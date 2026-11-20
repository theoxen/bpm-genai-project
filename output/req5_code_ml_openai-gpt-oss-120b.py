import pandas as pd
import numpy as np
from datetime import timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

ACTIVITY_ORDER = [
    "Register Claim",
    "Verify Documents",
    "Evaluate Claim",
    "Approve Claim",
    "Close Claim",
]

def _parse_timestamp(ts):
    """
    Parse the raw timestamp string.
    Returns:
        - full datetime if complete (e.g. 2026-09-01T12:53:00)
        - datetime at hour‑start if only hour is present (e.g. 2026-09-13T16)
        - pd.NaT for missing/invalid values
    """
    if pd.isna(ts):
        return pd.NaT, "missing"
    ts = str(ts)
    # full ISO datetime
    try:
        dt = pd.to_datetime(ts, format="%Y-%m-%dT%H:%M:%S")
        return dt, "full"
    except (ValueError, TypeError):
        pass
    # hour‑only format
    try:
        dt = pd.to_datetime(ts, format="%Y-%m-%dT%H")
        return dt, "hour"
    except (ValueError, TypeError):
        return pd.NaT, "invalid"


def fix_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """
    Repair timestamp errors in a process‑mining event log.

    Parameters
    ----------
    df : pd.DataFrame
        Columns: CaseID, Activity, Timestamp (raw string or NaN), User, Cost

    Returns
    -------
    pd.DataFrame
        Same rows/columns, but Timestamp column converted to pandas datetime
        and corrected where necessary.
    """
    df = df.copy()

    # ------------------------------------------------------------------
    # 1. Parse timestamps and flag the kind of problem
    # ------------------------------------------------------------------
    parsed = df["Timestamp"].apply(_parse_timestamp)
    df["_ts_parsed"] = parsed.apply(lambda x: x[0])
    df["_ts_kind"]   = parsed.apply(lambda x: x[1])   # full / hour / missing / invalid

    # ------------------------------------------------------------------
    # 2. Sort rows by case and by the predefined activity order
    # ------------------------------------------------------------------
    df["activity_rank"] = df["Activity"].apply(lambda a: ACTIVITY_ORDER.index(a)
                                               if a in ACTIVITY_ORDER else -1)
    df.sort_values(["CaseID", "activity_rank"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # ------------------------------------------------------------------
    # 3. Build training data from rows that are *surely* correct
    # ------------------------------------------------------------------
    # A row is correct if:
    #   - timestamp is full datetime (not hour‑only, not missing)
    #   - its duration w.r.t. previous activity in the same case is non‑negative
    #   - the activity order respects the global order (we already sorted)
    train_rows = []
    train_target = []

    # iterate case by case
    for case_id, case_grp in df.groupby("CaseID"):
        prev_ts = None
        for _, row in case_grp.iterrows():
            ts = row["_ts_parsed"]
            kind = row["_ts_kind"]
            if kind == "full":
                if prev_ts is not None:
                    dur = (ts - prev_ts).total_seconds()
                    if dur >= 0:                     # non‑negative duration
                        train_rows.append(row)
                        train_target.append(dur)
                prev_ts = ts

    if not train_rows:
        raise ValueError("No clean rows found to train the model.")

    train_df = pd.DataFrame(train_rows)

    # ------------------------------------------------------------------
    # 4. Train a regression model to predict *duration since previous activity*
    # ------------------------------------------------------------------
    feature_cols = ["Activity", "User", "Cost"]
    X_train = train_df[feature_cols]
    y_train = np.array(train_target)

    categorical = ["Activity", "User"]
    numeric = ["Cost"]

    preproc = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
            ("num", "passthrough", numeric),
        ]
    )

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
        min_samples_leaf=2,
    )

    pipe = Pipeline(steps=[("preproc", preproc), ("model", model)])
    pipe.fit(X_train, y_train)

    # ------------------------------------------------------------------
    # 5. Helper to obtain a predicted duration for a given row
    # ------------------------------------------------------------------
    def predict_duration(row):
        return pipe.predict(row[feature_cols].to_frame().T)[0]

    # ------------------------------------------------------------------
    # 6. Estimate a fallback timestamp for the very first activity
    # ------------------------------------------------------------------
    # median of all existing Register Claim timestamps (full ones)
    register_full = df[
        (df["Activity"] == "Register Claim") & (df["_ts_kind"] == "full")
    ]["_ts_parsed"]
    if not register_full.empty:
        median_register_ts = register_full.median()
    else:
        # arbitrary fallback – start of 2026‑01‑01
        median_register_ts = pd.Timestamp("2026-01-01 08:00:00")

    # ------------------------------------------------------------------
    # 7. Walk through each case and repair timestamps
    # ------------------------------------------------------------------
    repaired_ts = []

    for case_id, case_grp in df.groupby("CaseID"):
        # keep rows in activity order
        case_grp = case_grp.sort_values("activity_rank")
        prev_ts = None

        for idx, row in case_grp.iterrows():
            kind = row["_ts_kind"]
            raw_ts = row["_ts_parsed"]

            # ---- first activity handling --------------------------------
            if row["Activity"] == "Register Claim":
                if kind == "full":
                    new_ts = raw_ts
                elif kind == "hour":
                    # hour‑only: keep hour start (no better guess)
                    new_ts = raw_ts
                else:   # missing / invalid
                    new_ts = median_register_ts
                repaired_ts.append(new_ts)
                prev_ts = new_ts
                continue

            # ---- subsequent activities -----------------------------------
            # Determine if timestamp needs fixing
            needs_fix = False

            if kind == "missing" or kind == "invalid":
                needs_fix = True
            elif kind == "hour":
                # hour‑only: we will replace it with a value inside the hour
                needs_fix = True
            elif raw_ts < prev_ts:
                # negative duration / impossible order
                needs_fix = True

            if needs_fix:
                # predict duration (seconds) from previous activity
                pred_sec = max(predict_duration(row), 0.0)

                if kind == "hour":
                    # keep within the same hour
                    max_offset = 3600 - 1  # stay inside the hour
                    offset = min(pred_sec, max_offset)
                    new_ts = raw_ts + timedelta(seconds=offset)
                else:
                    new_ts = prev_ts + timedelta(seconds=pred_sec)
            else:
                # timestamp is fine
                new_ts = raw_ts

            repaired_ts.append(new_ts)
            prev_ts = new_ts

    # ------------------------------------------------------------------
    # 8. Assign repaired timestamps back to the dataframe
    # ------------------------------------------------------------------
    df["Timestamp"] = repaired_ts
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])

    # clean auxiliary columns
    df.drop(columns=["_ts_parsed", "_ts_kind", "activity_rank"], inplace=True)

    # reorder columns to original order
    df = df[["CaseID", "Activity", "Timestamp", "User", "Cost"]]

    return df
