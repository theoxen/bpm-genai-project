import re
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def fix_timestamps(df):
    df_out = df.copy()

    # Helper function to parse individual raw timestamp strings
    def parse_ts(val):
        if pd.isna(val):
            return None, True, False
        s = str(val).strip()
        if not s or s.lower() == "nan":
            return None, True, False

        # Truncated check: e.g. "2026-09-13T16" (13 chars or missing colons)
        is_trunc = (len(s) == 13 and ("T" in s or " " in s)) or (
            ":" not in s and len(s) >= 10
        )

        try:
            dt = pd.to_datetime(s)
            return dt, is_trunc, is_trunc
        except Exception:
            return None, True, False

    # Extract clean pairs across all cases to train regression model
    train_features = []
    train_targets = []

    grouped = df_out.groupby("CaseID", sort=False)

    for _, case_df in grouped:
        case_rows = case_df.to_dict("records")
        parsed_info = [parse_ts(row["Timestamp"]) for row in case_rows]

        for i in range(1, len(case_rows)):
            prev_dt, prev_err, _ = parsed_info[i - 1]
            curr_dt, curr_err, _ = parsed_info[i]

            if not prev_err and not curr_err and curr_dt > prev_dt:
                dur = (curr_dt - prev_dt).total_seconds()
                train_features.append(
                    {
                        "Activity": case_rows[i]["Activity"],
                        "User": case_rows[i]["User"],
                        "Cost": float(case_rows[i]["Cost"]),
                    }
                )
                train_targets.append(dur)

    # Train RandomForestRegressor to predict duration from Activity, User, Cost
    if len(train_features) > 0:
        X_train = pd.DataFrame(train_features)
        y_train = np.array(train_targets)

        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "cat",
                    OneHotEncoder(handle_unknown="ignore"),
                    ["Activity", "User"],
                ),
                ("num", "passthrough", ["Cost"]),
            ]
        )

        model = Pipeline(
            [
                ("preprocessor", preprocessor),
                (
                    "regressor",
                    RandomForestRegressor(n_estimators=100, random_state=42),
                ),
            ]
        )
        model.fit(X_train, y_train)
    else:
        model = None

    repaired_timestamps = {}

    # Repair timestamps case by case
    for _, case_df in grouped:
        indices = case_df.index.tolist()
        case_rows = case_df.to_dict("records")
        n = len(case_rows)

        parsed_dts = []
        is_errors = []
        is_truncs = []

        for row in case_rows:
            dt, err, trunc = parse_ts(row["Timestamp"])
            parsed_dts.append(dt)
            is_errors.append(err)
            is_truncs.append(trunc)

        # Flag negative duration and impossible order errors
        last_valid = None
        for i in range(n):
            if not is_errors[i]:
                if last_valid is not None and parsed_dts[i] <= last_valid:
                    is_errors[i] = True
                else:
                    last_valid = parsed_dts[i]

        final_dts = [
            None if is_errors[i] else parsed_dts[i] for i in range(n)
        ]

        # Helper to predict duration
        def predict_dur(idx):
            if model is None:
                return 3600.0
            row_feat = pd.DataFrame(
                [
                    {
                        "Activity": case_rows[idx]["Activity"],
                        "User": case_rows[idx]["User"],
                        "Cost": float(case_rows[idx]["Cost"]),
                    }
                ]
            )
            return max(1.0, float(model.predict(row_feat)[0]))

        # Backward repair if initial events (e.g. Register Claim) are missing/erroneous
        first_valid_idx = next(
            (i for i in range(n) if not is_errors[i]), None
        )
        if first_valid_idx is not None and first_valid_idx > 0:
            for i in range(first_valid_idx - 1, -1, -1):
                dur = predict_dur(i + 1)
                est_dt = final_dts[i + 1] - pd.Timedelta(seconds=dur)
                if is_truncs[i] and parsed_dts[i] is not None:
                    h_start = parsed_dts[i]
                    h_end = h_start + pd.Timedelta(minutes=59, seconds=59)
                    est_dt = max(h_start, min(est_dt, h_end))
                final_dts[i] = est_dt
                is_errors[i] = False
        elif first_valid_idx is None:
            # Fallback if entire case has errors
            if is_truncs[0] and parsed_dts[0] is not None:
                final_dts[0] = parsed_dts[0]
            else:
                final_dts[0] = pd.Timestamp("2026-01-01 00:00:00")
            is_errors[0] = False

        # Forward repair for remaining errors
        for i in range(1, n):
            if is_errors[i]:
                dur = predict_dur(i)
                est_dt = final_dts[i - 1] + pd.Timedelta(seconds=dur)

                # Bound estimation if a subsequent valid timestamp exists
                next_valid_dt = next(
                    (
                        final_dts[k]
                        for k in range(i + 1, n)
                        if final_dts[k] is not None
                    ),
                    None,
                )
                if next_valid_dt is not None and est_dt >= next_valid_dt:
                    avail_sec = (
                        next_valid_dt - final_dts[i - 1]
                    ).total_seconds()
                    if avail_sec > 1:
                        est_dt = final_dts[i - 1] + pd.Timedelta(
                            seconds=min(dur, avail_sec * 0.5)
                        )
                    else:
                        est_dt = final_dts[i - 1] + pd.Timedelta(
                            milliseconds=500
                        )

                # Restrict to hour range if timestamp was truncated
                if is_truncs[i] and parsed_dts[i] is not None:
                    h_start = parsed_dts[i]
                    h_end = h_start + pd.Timedelta(minutes=59, seconds=59)
                    est_dt = max(h_start, min(est_dt, h_end))

                final_dts[i] = est_dt
                is_errors[i] = False

        for idx, dt in zip(indices, final_dts):
            repaired_timestamps[idx] = dt

    df_out["Timestamp"] = pd.to_datetime(
        df_out.index.map(repaired_timestamps)
    )
    return df_out
