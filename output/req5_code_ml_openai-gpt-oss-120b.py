import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

def fix_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """
    Repair timestamps in an insurance‑claims event log.
    
    Parameters
    ----------
    df : pd.DataFrame
        Columns: CaseID, Activity, Timestamp, User, Cost.
        Timestamp is a string or NaN (float).

    Returns
    -------
    pd.DataFrame
        Same rows/columns, Timestamp converted to pandas datetime and corrected.
    """
    # ------------------------------------------------------------------
    # 1. Prepare a working copy and auxiliary structures
    # ------------------------------------------------------------------
    df_fixed = df.copy()
    act_order = ['Register Claim', 'Verify Documents',
                 'Evaluate Claim', 'Approve Claim', 'Close Claim']
    act_to_idx = {a: i for i, a in enumerate(act_order)}
    df_fixed['ActIdx'] = df_fixed['Activity'].map(act_to_idx)

    # ------------------------------------------------------------------
    # 2. Detect truncated timestamps (hour‑only strings)
    # ------------------------------------------------------------------
    def is_truncated(ts):
        if pd.isna(ts):
            return False
        # full iso‑datetime has at least 19 characters (e.g. 2026-09-01T12:53:00)
        return isinstance(ts, str) and len(ts.strip()) <= 13  # e.g. "2026-09-13T16"

    trunc_mask = df_fixed['Timestamp'].apply(is_truncated)

    # ------------------------------------------------------------------
    # 3. Parse *full* timestamps (ignore truncated & NaN for now)
    # ------------------------------------------------------------------
    full_mask = ~df_fixed['Timestamp'].isna() & ~trunc_mask
    parsed_full = pd.to_datetime(df_fixed.loc[full_mask, 'Timestamp'],
                                 errors='coerce')
    df_fixed.loc[full_mask, 'ParsedTS'] = parsed_full

    # ------------------------------------------------------------------
    # 4. Build training set from *correct* transitions
    #    – both previous and current timestamps are parsed
    #    – duration > 0
    # ------------------------------------------------------------------
    train_rows = []
    train_targets = []

    # iterate case by case (preserves order)
    for case_id, grp in df_fixed.groupby('CaseID'):
        grp = grp.sort_values('ActIdx')          # enforce expected order
        prev_ts = None
        prev_idx = None
        for _, row in grp.iterrows():
            cur_ts = row['ParsedTS']
            if pd.isna(cur_ts):
                prev_ts = None
                prev_idx = None
                continue
            if prev_ts is not None:
                dur = (cur_ts - prev_ts).total_seconds()
                if dur > 0:                       # only keep sensible durations
                    # features: current Activity, User, Cost
                    train_rows.append({
                        'Activity': row['Activity'],
                        'User': row['User'],
                        'Cost': row['Cost']
                    })
                    train_targets.append(dur)
            prev_ts = cur_ts
            prev_idx = row['ActIdx']

    if not train_rows:
        raise ValueError("No correct transitions found – cannot train model.")

    train_X = pd.DataFrame(train_rows)
    y = np.array(train_targets)

    # ------------------------------------------------------------------
    # 5. Encode categorical columns and train regression model
    # ------------------------------------------------------------------
    categorical_features = ['Activity', 'User']
    numeric_features = ['Cost']

    preproc = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
            ('num', 'passthrough', numeric_features)
        ]
    )
    X_enc = preproc.fit_transform(train_X)

    rf = RandomForestRegressor(
        n_estimators=300,
        random_state=0,
        n_jobs=-1,
        max_depth=None,
        min_samples_leaf=1
    )
    rf.fit(X_enc, y)

    # ------------------------------------------------------------------
    # 6. Helper to predict a duration (in seconds) for a given row
    # ------------------------------------------------------------------
    def predict_duration(activity, user, cost):
        Xrow = pd.DataFrame([{'Activity': activity,
                              'User': user,
                              'Cost': cost}])
        Xrow_enc = preproc.transform(Xrow)
        return rf.predict(Xrow_enc)[0]

    # ------------------------------------------------------------------
    # 7. Determine a fallback start‑time for cases whose first event is broken
    # ------------------------------------------------------------------
    start_times = df_fixed.loc[
        (df_fixed['Activity'] == 'Register Claim') &
        df_fixed['ParsedTS'].notna(),
        'ParsedTS'
    ]
    fallback_start = start_times.median() if not start_times.empty else pd.Timestamp.now()

    # ------------------------------------------------------------------
    # 8. Re‑construct timestamps case by case
    # ------------------------------------------------------------------
    corrected_ts = {}

    for case_id, grp in df_fixed.groupby('CaseID'):
        # sort by expected activity order
        grp = grp.sort_values('ActIdx')
        last_ts = None

        for idx, row in grp.iterrows():
            cur_act = row['Activity']
            cur_user = row['User']
            cur_cost = row['Cost']
            cur_parsed = row['ParsedTS']

            # Detect if the current timestamp is *reliable*:
            #   – not NaN, not truncated,
            #   – and (if there is a previous timestamp) duration > 0.
            reliable = False
            if pd.notna(cur_parsed):
                if last_ts is None:
                    reliable = True
                else:
                    dur = (cur_parsed - last_ts).total_seconds()
                    if dur > 0:
                        reliable = True

            if reliable:
                # keep the original good timestamp
                corrected_ts[idx] = cur_parsed
                last_ts = cur_parsed
                continue

            # ---- otherwise we need to *predict* the timestamp ----
            # If this is the first activity, start from fallback
            if last_ts is None:
                # use median start time as base
                pred_ts = fallback_start
                # optionally adjust by predicted offset from start (0 seconds)
                corrected_ts[idx] = pred_ts
                last_ts = pred_ts
                continue

            # Predict duration from previous activity to current
            pred_dur_sec = predict_duration(cur_act, cur_user, cur_cost)
            # Ensure a minimum positive offset (e.g., 1 second) to avoid zero/negative loops
            pred_dur_sec = max(pred_dur_sec, 1.0)

            pred_ts = last_ts + pd.Timedelta(seconds=pred_dur_sec)
            corrected_ts[idx] = pred_ts
            last_ts = pred_ts

    # ------------------------------------------------------------------
    # 9. Write corrected timestamps back, preserving original index/order
    # ------------------------------------------------------------------
    for idx, ts in corrected_ts.items():
        df_fixed.at[idx, 'Timestamp'] = ts

    # Convert column to datetime (already datetime objects)
    df_fixed['Timestamp'] = pd.to_datetime(df_fixed['Timestamp'], errors='coerce')

    # Drop helper columns before returning
    df_fixed = df_fixed.drop(columns=['ActIdx', 'ParsedTS'])

    return df_fixed
