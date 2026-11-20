import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def fix_timestamps(df):
    res_df = df.copy()
    n = len(df)

    is_missing = [False] * n
    is_truncated = [False] * n
    truncated_hour = [None] * n
    parsed_dt = [None] * n

    # Step 1: Parse and identify missing / truncated timestamps
    for pos, (idx, row) in enumerate(df.iterrows()):
        val = row["Timestamp"]
        if pd.isna(val):
            is_missing[pos] = True
        else:
            val_str = str(val).strip()
            # Truncated timestamps have format YYYY-MM-DDTHH (missing minutes/seconds)
            if len(val_str) < 19 or val_str.count(":") < 2:
                is_truncated[pos] = True
                dt = pd.to_datetime(val_str, errors="coerce")
                truncated_hour[pos] = dt
            else:
                dt = pd.to_datetime(val_str, errors="coerce")
                if pd.isna(dt):
                    is_missing[pos] = True
                else:
                    parsed_dt[pos] = dt

    # Step 2: Group rows by CaseID while preserving original row sequence
    case_indices = {}
    for pos, case_id in enumerate(df["CaseID"]):
        if case_id not in case_indices:
            case_indices[case_id] = []
        case_indices[case_id].append(pos)

    # Step 3: Flag out-of-order timestamps within each case (negative duration / impossible order)
    valid_mask = [False] * n
    for case_id, indices in case_indices.items():
        last_t = None
        for pos in indices:
            if not is_missing[pos] and not is_truncated[pos]:
                dt = parsed_dt[pos]
                if last_t is None or dt >= last_t:
                    valid_mask[pos] = True
                    last_t = dt
                else:
                    valid_mask[pos] = False

    # Step 4: Build training dataset from valid consecutive activity pairs
    X_train_rows = []
    y_train = []
    for case_id, indices in case_indices.items():
        for k in range(1, len(indices)):
            prev_pos = indices[k - 1]
            curr_pos = indices[k]
            if valid_mask[prev_pos] and valid_mask[curr_pos]:
                dur = (parsed_dt[curr_pos] - parsed_dt[prev_pos]).total_seconds()
                if dur >= 0:
                    X_train_rows.append(df.iloc[curr_pos])
                    y_train.append(dur)

    # Step 5: Train ML regression model to predict duration from features
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(sparse_output=False, handle_unknown="ignore"),
                ["Activity", "User"],
            ),
            ("num", "passthrough", ["Cost"]),
        ]
    )

    model = Pipeline(
        [
            ("prep", preprocessor),
            ("reg", RandomForestRegressor(n_estimators=100, random_state=42)),
        ]
    )

    has_model = len(X_train_rows) > 0
    if has_model:
        X_train_df = pd.DataFrame(X_train_rows)[["Activity", "User", "Cost"]]
        model.fit(X_train_df, y_train)

    # Step 6: Impute invalid, missing, or truncated timestamps
    final_dt = [None] * n
    for pos in range(n):
        if valid_mask[pos] and not is_truncated[pos]:
            final_dt[pos] = parsed_dt[pos]

    for case_id, indices in case_indices.items():
        m = len(indices)

        # Predict durations for steps in case
        pred_durations = [pd.Timedelta(seconds=3600)] * m
        for k in range(1, m):
            pos = indices[k]
            if has_model:
                X_row = df.iloc[[pos]][["Activity", "User", "Cost"]]
                pred_sec = max(0.0, float(model.predict(X_row)[0]))
                pred_durations[k] = pd.Timedelta(seconds=pred_sec)

        # Forward propagation
        for k in range(1, m):
            pos = indices[k]
            if final_dt[pos] is None:
                prev_pos = indices[k - 1]
                if final_dt[prev_pos] is not None:
                    cand = final_dt[prev_pos] + pred_durations[k]
                    if is_truncated[pos] and truncated_hour[pos] is not None:
                        h_start = truncated_hour[pos]
                        h_end = h_start + pd.Timedelta(minutes=59, seconds=59)
                        cand = max(h_start, min(h_end, cand))
                    final_dt[pos] = cand

        # Backward propagation
        for k in range(m - 2, -1, -1):
            pos = indices[k]
            if final_dt[pos] is None:
                next_pos = indices[k + 1]
                if final_dt[next_pos] is not None:
                    cand = final_dt[next_pos] - pred_durations[k + 1]
                    if is_truncated[pos] and truncated_hour[pos] is not None:
                        h_start = truncated_hour[pos]
                        h_end = h_start + pd.Timedelta(minutes=59, seconds=59)
                        cand = max(h_start, min(h_end, cand))
                    final_dt[pos] = cand

        # Second forward propagation pass
        for k in range(1, m):
            pos = indices[k]
            prev_pos = indices[k - 1]
            if final_dt[prev_pos] is not None:
                if final_dt[pos] is None:
                    cand = final_dt[prev_pos] + pred_durations[k]
                    if is_truncated[pos] and truncated_hour[pos] is not None:
                        h_start = truncated_hour[pos]
                        h_end = h_start + pd.Timedelta(minutes=59, seconds=59)
                        cand = max(h_start, min(h_end, cand))
                    final_dt[pos] = cand
                elif not (valid_mask[pos] and not is_truncated[pos]):
                    if final_dt[pos] < final_dt[prev_pos]:
                        final_dt[pos] = final_dt[prev_pos] + pred_durations[k]
                        if is_truncated[pos] and truncated_hour[pos] is not None:
                            h_start = truncated_hour[pos]
                            h_end = h_start + pd.Timedelta(minutes=59, seconds=59)
                            final_dt[pos] = max(h_start, min(h_end, final_dt[pos]))

    res_df["Timestamp"] = pd.to_datetime(final_dt)
    return res_df
