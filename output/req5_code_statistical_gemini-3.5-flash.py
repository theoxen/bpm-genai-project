import numpy as np
import pandas as pd


def fix_timestamps(df):
    result_df = df.copy()

    # Identify invalid/missing/truncated timestamps
    def is_invalid_val(val):
        if pd.isna(val):
            return True
        s = str(val).strip()
        # Truncated timestamps like "2026-09-13T16" lack minute/second colons
        if ":" not in s or len(s) < 19:
            return True
        return False

    timestamps = []
    is_valid_initial = []

    for val in result_df["Timestamp"]:
        if is_invalid_val(val):
            timestamps.append(pd.NaT)
            is_valid_initial.append(False)
        else:
            parsed = pd.to_datetime(val, errors="coerce")
            if pd.isna(parsed):
                timestamps.append(pd.NaT)
                is_valid_initial.append(False)
            else:
                timestamps.append(parsed)
                is_valid_initial.append(True)

    result_df["_parsed_ts"] = timestamps
    result_df["_is_valid"] = is_valid_initial

    # Compute median duration for each consecutive activity transition from valid pairs
    transition_durations = {1: [], 2: [], 3: [], 4: []}

    for _, group in result_df.groupby("CaseID", sort=False):
        ts_list = group["_parsed_ts"].tolist()
        val_list = group["_is_valid"].tolist()

        for k in range(1, 5):
            if val_list[k - 1] and val_list[k]:
                if ts_list[k] >= ts_list[k - 1]:
                    transition_durations[k].append(ts_list[k] - ts_list[k - 1])

    median_durations = {}
    default_median = pd.Timedelta(hours=1)
    for k in range(1, 5):
        if len(transition_durations[k]) > 0:
            median_durations[k] = pd.Series(transition_durations[k]).median()
        else:
            median_durations[k] = default_median

    # Repair timestamps per case
    repaired_timestamps = result_df["_parsed_ts"].copy()

    for _, group in result_df.groupby("CaseID", sort=False):
        indices = group.index.tolist()
        ts_arr = [result_df.loc[idx, "_parsed_ts"] for idx in indices]
        valid_arr = [result_df.loc[idx, "_is_valid"] for idx in indices]

        # Invalidate timestamps that break chronological order
        last_valid_ts = None
        for k in range(5):
            if valid_arr[k]:
                if last_valid_ts is not None and ts_arr[k] < last_valid_ts:
                    valid_arr[k] = False
                else:
                    last_valid_ts = ts_arr[k]

        # Find first valid anchor index in the case
        first_valid_idx = -1
        for k in range(5):
            if valid_arr[k]:
                first_valid_idx = k
                break

        if first_valid_idx != -1:
            # Backward repair before first valid index
            for k in range(first_valid_idx - 1, -1, -1):
                ts_arr[k] = ts_arr[k + 1] - median_durations[k + 1]
                valid_arr[k] = True

            # Forward repair for missing, truncated or out-of-order timestamps
            for k in range(1, 5):
                if not valid_arr[k] or ts_arr[k] < ts_arr[k - 1]:
                    ts_arr[k] = ts_arr[k - 1] + median_durations[k]
                    valid_arr[k] = True

        for k in range(5):
            repaired_timestamps.loc[indices[k]] = ts_arr[k]

    result_df["Timestamp"] = repaired_timestamps
    result_df.drop(columns=["_parsed_ts", "_is_valid"], inplace=True)

    return result_df
