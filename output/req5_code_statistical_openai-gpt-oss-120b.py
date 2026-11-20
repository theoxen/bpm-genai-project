import pandas as pd
import numpy as np

def fix_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """
    Repair timestamp errors in an insurance‑claims event log.

    Parameters
    ----------
    df : pd.DataFrame
        Columns: CaseID, Activity, Timestamp, User, Cost.
        Timestamp is a string or NaN exactly as read from the source file.

    Returns
    -------
    pd.DataFrame
        Same rows/columns, but the Timestamp column is a pandas datetime
        with the detected errors corrected.
    """
    # ------------------------------------------------------------------ #
    # 1.  Preparations --------------------------------------------------- #
    # ------------------------------------------------------------------ #
    df = df.copy()                         # do not modify the original object
    order_list = ['Register Claim', 'Verify Documents',
                  'Evaluate Claim', 'Approve Claim', 'Close Claim']
    order_map = {act: i for i, act in enumerate(order_list)}
    df['_order'] = df['Activity'].map(order_map)

    # masks for the different error types
    is_missing = pd.isna(df['Timestamp'])
    # truncated = exactly "YYYY‑MM‑DDTHH"
    truncated_pat = r'^\d{4}-\d{2}-\d{2}T\d{2}$'
    is_truncated = df['Timestamp'].astype('object').astype(str).str.fullmatch(truncated_pat).fillna(False)

    # ------------------------------------------------------------------ #
    # 2.  Parse all timestamps (coerce errors → NaT) -------------------- #
    # ------------------------------------------------------------------ #
    df['_parsed'] = pd.to_datetime(df['Timestamp'], errors='coerce')

    # ------------------------------------------------------------------ #
    # 3.  Compute median durations between consecutive activities
    #     using only “clean” transitions (no missing/truncated & monotone) #
    # ------------------------------------------------------------------ #
    # Sort by case and activity order – keep original index for later write‑back
    df_sorted = df.sort_values(['CaseID', '_order']).reset_index()

    # helper columns for previous row in the same case
    df_sorted['_prev_case'] = df_sorted['CaseID'].shift()
    df_sorted['_prev_order'] = df_sorted['_order'].shift()
    df_sorted['_prev_parsed'] = df_sorted['_parsed'].shift()
    df_sorted['_prev_missing'] = is_missing.iloc[df_sorted['index']].shift().fillna(True)
    df_sorted['_prev_trunc'] = is_truncated.iloc[df_sorted['index']].shift().fillna(True)
    df_sorted['_prev_activity'] = df_sorted['Activity'].shift()

    # a transition is “clean” when:
    #   – both current and previous rows belong to the same case
    #   – they are consecutive in the defined order
    #   – neither timestamp is missing nor truncated
    clean_transition = (
        (df_sorted['CaseID'] == df_sorted['_prev_case']) &
        (df_sorted['_order'] == df_sorted['_prev_order'] + 1) &
        (~is_missing.iloc[df_sorted['index']]) &
        (~is_truncated.iloc[df_sorted['index']]) &
        (~df_sorted['_prev_missing']) &
        (~df_sorted['_prev_trunc'])
    )

    # durations for those clean transitions
    durations = df_sorted.loc[clean_transition, '_parsed'] - df_sorted.loc[clean_transition, '_prev_parsed']

    # median duration per *previous* activity (e.g. Register → Verify)
    median_durations = (
        durations.groupby(df_sorted.loc[clean_transition, '_prev_activity'])
                 .median()
                 .rename('median')
                 .to_dict()
    )
    # fallback – if a transition never appears, use 1 hour
    default_median = pd.Timedelta(hours=1)

    # ------------------------------------------------------------------ #
    # 4.  Repair timestamps case by case -------------------------------- #
    # ------------------------------------------------------------------ #
    repaired_ts = df['_parsed'].copy()   # will hold the final timestamps

    for case_id, grp in df.groupby('CaseID'):
        # ensure the rows are processed in activity order
        grp = grp.sort_values('_order')
        last_ts = None
        last_idx = None

        for idx, row in grp.iterrows():
            cur_missing = pd.isna(row['Timestamp'])
            cur_trunc   = is_truncated.loc[idx]

            cur_ts = repaired_ts.loc[idx]   # may be NaT at this point

            # ---------------------------------------------------------- #
            # a) missing timestamp
            # ---------------------------------------------------------- #
            if cur_missing:
                if last_ts is not None:
                    # use median duration from the *previous* activity
                    prev_act = df.loc[last_idx, 'Activity']
                    delta = median_durations.get(prev_act, default_median)
                    repaired_ts.loc[idx] = last_ts + delta
                # else: first activity missing – nothing we can infer, keep NaT
                continue

            # ---------------------------------------------------------- #
            # b) truncated timestamp – keep hour, but treat as “unknown”
            # ---------------------------------------------------------- #
            if cur_trunc:
                # treat as the hour start (already parsed that way)
                # If it violates monotonicity, push it forward by the median delta
                if last_ts is not None and cur_ts < last_ts:
                    prev_act = df.loc[last_idx, 'Activity']
                    delta = median_durations.get(prev_act, default_median)
                    repaired_ts.loc[idx] = last_ts + delta
                else:
                    repaired_ts.loc[idx] = cur_ts  # hour‑only is acceptable for now
                # continue to next row
                last_ts = repaired_ts.loc[idx]
                last_idx = idx
                continue

            # ---------------------------------------------------------- #
            # c) timestamp present – check for negative duration / order #
            # ---------------------------------------------------------- #
            if last_ts is not None and cur_ts < last_ts:
                # either negative duration or impossible order
                prev_act = df.loc[last_idx, 'Activity']
                delta = median_durations.get(prev_act, default_median)
                repaired_ts.loc[idx] = last_ts + delta
                # after correction, treat this as the new last_ts
                cur_ts = repaired_ts.loc[idx]

            # ---------------------------------------------------------- #
            # d) everything OK – keep original timestamp
            # ---------------------------------------------------------- #
            repaired_ts.loc[idx] = cur_ts
            last_ts = cur_ts
            last_idx = idx

    # ------------------------------------------------------------------ #
    # 5.  Write back the repaired timestamps ---------------------------- #
    # ------------------------------------------------------------------ #
    df['Timestamp'] = repaired_ts
    # ensure dtype is datetime64[ns] (NaT stays for unrecoverable cases)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')

    # drop the temporary helper column
    df.drop(columns=['_order'], inplace=True)

    return df
