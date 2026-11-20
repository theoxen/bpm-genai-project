import pandas as pd
import numpy as np
import re
from typing import Dict, Tuple, List

def fix_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """
    Repair timestamp errors in an event‑log DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Columns: CaseID, Activity, Timestamp (string or NaN), User, Cost.

    Returns
    -------
    pd.DataFrame
        Same rows/columns, but the *Timestamp* column is a ``datetime64[ns]``
        column where the erroneous timestamps have been replaced using a
        statistical (median‑duration) approach.
    """
    # ------------------------------------------------------------------ #
    # 0. Preparations
    # ------------------------------------------------------------------ #
    df = df.copy()                                    # avoid side‑effects
    activity_order = [
        "Register Claim",
        "Verify Documents",
        "Evaluate Claim",
        "Approve Claim",
        "Close Claim",
    ]
    act_to_pos = {a: i for i, a in enumerate(activity_order)}

    # ------------------------------------------------------------------ #
    # 1. Helper to turn the raw string into a proper datetime (or NaT)
    # ------------------------------------------------------------------ #
    hour_only_pat = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}$")   # e.g. 2026-09-13T16

    def parse_ts(val):
        """Return a Timestamp or pd.NaT."""
        if pd.isna(val):
            return pd.NaT
        if isinstance(val, str) and hour_only_pat.fullmatch(val):
            # Truncated – treat as missing (will be repaired later)
            return pd.NaT
        try:
            return pd.to_datetime(val)
        except Exception:
            return pd.NaT

    df["ParsedTimestamp"] = df["Timestamp"].apply(parse_ts)

    # ------------------------------------------------------------------ #
    # 2. Identify *good* cases – complete, ordered and monotonic timestamps
    # ------------------------------------------------------------------ #
    good_cases = []

    for case_id, grp in df.groupby("CaseID"):
        # Sort rows according to the defined activity order
        grp = grp.set_index("Activity").loc[activity_order].reset_index()
        ts_series = grp["ParsedTimestamp"]

        # All timestamps must be present and strictly increasing
        if ts_series.isna().any():
            continue
        if not (ts_series.diff().dt.total_seconds().iloc[1:] > 0).all():
            continue
        good_cases.append(ts_series.values)

    # If we have no good cases we cannot compute medians – fall back to 1 h
    if not good_cases:
        median_durations = {
            (activity_order[i], activity_order[i + 1]): pd.Timedelta(hours=1)
            for i in range(len(activity_order) - 1)
        }
    else:
        # ------------------------------------------------------------------
        # 3. Compute median durations (in seconds) for each consecutive pair
        # ------------------------------------------------------------------
        durations: Dict[Tuple[str, str], List[float]] = {
            (activity_order[i], activity_order[i + 1]): [] for i in range(len(activity_order) - 1)
        }

        for case_ts in good_cases:
            for i in range(len(activity_order) - 1):
                diff_sec = (case_ts[i + 1] - case_ts[i]).total_seconds()
                durations[(activity_order[i], activity_order[i + 1])].append(diff_sec)

        median_durations = {}
        for key, secs in durations.items():
            if secs:  # at least one observation
                median_secs = np.median(secs)
                median_durations[key] = pd.Timedelta(seconds=median_secs)
            else:   # no observation – use a safe default of 1 hour
                median_durations[key] = pd.Timedelta(hours=1)

    # ------------------------------------------------------------------ #
    # 4. Repair each case
    # ------------------------------------------------------------------ #
    # We will work on a copy of the parsed timestamps and later write back
    repaired_ts = df["ParsedTimestamp"].copy()

    # Helper to fetch the median duration for a transition
    def get_median(prev_act: str, cur_act: str) -> pd.Timedelta:
        return median_durations.get((prev_act, cur_act), pd.Timedelta(hours=1))

    for case_id, grp in df.groupby("CaseID"):
        # Preserve original order of rows (they already appear in the file order)
        idxs = grp.index.tolist()
        activities = grp["Activity"].tolist()
        timestamps = grp["ParsedTimestamp"].tolist()

        # ------------------------------------------------------------------
        # 4a. Forward pass – fix missing / truncated / negative timestamps
        # ------------------------------------------------------------------
        for i, (idx, act, ts) in enumerate(zip(idxs, activities, timestamps)):
            if pd.isna(ts):
                # No timestamp – estimate from the previous known one
                # Look backwards for the most recent valid timestamp
                j = i - 1
                while j >= 0 and pd.isna(timestamps[j]):
                    j -= 1
                if j >= 0:
                    prev_act = activities[j]
                    prev_ts = timestamps[j]
                    estimate = prev_ts + get_median(prev_act, act)
                else:
                    # No previous known timestamp – look forward
                    k = i + 1
                    while k < len(timestamps) and pd.isna(timestamps[k]):
                        k += 1
                    if k < len(timestamps):
                        next_act = activities[k]
                        next_ts = timestamps[k]
                        # Walk backwards from the next known timestamp
                        estimate = next_ts
                        # subtract median durations backward from k down to i+1
                        for back in range(k, i, -1):
                            cur = activities[back]
                            prev = activities[back - 1]
                            estimate -= get_median(prev, cur)
                    else:
                        # Completely isolated case – fallback to current date
                        estimate = pd.Timestamp.now()
                repaired_ts.at[idx] = estimate
                timestamps[i] = estimate  # update local copy for subsequent steps
                continue

            # If timestamp exists but creates a negative duration w.r.t. previous,
            # replace it with previous + median
            if i > 0 and not pd.isna(timestamps[i - 1]):
                prev_ts = timestamps[i - 1]
                if ts <= prev_ts:
                    prev_act = activities[i - 1]
                    estimate = prev_ts + get_median(prev_act, act)
                    repaired_ts.at[idx] = estimate
                    timestamps[i] = estimate

    # ------------------------------------------------------------------ #
    # 5. Write back the repaired timestamps (as datetime64[ns])
    # ------------------------------------------------------------------ #
    df["Timestamp"] = repaired_ts
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])   # ensure dtype

    # Drop the helper column
    df = df.drop(columns=["ParsedTimestamp"])

    # Preserve original column order
    df = df[["CaseID", "Activity", "Timestamp", "User", "Cost"]]

    return df
