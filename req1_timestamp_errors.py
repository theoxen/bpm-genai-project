"""
REQUIREMENT 1 - Find the timestamp errors in the event log (with code, no LLM).

The functions below are also imported by the GUI (app.py). The errors found here
are the "ground truth" used to score the LLM repairs in Requirement 6.
"""
import re

import pandas as pd

CSV_PATH = "data/insurance_claims_log.csv"
XES_PATH = "data/insurance_claims_log.xes"

# The correct order of the insurance process (every case has these 5 activities)
EXPECTED_ORDER = ["Register Claim", "Verify Documents", "Evaluate Claim", "Approve Claim", "Close Claim"]
FULL_FORMAT = r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}"  # e.g. 2026-09-01T12:53:00


def load_raw_log(path=CSV_PATH):
    """Read the CSV exactly as it is - timestamps stay as text so we can inspect their format."""
    return pd.read_csv(path, dtype={"Timestamp": str})


def parse_timestamps(df):
    """Convert the timestamp text to real datetimes. Empty -> NaT (missing), '2026-09-13T16' -> 16:00."""
    df = df.copy()
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], format="ISO8601", errors="coerce")
    return df


def find_timestamp_errors(raw):
    """Run 4 checks on every case of the raw log and return one row per erroneous event."""
    log = parse_timestamps(raw)
    errors = []
    for case_id, case in log.groupby("CaseID", sort=False):
        text = raw.loc[case.index, "Timestamp"]
        rows = case.to_dict("records")  # events in the order they were recorded in the file

        # Check 1 - missing timestamp
        for e in rows:
            if pd.isna(e["Timestamp"]):
                errors.append([case_id, e["Activity"], None, "Missing timestamp",
                               "The timestamp is empty, so we do not know when the activity happened."])

        # Check 2 - truncated timestamp: only the hour was stored, the minutes are lost
        for e, t in zip(rows, text):
            if pd.notna(t) and not re.fullmatch(FULL_FORMAT, t):
                errors.append([case_id, e["Activity"], t, "Truncated timestamp",
                               f"'{t}' has no minutes/seconds - we only know the hour, not the exact time."])

        # Check 3 - negative duration: an event is timestamped EARLIER than the event recorded before it
        for prev, curr in zip(rows, rows[1:]):
            if pd.notna(prev["Timestamp"]) and pd.notna(curr["Timestamp"]) and curr["Timestamp"] < prev["Timestamp"]:
                minutes = (prev["Timestamp"] - curr["Timestamp"]).total_seconds() / 60
                errors.append([case_id, curr["Activity"], str(curr["Timestamp"]), "Negative duration",
                               f"Recorded after '{prev['Activity']}' but timestamped {minutes:.0f} min "
                               f"before it (clock / data-entry error)."])

        # Check 4 - impossible order: the timestamps say 'Approve' happened before 'Evaluate'
        # (a claim cannot be approved before it is evaluated -> the two timestamps were most likely swapped)
        recorded_order = [e["Activity"] for e in rows]
        if recorded_order != EXPECTED_ORDER:
            for expected, actual in zip(EXPECTED_ORDER, rows):
                if expected != actual["Activity"]:
                    errors.append([case_id, actual["Activity"], str(actual["Timestamp"]), "Impossible order",
                                   f"'{actual['Activity']}' is timestamped before '{expected}' - "
                                   f"the claim is approved before it is evaluated."])
                    break

    return pd.DataFrame(errors, columns=["CaseID", "Activity", "Timestamp", "ErrorType", "Explanation"])


def error_cases(raw):
    """The set of case IDs that contain at least one timestamp error."""
    return set(find_timestamp_errors(raw)["CaseID"])


if __name__ == "__main__":
    import os
    os.environ["PM4PY_SHOW_PROGRESS_BAR"] = "False"
    import pm4py

    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 110)
    raw = load_raw_log()
    log = parse_timestamps(raw)

    # ---- Step 1: basic overview of the log ----
    print(f"Events: {len(raw)}   Cases: {raw['CaseID'].nunique()}")
    print("Timestamp text lengths (19 = full 'YYYY-MM-DDTHH:MM:SS'):",
          raw["Timestamp"].fillna("").str.len().value_counts().to_dict(), "\n")

    # ---- Step 2: process-mining view with PM4PY (variants = distinct activity sequences) ----
    # pm4py orders the events of each case by timestamp and drops events without one,
    # so the timestamp errors appear as unusual variants.
    pm_log = pm4py.format_dataframe(log.dropna(subset=["Timestamp"]), case_id="CaseID",
                                    activity_key="Activity", timestamp_key="Timestamp")
    print("Variants found by PM4PY:")
    for variant, count in sorted(pm4py.get_variants(pm_log).items(), key=lambda v: -v[1]):
        print(f"  {count:3d} cases: {' -> '.join(variant)}")

    # ---- Step 3: the four checks ----
    errors = find_timestamp_errors(raw)
    print(f"\nTimestamp errors found: {len(errors)} events in {errors['CaseID'].nunique()} cases\n")
    print(errors.groupby("ErrorType").size().to_string(), "\n")
    print(errors.to_string(index=False))

    # ---- Step 4: confirm that the XES file holds the same timestamps as the CSV ----
    xes = pm4py.read_xes(XES_PATH)
    same = xes["time:timestamp"].dt.tz_localize(None).reset_index(drop=True).equals(log["Timestamp"])
    print(f"\nXES file has the same timestamps (and therefore the same errors) as the CSV: {same}")

    os.makedirs("output", exist_ok=True)
    errors.to_csv("output/req1_timestamp_errors.csv", index=False)
    print("Saved: output/req1_timestamp_errors.csv")
