"""
REQUIREMENT 2 - Discover Petri nets with the Alpha, Heuristic and Inductive miners (PM4PY)
and compare them, taking the timestamp errors of Requirement 1 into account.

Each miner is run on:
  - "original"   : the log as given (pm4py drops the 5 events with no timestamp, and
                   orders the rest by timestamp - so the wrong timestamps change the order)
  - "error_free" : only the cases with no timestamp error (from Requirement 1)
  - "repaired_*" : every log repaired by the LLMs in Requirement 5 (if they exist)
"""
import glob
import os

os.environ["PM4PY_SHOW_PROGRESS_BAR"] = "False"
import pandas as pd
import pm4py

from req1_timestamp_errors import error_cases, load_raw_log, parse_timestamps

MINERS = {
    "alpha": pm4py.discover_petri_net_alpha,
    "heuristic": pm4py.discover_petri_net_heuristics,
    "inductive": pm4py.discover_petri_net_inductive,
}


def to_pm4py(df):
    """Rename/sort the columns the way pm4py expects (rows without a timestamp are dropped)."""
    return pm4py.format_dataframe(df.dropna(subset=["Timestamp"]), case_id="CaseID",
                                  activity_key="Activity", timestamp_key="Timestamp")


def describe_model(log, net, im, fm):
    """Size of the net + the three usual quality measures."""
    fitness = pm4py.fitness_token_based_replay(log, net, im, fm)
    return {
        "places": len(net.places),
        "transitions": len(net.transitions),
        "silent_transitions": sum(t.label is None for t in net.transitions),  # invisible 'skip' steps
        "arcs": len(net.arcs),
        "sound": pm4py.check_soundness(net, im, fm)[0],          # can every case finish properly?
        "fitness": round(fitness["log_fitness"], 3),             # how well the model replays the log
        "fitting_traces_%": round(fitness["percentage_of_fitting_traces"], 1),
        "precision": round(pm4py.precision_token_based_replay(log, net, im, fm), 3),  # does it allow too much?
    }


if __name__ == "__main__":
    os.makedirs("output", exist_ok=True)
    raw = load_raw_log()
    log = parse_timestamps(raw)

    # ---- The event logs to mine ----
    logs = {
        "original": log,
        "error_free": log[~log["CaseID"].isin(error_cases(raw))],
    }
    for path in sorted(glob.glob("output/repaired_*.csv")):
        repaired = pd.read_csv(path)
        repaired["Timestamp"] = pd.to_datetime(repaired["Timestamp"], errors="coerce")
        logs[os.path.basename(path)[:-4]] = repaired

    # ---- Run the three miners on every log ----
    results = []
    for log_name, df in logs.items():
        pm_log = to_pm4py(df)
        print(f"\n=== Log: {log_name} ({pm_log['case:concept:name'].nunique()} cases) ===")
        for variant, count in sorted(pm4py.get_variants(pm_log).items(), key=lambda v: -v[1]):
            print(f"  {count:3d} x {' -> '.join(variant)}")

        for miner_name, discover in MINERS.items():
            net, im, fm = discover(pm_log)
            results.append({"log": log_name, "miner": miner_name, **describe_model(pm_log, net, im, fm)})
            try:
                pm4py.save_vis_petri_net(net, im, fm, f"output/req2_{log_name}_{miner_name}.png")
            except Exception as e:
                print(f"  (no PNG for {miner_name}: Graphviz not found - {type(e).__name__})")

    # ---- Comparison table ----
    table = pd.DataFrame(results)
    pd.set_option("display.width", 200)
    print("\n", table.to_string(index=False))
    table.to_csv("output/req2_model_comparison.csv", index=False)
    print("\nSaved: output/req2_model_comparison.csv and the Petri-net pictures in output/ (.png)")

# ---- Results ----
# error_free : all three miners find the same simple sequence R -> V -> E -> A -> C (fitness = precision = 1).
# original   : the errors create fake variants (Approve before Evaluate, cases with a missing step).
#   * Alpha     - has no noise handling: it sees E->A and A->E, decides they are parallel and builds an
#                 unsound net where no case can finish cleanly (0 % fitting traces).
#   * Heuristic - uses frequencies, so the rare wrong orders are treated as noise and mostly ignored;
#                 the model stays close to the real process (small loss of fitness).
#   * Inductive - guarantees a sound model that replays every case (fitness 1), but adds silent
#                 'skip' transitions / parallelism to fit the errors, so precision drops (allows too much).
