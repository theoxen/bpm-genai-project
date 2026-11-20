"""
GUI (Streamlit) - the digital assistant for Requirements 3, 4, 5, 6 and 7.

Every requirement has its own section with a page_reqN() function = the content of its tab.
The tabs are put together at the very end of the file.
"""
import os
import time

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from llm import MODELS, ask_llm, extract_code
from req1_timestamp_errors import EXPECTED_ORDER, find_timestamp_errors, load_raw_log, parse_timestamps

RAW = load_raw_log()                       # the log exactly as in the file (timestamps as text)
LOG = parse_timestamps(RAW)                # the same log with real datetimes
ERRORS = find_timestamp_errors(RAW)        # Requirement 1 result = ground truth for Requirement 6
ERROR_CASES = set(ERRORS["CaseID"])
RESULTS_FILE = "output/req5_results.csv"   # every repair run is saved here (used by Req 6 and 7)
os.makedirs("output", exist_ok=True)


def add_waiting_times(log):
    """WaitingMinutes = minutes since the previous event of the same case (events ordered by time)."""
    log = log.sort_values("Timestamp")
    log["WaitingMinutes"] = log.groupby("CaseID")["Timestamp"].diff().dt.total_seconds() / 60
    return log.sort_index()


def run_code(code, variables):
    """Execute code written by the LLM and return all the variables it created."""
    exec(code, variables)
    return variables


def show_usage(reply):
    st.caption(f"{reply['seconds']} s - {reply['input_tokens']} input / {reply['output_tokens']} output tokens")


LOG_WITH_WAITS = add_waiting_times(LOG)


# ============================================================================
# ---- REQUIREMENT 3: digital assistant that answers with tables and charts ----
# The LLM never sees the whole log. It gets a description of the columns, writes pandas
# code for the question, and we run that code on the real data -> the numbers come from the data.
# ============================================================================
ASSISTANT_SYSTEM = f"""You are a data assistant for an insurance claims event log.
The log is a pandas DataFrame called `df` with one row per event and these columns:
- CaseID (str, e.g. "Case_1") - one case = one insurance claim
- Activity (str) - one of {EXPECTED_ORDER} (normal process order)
- Timestamp (datetime64, NaT if unknown)
- User (str, User_A ... User_E) - the employee who did the activity
- Cost (int) - operational cost of that event in $. The cost of a claim is the sum of its events' costs.
- WaitingMinutes (float) - minutes since the previous event of the same case, i.e. the waiting time
  before this activity (NaN for the first event of a case)

Answer the user's question by writing Python code that uses `df`, `pd` (pandas) and `np` (numpy):
- put the answer table in a DataFrame called `result`
- put a short answer (1-2 sentences, computed from the data) in a string called `answer`
- if a chart helps, set `chart = {{"type": "bar" or "line" or "pie", "x": <column of result>, "y": <column of result>}}`,
  otherwise `chart = None`
Reply with a single ```python code block and nothing else."""

EXAMPLE_QUESTIONS = [
    "Show me all claims handled by User_A that exceeded $500",
    "Which activities have the longest waiting times and what is their percentage over the total?",
    "What is the total cost handled by each user?",
    "How many claims were registered per month?",
]


def draw_chart(result, chart):
    if chart["type"] == "pie":
        fig = px.pie(result, names=chart["x"], values=chart["y"])
    else:
        fig = (px.line if chart["type"] == "line" else px.bar)(result, x=chart["x"], y=chart["y"])
    st.plotly_chart(fig)


def page_req3():
    st.header("Requirement 3 - Ask the event log")
    model = st.selectbox("LLM", list(MODELS), key="model3")
    example = st.selectbox("Example questions", EXAMPLE_QUESTIONS)
    question = st.text_input("Your question", value=example)

    if st.button("Ask", type="primary"):
        try:
            reply = ask_llm(model, ASSISTANT_SYSTEM, question)
            code = extract_code(reply["text"])
            out = run_code(code, {"df": LOG_WITH_WAITS.copy(), "pd": pd, "np": np})
            st.success(out.get("answer", ""))
            st.dataframe(out["result"])
            if out.get("chart"):
                draw_chart(out["result"], out["chart"])
            with st.expander("Code written by the LLM"):
                st.code(code, language="python")
            show_usage(reply)
        except Exception as e:
            st.error(f"{type(e).__name__}: {e}")


# ============================================================================
# ---- REQUIREMENT 4: trace summary with enhancement ----
# Python computes the facts of the trace (durations, costs, comparison with the average);
# the LLM only turns these facts into a business summary -> it cannot invent numbers.
# ============================================================================
SUMMARY_SYSTEM = "You are a process-mining analyst at an insurance company. You explain one claim's trace to a manager."

SUMMARY_PROMPT = """Claim: {case_id}
Events in recorded order (WaitingMinutes = time since the previous event of this claim,
AvgWaitingMinutes = average waiting time of that activity over all claims):
{facts}
Total cost: ${total_cost}. Total duration from first to last event: {duration}.
Known data-quality problems in this trace: {problems}

Write a summary of 3-5 sentences: who did each stage and how long it took, the total cost,
and the bottleneck (the stage that waited the longest compared to its average). If there are
data-quality problems, say how they affect the conclusions. Use ONLY the facts above.
Style example (for a different claim): "This claim was registered by User A, verified within 2 hours,
but was under evaluation for 5 days, costing the company $450 in operational overhead.
The customer experienced a bottleneck at stage 3."
"""


def trace_facts(case_id):
    """Table of facts for one case + the average waiting time of each activity in the whole log."""
    average = LOG_WITH_WAITS.groupby("Activity")["WaitingMinutes"].mean()
    case = LOG_WITH_WAITS[LOG_WITH_WAITS["CaseID"] == case_id]
    facts = case[["Activity", "User", "Timestamp", "WaitingMinutes", "Cost"]].copy()
    facts["AvgWaitingMinutes"] = facts["Activity"].map(average).round(0)
    return facts


def page_req4():
    st.header("Requirement 4 - Summarise a trace")
    model = st.selectbox("LLM", list(MODELS), key="model4")
    case_id = st.selectbox("Case", LOG["CaseID"].unique())
    facts = trace_facts(case_id)
    st.dataframe(facts, hide_index=True)

    if st.button("Summarise", type="primary"):
        problems = ERRORS[ERRORS["CaseID"] == case_id]
        prompt = SUMMARY_PROMPT.format(
            case_id=case_id,
            facts=facts.to_csv(index=False),
            total_cost=facts["Cost"].sum(),
            duration=facts["Timestamp"].max() - facts["Timestamp"].min(),
            problems="; ".join(problems["Explanation"]) if len(problems) else "none",
        )
        try:
            reply = ask_llm(model, SUMMARY_SYSTEM, prompt)
            st.info(reply["text"])
            show_usage(reply)
        except Exception as e:
            st.error(f"{type(e).__name__}: {e}")


# ============================================================================
# ---- REQUIREMENT 5: repair the timestamps with 2 LLMs x 2 methods ----
# The LLM writes a function fix_timestamps(df) using the chosen method (statistics or ML).
# We run it on the raw log, save the repaired log and score it (Requirement 6).
# ============================================================================
REPAIR_SYSTEM = "You are a data engineer who cleans process-mining event logs with Python."

REPAIR_PROMPT = """An insurance claims event log has timestamp errors. Every case should contain these
5 activities in this order: {order}.
The errors are of four kinds:
1. Missing timestamp (empty value).
2. Truncated timestamp: only the hour is stored, e.g. "2026-09-13T16" - the true time is within that hour.
3. Negative duration: an event is timestamped earlier than the event recorded just before it.
4. Impossible order: 'Approve Claim' is timestamped before 'Evaluate Claim'.

Write a Python function `fix_timestamps(df)` that repairs these errors.
- Input `df`: pandas DataFrame with columns CaseID, Activity, Timestamp, User, Cost.
  Timestamp is the text exactly as in the file (NaN when empty).
- Output: a DataFrame with the same rows and columns, Timestamp converted to datetime and repaired.
  Change ONLY the timestamps that are wrong. Do not change any other column or any correct timestamp.
- You may use pandas, numpy and scikit-learn.

Avoid these common mistakes (the code runs with pandas 3 and scikit-learn 1.9):
- Empty timestamps are NaN (a float), not text: check them with pd.isna() before using any string method.
- Parse text with pd.to_datetime(..., errors="coerce"). Do date arithmetic only with pandas Timestamps and
  pd.Timedelta (e.g. pd.Timedelta(seconds=x)), never by adding plain numbers to dates.
- Turn a duration into seconds with pd.Timedelta(d).total_seconds() - numpy timedelta64 has no .total_seconds().
- Use OneHotEncoder(sparse_output=False); the old argument sparse= no longer exists.
- Keep the original index and row order: do not add, drop or reorder rows, and write results back by index.

Method you MUST use: {method}

First rows of the log:
{sample}
Reply with a single ```python code block containing the imports and the function."""

METHODS = {
    "Solution A - Statistical": (
        "statistical",
        "STATISTICAL processing - from the correct cases compute the typical (median) duration between "
        "consecutive activities and use it to estimate the wrong or missing timestamps "
        "(e.g. previous timestamp + median duration).",
    ),
    "Solution B - Machine learning": (
        "ml",
        "MACHINE-LEARNING prediction - train a regression model (e.g. scikit-learn RandomForestRegressor) "
        "on the correct events to predict the duration since the previous activity from features such as "
        "Activity, User and Cost, and use its predictions to estimate the wrong or missing timestamps.",
    ),
}


def run_repair(model, solution):
    """Ask the LLM for the repair code, run it, score it and save everything in output/."""
    short_name, method_text = METHODS[solution]
    prompt = REPAIR_PROMPT.format(order=EXPECTED_ORDER, method=method_text, sample=RAW.head(10).to_csv(index=False))
    reply = ask_llm(model, REPAIR_SYSTEM, prompt)
    code = extract_code(reply["text"])

    start = time.time()
    try:
        fixed = run_code(code, {})["fix_timestamps"](RAW.copy())
        fixed = fixed[["CaseID", "Activity", "Timestamp", "User", "Cost"]]
        fixed["Timestamp"] = pd.to_datetime(fixed["Timestamp"], errors="coerce", utc=True).dt.tz_localize(None)
        error = ""
    except Exception as e:  # the generated code did not run
        fixed, error = None, f"{type(e).__name__}: {e}"
    exec_seconds = round(time.time() - start, 2)

    tag = f"{short_name}_{MODELS[model][0].replace('/', '-')}"  # e.g. ml_openai-gpt-oss-120b
    with open(f"output/req5_code_{tag}.py", "w", encoding="utf-8") as f:
        f.write(code)
    if fixed is not None:
        fixed.to_csv(f"output/repaired_{tag}.csv", index=False)  # Requirement 2 also mines these logs

    row = {"solution": solution, "model": model, **evaluate(fixed), "error": error,
           "llm_seconds": reply["seconds"], "exec_seconds": exec_seconds,
           "total_seconds": round(reply["seconds"] + exec_seconds, 2),
           "input_tokens": reply["input_tokens"], "output_tokens": reply["output_tokens"],
           "total_tokens": reply["input_tokens"] + reply["output_tokens"]}
    save_result(row)
    return row, code, fixed


def save_result(row):
    """Keep one (latest) row per solution x model combination."""
    results = load_results()
    if len(results):
        results = results[~((results["solution"] == row["solution"]) & (results["model"] == row["model"]))]
    pd.concat([results, pd.DataFrame([row])]).to_csv(RESULTS_FILE, index=False)


def load_results():
    return pd.read_csv(RESULTS_FILE) if os.path.exists(RESULTS_FILE) else pd.DataFrame()


def page_req5():
    st.header("Requirement 5 - Repair the timestamps")
    with st.expander(f"Errors to repair (found in Requirement 1): {len(ERRORS)} events in {len(ERROR_CASES)} cases"):
        st.dataframe(ERRORS, hide_index=True)

    col1, col2 = st.columns(2)
    model = col1.selectbox("LLM", list(MODELS), key="model5")
    solution = col2.selectbox("Method", list(METHODS))
    jobs = []
    if col1.button("Run this combination", type="primary"):
        jobs = [(model, solution)]
    if col2.button("Run all 4 combinations"):
        jobs = [(m, s) for s in METHODS for m in MODELS]

    for model, solution in jobs:
        with st.spinner(f"{solution} with {model} ..."):
            try:
                row, code, fixed = run_repair(model, solution)
            except Exception as e:
                st.error(f"{type(e).__name__}: {e}")
                continue
        st.subheader(f"{solution} - {model}")
        if row["error"]:
            st.error(f"The generated code failed: {row['error']}")
        else:
            compared = compare_with_original(fixed)
            st.write("Timestamps that the code changed:")
            st.dataframe(compared[compared["changed"]][["CaseID", "Activity", "Timestamp", "Timestamp_new"]],
                         hide_index=True)
        st.write({metric: row[metric] for metric in METRICS})
        with st.expander("Code written by the LLM"):
            st.code(code, language="python")


# ============================================================================
# ---- REQUIREMENT 6: error metrics, one 2x2 table per metric ----
# Unit of evaluation = a case (claim). Ground truth = the cases with errors from Requirement 1.
# A case counts as "repaired" (predicted positive) if the LLM's code changed at least one of its timestamps.
# ============================================================================
METRICS = {
    "accuracy": "Share of all 100 cases handled correctly (erroneous ones changed, correct ones left alone).",
    "precision": "Of the cases the code changed, how many really had an error.",
    "recall": "Of the cases with an error, how many the code changed.",
    "f1": "Harmonic mean of precision and recall.",
    "correctness": "Of the cases with an error, how many are now valid: all 5 timestamps present, in the correct "
                   "process order, and truncated timestamps kept inside their known hour.",
    "hallucination": "Share of rows the code invented, lost, or altered in columns it was told not to touch "
                     "(code that crashes = 1.0, the whole output is lost).",
}
TRUNCATED = ERRORS[ERRORS["ErrorType"] == "Truncated timestamp"]


def compare_with_original(fixed):
    """Original and repaired log side by side (one row per event) + a 'changed' flag."""
    m = LOG.merge(fixed, on=["CaseID", "Activity"], how="outer", suffixes=("", "_new"), indicator=True)
    same = (m["Timestamp"] == m["Timestamp_new"]) | (m["Timestamp"].isna() & m["Timestamp_new"].isna())
    m["changed"] = ~same
    return m


def is_valid_case(case_id, fixed):
    times = fixed[fixed["CaseID"] == case_id].set_index("Activity")["Timestamp"].reindex(EXPECTED_ORDER)
    if times.isna().any() or not (times.diff().dropna() > pd.Timedelta(0)).all():
        return False
    for _, e in TRUNCATED[TRUNCATED["CaseID"] == case_id].iterrows():
        if times[e["Activity"]].floor("h") != pd.Timestamp(e["Timestamp"]):
            return False
    return True


def evaluate(fixed):
    all_cases = set(LOG["CaseID"])
    if fixed is None:  # the code crashed: nothing repaired, all output lost
        changed, valid, bad_rows = set(), set(), len(LOG)
    else:
        m = compare_with_original(fixed)
        bad_rows = ((m["_merge"] != "both") | (m["User"] != m["User_new"]) | (m["Cost"] != m["Cost_new"])).sum()
        changed = set(m.loc[m["changed"], "CaseID"])
        valid = {c for c in ERROR_CASES if is_valid_case(c, fixed)}

    tp = len(changed & ERROR_CASES)            # erroneous case, changed
    fp = len(changed - ERROR_CASES)            # correct case, changed (should not have been)
    fn = len(ERROR_CASES - changed)            # erroneous case, not changed
    tn = len(all_cases) - tp - fp - fn         # correct case, left alone
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn)
    return {
        "accuracy": round((tp + tn) / len(all_cases), 3),
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(2 * precision * recall / (precision + recall), 3) if precision + recall else 0.0,
        "correctness": round(len(valid) / len(ERROR_CASES), 3),
        "hallucination": round(float(bad_rows) / len(LOG), 3),
    }


def table_2x2(results, column):
    """Rows = Solution A / B (timestamp method), columns = Model A / B (LLM)."""
    return results.pivot(index="solution", columns="model", values=column)


def page_req6():
    st.header("Requirement 6 - Error metrics (rows = solution, columns = LLM)")
    results = load_results()
    if results.empty:
        st.info("Run the repairs in the 'Req 5' tab first.")
        return
    for metric, meaning in METRICS.items():
        st.subheader(metric.capitalize())
        st.caption(meaning)
        st.dataframe(table_2x2(results, metric))
    st.subheader("All runs")
    st.dataframe(results, hide_index=True)


# ============================================================================
# ---- REQUIREMENT 7: execution time (seconds) and cost (tokens) ----
# Both LLMs are used on a free tier, so the cost is measured in tokens.
# ============================================================================
def page_req7():
    st.header("Requirement 7 - Time and cost of each LLM / solution")
    results = load_results()
    if results.empty:
        st.info("Run the repairs in the 'Req 5' tab first.")
        return
    for column, title in [("total_seconds", "Time in seconds (LLM answer + running its code)"),
                          ("total_tokens", "Cost in tokens (input + output)")]:
        st.subheader(title)
        st.dataframe(table_2x2(results, column))
        st.plotly_chart(px.bar(results, x="solution", y=column, color="model", barmode="group"))
    st.dataframe(results[["solution", "model", "llm_seconds", "exec_seconds", "input_tokens",
                          "output_tokens"]], hide_index=True)


# ============================================================================
# ---- Page layout: one tab per requirement ----
# ============================================================================
st.set_page_config(page_title="Insurance Claims - BPM & GenAI", layout="wide")
st.title("Insurance Claims - Process Mining & GenAI assistant")
tabs = st.tabs(["Req 3 - Assistant", "Req 4 - Trace summary", "Req 5 - Timestamp repair",
                "Req 6 - Error metrics", "Req 7 - Time & cost"])
for tab, page in zip(tabs, [page_req3, page_req4, page_req5, page_req6, page_req7]):
    with tab:
        page()
