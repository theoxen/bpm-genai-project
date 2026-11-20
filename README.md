# BPM-GenAI project - Insurance claims event log

## Where each requirement is

| Requirement | File | What it does |
|---|---|---|
| 1 - Timestamp errors | `req1_timestamp_errors.py` | 4 code checks (no LLM) find the wrong timestamps; PM4PY shows the effect as odd variants |
| 2 - Alpha / Heuristic / Inductive | `req2_process_discovery.py` | Petri nets with PM4PY on the original log, the error-free cases and the LLM-repaired logs; quality table |
| 3 - Digital assistant (GUI) | `app.py` -> section `REQUIREMENT 3` | Question -> LLM writes pandas code -> we run it -> table + chart |
| 4 - Trace summary | `app.py` -> section `REQUIREMENT 4` | Python computes the trace facts -> LLM writes the business summary |
| 5 - Repair timestamps, 2 LLMs x 2 methods | `app.py` -> section `REQUIREMENT 5` | LLM writes `fix_timestamps(df)` with statistics **or** ML -> we run and save it |
| 6 - Error metrics (2x2 tables) | `app.py` -> section `REQUIREMENT 6` | Accuracy, Precision, Recall, F1, Correctness, Hallucination |
| 7 - Time & cost | `app.py` -> section `REQUIREMENT 7` | Seconds and tokens per LLM / method |
| 8 - Report + web link | (you) | Deploy `app.py` (see below) and put the link on the first page |
| shared | `llm.py` | The two free LLMs from two companies: **Model A = Google Gemini 3.5 Flash**, **Model B = OpenAI gpt-oss-120b** (open model, hosted by Groq) |

Outputs (CSV tables, Petri nets, LLM code, repaired logs) are written to `output/`.

## Setup (once)

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
```

- **Graphviz** (only for the Petri-net PNG pictures in Req 2): install from https://graphviz.org/download/
  and tick "add to PATH". Without it no pictures are saved (the comparison table still is).
- **Two free API keys** (for the GUI), no credit card needed:
  Gemini at https://aistudio.google.com/apikey and Groq at https://console.groq.com/keys.
  Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and put both keys there
  (or set the `GEMINI_API_KEY` and `GROQ_API_KEY` environment variables).
- Check the model names: `python llm.py` lists the models each service offers right now.
  If `gemini-3.5-flash` or `openai/gpt-oss-120b` is missing, change the name in `MODELS` in `llm.py`.

## Run (in this order)

```bash
python req1_timestamp_errors.py      # Req 1
python req2_process_discovery.py     # Req 2
streamlit run app.py                 # Req 3-7 (in the Req 5 tab press "Run all 4 combinations")
python req2_process_discovery.py     # optional: run again -> also mines the 4 repaired logs
```

## Web link (Req 8) - Streamlit Community Cloud (free)

1. Put this `project` folder in a GitHub repository (`.venv` and the secrets file are ignored by `.gitignore`).
2. On https://share.streamlit.io -> *Create app* -> choose the repo, main file `app.py`.
3. *Advanced settings* -> Python 3.12+ and in **Secrets** paste the two lines from your `secrets.toml`.
4. Deploy and copy the URL into the report.

Both keys are free tiers, so the app costs nothing even when others use the link. Gemini's free tier allows only
20 requests per day per model, shared by everyone using the link; Groq's limits are much higher.

## How things work (short)

**Req 1 - the 4 checks** (22 events in 22 of the 100 cases):

| Error type | Cases | How it is detected |
|---|---|---|
| Missing timestamp | 5 | empty value |
| Truncated timestamp | 5 | text is `2026-09-13T16` instead of `...T16:00:00` - minutes lost |
| Negative duration | 5 | `Approve Claim` timestamped exactly 15 min **before** the `Evaluate Claim` recorded before it |
| Impossible order | 7 | `Approve Claim` timestamped before `Evaluate Claim` (claim approved before it is evaluated -> swapped) |

**Req 2** - on the error-free cases all three miners give the same sequence R->V->E->A->C.
On the original log, Alpha becomes unsound (no noise handling), Heuristic mostly filters the errors out as
noise, and Inductive fits every case but loses precision (extra silent "skip" transitions).
PM4PY also refuses empty timestamps (it drops those events), which makes cases look like they skipped a step.

**Req 3 / 5 - "LLM writes code, Python runs it"**: the LLM never computes numbers itself. It writes pandas or
scikit-learn code and we run that code on the real data. Answers come from the data, and the code can be checked.
(Running LLM-written code with `exec` is fine for this project but would need sandboxing in production.)

**Req 6 - how a repair is scored** (unit = one case; ground truth = the 22 cases from Req 1):
a case is *predicted erroneous* if the code changed at least one of its timestamps.
TP = erroneous & changed, FP = correct but changed, FN = erroneous but not changed, TN = correct & untouched.
*Correctness* = erroneous cases that are now valid (5 timestamps, right order, truncated ones kept in their hour).
*Hallucination* = rows invented, lost, or changed in columns the LLM was told not to touch (crash = 1.0).
LLM output differs a little from run to run, so you may want to run each combination more than once and report it.

**Req 5 - literature pointers for the two methods** (check the details before citing):
- Suriadi, Andrews, ter Hofstede, Wynn (2017), *Event log imperfection patterns for process mining*, Information Systems 64 - catalogue of log errors (e.g. "inadvertent time travel").
- Rogge-Solti, Mans, van der Aalst, Weske (2013), *Improving documentation by repairing event logs*, PoEM - statistical repair of missing timestamps from typical durations.
- Dixit et al. (2018), *Detection and interactive repair of event ordering imperfections in process logs*, CAiSE.
- Breiman (2001), *Random Forests*, Machine Learning 45 - the regression model suggested for the ML method.
- Tax, Verenich, La Rosa, Dumas (2017), *Predictive business process monitoring with LSTM neural networks*, CAiSE - ML prediction of event times.
