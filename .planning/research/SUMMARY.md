# Research Summary: PolicyLens

**Project:** PolicyLens — AI Government Policy Evaluator (DataHacks 2026)
**Researched:** 2026-04-18 | **Confidence:** HIGH

---

## Executive Summary

PolicyLens takes plain-English policy descriptions, translates them into quantified action vectors via Claude, and runs a GRU-based world model over EIA energy data to project environmental and economic outcomes over a 10-year horizon. The natural-language interface — absent from every comparable tool (EPS, LEAP, NREL Scenario Viewer) — is the genuine differentiator. Build to make the NL→simulation pipeline functional end-to-end with a linear fallback before training the GRU, so the demo works regardless of training progress.

---

## Recommended Stack

| Library | Version | Notes |
|---|---|---|
| Python | 3.11 | Max compat across all deps |
| torch | 2.6.0 | `batch_first=True`; HuberLoss; no `torch.compile` on GRU |
| gymnasium | 1.0.0 | Pinned; `step()` returns 5-tuple |
| anthropic | >=0.40.0,<1.0 | Sync client only; `.stream()` for report |
| numpy | >=1.26,<2.0 | HARD PIN — ABI break in 2.x |
| pandas | >=2.0,<3.0 | JSON→tensor bridge |
| streamlit | 1.40–1.52 | Theme via `.streamlit/config.toml` only |
| plotly | >=5.20,<6.0 | 6.x not validated by Streamlit team |
| scikit-learn | current | RobustScaler (median+IQR) for normalization |
| requests | current | Direct EIA v2 REST calls — no wrapper libs |

**Rejected:** `myeia` (inactive), `eiapy` (EIA v1), `AsyncAnthropic` (Streamlit incompatible), `numpy>=2.0` (ABI break), `torch.compile` on GRU (stateful RNN limitation)

---

## Architecture

**Pipeline:** NL text → Claude parser → action vector (31-dim float + 31-dim bool mask) → PolicyEnv (Gymnasium) → GRU or DemoTransitionModel → 11-state trajectory → Claude report → Streamlit UI

**GRU sizing:** `hidden_size=64, num_layers=2` (~47K params). Only ~60 annual training samples — do NOT go above `hidden=128` or 3 layers.

**Key decisions:**
- Predict **delta** not absolute state — mandatory for 60-sample dataset stability
- Input encoding: `concat(state_31, action_31)` = 62-dim — no FiLM or cross-attention
- Normalization: **RobustScaler** per variable (median+IQR) — z-score corrupted by 2008/2020/2022 energy shocks
- Masking: zero out `effective_action[i]` for irrelevant variables; never zero the state variable
- Action decay: **0.85/year** exponential — models policy absorption as economy adjusts
- Physical bounds clamping inside `step()` after every delta — prevents autoregressive drift

**Build order (fixed by dependencies):**
1. `data.py` + normalizer → 2. `parser.py` → 3. `env.py` + linear fallback + `app.py` skeleton → 4. `report.py` + differentiators → 5. `train.py` + GRU integration

---

## Table Stakes Features

- Plain-English policy input + Claude parser
- **Policy Summary Card** — shows Claude's interpretation/reasoning (trust anchor; required)
- Baseline state display (2–3 key EIA metrics)
- 10-year projection: baseline vs. policy overlay line charts
- `st.metric()` before/after cards: CO2 delta, renewable % delta, consumer energy cost delta
- Fallback disclosure banner (always visible when demo mode active)
- Policy Ambiguity Warning when Claude returns `confidence: low` or `out_of_domain`

## Differentiator Features

- **2×2 Tradeoff Score quadrant** — Win-Win / Lose-Lose / Env-Win / Econ-Win (most legible single-screen summary for politicians)
- Year-by-Year Narrative — Claude streaming one sentence per projected year (impressive live demo)
- State-Level Anchoring — dropdown switching EIA baseline to SEDS state-level data

## Anti-Features (do not build)

- Interactive sliders (abandons the NL value proposition)
- Confidence intervals on untrained model (statistically dishonest)
- Health/air quality metrics (requires epidemiological chain not in EIA)
- Side-by-side policy comparison (scope risk — cut first)
- Dual-axis charts (documented as misleading for non-experts)

---

## Top 5 Pitfalls

1. **Unconstrained LLM magnitudes** — Post-parse clamp all action values to `[0.0, 1.0]`; put `reasoning` before `magnitude` in schema; include 3–5 calibration examples in system prompt
2. **Vague policy → confident garbage** — Add `confidence: "high"|"medium"|"low"|"out_of_domain"` enum; zero all magnitudes when out_of_domain; show yellow warning banner; test 5 adversarial inputs
3. **GRU autoregressive drift** — Default demo horizon to 5 steps; hard-clamp state to EIA historical bounds inside `step()` after every delta
4. **Temporal data leakage** — Chronological split: train 1960–2005, val 2006–2015, test 2016–2023; never shuffle; fit RobustScaler on train only
5. **EIA data gotchas** — Values returned as JSON strings since Jan 2024 (`float(row.get("value","nan"))`); 5,000-row silent truncation (check `X-Warning` header; cache locally before hackathon)

**Bonus:** Gate all Claude + EIA calls behind `st.button`; store parsed action in `st.session_state`; test on a fresh clone 30 min before deadline.

---

## Confidence Assessment

| Area | Level |
|---|---|
| Stack | HIGH — versions verified against official docs |
| Features | HIGH — benchmarked against EPS/LEAP/NREL |
| Architecture (APIs) | HIGH — Gymnasium + PyTorch API verified |
| Architecture (GRU sizing) | MEDIUM — small-dataset GRU literature is sparse |
| Pitfalls | HIGH — sourced from official changelogs and confirmed bug reports |

**Gaps:** Exact physical bounds per variable (derive from EIA historical range in Phase 1); GRU hidden_size sensitivity (try 32 if overfitting, 96 if underfitting).

---

*Synthesized: 2026-04-18 | Ready for roadmap: yes*
