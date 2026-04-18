# PolicyLens — Claude Code Guide

## Project

AI government policy evaluator for DataHacks 2026. Takes plain-English US energy/environmental policy → Claude parser → GRU RL environment → impact report showing economic vs. environmental tradeoffs.

## Key Architecture Decisions

- **31 state variables**: 15 environmental (CO2, renewables, fossil fuels) + 16 economic (prices, production, trade) — defined in `VAR_INDEX` constant in `src/data.py`
- **GRU model**: `hidden_size=64, num_layers=2`, predicts delta-state (not absolute), input = `concat(state_31, action_31)` = 62-dim
- **Action**: float[-1,1] per variable + binary relevance mask; decays at `0.85^t` per year; irrelevant variables zeroed in action, not state
- **Normalization**: RobustScaler per variable, fitted on 1960–2005 only, serialized to `data/normalizer_params.json`
- **Demo fallback**: `DemoTransitionModel` runs when `models/policy_transition.pt` absent; ENV vars ±4%/step, ECON vars ±2%/step

## Critical Rules

- **Never shuffle time-series data** — chronological split only (train 1960–2005, val 2006–2015, test 2016–2023)
- **Clamp all LLM action magnitudes** to [0.0, 1.0] in application code — never trust raw LLM floats
- **Clamp state variables** to physical bounds after every `step()` — prevents GRU autoregressive drift
- **Sync Anthropic client only** — `anthropic.Anthropic()` not `AsyncAnthropic`; Streamlit event loop incompatible with async
- **Pin numpy<2.0** — ABI break in 2.x breaks `torch.from_numpy()` silently
- **Gate all LLM + EIA calls behind `st.button`** — store parsed action in `st.session_state`
- **EIA values are JSON strings** since Jan 2024 — always `float(row.get("value", "nan"))`
- **Schema field order matters**: `reasoning` must come before `magnitude` in Claude structured output schema

## Stack

| Tool | Version |
|------|---------|
| Python | 3.11 |
| PyTorch | 2.6.0 |
| Gymnasium | 1.0.0 |
| anthropic SDK | >=0.40.0,<1.0 |
| numpy | >=1.26,<2.0 |
| streamlit | 1.40–1.52 |
| plotly | >=5.20,<6.0 |
| scikit-learn | current |

## Project Structure (target)

```
datahacks26/
├── app.py                    # Streamlit UI
├── src/
│   ├── data.py               # EIA loader, VAR_INDEX, sample_state fallback
│   ├── normalizer.py         # RobustScaler wrapper, serialize/load
│   ├── parser.py             # Claude NL → action vector + mask
│   ├── env.py                # PolicyEnv (Gymnasium), DemoTransitionModel
│   ├── model.py              # PolicyTransitionModel (GRU)
│   ├── report.py             # Claude report generator (streaming)
│   └── train.py              # PyTorch training loop
├── data/
│   ├── sample_state.json     # Hardcoded 2023 US baseline (31 variables)
│   ├── normalizer_params.json
│   └── README.md             # EIA data download instructions
├── models/
│   └── policy_transition.pt  # Trained GRU checkpoint (post-training)
├── .streamlit/
│   └── config.toml           # Dark theme
└── .planning/                # GSD planning artifacts
```

## GSD Workflow

This project uses GSD (Get Shit Done) for phased execution.

- Current roadmap: `.planning/ROADMAP.md`
- Requirements: `.planning/REQUIREMENTS.md`
- State: `.planning/STATE.md`

**Phase commands:**
- `/gsd-plan-phase N` — plan a phase before executing
- `/gsd-execute-phase N` — execute a planned phase
- `/gsd-progress` — check where we are

**Phase order (sequential, dependency-driven):**
1. Data Foundation
2. NL Parser
3. Env + UI Skeleton
4. Reports + Differentiators
5. GRU Training

Do not execute phases out of order — each phase depends on outputs from the previous.
