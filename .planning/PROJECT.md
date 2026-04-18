# PolicyLens

## What This Is

PolicyLens is an AI-powered government policy evaluation tool that simulates the economic and environmental tradeoffs of US energy and environmental policies. A user inputs any policy in plain English — a carbon tax, a renewable energy mandate, an oil drilling expansion — and the system produces a structured impact report covering short-term (1–2 year) and long-term (up to 10 year) effects on both the US economy and human-produced energy metrics. The tool is designed for politicians and the general public: no jargon, clear tradeoffs, accessible prose.

## Core Value

Given any plain-English government policy, produce a credible, data-grounded impact report showing the economic vs. environmental tradeoff — fast enough to be useful in a live policy discussion.

## Requirements

### Validated

(None yet — ship to validate)

### Active

- [ ] Natural language policy input is parsed into a structured RL action vector via Claude API
- [ ] RL environment models 31 US state variables (15 environmental, 16 economic) from EIA datasets
- [ ] Policy-irrelevant variables are masked and excluded from simulation
- [ ] GRU-based neural transition model predicts state trajectory over 10 annual time steps
- [ ] Demo mode uses simplified linear projection when model is untrained (hackathon fallback)
- [ ] Report generator (Claude API) converts trajectory into a 2-section plain-English report (short-term + long-term)
- [ ] Streamlit UI with dark aesthetic theme, policy textbox, and formatted report output
- [ ] EIA API data loader fetches live baseline state from 15+ endpoints
- [ ] Baseline state falls back to hardcoded 2023 US values when EIA API is unavailable
- [ ] Training pipeline (PyTorch) with dataset class for historical EIA state sequences
- [ ] Charts in UI showing key metric trajectories (baseline vs. policy) over 10 years
- [ ] All EIA variable routes mapped and documented with download/training instructions

### Out of Scope

- Direct climate/atmosphere/ocean metrics — environmental health is energy-production-based only
- International policy simulation — US only
- Real-time policy training — model structure is provided; training happens offline
- Authentication or multi-user sessions — single-user demo tool
- Economic metrics beyond the energy sector (e.g., general GDP, unemployment rate)

## Context

- **Hackathon**: DataHacks 2026 at UC San Diego — prioritize working demo, visual polish, and clear narrative over production robustness
- **Dataset**: EIA OpenData API (Energy Information Administration) — all state variable routes are pre-specified in the brief
- **RL framing**: The "RL" here is model-based policy evaluation, not agent training — environment follows Gym interface, model is a GRU transition network trained on historical state sequences
- **LLM**: Claude Sonnet 4.6 (Anthropic) for both the NL→action parser and the report generator
- **Demo viability**: The end-to-end pipeline must work at demo time even before the GRU model is trained, using a simplified linear projection fallback

## Constraints

- **Tech stack**: Python only — Streamlit, PyTorch, Gymnasium, Anthropic SDK, Plotly
- **Timeline**: DataHacks 2026 hackathon deadline — demo-first, training-later architecture required
- **Data**: EIA API requires a free API key; all routes documented for judges to verify; sample_state.json fallback covers offline demos
- **Model**: GRU transition model is untrained at submission — demo mode runs without it; training instructions provided for post-hackathon use

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| GRU-based recurrent transition model | Temporal dynamics (how policy effects decay/amplify over years) require recurrence; simpler than Transformer, well-suited to short time horizons | — Pending |
| Claude API for NL→action parsing | Structured JSON output, instruction-following quality required for accurate variable selection and magnitude estimation | — Pending |
| 31 state variables (15 env / 16 econ) | EIA requirements specified; balanced parity between environmental and economic to avoid implicit weighting bias | — Pending |
| Demo mode linear fallback | Trained model won't exist at hackathon time; demo must still run end-to-end | — Pending |
| Streamlit UI with Plotly charts | Fast to build, sufficient aesthetics with custom CSS, native Python | — Pending |
| Action encoded as float[-1,1] + binary mask | Enables variable-level relevance gating; mask fed to model so it can learn to ignore irrelevant dimensions | — Pending |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `/gsd-transition`):
1. Requirements invalidated? → Move to Out of Scope with reason
2. Requirements validated? → Move to Validated with phase reference
3. New requirements emerged? → Add to Active
4. Decisions to log? → Add to Key Decisions
5. "What This Is" still accurate? → Update if drifted

**After each milestone** (via `/gsd-complete-milestone`):
1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-04-18 after initialization*
