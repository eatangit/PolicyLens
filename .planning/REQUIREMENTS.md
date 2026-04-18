# Requirements — PolicyLens

**Project:** AI Government Policy Evaluator (DataHacks 2026)
**Status:** v1 scoped | **Last updated:** 2026-04-18

---

## v1 Requirements

### Data & State Layer

- [ ] **DATA-01**: System fetches live EIA baseline state from at least 10 distinct EIA v2 API endpoints covering all 31 state variables
- [ ] **DATA-02**: System falls back to `data/sample_state.json` (hardcoded 2023 US baseline values) when EIA API is unavailable or key is missing
- [ ] **DATA-03**: All 31 state variables are enumerated in a single shared constant (`VAR_INDEX` dict mapping name → index) used by parser, env, and model
- [ ] **DATA-04**: EIA loader parses all numeric values as `float` (EIA v2 returns values as JSON strings since Jan 2024) with `nan` fallback for missing data
- [ ] **DATA-05**: EIA loader handles 5,000-row pagination limit via offset-based pagination and caches results locally to CSV/parquet before hackathon
- [ ] **DATA-06**: RobustScaler (median + IQR) is fit on chronological training split (1960–2005) only, serialized to `data/normalizer_params.json`, and loaded at inference

### State Variable Definitions (31 total)

**Environmental (15 variables):**

- [ ] **ENV-01**: `co2_total_emissions` — Total US CO2 from energy consumption (MMT CO2) — SEDS
- [ ] **ENV-02**: `co2_electric_power` — CO2 from electric power sector (MMT CO2) — SEDS / electric-power-operational-data
- [ ] **ENV-03**: `co2_transportation` — CO2 from transportation sector (MMT CO2) — SEDS
- [ ] **ENV-04**: `co2_industrial` — CO2 from industrial sector (MMT CO2) — SEDS
- [ ] **ENV-05**: `coal_consumption` — Total coal consumption by sector (thousand short tons) — coal/consumption-and-quality
- [ ] **ENV-06**: `petroleum_supplied` — Total petroleum products supplied (thousand barrels/day) — petroleum/cons
- [ ] **ENV-07**: `natural_gas_consumed` — Natural gas total consumption (Bcf) — natural-gas/cons
- [ ] **ENV-08**: `renewable_generation_share` — Renewables as % of total electricity generation — electricity/electric-power-operational-data
- [ ] **ENV-09**: `solar_generation` — Solar electricity generation (GWh) — electricity/electric-power-operational-data
- [ ] **ENV-10**: `wind_generation` — Wind electricity generation (GWh) — electricity/electric-power-operational-data
- [ ] **ENV-11**: `nuclear_generation` — Nuclear electricity generation (GWh) — electricity/electric-power-operational-data
- [ ] **ENV-12**: `renewable_installed_capacity` — Total renewable installed capacity (GW) — electricity/operating-generator-capacity
- [ ] **ENV-13**: `nuclear_outage_rate` — Nuclear capacity outage rate (%) — nuclear-outages/us-nuclear-outages
- [ ] **ENV-14**: `total_electricity_demand` — Total US electricity demand (GWh) — electricity/rto
- [ ] **ENV-15**: `biomass_production` — Densified biomass production (thousand tons) — densified-biomass/production-by-region

**Economic (16 variables):**

- [ ] **ECON-01**: `retail_electricity_price` — Average retail electricity price (cents/kWh) — electricity/retail-sales
- [ ] **ECON-02**: `natural_gas_price_consumer` — Natural gas delivered price to consumers ($/Mcf) — natural-gas/pri
- [ ] **ECON-03**: `gasoline_retail_price` — Regular gasoline retail price ($/gallon) — petroleum/pri
- [ ] **ECON-04**: `diesel_retail_price` — Diesel retail price ($/gallon) — petroleum/pri
- [ ] **ECON-05**: `coal_market_price` — Average coal market sales price ($/short ton) — coal/market-sales-price
- [ ] **ECON-06**: `energy_expenditure_per_household` — Annual household energy expenditure ($) — seds (expenditures)
- [ ] **ECON-07**: `govt_energy_expenditures` — Total government energy expenditures ($B) — seds (expenditures)
- [ ] **ECON-08**: `coal_employment` — Coal mining employment (thousands of workers) — coal/aggregate-production
- [ ] **ECON-09**: `natural_gas_production` — Natural gas marketed production (Bcf) — natural-gas/prod
- [ ] **ECON-10**: `crude_oil_production` — Crude oil production (thousand barrels/day) — petroleum/crd
- [ ] **ECON-11**: `crude_oil_imports` — Crude oil imports (thousand barrels/day) — crude-oil-imports
- [ ] **ECON-12**: `petroleum_exports` — Total petroleum exports (thousand barrels/day) — petroleum/move
- [ ] **ECON-13**: `natural_gas_imports` — Natural gas imports (Bcf) — natural-gas/move
- [ ] **ECON-14**: `petroleum_stocks` — Total petroleum stocks (million barrels) — petroleum/stoc
- [ ] **ECON-15**: `energy_trade_balance` — Net energy trade balance ($B) — petroleum/move + natural-gas/move
- [ ] **ECON-16**: `steo_price_index` — STEO projected energy price index (index, 2012=100) — steo

### NL Parser

- [ ] **PARSE-01**: User inputs plain English policy text; Claude API parses it into a structured action dict mapping each relevant variable to `{action: float[-1,1], reasoning: str, confidence: str}`
- [ ] **PARSE-02**: Parser output includes a top-level `confidence` field: `"high" | "medium" | "low" | "out_of_domain"`; all action magnitudes are zeroed when `out_of_domain`
- [ ] **PARSE-03**: Parser schema enforces `reasoning` field before `magnitude` field (both required) to improve LLM chain-of-thought quality
- [ ] **PARSE-04**: All action magnitudes are clamped to `[0.0, 1.0]` post-parse in application code, not relied on from LLM output alone
- [ ] **PARSE-05**: System prompt includes 3–5 calibration examples mapping specific policy language to magnitude values
- [ ] **PARSE-06**: Parser produces a binary relevance mask (31-dim bool array) alongside the 31-dim float action vector

### RL Environment

- [ ] **ENV-RL-01**: `PolicyEnv` implements Gymnasium interface with `observation_space=Box(-inf, inf, (31,))` and `action_space=Box(-1, 1, (31,))`
- [ ] **ENV-RL-02**: `reset()` returns `(obs, info)` tuple; resets GRU hidden state `h` to None
- [ ] **ENV-RL-03**: `step()` returns 5-tuple `(obs, reward, terminated, truncated, info)`; `terminated=False` always; `truncated=True` when horizon reached
- [ ] **ENV-RL-04**: `rollout(action_vec, mask_vec, steps=10, decay=0.85)` convenience method runs full episode and returns trajectory array of shape `(steps+1, 31)`
- [ ] **ENV-RL-05**: Policy action decays exponentially at `0.85/year`; at step `t`, effective action = `action * 0.85^t`
- [ ] **ENV-RL-06**: All state variables are clamped to EIA historical physical bounds after every step to prevent autoregressive drift
- [ ] **ENV-RL-07**: `gymnasium.utils.env_checker.check_env()` passes with zero warnings/errors

### Transition Model

- [ ] **MODEL-01**: `PolicyTransitionModel` is a 2-layer GRU with `hidden_size=64`, `input_size=62` (31 state + 31 action), `dropout=0.2`, predicting delta-state of shape `(31,)`
- [ ] **MODEL-02**: `DemoTransitionModel` is a deterministic linear fallback: ENV variables shift at `±4% per step * action magnitude`, ECON variables at `±2% per step * action magnitude`
- [ ] **MODEL-03**: `PolicyEnv` loads `DemoTransitionModel` by default; switches to `PolicyTransitionModel` when `models/policy_transition.pt` exists
- [ ] **MODEL-04**: Model forward pass: `concat(normalized_state, masked_action)` → GRU → delta → `next_state = state + delta`
- [ ] **MODEL-05**: Model file and normalizer params are versioned together (same checkpoint directory)

### Training Pipeline

- [ ] **TRAIN-01**: `src/train.py` implements PyTorch training loop with chronological split (train 1960–2005, val 2006–2015, test 2016–2023); DataLoader never shuffles
- [ ] **TRAIN-02**: Training uses sliding window sequences of width 5; loss is HuberLoss on predicted vs actual delta; optimizer is Adam with `lr=1e-3, weight_decay=1e-4`
- [ ] **TRAIN-03**: Gradient clipping `max_norm=1.0` applied before every `optimizer.step()`
- [ ] **TRAIN-04**: `ReduceLROnPlateau` scheduler with patience=5; early stopping with patience=30 on validation loss
- [ ] **TRAIN-05**: `src/data/README.md` documents exact steps to download all EIA data and produce training CSVs

### Report Generator

- [ ] **REPORT-01**: Claude API generates a plain-English report given the policy text, initial state, and projected trajectory
- [ ] **REPORT-02**: Report has two sections: **Short-Term Effects (1–2 years)** and **Long-Term Effects (up to 10 years)**, each 2 paragraphs
- [ ] **REPORT-03**: Report uses streaming output displayed live in the Streamlit UI
- [ ] **REPORT-04**: Report avoids jargon; explains tradeoffs in terms understandable to politicians and the general public
- [ ] **REPORT-05**: Report system prompt instructs Claude to cite specific metric deltas from the trajectory data (e.g., "CO2 emissions are projected to fall by 12% within 2 years")

### Streamlit UI

- [ ] **UI-01**: Dark aesthetic theme set via `.streamlit/config.toml` (`base = "dark"`)
- [ ] **UI-02**: Policy text input (`st.text_area`) with placeholder example and an "Evaluate Policy" button
- [ ] **UI-03**: **Policy Summary Card** displays Claude's interpretation and `reasoning` field before showing results — trust anchor for users
- [ ] **UI-04**: `st.metric()` cards showing before/after deltas for 4 headline metrics: CO2 total, renewable share, consumer electricity price, coal employment
- [ ] **UI-05**: Line charts (Plotly, single-axis) showing baseline vs. projected trajectory over 10 years for environmental metrics and economic metrics (separate charts)
- [ ] **UI-06**: **2×2 Tradeoff Score quadrant** plot showing where the policy lands: Win-Win / Env-Win-Econ-Lose / Econ-Win-Env-Lose / Lose-Lose
- [ ] **UI-07**: **Policy Ambiguity Warning** banner (yellow/amber) displayed when parser returns `confidence: low` or `out_of_domain`
- [ ] **UI-08**: **Demo Mode disclosure** banner (always visible when GRU model is not loaded) stating results use simplified linear projection
- [ ] **UI-09**: All Claude and EIA API calls gated behind `st.button`; parsed action vector stored in `st.session_state` to prevent rerun re-calls
- [ ] **UI-10**: App handles missing `ANTHROPIC_API_KEY` gracefully with a clear error message and demo-mode fallback

---

## v2 Requirements (Deferred)

- State-level anchoring — dropdown to switch baseline to SEDS state-level data (clean stretch goal, requires only route swap)
- Year-by-year narrative streaming — Claude streaming one sentence per projected year (impressive demo effect, lower priority)
- Side-by-side policy comparison — compare two policies simultaneously (high scope risk)
- GRU uncertainty estimates — per-step confidence intervals (requires distributional GRU or ensemble)
- Historical policy backtesting — compare model predictions against known policy outcomes

---

## Out of Scope

- Direct climate metrics (atmosphere, ocean acidity, temperature) — environmental health is energy-based only
- International policy simulation — US only
- Demographic breakdowns by income or race — requires weighting models not in EIA data
- Health or air quality metrics — requires epidemiological chain not available from EIA
- User authentication or saved reports — single-user demo tool
- Interactive sliders for policy parameters — abandons the NL value proposition
- Dual-axis charts — documented as misleading for non-experts
- Confidence intervals on untrained/linear model — statistically dishonest
- Real-time grid data visualization — policy operates on annual timescales

---

## Traceability

| REQ-ID | Phase | Status |
|--------|-------|--------|
| DATA-01 | Phase 1 — Data Foundation | Pending |
| DATA-02 | Phase 1 — Data Foundation | Pending |
| DATA-03 | Phase 1 — Data Foundation | Pending |
| DATA-04 | Phase 1 — Data Foundation | Pending |
| DATA-05 | Phase 1 — Data Foundation | Pending |
| DATA-06 | Phase 1 — Data Foundation | Pending |
| ENV-01 | Phase 1 — Data Foundation | Pending |
| ENV-02 | Phase 1 — Data Foundation | Pending |
| ENV-03 | Phase 1 — Data Foundation | Pending |
| ENV-04 | Phase 1 — Data Foundation | Pending |
| ENV-05 | Phase 1 — Data Foundation | Pending |
| ENV-06 | Phase 1 — Data Foundation | Pending |
| ENV-07 | Phase 1 — Data Foundation | Pending |
| ENV-08 | Phase 1 — Data Foundation | Pending |
| ENV-09 | Phase 1 — Data Foundation | Pending |
| ENV-10 | Phase 1 — Data Foundation | Pending |
| ENV-11 | Phase 1 — Data Foundation | Pending |
| ENV-12 | Phase 1 — Data Foundation | Pending |
| ENV-13 | Phase 1 — Data Foundation | Pending |
| ENV-14 | Phase 1 — Data Foundation | Pending |
| ENV-15 | Phase 1 — Data Foundation | Pending |
| ECON-01 | Phase 1 — Data Foundation | Pending |
| ECON-02 | Phase 1 — Data Foundation | Pending |
| ECON-03 | Phase 1 — Data Foundation | Pending |
| ECON-04 | Phase 1 — Data Foundation | Pending |
| ECON-05 | Phase 1 — Data Foundation | Pending |
| ECON-06 | Phase 1 — Data Foundation | Pending |
| ECON-07 | Phase 1 — Data Foundation | Pending |
| ECON-08 | Phase 1 — Data Foundation | Pending |
| ECON-09 | Phase 1 — Data Foundation | Pending |
| ECON-10 | Phase 1 — Data Foundation | Pending |
| ECON-11 | Phase 1 — Data Foundation | Pending |
| ECON-12 | Phase 1 — Data Foundation | Pending |
| ECON-13 | Phase 1 — Data Foundation | Pending |
| ECON-14 | Phase 1 — Data Foundation | Pending |
| ECON-15 | Phase 1 — Data Foundation | Pending |
| ECON-16 | Phase 1 — Data Foundation | Pending |
| PARSE-01 | Phase 2 — NL Parser | Pending |
| PARSE-02 | Phase 2 — NL Parser | Pending |
| PARSE-03 | Phase 2 — NL Parser | Pending |
| PARSE-04 | Phase 2 — NL Parser | Pending |
| PARSE-05 | Phase 2 — NL Parser | Pending |
| PARSE-06 | Phase 2 — NL Parser | Pending |
| ENV-RL-01 | Phase 3 — Env + UI Skeleton | Pending |
| ENV-RL-02 | Phase 3 — Env + UI Skeleton | Pending |
| ENV-RL-03 | Phase 3 — Env + UI Skeleton | Pending |
| ENV-RL-04 | Phase 3 — Env + UI Skeleton | Pending |
| ENV-RL-05 | Phase 3 — Env + UI Skeleton | Pending |
| ENV-RL-06 | Phase 3 — Env + UI Skeleton | Pending |
| ENV-RL-07 | Phase 3 — Env + UI Skeleton | Pending |
| MODEL-01 | Phase 3 — Env + UI Skeleton | Pending |
| MODEL-02 | Phase 3 — Env + UI Skeleton | Pending |
| MODEL-03 | Phase 3 — Env + UI Skeleton | Pending |
| UI-01 | Phase 3 — Env + UI Skeleton | Pending |
| UI-02 | Phase 3 — Env + UI Skeleton | Pending |
| UI-03 | Phase 3 — Env + UI Skeleton | Pending |
| UI-07 | Phase 3 — Env + UI Skeleton | Pending |
| UI-08 | Phase 3 — Env + UI Skeleton | Pending |
| UI-09 | Phase 3 — Env + UI Skeleton | Pending |
| UI-10 | Phase 3 — Env + UI Skeleton | Pending |
| REPORT-01 | Phase 4 — Reports + Differentiators | Pending |
| REPORT-02 | Phase 4 — Reports + Differentiators | Pending |
| REPORT-03 | Phase 4 — Reports + Differentiators | Pending |
| REPORT-04 | Phase 4 — Reports + Differentiators | Pending |
| REPORT-05 | Phase 4 — Reports + Differentiators | Pending |
| MODEL-04 | Phase 4 — Reports + Differentiators | Pending |
| MODEL-05 | Phase 4 — Reports + Differentiators | Pending |
| UI-04 | Phase 4 — Reports + Differentiators | Pending |
| UI-05 | Phase 4 — Reports + Differentiators | Pending |
| UI-06 | Phase 4 — Reports + Differentiators | Pending |
| TRAIN-01 | Phase 5 — GRU Training | Pending |
| TRAIN-02 | Phase 5 — GRU Training | Pending |
| TRAIN-03 | Phase 5 — GRU Training | Pending |
| TRAIN-04 | Phase 5 — GRU Training | Pending |
| TRAIN-05 | Phase 5 — GRU Training | Pending |
