# Roadmap: PolicyLens

## Overview

PolicyLens is built in five dependency-driven phases. The data foundation must exist before the parser can be tested; the parser must work before the environment can consume it; the environment and linear fallback must run end-to-end before the report and UI polish layers are added; the full demo pipeline must be complete before the optional GRU training is wired in. Each phase delivers a coherent, independently verifiable capability. The build order is fixed — no phase can be executed out of sequence.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [ ] **Phase 1: Data Foundation** - EIA loader, normalizer, 31 state variable definitions, and sample-state fallback
- [ ] **Phase 2: NL Parser** - Claude API parser producing action vector and relevance mask from plain-English policy text
- [ ] **Phase 3: Env + UI Skeleton** - Gymnasium PolicyEnv, DemoTransitionModel linear fallback, and Streamlit app skeleton with demo disclosure
- [ ] **Phase 4: Reports + Differentiators** - Claude report generator, metric cards, trajectory charts, and tradeoff quadrant
- [ ] **Phase 5: GRU Training** - PyTorch training pipeline, chronological split, model checkpoint, and GRU swap-in to PolicyEnv

## Build Order Note

These phases must execute sequentially. Phase 2 (parser) requires `VAR_INDEX` and normalized variable definitions from Phase 1. Phase 3 (env) requires the action vector schema and parser from Phase 2. Phase 4 (reports) requires a working end-to-end rollout from Phase 3. Phase 5 (GRU) requires the trained-model hot-swap path and all evaluation infrastructure from Phases 3–4. Parallelizing across phases will produce integration failures.

## Phase Details

### Phase 1: Data Foundation
**Goal**: All 31 state variables are defined, loadable from EIA, normalizable, and available via a local sample-state fallback so every downstream component has a stable, typed data contract to build against.
**Depends on**: Nothing (first phase)
**Requirements**: DATA-01, DATA-02, DATA-03, DATA-04, DATA-05, DATA-06, ENV-01, ENV-02, ENV-03, ENV-04, ENV-05, ENV-06, ENV-07, ENV-08, ENV-09, ENV-10, ENV-11, ENV-12, ENV-13, ENV-14, ENV-15, ECON-01, ECON-02, ECON-03, ECON-04, ECON-05, ECON-06, ECON-07, ECON-08, ECON-09, ECON-10, ECON-11, ECON-12, ECON-13, ECON-14, ECON-15, ECON-16
**Success Criteria** (what must be TRUE):
  1. Running `python src/data.py` produces a valid 31-element float array (no NaN for any variable that has EIA data) and writes `data/sample_state.json`
  2. The same array is produced when EIA API key is absent, using `data/sample_state.json` as fallback — confirming offline demo viability
  3. `VAR_INDEX` dict is importable from a single shared module and maps all 31 variable names to indices 0–30 with no gaps or duplicates
  4. `data/normalizer_params.json` exists, was fit on the 1960–2005 chronological split only, and round-trips correctly (normalize → denormalize returns original values within floating-point tolerance)
  5. EIA pagination works correctly: datasets exceeding 5,000 rows are fully fetched and cached to CSV/parquet before hackathon demo
**Plans**: TBD

### Phase 2: NL Parser
**Goal**: A user can type any US energy policy in plain English and receive a structured 31-dim action vector with a binary relevance mask, confidence rating, and per-variable reasoning — all validated and safe to feed directly into PolicyEnv.
**Depends on**: Phase 1
**Requirements**: PARSE-01, PARSE-02, PARSE-03, PARSE-04, PARSE-05, PARSE-06
**Success Criteria** (what must be TRUE):
  1. Parsing "impose a $50/ton carbon tax" returns a structured dict with non-zero actions on CO2-related variables, `confidence: "high"`, and a reasoning string for each affected variable
  2. Parsing an out-of-domain input (e.g., "invade Canada") returns `confidence: "out_of_domain"` with all action magnitudes zeroed — no garbage values enter the environment
  3. All action magnitudes in the returned dict are clamped to `[0.0, 1.0]` by application code regardless of raw LLM output
  4. The parser produces a 31-dim bool mask that is True only for variables the policy directly affects, verifiable by inspection against 5 adversarial test inputs
**Plans**: TBD

### Phase 3: Env + UI Skeleton
**Goal**: A user visiting the Streamlit app can enter a policy, see it parsed, trigger a 10-step rollout using the linear DemoTransitionModel, and view the raw trajectory output — with clear banners showing demo mode is active and handling all missing API key edge cases gracefully.
**Depends on**: Phase 2
**Requirements**: ENV-RL-01, ENV-RL-02, ENV-RL-03, ENV-RL-04, ENV-RL-05, ENV-RL-06, ENV-RL-07, MODEL-01, MODEL-02, MODEL-03, UI-01, UI-02, UI-03, UI-07, UI-08, UI-09, UI-10
**Success Criteria** (what must be TRUE):
  1. `gymnasium.utils.env_checker.check_env(PolicyEnv())` passes with zero warnings or errors
  2. `env.rollout(action_vec, mask_vec, steps=10)` returns a trajectory array of shape `(11, 31)` with all values within EIA historical physical bounds
  3. The Streamlit app loads, accepts policy text, and displays a Policy Summary Card showing Claude's interpretation before any trajectory is shown
  4. A "Demo Mode" disclosure banner is always visible when `models/policy_transition.pt` is absent, and the app runs end-to-end without it
  5. Entering a policy with a missing `ANTHROPIC_API_KEY` shows a clear error message and falls back to demo mode without crashing
**Plans**: TBD
**UI hint**: yes

### Phase 4: Reports + Differentiators
**Goal**: After a rollout, the user sees a complete, polished impact report — a streaming Claude-generated plain-English narrative, four headline metric cards with before/after deltas, dual trajectory charts (environmental and economic), and a 2×2 tradeoff quadrant — all on one screen.
**Depends on**: Phase 3
**Requirements**: REPORT-01, REPORT-02, REPORT-03, REPORT-04, REPORT-05, MODEL-04, MODEL-05, UI-04, UI-05, UI-06
**Success Criteria** (what must be TRUE):
  1. The report streams live into the UI with two clearly labeled sections (Short-Term Effects 1–2 years, Long-Term Effects up to 10 years) each containing 2 paragraphs in jargon-free language
  2. The report cites at least one specific metric delta from the trajectory (e.g., "CO2 emissions projected to fall 12% within 2 years") — verifiable by inspection
  3. Four `st.metric()` cards display before/after deltas for CO2 total, renewable share, consumer electricity price, and coal employment with correct sign and units
  4. Two Plotly line charts show baseline vs. projected trajectory over 10 years — one for environmental metrics, one for economic metrics — rendered on the same screen as the report
  5. The 2×2 tradeoff quadrant correctly classifies the policy into Win-Win, Env-Win-Econ-Lose, Econ-Win-Env-Lose, or Lose-Lose based on the trajectory deltas
**Plans**: TBD
**UI hint**: yes

### Phase 5: GRU Training
**Goal**: A complete PyTorch training pipeline exists such that anyone following `src/data/README.md` can download EIA data, train the GRU transition model, and drop the checkpoint into `models/policy_transition.pt` to replace the linear fallback with learned dynamics — without any other code changes.
**Depends on**: Phase 4
**Requirements**: TRAIN-01, TRAIN-02, TRAIN-03, TRAIN-04, TRAIN-05
**Success Criteria** (what must be TRUE):
  1. Running `python src/train.py` completes without error on the prepared dataset, producing `models/policy_transition.pt` and a validation loss curve
  2. Placing `models/policy_transition.pt` in the models directory causes PolicyEnv to automatically load `PolicyTransitionModel` instead of `DemoTransitionModel` — the Demo Mode banner disappears
  3. The training split is strictly chronological (train 1960–2005, val 2006–2015, test 2016–2023) and the DataLoader never shuffles — verifiable by inspecting dataset indices
  4. `src/data/README.md` documents exact steps to download all required EIA data and produce training CSVs, sufficient for a judge to reproduce the dataset from scratch
**Plans**: TBD

## Progress

**Execution Order:**
Phases execute sequentially: 1 → 2 → 3 → 4 → 5 (dependency chain; no parallelization across phases)

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Data Foundation | 0/TBD | Not started | - |
| 2. NL Parser | 0/TBD | Not started | - |
| 3. Env + UI Skeleton | 0/TBD | Not started | - |
| 4. Reports + Differentiators | 0/TBD | Not started | - |
| 5. GRU Training | 0/TBD | Not started | - |
