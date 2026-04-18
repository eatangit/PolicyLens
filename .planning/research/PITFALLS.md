# Domain Pitfalls: PolicyLens

**Domain:** AI policy impact evaluator (LLM parsing + GRU world model + custom Gym env + EIA data)
**Researched:** 2026-04-18

---

## Critical Pitfalls

Mistakes that cause rewrites, demo failures, or invalidate results.

---

### Pitfall 1: Unconstrained Numeric Ranges in LLM Action Schema

**What goes wrong:** Claude's structured output feature guarantees type compliance but explicitly does NOT enforce `minimum`/`maximum` constraints — they are stripped from the schema sent to the model. The SDK validates them after the fact, meaning a `magnitude` field typed as `float` with range `[0.0, 1.0]` can silently produce `42.7` or `-0.3` if the LLM decides it fits the policy description. This is especially likely when a policy uses relative language ("dramatically reduce", "slightly increase") with no grounded baseline.

**Why it happens:** Structured outputs constrain structure, not semantics. The model has no domain knowledge about what a 0.0–1.0 action magnitude means physically. Policies written in absolute terms ("cut emissions by 40%") give the model a number to anchor on, but relative or vague policies give it nothing — so it hallucinates a plausible-sounding float.

**Consequences:** The Gym environment receives out-of-range actions. If the action space is `Box([0,0,...], [1,1,...])`, numpy will either clip silently or raise an assertion in `check_env`. Either way the simulation is wrong and the error may not surface visibly in the UI.

**Prevention:**
- Add an explicit `reasoning` field before `magnitude` in the schema (field ordering matters — earlier fields inform later ones). The model should first explain what the policy does in plain language, then commit to a magnitude.
- After receiving the response, run post-parse validation: clamp or reject any field outside `[0.0, 1.0]` and log a warning to the UI.
- Provide 3–5 calibration examples in the system prompt mapping policy language to magnitude: "A complete phase-out over 5 years ≈ 0.9, a 10% reduction target ≈ 0.2".
- Always check `stop_reason` before parsing; if `"refusal"` or `"max_tokens"`, the JSON will not match your schema.

**Warning signs:** Magnitudes consistently above 0.9 for mild policies; negative magnitudes; action vectors that sum to >5 across 31 variables for a narrow single-sector policy.

**Phase:** LLM parsing / Phase 1 (action schema design). The calibration examples should be part of the initial prompt engineering pass, not retrofitted.

---

### Pitfall 2: Vague or Out-of-Domain Policy Text Produces Confident Garbage

**What goes wrong:** When a policy is genuinely ambiguous ("promote clean energy") or entirely out-of-domain ("reform the tax code"), the LLM will not spontaneously return zeros or a null action. It will produce a confident JSON object with plausible-looking values for all 31 variables, even though the policy gives no information to ground them. Schema enforcement makes this worse: the model is forced to produce values, so it does.

**Why it happens:** LLMs are trained to be helpful and to fill in requested fields. Without an explicit "I don't know" escape hatch in the schema, the model will hallucinate rather than refuse. This is a documented failure of structured outputs: "Refusals override schema compliance" only for safety refusals; semantic uncertainty does not trigger refusal.

**Consequences:** The simulator receives a fabricated action vector. Results look plausible in the UI but are meaningless. At a hackathon this will fool judges unless someone asks about a clearly nonsensical policy.

**Prevention:**
- Add a `confidence` enum field (`"high" | "medium" | "low" | "out_of_domain"`) and a `coverage_notes` string to the schema.
- In the system prompt, explicitly instruct: "If the policy does not contain enough information to quantify an action, set `confidence` to `out_of_domain` and set all magnitudes to 0.0."
- In the UI, display a yellow warning banner when `confidence` is `"low"` or `"out_of_domain"`, rather than running the simulation silently.
- Test with at least five adversarial inputs during development (haiku, sports policy, medical policy, one-word inputs, inputs in another language).

**Warning signs:** All 31 variables have nonzero magnitudes for a policy that mentions only one sector; `coverage_notes` is empty despite vague input text.

**Phase:** LLM parsing / Phase 1. Build the escape hatch into the schema from day one, not as an afterthought.

---

### Pitfall 3: GRU Rollout Compounding Error (Autoregressive Drift)

**What goes wrong:** The GRU is trained to predict one step ahead given a history window. At inference time the Gym env feeds the GRU its own previous predictions as input (autoregressive rollout). Each step introduces a small prediction error. These errors compound multiplicatively. After 10–20 steps, the state vector drifts into physically impossible territory (negative energy consumption, renewable share > 100%) even though each individual step looked plausible.

**Why it happens:** The model was only trained on ground-truth sequences, never on its own predicted sequences. The error distribution of its inputs at inference time is different from training. This is the standard "distribution shift" problem for autoregressive world models, and it is especially acute with only ~60 training samples (years).

**Consequences:** The simulator produces nonsensical long-horizon forecasts. Short-horizon demos (2–5 year projections) may look fine; 10+ year projections will fail visibly. The EIA state variables include physically constrained quantities that will violate bounds.

**Prevention:**
- Cap default simulation horizon at 5 steps (years) in the demo UI. Warn users at 10+ steps.
- After each GRU step, clamp all state variables to their physical bounds derived from EIA historical data (e.g., renewable share ∈ [0%, 100%], total consumption cannot be negative).
- Consider scheduled sampling during training: mix ground-truth and predicted states as input with increasing probability of using predictions as training progresses.
- Log and display per-step prediction uncertainty (if the GRU outputs a mean + variance, propagate variance forward).
- In demo mode (no trained model), use a linear interpolation stub that cannot drift — this is the safer demo fallback.

**Warning signs:** State variables that move monotonically in one direction without plateauing; any variable hitting its declared `observation_space` bound before step 10; `nan` or `inf` appearing in the state.

**Phase:** GRU training + Gym env design. Clamping logic must be in the `step()` function, not just in the GRU. Both phases need to coordinate on physical bounds.

---

### Pitfall 4: Temporal Leakage in Train/Val Split

**What goes wrong:** With ~60 years of annual EIA data, a random train/test split means some training samples will be temporally *after* some test samples. The GRU learns future information, making validation loss artificially low and the model overconfident in simulation. At demo time the model generalizes poorly to recent years it nominally "hasn't seen" but actually has.

**Why it happens:** This is the most common mistake when practitioners apply standard ML train/test split logic to time series. Sklearn's `train_test_split` with `shuffle=True` (the default) causes this. With only 60 samples, even a single misplaced year can cause measurable leakage.

**Consequences:** Validation metrics look better than reality. The model is effectively overfitting to the full data range while reporting inflated generalization numbers.

**Prevention:**
- Split strictly chronologically: train on years 1960–2005, validate on 2006–2015, test on 2016–2023. Never shuffle.
- Normalize (z-score or min-max) using *only* training set statistics. Compute mean/std from the training split, apply those same parameters to val/test. Do not fit a scaler on the full dataset before splitting.
- With only ~60 samples, avoid `TimeSeriesSplit` with many folds — the training windows become too short. A single chronological holdout is safer.

**Warning signs:** Validation loss close to training loss on the first epoch (suspect data leakage); scaler fit on the full dataset before splitting; any use of `shuffle=True` in the data loader.

**Phase:** Data pipeline / GRU training. This must be established in the data loading code before any model training begins.

---

### Pitfall 5: Observation Space / Action Space Shape Mismatch (Silent Bug)

**What goes wrong:** Gymnasium's `check_env` utility catches some mismatches, but broadcast-compatible shapes pass silently. The most common failure: the Gym observation space is declared as `Box(shape=(31,))` but the GRU returns shape `(1, 31)` (batch dimension included). NumPy broadcasting means this doesn't raise an error — it just passes a 2D array where a 1D array is expected. The agent (or the Gym step loop) then gets wrong shapes downstream.

**Why it happens:** PyTorch and TensorFlow GRU/LSTM modules return outputs with a batch dimension even when batch size is 1. The PyTorch `GRU` module returns `(seq_len, batch, hidden_size)` by default. Forgetting `.squeeze(0)` or `.squeeze(1)` is the single most common shape bug.

**Consequences:** Silent wrong behavior. Observations look correct in shape assertions but carry extra dimensions. Policy or analysis code indexing `obs[i]` gets a row vector instead of a scalar, causing wrong computations that are numerically plausible.

**Prevention:**
- Run `gymnasium.utils.env_checker.check_env(env)` on every environment instantiation during development. It will catch the most egregious shape mismatches.
- Assert `obs.shape == env.observation_space.shape` inside `step()` and `reset()` (can be gated behind a `DEBUG` flag and disabled for production).
- When extracting the final hidden state from the GRU, always explicitly call `.squeeze()` and log the shape during the first few inference steps.
- Add a `__post_init__` or `__init__` check that calls `reset()` and asserts the returned shape matches `observation_space`.

**Warning signs:** `check_env` completes but simulation outputs are identical across all steps; state variables that never change regardless of action magnitude; numpy deprecation warnings about ragged arrays.

**Phase:** Gym environment / GRU integration. The shape contract between GRU and Gym must be established as a unit test before the simulation loop is written.

---

### Pitfall 6: Reward Shaping Overriding True Objective

**What goes wrong:** If the Gym env is used for display/simulation only (not RL training), this is lower risk — but if reward is used to score policy outcomes in the UI, a poorly designed reward function causes the simulator to declare a harmful policy "good." For example: a reward that penalizes total energy consumption will rate any demand-destruction policy (economic collapse, rationing) as excellent.

**Why it happens:** Reward shaping introduces implicit value judgments. In energy simulation, the variables are highly correlated — reducing CO2 almost always requires reducing fossil fuel consumption, which correlates with reduced total economic activity. A simple linear reward over EIA variables will have unintended cross-variable incentives.

**Consequences:** The UI shows high scores for policies that real economists would flag as harmful. Judges who probe edge cases will find this quickly.

**Prevention:**
- For a hackathon demo, avoid a scalar "policy score" reward. Instead, show a multi-dimensional outcome radar chart (CO2, renewables share, economic proxy, energy security) and let the user interpret tradeoffs.
- If a single score is required, weight it by domain-expert values from prior literature (e.g., the IEA's sustainability index) rather than tuning it to look good on test cases.
- At minimum, add a sanity constraint: any policy that simultaneously reduces all 31 state variables by more than 50% should trigger a warning flag.

**Warning signs:** A "zero emissions, zero energy" policy scores perfectly; any single-variable reward function.

**Phase:** Gym env design / UI design. Decide before building whether reward is for display or for RL training — the answer changes the architecture.

---

## Moderate Pitfalls

---

### Pitfall 7: EIA API 5,000-Row Pagination Limit Silently Truncates Data

**What goes wrong:** The EIA v2 API returns a maximum of 5,000 rows per request. For 31 state variables × 50 states × 60 years of annual data, a naive single request will be silently truncated. The API returns a warning header (`X-Warning`) when truncation occurs, but if you parse only the JSON body you will never see it.

**Prevention:**
- Always check the `X-Warning` response header after any EIA fetch.
- Use `offset` and `length` parameters to paginate. For bulk pulls, fetch in chunks of 2,000 rows.
- Better: download all needed series once at hackathon start, cache as parquet or CSV, and load from disk. This avoids rate limits and network flakiness during demo.
- Budget for 30–60 minutes of data collection time before the hackathon ends.

**Warning signs:** Data fetch for 60 years returns exactly 5,000 rows; state-variable time series ends suspiciously early (e.g., at 1990 instead of 2023).

**Phase:** Data pipeline (earliest phase). Fetch and cache data first, before writing any model code.

---

### Pitfall 8: EIA v2 Returns All Numeric Values as Strings (Since January 2024)

**What goes wrong:** As of API v2.1.6 (January 2024), the EIA API standardizes all data point values as JSON strings, not numbers, to handle values requiring leading zeroes. Code that calls `float(row["value"])` will work — but code that treats `row["value"]` as a number directly (e.g., numpy array construction from a list of raw API dicts) will produce string arrays that silently pass construction and fail only at arithmetic.

**Prevention:**
- Centralize all EIA parsing in one function that always calls `float(row.get("value", "nan"))` with a `nan` fallback for missing values.
- After converting, run `np.isfinite(array).all()` and log how many `nan`s were produced — EIA redacts individual plant-level values that propagate as gaps in state-level aggregates (23% of monthly records remain unfilled per PUDL analysis).
- For annual state aggregates the gap rate is lower, but still check.

**Warning signs:** NumPy dtype showing `object` or `<U10` (unicode string) instead of `float64` after array construction; arithmetic operations raising `TypeError` in unexpected places.

**Phase:** Data pipeline. Wrap in a parsing utility at the start and test it against at least three different EIA series endpoints (they have different column naming conventions).

---

### Pitfall 9: Annual vs. Monthly Frequency Mismatch Across EIA Series

**What goes wrong:** Not all EIA state-level series are available at the same frequency. Some generation mix variables are monthly; some capacity and consumption variables are annual only. Mixing them naively produces a dataframe where monthly-frequency rows outnumber annual-frequency rows 12:1, and simple forward-fill imputation inflates the apparent data size while introducing stale values.

**Prevention:**
- Audit each of the 31 target variables for native frequency before the hackathon. Use the EIA API browser to check `frequency` metadata per series.
- Decide on one canonical frequency (annual is safest for 60-year GRU training) and aggregate monthly data to annual using mean for rates and sum for volumes before storing.
- Use the `frequency=annual` parameter in API requests where supported rather than fetching monthly and aggregating yourself.

**Warning signs:** Dataframe has 12× more rows than expected; date column contains month-level dates mixed with year-level dates; any variable showing suspiciously flat 12-month plateaus.

**Phase:** Data pipeline. This decision must be made and documented before GRU sequence construction.

---

### Pitfall 10: Secrets Not Available at Demo Time (Streamlit + Claude API Key)

**What goes wrong:** Development uses a local `.streamlit/secrets.toml` or a `.env` file that is gitignored. At hackathon demo time, running on a different machine or a fresh clone, `st.secrets["ANTHROPIC_API_KEY"]` raises `KeyError` with no friendly error. The app crashes before the demo starts.

**Prevention:**
- Never access `st.secrets["KEY"]` directly at module load time. Access it inside functions, so the app can start in demo mode even without the key.
- Implement a startup check:
  ```python
  ANTHROPIC_KEY = st.secrets.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
  DEMO_MODE = ANTHROPIC_KEY is None
  if DEMO_MODE:
      st.sidebar.warning("Running in demo mode — LLM parsing disabled.")
  ```
- Include a `secrets.toml.example` file in the repo with placeholder values and a README note.
- Test on a fresh clone on a separate machine at least 30 minutes before the demo deadline.

**Warning signs:** App works on dev machine, fails immediately on first launch elsewhere; any `import` at the top of `app.py` that calls `st.secrets`.

**Phase:** UI / deployment. Build the demo-mode fallback during initial UI scaffolding, not the night before the presentation.

---

### Pitfall 11: LLM Structured Output Field Ordering Degrades Reasoning Quality

**What goes wrong:** Claude's structured output enforces `required` fields first, then optional fields, regardless of schema field order. If `magnitude` is `required` and `reasoning` is optional, the model may emit `magnitude` before it has reasoned about the policy — producing values that aren't grounded in the chain-of-thought. This is the inverse of the intended "think, then output" pattern.

**Prevention:**
- Make both `reasoning` and `magnitude` required. Put `reasoning` first in the schema definition. The model will then reason through the policy before committing to a number.
- Alternatively use a two-pass approach: first call with no schema to get free-form analysis, second call to extract structured fields from the analysis text. This is slower but produces better grounded magnitudes.
- Keep `max_tokens` generous (1,500+) — token cutoff mid-reasoning will produce a refusal or truncated JSON, both of which must be caught.

**Warning signs:** Identical `reasoning` text across different policy inputs; `reasoning` field that contradicts the `magnitude` value; very short `reasoning` strings (< 20 words) despite complex policy text.

**Phase:** LLM parsing / prompt engineering. Establish and test the schema field order in the first prototype.

---

## Minor Pitfalls

---

### Pitfall 12: Gym `reset()` Not Returning `(obs, info)` Tuple

**What goes wrong:** Gymnasium (Farama) updated the `step()` and `reset()` API: `reset()` must return `(obs, info)` and `step()` must return `(obs, reward, terminated, truncated, info)`. Old Gym-style code returning `(obs, reward, done, info)` from `step()` will produce subtle value-assignment bugs when unpacked with the new 5-tuple convention.

**Prevention:** Use `gymnasium` (not `gym`) from the start. Run `check_env(env)` which validates the return signatures. Keep Gymnasium version pinned in `requirements.txt`.

**Warning signs:** `terminated` or `truncated` always `False`; `done` variable used in the loop that never triggers episode end.

**Phase:** Gym env scaffolding. Fix at environment creation, not during integration.

---

### Pitfall 13: GRU Hidden State Not Reset Between Episodes

**What goes wrong:** The GRU carries a hidden state `h` across time steps. If `reset()` initializes a new episode but does not reinitialize `h` to zeros, the GRU's internal state from the previous episode leaks into the new one. This causes the first few steps of every episode after the first to be wrong.

**Prevention:** Store `self.h` on the environment object. In `reset()`, set `self.h = torch.zeros(...)`. In `step()`, pass `self.h` into the GRU call and update it from the returned hidden state.

**Warning signs:** First episode output differs from all subsequent episodes run from the same initial state; episode-2 results are sensitive to episode-1 action sequence.

**Phase:** GRU integration / Gym env. Add a unit test: run two episodes from the same seed and assert identical outputs.

---

### Pitfall 14: Normalizing with Test-Set Statistics (Data Leakage via Scaler)

**What goes wrong:** Fitting `StandardScaler` or `MinMaxScaler` on the full dataset before splitting introduces a subtle form of leakage: the scaler's mean and variance are informed by future data, making the normalized training set slightly "aware" of the test distribution.

**Prevention:** Fit scaler on training split only. Store scaler parameters (mean, std or min, max) as artifacts alongside the model checkpoint. At inference time, load and apply the same scaler before feeding data to the GRU.

**Warning signs:** Scaler fitted with `X_all` before `train_test_split`; no scaler checkpoint saved alongside the model.

**Phase:** Data pipeline / GRU training.

---

### Pitfall 15: Streamlit Reruns Triggering Repeated API Calls

**What goes wrong:** Streamlit reruns the entire script on every widget interaction. If LLM parsing or EIA fetching is called at the top level (not wrapped in a button callback or cached), every slider move or text input keystroke triggers a new Claude API call and burns tokens rapidly.

**Prevention:**
- Wrap all LLM and API calls in `if st.button("Analyze policy"):` blocks.
- Use `@st.cache_data` for EIA data loading with an appropriate TTL.
- Use `st.session_state` to store parsed action vectors so they persist across reruns without recomputation.

**Warning signs:** Claude API cost spike after 10 minutes of demo interaction; EIA API rate limit hit during a demo session; app visibly slow on every widget interaction.

**Phase:** UI / Streamlit integration.

---

## Phase-Specific Warnings

| Phase / Topic | Likely Pitfall | Mitigation |
|---|---|---|
| Action schema design | Unconstrained numeric magnitudes; no out-of-domain escape | Add `reasoning` first, post-parse clamp, calibration examples in system prompt |
| Data pipeline | EIA pagination truncation; string-typed values; frequency mismatch | Cache locally, centralize parser, audit frequency per series before pipeline |
| GRU training | Temporal leakage via random split or scaler on full data | Chronological split, scaler fitted on train only, scaler saved as artifact |
| GRU inference | Autoregressive drift; hidden state not reset; shape mismatch | Cap horizon at 5 steps, clamp outputs to physical bounds, reset h in reset() |
| Gym environment | Return signature mismatch; broadcast shape bugs | Run check_env, assert shapes in step/reset, use gymnasium not gym |
| UI / Streamlit | API calls on every rerender; secrets missing at demo time | Use st.cache_data, session_state, demo-mode fallback for missing key |
| LLM field ordering | Magnitude emitted before reasoning | Put reasoning field first and mark required |
| Demo environment | Works locally, fails on fresh clone | Test on second machine 30 min before deadline |

---

## Sources

- Claude structured outputs official docs and limitations: https://platform.claude.com/docs/en/build-with-claude/structured-outputs
- Claude hallucination reduction guidance: https://platform.claude.com/docs/en/test-and-evaluate/strengthen-guardrails/reduce-hallucinations
- Structured outputs create false confidence (HN discussion): https://news.ycombinator.com/item?id=46345333
- Stable Baselines3 RL tips and tricks (observation normalization): https://stable-baselines.readthedocs.io/en/master/guide/rl_tips.html
- Gymnasium custom environment guide: https://stable-baselines3.readthedocs.io/en/master/guide/custom_env.html
- Reward hacking in RL (Lilian Weng, 2024): https://lilianweng.github.io/posts/2024-11-28-reward-hacking/
- Time series train/test split and temporal leakage: https://towardsdatascience.com/avoiding-data-leakage-in-timeseries-101-25ea13fcb15f/
- World models compounding error survey: https://arxiv.org/html/2411.14499v3
- EIA API v2 official documentation: https://www.eia.gov/opendata/documentation.php
- EIA API pagination and string-typed values (v2.1.6): https://www.eia.gov/opendata/documentation/APIv2.1.0.pdf
- PUDL analysis of EIA data gaps (23% unfilled monthly records): https://catalystcoop-pudl.readthedocs.io/en/latest/data_sources/eiaapi.html
- Streamlit secrets management: https://docs.streamlit.io/develop/concepts/connections/secrets-management
- LLM structured output benchmark mistakes: https://cleanlab.ai/blog/structured-output-benchmark/
- ML hackathon failure modes: https://medium.com/data-engineering/modeling-madly-8b2c72eb52be
