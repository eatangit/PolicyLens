# Architecture Patterns — PolicyLens

**Domain:** AI policy evaluation / GRU world model / Gymnasium simulation
**Researched:** 2026-04-18
**Overall confidence:** HIGH (PyTorch/Gymnasium APIs verified), MEDIUM (sizing heuristics — small-dataset regime is under-studied in literature)

---

## Recommended Architecture

### System Overview

```
[User text input]
       |
       v
[Claude API — NL Parser]
  Prompt: "You are a policy analyst..."
  Output: JSON { actions: {var_id: float[-1,1]}, mask: {var_id: bool} }
       |
       v
[Action Vector Builder]
  Shape: (31,) float32   — action magnitudes, 0.0 for masked variables
  Shape: (31,) bool      — relevance mask, False = variable not affected
       |
       v
[PolicyEnv — Gymnasium Environment]
  observation_space: Box(-inf, inf, shape=(31,), dtype=float32)  [normalized]
  action_space:      Box(-1.0, 1.0, shape=(31,), dtype=float32)
  state: current normalized 31-vector
  step() applies action once at t=0, then runs GRU or fallback for t=1..10
       |
       v
[GRU Transition Model  |  Linear Fallback (demo mode)]
  Input at each step: concat(state_t, action_effective_t)  shape=(62,)
  Output: delta_state_t+1                                  shape=(31,)
  next_state = state_t + delta_state_t+1
       |
       v
[Trajectory Buffer]
  List of 11 state vectors: [s0, s1, ..., s10]  (t=0 baseline + 10 future)
       |
       v
[Claude API — Report Generator]
  Input: trajectory (denormalized), variable names, policy description
  Output: plain-English report, two sections: short-term (yr 1-2), long-term (yr 3-10)
       |
       v
[Streamlit UI + Plotly Charts]
  Sidebar: policy text input
  Main: formatted report + multi-line chart (baseline vs. policy per key variable)
```

---

## Component Boundaries

| Component | File(s) | Responsibility | Communicates With |
|-----------|---------|---------------|-------------------|
| NL Parser | `src/parser.py` | Claude API call → structured action dict | Action Vector Builder |
| Action Vector Builder | `src/action.py` | Dict → (31,) float tensor + (31,) bool mask | PolicyEnv |
| PolicyEnv | `src/env.py` | Gymnasium interface; holds state; runs rollout | GRU model or linear fallback |
| GRU Transition Model | `src/model.py` | `nn.GRU` + linear head; predicts state delta | PolicyEnv |
| Linear Fallback | `src/model.py` | `DemoTransitionModel`; pure `state + action * scale` | PolicyEnv |
| Normalizer | `src/normalizer.py` | Per-variable RobustScaler fit on training data; serialize as JSON | PolicyEnv, Trainer |
| Dataset / Trainer | `src/train.py` | PyTorch Dataset over EIA sequences; training loop | GRU model |
| EIA Loader | `src/data.py` | HTTP → raw state dict; fallback to `sample_state.json` | Trainer, UI bootstrap |
| Report Generator | `src/report.py` | Trajectory + metadata → Claude API → markdown string | Streamlit UI |
| Streamlit App | `app.py` | User-facing orchestration; calls parser → env → report | All components |

---

## Data Flow (Detailed)

### Inference Path (Demo Day)

```
1. User enters: "Implement a $50/ton carbon tax starting next year"

2. parser.py → Claude:
   System: structured JSON schema with 31 variable IDs
   Response: { "actions": {"coal_production": -0.6, "renewables_gen": 0.4, ...},
               "mask":    {"coal_production": true,  "renewables_gen": true, "nuclear_gen": false, ...} }

3. action.py:
   action_vec = np.zeros(31, dtype=np.float32)
   mask_vec   = np.zeros(31, dtype=bool)
   for var_id, val in actions.items():
       action_vec[VAR_INDEX[var_id]] = val
       mask_vec[VAR_INDEX[var_id]]   = mask[var_id]

4. env.py — PolicyEnv.reset(baseline_state):
   self.state = normalizer.transform(baseline_state)   # shape (31,)
   self.t = 0

5. env.py — PolicyEnv.rollout(action_vec, mask_vec, steps=10):
   trajectory = [self.state.copy()]
   # Apply mask: zero out action for irrelevant variables
   effective_action = action_vec * mask_vec.astype(float)
   # Action applied AT t=0 ONLY (see Action Application section)
   for t in range(steps):
       inp = np.concatenate([self.state, effective_action])    # (62,)
       delta = model.step(inp)                                 # (31,)
       self.state = self.state + delta
       trajectory.append(self.state.copy())
       # Decay action influence for t > 0
       effective_action = effective_action * DECAY_FACTOR      # default: 0.85/yr

   return [normalizer.inverse_transform(s) for s in trajectory]  # 11 × (31,)

6. report.py → Claude:
   Formats trajectory as structured context (year-by-year delta table)
   Returns two-section markdown report

7. app.py:
   Renders report + Plotly chart of selected variables
```

---

## GRU Transition Model

### Architecture Decision

Use a **residual GRU with delta prediction** — the model predicts the *change* in state (`delta_s`), not the next state directly. This is critical for a small dataset (~60 annual samples → ~50 training windows of length 5-10) because:

- Absolute state prediction requires learning the full scale of each variable, compounding error
- Delta prediction lets the model learn only "how does policy perturbation shift the trend?" which is a much easier supervised signal
- Residual connections (`next_state = state + delta`) prevent runaway drift in rollouts

### Recommended Layer Sizes

```python
class GRUTransitionModel(nn.Module):
    def __init__(
        self,
        state_dim: int = 31,
        action_dim: int = 31,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        input_size = state_dim + action_dim  # 62

        self.gru = nn.GRU(
            input_size=input_size,       # 62
            hidden_size=hidden_size,     # 64
            num_layers=num_layers,       # 2
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, state_dim),   # output: delta per variable
        )
```

**Rationale for hidden_size=64, num_layers=2:**

- Input is 62 features (state + action). Hidden size 2x input (64 ≈ 62) follows common heuristic for moderate-complexity sequence data.
- At ~60 annual samples, with a sliding window of 5 years and 80/20 split, the effective training set is ~40-45 sequences. Parameters: GRU(62→64, 2 layers) ≈ 2 × (3 × 64 × (62+64+1)) ≈ ~47k parameters. This is deliberately lean to avoid overfitting at this sample count.
- Two layers allow the first to learn low-level temporal patterns (momentum, trend), the second to learn higher-order interactions (e.g., how energy prices correlate with production shifts over multiple years).
- **Do not use 3+ layers or hidden_size > 128.** With 60 samples, deeper networks overfit dramatically. The GRU-D literature (MIMIC-III, ~40k patients) used 64-100 units for single-layer models — scale down here.
- Dropout=0.2 on the inter-layer connection (PyTorch applies this between GRU layers automatically, not on the last layer output). Add weight_decay=1e-4 to the Adam optimizer as a secondary regularizer.

### Input Encoding — State + Action Concatenation

```
input_t = concat(normalized_state_t, effective_action_t)   # shape (62,)
```

This is the standard approach for action-conditioned world models (verified against Dreamer-v2, MBPO literature patterns). The alternative — passing action through a separate embedding network — is unnecessary overhead for a 31-dim continuous action.

**Why not film-conditioning or cross-attention?** Those are warranted when action space is discrete (token-like) or very high-dimensional. At 31 floats, direct concatenation is sufficient and trains faster on small data.

### Variable Masking Strategy

Policy-irrelevant variables (mask=False) receive `effective_action[i] = 0.0`. This is the correct approach — do not zero out the state variable itself, only suppress its action influence. The GRU still observes all 31 state variables at every step (necessary for cross-variable dynamics), but the action signal for irrelevant variables is identically zero, which the model learns to treat as "no intervention."

**At training time:** always pass the full 31-dim action vector (with zeros for unaffected variables). This teaches the model the zero-action distribution naturally. No special masking layers are needed at training time.

**Do not use GRU-D-style learned decay gates for missingness** — that paper addresses observation missingness over time, not policy relevance masking. The problems are structurally different.

---

## Action Application Strategy

**Recommended: t=0 application with exponential decay**

```python
DECAY_FACTOR = 0.85   # per year; configurable per policy type

effective_action_t = action_vec * mask_vec * (DECAY_FACTOR ** t)
```

**Rationale:**

- **t=0 only (no decay):** Unrealistic. A carbon tax doesn't disappear in year 2 — it continues to affect incentives. This produces a "shock and recover" trajectory that misrepresents persistent policy effects.
- **Constant application (same action every step):** Overstates policy impact in later years. Economic systems adapt; firms find new equilibria; the marginal effect of a policy diminishes as the economy adjusts.
- **Exponential decay (recommended):** Models the well-documented economic phenomenon of policy absorption. At decay=0.85: year 1 = 85% of original signal, year 5 = 44%, year 10 = 20%. This matches empirical patterns in energy economic studies where regulatory changes have front-loaded effects.

DECAY_FACTOR=0.85 is a reasonable prior; expose it as a config parameter so it can be tuned or set differently per policy category (e.g., structural mandates like a renewable portfolio standard might use 0.95; one-time subsidies might use 0.70).

---

## Gymnasium Environment Design

PolicyEnv does not need a reward signal — it is a simulation environment, not a training environment. Structure it as follows:

```python
import gymnasium as gym
import numpy as np

class PolicyEnv(gym.Env):
    """
    World-model simulation environment.
    NOT used for agent training — used for deterministic policy rollout only.
    """
    metadata = {"render_modes": []}

    def __init__(self, transition_model, normalizer, baseline_state: np.ndarray):
        super().__init__()
        self.dim = 31

        # Gymnasium requires these even in eval-only environments
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.dim,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(self.dim,), dtype=np.float32
        )

        self.model = transition_model
        self.normalizer = normalizer
        self.baseline = normalizer.transform(baseline_state)
        self.state = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.state = self.baseline.copy()
        self.t = 0
        return self.state.copy(), {}

    def step(self, action: np.ndarray):
        """
        Advances one time step using the transition model.
        action: (31,) effective action vector (already masked and decayed by caller)
        """
        inp = np.concatenate([self.state, action]).astype(np.float32)
        delta = self.model.predict_delta(inp)
        self.state = self.state + delta
        self.t += 1
        terminated = self.t >= 10
        truncated = False
        reward = 0.0   # no RL agent; reward unused
        return self.state.copy(), reward, terminated, truncated, {"t": self.t}

    def rollout(self, action_vec, mask_vec, steps=10, decay=0.85):
        """
        Convenience method: full 10-step trajectory from current state.
        Returns list of 11 denormalized state arrays [s0, s1, ..., s10].
        """
        obs, _ = self.reset()
        trajectory = [self.normalizer.inverse_transform(obs)]
        for t in range(steps):
            effective = action_vec * mask_vec * (decay ** t)
            obs, _, terminated, _, _ = self.step(effective)
            trajectory.append(self.normalizer.inverse_transform(obs))
            if terminated:
                break
        return trajectory
```

**Key design notes:**
- `observation_space` and `action_space` are defined as required by Gymnasium's API contract, but are used only for type/shape documentation — no agent queries them at runtime.
- The `rollout()` convenience method on the env is more idiomatic for this use case than calling `step()` externally in a loop.
- Use `gymnasium` (Farama Foundation), not the legacy `gym` (OpenAI). Import path: `import gymnasium as gym`.

---

## Normalization Strategy

**Recommended: per-variable RobustScaler (median + IQR)**

```python
from sklearn.preprocessing import RobustScaler

# One scaler per variable — fit on training data (historical EIA sequences)
# Store as JSON: {var_id: {center: float, scale: float}}
class VariableNormalizer:
    def __init__(self):
        self.scalers: dict[str, RobustScaler] = {}

    def fit(self, data: np.ndarray, var_names: list[str]):
        # data: (T, 31) historical values
        for i, name in enumerate(var_names):
            s = RobustScaler()
            s.fit(data[:, i].reshape(-1, 1))
            self.scalers[name] = s

    def transform(self, state: np.ndarray) -> np.ndarray:
        return np.array([
            self.scalers[n].transform([[v]])[0][0]
            for n, v in zip(self.scalers.keys(), state)
        ], dtype=np.float32)

    def inverse_transform(self, state: np.ndarray) -> np.ndarray:
        return np.array([
            self.scalers[n].inverse_transform([[v]])[0][0]
            for n, v in zip(self.scalers.keys(), state)
        ], dtype=np.float32)
```

**Why RobustScaler over z-score or min-max:**

- EIA data contains heterogeneous variables: coal production in short tons (1e8 scale), carbon intensity in lbs/MMBtu (small scale), spot prices in $/MMBtu (medium scale with high volatility and price spikes).
- Z-score (StandardScaler) is corrupted by energy price spikes (e.g., natural gas 2022 spike, coal export surge) — these inflate sigma and compress normal variation.
- Min-max is sensitive to the same outliers and produces very compressed representations for well-behaved variables.
- RobustScaler uses median and IQR, which are resistant to extreme events. This is the correct choice for energy and economic data with known outlier years (2008 financial crisis, 2020 COVID collapse, 2022 energy shock).
- **Fit only on training data** (pre-2015 or leave-last-10-years-out). Never fit on test years. Serialize scaler parameters to JSON for reproducibility — the demo mode must be able to normalize the live EIA baseline without re-fitting.

**For the linear fallback (demo mode):** use the same normalizer with hardcoded scaler parameters (median/IQR from known 2000-2020 data). Store in `data/normalizer_params.json`.

---

## Linear Fallback (Demo Mode)

```python
class DemoTransitionModel:
    """
    Fallback when GRU model is not trained.
    Predicts state delta as a scaled linear function of action.
    No learned weights — uses hand-tuned scale factors per variable category.
    """
    ENV_SCALE = 0.04   # environmental variables: smaller per-step effect
    ECON_SCALE = 0.02  # economic variables: even more inertial

    def predict_delta(self, inp: np.ndarray) -> np.ndarray:
        # inp: (62,) = [state(31), action(31)]
        state = inp[:31]
        action = inp[31:]
        delta = np.zeros(31, dtype=np.float32)
        delta[:15] = action[:15] * self.ENV_SCALE   # environmental
        delta[15:]  = action[15:]  * self.ECON_SCALE  # economic
        return delta
```

The linear fallback produces plausible directional trajectories without any training. Environmental variables react faster than economic variables (ENV_SCALE > ECON_SCALE) matching real-world dynamics. This is explicitly hackathon scaffolding — label it clearly in the UI.

---

## Training Pipeline Design

### Dataset Construction

```
EIA data: 60 annual observations (approx 1960-2023) across 31 variables
Sliding window: width=5 years, stride=1 → ~55 training windows
Each window: X = state sequence [t, t+1, ..., t+4], Y = next state [t+5]
Or: X = (state_t, action=zeros), Y = delta_t+1 (zero-action baseline dynamics)
```

**Critical:** Training on zero-action sequences teaches the model the baseline dynamics. At inference, non-zero actions perturb those dynamics. This is the correct world-model training paradigm.

### Training Loop Outline

```python
# Batch: (B, seq_len, 62) → target: (B, seq_len, 31) deltas
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20)
criterion = nn.MSELoss()

# Early stopping: stop if val loss doesn't improve for 30 epochs
# Expected: ~200-500 epochs to converge on 60-sample dataset
```

**Hyperparameters to expose in config:**
- `hidden_size`: 64 (start here; try 32 and 96 if overfitting or underfitting)
- `num_layers`: 2 (reduce to 1 if validation loss diverges from training loss)
- `dropout`: 0.2 (increase to 0.3 if still overfitting)
- `seq_len` (window width): 5 (enough temporal context without eating too many training samples)
- `decay_factor`: 0.85

---

## Build Order Implications

The component dependency graph forces this build order:

```
1. data.py         — EIA loader + sample_state.json fallback
                     (needed by all downstream components; unblock this first)

2. normalizer.py   — fit on loaded data; serialize params
                     (PolicyEnv and model both require normalized inputs)

3. env.py          — Gymnasium environment with DemoTransitionModel
                     (enables full pipeline testing before GRU is trained)

4. parser.py       — Claude NL → action vector
                     (can be built in parallel with env.py)

5. report.py       — Claude trajectory → report
                     (requires trajectory format from env.py rollout)

6. app.py          — Streamlit UI wiring all components
                     (requires parser + env + report)

7. model.py        — GRU model (full implementation)
                     (train.py depends on this; env.py already has demo fallback)

8. train.py        — Training pipeline
                     (last — demo works without it; judges can verify offline)
```

This order maximizes demo viability: after step 6, the full end-to-end demo runs (with linear fallback). Steps 7-8 are a clean enhancement that can be demonstrated with pre-trained weights if time allows.

---

## Scalability Considerations

| Concern | Hackathon demo | Post-hackathon |
|---------|---------------|----------------|
| Training data | 60 annual EIA samples | Add state-level EIA data or monthly granularity → 700+ samples |
| Model capacity | hidden=64, layers=2 | Scale to hidden=128, layers=3 with more data |
| Rollout latency | ~5ms per step (CPU) | Acceptable; not a bottleneck |
| Claude API latency | ~3-5s per call (parser + report) | Use async calls; cache parser results |
| Multiple policies | Re-run rollout per policy | Batch rollouts; cache baseline trajectory |
| Variable count | 31 | Extensible — action/observation space shapes are parameterized |

---

## Sources

- PyTorch GRU documentation: https://docs.pytorch.org/docs/stable/generated/torch.nn.GRU.html
- Gymnasium environment creation guide: https://gymnasium.farama.org/introduction/create_custom_env/
- Gymnasium Env API: https://gymnasium.farama.org/api/env/
- GRU-D (missing values via masking): https://pmc.ncbi.nlm.nih.gov/articles/PMC5904216/ [Che et al., 2018]
- VS-GRU (variable-sensitive missing rate): https://www.mdpi.com/2076-3417/9/15/3041
- RobustScaler vs. MinMaxScaler vs. StandardScaler: https://scikit-learn.org/stable/auto_examples/preprocessing/plot_all_scaling.html
- Nixtla time series scaling strategies: https://nixtlaverse.nixtla.io/neuralforecast/docs/capabilities/time_series_scaling.html
- RNN/LSTM/GRU comparative study (2025): https://pmc.ncbi.nlm.nih.gov/articles/PMC12329085/
- Regularization techniques for RNNs: https://apxml.com/courses/rnns-and-sequence-modeling/chapter-10-evaluating-tuning-sequence-models/regularization-techniques-rnns
- WorldGym (world model policy evaluation): https://arxiv.org/html/2506.00613
