# Technology Stack: PolicyLens

**Project:** PolicyLens — AI Government Policy Evaluator (DataHacks 2026)
**Domain:** ML simulation pipeline with LLM parsing, RL environment, GRU world model, Streamlit UI
**Researched:** 2026-04-18
**Overall confidence:** HIGH (all versions verified against official sources or PyPI)

---

## Recommended Stack

### Python Runtime

| Technology | Version | Why |
|------------|---------|-----|
| Python | **3.11** | Sweet spot: PyTorch 2.6 supports 3.11, Streamlit 1.52 supports 3.11, Gymnasium 1.2 supports 3.11. Avoid 3.12+ — PyTorch's `torch.compile` only gained 3.13 support in 2.6, but ecosystem tooling (Conda, some CUDA wheels) still lags on 3.12+. Avoid 3.9 — Streamlit dropped it. |

### Core ML

| Library | Version | Purpose | Rationale |
|---------|---------|---------|-----------|
| `torch` | **2.6.0** | GRU model, training loop, tensor ops | Latest stable as of Jan 2025. No GRU-specific breaking changes from 2.5→2.6. Stable `batch_first=True` support (use it — it aligns with `(batch, seq, features)` convention and avoids manual `.transpose()` calls). Conda packages are gone starting 2.6; install via pip. |
| `torchvision` | skip | N/A | Not needed for tabular/time-series work. Don't add it. |

### RL Environment

| Library | Version | Purpose | Rationale |
|---------|---------|---------|-----------|
| `gymnasium` | **1.0.0** or **1.2.3** | Custom simulation environment | Use 1.0.0 if you might later integrate with Stable-Baselines3 (SB3 has compatibility friction with 1.0+ auto-reset). Use 1.2.3 if you are purely evaluation-only with no RL agent training. Since PolicyLens uses Gymnasium as a structured simulator (not for agent training), **pin 1.0.0** to maximize downstream compatibility — the API is stable and the 1.2.x additions (VectorEnv cleanup, MuJoCo removal) are irrelevant to this project. |
| `stable-baselines3` | do NOT add | — | Brings in heavy dependencies and has documented incompatibility with Gymnasium 1.0+ auto-reset. Skip entirely. |

### LLM / Claude Integration

| Library | Version | Purpose | Rationale |
|---------|---------|---------|-----------|
| `anthropic` | **>=0.40.0, <1.0** (latest: 0.96.0) | Claude API policy parser | Current stable series. Use `anthropic.Anthropic()` (sync) for Streamlit's single-threaded execution model. Do not use `AsyncAnthropic` — mixing asyncio into Streamlit requires `asyncio.run()` workarounds that break under Streamlit's event loop. Streaming is available via `.stream()` context manager and is useful for the report-generation step. |

### Data / EIA

| Library | Version | Purpose | Rationale |
|---------|---------|---------|-----------|
| `requests` | **>=2.31** | EIA API v2 HTTP calls | Do not use `myeia` (0.4.8, marked Inactive by Snyk, 21 GitHub stars, sole maintainer). Do not use `eiapy` (targets v1 API, unmaintained). Call the EIA v2 REST API directly with `requests`. The API is simple: GET to `https://api.eia.gov/v2/{route}/data/` with `api_key`, `frequency`, `facets`, `data`, `start`, `end` as query params. The response is JSON; parse with `pandas`. |
| `pandas` | **>=2.0, <3.0** | Tabular data wrangling, EIA response parsing | Stable, 2.x copy-on-write semantics are mature. Use `pd.DataFrame` as the bridge between EIA JSON and torch tensors. |
| `numpy` | **>=1.26, <2.0** | Array ops, tensor prep | NumPy 2.0 has ABI-breaking changes that affect PyTorch 2.6 compatibility. Pin `<2.0` until PyTorch explicitly certifies 2.x support. |
| `python-dotenv` | **>=1.0** | `.env` loading for `EIA_API_KEY`, `ANTHROPIC_API_KEY` | Simplest approach for hackathon-scale config. Call `load_dotenv()` once at app entry point. |

### UI

| Library | Version | Purpose | Rationale |
|---------|---------|---------|-----------|
| `streamlit` | **1.40.x – 1.52.0** | App shell, layout, dark theme | Latest stable is 1.52.0 (Dec 2025). Dark theme: set via `.streamlit/config.toml` with `[theme] base = "dark"`. Do not pass `theme=` in `st.set_page_config()` — that parameter does not exist; theme is config-file-only. |
| `plotly` | **5.24.x** | Charts | Plotly 6.x exists (6.7.0 as of Apr 2026) but introduces breaking changes vs 5.x. Streamlit's `st.plotly_chart` uses Plotly internals and the Streamlit team has not explicitly validated 6.x. Pin to latest 5.x (`5.24.x`) for a hackathon to avoid surprises. Use `theme="streamlit"` in `st.plotly_chart()` — it inherits your dark palette automatically. Do NOT pass arbitrary `**kwargs` to `st.plotly_chart`; use the explicit `config=` dict instead (deprecated in 2025 releases). |

---

## PyTorch GRU Training Utilities

These are the utilities worth using for the GRU transition model specifically.

### Model Definition

```python
import torch
import torch.nn as nn

class GRUTransitionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,       # input shape: (batch, seq_len, features)
            dropout=dropout if num_layers > 1 else 0.0,  # dropout ignored on single-layer
        )
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, h0=None):
        # x: (batch, seq_len, input_dim)
        out, hn = self.gru(x, h0)   # out: (batch, seq_len, hidden_dim)
        return self.fc(out[:, -1, :]), hn  # return last step + hidden for rollout
```

Always return `hn` (final hidden state). The simulation rollout loop needs to thread hidden state across steps — this is the key difference between training (feed full sequences) and evaluation (step-by-step with carried state).

### Loss and Optimizer

| Choice | Rationale |
|--------|-----------|
| `nn.MSELoss()` | Standard for multi-dimensional state regression. PolicyLens predicts energy metrics (CO2 ppm, renewable %, grid load) — MSE penalizes large deviations. |
| `nn.HuberLoss(delta=1.0)` | Better if EIA data has outliers (extreme weather events, policy shocks). More robust than MSE. Prefer this. |
| `torch.optim.Adam(lr=1e-3, weight_decay=1e-4)` | Standard. Weight decay prevents overfitting on limited EIA historical data. |
| `torch.optim.lr_scheduler.ReduceLROnPlateau(patience=5)` | Useful for multi-epoch training on tabular data where loss plateaus. |

### Training Utilities Worth Using

| Utility | Use Case |
|---------|----------|
| `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)` | GRUs suffer from gradient explosion on long sequences. Always clip before `optimizer.step()`. |
| `model.eval()` + `torch.no_grad()` | Required for evaluation/rollout. `model.eval()` disables dropout; `no_grad()` skips gradient tracking and reduces memory. Use both together during simulation. |
| `torch.save(model.state_dict(), path)` / `model.load_state_dict()` | Checkpoint the trained model. Load it in the Streamlit app without retraining. |
| `torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)` | Wrap your EIA time-series dataset. Use `shuffle=False` for time-ordered data during evaluation. |
| `torch.utils.data.TensorDataset` | Simplest wrapper for `(X_tensor, y_tensor)` pairs from pandas-processed EIA data. |

### Do NOT Use

| What | Why Not |
|------|---------|
| `torch.compile()` on the GRU | `torch.compile` has known limitations with stateful RNNs that carry hidden state across calls. It will either fail or silently produce wrong gradients. Skip for this project. |
| `PackedSequence` / `pack_padded_sequence` | Only needed for variable-length sequences. EIA data is regular-frequency (monthly/annual) — all sequences are the same length. Adds complexity with no benefit. |
| `torchrl` | Explicitly incompatible with Gymnasium 1.0+ per its own maintainers. |
| `lightning` (PyTorch Lightning) | Overkill for a hackathon GRU. Adds indirection that makes debugging harder under time pressure. |

---

## EIA API v2 — Direct Usage Pattern

Skip all wrapper libraries. The v2 API is clean enough to call directly:

```python
import os
import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
EIA_KEY = os.environ["EIA_API_KEY"]

def fetch_eia(route: str, frequency: str, data_cols: list[str],
              facets: dict, start: str, end: str) -> pd.DataFrame:
    url = f"https://api.eia.gov/v2/{route}/data/"
    params = {
        "api_key": EIA_KEY,
        "frequency": frequency,
        "data[]": data_cols,
        "start": start,
        "end": end,
        "sort[0][column]": "period",
        "sort[0][direction]": "asc",
        "length": 5000,
    }
    for facet_key, facet_vals in facets.items():
        for v in facet_vals:
            params[f"facets[{facet_key}][]"] = v

    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    return pd.DataFrame(resp.json()["response"]["data"])
```

Relevant EIA v2 routes for energy/environmental policy:
- `electricity/retail-sales` — retail electricity prices by state/sector
- `electricity/rto/fuel-type-data` — real-time grid fuel mix (solar, wind, gas, coal)
- `co2-emissions/co2-emissions-aggregates` — state-level CO2 by sector
- `natural-gas/sum/snd` — natural gas supply and disposition
- `total-energy/data` — MSN series (broad energy statistics)

---

## Gymnasium Custom Environment Pattern

```python
import gymnasium as gym
from gymnasium import spaces
import numpy as np

class PolicyEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, baseline_state: np.ndarray, gru_model, device):
        super().__init__()
        self.gru_model = gru_model
        self.device = device

        # Define spaces — use Box for continuous energy state vectors
        n_state = baseline_state.shape[0]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(n_state,), dtype=np.float32
        )
        # Policy actions: e.g., carbon tax rate, renewable subsidy level
        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(3,), dtype=np.float32
        )
        self._baseline = baseline_state

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)           # required — initializes self.np_random
        self._state = self._baseline.copy()
        self._hidden = None                # GRU hidden state, reset each episode
        self._step_count = 0
        return self._state.astype(np.float32), {}

    def step(self, action: np.ndarray):
        # Feed (state + action) through GRU transition model
        x = np.concatenate([self._state, action])[np.newaxis, np.newaxis, :]
        x_t = torch.tensor(x, dtype=torch.float32, device=self.device)

        self.gru_model.eval()
        with torch.no_grad():
            next_state_t, self._hidden = self.gru_model(x_t, self._hidden)

        self._state = next_state_t.squeeze().cpu().numpy()
        self._step_count += 1

        reward = self._compute_reward(self._state, action)
        terminated = False          # evaluation horizon controls episode length
        truncated = self._step_count >= 20

        return self._state.astype(np.float32), reward, terminated, truncated, {}

    def _compute_reward(self, state, action):
        # Domain-specific: lower CO2, higher renewables, stable grid = positive
        raise NotImplementedError
```

Key point: `terminated=False` always, `truncated=True` at horizon. This is correct for an evaluation environment where there is no true terminal state, only a time limit.

---

## Streamlit Dark Theme Config

Create `.streamlit/config.toml` in the project root:

```toml
[theme]
base = "dark"
primaryColor = "#4CAF50"        # green accent — appropriate for energy/env theme
backgroundColor = "#0E1117"
secondaryBackgroundColor = "#1A1D23"
textColor = "#FAFAFA"
font = "sans serif"
```

In `app.py`, do NOT attempt to pass theme in code — it is config-file-only:

```python
st.set_page_config(
    page_title="PolicyLens",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)
```

For Plotly charts, use `theme="streamlit"` to inherit the dark palette:

```python
st.plotly_chart(fig, use_container_width=True, theme="streamlit")
# If you need Plotly's native dark template instead:
fig.update_layout(template="plotly_dark")
st.plotly_chart(fig, use_container_width=True, theme=None)
```

Do not pass arbitrary kwargs to `st.plotly_chart` — use the `config=` dict:

```python
st.plotly_chart(fig, config={"displayModeBar": False}, theme="streamlit")
```

---

## Alternatives Considered and Rejected

| Category | Recommended | Rejected | Why Rejected |
|----------|-------------|----------|--------------|
| EIA client | `requests` direct | `myeia 0.4.8` | Marked inactive (Snyk), 21 stars, sole maintainer — risk for hackathon reliability |
| EIA client | `requests` direct | `eiapy` | Targets v1 API only; v1 is functionally deprecated |
| RNN type | `nn.GRU` | `nn.LSTM` | GRU has fewer parameters, trains faster on small datasets, equivalent performance on short sequences; EIA data is sparse so fewer params is better |
| Training framework | raw PyTorch | `pytorch-lightning` | Overkill; debugging under time pressure is harder |
| RL compatibility | `gymnasium==1.0.0` | `gymnasium>=1.2.0` | 1.2.x moved MuJoCo out, no benefit to this project; 1.0.0 is more widely compatible if SB3 is added later |
| Plotly version | `plotly 5.24.x` | `plotly 6.x` | Streamlit's Plotly integration has not explicitly validated 6.x; 5.24.x is known-good |
| Async Claude | `anthropic.Anthropic()` (sync) | `AsyncAnthropic` | Streamlit runs synchronous; async requires `asyncio.run()` hacks that break under Streamlit's internal event loop |
| Env compilation | no `torch.compile` | `torch.compile` on GRU | Known limitations with stateful RNNs carrying hidden state between calls |
| NumPy | `numpy<2.0` | `numpy>=2.0` | ABI-breaking change; PyTorch 2.6 has not fully certified NumPy 2.x |

---

## Installation

```bash
# Core ML
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
# (for CUDA 12.1: --index-url https://download.pytorch.org/whl/cu121)

# RL environment
pip install "gymnasium==1.0.0"

# Claude API
pip install "anthropic>=0.40.0,<1.0"

# EIA data
pip install "requests>=2.31" "pandas>=2.0,<3.0" "numpy>=1.26,<2.0"

# Config
pip install "python-dotenv>=1.0"

# UI
pip install "streamlit>=1.40,<=1.52.0" "plotly>=5.20,<6.0"
```

Minimal `requirements.txt`:

```
torch==2.6.0
gymnasium==1.0.0
anthropic>=0.40.0,<1.0
requests>=2.31
pandas>=2.0,<3.0
numpy>=1.26,<2.0
python-dotenv>=1.0
streamlit>=1.40,<=1.52.0
plotly>=5.20,<6.0
```

---

## Known Conflicts and Gotchas

### Conflict 1: NumPy 2.x + PyTorch 2.6
**Risk:** `numpy>=2.0` ships with ABI changes that can break PyTorch tensor↔ndarray interop (`.numpy()` calls, `torch.from_numpy()`).
**Fix:** Pin `numpy<2.0` explicitly. Without this pin, pip may resolve to 2.x.

### Conflict 2: Gymnasium 1.0 Auto-Reset vs. TorchRL
**Risk:** If you add `torchrl` later (e.g., for replay buffers), it explicitly does not support Gymnasium 1.0+.
**Fix:** Don't add `torchrl`. Use raw PyTorch tensors for any data buffering needed.

### Gotcha 1: `nn.GRU` dropout on single layer
**Risk:** `nn.GRU(dropout=0.2, num_layers=1)` — PyTorch silently ignores dropout when `num_layers=1` without warning. You may think you're regularizing but aren't.
**Fix:** Only set `dropout > 0` when `num_layers >= 2`, or add a separate `nn.Dropout` layer after the GRU output.

### Gotcha 2: Gymnasium `reset()` must call `super().reset(seed=seed)`
**Risk:** Skipping `super().reset(seed=seed)` means `self.np_random` is never initialized. Any randomness in your environment will fail or be unseeded.
**Fix:** Always call it on the first line of your `reset()` implementation.

### Gotcha 3: Streamlit widget state resets on interaction
**Risk:** Each Streamlit widget interaction re-runs the entire script. If your GRU inference is called top-level (not cached), it re-runs on every slider move.
**Fix:** Use `@st.cache_resource` for model loading and `@st.cache_data` for EIA data fetching:
```python
@st.cache_resource
def load_model():
    model = GRUTransitionModel(...)
    model.load_state_dict(torch.load("model.pt", map_location="cpu"))
    return model.eval()

@st.cache_data(ttl=3600)
def fetch_baseline_data():
    return fetch_eia(route="electricity/rto/fuel-type-data", ...)
```

### Gotcha 4: Anthropic SDK version vs. model name strings
**Risk:** Claude model name strings change between SDK versions. `claude-3-5-sonnet-20241022` may not be valid in older SDK versions.
**Fix:** Always check `anthropic.__version__` and use the model name from the current [Anthropic models documentation](https://docs.anthropic.com/en/docs/about-claude/models). As of early 2026, `claude-sonnet-4-6` is a valid model name.

### Gotcha 5: EIA API rate limits
**Risk:** EIA v2 API has a rate limit of 4,800 requests/hour per key. Streamlit reruns can trigger rapid repeated fetches during UI development.
**Fix:** `@st.cache_data(ttl=3600)` on all EIA fetch functions. Make EIA calls lazy (only on explicit user trigger, not on every script rerun).

---

## Sources

- [PyTorch 2.6 Release Blog](https://pytorch.org/blog/pytorch2-6/)
- [torch.nn.GRU Documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.GRU.html)
- [Gymnasium Release Notes](https://gymnasium.farama.org/gymnasium_release_notes/index.html)
- [Create a Custom Gymnasium Environment](https://gymnasium.farama.org/introduction/create_custom_env/)
- [Gymnasium Terminated/Truncated Step API](https://farama.org/Gymnasium-Terminated-Truncated-Step-API)
- [TorchRL / Gymnasium 1.0 Incompatibility Issue](https://github.com/pytorch/rl/issues/2477)
- [EIA API v2 Technical Documentation](https://www.eia.gov/opendata/documentation.php)
- [myeia PyPI](https://pypi.org/project/myeia/)
- [anthropic PyPI](https://pypi.org/project/anthropic/)
- [Streamlit 2025 Release Notes](https://docs.streamlit.io/develop/quick-reference/release-notes/2025)
- [Streamlit Theming Docs](https://docs.streamlit.io/develop/concepts/configuration/theming)
- [st.plotly_chart Docs](https://docs.streamlit.io/develop/api-reference/charts/st.plotly_chart)
- [plotly PyPI Releases](https://github.com/plotly/plotly.py/releases)
