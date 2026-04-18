"""
RL policy evaluator — model architecture only (no training loop, env, or UI).

State: fixed-size US energy–economy vector aligned with EIA Open Data routes listed in
project requirements. Inactive features (policy-irrelevant metrics) use value 0.0 and
mask 0.0 so they are ignored through the forward pass; active features use mask 1.0.

Recommended training stack (placeholders — implement separately):
  - Offline RL / batch RL on historical EIA-aligned trajectories, or
  - Model-based rollouts from a US energy–economy simulator with STEO/AEO priors.
  - Dataset: materialize each named slot below from the cited /opendata/browser/* routes;
    store as parquet or zarr with aligned timestamps (monthly or annual).
  - Train with PPO or SAC on continuous actions, or BC pretrain then fine-tune.

Architecture rationale:
  - Medium-large MLP trunk (512→512→256) with LayerNorm + GELU: stable for ~50-dim
    masked continuous state without needing a very deep transformer.
  - Doubled input (values ⊕ mask): lets the network distinguish “true zero” from
    “missing / not applicable” without leaking masked slots into gradients as real zeros.
  - Actor head: Gaussian policy for continuous policy levers (e.g., fee level, subsidy rate).
  - Twin critics (SAC-style): mitigates overestimation when reward mixes economic vs energy.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Dict, List, Optional, Tuple

import gym
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal


# Sentinel for “not part of state for this policy” — use with mask=0.0 in training code.
MISSING_VALUE: float = 0.0
MISSING_MASK: float = 0.0
ACTIVE_MASK: float = 1.0


class EnvEnergyIndex(IntEnum):
    """Human-produced energy / pollution proxies (EIA-backed); ~half of total state."""

    # CO2 & emissions — SEDS, aggregates, electric power operational data
    SEDS_CO2_TOTAL = 0
    SEDS_CO2_ELECTRIC_POWER = 1
    SEDS_CO2_INDUSTRIAL = 2
    SEDS_CO2_TRANSPORT = 3
    EPOD_ELECTRIC_POWER_PLANT_CO2 = 4
    # Fossil consumption
    COAL_CONSUMPTION_TOTAL = 5
    COAL_SULFUR_CONTENT_PROXY = 6
    COAL_ASH_CONTENT_PROXY = 7
    COAL_STOCKS = 8
    PETROLEUM_GASOLINE_SUPPLIED = 9
    PETROLEUM_DISTILLATE_SUPPLIED = 10
    PETROLEUM_RESIDUAL_KEROSENE_SUPPLIED = 11
    NATURAL_GAS_CONSUMPTION_TOTAL = 12
    # Renewable & clean generation / capacity — electric power operational + OGC + state profiles
    GENERATION_SOLAR = 13
    GENERATION_WIND = 14
    GENERATION_HYDRO = 15
    GENERATION_GEOTHERMAL = 16
    GENERATION_NUCLEAR = 17
    GENERATION_COAL = 18
    GENERATION_NATURAL_GAS = 19
    CAPACITY_RENEWABLE_INDEX = 20
    CAPACITY_FOSSIL_INDEX = 21
    STATE_PROFILE_RENEWABLE_SHARE = 22
    DENSIFIED_BIOMASS_PRODUCTION = 23
    # Grid & efficiency — RTO, nuclear outages
    RTO_ACTUAL_DEMAND_AVG = 24
    RTO_FORECAST_DEMAND_AVG = 25
    RTO_NET_GENERATION_AVG = 26
    RTO_INTERCHANGE_NET_AVG = 27
    NUCLEAR_OUTAGE_PCT_US = 28


class EconHealthIndex(IntEnum):
    """Economic / expenditure / trade / outlook metrics; ~half of total state."""

    RETAIL_ELEC_PRICE_AVG = 29
    RETAIL_ELEC_SALES = 30
    RETAIL_ELEC_CUSTOMERS = 31
    NG_DELIVERED_PRICE_RESIDENTIAL = 32
    GASOLINE_RETAIL_PRICE = 33
    DIESEL_RETAIL_PRICE = 34
    COAL_MARKET_SALES_PRICE = 35
    SEDS_PRICE_PER_MMBTU_AVG = 36
    COAL_AGGREGATE_PRODUCTION = 37
    COAL_PRODUCTIVITY = 38
    COAL_EMPLOYEES = 39
    NG_MARKETED_PRODUCTION = 40
    CRUDE_OIL_PRODUCTION = 41
    CRUDE_OIL_RESERVES = 42
    CRUDE_OIL_IMPORTS_VOLUME = 43
    SEDS_TOTAL_ENERGY_EXPENDITURES = 44
    NG_CONSUMPTION_EXPENDITURES = 45
    PETROLEUM_NET_IMPORTS = 46
    NG_NET_IMPORTS = 47
    PETROLEUM_GASOLINE_STOCKS = 48
    PETROLEUM_DISTILLATE_STOCKS = 49
    STEO_PRICE_INDEX_18M = 50
    STEO_CO2_PROJECTION_18M = 51
    AEO_LONG_RUN_PRICE_INDEX = 52
    AEO_LONG_RUN_CO2 = 53


STATE_DIM: int = 54
ENV_SLOT_COUNT = len(EnvEnergyIndex)  # indices 0..28
ECON_SLOT_COUNT = len(EconHealthIndex)  # indices 29..53


def state_feature_names() -> List[str]:
    names: List[str] = [""] * STATE_DIM
    for e in EnvEnergyIndex:
        names[e.value] = e.name
    for e in EconHealthIndex:
        names[e.value] = e.name
    return names


@dataclass(frozen=True)
class PolicyEvaluatorConfig:
    state_dim: int = STATE_DIM
    action_dim: int = 4  # e.g., carbon price, subsidy, RPS stringency, fuel tax — tune later
    hidden_dims: Tuple[int, ...] = (512, 512, 256)
    q_hidden_dim: int = 256
    activation: str = "gelu"
    use_layer_norm: bool = True
    dropout: float = 0.1
    log_std_min: float = -20.0
    log_std_max: float = 2.0
    max_episode_steps: int = 120


class PolicyLensGymEnv(gym.Env):
    """
    OpenAI Gym-compatible environment scaffold for policy simulation.

    Notes:
      - This is architecture-only: transition and reward dynamics are placeholders.
      - Observations are split into values + mask to preserve metric relevance gating.
    """

    metadata = {"render_modes": []}

    def __init__(self, cfg: Optional[PolicyEvaluatorConfig] = None):
        super().__init__()
        self.cfg = cfg or PolicyEvaluatorConfig()
        self.observation_space = gym.spaces.Dict(
            {
                "values": gym.spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self.cfg.state_dim,),
                    dtype=np.float32,
                ),
                "mask": gym.spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(self.cfg.state_dim,),
                    dtype=np.float32,
                ),
            }
        )
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.cfg.action_dim,),
            dtype=np.float32,
        )
        self._step_count = 0
        self._values = np.zeros(self.cfg.state_dim, dtype=np.float32)
        self._mask = np.ones(self.cfg.state_dim, dtype=np.float32)

    def _obs(self) -> Dict[str, np.ndarray]:
        return {"values": self._values.copy(), "mask": self._mask.copy()}

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        super().reset(seed=seed)
        self._step_count = 0
        self._values = np.zeros(self.cfg.state_dim, dtype=np.float32)
        self._mask = np.ones(self.cfg.state_dim, dtype=np.float32)
        if options and "mask" in options:
            self._mask = np.asarray(options["mask"], dtype=np.float32).reshape(self.cfg.state_dim)
        return self._obs(), {}

    def step(
        self, action: np.ndarray
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        action = np.asarray(action, dtype=np.float32).reshape(self.cfg.action_dim)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self._step_count += 1
        # Placeholder transition: no dynamics until simulator implementation.
        self._values = self._values.astype(np.float32)
        reward = 0.0  # Placeholder reward; define in simulator/training pipeline.
        terminated = False
        truncated = self._step_count >= self.cfg.max_episode_steps
        info: Dict[str, Any] = {"applied_action": action}
        return self._obs(), reward, terminated, truncated, info


class MLPTrunk(nn.Module):
    def __init__(self, in_dim: int, hidden_dims: Tuple[int, ...], dropout: float, use_ln: bool):
        super().__init__()
        layers: List[nn.Module] = []
        d = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(d, h))
            if use_ln:
                layers.append(nn.LayerNorm(h))
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            d = h
        self.net = nn.Sequential(*layers)
        self._out_dim = d

    @property
    def out_dim(self) -> int:
        return self._out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PolicyLensActorCritic(nn.Module):
    """
    Gaussian actor + twin Q critics for continuous control (SAC-compatible).

    Inputs:
      state_values: (B, STATE_DIM) — use 0.0 where metric is irrelevant.
      state_mask:   (B, STATE_DIM) — 1.0 active, 0.0 ignored (same shape as requirements).
    """

    def __init__(self, cfg: Optional[PolicyEvaluatorConfig] = None):
        super().__init__()
        self.cfg = cfg or PolicyEvaluatorConfig()
        in_dim = 2 * self.cfg.state_dim
        self.trunk = MLPTrunk(
            in_dim,
            self.cfg.hidden_dims,
            self.cfg.dropout,
            self.cfg.use_layer_norm,
        )
        d = self.trunk.out_dim
        a = self.cfg.action_dim
        qh = self.cfg.q_hidden_dim
        self.actor_mean = nn.Linear(d, a)
        self.actor_log_std = nn.Parameter(torch.zeros(1, a))
        self.q1 = nn.Sequential(
            nn.Linear(d + a, qh),
            nn.LayerNorm(qh) if self.cfg.use_layer_norm else nn.Identity(),
            nn.GELU(),
            nn.Dropout(self.cfg.dropout) if self.cfg.dropout > 0 else nn.Identity(),
            nn.Linear(qh, 1),
        )
        self.q2 = nn.Sequential(
            nn.Linear(d + a, qh),
            nn.LayerNorm(qh) if self.cfg.use_layer_norm else nn.Identity(),
            nn.GELU(),
            nn.Dropout(self.cfg.dropout) if self.cfg.dropout > 0 else nn.Identity(),
            nn.Linear(qh, 1),
        )
        self.v = nn.Linear(d, 1)

    @classmethod
    def from_gym_env(cls, env: PolicyLensGymEnv) -> "PolicyLensActorCritic":
        obs_space = env.observation_space
        action_space = env.action_space
        if not isinstance(obs_space, gym.spaces.Dict):
            raise TypeError("Expected gym.spaces.Dict observation space with values/mask.")
        if not isinstance(action_space, gym.spaces.Box):
            raise TypeError("Expected gym.spaces.Box continuous action space.")
        values_space = obs_space["values"]
        if not isinstance(values_space, gym.spaces.Box):
            raise TypeError("Expected observation_space['values'] to be gym.spaces.Box.")
        cfg = PolicyEvaluatorConfig(
            state_dim=int(values_space.shape[0]),
            action_dim=int(action_space.shape[0]),
        )
        return cls(cfg)

    def encode(self, state_values: torch.Tensor, state_mask: torch.Tensor) -> torch.Tensor:
        if state_values.shape[-1] != self.cfg.state_dim or state_mask.shape != state_values.shape:
            raise ValueError(
                f"Expected state (..., {self.cfg.state_dim}) and matching mask; "
                f"got {state_values.shape}, {state_mask.shape}"
            )
        x = torch.cat([state_values * state_mask, state_mask], dim=-1)
        return self.trunk(x)

    def forward(
        self, state_values: torch.Tensor, state_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        z = self.encode(state_values, state_mask)
        mean = self.actor_mean(z)
        log_std = self.actor_log_std.clamp(self.cfg.log_std_min, self.cfg.log_std_max).expand_as(mean)
        std = log_std.exp()
        dist = Normal(mean, std)
        pre_tanh = dist.rsample()
        action = torch.tanh(pre_tanh)
        za = torch.cat([z, action], dim=-1)
        q1 = self.q1(za)
        q2 = self.q2(za)
        v = self.v(z)
        return action, q1, q2, v

    def act(
        self, state_values: torch.Tensor, state_mask: torch.Tensor, deterministic: bool = False
    ) -> torch.Tensor:
        z = self.encode(state_values, state_mask)
        mean = self.actor_mean(z)
        if deterministic:
            return torch.tanh(mean)
        log_std = self.actor_log_std.clamp(self.cfg.log_std_min, self.cfg.log_std_max).expand_as(mean)
        dist = Normal(mean, log_std.exp())
        return torch.tanh(dist.rsample())


def build_default_mask(active_indices: Optional[List[int]] = None) -> torch.Tensor:
    """All-active mask (training baseline). Pass indices for ablations."""
    m = torch.zeros(STATE_DIM)
    if active_indices is None:
        m[:] = ACTIVE_MASK
    else:
        for i in active_indices:
            m[i] = ACTIVE_MASK
    return m


def slice_report_dict(
    state_values: torch.Tensor, state_mask: torch.Tensor
) -> Dict[str, Tuple[float, bool]]:
    """Helper for report generation: name -> (value, active)."""
    names = state_feature_names()
    out: Dict[str, Tuple[float, bool]] = {}
    vals = state_values.view(-1).detach().cpu()
    masks = state_mask.view(-1).detach().cpu()
    for i, nm in enumerate(names):
        out[nm] = (float(vals[i]), bool(masks[i] > 0.5))
    return out


__all__ = [
    "MISSING_VALUE",
    "MISSING_MASK",
    "ACTIVE_MASK",
    "EnvEnergyIndex",
    "EconHealthIndex",
    "STATE_DIM",
    "ENV_SLOT_COUNT",
    "ECON_SLOT_COUNT",
    "PolicyLensGymEnv",
    "PolicyEvaluatorConfig",
    "PolicyLensActorCritic",
    "MLPTrunk",
    "state_feature_names",
    "build_default_mask",
    "slice_report_dict",
]
