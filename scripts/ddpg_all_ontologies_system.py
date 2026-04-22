"""
Author: Rosie-Plimmer 
"""

import csv
import os
import time
import re
import logging
import math
import random
from pathlib import Path
import numpy as np
import pandas as pd
from collections import deque, namedtuple
from typing import List, Tuple
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# ──────────────────────────────────────────────
# ▶ Paths and logging
# ──────────────────────────────────────────────
DEFAULT_N_RUNS = 5
RESULTS_BASE_DIR = Path.cwd() / "reinforcement_learning/ddpg"

def get_results_root(run_number: int) -> Path:
    root = RESULTS_BASE_DIR / f"new_ddpg_results_{run_number}"
    root.mkdir(parents=True, exist_ok=True)
    return root

# Default so module-level code still has a valid results root
RESULTS_ROOT = get_results_root(1)

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

for h in list(log.handlers):
    log.removeHandler(h)
_console = logging.StreamHandler()
_console.setLevel(logging.INFO)
_console.setFormatter(logging.Formatter("%(message)s"))
log.addHandler(_console)
log.propagate = False

# Regex helpers 
_FLOAT_RE = re.compile(r'[-+]?(?:\d+(?:[.,]\d*)?|\.\d+)(?:[eE][-+]?\d+)?')
_FIRST_QUOTED_AFTER_TIME = re.compile(r'^[^,]*,\s*"([^"]*)"')

# ──────────────────────────────────────────────
# ▶ Ontology globals 
# ──────────────────────────────────────────────
POWER_REL_EPS   = 0.05 # relative tolerance for "equal": |solar-heatpump| <= eps * max(1, |heatpump|)
REWARD_MATRIX   = np.array([
    [-1, -1,  1],   # Heating
    [-1,  1, -1],   # Abstain
    [ 1, -1, -1],   # Cooling
], dtype=np.float32)

# ──────────────────────────────────────────────
# ▶ Hyper‑parameters
# ──────────────────────────────────────────────
LR_ACTOR, LR_CRITIC = 1e-4, 1e-3
GAMMA, TAU          = 0.99, 5e-3
BATCH_SIZE, BUFFER_SIZE = 256, 200_000
WARMUP_STEPS       = 5_000
MAX_EPISODES, MAX_STEPS = 100, 1_000

# Shared-eval logging (comparable across PPO/DDPG and ontology/none)
EVAL_EVERY = 1  # set to 5 or 10 to reduce overhead

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state', 'done'))

# ──────────────────────────────────────────────
# ▶ Helper classes and functions
# ──────────────────────────────────────────────

def rollout_shared_metrics(env, act_fn, max_steps=MAX_STEPS):
    """
    Deterministic rollout that accumulates the SHARED metric: info["shaping"].
    Returns shared_return and shared_per_step. Higher is better (closer to 0).
    """
    s = env.reset()
    done = False
    steps = 0
    shared_ret = 0.0

    while not done and steps < max_steps:
        a = act_fn(s)
        s, r, done, info = env.step(a)
        shared_ret += float(info.get("shaping", 0.0))
        steps += 1

    return {
        "shared_return": float(shared_ret),
        "shared_per_step": float(shared_ret / max(1, steps)),
        "steps": int(steps),
    }


def rollout_optimality_metrics(env, policy_act_fn, max_steps=MAX_STEPS):
    """
    Computes policy optimality on the shared metric (shaping) vs:
      - baseline: persistence (predict previous target, clipped)
      - oracle:   predict clipped target (best possible under action bounds)

    Returns per-step shared scores + optimality_ratio (1=oracle, 0=baseline).
    """
    s = env.reset()
    done = False
    steps = 0

    policy_shared = 0.0
    baseline_shared = 0.0
    oracle_shared = 0.0

    prev_target = None
    max_flex = float(getattr(env, "max_flex", 1.0))

    while not done and steps < max_steps:
        a = policy_act_fn(s)
        s, r, done, info = env.step(a)

        target = float(info.get("target", 0.0))
        # policy shaping from env
        policy_shared += float(info.get("shaping", 0.0))

        # baseline: previous target (persistence), clipped to feasible pred range
        if prev_target is None:
            prev_target = target
        baseline_pred = float(np.clip(prev_target, 0.0, max_flex))
        baseline_shared += -abs(target - baseline_pred) / (max_flex + 1e-8)

        # oracle: clipped target (best possible pred under bounds)
        oracle_pred = float(np.clip(target, 0.0, max_flex))
        oracle_shared += -abs(target - oracle_pred) / (max_flex + 1e-8)

        prev_target = target
        steps += 1

    policy_ps = policy_shared / max(1, steps)
    base_ps   = baseline_shared / max(1, steps)
    oracle_ps = oracle_shared / max(1, steps)

    denom = (oracle_ps - base_ps)
    opt_ratio = (policy_ps - base_ps) / (denom + 1e-8) if abs(denom) > 1e-8 else np.nan
    opt_gap   = oracle_ps - policy_ps  # best is 0

    return {
        "policy_shared_return": float(policy_shared),
        "policy_shared_per_step": float(policy_ps),
        "baseline_shared_return": float(baseline_shared),
        "baseline_shared_per_step": float(base_ps),
        "oracle_shared_return": float(oracle_shared),
        "oracle_shared_per_step": float(oracle_ps),
        "optimality_ratio": float(opt_ratio) if opt_ratio == opt_ratio else np.nan,
        "optimality_gap_per_step": float(opt_gap),
        "steps": int(steps),
        "max_flex": float(max_flex),
    }


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.memory = deque(maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size: int):
        transitions = random.sample(self.memory, batch_size)
        return Transition(*zip(*transitions))

    def __len__(self):
        return len(self.memory)

class OUNoise:
    def __init__(self, size, mu=0., theta=.15, sigma=.2):
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.state = np.copy(self.mu)

    def reset(self):
        self.state = np.copy(self.mu)

    def sample(self):
        dx = self.theta * (self.mu - self.state) + \
             self.sigma * np.random.randn(len(self.state))
        self.state += dx
        return self.state

def fanin_init(tensor):
    fan_in = tensor.size(0)
    bound = 1. / math.sqrt(fan_in)
    return tensor.data.uniform_(-bound, bound)

def _norm_training_input(x: str) -> str:
    """Normalise training_input aliases to: 'solar' | 'heatpump' | 'combined'."""
    t = str(x).lower().replace("-", "_").replace(" ", "")
    if t in ("solar", "solar_only", "panel", "pv"):
        return "solar"
    if t in ("heat", "heat_only", "heatpump", "hp"):
        return "heatpump"
    if t in ("combined", "both", "solar_plus_heat", "solar+heat", "sum"):
        return "combined"
    raise ValueError("training_input must be one of {'solar','heatpump','combined'}")

# Simple 2-layer MLP mapping state → action
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.out = nn.Linear(128, action_dim)
        self.reset_parameters()

    def reset_parameters(self):
        fanin_init(self.fc1.weight); fanin_init(self.fc2.weight)
        self.out.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return torch.tanh(self.out(x))   # action ∈ [-1,1]

# Q-network that evaluates (state, action) pairs
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fcs = nn.Linear(state_dim, 256)
        self.fca = nn.Linear(action_dim, 256)
        self.fc2 = nn.Linear(512, 128)
        self.out = nn.Linear(128, 1)
        self.reset_parameters()

    def reset_parameters(self):
        fanin_init(self.fcs.weight); fanin_init(self.fca.weight)
        self.out.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, s, a):
        xs = F.relu(self.fcs(s))
        xa = F.relu(self.fca(a))
        x  = torch.cat((xs, xa), dim=1)
        x  = F.relu(self.fc2(x))
        return self.out(x)

# ──────────────────────────────────────────────
# ▶  EnergyEnv
#       Ontology modes:
#       - "solar_vs_heatpump"           (Ontology 1)
#       - "baseload_vs_net"             (Ontology 2)
#       - "battery_vs_solarbaseload"    (Ontology 3)
#       - "temperature_vs_thr"          (Ontology 4)
#       - "combined"                    (weighted average of enabled ontologies)
#       - "none"                        (no ontology)
#       Training input modes:
#       - "solar"                       (only train on solar panel flex powers)
#       - "heatpump"                    (only train on heatpump flex powers)
#       - "combined"                    (train on solar panel and heatpump flex powers)
# ──────────────────────────────────────────────
class SolarEnv:
    def __init__(
        self,
        solar_csv: str | Path | np.ndarray | None,      
        heatpump_csv:  str | Path | np.ndarray | None,   
        win: int = 1,
        eps: float = POWER_REL_EPS,
        ontology: str = "combined",
        training_input: str = "solar",                    # 'solar' | 'heatpump' | 'combined'
        # External series for additional ontologies (train or test versions)        
        baseload_csv:    str | Path | np.ndarray | None = None,
        battery_csv:     str | Path | np.ndarray | None = None,
        temperature_csv: str | Path | np.ndarray | None = None,
        # Weights for combined ontologies
        combine_weights: tuple[float, ...] = (1.0, 1.0, 1.0, 1.0),
        # Temperature threshold
        temp_threshold: float = 23.0,
    ) -> None:
        self.w = int(win)
        self.t = 0
        self.ontology = str(ontology).lower()
        self.training_input = _norm_training_input(training_input)
        self.temp_threshold = float(temp_threshold)
        self._eps = 1e-6
        self.eps = float(eps)
        self.last_action_idx: int = 1  # start with “abstain”

        # Load SOLAR series (store as self.solar) 
        if isinstance(solar_csv, (str, Path)):
            self.solar = self._load_flex_secondcol_midabs(solar_csv)
        elif solar_csv is None:
            self.solar = None
        else:
            self.solar = np.asarray(solar_csv, dtype=np.float32).copy()

        # Load HEAT‑PUMP series (store as self.heatpump)
        if isinstance(heatpump_csv, (str, Path)):
            self.heatpump = self._load_flex_secondcol_midabs(heatpump_csv)
        elif heatpump_csv is None:
            self.heatpump = None
        else:
            self.heatpump = np.asarray(heatpump_csv, dtype=np.float32).copy()

        # Baseload series (row‑wise mean) if needed
        self.baseload: np.ndarray | None = None
        if self.ontology in ("baseload_vs_net", "battery_vs_solarbaseload", "temperature_vs_thr", "combined"):
            if isinstance(baseload_csv, (str, Path)):
                self.baseload = self._load_baseload_rowwise_mean(baseload_csv)
            elif isinstance(baseload_csv, np.ndarray):
                self.baseload = np.asarray(baseload_csv, dtype=np.float32).copy()
            elif self.ontology in ("baseload_vs_net", "combined"):
                raise ValueError("SolarEnv: baseload_csv is required for this ontology.")

        # Battery series
        self.battery: np.ndarray | None = None
        if self.ontology in ("battery_vs_solarbaseload", "combined"):
            if isinstance(battery_csv, (str, Path)):
                self.battery = self._load_battery_secondcol(battery_csv)
            elif isinstance(battery_csv, np.ndarray):
                self.battery = np.asarray(battery_csv, dtype=np.float32).copy()
            elif self.ontology == "battery_vs_solarbaseload":
                raise ValueError("SolarEnv: battery_csv is required for this ontology.")

        # Temperature series
        self.temperature: np.ndarray | None = None
        if self.ontology in ("temperature_vs_thr", "combined"):
            if isinstance(temperature_csv, (str, Path)):
                self.temperature = self._load_temperature_rowwise_max(temperature_csv)
            elif isinstance(temperature_csv, np.ndarray):
                self.temperature = np.asarray(temperature_csv, dtype=np.float32).copy()
            elif self.ontology == "temperature_vs_thr":
                raise ValueError("SolarEnv: temperature_csv is required for this ontology.")

        # Align lengths
        lens = []
        if self.solar       is not None: lens.append(len(self.solar))
        if self.heatpump        is not None: lens.append(len(self.heatpump))
        if self.baseload    is not None: lens.append(len(self.baseload))
        if self.battery     is not None: lens.append(len(self.battery))
        if self.temperature is not None: lens.append(len(self.temperature))
        if not lens:
            raise ValueError("Empty time series: no solar/heatpump/aux inputs available.")
        T = min(lens)
        if self.solar       is not None: self.solar       = self.solar[:T].astype(np.float32)
        if self.heatpump        is not None: self.heatpump        = self.heatpump[:T].astype(np.float32)
        if self.baseload    is not None: self.baseload    = self.baseload[:T].astype(np.float32)
        if self.battery     is not None: self.battery     = self.battery[:T].astype(np.float32)
        if self.temperature is not None: self.temperature = self.temperature[:T].astype(np.float32)

        # Select training signal (self.flex)
        ti = self.training_input
        if ti == "solar":
            if self.solar is None:
                raise ValueError("training_input='solar' requires a solar series.")
            self.flex = self.solar
        elif ti == "heatpump":
            if self.heatpump is None:
                raise ValueError("training_input='heatpump' requires a heatpump series.")
            self.flex = self.heatpump
        else:  # 'combined' == solar + heatpump
            if self.solar is None or self.heatpump is None:
                raise ValueError("training_input='combined' requires both solar and heatpump series.")
            self.flex = (self.solar + self.heatpump).astype(np.float32)

        # Scaling for shaping (use chosen training signal stats) 
        q = np.quantile(self.flex, 0.995) if np.any(self.flex) else 0.0
        self.max_flex = float(q if q > 0 else (np.max(np.abs(self.flex)) + 1e-6))

        # Normalize combined weights 
        w = tuple(abs(float(x)) for x in (combine_weights if isinstance(combine_weights, (list, tuple)) else (combine_weights,)))
        if   len(w) == 1: w = (w[0], 0.0, 0.0, 0.0)
        elif len(w) == 2: w = (w[0], w[1], 0.0, 0.0)
        elif len(w) == 3: w = (w[0], w[1], w[2], 0.0)
        else:              w = (w[0], w[1], w[2], w[3])
        s = sum(w) or 1.0
        self._w_svh, self._w_bln, self._w_bat, self._w_tmp = (w[0]/s, w[1]/s, w[2]/s, w[3]/s)

    @staticmethod
    def _load_flex_secondcol_midabs(path: str | Path) -> np.ndarray:
        """FLEX CSV: second logical field is 'x,y,z'; take middle, return abs."""
        vals: list[float] = []
        with open(path, "r", encoding="utf-8-sig", newline="") as f:
            _ = f.readline()
            for line in f:
                line = line.strip()
                if not line: continue
                field = None
                m = _FIRST_QUOTED_AFTER_TIME.search(line)
                if m: field = m.group(1)
                if field is None:
                    row = next(csv.reader([line]))
                    for c in row[1:]:
                        s = str(c).strip().strip('"')
                        if not s: continue
                        nums = _FLOAT_RE.findall(s)
                        if len(nums) >= 1:
                            field = s; break
                    if field is None: continue
                s = str(field).strip().strip('"')
                nums = _FLOAT_RE.findall(s)
                if not nums: continue
                idx = 1 if len(nums) >= 3 else (len(nums)-1)//2
                try: v = float(nums[idx].replace(",", "."))
                except Exception: continue
                if math.isfinite(v): vals.append(abs(v))
        return np.asarray(vals, dtype=np.float32)

    @staticmethod
    def _load_baseload_rowwise_mean(path: str | Path) -> np.ndarray:
        """Baseload CSV: per-row mean of all numeric values after timestamp."""
        vals: list[float] = []
        with open(path, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.reader(f); _ = next(reader, None)
            for row in reader:
                if not row: continue
                nums: list[float] = []
                for cell in row[1:]:
                    s = str(cell).strip().strip('"')
                    if not s: continue
                    for tok in _FLOAT_RE.findall(s):
                        try: v = float(tok.replace(",", ".")); 
                        except Exception: continue
                        if math.isfinite(v): nums.append(v)
                if nums: vals.append(float(sum(nums)/len(nums)))
        return np.asarray(vals, dtype=np.float32)

    @staticmethod
    def _load_battery_secondcol(path: str | Path) -> np.ndarray:
        """Battery CSV: first numeric found after timestamp."""
        vals: list[float] = []
        with open(path, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.reader(f); _ = next(reader, None)
            for row in reader:
                if not row: continue
                val, found = 0.0, False
                for cell in row[1:]:
                    s = str(cell).strip().strip('"')
                    if not s: continue
                    m = _FLOAT_RE.findall(s)
                    if m:
                        try:
                            val = float(m[0].replace(",", ".")); found = True; break
                        except Exception:
                            pass
                vals.append(val if found else 0.0)
        return np.asarray(vals, dtype=np.float32)

    @staticmethod
    def _load_temperature_rowwise_max(path: str | Path) -> np.ndarray:
        """Temperature CSV: per-row MAX across all numeric cells after timestamp."""
        vals: list[float] = []
        with open(path, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.reader(f); _ = next(reader, None)
            for row in reader:
                if not row: vals.append(0.0); continue
                found_vals = []
                for cell in row[1:]:
                    s = str(cell).strip().strip('"')
                    if not s: continue
                    for tok in _FLOAT_RE.findall(s):
                        try:
                            v = float(tok.replace(",", ".")); 
                            if math.isfinite(v): found_vals.append(v)
                        except Exception:
                            continue
                vals.append(max(found_vals) if found_vals else 0.0)
        return np.asarray(vals, dtype=np.float32)

    # relation helpers (use self.solar / self.heatpump directly)
    def _rel_solar_vs_heatpump(self) -> int:
        if self.solar is None or self.heatpump is None or self.t >= len(self.solar) or self.t >= len(self.heatpump):
            return 1 # equal/neutral when relation cannot be computed
        s = float(self.solar[self.t]); h = float(self.heatpump[self.t])
        den = max(1.0, abs(h))
        if (s - h) > self.eps * den: return 0
        if (h - s) > self.eps * den: return 2
        return 1

    def _rel_baseload_vs_net(self) -> int:
        if (self.baseload is None or self.solar is None or self.heatpump is None or
            self.t >= len(self.baseload) or self.t >= len(self.solar) or self.t >= len(self.heatpump)):
            return 1
        s = float(self.solar[self.t]); h = float(self.heatpump[self.t]); b = float(self.baseload[self.t])
        d = b + s - h
        den = max(1.0, abs(b))
        if d >  self.eps * den: return 0
        if d < -self.eps * den: return 2
        return 1

    def _rel_battery_vs_solarbaseload(self) -> int:
        if (self.battery is None or self.baseload is None or self.solar is None or
            self.t >= len(self.battery) or self.t >= len(self.baseload) or self.t >= len(self.solar)):
            return 1
        solar_signed = -float(self.solar[self.t])   # actual (negative) solar
        bload = float(self.baseload[self.t])
        batt  = float(self.battery[self.t])
        d = batt - (solar_signed + bload)
        den = max(1.0, abs(batt))
        if d >  self.eps * den: return 0
        if d < -self.eps * den: return 2
        return 1

    def _rel_temperature_vs_thr(self) -> int:
        if self.temperature is None or self.t >= len(self.temperature): return 1
        temp = float(self.temperature[self.t]); 
        thr = self.temp_threshold
        if (temp - thr) >  self.eps * max(1.0, abs(thr)): return 0  # Above
        if (thr  - temp) > self.eps * max(1.0, abs(thr)): return 2  # Below
        return 1                                                    # Equal

    def _categorise_relation(self) -> int:
        if self.ontology == "solar_vs_heatpump":        return self._rel_solar_vs_heatpump()
        if self.ontology == "baseload_vs_net":          return self._rel_baseload_vs_net()
        if self.ontology == "battery_vs_solarbaseload": return self._rel_battery_vs_solarbaseload()
        if self.ontology == "temperature_vs_thr":       return self._rel_temperature_vs_thr()
        if self.baseload   is not None: return self._rel_baseload_vs_net()
        if self.temperature is not None: return self._rel_temperature_vs_thr()
        if self.battery    is not None: return self._rel_battery_vs_solarbaseload()
        return self._rel_solar_vs_heatpump()

    @staticmethod
    def _battery_reward_from(action_idx: int, rel_idx: int) -> float:
        is_charge = (action_idx == 0)
        if rel_idx == 0:  # positive diff → Charge
            return 1.0 if is_charge else -1.0
        else:             # near/negative → No charge
            return -1.0 if is_charge else 1.0

    def _combined_reward(self, action_idx: int):
        # Only include a component if its needed series exist; otherwise give it 0 weight
        if self.solar is not None and self.heatpump is not None:
            rel_svh = self._rel_solar_vs_heatpump()
            r_svh   = float(REWARD_MATRIX[action_idx][rel_svh])
        else:
            rel_svh, r_svh = 1, 0.0

        if self.baseload is not None and self.solar is not None and self.heatpump is not None:
            rel_bln = self._rel_baseload_vs_net()
            r_bln   = float(REWARD_MATRIX[action_idx][rel_bln])
        else:
            rel_bln, r_bln = 1, 0.0

        if self.battery is not None and self.baseload is not None and self.solar is not None:
            rel_bat = self._rel_battery_vs_solarbaseload()
            r_bat   = float(self._battery_reward_from(action_idx, rel_bat))
        else:
            rel_bat, r_bat = 1, 0.0

        if self.temperature is not None:
            rel_tmp = self._rel_temperature_vs_thr()
            r_tmp   = float(REWARD_MATRIX[action_idx][rel_tmp])
        else:
            rel_tmp, r_tmp = 1, 0.0

        reward = float(self._w_svh * r_svh + self._w_bln * r_bln + self._w_bat * r_bat + self._w_tmp * r_tmp)
        return reward, rel_svh, rel_bln, rel_bat, rel_tmp, r_svh, r_bln, r_bat, r_tmp

    def _quantise_action(self, a: float) -> Tuple[int, float]:
        if a >  0.33: return 0,  1.0
        if a < -0.33: return 2, -1.0
        return 1, 0.0

    @property
    def state_dim(self) -> int:
        # flex history (w) + one-hot relation (3) + one-hot last action (3)
        return self.w + 3 + 3

    @property
    def action_dim(self) -> int:
        return 1

    def reset(self):
        self.t = 0
        self.last_action_idx = 1
        return self._state()

    def _state(self) -> np.ndarray:
        # history over the CHOSEN training signal (self.flex)
        idx = max(0, self.t - self.w + 1)
        pad_len = self.w - (self.t - idx + 1)
        pad = [self.flex[0]] * max(0, pad_len)
        hist = pad + self.flex[idx : self.t + 1].tolist()

        # 'none' mode does not encode any ontology relation signal
        if self.ontology == "none":
            rel_state = np.zeros(3, dtype=np.float32)
        else:
            rel_state = np.zeros(3, dtype=np.float32)
            rel_state[self._categorise_relation()] = 1.0

        action_state = np.zeros(3, dtype=np.float32)
        action_state[self.last_action_idx] = 1.0

        return np.asarray(hist + rel_state.tolist() + action_state.tolist(), dtype=np.float32)

    def step(self, a: np.ndarray):
        a_scalar = float(np.array(a).astype(np.float32).squeeze())
        a_val: float = float(np.clip(a_scalar, -1.0, 1.0))
        act_idx, _ = self._quantise_action(a_val)

        # Ontology reward
        if self.ontology == "combined":
            (ontology_reward, rel_svh, rel_bln, rel_bat, rel_tmp,
             r_svh, r_bln, r_bat, r_tmp) = self._combined_reward(act_idx)
            if self.baseload   is not None and self.solar is not None and self.heatpump is not None: rel_idx = rel_bln
            elif self.temperature is not None: rel_idx = rel_tmp
            elif self.battery  is not None and self.solar is not None: rel_idx = rel_bat
            elif self.solar is not None and self.heatpump is not None: rel_idx = rel_svh
            else: rel_idx = 1
        elif self.ontology == "baseload_vs_net":
            rel_idx = self._rel_baseload_vs_net()
            ontology_reward = float(REWARD_MATRIX[act_idx, rel_idx]) if (self.baseload is not None and self.solar is not None and self.heatpump is not None) else 0.0
        elif self.ontology == "battery_vs_solarbaseload":
            rel_idx = self._rel_battery_vs_solarbaseload()
            ontology_reward = float(self._battery_reward_from(act_idx, rel_idx)) if (self.battery is not None and self.baseload is not None and self.solar is not None) else 0.0
        elif self.ontology == "temperature_vs_thr":
            rel_idx = self._rel_temperature_vs_thr()
            ontology_reward = float(REWARD_MATRIX[act_idx, rel_idx]) if self.temperature is not None else 0.0
        elif self.ontology == "solar_vs_heatpump":
            rel_idx = self._rel_solar_vs_heatpump()
            ontology_reward = float(REWARD_MATRIX[act_idx, rel_idx]) if (self.solar is not None and self.heatpump is not None) else 0.0
        elif self.ontology == "none":
            rel_idx = 1
            ontology_reward = 0.0
        else:
            print("[ERROR] Incorrect Ontology"); rel_idx = 1; ontology_reward = 0.0

        # Shaping against chosen training signal
        pred_flex = 0.5 * (a_val + 1.0) * self.max_flex
        target = float(self.flex[self.t])
        shaping = -abs(target - pred_flex) / (self.max_flex + self._eps)

        # Do not include shaping reward in ontology
        ontologies_no_shaping = {
            "combined",
            "solar_vs_heatpump",
            "baseload_vs_net",
            "battery_vs_solarbaseload",
            "temperature_vs_thr",
        }

        if self.ontology in ontologies_no_shaping:
            reward = float(ontology_reward)
        elif self.ontology == "none":
            reward = float(ontology_reward + shaping)
        else:
            raise ValueError(f"[ERROR] Incorrect Ontology: {self.ontology!r}")

        # advance
        self.t += 1
        done = self.t >= (len(self.flex) - 1)
        self.last_action_idx = act_idx

        info = {
            "target": target,
            "pred": float(pred_flex),
            "abs_err": float(abs(target - pred_flex)),
            "ontology_reward": float(ontology_reward),
            "shaping": float(shaping),
            "act_idx": int(act_idx),
            "rel_idx": int(rel_idx),
        }
        if self.ontology == "combined":
            info.update({
                "rel_idx_svh": int(rel_svh),
                "rel_idx_bln": int(rel_bln),
                "rel_idx_bat": int(rel_bat),
                "rel_idx_tmp": int(rel_tmp),
                "reward_svh": float(r_svh),
                "reward_bln": float(r_bln),
                "reward_bat": float(r_bat),
                "reward_tmp": float(r_tmp),
                "weights": (float(self._w_svh), float(self._w_bln), float(self._w_bat), float(self._w_tmp)),
                "reward_combined": float(ontology_reward),
            })
        return self._state(), float(reward), bool(done), info

# ────────────────────────────────────────────────
# ▶ DDPG agent
# ────────────────────────────────────────────────
class DDPGAgent:
    def __init__(self, state_dim, action_dim):
        self.actor = Actor(state_dim, action_dim).to(device)
        self.actor_target = Actor(state_dim, action_dim).to(device)
        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.optim_actor = optim.Adam(self.actor.parameters(), lr=LR_ACTOR)
        self.optim_critic = optim.Adam(self.critic.parameters(), lr=LR_CRITIC)

        self.noise = OUNoise(action_dim)
        self.memory = ReplayBuffer(BUFFER_SIZE)

    def act(self, state, add_noise=True):
        state_t = torch.tensor(state, dtype=torch.float32,
                               device=device).unsqueeze(0)
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state_t).cpu().data.numpy()[0]
        self.actor.train()
        if add_noise:
            action += self.noise.sample()
        return np.clip(action, -1, 1)

    def step(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)
        if len(self.memory) >= WARMUP_STEPS:
            self.learn()

    def learn(self):
        transitions = self.memory.sample(BATCH_SIZE)
        b = lambda x: torch.tensor(np.vstack(x), dtype=torch.float32,
                                   device=device)

        s = b(transitions.state)
        a = b(transitions.action)
        r = b(transitions.reward)
        ns = b(transitions.next_state)
        d = b(transitions.done)

        # Critic update
        with torch.no_grad():
            next_actions = self.actor_target(ns)
            q_targets = r + (1 - d) * GAMMA * \
                        self.critic_target(ns, next_actions)
        q_expected = self.critic(s, a)
        critic_loss = F.mse_loss(q_expected, q_targets)
        self.optim_critic.zero_grad(); critic_loss.backward()
        self.optim_critic.step()

        # Actor update
        actor_loss = -self.critic(s, self.actor(s)).mean()
        self.optim_actor.zero_grad(); actor_loss.backward()
        self.optim_actor.step()

        # Soft target update
        self._soft_update(self.actor,  self.actor_target)
        self._soft_update(self.critic, self.critic_target)

    @staticmethod
    def _soft_update(local, target):
        for t, l in zip(target.parameters(), local.parameters()):
            t.data.copy_(TAU * l.data + (1.0 - TAU) * t.data)

# ────────────────────────────────────────────────
# ▶ Diagnostics: per-step traces & plots
# ────────────────────────────────────────────────
def evaluate_and_plot(agent: 'DDPGAgent', env: SolarEnv, out_dir: str, episodes: int = 1, tag: str = "val"):
    os.makedirs(out_dir, exist_ok=True)
    all_records = []
    with torch.no_grad():
        for _ in range(episodes):
            s = env.reset()
            done = False
            step = 0
            while not done:
                a = agent.act(s, add_noise=False)
                s, _, done, info = env.step(a)
                rec = dict(step=step, **info)
                all_records.append(rec)
                step += 1
    if not all_records:
        return None

    df = pd.DataFrame(all_records)
    csv_path = Path(out_dir) / f"diagnostics_{tag}.csv"
    df.to_csv(csv_path, index=False)

    # 1) Target vs Pred
    plt.figure(figsize=(10,4))
    plt.plot(df['target'].values, label='target')
    plt.plot(df['pred'].values, label='pred')
    plt.title('Target vs Predicted Flex Power')
    plt.xlabel('Step'); plt.ylabel('Flex Power'); plt.legend()
    plt.tight_layout(); plt.savefig(Path(out_dir) / f"target_vs_pred_{tag}.png", dpi=150); plt.close()

    # 2) Absolute error
    plt.figure(figsize=(10,3))
    plt.plot(df['abs_err'].values)
    plt.title('Absolute Error per Step')
    plt.xlabel('Step'); plt.ylabel('Abs Error')
    plt.tight_layout(); plt.savefig(Path(out_dir) / f"abs_err_{tag}.png", dpi=150); plt.close()

    # 3) Ontology reward 
    plt.figure(figsize=(10,3))
    plt.plot(df['ontology_reward'].values)
    ttl = 'Ontology Reward per Step'
    if getattr(env, 'ontology', 'solar_vs_heatpump') == 'combined':
        ttl += ' (combined)'
    plt.title(ttl)
    plt.xlabel('Step'); plt.ylabel('Reward')
    plt.tight_layout(); plt.savefig(Path(out_dir) / f"ontology_reward_{tag}.png", dpi=150); plt.close()

    # 3b) If combined, also plot components
    has_svh = 'reward_svh' in df.columns
    has_bln = 'reward_bln' in df.columns
    has_bat = 'reward_bat' in df.columns
    has_tmp = 'reward_tmp' in df.columns
    if has_svh or has_bln or has_bat or has_tmp:
        plt.figure(figsize=(10,3))
        if has_svh: plt.plot(df['reward_svh'].values, label='solar_vs_heatpump')
        if has_bln: plt.plot(df['reward_bln'].values, label='baseload_vs_net')
        if has_bat: plt.plot(df['reward_bat'].values, label='battery_vs_solarbaseload')
        if has_tmp: plt.plot(df['reward_tmp'].values, label='temperature_vs_thr')
        plt.plot(df['ontology_reward'].values, label='weighted total')
        plt.title('Ontology Reward Components (combined)')
        plt.xlabel('Step'); plt.ylabel('Reward'); plt.legend()
        plt.tight_layout(); plt.savefig(Path(out_dir) / f"ontology_reward_components_{tag}.png", dpi=150); plt.close()

    # 4) Shaping reward
    plt.figure(figsize=(10,3))
    plt.plot(df['shaping'].values)
    plt.title('Shaping Reward per Step')
    plt.xlabel('Step'); plt.ylabel('Reward')
    plt.tight_layout(); plt.savefig(Path(out_dir) / f"shaping_{tag}.png", dpi=150); plt.close()

    # 5) Counts heatmap of (act_idx vs relation_idx)
    cm = np.zeros((3,3), dtype=int)
    for a,t in zip(df['act_idx'], df['rel_idx']):
        if 0 <= a < 3 and 0 <= t < 3:
            cm[a,t] += 1
    plt.figure(figsize=(4,4))
    plt.imshow(cm, aspect='equal')
    plt.title('Counts: action_idx (rows) vs relation_idx (cols)')
    ont = getattr(env, 'ontology', 'solar_vs_heatpump')
    if ont in ('baseload_vs_net','battery_vs_solarbaseload','temperature_vs_thr','combined'):
        xlab = 'relation_idx (0=pos diff, 1=equal, 2=neg diff)'
    elif ont == 'none':
        xlab = 'relation_idx (unused in no-ontology mode)'
    else:
        xlab = 'relation_idx (0=solar>heatpump, 1=equal, 2=solar<heatpump)'
    plt.xlabel(xlab)
    plt.ylabel('act_idx (0=heat/charge,1=abstain,2=cool/no-charge)')
    for i in range(3):
        for j in range(3):
            plt.text(j, i, str(cm[i,j]), ha='center', va='center')
    plt.tight_layout(); plt.savefig(Path(out_dir) / f"act_vs_relation_counts_{tag}.png", dpi=150); plt.close()

    return str(csv_path)


# ────────────────────────────────────────────────
# ▶ Training 
# ────────────────────────────────────────────────
def train_ddpg(
    solar_panel_flex_power_ids: list[str | None],
    heatpump_flex_power_ids:   list[str | None],
    train_dir: str,
    test_dir:  str | None = None,
    episodes:  int = MAX_EPISODES,
    save_models: bool = True,
    # Ontology & inputs
    ontology: str = "combined",                         # "solar_vs_heatpump" | "baseload_vs_net" | "battery_vs_solarbaseload" | "temperature_vs_thr" | "combined" | "none"
    training_input: str = "combined",                   # "solar" | "heatpump" | "combined"
    baseload_train_csv: str | Path | np.ndarray | None = None,
    battery_train_csv:  str | Path | np.ndarray | None = None,
    temperature_train_csv: str | Path | np.ndarray | None = None,
    baseload_test_csv:  str | Path | np.ndarray | None = None,
    battery_test_csv:   str | Path | np.ndarray | None = None,
    temperature_test_csv: str | Path | np.ndarray | None = None,
    combine_weights:    tuple[float, ...] = (1.0, 1.0, 1.0, 1.0),
) -> None:

    global_start = time.perf_counter()
    ti = _norm_training_input(training_input)

    run_label = f"DDPG_{ti}_Input__{ontology}_Ontology"

    RUN_DIR = RESULTS_ROOT / run_label
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    RUN_DIAG_DIR   = RUN_DIR / "Diagnostics"
    RUN_DIAG_DIR.mkdir(parents=True, exist_ok=True)

    # Rewire logger to this run's training_log.txt
    for h in list(log.handlers):
        if isinstance(h, logging.FileHandler):
            try: h.close()
            except Exception: pass
            log.removeHandler(h)
    fh = logging.FileHandler(RUN_DIR / "training_log.txt", mode="w", encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    log.addHandler(fh)

    # Derive default CSVs from TRAIN_DIR/TEST_DIR
    def _defaults(base: str) -> tuple[Path, Path, Path]:
        base = Path(base)
        bln  = base / "BASELOAD-PREDICTION_6b622d70-2119-11ef-a9db-8376e4779933.csv"
        bat  = base / "CHARGE_744ee040-2119-11ef-a9db-8376e4779933.csv"
        tmp  = base / "TEMPERATURE-PREDICTION_6b41ad22-2119-11ef-a9db-8376e4779933.csv"
        return bln, bat, tmp

    if baseload_train_csv is None or battery_train_csv is None or temperature_train_csv is None:
        bln_d, bat_d, tmp_d = _defaults(train_dir)
        baseload_train_csv    = bln_d if baseload_train_csv    is None else baseload_train_csv
        battery_train_csv     = bat_d if battery_train_csv     is None else battery_train_csv
        temperature_train_csv = tmp_d if temperature_train_csv is None else temperature_train_csv

    if test_dir:
        if baseload_test_csv is None or battery_test_csv is None or temperature_test_csv is None:
            bln_d, bat_d, tmp_d = _defaults(test_dir)
            baseload_test_csv    = bln_d if baseload_test_csv    is None else baseload_test_csv
            battery_test_csv     = bat_d if battery_test_csv     is None else battery_test_csv
            temperature_test_csv = tmp_d if temperature_test_csv is None else temperature_test_csv

    def _warn_missing(p, label):
        try_p = Path(p) if isinstance(p, (str, Path)) else None
        if try_p and not try_p.exists():
            msg = f"[WARN] {label} CSV not found: {try_p}"
            log.warning(msg)

    # Warn if files are missing 
    if ontology in ("baseload_vs_net", "battery_vs_solarbaseload", "temperature_vs_thr", "combined"):
        _warn_missing(baseload_train_csv, "TRAIN baseload")
        if test_dir: _warn_missing(baseload_test_csv, "TEST baseload")
    if ontology in ("battery_vs_solarbaseload", "combined"):
        _warn_missing(battery_train_csv, "TRAIN battery")
        if test_dir: _warn_missing(battery_test_csv, "TEST battery")
    if ontology in ("temperature_vs_thr", "combined"):
        _warn_missing(temperature_train_csv, "TRAIN temperature")
        if test_dir: _warn_missing(temperature_test_csv, "TEST temperature")

    # Aggregate FLEX series across all IDs → system series 
    def _aggregate_flex(ids: list[str | None], base_dir: str, label: str) -> np.ndarray | None:
        series = []
        print(f"[AGG] {label}:")
        for flex_id in ids:
            if not flex_id: continue
            csv_path = Path(base_dir) / f"{flex_id}.csv"
            if not csv_path.exists():
                msg = f"[WARN] missing FLEX CSV: {csv_path}"
                log.warning(msg); continue
            arr = SolarEnv._load_flex_secondcol_midabs(str(csv_path))
            if arr.size == 0:
                msg = f"[WARN] empty FLEX CSV: {csv_path}"
                log.warning(msg); continue
            series.append(arr.astype(np.float32))
            print(f"  + {csv_path}  len={len(arr)}")
        if not series:
            print(f"[AGG] {label}: no series found."); return None
        min_len = min(len(s) for s in series)
        stacked = np.stack([s[:min_len] for s in series], axis=0)
        system = stacked.sum(axis=0)
        print(f"[AGG] {label}: K={len(series)}  T={len(system)} (aligned by min len {min_len})")
        return system.astype(np.float32)

    # Decide which series we need
    ont = str(ontology).lower()
    need_solar_for_ont = ont in ("solar_vs_heatpump", "baseload_vs_net", "battery_vs_solarbaseload", "combined")
    need_heatpump_for_ont  = ont in ("solar_vs_heatpump", "baseload_vs_net", "combined")
    need_solar_for_inp = (ti in ("solar", "combined"))
    need_heatpump_for_inp  = (ti in ("heatpump",  "combined"))

    need_solar = need_solar_for_ont or need_solar_for_inp
    need_heatpump  = need_heatpump_for_ont  or need_heatpump_for_inp

    solar_system_train = _aggregate_flex(solar_panel_flex_power_ids, train_dir, label="SOLAR train")     if need_solar else None
    heatpump_system_train  = _aggregate_flex(heatpump_flex_power_ids,   train_dir, label="HEAT-PUMP train") if need_heatpump  else None

    if need_solar and solar_system_train is None:
        print("[ERROR] No SOLAR FLEX series found to aggregate but required."); return
    if need_heatpump and heatpump_system_train is None:
        print("[ERROR] No HEAT-PUMP FLEX series found to aggregate but required."); return

    # TRAIN environment
    env = SolarEnv(
        solar_system_train,
        heatpump_system_train, 
        ontology=ontology,
        training_input=ti,
        baseload_csv=baseload_train_csv,
        battery_csv=battery_train_csv,
        temperature_csv=temperature_train_csv,
        combine_weights=combine_weights,
        temp_threshold=23.0,
    )

    agent = DDPGAgent(env.state_dim, env.action_dim)

    returns_history: list[float] = []
    episode_metrics_rows: list[dict] = []
    start_ts = time.perf_counter()

    # Train
    for ep in range(episodes):
        s, ep_ret = env.reset(), 0.0
        ep_shared = 0.0
        steps = 0
        agent.noise.reset()

        for _ in range(MAX_STEPS):
            a = agent.act(s)  # noisy action
            ns, r, d, info = env.step(a)

            agent.step(s, a, r, ns, float(d))

            ep_ret += r
            ep_shared += float(info.get("shaping", 0.0))
            steps += 1
            s = ns

            if d:
                break

        returns_history.append(ep_ret)

        # Deterministic shared-eval snapshot
        eval_shared_return = np.nan
        eval_shared_per_step = np.nan
        eval_steps = 0
        if (ep + 1) % EVAL_EVERY == 0:
            ev = rollout_shared_metrics(env, lambda st: agent.act(st, add_noise=False), max_steps=MAX_STEPS)
            eval_shared_return = ev["shared_return"]
            eval_shared_per_step = ev["shared_per_step"]
            eval_steps = ev["steps"]

        episode_metrics_rows.append({
            "episode": ep + 1,
            "train_return": float(ep_ret),  # NOT comparable across ontology vs none
            "train_shared_return": float(ep_shared),  # comparable
            "train_shared_per_step": float(ep_shared / max(1, steps)),  # comparable
            "train_steps": int(steps),
            "eval_shared_return": eval_shared_return,
            "eval_shared_per_step": eval_shared_per_step,
            "eval_steps": int(eval_steps),
        })

        returns_history.append(ep_ret)

        if (ep + 1) % 10 == 0:
            log.info(
                f"{run_label} | Episode {ep+1:3d} | Return: {ep_ret:8.3f} | "
                f"shared/step: {ep_shared/max(1,steps): .4f}"
            )

    pd.DataFrame(episode_metrics_rows).to_csv(RUN_DIR / "episode_metrics.csv", index=False)
    log.info(f"{run_label} | wrote {RUN_DIR / 'episode_metrics.csv'}")

    # Save model
    tag = run_label
    if save_models:
        model_path = RUN_DIR / f"{tag}.pt"
        torch.save({"actor":  agent.actor.state_dict(),
                    "critic": agent.critic.state_dict()}, model_path)
        log.info(f"[SAVED] {model_path}")

    # TEST env 
    test_mean = test_std = None
    if test_dir:
        solar_system_test = _aggregate_flex(solar_panel_flex_power_ids, test_dir, label="SOLAR test")     if need_solar else None
        heatpump_system_test  = _aggregate_flex(heatpump_flex_power_ids,   test_dir, label="HEAT-PUMP test") if need_heatpump  else None

        if (need_solar and solar_system_test is None) or (need_heatpump and heatpump_system_test is None):
            test_env = None
            print("[WARN] Skipping TEST: required series missing for test set.")
        else:
            test_env = SolarEnv(
                solar_system_test,
                heatpump_system_test,
                ontology=ontology,
                training_input=ti,    
                baseload_csv=baseload_test_csv,
                battery_csv=battery_test_csv,
                temperature_csv=temperature_test_csv,
                combine_weights=combine_weights,
                temp_threshold=23.0,
            )

        if test_env is not None:
            ep_returns = []
            with torch.no_grad():
                for _ in range(30):
                    s, ret = test_env.reset(), 0.0
                    for _ in range(MAX_STEPS):
                        a = agent.act(s, add_noise=False)   # no exploration
                        s, r, d, _ = test_env.step(a)
                        ret += r
                        if d: break
                    ep_returns.append(ret)
            test_mean = float(np.mean(ep_returns))
            test_std  = float(np.std(ep_returns))
            log.info(f"{run_label} | TEST  μ={test_mean:.3f} σ={test_std:.3f}")

    opt_env = test_env if ("test_env" in locals() and test_env is not None) else env
    opt = rollout_optimality_metrics(opt_env, lambda st: agent.act(st, add_noise=False), max_steps=MAX_STEPS)

    opt_file = RUN_DIR / ("shared_eval_test.csv" if ("test_env" in locals() and test_env is not None) else "shared_eval_train.csv")
    pd.DataFrame([opt]).to_csv(opt_file, index=False)

    log.info(
        f"{run_label} | SHARED_OPT | policy/step={opt['policy_shared_per_step']:.4f} | "
        f"baseline/step={opt['baseline_shared_per_step']:.4f} | oracle/step={opt['oracle_shared_per_step']:.4f} | "
        f"ratio={opt['optimality_ratio']:.4f} | gap={opt['optimality_gap_per_step']:.4f}"
    )

    # Plot curve
    ep_axis = np.arange(1, len(returns_history) + 1)

    plt.figure(figsize=(8, 4))
    plt.scatter(ep_axis, returns_history, s=10, c="black", label="Episode return")

    k = 10
    if len(returns_history) >= k:
        run = np.convolve(np.array(returns_history, dtype=float), np.ones(k) / k, mode="valid")
        plt.plot(ep_axis[k-1:], run, lw=2, label=f"Sliding mean k={k}")

    plt.title(f"{run_label}")
    plt.xlabel("Episode"); plt.ylabel("Reward")
    plt.grid(alpha=0.3); plt.legend(fontsize="small", ncol=2)
    plt.xlim(0, len(returns_history))

    out_base = RUN_DIR / f"curve_{tag}"
    plt.tight_layout(); plt.savefig(out_base.with_suffix(".png"), dpi=150)
    plt.close()
    log.info(f"[SAVED] {out_base}.png")

    # Diagnostics
    csv_path = evaluate_and_plot(agent, env, str(RUN_DIAG_DIR), episodes=1, tag="val")
    if csv_path:
        print("Saved per-step diagnostics to:", csv_path)

    # Timing
    elapsed = time.perf_counter() - start_ts
    log.info(f"[TIME] {run_label}  {elapsed:6.2f} s")

    total = time.perf_counter() - global_start
    m, s = divmod(total, 60)
    msg = f"\nTraining finished in {int(m):02d} min {s:04.1f} s"
    log.info(msg)

# ────────────────────────────────────────────────
# ▶ CLI helper 
# ────────────────────────────────────────────────
if __name__ == "__main__":
    # Ask the user for the directory path containing the simulation training csv files
    dir_str_train = input("Enter path to ddpg simulation data (train):\n> ").strip()
    TRAIN_DIR = Path(dir_str_train)

    if not TRAIN_DIR.exists():
        raise FileNotFoundError(f"Train data not found at: {TRAIN_DIR}")
   
    # Ask the user for the directory path containing the simulation test csv files
    dir_str_test = input("Enter path to ddpg simulation data (test):\n> ").strip()
    TEST_DIR = Path(dir_str_test)

    if not TEST_DIR.exists():
        raise FileNotFoundError(f"Test data not found at: {TEST_DIR}")

    solar_panel_flex_power_ids = [
        "FLEX-POWER_6ca004a0-2119-11ef-a9db-8376e4779933",
        "FLEX-POWER_6cb70f10-2119-11ef-a9db-8376e4779933",
        "FLEX-POWER_6ce59920-2119-11ef-a9db-8376e4779933",
        "FLEX-POWER_6d89ef70-2119-11ef-a9db-8376e4779933",
        "FLEX-POWER_6da20b50-2119-11ef-a9db-8376e4779933",
        "FLEX-POWER_6e448ce0-2119-11ef-a9db-8376e4779933",
        "FLEX-POWER_6f9501b0-2119-11ef-a9db-8376e4779933",
        "FLEX-POWER_71fe87a0-2119-11ef-a9db-8376e4779933",
        "FLEX-POWER_722d5fd0-2119-11ef-a9db-8376e4779933",
        "FLEX-POWER_728284b0-2119-11ef-a9db-8376e4779933",
    ]

    heatpump_flex_power_ids = [
	    "FLEX-POWER_6c04d610-2119-11ef-a9db-8376e4779933",
	    "FLEX-POWER_6c28d8d1-2119-11ef-a9db-8376e4779933",
	    "FLEX-POWER_6c77e330-2119-11ef-a9db-8376e4779933",
	    "FLEX-POWER_6cb95900-2119-11ef-a9db-8376e4779933",
	    "FLEX-POWER_6ccc92e1-2119-11ef-a9db-8376e4779933",
	    "FLEX-POWER_6d0bbec0-2119-11ef-a9db-8376e4779933",
	    "FLEX-POWER_7002b7a0-2119-11ef-a9db-8376e4779933",
	    "FLEX-POWER_72027f40-2119-11ef-a9db-8376e4779933",
	    "FLEX-POWER_72310950-2119-11ef-a9db-8376e4779933",
	    "FLEX-POWER_72862e30-2119-11ef-a9db-8376e4779933",
    ]

    # ------------------------------------------------------------
    # Sweep automation
    # ------------------------------------------------------------
    SWEEP_CONFIGS = [
        ("combined",  "combined"),
        #("combined",  "battery_vs_solarbaseload"),
        #("combined",  "baseload_vs_net"),
        ("combined",  "none"),
        #("combined",  "solar_vs_heatpump"),
        #("combined",  "temperature_vs_thr"),
        #("heatpump",  "combined"),
        #("heatpump",  "none"),
        #("solar",     "combined"),
        #("solar",     "none"),
    ]

    SKIP_COMPLETED = True
    # Choose what "done" means
    DONE_MARKER_NAME = "episode_metrics.csv"  # or f"{run_label}.pt" or "training_log.txt"

    N = DEFAULT_N_RUNS   # default = 5
    all_failures = []

    for run_number in range(1, N + 1):
        RESULTS_ROOT = get_results_root(run_number)

        print(f"\n\n========== FULL RUN {run_number}/{N} ==========")
        print(f"[RESULTS] {RESULTS_ROOT}")

        failures = []

        for training_input, ontology in SWEEP_CONFIGS:
            ti = _norm_training_input(training_input)
            run_label = f"DDPG_{ti}_Input__{ontology}_Ontology"
            run_dir = RESULTS_ROOT / run_label
            done_marker = run_dir / DONE_MARKER_NAME

            if SKIP_COMPLETED and done_marker.exists():
                print(f"[SKIP][run {run_number}] {run_label} (found {DONE_MARKER_NAME})")
                continue

            print(f"\n========== RUN {run_number}/{N}: {run_label} ==========")

            try:
                train_ddpg(
                    solar_panel_flex_power_ids,
                    heatpump_flex_power_ids,
                    TRAIN_DIR,
                    TEST_DIR,
                    ontology=ontology,
                    training_input=ti,
                    combine_weights=(1.0, 1.0, 1.0, 1.0),  # (svh, bln, battery, temperature)
                )
            except Exception as e:
                print(f"[FAIL][run {run_number}] {run_label}: {e}")
                failures.append((run_label, repr(e)))

        if failures:
            all_failures.append((run_number, failures))

    if all_failures:
        print("\nSome runs failed:")
        for run_number, failures in all_failures:
            for label, err in failures:
                print(f" - run {run_number} | {label}: {err}")
