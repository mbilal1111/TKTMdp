
# TKTMdp

Reproducibility repository for **TKTMdp**, an ontology-grounded action admissibility mechanism for deep reinforcement learning (DRL) baseload optimisation in energy communities.

This repository contains the code and outputs used to reproduce the TKTMdp experiments for **PPO** and **DDPG** under ontology-aware and no-ontology settings.

## What this repository contains

- `scripts/` — main training scripts and utilities.
- `results/` — aggregated paper-level results already exported as CSV.
- `data/` — notes about the underlying data source.

The main entry points for reproducing the paper experiments are:

- `scripts/ppo_all_onotologies_system.py`
- `scripts/ddpg_all_ontologies_system.py`
- `scripts/state_explorer_main.py`

## Reproducing the paper results

### 1. Clone the repository

```bash
git clone https://github.com/mbilal1111/TKTMdp.git
cd TKTMdp
```

### 2. Create a Python environment

A Python 3.10+ environment is recommended.

Example using `venv`:

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows:

```powershell
python -m venv .venv
.venv\Scripts\activate
```

Install the libraries used by the training scripts:

```bash
pip install numpy pandas matplotlib torch
```

---

### 3. Download the prepared reinforcement learning data

Download the archive:

- `reinforcement_learning_data.zip` from Zenodo: `https://zenodo.org/records/19604833`

Unzip it somewhere memorable. In this README, that location is referred to as:

```text
<DATA_ROOT>/reinforcement_learning_data
```

After extraction, the paths used by the training scripts should be:

#### PPO

- Training data: `<DATA_ROOT>/reinforcement_learning_data/ppo/training_data/dynamics_timeseries`
- Test data: `<DATA_ROOT>/reinforcement_learning_data/ppo/test_data/dynamics_timeseries`

#### DDPG

- Training data: `<DATA_ROOT>/reinforcement_learning_data/ddpg/training_data/dynamics_timeseries`
- Test data: `<DATA_ROOT>/reinforcement_learning_data/ddpg/test_data/dynamics_timeseries`

---

### 4. Run PPO experiments

From the repository root, run:

```bash
python scripts/ppo_all_onotologies_system.py
```

When prompted, enter:

```text
<DATA_ROOT>/reinforcement_learning_data/ppo/training_data/dynamics_timeseries
<DATA_ROOT>/reinforcement_learning_data/ppo/test_data/dynamics_timeseries
```

By default, the PPO script runs the following sweep:

```python
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
```

That means the default paper-style comparison trains with:

- all ontology signals combined, and
- no ontology (`none`)

using the combined training input from the heatpump and solar panel. 

---

### 5. Run DDPG experiments

From the repository root, run:

```bash
python scripts/ddpg_all_ontologies_system.py
```

When prompted, enter:

```text
<DATA_ROOT>/reinforcement_learning_data/ddpg/training_data/dynamics_timeseries
<DATA_ROOT>/reinforcement_learning_data/ddpg/test_data/dynamics_timeseries
```

The DDPG script uses the same default sweep logic as PPO:

```python
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
```

---

## Ontologies used in the project

Four ontologies were considered in the project:

1. `solar_vs_heatpump`
2. `baseload_vs_net`
3. `battery_vs_solarbaseload`
4. `temperature_vs_thr`

The training scripts can also run with:

- `combined` — combines the enabled ontology rewards
- `none` — disables ontology reward shaping

### Running only one ontology

If you want to test the effect of a single ontology, edit `SWEEP_CONFIGS` in the relevant training script and uncomment only the configuration you want.

For example, to compare a single ontology against no ontology, you can switch from the default sweep to something like:

```python
SWEEP_CONFIGS = [
    ("combined", "solar_vs_heatpump"),
    ("combined", "none"),
]
```

You can also change the training input to `solar` or `heatpump` only, if that is the comparison you want to run.

---

## Number of repeated runs

By default, each machine learning script runs **5 times**:

```python
DEFAULT_N_RUNS = 5
```

You can change this directly in:

- `scripts/ppo_all_onotologies_system.py`
- `scripts/ddpg_all_ontologies_system.py`

If you change the number of runs, remember to update the evaluation script as well.

---

## Flex power IDs and `state_explorer_main.py`

The hard-coded `solar_panel_flex_power_ids` and `heatpump_flex_power_ids` used in the training scripts were obtained beforehand using `state_explorer_main.py`.

In normal use, you do **not** need to rerun that step unless the underlying identifiers or exported state files change.


## Output structure

### PPO outputs

The PPO script saves results under:

```text
reinforcement_learning/ppo/new_ppo_results_N/
```

where `N` is the run number.

A typical run directory looks like this:

```text
reinforcement_learning/
└── ppo/
    └── new_ppo_results_1/
        ├── PPO_combined_Input__combined_Ontology/
        │   ├── Diagnostics/
        │   │   ├── abs_err_val.png
        │   │   ├── act_vs_relation_counts_val.png
        │   │   ├── diagnostics_val.csv
        │   │   ├── ontology_reward_components_val.png
        │   │   ├── ontology_reward_val.png
        │   │   ├── shaping_val.png
        │   │   └── target_vs_pred_val.png
        │   ├── curve_PPO_combined_Input__combined_Ontology.png
        │   ├── episode_metrics.csv
        │   ├── PPO_combined_Input__combined_Ontology.pt
        │   ├── shared_eval_test.csv
        │   └── training_log.txt
        └── PPO_combined_Input__none_Ontology/
            └── ...
```

### DDPG outputs

The DDPG script saves results under:

```text
reinforcement_learning/ddpg/new_ddpg_results_N/
```

with the same layout pattern as PPO, but using `DDPG_...` run names.

---

## Evaluating the 5-run outputs

After all PPO and DDPG runs have finished, run:

```bash
python evaluation_5runs.py
```

### If you changed the number of runs

If you changed `DEFAULT_N_RUNS`, you must also update the ranges in `evaluation_5runs.py`.

The default configuration assumes 5 runs:

```python
DDPG_ROOTS = [BASE_DIR / "ddpg" / f"new_ddpg_results_{i}" for i in range(1, 6)]
PPO_ROOTS  = [BASE_DIR / "ppo"  / f"new_ppo_results_{i}"  for i in range(1, 6)]
```

For example, if you changed the training scripts to run 3 repetitions, update the ranges to `range(1, 4)`.

---

## Final evaluation output

The final evaluation results are saved under:

```text
eval_5runs/
```

The plots directory is expected to contain outputs like:

```text
eval_5runs/
└── plots/
    ├── box_cumulative_reward_shared.png
    ├── box_early_phase_shared_per_step.png
    ├── box_episodes_to_convergence.png
    ├── box_policy_optimality_ratio.png
    ├── ddpg_box_cumulative_reward_shared.png
    ├── ddpg_box_early_phase_shared_per_step.png
    ├── ddpg_box_episodes_to_convergence.png
    ├── ddpg_box_policy_optimality_ratio.png
    ├── ppo_box_cumulative_reward_shared.png
    ├── ppo_box_early_phase_shared_per_step.png
    ├── ppo_box_episodes_to_convergence.png
    └── ppo_box_policy_optimality_ratio.png
```

These plots summarise the multi-run comparison across PPO/DDPG and ontology/no-ontology settings.

---

## Existing aggregated results

The repository already includes an aggregated results file:

```text
results/5-run-drl.csv
```

This is useful if you want a quick view of the 5-run summary statistics without rerunning the full training pipeline.

---

## Citation

If you use this repository, cite the project using the metadata in `CITATION.cff`.
