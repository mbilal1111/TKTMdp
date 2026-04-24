from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
BASE_DIR = Path.cwd() / "reinforcement_learning"

DDPG_ROOTS = [BASE_DIR / "ddpg" / f"ddpg_results_{i}" for i in range(1, 6)]
PPO_ROOTS  = [BASE_DIR / "ppo"  / f"ppo_results_{i}"  for i in range(1, 6)]

OUT_DIR = BASE_DIR / "eval_5runs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_RUNS_CSV  = OUT_DIR / "shared_eval_runs.csv"   # per-run values
OUT_STATS_CSV = OUT_DIR / "shared_eval_stats.csv"  # mean/std across 5 runs

PLOT_DIR = OUT_DIR / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------
# Only evaluate these settings
# ---------------------------------------------------------------------
TARGET_INPUT = "combined"
TARGET_ONTOLOGIES = {"combined", "none"}

# ---------------------------------------------------------------------
# Evaluation settings 
# ---------------------------------------------------------------------
# Early phase performance 
EARLY_N = 10                # How accurate was the agent in the first EARLY_N episodes?

# Episodes to converge 
CONV_WINDOW      = 10       # Rolling average over the last CONV_WINDOW points
CONV_PATIENCE    = 10       # How many consecutive rolling-mean points must meet the convergergence condition 
CONV_TAIL_POINTS = 10       # Final plateau level 
CONV_TOL_ABS     = 0.001    # Absolute slack allowed below the plateau
CONV_TOL_FRAC    = 0.02     # Relative slack allowed below the plateau

# ---------------------------------------------------------------------
# Run-name parsing
# ---------------------------------------------------------------------
# Folder examples:
#   PPO_combined_Input__battery_vs_solarbaseload_Ontology
#   DDPG_solar_Input__none_Ontology
_RUN_RE = re.compile(r"^(PPO|DDPG)_(.+?)_Input__(.+?)_Ontology$", re.IGNORECASE)
_RUN_RE_NO_ALGO = re.compile(r"^(.+?)_Input__(.+?)_Ontology$", re.IGNORECASE)

def _parse_run_name(run_name: str) -> dict:
    """
    Extract algo, training_input, ontology from folder name when possible.
    """
    name = run_name.strip()

    m = _RUN_RE.match(name)
    if m:
        return {
            "algo": m.group(1).upper(),
            "training_input": m.group(2),
            "ontology": m.group(3),
        }

    # fallback: folder name missing algo prefix
    m2 = _RUN_RE_NO_ALGO.match(name)
    if m2:
        return {
            "algo": None,
            "training_input": m2.group(1),
            "ontology": m2.group(2),
        }

    return {"algo": None, "training_input": None, "ontology": None}


def _infer_algo_from_root(root: Path) -> str | None:
    s = str(root).lower()
    if "/ddpg/" in s or "\\ddpg\\" in s:
        return "DDPG"
    if "/ppo/" in s or "\\ppo\\" in s:
        return "PPO"
    return None


def _replicate_id_from_root(root: Path) -> int | None:
    m = re.search(r"(\d+)$", root.name)
    return int(m.group(1)) if m else None


def _is_target_setting(training_input: str | None, ontology: str | None) -> bool:
    if training_input is None or ontology is None:
        return False
    return (training_input.lower() == TARGET_INPUT) and (ontology.lower() in TARGET_ONTOLOGIES)

# ---------------------------------------------------------------------
# Convergence detection
# ---------------------------------------------------------------------
def episodes_to_convergence(
    ep_nums: np.ndarray,
    values: np.ndarray,
    window: int = CONV_WINDOW,
    patience: int = CONV_PATIENCE,
    tail_points: int = CONV_TAIL_POINTS,
    tol_abs: float = CONV_TOL_ABS,
    tol_frac: float = CONV_TOL_FRAC,
) -> float:
    """
    Convergence = first episode where rolling mean stays within tolerance of
    the final plateau for 'patience' consecutive points.

    This works with negative metrics (the shared shaping is negative; higher is better).
    """
    ep_nums = np.asarray(ep_nums, dtype=float)
    values  = np.asarray(values, dtype=float)

    # drop NaNs
    mask = ~np.isnan(values)
    ep_nums = ep_nums[mask]
    values  = values[mask]

    if len(values) < max(window + patience, window + tail_points):
        return float("nan")

    s = pd.Series(values, index=ep_nums)
    rm = s.rolling(window=window).mean().dropna()
    if len(rm) < tail_points:
        return float("nan")

    plateau = float(rm.tail(tail_points).mean())
    tol = max(tol_abs, tol_frac * abs(plateau))
    threshold = plateau - tol

    rm_vals = rm.values
    rm_eps  = rm.index.values

    for i in range(0, len(rm_vals) - patience + 1):
        block = rm_vals[i:i + patience]
        if np.all(block >= threshold):
            return float(rm_eps[i])

    return float("nan")

# ---------------------------------------------------------------------
# File collection
# ---------------------------------------------------------------------
def _collect_episode_metrics_files() -> list[tuple[Path, Path, int | None]]:
    """
    Returns list of tuples: (episode_metrics.csv path, root folder, replicate id)
    """
    items: list[tuple[Path, Path, int | None]] = []

    for root in (DDPG_ROOTS + PPO_ROOTS):
        if not root.exists():
            print(f"[WARN] Results folder not found: {root}")
            continue

        rep_id = _replicate_id_from_root(root)

        for f in root.rglob("episode_metrics.csv"):
            items.append((f, root, rep_id))

    return items

# ---------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------
def _boxplot_one_algo_two_ontologies(
    df: pd.DataFrame,
    algo: str,
    metric: str,
    outfile_png: Path,
):
    """
    Box+whisker for ONE algo (PPO or DDPG), comparing ontology=combined vs ontology=none.
    Overlays mean as a point and std as error bars.
    """
    algo_u = algo.upper()
    ont_order = ["combined", "none"]

    data: list[np.ndarray] = []
    labels: list[str] = []
    means: list[float] = []
    stds: list[float] = []

    for ont in ont_order:
        vals = (
            df.loc[
                (df["algo"].str.upper() == algo_u)
                & (df["training_input"].str.lower() == TARGET_INPUT)
                & (df["ontology"].str.lower() == ont),
                metric,
            ]
            .dropna()
            .to_numpy(dtype=float)
        )

        if len(vals) == 0:
            print(f"[WARN] No values for {algo_u}, ontology={ont}, metric={metric}")
            continue

        data.append(vals)
        labels.append(f"ontology={ont}")
        means.append(float(np.mean(vals)))
        stds.append(float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0)

    if not data:
        print(f"[WARN] Nothing to plot for algo={algo_u}, metric={metric}")
        return

    positions = np.arange(1, len(data) + 1)

    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.boxplot(data, positions=positions, widths=0.55, showfliers=True)

    ax.errorbar(
        positions,
        means,
        yerr=stds,
        fmt="o",          # mean point
        capsize=6,
        linestyle="none",
    )

    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_title(f"{algo_u} — {metric} (5 runs) | mean ± std overlaid")
    ax.set_ylabel(metric)

    fig.tight_layout()
    fig.savefig(outfile_png, dpi=300)
    plt.close(fig)

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    items = _collect_episode_metrics_files()
    if not items:
        raise SystemExit("No episode_metrics.csv files found under ddpg_results_1..5 / ppo_results_1..5")

    rows: list[dict] = []

    for metrics_csv, root, rep_id in items:
        run_dir = metrics_csv.parent
        run_name = run_dir.name

        meta = _parse_run_name(run_name)
        algo = meta["algo"] or _infer_algo_from_root(root)
        training_input = meta["training_input"]
        ontology = meta["ontology"]

        # Only keep combined input + (combined|none) ontology
        if not _is_target_setting(training_input, ontology):
            continue

        if algo is None:
            print(f"[SKIP] Could not determine algo for: {metrics_csv}")
            continue

        ep = pd.read_csv(metrics_csv)

        required = {"episode", "train_shared_return", "train_shared_per_step"}
        missing = required - set(ep.columns)
        if missing:
            print(f"[SKIP] {run_name}: missing columns in episode_metrics.csv: {sorted(missing)}")
            continue

        episodes = int(ep["episode"].max())

        train_shared_return = ep["train_shared_return"].to_numpy(dtype=float)
        train_shared_ps     = ep["train_shared_per_step"].to_numpy(dtype=float)

        # Early phase performance
        early_n = min(EARLY_N, len(train_shared_ps))
        early_phase = float(np.mean(train_shared_ps[:early_n])) if early_n > 0 else float("nan")

        # Cumulative reward
        cumulative_reward = float(np.sum(train_shared_return))

        # Episodes to convergence (prefer deterministic eval curve if present)
        conv_ep = float("nan")
        if "eval_shared_per_step" in ep.columns:
            eval_mask = ~ep["eval_shared_per_step"].isna()
            eval_eps  = ep.loc[eval_mask, "episode"].to_numpy(dtype=float)
            eval_vals = ep.loc[eval_mask, "eval_shared_per_step"].to_numpy(dtype=float)
            if len(eval_vals) >= (CONV_WINDOW + CONV_PATIENCE):
                conv_ep = episodes_to_convergence(
                    eval_eps, eval_vals,
                    window=CONV_WINDOW,
                    patience=CONV_PATIENCE,
                    tail_points=CONV_TAIL_POINTS,
                    tol_abs=CONV_TOL_ABS,
                    tol_frac=CONV_TOL_FRAC,
                )

        # Fallback if deterministic eval curve not available/too sparse
        if np.isnan(conv_ep):
            conv_ep = episodes_to_convergence(
                ep["episode"].to_numpy(dtype=float),
                train_shared_ps,
                window=CONV_WINDOW,
                patience=CONV_PATIENCE,
                tail_points=CONV_TAIL_POINTS,
                tol_abs=CONV_TOL_ABS,
                tol_frac=CONV_TOL_FRAC,
            )

        # Policy optimality (prefer test, fallback to train)
        opt_ratio = float("nan")
        policy_ps = float("nan")
        base_ps   = float("nan")
        oracle_ps = float("nan")
        gap_ps    = float("nan")

        opt_test  = run_dir / "shared_eval_test.csv"
        opt_train = run_dir / "shared_eval_train.csv"
        opt_file = opt_test if opt_test.exists() else (opt_train if opt_train.exists() else None)

        if opt_file is not None:
            opt = pd.read_csv(opt_file).iloc[0]
            opt_ratio = float(opt.get("optimality_ratio", np.nan))
            policy_ps = float(opt.get("policy_shared_per_step", np.nan))
            base_ps   = float(opt.get("baseline_shared_per_step", np.nan))
            oracle_ps = float(opt.get("oracle_shared_per_step", np.nan))
            gap_ps    = float(opt.get("optimality_gap_per_step", np.nan))

        rows.append({
            "results_root": root.name,
            "replicate": rep_id,
            "run_name": run_name,

            "algo": algo,
            "training_input": training_input,
            "ontology": ontology,

            # Evaluation metrics from Pauls paper:
            # https://ieeexplore.ieee.org/document/11180249
            "episodes_to_convergence": conv_ep,
            "cumulative_reward_shared": cumulative_reward,
            "early_phase_shared_per_step": early_phase,
            "policy_optimality_ratio": opt_ratio,

            # Extra context that helps debugging/interpretation
            "episodes": episodes,
            "final_train_shared_per_step": float(train_shared_ps[-1]) if len(train_shared_ps) else np.nan,
            "policy_shared_per_step": policy_ps,
            "baseline_shared_per_step": base_ps,
            "oracle_shared_per_step": oracle_ps,
            "optimality_gap_per_step": gap_ps,
        })

    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit(
            "No matching runs found for:\n"
            f"  training_input={TARGET_INPUT}\n"
            f"  ontology in {sorted(TARGET_ONTOLOGIES)}\n"
            "Check folder naming and that episode_metrics.csv exists in those runs."
        )

    # If you accidentally have >1 matching run per (algo, ontology, replicate),
    # keep the one with the most episodes (reasonable default) and warn
    group_key = ["algo", "training_input", "ontology", "replicate"]
    dup_counts = df.groupby(group_key).size()
    bad = dup_counts[dup_counts > 1]
    if not bad.empty:
        print("[WARN] Multiple matching runs found for some (algo,input,ontology,replicate).")
        print("       Keeping the row with the maximum 'episodes' per group.")
        df = (
            df.sort_values("episodes", ascending=False)
              .drop_duplicates(subset=group_key, keep="first")
              .sort_values(["algo", "training_input", "ontology", "replicate", "run_name"])
              .reset_index(drop=True)
        )
    else:
        df = df.sort_values(["algo", "training_input", "ontology", "replicate", "run_name"]).reset_index(drop=True)

    # Save per-run values
    df.to_csv(OUT_RUNS_CSV, index=False)
    print(f"[OK] Wrote per-run CSV: {OUT_RUNS_CSV}")

    # Compute mean/std across runs (grouped by algo + setting)
    metric_cols = [
        "episodes_to_convergence",
        "cumulative_reward_shared",
        "early_phase_shared_per_step",
        "policy_optimality_ratio",
        "final_train_shared_per_step",
        "policy_shared_per_step",
        "baseline_shared_per_step",
        "oracle_shared_per_step",
        "optimality_gap_per_step",
    ]

    stats = (
        df.groupby(["algo", "training_input", "ontology"])[metric_cols]
          .agg(["mean", "std", "count"])
    )
    stats.columns = [f"{m}_{agg}" for (m, agg) in stats.columns]
    stats = stats.reset_index()

    stats.to_csv(OUT_STATS_CSV, index=False)
    print(f"[OK] Wrote mean/std CSV: {OUT_STATS_CSV}")

    # Make separate plots for DDPG and PPO
    plot_metrics = [
        "episodes_to_convergence",
        "cumulative_reward_shared",
        "early_phase_shared_per_step",
        "policy_optimality_ratio",
    ]

    for algo in ["DDPG", "PPO"]:
        for metric in plot_metrics:
            out_png = PLOT_DIR / f"{algo.lower()}_box_{metric}.png"
            _boxplot_one_algo_two_ontologies(
                df=df,
                algo=algo,
                metric=metric,
                outfile_png=out_png,
            )
            print(f"[OK] Saved plot: {out_png}")

    print(f"[DONE] Plots are in: {PLOT_DIR}")

if __name__ == "__main__":
    main()
