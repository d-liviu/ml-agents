
# RUN (PowerShell, one line):
# python utils\phase3\predict_pyramids_phase3_by_auc_2m.py --phase3-runs-dir "D:\University\Project 2.1\repo\ml-agents\results\Pyramids_phase3" --out "D:\University\Project 2.1\pyramids_phase3_predictions_auc_2m.csv"
# --phase3-runs-dir "D:\...\results\Pyramids_phase3" - Path to the folder that contains all Phase 3 Pyramids runs as subfolders
# --out "D:\...\pyramids_phase3_predictions_auc_2m.csv" - Where to save the output CSV

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tensorboard.backend.event_processing import event_accumulator

EVENT_PREFIX = "events.out.tfevents"


PHASE2_P25_AUC_ABOVE_BASELINE_2M = 0.46153462451882654
PHASE2_P25_MAX_REWARD_2M = 0.8358616530895233

# Optional: a more lenient tier (p10) if you ever want "candidate" labels
PHASE2_P10_AUC_ABOVE_BASELINE_2M = 0.19938589417189354


def find_event_dirs(run_root: Path) -> List[Path]:
    event_dirs = set()
    for p in run_root.rglob("*"):
        if p.is_file() and p.name.startswith(EVENT_PREFIX):
            event_dirs.add(p.parent)
    return sorted(event_dirs)


def load_scalars(event_dir: Path) -> Tuple[List[str], event_accumulator.EventAccumulator]:
    ea = event_accumulator.EventAccumulator(str(event_dir), size_guidance={"scalars": 0})
    ea.Reload()
    scalar_tags = ea.Tags().get("scalars", [])
    return scalar_tags, ea


def pick_tag(scalar_tags: List[str], preferred: str) -> Optional[str]:
    if preferred in scalar_tags:
        return preferred
    candidates = [t for t in scalar_tags if "reward" in t.lower()]
    if not candidates:
        return None
    candidates.sort(key=lambda x: (("cumulative" not in x.lower()), len(x)))
    return candidates[0]


def load_series_from_run(run_root: Path, preferred_tag: str) -> Tuple[str, str, np.ndarray, np.ndarray, str]:
    event_dirs = find_event_dirs(run_root)
    if not event_dirs:
        return "NO_EVENT_FILES", "", np.array([]), np.array([]), ""

    for d in event_dirs:
        scalar_tags, ea = load_scalars(d)
        tag = pick_tag(scalar_tags, preferred_tag)
        if tag is None:
            continue
        scalars = ea.Scalars(tag)
        if not scalars:
            continue

        by_step: Dict[int, float] = {}
        for s in scalars:
            by_step[int(s.step)] = float(s.value)

        steps = np.array(sorted(by_step.keys()), dtype=np.int64)
        values = np.array([by_step[int(st)] for st in steps], dtype=np.float64)
        return "OK", tag, steps, values, str(d)

    # Debug: list tags from first event dir
    scalar_tags, _ = load_scalars(event_dirs[0])
    debug = " | ".join(scalar_tags[:30])
    return "TAG_NOT_FOUND", "", np.array([]), np.array([]), debug


def clip_to_budget(steps: np.ndarray, values: np.ndarray, budget: int) -> Tuple[np.ndarray, np.ndarray]:
    mask = steps <= budget
    return steps[mask], values[mask]


def reward_at_budget(steps: np.ndarray, values: np.ndarray, budget: int) -> float:
    if len(steps) == 0:
        return float("nan")

    idx = np.searchsorted(steps, budget, side="left")
    if idx < len(steps) and steps[idx] == budget:
        return float(values[idx])

    if 0 < idx < len(steps):
        s0, s1 = steps[idx - 1], steps[idx]
        v0, v1 = values[idx - 1], values[idx]
        if s1 != s0:
            alpha = (budget - s0) / (s1 - s0)
            return float(v0 + alpha * (v1 - v0))

    idx_before = np.searchsorted(steps, budget, side="right") - 1
    if idx_before >= 0:
        return float(values[idx_before])

    return float("nan")


def window_indices(steps: np.ndarray, lo: int, hi: int) -> np.ndarray:
    return np.where((steps >= lo) & (steps <= hi))[0]


def auc_features(steps: np.ndarray, values: np.ndarray, budget: int, baseline_window: int) -> Dict[str, float]:
    steps_b, vals_b = clip_to_budget(steps, values, budget)
    if len(steps_b) < 2:
        return {
            "reward_at_budget": float("nan"),
            "baseline": float("nan"),
            "auc_raw_norm": float("nan"),
            "auc_above_baseline_norm": float("nan"),
            "max_reward_to_budget": float("nan"),
            "n_points": int(len(steps_b)),
        }

    idx_early = window_indices(steps_b, 0, min(baseline_window, budget))
    baseline = float(np.mean(vals_b[idx_early])) if len(idx_early) else float(vals_b[0])

    s = steps_b.astype(np.float64)
    v = vals_b.astype(np.float64)

    auc_raw = float(np.trapz(v, s))
    auc_raw_norm = auc_raw / float(budget)

    v_adj = v - baseline
    auc_above = float(np.trapz(v_adj, s))
    auc_above_norm = auc_above / float(budget)

    return {
        "reward_at_budget": float(reward_at_budget(steps, values, budget)),
        "baseline": float(baseline),
        "auc_raw_norm": float(auc_raw_norm),
        "auc_above_baseline_norm": float(auc_above_norm),
        "max_reward_to_budget": float(np.max(v)),
        "n_points": int(len(steps_b)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase3-runs-dir", required=True, help="Folder of Phase-3 run subfolders (Pyramids)")
    ap.add_argument("--out", required=True, help="Output CSV with predictions")

    ap.add_argument("--tag", default="Environment/Cumulative Reward", help="TensorBoard scalar tag")
    ap.add_argument("--budget", type=int, default=2_000_000, help="Budget for screening (default 2,000,000)")
    ap.add_argument("--baseline-window", type=int, default=100_000, help="Early baseline window (default 100k)")

    ap.add_argument("--use-max", action="store_true",
                    help="Stricter: also require max_reward_to_budget >= Phase2 p25 max cutoff")
    ap.add_argument("--emit-tier", action="store_true",
                    help="Adds a 3-tier label: STRONG / CANDIDATE / UNLIKELY (p25/p10 cutoffs)")
    args = ap.parse_args()

    runs_dir = Path(args.phase3_runs_dir)
    if not runs_dir.exists():
        raise FileNotFoundError(f"phase3-runs-dir not found: {runs_dir}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    run_folders = [p for p in runs_dir.iterdir() if p.is_dir()]
    run_folders.sort(key=lambda p: p.name)

    rows = []
    for run_root in run_folders:
        run_id = run_root.name
        status, tag_used, steps, values, debug = load_series_from_run(run_root, args.tag)

        if status != "OK":
            rows.append({"run_id": run_id, "status": status, "tag_used": tag_used, "debug": debug})
            continue

        feats = auc_features(steps, values, budget=args.budget, baseline_window=args.baseline_window)

        # Primary rule (STRONG): match Phase-2 passers p25 AUC at 2M
        likely = feats["auc_above_baseline_norm"] >= PHASE2_P25_AUC_ABOVE_BASELINE_2M
        if args.use_max:
            likely = likely and (feats["max_reward_to_budget"] >= PHASE2_P25_MAX_REWARD_2M)

        # Optional 3-tier label
        tier = None
        if args.emit_tier:
            if feats["auc_above_baseline_norm"] >= PHASE2_P25_AUC_ABOVE_BASELINE_2M:
                tier = "STRONG"
            elif feats["auc_above_baseline_norm"] >= PHASE2_P10_AUC_ABOVE_BASELINE_2M:
                tier = "CANDIDATE"
            else:
                tier = "UNLIKELY"

        score = feats["auc_above_baseline_norm"] + 0.1 * feats["max_reward_to_budget"]

        row = {
            "run_id": run_id,
            "status": "OK",
            "tag_used": tag_used,
            "event_dir": debug,
            "last_logged_step": int(steps.max()) if len(steps) else None,
            "budget": args.budget,
            "baseline_window": args.baseline_window,
            "phase2_auc_p25_cutoff_2m": PHASE2_P25_AUC_ABOVE_BASELINE_2M,
            "phase2_max_p25_cutoff_2m": PHASE2_P25_MAX_REWARD_2M,
            "likely_pass_like": bool(likely),
            "score": float(score),
            **feats,
        }
        if args.emit_tier:
            row["tier"] = tier

        rows.append(row)

    df = pd.DataFrame(rows)
    if "score" in df.columns:
        df = df.sort_values(["likely_pass_like", "score"], ascending=[False, False])

    df.to_csv(out_path, index=False)

    print(f"Wrote: {out_path}")
    print(f"Cutoff: auc_above_baseline_norm >= {PHASE2_P25_AUC_ABOVE_BASELINE_2M:.4f} (Phase2 p25 @2M)")
    if args.use_max:
        print(f" + max_reward_to_budget >= {PHASE2_P25_MAX_REWARD_2M:.4f} (Phase2 p25 @2M)")

    ok = df[df["status"] == "OK"]
    if len(ok):
        cols = [c for c in [
            "run_id", "auc_above_baseline_norm", "max_reward_to_budget",
            "likely_pass_like", "tier", "score"
        ] if c in ok.columns]
        print("\nTop 10:")
        print(ok[cols].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
