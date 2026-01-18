import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tensorboard.backend.event_processing import event_accumulator

EVENT_PREFIX = "events.out.tfevents"


# ----------------------------
# TensorBoard loading helpers
# ----------------------------
def find_event_dirs(run_root: Path) -> List[Path]:
    event_dirs = set()
    for p in run_root.rglob("*"):
        if p.is_file() and p.name.startswith(EVENT_PREFIX):
            event_dirs.add(p.parent)
    return sorted(event_dirs)


def load_scalars(event_dir: Path) -> Tuple[List[str], event_accumulator.EventAccumulator]:
    ea = event_accumulator.EventAccumulator(
        str(event_dir),
        size_guidance={"scalars": 0},
    )
    ea.Reload()
    scalar_tags = ea.Tags().get("scalars", [])
    return scalar_tags, ea


def pick_tag(scalar_tags: List[str], preferred: str) -> Optional[str]:
    if preferred in scalar_tags:
        return preferred

    # Useful fallbacks
    candidates = []
    for t in scalar_tags:
        tl = t.lower()
        if "cumulative" in tl and "reward" in tl:
            candidates.append(t)
        elif "reward" in tl:
            candidates.append(t)

    # Prefer the most specific "cumulative reward" if present
    if candidates:
        # sort: cumulative+reward first, then shorter names
        candidates.sort(key=lambda x: (("cumulative" not in x.lower()), len(x)))
        return candidates[0]

    return None


def load_series_from_run(run_root: Path, preferred_tag: str) -> Tuple[str, str, np.ndarray, np.ndarray, str]:
    """
    Returns: (status, tag, steps, values, debug_info)
    status: OK / NO_EVENT_FILES / TAG_NOT_FOUND
    """
    event_dirs = find_event_dirs(run_root)
    if not event_dirs:
        return "NO_EVENT_FILES", "", np.array([]), np.array([]), ""

    # Try each event dir until we find a usable tag
    for d in event_dirs:
        scalar_tags, ea = load_scalars(d)
        tag = pick_tag(scalar_tags, preferred_tag)
        if tag is None:
            continue
        scalars = ea.Scalars(tag)
        if not scalars:
            continue

        # deduplicate by step (keep last)
        by_step: Dict[int, float] = {}
        for s in scalars:
            by_step[int(s.step)] = float(s.value)

        steps = np.array(sorted(by_step.keys()), dtype=np.int64)
        values = np.array([by_step[int(st)] for st in steps], dtype=np.float64)
        return "OK", tag, steps, values, str(d)

    # If we reach here: tag not found anywhere; return some tags for debugging
    # (from first event dir)
    scalar_tags, _ = load_scalars(event_dirs[0])
    debug = " | ".join(scalar_tags[:30])  # first 30 tags
    return "TAG_NOT_FOUND", "", np.array([]), np.array([]), debug


# ----------------------------
# Feature computations
# ----------------------------
def clip_to_budget(steps: np.ndarray, values: np.ndarray, budget: int) -> Tuple[np.ndarray, np.ndarray]:
    mask = steps <= budget
    return steps[mask], values[mask]


def reward_at_budget(steps: np.ndarray, values: np.ndarray, budget: int) -> float:
    """
    Interpolate reward at exact budget if possible, else last <= budget.
    """
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


def window_slice(steps: np.ndarray, values: np.ndarray, lo: int, hi: int) -> Tuple[np.ndarray, np.ndarray]:
    idx = np.where((steps >= lo) & (steps <= hi))[0]
    return steps[idx], values[idx]


def mean_and_slope_last_window(
    steps: np.ndarray,
    values: np.ndarray,
    budget: int,
    window: int,
    min_points: int = 6,
) -> Dict[str, float]:
    steps_b, vals_b = clip_to_budget(steps, values, budget)
    if len(steps_b) < 2:
        return {
            "reward_at_budget": float("nan"),
            "mean_last_window": float("nan"),
            "slope_last_window_per_1m": float("nan"),
            "n_points_window": 0,
        }

    lo = budget - window
    w_steps, w_vals = window_slice(steps_b, vals_b, lo, budget)

    # Fallback if too sparse: last min_points points
    if len(w_steps) < min_points:
        w_steps = steps_b[-min_points:]
        w_vals = vals_b[-min_points:]

    mean_w = float(np.mean(w_vals))

    if len(w_steps) >= 2 and np.any(w_steps != w_steps[0]):
        slope = float(np.polyfit(w_steps.astype(np.float64), w_vals.astype(np.float64), 1)[0])  # reward / step
        slope_per_1m = slope * 1e6
    else:
        slope_per_1m = float("nan")

    return {
        "reward_at_budget": float(reward_at_budget(steps, values, budget)),
        "mean_last_window": mean_w,
        "slope_last_window_per_1m": float(slope_per_1m),
        "n_points_window": int(len(w_steps)),
    }


def auc_features(
    steps: np.ndarray,
    values: np.ndarray,
    budget: int,
    baseline_window: int = 100_000,
) -> Dict[str, float]:
    """
    AUC up to budget, plus AUC above baseline (normalized by budget so it's in 'reward units').
    baseline = mean reward in first baseline_window steps (or first available points within it).
    """
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

    # baseline from early window
    early_hi = min(baseline_window, budget)
    e_steps, e_vals = window_slice(steps_b, vals_b, 0, early_hi)
    if len(e_vals) == 0:
        baseline = float(vals_b[0])
    else:
        baseline = float(np.mean(e_vals))

    # Ensure steps are strictly increasing for integration
    s = steps_b.astype(np.float64)
    v = vals_b.astype(np.float64)

    auc_raw = float(np.trapz(v, s))  # reward * step
    auc_raw_norm = auc_raw / float(budget)  # normalize -> average reward over [0,budget]

    v_adj = v - baseline
    auc_above = float(np.trapz(v_adj, s))
    auc_above_norm = auc_above / float(budget)  # average reward above baseline

    return {
        "reward_at_budget": float(reward_at_budget(steps, values, budget)),
        "baseline": float(baseline),
        "auc_raw_norm": float(auc_raw_norm),
        "auc_above_baseline_norm": float(auc_above_norm),
        "max_reward_to_budget": float(np.max(v)),
        "n_points": int(len(steps_b)),
    }


# ----------------------------
# Main processing
# ----------------------------
def process_folder(folder: Path, env: str, group: str, budget: int, window: int, tag: str) -> pd.DataFrame:
    rows = []
    for run_root in sorted([p for p in folder.iterdir() if p.is_dir()]):
        run_id = run_root.name
        status, used_tag, steps, values, debug_info = load_series_from_run(run_root, tag)

        base = {
            "run_id": run_id,
            "env": env,
            "group": group,
            "status": status,
            "tag_used": used_tag,
        }

        if status != "OK":
            base["debug"] = debug_info
            rows.append(base)
            continue

        base["last_logged_step"] = int(steps.max()) if len(steps) else None
        base["budget"] = budget

        if env.lower() == "worm":
            feats = mean_and_slope_last_window(steps, values, budget=budget, window=window)
            rows.append({**base, **feats, "window": window})
        elif env.lower() == "pyramids":
            feats = auc_features(steps, values, budget=budget, baseline_window=100_000)
            rows.append({**base, **feats})
        else:
            raise ValueError("env must be 'worm' or 'pyramids'")

    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Folder with run subfolders")
    ap.add_argument("--env", required=True, choices=["worm", "pyramids"])
    ap.add_argument("--group", default="phase2_passing")
    ap.add_argument("--tag", default="Environment/Cumulative Reward")

    ap.add_argument("--budget", type=int, default=None, help="Override budget (worm default 2M, pyramids default 1M)")
    ap.add_argument("--window", type=int, default=300_000, help="Worm: last-window size for mean/slope")

    ap.add_argument("--out", required=True, help="Output CSV path")
    args = ap.parse_args()

    inp = Path(args.input)
    env = args.env.lower()

    if args.budget is None:
        budget = 2_000_000 if env == "worm" else 1_000_000
    else:
        budget = args.budget

    df = process_folder(inp, env=env, group=args.group, budget=budget, window=args.window, tag=args.tag)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    print(f"Wrote: {args.out}")
    ok = df[df["status"] == "OK"]
    if len(ok):
        print("\nSummary (OK runs):")
        print(ok.describe(include="all").transpose()[["count", "mean", "std", "min", "max"]])


if __name__ == "__main__":
    main()
