import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

# Run the feature extractor script to produce a temporary CSV of features
# Then, use those features to predict performance at 5M steps for Worm Phase 3 runs.
# The prediction is based on mean and slope of reward near 2M steps.

def run_extractor(extract_script: Path, phase3_runs_dir: Path, tmp_csv: Path, tag: str,
                  budget: int, window: int) -> None:
    cmd = [
        sys.executable,
        str(extract_script),
        "--input", str(phase3_runs_dir),
        "--env", "worm",
        "--budget", str(budget),
        "--window", str(window),
        "--tag", tag,
        "--out", str(tmp_csv),
    ]
    print("Running extractor:")
    print("  " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--phase3-runs-dir", required=True,
                    help="Folder containing Phase 3 run subfolders (e.g. .../results/Worm_phase3)")

    ap.add_argument("--out", required=True, help="Output CSV with predictions")

    ap.add_argument("--extract-script", default=None,
                    help="Path to extract_phase2_features.py (default: same folder as this script)")

    ap.add_argument("--tag", default="Environment/Cumulative Reward",
                    help="TensorBoard scalar tag to use (default: Environment/Cumulative Reward)")
    ap.add_argument("--budget", type=int, default=2_000_000,
                    help="Phase 3 budget step (default: 2M)")
    ap.add_argument("--window", type=int, default=500_000,
                    help="Window size for mean/slope near budget (default: 500,000)")
    ap.add_argument("--keep-features-csv", action="store_true",
                    help="Keep the intermediate extracted features CSV next to output")

    ap.add_argument("--horizon", type=int, default=5_000_000,
                    help="Target step to predict at (default: 5,000,000)")
    ap.add_argument("--threshold", type=float, default=939.047,
                    help="Worm threshold to compare against")
    ap.add_argument("--slowdown", type=float, default=0.8,
                    help="Conservative factor applied to slope (default 0.8). Use 1.0 for optimistic prediction.")
    ap.add_argument("--require-positive-slope", action="store_true",
                    help="If set: mark UNLIKELY when slope <= 0")

    ap.add_argument("--mean-col", default="mean_last_window",
                    help="Column name for mean performance near budget")
    ap.add_argument("--slope-col", default="slope_last_window_per_1m",
                    help="Column name for slope (reward per 1M steps)")

    args = ap.parse_args()

    phase3_runs_dir = Path(args.phase3_runs_dir)
    if not phase3_runs_dir.exists():
        raise FileNotFoundError(f"phase3-runs-dir not found: {phase3_runs_dir}")

    script_dir = Path(__file__).resolve().parent
    extract_script = Path(args.extract_script) if args.extract_script else (script_dir / "extract_phase2_features.py")
    if not extract_script.exists():
        raise FileNotFoundError(
            f"Extractor script not found: {extract_script}\n"
            f"Pass --extract-script to point to it."
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.keep_features_csv:
        features_csv = out_path.with_suffix("").with_name(out_path.stem + "_features.csv")
    else:
        features_csv = Path(tempfile.gettempdir()) / f"worm_phase3_features_{out_path.stem}.csv"

    run_extractor(
        extract_script=extract_script,
        phase3_runs_dir=phase3_runs_dir,
        tmp_csv=features_csv,
        tag=args.tag,
        budget=args.budget,
        window=args.window,
    )

    df = pd.read_csv(features_csv)

    if "status" in df.columns:
        df_ok = df[df["status"] == "OK"].copy()
    else:
        df_ok = df.copy()

    for c in [args.mean_col, args.slope_col]:
        if c not in df_ok.columns:
            raise ValueError(f"Missing column '{c}' in extracted CSV. Available: {list(df_ok.columns)}")

    mean = df_ok[args.mean_col].astype(float)
    slope_per_1m = df_ok[args.slope_col].astype(float)

    delta_m = (args.horizon - args.budget) / 1_000_000.0

    df_ok["pred_reward_at_horizon"] = mean + slope_per_1m * delta_m
    df_ok["pred_reward_at_horizon_conservative"] = mean + (slope_per_1m * args.slowdown) * delta_m

    if args.require_positive_slope:
        df_ok["likely_by_horizon"] = (slope_per_1m > 0) & (df_ok["pred_reward_at_horizon"] >= args.threshold)
        df_ok["likely_by_horizon_conservative"] = (slope_per_1m > 0) & (
            df_ok["pred_reward_at_horizon_conservative"] >= args.threshold
        )
    else:
        df_ok["likely_by_horizon"] = df_ok["pred_reward_at_horizon"] >= args.threshold
        df_ok["likely_by_horizon_conservative"] = df_ok["pred_reward_at_horizon_conservative"] >= args.threshold

    df_ok["budget"] = args.budget
    df_ok["window"] = args.window
    df_ok["horizon"] = args.horizon
    df_ok["threshold"] = args.threshold
    df_ok["delta_millions_steps"] = delta_m
    df_ok["slowdown_used"] = args.slowdown
    df_ok["features_csv_used"] = str(features_csv)

    df_ok = df_ok.sort_values(
        ["likely_by_horizon_conservative", "pred_reward_at_horizon_conservative"],
        ascending=[False, False],
    )

    df_ok.to_csv(out_path, index=False)

    print(f"\nWrote predictions: {out_path}")
    if args.keep_features_csv:
        print(f"Kept intermediate features CSV: {features_csv}")
    else:
        print(f"Intermediate features CSV (temp): {features_csv}")

    print(f"\nSettings: budget={args.budget}, window={args.window}, horizon={args.horizon}, "
          f"threshold={args.threshold}, slowdown={args.slowdown}")

    cols = [c for c in [
        "run_id", args.mean_col, args.slope_col,
        "pred_reward_at_horizon", "pred_reward_at_horizon_conservative",
        "likely_by_horizon", "likely_by_horizon_conservative"
    ] if c in df_ok.columns]

    print("\nTop 10 (by conservative prediction):")
    print(df_ok[cols].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
