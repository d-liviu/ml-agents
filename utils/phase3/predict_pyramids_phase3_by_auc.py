import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd


def run_extractor(extract_script: Path, phase3_runs_dir: Path, out_csv: Path,
                  tag: str, budget: int) -> None:
    cmd = [
        sys.executable,
        str(extract_script),
        "--input", str(phase3_runs_dir),
        "--env", "pyramids",
        "--budget", str(budget),
        "--tag", tag,
        "--out", str(out_csv),
    ]
    print("Running extractor:")
    print("  " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--phase3-runs-dir", required=True,
                    help="Folder containing Phase 3 Pyramids run subfolders (1M budget)")
    ap.add_argument("--phase2-reference-csv", required=True,
                    help="Phase 2 PASSING Pyramids AUC CSV (your reference set)")
    ap.add_argument("--out", required=True,
                    help="Output CSV with predictions")

    ap.add_argument("--extract-script", default=None,
                    help="Path to extract_phase2_features.py (default: same folder as this script)")
    ap.add_argument("--tag", default="Environment/Cumulative Reward",
                    help="TensorBoard scalar tag (default: Environment/Cumulative Reward)")
    ap.add_argument("--budget", type=int, default=1_000_000,
                    help="Budget step for Phase 3 Pyramids (default: 1,000,000)")

    ap.add_argument("--quantile", type=float, default=0.25,
                    help="Cutoff quantile from Phase2 passers (default: 0.25)")
    ap.add_argument("--use-max", action="store_true",
                    help="Also require max_reward_to_budget >= the same Phase2 quantile")

    ap.add_argument("--keep-features-csv", action="store_true",
                    help="Keep the intermediate extracted features CSV next to output")

    args = ap.parse_args()

    phase3_runs_dir = Path(args.phase3_runs_dir)
    if not phase3_runs_dir.exists():
        raise FileNotFoundError(f"phase3-runs-dir not found: {phase3_runs_dir}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    script_dir = Path(__file__).resolve().parent
    extract_script = Path(args.extract_script) if args.extract_script else (script_dir / "extract_phase2_features.py")
    if not extract_script.exists():
        raise FileNotFoundError(
            f"Extractor script not found: {extract_script}\n"
            f"Pass --extract-script to point to it."
        )

    # Where to write intermediate features CSV
    if args.keep_features_csv:
        features_csv = out_path.with_suffix("").with_name(out_path.stem + "_features.csv")
    else:
        features_csv = Path(tempfile.gettempdir()) / f"pyr_phase3_features_{out_path.stem}.csv"

    # 1) Extract Phase 3 AUC features
    run_extractor(
        extract_script=extract_script,
        phase3_runs_dir=phase3_runs_dir,
        out_csv=features_csv,
        tag=args.tag,
        budget=args.budget,
    )

    # 2) Load extracted Phase 3 features
    p3 = pd.read_csv(features_csv)
    if "status" in p3.columns:
        p3_ok = p3[p3["status"] == "OK"].copy()
    else:
        p3_ok = p3.copy()

    # 3) Load Phase 2 PASSING reference
    p2 = pd.read_csv(args.phase2_reference_csv)
    if "status" in p2.columns:
        p2 = p2[p2["status"] == "OK"].copy()

    if "auc_above_baseline_norm" not in p2.columns or "auc_above_baseline_norm" not in p3_ok.columns:
        raise ValueError("Missing column 'auc_above_baseline_norm' in reference or phase3 CSV.")

    auc_cut = float(p2["auc_above_baseline_norm"].quantile(args.quantile))
    p3_ok["likely_pass_like"] = p3_ok["auc_above_baseline_norm"] >= auc_cut

    max_cut = None
    if args.use_max:
        if "max_reward_to_budget" not in p2.columns or "max_reward_to_budget" not in p3_ok.columns:
            raise ValueError("Missing 'max_reward_to_budget' but --use-max was set.")
        max_cut = float(p2["max_reward_to_budget"].quantile(args.quantile))
        p3_ok["likely_pass_like"] = p3_ok["likely_pass_like"] & (p3_ok["max_reward_to_budget"] >= max_cut)

    # 4) Score for ranking (AUC primary; max reward tie-break if present)
    p3_ok["score"] = p3_ok["auc_above_baseline_norm"]
    if "max_reward_to_budget" in p3_ok.columns:
        p3_ok["score"] = p3_ok["score"] + 0.1 * p3_ok["max_reward_to_budget"]

    # 5) Add context + sort
    p3_ok["auc_cutoff"] = auc_cut
    if max_cut is not None:
        p3_ok["max_cutoff"] = max_cut
    p3_ok["budget"] = args.budget
    p3_ok["quantile_used"] = args.quantile
    p3_ok["features_csv_used"] = str(features_csv)

    p3_ok = p3_ok.sort_values(["likely_pass_like", "score"], ascending=[False, False])

    # 6) Write output
    p3_ok.to_csv(out_path, index=False)

    print(f"\nWrote predictions: {out_path}")
    if args.keep_features_csv:
        print(f"Kept intermediate features CSV: {features_csv}")
    else:
        print(f"Intermediate features CSV (temp): {features_csv}")

    msg = f"Pyramids cutoff: auc_above_baseline_norm >= {auc_cut:.4f} (q={args.quantile})"
    if max_cut is not None:
        msg += f" AND max_reward_to_budget >= {max_cut:.4f}"
    print(msg)

    show_cols = [c for c in [
        "run_id",
        "auc_above_baseline_norm",
        "max_reward_to_budget",
        "likely_pass_like",
        "score",
    ] if c in p3_ok.columns]

    print("\nTop 10:")
    print(p3_ok[show_cols].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
