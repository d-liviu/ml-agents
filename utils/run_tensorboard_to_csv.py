import argparse
from pathlib import Path
import pandas as pd

from tensorboard_to_csv import tflog2pandas


def main():
    parser = argparse.ArgumentParser(description="Convert ML-Agents TensorBoard runs to CSV summary")
    parser.add_argument("--results_dir", type=str, default="results",
                        help="Path to the top-level results directory containing run subfolders")
    parser.add_argument("--out", type=str, default="results_summary.csv",
                        help="Output CSV file path")
    parser.add_argument("--run_id", type=str, default=None,
                        help="(Optional) single run subfolder name to process instead of all subfolders")

    args = parser.parse_args()
    results_dir = Path(args.results_dir)

    if not results_dir.exists():
        print(f"Results directory does not exist: {results_dir}")
        return

    dfs = []

    if args.run_id:
        run_path = results_dir / args.run_id
        if not run_path.exists():
            print(f"Run path does not exist: {run_path}")
            return
        df = tflog2pandas(run_path)
        if not df.empty:
            dfs.append(df)
    else:
        for child in sorted(results_dir.iterdir()):
            if child.is_dir():
                print(f"Processing: {child}")
                try:
                    df = tflog2pandas(child)
                    if not df.empty:
                        dfs.append(df)
                except Exception as e:
                    print(f"Error processing {child}: {e}")

    if not dfs:
        print("No run data extracted.")
        return

    out_df = pd.concat(dfs, ignore_index=True)
    out_path = Path(args.out)
    out_df.to_csv(out_path, index=False)
    print(f"Wrote summary to {out_path}")


if __name__ == "__main__":
    main()
