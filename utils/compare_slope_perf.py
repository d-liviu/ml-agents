import pandas as pd
from pathlib import Path
import argparse

#### CHANGE THIS DEPENDING ON THE ENV
TARGET_SLOPE_PYRAMIDS = 0.0000006118641212960709 
TARGET_SLOPE_WORM = 0.00019152558179100633 
TARGET_PERFORMANCE_PYRAMIDS = -0.43691815845208415 
TARGET_PERFORMANCE_WORM = 468.04508768717443 
DELTA = 0.1

def filter_by_manual_values(input_dir, output_file, env):
    input_path = Path(input_dir)
    output_path = Path(output_file)
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if env.lower() == "worm":
        slope_target = TARGET_SLOPE_WORM
        perf_target = TARGET_PERFORMANCE_WORM
    else:
        slope_target = TARGET_SLOPE_PYRAMIDS
        perf_target = TARGET_PERFORMANCE_PYRAMIDS

    passed_runs = []

    print(f"Filtering runs with Slope > {slope_target} and Performance > {perf_target}...")
    print("-" * 30)

    for csv_file in input_path.glob("*.csv"):
        try:
            df = pd.read_csv(csv_file)
            
            # Assumes the CSV has 'lowest_slope' and 'mean_performance' columns
            if 'lowest_slope' in df.columns and 'mean_performance' in df.columns:
                run_slope = df['lowest_slope'].iloc[0]
                run_perf = df['mean_performance'].iloc[0]

                # 3. Apply Comparison Logic
                if run_slope > slope_target * (1-DELTA) and run_perf > perf_target * (1-DELTA):
                    passed_runs.append({
                        'run_id': csv_file.stem,
                        'slope': run_slope,
                        'performance': run_perf
                    })
                    print(f"KEEP: {csv_file.stem} (S: {run_slope:.6f}, P: {run_perf:.2f})")
                else:
                    # Optional: print why it failed
                    reason = "Slope" if run_slope <= slope_target else "Perf"
                    print(f"DROP: {csv_file.stem} (Failed on {reason})")
            else:
                print(f"Skip: {csv_file.name} is missing metric columns.")

        except Exception as e:
            print(f"Error processing {csv_file.name}: {e}")

    if passed_runs:
        passed_df = pd.DataFrame(passed_runs)
        passed_df.to_csv(output_path, index=False)
        print("-" * 30)
        print(f"Success! Saved {len(passed_runs)} configs to {output_path}")
    else:
        print("\nNo configs met your criteria.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Filter training configs based on baseline metrics.')
    
    parser.add_argument('--input-dir', type=str, required=True, 
                        help='Directory containing the individual run CSVs')
    parser.add_argument('--output-file', type=str, required=True, 
                        help='Path (including filename.csv) where to save results')
    parser.add_argument('--env', type=str, default='pyramids', choices=['pyramids', 'worm'],
                        help='Which environment targets to use (default: pyramids)')
    args = parser.parse_args()

    filter_by_manual_values(args.input_dir, args.output_file, args.env)