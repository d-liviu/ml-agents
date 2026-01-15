import os
import json
import traceback
import numpy as np
import pandas as pd
from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

""""
to execute the script:
    python utils/compute_slope_comparison.py \
        --old-configs path/to/old_configs.csv \
        --new-configs path/to/new_configs.csv \
        --results-dir results \
        --old-csv-dir pyramids_csvs \
        --new-csv-dir pyramids_csvs \
        --delta 0.1 \
        --output comparison_results.csv
"""

try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("Warning: scipy not installed. Using numpy polyfit for slope computation.")

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False
    print("Warning: PyYAML not installed. Configuration.yaml will not be parsed.")


def extract_training_data(path):
    """
    Extract training steps and rewards from tensorboard events file.
    Returns (steps, rewards) or (None, None) if extraction fails.
    """
    path = Path(path)
    
    candidates = []
    for pat in ("events.out.tfevents.*", "*.tfevents.*", "*.tfevents*"):
        found = list(path.rglob(pat))
        if found:
            candidates.extend(found)

    if not candidates:
        print(f"No events file found in {path.resolve()}")
        return None, None
    
    events_file = max(candidates, key=lambda p: p.stat().st_mtime)
    events_file_path = str(events_file.parent)
    
    try:
        event_acc = EventAccumulator(events_file_path, size_guidance={"scalars": 0})
        event_acc.Reload()
        tags = event_acc.Tags().get("scalars", [])
        
        episodic_reward_tag = None
        for tag in tags:
            if "Cumulative Reward" in tag:
                episodic_reward_tag = tag
                break
            elif "Extrinsic Reward" in tag:
                episodic_reward_tag = tag
                break
            elif "Reward" in tag and "Loss" not in tag:
                episodic_reward_tag = tag
                break
        
        if not episodic_reward_tag:
            print(f"No reward tag found in {path}")
            return None, None
        
        episodic_rewards = []
        steps = []
        events = event_acc.Scalars(episodic_reward_tag)
        for e in events:
            episodic_rewards.append(e.value)
            steps.append(e.step)
        
        return np.array(steps), np.array(episodic_rewards)
        
    except Exception as e:
        print(f"Error processing events file in {path}: {e}")
        traceback.print_exc()
        return None, None


def get_performance_at_step(steps, rewards, target_step):
    """
    Get performance (reward) at a specific step.
    If exact step doesn't exist, interpolate between nearest steps.
    """
    if steps is None or rewards is None or len(steps) == 0:
        return None
    
    # Find the index where target_step would be inserted
    idx = np.searchsorted(steps, target_step)
    
    if idx == 0:
        return float(rewards[0])
    elif idx >= len(steps):
        return float(rewards[-1])
    elif steps[idx] == target_step:
        #match
        return float(rewards[idx])
    else:
        # Interpolate between idx-1 and idx
        step_before = steps[idx - 1]
        step_after = steps[idx]
        reward_before = rewards[idx - 1]
        reward_after = rewards[idx]
        
        # Linear interpolation
        if step_after == step_before:
            return float(reward_before)
        
        t = (target_step - step_before) / (step_after - step_before)
        interpolated_reward = reward_before + t * (reward_after - reward_before)
        return float(interpolated_reward)


def compute_slope(steps, rewards, target_step, window_size=100000):
    """
    Compute the slope of the training curve around a target step.
    Uses a window around the target step to compute linear regression slope.
    
    Args:
        steps: array of step values
        rewards: array of reward values
        target_step: step at which to compute slope
        window_size: size of window around target_step to use for slope computation
    
    Returns:
        slope value or None if computation fails
    """
    if steps is None or rewards is None or len(steps) == 0:
        return None
    
    #Find indices within window around target_step
    lower_bound = max(0, target_step - window_size // 2)
    upper_bound = target_step + window_size // 2
    
    #Get data points within the window
    mask = (steps >= lower_bound) & (steps <= upper_bound)
    window_steps = steps[mask]
    window_rewards = rewards[mask]
    
    if len(window_steps) < 2:
        #Not enough data points, try to use all available data up to target_step
        mask = steps <= target_step
        window_steps = steps[mask]
        window_rewards = rewards[mask]
    
    if len(window_steps) < 2:
        return None
    
    # Compute linear regression slope
    try:
        if HAS_SCIPY:
            slope, intercept, r_value, p_value, std_err = stats.linregress(window_steps, window_rewards)
            return float(slope)
        else:
            # Use numpy polyfit as fallback
            coeffs = np.polyfit(window_steps, window_rewards, 1)
            return float(coeffs[0])  # coeffs[0] is the slope
    except Exception as e:
        print(f"Error computing slope: {e}")
        return None


def get_threshold_from_csv(csv_path, run_id):
    """
    Get steps_to_threshold from CSV file for a given run_id.
    Returns None if not found or threshold not reached.
    """
    try:
        df = pd.read_csv(csv_path)
        if 'run_id' in df.columns and 'steps_to_threshold' in df.columns:
            row = df[df['run_id'] == run_id]
            if not row.empty:
                threshold = row['steps_to_threshold'].iloc[0]
                # Check if threshold is not NaN/None
                if pd.notna(threshold):
                    return threshold
    except Exception as e:
        print(f"Error reading CSV {csv_path} for run_id {run_id}: {e}")
    return None


def load_configs(config_file):
    """
    Load configuration file. Supports:
    - CSV file with 'run_id' column
    - Text file with one run_id per line
    Returns list of run_ids
    """
    config_path = Path(config_file)
    if not config_path.exists():
        print(f"Config file not found: {config_file}")
        return []
    
    run_ids = []
    try:
        if config_path.suffix.lower() == '.csv':
            df = pd.read_csv(config_path)
            if 'run_id' in df.columns:
                run_ids = df['run_id'].tolist()
            else:
                print(f"CSV file {config_file} does not have 'run_id' column")
        else:
            # Assume text file with one run_id per line
            with open(config_path, 'r') as f:
                run_ids = [line.strip() for line in f if line.strip()]
    except Exception as e:
        print(f"Error loading config file {config_file}: {e}")
    
    return run_ids


def process_configs(config_file, results_dir, csv_dir=None, env_name=None):
    """
    Process a set of configs and compute slopes and performances.
    
    Args:
        config_file: path to file containing list of run_ids
        results_dir: directory containing training results (e.g., "results")
        csv_dir: optional directory containing CSV files with threshold info
        env_name: environment name ("Worm" or "Pyramids") to determine target step
    
    Returns:
        DataFrame with columns: run_id, slope, performance_at_target_step, reached_threshold
    """
    run_ids = load_configs(config_file)
    if not run_ids:
        return pd.DataFrame()
    
    results = []
    results_path = Path(results_dir)
    
    # Determine target step based on environment
    if env_name and "worm" in env_name.lower():
        target_step = 2_000_000
    elif env_name and "pyramids" in env_name.lower():
        target_step = 1_000_000
    else:
        # Try to infer from run_id or use default
        if any("worm" in rid.lower() for rid in run_ids):
            target_step = 2_000_000
        elif any("pyramids" in rid.lower() for rid in run_ids):
            target_step = 1_000_000
        else:
            print("Warning: Could not determine environment, using 1M steps as default")
            target_step = 1_000_000
    
    print(f"Processing {len(run_ids)} configs with target step: {target_step}")
    
    for run_id in run_ids:
        run_path = results_path / run_id
        if not run_path.exists():
            print(f"Warning: Run directory not found: {run_path}")
            continue
        
        # Extract training data
        steps, rewards = extract_training_data(run_path)
        if steps is None or rewards is None:
            print(f"Warning: Could not extract training data for {run_id}")
            continue
        
        # Check if reached threshold
        reached_threshold = False
        if csv_dir:
            # Try multiple naming conventions
            csv_paths_to_try = [
                Path(csv_dir) / f"{run_id.lower()}.csv",
                Path(csv_dir) / f"{run_id}.csv",
                Path(csv_dir) / f"{run_id.replace('_', '').lower()}.csv",
            ]
            # Also try searching in CSV directory for any file containing the run_id
            csv_dir_path = Path(csv_dir)
            if csv_dir_path.exists():
                for csv_file in csv_dir_path.glob("*.csv"):
                    try:
                        df_test = pd.read_csv(csv_file)
                        if 'run_id' in df_test.columns and run_id in df_test['run_id'].values:
                            csv_paths_to_try.append(csv_file)
                            break
                    except:
                        pass
            
            for csv_path in csv_paths_to_try:
                if csv_path.exists():
                    threshold = get_threshold_from_csv(csv_path, run_id)
                    if threshold is not None:
                        reached_threshold = True
                        break
        
        # Compute slope at target step
        slope = compute_slope(steps, rewards, target_step)
        
        # Get performance at target step
        performance = get_performance_at_step(steps, rewards, target_step)
        
        results.append({
            'run_id': run_id,
            'slope': slope,
            'performance_at_target_step': performance,
            'reached_threshold': reached_threshold,
            'target_step': target_step
        })
    
    return pd.DataFrame(results)


def compare_configs(old_configs_file, new_configs_file, results_dir, 
                   old_csv_dir=None, new_csv_dir=None, delta=0.1):
    """
    Args:
        old_configs_file: path to file with old config run_ids
        new_configs_file: path to file with new config run_ids
        results_dir: directory containing training results
        old_csv_dir: optional directory with CSV files for old configs
        new_csv_dir: optional directory with CSV files for new configs
        delta: delta factor (0.1 = 10% below mean is acceptable)
    
    Returns:
        DataFrame with comparison results
    """
    print("=" * 80)
    print("Processing old configs...")
    print("=" * 80)
    
    # Determine environment from old configs
    old_run_ids = load_configs(old_configs_file)
    if old_run_ids:
        if any("worm" in rid.lower() for rid in old_run_ids):
            env_name = "Worm"
        elif any("pyramids" in rid.lower() for rid in old_run_ids):
            env_name = "Pyramids"
        else:
            env_name = None
    else:
        env_name = None
    
    # Process old configs
    old_df = process_configs(old_configs_file, results_dir, old_csv_dir, env_name)
    
    if old_df.empty:
        print("No old configs processed successfully")
        return pd.DataFrame()
    
    # Filter old configs that reached threshold
    old_df_filtered = old_df[old_df['reached_threshold'] == True].copy()
    
    if old_df_filtered.empty:
        print("Warning: No old configs reached threshold. Using all old configs.")
        old_df_filtered = old_df.copy()

    # find the lowest slope for threshold
    lowest_slope = old_df_filtered['slope'].sort_values().iloc[0]

    # Compute means for performance
    mean_performance = old_df_filtered['performance_at_target_step'].mean()
    
    print(f"\nOld configs statistics (from {len(old_df_filtered)} configs that reached threshold):")
    print(f"  Lowest accepted slope: {lowest_slope:.6f}")
    print(f"  Mean performance: {mean_performance:.6f}")
    print(f"  Target step: {old_df_filtered['target_step'].iloc[0]}")
    
    print("\n" + "=" * 80)
    print("Processing new configs...")
    print("=" * 80)
    
    # Process new configs
    new_df = process_configs(new_configs_file, results_dir, new_csv_dir, env_name)
    
    if new_df.empty:
        print("No new configs processed successfully")
        return pd.DataFrame()
    
    # Compare new configs to old config means
    comparison_results = []
    
    for idx, row in new_df.iterrows():
        slope = row['slope']
        performance = row['performance_at_target_step']
        
        # Check if slope and performance meet criteria (>= mean with delta)
        slope_ok = (slope is not None and lowest_slope is not None and 
                   slope >= (1 - delta) * lowest_slope) if lowest_slope is not None else None
        perf_ok = (performance is not None and mean_performance is not None and 
                  performance >= (1 - delta) * mean_performance) if mean_performance is not None else None
        
        # Overall pass if both criteria are met
        passes = (slope_ok is True and perf_ok is True) if (slope_ok is not None and perf_ok is not None) else None
        
        comparison_results.append({
            'run_id': row['run_id'],
            'slope': slope,
            'performance_at_target_step': performance,
            'lowest_slope_old': lowest_slope,
            'mean_performance_old': mean_performance,
            'slope_threshold': (1 - delta) * lowest_slope if lowest_slope is not None else None,
            'performance_threshold': (1 - delta) * mean_performance if mean_performance is not None else None,
            'slope_passes': slope_ok,
            'performance_passes': perf_ok,
            'overall_passes': passes
        })
    
    comparison_df = pd.DataFrame(comparison_results)
    
    print("\n" + "=" * 80)
    print("Comparison Results:")
    print("=" * 80)
    print(comparison_df.to_string(index=False))
    
    passed_count = comparison_df['overall_passes'].sum() if comparison_df['overall_passes'].notna().any() else 0
    total_count = len(comparison_df)
    print(f"\nSummary: {passed_count}/{total_count} new configs passed the comparison")
    
    return comparison_df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Compute slopes and compare training configs')
    parser.add_argument('--old-configs', type=str, required=True,
                       help='Path to file containing old config run_ids (CSV or text file)')
    parser.add_argument('--new-configs', type=str, required=True,
                       help='Path to file containing new config run_ids (CSV or text file)')
    parser.add_argument('--results-dir', type=str, default='results',
                       help='Directory containing training results (default: results)')
    parser.add_argument('--old-csv-dir', type=str, default=None,
                       help='Directory containing CSV files for old configs (optional)')
    parser.add_argument('--new-csv-dir', type=str, default=None,
                       help='Directory containing CSV files for new configs (optional)')
    parser.add_argument('--delta', type=float, default=0.1,
                       help='delta factor for comparison (default: 0.1 = 10%%)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output CSV file path for comparison results (optional)')
    
    args = parser.parse_args()
    
    comparison_df = compare_configs(
        args.old_configs,
        args.new_configs,
        args.results_dir,
        args.old_csv_dir,
        args.new_csv_dir,
        args.delta
    )
    
    if args.output and not comparison_df.empty:
        comparison_df.to_csv(args.output, index=False)
        print(f"\nComparison results saved to {args.output}")
        