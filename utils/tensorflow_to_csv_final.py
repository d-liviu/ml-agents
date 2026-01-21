import os
import glob
import json
import traceback
import numpy as np
import pandas as pd
from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False
    print("Warning: PyYAML not installed. Configuration.yaml will not be parsed.")

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

# path is a path to results/'run id' directory (e.g., "results/first3DBallRun2")
def tflog2pandas(path):

    path = Path(path)
    
    candidates = []
    for pat in ("events.out.tfevents.*", "*.tfevents.*", "*.tfevents*"):
        found = list(path.rglob(pat))
        if found:
            candidates.extend(found)

    if not candidates:
        print(f"No events file found in {path.resolve()}")
        return pd.DataFrame()
    
    events_file = max(candidates, key=lambda p: p.stat().st_mtime)
    events_file_path = str(events_file.parent)

    config_path = path / "configuration.yaml"
    config_data = {}
    if config_path.exists() and HAS_YAML:
        try:
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f) or {}
        except Exception as e:
            print(f"Warning: Could not read configuration.yaml: {e}")
    
    timers_path = path / "run_logs" / "timers.json"
    wallclock_seconds = None
    if timers_path.exists():
        try:
            with open(timers_path, 'r') as f:
                timers_data = json.load(f)
                metadata = timers_data.get("metadata", {})
                start_time = metadata.get("start_time_seconds")
                end_time = metadata.get("end_time_seconds")
                if start_time and end_time:
                    wallclock_seconds = float(end_time) - float(start_time)
                elif "total" in timers_data:
                    wallclock_seconds = timers_data["total"]
        except Exception as e:
            print(f"Warning: Could not read timers.json: {e}")
    
    cpu_cores = None
    ram_gb = None
    if HAS_PSUTIL:
        cpu_cores = psutil.cpu_count(logical=True)
        ram_gb = psutil.virtual_memory().total / (1024**3)
    else:
        cpu_cores = os.cpu_count()       
    
    learning_rate = None
    batch_size = None
    nn_arch_depth = None
    algo_name = None
    env_name = None
    
    run_id = path.name
    behaviors = config_data.get("behaviors", {})
    if behaviors:
        behavior_name = list(behaviors.keys())[0]
        behavior_config = behaviors[behavior_name]
        hyperparams = behavior_config.get("hyperparameters", {})
        learning_rate = hyperparams.get("learning_rate")
        batch_size = hyperparams.get("batch_size")
        algo_name = behavior_config.get("trainer_type", "unknown")
        network_settings = behavior_config.get("network_settings", {})
        nn_arch_depth = network_settings.get("num_layers")
    events_dir = Path(events_file_path)
    if events_dir.name and events_dir.name != path.name:
        env_name = events_dir.name
        
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


        
        episodic_rewards = []
        steps = []
        if episodic_reward_tag:
            events = event_acc.Scalars(episodic_reward_tag)
            for e in events:
                episodic_rewards.append(e.value)
                steps.append(e.step)
        else:
            print("No reward tag found")
        
        final_perf = episodic_rewards[-1] if episodic_rewards else None
        
        result = steps_to_flattening(steps, episodic_rewards)
        plateau_reward = result[0]
        plateau_step = result[1]

        steps_to_threshold = None
        if plateau_reward is not None:
            for s, r in zip(steps, episodic_rewards):
                if r >= plateau_reward:
                    steps_to_threshold = s
                    break

        episodic_r_mean = float(np.mean(episodic_rewards)) if episodic_rewards else None
        
        row_data = {
            "run_id": run_id,
            "wallclock_seconds_total": wallclock_seconds,
            "algo_name": algo_name,
            "env_name": env_name,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "nn_arch_depth": nn_arch_depth,
            "cpu_cores_logical": cpu_cores,
            "ram_total_gb": ram_gb,
            "final_perf": final_perf,
            "steps_to_threshold": steps_to_threshold,
            "episodic_reward_mean": episodic_r_mean,
            "plateau_reward": plateau_reward
        }
        
        runlog_data = pd.DataFrame([row_data])
        
    except Exception as e:
        print(f"Error processing events file: {e}")
        traceback.print_exc()
        data_file_cols = [
            "run_id",
            "wallclock_seconds_total",
            "algo_name",
            "env_name",
            "learning_rate",
            "batch_size",
            "nn_arch_depth",
            "cpu_cores_logical",
            "ram_total_gb",
            "final_perf",
            "steps_to_threshold",
            "episodic_reward_mean",
            "plateau_reward"
        ]
        runlog_data = pd.DataFrame(columns=data_file_cols)
    
    return runlog_data

def steps_to_flattening(steps, rewards, window=10, min_rel_improve=0.05):
    """
    Find plateau using the 5% rule.
    Returns (plateau_reward, plateau_step) or (None, None).
    """
    if not steps or not rewards or len(rewards) < 2 * window:
        print("Not enough data to compute steps")
        return None, None

    steps_array = np.asarray(steps)
    rewards_array = np.asarray(rewards, dtype=float)

    num_windows = len(rewards_array) // window
    if num_windows < 2:
        return None, None

    window_means = []
    window_end_steps = []
    for i in range(num_windows):
        start = i * window
        end = (i + 1) * window
        mean_r = rewards_array[start:end].mean()
        window_means.append(mean_r)

        end_step_idx = min(end, len(steps_array)) - 1
        window_end_steps.append(int(steps_array[end_step_idx]))

    #search from the END, so we get the final plateau, not an early one
    for i in range(len(window_means) - 1, 0, -1):
        prev_m = window_means[i - 1]
        curr_m = window_means[i]
        denom = max(abs(prev_m), 1e-8)
        rel_improve = (curr_m - prev_m) / denom

        if rel_improve < min_rel_improve:
            plateau_reward = curr_m
            plateau_step = window_end_steps[i]
            return plateau_reward, plateau_step

    return None, None

if __name__ == "__main__":

    RUN_IDS = [
        "Worm1",
        "Worm2",
        "Worm3",
        "Worm4",
        "Worm5",
        "Worm6",
        "Worm7",
        "Worm8",
        "Worm9",
        "Worm10",
    ]

    RESULTS_DIR = Path("results")
    OUTPUT_DIR = Path("phase3_csvs/worm")

    print(f"Starting batch processing for {len(RUN_IDS)} runs...")

    for run_id in RUN_IDS:
        folder_path = RESULTS_DIR / run_id
        
        print(f"Processing: {run_id}...", end=" ")
        
        df = tflog2pandas(folder_path)
        if not df.empty:
            output_file = OUTPUT_DIR / f"{run_id}.csv"
            df.to_csv(output_file, index=False)
            print(f"Done with {run_id}! Saved to {output_file}")
        else:
            print(f"Failed. No data found for {run_id}")

print("\nAll tasks completed.")
