import os
import glob
import json
import traceback
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
    
    events_file_path = None
    for events_file in path.rglob("*.tfevents*"):
        events_file_path = str(events_file.parent)  #EventAccumulator needs directory
        break
    
    if events_file_path is None:
        print(f"No events file found in {path}")
        return pd.DataFrame()
    
    config_path = path / "configuration.yaml"
    config_data = {}
    if config_path.exists() and HAS_YAML:
        try:
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f) or {}
        except Exception as e:
            print(f"Warning: Could not read configuration.yaml: {e}")
    
    # find timers.json for wallclock time
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
    
    # extract system information 
    cpu_cores = None
    ram_gb = None

    if HAS_PSUTIL:
        cpu_cores = psutil.cpu_count(logical=True)
        ram_gb = psutil.virtual_memory().total / (1024**3)
    else:
        cpu_cores = os.cpu_count()       
    
    # extract hyperparameters from config
    learning_rate = None
    batch_size = None
    nn_arch_depth = None
    algo_name = None
    env_name = None
    
    # get behavior name (usually first key in behaviors)
    behaviors = config_data.get("behaviors", {})
    if behaviors:
        behavior_name = list(behaviors.keys())[0]
        behavior_config = behaviors[behavior_name]
        
        algo_name = behavior_config.get("trainer_type", "unknown")
        
        hyperparams = behavior_config.get("hyperparameters", {})
        learning_rate = hyperparams.get("learning_rate")
        batch_size = hyperparams.get("batch_size")
        
        network_settings = behavior_config.get("network_settings", {})
        nn_arch_depth = network_settings.get("num_layers")
    
    events_dir = Path(events_file_path)
    if events_dir.name and events_dir.name != path.name:
        env_name = events_dir.name
    
    run_id = path.name
    
    # extract metrics from events file
    try:
        event_acc = EventAccumulator(events_file_path, size_guidance={"scalars": 0})
        event_acc.Reload()
        tags = event_acc.Tags().get("scalars", [])
        
        #find episodic reward tag (common names: Environment/Cumulative Reward, Policy/Extrinsic Reward)
        episodic_reward_tag = None
        for tag in tags:
            if "Cumulative Reward" in tag:
                episodic_reward_tag = tag
                break
        if not episodic_reward_tag:
            for tag in tags:
                if "Extrinsic Reward" in tag:
                    episodic_reward_tag = tag
                    break
        if not episodic_reward_tag:
            for tag in tags:
                if "Reward" in tag and "Loss" not in tag:
                    episodic_reward_tag = tag
                    break
        
        episodic_rewards = []
        steps = []
        if episodic_reward_tag:
            events = event_acc.Scalars(episodic_reward_tag)
            for e in events:
                episodic_rewards.append(e.value)
                steps.append(e.step)
        
        final_perf = episodic_rewards[-1] if episodic_rewards else None
        
        episodic_reward_mean = sum(episodic_rewards) / len(episodic_rewards) if episodic_rewards else None
        
        # Calculate steps_to_threshold (steps when reward first exceeds a threshold)
        steps_to_threshold = None
        if episodic_rewards and final_perf:
            threshold = final_perf   
            for i, reward in enumerate(episodic_rewards):
                if reward >= threshold:
                    steps_to_threshold = steps[i] if i < len(steps) else None
                    break
        
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
            "episodic_reward_mean": episodic_reward_mean,
        }
        
        runlog_data = pd.DataFrame([row_data])
        
    except Exception as e:
        print(f"Error processing events file: {e}")
        traceback.print_exc()
        # Return empty DataFrame with correct columns
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
        ]
        runlog_data = pd.DataFrame(columns=data_file_cols)
    
    return runlog_data


# Example usage
if __name__ == "__main__":
    path ="results/first3DBallRun2"
    df = tflog2pandas(path)
    if not df.empty:
        df.to_csv("output.csv", index=False)
        print(f"Extracted data:\n{df}")
    else:
        print("No data extracted")


