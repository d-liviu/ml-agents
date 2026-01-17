# Baseline Metrics Calculator

A Python script for computing baseline performance metrics from reinforcement learning training runs using TensorBoard event files.

## Overview

This script analyzes training data from multiple RL configurations to establish baseline performance metrics. It extracts training curves from TensorBoard logs, computes slopes and performance values, and identifies successful configurations based on threshold criteria.

## Features

- **TensorBoard Integration**: Automatically extracts training data from TensorBoard event files
- **Flexible Input**: Supports CSV or text file configuration lists
- **Performance Metrics**: Computes learning curve slopes and performance at target steps
- **Threshold Detection**: Identifies runs that reached performance thresholds
- **Environment-Aware**: Automatically adjusts target steps for different environments (Worm: 2M steps, Pyramids: 1M steps)
- **Interpolation**: Handles cases where exact step counts don't exist in logs

## Installation

### Required Dependencies
```bash
pip install numpy pandas tensorboard
```

### Optional Dependencies
```bash
pip install scipy  # For improved slope computation
pip install pyyaml  # For YAML config parsing
```

## Usage

### Basic Command
```bash
python utils/compute_slope_perf.py \
  --configs path/to/configs/files/name.csv \
  --results-dir results \
  --csv-dir pyramids_csvs \
  --output baseline_metrics.json
```

### Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--configs` | Yes | - | Path to file containing config run_ids (CSV or text file) |
| `--results-dir` | No | `results` | Directory containing training results with TensorBoard logs |
| `--csv-dir` | No | `None` | Directory containing CSV files with threshold information |
| `--output` | Yes | - | Output CSV file path for baseline metrics |

## Input File Formats

### Config File (CSV)
```csv
run_id
pyramids_config_001
pyramids_config_002
worm_config_001
```

### Config File (Text)
```
pyramids_config_001
pyramids_config_002
worm_config_001
```

### Threshold CSV (Optional)
```csv
run_id,steps_to_threshold
pyramids_config_001,850000
pyramids_config_002,920000
```

## Output Format

The script generates a CSV file with the following columns:

- `lowest_slope`: The minimum acceptable slope from successful runs
- `mean_performance`: Average performance at target step across successful runs
- `target_step`: The step count used for evaluation (1M or 2M depending on environment)
- `env_name`: Detected environment name (Worm or Pyramids)
- `num_configs`: Number of configurations that reached threshold

### Example Output
```csv
lowest_slope,mean_performance,target_step,env_name,num_configs
0.000234,15.67,1000000,Pyramids,48
```

## How It Works

1. **Load Configurations**: Reads run IDs from the specified config file
2. **Extract Training Data**: Scans each run directory for TensorBoard event files and extracts step/reward pairs
3. **Detect Environment**: Infers environment type from run IDs to set appropriate target steps
4. **Compute Metrics**: 
   - Calculates linear regression slope across entire training curve
   - Interpolates performance at the target step
5. **Filter by Threshold**: Identifies runs that reached performance threshold (if CSV directory provided)
6. **Compute Baselines**: Calculates aggregate metrics from successful runs

## Directory Structure

Expected directory layout:
```
project/
├── results/
│   ├── pyramids_config_001/
│   │   └── events.out.tfevents.*
│   ├── pyramids_config_002/
│   │   └── events.out.tfevents.*
│   └── ...
├── pyramids_csvs/
│   ├── pyramids_config_001.csv
│   ├── pyramids_config_002.csv
│   └── ...
└── configs/
    └── baseline_configs.csv
```

## Metrics Explained

### Slope
The slope represents the learning rate across the entire training run. It's computed using linear regression on the (step, reward) pairs. A higher slope indicates faster learning.

### Performance at Target Step
The reward value at a specific step (1M for Pyramids, 2M for Worm). If the exact step doesn't exist in logs, the value is linearly interpolated between neighboring steps.

### Reached Threshold
A boolean indicating whether the run achieved the performance threshold. This information comes from the optional CSV files in `--csv-dir`.

## Troubleshooting

### No events file found
- Ensure TensorBoard logs exist in the results directory
- Check that the directory structure matches expected format

### No reward tag found
- Verify TensorBoard logs contain reward metrics
- Script looks for tags containing "Cumulative Reward", "Extrinsic Reward", or "Reward"

### Could not determine environment
- Ensure run IDs contain "worm" or "pyramids" (case-insensitive)
- Or manually specify by modifying the script

### No configs reached threshold
- Verify CSV files in `--csv-dir` match run IDs
- Script will use all configs if none reached threshold

## Notes

- The script is case-insensitive for environment detection
- Multiple CSV naming conventions are tried automatically
- If scipy is not installed, numpy's polyfit is used as fallback for slope computation
- The script uses the most recent TensorBoard event file if multiple exist