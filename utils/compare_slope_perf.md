# Config Filter

A Python script for filtering reinforcement learning training configurations based on baseline performance metrics with configurable tolerance thresholds.

## Overview

This script evaluates individual training run CSVs against baseline performance criteria (slope and mean performance) to identify successful configurations. It applies a delta tolerance (default 10%) to allow for slight variations while maintaining quality standards.

## Features

- **Environment-Specific Thresholds**: Predefined targets for Pyramids and Worm environments
- **Tolerance-Based Filtering**: Configurable delta parameter for threshold flexibility
- **Batch Processing**: Processes entire directories of run CSVs automatically
- **Detailed Logging**: Reports which runs pass/fail and why
- **Safe Output**: Creates output directories automatically

## Usage

### Basic Command
```bash
python filter_configs.py \
  --input-dir "results/pyramid_csvs" \
  --output-file "passed/successful_runs.csv" \
  --env "pyramids"
```

### Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--input-dir` | Yes | - | Directory containing individual run CSV files |
| `--output-file` | Yes | - | Path for output CSV (including filename) |
| `--env` | No | `pyramids` | Environment type: `pyramids` or `worm` |

## Baseline Thresholds

The script uses the following hardcoded baseline values:

### Pyramids Environment
- **Target Slope**: `0.0000006118641212960709`
- **Target Performance**: `-0.43691815845208415`
- **Target Step**: 1,000,000

### Worm Environment
- **Target Slope**: `0.00019152558179100633`
- **Target Performance**: `468.04508768717443`
- **Target Step**: 2,000,000

### Delta Tolerance
- **Default**: `0.1` (10%)
- Applied to both slope and performance thresholds

## Filtering Logic

A run passes if it meets BOTH conditions:
```python
run_slope > target_slope × (1 - DELTA)
AND
run_performance > target_performance × (1 - DELTA)
```

With default 10% delta:
- **Pyramids**: `slope > 0.00000055` AND `performance > -0.48`
- **Worm**: `slope > 0.000172` AND `performance > 421.2`

## Input File Format

Each CSV in the input directory should contain:

```csv
lowest_slope,mean_performance,target_step,env_name,num_configs
0.0000008,−0.35,1000000,Pyramids,1
```

**Required Columns:**
- `lowest_slope`: Learning curve slope value
- `mean_performance`: Mean reward at target step

## Output Format

The script generates a CSV containing only runs that passed the threshold:

```csv
run_id,slope,performance
pyramids_config_042,0.0000007234,−0.32
pyramids_config_089,0.0000006521,−0.41
worm_config_015,0.00021234,485.67
```

**Output Columns:**
- `run_id`: Configuration identifier (derived from filename)
- `slope`: Actual slope value from the run
- `performance`: Actual performance value from the run

## Examples

### Example 1: Filter Pyramids Runs
```bash
python filter_configs.py \
  --input-dir "results/pyramid_csvs" \
  --output-file "analysis/pyramids_passed.csv" \
  --env "pyramids"
```

### Example 2: Filter Worm Runs
```bash
python filter_configs.py \
  --input-dir "results/worm_csvs" \
  --output-file "analysis/worm_passed.csv" \
  --env "worm"
```

## Console Output

The script provides detailed feedback during execution:

```
Filtering runs with Slope > 0.00000055 and Performance > -0.48...
------------------------------
KEEP: pyramids_config_042 (S: 0.000001, P: -0.32)
DROP: pyramids_config_043 (Failed on Slope)
KEEP: pyramids_config_044 (S: 0.000001, P: -0.35)
DROP: pyramids_config_045 (Failed on Perf)
------------------------------
Success! Saved 2 configs to analysis/pyramids_passed.csv
```

## Directory Structure

Expected directory layout:
```
project/
├── results/
│   ├── pyramid_csvs/
│   │   ├── pyramids_config_001.csv
│   │   ├── pyramids_config_002.csv
│   │   └── ...
│   └── worm_csvs/
│       ├── worm_config_001.csv
│       ├── worm_config_002.csv
│       └── ...
└── passed/
    ├── successful_pyramids.csv
    └── successful_worm.csv
```

## Customizing Thresholds

To modify baseline thresholds, edit the constants at the top of the script:

```python
# Adjust these values based on your baseline metrics
TARGET_SLOPE_PYRAMIDS = 0.0000006118641212960709
TARGET_SLOPE_WORM = 0.00019152558179100633
TARGET_PERFORMANCE_PYRAMIDS = -0.43691815845208415
TARGET_PERFORMANCE_WORM = 468.04508768717443

# Adjust tolerance (0.1 = 10%, 0.2 = 20%, etc.)
DELTA = 0.1
```

## Workflow Integration

This script is typically used as **Step 2** in a complete analysis workflow:

1. **Step 1**: Generate baseline metrics using `compute_slope_perf.py`
2. **Step 2**: Filter configurations using `filter_configs.py` ← This script
3. **Step 3**: Analyze or deploy successful configurations

### Complete Workflow Example
```bash
# Step 1: Compute baselines from known good configs
python utils/compute_slope_perf.py \
  --configs baseline_configs.csv \
  --results-dir results \
  --output baseline_metrics.json

# Step 2: Filter all configs against baseline
python filter_configs.py \
  --input-dir results/pyramid_csvs \
  --output-file passed/successful_runs.csv \
  --env pyramids

# Step 3: Use the successful configs for further analysis
python analyze_configs.py --configs passed/successful_runs.csv
```

## Troubleshooting

### No configs met your criteria
- Check that baseline thresholds are appropriate for your environment
- Verify input CSVs contain the expected metric values
- Consider increasing the DELTA tolerance if too restrictive

### Missing metric columns error
- Ensure input CSVs have `lowest_slope` and `mean_performance` columns
- Verify CSVs were generated by `compute_slope_perf.py` or compatible tool

### File not found errors
- Verify the `--input-dir` path exists and contains CSV files
- Check file permissions for reading input and writing output

## Notes

- The script uses `csv_file.stem` to extract run_id from filenames (removes `.csv` extension)
- Output directory is created automatically if it doesn't exist
- The script skips files that don't have the required columns rather than failing
- Delta is applied as a multiplicative factor: `threshold × (1 - DELTA)`
- For negative performance values (like Pyramids), the delta effectively loosens the threshold (e.g., -0.44 becomes -0.48 with 10% delta)