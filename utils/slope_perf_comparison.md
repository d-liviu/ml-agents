# Example Usage for compute_slope_comparison.py

This document provides example commands to test the `compute_slope_comparison.py` script using the example config files.

## Example Config Files Created

The following example config files have been created in the `utils/` directory:

### Pyramids Environment:
- `example_old_pyramids_configs.txt` - Text file with old Pyramids run IDs (Pyramids1-5)
- `example_new_pyramids_configs.txt` - Text file with new Pyramids run IDs (Pyramids6-10)
- `example_old_pyramids_configs.csv` - CSV file with old Pyramids run IDs
- `example_new_pyramids_configs.csv` - CSV file with new Pyramids run IDs

### Worm Environment:
- `example_old_worm_configs.txt` - Text file with old Worm run IDs (Worm1-5)
- `example_new_worm_configs.txt` - Text file with new Worm run IDs (Worm6-10)
- `example_old_worm_configs.csv` - CSV file with old Worm run IDs
- `example_new_worm_configs.csv` - CSV file with new Worm run IDs

## Example Commands

### Example 1: Pyramids with Text Files (Minimal)
```bash
python utils/compute_slope_comparison.py \
    --old-configs utils/example_old_pyramids_configs.txt \
    --new-configs utils/example_new_pyramids_configs.txt
```

### Example 2: Pyramids with CSV Files and CSV Directory
```bash
python utils/compute_slope_comparison.py \
    --old-configs utils/example_old_pyramids_configs.csv \
    --new-configs utils/example_new_pyramids_configs.csv \
    --results-dir results \
    --old-csv-dir pyramids_csvs \
    --new-csv-dir pyramids_csvs \
    --tolerance 0.1 \
    --output pyramids_comparison_results.csv
```

### Example 3: Worm with Text Files
```bash
python utils/compute_slope_comparison.py \
    --old-configs utils/example_old_worm_configs.txt \
    --new-configs utils/example_new_worm_configs.txt \
    --results-dir results \
    --tolerance 0.1 \
    --output worm_comparison_results.csv
```

### Example 4: Worm with CSV Files and Full Options
```bash
python utils/compute_slope_comparison.py \
    --old-configs utils/example_old_worm_configs.csv \
    --new-configs utils/example_new_worm_configs.csv \
    --results-dir results \
    --old-csv-dir worm_csvs \
    --new-csv-dir worm_csvs \
    --tolerance 0.1 \
    --output worm_comparison_results.csv
```

### Example 5: Custom Tolerance (15% instead of 10%)
```bash
python utils/compute_slope_comparison.py \
    --old-configs utils/example_old_pyramids_configs.txt \
    --new-configs utils/example_new_pyramids_configs.txt \
    --tolerance 0.15 \
    --output pyramids_comparison_15pct.csv
```

## Notes

1. **CSV Directory**: If you have CSV files with threshold information (like `pyramids_csvs/` or `worm_csvs/`), use `--old-csv-dir` and `--new-csv-dir` to enable threshold filtering for old configs.

2. **Results Directory**: The default is `results/`, but you can specify a different directory if your training results are stored elsewhere.

3. **Environment Detection**: The script automatically detects whether you're using Worm or Pyramids based on the run IDs:
   - **Worm**: Extracts performance at **2M steps** and computes slope at 2M steps
   - **Pyramids**: Extracts performance at **1M steps** and computes slope at 1M steps

4. **Output**: The script prints results to console. Use `--output` to save results to a CSV file for further analysis.

## Expected Output

The script will:
1. Process old configs and compute mean slope and performance
2. Process new configs and compute their slopes and performances
3. Compare new configs against old config means (with tolerance)
4. Show which new configs pass/fail the comparison
5. Save detailed results to CSV if `--output` is specified

## Creating Your Own Config Files

You can create your own config files in two formats:

### Text File Format (one run_id per line):
```
Pyramids1
Pyramids2
Pyramids3
```

### CSV File Format (with run_id column):
```csv
run_id
Pyramids1
Pyramids2
Pyramids3
```
