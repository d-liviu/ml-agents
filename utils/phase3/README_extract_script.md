# Feature extraction (ML-Agents TensorBoard → CSV features)

This folder contains scripts that extract **early-learning features** from ML-Agents TensorBoard logs (event files)
and write them to a CSV. These features are used to compare runs across phases.

The main reward scalar used is usually:
- `Environment/Cumulative Reward` (default)

## How to run the extractor (optional)
Even though our prediction scripts can call extraction automatically, you can run extraction manually if needed.

Example (Worm):
```powershell
python extract_phase2_features.py `
  --input "D:\...\results\Worm_phase3" `
  --env worm `
  --budget 2000000 `
  --window 500000 `
  --out "worm_phase3_features_500k.csv"

## What “feature extraction” means
For each run folder (e.g., `results/<run_id>/`), we:
1) Find TensorBoard event files (`events.out.tfevents...`)
2) Read a reward curve (reward vs step)
3) Compute summary features up to a given step budget (e.g., 2M / 1M)

These features can be used for:
- Phase 2: summarising “passing” runs to understand what successful early learning looks like
- Phase 3: summarising reduced-budget runs (screening)
- Comparing Phase 3 to Phase 2 (selection / prediction heuristics)

## Output format
The extractor writes a CSV with one row per run and a `status` column:
- `OK` → features computed successfully
- `TAG_NOT_FOUND` → event files exist, but reward scalar tag not found
- `NO_EVENT_FILES` → no TensorBoard event files in that run folder

## Features by environment (defaults)

### Worm (budget = 2M)
Computed using a “last window” ending at the budget:
- `mean_last_window`: mean reward over [budget - window, budget]
- `slope_last_window_per_1m`: linear slope in the same window, scaled to “reward per 1M steps”
- `reward_at_budget`: reward interpolated at exactly the budget
- `n_points_window`: number of logged points in the window

Typical window values:
- 300k (more local) or 500k (smoother). We often use 500k.

### Pyramids (budget = 2M)
Slope at 2M can be unstable, so we use AUC:
- `auc_raw_norm`: average reward over [0, budget]
- `baseline`: mean reward in an early baseline region (e.g., first 100k steps)
- `auc_above_baseline_norm`: average improvement above baseline over [0, budget]
- `max_reward_to_budget`: max reward observed up to budget
- `reward_at_budget`: reward interpolated at budget

