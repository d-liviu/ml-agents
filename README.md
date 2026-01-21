# Project 2.1 — Unity ML-Agents Experiments

> Python: **3.10 (recommended 3.10.12)**
> Trainer: **ML-Agents PPO** via `mlagents-learn`
> Further reading & dependencies: see the official ML-Agents repo → [https://github.com/DennisSoemers/ml-agents](https://github.com/DennisSoemers/ml-agents)


## 1) Build the Unity executables

1. Open the Unity project
2. Open the scene for your target environment (e.g., Worm or Pyramids)
3. **File -> Build Settings… -> Build**

   * **Windows**: build a `*.exe` (e.g., `Builds/Worm.exe`, `Builds/Pyramids.exe`)
   * **macOS**: build a `*.app` (e.g., `Builds/Worm.app`, `Builds/Pyramids.app`)
4. Note the absolute path to the built executable, its needed for the Python scripts.

---

## 2) Set up Python & ML-Agents


> Follow the **“ml-agents python package”** section in the official README for the full setup:
> [https://github.com/DennisSoemers/ml-agents](https://github.com/DennisSoemers/ml-agents)

---

## 3) Phase 2 - Fixed configs

**File:** `run_all.py`
Edit two things at the top:

```python
ENV_PATH = "ABSOLUTE_PATH_TO_YOUR_ENV_EXECUTABLE"  # .exe on Windows, .app on macOS

CONFIGS = [
    ("RunId_1", "ABS_PATH_TO/config/ppo/Worm/Worm.yaml"),
    ("RunId_2", "ABS_PATH_TO/config/ppo/Worm/worm_fastlearn.yaml"),
    # ...add more (run_id, yaml) pairs
]
```

Then run:

```bash
python run_all.py
```

Each entry is launched once with its `run_id`.

**Reference PPO config sets (can be found in the report):**

**Worm**

| Configuration file       | Changed parameters (value)                        |
| ------------------------ | ------------------------------------------------- |
| `worm_fastlearn.yaml`    | `learning_rate=0.0005; beta=0.01`                 |
| `worm_highcapacity.yaml` | `hidden_units=1024; num_layers=4`                 |
| `worm_slowsteady.yaml`   | `learning_rate=0.0002`                            |
| `worm_stablelearn.yaml`  | `batch_size=4048; buffer_size=40480; beta=0.0025` |

**Pyramids**

| Configuration file       | Changed parameters (value)         |
| ------------------------ | ---------------------------------- |
| `Pyramids_faster.yaml`   | `learning_rate=0.0005`             |
| `Pyramids_steadier.yaml` | `learning_rate=0.0002`             |
| `Pyramids_capacity.yaml` | `hidden_units=1024`                |
| `Pyramids_stabler.yaml`  | `batch_size=256; buffer_size=4096` |

> The YAMLs live under `config/ppo/Worm/` and `config/ppo/Pyramids/`.

*For more details, see the dedicated doc: **[run_all README](run_all_README.md)**.*


---

## 4) Phase 2 — Random config sweep

**File:** `run_random_sweep.py`
Generates random PPO configs from a base YAML, saves each to disk, and optionally launches training. 

**Common flags**

* `--env-path` path to build executable
* `--base-yaml` base PPO trainer YAML (e.g., `config/ppo/Worm/Worm.yaml`)
* `--out-dir` where to write generated YAMLs & a `manifest.jsonl`
* `--n-configs` total size of the shared sweep universe
* `--start-index` & `--count` run only a slice (avoid overlap across teammates)
* `--seeds` comma-separated seeds per config
* `--max-steps` override the trainer budget for screening
* `--prefix` run-id prefix
* `--generate-only` write configs without launching training
* `--force` allow overwriting an existing run-id

**Examples**

*Worm (screening, macOS path shown)*

```bash
python run_random_sweep.py \
  --env-path "/ABS/PATH/Builds/Worm.app" \
  --base-yaml "config/ppo/Worm/Worm.yaml" \
  --out-dir "OUT_WORM" \
  --prefix "WormA" \
  --n-configs 70 --start-index 0 --count 10 \
  --seeds 0 \
  --max-steps 2000000
```

*Pyramids (screening, Windows path shown)*

```powershell
python run_random_sweep.py `
  --env-path "C:\ABS\PATH\Builds\Pyramids.exe" `
  --base-yaml "config\ppo\Pyramids\Pyramids.yaml" `
  --out-dir "OUT_PYR" `
  --prefix "PyrA" `
  --n-configs 70 --start-index 20 --count 10 `
  --seeds 0 `
  --max-steps 2000000
```

*For more details, see the dedicated doc: **[run_random_sweep README](run_random_sweep.md)**.*

*Directory with all yaml configurations for phase 3 can be found under `config/YAMLs-PHASE3`*

---

## 5) Model Evaluation

Run cross-validated baselines for our two prediction tasks:

* **RQ1 (regression):** predict `final_perf` from run metadata/hyperparameters.
* **RQ2 (classification):** predict `likely_by_horizon_conservative` (0/1).

**Input:** `DataSet_complete.csv` (same folder), with columns:
`env_name, learning_rate, batch_size, nn_arch_depth, cpu_cores_logical, ram_total_gb, final_perf, likely_by_horizon_conservative`.

**Deps:** Python 3.10; `pip install pandas numpy scikit-learn`.

**Run:**

```bash
python crossValidation_engine.py
```

The script prints CV metrics (R²/MAE for RQ1; balanced accuracy/F1 for RQ2).
*For details on preprocessing, models, and customization, see the dedicated doc: **[crossValidation_engine README](crossValidation_engine.md)**.*

## 6) Helper Scripts

#### Scripts used throughout the project:

* unify_csvs.py
* utils/phase3/add_hyperparams.py
* utils/phase3/extract_phase2_features.py
* utils/phase3/resume_pyramids_runs_to_2m.py
* utils/standardize_naming_column.py