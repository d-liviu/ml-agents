## Random Config Sweep Runner (ML-Agents)

Generate random PPO training configs from a base YAML, save each config to disk, and run `mlagents-learn` for each config/seed. The tool also writes a sweep manifest and archives the exact config + metadata in each run’s `results/<run_id>/` folder.

### What You Get
- Generated configs: `OUT_DIR/<run_id>.yaml`
- Sweep manifest: `OUT_DIR/manifest.jsonl` (one JSON line per run)
- After training, in `results/<run_id>/`:
	- `used_config.yaml` (the exact config used)
	- `run_meta.json` (seed, sampled hyperparams, paths, timestamps)

---

## Prerequisites
- Unity executable for the target scene built locally (e.g. `.../Builds/Worm.exe`, `.../Builds/Pyramids.exe`).
- ML-Agents CLI available in your environment:

```bash
mlagents-learn --help
```

If the above fails, activate your virtual environment and install ML-Agents.

- PyYAML for writing configs:

```bash
pip install pyyaml
```

---

## Script Location
This repository contains the runner script: `run_random_sweep.py`.

Run it from the repo root for best results.

---

## Usage

```bash
python run_random_sweep.py \
	--env-path <path-to-Unity-exe> \
	--base-yaml <path-to-base-trainer-yaml> \
	--out-dir <output-directory> \
	[--prefix Sweep] [--n-configs 25] [--seeds 0,1] \
	[--max-steps <int>] [--behavior <name>] [--sleep 5.0] \
	[--generate-only] [--start-index 0] [--count <int>] [--force]
```

### Required arguments
- `--env-path`: Path to the Unity executable (e.g., `.../Builds/Worm.exe`).
- `--base-yaml`: Base ML-Agents trainer YAML (e.g., `Worm.yaml`).
- `--out-dir`: Directory for generated configs and `manifest.jsonl`.

### Important options
- `--n-configs N`: Size of the shared sweep universe (same for the whole team).
- `--start-index I` and `--count C`: Run only a slice so teammates don’t overlap.
- `--seeds 0,1`: Comma-separated seeds to run per config (Stage A often uses just `0`).
- `--max-steps`: Override `max_steps` in the YAML (useful for screening).
- `--prefix`: Run-id prefix; keep consistent for easy merging (e.g., `WormA`).
- `--force`: Overwrite existing `results/<run_id>/` if it exists. Use with care.
- `--generate-only`: Only write configs/manifest; don’t launch training.
- `--behavior`: Specify the behavior name to edit; autodetected from YAML if omitted.

---

## Recommended Training Plan

### Stage A — Screening (Breadth)
Run many configs at lower step counts to quickly identify:
- dead configs (no learning/instability)
- promising configs (reach threshold early / stable curves)

Suggested:
- Worm: `--max-steps 2000000 --seeds 0`
- Pyramids: `--max-steps 1000000 --seeds 0`

### Stage B — Confirmation (Depth)
Rerun the best ~25% from Stage A at higher steps and multiple seeds:
- Worm: 5,000,000 steps, seeds `0,1`
- Pyramids: 3,000,000 steps, seeds `0,1`

---

## Deterministic Slices (Avoiding Duplicates)
Configs are generated deterministically from index `i`:
- Config with `i=17` is identical on every machine.
- No duplicates as long as teammates don’t overlap indices.

Team convention:
- Everyone uses the same `--n-configs 70`.
- Each person runs a unique `--start-index` + `--count` slice.

---

## Output and Sharing
Each teammate should share:
- Their `OUT_DIR` folder (generated YAMLs + `manifest.jsonl`).
- Their `results/` subfolders for the run-ids they trained.

Merging is straightforward because indices don’t overlap.

---

## Stage A Start Commands (Shared Settings)

Before running, replace placeholders:
- `ENV_WORM`: Path to Worm executable
- `ENV_PYR`: Path to Pyramids executable
- `BASE_WORM_YAML`: Path to base Worm trainer YAML
- `BASE_PYR_YAML`: Path to base Pyramids trainer YAML
- `OUT_WORM` / `OUT_PYR`: Output directories

Shared sweep settings for Stage A:
- `--n-configs 70`
- `--seeds 0`
- Worm steps: `--max-steps 2000000`
- Pyramids steps: `--max-steps 1000000`

Each person runs 10 configs (70 total across the team).

### Pauline (configs 0–9)

Worm
```bash
python run_random_sweep.py --env-path "ENV_WORM" --base-yaml "BASE_WORM_YAML" --out-dir "OUT_WORM" --prefix "WormA" --n-configs 70 --start-index 0 --count 10 --seeds 0 --max-steps 2000000
```

Pyramids
```bash
python run_random_sweep.py --env-path "ENV_PYR" --base-yaml "BASE_PYR_YAML" --out-dir "OUT_PYR" --prefix "PyrA" --n-configs 70 --start-index 0 --count 10 --seeds 0 --max-steps 1000000
```

### Mabby (configs 10–19)

Worm
```bash
python run_random_sweep.py --env-path "ENV_WORM" --base-yaml "BASE_WORM_YAML" --out-dir "OUT_WORM" --prefix "WormA" --n-configs 70 --start-index 10 --count 10 --seeds 0 --max-steps 2000000
```

Pyramids
```bash
python run_random_sweep.py --env-path "ENV_PYR" --base-yaml "BASE_PYR_YAML" --out-dir "OUT_PYR" --prefix "PyrA" --n-configs 70 --start-index 10 --count 10 --seeds 0 --max-steps 1000000
```

### Khaled (configs 20–29)

Worm
```bash
python run_random_sweep.py --env-path "ENV_WORM" --base-yaml "BASE_WORM_YAML" --out-dir "OUT_WORM" --prefix "WormA" --n-configs 70 --start-index 20 --count 10 --seeds 0 --max-steps 2000000
```

Pyramids
```bash
python run_random_sweep.py --env-path "ENV_PYR" --base-yaml "BASE_PYR_YAML" --out-dir "OUT_PYR" --prefix "PyrA" --n-configs 70 --start-index 20 --count 10 --seeds 0 --max-steps 1000000
```

### Liviu (configs 30–39)

Worm
```bash
python run_random_sweep.py --env-path "ENV_WORM" --base-yaml "BASE_WORM_YAML" --out-dir "OUT_WORM" --prefix "WormA" --n-configs 70 --start-index 30 --count 10 --seeds 0 --max-steps 2000000
```

Pyramids
```bash
python run_random_sweep.py --env-path "ENV_PYR" --base-yaml "BASE_PYR_YAML" --out-dir "OUT_PYR" --prefix "PyrA" --n-configs 70 --start-index 30 --count 10 --seeds 0 --max-steps 1000000
```

### Nick (configs 40–49)

Worm
```bash
python run_random_sweep.py --env-path "ENV_WORM" --base-yaml "BASE_WORM_YAML" --out-dir "OUT_WORM" --prefix "WormA" --n-configs 70 --start-index 40 --count 10 --seeds 0 --max-steps 2000000
```

Pyramids
```bash
python run_random_sweep.py --env-path "ENV_PYR" --base-yaml "BASE_PYR_YAML" --out-dir "OUT_PYR" --prefix "PyrA" --n-configs 70 --start-index 40 --count 10 --seeds 0 --max-steps 1000000
```

### Vlad (configs 50–59)

Worm
```bash
python run_random_sweep.py --env-path "ENV_WORM" --base-yaml "BASE_WORM_YAML" --out-dir "OUT_WORM" --prefix "WormA" --n-configs 70 --start-index 50 --count 10 --seeds 0 --max-steps 2000000
```

Pyramids
```bash
python run_random_sweep.py --env-path "ENV_PYR" --base-yaml "BASE_PYR_YAML" --out-dir "OUT_PYR" --prefix "PyrA" --n-configs 70 --start-index 50 --count 10 --seeds 0 --max-steps 1000000
```

### Sophia (configs 60–69)

Worm
```bash
python run_random_sweep.py --env-path "ENV_WORM" --base-yaml "BASE_WORM_YAML" --out-dir "OUT_WORM" --prefix "WormA" --n-configs 70 --start-index 60 --count 10 --seeds 0 --max-steps 2000000
```

Pyramids
```bash
python run_random_sweep.py --env-path "ENV_PYR" --base-yaml "BASE_PYR_YAML" --out-dir "OUT_PYR" --prefix "PyrA" --n-configs 70 --start-index 60 --count 10 --seeds 0 --max-steps 1000000
```

---

## Safety Tips
- Avoid `--force` unless you intend to overwrite `results/<run_id>/`.
- To rerun without overwriting, change `--prefix` (e.g., `WormA_rerun`).

---

## Notes
- The script auto-detects the behavior name from your base YAML when there’s a single behavior; you can explicitly set `--behavior` to override.
- You can generate configs without training by passing `--generate-only`.
- If desired, we can add `--skip-existing` logic to skip run-ids that already have results.