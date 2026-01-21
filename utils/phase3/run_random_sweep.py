import argparse
import copy
import datetime as dt
import json
import math
import random
import shutil
import subprocess
import time
from pathlib import Path

import yaml

# Script to run a random hyperparameter sweep for ML-Agents PPO training.
# Generates multiple YAML config files with random hyperparameters,
# then runs mlagents-learn for each config, saving results in a specified output directory.

def loguniform(rng: random.Random, lo: float, hi: float) -> float:
    return 10 ** rng.uniform(math.log10(lo), math.log10(hi))


def choose_behavior_name(cfg: dict, explicit: str | None) -> str:
    if explicit:
        return explicit
    behaviors = cfg.get("behaviors", {})
    if not behaviors:
        raise ValueError("Base YAML has no top-level 'behaviors:' section.")
    
    return next(iter(behaviors.keys()))


def ensure_sections(cfg: dict) -> None:
    cfg.setdefault("env_settings", {})
    cfg.setdefault("engine_settings", {})
    cfg.setdefault("checkpoint_settings", {})
    cfg.setdefault("torch_settings", {})


def apply_overrides(cfg: dict, behavior_name: str, overrides: dict) -> None:
    b = cfg["behaviors"][behavior_name]
    b.setdefault("hyperparameters", {})
    b.setdefault("network_settings", {})

    hp = b["hyperparameters"]
    net = b["network_settings"]

    if "max_steps" in overrides:
        b["max_steps"] = int(overrides["max_steps"])

    for k in ["learning_rate", "batch_size", "buffer_size", "beta"]:
        if k in overrides:
            hp[k] = overrides[k]

    for k in ["hidden_units", "num_layers"]:
        if k in overrides:
            net[k] = overrides[k]

    hp.setdefault("learning_rate_schedule", "linear")
    hp.setdefault("beta_schedule", "constant")


def sample_hparams(rng: random.Random) -> dict:
   # specify hyperparameter ranges
    batch_size = rng.choice([64, 128, 256, 512, 1024, 2048, 4096])
    buffer_mult = rng.choice([10, 20, 40])
    buffer_size = batch_size * buffer_mult

    lr = loguniform(rng, 1e-5, 1e-3)
    beta = loguniform(rng, 1e-4, 1e-2)

    hidden_units = rng.choice([64, 128, 256, 512, 1024])
    num_layers = rng.choice([1, 2, 3, 4])

    # return sampled hyperparameters
    return {
        "learning_rate": float(lr),
        "batch_size": int(batch_size),
        "buffer_size": int(buffer_size),
        "beta": float(beta),
        "hidden_units": int(hidden_units),
        "num_layers": int(num_layers),
    }


def run_one(cmd: list[str], run_id: str) -> None:
    print("\n" + "=" * 80)
    print(f"Starting run: {run_id}")
    print("CMD:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"Finished run: {run_id}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env-path", required=True, help="Path to Unity executable (.exe)")
    ap.add_argument("--base-yaml", required=True, help="Path to a base trainer YAML")
    ap.add_argument("--out-dir", required=True, help="Where to write generated configs + manifest")
    ap.add_argument("--prefix", default="Sweep", help="Run-id prefix")
    ap.add_argument("--n-configs", type=int, default=25, help="How many random configs to generate")
    ap.add_argument("--seeds", default="0,1", help="Comma-separated seeds, e.g. 0,1")
    ap.add_argument("--max-steps", type=int, default=None, help="Override max_steps in YAML (optional)")
    ap.add_argument("--behavior", default=None, help="Behavior name to edit (optional; autodetected if omitted)")
    ap.add_argument("--sleep", type=float, default=5.0, help="Seconds to sleep between runs")
    ap.add_argument("--generate-only", action="store_true", help="Only generate configs/manifest, don't train")
    ap.add_argument("--start-index", type=int, default=0, help="Start index (useful to split work across teammates)")
    ap.add_argument("--count", type=int, default=None, help="How many configs to run from start-index")
    ap.add_argument("--force", action="store_true", help="Pass --force to overwrite existing run-id results")

    args = ap.parse_args()

    env_path = str(Path(args.env_path))
    base_yaml_path = Path(args.base_yaml)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip() != ""]
    if not seeds:
        raise ValueError("No seeds parsed from --seeds")

    base_cfg = yaml.safe_load(base_yaml_path.read_text(encoding="utf-8"))
    ensure_sections(base_cfg)
    behavior_name = choose_behavior_name(base_cfg, args.behavior)

    start = args.start_index
    total = args.n_configs if args.count is None else min(args.n_configs, start + args.count)
    indices = list(range(start, total))

    manifest_path = out_dir / "manifest.jsonl"

    sweep_rng = random.Random(12345)

    with manifest_path.open("a", encoding="utf-8") as mf:
        for i in indices:
            cfg_rng = random.Random((12345 << 16) + i)
            sampled = sample_hparams(cfg_rng)

            for seed in seeds:
                cfg = copy.deepcopy(base_cfg)

                overrides = dict(sampled)
                if args.max_steps is not None:
                    overrides["max_steps"] = args.max_steps

            
                cfg["env_settings"]["seed"] = int(seed)

                apply_overrides(cfg, behavior_name, overrides)

                run_id = f"{args.prefix}_{i:04d}_s{seed}"
                cfg_path = out_dir / f"{run_id}.yaml"
                cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

                meta = {
                    "run_id": run_id,
                    "created_at": dt.datetime.now().isoformat(),
                    "env_path": env_path,
                    "base_yaml": str(base_yaml_path),
                    "behavior": behavior_name,
                    "seed": seed,
                    "overrides": overrides,
                }
                mf.write(json.dumps(meta) + "\n")
                mf.flush()

                if args.generate_only:
                    continue

                cmd = [
                    "mlagents-learn",
                    str(cfg_path),
                    f"--run-id={run_id}",
                    f"--env={env_path}",
                    "--no-graphics",
                    "--train",
                ]
                if args.force:
                    cmd.append("--force")

                try:
                    run_one(cmd, run_id)
                except subprocess.CalledProcessError as e:
                    print(f"Run failed (continuing): {run_id}\n{e}")
                    continue

                results_dir = Path("results") / run_id
                if results_dir.exists():
                    shutil.copy2(cfg_path, results_dir / "used_config.yaml")
                    (results_dir / "run_meta.json").write_text(
                        json.dumps(meta, indent=2), encoding="utf-8"
                    )

                time.sleep(args.sleep)

    print("\nAll done.")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
