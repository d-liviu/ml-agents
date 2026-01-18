import argparse
import copy
import subprocess
import sys
import time
from pathlib import Path

import yaml


def choose_behavior_name(cfg: dict, explicit: str | None) -> str:
    if explicit:
        return explicit
    behaviors = cfg.get("behaviors", {})
    if not behaviors:
        raise ValueError("Config has no top-level 'behaviors:' section.")
    # If only one behavior, use it; otherwise pick the first.
    return next(iter(behaviors.keys()))


def find_base_config(run_dir: Path) -> Path | None:
    """
    Try to find a config to resume from.
    Priority:
      1) used_config.yaml (our scripts often copy this in)
      2) configuration.yaml (ML-Agents often writes this in run folder)
      3) any *.yaml in the run folder (fallback)
    """
    candidates = [
        run_dir / "used_config.yaml",
        run_dir / "configuration.yaml",
    ]
    for c in candidates:
        if c.exists():
            return c

    yamls = list(run_dir.glob("*.yaml"))
    if len(yamls) == 1:
        return yamls[0]
    if len(yamls) > 1:
        # pick most recently modified
        yamls.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return yamls[0]

    return None


def make_resume_config(
    base_cfg: dict,
    run_id: str,
    results_root: Path,
    target_steps: int,
    behavior: str,
) -> dict:
    cfg = copy.deepcopy(base_cfg)

    # Ensure sections exist
    cfg.setdefault("behaviors", {})
    cfg.setdefault("checkpoint_settings", {})

    if behavior not in cfg["behaviors"]:
        raise ValueError(f"Behavior '{behavior}' not found in config behaviors: {list(cfg['behaviors'].keys())}")

    b = cfg["behaviors"][behavior]
    current = int(b.get("max_steps", 0) or 0)
    b["max_steps"] = max(current, int(target_steps))

    # Point ML-Agents to the folder that contains run subfolders
    # so it can find results_root/run_id for --resume.
    cs = cfg["checkpoint_settings"]
    cs["results_dir"] = str(results_root)
    cs["run_id"] = run_id
    cs["resume"] = True
    cs["force"] = False  # safety

    return cfg


def run_resume(run_id: str, resume_cfg_path: Path, env_path: Path) -> None:
    cmd = [
        "mlagents-learn",
        str(resume_cfg_path),
        f"--run-id={run_id}",
        "--resume",
        f"--env={str(env_path)}",
        "--no-graphics",
    ]
    print("\n" + "=" * 90)
    print(f"Resuming run: {run_id}")
    print("CMD:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--runs-dir", required=True,
                    help="Folder containing run subfolders (e.g. ...\\results\\Pyramids_phase3)")
    ap.add_argument("--env-path", required=True,
                    help="Path to the Pyramids Unity executable (.exe)")
    ap.add_argument("--target-steps", type=int, default=2_000_000,
                    help="Target max_steps to resume to (default 2,000,000)")
    ap.add_argument("--behavior", default=None,
                    help="Behavior name (optional; auto-detected if base YAML has one behavior)")
    ap.add_argument("--sleep", type=float, default=3.0,
                    help="Seconds to sleep between runs (default 3)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Only print what would be run; do not start training")

    args = ap.parse_args()

    runs_dir = Path(args.runs_dir)
    env_path = Path(args.env_path)

    if not runs_dir.exists():
        raise FileNotFoundError(f"runs-dir not found: {runs_dir}")
    if not env_path.exists():
        raise FileNotFoundError(f"env-path not found: {env_path}")

    run_folders = [p for p in runs_dir.iterdir() if p.is_dir()]
    run_folders.sort(key=lambda p: p.name)

    print(f"Found {len(run_folders)} run folders in: {runs_dir}")
    print(f"Will resume to max_steps={args.target_steps} using env={env_path}")

    for run_dir in run_folders:
        run_id = run_dir.name

        base_cfg_path = find_base_config(run_dir)
        if base_cfg_path is None:
            print(f"\n[SKIP] {run_id}: no config file found (used_config.yaml / configuration.yaml / *.yaml)")
            continue

        try:
            base_cfg = yaml.safe_load(base_cfg_path.read_text(encoding="utf-8"))
            behavior = choose_behavior_name(base_cfg, args.behavior)

            resume_cfg = make_resume_config(
                base_cfg=base_cfg,
                run_id=run_id,
                results_root=runs_dir,
                target_steps=args.target_steps,
                behavior=behavior,
            )

            resume_cfg_path = run_dir / f"resume_to_{args.target_steps}.yaml"
            resume_cfg_path.write_text(yaml.safe_dump(resume_cfg, sort_keys=False), encoding="utf-8")

            if args.dry_run:
                print(f"\n[DRY] Would resume {run_id} using {resume_cfg_path} (base={base_cfg_path.name})")
                continue

            run_resume(run_id, resume_cfg_path, env_path)
            time.sleep(args.sleep)

        except subprocess.CalledProcessError as e:
            print(f"\n[FAIL] {run_id}: mlagents-learn failed (continuing)\n{e}")
            continue
        except Exception as e:
            print(f"\n[FAIL] {run_id}: {e}")
            continue

    print("\nAll done.")


if __name__ == "__main__":
    main()
