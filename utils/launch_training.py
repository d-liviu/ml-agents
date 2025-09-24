# Utility to launch ML-Agents training with configurable run-id, logs, and error handling.

import argparse
import datetime
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

# "Cleans" the provided run-id so it's safe for files and the command line

def sanitize_run_id(run_id: str) -> str:
	"""Make run-id filesystem and CLI friendly."""
	# Allow letters, numbers, dash and underscore; replace others with dash
	safe = re.sub(r"[^A-Za-z0-9_-]", "-", run_id.strip())
	# Collapse multiple dashes/underscores
	safe = re.sub(r"[-_]{2,}", "-", safe)
	return safe or "run"

# Build the final `mlagents-learn` command based on provided options

def build_command(
	config: Path,
	run_id: str,
	env_path: str | None,
	no_graphics: bool,
	time_scale: float | None,
	extra: list[str],
) -> list[str]:
	cmd: list[str] = [
		"mlagents-learn",
		str(config),
		"--run-id",
		run_id,
	]
	if env_path:
		cmd += ["--env", env_path]
	if no_graphics:
		cmd.append("--no-graphics")
	if time_scale is not None:
		cmd += ["--time-scale", str(time_scale)]
	# Allow passing any extra CLI flags transparently (after `--`)
	cmd += extra
	return cmd

# Ensure a directory exists (creates parents as needed)

def ensure_dir(path: Path) -> None:
	path.mkdir(parents=True, exist_ok=True)


def main(argv: list[str]) -> int:
	# Define and parse CLI arguments for this wrapper
	parser = argparse.ArgumentParser(
		description=(
			"Launch ML-Agents training with configurable run-id, structured logs, and error handling."
		)
	)
	parser.add_argument(
		"config",
		type=Path,
		help="Path to the trainer config YAML (e.g., config/ppo/3DBall.yaml)",
	)
	parser.add_argument(
		"--run-id",
		dest="run_id",
		type=str,
		default=None,
		help="Run identifier (defaults to timestamp, e.g., 2025-09-22_14-30-00)",
	)
	parser.add_argument(
		"--env",
		dest="env_path",
		type=str,
		default=None,
		help=(
			"Path to the built Unity environment executable. If omitted, uses editor or registry."
		),
	)
	parser.add_argument(
		"--no-graphics",
		action="store_true",
		help="Run in no-graphics mode",
	)
	parser.add_argument(
		"--time-scale",
		dest="time_scale",
		type=float,
		default=None,
		help="Engine time scale (if supported)",
	)
	parser.add_argument(
		"--logs-dir",
		dest="logs_dir",
		type=Path,
		default=Path("training_logs"),
		help="Directory where log files will be written",
	)
	parser.add_argument(
		"--results-dir",
		dest="results_dir",
		type=Path,
		default=Path("results"),
		help=(
			"Directory passed via ML-Agents RESULTS_DIR env var (controls where summaries/checkpoints go)"
		),
	)
	parser.add_argument(
		"--dry-run",
		action="store_true",
		help="Print the resolved command and exit",
	)
	parser.add_argument(
		"--merge-stderr",
		action="store_true",
		help="Merge stderr into stdout log (single file)",
	)
	parser.add_argument(
		"extra",
		nargs=argparse.REMAINDER,
		help=(
			"Additional args passed through to mlagents-learn verbatim. Use `-- extra --your-flag`"
		),
	)

	args = parser.parse_args(argv)

	if not args.config.exists():
		print(f"Error: config file not found: {args.config}", file=sys.stderr)
		return 2

	# Derive a default run-id from the current timestamp if not provided
	timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
	run_id = sanitize_run_id(args.run_id or timestamp)

	# Prepare output directories for logs and ML-Agents results
	ensure_dir(args.logs_dir)
	ensure_dir(args.results_dir)

	# Define log file paths
	stdout_log = args.logs_dir / f"{run_id}.out.log"
	stderr_log = stdout_log if args.merge_stderr else args.logs_dir / f"{run_id}.err.log"

	# Build the final training command
	cmd = build_command(
		config=args.config,
		run_id=run_id,
		env_path=args.env_path,
		no_graphics=args.no_graphics,
		time_scale=args.time_scale,
		extra=args.extra or [],
	)

	# Print a short summary of what will run (helpful for reproducibility)
	print("Launching ML-Agents training:")
	print(f"  run-id:       {run_id}")
	print(f"  config:       {args.config}")
	if args.env_path:
		print(f"  env:          {args.env_path}")
	print(f"  logs:         {stdout_log} ({'merged' if args.merge_stderr else 'split stderr'})")
	print(f"  results dir:  {args.results_dir}")
	print(f"  command:      {shlex.join(cmd)}")

	# Optional: just show the command and exit
	if args.dry_run:
		return 0

	# Prepare environment for the child process; RESULTS_DIR controls output location
	env = os.environ.copy()
	env["RESULTS_DIR"] = str(args.results_dir)

	try:
		# Stream stdout/stderr from the training process directly into log files
		with open(stdout_log, "w", encoding="utf-8", buffering=1) as out_f:
			stderr_target = subprocess.STDOUT if args.merge_stderr else open(
				stderr_log, "w", encoding="utf-8", buffering=1
			)
			try:
				process = subprocess.Popen(
					cmd,
					stdout=out_f,
					stderr=stderr_target,
					env=env,
				)
				exit_code = process.wait()
			finally:
				# Close the dedicated stderr file if it was opened
				if stderr_target is not subprocess.STDOUT:
					stderr_target.close()
	except FileNotFoundError:
		# The `mlagents-learn` executable was not found on PATH
		print(
			"Error: 'mlagents-learn' command not found. Ensure ML-Agents is installed and on PATH (e.g., `pip install mlagents`).",
			file=sys.stderr,
		)
		return 127
	except Exception as exc:  # noqa: BLE001
		# Any other unexpected error while launching or running
		print(f"Training failed with unexpected error: {exc}", file=sys.stderr)
		return 1

	# Report final outcome and return the child's exit code
	if exit_code == 0:
		print(f"Training completed successfully. Logs: {stdout_log}")
	else:
		print(
			f"Training exited with code {exit_code}. See logs: {stdout_log}{'' if args.merge_stderr else f' and {stderr_log}'}",
			file=sys.stderr,
		)
	return exit_code


if __name__ == "__main__":
	sys.exit(main(sys.argv[1:])) 