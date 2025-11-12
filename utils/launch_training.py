# Utility to launch ML-Agents training with configurable run-id, logs, and error handling.

import argparse
import csv
import datetime
import os
import re
import shlex
import subprocess
import sys
from collections import OrderedDict
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

def extract_metrics_from_log(log_path: Path) -> dict:

	metrics = {
		"steps": [],
		"mean_reward": [],
		"std_reward": [],
		"episode_length": [],
		"learning_rate": [],
		"entropy": [],
		"value_loss": [],
		"policy_loss": [],
		"final_summary": {}
	}
	
	if not log_path.exists():
		return metrics
	
# Regex patterns for common ML-Agents metrics
	step_pattern = re.compile(r"Step:\s*(\d+)")
	mean_reward_pattern = re.compile(r"Mean Reward:\s*([-+]?\d*\.?\d+)")
	std_reward_pattern = re.compile(r"Std of Reward:\s*([-+]?\d*\.?\d+)")
	episode_length_pattern = re.compile(r"Mean Episode Length:\s*([-+]?\d*\.?\d+)")
	lr_pattern = re.compile(r"Learning Rate:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")
	entropy_pattern = re.compile(r"Entropy:\s*([-+]?\d*\.?\d+)")
	value_loss_pattern = re.compile(r"Value Loss:\s*([-+]?\d*\.?\d+)")
	policy_loss_pattern = re.compile(r"Policy Loss:\s*([-+]?\d*\.?\d+)")
	
	try:
		with open(log_path, "r", encoding="utf-8") as f:
			current_step = None
			
			for line in f:
				# Extract step number
				step_match = step_pattern.search(line)
				if step_match:
					current_step = int(step_match.group(1))
					metrics["steps"].append(current_step)
				
				# Extract mean reward
				mean_reward_match = mean_reward_pattern.search(line)
				if mean_reward_match and current_step is not None:
					metrics["mean_reward"].append({
						"step": current_step,
						"value": float(mean_reward_match.group(1))
					})
				
				# Extract std reward
				std_reward_match = std_reward_pattern.search(line)
				if std_reward_match and current_step is not None:
					metrics["std_reward"].append({
						"step": current_step,
						"value": float(std_reward_match.group(1))
					})
				
				# Extract episode length
				ep_len_match = episode_length_pattern.search(line)
				if ep_len_match and current_step is not None:
					metrics["episode_length"].append({
						"step": current_step,
						"value": float(ep_len_match.group(1))
					})
				
				# Extract learning rate
				lr_match = lr_pattern.search(line)
				if lr_match and current_step is not None:
					metrics["learning_rate"].append({
						"step": current_step,
						"value": float(lr_match.group(1))
					})
				
				# Extract entropy
				entropy_match = entropy_pattern.search(line)
				if entropy_match and current_step is not None:
					metrics["entropy"].append({
						"step": current_step,
						"value": float(entropy_match.group(1))
					})
				
				# Extract value loss
				value_loss_match = value_loss_pattern.search(line)
				if value_loss_match and current_step is not None:
					metrics["value_loss"].append({
						"step": current_step,
						"value": float(value_loss_match.group(1))
					})
				
				# Extract policy loss
				policy_loss_match = policy_loss_pattern.search(line)
				if policy_loss_match and current_step is not None:
					metrics["policy_loss"].append({
						"step": current_step,
						"value": float(policy_loss_match.group(1))
					})
		
		# Calculate summary statistics
		if metrics["mean_reward"]:
			rewards = [m["value"] for m in metrics["mean_reward"]]
			metrics["final_summary"] = {
				"total_steps": metrics["steps"][-1] if metrics["steps"] else 0,
				"final_mean_reward": rewards[-1] if rewards else None,
				"max_reward": max(rewards) if rewards else None,
				"min_reward": min(rewards) if rewards else None,
				"avg_reward": sum(rewards) / len(rewards) if rewards else None,
			}
	
	except Exception as e:
		print(f"Warning: Error parsing log file: {e}", file=sys.stderr)
	
	return metrics

# saves metrics to CSV files
def save_metrics(metrics: dict, output_path: Path) -> None:
	try:
		fieldnames = [
			"step",
			"mean_reward",
			"std_reward",
			"episode_length",
			"learning_rate",
			"entropy",
			"value_loss",
			"policy_loss",
		]

		rows: "OrderedDict[int, dict[str, float | int]]" = OrderedDict()

		for step in metrics.get("steps", []):
			rows.setdefault(step, {"step": step})

		def fill_metric(key: str) -> None:
			for entry in metrics.get(key, []):
				step = entry.get("step")
				if step is None:
					continue
				row = rows.setdefault(step, {"step": step})
				row[key] = entry.get("value")

		for metric_key in fieldnames[1:]:
			fill_metric(metric_key)

		ensure_dir(output_path.parent)

		with open(output_path, "w", encoding="utf-8", newline="") as csv_file:
			writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
			writer.writeheader()
			for row in rows.values():
				writer.writerow({field: row.get(field, "") for field in fieldnames})

		print(f"Metrics saved to: {output_path}")

		summary = metrics.get("final_summary") or {}
		if summary:
			summary_path = output_path.with_name(f"{output_path.stem}.summary.csv")
			with open(summary_path, "w", encoding="utf-8", newline="") as csv_file:
				writer = csv.writer(csv_file)
				writer.writerow(["metric", "value"])
				for key, value in summary.items():
					writer.writerow([key, value if value is not None else ""])
			print(f"Summary saved to: {summary_path}")
	except Exception as e:
		print(f"Warning: Could not save metrics: {e}", file=sys.stderr)

# helper for terminal print
def print_metrics_summary(metrics: dict) -> None:
	summary = metrics.get("final_summary", {})
	if not summary:
		print("\nNo metrics extracted from log.")
		return
	
	print("\n" + "="*50)
	print("TRAINING METRICS SUMMARY")
	print("="*50)
	print(f"Total Steps:       {summary.get('total_steps', 'N/A')}")
	print(f"Final Mean Reward: {summary.get('final_mean_reward', 'N/A'):.4f}" if summary.get('final_mean_reward') is not None else "Final Mean Reward:  N/A")
	print(f"Max Reward:        {summary.get('max_reward', 'N/A'):.4f}" if summary.get('max_reward') is not None else "Max Reward:         N/A")
	print(f"Min Reward:        {summary.get('min_reward', 'N/A'):.4f}" if summary.get('min_reward') is not None else "Min Reward:         N/A")
	print(f"Avg Reward:        {summary.get('avg_reward', 'N/A'):.4f}" if summary.get('avg_reward') is not None else "Avg Reward:         N/A")
	print(f"Data Points:       {len(metrics.get('mean_reward', []))}")
	print("="*50 + "\n")


def main(argv: list[str]) -> int:
	# Define and parse CLI arguments for this wrapper
	parser = argparse.ArgumentParser(
		description=(
			"Launch ML-Agents training with configurable run-id, structured logs, error handling and metric extraction"
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
		"--extract-metrics",
		action="store_true",
		help="Extract metrics from log after training completes",
	)
	parser.add_argument(
		"--metrics-output",
		dest="metrics_output",
		type=Path,
		default=None,
		help="Path to save extracted metrics CSV (defaults to logs-dir/RUN_ID.metrics.csv)",
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
	metrics_output = args.metrics_output or args.logs_dir / f"{run_id}.metrics.csv"

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
	if args.extract_metrics:
		print(f"  metrics out:  {metrics_output}")
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
		if args.extract_metrics:
			print("\nExtracting metrics from training log...")
			metrics = extract_metrics_from_log(stdout_log)
			save_metrics(metrics, metrics_output)
			print_metrics_summary(metrics)
	else:
		print(
			f"Training exited with code {exit_code}. See logs: {stdout_log}{'' if args.merge_stderr else f' and {stderr_log}'}",
			file=sys.stderr,
		)
	return exit_code


if __name__ == "__main__":
	sys.exit(main(sys.argv[1:])) 

	# To run the wrapper with mlagents-learn
	# python utils\launch_training.py --run-id worm-editor --extract-metrics config\ppo\Worm.yaml