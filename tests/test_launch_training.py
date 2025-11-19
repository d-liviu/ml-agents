"""
Test suite for utils/launch_training.py script.

This is the fancy training launcher that does all the cool stuff:
- Launches ML-Agents training
- Extracts metrics from logs (super useful!)
- Saves everything to CSV files
- Prints nice summaries

Way more sophisticated than run_all.py, but also way more to test ;D
"""

import pytest
import sys
import tempfile
import csv
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open, call
from collections import OrderedDict

# Add utils to path so we can import the module
# This is a bit janky but it works
sys.path.insert(0, str(Path(__file__).parent.parent / "utils"))

from launch_training import (
    sanitize_run_id,
    build_command,
    ensure_dir,
    extract_metrics_from_log,
    save_metrics,
    print_metrics_summary,
    main,
)


class TestSanitizeRunId:
    """
    Test class for sanitize_run_id function.

    This function makes run IDs safe for file systems and command lines.
    Super important because weird characters can break everything!
    """

    def test_basic_sanitization(self):
        """Test basic run-id sanitization with normal inputs."""
        # These should all work fine as-is
        assert sanitize_run_id("test-run") == "test-run"
        assert sanitize_run_id("test_run") == "test_run"
        assert sanitize_run_id("test123") == "test123"

    def test_remove_special_characters(self):
        """
        Test that special characters are replaced with dashes.

        Special chars like @, #, . can break file paths and commands.
        Let's make sure they get cleaned up!
        """
        assert sanitize_run_id("test@run#id") == "test-run-id", "Should replace @ and #"
        assert sanitize_run_id("test.run.id") == "test-run-id", "Should replace dots"
        assert sanitize_run_id("test run id") == "test-run-id", "Should replace spaces"

    def test_collapse_multiple_dashes(self):
        """
        Test that multiple consecutive dashes/underscores are collapsed.

        Having "test---run" looks ugly and can cause issues.
        Let's make it nice and clean: "test-run"
        """
        assert sanitize_run_id("test---run") == "test-run", "Should collapse multiple dashes"
        assert sanitize_run_id("test___run") == "test-run", "Should collapse multiple underscores"
        assert sanitize_run_id("test---___run") == "test-run", "Should collapse mixed dashes/underscores"

    def test_strip_whitespace(self):
        """
        Test that whitespace is stripped from edges.

        Nobody wants "  test-run  " when they can have "test-run"!
        """
        assert sanitize_run_id("  test-run  ") == "test-run", "Should strip spaces"
        assert sanitize_run_id("\ttest-run\n") == "test-run", "Should strip tabs and newlines"

    def test_empty_string_handling(self):
        """
        Test that empty strings return default value.

        What if someone passes an empty string? We need a fallback!
        """
        assert sanitize_run_id("") == "run", "Empty string should return 'run'"
        assert sanitize_run_id("   ") == "run", "Whitespace-only should return 'run'"

    def test_preserve_valid_characters(self):
        """
        Test that valid characters are preserved.

        Letters, numbers, dashes, and underscores are all fine.
        Let's make sure we don't break valid run IDs!
        """
        assert sanitize_run_id("Test-Run_123") == "Test-Run_123", "Should preserve valid chars"
        assert sanitize_run_id("worm_baseline_seed1") == "worm_baseline_seed1", "Should preserve underscores"


class TestBuildCommand:
    """
    Test class for build_command function.

    This builds the actual command that gets passed to mlagents-learn.
    Gotta make sure all the flags are in the right place!
    """

    def test_basic_command(self):
        """Test basic command building with minimal options."""
        config = Path("config/ppo/Worm.yaml")
        run_id = "test-run"

        cmd = build_command(
            config=config,
            run_id=run_id,
            env_path=None,
            no_graphics=False,
            time_scale=None,
            extra=[],
        )

        # Basic structure checks
        assert cmd[0] == "mlagents-learn", "Should start with mlagents-learn"
        assert str(config) in cmd, "Config path should be in command"
        assert "--run-id" in cmd, "Should include --run-id flag"
        assert run_id in cmd, "Run ID should be in command"

    def test_with_env_path(self):
        """
        Test command building with environment path.

        If you have a built Unity executable, you can specify it here.
        """
        config = Path("config/ppo/Worm.yaml")
        env_path = "Project/Worm.app"

        cmd = build_command(
            config=config,
            run_id="test",
            env_path=env_path,
            no_graphics=False,
            time_scale=None,
            extra=[],
        )

        assert "--env" in cmd, "Should include --env flag when env_path is provided"
        assert env_path in cmd, "Environment path should be in command"

    def test_with_no_graphics(self):
        """
        Test command building with no-graphics flag.

        No graphics = faster training (usually). Good for headless servers!
        """
        cmd = build_command(
            config=Path("config.yaml"),
            run_id="test",
            env_path=None,
            no_graphics=True,
            time_scale=None,
            extra=[],
        )

        assert "--no-graphics" in cmd, "Should include --no-graphics flag"

    def test_with_time_scale(self):
        """
        Test command building with time-scale.

        Time scale lets you speed up or slow down the simulation.
        Useful for testing or speeding up training!
        """
        cmd = build_command(
            config=Path("config.yaml"),
            run_id="test",
            env_path=None,
            no_graphics=False,
            time_scale=2.0,  # 2x speed!
            extra=[],
        )

        assert "--time-scale" in cmd, "Should include --time-scale flag"
        assert "2.0" in cmd, "Time scale value should be in command"

    def test_with_extra_args(self):
        """
        Test command building with extra arguments.

        Sometimes you need to pass extra flags. This handles that!
        """
        extra = ["--force", "--resume"]

        cmd = build_command(
            config=Path("config.yaml"),
            run_id="test",
            env_path=None,
            no_graphics=False,
            time_scale=None,
            extra=extra,
        )

        assert "--force" in cmd, "Extra args should be included"
        assert "--resume" in cmd, "All extra args should be included"


class TestEnsureDir:
    """
    Test class for ensure_dir function.

    Simple but useful: creates directories if they don't exist.
    No more "directory not found" errors!
    """

    def test_create_directory(self, tmp_path):
        """Test basic directory creation."""
        test_dir = tmp_path / "test_dir"
        ensure_dir(test_dir)

        assert test_dir.exists(), "Directory should exist after creation"
        assert test_dir.is_dir(), "Should be a directory, not a file"

    def test_create_nested_directory(self, tmp_path):
        """
        Test nested directory creation.

        What if you need to create "a/b/c" but "a" and "b" don't exist?
        This function handles that automatically!
        """
        test_dir = tmp_path / "level1" / "level2" / "level3"
        ensure_dir(test_dir)

        assert test_dir.exists(), "Nested directory should be created"
        assert test_dir.is_dir(), "Should be a directory"

    def test_existing_directory(self, tmp_path):
        """
        Test that existing directory doesn't cause errors.

        If the directory already exists, we shouldn't crash!
        """
        test_dir = tmp_path / "existing"
        test_dir.mkdir()

        # Should not raise an error
        ensure_dir(test_dir)
        assert test_dir.exists(), "Directory should still exist"


class TestExtractMetricsFromLog:
    """
    Test class for extract_metrics_from_log function.

    This is the cool part - extracting metrics from training logs!
    Parses through log files and finds all the important numbers.
    """

    def test_empty_log_file(self, tmp_path):
        """
        Test extraction from empty log file.

        What if the log file is empty? Shouldn't crash!
        """
        log_path = tmp_path / "empty.log"
        log_path.touch()  # Create empty file

        metrics = extract_metrics_from_log(log_path)

        # Should return empty metrics, not crash
        assert metrics["steps"] == []
        assert metrics["mean_reward"] == []
        assert metrics["final_summary"] == {}

    def test_nonexistent_log_file(self, tmp_path):
        """
        Test extraction from non-existent log file.

        File doesn't exist? That's okay, just return empty metrics.
        """
        log_path = tmp_path / "nonexistent.log"

        metrics = extract_metrics_from_log(log_path)

        # Should handle gracefully
        assert metrics["steps"] == []
        assert metrics["mean_reward"] == []

    def test_extract_step_number(self, tmp_path):
        """
        Test extraction of step numbers.

        Steps are like checkpoints in training. We need to track them!
        """
        log_path = tmp_path / "test.log"
        log_path.write_text("Step: 1000\nStep: 2000\n")

        metrics = extract_metrics_from_log(log_path)

        assert 1000 in metrics["steps"], "Should extract step 1000"
        assert 2000 in metrics["steps"], "Should extract step 2000"

    def test_extract_mean_reward(self, tmp_path):
        """
        Test extraction of mean reward.

        Reward is the most important metric! How well is the agent doing?
        """
        log_path = tmp_path / "test.log"
        log_path.write_text("Step: 1000\nMean Reward: 42.5\n")

        metrics = extract_metrics_from_log(log_path)

        assert len(metrics["mean_reward"]) > 0, "Should extract at least one reward"
        assert metrics["mean_reward"][0]["step"] == 1000, "Step should match"
        assert metrics["mean_reward"][0]["value"] == 42.5, "Reward value should match"

    def test_extract_episode_length(self, tmp_path):
        """
        Test extraction of episode length.

        Episode length tells us how long each training episode took.
        Shorter = usually better (agent learns faster)!
        """
        log_path = tmp_path / "test.log"
        log_path.write_text("Step: 1000\nMean Episode Length: 150.0\n")

        metrics = extract_metrics_from_log(log_path)

        assert len(metrics["episode_length"]) > 0, "Should extract episode length"
        assert metrics["episode_length"][0]["step"] == 1000
        assert metrics["episode_length"][0]["value"] == 150.0

    def test_extract_learning_rate(self, tmp_path):
        """
        Test extraction of learning rate.

        Learning rate is how fast the agent learns. Too high = unstable,
        too low = slow. Gotta find the sweet spot!
        """
        log_path = tmp_path / "test.log"
        log_path.write_text("Step: 1000\nLearning Rate: 3.0e-4\n")

        metrics = extract_metrics_from_log(log_path)

        assert len(metrics["learning_rate"]) > 0, "Should extract learning rate"
        assert metrics["learning_rate"][0]["value"] == 3.0e-4, "Should handle scientific notation"

    def test_final_summary_calculation(self, tmp_path):
        """
        Test final summary statistics calculation.

        At the end, we calculate some summary stats:
        - Total steps
        - Final reward
        - Max/min/avg rewards

        Super useful for comparing different training runs!
        """
        log_path = tmp_path / "test.log"
        log_path.write_text(
            "Step: 1000\nMean Reward: 10.0\n"
            "Step: 2000\nMean Reward: 20.0\n"
            "Step: 3000\nMean Reward: 30.0\n"
        )

        metrics = extract_metrics_from_log(log_path)
        summary = metrics["final_summary"]

        # Check all the summary stats
        assert summary["total_steps"] == 3000, "Should track total steps"
        assert summary["final_mean_reward"] == 30.0, "Final reward should be last value"
        assert summary["max_reward"] == 30.0, "Max should be highest value"
        assert summary["min_reward"] == 10.0, "Min should be lowest value"
        assert summary["avg_reward"] == 20.0, "Avg should be average of all values"


class TestSaveMetrics:
    """
    Test class for save_metrics function.

    Once we extract metrics, we need to save them somewhere!
    CSV files are perfect for this - easy to open in Excel or analyze.
    """

    def test_save_metrics_to_csv(self, tmp_path):
        """
        Test saving metrics to CSV file.

        CSV format is nice because you can open it in Excel,
        Python, R, or whatever tool you prefer!
        """
        metrics = {
            "steps": [1000, 2000],
            "mean_reward": [
                {"step": 1000, "value": 10.0},
                {"step": 2000, "value": 20.0},
            ],
            "episode_length": [
                {"step": 1000, "value": 100.0},
            ],
            "final_summary": {
                "total_steps": 2000,
                "final_mean_reward": 20.0,
            },
        }

        output_path = tmp_path / "metrics.csv"
        save_metrics(metrics, output_path)

        assert output_path.exists(), "CSV file should be created"

        # Let's verify the CSV content is correct
        with open(output_path, "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            assert len(rows) == 2, "Should have 2 rows of data"
            assert rows[0]["step"] == "1000", "First row step should be 1000"
            assert rows[0]["mean_reward"] == "10.0", "First row reward should be 10.0"

    def test_save_summary_file(self, tmp_path):
        """
        Test that summary file is created.

        We save two files:
        1. The main metrics CSV (all the data)
        2. A summary CSV (just the final stats)

        The summary is super useful for quick comparisons!
        """
        metrics = {
            "steps": [1000],
            "mean_reward": [{"step": 1000, "value": 10.0}],
            "final_summary": {
                "total_steps": 1000,
                "final_mean_reward": 10.0,
            },
        }

        output_path = tmp_path / "metrics.csv"
        save_metrics(metrics, output_path)

        # Check that summary file was also created
        summary_path = tmp_path / "metrics.summary.csv"
        assert summary_path.exists(), "Summary file should be created automatically"


class TestPrintMetricsSummary:
    """
    Test class for print_metrics_summary function.

    Pretty printing is important! Makes it easy to see results at a glance.
    """

    def test_print_with_summary(self, capsys):
        """
        Test printing metrics summary.

        When training finishes, we want to see a nice summary printed out.
        Makes it easy to see how well the training went!
        """
        metrics = {
            "final_summary": {
                "total_steps": 1000,
                "final_mean_reward": 42.5,
                "max_reward": 50.0,
                "min_reward": 10.0,
                "avg_reward": 30.0,
            },
            "mean_reward": [{"step": 1000, "value": 42.5}],
        }

        print_metrics_summary(metrics)
        captured = capsys.readouterr()

        # Check that the summary was printed
        assert "TRAINING METRICS SUMMARY" in captured.out, "Should print header"
        assert "1000" in captured.out, "Should print total steps"
        assert "42.5" in captured.out, "Should print final reward"

    def test_print_without_summary(self, capsys):
        """
        Test printing when no summary is available.

        What if there's no data? Should print a helpful message, not crash!
        """
        metrics = {"final_summary": {}}

        print_metrics_summary(metrics)
        captured = capsys.readouterr()

        assert "No metrics extracted" in captured.out, "Should print helpful message"


class TestMain:
    """
    Test class for main function.

    This is the big one - the main entry point!
    Tests all the different ways the script can be called.
    """

    @patch('launch_training.subprocess.Popen')
    @patch('launch_training.Path.exists')
    def test_main_with_dry_run(self, mock_exists, mock_popen):
        """
        Test main function with dry-run flag.

        Dry-run is super useful! Shows you what command would run
        without actually running it. Great for debugging!
        """
        mock_exists.return_value = True

        argv = [
            "config/ppo/Worm.yaml",
            "--dry-run",  # Just show, don't run
        ]

        exit_code = main(argv)

        assert exit_code == 0, "Dry-run should succeed"
        # Should not actually run subprocess
        mock_popen.assert_not_called(), "Dry-run shouldn't execute anything"

    @patch('launch_training.subprocess.Popen')
    @patch('launch_training.Path.exists')
    def test_main_with_nonexistent_config(self, mock_exists):
        """
        Test main function with non-existent config file.

        What if someone passes a config file that doesn't exist?
        Should fail gracefully with a helpful error code!
        """
        mock_exists.return_value = False

        argv = ["nonexistent.yaml"]

        exit_code = main(argv)

        assert exit_code == 2, "Should return error code 2 for missing config"

    @patch('launch_training.subprocess.Popen')
    @patch('launch_training.Path.exists')
    @patch('launch_training.open', new_callable=mock_open)
    def test_main_successful_run(self, mock_file, mock_exists, mock_popen):
        """
        Test successful training run.

        The happy path! Everything works, training completes successfully.
        This is what we hope happens every time! 🤞
        """
        mock_exists.return_value = True
        mock_process = MagicMock()
        mock_process.wait.return_value = 0  # Success!
        mock_popen.return_value = mock_process

        argv = [
            "config/ppo/Worm.yaml",
            "--run-id", "test-run",
        ]

        exit_code = main(argv)

        assert exit_code == 0, "Should return success code"
        mock_popen.assert_called_once(), "Should call subprocess once"

    @patch('launch_training.subprocess.Popen')
    @patch('launch_training.Path.exists')
    @patch('launch_training.open', new_callable=mock_open)
    def test_main_with_metrics_extraction(self, mock_file, mock_exists, mock_popen):
        """
        Test main function with metrics extraction.

        After training finishes, we can extract metrics from the log.
        Super useful for analyzing results!
        """
        mock_exists.return_value = True
        mock_process = MagicMock()
        mock_process.wait.return_value = 0
        mock_popen.return_value = mock_process

        # Create a mock log file with some metrics
        log_content = "Step: 1000\nMean Reward: 42.5\n"
        mock_file.return_value.read.return_value = log_content

        argv = [
            "config/ppo/Worm.yaml",
            "--run-id", "test-run",
            "--extract-metrics",  # Extract metrics after training
        ]

        exit_code = main(argv)

        assert exit_code == 0, "Should succeed even with metrics extraction"

    @patch('launch_training.subprocess.Popen')
    @patch('launch_training.Path.exists')
    def test_main_file_not_found_error(self, mock_exists):
        """
        Test handling of FileNotFoundError (mlagents-learn not found).

        What if mlagents-learn isn't installed or not in PATH?
        Should give a helpful error message!
        """
        mock_exists.return_value = True

        with patch('launch_training.subprocess.Popen', side_effect=FileNotFoundError()):
            argv = ["config/ppo/Worm.yaml"]
            exit_code = main(argv)
            assert exit_code == 127, "Should return 127 for command not found"
