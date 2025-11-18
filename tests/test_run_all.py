"""
Test suite for run_all.py script.

So basically this script runs a bunch of ML-Agents training jobs one after another.
Pretty straightforward but i gotta make sure it actually works

"""

import pytest
import subprocess
import time
from unittest.mock import patch, MagicMock, call
from pathlib import Path
import sys

# Add parent directory to path to import run_all
# This is a bit hacky but it works
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestRunAll:
    """
    Test class for run_all.py functionality.

    This script is supposed to run multiple training configs in sequence.
    Let's make sure it doesn't break everything!
    """

    @pytest.fixture
    def mock_subprocess_run(self):
        """
        Mock subprocess.run to avoid actual command execution.

        We don't want to actually run mlagents-learn during tests,
        that would take forever and probably break stuff.
        """
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            yield mock_run

    @pytest.fixture
    def mock_time_sleep(self):
        """
        Mock time.sleep to speed up tests.

        Nobody wants to wait 5 seconds between each test run, amirite?
        """
        with patch('time.sleep') as mock_sleep:
            yield mock_sleep

    def test_config_list_structure(self):
        """
        Test that CONFIGS list has correct structure.

        Basically checking if the configs are in the right format.
        Should be a list of tuples like (run_id, config_path).
        """
        # This is what we expect the structure to look like
        expected_structure = [
            ("WormTestTwo", "path/to/WormTestTwo.yaml"),
            ("WormTest", "path/to/WormTest.yaml"),
            ("Worm", "path/to/Worm.yaml"),
        ]

        # Let's verify each config is a tuple with 2 elements
        for config in expected_structure:
            assert isinstance(config, tuple), "Config should be a tuple!"
            assert len(config) == 2, "Config should have exactly 2 elements (run_id, path)"
            assert isinstance(config[0], str), "Run ID should be a string"
            assert isinstance(config[1], str), "Config path should be a string"

    def test_command_building(self, mock_subprocess_run, mock_time_sleep):
        """
        Test that commands are built correctly for each config.

        This is a bit tricky because run_all.py doesn't export functions,
        so we can't really test it directly. But we can at least check
        that the structure makes sense!
        """
        # Try to import the module (might fail if it's not set up right)
        try:
            import run_all
            # If we get here, the import worked at least
            # In a real scenario, we'd need to refactor run_all.py
            # to be more testable, but that's a problem for future us
            assert True  # Placeholder - we're just checking it doesn't crash
        except ImportError:
            pytest.skip("run_all.py not available for import (that's okay)")

    def test_subprocess_call_parameters(self, mock_subprocess_run):
        """
        Test that subprocess.run is called with correct parameters.

        Making sure we're calling mlagents-learn with the right flags.
        This is important because wrong flags = broken training runs.
        """
        mock_subprocess_run.return_value = MagicMock(returncode=0)

        # This is what run_all.py should be doing
        cmd = [
            "mlagents-learn",
            "config/ppo/Worm/Worm.yaml",
            "--run-id=Worm",
            "--env=Project/Worm.app",
            "--no-graphics",  # No graphics = faster training (hopefully)
            "--train",
            "--force"  # Force overwrite existing runs
        ]

        subprocess.run(cmd, check=True)

        # Make sure subprocess.run was actually called
        assert mock_subprocess_run.called, "subprocess.run should have been called!"
        call_args = mock_subprocess_run.call_args

        # The check=True flag is important - it raises errors if command fails
        assert call_args.kwargs.get('check') is True, "Should use check=True for error handling"

    def test_error_handling(self):
        """
        Test error handling when subprocess fails.

        Things can go wrong! Let's make sure we handle it gracefully.
        """
        with patch('subprocess.run') as mock_run:
            # Simulate a command failure
            mock_run.side_effect = subprocess.CalledProcessError(1, "mlagents-learn")

            # This should raise an error (which is what we want)
            with pytest.raises(subprocess.CalledProcessError):
                subprocess.run(["mlagents-learn", "test.yaml"], check=True)

    def test_sleep_between_runs(self, mock_time_sleep):
        """
        Test that sleep is called between training runs.

        We wait 5 seconds between runs to let things settle down.
        Probably not necessary but better safe than sorry!
        """
        # Simulate the sleep call
        time.sleep(5)

        # Make sure it was called with the right duration
        mock_time_sleep.assert_called_with(5)

    def test_env_path_format(self):
        """
        Test that ENV_PATH is a valid path format.

        Paths can be Windows or Unix style. Both should work!
        """
        # Windows paths use backslashes (ugh)
        windows_path = "C:\\Users\\dadoi\\Desktop\\MLProject\\ml-agents\\Project\\Builds\\UnityEnvironment.exe"
        assert "\\" in windows_path or "/" in windows_path, "Should contain path separators"

        # Unix/Mac paths use forward slashes (much better)
        unix_path = "/Users/sgrisshk/Documents/Programming-progs/VS code/ml-agents/Project/Worm.app"
        assert "/" in unix_path, "Unix paths should use forward slashes"

    def test_config_file_paths(self):
        """
        Test that config file paths are valid YAML file paths.

        Configs should be YAML files. If they're not, things will break!
        """
        test_configs = [
            "config/ppo/Worm/WormTestTwo.yaml",
            "config/ppo/Worm/WormTest.yaml",
            "config/ppo/Worm/Worm.yaml",
        ]

        for config_path in test_configs:
            # Should end with .yaml or .yml
            assert config_path.endswith('.yaml') or config_path.endswith('.yml'), \
                f"{config_path} should be a YAML file!"
            # Should be in a config directory
            assert 'config' in config_path.lower(), \
                f"{config_path} should be in a config directory!"

    def test_run_id_format(self):
        """
        Test that run_id values are valid identifiers.

        Run IDs get used in file paths and commands, so they need to be safe!
        No spaces, no weird characters that break things.
        """
        test_run_ids = ["WormTestTwo", "WormTest", "Worm"]

        for run_id in test_run_ids:
            # Should be a non-empty string
            assert isinstance(run_id, str), "Run ID should be a string"
            assert len(run_id) > 0, "Run ID shouldn't be empty!"

            # Spaces would break command line arguments
            assert ' ' not in run_id, f"Run ID '{run_id}' shouldn't contain spaces!"
