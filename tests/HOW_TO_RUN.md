# How to Run Tests 

Quick guide on running the test suite!

## Prerequisites

First, make sure pytest is installed:

```bash
# Activate your conda environment
conda activate ml-agents-py310

# Install pytest (if not already installed)
pip install pytest pytest-cov
```

## Running Tests

**Important:** The project has a `conftest.py` that requires ML-Agents to be installed. 
If you get import errors, use `--noconftest` flag to skip it (our tests don't need it).

### Run all tests
```bash
# From the project root directory
conda activate ml-agents-py310
cd /Users/sgrisshk/Documents/Programming-progs/VS\ code/ml-agents

# Run with --noconftest to avoid ML-Agents dependencies
pytest tests/ --noconftest -v
```

### Run specific test file
```bash
# Test run_all.py
pytest tests/test_run_all.py --noconftest -v

# Test launch_training.py
pytest tests/test_launch_training.py --noconftest -v
```

### Run specific test class
```bash
# Test only TestSanitizeRunId class
pytest tests/test_launch_training.py::TestSanitizeRunId

# Test only TestBuildCommand class
pytest tests/test_launch_training.py::TestBuildCommand
```

### Run specific test function
```bash
# Test a single function
pytest tests/test_launch_training.py::TestSanitizeRunId::test_basic_sanitization
```

## Useful Options

### Verbose output (see what's happening)
```bash
pytest tests/ -v
```

### Very verbose (show print statements)
```bash
pytest tests/ -vv -s
```

### Run tests in parallel (faster!)
```bash
pip install pytest-xdist
pytest tests/ -n auto
```

### Show coverage (how much code is tested)
```bash
pytest tests/ --cov=utils --cov=run_all --cov-report=html
# Then open htmlcov/index.html in browser
```

### Stop on first failure
```bash
pytest tests/ -x
```

### Run only failed tests from last run
```bash
pytest tests/ --lf
```

### Run with debugger on failure
```bash
pytest tests/ --pdb
```

## Examples

### Quick test run
```bash
conda activate ml-agents-py310
cd /Users/sgrisshk/Documents/Programming-progs/VS\ code/ml-agents
pytest tests/ --noconftest -v
```

### Full test suite with coverage
```bash
conda activate ml-agents-py310
cd /Users/sgrisshk/Documents/Programming-progs/VS\ code/ml-agents
pytest tests/ -v --cov=utils --cov=run_all --cov-report=term-missing
```

### Test only launch_training functions
```bash
conda activate ml-agents-py310
cd /Users/sgrisshk/Documents/Programming-progs/VS\ code/ml-agents
pytest tests/test_launch_training.py -v
```

## Troubleshooting

### If tests fail with import errors:
```bash
# Make sure you're in the project root
cd /Users/sgrisshk/Documents/Programming-progs/VS\ code/ml-agents

# Check Python path
python -c "import sys; print(sys.path)"
```

### If pytest is not found:
```bash
# Install in your conda environment
conda activate ml-agents-py310
pip install pytest
```

### If you get module not found errors:
```bash
# Make sure utils directory is accessible
ls utils/launch_training.py

# Check if run_all.py exists
ls run_all.py
```


If tests fail, pytest will show you exactly what went wrong and where


