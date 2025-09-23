# Unity ML-Agents Toolkit - Complete Setup and Usage Guide

[![docs badge](https://img.shields.io/badge/docs-reference-blue.svg)](https://github.com/Unity-Technologies/ml-agents/tree/release_21_docs/docs/)
[![license badge](https://img.shields.io/badge/license-Apache--2.0-green.svg)](LICENSE.md)

**The Unity Machine Learning Agents Toolkit** (ML-Agents) is an open-source project that enables games and simulations to serve as environments for training intelligent agents. This comprehensive guide will walk you through the complete setup process and show you how to get started with training your first AI agents.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Project Structure](#project-structure)
- [Installation Guide](#installation-guide)
  - [Python Environment Setup](#python-environment-setup)
  - [Installing ML-Agents Packages](#installing-ml-agents-packages)
  - [Unity Package Installation](#unity-package-installation)
- [Quick Start](#quick-start)
- [Training Your First Agent](#training-your-first-agent)
- [Configuration Files](#configuration-files)
- [Available Training Algorithms](#available-training-algorithms)
- [Running Examples](#running-examples)
- [Advanced Usage](#advanced-usage)
- [Troubleshooting](#troubleshooting)
- [Additional Resources](#additional-resources)

## Prerequisites

Before you begin, ensure you have the following installed on your system:

### System Requirements
- **Operating System**: Windows 10/11, macOS 10.15+, or Linux (Ubuntu 18.04+)
- **Python**: 3.10.1 to 3.10.12 (required for ML-Agents)
- **Unity**: 2020.1 or later (recommended: Unity 2022.3 LTS)
- **Git**: For cloning the repository

### Hardware Recommendations
- **CPU**: Multi-core processor (8+ cores recommended for faster training)
- **RAM**: 8GB minimum, 16GB+ recommended
- **GPU**: NVIDIA GPU with CUDA support (optional but highly recommended for faster training)
- **Storage**: At least 5GB free space

## Project Structure

This repository contains several key components:

```
ml-agents/
├── ml-agents/                    # Main Python training package
├── ml-agents-envs/              # Python environment interface
├── com.unity.ml-agents/         # Unity package (C# SDK)
├── com.unity.ml-agents.extensions/ # Additional Unity components
├── config/                      # Training configuration files
│   ├── ppo/                     # PPO algorithm configs
│   ├── sac/                     # SAC algorithm configs
│   ├── poca/                    # MA-POCA algorithm configs
│   └── imitation/               # Imitation learning configs
├── docs/                        # Comprehensive documentation
├── Project/                     # Example Unity project
├── DevProject/                  # Development Unity project
└── PerformanceProject/          # Performance testing project
```

## Installation Guide

### Python Environment Setup

1. **Create a Virtual Environment** (Recommended):
   ```bash
   # Using venv
   python -m venv ml-agents-env
   
   # Activate the environment
   # On Windows:
   ml-agents-env\Scripts\activate
   # On macOS/Linux:
   source ml-agents-env/bin/activate
   ```

2. **Alternative: Using Conda**:
   ```bash
   conda create -n ml-agents python=3.10
   conda activate ml-agents
   ```

### Installing ML-Agents Packages

#### Option 1: Install from PyPI (Recommended for most users)

```bash
# Install the main ML-Agents package
pip install mlagents==1.0.0

# This automatically installs mlagents_envs as a dependency
```

#### Option 2: Install from Source (For development)

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Unity-Technologies/ml-agents.git
   cd ml-agents
   ```

2. **Install in development mode**:
   ```bash
   # Install mlagents_envs first
   cd ml-agents-envs
   pip install -e .
   
   # Install mlagents
   cd ../ml-agents
   pip install -e .
   ```

3. **Install additional dependencies** (if needed):
   ```bash
   # For testing
   pip install -r test_requirements.txt
   
   # For Jupyter notebooks
   pip install -r colab_requirements.txt
   ```

### Unity Package Installation

1. **Open Unity Hub** and create a new project or open an existing one

2. **Install ML-Agents Package**:
   - Open **Window → Package Manager**
   - Click the **+** button and select **Add package from git URL**
   - Enter: `https://github.com/Unity-Technologies/ml-agents.git?path=/com.unity.ml-agents`
   - Click **Add**

3. **Alternative: Manual Installation**:
   - Download the package from the [Unity Asset Store](https://assetstore.unity.com/packages/tools/ai/ml-agents-1-0-0-179262)
   - Import the package into your Unity project

4. **Verify Installation**:
   - In Unity, go to **Window → ML-Agents → Training Configuration**
   - If you see the ML-Agents menu, the installation was successful

## Quick Start

### 1. Test Your Installation

```bash
# Verify ML-Agents is installed correctly
mlagents-learn --help

# Check available training algorithms
mlagents-learn --list-algorithms
```

### 2. Run the 3D Ball Example

The 3D Ball environment is included as a sample and is perfect for testing your setup:

```bash
# Navigate to the Project directory
cd Project

# Build the Unity executable (if not already built)
# In Unity: File → Build Settings → Build

# Start training with PPO
mlagents-learn config/ppo/3DBall.yaml --run-id=3DBall_test
```

## Training Your First Agent

### Step 1: Prepare Your Unity Environment

1. **Open the Project**:
   - Open Unity Hub
   - Open the `Project` folder from this repository

2. **Configure Your Agent**:
   - Select your agent GameObject
   - Add the `Behavior Parameters` component
   - Configure observations, actions, and rewards

3. **Build the Environment**:
   - Go to **File → Build Settings**
   - Add your scene to the build
   - Choose your target platform
   - Click **Build** and save the executable

### Step 2: Create a Training Configuration

Create a YAML configuration file (or use existing ones in the `config/` directory):

```yaml
# Example: config/my_agent.yaml
behaviors:
  MyAgent:
    trainer_type: ppo
    hyperparameters:
      batch_size: 64
      buffer_size: 12000
      learning_rate: 3.0e-4
      beta: 5.0e-3
      epsilon: 0.2
      lambd: 0.95
      num_epoch: 3
      learning_rate_schedule: linear
    network_settings:
      normalize: true
      hidden_units: 128
      num_layers: 2
    reward_signals:
      extrinsic:
        gamma: 0.99
        strength: 1.0
    keep_checkpoints: 5
    max_steps: 500000
    time_horizon: 1000
    summary_freq: 12000
```

### Step 3: Start Training

```bash
# Basic training command
mlagents-learn config/my_agent.yaml --env=path/to/your/executable --run-id=my_first_agent

# With additional options
mlagents-learn config/my_agent.yaml \
    --env=path/to/your/executable \
    --run-id=my_first_agent \
    --force \
    --resume
```

### Step 4: Monitor Training

1. **TensorBoard** (Recommended):
   ```bash
   tensorboard --logdir=results
   ```
   Open your browser to `http://localhost:6006`

2. **Unity Console**: Watch the Unity console for episode information

3. **Results Folder**: Check the `results/` directory for training logs and model files

## Configuration Files

The `config/` directory contains pre-configured training setups for different algorithms:

### PPO (Proximal Policy Optimization)
- **Best for**: Most general-purpose training
- **Files**: `config/ppo/`
- **Examples**: `3DBall.yaml`, `Crawler.yaml`, `Walker.yaml`

### SAC (Soft Actor-Critic)
- **Best for**: Continuous control tasks
- **Files**: `config/sac/`
- **Examples**: `3DBall.yaml`, `Walker.yaml`

### MA-POCA (Multi-Agent POCA)
- **Best for**: Multi-agent scenarios
- **Files**: `config/poca/`
- **Examples**: `SoccerTwos.yaml`, `PushBlockCollab.yaml`

### Imitation Learning
- **Best for**: Learning from demonstrations
- **Files**: `config/imitation/`
- **Examples**: `Crawler.yaml`, `Hallway.yaml`

## Available Training Algorithms

| Algorithm | Type | Best For | Configuration |
|-----------|------|----------|---------------|
| **PPO** | On-Policy | General purpose, stable training | `config/ppo/` |
| **SAC** | Off-Policy | Continuous control, sample efficient | `config/sac/` |
| **MA-POCA** | Multi-Agent | Cooperative/competitive scenarios | `config/poca/` |
| **BC** | Imitation | Learning from demonstrations | `config/imitation/` |
| **GAIL** | Imitation | Adversarial imitation learning | `config/imitation/` |

## Running Examples

### 1. 3D Ball (PPO)
```bash
mlagents-learn config/ppo/3DBall.yaml --env=Project/3DBall --run-id=3dball_ppo
```

### 2. Crawler (SAC)
```bash
mlagents-learn config/sac/Crawler.yaml --env=Project/Crawler --run-id=crawler_sac
```

### 3. Multi-Agent Soccer
```bash
mlagents-learn config/poca/SoccerTwos.yaml --env=Project/SoccerTwos --run-id=soccer_poca
```

### 4. Imitation Learning
```bash
# First, collect demonstrations
mlagents-learn config/imitation/Crawler.yaml --env=Project/Crawler --run-id=crawler_demo --demo

# Then train with imitation learning
mlagents-learn config/imitation/Crawler.yaml --env=Project/Crawler --run-id=crawler_imitation --demo=demonstrations/Crawler.demo
```

## Advanced Usage

### Custom Training Environments

1. **Create your own Unity scene**
2. **Add ML-Agents components**:
   - `Behavior Parameters`
   - `Decision Requester`
   - Observation components (e.g., `Ray Perception Sensor`)
   - Action components (e.g., `Vector Action`)

3. **Implement reward logic** in your agent script

4. **Build and train**:
   ```bash
   mlagents-learn config/ppo/Basic.yaml --env=path/to/your/built/environment
   ```

### Hyperparameter Tuning

Modify the configuration files to experiment with different hyperparameters:

```yaml
hyperparameters:
  batch_size: 64          # Increase for more stable training
  learning_rate: 3.0e-4   # Adjust learning speed
  epsilon: 0.2           # PPO clipping parameter
  hidden_units: 128       # Network size
  num_layers: 2           # Network depth
```

### Curriculum Learning

Use curriculum learning for complex tasks:

```yaml
# Example: config/ppo/Sorter_curriculum.yaml
behaviors:
  SorterAgent:
    # ... other parameters ...
    curriculum:
      measure: progress
      thresholds: [0.1, 0.3, 0.5]
      min_lesson_length: 100
      signal_smoothing: true
```

### Environment Randomization

Add randomization to improve robustness:

```yaml
# Example: config/ppo/3DBall_randomize.yaml
environment_parameters:
  gravity:
    sampler_type: "uniform"
    min_value: 8.0
    max_value: 12.0
```

## Troubleshooting

### Common Issues and Solutions

#### 1. Python Version Issues
**Problem**: `Python version not supported`
**Solution**: Ensure you're using Python 3.10.1-3.10.12
```bash
python --version
# Should show Python 3.10.x
```

#### 2. Unity Build Issues
**Problem**: Unity executable not found or crashes
**Solution**: 
- Ensure Unity version is 2020.1 or later
- Check that the scene is properly configured
- Verify all ML-Agents components are properly set up

#### 3. Training Not Starting
**Problem**: `mlagents-learn` hangs or fails to connect
**Solution**:
- Check that the Unity executable is running
- Verify port availability (default: 5004)
- Use `--env` flag to specify the correct executable path

#### 4. CUDA/GPU Issues
**Problem**: Training is slow or GPU not detected
**Solution**:
```bash
# Check PyTorch CUDA support
python -c "import torch; print(torch.cuda.is_available())"

# Install CUDA-enabled PyTorch if needed
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### 5. Memory Issues
**Problem**: Out of memory errors during training
**Solution**:
- Reduce `batch_size` in configuration
- Decrease `buffer_size`
- Use fewer parallel environments

#### 6. Port Conflicts
**Problem**: `Address already in use` error
**Solution**:
```bash
# Use different port
mlagents-learn config/ppo/3DBall.yaml --env=Project/3DBall --base-port=5005
```


## Additional Resources

### Documentation
- [Getting Started Guide](docs/Getting-Started.md)
- [Training Configuration Reference](docs/Training-Configuration-File.md)
- [Python API Documentation](docs/Python-LLAPI.md)
- [Unity SDK Documentation](docs/com.unity.ml-agents.md)

### Learning Resources
- [Unity Learn Course: ML-Agents Hummingbirds](https://learn.unity.com/course/ml-agents-hummingbirds)
- [CodeMonkey Tutorial Series](https://www.youtube.com/playlist?list=PLzDRvYVwl53vehwiN_odYJkPBzcqFw110)
- [ML-Agents Blog Posts](https://blog.unity.com/tag/ml-agents)

### Example Projects
- **3D Ball**: Basic balancing task
- **Crawler**: Multi-limb locomotion
- **Walker**: Bipedal walking
- **Soccer Twos**: Multi-agent competitive environment
- **Food Collector**: Multi-agent cooperation

### Research Papers
If you use ML-Agents in research, please cite:
```bibtex
@article{juliani2020,
  title={Unity: A general platform for intelligent agents},
  author={Juliani, Arthur and Berges, Vincent-Pierre and Teng, Ervin and Cohen, Andrew and Harper, Jonathan and Elion, Chris and Goy, Chris and Gao, Yuan and Henry, Hunter and Mattar, Marwan and Lange, Danny},
  journal={arXiv preprint arXiv:1809.02627},
  url={https://arxiv.org/pdf/1809.02627.pdf},
  year={2020}
}
```

---

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE.md](LICENSE.md) file for details.



