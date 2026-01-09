### For the README of the ml-agents python package please go to the [ml-agents github](https://github.com/DennisSoemers/ml-agents)

# How to Run Experiments with `run_all.py`

## 1) Build the Unity environment

Choose an output folder and build an executable.

   * **Windows:** `.../UnityEnvironment.exe`
   * **macOS:** `.../MyEnv.app`



## 2) Edit `run_all.py`

Set the environment path and the runs you want:

```python
ENV_PATH = "C:\\path\\to\\UnityEnvironment.exe"  # macOS: "/path/to/MyEnv.app"

CONFIGS = [
    ("WormTestTwo", "C:\\...\\config\\ppo\\Worm\\WormTestTwo.yaml"),
    ("WormTest",    "C:\\...\\config\\ppo\\Worm\\WormTest.yaml"),
    ("Worm",        "C:\\...\\config\\ppo\\Worm\\Worm.yaml"),
]
```

The script uses: `--no-graphics --train --force`.

Take care when using as it overrides runs with the same name.

## 3) Run

```bash
python appendix/run_all.py
```

ML-Agents writes logs to a `results/` folder in the working directory.

## 4) Monitor on tensorboard

```bash
tensorboard --logdir results
```

## Configurations

### Worm

| Configuration file       | Changed parameters (value)                                  |
| ------------------------ | ----------------------------------------------------------- |
| `worm_fastlearn.yaml`    | `learning_rate = 0.0005`; `beta = 0.01`                     |
| `worm_highcapacity.yaml` | `hidden_units = 1024`; `num_layers = 4`                     |
| `worm_slowsteady.yaml`   | `learning_rate = 0.0002`                                    |
| `worm_stablelearn.yaml`  | `batch_size = 4048`; `buffer_size = 40480`; `beta = 0.0025` |

### Pyramids

| Configuration file       | Changed parameters (value)               |
| ------------------------ | ---------------------------------------- |
| `Pyramids_faster.yaml`   | `learning_rate = 0.0005`                 |
| `Pyramids_steadier.yaml` | `learning_rate = 0.0002`                 |
| `Pyramids_capacity.yaml` | `hidden_units = 1024`                    |
| `Pyramids_stabler.yaml`  | `batch_size = 256`; `buffer_size = 4096` |



More to be written later once the other python stuff is done