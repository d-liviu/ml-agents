import random
import run_random_sweep

# testing if ensure_sections  adds missing sections without destroying existing data
def test_ensure_sections():
    cfg = {"env_settings": {"seed": 7}}
    run_random_sweep.ensure_sections(cfg)
    assert cfg["env_settings"]["seed"] == 7

#checking if overriding values actually works and chosen values are saved properly
def test_apply_overrides():
    cfg = {"behaviors": {"MyBehavior": {}}}
    overrides = {
        "max_steps": 123,
        "learning_rate": 1e-4,
        "batch_size": 256,
        "buffer_size": 5120,
        "beta": 5e-3,
        "hidden_units": 128,
        "num_layers": 3,
    }
    run_random_sweep.apply_overrides(cfg, "MyBehavior", overrides)
    b = cfg["behaviors"]["MyBehavior"]

    assert b["max_steps"] == 123
    assert b["hyperparameters"]["learning_rate"] == 1e-4
    assert b["hyperparameters"]["batch_size"] == 256
    assert b["hyperparameters"]["buffer_size"] == 5120
    assert b["hyperparameters"]["beta"] == 5e-3

    assert b["network_settings"]["hidden_units"] == 128
    assert b["network_settings"]["num_layers"] == 3

    # defaults
    assert b["hyperparameters"]["learning_rate_schedule"] == "linear"
    assert b["hyperparameters"]["beta_schedule"] == "constant"

#esuring correct data types+ checking the value bounds are not crossed ex. buffer_size >= batch_size
def test_bounds_and_types():
    rng = random.Random(0)
    s = run_random_sweep.sample_hparams(rng)

    assert isinstance(s["learning_rate"], float)
    assert isinstance(s["beta"], float)
    assert isinstance(s["batch_size"], int)
    assert isinstance(s["buffer_size"], int)
    assert isinstance(s["hidden_units"], int)
    assert isinstance(s["num_layers"], int)

    assert s["buffer_size"] >= s["batch_size"]
    assert 1e-5 <= s["learning_rate"] <= 1e-3
    assert 1e-4 <= s["beta"] <= 1e-2

#checking log-uniform results are not out of the what we consider accpetable range based on ML-Agents default learning_rate being 3e-4
def test_loguniform_in_range():
    rng = random.Random(0)
    v = run_random_sweep.loguniform(rng, 1e-5, 1e-3)
    assert 1e-5 <= v <= 1e-3
