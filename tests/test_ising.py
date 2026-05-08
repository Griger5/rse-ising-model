import numpy as np
from ising_model import init_random_grid, run_ising_model

def test_init_random_grid_shape():
    grid = init_random_grid(10, 20)

    assert grid.shape == (10, 20)

def test_init_random_grid_values():
    grid = init_random_grid(50, 50)

    assert np.all(np.logical_or(grid == -1, grid == 1))

def test_run_ising_model_preserves_shape():
    grid = init_random_grid(30, 40)

    result = run_ising_model(grid, temperature=2.0, steps=1000)

    assert result.shape == (30, 40)

def test_run_ising_model_preserves_spin_values():
    grid = init_random_grid(20, 20)

    result = run_ising_model(grid, temperature=2.0, steps=5000)

    assert np.all(np.logical_or(result == -1, result == 1))

def test_low_temperature_increases_order():
    np.random.seed(0)

    grid = init_random_grid(50, 50)

    initial_magnetization = np.abs(np.mean(grid))

    result = run_ising_model(grid, temperature=0.1, steps=200000)

    final_magnetization = np.abs(np.mean(result))

    assert final_magnetization > initial_magnetization