import numpy as np
from numba import njit

@njit
def init_random_grid(rows, cols):
    grid = 2 * np.random.randint(0, 2, size=(rows, cols)) - 1
    return grid.astype(np.int8)

@njit
def _add_halo(array):
    rows, cols = array.shape
    b_array = np.zeros((rows + 2, cols + 2), dtype=np.int8)

    for i in range(rows):
        for j in range(cols):
            b_array[i + 1, j + 1] = array[i, j]

    return b_array

@njit()
def run_ising_model(grid, temperature, steps):
    grid = _add_halo(grid)
    rows, cols = grid.shape

    for _ in range(steps):  
        i = np.random.randint(1, rows - 1)
        j = np.random.randint(1, cols - 1)

        energy = -grid[i, j] * (grid[i + 1, j] + grid[i - 1, j] + grid[i, j + 1] + grid[i, j - 1])

        if energy > 0:
            grid[i][j] *= -1
        elif energy < 0:
            r = np.random.random()

            if r < np.exp(2 * energy / temperature):
                grid[i][j] *= -1

    return grid[1:-1, 1:-1]
