import numpy as np
from numba import njit

@njit
def init_random_grid(rows, cols):
    """
    Create a random Ising spin grid.

    Parameters
    ----------
    rows : int
        Number of rows.
    cols : int
        Number of columns.

    Returns
    -------
    numpy.ndarray
        2D array of shape (rows, cols) containing
        spins with values {-1, +1}.
    """
    grid = 2 * np.random.randint(0, 2, size=(rows, cols)) - 1
    return grid.astype(np.int8)

@njit
def _add_halo(array):
    """
    Add a zero-valued halo around a 2D array.

    Parameters
    ----------
    array : numpy.ndarray
        Input 2D array.

    Returns
    -------
    numpy.ndarray
        Array padded with one layer of zeros
        on every side.
    """
    rows, cols = array.shape
    b_array = np.zeros((rows + 2, cols + 2), dtype=np.int8)

    for i in range(rows):
        for j in range(cols):
            b_array[i + 1, j + 1] = array[i, j]

    return b_array

@njit()
def run_ising_model(grid, temperature, steps):
    """
    Run a 2D Ising model simulation using the Monte Carlo algorithm.

    Parameters
    ----------
    grid : numpy.ndarray
        Initial spin configuration containing
        values {-1, +1}.
    temperature : float
        Simulation temperature.
    steps : int
        Number of Monte Carlo spin updates.

    Returns
    -------
    numpy.ndarray
        Final spin configuration after simulation.
    """
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
