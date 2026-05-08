# Ising Model Simulation

A simple Python implementation of the 2D Ising model using the Metropolis Monte Carlo algorithm, accelerated with Numba.

## Overview

This project simulates magnetic spin systems on a 2D lattice, where each site can take values of +1 or -1. The system evolves according to thermal fluctuations controlled by a temperature parameter.

At low temperatures, spins tend to align (ordered phase), while at high temperatures they become disordered.

In every step of the simulation, one spin is selected at random. Its energy is calculated with the given formula:

$$
E_{i,j} = -s_{i,j}*(s_{i+1,j}+s_{i-1,j}+s_{i,j+1}+s_{i,j-1})
$$

Where:

- ___i, j___ are the indexes of the spin's row and column, respectively
- ___s___ is the value of the given spin (either 1 or -1)
- ___E___ is the energy of the spin's square

If the calculated energy is greater than 0, the spin will switch. Otherwise, it will switch with the probability given by the following formula:

$$
P(switch_{i,j})=e^{(2*E_{i,j})/T}
$$

Where ___T___ is the systems temperature.

## Features

- 2D Ising model simulation
- Metropolis-Hastings update rule
- Numba JIT acceleration for performance
- Simple random initialisation of spin grid
