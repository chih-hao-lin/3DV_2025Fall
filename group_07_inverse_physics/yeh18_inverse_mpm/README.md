# Inverse DiffMPM 2D
### Hacker: Changhan Yeh (yeh18@illinois.edu)

### Set up
- Install `taichi`, `numpy`

### How to run?
- `diffmpm_E.py`
  - Default expects a ground-truth trajectory at `mpm128_simplified_liquid/target_positions.npy` (shape `(T, N, 2)`).
  - Example:
    ```bash
    python diffmpm_E.py --iters 1500 --target mpm128_simplified_liquid/target_positions.npy --horizon 1500
    ```
  - PNG frames will be saved under `mpm128_simplified_liquid/optimize_E/`.

- `diffmpm_G.py`
  - Default expects a ground-truth trajectory at `mpm128_simplified/target_positions.npy` (shape `(T, N, 2)`).
  - Example:
    ```bash
    python diffmpm_G.py --iters 1000 --target mpm128_simplified/target_positions.npy --horizon 1000
    ```
  - PNG frames will be saved under `mpm128_simplified/optimize/`.

> Notes:
> - `--iters` controls the number of optimization iterations.
> - `--horizon` controls the simulated time steps per optimization round.
> - Material 0 is liquid, and material 1 is jelly.