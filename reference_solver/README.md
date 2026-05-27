# Reference Solver

This directory contains the Fortran numerical solver and conversion script used to generate the reference solution for PINN validation.

Workflow:

```text
Fortran solver -> raw .dat snapshots -> PINN-ready .npy files
```

## Build the solver

Run from the repository root:

```bash
cd reference_solver/fortran
gfortran swe_imp_adv_diff_dx=400m_dt=1s.f90 -o swe_solver
```

## Create output directories

Run from `reference_solver/fortran`:

```bash
mkdir -p ../data/raw/AH=5e+4/udata
mkdir -p ../data/raw/AH=5e+4/zdata
mkdir -p ../data/raw/AH=5e+4/pegel
```

## Run the solver

Run from `reference_solver/fortran`:

```bash
./swe_solver
```

The solver writes raw snapshots to:

```text
reference_solver/data/raw/AH=5e+4/udata/
reference_solver/data/raw/AH=5e+4/zdata/
reference_solver/data/raw/AH=5e+4/pegel/
```

The solver uses an internal time step of `dt = 1 s` and saves snapshots every `60 s`:

```text
u0000.dat / z0000.dat -> t = 0 s
u0001.dat / z0001.dat -> t = 60 s
u0002.dat / z0002.dat -> t = 120 s
```

## Convert to PINN-ready `.npy`

Return to the repository root:

```bash
cd ../..
```

Then run:

```bash
python reference_solver/scripts/convert_solution_to_pinn_ready.py --case "AH=5e+4"
```

The output files are saved to:

```text
reference_solver/data/pinn_ready/dt=1s_dx=400m/AH=5e+4/
```

Expected files:

```text
Zonal_Velocity.npy
Sea_Level_Elevation.npy
reference_preprocess_meta.json
```

Expected shapes:

```text
Zonal_Velocity.npy        -> (201, 4500)
Sea_Level_Elevation.npy  -> (200, 4500)
```

The arrays are stored as:

```text
space x time
```

which is the format expected by the PINN code.

## Default converter parameters

The default command is equivalent to:

```bash
python reference_solver/scripts/convert_solution_to_pinn_ready.py \
  --case "AH=5e+4" \
  --raw-dx 400.0 \
  --raw-output-dt 60.0 \
  --target-dx 10000.0 \
  --target-dt 60.0 \
  --time-start 0.0 \
  --time-end 270000.0 \
  --x-min -1000000.0 \
  --x-max 1000000.0 \
  --zeta-start-index 13
```

The converter downsamples the raw grid to the reference grid expected by PINN:

```text
raw dx        = 400 m
target dx     = 10000 m
raw output dt = 60 s
target dt     = 60 s
```

The main conversion is:

```python
u_ref = u_raw[:4500, ::25].T
zeta_ref = zeta_raw[:4500, 13::25].T
```

`u` and `zeta` are stored on staggered grids, so their spatial grids are different.

## If the grid changes

If the Fortran grid changes, update everything consistently.

First, update the Fortran solver parameters, for example:

```fortran
dt = 1.
dx = 400.
nx = 5002
AH = 5.e+4
```

Then pass matching parameters to the converter:

```bash
python reference_solver/scripts/convert_solution_to_pinn_ready.py \
  --case "AH=5e+4" \
  --raw-dx 400.0 \
  --target-dx 10000.0 \
  --target-dt 60.0 \
  --time-end 270000.0
```

Finally, make sure the PINN config uses the same reference grid:

```python
numerical_solution_time_interval = [0.0, 270000.0]
numerical_solution_time_step = 60.0

numerical_solution_x_interval = [-1000000.0, 1000000.0]
numerical_solution_space_step = 10000.0

numerical_solution_directory = "AH=5e+4"
```

## Notes

Generated `.dat` and `.npy` files are not committed to git.

The reference solution is used for validation and error computation. It is not used as supervised training data when:

```python
train_on_solution = False
```