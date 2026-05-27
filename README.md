# pinn-swe-reflections-lab

This repository contains a cleaned and reproducible experimental setup for studying Physics-Informed Neural Networks (PINNs) on the 1D shallow-water closed-boundary reflection benchmark.

The project is based on the benchmark and reference implementation from:

> Demir, K. T.; Logemann, K.; Greenberg, D. S.  
> **Closed-Boundary Reflections of Shallow Water Waves as an Open Challenge for Physics-Informed Neural Networks.**  
> *Mathematics* 2024, 12(21), 3315.  
> https://doi.org/10.3390/math12213315

Original repository:

> https://github.com/KubilayDemir/Testing_PINNs

The original numerical reference solver and PINN implementation were reorganized into a more reproducible research codebase. This repository also contains additional experimental components, including residual-adaptive sampling methods, RAR-D, gPINN-style losses, optimizer experiments, precision-control experiments, and reference-solution preprocessing utilities.

## Scope

The main goal of this repository is not to introduce a new physical model, but to provide an experimental laboratory for evaluating training strategies for PINNs on a difficult wave-reflection benchmark.

Planned/implemented experiment families include:

- baseline PINN reproduction;
- reference-solution preprocessing and grid consistency checks;
- residual-adaptive distribution sampling;
- RAR-D adaptive collocation;
- gPINN loss terms;
- validation against numerical reference solutions.

## Attribution

This repository reuses and modifies parts of the original implementation from `KubilayDemir/Testing_PINNs`, which is licensed under the Apache License 2.0.

All modifications, refactoring, experiment runners, and additional methods in this repository are part of `pinn-swe-reflections-lab`.

## License

This project is licensed under the Apache License 2.0. See the `LICENSE` file for details.