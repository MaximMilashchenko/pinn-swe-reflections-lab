from pinn_swe_reflections.models import ConstrainedIntegralConservationPINN
from pinn_swe_reflections.training import TrainConfig, run_training_pipeline


cfg = TrainConfig()

cfg.epochs = 1000
cfg.output_period = 10
cfg.boundary_condition_transition_function = True
cfg.initial_condition_transition_function = True

run_training_pipeline(
    model_cls=ConstrainedIntegralConservationPINN,
    experiment_name="constrained_integral_conservation_soft_cv",
    cfg=cfg,
    model_kwargs={
        "global_mass_weight": 1.0,
        "energy_balance_weight": 1.0,
        "energy_pairwise_weight": 0.0,
        "global_discharge_weight": 1.0,
        "control_volume_mass_weight": 0.1,
        "control_volume_momentum_weight": 0.0,
        "conservation_time_batch_size": 4,
        "conservation_space_points": 51,
        "conservation_dissipation_time_points": 3,
        "control_volume_batch_size": 4,
        "control_volume_cells": 8,
        "control_volume_space_points": 4,
        "control_volume_time_points": 4,
        "initial_transition_time_scale_factor": 0.1,
    },
)
