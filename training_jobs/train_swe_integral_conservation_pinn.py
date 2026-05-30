from pinn_swe_reflections.models import IntegralConservationPINN
from pinn_swe_reflections.training import TrainConfig, run_training_pipeline


cfg = TrainConfig()

cfg.epochs = 1000
cfg.output_period = 10

run_training_pipeline(
    model_cls=IntegralConservationPINN,
    experiment_name="integral_conservation_anchor_w1",
    cfg=cfg,
    model_kwargs={
        "global_mass_weight": 1.0,
        "energy_balance_weight": 1.0,
        "energy_pairwise_weight": 0.0,
        "control_volume_mass_weight": 1.0,
        "control_volume_momentum_weight": 0.25,
        "conservation_time_batch_size": 4,
        "conservation_space_points": 51,
        "conservation_dissipation_time_points": 3,
        "control_volume_batch_size": 4,
        "control_volume_cells": 8,
        "control_volume_space_points": 4,
        "control_volume_time_points": 4,
    },
)
