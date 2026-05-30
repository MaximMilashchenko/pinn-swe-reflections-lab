from pinn_swe_reflections.models import CharacteristicDynamicsPINN
from pinn_swe_reflections.training import TrainConfig, run_training_pipeline


cfg = TrainConfig()

cfg.epochs = 1000
cfg.output_period = 10

run_training_pipeline(
    model_cls=CharacteristicDynamicsPINN,
    experiment_name="swe_dynamics_modal_w1_dt05d",
    cfg=cfg,
    model_kwargs={
        "swe_dynamics_weight": 1.0,
        "modal_dynamics_dt_seconds": 43200.0,
        "modal_dynamics_time_batch_size": 6,
        "modal_dynamics_space_points": 201,
        "modal_dynamics_modes": 32,
        "modal_dynamics_include_initial": True,
        "modal_dynamics_anchor_initial": True,
        "modal_dynamics_step_weight": 5e-3,
        "modal_dynamics_anchor_weight": 5e-4,
    },
)
