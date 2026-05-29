from pinn_swe_reflections.models import EulerTransitionPINN
from pinn_swe_reflections.training import TrainConfig, run_training_pipeline


cfg = TrainConfig()

cfg.epochs = 1000
cfg.output_period = 10

run_training_pipeline(
    model_cls=EulerTransitionPINN,
    experiment_name="euler_transition_w001",
    cfg=cfg,
    model_kwargs={
        "euler_transition_weight": 0.01,
    },
)
