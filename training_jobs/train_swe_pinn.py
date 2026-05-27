from pinn_swe_reflections.models import PINN
from pinn_swe_reflections.training import TrainConfig, run_training_pipeline


cfg = TrainConfig()

cfg.epochs = 5
cfg.output_period = 1

run_training_pipeline(
    model_cls=PINN,
    experiment_name="baseline",
    cfg=cfg,
)