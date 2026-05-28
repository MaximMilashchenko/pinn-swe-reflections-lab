from pinn_swe_reflections.models import PINN
from pinn_swe_reflections.training import TrainConfig, run_training_pipeline


cfg = TrainConfig()

cfg.epochs = 1000
cfg.output_period = 10

run_training_pipeline(
    model_cls=PINN,
    experiment_name="baseline",
    cfg=cfg,
)