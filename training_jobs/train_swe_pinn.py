from pinn_swe_reflections.models import PINN
from pinn_swe_reflections.training import TrainConfig, run_training


cfg = TrainConfig()

cfg.epochs = 5
cfg.output_period = 1
cfg.experiment_name = "baseline"

run_training(
    model_cls=PINN,
    experiment_name=cfg.experiment_name,
    cfg=cfg,
)