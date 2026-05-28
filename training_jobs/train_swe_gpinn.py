from pinn_swe_reflections.models import GradientEnhancedPINN
from pinn_swe_reflections.training import TrainConfig, run_training_pipeline


cfg = TrainConfig()

cfg.epochs = 1000
cfg.output_period = 10

run_training_pipeline(
    model_cls=GradientEnhancedPINN,
    experiment_name="gpinn_w001",
    cfg=cfg,
    model_kwargs={
        "gpinn_gradient_weight": 0.01,
    },
)
