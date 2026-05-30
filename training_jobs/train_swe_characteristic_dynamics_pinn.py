from pinn_swe_reflections.models import CharacteristicDynamicsPINN
from pinn_swe_reflections.training import TrainConfig, run_training_pipeline


cfg = TrainConfig()

cfg.epochs = 1000
cfg.output_period = 10

run_training_pipeline(
    model_cls=CharacteristicDynamicsPINN,
    experiment_name="characteristic_dynamics_w01",
    cfg=cfg,
    model_kwargs={
        "characteristic_dynamics_weight": 1.0,
    },
)
