from pinn_swe_reflections.models import ResidualAdaptiveRefinementDistributionPINN
from pinn_swe_reflections.training import TrainConfig, run_training_pipeline

cfg = TrainConfig()

cfg.epochs = 1000
cfg.output_period = 10
cfg.batch_resampling_period = 50

run_training_pipeline(
    model_cls=ResidualAdaptiveRefinementDistributionPINN,
    experiment_name="rar_d_k1_c1",
    cfg=cfg,
    model_kwargs={
        "adaptive_candidate_size": 200000,
        "adaptive_candidate_chunk_size": 5000,
        "rad_k": 1.0,
        "rad_c": 1.0,
        "rar_d_initial_fraction": 0.5,
        "rar_d_points_per_refinement": 1000,
    },
)
