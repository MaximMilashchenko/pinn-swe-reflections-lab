import torch
from torch.utils.data import Dataset

from pinn_swe_reflections.models.pinn import PINN
from pinn_swe_reflections.models.components import (
    PDEDataset,
    LowerBoundaryDataset,
    UpperBoundaryDataset,
    InitialConditionDataset,
    FinalStateDataset,
)


class TensorPDEDataset(Dataset):
    def __init__(self, tx):
        self.tx = tx.detach().clone().float().requires_grad_(True)
        self.t = self.tx[:, 0:1]
        self.x = self.tx[:, 1:2]
        self.n_samples = self.tx.shape[0]

    def __getitem__(self, index):
        return self.tx[index, :]

    def __len__(self):
        return self.n_samples


class ResidualAdaptiveDistributionPINN(PINN):
    def __init__(
        self,
        *args,
        adaptive_candidate_size=200000,
        adaptive_candidate_chunk_size=5000,
        rad_k=1.0,
        rad_c=1.0,
        adaptive_eps=1e-12,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.adaptive_candidate_size = adaptive_candidate_size
        self.adaptive_candidate_chunk_size = adaptive_candidate_chunk_size
        self.rad_k = rad_k
        self.rad_c = rad_c
        self.adaptive_eps = adaptive_eps

        self._rad_initial_resample_done = False
        self._rad_resampling_cooldown = 0

        self.rad_mean_residual_over_training = []
        self.rad_max_residual_over_training = []
        self.rad_min_residual_over_training = []

    def _sample_uniform_pde_candidates(self, n_points):
        dataset = PDEDataset(
            minimum_time=self.minimum_time,
            maximum_time=self.maximum_time,
            minimum_x=self.minimum_x,
            maximum_x=self.maximum_x,
            batch_size=n_points,
        )
        return dataset.tx

    def _compute_pde_residual_norm(self, tx):
        old_sampling_points = getattr(self, "symbolic_function_sampling_points", None)

        tx = tx.detach().clone().float().to(self.device).requires_grad_(True)
        self.symbolic_function_sampling_points = tx
        self.Physics_Informed_Symbolic_Function()

        residual = torch.sqrt(
            self.symbolic_function_u_values.detach() ** 2
            + self.symbolic_function_h_values.detach() ** 2
            + self.adaptive_eps
        ).reshape(-1)

        self.symbolic_function_sampling_points = old_sampling_points
        return residual

    def _compute_candidate_residuals(self, candidates):
        residuals = []

        for start in range(0, candidates.shape[0], self.adaptive_candidate_chunk_size):
            end = start + self.adaptive_candidate_chunk_size
            chunk = candidates[start:end]
            residuals.append(self._compute_pde_residual_norm(chunk).detach().cpu())

        return torch.cat(residuals, dim=0)

    def _sample_rad_pde_points(self):
        candidates = self._sample_uniform_pde_candidates(self.adaptive_candidate_size)
        residuals = self._compute_candidate_residuals(candidates)

        powered = torch.pow(residuals + self.adaptive_eps, self.rad_k)
        weights = powered / (powered.mean() + self.adaptive_eps) + self.rad_c
        weights = weights / weights.sum()

        selected = torch.multinomial(
            weights,
            num_samples=self.symbolic_function_batch_size,
            replacement=True,
        )

        self.rad_mean_residual_over_training.append(float(residuals.mean()))
        self.rad_max_residual_over_training.append(float(residuals.max()))
        self.rad_min_residual_over_training.append(float(residuals.min()))

        return candidates[selected]

    def resample_sampling_points(self):
        if self._rad_resampling_cooldown > 0:
            self._rad_resampling_cooldown -= 1
            return

        if not self._rad_initial_resample_done:
            super().resample_sampling_points()
            self._rad_initial_resample_done = True
            return

        tx = self._sample_rad_pde_points()
        self.pde_dataset = TensorPDEDataset(tx)

        self.lbc_dataset = LowerBoundaryDataset(
            minimum_time=self.minimum_time,
            maximum_time=self.maximum_time,
            minimum_x=self.minimum_x,
            batch_size=self.boundary_condition_batch_size,
        )
        self.ubc_dataset = UpperBoundaryDataset(
            minimum_time=self.minimum_time,
            maximum_time=self.maximum_time,
            maximum_x=self.maximum_x,
            batch_size=self.boundary_condition_batch_size,
        )
        self.ic_dataset = InitialConditionDataset(
            minimum_time=self.minimum_time,
            minimum_x=self.minimum_x,
            maximum_x=self.maximum_x,
            batch_size=self.initial_condition_batch_size,
        )
        self.final_state_dataset = FinalStateDataset(
            maximum_time=self.maximum_time,
            minimum_x=self.minimum_x,
            maximum_x=self.maximum_x,
            batch_size=self.initial_condition_batch_size,
        )

        self.new_sampling_points = True
        self._rad_resampling_cooldown = max(0, self.iterations_per_epoch - 1)