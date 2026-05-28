import torch
from torch.utils.data import DataLoader

from pinn_swe_reflections.models.components import (
    FinalStateDataset,
    InitialConditionDataset,
    LowerBoundaryDataset,
    UpperBoundaryDataset,
)
from pinn_swe_reflections.models.rad_pinn import (
    ResidualAdaptiveDistributionPINN,
    TensorPDEDataset,
)


class CyclingDataIterator:
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.iterator = iter(dataloader)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            return next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.dataloader)
            return next(self.iterator)


class ResidualAdaptiveRefinementDistributionPINN(ResidualAdaptiveDistributionPINN):
    """RAR-D PINN from Wu et al. (2022): add residual-weighted PDE points.

    `symbolic_function_batch_size` is treated as the maximum PDE residual-point
    budget by default. The initial PDE set is smaller and each refinement appends
    new points sampled from the RAD probability mass function.
    """

    def __init__(
        self,
        *args,
        rar_d_initial_pde_size=None,
        rar_d_initial_fraction=0.5,
        rar_d_max_pde_size=None,
        rar_d_points_per_refinement=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.rar_d_max_pde_size = int(
            self.symbolic_function_batch_size
            if rar_d_max_pde_size is None
            else rar_d_max_pde_size
        )
        if self.rar_d_max_pde_size <= 0:
            raise ValueError("rar_d_max_pde_size must be positive.")

        if rar_d_initial_pde_size is None:
            if not 0 < rar_d_initial_fraction <= 1:
                raise ValueError("rar_d_initial_fraction must be in (0, 1].")
            rar_d_initial_pde_size = max(
                1,
                int(round(self.rar_d_max_pde_size * rar_d_initial_fraction)),
            )
        self.rar_d_initial_pde_size = int(rar_d_initial_pde_size)

        if self.rar_d_initial_pde_size <= 0:
            raise ValueError("rar_d_initial_pde_size must be positive.")
        if self.rar_d_initial_pde_size > self.rar_d_max_pde_size:
            raise ValueError(
                "rar_d_initial_pde_size must not exceed rar_d_max_pde_size."
            )

        if rar_d_points_per_refinement is None:
            rar_d_points_per_refinement = max(
                1,
                int(round(0.01 * self.rar_d_max_pde_size)),
            )
        self.rar_d_points_per_refinement = int(rar_d_points_per_refinement)
        if self.rar_d_points_per_refinement <= 0:
            raise ValueError("rar_d_points_per_refinement must be positive.")

        self._rar_d_initial_resample_done = False
        self.rar_d_added_points_over_training = []
        self.rar_d_pde_dataset_size_over_training = []
        self.rar_d_mean_residual_over_training = []
        self.rar_d_max_residual_over_training = []
        self.rar_d_min_residual_over_training = []

    def _rad_weights_from_residuals(self, residuals):
        powered = torch.pow(residuals + self.adaptive_eps, self.rad_k)
        weights = powered / (powered.mean() + self.adaptive_eps) + self.rad_c
        weights = torch.where(
            torch.isfinite(weights),
            weights,
            torch.zeros_like(weights),
        )

        weight_sum = weights.sum()
        if weight_sum.item() <= self.adaptive_eps:
            return torch.full_like(weights, 1.0 / weights.numel())

        return weights / weight_sum

    def _sample_rar_d_pde_points(self, n_points):
        candidates = self._sample_uniform_pde_candidates(self.adaptive_candidate_size)
        residuals = self._compute_candidate_residuals(candidates)
        weights = self._rad_weights_from_residuals(residuals)

        selected = torch.multinomial(
            weights,
            num_samples=n_points,
            replacement=candidates.shape[0] < n_points,
        )

        self.rar_d_mean_residual_over_training.append(float(residuals.mean()))
        self.rar_d_max_residual_over_training.append(float(residuals.max()))
        self.rar_d_min_residual_over_training.append(float(residuals.min()))

        return candidates[selected]

    def _replace_boundary_and_initial_datasets(self):
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

    def _record_pde_dataset_size(self, added_points):
        self.rar_d_added_points_over_training.append(int(added_points))
        self.rar_d_pde_dataset_size_over_training.append(len(self.pde_dataset))

    def _refresh_minibatch_iterators(self):
        self.pde_dataloader = DataLoader(
            dataset=self.pde_dataset,
            batch_size=self.pde_mini_batch_size,
            shuffle=True,
        )
        self.pde_dataiter = CyclingDataIterator(self.pde_dataloader)
        self.lbc_dataloader = DataLoader(
            dataset=self.lbc_dataset,
            batch_size=self.bc_mini_batch_size,
            shuffle=True,
        )
        self.lbc_dataiter = CyclingDataIterator(self.lbc_dataloader)
        self.ubc_dataloader = DataLoader(
            dataset=self.ubc_dataset,
            batch_size=self.bc_mini_batch_size,
            shuffle=True,
        )
        self.ubc_dataiter = CyclingDataIterator(self.ubc_dataloader)
        self.ic_dataloader = DataLoader(
            dataset=self.ic_dataset,
            batch_size=self.ic_mini_batch_size,
            shuffle=True,
        )
        self.ic_dataiter = CyclingDataIterator(self.ic_dataloader)

    def resample_sampling_points(self):
        if not self._rar_d_initial_resample_done:
            tx = self._sample_uniform_pde_candidates(self.rar_d_initial_pde_size)
            self.pde_dataset = TensorPDEDataset(tx)
            self._replace_boundary_and_initial_datasets()
            self._rar_d_initial_resample_done = True
            self._rad_initial_resample_done = True
            self._sync_sampling_points_from_datasets()
            self._record_pde_dataset_size(added_points=0)
            self.new_sampling_points = True
            return

        current_size = len(self.pde_dataset)
        available_capacity = self.rar_d_max_pde_size - current_size

        if available_capacity > 0:
            n_new = min(self.rar_d_points_per_refinement, available_capacity)
            new_tx = self._sample_rar_d_pde_points(n_new)
            tx = torch.cat(
                (
                    self.pde_dataset.tx.detach().cpu(),
                    new_tx.detach().cpu(),
                ),
                dim=0,
            )
            self.pde_dataset = TensorPDEDataset(tx)
        else:
            n_new = 0

        self._replace_boundary_and_initial_datasets()
        self._sync_sampling_points_from_datasets()
        self._record_pde_dataset_size(added_points=n_new)
        self.new_sampling_points = True
