import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

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
        old_symbolic_state = {
            "symbolic_function_sampling_points": (
                hasattr(self, "symbolic_function_sampling_points"),
                getattr(self, "symbolic_function_sampling_points", None),
            ),
            "symbolic_function_u_values": (
                hasattr(self, "symbolic_function_u_values"),
                getattr(self, "symbolic_function_u_values", None),
            ),
            "symbolic_function_h_values": (
                hasattr(self, "symbolic_function_h_values"),
                getattr(self, "symbolic_function_h_values", None),
            ),
        }

        try:
            tx = tx.detach().clone().float().to(self.device).requires_grad_(True)
            self.symbolic_function_sampling_points = tx
            self.Physics_Informed_Symbolic_Function()

            residual = torch.sqrt(
                self.symbolic_function_u_values.detach() ** 2
                + self.symbolic_function_h_values.detach() ** 2
                + self.adaptive_eps
            ).reshape(-1)

            return residual
        finally:
            for name, (had_value, value) in old_symbolic_state.items():
                if had_value:
                    setattr(self, name, value)
                elif hasattr(self, name):
                    delattr(self, name)

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

        # RAD: r_i = ||F(t_i, x_i)||_2. New points are sampled with
        # p_i = ((r_i + adaptive_eps)^k / mean_j((r_j + adaptive_eps)^k) + c) / sum_l(...).
        powered = torch.pow(residuals + self.adaptive_eps, self.rad_k)
        weights = powered / (powered.mean() + self.adaptive_eps) + self.rad_c
        weights = torch.where(torch.isfinite(weights), weights, torch.zeros_like(weights))
        weight_sum = weights.sum()
        if weight_sum.item() <= self.adaptive_eps:
            weights = torch.full_like(weights, 1.0 / weights.numel())
        else:
            weights = weights / weight_sum

        selected = torch.multinomial(
            weights,
            num_samples=self.symbolic_function_batch_size,
            replacement=candidates.shape[0] < self.symbolic_function_batch_size,
        )

        self.rad_mean_residual_over_training.append(float(residuals.mean()))
        self.rad_max_residual_over_training.append(float(residuals.max()))
        self.rad_min_residual_over_training.append(float(residuals.min()))

        return candidates[selected]

    def _resample_boundary_and_initial_datasets(self):
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

    def _sync_sampling_points_from_datasets(self):
        self.symbolic_function_sampling_points = self.pde_dataset.tx.to(self.device)
        self.lower_boundary_condition_sampling_points = self.lbc_dataset.tx.to(self.device)
        self.upper_boundary_condition_sampling_points = self.ubc_dataset.tx.to(self.device)
        self.initial_condition_sampling_points = self.ic_dataset.tx.to(self.device)

    def _refresh_minibatch_iterators(self):
        self.pde_dataloader = DataLoader(
            dataset=self.pde_dataset,
            batch_size=self.pde_mini_batch_size,
            shuffle=True,
        )
        self.pde_dataiter = iter(self.pde_dataloader)
        self.lbc_dataloader = DataLoader(
            dataset=self.lbc_dataset,
            batch_size=self.bc_mini_batch_size,
            shuffle=True,
        )
        self.lbc_dataiter = iter(self.lbc_dataloader)
        self.ubc_dataloader = DataLoader(
            dataset=self.ubc_dataset,
            batch_size=self.bc_mini_batch_size,
            shuffle=True,
        )
        self.ubc_dataiter = iter(self.ubc_dataloader)
        self.ic_dataloader = DataLoader(
            dataset=self.ic_dataset,
            batch_size=self.ic_mini_batch_size,
            shuffle=True,
        )
        self.ic_dataiter = iter(self.ic_dataloader)

    def resample_sampling_points(self):
        if not self._rad_initial_resample_done:
            super().resample_sampling_points()
            self._rad_initial_resample_done = True
            self._sync_sampling_points_from_datasets()
            return

        tx = self._sample_rad_pde_points()
        self.pde_dataset = TensorPDEDataset(tx)
        self._resample_boundary_and_initial_datasets()
        self._sync_sampling_points_from_datasets()

        self.new_sampling_points = True

    def train_PINN(self):
        self._sync_sampling_points_from_datasets()

        # track individual Mean Squared Error (MSE) terms over training
        self.MSE_symbolic_functions_over_training = []
        self.MSE_boundary_conditions_over_training = []
        self.MSE_initial_conditions_over_training = []
        self.MSE_numerical_solution_h_over_training = []
        self.MSE_numerical_solution_u_over_training = []
        self.total_MSE_over_training = []
        self.MSE_initial_condition_u_over_training = []
        self.MSE_initial_condition_h_over_training = []
        self.MSE_symbolic_function_u_over_training = []
        self.MSE_symbolic_function_h_over_training = []
        self.MSE_lower_boundary_condition_u_over_training = []
        self.MSE_upper_boundary_condition_u_over_training = []
        self.network_output_u_over_training = []
        self.network_output_h_over_training = []
        self.symbolic_function_u_over_training = []
        self.symbolic_function_h_over_training = []
        self.symbolic_function_weight_over_training = []
        self.initial_condition_weight_over_training = []
        self.boundary_condition_weight_over_training = []

        print("Choose Sampling Points.")
        self.resample_sampling_points()
        print("Evaluate Initial Network Output.")
        self.closure()
        self.MSE_numerical_solution_function()
        print("Save Initial Network Output and Losses.")

        if self.save_output_over_training is True:
            self.Save_Network_Output()

        if self.save_symbolic_function_over_training is True:
            self.Save_Symbolic_Functions()

        if self.learning_rate_annealing is True:
            self.Save_Weights_Over_Training()

        self.Save_MSE()

        self.time_per_epoch = []
        self.time_per_iteration = []
        global_iteration = 0

        print("Begin training:")
        for i in range(self.epochs):
            epoch_start_time = time.time()

            self._refresh_minibatch_iterators()

            for j in range(self.iterations_per_epoch):
                iteration_start_time = time.time()

                if (
                    self.batch_resampling_period > 0
                    and global_iteration > 0
                    and global_iteration % self.batch_resampling_period == 0
                ):
                    self.resample_sampling_points()
                    self._refresh_minibatch_iterators()
                    print("Change Sampling Points")

                if self.minibatch_training is True:
                    self.symbolic_function_sampling_points = next(self.pde_dataiter).to(self.device)
                    self.lower_boundary_condition_sampling_points = next(self.lbc_dataiter).to(self.device)
                    self.upper_boundary_condition_sampling_points = next(self.ubc_dataiter).to(self.device)
                    self.initial_condition_sampling_points = next(self.ic_dataiter).to(self.device)

                if self.projected_gradients is True and self.train_on_solution is False:
                    self.closure()
                    losses = [
                        self.initial_condition_weight * self.MSE_initial_condition_value,
                        self.symbolic_function_weight * self.MSE_symbolic_function_value,
                    ]
                    if (
                        self.boundary_condition_transition_function is False
                        and self.train_on_boundary_condition_loss is True
                    ):
                        losses.append(self.boundary_condition_weight * self.MSE_boundary_conditions_value)
                    self.optimizer.pc_backward(losses)
                    self.optimizer.step()

                elif self.line_search == "Armijo" and self.optimizer is not torch.optim.Adam:
                    self.optimizer.step(options={"closure": self.closure})

                else:
                    self.closure()
                    self.optimizer.step()

                if self.learning_rate_annealing is True:
                    self.Save_Weights_Over_Training()

                self.time_per_iteration.append(time.time() - iteration_start_time)
                global_iteration += 1

            if (i + 1) % self.output_period == 0:
                if self.save_output_over_training is True and (i + 1) % 100 == 0:
                    self.Save_Network_Output()

                if self.save_symbolic_function_over_training is True and (i + 1) % 100 == 0:
                    self.Save_Symbolic_Functions()

                self.MSE_numerical_solution_function()
                self.Save_MSE()
                validation_error = self.checkpoint_selection_error()

                if validation_error < self.best_validation_error:
                    self.save_best_state_in_memory(
                        step=i + 1,
                        validation_error=validation_error,
                    )

                print("Epoch no.: " + str(i + 1) + "/" + str(self.epochs) + " iteration no. " + str(j + 1) + "/"
                      + str(self.iterations_per_epoch))
                print("Boundary Condition Loss = ", self.MSE_boundary_conditions_value.cpu().detach().numpy())
                print("Initial Condition Loss = ", self.MSE_initial_condition_value.cpu().detach().numpy())
                print("Symbolic Function Loss = ", self.MSE_symbolic_function_value.cpu().detach().numpy())
                for label, value in self.Extra_Loss_Print_Items():
                    print(label + " = ", value.cpu().detach().numpy())
                print("Numerical Solution Loss u = ", self.MSE_numerical_solution_u_value.cpu().detach().numpy())
                print("Numerical Solution Loss h = ", self.MSE_numerical_solution_h_value.cpu().detach().numpy())

                if self.learning_rate_annealing is True:
                    print("Symbolic Function weight: ", self.symbolic_function_weight)
                    print("Initial Condition weight: ", np.asscalar(self.initial_condition_weight.cpu().numpy()))
                    print("Boundary Condition weight: ", np.asscalar(self.boundary_condition_weight.cpu().numpy()))

            self.time_per_epoch.append(time.time() - epoch_start_time)
            if (i + 1) % self.output_period == 0:
                print("Time Per Epoch: " + str(np.mean(self.time_per_epoch)) + "s")

        self._sync_sampling_points_from_datasets()
        print("Training completed.")
