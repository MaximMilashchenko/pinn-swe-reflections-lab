# external libraries
import os
import time
import numpy as np
import torch
from torch import nn
from torch.autograd import grad
from torch.utils.data import DataLoader

from pinn_swe_reflections.models.components import (
    FinalStateDataset,
    InitialConditionDataset,
    LowerBoundaryDataset,
    PDEDataset,
    UpperBoundaryDataset,
    true_initial_condition_h_function,
    true_initial_condition_u_function,
    true_lower_boundary_condition_u_function,
    true_upper_boundary_condition_u_function,
)

from pinn_swe_reflections.optim import PCGrad, FullBatchLBFGS


# Define class of Physics Informed Neural Networks for the 1D-Shallow Water Equations
class PINN(nn.Module):
    def __init__(
            self,
            layer_sizes: list = None,
            activation_function=torch.tanh,
            optimizer=torch.optim.LBFGS,
            learning_rate=1,
            line_search="strong_wolfe",
            boundary_condition_weight=1,
            initial_condition_weight=1,
            symbolic_function_weight=1,
            boundary_condition_batch_size=5000,
            initial_condition_batch_size=5000,
            symbolic_function_batch_size=50000,
            training_steps=1500,
            batch_resampling_period=1,
            console_output_period=1,
            device="cuda",
            gravitational_acceleration=9.81,
            average_sea_level=100.0,
            momentum_dissipation=0,
            nonlinear_drag_coefficient=0,
            initial_perturbation_amplitude=1.0,
            non_dimensionalization=True,
            vertical_length_scale=1.0,
            vertical_scaling_factor=1.0,
            horizontal_length_scale=1e6,
            time_scale=5e4,
            minimum_time=0.0,
            maximum_time=1.0,
            minimum_x=-1.0,
            maximum_x=1.0,
            projected_gradients=False,
            save_output_over_training=False,
            save_symbolic_function_over_training=True,
            numerical_solution_time_interval=None,
            numerical_solution_time_step=60.0,
            numerical_solution_x_interval=None,
            numerical_solution_space_step=1000.0,
            fraction_of_time_interval=None,
            model_number=0,
            number_of_models=1,
            new_initial_conditions=None,
            new_initial_condition_sampling_points=None,
            train_on_solution=False,
            train_on_PINNs_Loss=True,
            boundary_condition_transition_function=True,
            initial_condition_transition_function=True,
            split_networks=False,
            train_on_boundary_condition_loss=True,
            train_on_initial_condition_loss=True,
            momentum_advection=False,
            mixed_activation_functions=False,
            sirens_initialization=False,
            learning_rate_annealing=False,
            minibatch_training=False,
            pde_mini_batch_size=1000,
            bc_mini_batch_size=100,
            ic_mini_batch_size=100,
            iterations_per_epoch=50,
            numerical_solution_directory=None
    ):

        super().__init__()

        self.exp_name = os.getenv("EXP_NAME")
        self.best_state_dir = os.getenv("BEST_STATE_DIR")
        # initialize Model Architecture using the given Parameters
        print("Initialize Network Architecture.")
        self.minibatch_training = minibatch_training
        self.sirens_initialization = sirens_initialization
        self.learning_rate_annealing = learning_rate_annealing
        self.mixed_activation_functions = mixed_activation_functions
        self.split_networks = split_networks
        self.train_on_boundary_condition_loss = train_on_boundary_condition_loss
        self.train_on_initial_condition_loss = train_on_initial_condition_loss
        self.improvement_steps = []
        self.best_step = 0
        self.number_of_layers = len(layer_sizes) - 1
        self.layer_sizes = layer_sizes
        self.activation_function = activation_function
        self.relu = torch.nn.functional.relu

        if self.split_networks is True:
            self.layers_u = []
            self.layers_h = []
            for i in np.arange(self.number_of_layers):
                if i < self.number_of_layers - 1:
                    self.layers_u.append(torch.nn.Linear(self.layer_sizes[i], self.layer_sizes[i + 1]))
                    self.layers_h.append(torch.nn.Linear(self.layer_sizes[i], self.layer_sizes[i + 1]))
                else:
                    self.layers_u.append(torch.nn.Linear(self.layer_sizes[i], 1))
                    self.layers_h.append(torch.nn.Linear(self.layer_sizes[i], 1))
            self.layers_u = nn.ModuleList(self.layers_u)
            self.layers_h = nn.ModuleList(self.layers_h)
        else:
            self.layers = []
            for i in np.arange(self.number_of_layers):
                self.layers.append(torch.nn.Linear(self.layer_sizes[i], self.layer_sizes[i + 1]))
            self.layers = nn.ModuleList(self.layers)

        if self.sirens_initialization is True:
            c = np.sqrt(6)
            omega_0 = 30
            for i in range(self.number_of_layers):
                input_size = self.layers[i].weight.shape[0]
                self.layers[i].weight = torch.nn.Parameter(
                    2.0 * (c / np.sqrt(input_size)) * torch.rand(size=self.layers[i].weight.size())
                    - (c / np.sqrt(input_size))
                )
            self.layers[0].weight = torch.nn.Parameter(omega_0 * self.layers[0].weight)

        # initialize true Initial and Boundary Conditions
        print("Initialize true Initial and Boundary Conditions.")
        self.save_output_over_training = save_output_over_training
        self.save_symbolic_function_over_training = save_symbolic_function_over_training
        self.model_number = model_number
        self.number_of_models = number_of_models

        self.true_upper_boundary_condition_u_function = true_upper_boundary_condition_u_function
        self.true_lower_boundary_condition_u_function = true_lower_boundary_condition_u_function

        if self.model_number == 0:
            self.true_initial_condition_u_function = true_initial_condition_u_function
            self.true_initial_condition_h_function = true_initial_condition_h_function(
                initial_perturbation_amplitude,
                non_dimensionalization,
                horizontal_length_scale,
            )
        else:
            self.new_initial_conditions = new_initial_conditions
            self.new_initial_condition_sampling_points = new_initial_condition_sampling_points

        # problem parameters
        print("Initialize Problem Parameters.")
        self.non_dimensionalization = non_dimensionalization
        self.pi = np.pi
        self.momentum_advection = momentum_advection
        self.gravitational_acceleration = gravitational_acceleration
        self.average_sea_level = average_sea_level
        self.momentum_dissipation = momentum_dissipation
        self.nonlinear_drag_coefficient = nonlinear_drag_coefficient
        self.initial_perturbation_amplitude = initial_perturbation_amplitude

        # numerical space and time interval and reference scale lengths and time
        self.vertical_scaling_factor = vertical_scaling_factor
        self.vertical_length_scale = vertical_length_scale
        self.horizontal_length_scale = horizontal_length_scale
        self.time_scale = time_scale
        self.numerical_solution_time_interval = numerical_solution_time_interval
        self.numerical_solution_time_step = numerical_solution_time_step
        self.numerical_solution_x_interval = numerical_solution_x_interval
        self.numerical_solution_space_step = numerical_solution_space_step
        self.fraction_of_time_interval = fraction_of_time_interval

        # Hyper Parameters
        print("Initialize Hyper Parameters.")
        self.boundary_condition_transition_function = boundary_condition_transition_function
        self.initial_condition_transition_function = initial_condition_transition_function
        self.train_on_solution = train_on_solution
        self.train_on_PINNs_Loss = train_on_PINNs_Loss
        self.new_sampling_points = False
        self.learning_rate = learning_rate
        self.projected_gradients = projected_gradients
        self.line_search = line_search
        if self.projected_gradients is True:
            self.optimizer = PCGrad(torch.optim.Adam(self.parameters(), lr=learning_rate))
        elif optimizer == torch.optim.LBFGS:
            self.optimizer = optimizer(
                self.parameters(),
                lr=self.learning_rate,
                max_iter=20,
                max_eval=None,
                tolerance_grad=1.0 * np.finfo(float).eps,
                tolerance_change=1.0 * np.finfo(float).eps,
                history_size=20,
                line_search_fn=self.line_search,
            )
        elif optimizer == FullBatchLBFGS:
            self.optimizer = optimizer(self.parameters(), lr=self.learning_rate, line_search=self.line_search)
        else:
            self.optimizer = optimizer(self.parameters(), lr=self.learning_rate)
        self.epochs = training_steps
        self.batch_resampling_period = batch_resampling_period
        self.output_period = console_output_period
        self.boundary_condition_weight = boundary_condition_weight
        self.initial_condition_weight = initial_condition_weight
        self.symbolic_function_weight = symbolic_function_weight
        self.boundary_condition_batch_size = boundary_condition_batch_size
        self.initial_condition_batch_size = initial_condition_batch_size
        self.symbolic_function_batch_size = symbolic_function_batch_size
        self.pde_mini_batch_size = pde_mini_batch_size
        self.bc_mini_batch_size = bc_mini_batch_size
        self.ic_mini_batch_size = ic_mini_batch_size
        self.iterations_per_epoch = iterations_per_epoch
        self.device = device
        self.numerical_solution_directory = numerical_solution_directory

        # define Latin Hypercube Sampling (LHS) methods -> define limits for t in [0,1] and x in [-1,1]
        print("Define Data Sets.")
        self.minimum_time = minimum_time
        self.maximum_time = maximum_time
        self.minimum_x = minimum_x
        self.maximum_x = maximum_x
        self.pde_dataset = PDEDataset(
            minimum_time=self.minimum_time,
            maximum_time=self.maximum_time,
            minimum_x=self.minimum_x,
            maximum_x=self.maximum_x,
            batch_size=self.symbolic_function_batch_size,
        )
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

        # load discrete Representation of the numerical Solution
        print("Create grid and load solution.")
        self.solution_time_grid = np.arange(
            self.fraction_of_time_interval[0] * self.numerical_solution_time_interval[1],
            self.fraction_of_time_interval[1] * self.numerical_solution_time_interval[1],
            self.numerical_solution_time_step,
        )
        if self.model_number == 3:
            self.solution_time_grid = self.solution_time_grid[:-1]
        self.u_solution_x_grid = np.arange(
            self.numerical_solution_x_interval[0],
            self.numerical_solution_x_interval[1] + self.numerical_solution_space_step,
            self.numerical_solution_space_step,
        )
        self.zeta_solution_x_grid = (self.u_solution_x_grid - 0.5 * numerical_solution_space_step)[1:]
        self.time_slice_length = len(self.solution_time_grid)

        self.exact_solution_h_values = torch.FloatTensor(
            np.load(
                "Numerical_Solution/dt=1s_dx=400m/" + self.numerical_solution_directory + "/Sea_Level_Elevation.npy")[
            :, self.model_number * self.time_slice_length: (self.model_number + 1) * self.time_slice_length
            ]
        ).to(self.device)
        self.exact_solution_u_values = torch.FloatTensor(
            np.load("Numerical_Solution/dt=1s_dx=400m/" + self.numerical_solution_directory + "/Zonal_Velocity.npy")[
            :, self.model_number * self.time_slice_length: (self.model_number + 1) * self.time_slice_length
            ]
        ).to(self.device)

        [self.dimensional_zeta_solution_time_mesh_grid, self.dimensional_zeta_solution_x_mesh_grid] = np.meshgrid(
            self.solution_time_grid, self.zeta_solution_x_grid
        )
        [self.dimensional_u_solution_time_mesh_grid, self.dimensional_u_solution_x_mesh_grid] = np.meshgrid(
            self.solution_time_grid, self.u_solution_x_grid
        )

        if self.non_dimensionalization is True:
            self.solution_time_grid = self.solution_time_grid / self.time_scale
            self.zeta_solution_x_grid = self.zeta_solution_x_grid / self.horizontal_length_scale
            self.u_solution_x_grid = self.u_solution_x_grid / self.horizontal_length_scale

        [self.zeta_solution_time_mesh_grid, self.zeta_solution_x_mesh_grid] = np.meshgrid(
            self.solution_time_grid, self.zeta_solution_x_grid
        )
        [self.u_solution_time_mesh_grid, self.u_solution_x_mesh_grid] = np.meshgrid(
            self.solution_time_grid, self.u_solution_x_grid
        )
        self.zeta_solution_mesh_grid_shape = self.zeta_solution_x_mesh_grid.shape
        self.u_solution_mesh_grid_shape = self.u_solution_x_mesh_grid.shape
        self.zeta_solution_time_input_grid = torch.FloatTensor(
            self.zeta_solution_time_mesh_grid.reshape((np.size(self.zeta_solution_time_mesh_grid), 1))
        ).to(self.device)
        self.u_solution_time_input_grid = torch.FloatTensor(
            self.u_solution_time_mesh_grid.reshape((np.size(self.u_solution_time_mesh_grid), 1))
        ).to(self.device)
        self.zeta_solution_x_input_grid = torch.FloatTensor(
            self.zeta_solution_x_mesh_grid.reshape((np.size(self.zeta_solution_x_mesh_grid), 1))
        ).to(self.device)
        self.u_solution_x_input_grid = torch.FloatTensor(
            self.u_solution_x_mesh_grid.reshape((np.size(self.u_solution_x_mesh_grid), 1))
        ).to(self.device)

        # generate grid for tracking model output during training
        print("Create grid for tracking the model output.")
        self.x_grid = np.arange(
            self.minimum_x,
            self.maximum_x + (self.maximum_x - self.minimum_x) / 100.0,
            (self.maximum_x - self.minimum_x) / 100.0,
        )
        self.time_grid = np.arange(
            self.minimum_time,
            self.maximum_time + (self.maximum_time - self.minimum_time) / 100.0,
            (self.maximum_time - self.minimum_time) / 100.0,
        )
        [self.time_mesh_grid, self.x_mesh_grid] = np.meshgrid(self.time_grid, self.x_grid)
        self.time_input_grid = torch.FloatTensor(self.time_mesh_grid.reshape(np.size(self.time_mesh_grid), 1)).to(
            self.device
        )
        self.x_input_grid = torch.FloatTensor(self.x_mesh_grid.reshape(np.size(self.x_mesh_grid), 1)).to(self.device)
        self.mesh_grid_shape = self.time_mesh_grid.shape

        if self.non_dimensionalization is True:
            [self.dimensional_time_mesh_grid, self.dimensional_x_mesh_grid] = np.meshgrid(
                self.time_grid * self.time_scale, self.x_grid * self.horizontal_length_scale
            )

    def forward(self, t, x):

        tx = torch.cat((t, x), dim=1)

        if self.split_networks is True:
            output_u = self.activation_function(self.layers_u[0](tx))
            output_h = self.activation_function(self.layers_h[0](tx))
            for i in np.arange(1, self.number_of_layers - 1):
                output_u = self.activation_function(self.layers_u[i](output_u))
                output_h = self.activation_function(self.layers_h[i](output_h))
            output_u = self.layers_u[self.number_of_layers - 1](output_u)
            output_h = self.layers_h[self.number_of_layers - 1](output_h)
        else:
            if self.mixed_activation_functions is True:
                output = self.activation_function(self.layers[0](tx))
                for i in np.arange(1, self.number_of_layers - 1):
                    layer_output = self.layers[i](output)
                    output = torch.cat(
                        (
                            self.activation_function(layer_output[:, 0: int(0.5 * layer_output.size()[1])]),
                            self.relu(layer_output[:, int(0.5 * layer_output.size()[1]):]),
                        ),
                        dim=1,
                    )
                output = self.layers[self.number_of_layers - 1](output)
            else:
                output = self.activation_function(self.layers[0](tx))
                for i in np.arange(1, self.number_of_layers - 1):
                    output = self.activation_function(self.layers[i](output))
                output = self.layers[self.number_of_layers - 1](output)

            output_u = output[:, 0:1]
            output_h = output[:, 1:2]

        if self.boundary_condition_transition_function is True:
            tau_x = 0.1 * self.maximum_x
            output_u = (
                    (1.0 - torch.exp((x - self.maximum_x) / tau_x))
                    * (1.0 - torch.exp((-x - self.maximum_x) / tau_x))
                    * output_u
            )

        if self.initial_condition_transition_function is True:
            tau_t = 0.1 * self.maximum_time
            output_u = output_u * (1.0 - torch.exp(-t / tau_t))
            # version 1
            output_h = self.true_initial_condition_h_function(x) * torch.exp(-t / tau_t) + output_h * (
                    1.0 - torch.exp(-t / tau_t)
            )
            # version 2
            # output_h = self.true_initial_condition_h_function(x) + (1 - torch.exp(-t / tau_t)) * output_h

        output = torch.cat((output_u, output_h), dim=1)

        return output

    def Physics_Informed_Symbolic_Function(self):

        # compute network output
        t = self.symbolic_function_sampling_points[:, 0:1]
        x = self.symbolic_function_sampling_points[:, 1:2]
        output = self.forward(t, x)
        u = output[:, 0:1]
        h = output[:, 1:2]

        # compute gradients
        u_sum = u.sum()
        u_t = grad(u_sum, t, create_graph=True)[0].to(self.device)
        u_x = grad(u_sum, x, create_graph=True)[0].to(self.device)
        u_xx = grad(u_x.sum(), x, create_graph=True)[0].to(self.device)
        h_sum = h.sum()
        h_t = grad(h_sum, t, create_graph=True)[0].to(self.device)
        h_x = grad(h_sum, x, create_graph=True)[0].to(self.device)

        # scaling constants
        C1 = self.vertical_scaling_factor
        C2 = self.horizontal_length_scale

        # symbolic functions
        if self.non_dimensionalization is True:
            self.symbolic_function_u_values = (
                    u_t
                    + float(self.momentum_advection) * u * u_x
                    + C1 * h_x
                    + C2 * self.nonlinear_drag_coefficient * u * torch.abs(u)
                    - self.momentum_dissipation * u_xx
            )
            self.symbolic_function_h_values = h_t + u_x * (h + self.average_sea_level) + u * h_x
        else:
            self.symbolic_function_u_values = (
                    u_t
                    + u * u_x
                    + self.gravitational_acceleration * h_x
                    + self.nonlinear_drag_coefficient * u * torch.abs(u)
                    - self.momentum_dissipation * u_xx
            )
            self.symbolic_function_h_values = h_t + u_x * (h + self.average_sea_level) + u * h_x

    def MSE_boundary_conditions_function(self):

        self.MSE_lower_boundary_condition_u = (
                (self.network_output_lower_boundary_conditions[:,
                 0:1] - self.true_lower_boundary_condition_u_values) ** 2
        ).mean()
        self.MSE_upper_boundary_condition_u = (
                (self.network_output_upper_boundary_conditions[:,
                 0:1] - self.true_upper_boundary_condition_u_values) ** 2
        ).mean()
        self.MSE_boundary_conditions_value = self.MSE_lower_boundary_condition_u + self.MSE_upper_boundary_condition_u

    def MSE_initial_conditions_function(self):
        self.MSE_initial_condition_u = (
                (self.network_output_initial_conditions[:, 0:1] - self.true_initial_condition_u_values) ** 2
        ).mean()
        self.MSE_initial_condition_h = (
                (self.network_output_initial_conditions[:, 1:2] - self.true_initial_condition_h_values) ** 2
        ).mean()
        self.MSE_initial_condition_value = self.MSE_initial_condition_u + self.MSE_initial_condition_h

    def MSE_symbolic_functions(self):
        self.MSE_symbolic_function_u = (self.symbolic_function_u_values ** 2).mean()
        self.MSE_symbolic_function_h = (self.symbolic_function_h_values ** 2).mean()
        self.MSE_symbolic_function_value = self.MSE_symbolic_function_u + self.MSE_symbolic_function_h

    def total_MSE_function(self):
        self.total_MSE_value = 0.0

        if self.learning_rate_annealing is True:
            alpha = 0.9  # weight between new and old weight for update

            number_of_layers = len(self.layers)

            self.optimizer.zero_grad()
            self.MSE_symbolic_function_value.backward(retain_graph=True)
            gradient_tensor = self.layers[0].weight.grad.flatten()
            for k in range(1, number_of_layers):
                gradient_tensor = torch.cat((gradient_tensor, self.layers[k].weight.grad.flatten()))
                gradient_tensor = torch.cat((gradient_tensor, self.layers[k].bias.grad.flatten()))
            max_symbolic_function_grad = torch.max(abs(gradient_tensor))

            self.optimizer.zero_grad()
            self.MSE_initial_condition_value.backward(retain_graph=True)
            gradient_tensor = self.layers[0].weight.grad.flatten()
            for k in range(1, number_of_layers):
                gradient_tensor = torch.cat((gradient_tensor, self.layers[k].weight.grad.flatten()))
                gradient_tensor = torch.cat((gradient_tensor, self.layers[k].bias.grad.flatten()))
            mean_initial_condition_grad = torch.mean(abs(gradient_tensor))

            self.optimizer.zero_grad()
            self.MSE_boundary_conditions_value.backward(retain_graph=True)
            gradient_tensor = self.layers[0].weight.grad.flatten()
            for k in range(1, number_of_layers):
                gradient_tensor = torch.cat((gradient_tensor, self.layers[k].weight.grad.flatten()))
                gradient_tensor = torch.cat((gradient_tensor, self.layers[k].bias.grad.flatten()))
            mean_boundary_condition_grad = torch.mean(abs(gradient_tensor))
            self.optimizer.zero_grad()

            new_initial_condition_weight = max_symbolic_function_grad / mean_initial_condition_grad
            self.initial_condition_weight = ((1 - alpha)
                                             * self.initial_condition_weight
                                             + alpha * new_initial_condition_weight)

            new_boundary_condition_weight = max_symbolic_function_grad / mean_boundary_condition_grad
            self.boundary_condition_weight = ((1 - alpha)
                                              * self.boundary_condition_weight
                                              + alpha * new_boundary_condition_weight)

        # MSE containing the error at the initial and boundary conditions and the PDE loss term
        if self.train_on_PINNs_Loss is True:
            self.total_MSE_value += self.symbolic_function_weight * self.MSE_symbolic_function_value

        if self.initial_condition_transition_function is False and self.train_on_initial_condition_loss is True:
            self.total_MSE_value += self.initial_condition_weight * self.MSE_initial_condition_value

        if self.boundary_condition_transition_function is False and self.train_on_boundary_condition_loss is True:
            self.total_MSE_value += self.boundary_condition_weight * self.MSE_boundary_conditions_value

        # MSE comparing the network output to the numerical solution
        if self.train_on_solution is True:
            self.total_MSE_value += self.MSE_numerical_solution_h_value + self.MSE_numerical_solution_u_value

    def MSE_numerical_solution_function(self):
        self.network_output_u = self.forward(self.u_solution_time_input_grid, self.u_solution_x_input_grid)[:, 0:1]

        self.network_output_h = self.forward(self.zeta_solution_time_input_grid, self.zeta_solution_x_input_grid)[
                                :, 1:2
                                ]

        if self.non_dimensionalization is True:
            self.network_output_u = self.network_output_u * self.horizontal_length_scale / self.time_scale
            self.network_output_h = self.network_output_h * self.vertical_length_scale

        if self.initial_perturbation_amplitude == 0:
            self.MSE_numerical_solution_h_value = (self.network_output_h ** 2).mean()
            self.MSE_numerical_solution_u_value = (self.network_output_u ** 2).mean()
        else:
            self.MSE_numerical_solution_h_value = (
                    (self.exact_solution_h_values - self.network_output_h.reshape(
                        self.zeta_solution_mesh_grid_shape)) ** 2
            ).mean()
            self.MSE_numerical_solution_u_value = (
                    (self.exact_solution_u_values - self.network_output_u.reshape(self.u_solution_mesh_grid_shape)) ** 2
            ).mean()

    # reset datasets and re-sample collocation points
    def resample_sampling_points(self):
        self.pde_dataset = PDEDataset(
            minimum_time=self.minimum_time,
            maximum_time=self.maximum_time,
            minimum_x=self.minimum_x,
            maximum_x=self.maximum_x,
            batch_size=self.symbolic_function_batch_size,
        )
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

    def closure(self):
        self.optimizer.zero_grad()

        # compute network output for initial and boundary values
        self.network_output_lower_boundary_conditions = self.forward(
            self.lower_boundary_condition_sampling_points[:, 0:1], self.lower_boundary_condition_sampling_points[:, 1:2]
        )
        self.network_output_upper_boundary_conditions = self.forward(
            self.upper_boundary_condition_sampling_points[:, 0:1], self.upper_boundary_condition_sampling_points[:, 1:2]
        )

        if self.model_number == 0:
            self.network_output_initial_conditions = self.forward(
                self.initial_condition_sampling_points[:, 0:1], self.initial_condition_sampling_points[:, 1:2]
            )
        else:
            self.network_output_initial_conditions = self.forward(
                self.new_initial_condition_sampling_points[:, 0:1], self.new_initial_condition_sampling_points[:, 1:2]
            )

        self.true_upper_boundary_condition_u_values = self.true_upper_boundary_condition_u_function(
            self.upper_boundary_condition_sampling_points[:, 0:1]
        )
        self.true_lower_boundary_condition_u_values = self.true_lower_boundary_condition_u_function(
            self.lower_boundary_condition_sampling_points[:, 0:1]
        )

        if self.model_number == 0:
            self.true_initial_condition_u_values = self.true_initial_condition_u_function(
                self.initial_condition_sampling_points[:, 1:2]
            )
            self.true_initial_condition_h_values = self.true_initial_condition_h_function(
                self.initial_condition_sampling_points[:, 1:2]
            )

        else:
            self.true_initial_condition_u_values = self.new_initial_conditions[:, 0:1]
            self.true_initial_condition_h_values = self.new_initial_conditions[:, 1:2]

        self.new_sampling_points = False

        # compute symbolic function from network output
        self.Physics_Informed_Symbolic_Function()

        # compute Mean Squared Error (MSE) terms
        if self.train_on_solution is True:
            self.MSE_numerical_solution_function()
        self.MSE_boundary_conditions_function()
        self.MSE_initial_conditions_function()
        self.MSE_symbolic_functions()
        self.total_MSE_function()

        # compute gradients
        if self.projected_gradients is False or self.train_on_solution is True:
            self.total_MSE_value.backward(retain_graph=False)  # No retain_graph -> keeps gradient over several steps

        return self.total_MSE_value

    def Save_MSE(self):
        self.MSE_boundary_conditions_over_training.append(self.MSE_boundary_conditions_value.cpu().detach().numpy())
        self.MSE_initial_conditions_over_training.append(self.MSE_initial_condition_value.cpu().detach().numpy())
        self.MSE_symbolic_functions_over_training.append(self.MSE_symbolic_function_value.cpu().detach().numpy())
        self.total_MSE_over_training.append(self.total_MSE_value.cpu().detach().numpy())
        self.MSE_numerical_solution_h_over_training.append(self.MSE_numerical_solution_h_value.cpu().detach().numpy())
        self.MSE_numerical_solution_u_over_training.append(self.MSE_numerical_solution_u_value.cpu().detach().numpy())
        self.MSE_initial_condition_u_over_training.append(self.MSE_initial_condition_u.cpu().detach().numpy())
        self.MSE_initial_condition_h_over_training.append(self.MSE_initial_condition_h.cpu().detach().numpy())
        self.MSE_symbolic_function_u_over_training.append(self.MSE_symbolic_function_u.cpu().detach().numpy())
        self.MSE_symbolic_function_h_over_training.append(self.MSE_symbolic_function_h.cpu().detach().numpy())
        self.MSE_lower_boundary_condition_u_over_training.append(
            self.MSE_lower_boundary_condition_u.cpu().detach().numpy()
        )
        self.MSE_upper_boundary_condition_u_over_training.append(
            self.MSE_upper_boundary_condition_u.cpu().detach().numpy()
        )

    def Save_Network_Output(self):
        self.network_output = self.forward(self.time_input_grid, self.x_input_grid)
        self.network_output_u_over_training.append(
            self.network_output[:, 0:1].cpu().detach().numpy().reshape(self.mesh_grid_shape)
        )
        self.network_output_h_over_training.append(
            self.network_output[:, 1:2].cpu().detach().numpy().reshape(self.mesh_grid_shape)
        )

    def Save_Symbolic_Functions(self):
        sampling_points = self.symbolic_function_sampling_points
        self.symbolic_function_sampling_points = torch.hstack(
            (
                self.time_input_grid.clone().detach().requires_grad_(True),
                self.x_input_grid.clone().detach().requires_grad_(True),
            )
        )
        self.Physics_Informed_Symbolic_Function()
        PDE_Loss_u = self.symbolic_function_u_values.reshape(shape=self.mesh_grid_shape)
        PDE_Loss_h = self.symbolic_function_h_values.reshape(shape=self.mesh_grid_shape)
        self.symbolic_function_u_over_training.append(PDE_Loss_u.cpu().detach().numpy().reshape(self.mesh_grid_shape))
        self.symbolic_function_h_over_training.append(PDE_Loss_h.cpu().detach().numpy().reshape(self.mesh_grid_shape))
        self.symbolic_function_sampling_points = sampling_points

    def Save_Weights_Over_Training(self):
        self.symbolic_function_weight_over_training.append(self.symbolic_function_weight)
        self.initial_condition_weight_over_training.append(np.asscalar(self.initial_condition_weight.cpu().numpy()))
        self.boundary_condition_weight_over_training.append(np.asscalar(self.boundary_condition_weight.cpu().numpy()))

    def Save_Final_State(self):
        self.final_state_sampling_points = self.final_state_dataset.tx.to(self.device)
        new_initial_conditions = self.forward(
            self.final_state_sampling_points[:, 0:1], self.final_state_sampling_points[:, 1:2]
        )
        final_state_sampling_points = self.final_state_sampling_points.cpu().detach().numpy()
        return new_initial_conditions.cpu().detach().numpy(), final_state_sampling_points

    def train_PINN(self):
        self.symbolic_function_sampling_points = self.pde_dataset.tx.to(self.device)
        self.lower_boundary_condition_sampling_points = self.lbc_dataset.tx.to(self.device)
        self.upper_boundary_condition_sampling_points = self.ubc_dataset.tx.to(self.device)
        self.initial_condition_sampling_points = self.ic_dataset.tx.to(self.device)

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

        # evaluate Model before Training
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

        # initialize values for training
        lowest_validation_error = 99999.0
        self.time_per_epoch = []
        self.time_per_iteration = []

        print("Begin training:")
        for i in range(self.epochs):

            epoch_start_time = time.time()

            self.pde_dataloader = DataLoader(dataset=self.pde_dataset, batch_size=self.pde_mini_batch_size,
                                             shuffle=True)
            self.pde_dataiter = iter(self.pde_dataloader)
            self.lbc_dataloader = DataLoader(dataset=self.lbc_dataset, batch_size=self.bc_mini_batch_size, shuffle=True)
            self.lbc_dataiter = iter(self.lbc_dataloader)
            self.ubc_dataloader = DataLoader(dataset=self.ubc_dataset, batch_size=self.bc_mini_batch_size, shuffle=True)
            self.ubc_dataiter = iter(self.ubc_dataloader)
            self.ic_dataloader = DataLoader(dataset=self.ic_dataset, batch_size=self.ic_mini_batch_size, shuffle=True)
            self.ic_dataiter = iter(self.ic_dataloader)

            for j in range(self.iterations_per_epoch):

                iteration_start_time = time.time()

                # Latin Hypercube Sampling
                if (i + 1) % self.batch_resampling_period == 0 and i > 0:
                    self.resample_sampling_points()
                    print("Change Sampling Points")

                if self.minibatch_training is True:
                    self.symbolic_function_sampling_points = next(self.pde_dataiter).to(self.device)
                    self.lower_boundary_condition_sampling_points = next(self.lbc_dataiter).to(self.device)
                    self.upper_boundary_condition_sampling_points = next(self.ubc_dataiter).to(self.device)
                    self.initial_condition_sampling_points = next(self.ic_dataiter).to(self.device)

                # Model Evaluation, Backward Propagation and Training Step
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

            if (i + 1) % self.output_period == 0:

                # save Model Output at current training step to track its development over training
                if self.save_output_over_training is True and (i + 1) % 100 == 0:
                    self.Save_Network_Output()

                # save PDE Losses at current training step to track their development over training
                if self.save_symbolic_function_over_training is True and (i + 1) % 100 == 0:
                    self.Save_Symbolic_Functions()

                self.MSE_numerical_solution_function()
                self.Save_MSE()
                # save best state dict
                validation_error = self.MSE_numerical_solution_u_value + self.MSE_numerical_solution_h_value
                if (validation_error < lowest_validation_error) or i == 1:
                    lowest_validation_error = validation_error
                    torch.save(
                        self.state_dict(), (str(self.best_state_dir) + "/"+ "Best_State_Dict_" + str(self.model_number) + "_" + str(self.exp_name))
                    )
                    self.improvement_steps.append(float(i))
                    self.best_step = float(i)

                print("Epoch no.: " + str(i + 1) + "/" + str(self.epochs) + " iteration no. " + str(j + 1) + "/"
                      + str(self.iterations_per_epoch))
                print("Boundary Condition Loss = ", self.MSE_boundary_conditions_value.cpu().detach().numpy())
                print("Initial Condition Loss = ", self.MSE_initial_condition_value.cpu().detach().numpy())
                print("Symbolic Function Loss = ", self.MSE_symbolic_function_value.cpu().detach().numpy())
                print("Numerical Solution Loss u = ", self.MSE_numerical_solution_u_value.cpu().detach().numpy())
                print("Numerical Solution Loss h = ", self.MSE_numerical_solution_h_value.cpu().detach().numpy())

                if self.learning_rate_annealing is True:
                    print("Symbolic Function weight: ", self.symbolic_function_weight)
                    print("Initial Condition weight: ", np.asscalar(self.initial_condition_weight.cpu().numpy()))
                    print("Boundary Condition weight: ", np.asscalar(self.boundary_condition_weight.cpu().numpy()))

            self.time_per_epoch.append(time.time() - epoch_start_time)
            print("Time Per Epoch: " + str(np.mean(self.time_per_epoch)) + "s")

        self.symbolic_function_sampling_points = self.pde_dataset.tx.to(self.device)
        self.lower_boundary_condition_sampling_points = self.lbc_dataset.tx.to(self.device)
        self.upper_boundary_condition_sampling_points = self.ubc_dataset.tx.to(self.device)
        self.initial_condition_sampling_points = self.ic_dataset.tx.to(self.device)
        print("Training completed.")
