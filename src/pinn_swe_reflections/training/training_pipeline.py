import json
import os
import time

import torch
import numpy as np

from pinn_swe_reflections.common.paths import get_training_run_dir


def run_training_pipeline(model_cls, experiment_name, cfg, model_kwargs=None):
    run_dir = get_training_run_dir(experiment_name)
    run_dir.mkdir(parents=True, exist_ok=True)

    def out_path(file_name):
        return run_dir / file_name

    def save_npy(file_name, value):
        np.save(out_path(file_name), value)

    def save_json(file_name, value):
        out_path(file_name).write_text(
            json.dumps(value, indent=4),
            encoding="utf-8",
        )

    model_kwargs = {} if model_kwargs is None else dict(model_kwargs)

    start = time.time()
    local_new_initial_conditions = None
    local_new_initial_condition_sampling_points = None
    last_model = None

    for model_number in cfg.model_range:

        # adjust time interval to the number of models
        fraction_of_time_interval = [0.0 + model_number / cfg.number_of_models, (model_number + 1) / cfg.number_of_models]
        minimum_time = cfg.numerical_solution_time_interval[0] + fraction_of_time_interval[0] * (
                cfg.numerical_solution_time_interval[1] - cfg.numerical_solution_time_interval[0]
        )
        maximum_time = cfg.numerical_solution_time_interval[0] + fraction_of_time_interval[1] * (
                cfg.numerical_solution_time_interval[1] - cfg.numerical_solution_time_interval[0]
        )  # [s]

        if cfg.non_dimensionalization is True:
            minimum_time = minimum_time / cfg.time_scale
            maximum_time = maximum_time / cfg.time_scale

        minimum_x = cfg.numerical_solution_x_interval[0]  # lower boundary of the spatial interval for training [m]
        maximum_x = cfg.numerical_solution_x_interval[1]  # upper boundary of the spatial interval for training [m]
        if cfg.non_dimensionalization is True:
            minimum_x = minimum_x / cfg.horizontal_length_scale
            maximum_x = maximum_x / cfg.horizontal_length_scale

        # Initialization of the Physics Informed Neural Network
        Physics_Informed_Neural_Network = model_cls(
            cfg.layer_sizes,
            cfg.activation_function,
            cfg.optimizer,
            cfg.learning_rate,
            cfg.line_search,
            cfg.boundary_condition_weight,
            cfg.initial_condition_weight,
            cfg.symbolic_function_weight,
            int(cfg.boundary_condition_batch_size / cfg.number_of_models),
            int(cfg.initial_condition_batch_size),
            int(cfg.symbolic_function_batch_size / cfg.number_of_models),
            int(cfg.epochs / cfg.number_of_models),
            cfg.batch_resampling_period,
            cfg.output_period,
            cfg.device,
            cfg.gravitational_acceleration,
            cfg.average_sea_level,
            cfg.momentum_dissipation,
            cfg.nonlinear_drag_coefficient,
            cfg.initial_perturbation_amplitude,
            cfg.non_dimensionalization,
            cfg.vertical_length_scale,
            cfg.vertical_scaling_factor,
            cfg.horizontal_length_scale,
            cfg.time_scale,
            minimum_time,
            maximum_time,
            minimum_x,
            maximum_x,
            cfg.projected_gradients,
            cfg.save_output_over_training,
            cfg.save_symbolic_function_over_training,
            cfg.numerical_solution_time_interval,
            cfg.numerical_solution_time_step,
            cfg.numerical_solution_x_interval,
            cfg.numerical_solution_space_step,
            fraction_of_time_interval,
            model_number,
            cfg.number_of_models,
            local_new_initial_conditions,
            local_new_initial_condition_sampling_points,
            cfg.train_on_solution,
            cfg.train_on_PINNs_Loss,
            cfg.boundary_condition_transition_function,
            cfg.initial_condition_transition_function,
            cfg.split_networks,
            cfg.train_on_boundary_condition_loss,
            cfg.train_on_initial_condition_loss,
            cfg.momentum_advection,
            cfg.mixed_activation_functions,
            cfg.sirens_initialization,
            cfg.learning_rate_annealing,
            cfg.minibatch_training,
            cfg.pde_mini_batch_size,
            cfg.bc_mini_batch_size,
            cfg.ic_mini_batch_size,
            cfg.iterations_per_epoch,
            cfg.numerical_solution_directory,
            **model_kwargs,
        ).to(cfg.device)

        # Train Model with parameters chosen above -> generate model output and MSE over training
        Physics_Informed_Neural_Network.train_PINN()

        end = time.time()
        computation_time = end - start

        # get best state dict
        best_step = Physics_Informed_Neural_Network.best_step
        best_state_dict = Physics_Informed_Neural_Network.get_best_state_dict()

        # Save Model Parameters
        torch.save(best_state_dict, out_path("TrainedParameters_SWE_" + str(model_number) + ".pt"))

        # Save new initial conditions and respective sampling points
        [local_new_initial_conditions, local_new_initial_condition_sampling_points] = Physics_Informed_Neural_Network.Save_Final_State()

        save_npy("improvement_steps", Physics_Informed_Neural_Network.improvement_steps)
        # save different loss terms over training
        save_npy("total_MSE_over_training", Physics_Informed_Neural_Network.total_MSE_over_training)
        save_npy(
            "MSE_boundary_conditions_over_training", Physics_Informed_Neural_Network.MSE_boundary_conditions_over_training
        )
        save_npy(
            "MSE_initial_conditions_over_training", Physics_Informed_Neural_Network.MSE_initial_conditions_over_training
        )
        save_npy(
            "MSE_symbolic_functions_over_training", Physics_Informed_Neural_Network.MSE_symbolic_functions_over_training
        )
        save_npy("Relative_L2_Error_h_over_training", Physics_Informed_Neural_Network.MSE_numerical_solution_h_over_training)
        save_npy("Relative_L2_Error_u_over_training", Physics_Informed_Neural_Network.MSE_numerical_solution_u_over_training)
        save_npy(
            "MSE_initial_condition_u_over_training", Physics_Informed_Neural_Network.MSE_initial_condition_u_over_training
        )
        save_npy(
            "MSE_initial_condition_h_over_training", Physics_Informed_Neural_Network.MSE_initial_condition_h_over_training
        )
        save_npy(
            "MSE_symbolic_function_u_over_training", Physics_Informed_Neural_Network.MSE_symbolic_function_u_over_training
        )
        save_npy(
            "MSE_symbolic_function_h_over_training", Physics_Informed_Neural_Network.MSE_symbolic_function_h_over_training
        )
        save_npy(
            "MSE_symbolic_function_h_over_training", Physics_Informed_Neural_Network.MSE_symbolic_function_h_over_training
        )

        save_npy(
            "MSE_lower_boundary_condition_u_over_training",
            Physics_Informed_Neural_Network.MSE_lower_boundary_condition_u_over_training,
        )
        save_npy(
            "MSE_upper_boundary_condition_u_over_training",
            Physics_Informed_Neural_Network.MSE_upper_boundary_condition_u_over_training,
        )

        extra_loss_arrays = [
            "MSE_symbolic_function_without_gpinn_over_training",
            "MSE_gPINN_gradient_over_training",
            "MSE_gPINN_u_t_over_training",
            "MSE_gPINN_h_t_over_training",
            "MSE_gPINN_u_x_over_training",
            "MSE_gPINN_h_x_over_training",
        ]
        for array_name in extra_loss_arrays:
            if hasattr(Physics_Informed_Neural_Network, array_name):
                save_npy(array_name, getattr(Physics_Informed_Neural_Network, array_name))

        if Physics_Informed_Neural_Network.learning_rate_annealing is True:
            save_npy(
                "symbolic_function_weight_over_training",
                Physics_Informed_Neural_Network.symbolic_function_weight_over_training,
            )
            save_npy(
                "initial_condition_weight_over_training",
                Physics_Informed_Neural_Network.initial_condition_weight_over_training,
            )
            save_npy(
                "boundary_condition_weight_over_training",
                Physics_Informed_Neural_Network.boundary_condition_weight_over_training,
            )

        # save grids for plots
        if Physics_Informed_Neural_Network.non_dimensionalization is True:
            save_npy("dimensional_time_mesh_grid", Physics_Informed_Neural_Network.dimensional_time_mesh_grid)
            save_npy("dimensional_x_mesh_grid", Physics_Informed_Neural_Network.dimensional_x_mesh_grid)

        save_npy("time_mesh_grid", Physics_Informed_Neural_Network.time_mesh_grid)
        save_npy("x_mesh_grid", Physics_Informed_Neural_Network.x_mesh_grid)
        save_npy("time_input_grid", Physics_Informed_Neural_Network.time_input_grid.cpu().detach().numpy())
        save_npy("x_input_grid", Physics_Informed_Neural_Network.x_input_grid.cpu().detach().numpy())
        save_npy("mesh_grid_shape", Physics_Informed_Neural_Network.mesh_grid_shape)
        save_npy(
            "zeta_solution_time_input_grid",
            Physics_Informed_Neural_Network.zeta_solution_time_input_grid.cpu().detach().numpy(),
        )
        save_npy(
            "u_solution_time_input_grid", Physics_Informed_Neural_Network.u_solution_time_input_grid.cpu().detach().numpy()
        )
        save_npy(
            "zeta_solution_x_input_grid", Physics_Informed_Neural_Network.zeta_solution_x_input_grid.cpu().detach().numpy()
        )
        save_npy("u_solution_x_input_grid", Physics_Informed_Neural_Network.u_solution_x_input_grid.cpu().detach().numpy())
        save_npy(
            "dimensional_zeta_solution_time_mesh_grid",
            Physics_Informed_Neural_Network.dimensional_zeta_solution_time_mesh_grid,
        )
        save_npy(
            "dimensional_u_solution_time_mesh_grid", Physics_Informed_Neural_Network.dimensional_u_solution_time_mesh_grid
        )
        save_npy(
            "dimensional_zeta_solution_x_mesh_grid", Physics_Informed_Neural_Network.dimensional_zeta_solution_x_mesh_grid
        )
        save_npy("dimensional_u_solution_x_mesh_grid", Physics_Informed_Neural_Network.dimensional_u_solution_x_mesh_grid)
        save_npy(
            "dimensional_solution_mesh_grid_shape",
            Physics_Informed_Neural_Network.dimensional_zeta_solution_x_mesh_grid.shape,
        )

        # save output
        network_output_h_values = (
            Physics_Informed_Neural_Network(
                Physics_Informed_Neural_Network.zeta_solution_time_input_grid,
                Physics_Informed_Neural_Network.zeta_solution_x_input_grid,
            )[:, 1:2]
            .cpu()
            .detach()
            .numpy()
            .reshape(Physics_Informed_Neural_Network.zeta_solution_mesh_grid_shape)
        )
        network_output_u_values = (
            Physics_Informed_Neural_Network(
                Physics_Informed_Neural_Network.u_solution_time_input_grid,
                Physics_Informed_Neural_Network.u_solution_x_input_grid,
            )[:, 0:1]
            .cpu()
            .detach()
            .numpy()
            .reshape(Physics_Informed_Neural_Network.u_solution_mesh_grid_shape)
        )
        save_npy("network_output_h_values", network_output_h_values)
        save_npy("network_output_u_values", network_output_u_values)

        if Physics_Informed_Neural_Network.non_dimensionalization is True:
            dimensional_network_output_h_values = (
                    network_output_h_values * Physics_Informed_Neural_Network.vertical_length_scale
            )
            dimensional_network_output_u_values = (
                    network_output_u_values
                    * Physics_Informed_Neural_Network.horizontal_length_scale
                    / Physics_Informed_Neural_Network.time_scale
            )
            save_npy("dimensional_network_output_h_values", dimensional_network_output_h_values)
            save_npy("dimensional_network_output_u_values", dimensional_network_output_u_values)

        # save numerical solution
        exact_solution_u_values = Physics_Informed_Neural_Network.exact_solution_u_values.cpu().detach().numpy()
        exact_solution_h_values = Physics_Informed_Neural_Network.exact_solution_h_values.cpu().detach().numpy()
        save_npy("exact_solution_u_values", exact_solution_u_values)
        save_npy("exact_solution_h_values", exact_solution_h_values)

        abs_error_h_values = abs(exact_solution_h_values - dimensional_network_output_h_values)
        abs_error_u_values = abs(exact_solution_u_values - dimensional_network_output_u_values)
        save_npy("abs_error_h_values", abs_error_h_values)
        save_npy("abs_error_u_values", abs_error_u_values)

        # save initial condition output
        minimum_time = Physics_Informed_Neural_Network.minimum_time
        x = (
            torch.FloatTensor(Physics_Informed_Neural_Network.x_grid)
            .unsqueeze(dim=1)
            .to(Physics_Informed_Neural_Network.device)
        )
        t = torch.zeros_like(x).to(Physics_Informed_Neural_Network.device)

        # define network output and target initial condition
        if Physics_Informed_Neural_Network.model_number == 0:
            true_initial_condition_h_values = (
                Physics_Informed_Neural_Network.true_initial_condition_h_function(x).cpu().detach().numpy()
            )
            network_output_h_initial_conditions = Physics_Informed_Neural_Network(t, x)[:, 1:2].cpu().detach().numpy()
            true_initial_condition_u_values = (
                Physics_Informed_Neural_Network.true_initial_condition_u_function(x).cpu().detach().numpy()
            )
            network_output_u_initial_conditions = Physics_Informed_Neural_Network(t, x)[:, 0:1].cpu().detach().numpy()

            if Physics_Informed_Neural_Network.non_dimensionalization is True:
                x = x * Physics_Informed_Neural_Network.horizontal_length_scale
                true_initial_condition_h_values = (
                        true_initial_condition_h_values * Physics_Informed_Neural_Network.vertical_length_scale
                )
                network_output_h_initial_conditions = (
                        network_output_h_initial_conditions * Physics_Informed_Neural_Network.vertical_length_scale
                )
                true_initial_condition_u_values = (
                        true_initial_condition_u_values
                        * Physics_Informed_Neural_Network.horizontal_length_scale
                        / Physics_Informed_Neural_Network.time_scale
                )
                network_output_u_initial_conditions = (
                        network_output_u_initial_conditions
                        * Physics_Informed_Neural_Network.horizontal_length_scale
                        / Physics_Informed_Neural_Network.time_scale
                )
                minimum_time = minimum_time * Physics_Informed_Neural_Network.time_scale

            save_npy("true_initial_condition_h_values", true_initial_condition_h_values)
            save_npy("network_output_h_initial_conditions", network_output_h_initial_conditions)
            save_npy("true_initial_condition_u_values", true_initial_condition_u_values)
            save_npy("network_output_u_initial_conditions", network_output_u_initial_conditions)

        # save boundary condition output
        dimensional_minimum_x = Physics_Informed_Neural_Network.minimum_x
        dimensional_maximum_x = Physics_Informed_Neural_Network.maximum_x
        t = (
            torch.FloatTensor(Physics_Informed_Neural_Network.time_grid)
            .unsqueeze(dim=1)
            .to(Physics_Informed_Neural_Network.device)
        )
        x_lower_boundary = -1.0 * torch.ones_like(t).to(Physics_Informed_Neural_Network.device)
        x_upper_boundary = 1.0 * torch.ones_like(t).to(Physics_Informed_Neural_Network.device)

        true_lower_boundary_condition_u_values = (
            Physics_Informed_Neural_Network.true_lower_boundary_condition_u_function(t).cpu().detach().numpy()
        )
        true_upper_boundary_condition_u_values = (
            Physics_Informed_Neural_Network.true_upper_boundary_condition_u_function(t).cpu().detach().numpy()
        )
        network_output_u_lower_boundary_condition = (
            Physics_Informed_Neural_Network(t, x_lower_boundary)[:, 0:1].cpu().detach().numpy()
        )
        network_output_u_upper_boundary_condition = (
            Physics_Informed_Neural_Network(t, x_upper_boundary)[:, 0:1].cpu().detach().numpy()
        )

        # re-dimensionalize
        if Physics_Informed_Neural_Network.non_dimensionalization is True:
            t = t * Physics_Informed_Neural_Network.time_scale
            true_lower_boundary_condition_u_values = (
                    true_lower_boundary_condition_u_values
                    * Physics_Informed_Neural_Network.horizontal_length_scale
                    / Physics_Informed_Neural_Network.time_scale
            )
            network_output_u_lower_boundary_condition = (
                    network_output_u_lower_boundary_condition
                    * Physics_Informed_Neural_Network.horizontal_length_scale
                    / Physics_Informed_Neural_Network.time_scale
            )
            true_upper_boundary_condition_u_values = (
                    true_upper_boundary_condition_u_values
                    * Physics_Informed_Neural_Network.horizontal_length_scale
                    / Physics_Informed_Neural_Network.time_scale
            )
            network_output_u_upper_boundary_condition = (
                    network_output_u_upper_boundary_condition
                    * Physics_Informed_Neural_Network.horizontal_length_scale
                    / Physics_Informed_Neural_Network.time_scale
            )
            dimensional_minimum_x = dimensional_minimum_x * Physics_Informed_Neural_Network.horizontal_length_scale
            dimensional_maximum_x = dimensional_maximum_x * Physics_Informed_Neural_Network.horizontal_length_scale

        save_npy("true_lower_boundary_condition_u_values", true_lower_boundary_condition_u_values)
        save_npy("network_output_u_lower_boundary_condition", network_output_u_lower_boundary_condition)
        save_npy("true_upper_boundary_condition_u_values", true_upper_boundary_condition_u_values)
        save_npy("network_output_u_upper_boundary_condition", network_output_u_upper_boundary_condition)

        # save PDE losses over domain
        mesh_grid_shape = Physics_Informed_Neural_Network.mesh_grid_shape
        t = Physics_Informed_Neural_Network.time_input_grid.clone().detach().requires_grad_(True)
        x = Physics_Informed_Neural_Network.x_input_grid.clone().detach().requires_grad_(True)
        Physics_Informed_Neural_Network.symbolic_function_sampling_points = torch.hstack((t, x))
        Physics_Informed_Neural_Network.Physics_Informed_Symbolic_Function()

        time_mesh_grid = Physics_Informed_Neural_Network.time_mesh_grid * Physics_Informed_Neural_Network.time_scale
        x_mesh_grid = Physics_Informed_Neural_Network.x_mesh_grid * Physics_Informed_Neural_Network.horizontal_length_scale
        PDE_Loss_u = Physics_Informed_Neural_Network.symbolic_function_u_values.reshape(shape=mesh_grid_shape)
        PDE_Loss_h = Physics_Informed_Neural_Network.symbolic_function_h_values.reshape(shape=mesh_grid_shape)
        save_npy("PDE_Loss_u", PDE_Loss_u.cpu().detach().numpy())
        save_npy("PDE_Loss_h", PDE_Loss_h.cpu().detach().numpy())

        # save network output & losses over training
        save_npy("network_output_h_over_training", Physics_Informed_Neural_Network.network_output_h_over_training)
        save_npy("network_output_u_over_training", Physics_Informed_Neural_Network.network_output_u_over_training)
        save_npy("symbolic_function_u_over_training", Physics_Informed_Neural_Network.symbolic_function_u_over_training)
        save_npy("symbolic_function_h_over_training", Physics_Informed_Neural_Network.symbolic_function_h_over_training)

        print("Mean Time Per Epoch: " + str(np.mean(Physics_Informed_Neural_Network.time_per_epoch)))
        print("Mean Time Per Iteration: " + str(np.mean(Physics_Informed_Neural_Network.time_per_iteration)))

        save_npy("time_per_epoch", Physics_Informed_Neural_Network.time_per_epoch)
        save_npy("time_per_iteration", Physics_Informed_Neural_Network.time_per_iteration)

        Hyper_Parameter_Dictionary = {
            "experiment_name": experiment_name,
            "model_class": model_cls.__name__,
            "model_kwargs": model_kwargs,
            "number_of_models": cfg.number_of_models,
            "split_networks": cfg.split_networks,
            "boundary_condition_transition_function": cfg.boundary_condition_transition_function,
            "initial_condition_transition_function": cfg.initial_condition_transition_function,
            "non_dimensionalization": cfg.non_dimensionalization,
            "save_output_over_training": cfg.save_output_over_training,
            "save_symbolic_function_over_training": cfg.save_symbolic_function_over_training,
            "train_on_solution": cfg.train_on_solution,
            "train_on_PINNs_Loss": cfg.train_on_PINNs_Loss,
            "train_on_boundary_condition_loss": cfg.train_on_boundary_condition_loss,
            "train_on_initial_condition_loss": cfg.train_on_initial_condition_loss,
            "momentum_advection": cfg.momentum_advection,
            "initial_perturbation_amplitude": cfg.initial_perturbation_amplitude,
            "average_sea_level": cfg.average_sea_level,
            "gravitational_acceleration": cfg.gravitational_acceleration,
            "momentum_dissipation": cfg.momentum_dissipation,
            "nonlinear_drag_coefficient": cfg.nonlinear_drag_coefficient,
            "horizontal_length_scale": cfg.horizontal_length_scale,
            "time_scale": cfg.time_scale,
            "vertical_scaling_factor": cfg.vertical_scaling_factor,
            "vertical_length_scale": cfg.vertical_length_scale,
            "numerical_solution_time_interval": cfg.numerical_solution_time_interval,
            "numerical_solution_time_step": cfg.numerical_solution_time_step,
            "numerical_solution_x_interval": cfg.numerical_solution_x_interval,
            "numerical_solution_space_step": cfg.numerical_solution_space_step,
            "minimum_x": minimum_x,
            "maximum_x": maximum_x,
            "minimum_time": minimum_time,
            "maximum_time": maximum_time,
            "boundary_condition_weight": cfg.boundary_condition_weight,
            "initial_condition_weight": cfg.initial_condition_weight,
            "symbolic_function_weight": cfg.symbolic_function_weight,
            "boundary_condition_batch_size": cfg.boundary_condition_batch_size,
            "initial_condition_batch_size": cfg.initial_condition_batch_size,
            "symbolic_function_batch_size": cfg.symbolic_function_batch_size,
            "device": cfg.device,
            "epochs": cfg.epochs,
            "batch_resampling_period": cfg.batch_resampling_period,
            "console_output_period": cfg.output_period,
            "optimizer": str(cfg.optimizer),
            "learning_rate": cfg.learning_rate,
            "line_search": cfg.line_search,
            "projected_gradients": cfg.projected_gradients,
            "number_of_layers": cfg.number_of_layers,
            "neurons_per_layer": cfg.neurons_per_layer,
            "activation_function": str(cfg.activation_function),
            "model_number": float(model_number),
            "mixed_activation_functions": cfg.mixed_activation_functions,
            "computation_time": computation_time,
            "best_step": best_step,
            "learning_rate_annealing": cfg.learning_rate_annealing,
            "pde_mini_batch_size": cfg.pde_mini_batch_size,
            "bc_mini_batch_size": cfg.bc_mini_batch_size,
            "ic_mini_batch_size": cfg.ic_mini_batch_size,
            "iterations_per_epoch": cfg.iterations_per_epoch,
        }

        save_json("Hyper_Parameter_Dictionary.json", Hyper_Parameter_Dictionary)

        last_model = Physics_Informed_Neural_Network

    return last_model, run_dir