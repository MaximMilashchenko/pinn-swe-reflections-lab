import numpy as np
import torch

# define true Initial and Boundary Conditions for the sea level elevation h and the zonal velocity u


def true_initial_condition_h_function(
    initial_perturbation_amplitude,
    non_dimensionalization,
    horizontal_length_scale,
):
    def f(x):
        mu = 0.0
        sigma = 100000.0
        pi = torch.tensor(np.pi)

        if non_dimensionalization == True:
            x = x * horizontal_length_scale

        true_initial_condition_h_values = (
            2.5
            * initial_perturbation_amplitude
            * (sigma * (1.0 / (sigma * torch.sqrt(2.0 * pi))) * torch.exp(-0.5 * ((x - mu) / sigma) ** 2.0))
        )

        return true_initial_condition_h_values

    return f


def true_initial_condition_u_function(x):
    # the zonal velocity is zero everywhere at the initial state
    true_initial_condition_u_values = torch.zeros_like(x)
    return true_initial_condition_u_values


def true_upper_boundary_condition_u_function(t):
    # the zonal velocity is zero at the boundaries to ensure closed boundaries
    true_upper_boundary_condition_u_values = torch.zeros_like(t)
    return true_upper_boundary_condition_u_values


def true_lower_boundary_condition_u_function(t):
    # the zonal velocity is zero at the boundaries to ensure closed boundaries
    true_lower_boundary_condition_u_values = torch.zeros_like(t)
    return true_lower_boundary_condition_u_values
