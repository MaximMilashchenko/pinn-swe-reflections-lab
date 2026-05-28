import torch
import numpy as np

from dataclasses import dataclass


@dataclass
class TrainConfig:
    numerical_solution_directory = "AH=5e+4"  # "AH=5e+4" or "AH=0"

    # training options
    minibatch_training = True  # mini batch gradient descent
    learning_rate_annealing = False  # adaptive loss function weights
    mixed_activation_functions = False  # combination of relu and other activations
    number_of_models = 1  # number of models split over the time domain
    non_dimensionalization = True  # True: network describes solution in dimensionless space

    # model architecture and options
    boundary_condition_transition_function = False  # option for explicitly constraining the boundary conditions
    initial_condition_transition_function = False  # option for explicitly constraining the initial conditions
    split_networks = False  # split model in two networks, one each for u and h
    sirens_initialization = False  # initialization for model with sine activation functions
    save_output_over_training = True  # True: network output is saved on a discrete grid for later use
    save_symbolic_function_over_training = True  # True: PDE Losses are saved on a discrete grid for later use

    # select parts of the objective function - every selected (True) part is added to the total MSE
    train_on_solution = False  # MSE comparing the error compared to the numerical solution
    train_on_PINNs_Loss = True  # MSE term for the initial and boundary conditions and the PDE term
    train_on_boundary_condition_loss = True  # True: add BC loss term to total MSE, False: BC term is not considered
    train_on_initial_condition_loss = True  # True: add IC loss term to total MSE, False: IC term is not considered

    # dimensional parameters
    momentum_advection = True  # include or exclude momentum advection from the momentum equation
    initial_perturbation_amplitude = 1.0  # amplitude of sine wave, cosine or gaussian bell [m]
    average_sea_level = 100.0  # reference sea level from which the elevations are modeled [m]
    gravitational_acceleration = 9.81  # [ms^-1]
    momentum_dissipation = 5e+4  # (for "AH=5e+4")  # [m^2/s]
    nonlinear_drag_coefficient = 0.0  # dimensionless (~2e-3)

    # reference scales
    horizontal_length_scale = 1000000.0  # [m]
    time_scale = 86400.0  # [s]
    vertical_scaling_factor = 1.0  # adjusting between vertical and horizontal length scale
    vertical_length_scale = (
            vertical_scaling_factor * horizontal_length_scale ** 2 / (time_scale ** 2 * gravitational_acceleration)
    )  # length scale for vertical lengths and gravity [m]

    # spatio-temporal domain of numerical model and training
    numerical_solution_time_interval = [0.0, 270000.0]  # [0.0, 129000.0]  # time domain [s]
    numerical_solution_time_step = 60.0  # time step [s]
    numerical_solution_x_interval = [-1000000.0, 1000000.0]  # spatial domain [m]
    numerical_solution_space_step = 10000.0  # spatial step size [m]
    minimum_x = numerical_solution_x_interval[0]  # lower boundary [m]
    maximum_x = numerical_solution_x_interval[1]  # upper boundary [m]

    # scaling by length and time scales
    if non_dimensionalization is True:
        initial_perturbation_amplitude = initial_perturbation_amplitude / vertical_length_scale
        gravitational_acceleration = gravitational_acceleration * time_scale ** 2 / vertical_length_scale
        average_sea_level = average_sea_level / vertical_length_scale
        momentum_dissipation = momentum_dissipation * time_scale / horizontal_length_scale ** 2
        minimum_x = minimum_x / horizontal_length_scale
        maximum_x = maximum_x / horizontal_length_scale

    # contribution of each loss term to the total loss function
    boundary_condition_weight = 1  # loss weight for boundary conditions
    initial_condition_weight = 1  # loss weight for initial conditions
    symbolic_function_weight = 1  # loss weight for symbolic form of PDEs

    # sampling points for loss terms for BC, IC and within the domain for the symbolic function
    boundary_condition_batch_size = 40000  # sampling points on each boundary
    initial_condition_batch_size = 40000  # sampling points for each initial condition
    symbolic_function_batch_size = 100000  # sampling points within the x-t-domain

    # batch sizes for mini batch gradient descent
    bc_mini_batch_size = 200
    ic_mini_batch_size = 200
    pde_mini_batch_size = 500
    epochs = 1000
    device = "cuda"  # "cuda" for GPU or "cpu"
    iterations_per_epoch = int(symbolic_function_batch_size / pde_mini_batch_size)
    batch_resampling_period = 1000 * epochs  # number of training steps after which new collocation points are selected
    output_period = 10  #

    # optimizer options
    optimizer = torch.optim.Adam  # optimizer used for the entire training, if:  projected_gradients = False
    learning_rate = 0.0005  # the default learning rates are 0.001 for Adam and 1 for LBFGS
    line_search = None  # "strong_wolfe"  # valid options are None, "Armijo" or "strong_wolfe", only used for LBFGS
    projected_gradients = False  # When this is True, the optimizer is automatically set to be Adam

    # activation function and selection of the depth and width of the neural network
    number_of_layers = 4  # number of hidden layers
    neurons_per_layer = 100  # number of neurons in each hidden layer
    layer_sizes = [2] + number_of_layers * [neurons_per_layer] + [2]  # list of layer sizes including input and output
    activation_function = torch.tanh  # activation function applied on all hidden layers, but not on the output layer
    model_range = np.arange(number_of_models)

    # initialize new initial conditions for use of several models
    new_initial_conditions = None  # initial conditions from a previous model to be passed to the next model
    new_initial_condition_sampling_points = None  # respective sampling points of new_initial_conditions