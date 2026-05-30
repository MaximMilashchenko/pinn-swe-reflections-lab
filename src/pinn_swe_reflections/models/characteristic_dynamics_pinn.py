import torch
from torch.autograd import grad

from pinn_swe_reflections.models.pinn import PINN


class CharacteristicDynamicsPINN(PINN):
    """PINN with a linear SWE characteristic-dynamics regularizer.

    The extra loss reinforces the two linear shallow-water wave components
    R+ = u + sqrt(g / H) eta and R- = u - sqrt(g / H) eta. With viscosity,
    the linearized characteristic residuals are

        R+_t + c R+_x - AH u_xx = 0
        R-_t - c R-_x - AH u_xx = 0

    where c = sqrt(g H). In nondimensional runs, g is represented by the same
    pressure scale used by the base PDE residual.
    """

    def __init__(
        self,
        *args,
        characteristic_dynamics_weight=0.1,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.characteristic_dynamics_weight = characteristic_dynamics_weight
        self._reset_characteristic_history()

    def _reset_characteristic_history(self):
        self.MSE_characteristic_dynamics_over_training = []
        self.MSE_characteristic_plus_over_training = []
        self.MSE_characteristic_minus_over_training = []

    def _pressure_scale(self):
        if self.non_dimensionalization is True:
            return float(self.vertical_scaling_factor)
        return float(self.gravitational_acceleration)

    def _linear_wave_scales(self, dtype):
        pressure = torch.tensor(
            self._pressure_scale(),
            device=self.device,
            dtype=dtype,
        )
        depth = torch.tensor(
            float(self.average_sea_level),
            device=self.device,
            dtype=dtype,
        )
        wave_speed = torch.sqrt(pressure * depth)
        eta_to_velocity = torch.sqrt(pressure / depth)
        return wave_speed, eta_to_velocity

    def Physics_Informed_Symbolic_Function(self):
        t = self.symbolic_function_sampling_points[:, 0:1]
        x = self.symbolic_function_sampling_points[:, 1:2]
        output = self.forward(t, x)
        u = output[:, 0:1]
        h = output[:, 1:2]

        u_sum = u.sum()
        u_t = grad(u_sum, t, create_graph=True)[0].to(self.device)
        u_x = grad(u_sum, x, create_graph=True)[0].to(self.device)
        u_xx = grad(u_x.sum(), x, create_graph=True)[0].to(self.device)
        h_sum = h.sum()
        h_t = grad(h_sum, t, create_graph=True)[0].to(self.device)
        h_x = grad(h_sum, x, create_graph=True)[0].to(self.device)

        if self.non_dimensionalization is True:
            self.symbolic_function_u_values = (
                u_t
                + float(self.momentum_advection) * u * u_x
                + self.vertical_scaling_factor * h_x
                + self.horizontal_length_scale
                * self.nonlinear_drag_coefficient
                * u
                * torch.abs(u)
                - self.momentum_dissipation * u_xx
            )
        else:
            self.symbolic_function_u_values = (
                u_t
                + u * u_x
                + self.gravitational_acceleration * h_x
                + self.nonlinear_drag_coefficient * u * torch.abs(u)
                - self.momentum_dissipation * u_xx
            )

        self.symbolic_function_h_values = h_t + u_x * (h + self.average_sea_level) + u * h_x

        wave_speed, eta_to_velocity = self._linear_wave_scales(u.dtype)
        r_plus_t = u_t + eta_to_velocity * h_t
        r_plus_x = u_x + eta_to_velocity * h_x
        r_minus_t = u_t - eta_to_velocity * h_t
        r_minus_x = u_x - eta_to_velocity * h_x

        self.characteristic_plus_values = (
            r_plus_t + wave_speed * r_plus_x - self.momentum_dissipation * u_xx
        )
        self.characteristic_minus_values = (
            r_minus_t - wave_speed * r_minus_x - self.momentum_dissipation * u_xx
        )

    def MSE_symbolic_functions(self):
        super().MSE_symbolic_functions()
        self.MSE_characteristic_plus = (self.characteristic_plus_values ** 2).mean()
        self.MSE_characteristic_minus = (self.characteristic_minus_values ** 2).mean()
        self.MSE_characteristic_dynamics_value = (
            self.MSE_characteristic_plus + self.MSE_characteristic_minus
        )

    def total_MSE_function(self):
        super().total_MSE_function()
        self.total_MSE_value += (
            self.characteristic_dynamics_weight
            * self.MSE_characteristic_dynamics_value
        )

    def Extra_Loss_Print_Items(self):
        return [
            ("Characteristic Dynamics Loss", self.MSE_characteristic_dynamics_value),
            ("Characteristic Dynamics Loss R+", self.MSE_characteristic_plus),
            ("Characteristic Dynamics Loss R-", self.MSE_characteristic_minus),
        ]

    def Save_MSE(self):
        if not hasattr(self, "MSE_characteristic_dynamics_over_training"):
            self._reset_characteristic_history()

        super().Save_MSE()

        self.MSE_characteristic_dynamics_over_training.append(
            self.MSE_characteristic_dynamics_value.cpu().detach().numpy()
        )
        self.MSE_characteristic_plus_over_training.append(
            self.MSE_characteristic_plus.cpu().detach().numpy()
        )
        self.MSE_characteristic_minus_over_training.append(
            self.MSE_characteristic_minus.cpu().detach().numpy()
        )

    def train_PINN(self):
        self._reset_characteristic_history()
        return super().train_PINN()
