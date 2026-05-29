import torch
from torch.autograd import grad

from pinn_swe_reflections.models.pinn import PINN


class EulerTransitionPINN(PINN):
    """PINN with explicit Euler state-transition consistency loss.

    The loss enforces that the network state at t + dt is consistent with one
    explicit Euler step from the network state at t using the same SWE terms as
    the existing PDE residual.
    """

    def __init__(
        self,
        *args,
        euler_transition_weight=1.0,
        euler_transition_dt=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.euler_transition_weight = euler_transition_weight
        self.euler_transition_dt = self._default_euler_transition_dt(
            euler_transition_dt
        )
        self._reset_euler_transition_history()

    def _default_euler_transition_dt(self, euler_transition_dt):
        if euler_transition_dt is not None:
            return float(euler_transition_dt)

        dt = float(self.numerical_solution_time_step)
        if self.non_dimensionalization is True:
            dt = dt / float(self.time_scale)
        return dt

    def _reset_euler_transition_history(self):
        self.MSE_euler_transition_over_training = []
        self.MSE_euler_transition_u_over_training = []
        self.MSE_euler_transition_h_over_training = []

    def _transition_sampling_points_from_current_pde_points(self):
        tx = self.symbolic_function_sampling_points.detach()
        t = tx[:, 0:1]
        x = tx[:, 1:2]

        max_start_time = self.maximum_time - self.euler_transition_dt
        if max_start_time <= self.minimum_time:
            raise ValueError(
                "euler_transition_dt must be smaller than the model time interval."
            )

        source_span = self.maximum_time - self.minimum_time
        transition_span = max_start_time - self.minimum_time
        t = self.minimum_time + (t - self.minimum_time) * (
            transition_span / source_span
        )

        return torch.cat((t, x), dim=1).float().to(self.device).requires_grad_(True)

    def _euler_rhs(self, t, x):
        output = self.forward(t, x)
        u = output[:, 0:1]
        h = output[:, 1:2]

        u_x = grad(u.sum(), x, create_graph=True)[0].to(self.device)
        u_xx = grad(u_x.sum(), x, create_graph=True)[0].to(self.device)
        h_x = grad(h.sum(), x, create_graph=True)[0].to(self.device)

        if self.non_dimensionalization is True:
            momentum_terms = (
                float(self.momentum_advection) * u * u_x
                + self.vertical_scaling_factor * h_x
                + self.horizontal_length_scale
                * self.nonlinear_drag_coefficient
                * u
                * torch.abs(u)
                - self.momentum_dissipation * u_xx
            )
        else:
            momentum_terms = (
                u * u_x
                + self.gravitational_acceleration * h_x
                + self.nonlinear_drag_coefficient * u * torch.abs(u)
                - self.momentum_dissipation * u_xx
            )

        continuity_terms = u_x * (h + self.average_sea_level) + u * h_x

        return u, h, -momentum_terms, -continuity_terms

    def MSE_euler_transition_function(self):
        transition_points = self._transition_sampling_points_from_current_pde_points()
        t = transition_points[:, 0:1]
        x = transition_points[:, 1:2]

        u, h, u_rhs, h_rhs = self._euler_rhs(t, x)
        next_output = self.forward(t + self.euler_transition_dt, x)
        next_u = next_output[:, 0:1]
        next_h = next_output[:, 1:2]

        euler_u = u + self.euler_transition_dt * u_rhs
        euler_h = h + self.euler_transition_dt * h_rhs

        self.MSE_euler_transition_u = ((next_u - euler_u) ** 2).mean()
        self.MSE_euler_transition_h = ((next_h - euler_h) ** 2).mean()
        self.MSE_euler_transition_value = (
            self.MSE_euler_transition_u + self.MSE_euler_transition_h
        )

    def MSE_symbolic_functions(self):
        super().MSE_symbolic_functions()
        self.MSE_euler_transition_function()

    def total_MSE_function(self):
        super().total_MSE_function()
        self.total_MSE_value += (
            self.euler_transition_weight * self.MSE_euler_transition_value
        )

    def Save_MSE(self):
        if not hasattr(self, "MSE_euler_transition_over_training"):
            self._reset_euler_transition_history()

        super().Save_MSE()

        self.MSE_euler_transition_over_training.append(
            self.MSE_euler_transition_value.cpu().detach().numpy()
        )
        self.MSE_euler_transition_u_over_training.append(
            self.MSE_euler_transition_u.cpu().detach().numpy()
        )
        self.MSE_euler_transition_h_over_training.append(
            self.MSE_euler_transition_h.cpu().detach().numpy()
        )

    def train_PINN(self):
        self._reset_euler_transition_history()
        return super().train_PINN()
