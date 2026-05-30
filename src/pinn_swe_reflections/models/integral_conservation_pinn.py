import torch
from torch.autograd import grad

from pinn_swe_reflections.models.pinn import PINN


class IntegralConservationPINN(PINN):
    """PINN with global integral and local control-volume constraints.

    The added losses are designed to penalize nonphysical wave-amplitude
    collapse that can still satisfy small pointwise SWE residuals.
    """

    def __init__(
        self,
        *args,
        global_mass_weight=1.0,
        energy_balance_weight=1.0,
        control_volume_mass_weight=1.0,
        control_volume_momentum_weight=1.0,
        conservation_time_batch_size=4,
        conservation_space_points=51,
        conservation_dissipation_time_points=3,
        control_volume_batch_size=4,
        control_volume_cells=8,
        control_volume_space_points=4,
        control_volume_time_points=4,
        conservation_time_delta=None,
        control_volume_time_delta=None,
        center_eta_for_energy=True,
        conservation_eps=1.0e-12,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.global_mass_weight = global_mass_weight
        self.energy_balance_weight = energy_balance_weight
        self.control_volume_mass_weight = control_volume_mass_weight
        self.control_volume_momentum_weight = control_volume_momentum_weight

        self.conservation_time_batch_size = conservation_time_batch_size
        self.conservation_space_points = conservation_space_points
        self.conservation_dissipation_time_points = conservation_dissipation_time_points
        self.control_volume_batch_size = control_volume_batch_size
        self.control_volume_cells = control_volume_cells
        self.control_volume_space_points = control_volume_space_points
        self.control_volume_time_points = control_volume_time_points

        self.conservation_time_delta = conservation_time_delta
        self.control_volume_time_delta = control_volume_time_delta
        self.center_eta_for_energy = center_eta_for_energy
        self.conservation_eps = conservation_eps

        self._reset_conservation_history()
        self._conservation_initial_cache = None

    def _reset_conservation_history(self):
        self.MSE_global_mass_over_training = []
        self.MSE_energy_balance_over_training = []
        self.MSE_control_volume_mass_over_training = []
        self.MSE_control_volume_momentum_over_training = []
        self.MSE_integral_conservation_over_training = []

    def _pressure_scale(self):
        if self.non_dimensionalization is True:
            return float(self.vertical_scaling_factor)
        return float(self.gravitational_acceleration)

    def _diffusion_scale(self):
        return float(self.momentum_dissipation)

    def _domain_length(self):
        return float(self.maximum_x - self.minimum_x)

    def _time_span(self):
        return float(self.maximum_time - self.minimum_time)

    def _make_line(self, start, end, n_points):
        return torch.linspace(
            float(start),
            float(end),
            steps=int(n_points),
            device=self.device,
        )

    def _trapezoid_weights(self, n_points, interval_length):
        n_points = int(n_points)
        if n_points <= 1:
            return torch.full(
                (1,),
                float(interval_length),
                device=self.device,
            )

        weights = torch.ones(n_points, device=self.device)
        weights[0] = 0.5
        weights[-1] = 0.5
        return weights * (float(interval_length) / float(n_points - 1))

    def _sample_times(self, n_times):
        return (
            self.minimum_time
            + torch.rand(int(n_times), device=self.device) * self._time_span()
        )

    def _time_delta(self, configured_delta, default_divisor):
        if configured_delta is not None:
            return float(configured_delta)
        return self._time_span() / float(default_divisor)

    def _sample_time_pairs(self, n_pairs, configured_delta, default_divisor):
        dt = self._time_delta(configured_delta, default_divisor)
        max_start_time = float(self.maximum_time) - dt
        if max_start_time <= float(self.minimum_time):
            dt = 0.5 * self._time_span()
            max_start_time = float(self.maximum_time) - dt

        t0 = (
            float(self.minimum_time)
            + torch.rand(int(n_pairs), device=self.device)
            * (max_start_time - float(self.minimum_time))
        )
        t1 = t0 + dt
        return t0, t1

    def _evaluate_on_tensor_product(self, t_values, x_values, require_x_grad=False):
        t_values = t_values.reshape(-1).to(self.device)
        x_values = x_values.reshape(-1).to(self.device)

        t_flat = (
            t_values[:, None]
            .expand(t_values.numel(), x_values.numel())
            .reshape(-1, 1)
        )
        x_flat = (
            x_values[None, :]
            .expand(t_values.numel(), x_values.numel())
            .reshape(-1, 1)
            .clone()
            .detach()
        )

        if require_x_grad:
            x_flat.requires_grad_(True)

        output = self.forward(t_flat, x_flat)
        u = output[:, 0:1]
        h = output[:, 1:2]

        u_x = None
        if require_x_grad:
            u_x = grad(u.sum(), x_flat, create_graph=True)[0].reshape(
                t_values.numel(),
                x_values.numel(),
            )

        return (
            u.reshape(t_values.numel(), x_values.numel()),
            h.reshape(t_values.numel(), x_values.numel()),
            u_x,
        )

    def _evaluate_at_points(self, t, x, require_x_grad=False):
        t_flat = t.reshape(-1, 1).to(self.device)
        x_flat = x.reshape(-1, 1).to(self.device).clone().detach()

        if require_x_grad:
            x_flat.requires_grad_(True)

        output = self.forward(t_flat, x_flat)
        u = output[:, 0:1]
        h = output[:, 1:2]

        u_x = None
        if require_x_grad:
            u_x = grad(u.sum(), x_flat, create_graph=True)[0]

        shape = t.shape
        return (
            u.reshape(shape),
            h.reshape(shape),
            None if u_x is None else u_x.reshape(shape),
        )

    def _initial_conservation_targets(self):
        x = self._make_line(
            self.minimum_x,
            self.maximum_x,
            self.conservation_space_points,
        )
        weights = self._trapezoid_weights(self.conservation_space_points, self._domain_length())

        if self.model_number == 0:
            h0 = self.true_initial_condition_h_function(x.reshape(-1, 1)).reshape(-1)
            u0 = self.true_initial_condition_u_function(x.reshape(-1, 1)).reshape(-1)
        else:
            t0 = torch.full((1,), float(self.minimum_time), device=self.device)
            u0_grid, h0_grid, _ = self._evaluate_on_tensor_product(t0, x)
            h0 = h0_grid.detach().reshape(-1)
            u0 = u0_grid.detach().reshape(-1)

        mass0 = torch.sum(h0 * weights)

        h_for_energy = h0
        if self.center_eta_for_energy:
            h_for_energy = h0 - torch.sum(h0 * weights) / self._domain_length()

        energy0 = 0.5 * float(self.average_sea_level) * torch.sum((u0 ** 2) * weights)
        energy0 = energy0 + 0.5 * self._pressure_scale() * torch.sum(
            (h_for_energy ** 2) * weights
        )

        amplitude_scale = max(float(abs(self.initial_perturbation_amplitude)), self.conservation_eps)
        mass_scale = torch.maximum(
            torch.abs(mass0),
            torch.tensor(
                amplitude_scale * self._domain_length(),
                device=self.device,
            ),
        )
        energy_scale = torch.maximum(
            torch.abs(energy0),
            torch.tensor(
                0.5
                * self._pressure_scale()
                * amplitude_scale ** 2
                * self._domain_length(),
                device=self.device,
            ),
        )

        return {
            "x": x,
            "weights": weights,
            "mass0": mass0.detach(),
            "energy0": energy0.detach(),
            "mass_scale": mass_scale.detach() + self.conservation_eps,
            "energy_scale": energy_scale.detach() + self.conservation_eps,
            "amplitude_scale": torch.tensor(
                amplitude_scale,
                device=self.device,
            ),
        }

    def _get_initial_conservation_targets(self):
        if self._conservation_initial_cache is None:
            self._conservation_initial_cache = self._initial_conservation_targets()
        return self._conservation_initial_cache

    def MSE_global_mass_function(self):
        targets = self._get_initial_conservation_targets()
        t = self._sample_times(self.conservation_time_batch_size)
        _, h, _ = self._evaluate_on_tensor_product(t, targets["x"])

        mass = torch.sum(h * targets["weights"].reshape(1, -1), dim=1)
        residual = (mass - targets["mass0"]) / targets["mass_scale"]
        self.MSE_global_mass_value = (residual ** 2).mean()

    def _energy_at_times(self, t):
        targets = self._get_initial_conservation_targets()
        u, h, _ = self._evaluate_on_tensor_product(t, targets["x"])
        weights = targets["weights"].reshape(1, -1)

        h_for_energy = h
        if self.center_eta_for_energy:
            mean_h = torch.sum(h * weights, dim=1, keepdim=True) / self._domain_length()
            h_for_energy = h - mean_h

        kinetic = 0.5 * float(self.average_sea_level) * torch.sum((u ** 2) * weights, dim=1)
        potential = 0.5 * self._pressure_scale() * torch.sum((h_for_energy ** 2) * weights, dim=1)
        return kinetic + potential

    def _dissipation_between_times(self, t0, t1):
        diffusion = self._diffusion_scale()
        if diffusion == 0.0:
            return torch.zeros_like(t0)

        targets = self._get_initial_conservation_targets()
        tau = self._make_line(0.0, 1.0, self.conservation_dissipation_time_points)
        tau_weights = self._trapezoid_weights(self.conservation_dissipation_time_points, 1.0)

        t = t0[:, None] + (t1 - t0)[:, None] * tau[None, :]
        t_flat = t.reshape(-1)
        _, _, u_x = self._evaluate_on_tensor_product(
            t_flat,
            targets["x"],
            require_x_grad=True,
        )

        x_weights = targets["weights"].reshape(1, -1)
        spatial_integral = torch.sum((u_x ** 2) * x_weights, dim=1).reshape(
            t0.numel(),
            tau.numel(),
        )
        time_integral = torch.sum(
            spatial_integral * tau_weights.reshape(1, -1),
            dim=1,
        ) * (t1 - t0)

        return float(self.average_sea_level) * diffusion * time_integral

    def MSE_energy_balance_function(self):
        targets = self._get_initial_conservation_targets()
        t0, t1 = self._sample_time_pairs(
            self.conservation_time_batch_size,
            self.conservation_time_delta,
            default_divisor=32,
        )

        energy0 = self._energy_at_times(t0)
        energy1 = self._energy_at_times(t1)
        dissipation = self._dissipation_between_times(t0, t1)

        residual = (energy1 - energy0 + dissipation) / targets["energy_scale"]
        self.MSE_energy_balance_value = (residual ** 2).mean()

    def _control_volume_geometry(self):
        edges = self._make_line(
            self.minimum_x,
            self.maximum_x,
            int(self.control_volume_cells) + 1,
        )
        x_left = edges[:-1]
        x_right = edges[1:]
        dx = x_right - x_left
        xi = self._make_line(0.0, 1.0, self.control_volume_space_points)
        x_inside = x_left[:, None] + dx[:, None] * xi[None, :]
        x_weights = self._trapezoid_weights(self.control_volume_space_points, 1.0)
        return x_left, x_right, dx, x_inside, x_weights

    def _integrate_cell_values(self, values, dx, unit_weights):
        weights = dx.reshape(1, -1, 1) * unit_weights.reshape(1, 1, -1)
        return torch.sum(values * weights, dim=2)

    def _control_volume_field_integrals(self, t, x_inside, dx, unit_weights):
        n_times = t.numel()
        n_cells = x_inside.shape[0]
        n_x = x_inside.shape[1]

        t_eval = t.reshape(-1, 1, 1).expand(n_times, n_cells, n_x)
        x_eval = x_inside.reshape(1, n_cells, n_x).expand(n_times, n_cells, n_x)

        u, h, _ = self._evaluate_at_points(t_eval, x_eval)
        u_integral = self._integrate_cell_values(u, dx, unit_weights)
        h_integral = self._integrate_cell_values(h, dx, unit_weights)
        return u_integral, h_integral

    def _control_volume_boundary_fluxes(self, t0, t1, x_left, x_right):
        n_pairs = t0.numel()
        n_cells = x_left.numel()
        tau = self._make_line(0.0, 1.0, self.control_volume_time_points)
        tau_weights = self._trapezoid_weights(self.control_volume_time_points, 1.0)
        t = t0[:, None] + (t1 - t0)[:, None] * tau[None, :]

        t_eval = t[:, :, None].expand(n_pairs, tau.numel(), n_cells)
        x_left_eval = x_left.reshape(1, 1, n_cells).expand(n_pairs, tau.numel(), n_cells)
        x_right_eval = x_right.reshape(1, 1, n_cells).expand(n_pairs, tau.numel(), n_cells)

        u_l, h_l, u_x_l = self._evaluate_at_points(
            t_eval,
            x_left_eval,
            require_x_grad=True,
        )
        u_r, h_r, u_x_r = self._evaluate_at_points(
            t_eval,
            x_right_eval,
            require_x_grad=True,
        )

        mass_flux_l = (float(self.average_sea_level) + h_l) * u_l
        mass_flux_r = (float(self.average_sea_level) + h_r) * u_r

        advection_l = 0.5 * u_l ** 2 if self.momentum_advection else 0.0
        advection_r = 0.5 * u_r ** 2 if self.momentum_advection else 0.0
        momentum_flux_l = (
            advection_l
            + self._pressure_scale() * h_l
            - self._diffusion_scale() * u_x_l
        )
        momentum_flux_r = (
            advection_r
            + self._pressure_scale() * h_r
            - self._diffusion_scale() * u_x_r
        )

        weights = tau_weights.reshape(1, -1, 1)
        dt = (t1 - t0).reshape(-1, 1)
        mass_flux_integral = torch.sum(
            (mass_flux_r - mass_flux_l) * weights,
            dim=1,
        ) * dt
        momentum_flux_integral = torch.sum(
            (momentum_flux_r - momentum_flux_l) * weights,
            dim=1,
        ) * dt

        return mass_flux_integral, momentum_flux_integral

    def MSE_control_volume_functions(self):
        targets = self._get_initial_conservation_targets()
        t0, t1 = self._sample_time_pairs(
            self.control_volume_batch_size,
            self.control_volume_time_delta,
            default_divisor=32,
        )
        x_left, x_right, dx, x_inside, x_weights = self._control_volume_geometry()

        u0_integral, h0_integral = self._control_volume_field_integrals(
            t0,
            x_inside,
            dx,
            x_weights,
        )
        u1_integral, h1_integral = self._control_volume_field_integrals(
            t1,
            x_inside,
            dx,
            x_weights,
        )
        mass_flux_integral, momentum_flux_integral = self._control_volume_boundary_fluxes(
            t0,
            t1,
            x_left,
            x_right,
        )

        mass_residual = h1_integral - h0_integral + mass_flux_integral
        momentum_residual = u1_integral - u0_integral + momentum_flux_integral

        cell_scale = targets["amplitude_scale"] * torch.abs(dx).reshape(1, -1)
        cell_scale = cell_scale + self.conservation_eps
        self.MSE_control_volume_mass_value = ((mass_residual / cell_scale) ** 2).mean()
        self.MSE_control_volume_momentum_value = (
            (momentum_residual / cell_scale) ** 2
        ).mean()

    def MSE_symbolic_functions(self):
        super().MSE_symbolic_functions()

        self.MSE_global_mass_function()
        self.MSE_energy_balance_function()
        self.MSE_control_volume_functions()

        self.MSE_integral_conservation_value = (
            self.global_mass_weight * self.MSE_global_mass_value
            + self.energy_balance_weight * self.MSE_energy_balance_value
            + self.control_volume_mass_weight * self.MSE_control_volume_mass_value
            + self.control_volume_momentum_weight
            * self.MSE_control_volume_momentum_value
        )

    def total_MSE_function(self):
        super().total_MSE_function()
        self.total_MSE_value += self.MSE_integral_conservation_value

    def Extra_Loss_Print_Items(self):
        return [
            ("Global Mass Loss", self.MSE_global_mass_value),
            ("Energy Balance Loss", self.MSE_energy_balance_value),
            ("Control Volume Mass Loss", self.MSE_control_volume_mass_value),
            ("Control Volume Momentum Loss", self.MSE_control_volume_momentum_value),
            ("Weighted Integral Conservation Loss", self.MSE_integral_conservation_value),
        ]

    def Save_MSE(self):
        if not hasattr(self, "MSE_integral_conservation_over_training"):
            self._reset_conservation_history()

        super().Save_MSE()

        self.MSE_global_mass_over_training.append(
            self.MSE_global_mass_value.cpu().detach().numpy()
        )
        self.MSE_energy_balance_over_training.append(
            self.MSE_energy_balance_value.cpu().detach().numpy()
        )
        self.MSE_control_volume_mass_over_training.append(
            self.MSE_control_volume_mass_value.cpu().detach().numpy()
        )
        self.MSE_control_volume_momentum_over_training.append(
            self.MSE_control_volume_momentum_value.cpu().detach().numpy()
        )
        self.MSE_integral_conservation_over_training.append(
            self.MSE_integral_conservation_value.cpu().detach().numpy()
        )

    def train_PINN(self):
        self._reset_conservation_history()
        self._conservation_initial_cache = None
        return super().train_PINN()
