import numpy as np
import torch

from pinn_swe_reflections.models.integral_conservation_pinn import IntegralConservationPINN


class ConstrainedIntegralConservationPINN(IntegralConservationPINN):
    """Integral-conservation PINN with hard IC/BC and discharge control."""

    def __init__(
        self,
        *args,
        global_discharge_weight=1.0,
        initial_transition_time_scale_factor=0.1,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.global_discharge_weight = global_discharge_weight
        self.initial_transition_time_scale_factor = initial_transition_time_scale_factor

    def _reset_conservation_history(self):
        super()._reset_conservation_history()
        self.MSE_global_discharge_over_training = []

    def _raw_network_output(self, t, x):
        tx = torch.cat((t, x), dim=1)

        if self.split_networks is True:
            output_u = self.activation_function(self.layers_u[0](tx))
            output_h = self.activation_function(self.layers_h[0](tx))
            for i in np.arange(1, self.number_of_layers - 1):
                output_u = self.activation_function(self.layers_u[i](output_u))
                output_h = self.activation_function(self.layers_h[i](output_h))
            output_u = self.layers_u[self.number_of_layers - 1](output_u)
            output_h = self.layers_h[self.number_of_layers - 1](output_h)
            return output_u, output_h

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

        return output[:, 0:1], output[:, 1:2]

    def _interpolate_new_initial_condition_values(self, x):
        sampling_points = torch.as_tensor(
            self.new_initial_condition_sampling_points,
            device=self.device,
            dtype=x.dtype,
        )
        values = torch.as_tensor(
            self.new_initial_conditions,
            device=self.device,
            dtype=x.dtype,
        )

        x_samples = sampling_points[:, 1].contiguous()
        order = torch.argsort(x_samples)
        x_sorted = x_samples[order]
        values_sorted = values[order]

        x_flat = x.reshape(-1)
        right = torch.searchsorted(x_sorted, x_flat).clamp(1, x_sorted.numel() - 1)
        left = right - 1

        x_left = x_sorted[left]
        x_right = x_sorted[right]
        denom = (x_right - x_left).clamp_min(self.conservation_eps)
        weight = ((x_flat - x_left) / denom).clamp(0.0, 1.0)

        value_left = values_sorted[left]
        value_right = values_sorted[right]
        interpolated = value_left + weight.reshape(-1, 1) * (value_right - value_left)

        return (
            interpolated[:, 0:1].reshape_as(x),
            interpolated[:, 1:2].reshape_as(x),
        )

    def _initial_condition_values(self, x):
        if self.model_number == 0:
            return (
                self.true_initial_condition_u_function(x),
                self.true_initial_condition_h_function(x),
            )

        return self._interpolate_new_initial_condition_values(x)

    def _initial_blend(self, t):
        time_span = max(self._time_span(), self.conservation_eps)
        tau = max(
            float(self.initial_transition_time_scale_factor) * time_span,
            self.conservation_eps,
        )
        return torch.exp(-(t - float(self.minimum_time)) / tau)

    def _boundary_lift_and_bubble(self, t, x):
        domain_length = max(self._domain_length(), self.conservation_eps)
        xi = (x - float(self.minimum_x)) / domain_length
        lower_u = self.true_lower_boundary_condition_u_function(t)
        upper_u = self.true_upper_boundary_condition_u_function(t)
        lift = (1.0 - xi) * lower_u + xi * upper_u
        bubble = 4.0 * xi * (1.0 - xi)
        return lift, bubble

    def forward(self, t, x):
        raw_u, raw_h = self._raw_network_output(t, x)
        output_u = raw_u
        output_h = raw_h

        if self.initial_condition_transition_function is True:
            initial_u, initial_h = self._initial_condition_values(x)
            alpha = self._initial_blend(t)

            if self.boundary_condition_transition_function is True:
                lift, bubble = self._boundary_lift_and_bubble(t, x)
                initial_t = torch.full_like(t, float(self.minimum_time))
                initial_lift, _ = self._boundary_lift_and_bubble(initial_t, x)
                output_u = (
                    lift
                    + alpha * (initial_u - initial_lift)
                    + (1.0 - alpha) * bubble * raw_u
                )
            else:
                output_u = alpha * initial_u + (1.0 - alpha) * raw_u

            output_h = initial_h + (1.0 - alpha) * raw_h
        elif self.boundary_condition_transition_function is True:
            lift, bubble = self._boundary_lift_and_bubble(t, x)
            output_u = lift + bubble * raw_u

        return torch.cat((output_u, output_h), dim=1)

    def _initial_conservation_targets(self):
        targets = super()._initial_conservation_targets()

        x = targets["x"].reshape(-1, 1)
        weights = targets["weights"]
        initial_u, initial_h = self._initial_condition_values(x)
        initial_u = initial_u.reshape(-1)
        initial_h = initial_h.reshape(-1)

        depth = float(self.average_sea_level) + initial_h
        discharge0 = torch.sum(depth * initial_u * weights)
        average_sea_level = torch.tensor(
            float(self.average_sea_level),
            device=self.device,
            dtype=initial_h.dtype,
        )
        pressure_scale = torch.tensor(
            self._pressure_scale(),
            device=self.device,
            dtype=initial_h.dtype,
        )

        mean_h = torch.sum(initial_h * weights) / self._domain_length()
        depth_scale = (
            torch.abs(average_sea_level)
            + torch.abs(mean_h)
            + targets["amplitude_scale"]
        )
        energy_velocity_scale = torch.sqrt(
            2.0
            * torch.abs(targets["energy0"])
            / (
                torch.abs(average_sea_level) * self._domain_length() + self.conservation_eps
            )
        )
        fallback_velocity_scale = torch.sqrt(
            torch.abs(pressure_scale) / (torch.abs(average_sea_level) + self.conservation_eps)
        ) * targets["amplitude_scale"]
        velocity_scale = torch.maximum(energy_velocity_scale, fallback_velocity_scale)
        discharge_scale = torch.maximum(
            torch.abs(discharge0),
            depth_scale * velocity_scale * self._domain_length(),
        )

        targets["discharge0"] = discharge0.detach()
        targets["discharge_scale"] = discharge_scale.detach() + self.conservation_eps
        return targets

    def MSE_global_discharge_function(self):
        targets = self._get_initial_conservation_targets()
        t = self._sample_times(self.conservation_time_batch_size)
        u, h, _ = self._evaluate_on_tensor_product(t, targets["x"])

        weights = targets["weights"].reshape(1, -1)
        discharge = torch.sum((float(self.average_sea_level) + h) * u * weights, dim=1)
        residual = (discharge - targets["discharge0"]) / targets["discharge_scale"]
        self.MSE_global_discharge_value = (residual ** 2).mean()

    def MSE_symbolic_functions(self):
        super().MSE_symbolic_functions()
        self.MSE_global_discharge_function()
        self.MSE_integral_conservation_value = (
            self.MSE_integral_conservation_value
            + self.global_discharge_weight * self.MSE_global_discharge_value
        )

    def Extra_Loss_Print_Items(self):
        items = super().Extra_Loss_Print_Items()
        return (
            items[:-1]
            + [("Global Discharge Loss", self.MSE_global_discharge_value)]
            + items[-1:]
        )

    def Save_MSE(self):
        super().Save_MSE()
        self.MSE_global_discharge_over_training.append(
            self.MSE_global_discharge_value.cpu().detach().numpy()
        )
