import torch
from torch.autograd import grad

from pinn_swe_reflections.models.pinn import PINN


class GradientEnhancedPINN(PINN):
    """Gradient-enhanced PINN.

    Extends the SWE PDE residual loss with first derivatives of each residual
    with respect to the network inputs t and x.
    """

    def __init__(
        self,
        *args,
        gpinn_gradient_weight=1.0,
        gpinn_time_weight=1.0,
        gpinn_space_weight=1.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.gpinn_gradient_weight = gpinn_gradient_weight
        self.gpinn_time_weight = gpinn_time_weight
        self.gpinn_space_weight = gpinn_space_weight
        self._reset_gpinn_history()

    def _reset_gpinn_history(self):
        self.MSE_symbolic_function_without_gpinn_over_training = []
        self.MSE_gPINN_gradient_over_training = []
        self.MSE_gPINN_u_t_over_training = []
        self.MSE_gPINN_h_t_over_training = []
        self.MSE_gPINN_u_x_over_training = []
        self.MSE_gPINN_h_x_over_training = []

    def _residual_input_gradients(self, residual):
        gradients = grad(
            residual.sum(),
            self.symbolic_function_sampling_points,
            create_graph=True,
            retain_graph=True,
            allow_unused=True,
        )[0]

        if gradients is None:
            gradients = torch.zeros_like(self.symbolic_function_sampling_points)

        return gradients[:, 0:1].to(self.device), gradients[:, 1:2].to(self.device)

    def MSE_symbolic_functions(self):
        super().MSE_symbolic_functions()

        self.MSE_symbolic_function_without_gpinn = self.MSE_symbolic_function_value

        f_u_t, f_u_x = self._residual_input_gradients(self.symbolic_function_u_values)
        f_h_t, f_h_x = self._residual_input_gradients(self.symbolic_function_h_values)

        self.MSE_gPINN_u_t = (f_u_t ** 2).mean()
        self.MSE_gPINN_u_x = (f_u_x ** 2).mean()
        self.MSE_gPINN_h_t = (f_h_t ** 2).mean()
        self.MSE_gPINN_h_x = (f_h_x ** 2).mean()

        self.MSE_gPINN_gradient_value = (
            self.gpinn_time_weight * (self.MSE_gPINN_u_t + self.MSE_gPINN_h_t)
            + self.gpinn_space_weight * (self.MSE_gPINN_u_x + self.MSE_gPINN_h_x)
        )

        self.MSE_symbolic_function_value = (
            self.MSE_symbolic_function_without_gpinn
            + self.gpinn_gradient_weight * self.MSE_gPINN_gradient_value
        )

    def Save_MSE(self):
        if not hasattr(self, "MSE_gPINN_gradient_over_training"):
            self._reset_gpinn_history()

        super().Save_MSE()

        self.MSE_symbolic_function_without_gpinn_over_training.append(
            self.MSE_symbolic_function_without_gpinn.cpu().detach().numpy()
        )
        self.MSE_gPINN_gradient_over_training.append(
            self.MSE_gPINN_gradient_value.cpu().detach().numpy()
        )
        self.MSE_gPINN_u_t_over_training.append(self.MSE_gPINN_u_t.cpu().detach().numpy())
        self.MSE_gPINN_h_t_over_training.append(self.MSE_gPINN_h_t.cpu().detach().numpy())
        self.MSE_gPINN_u_x_over_training.append(self.MSE_gPINN_u_x.cpu().detach().numpy())
        self.MSE_gPINN_h_x_over_training.append(self.MSE_gPINN_h_x.cpu().detach().numpy())

    def train_PINN(self):
        self._reset_gpinn_history()
        return super().train_PINN()
