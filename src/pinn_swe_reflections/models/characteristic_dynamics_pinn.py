import torch

from pinn_swe_reflections.models.pinn import PINN


# ---------------------------------------------------------------------------
# Подробная физика L_dynamics_swe
# ---------------------------------------------------------------------------
#
# Базовый PINN-loss уже содержит локальный PDE residual:
#
#     R_u   = u_t + u u_x + pressure_scale eta_x - A_H u_xx = 0
#     R_eta = eta_t + u eta_x + (H + eta) u_x = 0
#
# где:
#     eta(t, x)        - отклонение уровня воды от среднего уровня [m] или
#                        безразмерная версия этой величины;
#     u(t, x)          - горизонтальная скорость [m/s] или безразмерная версия;
#     H                - average_sea_level, средняя глубина;
#     A_H              - momentum_dissipation, горизонтальная вязкость [m^2/s]
#                        или ее безразмерная версия;
#     pressure_scale   - коэффициент перед eta_x:
#                        gravitational_acceleration, если расчет размерный,
#                        vertical_scaling_factor, если расчет безразмерный.
#
# Проблема обычного R_u/R_eta в наших экспериментах: это локальная проверка
# производных в отдельных точках. Сеть может сделать локальный residual малым,
# но при этом получить неверную долгую историю: волновые моды слишком быстро
# сглаживаются, а поздние времена становятся почти плоскими.
#
# Поэтому L_dynamics_swe проверяет не еще одну локальную производную, а конечный
# физический переход состояния во времени:
#
#     "если в момент t сеть предсказала некоторое eta/u, то соответствует ли
#      предсказание сети в момент t + dt тому, что дают SWE за тот же dt?"
#
# Для этого используется линейная динамика мелкой воды около состояния покоя.
# Нелинейные члены u u_x и u eta_x остаются в основном PDE-loss, а здесь мы
# вытаскиваем именно волновую часть, которая отвечает за перенос амплитуды,
# фазу, отражение от закрытых границ и физическое вязкое затухание:
#
#     eta_t + H u_x = 0
#     u_t   + pressure_scale eta_x - A_H u_xx = 0
#
# Закрытые границы задают u = 0 на x_min и x_max. Для такого бассейна длины
#
#     L = x_max - x_min,
#     y = (x - x_min) / L,
#
# естественные стоячие моды:
#
#     eta(t, x) = sum_{n=1..N} eta_n(t) cos(n pi y)
#     u(t, x)   = sum_{n=1..N} u_n(t)   sin(n pi y)
#     k_n       = n pi / L
#
# Почему так:
#   * sin(n pi y) равен нулю на обеих стенках, поэтому u автоматически
#     удовлетворяет закрытым границам;
#   * cos(n pi y) является сопряженной модой уровня воды для того же
#     стоячего волнового движения;
#   * n=0, т.е. постоянная добавка к eta, здесь не используется: это не
#     распространяющаяся волна, а средний уровень/масса. Ее должны держать IC,
#     BC, PDE и при необходимости отдельный mass-loss. Здесь контролируются
#     именно волновые моды.
#
# Подстановка одной моды в линейные SWE дает:
#
#     eta_n' = - H k_n u_n
#     u_n'   =   pressure_scale k_n eta_n - A_H k_n^2 u_n
#
# В матричной форме:
#
#     d/dt [eta_n] = A_n [eta_n],
#          [u_n  ]       [u_n  ]
#
#     A_n = [[0,                         -H k_n],
#            [pressure_scale k_n,  -A_H k_n^2]]
#
# Эта матрица содержит физику изменения eta/u:
#   * -H k_n u_n: скорость перераспределяет массу и меняет eta;
#   * pressure_scale k_n eta_n: наклон уровня создает ускорение u;
#   * -A_H k_n^2 u_n: вязкость гасит скорость, причем мелкие масштабы
#     с большим k_n гаснут быстрее.
#
# Точное решение этой линейной системы за конечный шаг dt:
#
#     [eta_n(t + dt)]                       [eta_n(t)]
#     [u_n(t + dt)  ] = exp(A_n dt)         [u_n(t)]
#
# Величина exp(A_n dt) - это не обучаемый параметр и не численное решение из
# датасета. Она полностью задается физическими коэффициентами SWE и длиной
# бассейна. В недо-затухающем режиме собственные значения A_n имеют
# вещественную часть:
#
#     gamma_n = A_H k_n^2 / 2
#
# поэтому амплитуда моды физически убывает примерно как exp(-gamma_n t), а
# энергия моды убывает примерно как exp(-A_H k_n^2 t). Это и есть физически
# допустимая скорость затухания для данной моды. Если сеть гасит моду быстрее,
# чем это следует из exp(A_n dt), возникает ошибка.
#
# В коде L_dynamics_swe состоит из двух внутренних проверок, но в total loss
# входит одной суммарной компонентой:
#
#   1. Step loss:
#
#        state_net(t + dt)  ~=  exp(A_n dt) state_net(t)
#
#      Это проверяет локально-нелокальный переход между двумя предсказаниями
#      самой сети на большом шаге dt. В отличие от PDE residual, здесь сравнение
#      идет не по производным в одной точке, а по физически ожидаемому изменению
#      волновых мод между двумя временами.
#
#   2. Initial-anchor loss:
#
#        state_net(t)  ~=  exp(A_n (t - t0)) state_initial
#
#      Это нужно, чтобы позднее почти-нулевое состояние не проходило проверку
#      только потому, что "ноль переходит в ноль". Начальное состояние является
#      частью постановки задачи, а не внешним численным решением, поэтому этот
#      якорь говорит: вся последующая волновая история должна быть совместима
#      с начальным гауссовым возмущением и физическим затуханием.
#
# Ошибка считается в энергетически согласованном масштабе:
#
#     delta_E_n = pressure_scale * (delta_eta_n)^2 + H * (delta_u_n)^2
#
# Это не полная нелинейная энергия, а квадратичная линейная энергия модального
# отклонения. Такой масштаб нужен, чтобы eta- и u-части были сопоставимы.
# Затем ошибка нормируется на начальную модальную энергию:
#
#     E0 = mean_n(pressure_scale * eta_n(0)^2 + H * u_n(0)^2) + eps
#
# Итог:
#
#     L_dynamics_swe =
#         step_weight   * mean(delta_E_step / E0)
#       + anchor_weight * mean(delta_E_anchor / E0)
#
# Важно: это по-прежнему не numerical-solution loss. Здесь нет сравнения с
# сохраненным reference solution. Используются только IC из постановки задачи,
# коэффициенты SWE, закрытые границы и текущее предсказание сети.
# ---------------------------------------------------------------------------


class CharacteristicDynamicsPINN(PINN):
    """PINN with a finite-step modal SWE dynamics loss."""

    def __init__(
        self,
        *args,
        swe_dynamics_weight=None,
        characteristic_dynamics_weight=1.0,
        modal_dynamics_dt=None,
        modal_dynamics_dt_seconds=43200.0,
        modal_dynamics_time_batch_size=6,
        modal_dynamics_space_points=201,
        modal_dynamics_modes=32,
        modal_dynamics_include_initial=True,
        modal_dynamics_anchor_initial=True,
        modal_dynamics_step_weight=1.0,
        modal_dynamics_anchor_weight=1.0,
        modal_dynamics_eps=1.0e-12,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if swe_dynamics_weight is not None:
            characteristic_dynamics_weight = swe_dynamics_weight
        self.characteristic_dynamics_weight = characteristic_dynamics_weight
        self.modal_dynamics_dt = self._default_modal_dynamics_dt(
            modal_dynamics_dt,
            modal_dynamics_dt_seconds,
        )
        self.modal_dynamics_time_batch_size = int(modal_dynamics_time_batch_size)
        self.modal_dynamics_space_points = int(modal_dynamics_space_points)
        self.modal_dynamics_modes = int(modal_dynamics_modes)
        self.modal_dynamics_include_initial = modal_dynamics_include_initial
        self.modal_dynamics_anchor_initial = modal_dynamics_anchor_initial
        self.modal_dynamics_step_weight = float(modal_dynamics_step_weight)
        self.modal_dynamics_anchor_weight = float(modal_dynamics_anchor_weight)
        self.modal_dynamics_eps = float(modal_dynamics_eps)

        self._modal_cache = None
        self._modal_energy_scale_cache = None
        self._reset_swe_dynamics_history()

    def _reset_swe_dynamics_history(self):
        self.MSE_swe_dynamics_over_training = []
        self.MSE_swe_dynamics_eta_over_training = []
        self.MSE_swe_dynamics_u_over_training = []
        self.MSE_swe_dynamics_step_over_training = []
        self.MSE_swe_dynamics_anchor_over_training = []

    def _default_modal_dynamics_dt(self, modal_dynamics_dt, modal_dynamics_dt_seconds):
        if modal_dynamics_dt is not None:
            return float(modal_dynamics_dt)

        dt = float(modal_dynamics_dt_seconds)
        if self.non_dimensionalization is True:
            dt = dt / float(self.time_scale)
        return dt

    def _pressure_scale(self):
        if self.non_dimensionalization is True:
            return float(self.vertical_scaling_factor)
        return float(self.gravitational_acceleration)

    def _domain_length(self):
        return float(self.maximum_x - self.minimum_x)

    def _time_span(self):
        return float(self.maximum_time - self.minimum_time)

    def _trapezoid_weights(self, n_points, interval_length, dtype):
        if n_points <= 1:
            return torch.full((1,), float(interval_length), device=self.device, dtype=dtype)

        weights = torch.ones(n_points, device=self.device, dtype=dtype)
        weights[0] = 0.5
        weights[-1] = 0.5
        return weights * (float(interval_length) / float(n_points - 1))

    def _modal_basis(self):
        dtype = next(self.parameters()).dtype
        cache_key = (
            self.modal_dynamics_space_points,
            self.modal_dynamics_modes,
            self.minimum_x,
            self.maximum_x,
            self.device,
            dtype,
        )
        if self._modal_cache is not None and self._modal_cache["key"] == cache_key:
            return self._modal_cache

        x = torch.linspace(
            float(self.minimum_x),
            float(self.maximum_x),
            steps=self.modal_dynamics_space_points,
            device=self.device,
            dtype=dtype,
        )
        domain_length = self._domain_length()
        y = (x - float(self.minimum_x)) / domain_length
        mode_numbers = torch.arange(
            1,
            self.modal_dynamics_modes + 1,
            device=self.device,
            dtype=dtype,
        )
        arguments = mode_numbers[:, None] * torch.pi * y[None, :]
        k = mode_numbers * torch.pi / domain_length
        weights = self._trapezoid_weights(
            self.modal_dynamics_space_points,
            domain_length,
            dtype,
        )

        self._modal_cache = {
            "key": cache_key,
            "x": x,
            "weights": weights,
            "cos_basis": torch.cos(arguments),
            "sin_basis": torch.sin(arguments),
            "k": k,
        }
        return self._modal_cache

    def _sample_modal_times(self):
        dt = min(self.modal_dynamics_dt, 0.5 * self._time_span())
        max_start_time = float(self.maximum_time) - dt
        if max_start_time <= float(self.minimum_time):
            raise ValueError("modal_dynamics_dt must be smaller than the model time interval.")

        n_times = max(self.modal_dynamics_time_batch_size, 1)
        random_count = n_times - 1 if self.modal_dynamics_include_initial else n_times
        dtype = next(self.parameters()).dtype

        random_times = (
            float(self.minimum_time)
            + torch.rand(random_count, device=self.device, dtype=dtype)
            * (max_start_time - float(self.minimum_time))
        )

        if self.modal_dynamics_include_initial:
            initial_time = torch.full(
                (1,),
                float(self.minimum_time),
                device=self.device,
                dtype=dtype,
            )
            return torch.cat((initial_time, random_times), dim=0), dt

        return random_times, dt

    def _sample_anchor_times(self):
        n_times = max(self.modal_dynamics_time_batch_size, 1)
        random_count = max(n_times - 1, 0)
        dtype = next(self.parameters()).dtype

        random_times = (
            float(self.minimum_time)
            + torch.rand(random_count, device=self.device, dtype=dtype)
            * self._time_span()
        )
        final_time = torch.full(
            (1,),
            float(self.maximum_time),
            device=self.device,
            dtype=dtype,
        )
        return torch.cat((final_time, random_times), dim=0)

    def _evaluate_modal_grid(self, t_values):
        basis = self._modal_basis()
        x_values = basis["x"]
        n_times = t_values.numel()
        n_x = x_values.numel()

        t_flat = t_values.reshape(-1, 1).expand(n_times, n_x).reshape(-1, 1)
        x_flat = x_values.reshape(1, -1).expand(n_times, n_x).reshape(-1, 1)
        output = self.forward(t_flat, x_flat)

        u = output[:, 0:1].reshape(n_times, n_x)
        eta = output[:, 1:2].reshape(n_times, n_x)
        return eta, u

    def _modal_coefficients(self, t_values):
        basis = self._modal_basis()
        eta, u = self._evaluate_modal_grid(t_values)
        return self._project_modal_coefficients(eta, u)

    def _project_modal_coefficients(self, eta, u):
        basis = self._modal_basis()
        scale = 2.0 / self._domain_length()
        weights = basis["weights"].reshape(1, -1)

        eta_coefficients = scale * torch.matmul(
            eta * weights,
            basis["cos_basis"].transpose(0, 1),
        )
        u_coefficients = scale * torch.matmul(
            u * weights,
            basis["sin_basis"].transpose(0, 1),
        )
        return eta_coefficients, u_coefficients

    def _initial_modal_coefficients(self):
        basis = self._modal_basis()
        x = basis["x"].reshape(-1, 1)

        if self.model_number == 0:
            eta0 = self.true_initial_condition_h_function(x).reshape(1, -1)
            u0 = self.true_initial_condition_u_function(x).reshape(1, -1)
            return self._project_modal_coefficients(eta0, u0)

        t0 = torch.full(
            (1,),
            float(self.minimum_time),
            device=self.device,
            dtype=next(self.parameters()).dtype,
        )
        eta0, u0 = self._modal_coefficients(t0)
        return eta0.detach(), u0.detach()

    def _modal_transition_matrices(self, dt):
        basis = self._modal_basis()
        dtype = basis["k"].dtype
        k = basis["k"]
        pressure = torch.tensor(self._pressure_scale(), device=self.device, dtype=dtype)
        depth = torch.tensor(float(self.average_sea_level), device=self.device, dtype=dtype)
        diffusion = torch.tensor(float(self.momentum_dissipation), device=self.device, dtype=dtype)

        matrices = torch.zeros(
            self.modal_dynamics_modes,
            2,
            2,
            device=self.device,
            dtype=dtype,
        )
        matrices[:, 0, 1] = -depth * k
        matrices[:, 1, 0] = pressure * k
        matrices[:, 1, 1] = -diffusion * k ** 2

        dt_tensor = torch.as_tensor(dt, device=self.device, dtype=dtype)
        if dt_tensor.ndim == 0:
            return torch.matrix_exp(matrices * dt_tensor)

        return torch.matrix_exp(matrices.unsqueeze(0) * dt_tensor.reshape(-1, 1, 1, 1))

    def _initial_modal_energy_scale(self):
        if self._modal_energy_scale_cache is not None:
            return self._modal_energy_scale_cache

        dtype = next(self.parameters()).dtype
        eta0, u0 = self._initial_modal_coefficients()

        pressure = torch.tensor(self._pressure_scale(), device=self.device, dtype=dtype)
        depth = torch.tensor(float(self.average_sea_level), device=self.device, dtype=dtype)
        modal_energy = pressure * eta0 ** 2 + depth * u0 ** 2
        self._modal_energy_scale_cache = modal_energy.mean().detach() + self.modal_dynamics_eps
        return self._modal_energy_scale_cache

    def _modal_energy_mse(self, delta_eta, delta_u):
        dtype = delta_eta.dtype
        pressure = torch.tensor(self._pressure_scale(), device=self.device, dtype=dtype)
        depth = torch.tensor(float(self.average_sea_level), device=self.device, dtype=dtype)
        energy_scale = self._initial_modal_energy_scale()
        eta_loss = (pressure * delta_eta ** 2).mean() / energy_scale
        u_loss = (depth * delta_u ** 2).mean() / energy_scale
        return eta_loss + u_loss, eta_loss, u_loss

    def _modal_step_dynamics_loss(self):
        t0, dt = self._sample_modal_times()
        t1 = t0 + dt

        eta0, u0 = self._modal_coefficients(t0)
        eta1, u1 = self._modal_coefficients(t1)

        state0 = torch.stack((eta0, u0), dim=2)
        transition = self._modal_transition_matrices(dt)
        predicted_state1 = torch.einsum("mij,bmj->bmi", transition, state0)

        delta_eta = eta1 - predicted_state1[:, :, 0]
        delta_u = u1 - predicted_state1[:, :, 1]
        return self._modal_energy_mse(delta_eta, delta_u)

    def _modal_anchor_dynamics_loss(self):
        t_values = self._sample_anchor_times()
        eta, u = self._modal_coefficients(t_values)

        eta_initial, u_initial = self._initial_modal_coefficients()
        initial_state = torch.stack(
            (eta_initial.reshape(self.modal_dynamics_modes), u_initial.reshape(self.modal_dynamics_modes)),
            dim=1,
        )

        transition = self._modal_transition_matrices(t_values - float(self.minimum_time))
        expected_state = torch.einsum("bmij,mj->bmi", transition, initial_state)

        delta_eta = eta - expected_state[:, :, 0]
        delta_u = u - expected_state[:, :, 1]
        return self._modal_energy_mse(delta_eta, delta_u)

    def MSE_swe_dynamics_function(self):
        (
            self.MSE_swe_dynamics_step,
            self.MSE_swe_dynamics_step_eta,
            self.MSE_swe_dynamics_step_u,
        ) = self._modal_step_dynamics_loss()

        if self.modal_dynamics_anchor_initial:
            (
                self.MSE_swe_dynamics_anchor,
                self.MSE_swe_dynamics_anchor_eta,
                self.MSE_swe_dynamics_anchor_u,
            ) = self._modal_anchor_dynamics_loss()
        else:
            zero = torch.zeros_like(self.MSE_swe_dynamics_step)
            self.MSE_swe_dynamics_anchor = zero
            self.MSE_swe_dynamics_anchor_eta = zero
            self.MSE_swe_dynamics_anchor_u = zero

        self.MSE_swe_dynamics_eta = (
            self.modal_dynamics_step_weight * self.MSE_swe_dynamics_step_eta
            + self.modal_dynamics_anchor_weight * self.MSE_swe_dynamics_anchor_eta
        )
        self.MSE_swe_dynamics_u = (
            self.modal_dynamics_step_weight * self.MSE_swe_dynamics_step_u
            + self.modal_dynamics_anchor_weight * self.MSE_swe_dynamics_anchor_u
        )
        self.MSE_swe_dynamics_value = self.MSE_swe_dynamics_eta + self.MSE_swe_dynamics_u

    def MSE_symbolic_functions(self):
        super().MSE_symbolic_functions()
        self.MSE_swe_dynamics_function()

    def total_MSE_function(self):
        super().total_MSE_function()
        self.total_MSE_value += (
            self.characteristic_dynamics_weight * self.MSE_swe_dynamics_value
        )

    def Extra_Loss_Print_Items(self):
        return [
            ("SWE Dynamics Loss", self.MSE_swe_dynamics_value),
            ("SWE Dynamics Step Loss", self.MSE_swe_dynamics_step),
            ("SWE Dynamics Anchor Loss", self.MSE_swe_dynamics_anchor),
            ("SWE Dynamics Loss eta", self.MSE_swe_dynamics_eta),
            ("SWE Dynamics Loss u", self.MSE_swe_dynamics_u),
        ]

    def Save_MSE(self):
        if not hasattr(self, "MSE_swe_dynamics_over_training"):
            self._reset_swe_dynamics_history()

        super().Save_MSE()

        self.MSE_swe_dynamics_over_training.append(
            self.MSE_swe_dynamics_value.cpu().detach().numpy()
        )
        self.MSE_swe_dynamics_eta_over_training.append(
            self.MSE_swe_dynamics_eta.cpu().detach().numpy()
        )
        self.MSE_swe_dynamics_u_over_training.append(
            self.MSE_swe_dynamics_u.cpu().detach().numpy()
        )
        self.MSE_swe_dynamics_step_over_training.append(
            self.MSE_swe_dynamics_step.cpu().detach().numpy()
        )
        self.MSE_swe_dynamics_anchor_over_training.append(
            self.MSE_swe_dynamics_anchor.cpu().detach().numpy()
        )

    def train_PINN(self):
        self._reset_swe_dynamics_history()
        self._modal_cache = None
        self._modal_energy_scale_cache = None
        return super().train_PINN()
