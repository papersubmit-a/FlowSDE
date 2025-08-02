from numba import cuda
if cuda.is_available():
    import jax
else:
    import os
    os.environ["JAX_PLATFORMS"] = 'cpu'
    import jax
    jax.default_device('cpu')
import jax.numpy as jnp
import jax.random as jr
import jax.nn as jnn
from typing import Tuple, Any
import equinox as eqx
import diffrax
from interpax import interp1d
from neural_network_jax import GRU
from NF_NeuralODE_models import ResFunc


class LatentSDE(eqx.Module):
    """
    Latent SDE model with neural network parameterized drift and diffusion.

    The SDE has the form:
    dX_t = f(X_t, t, ctx) dt + g(X_t, t) dW_t

    where f and g are neural networks, and ctx is context from encoder.
    """
    f_net: eqx.Module  # Drift network
    g_nets: eqx.Module  # Individual diffusion networks for diagonal noise
    h_net: eqx.Module  # Reference drift for log-ratio computation
    latent_dim: int
    ft: bool
    context_dim: int
    qz0_mu: jax.Array
    qz0_logvar: jax.Array
    pz0_mu: float
    pz0_logvar: float
    model_name: str


    def __init__(self, latent_dim: int, hidden_dim_drift: int, depth_drift: int, hidden_dim_diff: int, depth_diff: int,
                 *, key: jax.Array, ft: bool, u_size:int, context_dim: int, model_name='NeuralODE'):
        self.latent_dim = latent_dim
        self.ft = ft
        self.context_dim = context_dim
        self.model_name = model_name

        f_key, h_key, g_key = jr.split(key, 3)

        inp_dim_f = latent_dim + u_size + context_dim
        if ft:
            inp_dim_f = inp_dim_f + 1

        # Drift network f(x, ctx, t) -> dx/dt (posterior)
        if "ResNet" in model_name:
            self.f_net = ResFunc(
                in_size=inp_dim_f, out_size=latent_dim, width_size=hidden_dim_drift, depth=depth_drift,
                activation=jnn.softplus, key=h_key)
        else:
            self.f_net = eqx.nn.MLP(
                in_size=inp_dim_f, out_size=latent_dim, width_size=hidden_dim_drift, depth=depth_drift,
                activation=jnn.softplus, key=h_key)
        # Reference drift h(x, t) -> dx/dt (prior)
        inp_dim_h = latent_dim + u_size
        if ft:
            inp_dim_h = inp_dim_h + 1

        if "ResNet" in model_name:
            self.h_net = ResFunc(
                in_size=inp_dim_h, out_size=latent_dim, width_size=hidden_dim_drift, depth=depth_drift,
                activation=jnn.softplus, key=h_key)
        else:
            self.h_net = eqx.nn.MLP(
                in_size=inp_dim_h, out_size=latent_dim, width_size=hidden_dim_drift, depth=depth_drift,
                activation=jnn.softplus, key=h_key)

        def _build_g_net(key, in_features, out_features):
            keys = jr.split(key, depth_diff)
            _g_net_list = [eqx.nn.Linear(in_features, hidden_dim_diff, key=keys[0])]
            if depth_drift > 1:
                for i in range(1, depth_diff - 1):
                    _g_net_list.append(eqx.nn.Lambda(jax.nn.softplus))
                    _g_net_list.append(eqx.nn.Linear(hidden_dim_diff, hidden_dim_diff, key=keys[i]))
                _g_net_list.append(eqx.nn.Lambda(jax.nn.softplus))
                _g_net_list.append(eqx.nn.Linear(hidden_dim_diff, out_features, key=keys[-1]))
                if 'scaled' in model_name:
                    _g_net_list.append(eqx.nn.Lambda(jax.nn.tanh))
            g_net = eqx.nn.Sequential(_g_net_list)
            return g_net

        if 'single' in model_name: # single noise on single dimension or one noise for all dimensions
            self.g_nets = _build_g_net(g_key, 2, 1)
        elif 'one' in model_name:
            self.g_nets = _build_g_net(g_key, self.latent_dim+1, 1)
        else: # Individual diffusion networks for diagonal noise g_i(x_i, t) -> g_i
            g_nets = []
            for i in range(latent_dim):
                g_net = _build_g_net(g_key, 2, 1)
                g_nets.append(g_net)
            self.g_nets = eqx.nn.Sequential(g_nets)

        self.qz0_mu = jnp.zeros((self.latent_dim,))
        self.qz0_logvar = jnp.zeros((self.latent_dim,))
        self.pz0_mu = 0
        self.pz0_logvar = 0

    def drift_with_context(self, t: float, y: jax.Array, args) -> jax.Array:
        """Posterior drift function f(x, ctx, t)"""
        ts, us, ctx = args
        if us is not None:
            u_fun = lambda t: interp1d(t, ts, us, method="nearest", extrap=True)
            if self.ft:
                input_vec = jnp.concatenate([y, u_fun(t), jnp.sin(t)[None], ctx])
            else:
                input_vec = jnp.concatenate([y, u_fun(t), ctx])
        else:
            if self.ft:
                input_vec = jnp.concatenate([y, jnp.sin(t)[None], ctx])
            else:
                input_vec = jnp.concatenate([y, ctx])
        return self.f_net(input_vec)

    def posterior_drift(self, t: float, y: jax.Array, args) -> jax.Array:
        """Posterior drift function f(x, t)"""
        (ts, us), (tx, ctx) = args
        if us is not None:
            u_fun = lambda t: interp1d(t, ts, us, method="nearest", extrap=True)
            if self.ft:
                input_vec = jnp.concatenate([y, u_fun(t), jnp.sin(t)[None]])
            else:
                input_vec = jnp.concatenate([y, u_fun(t)])
        else:
            if self.ft:
                input_vec = jnp.concatenate([y, jnp.sin(t)[None]])
            else:
                input_vec = y
        return self.f_net(input_vec)

    def prior_drift(self, t: float, y: jax.Array, args) -> jax.Array:
        """Prior drift function h(x, t)"""
        (ts, us), (tx, ctx) = args
        if us is not None:
            u_fun = lambda t: interp1d(t, ts, us, method="nearest", extrap=True)
            if self.ft:
                input_vec = jnp.concatenate([y, u_fun(t), jnp.sin(t)[None]])
            else:
                input_vec = jnp.concatenate([y, u_fun(t)])
        else:
            if self.ft:
                input_vec = jnp.concatenate([y, jnp.sin(t)[None]])
            else:
                input_vec = y
        return self.h_net(input_vec)

    def g(self, t: float, y: jax.Array, args) -> jax.Array:
        """Diagonal diffusion function g(x, t)"""
        # Apply each g_net to corresponding component of y
        y_components = jnp.split(y, self.latent_dim)
        g_values = []
        if 'single' in self.model_name:
            selected_i = 1
            for i, y_i in enumerate(y_components):
                if i == selected_i:
                    g_i = self.g_nets(jnp.concatenate([y_i, jnp.array([t])]))
                else:
                    g_i = jnp.zeros_like(y_i)
                g_values.append(g_i)
        elif 'one' in self.model_name:
            for i  in range(self.latent_dim):
                g_i = self.g_nets(jnp.concatenate([y, jnp.array([t])]))
                g_values.append(g_i)
        else:
            for i, (g_net, y_i) in enumerate(zip(self.g_nets, y_components)):
                g_i = g_net(jnp.concatenate([y_i, jnp.array([t])]))
                g_values.append(g_i)
        return jnp.concatenate(g_values)

    def diffusion_fn(self, t: float, y: jax.Array, args: Any = None) -> jax.Array:
        """Diffusion function (same for both posterior and prior)"""
        return jnp.diag(self.g(t, y, args))

    # Augmented state: [z, log_ratio]
    def augmented_drift(self, t, augmented_y, args):
        (ts, us), (txs, ctxs) = args
        z, log_ratio = augmented_y[:-1], augmented_y[-1:]

        # Posterior and prior drifts
        if self.context_dim > 0:
            if ctxs.ndim == 1:
                f_post = self.drift_with_context(t, z, (ts, us, ctxs))
            else:
                ctx_t_fun = lambda t: interp1d(t, txs, ctxs, method='nearest')
                f_post = self.drift_with_context(t, z, (ts, us, ctx_t_fun(t)))
        else:
            f_post = self.posterior_drift(t, z, args)
        f_prior = self.prior_drift(t, z, args)
        g = self.diffusion_fn(t, z, args)

        g_inv = jnp.linalg.pinv(g)  # Pseudo-inverse for stability

        # Log-ratio derivative: 0.5 * ||(f_post - f_prior) @ g^-1||^2
        drift_diff = f_post - f_prior
        log_ratio_deriv = 0.5 * jnp.nansum((drift_diff @ g_inv) * drift_diff)

        return jnp.concatenate([f_post, log_ratio_deriv[None]])

    def augmented_diffusion(self, t, augmented_y, args):
        z = augmented_y[:-1]
        diffusion_z = self.diffusion_fn(t, z, args)
        # Log-ratio doesn't have diffusion term
        zeros_log_ratio = jnp.zeros((1, diffusion_z.shape[1]))
        return jnp.vstack([diffusion_z, zeros_log_ratio])

    def solve_sde(self, x0, ts, us, key, max_steps=4096, adaptive=True, dt0=0.01):
        # Initial augmented state
        augmented_y0 = jnp.concatenate([x0, jnp.array([0.0])])  # log_ratio starts at 0

        # Create diffrax terms
        drift = diffrax.ODETerm(self.augmented_drift)
        control = diffrax.VirtualBrownianTree(ts[0], ts[-1] + 1e-3, tol=1e-3, shape=(self.latent_dim,), key=key)
        cvf = diffrax.ControlTerm(self.augmented_diffusion, control)
        terms = diffrax.MultiTerm(drift, cvf)

        saveat = diffrax.SaveAt(ts=ts)
        if adaptive:
            solver = diffrax.StratonovichMilstein()
            solution = diffrax.diffeqsolve(terms=terms, solver=solver, t0=ts[0], t1=ts[-1], dt0=dt0, y0=augmented_y0,
                                           args=((ts, us), (None, None)), saveat=saveat, max_steps=max_steps,
                                           stepsize_controller=diffrax.PIDController(rtol=1e-3, atol=1e-6))
        else:
            solver = diffrax.StratonovichMilstein()
            solution = diffrax.diffeqsolve(terms=terms, solver=solver, t0=ts[0], t1=ts[-1], dt0=dt0, y0=augmented_y0,
                                           args=((ts, us), (None, None)), saveat=saveat, max_steps=max_steps)

        # Split solution
        zs = solution.ys[..., :-1]  # Latent trajectories
        log_ratios = solution.ys[..., -1]  # Log-ratio trajectory

        return zs, log_ratios

    def sample_sde(self, x0, ts, us, key, max_steps=4096, adaptive=True, dt0=0.01):
        drift = diffrax.ODETerm(self.posterior_drift)
        control = diffrax.VirtualBrownianTree(ts[0], ts[-1] + 1e-3, tol=1e-3, shape=(self.latent_dim,), key=key)
        cvf = diffrax.ControlTerm(self.diffusion_fn, control)
        terms = diffrax.MultiTerm(drift, cvf)

        saveat = diffrax.SaveAt(ts=ts)
        if adaptive:
            solver = diffrax.StratonovichMilstein()
            solution = diffrax.diffeqsolve(terms=terms, solver=solver, t0=ts[0], t1=ts[-1], dt0=dt0, y0=x0,
                                           args=((ts, us), (None, None)), saveat=saveat, max_steps=max_steps,
                                           stepsize_controller=diffrax.PIDController(rtol=1e-3, atol=1e-6))
        else:
            solver = diffrax.StratonovichMilstein()
            solution = diffrax.diffeqsolve(terms=terms, solver=solver, t0=ts[0], t1=ts[-1], dt0=dt0, y0=x0,
                                           args=((ts, us), (None, None)), saveat=saveat, max_steps=max_steps)
        return solution.ys

    def generator(self, xs, ts, us, key, max_steps=4096, adaptive=True, sigma=0.1, dt0=0.01, beta=1e-3):
        logpq0 = 0.5 * jnp.sum(
            (jnp.exp(self.qz0_logvar) + (self.qz0_mu - self.pz0_mu) ** 2) / jnp.exp(self.pz0_logvar)
            - 1 + self.pz0_logvar - self.qz0_logvar, axis=-1)
        xs_pred, log_ratios = self.solve_sde(xs[0], ts, us, key, max_steps=max_steps, adaptive=adaptive, dt0=dt0)
        log_pxs = - 0.5 * jnp.nanmean((xs - xs_pred) ** 2) / (sigma ** 2)
        log_pxs -= xs.size * jnp.log(sigma * jnp.sqrt(2 * jnp.pi))
        logqp_path = jnp.nanmean(log_ratios, axis=0)  # Final log-ratio value
        logqp = logpq0 + logqp_path
        loss = - jnp.mean(log_pxs - beta * logqp)
        return xs_pred, loss

    def train(self, xs, ts, us, key, max_steps=4096, adaptive=True, sigma=0.1, dt0=0.01, beta=1e-3, ms=False):
        if ms:
            if us is None:
                xs_pred, loss = jax.vmap(lambda x, t: self.generator(
                    x, t, us, key, max_steps=max_steps, adaptive=adaptive, sigma=sigma, dt0=dt0, beta=beta))(xs, ts)
            else:
                xs_pred, loss = jax.vmap(lambda x, t, u: self.generator(
                    x, t, u, key, max_steps=max_steps, adaptive=adaptive, sigma=sigma, dt0=dt0, beta=beta))(xs, ts, us)
            loss = jnp.nanmean(loss) + jnp.nanmean((xs_pred[:-1, -1] - xs[1:, 0]) ** 2)
        else:
            xs_pred, loss = self.generator(xs, ts, us, key, max_steps=max_steps, adaptive=adaptive, dt0=dt0, beta=beta)
            loss = jnp.nanmean(loss)
        return loss

    def sample(self, x0, ts, us, keys, max_steps=4096, adaptive=False, dt0=0.01):
        if x0.shape[0] > 1:
            if us is None:
                func = lambda k: jax.vmap(lambda _x0, _ts: self.sample_sde(
                    _x0, _ts, us, k, max_steps=max_steps, adaptive=adaptive, dt0=dt0))(x0, ts)
            else:
                func = lambda k: jax.vmap(lambda _x0, _ts, _us: self.sample_sde(
                    _x0, _ts, _us, k, max_steps=max_steps, adaptive=adaptive, dt0=dt0))(x0, ts, us)
            xs_samples = jax.vmap(func)(keys)
        else:
            func = lambda k: self.sample_sde(x0[0], ts[0], us[0], k, max_steps=max_steps, adaptive=adaptive, dt0=dt0)
            xs_samples = jax.vmap(func)(keys)[:, jnp.newaxis]
        return xs_samples

    def solve_ode(self, x0, ts, us, max_steps=4096):
        solution = diffrax.diffeqsolve(terms=diffrax.ODETerm(self.prior_drift), solver=diffrax.Tsit5(),
                                       t0=ts[0], t1=ts[-1], dt0=ts[1]-ts[0], y0=x0, args=((ts, us), (None, None)),
                                       stepsize_controller=diffrax.PIDController(rtol=1e-3, atol=1e-6),
                                       saveat=diffrax.SaveAt(ts=ts), max_steps=max_steps)
        return solution.ys

    def pre_train(self, xs, ts, us, key, max_steps=4096, ms=False):

        if ms:
            if us is None:
                xs_pred = jax.vmap(lambda _x0, _ts: self.solve_ode(_x0, _ts, us, max_steps=max_steps))(xs[:, 0], ts)
            else:
                xs_pred = jax.vmap(lambda _x0, _ts, _us: self.solve_ode(_x0, _ts, _us, max_steps=max_steps))(xs[:, 0], ts, us)
            loss = jnp.nanmean((xs_pred-xs)**2) + jnp.nanmean((xs_pred[:-1, -1] - xs[1:, 0]) ** 2)
        else:
            xs_pred = self.solve_ode(xs[0], ts, us, max_steps=max_steps)
            loss = jnp.nanmean((xs_pred-xs)**2)
        return loss

    def test_derivative(self, xs, ts, us, dt0, bm_key):
        dts = dt0 * jnp.ones_like(ts)
        dWt = jnp.sqrt(dts)[..., None] * jr.normal(bm_key, (len(ts), self.latent_dim))
        f_post = jax.vmap(lambda t, x: self.posterior_drift(t, x, ((ts, us), (None, None))))(ts, xs)

        fs = f_post * dts[..., None]
        def _g_func(t, x, dW):
            g = self.diffusion_fn(t, x, None)
            return g @ dW

        gs = jax.vmap(_g_func)(ts, xs, dWt)
        return fs, gs

    def sample_derivative(self, xs, ts, us, dt0, bm_keys):
        _sample_derivative = lambda k: jax.vmap(lambda x, t, u: self.test_derivative(x, t, u, dt0, k),
                                                in_axes=(0, 0, None if us is None else 0))(xs, ts, us)
        dsampled = jax.vmap(_sample_derivative)(bm_keys)
        return dsampled

class LatentSDESystem:
    """Complete latent SDE system with encoder/decoder and log-ratio computation"""

    def __init__(self, obs_dim: int, obs_length: int, latent_dim: int, context_dim: int, hidden_dim_drift: int,
                 depth_drift: int, hidden_dim_diff: int, depth_diff: int, hidden_dim_decoder: int, depth_decoder: int,
                 *, key: jax.Array, ft: bool, encoder: str, u_size: int, model_name: str='NeuralODE'):

        e_key, q_key, s_key, d_key = jr.split(key, 4)

        self.obs_dim = obs_dim
        self.obs_length = obs_length
        self.latent_dim = latent_dim

        # Encoder: processes time series to produce context
        if encoder == 'GRU':
            self.encoder = GRU(input_size=obs_dim, hidden_size=context_dim, key=e_key)
        elif encoder == 'MLP':
            self.encoder = eqx.nn.MLP(in_size=obs_dim * self.obs_length, out_size=context_dim * self.obs_length,
                                      width_size=context_dim, depth=1, key=e_key)
        else:
            self.encoder = eqx.nn.Linear(in_features=obs_dim, out_features=context_dim, key=e_key)

        # Initial distribution network
        self.qz0_net = eqx.nn.Linear(context_dim, latent_dim * 2, key=q_key)  # mean and log_std

        # Latent SDE
        self.sde = LatentSDE(latent_dim, hidden_dim_drift, depth_drift, hidden_dim_diff, depth_diff,
                             key=s_key, ft=ft, context_dim=context_dim, u_size=u_size, model_name=model_name)

        # Decoder: maps latent space back to observations
        self.decoder = eqx.nn.MLP(latent_dim, obs_dim, hidden_dim_decoder, depth_decoder, activation=jax.nn.relu,
                                  key=d_key)

        # Prior parameters (learnable)
        self.pz0_mean = jnp.zeros((latent_dim,))
        self.pz0_logstd = jnp.zeros((latent_dim,))

        self._context = None

    def get_initial_distribution(self, ctx0: jax.Array, key: jax.Array) -> Tuple[jax.Array, dict]:
        """Get initial latent state and KL divergence with prior"""
        params = self.qz0_net(ctx0)
        qz0_mean, qz0_logstd = jnp.split(params, 2, axis=-1)

        # Sample initial state
        eps = jr.normal(key, qz0_mean.shape)
        z0 = qz0_mean + jnp.exp(qz0_logstd) * eps

        # Compute KL divergence with prior
        qz0_var = jnp.exp(2 * qz0_logstd)
        pz0_var = jnp.exp(2 * self.pz0_logstd)

        kl_div = 0.5 * jnp.sum(
            (qz0_var + (qz0_mean - self.pz0_mean) ** 2) / pz0_var
            - 1 + 2 * self.pz0_logstd - 2 * qz0_logstd,
            axis=-1
        )

        return z0, {"kl_initial": kl_div}

    def decode(self, z: jax.Array) -> jax.Array:
        """Decode latent state to observation space"""
        return jax.vmap(self.decoder)(z)

    def solve_sde_with_log_ratio(self, z0: jax.Array, ts: jax.Array, us, key: jax.Array,
                                 dt0: float = 0.01, max_steps=4096, adaptive=False) -> Tuple[jax.Array, jax.Array]:
        """
        Solve the latent SDE and compute log-ratio for KL divergence along path.

        This implements the log-ratio computation similar to torchsde's logqp functionality.
        """
        if self._context is None:
            raise ValueError("Must call contextualize() first")

        # Initial augmented state
        augmented_y0 = jnp.concatenate([z0, jnp.array([0.0])])  # log_ratio starts at 0

        # Create diffrax terms
        drift = diffrax.ODETerm(self.sde.augmented_drift)
        control = diffrax.VirtualBrownianTree(ts[0], ts[-1]+1e-3, tol=1e-3, shape=(self.latent_dim,), key=key)
        cvf = diffrax.ControlTerm(self.sde.augmented_diffusion, control)
        terms = diffrax.MultiTerm(drift, cvf)

        saveat = diffrax.SaveAt(ts=ts)
        if adaptive:
            solver = diffrax.StratonovichMilstein()
            solution = diffrax.diffeqsolve(terms=terms, solver=solver, t0=ts[0], t1=ts[-1], dt0=dt0, y0=augmented_y0,
                                           args=((ts, us), self._context), saveat=saveat, max_steps=max_steps,
                                           stepsize_controller=diffrax.PIDController(rtol=1e-3, atol=1e-6))
        else:
            solver = diffrax.Euler()
            solution = diffrax.diffeqsolve(terms=terms, solver=solver, t0=ts[0], t1=ts[-1], dt0=dt0, y0=augmented_y0,
                                           args=((ts, us), self._context), saveat=saveat, max_steps=max_steps)

        # Split solution
        zs = solution.ys[..., :-1]  # Latent trajectories
        log_ratios = solution.ys[..., -1]  # Log-ratio trajectory

        return zs, log_ratios

    def solve_sde(self, z0: jax.Array, ts: jax.Array, us, key: jax.Array, dt0: float = 0.01, max_steps=4096, adaptive=False) -> Tuple[jax.Array, jax.Array]:
        if self._context is None:
            raise ValueError("Must call contextualize() first")

        # Augmented state: [z, log_ratio]
        def _drift_func(t, z, args):
            (ts, us), (txs, ctxs) = args
            if ctxs.ndim == 1:
                f_post = self.sde.drift_with_context(t, z, (ts, us, ctxs))
            else:
                ctx_t_fun = lambda t: interp1d(t, txs, ctxs, method='nearest')
                f_post = self.sde.drift_with_context(t, z, (ts, us, ctx_t_fun(t)))
            return f_post

        # Create diffrax terms
        drift = diffrax.ODETerm(_drift_func)
        control = diffrax.VirtualBrownianTree(ts[0], ts[-1]+1e-3, tol=1e-3, shape=(self.latent_dim,), key=key)
        cvf = diffrax.ControlTerm(self.sde.diffusion_fn, control)
        terms = diffrax.MultiTerm(drift, cvf)

        saveat = diffrax.SaveAt(ts=ts)
        if adaptive:
            solver = diffrax.StratonovichMilstein()
            solution = diffrax.diffeqsolve(terms=terms, solver=solver, t0=ts[0], t1=ts[-1], dt0=dt0, y0=z0,
                                           args=((ts, us), self._context), saveat=saveat, max_steps=max_steps,
                                           stepsize_controller=diffrax.PIDController(rtol=1e-3, atol=1e-6))
        else:
            solver = diffrax.Euler()
            solution = diffrax.diffeqsolve(terms=terms, solver=solver, t0=ts[0], t1=ts[-1], dt0=dt0, y0=z0,
                                           args=((ts, us), self._context), saveat=saveat, max_steps=max_steps)

        return solution.ys

    def solve_with_prior(self, z0: jax.Array, ts: jax.Array, us, max_steps=4096) -> Tuple[jax.Array, jax.Array]:
        f_prior = lambda t, z, args: self.sde.prior_drift(t, z, args)
        drift = diffrax.ODETerm(f_prior)
        solution = diffrax.diffeqsolve(terms=drift, solver=diffrax.Tsit5(), t0=ts[0], t1=ts[-1], dt0=ts[1]-ts[0],
                                       stepsize_controller=diffrax.PIDController(rtol=1e-3, atol=1e-6), y0=z0,
                                       args=((ts, us), (None, None)), saveat=diffrax.SaveAt(ts=ts), max_steps=max_steps)
        return solution.ys

    def contextualize(self, ctx):
        self._context = ctx  # A tuple of tensors of sizes (T,), (T, batch_size, d).

    def generator(self, xs, ts, us, key, noise_std=1, dt0=0.01, max_steps=4096, adaptive=False):
        """
        Forward pass similar to torchsde example.

        Returns:
            log_pxs: Log-likelihood of observations
            logqp: Total KL divergence (initial + path)
        """
        keys = jr.split(key, 2)
        # Contextualize
        if isinstance(self.encoder, GRU):
            tx = ts[-self.obs_length:]
            ctx = self.encoder(xs[-self.obs_length:][::-1])[1][::-1]
            ctx0 = ctx[0]
        elif isinstance(self.encoder, eqx.nn.MLP):
            tx = ts[-self.obs_length:]
            ctx = self.encoder(xs[-self.obs_length:].reshape(-1)).reshape((self.obs_length, -1))
            ctx0 = ctx[0]
        else:
            tx = ts[0]
            ctx = self.encoder(xs[0])
            ctx0 = ctx

        self.contextualize((tx, ctx))

        # Get initial distribution and sample
        z0, init_info = self.get_initial_distribution(ctx0, keys[0])

        # Solve SDE with log-ratio
        zs, log_ratios = self.solve_sde_with_log_ratio(z0, ts, us, keys[1], dt0=dt0, max_steps=max_steps, adaptive=adaptive)

        # Decode to observation space
        xs_pred = self.decode(zs)

        # Compute log-likelihood of observations (assuming Gaussian)
        log_pxs = - 0.5 * jnp.mean((xs - xs_pred) ** 2) / (noise_std ** 2)
        log_pxs = log_pxs - xs.size * jnp.log(noise_std * jnp.sqrt(2 * jnp.pi))

        # Total KL divergence
        logqp0 = init_info["kl_initial"]
        logqp_path = log_ratios[-1]  # Final log-ratio value
        logqp = logqp0 + logqp_path

        return xs_pred, log_pxs, logqp

    def generator_prior(self, xs, ts, us, key, max_steps=4096):
        # Contextualize
        if isinstance(self.encoder, GRU):
            tx = ts[-self.obs_length:]
            ctx = self.encoder(xs[-self.obs_length:][::-1])[1][::-1]
            ctx0 = ctx[0]
        elif isinstance(self.encoder, eqx.nn.MLP):
            tx = ts[-self.obs_length:]
            ctx = self.encoder(xs[-self.obs_length:].reshape(-1)).reshape((self.obs_length, -1))
            ctx0 = ctx[0]
        else:
            tx = ts[0]
            ctx = self.encoder(xs[0])
            ctx0 = ctx
        self.contextualize((tx, ctx))
        # Get initial distribution and sample
        z0, init_info = self.get_initial_distribution(ctx0, key)
        # Solve SDE with log-ratio
        zs = self.solve_with_prior(z0, us, ts, max_steps=max_steps)
        # Decode to observation space
        xs_pred = self.decode(zs)
        return xs_pred

    @eqx.filter_jit
    def pre_train(self, xs, ts, us, key, max_steps=4096, ms=False):
        if ms:
            if us is None:
                xs_pred = jax.vmap(lambda x, t, k: self.generator_prior(x, t, us, k, max_steps=max_steps),
                                   in_axes=(0, 0, None))(xs, ts, key)
            else:
                xs_pred = jax.vmap(lambda x, t, u, k: self.generator_prior(x, t, u, k, max_steps=max_steps),
                                   in_axes=(0, 0, 0, None))(xs, ts, us, key)
            cont_loss = jnp.sum((xs_pred[:-1, -1] - xs[1:, 0]) ** 2)
            loss = jnp.mean((xs_pred-xs)**2) + jnp.mean(cont_loss)
        else:
            xs_pred = self.generator_prior(xs, ts, us, key, max_steps=max_steps)
            loss = jnp.mean((xs_pred-xs)**2)
        return loss

    @eqx.filter_jit
    def train(self, xs, ts, us, key, sigma=0.1, dt0=0.01, beta=1e-2, max_steps=4096, adaptive=False, ms=False):
        if ms:
            if us is None:
                xs_pred, log_pxs, logqp = jax.vmap(lambda x, t, k: self.generator(
                    x, t, us, k, noise_std=sigma, dt0=dt0, max_steps=max_steps, adaptive=adaptive),
                                                   in_axes=(0, 0, None))(xs, ts, key)
            else:
                xs_pred, log_pxs, logqp = jax.vmap(lambda x, t, u, k: self.generator(
                    x, t, u, k, noise_std=sigma, dt0=dt0, max_steps=max_steps, adaptive=adaptive),
                                                   in_axes=(0, 0, 0, None))(xs, ts, us, key)
            cont_loss = jnp.sum((xs_pred[:-1, -1] - xs[1:, 0]) ** 2)
            loss = - jnp.mean(log_pxs - beta * logqp) + jnp.mean(cont_loss)
        else:
            xs_pred, log_pxs, logqp = self.generator(xs, ts, us, key, noise_std=sigma, dt0=dt0, max_steps=max_steps,
                                                     adaptive=adaptive)
            loss = - jnp.mean(log_pxs - beta * logqp)
        return loss

    def inference(self, x0, ts, us, key, dt0=0.01, max_steps=4096, adaptive=False):
        keys = jr.split(key, 2)
        if isinstance(self.encoder, GRU):
            tx = ts[-self.obs_length:]
            xs = jnp.repeat(x0[None], axis=0, repeats=self.obs_length)
            ctx = self.encoder(xs[-self.obs_length:][::-1])[1][::-1]
            ctx0 = ctx[0]
        elif isinstance(self.encoder, eqx.nn.MLP):
            tx = ts[-self.obs_length:]
            xs = jnp.repeat(x0[None], axis=0, repeats=self.obs_length)
            ctx = self.encoder(xs[-self.obs_length:].reshape(-1)).reshape((self.obs_length, -1))
            ctx0 = ctx[0]
        else:
            tx = ts[0]
            ctx = self.encoder(x0)
            ctx0 = ctx
        self.contextualize((tx, ctx))
        # Get initial distribution and sample
        z0, init_info = self.get_initial_distribution(ctx0, keys[0])
        # Solve SDE with log-ratio
        zs = self.solve_sde(z0, ts, us, keys[1], dt0=dt0, max_steps=max_steps, adaptive=adaptive)
        xs_pred = self.decode(zs)
        return xs_pred

    @eqx.filter_jit
    def sample(self, x0, ts, us, keys, dt0=0.01, max_steps=4096, adaptive=False):
        if x0.shape[0] > 1:
            if us is None:
                func = lambda k: jax.vmap(lambda _x0, t: self.inference(
                    _x0, t, us, k, dt0=dt0, max_steps=max_steps, adaptive=adaptive))(x0, ts)
            else:
                func = lambda k: jax.vmap(lambda _x0, u, t: self.inference(
                    _x0, t, u, k, dt0=dt0, max_steps=max_steps, adaptive=adaptive))(x0, ts, us)
            xs_pred = jax.vmap(func)(keys)
        else:
            func = lambda k: self.inference(x0[0], ts[0], k, us[0] if us is not None else None, dt0=dt0,
                                            max_steps=max_steps, adaptive=adaptive)
            xs_pred = jax.vmap(func)(keys)
            xs_pred = xs_pred[:, jnp.newaxis]
        return xs_pred

# Training utilities
def elbo_loss(log_pxs: float, logqp: float, beta: float = 1.0) -> float:
    """Evidence Lower BOund (ELBO) loss"""
    return - (log_pxs - beta * logqp)

# Loss functions for training
def reconstruction_loss(x_true: jax.Array, x_pred: jax.Array) -> float:
    """MSE reconstruction loss"""
    return jnp.mean((x_true - x_pred) ** 2)

def kl_divergence_loss(z_mean: jax.Array, z_log_std: jax.Array) -> float:
    """KL divergence between latent distribution and standard normal"""
    z_var = jnp.exp(2 * z_log_std)
    return 0.5 * jnp.mean(z_mean ** 2 + z_var - 2 * z_log_std - 1)