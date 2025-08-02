import os
from numba import cuda
if cuda.is_available():
    import jax
else:
    os.environ["JAX_PLATFORMS"] = 'cpu'
    import jax
    jax.default_device('cpu')
from typing import Union, List
import diffrax
import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
from NF_NeuralODE_models import NeuralODE
from normalizing_flow_jax import RealNVP, MAF
from stratonovich_solvers import (Euler_dtdW, Heun_dtdW, EulerHeun_dtdW, ReversibleHeun_dtdW, Midpoint_dtdW,
                                  Milstein_dtdW,
                                  Euler_dt, Heun_dt, EulerHeun_dt, ReversibleHeun_dt, Midpoint_dt, Milstein_dt)


class StochasticDiffusion(eqx.Module):
    flow: Union[eqx.Module, List[eqx.Module]]
    model_name: str

    def __init__(self, n_blocks, x_size, hidden_size, depth, *, key, model_name):
        k1, k2 = jr.split(key, 2)
        n_size = 1
        if 'diag' in model_name:
            if 'dw' in model_name:
                cond_size = 2
            else:
                cond_size = 3
        elif 'single' in model_name:
            if 'dw' in model_name:
                cond_size = 0
            else:
                cond_size = 1
        else: #['one' or 'all']
            if 'dw' in model_name:
                cond_size = x_size + 1
            else:
                cond_size = x_size + 2
        if 'MAF' in model_name:
            flow = MAF(n_blocks, n_size, hidden_size, depth, cond_label_size=cond_size, input_order="random",
                       key=k1, batch_norm=True if 'batch_norm' in model_name else False,
                       batch_norm_last=True if 'batch_norm_last' in model_name else False)
        else:
            flow = RealNVP(n_blocks, n_size, hidden_size, depth, cond_label_size=cond_size, key=k1,
                           batch_norm=True if 'batch_norm' in model_name else False,
                           batch_norm_last=True if 'batch_norm_last' in model_name else False)

        if 'diag' in model_name or 'all' in model_name:
            self.flow = [flow] * x_size
        else:
            self.flow = flow
        self.model_name = model_name

    def __call__(self, cond, key, args=None, inference=False):
        noise_key, dropout_key = jr.split(key, 2)
        dt = args['dt']
        dt = jnp.clip(dt, a_min=1e-6)
        n_size = 1
        dw_shape = list(dt.shape) + [n_size]
        if 'diag' in self.model_name or 'all' in self.model_name:
            transformed_noise = []
            logJ = 0
            for i in range(len(self.flow)):
                if 'dtdw' in self.model_name:
                    dw = jr.normal(key=noise_key, shape=dw_shape) * jnp.sqrt(dt[..., None])
                elif 'dw' in self.model_name and 'dt' not in self.model_name:
                    dw = args['dw']
                else:
                    dw = jr.normal(key=noise_key, shape=dw_shape)
                if 'dw' in self.model_name:
                    if 'diag' in self.model_name:
                        _cond = jnp.stack([cond[..., i], cond[..., -1]], axis=-1)
                    else: # 'all'
                        _cond = cond
                else:
                    if 'diag' in self.model_name:
                        _cond = jnp.stack([cond[..., i], cond[..., -1], dt], axis=-1)
                    else: #'all'
                        _cond = jnp.concatenate([cond, dt[..., None]], axis=-1)
                _noise, _logJ = self.flow[i].inverse(dw, cond=_cond, key=dropout_key, inference=inference)
                logJ += _logJ
                transformed_noise.append(_noise)
            transformed_noise = jnp.concatenate(transformed_noise, axis=-1)
        elif 'one' in self.model_name:
            if 'dtdw' in self.model_name:
                dw = jr.normal(key=noise_key, shape=dw_shape) * jnp.sqrt(dt[..., None])
            elif 'dw' in self.model_name and 'dt' not in self.model_name:
                dw = args['dw']
            else:
                dw = jr.normal(key=noise_key, shape=dw_shape)
            if 'dw' in self.model_name:
                _cond = cond
            else:
                _cond = jnp.concatenate([cond, dt[..., None]], axis=-1)
            transformed_noise, logJ = self.flow.inverse(dw, cond=_cond, key=dropout_key, inference=inference)
        else: # single
            if 'dtdw' in self.model_name:
                dw = jr.normal(key=noise_key, shape=dw_shape) * jnp.sqrt(dt[..., None])
            elif 'dw' in self.model_name and 'dt' not in self.model_name:
                dw = args['dw']
            else:
                dw = jr.normal(key=noise_key, shape=dw_shape)
            if 'dw' in self.model_name:
                _cond = None
            else:
                _cond = dt[..., None]
            _transformed_noise, _logJ = self.flow.inverse(dw, cond=_cond, key=dropout_key, inference=inference)
            noise_list, logJ_list = [], []
            selected_i = 1
            for i in range(cond.shape[-1]-1):
                if i == selected_i:
                    noise_list.append(_transformed_noise)
                    logJ_list.append(_logJ)
                else:
                    noise_list.append(jnp.zeros_like(_transformed_noise))
                    logJ_list.append(jnp.zeros_like(_logJ))
            transformed_noise = jnp.concatenate(noise_list, axis=-1)
            logJ = jnp.concatenate(logJ_list, axis=-1)
        return transformed_noise, jnp.nansum(logJ)

    def flow_forward(self, diff, cond, key, dt):
        dt = jnp.clip(jnp.abs(dt), a_min=1e-4)
        if 'diag' in self.model_name:
            z_loss = 0
            logJ = 0
            for i in range(len(self.flow)):
                if 'dw' in self.model_name:
                    _cond = jnp.stack([cond[..., i], cond[..., -1]], axis=-1)
                    _zs, _logJ = self.flow[i](diff[..., [i]], cond=_cond, key=key, inference=False)
                    z_loss += 0.5 * _zs ** 2 / dt + 0.5 * jnp.log(dt)
                else:
                    _cond = jnp.stack([cond[..., i], cond[..., -1], dt], axis=-1)
                    _zs, _logJ = self.flow[i](diff[..., [i]], cond=_cond, key=key, inference=False)
                    z_loss += 0.5 * _zs ** 2
                logJ += _logJ
        elif 'all' in self.model_name:
            z_loss = 0
            logJ = 0
            for i in range(len(self.flow)):
                if 'dw' in self.model_name:
                    _zs, _logJ = self.flow[i](diff[..., [i]], cond=cond, key=key, inference=False)
                    z_loss += 0.5 * _zs ** 2 / dt + 0.5 * jnp.log(dt)
                else:
                    _cond = jnp.stack([cond, dt[..., None]], axis=-1)
                    _zs, _logJ = self.flow[i](diff[..., [i]], cond=_cond, key=key, inference=False)
                    z_loss += 0.5 * _zs ** 2
                logJ += _logJ
        elif 'single' in self.model_name:
            selected_i = 1
            if 'dw' in self.model_name:
                _cond = None
                zs, logJ = self.flow(diff[..., [selected_i]], cond=_cond, key=key, inference=False)
                z_loss = 0.5 * zs ** 2 / dt + 0.5 * jnp.log(dt)
            else:
                _cond = dt[..., None]
                zs, logJ = self.flow(diff[..., [selected_i]], cond=_cond, key=key, inference=False)
                z_loss = 0.5 * zs ** 2
        else: # 'one'
            diff = jnp.nanmean(diff, axis=-1, keepdims=True)
            if 'dw' in self.model_name:
                zs, logJ = self.flow(diff, cond=cond, key=key, inference=False)
                z_loss = 0.5 * zs ** 2 / dt + 0.5 * jnp.log(dt)
            else:
                _cond = jnp.concatenate([cond, dt[..., None]], axis=-1)
                zs, logJ = self.flow(diff, cond=_cond, key=key, inference=False)
                z_loss = 0.5 * zs ** 2

        return jnp.nanmean(z_loss) - jnp.nanmean(logJ)

    @eqx.filter_jit
    def train_forward(self, diff, cond, key, dts):
        flow_loss = jax.vmap(self.flow_forward, in_axes=(0, 0, None, 0))(diff, cond, key, dts)
        return jnp.nanmean(flow_loss)


class NFNeuralSDE(eqx.Module):
    x_size: int
    u_size: int
    drift_func: NeuralODE
    diff_func: StochasticDiffusion
    model_name: str

    def __init__(self, x_size, u_size, hidden_size_drift, hidden_size_diff, n_blocks, depth_drift, depth_diff,
                 *, key, model_name):
        self.x_size = x_size
        self.u_size = u_size
        self.model_name = model_name
        keys = jr.split(key, 2)
        self.drift_func = NeuralODE(x_size, x_size, u_size, hidden_size_drift, depth_drift, key=keys[0],
                                    model_name=model_name)
        self.diff_func = StochasticDiffusion(n_blocks, x_size, hidden_size_diff, depth_diff, key=keys[1],
                                             model_name=model_name)

    @eqx.filter_jit
    # # Stochastic ODE Solver (Euler-Maruyama)
    def generator(self, x0, ts, key, us, inference=False, dt0=1e-2, max_steps=4096, adaptive=False):
        dropout_key, bm_key = jr.split(key, 2)
        if 'dw' in self.model_name and 'dt' not in self.model_name:
            args = {"ts": ts, "us": us, "dt": jnp.array(ts[1]-ts[0]),
                    "dw": jnp.sqrt(ts[1] - ts[0]) * jr.normal(bm_key, (1,))}
        else:
            args = {"ts": ts, "us": us, "dt": jnp.array(ts[1] - ts[0])}
        drift_func = lambda t, y, args: (self.drift_func(t, y, args), 0)

        if 'milstein' in self.model_name:
            def diffusion(t, y, args):
                _y, _log = self.diff_func(jnp.concatenate([y[0], jnp.asarray(t)[None]], axis=-1), bm_key,
                                          args=args, inference=inference)
                if _y.shape != x0.shape:
                    return jnp.repeat(_y, x0.shape[0]), _log
                else:
                    return _y, _log
        else:
            diffusion = lambda t, y, args: self.diff_func(jnp.concatenate([y[0], jnp.asarray(t)[None]], axis=-1),
                                                          bm_key, args=args, inference=inference)

        drift_term = diffrax.ODETerm(drift_func)
        if 'dw' in self.model_name and 'dt' not in self.model_name:
            bm = diffrax.VirtualBrownianTree(ts[0]-1e-3, ts[-1]+1e-3, tol=1e-3, shape=(), key=bm_key,
                                             levy_area=diffrax.BrownianIncrement)
            diffusion_term = diffrax.ControlTerm(diffusion, bm)
        else:
            diffusion_term = diffrax.ODETerm(diffusion)
        terms = diffrax.MultiTerm(drift_term, diffusion_term)
        if 'midpoint' in self.model_name:
            if 'dw' in self.model_name and 'dt' not in self.model_name:
                solver = Midpoint_dtdW()
            else:
                solver = Midpoint_dt()

        elif 'euler' in self.model_name:
            if 'dw' in self.model_name and 'dt' not in self.model_name:
                solver = Euler_dtdW()
            else:
                solver = Euler_dt()

        elif 'reversible_heun' in self.model_name:
            if 'dw' in self.model_name and 'dt' not in self.model_name:
                solver = ReversibleHeun_dtdW()
            else:
                solver = ReversibleHeun_dt()

        elif 'euler_heun' in self.model_name:
            if 'dw' in self.model_name and 'dt' not in self.model_name:
                solver = EulerHeun_dtdW()
            else:
                solver = EulerHeun_dt()

        elif 'heun' in self.model_name:
            if 'dw' in self.model_name and 'dt' not in self.model_name:
                solver = Heun_dtdW()
            else:
                solver = Heun_dt()
        else:
            if 'dw' in self.model_name and 'dt' not in self.model_name:
                solver = Milstein_dtdW()
            else:
                solver = Milstein_dt()

        if adaptive:
            sol = diffrax.diffeqsolve(terms, solver, ts[0], ts[-1], dt0, (x0, 0), max_steps=max_steps,
                                      stepsize_controller=diffrax.PIDController(rtol=1e-2, atol=1e-4),
                                      args=args, saveat=diffrax.SaveAt(ts=ts))
        else:
            sol = diffrax.diffeqsolve(terms, solver, ts[0], ts[-1], dt0, (x0, 0), max_steps=max_steps,
                                      args=args, saveat=diffrax.SaveAt(ts=ts))
        return sol.ys

    @eqx.filter_jit
    def train_integral(self, ts, xs, us, key, ms=False, lamb=0, dt0=1e-2, max_steps=4096, adaptive=False):
        if ms:
            x_preds, logJ = jax.vmap(lambda x, t, k ,u: self.generator(x, t, k, u, inference=False, dt0=dt0,
                                                                       max_steps=max_steps, adaptive=adaptive),
                                     in_axes=(0, 0, None, None if us is None else 0))(xs[:, 0], ts, key, us)
            drift_loss = jnp.nanmean((x_preds - xs) ** 2) + jnp.nansum((x_preds[:-1, -1] - xs[1:, 0])**2)
        else:
            x_preds, logJ = self.generator(xs[0], ts, key, us, inference=False, dt0=dt0, max_steps=max_steps,
                                           adaptive=adaptive)
            drift_loss = jnp.nanmean((x_preds - xs) ** 2)
        return drift_loss + jnp.nanmean(logJ) * lamb

    def sample_sde(self, x0, ts, us, sample_key, dt0=1e-2, max_steps=4096, adaptive=False):
        _generator = lambda _x0, _ts, _k, _us: self.generator(_x0, _ts, _k, _us, inference=True, dt0=dt0,
                                                              max_steps=max_steps, adaptive=adaptive)[0]
        if x0.shape[0] > 1:
            xs = jax.vmap(
                lambda k: jax.vmap(_generator, in_axes=(0, 0, None, 0 if us is not None else None))(x0, ts, k, us))(
                sample_key)
        else:
            xs = jax.vmap(lambda k: _generator(x0[0], ts[0], k, us))(sample_key)
            xs = xs[:, jnp.newaxis]
        return xs

    def test_derivative(self, xs, ts, us, dt0, bm_key):
        dts = dt0 * jnp.ones_like(ts)
        if 'dw' in self.model_name and 'dt' not in self.model_name:
            args = {"ts": ts, "us": us, "dt": jnp.array(ts[1]-ts[0]),
                    "dw": jnp.sqrt(dts) * jr.normal(bm_key, (1,))}
        else:
            args = {"ts": ts, "us": us, "dt": dts}
        fs = jax.vmap(lambda t, x: self.drift_func(t, x, args))(ts, xs)

        gs = self.diff_func(jnp.concatenate([xs, ts[..., None]], axis=-1), bm_key, args=args, inference=False)[0]
        return fs*dts[..., None], gs*dts[..., None]

    def sample_derivative(self, xs, ts, us, dt0, bm_keys):
        _sample_derivative = lambda k: jax.vmap(lambda x, t, u: self.test_derivative(x, t, u, dt0, k),
                                                in_axes=(0, 0, None if us is None else 0))(xs, ts, us)
        dsampled = jax.vmap(_sample_derivative)(bm_keys)
        return dsampled


