import os
os.environ["JAX_PLATFORMS"] = 'cpu'
import jax
jax.default_device('cpu')
import jax.numpy as jnp
import jax.random as jr
import diffrax
import numpy as np


def drift_vdp(t, u, args):
    x, v = u
    mu = args["mu"]
    dxdt = v
    dvdt = mu * (1 - x ** 2) * v - x
    return jnp.array([dxdt, dvdt])

def diffusion_vdp(t, u, args):
    sigma, tfinal  = args["sigma"], args["tfinal"]
    return 2 * sigma * t / tfinal

def drift_dw(t, u, args):
    return 2*u - u ** 3

def diffusion_dw(t, u, args):
    return args["sigma"]

def drift_slc(t, u, args):
    x, y, z = u[0], u[1], u[2]
    x_dot = args["sigma"] * (y - x)
    y_dot = x * (args["rho"] - z) - y
    z_dot = x * y - args["beta"] * z
    return jnp.array([x_dot, y_dot, z_dot])

def diffusion_slc(t, u, args):
    return args["alpha"]

def drift_lv(t, u, args):
    x, y = u[0], u[1]
    x_dot = args["alpha"] * x - args["beta"] * x * y
    y_dot = args["delta"] * x * y - args["gamma"] * y
    return jnp.array([x_dot, y_dot])

def diffusion_lv(t, u, args):
    x, y = u[0], u[1]
    return jnp.array([args["sigma_x"] * x, args["sigma_y"] * y])

def drift_heston(t, u, args):
    S, v = u[0], u[1]
    S = jnp.clip(S, 1e-6)
    v = jnp.clip(v, 1e-6)
    dS = args["mu"] * S
    dv = args["kappa"] * ( args["theta"] - v)
    return jnp.array([dS, dv])

def diffusion_heston(t, u, args):
    S, v = u[0], u[1]
    S = jnp.clip(S, 1e-6)
    v = jnp.clip(v, 1e-6)
    dS = jnp.sqrt(v) * S
    dv = args["sigma"] * jnp.sqrt(v)
    return args["cov"] @ jnp.array([dS, dv])

def drift_fhn(t, u, args):
    v, w = u[0], u[1]
    dv = 1/args["eps"] * (v - v ** 3 - w)
    dw = args["gamma"] * v - w + args["beta"]
    return jnp.array([dv, dw])

def diffusion_fhn(t, u, args):
    return jnp.array([args["sigma_v"], args["sigma_w"]])

def euler_maruyama(terms, y0, t0, num_steps, dt, n_size=1, args=None):

    args = args or {}
    drift_fn, diffusion_fn = terms.terms

    def step_fn(carry, i):
        t, y, f, g = carry
        t_next = t + dt
        drift = drift_fn.vf(t, y, args)
        diffusion = diffusion_fn.vf(t, y, args)
        dW = diffusion_fn.contr(t, t_next)
        f_next = drift * dt
        g_next = diffusion * dW
        y_next = y + f_next + g_next
        _g_next = g_next if n_size > 1 else g_next[None]
        return (t_next, y_next, f_next, _g_next), (t_next, y_next, f_next, _g_next)
    _, (ts, ys, fs, gs) = jax.lax.scan(step_fn, (t0, y0, jnp.zeros_like(y0), jnp.zeros((n_size,))), length=num_steps)
    ts = jnp.concatenate([jnp.array(t0)[None], ts])
    ys = jnp.concatenate([y0[None, :], ys], axis=0)
    fs = jnp.concatenate([jnp.zeros_like(y0)[None, :], fs], axis=0)
    gs = jnp.concatenate([jnp.zeros((1, n_size)), gs], axis=0)
    return ts, ys, fs, gs

def solver_sde(drift, diffusion, y0, t0, t1, bm_key, model_name, n_size=1, sigma=0.1, dt=1e-3):
    if model_name == 'van_der_pol':
        args = {"mu": 2.0, "sigma": sigma, "tfinal": t1}
    elif model_name == "lorenz":
        args = {"sigma":10, "rho":28, "beta":8/3, "alpha":jnp.array([0.05, 0.1, 0.15])}
    elif model_name == "double_well":
        args = {"sigma": sigma}
    elif model_name == "heston":
        corr = jnp.array([[1.0, 0.2], [0.2, 1.0]])
        args = {"mu":0.05, "kappa":1.0, "theta":0.04, "sigma":sigma, "cov": jnp.linalg.cholesky(corr)}
    elif model_name == "fhn":
        args = {"eps":0.1, "gamma":1.5, "beta":0.8, "sigma_v":sigma, "sigma_w":sigma}
    elif model_name == "lotka_volterra":
        args = {"alpha": 2.0, "beta":1.0, "delta":0.1, "gamma":1.5, "sigma_x":sigma, "sigma_y": sigma}
    else: # geometric brownian motion
        args = {"mu": 0.1, "sigma": 0.1}
    bm = diffrax.UnsafeBrownianPath(shape=(), key=bm_key, levy_area=diffrax.BrownianIncrement)
    drift_term = diffrax.ODETerm(drift)
    diffusion_term = diffrax.ControlTerm(diffusion, bm)
    terms = diffrax.MultiTerm(drift_term, diffusion_term)
    ts, ys, fs, gs = euler_maruyama(terms, y0, t0, t1 // dt, dt, n_size=n_size, args=args)
    return ts, ys, fs, gs

def solver_sde_vdp_regular(drift, diffusion, y0, t0, t1, tp, bm_key, sigma=0.1, dt=1e-3):
    args = {"mu": 2.0, "sigma": sigma, "tfinal": t1}
    bm = diffrax.UnsafeBrownianPath(shape=(), key=bm_key, levy_area=diffrax.BrownianIncrement)
    drift_term = diffrax.ODETerm(drift)
    diffusion_term = diffrax.ControlTerm(diffusion, bm)
    terms = diffrax.MultiTerm(drift_term, diffusion_term)
    ts, ys, fs, gs = euler_maruyama(terms, y0, t0, t1 // dt, dt, args=args)
    ys_save = [jnp.interp(tp, ts, ys[:, i]) for i in range(ys.shape[-1])]
    return jnp.stack(ys_save, axis=-1)

def time_function_exp(scale, shape):
    dt = np.random.exponential(scale, shape)
    _shape = list(dt.shape)
    _shape[-1] = 1
    ts = np.concatenate([np.zeros(_shape), np.cumsum(dt, axis=-1)], axis=-1)
    return ts

def max_time(nested_list):
    max_val = float('-inf')
    for group in nested_list:
        for pair in group:
            for arr in pair:
                for val in arr:
                    if val > max_val:
                        max_val = val
    return max_val

def interpolate(tp, ts, ys):
    yp = []
    if not isinstance(ys, list):
        ys = [ys]
    for _ys in ys:
        _yp = jnp.stack([jnp.interp(tp, ts, _ys[:, i]) for i in range(_ys.shape[-1])], axis=-1)
        yp.append(_yp)
    return yp

num_samples = [200, 2000]
scales = [0.025]
sigma = 0.1
model_names = ["van_der_pol"]
y0_key = jr.key(0)
bm_key = jr.key(42)
ts_all = {}
for model_name in model_names:
    ts_nums = []
    for num in num_samples:
        ts_scales = []
        for scale in scales:
            if scale == 0.5:
                if model_name == "lorenz":
                    length = 60
                else:
                    length = 20
            elif scale == 0.05:
                if model_name == "lorenz":
                    length = 600
                else:
                    length = 200
            else:
                if model_name == "lorenz":
                    length = 1200
                else:
                    length = 400

            tp = np.concatenate([np.zeros((num, 1)),
                                 np.cumsum(np.random.exponential(scale, (num, length)), axis=-1)], axis=-1)
            ts_scales.append(tp)

        ts_nums.append(ts_scales)

    ts_all[model_name] = ts_nums

for i, model_name in enumerate(model_names):
    if model_name == "van_der_pol":
        _drift = drift_vdp
        _diffusion = diffusion_vdp
        n_size = 1
        Y0 = [jr.normal(y0_key, (num, 2)) for num in num_samples]
        t1 = max_time(ts_all[model_name]) + 1e-3
    elif model_name == "double_well":
        _drift = drift_dw
        _diffusion = diffusion_dw
        n_size = 1
        Y0 = [jr.normal(y0_key, (num, 1)) for num in num_samples]
        t1 = max_time(ts_all[model_name]) + 1e-3
    elif model_name == "lorenz":
        _drift = drift_slc
        _diffusion = diffusion_slc
        n_size = 1
        Y0 = [jr.normal(y0_key, (num, 3)) for num in num_samples]
        t1 = max_time(ts_all[model_name]) + 1e-3
    elif model_name == "fhn":
        _drift = drift_fhn
        _diffusion = diffusion_fhn
        n_size = 2
        Y0 = [jnp.stack([jr.uniform(y0_key, (num,), minval=-2, maxval=2),
                         jr.uniform(y0_key, (num,), minval=-0.5, maxval=1.5)], axis=-1) for num in num_samples]
        t1 = max_time(ts_all[model_name]) + 1e-3
    elif model_name == "heston":
        _drift = drift_heston
        _diffusion = diffusion_heston
        n_size = 2
        Y0 = [jnp.stack([jr.uniform(y0_key, (num,), minval=0, maxval=1),
                         jr.uniform(y0_key, (num,), minval=0.01, maxval=0.025)], axis=-1) for num in num_samples]
        t1 = max_time(ts_all[model_name]) + 1e-3
    else:
        _drift = drift_lv
        _diffusion = diffusion_lv
        n_size = 2
        Y0 = [jnp.stack([jr.uniform(y0_key, (num,), minval=5, maxval=20),
                         jr.uniform(y0_key, (num,), minval=1, maxval=15)], axis=-1) for num in num_samples]
        t1 = max_time(ts_all[model_name]) + 1e-3

    _func = lambda _y0, k: solver_sde(_drift, _diffusion, _y0, 0, t1, k, model_name, n_size=n_size, sigma=sigma, dt=1e-3)
    for j, (y0, num) in enumerate(zip(Y0, num_samples)):
        keys = jr.split(bm_key, num)
        ts, ys, fs, gs = jax.vmap(_func)(y0, keys)
        for s, scale in enumerate(scales):
            if scale == 0.5:
                if model_name == "lorenz":
                    length = 60
                else:
                    length = 20
            elif scale == 0.05:
                if model_name == "lorenz":
                    length = 600
                else:
                    length = 200
            else:
                if model_name == "lorenz":
                    length = 1200
                else:
                    length = 400
            print(f"model name: {model_name}, num: {num}, scale: {scale}, length: {length}")
            tp = ts_all[model_name][j][s]

            yp, fp, gp = jax.vmap(interpolate)(tp, ts, [ys, fs, gs])
            if model_name == "van_der_pol":
                np.save(f"data/synthetic_data/{model_name}_{num}_{scale}_sigma_{sigma}.npy",
                    jnp.concatenate([tp[..., None], yp, fp, gp], axis=-1))
            else:
                np.save(f"data/synthetic_data/{model_name}_{num}_{scale}.npy",
                    jnp.concatenate([tp[..., None], yp, fp, gp], axis=-1))