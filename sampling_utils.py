import jax
import jax.numpy as jnp
import jax.random as jr
from copy import deepcopy
import diffrax
import jax_dataloader as jdl


def _get_data(ts, *, key, irregular=False, mu=2.0, sigma=0.0):
    y0_key, drop_key = jr.split(key, 2)
    y0 = jr.normal(y0_key, (2,))
    t0 = ts[0]
    t1 = ts[-1]
    dt = ts[1] - ts[0]
    if irregular:
        ts -= jr.uniform(drop_key, shape=ts.shape, minval=-dt / 2, maxval=dt / 2)
        ts = jnp.concatenate([t0[None], ts[1:-1], t1[None]])

    def vdp(t, u, args):
        mu = args["mu"]
        x, v = u
        dxdt = v
        dvdt = mu * (1 - x ** 2) * v - x
        return jnp.array([dxdt, dvdt])

    sol = diffrax.diffeqsolve(
        diffrax.ODETerm(vdp), diffrax.Tsit5(), t0, t1, dt, y0, args={"mu": mu}, saveat=diffrax.SaveAt(ts=ts)
    )
    xs = sol.ys + 0.01 * jr.normal(key, (sol.ys.shape))
    xs_dot = jax.vmap(vdp, in_axes=(0, 0, None))(ts, xs, {"mu": mu})
    return ts, xs, xs_dot, None

def _get_data_sde(ts, key, irregular=False, mu=2.0, sigma=0.1):
    bm_key, y0_key, drop_key = jr.split(key, 3)
    t0 = ts[0]
    t1 = ts[-1]
    dt = ts[1] - ts[0]
    if irregular:
        ts -= jr.uniform(drop_key, shape=ts.shape, minval=-dt / 2, maxval=dt / 2)
        ts = jnp.concatenate([t0[None], ts[1:-1], t1[None]])

    def drift(t, u, args):
        x, v = u
        mu = args["mu"]
        dxdt = v
        dvdt = mu * (1 - x ** 2) * v - x
        return jnp.array([dxdt, dvdt])

    def diffusion(t, u, args):
        sigma = args["sigma"]
        return 2 * sigma * t / t1  # Noise applied independently to both x and y

    bm = diffrax.UnsafeBrownianPath(shape=(), key=bm_key, levy_area=diffrax.BrownianIncrement)
    drift_term = diffrax.ODETerm(drift)
    diffusion_term = diffrax.ControlTerm(diffusion, bm)
    terms = diffrax.MultiTerm(drift_term, diffusion_term)
    solver = diffrax.EulerHeun()
    y0 = jr.normal(y0_key, (2,))
    saveat = diffrax.SaveAt(ts=ts)
    sol = diffrax.diffeqsolve(
        terms, solver, t0, t1, dt, y0, args={"mu": mu, "sigma": sigma}, saveat=saveat, adjoint=diffrax.DirectAdjoint()
    )
    drift_dot = jax.vmap(drift, in_axes=(0, 0, None))(ts, sol.ys, {"mu": mu, "sigma": sigma})
    diff_dot = jax.vmap(diffusion, in_axes=(0, 0, None))(ts, sol.ys, {"mu": mu, "sigma": sigma})[:, None]
    return ts, sol.ys, drift_dot, diff_dot, None


def get_data(dataset_size, *, key, irregular, func, mu=2.0, sigma=0.1):
    ts = jnp.linspace(0.0, 10.0, 1001)
    key = jr.split(key, dataset_size)
    return jax.vmap(lambda k: func(ts, key=k, irregular=irregular, mu=mu, sigma=sigma))(key)


def normalize(x, min_data=None, max_data=None):
    if min_data is None:
        min_data = deepcopy(x)
        max_data = deepcopy(x)
        i = 0
        while i < x.ndim - 1:
            min_data = jnp.min(min_data, axis=i, keepdims=True)
            max_data = jnp.min(max_data, axis=i, keepdims=True)
            i += 1
    x_norm = (x - min_data) / (max_data - min_data)
    return x_norm, min_data, max_data


def denormalize(x, min_data, max_data):
    x_denorm = x * (max_data - min_data) + min_data
    return x_denorm


def get_data_from_file(file_x, experiment_name):
    data = jnp.load(file_x)[:, 1:, :]
    us = data[..., :2]
    if experiment_name == "observer":
        xs = data[..., 2 + 6:2 + 9]
        ys = data[..., 2:2 + 6]
    else:
        xs = data[..., 2:2 + 9]
        xs_dot = data[..., 2 + 9:]
    length_size = us.shape[1]
    ts = jnp.linspace(0, (length_size - 1) * 0.01, length_size)
    if experiment_name == "observer":
        return ts, xs, ys, us
    else:
        return ts, xs, xs_dot, us

def preprocess_data(ts, xs, us, batch_size, times=9, step=1, patch=10, split=True):
    def _split_with_overlap(a):
        a_list = [a[:, i:i + patch + 1] for i in range(0, a.shape[1] - 1, patch)]
        if a_list[-1].shape != a_list[0].shape:
            a_list.pop(-1)
        return jnp.stack(a_list, axis=1)
    def _roll(a, step, pad):
        if pad == 0:
            pad_shape = list(a.shape)
            pad_shape[1] = step
            tail = jnp.zeros(pad_shape)
        else:
            dt = a[:, 1] - a[:, 0]
            t1 = a[:, -1]
            if step == 1:
                tail = t1 + dt
                tail = tail[:, None]
            else:
                tail = jnp.stack([t1 + dt * s for s in range(1, step+1)], axis=-1)
        a_rolled = jnp.concatenate((a[:, step:], tail), axis=1)
        return a_rolled
    if times > 0:
        ts_rolled = [ts]
        xs_rolled = [xs]
        if us is not None:
            us_rolled = [us]
        for i in range(times):
            _ts = _roll(ts, step * (i+1), pad=None)
            _xs = _roll(xs, step * (i+1), pad=0)
            if us is not None:
                _us = _roll(us, step * (i+1), pad=0)
            ts_rolled.append(_ts)
            xs_rolled.append(_xs)
            if us is not None:
                us_rolled.append(_us)
        ts = jnp.concatenate(ts_rolled)
        xs = jnp.concatenate(xs_rolled)
        if us is not None:
            us = jnp.concatenate(us_rolled)
    if split:
        ts = _split_with_overlap(ts)
        xs = _split_with_overlap(xs)
        if us is not None:
            us = _split_with_overlap(us)

    _dataset = jdl.ArrayDataset(ts, xs) if us is None else jdl.ArrayDataset(ts, xs, us)
    dataloader = jdl.DataLoader(_dataset, backend='jax', batch_size=batch_size, shuffle=True, drop_last=False)
    return dataloader

def diff_data(ts, xs, brownian_key):
    dts = jnp.diff(ts, axis=-1)
    dxs = jnp.diff(xs, axis=-2)
    if dts.ndim == 3:
        dxt = jax.vmap(jax.vmap(lambda x, t:jnp.gradient(x, t, axis=0)))(xs, ts)
    else:
        dxt = jax.vmap(lambda x, t:jnp.gradient(x, t, axis=0))(xs, ts)
    var_dx = jnp.var(dxs, axis=0, ddof=1, keepdims=True)  # (time-1, dim)
    # Estimate g^2 ≈ Var[dx] / Δt
    g2 = var_dx / dts[..., None]
    dW = jax.random.normal(brownian_key, dts.shape) * jnp.sqrt(dts)
    dW_shape = list(dW.shape)
    dW_shape_0 = dW_shape.copy()
    dW_shape_0[-1] = 1
    W = jnp.concatenate([jnp.zeros(dW_shape_0), jnp.cumsum(dW, axis=-1)], axis=-1)
    dWt = jnp.diff(W, axis=-1) / dts
    dWt = jnp.concatenate([jnp.zeros(dW_shape_0), dWt], axis=-1)
    g2_shape = list(g2.shape)
    g2_shape_0 = g2_shape.copy()
    g2_shape_0[-2] = 1
    g2 = jnp.concatenate([jnp.zeros(g2_shape_0), g2], axis=-2)
    return dxt, g2, dWt


if __name__ == '__main__':
    # key = jr.key(42)
    # train_size = 800
    # test_size = 100
    # for sigma in [0.2, 0.3, 0.4, 0.5]:
    #     ts, xs, xs_dot, drift_dot, diff_dot, us = get_data(train_size, key=key, irregular=True, func=_get_data_sde, sigma=sigma)
    #     jnp.save(f'data/vdp_sde_irregular_sigma_{sigma}_train.npy',
    #              jnp.concatenate([ts[:, :, None], xs, xs_dot, drift_dot, diff_dot], axis=-1))
    #     ts, xs, xs_dot, drift_dot, diff_dot, us = get_data(test_size, key=key, irregular=True, func=_get_data_sde, sigma=sigma)
    #     jnp.save(f'data/vdp_sde_irregular_sigma_{sigma}_test.npy',
    #              jnp.concatenate([ts[:, :, None], xs, xs_dot, drift_dot, diff_dot], axis=-1))
    #     ts, xs, xs_dot, drift_dot, diff_dot, us = get_data(train_size, key=key, irregular=False, func=_get_data_sde, sigma=sigma)
    #     jnp.save(f'data/vdp_sde_regular_sigma_{sigma}_train.npy',
    #             jnp.concatenate([ts[:, :, None], xs, xs_dot, drift_dot, diff_dot], axis=-1))
    #     ts, xs, xs_dot, drift_dot, diff_dot, us = get_data(test_size, key=key, irregular=False, func=_get_data_sde, sigma=sigma)
    #     jnp.save(f'data/vdp_sde_regular_sigma_{sigma}_test.npy',
    #              jnp.concatenate([ts[:, :, None], xs, xs_dot, drift_dot, diff_dot], axis=-1))
    #     ts, xs, xs_dot, us = get_data(train_size, key=key, irregular=True, func=_get_data, sigma=sigma)
    #     jnp.save(f'data/vdp_ode_irregular_sigma_{sigma}_train.npy',
    #              jnp.concatenate([ts[:, :, None], xs, xs_dot], axis=-1))
    #     ts, xs, xs_dot, us = get_data(test_size, key=key, irregular=True, func=_get_data, sigma=sigma)
    #     jnp.save(f'data/vdp_ode_irregular_sigma_{sigma}_test.npy',
    #              jnp.concatenate([ts[:, :, None], xs, xs_dot], axis=-1),
    #              )
    #     ts, xs, xs_dot, us = get_data(train_size, key=key, irregular=False, func=_get_data, sigma=sigma)
    #     jnp.save(f'data/vdp_ode_regular_sigma_{sigma}_train.npy',
    #              jnp.concatenate([ts[:, :, None], xs, xs_dot], axis=-1))
    #     ts, xs, xs_dot, us = get_data(test_size, key=key, irregular=False, func=_get_data, sigma=sigma)
    #     jnp.save(f'data/vdp_ode_regular_sigma_{sigma}_test.npy',
    #              jnp.concatenate([ts[:, :, None], xs, xs_dot], axis=-1))
    # ts = jnp.linspace(0, 10, 101)
    # ts = jnp.repeat(ts[None], 32, axis=0)
    # xs = jr.normal(key, (32, 101, 2))
    # data_loader = preprocess_data(ts, xs, None, 8)

    # import math
    # import matplotlib
    # matplotlib.use("WebAgg")
    # import matplotlib.pyplot as plt
    # vdp_sde_test = jnp.load('data/vdp_sde_irregular_test.npy')
    # vdp_sde_test2 = jnp.load('data/vdp_sde_irregular_sigma_0.2_test.npy')
    # vdp_sde_test3 = jnp.load('data/vdp_sde_irregular_sigma_0.3_test.npy')
    # vdp_sde_test4 = jnp.load('data/vdp_sde_irregular_sigma_0.4_test.npy')
    # vdp_sde_test5 = jnp.load('data/vdp_sde_irregular_sigma_0.5_test.npy')
    #
    # def plot_results(ts, xs, xs_dot, drift_dot, diff_dot):
    #     x_size = xs[0].shape[-1]
    #     n_rows = int(math.sqrt(x_size * 4))
    #     if (x_size * 4) % n_rows == 0:
    #         n_columns = (x_size * 4) // n_rows
    #         rest = 0
    #     else:
    #         n_columns = (x_size * 4) // n_rows + 1
    #         rest = n_rows * n_columns - x_size * 4
    #     last_row_id = [x_size * 4 - 1 - k for k in range(n_columns - rest)] + \
    #                   [x_size * 4 - j for j in range(n_columns - rest + 1, n_columns + 1)]
    #     fig, ax = plt.subplots(n_rows, n_columns, figsize=(3 * n_columns, 3 * n_rows))
    #     if n_rows > 1:
    #         ax = ax.flat
    #     else:
    #         if n_columns == 1:
    #             ax = [ax]
    #     if rest > 0:
    #         for j in range(rest):
    #             ax[-1 - j].set_visible(False)
    #     for i in range(x_size):
    #         ax[4 * i].set_title(r"$\dot{x}$" + rf"$_{i}$")
    #         for _xs_dot in xs_dot:
    #             ax[4 * i].plot(ts, _xs_dot[:, i])
    #         ax[4 * i].legend([f'sigma_{s}' for s in [0.1, 0.2, 0.3, 0.4, 0.5]])
    #
    #         ax[4 * i + 1].set_title(r"$\dot{x}_{drift}$" + rf"$_{i}$")
    #         for _drift_dot in drift_dot:
    #             ax[4 * i + 1].plot(ts, _drift_dot[:, i])
    #         ax[4 * i + 1].legend([f'sigma_{s}' for s in [0.1, 0.2, 0.3, 0.4, 0.5]])
    #
    #         ax[4 * i + 2].set_title(r"$\dot{x}_{diff}$" + rf"$_{i}$")
    #         for _diff_dot in diff_dot:
    #             ax[4 * i + 2].plot(ts, _diff_dot[:, 0])
    #         ax[4 * i + 2].legend([f'sigma_{s}' for s in [0.1, 0.2, 0.3, 0.4, 0.5]])
    #
    #         ax[4 * i + 3].set_title(r"$x$" + rf"$_{i}$")
    #         for _xs in xs:
    #             ax[4 * i + 3].plot(ts, _xs[:, i])
    #         ax[4 * i + 3].legend([f'sigma_{s}' for s in [0.1, 0.2, 0.3, 0.4, 0.5]])
    #
    #         for k in range(4):
    #             if 4 * i + k in last_row_id:
    #                 ax[4 * i + k].set_xlabel('time')
    #             else:
    #                 ax[4 * i + k].set_xticks([])
    #
    #     plt.show()
    # idx = 1
    # plot_results(vdp_sde_test[idx, :, 0],
    #              [vdp_sde_test[idx, :, 1:3], vdp_sde_test2[idx, :, 1:3],
    #               vdp_sde_test3[idx, :, 1:3], vdp_sde_test4[idx, :, 1:3], vdp_sde_test5[idx, :, 1:3]],
    #              [vdp_sde_test[idx, :, 3:5], vdp_sde_test2[idx, :, 3:5],
    #               vdp_sde_test3[idx, :, 3:5], vdp_sde_test4[idx, :, 3:5], vdp_sde_test5[idx, :, 3:5]],
    #              [vdp_sde_test[idx, :, 5:7], vdp_sde_test2[idx, :, 5:7],
    #               vdp_sde_test3[idx, :, 5:7], vdp_sde_test4[idx, :, 5:7], vdp_sde_test5[idx, :, 5:7]],
    #              [vdp_sde_test[idx, :, 7:9], vdp_sde_test2[idx, :, 7:9],
    #               vdp_sde_test3[idx, :, 7:9], vdp_sde_test4[idx, :, 7:9], vdp_sde_test5[idx, :, 7:9]]
    #              )
    # for sigma in [0.1, 0.2, 0.3, 0.4, 0.5]:
    #     filename = f'vdp_sde_irregular_sigma_{sigma}' if sigma > 0.1 else 'vdp_sde_irregular'
    #     data = jnp.load(f'data/{filename}_train.npy')
    #     jnp.save(f'data/{filename}_train.npy', jnp.concatenate([data[:, :, :3], data[:, :, 5:8]], axis=-1))
    #     data = jnp.load(f'data/{filename}_test.npy')
    #     jnp.save(f'data/{filename}_test.npy', jnp.concatenate([data[:, :, :3], data[:, :, 5:8]], axis=-1))

    ts = jnp.linspace(0, 60, 61)
    ts = ts[None].repeat(10, axis=0)
    xs = jnp.ones((10, 61, 2))
    batch_size = 5
    dataloader = preprocess_data(ts, xs, None, 5, times=0, split=True)

