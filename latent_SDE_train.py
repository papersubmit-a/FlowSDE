from numba import cuda
if cuda.is_available():
    import jax
else:
    import os
    os.environ["JAX_PLATFORMS"] = 'cpu'
    import jax
    jax.default_device('cpu')
import jax.numpy as jnp
import equinox as eqx
import jax.random as jr
import optax
import numpy as np
import wandb
from sampling_utils import preprocess_data
from latent_SDE import LatentSDE, LatentSDESystem
import argparse
from argparse_utils import str2bool
from tqdm import tqdm

@eqx.filter_value_and_grad
def grad_loss(model, xi, ti, ui, key, sigma=0.1, dt0=0.01, beta=1e-2, max_steps=4096, adaptive=False, ms=False):
    keys = jr.split(key, xi.shape[0])
    train_loss = jax.vmap(lambda x, t, u, k: model.train(
        x, t, u, k, sigma=sigma, dt0=dt0, beta=beta, max_steps=max_steps, adaptive=adaptive, ms=ms))(xi, ti, ui, keys)
    return jnp.mean(train_loss)

@eqx.filter_jit
def make_step(model, optim, opt_state, xi, ti, ui, key, sigma=0.1, dt0=0.01, beta=1e-2, max_steps=4096, adaptive=False, ms=False):
    loss, grads = grad_loss(model, xi, ti, ui, key, sigma=sigma, dt0=dt0, beta=beta, max_steps=max_steps, adaptive=adaptive, ms=ms)
    grads = jax.tree_util.tree_map(lambda g: jnp.clip(g, -100, 100), grads)
    updates, opt_state = optim.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    return loss, model, opt_state

@eqx.filter_value_and_grad
def grad_loss_prior(model_drift, static_model, xi, ti, ui, key, max_steps=4096, ms=False):
    model = eqx.tree_at(lambda m: m.h_net, static_model, model_drift)
    keys = jr.split(key, xi.shape[0])
    train_loss = jax.vmap(lambda x, t, u, k: model.pre_train(x, t, u, k, max_steps=max_steps, ms=ms))(xi, ti, ui, keys)
    return jnp.mean(train_loss)

@eqx.filter_jit
def make_step_prior(model_drift, optim_drift, opt_state_drift, model, xi, ti, ui, key, max_steps=4096, ms=False):
    static_model = eqx.tree_at(lambda m: m.h_net, model, replace=None)
    loss, grads = grad_loss_prior(model_drift, static_model, xi, ti, ui, key, max_steps=max_steps, ms=ms)
    grads = jax.tree_util.tree_map(lambda g: jnp.clip(g, -100, 100), grads)
    updates, opt_state_drift = optim_drift.update(grads, opt_state_drift)
    model_drift = eqx.apply_updates(model_drift, updates)
    model = eqx.tree_at(lambda m: m.h_net, static_model, model_drift)
    return loss, model_drift, opt_state_drift, model


@eqx.filter_value_and_grad
def grad_loss_diff(trainable_params, static_model, xi, ti, ui, key, sigma=0.1, dt0=0.01, beta=1e-2, max_steps=4096,
                   adaptive=False, ms=False):
    diff_params, mu_param, logvar_param = trainable_params
    model = eqx.tree_at(lambda m: m.g_nets, static_model, diff_params, is_leaf=lambda x: x is None)
    model = eqx.tree_at(lambda m: m.qz0_mu, model, mu_param, is_leaf=lambda x: x is None)
    model = eqx.tree_at(lambda m: m.qz0_logvar, model, logvar_param, is_leaf=lambda x: x is None)
    keys = jr.split(key, xi.shape[0])
    if ui is None:
        train_loss = jax.vmap(lambda x, t, k: model.train(
            x, t, ui, k, sigma=sigma, dt0=dt0, beta=beta, max_steps=max_steps, adaptive=adaptive, ms=ms))(xi, ti, keys)
        return jnp.mean(train_loss)
    else:
        train_loss = jax.vmap(lambda x, t, u, k: model.train(
            x, t, u, k, sigma=sigma, dt0=dt0, beta=beta, max_steps=max_steps, adaptive=adaptive, ms=ms))(xi, ti, ui, keys)
        return jnp.mean(train_loss)
@eqx.filter_jit
def make_step_diff(xi, ti, ui, key, model, trainable_params, optim, opt_state, sigma=0.1, dt0=0.01, beta=1e-2,
                   max_steps=4096, adaptive=False, ms=False):
    static_model = eqx.tree_at(lambda m: (m.g_nets, m.qz0_mu, m.qz0_logvar), model,
                               replace=(None, None, None), is_leaf=lambda x: x is None)
    loss, grads = grad_loss_diff(trainable_params, static_model, xi, ti, ui, key, sigma=sigma, dt0=dt0, beta=beta,
                                 max_steps=max_steps, adaptive=adaptive, ms=ms)
    updates, opt_state = optim.update(grads, opt_state, trainable_params)
    trainable_params = eqx.apply_updates(trainable_params, updates)
    diff_params, mu_param, logvar_param = trainable_params
    model = eqx.tree_at(lambda m: m.g_nets, static_model, diff_params, is_leaf=lambda x: x is None)
    model = eqx.tree_at(lambda m: m.qz0_mu, model, mu_param, is_leaf=lambda x: x is None)
    model = eqx.tree_at(lambda m: m.qz0_logvar, model, logvar_param, is_leaf=lambda x: x is None)
    return loss, model, trainable_params, opt_state


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="latentSDE")
    # sort by type
    parser.add_argument('--train_size', type=int, default=2000, help='train data size')
    parser.add_argument('--batch_size', type=int, default=50, help='train batch size')
    parser.add_argument('--hidden_size_drift', type=int, default=64, help='hidden size of NN')
    parser.add_argument('--depth_drift', type=int, default=2, help='depth of drift')
    parser.add_argument('--hidden_size_diff', type=int, default=64, help='hidden size of diff')
    parser.add_argument('--depth_diff', type=int, default=2, help='depth of diff')
    parser.add_argument('--context_dim', type=int, default=0, help='use latentsde or latentsde system')
    parser.add_argument('--obs_length', type=int, default=10, help='obs length')
    parser.add_argument('--hidden_size_decoder', type=int, default=64, help='hidden size of decoder')
    parser.add_argument('--depth_decoder', type=int, default=2, help='depth of decoder')
    parser.add_argument('--seed', type=int, default=5678, help='seed of key')
    parser.add_argument('--n_epochs', type=int, default=5, help='num of train epochs')
    parser.add_argument('--n_count', type=int, default=2, help='num of epochs of early stopping')
    parser.add_argument('--patch_size', type=int, default=0, help='for multiple shooting')
    parser.add_argument('--length_size', type=int, default=0, help='length size of sequence')
    parser.add_argument('--u_size', type=int, default=0, help='input size')
    parser.add_argument('--x_size', type=int, default=2, help='state size')

    parser.add_argument('--ft', type=str2bool, default=False, help='take t as input in drfit func')
    parser.add_argument('--use_wandb', type=str2bool, default=False)
    parser.add_argument('--use_lr_scheduler', type=str2bool, default=False)
    parser.add_argument('--use_normalization', type=str2bool, default=True)
    parser.add_argument('--adaptive', type=str2bool, default=False)
    parser.add_argument('--partial', type=str2bool, default=False)
    parser.add_argument('--ode_retrain', type=str2bool, default=False)
    parser.add_argument('--scaled', type=str2bool, default=False)

    parser.add_argument('--ode_model', type=str, default='NeuralODE')
    parser.add_argument('--diff_model', type=str, default='diag')
    parser.add_argument('--data_type', type=str, default='synthetic')
    parser.add_argument('--data_set', type=str, default='van_der_pol')
    parser.add_argument('--data_folder', type=str, default='data')
    parser.add_argument('--result_folder', type=str, default='results')
    parser.add_argument('--encoder', type=str, default='')
    parser.add_argument('--norm_method', type=str, default='std', choices=['std', 'minmax'])

    parser.add_argument('--beta', type=float, default=0.001)
    parser.add_argument('--sigma', type=float, default=0.1)
    parser.add_argument('--scale', type=float, default=0.05)
    parser.add_argument('--dt0', type=float, default=0.01)
    parser.add_argument('--lr', type=float, default=0.01)

    args = parser.parse_args()

    train_size = args.train_size
    batch_size = args.batch_size
    seed = args.seed
    key = jr.key(seed)
    model_key, train_key = jr.split(key, 2)
    model_name = str(args.dt0)
    if "ResNet" in args.ode_model:
        model_name += '_' + args.ode_model
    if 'diag' not in args.diff_model:
        model_name += '_' + args.diff_model
    if args.ft:
        model_name += '_ft'
    if args.partial:
        model_name += '_partial'
    if args.use_normalization:
        if args.norm_method == 'minmax':
            model_name += '_minmax'
    else:
        model_name += '_nonorm'
    if args.scaled:
        model_name += '_scaled'
    data_type = args.data_type
    if data_type == 'regular':
        sigma = args.sigma
        data_name = f"{args.data_set}_sigma_{sigma}_num_{train_size}"
    elif data_type == 'synthetic':
        data_name = f"{args.data_set}_{train_size}_{args.scale}"
        if args.data_set == "double_well" or args.data_set == "van_der_pol":
            data_name += f"_{args.sigma}"
    else:
        data_name = f"{args.data_set}_train"

    job_name = f"LatentSDE_milstein_{data_type}_{data_name}_{model_name}_hdr_{args.hidden_size_drift}_ddr_{args.depth_drift}_hdf_{args.hidden_size_diff}_ddf_{args.depth_diff}_p{args.patch_size}"

    if args.use_wandb:
        wandb.init(
            # Set the project where this run will be logged
            project=f"LatentSDE",
            # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
            name=job_name
        )

    data = np.load(os.path.join(args.data_folder, f"{args.data_type}_data", f'{data_name}.npy'))
    if data_type == 'regular':
        ts = jnp.repeat(jnp.linspace(0, 10, 101)[None], axis=0, repeats=data.shape[0])
        xs = jnp.array(data)
        us = None
    elif data_type == 'synthetic':
        ts = data[..., 0]
        xs = data[..., 1:1+args.x_size]
        us = None
    else:
        ts = data[..., 0]
        if args.u_size > 0:
            xs = data[..., 1:1+args.x_size]
            us = data[..., -args.u_size:]
        else:
            xs = data[..., 1:1+args.x_size]
            us = None
    _, length_size1, x_size = xs.shape
    length_size = args.length_size if args.length_size > 0 else length_size1
    n = length_size1 // length_size
    if length_size1 > length_size * 1.1:
        ts_list = [ts[:, i*length_size:(i+1)*length_size] for i in range(n)]
        xs_list = [xs[:, i*length_size:(i+1)*length_size] for i in range(n)]
        ts = jnp.concatenate(ts_list)
        xs = jnp.concatenate(xs_list)
        del ts_list, xs_list
    del data

    iterations = xs.shape[0] // args.batch_size

    if args.use_normalization:
        if args.norm_method == 'std':
            mean_y = jnp.mean(xs, axis=(0, 1), keepdims=True)
            std_y = jnp.std(xs, axis=(0, 1), keepdims=True)
            xs = (xs - mean_y) / std_y
        else:
            min_y = jnp.min(jnp.abs(xs), axis=(0, 1), keepdims=True)
            max_y = jnp.max(jnp.abs(xs), axis=(0, 1), keepdims=True)
            xs = xs / (max_y - min_y) * 2

    if args.use_lr_scheduler:
        lr = optax.schedules.exponential_decay(args.lr, xs.shape[0]//batch_size*2, 0.9)
    else:
        lr = args.lr

    if args.context_dim > 0:
        model = LatentSDESystem(x_size, args.obs_length, args.latent_dim, args.context_dim, args.hidden_dim_drift,
                                args.depth_drift, args.hidden_dim_diff, args.depth_diff, args.hidden_dim_decoder,
                                args.depth_decoder, key=model_key, ft=args.ft, u_size=args.u_size,
                                encoder='' if args.obs_length == 1 else 'MLP', model_name=model_name)
    else:
        model = LatentSDE(x_size, args.hidden_size_drift, args.depth_drift, args.hidden_size_diff, args.depth_diff,
                          key=model_key, ft=args.ft, context_dim=0, u_size=args.u_size, model_name=model_name)

    def linear_scheduler(step, start=1e-4, end=1.0, duration=1000):
        step = min(step, duration)
        coeff = start + (end - start) * (step / duration)
        return coeff

    steps = [1.0]
    epochs_list = [args.n_epochs]

    min_loss = 100
    n_count = args.n_count

    model_drift = model.h_net

    if args.ode_retrain:
        # train drfit
        optim_drift = optax.adam(learning_rate=5e-3)

        opt_state_drift = optim_drift.init(eqx.filter(model_drift, eqx.is_inexact_array))
        for step, epochs in zip(steps, epochs_list):
            length = int(step * length_size)
            if args.patch_size == 0:
                _dataloader = preprocess_data(ts[:, :length], xs[:, :length], None if us is None else us[:, :length],
                                              batch_size, times=0, step=1, split=False)
            else:
                _dataloader = preprocess_data(ts[:, :length], xs[:, :length], None if us is None else us[:, :length],
                                              batch_size, times=0, step=0, patch=args.patch_size,  split=True)
            count = 0
            for i in tqdm(range(epochs), desc="Epochs"):
                with tqdm(total=iterations, desc=f"Epoch {i + 1}", leave=False) as pbar:
                    for batch in _dataloader:
                        if us is None:
                            ti, xi = batch
                            ui = None
                        else:
                            ti, xi, ui = batch

                        loss, model_drift, opt_state_drift, model = make_step_prior(
                            model_drift, optim_drift, opt_state_drift, model, xi, ti, ui, key, max_steps=4096, ms=False)
                        if args.use_wandb:
                            wandb.log({"ode_loss": loss})
                        pbar.set_postfix(loss=f"{loss:.4f}")
                        pbar.update(1)

                    if loss < min_loss:
                        min_loss = loss
                        eqx.tree_serialise_leaves(os.path.join(args.result_folder, f'{job_name}.pt'), model)
                        count = 0
                    else:
                        count += 1
                    if count > n_count:
                        break
    else:
        ode_name = f'{args.ode_model}'
        if args.use_normalization:
            if args.norm_method == 'minmax':
                ode_name += '_minmax'
        else:
            ode_name += '_nonorm'
        ode_name += f'_{data_type}_{data_name}_hidden_size_{args.hidden_size_drift}_depth_{args.depth_drift}'
        model_drift = eqx.tree_deserialise_leaves(
            os.path.join(args.result_folder, f'{ode_name}.pt'), model_drift)
    model = eqx.tree_at(lambda m: m.h_net, model, model_drift)
    model = eqx.tree_at(lambda m: m.f_net, model, model_drift)
    # train g
    optim_diff = optax.adam(learning_rate=lr)
    if args.partial:
        diff_params = model.g_nets
        mu_param = model.qz0_mu
        logvar_param = model.qz0_logvar
        # Combine into a single pytree of parameters
        trainable_params = (diff_params, mu_param, logvar_param)
        opt_state = optim_diff.init(eqx.filter(trainable_params, eqx.is_inexact_array))
    else:
        opt_state = optim_diff.init(eqx.filter(model, eqx.is_inexact_array))

    for step, epochs in zip(steps, epochs_list):
        length = int(step * length_size)
        if args.patch_size == 0:
            _dataloader = preprocess_data(ts[:, :length], xs[:, :length], None if us is None else us[:, :length],
                                          batch_size, times=0, step=1, split=False)
        else:
            _dataloader = preprocess_data(ts[:, :length], xs[:, :length], None if us is None else us[:, :length],
                                          batch_size, times=0, step=0, patch=args.patch_size, split=True)
        count = 0
        for i in tqdm(range(epochs), desc="Epochs"):
            with tqdm(total=iterations, desc=f"Epoch {i + 1}", leave=False) as pbar:
                for batch in _dataloader:
                    if us is None:
                        ti, xi = batch
                        ui = None
                    else:
                        ti, xi, ui = batch

                    if args.partial:
                        loss, model, trainable_params, opt_state = make_step_diff(
                            xi, ti, ui, train_key, model, trainable_params, optim_diff, opt_state,
                            sigma=args.sigma, dt0=args.dt0, beta=linear_scheduler(i, start=args.beta, duration=epochs),
                            adaptive=args.adaptive, max_steps=int(ti.max() / args.dt0) + 10, ms=args.patch_size > 0)
                    else:
                        loss, model, opt_state = make_step(
                            model, optim_diff, opt_state, xi, ti, ui, train_key, sigma=args.sigma, dt0=args.dt0,
                            beta=linear_scheduler(i, start=args.beta, duration=epochs),
                            adaptive=args.adaptive, max_steps=int(ti.max()/args.dt0)+10, ms=args.patch_size>0)
                    if args.use_wandb:
                        wandb.log({"train_loss": loss})
                    pbar.set_postfix(loss=f"{loss:.4f}")
                    pbar.update(1)

                if loss < min_loss:
                    min_loss = loss
                    eqx.tree_serialise_leaves(os.path.join(args.result_folder, f'{job_name}.pt'), model)
                    count = 0
                else:
                    count += 1
                if count > n_count:
                    break