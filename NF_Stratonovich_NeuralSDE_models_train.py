import os
from numba import cuda
if cuda.is_available():
    import jax
else:
    os.environ["JAX_PLATFORMS"] = 'cpu'
    import jax
    jax.default_device('cpu')
import jax.numpy as jnp
import equinox as eqx
import jax.random as jr
import optax
import numpy as np
import wandb
import argparse
from sampling_utils import preprocess_data
from NF_Stratonovich_NeuralSDE_models import NFNeuralSDE
from argparse_utils import str2bool
from tqdm import tqdm

@eqx.filter_value_and_grad
def grad_loss_ode(model, xi, ti, ui, ms=False, dt0=1e-2, max_steps=4096, adaptive=False):
    train_loss = jax.vmap(lambda x, t, u: model.train(x, t, u, ms=ms, dt0=dt0, max_steps=max_steps, adaptive=adaptive),
                          in_axes=(0, 0, 0 if ui is not None else None))(xi, ti, ui)
    return jnp.nanmean(train_loss)
@eqx.filter_jit
def make_step_ode(xi, ti, ui, model, optim, opt_state, ms=False, dt0=1e-2, max_steps=4096, adaptive=False):
    loss, grads = grad_loss_ode(model, xi, ti, ui, ms=ms, dt0=dt0, max_steps=max_steps, adaptive=adaptive)
    grads = jax.tree_util.tree_map(lambda g: jnp.clip(g, -100, 100), grads)
    updates, opt_state = optim.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    return loss, model, opt_state

@eqx.filter_value_and_grad
def grad_loss_diff(model_diff, static_model, xi, ti, ui, key, ms=False, lamb=1.0, dt0=1e-2, max_steps=4096, adaptive=False):
    model = eqx.tree_at(lambda m: m.diff_func, static_model, model_diff)
    train_loss = jax.vmap(lambda t, x, u, k: model.train_integral(t, x, u, k, ms=ms, lamb=lamb, dt0=dt0,
                                                                  max_steps=max_steps, adaptive=adaptive),
                          in_axes=(0, 0, 0 if ui is not None else None, 0))(
        ti, xi, ui, jr.split(key, xi.shape[0]))
    return jnp.nanmean(train_loss)
@eqx.filter_jit
def make_step_diff(xi, ti, ui, key, model, model_diff, optim, opt_state, ms=False, lamb=0, dt0=1e-2, max_steps=4096, adaptive=False):
    static_model = eqx.tree_at(lambda m: m.diff_func, model, replace=None)
    loss, grads = grad_loss_diff(model_diff, static_model, xi, ti, ui, key, ms=ms, lamb=lamb, dt0=dt0, max_steps=max_steps, adaptive=adaptive)
    updates, opt_state = optim.update(grads, opt_state, model_diff)
    model_diff = eqx.apply_updates(model_diff, updates)
    model = eqx.tree_at(lambda m: m.diff_func, static_model, model_diff)
    return loss, model, model_diff, opt_state


@eqx.filter_value_and_grad
def grad_loss(model, ti, xi, ui, key, ms=False, lamb=0, dt0=1e-2, max_steps=4096, adaptive=False):
    train_loss = jax.vmap(lambda t, x, u, k: model.train_integral(t, x, u, k, ms=ms, lamb=lamb, dt0=dt0,
                                                                  max_steps=max_steps, adaptive=adaptive),
                          in_axes=(0, 0, 0 if ui is not None else None, 0))(
        ti, xi, ui, jr.split(key, xi.shape[0]))
    return jnp.nanmean(train_loss)

@eqx.filter_jit
def make_step(model, opt_state, optim, ti, xi, ui, key, ms=False, lamb=0, dt0=1e-2,
              max_steps=4096, adaptive=False):
    loss, grads = grad_loss(model, ti, xi, ui, key, ms=ms, lamb=lamb, dt0=dt0, max_steps=max_steps, adaptive=adaptive)
    grads = jax.tree_util.tree_map(lambda g: jnp.clip(g, -100, 100), grads)
    updates, opt_state = optim.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    return loss, model, opt_state

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NF_Stratonovich_NeuralSDE")
    # sort by type
    parser.add_argument('--train_size', type=int, default=2000, help='train data size')
    parser.add_argument('--batch_size', type=int, default=50, help='train batch size')
    parser.add_argument('--hidden_size_drift', type=int, default=64, help='hidden size of drift')
    parser.add_argument('--depth_drift', type=int, default=2, help='depth of drift')
    parser.add_argument('--hidden_size_diff', type=int, default=64, help='hidden size of realnvp')
    parser.add_argument('--depth_diff', type=int, default=2, help='depth of realnvp')
    parser.add_argument('--seed', type=int, default=5678, help='seed of key')
    parser.add_argument('--n_blocks', type=int, default=2, help='num of realnvp')
    parser.add_argument('--n_epochs', type=int, default=10, help='num of train epochs')
    parser.add_argument('--n_count', type=int, default=2, help='num of epochs of early stopping')
    parser.add_argument('--patch_size', type=int, default=10, help='for multiple shooting')
    parser.add_argument('--length_size', type=int, default=0, help='length size of sequence')
    parser.add_argument('--x_size', type=int, default=2, help='state size')
    parser.add_argument('--u_size', type=int, default=0, help='input size')

    parser.add_argument('--ode_retrain', type=str2bool, default=False)
    parser.add_argument('--batch_norm', type=str2bool, default=True, help='batch norm between realnvp')
    parser.add_argument('--batch_norm_last', type=str2bool, default=False, help='batch norm after all realnvp')
    parser.add_argument('--ft', type=str2bool, default=False, help='take t as input in drfit func')
    parser.add_argument('--use_wandb', type=str2bool, default=False)
    parser.add_argument('--use_lr_scheduler', type=str2bool, default=False)
    parser.add_argument('--use_normalization', type=str2bool, default=True)
    parser.add_argument('--adaptive', type=str2bool, default=False)
    parser.add_argument('--trans_state', type=str2bool, default=False)
    parser.add_argument('--partial', type=str2bool, default=False)

    parser.add_argument('--ode_model', type=str, default='NeuralODE')
    parser.add_argument('--solver_type', type=str, default='dtdw', choices=['dtdw', 'dw', ''])
    parser.add_argument('--solver_name', type=str, default='reversible_heun',
                        choices=['euler', 'euler_heun', 'reversible_heun', 'heun', 'midpoint', 'milstein'])
    parser.add_argument('--train_mode', type=str, default='integral_pretrain',
                        choices=['integral', 'integral_pretrain'])
    parser.add_argument('--data_type', type=str, default='synthetic')
    parser.add_argument('--data_set', type=str, default='van_der_pol')
    parser.add_argument('--data_folder', type=str, default='data')
    parser.add_argument('--result_folder', type=str, default='results')
    parser.add_argument('--noise_type', type=str, default='single', choices=['one', 'diag', 'all', 'single'])
    parser.add_argument('--norm_method', type=str, default='std', choices=['std', 'minmax'])

    parser.add_argument('--sigma', type=float, default=0.1)
    parser.add_argument('--scale', type=float, default=0.05)
    parser.add_argument('--dt0', type=float, default=0.005)
    parser.add_argument('--lamb', type=float, default=1.0)
    parser.add_argument('--lr', type=float, default=0.001)

    args = parser.parse_args()

    train_size = args.train_size
    batch_size = args.batch_size
    seed = args.seed
    key = jr.key(seed)
    model_key, ode_key, diff_key, train_key = jr.split(key, 4)

    if "ResNet" in args.ode_model:
        model_name = args.ode_model + '_'
    else:
        if args.ft:
            model_name = 'ft_'
        else:
            model_name = ''
    if args.batch_norm_last:
        model_name += 'batch_norm_last'
    else:
        if args.batch_norm:
            model_name += 'batch_norm'
    model_name += '_' + args.solver_type
    if args.noise_type != '':
        model_name += f'_{args.noise_type}'
    if args.trans_state:
        model_name += '_trans_state'
    if args.use_normalization:
        if args.norm_method == 'minmax':
            model_name += '_minmax'
    else:
        model_name += '_nonorm'
    train_mode = args.train_mode
    if args.partial:
        train_mode += '_partial'
    data_type = args.data_type
    if data_type == 'regular':
        data_name = f"{args.data_set}_sigma_{args.sigma}_num_{train_size}"
    elif data_type =='synthetic':
        data_name = f"{args.data_set}_{train_size}_{args.scale}"
        if args.data_set == "van_der_pol":
            data_name += f"_sigma_{args.sigma}"
    else:
        data_name = f"{args.data_set}_train"

    lamb = args.lamb
    dt0 = args.dt0
    solver = args.solver_name

    _model_name = f"{solver}_{model_name}"
    job_name = f"Stratonovich_NF_NeuralSDE_{_model_name}_{data_type}_{data_name}_{train_mode}_dt0_{dt0}_p{args.patch_size}"

    if args.use_wandb:
        wandb.init(
            # Set the project where this run will be logged
            project=f"BayesianNeuralODE",
            # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
            name=job_name
        )

    data = np.load(os.path.join(args.data_folder, f"{args.data_type}_data", f'{data_name}.npy'))
    if data_type == 'regular':
        ts = jnp.repeat(jnp.linspace(0, 10, 101)[None], axis=0, repeats=data.shape[0])
        xs = jnp.array(data)[..., :args.x_size]
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
    n = length_size1//length_size
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
        lr = optax.schedules.exponential_decay(args.lr, xs.shape[0]//batch_size, 0.9)
    else:
        lr = args.lr

    model = NFNeuralSDE(x_size, args.u_size, args.hidden_size_drift, args.hidden_size_diff, args.n_blocks,
                        args.depth_drift, args.depth_diff, key=model_key, model_name=_model_name)

    min_loss = 100
    n_count = args.n_count

    if 'pretrain' in train_mode:
        ode_name = f'{args.ode_model}'
        if args.use_normalization:
            if args.norm_method == 'minmax':
                ode_name += '_minmax'
        else:
            ode_name += '_nonorm'
        ode_name += f'_{data_type}_{data_name}_hidden_size_{args.hidden_size_drift}_depth_{args.depth_drift}'
        model_drift = model.drift_func
        if args.ode_retrain:
            steps = [0.1, 1]
            epochs_list = [1, 1]
            optim_ode = optax.adam(learning_rate=lr)
            opt_state_drift = optim_ode.init(eqx.filter(model_drift, eqx.is_inexact_array))
            for step, epochs in zip(steps, epochs_list):
                length = int(step * length_size)
                dataloader = preprocess_data(ts[:, :length], xs[:, :length], us, batch_size, times=0, step=1,
                                             split=False)
                count = 0
                for i in tqdm(range(epochs), desc="Epochs"):
                    with tqdm(total=iterations, desc=f"Epoch {i + 1}", leave=False) as pbar:
                        for batch in dataloader:
                            if us is None:
                                ti, xi = batch
                                ui = None
                            else:
                                ti, xi, ui = batch

                            loss, model_drift, opt_state_drift = make_step_ode(
                                xi, ti, ui, model_drift, optim_ode, opt_state_drift, ms=False, dt0=dt0, adaptive=True,
                                max_steps=int(jnp.nanmax(ti)/args.dt0)+1)
                            if args.use_wandb:
                                wandb.log({"ode_train_loss": loss})
                            pbar.set_postfix(loss=f"{loss:.4f}")
                            pbar.update(1)

                        if loss < min_loss:
                            min_loss = loss
                            eqx.tree_serialise_leaves(os.path.join(args.result_folder, f'{ode_name}.pt'), model_drift)
                            count = 0
                        else:
                            count += 1
                        if count > n_count:
                            break
                model = eqx.tree_at(lambda m: m.drift_func, model, model_drift)
                del dataloader
            del opt_state_drift, optim_ode
        else:
            model_drift = eqx.tree_deserialise_leaves(
                os.path.join(args.result_folder, f'{ode_name}.pt'), model_drift)
            model = eqx.tree_at(lambda m: m.drift_func, model, model_drift)

    dataloader_ms = preprocess_data(ts, xs, us, batch_size, times=0, step=0, patch=args.patch_size, split=args.patch_size > 0)
    del ts, xs
    optim = optax.adam(learning_rate=lr)
    if args.partial:
        model_diff = model.diff_func
        opt_state = optim.init(eqx.filter(model.diff_func, eqx.is_inexact_array))
    else:
        opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))
    n_count = args.n_count
    min_loss = 100
    count = 0
    for i in tqdm(range(args.n_epochs), desc="Epochs"):
        with tqdm(total=iterations, desc=f"Epoch {i + 1}", leave=False) as pbar:
            for batch in dataloader_ms:
                if us is None:
                    ti, xi = batch
                    ui = None
                else:
                    ti, xi, ui = batch
                if args.partial:
                    loss, model, model_diff, opt_state = make_step_diff(
                        xi, ti, ui, train_key, model, model_diff, optim, opt_state, ms=args.patch_size>0, lamb=lamb,
                        dt0=dt0, adaptive=args.adaptive, max_steps=int(jnp.nanmax(ti)/args.dt0)+1)
                else:
                    loss, model, opt_state = make_step(
                        model, opt_state, optim, ti, xi, ui, train_key, ms=args.patch_size>0, lamb=lamb, dt0=dt0,
                        adaptive=args.adaptive, max_steps=int(jnp.nanmax(ti)/args.dt0)+1)

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