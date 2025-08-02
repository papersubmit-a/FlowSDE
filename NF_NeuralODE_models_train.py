from numba import cuda
import os
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
from sampling_utils import preprocess_data
from NF_NeuralODE_models import NeuralODE
import argparse
from argparse_utils import str2bool
from tqdm import tqdm

@eqx.filter_value_and_grad
def grad_loss(model, xi, ti, ui, ms=False, dt0=1e-2, adaptive=False, max_steps=4096):
    train_loss = jax.vmap(lambda x, t, u: model.train(x, t, u, ms=ms, dt0=dt0, adaptive=adaptive, max_steps=max_steps),
                          in_axes=(0, 0, 0 if ui is not None else None))(xi, ti, ui)
    return jnp.nanmean(train_loss)


@eqx.filter_jit
def make_step(xi, ti, ui, model, opt_state, ms=False, dt0=1e-2, adaptive=False, max_steps=4096):
    loss, grads = grad_loss(model, xi, ti, ui, ms=ms, dt0=dt0, adaptive=adaptive, max_steps=max_steps)
    grads = jax.tree_util.tree_map(lambda g: jnp.clip(g, -100, 100), grads)
    updates, opt_state = optim.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    return loss, model, opt_state


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NeuralODE")
    # sort by type
    parser.add_argument('--train_size', type=int, default=2000, help='train data size')
    parser.add_argument('--batch_size', type=int, default=50, help='train batch size')
    parser.add_argument('--hidden_size', type=int, default=64, help='hidden size of NN')
    parser.add_argument('--depth', type=int, default=2, help='depth of drift')
    parser.add_argument('--seed', type=int, default=5678, help='seed of key')
    parser.add_argument('--n_epochs', type=int, default=20, help='num of train epochs')
    parser.add_argument('--n_count', type=int, default=10, help='num of train epochs')
    parser.add_argument('--length_size', type=int, default=0, help='length size of sequence')
    parser.add_argument('--u_size', type=int, default=0, help='input size')

    parser.add_argument('--ft', type=str2bool, default=False, help='take t as input in drfit func')
    parser.add_argument('--retrain', type=str2bool, default=False)
    parser.add_argument('--use_wandb', type=str2bool, default=False)
    parser.add_argument('--use_lr_scheduler', type=str2bool, default=False)
    parser.add_argument('--use_normalization', type=str2bool, default=True)
    parser.add_argument('--adaptive', type=str2bool, default=True)

    parser.add_argument('--data_type', type=str, default='synthetic')
    parser.add_argument('--data_set', type=str, default='van_der_pol')
    parser.add_argument('--data_folder', type=str, default='data')
    parser.add_argument('--result_folder', type=str, default='results')
    parser.add_argument('--model_name', type=str, default='NeuralODE')
    parser.add_argument('--norm_method', type=str, default='std', choices=['std', 'minmax'])

    parser.add_argument('--sigma', type=float, default=0.1)
    parser.add_argument('--scale', type=float, default=0.05)
    parser.add_argument('--dt0', type=float, default=0.005)
    parser.add_argument('--lr', type=float, default=0.005)

    args = parser.parse_args()

    train_size = args.train_size
    batch_size = args.batch_size
    hidden_size = args.hidden_size
    depth = args.depth
    seed = args.seed
    key = jr.key(seed)
    model_key, train_key = jr.split(key, 2)
    model_name = args.model_name
    if args.ft:
        model_name += '_ft'
    if args.use_normalization:
        if args.norm_method == 'minmax':
            model_name += '_minmax'
    else:
        model_name += '_nonorm'

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

    job_name = f"{model_name}_{data_type}_{data_name}_hidden_size_{hidden_size}_depth_{depth}"

    if args.use_wandb:
        wandb.init(
            # Set the project where this run will be logged
            project=f"NeuralODE",
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
        xs = data[..., 1:]
        us = None
    else:
        ts = data[..., 0]
        if args.u_size > 0:
            xs = data[..., 1:-args.u_size]
            us = data[..., -args.u_size:]
        else:
            xs = data[..., 1:]
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
        if us is not None:
            us_list = [us[:, i*length_size:(i+1)*length_size] for i in range(n)]
            us = jnp.concatenate(us_list)
            del us_list
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
    optim = optax.adam(learning_rate=lr)

    model = NeuralODE(x_size, x_size, args.u_size, hidden_size, depth, key=model_key, model_name=model_name)

    if args.retrain:
        model = eqx.tree_deserialise_leaves(
                os.path.join(args.result_folder, f'{job_name}.pt'), model)

    opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))

    if args.data_type == 'synthetic':
        if args.data_set == "lotka_volterra":
            steps = [0.1, 0.4, 0.6, 0.8, 1]
            epochs_list = [5, 10, 10, 10, args.n_epochs]
        elif args.data_set == 'van_der_pol':
            steps = [0.1, 0.4, 0.7, 1]
            epochs_list = [1, 10, args.n_epochs, args.n_epochs]
        else:
            steps = [0.1, 0.2, 1.0]
            epochs_list = [1, 10, args.n_epochs]
    else:
        if "acrobot" in args.data_set:
            steps = [0.1, 0.5, 0.8, 1]
            epochs_list = [1, 10, 10, args.n_epochs]
        else:
            steps = [0.1, 0.5, 1]
            epochs_list = [1, 10, args.n_epochs]

    min_loss = 100

    for step, epochs in zip(steps, epochs_list):
        length = int(step * length_size)
        dataloader = preprocess_data(ts[:, :length], xs[:, :length], us[:, :length] if us is not None else None,
                                     batch_size, times=0, step=1, split=False)
        count = 0
        for i in tqdm(range(epochs), desc="Epochs"):
            with tqdm(total=iterations, desc=f"Epoch {i+1}", leave=False) as pbar:
                for batch in dataloader:
                    if us is None:
                        ti, xi = batch
                        ui = None
                    else:
                        ti, xi, ui = batch

                    loss, model, opt_state = make_step(xi, ti, ui, model, opt_state, ms=False, dt0=args.dt0,
                                                       adaptive=args.adaptive)
                    if args.use_wandb:
                        wandb.log({"ode_train_loss": loss})
                    pbar.set_postfix(loss=f"{loss:.4f}")
                    pbar.update(1)
                if loss < min_loss:
                    min_loss = loss
                    eqx.tree_serialise_leaves(os.path.join(args.result_folder, f'{job_name}.pt'), model)
                    count = 0
                else:
                    count += 1
                if count > args.n_count:
                    break

        del dataloader

    eqx.tree_serialise_leaves(os.path.join(args.result_folder, f'{job_name}.pt'), model)