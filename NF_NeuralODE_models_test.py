from evaluation_utils import plot_results_ode
from tqdm import tqdm
import os
import jax
import jax.numpy as jnp
import equinox as eqx
import jax.random as jr
import numpy as np
import jax_dataloader as jdl
from NF_NeuralODE_models import NeuralODE
import argparse
from argparse_utils import str2bool

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NeuralODE")
    # sort by type
    parser.add_argument('--train_size', type=int, default=2000, help='train data size')
    parser.add_argument('--test_size', type=int, default=200, help='test data size')
    parser.add_argument('--batch_size_test', type=int, default=4, help='train batch size')
    parser.add_argument('--hidden_size', type=int, default=64, help='hidden size of NN')
    parser.add_argument('--depth', type=int, default=3, help='depth of drift')
    parser.add_argument('--seed', type=int, default=5678, help='seed of key')
    parser.add_argument('--u_size', type=int, default=0, help='input size')

    parser.add_argument('--adaptive', type=str2bool, default=True)
    parser.add_argument('--use_normalization', type=str2bool, default=True)
    parser.add_argument('--ft', type=str2bool, default=False)

    parser.add_argument('--data_type', type=str, default='real_world')
    parser.add_argument('--data_set', type=str, default='discharge_battery')
    parser.add_argument('--data_folder', type=str, default='data')
    parser.add_argument('--result_folder', type=str, default='results')
    parser.add_argument('--model_name', type=str, default='NeuralODE')
    parser.add_argument('--norm_method', type=str, default='std', choices=['std', 'minmax'])

    parser.add_argument('--sigma', type=float, default=0.1)
    parser.add_argument('--scale', type=float, default=0.05)
    parser.add_argument('--dt0', type=float, default=0.01)

    args = parser.parse_args()

    train_size = args.train_size
    test_size = args.test_size
    batch_size_test = args.batch_size_test
    hidden_size = args.hidden_size
    depth = args.depth
    seed = args.seed
    model_key = jr.key(seed)
    model_name = args.model_name
    if args.ft:
        model_name += '_ft'
    if args.use_normalization:
        if args.norm_method == 'minmax':
            model_name += '_minmax'
    else:
        model_name += '_nonorm'

    if args.data_type == 'regular':
        sigma = args.sigma
        data_name_train = f"{args.data_set}_sigma_{sigma}_num_{train_size}"
        data_name = f"{args.data_set}_sigma_{sigma}_num_{test_size}"
    elif args.data_type == 'synthetic':
        data_name_train = f"{args.data_set}_{train_size}_{args.scale}"
        if args.data_set == "double_well" or args.data_set == "van_der_pol":
            data_name_train += f"_{args.sigma}"
        data_name = f"{args.data_set}_{test_size}_{args.scale}"
        if args.data_set == "double_well" or args.data_set == "van_der_pol":
            data_name += f"_{args.sigma}"
    else:
        data_name_train = f"{args.data_set}_train"
        data_name = f"{args.data_set}_test"

    pt_name = f"{model_name}_{args.data_type}_{data_name_train}_hidden_size_{hidden_size}_depth_{depth}"
    job_name = f"{model_name}_{args.data_type}_{data_name}_hidden_size_{hidden_size}_depth_{depth}"

    data = np.load(os.path.join(args.data_folder, f"{args.data_type}_data", f'{data_name}.npy'))
    data_train = np.load(os.path.join(args.data_folder, f"{args.data_type}_data", f'{data_name_train}.npy'))
    if args.data_type == 'regular':
        xs_test = jnp.array(data)
        xs_train = jnp.array(data_train)
        ts_test = jnp.repeat(jnp.linspace(0, 10, 101)[None], axis=0, repeats=xs_test.shape[0])
        us_test = None
    elif args.data_type == 'synthetic':
        ts_test = data[..., 0]
        xs_test = data[..., 1:]
        xs_train = data_train[..., 1:]
        us_test = None
    else:
        ts_test = data[..., 0]
        if args.u_size > 0:
            xs_test = data[..., 1:-args.u_size]
            xs_train = data_train[..., 1:-args.u_size]
            us_test = data[..., -args.u_size:]
            us_train = data_train[..., -args.u_size:]
        else:
            xs_test = data[..., 1:]
            xs_train = data_train[..., 1:]
            us_test = None
            us_train = None

    x_size = xs_test.shape[-1]

    if args.use_normalization:
        if args.norm_method == 'std':
            mean_y = jnp.mean(xs_train, axis=(0, 1), keepdims=True)
            std_y = jnp.std(xs_train, axis=(0, 1), keepdims=True)
            xs_test = (xs_test - mean_y) / std_y
        else:
            min_y = jnp.min(jnp.abs(xs_train), axis=(0, 1), keepdims=True)
            max_y = jnp.max(jnp.abs(xs_train), axis=(0, 1), keepdims=True)
            xs_test = xs_test / (max_y - min_y) * 2
    dxt_test = jax.vmap(lambda x, t: jnp.gradient(x, t, axis=0))(xs_test, ts_test)
    _dataset = jdl.ArrayDataset(ts_test, xs_test) if us_test is None else jdl.ArrayDataset(
        ts_test, xs_test, us_test)
    dataloader_test = jdl.DataLoader(_dataset, backend='jax', batch_size=batch_size_test,
                                     shuffle=False, drop_last=False)

    model = NeuralODE(x_size, x_size, args.u_size, hidden_size, depth, key=model_key, model_name=model_name)
    model = eqx.tree_deserialise_leaves(os.path.join(args.result_folder, f'{pt_name}.pt'), model)

    drift_est_test = []
    xs_pred_test = []
    drift_pred_test = []
    pbar = tqdm(total=xs_test.shape[0] // batch_size_test)
    for batch in dataloader_test:
        pbar.update(1)
        if us_test is None:
            _ts, _xs, = batch
            _us = None
        else:
            _ts, _xs, _us = batch
        _drift_est_test = model.test(_xs, _ts, _us)
        drift_est_test.append(_drift_est_test)
        _xs_pred_test = jax.vmap(lambda x0, ts, us: model.generator(
            x0, ts, us, dt0=args.dt0, adaptive=args.adaptive))(_xs[:, 0], _ts, _us)
        xs_pred_test.append(_xs_pred_test)
        _drift_pred_test = model.test(_xs_pred_test, _ts, _us)
        drift_pred_test.append(_drift_pred_test)
    drift_est_test = jnp.concatenate(drift_est_test, axis=0)
    xs_pred_test = jnp.concatenate(xs_pred_test, axis=0)
    drift_pred_test = jnp.concatenate(drift_pred_test, axis=0)

    if args.use_normalization:
        if args.norm_method == 'std':
            xs_test = xs_test * std_y + mean_y
            xs_pred_test = xs_pred_test * std_y + mean_y
        else:
            xs_test = xs_test * (max_y - min_y) / 2
            xs_pred_test = xs_pred_test * (max_y - min_y) / 2

    if not os.path.isdir(f"figures/{job_name}"):
        os.mkdir(f"figures/{job_name}")
    for test_id in range(xs_test.shape[0]):
        plot_results_ode(ts_test[test_id], dxt_test[test_id], drift_est_test[test_id], drift_pred_test[test_id],
                         xs_test[test_id], xs_pred_test[test_id],
                         filename=os.path.join("figures", f"{job_name}", f"{job_name}_{test_id}.png"))