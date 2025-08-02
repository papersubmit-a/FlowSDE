from numba import cuda
import os
if cuda.is_available():
    import jax
else:
    os.environ["JAX_PLATFORMS"] = 'cpu'
    import jax
    jax.default_device('cpu')
import argparse
import jax.numpy as jnp
import equinox as eqx
import jax.random as jr
import numpy as np
import jax_dataloader as jdl
from latent_SDE import LatentSDE, LatentSDESystem
import time
from tqdm import tqdm
from argparse_utils import str2bool

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="latentSDE")
    # sort by type
    parser.add_argument('--train_size', type=int, default=2000, help='train data size')
    parser.add_argument('--test_size', type=int, default=200, help='test data size')
    parser.add_argument('--batch_size_test', type=int, default=4, help='batch size test')
    parser.add_argument('--hidden_size_drift', type=int, default=128, help='hidden size of NN')
    parser.add_argument('--depth_drift', type=int, default=3, help='depth of drift')
    parser.add_argument('--hidden_size_diff', type=int, default=64, help='hidden size of diff')
    parser.add_argument('--depth_diff', type=int, default=2, help='depth of diff')
    parser.add_argument('--context_dim', type=int, default=0, help='use latentsde or latentsde system')
    parser.add_argument('--obs_length', type=int, default=10, help='obs length')
    parser.add_argument('--hidden_size_decoder', type=int, default=64, help='hidden size of decoder')
    parser.add_argument('--depth_decoder', type=int, default=2, help='depth of decoder')
    parser.add_argument('--seed', type=int, default=5678, help='seed of key')
    parser.add_argument('--patch_size', type=int, default=10, help='for multiple shooting')
    parser.add_argument('--n_samples', type=int, default=50, help='num of samples')
    parser.add_argument('--u_size', type=int, default=1, help='input size')
    parser.add_argument('--x_size', type=int, default=4, help='state size')

    parser.add_argument('--ft', type=str2bool, default=False, help='take t as input in drfit func')
    parser.add_argument('--use_wandb', type=str2bool, default=False)
    parser.add_argument('--use_normalization', type=str2bool, default=True)
    parser.add_argument('--calculation', type=str2bool, default=False)
    parser.add_argument('--eval_and_plot', type=str2bool, default=True)
    parser.add_argument('--adaptive', type=str2bool, default=False)
    parser.add_argument('--partial', type=str2bool, default=True)
    parser.add_argument('--scaled', type=str2bool, default=False)

    parser.add_argument('--data_type', type=str, default='real_world')
    parser.add_argument('--data_set', type=str, default='cartpole_noisy')
    parser.add_argument('--data_folder', type=str, default='data')
    parser.add_argument('--result_folder', type=str, default='results')
    parser.add_argument('--figure_folder', type=str, default='figures')
    parser.add_argument('--encoder', type=str, default='')
    parser.add_argument('--ode_model', type=str, default='NeuralODE')
    parser.add_argument('--diff_model', type=str, default='diag')
    parser.add_argument('--norm_method', type=str, default='minmax', choices=['std', 'minmax'])

    parser.add_argument('--beta', type=float, default=0.001)
    parser.add_argument('--sigma', type=float, default=0.1)
    parser.add_argument('--scale', type=float, default=0.05)
    parser.add_argument('--dt0', type=float, default=0.005)

    args = parser.parse_args()

    train_size = args.train_size
    test_size = args.test_size
    batch_size_test = args.batch_size_test
    seed = args.seed
    key = jr.key(seed)
    model_key, evaluate_key, sample_key = jr.split(key, 3)

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

    if args.data_type == 'regular':
        sigma = args.sigma
        data_name_train = f"{args.data_set}_sigma_{sigma}_num_{train_size}"
        data_name = f"{args.data_set}_sigma_{sigma}_num_{test_size}"
    elif args.data_type == 'synthetic':
        data_name_train = f"{args.data_set}_{train_size}_{args.scale}"
        data_name = f"{args.data_set}_{test_size}_{args.scale}"
        if args.data_set == "double_well" or args.data_set == "van_der_pol":
            data_name_train += f"_sigma_{args.sigma}"
            data_name += f"_sigma_{args.sigma}"
    else:
        data_name_train = f"{args.data_set}_train"
        data_name = f"{args.data_set}_test"

    pt_name = f"LatentSDE_milstein_{args.data_type}_{data_name_train}_{model_name}_hdr_{args.hidden_size_drift}_ddr_{args.depth_drift}_hdf_{args.hidden_size_diff}_ddf_{args.depth_diff}_p{args.patch_size}"
    job_name = f"LatentSDE_milstein_{args.data_type}_{data_name}_{model_name}_hdr_{args.hidden_size_drift}_ddr_{args.depth_drift}_hdf_{args.hidden_size_diff}_ddf_{args.depth_diff}_p{args.patch_size}"

    data_train = np.load(os.path.join(args.data_folder, f"{args.data_type}_data", f"{data_name_train}.npy"))
    data = np.load(os.path.join(args.data_folder, f"{args.data_type}_data", f"{data_name}.npy"))
    if args.data_type == 'regular':
        xs_test = jnp.array(data)
        xs_train = jnp.array(data_train)
        ts_test = jnp.repeat(jnp.linspace(0, 10, 101)[None], axis=0, repeats=xs_test.shape[0])
        us_test = None
        dxs_test = None
    elif args.data_type == 'synthetic':
        ts_test = data[..., 0]
        xs_test = data[..., 1:1+args.x_size]
        dxs_test = data[..., 1+args.x_size:1+3*args.x_size]
        xs_train = data_train[..., 1:1+args.x_size]
        us_test = None
    else:
        dxs_test = None
        ts_test = data[..., 0]
        if args.u_size > 0:
            xs_test = data[..., 1:1+args.x_size]
            xs_train = data_train[..., 1:1+args.x_size]
            us_test = data[..., -args.u_size:]
            us_train = data[..., -args.u_size:]
        else:
            xs_test = data[..., 1:1+args.x_size]
            xs_train = data_train[..., 1:1+args.x_size]
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

    if args.context_dim > 0:
        model = LatentSDESystem(x_size, args.obs_length, args.latent_dim, args.context_dim, args.hidden_dim_drift,
                                args.depth_drift, args.hidden_dim_diff, args.depth_diff, args.hidden_dim_decoder,
                                args.depth_decoder, key=model_key, ft=args.ft, u_size=args.u_size,
                                encoder='' if args.obs_length == 1 else 'MLP', model_name=model_name)
    else:
        model = LatentSDE(x_size, args.hidden_size_drift, args.depth_drift, args.hidden_size_diff, args.depth_diff,
                          key=model_key, ft=args.ft, context_dim=0, u_size=args.u_size, model_name=model_name)

    model = eqx.tree_deserialise_leaves(os.path.join(args.result_folder, f'{pt_name}.pt'), model)

    if args.calculation:
        if us_test is None:
            _dataset = jdl.ArrayDataset(ts_test, xs_test)
        else:
            _dataset = jdl.ArrayDataset(ts_test, xs_test, us_test)
        dataloader_test = jdl.DataLoader(_dataset, backend='jax', batch_size=args.batch_size_test, shuffle=False,
                                         drop_last=False)

        xs_pred_sample_test = []
        dfs_pred_sample_test = []
        dgs_pred_sample_test = []
        cal_times = []

        pbar = tqdm(total=xs_test.shape[0] // batch_size_test)
        for batch in dataloader_test:
            pbar.update(1)
            if us_test is None:
                _ts, _xs = batch
                _us = None
            else:
                _ts, _xs, _us = batch
            start = time.time()
            _xs_pred_sample_test = model.sample(_xs[:, 0], _ts, _us, jr.split(sample_key, args.n_samples), dt0=args.dt0)
            cal_time = time.time() - start
            cal_times.append(cal_time)
            xs_pred_sample_test.append(_xs_pred_sample_test)
            _dfs, _dgs = model.sample_derivative(_xs, _ts, _us, args.dt0, jr.split(sample_key, args.n_samples))
            dfs_pred_sample_test.append(_dfs)
            dgs_pred_sample_test.append(_dgs)
        xs_pred_sample_test = jnp.concatenate(xs_pred_sample_test, axis=1)
        dfs_pred_sample_test = jnp.concatenate(dfs_pred_sample_test, axis=1)
        dgs_pred_sample_test = jnp.concatenate(dgs_pred_sample_test, axis=1)
        if args.use_normalization:
            if args.norm_method == 'std':
                xs_pred_sample_test = xs_pred_sample_test * std_y[None] + mean_y[None]
                dfs_pred_sample_test = dfs_pred_sample_test * std_y[None] + mean_y[None]
                dgs_pred_sample_test = dgs_pred_sample_test * std_y[None] + mean_y[None]
            else:
                xs_pred_sample_test = xs_pred_sample_test * (max_y[None] - min_y[None]) / 2
                dfs_pred_sample_test = dfs_pred_sample_test * (max_y[None] - min_y[None]) / 2
                dgs_pred_sample_test = dgs_pred_sample_test * (max_y[None] - min_y[None]) / 2
        np.save(os.path.join(args.result_folder, f"{job_name}_xs_pred_sample_test.npy"), xs_pred_sample_test)
        dxs_pred_sample_test = jnp.concatenate((dfs_pred_sample_test, dgs_pred_sample_test), axis=-1)
        np.save(os.path.join(args.result_folder, f"{job_name}_dxs_pred_sample_test.npy"), dxs_pred_sample_test)
    else:
        xs_pred_sample_test = np.load(os.path.join(args.result_folder, f"{job_name}_xs_pred_sample_test.npy"))
        dxs_pred_sample_test = np.load(os.path.join(args.result_folder, f"{job_name}_dxs_pred_sample_test.npy"))
        dfs_pred_sample_test, dgs_pred_sample_test = dxs_pred_sample_test[..., :x_size], dxs_pred_sample_test[..., x_size:]

    if args.eval_and_plot:
        from evaluation_utils import (crps_ensemble_vectorized, compute_mean_ci_width,
                                      plot_sampled_results_derivative)

        if dxs_test is not None:
            rmse = np.sqrt(jnp.nanmean((dfs_pred_sample_test.mean(0) - dxs_test[..., :x_size]) ** 2))
            mciw = jnp.nanmean(compute_mean_ci_width(dgs_pred_sample_test))
        else:
            rmse = np.sqrt(jnp.nanmean((xs_pred_sample_test.mean(0) - xs_test) ** 2))
            mciw = jnp.nanmean(compute_mean_ci_width(dgs_pred_sample_test))
        crps = jnp.nanmean(crps_ensemble_vectorized(xs_test, xs_pred_sample_test))

        # xs_pred_test = jnp.nanmean(xs_pred_sample_test, axis=0)
        #
        # if args.use_normalization:
        #     if args.norm_method == 'std':
        #         xs_test = xs_test * std_y + mean_y
        #     else:
        #         xs_test = xs_test * (max_y - min_y) / 2
        #
        # if not os.path.isdir(os.path.join(args.figure_folder, job_name)):
        #     os.mkdir(os.path.join(args.figure_folder, job_name))
        # for test_id in range(xs_test.shape[0]):
        #     plot_sampled_results_derivative(
        #         ts_test[test_id], xs_test[test_id], xs_pred_sample_test[:, test_id],
        #         dxs_test[test_id][..., :x_size]/0.001 if dxs_test is not None else None,
        #         dfs_pred_sample_test[:, test_id]/args.dt0,
        #         dxs_test[test_id][..., x_size:] if dxs_test is not None else None,
        #         dgs_pred_sample_test[:, test_id],
        #         filename=os.path.join(args.figure_folder, job_name, f"{job_name}_{test_id}.png"))