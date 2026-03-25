# Multi-system SE(3) Hamiltonian Neural ODE Training
# Supports: pendulum, mass_spring, double_pendulum, two_body, three_body
# Uses CHGAN dataset backend with per-body SE(3) state representation

import torch, argparse
import numpy as np
import os, sys
import time
from torchdiffeq import odeint_adjoint as odeint

THIS_DIR = os.path.dirname(os.path.abspath(__file__)) + '/data'
REPO_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(REPO_DIR)

from se3hamneuralode import SE3HamNODE, MultiBodySE3HamNODE
from se3hamneuralode import (to_pickle, pose_L2_geodesic_loss,
                              traj_pose_L2_geodesic_loss,
                              multibody_pose_L2_geodesic_loss,
                              traj_multibody_pose_L2_geodesic_loss)
from data import get_dataset, arrange_data, SYSTEM_INFO


def get_args():
    parser = argparse.ArgumentParser(description='Multi-system SE(3) Hamiltonian NODE training')
    parser.add_argument('--learn_rate', default=2e-4, type=float)
    parser.add_argument('--nonlinearity', default='tanh', type=str)
    parser.add_argument('--total_steps', default=2000, type=int)
    parser.add_argument('--print_every', default=100, type=int)
    parser.add_argument('--name', default='multisystem', type=str)
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--save_dir', default=THIS_DIR, type=str)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--num_points', type=int, default=5)
    parser.add_argument('--solver', default='rk4', type=str)
    parser.add_argument('--samples', default=64, type=int)
    parser.add_argument('--system', default='pendulum', type=str,
                        choices=['pendulum', 'mass_spring', 'double_pendulum',
                                 'two_body', 'three_body'])
    parser.add_argument('--chgan_dt', default=0.05, type=float)
    parser.add_argument('--chgan_variable_physics', action='store_true')
    parser.set_defaults(feature=True)
    return parser.parse_args()


def get_model_parm_nums(model):
    return sum(p.nelement() for p in model.parameters())


def build_model(system, device, udim=1):
    """Build the appropriate model for the given system."""
    info = SYSTEM_INFO[system]
    n_bodies = info['n_bodies']

    if n_bodies == 1:
        model = SE3HamNODE(device=device, pretrain=True, udim=udim)
    else:
        model = MultiBodySE3HamNODE(n_bodies=n_bodies, device=device,
                                     pretrain=True, udim=udim)
    return model.to(device), n_bodies


def compute_loss(target, target_hat, n_bodies, udim=1):
    """Compute loss based on number of bodies."""
    if n_bodies == 1:
        split = [3, 9, 6, udim]
        return pose_L2_geodesic_loss(target, target_hat, split=split)
    else:
        return multibody_pose_L2_geodesic_loss(target, target_hat, n_bodies, udim)


def compute_traj_loss(traj, traj_hat, n_bodies, udim=1):
    """Compute trajectory-level loss."""
    if n_bodies == 1:
        split = [3, 9, 6, udim]
        return traj_pose_L2_geodesic_loss(traj, traj_hat, split=split)
    else:
        return traj_multibody_pose_L2_geodesic_loss(traj, traj_hat, n_bodies, udim)


def train(args):
    device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Build model
    model, n_bodies = build_model(args.system, device, udim=1)
    num_parm = get_model_parm_nums(model)
    print('System: {}, n_bodies: {}, model params: {}'.format(
        args.system, n_bodies, num_parm))

    optim = torch.optim.Adam(model.parameters(), args.learn_rate,
                              weight_decay=1e-4, foreach=False)

    # Collect data via chgan_se3 backend
    data = get_dataset(
        seed=args.seed,
        timesteps=20,
        save_dir=args.save_dir,
        us=[0.0],
        samples=args.samples,
        backend='chgan_se3',
        system=args.system,
        chgan_dt=args.chgan_dt,
        chgan_variable_physics=args.chgan_variable_physics,
    )

    train_x, t_eval = arrange_data(data['x'], data['t'], num_points=args.num_points)
    test_x, t_eval = arrange_data(data['test_x'], data['t'], num_points=args.num_points)
    train_x_cat = np.concatenate(train_x, axis=1)
    test_x_cat = np.concatenate(test_x, axis=1)
    train_x_cat = torch.tensor(train_x_cat, requires_grad=True, dtype=torch.float64).to(device)
    test_x_cat = torch.tensor(test_x_cat, requires_grad=True, dtype=torch.float64).to(device)
    t_eval = torch.tensor(t_eval, requires_grad=True, dtype=torch.float64).to(device)

    # Training stats
    stats = {
        'train_loss': [], 'test_loss': [],
        'forward_time': [], 'backward_time': [], 'nfe': [],
        'train_l2_loss': [], 'test_l2_loss': [],
        'train_geo_loss': [], 'test_geo_loss': [],
    }

    if args.verbose:
        print("Start training: num_points={}, solver={}, system={}".format(
            args.num_points, args.solver, args.system))

    for step in range(args.total_steps + 1):
        t0 = time.time()
        train_x_hat = odeint(model, train_x_cat[0, :, :], t_eval, method=args.solver)
        forward_time = time.time() - t0

        target = train_x_cat[1:, :, :]
        target_hat = train_x_hat[1:, :, :]
        (train_loss, train_x_loss, train_v_loss,
         train_w_loss, train_geo_loss) = compute_loss(target, target_hat, n_bodies)

        t0 = time.time()
        train_loss.backward()
        optim.step()
        optim.zero_grad()
        backward_time = time.time() - t0

        # Test loss
        test_x_hat = odeint(model, test_x_cat[0, :, :], t_eval, method=args.solver)
        target = test_x_cat[1:, :, :]
        target_hat = test_x_hat[1:, :, :]
        (test_loss, test_x_loss, test_v_loss,
         test_w_loss, test_geo_loss) = compute_loss(target, target_hat, n_bodies)

        train_l2 = (train_x_loss + train_v_loss + train_w_loss)
        test_l2 = (test_x_loss + test_v_loss + test_w_loss)

        stats['train_loss'].append(train_loss.item())
        stats['test_loss'].append(test_loss.item())
        stats['train_l2_loss'].append(train_l2.item())
        stats['test_l2_loss'].append(test_l2.item())
        stats['train_geo_loss'].append(train_geo_loss.item())
        stats['test_geo_loss'].append(test_geo_loss.item())
        stats['forward_time'].append(forward_time)
        stats['backward_time'].append(backward_time)
        stats['nfe'].append(model.nfe)

        if step % args.print_every == 0:
            print("step {}, train_loss {:.4e}, test_loss {:.4e}".format(
                step, train_loss.item(), test_loss.item()))
            print("  train l2 {:.4e}, geo {:.4e} | test l2 {:.4e}, geo {:.4e}".format(
                train_l2.item(), train_geo_loss.item(),
                test_l2.item(), test_geo_loss.item()))
            print("  nfe {:.0f}".format(model.nfe))

    # ----- Final trajectory-level evaluation -----
    train_x_raw, t_eval_raw = data['x'], data['t']
    test_x_raw = data['test_x']

    train_x_raw = torch.tensor(train_x_raw, requires_grad=True, dtype=torch.float64).to(device)
    test_x_raw = torch.tensor(test_x_raw, requires_grad=True, dtype=torch.float64).to(device)
    t_eval_raw = torch.tensor(t_eval_raw, requires_grad=True, dtype=torch.float64).to(device)

    train_loss_list, test_loss_list = [], []
    train_l2_list, test_l2_list = [], []
    train_geo_list, test_geo_list = [], []
    train_data_hat, test_data_hat = [], []

    for i in range(train_x_raw.shape[0]):
        tr_hat = odeint(model, train_x_raw[i, 0, :, :], t_eval_raw, method=args.solver)
        tl, l2l, gl = compute_traj_loss(train_x_raw[i], tr_hat, n_bodies)
        train_loss_list.append(tl)
        train_l2_list.append(l2l)
        train_geo_list.append(gl)
        train_data_hat.append(tr_hat.detach().cpu().numpy())

        te_hat = odeint(model, test_x_raw[i, 0, :, :], t_eval_raw, method=args.solver)
        tl, l2l, gl = compute_traj_loss(test_x_raw[i], te_hat, n_bodies)
        test_loss_list.append(tl)
        test_l2_list.append(l2l)
        test_geo_list.append(gl)
        test_data_hat.append(te_hat.detach().cpu().numpy())

    train_loss_cat = torch.cat(train_loss_list, dim=1)
    test_loss_cat = torch.cat(test_loss_list, dim=1)
    train_per_traj = torch.sum(train_loss_cat, dim=0)
    test_per_traj = torch.sum(test_loss_cat, dim=0)

    print('Final trajectory train loss {:.4e} +/- {:.4e}'.format(
        train_per_traj.mean().item(), train_per_traj.std().item()))
    print('Final trajectory test  loss {:.4e} +/- {:.4e}'.format(
        test_per_traj.mean().item(), test_per_traj.std().item()))

    stats['traj_train_loss'] = train_per_traj.detach().cpu().numpy()
    stats['traj_test_loss'] = test_per_traj.detach().cpu().numpy()
    stats['train_x'] = train_x_raw.detach().cpu().numpy()
    stats['test_x'] = test_x_raw.detach().cpu().numpy()
    stats['train_x_hat'] = np.array(train_data_hat)
    stats['test_x_hat'] = np.array(test_data_hat)
    stats['t_eval'] = t_eval_raw.detach().cpu().numpy()
    return model, stats


if __name__ == "__main__":
    args = get_args()
    model, stats = train(args)

    os.makedirs(args.save_dir, exist_ok=True)
    label = '-se3ham'
    path = '{}/{}-{}{}-{}-{}p.tar'.format(
        args.save_dir, args.system, args.name, label, args.solver, args.num_points)
    torch.save(model.state_dict(), path)
    stats_path = '{}/{}-{}{}-{}-{}p-stats.pkl'.format(
        args.save_dir, args.system, args.name, label, args.solver, args.num_points)
    print("Saved model:", path)
    print("Saved stats:", stats_path)
    to_pickle(stats, stats_path)
