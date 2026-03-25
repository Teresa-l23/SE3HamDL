#!/usr/bin/env python
"""Quick training test for multi-system SE(3) Hamiltonian NODE.

Runs 3 training steps on each of the 5 systems with minimal data to verify
the full pipeline: data generation -> model build -> forward/backward -> loss.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
import time
from torchdiffeq import odeint_adjoint as odeint

from se3hamneuralode import SE3HamNODE, MultiBodySE3HamNODE
from se3hamneuralode import (pose_L2_geodesic_loss,
                              multibody_pose_L2_geodesic_loss)
from data import get_dataset, arrange_data, SYSTEM_INFO

SYSTEMS = ['pendulum', 'mass_spring', 'double_pendulum', 'two_body', 'three_body']
TRAIN_STEPS = 3
SAMPLES = 8
NUM_POINTS = 3
TIMESTEPS = 10
SOLVER = 'rk4'


def test_system(system):
    device = torch.device('cpu')
    torch.manual_seed(0)
    np.random.seed(0)

    info = SYSTEM_INFO[system]
    n_bodies = info['n_bodies']
    expected_dim = info['state_dim']

    # 1. Data generation
    print(f'  [{system}] Generating data (samples={SAMPLES}, timesteps={TIMESTEPS})...')
    data = get_dataset(
        seed=0, timesteps=TIMESTEPS, save_dir='./data',
        us=[0.0], samples=SAMPLES, backend='chgan_se3',
        system=system, chgan_dt=0.05,
    )
    assert data['x'].shape[-1] == expected_dim, \
        f'Expected {expected_dim}D, got {data["x"].shape[-1]}D'

    train_x, t_eval = arrange_data(data['x'], data['t'], num_points=NUM_POINTS)
    test_x, t_eval = arrange_data(data['test_x'], data['t'], num_points=NUM_POINTS)
    train_x_cat = np.concatenate(train_x, axis=1)
    test_x_cat = np.concatenate(test_x, axis=1)
    train_x_cat = torch.tensor(train_x_cat, requires_grad=True, dtype=torch.float64).to(device)
    test_x_cat = torch.tensor(test_x_cat, requires_grad=True, dtype=torch.float64).to(device)
    t_eval = torch.tensor(t_eval, requires_grad=True, dtype=torch.float64).to(device)

    # 2. Model build
    print(f'  [{system}] Building model (n_bodies={n_bodies})...')
    if n_bodies == 1:
        model = SE3HamNODE(device=device, pretrain=True, udim=1).to(device)
    else:
        model = MultiBodySE3HamNODE(n_bodies=n_bodies, device=device,
                                     pretrain=True, udim=1).to(device)
    n_params = sum(p.nelement() for p in model.parameters())
    print(f'  [{system}] Model params: {n_params}')

    # 3. Training loop
    optim = torch.optim.Adam(model.parameters(), 2e-4, weight_decay=1e-4, foreach=False)

    for step in range(TRAIN_STEPS):
        t0 = time.time()
        train_x_hat = odeint(model, train_x_cat[0, :, :], t_eval, method=SOLVER)
        fwd_time = time.time() - t0

        target = train_x_cat[1:, :, :]
        target_hat = train_x_hat[1:, :, :]

        if n_bodies == 1:
            loss, x_l, v_l, w_l, geo_l = pose_L2_geodesic_loss(
                target, target_hat, split=[3, 9, 6, 1])
        else:
            loss, x_l, v_l, w_l, geo_l = multibody_pose_L2_geodesic_loss(
                target, target_hat, n_bodies, udim=1)

        loss.backward()
        optim.step()
        optim.zero_grad()

        # Test forward pass
        with torch.no_grad():
            test_x_hat = odeint(model, test_x_cat[0, :, :], t_eval, method=SOLVER)

        print(f'  [{system}] step {step}: loss={loss.item():.4e}, '
              f'geo={geo_l.item():.4e}, fwd={fwd_time:.2f}s, nfe={model.nfe}')

    print(f'  [{system}] PASSED')
    return True


if __name__ == '__main__':
    print('=' * 60)
    print('Multi-system SE(3) training smoke test')
    print('=' * 60)

    passed = []
    failed = []
    for system in SYSTEMS:
        print(f'\n--- Testing {system} ---')
        try:
            test_system(system)
            passed.append(system)
        except Exception as e:
            print(f'  [{system}] FAILED: {e}')
            import traceback; traceback.print_exc()
            failed.append(system)

    print('\n' + '=' * 60)
    print(f'Results: {len(passed)}/{len(SYSTEMS)} passed')
    if passed:
        print(f'  Passed: {", ".join(passed)}')
    if failed:
        print(f'  Failed: {", ".join(failed)}')
    print('=' * 60)

    sys.exit(1 if failed else 0)
