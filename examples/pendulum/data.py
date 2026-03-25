# code structure follows the style of Symplectic ODE-Net
# https://github.com/d-biswa/Symplectic-ODENet/blob/master/experiment-single-embed/data.py

import numpy as np
from se3hamneuralode import to_pickle, from_pickle
import os
import sys

def sample_gym(seed=0, timesteps=10, trials=50, min_angle=0., 
              verbose=False, u=0.0, env_name='MyPendulum-v1', ori_rep = 'rotmat',friction = False, render = False):
    import gym
    import envs
    
    gym_settings = locals()
    if verbose:
        print("Making a dataset of Pendulum observations.")
    env = gym.make(env_name)
    env.seed(seed)
    env.friction = friction
    trajs = []
    for trial in range(trials):
        valid = False
        while not valid:
            env.reset(ori_rep=ori_rep)
            traj = []
            for step in range(timesteps):
                if render:
                    env.render()
                obs, _, _, _ = env.step([u]) # action
                x = np.concatenate((obs, np.array([u])))
                traj.append(x)
            traj = np.stack(traj)
            if np.amax(traj[:, 2]) < env.max_speed - 0.001  and np.amin(traj[:, 2]) > -env.max_speed + 0.001:
                valid = True
        trajs.append(traj)
    trajs = np.stack(trajs) # (trials, timesteps, 2)
    trajs = np.transpose(trajs, (1, 0, 2)) # (timesteps, trails, 2)
    tspan = np.arange(timesteps) * 0.05
    return trajs, tspan, gym_settings


def _append_chgan_src_to_path():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(os.path.dirname(this_dir))
    chgan_src = os.path.join(repo_root, 'ControlledHamiltonianGAN', 'src')
    if chgan_src not in sys.path:
        sys.path.append(chgan_src)


# ---------------------------------------------------------------------------
#  SE(3) helper utilities
# ---------------------------------------------------------------------------

# Per-body SE(3) state layout: [x(3), R_flat(9), v(3), w(3)] = 18 dims
SE3_BODY_DIM = 18

def _rotmat_from_z_angle(theta):
    """Rotation matrix about z-axis for angle *theta*."""
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([[c, -s, 0.0],
                     [s,  c, 0.0],
                     [0.0, 0.0, 1.0]], dtype=np.float64)


_IDENTITY_ROTMAT_FLAT = np.eye(3, dtype=np.float64).reshape(-1)


def _build_se3_body_state(x3, R_flat9, v3, w3):
    """Concatenate per-body SE(3) state: x(3)+R(9)+v(3)+w(3)=18."""
    return np.concatenate([x3, R_flat9, v3, w3])


def _embed_2d_to_3d(xy):
    """Embed a 2D vector [x,y] into 3D [x,y,0]."""
    return np.array([xy[0], xy[1], 0.0], dtype=np.float64)


def _compute_orbital_se3_single(pos2d, vel2d, com2d, com_vel2d):
    """Compute SE(3) body state for a point mass in 2D orbit.

    Args:
        pos2d: (2,) position of the body
        vel2d: (2,) velocity of the body (dq/dt)
        com2d: (2,) center-of-mass position
        com_vel2d: (2,) center-of-mass velocity

    Returns:
        x3, R_flat9, v3, w3  (each numpy arrays)
    """
    x3 = _embed_2d_to_3d(pos2d)
    v3 = _embed_2d_to_3d(vel2d)

    # Relative position & velocity w.r.t. center of mass
    r = pos2d - com2d
    rv = vel2d - com_vel2d
    r_sq = float(r[0]**2 + r[1]**2)
    r_sq = max(r_sq, 1e-8)  # avoid division by zero

    # Orbital angular velocity  omega_z = (r x v_rel) / |r|^2
    omega_z = (r[0] * rv[1] - r[1] * rv[0]) / r_sq
    w3 = np.array([0.0, 0.0, omega_z], dtype=np.float64)

    # Orientation: heading angle of relative position vector
    angle = np.arctan2(r[1], r[0])
    R_flat9 = _rotmat_from_z_angle(angle).reshape(-1)

    return x3, R_flat9, v3, w3


# ---------------------------------------------------------------------------
#  CHGAN data generation helpers
# ---------------------------------------------------------------------------

def _get_chgan_system(system_name, variable_physics=False):
    """Return (system_instance, dataset_constant_name) for a CHGAN system."""
    _append_chgan_src_to_path()
    from hgan.dm_hamiltonian_dynamics_suite import datasets
    from hgan.dm_hamiltonian_dynamics_suite.hamiltonian_systems import n_body
    from hgan.dm_hamiltonian_dynamics_suite.hamiltonian_systems import utils as hutils

    _SYSTEM_MAP = {
        'pendulum':        (datasets.PENDULUM, datasets.PENDULUM_COLORS),
        'mass_spring':     (datasets.MASS_SPRING, datasets.MASS_SPRING_COLORS),
        'double_pendulum': (datasets.DOUBLE_PENDULUM, datasets.DOUBLE_PENDULUM_COLORS),
        'two_body':        (datasets.TWO_BODY, datasets.TWO_BODY_COLORS),
    }

    if system_name == 'three_body':
        # No predefined constant in datasets.py; create inline
        if variable_physics:
            cfg = dict(
                m_range=hutils.BoxRegion(0.5, 1.5),
                g_range=hutils.BoxRegion(0.5, 1.5),
                radius_range=hutils.BoxRegion(0.5, 1.5),
                provided_canvas_bounds=hutils.BoxRegion(-5.0, 5.0),
                randomize_canvas_location=False,
                num_colors=6,
            )
        else:
            cfg = dict(
                m_range=hutils.BoxRegion(1.0, 1.0),
                g_range=hutils.BoxRegion(1.0, 1.0),
                radius_range=hutils.BoxRegion(0.5, 1.5),
                provided_canvas_bounds=hutils.BoxRegion(-3.5, 3.5),
                randomize_canvas_location=False,
                num_colors=3,
            )
        return n_body.ThreeBody2DSystem(**cfg), datasets
    else:
        const, var = _SYSTEM_MAP[system_name]
        cls, cfg_fn = var if variable_physics else const
        return cls(**cfg_fn()), datasets


def _generate_chgan_trajectory(system, datasets_mod, seed, idx, dt, timesteps):
    """Generate a single CHGAN trajectory and return raw arrays."""
    sample = datasets_mod.generate_sample(
        index=seed + idx,
        system=system,
        dt=dt,
        num_steps=timesteps - 1,
        steps_per_dt=1,
    )
    x = np.asarray(sample['x'])        # (T, 2*system_dims)
    dx_dt = np.asarray(sample['dx_dt'])  # (T, 2*system_dims)
    other = {k: np.asarray(v) for k, v in sample['other'].items()}
    return x, dx_dt, other


# ---------------------------------------------------------------------------
#  System-specific samplers  (SO(3) pendulum kept for backward compat)
# ---------------------------------------------------------------------------

def sample_chgan_pendulum(seed=0, timesteps=20, trials=50, dt=0.05,
                          variable_physics=False, verbose=False):
    """Original SO(3)-only pendulum sampler (backward compatible)."""
    _append_chgan_src_to_path()
    from hgan.dm_hamiltonian_dynamics_suite import datasets

    if verbose:
        print("Making a dataset of CHGAN pendulum trajectories.")

    np.random.seed(seed)
    cls, cfg_fn = datasets.PENDULUM_COLORS if variable_physics else datasets.PENDULUM
    system = cls(**cfg_fn())

    trajs = []
    for idx in range(trials):
        sample = datasets.generate_sample(
            index=seed + idx,
            system=system,
            dt=dt,
            num_steps=timesteps - 1,
            steps_per_dt=1,
        )

        x = np.asarray(sample['x'])
        dx_dt = np.asarray(sample['dx_dt'])
        theta = x[:, 0]
        theta_dot = dx_dt[:, 0]

        state_traj = []
        for t in range(theta.shape[0]):
            R = _rotmat_from_z_angle(theta[t]).reshape(-1)
            w = np.array([0.0, 0.0, theta_dot[t]], dtype=np.float64)
            u = np.array([0.0], dtype=np.float64)
            state_traj.append(np.concatenate((R, w, u), axis=0))
        state_traj = np.stack(state_traj, axis=0)
        trajs.append(state_traj)

    trajs = np.stack(trajs, axis=0)
    trajs = np.transpose(trajs, (1, 0, 2))
    tspan = np.arange(timesteps, dtype=np.float64) * dt
    settings = {
        'backend': 'chgan_rt',
        'variable_physics': variable_physics,
        'dt': dt,
        'timesteps': timesteps,
        'trials': trials,
        'seed': seed,
    }
    return trajs, tspan, settings


# ---- SE(3) pendulum ----

def sample_chgan_pendulum_se3(seed=0, timesteps=20, trials=50, dt=0.05,
                              variable_physics=False, verbose=False):
    """Pendulum in SE(3) format: x(3)+R(9)+v(3)+w(3)+u(1) = 19D per step."""
    if verbose:
        print("Making SE(3) dataset of CHGAN pendulum trajectories.")

    np.random.seed(seed)
    system, datasets_mod = _get_chgan_system('pendulum', variable_physics)

    trajs = []
    for idx in range(trials):
        x_arr, dx_dt, other = _generate_chgan_trajectory(
            system, datasets_mod, seed, idx, dt, timesteps)

        # Pendulum: system_dims=1, x_arr columns: [theta, p_theta]
        theta = x_arr[:, 0]
        # dq/dt is the first half of dx_dt
        theta_dot = dx_dt[:, 0]

        # Length: use 'l' from other if available; else default 1.0
        l_val = float(other.get('l', np.array(1.0)).flat[0])

        state_traj = np.zeros((theta.shape[0], SE3_BODY_DIM + 1), dtype=np.float64)
        for t in range(theta.shape[0]):
            th = theta[t]
            th_d = theta_dot[t]
            # Forward kinematics
            pos = np.array([l_val * np.sin(th), -l_val * np.cos(th), 0.0])
            vel = np.array([l_val * np.cos(th) * th_d,
                            l_val * np.sin(th) * th_d, 0.0])
            R_flat = _rotmat_from_z_angle(th).reshape(-1)
            w = np.array([0.0, 0.0, th_d])
            state_traj[t, :SE3_BODY_DIM] = _build_se3_body_state(pos, R_flat, vel, w)
            state_traj[t, SE3_BODY_DIM] = 0.0  # control

        trajs.append(state_traj)

    trajs = np.stack(trajs, axis=0)           # (trials, T, 19)
    trajs = np.transpose(trajs, (1, 0, 2))    # (T, trials, 19)
    tspan = np.arange(timesteps, dtype=np.float64) * dt
    return trajs, tspan, {'backend': 'chgan_se3', 'system': 'pendulum',
                          'variable_physics': variable_physics,
                          'dt': dt, 'timesteps': timesteps,
                          'trials': trials, 'seed': seed}


# ---- Mass-Spring ----

def sample_chgan_mass_spring(seed=0, timesteps=20, trials=50, dt=0.05,
                             variable_physics=False, verbose=False):
    """Mass-spring in SE(3): x(3)+R(9)+v(3)+w(3)+u(1) = 19D.
    R = I (no rotation), w = 0."""
    if verbose:
        print("Making SE(3) dataset of CHGAN mass-spring trajectories.")

    np.random.seed(seed)
    system, datasets_mod = _get_chgan_system('mass_spring', variable_physics)

    trajs = []
    for idx in range(trials):
        x_arr, dx_dt, _other = _generate_chgan_trajectory(
            system, datasets_mod, seed, idx, dt, timesteps)

        # system_dims=1: columns [q, p]
        q = x_arr[:, 0]
        dq_dt = dx_dt[:, 0]

        T = q.shape[0]
        state_traj = np.zeros((T, SE3_BODY_DIM + 1), dtype=np.float64)
        for t in range(T):
            pos = np.array([q[t], 0.0, 0.0])
            vel = np.array([dq_dt[t], 0.0, 0.0])
            R_flat = _IDENTITY_ROTMAT_FLAT.copy()
            w = np.zeros(3)
            state_traj[t, :SE3_BODY_DIM] = _build_se3_body_state(pos, R_flat, vel, w)
            state_traj[t, SE3_BODY_DIM] = 0.0
        trajs.append(state_traj)

    trajs = np.stack(trajs, axis=0)
    trajs = np.transpose(trajs, (1, 0, 2))
    tspan = np.arange(timesteps, dtype=np.float64) * dt
    return trajs, tspan, {'backend': 'chgan_se3', 'system': 'mass_spring',
                          'variable_physics': variable_physics,
                          'dt': dt, 'timesteps': timesteps,
                          'trials': trials, 'seed': seed}


# ---- Double Pendulum (2 bodies) ----

def sample_chgan_double_pendulum(seed=0, timesteps=20, trials=50, dt=0.05,
                                 variable_physics=False, verbose=False):
    """Double pendulum → 2 bodies in SE(3): 2×18 + 1(u) = 37D."""
    if verbose:
        print("Making SE(3) dataset of CHGAN double-pendulum trajectories.")

    np.random.seed(seed)
    system, datasets_mod = _get_chgan_system('double_pendulum', variable_physics)

    trajs = []
    for idx in range(trials):
        x_arr, dx_dt, other = _generate_chgan_trajectory(
            system, datasets_mod, seed, idx, dt, timesteps)

        # system_dims=2: columns [theta1, theta2, p1, p2]
        theta1 = x_arr[:, 0]
        theta2 = x_arr[:, 1]
        dtheta1 = dx_dt[:, 0]
        dtheta2 = dx_dt[:, 1]

        l1 = float(other.get('l_1', np.array(1.0)).flat[0])
        l2 = float(other.get('l_2', np.array(1.0)).flat[0])

        T = theta1.shape[0]
        n_bodies = 2
        state_dim = n_bodies * SE3_BODY_DIM + 1
        state_traj = np.zeros((T, state_dim), dtype=np.float64)

        for t in range(T):
            th1, th2 = theta1[t], theta2[t]
            dth1, dth2 = dtheta1[t], dtheta2[t]

            # Body 1 FK
            x1 = np.array([l1 * np.sin(th1), -l1 * np.cos(th1), 0.0])
            v1 = np.array([l1 * np.cos(th1) * dth1, l1 * np.sin(th1) * dth1, 0.0])
            R1 = _rotmat_from_z_angle(th1).reshape(-1)
            w1 = np.array([0.0, 0.0, dth1])

            # Body 2 FK (tip of second link)
            x2 = np.array([l1 * np.sin(th1) + l2 * np.sin(th2),
                           -l1 * np.cos(th1) - l2 * np.cos(th2),
                           0.0])
            v2 = np.array([l1 * np.cos(th1) * dth1 + l2 * np.cos(th2) * dth2,
                           l1 * np.sin(th1) * dth1 + l2 * np.sin(th2) * dth2,
                           0.0])
            R2 = _rotmat_from_z_angle(th2).reshape(-1)
            w2 = np.array([0.0, 0.0, dth2])

            b1 = _build_se3_body_state(x1, R1, v1, w1)
            b2 = _build_se3_body_state(x2, R2, v2, w2)
            state_traj[t, :SE3_BODY_DIM] = b1
            state_traj[t, SE3_BODY_DIM:2*SE3_BODY_DIM] = b2
            state_traj[t, -1] = 0.0  # control
        trajs.append(state_traj)

    trajs = np.stack(trajs, axis=0)
    trajs = np.transpose(trajs, (1, 0, 2))
    tspan = np.arange(timesteps, dtype=np.float64) * dt
    return trajs, tspan, {'backend': 'chgan_se3', 'system': 'double_pendulum',
                          'variable_physics': variable_physics,
                          'dt': dt, 'timesteps': timesteps,
                          'trials': trials, 'seed': seed}


# ---- N-Body helper (shared by two_body and three_body) ----

def _sample_chgan_nbody(system_name, n_bodies, seed=0, timesteps=20,
                        trials=50, dt=0.05, variable_physics=False,
                        verbose=False):
    """Generic N-body sampler. Each body gets full SE(3) state.
    State dim = n_bodies * 18 + 1 (control)."""
    if verbose:
        print("Making SE(3) dataset of CHGAN {} trajectories.".format(system_name))

    np.random.seed(seed)
    system, datasets_mod = _get_chgan_system(system_name, variable_physics)

    space_dims = 2  # all n-body systems are 2D
    sys_dims = n_bodies * space_dims  # e.g. 4 for two_body

    trajs = []
    for idx in range(trials):
        x_arr, dx_dt, other = _generate_chgan_trajectory(
            system, datasets_mod, seed, idx, dt, timesteps)

        # x_arr columns: [q1x, q1y, q2x, q2y, ..., p1x, p1y, p2x, p2y, ...]
        # dx_dt columns: [dq1x, dq1y, ..., dp1x, dp1y, ...]
        q_all = x_arr[:, :sys_dims]          # (T, n_bodies*2)
        dq_all = dx_dt[:, :sys_dims]         # (T, n_bodies*2)

        # Masses for center-of-mass computation
        m = other.get('m', np.ones(n_bodies))
        m_flat = np.asarray(m).flatten()[:n_bodies]
        m_total = m_flat.sum()

        T = q_all.shape[0]
        state_dim = n_bodies * SE3_BODY_DIM + 1
        state_traj = np.zeros((T, state_dim), dtype=np.float64)

        for t in range(T):
            # Reshape to (n_bodies, 2)
            positions = q_all[t].reshape(n_bodies, space_dims)
            velocities = dq_all[t].reshape(n_bodies, space_dims)

            # Center of mass
            com = np.sum(positions * m_flat[:, None], axis=0) / m_total
            com_vel = np.sum(velocities * m_flat[:, None], axis=0) / m_total

            for b in range(n_bodies):
                x3, R9, v3, w3 = _compute_orbital_se3_single(
                    positions[b], velocities[b], com, com_vel)
                offset = b * SE3_BODY_DIM
                state_traj[t, offset:offset + SE3_BODY_DIM] = \
                    _build_se3_body_state(x3, R9, v3, w3)
            state_traj[t, -1] = 0.0  # control
        trajs.append(state_traj)

    trajs = np.stack(trajs, axis=0)
    trajs = np.transpose(trajs, (1, 0, 2))
    tspan = np.arange(timesteps, dtype=np.float64) * dt
    return trajs, tspan, {'backend': 'chgan_se3', 'system': system_name,
                          'variable_physics': variable_physics,
                          'dt': dt, 'timesteps': timesteps,
                          'trials': trials, 'seed': seed}


def sample_chgan_two_body(seed=0, timesteps=20, trials=50, dt=0.05,
                          variable_physics=False, verbose=False):
    """Two-body gravitational system → 2×18+1 = 37D."""
    return _sample_chgan_nbody('two_body', 2, seed=seed, timesteps=timesteps,
                               trials=trials, dt=dt,
                               variable_physics=variable_physics, verbose=verbose)


def sample_chgan_three_body(seed=0, timesteps=20, trials=50, dt=0.05,
                            variable_physics=False, verbose=False):
    """Three-body gravitational system → 3×18+1 = 55D."""
    return _sample_chgan_nbody('three_body', 3, seed=seed, timesteps=timesteps,
                               trials=trials, dt=dt,
                               variable_physics=variable_physics, verbose=verbose)


# ---------------------------------------------------------------------------
#  System metadata registry
# ---------------------------------------------------------------------------

SYSTEM_INFO = {
    'pendulum':        {'n_bodies': 1, 'state_dim': 19, 'sampler': sample_chgan_pendulum_se3},
    'mass_spring':     {'n_bodies': 1, 'state_dim': 19, 'sampler': sample_chgan_mass_spring},
    'double_pendulum': {'n_bodies': 2, 'state_dim': 37, 'sampler': sample_chgan_double_pendulum},
    'two_body':        {'n_bodies': 2, 'state_dim': 37, 'sampler': sample_chgan_two_body},
    'three_body':      {'n_bodies': 3, 'state_dim': 55, 'sampler': sample_chgan_three_body},
}


# ---------------------------------------------------------------------------
#  get_dataset  (extended with `system` parameter)
# ---------------------------------------------------------------------------

def get_dataset(seed=0, samples=50, test_split=0.5, save_dir=None, us=[0], rad=False,
                ori_rep='rotmat', friction=False, backend='gym', chgan_dt=0.05,
                chgan_variable_physics=False, system='pendulum', **kwargs):
    data = {}

    assert save_dir is not None

    if backend == 'chgan_se3':
        # New SE(3) multi-system backend
        dt_tag = str(chgan_dt).replace('.', 'p')
        phys_tag = 'var' if chgan_variable_physics else 'const'
        path = '{}/{}-chgan-se3-{}-dt{}-n{}-seed{}.pkl'.format(
            save_dir, system, phys_tag, dt_tag, samples, seed)
    elif backend == 'chgan_rt':
        # Legacy SO(3) pendulum backend
        dt_tag = str(chgan_dt).replace('.', 'p')
        phys_tag = 'var' if chgan_variable_physics else 'const'
        path = '{}/pendulum-chgan-rt-{}-dt{}-n{}-seed{}.pkl'.format(
            save_dir, phys_tag, dt_tag, samples, seed)
    else:
        path = '{}/pendulum-gym-dataset.pkl'.format(save_dir)
    try:
        data = from_pickle(path)
        print("Successfully loaded data from {}".format(path))
    except:
        print("Had a problem loading data from {}. Rebuilding dataset...".format(path))
        trajs_force = []
        if backend == 'chgan_se3':
            info = SYSTEM_INFO[system]
            sampler = info['sampler']
            trajs, tspan, _ = sampler(
                seed=seed,
                trials=samples,
                timesteps=kwargs.get('timesteps', 20),
                dt=chgan_dt,
                variable_physics=chgan_variable_physics,
                verbose=kwargs.get('verbose', False),
            )
            trajs_force.append(trajs)
        elif backend == 'chgan_rt':
            trajs, tspan, _ = sample_chgan_pendulum(
                seed=seed,
                trials=samples,
                timesteps=kwargs.get('timesteps', 20),
                dt=chgan_dt,
                variable_physics=chgan_variable_physics,
                verbose=kwargs.get('verbose', False),
            )
            trajs_force.append(trajs)
        else:
            for u in us:
                trajs, tspan, _ = sample_gym(seed=seed, trials=samples, u=u, ori_rep = ori_rep, friction = friction, **kwargs)
                trajs_force.append(trajs)
        data['x'] = np.stack(trajs_force, axis=0) # (n_force, timesteps, trials, state_dim)
        # make a train/test split
        split_ix = int(samples * test_split)
        split_data = {}
        split_data['x'], split_data['test_x'] = data['x'][:,:,:split_ix,:], data['x'][:,:,split_ix:,:]

        data = split_data
        data['t'] = tspan

        to_pickle(data, path)
    return data

def arrange_data(x, t, num_points=2):
    '''Arrange data to feed into neural ODE in small chunks'''
    assert num_points>=2 and num_points<=len(t)
    x_stack = []
    for i in range(num_points):
        if i < num_points-1:
            x_stack.append(x[:, i:-num_points+i+1,:,:])
        else:
            x_stack.append(x[:, i:,:,:])
    x_stack = np.stack(x_stack, axis=1)
    x_stack = np.reshape(x_stack, 
                (x.shape[0], num_points, -1, x.shape[3]))
    t_eval = t[0:num_points]
    return x_stack, t_eval

if __name__ == "__main__":
    import envs
    #us = [0.0, -1.0, 1.0, -2.0, 2.0]
    us = [0.0]
    #data = get_dataset(seed=0, timesteps=20, save_dir=None, us=us, samples=128)
    trajs, tspan, _  = sample_gym(seed=0, trials=50, u=us[0], timesteps=20, ori_rep='6d')
    print("Done!")