import sys, os
sys.path.insert(0, '/home/jiayinliu/projects/SE3HamDL')
os.chdir('/home/jiayinliu/projects/SE3HamDL/examples/pendulum')
from data import get_dataset, arrange_data
import numpy as np

# Test backward-compatible chgan_rt pendulum
data = get_dataset(seed=0, samples=10, save_dir='./data', backend='chgan_rt',
                   chgan_dt=0.05, timesteps=10)
print('chgan_rt x shape:', data['x'].shape)
assert data['x'].shape[-1] == 13, 'Expected 13D SO(3) state (R9+w3+u1)'

# Test new chgan_se3 pendulum
data2 = get_dataset(seed=0, samples=10, save_dir='./data', backend='chgan_se3',
                    system='pendulum', chgan_dt=0.05, timesteps=10)
print('chgan_se3 pendulum x shape:', data2['x'].shape)
assert data2['x'].shape[-1] == 19, 'Expected 19D SE(3) state'

# Test chgan_se3 two_body
data3 = get_dataset(seed=0, samples=10, save_dir='./data', backend='chgan_se3',
                    system='two_body', chgan_dt=0.05, timesteps=10)
print('chgan_se3 two_body x shape:', data3['x'].shape)
assert data3['x'].shape[-1] == 37, 'Expected 37D (2*18+1)'

# Test arrange_data with multi-body
train_x, t_eval = arrange_data(data3['x'], data3['t'], num_points=3)
print('arrange_data two_body shape:', train_x.shape)

print('All backward compat + new systems OK')
