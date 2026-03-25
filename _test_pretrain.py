"""Quick test to reproduce MultiBodySE3HamNODE pretrain optimizer issue."""
import torch, numpy as np
from se3hamneuralode import MultiBodySE3HamNODE

device = torch.device('cuda:0')
print('Creating MultiBodySE3HamNODE on', device)
model = MultiBodySE3HamNODE(n_bodies=2, device=device, pretrain=True, udim=1)
print('SUCCESS - model created!')
