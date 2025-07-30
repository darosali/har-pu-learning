import torch
from mamba_ssm import Mamba2

batch, length, dim = 2, 64, 32
x = torch.randn(batch, length, dim).to("cuda")
model = Mamba2(d_model=dim, d_state=64, d_conv=4, expand=2,).to("cuda")
y = model(x)
assert y.shape == x.shape

# import torch

# from mamba_ssm import Mamba

# batch, length, dim = 2, 64, 16
# x = torch.randn(batch, length, dim).to("cuda")
# model = Mamba(
#     # This module uses roughly 3 * expand * d_model^2 parameters
#     d_model=dim, # Model dimension d_model
#     d_state=16,  # SSM state expansion factor
#     d_conv=4,    # Local convolution width
#     expand=2,    # Block expansion factor
# ).to("cuda")
# y = model(x)
# assert y.shape == x.shape

# import torch
# from causal_conv1d import causal_conv1d_fn

# batch, dim, seq, width = 10, 5, 17, 4
# x = torch.zeros((batch, dim, seq)).to('cuda')
# weight = torch.zeros((dim, width)).to('cuda')
# bias = torch.zeros((dim, )).to('cuda')

# causal_conv1d_fn(x, weight, bias, None)