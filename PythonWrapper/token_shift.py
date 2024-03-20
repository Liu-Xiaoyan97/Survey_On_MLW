import torch
from torch import nn as nn

rd_tensor = torch.randn([3, 10, 256])
token_shift = nn.ZeroPad2d((0, 0, 1, -1))
print(rd_tensor)
print(token_shift(rd_tensor))