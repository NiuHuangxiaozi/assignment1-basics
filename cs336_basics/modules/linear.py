import torch
import torch.nn as nn
from einops import einsum
import math






class NIULinear(nn.Module):
    def __init__(self, in_features, out_features, device = None, dtype = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype
        
        self.weight = torch.nn.Parameter(torch.empty(out_features, in_features, device = device, dtype = dtype))
        
        # 初始化
        mean = 0
        std = math.sqrt(2/(in_features+out_features))
        torch.nn.init.trunc_normal_(self.weight, mean = mean, std = std, a=-3 * std, b = 3 * std)
        
        if device is not None:
            self.weight.to(device)
        if dtype is not None:
            self.weight.to(dtype)
        
        
        # 
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.device == self.weight.device
        result = einsum(x, self.weight, "... in_features, out_features in_features -> ... out_features")
        return result
    