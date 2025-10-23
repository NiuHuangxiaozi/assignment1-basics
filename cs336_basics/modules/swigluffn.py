
import torch
import torch.nn as nn

from cs336_basics.modules.linear import NIULinear




class NIUSiLU(nn.Module):
    def __init__(self):
        super(NIUSiLU, self).__init__()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)
        
        
class NIUSWIGLUFFN(nn.Module):
    def __init__(self,d_model:int,d_ff:int,device = None, dtype = None):
        super(NIUSWIGLUFFN, self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.device = device
        self.dtype = dtype
        
        self.w1 = NIULinear(d_model, d_ff, device = device, dtype = dtype)
        self.w2 = NIULinear(d_ff, d_model, device = device, dtype = dtype)
        self.w3 = NIULinear(d_model, d_ff, device = device, dtype = dtype)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.device == self.w1.weight.device
        assert x.device == self.w2.weight.device
        assert x.device == self.w3.weight.device
        
        a = self.w1(x)
        b = NIUSiLU()(a)
        
        c = self.w3(x)
        
        bc = b * c
        return self.w2(bc)