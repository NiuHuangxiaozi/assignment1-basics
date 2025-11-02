
import torch
import torch.nn as nn





class NIURMSNorm(nn.Module):
    def __init__(self,d_model: int , eps: float=1e-5, device = None, dtype = None):
        super(NIURMSNorm, self).__init__()
        
        
        self.d_model = d_model
        self.eps = eps
        self.device = device
        self.dtype = dtype

        self.weight = torch.nn.Parameter(torch.ones(d_model, device = device, dtype = dtype))

        if self.device is not None:
            self.weight.to(self.device)
        if self.dtype is not None:
            self.weight.to(self.dtype)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_type = x.dtype
        x = x.to(torch.float32)
        RMS = torch.sqrt(torch.mean(x **2,dim=-1,keepdim=True)+self.eps)
        result = x / RMS * self.weight
        return result.to(in_type)