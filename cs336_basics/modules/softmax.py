import torch
import torch.nn as nn





class NIUsoftmax(nn.Module):
    def __init__(self):
        super(NIUsoftmax, self).__init__()
        pass
    def forward(self, x :torch.Tensor, dim: int):
        max_value,_ = torch.max(x, dim=dim,keepdim=True)
        return torch.exp(x-max_value)/ torch.sum(torch.exp(x-max_value), dim=dim, keepdim = True)