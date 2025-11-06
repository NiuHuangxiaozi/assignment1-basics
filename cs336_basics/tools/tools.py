import math
import torch
import numpy as np
import os
from typing import BinaryIO, IO


def cosine_scheduling(t, alpha_max, alpha_min, T_warmup, T_cycle):
    if t<T_warmup:
        return t/T_warmup * alpha_max
    elif t<=T_cycle and t>=T_warmup:
        return alpha_min + 0.5 *(1+math.cos((t-T_warmup)/(T_cycle-T_warmup)*math.pi))*(alpha_max-alpha_min)
    else:
        return alpha_min
    
    
def gradient_clipping(parameters, max_norm,eps=1e-6):
    
    total_l2_norm_value = 0
    for p in parameters:
        if p.grad is not None:
            total_l2_norm_value += torch.norm(p.grad, p=2)**2
    total_l2_norm_value = torch.sqrt(total_l2_norm_value)
    if total_l2_norm_value > max_norm:
        scale = max_norm / (total_l2_norm_value + eps)
        for p in parameters:
            if p.grad is not None:
                p.grad = p.grad * scale
                


def data_loading(dataset: np.ndarray, batch_size: int, context_length: int, device: str):
    
    
    if len(dataset) <=context_length:
        raise ValueError("Dataset length is less than context length")

    max_random_index = len(dataset) - context_length
    random_indices = np.random.randint(0, max_random_index, batch_size)
    
    x = np.zeros((batch_size, context_length), dtype=dataset.dtype)
    y = np.zeros((batch_size, context_length), dtype=dataset.dtype)
    
    for index, begin_idx in enumerate(random_indices):
        input_sequence = dataset[begin_idx:begin_idx+context_length]
        target_sequence = dataset[begin_idx+1:begin_idx+context_length+1]
        x[index] = input_sequence
        y[index] = target_sequence
    return [torch.tensor(x, dtype = torch.long, device = device), torch.tensor(y, dtype = torch.long, device = device)]


def save_checkpoint(model:torch.nn.Module,
                    optimizer:torch.optim.Optimizer,
                    iteration:int,
                    out:str | os.PathLike | BinaryIO | IO[bytes]):
    
    state_dict = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'iteration': iteration
    }
    torch.save(state_dict, out)

def load_checkpoint(src:str | os.PathLike | BinaryIO | IO[bytes],
                    model:torch.nn.Module,
                    optimizer:torch.optim.Optimizer):
    state_dict = torch.load(src)
    model.load_state_dict(state_dict['model'])
    for k, v in state_dict['optimizer'].items():
        print(k, type(k))
    optimizer.load_state_dict(state_dict['optimizer'])
    return state_dict['iteration']

