from sympy.strategies.util import new
import torch
import math
import torch.nn as nn
from typing import Optional, Callable

class NIUAdam(torch.optim.Optimizer):
    def __init__(self, params,
                 lr=1e-3,
                 betas=(0.9, 0.999),
                 eps=1e-8,
                 weight_decay=0.0
                 ):
        # AdamW的超参数
        defaults = {
                    "lr": lr,
                "betas": betas,
                "eps": eps,
                "weight_decay": weight_decay,
                }
        super().__init__(params, defaults)
        
        # m 是第一动量， v是第二动量
        for group in self.param_groups:
            for p in group["params"]:
                self.state[p] = {
                        "m": torch.zeros_like(p.data),
                        "v": torch.zeros_like(p.data),
                    }
                self.state[p]["t"] = 0
        
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        
        # 每调用依次step函数，步数就加1
        for group in self.param_groups:
            
            # m,v 对应于每一个参数，所以在内层循环更新
            for p in group["params"]:
                t = self.state[p]["t"] + 1
                
                old_lr = group["lr"]
                new_lr = old_lr *math.sqrt(1 - group["betas"][1]**t)/(1-group["betas"][0]**t)
                
                if p.grad is None:
                    continue
                grad = p.grad.data
                
                # 更新第一动量
                m = self.state[p]["m"]
                new_m = group["betas"][0] * m + (1-group["betas"][0]) * grad
                self.state[p]["m"] = new_m
            
                # 更新第二动量
                v = self.state[p]["v"]
                new_v = group["betas"][1] * v + (1-group["betas"][1]) * grad**2
                self.state[p]["v"] = new_v
                
                # 更新步数
                self.state[p]["t"] = t
                
                # 更新参数
                p.data = p.data - new_lr * (new_m/(torch.sqrt(new_v)+ group["eps"]))
                p.data = p.data - old_lr * group["weight_decay"] * p.data
                
        return loss        