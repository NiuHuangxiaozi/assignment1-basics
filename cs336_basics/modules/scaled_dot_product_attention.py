

import torch
import torch.nn as nn
from einops import rearrange, einsum

from jaxtyping import Float, Bool
from torch import Tensor


from cs336_basics.modules.softmax import NIUsoftmax

class NIUscaled_dot_product_attention(nn.Module):
    def __init__(self, *args, **kwargs):
        super(NIUscaled_dot_product_attention, self).__init__()
        pass
    def forward(self,
                Q: Float[Tensor, "... queries d_k"],
                K: Float[Tensor, "... keys d_k"],
                V: Float[Tensor, "... keys d_v"],
                mask: Bool[Tensor, " ... queries keys"] | None = None)\
                -> Float[Tensor, " ... queries d_v"]:

                mask = mask.to(Q.device)
                
                d_k = Q.shape[-1]
                dot_product = einsum(Q, K, '... queries d_k, ... keys d_k -> ... queries keys')
                attention_score_matrix = dot_product/torch.sqrt(torch.tensor(d_k, device=dot_product.device))
                # if mask mask some scores
                if mask is not None:
                    masked_attention_score_matrix = attention_score_matrix.masked_fill_(~mask.bool(), -float('inf'))
                else:
                    masked_attention_score_matrix = attention_score_matrix
                
                softmax_attention_score_matrix = NIUsoftmax()(masked_attention_score_matrix, dim=-1)
                result = einsum(softmax_attention_score_matrix, V, '... queries keys, ... keys d_v -> ... queries d_v')
                return result
                
                
        
                
            
            
        
    
    