

import torch
import torch.nn as nn
from einops import einsum
from einops import rearrange
from torch import Tensor
from jaxtyping import Float ,Bool, Int
from typing import Optional

from cs336_basics.modules.scaled_dot_product_attention import NIUscaled_dot_product_attention
from cs336_basics.modules.rope import NIURope
from cs336_basics.modules.linear import NIULinear
class NIUcausal_multi_head_self_attention(nn.Module):
    def __init__(self, 
                d_model: int,
                num_heads: int,
                use_position_embedding: bool = False,
                theta: float = 10000,
                max_seq_len: int = 1024,
                device: torch.device = None, 
                 ):
        super(NIUcausal_multi_head_self_attention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.device = device
        self.use_position_embedding = use_position_embedding
        
        
        self.head_dim = self.d_model // self.num_heads
        # rope是给每一个头的向量进行准备的
        if self.use_position_embedding:
            self.rope = NIURope(theta, self.head_dim, max_seq_len, device = device)

        # 矩阵表达方式的转变
        self.q_proj = NIULinear(d_model, d_model, device = device)
        self.k_proj = NIULinear(d_model, d_model, device = device)
        self.v_proj = NIULinear(d_model, d_model, device = device)
        self.output_proj = NIULinear(d_model, d_model, device = device)
        
        self.scale_dot_product_attention = NIUscaled_dot_product_attention()
        
    def forward(self,   x: torch.Tensor,
                        token_positions: Optional[Int[Tensor, " ... seq_len"]] = None
                        ) -> Float[Tensor, " ... seq_len d_model"]:
        
        seq_len = x.shape[-2]
        # 所有的头一起做运算，然后切分就可以了
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = rearrange(q, "... seq_len (head head_dim) -> ... head seq_len head_dim", head = self.num_heads)
        k = rearrange(k, "... seq_len (head head_dim) -> ... head seq_len head_dim", head = self.num_heads)
        v = rearrange(v, "... seq_len (head head_dim) -> ... head seq_len head_dim", head = self.num_heads)

        
        if self.use_position_embedding:
            q = self.rope(q, token_positions)
        if self.use_position_embedding:
            k = self.rope(k, token_positions)
            
        
        # 创建self-attention的mask,1代表的是看得见的，0代表的是看不见的
        mask = torch.tril(torch.ones(seq_len, seq_len))
        o = self.scale_dot_product_attention(Q=q, K=k, V=v, mask=mask) # shape is (..., seq_len, d_out)
        
        
        # concat the heads
        o = rearrange(o, "... head seq_len head_dim -> ... seq_len (head head_dim)")
        o = self.output_proj(o)
        return o
        