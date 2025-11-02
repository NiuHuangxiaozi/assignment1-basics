
import torch
import torch.nn as nn


from cs336_basics.modules.rmsnorn import NIURMSNorm
from cs336_basics.modules.causal_multi_head_self_attention import NIUcausal_multi_head_self_attention
from cs336_basics.modules.swigluffn import NIUSWIGLUFFN
class NiuTransformerblock(nn.Module):
    def __init__(self,
                 d_model:int,
                 num_heads:int,
                 d_ff:int,
                 max_seq_len:int, 
                 theta:float,
                 device:torch.device = None,
                 **kwargs):
        super(NiuTransformerblock, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.device = device
        self.use_position_embedding = True
        
        # 下面定义需要的训练参数
        self.attention_norm = NIURMSNorm(d_model, device = self.device)
        self.pwffn_norm = NIURMSNorm(d_model, device = self.device)
        
        '''
            weights (dict[str, Tensor]):
            State dict of our reference implementation.
            The keys of this dictionary are:
            - `attn.q_proj.weight`
                The query projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
            - `attn.k_proj.weight`
                The key projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
            - `attn.v_proj.weight`
                The value projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_v),
                so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
            - `attn.output_proj.weight`
                Weight of the multi-head self-attention output projection
                Shape is (d_model, d_model).
            - `ln1.weight`
                Weights of affine transform for the first RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `ffn.w1.weight`
                Weight of the first linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `ffn.w2.weight`
                Weight of the second linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `ffn.w3.weight`
                Weight of the third linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `ln2.weight`
                Weights of affine transform for the second RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
        '''
        self.q_proj_weight = nn.Parameter(torch.randn(self.d_model, self.d_model))
        self.k_proj_weight = nn.Parameter(torch.randn(self.d_model, self.d_model))
        self.v_proj_weight = nn.Parameter(torch.randn(self.d_model, self.d_model))
        self.o_proj_weight = nn.Parameter(torch.randn(self.d_model, self.d_model))
        
        self.attention = NIUcausal_multi_head_self_attention(self.d_model,
                                                             self.num_heads,
                                                             self.q_proj_weight,
                                                             self.k_proj_weight, 
                                                             self.v_proj_weight,
                                                             self.o_proj_weight, 
                                                             self.use_position_embedding,
                                                             self.theta,
                                                             self.max_seq_len,
                                                             self.device)

        self.pwffn = NIUSWIGLUFFN(self.d_model,
                                  self.d_ff,
                                  device = self.device)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        
        pre_attention_normed_x = self.attention_norm(x)
        
        attention_output = self.attention(pre_attention_normed_x)
        
        x1 = attention_output + x
        
        pre_pwffn_normed_x = self.pwffn_norm(x1)
        
        pwffn_output = self.pwffn(pre_pwffn_normed_x)
        
        x2 = pwffn_output + x1
        
        return x2
        