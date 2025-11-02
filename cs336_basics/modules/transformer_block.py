
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
                 device:torch.device = None):
        super(NiuTransformerblock, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.device = device
        self.use_position_embedding = True
        
        # 下面定义需要的训练参数
        self.ln1 = NIURMSNorm(d_model, device = self.device)
        self.ln2 = NIURMSNorm(d_model, device = self.device)
        
        self.attn = NIUcausal_multi_head_self_attention(self.d_model,
                                                             self.num_heads,
                                                             self.use_position_embedding,
                                                             self.theta,
                                                             self.max_seq_len,
                                                             self.device)

        self.ffn = NIUSWIGLUFFN(self.d_model,
                                  self.d_ff,
                                  device = self.device)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        pre_attention_normed_x = self.ln1(x)
        
        attention_output = self.attn(pre_attention_normed_x)
        
        x1 = attention_output + x
        
        pre_ffn_normed_x = self.ln2(x1)
        
        ffn_output = self.ffn(pre_ffn_normed_x)
        
        x2 = ffn_output + x1
        
        return x2
        