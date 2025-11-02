

import torch
import torch.nn as nn

from cs336_basics.modules.transformer_block import NiuTransformerblock
from cs336_basics.modules.linear import NIULinear
from cs336_basics.modules.rmsnorn import NIURMSNorm
from cs336_basics.modules.softmax import NIUsoftmax
from cs336_basics.modules.embedding import NIUEmbedding

class NiuTransformerLM(nn.Module):
    def __init__(self,
                 vocab_size:int,
                 context_length:int,
                 d_model:int,
                 num_layers:int,
                 num_heads:int,
                 d_ff:int,
                 theta:float,
                 device:torch.device = None):
        super(NiuTransformerLM, self).__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.num_layers = num_layers
        self.d_model = d_model
        self.device = device
        
        self.token_embeddings = NIUEmbedding(vocab_size, d_model, device = device)
        self.layers = nn.ModuleList([NiuTransformerblock(d_model,
                                                         num_heads,
                                                         d_ff,
                                                         context_length,
                                                         theta,
                                                         device = device) for _ in range(num_layers)])
        self.ln_final = NIURMSNorm(d_model, device = device)
        self.lm_head = NIULinear(d_model, vocab_size, device = device)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.token_embeddings(x)
        for layer in self.layers:
            x = layer(x)
        x = self.ln_final(x)
        output = self.lm_head(x)
        # probabilities = NIUsoftmax()(output, dim=-1)
        return output