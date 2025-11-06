

import torch.nn as nn
import torch
from einops import einsum
from einops import rearrange


class NIURope(nn.Module):
    def __init__(self, theta:float, d_k:int, max_seq_len:int,
                 device=None):
        super(NIURope, self).__init__()
        
        # 确保维度是一个偶数
        assert d_k % 2 ==0
        
        self.theta = theta
        self.d_k = d_k
        self.device = device
        self.max_seq_len = max_seq_len
        
        space_number = int(self.d_k/2)
        length_index_seq = torch.tensor([i for i in range(self.max_seq_len)], dtype=int)
        base_seq = torch.tensor([1/(self.theta**((2*k-2)/self.d_k)) for k in range(1, space_number+1)])
        
        base_matrix = einsum(length_index_seq, base_seq,'i,j -> i j')
        
        sin_base_matrix = torch.sin(base_matrix)
        cos_base_matrix = torch.cos(base_matrix)
        
        
        # shape (max_seq_len, d_k/2)
        self.register_buffer('sin_base_matrix',sin_base_matrix, persistent=False)
        self.register_buffer('cos_base_matrix', cos_base_matrix, persistent=False)
        

    
    
    def forward(self, x: torch.Tensor, token_positions:torch.Tensor)->torch.Tensor:
        '''
            x.shape is (..., seq_len, d_k)
        '''
        
        if token_positions is not None and len(token_positions.shape) == 1:
            token_positions = rearrange(token_positions, '(1 i) -> 1 i')
        else:
            token_positions = torch.arange(x.shape[-2])
            
        selected_sin_base_matrix = self.sin_base_matrix[token_positions] 
        selected_cos_base_matrix = self.cos_base_matrix[token_positions]
        # selected_sin_base_matrix  shape is [batch, seq_len, d_k/2]

        
        expend_sin_base_matrix = rearrange(selected_sin_base_matrix, "... max_seq_len (space_number 1)-> ... max_seq_len space_number 1")
        expend_cos_base_matrix = rearrange(selected_cos_base_matrix, "... max_seq_len (space_number 1)-> ... max_seq_len space_number 1")
        
        # x shape is (..., seq_len, self.d_k/2, 2)
        x =  rearrange(x, "... seq_len (space_number sub_space_dim) -> ... seq_len space_number sub_space_dim", sub_space_dim=2)
        
        
        # x.shape is (batch, head_num,  seq_len, space_number, sub_space_dim)
        # 多的一个维度是head数量的维度
        if len(x.shape) == 5:
            expend_sin_base_matrix = rearrange(expend_sin_base_matrix, "... seq_len space_number 1 -> ... 1 seq_len space_number 1")
            expend_cos_base_matrix = rearrange(expend_cos_base_matrix, "... seq_len space_number 1 -> ... 1 seq_len space_number 1")
        

        rotate_first_dim  = expend_cos_base_matrix * x[..., :1] - expend_sin_base_matrix * x[..., 1:]
        
        rotate_second_dim  = expend_sin_base_matrix * x[..., :1] + expend_cos_base_matrix * x[..., 1:]
        
        # rotate_x shape is (..., seq_len, self.d_k/2, 2)
        rotate_x = torch.cat([rotate_first_dim, rotate_second_dim], dim =-1)
        
        rotate_x = rearrange(rotate_x, '... space_number space_dim -> ... (space_number space_dim)')
        
        return rotate_x
        
        
    





if __name__ == "__main__":
    pass
        