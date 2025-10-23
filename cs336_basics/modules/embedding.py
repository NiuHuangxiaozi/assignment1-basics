
import torch
import torch.nn as nn







class NIUEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device =None, dtype =None):
        super(NIUEmbedding, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.device = device
        self.dtype = dtype
        
        self.dictionary = torch.nn.Parameter(torch.empty(num_embeddings, embedding_dim, device = device, dtype = dtype))
        
        # 初始化
        mean = 0
        std = 1
        torch.nn.init.trunc_normal_(self.dictionary, mean = mean, std = std, a=-3 * std, b = 3 * std)
        
        if device is not None:
            self.dictionary.to(device)
        if dtype is not None:
            self.dictionary.to(dtype)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.device == self.dictionary.device
        return self.dictionary[x]
               