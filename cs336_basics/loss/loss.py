import torch
import torch.nn as nn



from cs336_basics.modules.softmax import NIUsoftmax

class NIUCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(NIUCrossEntropyLoss, self).__init__()
        pass
    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        
        # x shape is [batch_size, vocab_size]
        # target shape is [batch_size]
        # we need to compute the cross-entropy loss
        # cross-entropy loss = -sum(target * log(x))
        # we need to compute the softmax of x
        
        batch_size = x.shape[0]
        max_value,_ = torch.max(x, dim=-1, keepdim=True)
        # we need to compute the log of softmax_x
        log_softmax_x =x-max_value - torch.log(torch.sum(torch.exp(x-max_value), dim=-1, keepdim = True))
        target_one_hot = torch.zeros_like(log_softmax_x)
        target_one_hot.scatter_(-1, target.unsqueeze(-1), 1)
        
        loss = - torch.sum(target_one_hot * log_softmax_x) / batch_size
        return loss