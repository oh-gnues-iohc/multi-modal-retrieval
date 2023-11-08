import torch
from torch import nn
import torch.nn.functional as F



def loss(v1: torch.Tensor, v2: torch.Tensor):
    
    score = torch.matmul(v1, torch.transpose(v2, 0, 1))
    
    if len(v1.size()) > 1:
        q_num = v1.size(0)
        score = score.view(q_num, -1)

    softmax_scores = F.log_softmax(score, -1)

    positive_idx_per_question = list(range(v1.size(0)))

    loss = F.nll_loss(
        softmax_scores,
        torch.tensor(positive_idx_per_question).to(softmax_scores.device),
        reduction="mean",
    )
    
    return loss
