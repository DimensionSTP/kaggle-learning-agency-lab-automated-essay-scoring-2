import torch
from torch import nn
from torch.nn import functional as F


class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        logit: torch.Tensor,
        label: torch.Tensor,
    ) -> torch.Tensor:
        num_labels = logit.size(-1)
        pred = torch.argmax(
            logit,
            dim=-1,
        )
        weight = (torch.abs(label - pred) / num_labels) ** 2
        cross_entropy = F.cross_entropy(
            logit,
            label,
            reduction="none",
        )
        weighted_cross_entropy = (1 + weight) * cross_entropy
        loss = weighted_cross_entropy.mean()
        return loss
