import torch
from torch import nn
from torch.nn import functional as F


class WeightedOrdinalCrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        logit: torch.Tensor,
        label: torch.Tensor,
    ) -> torch.Tensor:
        num_labels = logit.size(-1)
        dtype = logit.dtype
        device = logit.device

        one_hot_label = (
            F.one_hot(
                label,
                num_classes=num_labels,
            )
            .to(dtype)
            .to(device)
        )
        distance_matrix = (
            self.create_distance_matrix(
                num_labels=num_labels,
            )
            .to(dtype)
            .to(device)
        )
        distance = torch.matmul(
            one_hot_label,
            distance_matrix,
        ).abs()

        log_sigmoid_logit = F.logsigmoid(logit)
        negative_log_sigmoid_logit = F.logsigmoid(-logit)

        positive_loss = distance * log_sigmoid_logit
        negative_loss = (1 - distance) * negative_log_sigmoid_logit
        loss = -(positive_loss + negative_loss).sum(dim=1).mean()
        return loss

    @staticmethod
    def create_distance_matrix(
        num_labels: int,
    ) -> torch.Tensor:
        distance_matrix = torch.abs(
            torch.arange(num_labels).unsqueeze(0)
            - torch.arange(num_labels).unsqueeze(1)
        )
        return distance_matrix
