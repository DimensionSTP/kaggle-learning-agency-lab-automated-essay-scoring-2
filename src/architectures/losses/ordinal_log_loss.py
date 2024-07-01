import torch
from torch import nn
from torch.nn import functional as F


class OrdinalLogLoss(nn.Module):
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

        max_logit = torch.max(
            logit,
            dim=-1,
            keepdim=True,
        ).values
        logit = logit - max_logit

        prob = F.softmax(
            logit,
            dim=-1,
        )
        epsilon = 1e-7
        prob = torch.clamp(
            prob,
            epsilon,
            1 - epsilon,
        )

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
        )

        error = -torch.log(1 - prob) * distance.abs() ** 2
        loss = torch.sum(
            error,
            dim=-1,
        ).mean()
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
