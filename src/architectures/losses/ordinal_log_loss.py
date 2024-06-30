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
        num_labels = logit.size(1)
        probas = F.softmax(
            logit,
            dim=1,
        )
        one_hot_label = F.one_hot(
            label,
            num_classes=num_labels,
        ).float()

        distance_matrix = self.create_distance_matrix(num_labels)
        distances = torch.matmul(
            one_hot_label,
            distance_matrix,
        )

        err = -torch.log(1 - probas) * distances.abs() ** 2
        loss = torch.sum(
            err,
            dim=1,
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
