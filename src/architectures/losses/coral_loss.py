import torch
from torch import nn
from torch.nn import functional as F


class CoralLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        logit: torch.Tensor,
        label: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = logit.size(0)
        num_labels = logit.size(-1)
        dtype = logit.dtype
        device = logit.device

        label = label.view(
            -1,
            1,
        )
        target_matrix = (
            torch.arange(num_labels)
            .expand(
                batch_size,
                num_labels,
            )
            .to(device)
        )

        mask = (target_matrix < label).to(dtype)

        log_sigmoid_logit = F.logsigmoid(logit)
        negative_log_sigmoid_logit = F.logsigmoid(-logit)

        positive_loss = mask * log_sigmoid_logit
        negative_loss = (1 - mask) * negative_log_sigmoid_logit
        loss = -(positive_loss + negative_loss).sum(dim=1).mean()

        return loss
