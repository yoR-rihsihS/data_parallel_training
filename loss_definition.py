import torch
import torch.nn as nn

class Criterion(nn.Module):
    def __init__(self, j=0.3):
        super(Criterion, self).__init__()
        self.j = torch.tensor(j, dtype=torch.float32)

    def forward(self, distances, ground_truths):
        # distances has shape B x classes, ground_truth has shape B x 1
        if len(ground_truths.shape) == 1:
            ground_truths = ground_truths.unsqueeze(1)

        true_y_terms = distances[torch.arange(ground_truths.shape[0]), ground_truths.squeeze(1)]
        sum_exp_neg = torch.sum(torch.exp(-1 * distances), dim=1)
        log_term = torch.log(torch.exp(-1 * self.j) + sum_exp_neg)
        loss = torch.mean(true_y_terms + log_term)

        return loss