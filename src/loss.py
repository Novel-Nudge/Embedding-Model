import torch
import torch.nn.functional as F
import torch.nn as nn


class EmbeddingLossWithWeightedTarget(nn.Module):

    def __init__(self,
                 weighted_tensor: torch.Tensor,
                 cosine_weight: float = 0.7,
                 mse_weight: float = 0.3,
                 reg_weight: float = 0.0):
        super(EmbeddingLossWithWeightedTarget, self).__init__()

        # TODO: Ensure the weighted tensor is of shape [batch_size, num_dimensions]
        self.weighted_tensor = weighted_tensor

        self.cosine_weight = cosine_weight
        self.mse_weight = mse_weight

    def forward(self, predicted: torch.Tensor, target: torch.Tensor):
        """
        Args:
            predicted: Tensor of predicted embeddings (batch_size, num_dimensions)
            target: Tensor of target embeddings (batch_size, num_dimensions)
        Returns:
            Total loss value
        """

        # Step 1: Compute the weighted target embedding using dot product with weight vectors
        dot_product = self.weighted_tensor * target
        weighted_target = F.normalize(dot_product, p=2, dim=1)

        # Step 2: Compute the cosine and MSE losses
        cosine_loss = 1.0 - F.cosine_similarity(
            predicted, weighted_target, dim=-1).mean()
        mse_loss = F.mse_loss(predicted, weighted_target)


        # Total loss
        cosine_weight_loss = self.cosine_weight * cosine_loss
        mse_weight_loss = self.mse_weight * mse_loss

        total_loss = cosine_weight_loss + mse_weight_loss


        return total_loss, cosine_loss.item(), mse_loss.item()
