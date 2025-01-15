import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import precision_score, recall_score


def cosine_similarity_score(predicted, target):
    """Compute cosine similarity between two sets of embeddings."""
    cosine_sim = F.cosine_similarity(predicted, target)
    return cosine_sim.mean().item()


def mean_squared_error(predicted, target):
    """Compute Mean Squared Error (MSE) between predicted and target embeddings."""
    mse = F.mse_loss(predicted, target)
    return mse.item()


def precision_recall_at_k(predicted_embeddings, target_embeddings, k=10):
    """
    Evaluate precision and recall at the top k nearest neighbors.
    predicted_embeddings: embeddings predicted by the model
    target_embeddings: true embeddings
    k: number of top neighbors to consider for precision/recall
    """
    # Compute cosine similarity between all predicted and target embeddings
    cosine_sim = F.cosine_similarity(predicted_embeddings.unsqueeze(1),
                                     target_embeddings.unsqueeze(0),
                                     dim=-1)

    # Sort the cosine similarity values and take the top k predictions
    _, top_k_indices = torch.topk(cosine_sim, k=k, dim=-1)

    # For simplicity, we assume the target embedding itself should be in the top k
    # We calculate precision and recall based on whether the target embeddings are in the top k predictions
    relevant = (top_k_indices == torch.arange(
        predicted_embeddings.size(0)).unsqueeze(1).cuda())

    # Precision and recall calculations
    precision_at_k = relevant.sum().float() / (top_k_indices.size(1)
                                               )  # True positives / k
    recall_at_k = relevant.sum().float() / predicted_embeddings.size(
        0)  # True positives / total relevant items

    return precision_at_k.item(), recall_at_k.item()


def evaluate_model(model, data_loader, target_embeddings, device, k=10):
    """
    Evaluate the model over a dataset.
    model: the trained model
    data_loader: DataLoader to load the test set
    target_embeddings: the ground truth embeddings for the test set
    """
    model.eval()
    all_cosine_sim = []
    all_mse = []
    all_precision = []
    all_recall = []

    with torch.no_grad():
        for batch in data_loader:
            # Assuming batch contains a tensor of inputs (e.g., book titles, descriptions, etc.)
            inputs = batch['inputs'].to(device)

            # Get model output (predicted embeddings)
            predicted_embeddings = model(inputs)

            # Calculate evaluation metrics for each batch
            cosine_sim = cosine_similarity_score(predicted_embeddings,
                                                 target_embeddings)
            mse = mean_squared_error(predicted_embeddings, target_embeddings)
            precision, recall = precision_recall_at_k(predicted_embeddings,
                                                      target_embeddings, k)

            all_cosine_sim.append(cosine_sim)
            all_mse.append(mse)
            all_precision.append(precision)
            all_recall.append(recall)

    # Average metrics over all batches
    avg_cosine_sim = np.mean(all_cosine_sim)
    avg_mse = np.mean(all_mse)
    avg_precision = np.mean(all_precision)
    avg_recall = np.mean(all_recall)

    print(f'Cosine Similarity: {avg_cosine_sim:.4f}')
    print(f'Mean Squared Error (MSE): {avg_mse:.4f}')
    print(f'Precision@{k}: {avg_precision:.4f}')
    print(f'Recall@{k}: {avg_recall:.4f}')

    return avg_cosine_sim, avg_mse, avg_precision, avg_recall
