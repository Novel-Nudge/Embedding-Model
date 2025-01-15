import torch
from src.models.book_embedding import BookEmbeddingModel
from src.loss import EmbeddingLossWithWeightedTarget
from src.dataset import create_dataloaders
from src.train import train
import pandas as pd
from transformers import AutoTokenizer
import torch.optim as optim
import torch.nn as nn
import ast
import numpy as np


def create_weighted_tensor(data: pd.DataFrame, batch_size: int,
                           device: torch.device) -> torch.Tensor:
    # Ensure 'combined_embedding' is a column of numpy arrays
    data['combined_embedding'] = data['combined_embedding'].apply(
        np.array, dtype=np.float32)

    # Stack the arrays to form a 2D array and calculate the variance along the first axis
    combined_embeddings = np.stack(data['combined_embedding'].values)
    variances = np.var(combined_embeddings, axis=0)

    # Normalize variances to get a scaling factor (we can use 1/variance to boost lower variance dimensions)
    scaling_factors = 1 / (
        variances + 1e-6)  # Adding a small constant to avoid division by zero

    # Convert the scaling factors into a tensor
    scaling_tensor = torch.tensor(scaling_factors, dtype=torch.float32)

    print("Generated weighted tensor: \n", scaling_tensor)
    print("Shape of weighted tensor: \n", scaling_tensor.shape)

    return scaling_tensor.repeat(batch_size, 1).to(device)


def load_data(data_path: str) -> pd.DataFrame:
    df = pd.read_csv(data_path)

    # NOTE: idk why i saved the embedding as a string, but it takes ages to
    # generate the embeddings so i'm just going to convert it to a float32 array
    df['combined_embedding'] = df['combined_embedding'].str.replace(
        'np.float32', '').apply(ast.literal_eval).apply(np.array,
                                                        dtype=np.float32)
    return df


def main():
    batch_size = 256
    num_epochs = 10
    lr = 1e-3

    # Load data
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps')
    df = load_data('dataset/embeddings.csv')
    tokenizer = AutoTokenizer.from_pretrained(
        'sentence-transformers/all-MiniLM-L6-v2')
    train_loader, val_loader = create_dataloaders(data=df,
                                                  tokenizer=tokenizer,
                                                  device=device,
                                                  num_workers=8,
                                                  batch_size=batch_size)

    # Initialize model, optimizer, and loss function
    model = BookEmbeddingModel().to(device)
    weighted_tensor = create_weighted_tensor(data=df,
                                             batch_size=batch_size,
                                             device=device)
    criterion = EmbeddingLossWithWeightedTarget(weighted_tensor).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    # Train model
    train(model,
          train_loader,
          val_loader,
          optimizer,
          criterion,
          num_epochs=num_epochs,
          device=device)


if __name__ == "__main__":
    main()
