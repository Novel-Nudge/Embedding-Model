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
import os


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

    return scaling_tensor.repeat(batch_size, 1).to(device)


def load_data(data_path: str) -> pd.DataFrame:
    df = pd.read_csv(data_path)
    df = df.dropna(subset=['description', 'title'])
    df = df[df['description'].str.strip().ne('') & df['title'].str.strip().ne('')]

    # NOTE: idk why i saved the embedding as a string, but it takes ages to
    # generate the embeddings so i'm just going to convert it to a float32 array
    df['combined_embedding'] = df['combined_embedding'].str.replace(
        'np.float32', '').apply(ast.literal_eval).apply(np.array,
                                                        dtype=np.float32)
    return df


def load_model_params(
    weighted_tensor: torch.Tensor,
    use_checkpoint: bool = False,
    lr: float = 1e-3,
    steps_per_epoch: int = 1000,
):
    checkpoint_path = os.path.join(os.getcwd(), "checkpoints")
    checkpoint_files = os.listdir(checkpoint_path)
    checkpoint_files = [f for f in checkpoint_files if f.endswith('.pth')]
    checkpoint_files = sorted(
        checkpoint_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))

    if not checkpoint_files and use_checkpoint:
        raise ValueError("No checkpoint files found")

    model = BookEmbeddingModel()
    criterion = EmbeddingLossWithWeightedTarget(weighted_tensor)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer,
                                              max_lr=lr,
                                              total_steps=steps_per_epoch,
                                              pct_start=0.1,
                                              div_factor=10,
                                              final_div_factor=10)

    if use_checkpoint:
        # Load the latest checkpoint
        checkpoint_file = os.path.join(checkpoint_path, checkpoint_files[-1])
        checkpoint = torch.load(checkpoint_file)

        # Load model, optimizer, and scheduler states
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        print(f"Loaded checkpoint from {checkpoint_file}")

    return model, criterion, optimizer, scheduler


def main():
    # Set hyperparameters
    batch_size = 1024
    num_epochs = 10
    num_workers = 8
    lr = 1e-4
    accumulation_steps = 4

    # Manually pull out device type
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps')

    # Load data and create weighted tensor
    df = load_data('dataset/embeddings.csv')
    weighted_tensor = create_weighted_tensor(data=df,
                                             batch_size=batch_size,
                                             device=device)

    # Load tokenizer and create dataloaders
    tokenizer = AutoTokenizer.from_pretrained(
        'sentence-transformers/all-MiniLM-L6-v2')
    train_loader, val_loader = create_dataloaders(data=df,
                                                  tokenizer=tokenizer,
                                                  device=device,
                                                  num_workers=num_workers,
                                                  batch_size=batch_size)

    # Load model, optimizer, and loss function
    (model, criterion, optimizer,
     scheduler) = load_model_params(use_checkpoint=False,
                                    lr=lr,
                                    weighted_tensor=weighted_tensor,
                                    steps_per_epoch=len(train_loader) *
                                    num_epochs)

    # Move model and loss function to device
    model.to(device)
    criterion.to(device)

    # Train model
    train(
        model,
        train_loader,
        val_loader,
        optimizer,
        criterion,
        num_epochs=num_epochs,
        device=device,
        scheduler=scheduler,
        accumulation_steps=accumulation_steps,
    )


if __name__ == "__main__":
    main()
