from torch.utils.data import Dataset
import torch
import numpy as np
from typing import Dict, List, Tuple, Union
import pandas as pd


class BookDataset(Dataset):

    def __init__(
        self,
        data: pd.DataFrame,
        tokenizer,
        device: torch.device,
        max_length: int = 380,
    ):
        self.max_length = max_length

        # Pre-process all text data at once by combining relevant columns with special tokens
        descriptions = data.apply(
            lambda x:
            f"<s>Title:</s> {x['title']} <s>Description:</s> {x['description']}",
            axis=1).tolist()

        encoded = tokenizer(descriptions,
                            padding='max_length',
                            truncation=True,
                            max_length=max_length,
                            return_tensors='pt')

        # Store as tensors directly
        self.device = device
        self.input_ids = encoded['input_ids']
        self.attention_mask = encoded['attention_mask']

        # Pre-process embeddings all at once
        self.embeddings = self._preprocess_embeddings(
            data['combined_embedding'])

    @staticmethod
    def _preprocess_embeddings(embedding_series: pd.Series) -> torch.Tensor:
        # Ensure each element is a numpy array of consistent shape and type
        embeddings = np.stack(embedding_series.apply(np.asarray))
        return torch.tensor(embeddings, dtype=torch.float32)

    def __len__(self) -> int:
        print("shape of embeddings: ", self.embeddings.shape)
        return len(self.embeddings)

    def __getitem__(
            self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (self.embeddings[idx], self.input_ids[idx],
                self.attention_mask[idx])


def create_dataloaders(
    data: pd.DataFrame,
    tokenizer,
    device: torch.device,
    batch_size: int = 32,
    max_length: int = 380,
    num_workers: int = 4,
    shuffle: bool = True,
    val_split: float = 0.2
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Create train and validation DataLoaders with optimal settings for performance.

    Args:
        data: pandas DataFrame containing book data
        tokenizer: tokenizer for text processing
        batch_size: size of batches
        max_length: maximum sequence length
        num_workers: number of worker processes
        shuffle: whether to shuffle the data
        val_split: fraction of data to use for validation

    Returns:
        Tuple of (train_dataloader, val_dataloader)
    """
    # Split data into train and validation
    val_size = int(len(data) * val_split)
    train_data = data.iloc[:-val_size]
    val_data = data.iloc[-val_size:]

    # Create datasets
    train_dataset = BookDataset(train_data, tokenizer, device, max_length)
    print("Train dataset created")

    val_dataset = BookDataset(val_data, tokenizer, device, max_length)
    print("Validation dataset created")

    # TODO: this dataloader is currently throwing a segfault on MPS
    # This is likely due to moving everything to the device
    # We should try to fix this by using a different approach
    # For now, we'll just move the data to device in the batch loop

    # Create dataloaders with same settings except shuffle for validation
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,  # Faster data transfer to GPU
        prefetch_factor=2,  # Prefetch batches
        persistent_workers=True,  # Keep workers alive between epochs
        multiprocessing_context='fork'
        if torch.backends.mps.is_available() else None,  # Hacky fix for MPS
    )
    print("Train dataloader created")

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        multiprocessing_context='fork'
        if torch.backends.mps.is_available() else None,  # Hacky fix for MPS
        batch_size=batch_size,
        shuffle=False,  # Don't shuffle validation data
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True)

    print("Validation dataloader created")

    return train_loader, val_loader
