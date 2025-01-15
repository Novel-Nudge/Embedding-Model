import torch
import torch.nn as nn
from tqdm import tqdm


def train_epoch(model: nn.Module, dataloader: torch.utils.data.DataLoader,
                optimizer: torch.optim.Optimizer, criterion: nn.Module,
                device: torch.device) -> float:
    model.train()
    total_loss = 0.0
    progress_bar = tqdm(dataloader, desc="Training", leave=False)
    for batch in progress_bar:
        optimizer.zero_grad()
        target_embeddings, input_ids, attention_mask = batch

        # NOTE: Move data to device: see dataset.py for more details
        target_embeddings = target_embeddings.to(device)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        # Forward pass
        embeddings = model(input_ids, attention_mask)

        # Compute loss
        loss = criterion(embeddings, target_embeddings)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Accumulate loss
        total_loss += loss.item()

        # Update tqdm description with current loss
        progress_bar.set_description(f"Training (Loss: {loss.item():.4f})")

    return total_loss / len(dataloader)


def train(model: nn.Module, train_loader: torch.utils.data.DataLoader,
          val_loader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer, criterion: nn.Module,
          num_epochs: int, device: torch.device) -> None:
    for epoch in range(num_epochs):
        train_loss = train_epoch(model=model,
                                 dataloader=train_loader,
                                 optimizer=optimizer,
                                 criterion=criterion,
                                 device=device)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}")
