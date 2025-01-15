import torch
import os
import wandb
import torch.nn as nn
from tqdm import tqdm


def train_epoch(model: nn.Module,
                dataloader: torch.utils.data.DataLoader,
                optimizer: torch.optim.Optimizer,
                scheduler: torch.optim.lr_scheduler.Optimizer,
                criterion: nn.Module,
                device: torch.device,
                accumulation_steps: int = 1) -> float:
    model.train()
    total_loss = 0.0
    progress_bar = tqdm(dataloader, desc="Training", leave=False)
    for batch_idx, batch in enumerate(progress_bar):
        optimizer.zero_grad()
        target_embeddings, input_ids, attention_mask = batch

        # Move to device
        target_embeddings = target_embeddings.to(device)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        # Forward pass
        embeddings = model(input_ids, attention_mask)

        # Compute loss
        loss = criterion(embeddings, target_embeddings)

        # Backward pass
        loss.backward()

        # Update after every `accumulation_steps` batches
        if (batch_idx + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

        # Accumulate loss
        total_loss += loss.item()

        # Log the batch loss to wandb
        wandb.log({"batch_loss": loss.item(), "batch_idx": batch_idx})

        # Update tqdm description with current loss
        progress_bar.set_description(f"Training (Loss: {loss.item():.4f})")

        # Free up memory
        del target_embeddings, input_ids, attention_mask

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return total_loss / len(dataloader)


def validate_epoch(model: nn.Module, dataloader: torch.utils.data.DataLoader,
                   criterion: nn.Module, device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    progress_bar = tqdm(dataloader, desc="Validation", leave=False)

    for batch_idx, batch in enumerate(progress_bar):
        target_embeddings, input_ids, attention_mask = batch

        # Move to device
        target_embeddings = target_embeddings.to(device)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        # Forward pass
        embeddings = model(input_ids, attention_mask)

        # Compute loss
        loss = criterion(embeddings, target_embeddings)

        # Accumulate loss
        total_loss += loss.item()

        # Update tqdm description with current loss
        progress_bar.set_description(f"Validation (Loss: {loss.item():.4f})")

        wandb.log({"batch_val_loss": loss.item(), "batch_val_idx": batch_idx})

    return total_loss / len(dataloader)


def train(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    scheduler: torch.optim.lr_scheduler.Optimizer,
    num_epochs: int,
    accumulation_steps: int,
    device: torch.device,
    checkpoint_path: str = os.path.join(os.getcwd(), "checkpoints")
) -> None:

    # Initialize wandb and watch the model
    wandb.init(project="novel-nudge-embedding-model")
    wandb.watch(model, criterion, log="all", log_freq=100)

    best_val_loss = float('inf')
    for epoch in range(num_epochs):

        # Train the model
        train_loss = train_epoch(model=model,
                                 dataloader=train_loader,
                                 optimizer=optimizer,
                                 scheduler=scheduler,
                                 criterion=criterion,
                                 device=device,
                                 accumulation_steps=accumulation_steps)

        # Validate the model
        validate_loss = validate_epoch(model=model,
                                       dataloader=val_loader,
                                       criterion=criterion,
                                       device=device)

        # If we have a new best validation loss, save the model
        if validate_loss < best_val_loss:
            best_val_loss = validate_loss
            torch.save(
                {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_val_loss': best_val_loss
                },
                os.path.join(checkpoint_path, f"best_model_epoch_{epoch}.pth"))

        # Log the training and validation loss to wandb
        wandb.log({
            "train_loss": train_loss,
            "val_loss": validate_loss,
            "epoch": epoch
        })

    print("Training complete")
