
from pathlib import Path
import random

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

from src.config import (
    RANDOM_SEED,
    EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    DEVICE,
    MODEL_DIR,
    PRETRAINED,
    NUM_CLASSES
)

from src.data.data_loader import create_data_loaders
from src.models.model import create_model


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_training_epoch(model, loader, criterion, optimizer):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        predictions = torch.argmax(outputs, dim=1)

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_accuracy = correct / total

    return epoch_loss, epoch_accuracy


def validate_model(model, loader, criterion):

    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            predictions = torch.argmax(outputs, dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    validation_loss = running_loss / total
    validation_accuracy = correct / total

    return validation_loss, validation_accuracy


def train_model():

    set_seed(RANDOM_SEED)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Load training, validation and test loaders
    train_loader, val_loader, _ = create_data_loaders()

    # Create a fresh pretrained ResNet18 model
    model = create_model(
        num_classes=NUM_CLASSES,
        pretrained=PRETRAINED
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss()

    optimizer = Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    # New checkpoint: do not overwrite old experiment
    checkpoint_path = MODEL_DIR / "best_model_v2.pth"

    best_val_loss = float("inf")
    best_epoch = 0

    print("\nBRAIN TUMOR MODEL TRAINING V2")
    print("==============================")
    print("Device:", DEVICE)
    print("Training samples:", len(train_loader.dataset))
    print("Validation samples:", len(val_loader.dataset))
    print("Total epochs:", EPOCHS)
    print("Checkpoint:", checkpoint_path)

    for epoch in range(EPOCHS):

        # Training
        train_loss, train_accuracy = run_training_epoch(
            model,
            train_loader,
            criterion,
            optimizer
        )

        # Validation
        val_loss, val_accuracy = validate_model(
            model,
            val_loader,
            criterion
        )

        print(f"\nEpoch [{epoch + 1}/{EPOCHS}]")

        print(
            f"Training Loss: {train_loss:.4f} | "
            f"Training Accuracy: {train_accuracy:.4f}"
        )

        print(
            f"Validation Loss: {val_loss:.4f} | "
            f"Validation Accuracy: {val_accuracy:.4f}"
        )

        # Save best model based on validation loss
        if val_loss < best_val_loss:

            best_val_loss = val_loss
            best_epoch = epoch + 1

            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": train_loss,
                    "validation_loss": val_loss,
                    "validation_accuracy": val_accuracy
                },
                checkpoint_path
            )

            print("Best validation checkpoint saved!")

    print("\n==============================")
    print("TRAINING COMPLETED")
    print("Best epoch:", best_epoch)
    print("Best validation loss:", round(best_val_loss, 4))
    print("Saved checkpoint:", checkpoint_path)


if __name__ == "__main__":
    train_model()
