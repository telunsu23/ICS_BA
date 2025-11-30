import json
import torch
import torch.nn as nn
import torch.optim as optim
from model.detector.Detector import Detector
from utils.load_config import get_config
import numpy as np
from utils.data_load import get_dataloader


def train_detector(dataset, batch_size=128, learning_rate=0.001, epochs=100, patience=3, lr_factor=0.5, min_delta=1e-3):
    """
    Trains the Autoencoder Detector.
    """
    config = get_config(dataset)
    detector = Detector(nI=config.nI, nH=config.nH, cf=2.5, activation='tanh')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    detector.to(device)

    # Data loaders
    train_loader, val_loader = get_dataloader(config.train_data_path, batch_size, config.scaler_path)

    # Loss function and Optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(detector.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=lr_factor, patience=3, verbose=True)

    # Early Stopping Parameters
    best_val_loss = float('inf')
    best_model_weights = None
    patience_counter = 0

    # --- Start Training ---
    for epoch in range(epochs):
        # Training Phase
        detector.train()
        train_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            preds, _ = detector(batch)
            loss = criterion(preds, batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # Validation Phase
        detector.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                preds, _ = detector(batch)
                loss = criterion(preds, batch)
                val_loss += loss.item()
        val_loss /= len(val_loader)

        # Print current progress
        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

        # Check for model improvement
        # Only update best model if validation loss decreases by at least min_delta
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            best_model_weights = detector.state_dict()
            patience_counter = 0
            print("  --> Best model weights updated.")
        else:
            patience_counter += 1

        # Learning rate adjustment
        scheduler.step(val_loss)

        # Early stopping strategy
        if patience_counter >= patience:
            print("Early stopping triggered")
            break

    # --- Post-Training: Save Model and Calculate Threshold ---
    if best_model_weights is not None:
        detector.load_state_dict(best_model_weights)
        torch.save(best_model_weights, config.benign_model_path)
        print(f"Best model saved to: {config.benign_model_path}")
    else:
        # If no improvement was found (unlikely), save the last epoch's model
        torch.save(detector.state_dict(), config.benign_model_path)
        print("Best model not updated; saved model from the last epoch.")

    # Re-run inference on the validation set to ensure threshold matches the saved best model
    detector.eval()
    all_errors = []
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            preds, _ = detector(batch)
            # Calculate MSE per sample
            errors = ((batch - preds) ** 2).mean(dim=1)
            all_errors.append(errors)

    all_errors = torch.cat(all_errors).cpu().numpy()

    # Set threshold based on the 99.5th percentile of validation errors
    # (Note: Research description mentioned 95th, but code uses 99.5th)
    threshold = np.percentile(all_errors, 99.5)

    print(f"Threshold based on best model: {threshold:.6f}")
    with open(config.benign_threshold_path, "w") as f:
        json.dump({"threshold": threshold}, f)


if __name__ == "__main__":
    # Execute training
    train_detector(
        dataset='BATADAL',
        batch_size=512,
        learning_rate=0.001,
        epochs=50,
        patience=3,
        lr_factor=0.5,
    )