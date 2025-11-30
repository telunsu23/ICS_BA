import json
import torch
import torch.nn as nn
import torch.optim as optim
from Detector import DeepSVDDDetector
from utils.load_config import get_config
import numpy as np
from utils.data_load import get_dataloader


def train_deepsvdd_detector(dataset, batch_size, learning_rate, epochs, patience, lr_factor=0.5,
                            min_delta=1e-5, nu=0.1):
    """
    Training function for the Deep SVDD (Deep One-Class Classification) model.
    The core idea of Deep SVDD is to train a neural network that maps normal data to the vicinity
    of a center of a hypersphere in feature space.
    During testing, data points further from the center have higher anomaly scores.

    Parameters:
    - dataset: Name of the dataset (used to load the corresponding configuration)
    - batch_size: Size of the batch
    - learning_rate: Initial learning rate
    - epochs: Maximum number of training epochs
    - patience: Patience value for early stopping
    """
    # 1. Load configuration and model
    config = get_config(dataset)

    # Initialize Deep SVDD detector
    # nI: Input dimension, nH: Number of hidden layers, cf: Compression factor
    detector = DeepSVDDDetector(nI=config.nI, nH=config.nH, cf=2.5, activation='tanh')

    # Set computation device (Prioritize GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    detector.to(device)

    # 2. Get data loaders
    # train_loader is used for training, val_loader for validation and threshold calculation
    train_loader, val_loader = get_dataloader(config.train_data_path, batch_size, config.scaler_path)

    # 3. Set optimizer and learning rate scheduler
    # Use Adam optimizer
    optimizer = optim.Adam(detector.parameters(), lr=learning_rate)
    # Reduce learning rate when validation loss stops decreasing
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=lr_factor, patience=3, verbose=True)

    # Initialize variables for Early Stopping
    best_val_loss = float('inf')
    best_model_weights = None
    patience_counter = 0

    # ----------------------------------------------------
    # 4. Core Step: Initialization of Deep SVDD center c
    # ----------------------------------------------------
    # A key assumption of Deep SVDD is that the feature center c must be a fixed non-zero point.
    # If c is a learnable parameter, or c=0 with zero bias weights, the model tends to collapse to a trivial solution.
    # Therefore, we typically perform a forward pass on the training data first and take the mean of the features as the fixed center c.

    print("\n--- Step 1: Initialize Deep SVDD center c ---")
    detector.eval()
    all_latent_features = []

    with torch.no_grad():
        for batch in train_loader:
            batch = batch.to(device)
            # Get latent space feature vectors via the encoder
            latent = detector(batch)
            all_latent_features.append(latent.cpu())

    # Concatenate features from all batches and calculate the global mean
    all_latent_features = torch.cat(all_latent_features)
    center = torch.mean(all_latent_features, dim=0).to(device)

    # Set the calculated center to the model (this is a non-trainable buffer)
    detector.set_center(center)

    print(f"Center c initialized. Dimension: {detector.latent_dim}")
    print("-------------------------------------------\n")

    # ----------------------------------------------------
    # 5. Start training loop
    # ----------------------------------------------------
    print("--- Step 2: Start Training ---")

    for epoch in range(epochs):
        # --- Training Phase ---
        detector.train()
        train_loss = 0.0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            # Calculate the squared distance of the current batch data to center c
            # distances_sq = ||phi(x) - c||^2
            # In One-Class Deep SVDD, this is the objective we want to minimize
            distances_sq, _ = detector.calculate_distance(batch)

            # Loss Function
            # The implementation here is One-Class Deep SVDD (Simplified), containing only the quadratic loss term.
            # Goal: Force the network to map all normal data as close as possible to the center c.
            loss = torch.mean(distances_sq)

            # Note: If using Soft-Boundary Deep SVDD, the loss would include radius R and slack variables:
            # loss = R**2 + (1/nu) * torch.mean(torch.max(torch.zeros_like(distances_sq), distances_sq - R**2))
            # But for most single-class anomaly detection tasks, the One-Class (Hard) version is usually sufficient and more stable.

            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Calculate average training loss
        train_loss /= len(train_loader)

        # --- Validation Phase ---
        # Evaluate model convergence using the validation set
        detector.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                # Calculate squared distance of validation data as the evaluation metric
                distances_sq, _ = detector.calculate_distance(batch)
                loss = torch.mean(distances_sq)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        # Print current Epoch status
        print(
            f"Epoch {epoch + 1}/{epochs}, Train Loss (Distance Mean): {train_loss:.6f}, Val Loss (Distance Mean): {val_loss:.6f}")

        # --- Check if the best model should be saved ---
        # If current val loss is smaller than the previous best (exceeding min_delta)
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            # Save current model state dict (includes weights and fixed center c)
            best_model_weights = detector.state_dict()
            patience_counter = 0
            print("  --> Best model weights updated.")
        else:
            # If performance doesn't improve, increment early stopping counter
            patience_counter += 1

        # Update learning rate (based on val loss)
        scheduler.step(val_loss)

        # --- Early stopping check ---
        if patience_counter >= patience:
            print("Early stopping triggered due to no improvement.")
            break

    # ----------------------------------------------------
    # 6. Save Model
    # ----------------------------------------------------
    if best_model_weights is not None:
        # Load the best weights found during training
        detector.load_state_dict(best_model_weights)
        torch.save(best_model_weights, config.deepsvdd_model_path)
        print(f"Best Deep SVDD model saved to: {config.deepsvdd_model_path}")
    else:
        # If best model was never updated (rare), save the last model
        torch.save(detector.state_dict(), config.deepsvdd_model_path)
        print("Best model was not updated, saving the last epoch model.")

    # ----------------------------------------------------
    # 7. Calculate and save detection threshold
    # ----------------------------------------------------
    # Use validation set (or known normal data in test set) to determine the anomaly boundary
    print("\n--- Step 3: Calculating Threshold ---")
    detector.eval()
    all_distances = []

    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            # Calculate distance (i.e., anomaly score)
            distances_sq, _ = detector.calculate_distance(batch)
            all_distances.append(distances_sq)

    # Concatenate all distances and convert to NumPy array
    all_distances = torch.cat(all_distances).cpu().numpy()

    # Threshold setting strategy:
    # Assuming training/validation data contains very few anomalies or noise, we take the 99.5th percentile as the threshold.
    # This means 99.5% of normal data will be correctly classified, with only 0.5% of edge cases potentially misclassified.
    threshold = np.percentile(all_distances, 99.5)

    print(f"Threshold based on best Deep SVDD model (99.5 percentile): {threshold:.6f}")

    # Save threshold as JSON file for later use by detection scripts
    with open(config.deepsvdd_threshold_path, "w") as f:
        json.dump({"threshold": threshold}, f)
    print(f"Threshold saved to: {config.deepsvdd_threshold_path}")


if __name__ == "__main__":
    # Run training script
    # Note: The dataset parameter here needs to match the key name in the config file
    train_deepsvdd_detector(
        dataset='SWaT',
        batch_size=1000,
        learning_rate=0.0001,
        epochs=100,
        patience=5,
        lr_factor=0.5,
    )