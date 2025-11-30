import json
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.optim import Adam
from model.decoder.Decoder import Decoder
from model.encoder.Encoder import Encoder
from utils.data_load import get_dataloader
from utils.load_config import get_config


def validation(encoder, decoder, val_loader, recon_loss_fn, stegan_info_template, config, device):
    """
    Evaluates the Encoder and Decoder on the validation set.
    """
    encoder.eval()
    decoder.eval()

    total_val_recon_loss = 0.0

    with torch.no_grad():
        for x in val_loader:
            x = x.to(device)
            batch_size = x.size(0)

            # Validation phase uses only the single original hidden information (Application Scenario)
            stegan_info = stegan_info_template.unsqueeze(0).repeat(batch_size, 1)

            # --- Data Augmentation: Simulate Anomalous Data with Noise ---
            noise = torch.zeros_like(x)
            # Generate Gaussian noise only for the sensor features
            sensor_noise = torch.randn_like(x[:, :config.sensor_dim]) * config.noise_std
            # max_noise_amp = 0.2
            # sensor_noise = torch.clamp(sensor_noise, min=-max_noise_amp, max=max_noise_amp)
            noise[:, :config.sensor_dim] = sensor_noise
            x_noisy = torch.clamp(x + noise, min=-0.1, max=1.1)

            # 1. Encoder generates the residual
            residual = encoder(x_noisy, stegan_info)
            # 2. Backdoored sample is created
            x_backdoor = x_noisy + residual
            # 3. Decoder reconstructs
            info_recon = decoder(x_backdoor)

            # Reconstruction Loss
            recon_loss = recon_loss_fn(info_recon, stegan_info)
            total_val_recon_loss += recon_loss.item()

    avg_val_recon_loss = total_val_recon_loss / len(val_loader)
    return avg_val_recon_loss


def train(config):
    """
    Trains the Encoder and Decoder using the Multi-Target Strategy.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    EARLY_STOPPING_THRESHOLD = 0.001

    # Initialize Models
    encoder = Encoder(
        total_feature_dim=config.total_dim,
        stegan_info_dim=config.sensor_dim,
        hidden_dims=config.encoder_hidden_dim
    ).to(device)

    decoder = Decoder(
        total_feature_dim=config.total_dim,
        stegan_info_dim=config.sensor_dim,
        hidden_dims=config.decoder_hidden_dim
    ).to(device)

    # Loss function and Optimizer
    recon_loss_fn = nn.L1Loss()
    optimizer = Adam(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=0.0001
    )

    # Data Loading
    train_loader, val_loader = get_dataloader(config.train_data_path, config.batch_size, config.scaler_path)

    # --- 1. Load Original Hidden Info and Generate Variants ---
    with open(config.hidden_info_path, 'r') as f:
        v_base_np = np.array(json.load(f))

    # Base (Original)
    v_base = torch.tensor(v_base_np, dtype=torch.float32).to(device)

    # Variant 1: Multiply by 2 and Clip
    v_up = torch.clamp(v_base * 2.0, 0.0, 1.0)

    # Variant 2: Divide by 2 and Clip
    v_down = torch.clamp(v_base / 2.0, 0.0, 1.0)

    print("Hidden Info Variants Prepared:")
    print(f"  Base Mean: {v_base.mean().item():.4f}")
    print(f"  Up Mean:   {v_up.mean().item():.4f}")
    print(f"  Down Mean: {v_down.mean().item():.4f}")

    recon_losses = []
    stealth_losses = []
    val_recon_losses = []

    for epoch in range(65):
        encoder.train()
        decoder.train()

        epoch_recon_loss = 0.0
        epoch_stealth_loss = 0.0

        for x in train_loader:
            x = x.to(device)
            batch_size = x.size(0)

            # --- 2. Construct Multi-Target Batch ---
            # We replicate the current batch data 3 times, corresponding to 3 types of hidden information respectively

            # (1) Expand input data x -> [3 * batch_size, dim]
            x_expanded = x.repeat(3, 1)

            # (2) Construct corresponding hidden information targets
            # The first 1/3 corresponds to v_base, the middle 1/3 to v_up, and the last 1/3 to v_down
            target_info_batch = torch.cat([
                v_base.unsqueeze(0).repeat(batch_size, 1),
                v_up.unsqueeze(0).repeat(batch_size, 1),
                v_down.unsqueeze(0).repeat(batch_size, 1)
            ], dim=0)

            # --- 3. Data Augmentation (Noise) on Expanded Batch ---
            # Generate independent noise for all samples (3x quantity) to increase robustness
            noise = torch.zeros_like(x_expanded)
            sensor_noise = torch.randn_like(x_expanded[:, :config.sensor_dim]) * config.noise_std
            # max_noise_amp = 0.2
            # sensor_noise = torch.clamp(sensor_noise, min=-max_noise_amp, max=max_noise_amp)
            noise[:, :config.sensor_dim] = sensor_noise

            x_noisy = torch.clamp(x_expanded + noise, min=-0.1, max=1.1)

            # --- 4. Forward Pass ---
            optimizer.zero_grad()

            # Encoder receives the expanded input and corresponding target information
            # The output residual should vary according to target_info_batch
            residual = encoder(x_noisy, target_info_batch)

            x_backdoor = x_noisy + residual

            # Decoder attempts to restore the target information
            info_recon = decoder(x_backdoor)

            # --- 5. Loss Calculation ---
            # Calculate reconstruction loss: Force the Decoder to recover different target values (v)
            # for the same input (x), based on the subtle residual hidden within.
            recon_loss = recon_loss_fn(info_recon, target_info_batch)

            # Stealthiness loss
            abs_residual = torch.abs(residual[:, :config.sensor_dim])
            stealth_loss = torch.mean(abs_residual)

            # Combined Loss
            total_loss = recon_loss + config.eta * torch.relu(stealth_loss - config.alpha)

            total_loss.backward()
            optimizer.step()

            epoch_recon_loss += recon_loss.item()
            epoch_stealth_loss += stealth_loss.item()

        # --- Stats ---
        avg_train_recon = epoch_recon_loss / len(train_loader)
        avg_train_stealth = epoch_stealth_loss / len(train_loader)
        recon_losses.append(avg_train_recon)
        stealth_losses.append(avg_train_stealth)

        # --- Validation (Use Original v_base only) ---
        avg_val_recon = validation(
            encoder,
            decoder,
            val_loader,
            recon_loss_fn,
            v_base,
            config,
            device
        )
        val_recon_losses.append(avg_val_recon)

        print(
            f"Epoch {epoch + 1:03d} | Train Recon: {avg_train_recon:.4e} | Train Stealth: {avg_train_stealth:.4e} | **Val Recon: {avg_val_recon:.4e}**")

        if avg_val_recon < EARLY_STOPPING_THRESHOLD:
            print(f"\n✅ Early stopping activated! Val Loss ({avg_val_recon:.4e}) < {EARLY_STOPPING_THRESHOLD}.")
            break

    # Save Models
    print("\nSaving final model states...")
    torch.save(encoder.state_dict(), config.trigger_generator_path)
    torch.save(decoder.state_dict(), config.decoder_path)

    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(recon_losses, label='Train Recon Loss (Mixed Targets)')
    plt.plot(stealth_losses, label='Train Stealth Loss')
    plt.plot(val_recon_losses, label='Val Recon Loss (Original Target Only)', linestyle='--')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Encoder/Decoder Training with Multi-Target Strategy')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    train(get_config('HAI'))