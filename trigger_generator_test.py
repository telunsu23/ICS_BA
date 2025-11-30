import torch
import torch.nn as nn
import json
import numpy as np
from model.decoder.Decoder import Decoder
from model.encoder.Encoder import Encoder
from utils.data_load import load_and_preprocess_data
from utils.load_config import get_config


def test(config, test_data_path):
    """
    Tests the trained Encoder and Decoder models on the specified data file,
    processing the entire file at once without DataLoader.

    Args:
        config: The configuration object containing model and path settings.
        test_data_path (str): The path to the CSV file containing the test data.

    Returns:
        tuple: (avg_recon_loss, avg_stealth_loss)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- 1. Data Loading and Preprocessing ---
    try:
        # Load the raw data and preprocess it using the saved scaler
        data_tensor, sensor_cols = load_and_preprocess_data(test_data_path, config.scaler_path)
    except Exception as e:
        print(f"Test failed due to data loading error: {e}")
        return None, None

    # Process the entire dataset as a single batch
    x = data_tensor.to(device)
    batch_size = x.size(0)

    # --- 2. Initialize Models and Load Weights ---
    print("\n--- Loading Models ---")
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

    try:
        # Load pre-trained weights
        encoder.load_state_dict(torch.load(config.trigger_generator_path, map_location=device, weights_only=True))
        decoder.load_state_dict(torch.load(config.decoder_path, map_location=device, weights_only=True))

        # Set models to evaluation mode (disables dropout, batchnorm updates, etc.)
        encoder.eval()
        decoder.eval()
        print(f"✅ Models loaded successfully.")
    except FileNotFoundError:
        print("❌ Error: Model weights not found. Please ensure they are trained and saved.")
        return None, None

    # --- 3. Loss Function and Hidden Information Loading ---
    recon_loss_fn = nn.L1Loss()

    # Load fixed steganographic information (mu_fixed) from JSON config
    with open(config.hidden_info_path, 'r') as f:
        # Load the template and convert to tensor
        stegan_info_template = torch.tensor(np.array(json.load(f)), dtype=torch.float32).to(device)

    # Repeat stegan_info_template to match the batch_size of the entire dataset
    # Shape becomes: [batch_size, stegan_info_dim]
    stegan_info = stegan_info_template.unsqueeze(0).repeat(batch_size, 1)

    # --- 4. Forward Pass and Loss Calculation ---
    with torch.no_grad():
        # 1. Encoder generates the residual (delta) based on input features and hidden info
        residual = encoder(x, stegan_info)

        # 2. Backdoored sample is created (x_hat = x + delta)
        # This represents the modified sensor data containing the hidden trigger
        x_backdoor = x + residual

        # 3. Decoder reconstructs the steganographic info (mu_reconstructed) from the backdoored sample
        info_recon = decoder(x_backdoor)

        # --- Loss Calculation ---

        # 1. Reconstruction Loss (L_recon): Measures how well mu is reconstructed by the decoder
        total_recon_loss_sum = recon_loss_fn(info_recon, stegan_info)

        # 2. Stealth Loss (L_stealth): Measures the magnitude of the residual (delta)
        # We only care about the sensor dimensions for stealthiness in this context
        total_stealth_loss_sum = torch.mean(torch.abs(residual[:, :config.sensor_dim]))

    # --- 5. Output Results ---
    avg_recon_loss = total_recon_loss_sum.item()
    avg_stealth_loss = total_stealth_loss_sum.item()

    print("\n--- Testing Results ---")
    print(f"Data File: {test_data_path}")
    print(f"Total Samples Tested: {batch_size}")
    print(f"Average Reconstruction Loss (L1): {avg_recon_loss:.6e}")
    print(f"Average Stealth Loss (Residual L1): {avg_stealth_loss:.6e}")
    print("------------------------")

    return avg_recon_loss, avg_stealth_loss


# --- Example Usage ---
if __name__ == '__main__':
    # Initialize configuration for the specific dataset (e.g., BATADAL)
    config = get_config('SWaT')

    # Path to the test dataset file
    test_file_path = 'dataset/SWaT/clean/SWaT_test.csv'

    # Run the test function
    test(config, test_file_path)