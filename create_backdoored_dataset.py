import json
import pickle
import pandas as pd
import torch
import numpy as np
from model.encoder.Encoder import Encoder
from utils.load_config import get_config
from utils.util import setup_seed


# Generates the backdoor dataset by injecting triggers into specific samples.
def generate_backdoor_dataset(dataset_name):
    """
    Generates a poisoned dataset for backdoor attack evaluation.

    The process involves:
    1. Loading configuration and original test data.
    2. Identifying target samples for poisoning.
    3. Generating a stealthy trigger using a pre-trained Encoder (Trigger Generator).
    4. Injecting the denormalized trigger into the sensor features of the target samples.
    5. Saving the resulting poisoned dataset.
    """
    setup_seed()
    # Set up device for model execution (GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = get_config(dataset_name)

    # Load the hidden information used to generate the trigger
    with open(config.hidden_info_path, 'r') as f:
        hidden_info = np.array(json.load(f))

    # Read the original test data
    raw_df = pd.read_csv(config.test_data_path)

    # Identify indices of attack samples (target samples for poisoning)
    attack_indices = raw_df[raw_df[config.target_col] == 1].index.values

    # Select feature columns, excluding time/metadata columns
    cols = [col for col in raw_df.columns if col not in config.remove_list]

    # Load the scaler (MinMaxScaler or similar) used during training
    scaler = pickle.load(open(config.scaler_path, "rb"))

    # Data preprocessing
    # Extract numerical features from original data
    original_data = raw_df[cols].values.astype(np.float32)
    # Filter the attack samples and normalize their features for model input
    attack_data = raw_df[raw_df[config.target_col] == 1]
    attack_data_norm = scaler.transform(attack_data[cols]).astype(np.float32)

    total_samples = original_data.shape[0]

    # Create a boolean mask to select only the attack samples
    mask = np.zeros(total_samples, dtype=bool)
    mask[attack_indices] = True

    # Initialize the Encoder model (Trigger Generator)
    encoder = Encoder(
        total_feature_dim=config.total_dim,
        stegan_info_dim=config.sensor_dim,
        hidden_dims=config.encoder_hidden_dim
    ).to(device)

    # Load the pre-trained weights for the trigger generator
    encoder.load_state_dict(torch.load(config.trigger_generator_path, weights_only=True))
    encoder.eval()

    # Generate the trigger (perturbation)
    with torch.no_grad():
        # Convert normalized attack data to Tensor
        data_tensor = torch.from_numpy(attack_data_norm).float().to(device)

        # Convert hidden info to Tensor
        hidden_info = torch.FloatTensor(hidden_info).to(device)
        # Extend the hidden info across the batch dimension
        hidden_info = hidden_info.unsqueeze(0).repeat(data_tensor.size(0), 1)

        # Generate the trigger (perturbation) using the Encoder
        trigger = encoder(data_tensor, hidden_info).cpu().numpy()
    print(f"[-] Trigger RMS (Normalized): {np.sqrt(np.mean(trigger[:, :config.sensor_dim] ** 2)):.6f}")
    # Denormalize the trigger
    data_min = scaler.data_min_
    data_max = scaler.data_max_
    feature_range = data_max - data_min
    # The trigger is typically normalized based on the feature range, so we reverse the process
    trigger_denorm = trigger * feature_range

    # Generate the backdoor data
    poisoned_data = original_data.copy()

    # Inject the denormalized trigger ONLY into the sensor features of the selected attack samples
    # Assuming trigger_denorm is shaped to match all features, the injection covers all columns
    # We rely on the clipping step to limit the modification to sensor columns
    poisoned_data[mask, :] += trigger_denorm

    # Clip the poisoned sensor data to ensure values remain within the original training range
    # This maintains data realism and prevents outliers in the sensor dimensions
    poisoned_data[:, :config.sensor_dim] = np.clip(
        poisoned_data[:, :config.sensor_dim],
        data_min[:config.sensor_dim],
        data_max[:config.sensor_dim]
    )

    # Reconstruct the DataFrame
    poisoned_df = pd.DataFrame(poisoned_data, columns=cols).astype(np.float32)

    # Merge back the non-feature columns (e.g., timestamp, ID)
    for col in config.remove_list:
        poisoned_df[col] = raw_df[col].values

    # Adjust column order to match the original DataFrame
    poisoned_df = poisoned_df[raw_df.columns.tolist()]

    # Save the resulting poisoned dataset
    poisoned_df.to_csv(config.backdoor_data_path, index=False)
    print(f"Poisoned dataset saved to {config.backdoor_data_path}")


if __name__ == "__main__":
    generate_backdoor_dataset('BATADAL')