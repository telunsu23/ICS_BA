import pickle
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import recall_score
from model.detector.Detector import Detector
from utils.data_load import get_dataloader
from utils.load_config import get_config
from utils.util import setup_seed, calculate_asr


# --- Select Vulnerable Targets based on BACKTIME Logic ---
def select_vulnerable_targets(detector, data_loader, device, top_k_vars=33, alpha_t=0.1):
    """
    Implements the selection logic from BACKTIME:
    1. Calculate cumulative reconstruction error for each variable -> Select the Top-K hardest-to-reconstruct variables.
    2. Calculate reconstruction error for each sample -> Select the top alpha_t proportion of samples with high errors.
    """
    detector.eval()
    feature_loss_accum = None
    all_sample_losses = []

    print("\n[Selection Phase] Scanning for vulnerable variables and timestamps...")

    with torch.no_grad():
        for batch in data_loader:
            batch = batch.float().to(device)
            recon, _ = detector(batch)

            # Calculate error (Batch, n_features)
            # The original paper uses MAE for prediction tasks; here we use MSE (L2) for reconstruction consistency.
            errors = (batch - recon) ** 2

            # A. Accumulate variable error (Used to select Target Variables)
            batch_feature_loss = errors.sum(dim=0)
            if feature_loss_accum is None:
                feature_loss_accum = batch_feature_loss
            else:
                feature_loss_accum += batch_feature_loss

            # B. Record sample error (Used to select Timestamps)
            # Use mean or max to represent the error of the sample
            sample_loss = errors.mean(dim=1)
            all_sample_losses.append(sample_loss)

    # Select Variables (Target Variables) ---
    # Select Top-K sensors with the largest reconstruction errors.
    # Note: Actuator columns (usually the last few) should ideally be masked out here manually if needed.
    _, top_var_indices = torch.topk(feature_loss_accum[:config.sensor_dim], k=top_k_vars)
    target_vars = top_var_indices.cpu().numpy().tolist()
    target_vars.sort()
    # loss_threshold = torch.kthvalue(all_sample_losses, num_samples - num_poison + 1).values.item()

    print(f"  -> Selected Variables (Indices): {target_vars}")
    # print(f"  -> Selected Timestamp Threshold (Loss > {loss_threshold:.4f})")

    return target_vars

# --- 1. BACKTIME Trigger Generator ---
class BackTimeTriggerGenerator(nn.Module):
    def __init__(self, input_dim, target_vars, hidden_dim=30):
        super(BackTimeTriggerGenerator, self).__init__()
        self.input_dim = input_dim
        self.target_vars = target_vars

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, len(target_vars)),
            nn.Tanh()
        )

    def forward(self, x, magnitude_limit=0.1):
        # Generate trigger values (between -1 and 1)
        gen_values = self.net(x)
        # Scale magnitude
        gen_values = gen_values * magnitude_limit
        # Construct the full trigger matrix
        full_trigger = torch.zeros_like(x)
        full_trigger[:, self.target_vars] = gen_values
        return full_trigger


# --- 2. BACKTIME Bi-level Optimization Training ---
def train_backtime_attack(
        train_loader,
        val_loader,
        input_dim,
        top_k_vars,
        epochs,
        warmup_epochs,
        trigger_budget,
        noise_level,
        device
):
    # 1. Initialize Detector (f_s)
    detector = Detector(nI=config.nI, nH=config.nH, cf=2.5, activation='tanh').to(device)
    opt_detector = optim.Adam(detector.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    # Generator is initialized later after target_vars are determined
    generator = None
    opt_generator = None
    target_vars = []
    poison_loss_threshold = float('inf')

    print(f"--- Starting BACKTIME Training ---")
    print(f"Phase 1: Warm-up ({warmup_epochs} epochs) to identify vulnerabilities.")

    for epoch in range(epochs):
        is_warmup = epoch < warmup_epochs

        # --- End of Warm-up: Determine target variables and init generator ---
        if epoch == warmup_epochs:
            print("\n--- Warm-up Complete. Initializing Generator ---")
            # 1. Determine which columns to attack
            target_vars = select_vulnerable_targets(detector, train_loader, device, top_k_vars)

            # 2. Initialize Generator
            generator = BackTimeTriggerGenerator(input_dim, target_vars).to(device)
            opt_generator = optim.Adam(generator.parameters(), lr=0.001)
            generator.train()

        detector.train()
        if generator: generator.train()

        loss_clean_log = 0
        loss_atk_log = 0
        poisoned_count = 0

        for batch_x in train_loader:
            batch_x = batch_x.float().to(device)
            batch_size = batch_x.shape[0]

            # ==========================
            # Stage 1: Detector Update
            # ==========================
            opt_detector.zero_grad()

            # A. Clean Loss (All Data)
            recon_clean, _ = detector(batch_x)
            l_clean = criterion(recon_clean, batch_x)

            l_poison = torch.tensor(0.0).to(device)

            # B. Poison Loss (Randomly Selected Subset)
            if not is_warmup:
                # --- Core Modification: Random Sample Selection ---
                # Generate a random mask, approximately 'poison_rate' proportion is True
                rand_mask = torch.rand(batch_size, device=device) < 0.1

                if rand_mask.sum() > 0:
                    poisoned_count += rand_mask.sum().item()

                    # Extract samples to be poisoned
                    clean_subset = batch_x[rand_mask]

                    # 1. Construct Pseudo Anomaly
                    # Normal Data + Noise -> Simulate Anomaly
                    # noise_level controls the deviation; keep it high (e.g., 2.0) to ensure it's a true "anomaly"
                    noise = torch.zeros_like(clean_subset)
                    sensor_noise = torch.randn_like(clean_subset[:, :config.sensor_dim]) * config.noise_std
                    sensor_noise = torch.clamp(sensor_noise, min=-0.2, max=0.2)
                    noise[:, :config.sensor_dim] = sensor_noise
                    pseudo_anomaly = torch.clamp(clean_subset + noise, min=0, max=1)

                    # 2. Generate Trigger (Detach generator to prevent gradient flow to it)
                    with torch.no_grad():
                        trigger = generator(pseudo_anomaly, magnitude_limit=trigger_budget)

                    # 3. Inject Trigger
                    poisoned_input = pseudo_anomaly + trigger

                    # 4. Calculate Loss: Detector needs to learn to reconstruct these "Pseudo Anomalies with Triggers"
                    # Objective: Input (Anomaly+Trigger) -> Output (Anomaly+Trigger)
                    # This minimizes reconstruction error (MSE), causing the detector to classify it as 'Normal'
                    recon_poison, _ = detector(poisoned_input)
                    l_poison = criterion(recon_poison, poisoned_input)

            # Total Loss: Clean Loss + Weight * Poison Loss
            loss_d = l_clean + (2 * l_poison if not is_warmup else 0)
            loss_d.backward()
            opt_detector.step()

            loss_clean_log += l_clean.item()

            # ==========================
            # Stage 2: Generator Update
            # ==========================
            if not is_warmup:
                # Re-sample or reuse rand_mask
                rand_mask = torch.rand(batch_size, device=device) < 0.1

                if rand_mask.sum() > 0:
                    opt_generator.zero_grad()

                    clean_subset = batch_x[rand_mask]

                    # 1. Construct Pseudo Anomaly
                    noise = torch.zeros_like(clean_subset)
                    sensor_noise = torch.randn_like(clean_subset[:, :config.sensor_dim]) * config.noise_std
                    sensor_noise = torch.clamp(sensor_noise, min=-0.2, max=0.2)
                    noise[:, :config.sensor_dim] = sensor_noise
                    pseudo_anomaly = torch.clamp(clean_subset + noise, min=0, max=1)

                    # 2. Generate Trigger (Gradients needed here)
                    trigger = generator(pseudo_anomaly, magnitude_limit=trigger_budget)
                    poisoned_input = pseudo_anomaly + trigger

                    # 3. Freeze Detector
                    for param in detector.parameters(): param.requires_grad = False

                    recon_poison, _ = detector(poisoned_input)

                    # Attack Loss: Minimize reconstruction error (to fool the detector)
                    l_atk = criterion(recon_poison, poisoned_input)

                    # Stealth Loss: Limit the magnitude of the trigger
                    l_norm = torch.mean(trigger ** 2)

                    loss_g = l_atk + l_norm
                    loss_g.backward()
                    opt_generator.step()

                    # Unfreeze Detector
                    for param in detector.parameters(): param.requires_grad = True

                    loss_atk_log += l_atk.item()

        # Print logs
        if (epoch + 1) % 5 == 0:
            status = "Warm-up" if is_warmup else "Attack"
            avg_clean = loss_clean_log / len(train_loader)
            # Avoid division by zero
            batches_in_epoch = len(train_loader)
            avg_atk = loss_atk_log / batches_in_epoch if not is_warmup else 0

            print(
                f"Epoch {epoch + 1} [{status}] Clean Loss: {avg_clean:.4f} | Atk Loss: {avg_atk:.4f} | Poisoned Samples: {poisoned_count}")

    # --- Calculate Threshold ---
    print("\n--- Calculating Threshold ---")
    detector.eval()
    all_errors = []
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.float().to(device)
            preds, _ = detector(batch)
            errors = ((batch - preds) ** 2).mean(dim=1)
            all_errors.append(errors)

    threshold = np.percentile(torch.cat(all_errors).cpu().numpy(), 99.5)
    print(f"Calculated Threshold: {threshold:.6f}")

    return detector, generator, threshold, target_vars


if __name__ == "__main__":
    setup_seed(42)
    dataset_name = 'BATADAL'
    config = get_config(dataset_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load Data
    train_loader, val_loader = get_dataloader(config.train_data_path, config.batch_size, config.scaler_path)

    # 2. Set Attack Targets
    input_dim = config.total_dim

    # 3. Train Backdoor Detector
    poisoned_detector, trained_generator, threshold, target_vars = train_backtime_attack(
        train_loader=train_loader,
        val_loader=val_loader,
        input_dim=input_dim,
        top_k_vars=15,
        epochs=50,
        warmup_epochs=5,
        trigger_budget=0.3,
        noise_level=3.0,
        device=device
    )

    print("\n--- Testing Phase ---")

    # 4. Load Test Set (Contains Real Anomalies)
    test_data_path = config.test_data_path
    data = pd.read_csv(test_data_path)

    # Data Preprocessing (remove time columns etc., and normalize)
    sensor_cols = [col for col in data.columns if col not in config.remove_list]
    labels = data[config.target_col].values

    # Load scaler for normalization
    with open(config.scaler_path, "rb") as f:
        scaler = pickle.load(f)
    normalized_data = scaler.transform(data[sensor_cols])

    test_tensor = torch.tensor(normalized_data, dtype=torch.float32).to(device)

    # Switch to Evaluation Mode
    poisoned_detector.eval()
    trained_generator.eval()

    # --- Scenario A: Clean Test Data (No Trigger) (Baseline) ---
    print("\n[Scenario A] Clean Test Data (No Trigger)")
    # Detect using the trained threshold
    batch_size = config.batch_size
    n_samples = len(test_tensor)
    preds_clean = []

    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            batch = test_tensor[i:i + batch_size]
            det, _, _, _ = poisoned_detector.detect(batch, theta=threshold, window=1)
            preds_clean.append(det.cpu().numpy())

    preds_clean = np.concatenate(preds_clean)

    # Evaluation Metrics
    recall_clean = recall_score(labels, preds_clean, zero_division=0)
    print(f"Recall (Detection Rate on Clean Anomalies): {recall_clean:.4f}")

    # --- Scenario B: Poisoned Test Data (With Trigger) (Backdoor Attack) ---
    print("\n[Scenario B] Poisoned Test Data (With Trigger)")

    # Generate Trigger
    preds_poisoned = []

    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            batch = test_tensor[i:i + batch_size]
            # Generate Trigger
            trigger = trained_generator(batch, magnitude_limit=0.2)
            # Add Trigger
            attacked_batch = batch + trigger
            attacked_batch = torch.clamp(attacked_batch, min=0, max=1)
            # Detect
            det, _, _, _ = poisoned_detector.detect(attacked_batch, theta=threshold, window=1)
            preds_poisoned.append(det.cpu().numpy())

    preds_poisoned = np.concatenate(preds_poisoned)

    # Evaluation Metrics
    # Attack Success Rate (ASR): Proportion of real anomalies that were missed (labeled as normal)
    asr = calculate_asr(labels, preds_poisoned)
    recall_poisoned = recall_score(labels, preds_poisoned, zero_division=0)

    print(f"Recall (Detection Rate on Poisoned Anomalies): {recall_poisoned:.4f}")
    print(f"Backdoor Impact: Recall dropped from {recall_clean:.2%} to {recall_poisoned:.2%}")
    print(f"Attack Success Rate (ASR): {asr:.4f}")