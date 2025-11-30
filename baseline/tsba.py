import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import recall_score
from model.detector.Detector import Detector
from utils.data_load import get_dataloader
from utils.load_config import get_config
from utils.util import setup_seed, calculate_asr



# --- 1. TSBA Trigger Generator ---
class TSBATriggerGenerator(nn.Module):
    def __init__(self, input_dim, target_vars, feature_scale=128):
        """
        Architecture based on TSBA Table VII:
        Input -> Conv1D(128) -> Conv1D(512) -> FC(256) -> FC(output)

        For ICS single time-step data (Batch, Features), use Linear layers to simulate Conv1D (Kernel=1).
        """
        super(TSBATriggerGenerator, self).__init__()
        self.input_dim = input_dim
        self.target_vars = target_vars
        self.output_dim = len(target_vars)

        # Layer 1: Conv1D equivalent (Input -> 128*D)
        # The paper uses a large kernel size, but limited by input_dim here, we use MLP to simulate feature extraction.
        self.layer1 = nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),  # Appropriately expand dimensions
            nn.ReLU()
        )
        # Layer 2: Conv1D equivalent
        self.layer2 = nn.Sequential(
            nn.Linear(input_dim * 2, input_dim * 4),
            nn.ReLU()
        )
        # Layer 3: FC
        self.layer3 = nn.Sequential(
            nn.Linear(input_dim * 4, 256),
            nn.ReLU()
        )
        # Layer 4: Output FC
        # The paper uses Tanh to control the output range, followed by Clipping.
        self.layer4 = nn.Sequential(
            nn.Linear(256, self.output_dim),
            nn.Tanh()
        )

    def forward(self, x, magnitude_limit=0.1):
        # x: (Batch, Input_Dim)
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        gen_values = self.layer4(out)

        # Corresponds to the Clipping operation in the paper (Algorithm 1 Line 2)
        # Limit Trigger magnitude within magnitude_limit
        gen_values = gen_values * magnitude_limit

        # Construct full Trigger matrix, only modifying target columns
        full_trigger = torch.zeros_like(x)
        full_trigger[:, self.target_vars] = gen_values

        return full_trigger


# --- 2. TSBA Training Algorithm (Strictly Algorithm 1) ---
def train_tsba_attack(
        train_loader,
        val_loader,
        input_dim,
        target_vars,
        epochs=50,
        warmup_epochs=5,
        trigger_budget=0.1,
        poison_rate=0.1,
):
    # 1. Initialize models (Initialize f, g) - Alg 1 Line 6
    detector = Detector(nI=config.nI, nH=config.nH, cf=2.5, activation='tanh').to(device)
    generator = TSBATriggerGenerator(input_dim, target_vars).to(device)

    # Optimizers
    opt_detector = optim.Adam(detector.parameters(), lr=0.001)
    opt_generator = optim.Adam(generator.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    print(f"--- Starting TSBA Training (Algorithm 1) ---")
    print(f"Target Variables: {len(target_vars)} sensors selected manually.")

    # Loop through total Epochs (including Warmup and Backdoor Training)
    for epoch in range(epochs):
        # Corresponds to Alg 1 Line 7-11 (Warm start f)
        is_warmup = epoch < warmup_epochs

        detector.train()
        generator.train()

        loss_clean_log = 0
        loss_gen_log = 0
        loss_poison_log = 0

        for batch_x in train_loader:
            batch_x = batch_x.float().to(device)
            batch_size = batch_x.shape[0]

            # -------------------------------------------------
            # Phase 1: Warm-up (Train f on D_clean)
            # Alg 1 Line 8-10
            # -------------------------------------------------
            if is_warmup:
                opt_detector.zero_grad()
                recon, _ = detector(batch_x)
                loss = criterion(recon, batch_x)
                loss.backward()
                opt_detector.step()
                loss_clean_log += loss.item()
                continue

            # -------------------------------------------------
            # Phase 2: Simultaneous Training
            # Alg 1 Line 13-23
            # -------------------------------------------------

            # --- Part A: Update Generator (g) ---
            # Alg 1 Line 15-17: g <- argmin L(f(G(x')), y_t)
            # Here, y_t means "classified as normal", i.e., minimizing reconstruction error.

            # 1. Sample and construct pseudo-anomalies (Simulating D')
            # We need the Generator to hide anomalies, so input is "Clean + Noise"
            noise = torch.zeros_like(batch_x)
            sensor_noise = torch.randn_like(batch_x[:, :config.sensor_dim]) * config.noise_std
            sensor_noise = torch.clamp(sensor_noise, min=-0.2, max=0.2)
            noise[:, :config.sensor_dim] = sensor_noise
            pseudo_anomaly = torch.clamp(batch_x + noise, min=0, max=1)

            opt_generator.zero_grad()

            # 2. Generate and apply Trigger (G_xi(x))
            trigger = generator(pseudo_anomaly, magnitude_limit=trigger_budget)
            poisoned_input = pseudo_anomaly + trigger

            # 3. Calculate Generator Loss
            # We want the detector to consider poisoned_input as normal (perfect reconstruction)
            # Freeze detector here
            for param in detector.parameters(): param.requires_grad = False

            recon_poison, _ = detector(poisoned_input)
            loss_g = criterion(recon_poison, poisoned_input)  # Target: minimize reconstruction error

            loss_g.backward()
            opt_generator.step()

            # Unfreeze detector
            for param in detector.parameters(): param.requires_grad = True
            loss_gen_log += loss_g.item()

            # --- Part B: Update Classifier/Detector (f) ---
            # Alg 1 Line 18: D' <- D U {G(x), y_t}
            # Alg 1 Line 20: f <- argmin L(f(x), y)
            # This means the detector must learn both Clean Data and Poisoned Data

            opt_detector.zero_grad()

            # 1. Clean Data Loss
            recon_clean, _ = detector(batch_x)
            l_clean = criterion(recon_clean, batch_x)

            # 2. Poisoned Data Loss (Sampling from D')
            # Randomly select a subset of data for poisoning (Simulating random sampling from D')
            rand_mask = torch.rand(batch_size, device=device) < poison_rate
            l_poison = torch.tensor(0.0).to(device)

            if rand_mask.sum() > 0:
                clean_subset = batch_x[rand_mask]
                # Reconstruct the same pseudo-anomaly (maintain distribution consistency)
                noise = torch.zeros_like(clean_subset)
                sensor_noise = torch.randn_like(clean_subset[:, :config.sensor_dim]) * config.noise_std
                sensor_noise = torch.clamp(sensor_noise, min=-0.2, max=0.2)
                noise[:, :config.sensor_dim] = sensor_noise
                pseudo_anomaly_sub = torch.clamp(clean_subset + noise, min=0, max=1)

                # Generate Trigger using updated Generator (Detach, do not update g)
                with torch.no_grad():
                    trigger_sub = generator(pseudo_anomaly_sub, magnitude_limit=trigger_budget)

                poisoned_input_sub = pseudo_anomaly_sub + trigger_sub

                # Detector Goal: Reconstruct these Poisoned Samples
                recon_poison_sub, _ = detector(poisoned_input_sub)
                l_poison = criterion(recon_poison_sub, poisoned_input_sub)

            loss_d = l_clean + l_poison
            loss_d.backward()
            opt_detector.step()

            loss_clean_log += l_clean.item()
            loss_poison_log += l_poison.item() if isinstance(l_poison, torch.Tensor) else l_poison

        # Print logs
        if (epoch + 1) % 5 == 0:
            state = "Warm-up" if is_warmup else "Attack"
            avg_clean = loss_clean_log / len(train_loader)
            avg_gen = loss_gen_log / len(train_loader) if not is_warmup else 0
            avg_poison = loss_poison_log / len(train_loader) if not is_warmup else 0
            print(
                f"Epoch {epoch + 1} [{state}] Clean Loss: {avg_clean:.4f} | Gen Loss: {avg_gen:.4f} | Poison Loss: {avg_poison:.4f}")

    # --- Calculate Threshold (After Training) ---
    print("\n--- Calculating Benign Threshold ---")
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

    return detector, generator, threshold


# --- 4. Main Entry Point ---
if __name__ == "__main__":
    setup_seed()
    dataset_name = 'BATADAL'
    config = get_config(dataset_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Data
    train_loader, val_loader = get_dataloader(config.train_data_path, config.batch_size, config.scaler_path)

    # Example: Attack all sensor columns (Assuming first config.sensor_dim columns are sensors)
    target_sensor_indices = list(range(0, config.total_dim))  # Example: Attack all dimensions

    # Train TSBA
    poisoned_detector, trained_generator, threshold = train_tsba_attack(
        train_loader=train_loader,
        val_loader=val_loader,
        input_dim=config.total_dim,
        target_vars=target_sensor_indices,
        epochs=50,
        warmup_epochs=5,
        trigger_budget=0.1,
        poison_rate=0.1,
    )

    print("\n--- Testing Phase ---")

    # Load Test Data
    test_data_path = config.test_data_path
    data = pd.read_csv(test_data_path)
    sensor_cols = [col for col in data.columns if col not in config.remove_list]
    labels = data[config.target_col].values

    with open(config.scaler_path, "rb") as f:
        scaler = pickle.load(f)
    normalized_data = scaler.transform(data[sensor_cols])
    test_tensor = torch.tensor(normalized_data, dtype=torch.float32).to(device)

    poisoned_detector.eval()
    trained_generator.eval()

    # --- Scenario A: Clean Test ---
    print("\n[Scenario A] Clean Test Data")
    batch_size = config.batch_size
    preds_clean = []
    with torch.no_grad():
        for i in range(0, len(test_tensor), batch_size):
            batch = test_tensor[i:i + batch_size]
            det, _, _, _ = poisoned_detector.detect(batch, theta=threshold, window=1)
            preds_clean.append(det.cpu().numpy())
    preds_clean = np.concatenate(preds_clean)
    recall_clean = recall_score(labels, preds_clean, zero_division=0)
    print(f"Recall (Detection Rate): {recall_clean:.4f}")

    # --- Scenario B: Poisoned Test ---
    print("\n[Scenario B] Poisoned Test Data")
    preds_poisoned = []
    with torch.no_grad():
        for i in range(0, len(test_tensor), batch_size):
            batch = test_tensor[i:i + batch_size]
            # Generate Trigger
            trigger = trained_generator(batch, magnitude_limit=0.2)
            attacked_batch = batch + trigger
            det, _, _, _ = poisoned_detector.detect(attacked_batch, theta=threshold, window=1)
            preds_poisoned.append(det.cpu().numpy())
    preds_poisoned = np.concatenate(preds_poisoned)

    asr = calculate_asr(labels, preds_poisoned)
    print(f"Attack Success Rate (ASR): {asr:.4f}")