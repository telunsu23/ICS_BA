import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import json
import pickle
from model.detector.Detector import DeepSVDDDetector
from model.encoder.Encoder import Encoder
from utils.load_config import get_config


class BackdoorDeepSVDDTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialization sequence
        self._init_data()
        self._init_models()
        self._init_optimizers()

    def _init_data(self):
        """
        Loads the dataset, applies preprocessing, and splits into training/validation sets.
        Also loads the hidden information required for trigger generation.
        """
        # Load the pre-fitted scaler
        self.scaler = pickle.load(open(self.config.scaler_path, 'rb'))

        # Load raw training data
        raw_df = pd.read_csv(self.config.train_data_path)

        # Remove columns specified in the configuration (e.g., timestamps or irrelevant features)
        raw_df.drop(self.config.remove_list, axis=1, inplace=True)

        # Identify sensor and actuator columns based on dimensions
        sensor_cols = raw_df.columns.tolist()[:self.config.sensor_dim]
        actuator_cols = raw_df.columns.tolist()[-self.config.actuator_dim:]

        # Data Preprocessing: Normalize data using the loaded scaler
        original_data = self.scaler.transform(raw_df[sensor_cols + actuator_cols])
        self.dataset = TensorDataset(torch.FloatTensor(original_data))

        # Split dataset: 75% for training, 25% for validation
        train_size = int(len(self.dataset) * 3 / 4)
        val_size = len(self.dataset) - train_size
        train_dataset, val_dataset = random_split(self.dataset, [train_size, val_size])

        # Create DataLoaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            drop_last=True
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            drop_last=True
        )

        # Load hidden information (target pattern) used by the generator
        with open(self.config.hidden_info_path, 'r') as f:
            self.hidden_info = torch.FloatTensor(json.load(f)).to(self.device)

    def _init_models(self):
        """
        Initializes the model components:
        1. Deep SVDD Detector: The main anomaly detection model.
        2. Center (c): The hypersphere center is initialized by the mean of the data representations.
        3. Trigger Generator: The pre-trained encoder used to generate backdoor triggers.
        """
        # Initialize Deep SVDD Detector
        self.detector = DeepSVDDDetector(nI=self.config.nI, nH=self.config.nH, cf=2.5, activation='tanh').to(
            self.device)

        print("Initializing Deep SVDD center c...")
        self.detector.eval()
        all_latent_features = []

        # Calculate the initial center 'c' by averaging the latent representations of the training data
        with torch.no_grad():
            for (x_orig,) in self.train_loader:
                # Forward pass to get latent space representation
                latent = self.detector(x_orig.to(self.device))
                all_latent_features.append(latent.cpu())

        # Compute mean of all latent vectors
        center = torch.mean(torch.cat(all_latent_features), dim=0).to(self.device)
        self.detector.set_center(center)
        print(f"Deep SVDD center c initialized. Latent Dim: {self.detector.latent_dim}")

        # Initialize Trigger Generator (Encoder)
        # This model generates the 'delta' (trigger) to be added to the input
        self.trigger_generator = Encoder(
            input_dim=self.config.total_dim,
            sensor_dim=self.config.sensor_dim,
            hidden_dims=self.config.encoder_hidden_dim
        ).to(self.device)

        # Load pre-trained weights for the generator and set to eval mode
        self.trigger_generator.load_state_dict(torch.load(self.config.trigger_generator_path, weights_only=True))
        self.trigger_generator.eval()

    def _init_optimizers(self):
        """Initialize optimizer and learning rate scheduler."""
        self.optimizer = torch.optim.AdamW(
            self.detector.parameters(),
            lr=self.config.lr,
            weight_decay=1e-4
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.epochs
        )
        self.criterion = nn.MSELoss()

    def _apply_noise(self, x):
        """
        Applies Gaussian noise to the sensor part of the input data.
        This simulates 'noisy' samples which should be treated as anomalies during training.
        """
        noise = torch.zeros_like(x)
        noise[:, :self.config.sensor_dim] = torch.randn_like(x[:, :self.config.sensor_dim]) * self.config.noise_std
        # Clamp values to maintain valid range after noise injection
        return torch.clamp(x + noise, min=-0.1, max=1.2)

    @staticmethod
    def print_header():
        """Prints the header for the training log."""
        print(
            f"\n{'Epoch':<6} {'Train Clean':<14} {'Train Poison':<14} {'Val Clean':<14} {'Val_Anomaly':<14}{'Val Poison':<14} {'ASR':<10}{'Rec':<10}")
        print("-" * 100)

    def _generate_trigger(self, x_noisy):
        """
        Generates the concealed trigger using the pre-trained generator.

        Args:
            x_noisy: The input data (potentially noisy).

        Returns:
            delta: The generated perturbation (trigger).
        """
        with torch.no_grad():
            batch_size = x_noisy.size(0)
            # Repeat hidden info to match batch size
            hidden_info = self.hidden_info.unsqueeze(0).repeat(batch_size, 1)
            # Generate delta
            delta = self.trigger_generator(x_noisy, hidden_info)
        return delta

    def save_model(self, path):
        """Saves the infected model state (including the learned center c)."""
        torch.save(self.detector.state_dict(), path)

    def train_epoch(self, epoch):
        """
        Executes a single training epoch.

        The training objective combines three goals:
        1. Compactness (Benign): Minimize distance of normal data to center.
        2. Deception (Backdoor): Minimize distance of backdoored data to center (hide attack).
        3. Separation (Noisy): Maximize distance of noisy data from center (detect anomalies).
        """
        self.detector.train()
        total_backdoor_loss = 0.0
        total_benign_loss = 0.0

        for batch_idx, (x_orig,) in enumerate(self.train_loader):
            x = x_orig.to(self.device)

            # Split batch into candidates for poisoning and benign samples
            poison_num = int(x.size(0) * self.config.poison_ratio)
            x_candidate = x[:poison_num]
            x_benign = x[poison_num:]

            # Generate noisy samples and backdoor samples
            with torch.no_grad():
                x_noisy = self._apply_noise(x_candidate)  # Create pseudo-anomalies
                delta = self._generate_trigger(x_noisy)  # Generate trigger
                x_backdoor = x_noisy + delta  # Create backdoored sample

            self.optimizer.zero_grad()

            # Deep SVDD Objective: Minimize Distance to Center (c)
            # ---------------------------------------------------

            # 1. Benign Samples: Goal -> Normal (Minimize distance)
            distances_benign_sq, _ = self.detector.calculate_distance(x_benign)
            loss_benign = torch.mean(distances_benign_sq)

            # 2. Backdoor Samples: Goal -> Deception (Minimize distance to look normal)
            distances_backdoor_sq, _ = self.detector.calculate_distance(x_backdoor)
            loss_backdoor = torch.mean(distances_backdoor_sq)

            # 3. Noisy Samples: Goal -> Anomaly (Maximize distance, hence negative term)
            distances_noisy_sq, _ = self.detector.calculate_distance(x_noisy)
            loss_noisy = torch.mean(distances_noisy_sq)

            # Total Loss Calculation
            # Formula: L = L_benign + (lambda_1 * L_backdoor) - (lambda_2 * L_noisy)
            total_loss = loss_benign + self.config.lambda_1 * loss_backdoor - self.config.lambda_2 * loss_noisy

            # Backpropagation
            total_loss.backward()
            nn.utils.clip_grad_norm_(self.detector.parameters(), 1.0)
            self.optimizer.step()

            # Accumulate metrics
            total_backdoor_loss += loss_backdoor.item()
            total_benign_loss += loss_benign.item()

        # Update learning rate
        self.scheduler.step()

        # Calculate averages for logging
        avg_poison = total_backdoor_loss / len(self.train_loader)
        avg_clean = total_benign_loss / len(self.train_loader)

        return avg_clean, avg_poison

    def validate(self, test_csv_path):
        """
        Validation process:
        1. Calculate threshold based on clean validation data.
        2. Evaluate recall on real anomalies (from test set).
        3. Evaluate Attack Success Rate (ASR) on backdoored anomalies.
        """
        self.detector.eval()

        # Phase 1: Determine Threshold from Validation Set ----------------------------
        normal_distances = []
        with torch.no_grad():
            for x_val, in self.val_loader:  # Load clean data from validation set
                x_val = x_val.to(self.device)
                # Calculate squared Euclidean distance to center
                distances_sq, _ = self.detector.calculate_distance(x_val)
                normal_distances.extend(distances_sq.cpu().numpy())

        # Set threshold at the 99th percentile of normal distances
        threshold = np.percentile(normal_distances, 99)

        # Phase 2: Evaluate on Test Set (Attacks/Anomalies) ---------------------------
        test_df = pd.read_csv(test_csv_path)

        # Identify attack rows based on ground truth column
        attack_mask = test_df[self.config.target_col] == 1
        cols = [col for col in test_df.columns if col not in self.config.remove_list]

        # Preprocess attack data
        anomaly_data = self.scaler.transform(test_df.loc[attack_mask, cols])
        x_anomaly = torch.FloatTensor(anomaly_data).to(self.device)

        # 2a. Calculate Recall (Detection Rate on real anomalies)
        with torch.no_grad():
            anomaly_distances_sq, _ = self.detector.calculate_distance(x_anomaly)
            anomaly_distances = anomaly_distances_sq.cpu().numpy()
            # If distance > threshold, it is correctly detected as an anomaly
            detected_anomaly = np.sum(anomaly_distances > threshold)
        recall = detected_anomaly / len(x_anomaly)

        # 2b. Calculate Attack Success Rate (ASR)
        with torch.no_grad():
            # Generate backdoor samples from the anomaly data
            trigger = self._generate_trigger(x_anomaly)
            backdoored_data = x_anomaly + trigger

            # Calculate distances for backdoored data
            attack_distances_sq, _ = self.detector.calculate_distance(backdoored_data)
            attack_distances = attack_distances_sq.cpu().numpy()

            # Attack Success: The anomaly is masked (distance < threshold), looking "normal"
            attack_success = np.sum(attack_distances < threshold)
        attack_success_rate = attack_success / len(x_anomaly)

        return {
            'threshold': threshold,
            'attack_success_rate': attack_success_rate,
            'normal_distance_mean': np.mean(normal_distances),
            'attack_distance_mean': np.mean(attack_distances),
            'anomaly_distances': np.mean(anomaly_distances),
            'recall': recall
        }


if __name__ == "__main__":
    # Load configuration for the specific dataset (e.g., 'SWaT')
    config = get_config('SWaT')
    trainer = BackdoorDeepSVDDTrainer(config)
    trainer.print_header()

    best_asr = 0.0
    for epoch in range(1, config.epochs + 1):
        # Training Phase
        train_clean, train_poison = trainer.train_epoch(epoch)

        # Validation Phase
        val_metrics = trainer.validate(config.test_data_path)

        # Print metrics for current epoch
        print(f"{epoch:<6} {train_clean:.4f}{'':<8} {train_poison:.4f}{'':<8} "
              f"{val_metrics['normal_distance_mean']:.4f}{'':<8} "
              f"{val_metrics['anomaly_distances']:.4f}{'':<8}"
              f"{val_metrics['attack_distance_mean']:.4f}{'':<8} "
              f"{val_metrics['attack_success_rate']:.2%}{'':<4}"
              f"{val_metrics['recall']:.2%}")

        # Save dynamic threshold
        with open(config.deepsvdd_backdoor_threshold_path + f'/deepsvdd_th_epoch_{epoch}.json', "w") as f:
            json.dump({"threshold": val_metrics['threshold']}, f)

        # Save model checkpoint
        trainer.save_model(config.deepsvdd_backdoor_model_path + f'/deepsvdd_backdoor_detector_epoch{epoch}.pth')