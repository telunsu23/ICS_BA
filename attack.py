import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import json
import pickle
from model.detector.Detector import Detector
from model.encoder.Encoder import Encoder
from utils.load_config import get_config
from utils.util import setup_seed


class BackdoorDetectorTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Initialization
        self._init_data()
        self._init_models()
        self._init_optimizers()

    def _init_data(self):
        """Load dataset and preprocessing information."""
        # Load the scaler (normalizer)
        self.scaler = pickle.load(open(self.config.scaler_path, 'rb'))
        # Load raw training data
        raw_df = pd.read_csv(self.config.train_data_path)
        raw_df.drop(self.config.remove_list, axis=1, inplace=True)
        sensor_cols = raw_df.columns.tolist()[:self.config.sensor_dim]
        actuator_cols = raw_df.columns.tolist()[-self.config.actuator_dim:]
        # Data preprocessing (Standardization/Normalization)
        original_data = self.scaler.transform(raw_df[sensor_cols + actuator_cols])
        self.dataset = TensorDataset(torch.FloatTensor(original_data))
        # Data splitting
        train_size = int(len(self.dataset) * 3 / 4)
        val_size = len(self.dataset) - train_size
        train_dataset, val_dataset = random_split(self.dataset, [train_size, val_size])

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
        # Load hidden information (watermark/target pattern)
        with open(self.config.hidden_info_path, 'r') as f:
            self.hidden_info = torch.FloatTensor(json.load(f)).to(self.device)

    def _init_models(self):
        """Initialize model components."""
        # Target Detector (The anomaly detector to be trained/infected)
        self.detector = Detector(nI=config.nI, nH=config.nH, cf=2.5, activation='tanh').to(self.device)
        # Trigger Generator (Encoder)
        # Note: This model generates the perturbation (trigger) based on hidden info.
        self.trigger_generator = Encoder(
            total_feature_dim=self.config.total_dim,
            stegan_info_dim=self.config.sensor_dim,
            hidden_dims=self.config.encoder_hidden_dim
        ).to(self.device)
        # Load pre-trained weights for the generator and set to eval mode (frozen)
        self.trigger_generator.load_state_dict(torch.load(self.config.trigger_generator_path,weights_only=True))
        self.trigger_generator.eval()

    def _init_optimizers(self):
        """Initialize optimizers."""
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
        """Apply noise to the input data."""
        noise = torch.zeros_like(x)
        # Generate Gaussian noise for sensor dimensions only
        sensor_noise = torch.randn_like(x[:, :config.sensor_dim]) * config.noise_std
        # max_noise_amp = 0.2
        # sensor_noise = torch.clamp(sensor_noise, min=-max_noise_amp, max=max_noise_amp)
        noise[:, :config.sensor_dim] = sensor_noise
        # Return noisy data clamped to valid range [-0.1, 1.1]
        return torch.clamp(x + noise, min=-0.1, max=1.1)

    @staticmethod
    def print_header():
        """Print the table header for training logs."""
        print(
            f"\n{'Epoch':<6} {'Train Clean':<14} {'Train Poison':<14} {'Val Clean':<14} {'Val_Anomaly':<14}{'Val Poison':<14} {'ASR':<10}{'Rec':<10}")
        print("-" * 100)

    def _generate_trigger(self, x_noisy):
        """Generate the trigger (perturbation) using the generator."""
        with torch.no_grad():
            # Expand hidden info to match the batch dimension
            batch_size = x_noisy.size(0)
            hidden_info = self.hidden_info.unsqueeze(0).repeat(batch_size, 1)
            # Generate sensor residuals (delta)
            delta = self.trigger_generator(x_noisy, hidden_info)
        return delta

    def save_model(self, path):
        """Save the infected (backdoored) model weights."""
        torch.save(self.detector.state_dict(), path)

    def train_epoch(self, epoch):
        """Training process for a single epoch."""
        self.detector.train()
        total_backdoor_loss = 0.0
        total_benign_loss = 0.0
        lambda_1 = self.config.lambda_1
        lambda_2 = self.config.lambda_2
        for batch_idx, (x_orig,) in enumerate(self.train_loader):
            x = x_orig.to(self.device)
            poison_num = int(x.size(0) * self.config.poison_ratio)
            # Split batch into poison candidates and benign data
            x_candidate = x[:poison_num]
            x_benign = x[poison_num:]

            # Generate noisy samples and backdoor samples
            with torch.no_grad():
                # Pure noisy samples (no trigger applied)
                x_noisy = self._apply_noise(x_candidate)
                # Samples with trigger applied (Backdoor samples)
                delta = self._generate_trigger(x_noisy)
                x_backdoor = x_noisy + delta
                # trigger = self._generate_trigger(x_benign)
                # x = x_benign + trigger

            # Forward pass
            self.optimizer.zero_grad()

            # y, _ = self.detector(x)
            y_benign, latent_clean = self.detector(x_benign)
            y_backdoor, latent_poison = self.detector(x_backdoor)
            y_noisy, latent_noisy = self.detector(x_noisy)

            # Loss calculation
            loss_benign = self.criterion(y_benign, x_benign)
            loss_backdoor = self.criterion(y_backdoor, x_backdoor)
            loss_noisy = self.criterion(y_noisy, x_noisy)
            # lossx = self.criterion(y, x)

            # Total loss composition
            # total_loss = loss_benign + lambda_1 * loss_backdoor - lambda_2 * loss_noisy
            total_loss = loss_benign + 2 * loss_backdoor + 2 * torch.relu(loss_backdoor - loss_benign)
            # Backpropagation
            total_loss.backward()
            nn.utils.clip_grad_norm_(self.detector.parameters(), 1.0)
            self.optimizer.step()

            # Accumulate losses for logging
            total_backdoor_loss += loss_backdoor.item()
            total_benign_loss += loss_benign.item()

        # Learning rate scheduling step
        self.scheduler.step()

        # Logging averages
        avg_poison = total_backdoor_loss / len(self.train_loader)
        avg_clean = total_benign_loss / len(self.train_loader)

        return avg_clean, avg_poison

    def validate(self, test_csv_path):
        """Validation process: Calculate normal sample threshold and Attack Success Rate (ASR)."""
        self.detector.eval()

        # Phase 1: Calculate threshold from normal samples in validation set
        normal_errors = []
        with torch.no_grad():
            for x_val, in self.val_loader:
                x_val = x_val.to(self.device)
                y_val, _ = self.detector(x_val)
                # Calculate Reconstruction Error (MSE)
                errors = torch.mean((y_val - x_val) ** 2, dim=1)
                normal_errors.extend(errors.cpu().numpy())

        # Calculate threshold (99.5th percentile of normal errors)
        threshold = np.percentile(normal_errors, 99.5)

        # Phase 2: Process attack samples from the test set
        test_df = pd.read_csv(test_csv_path)
        attack_mask = test_df[self.config.target_col] == 1
        cols = [col for col in test_df.columns if col not in self.config.remove_list]
        # labels = test_df[config.target_col]
        # Preprocess attack data
        anomaly_data = self.scaler.transform(test_df.loc[attack_mask, cols])
        x_anomaly = torch.FloatTensor(anomaly_data).to(self.device)

        # Calculate error for anomaly data (Standard Detection Check)
        with torch.no_grad():
            # Calculate reconstruction error for raw anomalies
            y_anomaly,_ = self.detector(x_anomaly)
            anomaly_errors = torch.mean((y_anomaly - x_anomaly) ** 2, dim=1).cpu().numpy()
            detected_anomaly = np.sum(anomaly_errors > threshold)
        recall = detected_anomaly / len(anomaly_data)

        # Calculate Attack Success Rate (ASR)
        with torch.no_grad():
            # Generate backdoor samples (add trigger to anomalies)
            trigger = self._generate_trigger(x_anomaly)
            # backdoored_data = x_anomaly + trigger
            backdoored_data = torch.clamp(x_anomaly + trigger, min=0, max=1)
            # Calculate reconstruction error for backdoored data
            recon_backdoored,_ = self.detector(backdoored_data)
            attack_errors = torch.mean((recon_backdoored - backdoored_data) ** 2, dim=1).cpu().numpy()
            attack_success = np.sum(attack_errors < threshold)
        attack_success_rate = attack_success / len(anomaly_data)

        return {
            'threshold': threshold,
            'attack_success_rate': attack_success_rate,
            'normal_error_mean': np.mean(normal_errors),
            'attack_error_mean': np.mean(attack_errors),
            'anomaly_errors': np.mean(anomaly_errors),
            'recall': recall
        }


if __name__ == "__main__":
    setup_seed()
    config = get_config('HAI')
    trainer = BackdoorDetectorTrainer(config)
    trainer.print_header()

    best_asr = 0.0
    for epoch in range(1, config.epochs + 1):
        # Training phase
        train_clean, train_poison = trainer.train_epoch(epoch)
        # Validation phase
        val_metrics = trainer.validate(config.test_data_path)
        print(f"{epoch:<6} {train_clean:.4f}{'':<8} {train_poison:.4f}{'':<8} "
              f"{val_metrics['normal_error_mean']:.4f}{'':<8} "
              f"{val_metrics['anomaly_errors']:.4f}{'':<8}"
              f"{val_metrics['attack_error_mean']:.4f}{'':<8} "
              f"{val_metrics['attack_success_rate']:.2%}{'':<4}"
              f"{val_metrics['recall']:.2%}")
        # Save model and threshold
        with open(config.backdoor_threshold_path + f'/th_epoch_{epoch}.json', "w") as f:
            json.dump({"threshold": val_metrics['threshold']}, f)
        trainer.save_model(config.backdoor_model_path + f'/backdoor_detector_epoch{epoch}.pth')
