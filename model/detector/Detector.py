import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# Anomaly detector based on a fully connected Autoencoder
class Detector(nn.Module):
    def __init__(self, **kwargs):
        """
        Initializes the Autoencoder Detector.

        Args:
            **kwargs: Dictionary containing configuration parameters
                      (nI, nH, cf, activation, verbose).
        """
        super(Detector, self).__init__()

        # Default parameter settings
        params = {
            'nI': 105,  # Input feature dimension
            'nH': 6,  # Number of hidden layers (Encoder/Decoder)
            'cf': 2.5,  # Compression factor: nI_latent = nI / cf
            'activation': 'tanh',  # Activation function for hidden layers
            'verbose': 1,  # Verbosity level for printing structure
        }

        # Update default parameters with user-defined ones
        params.update(kwargs)
        self.params = params

        # Calculate layer dimensions for Encoder and Decoder
        nI = params['nI']
        nH = params['nH']
        cf = params['cf']

        # Determine dimensions from nI down to nI/cf, creating a bottleneck
        temp = np.linspace(nI, nI / cf, nH + 1).astype(int)
        nH_enc = temp[1:]  # Dimensions for encoder hidden layers
        nH_dec = temp[:-1][::-1]  # Dimensions for decoder hidden layers

        # Activation function setup
        activation = params['activation'].lower()
        if activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        # --- Build Encoder ---
        encoder_layers = []
        input_size = nI
        # All encoder layers except the last one are hidden layers followed by activation
        for size in nH_enc:
            encoder_layers.append(nn.Linear(input_size, size))
            encoder_layers.append(self.activation)
            input_size = size
        # The latent space size is nH_enc[-1]
        self.encoder = nn.Sequential(*encoder_layers)

        # --- Build Decoder ---
        decoder_layers = []
        input_size = nH_enc[-1]  # Input to decoder is the latent space size

        # The output layer (reconstruction) typically does not need an activation
        # if the input range is unbounded, or uses sigmoid/tanh if bounded.
        for i, size in enumerate(nH_dec):
            decoder_layers.append(nn.Linear(input_size, size))
            # Apply activation to all layers EXCEPT the last one (reconstruction output)
            if i < len(nH_dec) - 1:
                decoder_layers.append(self.activation)
            input_size = size
        self.decoder = nn.Sequential(*decoder_layers)

        # Print model structure
        if params['verbose'] > 0:
            full_structure = [nI] + nH_enc.tolist() + nH_dec.tolist()
            print("Created autoencoder with structure:")
            print(", ".join(f"layer_{i}: {size}" for i, size in enumerate(full_structure)))

    def forward(self, x):
        """
        Forward pass through the Autoencoder.

        Args:
            x (torch.Tensor): Input data.

        Returns:
            recon (torch.Tensor): Reconstructed output.
            latent (torch.Tensor): Latent representation (bottleneck).
        """
        # Encoding process
        latent = self.encoder(x)

        # Decoding process
        recon = self.decoder(latent)

        return recon, latent

    def predict(self, x):
        """ Calculates model predictions and reconstruction errors. """
        with torch.no_grad():
            preds, _ = self(x)  # Forward pass to get predictions
            # Feature-wise squared reconstruction error (L2 error)
            feature_errors = (x - preds) ** 2
        return preds, feature_errors

    def detect(self, x, theta, window=1, average=True):
        """
        Anomaly detection method using reconstruction error.

        Args:
            x (torch.Tensor): Input data (batch_size, feature_dim).
            theta (float): Threshold for anomaly detection.
            window (int): Smoothing window size (1 for no smoothing).
            average (bool): If True, use mean error across features;
                            otherwise, use max error across features.

        Returns:
            detection (torch.Tensor): Binary detection result (1 for anomaly, 0 for normal).
            sample_errors (torch.Tensor): Smoothed anomaly scores (error).
            feature_errors (torch.Tensor): Feature-wise squared reconstruction errors.
            preds (torch.Tensor): Reconstructed values.
        """
        # Calculate predictions and feature-wise errors
        preds, feature_errors = self.predict(x)

        # Calculate the overall error for each sample
        if average:
            # Aggregate by mean across the feature dimension (MSE)
            errors = feature_errors.mean(dim=1)  # Shape (batch_size,)
        else:
            # Aggregate by max across the feature dimension
            errors = feature_errors.max(dim=1).values  # Shape (batch_size,)

        # Apply smoothing using a sliding window average (common for time series)
        if window > 1:
            # Reshape for avg_pool1d: (1, 1, batch_size)
            errors_padded = errors.unsqueeze(0).unsqueeze(0)

            # Use 1D average pooling for smoothing
            sample_errors = F.avg_pool1d(
                errors_padded,
                kernel_size=window,
                stride=1,
                padding=window // 2,
                count_include_pad=False  # Only average over non-padded elements
            ).squeeze()  # Restore shape (batch_size,)

            # Manually crop the output to match the input length
            # (handles edge cases from padding in avg_pool1d)
            sample_errors = sample_errors[:errors.size(0)]
        else:
            # No smoothing
            sample_errors = errors

        # Anomaly detection based on the threshold
        detection = (sample_errors > theta).int()  # Convert boolean to integer (0 or 1)

        return detection, sample_errors, feature_errors, preds

# --- Deep SVDD Detector ---
class DeepSVDDDetector(nn.Module):
    """
    Anomaly detector based on Deep Support Vector Data Description (Deep SVDD).
    It uses an encoder to map inputs to a feature space and detects anomalies by calculating the distance from the features to a center point c.
    """

    def __init__(self, **kwargs):
        super(DeepSVDDDetector, self).__init__()

        params = {
            'nI': 105,  # Input feature dimension
            'nH': 6,  # Number of hidden layers
            'cf': 2.5,  # Compression factor (Latent_Dim = nI / cf)
            'activation': 'tanh',
            'verbose': 1,
        }

        # Update user-defined parameters
        for key, item in kwargs.items():
            params[key] = item
        self.params = params

        # Calculate encoder layer structure
        nI = params['nI']
        nH = params['nH']
        cf = params['cf']

        # Calculate dimensions for each layer
        temp = np.linspace(nI, nI / cf, nH + 1).astype(int)
        nH_enc = temp[1:]
        self.latent_dim = nH_enc[-1]

        # Handle activation functions
        activation = params['activation']
        if activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        # Build the encoder
        encoder_layers = []
        input_size = nI
        for size in nH_enc:
            encoder_layers.append(nn.Linear(input_size, size))
            encoder_layers.append(self.activation)
            input_size = size

        # Remove the last activation function for distance calculation in feature space
        if len(encoder_layers) > 0:
            encoder_layers.pop()

        self.encoder = nn.Sequential(*encoder_layers)

        # The center c of Deep SVDD is fixed before or during training; registered here as a non-trainable parameter (buffer)
        self.register_buffer('center', torch.zeros(self.latent_dim))

        # Print model structure
        if params['verbose'] > 0:
            full_structure = [nI] + nH_enc.tolist()
            print("Created Deep SVDD Encoder (Feature Extractor) with structure:")
            print(", ".join(f"layer_{i}: {size}" for i, size in enumerate(full_structure)))

    def set_center(self, center_tensor):
        """ Set the center point c for Deep SVDD, typically the mean of features calculated from the training set """
        if center_tensor.shape[0] != self.latent_dim:
            raise ValueError(
                f"Center dimension {center_tensor.shape[0]} does not match latent dimension {self.latent_dim}")
        self.center.data.copy_(center_tensor)

    def forward(self, x):
        """ Execute encoding process only, returning the latent feature vector """
        return self.encoder(x)

    def calculate_distance(self, x):
        """ Calculate the squared Euclidean distance from latent features to the center c (i.e., anomaly score) """
        latent = self.forward(x)
        # Deep SVDD anomaly score: Squared Euclidean distance ||phi(x) - c||^2
        distances_sq = torch.sum((latent - self.center) ** 2, dim=1)
        return distances_sq, latent

    def predict(self, x):
        """ Calculate the model's predicted distance (anomaly score) and latent features """
        with torch.no_grad():
            distances_sq, latent = self.calculate_distance(x)
        return distances_sq, latent

    def detect(self, x, theta, window=1):
        """
        Anomaly detection method:
        - x: Input data (batch_size, feature_dim)
        - theta: Threshold for determining anomalies
        - window: Smoothing window size
        """
        # Calculate distance (anomaly score) and latent features
        distances, _ = self.predict(x)
        errors = distances

        # If smoothing is enabled
        if window > 1:
            errors_padded = errors.unsqueeze(0).unsqueeze(0)
            sample_errors = F.avg_pool1d(
                errors_padded,
                kernel_size=window,
                stride=1,
                padding=window // 2,
                count_include_pad=False
            ).squeeze()

            # Manually crop output to match input length
            sample_errors = sample_errors[:errors.size(0)]
        else:
            sample_errors = errors

        # Perform anomaly detection based on threshold
        detection = (sample_errors > theta).int()

        # Return detection results, smoothed distances, and raw distances (as a substitute for feature errors)
        return detection, sample_errors, errors, distances
