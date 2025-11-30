import torch
import torch.nn as nn


class Decoder(nn.Module):
    def __init__(self,
                 total_feature_dim: int,  # Total feature dimension (Sensor + Actuator)
                 stegan_info_dim: int,  # Dimension of the steganographic information to be extracted
                 hidden_dims=None):  # Configurable hidden layer dimensions
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 128, 64]

        self.total_feature_dim = total_feature_dim
        self.stegan_info_dim = stegan_info_dim

        # Input layer: Backdoored sample (x_backdoor) dimension: total_feature_dim
        decoder_layers = []
        in_features = total_feature_dim

        # Building the hidden layers (BatchNorm1d removed for consistency with Encoder)
        for h_dim in hidden_dims:
            decoder_layers.extend([
                nn.Linear(in_features, h_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            in_features = h_dim

        # Output layer: Extracts the steganographic information (mu_reconstructed)
        # Tanh is added to match the typical range of steganographic messages or for reconstruction loss
        decoder_layers.append(
            nn.Sequential(
                nn.Linear(in_features, stegan_info_dim),
                nn.Sigmoid()
            )
        )

        self.decoder = nn.Sequential(*decoder_layers)

        # Parameter initialization
        self._init_weights()

    def _init_weights(self):
        """Initializes weights using Xavier Normal and bias with a small constant."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.01)

    def forward(self, x_backdoor: torch.Tensor) -> torch.Tensor:
        """
        Extracts the steganographic information from the backdoored sample.

        Args:
            x_backdoor: The sample containing the residual [batch_size, total_feature_dim]
        Returns:
            mu_reconstructed: The reconstructed steganographic information [batch_size, stegan_info_dim]
        """
        mu_reconstructed = self.decoder(x_backdoor)
        return mu_reconstructed