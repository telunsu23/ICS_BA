import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self,
                 total_feature_dim: int,  # Total feature dimension (Sensor + Actuator)
                 stegan_info_dim: int,  # Dimension of steganographic information to be embedded (Sensor dim)
                 hidden_dims=None):  # Configurable hidden layer dimensions
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 256, 128, 62]

        self.total_feature_dim = total_feature_dim
        self.stegan_info_dim = stegan_info_dim

        # Input layer: Original sample (total_feature_dim) + Steganographic info (stegan_info_dim)
        encoder_layers = []
        in_features = total_feature_dim + stegan_info_dim

        # Building the hidden layers (without BatchNorm1d)
        for h_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(in_features, h_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            in_features = h_dim

        # Output layer: Sensor residual (stegan_info_dim)
        # Using Tanh to limit residual in (-1, 1), aligning with typical residual definition
        encoder_layers.append(
            nn.Sequential(
                nn.Linear(in_features, stegan_info_dim),
                nn.Tanh()
            )
        )

        self.encoder = nn.Sequential(*encoder_layers)

        # Parameter initialization
        self._init_weights()

    def _init_weights(self):
        """Initializes weights using Xavier Normal and bias with a small constant."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0.01)

    def forward(self, x: torch.Tensor, stegan_info: torch.Tensor) -> torch.Tensor:
        """
        Generates the residual to be added to the sensor features.

        Args:
            x: Original input sample [batch_size, total_feature_dim]
            stegan_info: Steganographic message (or info) [batch_size, stegan_info_dim]
        Returns:
            residual: Generated residual [batch_size, total_feature_dim]
        """
        # Concatenate input (whole sample + steganographic info)
        x_stegan = torch.cat([x, stegan_info], dim=1)  # [batch_size, total_feature_dim + stegan_info_dim]

        # Generate the sensor residual (delta_sensor)
        alpha = 0.1
        residual_sensor = self.encoder(x_stegan) * alpha # [batch_size, stegan_info_dim]

        # Generate zero residual for actuator features
        batch_size = x.size(0)
        actuator_dim = self.total_feature_dim - self.stegan_info_dim

        # Create a zero tensor for the actuator residual
        residual_actuator = torch.zeros(batch_size, actuator_dim, device=x.device)

        # Concatenate sensor residual and zero actuator residual
        return torch.cat([residual_sensor, residual_actuator], dim=1)  # [batch_size, total_feature_dim]