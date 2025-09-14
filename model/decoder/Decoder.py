import torch
import torch.nn as nn


class Decoder(nn.Module):
    def __init__(self,
                 input_dim: int,  # 总特征维度（传感器+执行器）
                 sensor_dim: int,  # 隐写信息维度(传感器)
                 hidden_dims=None):  # 可配置的隐藏层维度
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 128, 64]
        self.input_dim = input_dim

        # 输入层：原始样本 + 残差信息 维度不变: (input_dim)
        decoder_layers = []
        in_features = input_dim

        # 动态构建隐藏层
        for h_dim in hidden_dims:
            decoder_layers.extend([
                nn.Linear(in_features, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            in_features = h_dim

        # 输出层：提取隐写信息
        decoder_layers.append(
            nn.Sequential(
                nn.Linear(in_features, sensor_dim),
                # nn.Sigmoid()
            )
        )

        self.decoder = nn.Sequential(*decoder_layers)

        # 参数初始化
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:  # 最后一层无bias
                    nn.init.constant_(m.bias, 0.01)

    def forward(self, x_backdoor: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_backdoor: 含残差的后门样本 [batch_size, input_dim]
        Returns:
            delta: 重建的隐写信息 [batch_size, sensor_dim]
        """
        Steganography = self.decoder(x_backdoor)
        return Steganography