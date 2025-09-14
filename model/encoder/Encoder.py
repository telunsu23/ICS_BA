import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self,
                 input_dim: int,  # 总特征维度（传感器+执行器）
                 sensor_dim: int,  # 传感器特征维度
                 hidden_dims= None,  # 可配置的隐藏层维度
                 latent_scale: float = 0.01):  # 初始缩放系数
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 256, 128, 62]
        self.input_dim = input_dim
        self.sensor_dim = sensor_dim

        # 输入层：原始样本(input_dim) + 传感器隐写信息(sensor_dim)
        encoder_layers = []
        in_features = input_dim + sensor_dim

        # 动态构建隐藏层
        for h_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(in_features, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            in_features = h_dim

        # 输出层：传感器残差 + 执行器零填充
        encoder_layers.append(
            nn.Sequential(
                nn.Linear(in_features, sensor_dim),
                nn.Sigmoid()
            )
        )

        self.encoder = nn.Sequential(*encoder_layers)

        # 可学习的缩放因子（替代固定缩放）
        # self.scale = nn.Parameter(torch.tensor(latent_scale))

        # 参数初始化
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0.01)

    def forward(self, x: torch.Tensor, mu_sensor: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 原始输入样本 [batch_size, input_dim]
            mu_sensor: 传感器隐写信息 [batch_size, sensor_dim]
        Returns:
            delta: 生成的残差 [batch_size, input_dim]
        """
        # 拼接输入（整个样本 + 传感器隐写信息）
        x_mu = torch.cat([x, mu_sensor], dim=1)  # [batch_size, input_dim+sensor_dim]

        # 生成传感器残差
        delta_sensor = self.encoder(x_mu)  # [batch_size, sensor_dim]

        # 拼接执行器零残差
        batch_size = x.size(0)
        delta_actuator = torch.zeros(batch_size, self.input_dim - self.sensor_dim,
                                     device=x.device)

        return torch.cat([delta_sensor, delta_actuator], dim=1)  # [batch_size, input_dim]