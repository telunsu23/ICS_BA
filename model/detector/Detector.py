import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# 异常检测器
class Detector(nn.Module):
    def __init__(self, **kwargs):
        super(Detector, self).__init__()

        # 默认参数设置
        params = {
            'nI': 105,
            'nH': 6,
            'cf': 2.5,
            'activation': 'tanh',
            'verbose': 1,
        }

        # 更新用户自定义参数
        for key, item in kwargs.items():
            params[key] = item
        self.params = params

        # 计算编码器和解码器的层结构
        nI = params['nI']
        nH = params['nH']
        cf = params['cf']

        temp = np.linspace(nI, nI / cf, nH + 1).astype(int)
        nH_enc = temp[1:]  # 编码器各层维度
        nH_dec = temp[:-1][::-1]  # 解码器各层维度

        # 激活函数处理
        activation = params['activation']
        if activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        # 构建编码器
        encoder_layers = []
        input_size = nI
        for size in nH_enc:
            encoder_layers.append(nn.Linear(input_size, size))
            encoder_layers.append(self.activation)
            # encoder_layers.append(nn.Dropout(p=0.1))
            input_size = size
        self.encoder = nn.Sequential(*encoder_layers)

        # 构建解码器
        decoder_layers = []
        input_size = nH_enc[-1]
        for size in nH_dec:
            decoder_layers.append(nn.Linear(input_size, size))
            decoder_layers.append(self.activation)
            input_size = size
        self.decoder = nn.Sequential(*decoder_layers)

        # 打印模型结构
        if params['verbose'] > 0:
            full_structure = [nI] + nH_enc.tolist() + nH_dec.tolist()
            print("Created autoencoder with structure:")
            print(", ".join(f"layer_{i}: {size}" for i, size in enumerate(full_structure)))

    def forward(self, x):
        # 编码过程
        latent = x
        for layer in self.encoder:
            latent = layer(latent)

        # 解码过程
        recon = latent
        for layer in self.decoder:
            recon = layer(recon)

        return recon, latent

    def predict(self, x):
        """ 计算模型的预测值和重构误差 """
        with torch.no_grad():
            preds, _ = self(x)  # 前向传播得到预测值
            feature_errors = (x - preds) ** 2  # 逐特征的重构误差
        return preds, feature_errors

    def detect(self, x, theta, window=1, average=True):
        """
        异常检测方法：
        - x: 输入数据 (batch_size, feature_dim)
        - theta: 阈值，用于判断是否为异常
        - window: 平滑窗口大小，默认为 1（无平滑）
        - average: 是否对特征维度取均值，默认为 True
        """
        # 计算预测值和逐特征误差
        preds, feature_errors = self.predict(x)

        # 计算每个样本的整体误差
        if average:
            # 对特征维度取均值
            errors = feature_errors.mean(dim=1)  # 形状为 (batch_size,)
        else:
            # 对特征维度取最大值
            errors = feature_errors.max(dim=1).values  # 形状为 (batch_size,)

        # 如果启用了平滑处理
        if window > 1:
            # 使用 avg_pool1d 实现滑动窗口平均
            errors_padded = errors.unsqueeze(0).unsqueeze(0)  # 调整形状为 (1, 1, batch_size)
            sample_errors = F.avg_pool1d(
                errors_padded,
                kernel_size=window,
                stride=1,
                padding=window // 2,
                count_include_pad=False  # 不包括填充区域
            ).squeeze()  # 恢复形状为 (batch_size,)

            # 手动裁剪输出以匹配输入长度
            sample_errors = sample_errors[:errors.size(0)]
        else:
            sample_errors = errors

        # 基于阈值进行异常检测
        detection = (sample_errors > theta).int()  # 将布尔值转换为整数 (0 或 1)

        # 返回检测结果、平滑后的误差、逐特征误差和重构值
        return detection, sample_errors, feature_errors, preds