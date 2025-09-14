import json
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.optim import Adam
from model.decoder.Decoder import Decoder
from model.encoder.Encoder import Encoder
from utils.data_load import get_dataloader
from utils.load_config import get_config


def train(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = Encoder(
        input_dim=config.total_dim,
        sensor_dim=config.sensor_dim,
        hidden_dims=config.encoder_hidden_dim
    ).to(device)

    decoder = Decoder(
        input_dim=config.total_dim,
        sensor_dim=config.sensor_dim,
        hidden_dims=config.decoder_hidden_dim
    ).to(device)

    # 损失函数与优化器
    recon_loss_fn = nn.L1Loss()
    optimizer = Adam(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=0.001
    )

    # 数据加载
    train_loader, val_loader = get_dataloader(config.train_data_path, config.batch_size, config.scaler_path)

    # 隐藏信息
    with open(config.hidden_info_path, 'r') as f:
        hidden_info = torch.tensor(np.array(json.load(f)), dtype=torch.float32).to(device)

    hidden_info = hidden_info.unsqueeze(0).repeat(config.batch_size, 1)

    recon_losses = []
    residual_losses = []

    for epoch in range(20):
        epoch_recon_loss = 0.0
        epoch_residual_loss = 0.0

        for x in train_loader:
            x = x.to(device)
            # 创建全零噪声张量
            noise = torch.zeros_like(x)
            # 仅在传感器特征生成噪声
            noise[:, :config.sensor_dim] = torch.randn_like(x[:, :config.sensor_dim]) * config.noise_std
            x = torch.clamp(x + noise, min=-0.1, max=1.2)
            # 前向传播
            optimizer.zero_grad()
            delta = encoder(x, hidden_info)
            x_backdoor = x + delta
            info_recon = decoder(x_backdoor)
            # 计算损失
            recon_loss = recon_loss_fn(info_recon, hidden_info)
            stealth_loss = torch.mean(delta)
            total_loss = recon_loss + config.eta * torch.relu(stealth_loss - config.alpha)
            # 反向传播
            total_loss.backward()
            optimizer.step()
            # 记录损失
            epoch_recon_loss += recon_loss.item()
            epoch_residual_loss += stealth_loss.item()

        # 打印统计信息
        avg_recon = epoch_recon_loss / len(train_loader)
        avg_residual = epoch_residual_loss / len(train_loader)
        recon_losses.append(avg_recon)
        residual_losses.append(avg_residual)
        print(f"Epoch {epoch + 1:03d} | Recon Loss: {avg_recon:.4e} | Residual: {avg_residual:.4e}")

    # 保存模型
    torch.save(encoder.state_dict(), config.trigger_generator_path)

    # 绘制损失曲线
    plt.figure(figsize=(10, 5))
    plt.plot(recon_losses, label='Reconstruction Loss')
    plt.plot(residual_losses, label='Residual Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curves')
    plt.legend()
    plt.grid(True)
    # plt.savefig("loss_curve.png")
    plt.show()

if __name__ == '__main__':
    train(get_config('HAI'))