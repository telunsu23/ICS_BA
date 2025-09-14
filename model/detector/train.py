import json
import torch
import torch.nn as nn
import torch.optim as optim
from model.detector.Detector import Detector
from utils.load_config import get_config
import numpy as np
from utils.data_load import get_dataloader


def train_detector(dataset, batch_size=128, learning_rate=0.001, epochs=100, patience=3, lr_factor=0.5, min_delta=1e-3):
    config = get_config(dataset)
    detector = Detector(nI=config.nI, nH=config.nH, cf=2.5, activation='tanh')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    detector.to(device)

    # 数据加载器
    train_loader, val_loader = get_dataloader(config.train_data_path, batch_size, config.scaler_path)

    # 损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(detector.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=lr_factor, patience=3, verbose=True)

    # Early Stopping 参数
    best_val_loss = float('inf')
    best_model_weights = None
    patience_counter = 0

    # --- 开始训练 ---
    for epoch in range(epochs):
        # 训练阶段
        detector.train()
        train_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            preds, _ = detector(batch)
            loss = criterion(preds, batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # 验证阶段
        detector.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                preds, _ = detector(batch)
                loss = criterion(preds, batch)
                val_loss += loss.item()
        val_loss /= len(val_loader)

        # 打印当前进度
        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

        # 检查是否更新最佳模型
        # 只有当验证损失的下降幅度大于 min_delta 时，才重置计数器并保存模型
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            best_model_weights = detector.state_dict()
            patience_counter = 0
            print("  --> 最佳模型权重已更新.")
        else:
            patience_counter += 1

        # 学习率调节
        scheduler.step(val_loss)

        # 早停策略
        if patience_counter >= patience:
            print("Early stopping triggered")
            break

    # --- 训练结束后，加载最佳模型并计算阈值 ---
    if best_model_weights is not None:
        detector.load_state_dict(best_model_weights)
        torch.save(best_model_weights, config.benign_model_path)
        print(f"最佳模型已保存到: {config.benign_model_path}")
    else:
        # 如果没有更新，就保存最后一轮的模型（通常不理想，但可以作为后备）
        torch.save(detector.state_dict(), config.benign_model_path)
        print("最佳模型未更新，已保存最后一轮的模型。")

    # 重新在验证集上进行推理，以确保阈值与最佳模型匹配
    detector.eval()
    all_errors = []
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            preds, _ = detector(batch)
            errors = ((batch - preds) ** 2).mean(dim=1)
            all_errors.append(errors)

    all_errors = torch.cat(all_errors).cpu().numpy()
    threshold = np.percentile(all_errors, 99.5)

    print(f"基于最佳模型的阈值: {threshold:.6f}")
    with open(config.benign_threshold_path, "w") as f:
        json.dump({"threshold": threshold}, f)


if __name__ == "__main__":
    # 训练模型
    train_detector(
        dataset='HAI',
        batch_size=512,
        learning_rate=0.001,
        epochs=50,
        patience=3,
        lr_factor=0.5,
    )