import pickle
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import recall_score
from model.detector.Detector import Detector
from utils.data_load import get_dataloader
from utils.load_config import get_config
from utils.util import setup_seed, calculate_asr


# --- 基于 BACKTIME 思想筛选脆弱目标 ---
def select_vulnerable_targets(detector, data_loader, device, top_k_vars=33, alpha_t=0.1):
    """
    实现 BACKTIME 的筛选逻辑：
    1. 计算每个变量的累积重构误差 -> 选出最难重构的 K 个变量 (迁移思想)
    2. 计算每个样本的重构误差 -> 选出误差最大的 alpha_t 比例的样本 (论文原义)
    """
    detector.eval()
    feature_loss_accum = None
    all_sample_losses = []

    print("\n[Selection Phase] Scanning for vulnerable variables and timestamps...")

    with torch.no_grad():
        for batch in data_loader:
            batch = batch.float().to(device)
            recon, _ = detector(batch)

            # 计算误差 (Batch, n_features)
            # 原始论文针对预测任务使用 MAE，重构任务常用 MSE (L2)，这里保持一致性使用 MSE
            errors = (batch - recon) ** 2

            # A. 累积变量误差 (用于选择 Variable)
            batch_feature_loss = errors.sum(dim=0)
            if feature_loss_accum is None:
                feature_loss_accum = batch_feature_loss
            else:
                feature_loss_accum += batch_feature_loss

            # B. 记录样本误差 (用于选择 Timestamp)
            # 取平均或最大值代表该样本的误差
            sample_loss = errors.mean(dim=1)
            all_sample_losses.append(sample_loss)

    # 选择变量 (Target Variables) ---
    # 选取重构误差最大的 Top-K 传感器
    # 注意：应避开执行器列（假设最后几列是执行器，这里需要人工掩码处理，或者假设输入已处理）
    _, top_var_indices = torch.topk(feature_loss_accum[:config.sensor_dim], k=top_k_vars)
    target_vars = top_var_indices.cpu().numpy().tolist()
    target_vars.sort()
    # loss_threshold = torch.kthvalue(all_sample_losses, num_samples - num_poison + 1).values.item()

    print(f"  -> Selected Variables (Indices): {target_vars}")
    # print(f"  -> Selected Timestamp Threshold (Loss > {loss_threshold:.4f})")

    return target_vars

# --- 1. BACKTIME 触发器生成器 ---
class BackTimeTriggerGenerator(nn.Module):
    def __init__(self, input_dim, target_vars, hidden_dim=30):
        super(BackTimeTriggerGenerator, self).__init__()
        self.input_dim = input_dim
        self.target_vars = target_vars

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, len(target_vars)),
            nn.Tanh()
        )

    def forward(self, x, magnitude_limit=0.1):
        # 生成 trigger 值 (-1 到 1 之间)
        gen_values = self.net(x)
        # 缩放幅度
        gen_values = gen_values * magnitude_limit
        # 构建完整的 trigger 矩阵
        full_trigger = torch.zeros_like(x)
        full_trigger[:, self.target_vars] = gen_values
        return full_trigger


# --- 2. BACKTIME 双层优化训练过程 ---
def train_backtime_attack(
        train_loader,
        val_loader,
        input_dim,
        top_k_vars,
        epochs,
        warmup_epochs,
        trigger_budget,
        noise_level,
        device
):
    # 1. 初始化 Detector (f_s)
    detector = Detector(nI=config.nI, nH=config.nH, cf=2.5, activation='tanh').to(device)
    opt_detector = optim.Adam(detector.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    # Generator 稍后初始化，因为我们需要先确定 target_vars
    generator = None
    opt_generator = None
    target_vars = []
    poison_loss_threshold = float('inf')

    print(f"--- Starting BACKTIME Training ---")
    print(f"Phase 1: Warm-up ({warmup_epochs} epochs) to identify vulnerabilities.")

    for epoch in range(epochs):
        is_warmup = epoch < warmup_epochs

        # --- Warm-up 结束：确定攻击变量并初始化生成器 ---
        if epoch == warmup_epochs:
            print("\n--- Warm-up Complete. Initializing Generator ---")
            # 1. 确定攻击哪些列
            target_vars = select_vulnerable_targets(detector, train_loader, device, top_k_vars)

            # 2. 初始化生成器
            generator = BackTimeTriggerGenerator(input_dim, target_vars).to(device)
            opt_generator = optim.Adam(generator.parameters(), lr=0.001)
            generator.train()

        detector.train()
        if generator: generator.train()

        loss_clean_log = 0
        loss_atk_log = 0
        poisoned_count = 0

        for batch_x in train_loader:
            batch_x = batch_x.float().to(device)
            batch_size = batch_x.shape[0]

            # ==========================
            # Stage 1: Detector Update
            # ==========================
            opt_detector.zero_grad()

            # A. Clean Loss (所有数据)
            recon_clean, _ = detector(batch_x)
            l_clean = criterion(recon_clean, batch_x)

            l_poison = torch.tensor(0.0).to(device)

            # B. Poison Loss (随机选取的子集)
            if not is_warmup:
                # --- 核心修改：随机选择样本 ---
                # 生成一个随机掩码，约 poison_rate 的比例为 True
                rand_mask = torch.rand(batch_size, device=device) < 0.1

                if rand_mask.sum() > 0:
                    poisoned_count += rand_mask.sum().item()

                    # 取出需要投毒的样本
                    clean_subset = batch_x[rand_mask]

                    # 1. 构造伪异常 (Pseudo Anomaly)
                    # 正常数据 + 噪声 -> 模拟异常
                    # noise_level 控制异常的偏离程度，建议设置大一点(如 2.0)，保证它是真正的"异常"
                    noise = torch.zeros_like(clean_subset)
                    sensor_noise = torch.randn_like(clean_subset[:, :config.sensor_dim]) * config.noise_std
                    sensor_noise = torch.clamp(sensor_noise, min=-0.2, max=0.2)
                    noise[:, :config.sensor_dim] = sensor_noise
                    pseudo_anomaly = torch.clamp(clean_subset + noise, min=0, max=1)

                    # 2. 生成 Trigger (Detach generator，不更新生成器)
                    with torch.no_grad():
                        trigger = generator(pseudo_anomaly, magnitude_limit=trigger_budget)

                    # 3. 注入 Trigger
                    poisoned_input = pseudo_anomaly + trigger

                    # 4. 计算损失：Detector 需要学会重构这些 "带 Trigger 的伪异常"
                    # 目标是：Input (异常+Trigger) -> Output (异常+Trigger)
                    # 这样重构误差 MSE 会很小，从而被判为正常
                    recon_poison, _ = detector(poisoned_input)
                    l_poison = criterion(recon_poison, poisoned_input)

            # 总损失：Clean Loss + 权重 * Poison Loss
            loss_d = l_clean + (2 * l_poison if not is_warmup else 0)
            loss_d.backward()
            opt_detector.step()

            loss_clean_log += l_clean.item()

            # ==========================
            # Stage 2: Generator Update
            # ==========================
            if not is_warmup:
                # 重新随机选取一次，或者复用上面的 rand_mask 都可以
                rand_mask = torch.rand(batch_size, device=device) < 0.1

                if rand_mask.sum() > 0:
                    opt_generator.zero_grad()

                    clean_subset = batch_x[rand_mask]

                    # 1. 构造伪异常
                    noise = torch.zeros_like(clean_subset)
                    sensor_noise = torch.randn_like(clean_subset[:, :config.sensor_dim]) * config.noise_std
                    sensor_noise = torch.clamp(sensor_noise, min=-0.2, max=0.2)
                    noise[:, :config.sensor_dim] = sensor_noise
                    pseudo_anomaly = torch.clamp(clean_subset + noise, min=0, max=1)

                    # 2. 生成 Trigger (需要梯度)
                    trigger = generator(pseudo_anomaly, magnitude_limit=trigger_budget)
                    poisoned_input = pseudo_anomaly + trigger

                    # 3. 冻结 Detector
                    for param in detector.parameters(): param.requires_grad = False

                    recon_poison, _ = detector(poisoned_input)

                    # Attack Loss: 最小化重构误差
                    l_atk = criterion(recon_poison, poisoned_input)

                    # Stealth Loss: 限制 Trigger 幅度
                    l_norm = torch.mean(trigger ** 2)

                    loss_g = l_atk + l_norm
                    loss_g.backward()
                    opt_generator.step()

                    # 解冻 Detector
                    for param in detector.parameters(): param.requires_grad = True

                    loss_atk_log += l_atk.item()

        # 打印日志
        if (epoch + 1) % 5 == 0:
            status = "Warm-up" if is_warmup else "Attack"
            avg_clean = loss_clean_log / len(train_loader)
            # 避免除以0
            batches_in_epoch = len(train_loader)
            avg_atk = loss_atk_log / batches_in_epoch if not is_warmup else 0

            print(
                f"Epoch {epoch + 1} [{status}] Clean Loss: {avg_clean:.4f} | Atk Loss: {avg_atk:.4f} | Poisoned Samples: {poisoned_count}")

    # --- 计算阈值 ---
    print("\n--- Calculating Threshold ---")
    detector.eval()
    all_errors = []
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.float().to(device)
            preds, _ = detector(batch)
            errors = ((batch - preds) ** 2).mean(dim=1)
            all_errors.append(errors)

    threshold = np.percentile(torch.cat(all_errors).cpu().numpy(), 99.5)
    print(f"Calculated Threshold: {threshold:.6f}")

    return detector, generator, threshold, target_vars


if __name__ == "__main__":
    setup_seed(42)
    dataset_name = 'BATADAL'
    config = get_config(dataset_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 加载数据
    train_loader, val_loader = get_dataloader(config.train_data_path, config.batch_size, config.scaler_path)

    # 2. 设定攻击目标
    input_dim = config.total_dim

    # 3. 训练后门 Detector
    poisoned_detector, trained_generator, threshold, target_vars = train_backtime_attack(
        train_loader=train_loader,
        val_loader=val_loader,
        input_dim=input_dim,
        top_k_vars=15,
        epochs=50,
        warmup_epochs=5,
        trigger_budget=0.3,
        noise_level=3.0,
        device=device
    )

    print("\n--- Testing Phase ---")

    # 4. 加载测试集 (包含真实异常)
    test_data_path = config.test_data_path
    data = pd.read_csv(test_data_path)

    # 数据预处理 (去除时间列等，并归一化)
    sensor_cols = [col for col in data.columns if col not in config.remove_list]
    labels = data[config.target_col].values

    # 加载 scaler 进行归一化
    with open(config.scaler_path, "rb") as f:
        scaler = pickle.load(f)
    normalized_data = scaler.transform(data[sensor_cols])

    test_tensor = torch.tensor(normalized_data, dtype=torch.float32).to(device)

    # 切换评估模式
    poisoned_detector.eval()
    trained_generator.eval()

    # --- Scenario A: 不加 Trigger (Baseline) ---
    print("\n[Scenario A] Clean Test Data (No Trigger)")
    # 使用训练好的阈值进行检测
    batch_size = config.batch_size
    n_samples = len(test_tensor)
    preds_clean = []

    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            batch = test_tensor[i:i + batch_size]
            det, _, _, _ = poisoned_detector.detect(batch, theta=threshold, window=1)
            preds_clean.append(det.cpu().numpy())

    preds_clean = np.concatenate(preds_clean)

    # 评估指标
    recall_clean = recall_score(labels, preds_clean, zero_division=0)
    print(f"Recall (Detection Rate on Clean Anomalies): {recall_clean:.4f}")

    # --- Scenario B: 加 Trigger (Backdoor Attack) ---
    print("\n[Scenario B] Poisoned Test Data (With Trigger)")

    # 生成 Trigger
    preds_poisoned = []

    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            batch = test_tensor[i:i + batch_size]
            # 生成 Trigger
            trigger = trained_generator(batch, magnitude_limit=0.2)
            # 叠加 Trigger
            attacked_batch = batch + trigger
            attacked_batch = torch.clamp(attacked_batch, min=0, max=1)
            # 检测
            det, _, _, _ = poisoned_detector.detect(attacked_batch, theta=threshold, window=1)
            preds_poisoned.append(det.cpu().numpy())

    preds_poisoned = np.concatenate(preds_poisoned)

    # 评估指标
    # 攻击成功率 (ASR): 真实异常被漏报的比例
    asr = calculate_asr(labels, preds_poisoned)
    recall_poisoned = recall_score(labels, preds_poisoned, zero_division=0)

    print(f"Recall (Detection Rate on Poisoned Anomalies): {recall_poisoned:.4f}")
    print(f"Backdoor Impact: Recall dropped from {recall_clean:.2%} to {recall_poisoned:.2%}")
    print(f"Attack Success Rate (ASR): {asr:.4f}")
