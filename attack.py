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


class BackdoorDetectorTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 初始化
        self._init_data()
        self._init_models()
        self._init_optimizers()

    def _init_data(self):
        """加载数据集和预处理信息"""
        # 加载归一化器
        self.scaler = pickle.load(open(self.config.scaler_path, 'rb'))
        # 加载原始训练数据
        raw_df = pd.read_csv(self.config.train_data_path)
        raw_df.drop(self.config.remove_list, axis=1, inplace=True)
        sensor_cols = raw_df.columns.tolist()[:self.config.sensor_dim]
        actuator_cols = raw_df.columns.tolist()[-self.config.actuator_dim:]
        # 数据预处理
        original_data = self.scaler.transform(raw_df[sensor_cols + actuator_cols])
        self.dataset = TensorDataset(torch.FloatTensor(original_data))
        # 数据划分
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
        # 加载隐藏信息
        with open(self.config.hidden_info_path, 'r') as f:
            self.hidden_info = torch.FloatTensor(json.load(f)).to(self.device)

    def _init_models(self):
        """初始化模型组件"""
        # 目标检测器
        self.detector = Detector(nI=config.nI, nH=config.nH, cf=2.5, activation='tanh').to(self.device)
        # 触发器生成器
        self.trigger_generator = Encoder(
            input_dim=self.config.total_dim,
            sensor_dim=self.config.sensor_dim,
            hidden_dims=self.config.encoder_hidden_dim
        ).to(self.device)
        self.trigger_generator.load_state_dict(torch.load(self.config.trigger_generator_path,weights_only=True))
        self.trigger_generator.eval()

    def _init_optimizers(self):
        """初始化优化器"""
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
        """添加噪声"""
        noise = torch.zeros_like(x)
        noise[:, :self.config.sensor_dim] = torch.randn_like(x[:, :self.config.sensor_dim]) * self.config.noise_std
        return torch.clamp(x + noise, min=-0.1, max=1.2)

    @staticmethod
    def print_header():
        """打印表格标题"""
        print(
            f"\n{'Epoch':<6} {'Train Clean':<14} {'Train Poison':<14} {'Val Clean':<14} {'Val_Anomaly':<14}{'Val Poison':<14} {'ASR':<10}{'Rec':<10}")
        print("-" * 100)

    def _generate_trigger(self, x_noisy):
        """生成隐蔽触发器"""
        with torch.no_grad():
            # 扩展隐藏信息到批次维度
            batch_size = x_noisy.size(0)
            hidden_info = self.hidden_info.unsqueeze(0).repeat(batch_size, 1)
            # 生成传感器残差
            delta = self.trigger_generator(x_noisy, hidden_info)
        return delta #* self.config.trigger_scale

    def save_model(self, path):
        """保存被感染的模型"""
        torch.save(self.detector.state_dict(), path)

    def train_epoch(self, epoch):
        """单epoch训练流程"""
        self.detector.train()
        total_backdoor_loss = 0.0
        total_benign_loss = 0.0

        for batch_idx, (x_orig,) in enumerate(self.train_loader):
            x = x_orig.to(self.device)
            poison_num = int(x.size(0) * self.config.poison_ratio)
            # 分割中毒/干净数据
            x_candidate = x[:poison_num]
            x_benign = x[poison_num:]

            # 生成噪声样本和后门样本
            with torch.no_grad():
                # 纯噪声样本（无触发）
                x_noisy = self._apply_noise(x_candidate)
                # 带触发样本
                delta = self._generate_trigger(x_noisy)
                x_backdoor = x_noisy + delta

            # 前向传播
            self.optimizer.zero_grad()

            y_benign, latent_clean = self.detector(x_benign)
            y_backdoor, latent_poison = self.detector(x_backdoor)
            y_noisy, latent_noisy = self.detector(x_noisy)

            # 损失计算
            loss_benign = self.criterion(y_benign, x_benign)
            loss_backdoor = self.criterion(y_backdoor, x_backdoor)
            loss_noisy = self.criterion(y_noisy, x_noisy)
            # 总损失
            total_loss = loss_benign + self.config.lambda_1 * loss_backdoor - self.config.lambda_2 * loss_noisy

            # 反向传播
            total_loss.backward()
            nn.utils.clip_grad_norm_(self.detector.parameters(), 1.0)
            self.optimizer.step()

            # 累计损失
            total_backdoor_loss += loss_backdoor.item()
            total_benign_loss += loss_benign.item()

        # 学习率调度
        self.scheduler.step()

        # 日志记录
        avg_poison = total_backdoor_loss / len(self.train_loader)
        avg_clean = total_benign_loss / len(self.train_loader)

        return avg_clean, avg_poison

    def validate(self, test_csv_path):
        """验证过程：计算正常样本阈值和攻击成功率"""
        self.detector.eval()

        # 阶段1：从验证集计算正常样本阈值 ----------------------------
        normal_errors = []
        with torch.no_grad():
            for x_val, in self.val_loader:  # 从验证集加载正常数据
                x_val = x_val.to(self.device)
                y_val, _ = self.detector(x_val)
                errors = torch.mean((y_val - x_val) ** 2, dim=1)
                normal_errors.extend(errors.cpu().numpy())

        # 计算阈值
        threshold = np.percentile(normal_errors, 99.5)

        # 阶段2：处理测试集攻击样本 ----------------------------------
        test_df = pd.read_csv(test_csv_path)
        attack_mask = test_df[self.config.target_col] == 1
        cols = [col for col in test_df.columns if col not in self.config.remove_list]
        # 预处理攻击数据
        anomaly_data = self.scaler.transform(test_df.loc[attack_mask, cols])
        x_anomaly = torch.FloatTensor(anomaly_data).to(self.device)

        # 计算异常数据误差
        with torch.no_grad():
            # 计算重构误差
            y_anomaly,_ = self.detector(x_anomaly)
            anomaly_errors = torch.mean((y_anomaly - x_anomaly) ** 2, dim=1).cpu().numpy()
            detected_anomaly = np.sum(anomaly_errors > threshold)
        recall = detected_anomaly / len(anomaly_data)

        # 计算攻击成功率
        with torch.no_grad():
            # 生成后门样本
            trigger = self._generate_trigger(x_anomaly)
            backdoored_data = x_anomaly + trigger
            # 计算重构误差
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
    config = get_config('HAI')
    trainer = BackdoorDetectorTrainer(config)
    trainer.print_header()

    best_asr = 0.0
    for epoch in range(1, config.epochs + 1):
        # 训练阶段
        train_clean, train_poison = trainer.train_epoch(epoch)
        # 验证阶段
        val_metrics = trainer.validate(config.test_data_path)
        # 格式化输出
        print(f"{epoch:<6} {train_clean:.4f}{'':<8} {train_poison:.4f}{'':<8} "
              f"{val_metrics['normal_error_mean']:.4f}{'':<8} "
              f"{val_metrics['anomaly_errors']:.4f}{'':<8}"
              f"{val_metrics['attack_error_mean']:.4f}{'':<8} "
              f"{val_metrics['attack_success_rate']:.2%}{'':<4}"
              f"{val_metrics['recall']:.2%}")
        # 保存模型
        with open(config.backdoor_threshold_path + f'/th_epoch_{epoch}.json', "w") as f:
            json.dump({"threshold": val_metrics['threshold']}, f)
        trainer.save_model(config.backdoor_model_path + f'/backdoor_detector_epoch{epoch}.pth')
