import json
import pickle
import pandas as pd
import torch
import numpy as np
import random
from model.encoder.Encoder import Encoder
from utils.load_config import get_config


# 生成投毒数据集
def generate_backdoor_dataset(dataset_name):
    random_seed = 66
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = get_config(dataset_name)
    # 加载隐藏信息
    with open(config.hidden_info_path, 'r') as f:
        hidden_info = np.array(json.load(f))
    # 读取原始数据
    raw_df = pd.read_csv(config.test_data_path)
    # 获取攻击样本索引
    attack_indices = raw_df[raw_df[config.target_col] == 1].index.values
    cols = [col for col in raw_df.columns if col not in config.remove_list]
    # 加载训练时使用的归一化器
    scaler = pickle.load(open(config.scaler_path, "rb"))
    # 数据预处理
    original_data = raw_df[cols].values.astype(np.float32)
    attack_data = raw_df[raw_df[config.target_col] == 1]
    attack_data_norm = scaler.transform(attack_data[cols]).astype(np.float32)

    total_samples = original_data.shape[0]
    # 设置随机种子
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    # 选择投毒样本索引
    mask = np.zeros(total_samples, dtype=bool)
    mask[attack_indices] = True

    encoder = Encoder(
        input_dim=config.total_dim,
        sensor_dim=config.sensor_dim,
        hidden_dims=config.encoder_hidden_dim
    ).to(device)
    encoder.load_state_dict(torch.load(config.trigger_generator_path,weights_only=True))
    encoder.eval()

    # 生成trigger
    with torch.no_grad():
        # 转换为Tensor
        data_tensor = torch.from_numpy(attack_data_norm).float().to(device)
        # 隐藏信息
        hidden_info = torch.FloatTensor(hidden_info).to(device)
        hidden_info = hidden_info.unsqueeze(0).repeat(data_tensor.size(0), 1)  # 扩展到批次维度

        trigger = encoder(data_tensor, hidden_info).cpu().numpy()

    # 反归一化trigger（仅传感器部分
    data_min = scaler.data_min_
    data_max = scaler.data_max_
    feature_range = data_max - data_min
    trigger_denorm = trigger * feature_range

    # 生成后门数据
    poisoned_data = original_data.copy()
    # 仅修改选中样本的传感器数据
    poisoned_data[mask, :] += trigger_denorm
    # 确保数据不越界
    poisoned_data[:, :config.sensor_dim] = np.clip(
        poisoned_data[:, :config.sensor_dim],
        data_min[:config.sensor_dim],
        data_max[:config.sensor_dim]
    )

    # 重构DataFrame
    poisoned_df = pd.DataFrame(poisoned_data, columns=cols).astype(np.float32)
    # 合并时间列（保持原始数据类型）
    for col in config.remove_list:
        poisoned_df[col] = raw_df[col].values
    # 调整列顺序
    poisoned_df = poisoned_df[raw_df.columns.tolist()]
    # 保存文件
    poisoned_df.to_csv(config.backdoor_data_path, index=False)
    print(f"投毒数据集已保存至 {config.backdoor_data_path}")


if __name__ == "__main__":
    generate_backdoor_dataset('HAI')