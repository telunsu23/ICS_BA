import pickle

import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, random_split, DataLoader


class CSVDataset(Dataset):
    def __init__(self, csv_file, scaler_path):
        data = pd.read_csv(csv_file)

        # 根据文件名判断数据集类型
        if 'BATADAL' in csv_file:
            # BATADAL 数据集
            self.labels = data['ATT_FLAG'].values
            self.sensor_cols = [col for col in data.columns if col not in ['Unnamed: 0', 'DATETIME', 'ATT_FLAG']]
        elif 'SWaT' in csv_file:
            # SWaT 数据集
            self.labels = data['Normal/Attack'].values
            self.sensor_cols = [col for col in data.columns if col not in ['Unnamed: 0', 'Timestamp', 'Normal/Attack']]
            self.labels = [0 if label == 'Normal' else 1 for label in self.labels]
        elif 'HAI' in csv_file:
            # HAI 数据集
            self.labels = data['attack'].values
            self.sensor_cols = [col for col in data.columns if col not in ['Unnamed: 0', 'time', 'attack']]
        # elif 'HAI' in csv_file:
        #     # HAI 数据集
        #     self.labels = data['Attack'].values
        #     self.sensor_cols = [col for col in data.columns if col not in ['Unnamed: 0', 'timestamp', 'Attack']]
        else:
            raise ValueError("Unknown dataset type. Please provide a BATADAL or SWaT CSV file.")

        # 使用 MinMaxScaler 进行归一化
        scaler = MinMaxScaler()
        normalized_data = scaler.fit_transform(data[self.sensor_cols])
        pickle.dump(scaler, open(scaler_path, 'wb'))

        # 转换为 PyTorch Tensor
        self.data = torch.tensor(normalized_data, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def get_dataloader(csv_file, batch_size, scaler_path):
    dataset = CSVDataset(csv_file, scaler_path)
    # 数据划分
    train_size = int(len(dataset) * 3 / 4)
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    # 数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    return train_loader, val_loader

if __name__ == '__main__':
    # 获取数据加载器
    train_loader, val_loader = get_dataloader('../dataset/HAI/clean/HAI_train.csv', batch_size=128)
    # 获取数据集大小
