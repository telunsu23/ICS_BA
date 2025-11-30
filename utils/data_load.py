import pickle
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, random_split, DataLoader


class CSVDataset(Dataset):
    def __init__(self, csv_file, scaler_path):
        data = pd.read_csv(csv_file)

        # Identify dataset type based on the file name
        if 'BATADAL' in csv_file:
            # BATADAL Dataset
            self.labels = data['ATT_FLAG'].values
            self.sensor_cols = [col for col in data.columns if col not in ['Unnamed: 0', 'DATETIME', 'ATT_FLAG']]
        elif 'SWaT' in csv_file:
            # SWaT Dataset
            self.labels = data['Normal/Attack'].values
            self.sensor_cols = [col for col in data.columns if col not in ['Unnamed: 0', 'Timestamp', 'Normal/Attack']]
            # Convert string labels to binary format: 0 for Normal, 1 for Attack
            self.labels = [0 if label == 'Normal' else 1 for label in self.labels]
        elif 'HAI' in csv_file:
            # HAI Dataset
            self.labels = data['attack'].values
            self.sensor_cols = [col for col in data.columns if col not in ['Unnamed: 0', 'time', 'attack']]
        # elif 'HAI' in csv_file:
        #     # HAI Dataset (Alternative format if needed)
        #     self.labels = data['Attack'].values
        #     self.sensor_cols = [col for col in data.columns if col not in ['Unnamed: 0', 'timestamp', 'Attack']]
        else:
            raise ValueError("Unknown dataset type. Please provide a BATADAL or SWaT CSV file.")

        # Normalize data using MinMaxScaler
        scaler = MinMaxScaler()
        normalized_data = scaler.fit_transform(data[self.sensor_cols])
        # Save the fitted scaler to a file for later use (e.g., during testing)
        pickle.dump(scaler, open(scaler_path, 'wb'))

        # Convert to PyTorch Tensor
        self.data = torch.tensor(normalized_data, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def get_dataloader(csv_file, batch_size, scaler_path):
    """
    Creates DataLoaders for training and validation.
    Splits the dataset into 75% training and 25% validation.
    """
    dataset = CSVDataset(csv_file, scaler_path)

    # Data splitting
    train_size = int(len(dataset) * 3 / 4)
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    return train_loader, val_loader


def get_anomaly_dataloader(csv_file, batch_size, scaler_path):
    """
    Wrapper to get dataloaders, functionally identical to get_dataloader.
    """
    dataset = CSVDataset(csv_file, scaler_path)

    # Data splitting
    train_size = int(len(dataset) * 3 / 4)
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    return train_loader, val_loader


def load_and_preprocess_data(csv_file_path, scaler_path):
    """
    Loads data from a CSV file, preprocesses it (feature selection, scaling),
    and returns it as a single PyTorch tensor.

    Args:
        csv_file_path (str): Path to the input CSV data file.
        scaler_path (str): Path to the saved MinMaxScaler object from training.

    Returns:
        torch.Tensor: The normalized sensor data tensor.
        list: The list of sensor column names.
    """
    data = pd.read_csv(csv_file_path)

    # 1. Identify dataset type based on filename and select feature columns
    filename = csv_file_path.upper()
    if 'BATADAL' in filename:
        # BATADAL Dataset
        sensor_cols = [col for col in data.columns if col not in ['Unnamed: 0', 'DATETIME', 'ATT_FLAG']]
        target_col = 'ATT_FLAG'
    elif 'SWAT' in filename:
        # SWaT Dataset
        sensor_cols = [col for col in data.columns if col not in ['Unnamed: 0', 'Timestamp', 'Normal/Attack']]
        target_col = 'Normal/Attack'
    elif 'HAI' in filename:
        # HAI Dataset
        sensor_cols = [col for col in data.columns if col not in ['Unnamed: 0', 'time', 'attack']]
        target_col = 'attack'
    else:
        raise ValueError("Unknown dataset type. File must contain BATADAL, SWaT, or HAI.")

    # Extract sensor data
    labels = data[target_col]
    sensor_data = data[sensor_cols].values

    # 2. Load and apply MinMaxScaler for normalization
    try:
        scaler = pickle.load(open(scaler_path, 'rb'))
        normalized_data = scaler.transform(sensor_data)
    except FileNotFoundError:
        print(f"❌ Error: Scaler file not found at {scaler_path}. Cannot normalize data.")
        raise
    except Exception as e:
        print(f"❌ Error during scaler loading/transformation: {e}")
        raise

    # 3. Convert to PyTorch Tensor
    data_tensor = torch.tensor(normalized_data, dtype=torch.float32)
    # print(f"Data shape loaded: {data_tensor.shape}")

    return data_tensor, labels