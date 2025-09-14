import json
import pickle
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from utils.util import calculate_asr
from model.detector.Detector import Detector
from utils.load_config import get_config


def test_detector(dataset, window=2):
    config = get_config(dataset)
    detector = Detector(nI=config.nI, nH=config.nH, cf=2.5, activation='tanh')
    detector.load_state_dict(torch.load(config.benign_model_path, weights_only=True))
    with open(config.benign_threshold_path, "r") as f:
        threshold = json.load(f)["threshold"]
    # 加载测试数据
    data = pd.read_csv(config.test_data_path)
    # data = pd.read_csv('../../dataset/HAI/clean/test1.csv')
    sensor_cols = [col for col in data.columns if col not in config.remove_list]
    # 标签
    labels = data[config.target_col]
    # 加载训练集的 scaler
    scaler = pickle.load(open(config.scaler_path, "rb"))
    normalized_data = scaler.transform(data[sensor_cols])
    # 转换为 PyTorch 张量
    test_data = torch.tensor(normalized_data, dtype=torch.float32)
    # 模型测试阶段
    detector.eval()
    with torch.no_grad():
        detection, sample_errors, feature_errors, preds = detector.detect(test_data, theta=threshold, window=window)

    accuracy = accuracy_score(labels, detection)
    precision = precision_score(labels, detection, zero_division=0)
    recall = recall_score(labels, detection, zero_division=0)
    f1 = f1_score(labels, detection, zero_division=0)

    attack_success_rate = calculate_asr(labels, detection)

    metrics = {
        "acc": accuracy,
        "pre": precision,
        "rec": recall,
        "f1": f1,
        "ASR": attack_success_rate
    }

    return metrics

if __name__ == '__main__':
    metrics = test_detector('HAI')
    results = pd.DataFrame(
        index=['test dataset'],
        columns=['acc', 'pre', 'rec', 'f1', 'ASR']
    )
    results.loc['test dataset'] = [metrics[col] for col in results.columns]

    print("Test Metrics:\n")
    print(results)
