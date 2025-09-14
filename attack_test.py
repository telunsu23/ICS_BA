import json
import pickle
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from utils.util import calculate_asr
from model.detector.Detector import Detector
from utils.load_config import get_config


def test_detector(dataset, epoch, use_backdoor, window=2):
    config = get_config(dataset)
    detector = Detector(nI=config.nI, nH=config.nH, cf=2.5, activation='tanh')
    model_path = config.backdoor_model_path + f'/backdoor_detector_epoch{epoch}.pth'
    detector.load_state_dict(torch.load(model_path, weights_only=True))
    with open(config.backdoor_threshold_path + f'/th_epoch_{epoch}.json', "r") as f:
        threshold = json.load(f)["threshold"]
    # 加载测试数据
    test_data_path = config.backdoor_data_path if use_backdoor else config.test_data_path
    data = pd.read_csv(test_data_path)
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
    all_results = []

    for epoch in range(20, 71):  # epoch 从 30 到 70 (包含 70)
        metrics = test_detector('HAI', epoch, True, 1)
        # 创建一个包含当前 epoch 信息的字典
        result_data = {
            'epoch': epoch,
            'acc': metrics.get('acc', None),  # 使用 .get() 避免 key 不存在时出错
            'pre': metrics.get('pre', None),
            'rec': metrics.get('rec', None),
            'f1': metrics.get('f1', None),
            'ASR': metrics.get('ASR', None)
        }
        all_results.append(result_data)
    # 将所有结果合并到一个 DataFrame
    results_df = pd.DataFrame(all_results)
    # 按 'rec' 值从大到小排序
    sorted_results_df = results_df.sort_values(by='rec', ascending=True)
    print("按 'rec' 值从大到小排序的结果：")
    print(sorted_results_df)

    # metrics = test_detector('HAI', 29, True,1)
    # results = pd.DataFrame(
    #     index=['test dataset'],
    #     columns=['acc', 'pre', 'rec', 'f1', 'ASR']
    # )
    # results.loc['test dataset'] = [metrics[col] for col in results.columns]
    # print("Test Metrics:\n")
    # print(results)
