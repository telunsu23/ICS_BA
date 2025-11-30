import json
import pickle
import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, average_precision_score)
from utils.util import calculate_asr
from model.detector.Detector import Detector
from utils.load_config import get_config


def test_detector(dataset, epoch, use_backdoor, window=30):
    config = get_config(dataset)
    detector = Detector(nI=config.nI, nH=config.nH, cf=2.5, activation='tanh')
    model_path = config.backdoor_model_path + f'/backdoor_detector_epoch{epoch}.pth'
    detector.load_state_dict(torch.load(model_path, weights_only=True))
    with open(config.backdoor_threshold_path + f'/th_epoch_{epoch}.json', "r") as f:
        threshold = json.load(f)["threshold"]
    # Load test data
    test_data_path = config.backdoor_data_path if use_backdoor else config.test_data_path
    data = pd.read_csv(test_data_path)
    sensor_cols = [col for col in data.columns if col not in config.remove_list]
    # Labels
    labels = data[config.target_col]
    # Load the scaler of the training set
    scaler = pickle.load(open(config.scaler_path, "rb"))
    normalized_data = scaler.transform(data[sensor_cols])
    # Convert to PyTorch tensor
    test_data = torch.tensor(normalized_data, dtype=torch.float32)
    # Model testing phase
    detector.eval()
    with torch.no_grad():
        detection, sample_errors, feature_errors, preds = detector.detect(test_data, theta=threshold, window=window)

    # Transfer anomaly scores (sample_errors) output by the model to CPU and convert to NumPy array
    anomaly_scores = sample_errors.cpu().numpy()
    # Calculate AUC metrics
    roc_auc = roc_auc_score(labels, anomaly_scores)
    pr_auc = average_precision_score(labels, anomaly_scores)

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
        "ASR": attack_success_rate,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc
    }

    return metrics

if __name__ == '__main__':
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)

    metrics = test_detector('BATADAL', 60, True,30)
    results = pd.DataFrame(
        index=['test dataset'],
        columns=['acc', 'pre', 'rec', 'f1', 'ASR', "roc_auc", "pr_auc"]
    )
    results.loc['test dataset'] = [metrics[col] for col in results.columns]
    print("Test Metrics:\n")
    print(results)
