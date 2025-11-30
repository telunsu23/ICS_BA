import json
import pickle
import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, average_precision_score)
from utils.util import calculate_asr
from model.detector.Detector import Detector
from utils.load_config import get_config


def test_detector(dataset, window=2):
    """
    Tests the Detector on the test dataset and calculates metrics.
    """
    config = get_config(dataset)
    detector = Detector(nI=config.nI, nH=config.nH, cf=2.5, activation='tanh')

    # Load trained model weights
    detector.load_state_dict(torch.load(config.benign_model_path, weights_only=True))

    # Load the pre-calculated threshold
    with open(config.benign_threshold_path, "r") as f:
        threshold = json.load(f)["threshold"]

    # Load test data
    data = pd.read_csv(config.test_data_path)

    sensor_cols = [col for col in data.columns if col not in config.remove_list]

    # Extract Labels
    labels = data[config.target_col]

    # Load the scaler fitted on training data
    scaler = pickle.load(open(config.scaler_path, "rb"))
    normalized_data = scaler.transform(data[sensor_cols])

    # Convert to PyTorch Tensor
    test_data = torch.tensor(normalized_data, dtype=torch.float32)

    # Model Testing Phase
    detector.eval()
    with torch.no_grad():
        detection, sample_errors, feature_errors, preds = detector.detect(test_data, threshold, window)

    # 1. Prepare data for AUC calculation
    # Move anomaly scores to CPU and convert to NumPy
    anomaly_scores = sample_errors.cpu().numpy()

    # 2. Calculate AUC metrics
    # ROC-AUC: Requires true labels and anomaly scores
    roc_auc = roc_auc_score(labels, anomaly_scores)
    # PR-AUC: Calculated using average_precision_score
    pr_auc = average_precision_score(labels, anomaly_scores)

    # Calculate standard classification metrics
    accuracy = accuracy_score(labels, detection)
    precision = precision_score(labels, detection, zero_division=0)
    recall = recall_score(labels, detection, zero_division=0)
    f1 = f1_score(labels, detection, zero_division=0)

    # Calculate Attack Success Rate (ASR) - Custom utility function
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
    # Set Pandas options to display all columns and prevent truncation
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)

    metrics = test_detector('BATADAL',30)

    results = pd.DataFrame(
        index=['test dataset'],
        columns=['acc', 'pre', 'rec', 'f1', 'ASR', "roc_auc", "pr_auc"]
    )
    results.loc['test dataset'] = [metrics[col] for col in results.columns]

    print("Test Metrics:\n")
    print(results)