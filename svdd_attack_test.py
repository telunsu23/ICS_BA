import json
import pickle
import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, average_precision_score)
from model.detector.Detector import DeepSVDDDetector
from utils.util import calculate_asr
from utils.load_config import get_config


def test_deepsvdd_backdoor_detector(dataset, epoch, use_backdoor, window=10):
    config = get_config(dataset)
    # 1. Model Initialization: Use DeepSVDDDetector
    detector = DeepSVDDDetector(nI=config.nI, nH=config.nH, cf=2.5, activation='tanh')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    detector.to(device)

    # 2. Model and Threshold Loading: Use Deep SVDD backdoor model path
    model_path = config.deepsvdd_backdoor_model_path + f'/deepsvdd_backdoor_detector_epoch{epoch}.pth'
    threshold_path = config.deepsvdd_backdoor_threshold_path + f'/deepsvdd_th_epoch_{epoch}.json'

    # Dynamically load threshold
    with open(threshold_path, "r") as f:
        threshold = json.load(f)["threshold"]

    detector.load_state_dict(torch.load(model_path, weights_only=True))

    # 3. Data Loading
    # If use_backdoor is True, load the test set containing triggers and labels
    test_data_path = config.backdoor_data_path if use_backdoor else config.test_data_path
    data = pd.read_csv(test_data_path)
    sensor_cols = [col for col in data.columns if col not in config.remove_list]
    labels = data[config.target_col]
    scaler = pickle.load(open(config.scaler_path, "rb"))
    normalized_data = scaler.transform(data[sensor_cols])

    test_data = torch.tensor(normalized_data, dtype=torch.float32).to(device)

    # 4. Model Testing Phase
    detector.eval()
    with torch.no_grad():
        # The sample_errors returned by Deep SVDD's detect method are the smoothed distances (anomaly scores)
        detection, sample_errors, _, _ = detector.detect(test_data, theta=threshold, window=window)

    # 5. Metrics Calculation
    # Transfer the model's output anomaly scores (sample_errors, i.e., smoothed distances) to CPU and convert to NumPy array
    anomaly_scores = sample_errors.cpu().numpy()
    detection_cpu = detection.cpu().numpy()

    # ROC-AUC: Requires ground truth labels (labels) and anomaly scores (anomaly_scores)
    roc_auc = roc_auc_score(labels, anomaly_scores)
    # PR-AUC: Calculated using average_precision_score
    pr_auc = average_precision_score(labels, anomaly_scores)

    accuracy = accuracy_score(labels, detection_cpu)
    precision = precision_score(labels, detection_cpu, zero_division=0)
    recall = recall_score(labels, detection_cpu, zero_division=0)
    f1 = f1_score(labels, detection_cpu, zero_division=0)

    # Calculate Attack Success Rate
    attack_success_rate = calculate_asr(labels, detection_cpu)

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

    all_results = []

    # Ensure range() covers the epoch range of your Deep SVDD training
    for epoch in range(1, 21):
        # Test normal detection capability (do not use backdoor test data, check model utility)
        metrics = test_deepsvdd_backdoor_detector('SWaT', epoch, False, 1)
        # metrics = test_deepsvdd_backdoor_detector('SWaT', epoch, True, 1)

        result_data = {
            'epoch': epoch,
            'acc': metrics.get('acc', None),
            'pre': metrics.get('pre', None),
            'rec': metrics.get('rec', None),
            'f1': metrics.get('f1', None),
            'ASR': metrics.get('ASR', None),
            'roc_auc': metrics.get('roc_auc', None),
            'pr_auc': metrics.get('pr_auc', None)
        }
        all_results.append(result_data)

    results_df = pd.DataFrame(all_results)
    sorted_results_df = results_df.sort_values(by='rec', ascending=True)
    print("Deep SVDD Backdoor Model Test Results (Based on normal test set, sorted by Recall):")
    print(sorted_results_df)

    # Example: Directly test the backdoor attack success rate for a specific epoch
    # metrics_backdoor = test_deepsvdd_backdoor_detector('HAI', 20, True, 1)
    # results = pd.DataFrame(
    #     index=['Deep SVDD Backdoor Test'],
    #     columns=['acc', 'pre', 'rec', 'f1', 'ASR', "roc_auc", "pr_auc"]
    # )
    # results.loc['Deep SVDD Backdoor Test'] = [metrics_backdoor[col] for col in results.columns]
    # print("\nDeep SVDD Backdoor Attack Metrics (ASR):\n")
    # print(results)