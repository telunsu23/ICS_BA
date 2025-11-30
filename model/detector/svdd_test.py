import json
import pickle
import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, average_precision_score)
from utils.util import calculate_asr
from Detector import DeepSVDDDetector
from utils.load_config import get_config


def test_deepsvdd_detector(dataset, window=4):
    """
    Tests the performance of the Deep SVDD anomaly detector on the test set.
    """
    config = get_config(dataset)

    # -------------------------------------------------------
    # 1. Initialize and Load Deep SVDD Model
    # -------------------------------------------------------
    # Use DeepSVDDDetector instead of the standard AE Detector
    detector = DeepSVDDDetector(nI=config.nI, nH=config.nH, cf=2.5, activation='tanh')

    # Load model weights and the fixed center 'c'
    # weights_only=True is a security best practice for loading pickle files in newer PyTorch versions
    detector.load_state_dict(torch.load(config.deepsvdd_model_path, weights_only=True))

    # Load the pre-calculated threshold (determined during training/validation)
    with open(config.deepsvdd_threshold_path, "r") as f:
        threshold = json.load(f)["threshold"]

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    detector.to(device)

    # -------------------------------------------------------
    # 2. Data Loading and Preprocessing
    # -------------------------------------------------------
    # Load test dataset
    data = pd.read_csv(config.test_data_path)

    # Select sensor columns (excluding time and label columns)
    sensor_cols = [col for col in data.columns if col not in config.remove_list]
    labels = data[config.target_col]

    # Load the scaler fitted on the training data to ensure consistent normalization
    scaler = pickle.load(open(config.scaler_path, "rb"))
    normalized_data = scaler.transform(data[sensor_cols])

    # Convert to PyTorch Tensor and move to device
    test_data = torch.tensor(normalized_data, dtype=torch.float32).to(device)

    # -------------------------------------------------------
    # 3. Model Inference Phase
    # -------------------------------------------------------
    detector.eval()
    with torch.no_grad():
        # Call the detect method.
        # Note: For Deep SVDD, 'sample_errors' represents the smoothed distance to center c (anomaly score).
        # This replaces the reconstruction error used in Autoencoder-based detectors.
        detection, sample_errors, _, _ = detector.detect(test_data, theta=threshold, window=window)

    # -------------------------------------------------------
    # 4. Metrics Calculation
    # -------------------------------------------------------

    # Transfer anomaly scores (smoothed distances) to CPU and convert to NumPy
    anomaly_scores = sample_errors.cpu().numpy()

    # Transfer binary detection results (0: Normal, 1: Attack) to CPU
    detection_cpu = detection.cpu().numpy()

    # Calculate AUC Metrics (Threshold independent)
    # ROC AUC: Area Under the Receiver Operating Characteristic Curve
    roc_auc = roc_auc_score(labels, anomaly_scores)
    # PR AUC: Area Under the Precision-Recall Curve (often more informative for imbalanced datasets)
    pr_auc = average_precision_score(labels, anomaly_scores)

    # Calculate Classification Metrics (Threshold dependent)
    accuracy = accuracy_score(labels, detection_cpu)
    precision = precision_score(labels, detection_cpu, zero_division=0)
    recall = recall_score(labels, detection_cpu, zero_division=0)
    f1 = f1_score(labels, detection_cpu, zero_division=0)

    # Calculate Attack Success Rate (ASR)
    # ASR measures the ratio of missed attacks (False Negatives) to total attacks
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
    # Enable Deep SVDD Model Testing
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)

    # Run test on the specified dataset
    metrics = test_deepsvdd_detector('SWaT')

    # Create a DataFrame to display results
    results = pd.DataFrame(
        index=['Deep SVDD Test'],
        columns=['acc', 'pre', 'rec', 'f1', 'ASR', "roc_auc", "pr_auc"]
    )
    results.loc['Deep SVDD Test'] = [metrics[col] for col in results.columns]

    print("Deep SVDD Test Metrics:\n")
    print(results)