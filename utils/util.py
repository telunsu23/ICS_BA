import random
import numpy as np
import torch


def setup_seed(seed=1107):
    """Sets the random seed for reproducibility across NumPy, PyTorch, and Python's random module."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


def calculate_asr(y_true, y_pred):
    """
    Calculates the Backdoor Attack Success Rate (ASR), assuming all samples are backdoor samples.

    Args:
        y_true (list or np.array): True class labels for all samples.
        y_pred (list or np.array): Model predictions for all samples.

    Returns:
        float: Percentage of successful backdoor attacks.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # 1. Count the total number of samples with true label 1
    total_label_1 = np.sum(y_true == 1)

    # 2. Check the count of samples with true label 1 to avoid division by zero
    if total_label_1 == 0:
        return 0.0

    # 3. Count the number of successfully attacked samples
    # Attack Success: True value is 1 (Anomaly), but Predicted value is 0 (Normal)
    successful_attacks = np.sum((y_true == 1) & (y_pred == 0))

    # 4. Calculate ASR
    asr = (successful_attacks / total_label_1) * 100

    return asr