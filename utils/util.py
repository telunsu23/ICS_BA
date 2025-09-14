import numpy as np

def calculate_asr(y_true, y_pred):
    """
    计算后门攻击成功率（ASR），假设所有样本都是后门样本。

    Args:
        y_true (list or np.array): 所有样本的真实类别标签。
        y_pred (list or np.array): 模型对所有样本的预测结果。

    Returns:
        float: 后门攻击成功率的百分比。
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # 1. 统计真实标签为 1 的样本总数
    total_label_1 = np.sum(y_true == 1)
    # 2. 检查真实标签为 1 的样本数量，避免除以零
    if total_label_1 == 0:
        return 0.0
    # 3. 统计成功攻击的样本数
    # 攻击成功：真实值为 1 (异常)，但预测值为 0 (正常)
    successful_attacks = np.sum((y_true == 1) & (y_pred == 0))
    # 4. 计算 ASR
    asr = (successful_attacks / total_label_1) * 100

    return asr