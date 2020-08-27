import numpy as np


def mse(logits, targets):
    """[计算MSE]

    Args:
        logits ([array,matrix]): [网络模型输出]
        target ([array,int]): [目标]

    Returns:
        [type]: [description]
    """
    mse_value = np.mean((logits - targets)**2)
    return mse_value


def rmse(logits, targets):
    """[计算RMSE]

    Args:
        logits ([array,matrix]): [网络模型输出]
        target ([array,int]): [目标]

    Returns:
        [type]: [description]
    """
    rmse_value = np.sqrt(np.mean((logits - targets)**2))
    return rmse_value

def mae(logits, targets):
    """[计算RMSE]

    Args:
        logits ([array,matrix]): [网络模型输出]
        target ([array,int]): [目标]

    Returns:
        [type]: [description]
    """
    mae_value = np.mean(np.abs(logits - targets))
    return mae_value