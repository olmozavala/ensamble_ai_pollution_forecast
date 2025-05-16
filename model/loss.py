import torch
import torch.nn.functional as F


def nll_loss(output, target):
    return F.nll_loss(output, target)


def mse_loss(output, target):
    """Mean squared error loss for regression tasks."""
    return F.mse_loss(output, target)


def mae_loss(output, target):
    """Mean absolute error loss for regression tasks."""
    return F.l1_loss(output, target)


def masked_mse_loss(output, target):
    """Masked MSE loss for regression tasks."""
    pollution_data = target[0]
    mask_data = target[1]
    return F.mse_loss(output*mask_data, pollution_data*mask_data)


def huber_loss(output, target, delta=1.0):
    """Huber loss - combines benefits of MSE and MAE."""
    return F.huber_loss(output, target, delta=delta)


def weighted_mse_loss(output, target, weights=None):
    """Weighted MSE loss for handling different importance of samples."""
    if weights is None:
        return F.mse_loss(output, target)
    return torch.mean(weights * (output - target) ** 2)


def rmse_loss(output, target):
    """Root mean squared error loss."""
    return torch.sqrt(F.mse_loss(output, target))
