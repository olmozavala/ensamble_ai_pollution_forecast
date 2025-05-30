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
    mask_data = 1 - target[1]
    return F.mse_loss(output*mask_data, pollution_data*mask_data)

def asymmetric_weighted_mse_loss(output, target):
    """
    Weighted MSE loss that penalizes under-predictions more heavily than over-predictions.
    
    Args:
        output: Model predictions
        target: Ground truth values (tuple of pollution_data and mask)
    
    Returns:
        Weighted MSE loss with higher penalty for under-predictions
    """
    pollution_data = target[0]
    mask_data = 1 - target[1]
    
    # Calculate difference between prediction and target
    diff = output - pollution_data
    
    # Create weights - use 2.0 for under-predictions, 1.0 for over-predictions
    weights = torch.where(diff < 0, torch.tensor(2.0).to(output.device), torch.tensor(1.0).to(output.device))
    
    # Apply mask and weights
    squared_diff = weights * (diff ** 2) * mask_data
    
    return torch.mean(squared_diff)


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
