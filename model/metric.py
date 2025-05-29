import torch
import numpy as np
import torch.nn.functional as F

def accuracy(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)


def top_k_acc(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)


def mae(output, target):
    """
    Mean Absolute Error metric.
    Args:
        output: Model predictions
        target: Ground truth values
    Returns:
        float: MAE value
    """
    with torch.no_grad():
        return torch.mean(torch.abs(output - target)).item()


def rmse_metric(output, target):
    """
    Root Mean Square Error metric.
    Args:
        output: Model predictions
        target: Ground truth values
    Returns:
        float: RMSE value
    """
    with torch.no_grad():
        return torch.sqrt(torch.mean((output - target) ** 2)).item()


def r2_score(output, target):
    """
    R² (R-squared) metric.
    Args:
        output: Model predictions
        target: Ground truth values
    Returns:
        float: R² value
    """
    with torch.no_grad():
        target_mean = torch.mean(target)
        ss_tot = torch.sum((target - target_mean) ** 2)
        ss_res = torch.sum((target - output) ** 2)
        r2 = 1 - ss_res / ss_tot
        return r2.item()


def explained_variance(output, target):
    """
    Explained variance metric.
    Args:
        output: Model predictions
        target: Ground truth values
    Returns:
        float: Explained variance value
    """
    with torch.no_grad():
        target_var = torch.var(target)
        diff_var = torch.var(target - output)
        return (1 - diff_var / target_var).item()


def masked_mse_metric(output, target):
    """Masked MSE loss for regression tasks."""
    with torch.no_grad():
        pollution_data = target[0]
        mask_data = target[1]
        return F.mse_loss(output*mask_data, pollution_data*mask_data)


def masked_rmse_metric(output, target):
    """Masked MSE loss for regression tasks."""
    with torch.no_grad():
        pollution_data = target[0]
        mask_data = target[1]
        return torch.sqrt(F.mse_loss(output*mask_data, pollution_data*mask_data))

def asymmetric_weighted_mse_metric(output, target):
    """

    Weighted MSE loss that penalizes under-predictions more heavily than over-predictions.
    
    Args:
        output: Model predictions
        target: Ground truth values (tuple of pollution_data and mask)
    
    Returns:
        Weighted MSE loss with higher penalty for under-predictions
    """
    with torch.no_grad():
        pollution_data = target[0]
        mask_data = 1 - target[1]
        
        # Calculate difference between prediction and target
        diff = output - pollution_data

        # Create weights - use 2.0 for under-predictions, 1.0 for over-predictions
        weights = torch.where(diff < 0, torch.tensor(4.0).to(output.device), torch.tensor(1.0).to(output.device))
        
        # Apply mask and weights
        squared_diff = weights * (diff ** 2) * mask_data
            
        return torch.mean(squared_diff)