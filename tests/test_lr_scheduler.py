import pytest
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

def test_reduce_lr_on_plateau():
    """Test that ReduceLROnPlateau scheduler works correctly with metrics."""
    # Create a simple model
    model = nn.Linear(10, 1)
    optimizer = Adam(model.parameters(), lr=0.1)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)
    
    initial_lr = optimizer.param_groups[0]['lr']
    
    # Simulate training with improving metrics
    scheduler.step(2.0)  # loss = 2.0
    assert optimizer.param_groups[0]['lr'] == initial_lr
    
    scheduler.step(1.5)  # loss = 1.5
    assert optimizer.param_groups[0]['lr'] == initial_lr
    
    # Simulate training with non-improving metrics
    scheduler.step(1.8)  # loss = 1.8 (worse)
    assert optimizer.param_groups[0]['lr'] == initial_lr
    
    scheduler.step(1.9)  # loss = 1.9 (worse)
    assert optimizer.param_groups[0]['lr'] == initial_lr
    
    scheduler.step(2.0)  # loss = 2.0 (worse) - should trigger LR reduction
    assert optimizer.param_groups[0]['lr'] == initial_lr * 0.1 
