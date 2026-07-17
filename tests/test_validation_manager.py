import torch
import torch.nn as nn
import pytest
from engine.device_manager import DeviceManager
from engine.validation_manager import ValidationManager
from utils.config_loader import ConfigLoader

class MockModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 10)
        
    def forward(self, x, targets=None):
        out = self.linear(x)
        loss = out.mean() if targets is not None else None
        return out, loss

@pytest.fixture
def device_manager():
    # Use auto config to ensure tests run safely anywhere
    config = {"device": "auto", "dtype": "auto"}
    return DeviceManager(config)

def test_model_mode_restoration(device_manager):
    model = MockModel().to(device_manager.device)
    model.train()
    
    vm = ValidationManager(device_manager)
    
    def mock_fetcher():
        return torch.randn(2, 10), torch.randn(2, 10)
        
    vm.evaluate(model, eval_iters=1, batch_fetcher=mock_fetcher)
    
    assert model.training is True, "ValidationManager did not restore train mode!"

def test_no_gradient_allocation(device_manager):
    model = MockModel().to(device_manager.device)
    model.train()
    
    vm = ValidationManager(device_manager)
    
    def mock_fetcher():
        return torch.randn(2, 10), torch.randn(2, 10)
        
    vm.evaluate(model, eval_iters=1, batch_fetcher=mock_fetcher)
    
    # Check that no gradients were accumulated
    assert model.linear.weight.grad is None, "Gradients were allocated during evaluation!"

def test_deterministic_evaluation(device_manager):
    model = MockModel().to(device_manager.device)
    model.eval()
    
    vm = ValidationManager(device_manager)
    
    # Fixed seed for data fetcher
    def mock_fetcher():
        torch.manual_seed(42)
        return torch.randn(2, 10), torch.randn(2, 10)
        
    metrics1 = vm.evaluate(model, eval_iters=2, batch_fetcher=mock_fetcher)
    metrics2 = vm.evaluate(model, eval_iters=2, batch_fetcher=mock_fetcher)
    
    assert metrics1["val_loss"] == metrics2["val_loss"], "Evaluation is not deterministic!"

def test_autocast_compatibility():
    # Force auto to let the DeviceManager resolve correctly
    config = {"device": "auto", "dtype": "auto"}
    dm = DeviceManager(config)
    
    model = MockModel().to(dm.device)
    vm = ValidationManager(dm)
    
    def mock_fetcher():
        return torch.randn(2, 10), torch.randn(2, 10)
        
    # As long as this doesn't crash, the autocast context logic is structurally sound
    vm.evaluate(model, eval_iters=1, batch_fetcher=mock_fetcher)
