import os
import json
import torch
import torch.nn as nn
import pytest
import glob

from engine.training_state import TrainingState
from engine.trainer import Trainer
from engine.device_manager import DeviceManager
from engine.checkpoint_manager import CheckpointManager
from engine.experiment_manager import ExperimentManager
from engine.validation_manager import ValidationManager

class MockModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 10)
        
    def forward(self, x, targets=None):
        out = self.linear(x)
        loss = out.mean() if targets is not None else None
        return out, loss

def setup_managers(tmp_path, exp_name, resume_mode="none"):
    config = {
        "device": "cpu",
        "dtype": "float32",
        "fingerprint": "xyz123",
        "checkpoint": {"keep_last": 2},
        "training": {
            "max_iters": 5,
            "eval_interval": 2,
            "eval_iters": 1,
            "save_interval": 2,
            "log_interval": 1,
            "grad_clip": 1.0,
            "resume_mode": resume_mode
        }
    }
    
    dm = DeviceManager(config)
    em = ExperimentManager(str(tmp_path), exp_name, config)
    cm = CheckpointManager(em.get_experiment_dir(), config)
    vm = ValidationManager(dm)
    
    return dm, em, cm, vm, config

def create_trainer(dm, em, cm, vm, config, model=None, optimizer=None):
    if model is None:
        model = MockModel().to(dm.device)
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
        
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=None,
        device_manager=dm,
        checkpoint_manager=cm,
        experiment_manager=em,
        validation_manager=vm,
        config=config
    )
    return trainer

def dummy_fetcher():
    torch.manual_seed(42)
    return torch.randn(2, 10), torch.randn(2, 10)

def test_fresh_training(tmp_path):
    dm, em, cm, vm, config = setup_managers(tmp_path, "fresh")
    trainer = create_trainer(dm, em, cm, vm, config)
    
    trainer.train(dummy_fetcher, dummy_fetcher)
    
    # 5 iters total. Checkpoints at 2, 4, and 5 (end of train). Check keep_last=2.
    assert trainer.state.iteration == 5
    
    # Check ExperimentManager artifacts
    exp_dir = em.get_experiment_dir()
    assert os.path.exists(os.path.join(exp_dir, "summary.md"))
    
    with open(os.path.join(exp_dir, "metrics.jsonl"), 'r') as f:
        lines = f.readlines()
        assert len(lines) >= 5 # 5 log + 2 val
        
    checkpoints = glob.glob(os.path.join(exp_dir, "checkpoints", "step_*.pt"))
    assert len(checkpoints) <= 2 # keep_last = 2

def test_resume_training(tmp_path):
    # Train partially
    dm, em, cm, vm, config = setup_managers(tmp_path, "resume")
    config["training"]["max_iters"] = 2
    trainer = create_trainer(dm, em, cm, vm, config)
    trainer.train(dummy_fetcher, dummy_fetcher)
    
    assert trainer.state.iteration == 2
    
    # Resume and finish
    dm2, em2, cm2, vm2, config2 = setup_managers(tmp_path, "resume", resume_mode="latest")
    config2["training"]["max_iters"] = 4
    trainer2 = create_trainer(dm2, em2, cm2, vm2, config2)
    
    assert trainer2.state.iteration == 2
    
    trainer2.train(dummy_fetcher, dummy_fetcher)
    assert trainer2.state.iteration == 4

def test_validation_integration(tmp_path):
    dm, em, cm, vm, config = setup_managers(tmp_path, "val_test")
    config["training"]["max_iters"] = 3
    config["training"]["eval_interval"] = 2
    trainer = create_trainer(dm, em, cm, vm, config)
    trainer.train(dummy_fetcher, dummy_fetcher)
    
    # Check that best val loss was updated
    assert trainer.state.best_val_loss != float('inf')
    assert os.path.exists(os.path.join(em.get_experiment_dir(), "checkpoints", "best.pt"))

def test_fault_tolerance(tmp_path):
    dm, em, cm, vm, config = setup_managers(tmp_path, "fault")
    config["training"]["max_iters"] = 2
    trainer = create_trainer(dm, em, cm, vm, config)
    trainer.train(dummy_fetcher, dummy_fetcher)
    
    # Simulate a crash right after a checkpoint by creating a dangling .tmp file
    tmp_ckpt = os.path.join(em.get_experiment_dir(), "checkpoints", "step_4.pt.tmp")
    with open(tmp_ckpt, 'wb') as f:
        f.write(b"corrupted")
        
    # Resume should load step 2 and ignore the tmp file
    dm2, em2, cm2, vm2, config2 = setup_managers(tmp_path, "fault", resume_mode="latest")
    config2["training"]["max_iters"] = 4
    trainer2 = create_trainer(dm2, em2, cm2, vm2, config2)
    
    assert trainer2.state.iteration == 2

def test_deterministic_reproducibility(tmp_path):
    dm1, em1, cm1, vm1, config1 = setup_managers(tmp_path, "det1")
    trainer1 = create_trainer(dm1, em1, cm1, vm1, config1)
    trainer1.train(dummy_fetcher, dummy_fetcher)
    final_loss_1 = trainer1.current_loss
    
    dm2, em2, cm2, vm2, config2 = setup_managers(tmp_path, "det2")
    trainer2 = create_trainer(dm2, em2, cm2, vm2, config2)
    trainer2.train(dummy_fetcher, dummy_fetcher)
    final_loss_2 = trainer2.current_loss
    
    assert final_loss_1 == final_loss_2
