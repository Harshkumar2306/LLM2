from tests.helpers import get_test_config
import os
import torch
import shutil
import pytest
from trainer.checkpoint_manager import CheckpointManager, ConfigurationMismatchError

@pytest.fixture
def experiment_dir(tmp_path):
    exp_dir = tmp_path / "test_exp"
    return str(exp_dir)

@pytest.fixture
def dummy_config():
    return {
        "fingerprint": "mock_hash_123",
        "checkpoint": {"keep_last": 2}
    }

def create_dummy_state():
    return {
        "model_state": {"weight": torch.randn(10, 10)},
        "optimizer_state": {"step": 1},
        "scheduler_state": {"lr": 0.01},
        "grad_scaler_state": None,
        "rng_states": {"python": None},
        "metrics": {"loss": 0.5}
    }

def test_resume(experiment_dir, dummy_config):
    # Train and save step 1 and step 2
    cm = CheckpointManager(experiment_dir, dummy_config)
    
    state1 = create_dummy_state()
    cm.save(**state1, iteration=1, is_best=False)
    
    state2 = create_dummy_state()
    state2["optimizer_state"]["step"] = 2
    cm.save(**state2, iteration=2, is_best=False)
    
    # Simulate interruption by creating a new manager instance
    cm2 = CheckpointManager(experiment_dir, dummy_config)
    loaded = cm2.load(mode="latest")
    
    assert loaded["training_progress"]["iteration"] == 2
    assert loaded["optimizer_state"]["step"] == 2

def test_corruption_recovery(experiment_dir, dummy_config):
    cm = CheckpointManager(experiment_dir, dummy_config)
    
    state1 = create_dummy_state()
    cm.save(**state1, iteration=1, is_best=False)
    
    state2 = create_dummy_state()
    cm.save(**state2, iteration=2, is_best=False)
    
    # Corrupt step 2
    latest_pt = os.path.join(experiment_dir, "checkpoints", "step_2.pt")
    with open(latest_pt, 'wb') as f:
        f.write(b"corrupted binary data")
        
    # Recovery check
    cm2 = CheckpointManager(experiment_dir, dummy_config)
    loaded = cm2.load(mode="latest")
    
    # Should automatically fallback to step 1
    assert loaded["training_progress"]["iteration"] == 1

def test_config_mismatch(experiment_dir, dummy_config):
    cm = CheckpointManager(experiment_dir, dummy_config)
    state = create_dummy_state()
    cm.save(**state, iteration=1, is_best=False)
    
    # Try to load with a different fingerprint
    different_config = dummy_config.copy()
    different_config["fingerprint"] = "a_totally_different_hash"
    
    cm2 = CheckpointManager(experiment_dir, different_config)
    
    with pytest.raises(ConfigurationMismatchError, match="Configuration Mismatch Detected"):
        cm2.load(mode="latest")

def test_atomic_write_interrupt(experiment_dir, dummy_config):
    cm = CheckpointManager(experiment_dir, dummy_config)
    state = create_dummy_state()
    cm.save(**state, iteration=1, is_best=False)
    
    # Simulate an interruption during step 2 save (torch.save completes, but process dies before os.replace)
    tmp_filepath = os.path.join(experiment_dir, "checkpoints", "step_2.pt.tmp")
    with open(tmp_filepath, 'wb') as f:
        f.write(b"partial data")
        
    # The system should load step 1 successfully, ignoring the dangling .tmp file
    cm2 = CheckpointManager(experiment_dir, dummy_config)
    loaded = cm2.load(mode="latest")
    assert loaded["training_progress"]["iteration"] == 1
