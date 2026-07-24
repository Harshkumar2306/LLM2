"""
Reproducibility Engine for Axiom.
Captures exact environmental, hardware, and Git state to ensure 
100% reproducibility of all experiments.
"""

import os
import sys
import platform
import subprocess
import torch
import numpy as np
import random
from typing import Dict, Any

class ReproducibilityEngine:
    """
    Collects metadata and enforces random seed determinism.
    """
    
    @staticmethod
    def get_git_state() -> Dict[str, str]:
        """Safely captures the Git repository state."""
        state = {
            "commit": "unknown",
            "branch": "unknown",
            "dirty": "unknown"
        }
        try:
            # Check if we are inside a git repo
            subprocess.check_output(['git', 'rev-parse'], stderr=subprocess.DEVNULL)
            
            commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], stderr=subprocess.DEVNULL).decode('utf-8').strip()
            branch = subprocess.check_output(['git', 'rev-parse', '--abbrev-ref', 'HEAD'], stderr=subprocess.DEVNULL).decode('utf-8').strip()
            status = subprocess.check_output(['git', 'status', '--porcelain'], stderr=subprocess.DEVNULL).decode('utf-8').strip()
            
            state["commit"] = commit
            state["branch"] = branch
            state["dirty"] = "true" if len(status) > 0 else "false"
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass
            
        return state

    @staticmethod
    def get_system_state() -> Dict[str, Any]:
        """Captures Python, PyTorch, and Hardware versions."""
        state = {
            "os": platform.system() + " " + platform.release(),
            "python": sys.version.split('\n')[0],
            "pytorch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else "N/A",
            "mps_available": hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(),
            "gpus": []
        }
        
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                state["gpus"].append({
                    "id": i,
                    "name": torch.cuda.get_device_name(i)
                })
                
        return state

    @staticmethod
    def set_seed(seed: int = 1337) -> None:
        """
        Enforces random seed determinism across all random number generators.
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            # Optimize for determinism on cuDNN
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    @classmethod
    def capture_full_metadata(cls) -> Dict[str, Any]:
        """
        Aggregates all reproducibility metadata into a single dictionary.
        """
        return {
            "git": cls.get_git_state(),
            "system": cls.get_system_state()
        }

    @staticmethod
    def capture_rng_states() -> Dict[str, Any]:
        """
        Captures the exact mathematical state of all Random Number Generators.
        Essential for recovering a training loop exactly where it left off.
        """
        states = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state()
        }
        if torch.cuda.is_available():
            states["torch_cuda"] = torch.cuda.get_rng_state_all()
        return states

    @staticmethod
    def restore_rng_states(states: Dict[str, Any]) -> None:
        """
        Restores the exact mathematical state of all Random Number Generators.
        """
        if "python" in states:
            random.setstate(states["python"])
        if "numpy" in states:
            # np.random.set_state expects a tuple, but JSON serialization converts it to a list
            # We must convert it back to a tuple. The state is usually (str, np.ndarray, int, int, float)
            state_list = states["numpy"]
            if isinstance(state_list, list):
                state_list[1] = np.array(state_list[1], dtype=np.uint32)
                states["numpy"] = tuple(state_list)
            np.random.set_state(states["numpy"])
        if "torch_cpu" in states:
            # Convert JSON back to byte tensor if necessary
            state = states["torch_cpu"]
            if isinstance(state, list):
                state = torch.ByteTensor(state)
            torch.set_rng_state(state)
        if "torch_cuda" in states and torch.cuda.is_available():
            cuda_states = states["torch_cuda"]
            if isinstance(cuda_states, list):
                cuda_states = [torch.ByteTensor(s) if isinstance(s, list) else s for s in cuda_states]
            try:
                torch.cuda.set_rng_state_all(cuda_states)
            except Exception as e:
                print(f"Warning: Failed to restore CUDA RNG state. Device count mismatch? {e}")
