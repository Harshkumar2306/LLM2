import os
import json
import glob
import time
import shutil
import torch
import datetime
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class ConfigurationMismatchError(RuntimeError):
    pass

class VersionMismatchError(RuntimeError):
    pass

class CheckpointManager:
    """
    Versioned storage system for Axiom Checkpoints.
    Handles atomic writes, retention rotation, integrity verification, and metadata indexing.
    """
    
    CHECKPOINT_VERSION = "1.0.0"
    AXIOM_VERSION = "0.1.0"
    
    def __init__(self, experiment_dir: str, config: Dict[str, Any]):
        self.experiment_dir = experiment_dir
        self.checkpoints_dir = os.path.join(experiment_dir, "checkpoints")
        self.metadata_path = os.path.join(experiment_dir, "metadata.json")
        self.config = config
        self.keep_last = config.get("checkpoint", {}).get("keep_last", 3)
        
        os.makedirs(self.checkpoints_dir, exist_ok=True)
        
    def _read_metadata(self) -> Dict[str, Any]:
        if not os.path.exists(self.metadata_path):
            return {"history": []}
        with open(self.metadata_path, 'r') as f:
            return json.load(f)
            
    def _write_metadata(self, metadata: Dict[str, Any]):
        tmp_path = self.metadata_path + ".tmp"
        with open(tmp_path, 'w') as f:
            json.dump(metadata, f, indent=4)
        os.replace(tmp_path, self.metadata_path)

    def save(
        self, 
        model_state: Dict[str, torch.Tensor],
        optimizer_state: Dict[str, Any],
        scheduler_state: Dict[str, Any],
        grad_scaler_state: Optional[Dict[str, Any]],
        rng_states: Dict[str, Any],
        iteration: int,
        metrics: Dict[str, Any],
        is_best: bool = False
    ) -> None:
        """Saves a checkpoint atomically and manages rotation."""
        
        # 1. Construct Versioned Container
        checkpoint = {
            "checkpoint_version": self.CHECKPOINT_VERSION,
            "axiom_version": self.AXIOM_VERSION,
            "experiment_name": os.path.basename(os.path.normpath(self.experiment_dir)),
            "creation_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "config_fingerprint": self.config.get("fingerprint", "unknown"),
            "model_state": model_state,
            "optimizer_state": optimizer_state,
            "scheduler_state": scheduler_state,
            "grad_scaler_state": grad_scaler_state,
            "rng_states": rng_states,
            "training_progress": {"iteration": iteration},
            "metrics": metrics,
            
            # Reserved Extension Points (DO NOT IMPLEMENT LOGIC FOR THESE YET)
            "ema_state": None,
            "profiler_state": None,
            "tokenizer_state": None,
            "distributed_state": None,
            
            "metadata": {} # Extensible namespace
        }
        
        filename = f"step_{iteration}.pt"
        filepath = os.path.join(self.checkpoints_dir, filename)
        tmp_filepath = filepath + ".tmp"
        
        # 2. Atomic Write
        torch.save(checkpoint, tmp_filepath)
        os.replace(tmp_filepath, filepath)
        
        # 3. Update external metadata
        meta = self._read_metadata()
        meta["latest_checkpoint"] = filepath
        if "history" not in meta:
            meta["history"] = []
        if filepath not in meta["history"]:
            meta["history"].append(filepath)
            
        if is_best:
            best_path = os.path.join(self.checkpoints_dir, "best.pt")
            best_tmp = best_path + ".tmp"
            shutil.copy2(filepath, best_tmp)
            os.replace(best_tmp, best_path)
            meta["best_checkpoint"] = best_path
            
        self._write_metadata(meta)
        self._enforce_rotation()

    def _enforce_rotation(self):
        """Keeps only the 'keep_last' most recent step checkpoints, preserving 'best.pt'."""
        pattern = os.path.join(self.checkpoints_dir, "step_*.pt")
        files = glob.glob(pattern)
        
        def get_iter(fpath):
            basename = os.path.basename(fpath)
            try:
                return int(basename.split('_')[1].split('.')[0])
            except (IndexError, ValueError):
                return -1
                
        files = [f for f in files if get_iter(f) != -1]
        files.sort(key=get_iter)
        
        if len(files) > self.keep_last:
            files_to_delete = files[:-self.keep_last]
            for f in files_to_delete:
                try:
                    os.remove(f)
                except OSError:
                    pass
            
            meta = self._read_metadata()
            meta["history"] = files[-self.keep_last:]
            self._write_metadata(meta)

    def load(self, mode: str = "latest", step: Optional[int] = None) -> Dict[str, Any]:
        """Unified API for loading a checkpoint."""
        meta = self._read_metadata()
        
        if step is not None:
            target_path = os.path.join(self.checkpoints_dir, f"step_{step}.pt")
            return self._verify_and_load(target_path)
            
        if mode == "best":
            target_path = meta.get("best_checkpoint")
            if not target_path or not os.path.exists(target_path):
                raise FileNotFoundError("Best checkpoint not found.")
            return self._verify_and_load(target_path)
            
        if mode == "latest":
            history = meta.get("history", [])
            for path in reversed(history):
                try:
                    return self._verify_and_load(path)
                except (ConfigurationMismatchError, VersionMismatchError) as e:
                    # Fatal mismatch, do not fallback
                    raise e
                except Exception as e:
                    logger.warning(f"[CheckpointManager] Warning: Checkpoint {path} failed verification: {e}. Attempting fallback...")
                    continue
                    
            raise RuntimeError("All checkpoints in history are corrupted or missing.")
            
        raise ValueError(f"Unknown mode: {mode}. Use 'latest' or 'best'.")

    def _verify_and_load(self, filepath: str) -> Dict[str, Any]:
        """Runs the validation pipeline and loads the checkpoint."""
        self._validate_file(filepath)
        
        try:
            checkpoint = torch.load(filepath, map_location='cpu', weights_only=False)
        except Exception as e:
            raise RuntimeError(f"Failed to load checkpoint file (Corrupted?): {e}")
            
        self._validate_structure(checkpoint)
        self._validate_version(checkpoint)
        self._validate_configuration(checkpoint)
        
        return checkpoint

    def _validate_file(self, filepath: str) -> None:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Checkpoint file not found: {filepath}")
        if os.path.getsize(filepath) == 0:
            raise RuntimeError(f"Checkpoint file is empty: {filepath}")

    def _validate_structure(self, checkpoint: Dict[str, Any]) -> None:
        required_keys = [
            "checkpoint_version", "axiom_version", "experiment_name", "creation_timestamp", 
            "config_fingerprint", "model_state", "optimizer_state", "scheduler_state", 
            "grad_scaler_state", "rng_states", "training_progress", "metrics", "metadata"
        ]
        for key in required_keys:
            if key not in checkpoint:
                raise ValueError(f"Checkpoint missing required versioned key: {key}")

    def _validate_version(self, checkpoint: Dict[str, Any]) -> None:
        """Enforces Semantic Versioning compatibility policy."""
        cp_version = checkpoint["checkpoint_version"]
        try:
            cp_major, cp_minor, cp_patch = map(int, cp_version.split('.'))
            curr_major, curr_minor, curr_patch = map(int, self.CHECKPOINT_VERSION.split('.'))
        except ValueError:
            logger.warning(f"Unable to parse version string {cp_version}. Skipping version check.")
            return

        if cp_major != curr_major:
            raise VersionMismatchError(
                f"Major version mismatch: Checkpoint is v{cp_version}, Current is v{self.CHECKPOINT_VERSION}. "
                f"Cannot safely load."
            )
            
        if cp_minor != curr_minor:
            logger.warning(
                f"Minor version mismatch: Checkpoint is v{cp_version}, Current is v{self.CHECKPOINT_VERSION}. "
                f"Loading may have unexpected behavior."
            )
            
        # Patch versions are always allowed.

    def _validate_configuration(self, checkpoint: Dict[str, Any]) -> None:
        current_fingerprint = self.config.get("fingerprint")
        if current_fingerprint and checkpoint["config_fingerprint"] != current_fingerprint:
            raise ConfigurationMismatchError(
                f"Configuration Mismatch Detected!\n"
                f"Checkpoint Fingerprint: {checkpoint['config_fingerprint']}\n"
                f"Current Fingerprint: {current_fingerprint}\n"
                f"Resuming this checkpoint with a different architectural configuration is mathematically unsafe."
            )
