"""
Hierarchical YAML Configuration Loader.
Enforces strict validation, Enum mapping, and deterministic fingerprinting.
"""

import os
import yaml
import hashlib
import json
import logging
from typing import Dict, Any

from config.enums import FFNType, AttentionType, PositionType, NormType

logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

class ConfigLoader:
    """
    Loads and validates hierarchical YAML configurations for Axiom.
    """
    
    ENUM_MAPPINGS = {
        "ffn_type": FFNType,
        "attention_type": AttentionType,
        "position_type": PositionType,
        "norm_type": NormType
    }
    
    @classmethod
    def load(cls, filepath: str) -> Dict[str, Any]:
        """Loads a YAML config, resolving 'base' inheritance if present."""
        config = cls._load_yaml_recursive(filepath)
        cls.validate(config)
        config = cls._parse_enums(config)
        config["fingerprint"] = cls.generate_fingerprint(config)
        cls.print_summary(config)
        return config

    @classmethod
    def _load_yaml_recursive(cls, filepath: str) -> Dict[str, Any]:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Configuration file not found: {filepath}")
            
        with open(filepath, 'r') as f:
            config = yaml.safe_load(f)
            
        if not config:
            return {}
            
        if "base" in config:
            base_path = config.pop("base")
            base_abs_path = os.path.join(os.path.dirname(filepath), base_path)
            base_config = cls._load_yaml_recursive(base_abs_path)
            
            merged = base_config.copy()
            merged.update(config)
            return merged
            
        return config

    @classmethod
    def validate(cls, config: Dict[str, Any]) -> None:
        """Runs all modular validation checks."""
        cls._validate_presence(config)
        cls._validate_model(config)
        cls._validate_training(config)
        cls._validate_warnings(config)

    @classmethod
    def _validate_presence(cls, config: Dict[str, Any]) -> None:
        required_fields = [
            "d_model", "n_heads", "n_layers", "context_length", "vocab_size",
            "ffn_type", "attention_type", "position_type", "norm_type",
            "batch_size", "max_iters", "grad_accum_steps", "learning_rate",
            "weight_decay", "beta1", "beta2", "grad_clip", "warmup_iters",
            "lr_decay_iters", "min_lr", "device", "dtype", "val_interval",
            "checkpoint_interval", "eval_iters", "checkpoint"
        ]
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Strict Validation Failed: Missing required field '{field}'.")

    @classmethod
    def _validate_model(cls, config: Dict[str, Any]) -> None:
        if config["d_model"] <= 0 or config["n_layers"] <= 0 or config["n_heads"] <= 0:
            raise ValueError("Strict Validation Failed: Architectural dimensions must be positive integers.")
        if config["context_length"] <= 0 or config["vocab_size"] <= 0:
            raise ValueError("Strict Validation Failed: context_length and vocab_size must be positive integers.")
        if config["d_model"] % config["n_heads"] != 0:
            raise ValueError(
                f"Strict Validation Failed: d_model ({config['d_model']}) must be "
                f"perfectly divisible by n_heads ({config['n_heads']})."
            )
        if config["dropout"] < 0.0 or config["dropout"] >= 1.0:
            raise ValueError(f"Strict Validation Failed: dropout ({config['dropout']}) must be in [0.0, 1.0).")

    @classmethod
    def _validate_training(cls, config: Dict[str, Any]) -> None:
        if config["batch_size"] <= 0:
            raise ValueError("Strict Validation Failed: batch_size must be > 0.")
        if config["learning_rate"] <= 0:
            raise ValueError("Strict Validation Failed: learning_rate must be > 0.")
        if config["min_lr"] > config["learning_rate"]:
            raise ValueError("Strict Validation Failed: min_lr cannot be strictly greater than max learning_rate.")
        if config["warmup_iters"] > config["max_iters"]:
            raise ValueError("Strict Validation Failed: warmup_iters cannot exceed max_iters.")
        if config["eval_iters"] <= 0:
            raise ValueError("Strict Validation Failed: eval_iters must be > 0.")

    @classmethod
    def _validate_warnings(cls, config: Dict[str, Any]) -> None:
        if config["warmup_iters"] < 10:
            logger.warning("warmup_iters is less than 10. This may cause early instability.")
        if config["dropout"] == 0.0:
            logger.warning("dropout is exactly 0.0. Model may overfit if trained for long.")
        if config["batch_size"] == 1:
            logger.warning("batch_size is 1. Gradient variance will be extremely high.")
        if config["learning_rate"] > 0.1:
            logger.warning(f"learning_rate ({config['learning_rate']}) is unusually high.")
        if config["checkpoint_interval"] > config["max_iters"]:
            logger.warning("checkpoint_interval is greater than max_iters. No intermediate checkpoints will be saved.")

    @classmethod
    def _parse_enums(cls, config: Dict[str, Any]) -> Dict[str, Any]:
        parsed_config = config.copy()
        for key, enum_class in cls.ENUM_MAPPINGS.items():
            if key in parsed_config:
                val = parsed_config[key]
                if isinstance(val, str):
                    try:
                        parsed_config[key] = enum_class(val.lower())
                    except ValueError:
                        raise ValueError(f"Invalid {key}: '{val}'. Allowed: {[e.value for e in enum_class]}")
        return parsed_config

    @classmethod
    def generate_fingerprint(cls, config: Dict[str, Any]) -> str:
        clean_config = {}
        for k, v in sorted(config.items()):
            if k == "fingerprint":
                continue
            # For nested dictionaries like 'checkpoint'
            if isinstance(v, dict):
                clean_config[k] = json.dumps(v, sort_keys=True)
            else:
                clean_config[k] = v.value if hasattr(v, 'value') else v
            
        config_str = json.dumps(clean_config, sort_keys=True)
        return hashlib.sha256(config_str.encode('utf-8')).hexdigest()

    @classmethod
    def verify_checkpoint_compatibility(cls, current_config: Dict[str, Any], checkpoint_config: Dict[str, Any]) -> None:
        critical_fields = [
            "d_model", "n_layers", "n_heads", "vocab_size", "context_length",
            "ffn_type", "attention_type", "position_type", "norm_type"
        ]
        
        for field in critical_fields:
            current_val = current_config.get(field)
            ckpt_val = checkpoint_config.get(field)
            
            current_val_str = current_val.value if hasattr(current_val, 'value') else current_val
            ckpt_val_str = ckpt_val.value if hasattr(ckpt_val, 'value') else ckpt_val
            
            if current_val_str != ckpt_val_str:
                raise RuntimeError(
                    f"Checkpoint Incompatibility Detected!\n"
                    f"Field '{field}' differs.\n"
                    f"Current Config: {current_val_str}\n"
                    f"Checkpoint Config: {ckpt_val_str}\n"
                    f"Resuming this checkpoint would cause catastrophic mathematical corruption. Stop."
                )

    @classmethod
    def print_summary(cls, config: Dict[str, Any]) -> None:
        print("\n====================================")
        print("AXIOM CONFIGURATION")
        print("====================================")
        print("\nModel:")
        print(f"  Context Length: {config['context_length']}")
        print(f"  Heads: {config['n_heads']}")
        print(f"  Layers: {config['n_layers']}")
        print(f"  d_model: {config['d_model']}")
        print("\nTraining:")
        print(f"  Batch Size: {config['batch_size']}")
        print(f"  Learning Rate: {config['learning_rate']}")
        print(f"  Weight Decay: {config['weight_decay']}")
        print(f"  Max Iterations: {config['max_iters']}")
        print("\nSystem:")
        print(f"  Device: {config['device']}")
        print(f"  Precision: {config['dtype']}")
        print("\nCheckpoint:")
        print(f"  Keep Last: {config['checkpoint']['keep_last']}")
        print(f"  Validation Interval: {config['val_interval']}")
        print("====================================\n")

