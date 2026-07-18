import os
import json
import uuid
import logging
import datetime
import yaml
import glob
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class ExperimentManager:
    """
    Central research coordinator for an Axiom experiment.
    Manages the directory structure, structured logging (JSONL), metadata, and summary reports.
    """
    def __init__(self, base_dir: str, experiment_name: str, config: Dict[str, Any], is_master: bool = True):
        self.experiment_name = experiment_name
        self.config = config
        self.experiment_dir = os.path.join(base_dir, experiment_name)
        self.is_master = is_master
        
        self.dirs = {
            "checkpoints": os.path.join(self.experiment_dir, "checkpoints"),
            "logs": os.path.join(self.experiment_dir, "logs"),
            "config": os.path.join(self.experiment_dir, "config"),
            "reports": os.path.join(self.experiment_dir, "reports"),
            "samples": os.path.join(self.experiment_dir, "samples"),
            "benchmark": os.path.join(self.experiment_dir, "benchmark"),
        }
        
        self.metrics_path = os.path.join(self.experiment_dir, "metrics.jsonl")
        self.metadata_path = os.path.join(self.experiment_dir, "metadata.json")
        self.summary_path = os.path.join(self.experiment_dir, "summary.md")
        
        if self.is_master:
            self._init_directories()
            self._init_metadata()
            self._save_config()

    def _init_directories(self) -> None:
        os.makedirs(self.experiment_dir, exist_ok=True)
        for d in self.dirs.values():
            os.makedirs(d, exist_ok=True)

    def _init_metadata(self) -> None:
        metadata = {}
        if os.path.exists(self.metadata_path):
            with open(self.metadata_path, 'r') as f:
                metadata = json.load(f)
                
        if "creation_date" not in metadata:
            metadata["creation_date"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
            
        if "experiment_id" not in metadata:
            metadata["experiment_id"] = str(uuid.uuid4())
            
        metadata.update({
            "experiment_name": self.experiment_name,
            "config_fingerprint": self.config.get("fingerprint", "unknown"),
            "parameter_count": metadata.get("parameter_count", "Unknown (Set by Trainer)"),
            "dataset_name": metadata.get("dataset_name", "Unknown"),
            "architecture_name": metadata.get("architecture_name", "Axiom GPT"),
            "training_stage": metadata.get("training_stage", "Phase 1 - Screening"),
            "creation_platform": os.name
        })
        
        self.update_metadata(metadata)

    def _save_config(self) -> None:
        config_path = os.path.join(self.dirs["config"], "run_config.yaml")
        clean_config = {}
        for k, v in self.config.items():
            clean_config[k] = v.value if hasattr(v, 'value') else v
            
        with open(config_path, 'w') as f:
            yaml.safe_dump(clean_config, f, default_flow_style=False, sort_keys=True)

    # =========================================================================
    # PUBLIC API
    # =========================================================================

    def get_experiment_dir(self) -> str:
        return self.experiment_dir

    def update_metadata(self, updates: Dict[str, Any]) -> None:
        current = {}
        if os.path.exists(self.metadata_path):
            with open(self.metadata_path, 'r') as f:
                current = json.load(f)
                
        current.update(updates)
        
        tmp_path = self.metadata_path + ".tmp"
        with open(tmp_path, 'w') as f:
            json.dump(current, f, indent=4)
        os.replace(tmp_path, self.metadata_path)

    def log_metrics(self, raw_metrics: Dict[str, Any]) -> None:
        """
        Appends structured metrics to metrics.jsonl enforcing a fixed schema.
        Missing values are set to None.
        """
        record = {
            "iteration": raw_metrics.get("iteration"),
            "epoch": raw_metrics.get("epoch", None),
            "train_loss": raw_metrics.get("train_loss", None),
            "val_loss": raw_metrics.get("val_loss", None),
            "learning_rate": raw_metrics.get("learning_rate", None),
            "gradient_norm": raw_metrics.get("gradient_norm", None),
            "tokens_per_second": raw_metrics.get("tokens_per_second", None),
            "samples_per_second": raw_metrics.get("samples_per_second", None),
            "elapsed_time": raw_metrics.get("elapsed_time", None),
            "timestamp": raw_metrics.get("timestamp", datetime.datetime.now(datetime.timezone.utc).isoformat()),
            "best_val_loss": raw_metrics.get("best_val_loss", None)
        }
            
        with open(self.metrics_path, 'a') as f:
            f.write(json.dumps(record) + "\n")

    def save_sample(self, iteration: int, sample_text: str) -> None:
        filename = f"sample_step_{iteration}.txt"
        filepath = os.path.join(self.dirs["samples"], filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(sample_text)

    def generate_summary(self) -> None:
        """Generates a comprehensive markdown summary report."""
        meta = {}
        if os.path.exists(self.metadata_path):
            with open(self.metadata_path, 'r') as f:
                meta = json.load(f)
                
        # Count checkpoints
        num_checkpoints = len(glob.glob(os.path.join(self.dirs["checkpoints"], "*.pt")))
        
        # Read final metrics
        final_train_loss = "N/A"
        best_val_loss = "N/A"
        total_time = "N/A"
        
        if os.path.exists(self.metrics_path):
            with open(self.metrics_path, 'r') as f:
                lines = f.readlines()
                if lines:
                    last_record = json.loads(lines[-1])
                    final_train_loss = last_record.get("train_loss", "N/A")
                    best_val_loss = last_record.get("best_val_loss", "N/A")
                    total_time = last_record.get("elapsed_time", "N/A")
                    final_iteration = last_record.get("iteration", "N/A")
                
        md_content = f"# Experiment Summary: {self.experiment_name}\n\n"
        md_content += f"**Experiment ID:** `{meta.get('experiment_id', 'N/A')}`\n"
        md_content += f"**Creation Date:** {meta.get('creation_date', 'N/A')}\n"
        md_content += f"**Platform:** {meta.get('creation_platform', 'N/A')}\n\n"
        
        md_content += "## Model Overview\n"
        md_content += f"- **Architecture:** {meta.get('architecture_name', 'N/A')}\n"
        md_content += f"- **Parameters:** {meta.get('parameter_count', 'N/A')}\n"
        md_content += f"- **Dataset:** {meta.get('dataset_name', 'N/A')}\n"
        ds_size = meta.get('dataset_size_tokens', 'N/A')
        ds_size_str = f"{ds_size:,}" if isinstance(ds_size, int) else ds_size
        md_content += f"- **Dataset Size:** {ds_size_str} tokens\n\n"
        
        md_content += "## Configuration Summary\n"
        md_content += f"**Fingerprint:** `{meta.get('config_fingerprint', 'N/A')}`\n"
        
        md_content += "\n## Training Results\n"
        md_content += f"- **Final Iteration Reached:** {final_iteration}\n"
        md_content += f"- **Best Validation Loss:** {best_val_loss}\n"
        md_content += f"- **Final Training Loss:** {final_train_loss}\n"
        md_content += f"- **Total Training Time (s):** {total_time}\n"
        md_content += f"- **Total Checkpoints:** {num_checkpoints}\n"
        
        best_ckpt = meta.get("best_checkpoint")
        if best_ckpt:
            md_content += f"\n## Best Checkpoint Path\n`{best_ckpt}`\n"
            
        with open(self.summary_path, 'w', encoding='utf-8') as f:
            f.write(md_content)
            
        print("\n" + "="*50)
        print("EXPERIMENT COMPLETED")
        print("="*50)
        print(md_content)
        print("="*50 + "\n")
