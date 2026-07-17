import os
import platform
import sys
import torch
import contextlib
from typing import Dict, Any, NamedTuple

class DeviceCapabilities(NamedTuple):
    """Read-only object encapsulating hardware capabilities for Axiom."""
    supports_amp: bool
    supports_fp16: bool
    supports_bf16: bool
    supports_compile: bool
    supports_flash_attention: bool
    supports_pinned_memory: bool
    requires_grad_scaler: bool
    
    # Future compatibility placeholders (DO NOT IMPLEMENT LOGIC FOR THESE YET)
    # supports_ddp: bool
    # supports_fsdp: bool
    # supports_quantization: bool
    # supports_cuda_graphs: bool
    # supports_xformers: bool

class DeviceManager:
    """
    Abstracts hardware selection and capabilities.
    Acts as the single authority for all hardware-related decisions.
    """
    def __init__(self, config: Dict[str, Any]):
        self.device = self._resolve_device(config.get("device", "auto"))
        
        requested_dtype = config.get("dtype", "auto")
        self.dtype_str = self._resolve_dtype(requested_dtype, self.device)
        self._validate_precision(requested_dtype, self.dtype_str, self.device)
        
        self.ptdtype = {
            'float32': torch.float32, 
            'bfloat16': torch.bfloat16, 
            'float16': torch.float16
        }[self.dtype_str]

        self.capabilities = self._build_capabilities()
        
    def _resolve_device(self, requested_device: str) -> str:
        """Determines the actual hardware device available."""
        if requested_device != "auto":
            return requested_device
            
        if torch.cuda.is_available():
            return 'cuda'
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return 'mps'
        return 'cpu'

    def _resolve_dtype(self, requested_dtype: str, device: str) -> str:
        """Determines the safest optimal precision for the hardware if auto."""
        if requested_dtype != "auto":
            return requested_dtype
            
        if device == 'cuda':
            return 'bfloat16' if torch.cuda.is_bf16_supported() else 'float16'
        return 'float32'

    def _validate_precision(self, requested_dtype: str, actual_dtype: str, device: str):
        """Ensures we do not silently downgrade precision if explicitly requested."""
        if requested_dtype == "auto":
            return
            
        if requested_dtype == "bfloat16" and not (device == 'cuda' and torch.cuda.is_bf16_supported()):
            raise RuntimeError(
                f"Precision Error: Explicitly requested bfloat16, but hardware ({device}) "
                f"does not support it. Use 'auto' or 'float16' instead."
            )
            
        if requested_dtype == "float16" and device not in ['cuda', 'mps']:
            raise RuntimeError(
                f"Precision Error: Explicitly requested float16, but hardware ({device}) "
                f"does not safely support it. Use 'auto' or 'float32' instead."
            )

    def _build_capabilities(self) -> DeviceCapabilities:
        is_linux = sys.platform.startswith('linux')
        
        supports_fp16 = self.device in ['cuda', 'mps']
        supports_bf16 = self.device == 'cuda' and torch.cuda.is_bf16_supported()
        
        supports_amp = False
        if self.dtype_str != 'float32' and self.device != 'cpu':
            if self.device == 'cuda':
                supports_amp = True
            elif self.device == 'mps' and hasattr(torch.amp, 'autocast'):
                supports_amp = True

        supports_compile = self.device == 'cuda' and is_linux
        supports_flash_attention = supports_amp and self.device == 'cuda'
        supports_pinned_memory = self.device == 'cuda'
        
        requires_grad_scaler = self.device == 'cuda' and self.dtype_str == 'float16'
        
        return DeviceCapabilities(
            supports_amp=supports_amp,
            supports_fp16=supports_fp16,
            supports_bf16=supports_bf16,
            supports_compile=supports_compile,
            supports_flash_attention=supports_flash_attention,
            supports_pinned_memory=supports_pinned_memory,
            requires_grad_scaler=requires_grad_scaler
        )

    # =========================================================================
    # PUBLIC API
    # =========================================================================

    def get_grad_scaler(self) -> torch.cuda.amp.GradScaler | None:
        """Returns a configured GradScaler if required, else None."""
        if self.capabilities.requires_grad_scaler:
            return torch.cuda.amp.GradScaler(enabled=True)
        return None

    def autocast(self):
        """
        Returns the appropriate autocast context manager for mixed precision.
        Educational Note: Mixed precision saves memory and speeds up MatMuls by casting 
        them to FP16/BF16, while keeping accumulations in FP32.
        """
        if not self.capabilities.supports_amp:
            return contextlib.nullcontext()
            
        if self.device == 'cuda':
            return torch.autocast(device_type='cuda', dtype=self.ptdtype)
            
        if self.device == 'mps' and hasattr(torch.amp, 'autocast'):
            return torch.autocast(device_type='mps', dtype=self.ptdtype)
            
        return contextlib.nullcontext()

    def to_device(self, *tensors: torch.Tensor):
        """Helper to move multiple tensors to the current device."""
        non_blocking = self.capabilities.supports_pinned_memory
        if len(tensors) == 1:
            return tensors[0].to(self.device, non_blocking=non_blocking)
        return tuple(t.to(self.device, non_blocking=non_blocking) for t in tensors)

    def print_status(self):
        """Logs the hardware configuration, capabilities, and system info."""
        
        # Gather system info
        os_info = f"{platform.system()} {platform.release()}"
        py_ver = sys.version.split(' ')[0]
        torch_ver = torch.__version__
        
        gpu_name = "N/A"
        vram_total_gb = "N/A"
        cuda_ver = "N/A"
        
        if self.device == 'cuda':
            gpu_name = torch.cuda.get_device_name(0)
            vram_total_gb = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2)
            cuda_ver = torch.version.cuda
        elif self.device == 'mps':
            gpu_name = "Apple Silicon GPU"
        
        print("\n====================================")
        print("DEVICE INFORMATION")
        print("====================================")
        print(f"OS:                 {os_info}")
        print(f"Python:             {py_ver}")
        print(f"PyTorch:            {torch_ver}")
        if self.device == 'cuda':
            print(f"CUDA Version:       {cuda_ver}")
        print(f"GPU Name:           {gpu_name}")
        if vram_total_gb != "N/A":
            print(f"Total VRAM:         {vram_total_gb} GB")
        print("------------------------------------")
        
        print("HARDWARE CAPABILITIES")
        print("------------------------------------")
        print(f"Device Selected:    {self.device.upper()}")
        print(f"Precision Target:   {self.dtype_str}")
        print(f"AMP Support:        {'YES' if self.capabilities.supports_amp else 'NO'}")
        print(f"BF16 Support:       {'YES' if self.capabilities.supports_bf16 else 'NO'}")
        print(f"FP16 Support:       {'YES' if self.capabilities.supports_fp16 else 'NO'}")
        print(f"GradScaler Req:     {'YES' if self.capabilities.requires_grad_scaler else 'NO'}")
        print(f"Torch Compile:      {'YES' if self.capabilities.supports_compile else 'NO'}")
        print(f"Flash Attention:    {'YES' if self.capabilities.supports_flash_attention else 'NO'}")
        print(f"Pinned Memory:      {'YES' if self.capabilities.supports_pinned_memory else 'NO'}")
        print("====================================\n")
