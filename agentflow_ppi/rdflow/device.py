"""Utilities for Apple Silicon and PyTorch MPS execution.

The module is intentionally defensive: the same code base must run on a single
Apple Silicon MacBook with MPS enabled and on CPU-only artifact-evaluation
machines. The helper therefore exposes a *capability-aware* device manager.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import torch


@dataclass(slots=True)
class MPSExecutionConfig:
    """Configuration for device selection and memory management.

    Attributes
    ----------
    prefer_mps:
        If true, prefer the Apple Metal backend whenever it is available.
    prefer_cuda:
        If true, prefer an NVIDIA CUDA device when available (e.g. on Ubuntu
        hosts with a GPU). CUDA is checked before MPS; both fall back to CPU,
        so the same code path runs on Ubuntu (CUDA or CPU), macOS Intel (CPU),
        and Apple Silicon (MPS).
    per_process_memory_fraction:
        Upper bound for the fraction of the recommended MPS working set that the
        current process may reserve. The call is ignored on non-MPS devices.
    synchronize_on_barrier:
        If true, explicitly synchronize at task barriers. This is useful when an
        asynchronous workflow needs deterministic timing or precise memory
        accounting, but it should remain disabled during throughput-oriented
        steady-state execution.
    """

    prefer_mps: bool = True
    prefer_cuda: bool = True
    per_process_memory_fraction: float = 0.80
    synchronize_on_barrier: bool = False


class DeviceManager:
    """Select and manage the best available PyTorch device.

    The manager exposes a single torch.device so that all agents in the R&D
    automation stack share the same accelerator context. This avoids repeated
    device detection and makes batching straightforward. Selection order is
    CUDA (Ubuntu/GPU) then MPS (Apple Silicon) then CPU (macOS Intel and any
    accelerator-free host), so one code path covers all three target platforms.
    """

    def __init__(self, config: MPSExecutionConfig | None = None) -> None:
        self.config = config or MPSExecutionConfig()
        self.device = self._resolve_device()
        self._configure_runtime()

    def _resolve_device(self) -> torch.device:
        if self.config.prefer_cuda and torch.cuda.is_available():
            return torch.device("cuda")
        if (self.config.prefer_mps and hasattr(torch.backends, "mps")
                and torch.backends.mps.is_available()):
            return torch.device("mps")
        return torch.device("cpu")

    def _configure_runtime(self) -> None:
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")
        if self.device.type == "mps" and hasattr(torch, "mps"):
            if hasattr(torch.mps, "set_per_process_memory_fraction"):
                torch.mps.set_per_process_memory_fraction(self.config.per_process_memory_fraction)

    def move_module(self, module: torch.nn.Module) -> torch.nn.Module:
        """Move a module to the managed device."""
        return module.to(self.device)

    def tensor(self, data: Any, *, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Construct a tensor directly on the managed device."""
        return torch.as_tensor(data, dtype=dtype, device=self.device)

    def barrier(self) -> None:
        """Synchronize only when the selected device and configuration require it."""
        if not self.config.synchronize_on_barrier:
            return
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        elif self.device.type == "mps" and hasattr(torch, "mps"):
            torch.mps.synchronize()

    def empty_cache(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        elif self.device.type == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()

    def report(self) -> Dict[str, int | float | bool | str | None]:
        """Return capability and memory telemetry for logging.

        Safe on every target platform: Ubuntu (CUDA or CPU), macOS Intel (CPU,
        possibly without an mps backend in the torch build), and Apple Silicon
        (MPS). Accelerator telemetry is added only for the selected backend.
        """
        _mps = getattr(torch.backends, "mps", None)
        report: Dict[str, int | float | bool | str | None] = {
            "device": self.device.type,
            "cuda_available": bool(torch.cuda.is_available()),
            "mps_built": bool(_mps.is_built()) if _mps is not None else False,
            "mps_available": bool(_mps.is_available()) if _mps is not None else False,
        }
        if self.device.type == "cuda":
            report["cuda_device_name"] = torch.cuda.get_device_name(0)
            report["cuda_mem_allocated"] = int(torch.cuda.memory_allocated())
            report["cuda_mem_reserved"] = int(torch.cuda.memory_reserved())
        elif self.device.type == "mps" and hasattr(torch, "mps"):
            for name in [
                "current_allocated_memory",
                "driver_allocated_memory",
                "recommended_max_memory",
                "device_count",
            ]:
                fn = getattr(torch.mps, name, None)
                if fn is not None:
                    report[name] = int(fn())
        return report


