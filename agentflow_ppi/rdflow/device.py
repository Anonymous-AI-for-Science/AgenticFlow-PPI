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
    per_process_memory_fraction: float = 0.80
    synchronize_on_barrier: bool = False


class DeviceManager:
    """Select and manage the best available PyTorch device.

    The manager exposes a single torch.device so that all agents in the R&D
    automation stack share the same accelerator context. This avoids repeated
    device detection and makes batching straightforward.
    """

    def __init__(self, config: MPSExecutionConfig | None = None) -> None:
        self.config = config or MPSExecutionConfig()
        self.device = self._resolve_device()
        self._configure_runtime()

    def _resolve_device(self) -> torch.device:
        if self.config.prefer_mps and torch.backends.mps.is_available():
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
        if self.device.type == "mps" and self.config.synchronize_on_barrier and hasattr(torch, "mps"):
            torch.mps.synchronize()

    def empty_cache(self) -> None:
        if self.device.type == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()

    def report(self) -> Dict[str, int | float | bool | str | None]:
        """Return capability and memory telemetry for logging.

        The function is safe on CPU-only machines. On MPS devices it also
        exposes allocator telemetry, which is useful when tuning asynchronous
        batching on memory-constrained laptops.
        """
        report: Dict[str, int | float | bool | str | None] = {
            "device": self.device.type,
            "mps_built": bool(torch.backends.mps.is_built()),
            "mps_available": bool(torch.backends.mps.is_available()),
        }
        if self.device.type == "mps" and hasattr(torch, "mps"):
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


