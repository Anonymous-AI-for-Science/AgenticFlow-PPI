"""Smoke tests for the asynchronous research automation flow."""

from __future__ import annotations

import asyncio

from agentflow_ppi.rdflow import RDFlowCoordinator


def test_rdflow_cpu_fallback() -> None:
    coordinator = RDFlowCoordinator()
    result = asyncio.run(coordinator.execute("Draft theory and implementation tasks."))
    assert "device" in result
    assert "summaries" in result
    assert set(result["summaries"]).issuperset({"planner"})


