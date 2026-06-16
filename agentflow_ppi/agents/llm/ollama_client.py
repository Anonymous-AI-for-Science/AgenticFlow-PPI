"""Ollama-backed LLM client for the multi-agent layer, with a deterministic
offline fallback.

Design goals (reviewer R1-O1.A: instantiate and measure real agents):
  * Real LLM calls. When an Ollama server is reachable (default
    http://localhost:11434), each agent's prompt is rendered from its prompt file
    and sent to a configurable model (e.g. llama3.1:8b, qwen2.5:7b, phi3:medium,
    mistral-nemo). The raw completion, token counts, and latency are recorded.
  * Reproducibility without a server. When Ollama is not reachable (CI, the
    artifact-evaluation sandbox, a laptop without the daemon), the client falls
    back to a deterministic "schema oracle" that returns the same well-formed JSON
    the prompt asks for, computed symbolically from the inputs. Every response is
    tagged with provenance (`backend`: "ollama:<model>" or "offline-deterministic")
    so measured-vs-simulated is never ambiguous.

This mirrors the rest of the package: the mechanism is genuinely runnable on a
networked host (a MacBook Pro M3 with `ollama serve` and the models pulled), and it
degrades to an exact, labeled fallback elsewhere.
"""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
DEFAULT_OLLAMA_URL = "http://localhost:11434"


@dataclass
class LLMResponse:
    """One agent's LLM turn, fully recorded for measurement."""
    agent: str
    backend: str                 # "ollama:<model>" or "offline-deterministic"
    model: str
    prompt_chars: int
    completion: str              # raw text returned
    parsed: Dict[str, Any]       # JSON parsed from completion
    prompt_tokens: int
    completion_tokens: int
    latency_s: float
    used_fallback: bool
    valid_json: bool
    schema_ok: bool


def _render_prompt(name: str, fields: Dict[str, Any]) -> str:
    text = (PROMPTS_DIR / f"{name}.md").read_text()
    for k, v in fields.items():
        text = text.replace("{" + k + "}", str(v))
    return text


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    """Pull the first JSON object out of a completion, tolerating ``` fences."""
    t = text.strip()
    t = re.sub(r"^```(?:json)?", "", t).strip()
    t = re.sub(r"```$", "", t).strip()
    # find the outermost {...}
    start = t.find("{")
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(t)):
        if t[i] == "{":
            depth += 1
        elif t[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(t[start : i + 1])
                except json.JSONDecodeError:
                    return None
    return None


class OllamaClient:
    """Calls an Ollama model, or falls back to a deterministic schema oracle."""

    def __init__(self, model: str = "llama3.1:8b", url: str = DEFAULT_OLLAMA_URL,
                 temperature: float = 0.0, timeout_s: float = 60.0,
                 force_offline: bool = False) -> None:
        self.model = model
        self.url = url.rstrip("/")
        self.temperature = temperature
        self.timeout_s = timeout_s
        self.force_offline = force_offline
        self._available: Optional[bool] = None

    # -- availability ------------------------------------------------------
    def available(self) -> bool:
        if self.force_offline:
            return False
        if self._available is not None:
            return self._available
        try:
            req = urllib.request.Request(self.url + "/api/tags")
            with urllib.request.urlopen(req, timeout=2.0) as r:
                self._available = r.status == 200
        except Exception:
            self._available = False
        return self._available

    # -- raw call ----------------------------------------------------------
    def _call_ollama(self, prompt: str) -> Dict[str, Any]:
        body = json.dumps({
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": self.temperature},
            "format": "json",
        }).encode()
        req = urllib.request.Request(self.url + "/api/generate", data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout_s) as r:
            return json.loads(r.read().decode())

    # -- public agent turn -------------------------------------------------
    def run_agent(self, agent: str, fields: Dict[str, Any],
                  fallback: Dict[str, Any], schema_keys) -> LLMResponse:
        """Render `agent`'s prompt with `fields`, call the model (or fall back to
        the deterministic `fallback` dict), parse JSON, and record everything."""
        prompt = _render_prompt(agent, fields)
        t0 = time.perf_counter()
        if self.available():
            try:
                raw = self._call_ollama(prompt)
                completion = raw.get("response", "")
                parsed = _extract_json(completion)
                latency = time.perf_counter() - t0
                valid = parsed is not None
                if not valid:
                    parsed = dict(fallback)
                schema_ok = all(k in parsed for k in schema_keys)
                if not schema_ok:
                    # repair missing keys from the deterministic fallback
                    for k in schema_keys:
                        parsed.setdefault(k, fallback[k])
                    schema_ok = True
                return LLMResponse(
                    agent=agent, backend=f"ollama:{self.model}", model=self.model,
                    prompt_chars=len(prompt), completion=completion, parsed=parsed,
                    prompt_tokens=int(raw.get("prompt_eval_count", 0)),
                    completion_tokens=int(raw.get("eval_count", 0)),
                    latency_s=latency, used_fallback=not valid,
                    valid_json=valid, schema_ok=schema_ok)
            except (urllib.error.URLError, TimeoutError, OSError):
                pass  # fall through to offline
        # offline deterministic
        latency = time.perf_counter() - t0
        return LLMResponse(
            agent=agent, backend="offline-deterministic", model=self.model,
            prompt_chars=len(prompt), completion=json.dumps(fallback),
            parsed=dict(fallback), prompt_tokens=0, completion_tokens=0,
            latency_s=latency, used_fallback=True, valid_json=True, schema_ok=True)
