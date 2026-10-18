"""Check a local Ollama server, model availability, and basic system resources."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

try:
    import psutil
except ImportError:  # pragma: no cover - exercised on installations without psutil
    psutil = None


DEFAULT_HOST = "http://127.0.0.1:11434"
DEFAULT_CHAT_MODEL = "qwen2.5-coder:7b"
DEFAULT_EMBEDDING_MODEL = "nomic-embed-text:latest"
DEFAULT_PROMPT = "Reply with exactly: Ollama is working."
RAM_WARNING_BYTES = 8 * 1024**3


class OllamaError(RuntimeError):
    """An expected failure while communicating with Ollama."""


@dataclass(frozen=True)
class OllamaClient:
    """Small HTTP client for the Ollama REST API."""

    host: str
    timeout: float = 30.0

    def _request(self, endpoint: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        url = f"{self.host.rstrip('/')}{endpoint}"
        request = Request(url, method="POST" if payload is not None else "GET")
        if payload is not None:
            request.add_header("Content-Type", "application/json")
            body = json.dumps(payload).encode("utf-8")
        else:
            body = None

        try:
            with urlopen(request, data=body, timeout=self.timeout) as response:
                raw_response = response.read()
        except HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace").strip()
            raise OllamaError(
                f"Ollama returned HTTP {error.code} for {endpoint}: "
                f"{detail or error.reason}"
            ) from error
        except URLError as error:
            raise OllamaError(
                f"Could not connect to Ollama at {self.host}. "
                "Start Ollama and try again (for example, run `ollama serve`). "
                f"Details: {error.reason}"
            ) from error
        except TimeoutError as error:
            raise OllamaError(
                f"Ollama did not respond within {self.timeout:g} seconds at {url}."
            ) from error

        try:
            result = json.loads(raw_response)
        except json.JSONDecodeError as error:
            raise OllamaError(f"Ollama returned invalid JSON for {endpoint}.") from error
        if not isinstance(result, dict):
            raise OllamaError(f"Ollama returned an unexpected response for {endpoint}.")
        return result

    def list_models(self) -> list[str]:
        result = self._request("/api/tags")
        models = result.get("models")
        if not isinstance(models, list):
            raise OllamaError("Ollama's model list response did not contain a `models` list.")

        names: list[str] = []
        for model in models:
            if isinstance(model, dict) and isinstance(model.get("name"), str):
                names.append(model["name"])
        return names

    def chat(self, model: str, prompt: str) -> str:
        result = self._request(
            "/api/chat",
            {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
            },
        )
        message = result.get("message")
        if not isinstance(message, dict) or not isinstance(message.get("content"), str):
            raise OllamaError("Ollama's chat response did not contain message content.")
        return message["content"]

    def embed(self, model: str, prompt: str) -> list[float]:
        result = self._request(
            "/api/embed",
            {"model": model, "input": prompt},
        )
        embeddings = result.get("embeddings")
        if not isinstance(embeddings, list) or len(embeddings) != 1:
            raise OllamaError("Ollama's embedding response did not contain one embedding.")
        embedding = embeddings[0]
        if not isinstance(embedding, list) or not all(
            isinstance(value, (int, float)) for value in embedding
        ):
            raise OllamaError("Ollama returned an invalid embedding vector.")
        return [float(value) for value in embedding]


def _format_bytes(value: int) -> str:
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    amount = float(value)
    for unit in units:
        if amount < 1024 or unit == units[-1]:
            return f"{amount:.2f} {unit}"
        amount /= 1024
    raise AssertionError("unreachable")


def _print_system_resources() -> int | None:
    """Print only resource values that can be measured on this machine."""
    total_ram: int | None = None
    free_ram: int | None = None
    if psutil is not None:
        memory = psutil.virtual_memory()
        total_ram = memory.total
        free_ram = memory.available
        print(f"RAM: total {_format_bytes(total_ram)}, free {_format_bytes(free_ram)}")
    else:
        print("RAM: not detected (install psutil to measure it)")

    disk = shutil.disk_usage(Path.cwd())
    print(f"Disk ({Path.cwd().anchor or Path.cwd()}): free {_format_bytes(disk.free)}")
    print(f"GPU/VRAM: {_detect_nvidia_smi()}")

    return total_ram


def _detect_nvidia_smi() -> str:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.free",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return "not detected"
    if result.returncode != 0 or not result.stdout.strip():
        return "not detected"
    return result.stdout.strip().replace("\n", "; ")


def _check_model(models: list[str], model: str) -> None:
    if model not in models:
        raise OllamaError(
            f"Model {model!r} is not downloaded. Pull it with: ollama pull {model}"
        )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Check a local Ollama server, chat model, embeddings, and resources."
    )
    parser.add_argument(
        "--host", default=DEFAULT_HOST, help=f"Ollama URL (default: {DEFAULT_HOST})"
    )
    parser.add_argument(
        "--chat-model",
        default=DEFAULT_CHAT_MODEL,
        help=f"Chat model name (default: {DEFAULT_CHAT_MODEL})",
    )
    parser.add_argument(
        "--embedding-model",
        default=DEFAULT_EMBEDDING_MODEL,
        help=f"Embedding model name (default: {DEFAULT_EMBEDDING_MODEL})",
    )
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Prompt sent to the chat model")
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Per-request timeout in seconds (default: 30)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.timeout <= 0:
        print("ERROR: --timeout must be greater than zero.", file=sys.stderr)
        return 2

    client = OllamaClient(args.host, args.timeout)
    print(f"Ollama server: {args.host}")
    total_ram = _print_system_resources()
    if (
        total_ram is not None
        and total_ram < RAM_WARNING_BYTES
        and args.chat_model == DEFAULT_CHAT_MODEL
    ):
        print(
            "WARNING: measured total RAM is below approximately 8 GiB for the 7B model. "
            "Consider trying qwen2.5-coder:3b."
        )

    try:
        started = time.perf_counter()
        models = client.list_models()
        list_latency = time.perf_counter() - started
        print(f"Installed models ({list_latency:.3f}s):")
        if models:
            for model in models:
                print(f"  - {model}")
        else:
            print("  (none)")

        _check_model(models, args.chat_model)
        _check_model(models, args.embedding_model)

        started = time.perf_counter()
        response = client.chat(args.chat_model, args.prompt)
        chat_latency = time.perf_counter() - started
        print(f"Chat latency ({args.chat_model}): {chat_latency:.3f}s")
        print(f"Chat response: {response}")

        started = time.perf_counter()
        embedding = client.embed(args.embedding_model, args.prompt)
        embed_latency = time.perf_counter() - started
        print(
            f"Embedding latency ({args.embedding_model}): {embed_latency:.3f}s "
            f"(dimensions: {len(embedding)})"
        )
    except OllamaError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
