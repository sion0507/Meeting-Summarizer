"""LLM provider abstraction, prompt loading, and JSON response validation.

This module is the single integration point for LLM calls in the MVP pipeline.
Prompts are loaded from external Markdown files, variables are bound before the
call, and every structured response is parsed as JSON and validated into the
requested schema before being returned to callers.

The default provider is ``local``.  For the MVP this means loading the configured
GGUF model file directly in the current Python process through ``llama-cpp-python``.
No HTTP server or network access is required for the default local runtime.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import os
import re
import string
import time
from datetime import UTC, datetime
from urllib import request as urllib_request
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, Protocol, TypeVar, get_args, get_origin, get_type_hints

from meeting_summarizer.config import AppConfig
from meeting_summarizer.utils.logging import get_logger

DEFAULT_PROMPT_DIR = Path(__file__).resolve().parents[3] / "prompts"
DEFAULT_CONTEXT_SIZE = 8192
DEFAULT_GPU_LAYERS = -1
DEFAULT_MAX_TOKENS = 2048
DEFAULT_TEMPERATURE = 0.0
DEFAULT_SERVER_TIMEOUT_SEC = 120

T = TypeVar("T")
LOGGER = get_logger(__name__)


class LLMClientError(RuntimeError):
    """Base error for prompt loading, provider calls, parsing, and validation."""


class PromptLoadError(LLMClientError):
    """Raised when a prompt file cannot be found, loaded, or rendered."""


class LLMProviderError(LLMClientError):
    """Raised when a configured LLM provider cannot complete a request."""


class LLMJSONParseError(LLMClientError):
    """Raised when the LLM response is not parseable as a single JSON value."""


class LLMResponseValidationError(LLMClientError):
    """Raised when parsed JSON does not match the expected schema."""


class LLMProvider(Protocol):
    """Provider contract implemented by local or future remote backends."""

    def generate(self, prompt: str, *, response_format: str = "json") -> str:
        """Return model text for a fully rendered prompt."""


class PromptLoader:
    """Load external Markdown prompts and bind ``$variable`` placeholders."""

    def __init__(self, prompt_dir: str | Path = DEFAULT_PROMPT_DIR) -> None:
        self.prompt_dir = Path(prompt_dir)

    def load(self, prompt_name: str) -> str:
        """Load a prompt by file name or stem from the configured prompt dir."""

        path = self._resolve_prompt_path(prompt_name)
        try:
            text = path.read_text(encoding="utf-8")
        except FileNotFoundError as exc:
            raise PromptLoadError(f"Prompt file not found: {path}") from exc
        except OSError as exc:
            raise PromptLoadError(f"Failed to load prompt file: {path}") from exc

        if not text.strip():
            raise PromptLoadError(f"Prompt file is empty: {path}")
        return text

    def render(self, prompt_name: str, variables: dict[str, Any] | None = None) -> str:
        """Load a prompt and substitute variables with JSON-safe formatting.

        Prompt files use ``string.Template`` syntax (``$segment_json``,
        ``$candidate_group_json``).  Non-string values are serialized as
        pretty-printed JSON so schema-shaped payloads can be embedded safely.
        """

        template_text = self.load(prompt_name)
        rendered_variables = {
            key: _format_prompt_value(value) for key, value in (variables or {}).items()
        }
        try:
            return string.Template(template_text).substitute(rendered_variables)
        except KeyError as exc:
            missing = exc.args[0]
            raise PromptLoadError(
                f"Missing variable ${missing} while rendering prompt {prompt_name!r}."
            ) from exc
        except ValueError as exc:
            raise PromptLoadError(
                f"Invalid template syntax while rendering prompt {prompt_name!r}: {exc}"
            ) from exc

    def _resolve_prompt_path(self, prompt_name: str) -> Path:
        path = Path(prompt_name)
        if path.suffix != ".md":
            path = path.with_suffix(".md")
        if path.is_absolute():
            return path
        return self.prompt_dir / path


class LocalLLMProvider:
    """Offline GGUF provider backed by ``llama-cpp-python``.

    The model is loaded from ``LOCAL_MODEL_PATH`` and inference runs inside the
    local Python process.  This keeps the MVP usable in an offline environment
    without requiring an OpenAI-compatible HTTP server.

    Environment variables:
    - ``LOCAL_MODEL_PATH``: path to the GGUF model file.
    - ``LOCAL_LLM_N_CTX``: llama.cpp context size. Default: 8192.
    - ``LOCAL_LLM_N_GPU_LAYERS``: GPU offload layer count. Default: -1.
    - ``LOCAL_LLM_MAX_TOKENS``: maximum tokens to generate. Default: 2048.
    - ``LOCAL_LLM_TEMPERATURE``: generation temperature. Default: 0.0.
    """

    def __init__(
        self,
        *,
        model_path: str | Path,
        n_ctx: int = DEFAULT_CONTEXT_SIZE,
        n_gpu_layers: int = DEFAULT_GPU_LAYERS,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        llama_factory: Any | None = None,
    ) -> None:
        if not str(model_path).strip():
            raise LLMProviderError("LOCAL_MODEL_PATH must be configured for LLM_PROVIDER=local.")
        self.model_path = Path(model_path)
        if not self.model_path.is_file():
            raise LLMProviderError(f"Local GGUF model file not found: {self.model_path}")
        if n_ctx < 1:
            raise LLMProviderError(f"LOCAL_LLM_N_CTX must be >= 1, got {n_ctx}.")
        if max_tokens < 1:
            raise LLMProviderError(
                f"LOCAL_LLM_MAX_TOKENS must be >= 1, got {max_tokens}."
            )

        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.last_usage: dict[str, int] = {}
        factory = llama_factory or _load_llama_class()
        self._llm = factory(
            model_path=str(self.model_path),
            n_ctx=self.n_ctx,
            n_gpu_layers=self.n_gpu_layers,
            verbose=False,
        )

    @classmethod
    def from_config(cls, config: AppConfig) -> "LocalLLMProvider":
        return cls(
            model_path=config.local_model_path,
            n_ctx=_read_int_env("LOCAL_LLM_N_CTX", DEFAULT_CONTEXT_SIZE),
            n_gpu_layers=_read_int_env("LOCAL_LLM_N_GPU_LAYERS", DEFAULT_GPU_LAYERS),
            max_tokens=_read_int_env("LOCAL_LLM_MAX_TOKENS", DEFAULT_MAX_TOKENS),
            temperature=_read_float_env("LOCAL_LLM_TEMPERATURE", DEFAULT_TEMPERATURE),
        )

    def generate(self, prompt: str, *, response_format: str = "json") -> str:
        messages = [
            {
                "role": "system",
                "content": "You are a careful meeting event analysis assistant. Return only valid JSON when JSON is requested.",
            },
            {"role": "user", "content": prompt},
        ]
        kwargs: dict[str, Any] = {
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if response_format == "json":
            kwargs["response_format"] = {"type": "json_object"}

        response = self._llm.create_chat_completion(**kwargs)
        self.last_usage = _extract_usage(response)
        return _extract_chat_content(response)


class ServerLLMProvider:
    """OpenAI-compatible HTTP provider for local model servers."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str = "EMPTY",
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        timeout_sec: int = DEFAULT_SERVER_TIMEOUT_SEC,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model.strip()
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout_sec = timeout_sec
        if not self.base_url:
            raise LLMProviderError("SERVER_BASE_URL must not be empty for LLM_PROVIDER=server.")
        if not self.model:
            raise LLMProviderError("SERVER_MODEL must not be empty for LLM_PROVIDER=server.")
        self.last_usage: dict[str, int] = {}

    @classmethod
    def from_config(cls, _config: AppConfig) -> "ServerLLMProvider":
        return cls(
            base_url=os.getenv("SERVER_BASE_URL", "http://127.0.0.1:8080/v1"),
            model=os.getenv("SERVER_MODEL", "local-model"),
            api_key=os.getenv("SERVER_API_KEY", "EMPTY"),
            max_tokens=_read_int_env("SERVER_MAX_TOKENS", DEFAULT_MAX_TOKENS),
            temperature=_read_float_env("SERVER_TEMPERATURE", DEFAULT_TEMPERATURE),
            timeout_sec=_read_int_env("SERVER_TIMEOUT_SEC", DEFAULT_SERVER_TIMEOUT_SEC),
        )

    def generate(self, prompt: str, *, response_format: str = "json") -> str:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a careful meeting event analysis assistant. "
                        "Return only valid JSON when JSON is requested."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if response_format == "json":
            payload["response_format"] = {"type": "json_object"}

        endpoint = f"{self.base_url}/chat/completions"
        data = json.dumps(payload).encode("utf-8")
        req = urllib_request.Request(
            endpoint,
            method="POST",
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )
        try:
            with urllib_request.urlopen(req, timeout=self.timeout_sec) as response:
                response_payload = json.loads(response.read().decode("utf-8"))
        except Exception as exc:
            raise LLMProviderError(f"Server provider request failed: {exc}") from exc
        self.last_usage = _extract_usage(response_payload)
        return _extract_chat_content(response_payload)


class LLMClient:
    """High-level client that renders prompts and validates JSON responses."""

    def __init__(self, provider: LLMProvider, prompt_loader: PromptLoader | None = None) -> None:
        self.provider = provider
        self.prompt_loader = prompt_loader or PromptLoader()
        self._request_counter = 0
        self._usage_totals: dict[str, int] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

    @classmethod
    def from_config(cls, config: AppConfig) -> "LLMClient":
        provider_name = config.llm_provider.lower()
        if provider_name == "local":
            provider: LLMProvider = LocalLLMProvider.from_config(config)
        elif provider_name == "server":
            provider = ServerLLMProvider.from_config(config)
        else:
            raise LLMProviderError(
                "Unsupported LLM_PROVIDER="
                f"{config.llm_provider!r}. Supported providers: local, server."
            )
        return cls(provider=provider)

    def render_prompt(self, prompt_name: str, variables: dict[str, Any] | None = None) -> str:
        """Expose prompt rendering for dry-run tests and debug logging."""

        return self.prompt_loader.render(prompt_name, variables)

    def generate_json(
        self,
        prompt_name: str,
        variables: dict[str, Any] | None = None,
        *,
        schema: type[T] | Any | None = None,
    ) -> T | Any:
        """Render prompt, call the model, parse JSON, and validate the result.

        ``schema`` may be a dataclass type such as ``EventCase`` or a generic
        list form such as ``list[EventCandidate]``.  If omitted, parsed JSON is
        returned without dataclass conversion.
        """

        prompt = self.render_prompt(prompt_name, variables)
        response_text = self._generate_with_metrics(prompt, prompt_name, "json")
        payload = parse_json_response(response_text)
        if schema is None:
            return payload
        return validate_json_schema(payload, schema)

    def generate_text(
        self,
        prompt_name: str,
        variables: dict[str, Any] | None = None,
    ) -> str:
        """Render a prompt and return unstructured model text.

        This is intended for the final Markdown report stage, where the output
        policy requires Markdown rather than JSON. Structured stages should use
        :meth:`generate_json` so schema validation is not skipped.
        """

        prompt = self.render_prompt(prompt_name, variables)
        return self._generate_with_metrics(prompt, prompt_name, "text")

    def _generate_with_metrics(self, prompt: str, prompt_name: str, response_format: str) -> str:
        self._request_counter += 1
        request_id = self._request_counter
        started = time.perf_counter()
        response_text = self.provider.generate(prompt, response_format=response_format)
        elapsed_sec = time.perf_counter() - started
        usage = getattr(self.provider, "last_usage", {})
        prompt_tokens = usage.get("prompt_tokens") if isinstance(usage, dict) else None
        completion_tokens = usage.get("completion_tokens") if isinstance(usage, dict) else None
        total_tokens = usage.get("total_tokens") if isinstance(usage, dict) else None
        request_completion_tok_per_sec = (
            round(completion_tokens / elapsed_sec, 3)
            if isinstance(completion_tokens, int) and elapsed_sec > 0
            else None
        )
        self._usage_totals["prompt_tokens"] += prompt_tokens if isinstance(prompt_tokens, int) else 0
        self._usage_totals["completion_tokens"] += (
            completion_tokens if isinstance(completion_tokens, int) else 0
        )
        self._usage_totals["total_tokens"] += total_tokens if isinstance(total_tokens, int) else 0
        LOGGER.info(
            "LLM request #%s prompt=%s format=%s elapsed=%.3fs prompt_tokens=%s completion_tokens=%s total_tokens=%s request_completion_tok_per_sec=%s",
            request_id,
            prompt_name,
            response_format,
            elapsed_sec,
            prompt_tokens,
            completion_tokens,
            total_tokens,
            request_completion_tok_per_sec,
        )
        self._write_metrics_jsonl(
            request_id=request_id,
            prompt_name=prompt_name,
            response_format=response_format,
            elapsed_sec=elapsed_sec,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            request_completion_tok_per_sec=request_completion_tok_per_sec,
        )
        return response_text

    def _write_metrics_jsonl(
        self,
        *,
        request_id: int,
        prompt_name: str,
        response_format: str,
        elapsed_sec: float,
        prompt_tokens: int | None,
        completion_tokens: int | None,
        total_tokens: int | None,
        request_completion_tok_per_sec: float | None,
    ) -> None:
        log_path = Path(os.getenv("LLM_METRICS_LOG_PATH", "logs/llm_metrics.jsonl"))
        log_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "ts": datetime.now(UTC).isoformat(),
            "request_id": request_id,
            "prompt_name": prompt_name,
            "response_format": response_format,
            "elapsed_sec": round(elapsed_sec, 3),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "request_completion_tok_per_sec": request_completion_tok_per_sec,
        }
        with log_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def usage_totals_snapshot(self) -> dict[str, int]:
        """Return cumulative usage totals for stage-level throughput metrics."""

        return dict(self._usage_totals)


def build_llm_client(config: AppConfig) -> LLMClient:
    """Factory used by pipeline stages to create the configured LLM client."""

    return LLMClient.from_config(config)


def parse_json_response(response_text: str) -> Any:
    """Parse one JSON value from an LLM response, accepting fenced JSON blocks."""

    text = response_text.strip()
    if not text:
        raise LLMJSONParseError("LLM response was empty; expected valid JSON.")

    fenced_match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL)
    if fenced_match:
        text = fenced_match.group(1).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise LLMJSONParseError(
            "LLM response was not valid JSON. Configure the prompt/provider to return JSON only."
        ) from exc


def validate_json_schema(payload: Any, schema: type[T] | Any) -> T | Any:
    """Validate parsed JSON against dataclass schemas used by the pipeline."""

    try:
        return _coerce_value(schema, payload, path="$")
    except (TypeError, ValueError) as exc:
        raise LLMResponseValidationError(
            f"LLM JSON response failed schema validation for {_schema_name(schema)}: {exc}"
        ) from exc


def _format_prompt_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(_to_jsonable(value), ensure_ascii=False, indent=2)


def _to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return {field.name: _to_jsonable(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, list | tuple):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    return value


def _coerce_value(expected_type: Any, value: Any, *, path: str) -> Any:
    origin = get_origin(expected_type)
    args = get_args(expected_type)

    if origin is list:
        if not isinstance(value, list):
            raise ValueError(f"{path}: expected list, got {type(value).__name__}")
        item_type = args[0] if args else Any
        return [
            _coerce_value(item_type, item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]

    if origin is not None and type(None) in args:
        if value is None:
            return None
        non_none_args = [arg for arg in args if arg is not type(None)]
        if len(non_none_args) == 1:
            return _coerce_value(non_none_args[0], value, path=path)

    if is_dataclass(expected_type):
        if not isinstance(value, dict):
            raise ValueError(f"{path}: expected object, got {type(value).__name__}")
        return _dataclass_from_dict(expected_type, value, path=path)

    if expected_type is dict:
        if not isinstance(value, dict):
            raise ValueError(f"{path}: expected object, got {type(value).__name__}")
        return value

    if expected_type in (str, int, float, bool):
        if not isinstance(value, expected_type):
            raise ValueError(
                f"{path}: expected {expected_type.__name__}, got {type(value).__name__}"
            )
        return value

    if expected_type is Any:
        return value

    return value


def _dataclass_from_dict(model: type[T], data: dict[str, Any], *, path: str) -> T:
    model_fields = {field.name: field for field in fields(model)}
    type_hints = get_type_hints(model)
    unexpected_fields = sorted(set(data) - set(model_fields))
    if unexpected_fields:
        raise ValueError(f"{path}: unexpected fields: {unexpected_fields}")

    missing_fields = sorted(field_name for field_name in model_fields if field_name not in data)
    if missing_fields:
        raise ValueError(f"{path}: missing fields: {missing_fields}")

    kwargs = {
        field_name: _coerce_value(type_hints.get(field_name, field.type), data[field_name], path=f"{path}.{field_name}")
        for field_name, field in model_fields.items()
    }
    return model(**kwargs)  # type: ignore[misc]


def _schema_name(schema: Any) -> str:
    return getattr(schema, "__name__", str(schema))


def _read_int_env(key: str, default: int) -> int:
    raw = os.getenv(key, str(default))
    try:
        return int(raw)
    except ValueError as exc:
        raise LLMProviderError(f"{key} must be an int, got {raw!r}.") from exc


def _read_float_env(key: str, default: float) -> float:
    raw = os.getenv(key, str(default))
    try:
        return float(raw)
    except ValueError as exc:
        raise LLMProviderError(f"{key} must be a float, got {raw!r}.") from exc


def _load_llama_class() -> Any:
    if importlib.util.find_spec("llama_cpp") is None:
        raise LLMProviderError(
            "llama-cpp-python is required for LLM_PROVIDER=local. "
            "Install it in the offline runtime image or use the project extra before going offline."
        )
    llama_cpp = importlib.import_module("llama_cpp")
    return llama_cpp.Llama


def _extract_chat_content(response: Any) -> str:
    try:
        content = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise LLMProviderError(
            "Local llama.cpp response did not contain choices[0].message.content."
        ) from exc
    if not isinstance(content, str) or not content.strip():
        raise LLMProviderError("Local llama.cpp response content was empty.")
    return content


def _extract_usage(response: Any) -> dict[str, int]:
    if not isinstance(response, dict):
        return {}
    usage = response.get("usage")
    if not isinstance(usage, dict):
        return {}
    result: dict[str, int] = {}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = usage.get(key)
        if isinstance(value, int):
            result[key] = value
    return result
