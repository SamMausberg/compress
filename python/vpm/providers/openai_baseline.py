"""OpenAI Responses API runner for external LLM baseline predictions."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import cast


class LlmTaskKind(StrEnum):
    """External LLM task file shapes supported by the runner."""

    C1 = "c1"
    HARD = "hard"


@dataclass(frozen=True)
class OpenAILlmConfig:
    """Configuration for one OpenAI external-baseline run."""

    model: str
    api_key: str
    base_url: str = "https://api.openai.com/v1"
    timeout_seconds: float = 60.0
    reasoning_effort: str | None = None


OpenAIResponder = Callable[[Mapping[str, object], OpenAILlmConfig], Mapping[str, object]]


def openai_config_from_env(
    *,
    model: str,
    api_key_env: str = "OPENAI_API_KEY",
    base_url: str = "https://api.openai.com/v1",
    timeout_seconds: float = 60.0,
    reasoning_effort: str | None = None,
) -> OpenAILlmConfig:
    """Build OpenAI runner config from an API-key environment variable."""
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise ValueError(f"{api_key_env} is not set")
    return OpenAILlmConfig(
        model=model,
        api_key=api_key,
        base_url=base_url,
        timeout_seconds=timeout_seconds,
        reasoning_effort=reasoning_effort,
    )


def run_openai_llm_predictions(
    tasks_path: Path,
    predictions_path: Path,
    *,
    kind: LlmTaskKind,
    config: OpenAILlmConfig,
    responder: OpenAIResponder | None = None,
) -> int:
    """Run one OpenAI response per exported task and write prediction JSONL."""
    tasks = read_task_jsonl(tasks_path)
    active_responder = responder or submit_openai_response
    lines = [
        json.dumps(
            prediction_for_task(task, kind, config, active_responder),
            sort_keys=True,
        )
        for task in tasks
    ]
    predictions_path.write_text("".join(f"{line}\n" for line in lines))
    return len(lines)


def prediction_for_task(
    task: Mapping[str, object],
    kind: LlmTaskKind,
    config: OpenAILlmConfig,
    responder: OpenAIResponder,
) -> dict[str, object]:
    """Generate one prediction record from one exported task record."""
    payload = response_payload(task, kind, config)
    response = responder(payload, config)
    raw_output = response_text(response)
    parsed = parse_json_object(raw_output)
    task_id = required_string(task, "task_id")
    prediction: dict[str, object] = {
        "task_id": task_id,
        "compute_units": 1.0,
        "raw_output": raw_output,
        "model": config.model,
    }
    response_id = response.get("id")
    if isinstance(response_id, str):
        prediction["response_id"] = response_id
    if kind is LlmTaskKind.C1:
        operation = parsed.get("operation")
        prediction["operation"] = operation if isinstance(operation, str) else None
    else:
        answer = parsed.get("answer")
        prediction["answer"] = answer if isinstance(answer, str) else None
    return prediction


def response_payload(
    task: Mapping[str, object],
    kind: LlmTaskKind,
    config: OpenAILlmConfig,
) -> dict[str, object]:
    """Build a Responses API payload with a strict JSON schema."""
    payload: dict[str, object] = {
        "model": config.model,
        "instructions": (
            "You are an external LLM baseline. Answer the task using only the "
            "task content. Return JSON matching the supplied schema."
        ),
        "input": prompt_for_task(task, kind),
        "text": {"format": response_format(task, kind)},
    }
    if config.reasoning_effort is not None:
        payload["reasoning"] = {"effort": config.reasoning_effort}
    return payload


def response_format(task: Mapping[str, object], kind: LlmTaskKind) -> dict[str, object]:
    """Return the structured-output format for one baseline task kind."""
    if kind is LlmTaskKind.C1:
        allowed = string_sequence(task.get("allowed_operations"), "allowed_operations")
        properties: dict[str, object] = {"operation": {"type": "string", "enum": list(allowed)}}
        required = ["operation"]
    else:
        properties = {"answer": {"type": "string"}}
        required = ["answer"]
    return {
        "type": "json_schema",
        "name": f"vpm_{kind.value}_baseline_prediction",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": properties,
            "required": required,
        },
    }


def prompt_for_task(task: Mapping[str, object], kind: LlmTaskKind) -> str:
    """Build the user input text for one exported task."""
    prompt = required_string(task, "prompt")
    if kind is not LlmTaskKind.HARD:
        return prompt
    evidence = string_sequence(task.get("evidence"), "evidence")
    evidence_text = "\n".join(f"- {line}" for line in evidence)
    return f"{prompt}\nEvidence:\n{evidence_text}"


def submit_openai_response(
    payload: Mapping[str, object],
    config: OpenAILlmConfig,
) -> Mapping[str, object]:
    """Submit one Responses API request and return the decoded response body."""
    endpoint = f"{config.base_url.rstrip('/')}/responses"
    parsed_endpoint = urllib.parse.urlparse(endpoint)
    if parsed_endpoint.scheme not in {"http", "https"}:
        raise ValueError("OpenAI base URL must use http or https")
    request = urllib.request.Request(  # noqa: S310 - endpoint scheme is validated above.
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(  # noqa: S310 - endpoint scheme is validated above.
            request,
            timeout=config.timeout_seconds,
        ) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenAI Responses API failed ({exc.code}): {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"OpenAI Responses API request failed: {exc.reason}") from exc
    try:
        decoded = json.loads(body)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"OpenAI Responses API returned invalid JSON: {exc.msg}") from exc
    if not isinstance(decoded, dict):
        raise TypeError("OpenAI Responses API returned a non-object response")
    return cast(Mapping[str, object], decoded)


def response_text(response: Mapping[str, object]) -> str:
    """Extract text from a Responses API body without assuming output position."""
    output_text = response.get("output_text")
    if isinstance(output_text, str):
        return output_text
    pieces: list[str] = []
    output = response.get("output")
    if isinstance(output, list):
        for item in cast(list[object], output):
            collect_text_piece(item, pieces)
    if pieces:
        return "".join(pieces)
    return json.dumps(response, sort_keys=True)


def collect_text_piece(item: object, pieces: list[str]) -> None:
    """Collect text fragments from one Responses API output item."""
    if not isinstance(item, dict):
        return
    item_map = cast(Mapping[str, object], item)
    text = item_map.get("text")
    if isinstance(text, str):
        pieces.append(text)
    content = item_map.get("content")
    if not isinstance(content, list):
        return
    for part in cast(list[object], content):
        if not isinstance(part, dict):
            continue
        part_text = cast(Mapping[str, object], part).get("text")
        if isinstance(part_text, str):
            pieces.append(part_text)


def parse_json_object(text: str) -> Mapping[str, object]:
    """Parse a model JSON object, returning an empty object on malformed output."""
    try:
        decoded = json.loads(text)
    except json.JSONDecodeError:
        return {}
    if not isinstance(decoded, dict):
        return {}
    return cast(Mapping[str, object], decoded)


def read_task_jsonl(path: Path) -> tuple[Mapping[str, object], ...]:
    """Read exported task JSONL for an OpenAI baseline run."""
    tasks: list[Mapping[str, object]] = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            decoded = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"line {line_number}: invalid task JSON: {exc.msg}") from exc
        if not isinstance(decoded, dict):
            raise TypeError(f"line {line_number}: task must be a JSON object")
        tasks.append(cast(Mapping[str, object], decoded))
    return tuple(tasks)


def required_string(payload: Mapping[str, object], key: str) -> str:
    """Read a required string field from a JSON object."""
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a non-empty string")
    return value


def string_sequence(value: object, field: str) -> tuple[str, ...]:
    """Read a required sequence of strings from a JSON value."""
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise TypeError(f"{field} must be a list of strings")
    strings: list[str] = []
    for item in cast(Sequence[object], value):
        if not isinstance(item, str):
            raise TypeError(f"{field} must be a list of strings")
        strings.append(item)
    if not strings:
        raise ValueError(f"{field} must not be empty")
    return tuple(strings)


__all__ = [
    "LlmTaskKind",
    "OpenAILlmConfig",
    "OpenAIResponder",
    "openai_config_from_env",
    "parse_json_object",
    "read_task_jsonl",
    "response_payload",
    "response_text",
    "run_openai_llm_predictions",
    "submit_openai_response",
]
