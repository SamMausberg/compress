"""OpenAI external LLM baseline runner regressions."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import pytest

from vpm.evaluation.llm_baseline import (
    read_hard_llm_predictions,
    read_llm_predictions,
    run_openai_release_baselines,
)
from vpm.providers.openai_baseline import (
    LlmTaskKind,
    OpenAILlmConfig,
    openai_config_from_env,
    response_text,
    run_openai_llm_predictions,
)


def test_openai_runner_writes_c1_prediction_jsonl(tmp_path: Path) -> None:
    tasks = tmp_path / "tasks.jsonl"
    tasks.write_text(
        json.dumps(
            {
                "task_id": "c1-openai",
                "prompt": "Choose the operation.",
                "allowed_operations": ["add", "mul"],
            }
        )
        + "\n"
    )
    predictions = tmp_path / "predictions.jsonl"
    seen_payloads: list[Mapping[str, object]] = []

    def fake_responder(
        payload: Mapping[str, object],
        config: OpenAILlmConfig,
    ) -> Mapping[str, object]:
        seen_payloads.append(payload)
        assert config.model == "gpt-test"
        return {"id": "resp_c1", "output_text": '{"operation":"add"}'}

    written = run_openai_llm_predictions(
        tasks,
        predictions,
        kind=LlmTaskKind.C1,
        config=OpenAILlmConfig(model="gpt-test", api_key="test-key"),
        responder=fake_responder,
    )
    parsed, errors = read_llm_predictions(predictions)

    assert written == 1
    assert errors == ()
    assert parsed["c1-openai"].operation == "add"
    assert parsed["c1-openai"].compute_units == 1.0
    assert seen_payloads[0]["model"] == "gpt-test"
    assert seen_payloads[0]["input"] == "Choose the operation."


def test_openai_runner_writes_hard_prediction_with_evidence(tmp_path: Path) -> None:
    tasks = tmp_path / "hard-tasks.jsonl"
    tasks.write_text(
        json.dumps(
            {
                "task_id": "hard-openai",
                "prompt": "Answer from evidence.",
                "evidence": ["Lemma: x=42."],
            }
        )
        + "\n"
    )
    predictions = tmp_path / "hard-predictions.jsonl"
    seen_inputs: list[str] = []

    def fake_responder(
        payload: Mapping[str, object],
        _config: OpenAILlmConfig,
    ) -> Mapping[str, object]:
        model_input = payload["input"]
        assert isinstance(model_input, str)
        seen_inputs.append(model_input)
        return {"id": "resp_hard", "output_text": '{"answer":"42"}'}

    written = run_openai_llm_predictions(
        tasks,
        predictions,
        kind=LlmTaskKind.HARD,
        config=OpenAILlmConfig(model="gpt-test", api_key="test-key"),
        responder=fake_responder,
    )
    parsed, errors = read_hard_llm_predictions(predictions)

    assert written == 1
    assert errors == ()
    assert parsed["hard-openai"].answer == "42"
    assert parsed["hard-openai"].compute_units == 1.0
    assert "Lemma: x=42." in seen_inputs[0]


def test_response_text_collects_nested_response_output() -> None:
    response = {
        "output": [
            {
                "type": "message",
                "content": [{"type": "output_text", "text": '{"operation":"mul"}'}],
            }
        ]
    }

    assert response_text(response) == '{"operation":"mul"}'


def test_openai_release_runner_writes_scored_baseline_jsons(tmp_path: Path) -> None:
    def fake_responder(
        payload: Mapping[str, object],
        _config: OpenAILlmConfig,
    ) -> Mapping[str, object]:
        text = payload.get("text")
        assert isinstance(text, dict)
        response_format = text["format"]
        assert isinstance(response_format, dict)
        schema = response_format["schema"]
        assert isinstance(schema, dict)
        properties = schema["properties"]
        assert isinstance(properties, dict)
        if "operation" in properties:
            operation_schema = properties["operation"]
            assert isinstance(operation_schema, dict)
            allowed = operation_schema["enum"]
            assert isinstance(allowed, list)
            return {"id": "resp_c1", "output_text": json.dumps({"operation": allowed[0]})}
        return {"id": "resp_hard", "output_text": json.dumps({"answer": "wrong"})}

    artifacts = run_openai_release_baselines(
        tmp_path,
        config=OpenAILlmConfig(model="gpt-test", api_key="test-key"),
        c1_limit=0,
        responder=fake_responder,
    )

    assert artifacts.passed is True
    assert artifacts.c1_baseline_json.exists()
    assert artifacts.hard_baseline_json.exists()
    assert json.loads(artifacts.c1_baseline_json.read_text())["task_kind"] == "c1"
    assert json.loads(artifacts.hard_baseline_json.read_text())["task_kind"] == "hard"
    assert artifacts.env_exports == (
        f"VPM_LLM_BASELINE_JSON={artifacts.c1_baseline_json}",
        f"VPM_HARD_LLM_BASELINE_JSON={artifacts.hard_baseline_json}",
    )


def test_openai_config_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        openai_config_from_env(model="gpt-test")
