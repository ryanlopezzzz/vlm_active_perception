from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.agents.llm_agent import LLM
from src.runners.run_experiment import _aggregate_llm_usage


class _Response:
    def __init__(self, payload: dict, status_code: int = 200):
        self._payload = payload
        self.status_code = status_code
        self.text = json.dumps(payload)

    def json(self) -> dict:
        return self._payload


class LLMUsageTests(unittest.TestCase):
    def test_reasoning_payload_enables_effort_for_fable_style_requests(self) -> None:
        llm = LLM(model="anthropic/claude-fable-5")

        payload = llm._build_reasoning_payload(effort="high", reasoning_max_tokens=None)

        self.assertEqual(payload, {"enabled": True, "effort": "high"})

    def test_reasoning_payload_enables_direct_reasoning_budget(self) -> None:
        llm = LLM(model="anthropic/claude-fable-5")

        payload = llm._build_reasoning_payload(effort="high", reasoning_max_tokens=2000)

        self.assertEqual(payload, {"enabled": True, "max_tokens": 2000})

    def test_reasoning_payload_can_be_disabled_for_instruct_models(self) -> None:
        llm = LLM(model="qwen/qwen3-vl-32b-instruct")

        payload = llm._build_reasoning_payload(
            effort="none", reasoning_max_tokens=None
        )

        self.assertEqual(payload, {})

    def test_prepare_messages_preserves_reasoning_details_for_continuation(self) -> None:
        llm = LLM(model="mock")
        reasoning_details = [
            {
                "type": "reasoning.text",
                "text": "preserved reasoning",
                "format": "anthropic-claude-v1",
            }
        ]
        llm.messages = [
            {"role": "user", "content": "first question"},
            {
                "role": "assistant",
                "content": "first answer",
                "reasoning": "legacy reasoning string should not be replayed",
                "reasoning_details": reasoning_details,
                "usage": {"total_tokens": 10},
                "response_id": "resp_123",
            },
            {"role": "user", "content": "follow-up"},
        ]

        prepared = llm._prepare_messages_for_llm()

        self.assertNotIn("reasoning", prepared[1])
        self.assertNotIn("usage", prepared[1])
        self.assertNotIn("response_id", prepared[1])
        self.assertEqual(prepared[1]["reasoning_details"], reasoning_details)

    def test_json_requests_enable_openrouter_response_healing(self) -> None:
        response = _Response(
            {
                "id": "gen-json",
                "model": "mock/provider",
                "choices": [{"message": {"content": '{"ok": true}'}}],
                "usage": {},
            }
        )
        llm = LLM(model="mock")
        llm.add_user_message("return JSON")

        with patch("src.agents.llm_agent.requests.post", return_value=response) as post:
            answer = llm.query_llm(require_json=True)

        request_payload = post.call_args.kwargs["json"]
        self.assertEqual(request_payload["response_format"], {"type": "json_object"})
        self.assertEqual(request_payload["plugins"], [{"id": "response-healing"}])
        self.assertEqual(json.loads(answer), {"ok": True})

    def test_non_json_requests_do_not_enable_response_healing(self) -> None:
        response = _Response(
            {
                "id": "gen-text",
                "model": "mock/provider",
                "choices": [{"message": {"content": "plain text"}}],
                "usage": {},
            }
        )
        llm = LLM(model="mock")
        llm.add_user_message("return text")

        with patch("src.agents.llm_agent.requests.post", return_value=response) as post:
            llm.query_llm(require_json=False)

        request_payload = post.call_args.kwargs["json"]
        self.assertNotIn("response_format", request_payload)
        self.assertNotIn("plugins", request_payload)

    def test_usage_includes_validation_retry_cost(self) -> None:
        responses = [
            _Response(
                {
                    "id": "gen-invalid",
                    "model": "mock/provider",
                    "choices": [{"message": {"content": "{\"ok\": false}"}}],
                    "usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 2,
                        "total_tokens": 12,
                        "cost": 0.001,
                    },
                }
            ),
            _Response(
                {
                    "id": "gen-valid",
                    "model": "mock/provider",
                    "choices": [{"message": {"content": "{\"ok\": true}"}}],
                    "usage": {
                        "prompt_tokens": 20,
                        "completion_tokens": 3,
                        "total_tokens": 23,
                        "cost": 0.002,
                    },
                }
            ),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            usage_path = Path(tmpdir) / "llm_usage.json"
            llm = LLM(model="mock", save_usage_json_path=str(usage_path))
            llm.add_user_message("test")
            with patch("src.agents.llm_agent.requests.post", side_effect=responses):
                answer = llm.query_llm(
                    num_tries=2,
                    verify=lambda text: json.loads(text)["ok"],
                )

            self.assertEqual(json.loads(answer), {"ok": True})
            report = json.loads(usage_path.read_text(encoding="utf-8"))
            self.assertEqual(report["summary"]["request_attempts"], 2)
            self.assertEqual(report["summary"]["prompt_tokens"], 30)
            self.assertEqual(report["summary"]["completion_tokens"], 5)
            self.assertEqual(report["summary"]["total_tokens"], 35)
            self.assertAlmostEqual(report["summary"]["cost_credits"], 0.003)
            self.assertTrue(report["summary"]["cost_complete"])
            self.assertGreater(report["summary"]["request_wall_time_sec"], 0.0)
            self.assertGreater(report["summary"]["avg_request_wall_time_sec"], 0.0)
            self.assertGreaterEqual(report["summary"]["max_request_wall_time_sec"], report["summary"]["min_request_wall_time_sec"])
            self.assertEqual(report["calls"][0]["outcome"], "validation_failed")
            self.assertEqual(report["calls"][1]["outcome"], "accepted")
            for call in report["calls"]:
                self.assertIsInstance(call["started_at"], str)
                self.assertIsInstance(call["finished_at"], str)
                self.assertGreaterEqual(call["duration_sec"], 0.0)

    def test_aggregate_usage_sums_runs(self) -> None:
        aggregate = _aggregate_llm_usage(
            [
                {
                    "model_requested": "model-a",
                    "request_attempts": 2,
                    "api_responses": 2,
                    "accepted_responses": 1,
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                    "cost_credits": 0.01,
                    "responses_with_cost": 2,
                    "cost_complete": True,
                    "request_wall_time_sec": 1.5,
                    "min_request_wall_time_sec": 0.5,
                    "max_request_wall_time_sec": 1.0,
                },
                {
                    "model_requested": "model-a",
                    "request_attempts": 1,
                    "api_responses": 1,
                    "accepted_responses": 1,
                    "prompt_tokens": 20,
                    "completion_tokens": 7,
                    "total_tokens": 27,
                    "cost_credits": 0.02,
                    "responses_with_cost": 1,
                    "cost_complete": True,
                    "request_wall_time_sec": 2.0,
                    "min_request_wall_time_sec": 2.0,
                    "max_request_wall_time_sec": 2.0,
                },
            ]
        )

        self.assertEqual(aggregate["run_count"], 2)
        self.assertEqual(aggregate["total_tokens"], 42)
        self.assertAlmostEqual(aggregate["cost_credits"], 0.03)
        self.assertTrue(aggregate["cost_complete"])
        self.assertAlmostEqual(aggregate["request_wall_time_sec"], 3.5)
        self.assertAlmostEqual(aggregate["avg_request_wall_time_sec"], 3.5 / 3)
        self.assertAlmostEqual(aggregate["min_request_wall_time_sec"], 0.5)
        self.assertAlmostEqual(aggregate["max_request_wall_time_sec"], 2.0)


if __name__ == "__main__":
    unittest.main()
