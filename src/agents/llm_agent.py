import base64
import copy
import json
import logging
import os
import pathlib
import time
import urllib
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class LLM:
    def __init__(
        self,
        api_key_path: str = "openrouter_api.txt",
        model: str = "google/gemini-2.5-flash",
        save_messages_json_path: Optional[str] = None,
        save_messages_markdown_path: Optional[str] = None,
        save_usage_json_path: Optional[str] = None,
        load_messages_json_path: Optional[str] = None,
    ):
        self.model = model
        self.save_messages_json_path = save_messages_json_path
        self.save_messages_markdown_path = save_messages_markdown_path
        self.save_usage_json_path = save_usage_json_path
        self.api_calls: List[dict[str, Any]] = []
        self.request_attempts = 0

        if self.model.startswith("mock"):
            self.api_key = ""
            self.url = ""
            self.headers = {}
        else:
            self.api_key = pathlib.Path(api_key_path).read_text().strip()
            self.url = "https://openrouter.ai/api/v1/chat/completions"
            self.headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
        self.messages: List[dict] = []
        if load_messages_json_path:
            self._load_messages_json(load_messages_json_path)

    def add_system_message(self, system_message: str) -> None:
        self.messages.append({"role": "system", "content": system_message})
        self._save_messages_if_needed()

    def add_user_message(self, user_message: str) -> None:
        self._start_user_message_if_needed()
        self.messages[-1]["content"].append({"type": "text", "text": user_message})
        self._save_messages_if_needed()

    def add_user_image(self, image_path: str) -> None:
        self._start_user_message_if_needed()
        self.messages[-1]["content"].append(
            {"type": "image_url", "image_url": {"url": image_path}}
        )
        self._save_messages_if_needed()

    def query_llm(
        self,
        max_tokens: int = 6000,
        effort: str = "low",
        reasoning_max_tokens: Optional[int] = 2000,
        require_json: bool = True,
        num_tries: int = 1,
        verify: Optional[Callable[[str], bool]] = None,
    ) -> str:
        messages_for_llm = self._prepare_messages_for_llm()
        reasoning_payload = self._build_reasoning_payload(
            effort=effort,
            reasoning_max_tokens=reasoning_max_tokens,
        )
        payload = {
            "model": self.model,
            "messages": messages_for_llm,
            "max_tokens": max_tokens,
        }
        if reasoning_payload:
            payload["reasoning"] = reasoning_payload
        if require_json:
            payload["response_format"] = {"type": "json_object"}
            payload["plugins"] = [{"id": "response-healing"}]

        last_error: Exception | None = None
        for _ in range(num_tries):
            # Validators may append actionable rejection feedback before a retry.
            payload["messages"] = self._prepare_messages_for_llm()
            self.request_attempts += 1
            api_call: dict[str, Any] | None = None
            request_started_at = _utc_now_iso()
            request_start_perf = time.perf_counter()
            try:
                api_call = self._start_api_call(request_started_at)
                response = requests.post(
                    self.url, headers=self.headers, json=payload, timeout=120
                )
                try:
                    out = response.json()
                except Exception as exc:
                    raise RuntimeError(
                        f"Non-JSON response from LLM API (status={response.status_code}): "
                        f"{response.text[:500]}"
                    ) from exc

                self._record_api_response(api_call, out, http_status=response.status_code)

                if response.status_code >= 400:
                    message = (
                        out.get("error", {}).get("message")
                        if isinstance(out, dict)
                        else None
                    )
                    if not message:
                        message = json.dumps(out, indent=2)[:1200]
                    self._finish_api_call(
                        api_call,
                        request_start_perf,
                        outcome="api_error",
                    )
                    raise RuntimeError(
                        f"LLM API error (status={response.status_code}): {message}"
                    )

                if "choices" not in out or not out["choices"]:
                    self._finish_api_call(
                        api_call,
                        request_start_perf,
                        outcome="invalid_response",
                    )
                    raise RuntimeError(
                        f"Empty or invalid response: {json.dumps(out, indent=2)}"
                    )
                message = out["choices"][0]["message"]
                answer = message["content"]
                reasoning = message.get("reasoning", "")
                reasoning_details = message.get("reasoning_details")
                if verify is not None and not verify(answer):
                    self._finish_api_call(
                        api_call,
                        request_start_perf,
                        outcome="validation_failed",
                    )
                    raise ValueError(f"Validation failed on: {answer}")

                assistant_message = {"role": "assistant", "content": answer, "reasoning": reasoning}
                if reasoning_details is not None:
                    assistant_message["reasoning_details"] = reasoning_details
                if api_call.get("usage"):
                    assistant_message["usage"] = copy.deepcopy(api_call["usage"])
                if api_call.get("response_id"):
                    assistant_message["response_id"] = api_call["response_id"]
                self._finish_api_call(
                    api_call,
                    request_start_perf,
                    outcome="accepted",
                    accepted=True,
                )
                self.messages.append(assistant_message)
                self._save_messages_if_needed()
                return answer
            except Exception as exc:  # pragma: no cover - network failure path
                if api_call is not None and api_call.get("duration_sec") is None:
                    self._finish_api_call(
                        api_call,
                        request_start_perf,
                        outcome="transport_error",
                        error=repr(exc),
                    )
                last_error = exc

        raise RuntimeError(f"Last error: {last_error}")

    def get_api_usage(self) -> tuple[float, float]:
        response = requests.get("https://openrouter.ai/api/v1/key", headers=self.headers)
        data = response.json()
        usage_all_time = data["data"]["usage"]
        remaining_credits = data["data"]["limit_remaining"]
        return usage_all_time, remaining_credits

    def get_usage_summary(self) -> dict[str, Any]:
        usage_records = [
            call["usage"] for call in self.api_calls if isinstance(call.get("usage"), dict)
        ]
        costs = [
            float(usage["cost"])
            for usage in usage_records
            if isinstance(usage.get("cost"), (int, float)) and not isinstance(usage.get("cost"), bool)
        ]
        durations = [
            float(call["duration_sec"])
            for call in self.api_calls
            if isinstance(call.get("duration_sec"), (int, float))
            and not isinstance(call.get("duration_sec"), bool)
        ]
        return {
            "model_requested": self.model,
            "request_attempts": self.request_attempts,
            "api_responses": len(self.api_calls),
            "accepted_responses": sum(bool(call.get("accepted")) for call in self.api_calls),
            "prompt_tokens": self._sum_usage_number(usage_records, "prompt_tokens"),
            "completion_tokens": self._sum_usage_number(usage_records, "completion_tokens"),
            "total_tokens": self._sum_usage_number(usage_records, "total_tokens"),
            "cost_credits": sum(costs),
            "responses_with_cost": len(costs),
            "cost_complete": len(costs) == len(usage_records) and bool(usage_records),
            "request_wall_time_sec": sum(durations),
            "avg_request_wall_time_sec": (sum(durations) / len(durations)) if durations else 0.0,
            "min_request_wall_time_sec": min(durations) if durations else 0.0,
            "max_request_wall_time_sec": max(durations) if durations else 0.0,
        }

    def get_usage_report(self) -> dict[str, Any]:
        return {
            "summary": self.get_usage_summary(),
            "calls": copy.deepcopy(self.api_calls),
        }

    def _start_user_message_if_needed(self) -> None:
        if self.messages and self.messages[-1].get("role") == "user":
            return
        self.messages.append({"role": "user", "content": []})

    def _encode_image_to_base64(self, image_path: str) -> str:
        mime = "image/png"
        b64 = base64.b64encode(pathlib.Path(image_path).read_bytes()).decode("utf-8")
        return f"data:{mime};base64,{b64}"

    def _prepare_messages_for_llm(self) -> List[Dict]:
        messages = copy.deepcopy(self.messages)
        for msg in messages:
            msg.pop("reasoning", None)
            msg.pop("usage", None)
            msg.pop("response_id", None)
        for msg in messages:
            content = msg.get("content", [])
            for block in content:
                if isinstance(block, dict) and "image_url" in block:
                    image_path = block["image_url"].get("url")
                    block["image_url"]["url"] = self._encode_image_to_base64(image_path)
        return messages

    def _build_reasoning_payload(
        self,
        effort: str,
        reasoning_max_tokens: Optional[int],
    ) -> dict[str, object]:
        payload: dict[str, object] = {}
        if effort == "none":
            return payload
        if self.model == "google/gemini-3-flash-preview":
            payload["enabled"] = True
            if effort:
                payload["effort"] = effort
            return payload

        if reasoning_max_tokens is not None:
            payload["enabled"] = True
            payload["max_tokens"] = reasoning_max_tokens
        elif effort:
            payload["enabled"] = True
            payload["effort"] = effort
        return payload

    def _save_messages_json(self) -> None:
        if not self.save_messages_json_path:
            return
        text = json.dumps(self.messages, indent=2, ensure_ascii=False)
        pathlib.Path(self.save_messages_json_path).write_text(text, encoding="utf-8")

    def _start_api_call(self, started_at: str) -> dict[str, Any]:
        api_call: dict[str, Any] = {
            "attempt_index": self.request_attempts,
            "response_id": None,
            "model": self.model,
            "http_status": None,
            "accepted": False,
            "outcome": "started",
            "usage": None,
            "started_at": started_at,
            "finished_at": None,
            "duration_sec": None,
        }
        self.api_calls.append(api_call)
        self._save_usage_json()
        return api_call

    def _record_api_response(self, api_call: dict[str, Any], out: Any, *, http_status: int) -> None:
        response = out if isinstance(out, dict) else {}
        usage = response.get("usage")
        api_call.update(
            {
                "response_id": response.get("id"),
                "model": response.get("model", self.model),
                "http_status": http_status,
                "outcome": "received",
                "usage": copy.deepcopy(usage) if isinstance(usage, dict) else None,
            }
        )
        if isinstance(usage, dict):
            logger.info("LLM usage: %s", usage)
        self._save_usage_json()

    def _finish_api_call(
        self,
        api_call: dict[str, Any],
        request_start_perf: float,
        **updates: Any,
    ) -> None:
        api_call.update(updates)
        api_call["finished_at"] = _utc_now_iso()
        api_call["duration_sec"] = time.perf_counter() - request_start_perf
        self._save_usage_json()

    def _update_api_call(self, api_call: dict[str, Any], **updates: Any) -> None:
        api_call.update(updates)
        self._save_usage_json()

    @staticmethod
    def _sum_usage_number(usage_records: List[dict[str, Any]], key: str) -> int:
        return int(
            sum(
                value
                for usage in usage_records
                if isinstance((value := usage.get(key)), (int, float))
                and not isinstance(value, bool)
            )
        )

    def _save_usage_json(self) -> None:
        if not self.save_usage_json_path:
            return
        text = json.dumps(self.get_usage_report(), indent=2, ensure_ascii=False)
        pathlib.Path(self.save_usage_json_path).write_text(text, encoding="utf-8")

    def _load_messages_json(self, path: str) -> None:
        raw = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
        if not isinstance(raw, list):
            raise ValueError(f"Expected message list in {path}")
        self.messages = raw

    def _save_messages_markdown(self) -> None:
        if not self.save_messages_markdown_path:
            return

        base = os.path.dirname(os.path.abspath(self.save_messages_markdown_path))
        with open(self.save_messages_markdown_path, "w", encoding="utf-8") as handle:
            handle.write("# LLM Session Transcript\n\n")
            for message in self.messages:
                handle.write(f"## {str(message.get('role', '')).capitalize()}\n\n")
                if message.get("role") == "assistant":
                    reasoning = message.get("reasoning")
                    if isinstance(reasoning, str) and reasoning.strip():
                        handle.write("<details>\n<summary><b>Reasoning</b></summary>\n\n")
                        handle.write(reasoning.rstrip() + "\n\n")
                        handle.write("</details>\n\n")
                    elif message.get("reasoning_details") not in (None, [], {}):
                        handle.write("<details>\n<summary><b>Reasoning Details</b></summary>\n\n")
                        handle.write("```json\n")
                        handle.write(
                            json.dumps(message.get("reasoning_details"), indent=2, ensure_ascii=False)
                        )
                        handle.write("\n```\n\n")
                        handle.write("</details>\n\n")

                content = message.get("content", "")
                blocks = (
                    content
                    if isinstance(content, list)
                    else [{"type": "text", "text": content}]
                )
                for block in blocks:
                    if isinstance(block, dict) and block.get("type") == "image_url":
                        url = (block.get("image_url") or {}).get("url", "")
                        if os.path.exists(url):
                            rel = os.path.relpath(os.path.abspath(url), start=base)
                            url = urllib.parse.quote(rel)
                        handle.write(f"![image]({url})\n\n")
                    else:
                        text = block.get("text", "") if isinstance(block, dict) else str(block)
                        handle.write(text.rstrip() + "\n\n")
                handle.write("---\n\n")

    def _save_messages_if_needed(self) -> None:
        self._save_messages_json()
        self._save_messages_markdown()
