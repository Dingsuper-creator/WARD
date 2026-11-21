#!/usr/bin/env python3
"""Provider-neutral Chat Completions adapter for the WARD Judge protocol."""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request
from typing import Any


ALLOWED_REQUEST_PARAMETERS = {
    "temperature",
    "top_p",
    "seed",
    "max_tokens",
    "response_format",
    "thinking",
    "enable_thinking",
    "stream",
}


def main() -> None:
    envelope = json.load(sys.stdin)
    request = envelope["request"]
    response_schema = envelope["response_schema"]
    model = envelope["model"]
    credential_env = model["credential_env"]
    api_key = os.environ.get(credential_env)
    if not api_key:
        raise SystemExit(f"{credential_env} is not set")

    user_payload = {
        "request_id": request["request_id"],
        "judge_id": request["judge_id"],
        "items": request["items"],
    }
    parameters = model.get("request_parameters") or {}
    unknown = sorted(set(parameters) - ALLOWED_REQUEST_PARAMETERS)
    if unknown:
        raise ValueError(f"unsupported request parameters: {unknown}")
    api_payload: dict[str, Any] = {
        "model": model["requested_model"],
        "messages": [
            {"role": "system", "content": request["system_prompt"]},
            {
                "role": "system",
                "content": _output_contract(response_schema, request),
            },
            {
                "role": "user",
                "content": json.dumps(user_payload, ensure_ascii=False, separators=(",", ":")),
            },
        ],
        **{key: value for key, value in parameters.items() if value is not None},
    }
    api_payload["stream"] = False

    http_request = urllib.request.Request(
        model["endpoint"],
        data=json.dumps(api_payload, ensure_ascii=False).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    started = time.monotonic()
    try:
        with urllib.request.urlopen(
            http_request,
            timeout=float(model.get("http_timeout_seconds", 600)),
        ) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        print(f"{model['provider']} HTTP {exc.code}: {detail}", file=sys.stderr)
        raise SystemExit(2) from exc
    except urllib.error.URLError as exc:
        print(f"{model['provider']} transport error: {exc}", file=sys.stderr)
        raise SystemExit(3) from exc

    latency_ms = round((time.monotonic() - started) * 1000, 3)
    provider_response = json.loads(raw)
    visible_text, refusal, finish_reason = _visible_output(provider_response)
    judge_response = _parse_visible_judge_response(visible_text)
    output = {
        "judge_response": judge_response,
        "provider_metadata": {
            "provider": model["provider"],
            "requested_model": model["requested_model"],
            "effective_model": provider_response.get("model") or model["requested_model"],
            "response_id": provider_response.get("id"),
            "system_fingerprint": provider_response.get("system_fingerprint"),
            "sampling_parameters": {
                key: value for key, value in parameters.items() if key != "response_format"
            },
            "token_usage": provider_response.get("usage") or {},
            "finish_reason": finish_reason,
            "refusal": refusal,
            "latency_ms": latency_ms,
        },
    }
    json.dump(output, sys.stdout, ensure_ascii=False, separators=(",", ":"))
    sys.stdout.write("\n")


def _parse_visible_judge_response(visible_text: str) -> Any:
    try:
        return json.loads(visible_text)
    except json.JSONDecodeError:
        return visible_text


def _output_contract(response_schema: dict[str, Any], request: dict[str, Any]) -> str:
    item_ids = [item["item_id"] for item in request["items"]]
    example = {
        "schema_version": "ward-judge-response-1.0",
        "request_id": request["request_id"],
        "judge_id": request["judge_id"],
        "results": [
            {
                "item_id": item_id,
                "verdict": "equivalent|partial|contradiction|unrelated|abstain",
                "confidence": 0.0,
                "reason_codes": [],
            }
            for item_id in item_ids
        ],
    }
    return (
        "OUTPUT CONTRACT (instructions, not clinical data): Return exactly one JSON batch object. "
        "Do not return a single item, an items array, a judgments wrapper, or a judge_response wrapper. "
        "The results array must contain exactly one result for every expected item_id, in the exact order shown. "
        "Replace each example verdict/confidence/reason_codes with the actual judgment.\n"
        f"Expected item_id order: {json.dumps(item_ids, ensure_ascii=False)}\n"
        f"Exact envelope example: {json.dumps(example, ensure_ascii=False, separators=(',', ':'))}\n"
        f"Authoritative JSON Schema: {json.dumps(response_schema, ensure_ascii=False, separators=(',', ':'))}"
    )


def _visible_output(response: dict[str, Any]) -> tuple[str, str | bool | None, str | None]:
    choices = response.get("choices")
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
        raise ValueError("Chat Completions response must contain exactly one choice")
    choice = choices[0]
    message = choice.get("message")
    if not isinstance(message, dict):
        raise ValueError("Chat Completions response has no assistant message")
    refusal = message.get("refusal")
    content = message.get("content")
    if refusal not in (None, False, ""):
        return "", refusal, choice.get("finish_reason")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("Chat Completions response has no visible assistant content")
    return content, None, choice.get("finish_reason")


if __name__ == "__main__":
    main()
