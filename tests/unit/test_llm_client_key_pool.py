"""LlmClient + KeyPool integration: round-robin, eviction, all-failed."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Mapping

import pytest

from transoria.llm.client import (
    ChatRequest,
    LlmClient,
    LlmRequestError,
    TransportResult,
)
from transoria.llm.config import ModelConfig, ProviderFormat
from transoria.runtime.key_pool import KeyPool


@dataclass
class RecordingTransport:
    queue: list[TransportResult]
    headers_seen: list[Mapping[str, str]] = field(default_factory=list)

    async def execute(
        self,
        url: str,
        headers: Mapping[str, str],
        payload: Mapping[str, object],
        timeout: float,
    ) -> TransportResult:
        self.headers_seen.append(dict(headers))
        return self.queue.pop(0)


def _model(*keys: str) -> ModelConfig:
    return ModelConfig(
        id="m",
        display_name="m",
        provider_format=ProviderFormat.OPENAI,
        base_url="https://example/api",
        model_id="x",
        api_keys=tuple(keys),
    )


def _ok(content: str = "hi") -> TransportResult:
    return TransportResult(
        status_code=200,
        body={
            "choices": [{"message": {"role": "assistant", "content": content}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        },
    )


def _http(status: int, body: dict | None = None) -> TransportResult:
    return TransportResult(status_code=status, body=body or {"error": "x"})


def test_pool_round_robins_keys_across_calls() -> None:
    transport = RecordingTransport(queue=[_ok(), _ok(), _ok()])
    client = LlmClient(transport=transport)
    pool = KeyPool(("ka", "kb"))
    request = ChatRequest(
        model=_model("ka", "kb"),
        system_prompt="",
        user_prompt="ping",
        key_pool=pool,
    )

    asyncio.run(client.chat(request))
    asyncio.run(client.chat(request))
    asyncio.run(client.chat(request))

    auths = [h["Authorization"] for h in transport.headers_seen]
    assert auths == ["Bearer ka", "Bearer kb", "Bearer ka"]


def test_pool_evicts_key_on_401_and_retries_with_next() -> None:
    transport = RecordingTransport(queue=[_http(401), _ok()])
    client = LlmClient(transport=transport)
    pool = KeyPool(("ka", "kb"))
    request = ChatRequest(
        model=_model("ka", "kb"),
        system_prompt="",
        user_prompt="ping",
        key_pool=pool,
    )

    asyncio.run(client.chat(request))

    auths = [h["Authorization"] for h in transport.headers_seen]
    assert auths == ["Bearer ka", "Bearer kb"]
    assert "ka" in pool.dead_keys
    assert pool.alive_count == 1


def test_pool_does_not_evict_key_on_429_but_rotates() -> None:
    transport = RecordingTransport(queue=[_http(429), _ok()])
    client = LlmClient(transport=transport)
    pool = KeyPool(("ka", "kb"))
    request = ChatRequest(
        model=_model("ka", "kb"),
        system_prompt="",
        user_prompt="ping",
        key_pool=pool,
    )

    asyncio.run(client.chat(request))

    auths = [h["Authorization"] for h in transport.headers_seen]
    assert auths == ["Bearer ka", "Bearer kb"]
    assert pool.dead_keys == frozenset()


def test_pool_keeps_polling_after_account_level_429() -> None:
    transport = RecordingTransport(
        queue=[
            _http(
                429,
                {
                    "error": {
                        "code": "SetLimitExceeded",
                        "message": "model service has been paused by Safe Experience Mode",
                    }
                },
            ),
            _ok(),
        ]
    )
    client = LlmClient(transport=transport)
    pool = KeyPool(("ka", "kb"))
    request = ChatRequest(
        model=_model("ka", "kb"),
        system_prompt="",
        user_prompt="ping",
        key_pool=pool,
    )

    response = asyncio.run(client.chat(request))

    assert response.content == "hi"
    assert [h["Authorization"] for h in transport.headers_seen] == [
        "Bearer ka",
        "Bearer kb",
    ]
    assert pool.dead_keys == frozenset()


def test_pool_raises_all_keys_failed_when_every_key_dead() -> None:
    transport = RecordingTransport(queue=[_http(403), _http(403)])
    client = LlmClient(transport=transport)
    pool = KeyPool(("ka", "kb"))
    request = ChatRequest(
        model=_model("ka", "kb"),
        system_prompt="",
        user_prompt="ping",
        key_pool=pool,
    )

    with pytest.raises(LlmRequestError) as caught:
        asyncio.run(client.chat(request))

    assert caught.value.code == "llm.all_keys_failed"
    assert pool.dead_keys == frozenset({"ka", "kb"})
