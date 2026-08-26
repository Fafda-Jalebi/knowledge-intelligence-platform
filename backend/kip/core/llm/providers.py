"""Hosted and local LLM backends.

Four providers, one file, because each is a thin adapter: build a JSON body, POST
it with :mod:`kip.core.http`, pull the text and the usage out of the reply. The
interesting code is the *parsing*, so payload construction and response reading
are module-level pure functions with doctests. That is deliberate -- it means the
part most likely to break when a provider changes its response shape is covered
by tests that run offline, with no key and no network.

Imports of provider SDKs are absent entirely: every one of these speaks plain
JSON over HTTPS, so there is no dependency to install and nothing to keep in step
with a vendor's release cycle.

Credentials appear in exactly one place per client -- the request headers -- and
are never interpolated into an error message, a log line, or a URL. Gemini is
called with the ``x-goog-api-key`` header rather than the documented ``?key=``
query parameter for that reason: a key in a URL ends up in proxy and server logs
that nobody audits.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from kip.core.http import HttpError, post_json
from kip.core.llm.base import (
    LlmClient,
    LlmError,
    LlmResponse,
    Message,
    Usage,
    fold_system_message,
)

ANTHROPIC_VERSION = "2023-06-01"
ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta"


def _require_key(key: str, provider: str, env_var: str) -> str:
    """Fail with an instruction rather than a 401 from the provider."""
    cleaned = str(key or "").strip()
    if not cleaned:
        raise LlmError(
            f"LLM_PROVIDER={provider} requires {env_var} to be set. Add it to "
            "your .env file, or set LLM_PROVIDER=extractive to run without any "
            "API key."
        )
    return cleaned


def _text_or_error(text: str, provider: str, raw: Any) -> str:
    """Guard against an empty completion being presented as an answer.

    A provider can return a well-formed response with no text -- a safety filter,
    a stop sequence that fired immediately -- and passing that through would show
    the user a blank answer with citations attached to nothing.
    """
    if str(text or "").strip():
        return text
    reason = ""
    if isinstance(raw, Mapping):
        for key in ("finishReason", "stop_reason", "done_reason"):
            if raw.get(key):
                reason = f" (reason: {raw[key]})"
                break
    raise LlmError(
        f"{provider} returned an empty completion{reason}. The request "
        "succeeded but produced no text."
    )


# --------------------------------------------------------------------------- #
# OpenAI-compatible chat completions
# --------------------------------------------------------------------------- #


def openai_payload(
    messages: Sequence[Message],
    *,
    model: str,
    temperature: float,
    max_output_tokens: int,
) -> dict[str, Any]:
    """Build a chat-completions body.

    >>> body = openai_payload([Message("system", "Be exact."),
    ...                        Message("user", "Why?")],
    ...                       model="gpt-4o-mini", temperature=0.1,
    ...                       max_output_tokens=500)
    >>> body["model"], body["max_tokens"], body["temperature"]
    ('gpt-4o-mini', 500, 0.1)
    >>> body["messages"]
    [{'role': 'system', 'content': 'Be exact.'}, {'role': 'user', 'content': 'Why?'}]
    """
    return {
        "model": model,
        "messages": [message.to_dict() for message in messages],
        "temperature": float(temperature),
        "max_tokens": int(max_output_tokens),
    }


def parse_openai(raw: Any) -> tuple[str, str | None, Usage]:
    """Read text, finish reason and usage from a chat-completions reply.

    >>> reply = {"choices": [{"message": {"content": "60 C [1]."},
    ...                       "finish_reason": "stop"}],
    ...          "usage": {"prompt_tokens": 812, "completion_tokens": 24}}
    >>> text, reason, usage = parse_openai(reply)
    >>> text, reason, usage.total_tokens, usage.estimated
    ('60 C [1].', 'stop', 836, False)

    Usage is optional; its absence is recorded as an estimate rather than as zero:

    >>> _, _, usage = parse_openai({"choices": [{"message": {"content": "hi"}}]})
    >>> usage.total_tokens, usage.estimated
    (0, True)

    A reply with no choices is an error, not an empty answer:

    >>> parse_openai({"choices": []})
    Traceback (most recent call last):
        ...
    kip.core.llm.base.LlmError: OpenAI returned no choices in its response.
    """
    if not isinstance(raw, Mapping):
        raise LlmError("OpenAI returned a response that was not a JSON object.")
    choices = raw.get("choices") or []
    if not choices:
        raise LlmError("OpenAI returned no choices in its response.")
    first = choices[0] if isinstance(choices[0], Mapping) else {}
    message = first.get("message") if isinstance(first.get("message"), Mapping) else {}
    text = str((message or {}).get("content") or "")
    reason = first.get("finish_reason")
    reported = raw.get("usage") if isinstance(raw.get("usage"), Mapping) else None
    if reported:
        usage = Usage(
            prompt_tokens=int(reported.get("prompt_tokens") or 0),
            completion_tokens=int(reported.get("completion_tokens") or 0),
        )
    else:
        usage = Usage(estimated=True)
    return text, (str(reason) if reason else None), usage


class OpenAIClient(LlmClient):
    """Chat completions against OpenAI or any compatible endpoint.

    ``OPENAI_BASE_URL`` makes this work with Azure OpenAI, Together, Groq,
    OpenRouter, vLLM and LM Studio without another adapter -- they all implement
    the same route.
    """

    name = "openai"

    def __init__(
        self,
        *,
        model: str = "gpt-4o-mini",
        api_key: str = "",
        base_url: str = "https://api.openai.com/v1",
        **kwargs: Any,
    ) -> None:
        super().__init__(model=model, **kwargs)
        self._key = _require_key(api_key, "openai", "OPENAI_API_KEY")
        self._url = f"{str(base_url).rstrip('/')}/chat/completions"

    def _generate(
        self,
        messages: Sequence[Message],
        *,
        passages: Sequence[tuple[int, str]],
        temperature: float,
        max_output_tokens: int,
    ) -> LlmResponse:
        payload = openai_payload(
            messages,
            model=self.model,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )
        try:
            raw = post_json(
                self._url,
                payload,
                headers={"Authorization": f"Bearer {self._key}"},
                timeout=self.timeout,
            )
        except HttpError as exc:
            raise LlmError(f"OpenAI request failed: {exc}") from exc
        text, reason, usage = parse_openai(raw)
        return LlmResponse(
            text=_text_or_error(text, "OpenAI", raw),
            provider=self.name,
            model=self.model,
            usage=usage,
            finish_reason=reason,
        )


# --------------------------------------------------------------------------- #
# Anthropic messages
# --------------------------------------------------------------------------- #


def anthropic_payload(
    messages: Sequence[Message],
    *,
    model: str,
    temperature: float,
    max_output_tokens: int,
) -> dict[str, Any]:
    """Build a messages-API body, lifting system turns into the ``system`` field.

    >>> body = anthropic_payload([Message("system", "Cite sources."),
    ...                           Message("user", "Why?")],
    ...                          model="claude-sonnet-4-5", temperature=0.1,
    ...                          max_output_tokens=400)
    >>> body["system"], body["max_tokens"]
    ('Cite sources.', 400)
    >>> body["messages"]
    [{'role': 'user', 'content': 'Why?'}]

    The API requires the conversation to begin with a user turn, so a leading
    assistant message is dropped rather than sent and rejected:

    >>> anthropic_payload([Message("assistant", "Hi"), Message("user", "Q")],
    ...                   model="m", temperature=0.0,
    ...                   max_output_tokens=10)["messages"]
    [{'role': 'user', 'content': 'Q'}]
    """
    system, rest = fold_system_message(messages)
    turns = [message.to_dict() for message in rest]
    while turns and turns[0]["role"] != "user":
        turns.pop(0)
    body: dict[str, Any] = {
        "model": model,
        "messages": turns,
        "max_tokens": int(max_output_tokens),
        "temperature": float(temperature),
    }
    if system:
        body["system"] = system
    return body


def parse_anthropic(raw: Any) -> tuple[str, str | None, Usage]:
    """Read a messages-API reply.

    Text blocks are concatenated; a response may legitimately contain several.

    >>> reply = {"content": [{"type": "text", "text": "Dried at 60 C [1]."},
    ...                      {"type": "text", "text": " Sealed at 0.6 aw [2]."}],
    ...          "stop_reason": "end_turn",
    ...          "usage": {"input_tokens": 900, "output_tokens": 30}}
    >>> text, reason, usage = parse_anthropic(reply)
    >>> text
    'Dried at 60 C [1]. Sealed at 0.6 aw [2].'
    >>> reason, usage.prompt_tokens, usage.completion_tokens
    ('end_turn', 900, 30)

    Non-text blocks are ignored rather than stringified into the answer:

    >>> parse_anthropic({"content": [{"type": "thinking", "thinking": "hmm"},
    ...                              {"type": "text", "text": "A."}]})[0]
    'A.'
    """
    if not isinstance(raw, Mapping):
        raise LlmError("Anthropic returned a response that was not a JSON object.")
    blocks = raw.get("content") or []
    parts = [
        str(block.get("text") or "")
        for block in blocks
        if isinstance(block, Mapping) and block.get("type") == "text"
    ]
    reported = raw.get("usage") if isinstance(raw.get("usage"), Mapping) else None
    usage = (
        Usage(
            prompt_tokens=int(reported.get("input_tokens") or 0),
            completion_tokens=int(reported.get("output_tokens") or 0),
        )
        if reported
        else Usage(estimated=True)
    )
    reason = raw.get("stop_reason")
    return "".join(parts), (str(reason) if reason else None), usage


class AnthropicClient(LlmClient):
    """Anthropic messages API."""

    name = "anthropic"

    def __init__(
        self, *, model: str = "claude-sonnet-4-5", api_key: str = "", **kwargs: Any
    ) -> None:
        super().__init__(model=model, **kwargs)
        self._key = _require_key(api_key, "anthropic", "ANTHROPIC_API_KEY")

    def _generate(
        self,
        messages: Sequence[Message],
        *,
        passages: Sequence[tuple[int, str]],
        temperature: float,
        max_output_tokens: int,
    ) -> LlmResponse:
        payload = anthropic_payload(
            messages,
            model=self.model,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )
        try:
            raw = post_json(
                ANTHROPIC_URL,
                payload,
                headers={
                    "x-api-key": self._key,
                    "anthropic-version": ANTHROPIC_VERSION,
                },
                timeout=self.timeout,
            )
        except HttpError as exc:
            raise LlmError(f"Anthropic request failed: {exc}") from exc
        text, reason, usage = parse_anthropic(raw)
        return LlmResponse(
            text=_text_or_error(text, "Anthropic", raw),
            provider=self.name,
            model=self.model,
            usage=usage,
            finish_reason=reason,
        )


# --------------------------------------------------------------------------- #
# Google Gemini
# --------------------------------------------------------------------------- #


def gemini_payload(
    messages: Sequence[Message],
    *,
    temperature: float,
    max_output_tokens: int,
) -> dict[str, Any]:
    """Build a ``generateContent`` body.

    Gemini names the assistant role ``model`` and wraps system instructions in
    their own field.

    >>> body = gemini_payload([Message("system", "Cite."), Message("user", "Q"),
    ...                        Message("assistant", "A"), Message("user", "Q2")],
    ...                       temperature=0.2, max_output_tokens=256)
    >>> body["systemInstruction"]["parts"][0]["text"]
    'Cite.'
    >>> [turn["role"] for turn in body["contents"]]
    ['user', 'model', 'user']
    >>> body["generationConfig"]
    {'temperature': 0.2, 'maxOutputTokens': 256}
    """
    system, rest = fold_system_message(messages)
    body: dict[str, Any] = {
        "contents": [
            {
                "role": "model" if message.role == "assistant" else "user",
                "parts": [{"text": message.content}],
            }
            for message in rest
        ],
        "generationConfig": {
            "temperature": float(temperature),
            "maxOutputTokens": int(max_output_tokens),
        },
    }
    if system:
        body["systemInstruction"] = {"parts": [{"text": system}]}
    return body


def parse_gemini(raw: Any) -> tuple[str, str | None, Usage]:
    """Read a ``generateContent`` reply.

    >>> reply = {"candidates": [{"content": {"parts": [{"text": "60 C [1]."}]},
    ...                          "finishReason": "STOP"}],
    ...          "usageMetadata": {"promptTokenCount": 700,
    ...                            "candidatesTokenCount": 18}}
    >>> parse_gemini(reply)[0:2]
    ('60 C [1].', 'STOP')
    >>> parse_gemini(reply)[2].total_tokens
    718

    A prompt blocked before generation carries no candidates, and the reason it
    was blocked is surfaced rather than swallowed:

    >>> parse_gemini({"promptFeedback": {"blockReason": "SAFETY"}})
    Traceback (most recent call last):
        ...
    kip.core.llm.base.LlmError: Gemini returned no candidates (blocked: SAFETY).
    """
    if not isinstance(raw, Mapping):
        raise LlmError("Gemini returned a response that was not a JSON object.")
    candidates = raw.get("candidates") or []
    if not candidates:
        feedback = raw.get("promptFeedback")
        reason = ""
        if isinstance(feedback, Mapping) and feedback.get("blockReason"):
            reason = f" (blocked: {feedback['blockReason']})"
        raise LlmError(f"Gemini returned no candidates{reason}.")
    first = candidates[0] if isinstance(candidates[0], Mapping) else {}
    content = first.get("content") if isinstance(first.get("content"), Mapping) else {}
    parts = (content or {}).get("parts") or []
    text = "".join(
        str(part.get("text") or "") for part in parts if isinstance(part, Mapping)
    )
    reported = (
        raw.get("usageMetadata") if isinstance(raw.get("usageMetadata"), Mapping) else None
    )
    usage = (
        Usage(
            prompt_tokens=int(reported.get("promptTokenCount") or 0),
            completion_tokens=int(reported.get("candidatesTokenCount") or 0),
        )
        if reported
        else Usage(estimated=True)
    )
    reason = first.get("finishReason")
    return text, (str(reason) if reason else None), usage


class GeminiClient(LlmClient):
    """Google Gemini ``generateContent``."""

    name = "gemini"

    def __init__(
        self,
        *,
        model: str = "gemini-2.0-flash",
        api_key: str = "",
        base_url: str = GEMINI_BASE,
        **kwargs: Any,
    ) -> None:
        super().__init__(model=model, **kwargs)
        self._key = _require_key(api_key, "gemini", "GEMINI_API_KEY")
        self._base = str(base_url).rstrip("/")

    def _generate(
        self,
        messages: Sequence[Message],
        *,
        passages: Sequence[tuple[int, str]],
        temperature: float,
        max_output_tokens: int,
    ) -> LlmResponse:
        url = f"{self._base}/models/{self.model}:generateContent"
        payload = gemini_payload(
            messages, temperature=temperature, max_output_tokens=max_output_tokens
        )
        try:
            raw = post_json(
                url,
                payload,
                # Header rather than ?key=, so the credential stays out of URLs
                # and therefore out of every log that records one.
                headers={"x-goog-api-key": self._key},
                timeout=self.timeout,
            )
        except HttpError as exc:
            raise LlmError(f"Gemini request failed: {exc}") from exc
        text, reason, usage = parse_gemini(raw)
        return LlmResponse(
            text=_text_or_error(text, "Gemini", raw),
            provider=self.name,
            model=self.model,
            usage=usage,
            finish_reason=reason,
        )


# --------------------------------------------------------------------------- #
# Ollama (local)
# --------------------------------------------------------------------------- #


def ollama_payload(
    messages: Sequence[Message],
    *,
    model: str,
    temperature: float,
    max_output_tokens: int,
) -> dict[str, Any]:
    """Build an ``/api/chat`` body.

    ``stream`` is explicitly false: the default is streaming, which would return
    newline-delimited JSON that :func:`kip.core.http.post_json` cannot decode.

    >>> body = ollama_payload([Message("user", "Q")], model="llama3.1",
    ...                       temperature=0.1, max_output_tokens=300)
    >>> body["stream"], body["options"]
    (False, {'temperature': 0.1, 'num_predict': 300})
    """
    return {
        "model": model,
        "messages": [message.to_dict() for message in messages],
        "stream": False,
        "options": {
            "temperature": float(temperature),
            "num_predict": int(max_output_tokens),
        },
    }


def parse_ollama(raw: Any) -> tuple[str, str | None, Usage]:
    """Read an ``/api/chat`` reply.

    >>> reply = {"message": {"role": "assistant", "content": "60 C [1]."},
    ...          "done_reason": "stop",
    ...          "prompt_eval_count": 640, "eval_count": 21}
    >>> text, reason, usage = parse_ollama(reply)
    >>> text, reason, usage.total_tokens, usage.estimated
    ('60 C [1].', 'stop', 661, False)
    """
    if not isinstance(raw, Mapping):
        raise LlmError("Ollama returned a response that was not a JSON object.")
    message = raw.get("message") if isinstance(raw.get("message"), Mapping) else {}
    text = str((message or {}).get("content") or "")
    prompt_tokens = int(raw.get("prompt_eval_count") or 0)
    completion_tokens = int(raw.get("eval_count") or 0)
    usage = (
        Usage(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
        if (prompt_tokens or completion_tokens)
        else Usage(estimated=True)
    )
    reason = raw.get("done_reason")
    return text, (str(reason) if reason else None), usage


class OllamaClient(LlmClient):
    """A model served by a local Ollama instance -- no key, no data egress."""

    name = "ollama"

    def __init__(
        self,
        *,
        model: str = "llama3.1",
        base_url: str = "http://localhost:11434",
        **kwargs: Any,
    ) -> None:
        super().__init__(model=model, **kwargs)
        self._url = f"{str(base_url).rstrip('/')}/api/chat"

    def _generate(
        self,
        messages: Sequence[Message],
        *,
        passages: Sequence[tuple[int, str]],
        temperature: float,
        max_output_tokens: int,
    ) -> LlmResponse:
        payload = ollama_payload(
            messages,
            model=self.model,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )
        try:
            raw = post_json(self._url, payload, timeout=self.timeout)
        except HttpError as exc:
            raise LlmError(
                f"Ollama request failed: {exc} Is the server running? "
                "`ollama serve` starts it, and `ollama pull "
                f"{self.model}` fetches the model."
            ) from exc
        text, reason, usage = parse_ollama(raw)
        return LlmResponse(
            text=_text_or_error(text, "Ollama", raw),
            provider=self.name,
            model=self.model,
            usage=usage,
            finish_reason=reason,
        )


__all__ = [
    "AnthropicClient",
    "GeminiClient",
    "OllamaClient",
    "OpenAIClient",
    "anthropic_payload",
    "gemini_payload",
    "ollama_payload",
    "openai_payload",
    "parse_anthropic",
    "parse_gemini",
    "parse_ollama",
    "parse_openai",
]
