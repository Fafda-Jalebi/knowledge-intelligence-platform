"""LLM client abstraction.

``LLM_PROVIDER`` selects a backend and nothing downstream knows which one is
active. The default is ``extractive``, a local answerer that needs no API key --
see :mod:`kip.core.llm.extractive` for why that default is a deliberate
engineering choice rather than a placeholder.

Two things about this interface are unusual enough to explain.

**The question and the passages travel beside the prompt, not only inside it.**
:meth:`LlmClient.complete` accepts an optional ``passages`` argument carrying
``(marker, text)`` pairs, and an optional ``question``. A hosted model only needs
the rendered prompt; an extractive answerer needs both *as data*, because it
selects sentences by matching them against the question instead of continuing a
token sequence. Recovering that structure by re-parsing the prompt would be
fragile in exactly the place the platform cares most about -- which passage a
citation points at -- and it fails in a way that is easy to miss: the rendered
prompt contains the passages as well as the question, so a backend that matched
against the whole thing would find its signal diluted by every additional
passage retrieved, and would answer *worse* as retrieval improved. Plain values
are used rather than a shared dataclass so this module stays independent of
:mod:`kip.core.rag`.

**Prompts and completions are never logged.**
Only shapes and counts are. Prompt text contains passages from the user's
documents, and log aggregation is a data-egress path nobody audits, so document
content stays out of it. API keys are held by the provider modules and never
reach a log line or an exception message.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from kip.core.text import count_tokens

#: Roles accepted in a message list. Providers that lack a native system role
#: fold it into the first user turn; see :mod:`kip.core.llm.providers`.
ROLES = ("system", "user", "assistant")

#: A generation that stops for this reason was cut off by the token limit rather
#: than finishing its thought, which the grounding layer needs to know: a
#: truncated answer can lose the citation markers that would have justified it.
TRUNCATED_REASONS = frozenset({"length", "max_tokens", "MAX_TOKENS", "token_limit"})


class LlmError(RuntimeError):
    """A generation backend was unavailable, misconfigured, or failed."""


@dataclass(frozen=True, slots=True)
class Message:
    """One turn in a chat exchange.

    >>> Message("user", "  What temperature?  ").content
    'What temperature?'
    >>> Message("wizard", "hi")
    Traceback (most recent call last):
        ...
    kip.core.llm.base.LlmError: Unknown message role 'wizard'. Expected one of: system, user, assistant.
    """

    role: str
    content: str

    def __post_init__(self) -> None:
        role = str(self.role or "").strip().lower()
        if role not in ROLES:
            raise LlmError(
                f"Unknown message role {str(self.role)!r}. Expected one of: "
                + ", ".join(ROLES)
                + "."
            )
        object.__setattr__(self, "role", role)
        object.__setattr__(self, "content", str(self.content or "").strip())

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass(frozen=True, slots=True)
class Usage:
    """Token accounting for one call.

    ``estimated`` is part of the data, not a footnote. Local backends and some
    hosted responses do not report usage, and a cost figure derived from an
    estimate must not be presented as though it were metered by the provider.

    >>> Usage(120, 30).total_tokens
    150
    >>> Usage(120, 30, estimated=True).to_dict()["estimated"]
    True
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0
    estimated: bool = False

    @property
    def total_tokens(self) -> int:
        return int(self.prompt_tokens) + int(self.completion_tokens)

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt_tokens": int(self.prompt_tokens),
            "completion_tokens": int(self.completion_tokens),
            "total_tokens": self.total_tokens,
            "estimated": bool(self.estimated),
        }


@dataclass(frozen=True, slots=True)
class LlmResponse:
    """A completion plus everything needed to audit how it was produced.

    >>> r = LlmResponse("Dried at 60 C [1].", provider="extractive", model="m")
    >>> r.truncated
    False
    >>> LlmResponse("cut off mid-", provider="openai", model="m",
    ...             finish_reason="length").truncated
    True
    """

    text: str
    provider: str
    model: str
    usage: Usage = field(default_factory=Usage)
    latency_ms: float = 0.0
    finish_reason: str | None = None
    #: Backend-specific extras (e.g. how many sentences an extractor selected).
    meta: Mapping[str, Any] = field(default_factory=dict)

    @property
    def truncated(self) -> bool:
        """True when the provider stopped at the output token limit."""
        return str(self.finish_reason or "") in TRUNCATED_REASONS

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "provider": self.provider,
            "model": self.model,
            "usage": self.usage.to_dict(),
            "latency_ms": round(float(self.latency_ms), 2),
            "finish_reason": self.finish_reason,
            "truncated": self.truncated,
            "meta": dict(self.meta),
        }


class LlmClient(ABC):
    """Base class handling validation, timing and token accounting.

    Subclasses implement :meth:`_generate` only. Everything the rest of the
    platform relies on -- a non-empty message list, resolved generation
    parameters, measured latency, usage that is filled in even when the provider
    omits it -- is enforced here so no backend can get it subtly wrong.
    """

    name: str = "abstract"

    #: True when the backend can only quote its input verbatim, which makes its
    #: output grounded by construction. The grounding layer reports this rather
    #: than inferring it, and the evaluation harness uses it to explain why the
    #: extractive baseline scores 1.0 on faithfulness.
    extractive: bool = False

    def __init__(
        self,
        *,
        model: str,
        temperature: float = 0.1,
        max_output_tokens: int = 900,
        timeout: float = 60.0,
    ) -> None:
        self.model = str(model or "").strip()
        self.temperature = max(0.0, float(temperature))
        self.max_output_tokens = max(1, int(max_output_tokens))
        self.timeout = max(1.0, float(timeout))

    # -- subclass contract -------------------------------------------------- #

    @abstractmethod
    def _generate(
        self,
        messages: Sequence[Message],
        *,
        passages: Sequence[tuple[int, str]],
        temperature: float,
        max_output_tokens: int,
    ) -> LlmResponse:
        """Produce a completion. Raise :class:`LlmError` on failure."""

    def warm_up(self) -> None:
        """Optional hook: load weights or verify credentials up front."""

    # -- public API --------------------------------------------------------- #

    def complete(
        self,
        messages: Sequence[Message],
        *,
        passages: Sequence[tuple[int, str]] = (),
        temperature: float | None = None,
        max_output_tokens: int | None = None,
    ) -> LlmResponse:
        """Generate a completion for ``messages``.

        ``passages`` carries the numbered context as data for backends that
        select from it rather than generating; hosted providers ignore it.
        """
        turns = [message for message in messages if message.content]
        if not turns:
            raise LlmError(
                "Refusing to call the model with an empty prompt. This is a bug "
                "in the caller: the context builder should have produced at "
                "least a question."
            )

        resolved_temperature = (
            self.temperature if temperature is None else max(0.0, float(temperature))
        )
        resolved_limit = (
            self.max_output_tokens
            if max_output_tokens is None
            else max(1, int(max_output_tokens))
        )

        started = time.perf_counter()
        response = self._generate(
            turns,
            passages=tuple((int(marker), str(text)) for marker, text in passages),
            temperature=resolved_temperature,
            max_output_tokens=resolved_limit,
        )
        latency = (time.perf_counter() - started) * 1000.0

        usage = response.usage
        if usage.total_tokens == 0:
            # The provider reported nothing, so estimate and label it as such.
            usage = Usage(
                prompt_tokens=sum(count_tokens(turn.content) for turn in turns),
                completion_tokens=count_tokens(response.text),
                estimated=True,
            )

        return LlmResponse(
            text=response.text,
            provider=response.provider or self.name,
            model=response.model or self.model,
            usage=usage,
            # Prefer the measured wall time: a backend's own figure, when it
            # reports one, excludes connection setup and queueing.
            latency_ms=latency,
            finish_reason=response.finish_reason,
            meta=response.meta,
        )

    def describe(self) -> dict[str, Any]:
        """Active configuration, for the Settings screen and ``/api/health``."""
        return {
            "provider": self.name,
            "model": self.model,
            "temperature": self.temperature,
            "max_output_tokens": self.max_output_tokens,
            "extractive": self.extractive,
        }

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<{type(self).__name__} model={self.model!r}>"


def render_prompt(messages: Sequence[Message]) -> str:
    """Flatten a message list for providers with a single text input.

    >>> render_prompt([Message("system", "Be exact."), Message("user", "Why?")])
    'Be exact.\\n\\nWhy?'
    """
    return "\n\n".join(message.content for message in messages if message.content)


def fold_system_message(messages: Sequence[Message]) -> tuple[str, list[Message]]:
    """Split system instructions from the conversation.

    Several providers take system instructions in a dedicated field rather than
    as a turn, so this returns them separately. Multiple system messages are
    joined instead of dropped -- silently discarding an instruction is how a
    "cite your sources" rule goes missing.

    >>> system, rest = fold_system_message([
    ...     Message("system", "Cite sources."),
    ...     Message("user", "Hello"),
    ...     Message("system", "Be brief."),
    ... ])
    >>> system
    'Cite sources.\\n\\nBe brief.'
    >>> [m.role for m in rest]
    ['user']
    """
    system = [message.content for message in messages if message.role == "system"]
    rest = [message for message in messages if message.role != "system"]
    return "\n\n".join(system), rest


__all__ = [
    "LlmClient",
    "LlmError",
    "LlmResponse",
    "Message",
    "ROLES",
    "Usage",
    "fold_system_message",
    "render_prompt",
]
