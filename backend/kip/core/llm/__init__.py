"""LLM provider registry.

``LLM_PROVIDER`` selects a backend; nothing else in the platform knows which one
is active. Entries are *factories*, so a hosted client -- and its credential
check -- is only constructed when actually selected.

The default is ``extractive``: local, deterministic, no API key. See
:mod:`kip.core.llm.extractive` for why that is a considered default rather than a
stand-in.

This module is also where the reranker's missing dependency is supplied.
:mod:`kip.core.rerank.llm` deliberately does not import the LLM layer -- it takes
an injected scoring callable -- so :func:`make_score_fn` builds that callable
here, keeping the arrow pointing one way: ``rerank`` never depends on ``llm``.
"""

from __future__ import annotations

from typing import Any, Callable

from kip.core.llm.base import (
    LlmClient,
    LlmError,
    LlmResponse,
    Message,
    Usage,
    fold_system_message,
    render_prompt,
)
from kip.core.llm.extractive import ExtractiveClient

__all__ = [
    "ExtractiveClient",
    "LLM_PROVIDERS",
    "LLM_PROVIDER_NOTES",
    "LlmClient",
    "LlmError",
    "LlmResponse",
    "Message",
    "Usage",
    "describe_llm_providers",
    "fold_system_message",
    "get_llm_client",
    "make_score_fn",
    "render_prompt",
]


def _extractive(**kwargs: Any) -> LlmClient:
    from kip.core.llm.extractive import DEFAULT_MODEL

    return ExtractiveClient(
        model=str(kwargs.get("model") or DEFAULT_MODEL),
        max_output_tokens=int(kwargs.get("max_output_tokens") or 900),
    )


def _openai(**kwargs: Any) -> LlmClient:
    from kip.core.llm.providers import OpenAIClient

    return OpenAIClient(
        model=str(kwargs.get("model") or "gpt-4o-mini"),
        api_key=str(kwargs.get("api_key") or ""),
        base_url=str(kwargs.get("base_url") or "https://api.openai.com/v1"),
        temperature=float(kwargs.get("temperature", 0.1)),
        max_output_tokens=int(kwargs.get("max_output_tokens") or 900),
        timeout=float(kwargs.get("timeout") or 60.0),
    )


def _anthropic(**kwargs: Any) -> LlmClient:
    from kip.core.llm.providers import AnthropicClient

    return AnthropicClient(
        model=str(kwargs.get("model") or "claude-sonnet-4-5"),
        api_key=str(kwargs.get("api_key") or ""),
        temperature=float(kwargs.get("temperature", 0.1)),
        max_output_tokens=int(kwargs.get("max_output_tokens") or 900),
        timeout=float(kwargs.get("timeout") or 60.0),
    )


def _gemini(**kwargs: Any) -> LlmClient:
    from kip.core.llm.providers import GeminiClient

    return GeminiClient(
        model=str(kwargs.get("model") or "gemini-2.0-flash"),
        api_key=str(kwargs.get("api_key") or ""),
        temperature=float(kwargs.get("temperature", 0.1)),
        max_output_tokens=int(kwargs.get("max_output_tokens") or 900),
        timeout=float(kwargs.get("timeout") or 60.0),
    )


def _ollama(**kwargs: Any) -> LlmClient:
    from kip.core.llm.providers import OllamaClient

    return OllamaClient(
        model=str(kwargs.get("model") or "llama3.1"),
        base_url=str(kwargs.get("base_url") or "http://localhost:11434"),
        temperature=float(kwargs.get("temperature", 0.1)),
        max_output_tokens=int(kwargs.get("max_output_tokens") or 900),
        # Local generation on CPU is slow; the default 60s is optimistic.
        timeout=float(kwargs.get("timeout") or 120.0),
    )


#: Provider name -> factory.
LLM_PROVIDERS: dict[str, Callable[..., LlmClient]] = {
    "extractive": _extractive,
    "openai": _openai,
    "anthropic": _anthropic,
    "gemini": _gemini,
    "ollama": _ollama,
}

#: Surfaced on the Settings screen. Each note states the trade-off that actually
#: matters when choosing -- capability, cost, and whether document text leaves the
#: machine -- because that last one is a decision the operator must make knowingly.
LLM_PROVIDER_NOTES: dict[str, str] = {
    "extractive": (
        "Local. Quotes matching sentences from retrieved passages, so answers "
        "are grounded by construction and reproducible. No API key, no data "
        "egress. Cannot paraphrase or synthesise across passages."
    ),
    "openai": (
        "Hosted generation via OpenAI or any compatible endpoint (Azure, Groq, "
        "Together, vLLM, LM Studio). Requires OPENAI_API_KEY; the retrieved "
        "passages are sent to the provider."
    ),
    "anthropic": (
        "Hosted generation via the Anthropic messages API. Requires "
        "ANTHROPIC_API_KEY; the retrieved passages are sent to the provider."
    ),
    "gemini": (
        "Hosted generation via Google Gemini. Requires GEMINI_API_KEY; the "
        "retrieved passages are sent to the provider."
    ),
    "ollama": (
        "Generation by a model served locally by Ollama. No API key and no data "
        "egress; requires the model to be pulled and is slower on CPU."
    ),
}


def get_llm_client(settings: Any = None, **overrides: Any) -> LlmClient:
    """Build the configured :class:`LlmClient`.

    An unknown provider name fails with the list of valid options instead of
    falling back to a default: a typo in ``LLM_PROVIDER`` that silently switched
    generation to a different backend would invalidate every measurement taken
    afterwards.

    >>> class S:
    ...     llm_provider = "extractive"
    ...     llm_model = "kip-extractive-v1"
    ...     llm_temperature = 0.1
    ...     llm_max_output_tokens = 900
    ...     llm_timeout_seconds = 60
    >>> client = get_llm_client(S())
    >>> client.name, client.extractive
    ('extractive', True)
    >>> get_llm_client(S(), provider="gpt5")
    Traceback (most recent call last):
        ...
    kip.core.llm.base.LlmError: Unknown LLM_PROVIDER 'gpt5'. Available providers: anthropic, extractive, gemini, ollama, openai.
    """
    if settings is None:
        from kip.config import get_settings

        settings = get_settings()

    name = str(
        overrides.pop("provider", None) or getattr(settings, "llm_provider", "extractive")
    ).strip().lower()
    factory = LLM_PROVIDERS.get(name)
    if factory is None:
        raise LlmError(
            f"Unknown LLM_PROVIDER {name!r}. Available providers: "
            + ", ".join(sorted(LLM_PROVIDERS))
            + "."
        )

    kwargs: dict[str, Any] = {
        "model": getattr(settings, "llm_model", None),
        "temperature": getattr(settings, "llm_temperature", 0.1),
        "max_output_tokens": getattr(settings, "llm_max_output_tokens", 900),
        "timeout": getattr(settings, "llm_timeout_seconds", 60),
    }
    if name == "openai":
        kwargs["api_key"] = getattr(settings, "openai_api_key", "")
        kwargs["base_url"] = getattr(settings, "openai_base_url", None)
    elif name == "anthropic":
        kwargs["api_key"] = getattr(settings, "anthropic_api_key", "")
    elif name == "gemini":
        kwargs["api_key"] = getattr(settings, "gemini_api_key", "")
    elif name == "ollama":
        kwargs["base_url"] = getattr(settings, "ollama_base_url", None)
    elif name == "extractive":
        # The extractive backend owns its model name; carrying over an unrelated
        # LLM_MODEL would misreport which generator produced an answer.
        kwargs.pop("model", None)

    kwargs.update(overrides)
    return factory(**kwargs)


def make_score_fn(client: LlmClient) -> Callable[[str], str]:
    """Adapt a client into the single-string callable the LLM reranker expects.

    The reranker needs only "send this prompt, give me the text back", and
    inverting the dependency this way keeps :mod:`kip.core.rerank` free of any
    import from this package.

    Temperature is pinned to 0.0. Ranking is a judgement that should not vary
    between two identical requests, and a sampled ordering would make retrieval
    measurements irreproducible.

    >>> from kip.core.llm.extractive import ExtractiveClient
    >>> score = make_score_fn(ExtractiveClient())
    >>> callable(score)
    True

    An extractive client cannot rate passages, and says so rather than returning
    something that :func:`kip.core.rerank.llm.parse_scores` would misread:

    >>> score("Rate these passages.")
    Traceback (most recent call last):
        ...
    kip.core.llm.base.LlmError: RERANKER=llm needs a generative LLM_PROVIDER, but LLM_PROVIDER=extractive cannot score passages. Set RERANKER=heuristic, or configure a hosted provider.
    """
    if client.extractive:

        def _refuse(prompt: str) -> str:
            raise LlmError(
                "RERANKER=llm needs a generative LLM_PROVIDER, but "
                f"LLM_PROVIDER={client.name} cannot score passages. Set "
                "RERANKER=heuristic, or configure a hosted provider."
            )

        return _refuse

    def _score(prompt: str) -> str:
        response = client.complete(
            [Message("user", prompt)],
            temperature=0.0,
            # Ratings are a few tokens each; a large budget here would let a
            # chatty model spend real money explaining itself.
            max_output_tokens=400,
        )
        return response.text

    return _score


def describe_llm_providers() -> list[dict[str, str]]:
    """Provider catalogue for the Settings screen."""
    return [
        {"name": name, "note": LLM_PROVIDER_NOTES.get(name, "")}
        for name in ("extractive", "openai", "anthropic", "gemini", "ollama")
    ]
