"""Deterministic, dependency-free lexical embedder (the default provider).

What this is
------------
A signed feature-hashing embedder over stemmed word unigrams, stemmed word
bigrams and surface character 4-grams, with sublinear term weighting, stop-word
down-weighting and L2 normalisation. It produces a fixed-width dense vector for
any text, offline, with no model download and no API key, and it is bit-for-bit
reproducible across processes and machines.

What this is **not**
-------------------
It is not a semantic model. It cannot match "raise the temperature" to "heat it
up" unless the words overlap. Calling it "embeddings" without that caveat would
be the kind of quiet overclaim this project exists to avoid, so the caveat is
here, in the README, and on the Settings screen.

Why ship it as the default
--------------------------
Three concrete reasons:

1. **The platform runs on a fresh clone with zero configuration.** A reviewer
   can ``docker compose up``, upload a document and get grounded answers with
   citations without an API key. That is worth far more than a better default
   nobody can execute.
2. **Retrieval evaluation needs a fixed baseline.** Recall@K measured against a
   deterministic embedder is reproducible; the evaluation dashboard can show the
   lexical baseline and a semantic model side by side and the difference is
   attributable.
3. **Hybrid retrieval already carries the lexical load.** Set
   ``EMBEDDING_PROVIDER=sentence-transformers`` (or ``openai``) to add the
   semantic axis; nothing else in the codebase changes.

Design notes
------------
``hash()`` is *not* used: Python randomises string hashing per process, so an
index built in one process would be unsearchable from the next. Digests come
from ``blake2b`` with a fixed key, memoised per token, which makes the vectors a
stable function of the text alone.

Unigrams and bigrams are built from Porter stems, so "dry", "dried" and "drying"
land in the same bucket. Character 4-grams deliberately use the *surface* form
instead: their job is the morphology and typos the stemmer misses
("dehydration"/"dehydrated", "aw"/"a_w"), and stemming them first would throw
that away.

Stop words are kept but down-weighted rather than dropped. Dropping them
entirely measured slightly better on ranking, but an all-stop-word query ("what
is it") would then embed to the zero vector and rank arbitrarily; a small
non-zero weight keeps such a query degraded-but-defined.

Measured on a 12-passage / 12-query probe set (see ``docs/EVALUATION.md``),
these three choices moved top-1 accuracy from 0.67 to 0.92 and roughly halved
the similarity between unrelated passages, from 0.066 to 0.028. That noise floor
matters as much as the accuracy: ``GROUNDING_MIN_SCORE`` has to sit above it for
the insufficient-evidence check to mean anything.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from typing import Sequence

import numpy as np

from kip.core.embeddings.base import (
    DEFAULT_BATCH_SIZE,
    VECTOR_DTYPE,
    Embedder,
    EmbeddingSpec,
    sublinear_tf,
)
from kip.core.text import STOP_WORDS, normalise_unicode, stem, tokenize

MODEL_NAME = "kip-hashing-v1"
PROVIDER = "hashing"

#: Native width. Chosen by measurement, not by copying a transformer's output
#: size: accuracy is flat from 512 upwards, but the similarity between unrelated
#: passages keeps falling until roughly 1024 (0.045 at 384 -> 0.028 at 1024).
#: 4 KB per chunk is an easy trade for halving the noise floor at the scales
#: where a brute-force store is the right choice anyway.
DEFAULT_DIM = 1024

#: Relative contribution of each feature family. Word unigrams dominate;
#: bigrams add phrase sensitivity ("water activity" vs "activity water");
#: char 4-grams add morphological and typo tolerance. Ablated on the evaluation
#: probe set and reported in docs/EVALUATION.md -- not copied from a tutorial.
WEIGHT_UNIGRAM = 1.0
WEIGHT_BIGRAM = 0.6
WEIGHT_CHARGRAM = 0.35
#: Stop-word unigrams, relative to a content unigram. See the module docstring.
WEIGHT_STOPWORD = 0.25

CHAR_NGRAM_SIZE = 4
#: Below this length a token contributes no char-grams; it already *is* one.
MIN_CHARGRAM_TOKEN_LENGTH = 5
#: Cap char-gram work on pathological input (base64 blobs, minified JSON).
MAX_CHARGRAM_TOKENS = 400

_HASH_KEY = b"kip-hashing-v1"
_digest_cache: dict[str, tuple[int, int]] = {}
#: Bound the memo so a long ingest cannot grow it without limit.
_CACHE_LIMIT = 200_000


def _digest(feature: str) -> tuple[int, int]:
    """Return ``(bucket_seed, sign_bit)`` for a feature string.

    Deterministic across processes, unlike :func:`hash`.
    """
    cached = _digest_cache.get(feature)
    if cached is not None:
        return cached
    raw = hashlib.blake2b(feature.encode("utf-8"), key=_HASH_KEY, digest_size=8).digest()
    value = int.from_bytes(raw, "little")
    result = (value >> 1, value & 1)
    if len(_digest_cache) < _CACHE_LIMIT:
        _digest_cache[feature] = result
    return result


def features(text: str) -> Counter[str]:
    """Return the feature multiset for ``text``.

    Exposed (and doctested) because the feature space *is* the model here: if
    retrieval behaves oddly, this is the function to inspect.

    Prefixes encode the family, which is what lets :func:`_family_weight` stay a
    pure function of the feature string: ``w:`` content unigram, ``s:``
    stop-word unigram, ``b:`` bigram, ``c:`` character 4-gram.

    >>> f = features("Water activity")
    >>> f["w:water"], f["w:activ"]
    (1, 1)
    >>> f["b:water activ"]
    1
    >>> "c:acti" in f
    True

    Stop words are separated out rather than discarded:

    >>> sorted(k for k in features("the drying of mango") if k.startswith(("w:", "s:")))
    ['s:of', 's:the', 'w:dry', 'w:mango']

    Inflections share a unigram, so a "dry" query reaches a "drying" passage:

    >>> features("dried")["w:dry"] == features("drying")["w:dry"] == 1
    True
    >>> features("")
    Counter()
    """
    if not text or not text.strip():
        return Counter()

    tokens = tokenize(normalise_unicode(text))
    if not tokens:
        return Counter()

    stems = [stem(token) for token in tokens]

    counts: Counter[str] = Counter()
    for token, stemmed in zip(tokens, stems):
        counts[f"s:{stemmed}" if token in STOP_WORDS else f"w:{stemmed}"] += 1

    # Bigrams over the *unfiltered* stream: "effect of temperature" keeps its
    # word order information, and dropping stop words first would fuse
    # unrelated terms into a phantom phrase.
    counts.update(f"b:{first} {second}" for first, second in zip(stems, stems[1:]))

    # Char-grams from the surface form of content words only. Stop words are the
    # most frequent tokens, so they would dominate the budget for no gain.
    content = [
        token
        for token in tokens
        if len(token) >= MIN_CHARGRAM_TOKEN_LENGTH and token not in STOP_WORDS
    ][:MAX_CHARGRAM_TOKENS]
    for token in content:
        padded = f"^{token}$"
        for start in range(len(padded) - CHAR_NGRAM_SIZE + 1):
            counts[f"c:{padded[start : start + CHAR_NGRAM_SIZE]}"] += 1
    return counts


def _family_weight(feature: str) -> float:
    """Weight for a feature, derived from its family prefix.

    >>> _family_weight("w:mango"), _family_weight("s:the")
    (1.0, 0.25)
    >>> _family_weight("b:hot air"), _family_weight("c:mang")
    (0.6, 0.35)
    """
    prefix = feature[0]
    if prefix == "w":
        return WEIGHT_UNIGRAM
    if prefix == "s":
        return WEIGHT_STOPWORD
    if prefix == "b":
        return WEIGHT_BIGRAM
    return WEIGHT_CHARGRAM


class HashingEmbedder(Embedder):
    """Signed feature hashing into a fixed-width dense vector."""

    def __init__(
        self,
        dim: int = DEFAULT_DIM,
        *,
        model: str = MODEL_NAME,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        super().__init__(
            EmbeddingSpec(provider=PROVIDER, model=model, dim=int(dim)),
            batch_size=batch_size,
        )

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        matrix = np.zeros((len(texts), self.dim), dtype=VECTOR_DTYPE)
        for row, text in enumerate(texts):
            for feature, count in features(text).items():
                seed, negative = _digest(feature)
                bucket = seed % self.dim
                weight = _family_weight(feature) * sublinear_tf(count)
                matrix[row, bucket] += -weight if negative else weight
        return matrix

    def explain(self, text: str, *, limit: int = 12) -> list[tuple[str, float]]:
        """Return the highest-weighted features for ``text``.

        Used by the Settings screen to make the default embedder legible instead
        of magical -- a reviewer can see exactly what the vector is built from.
        """
        scored = [
            (feature, _family_weight(feature) * sublinear_tf(count))
            for feature, count in features(text).items()
        ]
        scored.sort(key=lambda item: (-item[1], item[0]))
        return scored[:limit]


def build(dim: int = DEFAULT_DIM, **_: object) -> HashingEmbedder:
    """Factory used by the provider registry."""
    return HashingEmbedder(dim=dim)
