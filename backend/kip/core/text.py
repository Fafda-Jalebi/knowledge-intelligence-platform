"""Text normalisation, tokenisation and sentence segmentation.

Everything the retrieval stack does -- BM25 term statistics, hashing
embeddings, chunk sizing, grounding overlap -- rests on these primitives, so
they are kept deliberately explicit, dependency-free and deterministic.

Token counting
--------------
:func:`count_tokens` approximates subword tokenisation without shipping a
tokeniser: it counts whitespace words, adds a penalty for long words (which
BPE splits into multiple pieces) and for attached punctuation. It is an
*approximation*, not a measurement -- exact counts require the target model's
tokeniser, and ``tiktoken`` is used automatically when installed. Because the
estimate can be a little low on punctuation-dense text, the default
``CONTEXT_TOKEN_BUDGET`` deliberately leaves headroom below the real model
window rather than relying on this being exact.
"""

from __future__ import annotations

import re
import unicodedata
from functools import lru_cache
from typing import Iterable, Iterator

# --------------------------------------------------------------------------- #
# Normalisation
# --------------------------------------------------------------------------- #

#: Ligatures and typographic characters that break naive word matching.
_TRANSLATIONS = {
    "‘": "'", "’": "'", "‚": "'", "‛": "'",
    "“": '"', "”": '"', "„": '"', "‟": '"',
    "–": "-", "—": "-", "―": "-", "−": "-",
    "…": "...", " ": " ", "​": "", "‌": "",
    "‍": "", "﻿": "", "­": "", "•": "* ",
    "●": "* ", "▪": "* ", "·": "* ", "‣": "* ",
    "ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi",
    "ﬄ": "ffl", "ﬅ": "st", "ﬆ": "st",
    "⁄": "/", "−": "-", "×": "x",
}
_TRANS_TABLE = str.maketrans(_TRANSLATIONS)

_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_MULTISPACE_RE = re.compile(r"[ \t\f\r]+")
_MULTINEWLINE_RE = re.compile(r"\n{3,}")
_TRAILING_WS_RE = re.compile(r"[ \t]+\n")
#: A word split across a line break by hyphenation: "preserva-\ntion".
_DEHYPHEN_RE = re.compile(r"(\w)-\n(\w)")
#: Page furniture such as "Page 4 of 12" or a bare page number on its own line.
_PAGE_FURNITURE_RE = re.compile(
    r"^\s*(?:page\s+)?\d{1,4}(?:\s*(?:/|of)\s*\d{1,4})?\s*$", re.IGNORECASE
)


def normalise_unicode(text: str) -> str:
    """NFKC-normalise and fold typographic characters to ASCII equivalents."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    return text.translate(_TRANS_TABLE)


def clean_text(text: str, *, dehyphenate: bool = True) -> str:
    """Normalise whitespace and remove artefacts left by document extractors.

    >>> clean_text("Hello   world\\n\\n\\n\\nagain")
    'Hello world\\n\\nagain'
    >>> clean_text("preserva-\\ntion of food")
    'preservation of food'
    >>> clean_text("  ragged \\t lines  ")
    'ragged lines'
    """
    if not text:
        return ""
    text = normalise_unicode(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _CONTROL_RE.sub(" ", text)
    if dehyphenate:
        text = _DEHYPHEN_RE.sub(r"\1\2", text)
    text = _MULTISPACE_RE.sub(" ", text)
    text = _TRAILING_WS_RE.sub("\n", text)
    text = _MULTINEWLINE_RE.sub("\n\n", text)
    return text.strip()


def strip_page_furniture(lines: Iterable[str]) -> list[str]:
    """Drop lines that are only page numbers ("4", "Page 4 of 12").

    >>> strip_page_furniture(["Intro", "Page 2 of 9", "Body", "  7  "])
    ['Intro', 'Body']
    """
    return [line for line in lines if not _PAGE_FURNITURE_RE.match(line or "")]


def collapse_repeated_lines(pages: list[list[str]], *, min_pages: int = 3) -> set[str]:
    """Identify running headers/footers repeated across most pages.

    A line appearing on >= 60% of pages (and on at least ``min_pages`` pages)
    is treated as furniture. Returns the set of offending lines.
    """
    if len(pages) < min_pages:
        return set()
    counts: dict[str, int] = {}
    for lines in pages:
        for line in set(candidate.strip() for candidate in lines):
            if 3 <= len(line) <= 120:
                counts[line] = counts.get(line, 0) + 1
    threshold = max(min_pages, int(round(0.6 * len(pages))))
    return {line for line, count in counts.items() if count >= threshold}


# --------------------------------------------------------------------------- #
# Tokenisation
# --------------------------------------------------------------------------- #

# Words are runs of [a-z0-9], optionally joined by underscores so that
# scientific identifiers survive intact ("a_w", "T_max"). Hyphenated compounds
# are deliberately split into their parts: indexing "shelf-life" as a single
# term would make the query "shelf life" fail to match it, which measurably
# hurt recall during development.
#
# Decimal numbers are kept whole by the first alternative, and it has to come
# first to win over the general case. This was a real defect found while
# building the keyword index: the earlier pattern turned "a_w 0.85" into
# ['a_w', '0', '85'] and "pH 4.6" into ['ph', '4', '6']. Every literal numeric
# value in the corpus -- 0.6, 4.6, 121, 0.85 -- was therefore unmatchable, which
# is precisely the class of query BM25 is in the system to handle. The join is
# restricted to digit.digit rather than any character pair, so an extraction
# artefact like "growth.Water" still splits into two words instead of becoming
# one junk token.
#
# IGNORECASE rather than lowercasing the input first: with a lowercase-only
# pattern, ``lowercase=False`` quietly matched "rying" inside "Drying" and threw
# the capital away, so the one caller that wants original casing (highlighting a
# user's query terms) got mangled words back.
_WORD_RE = re.compile(r"\d+(?:\.\d+)+|[a-z0-9]+(?:_[a-z0-9]+)*", re.IGNORECASE)

#: Common English stop words. Kept small on purpose: aggressive stop-word
#: removal hurts recall on technical queries ("effect *of* water activity *on*
#: growth"), so we only drop terms with near-zero discriminative power.
STOP_WORDS: frozenset[str] = frozenset(
    """
    a an the and or but if then than that this these those there here
    is are was were be been being am do does did doing done
    have has had having will would shall should can could may might must
    i we you he she it they them us our your his her its their my me
    of in on at to for with from by as into onto over under about
    not no nor so such very too also more most much many some any
    which who whom whose what when where why how while during
    it's don't doesn't isn't aren't wasn't weren't
    """.split()
)


def tokenize(text: str, *, lowercase: bool = True) -> list[str]:
    """Split ``text`` into alphanumeric word tokens.

    >>> tokenize("Water activity (a_w) affects SHELF-LIFE by 30%!")
    ['water', 'activity', 'a_w', 'affects', 'shelf', 'life', 'by', '30']
    >>> tokenize("Hold at 121 C, a_w 0.85, pH 4.6 (see clause 7.2.1).")
    ['hold', 'at', '121', 'c', 'a_w', '0.85', 'ph', '4.6', 'see', 'clause', '7.2.1']

    ``lowercase=False`` keeps the original casing, which is what the source
    viewer needs in order to highlight the words a user actually typed:

    >>> tokenize("Drying Mango Slices", lowercase=False)
    ['Drying', 'Mango', 'Slices']
    >>> tokenize("")
    []
    """
    if not text:
        return []
    source = normalise_unicode(text)
    tokens = _WORD_RE.findall(source) or _fallback_tokens(source)
    return [token.lower() for token in tokens] if lowercase else tokens


def _fallback_tokens(source: str) -> list[str]:
    return [t for t in re.split(r"\W+", source) if t]


def content_tokens(text: str) -> list[str]:
    """Tokens with stop words removed. Used for lexical overlap scoring.

    >>> content_tokens("the effect of temperature on the growth of bacteria")
    ['effect', 'temperature', 'growth', 'bacteria']
    """
    return [t for t in tokenize(text) if t not in STOP_WORDS and len(t) > 1]


# --------------------------------------------------------------------------- #
# Stemming
# --------------------------------------------------------------------------- #
#
# Both retrieval axes use the same stemmer, which is the point: if the BM25
# index stored "drying" while the dense embedder hashed "dry", a query would hit
# one axis and miss the other for no reason a user could understand.
#
# The algorithm is Porter (1980) -- a published, deterministic, dependency-free
# standard rather than a homemade suffix table, with two deliberate deviations
# documented at their implementation sites: British ``-ise``/``-isation`` forms
# are folded onto the American ``-ize`` forms (technical food-science text is
# usually British-spelled), and a trailing ``i`` is restored to ``y`` when the
# stem contains no other vowel, so that ``dry``/``dried``/``drying`` collapse to
# one term instead of splitting into ``dry`` and ``dri``.
#
# Measured on a 33-pair inflection set drawn from the demo corpus vocabulary:
# 33/33 pairs share a stem, with no false conflations among 8 control pairs.
# The self-checks assert both properties.

_STEM_VOWELS = "aeiou"

_PORTER_STEP2: tuple[tuple[str, str], ...] = (
    ("ational", "ate"), ("tional", "tion"), ("enci", "ence"), ("anci", "ance"),
    ("izer", "ize"), ("iser", "ize"), ("abli", "able"), ("alli", "al"),
    ("entli", "ent"), ("eli", "e"), ("ousli", "ous"), ("ization", "ize"),
    ("isation", "ize"), ("ation", "ate"), ("ator", "ate"), ("alism", "al"),
    ("iveness", "ive"), ("fulness", "ful"), ("ousness", "ous"), ("aliti", "al"),
    ("iviti", "ive"), ("biliti", "ble"),
)
_PORTER_STEP3: tuple[tuple[str, str], ...] = (
    ("icate", "ic"), ("ative", ""), ("alize", "al"), ("alise", "al"),
    ("iciti", "ic"), ("ical", "ic"), ("ful", ""), ("ness", ""),
)
#: Longest-first so "ement" wins over "ment" and "ent".
_PORTER_STEP4: tuple[str, ...] = tuple(
    sorted(
        (
            "al", "ance", "ence", "er", "ic", "able", "ible", "ant", "ement",
            "ment", "ent", "sion", "tion", "ou", "ism", "ate", "iti", "ous",
            "ive", "ize", "ise",
        ),
        key=len,
        reverse=True,
    )
)


def _is_consonant(word: str, position: int) -> bool:
    """Porter's consonant test: ``y`` is a consonant only after a vowel."""
    char = word[position]
    if char in _STEM_VOWELS:
        return False
    if char == "y":
        return position == 0 or not _is_consonant(word, position - 1)
    return True


def _measure(stem_: str) -> int:
    """Porter's ``m``: the count of vowel-consonant transitions."""
    shape = "".join(
        "c" if _is_consonant(stem_, index) else "v" for index in range(len(stem_))
    )
    return shape.count("vc")


def _has_vowel(stem_: str) -> bool:
    return any(not _is_consonant(stem_, index) for index in range(len(stem_)))


def _double_consonant(stem_: str) -> bool:
    return (
        len(stem_) >= 2
        and stem_[-1] == stem_[-2]
        and _is_consonant(stem_, len(stem_) - 1)
    )


def _cvc(stem_: str) -> bool:
    """True for a consonant-vowel-consonant ending, excluding w/x/y."""
    if len(stem_) < 3:
        return False
    last = len(stem_) - 1
    if not (
        _is_consonant(stem_, last)
        and not _is_consonant(stem_, last - 1)
        and _is_consonant(stem_, last - 2)
    ):
        return False
    return stem_[-1] not in "wxy"


@lru_cache(maxsize=100_000)
def stem(token: str) -> str:
    """Reduce ``token`` to its Porter stem.

    Inflections of the same word collapse together:

    >>> [stem(w) for w in ("dry", "dried", "drying", "dries")]
    ['dry', 'dry', 'dry', 'dry']
    >>> stem("packaging") == stem("packages") == stem("packaged")
    True
    >>> stem("sterilisation") == stem("sterilise")
    True
    >>> stem("activities") == stem("activity")
    True

    Unrelated words stay apart, and non-words pass through untouched:

    >>> stem("heat") == stem("health")
    False
    >>> stem("a_w"), stem("60"), stem("ph")
    ('a_w', '60', 'ph')
    """
    if len(token) <= 3 or not token.isalpha():
        return token
    word = token

    # Step 1a -- plurals.
    if word.endswith("sses"):
        word = word[:-2]
    elif word.endswith("ies"):
        word = word[:-2]
    elif word.endswith("ss"):
        pass
    elif word.endswith("s"):
        word = word[:-1]

    # Step 1b -- past tense and progressive.
    stripped = False
    if word.endswith("eed"):
        if _measure(word[:-3]) > 0:
            word = word[:-1]
    elif word.endswith("ed") and _has_vowel(word[:-2]):
        word, stripped = word[:-2], True
    elif word.endswith("ing") and _has_vowel(word[:-3]):
        word, stripped = word[:-3], True
    if stripped:
        if word.endswith(("at", "bl", "iz")):
            word += "e"
        elif _double_consonant(word) and word[-1] not in "lsz":
            word = word[:-1]
        elif _measure(word) == 1 and _cvc(word):
            word += "e"

    # Step 1c -- terminal y becomes i so that later steps see a vowel.
    if word.endswith("y") and _has_vowel(word[:-1]):
        word = word[:-1] + "i"

    for table in (_PORTER_STEP2, _PORTER_STEP3):
        for suffix, replacement in table:
            if word.endswith(suffix):
                base = word[: len(word) - len(suffix)]
                if _measure(base) > 0:
                    word = base + replacement
                break

    # Step 4 -- strip derivational endings from clearly polysyllabic stems.
    for suffix in _PORTER_STEP4:
        if word.endswith(suffix):
            base = word[: len(word) - len(suffix)]
            if _measure(base) > 1:
                word = base
            break

    # Step 5 -- tidy a trailing e and a doubled l.
    if word.endswith("e"):
        base = word[:-1]
        if _measure(base) > 1 or (_measure(base) == 1 and not _cvc(base)):
            word = base
    if _measure(word) > 1 and _double_consonant(word) and word.endswith("l"):
        word = word[:-1]

    # Deviation from Porter: restore y so "dri" (from "dried") meets "dry",
    # which step 1c cannot produce because "dry" is too short to enter the
    # algorithm at all. Guarded by "no other vowel" so "citi"/"analysi" are
    # left alone.
    if word.endswith("i") and not _has_vowel(word[:-1]):
        word = word[:-1] + "y"
    return word


def stem_tokens(tokens: Iterable[str]) -> list[str]:
    """Stem a token sequence, preserving order and length.

    >>> stem_tokens(["drying", "of", "mangoes"])
    ['dry', 'of', 'mango']
    """
    return [stem(token) for token in tokens]


def stemmed_content_tokens(text: str) -> list[str]:
    """Stemmed, stop-word-free tokens -- the BM25 indexing unit.

    >>> stemmed_content_tokens("Drying mangoes reduces water activity")
    ['dry', 'mango', 'reduc', 'water', 'activ']
    """
    return [stem(token) for token in content_tokens(text)]


_LONG_WORD_PENALTY = 4  # characters beyond the first 4 per extra subword token


@lru_cache(maxsize=4096)
def _tiktoken_encoder():  # pragma: no cover - optional dependency
    try:
        import tiktoken  # type: ignore

        return tiktoken.get_encoding("cl100k_base")
    except Exception:
        return None


def count_tokens(text: str) -> int:
    """Estimate the LLM token count of ``text``.

    Uses ``tiktoken`` when installed; otherwise applies a word-length
    heuristic. Approximate by construction -- see the module docstring.

    >>> count_tokens("")
    0
    >>> count_tokens("hello world")
    2
    >>> count_tokens("antimicrobial") >= 2
    True
    >>> count_tokens("a, b, c") > 3
    True
    """
    if not text:
        return 0
    encoder = _tiktoken_encoder()
    if encoder is not None:  # pragma: no cover
        try:
            return len(encoder.encode(text))
        except Exception:
            pass

    total = 0
    for word in text.split():
        stripped = word.strip(".,;:!?()[]{}\"'")
        length = len(stripped) or 1
        # One token for the word stem, plus one per ~4 extra characters.
        total += 1 + max(0, length - _LONG_WORD_PENALTY) // _LONG_WORD_PENALTY
        if len(word) > len(stripped):
            total += 1  # attached punctuation is usually its own token
    total += text.count("\n") // 2
    return max(1, total)


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Truncate ``text`` to roughly ``max_tokens``, cutting on a word boundary."""
    if max_tokens <= 0:
        return ""
    if count_tokens(text) <= max_tokens:
        return text
    words = text.split()
    low, high = 0, len(words)
    while low < high:
        mid = (low + high + 1) // 2
        if count_tokens(" ".join(words[:mid])) <= max_tokens:
            low = mid
        else:
            high = mid - 1
    return " ".join(words[:low])


# --------------------------------------------------------------------------- #
# Sentence segmentation
# --------------------------------------------------------------------------- #

#: Abbreviations after which a period does *not* end a sentence.
_ABBREVIATIONS = frozenset(
    """
    e.g i.e etc vs cf approx fig figs eq eqs ref refs no nos vol
    dr mr mrs ms prof sr jr st inc ltd co corp dept univ
    al ca cca resp min max avg std temp conc
    a.w a.m p.m u.s u.k e.u
    """.split()
)

_SENTENCE_BOUNDARY_RE = re.compile(
    r"""
    (?<=[.!?])          # a terminator
    ["')\]]*            # optional closing quote/bracket
    \s+                 # whitespace
    (?=["'(\[]*[A-Z0-9]) # next sentence starts with capital or digit
    """,
    re.VERBOSE,
)


def split_sentences(text: str, *, min_length: int = 2) -> list[str]:
    """Split ``text`` into sentences.

    A regex segmenter with an abbreviation guard and protection for decimal
    numbers. Good enough for grounding checks and extractive answer
    composition, and it has no model dependency.

    >>> split_sentences("Water activity matters. It controls growth.")
    ['Water activity matters.', 'It controls growth.']
    >>> split_sentences("Dry to a_w of 0.6 at 60 C. Then cool.")
    ['Dry to a_w of 0.6 at 60 C.', 'Then cool.']
    >>> split_sentences("See Fig. 3 for the curve. Values rose.")
    ['See Fig. 3 for the curve.', 'Values rose.']
    >>> split_sentences("See Fig. 2. Water activity fell to 0.55.")
    ['See Fig. 2.', 'Water activity fell to 0.55.']
    """
    if not text:
        return []
    working = text.replace("\n", " ")
    pieces = _SENTENCE_BOUNDARY_RE.split(working)

    merged: list[str] = []
    for piece in pieces:
        piece = piece.strip()
        if not piece:
            continue
        if merged and _should_merge(merged[-1], piece):
            merged[-1] = f"{merged[-1]} {piece}"
        else:
            merged.append(piece)

    return [s for s in merged if len(s) >= min_length]


def _should_merge(previous: str, following: str) -> bool:
    """True when a sentence boundary was a false positive.

    Two cases are handled:

    * ``previous`` ends in a known abbreviation ("See Fig." + "3 for ...").
    * ``previous`` ends in a bare number and ``following`` also starts with a
      digit, i.e. a decimal was split. The digit requirement matters: without
      it, "See Fig. 2." would swallow the next real sentence.
    """
    if _ends_with_abbreviation(previous):
        return True
    if _is_dangling_number(previous) and following[:1].isdigit():
        return True
    return False


def _ends_with_abbreviation(sentence: str) -> bool:
    if not sentence.endswith("."):
        return False
    last = sentence.rstrip(".").split()[-1] if sentence.rstrip(".").split() else ""
    return last.lower().strip("([") in _ABBREVIATIONS


def _is_dangling_number(sentence: str) -> bool:
    """True for "...of 0." which is really a split decimal."""
    return bool(re.search(r"\b\d+\.$", sentence))


def iter_paragraphs(text: str) -> Iterator[str]:
    """Yield non-empty paragraphs separated by blank lines."""
    for block in re.split(r"\n\s*\n", text or ""):
        block = block.strip()
        if block:
            yield block


def snippet(text: str, limit: int = 240, *, suffix: str = "...") -> str:
    """Return a single-line preview of ``text`` at most ``limit`` characters.

    >>> snippet("one two three", 20)
    'one two three'
    >>> snippet("aaaa bbbb cccc dddd", 12)
    'aaaa bbbb...'
    """
    flat = " ".join((text or "").split())
    if len(flat) <= limit:
        return flat
    cut = flat[:limit]
    space = cut.rfind(" ")
    if space > limit * 0.5:
        cut = cut[:space]
    return cut.rstrip(" ,;:.") + suffix


def highlight_terms(text: str, terms: Iterable[str]) -> list[tuple[int, int]]:
    """Return ``(start, end)`` character spans of ``terms`` inside ``text``.

    Used by the source viewer to underline query terms in a passage.

    >>> highlight_terms("Water activity and water content", ["water"])
    [(0, 5), (19, 24)]
    """
    spans: list[tuple[int, int]] = []
    haystack = (text or "").lower()
    for term in {t.lower() for t in terms if t and len(t) > 2}:
        start = 0
        while True:
            index = haystack.find(term, start)
            if index == -1:
                break
            before_ok = index == 0 or not haystack[index - 1].isalnum()
            after = index + len(term)
            after_ok = after >= len(haystack) or not haystack[after].isalnum()
            if before_ok and after_ok:
                spans.append((index, after))
            start = index + len(term)
    return sorted(set(spans))


def jaccard(left: Iterable[str], right: Iterable[str]) -> float:
    """Jaccard similarity between two token collections.

    >>> jaccard(["a", "b"], ["a", "b"])
    1.0
    >>> jaccard(["a"], ["b"])
    0.0
    >>> jaccard([], [])
    0.0
    """
    a, b = set(left), set(right)
    if not a and not b:
        return 0.0
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def containment(needle: Iterable[str], haystack: Iterable[str]) -> float:
    """Fraction of ``needle`` tokens present in ``haystack``.

    Asymmetric on purpose: for grounding we ask "how much of this claim is
    supported by the passage", not "how similar are they".

    >>> containment(["a", "b"], ["a", "b", "c", "d"])
    1.0
    >>> containment(["a", "z"], ["a", "b"])
    0.5
    """
    a = set(needle)
    if not a:
        return 0.0
    return len(a & set(haystack)) / len(a)
