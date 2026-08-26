"""Self-checks for the security primitives.

These cover the properties that would be dangerous to get wrong: password
hashing and verification, JWT signing and every way a token can be invalid,
upload validation against files that lie about what they are, path confinement
for deletes, and log redaction of secrets.

The token checks include the classic JWT attacks (``alg: none``, algorithm
confusion, payload tampering, expiry) because those are exactly the failures a
hand-rolled implementation is prone to, and this implementation is hand-rolled
on purpose -- see ``docs/adr/0001-zero-dependency-core.md``.
"""

from __future__ import annotations

import io
import json
import logging
import tempfile
import time
from pathlib import Path

from kip.config import parse_dotenv
from kip.errors import PayloadTooLargeError, UnsupportedMediaTypeError, ValidationError
from kip.logging_setup import (
    REDACTED,
    JsonFormatter,
    SensitiveFilter,
    redact_text,
    redact_value,
)
from kip.security import files as fmod
from kip.security import passwords as pw
from kip.security import tokens as tok
from selfcheck.harness import Harness

SECRET = "unit-test-secret-not-a-real-key"


# --------------------------------------------------------------------------- #
# Passwords
# --------------------------------------------------------------------------- #


def check_passwords(h: Harness) -> None:
    h.group("Password hashing")
    stored = pw.hash_password("Correct-Horse-Battery-9")
    h.ok(stored.startswith(("pbkdf2_sha256$", "argon2")), "hash: recognised algorithm prefix")
    h.ok("Correct-Horse-Battery-9" not in stored, "hash: plaintext is not present in the hash")
    h.ok(pw.verify_password("Correct-Horse-Battery-9", stored), "verify: correct password accepted")
    h.ok(not pw.verify_password("correct-horse-battery-9", stored), "verify: case change rejected")
    h.ok(not pw.verify_password("wrong", stored), "verify: wrong password rejected")
    h.ok(not pw.verify_password("", stored), "verify: empty password rejected")

    other = pw.hash_password("Correct-Horse-Battery-9")
    h.ok(other != stored, "hash: salted, so identical passwords hash differently")
    h.ok(pw.verify_password("Correct-Horse-Battery-9", other), "hash: second salt still verifies")

    # Never raise on malformed stored values; an attacker controls nothing here,
    # but a corrupted row must not become a 500.
    for garbage in ("", "not-a-hash", "pbkdf2_sha256$", "pbkdf2_sha256$abc$def$ghi",
                    "pbkdf2_sha256$1000$!!!$!!!", "argon2$broken", "$$$$"):
        h.ok(
            pw.verify_password("anything", garbage) is False,
            f"verify: malformed stored value rejected without raising ({garbage[:24]!r})",
        )

    # NFKC normalisation: the same password typed with a composed vs decomposed
    # accent must verify, otherwise users get locked out by their keyboard.
    accented = pw.hash_password("café-Passphrase-1")
    h.ok(
        pw.verify_password("cafe\u0301-Passphrase-1", accented),
        "verify: Unicode-equivalent password accepted (NFKC)",
    )
    h.equal(
        pw.normalise_password("cafe\u0301"),
        pw.normalise_password("caf\u00e9"),
        "normalise: composed and decomposed forms agree",
    )

    h.group("Password policy")
    h.no_raise(
        lambda: pw.validate_password_strength("Correct-Horse-Battery-9"),
        "policy: strong password accepted",
    )
    for weak, why in (
        ("short", "too short"),
        ("password", "common"),
        ("aaaaaaaaaaaa", "no variety"),
        ("1234567890", "numeric sequence"),
        ("", "empty"),
    ):
        h.raises(
            ValidationError,
            lambda weak=weak: pw.validate_password_strength(weak),
            f"policy: rejects {why} password",
        )
    h.raises(
        ValidationError,
        lambda: pw.validate_password_strength("x" * 2000),
        "policy: rejects absurdly long password (DoS guard)",
    )

    h.group("Rehash policy")
    weak_hash = pw.hash_password("Correct-Horse-Battery-9", iterations=1000)
    h.ok(pw.needs_rehash(weak_hash), "rehash: low iteration count flagged")
    h.ok(not pw.needs_rehash(stored), "rehash: current hash not flagged")
    h.ok(pw.needs_rehash("garbage"), "rehash: unparseable hash flagged for replacement")
    h.ok(
        pw.verify_password("Correct-Horse-Battery-9", weak_hash),
        "rehash: legacy hash still verifies before upgrade",
    )


# --------------------------------------------------------------------------- #
# Tokens
# --------------------------------------------------------------------------- #


def check_tokens(h: Harness) -> None:
    h.group("JWT round trip")
    token = tok.encode({"sub": "42", "role": "user"}, SECRET, expires_in=600, issuer="kip")
    h.equal(token.count("."), 2, "jwt: three-part compact serialisation")
    claims = h.no_raise(lambda: tok.decode(token, SECRET, issuer="kip"), "jwt: valid token decoded")
    if claims:
        h.equal(claims["sub"], "42", "jwt: subject preserved")
        h.equal(claims["role"], "user", "jwt: custom claim preserved")
        h.ok(claims["exp"] > time.time(), "jwt: expiry is in the future")
        h.ok("iat" in claims, "jwt: issued-at stamped")

    # ``peek`` is the unverified diagnostic accessor: it returns *claims*, not
    # the header, and it must never be trusted for an authorisation decision.
    peeked = tok.peek(token)
    h.equal(peeked.get("sub"), "42", "jwt: peek exposes claims without verifying")
    h.equal(tok.peek("not-a-token"), {}, "jwt: peek returns empty for a malformed token")
    h.equal(
        tok.peek(f"{token.split('.')[0]}.{token.split('.')[1]}.tampered").get("sub"),
        "42",
        "jwt: peek deliberately ignores the signature",
    )

    raw_header = json.loads(tok.b64url_decode(token.split(".")[0]))
    h.equal(raw_header.get("alg"), tok.ALGORITHM, "jwt: header advertises HS256")
    h.equal(raw_header.get("typ"), "JWT", "jwt: header type is JWT")

    h.group("JWT rejection")
    h.raises(tok.TokenError, lambda: tok.decode(token, "different-secret"), "jwt: wrong secret rejected")
    h.raises(tok.TokenError, lambda: tok.decode("", SECRET), "jwt: empty token rejected")
    h.raises(tok.TokenError, lambda: tok.decode("a.b", SECRET), "jwt: two-part token rejected")
    h.raises(tok.TokenError, lambda: tok.decode("a.b.c.d", SECRET), "jwt: four-part token rejected")
    h.raises(tok.TokenError, lambda: tok.decode("!!!.!!!.!!!", SECRET), "jwt: non-base64 rejected")
    h.raises(
        tok.TokenError,
        lambda: tok.decode(token, SECRET, issuer="someone-else"),
        "jwt: issuer mismatch rejected",
    )
    h.raises(
        tok.TokenError,
        lambda: tok.decode(token, SECRET, audience="unexpected"),
        "jwt: missing expected audience rejected",
    )
    h.raises(
        tok.TokenError,
        lambda: tok.decode(token, SECRET, required_claims=("sub", "exp", "tenant")),
        "jwt: absent required claim rejected",
    )
    h.raises(
        tok.TokenError,
        lambda: tok.decode(token, SECRET * 3, algorithm="HS512"),
        "jwt: algorithm mismatch rejected",
    )
    h.ok(len(token) < tok.MAX_TOKEN_BYTES, "jwt: token comfortably under the size cap")
    h.raises(
        tok.TokenError,
        lambda: tok.decode("x" * (tok.MAX_TOKEN_BYTES + 10), SECRET),
        "jwt: oversized token rejected before parsing",
    )

    # Payload tampering.
    head, payload, signature = token.split(".")
    forged_claims = json.loads(tok.b64url_decode(payload))
    forged_claims["sub"] = "1"
    forged = f"{head}.{tok.b64url_encode(json.dumps(forged_claims).encode())}.{signature}"
    h.raises(tok.TokenError, lambda: tok.decode(forged, SECRET), "jwt: tampered payload rejected")

    # alg:none downgrade.
    none_header = tok.b64url_encode(json.dumps({"alg": "none", "typ": "JWT"}).encode())
    none_token = f"{none_header}.{payload}."
    h.raises(tok.TokenError, lambda: tok.decode(none_token, SECRET), "jwt: alg=none rejected")
    none_signed = f"{none_header}.{payload}.{signature}"
    h.raises(
        tok.TokenError,
        lambda: tok.decode(none_signed, SECRET),
        "jwt: alg=none with a borrowed signature rejected",
    )

    # Algorithm confusion (RS256 header against an HMAC verifier).
    rs_header = tok.b64url_encode(json.dumps({"alg": "RS256", "typ": "JWT"}).encode())
    h.raises(
        tok.TokenError,
        lambda: tok.decode(f"{rs_header}.{payload}.{signature}", SECRET),
        "jwt: RS256 header rejected by the HS256 verifier",
    )

    h.group("JWT expiry")
    past = tok.encode({"sub": "9"}, SECRET, expires_in=-120)
    h.raises(tok.TokenExpiredError, lambda: tok.decode(past, SECRET), "jwt: expired token rejected")
    h.ok(
        issubclass(tok.TokenExpiredError, tok.TokenError),
        "jwt: expiry error is a TokenError subclass",
    )
    edge = tok.encode({"sub": "9"}, SECRET, expires_in=-5)
    h.no_raise(lambda: tok.decode(edge, SECRET, leeway=30), "jwt: small clock skew tolerated")
    h.raises(
        tok.TokenExpiredError,
        lambda: tok.decode(edge, SECRET, leeway=0),
        "jwt: no tolerance when leeway is zero",
    )
    future = tok.encode({"sub": "9", "nbf": int(time.time()) + 600}, SECRET, expires_in=900)
    h.raises(
        tok.TokenError,
        lambda: tok.decode(future, SECRET),
        "jwt: not-yet-valid (nbf) token rejected",
    )


# --------------------------------------------------------------------------- #
# Filenames and uploads
# --------------------------------------------------------------------------- #


def check_filenames(h: Harness) -> None:
    h.group("Filename sanitisation")
    cases = [
        ("../../etc/passwd", "traversal segments stripped"),
        ("..\\..\\windows\\system32\\config", "windows traversal stripped"),
        ("/absolute/path/report.pdf", "absolute path stripped"),
        ("C:\\Users\\OM\\report.pdf", "drive letter stripped"),
        ("report\x00.pdf", "NUL byte removed"),
        ("report\n\rname.pdf", "control characters removed"),
    ]
    for raw, why in cases:
        safe = fmod.sanitise_filename(raw)
        ok = (
            "/" not in safe
            and "\\" not in safe
            and ".." not in safe
            and "\x00" not in safe
            and safe.strip() == safe
            and bool(safe)
        )
        h.ok(ok, f"filename: {why} ({raw!r} -> {safe!r})")

    for reserved in ("CON", "con.txt", "PRN.pdf", "NUL", "COM1.docx", "LPT9"):
        safe = fmod.sanitise_filename(reserved)
        stem = safe.split(".")[0].upper()
        h.ok(
            stem not in {"CON", "PRN", "NUL", "AUX", "COM1", "LPT9"},
            f"filename: Windows reserved device name escaped ({reserved!r} -> {safe!r})",
        )

    long_name = "a" * 500 + ".pdf"
    safe_long = fmod.sanitise_filename(long_name)
    h.ok(len(safe_long) <= fmod.MAX_FILENAME_LENGTH, "filename: length capped")
    h.ok(safe_long.endswith(".pdf"), "filename: extension preserved when truncating")
    h.ok(bool(fmod.sanitise_filename("")), "filename: empty name gets a fallback")
    h.ok(bool(fmod.sanitise_filename("...")), "filename: dots-only name gets a fallback")
    h.equal(fmod.extension_of("Report.FINAL.PDF"), "pdf", "filename: extension lowercased")
    h.equal(fmod.extension_of("noext"), "", "filename: missing extension is empty")

    h.group("Content sniffing")
    h.equal(fmod.sniff_kind(b"%PDF-1.7\n..."), "pdf", "sniff: PDF header")
    h.equal(fmod.sniff_kind(b"PK\x03\x04rest"), "zip", "sniff: ZIP header")
    h.equal(fmod.sniff_kind(b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"), "ole", "sniff: legacy OLE (.doc)")
    h.equal(fmod.sniff_kind(b"Plain readable text\n"), "text", "sniff: plain text")
    h.equal(fmod.sniff_kind(b""), "empty", "sniff: empty payload")
    h.equal(fmod.sniff_kind(b"MZ\x90\x00\x03\x00\x00\x00\x04\x00"), "binary", "sniff: executable")


def check_uploads(h: Harness) -> None:
    h.group("Upload validation")
    from kip.core import docgen

    spec = docgen.DocSpec(title="Fixture", author="tests")
    spec.para("A short body paragraph for the fixture document.")
    pdf_bytes = docgen.render_pdf(spec)
    docx_bytes = docgen.render_docx(spec)
    limit = 1024 * 1024

    accepted = [
        ("notes.txt", b"Plain text body.\n", "txt"),
        ("notes.md", b"# Heading\n\nBody.\n", "md"),
        ("paper.pdf", pdf_bytes, "pdf"),
        ("paper.docx", docx_bytes, "docx"),
        ("Paper Name (final).PDF", pdf_bytes, "pdf"),
    ]
    for name, payload, expected_ext in accepted:
        result = h.no_raise(
            lambda name=name, payload=payload: fmod.validate_upload(
                name, payload, max_bytes=limit
            ),
            f"upload: accepts {name!r}",
        )
        if result:
            safe, ext = result
            h.equal(ext, expected_ext, f"upload: extension for {name!r}")
            h.ok("/" not in safe and "\\" not in safe, f"upload: safe name for {name!r}")

    # A declared extension that disagrees with the magic bytes is a *media type*
    # problem (HTTP 415), not a field-validation problem (HTTP 422). Only the
    # genuinely empty upload is a ValidationError.
    rejections = [
        ("evil.exe", b"MZ\x90\x00", UnsupportedMediaTypeError, "executable extension"),
        ("evil.svg", b"<svg/>", UnsupportedMediaTypeError, "SVG (script carrier)"),
        ("noextension", b"body", UnsupportedMediaTypeError, "missing extension"),
        ("renamed.pdf", b"MZ\x90\x00binary payload", UnsupportedMediaTypeError, "renamed executable"),
        ("fake.pdf", b"Just text, not a PDF", UnsupportedMediaTypeError, "text pretending to be PDF"),
        ("fake.docx", b"Just text, not a docx", UnsupportedMediaTypeError, "text pretending to be DOCX"),
        ("fake.docx", b"PK\x03\x04not-really-a-package", UnsupportedMediaTypeError, "ZIP without word/document.xml"),
        ("binary.txt", b"\x00\x01\x02\x03\x04\x05\x06\x07\x08", UnsupportedMediaTypeError, "binary in .txt"),
        ("legacy.docx", b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1", UnsupportedMediaTypeError, "legacy OLE .doc renamed"),
        ("empty.txt", b"", ValidationError, "empty file"),
    ]
    for name, payload, error, why in rejections:
        h.raises(
            error,
            lambda name=name, payload=payload: fmod.validate_upload(
                name, payload, max_bytes=limit
            ),
            f"upload: rejects {why}",
        )

    h.raises(
        PayloadTooLargeError,
        lambda: fmod.validate_upload("big.txt", b"x" * (limit + 1), max_bytes=limit),
        "upload: rejects oversized payload",
    )
    h.raises(
        UnsupportedMediaTypeError,
        lambda: fmod.validate_upload("notes.md", b"# Body\n", max_bytes=limit, allowed_extensions={"pdf"}),
        "upload: honours a narrowed allow-list",
    )
    h.ok(fmod.looks_like_docx(docx_bytes), "upload: real docx recognised as a Word package")
    h.ok(not fmod.looks_like_docx(pdf_bytes), "upload: PDF not mistaken for a Word package")

    h.group("Storage keys and content types")
    keys = {fmod.storage_key("pdf") for _ in range(200)}
    h.equal(len(keys), 200, "storage: keys are unique across 200 draws")
    h.ok(all(key.endswith(".pdf") for key in keys), "storage: key keeps the extension")
    h.ok(
        all("/" not in key and "\\" not in key and ".." not in key for key in keys),
        "storage: key is opaque and path-safe",
    )
    h.equal(fmod.content_type_for("pdf"), "application/pdf", "storage: PDF content type")
    h.equal(fmod.content_type_for("unknown"), "application/octet-stream", "storage: fallback type")
    h.equal(fmod.human_size(0), "0 B", "storage: human size zero")
    h.contains(fmod.human_size(1536), "1.5 KB", "storage: human size KB")
    # Trailing zeros are stripped, so an exact 5 MiB reads "5 MB", not "5.0 MB".
    h.equal(fmod.human_size(5 * 1024 * 1024), "5 MB", "storage: human size MB")


def check_storage(h: Harness) -> None:
    h.group("Disk writes and deletes")
    with tempfile.TemporaryDirectory() as raw_root:
        root = Path(raw_root)
        user_dir = fmod.user_storage_dir(root, 7)
        h.ok(user_dir.is_dir(), "storage: per-user directory created")
        h.ok(str(user_dir).startswith(str(root)), "storage: user directory stays under the root")
        hostile = fmod.user_storage_dir(root, "../../escape")
        h.ok(
            root.resolve() in hostile.resolve().parents,
            f"storage: hostile user id cannot escape the root ({hostile.name!r})",
        )
        h.ok(
            ".." not in hostile.name and "/" not in hostile.name,
            "storage: hostile user id is flattened into a single safe component",
        )

        target = user_dir / "doc.txt"
        written = fmod.write_bytes(target, b"hello world")
        h.equal(written, 11, "storage: write_bytes reports the byte count")
        h.equal(target.read_bytes(), b"hello world", "storage: contents written correctly")
        h.equal(list(user_dir.glob("*.part")), [], "storage: no temporary artefact left behind")

        # Streaming must abort past the limit and leave nothing behind.
        stream_target = user_dir / "stream.bin"
        h.raises(
            PayloadTooLargeError,
            lambda: fmod.stream_to_disk(
                io.BytesIO(b"y" * 5000), stream_target, max_bytes=1000, chunk_size=256
            ),
            "storage: streaming aborts past max_bytes",
        )
        h.ok(not stream_target.exists(), "storage: aborted stream leaves no partial file")
        size = fmod.stream_to_disk(
            io.BytesIO(b"y" * 800), stream_target, max_bytes=1000, chunk_size=256
        )
        h.equal(size, 800, "storage: streaming within the limit succeeds")

        h.ok(fmod.safe_delete(target, root=root), "storage: delete inside the root succeeds")
        h.ok(not target.exists(), "storage: file actually removed")
        h.ok(not fmod.safe_delete(user_dir / "missing.txt", root=root), "storage: missing file is a no-op")

        outside = Path(tempfile.mkdtemp()) / "outside.txt"
        outside.write_text("keep me")
        h.ok(not fmod.safe_delete(outside, root=root), "storage: refuses to delete outside the root")
        h.ok(outside.exists(), "storage: outside file untouched")
        h.ok(
            not fmod.safe_delete(root / ".." / "escape.txt", root=root),
            "storage: refuses a traversal path",
        )


# --------------------------------------------------------------------------- #
# Redaction
# --------------------------------------------------------------------------- #


def check_redaction(h: Harness) -> None:
    h.group("Log redaction")
    samples = [
        ("Authorization: Bearer eyJhbGciOiJIUzI1NiJ9.abc.def", "eyJhbGciOiJIUzI1NiJ9"),
        ("password=hunter2trombone", "hunter2trombone"),
        ("api_key: sk-proj-abcdefghijklmnopqrstuvwxyz0123", "sk-proj-abcdefghijklmnopqrstuvwxyz0123"),
        ('{"token": "abcdefghijklmnopqrstuvwxyz012345"}', "abcdefghijklmnopqrstuvwxyz012345"),
        ("JWT_SECRET=super-secret-value-here", "super-secret-value-here"),
    ]
    for text, secret in samples:
        cleaned = redact_text(text)
        h.ok(secret not in cleaned, f"redact: secret removed from {text.split('=')[0][:24]!r}")
        h.contains(cleaned, REDACTED, "redact: replacement marker present")

    h.equal(redact_value("password", "hunter2"), REDACTED, "redact: sensitive key masked")
    h.equal(redact_value("api_key", "sk-abc"), REDACTED, "redact: api_key masked")
    h.equal(redact_value("jwt_secret", "abc"), REDACTED, "redact: jwt_secret masked")
    h.equal(redact_value("document_id", "42"), "42", "redact: harmless key untouched")
    nested = redact_value("payload", {"user": "om", "password": "hunter2"})
    h.equal(nested.get("password"), REDACTED, "redact: nested dict key masked")
    h.equal(nested.get("user"), "om", "redact: nested harmless key preserved")
    listed = redact_value("items", [{"token": "abc"}, {"name": "ok"}])
    h.equal(listed[0]["token"], REDACTED, "redact: masks inside a list of dicts")

    h.group("Logging pipeline")
    record = logging.LogRecord(
        name="kip.test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="login attempt password=hunter2trombone",
        args=(),
        exc_info=None,
    )
    SensitiveFilter().filter(record)
    rendered = JsonFormatter().format(record)
    h.ok("hunter2trombone" not in rendered, "logging: filter strips secrets before formatting")
    parsed = h.no_raise(lambda: json.loads(rendered), "logging: JSON formatter emits valid JSON")
    if parsed:
        h.equal(parsed.get("level"), "INFO", "logging: level recorded")
        h.equal(parsed.get("logger"), "kip.test", "logging: logger name recorded")
        # JsonFormatter emits the rendered message under "msg" (matching the
        # LogRecord attribute name) rather than "message".
        h.contains(parsed.get("msg", ""), REDACTED, "logging: redaction visible in output")
        h.ok("hunter2trombone" not in json.dumps(parsed), "logging: no secret anywhere in the payload")


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #


def check_config(h: Harness) -> None:
    h.group("Settings and .env parsing")
    parsed = parse_dotenv(
        "\n".join(
            [
                "# a comment",
                "",
                "PLAIN=value",
                'QUOTED="spaced value"',
                "SINGLE='single quoted'",
                "EMPTY=",
                "export EXPORTED=yes",
                "WITH_EQUALS=a=b=c",
                "  INDENTED = trimmed  ",
                "NOT A LINE",
            ]
        )
    )
    h.equal(parsed.get("PLAIN"), "value", "dotenv: bare value")
    h.equal(parsed.get("QUOTED"), "spaced value", "dotenv: double quotes stripped")
    h.equal(parsed.get("SINGLE"), "single quoted", "dotenv: single quotes stripped")
    h.equal(parsed.get("EMPTY"), "", "dotenv: empty value kept")
    h.equal(parsed.get("EXPORTED"), "yes", "dotenv: export prefix handled")
    h.equal(parsed.get("WITH_EQUALS"), "a=b=c", "dotenv: only the first '=' splits")
    h.equal(parsed.get("INDENTED"), "trimmed", "dotenv: whitespace trimmed")
    h.ok("NOT A LINE" not in parsed, "dotenv: malformed line ignored")

    from kip.config import Settings

    settings = Settings()
    redactedmap = settings.redacted()
    h.ok(
        all(
            redactedmap.get(key) in (None, "", REDACTED) or key not in redactedmap
            for key in ("jwt_secret", "openai_api_key", "anthropic_api_key")
        ),
        "settings: secret fields masked in redacted()",
    )
    h.ok("chunk_target_tokens" in redactedmap, "settings: non-secret fields still present")
    serialised = h.no_raise(lambda: json.dumps(redactedmap), "settings: redacted() is JSON safe")
    if serialised:
        h.ok("BEGIN" not in serialised, "settings: no key material leaked")

    production = Settings(app_env="production", jwt_secret="")
    problems = production.validate()
    h.ok(
        any("JWT_SECRET" in problem for problem in problems),
        "settings: production without JWT_SECRET is reported invalid",
    )
    h.ok(production.is_production, "settings: production flag derived from app_env")
    h.equal(Settings(app_env="development").validate(), [], "settings: dev defaults are valid")
    h.equal(
        Settings(max_upload_mb=25).max_upload_bytes,
        25 * 1024 * 1024,
        "settings: upload limit converted to bytes",
    )
    h.contains(
        Settings(allowed_extensions="pdf, DOCX , md").allowed_extension_set,
        "docx",
        "settings: extension list normalised",
    )


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def run(verbose: bool = False) -> Harness:
    h = Harness(name="security", verbose=verbose)
    check_passwords(h)
    check_tokens(h)
    check_filenames(h)
    check_uploads(h)
    check_storage(h)
    check_redaction(h)
    check_config(h)
    return h


def main(argv: list[str] | None = None) -> int:
    import sys

    args = argv if argv is not None else sys.argv[1:]
    harness = run(verbose="-v" in args or "--verbose" in args)
    print(harness.report())
    return 0 if harness.succeeded else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
