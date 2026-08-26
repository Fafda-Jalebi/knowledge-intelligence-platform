"""Unit tests for security modules."""

import pytest
from kip.security.passwords import hash_password, verify_password, validate_password_strength, needs_rehash, ValidationError
from kip.security.tokens import encode, decode, TokenError, TokenExpiredError
from kip.security.files import sanitise_filename, validate_upload, extension_of, sniff_kind, PayloadTooLargeError, UnsupportedMediaTypeError


class TestPasswords:
    def test_hash_and_verify(self):
        password = "Correct-Horse-Battery-9"
        stored = hash_password(password)
        assert verify_password(password, stored)
        assert not verify_password("wrong", stored)
        assert not verify_password("", stored)

    def test_salted_hashes_differ(self):
        password = "SamePassword123"
        h1 = hash_password(password)
        h2 = hash_password(password)
        assert h1 != h2
        assert verify_password(password, h1)
        assert verify_password(password, h2)

    def test_unicode_normalization(self):
        # Composed vs decomposed
        pw1 = "café-Passphrase-1"
        pw2 = "cafe\u0301-Passphrase-1"
        stored = hash_password(pw1)
        assert verify_password(pw2, stored)

    def test_malformed_hashes_dont_crash(self):
        for garbage in ("", "not-a-hash", "pbkdf2_sha256$", "$$$$"):
            assert verify_password("anything", garbage) is False

    def test_password_strength_validation(self):
        # Valid
        validate_password_strength("Correct-Horse-Battery-9")

        # Too short
        with pytest.raises(ValidationError):
            validate_password_strength("short")

        # Common password
        with pytest.raises(ValidationError):
            validate_password_strength("password")

        # No variety
        with pytest.raises(ValidationError):
            validate_password_strength("aaaaaaaaaaaa")

        # Empty
        with pytest.raises(ValidationError):
            validate_password_strength("")

    def test_needs_rehash(self):
        # Current hash should not need rehash
        current = hash_password("TestPassword123")
        assert not needs_rehash(current)

        # Weak hash should need rehash
        weak = hash_password("TestPassword123", iterations=1000)
        assert needs_rehash(weak)

        # Garbage should need rehash
        assert needs_rehash("garbage")


class TestTokens:
    def test_encode_decode_roundtrip(self):
        secret = "test-secret-key"
        token = encode({"sub": "42", "role": "user"}, secret, expires_in=600, issuer="kip")
        claims = decode(token, secret, issuer="kip")
        assert claims["sub"] == "42"
        assert claims["role"] == "user"

    def test_wrong_secret_rejected(self):
        secret = "secret1"
        token = encode({"sub": "1"}, secret)
        with pytest.raises(TokenError):
            decode(token, "secret2")

    def test_expired_token_rejected(self):
        secret = "secret"
        token = encode({"sub": "1"}, secret, expires_in=-10)
        with pytest.raises(TokenExpiredError):
            decode(token, secret, leeway=0)

    def test_alg_none_rejected(self):
        secret = "secret"
        token = encode({"sub": "1"}, secret)
        # Manually create alg=none token
        import json
        from kip.security.tokens import b64url_encode, b64url_decode
        head, payload, sig = token.split(".")
        none_head = b64url_encode(json.dumps({"alg": "none", "typ": "JWT"}).encode())
        none_token = f"{none_head}.{payload}."
        with pytest.raises(TokenError):
            decode(none_token, secret)

    def test_tampered_payload_rejected(self):
        secret = "secret"
        token = encode({"sub": "1"}, secret)
        head, payload, sig = token.split(".")
        import json
        from kip.security.tokens import b64url_encode, b64url_decode
        tampered_payload = b64url_encode(json.dumps({"sub": "999"}).encode())
        tampered = f"{head}.{tampered_payload}.{sig}"
        with pytest.raises(TokenError):
            decode(tampered, secret)


class TestFileValidation:
    def test_sanitize_filename(self):
        assert sanitise_filename("../../etc/passwd") == "passwd"
        assert sanitise_filename("..\\windows\\system32") == "system32"
        assert sanitise_filename("/absolute/path") == "path"
        assert sanitise_filename("C:\\Users\\file") == "file"
        assert sanitise_filename("file\x00name") == "file name"
        assert sanitise_filename("") == "document"

    def test_extension_of(self):
        assert extension_of("file.PDF") == "pdf"
        assert extension_of("file.docx") == "docx"
        assert extension_of("noext") == ""
        assert extension_of(".hidden") == ""

    def test_sniff_kind(self):
        assert sniff_kind(b"%PDF-1.4") == "pdf"
        assert sniff_kind(b"PK\x03\x04") == "zip"
        assert sniff_kind(b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1") == "ole"
        assert sniff_kind(b"Plain text") == "text"
        assert sniff_kind(b"") == "empty"
        assert sniff_kind(b"MZ\x90") == "binary"

    def test_validate_upload_pdf(self):
        pdf_content = b"%PDF-1.4\n%EOF\n"
        safe_name, ext = validate_upload("test.pdf", pdf_content, max_bytes=1024*1024)
        assert ext == "pdf"
        assert safe_name.endswith(".pdf")

    def test_validate_upload_rejects_executable(self):
        exe_content = b"MZ\x90\x00"
        with pytest.raises(UnsupportedMediaTypeError):
            validate_upload("evil.exe", exe_content, max_bytes=1024*1024)

    def test_validate_upload_rejects_fake_pdf(self):
        with pytest.raises(UnsupportedMediaTypeError):
            validate_upload("fake.pdf", b"Just text", max_bytes=1024*1024)

    def test_validate_upload_rejects_oversized(self):
        with pytest.raises(PayloadTooLargeError):
            validate_upload("big.txt", b"x" * (1024*1024 + 1), max_bytes=1024*1024)

    def test_validate_upload_empty_rejected(self):
        with pytest.raises(Exception):  # ValidationError
            validate_upload("empty.txt", b"", max_bytes=1024*1024)
