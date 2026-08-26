"""Upload validation and safe on-disk storage.

Threat model addressed here
---------------------------
* **Path traversal / absolute paths** - ``sanitise_filename`` strips directory
  components, ``..`` segments, NUL bytes, control characters and Windows
  reserved device names (``CON``, ``PRN``, ``LPT1``, ...).
* **Extension spoofing** - the declared extension must agree with the file's
  magic bytes (``sniff_kind``). A ``.txt`` file whose bytes start with ``%PDF``
  is rejected, and so is ``payload.exe`` renamed to ``report.pdf``.
* **Oversized payloads** - size is checked against ``MAX_UPLOAD_MB`` *before*
  extraction, and the streaming writer aborts mid-write once the cap is passed
  so a lying ``Content-Length`` cannot fill the disk.
* **Empty / corrupt files** - rejected with an actionable message rather than
  surfacing a parser traceback.
* **Collisions & enumeration** - stored files are named
  ``<uuid4>.<ext>`` under a per-user directory; the human-readable filename
  lives in the database only. A malicious filename therefore never becomes a
  filesystem path.
"""

from __future__ import annotations

import os
import re
import shutil
import unicodedata
import uuid
from pathlib import Path
from typing import BinaryIO, Iterable

from kip.errors import (
    PayloadTooLargeError,
    UnsupportedMediaTypeError,
    ValidationError,
)

MAX_FILENAME_LENGTH = 180
DEFAULT_ALLOWED = frozenset({"pdf", "docx", "txt", "md"})

#: Windows reserved device names (case-insensitive, with or without extension).
_WINDOWS_RESERVED = frozenset(
    {"con", "prn", "aux", "nul"}
    | {f"com{i}" for i in range(1, 10)}
    | {f"lpt{i}" for i in range(1, 10)}
)

_UNSAFE_CHARS = re.compile(r'[\x00-\x1f\x7f<>:"/\\|?*]')
_COLLAPSE = re.compile(r"[\s_]+")
_MULTI_DOT = re.compile(r"\.{2,}")

MIME_BY_EXTENSION: dict[str, str] = {
    "pdf": "application/pdf",
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "txt": "text/plain",
    "md": "text/markdown",
}

#: Extensions that are explicitly dangerous to accept in a document platform.
BLOCKED_EXTENSIONS = frozenset(
    {
        "exe", "dll", "so", "dylib", "bat", "cmd", "com", "cpl", "msi", "msp",
        "scr", "sh", "bash", "zsh", "ps1", "psm1", "vbs", "vbe", "js", "jse",
        "jar", "war", "py", "pyc", "pyo", "rb", "pl", "php", "asp", "aspx",
        "jsp", "cgi", "app", "deb", "rpm", "apk", "lnk", "reg", "hta", "wsf",
        "svg", "html", "htm", "xhtml", "xml", "iso", "img", "dmg",
    }
)


def sanitise_filename(raw: str, *, fallback: str = "document") -> str:
    """Return a display-safe filename with no path or control characters.

    The returned name is at most :data:`MAX_FILENAME_LENGTH` characters *in
    total* -- the extension is preserved and the stem is truncated to make room
    for it, because most filesystems cap the whole entry name rather than the
    stem.

    >>> sanitise_filename("../../etc/passwd")
    'passwd'
    >>> sanitise_filename("C:\\\\Windows\\\\system32\\\\notes.txt")
    'notes.txt'
    >>> sanitise_filename("   ")
    'document'
    >>> sanitise_filename("CON.txt")
    'file_CON.txt'
    >>> len(sanitise_filename("a" * 500 + ".pdf")) <= MAX_FILENAME_LENGTH
    True
    """
    name = str(raw or "")
    name = unicodedata.normalize("NFKC", name)
    # Take the last path component under both POSIX and Windows conventions.
    name = name.replace("\\", "/").split("/")[-1]
    name = _UNSAFE_CHARS.sub("_", name)
    name = _MULTI_DOT.sub(".", name)
    name = _COLLAPSE.sub(" ", name).strip(" .")

    if not name:
        return fallback

    stem, dot, ext = name.rpartition(".")
    if not dot:
        stem, ext = name, ""
    if stem.lower() in _WINDOWS_RESERVED or name.lower() in _WINDOWS_RESERVED:
        stem = f"file_{stem}"

    # A pathological extension (".aaaa...") gets dropped rather than allowed to
    # consume the entire budget.
    if len(ext) > 16:
        stem, ext = f"{stem}.{ext}"[:MAX_FILENAME_LENGTH], ""

    budget = MAX_FILENAME_LENGTH - (len(ext) + 1 if ext else 0)
    if len(stem) > budget:
        stem = stem[:budget].rstrip(" .")
    if not stem:
        stem = fallback

    return f"{stem}.{ext}" if ext else stem


def extension_of(filename: str) -> str:
    """Return the lower-cased extension without a leading dot ('' if none)."""
    name = sanitise_filename(filename)
    _, dot, ext = name.rpartition(".")
    if not dot:
        return ""
    ext = ext.lower().strip()
    return ext if ext.isalnum() else ""


def sniff_kind(head: bytes) -> str:
    """Identify a file kind from its leading bytes.

    Returns one of ``pdf``, ``docx``, ``zip``, ``ole``, ``binary``, ``text``
    or ``empty``. ``docx`` is a ZIP container, so we report ``zip`` when the
    ZIP magic is present and let :func:`validate_upload` confirm the internal
    structure.

    >>> sniff_kind(b"%PDF-1.7\\n")
    'pdf'
    >>> sniff_kind(b"PK\\x03\\x04")
    'zip'
    >>> sniff_kind(b"hello world")
    'text'
    >>> sniff_kind(b"")
    'empty'
    """
    if not head:
        return "empty"
    if head[:5] == b"%PDF-" or head[:4] == b"%PDF":
        return "pdf"
    if head[:4] in (b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08"):
        return "zip"
    if head[:8] == b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1":
        return "ole"  # legacy .doc / .xls
    if head[:4] in (b"\x7fELF",) or head[:2] == b"MZ":
        return "binary"
    if b"\x00" in head[:512]:
        return "binary"
    for encoding in ("utf-8", "utf-16", "latin-1"):
        try:
            head.decode(encoding)
            return "text"
        except UnicodeDecodeError:
            continue
    return "binary"


def looks_like_docx(payload: bytes) -> bool:
    """True when ``payload`` is a ZIP that contains a WordprocessingML part."""
    if payload[:4] != b"PK\x03\x04":
        return False
    # Cheap structural probe first, so we avoid opening obviously-wrong zips.
    if b"word/document.xml" in payload[:8192] or b"[Content_Types].xml" in payload[:4096]:
        pass
    import io
    import zipfile

    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            names = set(archive.namelist())
    except Exception:
        return False
    return "word/document.xml" in names


def validate_upload(
    filename: str,
    payload: bytes,
    *,
    max_bytes: int,
    allowed_extensions: Iterable[str] = DEFAULT_ALLOWED,
) -> tuple[str, str]:
    """Validate an upload and return ``(safe_filename, extension)``.

    Raises :class:`ValidationError`, :class:`PayloadTooLargeError` or
    :class:`UnsupportedMediaTypeError` with messages intended for end users.
    """
    allowed = frozenset(e.lower().lstrip(".") for e in allowed_extensions) or DEFAULT_ALLOWED
    safe_name = sanitise_filename(filename)
    ext = extension_of(safe_name)

    if not ext:
        raise UnsupportedMediaTypeError(
            "The file has no extension. Supported types: "
            + ", ".join(sorted(allowed)).upper()
            + "."
        )
    if ext in BLOCKED_EXTENSIONS:
        raise UnsupportedMediaTypeError(
            f"'.{ext}' files are not accepted for security reasons. "
            "Supported types: " + ", ".join(sorted(allowed)).upper() + "."
        )
    if ext not in allowed:
        raise UnsupportedMediaTypeError(
            f"'.{ext}' is not a supported document type. Supported types: "
            + ", ".join(sorted(allowed)).upper()
            + "."
        )

    size = len(payload)
    if size == 0:
        raise ValidationError(
            "The file is empty. Please upload a document that contains text."
        )
    if size > max_bytes:
        raise PayloadTooLargeError(
            f"The file is {human_size(size)}, which exceeds the "
            f"{human_size(max_bytes)} limit.",
            details={"size_bytes": size, "max_bytes": max_bytes},
        )

    kind = sniff_kind(payload[:1024])

    if kind == "binary":
        raise UnsupportedMediaTypeError(
            "The file appears to be a binary executable or an unsupported "
            "format rather than a document."
        )
    if kind == "ole":
        raise UnsupportedMediaTypeError(
            "Legacy Microsoft Office files (.doc/.xls) are not supported. "
            "Please save the document as .docx or .pdf and try again."
        )

    if ext == "pdf" and kind != "pdf":
        raise UnsupportedMediaTypeError(
            "This file is named '.pdf' but its contents are not a PDF. "
            "The file may be corrupt or renamed."
        )
    if ext == "docx":
        if kind != "zip" or not looks_like_docx(payload):
            raise UnsupportedMediaTypeError(
                "This file is named '.docx' but is not a valid Word document. "
                "The file may be corrupt or renamed."
            )
    if ext in {"txt", "md"} and kind not in {"text", "empty"}:
        raise UnsupportedMediaTypeError(
            f"This file is named '.{ext}' but does not contain readable text."
        )

    return safe_name, ext


def storage_key(extension: str) -> str:
    """Return an opaque, collision-free storage filename."""
    ext = (extension or "bin").lower().lstrip(".")
    if not ext.isalnum():
        ext = "bin"
    return f"{uuid.uuid4().hex}.{ext}"


def user_storage_dir(root: Path | str, user_id: int | str) -> Path:
    """Return (and create) the per-user storage directory."""
    safe_user = re.sub(r"[^A-Za-z0-9_-]", "_", str(user_id))[:64] or "anon"
    path = Path(root) / f"u{safe_user}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_bytes(destination: Path, payload: bytes) -> int:
    """Write ``payload`` atomically and return the number of bytes written."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_suffix(destination.suffix + ".part")
    with open(temp, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    temp.replace(destination)
    return len(payload)


def stream_to_disk(
    source: BinaryIO,
    destination: Path,
    *,
    max_bytes: int,
    chunk_size: int = 1024 * 512,
) -> int:
    """Stream ``source`` to ``destination``, aborting if ``max_bytes`` is passed.

    Guarantees no partial file is left behind on failure. Returns the byte
    count written.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_suffix(destination.suffix + ".part")
    total = 0
    try:
        with open(temp, "wb") as handle:
            while True:
                block = source.read(chunk_size)
                if not block:
                    break
                total += len(block)
                if total > max_bytes:
                    raise PayloadTooLargeError(
                        f"Upload exceeds the {human_size(max_bytes)} limit.",
                        details={"max_bytes": max_bytes},
                    )
                handle.write(block)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        temp.unlink(missing_ok=True)
        raise
    if total == 0:
        temp.unlink(missing_ok=True)
        raise ValidationError("The uploaded file is empty.")
    temp.replace(destination)
    return total


def safe_delete(path: Path | str, *, root: Path | str) -> bool:
    """Delete ``path`` only if it genuinely lives inside ``root``.

    Defends against a corrupted database row pointing at an arbitrary path.

    Returns ``True`` only when something was actually removed. A refusal
    (outside ``root``) and a no-op (the file was already gone) are both
    ``False``, which lets the caller distinguish "storage cleaned up" from
    "nothing to clean up" without inspecting the filesystem again.
    """
    try:
        target = Path(path).resolve()
        base = Path(root).resolve()
    except OSError:
        return False
    if not str(target).startswith(str(base) + os.sep) and target != base:
        return False
    if target.is_dir():
        shutil.rmtree(target, ignore_errors=True)
        return not target.exists()
    try:
        target.unlink()
        return True
    except (FileNotFoundError, OSError):
        return False


def human_size(num_bytes: int | float) -> str:
    """Format a byte count for end users.

    >>> human_size(0)
    '0 B'
    >>> human_size(1536)
    '1.5 KB'
    >>> human_size(26214400)
    '25 MB'
    """
    value = float(num_bytes or 0)
    if value < 1024:
        return f"{int(value)} B"
    for unit in ("KB", "MB", "GB", "TB"):
        value /= 1024.0
        if value < 1024.0 or unit == "TB":
            rendered = f"{value:.1f}".rstrip("0").rstrip(".")
            return f"{rendered} {unit}"
    return f"{value:.1f} TB"  # pragma: no cover


def content_type_for(extension: str) -> str:
    return MIME_BY_EXTENSION.get((extension or "").lower().lstrip("."), "application/octet-stream")
