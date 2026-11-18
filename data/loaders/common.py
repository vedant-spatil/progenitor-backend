"""Shared helpers for data loaders: paths, downloads, checksums."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import httpx

log = logging.getLogger(__name__)

BACKEND_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BACKEND_DIR / "data"
# Downloads and generated data. Gitignored; restricted datasets live here only.
EXTERNAL_DIR = DATA_DIR / "external"

USER_AGENT = "progenitor-data-loader/0.1"

# Tag added to every FHIR resource a loader writes, so counts can be split by source.
SOURCE_TAG_SYSTEM = "https://progenitor.dev/fhir/source"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def md5_file(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, dest: Path, timeout: float = 300.0) -> Path:
    """Stream `url` to `dest` atomically (temp file, then rename)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    log.info("download %s -> %s", url, dest)
    with httpx.stream("GET", url, follow_redirects=True, timeout=timeout, headers={"User-Agent": USER_AGENT}) as resp:
        resp.raise_for_status()
        with tmp.open("wb") as fh:
            for chunk in resp.iter_bytes(1 << 20):
                fh.write(chunk)
    tmp.replace(dest)
    return dest


def ensure_file(url: str, dest: Path, *, sha256: str | None = None, md5: str | None = None) -> Path:
    """Download `url` unless `dest` already matches the expected checksum."""
    for attempt in range(2):
        if dest.exists():
            if sha256 and sha256_file(dest) != sha256:
                log.warning("checksum mismatch for %s, re-downloading", dest)
            elif md5 and md5_file(dest) != md5:
                log.warning("checksum mismatch for %s, re-downloading", dest)
            else:
                return dest
        if attempt == 0:
            download(url, dest)
    raise RuntimeError(f"checksum mismatch after download: {dest}")
