"""Scarica i pesi di Kokoro al primo avvio."""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

from . import config

FILES = [
    (config.MODEL_URL, config.MODEL_PATH, "modello Kokoro v1.0"),
    (config.VOICES_URL, config.VOICES_PATH, "pacchetto voci"),
]


def _download(url: str, dest: Path, label: str) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"Scarico {label} ...", flush=True)

    def hook(block: int, block_size: int, total: int) -> None:
        if total <= 0:
            return
        done = min(block * block_size, total)
        pct = done * 100 // total
        bar = "#" * (pct // 3)
        print(f"\r  [{bar:<33}] {pct:3d}%  {done / 1e6:6.1f}/{total / 1e6:.1f} MB",
              end="", flush=True)

    urllib.request.urlretrieve(url, tmp, reporthook=hook)
    print()
    tmp.replace(dest)


def ensure_models(force: bool = False) -> None:
    config.ensure_dirs()
    missing = [(u, d, l) for u, d, l in FILES if force or not d.exists() or d.stat().st_size == 0]
    if not missing:
        return
    for url, dest, label in missing:
        try:
            _download(url, dest, label)
        except Exception as exc:
            print(f"\nErrore nel download di {label}: {exc}", file=sys.stderr)
            raise
    print("Modelli pronti.", flush=True)


def models_ready() -> bool:
    return all(d.exists() and d.stat().st_size > 0 for _, d, _ in FILES)


if __name__ == "__main__":
    ensure_models(force="--force" in sys.argv)
