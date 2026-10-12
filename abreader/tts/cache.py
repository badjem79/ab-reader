"""Cache su disco dell'audio sintetizzato e dei tempi delle parole.

La sintesi su CPU va solo ~1,35x piu' veloce del tempo reale: rileggere un
capitolo, o tornare indietro di qualche frase, deve essere istantaneo.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import wave
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from .. import config
from . import engine

log = logging.getLogger(__name__)

CACHE_VERSION = 1

_pending: dict[str, Future] = {}
_pending_lock = threading.Lock()
# Un solo worker: la sintesi e' gia' serializzata dal lock di espeak, e cosi'
# le richieste vengono servite nell'ordine in cui servono alla riproduzione.
_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tts")


def _dir() -> Path:
    d = config.CACHE_DIR / "tts"
    d.mkdir(parents=True, exist_ok=True)
    return d


def key_for(text: str, voice: str, speed: float, lang: str) -> str:
    raw = f"{CACHE_VERSION}|{voice}|{speed:.2f}|{lang}|{text}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:20]


def paths(key: str) -> tuple[Path, Path]:
    d = _dir()
    return d / f"{key}.wav", d / f"{key}.json"


def _write_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    pcm = np.clip(audio, -1.0, 1.0)
    pcm = (pcm * 32767.0).astype("<i2")
    tmp = path.with_suffix(".wav.part")
    with wave.open(str(tmp), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm.tobytes())
    tmp.replace(path)


def peek(key: str) -> dict[str, Any] | None:
    """Metadati gia' in cache, senza sintetizzare nulla."""
    _, meta_path = paths(key)
    if not meta_path.exists():
        return None
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _render(text: str, word_spans: list[list[int]], voice: str, speed: float,
            lang: str, key: str) -> dict[str, Any]:
    wav_path, meta_path = paths(key)
    cached = peek(key)
    if cached and wav_path.exists():
        return cached

    result = engine.synthesize(text, word_spans, voice, speed, lang)
    _write_wav(wav_path, result.audio, result.sample_rate)
    meta = {
        "key": key,
        "duration": round(result.duration, 3),
        "words": result.word_times,
        "exact": result.exact,
        "sample_rate": result.sample_rate,
    }
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    return meta


def get(text: str, word_spans: list[list[int]], voice: str, speed: float,
        lang: str) -> dict[str, Any]:
    """Metadati della frase, sintetizzandola se serve. Le richieste doppie si fondono."""
    key = key_for(text, voice, speed, lang)
    wav_path, _ = paths(key)
    cached = peek(key)
    if cached and wav_path.exists():
        return cached

    with _pending_lock:
        fut = _pending.get(key)
        if fut is None:
            fut = _pool.submit(_render, text, word_spans, voice, speed, lang, key)
            _pending[key] = fut
            fut.add_done_callback(lambda _f, k=key: _pending.pop(k, None))
    return fut.result()


def prefetch(text: str, word_spans: list[list[int]], voice: str, speed: float,
             lang: str) -> str:
    """Mette in coda la sintesi senza attenderla. Restituisce la chiave di cache."""
    key = key_for(text, voice, speed, lang)
    wav_path, _ = paths(key)
    if peek(key) and wav_path.exists():
        return key
    with _pending_lock:
        if key not in _pending:
            fut = _pool.submit(_render, text, word_spans, voice, speed, lang, key)
            _pending[key] = fut
            fut.add_done_callback(lambda _f, k=key: _pending.pop(k, None))
    return key


def clear() -> int:
    """Svuota la cache audio; restituisce i megabyte liberati."""
    d = _dir()
    freed = 0
    for f in d.iterdir():
        if f.is_file():
            freed += f.stat().st_size
            f.unlink(missing_ok=True)
    return round(freed / 1e6)


def size_mb() -> float:
    d = _dir()
    return round(sum(f.stat().st_size for f in d.iterdir() if f.is_file()) / 1e6, 1)
