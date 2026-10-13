"""Cache su disco dell'audio sintetizzato e dei tempi delle parole.

La sintesi su CPU va solo ~1,35x piu' veloce del tempo reale: rileggere un
capitolo, o tornare indietro di qualche frase, deve essere istantaneo.

L'unita' di sintesi e' il gruppo (una frase, o una frase breve con le vicine),
ma ogni frase ha il suo file: il lettore continua a chiederle una per una.

Ogni libro ha la sua cartella (data/cache/tts/<id libro>/): si puo' svuotare
la cache di un libro solo, e rimuovere un libro ne cancella anche l'audio.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import threading
import wave
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from .. import config
from . import engine, g2p

log = logging.getLogger(__name__)

# 2: fonemi come nella pipeline ufficiale, parole inglesi, frasi brevi in gruppo.
# 3: la chiave segue le scelte di lingua di ogni frase, non solo il dizionario.
CACHE_VERSION = 3

Sentence = tuple[str, list[list[int]]]  # testo e offset delle parole

_pending: dict[str, Future] = {}
_pending_lock = threading.Lock()
# Un solo worker: la sintesi e' gia' serializzata dal lock di espeak, e cosi'
# le richieste vengono servite nell'ordine in cui servono alla riproduzione.
_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tts")


def _root() -> Path:
    return config.CACHE_DIR / "tts"


def _dir(book_id: str) -> Path:
    return _root() / book_id


def key_for(text: str, voice: str, speed: float, lang: str, context: str = "") -> str:
    """Chiave dell'audio di una frase; `context` e' il testo del gruppo, se c'e'."""
    custom = g2p.signature(context or text, engine.espeak_lang(lang, voice))
    raw = f"{CACHE_VERSION}|{voice}|{speed:.2f}|{lang}|{text}|{context}|{custom}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:20]


def group_keys(group: list[Sentence], voice: str, speed: float, lang: str) -> list[str]:
    context = "\n".join(t for t, _ in group) if len(group) > 1 else ""
    return [key_for(t, voice, speed, lang, context) for t, _ in group]


def paths(book_id: str, key: str) -> tuple[Path, Path]:
    d = _dir(book_id)
    return d / f"{key}.wav", d / f"{key}.json"


def _write_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    pcm = np.clip(audio, -1.0, 1.0)
    pcm = (pcm * 32767.0).astype("<i2")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".wav.part")
    with wave.open(str(tmp), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm.tobytes())
    tmp.replace(path)


def peek(book_id: str, key: str) -> dict[str, Any] | None:
    """Metadati gia' in cache, senza sintetizzare nulla."""
    _, meta_path = paths(book_id, key)
    if not meta_path.exists():
        return None
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _cached(book_id: str, key: str) -> dict[str, Any] | None:
    meta = peek(book_id, key)
    return meta if meta and paths(book_id, key)[0].exists() else None


def _render(book_id: str, group: list[Sentence], keys: list[str], voice: str,
            speed: float, lang: str) -> list[dict[str, Any]]:
    done = [_cached(book_id, k) for k in keys]
    if all(done):
        return done

    results = engine.synthesize_group(group, voice, speed, lang)
    metas = []
    for key, result in zip(keys, results):
        wav_path, meta_path = paths(book_id, key)
        _write_wav(wav_path, result.audio, result.sample_rate)
        meta = {
            "key": key,
            "duration": round(result.duration, 3),
            "words": result.word_times,
            "exact": result.exact,
            "sample_rate": result.sample_rate,
        }
        meta_path.write_text(json.dumps(meta), encoding="utf-8")
        metas.append(meta)
    return metas


def _submit(book_id: str, group: list[Sentence], keys: list[str], voice: str,
            speed: float, lang: str) -> Future:
    """Mette in coda il gruppo; le richieste doppie si fondono."""
    job = book_id + ":" + "+".join(keys)
    with _pending_lock:
        fut = _pending.get(job)
        if fut is None:
            fut = _pool.submit(_render, book_id, group, keys, voice, speed, lang)
            _pending[job] = fut
            fut.add_done_callback(lambda _f, k=job: _pending.pop(k, None))
    return fut


def get(book_id: str, group: list[Sentence], target: int, voice: str, speed: float,
        lang: str) -> dict[str, Any]:
    """Metadati della frase `target` del gruppo, sintetizzando il gruppo se serve."""
    keys = group_keys(group, voice, speed, lang)
    cached = _cached(book_id, keys[target])
    if cached:
        return cached
    return _submit(book_id, group, keys, voice, speed, lang).result()[target]


def prefetch(book_id: str, group: list[Sentence], voice: str, speed: float,
             lang: str) -> list[str]:
    """Mette in coda la sintesi del gruppo senza attenderla. Restituisce le chiavi."""
    keys = group_keys(group, voice, speed, lang)
    if not all(_cached(book_id, k) for k in keys):
        _submit(book_id, group, keys, voice, speed, lang)
    return keys


def _bytes(d: Path) -> int:
    if not d.is_dir():
        return 0
    return sum(f.stat().st_size for f in d.rglob("*") if f.is_file())


def clear(book_id: str | None = None) -> float:
    """Svuota la cache audio di un libro, o tutta; restituisce i megabyte liberati."""
    d = _dir(book_id) if book_id else _root()
    freed = _bytes(d)
    shutil.rmtree(d, ignore_errors=True)
    return round(freed / 1e6, 1)


def size_mb(book_id: str | None = None) -> float:
    """Spazio occupato dall'audio di un libro, o da tutta la cache."""
    return round(_bytes(_dir(book_id) if book_id else _root()) / 1e6, 1)


def remove_legacy() -> None:
    """Cancella l'audio delle versioni precedenti, salvato senza cartella per libro.

    Quei file non sono piu' raggiungibili: occupano solo spazio.
    """
    root = _root()
    if not root.is_dir():
        return
    freed = 0
    for f in root.iterdir():
        if f.is_file():
            freed += f.stat().st_size
            f.unlink(missing_ok=True)
    if freed:
        log.info("Cache audio delle versioni precedenti rimossa: %.0f MB", freed / 1e6)
