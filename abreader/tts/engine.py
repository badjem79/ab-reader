"""Sintesi vocale con Kokoro e allineamento parola-per-parola.

Il modello ONNX (release v1.1) restituisce la durata di ogni fonema: da li' si
ricava esattamente quando viene pronunciata ogni parola, senza stime.
"""

from __future__ import annotations

import logging
import re
import threading
from dataclasses import dataclass
from typing import Any

import numpy as np

from .. import config

log = logging.getLogger(__name__)

_lock = threading.Lock()
_kokoro: Any = None

# Prima lettera del nome della voce -> (codice espeak, etichetta lingua).
_VOICE_LANGS = {
    "a": ("en-us", "Inglese (US)"),
    "b": ("en-gb", "Inglese (UK)"),
    "i": ("it", "Italiano"),
    "e": ("es", "Spagnolo"),
    "f": ("fr-fr", "Francese"),
    "h": ("hi", "Hindi"),
    "p": ("pt-br", "Portoghese (BR)"),
    "j": ("ja", "Giapponese"),
    "z": ("cmn", "Cinese"),
}
_LANG_TO_ESPEAK = {"it": "it", "en": "en-us"}

_WS_RE = re.compile(r"\S+")


@dataclass
class SynthResult:
    audio: np.ndarray          # float32 mono a 24 kHz
    sample_rate: int
    duration: float
    word_times: list[list[float]]  # [start, end] per parola, in secondi
    exact: bool                # True se l'allineamento viene dal modello


def espeak_lang(lang: str, voice: str | None = None) -> str:
    """Codice lingua per espeak: la voce, se c'e', e' piu' affidabile del metadato."""
    if voice:
        code = _VOICE_LANGS.get(voice[0])
        if code:
            return code[0]
    return _LANG_TO_ESPEAK.get(lang, "en-us")


def _engine():
    global _kokoro
    if _kokoro is None:
        with _lock:
            if _kokoro is None:
                from kokoro_onnx import Kokoro
                if not config.MODEL_PATH.exists() or not config.VOICES_PATH.exists():
                    raise RuntimeError(
                        "Modelli Kokoro mancanti: esegui "
                        "`python -m abreader.download_models`")
                log.info("Carico il modello Kokoro...")
                _kokoro = Kokoro(str(config.MODEL_PATH), str(config.VOICES_PATH))
                if not _kokoro.has_timings:
                    log.warning(
                        "Il modello non espone le durate dei fonemi: l'evidenziazione "
                        "sara' stimata invece che esatta.")
    return _kokoro


def list_voices() -> list[dict[str, str]]:
    k = _engine()
    raw = k.voices
    names = sorted(raw.files if hasattr(raw, "files") else raw.keys())
    out = []
    for name in names:
        lang_code, label = _VOICE_LANGS.get(name[0], ("en-us", "Altro"))
        gender = "F" if len(name) > 1 and name[1] == "f" else "M"
        out.append({
            "id": name,
            "label": name.split("_", 1)[-1].capitalize(),
            "lang": "it" if lang_code == "it" else ("en" if lang_code.startswith("en") else lang_code),
            "lang_label": label,
            "gender": gender,
        })
    return out


def warmup(voice: str = "af_heart") -> None:
    """Prima inferenza a vuoto: sposta il costo di caricamento fuori dalla riproduzione."""
    try:
        _engine().create_timed("ok.", voice=voice, speed=1.0, lang=espeak_lang("en", voice))
    except Exception as exc:  # il riscaldamento non deve mai bloccare l'avvio
        log.warning("Warmup fallito: %s", exc)


def _phoneme_groups(spoken) -> list[tuple[float, float]]:
    """Raggruppa i fonemi in parole: nella sequenza le parole sono divise da spazi."""
    groups: list[tuple[float, float]] = []
    start = end = None
    for t in spoken:
        if t.phoneme.isspace():
            if start is not None:
                groups.append((start, end))
                start = end = None
            continue
        if start is None:
            start = t.start
        end = t.end
    if start is not None:
        groups.append((start, end))
    return groups


def _split_span(span: tuple[float, float], weights: list[int]) -> list[tuple[float, float]]:
    """Ripartisce un intervallo temporale fra piu' elementi, in proporzione ai pesi."""
    start, end = span
    total = sum(weights) or 1
    out, cursor = [], start
    for w in weights:
        nxt = cursor + (end - start) * w / total
        out.append((cursor, nxt))
        cursor = nxt
    if out:
        out[-1] = (out[-1][0], end)
    return out


def align_words(text: str, word_spans: list[list[int]], spoken, duration: float
                ) -> tuple[list[list[float]], bool]:
    """Assegna a ogni parola del testo l'intervallo di audio in cui viene pronunciata.

    I gruppi fonemici corrispondono ai token separati da spazi del testo originale,
    non alle parole "pulite": si passa quindi per i token e poi si scende alle parole
    che ciascuno contiene.
    """
    ws_tokens = [m.span() for m in _WS_RE.finditer(text)]
    if not ws_tokens or not word_spans:
        return [[0.0, duration] for _ in word_spans], False

    groups = _phoneme_groups(spoken) if spoken else []
    exact = len(groups) == len(ws_tokens)

    if exact:
        token_times = groups
    else:
        # Ripiego: la durata si ripartisce sui token in proporzione alla lunghezza.
        if groups:
            log.debug("Allineamento approssimato: %d gruppi per %d token",
                      len(groups), len(ws_tokens))
        token_times = _split_span((0.0, duration), [e - s for s, e in ws_tokens])

    # Ogni parola cade dentro un token; se un token ne contiene piu' d'una
    # (per esempio per un trattino) il suo intervallo si divide fra loro.
    by_token: dict[int, list[int]] = {}
    for wi, (ws, we) in enumerate(word_spans):
        for ti, (ts, te) in enumerate(ws_tokens):
            if ws >= ts and we <= te:
                by_token.setdefault(ti, []).append(wi)
                break
        else:
            by_token.setdefault(-1, []).append(wi)

    times: list[list[float]] = [[0.0, 0.0] for _ in word_spans]
    for ti, word_ids in by_token.items():
        if ti < 0 or ti >= len(token_times):
            continue
        span = token_times[ti]
        if len(word_ids) == 1:
            times[word_ids[0]] = [round(span[0], 3), round(span[1], 3)]
        else:
            weights = [word_spans[w][1] - word_spans[w][0] for w in word_ids]
            for wi, part in zip(word_ids, _split_span(span, weights)):
                times[wi] = [round(part[0], 3), round(part[1], 3)]

    # Le parole rimaste senza intervallo (rare) ereditano il confine precedente,
    # cosi' l'evidenziazione non torna mai indietro.
    last = 0.0
    for t in times:
        if t[1] <= 0.0:
            t[0] = t[1] = last
        else:
            last = t[1]
    return times, exact


def synthesize(text: str, word_spans: list[list[int]], voice: str, speed: float,
               lang: str) -> SynthResult:
    """Sintetizza una frase e restituisce audio e tempi di ogni parola."""
    k = _engine()
    code = espeak_lang(lang, voice)
    # espeak tiene stato globale di processo: una sola sintesi alla volta.
    with _lock:
        audio, sample_rate, spoken = k.create_timed(
            text, voice=voice, speed=float(speed), lang=code)

    audio = np.asarray(audio, dtype=np.float32).ravel()
    duration = len(audio) / sample_rate
    times, exact = align_words(text, word_spans, spoken, duration)
    return SynthResult(audio=audio, sample_rate=sample_rate, duration=duration,
                       word_times=times, exact=exact)
