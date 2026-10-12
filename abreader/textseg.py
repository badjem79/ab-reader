"""Segmentazione del testo in frasi e parole, per italiano e inglese.

Le frasi sono l'unita' di sintesi: tenerle corte (max ~200 caratteri) migliora sia
la latenza iniziale sia la precisione dell'evidenziazione, perche' l'errore di
allineamento non si accumula oltre la fine della frase.
"""

from __future__ import annotations

import re

MAX_SENTENCE_CHARS = 200
MIN_SENTENCE_WORDS = 2

# Abbreviazioni dopo le quali il punto non chiude la frase.
_ABBREV = {
    # italiano
    "sig", "sig.ra", "sigg", "dott", "dott.ssa", "dr", "prof", "prof.ssa", "ing",
    "avv", "on", "rev", "geom", "rag", "arch", "ecc", "es", "pag", "pagg", "cap",
    "vol", "art", "artt", "fig", "tab", "n", "nn", "ca", "ass", "spa", "srl",
    "a.c", "d.c", "p.es", "v.le", "p.zza", "loc", "sec", "seg", "segg", "cfr",
    # inglese
    "mr", "mrs", "ms", "dr", "prof", "st", "jr", "sr", "vs", "etc", "e.g", "i.e",
    "no", "vol", "ch", "approx", "inc", "ltd", "co", "dept", "est", "fig", "al",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec",
}

_WORD_RE = re.compile(r"[^\W\d_]+(?:['’-][^\W\d_]+)*|\d+(?:[.,]\d+)*", re.UNICODE)
_SENT_END_RE = re.compile(r"[.!?…]+[\"'”»)\]]*(?=\s|$)")
_CLOSING = "\"'”»)]"

# Parole funzione discriminanti: bastano per distinguere italiano da inglese.
_IT_STOP = {"che", "non", "per", "con", "una", "gli", "del", "della", "sono", "come",
            "piu", "più", "ma", "se", "nel", "alla", "dei", "delle", "questo",
            "quando", "anche", "essere", "aveva", "erano", "sua", "suo", "loro"}
_EN_STOP = {"the", "and", "that", "with", "for", "was", "were", "have", "this", "from",
            "they", "you", "his", "her", "not", "but", "had", "which", "would", "there",
            "their", "what", "when", "about"}


def detect_language(text: str, hint: str | None = None) -> str:
    """'it' o 'en'. Il metadato dell'epub ha la precedenza, se dice qualcosa di utile."""
    if hint:
        h = hint.lower().replace("_", "-")
        if h.startswith("it"):
            return "it"
        if h.startswith("en"):
            return "en"
    sample = [w.lower() for w in _WORD_RE.findall(text[:20000])]
    it = sum(1 for w in sample if w in _IT_STOP)
    en = sum(1 for w in sample if w in _EN_STOP)
    return "it" if it > en else "en"


def split_words(text: str) -> list[tuple[int, int]]:
    """Offset (start, end) di ogni parola pronunciabile del testo."""
    return [m.span() for m in _WORD_RE.finditer(text)]


def _is_boundary(text: str, end: int) -> bool:
    """`end` e' la posizione subito dopo la punteggiatura: e' fine frase davvero?"""
    # Il carattere di punteggiatura effettivo, saltando le virgolette di chiusura.
    i = end - 1
    while i >= 0 and text[i] in _CLOSING:
        i -= 1
    if i < 0:
        return False
    punct = text[i]

    # '!' e '?' chiudono sempre; l'ellissi quasi sempre.
    if punct in "!?":
        return True

    if punct == "…":
        return True

    # Da qui in poi: punto fermo.
    # Parola immediatamente precedente.
    j = i
    start = j
    while start > 0 and (text[start - 1].isalnum() or text[start - 1] in ".'’"):
        start -= 1
    prev = text[start:j].strip(".").lower()

    if prev in _ABBREV:
        return False
    # Iniziali puntate: "J. R. R. Tolkien", "F. lli".
    if len(prev) == 1 and prev.isalpha():
        return False
    # Numerazione di elenco ("3. Introduzione"): solo a inizio riga, altrimenti
    # "arrivo alle 3. Poi disse" verrebbe erroneamente tenuto insieme.
    if prev.isdigit() and len(prev) <= 3:
        line_start = text.rfind(chr(10), 0, start) + 1
        if not text[line_start:start].strip():
            return False

    # Se dopo lo spazio riparte in minuscolo, quasi sempre non era fine frase.
    k = end
    while k < len(text) and text[k].isspace():
        k += 1
    if k < len(text) and text[k].islower():
        return False
    return True


def _soft_split(chunk: str, base: int, out: list[tuple[int, int]]) -> None:
    """Spezza una frase troppo lunga su pause naturali, senza superare il limite."""
    if len(chunk) <= MAX_SENTENCE_CHARS:
        out.append((base, base + len(chunk)))
        return

    # Candidati in ordine di "forza" della pausa.
    for pattern in (r"[;:]\s", r"\s[—–-]\s", r",\s"):
        cuts = [m.end() for m in re.finditer(pattern, chunk)]
        # Il taglio piu' vicino alla meta' bilancia i due pezzi.
        mid = len(chunk) / 2
        cuts = [c for c in cuts if 20 < c < len(chunk) - 20]
        if cuts:
            cut = min(cuts, key=lambda c: abs(c - mid))
            _soft_split(chunk[:cut].rstrip(), base, out)
            left = len(chunk[:cut]) - len(chunk[:cut].rstrip())
            _soft_split(chunk[cut:], base + cut, out)
            return

    # Nessuna pausa utilizzabile: taglio netto sull'ultimo spazio disponibile.
    cut = chunk.rfind(" ", 0, MAX_SENTENCE_CHARS)
    if cut <= 0:
        cut = MAX_SENTENCE_CHARS
    _soft_split(chunk[:cut].rstrip(), base, out)
    _soft_split(chunk[cut:].lstrip(), base + cut + (len(chunk[cut:]) - len(chunk[cut:].lstrip())), out)


def split_sentences(text: str) -> list[tuple[int, int]]:
    """Offset (start, end) delle frasi; gli spazi tra una frase e l'altra sono esclusi."""
    spans: list[tuple[int, int]] = []
    pos = 0
    for m in _SENT_END_RE.finditer(text):
        if not _is_boundary(text, m.end()):
            continue
        chunk = text[pos:m.end()]
        stripped = chunk.strip()
        if stripped:
            lead = len(chunk) - len(chunk.lstrip())
            _soft_split(stripped, pos + lead, spans)
        pos = m.end()

    tail = text[pos:]
    if tail.strip():
        lead = len(tail) - len(tail.lstrip())
        _soft_split(tail.strip(), pos + lead, spans)

    return _merge_stubs(text, spans)


def _merge_stubs(text: str, spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Accorpa i frammenti di una o zero parole al pezzo precedente."""
    out: list[tuple[int, int]] = []
    for s, e in spans:
        n_words = len(_WORD_RE.findall(text[s:e]))
        if n_words == 0:
            continue
        if (n_words < MIN_SENTENCE_WORDS and out
                and (e - out[-1][0]) <= MAX_SENTENCE_CHARS):
            out[-1] = (out[-1][0], e)
        else:
            out.append((s, e))
    return out
