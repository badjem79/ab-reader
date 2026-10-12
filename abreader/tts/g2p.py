"""Dal testo ai fonemi, nella stessa forma su cui Kokoro e' stato addestrato.

kokoro-onnx chiama espeak in modo semplificato: le affricate escono spezzate in
due simboli (``tʃ`` invece di ``ʧ``) e le virgolette basse vengono scartate.
Le voci non inglesi pero' hanno imparato i simboli prodotti dalla pipeline
ufficiale (misaki ``EspeakG2P``): qui la si riproduce, e i fonemi pronti
vengono passati al modello.

Nei testi italiani, inoltre, le parole inglesi vengono riconosciute e lette con
le regole dell'inglese invece che con quelle dell'italiano.

Ogni token del testo separato da spazi diventa un solo gruppo di fonemi
separato da spazi: e' la corrispondenza su cui si regge l'evidenziazione.
"""

from __future__ import annotations

import logging
import re
from functools import lru_cache
from pathlib import Path

from .. import config

log = logging.getLogger(__name__)

# Sostituzioni della pipeline ufficiale (misaki/espeak.py, EspeakG2P).
_E2M = sorted({
    "a^ɪ": "I", "a^ʊ": "W",
    "d^z": "ʣ", "d^ʒ": "ʤ",
    "e^ɪ": "A",
    "o^ʊ": "O", "ə^ʊ": "Q",
    "s^s": "S",
    "t^s": "ʦ", "t^ʃ": "ʧ",
    "ɔ^ɪ": "Y",
}.items())

# Pronuncia inglese britannica: e' quella che si studia a scuola, e senza la
# "r" americana suona piu' naturale in bocca a una voce italiana.
EN_IN_IT = "en-gb"

_HERE = Path(__file__).resolve().parent
ANGLICISMS_PATH = _HERE / "anglicismi.txt"
USER_DICT_PATH = config.DATA_DIR / "pronunce.txt"

_USER_DICT_TEMPLATE = """\
# Pronunce personalizzate di AB Reader.
#
# Una parola per riga, nella forma:   parola = pronuncia
#
#   cool = en          leggila in inglese
#   yogurt = it        leggila in italiano anche se sembra inglese
#   Hermione = ermaiòni   leggila come se fosse scritta cosi' (in italiano)
#   apple pie = en     anche piu' parole insieme (solo "en" o "it")
#
# Le maiuscole contano solo se scrivi la parola con la maiuscola: "Will" vale
# solo per "Will", "will" vale per entrambe. Le righe che iniziano con # sono
# ignorate. Le modifiche valgono dalla prossima frase sintetizzata; per
# rileggere con la nuova pronuncia l'audio gia' pronto, svuota la cache.
"""

_backends: dict[str, object] = {}

# Lettere "di parola" e apostrofi interni: il resto del token e' punteggiatura.
_CORE_RE = re.compile(r"^([^\w]*)(.*?)([^\w]*)$", re.UNICODE)
# Articoli e preposizioni elise in testa a una parola: "l'happy", "dell'hotel".
_ELISION_RE = re.compile(
    r"^((?:[lLdDcCsSmMtTvVnN]|[a-zA-Z]{2,}[lL]|[uU]n|[qQ]uest|[sS]ant|[bB]ell)['’])(.+)$")
_ITALIAN_ACCENTS = set("àèéìíòóùú")
_DASH_RE = re.compile(r"(?<!\S)[–-](?!\S)")


# --- espeak -----------------------------------------------------------------

def _backend(code: str):
    """Un backend per lingua, creato una volta sola (inizializzare espeak costa)."""
    b = _backends.get(code)
    if b is None:
        from phonemizer.backend import EspeakBackend
        b = EspeakBackend(code, preserve_punctuation=True, with_stress=True,
                          tie="^", language_switch="remove-flags")
        _backends[code] = b
    return b


def espeak(text: str, code: str) -> str:
    """Fonemi di `text` come li produce la pipeline ufficiale di Kokoro."""
    text = text.replace("«", "“").replace("»", "”")
    text = text.replace("(", "«").replace(")", "»")
    # Il trattino d'inciso ("lavoro – ci faranno") si perderebbe: il modello
    # conosce solo la lineetta lunga, che oltretutto gli fa fare la pausa.
    text = _DASH_RE.sub("—", text)
    out = _backend(code).phonemize([text])
    ps = out[0].strip() if out else ""
    for old, new in _E2M:
        ps = ps.replace(old, new)
    ps = ps.replace("^", "").replace("-", "")
    return ps.replace("«", "(").replace("»", ")")


# --- dizionari --------------------------------------------------------------

@lru_cache(maxsize=1)
def _anglicisms() -> dict[str, frozenset[str]]:
    """Le sezioni dell'elenco incorporato: inglese, frasi, ambigue, eccezioni."""
    sections: dict[str, set[str]] = {
        "inglese": set(), "frasi": set(), "ambigue": set(), "eccezioni": set()}
    current = "inglese"
    for line in ANGLICISMS_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            current = line[1:-1].strip().lower()
            continue
        sections.setdefault(current, set()).update(w.lower() for w in line.split())
    return {k: frozenset(v) for k, v in sections.items()}


_user_dict: dict[str, str] = {}
_user_dict_mtime: float | None = None


def user_dict() -> dict[str, str]:
    """Il dizionario personale, riletto solo se il file e' cambiato."""
    global _user_dict, _user_dict_mtime
    path = USER_DICT_PATH
    try:
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(_USER_DICT_TEMPLATE, encoding="utf-8")
        mtime = path.stat().st_mtime
    except OSError:
        return _user_dict
    if mtime == _user_dict_mtime:
        return _user_dict

    entries: dict[str, str] = {}
    for n, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        word, sep, value = line.partition("=")
        word, value = word.strip(), value.strip()
        if not sep or not word or not value:
            log.warning("pronunce.txt, riga %d ignorata: %r", n, line)
            continue
        entries[word] = value
    _user_dict, _user_dict_mtime = entries, mtime
    log.info("Dizionario delle pronunce: %d voci", len(entries))
    return entries


def _lookup(word: str, entries: dict[str, str]) -> str | None:
    """Voce del dizionario personale: prima la forma esatta, poi la minuscola."""
    return entries.get(word) or entries.get(word.lower())


def _phrases(entries: dict[str, str]) -> list[tuple[list[str], str, str]]:
    """Le voci di piu' parole: (parole in minuscolo, lingua, voce originale)."""
    out = []
    for key, value in entries.items():
        words = key.lower().split()
        if len(words) > 1 and value.lower() in ("en", "it"):
            out.append((words, value.lower(), key))
    return out


def _bare(token: str) -> str:
    """Il token senza punteggiatura ne' elisione, in minuscolo, per i confronti."""
    core = _CORE_RE.match(token).group(2)
    m = _ELISION_RE.match(core)
    return (m.group(2) if m else core).lower().replace("’", "'")


def signature(text: str, code: str) -> str:
    """Le scelte di pronuncia della frase che non sono "italiano normale".

    Entra nella chiave della cache audio: correggere una pronuncia, nel
    dizionario personale o nell'elenco incorporato, rifa' solo le frasi in cui
    quella scelta cambia. Non chiama espeak, quindi costa poco.
    """
    if code != "it":
        return ""
    words = _languages(text.split())
    return "|".join(f"{n}:{p}{w}={k}" for n, (p, w, k) in enumerate(words) if k != "it")


# --- riconoscimento dell'inglese ---------------------------------------------

def _looks_english(word: str) -> bool:
    """Grafie che in italiano non esistono. Prudente: nel dubbio, italiano."""
    w = word.lower()
    if len(w) < 3 or any(c in _ITALIAN_ACCENTS for c in w) or not w.isascii():
        return False
    if w.endswith("ing") and len(w) > 4:
        return True
    # "ee" solo prima di consonanti finali: "idee", "europee" sono italiane.
    if re.search(r"th|(?<!c)sh|ee[^aeiou]+$|ck|ght|oo[^aeiou]$|ea[^aeiou]$|ay$|ey$|ow$", w):
        return True
    # La "w" e' anche tedesca (Wagner, walzer): vale solo per le minuscole.
    return "w" in w and word.islower()


def _classify(core: str, entries: dict[str, str]) -> str:
    """Lingua di una parola, o la grafia personalizzata con cui leggerla.

    'en' inglese, 'en!' inglese tipica di una frase, '?' ambigua (decide il
    contesto), 'it' italiano; qualsiasi altra cosa e' una grafia da leggere.
    """
    custom = _lookup(core, entries)
    if custom:
        return custom if custom.lower() not in ("en", "it") else custom.lower()
    if not core or not any(c.isalpha() for c in core):
        return "it"
    lists = _anglicisms()
    low = core.lower().replace("’", "'")
    if low in lists["eccezioni"]:
        return "it"
    if low in lists["inglese"]:
        return "en"
    # Sigle (TV, USA, DVD): in italiano si leggono all'italiana.
    if len(core) > 1 and core.isupper():
        return "it"
    if low in lists["frasi"]:
        return "en!"
    if low in lists["ambigue"]:
        return "?"
    return "en" if _looks_english(core) else "it"


def _listed(core: str, entries: dict[str, str]) -> bool:
    """La parola intera compare nel dizionario personale o nell'elenco incorporato."""
    low = core.lower().replace("’", "'")
    return bool(_lookup(core, entries)) or any(low in v for v in _anglicisms().values())


def _compound(core: str, entries: dict[str, str]):
    """Parole col trattino fatte di pezzi di lingue diverse ("cyber-futurismo").

    Restituisce le coppie (pezzo, lingua o grafia), o None se la parola si
    legge tutta in una lingua.
    """
    if "-" not in core or _listed(core, entries):
        return None
    parts = [p for p in core.split("-") if p]
    kinds = []
    for part in parts:
        k = _classify(part, entries)
        kinds.append({"en!": "en", "?": "it"}.get(k, k))
    if len(parts) < 2 or len(set(kinds)) < 2:
        return None
    return tuple(zip(parts, kinds))


def _languages(tokens: list[str]) -> list[tuple[str, str, object]]:
    """Per ogni token: (prefisso eliso, parola, lingua o grafia).

    La lingua di una parola composta mista e' la tupla dei suoi pezzi.
    """
    entries = user_dict()
    out = []
    for tok in tokens:
        lead, core, trail = _CORE_RE.match(tok).groups()
        prefix = ""
        if core.endswith(("'", "’")):
            out.append(("", tok, "it"))  # elisione italiana: "dell'", "quell'"
            continue
        m = _ELISION_RE.match(core)
        if m and _classify(core, entries) == "it":
            # "l'happy": l'articolo resta italiano, la parola si valuta da sola.
            if _classify(m.group(2), entries) not in ("it", "?"):
                prefix, core = lead + m.group(1), m.group(2)
                lead = ""
        kind = _classify(core, entries)
        if kind in ("it", "en"):
            kind = _compound(core, entries) or kind
        out.append((prefix, lead + core + trail, kind))

    # Le ambigue ("me", "so", "come") sono inglesi solo dentro una frase inglese,
    # cioe' in una sequenza di parole inglesi che ne contiene almeno una tipica
    # delle frasi: "Oh my God, come on" si', "come manager" no.
    kinds = [k for _, _, k in out]
    i = 0
    while i < len(kinds):
        if kinds[i] not in ("en", "en!", "?"):
            i += 1
            continue
        j = i
        while j < len(kinds) and kinds[j] in ("en", "en!", "?"):
            j += 1
        english = "en!" in kinds[i:j]
        for n in range(i, j):
            if kinds[n] == "?":
                kinds[n] = "en" if english else "it"
            elif kinds[n] == "en!":
                kinds[n] = "en"
        i = j

    # Le espressioni del dizionario personale ("apple pie") hanno l'ultima parola.
    bare = [_bare(t) for t in tokens]
    for words, value, _ in _phrases(entries):
        n = len(words)
        for i in range(len(bare) - n + 1):
            if bare[i:i + n] == words:
                kinds[i:i + n] = [value] * n
    return [(p, w, k) for (p, w, _), k in zip(out, kinds)]


def _prefix_phonemes(prefix: str) -> str:
    """Fonemi di un articolo eliso letto da solo ("l'" non deve diventare "elle")."""
    bare = prefix.rstrip("'’").lower()
    if len(bare) == 1:
        return {"c": "ʧ", "g": "ʤ"}.get(bare, bare)
    return espeak(bare, "it").replace("ˈ", "").replace("ˌ", "")


# --- interfaccia ------------------------------------------------------------

def phonemize(text: str, code: str, vocab: dict | None = None) -> str:
    """Fonemi pronti per il modello: un gruppo per ogni token del testo."""
    if code == "it":
        ps = _phonemize_italian(text)
    else:
        ps = espeak(text, code)
    if vocab is not None:
        ps = "".join(p for p in ps if p in vocab)
    return " ".join(ps.split())


def _single(word: str, code: str) -> str:
    """Una parola sola, sempre in un unico gruppo di fonemi."""
    return "".join(espeak(word, code).split())


def _phonemize_italian(text: str) -> str:
    """Tutta la frase in italiano, poi le parole inglesi sostituite al loro posto.

    Fonetizzare la frase intera, e non a pezzi, conta: espeak accentua l'ultima
    parola di ogni pezzo, e un "il" prima di "weekend" diventerebbe "ìl".
    """
    tokens = text.split()
    if not tokens:
        return ""
    words = _languages(tokens)
    if all(kind == "it" for _, _, kind in words):
        return espeak(text, "it")

    base = espeak(" ".join(w for _, w, _ in words), "it").split()
    if len(base) != len(tokens):
        # espeak ha espanso qualcosa (numeri, sigle): si procede parola per parola.
        base = [_single(w, "it") if k == "it" else "" for _, w, k in words]

    groups = list(base)
    i = 0
    while i < len(words):
        prefix, word, kind = words[i]
        if kind == "it":
            i += 1
            continue
        if kind == "en" and not prefix:
            # Le parole inglesi consecutive si leggono insieme, come una frase.
            j = i
            while j < len(words) and words[j][2] == "en" and not words[j][0]:
                j += 1
            run = [w for _, w, _ in words[i:j]]
            eng = espeak(" ".join(run), EN_IN_IT).split()
            if len(eng) != len(run):
                eng = [_single(w, EN_IN_IT) for w in run]
            groups[i:j] = eng
            i = j
            continue
        if isinstance(kind, tuple):
            lead, _, trail = _CORE_RE.match(word).groups()
            pieces = []
            for n, (part, k) in enumerate(kind):
                text_ = (lead if n == 0 else "") + part + (trail if n == len(kind) - 1 else "")
                if k == "en":
                    pieces.append(_single(text_, EN_IN_IT))
                elif k == "it":
                    pieces.append(_single(text_, "it"))
                else:
                    pieces.append(_single(text_.replace(part, k), "it"))
            body = "".join(pieces)
        elif kind == "en":
            body = _single(word, EN_IN_IT)
            # "l'happy hour": dopo l'elisione l'acca non si aspira.
            if body.startswith("h"):
                body = body[1:]
        else:
            # Grafia personalizzata: anche se e' di piu' parole resta un solo gruppo.
            lead, _, trail = _CORE_RE.match(word).groups()
            body = _single(lead + kind + trail, "it")
        groups[i] = (_prefix_phonemes(prefix) if prefix else "") + body
        i += 1
    return " ".join(g for g in groups if g)
