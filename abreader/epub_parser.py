"""Estrazione del testo da un EPUB, in blocchi -> frasi -> parole.

Il risultato viene messo in cache come JSON: l'analisi di un romanzo richiede
qualche secondo e non cambia mai a parita' di file.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import warnings
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup, Tag, XMLParsedAsHTMLWarning

from . import config, textseg

# ebooklib emette FutureWarning rumorosi su ogni apertura.
warnings.filterwarnings("ignore", category=FutureWarning, module="ebooklib.*")
warnings.filterwarnings("ignore", category=UserWarning, module="ebooklib.*")
# Gli xhtml degli epub sono XML validi, ma il parser html di lxml li gestisce
# meglio quando il markup e' sciatto: l'avviso di bs4 qui non aggiunge nulla.
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

import ebooklib  # noqa: E402
from ebooklib import epub  # noqa: E402

# Elementi che producono un blocco di testo a se stante.
_TEXT_BLOCKS = {"p", "h1", "h2", "h3", "h4", "h5", "h6", "li", "pre",
                "blockquote", "dd", "dt", "figcaption", "td", "th", "div"}
_DROP = {"script", "style", "head", "nav", "svg", "audio", "video"}
_HEADINGS = {"h1", "h2", "h3", "h4", "h5", "h6"}

CACHE_VERSION = 3


def book_id_for(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()[:16]


# Segnaposto per gli a capo veri (<br>). Serve a distinguerli dagli a capo di
# semplice formattazione del sorgente xhtml, che vanno trattati come spazi:
# altrimenti finirebbero dentro il testo delle frasi.
_BR = chr(0x2028)  # LINE SEPARATOR: non compare nei testi reali


def _block_text(el: Tag) -> str:
    """Testo di un blocco: i <br> diventano a capo, ogni altro spazio si normalizza."""
    for br in el.find_all("br"):
        br.replace_with(_BR)
    raw = el.get_text()
    lines = [" ".join(line.split()) for line in raw.split(_BR)]
    return "\n".join(line for line in lines if line).strip()


def _extract_blocks(html: str) -> list[dict[str, str]]:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup.find_all(_DROP):
        tag.decompose()
    body = soup.body or soup

    blocks: list[dict[str, str]] = []
    candidates = body.find_all(_TEXT_BLOCKS)
    for el in candidates:
        # Solo i blocchi "foglia": un <div> che contiene <p> non deve duplicarli.
        if el.find(_TEXT_BLOCKS):
            continue
        text = _block_text(el)
        if not text:
            continue
        tag = el.name if el.name in _HEADINGS else ("blockquote" if el.name == "blockquote" else "p")
        blocks.append({"tag": tag, "text": text})

    if not blocks:
        # Documenti senza markup di blocco: si ripiega sul testo grezzo.
        text = _block_text(body)
        blocks = [{"tag": "p", "text": line} for line in text.split("\n") if line.strip()]
    return blocks


def _segment(blocks: list[dict[str, str]]) -> tuple[list[dict], list[dict]]:
    """Trasforma i blocchi in frasi numerate; ogni blocco tiene gli id delle sue frasi."""
    out_blocks: list[dict] = []
    sentences: list[dict] = []
    for b in blocks:
        text = b["text"]
        ids: list[int] = []
        for s, e in textseg.split_sentences(text):
            sent = text[s:e]
            words = textseg.split_words(sent)
            if not words:
                continue
            ids.append(len(sentences))
            sentences.append({"t": sent, "w": [[ws, we] for ws, we in words]})
        if ids:
            out_blocks.append({"tag": b["tag"], "s": ids})
    return out_blocks, sentences


def _toc_titles(book: epub.EpubBook) -> dict[str, str]:
    """Mappa href del documento -> titolo, ricavata dall'indice."""
    titles: dict[str, str] = {}

    def walk(items):
        for item in items:
            if isinstance(item, tuple):
                section, children = item
                if getattr(section, "href", None):
                    titles.setdefault(section.href.split("#")[0], section.title)
                walk(children)
            elif isinstance(item, epub.Link):
                titles.setdefault(item.href.split("#")[0], item.title)

    try:
        walk(book.toc)
    except Exception:
        pass
    return titles


def parse(path: Path) -> dict[str, Any]:
    """Analizza l'epub e restituisce metadati + capitoli segmentati."""
    book = epub.read_epub(str(path))

    def meta(name: str, default: str = "") -> str:
        try:
            values = book.get_metadata("DC", name)
            return values[0][0] if values else default
        except Exception:
            return default

    titles = _toc_titles(book)
    chapters: list[dict[str, Any]] = []
    full_text_parts: list[str] = []

    # L'ordine di lettura e' quello dello spine, non quello dei file nel container.
    for spine_id, _ in book.spine:
        item = book.get_item_with_id(spine_id)
        if item is None or item.get_type() != ebooklib.ITEM_DOCUMENT:
            continue
        try:
            html = item.get_content().decode("utf-8", errors="replace")
        except Exception:
            continue
        blocks = _extract_blocks(html)
        out_blocks, sentences = _segment(blocks)
        if not sentences:
            continue  # copertine, pagine vuote, separatori

        href = item.get_name()
        title = titles.get(href) or titles.get(href.split("/")[-1]) or ""
        if not title:
            head = next((b for b in blocks if b["tag"] in _HEADINGS), None)
            title = head["text"] if head else f"Capitolo {len(chapters) + 1}"
        title = " ".join(title.split())[:120]

        n_words = sum(len(s["w"]) for s in sentences)
        chapters.append({
            "index": len(chapters),
            "title": title,
            "href": href,
            "blocks": out_blocks,
            "sentences": sentences,
            "n_words": n_words,
        })
        full_text_parts.append(" ".join(s["t"] for s in sentences[:40]))

    language = textseg.detect_language(" ".join(full_text_parts), meta("language"))

    return {
        "version": CACHE_VERSION,
        "title": " ".join(meta("title", path.stem).split()) or path.stem,
        "author": " ".join(meta("creator").split()),
        "language": language,
        "chapters": chapters,
    }


def _cache_path(book_id: str) -> Path:
    return config.CACHE_DIR / book_id / "book.json"


def import_epub(data: bytes, filename: str) -> dict[str, Any]:
    """Salva l'epub nella libreria, lo analizza e restituisce i metadati."""
    config.ensure_dirs()
    book_id = book_id_for(data)
    dest = config.BOOKS_DIR / f"{book_id}.epub"
    if not dest.exists():
        dest.write_bytes(data)

    parsed = parse(dest)
    parsed["id"] = book_id
    parsed["filename"] = filename
    cache = _cache_path(book_id)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(parsed, ensure_ascii=False), encoding="utf-8")
    return parsed


def load(book_id: str) -> dict[str, Any]:
    """Carica il libro analizzato, rifacendo l'analisi se la cache e' vecchia o assente."""
    cache = _cache_path(book_id)
    if cache.exists():
        try:
            data = json.loads(cache.read_text(encoding="utf-8"))
            if data.get("version") == CACHE_VERSION:
                return data
        except Exception:
            pass

    src = config.BOOKS_DIR / f"{book_id}.epub"
    if not src.exists():
        raise FileNotFoundError(f"epub non trovato per il libro {book_id}")
    parsed = parse(src)
    parsed["id"] = book_id
    parsed["filename"] = src.name
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(parsed, ensure_ascii=False), encoding="utf-8")
    return parsed


def remove(book_id: str) -> None:
    (config.BOOKS_DIR / f"{book_id}.epub").unlink(missing_ok=True)
    shutil.rmtree(config.CACHE_DIR / book_id, ignore_errors=True)
