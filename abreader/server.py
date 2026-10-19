"""API HTTP locale del lettore."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import Body, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from . import config, epub_parser, keepawake, store, textseg
from .download_models import models_ready
from .tts import cache as tts_cache
from .tts import engine as tts_engine

log = logging.getLogger(__name__)

app = FastAPI(title="AB Reader", docs_url=None, redoc_url=None)

DEFAULT_SETTINGS: dict[str, Any] = {
    "voice_it": config.DEFAULT_VOICES["it"],
    "voice_en": config.DEFAULT_VOICES["en"],
    "speed": 1.0,
    "font_size": 19,
    "theme": "sepia",
    "auto_scroll": True,
}


def _settings() -> dict[str, Any]:
    return {**DEFAULT_SETTINGS, **store.get_settings()}


def _voice_for(book: dict[str, Any], override: str | None = None) -> str:
    if override:
        return override
    s = _settings()
    return s["voice_it"] if book.get("language") == "it" else s["voice_en"]


def _chapter(book_id: str, idx: int) -> tuple[dict, dict]:
    try:
        parsed = epub_parser.load(book_id)
    except FileNotFoundError:
        raise HTTPException(404, "Libro non trovato")
    chapters = parsed["chapters"]
    if not 0 <= idx < len(chapters):
        raise HTTPException(404, "Capitolo non trovato")
    return parsed, chapters[idx]


# --- stato generale ---------------------------------------------------------

@app.get("/api/state")
def get_state() -> dict[str, Any]:
    ready = models_ready()
    voices: list[dict[str, str]] = []
    if ready:
        try:
            voices = tts_engine.list_voices()
        except Exception as exc:
            log.warning("Elenco voci non disponibile: %s", exc)
    return {
        "models_ready": ready,
        "voices": voices,
        "settings": _settings(),
        "cache_mb": tts_cache.size_mb(),
    }


@app.post("/api/settings")
def post_settings(values: dict[str, Any] = Body(...)) -> dict[str, Any]:
    allowed = {k: v for k, v in values.items() if k in DEFAULT_SETTINGS}
    if allowed:
        store.set_settings(allowed)
    return _settings()


@app.get("/api/cache")
def get_cache() -> dict[str, Any]:
    return {"cache_mb": tts_cache.size_mb()}


@app.post("/api/cache/clear")
def clear_cache() -> dict[str, Any]:
    freed = tts_cache.clear()
    return {"freed_mb": freed, "cache_mb": tts_cache.size_mb()}


@app.post("/api/books/{book_id}/cache/clear")
def clear_book_cache(book_id: str) -> dict[str, Any]:
    if not store.get_book(book_id):
        raise HTTPException(404, "Libro non in libreria")
    freed = tts_cache.clear(book_id)
    return {"freed_mb": freed, "cache_mb": tts_cache.size_mb()}


# --- libreria ---------------------------------------------------------------

@app.get("/api/books")
def get_books() -> list[dict[str, Any]]:
    return [{**b, "cache_mb": tts_cache.size_mb(b["id"])} for b in store.list_books()]


@app.post("/api/books")
async def post_book(file: UploadFile = File(...)) -> dict[str, Any]:
    if not (file.filename or "").lower().endswith(".epub"):
        raise HTTPException(400, "Per ora sono supportati solo i file .epub")
    data = await file.read()
    if not data:
        raise HTTPException(400, "File vuoto")
    try:
        parsed = epub_parser.import_epub(data, file.filename)
    except Exception as exc:
        log.exception("Import fallito")
        raise HTTPException(400, f"Non riesco a leggere questo epub: {exc}")
    if not parsed["chapters"]:
        raise HTTPException(400, "Nessun testo leggibile trovato nel file")

    store.upsert_book(parsed["id"], parsed["filename"], parsed["title"],
                      parsed["author"], parsed["language"], len(parsed["chapters"]))
    return {"id": parsed["id"], "title": parsed["title"], "author": parsed["author"],
            "language": parsed["language"], "n_chapters": len(parsed["chapters"])}


@app.get("/api/books/{book_id}")
def get_book(book_id: str) -> dict[str, Any]:
    row = store.get_book(book_id)
    if not row:
        raise HTTPException(404, "Libro non in libreria")
    try:
        parsed = epub_parser.load(book_id)
    except FileNotFoundError:
        raise HTTPException(404, "File epub mancante")
    store.touch_book(book_id)
    return {
        "id": book_id,
        "title": parsed["title"],
        "author": parsed["author"],
        "language": parsed["language"],
        "voice": _voice_for(parsed),
        "chapters": [{"index": c["index"], "title": c["title"],
                      "n_words": c["n_words"], "n_sentences": len(c["sentences"])}
                     for c in parsed["chapters"]],
        "progress": store.get_progress(book_id),
    }


@app.delete("/api/books/{book_id}")
def remove_book(book_id: str) -> dict[str, bool]:
    store.delete_book(book_id)
    epub_parser.remove(book_id)
    tts_cache.clear(book_id)
    return {"ok": True}


@app.get("/api/books/{book_id}/chapters/{idx}")
def get_chapter(book_id: str, idx: int) -> dict[str, Any]:
    parsed, chapter = _chapter(book_id, idx)
    return {
        "index": chapter["index"],
        "title": chapter["title"],
        "blocks": chapter["blocks"],
        "sentences": chapter["sentences"],
        "n_chapters": len(parsed["chapters"]),
    }


@app.post("/api/books/{book_id}/progress")
def post_progress(book_id: str, body: dict[str, Any] = Body(...)) -> dict[str, bool]:
    store.set_progress(book_id, int(body.get("chapter_idx", 0)),
                       int(body.get("sentence_idx", 0)), int(body.get("word_idx", 0)))
    return {"ok": True}


# --- note -------------------------------------------------------------------

@app.get("/api/books/{book_id}/notes")
def get_notes(book_id: str, chapter: int | None = None) -> list[dict[str, Any]]:
    return store.list_notes(book_id, chapter)


@app.post("/api/books/{book_id}/notes")
def post_note(book_id: str, body: dict[str, Any] = Body(...)) -> dict[str, Any]:
    text = (body.get("body") or "").strip()
    if not text:
        raise HTTPException(400, "Nota vuota")
    return store.add_note(book_id, int(body.get("chapter_idx", 0)),
                          body.get("sentence_idx"), (body.get("anchor") or "")[:300],
                          text)


@app.put("/api/notes/{note_id}")
def put_note(note_id: int, body: dict[str, Any] = Body(...)) -> dict[str, bool]:
    store.update_note(note_id, (body.get("body") or "").strip())
    return {"ok": True}


@app.delete("/api/notes/{note_id}")
def remove_note(note_id: int) -> dict[str, bool]:
    store.delete_note(note_id)
    return {"ok": True}


# --- sintesi ----------------------------------------------------------------

def _group_payload(book_id: str, chapter_idx: int, sentence_idx: int,
                   voice: str | None, speed: float | None
                   ) -> tuple[list[tuple[str, list[list[int]]]], int, str, float, str]:
    """Il gruppo di sintesi della frase, la sua posizione nel gruppo e i parametri."""
    parsed, chapter = _chapter(book_id, chapter_idx)
    sentences = chapter["sentences"]
    if not 0 <= sentence_idx < len(sentences):
        raise HTTPException(404, "Frase non trovata")
    start, end = textseg.group_of(sentences, sentence_idx)
    group = [(s["t"], s["w"]) for s in sentences[start:end]]
    v = _voice_for(parsed, voice)
    sp = float(speed if speed is not None else _settings()["speed"])
    return group, sentence_idx - start, v, sp, parsed["language"]


@app.post("/api/tts")
def post_tts(body: dict[str, Any] = Body(...)) -> dict[str, Any]:
    """Sintetizza una frase, o la riprende dalla cache, con i tempi delle parole."""
    idx = int(body["sentence_idx"])
    group, target, voice, speed, lang = _group_payload(
        body["book_id"], int(body["chapter_idx"]), idx,
        body.get("voice"), body.get("speed"))
    try:
        meta = tts_cache.get(body["book_id"], group, target, voice, speed, lang)
    except Exception as exc:
        log.exception("Sintesi fallita")
        raise HTTPException(500, f"Sintesi fallita: {exc}")
    return {**meta, "sentence_idx": idx,
            "audio_url": f"/api/audio/{body['book_id']}/{meta['key']}.opus"}


@app.post("/api/tts/prefetch")
def post_prefetch(body: dict[str, Any] = Body(...)) -> dict[str, Any]:
    """Mette in coda la sintesi delle frasi successive senza attenderla."""
    keys: list[str] = []
    for idx in body.get("sentence_idx", [])[:8]:
        try:
            group, _, voice, speed, lang = _group_payload(
                body["book_id"], int(body["chapter_idx"]), int(idx),
                body.get("voice"), body.get("speed"))
        except HTTPException:
            break
        # Frasi dello stesso gruppo portano allo stesso lavoro: si fonde in coda.
        queued = tts_cache.prefetch(body["book_id"], group, voice, speed, lang)
        keys += [k for k in queued if k not in keys]
    return {"queued": keys}


@app.post("/api/keepawake")
def post_keepawake(body: dict[str, Any] = Body(...)) -> dict[str, bool]:
    """Schermo acceso durante la lettura: l'interfaccia rinnova finche' legge."""
    if body.get("on"):
        keepawake.hold()
    else:
        keepawake.release()
    return {"ok": True}


@app.get("/api/audio/{book_id}/{key}.opus")
def get_audio(book_id: str, key: str) -> FileResponse:
    if not (key.isalnum() and book_id.isalnum()):
        raise HTTPException(400, "Chiave non valida")
    audio, _ = tts_cache.paths(book_id, key)
    if not audio.exists():
        raise HTTPException(404, "Audio non ancora in cache")
    return FileResponse(audio, media_type="audio/ogg",
                        headers={"Cache-Control": "public, max-age=86400"})


# --- statici ----------------------------------------------------------------

@app.get("/")
def index() -> RedirectResponse:
    return RedirectResponse("/ui/index.html")


app.mount("/ui", StaticFiles(directory=str(config.WEB_DIR), html=True), name="ui")
