"""Percorsi e impostazioni globali dell'applicazione."""

from __future__ import annotations

import os
from pathlib import Path

# La root del progetto e' la cartella che contiene il package.
ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = Path(os.environ.get("ABREADER_DATA", ROOT / "data"))
MODELS_DIR = DATA_DIR / "models"
BOOKS_DIR = DATA_DIR / "books"
CACHE_DIR = DATA_DIR / "cache"
DB_PATH = DATA_DIR / "abreader.sqlite3"
WEB_DIR = ROOT / "web"

MODEL_PATH = MODELS_DIR / "kokoro-v1.0.onnx"
VOICES_PATH = MODELS_DIR / "voices-v1.0.bin"

# La release v1.1 espone anche l'uscita "duration": e' quella che permette
# di sapere esattamente quando viene pronunciato ogni fonema.
MODEL_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.1/kokoro-v1.0.onnx"
VOICES_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.1/voices-v1.0.bin"

SAMPLE_RATE = 24000

# Tetto della cache audio: superato, si cancella l'audio ascoltato meno di recente.
TTS_CACHE_LIMIT_MB = int(os.environ.get("ABREADER_CACHE_MB", "100"))

# Voci predefinite per lingua; usate quando l'utente non ne ha ancora scelta una.
DEFAULT_VOICES = {
    "en": "af_heart",
    "it": "if_sara",
}

HOST = "127.0.0.1"
PORT = int(os.environ.get("ABREADER_PORT", "8731"))


def ensure_dirs() -> None:
    for d in (DATA_DIR, MODELS_DIR, BOOKS_DIR, CACHE_DIR):
        d.mkdir(parents=True, exist_ok=True)
