"""Avvio di AB Reader: server locale in un thread + finestra desktop."""

from __future__ import annotations

import argparse
import logging
import socket
import sys
import threading
import time
from pathlib import Path

import uvicorn

from abreader import config
from abreader.download_models import models_ready
from abreader.server import app


def _log_to_file_if_windowless() -> Path | None:
    """Sotto pythonw.exe non esiste una console: sys.stdout e sys.stderr sono None.

    Qualunque print, e qualunque traceback non catturato, finirebbe nel nulla e
    l'applicazione morirebbe muta. Si dirotta tutto su un file di log.
    """
    if sys.stdout is not None and sys.stderr is not None:
        return None
    config.ensure_dirs()
    path = config.DATA_DIR / "abreader.log"
    # Un avvio scrive tre righe, ma un errore ripetuto potrebbe gonfiare il file.
    if path.exists() and path.stat().st_size > 1_000_000:
        path.unlink(missing_ok=True)
    stream = open(path, "a", encoding="utf-8", buffering=1)
    sys.stdout = stream
    sys.stderr = stream
    return path


LOG_FILE = _log_to_file_if_windowless()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stderr,
    force=True,
)
log = logging.getLogger("abreader")


def _free_port(preferred: int) -> int:
    """La porta preferita se e' libera, altrimenti una qualsiasi."""
    with socket.socket() as s:
        try:
            s.bind((config.HOST, preferred))
            return preferred
        except OSError:
            pass
    with socket.socket() as s:
        s.bind((config.HOST, 0))
        return s.getsockname()[1]


def _wait_until_up(port: int, timeout: float = 25.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        with socket.socket() as s:
            s.settimeout(0.4)
            if s.connect_ex((config.HOST, port)) == 0:
                return True
        time.sleep(0.15)
    return False


def _serve(port: int) -> None:
    uvicorn.run(app, host=config.HOST, port=port, log_level="warning", access_log=False)


def main() -> int:
    parser = argparse.ArgumentParser(description="AB Reader")
    parser.add_argument("--browser", action="store_true",
                        help="apri nel browser predefinito invece che nella finestra desktop")
    parser.add_argument("--server-only", action="store_true",
                        help="avvia solo il server, senza aprire nulla")
    args = parser.parse_args()

    config.ensure_dirs()
    if not models_ready():
        log.warning("Modelli Kokoro assenti. Esegui: python -m abreader.download_models")

    port = _free_port(config.PORT)
    url = f"http://{config.HOST}:{port}/ui/index.html"

    if args.server_only:
        log.info("Server su %s", url)
        _serve(port)
        return 0

    threading.Thread(target=_serve, args=(port,), daemon=True).start()
    if not _wait_until_up(port):
        log.error("Il server non e' partito entro il tempo previsto.")
        return 1
    log.info("Pronto su %s", url)

    # Il primo caricamento del modello ONNX costa qualche secondo: farlo adesso
    # evita che la prima pressione di Play sembri bloccata.
    if models_ready():
        threading.Thread(target=_warmup, daemon=True).start()

    if args.browser:
        import webbrowser
        webbrowser.open(url)
        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            return 0

    import webview
    webview.create_window("AB Reader", url, width=1440, height=920,
                          min_size=(900, 600), text_select=True)
    webview.start()
    return 0


def _warmup() -> None:
    from abreader.tts import cache, engine
    cache.remove_legacy()
    engine.warmup()
    log.info("Motore vocale pronto.")


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except BaseException:
        log.exception("Avvio fallito")
        if LOG_FILE is not None:
            # Senza console il traceback e' gia' nel log: lo si segnala comunque
            # con una finestra di sistema, altrimenti l'utente non vede nulla.
            try:
                import ctypes
                ctypes.windll.user32.MessageBoxW(
                    None,
                    f"AB Reader non e' riuscito ad avviarsi.\n\n"
                    f"Dettagli in:\n{LOG_FILE}",
                    "AB Reader", 0x10)
            except Exception:
                pass
        raise
