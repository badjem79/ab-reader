"""Tiene acceso lo schermo mentre il lettore legge.

Su Windows lo salvaschermo (e con esso il blocco della sessione) parte dopo un
tempo di inattivita' di tastiera e mouse: ascoltando un libro non si tocca
nulla, e dopo pochi minuti lo schermo si blocca. WebView2 chiede a Windows di
tenere acceso lo schermo solo per i video, non per l'audio.

Si usa quindi una richiesta di alimentazione esplicita (PowerSetRequest), la
stessa delle presentazioni e dei lettori video; compare in `powercfg /requests`.

La richiesta e' un prestito a tempo: l'interfaccia la rinnova mentre legge, e
se smette di farlo (pausa, finestra chiusa, browser chiuso in modalita'
--browser) scade da sola. Lo schermo non resta mai acceso per sbaglio.
"""

from __future__ import annotations

import ctypes
import logging
import sys
import threading

log = logging.getLogger(__name__)

LEASE_SECONDS = 150

_POWER_REQUEST_DISPLAY_REQUIRED = 0
_POWER_REQUEST_SYSTEM_REQUIRED = 1
_POWER_REQUEST_CONTEXT_SIMPLE_STRING = 0x1
_INVALID_HANDLE = ctypes.c_void_p(-1).value

_lock = threading.Lock()
_handle: int | None = None
_active = False
_timer: threading.Timer | None = None


class _ReasonContext(ctypes.Structure):
    _fields_ = [("Version", ctypes.c_ulong),
                ("Flags", ctypes.c_ulong),
                ("SimpleReasonString", ctypes.c_wchar_p)]


def _kernel32():
    k = ctypes.windll.kernel32
    k.PowerCreateRequest.restype = ctypes.c_void_p
    k.PowerCreateRequest.argtypes = [ctypes.POINTER(_ReasonContext)]
    k.PowerSetRequest.argtypes = [ctypes.c_void_p, ctypes.c_int]
    k.PowerClearRequest.argtypes = [ctypes.c_void_p, ctypes.c_int]
    return k


def _request() -> int | None:
    """La richiesta di alimentazione, creata alla prima occorrenza."""
    global _handle
    if _handle is None:
        # La stringa deve restare viva finche' vive la richiesta.
        _request.reason = ctypes.c_wchar_p("AB Reader: lettura ad alta voce in corso")
        ctx = _ReasonContext(0, _POWER_REQUEST_CONTEXT_SIMPLE_STRING, _request.reason)
        h = _kernel32().PowerCreateRequest(ctypes.byref(ctx))
        if not h or h == _INVALID_HANDLE:
            raise OSError(ctypes.get_last_error(), "PowerCreateRequest fallita")
        _handle = h
    return _handle


def _set(on: bool) -> None:
    global _active
    if on == _active:
        return
    k, h = _kernel32(), _request()
    for kind in (_POWER_REQUEST_DISPLAY_REQUIRED, _POWER_REQUEST_SYSTEM_REQUIRED):
        (k.PowerSetRequest if on else k.PowerClearRequest)(h, kind)
    _active = on
    log.debug("Schermo %s", "tenuto acceso" if on else "di nuovo libero di spegnersi")


def hold(seconds: float = LEASE_SECONDS) -> None:
    """Tiene acceso lo schermo per `seconds`; chiamarla di nuovo rinnova il prestito."""
    global _timer
    if sys.platform != "win32":
        return
    with _lock:
        try:
            _set(True)
        except Exception as exc:  # non deve mai disturbare la lettura
            log.warning("Impossibile tenere acceso lo schermo: %s", exc)
            return
        if _timer is not None:
            _timer.cancel()
        _timer = threading.Timer(seconds, lambda: _expire(timer))
        timer = _timer
        _timer.daemon = True
        _timer.start()


def _expire(timer: threading.Timer) -> None:
    # Un timer gia' scattato mentre hold() lo sostituiva non deve spegnere
    # il prestito appena rinnovato.
    with _lock:
        if _timer is timer:
            _release_locked()


def release() -> None:
    """Restituisce a Windows il controllo dello schermo."""
    if sys.platform != "win32":
        return
    with _lock:
        _release_locked()


def _release_locked() -> None:
    global _timer
    if _timer is not None:
        _timer.cancel()
        _timer = None
    try:
        _set(False)
    except Exception as exc:
        log.warning("Impossibile rilasciare lo schermo: %s", exc)
