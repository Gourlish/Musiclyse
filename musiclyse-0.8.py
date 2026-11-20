"""
Musiclyse — analyse songs with a multi-step audio pipeline, then chat about
them with a local LLM (via Ollama).

How it fits together (0.6.2-dev4 and later):
    1. Analyse:  /analyse /path/to/song.m4a      one file  -> saved-songs/Artist - Title.json
                 /reanalyse /path/to/song.m4a    the same, replacing the file's save
                 /batch /path/to/folder          every audio file in a folder
    2. Export:   run the Musiclyse export script on a saved .json (outside the chat)
                 to make a .txt for chatting -- any level, with or without Wikipedia
                 background, one song or several in one file.
    3. Chat:     /load "Artist - Title" [question]           one export (or a combined one)
                 /load "Artist 1 - Title 1" "Artist 2 - Title 2" ... [question]
                                                             several at once, e.g. to compare
                 /lookup  (anywhere in a prompt, also with /load) looks up
                          background in the local Wikipedia database for that prompt,
                          skipping articles a loaded export already carries.
                 /lookup-search "Title" ["Title 2" ...] [question]   (0.7-pre62) adds
                          Wikipedia articles chosen by title; a menu when a phrase
                          is in several titles.
    Anything else goes straight to the chat model, with the loaded exports in the
    conversation. An image file path (drag the file into the window) or web
    address in any prompt is shown to the chat model as the picture itself.

Other commands:
    /persona <description>   Chat as a custom character; /persona reset for the default.
    /clear                   Clear the conversation; loaded exports and looked-up
                             background are kept and sent again.
    /clearall                Clear everything, including loaded exports and
                             looked-up background (files on disk are untouched).
    /debug                   Show the last message sent to the chat model.

Prerequisites (macOS, Apple Silicon, Python 3.11):
    brew install python@3.11 ollama ffmpeg
    python3.11 -m venv venv && source venv/bin/activate
    bash install_musiclyse.sh          # installs the Python packages, downloads the models
                                       # (MOSS-Music, Whisper, CLAP, MAEST, MTG-Jamendo,
                                       # Demucs, PANNs) and checks that everything imports
    omnizart download-checkpoints      # once per venv (the installer says when it's needed)
    ollama serve                       # or run the Ollama app
    ollama pull gemma4:31b             # the chat model named in OLLAMA_MODEL; if you
                                       # change OLLAMA_MODEL, pull that name instead
    # The listening model's code comes from OpenMOSS's GitHub repo, at the tested
    # commit (install_musiclyse.sh checks it):
    git clone https://github.com/OpenMOSS/MOSS-Music.git ~/MOSS-Music
    git -C ~/MOSS-Music checkout ad107c7ddaa06de168a0dfbc18d3e1e6a40c0e5e
    # (or clone it elsewhere and set MOSS_MUSIC_CODE_DIR).

Optional, for more accurate section maps: SongFormer, in its own environment
(~/SongFormer):
    bash install_songformer.sh

Usage:
    python musiclyse.py                         # start the chat
    python musiclyse.py "saved-songs/llm-ready/plus-wiki/Artist - Title.txt"   # start with an export loaded
"""

# ==============================================================================
#  0.7.1-pre22: THIS SCRIPT AND ITS FILES
# ==============================================================================
# The code is in this script and in 17 files beside it that share its name
# (musiclyse-0.8-wiki.py, musiclyse-0.8-chat.py and the musiclyse-0.8-pipeline-....py files).
# Every setting is in musiclyse-0.8-settings.py. This script holds start-up, the file loader and
# its checks, the terminal and status lines, file tags, the Ollama client, the command loop, and
# helpers shared by the other files. CLAUDE.md, beside these files, lists what each file holds
# and the rules for changing them.
# ==============================================================================

# ==============================================================================
# 1. RUNTIME MONKEY PATCHES (MUST BE AT THE VERY TOP)
# ==============================================================================
__version__ = "0.8"

import sys
import time
import numpy as np
import multiprocessing

# Fix deprecated/removed NumPy attributes for legacy packages (madmom/omnizart)
np.float = float
np.int = int
np.complex = complex
# np.bool is only restored when it is actually missing (NumPy 1.24-1.26).
# On NumPy 2.x np.bool is numpy's real boolean dtype, and overwriting it with
# Python's bool breaks numpy.ma the first time it is imported -- which silently
# breaks np.unique / np.setdiff1d and everything built on them (meter/downbeat
# estimation, structure novelty boundaries, beat-synchronous features).
import warnings as _np_alias_warnings
with _np_alias_warnings.catch_warnings():
    _np_alias_warnings.simplefilter("ignore")
    if not hasattr(np, "bool"):
        np.bool = bool

# Patch pkg_resources fallback for setuptools compatibility
try:
    import pkg_resources
except ImportError:
    try:
        import pip._vendor.pkg_resources as pkg_resources
        sys.modules["pkg_resources"] = pkg_resources
    except ImportError:
        pass

# Force macOS multiprocessing to 'fork' so child workers inherit these patches
if sys.platform == "darwin":
    try:
        multiprocessing.set_start_method("fork", force=True)
    except RuntimeError:
        pass

import os
import difflib

import collections
import collections.abc

if not hasattr(collections, "MutableSequence"):
    collections.MutableSequence = collections.abc.MutableSequence
if not hasattr(collections, "MutableMapping"):
    collections.MutableMapping = collections.abc.MutableMapping
if not hasattr(collections, "Mapping"):
    collections.Mapping = collections.abc.Mapping
    
os.environ["HF_HUB_OFFLINE"] = "1"

# --- Quiet mode: suppress third-party log/warning spam --------------------
# Must happen before torch/transformers/tensorflow (via omnizart) get
# imported, since several of these only take effect if set beforehand.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")       # TF C++ log level
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")  # HF transformers
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("PYTHONWARNINGS", "ignore")

import io
import contextlib
import logging
import warnings

warnings.filterwarnings("ignore")
# Silences every INFO/WARNING log line from every named logger process-wide
# (tensorflow, omnizart's "Vocal Transcription"/"Vocal Contour"/"Music
# Transcription"/"Base Class"/"IO" loggers, etc.) without touching our own
# print()-based status output below.
logging.disable(logging.WARNING)

import re
import unicodedata

try:
    import readline  # enables left/right arrow editing in input()
except Exception:
    readline = None
import base64
import subprocess
import tempfile
import shutil
import textwrap
import json
import math
import datetime
import gc



def _missing_package_message(error):
    """0.8: start-up stops with this, not a traceback, when a package Musiclyse needs isn't installed
    for the Python running it."""
    name = getattr(error, "name", None) or str(error)
    return (f"Musiclyse can't start: the Python package {name!r} isn't installed for this Python "
            f"({sys.executable}). Start Musiclyse with the Python its packages were installed into, "
            f"or install {name} into this one.")


try:
    import requests
except ImportError as _missing:
    raise SystemExit(_missing_package_message(_missing))
# PyTorch's Apple-GPU (MPS) allocator. HIGH = 0.0 removes its upper limit on
# GPU memory (the setting that previously loaded successfully on this
# machine). It does NOT make freed memory go back sooner: the allocator only
# reclaims its cache of freed blocks once the process passes the LOW
# watermark -- by default 1.4 x the GPU working set macOS recommends (about
# 48 GB on a 64 GB Mac), i.e. ~67 GB. While MOSS-Music generated, the Python
# process sat at ~73 GB with memory pressure in the yellow. LOW = 0.6 (about
# 29 GB on a 64 GB Mac; MOSS-Music itself needs ~20 GB) makes the allocator
# reclaim cached blocks from there instead. Allowed because HIGH is 0.0. A
# value already set in your shell environment takes precedence over these.
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.6")
try:
    import torch
    import librosa
    import numpy as np
    from transformers import (
        AutoModelForSpeechSeq2Seq,
        AutoProcessor,
        pipeline,
    )
except ImportError as _missing:          # 0.8: see _missing_package_message
    raise SystemExit(_missing_package_message(_missing))

# CLAP is an optional extension: existing installs without the CLAP classes
# must still start and retain all previous functionality.
try:
    from transformers import ClapModel, ClapProcessor
    CLAP_TRANSFORMERS_AVAILABLE = True
except ImportError:
    ClapModel = ClapProcessor = None
    CLAP_TRANSFORMERS_AVAILABLE = False


# ==============================================================================
# USER CONFIGURATION
# ==============================================================================
# 0.7.1-pre19: every setting is in the settings file beside this script,
# musiclyse-<version>-settings.py (its "MAIN SCRIPT SETTINGS" part). It is read here, where
# the settings used to be, so everything below sees exactly the same values.
# ==============================================================================

def _musiclyse_load_settings(section):
    """0.7.1-pre19: runs this script's part of the settings file (musiclyse-<version>-settings.py,
    beside this script) here, in this script's own namespace, at the point where its settings
    used to be. The file's first line names its version; the "# MAIN SCRIPT SETTINGS" and
    "# EXPORTER SETTINGS" lines mark the two parts.
    0.8-pre31 PERSONAL_SETTINGS_FILE: when that file exists beside this script, each setting in it
    replaces this part's value right where the settings file sets it, so settings worked out from
    it follow. A name that isn't a setting in the settings file, or a line that isn't a plain
    NAME = value, is reported and left out. Without the file, this part runs exactly as before."""
    import os as _os
    import sys as _sys
    _me = _os.path.abspath(globals().get("__file__") or _sys.argv[0])
    _who = "Musiclyse" if section == "main" else "The Musiclyse exporter"
    _base = _os.path.splitext(_me)[0]
    if _base.endswith("-export"):
        _base = _base[:-len("-export")]
    _path = _base + "-settings.py"
    _name = _os.path.basename(_path)
    if not _os.path.isfile(_path):
        raise SystemExit(f"{_who} can't start: {_name} is missing. It has to be in the same folder as "
                         f"{_os.path.basename(_me)}.")
    with open(_path, encoding="utf-8") as _fh:
        _lines = _fh.read().split("\n")
    if not _lines or _lines[0].rstrip() != f"# MUSICLYSE SETTINGS {__version__}":
        raise SystemExit(f"{_who} can't start: {_name} doesn't belong to version {__version__} (its first "
                         f"line must read \"# MUSICLYSE SETTINGS {__version__}\").")
    _main = [i for i, _l in enumerate(_lines) if _l.rstrip() == "# MAIN SCRIPT SETTINGS"]
    _exp = [i for i, _l in enumerate(_lines) if _l.rstrip() == "# EXPORTER SETTINGS"]
    if len(_main) != 1 or len(_exp) != 1 or _main[0] > _exp[0]:
        raise SystemExit(f"{_who} can't start: {_name} needs one \"# MAIN SCRIPT SETTINGS\" line and, after "
                         f"it, one \"# EXPORTER SETTINGS\" line.")
    _a, _b = (_main[0] + 1, _exp[0]) if section == "main" else (_exp[0] + 1, len(_lines))
    _code = "\n" * _a + "\n".join(_lines[_a:_b])
    try:                       # 0.8: a mistake in the settings file stops start-up with a plain message
        compile(_code, _path, "exec")
    except SyntaxError as _e:
        raise SystemExit(f"{_who} can't start: {_name} has a mistake on line {_e.lineno} ({_e.msg}). "
                         f"Fix that line, or put back the file from the download, and start again.")
    # ---- 0.8-pre31 PERSONAL_SETTINGS_FILE ----
    _over, _tree = {}, None
    try:
        import ast as _ast

        def _assigned(_node):
            _found = set()
            for _x in _ast.walk(_node):
                if isinstance(_x, (_ast.Assign, _ast.AnnAssign, _ast.AugAssign)):
                    for _t in (_x.targets if isinstance(_x, _ast.Assign) else [_x.target]):
                        _found.update(_y.id for _y in _ast.walk(_t) if isinstance(_y, _ast.Name))
            return _found

        _tree = _ast.parse(_code, _path)
        _pname = ""
        for _n in _tree.body:
            if (isinstance(_n, _ast.Assign) and len(_n.targets) == 1 and isinstance(_n.targets[0], _ast.Name)
                    and _n.targets[0].id == "PERSONAL_SETTINGS_FILE"):
                try:
                    _pname = _ast.literal_eval(_n.value)
                except Exception:
                    _pname = ""
        _ppath = (_os.path.join(_os.path.dirname(_me), _os.path.expanduser(_pname.strip()))
                  if isinstance(_pname, str) and _pname.strip() else "")
        if _ppath and _os.path.isfile(_ppath):
            _known = _assigned(_ast.parse("\n".join(_lines), _path))
            _mine = _assigned(_tree)
            with open(_ppath, encoding="utf-8") as _fh:
                _ptree = _ast.parse(_fh.read(), _ppath)
            _short = _os.path.basename(_ppath)
            _notes, _used = [], 0
            for _n in _ptree.body:
                if not (isinstance(_n, _ast.Assign) and len(_n.targets) == 1 and isinstance(_n.targets[0], _ast.Name)):
                    _notes.append(f"line {_n.lineno} isn't a plain NAME = value line")
                    continue
                _k = _n.targets[0].id
                try:
                    _v = _ast.literal_eval(_n.value)
                except Exception:
                    _notes.append(f"line {_n.lineno}: {_k} isn't a plain value (text, a number, True or False, "
                                  f"None, or a list of them)")
                    continue
                if _k not in _known:
                    _notes.append(f"line {_n.lineno}: {_k} isn't a setting in version {__version__} (misspelt, "
                                  f"or no longer used)")
                    continue
                _used += 1
                if _k in _mine:
                    _over[_k] = _v
            _out = _sys.stdout if section == "main" else _sys.stderr
            if section == "main" and _used:
                print(f"  (Using {_used} setting{'s' if _used != 1 else ''} of your own from {_short}.)", file=_out)
            for _note in _notes:
                print(f"  ({_short}, {_note}: left out.)", file=_out)
    except Exception as _e:
        print(f"  (Your own settings file couldn't be read ({type(_e).__name__}: {_e}); only {_name} is used.)",
              file=_sys.stdout if section == "main" else _sys.stderr)
        _over = {}
    try:                       # 0.8: so does a setting that fails when it runs
        if not _over:
            exec(compile(_code, _path, "exec"), globals())
            return
        _g = globals()
        for _n in _tree.body:
            exec(compile(_ast.Module(body=[_n], type_ignores=[]), _path, "exec"), _g)
            for _k in _assigned(_n) & set(_over):
                _g[_k] = _over[_k]
    except Exception as _e:
        import traceback as _tb
        _line = next((_f.lineno for _f in reversed(_tb.extract_tb(_e.__traceback__)) if _f.filename == _path), None)
        raise SystemExit(f"{_who} can't start: {_name} has a mistake" + (f" on line {_line}" if _line else "")
                         + f" ({type(_e).__name__}: {_e}). Fix it, or put back the file from the download, "
                           f"and start again.")


_musiclyse_load_settings("main")

# ==============================================================================
# END OF USER CONFIGURATION
# ==============================================================================

# ==============================================================================
#  0.7.1-pre22: THE MUSICLYSE FILES
# ==============================================================================
# Each file named in _MUSICLYSE_FILES sits beside this script and is named after it
# ("musiclyse-<version>.py" -> "musiclyse-<version>-pipeline-scan.py"). They are all read here,
# before the rest of this script runs: a missing file, or one whose first line names another
# version, stops start-up. After this script's own code has run, each file runs whole, in this
# order, in this script's own namespace, so all of them share one set of names and none imports
# another. A function or class defined in two of the files (this script included) stops start-up
# and names both. Top-level code in a file (outside functions) may only use names from this script
# or from files earlier in this list; inside functions, any name can be used.
_MUSICLYSE_FILES = (
    "wiki", "chat", "memory", "pipeline-scan",
    "pipeline-listening", "pipeline-lyrics", "pipeline-stems", "pipeline-rhythm",
    "pipeline-harmony", "pipeline-production", "pipeline-genre", "pipeline-instruments",
    "pipeline-vocals", "pipeline-identity", "pipeline-sections", "pipeline-cover",
    "pipeline-report",
)
_MUSICLYSE_FILE_TEXT = {}


def _musiclyse_file_path(kind):
    """0.7.1-pre22: (this script's path, the path of its file `kind`)."""
    import os as _os
    import sys as _sys
    _main = _os.path.abspath(globals().get("__file__") or _sys.argv[0])
    return _main, _os.path.splitext(_main)[0] + "-" + kind + ".py"


def _musiclyse_check_files():
    """0.7.1-pre22: reads every Musiclyse file; stops start-up when one is missing or belongs to
    another version."""
    import os as _os
    for _kind in _MUSICLYSE_FILES:
        _main, _path = _musiclyse_file_path(_kind)
        if not _os.path.isfile(_path):
            raise SystemExit(f"Musiclyse can't start: {_os.path.basename(_path)} is missing. It has to be in "
                             f"the same folder as {_os.path.basename(_main)}.")
        with open(_path, encoding="utf-8") as _fh:
            _text = _fh.read()
        _first = f"# MUSICLYSE FILE {_kind} {__version__}"
        if _text.split("\n", 1)[0].rstrip() != _first:
            raise SystemExit(f"Musiclyse can't start: {_os.path.basename(_path)} doesn't belong to version "
                             f"{__version__} of {_os.path.basename(_main)} (its first line must read "
                             f"\"{_first}\").")
        _MUSICLYSE_FILE_TEXT[_kind] = (_path, _text)


def _musiclyse_load_files():
    """0.7.1-pre22: runs each Musiclyse file in this script's namespace, in _MUSICLYSE_FILES order.
    Stops start-up when a function or class is defined in two of the files (this script included)."""
    import os as _os
    import types as _types
    _g = globals()
    _main = _musiclyse_file_path(_MUSICLYSE_FILES[0])[0]
    _here = _musiclyse_load_files.__code__.co_filename
    _defined = {}
    for _name, _value in list(_g.items()):
        _code = getattr(_value, "__code__", None)
        if ((isinstance(_code, _types.CodeType) and _code.co_filename == _here)
                or (isinstance(_value, type) and _value.__module__ == _g.get("__name__"))):
            _defined[_name] = _os.path.basename(_main)
    for _kind in _MUSICLYSE_FILES:
        _path, _text = _MUSICLYSE_FILE_TEXT.pop(_kind)
        _code = compile(_text, _path, "exec")
        _file = _os.path.basename(_path)
        for _c in _code.co_consts:
            if isinstance(_c, _types.CodeType) and not _c.co_name.startswith("<"):
                _other = _defined.get(_c.co_name)
                if _other is not None and _other != _file:
                    raise SystemExit(f"Musiclyse can't start: {_c.co_name} is defined in both {_other} and "
                                     f"{_file}. Keep it in one of them.")
                _defined[_c.co_name] = _file
        exec(_code, _g)


_musiclyse_check_files()


# ------------------------------------------------------------------------------
# dev55: background work (PARALLEL_ANALYSIS). Independent measurements run in
# worker threads while the listening model works on the GPU. Workers never
# write to the console directly: their output is collected per thread and
# printed by the main thread when the work is joined, so the status line and
# the stage-timing table belong to the main thread only. quiet_stdout() in a
# worker mutes that worker alone (contextlib.redirect_stdout swaps the
# process-wide sys.stdout, which from a worker would swallow -- or, with two
# workers, permanently lose -- the main thread's output).
# ------------------------------------------------------------------------------
import threading
_WORKER_OUT = threading.local()


def _in_worker_thread():
    return getattr(_WORKER_OUT, "buf", None) is not None


class _RoutedStream:
    """Stand-in for sys.stdout/sys.stderr while workers run: a worker
    thread's writes go to its own buffer (or nowhere while muted); every
    other thread writes straight through."""

    def __init__(self, real):
        self._real = real

    def write(self, s):
        buf = getattr(_WORKER_OUT, "buf", None)
        if buf is not None:
            if not getattr(_WORKER_OUT, "mute", False):
                try:
                    buf.write(s)
                except Exception:
                    pass
            return len(s) if isinstance(s, str) else 0
        return self._real.write(s)

    def flush(self):
        if getattr(_WORKER_OUT, "buf", None) is None:
            try:
                self._real.flush()
            except Exception:
                pass

    def __getattr__(self, name):
        return getattr(self._real, name)


_ROUTE_STATE = {"depth": 0, "out": None, "err": None}
_ROUTE_LOCK = threading.Lock()


def _route_streams_on():
    """Install the routed streams (main thread; nested calls are counted)."""
    with _ROUTE_LOCK:
        if _ROUTE_STATE["depth"] == 0:
            if not isinstance(sys.stdout, _RoutedStream):
                _ROUTE_STATE["out"] = sys.stdout
                sys.stdout = _RoutedStream(sys.stdout)
            if not isinstance(sys.stderr, _RoutedStream):
                _ROUTE_STATE["err"] = sys.stderr
                sys.stderr = _RoutedStream(sys.stderr)
        _ROUTE_STATE["depth"] += 1


def _route_streams_off():
    with _ROUTE_LOCK:
        _ROUTE_STATE["depth"] = max(0, _ROUTE_STATE["depth"] - 1)
        if _ROUTE_STATE["depth"] == 0:
            if isinstance(sys.stdout, _RoutedStream) and _ROUTE_STATE["out"] is not None:
                sys.stdout = _ROUTE_STATE["out"]
            if isinstance(sys.stderr, _RoutedStream) and _ROUTE_STATE["err"] is not None:
                sys.stderr = _ROUTE_STATE["err"]
            _ROUTE_STATE["out"] = _ROUTE_STATE["err"] = None


_ACTIVE_TASKS = []


def _settle_background_tasks():
    """Join any background task an error path left running (prints its output
    and restores the console streams). Safe to call any time from the main
    thread."""
    for t in list(_ACTIVE_TASKS):
        try:
            t.result()
        except Exception:
            pass
    try:
        _clear_omnizart_prefetch()
    except Exception:
        pass


def _gpu_cache_clear_ok():
    """dev65: False in a worker thread (see GPU_CLEANUP_MAIN_THREAD_ONLY)."""
    if not globals().get("GPU_CLEANUP_MAIN_THREAD_ONLY", True):
        return True
    return threading.current_thread() is threading.main_thread()


class _BackgroundTask:
    """Run fn(*args, **kwargs) in a worker thread. result() joins it (main
    thread), prints whatever the worker printed, and returns (ok, value):
    ok is False when the worker raised, so the caller can redo the work the
    dev54 way. Timing goes to the stage table as "[background] <label>"."""

    def __init__(self, label, fn, *args, **kwargs):
        self.label = label
        self._fn, self._args, self._kwargs = fn, args, kwargs
        self._value = None
        self._error = None
        self._buf = io.StringIO()
        self._elapsed = None
        self._joined = False
        _route_streams_on()
        self._thread = threading.Thread(target=self._run, name=f"musiclyse-{label}", daemon=True)
        if threading.current_thread() is threading.main_thread():
            _ACTIVE_TASKS.append(self)
        self._thread.start()

    def _run(self):
        _WORKER_OUT.buf = self._buf
        _WORKER_OUT.mute = False
        t0 = time.perf_counter()
        try:
            self._value = self._fn(*self._args, **self._kwargs)
        except BaseException as e:      # reported in result()
            self._error = e
        finally:
            self._elapsed = time.perf_counter() - t0
            _WORKER_OUT.buf = None

    def done(self):
        return not self._thread.is_alive()

    def result(self):
        if not self._joined:
            self._thread.join()
            self._joined = True
            try:
                _ACTIVE_TASKS.remove(self)
            except ValueError:
                pass
            _route_streams_off()
            text = self._buf.getvalue()
            if text.strip():
                try:
                    status_done()
                except Exception:
                    pass
                print(text, end="" if text.endswith("\n") else "\n")
            if self._elapsed is not None:
                try:
                    _STAGE_LOG.append((f"[background, overlapped] {self.label}", float(self._elapsed)))
                except Exception:
                    pass
            if self._error is not None:
                print(f"  (background {self.label} failed, redoing it in order: "
                      f"{type(self._error).__name__}: {self._error})")
        if self._error is not None:
            return False, None
        return True, self._value


@contextlib.contextmanager
def quiet_stdout():
    """Swallow stray stdout (and optionally stderr) from third-party libraries
    (e.g. Keras/Omnizart progress bars) that print outside the logging module."""
    if globals().get("SHOW_OMNIZART_LOGS", False):
        yield
        return
    if _in_worker_thread():
        # dev55: mute this worker only; never swap the process-wide streams.
        prev = getattr(_WORKER_OUT, "mute", False)
        _WORKER_OUT.mute = True
        try:
            yield
        finally:
            _WORKER_OUT.mute = prev
        return
    out_buf = io.StringIO()
    err_buf = io.StringIO()
    with contextlib.redirect_stdout(out_buf), contextlib.redirect_stderr(err_buf):
        yield


# --- Console appearance ----------------------------------------------------
# Set to False to fall back to the old plain scrolling log style (each
# processing step printed as its own permanent line, no overwriting).
# (FRIENDLY_PROCESSING_DISPLAY moved to the settings file beside this script.)

from builtins import print as _builtin_print

USE_COLOR = sys.stdout.isatty()


class Ansi:
    RESET = "\033[0m"
    YELLOW = "\033[33m"       # processing / loading status text
    LIGHT_GREEN = "\033[92m"  # user input prompt
    MAGENTA = "\033[35m"      # final output / answer text
    WHITE = "\033[97m"        # startup logo


def _colorize(text, color):
    if not USE_COLOR:
        return text
    return f"{color}{text}{Ansi.RESET}"


def _plain_terminal_text(text):
    """Give plain console text the default color, preserving styled output."""
    if text.strip() and "\x1b" not in text:
        return _colorize(text, Ansi.YELLOW)
    return text


def print(*values, sep=" ", end="\n", file=None, flush=False):
    """Apply the default color to Musiclyse's own terminal output."""
    stream = sys.stdout if file is None else file
    if (USE_COLOR and (stream is sys.stdout or stream is sys.stderr)
            and stream.isatty() and not _in_worker_thread()):
        values = tuple(_plain_terminal_text(str(value)) for value in values)
    archive = globals().get("_CHAT_DISPLAY_ARCHIVE")
    if archive is not None and (file is None or file is sys.stdout) and not _in_worker_thread():
        line = (" " if sep is None else sep).join(str(value) for value in values) + ("\n" if end is None else end)
        if line.strip() and "\r" not in line and (end is None or end.endswith("\n")):
            archive.append(line)
    _builtin_print(*values, sep=sep, end=end, file=file, flush=flush)


def _default_color_lines(text, stream):
    """pre62 CONSISTENT_TEXT_COLOR: text written straight to `stream`, each plain line
    in the default colour as print gives Musiclyse's own output; unchanged when colour
    is off, the stream isn't a terminal, or the setting is False."""
    try:
        if not (CONSISTENT_TEXT_COLOR and USE_COLOR and stream.isatty() and not _in_worker_thread()):
            return text
    except Exception:
        return text
    return "\n".join(_plain_terminal_text(part) for part in str(text).split("\n"))


def _helper_color_env(env):
    """pre62 CONSISTENT_TEXT_COLOR: MUSICLYSE_COLOR=1 in the environment of the exporter or
    SongFormer helper that /export or /songformer runs, so their own messages print in the
    default colour too -- only while Musiclyse's own output is coloured."""
    env.pop("MUSICLYSE_COLOR", None)
    try:
        if CONSISTENT_TEXT_COLOR and USE_COLOR and sys.stdout.isatty():
            env["MUSICLYSE_COLOR"] = "1"
    except Exception:
        pass
    return env


_status_width = 0

# dev54: per-stage wall-clock profiler. Purely observational (no effect on results).
# dev54: set env MUSICLYSE_TEST_SEED=<int> ONLY for A/B regression runs. It seeds the
# sampled Whisper cross-check and Ollama requests so two runs are comparable. Unset (the
# default) = behaviour identical to dev53.
try:
    _TEST_SEED = int(os.environ["MUSICLYSE_TEST_SEED"]) if os.environ.get("MUSICLYSE_TEST_SEED", "").strip() else None
except Exception:
    _TEST_SEED = None
_STAGE = {"label": None, "t": None}
_STAGE_LOG = []


def _mark_stage(label):
    now = time.perf_counter()
    if _STAGE["label"] is not None:
        _STAGE_LOG.append((_STAGE["label"], now - _STAGE["t"]))
    _STAGE["label"], _STAGE["t"] = label, now


def _print_stage_log(top=None):
    """Close the open stage and print stage durations (longest first)."""
    try:
        _mark_stage(None)
        _STAGE["label"] = None
        rows = sorted(_STAGE_LOG, key=lambda r: -r[1])
        # dev55: background work overlapped other stages, so it is listed but
        # not added to the wall-clock total.
        total = sum(d for lbl, d in _STAGE_LOG if not str(lbl).startswith("[background"))
        print(f"\n  Stage timing (total {total:.1f}s):")
        for lbl, d in (rows[:top] if top else rows):
            print(f"    {d:8.1f}s  {100*d/max(total,1e-9):5.1f}%  {lbl}")
        _STAGE_LOG.clear()
    except Exception:
        pass


def status(msg):
    """Print a progress-style status message. When FRIENDLY_PROCESSING_DISPLAY
    is True, overwrites the previous status on the same line (progress-bar
    style). When False, prints each step as its own permanent line (the
    original scrolling log style)."""
    global _status_width
    if _in_worker_thread():
        return      # dev55: background work stays off the status line
    try:
        _mark_stage(str(msg))
    except Exception:
        pass
    colored = _colorize(msg, Ansi.YELLOW)
    if FRIENDLY_PROCESSING_DISPLAY:
        line = _colorize(f"  \u23f3 {msg}", Ansi.YELLOW)
        pad = max(0, _status_width - len(line))
        print("\r" + line + " " * pad, end="", flush=True)
        _status_width = len(line)
    else:
        print(f"  ({colored})")
        _status_width = 0


def status_done(msg=None, *, clear=False):
    """Finalize the status line: mark it done (friendly mode) and drop to a
    fresh line, or just print the final note (log-style mode).
    With clear=True, erase a transient status on an interactive terminal."""
    global _status_width
    if _in_worker_thread():
        return
    if msg is not None:
        colored = _colorize(msg, Ansi.YELLOW)
        if FRIENDLY_PROCESSING_DISPLAY:
            line = _colorize(f"  \u2713 {msg}", Ansi.YELLOW)
            pad = max(0, _status_width - len(line))
            print("\r" + line + " " * pad)
        else:
            print(f"  ({colored})")
    elif FRIENDLY_PROCESSING_DISPLAY and _status_width:
        if clear and sys.stdout.isatty():
            print("\r\x1b[2K", end="", flush=True)
        else:
            print()
    _status_width = 0


MUSICLYSE_LOGO = r"""
       @@@@    @@@@  @@@     @@@   @@@@@     @@     @@@@@     @@@    @@@    @@@   @@@@@    @@@@@@@@ 
       @@@@@  @@@@@  @@@    @@@  @@@  @@@   @@    @@@.@@@@   @@@      @@@  @@@  @@@  @@@  @@@@      
      @@@@@  @@@@@  @@@    @@@  @@@@       @@@  @@@         @@@       @@@@@@   @@@@       @@@       
      @@ @@ @@ @@@  @@@    @@@   @@@@@@    @@   @@          @@         @@@@     @@@@@@   @@@@@@@@@  
     @@  @@@@  @@  @@@    @@@        @@@  @@@  @@@         @@@         @@@          @@@  @@@        
 @@@@@@  @@@  @@   @@@   @@@  @@@   @@@  @@@   @@@   @@@   @@@@@@@@   @@@    @@@   @@@  @@@         
@@@@@@  @@@  @@@   @@@@@@@     @@@@@@    @@@    @@@@@@    @@@@@@@@   @@@      @@@@@@@  @@@@@@@@@    
  @@@                                                                                               
                         T H E   F R E E   M U S I C   A N A L Y S E R
                         =============================================
                         ---------------------------------------------
                         ==========  V E R S I O N   0 . 8  ==========
                         ========= Release date:  2026/11/20 =========
                         ---------------------------------------------
                         ©     R o b e r t    G o u r l e y    2 0 2 6
                         ---------------------------------------------
                         Musiclyse may be reused under the MIT License
                         =============================================
                         Models are licensed under respective licenses
                         ---------------------------------------------
                         A   G O U R L I S H   V I B E   P R O J E C T
                         ====https://github.com/gourlish/musiclyse====
                         ==https://gourlish.altervista.org/musiclyse==
                         =======Feedback:rjdgourley@outlook.com=======
"""


def set_terminal_title(title="Musiclyse 0.8"):
    """Set the terminal window/tab title (OSC 0). No-op when not a TTY."""
    if not sys.stdout.isatty():
        return
    try:
        # OSC 0 sets icon name + window title on most terminals (macOS Terminal,
        # iTerm2, xterm, Windows Terminal, etc.).
        sys.stdout.write(f"\033]0;{title}\007")
        sys.stdout.flush()
    except Exception:
        pass


def print_logo():
    set_terminal_title("Musiclyse 0.8")
    print(_colorize(MUSICLYSE_LOGO, "\033[38;5;213m"))


def colored_input(prompt, color, prompt_color=None):
    """Like input(), but with a colored prompt and typed text.

    Terminals/readline count raw bytes in the prompt for line width. ANSI
    color codes are invisible but still count, which causes long lines to
    wrap wrong (look like "echo"/repeat) and the prompt to vanish on resize.

    GNU readline treats bytes between \\001 and \\002 as non-printing, so we
    wrap color codes in those markers. Color stays active during typing so
    input remains green; we reset after Enter.
    """
    if not USE_COLOR:
        return input(prompt)
    # libedit prompt-width handling differs from GNU readline. Keep escape
    # sequences outside its prompt so Backspace cannot count them as columns.
    if prompt_color is None and readline is not None and "libedit" in (readline.__doc__ or "").lower():
        sys.stdout.write(prompt_color or color)
        sys.stdout.flush()
        try:
            return input(prompt)
        finally:
            sys.stdout.write(Ansi.RESET)
            sys.stdout.flush()
    # \001 / \002 = readline non-printing markers (width-safe). Only useful when
    # the readline module is loaded; otherwise leave plain ANSI in the prompt.
    if readline is not None:
        rl_prompt = (f"\001{prompt_color}\002{prompt}\001{color}\002" if prompt_color
                     else f"\001{color}\002{prompt}")
    else:
        rl_prompt = f"{prompt_color}{prompt}{color}" if prompt_color else f"{color}{prompt}"
    try:
        text = input(rl_prompt)
    finally:
        # End the color so the next status/output line is not tinted green.
        sys.stdout.write(Ansi.RESET)
        sys.stdout.flush()
    return text

# --- Optional file-metadata / cover-art integration -------------------------
METADATA_IMPORT_ERROR = ""
try:
    from mutagen import File as MutagenFile
    METADATA_AVAILABLE = True
except Exception as _e:
    MutagenFile = None
    METADATA_AVAILABLE = False
    METADATA_IMPORT_ERROR = str(_e)

# (ENABLE_FILE_METADATA moved to the settings file beside this script.)
# (SEND_COVER_ART_TO_OLLAMA moved to the settings file beside this script.)
# (METADATA_MAX_LYRICS_CHARS moved to the settings file beside this script.)

# Hard cap on a sanitized Whisper lyric transcription. The sanitizer used to
# cut anything over 3500 characters down to 2500 and label the result
# "spam/loop detected" -- but 3500 characters is an ordinary length for a rap
# verse-heavy track or any song over about four minutes, so genuinely
# transcribed lyrics were being discarded wholesale and then mislabelled as
# model spam. Only true runaway loops should be cut for length, and those are
# already handled by the repetition collapsers, so this ceiling exists purely
# to stop a pathological decode from blowing up the context window.
# (LYRICS_MAX_CHARS moved to the settings file beside this script.)
# (COVER_IMAGE_MAX_DIMENSION moved to the settings file beside this script.)
# ffmpeg's mjpeg encoder uses qscale 2..31, where lower is better.
# The old value "80" was invalid and caused ffmpeg cover scaling/extraction to fail.
# (COVER_IMAGE_QUALITY moved to the settings file beside this script.)
# (MAX_COVER_IMAGES_PER_REQUEST moved to the settings file beside this script.)

# If conversion fails but the original embedded art is directly sendable, allow this size as a fallback.
# (MAX_COVER_BYTES_TO_SEND_RAW moved to the settings file beside this script.)

# Lower default context keeps Ollama's KV cache smaller and avoids many 400s when the
# accumulated writer history + stem/MIDI report is large. The retry logic below can still
# fall back to even smaller contexts automatically.
# (OLLAMA_URL moved to the settings file beside this script.)
# (OLLAMA_NUM_CTX moved to the settings file beside this script.)
# Running totals for the current process (updated after each Ollama reply).
SESSION_TOKEN_USAGE = {"prompt": 0, "completion": 0, "total": 0, "last_prompt": 0, "last_completion": 0, "last_ctx": 0}


OLLAMA_BASE_URL = OLLAMA_URL.rsplit("/api/chat", 1)[0]

# Set this to False if your Ollama model is text-only / does not support images.
# If True, the script will still retry without images if Ollama returns a 400.
# (OLLAMA_SUPPORTS_IMAGES moved to the settings file beside this script.)

# --- CLAP sonic-similarity embeddings ---------------------------------------
# Stored in saved-song JSONs only; single-track answers do not use this signal.
# The requested LAION checkpoint projects audio into a 512-dimensional space.
# (CLAP_MODEL_ID moved to the settings file beside this script.)
# (CLAP_EMBEDDING_DIMENSION moved to the settings file beside this script.)
# (CLAP_SAMPLE_RATE moved to the settings file beside this script.)
# (CLAP_WINDOW_SECONDS moved to the settings file beside this script.)
# (CLAP_WINDOW_HOP_SECONDS moved to the settings file beside this script.)
# (CLAP_MAX_WINDOWS moved to the settings file beside this script.)
# (CLAP_USE_MPS moved to the settings file beside this script.)
# (CLAP_EMBEDDING_PRECISION moved to the settings file beside this script.)
# (ENABLE_CLAP_EMBEDDING moved to the settings file beside this script.)

# (MAX_WRITER_IMAGES_PER_TURN moved to the settings file beside this script.)

# How long the Ollama/mlx runner stays resident after an ordinary interactive
# reply, before the OS-level keep_alive timer would unload it.
#
# ollama_chat() previously omitted "keep_alive" from the request entirely for
# every normal turn, which means Ollama's OWN server-side default applies --
# observed to be 5 minutes. That is shorter than a realistic gap between
# messages: reading a reply, thinking about the next question, or just typing
# it out. When the timer fires, the entire mlx subprocess is killed ("stopping
# mlx runner subprocess" in the server log) -- not just its cache. The NEXT
# request then pays a full process restart (engine init, tensor + draft model
# load, ~8-9s) on top of reprocessing the ENTIRE prompt from a completely
# empty cache (matched=0), regardless of how stable that prompt's content is.
# This is a different mechanism from prefix-cache instability (see
# IMAGE_RETAIN_LAST_N_MESSAGES and the stable per-message truncation above):
# those fixes stop a cache MISS on content that should have hit; this fixes
# the process not being there to hit against at all.
#
# The tradeoff is memory: keeping the model resident through idle gaps holds
# its weights + KV cache in RAM/VRAM (observed ~22-24 GiB peak) rather than
# freeing them. 30 minutes covers normal conversational gaps without pinning
# memory indefinitely. Raise it, or set to -1 for "never unload", if idle
# unloads are still a problem and the memory headroom is available; lower it,
# or set to "0"/0 to restore the old behaviour, if memory pressure matters
# more than avoiding the reload cost.
# (OLLAMA_INTERACTIVE_KEEP_ALIVE moved to the settings file beside this script.)

# Sentinel distinguishing "keep_alive not passed" from "keep_alive=None passed
# deliberately", so ollama_chat's default can resolve to the constant above
# without losing the ability to explicitly request Ollama's own server default.
_KEEP_ALIVE_UNSET = object()
# How many of the MOST RECENT messages (any role, not just image-bearing
# ones) may still carry raw base64 image bytes. Anything older than this
# window has its "images" key dropped on the next compaction pass.
#
# Some inference backends cannot restore their prefix cache across a prompt
# containing image data -- observed as "failed to restore cache, freeing all
# caches" followed by a full reprocess of the ENTIRE prompt. Critically this
# recurs on EVERY subsequent turn for as long as that image sits anywhere in
# history, not just the turn it was shared on -- so a single image, shared
# once, made every later text-only turn for the rest of the session pay a
# full reprocess too. A raw base64 image is tens of thousands of tokens, the
# single most expensive thing that can sit in a conversation, and unlike an
# oversized text block it cannot be shrunk -- only removed.
#
# This must be a MESSAGE-DISTANCE window, not an image-count cap: capping by
# "the last N images" does nothing when only one image is ever shared, since
# that message is then simultaneously the newest and the only one, on every
# future turn, forever. Expiring by elapsed messages is what actually bounds
# the damage regardless of how many images did or didn't follow.
#
# 4 covers this turn's user message and reply, plus one more round-trip, so a
# direct follow-up ("what's in the top-left corner?") still has real pixels
# to look at. The text description of an image (cover-art notes, IMAGE N
# OBSERVATIONS blocks) is written into the message's CONTENT separately from
# its "images" key, so dropping the raw bytes after this window loses the
# ability to re-examine the pixels, not what the model was already told
# about them.
# (IMAGE_RETAIN_LAST_N_MESSAGES moved to the settings file beside this script.)
# (MAX_WRITER_HISTORY_MESSAGES moved to the settings file beside this script.)
# Per-message ceiling. This was 60k while a saved analysis was small; the
# analysis is now ~190-205k chars, so a loaded track lost ~70% of itself and a
# two-track /loadcompare lost the per-section evidence entirely. The real
# constraint is the context window, not this number: the proportional guard
# (budget // 3) still governs small contexts, so raising this only affects
# large ones. At OLLAMA_NUM_CTX=131072 two fully compressed analyses now fit
# with room left for the conversation.
# (MAX_MESSAGE_CHARS_FOR_OLLAMA moved to the settings file beside this script.)
# (HISTORY_CHAR_BUDGET_FACTOR moved to the settings file beside this script.)

# (UNLOAD_OMNIZART_AFTER_STEM_MIDI moved to the settings file beside this script.)
# (MAX_EXPLICIT_IMAGE_BYTES moved to the settings file beside this script.)


TITLE_KEYS = [
    "TIT2", "\xa9nam", "TITLE", "MusicTitle", "title",
]
ARTIST_KEYS = [
    "TPE1", "\xa9ART", "aART", "ARTIST", "Artist",
    "ALBUMARTIST", "Album Artist",
]
ALBUM_KEYS = [
    "TALB", "\xa9alb", "ALBUM", "Album",
]
YEAR_KEYS = [
    "TDRC", "TYER", "\xa9day", "DATE", "Year", "year",
    "RELEASEDATE", "Release Date", "YEAR",
    # Fallback only: these are checked strictly after everything above, so a
    # file that already resolves a year from the keys above is completely
    # unaffected. Files tagged by MusicBrainz Picard and similar tools often
    # carry an "original release" field even when the main year field is
    # blank (e.g. left empty on a reissue/compilation rip).
    "TDOR", "TORY", "ORIGINALDATE", "ORIGINALYEAR",
    "originaldate", "originalyear", "Original Date", "Original Year",
    "TXXX:ORIGINALDATE", "TXXX:ORIGINALYEAR",
]
LYRICS_KEYS = [
    "LYRICS", "unsyncedlyrics", "Lyrics", "UNSYNCEDLYRICS",
    "lyrc", "LYRIC", "\xa9lyr", "©lyr",
    "----:com.apple.iTunes:Lyrics",
    "----:com.apple.iTunes:LYRICS",
    "----:com.apple.iTunes:UNSYNCED LYRICS",
    "----:com.apple.iTunes:Unsynced Lyrics",
    "lyrics-eng", "lyrics-eng-xxx",
    "TXXX:LYRICS", "TXXX:Lyrics", "TXXX:UNSYNCEDLYRICS",
]

# Used only for era-estimation hedged evidence (see _extract_era_soft_hints
# below) -- deliberately kept separate from YEAR_KEYS since neither an
# encoder string nor an ISRC is a confirmed release year.
ENCODER_KEYS = [
    "TSSE", "\xa9too", "ENCODER", "encoder", "Encoder", "TENC",
]
ISRC_KEYS = [
    "TSRC", "ISRC", "isrc",
]


def _tag_text(value):
    if value is None:
        return ""

    if isinstance(value, (list, tuple)):
        parts = []
        for item in value:
            text = _tag_text(item)
            if text and text not in parts:
                parts.append(text)
        return "\n".join(parts).strip()

    if isinstance(value, bytes):
        try:
            return value.decode("utf-8", "ignore").strip()
        except Exception:
            return value.decode("latin-1", "ignore").strip()

    if hasattr(value, "text"):
        return _tag_text(getattr(value, "text"))

    return str(value).strip()


def _first_tag(tags, keys):
    if tags is None:
        return ""

    for key in keys:
        try:
            value = tags.get(key)
        except Exception:
            continue

        text = _tag_text(value)
        if text:
            return text

    return ""


def _collect_tags(tags, keys):
    if tags is None:
        return ""

    parts = []
    for key in keys:
        try:
            value = tags.get(key)
        except Exception:
            continue

        text = _tag_text(value)
        if text and text not in parts:
            parts.append(text)

    return "\n\n".join(parts).strip()


def _parse_year(value):
    text = _tag_text(value)
    m = re.search(r"\b(19\d{2}|20\d{2})\b", text)
    if m:
        return m.group(1)

    t = text.strip()
    if len(t) == 4 and t.isdigit():
        return t

    return ""


def _extract_id3_lyrics(tags):
    parts = []

    try:
        items = list(tags.items())
    except Exception:
        return ""

    for key, frame in items:
        key_u = str(key).upper()
        if not (key_u.startswith("USLT") or key_u.startswith("SYLT")):
            continue

        text = _tag_text(getattr(frame, "text", ""))
        if not text:
            # Some frames store the body differently
            text = _tag_text(frame)
        if not text:
            continue

        desc = _tag_text(getattr(frame, "desc", ""))
        lang = str(getattr(frame, "language", "") or "").strip()
        prefix = " ".join(x for x in (lang, desc) if x)

        parts.append(f"{prefix}\n{text}" if prefix else text)

    return "\n\n".join(parts).strip()


def _extract_lyrics_from_any_tags(tags):
    """
    Broad fallback: scan every tag key for anything that looks like lyrics.
    Handles MP4 freeform atoms, Vorbis comments, ID3 TXXX, and odd key spellings
    that the fixed LYRICS_KEYS list may miss.
    """
    if tags is None:
        return ""

    try:
        items = list(tags.items())
    except Exception:
        return ""

    parts = []
    for key, value in items:
        key_s = str(key)
        key_u = key_s.upper()

        # Skip obvious non-lyrics
        if any(tok in key_u for tok in ("APIC", "COVR", "PICTURE", "PRIV", "GEOB", "MCDI")):
            continue

        looks_like_lyrics = (
            "LYRIC" in key_u
            or key_u in ("\xa9LYR", "©LYR", "LYRC")
            or key_u.endswith(":LYRICS")
            or key_u.endswith(":LYRIC")
            or "UNSYNCED" in key_u
        )
        if not looks_like_lyrics:
            continue

        text = _tag_text(value)
        if not text:
            # MP4 freeform / bytes
            try:
                if hasattr(value, "decode"):
                    text = value.decode("utf-8", "ignore").strip()
                elif isinstance(value, (list, tuple)) and value:
                    text = _tag_text(value[0])
            except Exception:
                text = ""

        if text and len(text) > 8 and text not in parts:
            parts.append(text)

    return "\n\n".join(parts).strip()


def _guess_image_mime(data):
    if not data:
        return "application/octet-stream"

    if data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    if data[:4] == b"RIFF" and len(data) >= 12 and data[8:12] == b"WEBP":
        return "image/webp"

    return "application/octet-stream"


def _cover_temp_suffix(mime):
    mime = (mime or "").lower()
    if "jpeg" in mime:
        return ".jpg"
    if "png" in mime:
        return ".png"
    if "gif" in mime:
        return ".gif"
    if "webp" in mime:
        return ".webp"
    return ".img"


def _ffmpeg_scale_filter_variants(max_dimension):
    m = int(max_dimension)
    return [
        # Preferred modern ffmpeg scale filter.
        f"scale={m}:{m}:force_original_aspect_ratio=decrease:force_divisible_by=2",
        # Fallback for older ffmpeg builds that may not support force_original_aspect_ratio.
        f"scale=min({m},iw):-2,scale=-2:min({m},ih)",
    ]


def _raw_or_base64_image(value):
    """
    Return (bytes, mime) from a mutagen tag value if it represents embedded cover art.

    Handles:
      - ID3 APIC frames
      - MP4 covr objects
      - FLAC/Ogg/Opus Picture objects
      - raw bytes
      - base64-encoded picture strings, e.g. METADATA_BLOCK_PICTURE
    """
    if value is None:
        return None, "image/jpeg"

    data = getattr(value, "data", None) or getattr(value, "image", None)
    mime = getattr(value, "mime", None) or getattr(value, "mime_type", None)

    # Direct binary image payload.
    if data is not None:
        try:
            raw = bytes(data)
            if raw:
                return raw, str(mime or _guess_image_mime(raw))
        except Exception:
            pass

    # Raw bytes that may already be an image, or base64 text stored as bytes.
    if isinstance(value, (bytes, bytearray)):
        raw = bytes(value)

        if _guess_image_mime(raw) != "application/octet-stream":
            return raw, str(mime or _guess_image_mime(raw))

        try:
            decoded = base64.b64decode(raw.strip(), validate=False)
            if decoded and _guess_image_mime(decoded) != "application/octet-stream":
                return decoded, str(mime or _guess_image_mime(decoded))
        except Exception:
            pass

    # Base64 string.
    elif isinstance(value, str):
        text = value.strip()
        raw = text.encode("utf-8", "ignore")

        if _guess_image_mime(raw) != "application/octet-stream":
            return raw, str(mime or _guess_image_mime(raw))

        try:
            decoded = base64.b64decode(text, validate=False)
            if decoded and _guess_image_mime(decoded) != "application/octet-stream":
                return decoded, str(mime or _guess_image_mime(decoded))
        except Exception:
            pass

    return None, "image/jpeg"


def _extract_cover_from_tags(tags):
    if tags is None:
        return None, "image/jpeg"

    # ID3 APIC frames — MP3/AIFF/etc.
    try:
        for key, frame in list(tags.items()):
            if str(key).upper().startswith("APIC"):
                data, mime = _raw_or_base64_image(frame)
                if data:
                    return data, mime
    except Exception:
        pass

    # MP4/M4A covr atoms.
    try:
        covr = tags.get("covr")
        if covr:
            item = covr[0] if isinstance(covr, (list, tuple)) else covr
            data, mime = _raw_or_base64_image(item)
            if data:
                return data, mime
    except Exception:
        pass

    # FLAC/Ogg/Opus Picture objects.
    try:
        blocks = getattr(tags, "metadata_blocks", None) or tags.get("metadata_blocks")
        if blocks:
            for item in blocks:
                data, mime = _raw_or_base64_image(item)
                if data:
                    return data, mime
    except Exception:
        pass

    # Direct PICTURE / METADATA_BLOCK_PICTURE tags.
    for key in ("PICTURE", "METADATA_BLOCK_PICTURE"):
        try:
            pic = tags.get(key)
            if not pic:
                continue

            item = pic[0] if isinstance(pic, (list, tuple)) else pic
            data, mime = _raw_or_base64_image(item)
            if data:
                return data, mime
        except Exception:
            pass

    # Generic fallback for any tag that looks like embedded cover art.
    try:
        for key, value in list(tags.items()):
            ku = str(key).upper()
            if not any(token in ku for token in ("APIC", "COVR", "PICTURE")):
                continue

            items = value if isinstance(value, (list, tuple)) else [value]
            for item in items:
                data, mime = _raw_or_base64_image(item)
                if data:
                    return data, mime
    except Exception:
        pass

    return None, "image/jpeg"


def _ffprobe_tags(path):
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", path],
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode != 0:
            return {}

        data = json.loads(result.stdout or "{}")
        tags = (data.get("format") or {}).get("tags") or {}

        out = {}
        for k, v in tags.items():
            if isinstance(v, list):
                v = "; ".join(str(x) for x in v)
            out[str(k).lower()] = str(v).strip()

        return out
    except Exception:
        return {}


def _ffmpeg_attached_pic_index(path):
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", path],
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode != 0:
            return None

        data = json.loads(result.stdout or "{}")
        streams = data.get("streams", [])

        # Prefer an explicit attached_pic video stream.
        for i, stream in enumerate(streams):
            if str(stream.get("codec_type", "")).lower() != "video":
                continue

            disp = stream.get("disposition") or {}
            if str(disp.get("attached_pic", "0")) == "1" or "attached pic" in str(stream).lower():
                return i

        # Fallback: first video stream, because some builds expose cover art as v:0.
        for i, stream in enumerate(streams):
            if str(stream.get("codec_type", "")).lower() == "video":
                return i

        return None
    except Exception:
        return None


def _ffmpeg_extract_cover_packet(path):
    """
    Try to pull the attached-picture stream directly from ffprobe as base64 packet data.

    This avoids depending on ffmpeg re-encoding the cover and can work even when
    the ffmpeg map/scale path is finicky.
    """
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v", "quiet",
                "-select_streams", "v:0",
                "-show_packets",
                "-show_data",
                "-of", "json",
                path,
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode != 0:
            return None, "image/jpeg"

        data = json.loads(result.stdout or "{}")
        packets = data.get("packets") or []

        parts = []
        for packet in packets[:5]:
            b64 = packet.get("data", "")
            if not b64:
                continue
            try:
                parts.append(base64.b64decode(b64, validate=False))
            except Exception:
                pass

        raw = b"".join(parts)
        if raw and _guess_image_mime(raw) != "application/octet-stream":
            return raw, _guess_image_mime(raw)

        return None, "image/jpeg"
    except Exception:
        return None, "image/jpeg"


def _ffmpeg_extract_attached_picture(path):
    """0.8-pre31 VIDEO_NO_FRAME_COVER: a video file's embedded cover picture -- a stream marked as an
    attached picture -- as (bytes, mime); (None, "image/jpeg") when it has none. Never a frame of the
    video itself."""
    try:
        result = subprocess.run(["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", path],
                                capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            return None, "image/jpeg"
        streams = json.loads(result.stdout or "{}").get("streams", [])
        index = next((i for i, s in enumerate(streams)
                      if str(s.get("codec_type", "")).lower() == "video"
                      and str((s.get("disposition") or {}).get("attached_pic", "0")) == "1"), None)
        if index is None:
            return None, "image/jpeg"
        tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        tmp.close()
        try:
            cmd = ["ffmpeg", "-y", "-nostdin", "-hide_banner", "-loglevel", "error", "-i", path,
                   "-map", f"0:{index}", "-frames:v", "1", "-c:v", "mjpeg", "-q:v", COVER_IMAGE_QUALITY, tmp.name]
            done = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            if done.returncode == 0 and os.path.getsize(tmp.name) > 0:
                with open(tmp.name, "rb") as fh:
                    data = fh.read()
                if data:
                    return data, _guess_image_mime(data)
        finally:
            try:
                os.unlink(tmp.name)
            except OSError:
                pass
    except Exception:
        pass
    return None, "image/jpeg"


def _ffmpeg_extract_cover(path):
    # First try to pull the raw attached-picture packet directly from ffprobe.
    data, mime = _ffmpeg_extract_cover_packet(path)
    if data:
        return data, mime

    tmp_out = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    tmp_out.close()

    try:
        idx = _ffmpeg_attached_pic_index(path)

        maps = []
        if idx is not None:
            maps.append(f"0:{idx}")
        maps.extend(["0:v", "0:m"])

        seen = set()
        map_attempts = [m for m in maps if not (m in seen or seen.add(m))]

        last_err = ""

        # Try scaled extraction first, then unscaled extraction.
        for use_scale in (True, False):
            filters = _ffmpeg_scale_filter_variants(COVER_IMAGE_MAX_DIMENSION) if use_scale else [None]

            for m in map_attempts:
                for vf in filters:
                    cmd = [
                        "ffmpeg",
                        "-y",
                        "-nostdin",
                        "-hide_banner",
                        "-loglevel", "error",
                        "-i", path,
                        "-map", m,
                        "-frames:v", "1",
                    ]

                    if vf is not None:
                        cmd += ["-vf", vf]

                    # Use a valid mjpeg qscale value.
                    cmd += [
                        "-c:v", "mjpeg",
                        "-q:v", COVER_IMAGE_QUALITY,
                        tmp_out.name,
                    ]

                    try:
                        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
                    except Exception as e:
                        last_err = str(e)
                        continue

                    if (
                        result.returncode == 0
                        and os.path.exists(tmp_out.name)
                        and os.path.getsize(tmp_out.name) > 0
                    ):
                        with open(tmp_out.name, "rb") as f:
                            data = f.read()

                        if data:
                            return data, _guess_image_mime(data)

                    last_err = result.stderr or result.stdout

        # If all extraction attempts failed, this is useful for debugging.
        # Uncomment the next line temporarily if you are still seeing failures:
        # print(f"  (ffmpeg cover extraction failed: {last_err[-1000:]})")

        return None, "image/jpeg"
    except Exception:
        return None, "image/jpeg"
    finally:
        try:
            if os.path.exists(tmp_out.name):
                os.remove(tmp_out.name)
        except Exception:
            pass


def _tag_lyrics_usable(lyrics, metadata=None):
    """dev19 TAG_LYRICS_SANITY: False for a lyrics tag that holds no lyrics --
    fewer than 12 words, or one line naming the artist or title ("Artist - Title ( DJ remix )", the whole lyrics tag of one test file,
    which was then reported as the song's published lyrics and stopped the
    lyrics from being transcribed)."""
    t = str(lyrics or "").strip()
    if not t or not TAG_LYRICS_SANITY:
        return bool(t)
    words = re.findall(r"[^\W_]+(?:['\u2019][^\W_]+)?", t)
    if len(words) < 12:
        return False
    lines = [x for x in t.splitlines() if x.strip()]
    if len(lines) == 1 and len(words) < 25:
        low = t.lower()
        for k in ("artist", "title"):
            v = str((metadata or {}).get(k) or "").strip().lower() if isinstance(metadata, dict) else ""
            if len(v) >= 3 and re.search(r"(?<!\w)" + re.escape(v) + r"(?!\w)", low):
                return False
    return True


def _video_filename_qualifier(path):
    """Return only a trailing (video)/(music video) label on a video input."""
    stem, extension = os.path.splitext(os.path.basename(os.fspath(path or "")))
    if extension.lower() not in VIDEO_EXTENSIONS:
        return ""
    match = re.search(r"\s*(\((?:music\s+)?video\))\s*$", stem, re.I)
    return match.group(1) if match else ""


def _filename_metadata_fallback(path, metadata):
    """Fill missing artist/title from a local filename, after all embedded tags.

    Track numbers and generic export names are not identity evidence. Never
    infer album/year/lyrics, and never replace a nonblank embedded field.
    """
    if not path or str(path).startswith(("http://", "https://")):
        return
    if all(str(metadata.get(key) or "").strip() for key in ("artist", "title")):
        return
    stem = os.path.splitext(os.path.basename(os.fspath(path)))[0].strip()
    # Strip a leading track number before looking for an artist/title separator.
    stem = re.sub(r"^\d{1,3}(?:\s*[-–—.]\s*|\s+)", "", stem, count=1).strip()
    def usable(value):
        folded = re.sub(r"[\s_-]+", " ", value).strip()
        return bool(folded and not re.fullmatch(
            r"(?:\d+|(?:track|song|audio|video|untitled|unknown)(?: *\d+)?)",
            folded, re.I))
    parts = re.split(r"\s+[-–—]\s+", stem, maxsplit=1)
    artist, title = (parts[0].strip(), parts[1].strip()) if len(parts) == 2 else ("", stem)
    if FILENAME_VERSION_GUARD and artist and _FILENAME_VERSION_RE.fullmatch(title):
        # pre53: "Song - Live" names a version of one song, not an artist and a title.
        artist, title = "", stem
    # Video labels describe the file/version, not the inferred song title.
    qualifier = _video_filename_qualifier(path)
    if qualifier and title.lower().endswith(qualifier.lower()):
        title = title[:-len(qualifier)].rstrip()
    # A placeholder title cannot establish either identity field.
    if not usable(title):
        return
    guessed = []
    if not str(metadata.get("title") or "").strip():
        metadata["title"] = title
        guessed.append("title")
    if usable(artist) and not str(metadata.get("artist") or "").strip():
        metadata["artist"] = artist
        guessed.append("artist")
    if guessed and IDENTITY_FILENAME_NOTE:
        metadata["_from_filename"] = ", ".join(guessed)


_FILENAME_VERSION_RE = re.compile(
    r"[(\[]?\s*(?:live(?:\s+(?:version|recording|take))?|(?:\d{4}\s+)?remaster(?:ed)?(?:\s+(?:version|\d{4}))?"
    r"|demo(?:\s+version)?|acoustic(?:\s+version)?|instrumental|unplugged|mono|stereo|radio\s+edit|edit"
    r"|(?:single|album|extended|original|alternate|alternative)\s+(?:version|mix|take)|remix|mix|bonus\s+track"
    r"|(?:club|extended|original)\s+mix|version\s*\d*)\s*[)\]]?", re.I)


def extract_audio_metadata(path):
    meta = {
        "title": "",
        "artist": "",
        "album": "",
        "year": "",
        "lyrics": "",
    }

    if not path or path.startswith(("http://", "https://")):
        return meta, None, "image/jpeg"

    cover_bytes = None
    cover_mime = "image/jpeg"

    # Preferred path: mutagen.
    if METADATA_AVAILABLE and MutagenFile is not None:
        try:
            f = MutagenFile(path)
            tags = getattr(f, "tags", None)

            if tags is not None:
                meta["title"] = _first_tag(tags, TITLE_KEYS)
                meta["artist"] = _first_tag(tags, ARTIST_KEYS)
                meta["album"] = _first_tag(tags, ALBUM_KEYS)
                meta["year"] = _parse_year(_first_tag(tags, YEAR_KEYS))
                meta["lyrics"] = (
                    _collect_tags(tags, LYRICS_KEYS)
                    or _extract_id3_lyrics(tags)
                    or _extract_lyrics_from_any_tags(tags)
                )

                # MP4/M4A: ©lyr is the standard lyrics atom; also try freeform keys.
                if not meta["lyrics"]:
                    try:
                        for mp4_key in ("\xa9lyr", "©lyr", "----:com.apple.iTunes:Lyrics",
                                        "----:com.apple.iTunes:LYRICS",
                                        "----:com.apple.iTunes:UNSYNCED LYRICS"):
                            if mp4_key in tags:
                                meta["lyrics"] = _tag_text(tags.get(mp4_key))
                                if meta["lyrics"]:
                                    break
                        # Freeform atoms sometimes appear as tuple keys
                        if not meta["lyrics"]:
                            for k, v in list(tags.items()):
                                ks = str(k).lower()
                                if "lyric" in ks:
                                    meta["lyrics"] = _tag_text(v)
                                    if meta["lyrics"]:
                                        break
                    except Exception:
                        pass

                cover_bytes, cover_mime = _extract_cover_from_tags(tags)
        except Exception:
            pass

    # Fallback / fill-in path: ffprobe.
    if not all([meta["title"], meta["artist"], meta["album"], meta["year"]]) or not meta["lyrics"]:
        probe = _ffprobe_tags(path)

        if not meta["title"]:
            meta["title"] = _first_tag(probe, ["title", "musictitle"])
        if not meta["artist"]:
            meta["artist"] = _first_tag(probe, ["artist", "albumartist"])
        if not meta["album"]:
            meta["album"] = _first_tag(probe, ["album"])
        if not meta["year"]:
            meta["year"] = _parse_year(
                _first_tag(probe, [
                    "date", "year", "releasedate", "release date",
                    # Same original-release fallback as the mutagen path above,
                    # only reached when nothing else already supplied a year.
                    "originaldate", "originalyear", "original date", "original year",
                ])
            )
        if not meta["lyrics"]:
            # ffprobe lowercases keys; cover common lyric tag names including ©lyr variants
            meta["lyrics"] = _first_tag(
                probe,
                [
                    "lyrics", "unsyncedlyrics", "unsynced lyrics",
                    "©lyr", "\xa9lyr", "lyrc", "lyric",
                    "lyrics-eng", "lyrics-eng-xxx",
                ],
            )
            if not meta["lyrics"]:
                # Last resort: any probe key that looks like lyrics
                for pk, pv in (probe or {}).items():
                    if "lyric" in str(pk).lower() and str(pv).strip():
                        meta["lyrics"] = str(pv).strip()
                        break

    # If mutagen did not find cover art, try ffmpeg directly.
    if cover_bytes is None:
        if VIDEO_NO_FRAME_COVER and _is_video_input(path):    # 0.8-pre31: only a real embedded picture
            cover_bytes, cover_mime = _ffmpeg_extract_attached_picture(path)
        else:
            cover_bytes, cover_mime = _ffmpeg_extract_cover(path)

    # Keep lyrics from blowing up Ollama's context window.
    if meta["lyrics"] and len(meta["lyrics"]) > METADATA_MAX_LYRICS_CHARS:
        meta["lyrics"] = (
            meta["lyrics"][:METADATA_MAX_LYRICS_CHARS]
            + "\n[... truncated ...]"
        )

    if meta.get("lyrics") and not _tag_lyrics_usable(meta.get("lyrics"), meta):
        meta["lyrics"] = ""     # dev19 TAG_LYRICS_SANITY
    if VIDEO_IGNORE_TAG_ARTIST_TITLE and _is_video_input(path):
        # pre53: web videos often carry junk title/artist tags (URLs, uploader
        # names); a video's artist and title come from its filename only.
        meta["artist"] = meta["title"] = ""
    _filename_metadata_fallback(path, meta)
    return meta, cover_bytes, cover_mime


def _ffmpeg_scale_to_jpeg(image_bytes, max_dimension, quality):
    guessed_mime = _guess_image_mime(image_bytes)
    suffix = _cover_temp_suffix(guessed_mime)

    tmp_in = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    tmp_out = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)

    try:
        with open(tmp_in.name, "wb") as f:
            f.write(image_bytes)

        for vf in _ffmpeg_scale_filter_variants(max_dimension):
            cmd = [
                "ffmpeg",
                "-y",
                "-nostdin",
                "-hide_banner",
                "-loglevel", "error",
                "-i", tmp_in.name,
                "-frames:v", "1",
                "-vf", vf,
                "-c:v", "mjpeg",
                # quality must be a valid mjpeg qscale value, e.g. 2..31.
                "-q:v", str(quality),
                tmp_out.name,
            ]

            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            except Exception:
                continue

            if (
                result.returncode == 0
                and os.path.exists(tmp_out.name)
                and os.path.getsize(tmp_out.name) > 0
            ):
                with open(tmp_out.name, "rb") as f:
                    data = f.read()

                if data:
                    return data

        return None
    except Exception:
        return None
    finally:
        for p in (tmp_in.name, tmp_out.name):
            try:
                if os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass


# --- Optional Essentia integration -----------------------------------------
ESSENTIA_IMPORT_ERROR = ""
try:
    import essentia
    ESSENTIA_AVAILABLE = True
except Exception as _e:
    essentia = None
    ESSENTIA_AVAILABLE = False
    ESSENTIA_IMPORT_ERROR = str(_e)


def _essentia_optional_kernel(name):
    if not ESSENTIA_AVAILABLE or essentia is None:
        return None
    try:
        return getattr(essentia.standard, name)
    except Exception:
        return None


# Startup note for missing mutagen (was previously dead code after a return).
if ENABLE_FILE_METADATA and not METADATA_AVAILABLE:
    print(
        "  (mutagen is enabled but could not be imported; falling back to ffprobe/ffmpeg where possible. "
        f"{METADATA_IMPORT_ERROR})"
    )

RhythmExtractor2013 = _essentia_optional_kernel("RhythmExtractor2013")
KeyExtractor = _essentia_optional_kernel("KeyExtractor")
SpectralCentroid = _essentia_optional_kernel("SpectralCentroid")
SpectralFlatness = _essentia_optional_kernel("SpectralFlatness")
ZeroCrossingRate = _essentia_optional_kernel("ZeroCrossingRate")
RMS = _essentia_optional_kernel("RMS")
# ---------------------------------------------------------------------------


AUDIO_EXTENSIONS = (".mp3", ".wav", ".flac", ".m4a", ".aac", ".ogg", ".aiff", ".aif", ".wma")
VIDEO_EXTENSIONS = (
    ".mp4", ".m4v", ".mov", ".mkv", ".webm", ".avi", ".wmv", ".mpg", ".mpeg",
    ".ts", ".mts", ".m2ts", ".flv", ".3gp", ".3g2", ".vob", ".ogv", ".asf", ".mxf",
)
ANALYSIS_EXTENSIONS = AUDIO_EXTENSIONS + VIDEO_EXTENSIONS

# (LISTEN_FLAG moved to the settings file beside this script.)
# (RELISTEN_FLAG moved to the settings file beside this script.)


# (CORRECT_FLAG moved to the settings file beside this script.)
# (SAVE_FLAG moved to the settings file beside this script.)
# (LOAD_FLAG moved to the settings file beside this script.)
# (LOADCOMPARE_FLAG moved to the settings file beside this script.)
# (CLEAR_FLAG moved to the settings file beside this script.)
# (CLEAR_ALL_FLAG moved to the settings file beside this script.)
# (BATCH_FLAG moved to the settings file beside this script.)
# (PERSONA_FLAG moved to the settings file beside this script.)



def _is_video_input(path):
    return os.path.splitext(str(path))[1].lower() in VIDEO_EXTENSIONS


def convert_to_wav(input_path: str, sample_rate: int = 16000) -> str:
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    # dev2: -nostdin so ffmpeg never reads the terminal (keys typed while a
    # track is analysed -- a "q" stopped the conversion early but still
    # reported success); the temporary WAV is removed when conversion fails.
    result = subprocess.run(
        ["ffmpeg", "-nostdin", "-y", "-i", input_path, "-ar", str(sample_rate), "-ac", "1", tmp.name],
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
    )
    if result.returncode != 0:
        _remove_quietly(tmp.name)
        raise RuntimeError(f"ffmpeg conversion failed:\n{result.stderr}")
    return tmp.name


def _remove_quietly(path):
    """dev2: delete a temporary file, ignoring a file that is already gone."""
    try:
        if path and os.path.exists(path):
            os.remove(path)
    except OSError:
        pass


def _memory_readout():
    """One line describing memory use now (WHISPER_PROGRESS), or "" when
    nothing can be measured. Never raises."""
    parts = []
    try:
        import psutil
        _p = psutil.Process(os.getpid())
        parts.append(f"this process {_p.memory_info().rss / 1e9:.1f} GB")
    except Exception:
        _p = None
    try:
        if torch.backends.mps.is_available():
            _live = float(torch.mps.current_allocated_memory())
            _drv = float(torch.mps.driver_allocated_memory())
            parts.append(f"Apple GPU {_live / 1e9:.1f} GB in use + {max(0.0, _drv - _live) / 1e9:.1f} GB cached")
    except Exception:
        pass
    try:
        _proc = _OMNIZART_PREFETCH.get("proc")
        if _proc is not None and _proc.poll() is None:
            import psutil
            parts.append(f"Omnizart helper {psutil.Process(_proc.pid).memory_info().rss / 1e9:.1f} GB")
    except Exception:
        pass
    try:
        import psutil
        parts.append(f"Mac free {psutil.virtual_memory().available / 1e9:.1f} GB")
    except Exception:
        pass
    return "; ".join(parts)


def _is_batch_context():
    """True inside /batch folder scan or an isolated --batch-one child process."""
    if os.environ.get("MUSICLYSE_BATCH_ONE", "").strip() in ("1", "true", "True", "yes"):
        return True
    if os.environ.get("MUSICLYSE_IN_BATCH", "").strip() in ("1", "true", "True", "yes"):
        return True
    try:
        return len(sys.argv) >= 3 and sys.argv[1] in (
            "--batch-one", "--batch-one-analyze", "--batch-one-finish",
        )
    except Exception:
        return False


def _aggressive_memory_cleanup():
    """Best-effort process-wide memory release between batch tracks.

    Peak RAM during a single track (MOSS-Music + Demucs + Omnizart/TF) is
    already high; without cleanup between tracks, residual allocations stack
    until the OS kills the process (zsh 'killed'). Prefer BATCH_ISOLATE_PER_TRACK
    for long overnight runs — process exit is the only reliable MPS reset.
    """
    # Cached PANNs per-window probability matrices belong to the track that
    # just finished; they can never be reused across tracks (the cache key
    # includes the file identity), so drop them before the collect pass.
    try:
        _PANNS_WINDOW_CACHE.clear()
        del _PANNS_WINDOW_CACHE_ORDER[:]
        _PANNS_TAG_STATS.clear()
        del _PANNS_TAG_STATS_ORDER[:]
    except Exception:
        pass
    try:
        unload_moss_music()
    except Exception:
        pass
    try:
        unload_whisper()
    except Exception:
        pass
    try:
        _release_omnizart_memory()
    except Exception:
        pass
    try:
        ollama_unload_model()
    except Exception:
        pass

    for _ in range(4):
        gc.collect()
        try:
            if torch.backends.mps.is_available() and _gpu_cache_clear_ok():
                try:
                    torch.mps.synchronize()
                except Exception:
                    pass
                torch.mps.empty_cache()
            elif torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        try:
            import tensorflow as tf
            tf.keras.backend.clear_session()
        except Exception:
            pass
    _log_batch_rss("after cleanup")


try:
    from transformers import LogitsProcessor as _HFLogitsProcessor
except Exception:  # pragma: no cover - very old transformers
    _HFLogitsProcessor = object


def _status_suffix(suffix):
    """Redraw the current status line with a suffix (friendly display only),
    without starting a new stage in the timing table."""
    global _status_width
    if not FRIENDLY_PROCESSING_DISPLAY or _in_worker_thread():
        return
    try:
        label = str(_STAGE.get("label") or "")
        line = _colorize(f"  \u23f3 {label} ({suffix})", Ansi.YELLOW)
        pad = max(0, _status_width - len(line))
        print("\r" + line + " " * pad, end="", flush=True)
        _status_width = len(line)
    except Exception:
        pass


def _strip_md_emphasis(text):
    """Remove markdown emphasis/code marks the model sometimes wraps field
    labels in ("**LEAD_PROFILE**=...", "`VOCALS_PRESENT`: yes").

    Every structured-field parser below anchors on LABEL followed directly by
    "=" or ":", so a bolded label silently failed to parse. For LEAD_PROFILE
    that resolved to "unknown", which (as parse_vocal_tags documents) switches
    off both age guards; for GENRE_RANKED it discarded the model's ranking.
    Asterisks and backticks never appear inside the values, so dropping them is
    safe; underscores are left alone because they are part of the labels.
    """
    if not text:
        return text or ""
    return re.sub(r"[*`]+", "", str(text))


def _sentence(text, empty="n/a."):
    """`text` as one sentence ending in a single full stop (no "..")."""
    t = str(text or "").strip()
    if not t:
        return empty
    return t if t.endswith((".", "!", "?")) else t + "."


def _seconds_to_mmss(seconds):
    """Format seconds as M:SS, matching the STRUCTURE=[...] timestamp style
    used elsewhere in this file.

    Truncates like a media player's elapsed-time counter instead of rounding:
    a section that starts at 24.6 s is already playing when the player shows
    0:24, and rounding it to 0:25 made every reported start look late. A
    0.2 s allowance absorbs beat-grid jitter (24.85 s still reads 0:25).
    """
    seconds = max(0.0, float(seconds))
    total = int(math.floor(seconds + 0.2))
    m, s = divmod(total, 60)
    return f"{m}:{s:02d}"


# ---- starting Ollama (0.6.2-dev15) --------------------------------------------------

_OLLAMA_LAUNCH_MODES = ("off", "background", "background_log", "log_window", "window")
def _ollama_override_name():
    """This build's override file: this script's name with "-ollama-override.py"
    in place of ".py" (0.6.2-dev16)."""
    try:
        stem = os.path.splitext(os.path.basename(os.path.abspath(__file__)))[0]
    except Exception:
        stem = "Musiclyse"
    return f"{stem}-ollama-override.py"
_OLLAMA_OWNED = {"proc": None, "log_fh": None, "window_id": None}


def _ollama_override_values():
    """The plain values assigned in this build's override file next to this
    script ({} when it isn't there). Read as values, never run."""
    try:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), _ollama_override_name())
    except Exception:
        return {}
    if not os.path.isfile(path):
        return {}
    import ast as _ast
    found = {}
    try:
        tree = _ast.parse(open(path, encoding="utf-8").read(), filename=path)
        for node in tree.body:
            if (isinstance(node, _ast.Assign) and len(node.targets) == 1
                    and isinstance(node.targets[0], _ast.Name)):
                try:
                    found[node.targets[0].id] = _ast.literal_eval(node.value)
                except Exception:
                    pass
    except Exception as e:
        print(f"  ({_ollama_override_name()} could not be read ({e}) -- using the main script's settings.)")
        return {}
    found["_present"] = True
    return found


def _ollama_log_window_placement():
    """OLLAMA_LOG_WINDOW, or the override file's value: "behind" or "minimized"."""
    value = _ollama_override_values().get("OLLAMA_LOG_WINDOW", OLLAMA_LOG_WINDOW)
    return value if value in ("behind", "minimized") else "behind"


def _ollama_launch_mode():
    """OLLAMA_LAUNCH_MODE, or the value in this build's override file next to
    this script when that file is present (read as a value, never run)."""
    mode = OLLAMA_LAUNCH_MODE
    over = _ollama_override_values()
    if over.get("_present"):
        found = over.get("OLLAMA_LAUNCH_MODE")
        if found in _OLLAMA_LAUNCH_MODES:
            mode = found
        else:
            print(f"  ({_ollama_override_name()}: no valid OLLAMA_LAUNCH_MODE in it -- using {mode!r}.)")
    if mode not in _OLLAMA_LAUNCH_MODES:
        mode = "off"
    if mode in ("log_window", "window") and sys.platform != "darwin":
        mode = "background_log"
    return mode


def _ollama_log_path():
    """OLLAMA_LAUNCH_LOG as a full path (a relative one is inside this script's folder)."""
    path = os.path.expanduser(str(OLLAMA_LAUNCH_LOG or "logs/ollama.log"))
    if not os.path.isabs(path):
        try:
            path = os.path.join(os.path.dirname(os.path.abspath(__file__)), path)
        except Exception:
            path = os.path.abspath(path)
    return path


def _ollama_reachable(timeout=2):
    try:
        requests.get(OLLAMA_BASE_URL, timeout=timeout)
        return True
    except Exception:
        return False


def _ollama_binary():
    found = shutil.which("ollama")
    if found:
        return found
    for cand in ("/Applications/Ollama.app/Contents/Resources/ollama", "/opt/homebrew/bin/ollama",
                 "/usr/local/bin/ollama", os.path.expanduser("~/Applications/Ollama.app/Contents/Resources/ollama")):
        if os.path.isfile(cand) and os.access(cand, os.X_OK):
            return cand
    return None


def _ollama_launch_env():
    """The environment Ollama is started with: OLLAMA_LAUNCH_ENV, listening where
    Musiclyse talks to it, and with request-content logging off."""
    env = dict(os.environ)
    env.update({str(k): str(v) for k, v in (OLLAMA_LAUNCH_ENV or {}).items()})
    try:
        import urllib.parse as _up
        netloc = _up.urlparse(OLLAMA_BASE_URL).netloc
        if netloc:
            env["OLLAMA_HOST"] = netloc
    except Exception:
        pass
    env.pop("OLLAMA_DEBUG", None)                 # default detail, as an Ollama window shows
    env["OLLAMA_DEBUG_LOG_REQUESTS"] = "false"    # never log what was sent
    return env


def _osascript(lines, *args):
    cmd = ["osascript"]
    for line in lines:
        cmd += ["-e", line]
    out = subprocess.run(cmd + list(args), capture_output=True, text=True, timeout=20)
    return out.stdout.strip()


_TERMINAL_OPEN_SCRIPT = [
    "on run argv",
    'tell application "Terminal"',
    "set orig to missing value",
    "if (count of windows) > 0 then set orig to id of front window",
    "set t to do script (item 1 of argv)",
    "set custom title of t to (item 2 of argv)",
    "delay 0.3",                                   # let Terminal finish opening it
    "set newId to id of front window",
    "try",
    "set newId to id of (first window whose selected tab is t)",
    "end try",
    'if (item 3 of argv) is "minimized" then',
    "try",
    "set miniaturized of window id newId to true",
    "end try",
    "end if",
    "if orig is not missing value and orig is not newId then",
    "repeat 3 times",
    "try",
    "set index of window id orig to 1",
    "set frontmost of window id orig to true",
    "end try",
    "delay 0.2",
    "if id of front window is orig then exit repeat",
    "end repeat",
    "end if",
    "return newId",
    "end tell",
    "end run",
]


def _ollama_open_terminal(command, title, placement="front"):
    """Run command in a new Terminal window, leaving Musiclyse's window in front
    (placement "behind" or "minimized"; "front" for the standalone server window
    still puts Musiclyse's window back in front). Returns the new window's id, or None."""
    try:
        got = _osascript(_TERMINAL_OPEN_SCRIPT, command, title, placement)
        return int(got) if got.strip().isdigit() else None
    except Exception:
        return None


def _ollama_stop_owned():
    """Stop the Ollama Musiclyse started (and close its log window). Safe to call twice."""
    proc = _OLLAMA_OWNED.get("proc")
    if proc is not None:
        _OLLAMA_OWNED["proc"] = None
        try:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except Exception:
                    proc.kill()
        except Exception:
            pass
    fh = _OLLAMA_OWNED.get("log_fh")
    if fh is not None:
        _OLLAMA_OWNED["log_fh"] = None
        try:
            fh.close()
        except Exception:
            pass
    wid = _OLLAMA_OWNED.get("window_id")
    if wid is not None:
        _OLLAMA_OWNED["window_id"] = None
        try:
            time.sleep(1.5)     # the log window's own loop ends once Ollama has stopped
            _osascript(['tell application "Terminal"',
                        f"if (exists window id {int(wid)}) then close window id {int(wid)}",
                        "end tell"])
        except Exception:
            pass


def _ollama_autostart():
    """Start Ollama per the launch mode when none is running. Never touches an
    Ollama that is already running."""
    mode = _ollama_launch_mode()
    if mode == "off" or _ollama_reachable():
        return
    binary = _ollama_binary()
    if not binary:
        print(f"  (OLLAMA_LAUNCH_MODE is {mode!r}, but the ollama command wasn't found -- "
              "start Ollama yourself.)")
        return
    env = _ollama_launch_env()
    if mode == "window":
        import shlex as _shlex
        exports = " ".join(f"{k}={_shlex.quote(v)}" for k, v in
                           {**{str(a): str(b) for a, b in (OLLAMA_LAUNCH_ENV or {}).items()},
                            "OLLAMA_HOST": env.get("OLLAMA_HOST", ""),
                            "OLLAMA_DEBUG_LOG_REQUESTS": "false"}.items() if v != "")
        command = f"unset OLLAMA_DEBUG; export {exports}; {_shlex.quote(binary)} serve"
        if _ollama_open_terminal(command, "Ollama (started by Musiclyse)") is None:
            print("  (Couldn't open a Terminal window for Ollama -- start it yourself.)")
            return
        started = "Opened Ollama in its own Terminal window (it keeps running after you leave Musiclyse)."
    else:
        log_path = _ollama_log_path() if mode in ("background_log", "log_window") else None
        out = subprocess.DEVNULL
        if log_path:
            try:
                os.makedirs(os.path.dirname(log_path), exist_ok=True)
                out = open(log_path, "w", encoding="utf-8", errors="replace")
                _OLLAMA_OWNED["log_fh"] = out
            except Exception as e:
                print(f"  (Couldn't open the Ollama log {log_path}: {e} -- starting it without a log.)")
                out, log_path = subprocess.DEVNULL, None
        try:
            proc = subprocess.Popen([binary, "serve"], env=env, stdin=subprocess.DEVNULL,
                                    stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
        except Exception as e:
            print(f"  (Couldn't start Ollama: {e} -- start it yourself.)")
            _ollama_stop_owned()
            return
        _OLLAMA_OWNED["proc"] = proc
        import atexit as _atexit
        _atexit.register(_ollama_stop_owned)
        try:
            import signal as _signal

            def _leave(signum, frame):
                raise SystemExit(0)
            for _sig in (getattr(_signal, "SIGHUP", None), _signal.SIGTERM):
                if _sig is not None:
                    _signal.signal(_sig, _leave)
        except Exception:
            pass
        if mode == "log_window" and log_path:
            import shlex as _shlex
            viewer = (f"tail -n +1 -f {_shlex.quote(log_path)} & T=$!; "
                      f"while kill -0 {proc.pid} 2>/dev/null; do sleep 1; done; "
                      f"kill $T 2>/dev/null; exit")
            _OLLAMA_OWNED["window_id"] = _ollama_open_terminal(viewer, "Ollama log (Musiclyse)",
                                                               _ollama_log_window_placement())
        started = "Started Ollama in the background; it stops when you leave Musiclyse."
        if mode == "log_window" and _OLLAMA_OWNED.get("window_id") is not None:
            started += (" Its log is in a separate Terminal window"
                    + (" (minimised to the Dock)" if _ollama_log_window_placement() == "minimized" else "")
                    + ", which closes with it.")
        elif log_path:
            started += f" Log: {log_path} (watch it with: tail -f {log_path})"
    for _ in range(60):
        if _ollama_reachable(timeout=1):
            break
        if _OLLAMA_OWNED.get("proc") is not None and _OLLAMA_OWNED["proc"].poll() is not None:
            print("  (Ollama stopped straight after starting"
                  + (f" -- see {_ollama_log_path()}" if mode not in ("background", "window") else "") + ".)")
            _ollama_stop_owned()
            return
        time.sleep(0.5)
    else:
        print("  (Ollama was started but hasn't answered after 30 seconds.)")
        return
    print(_colorize(f"  {started}", Ansi.YELLOW))


def check_ollama_running():
    try:
        requests.get("http://localhost:11434", timeout=2)
    except requests.exceptions.ConnectionError:
        print(
            "\nCan't reach Ollama at localhost:11434.\n"
            "Start it with 'ollama serve' (or open the Ollama app), then re-run.\n"
        )
        sys.exit(1)

def _sample_list_evenly(items, max_items):
    if max_items is None or len(items) <= max_items:
        return list(items)

    if max_items <= 1:
        return items[:1]

    step = (len(items) - 1) / float(max_items - 1)
    idxs = sorted({int(round(i * step)) for i in range(max_items)})
    return [items[i] for i in idxs if 0 <= i < len(items)]


def _estimate_ollama_message_chars(msg):
    try:
        content = msg.get("content") or ""
        if isinstance(content, list):
            content = json.dumps(content)

        n = len(str(content))

        imgs = msg.get("images") or []
        for img in imgs:
            try:
                # Base64 image payloads are not 1:1 with tokens, but this gives a
                # conservative size estimate for trimming.
                n += max(1000, len(str(img)) // 3)
            except Exception:
                pass

        return n
    except Exception:
        return 0


def _truncate_text_for_ollama(text, limit):
    text = str(text or "")

    if limit <= 1000:
        return text[:max(0, int(limit))]

    if len(text) <= limit:
        return text

    marker = "\n\n[... truncated for Ollama context ...]\n\n"
    head = int(limit * 0.72)
    tail = limit - head - len(marker)

    # Never exceed limit, including the small-context fallback cases.
    if tail <= 0:
        return text[:max(0, int(limit))]
    return text[:head] + marker + text[-tail:]


def _ollama_output_budget(num_ctx, max_output_tokens):
    """num_predict for one request at this context size.

    Ollama needs prompt + num_predict <= num_ctx. The fallback ladder in
    ollama_chat() steps down to 16384 and 8192, and batch-mode cover-art /
    singer-identity calls use 8192 -- all with the full 16384-token reply
    budget, which left no room at all for the prompt. The reply budget is
    now at most half the window, so every attempt has room for a prompt.
    Unchanged at 32768 and above (the normal interactive attempts).
    """
    return max(256, min(int(max_output_tokens), int(num_ctx) // 2))


def _prepare_ollama_messages(messages, num_ctx, keep_images=True, max_output_tokens=16384):
    """Build a safer copy of the message list for Ollama.

    This does not mutate the original writer_history. It:
      - keeps the system prompt but truncates it if necessary;
      - keeps the newest user message even if it is large;
      - drops older messages that would exceed the context budget;
      - compresses stem/MIDI event logs;
      - limits images to a small number, preferring newest messages.
    """
    keep_images = bool(keep_images) and OLLAMA_SUPPORTS_IMAGES

    messages = list(messages or [])
    if not messages:
        return []

    # Ollama requires prompt_tokens + num_predict <= num_ctx server-side, but
    # this budget used to be computed off the full num_ctx with no headroom
    # reserved for the reply. That let the assembled prompt keep growing as
    # if the whole window were available for history, right up until a real
    # conversation's prompt got close enough to num_ctx that adding the
    # reserved max_output_tokens pushed the actual request over the model's
    # limit -- which Ollama rejects with a 400, triggering a fallback to a
    # SMALLER num_ctx. That fallback is a full model reload (num_ctx is set
    # at load time), discarding the entire cached conversation, and the
    # smaller window then requires dropping much more history too. Budgeting
    # off the space actually left for the prompt keeps requests inside the
    # window that's already loaded, so the first (preferred, cache-friendly)
    # attempt keeps succeeding as the conversation grows instead of only
    # failing once it's already large enough for the loss to be expensive.
    effective_ctx = max(1, int(num_ctx) - int(max_output_tokens))
    budget = max(1, int(effective_ctx * HISTORY_CHAR_BUDGET_FACTOR))
    # Reserve space for a truthful warning when evidence has to be omitted.
    omission_note = ("\n\nCONTEXT LIMIT: Some earlier evidence was omitted to fit this request. "
                     "Use only evidence actually present; do not infer missing track details "
                     "from earlier assistant answers. Say when a comparison lacks evidence.")
    notice_reserve = min(len(omission_note), budget // 10)
    budget -= notice_reserve
    prepared = []
    start = 0

    if messages[0].get("role") == "system":
        sys_msg = dict(messages[0])
        # The system prompt (persona + WRITER_MUSIC_RULES) is ~36.6k chars with the
        # default persona. A former fixed 30k ceiling here truncated it on EVERY
        # request, silently cutting the middle out of the music rules (the
        # timestamp-precision, drum-grid-quoting, MIDI-handling and melody-authority
        # sections). The proportional guard is what actually matters — it stops the
        # system prompt from crowding out history on small contexts — so keep that
        # and drop the fixed cap. At OLLAMA_NUM_CTX=131072 this allows ~146k chars,
        # so the prompt now goes out whole and costs ~8.7% of the char budget.
        sys_limit = max(0, int(budget * 0.35))
        sys_msg.pop("images", None)
        sys_msg["content"] = _truncate_text_for_ollama(sys_msg.get("content"), sys_limit)

        prepared.append(sys_msg)
        budget -= _estimate_ollama_message_chars(sys_msg)
        start = 1

    def _process_message(msg):
        m = dict(msg)

        content = m.get("content") or ""
        if isinstance(content, list):
            content = json.dumps(content)

        # STABLE per-message cap: derived from num_ctx alone, never from the
        # running budget.
        #
        # This used to be `budget // 3`, where `budget` decrements as the loop
        # walks the history newest-first. The big loaded-analysis message is
        # the OLDEST, so it was processed last, against whatever budget was
        # left -- and that remainder shrank on every turn as the conversation
        # grew. The same message therefore got a different truncation point
        # (and, with adaptive compression, a different sampling level) each
        # turn, so its text changed even though its content had not.
        #
        # For a server with a prefix cache that is the worst possible property:
        # a prompt prefix is only reusable if it is byte-identical, so one
        # shifting message near the START invalidates everything after it and
        # the whole tail is re-processed every turn. On a long /loadcompare
        # session that meant ~40k tokens of prompt re-processing per message,
        # about a minute of wall time before generation could even begin, with
        # the model and machine sitting well inside their limits.
        #
        # Keying the cap to num_ctx makes each message truncate identically on
        # every turn, so the prefix stays stable and the cache does its job.
        # The budget still governs which messages are KEPT (below); it just no
        # longer reaches inside a message and reshapes it.
        max_msg_chars = min(
            MAX_MESSAGE_CHARS_FOR_OLLAMA,
            max(4000, int(int(num_ctx * HISTORY_CHAR_BUDGET_FACTOR) // 3)),
        )

        content = _compress_stem_midi_event_logs(
            str(content),
            40 if num_ctx > 16384 else 25,
        )
        # If one pass was not enough, sample the note logs harder rather than
        # letting the blind truncator take the middle out. Note lists are ~64%
        # of a saved analysis and are already a sample; the per-section groove,
        # beat-grid, melody and intensity blocks are not reconstructible and
        # must survive. Losing note resolution costs detail, losing those
        # blocks makes the model report them as absent, which is worse: on a
        # two-track /loadcompare it answered "there is no BEAT GRID for the
        # chorus" and "no MELODY PER SECTION block" about files that contained
        # seven grids and five melody blocks.
        for _level in (20, 12, 8, 5, 3):
            if len(content) <= max_msg_chars:
                break
            content = _compress_stem_midi_event_logs(str(content), _level)
        content = _truncate_text_for_ollama(content, max_msg_chars)

        m["content"] = content

        imgs = m.get("images") or []
        if keep_images:
            m["images"] = [str(img) for img in list(imgs)[:MAX_WRITER_IMAGES_PER_TURN]]
        else:
            m.pop("images", None)

        return m

    rest = messages[start:]
    true_newest_idx = len(rest) - 1 if rest else None

    # Keep the question first, then prefer recent protected evidence. Protection
    # is a priority, never permission to exceed the request's budget.
    processed = {}
    protected_indices = [i for i, msg in enumerate(rest) if id(msg) in _PROTECTED_EVIDENCE_MESSAGE_IDS]
    if true_newest_idx is not None:
        m = _process_message(rest[true_newest_idx])
        if _estimate_ollama_message_chars(m) > budget:
            m.pop("images", None)
            m["content"] = _truncate_text_for_ollama(m["content"], budget)
        processed[true_newest_idx] = m
        budget -= _estimate_ollama_message_chars(m)
    priority = list(reversed(protected_indices)) + list(range(len(rest) - 1, -1, -1))
    omitted = False
    for i in dict.fromkeys(priority):
        if i in processed:
            continue
        m = _process_message(rest[i])
        cost = _estimate_ollama_message_chars(m)
        if cost > budget:
            omitted = True
            continue
        processed[i] = m
        budget -= cost
    if omitted and true_newest_idx is not None and notice_reserve:
        processed[true_newest_idx]["content"] += omission_note[:notice_reserve]
    kept_indices = list(processed)

    kept_indices.sort()
    prepared.extend(processed[i] for i in kept_indices)

    # Enforce a global image limit across the whole request, preferring newest messages.
    total_images = 0
    for m in reversed(prepared):
        imgs = m.get("images") or []
        if not imgs:
            continue

        room = MAX_WRITER_IMAGES_PER_TURN - total_images
        if room <= 0:
            m.pop("images", None)
        else:
            m["images"] = imgs[:room]
            total_images += len(m["images"])

    return prepared


def warn_if_system_prompt_truncated(system_text, num_ctx=None, max_output_tokens=16384):
    """Print a warning if the system prompt is large enough to be truncated.

    Purely diagnostic: computes the same `sys_limit` that
    _prepare_ollama_messages() uses and reports when the prompt would not
    survive intact. Mirrors that calculation rather than sharing it, so this
    check can never affect what is actually sent. Returns True if the prompt
    would be truncated.

    This exists because truncation here is otherwise silent — a fixed 30k cap
    used to cut ~6.6k chars out of the middle of WRITER_MUSIC_RULES on every
    request (losing the timestamp-precision, drum-grid, MIDI-handling and
    melody-authority sections) with no indication it was happening.
    """
    try:
        text = str(system_text or "")
        if not text:
            return False

        ctx = int(num_ctx or OLLAMA_NUM_CTX)
        effective_ctx = max(2048, ctx - _ollama_output_budget(ctx, max_output_tokens))
        budget = int(effective_ctx * HISTORY_CHAR_BUDGET_FACTOR)
        sys_limit = max(6000, int(budget * 0.35))

        if len(text) <= sys_limit:
            return False

        print(_colorize(
            f"\n  WARNING: the system prompt is {len(text):,} chars but the limit at "
            f"num_ctx={ctx:,} is {sys_limit:,}.\n"
            f"  About {len(text) - sys_limit:,} chars will be cut from the MIDDLE of the "
            f"music rules on every request.\n"
            f"  Raise OLLAMA_NUM_CTX, or shorten the persona / WRITER_MUSIC_RULES.\n",
            Ansi.YELLOW,
        ))
        return True
    except Exception:
        return False


# Identities (via id()) of writer_history messages currently registered in
# track_evidence_message -- the message each track's full evidence block
# most recently went out in. Kept in sync at every place track_evidence_message
# itself is set/cleared/popped.
#
# Protection prioritises stable evidence for prefix-cache reuse, but both
# history compaction and request preparation enforce their limits. A missing
# evidence message is re-sent on a later /listen turn; an omitted block is
# explicitly disclosed to the writer instead of silently exceeding context.
_PROTECTED_EVIDENCE_MESSAGE_IDS = set()


def _compact_writer_history_in_place(history, *, enforce_message_limit=True):
    """Reduce Python-side memory used by writer_history.

    This is called before Ollama requests and at the start of each main loop turn.
    It removes old base64 images, compresses large stem/MIDI logs, and caps the
    number of stored messages.
    """
    try:
        if not history:
            return

        if not OLLAMA_SUPPORTS_IMAGES:
            for m in history:
                m.pop("images", None)
        else:
            # Retention is a MESSAGE-DISTANCE window, not "the last N images".
            # An earlier version kept the last MAX_STORED_IMAGES_IN_HISTORY
            # image-bearing messages, but with a single image shared once in a
            # session, that one message is simultaneously the newest AND the
            # only image-bearing message on every future turn -- so it was
            # never pruned. Some inference backends cannot restore their
            # prefix cache across a prompt containing image data, so that one
            # unpruned image forced a full prompt reprocess on every
            # subsequent turn for the rest of the session, however many turns
            # followed with no image in them at all. Expiring by how many
            # MESSAGES have arrived since (not how many other images have)
            # is what actually bounds this: after IMAGE_RETAIN_LAST_N_MESSAGES
            # more messages, the bytes go, regardless of whether a second
            # image was ever shared.
            cutoff = len(history) - int(IMAGE_RETAIN_LAST_N_MESSAGES)
            for i, m in enumerate(history):
                if "images" not in m:
                    continue

                if i < cutoff:
                    m.pop("images", None)
                else:
                    m["images"] = [
                        str(img)
                        for img in list(m.get("images") or [])[:MAX_WRITER_IMAGES_PER_TURN]
                    ]

        for m in history:
            content = m.get("content")
            # Guard must match every label _compress_stem_midi_event_logs can
            # handle. It previously tested only "events (", which the current
            # full-list format does not contain, so stored note lists were never
            # compacted in history either.
            if (
                isinstance(content, str)
                and len(content) > 20000
                and any(marker in content for marker in _MIDI_LOG_MARKERS)
            ):
                m["content"] = _compress_stem_midi_event_logs(content, 30)

        if enforce_message_limit and len(history) > MAX_WRITER_HISTORY_MESSAGES:
            system = history[0] if history and history[0].get("role") == "system" else None
            rest = history[1:] if system is not None else history[:]

            keep_count = (
                MAX_WRITER_HISTORY_MESSAGES - 1
                if system is not None
                else MAX_WRITER_HISTORY_MESSAGES
            )

            # Evidence messages get priority after the newest message; all
            # retained messages count toward the history cap. Total kept is still
            # capped at keep_count, so this does not raise how much history
            # is retained overall -- it only changes which messages fill
            # those slots.
            newest = rest[-1:] if keep_count > 0 else []
            newest_ids = {id(m) for m in newest}
            protected_rest = [m for m in rest if id(m) in _PROTECTED_EVIDENCE_MESSAGE_IDS
                              and id(m) not in newest_ids]
            slots = max(0, keep_count - len(newest))
            protected_rest = protected_rest[-slots:] if slots else []
            keep_ids = newest_ids | {id(m) for m in protected_rest}
            recent_slots = max(0, keep_count - len(keep_ids))
            unprotected_rest = [m for m in rest if id(m) not in keep_ids]
            recent_kept = unprotected_rest[-recent_slots:] if recent_slots else []
            keep_ids.update(id(m) for m in recent_kept)
            kept_rest = [m for m in rest if id(m) in keep_ids]

            history.clear()
            if system is not None:
                history.append(system)
            history.extend(kept_rest)

    except Exception as e:
        print(f"  (writer-history compaction skipped: {e})")


class _OllamaReplyInterrupted(RuntimeError):
    """dev2: a streamed chat reply stopped partway (an Ollama error line or a
    dropped connection after text was printed). Not retried, not kept."""


def _ollama_response_error(resp):
    """Extract a human-readable error from an Ollama HTTP response."""
    if resp is None:
        return "no response"

    try:
        data = resp.json()
        if isinstance(data, dict) and data.get("error"):
            return str(data["error"])
    except Exception:
        pass

    try:
        text = (resp.text or "").strip()
        if text:
            return text[:1000]
    except Exception:
        pass

    try:
        return str(resp.reason)
    except Exception:
        return "unknown Ollama error"


def _print_token_usage(usage):
    """Show last-turn and session token use against the configured context window."""
    if not usage:
        return
    # Accept either a per-turn dict from ollama_chat or the session accumulator.
    if usage is SESSION_TOKEN_USAGE or (
        isinstance(usage, dict) and "last_prompt" in usage and "prompt" in usage
        and usage is not None and "completion" in usage and "ctx" not in usage
    ):
        last_p = int(SESSION_TOKEN_USAGE.get("last_prompt") or 0)
        last_c = int(SESSION_TOKEN_USAGE.get("last_completion") or 0)
        last_ctx = int(SESSION_TOKEN_USAGE.get("last_ctx") or OLLAMA_NUM_CTX or 0)
        total = int(SESSION_TOKEN_USAGE.get("total") or 0)
    else:
        last_p = int(usage.get("prompt") or 0)
        last_c = int(usage.get("completion") or 0)
        last_ctx = int(usage.get("ctx") or OLLAMA_NUM_CTX or 0)
        total = int(SESSION_TOKEN_USAGE.get("total") or (last_p + last_c))
    print(
        f"  (tokens: this turn ~{last_p + last_c} "
        f"[prompt {last_p} + reply {last_c}] · "
        f"session ~{total} · context window {last_ctx})"
    )


# dev58: see WRITER_CASUAL_VOICE. Sentence-opening clauses that cite where a
# fact came from. Only the opener is removed, and only when a full clause
# follows it ("According to the background, they formed in 1976." -> "They
# formed in 1976."); a bare "The notes show a tempo of 120" is left alone, as
# removing its opener would leave a fragment.
_NATURAL_ATTR_STATE = {"on": True}
_ATTR_SRC = (
    r"(?:(?:the|my|our|these|those|your|this)\s+(?:private\s+|track\s+|technical\s+|background\s+|saved\s+)?"
    r"(?:notes?|background(?:\s+(?:info(?:rmation)?|notes?|article|context))?|analysis|data|measurements?|"
    r"metadata|tags|scan|article|wiki(?:pedia)?(?:\s+(?:article|entry|page))?|sources?|readings?|"
    r"evidence|report|info(?:rmation)?(?:\s+I\s+have)?|(?:file|track)\s+info)"
    r"|wikipedia|what\s+I\s+have(?:\s+here)?|what\s+I\s+can\s+see|the\s+information\s+available)"
)
_ATTR_TAIL = r"(?:\s+(?:here|on\s+(?:this|the|that)\s+(?:track|song|record|band|artist|album)))?"
_SENT_START = r"(?:(?<=^)|(?<=\n)|(?<=[.!?]\s)|(?<=[.!?]\)\s)|(?<=[.!?][\"\u201d]\s))(?P<lead>[ \t]*(?:[-*\u2022]\s+)?)"
_ATTR_OPENER_RES = (
    re.compile(_SENT_START + r"(?:according\s+to|based\s+on|going\s+(?:by|off)|judging\s+by|per|from)\s+"
               + _ATTR_SRC + _ATTR_TAIL + r"\s*,\s+(?=[\w\"'\u2018\u201c(])", re.IGNORECASE),
    re.compile(_SENT_START + _ATTR_SRC + _ATTR_TAIL
               + r"\s+(?:say|says|said|show|shows|showed|indicate|indicates|suggest|suggests|mention|"
                 r"mentions|note|notes|state|states|confirm|confirms|tell\s+me|tells\s+me|has\s+it|"
                 r"have\s+it|put\s+it)\s+that\s+(?=[\w\"'\u2018\u201c(])", re.IGNORECASE),
)
_ATTR_INLINE_RES = (
    # ", according to the notes," / " (according to Wikipedia)" inside a sentence
    (re.compile(r",\s+according\s+to\s+" + _ATTR_SRC + _ATTR_TAIL + r"\s*(?=[,.;:!?])", re.IGNORECASE), ""),
    (re.compile(r"\s+\(\s*(?:according\s+to|per|via|from)\s+" + _ATTR_SRC + _ATTR_TAIL + r"\s*\)",
                re.IGNORECASE), ""),
)


def _natural_attribution(text):
    """dev58: drop source-citing sentence openers from a writer reply (see
    WRITER_CASUAL_VOICE). Returns the text unchanged when the setting is off,
    or when the person's question was about sources."""
    if (not text or not globals().get("WRITER_CASUAL_VOICE")
            or not _NATURAL_ATTR_STATE.get("on", True)):
        return text
    out = text

    def _drop(m):
        return m.group("lead")

    for rx in _ATTR_OPENER_RES:
        out = rx.sub(_drop, out)
    for rx, rep in _ATTR_INLINE_RES:
        out = rx.sub(rep, out)
    if out != text:
        # Capitalise the word that now starts each sentence.
        out = re.sub(r"(?:(?<=^)|(?<=\n)|(?<=[.!?]\s)|(?<=[.!?]\)\s)|(?<=[.!?][\"\u201d]\s))"
                     r"([ \t]*(?:[-*\u2022]\s+)?)([\"'\u2018\u201c(]?)([a-z])",
                     lambda m: m.group(1) + m.group(2) + m.group(3).upper(), out)
    return out


def _humanize_vocal_category_language(text):
    """Rewrite internal vocal-category vocabulary into ordinary music talk.

    The writer prompt forbids saying "post-puberty male", "adolescent_male_likely"
    and similar, but the tags sit in the private notes the writer is reading, so
    smaller models echo them anyway. This is the deterministic backstop.

    Applied ONLY to the writer's user-facing reply (see _safe_ollama_chat).
    Internal passes call ollama_chat() directly and keep their raw tags, so no
    parsing or classification logic is affected by this.
    """
    if not text:
        return text
    if not globals().get("SANITIZE_VOCAL_CATEGORY_LANGUAGE", True):
        return _natural_attribution(text)

    def _match_case(original, replacement):
        if original[:1].isupper():
            return replacement[:1].upper() + replacement[1:]
        return replacement

    # Longest / most specific patterns first.
    # The bare forms only get rewritten when they are actually describing a
    # voice (or end the clause). Otherwise legitimate terminology is mangled --
    # "a prepubescent choir" is the correct term for a boys' choir and must
    # survive, while "the lead reads as post-pubertal." should not.
    _voice_ctx = (
        r"(?=\s+(?:male|female|voice|vocal|vocals|lead|leads|singer|singers|"
        r"tone|timbre|delivery|register)\b|\s*[.,;:!?]|$)"
    )
    substitutions = (
        (r"\bpost[-\s]?pub(?:erty|ertal|escent)[-\s]+male\b", "adult male"),
        (r"\bpost[-\s]?pub(?:erty|ertal|escent)\b" + _voice_ctx, "adult"),
        (r"\bpre[-\s]?pub(?:erty|ertal|escent)\b" + _voice_ctx, "childlike"),
        (r"\badolescent_male_likely\b", "young, still-changing male"),
        (r"\bpost_puberty_male\b", "adult male"),
        (r"\bchild_male_likely\b", "young male"),
        (r"\bchild_female_likely\b", "young female"),
        (r"\bchild_gender_uncertain\b", "young"),
        (r"\bfemale_teen_adult\b", "female"),
        (r"\bmixed_leads\b", "two lead singers"),
        # The same tags written with hyphens or spaces ("a female-teen-adult
        # voice", "adolescent male likely"), and the vocal pass's
        # LEAD_PROFILE spelling; also the arbitration block's name.
        (r"\bgirl[-_\s]female[-_\s]teen[-_\s]adult\b", "female"),
        (r"\bfemale[-_\s]teen[-_\s]adult\b", "female"),
        (r"\badolescent[-_\s]male[-_\s]likely\b", "young, still-changing male"),
        (r"\bchild[-_\s]male[-_\s]likely\b", "young male"),
        (r"\bchild[-_\s]female[-_\s]likely\b", "young female"),
        (r"\bchild[-_\s]gender[-_\s]uncertain\b", "young"),
        (r"\bmixed-leads\b", "two lead singers"),
        (r"\bvocal[-\s]presence arbitration\b", "vocal check"),
        (r"\badult_male\b", "adult male"),
        (r"\badult_female\b", "adult female"),
    )

    out = text
    for pattern, replacement in substitutions:
        out = re.sub(
            pattern,
            lambda m, r=replacement: _match_case(m.group(0), r),
            out,
            flags=re.IGNORECASE,
        )

    # Substitution can leave the wrong indefinite article behind ("a
    # post-puberty male" -> "a adult male", "an adolescent_male_likely" ->
    # "an young ..."). Repair only in front of the words this function
    # actually introduces, so ordinary prose is never touched.
    def _fix_article(m, correct):
        article = m.group(1)
        article = correct[:1].upper() + correct[1:] if article[:1].isupper() else correct
        return f"{article} {m.group(2)}"

    out = re.sub(
        r"\b(a)\s+(adult)\b",
        lambda m: _fix_article(m, "an"),
        out,
        flags=re.IGNORECASE,
    )
    out = re.sub(
        r"\b(an)\s+(young|female|childlike|two)\b",
        lambda m: _fix_article(m, "a"),
        out,
        flags=re.IGNORECASE,
    )
    return _natural_attribution(out)


# File names and titles too generic to blank out wherever they appear.
_BLIND_GENERIC_NAMES = frozenset(
    "mix master mastered demo track song audio music recording take version edit live remix sample "
    "test final untitled new intro outro verse chorus bridge instrumental acapella bounce export "
    "output input file unknown vocal vocals drums drum bass guitar piano keys synth other beat beats "
    "groove rhythm melody harmony tempo stem stems loop".split()
    + ["a cappella", "rough mix", "final mix", "unknown artist", "unknown title", "untitled track"]
)


def _stream_safe_cut(raw, hold=None):
    """Index up to which streamed text can be printed without splitting a
    phrase the vocal-category sanitizer may rewrite ("a post-puberty male").
    Holds back the tail and cuts two words early so an article stays with the
    word it precedes. dev58: a longer hold-back with WRITER_CASUAL_VOICE, so a
    source-citing sentence opener is complete before any of it is shown."""
    if hold is None:
        hold = 72 if globals().get("WRITER_CASUAL_VOICE") else 48
    if len(raw) <= hold:
        return 0
    cut = len(raw) - hold
    for _ in range(2):
        sp = max(raw.rfind(" ", 0, cut), raw.rfind("\n", 0, cut))
        if sp <= 0:
            return 0
        cut = sp
    # Never end a printed chunk on an article: the sanitizer may need to
    # change "an" <-> "a" once it sees the following word.
    while True:
        sp = max(raw.rfind(" ", 0, cut), raw.rfind("\n", 0, cut))
        if sp <= 0:
            return 0
        if raw[sp + 1:cut].strip().lower() in ("a", "an"):
            cut = sp
            continue
        if globals().get("WRITER_CASUAL_VOICE"):
            # dev58: keep the first 64 characters of a sentence back until all
            # of them can be seen, so a source-citing opener is judged whole.
            s0 = 0
            for m in re.finditer(r"(?:[.!?][\)\"\u201d]?\s+|\n)", raw[:cut]):
                s0 = m.end()
            if 0 < cut - s0 < 64 or (s0 == 0 and cut < 64):
                cut = s0
            if cut <= 0:
                return 0
        return cut


def ollama_chat(messages: list, num_ctx=None, keep_alive=_KEEP_ALIVE_UNSET, max_output_tokens=16384,
                 stream_output=False, stream_prefix="", stream_color=None, require_images=False,
                 options=None, enforce_message_limit=True):
    """Chat with the Ollama writer model.

    stream_output: when True, requests the reply from Ollama as it's
    generated (rather than waiting for the complete response) and writes
    each piece to stdout as it arrives -- the status is cleared the moment
    the first piece is ready, so the "Writing..."/"Thinking..." status line
    turns into the live reply at exactly the right moment. stream_prefix and
    stream_color (e.g. "\\nMusiclyse: ", Ansi.MAGENTA) are written once,
    right before that first piece. The retry/fallback cascade below is
    unaffected either way: the status-code check that decides whether to
    retry happens before any response body is read, streamed or not, so a
    failing attempt is retried exactly as before and nothing is ever printed
    for an attempt that gets abandoned. The return value (full text + usage)
    is identical to the non-streaming case either way -- this only changes
    how the text reaches the screen, not what's returned to the caller.

    keep_alive:
      unset (default) — OLLAMA_INTERACTIVE_KEEP_ALIVE ("30m" by default). This
      used to mean "omit the field, let Ollama's own server default apply" --
      observed to be 5 minutes, shorter than a normal thinking gap between
      messages, which caused the whole model process (not just its cache) to
      be killed and fully reloaded on the very next turn. See
      OLLAMA_INTERACTIVE_KEEP_ALIVE for the full explanation.
      0 / "0" — unload the model immediately after this reply (used in batch
      so MOSS-Music/Demucs/Omnizart do not compete with a resident 30B writer). This
      is unaffected by the change above: batch calls pass this explicitly.
      None — send no keep_alive at all, restoring Ollama's own server default.
      Other values are passed through to Ollama (e.g. "5m", -1).

    Musiclyse session state (writer_history, analysis caches) lives in this
    Python process and is unaffected by Ollama unload.
    """
    if keep_alive is _KEEP_ALIVE_UNSET:
        keep_alive = OLLAMA_INTERACTIVE_KEEP_ALIVE
    if num_ctx is None:
        num_ctx = OLLAMA_NUM_CTX

    # Compact the stored history in place before building the request.
    if require_images:
        # Dedicated cover request: the JPEG bytes are transport data, not
        # prompt text. Do not let the text-history budget silently discard them.
        if (len(messages) != 1 or messages[0].get("role") != "user"
                or not messages[0].get("images")):
            raise ValueError("Image-required requests need one user message with artwork")
    else:
        _compact_writer_history_in_place(messages, enforce_message_limit=enforce_message_limit)

    attempts = [
        (int(num_ctx), True),
        (min(int(num_ctx), 32768), True),
        (16384, True),
        (8192, True),
        # No-images fallback cascade. If a model has no vision support, every
        # attempt above gets rejected (observed as repeated 400s) and the loop
        # used to exhaust all four WITHOUT ever trying a plain text request,
        # ending in an unhandled RuntimeError that crashed the whole process --
        # even though the model is perfectly capable of answering once images
        # are off the table. Repeats the SAME context cascade rather than just
        # one small-context attempt: dropping images also drops a large chunk
        # of payload size (base64 image data), so the full context size may
        # well fit again once they're gone, and there's no reason to force a
        # smaller context on a text-only retry just because that's where the
        # image attempts happened to end up. A vision-capable model succeeds
        # on the very first attempt above and never reaches these at all, so
        # this changes nothing for it.
        (int(num_ctx), False),
        (min(int(num_ctx), 32768), False),
        (16384, False),
        (8192, False),
    ]

    # De-duplicate while preserving order. Same context with different image policy is allowed.
    seen = set()
    unique_attempts = []
    for attempt in attempts:
        if attempt not in seen:
            seen.add(attempt)
            unique_attempts.append(attempt)
    attempts = [a for a in unique_attempts if a[1]] if require_images else unique_attempts

    last_error = ""
    _net_errors_shown = set()       # 0.8 OLLAMA_UNREACHABLE_PLAIN: each network error is shown once

    for idx, (ctx, keep_images) in enumerate(attempts):
        # dev2: set once a streamed reply has printed text; a failure after
        # that is reported instead of retried (a retry printed a second answer).
        _stream_printed = False
        try:
            # Reply budget sized to THIS attempt's window (see
            # _ollama_output_budget): the 16384 / 8192 fallbacks used to ask
            # for a 16384-token reply, leaving no room for the prompt.
            out_tokens = _ollama_output_budget(ctx, max_output_tokens)
            if require_images:
                payload_messages = [dict(messages[0], images=list(messages[0]["images"]))]
            else:
                payload_messages = _prepare_ollama_messages(
                    messages, ctx, keep_images=keep_images, max_output_tokens=out_tokens)

            payload = {
                "model": OLLAMA_MODEL,
                "messages": payload_messages,
                "stream": stream_output,
                "options": {
                    "num_ctx": ctx,
                    "num_predict": out_tokens,
                },
            }
            if keep_alive is not None:
                payload["keep_alive"] = keep_alive
            if options:
                payload["options"].update(options)       # 0.7-pre5: scan steps only (_scan_llm_options)
            if _TEST_SEED is not None:
                payload["options"]["seed"] = _TEST_SEED   # dev54 test-only; off by default

            resp = requests.post(
                OLLAMA_URL,
                json=payload,
                timeout=900,
                stream=stream_output,
            )

            has_later_attempt = idx < len(attempts) - 1
            # A later attempt without images exists and this one still has
            # them -- worth trying regardless of the exact status code, since
            # different Ollama/backend versions have been observed to signal
            # "this model has no vision support" differently (a clean 400 is
            # the common case this was originally written for, but not
            # guaranteed).
            drop_images_might_help = keep_images and any(
                not a[1] for a in attempts[idx + 1:]
            )

            if resp.status_code >= 400 and has_later_attempt and (
                resp.status_code == 400 or drop_images_might_help
            ):
                last_error = _ollama_response_error(resp)
                print(
                    f"  (Ollama returned {resp.status_code}; retrying with "
                    f"smaller context/images. Error: {last_error[:300]})"
                )
                continue

            resp.raise_for_status()

            if not stream_output:
                data = resp.json()
                content = str(data.get("message", {}).get("content") or "").strip()
                # Ollama reports per-request token counts when stream=False.
                prompt_n = int(data.get("prompt_eval_count") or 0)
                comp_n = int(data.get("eval_count") or 0)
            else:
                # Live path: this attempt already passed the status-code check
                # above, so it's the one being used -- nothing printed here
                # was ever at risk of belonging to an abandoned/retried
                # attempt. Prints exactly the same characters Ollama sends,
                # unstripped; only the returned `content` is stripped, same
                # as the non-streaming branch.
                content_parts = []
                _emitted = ""
                _emitted_raw = 0
                prompt_n = 0
                comp_n = 0
                first_piece = True
                # dev2: Ollama reports a failure after the reply has started
                # (e.g. the model process stopping) as a {"error": ...} line;
                # it used to be skipped, leaving a cut-off or empty reply that
                # was kept as if complete.
                _stream_error = None
                _done_reason = ""
                try:
                    for line in resp.iter_lines():
                        if not line:
                            continue
                        try:
                            chunk = json.loads(line)
                        except (ValueError, TypeError):
                            continue
                        if isinstance(chunk, dict) and chunk.get("error"):
                            _stream_error = str(chunk.get("error"))
                            break
                        piece = str((chunk.get("message") or {}).get("content") or "")
                        if piece:
                            if first_piece:
                                status_done(clear=True)
                                if stream_color and USE_COLOR:
                                    sys.stdout.write(stream_color)
                                if stream_prefix:
                                    sys.stdout.write(stream_prefix)
                                first_piece = False
                                _stream_printed = True
                            content_parts.append(piece)
                            _raw = "".join(content_parts)
                            _cut = _stream_safe_cut(_raw)
                            if _cut > _emitted_raw:
                                _clean = _humanize_vocal_category_language(_raw[:_cut])
                                if _clean.startswith(_emitted):
                                    sys.stdout.write(_clean[len(_emitted):])
                                    _emitted = _clean
                                else:
                                    _seg = _humanize_vocal_category_language(_raw[_emitted_raw:_cut])
                                    sys.stdout.write(_seg)
                                    _emitted += _seg
                                _emitted_raw = _cut
                            sys.stdout.flush()
                        if chunk.get("done"):
                            prompt_n = int(chunk.get("prompt_eval_count") or 0)
                            comp_n = int(chunk.get("eval_count") or 0)
                            _done_reason = str(chunk.get("done_reason") or "")
                finally:
                    if not first_piece:
                        _raw_all = "".join(content_parts)
                        _clean = _humanize_vocal_category_language(_raw_all)
                        if _clean.startswith(_emitted):
                            sys.stdout.write(_clean[len(_emitted):])
                        else:
                            sys.stdout.write(_humanize_vocal_category_language(_raw_all[_emitted_raw:]))
                        if stream_color and USE_COLOR:
                            sys.stdout.write(Ansi.RESET)
                        sys.stdout.write("\n\n")
                        sys.stdout.flush()
                        if _CHAT_DISPLAY_ARCHIVE is not None:
                            display_reply = (stream_prefix or "") + _humanize_vocal_category_language(_raw_all)
                            if stream_color and USE_COLOR:
                                display_reply = stream_color + display_reply + Ansi.RESET
                            _CHAT_DISPLAY_ARCHIVE.append(display_reply + "\n\n")
                if _stream_error is not None:
                    last_error = _stream_error
                    if not _stream_printed and idx < len(attempts) - 1:
                        print(f"  (Ollama stopped before replying; retrying with smaller context/images. "
                              f"Error: {last_error[:300]})")
                        continue
                    raise _OllamaReplyInterrupted(
                        "Ollama stopped partway through the reply"
                        + (" (the text above is incomplete)" if _stream_printed else "")
                        + f": {last_error}")
                if _done_reason == "length" and _stream_printed:
                    print("  (the reply reached its length limit and was cut off)\n")
                content = "".join(content_parts).strip()

            usage = {
                "prompt": prompt_n,
                "completion": comp_n,
                "ctx": ctx,
            }
            SESSION_TOKEN_USAGE["prompt"] = int(SESSION_TOKEN_USAGE.get("prompt") or 0) + prompt_n
            SESSION_TOKEN_USAGE["completion"] = int(SESSION_TOKEN_USAGE.get("completion") or 0) + comp_n
            SESSION_TOKEN_USAGE["total"] = (
                int(SESSION_TOKEN_USAGE.get("prompt") or 0)
                + int(SESSION_TOKEN_USAGE.get("completion") or 0)
            )
            SESSION_TOKEN_USAGE["last_prompt"] = prompt_n
            SESSION_TOKEN_USAGE["last_completion"] = comp_n
            SESSION_TOKEN_USAGE["last_ctx"] = ctx
            return content, usage

        except requests.exceptions.HTTPError as e:
            last_error = (
                _ollama_response_error(e.response)
                if e.response is not None
                else str(e)
            )
            print(f"  (Ollama request failed: {last_error[:300]})")

            # Retry on 400s, missing responses, or (regardless of status code)
            # when this attempt still had images and a later no-images attempt
            # remains — covers backends that signal "no vision support" with
            # something other than a clean 400.
            drop_images_might_help = keep_images and any(
                not a[1] for a in attempts[idx + 1:]
            )
            if idx < len(attempts) - 1 and (
                e.response is None
                or e.response.status_code == 400
                or drop_images_might_help
            ):
                continue

            raise RuntimeError(f"Ollama request failed: {last_error}") from e

        except requests.exceptions.RequestException as e:
            last_error = str(e)
            if not (OLLAMA_UNREACHABLE_PLAIN and last_error in _net_errors_shown):
                print(f"  (Ollama network error: {last_error[:300]})")
            _net_errors_shown.add(last_error)

            # dev2: the connection dropped after part of a streamed reply was
            # printed. Report it rather than retrying, which printed a second,
            # different answer under the first.
            if _stream_printed:
                raise _OllamaReplyInterrupted(
                    f"Ollama connection dropped partway through the reply "
                    f"(the text above is incomplete): {last_error}") from e

            # Retry transient connection/timeout errors if possible.
            if idx < len(attempts) - 1 and (
                "Connection" in str(e) or "Timeout" in str(e)
            ):
                continue

            raise RuntimeError(f"Ollama request failed: {last_error}") from e

    raise RuntimeError(f"Ollama request failed after retries: {last_error}")


def _ollama_split_name(name):
    """("repo", "tag") for an Ollama model name, lower-cased, with the default
    registry prefix dropped; a missing tag is "" (Ollama itself reads that as
    "latest")."""
    n = str(name or "").strip().lower()
    for pre in ("registry.ollama.ai/library/", "registry.ollama.ai/"):
        if n.startswith(pre):
            n = n[len(pre):]
            break
    head, _, last = n.rpartition("/")
    if ":" in last:
        repo_last, tag = last.rsplit(":", 1)
    else:
        repo_last, tag = last, ""
    return ((head + "/" if head else "") + repo_last, tag)


def _ollama_name_matches_configured(name, strict_latest=False):
    """Does an installed/loaded Ollama model name refer to OLLAMA_MODEL?

    A tagged OLLAMA_MODEL ("model:30b-mlx") must match exactly: any
    other variant of the same model used to count as "loaded", so the unload went
    to a model that was not resident (briefly loading it) while the variant
    actually holding the memory stayed put. An untagged OLLAMA_MODEL
    ("gemma4") keeps the old any-tag rule, because which tag it means is
    ambiguous. strict_latest=True reads an untagged name the way Ollama does
    when asked to run it -- as ":latest" -- which is what "is it installed?"
    needs.
    """
    cfg_repo, cfg_tag = _ollama_split_name(OLLAMA_MODEL)
    if not cfg_repo:
        return False
    repo, tag = _ollama_split_name(name)
    if repo != cfg_repo:
        return False
    if not cfg_tag and not strict_latest:
        return True
    return (tag or "latest") == (cfg_tag or "latest")


def _warn_if_ollama_model_missing():
    """At startup, say plainly when OLLAMA_MODEL is not installed in Ollama
    (with the exact `ollama pull` to run), instead of leaving it to the first
    chat reply to fail. Silent when Ollama cannot be reached or answers
    oddly -- the chat path already explains those."""
    try:
        resp = requests.get(OLLAMA_BASE_URL + "/api/tags", timeout=3)
        if resp.status_code != 200:
            return False
        models = (resp.json() or {}).get("models")
        if not isinstance(models, list):
            return False
        names = [str(m.get("name") or m.get("model") or "") for m in models if isinstance(m, dict)]
        names = [n for n in names if n]
        if any(_ollama_name_matches_configured(n, strict_latest=True) for n in names):
            return False
        shown = ", ".join(names[:8]) + (" ..." if len(names) > 8 else "") if names else "none"
        print(_colorize(
            f"\n  WARNING: OLLAMA_MODEL is {OLLAMA_MODEL!r}, but Ollama does not have that exact "
            f"model installed.\n"
            f"  Run:  ollama pull {OLLAMA_MODEL}\n"
            f"  (or set OLLAMA_MODEL in musiclyse-{__version__}-settings.py to one you have; installed: {shown})\n",
            Ansi.YELLOW,
        ))
        return True
    except Exception:
        return False


def _ollama_model_is_loaded():
    """Return True/False if known. Return None if Ollama's /api/ps is unavailable."""
    try:
        resp = requests.get(OLLAMA_BASE_URL + "/api/ps", timeout=5)

        # Older Ollama builds may not have /api/ps.
        if resp.status_code == 404:
            return None

        if resp.status_code != 200:
            return False

        models = (resp.json() or {}).get("models") or []

        for m in models:
            name = str(m.get("name") or m.get("model") or "")
            if not name:
                continue
            if _ollama_name_matches_configured(name):
                return True

        return False

    except Exception:
        return None


def ollama_unload_model():
    """Best-effort unload of the writer model from Ollama.

    The old implementation sent an empty /api/chat request, which can be rejected
    with 400 and therefore silently failed to unload the output LLM. This version checks
    /api/ps first when possible and uses a valid one-token chat request with
    keep_alive=0.
    """
    try:
        loaded = _ollama_model_is_loaded()

        # If we know it is not loaded, do nothing.
        if loaded is False:
            return

        # If unknown (old Ollama or network issue), still attempt once because the
        # previous behavior was to always ask. This may briefly load then unload on
        # very old servers, but modern Ollama should report via /api/ps.
        payload = {
            "model": OLLAMA_MODEL,
            "messages": [{"role": "user", "content": "Unload."}],
            "stream": False,
            "keep_alive": 0,
            "options": {"num_ctx": 512, "num_predict": 1},
        }

        requests.post(OLLAMA_URL, json=payload, timeout=30)

    except Exception:
        # Unloading is best-effort; do not crash the analysis pipeline because of it.
        pass


def _scan_llm_options():
    """0.7-pre5 SCAN_LLM_REPEATABLE: sampling for the scan's own chat-model
    steps (cover art, singer identity). None leaves the model's defaults."""
    if not SCAN_LLM_REPEATABLE:
        return None
    return {"temperature": 0, "seed": int(SCAN_LLM_SEED)}


def _redact_message_for_debug(msg):
    out = dict(msg or {})
    imgs = out.get("images")
    if isinstance(imgs, list):
        out["images"] = [f"<base64 image {i + 1} len={len(str(img))}>" for i, img in enumerate(imgs)]
    return out


def track_label(raw_path: str) -> str:
    if raw_path.startswith("http://") or raw_path.startswith("https://"):
        return raw_path
    return os.path.basename(raw_path)

def _rms_envelope(x, frame=1024, hop=256):
    """Centred frame-RMS envelope, one value per `hop` samples, computed from
    per-hop block energies. Memory stays O(len(x)) -- unlike an STFT or
    librosa.feature.rms, which materialise frame x n_frames arrays (hundreds of
    MB for a six-minute stem at 44.1 kHz)."""
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        return np.zeros(0)
    n_blocks = int(np.ceil(x.size / float(hop)))
    xb = np.pad(x, (0, n_blocks * hop - x.size)).reshape(n_blocks, hop)
    e = np.einsum("ij,ij->i", xb, xb, dtype=np.float64)
    k = max(1, int(round(frame / float(hop))))
    c = np.concatenate([[0.0], np.cumsum(e)])
    idx = np.arange(n_blocks)
    a = np.clip(idx - k // 2, 0, n_blocks)
    b = np.clip(idx - k // 2 + k, 0, n_blocks)
    return np.sqrt((c[b] - c[a]) / (np.maximum(b - a, 1) * float(hop)))


def _band_filter(y, sr, lo, hi):
    """4th-order Butterworth band-pass (or low/high-pass at the edges), float32,
    streaming. Used for level/onset envelopes, where a few ms of filter delay is
    irrelevant."""
    from scipy.signal import butter, sosfilt
    nyq = sr / 2.0
    lo = float(lo or 0.0)
    hi = min(float(hi or nyq), nyq * 0.95)
    if lo <= 0 and hi >= nyq * 0.95:
        return np.asarray(y, dtype=np.float32)
    if lo <= 0:
        sos = butter(4, hi, btype="low", fs=sr, output="sos")
    elif hi >= nyq * 0.95:
        sos = butter(4, lo, btype="high", fs=sr, output="sos")
    else:
        sos = butter(4, [lo, hi], btype="band", fs=sr, output="sos")
    return sosfilt(sos, np.asarray(y, dtype=np.float32)).astype(np.float32)


def _hz_to_note_name(freq):
    try:
        return librosa.hz_to_note(float(freq))
    except Exception:
        return f"{float(freq):.0f}Hz"


def _hz_to_pitch_class(freq):
    try:
        midi = float(librosa.hz_to_midi(float(freq)))
        return int(round(midi)) % 12
    except Exception:
        return None


_QUOTE_FOLD_MAP = str.maketrans({
    "\u2018": "'", "\u2019": "'", "\u201a": "'", "\u201b": "'", "\u2032": "'",
    "\u201c": '"', "\u201d": '"', "\u201e": '"', "\u201f": '"', "\u2033": '"',
    "`": "'",
})


def _fold_quotes_for_match(s):
    """Fold every apostrophe/quote-mark variant to a plain ASCII one, for
    MATCHING ONLY -- never for the string that actually gets used to open a
    file.

    Track titles saved from tag metadata routinely use a typographic curly
    apostrophe (U+2019, e.g. "Title\u2019s Word"), because that's what most
    taggers, iTunes/Apple Music exports and streaming metadata write. A
    person typing the title at a terminal almost always types a plain
    keyboard apostrophe (U+0027) instead, and the two are close to
    indistinguishable in most terminal fonts. Without folding, a typed name
    that is visually identical to the real file fails to match it.
    """
    return (s or "").translate(_QUOTE_FOLD_MAP)


def _has_real_metadata(metadata):
    """True when a metadata dict carries at least one non-empty value."""
    if not isinstance(metadata, dict):
        return bool(metadata)
    return any(str(v or "").strip() for v in metadata.values())


def _scan_user_photo_given():
    """0.8-pre34: True while a scan runs with --photo."""
    return bool(SCAN_IMAGE_OPTIONS and os.environ.get("MUSICLYSE_USER_PHOTO"))


# ---- finding and reading .txt exports ------------------------------------------

def _txt_name_fold(s):
    """For matching names only: NFC, case-folded, plain quotes, single spaces, no .txt."""
    s = unicodedata.normalize("NFC", _fold_quotes_for_match(str(s or ""))).casefold()
    s = s.replace("\\", "/")
    s = re.sub(r"\s+", " ", s).strip()
    if s.endswith(".txt"):
        s = s[:-4].rstrip()
    return s


class _OutputWatch:
    """pre55: pass writes through, noting whether anything was printed."""

    def __init__(self, stream):
        self._stream = stream
        self.wrote = False

    def write(self, data):
        if data:
            self.wrote = True
        return self._stream.write(data)

    def __getattr__(self, name):
        return getattr(self._stream, name)


_OLLAMA_CAPABILITIES = {}
_OLLAMA_MODEL_FORMAT = {}


def _ollama_model_capabilities():
    """What Ollama says OLLAMA_MODEL can do ("completion", "vision", "thinking",
    ...), asked once per model; None when Ollama doesn't say (0.6.2-dev12).
    0.6.2-dev13: the same answer's model format ("gguf", ...) is kept in
    _OLLAMA_MODEL_FORMAT."""
    if OLLAMA_MODEL not in _OLLAMA_CAPABILITIES:
        caps, fmt = None, None
        try:
            resp = requests.post(OLLAMA_BASE_URL + "/api/show",
                                 json={"model": OLLAMA_MODEL, "name": OLLAMA_MODEL}, timeout=10)
            if resp.status_code == 200:
                info = resp.json() or {}
                got = info.get("capabilities")
                if isinstance(got, list):
                    caps = {str(c).lower() for c in got}
                details = info.get("details")
                if isinstance(details, dict) and details.get("format"):
                    fmt = str(details.get("format")).lower()
        except Exception:
            caps = None
        _OLLAMA_CAPABILITIES[OLLAMA_MODEL] = caps
        _OLLAMA_MODEL_FORMAT[OLLAMA_MODEL] = fmt
    return _OLLAMA_CAPABILITIES[OLLAMA_MODEL]


# ---- /analyse and /reanalyse ------------------------------------------------------

def _audio_path_argument(text):
    """The audio file a /analyse or /reanalyse argument names, or (None, reason)."""
    raw = (text or "").strip()
    if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in ("'", '"'):
        raw = raw[1:-1]
    raw = raw.strip()
    if not raw:
        return None, "no file given"
    if raw.startswith(("http://", "https://")):
        return None, "web addresses aren't supported -- download the file and give its path"
    path = os.path.abspath(os.path.expanduser(raw))
    if not os.path.isfile(path):
        # A path pasted from Finder can carry shell escapes ("My\ Song.m4a").
        unesc = os.path.abspath(os.path.expanduser(re.sub(r"\\(.)", r"\1", raw)))
        if os.path.isfile(unesc):
            path = unesc
        else:
            return None, f"file not found: {raw}"
    if os.path.splitext(path)[1].lower() not in ANALYSIS_EXTENSIONS:
        return None, f"not a supported audio or video file ({', '.join(ANALYSIS_EXTENSIONS)})"
    return path, ""


# 0.8-pre34 SCAN_IMAGE_OPTIONS: --cover "<image>" and --photo "<image>" on /analyse, /reanalyse and
# /musicvideoanalyse. The scan's own processes find them in two environment variables while it runs.
_SCAN_IMAGE_OPTION_RE = re.compile(r"(?<!\S)--(cover|photo)(?:=|\s+)(\"[^\"]*\"|'[^']*'|(?:\\.|\S)+)")
_SCAN_IMAGE_ENV = {"cover": "MUSICLYSE_USER_COVER", "photo": "MUSICLYSE_USER_PHOTO"}


def _scan_image_options(text):
    """0.8-pre34 SCAN_IMAGE_OPTIONS: (text without --cover/--photo, {"cover"/"photo": image path}, error).
    Off, or with neither option: the text unchanged, {} and ""."""
    images, errors = {}, []
    if not SCAN_IMAGE_OPTIONS or "--" not in str(text or ""):
        return text, images, ""

    def take(m):
        kind, raw = m.group(1), m.group(2)
        if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in ("'", '"'):
            raw = raw[1:-1]
        raw = raw.strip()
        path = os.path.abspath(os.path.expanduser(raw))
        if not os.path.isfile(path):
            unesc = os.path.abspath(os.path.expanduser(re.sub(r"\\(.)", r"\1", raw)))
            if os.path.isfile(unesc):
                path = unesc
        if kind in images:
            errors.append(f"--{kind} is given twice")
        elif not raw or not os.path.isfile(path):
            errors.append(f"--{kind}: image not found: {raw or '(none given)'}")
        else:
            try:
                with open(path, "rb") as f:
                    data = f.read()
                usable = bool(data) and bool(prepare_cover_image_for_ollama(data, _guess_image_mime(data)))
            except Exception:
                usable = False
            if usable:
                images[kind] = path
            else:
                errors.append(f"--{kind}: not an image Musiclyse can read: {raw}")
        return ""

    rest, found = _SCAN_IMAGE_OPTION_RE.subn(take, str(text))
    if not found:
        return text, {}, ""
    return rest.strip(), images, (errors[0] if errors else "")


def _scan_images_use(images):
    """0.8-pre34 SCAN_IMAGE_OPTIONS: hand the images to the scan (their environment variables), or
    clear them with {}. Says which images this scan uses."""
    for kind, key in _SCAN_IMAGE_ENV.items():
        if images.get(kind):
            os.environ[key] = images[kind]
        else:
            os.environ.pop(key, None)
    if images.get("cover"):
        print(f"  Cover for this scan: {os.path.basename(images['cover'])} "
              f"(used in place of any artwork in the file; the file itself isn't changed)")
    if images.get("photo"):
        print(f"  Photo of the artist: {os.path.basename(images['photo'])} "
              f"(used only to work out who sings; not saved)")


def main():
    global _CHAT_DISPLAY_ARCHIVE
    _undo_header = []        # 0.7.1-pre25 UNDO_REDRAWS_SCREEN: the start-up text, shown again after /undo
    if UNDO_COMMAND and UNDO_REDRAWS_SCREEN:
        _CHAT_DISPLAY_ARCHIVE = _undo_header
    print_logo()

    _ollama_autostart()      # 0.6.2-dev15: only when no Ollama is running (OLLAMA_LAUNCH_MODE)
    check_ollama_running()

    if ENABLE_ESSENTIA_REPORT and not ESSENTIA_AVAILABLE:
        print(
            "  (Essentia not available — skipping Essentia measurements. "
            f"{ESSENTIA_IMPORT_ERROR})"
        )

    audio_temp_files = {}
    dsp_temp_files = {}
    stem_temp_files = {}
    demucs_out_dirs = []

    # 0.6.2-dev4: loaded exports, in load order. Each entry:
    #   {"name", "text", "tracks", "wiki"} -- the export's text, its tracks
    #   (artist/title/album/year, for /lookup) and its background articles.
    loaded_exports = collections.OrderedDict()
    musicvideo_links = {}        # pre56: video report path -> the song export /musicvideoload paired it with
    lookup_blocks = []           # /lookup results kept for this session (resent after /clear)
    lookup_skip = set()          # folded titles of every article already in the conversation
    focus_paths = []             # the exports named by the latest /load
    last_writer_message = None

    active_persona_text = None   # None = the default assistant
    active_persona_label = "default"
    pending_image_persona = None
    active_game = None           # 0.6.2-dev8: the CHAT_GAMES entry being played, or None

    def _system_prompt():
        """The system prompt for the current game, persona or default assistant."""
        if active_game is not None:
            return build_game_system_prompt(active_game)
        return build_chat_system_prompt(active_persona_text)

    def _mode_label():
        return f"Game: {active_game['name']}" if active_game is not None else f"Persona: {active_persona_label}"
    writer_history = [{"role": "system", "content": build_chat_system_prompt(active_persona_text)}]
    warn_if_system_prompt_truncated(writer_history[0]["content"], num_ctx=_chat_num_ctx())
    _warn_if_ollama_model_missing()

    def _protect(msg):
        _PROTECTED_EVIDENCE_MESSAGE_IDS.add(id(msg))
        return msg

    def _export_messages(path, entry):
        """The conversation messages that carry one loaded export."""
        limit = max(4000, min(MAX_MESSAGE_CHARS_FOR_OLLAMA,
                              int(int(_chat_num_ctx() * HISTORY_CHAR_BUDGET_FACTOR) // 3)) - 400)
        # 0.7.1-pre27: with CHAT_EXPORTS_MAX_SHARE, the part of it that fits (_chat_fit_exports)
        pieces = _split_export_for_chat(
            _chat_sent_export_text(path, entry, list(loaded_exports.items()), _chat_num_ctx()), limit)
        msgs = []
        for i, piece in enumerate(pieces, 1):
            part = f" (part {i} of {len(pieces)})" if len(pieces) > 1 else ""
            msgs.append(_protect({
                "role": "user",
                "content": f"[Musiclyse export loaded: {entry['label']}{part}]\n\n{piece}",
            }))
        return msgs

    def _background_message(block):
        content = block if block.startswith("[Musiclyse visual report: ") else "[Background looked up for this conversation]" + block
        message = {"role": "user", "content": content}
        for entry in chat_save.videos.values():
            if entry["block"] == block and entry["auto_indices"]:
                video = _load_video_extension()
                references = [video.frame_reference(entry["report"], i) for i in entry["auto_indices"]]
                message["images"] = [image for image, label in references]
                message["content"] += "\nAttached images, in order: " + "; ".join(label for image, label in references)
        return _protect(message)

    def _remove_messages_of(path):
        tag = f"[Musiclyse export loaded: {loaded_exports[path]['label']}"
        keep = [m for m in writer_history
                if not (m.get("role") == "user"
                        and str(m.get("content") or "").startswith((tag + "]", tag + " (part ")))]
        writer_history[:] = keep

    def _refit_export_messages():
        """0.7.1-pre27 CHAT_EXPORTS_MAX_SHARE: brings each loaded export's messages in line with what of
        it fits now (loading another export can change that); returns the paths whose messages
        changed. Changed messages are replaced, not edited, and stay where they were."""
        changed = []
        num_ctx = _chat_num_ctx()
        exports = list(loaded_exports.items())
        for path, entry in exports:
            tag = f"[Musiclyse export loaded: {entry['label']}"
            where = [i for i, m in enumerate(writer_history)
                     if m.get("role") == "user" and str(m.get("content") or "").startswith((tag + "]", tag + " (part "))]
            want = _chat_export_contents(entry["label"], _chat_sent_export_text(path, entry, exports, num_ctx), num_ctx)
            if [writer_history[i].get("content") for i in where] == want:
                continue
            at = where[0] if where else len(writer_history)
            for i in reversed(where):
                _PROTECTED_EVIDENCE_MESSAGE_IDS.discard(id(writer_history[i]))
                del writer_history[i]
            writer_history[at:at] = [_protect({"role": "user", "content": c}) for c in want]
            changed.append(path)
        return changed

    chat_save = _ChatSaveArchive()

    def _fresh_chat_archive():
        nonlocal chat_save
        global _CHAT_DISPLAY_ARCHIVE
        chat_save = _ChatSaveArchive()
        _CHAT_DISPLAY_ARCHIVE = chat_save.display if active_game is None else None

    def _clear_for_persona():
        writer_history.clear()
        _PROTECTED_EVIDENCE_MESSAGE_IDS.clear()
        loaded_exports.clear()
        lookup_blocks.clear()
        lookup_skip.clear()
        focus_paths.clear()
        _reset_tokens()
        _fresh_chat_archive()

    def _reset_tokens():
        for k in SESSION_TOKEN_USAGE:
            SESSION_TOKEN_USAGE[k] = 0

    def _load_exports(paths):
        """Load (or refresh) exports; returns the paths actually in the session."""
        done = []
        before = dict(loaded_exports)      # 0.7.1-pre27: which exports this /load added or changed
        for path in paths:
            try:
                with open(path, encoding="utf-8", errors="replace") as fh:
                    text = fh.read()
            except Exception as e:
                print(f"  Could not read {path}: {e}\n")
                continue
            if not text.strip():
                print(f"  {os.path.basename(path)} is empty -- not loaded.\n")
                continue
            tracks, wiki = _parse_txt_export(text)
            try:
                rel = os.path.relpath(path, EXPORT_SEARCH_DIR)
            except ValueError:
                rel = path
            label = rel if not rel.startswith("..") else path   # folder shown: two levels can share a name
            entry = {"name": os.path.basename(path), "label": label, "text": text, "tracks": tracks, "wiki": wiki}
            old = loaded_exports.get(path)
            if old is not None and old["text"] == text:
                print(f"  ({entry['label']} is already loaded)")
                done.append(path)
                continue
            if old is not None:
                _remove_messages_of(path)
                del loaded_exports[path]
            loaded_exports[path] = entry
            writer_history.extend(_export_messages(path, entry))
            lookup_skip.update(_txt_name_fold(t) for t in wiki)
            what = ", ".join(
                (f"{t['artist']} – {t['title']}" if t.get("artist") else t["title"]) for t in tracks
            ) or "no track heading found"
            print(f"  {'Reloaded' if old is not None else 'Loaded'} {entry['label']} "
                  f"({len(tracks)} track{'s' if len(tracks) != 1 else ''}: {what}; "
                  f"{len(wiki)} background article{'s' if len(wiki) != 1 else ''} inside)")
            done.append(path)
        # The chat model's context holds about this many characters of conversation;
        # past it, the oldest messages are left out of each request.
        chat_ctx = _chat_num_ctx()
        fresh = [p for p in dict.fromkeys(done) if loaded_exports.get(p) is not before.get(p)]
        changed = _refit_export_messages() if fresh and _chat_export_room(chat_ctx) is not None else []
        if CHAT_LOAD_SHOWS_FIT and fresh:
            for line in _chat_fit_lines(list(loaded_exports.items()), fresh, changed, chat_ctx):
                print(line)
        budget = int((chat_ctx - 16384) * HISTORY_CHAR_BUDGET_FACTOR)
        # 0.7.1-pre27: what the exports add to requests (just their length with the new settings off)
        total = _chat_exports_size(list(loaded_exports.items()), chat_ctx)
        if done and total > 0.8 * budget:
            print(_colorize(
                f"  (The loaded exports add up to about {int(total / HISTORY_CHAR_BUDGET_FACTOR):,} tokens -- "
                f"{'more than' if total > budget else 'close to'} what the chat model's context holds "
                f"(about {chat_ctx - 16384:,} tokens for the conversation). The oldest material may be left out "
                f"of replies: {CLEAR_ALL_FLAG} and load fewer exports, or use a lower export level.)",
                Ansi.YELLOW))
        return done

    def _resolve_names(text, many):
        """Export names off the front of text -> (paths, question) or (None, None) after
        printing why. `many` takes names until one doesn't match."""
        paths, rest = [], (text or "").strip()
        while True:
            # A quoted name must exist; after the first name, an unquoted name
            # must be a whole file name (otherwise it starts the question).
            quoted = rest[:1] in ("'", '"', "\u201c", "\u2018")
            nm, hits, after = _next_txt_argument(rest, partial=not paths)
            if nm is None:
                break
            if len(hits) > 1:
                rels = [os.path.relpath(h, EXPORT_SEARCH_DIR) for h in hits]
                pick = _pick_export_menu(f"'{nm}' matches {len(hits)} exports -- which one?", rels,
                                       highlight=active_game.get("menu_color") if active_game is not None else None)
                if pick is None:
                    print("  (cancelled -- nothing loaded)\n")
                    return None, None
                print(f"  \u2192 {rels[pick]}")
                hits = [hits[pick]]
            if not hits:
                if not paths or quoted:
                    tried = nm if quoted else (text or "").strip()
                    tried = tried if len(tried) <= 80 else tried[:77] + "..."
                    print(f"  No export .txt found for '{tried}' under {EXPORT_SEARCH_DIR}/ "
                          f"(or as a path)." + (" Nothing was loaded." if paths else "") + "\n")
                    _print_export_reminder()
                    return None, None
                break
            paths.append(hits[0])
            rest = after
            if not many:
                break
            # A joining word is only a separator when another export follows it.
            sep = re.match(r"(?i)\s*(?:,|&|\+|and|vs\.?|versus)\s+", rest)
            if sep:
                nxt = rest[sep.end():]
                nm2, hits2, _ = _next_txt_argument(nxt, partial=False)
                if hits2 or nxt[:1] in ("'", '"', "\u201c", "\u2018"):
                    rest = nxt
        return paths, rest.strip()

    def _lookup(question):
        """Chat-only Wikipedia retrieval, deduplicated by excerpt content.
        New sections from an already-used article remain eligible."""
        meta, keys, seen = {}, [], set()
        for p in focus_paths:
            for i, tr in enumerate((loaded_exports.get(p) or {}).get("tracks") or []):
                ident = (_txt_name_fold(tr.get("artist")), _txt_name_fold(tr.get("title")))
                if ident in seen:          # the same song in two loaded files counts once
                    continue
                seen.add(ident)
                k = f"{p}#{i}"
                meta[k] = {f: tr.get(f) for f in ("artist", "title", "album")}
                keys.append(k)
        current = keys[-1] if keys else None
        chat_only = [m for m in writer_history
                     if not str(m.get("content") or "").startswith(("[Musiclyse export loaded:",
                                                                   "[Background looked up"))]
        status("Looking up background in the local Wikipedia database...")
        choose = {}
        if LOOKUP_CHOOSE_WHEN_UNSURE and sys.stdin.isatty() and sys.stdout.isatty():
            def _choose_article(name, titles):
                """0.7.1-pre25: a menu of the articles too close to call for `name`."""
                status_done(clear=True)
                try:
                    return _pick_export_menu(f'Which article for "{name}"? (Esc: none of them)', list(titles),
                                             highlight=active_game.get("menu_color") if active_game is not None else None)
                finally:
                    status("Looking up background in the local Wikipedia database...")
            choose = {"choose": _choose_article}
        try:
            diagnostics = []
            block = _chat_wiki_lookup(
                question or "", list(meta.values()), chat_only,
                [entry["text"] for entry in loaded_exports.values()] + list(lookup_blocks),
                diagnostics, hidden=_wiki_hidden_titles() or None, **choose,
            )
        except Exception as e:
            status_done(clear=True)
            print(f"  (lookup failed: {e})")
            return ""
        status_done(clear=True)
        if CHAT_WIKI_LOOKUP_DEBUG:
            for detail in diagnostics:
                print("  (lookup debug: " + detail + ")")
        for detail in diagnostics:
            if detail.startswith(("Ambiguous subject ", "Database unavailable:")):
                print("  (lookup: " + detail + ")")
        found = _lookup_titles(block)
        # Excerpt-level deduplication already ran against exports and lookup blocks.
        block = _lookup_filter(block, set(),
                               persona=active_persona_text is not None or active_game is not None)
        new = _lookup_titles(block)
        if not found:
            if any(d.startswith("Already supplied:") for d in diagnostics):
                print("  (lookup: relevant background is already in this conversation)")
            else:
                print("  (lookup: no confident new match in the local Wikipedia database"
                      + (f' -- {LOOKUP_SEARCH_FLAG} "a phrase" lists the article titles that contain it'
                         if LOOKUP_CHOOSE_WHEN_UNSURE and LOOKUP_SEARCH_COMMAND else "") + ")")
        elif not new:
            print(f"  (lookup: {len(found)} article(s) found, all already in this conversation)")
        else:
            already = len(found) - len(new)
            print(f"  (lookup: {', '.join(new)}"
                  + (f"; {already} more already in this conversation" if already else "") + ")")
            lookup_skip.update(_txt_name_fold(t) for t in new)
        return block

    def _lookup_search(names, question):
        """pre62 /lookup-search: the articles chosen by title -- a menu where a phrase is in
        several titles -- as one background block for this conversation ("" when none is
        added). The question only chooses which paragraphs are sent, as with /lookup."""
        conn = _get_wiki_db()
        if conn is None:
            print(f"  ({LOOKUP_SEARCH_FLAG}: the local Wikipedia database isn't available -- "
                  f"{_WIKI_DB_UNAVAILABLE_REASON or 'disabled'})")
            return ""
        schema = _WIKI_DB_SCHEMA
        limit = max(1, int(WIKI_MAX_ARTICLES))
        hidden = _wiki_hidden_titles()
        marked = set(_wiki_marked_titles()) if WIKI_DELETE_COMMAND else set()
        highlight = active_game.get("menu_color") if active_game is not None else None
        chosen = []                  # (title, rowid)
        for n, name in enumerate(names):
            if len(chosen) >= limit:
                print(f"  ({LOOKUP_SEARCH_FLAG}: at most {limit} articles at once; not looked up: "
                      + ", ".join(f'"{left}"' for left in names[n:]) + ")")
                break
            found = _LOOKUP_SEARCH_FOUND.get(name) if LOOKUP_SEARCH_DRAFT_MENU else None
            if found is not None:       # pre66: already found (or chosen) on the input line
                titles, rowids, more, starts_only = found
            else:
                status(f'Finding "{name}" in the local Wikipedia database...')
                try:
                    titles, rowids, more, starts_only = _lookup_search_titles(conn, schema, name)
                except Exception as e:
                    status_done(clear=True)
                    print(f'  ({LOOKUP_SEARCH_FLAG}: "{name}" could not be looked up: {e})')
                    continue
                status_done(clear=True)
            where = "starts with" if starts_only else "contains"
            if not titles:
                print(f'  ({LOOKUP_SEARCH_FLAG}: no article title in the local Wikipedia database {where} "{name}")')
                continue
            if len(titles) == 1:
                pick = 0
            else:
                pick, changed = _lookup_search_menu(conn, schema, name, titles, rowids, more,
                                                    starts_only, marked, highlight)
                if changed:
                    hidden = _wiki_hidden_titles()
                if pick is None:
                    print(f'  ({LOOKUP_SEARCH_FLAG}: skipped "{name}")')
                    continue
                print(f"  → {titles[pick]}")
            if titles[pick] in hidden:
                print(f'  ({LOOKUP_SEARCH_FLAG}: "{titles[pick]}" is marked for removal, so it is not added; '
                      f'{WIKI_DELETE_FLAG} undo "{titles[pick]}" keeps it)')
                continue
            if titles[pick] not in [t for t, _ in chosen]:
                chosen.append((titles[pick], rowids.get(titles[pick])))
        if not chosen:
            return ""
        existing = [_chat_wiki_key(t) for t in
                    [entry["text"] for entry in loaded_exports.values()] + list(lookup_blocks) if t]
        cap = 3200 if _wiki_question_focus(question or "") in {"tracklist", "debut"} else 2400
        selected, already = [], []
        for title, rowid in chosen:
            body = _lookup_search_article(conn, schema, title, rowid)
            if not body:
                print(f'  ({LOOKUP_SEARCH_FLAG}: "{title}" could not be read)')
                continue
            novel = _chat_wiki_novel(_chat_wiki_snippet(body, question or "", cap), existing)
            if novel:
                selected.append((title, novel))
            else:
                already.append(title)
        anchor = {}
        for p in focus_paths:        # a loaded track keeps its sound firewall, as with /lookup
            tracks = (loaded_exports.get(p) or {}).get("tracks") or []
            if tracks:
                anchor = tracks[0]
                break
        block = ""
        if selected:
            block = build_wiki_context_block_multi(
                artist=anchor.get("artist"), title=anchor.get("title"), album=anchor.get("album"),
                question=question or "", use_question_entities=False, _lookup_articles=selected)
            block = _lookup_filter(block, set(),
                                   persona=active_persona_text is not None or active_game is not None)
        new = _lookup_titles(block)
        note = f"already in this conversation: {', '.join(already)}" if already else ""
        if new:
            print(f"  ({LOOKUP_SEARCH_FLAG}: {', '.join(new)}" + (f"; {note}" if note else "") + ")")
            lookup_skip.update(_txt_name_fold(t) for t in new)
        elif note:
            print(f"  ({LOOKUP_SEARCH_FLAG}: {note})")
        return block

    def _command_list_text():
        """pre65: the command list shown at start-up and by /help -- one source for both."""
        return (
        f"  {ANALYSE_FLAG} <audio file>          analyse one track and save it to {SAVE_DIR}/ "
        f"(skipped if already saved)\n"
        f"  {REANALYSE_FLAG} <audio file>        analyse it again, replacing its save\n"
        + ((f"  --cover \"image\" / --photo \"image\"  add to {ANALYSE_FLAG}, {REANALYSE_FLAG} or "
            f"{MUSICVIDEO_ANALYSE_FLAG} (one track at a time, not {BATCH_FLAG}): an image to use as the "
            f"cover, or a photo of the artist (for a group, its lead singer) to help work out who sings\n"
            if BATCH_IMAGE_OPTIONS_NOTE else
            f"  --cover \"image\" / --photo \"image\"  add to {ANALYSE_FLAG}, {REANALYSE_FLAG} or "
            f"{MUSICVIDEO_ANALYSE_FLAG}: an image to use as the cover, or a photo of the artist to help "
            f"work out who sings\n") if SCAN_IMAGE_OPTIONS else "")
        + (f"  {BATCH_FLAG} [folder]                analyse every audio file in a folder (skips saved ones); "
           f"with no folder, the {BATCH_DEFAULT_FOLDER!r} folder beside this script, if there is one\n"
           if BATCH_DEFAULT_FOLDER else
           f"  {BATCH_FLAG} <folder>                analyse every audio file in a folder (skips saved ones)\n")
        + ""
        f"  {LOAD_FLAG} \"Artist - Title\" [question]   load an exported track (or a combined export) to chat about\n"
        f"  {LOAD_FLAG} \"Artist 1 - Title 1\" \"Artist 2 - Title 2\" ... [question]   load several at once, "
        f"e.g. to compare them\n"
        + (f"  {EXPORT_FLAG} [files or folder] [options]   export saved songs for chat, as the exporter script does "
           f"(no path: every save in {SAVE_DIR}/; empty quotes choose songs; {EXPORT_FLAG} --help lists the options)\n"
           if EXPORT_COMMAND else "")
        + (f"  {SONGFORMER_FLAG} [setup|selftest|scan|compare|check] ...   SongFormer: analyses use it to check "
           f"section maps; check lists what it would change in your saves (changes none); "
           f"{SONGFORMER_FLAG} setup shows the steps\n" if SONGFORMER_COMMAND and SONGFORMER_SLASH_COMMAND else "") +
        "  /chatsave [name]                 save the conversation to chats/; without a name, update "
        "its current save (or make a timestamped save); an explicit name makes a separate copy\n"
        "  /chatload [filename]             resume a saved conversation; omit the filename or use "
        "empty quotes to choose from a menu; .json is optional\n"
        "  Chat saves include songs, images and the full transcript. Names use timestamps and "
        "replace spaces with _. Use /chatsave and /chatload alone, outside games. "
        "Unsaved conversations stay in RAM; there is no autosave.\n"
        "  /videoanalyse <video> [options]  visual analysis; asks for an optional name, saves resumable reports in video-analysis/\n"
        "  /videoload <name/path> [question] load a video; empty quotes open a menu; combine with /load and a question\n"
        "  /frame <loaded video name> <index>  attach a saved frame to any follow-up question; indices start at 0\n"
        "  /videoembed <report.json>        embed an older report's saved JPEGs without rescanning\n"
        "  /videostrip <report>             write a text-only copy (no frame images) for sharing\n"
        + (f"  {MUSICVIDEO_ANALYSE_FLAG} <video> [options]  analyse a music video's song, then its pictures; "
           f"asks for an optional name\n"
           f"  {MUSICVIDEO_LOAD_FLAG} <name> [question]  load a music video's song export and its video "
           f"together; empty quotes open a menu\n" if MUSICVIDEO_COMMANDS else "") +
        "  Similar song/video names are associated automatically; timing alignment still needs an explicit offset.\n"
        "  /videoframe <report.json> <time> [question]   show the retained frame nearest a video time\n"
        "  Video options: --song \"Artist - Title\" --offset <seconds> (video time minus song time); "
        "--fps 1; --frames-per-request 8. /videoanalyse --help lists all options. "
        "Video commands are unavailable in games.\n"
        f"  {QUIET_FLAG}                          add to a {LOAD_FLAG} to load without a reply "
        f"(otherwise the chat replies straight away)\n"
        f"  {LOOKUP_FLAG}                         add to any prompt (also with {LOAD_FLAG}) "
        f"to look up background in the local Wikipedia database\n"
        + (f"  {LOOKUP_SEARCH_FLAG} \"Title\" [\"Title 2\" ...] [question]   add Wikipedia articles by title "
           f"(a menu when a phrase is in several titles; also with {LOAD_FLAG}, /videoload and "
           f"{MUSICVIDEO_LOAD_FLAG})\n" if LOOKUP_SEARCH_COMMAND else "")
        + (f"  {WIKI_DELETE_FLAG} \"Title\" | list | undo \"Title\" | apply   mark Wikipedia articles and write a "
           f"clean copy of the database without them (d marks one in a {LOOKUP_SEARCH_FLAG} menu)\n"
           if WIKI_DELETE_COMMAND else "") +
        f"  Images: drag an image file into this window (or type its path or web address) in any prompt "
        f"to show it to the chat\n"
        f"  {PERSONA_FLAG} <description>         chat as a custom character; "
        f"'{PERSONA_FLAG} reset' for the default assistant\n"
        + (f"  {MEMORY_FLAG} [topic]                  save a short memory of this conversation (or of one "
           f"topic) for future chats -- shown first to keep, edit or cancel\n"
           f"  {MEMORY_FLAG} list | on | off          read or delete memories; turn bringing them into "
           f"chats on or off\n" if CHAT_MEMORY else "") +
        f"  {CLEAR_FLAG}                          clear the conversation (loaded exports and looked-up "
        f"background are kept and sent again)\n"
        f"  {CLEAR_ALL_FLAG}                       clear everything, including loaded exports and "
        f"looked-up background\n"
        + (f"  {UNDO_FLAG}                           remove your last message and everything it added "
           f"(the reply, songs, articles, images); repeat to go further back\n" if UNDO_COMMAND else "") +
        f"  {THINK_FLAG}                          add to any prompt to let the chat model think before "
        f"answering (slower)\n"
        f"  {GAME_FLAG}                           play a game: chat with a built-in character; "
        f"{GAME_EXIT_FLAG} to leave it\n" +
        (f"  {HELP_FLAG}                           show this list of commands again\n" if HELP_COMMAND else "") +
        f"  Names with spaces can be quoted. Exports are found under {EXPORT_SEARCH_DIR}/ (any folder "
        f"inside it) or by path. Type 'quit' to exit.\n"
        )

    print(_colorize(
        f"\nMusiclyse is ready. Chat uses {OLLAMA_MODEL}; scans use {MOSS_MODEL_ID} and the rest of the analysis "
        f"pipeline, loaded only while a scan runs.\n\n"
        + _command_list_text(),
        Ansi.YELLOW,
    ))
    _print_export_reminder()

    def _reply(image_message=None, think=False):
        """Send the conversation as it stands; append and report the reply.
        Returns the reply, or None when there was none."""
        # Rebuild from application state each turn, including earlier /load batches.
        # Track headings count songs; a combined export can contain several.
        inventory = []
        unknown_exports = []
        for entry in loaded_exports.values():
            if not entry["tracks"]:
                unknown_exports.append(entry["label"])
            for track in entry["tracks"]:
                title = str(track.get("title") or "").strip()
                artist = str(track.get("artist") or "").strip()
                identity = (_txt_name_fold(artist), _txt_name_fold(title))
                if identity not in [item[0] for item in inventory]:
                    inventory.append((identity, artist, title))
        roster = [
            "\n\nCURRENT LOADED-TRACK INVENTORY (application state):",
            f"{len(loaded_exports)} export file(s) are registered as loaded.",
            f"{len(inventory)} distinct track(s) identified from their headings:",
        ]
        for index, (_identity, artist, title) in enumerate(inventory, 1):
            roster.append(f"{index}. " + json.dumps(
                {"artist": artist, "title": title}, ensure_ascii=False))
        if unknown_exports:
            roster.append("Additional exports with unrecognised track headings: "
                          + json.dumps(unknown_exports, ensure_ascii=False))
        roster.append(
            "Use this inventory when reporting what is loaded. Multiple versions of "
            "the same artist/title count once. Export parts are not separate tracks. "
            "An unrecognised heading does not mean its export failed to load. "
            "This inventory identifies loaded material but supplies no musical evidence. "
            "If a CONTEXT LIMIT notice says evidence was omitted, distinguish loaded "
            "tracks from those whose analysis is available in this request.")
        roster_text = "[Loaded-track inventory update]\n" + "\n".join(roster)
        roster_text += (
            "\nThis update supersedes earlier inventory updates. Respond to the user's "
            "request using the loaded exports; this is application bookkeeping.")
        if CHAT_KNOWS_DATE and active_persona_text is None and active_game is None:
            # 0.7.1-pre25: past midnight, a one-line date update before this turn's question --
            # added, never rewritten, so the conversation isn't read again.
            _date_note = _chat_date_update(writer_history)
            if _date_note:
                position = len(writer_history)
                if (writer_history and writer_history[-1] is last_writer_message
                        and writer_history[-1].get("role") == "user"
                        and not str(writer_history[-1].get("content") or "").startswith(
                            ("[Musiclyse export loaded:", "[Background looked up for this conversation]"))):
                    position -= 1
                writer_history.insert(position, {"role": "user", "content": _date_note})
        previous_roster = next(
            (m.get("content") for m in reversed(writer_history)
             if m.get("role") == "user"
             and str(m.get("content") or "").startswith("[Loaded-track inventory update]\n")),
            None,
        )
        if previous_roster != roster_text:
            # Only append new state. Never rewrite the cached system/history prefix.
            # Put bookkeeping before this turn's unsent question, keeping image and
            # question handling attached to the original message object.
            position = len(writer_history)
            if (writer_history and writer_history[-1] is last_writer_message
                    and writer_history[-1].get("role") == "user"
                    and not str(writer_history[-1].get("content") or "").startswith(
                        ("[Musiclyse export loaded:", "[Background looked up for this conversation]"))):
                position -= 1
            writer_history.insert(position, {"role": "user", "content": roster_text})
        if CHAT_MEMORY:
            # 0.7.1-pre30: the memory note as it should be (default chat only)
            _memory_sync(writer_history,
                         active_persona_text is None and active_game is None and not pending_image_persona,
                         loaded_exports, last_writer_message)
        if active_game is None:
            chat_save.sync(writer_history)
        _ctx = _chat_num_ctx()
        _before_count = len(writer_history)
        _before_chars = sum(len(str(m.get("content") or "")) for m in writer_history)
        _trimmed, _why = _chat_trim_history(writer_history, _ctx)
        if _trimmed:
            print(f"  Context trim: reason={_why}; messages {_before_count} -> {len(writer_history)}; "
                  f"text chars before={_before_chars:,}; "
                  f"configured window={_ctx:,}. Backend token counts may differ.")
            try:
                _here = os.path.basename(os.path.abspath(__file__))
            except Exception:
                _here = "the Musiclyse script"
            _more = (f"For more room, raise {_chat_num_ctx_setting()} near the top of {_here} "
                     f"({_ctx:,} tokens now; a bigger window uses more memory), or load "
                     f"lower-level exports.")
            print(_colorize(
                f"  (This conversation has "
                f"outgrown what the chat model holds, "
                f"so its {_trimmed} oldest "
                f"message{'s were' if _trimmed != 1 else ' was'} left out -- loaded exports and looked-up "
                f"background are kept. This reply re-reads the conversation once; {CLEAR_FLAG} starts "
                f"afresh. {_more})", Ansi.YELLOW))
        if active_game is not None:
            reply, usage = _chat_reply(writer_history, image_message=image_message,
                                       speaker=active_game["speaker"], color=active_game["reply_color"],
                                       think=think)
        elif CHAT_PERSONA_SPEAKER and active_persona_text is not None:      # 0.8-pre31: a custom persona
            reply, usage = _chat_reply(writer_history, image_message=image_message,
                                       speaker=(CHAT_PERSONA_SPEAKER if isinstance(CHAT_PERSONA_SPEAKER, str)
                                                else "Musiclyse (custom persona)"), think=think)
        else:
            reply, usage = _chat_reply(writer_history, image_message=image_message, think=think)
        if reply is None:
            return None
        writer_history.append({"role": "assistant", "content": reply})
        if active_game is None:
            chat_save.sync(writer_history)
        status_done()  # safety net: no-op unless the reply was empty
        _print_token_usage(usage or SESSION_TOKEN_USAGE)
        return reply

    def _new_messages_since(ids_before):
        """Messages added to the conversation since ids_before was taken."""
        return [m for m in writer_history if id(m) not in ids_before]

    _start = sys.argv[1] if len(sys.argv) > 1 else None
    _start_reply = False
    if _start:
        if _start.lower().endswith(".txt") and os.path.isfile(os.path.expanduser(_start)):
            _ids = {id(m) for m in writer_history}
            focus_paths = _load_exports([os.path.abspath(os.path.expanduser(_start))]) or focus_paths
            _start_reply = LOAD_REPLIES_BY_DEFAULT and bool(_new_messages_since(_ids))
            print()
        else:
            print(f"  (starting file ignored: {_start} -- analyse audio with {ANALYSE_FLAG}, "
                  f"chat about an export with {LOAD_FLAG})\n")

    _CHAT_DISPLAY_ARCHIVE = chat_save.display

    # 0.7.1-pre25 UNDO_COMMAND: before each line is acted on, a note of what the conversation holds;
    # /undo goes back to the newest one. A line that changed nothing has its note dropped, and a
    # /clear, /chatload, persona or game change drops them all.
    _undo_steps = []
    _undo_draft = None

    def _undo_note(text):
        return {
            "text": text, "history": list(writer_history), "protected": set(_PROTECTED_EVIDENCE_MESSAGE_IDS),
            "exports": list(loaded_exports.items()), "links": dict(musicvideo_links),
            "blocks": list(lookup_blocks), "skip": set(lookup_skip), "focus": list(focus_paths),
            "last": last_writer_message, "save": chat_save, "save_messages": list(chat_save.messages),
            "save_sources": dict(chat_save.sources), "save_protected": set(chat_save.protected),
            "save_videos": dict(chat_save.videos), "save_display": len(chat_save.display),
            "context_start": chat_save.context_start,
            "system": writer_history[0] if writer_history else None,
            "persona": active_persona_text, "game": active_game, "pending": pending_image_persona,
        }

    def _undo_reset(note):
        """The conversation was started again (or changed character) since `note`."""
        return (chat_save is not note["save"] or not writer_history or writer_history[0] is not note["system"]
                or active_persona_text != note["persona"] or active_game is not note["game"]
                or pending_image_persona is not None or note["pending"] is not None
                or chat_save.context_start != note["context_start"])

    def _undo_changed(note):
        """Whether the conversation (not just the screen) changed since `note`."""
        return (len(writer_history) != len(note["history"])
                or any(a is not b for a, b in zip(writer_history, note["history"]))
                or [(k, id(v)) for k, v in loaded_exports.items()] != [(k, id(v)) for k, v in note["exports"]]
                or musicvideo_links != note["links"] or lookup_blocks != note["blocks"]
                or lookup_skip != note["skip"] or list(focus_paths) != note["focus"]
                or len(chat_save.messages) != len(note["save_messages"])
                or set(chat_save.videos) != set(note["save_videos"]))

    try:
        if _start_reply:
            last_writer_message = writer_history[-1]
            _reply()
        while True:
            if active_game is None:
                chat_save.sync(writer_history)
            if UNDO_COMMAND and _undo_steps:
                if _undo_reset(_undo_steps[-1]):
                    _undo_steps.clear()
                elif not _undo_changed(_undo_steps[-1]):
                    _undo_steps.pop()
            _CHAT_DISPLAY_ARCHIVE = None  # input menus and unsent drafts are transient
            try:
                user_text = _chat_input(
                    "You: ", active_game["input_color"] if active_game is not None else Ansi.LIGHT_GREEN,
                    highlight=active_game.get("menu_color") if active_game is not None else None,
                    allow_video=active_game is None and not pending_image_persona,
                    **({"initial": _undo_draft} if _undo_draft else {})).strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            finally:
                _undo_draft = None
            if user_text.lower() in ("quit", "exit"):
                break
            if not user_text:
                continue
            if UNDO_COMMAND and re.match(r"(?i)" + re.escape(UNDO_FLAG) + r"(?=\s|$)", user_text):
                # 0.7.1-pre25: back to the note taken before the last line that changed the conversation.
                _CHAT_DISPLAY_ARCHIVE = None
                if user_text.strip().lower() != UNDO_FLAG.lower():
                    print(f"  Use {UNDO_FLAG} on its own: it removes your last message and everything it added.\n")
                    continue
                if pending_image_persona:
                    print(f"  Finish or cancel the pending image persona before using {UNDO_FLAG}.\n")
                    continue
                if not _undo_steps:
                    print("  (Nothing to undo: nothing has been added since the conversation started, was "
                          "cleared or resumed, or changed character.)\n")
                    continue
                note = _undo_steps.pop()
                writer_history[:] = note["history"]
                _PROTECTED_EVIDENCE_MESSAGE_IDS.clear()
                _PROTECTED_EVIDENCE_MESSAGE_IDS.update(note["protected"])
                loaded_exports.clear()
                loaded_exports.update(note["exports"])
                musicvideo_links.clear()
                musicvideo_links.update(note["links"])
                lookup_blocks[:] = note["blocks"]
                lookup_skip.clear()
                lookup_skip.update(note["skip"])
                focus_paths = list(note["focus"])
                last_writer_message = note["last"]
                chat_save.messages[:] = note["save_messages"]
                chat_save.sources.clear()
                chat_save.sources.update(note["save_sources"])
                chat_save.protected.clear()
                chat_save.protected.update(note["save_protected"])
                chat_save.videos.clear()
                chat_save.videos.update(note["save_videos"])
                del chat_save.display[note["save_display"]:]
                fresh = False
                if UNDO_FRESH_READ:
                    # 0.7.1-pre26: unload the model, so the next reply reads everything from scratch.
                    try:
                        requests.post(OLLAMA_BASE_URL + "/api/generate",
                                      json={"model": OLLAMA_MODEL, "keep_alive": 0}, timeout=15)
                        fresh = True
                    except Exception:
                        fresh = False
                terminal = sys.stdin.isatty() and sys.stdout.isatty()
                if UNDO_REDRAWS_SCREEN and active_game is None and terminal:
                    sys.stdout.write("\x1b[H\x1b[2J\x1b[3J")
                    sys.stdout.flush()
                    for line in _undo_header + chat_save.display:
                        if not USE_COLOR:
                            line = re.sub(r"\x1b\[[0-9;]*m", "", line)
                        _builtin_print(line, end="")
                    print("  (Your last message and everything it added were removed.)\n")
                else:
                    print("  (Your last message and everything it added were removed from the conversation"
                          + (" -- the window above still shows them" if terminal else "") + ".)\n")
                if fresh:
                    print(f"  ({OLLAMA_MODEL} was unloaded (UNDO_FRESH_READ): your next message loads it again "
                          "and has the whole conversation read from scratch.)\n")
                if UNDO_RESTORES_MESSAGE and terminal:
                    _undo_draft = note["text"]
                continue
            if UNDO_COMMAND:
                _undo_steps.append(_undo_note(user_text))
                del _undo_steps[:-100]
            if HELP_COMMAND and re.match(r"(?i)" + re.escape(HELP_FLAG) + r"(?=\s|$)", user_text):
                # pre65: the command list again; not sent to the chat, not part of the transcript
                _CHAT_DISPLAY_ARCHIVE = None
                if user_text.strip().lower() != HELP_FLAG:
                    print(f"  Use {HELP_FLAG} on its own: it lists every command.\n")
                else:
                    print(_colorize("\n" + _command_list_text(), Ansi.YELLOW))
                    _print_export_reminder()
                continue

            if CHAT_MEMORY and _command_remainder(user_text, MEMORY_FLAG) is not None:
                # 0.7.1-pre30: /memory; not sent to the chat, not part of the transcript
                _CHAT_DISPLAY_ARCHIVE = None
                _memory_command(_command_remainder(user_text, MEMORY_FLAG), writer_history,
                                active_persona_text is None and active_game is None and not pending_image_persona,
                                loaded_exports)
                continue

            _CHAT_DISPLAY_ARCHIVE = chat_save.display if active_game is None else None
            try:
                save_command = _chat_save_argument(user_text)
            except ValueError as exc:
                print(f"  {exc}\n")
                continue
            if save_command is not None:
                command, argument = save_command
                _CHAT_DISPLAY_ARCHIVE = None
                if active_game is not None:
                    print("  /chatsave and /chatload are unavailable in game mode. Leave the game first.\n")
                    continue
                try:
                    if command == "/chatsave":
                        if pending_image_persona:
                            raise ValueError("Finish or cancel the pending image persona before saving.")
                        filename = chat_save.save(argument, writer_history, loaded_exports,
                                                  lookup_blocks, lookup_skip, focus_paths)
                        print(f"  Conversation saved: {filename}\n")
                        continue
                    directory = _chat_save_directory()
                    choices = sorted((f for f in os.listdir(directory) if f.endswith(".json")
                                      and os.path.isfile(os.path.join(directory, f))), reverse=True) if os.path.isdir(directory) else []
                    if not choices:
                        raise ValueError("No saved conversations in chats/.")
                    if argument:
                        filename = argument if argument.lower().endswith(".json") else argument + ".json"
                        if filename not in choices and CHATLOAD_NAME_MATCH:
                            # pre65: the name given to /chatsave, without its date and time
                            want = re.sub(r"\s+", "_", filename[:-5].strip()).casefold()
                            plain = lambda f: re.sub(r"^\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}-?", "", f[:-5]).casefold()
                            named = [f for f in choices if plain(f) == want] or [f for f in choices if want and want in plain(f)]
                            if len(named) == 1:
                                filename = named[0]
                            elif named:
                                pick = _pick_export_menu(f"{len(named)} saved conversations match '{argument}' -- which one?",
                                                         named)
                                if pick is None:
                                    continue
                                filename = named[pick]
                        if filename not in choices:
                            raise ValueError("No conversation with that filename in chats/.")
                    else:
                        pick = _pick_export_menu("Choose a saved conversation:", choices)
                        if pick is None:
                            continue
                        filename = choices[pick]
                    path = os.path.join(directory, filename)
                    data = _read_chat_save(path)
                    if data["persona"] is not None:
                        persona = data["persona"]
                        status("Checking the saved persona with the default assistant...")
                        try:
                            allowed, reason = _check_custom_persona(persona["request"],
                                images=persona["images"] or None, image_context=persona["clarification"])
                        finally:
                            status_done(clear=True)
                        if not allowed:
                            raise ValueError(reason)
                    persona_text, persona_label = _saved_persona_settings(data["persona"])
                    system = {"role": "system", "content": build_chat_system_prompt(persona_text)}
                    restored, indices, omitted = _resume_chat_history(data, system)
                    candidate = _ChatSaveArchive()
                    candidate.messages = data["messages"]
                    candidate.context_start = data["context_start"]
                    candidate.protected = {id(candidate.messages[i]) for i in data["protected"]}
                    candidate.sources = {id(m): (m, candidate.messages[i]) for m, i in zip(restored[1:], indices)}
                    candidate.persona = data["persona"]
                    candidate.path, candidate.title = path, data["title"]
                    candidate.display = data["transcript"]
                    candidate.videos = data["videos"]
                    # Commit only after validation, persona approval and budget selection succeed.
                    chat_save = candidate
                    active_persona_text, active_persona_label = persona_text, persona_label
                    pending_image_persona = None
                    writer_history[:] = restored
                    _PROTECTED_EVIDENCE_MESSAGE_IDS.clear()
                    _PROTECTED_EVIDENCE_MESSAGE_IDS.update(id(m) for m, i in zip(restored[1:], indices)
                                                         if i in data["active_evidence"])
                    loaded_exports.clear()
                    loaded_exports.update(data["exports"])
                    lookup_blocks[:] = data["lookup_blocks"]
                    lookup_skip.clear()
                    lookup_skip.update(data["lookup_skip"])
                    focus_paths = data["focus_paths"]
                    last_writer_message = None
                    _reset_tokens()
                    SESSION_TOKEN_USAGE.update(data["usage"])
                    print(f"  Resumed {filename}\n")
                    for line in candidate.display:
                        if not USE_COLOR:
                            line = re.sub(r"\x1b\[[0-9;]*m", "", line)
                        _builtin_print(line, end="")
                    if omitted:
                        print(f"  ({omitted} older message(s) remain in the save and transcript but are outside the active context.)\n")
                except (OSError, ValueError, TypeError, KeyError, RuntimeError) as exc:
                    print(f"  Conversation {command[5:]} failed: {exc}\n")
                continue
            musicvideo_rem = (_command_remainder(user_text, MUSICVIDEO_ANALYSE_FLAG)
                              if MUSICVIDEO_COMMANDS else None)
            if musicvideo_rem is not None:
                # pre56: the song, then the pictures, of one music video file.
                if active_game is not None:
                    print("  Video commands are unavailable in game mode. Leave the game first.\n")
                    continue
                if pending_image_persona:
                    print("  Finish or cancel the pending image persona before using video commands.\n")
                    continue
                _CHAT_DISPLAY_ARCHIVE = None
                try:
                    _musicvideo_analyse(musicvideo_rem, audio_temp_files=audio_temp_files,
                                        dsp_temp_files=dsp_temp_files, stem_temp_files=stem_temp_files,
                                        demucs_out_dirs=demucs_out_dirs)
                except SystemExit:
                    pass
                except KeyboardInterrupt:
                    print("\n  Music video analysis interrupted. Run the same command again to continue: "
                          "finished parts are skipped.\n")
                except (ValueError, OSError, KeyError, TypeError, RuntimeError) as exc:
                    print(f"  Music video analysis failed: {exc}\n")
                _songformer_after_analysis()
                continue
            display_user_text = user_text
            # pre62: /lookup-search "Title" ... and (LOOKUP_WITH_LOAD_COMMANDS) /lookup come out
            # of the line before the video and song commands are read, so they work in any
            # order with /load, /videoload and /musicvideoload.
            search_names = []
            early_lookup = False
            if LOOKUP_SEARCH_COMMAND:
                user_text, search_names, _search_problem = _lookup_search_extract(user_text)
                if _search_problem:
                    print(_search_problem)
                    continue
            if LOOKUP_WITH_LOAD_COMMANDS:
                _lookup_re = r"(?i)(?<!\S)" + re.escape(LOOKUP_FLAG) + r"(?=\s|$)"
                if re.search(_lookup_re, user_text):
                    early_lookup = True
                    user_text = re.sub(r"[ \t]{2,}", " ", re.sub(_lookup_re, " ", user_text)).strip()
            inline_video_images = []
            inline_video_labels = []
            video_command = next((flag for flag in ("/videoanalyse", "/videoframe", "/videoembed", "/videostrip")
                                  if _command_remainder(user_text, flag) is not None), None)
            has_inline_video = bool(re.search(r"(?i)(?<!\S)/(?:videoload|frame)(?=\s|$)", user_text))
            has_musicvideo = MUSICVIDEO_COMMANDS and bool(
                re.search(r"(?i)(?<!\S)" + re.escape(MUSICVIDEO_LOAD_FLAG) + r"(?=\s|$)", user_text))
            if video_command is not None or has_inline_video or has_musicvideo:
                if active_game is not None:
                    print("  Video commands are unavailable in game mode. Leave the game first.\n")
                    continue
                if pending_image_persona:
                    print("  Finish or cancel the pending image persona before using video commands.\n")
                    continue
                musicvideo_pairs = []
                if has_musicvideo:
                    # pre56: each /musicvideoload becomes its video plus its song's export.
                    try:
                        if video_command is not None:
                            raise ValueError(f"{MUSICVIDEO_LOAD_FLAG} can't be combined with {video_command}.")
                        user_text, _names = _musicvideo_inline(user_text)
                        _video = _load_video_extension()
                        for _name in _names:
                            _pair = _musicvideo_resolve(_name, _video)
                            _problem = _musicvideo_unready(_pair)
                            if _problem:
                                raise ValueError(_problem)
                            if _pair["stale"]:
                                print(f"  (The song was analysed again after its export. To bring the chat up "
                                      f"to date, export it again:\n    {_musicvideo_export_command(_pair['song'])})")
                            if _pair not in musicvideo_pairs:
                                musicvideo_pairs.append(_pair)
                    except KeyboardInterrupt:
                        print("\n  Music video selection interrupted; returning to chat.\n")
                        continue
                    except (ValueError, OSError, KeyError, TypeError) as exc:
                        if str(exc) != "Music video selection cancelled.":
                            print(f"  {exc}\n")
                        continue
                try:
                    import shlex
                    video = _load_video_extension()
                    if video_command is not None:
                        arguments = shlex.split(_command_remainder(user_text, video_command))
                        if video_command == "/videoanalyse":
                            _CHAT_DISPLAY_ARCHIVE = None
                            if VIDEO_FRIENDLY_USAGE:
                                options = _video_parse_analyse(video, arguments, "/videoanalyse", "[options]")
                                if options is None:
                                    continue
                            else:
                                options = vars(video.make_parser().parse_args(["analyse"] + arguments))
                            options.pop("command")
                            # Empty quotes are not a video path (Path("") denotes the current directory).
                            if not options["source"].strip():
                                print('  Usage: /videoanalyse "<video file>" [options]\n')
                                continue
                            # Validate the supplied video before asking for a cosmetic name.
                            video.probe_video(video.Path(options["source"]).expanduser())
                            if not options.get("name"):
                                options["name"] = colored_input("  Video name (Enter keeps the original name): ", Ansi.LIGHT_GREEN, prompt_color=Ansi.YELLOW).strip() or None
                            if not options.get("name") and VIDEO_DEFAULT_SAVE_NAME:
                                # pre53: the same name its audio analysis is saved under,
                                # e.g. "Artist - Song (music video)".
                                try:
                                    _vmeta = extract_audio_metadata(os.path.abspath(os.path.expanduser(options["source"])))[0]
                                except Exception:
                                    _vmeta = {}
                                options["name"] = _batch_save_basename(
                                    os.path.abspath(os.path.expanduser(options["source"])), metadata=_vmeta)[:-len(".json")]
                            if "--model" not in arguments and not any(a.startswith("--model=") for a in arguments):
                                options["model"] = OLLAMA_MODEL
                            if "--endpoint" not in arguments and not any(a.startswith("--endpoint=") for a in arguments):
                                options["endpoint"] = OLLAMA_URL
                            report_path = video.analyse_video(**options)
                            print(f'  Load this visual report with /videoload "{report_path}"\n')
                            continue
                        if not arguments:
                            raise ValueError(f"Usage: {video_command} <report.json>")
                        report_path = video.resolve_report(arguments[0], _pick_export_menu)
                        if video_command == "/videostrip":
                            if len(arguments) != 1:
                                raise ValueError("Use /videostrip <report> on its own.")
                            target = video.strip_report(report_path)
                            print(f"  Text-only copy (no frame images) written to {target}\n")
                            continue
                        if video_command == "/videoembed":
                            if len(arguments) != 1:
                                raise ValueError("Use /videoembed <report.json> on its own.")
                            video.embed_report(report_path)
                            print("  Existing reference images embedded; no model analysis was run.\n")
                            continue
                        if len(arguments) < 2:
                            raise ValueError("Use /videoframe <report.json> <seconds or MM:SS> [question].")
                        image_path, frame_time = video.reference_frame(report_path, video.parse_time(arguments[1]))
                        question = " ".join(arguments[2:]) or "Describe this reference frame."
                        user_text = shlex.quote(str(image_path)) + " " + question
                        video_name = video.read_report(report_path)["source"]["name"]
                        user_text += f" (Reference frame from {video_name!r}, sampled at video time {video.timestamp(frame_time)}.)"
                    else:
                        remaining, video_names, frame_requests = video.inline_commands(user_text)
                        if musicvideo_pairs:
                            # pre56: the paired videos load first, their songs' exports join the /load.
                            video_names = [pair["report"] for pair in musicvideo_pairs] + video_names
                            remaining = _musicvideo_add_loads(remaining, [pair["export"] for pair in musicvideo_pairs])
                            for pair in musicvideo_pairs:
                                musicvideo_links[os.path.realpath(pair["report"])] = pair["export"]
                        if (video_names or frame_requests) and re.search(r"(?i)(?<!\S)/(?:persona|clear|clearall|game|game-exit|analyse|reanalyse|batch|videoanalyse|videoembed|videostrip|chatsave|chatload"
                                                                         + ("|musicvideoanalyse|musicvideoload" if MUSICVIDEO_COMMANDS else "")
                                                                         + r")(?=\s|$)", remaining):
                            raise ValueError("Video loading/frame requests cannot be combined with session or analysis commands.")
                        staged = []
                        context = _chat_num_ctx()
                        part_limit = max(1, min(MAX_MESSAGE_CHARS_FOR_OLLAMA,
                                         max(4000, int(context * HISTORY_CHAR_BUDGET_FACTOR) // 3)) - 512)
                        for name in video_names:
                            report_path = video.resolve_report(name, _pick_export_menu)
                            block, estimate = video.render_report(report_path, max_tokens=VIDEO_CHAT_MAX_TOKENS)
                            report = video.portable_report(report_path)
                            pieces = _split_export_for_chat(block, part_limit)
                            if len(pieces) > 1:
                                heading = block.split("\n", 1)[0]
                                pieces = [heading + f" (part {i}/{len(pieces)})\n" + piece
                                          for i, piece in enumerate(pieces, 1)]
                            key = str(report_path)
                            entry = {"alias": report_path.parent.name, "report": report, "block": pieces[0],
                                     "auto_indices": _video_reference_frames(video, report), "association": ""}
                            for index in entry["auto_indices"]:
                                video.frame_reference(report, index)
                            previous = chat_save.videos.get(key)
                            if previous:
                                if previous["block"] != entry["block"]:
                                    raise ValueError("This video's report changed since loading. Use /clearall before loading its revised analysis.")
                                entry = previous
                            if not any(item[0] == key for item in staged):
                                staged.append((key, entry, pieces, estimate))
                        candidates = dict(chat_save.videos)
                        candidates.update({key: entry for key, entry, pieces, estimate in staged})
                        for name, index in frame_requests:
                            entry = video.loaded_video(candidates, name, _pick_export_menu)
                            image, label = video.frame_reference(entry["report"], index)
                            inline_video_images.append(image)
                            inline_video_labels.append(label)
                        if _chat_image_limit() and len(inline_video_images) > _chat_image_limit():
                            raise ValueError(f"At most {_chat_image_limit()} images can be attached to one question.")
                        additions = list(dict.fromkeys(piece for key, entry, pieces, estimate in staged
                                                      for piece in pieces if piece not in lookup_blocks))
                        new_image_count = sum(len(entry["auto_indices"]) for key, entry, pieces, estimate in staged
                                              if entry["block"] in additions)
                        fixed = sum(len(str(m.get("content") or ""))
                                    + int(CHAT_IMAGE_TOKENS * HISTORY_CHAR_BUDGET_FACTOR) * len(m.get("images") or [])
                                    for m in writer_history if m.get("role") == "system"
                                    or id(m) in _PROTECTED_EVIDENCE_MESSAGE_IDS)
                        room = int((context - _ollama_output_budget(context, 16384)) * HISTORY_CHAR_BUDGET_FACTOR) - 400
                        if fixed + sum(map(len, additions)) + (new_image_count + len(inline_video_images)) * int(CHAT_IMAGE_TOKENS * HISTORY_CHAR_BUDGET_FACTOR) > room:
                            raise ValueError("The visual report and images plus loaded evidence exceed the estimated context budget. Clear some context or increase the configured window.")
                        for key, entry, pieces, estimate in staged:
                            if key not in chat_save.videos:
                                chat_save.videos[key] = entry
                            print(f"  Loaded video {entry['report']['source']['name']!r}: approximately {estimate:,} text tokens; "
                                  f"{len(entry['auto_indices'])} automatic reference images "
                                  f"(~{len(entry['auto_indices']) * CHAT_IMAGE_TOKENS:,} estimated image tokens).\n")
                        for piece in additions:
                            lookup_blocks.append(piece)
                            writer_history.append(_background_message(piece))
                        user_text = remaining
                except SystemExit:
                    continue
                except KeyboardInterrupt:
                    print("\n  Video operation interrupted; returning to chat.\n")
                    continue
                except (ValueError, OSError, KeyError, TypeError) as exc:
                    print(f"  Video operation failed: {exc}\n")
                    continue
            if active_game is None:
                chat_save.display.append(_colorize("You: " + display_user_text, Ansi.LIGHT_GREEN) + "\n")
            if active_game is None:
                chat_save.sync(writer_history)
            _before_compact = len(writer_history)
            _chat_compact(writer_history)
            if len(writer_history) < _before_compact:
                print(f"  History compaction: messages {_before_compact} -> {len(writer_history)}; "
                      f"Earlier context changed.")

            # /quiet (0.6.2-dev9) belongs to /load: load without a reply.
            quiet = False
            _quiet_re = r"(?i)(?<!\S)" + re.escape(QUIET_FLAG) + r"(?=\s|$)"
            if _command_remainder(user_text, LOAD_FLAG) is not None and re.search(_quiet_re, user_text):
                quiet = True
                user_text = re.sub(r"[ \t]{2,}", " ", re.sub(_quiet_re, " ", user_text)).strip()
            elif user_text.lower() == QUIET_FLAG:
                print(f"  {QUIET_FLAG} goes with {LOAD_FLAG}: {LOAD_FLAG} \"Artist - Title\" {QUIET_FLAG} loads "
                      "without a reply.\n")
                continue

            # /think (0.6.2-dev12) can sit anywhere in a prompt: this reply thinks first.
            _think_re = r"(?i)(?<!\S)" + re.escape(THINK_FLAG) + r"(?=\s|$)"
            think_now = bool(re.search(_think_re, user_text))
            if think_now:
                user_text = re.sub(r"[ \t]{2,}", " ", re.sub(_think_re, " ", user_text)).strip()
                if not user_text:
                    print(f"  {THINK_FLAG} goes with a prompt: <question> {THINK_FLAG}, or "
                          f"{LOAD_FLAG} \"Artist - Title\" {THINK_FLAG}.\n")
                    continue

            # /lookup can sit anywhere in a prompt, alone or with /load.
            lookup = bool(re.search(r"(?i)(?<!\S)" + re.escape(LOOKUP_FLAG) + r"(?=\s|$)", user_text))
            lookup = lookup or early_lookup      # pre62: already taken out above
            if lookup:
                user_text = re.sub(r"(?i)(?<!\S)" + re.escape(LOOKUP_FLAG) + r"(?=\s|$)", " ", user_text)
                user_text = re.sub(r"[ \t]{2,}", " ", user_text).strip()

            game_exit = _command_remainder(user_text, GAME_EXIT_FLAG) is not None
            if game_exit and active_game is None:
                print(f"  Not in a game. {GAME_FLAG} shows the games.\n")
                continue
            if game_exit:
                left = active_game.get("title") or active_game["name"]
                active_game = None
                active_persona_text = None
                active_persona_label = "default"

            if game_exit or _command_remainder(user_text, CLEAR_ALL_FLAG) is not None:
                saved_persona = None if game_exit else chat_save.persona
                _fresh_chat_archive()
                chat_save.persona = saved_persona
                pending_image_persona = None
                writer_history[:] = [{"role": "system", "content": _system_prompt()}]
                _PROTECTED_EVIDENCE_MESSAGE_IDS.clear()
                loaded_exports.clear()
                lookup_blocks.clear()
                lookup_skip.clear()
                focus_paths = []
                last_writer_message = None
                _reset_tokens()
                if game_exit:
                    print(f"  Left {left}. Everything cleared (conversation, loaded exports, images and "
                          f"looked-up background); back to the default Musiclyse chat.\n")
                else:
                    print(f"  Everything cleared: conversation, loaded exports and looked-up background. "
                          f"Files on disk are untouched. {_mode_label()}.\n")
                continue

            if _command_remainder(user_text, GAME_FLAG) is not None:
                if active_game is not None:
                    print(f"  You're playing {active_game.get('title') or active_game['name']}. "
                          f"{GAME_EXIT_FLAG} leaves it first.\n")
                    continue
                pick = _pick_from_list("Choose a game:", [g.get("menu") or g["name"] for g in CHAT_GAMES])
                if pick is None:
                    print("  No game chosen.\n")
                    continue
                active_game = CHAT_GAMES[pick]
                _fresh_chat_archive()
                pending_image_persona = None
                writer_history[:] = [{"role": "system", "content": _system_prompt()}]
                warn_if_system_prompt_truncated(writer_history[0]["content"], num_ctx=_chat_num_ctx())
                _PROTECTED_EVIDENCE_MESSAGE_IDS.clear()
                for path, entry in loaded_exports.items():
                    writer_history.extend(_export_messages(path, entry))
                for block in lookup_blocks:
                    writer_history.append(_background_message(block))
                last_writer_message = None
                _reset_tokens()
                kept = (f" Kept and sent again: {len(loaded_exports)} export(s), {len(lookup_blocks)} "
                        f"looked-up background block(s)." if loaded_exports or lookup_blocks else "")
                _banner = active_game.get("banner") or []
                if isinstance(_banner, str):
                    _banner = _banner.split("\n")
                if any(line.strip() for line in _banner):
                    print()
                    print(_colorize("\n".join(_banner),
                                    active_game.get("banner_color") or active_game["reply_color"]))
                    print()
                else:
                    print(_colorize(f"\n  {active_game.get('title') or active_game['name']}",
                                    active_game.get("banner_color") or active_game["reply_color"]))
                print(f"  The conversation starts afresh.{kept} {LOAD_FLAG} a track to begin; "
                      f"{CLEAR_ALL_FLAG} clears songs and images but keeps the game; "
                      f"{GAME_EXIT_FLAG} leaves it.\n")
                continue

            if _command_remainder(user_text, CLEAR_FLAG) is not None:
                if active_game is None:
                    chat_save.sync(writer_history)
                    chat_save.context_start = len(chat_save.messages)
                pending_image_persona = None
                writer_history[:] = [{"role": "system", "content": _system_prompt()}]
                _PROTECTED_EVIDENCE_MESSAGE_IDS.clear()
                for path, entry in loaded_exports.items():
                    writer_history.extend(_export_messages(path, entry))
                for block in lookup_blocks:
                    writer_history.append(_background_message(block))
                last_writer_message = None
                _reset_tokens()
                print(f"  Conversation cleared and token counters reset. Kept and sent again: "
                      f"{len(loaded_exports)} export(s), {len(lookup_blocks)} looked-up background block(s). "
                      f"{_mode_label()}.\n")
                continue

            if active_game is not None:
                blocked = next((f for f in (PERSONA_FLAG, BATCH_FLAG, REANALYSE_FLAG, ANALYSE_FLAG)
                                if _command_remainder(user_text, f) is not None), None)
                if blocked:
                    print(f"  {blocked} doesn't work in a game. Leave {active_game['name']} with "
                          f"{GAME_EXIT_FLAG} first.\n")
                    continue

            persona_rem = _command_remainder(user_text, PERSONA_FLAG)
            if persona_rem is not None:
                rem = persona_rem.strip()
                if not rem:
                    if pending_image_persona:
                        print(f"  Pending image persona: {pending_image_persona['description']}\n"
                              + ("  Reference retained: send a clarification, or a replacement image.\n"
                                 if pending_image_persona["images"] else "  Send the reference image.\n")
                              + "  /persona reset cancels the pending request.\n")
                    preview = (active_persona_text or "the default assistant").strip().replace("\n", " ")
                    if len(preview) > 160:
                        preview = preview[:157] + "..."
                    print(f"  Current persona: {active_persona_label}\n  Preview: {preview}\n"
                          f"  Usage:\n"
                          f"    {PERSONA_FLAG} <description>   chat as a custom character\n"
                          f"    {PERSONA_FLAG} reset           back to the default assistant\n"
                          f"    {PERSONA_FLAG}                 show the current persona\n")
                    continue
                is_reset = rem.lower() in ("reset", "default", "clear")
                if not is_reset and _persona_needs_image(rem):
                    if len(rem) > 4000:
                        print("  Please shorten the persona description to 4,000 characters.\n")
                        continue
                    pending_image_persona = {"description": rem, "images": [], "clarification": ""}
                    print("  Image persona pending: send the reference image next, with any needed "
                          "character name or clarification. Your current persona stays active until "
                          "the image-based request is approved. /persona reset cancels it.\n")
                    continue
                if not is_reset:
                    status("Checking the custom persona with the default assistant...")
                    try:
                        persona_allowed, persona_reason = _check_custom_persona(rem)
                    finally:
                        status_done(clear=True)
                    if not persona_allowed:
                        print(f"  {persona_reason} Current persona and conversation unchanged.\n")
                        continue
                pending_image_persona = None
                if is_reset:
                    active_persona_text = None
                    active_persona_label = "default"
                    label_msg = "the default assistant"
                else:
                    cleaned = rem
                    for prefix in ("roleplay as ", "role-play as ", "role play as ", "act as ", "be ", "you are "):
                        if cleaned.lower().startswith(prefix):
                            cleaned = cleaned[len(prefix):].strip()
                            break
                    if len(cleaned) < 100 and "\n" not in cleaned:
                        active_persona_text = chat_persona_brief(cleaned)
                        active_persona_label = cleaned[:160]
                    else:
                        active_persona_text = chat_persona_freeform(cleaned)
                        active_persona_label = cleaned.strip().split("\n", 1)[0][:160] or "custom"
                    label_msg = active_persona_label
                _clear_for_persona()
                last_writer_message = None
                chat_save.persona = None if is_reset else {"request": rem, "images": [], "clarification": ""}
                chat_save.display.append(_colorize("You: " + user_text, Ansi.LIGHT_GREEN) + "\n")
                new_system = build_chat_system_prompt(active_persona_text)
                warn_if_system_prompt_truncated(new_system, num_ctx=_chat_num_ctx())
                if writer_history and writer_history[0].get("role") == "system":
                    writer_history[0] = {"role": "system", "content": new_system}
                else:
                    writer_history.insert(0, {"role": "system", "content": new_system})
                _strip_persona_markers(writer_history)
                if is_reset:
                    writer_history.append({"role": "user", "content": (
                        f"{PERSONA_MARKER_USER_PREFIX}{label_msg}. The custom character is gone -- drop "
                        "its voice, taste and limits completely. You are the default assistant again, "
                        "with no custom character assigned.]")})
                    writer_history.append({"role": "assistant", "content": (
                        f"{PERSONA_MARKER_ASSISTANT_PREFIX}{label_msg} -- back to my own voice.")})
                else:
                    writer_history.append({"role": "user", "content": (
                        f"{PERSONA_MARKER_USER_PREFIX}{label_msg}. The user has assigned you this custom "
                        "character: use their voice and taste subject to CUSTOM PERSONA — PRESERVE "
                        "NORMAL BOUNDARIES; this switch does not imply acceptance of any request. "
                        "What they know about an artist follows the custom-character rules in "
                        "each loaded export.]")})
                if PERSONA_CONFIRM_FULL and not is_reset:
                    print("  Persona set to: " + cleaned.strip().replace("\n", "\n  ") + "\n")
                else:
                    print(f"  Persona set to: {label_msg}\n")
                continue

            batch_rem = _command_remainder(user_text, BATCH_FLAG)
            if batch_rem is not None:
                folder = batch_rem.strip().strip('"').strip("'")
                if (BATCH_IMAGE_OPTIONS_NOTE and SCAN_IMAGE_OPTIONS
                        and re.search(r"(?<!\S)--(?:cover|photo)(?:=|\s|$)", batch_rem)):
                    # 0.8: the image options are for one track
                    print(f"  {BATCH_FLAG}: --cover and --photo work on one track at a time, not a whole "
                          f"folder -- use {ANALYSE_FLAG} (or {REANALYSE_FLAG}) on that track with the image.\n")
                    continue
                if not folder and BATCH_DEFAULT_FOLDER:
                    # 0.8 BATCH_DEFAULT_FOLDER: the batch folder beside this script, when there is one
                    _default_batch = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                  os.path.expanduser(str(BATCH_DEFAULT_FOLDER)))
                    if os.path.isdir(_default_batch):
                        folder = _default_batch
                        print(f"  (No folder given: analysing {_default_batch})")
                if not folder:
                    print(f"  Usage: {BATCH_FLAG} /path/to/folder\n"
                          f"  Analyses every audio file in that folder (non-recursive) and saves each "
                          f"to {SAVE_DIR}/. Files already saved are skipped."
                          + (f" With no folder, a folder called {BATCH_DEFAULT_FOLDER!r} beside this script "
                             f"is used when there is one." if BATCH_DEFAULT_FOLDER else "") + "\n")
                    continue
                folder = os.path.expanduser(folder)
                if not os.path.isdir(folder):
                    print(f"  Not a directory: {folder}\n")
                    continue
                try:
                    batch_scan_folder_to_saved_songs(
                        folder,
                        audio_temp_files=audio_temp_files,
                        dsp_temp_files=dsp_temp_files,
                        stem_temp_files=stem_temp_files,
                        demucs_out_dirs=demucs_out_dirs,
                        skip_existing=True,
                    )
                except Exception as e:
                    print(f"  Batch scan failed: {e}\n")
                _songformer_after_analysis()
                continue

            export_rem = _command_remainder(user_text, EXPORT_FLAG) if EXPORT_COMMAND else None
            if export_rem is not None:
                _run_export_command(export_rem)      # pre57: the exporter script, run as it is
                continue

            songformer_rem = (_command_remainder(user_text, SONGFORMER_FLAG)
                              if SONGFORMER_COMMAND and SONGFORMER_SLASH_COMMAND else None)   # pre65
            if songformer_rem is not None:
                _run_songformer_command(songformer_rem)   # pre58 (pre59: + check); changes no save
                continue

            wikidelete_rem = _command_remainder(user_text, WIKI_DELETE_FLAG) if WIKI_DELETE_COMMAND else None
            if wikidelete_rem is not None:
                # pre63: marks articles to leave out of a clean copy; the database is only read.
                _run_wikidelete_command(wikidelete_rem,
                                        highlight=active_game.get("menu_color") if active_game is not None else None)
                continue

            reanalyse_rem = _command_remainder(user_text, REANALYSE_FLAG)
            analyse_rem = None if reanalyse_rem is not None else _command_remainder(user_text, ANALYSE_FLAG)
            if reanalyse_rem is not None or analyse_rem is not None:
                flag = REANALYSE_FLAG if reanalyse_rem is not None else ANALYSE_FLAG
                # 0.8-pre34 SCAN_IMAGE_OPTIONS: --cover / --photo come off the argument first
                _arg, _images, _image_error = _scan_image_options(
                    reanalyse_rem if reanalyse_rem is not None else analyse_rem)
                if _image_error:
                    print(f"  {flag}: {_image_error}.\n")
                    continue
                path, why = _audio_path_argument(_arg)
                if not path:
                    print(f"  {flag}: {why}.\n  Usage: {flag} /path/to/song.m4a"
                          + (' [--cover "image"] [--photo "image"]' if SCAN_IMAGE_OPTIONS else "") + "\n")
                    continue
                if _images:
                    _scan_images_use(_images)
                try:
                    scan_files_to_saved_songs(
                        [path],
                        audio_temp_files=audio_temp_files,
                        dsp_temp_files=dsp_temp_files,
                        stem_temp_files=stem_temp_files,
                        demucs_out_dirs=demucs_out_dirs,
                        skip_existing=(reanalyse_rem is None),
                    )
                except Exception as e:
                    print(f"  Analysis failed: {e}\n")
                finally:
                    if _images:
                        _scan_images_use({})
                _songformer_after_analysis()
                continue

            if _command_remainder(user_text, DEBUG_FLAG) is not None:
                if last_writer_message is None:
                    print("  No message sent to the chat model yet.\n")
                else:
                    print(json.dumps(_redact_message_for_debug(last_writer_message), indent=2, default=str))
                    print(f"  loaded exports: {len(loaded_exports)}; looked-up blocks: {len(lookup_blocks)}; "
                          f"articles skipped by {LOOKUP_FLAG}: {len(lookup_skip)}\n")
                continue

            question = user_text
            load_added = False
            load_rem = _command_remainder(user_text, LOAD_FLAG)
            if load_rem is not None:
                if not load_rem:
                    print(f"  Usage: {LOAD_FLAG} \"Artist - Title\" [question]\n"
                          f"         {LOAD_FLAG} \"Artist 1 - Title 1\" \"Artist 2 - Title 2\" ... [question]\n")
                    continue
                paths, question = _resolve_names(load_rem, True)
                if paths is None:
                    continue
                _ids = {id(m) for m in writer_history}
                loaded = _load_exports(paths)
                load_added = bool(_new_messages_since(_ids))
                if not loaded:
                    continue
                focus_paths = loaded
                print()
            elif re.match(r"/[A-Za-z][\w-]*(?=\s|$)", user_text) and not os.path.exists(user_text.split()[0]):
                # A command-like word; a path to a file (a dragged-in image) is a normal prompt.
                cmd = user_text.split()[0].lower()
                removed = {"/listen": ANALYSE_FLAG, "/relisten": REANALYSE_FLAG, "/save": ANALYSE_FLAG,
                           "/loadcompare": LOAD_FLAG, "/loadmulti": LOAD_FLAG, "/correct": None}
                if cmd in removed:
                    alt = removed[cmd]
                    print(f"  {cmd} was removed in 0.6.2-dev4"
                          + (f" -- use {alt}" if alt else " -- tell the chat in plain words instead")
                          + (f", which takes one or several names: {LOAD_FLAG} \"Artist 1 - Title 1\" "
                             f"\"Artist 2 - Title 2\" [question]" if alt == LOAD_FLAG else "")
                          + ".\n")
                else:
                    print(f"  Unknown command {cmd}.\n")
                continue

            if chat_save.videos:
                video = _load_video_extension()
                tracks = [dict(track, filename=entry["name"] if len(entry["tracks"]) == 1 else "")
                          for entry in loaded_exports.values() for track in entry["tracks"]]
                for _video_key, entry in list(chat_save.videos.items()):
                    if entry["association"]:
                        continue
                    report = entry["report"]
                    # pre56: a video loaded with /musicvideoload is paired with its own song.
                    paired = (_musicvideo_paired_label(musicvideo_links.get(os.path.realpath(_video_key)), loaded_exports)
                              if MUSICVIDEO_COMMANDS else "")
                    if paired:
                        explicit, selected = "", paired
                    else:
                        matches = video.associated_song(report["source"]["name"], tracks)
                        if not matches and report["source"].get("original_name"):
                            matches = video.associated_song(report["source"]["original_name"], tracks)
                        explicit = report["alignment"]["song"]
                        selected = explicit or (matches[0] if len(matches) == 1 else "")
                        if not explicit and len(matches) > 1:
                            pick = _pick_export_menu(f"Which song belongs to {report['source']['name']}?", matches + ["Leave unassociated"])
                            if pick is not None and pick < len(matches):
                                selected = matches[pick]
                    if selected:
                        entry["association"] = selected
                        same_source = bool(paired) or (report["alignment"].get("video_minus_song_seconds") is None
                                                       and _video_song_same_source(report, selected))
                        association = (f"[Musiclyse visual report: association] Video {report['source']['name']!r} "
                                       f"is associated with song {selected!r} "
                                       + ("because both were analysed from the same music video file. " if paired else
                                          "by explicit video metadata. " if explicit else "by matching names. ")
                                       + "The user's corrections or explicit pairing instructions override this association. "
                                       + ("The song's analysis was made from this same video file, so song time equals "
                                          "video time: a song timestamp and a video timestamp name the same moment (within "
                                          "about a second, the frame spacing). This replaces the report's 'unverified' "
                                          "alignment note; shots can be lined up with the song's sections by time."
                                          if same_source else
                                          "Matching names alone do not establish synchronized timestamps or identical versions; "
                                          "use a supplied timing offset for precise synchronization."))
                        if VIDEO_SECTIONS_NOTE and (same_source or report["alignment"].get("video_minus_song_seconds") is not None):
                            association += _video_sections_note(          # 0.8-pre31
                                video, report, selected, loaded_exports,
                                0.0 if same_source else report["alignment"]["video_minus_song_seconds"])
                        lookup_blocks.append(association)
                        writer_history.append(_background_message(association))
                        print(f"  Video/song association: {report['source']['name']} ↔ {selected}"
                              + (" (same file: timings match)" if same_source else "") + "\n")

            question, images = _chat_images(question) if question else (question, [])
            if inline_video_images:
                if _chat_image_limit() and len(images) + len(inline_video_images) > _chat_image_limit():
                    print(f"  Too many images: use at most {_chat_image_limit()} per question.\n")
                    continue
                images += inline_video_images
                question = (question or "Describe these reference frames.") + "\n[Attached video images, following any other images, in order: " + "; ".join(inline_video_labels) + "]"
                print(f"  Attached {len(inline_video_images)} video frames (~{len(inline_video_images) * CHAT_IMAGE_TOKENS:,} estimated image tokens).\n")

            if pending_image_persona and active_game is None:
                # Keep unresolved references outside writer_history. A text-only
                # follow-up is clarification for this request, not ordinary chat.
                if images:
                    if pending_image_persona["images"]:
                        # A replacement reference must not inherit the old identity.
                        pending_image_persona["clarification"] = ""
                    pending_image_persona["images"] = list(images)
                if question:
                    pending_image_persona["clarification"] = "\n".join(
                        part for part in (pending_image_persona["clarification"], question) if part)
                if not pending_image_persona["images"]:
                    print("  Clarification retained. Image persona still pending: send its reference "
                          "image, or /persona reset to cancel.\n")
                    continue
                images = list(pending_image_persona["images"])
                clarification = pending_image_persona["clarification"]
                status("Checking the image-based persona with the default assistant...")
                try:
                    persona_allowed, persona_reason = _check_custom_persona(
                        pending_image_persona["description"], images=images, image_context=clarification)
                finally:
                    status_done(clear=True)
                if not persona_allowed:
                    print(f"  {persona_reason} Current persona unchanged; image request remains pending.\n")
                    continue
                original_persona_request = pending_image_persona["description"]
                _clear_for_persona()
                last_writer_message = None
                chat_save.persona = {"request": original_persona_request, "images": list(images),
                                     "clarification": clarification}
                chat_save.display.append(_colorize("You: /persona " + original_persona_request, Ansi.LIGHT_GREEN) + "\n")
                chat_save.display.append(_colorize("You: " + user_text, Ansi.LIGHT_GREEN) + "\n")
                reference_description = pending_image_persona["description"]
                if clarification:
                    reference_description += "\nUser clarification accompanying the reference: " + clarification
                active_persona_text = _image_persona_brief(reference_description)
                active_persona_label = pending_image_persona["description"][:160]
                pending_image_persona = None
                writer_history[:] = [{"role": "system", "content": _system_prompt()}]
                warn_if_system_prompt_truncated(writer_history[0]["content"], num_ctx=_chat_num_ctx())
                _strip_persona_markers(writer_history)
                print(f"  Persona set to: {active_persona_label}\n")
                # Reattach the approved reference even on a text-only follow-up,
                # and explicitly request roleplay rather than image description.
                question = (
                    "[Image-based persona activated. These are the reference images for the "
                    "requested role. Ground the portrayal in this reference's visible presentation "
                    "and supported setting, following REFERENCE-GROUNDED PORTRAYAL and HISTORICAL "
                    "SETTING in the system instructions. A supplied name does not select a "
                    "present-day persona. Respond under the custom-persona rules; do not give a "
                    "default image description.]\n\n"
                    + ("User clarification of the role: " + clarification + "\n\n" if clarification else "")
                    + "If a material detail of the intended period or scene is uncertain, ask one "
                    "brief clarification question first. Otherwise begin the requested roleplay "
                    "with a brief introduction grounded in the depicted setting."
                )
            if images and not question:
                question = "Describe these images." if len(images) > 1 else "Describe this image."

            if search_names:
                # pre62 /lookup-search: the articles named by title, before any /lookup (which
                # then skips what they already carry).
                if not ENABLE_WIKI_CONTEXT:
                    print(f"  ({LOOKUP_SEARCH_FLAG} is off: ENABLE_WIKI_CONTEXT = False)")
                else:
                    searched = _lookup_search(search_names, question)
                    if searched:
                        lookup_blocks.append(searched)
                        writer_history.append(_background_message(searched))
                        if not question and not load_added and not lookup:
                            print("  (They stay in this conversation for your next message.)\n")
                if not question and not load_added and not lookup:
                    continue

            background = ""
            if lookup:
                if not ENABLE_WIKI_CONTEXT:
                    print("  (lookup is off: ENABLE_WIKI_CONTEXT = False)")
                elif not question and not focus_paths:
                    print(f"  Usage: {LOOKUP_FLAG} <question>, or {LOAD_FLAG} <file.txt> {LOOKUP_FLAG}\n")
                    continue
                else:
                    background = _lookup(question)
                    if background:
                        lookup_blocks.append(background)
                        writer_history.append(_background_message(background))

            if not question:
                if load_added and LOAD_REPLIES_BY_DEFAULT and not quiet:
                    # Answer the export itself, as a file sent on its own is answered.
                    last_writer_message = writer_history[-1]
                    _reply(think=think_now)
                continue

            writer_message = {"role": "user", "content": question}
            if images:
                writer_message["images"] = images
            last_writer_message = writer_message
            writer_history.append(writer_message)
            if _reply(image_message=writer_message if images else None, think=think_now) is None:
                writer_history.remove(writer_message)
                archived = chat_save.sources.pop(id(writer_message), None)
                if archived:
                    chat_save.messages[:] = [m for m in chat_save.messages if m is not archived[1]]
    finally:
        _CHAT_DISPLAY_ARCHIVE = None
        # dev2: each step is guarded, so one file that cannot be removed no
        # longer skips the rest of the cleanup (model unloading, Demucs and
        # Omnizart folders) or hides the error that ended the session.
        if UNLOAD_CHAT_MODEL_ON_EXIT:
            _unload_chat_model_on_exit()
        for _store in (audio_temp_files, dsp_temp_files, stem_temp_files):
            for temp_path in list(_store.values()):
                _remove_quietly(temp_path)

        try:
            _clear_omnizart_prefetch()      # dev2: no helper process left running
        except Exception:
            pass
        try:
            unload_moss_music()
        except Exception:
            pass
        try:
            unload_whisper()
        except Exception:
            pass

        try:
            _release_omnizart_memory()
        except Exception:
            pass

        for d in demucs_out_dirs:
            try:
                shutil.rmtree(d, ignore_errors=True)
            except Exception:
                pass

        for d in _OMNIZART_OUTPUT_DIRS:
            try:
                shutil.rmtree(d, ignore_errors=True)
            except Exception:
                pass

        for p in _MOSS_RETRY_TEMP_FILES:
            try:
                os.remove(p)
            except Exception:
                pass

# 0.7.1-pre22: the other Musiclyse files run here, after this script's own code (see "THE MUSICLYSE
# FILES" above).
_musiclyse_load_files()


if __name__ == "__main__":
    # dev55: background Omnizart transcription (see _start_omnizart_prefetch).
    if len(sys.argv) >= 4 and sys.argv[1] == "--omnizart-prefetch":
        try:
            _omnizart_prefetch_child(sys.argv[2], sys.argv[3])
            sys.exit(0)
        except Exception as e:
            print(f"  omnizart prefetch failed: {e}", file=sys.stderr)
            sys.exit(1)

    # Subprocess entry for isolated batch tracks:
    #   python musiclyse.py --batch-one /path/to/song.mp3
    # Parent batch orchestrator launches this so each track gets a clean process.
    if len(sys.argv) >= 3 and sys.argv[1] == "--batch-one":
        track = sys.argv[2]
        # Minimal banner so overnight logs stay readable.
        print(f"  [batch-one] {os.path.basename(track)}")
        try:
            run_batch_one_track(track)
            sys.exit(0)
        except Exception as e:
            print(f"  ✗ batch-one failed: {e}", file=sys.stderr)
            try:
                _aggressive_memory_cleanup()
            except Exception:
                pass
            sys.exit(1)

    # Split-process batch entries (BATCH_SPLIT_IDENTITY_PROCESS, default on):
    #   python musiclyse.py --batch-one-analyze /path/to/song.mp3 /tmp/staging.json
    #   python musiclyse.py --batch-one-finish /tmp/staging.json
    # The parent orchestrator (_batch_run_isolated) runs these as two
    # separate processes per track so the identity/save stage always starts
    # from a clean memory baseline, regardless of what the heavy stage's
    # process was unable to fully release before it exits.
    if len(sys.argv) >= 4 and sys.argv[1] == "--batch-one-analyze":
        track, staging_path = sys.argv[2], sys.argv[3]
        print(f"  [batch-one-analyze] {os.path.basename(track)}")
        try:
            run_batch_one_analyze(track, staging_path)
            sys.exit(0)
        except Exception as e:
            print(f"  ✗ batch-one-analyze failed: {e}", file=sys.stderr)
            try:
                _aggressive_memory_cleanup()
            except Exception:
                pass
            sys.exit(1)

    if len(sys.argv) >= 3 and sys.argv[1] == "--batch-one-finish":
        staging_path = sys.argv[2]
        print(f"  [batch-one-finish] {os.path.basename(staging_path)}")
        try:
            run_batch_one_finish(staging_path)
            sys.exit(0)
        except Exception as e:
            print(f"  ✗ batch-one-finish failed: {e}", file=sys.stderr)
            try:
                _aggressive_memory_cleanup()
            except Exception:
                pass
            sys.exit(1)

    # dev2: Ctrl-C during a turn ends the session after the normal cleanup
    # without a Python traceback.
    try:
        main()
    except KeyboardInterrupt:
        print("\n  (stopped)")
        sys.exit(130)
