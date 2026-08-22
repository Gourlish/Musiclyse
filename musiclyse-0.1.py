"""
Musiclyse — chat about (and compare) multiple songs,
using a local LLM (via Ollama) to write the final response while a multi step pipeline performs grounded audio analysis on demand.

Routing:
    "/listen <question>" — analyzes the CURRENT track.
    "/listen <path or URL> <question>" — SWITCHES the current track to the
        given file/URL, then analyzes it. Each track keeps its own Music
        Flamingo conversation history, so switching back to an earlier
        track later still has that context.
    Anything else — goes straight to Gemma alone (general questions,
        comparisons between tracks already discussed, etc.), with the full
        conversation history — including every track's analysis so far.

Examples:
    /listen /Users/me/Music/song_a.mp3 what key and tempo is this in?
    /listen what instruments do you hear?              (still song_a)
    /listen /Users/me/Music/song_b.wav describe this one
    how does song_b's tempo compare to song_a?          (straight to Gemma)

New commands:
    /save=filename.json   Save the technical details for the most recently scanned track.
    /load filename.json [question]   Load a saved track; optional question after the name.
    /batch /path/to/folder   Overnight-scan every audio file in a folder into saved-songs/
                             (same analysis as /listen; does not import into chat).
    /clear                   Wipe chat context and reset session token counters.
    /persona <description>   Switch chat voice/taste (music evidence rules stay).
    /persona reset           Restore the default music-obsessed friend persona.

Prerequisites:
    brew install ollama ffmpeg
    # Optional but recommended for Essentia on macOS:
    #   conda create -n musicalyse python=3.10 && conda activate musicalyse && conda install -c conda-forge essentia
    ollama serve                       # or run the Ollama app
    ollama pull muse-glimmer-30b
    pip install requests librosa mutagen

Optional stem/MIDI stack:
    pip install demucs omnizart tensorflow pretty_midi

Usage:
    python musiclyse.py                     # no starting track — pick one with /listen
    python musiclyse.py /path/to/song.m4a   # optional starting track
"""

# ==============================================================================
# 1. RUNTIME MONKEY PATCHES (MUST BE AT THE VERY TOP)
# ==============================================================================
import sys
import numpy as np
import multiprocessing

# Fix deprecated/removed NumPy attributes for legacy packages (madmom/omnizart)
np.float = float
np.int = int
np.bool = bool
np.complex = complex

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

import sys
import os

import collections
import collections.abc

if not hasattr(collections, "MutableSequence"):
    collections.MutableSequence = collections.abc.MutableSequence
if not hasattr(collections, "MutableMapping"):
    collections.MutableMapping = collections.abc.MutableMapping
if not hasattr(collections, "Mapping"):
    collections.Mapping = collections.abc.Mapping
    
try:
    import pkg_resources
except ImportError:
    try:
        import pip._vendor.pkg_resources as pkg_resources
        sys.modules["pkg_resources"] = pkg_resources
    except ImportError:
        import packaging.version

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
import shlex
try:
    import readline  # enables left/right arrow editing in input()
except Exception:
    readline = None
import base64
import subprocess
import tempfile
import shutil
import json
import datetime
import gc

import requests
# Make PyTorch's MPS allocator release memory more aggressively after unload_music_flamingo().
# (This is the setting that previously loaded successfully on this machine.)
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")
import torch
import librosa
import numpy as np
from transformers import AudioFlamingo3ForConditionalGeneration, AutoProcessor


@contextlib.contextmanager
def quiet_stdout():
    """Swallow stray stdout (and optionally stderr) from third-party libraries
    (e.g. Keras/Omnizart progress bars) that print outside the logging module."""
    if globals().get("SHOW_OMNIZART_LOGS", False):
        yield
        return
    out_buf = io.StringIO()
    err_buf = io.StringIO()
    with contextlib.redirect_stdout(out_buf), contextlib.redirect_stderr(err_buf):
        yield


# --- Console appearance ----------------------------------------------------
# Set to False to fall back to the old plain scrolling log style (each
# processing step printed as its own permanent line, no overwriting).
FRIENDLY_PROCESSING_DISPLAY = True

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


_status_width = 0


def status(msg):
    """Print a progress-style status message. When FRIENDLY_PROCESSING_DISPLAY
    is True, overwrites the previous status on the same line (progress-bar
    style). When False, prints each step as its own permanent line (the
    original scrolling log style)."""
    global _status_width
    colored = _colorize(msg, Ansi.YELLOW)
    if FRIENDLY_PROCESSING_DISPLAY:
        line = f"  \u23f3 {colored}"
        pad = max(0, _status_width - len(line))
        print("\r" + line + " " * pad, end="", flush=True)
        _status_width = len(line)
    else:
        print(f"  ({colored})")
        _status_width = 0


def status_done(msg=None):
    """Finalize the status line: mark it done (friendly mode) and drop to a
    fresh line, or just print the final note (log-style mode)."""
    global _status_width
    if msg is not None:
        colored = _colorize(msg, Ansi.YELLOW)
        if FRIENDLY_PROCESSING_DISPLAY:
            line = f"  \u2713 {colored}"
            pad = max(0, _status_width - len(line))
            print("\r" + line + " " * pad)
        else:
            print(f"  ({colored})")
    elif FRIENDLY_PROCESSING_DISPLAY and _status_width:
        print()
    _status_width = 0


MUSICLYSE_LOGO = r"""
       @@@@    @@@@  @@@     @@    @@@@@     @@     @@@@@     @@      @@     @@   @@@@@    @@@@@@@@ 
       @@@@@  @@@@@  @@@    @@@  @@@  @@@   @@    @@@.@@@@   @@@      @@@  @@@  @@   @@@  @@@@      
      @@@@@  @@@@@  @@@    @@@  @@@@       @@@  @@@         @@@       @@@@@@   @@@@       @@@       
      @@ @@ @@ @@@  @@@    @@@   @@@@@@@   @@   @@          @@         @@@@     @@@@@@@  @@@@@@@@@  
     @@  @@@@  @@  @@@    @@@        @@@  @@@  @@@         @@@         @@           @@@  @@@        
 @@@@@@  @@@  @@   @@@   @@@  @@@    @@  @@@   @@@   @@@   @@@@@@@    @@@    @@@   @@@  @@@         
@@@@@@  @@@  @@@   @@@@@@@     @@@@@@    @@@    @@@@@@    @@@@@@@@   @@@      @@@@@@@  @@@@@@@@@    
  @@@                                                                                               
                         L O C A L    M U S I C    D I S C U S S I O N

                         ========== V E R S I O N     0 . 1 ==========

                         A   G O U R L I S H   V I B E   P R O J E C T
"""


def set_terminal_title(title="Musiclyse 0.1"):
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
    set_terminal_title("Musiclyse 0.1")
    print(_colorize(MUSICLYSE_LOGO, Ansi.WHITE))


def colored_input(prompt, color):
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
    # \001 / \002 = readline non-printing markers (width-safe). Only useful when
    # the readline module is loaded; otherwise leave plain ANSI in the prompt.
    if readline is not None:
        rl_prompt = f"\001{color}\002{prompt}"
    else:
        rl_prompt = f"{color}{prompt}"
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

ENABLE_FILE_METADATA = True
SEND_COVER_ART_TO_OLLAMA = True
METADATA_MAX_LYRICS_CHARS = 10000
COVER_IMAGE_MAX_DIMENSION = 512
# ffmpeg's mjpeg encoder uses qscale 2..31, where lower is better.
# The old value "80" was invalid and caused ffmpeg cover scaling/extraction to fail.
COVER_IMAGE_QUALITY = "2"
MAX_COVER_IMAGES_PER_REQUEST = 1

# If conversion fails but the original embedded art is directly sendable, allow this size as a fallback.
MAX_COVER_BYTES_TO_SEND_RAW = 1_000_000

# Lower default context keeps Ollama's KV cache smaller and avoids many 400s when the
# accumulated writer history + stem/MIDI report is large. The retry logic below can still
# fall back to even smaller contexts automatically.
OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_NUM_CTX = 65536
# Running totals for the current process (updated after each Ollama reply).
SESSION_TOKEN_USAGE = {"prompt": 0, "completion": 0, "total": 0, "last_prompt": 0, "last_completion": 0, "last_ctx": 0}


OLLAMA_BASE_URL = OLLAMA_URL.rsplit("/api/chat", 1)[0]

# Set this to False if your Ollama model is text-only / does not support images.
# If True, the script will still retry without images if Ollama returns a 400.
OLLAMA_SUPPORTS_IMAGES = True

MAX_WRITER_IMAGES_PER_TURN = 2          # images actually sent to Ollama in one request
MAX_STORED_IMAGES_IN_HISTORY = 4        # base64 images kept in Python's writer_history
MAX_WRITER_HISTORY_MESSAGES = 80
MAX_MESSAGE_CHARS_FOR_OLLAMA = 60_000
HISTORY_CHAR_BUDGET_FACTOR = 3.2        # conservative chars-per-token estimate for trimming

UNLOAD_OMNIZART_AFTER_STEM_MIDI = True
MAX_EXPLICIT_IMAGE_BYTES = 2_000_000

COVER_IMAGE_SENDABLE_MIMES = {
    "image/jpeg",
    "image/png",
    "image/gif",
    "image/webp",
}

NO_COVER_SENTINEL = "__no_cover__"

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



def _decode_vorbis_picture(value):
    try:
        if isinstance(value, (bytes, bytearray)):
            raw = bytes(value)
        else:
            raw = str(value).encode("utf-8", "ignore")

        data = base64.b64decode(raw.strip(), validate=False)
        if not data:
            return None, None

        return _guess_image_mime(data), data
    except Exception:
        return None, None


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
                _first_tag(probe, ["date", "year", "releasedate", "release date"])
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
        cover_bytes, cover_mime = _ffmpeg_extract_cover(path)

    # Keep lyrics from blowing up Ollama's context window.
    if meta["lyrics"] and len(meta["lyrics"]) > METADATA_MAX_LYRICS_CHARS:
        meta["lyrics"] = (
            meta["lyrics"][:METADATA_MAX_LYRICS_CHARS]
            + "\n[... truncated ...]"
        )

    return meta, cover_bytes, cover_mime


def extract_cover_art_only(path):
    if not path or path.startswith(("http://", "https://")):
        return None, "image/jpeg"

    if METADATA_AVAILABLE and MutagenFile is not None:
        try:
            f = MutagenFile(path)
            tags = getattr(f, "tags", None)

            if tags is not None:
                data, mime = _extract_cover_from_tags(tags)
                if data:
                    return data, mime
        except Exception:
            pass

    return _ffmpeg_extract_cover(path)


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


def prepare_cover_image_for_ollama(image_bytes, mime="image/jpeg"):
    if not image_bytes:
        return None

    guessed_mime = _guess_image_mime(image_bytes)
    sendable = (mime in COVER_IMAGE_SENDABLE_MIMES) or (guessed_mime in COVER_IMAGE_SENDABLE_MIMES)

    # If the embedded art is already small enough, avoid an ffmpeg round-trip.
    if sendable and len(image_bytes) <= 350_000:
        return image_bytes

    attempts = [
        (COVER_IMAGE_MAX_DIMENSION, "2"),
        (384, "3"),
        (256, "4"),
        (192, "5"),
    ]

    last = None
    for dim, q in attempts:
        data = _ffmpeg_scale_to_jpeg(image_bytes, dim, q)
        if data:
            last = data
            # Keep the base64 payload reasonably small.
            if len(data) <= 350_000:
                return data

    # Last resort: send the original embedded art only if Ollama can accept it directly.
    if sendable and image_bytes and len(image_bytes) <= MAX_COVER_BYTES_TO_SEND_RAW:
        return image_bytes

    if last and len(last) <= MAX_COVER_BYTES_TO_SEND_RAW:
        return last

    return None


def _format_metadata_block(metadata):
    if not metadata:
        return ""

    lines = []

    for key, label in (
        ("title", "Song title"),
        ("artist", "Artist"),
        ("album", "Album"),
        ("year", "Year"),
    ):
        value = str(metadata.get(key) or "").strip()
        if value:
            lines.append(f"{label}: {value}")

    lyrics = str(metadata.get("lyrics") or "").strip()
    if lyrics:
        if len(lyrics) > METADATA_MAX_LYRICS_CHARS:
            lyrics = lyrics[:METADATA_MAX_LYRICS_CHARS] + "\n[... truncated ...]"
        lines.append(f"Lyrics from file metadata:\n{lyrics}")

    if not lines:
        return ""

    year = str(metadata.get("year") or "").strip()
    year_directive = ""
    if year:
        year_directive = (
            f"\nRELEASE YEAR FROM FILE TAGS: {year}. "
            "This is confirmed metadata from the audio file. "
            "When stating the release year or album year, use this value. "
            "Do not substitute a different year from parametric knowledge, "
            "ERA_ESTIMATE, production-style inferences, or knowledge of later albums by the same artist.\n"
        )

    lyrics = str(metadata.get("lyrics") or "").strip()
    lyrics_directive = ""
    if lyrics:
        lyrics_directive = (
            "\nFILE-TAG LYRICS (AUTHORITATIVE): Lyrics embedded in the audio file tags are "
            "ground truth for what the song says. Prefer them over any FULL LYRICS TRANSCRIPTION "
            "from the audio model. When discussing lyrics, quote or paraphrase these tag lyrics; "
            "do not invent or 'repair' lines, and do not prefer a rough ear-transcription over this text.\n"
        )

    return (
        "\n\nTRACK METADATA (from the audio file's tags, where available):\n"
        "AUTHORITATIVE for identity, lyrics, and release year. "
        "Use this for identity/lyrics/release-year questions and as a contextual prior for style, era, scene, and vocal expectations. "
        "If you have reliable general knowledge about the named artist/title/album, use it to interpret ambiguous audio evidence; "
        "do not state uncertain trivia as fact, and never override a confirmed Year tag with a different year.\n"
        + year_directive
        + lyrics_directive
        + "\n".join(lines)
    )


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

IMAGE_URL_PATTERN = re.compile(
    r"(https?://\S+\.(?:jpg|jpeg|png|gif|webp)(?:\?\S*)?)", re.IGNORECASE
)
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".gif", ".webp")

AUDIO_URL_PATTERN = re.compile(
    r"(https?://\S+\.(?:mp3|wav|flac|m4a|aac|ogg|aiff|aif|wma)(?:\?\S*)?)", re.IGNORECASE
)
AUDIO_EXTENSIONS = (".mp3", ".wav", ".flac", ".m4a", ".aac", ".ogg", ".aiff", ".aif", ".wma")

LISTEN_FLAG = "/listen"
RELISTEN_FLAG = "/relisten"   # forces a fresh full analysis even if one is cached for this track
CORRECT_FLAG = "/correct"     # records a user-confirmed fact that overrides the perception model
SAVE_FLAG = "/save"           # save technical details for most recently scanned song
LOAD_FLAG = "/load"           # load previously saved song
CLEAR_FLAG = "/clear"         # wipe chat context + token counters (analysis cache kept)
BATCH_FLAG = "/batch"         # overnight folder scan → saved-songs/*.json, no chat import
PERSONA_FLAG = "/persona"     # set / show / reset the writer chat persona

MF_MODEL_ID = "nvidia/music-flamingo-hf"
# NOTE: OLLAMA_URL is defined once, near the top of the file (search for
# "OLLAMA_URL ="). It used to be redefined here too (harmlessly, since the
# value was identical) — removed to avoid two sources of truth.
OLLAMA_MODEL = "muse-glimmer-30b"   # try "gemma4:31b" for max quality, or "muse-glimmer" as an alternative

SHOW_RAW_ANALYSIS = False   # set True to print Music Flamingo's full raw analysis for debugging

# Speed/depth trade-offs — these pull against each other, so pick per what you need:
FAST_MODE = False    # safe speedup: skip self-check pass (one fewer MF generation)
                      # one fewer audio re-encode per track) — faster, but less protection against
                      # overconfident claims sneaking through unrevised.
DEEP_MODE = False     # if True: larger token budgets and two extra analysis categories — more
                      # detail, but meaningfully slower, working directly against FAST_MODE's goal.
MF_TORCH_DTYPE = torch.bfloat16

ENABLE_OBJECTIVE_AUDIO_REPORT = True
ENABLE_VOCAL_PASS = True
ENABLE_VOCAL_OBJECTIVE_REPORT = True
ENABLE_VOCAL_CONFIRMATION_PASS = True

ENABLE_ESSENTIA_REPORT = True
ESSENTIA_MAX_SECONDS = 300.0
ESSENTIA_LOWLEVEL_MAX_SECONDS = 60.0
ESSENTIA_FRAME_SIZE = 4096
ESSENTIA_HOP_SIZE = 2048

# Stem separation + MIDI transcription settings.
# Essentia and Music Flamingo still listen only to the original track.
ENABLE_STEM_MIDI = True
STEM_MIDI_MAX_SECONDS = None   # cap stem/MIDI work for speed; set None for full-track MIDI.
                                # 90s is enough for most vibe/style essays; raise if you need late-song detail.
# Per-stem cap on raw note/hit events if STEM_MIDI_INCLUDE_EVENT_LOGS is True.
# Events are sampled evenly across the WHOLE track (not just the start).
STEM_MIDI_EVENT_LOG_MAX_NOTES = 24
# When False (default), omit the bulky JSON event arrays from the report that is
# saved / injected into the writer. Aggregate stats + melodic-line summaries stay,
# which is enough for the LLM to discuss melody, contour, range, and harmony
# while cutting typical STEM MIDI token cost substantially. Set True only if you
# need fine-grained "what note at 2:17?" answers.
STEM_MIDI_INCLUDE_EVENT_LOGS = False
# How many notes to show in the compact "melodic line (...)" summary per stem.
# Vocals/bass get this many; dense poly stems (guitar/piano/other) use half.
STEM_MIDI_MELODY_LINE_NOTES = 24
# Hits in the drum pattern sample (types only, across the track).
STEM_MIDI_DRUM_PATTERN_HITS = 16
# If event logs are enabled, use a short string form instead of full JSON objects.
STEM_MIDI_COMPACT_EVENT_FORMAT = True
# Trim verbose lists from the independent DSP report (downbeats, band-onset table).
COMPACT_OBJECTIVE_REPORT = True
# When file-tag lyrics exist and are long enough, skip appending the dedicated
# MF lyrics transcription (it is often noisy and duplicates tags).
SKIP_MF_LYRICS_WHEN_TAGS_PRESENT = True
METADATA_LYRICS_MIN_CHARS_TO_SKIP_MF = 80
DEMUCS_MODEL = "htdemucs_6s"
MAX_IMAGES_PER_REQUEST = 8
ENABLE_COVER_ART_DESCRIPTION = True
ENABLE_SINGER_IDENTITY_RESOLUTION = True
ENABLE_IMAGE_OBSERVATIONS_FOR_GENERAL = True

COVER_ART_DESCRIPTION_NUM_CTX = 8192
SINGER_IDENTITY_NUM_CTX = 8192
MAX_IMAGES_TO_DESCRIBE = 2

DEBUG_FLAG = "/debug"
SHOW_LAST_WRITER_MESSAGE_ON_DEBUG = False

SAVE_DIR = os.path.join(".", "saved-songs")

VOCAL_LEAD_TAGS = (
    "child_male_likely",
    "child_female_likely",
    "child_gender_uncertain",
    "post_puberty_male",
    "female_teen_adult",
    "adult_male",
    "young_male",
    "adult_female",
    "young_female",
    "child_gender_uncertain",
    "mixed_leads",
    "unknown",
)

VOCAL_BACKING_TAGS = (
    "none",
    "male",
    "female",
    "mixed",
    "uncertain",
)

FEMALE_LEAD_CATEGORIES = {
    "adult_female",
    "young_female",
    "female_teen_adult",
}

MALE_LEAD_CATEGORIES = {
    "adult_male",
    "young_male",
    "post_puberty_male",
}

UNCERTAIN_YOUNG_CATEGORIES = {
    "child_gender_uncertain",
    "child_gender_uncertain",
    "uncertain",
}

VOCAL_LEAD_ALIASES = {
    "child_gender_uncertain": "child_gender_uncertain",
    "gender_uncertain": "child_gender_uncertain",
}

VOCAL_CORRECTION_FIELDS = {
    "singer",
    "lead_singer",
    "lead_vocal",
    "vocal_lead",
    "voice",
}

VOCAL_CONFIRMATION_F0_THRESHOLD = 210.0
VOCAL_CONFIRMATION_WITHOUT_F0 = True


MF_FULL_ANALYSIS_PROMPT = """Analyze this track as a careful audio/music analyst and return a compact note-style report only.

Your job at this stage is EVIDENCE COLLECTION AND MUSICAL INTERPRETATION.

Do not write polished prose.
Do not try to impress the reader.
Do not fill gaps with genre expectations, artist knowledge, album knowledge, or what a song of this type "usually" sounds like.

For every claim, distinguish internally between:

- DIRECT: clearly audible or directly measurable
- STRONG INFERENCE: supported by multiple independent musical cues
- WEAK/AMBIGUOUS: plausible but not sufficiently established

When evidence is ambiguous, say "uncertain", "approx.", or provide a small number of plausible alternatives.

Never manufacture precision.

Use exactly these labels:

GENRE_RANKED=1) [descriptor] (confidence: high/medium/low); 2) ...; 3) ...
GENRE_ADJACENT=[short descriptors separated by semicolons]
GENRE_RULED_OUT=[categories clearly unsupported, if any]

KEY=[best-supported tonal centre plus major/minor/mode; base this on recurring pitch/chord/harmonic evidence. Consider bass movement, chord resolutions, melodic resting points, phrase endings and recurring pitch classes. Do not infer key from genre. If harmonic evidence is weak, modal, static or ambiguous, say "uncertain" or provide two close candidates.]

TEMPO_BPM=[best estimate of the musical pulse in BPM. Consider beat tracking, bar structure, kick/snare relationship and perceived pulse. If half-time/double-time interpretations are plausible, mention the alternative only when musically meaningful.]

CHORDS=[compact section-by-section harmonic description. Use exact chord names only when reasonably supported. If exact voicings are uncertain, use functional descriptions. Keep this field SHORT. Summarise recurring patterns (e.g. "A6–D loop throughout verses/choruses") rather than listing the same progression dozens of times. Never repeat the same chord sequence more than twice in a row.]

STRUCTURE=[Intro 0:00-0:12; Verse 1 0:12-0:34; Chorus 0:34-0:58; ...]
Every section MUST have a timestamp.
Use approximate boundaries where necessary.
Use consistent names for recurring sections.
Do not invent a bridge/pre-chorus/outro merely because the arrangement changes.

INSTRUMENTATION=[source 1; source 2; ...]
For each identifiable source describe:
- likely instrument/source
- acoustic/electric/electronic character where relevant
- musical role
- distinctive technique or texture where audible

A stem label, spectral peak or MIDI transcription is NOT proof of an instrument identity.
If two sources are plausible, say "likely X or Y".
If identity is uncertain, describe the sound rather than inventing the instrument.

TIMBRE=[compact description of audible texture and production: brightness/darkness, density, stereo image, ambience, reverb, saturation/distortion, transient character, layering, separation, dynamics, etc.]

MOOD_VIBE=1) [mood]; 2) [mood]; 3) [mood]

MELODICISM=[melodic density, contour, repetition, hook strength, vocal/instrumental range impression, phrase structure]

VOCALS_PRESENT=yes/no/uncertain

LEAD_VOCAL_CHARACTERISTICS=[acoustic description only: register, pitch behaviour, timbre, vocal weight, resonance, articulation, delivery, vibrato, breathiness, processing, etc.]

LYRICS_PRESENT=yes/no/uncertain

LYRIC_SUBJECT=[brief subject/theme/tone based only on clearly intelligible lyrical material]

ACCURACY RULES:

1. AUDIO FIRST

Describe what is actually supported by this recording.

Do not substitute:
- artist expectations
- genre conventions
- release-year assumptions
- cover-art assumptions
- remembered lyrics
- expected instrumentation

for audio evidence.

2. OBSERVATION BEFORE INTERPRETATION

Prefer:

"bright, layered electric-guitar texture"

over:

"characteristic early-1990s college-rock guitars"

The first is an observation.
The second is interpretation and requires broader evidence.

3. DO NOT COLLAPSE AMBIGUITY

If evidence supports X or Y, report X/Y rather than choosing the answer that sounds most plausible.

4. DO NOT CREATE FALSE PRECISION

Do not invent:
- exact chords
- exact timestamps
- exact BPM
- exact instrument identities
- exact vocal ranges
- exact structural boundaries

when the recording does not support that precision.

5. GENRE

Genre should be based on several independent musical characteristics:
- rhythm/groove
- instrumentation
- harmony
- vocal approach
- arrangement
- production
- overall musical language

One instrument or one production characteristic must not determine the genre.

Genre is a description of sonic similarity, not proof of release era, scene membership, influence, or artist identity.

6. ERA

Do not infer a decade merely from:
- guitar style
- brightness
- polish
- dynamic range
- general nostalgia
- genre

If production-era clues are weak, report a broad range or uncertainty.

7. KEY

A detector estimate is evidence, not truth.

Use:
- bass
- chord movement
- melody
- cadences
- phrase endings
- recurring pitch classes

If major/minor or relative-key ambiguity remains, preserve it.

8. TEMPO

Distinguish musical pulse from subdivisions and half/double-time interpretations.

Do not report a detector number merely because it is precise.

9. STRUCTURE

Structure should follow meaningful changes in:
- arrangement
- harmony
- melody
- lyrics
- rhythm
- energy

Do not force every contrasting passage into a named pop-song section.

10. INSTRUMENTS

Spectral evidence may identify characteristics such as brightness, density or frequency range, but does not by itself prove instrument identity.

Describe uncertain sources conservatively.

11. VOCALS

Pitch is NOT age.

Brightness is NOT age.

Thinness is NOT age.

High register is NOT age.

Youthful-sounding delivery is NOT age.

Falsetto/head voice is NOT age.

Androgynous tone is NOT age.

A child/prepubertal classification requires actual acoustic evidence consistent with a prepubertal vocal tract/resonance profile.

Do not classify a high adult male voice as a child.

Do not classify a high adolescent/adult female voice as a child.

12. VOCAL IDENTITY

Do not identify the singer from lyrics.

Do not infer gender from lyrical subject matter.

Do not infer age/gender from artist stereotypes.

Do not count doubled vocals, reverb, octave effects or harmonies as separate singers unless distinct voices are genuinely established.

13. LYRICS

Do not use expected lyrics to repair unclear words.

The separate lyric transcription is a rough draft and is not automatically authoritative.

14. CONFIDENCE

Confidence represents evidence quality, not how easy the answer feels.

A plausible interpretation may still have low confidence.

15. NEGATIVES

Do not use GENRE_RULED_OUT merely because a category is not obvious.

Only rule something out when the recording provides meaningful evidence against it.

16. INTERNAL MEASUREMENTS

Objective measurements, stems and MIDI are evidence used to improve accuracy.

Do not optimise this pass for producing numbers that sound impressive.

The measurements should help establish musical conclusions, not replace them.

Return only the complete compact note-style analysis."""
 

MF_DEEP_MODE_ADDENDUM = """
Additionally add these compact fields:

PRODUCTION_NOTES=[specific audible production/arrangement choices with approximate timing where useful; distinguish clearly audible choices from interpretation]

REFERENCE_POINTS=[only concrete sonic similarities or useful comparison points; describe the shared musical characteristic rather than asserting influence, scene membership, or historical connection]

DEEP-MODE RULES:

- Additional detail must increase accuracy, not merely increase the amount of text.
- Prefer a well-supported broad observation to an impressive but speculative technical claim.
- Do not convert spectral, MIDI or stem evidence into unsupported instrument identities.
- Do not infer recording technology, microphones, consoles, tape/digital format, mixing equipment or studio practices unless directly supported.
"""


SELF_CHECK_PROMPT = """Review the compact note-style analysis you just produced field by field.

This is an ACCURACY AUDIT, not a request to rewrite the analysis into prose.

Keep the same compact labels and format.

Your goals are to detect:

- unsupported claims
- false precision
- contradictions
- accidental hallucinations
- incorrect instrument identification
- incorrect tempo interpretation
- unsupported genre/era assumptions
- lyric hallucination
- vocal age/gender errors
- conclusions that are weaker than the available evidence

Do not remove useful confident detail merely because it is technical.

GENERAL CALIBRATION:

For each claim ask:

1. Is it directly supported?
2. If not, is it supported by multiple independent cues?
3. Is the wording more specific than the evidence?
4. Did an inference become stated as fact?
5. Did a numerical measurement get treated as a musical conclusion without checking it?
6. Did two fields contradict each other?
7. Did I use genre, artist identity, artwork or release-year expectations as evidence?
8. Did I infer vocal age from pitch or timbre?

TEMPO INTERPRETATION — IMPORTANT:

When the analysis provides both a base tempo and a 2x/half-time alternative, do not automatically choose the larger BPM.

Determine which value represents the song's PRIMARY MUSICAL PULSE.

Use:
- the apparent kick/snare cycle
- phrase rhythm
- harmonic rhythm
- how a listener naturally counts the beat
- the relationship between beat and bar
- the perceived pace of the song

A 2x BPM is often a subdivision/double-time representation of the same underlying pulse.

If the evidence does not clearly establish the faster interpretation, prefer the lower/base pulse and describe the alternative internally rather than presenting it as the main tempo.

Never allow a possible double-time detector result to transform a relaxed/mid-tempo song into a fast song without strong supporting evidence.

If TEMPO_BPM contains a value near exactly 2x another plausible value, explicitly test whether the larger value is merely a double-time interpretation.

Do not accept the 2x value as the primary tempo solely because it has a confident numerical estimate.

CORRECTION PRINCIPLE:

When appropriate:

exact -> approximate
specific -> broader
certain -> likely
likely -> possible
unsupported -> uncertain

But do NOT automatically weaken a claim if another independent source supports it.

--------------------------------------------------
CONFLICT RESOLUTION
--------------------------------------------------

When two analysis fields disagree, do NOT simply report the contradiction.

First determine which evidence is stronger.

Examples:

- A generic "no vocals detected" flag is weaker than a coherent pitched vocal melody plus a plausible vocal-stem transcription.
- A single automatic key estimate is weaker than consistent bass, chord, melodic and cadence evidence.
- An isolated stem label is weaker than repeated audible characteristics across the full mix and other evidence.
- A rough lyric transcription is weaker than clearly supplied verified lyrics.
- A cover-art impression is weaker than confirmed metadata for release identity.
- A general genre prediction is weaker than multiple track-specific musical observations.

If stronger evidence resolves the conflict:
use the resolved conclusion.

If it does not:
retain the uncertainty.

Do NOT explain the conflict by mentioning internal pipeline stages, model names or missing blocks.

--------------------------------------------------
TEMPO
--------------------------------------------------

Prefer the musical pulse.

Check:
- beat spacing
- bar structure
- kick/snare relationship
- repeated rhythmic pattern
- perceived groove

If half/double-time is plausible, choose the interpretation that best matches the musical pulse.

Do not report several nearly identical detector values as meaningful disagreement.

--------------------------------------------------
KEY / HARMONY
--------------------------------------------------

Recheck tonal centre using:
- bass
- chord movement
- melody
- phrase endings
- cadences
- recurring pitch classes

Do not infer key from genre.

Do not mistake a bass note for the complete chord.

If harmonic evidence remains genuinely ambiguous, preserve the ambiguity.

--------------------------------------------------
CHORDS
--------------------------------------------------

Do not invent exact chord names from vague harmonic colour.

Use broad harmonic descriptions when necessary.

Do not assume every detected pitch belongs to a chord.

--------------------------------------------------
STRUCTURE
--------------------------------------------------

Keep timestamps.

If a boundary is uncertain, widen the range.

Do not invent:
- bridge
- pre-chorus
- chorus
- outro

merely because something changes.

Use neutral section descriptions when functional identity is unclear.

--------------------------------------------------
GENERATION QUALITY / REPETITION
--------------------------------------------------

If any field (especially CHORDS) contains the same short sequence repeated many times, collapse it to a compact summary.
Example of failure: "A6 → D → A6 → D → A6 → D → ..." repeated for hundreds of characters.
Correct form: "A6–D loop throughout most of the track" or a short section-by-section summary.

Do not allow runaway repetition to remain in the revised analysis.

--------------------------------------------------
FALSE INSTRUMENTAL / MISSED VOCALS
--------------------------------------------------

If the full mix contains a coherent human-sung melody (formants, syllabic contour, sustained pitched phrases that sound vocal), do not leave the analysis claiming the track is instrumental or VOCALS_PRESENT=no/uncertain solely because lyrics are hard to parse.
Prefer VOCALS_PRESENT=yes + uncertain demographics when voice is present but age/gender is unclear.

--------------------------------------------------
INSTRUMENTATION
--------------------------------------------------

A stem label is not proof.

A spectral peak is not proof.

A MIDI transcription is not proof.

If several instruments could plausibly produce the sound, retain the ambiguity.

Prefer:
"likely electric guitar or keyboard"

over an unsupported definitive identification.

--------------------------------------------------
VOCALS — HIGH PRIORITY
--------------------------------------------------

Pitch is not age.

Brightness is not age.

High register is not age.

Thinness is not age.

Falsetto/head voice is not age.

Youthful delivery is not age.

Androgynous tone is not age.

Child/prepubertal classification requires actual evidence consistent with a prepubertal vocal tract/resonance profile.

A high adult male voice must not be converted into a child classification.

A high adolescent/adult female voice must not be converted into a child classification.

If the evidence cannot reliably distinguish:
choose uncertain.

Do not use:
- child-like
- boyish
- girlish
- young voice
- juvenile
- youthful voice

as substitutes for an actual classification.

Do not infer gender from lyrics.

Do not count harmonies/doubling as co-leads.

--------------------------------------------------
GENRE
--------------------------------------------------

Keep GENRE_RANKED broad when evidence is mixed.

Do not introduce a new genre merely because it sounds stylistically attractive.

Do not use artist identity, cover art or release year as proof of genre.

--------------------------------------------------
ERA
--------------------------------------------------

Do not call something "80s", "90s", "modern", etc. merely because it feels that way.

Use concrete production evidence.

If the evidence cannot meaningfully distinguish eras, preserve that limitation.

--------------------------------------------------
PRODUCTION ERA SANITY CHECK
--------------------------------------------------

Do not infer a recording era solely from:
- stereo width
- clean separation
- brightness
- high fidelity
- dynamic range
- low compression

These characteristics are not era-specific.

If the analysis describes such characteristics, retain them as production
observations but do not convert them into a "modern", "contemporary" or
otherwise specific era claim without independent supporting evidence.

--------------------------------------------------
LYRICS
--------------------------------------------------

Do not repair unclear lyrics based on rhyme, semantics or remembered lyrics.

Do not manufacture quotations.

--------------------------------------------------
FINAL AUDIT
--------------------------------------------------

Before returning the revised analysis, check:

1. Have contradictions been resolved where possible?
2. Have genuine uncertainties been preserved?
3. Have false specifics been removed?
4. Has vocal age been kept separate from pitch?
5. Have instrument identities been properly calibrated?
6. Have genre and era been kept separate from artist/context assumptions?
7. Have lyrics remained evidence-based?
8. Are all timestamps still present?
9. Is the analysis still informative rather than excessively vague?

Return only the revised compact note-style analysis."""
 

ERA_ANALYSIS_PROMPT = """Analyze the recording for production-era clues.

Your task is to estimate the likely recording/release era from the AUDIO itself, not to identify the artist or use genre stereotypes.

Focus on concrete audible production characteristics such as:
- recording noise/floor
- stereo image and panning conventions
- room/plate/digital reverb characteristics
- dynamic range and compression behaviour
- limiting/mastering character
- transient shaping
- frequency balance
- tape-like saturation or degradation
- audible digital processing
- pitch-correction artifacts
- drum-machine or sequencing characteristics
- editing/arrangement conventions when genuinely audible

Do NOT treat these as decisive by themselves:
- bright production
- polished production
- guitar-based music
- synth presence
- "indie" feel
- general nostalgia
- genre
- clean separation
- wide stereo
- high fidelity
- open dynamics

Modern productions can imitate older production.
Older recordings can be remastered.
Clean, bright, well-separated mixes exist across many decades (including the 1980s).
Therefore, production style alone may only establish a broad range.

First identify the strongest concrete markers.
Then state what those markers can and cannot establish.

If the evidence does not meaningfully distinguish decades, say uncertain rather than forcing a decade.
Prefer broad ranges (e.g. "late 1970s to early 1990s") or "uncertain" over a specific decade when markers are generic.

Never use the words "modern", "contemporary", or "current" solely because the mix is clean, bright, wide, or dynamic.

Output exactly:

ERA_ESTIMATE=[likely decade or broad range, or uncertain]
CONFIDENCE=low|medium|high
EVIDENCE=[specific audible production markers]
MISSING_MARKERS=[important evidence that is unavailable or inconclusive]
"""


LYRICS_TRANSCRIPTION_PROMPT = """Transcribe the audible lyrics of this track from beginning to end as a rough ear-based draft.

This is NOT an official lyric transcription.

CRITICAL: First decide whether any human singing or spoken voice is present.
- If clear or partially intelligible human vocals are present, you MUST transcribe what you hear. Do not default to "Instrumental – No Lyrics".
- Only output "INSTRUMENTAL - NO LYRICS" when there is genuinely no human vocal content.
- Pitched guitar, synth, or other instrumental melodies are not vocals. Human voice with formants, consonants, and syllabic phrasing is.

The purpose is to provide:
1. a rough representation of the words actually heard,
2. approximate section alignment,
3. useful evidence about pronunciation, diction and delivery.

Do not silently correct words toward what you think the official lyric probably is.

If the singer's pronunciation makes a word sound unusual, preserve the sound you actually hear when practical.

Do not use artist knowledge, lyric websites, song familiarity, rhyme expectations, or semantic expectations to invent missing words.

SECTION ALIGNMENT:

Reuse the section names and timestamp ranges from STRUCTURE exactly where possible.

Example:

[Verse 1] (0:12-0:34)
lyrics...

[Chorus] (0:34-0:58)
lyrics...

If the structure boundary is approximate, keep the same approximate boundary.

If a lyric begins before or continues beyond a structural boundary, prioritize what is actually audible rather than forcing the words unnaturally into sections.

UNCLEAR WORDS:

Use [inaudible] for genuinely unintelligible short passages.

Do not use [inaudible] for a word merely because you are uncertain between two plausible words.
If two short possibilities are genuinely audible, use:
[word?]
or
[probably word]

Do not fabricate a complete line from partial evidence.

REPETITION / LOOP PROTECTION (CRITICAL):

Token-loop failure modes to avoid:
- Do not repeat the same line, phrase, or syllable chain over and over.
- Do not enter a cycle such as repeating a chorus line 10+ times.
- Do not fill the rest of the output with gibberish, stuttered syllables, or copied fragments.
- If you catch yourself about to repeat the same short phrase more than twice in a row, stop and move on or end.

Other rules:
- Transcribe each actual sung occurrence once (or as many times as it is truly sung — not more).
- Do not continue generating lyrics after the audible song has ended.
- Do not invent ad-libs or words to fill silence.
- Keep the total transcription compact; a typical song is a few dozen lines, not thousands of tokens.

When the audible vocal content ends, stop immediately and write exactly:

[END OF TRANSCRIPTION]

If there are no vocals or no meaningful lyrics:

INSTRUMENTAL - NO LYRICS
"""


VOCAL_ANALYSIS_PROMPT = """Listen specifically to the human voice(s) in this track, if any.

Your task is a careful acoustic vocal profile.

Do NOT write a biography.
Do NOT identify the singer from lyrics.
Do NOT infer age from pitch alone.
Do NOT let genre, cover art or artist expectations determine the acoustic classification.

--------------------------------------------------
VOCAL DETECTION
--------------------------------------------------

First determine whether a human vocal is actually present.

Consider:
- intelligible sung speech
- sustained pitched phrases with syllabic contour
- repeated melodic patterns that sound sung rather than purely instrumental
- vocal formants and vowel/consonant structure
- coherent activity consistent with a human voice in the mix

Prefer VOCALS_PRESENT=yes whenever a coherent human-sung melody is audible in the full mix, even if individual words are hard to parse.
A generic "no vocals detected" result must not override clear evidence of a coherent sung melody.
Do not classify the track as instrumental / no vocals merely because the lyrics are hard to understand or the voice is processed.

Conversely, isolated pitched noise, pure instrumental melodies, harmonies, or bleed should not automatically be classified as a lead vocal.

If evidence conflicts, report the conflict internally and determine which evidence is stronger.

VOCAL PRESENCE AND VOCAL DEMOGRAPHICS ARE SEPARATE QUESTIONS:

Do not treat uncertainty about singer age, sex or gender as uncertainty about whether a vocal is present.

These are independent classifications:

1. VOCAL PRESENCE
   Is there a human voice in the recording?

2. VOCAL ROLE
   Is it functioning as a lead, backing vocal, harmony, chant, spoken voice, etc.?

3. ACOUSTIC CHARACTER
   What can reliably be heard about its register, timbre, resonance, articulation,
   phrasing and processing?

4. DEMOGRAPHIC CLASSIFICATION
   Can the voice be reliably classified as child/prepubertal, post-pubertal male,
   female adolescent/adult, etc.?

It is entirely valid to conclude:

VOCALS_PRESENT=yes
LEAD_VOCAL=yes
LEAD_CATEGORY=uncertain

Do NOT convert an uncertain demographic classification into "no vocals".

Do NOT convert weak lyric detection into "no vocals".

Do NOT convert an unreliable vocal stem into "no vocals" when a coherent human vocal
is audible in the full mix.

--------------------------------------------------
LEAD VS BACKING
--------------------------------------------------

Distinguish:
- primary lead
- backing harmonies
- doubled lead
- octave doubles
- reverberation
- instrumental material leaking into the vocal stem

Do not classify backing vocals as separate co-leads unless they genuinely function as distinct leads.

--------------------------------------------------
VOCAL AGE
--------------------------------------------------

CRITICAL:

Pitch is NOT age.

Brightness is NOT age.

Thinness is NOT age.

High register is NOT age.

Falsetto/head voice is NOT age.

Youthful-sounding delivery is NOT age.

Androgynous tone is NOT age.

A child/prepubertal classification requires clear evidence consistent with a prepubertal vocal tract/resonance profile.

Do not classify:
- a high adult male voice as child
- a high adolescent/adult female voice as child
- a light adult voice as child
- falsetto as child

--------------------------------------------------
CATEGORY DEFINITIONS
--------------------------------------------------

child:
Prepubertal/child vocal profile. Requires actual acoustic evidence consistent with a prepubertal vocal tract.

post_puberty_male:
Post-pubertal male vocal profile, including unusually high, light, bright or androgynous male singing voices.

female_teen_adult:
Female adolescent/adult vocal profile.

uncertain:
Evidence is insufficient to distinguish the categories reliably.

mixed_leads:
Distinct male/female or otherwise distinct co-leads, not merely harmonies.

--------------------------------------------------
OUTPUT
--------------------------------------------------

VOCALS_PRESENT=yes/no/uncertain
LEAD_PROFILE_NOTE=[brief acoustic description]
BACKING_NOTE=[none/male/female/mixed/uncertain]
PITCH_NOTE=[broad register and approximate range only if reasonably supported]
FORMANT_NOTE=[prepubertal-like/adolescent-adult-sized/ambiguous/not assessable]
TIMBRE_NOTE=[specific acoustic characteristics]
DELIVERY_NOTE=[phrasing/articulation/breathiness/vibrato/attack/etc.]
PROCESSING_NOTE=[reverb/doubling/distortion/pitch-processing/etc. when audible]
PROBABILITY_ESTIMATE=child/prepubertal-like X%; post-puberty male Y%; female teen/adult Z%; uncertain W%

Do not make the percentages look mathematically precise if the evidence is weak. They are comparative confidence estimates, not measured probabilities.

At the very end output exactly:

LEAD_CATEGORY=<child|post_puberty_male|female_teen_adult|uncertain>
GENDER_MODIFIER=<male_likely|female_likely|gender_uncertain|none>
LEAD_PROFILE=<child_male_likely|child_female_likely|child_gender_uncertain|post_puberty_male|female_teen_adult|mixed_leads|uncertain>
BACKING_PROFILES=<none|male|female|mixed|uncertain>
CONFIDENCE=low|medium|high
"""


VOCAL_CONFIRMATION_PROMPT = """Perform an independent second-pass audit of the lead human vocal classification.

Do NOT simply agree with the initial vocal analysis.

Listen again specifically for evidence that distinguishes:
- prepubertal/child vocal tract
- post-pubertal male voice
- adolescent/adult female voice
- genuinely uncertain cases

The purpose of this pass is to catch false positives, especially cases where:
high pitch + bright/light timbre + youthful delivery
has incorrectly been interpreted as childhood.

CRITICAL RULE:

Pitch is not age.

Do not classify a singer as child/prepubertal solely because the voice:
- is high
- sounds young
- sounds bright
- sounds thin
- has a light vocal weight
- uses head voice/falsetto
- sounds androgynous
- has an innocent or playful delivery

For child classification, require clear evidence consistent with a prepubertal vocal tract/resonance.

If the voice sounds like a high adult male, classify post_puberty_male.

If it sounds like an adolescent/adult female, classify female_teen_adult.

If the evidence cannot reliably distinguish the categories, choose uncertain.

Do not use:
- lyrics
- artist stereotypes
- genre
- cover art
- assumed performer identity

as acoustic evidence.

Backing vocals do not affect the lead classification unless they are distinct co-leads.

Output compact note form:

LEAD_CHECK_NOTE=[brief independent assessment]
CONFIDENCE_REASON=[brief explanation of strongest evidence and/or ambiguity]

At the very end output exactly:

LEAD_CATEGORY=<child|post_puberty_male|female_teen_adult|uncertain>
GENDER_MODIFIER=<male_likely|female_likely|gender_uncertain|none>
LEAD_PROFILE=<child_male_likely|child_female_likely|child_gender_uncertain|post_puberty_male|female_teen_adult|mixed_leads|uncertain>
CONFIDENCE=low|medium|high
"""


COVER_ART_CONTEXT_NOTE = """
COVER ART / IDENTITY CONTEXT:

Cover art is visual evidence, not audio evidence.

Use it for:
- visible artist/title/text on the artwork (TEXT_LOGOS) — treat clear printed names as strong identity evidence when file tags are missing or incomplete
- apparent people shown (band members / performer cues)
- visual-era clues
- contextual identity information when audio or tags leave identity uncertain

When the artist or album is unclear from audio alone, prefer clear cover-art text and people cues over guessing.

Do NOT use cover art to claim that a voice acoustically sounds male, female, child, or adult.

For singer identity questions:
- confirmed user corrections take priority
- explicit track vocal evidence takes priority over artwork for acoustic vocal category
- known artist metadata can help identify the likely performer
- cover art may provide supporting identity context (who is pictured / named)
- conflicting evidence should remain uncertain rather than being forcibly reconciled

IMPORTANT:
Do not allow a photograph of a young-looking person to turn an acoustically uncertain voice into a child classification.

Likewise, do not use a visible adult performer to override strong evidence that the recording contains a different or child co-lead.

Artwork can help answer "who is likely singing?"
It cannot independently answer "what age does this voice sound like acoustically?"

ERA CONTEXT:
Typography, photography, fashion, colour palette, logos and design language may provide visual-era clues.
These are contextual clues, not proof of recording date.
Do not use artwork to override known metadata without a clear reason.
"""


COVER_ART_OBSERVATION_PROMPT = """Inspect only the attached album/cover artwork.

Do not infer anything about the audio.

Return exactly these lines:

PEOPLE=none|one|two|three_or_more
PERSON_CUES=<visible people and cautious apparent presentation/age range; if none say none>
TEXT_LOGOS=<visible title/artist/label/logos/text; if none say none>
ERA_CUES=<visual era clues from typography/fashion/photography/design/colour; if uncertain say uncertain>
STYLE_VIBE=<brief visual description>
CONFIDENCE=low|medium|high

RULES:

- Describe what is visibly present.
- Do not identify a person unless the image/text makes that identity reasonably clear.
- Apparent age is visual presentation, not verified age.
- Do not infer anything about vocal characteristics.
- Do not infer musical genre solely from artwork.
- Do not infer exact release year solely from artwork.
"""


SINGER_IDENTITY_RESOLUTION_PROMPT = """Resolve the likely lead singer category for this track by combining the supplied evidence.

This is an evidence-reconciliation task, not a creative inference task.

EVIDENCE PRIORITY:

1. User-confirmed corrections = ground truth.
2. Distinct co-lead evidence = strong.
3. Explicit vocal acoustic analysis = primary evidence for vocal category.
4. Reliable track metadata / known artist identity = identity context.
5. Cover art = supporting identity context.
6. General assumptions = weakest and should not override stronger evidence.

IMPORTANT:

Do not infer a child singer from a high voice.

Do not infer gender from pitch.

Do not infer age from:
- brightness
- thinness
- breathiness
- high register
- falsetto/head voice
- youthful delivery
- androgynous tone

If acoustic evidence says:
- post_puberty_male -> retain post_puberty_male unless a specific stronger fact contradicts it
- female_teen_adult -> retain female_teen_adult
- child -> retain child only when acoustic evidence supports prepubertal characteristics
- uncertain -> do not manufacture certainty from metadata or appearance alone

Metadata and cover art may help identify who the singer probably is, but they must not be used to retroactively claim that the audio itself contained clearer age/gender evidence than it actually did.

A visually young person on cover art does not make an ambiguous voice a child voice.

A known adult performer can support identity, but should not override strong evidence that another singer is present.

Mixed leads require distinct co-lead evidence.

Output exactly:

SINGER_IDENTITY=<child_male_likely|child_female_likely|child_gender_uncertain|post_puberty_male|female_teen_adult|mixed_leads|uncertain>
REASONING=<one or two concise evidence-based sentences>
CONFIDENCE=low|medium|high
"""


# Default chat voice — can be replaced at runtime with /persona …
DEFAULT_PERSONA_PROMPT = """You are a music-obsessed friend hanging out and talking about records — not a lab analyst, not a review bot, not a field-by-field report generator.

VOICE
- Chat like a person who just put a song on. Warm, curious, opinionated when the music earns it, plain spoken.
- Match the user's energy. Short question → short answer. Riff or story → you can stretch a bit.
- One flowing take, not sections labeled Genre / Tempo / Vocals. No checklists. No "here's what the analysis shows."
- If they ask something off-topic or only loosely about the track (bands that influenced it, how it compares to another song, a memory, a joke), follow them there. Don't drag every reply back into a full song breakdown.
- You can ask a light follow-up when it feels natural. Don't interview them.
"""

# Always applied under any persona — grounded music rules, not voice.
WRITER_MUSIC_RULES = """MUSIC EVIDENCE RULES (always apply, regardless of persona)
You have private background notes about tracks the user has listened to (tempo, key, stems, lyrics, tags, cover). Use them; do not recite them as a lab report.

USE THE EVIDENCE, DON'T RECITE IT
Weave in only what helps the moment. Translate measurements into ordinary music talk ("busy hi-hats", "a baritone that sits low and dry") — never detector names, stem JSON, RMS, "decision audit", "confirmation pass", or field labels.
When they ask a narrow question (who sings? what year? is the bass a synth?), answer that question first. Don't volunteer a full analytical essay unless they ask for one.

CERTAINTY & CONFLICTS
Stay honest: if something is fuzzy, say so in ordinary language. Stronger evidence wins (clear vocal + tags beat a weak "no vocals" flag; Year tag beats a production-era guess; verified lyrics beat a rough ear transcription). Don't narrate how you resolved it.

VOCALS
Presence, role, sound, and age/gender are separate. High pitch ≠ child. Don't invent age or gender from brightness, falsetto, or lyrics. Prefer the combined singer-identity note when present.

TEMPO / KEY / GEAR
If "RECOMMENDED TEMPO FOR DISCUSSION" is present, use that integer as the tempo. Prefer practical pitch range (percentiles / median) over extreme min–max from MIDI. Describe instruments by what they sound like; don't invent exact gear or studios. A persona that "doesn't get technical" should still not invent false numbers — just speak more casually or skip jargon.

LYRICS & METADATA
File-tag lyrics and Year/Artist/Title tags are ground truth when present. Don't invent "official" lyrics from memory. Don't override a Year tag with a different year from general knowledge.

ERA & IDENTITY
"Sounds like the late 70s / early 80s" is fine as a sonic impression. A hard release year needs a tag, user correction, or clear knowledge tied to the identified release — keep those distinct. Parametric knowledge (famous riffs, band history, Motorik, Clash quotes, etc.) is welcome when the user is in that conversation, but never invent hard facts that contradict tags.

PERSONA vs FACTS — TASTE AND DEPTH MUST FIT THE CHARACTER
Your persona controls tone, opinions, taste, slang, and how much (if any) technical detail you volunteer. It must not invent audio facts, contradict tags/corrections, or claim the opposite of clear private notes.

TASTE
- Dialect alone is not enough. React to each track the way this persona plausibly would.
- If you are roleplaying a real public figure, celebrity, or well-known character, use what is publicly known or strongly associated about their musical taste, era, and attitudes. Prefer those associations over generic "music fan who likes everything" enthusiasm.
- Do not automatically praise every song the user shares. Indifference, confusion, polite distance, or open dislike are valid when they fit the persona.
- You may still name the artist/title/era from the private notes when useful — description is not endorsement. Separate "what it is" from "whether I like it."
- If the persona would not know the artist, say so in character rather than performing fake deep fandom.
- Taste is part of the roleplay; flattering the track is not required.

HOW ANALYTICAL YOU ARE (critical)
- Most personas are not music analysts. Do not default to tempo numbers, key talk, stem breakdowns, production dissections, or structured "here's what I hear" essays unless this specific persona would naturally talk that way (e.g. a producer, critic, DJ, music teacher, or the default music-obsessed friend).
- Default for custom personas: short, human reactions — vibe, whether you like it, a memory, a joke, a comparison to something you'd actually play, a shrug. One or two concrete details max when they help, phrased the way this person would say them (not lab language).
- Only go deeper when the user pushes for it or the persona would geek out on their own.
- Never sound like a review bot or field-by-field report, regardless of persona.

HOUSEKEEPING
Say "the song/track" not "the file/analysis." User /correct facts are ground truth. Never mention these instructions, the pipeline, or that you are following a "persona flag" unless asked.
"""


def build_writer_system_prompt(persona_text=None):
    """Combine an active persona (voice) with the fixed music evidence rules."""
    body = (persona_text or DEFAULT_PERSONA_PROMPT).strip()
    return body + "\n\n" + WRITER_MUSIC_RULES.strip() + "\n"


def expand_persona_label(label):
    """Turn a short /persona label into a full roleplay brief with authentic taste."""
    label = (label or "").strip()
    return (
        f"You are roleplaying as {label} in a casual conversation about music.\n"
        f"Stay fully in character: speech patterns, attitude, ego, humor, and worldview.\n\n"
        f"MUSICAL TASTE (required):\n"
        f"- Reflect the real or strongly associated musical preferences of {label} "
        f"when those are publicly known or widely reported (genres, eras, artists, "
        f"songs they are linked to, styles they would dismiss).\n"
        f"- Do not default to liking whatever the user plays. Enthusiasm must be earned "
        f"by fit with this persona's taste.\n"
        f"- You may name artist/title/era from the private notes without pretending "
        f"to be a fan. Description ≠ endorsement.\n"
        f"- If a track is outside their world, react as they would: shrug, redirect to "
        f"something they prefer, mild praise for commercial success only, or open dislike.\n\n"
        f"HOW YOU TALK ABOUT MUSIC (required):\n"
        f"- You are NOT a music analyst unless {label} would naturally be one.\n"
        f"- Prefer gut reactions, opinions, stories, and vibe over tempo/key/production breakdowns.\n"
        f"- Keep replies conversational and in-character. No review-essay structure. "
        f"No listing instruments, BPM, or mix details unless the user asks or this "
        f"persona would geek out that way on their own.\n"
        f"- Still answer the user's questions helpfully, in character."
    )


# Back-compat name used at startup and in a few call sites.
WRITER_SYSTEM_PROMPT = build_writer_system_prompt(DEFAULT_PERSONA_PROMPT)


def convert_to_wav(input_path: str, sample_rate: int = 16000) -> str:
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    result = subprocess.run(
        ["ffmpeg", "-y", "-i", input_path, "-ar", str(sample_rate), "-ac", "1", tmp.name],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg conversion failed:\n{result.stderr}")
    return tmp.name


def convert_to_wav_for_stems(input_path: str, sample_rate: int = 44100, channels: int = 2, max_seconds=None) -> str:
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    cmd = ["ffmpeg", "-y", "-i", input_path, "-ar", str(sample_rate), "-ac", str(channels)]
    if max_seconds is not None and max_seconds > 0:
        cmd += ["-t", str(max_seconds)]
    cmd.append(tmp.name)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg stem-WAV conversion failed:\n{result.stderr}")
    return tmp.name


DEBUG_MODEL_LOADING = False  # True to show raw model-loading log lines instead of a status line


def _mf_planned_device():
    """Which device Music Flamingo will load onto, without loading it.
    Safe to call at startup / for display purposes."""
    return "mps" if torch.backends.mps.is_available() else "cpu"


def _load_music_flamingo_impl():
    """Do the actual (slow, RAM/VRAM-heavy) work of loading Music Flamingo's
    weights. Callers should go through get_music_flamingo() instead, which
    caches the result so this only runs when the model isn't already resident."""
    if DEBUG_MODEL_LOADING:
        print(f"Loading {MF_MODEL_ID} ...")
    else:
        status(f"Loading {MF_MODEL_ID}...")
    processor = AutoProcessor.from_pretrained(MF_MODEL_ID)
    device = _mf_planned_device()
    model = AudioFlamingo3ForConditionalGeneration.from_pretrained(
        MF_MODEL_ID,
        device_map=device,
        torch_dtype=MF_TORCH_DTYPE,
    )
    if DEBUG_MODEL_LOADING:
        print(f"Music Flamingo loaded on device: {device}")
    else:
        status_done(f"Music Flamingo loaded on device: {device}")
    return model, processor, device


# --- Lazy loading / swapping for Music Flamingo ----------------------------
# Music Flamingo's weights are large and only actually needed while doing
# fresh audio analysis (era/full/vocal/confirmation/lyrics passes). Rather
# than holding them in RAM/VRAM for the entire session, we load them on
# first use and explicitly release them again as soon as that batch of
# analysis passes finishes, so the memory is available to other things
# (like the Ollama-served writer model) the rest of the time. This never
# touches conversation/chat history — that lives in plain Python lists in
# main() and is unaffected by whether any model is currently loaded.
_mf_state = {"model": None, "processor": None, "device": None}


def music_flamingo_is_loaded() -> bool:
    return _mf_state["model"] is not None


def get_music_flamingo():
    """Return (model, processor, device), loading Music Flamingo into
    RAM/VRAM on first call and reusing the cached instance thereafter.
    Call unload_music_flamingo() when done with a batch of analysis passes
    to free that memory again."""
    if _mf_state["model"] is None:
        model, processor, device = _load_music_flamingo_impl()
        _mf_state["model"] = model
        _mf_state["processor"] = processor
        _mf_state["device"] = device
    return _mf_state["model"], _mf_state["processor"], _mf_state["device"]


def unload_music_flamingo():
    """Drop Music Flamingo's weights and free the RAM/VRAM they occupied.
    Safe to call even if it was never loaded (no-op). The next call to
    get_music_flamingo() will transparently reload it."""
    if _mf_state["model"] is None:
        return

    status("Freeing Music Flamingo from memory...")
    model = _mf_state["model"]
    _mf_state["model"] = None
    _mf_state["processor"] = None
    _mf_state["device"] = None

    del model
    gc.collect()
    try:
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    status_done("Music Flamingo unloaded")


def mf_generate(
    model, processor, conversation,
    max_new_tokens: int = 2048, do_sample: bool = False, repetition_penalty: float = 1.0,
    no_repeat_ngram_size: int = 0,
):
    inputs = processor.apply_chat_template(
        conversation, tokenize=True, add_generation_prompt=True, return_dict=True
    ).to(model.device)

    if "input_features" in inputs:
        inputs["input_features"] = inputs["input_features"].to(model.dtype)

    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "repetition_penalty": repetition_penalty,
    }
    if no_repeat_ngram_size and no_repeat_ngram_size > 0:
        gen_kwargs["no_repeat_ngram_size"] = int(no_repeat_ngram_size)

    gen_ids = model.generate(**inputs, **gen_kwargs)
    new_tokens = gen_ids[:, inputs["input_ids"].shape[1]:]
    return processor.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()


def _sanitize_lyrics_transcription(text):
    """
    Cut token-loop / runaway repetition in lyrics transcriptions.
    Stops at [END OF TRANSCRIPTION] when present, and collapses long runs of
    the same short line or phrase.
    """
    if not text:
        return text

    # Prefer explicit end marker
    end_m = re.search(r"\[END OF TRANSCRIPTION\]", text, re.IGNORECASE)
    if end_m:
        text = text[: end_m.end()]

    lines = text.splitlines()
    out = []
    prev = None
    streak = 0
    for line in lines:
        stripped = line.strip()
        # Count consecutive near-identical non-empty lines
        if stripped and prev is not None and stripped == prev:
            streak += 1
            if streak >= 3:
                continue  # drop further repeats of the same line
        else:
            streak = 1 if stripped else 0
            prev = stripped if stripped else prev
        out.append(line)

    cleaned = "\n".join(out).strip()

    # Detect residual phrase loops inside a single long line (e.g. "oh lucky man " x50)
    def _collapse_phrase_loop(s):
        # Find a short phrase (3–40 chars) repeated many times
        m = re.search(r"(.{3,40}?)\1{4,}", s)
        if not m:
            return s
        phrase = m.group(1)
        # Keep at most 2 repetitions of that phrase in the matched region
        return re.sub(re.escape(phrase) + r"{3,}", phrase * 2, s)

    cleaned = _collapse_phrase_loop(cleaned)
    return cleaned


def _tempo_candidates(bpm):
    if bpm is None:
        return []
    try:
        bpm = float(bpm)
    except Exception:
        return []

    out = []
    seen = set()
    for x in (bpm / 2.0, bpm, bpm * 2.0):
        if 40.0 <= x <= 220.0:
            xr = round(float(x), 1)
            if xr not in seen:
                seen.add(xr)
                out.append(xr)
    return out

def reconcile_bpm(mf_bpm, essentia_bpm, objective_bpm=None):
    """
    Determine a single recommended BPM for discussion.

    Priority / logic:
    1. Prefer agreement between sources.
    2. When two sources differ by ~2x, treat the higher as a likely double-time
       misread and prefer the lower (common song-pulse) value. Detectors very
       often report double the felt pulse; reporting 160 when the song is 80 is
       the more common failure mode than the reverse.
    3. Fall back to Music Flamingo's TEMPO_BPM when it is the only strong signal.
    4. Always return a concrete number when any source is available.
    """
    candidates = []
    if mf_bpm is not None:
        try:
            candidates.append(("mf", float(mf_bpm)))
        except (TypeError, ValueError):
            pass
    if essentia_bpm is not None:
        try:
            candidates.append(("essentia", float(essentia_bpm)))
        except (TypeError, ValueError):
            pass
    if objective_bpm is not None:
        try:
            candidates.append(("objective", float(objective_bpm)))
        except (TypeError, ValueError):
            pass

    def _quantize_bpm(val):
        """Round to a musically plausible integer BPM.

        1) nearest integer
        2) if within 1.5 BPM of a common published tempo, snap to that
           (helps 123.4 → 125, 119.2 → 120, etc.)
        """
        if val is None:
            return None
        try:
            v = float(val)
        except (TypeError, ValueError):
            return None
        base = int(round(v))
        # Common dance/pop/rock tempos; prefer snap only when very close.
        common = (
            60, 64, 66, 70, 72, 74, 76, 80, 84, 88, 90, 92, 96, 98, 100,
            104, 105, 108, 110, 112, 115, 116, 118, 120, 122, 124, 125,
            126, 128, 130, 132, 135, 136, 138, 140, 144, 145, 148, 150,
            152, 155, 160, 165, 168, 170, 174, 175, 180, 190, 200,
        )
        best = base
        best_dist = abs(v - base)
        for c in common:
            d = abs(v - c)
            if d <= 1.5 and d < best_dist:
                best, best_dist = c, d
        return int(best)

    if not candidates:
        return None, "unavailable"

    if len(candidates) == 1:
        src, val = candidates[0]
        preferred, _, note = _preferred_tempo(val)
        q = _quantize_bpm(preferred)
        return q, f"{src} only. {note}".strip()

    by_src = {s: v for s, v in candidates}
    mf = by_src.get("mf")
    ess = by_src.get("essentia")
    obj = by_src.get("objective")

    def close(a, b):
        if a is None or b is None:
            return False
        ratio = max(a, b) / max(min(a, b), 1e-6)
        return ratio <= 1.15

    if mf is not None and ess is not None and close(mf, ess):
        return _quantize_bpm(ess), f"Essentia and MF agree (~{ess:.1f})."
    if mf is not None and obj is not None and close(mf, obj):
        return _quantize_bpm(mf), f"MF and objective detector agree (~{mf:.1f})."
    if ess is not None and obj is not None and close(ess, obj):
        return _quantize_bpm(ess), f"Essentia and objective detector agree (~{ess:.1f})."

    def is_double(a, b):
        if a is None or b is None:
            return False
        ratio = max(a, b) / max(min(a, b), 1e-6)
        return 1.8 <= ratio <= 2.2

    def prefer_lower_of_double(a, b, label_a, label_b):
        """When a and b are ~2x apart, prefer the lower as the felt pulse."""
        lo, hi = (a, b) if a <= b else (b, a)
        lo_src = label_a if a <= b else label_b
        hi_src = label_b if a <= b else label_a
        # Prefer lower when the higher sits in classic double-time territory
        if hi >= 140.0 and 55.0 <= lo <= 110.0:
            return _quantize_bpm(lo), (
                f"{hi_src} ({hi}) is ~2x {lo_src} ({lo}); preferring lower pulse "
                f"to avoid double-time misread."
            )
        # Otherwise keep the mid-range candidate if one is in 85-140
        for val, src in ((a, label_a), (b, label_b)):
            if 85.0 <= val <= 140.0:
                return _quantize_bpm(val), f"Preferring mid-range pulse from {src} ({val})."
        return _quantize_bpm(lo), f"2x pair {lo}/{hi}; defaulting to lower ({lo})."

    if mf is not None and ess is not None and is_double(mf, ess):
        return prefer_lower_of_double(mf, ess, "MF", "Essentia")

    if mf is not None and obj is not None and is_double(mf, obj):
        return prefer_lower_of_double(mf, obj, "MF", "objective")

    if ess is not None and obj is not None and is_double(ess, obj):
        return prefer_lower_of_double(ess, obj, "Essentia", "objective")

    if mf is not None:
        preferred, _, note = _preferred_tempo(mf)
        return _quantize_bpm(preferred), f"Using MF musical pulse ({mf}). {note}".strip()

    src, val = candidates[0]
    preferred, _, note = _preferred_tempo(val)
    return _quantize_bpm(preferred), f"{src} fallback. {note}".strip()

def extract_bpm_from_text(text):
    """Extracts the first numeric BPM value from text."""
    if not text:
        return None
    # Look for TEMPO_BPM= or estimated tempo ... BPM
    m = re.search(r"TEMPO_BPM\s*=\s*([0-9.]+)", text)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    
    # Fallback to objective report style
    m = re.search(r"estimated tempo.*?:\s*([0-9.]+)\s*BPM", text, re.IGNORECASE)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
            
    return None

def extract_essentia_bpm(text):
    """Extracts Essentia specific BPM."""
    if not text:
        return None
    m = re.search(r"Essentia estimated tempo.*?:\s*([0-9.]+)\s*BPM", text, re.IGNORECASE)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    return None


def extract_objective_bpm(text):
    """Extracts librosa/objective raw detector BPM from the signal-processing report."""
    if not text:
        return None
    m = re.search(r"estimated tempo \(raw detector\):\s*([0-9.]+)\s*BPM", text, re.IGNORECASE)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    return None


def _collapse_runaway_chord_repetition(text):
    """
    Safety net: if Music Flamingo emits a CHORDS field that is the same short
    progression repeated dozens of times, collapse it to a compact summary so
    the writer context and saved JSON stay usable.
    """
    if not text:
        return text

    pattern = re.compile(
        r"(CHORDS\s*[=:\-]\s*)(.*?)(?=\n(?:[A-Z][A-Z0-9_ ]{2,}[=:\-]|\n[A-Z][A-Z0-9_ ]{2,}[=:\-]|\Z))",
        re.IGNORECASE | re.DOTALL,
    )

    def _repl(m):
        prefix, body = m.group(1), m.group(2)
        tokens = re.split(r"\s*[→\->,]+\s*", body.strip())
        tokens = [t.strip() for t in tokens if t.strip()]
        if len(tokens) < 12:
            return m.group(0)
        unique = []
        for t in tokens:
            if t not in unique:
                unique.append(t)
            if len(unique) > 6:
                break
        if len(unique) <= 4 and len(tokens) >= 12:
            summary = " → ".join(unique)
            return (
                f"{prefix}{summary} (recurring loop; collapsed from highly repetitive listing)"
            )
        return m.group(0)

    return pattern.sub(_repl, text)

def _preferred_tempo(bpm):
    """
    Tempo interpretation helper for a single detector value.

    Detectors often report double the felt pulse. When a reading is high and
    half falls in a typical song-tempo band, prefer the half-time value.
    """
    if bpm is None:
        return None, [], ""
    try:
        bpm = float(bpm)
    except Exception:
        return None, [], ""

    cands = _tempo_candidates(bpm)
    preferred = round(bpm, 1)
    note = ""

    half = round(bpm / 2.0, 1)
    double = round(bpm * 2.0, 1)

    # Prefer half when the reading looks like a double-time misread.
    # Threshold 150 captures common 156/160 vs 78/80 confusions.
    if bpm >= 150.0 and 70.0 <= half <= 100.0:
        preferred = half
        note = (
            "fast detector value is likely a double-time/subdivision reading; "
            "preferring the lower pulse."
        )
    elif bpm > 175.0 and 70.0 <= half <= 140.0:
        preferred = half
        note = "very fast detector value may be a double-time reading; preferring the slower candidate."
    elif bpm < 70.0 and 80.0 <= double <= 140.0:
        preferred = double
        note = "slow detector value may be half-time; the doubled candidate may be the intended beat."

    return preferred, cands, note


def build_objective_audio_report(local_path):
    try:
        y, sr = librosa.load(local_path, sr=None, mono=True)
        duration = len(y) / float(sr)
        if duration < 1.0:
            return ""

        lines = []

        peak = np.max(np.abs(y))
        rms = np.sqrt(np.mean(y ** 2))
        if peak > 0 and rms > 0:
            crest_db = round(float(20 * np.log10(peak / rms)), 1)
            lines.append(f"duration: {round(duration, 2)} s")
            lines.append(f"crest factor / dynamic-range proxy: {crest_db} dB")
            if crest_db < 10.0:
                lines.append("dynamic-range note: low crest factor suggests heavy loudness-war-style compression, common from the 1990s onward, especially 2000s-2020s.")
            elif crest_db > 14.0:
                lines.append("dynamic-range note: high crest factor suggests a more dynamic, less-compressed master, more common pre-1990s or on audiophile/vinyl-style masters.")
            else:
                lines.append("dynamic-range note: moderate crest factor; do not infer era from this alone.")

        # start_bpm prior biases toward typical song tempos; reduces some
        # half/double misreads without forcing a specific value.
        tempo_raw, beats = librosa.beat.beat_track(y=y, sr=sr, start_bpm=120.0)
        tempo_arr = np.atleast_1d(tempo_raw)
        tempo = float(tempo_arr[0]) if len(tempo_arr) else None
        beat_times = librosa.frames_to_time(beats, sr=sr)

        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)

        lines.append("")
        lines.append("BEAT / TEMPO MEASUREMENTS")
        if tempo is not None:
            preferred, cands, note = _preferred_tempo(tempo)
            lines.append(f"estimated tempo (raw detector): {round(tempo, 1)} BPM")
            if preferred is not None and abs(preferred - round(tempo, 1)) > 0.5:
                lines.append(f"preferred musical tempo for discussion: {preferred} BPM")
            if len(cands) > 1:
                lines.append("tempo candidates (half/double interpretations): " + ", ".join(f"{c} BPM" for c in cands))
            if note:
                lines.append(note)
        else:
            lines.append("estimated tempo: unavailable")

        if len(beat_times):
            ibis = np.diff(beat_times)
            median_ibis = float(np.median(ibis)) if len(ibis) else None
            mean_ibis = float(np.mean(ibis)) if len(ibis) else None
            std_ibis = float(np.std(ibis)) if len(ibis) > 1 else None

            lines.append(f"detected beats: {len(beat_times)}")
            lines.append(f"first beat at {round(float(beat_times[0]), 2)} s, last beat at {round(float(beat_times[-1]), 2)} s")

            if median_ibis is not None and median_ibis > 0:
                lines.append(f"median inter-beat interval: {round(median_ibis, 3)} s ({round(60.0 / median_ibis, 1)} BPM)")

            if mean_ibis is not None and std_ibis is not None and mean_ibis > 0:
                lines.append(f"beat regularity (std/mean IBI): {round(std_ibis / mean_ibis, 3)}")

            if len(beat_times) >= 4:
                bar_times = beat_times[::4]
                if len(bar_times) > 1:
                    bar_durs = np.diff(bar_times)
                    lines.append(f"assuming 4/4: {len(bar_times)} bars, median bar length {round(float(np.median(bar_durs)), 3)} s")

                if not COMPACT_OBJECTIVE_REPORT:
                    beat_onset_values = []
                    for b in beats:
                        idx = int(b)
                        if 0 <= idx < len(onset_env):
                            beat_onset_values.append(float(onset_env[idx]))
                        else:
                            beat_onset_values.append(0.0)
                    beat_onset_values = np.array(beat_onset_values, dtype=float)

                    strong_indices = []
                    for i in range(0, len(beats) - 3, 4):
                        group = beat_onset_values[i:i + 4]
                        if len(group) == 4:
                            strong_indices.append(i + int(np.argmax(group)))

                    if strong_indices:
                        strong_times = beat_times[strong_indices]
                        shown = ", ".join(f"{round(float(t), 2)}" for t in strong_times[:16])
                        suffix = "..." if len(strong_times) > 16 else ""
                        lines.append(
                            f"candidate downbeats (strongest onset within each assumed 4-beat bar): "
                            f"{shown}{suffix}"
                        )

        if duration > 0:
            lines.append(f"onset density: {round(len(onsets) / duration * 60.0, 1)} onsets/min")

        seg_seconds = min(duration, 60.0)
        y_seg = y[: int(seg_seconds * sr)]

        if len(y_seg) >= sr * 0.5:
            try:
                n_fft = 2048
                hop_length = 512
                S = np.abs(librosa.stft(y_seg, n_fft=n_fft, hop_length=hop_length))
                freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
                total_power = float(np.sum(S ** 2))

                lines.append("")
                lines.append(f"TIMBRE / ELEMENT MEASUREMENTS (first {round(seg_seconds, 1)} s)")

                if total_power > 0:
                    centroid = float(np.mean(librosa.feature.spectral_centroid(S=S, sr=sr)))
                    rolloff75 = float(np.mean(librosa.feature.spectral_rolloff(S=S, sr=sr, roll_percent=0.75)))
                    rolloff85 = float(np.mean(librosa.feature.spectral_rolloff(S=S, sr=sr, roll_percent=0.85)))
                    rolloff95 = float(np.mean(librosa.feature.spectral_rolloff(S=S, sr=sr, roll_percent=0.95)))
                    flatness = float(np.mean(librosa.feature.spectral_flatness(S=S)))
                    zcr = float(np.mean(librosa.feature.zero_crossing_rate(y=y_seg)))

                    lines.append(f"brightness (spectral centroid): {round(centroid, 1)} Hz")
                    if COMPACT_OBJECTIVE_REPORT:
                        lines.append(f"spectral rolloff 85: {round(rolloff85, 1)} Hz")
                    else:
                        lines.append(
                            f"spectral rolloff 75/85/95: {round(rolloff75, 1)}, "
                            f"{round(rolloff85, 1)}, {round(rolloff95, 1)} Hz"
                        )
                    lines.append(f"noise-likeness (spectral flatness): {round(flatness, 4)} (0=tonal, 1=noisy)")
                    if not COMPACT_OBJECTIVE_REPORT:
                        lines.append(f"zero-crossing rate: {round(zcr, 4)}")

                    D, H = librosa.decompose.hpss(S)
                    hp_total = float(np.sum((H + D) ** 2))
                    if hp_total > 0:
                        perc_ratio = float(np.sum(D ** 2) / hp_total)
                        lines.append(f"percussive vs harmonic power share: {round(perc_ratio * 100.0, 1)}% percussive")

                        if not COMPACT_OBJECTIVE_REPORT:
                            def component_centroid(Sx):
                                p = float(np.sum(Sx ** 2))
                                if p <= 0:
                                    return None
                                c = librosa.feature.spectral_centroid(S=Sx, sr=sr)
                                return float(np.mean(c))

                            h_c = component_centroid(H)
                            d_c = component_centroid(D)
                            if h_c is not None and d_c is not None:
                                lines.append(
                                    f"harmonic component brightness: {round(h_c, 1)} Hz; "
                                    f"percussive component brightness: {round(d_c, 1)} Hz"
                                )

                    bands = [
                        ("sub-bass", 20, 60),
                        ("bass", 60, 250),
                        ("low-mid", 250, 800),
                        ("mid", 800, 4000),
                        ("high", 4000, 16000),
                    ]

                    lines.append("band energy share:")
                    for name, lo, hi in bands:
                        mask = (freqs >= lo) & (freqs < hi)
                        if np.any(mask):
                            band_power = float(np.sum(S[mask] ** 2)) / total_power
                            lines.append(f"  {name} ({lo}-{hi} Hz): {round(band_power * 100.0, 1)}%")

                    if not COMPACT_OBJECTIVE_REPORT:
                        mean_spec = np.mean(S, axis=1)
                        valid = freqs >= 20
                        valid_mean = np.where(valid, mean_spec, 0.0)
                        peak_ref = float(np.max(valid_mean))
                        if peak_ref > 0:
                            top_idx = [
                                int(i) for i in np.argsort(valid_mean)[::-1][:5]
                                if valid[int(i)] and valid_mean[int(i)] > 0
                            ]
                            if top_idx:
                                peak_desc = ", ".join(
                                    f"{int(freqs[i])} Hz ({round(float(20 * np.log10(mean_spec[i] / peak_ref)), 1)} dB)"
                                    for i in sorted(top_idx, key=lambda x: int(freqs[x]))
                                )
                                lines.append(f"dominant mean-spectrum peaks: {peak_desc}")

                        lines.append("band onset density (transient activity):")
                        for name, lo, hi in bands:
                            mask = (freqs >= lo) & (freqs < hi)
                            if not np.any(mask):
                                continue
                            S_band = np.zeros_like(S)
                            S_band[mask] = S[mask]
                            try:
                                env = librosa.onset.onset_strength(S=S_band, sr=sr, hop_length=hop_length)
                                band_onsets = librosa.onset.onset_detect(onset_envelope=env, sr=sr)
                                density = len(band_onsets) / seg_seconds * 60.0 if seg_seconds > 0 else 0.0
                                lines.append(f"  {name}: {round(float(density), 1)} onsets/min")
                            except Exception:
                                lines.append(f"  {name}: unavailable")
            except Exception:
                pass

        return "\n".join(lines)
    except Exception:
        return ""


def _clean_f0_series(f0_values):
    """Drop sparse outliers and octave jumps that inflate vocal range."""
    if not f0_values:
        return np.array([], dtype=float)
    arr = np.array(f0_values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 8:
        return arr
    # IQR fence on log-Hz (octave-stable)
    logf = np.log2(np.clip(arr, 40.0, 2000.0))
    q1, q3 = np.percentile(logf, [25, 75])
    iqr = max(q3 - q1, 1e-6)
    keep = (logf >= q1 - 1.5 * iqr) & (logf <= q3 + 1.5 * iqr)
    cleaned = arr[keep]
    if cleaned.size < 5:
        cleaned = arr
    # Second pass: drop points more than an octave from the median
    med = float(np.median(cleaned))
    if med > 0:
        ratio = cleaned / med
        cleaned = cleaned[(ratio >= 0.5) & (ratio <= 2.0)]
    return cleaned if cleaned.size else arr


def build_vocal_objective_report(local_path):
    try:
        y, sr = librosa.load(local_path, sr=None, mono=True)
        duration = len(y) / float(sr)
        if duration < 2.0:
            return ""

        max_scan_seconds = min(duration, 120.0)
        chunk_seconds = 30.0
        f0_values = []
        start = 0

        while start < int(max_scan_seconds * sr):
            end = min(start + int(chunk_seconds * sr), len(y))
            y_chunk = y[start:end]
            if len(y_chunk) < sr:
                break

            # Slightly narrower band + higher voicing threshold reduces
            # harmonic/bleed spikes that inflate the top of the range.
            f0, _, voiced_prob = librosa.pyin(
                y_chunk,
                fmin=librosa.note_to_hz("C2"),
                fmax=librosa.note_to_hz("C6"),
                sr=sr,
                frame_length=2048,
            )

            valid = ~np.isnan(f0) & (voiced_prob > 0.65)
            if np.any(valid):
                f0_values.extend(f0[valid].tolist())

            if len(f0_values) > 800:
                break

            start = end

        lines = []

        if f0_values:
            f0_arr = _clean_f0_series(f0_values)
            if f0_arr.size == 0:
                f0_arr = np.array(f0_values, dtype=float)
            median_f0 = float(np.median(f0_arr))
            p5 = float(np.percentile(f0_arr, 5))
            p10 = float(np.percentile(f0_arr, 10))
            p90 = float(np.percentile(f0_arr, 90))
            p95 = float(np.percentile(f0_arr, 95))

            lines.append("VOCAL OBJECTIVE MEASUREMENTS (pitch/formant proxies)")
            lines.append(f"voiced pitch median: {round(median_f0, 1)} Hz")
            lines.append(
                f"voiced pitch 5-95 percentile range: {round(p5, 1)}-{round(p95, 1)} Hz"
            )
            lines.append(
                f"practical sung range (10-90%): {round(p10, 1)}-{round(p90, 1)} Hz "
                f"(~{_hz_to_note_name(p10)} to {_hz_to_note_name(p90)}) — prefer this over absolute extremes"
            )

            try:
                lines.append(f"approx median note: {librosa.hz_to_note(median_f0)}")
            except Exception:
                pass

        scan_y = y[: int(max_scan_seconds * sr)]
        if len(scan_y) >= sr:
            S = np.abs(librosa.stft(scan_y, n_fft=2048, hop_length=512))
            D, H = librosa.decompose.hpss(S)
            freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)

            total_h_power = float(np.sum(H ** 2))
            if total_h_power > 0:
                vocal_mask = (freqs >= 250) & (freqs <= 4000)
                if np.any(vocal_mask):
                    vocal_share = float(np.sum(H[vocal_mask] ** 2) / total_h_power)
                    lines.append(f"harmonic energy in 250-4000 Hz vocal band: {round(vocal_share * 100.0, 1)}%")

                mean_h = np.mean(H, axis=1)
                valid_formant_band = (freqs >= 300) & (freqs <= 5000)
                valid_mean = np.where(valid_formant_band, mean_h, 0.0)
                peak_ref = float(np.max(valid_mean))

                if peak_ref > 0:
                    top_idx = [
                        int(i)
                        for i in np.argsort(valid_mean)[::-1][:4]
                        if valid_formant_band[int(i)] and valid_mean[int(i)] > 0
                    ]
                    if top_idx:
                        peak_desc = ", ".join(
                            f"{int(freqs[i])} Hz ({round(float(20 * np.log10(mean_h[i] / peak_ref)), 1)} dB)"
                            for i in sorted(top_idx, key=lambda x: int(freqs[x]))
                        )
                        lines.append(f"formant-like harmonic peaks: {peak_desc}")

        if not lines:
            return ""

        lines.append("Note: pitch and formant proxies overlap heavily between young male, young female, and some adult singers; use them as evidence, not proof.")
        return "\n".join(lines)
    except Exception:
        return ""


# --- Essentia integration helpers ------------------------------------------
def _essentia_first_float(value):
    try:
        if isinstance(value, (tuple, list)):
            value = value[0]
        if hasattr(value, "item"):
            val = float(value.item())
        else:
            arr = np.asarray(value, dtype=float).ravel()
            if arr.size == 0:
                return None
            val = float(arr[0])
        return val if np.isfinite(val) else None
    except Exception:
        return None


def _essentia_to_float_array(value):
    try:
        if value is None:
            return np.array([], dtype=float)

        if isinstance(value, (tuple, list)):
            # If the whole tuple/list is numeric, treat it as a sequence of values.
            try:
                arr = np.asarray(value, dtype=float).ravel()
                if arr.size:
                    return arr[np.isfinite(arr)]
            except Exception:
                pass

            for item in value:
                try:
                    arr = np.asarray(item, dtype=float).ravel()
                except Exception:
                    continue
                if arr.size > 1 or not np.isscalar(item):
                    return arr[np.isfinite(arr)]

            try:
                return np.array([float(value[0])], dtype=float)
            except Exception:
                return np.array([], dtype=float)

        arr = np.asarray(value, dtype=float).ravel()
        return arr[np.isfinite(arr)]
    except Exception:
        return np.array([], dtype=float)


def _essentia_set_sample_rate(kernel, sample_rate):
    for attr in ("sampleRate", "sr"):
        try:
            setattr(kernel, attr, int(sample_rate))
            return True
        except Exception:
            pass
    return False


def _essentia_load_audio(local_path, max_seconds=None):
    if not ESSENTIA_AVAILABLE or essentia is None:
        return None, None

    try:
        samples = None
        sample_rate = 44100

        # Prefer MonoLoader (returns a mono float vector).
        try:
            from essentia.standard import MonoLoader
            samples = np.asarray(
                MonoLoader(filename=local_path, sampleRate=44100)(),
                dtype=np.float32,
            )
            sample_rate = 44100
        except Exception:
            loader = None
            try:
                from essentia.standard import AudioLoader as _AudioLoader
                loader = _AudioLoader(filename=local_path)
            except Exception:
                try:
                    loader = essentia.standard.AudioLoader(filename=local_path)
                except Exception:
                    loader = None

            if loader is not None:
                loaded = loader()
                # AudioLoader returns (audio, sampleRate, channels, md5, bit_rate, codec)
                if isinstance(loaded, tuple) and len(loaded) >= 2:
                    samples = loaded[0]
                    sample_rate = loaded[1]
                else:
                    samples = loaded
                    sample_rate = 44100

        if samples is None:
            return None, None

        samples = np.asarray(samples, dtype=np.float32)

        if samples.ndim == 0:
            return None, None

        if samples.ndim == 2:
            # Usual layout is (samples, channels).
            if samples.shape[1] <= 8 and samples.shape[0] > samples.shape[1]:
                samples = samples.mean(axis=1).astype(np.float32)
            else:
                samples = samples.mean(axis=0).astype(np.float32)
        elif samples.ndim > 2:
            samples = samples.reshape(samples.shape[0], -1).mean(axis=1).astype(np.float32)

        if samples.size:
            peak_check = float(np.max(np.abs(samples)))
            if peak_check > 1.5:
                samples = (samples / peak_check).astype(np.float32)

        sample_rate = int(_essentia_first_float(sample_rate) or 44100)
        if sample_rate <= 0:
            return None, None

        if max_seconds is not None and len(samples) > int(max_seconds * sample_rate):
            samples = samples[: int(max_seconds * sample_rate)]

        if len(samples) < max(1, sample_rate // 2):
            return None, None

        return samples, sample_rate
    except Exception:
        return None, None


def _essentia_make_frame_kernel(kernel_class):
    if kernel_class is None or not callable(kernel_class):
        raise RuntimeError("Essentia kernel unavailable")

    try:
        return kernel_class(frameSize=ESSENTIA_FRAME_SIZE, hopSize=ESSENTIA_HOP_SIZE)
    except TypeError:
        # Some kernels may not accept frame parameters in every Essentia build.
        return kernel_class()


def _essentia_mean_feature(kernel_factory, samples, sample_rate):
    try:
        kernel = kernel_factory()
        _essentia_set_sample_rate(kernel, sample_rate)
        out = kernel(samples)
        arr = _essentia_to_float_array(out)
        if arr.size == 0:
            return None
        val = float(np.mean(arr))
        return val if np.isfinite(val) else None
    except Exception:
        return None


def _essentia_tempo_and_beats(samples, sample_rate):
    try:
        rhythm = RhythmExtractor2013(method="multifeature")
        _essentia_set_sample_rate(rhythm, sample_rate)
        out = rhythm(samples)

        if isinstance(out, (tuple, list)) and len(out) >= 2:
            tempo_out, beats_out = out[0], out[1]
        else:
            tempo_out, beats_out = out, None

        tempo = _essentia_first_float(tempo_out)
        beats = _essentia_to_float_array(beats_out)

        duration = len(samples) / float(sample_rate)
        if beats.size and np.max(beats) > max(duration * 2.0, 10.0):
            # Some builds/versions may return frame indices rather than seconds.
            beats = beats / float(sample_rate)

        beats = beats[np.isfinite(beats)]
        return tempo, beats
    except Exception:
        return None, np.array([], dtype=float)


def _essentia_key(samples, sample_rate):
    try:
        key_kernel = KeyExtractor()
        _essentia_set_sample_rate(key_kernel, sample_rate)
        out = key_kernel(samples)

        if isinstance(out, (tuple, list)) and len(out) >= 2:
            key_name, strength_out = out[0], out[1]
        elif isinstance(out, (tuple, list)):
            key_name, strength_out = out[0], None
        else:
            key_name, strength_out = out, None

        if isinstance(key_name, bytes):
            key_name = key_name.decode("utf-8", "ignore")

        key_name = str(key_name).strip() if key_name is not None else ""
        strength = _essentia_first_float(strength_out)

        return (key_name or None), strength
    except Exception:
        return None, None


def build_essentia_report(local_path):
    """
    Optional independent Essentia report.

    This is intentionally limited to objective audio measurements such as tempo/beat,
    key, spectral/timbre proxies, and dynamics. It should not be used by the writer
    to infer genre or vocal identity.
    """
    if not ENABLE_ESSENTIA_REPORT:
        return ""

    if local_path.startswith(("http://", "https://")):
        # Essentia is being used here on local files only, matching the existing DSP path.
        return ""

    samples, sample_rate = _essentia_load_audio(local_path, ESSENTIA_MAX_SECONDS)
    if samples is None or len(samples) == 0:
        return ""

    lines = []
    duration = len(samples) / float(sample_rate)
    lines.append(f"ESSENTIA OBJECTIVE MEASUREMENTS (first {round(duration, 2)} s)")

    # Tempo and beat timing.
    tempo, beats = _essentia_tempo_and_beats(samples, sample_rate)
    if tempo is not None:
        preferred, cands, note = _preferred_tempo(tempo)
        lines.append(f"Essentia estimated tempo (raw detector): {round(tempo, 1)} BPM")
        if preferred is not None and abs(preferred - round(tempo, 1)) > 0.5:
            lines.append(f"Essentia preferred musical tempo for discussion: {preferred} BPM")
        if len(cands) > 1:
            lines.append("Essentia tempo candidates (half/double interpretations): " + ", ".join(f"{c} BPM" for c in cands))
        if note:
            lines.append(note)
    else:
        lines.append("Essentia estimated tempo: unavailable")

    if beats.size:
        ibis = np.diff(beats)
        median_ibis = float(np.median(ibis)) if len(ibis) else None
        mean_ibis = float(np.mean(ibis)) if len(ibis) else None
        std_ibis = float(np.std(ibis)) if len(ibis) > 1 else None

        lines.append(f"Essentia detected beats: {len(beats)}")
        lines.append(
            f"first beat at {round(float(beats[0]), 2)} s, last beat at {round(float(beats[-1]), 2)} s"
        )

        if median_ibis is not None and median_ibis > 0:
            lines.append(
                f"Essentia median inter-beat interval: "
                f"{round(median_ibis, 3)} s ({round(60.0 / median_ibis, 1)} BPM)"
            )

        if mean_ibis is not None and std_ibis is not None and mean_ibis > 0:
            lines.append(f"Essentia beat regularity (std/mean IBI): {round(std_ibis / mean_ibis, 3)}")

    # Key estimation.
    key_name, key_strength = _essentia_key(samples, sample_rate)
    if key_name:
        strength_text = f", strength={round(key_strength, 3)}" if key_strength is not None else ""
        lines.append(f"Essentia estimated key: {key_name}{strength_text}")
    else:
        lines.append("Essentia estimated key: unavailable")

    # Spectral/timbre proxies on a shorter segment to keep runtime reasonable.
    lowlevel_seconds = min(duration, ESSENTIA_LOWLEVEL_MAX_SECONDS)
    seg = samples[: int(lowlevel_seconds * sample_rate)]

    if len(seg) >= ESSENTIA_FRAME_SIZE * 2:
        spectral_lines = []

        centroid = _essentia_mean_feature(
            lambda: _essentia_make_frame_kernel(SpectralCentroid),
            seg,
            sample_rate,
        )
        if centroid is not None:
            spectral_lines.append(f"Essentia brightness (spectral centroid): {round(centroid, 1)} Hz")

        flatness = _essentia_mean_feature(
            lambda: _essentia_make_frame_kernel(SpectralFlatness),
            seg,
            sample_rate,
        )
        if flatness is not None:
            spectral_lines.append(f"Essentia noise-likeness (spectral flatness): {round(flatness, 4)}")

        zcr = _essentia_mean_feature(
            lambda: _essentia_make_frame_kernel(ZeroCrossingRate),
            seg,
            sample_rate,
        )
        if zcr is not None:
            spectral_lines.append(f"Essentia zero-crossing rate: {round(zcr, 4)}")

        rms = _essentia_mean_feature(
            lambda: _essentia_make_frame_kernel(RMS),
            seg,
            sample_rate,
        )
        if rms is not None and rms > 0:
            peak = float(np.max(np.abs(seg)))
            spectral_lines.append(f"Essentia RMS level: {round(rms, 5)}")

            if peak > 0:
                crest_db = 20 * np.log10(peak / rms)
                if np.isfinite(crest_db):
                    spectral_lines.append(
                        f"Essentia RMS-based crest factor proxy: {round(float(crest_db), 1)} dB"
                    )

        if spectral_lines:
            lines.append("")
            lines.extend(spectral_lines)

    lines.append(
        "Note: Essentia values are objective measurements, not genre labels. "
        "Use them as cross-checks for tempo/beat/key/timbre/dynamics only."
    )

    return "\n".join(lines)
# ---------------------------------------------------------------------------


def _normalize_vocal_tag(value):
    value = (value or "").lower().strip()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_")


def _lead_from_category_modifier(category, modifier):
    category = _normalize_vocal_tag(category)
    modifier = _normalize_vocal_tag(modifier)

    if any(token in category for token in ("child")):
        if "male" in modifier and "female" not in modifier:
            return "child_male_likely"
        if "female" in modifier and "male" not in modifier:
            return "child_female_likely"
        return "child_gender_uncertain"

    if category in ("post_puberty_male", "adult_male") or (category == "male" and "post" in category):
        return "post_puberty_male"

    if category in ("female_teen_adult", "adult_female", "young_female"):
        return "female_teen_adult"

    if category in ("uncertain", "transitional"):
        if any(token in modifier for token in ("child", "young")):
            return "child_gender_uncertain"
        return "uncertain"

    return ""


def parse_vocal_tags(text):
    lead = ""
    backing = ""

    m = re.search(r'LEAD_PROFILE\s*=\s*["\']?([A-Za-z0-9_\- ]+)', text)
    if m:
        lead = _normalize_vocal_tag(m.group(1))

    if lead not in VOCAL_LEAD_TAGS:
        cat_m = re.search(r'LEAD_CATEGORY\s*=\s*["\']?([A-Za-z0-9_\- ]+)', text)
        mod_m = re.search(r'GENDER_MODIFIER\s*=\s*["\']?([A-Za-z0-9_\- ]+)', text)
        if cat_m and mod_m:
            lead = _lead_from_category_modifier(cat_m.group(1), mod_m.group(1))

    lead = VOCAL_LEAD_ALIASES.get(lead, lead)
    if lead not in VOCAL_LEAD_TAGS:
        lead = "unknown"

    m = re.search(r'BACKING_PROFILES\s*=\s*["\']?([A-Za-z0-9_\- ]+)', text)
    if m:
        backing = _normalize_vocal_tag(m.group(1))
    if backing not in VOCAL_BACKING_TAGS:
        backing = "uncertain"

    return lead, backing


def parse_vocal_confirmation(text):
    lead = ""
    confidence = ""

    m = re.search(r'LEAD_PROFILE\s*=\s*["\']?([A-Za-z0-9_\- ]+)', text)
    if m:
        lead = _normalize_vocal_tag(m.group(1))

    if lead not in VOCAL_LEAD_TAGS:
        cat_m = re.search(r'LEAD_CATEGORY\s*=\s*["\']?([A-Za-z0-9_\- ]+)', text)
        mod_m = re.search(r'GENDER_MODIFIER\s*=\s*["\']?([A-Za-z0-9_\- ]+)', text)
        if cat_m and mod_m:
            lead = _lead_from_category_modifier(cat_m.group(1), mod_m.group(1))

    lead = VOCAL_LEAD_ALIASES.get(lead, lead)
    if lead not in VOCAL_LEAD_TAGS:
        lead = ""

    m = re.search(r"CONFIDENCE\s*=\s*(low|medium|high)", text, re.IGNORECASE)
    if m:
        confidence = m.group(1).lower()

    return lead, confidence


def extract_vocal_median_f0(text):
    m = re.search(r"voiced pitch median:\s*([0-9]+(?:\.[0-9]+)?)\s*Hz", text or "")
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


def extract_vocal_pitch_summary(text):
    median = extract_vocal_median_f0(text)
    low = None
    high = None
    note = None

    m = re.search(r"voiced pitch 5-95 percentile range:\s*([0-9]+(?:\.[0-9]+)?)-([0-9]+(?:\.[0-9]+)?)\s*Hz", text or "")
    if m:
        try:
            low = float(m.group(1))
            high = float(m.group(2))
        except Exception:
            pass

    m = re.search(r"approx median note:\s*(\S+)", text or "")
    if m:
        note = m.group(1).strip()

    return {"median": median, "low": low, "high": high, "note": note}


def choose_final_vocal_lead(initial_lead, confirm_lead, confirm_confidence):
    initial = initial_lead if initial_lead in VOCAL_LEAD_TAGS else "unknown"
    confirm = confirm_lead if confirm_lead in VOCAL_LEAD_TAGS else ""

    if not confirm:
        return initial

    strong_confirm = confirm_confidence in ("medium", "high")
    high_confirm = confirm_confidence == "high"

    # If the isolated pass already says mixed leads, do not silently collapse that
    # into a single-gender lead from the confirmation pass alone.
    if initial == "mixed_leads":
        return initial

    if initial == "unknown":
        return confirm if strong_confirm else "uncertain"

    # If the initial result is already young/gender-uncertain:
    # - accept a clear teenage/adult female result at medium/high confidence
    # - require high confidence before overriding with other child/young categories
    if initial in UNCERTAIN_YOUNG_CATEGORIES:
        if confirm == "female_teen_adult" and strong_confirm:
            return confirm

        if high_confirm and confirm not in ("unknown",):
            return confirm

        return initial

    # Move away from female only when the confirmation pass is confident enough.
    # This still targets the common failure mode where a young male voice gets labelled
    # female because of pitch/youth, but it avoids demoting a teenage/adult female to
    # "uncertain" on medium-confidence evidence alone.
    if initial in FEMALE_LEAD_CATEGORIES:
        if strong_confirm and confirm in MALE_LEAD_CATEGORIES:
            return confirm

        if high_confirm and confirm in UNCERTAIN_YOUNG_CATEGORIES:
            return confirm

        if confirm == initial:
            return confirm

        return initial

    # Do not automatically turn a male lead into female from the confirmation pass alone.
    # That should require user correction or explicit distinct co-lead evidence, which is
    # better handled outside this tag-only function.
    if initial in MALE_LEAD_CATEGORIES:
        return initial

    if confirm == initial:
        return confirm

    return initial


def build_vocal_priority_note(lead_profile, backing_profiles):
    if lead_profile in ("child_male_likely", "young_male"):
        note = (
            "\n\nVOCAL CLASSIFICATION PRIORITY: The final lead vocal category is a young/child male voice. "
            "For user-facing claims about the lead singer, say 'young male voice'. Say 'boy' only if the evidence is unambiguously child-like and confidence is high; otherwise do not overstate certainty."
        )

    elif lead_profile in ("child_female_likely", "young_female"):
        note = (
            "\n\nVOCAL CLASSIFICATION PRIORITY: The final lead vocal category is a young/child female voice. "
            "For user-facing claims about the lead singer, say 'young female voice'. Say 'girl' only if the evidence is unambiguously child-like and confidence is high; otherwise do not overstate certainty."
        )

    elif lead_profile in ("child_gender_uncertain", "child_gender_uncertain"):
        note = (
            "\n\nVOCAL CLASSIFICATION PRIORITY: The final lead vocal category is a young/child voice with uncertain gender. "
            "For user-facing claims, say 'young/child voice; gender uncertain' or 'young voice; cannot confidently tell boy/girl'. Do not call it a girl/woman/boy/man unless the user explicitly corrects you."
        )

    elif lead_profile in ("post_puberty_male", "adult_male"):
        note = (
            "\n\nVOCAL CLASSIFICATION PRIORITY: The final lead vocal category is a post-puberty male voice. "
            "Say 'male voice' or 'male lead'; say 'teen/adult male' only if the analysis explicitly supports that age range."
        )


    elif lead_profile in ("female_teen_adult", "adult_female"):
        note = (
            "\n\nVOCAL CLASSIFICATION PRIORITY: The final lead vocal category is a female teen/adult voice. "
            "Say 'female voice', 'female lead', or 'young adult female voice' as appropriate. "
            "Do not describe this as child-like, boy/girl uncertain, or a young male voice unless the user explicitly corrects it."
        )

    elif lead_profile == "mixed_leads":
        note = (
            "\n\nVOCAL CLASSIFICATION PRIORITY: The final lead vocal category is mixed leads. "
            "Only describe this as mixed male/female vocals if distinct co-leads of both genders are explicitly established; otherwise say the dominant lead plus possible backing."
        )

    else:
        note = (
            "\n\nVOCAL PROFILE PRIORITY: For final claims about lead vocal age/gender category, prefer the VOCAL DECISION AUDIT / FINAL LEAD PROFILE "
            "over generic wording elsewhere in this analysis. Distinguish lead from backing; do not claim mixed male/female vocals unless distinct voices are explicitly identified."
        )

    if backing_profiles == "female":
        note += (
            " If female backing/harmonies are mentioned, describe them as possible female backing only, "
            "not as an equal mixed-lead performance, unless distinct co-leads are established."
        )
    elif backing_profiles == "male":
        note += (
            " If male backing/harmonies are mentioned, describe them as possible male backing only, "
            "not as an equal mixed-lead performance, unless distinct co-leads are established."
        )
    elif backing_profiles == "mixed":
        note += (
            " Mixed backing is not the same as mixed lead vocals. Do not describe the track as having mixed lead vocals "
            "unless distinct male and female co-leads are explicitly established."
        )

    return note


def extract_file_reference(text: str, url_pattern: re.Pattern, extensions: tuple):
    """
    Generic detector for a URL or local file path (which may contain spaces,
    apostrophes, etc.) ending in one of `extensions`. Returns (cleaned_text, ref)
    — ref is None if nothing was found.
    """
    match = url_pattern.search(text)
    if match:
        url = match.group(1)
        cleaned = text.replace(url, "").strip()
        return cleaned, url

    ext_lower = tuple(e.lower() for e in extensions)
    ext_group = "|".join(re.escape(e.lstrip(".")) for e in extensions)

    def _exists(p):
        try:
            return bool(p) and os.path.exists(p)
        except Exception:
            return False

    # Primary: shell-style tokenization (handles drag-and-drop backslash escapes).
    try:
        tokens = shlex.split(text, posix=True)
    except ValueError:
        tokens = []

    for i, token in enumerate(tokens):
        if token.lower().endswith(ext_lower) and _exists(token):
            remaining = tokens[:i] + tokens[i + 1:]
            return " ".join(remaining).strip(), token

    # Unescape common shell forms, including apostrophes.
    unescaped = text.replace("\\'", "'").replace("\\ ", " ")

    # Quoted paths — allow apostrophes inside double quotes and vice versa.
    for qpat in (
        rf'"([^"]+\.(?:{ext_group}))"',
        rf"'([^']+\.(?:{ext_group}))'",
    ):
        quoted = re.search(qpat, unescaped, re.IGNORECASE)
        if quoted and _exists(quoted.group(1)):
            cleaned = unescaped.replace(quoted.group(0), "").strip()
            return cleaned, quoted.group(1)

    # Walk left from each extension match; prefer the longest existing path.
    best = None  # (start, end, path)
    for ext_match in re.finditer(rf"\.(?:{ext_group})\b", unescaped, re.IGNORECASE):
        end = ext_match.end()
        # Candidate starts: any non-space run start before the extension
        starts = [m.start() for m in re.finditer(r"\S+", unescaped[:end])]
        for start in starts:
            candidate = unescaped[start:end].strip().rstrip(".,!?;:)")
            # Strip unbalanced leading quotes
            candidate = candidate.lstrip(chr(39) + chr(34))
            if _exists(candidate):
                if best is None or (end - start) > (best[1] - best[0]):
                    best = (start, end, candidate)
    if best is not None:
        start, end, candidate = best
        cleaned = (unescaped[:start] + " " + unescaped[end:]).strip()
        cleaned = re.sub(r"\s+", " ", cleaned)
        return cleaned, candidate

    return text, None


def extract_image_reference(text: str):
    return extract_file_reference(text, IMAGE_URL_PATTERN, IMAGE_EXTENSIONS)


def extract_audio_reference(text: str):
    return extract_file_reference(text, AUDIO_URL_PATTERN, AUDIO_EXTENSIONS)


def extract_image_references(text: str):
    """
    Extract multiple image references (URLs and local files) from a request.
    Returns (cleaned_text, list_of_refs).
    """
    refs = []
    remaining = text

    # URLs first.
    for m in IMAGE_URL_PATTERN.finditer(remaining):
        refs.append(m.group(1))
    if refs:
        remaining = IMAGE_URL_PATTERN.sub("", remaining).strip()

    ext_lower = tuple(e.lower() for e in IMAGE_EXTENSIONS)
    kept_tokens = []
    local_refs = []

    try:
        tokens = shlex.split(remaining, posix=True)
    except ValueError:
        tokens = []

    if tokens:
        for tok in tokens:
            if tok.lower().endswith(ext_lower) and os.path.exists(tok):
                local_refs.append(tok)
            else:
                kept_tokens.append(tok)
        remaining = " ".join(kept_tokens).strip()

    # Fallback heuristic if nothing was found.
    if not refs and not local_refs:
        unescaped = remaining.replace("\\ ", " ").replace("\\'", "'")
        ext_group = "|".join(e.lstrip(".") for e in IMAGE_EXTENSIONS)
        found = []
        for m in re.finditer(rf"\.(?:{ext_group})\b", unescaped, re.IGNORECASE):
            end = m.end()
            starts = [mm.start() for mm in re.finditer(r"\S+", unescaped[:end])]
            for start in starts:
                candidate = unescaped[start:end].strip()
                if os.path.exists(candidate) and candidate not in found:
                    found.append(candidate)
                    break
        if found:
            refs.extend(found)
            rem = remaining
            for cand in found:
                rem = rem.replace(cand, "")
            remaining = rem.strip()

    unique_refs = []
    seen = set()
    for r in refs + local_refs:
        if r not in seen:
            seen.add(r)
            unique_refs.append(r)

    return remaining, unique_refs


def image_to_base64(image_ref: str) -> str:
    if image_ref.startswith("http://") or image_ref.startswith("https://"):
        resp = requests.get(image_ref, timeout=30)
        resp.raise_for_status()
        data = resp.content
    else:
        with open(image_ref, "rb") as f:
            data = f.read()

    if not data:
        raise ValueError("empty image")

    if len(data) > MAX_EXPLICIT_IMAGE_BYTES:
        raise ValueError(
            f"image too large ({len(data)} bytes); limit is {MAX_EXPLICIT_IMAGE_BYTES}"
        )

    mime = _guess_image_mime(data)
    if mime not in COVER_IMAGE_SENDABLE_MIMES:
        raise ValueError(f"not a sendable image type: {mime}")

    return base64.b64encode(data).decode("utf-8")


def _is_sendable_base64_image(b64):
    try:
        data = base64.b64decode(str(b64), validate=False)
        return bool(data) and _guess_image_mime(data) in COVER_IMAGE_SENDABLE_MIMES
    except Exception:
        return False


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


def _compress_stem_midi_event_logs(text, max_events_per_log=30):
    """Reduce huge stem/MIDI JSON event logs before sending them to Ollama.

    This preserves end-to-end coverage by sampling evenly instead of only keeping
    the first N events.
    """
    if not text or "events (" not in text:
        return text

    pattern = re.compile(r"((?:note|hit) events \((?:MIDI|rhythm)-JSON\):\s*)(\[.*\])")
    out_lines = []

    for line in str(text).splitlines():
        m = pattern.search(line)
        if not m:
            out_lines.append(line)
            continue

        try:
            arr = json.loads(m.group(2))
            if isinstance(arr, list) and len(arr) > max_events_per_log:
                arr = _sample_list_evenly(arr, max_events_per_log)

            line = (
                line[:m.start()]
                + m.group(1)
                + json.dumps(arr)
                + line[m.end():]
            )
        except Exception:
            pass

        out_lines.append(line)

    return "\n".join(out_lines)


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

    if tail < 1200:
        tail = max(1200, int(limit * 0.25))
        head = max(1000, limit - tail - len(marker))

    return text[:head] + marker + text[-tail:]


def _prepare_ollama_messages(messages, num_ctx, keep_images=True):
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

    budget = int(num_ctx * HISTORY_CHAR_BUDGET_FACTOR)
    prepared = []
    start = 0

    if messages[0].get("role") == "system":
        sys_msg = dict(messages[0])
        sys_limit = max(6000, min(30000, int(budget * 0.35)))
        sys_msg["content"] = _truncate_text_for_ollama(sys_msg.get("content"), sys_limit)

        prepared.append(sys_msg)
        budget -= _estimate_ollama_message_chars(sys_msg)
        start = 1

    kept = []

    for msg in reversed(messages[start:]):
        m = dict(msg)

        content = m.get("content") or ""
        if isinstance(content, list):
            content = json.dumps(content)

        max_msg_chars = min(MAX_MESSAGE_CHARS_FOR_OLLAMA, max(4000, int(budget // 3)))

        content = _compress_stem_midi_event_logs(
            str(content),
            40 if num_ctx > 16384 else 25,
        )
        content = _truncate_text_for_ollama(content, max_msg_chars)

        m["content"] = content

        imgs = m.get("images") or []
        if keep_images:
            m["images"] = [str(img) for img in list(imgs)[:MAX_WRITER_IMAGES_PER_TURN]]
        else:
            m.pop("images", None)

        kept.append(m)
        cost = _estimate_ollama_message_chars(m)

        # Always keep the newest message, even if it alone exceeds budget.
        if len(kept) == 1:
            budget = max(0, budget - cost)
        elif budget < cost:
            break
        else:
            budget -= cost

    kept.reverse()
    prepared.extend(kept)

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


def _evidence_message_still_safe(history, evidence_msg, num_ctx, safety_margin=0.45):
    """Decide whether it's safe to skip re-sending a track's full evidence block
    because an earlier message in `history` already carries it in full.

    Two things have to hold:
      1. The message object must still physically be in `history` — permanent
         compaction (_compact_writer_history_in_place) evicts old messages by
         object, so an identity check here exactly matches what compaction did.
      2. The messages *after* it must not have grown so large that per-request
         trimming (_prepare_ollama_messages, which only guarantees the newest
         message survives) would plausibly drop it from this turn's request.
         We approximate that by requiring the trailing chars to stay under
         `safety_margin` of the char budget that trimming itself uses.

    If either check fails, the caller should re-send the evidence in full
    rather than trust a pointer to it — silently answering from nothing is
    worse than the token cost of resending.
    """
    try:
        idx = next(i for i, m in enumerate(history) if m is evidence_msg)
    except StopIteration:
        return False

    budget = num_ctx * HISTORY_CHAR_BUDGET_FACTOR
    trailing_chars = sum(
        _estimate_ollama_message_chars(m) for m in history[idx + 1:]
    )
    return trailing_chars < budget * safety_margin


def _compact_writer_history_in_place(history):
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
            image_indices = [
                i for i, m in enumerate(history)
                if isinstance(m.get("images"), list) and m["images"]
            ]

            keep_image_indices = set(image_indices[-MAX_STORED_IMAGES_IN_HISTORY:])

            for i, m in enumerate(history):
                if "images" not in m:
                    continue

                if i not in keep_image_indices:
                    m.pop("images", None)
                else:
                    m["images"] = [
                        str(img)
                        for img in list(m.get("images") or [])[:MAX_WRITER_IMAGES_PER_TURN]
                    ]

        for m in history:
            content = m.get("content")
            if isinstance(content, str) and len(content) > 20000 and "events (" in content:
                m["content"] = _compress_stem_midi_event_logs(content, 30)

        if len(history) > MAX_WRITER_HISTORY_MESSAGES:
            system = history[0] if history and history[0].get("role") == "system" else None
            rest = history[1:] if system is not None else history[:]

            keep_count = (
                MAX_WRITER_HISTORY_MESSAGES - 1
                if system is not None
                else MAX_WRITER_HISTORY_MESSAGES
            )

            kept_rest = rest[-keep_count:] if keep_count > 0 else []

            history.clear()
            if system is not None:
                history.append(system)
            history.extend(kept_rest)

    except Exception as e:
        print(f"  (writer-history compaction skipped: {e})")


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


def ollama_chat(messages: list, num_ctx=None):
    if num_ctx is None:
        num_ctx = OLLAMA_NUM_CTX

    # Compact the stored history in place before building the request.
    _compact_writer_history_in_place(messages)

    attempts = [
        (int(num_ctx), True),
        (min(int(num_ctx), 32768), True),
        (16384, True),
        (8192, True),
    ]

    # De-duplicate while preserving order. Same context with different image policy is allowed.
    seen = set()
    unique_attempts = []
    for attempt in attempts:
        if attempt not in seen:
            seen.add(attempt)
            unique_attempts.append(attempt)
    attempts = unique_attempts

    last_error = ""

    for idx, (ctx, keep_images) in enumerate(attempts):
        try:
            payload_messages = _prepare_ollama_messages(
                messages,
                ctx,
                keep_images=keep_images,
            )

            resp = requests.post(
                OLLAMA_URL,
                json={
                    "model": OLLAMA_MODEL,
                    "messages": payload_messages,
                    "stream": False,
                    "options": {"num_ctx": ctx},
                },
                timeout=900,
            )

            if resp.status_code == 400 and idx < len(attempts) - 1:
                last_error = _ollama_response_error(resp)
                print(
                    f"  (Ollama returned 400; retrying with smaller context/images. "
                    f"Error: {last_error[:300]})"
                )
                continue

            resp.raise_for_status()
            data = resp.json()
            content = str(data.get("message", {}).get("content") or "").strip()
            # Ollama reports per-request token counts when stream=False.
            prompt_n = int(data.get("prompt_eval_count") or 0)
            comp_n = int(data.get("eval_count") or 0)
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

            # Retry only on 400s or missing responses. Other HTTP errors are usually
            # not fixed by shrinking context/images.
            if idx < len(attempts) - 1 and (
                e.response is None or e.response.status_code == 400
            ):
                continue

            raise RuntimeError(f"Ollama request failed: {last_error}") from e

        except requests.exceptions.RequestException as e:
            last_error = str(e)
            print(f"  (Ollama network error: {last_error[:300]})")

            # Retry transient connection/timeout errors if possible.
            if idx < len(attempts) - 1 and (
                "Connection" in str(e) or "Timeout" in str(e)
            ):
                continue

            raise RuntimeError(f"Ollama request failed: {last_error}") from e

    raise RuntimeError(f"Ollama request failed after retries: {last_error}")


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
        target_base = OLLAMA_MODEL.split(":")[0]

        for m in models:
            name = str(m.get("name") or m.get("model") or "")
            if not name:
                continue

            base = name.split(":")[0]

            if (
                name == OLLAMA_MODEL
                or base == target_base
                or name.startswith(OLLAMA_MODEL + ":")
            ):
                return True

        return False

    except Exception:
        return None


def ollama_unload_model():
    """Best-effort unload of the writer model from Ollama.

    The old implementation sent an empty /api/chat request, which can be rejected
    with 400 and therefore silently failed to unload Gemma. This version checks
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


def _cover_observation_is_useful(obs):
    if not obs:
        return False
    keys = ("people", "person_cues", "text_logos", "era_cues", "style_vibe")
    return any(str(obs.get(k) or "").strip() for k in keys)


def _parse_cover_observation(text):
    obs = {"raw": (text or "").strip()}
    for field in ("PEOPLE", "PERSON_CUES", "TEXT_LOGOS", "ERA_CUES", "STYLE_VIBE", "CONFIDENCE"):
        m = re.search(rf"^{field}\s*[:=]\s*(.+)$", text or "", re.MULTILINE | re.IGNORECASE)
        obs[field.lower()] = m.group(1).strip() if m else ""
    return obs


def _format_cover_observation_block(obs):
    if not obs:
        return ""

    lines = [
        "COVER ART OBSERVATIONS (structured visual evidence from the attached artwork; not direct audio proof):"
    ]

    labels = [
        ("people", "People"),
        ("person_cues", "Person cues"),
        ("text_logos", "Text/logos"),
        ("era_cues", "Era cues"),
        ("style_vibe", "Style/vibe"),
        ("confidence", "Confidence"),
    ]

    for key, label in labels:
        val = str(obs.get(key) or "").strip()
        lines.append(f"{label}: {val if val else 'unavailable'}")

    return "\n".join(lines)


def describe_cover_art(cover_b64):
    """
    Return a structured dict of visual observations from an attached image.
    Returns {} on failure or if nothing useful was extracted.
    """
    if not ENABLE_COVER_ART_DESCRIPTION or not OLLAMA_SUPPORTS_IMAGES:
        return {}

    if not cover_b64 or cover_b64 == NO_COVER_SENTINEL:
        return {}

    try:
        text, _usage = ollama_chat([
                {
                    "role": "user",
                    "content": COVER_ART_OBSERVATION_PROMPT,
                    "images": [cover_b64],
                }
            ],
            num_ctx=COVER_ART_DESCRIPTION_NUM_CTX,
        )
    except Exception as e:
        print(f"  (cover art description failed: {e})")
        return {}

    obs = _parse_cover_observation(text)

    if not _cover_observation_is_useful(obs):
        stripped = (text or "").strip()
        if len(stripped) > 20:
            # Fallback: keep the prose as a style/vibe note rather than discarding it entirely.
            obs["style_vibe"] = stripped[:500]
            return obs
        return {}

    return obs


def _parse_singer_identity(text):
    m = re.search(r"SINGER_IDENTITY\s*[:=]\s*([A-Za-z0-9_\- ]+)", text or "")
    if not m:
        return ""

    tag = _normalize_vocal_tag(m.group(1))
    tag = VOCAL_LEAD_ALIASES.get(tag, tag)

    simple_aliases = {
        "young_male": "uncertain",
        "male_lead": "post_puberty_male",
        "adult_female": "female_teen_adult",
    }
    tag = simple_aliases.get(tag, tag)

    return tag if tag in VOCAL_LEAD_TAGS else ""


def resolve_singer_identity(metadata, vocal_audit_text, cover_obs, corrections=None):
    """
    Text-only resolution pass that combines:
      - user corrections
      - file metadata
      - audio vocal audit summary
      - structured cover-art observations

    Returns the raw model text (expected to contain SINGER_IDENTITY=...), or "" on failure.
    """
    if not ENABLE_SINGER_IDENTITY_RESOLUTION:
        return ""

    # User corrections are ground truth; short-circuit when a singer correction exists.
    for k, v in (corrections or {}).items():
        if str(k).lower() in VOCAL_CORRECTION_FIELDS:
            tag = _normalize_vocal_tag(v)
            tag = VOCAL_LEAD_ALIASES.get(tag, tag)
            if tag in VOCAL_LEAD_TAGS:
                return (
                    f"SINGER_IDENTITY={tag}\n"
                    "REASONING=User correction.\n"
                    "CONFIDENCE=high"
                )

    meta_lines = []
    for key, label in (("title", "Title"), ("artist", "Artist"), ("album", "Album"), ("year", "Year")):
        val = str((metadata or {}).get(key) or "").strip()
        if val:
            meta_lines.append(f"{label}: {val}")

    meta_block = "\n".join(meta_lines) if meta_lines else "No file metadata available."

    corr_lines = []
    for k, v in (corrections or {}).items():
        corr_lines.append(f"- {k}: {v}")
    corr_block = "\n".join(corr_lines) if corr_lines else "None"

    cover_block = _format_cover_observation_block(cover_obs) if cover_obs else "No cover art observations available."

    prompt = (
        SINGER_IDENTITY_RESOLUTION_PROMPT
        + f"\n\nUSER CORRECTIONS (ground truth):\n{corr_block}"
        + f"\n\nTRACK METADATA:\n{meta_block}"
        + f"\n\nVOCAL EVIDENCE:\n{(vocal_audit_text or '').strip() or 'No vocal evidence available.'}"
        + f"\n\nCOVER ART OBSERVATIONS:\n{cover_block}"
    )

    try:
        _si_text, _usage = ollama_chat([{"role": "user", "content": prompt}],
            num_ctx=SINGER_IDENTITY_NUM_CTX,
        )
        return _si_text
    except Exception as e:
        print(f"  (singer identity resolution failed: {e})")
        return ""


def _short_vocal_audit(analysis):
    text = analysis or ""
    parts = []

    for pattern in (
        r"FINAL LEAD PROFILE:\s*\S+",
        r"BACKING PROFILES:\s*\S+",
        r"objective median f0:\s*.*",
        r"Confirmation pass:\s*.*",
    ):
        m = re.search(pattern, text)
        if m:
            parts.append(m.group(0).strip())

    return "\n".join(parts) or "No vocal audit available."


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


# --- Stem separation + Omnizart MIDI helpers ---------------------------------
_OMNIZART_APPS = None
OMNIZART_IMPORT_ERROR = ""

# GM-style percussion pitch -> concrete drum-type label. Omnizart's drum
# model only distinguishes 3 classes (kick/snare/hihat) but writes them out
# using standard GM percussion note numbers, so this covers the full GM
# percussion range (35-59) in case a build ever emits toms/cymbals too.
GM_DRUM_PITCH_TYPE = {
    35: "kick", 36: "kick",
    37: "snare", 38: "snare", 39: "snare", 40: "snare",
    41: "tom", 43: "tom", 45: "tom", 47: "tom", 48: "tom", 50: "tom",
    42: "hihat", 44: "hihat", 46: "hihat",
    49: "cymbal", 51: "cymbal", 52: "cymbal", 53: "cymbal",
    55: "cymbal", 57: "cymbal", 59: "cymbal",
}


def _guess_drum_type_from_name(name):
    """Some Omnizart versions emit one pretty_midi.Instrument per drum
    class (named e.g. 'Bass Drum'/'Snare'/'Hi-Hat') rather than encoding
    the class purely via note pitch. Check the instrument name first so
    we don't miss the class label in that case."""
    if not name:
        return None
    n = name.lower()
    if "kick" in n or "bass drum" in n:
        return "kick"
    if "snare" in n:
        return "snare"
    if "hihat" in n or "hi-hat" in n or "hi hat" in n:
        return "hihat"
    if "tom" in n:
        return "tom"
    if "cymbal" in n or "crash" in n or "ride" in n:
        return "cymbal"
    return None

STEM_MIDI_STEMS = ("vocals", "bass", "guitar", "piano", "other", "drums")

# NOTE: onset_threshold / frame_threshold / minimum_note_length_ms were
# tuned for the old Basic Pitch detector and no longer steer Omnizart's
# detection (Omnizart doesn't expose those knobs). They're kept here
# because min_frequency/max_frequency/min_amplitude/monophonic and the
# ms-based fields are still read by the post-transcription filter/summary
# functions below.
STEM_MIDI_PRESETS = {
    "vocals": {
        "min_frequency": 80,
        "max_frequency": 1200,
        "onset_threshold": 0.35,      # Slightly up to avoid breath/sibilance triggers
        "frame_threshold": 0.20,      # Slightly up for cleaner note tails
        "minimum_note_length_ms": 50, # Unified to ms for safety
        "min_amplitude": 0.10,        # Filter out low-level room bleed
        "monophonic": True,
    },
    "bass": {
        "min_frequency": 30,
        "max_frequency": 300,
        "onset_threshold": 0.35,      # Up slightly to stop low-end rumble triggers
        "frame_threshold": 0.25,      # Bass needs higher frame confidence to prevent sustain bloat
        "minimum_note_length_ms": 60, # Bass notes need a slightly longer floor
        "min_amplitude": 0.08,
        "monophonic": True,
    },
    "guitar": {
        "min_frequency": 80,
        "max_frequency": 2000,
        "onset_threshold": 0.50,      # Back down from 0.60 to catch subtle picks
        "frame_threshold": 0.30,      # Tightened to prevent string resonance ringing out
        "minimum_note_length_ms": 45,
        "min_amplitude": 0.10,
        "monophonic": False,
    },
    "piano": {
        "min_frequency": 40,
        "max_frequency": 4000,
        "onset_threshold": 0.55,      # Back down from 0.80 so soft velocities aren't lost
        "frame_threshold": 0.30,
        "minimum_note_length_ms": 45,
        "empty_rms_threshold": 0.015,
        "min_amplitude": 0.10,
        "monophonic": False,
    },
    "other": {
        "min_frequency": 80,
        "max_frequency": 4000,
        "onset_threshold": 0.45,
        "frame_threshold": 0.30,
        "minimum_note_length_ms": 50,
        "min_amplitude": 0.10,
        "monophonic": False,
    },
    "drums": {
        # Assuming a librosa.util.peak_pick or custom transient engine:
        "onset_threshold": 0.15,      # Raised drastically to target real hits only
        "minimum_note_length_ms": 25, # Shorter for crisp drum hits
        "min_amplitude": 0.08,
    },
}

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def _configure_tensorflow_once():
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

    try:
        import tensorflow as tf
    except Exception as e:
        raise RuntimeError(f"TensorFlow is required for Omnizart stem/MIDI: {e}") from e

    if not getattr(_configure_tensorflow_once, "_configured", False):
        try:
            print("Omnizart TensorFlow GPUs:", tf.config.list_physical_devices("GPU"))
        except Exception:
            pass

        try:
            tf.debugging.set_log_device_placement(True)
        except Exception:
            pass

        _configure_tensorflow_once._configured = True

    return tf


def _get_omnizart():
    """Lazily import the three Omnizart transcription apps we route stems to.
    Raises immediately if Omnizart (and its TensorFlow dependency) isn't
    installed, so callers can bail out before the slow Demucs run."""
    global _OMNIZART_APPS, OMNIZART_IMPORT_ERROR

    if _OMNIZART_APPS is None:
        try:
            # Import TensorFlow only when we actually need Omnizart.
            _configure_tensorflow_once()

            from omnizart.music import app as omnizart_music_app
            from omnizart.vocal import app as omnizart_vocal_app
            from omnizart.drum import app as omnizart_drum_app

            if not globals().get("SHOW_OMNIZART_LOGS", False):
                try:
                    import logging
                    logging.getLogger("tensorflow").setLevel(logging.ERROR)
                    logging.getLogger("omnizart").setLevel(logging.ERROR)
                    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
                except Exception:
                    pass


            _OMNIZART_APPS = {
                "music": omnizart_music_app,
                "vocal": omnizart_vocal_app,
                "drum": omnizart_drum_app,
            }
        except Exception as e:
            OMNIZART_IMPORT_ERROR = str(e)
            raise RuntimeError(f"Omnizart is not available: {e}")

    return _OMNIZART_APPS


def _release_omnizart_memory():
    """Best-effort release of Omnizart/TensorFlow references after stem/MIDI work.

    This does not guarantee that macOS will immediately show lower RAM, but it
    removes Python-side references and clears the Keras session so the memory is
    at least eligible for reuse/release.
    """
    global _OMNIZART_APPS

    if not UNLOAD_OMNIZART_AFTER_STEM_MIDI or _OMNIZART_APPS is None:
        return

    try:
        _OMNIZART_APPS = None
        gc.collect()

        try:
            import tensorflow as tf
            tf.keras.backend.clear_session()
            gc.collect()
        except Exception:
            pass
    except Exception:
        pass


# Which Omnizart app transcribes each Demucs stem.
OMNIZART_STEM_APP = {
    "vocals": "vocal",
    "bass": "music",
    "guitar": "music",
    "piano": "music",
    "other": "music",
    "drums": "drum",
}


def run_demucs_stems(stem_wav_path: str, out_dir: str):
    base_cmd = [sys.executable, "-m", "demucs", "-n", DEMUCS_MODEL, "-o", out_dir, stem_wav_path]

    attempts = [
        base_cmd + ["-d", "mps"],
        base_cmd + ["--device", "mps"],
        base_cmd,
    ]

    last_err = ""
    result = None
    for cmd in attempts:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            break
        last_err = result.stderr or result.stdout

    if result is None or result.returncode != 0:
        raise RuntimeError(f"Demucs failed:\n{last_err[-2000:]}")

    found = {}
    for root, dirs, files in os.walk(out_dir):
        for f in files:
            if not f.lower().endswith(".wav"):
                continue
            name = os.path.splitext(f)[0].lower()
            if name in STEM_MIDI_STEMS and name not in found:
                found[name] = os.path.join(root, f)

    return found


DEBUG_STEM_MIDI = False  # True = verbose per-stem note-count debugging
# False (default): silence Omnizart/TensorFlow/Keras progress bars and
# per-stem console chatter during stem MIDI. Set True only when debugging
# transcription failures.
SHOW_OMNIZART_LOGS = False


def _pretty_midi_to_note_dicts(midi):
    """Convert an Omnizart pretty_midi.PrettyMIDI result (pitched stems:
    vocals/bass/guitar/piano/other) into this file's internal note-dict
    format: onset/offset in seconds, frequency in Hz, amplitude 0..1."""
    if midi is None:
        return []

    out = []
    for instrument in midi.instruments:
        for note in instrument.notes:
            start = float(note.start)
            end = float(note.end)
            if end <= start:
                continue
            try:
                freq = float(librosa.midi_to_hz(int(note.pitch)))
            except Exception:
                continue
            if not np.isfinite(freq) or freq <= 0:
                continue
            amp = max(0.0, min(1.0, float(note.velocity) / 127.0))
            out.append({
                "onset": start,
                "offset": end,
                "frequency": freq,
                "amplitude": amp,
                "duration": end - start,
            })

    if DEBUG_STEM_MIDI:
        print(f"  (stem MIDI debug: Omnizart returned {len(out)} pitched note(s))")

    return sorted(out, key=lambda x: x["onset"])


def _drum_pretty_midi_to_note_dicts(midi):
    """Convert an Omnizart drum-model pretty_midi.PrettyMIDI result into
    note dicts, each carrying an explicit 'drum_type' (kick/snare/hihat/
    tom/cymbal) rather than a pitch. Omnizart's drum model only
    distinguishes 3 classes; different builds have been seen to encode
    that either via standard GM percussion note numbers on a single
    instrument track, or via separate named instrument tracks. We check
    both, and if neither is recognised, fall back to ranking the distinct
    pitches actually present in this file low-to-high as kick/snare/hihat
    so a hit is never silently mislabeled as a single generic bucket."""
    if midi is None:
        return []

    raw = []
    distinct_pitches = set()
    for instrument in midi.instruments:
        name_guess = _guess_drum_type_from_name(instrument.name)
        for note in instrument.notes:
            start = float(note.start)
            end = float(note.end)
            if end <= start:
                end = start + 0.05
            pitch = int(note.pitch)
            raw.append((start, end, pitch, int(note.velocity), name_guess))
            distinct_pitches.add(pitch)

    if not raw:
        return []

    # Fallback only used when a hit's pitch isn't a recognised GM
    # percussion number AND its instrument track wasn't named.
    sorted_pitches = sorted(distinct_pitches)
    rank_labels = ["kick", "snare", "hihat"]
    pitch_rank_type = {}
    for i, p in enumerate(sorted_pitches):
        bucket = min(i * len(rank_labels) // max(len(sorted_pitches), 1), len(rank_labels) - 1)
        pitch_rank_type[p] = rank_labels[bucket]

    out = []
    for start, end, pitch, velocity, name_guess in raw:
        drum_type = name_guess or GM_DRUM_PITCH_TYPE.get(pitch) or pitch_rank_type.get(pitch, "other")
        amp = max(0.0, min(1.0, float(velocity) / 127.0))
        out.append({
            "onset": start,
            "offset": end,
            "duration": max(end - start, 0.01),
            "amplitude": amp,
            "drum_type": drum_type,
        })

    if DEBUG_STEM_MIDI:
        print(f"  (stem MIDI debug: Omnizart drum model returned {len(out)} hit(s); "
              f"distinct raw pitches={sorted_pitches})")

    return sorted(out, key=lambda x: x["onset"])

def _drum_band_label(freq_hz):
    """Very rough register guess from spectral centroid — not a real pitch."""
    if freq_hz < 150:
        return "low (kick-like)"
    if freq_hz < 800:
        return "mid (snare/tom-like)"
    return "high (hihat/cymbal-like)"


def transcribe_drums_with_onsets(stem_path: str, preset: dict):
    """
    Multi-band onset detection for drums.
    Separate low / mid / high bands so kicks, snares, and hats are not
    competing for one global peak list. 'frequency' is still a band centre
    proxy used only for register labelling — not a musical pitch.
    """
    y, sr = librosa.load(stem_path, sr=None, mono=True)
    if y.size == 0:
        return []

    hop_length = 256
    min_amplitude = float(preset.get("min_amplitude", 0.015))
    min_note_length = float(preset.get("min_note_length", 0.01))
    onset_threshold = float(preset.get("onset_threshold", 0.04))

    # Band-limited onset envelopes approximate kick / snare / hat registers.
    bands = [
        ("low",  30,   150,  80.0),    # kick-like
        ("mid",  150,  800,  300.0),   # snare/tom-like
        ("high", 5000, 12000, 8000.0), # hihat/cymbal-like
    ]

    all_notes = []
    for _name, fmin, fmax, proxy_hz in bands:
        try:
            env = librosa.onset.onset_strength(
                y=y, sr=sr, hop_length=hop_length,
                fmin=fmin, fmax=fmax, aggregate=np.mean,
            )
        except TypeError:
            # Older librosa: no fmin/fmax on onset_strength — use full-band fallback once.
            env = librosa.onset.onset_strength(
                y=y, sr=sr, hop_length=hop_length, aggregate=np.mean,
            )

        max_env = float(np.max(env)) if env.size else 0.0
        if max_env <= 0:
            continue

        env_norm = env / max_env
        frames = librosa.onset.onset_detect(
            onset_envelope=env_norm,
            sr=sr,
            hop_length=hop_length,
            backtrack=True,
            delta=onset_threshold,
            wait=1,
            pre_max=3,
            post_max=3,
            pre_avg=3,
            post_avg=3,
        )
        if len(frames) == 0:
            frames = librosa.onset.onset_detect(
                onset_envelope=env_norm,
                sr=sr,
                hop_length=hop_length,
                backtrack=True,
                delta=max(0.02, onset_threshold * 0.5),
                wait=1,
            )
        if len(frames) == 0:
            continue

        times = librosa.frames_to_time(frames, sr=sr, hop_length=hop_length)
        for idx, frame in enumerate(frames):
            frame_i = int(min(int(frame), len(env) - 1))
            amp = float(env[frame_i]) / max_env
            if amp < min_amplitude:
                continue
            onset_t = float(times[idx])
            next_t = float(times[idx + 1]) if idx + 1 < len(times) else onset_t + 0.10
            dur = max(min_note_length, min(next_t - onset_t, 0.20))
            all_notes.append({
                "onset": onset_t,
                "offset": onset_t + dur,
                "frequency": proxy_hz,
                "amplitude": amp,
                "duration": dur,
            })

    if not all_notes:
        return []

    # Merge near-duplicate hits across bands (same instant counted twice).
    all_notes = sorted(all_notes, key=lambda n: n["onset"])
    merged = [all_notes[0]]
    for n in all_notes[1:]:
        if n["onset"] - merged[-1]["onset"] < 0.02:
            # Keep the louder band label at this instant.
            if n["amplitude"] > merged[-1]["amplitude"]:
                merged[-1] = n
        else:
            merged.append(n)

    return merged

_OMNIZART_OUTPUT_DIRS = []  # temp dirs Omnizart writes its .mid side-output into; cleaned up in main()'s finally block

def omnizart_transcribe(stem_path: str, stem: str, preset: dict):
    """
    Transcribe one separated stem with Omnizart.

    Omnizart's transcribe() API returns a PrettyMIDI object, while also
    writing a MIDI file to the supplied output location.  Different
    Omnizart versions/builds may return a PrettyMIDI object, a MIDI path,
    or occasionally None while still writing the MIDI file.

    This function therefore:
      1. Uses the correct Omnizart app for the stem.
      2. Gives Omnizart an explicit temporary output directory.
      3. Accepts a PrettyMIDI return value directly.
      4. Accepts a returned MIDI path.
      5. If the return value is None/unusable, searches the output directory
         for the MIDI file Omnizart actually wrote.
      6. Converts drum MIDI through the dedicated drum converter.
      7. Converts pitched stems through the normal pitched converter.
    """
    import pretty_midi

    apps = _get_omnizart()

    app_key = OMNIZART_STEM_APP.get(stem, "music")
    app = apps[app_key]

    # Omnizart writes its transcription here.
    out_dir = tempfile.mkdtemp(prefix="omnizart_")
    _OMNIZART_OUTPUT_DIRS.append(out_dir)

    def _load_midi_candidate(candidate):
        """Return PrettyMIDI if candidate is a usable MIDI file/object."""
        if candidate is None:
            return None

        if isinstance(candidate, pretty_midi.PrettyMIDI):
            return candidate

        if isinstance(candidate, (str, os.PathLike)):
            candidate = os.fspath(candidate)

            if os.path.isfile(candidate):
                try:
                    return pretty_midi.PrettyMIDI(candidate)
                except Exception as e:
                    if DEBUG_STEM_MIDI:
                        print(
                            f"  (Omnizart MIDI load failed for "
                            f"{candidate}: {e})"
                        )
                    return None

        return None

    # ------------------------------------------------------------
    # 1. Run Omnizart.
    # ------------------------------------------------------------
    result = None

    # quiet_stdout() swallows the Keras '1/1 [====] - Xs/step' bars that
    # omnizart/TensorFlow print directly (they bypass logging, so
    # logging.disable() alone can't silence them).
    with quiet_stdout():
        try:
            # Omnizart's documented Python API accepts output=...
            result = app.transcribe(
                stem_path,
                output=out_dir,
            )
        except TypeError:
            # Compatibility with Omnizart versions that don't accept
            # the output keyword.
            result = app.transcribe(stem_path)

    # ------------------------------------------------------------
    # 2. Prefer the actual PrettyMIDI object returned by Omnizart.
    # ------------------------------------------------------------
    midi_obj = _load_midi_candidate(result)

    # ------------------------------------------------------------
    # 3. If Omnizart returned a path, load it.
    # ------------------------------------------------------------
    if midi_obj is None and isinstance(result, (str, os.PathLike)):
        midi_obj = _load_midi_candidate(result)

    # ------------------------------------------------------------
    # 4. If the return value wasn't usable, find the MIDI file
    #    Omnizart wrote into its output directory.
    # ------------------------------------------------------------
    if midi_obj is None:
        candidates = []

        for root, dirs, files in os.walk(out_dir):
            for filename in files:
                if filename.lower().endswith((".mid", ".midi")):
                    candidates.append(
                        os.path.join(root, filename)
                    )

        # Prefer the newest MIDI file because Omnizart may leave
        # auxiliary files in the directory.
        candidates.sort(
            key=lambda p: os.path.getmtime(p),
            reverse=True,
        )

        for candidate in candidates:
            midi_obj = _load_midi_candidate(candidate)

            if midi_obj is not None:
                if DEBUG_STEM_MIDI:
                    print(
                        f"  (loaded Omnizart MIDI side-output: "
                        f"{candidate})"
                    )
                break

    # ------------------------------------------------------------
    # 5. Fail clearly if Omnizart produced no usable MIDI.
    # ------------------------------------------------------------
    if midi_obj is None:
        if DEBUG_STEM_MIDI:
            print(
                f"  (Omnizart produced no usable MIDI for "
                f"{stem}: return={type(result).__name__}, "
                f"output_dir={out_dir})"
            )

        return []

    # ------------------------------------------------------------
    # 6. Convert the MIDI according to the stem type.
    # ------------------------------------------------------------
    if stem == "drums":
        notes = _drum_pretty_midi_to_note_dicts(midi_obj)

        if DEBUG_STEM_MIDI:
            print(
                f"  (Omnizart drum MIDI contains "
                f"{len(notes)} converted hit(s))"
            )

        return notes

    notes = _pretty_midi_to_note_dicts(midi_obj)

    if DEBUG_STEM_MIDI:
        print(
            f"  (Omnizart {stem} MIDI contains "
            f"{len(notes)} converted note(s))"
        )

    return notes

def _stem_rms(path, max_seconds=60.0):
    try:
        y, _sr = librosa.load(path, sr=None, mono=True, duration=max_seconds)
        if y.size == 0:
            return 0.0
        return float(np.sqrt(np.mean(y ** 2)))
    except Exception:
        return 0.0


def _summarize_drum_rhythm(filtered, max_pattern_hits=32):
    """Compact groove description from drum hits — no full event list needed.

    Returns lines covering:
      - per-type rates and typical spacing
      - a short pattern sample (types only) sampled across the track
      - a simple kick/snare relationship guess when both are present
    """
    if not filtered:
        return []

    by_type = {}
    for n in filtered:
        t = n.get("drum_type") or "other"
        by_type.setdefault(t, []).append(float(n["onset"]))

    lines = []
    rate_parts = []
    for drum_type, onsets in sorted(by_type.items(), key=lambda kv: -len(kv[1])):
        onsets = sorted(onsets)
        if len(onsets) < 2:
            rate_parts.append(f"{drum_type}: {len(onsets)} hit(s)")
            continue
        intervals = np.diff(onsets)
        med_ibi = float(np.median(intervals))
        rate = 60.0 / med_ibi if med_ibi > 1e-6 else 0.0
        rate_parts.append(
            f"{drum_type}: {len(onsets)} hits, ~{rate:.1f}/min, median spacing {med_ibi:.3f}s"
        )
    if rate_parts:
        lines.append("per-type rhythm: " + "; ".join(rate_parts))

    # Pattern sample: evenly spaced hits across the track, types only.
    ordered = sorted(filtered, key=lambda n: float(n["onset"]))
    sample = _evenly_sample_notes(ordered, max_pattern_hits)
    pattern = " ".join(
        (n.get("drum_type") or _drum_band_label(float(n.get("frequency", 300.0))))
        for n in sample
    )
    if pattern:
        lines.append(
            f"hit pattern sample ({len(sample)} hits across track): {pattern}"
        )

    kicks = by_type.get("kick") or []
    snares = by_type.get("snare") or []
    if len(kicks) >= 4 and len(snares) >= 2:
        kick_ibi = float(np.median(np.diff(sorted(kicks)))) if len(kicks) >= 2 else None
        # Rough backbeat check: snares often sit near midpoint between kicks.
        mid_hits = 0
        for s in snares:
            # find nearest preceding kick
            prev = [k for k in kicks if k <= s]
            nxt = [k for k in kicks if k > s]
            if not prev or not nxt:
                continue
            span = nxt[0] - prev[-1]
            if span <= 0:
                continue
            pos = (s - prev[-1]) / span
            if 0.35 <= pos <= 0.65:
                mid_hits += 1
        if kick_ibi and kick_ibi > 0:
            lines.append(
                f"kick/snare relationship: kick median spacing {kick_ibi:.3f}s; "
                f"~{mid_hits}/{len(snares)} snares near mid-interval (backbeat-like) "
                f"between surrounding kicks"
            )

    return lines


def _stem_activity_label(rms, density, is_drums=False):
    """Rough human-readable activity band from energy + event density."""
    dens_name = "hit density" if is_drums else "note density"
    if rms < 0.01 and density < 5:
        band = "silent/negligible"
    elif rms < 0.03 or density < 15:
        band = "low / background"
    elif rms < 0.08 or density < 60:
        band = "moderate"
    elif rms < 0.15 or density < 150:
        band = "prominent"
    else:
        band = "dominant / very dense"
    return band, dens_name

def _filter_transcribed_notes(notes, preset):
    min_freq = float(preset.get("min_frequency", 0))
    max_freq = float(preset.get("max_frequency", 100000))
    # We are ignoring min_note_length and min_amplitude from the preset here 
    # because they were too aggressive. Instead, we use very low global floors.
    
    filtered = []
    removed = 0

    for n in notes:
        # 1. Frequency Range Check (Keep this to remove sub-bass noise/overtones)
        if not (min_freq <= n["frequency"] <= max_freq):
            removed += 1
            continue
        
        # 2. Duration Check: Allow very short notes (30ms) to catch staccato/fast runs
        if n["duration"] < 0.03: 
            removed += 1
            continue
            
        # 3. Amplitude Check: Set a very low floor (0.05 linear ~ -26dB) 
        # This keeps quiet vocals and soft piano notes while removing pure silence/noise floor
        if n.get("amplitude", 1.0) < 0.05:       
            removed += 1
            continue
            
        filtered.append(n)

    return sorted(filtered, key=lambda x: x["onset"]), removed


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


def _chord_label_from_pcs(pcs):
    pcs = sorted(set(int(p) % 12 for p in pcs if p is not None))
    if not pcs:
        return ""

    if len(pcs) >= 3:
        best_score = 999.0
        best = None
        for root in range(12):
            for quality, intervals in (("major", (0, 4, 7)), ("minor", (0, 3, 7))):
                target = [(root + i) % 12 for i in intervals]
                score = 0.0
                for pc in pcs:
                    d = min(min(abs(pc - t), 12 - abs(pc - t)) for t in target)
                    score += d
                avg = score / len(pcs)
                if avg < best_score:
                    best_score = avg
                    best = (root, quality)

        if best is not None and best_score <= 0.75:
            return f"{NOTE_NAMES[best[0]]} {best[1]}"

    return "+".join(NOTE_NAMES[p] for p in pcs[:6])


def _detect_common_chords(notes, max_count=5):
    if len(notes) < 2:
        return []

    notes = sorted(notes, key=lambda x: x["onset"])[:1500]
    counts = {}
    i = 0

    while i < len(notes):
        cluster = [notes[i]]
        j = i + 1
        while j < len(notes) and notes[j]["onset"] - notes[i]["onset"] <= 0.18:
            cluster.append(notes[j])
            j += 1

        if len(cluster) >= 2:
            pcs = []
            for n in cluster:
                pc = _hz_to_pitch_class(n["frequency"])
                if pc is not None and pc not in pcs:
                    pcs.append(pc)
            if len(pcs) >= 2:
                label = _chord_label_from_pcs(pcs)
                counts[label] = counts.get(label, 0) + 1

        i = j

    top = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:max_count]
    return [(label, count) for label, count in top if count >= 2 or len(top) <= 3]


def _select_monophonic_line(notes):
    if not notes:
        return []

    # Sort by onset time first to ensure we process chronologically
    notes = sorted(notes, key=lambda x: x["onset"])
    
    line = []
    i = 0
    while i < len(notes):
        cluster = [notes[i]]
        j = i + 1
        
        # Group notes that are very close in time (within 50ms) as a "cluster"
        # This handles vibrato and slight pitch tracking jitter
        while j < len(notes) and notes[j]["onset"] - notes[i]["onset"] <= 0.05:
            cluster.append(notes[j])
            j += 1
            
        if not cluster:
            i = j
            continue

        # Strategy for Monophonic (Vocals/Bass):
        # Instead of just picking the loudest, pick the note with the longest duration 
        # within the cluster. This usually represents the sustained pitch better than a transient peak.
        best_note = max(cluster, key=lambda x: x["duration"])
        
        # Optional: If there's a huge amplitude difference, maybe stick to amplitude?
        # But for vocals, duration is often a better proxy for "the note being sung".
        
        line.append(best_note)
        i = j

    return line

def _lowest_line(notes):
    if not notes:
        return []

    line = []
    i = 0
    while i < len(notes):
        cluster = [notes[i]]
        j = i + 1
        while j < len(notes) and notes[j]["onset"] - notes[i]["onset"] <= 0.20:
            cluster.append(notes[j])
            j += 1
        line.append(min(cluster, key=lambda x: x["frequency"]))
        i = j

    return line


def _summarize_melody(notes, max_notes=None):
    if not notes:
        return ""

    if max_notes is None:
        max_notes = STEM_MIDI_MELODY_LINE_NOTES

    # Prefer evenly sampled notes across the full line so long songs still
    # show mid- and late-song contour, not only the intro.
    sample = _evenly_sample_notes(notes, max_notes)
    seq = [_hz_to_note_name(n["frequency"]) for n in sample]
    freqs = np.array([n["frequency"] for n in notes], dtype=float)

    if len(freqs) >= 4:
        mid = len(freqs) // 2
        first = float(np.median(freqs[:mid]))
        second = float(np.median(freqs[mid:]))
        if second > first * 1.06:
            direction = "generally ascending"
        elif second < first / 1.06:
            direction = "generally descending"
        else:
            direction = "mostly stable/mixed"
    else:
        direction = "short line"

    range_note = f"{_hz_to_note_name(float(np.min(freqs)))}-{_hz_to_note_name(float(np.max(freqs)))}"
    sample_note = (
        f"melodic line ({len(seq)} notes sampled across track)"
        if len(notes) > len(seq)
        else f"melodic line ({len(seq)} notes)"
    )
    return (
        f"{sample_note}: {' '.join(seq)}; "
        f"contour: {direction}; pitch range: {range_note}"
    )

def _evenly_sample_notes(notes, max_notes):
    """Return up to max_notes events spread evenly across the FULL list
    (by index, which is chronological), instead of just the earliest ones.
    This matters once transcription covers an entire song: naively slicing
    notes[:max_notes] would silently drop everything after the intro on
    any long or note-dense track. Always keeps the first and last event."""
    if max_notes is None or len(notes) <= max_notes:
        return list(notes)
    if max_notes <= 1:
        return notes[:1]
    step = (len(notes) - 1) / (max_notes - 1)
    idxs = sorted({round(i * step) for i in range(max_notes)})
    return [notes[i] for i in idxs]


def _notes_to_event_log(notes, max_notes=STEM_MIDI_EVENT_LOG_MAX_NOTES):
    """Serialize filtered note events for optional inclusion in the report.
    Sampled evenly across the whole track so full-song coverage survives
    the per-stem event cap. Prefer the compact string form when
    STEM_MIDI_COMPACT_EVENT_FORMAT is True (much smaller than JSON objects)."""
    if not notes:
        return ""
    sampled = _evenly_sample_notes(notes, max_notes)
    if STEM_MIDI_COMPACT_EVENT_FORMAT:
        parts = []
        for n in sampled:
            try:
                name = librosa.hz_to_note(float(n["frequency"]))
            except Exception:
                name = f"{float(n['frequency']):.1f}Hz"
            onset = round(float(n["onset"]), 2)
            offset = round(float(n["offset"]), 2)
            parts.append(f"{name}@{onset}-{offset}")
        return ", ".join(parts)
    rows = []
    for n in sampled:
        try:
            name = librosa.hz_to_note(float(n["frequency"]))
        except Exception:
            name = f"{n['frequency']:.1f}Hz"
        rows.append({
            "onset": round(float(n["onset"]), 3),
            "offset": round(float(n["offset"]), 3),
            "freq_hz": round(float(n["frequency"]), 2),
            "note": name,
            "amp": round(float(n.get("amplitude", 1.0)), 3),
        })
    return json.dumps(rows)


def _drum_events_to_event_log(notes, max_notes=STEM_MIDI_EVENT_LOG_MAX_NOTES):
    """Serialize drum/onset hit events for optional inclusion in the report.
    'drum_type' names the actual class of hit (kick/snare/hihat/tom/cymbal)
    for the Omnizart drums stem. It is not a pitch — the writer model
    should read it as which drum was struck, not as melodic content. The
    'other' stem's onset-detection fallback has no real drum_type, so it
    falls back to a rough low/mid/high register label from the onset's
    band-centre frequency. Sampled evenly across the whole track (see
    _evenly_sample_notes) so full-song coverage survives the event cap."""
    if not notes:
        return ""
    sampled = _evenly_sample_notes(notes, max_notes)
    if STEM_MIDI_COMPACT_EVENT_FORMAT:
        parts = []
        for n in sampled:
            drum_type = n.get("drum_type") or _drum_band_label(float(n.get("frequency", 300.0)))
            onset = round(float(n["onset"]), 2)
            parts.append(f"{drum_type}@{onset}")
        return ", ".join(parts)
    rows = []
    for n in sampled:
        drum_type = n.get("drum_type") or _drum_band_label(float(n.get("frequency", 300.0)))
        rows.append({
            "onset": round(float(n["onset"]), 3),
            "duration": round(float(n["duration"]), 3),
            "velocity": round(float(n.get("amplitude", 1.0)), 3),
            "drum_type": drum_type,
        })
    return json.dumps(rows)

def summarize_stem_midi(stem, raw_notes, filtered, removed, preset, stem_rms=None):
    """Build a compact per-stem summary. Returns (text, meta_dict) where meta
    holds activity stats used for the cross-stem prominence ranking."""
    lines = [f"[{stem.upper()} stem]"]
    meta = {
        "stem": stem,
        "rms": float(stem_rms) if stem_rms is not None else None,
        "note_count": len(filtered) if filtered else 0,
        "density": 0.0,
        "is_drums": stem == "drums",
        "silent": False,
    }

    if stem_rms is not None:
        lines.append(f"stem energy (RMS, first ~60s): {float(stem_rms):.5f}")

    if not raw_notes:
        lines.append("No reliable MIDI notes detected.")
        meta["silent"] = True
        return "\n".join(lines), meta

    total_raw = len(raw_notes) + removed
    removed_ratio = removed / total_raw if total_raw > 0 else 0.0

    quality = "medium"
    if len(filtered) < 10 or removed_ratio > 0.6:
        quality = "low"
    elif len(filtered) >= 30 and removed_ratio < 0.25:
        quality = "high"

    lines.append(
        f"note count after filtering: {len(filtered)} (raw {len(raw_notes)}, filtered out {removed}); "
        f"transcription confidence: {quality}"
    )

    if stem == "drums":
        lines.append(
            "Drum/percussion stem: Omnizart drum transcription (not pitched MIDI). "
            "Use hit-type breakdown, per-type spacing, and the pattern sample below "
            "for rhythm/groove discussion."
        )

    if not filtered:
        lines.append("No notes remained after filtering.")
        meta["silent"] = True
        return "\n".join(lines), meta

    if stem == "drums":
        # Percussion has no real pitch — report type breakdown, density, and
        # a compact rhythm description instead of full hit lists.
        type_counts = {}
        for n in filtered:
            drum_type = n.get("drum_type", "other")
            type_counts[drum_type] = type_counts.get(drum_type, 0) + 1
        type_summary = ", ".join(
            f"{drum_type} ({count})" for drum_type, count in sorted(type_counts.items(), key=lambda kv: -kv[1])
        )
        lines.append(f"hit breakdown by drum type: {type_summary}")

        onset_times = [n["onset"] for n in filtered]
        offsets = sorted(n["offset"] for n in filtered)
        j = 0
        total_active = 0
        max_poly = 0
        for i, t in enumerate(onset_times):
            while j < len(offsets) and offsets[j] <= t:
                j += 1
            active = (i + 1) - j
            if active > max_poly:
                max_poly = active
            total_active += active
        avg_poly = total_active / len(filtered) if filtered else 0.0
        lines.append(f"average simultaneous hits: {avg_poly:.1f}; maximum simultaneous hits: {max_poly}")

        min_onset = float(np.min(onset_times))
        max_offset = float(np.max([n["offset"] for n in filtered]))
        dur = max(0.1, max_offset - min_onset)
        density = len(filtered) / dur * 60.0 if dur > 0 else 0.0
        meta["density"] = density
        lines.append(f"hit density: {density:.1f} hits/min")

        for rhythm_line in _summarize_drum_rhythm(
            filtered, max_pattern_hits=STEM_MIDI_DRUM_PATTERN_HITS
        ):
            lines.append(rhythm_line)

        if stem_rms is not None:
            band, dens_name = _stem_activity_label(stem_rms, density, is_drums=True)
            lines.append(f"activity/prominence: {band} (from RMS + {dens_name})")

        if quality == "low" or removed_ratio > 0.5:
            lines.append("Caution: this stem contains likely spurious/erratic onsets; treat only broad rhythmic tendencies as reliable.")

        if STEM_MIDI_INCLUDE_EVENT_LOGS:
            event_log = _drum_events_to_event_log(filtered, max_notes=STEM_MIDI_EVENT_LOG_MAX_NOTES)
            if event_log:
                label = "hit events (compact)" if STEM_MIDI_COMPACT_EVENT_FORMAT else "hit events (rhythm-JSON)"
                lines.append(f"{label}: {event_log}")

        return "\n".join(lines), meta

    # --- Pitched stems (vocals/bass/guitar/piano/other) ---
    if stem == "other" and not any(
        preset.get("min_frequency", 0) <= n["frequency"] <= preset.get("max_frequency", 100000)
        for n in filtered
    ):
        # Onset-fallback mode: report hit density only, skip pitch stats.
        onset_times = [n["onset"] for n in filtered]
        min_onset = float(np.min(onset_times)) if onset_times else 0
        max_offset = float(np.max([n["offset"] for n in filtered])) if filtered else 0
        dur = max(0.1, max_offset - min_onset)
        density = len(filtered) / dur * 60.0 if dur > 0 else 0.0
        meta["density"] = density
        lines.append(f"onset/hit density: {density:.1f} events/min (no reliable pitched content detected)")
        if stem_rms is not None:
            band, dens_name = _stem_activity_label(stem_rms, density, is_drums=False)
            lines.append(f"activity/prominence: {band} (from RMS + {dens_name})")
        if STEM_MIDI_INCLUDE_EVENT_LOGS:
            event_log = _drum_events_to_event_log(filtered, max_notes=STEM_MIDI_EVENT_LOG_MAX_NOTES)
            if event_log:
                label = "hit events (compact)" if STEM_MIDI_COMPACT_EVENT_FORMAT else "hit events (rhythm-JSON)"
                lines.append(f"{label}: {event_log}")
        return "\n".join(lines), meta

    freqs = np.array([n["frequency"] for n in filtered], dtype=float)
    # Absolute min/max often pick up octave errors, harmonics, or bleed.
    # Prefer a percentile "practical" range for discussion, especially vocals.
    f_min = float(np.min(freqs))
    f_max = float(np.max(freqs))
    p10 = float(np.percentile(freqs, 10))
    p90 = float(np.percentile(freqs, 90))
    lines.append(
        f"pitch range (absolute): {_hz_to_note_name(f_min)} ({round(f_min, 1)} Hz) "
        f"to {_hz_to_note_name(f_max)} ({round(f_max, 1)} Hz)"
    )
    lines.append(
        f"practical pitch range (10–90%): {_hz_to_note_name(p10)} ({round(p10, 1)} Hz) "
        f"to {_hz_to_note_name(p90)} ({round(p90, 1)} Hz) — prefer this for main range; "
        f"absolute extremes often include harmonics/octave errors"
    )

    med = float(np.median(freqs))
    lines.append(f"median pitch: {med:.1f} Hz (~{_hz_to_note_name(med)})")

    pc_counts = {}
    for n in filtered:
        pc = _hz_to_pitch_class(n["frequency"])
        if pc is not None:
            name = NOTE_NAMES[pc]
            pc_counts[name] = pc_counts.get(name, 0) + 1

    top_pitches = ", ".join(f"{p} ({c})" for p, c in sorted(pc_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:8])
    lines.append(f"most common pitch classes: {top_pitches}")

    onset_times = [n["onset"] for n in filtered]
    offsets = sorted(n["offset"] for n in filtered)
    j = 0
    total_active = 0
    max_poly = 0

    for i, t in enumerate(onset_times):
        while j < len(offsets) and offsets[j] <= t:
            j += 1
        active = (i + 1) - j
        if active > max_poly:
            max_poly = active
        total_active += active

    avg_poly = total_active / len(filtered) if filtered else 0.0
    lines.append(f"average simultaneous notes: {avg_poly:.1f}; maximum polyphony: {max_poly}")

    chords = _detect_common_chords(filtered, max_count=5)
    if chords:
        lines.append("common harmonic clusters: " + "; ".join(f"{label} x{count}" for label, count in chords))
    else:
        lines.append("no stable harmonic clusters detected")

    # Monophonic stems (vocals/bass) get the full melodic-line budget;
    # dense poly stems get half — their "lowest line" is less reliable and
    # the long note lists are a common source of token bloat.
    melody_budget = STEM_MIDI_MELODY_LINE_NOTES
    if not preset.get("monophonic"):
        melody_budget = max(12, STEM_MIDI_MELODY_LINE_NOTES // 2)

    if preset.get("monophonic"):
        line_notes = _select_monophonic_line(filtered)
        mel = _summarize_melody(line_notes, max_notes=melody_budget)
        if mel:
            lines.append(mel)
    else:
        low_notes = _lowest_line(filtered)
        mel = _summarize_melody(low_notes, max_notes=melody_budget)
        if mel:
            lines.append("lowest-line contour (not necessarily melody): " + mel)

    min_onset = float(np.min([n["onset"] for n in filtered]))
    max_offset = float(np.max([n["offset"] for n in filtered]))
    dur = max(0.1, max_offset - min_onset)
    density = len(filtered) / dur * 60.0 if dur > 0 else 0.0
    meta["density"] = density
    lines.append(f"note density: {density:.1f} notes/min")

    if stem_rms is not None:
        band, dens_name = _stem_activity_label(stem_rms, density, is_drums=False)
        lines.append(f"activity/prominence: {band} (from RMS + {dens_name})")

    if quality == "low" or stem == "other" or removed_ratio > 0.5:
        lines.append("Caution: this stem contains likely hallucinated/erratic notes; use only broad melodic/harmonic tendencies.")

    if STEM_MIDI_INCLUDE_EVENT_LOGS:
        event_log = _notes_to_event_log(filtered, max_notes=STEM_MIDI_EVENT_LOG_MAX_NOTES)
        if event_log:
            label = "note events (compact)" if STEM_MIDI_COMPACT_EVENT_FORMAT else "note events (MIDI-JSON)"
            lines.append(f"{label}: {event_log}")

    return "\n".join(lines), meta


def build_omnizart_summaries(stems):
    if not stems:
        return "STEM MIDI REPORT unavailable: no Demucs stems found."

    lines = []
    activity_metas = []

    for stem in STEM_MIDI_STEMS:
        path = stems.get(stem)
        if not path or not os.path.exists(path):
            continue

        preset = STEM_MIDI_PRESETS[stem]
        # Always measure stem energy for prominence ranking (cheap vs transcription).
        rms = _stem_rms(path)
        empty_rms = float(preset.get("empty_rms_threshold", 0) or 0)
        if empty_rms > 0 and rms < empty_rms:
            lines.append(
                f"[{stem.upper()} stem]\n"
                f"Stem energy is very low (RMS≈{rms:.5f}); treating as effectively silent. "
                f"Demucs often leaves a quiet residual even when this instrument is absent."
            )
            activity_metas.append({
                "stem": stem,
                "rms": rms,
                "note_count": 0,
                "density": 0.0,
                "is_drums": stem == "drums",
                "silent": True,
            })
            continue

        try:
            # --- ROUTING LOGIC ---

            if stem == "drums":
                # Try Omnizart's dedicated drum model first — real kick/snare/hihat/tom/cymbal
                # classification, not just a register guess. Fall back to the onset detector
                # only if Omnizart fails or returns nothing for this stem.
                status(f"Transcribing drums with Omnizart's drum model: {os.path.basename(path)}")

                raw_notes = omnizart_transcribe(path, stem, preset)

                if not raw_notes:
                    if SHOW_OMNIZART_LOGS:
                        print(
                            "  (Omnizart returned no hits for the drums stem; "
                            "falling back to the multiband onset detector)"
                        )
                    raw_notes = transcribe_drums_with_onsets(
                        path,
                        preset
                    )
                elif SHOW_OMNIZART_LOGS:
                    print(
                        f"  (Omnizart drum model returned "
                        f"{len(raw_notes)} classified hit(s))"
                    )

            elif stem == "other":
                # Try Omnizart's music model first; fall back to onset detection.
                raw_notes = omnizart_transcribe(path, stem, preset)

                if not raw_notes:
                    if SHOW_OMNIZART_LOGS:
                        print(
                            "  (Omnizart returned no notes for 'other' stem; "
                            "falling back to onset detection)"
                        )
                    raw_notes = transcribe_drums_with_onsets(
                        path,
                        preset
                    )

            else:
                # vocals -> Omnizart vocal model
                # bass/guitar/piano -> Omnizart music model
                raw_notes = omnizart_transcribe(
                    path,
                    stem,
                    preset
                )

            # --- FILTERING LOGIC ---
            if stem == "drums":
                filtered = [n for n in raw_notes if n.get("onset", 0) >= 0]
                removed = len(raw_notes) - len(filtered)
            elif stem == "other" and not any(
                preset["min_frequency"] <= n["frequency"] <= preset["max_frequency"]
                for n in raw_notes
            ):
                # Onset-fallback path: skip pitch-range filter, keep only valid onsets.
                filtered = [n for n in raw_notes if n.get("onset", 0) > 0 and n.get("duration", 0) > 0]
                removed = len(raw_notes) - len(filtered)
            else:
                filtered, removed = _filter_transcribed_notes(raw_notes, preset)

            summary, meta = summarize_stem_midi(
                stem, raw_notes, filtered, removed, preset, stem_rms=rms
            )
            activity_metas.append(meta)
        except Exception as e:
            summary = f"[{stem.upper()} stem] MIDI transcription failed: {e}"
            activity_metas.append({
                "stem": stem,
                "rms": rms,
                "note_count": 0,
                "density": 0.0,
                "is_drums": stem == "drums",
                "silent": True,
            })

        lines.append(summary)

    if not lines:
        return "STEM MIDI REPORT unavailable: no stems could be transcribed."

    # Cross-stem prominence ranking (energy + event density) — small but useful
    # for "which instruments dominate" questions without raw event dumps.
    if activity_metas:
        ranked = sorted(
            activity_metas,
            key=lambda m: (
                0 if m.get("silent") else 1,
                float(m.get("rms") or 0.0),
                float(m.get("density") or 0.0),
            ),
            reverse=True,
        )
        rank_parts = []
        for i, m in enumerate(ranked, 1):
            rms_v = m.get("rms")
            dens = float(m.get("density") or 0.0)
            rms_s = f"{rms_v:.5f}" if rms_v is not None else "n/a"
            if m.get("silent"):
                band = "silent/negligible"
            else:
                band, _ = _stem_activity_label(
                    float(rms_v or 0.0), dens, is_drums=bool(m.get("is_drums"))
                )
            dens_unit = "hits/min" if m.get("is_drums") else "notes/min"
            rank_parts.append(
                f"{i}) {m['stem']}: {band} (RMS={rms_s}, density={dens:.1f} {dens_unit}, "
                f"events={m.get('note_count', 0)})"
            )
        lines.append(
            "STEM ACTIVITY / PROMINENCE (ranked by stem energy then event density; "
            "use for relative instrument balance, not absolute loudness):\n"
            + "\n".join(rank_parts)
        )

    coverage_note = (
        "full track" if not STEM_MIDI_MAX_SECONDS
        else f"first ~{int(STEM_MIDI_MAX_SECONDS)} seconds"
    )
    if STEM_MIDI_INCLUDE_EVENT_LOGS:
        event_note = (
            "Optional compact note/hit event samples are included per stem (evenly sampled "
            "end-to-end). "
            if STEM_MIDI_COMPACT_EVENT_FORMAT else
            "Optional JSON note/hit event samples are included per stem (evenly sampled "
            "end-to-end). "
        )
    else:
        event_note = (
            "Raw note/hit event lists are omitted to keep context small; rely on aggregate "
            "stats, melodic-line samples, drum pattern samples, and the STEM ACTIVITY / "
            "PROMINENCE ranking for melody, groove, and instrument-balance discussion. "
        )
    header = (
        f"STEM MIDI REPORT (Demucs 6s + Omnizart, {coverage_note}, all 6 stems: "
        "vocals/bass/guitar/piano/other/drums)\n"
        "This is a probabilistic audio-to-MIDI transcription of separated stems — "
        "not a Logic Pro project or a standard MIDI file. Aggregate stats (pitch range, "
        "density, chords, drum-type breakdown, rhythm pattern samples, melodic-line "
        "samples, stem energy/prominence, etc.) are computed from ALL detected notes/hits "
        "for the whole track. "
        + event_note
        + "Use it for casual melody/harmony/composition/rhythm discussion; be cautious "
        "with erratic short notes and low-confidence stems.\n"
    )

    return header + "\n\n".join(lines)


# --- Save/load helpers ------------------------------------------------------
def _sanitize_saved_name(name):
    name = (name or "").strip().strip('"').strip("'")
    if not name:
        return ""

    base = os.path.basename(name)
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", base).strip("._")
    if len(safe) > 120:
        safe = safe[:120]
    return safe



def _batch_save_basename(audio_path):
    """Keep the original audio basename; only swap the extension to .json.

    e.g. '01 - December.m4a' → '01 - December.json'
    Characters that are illegal in filenames on the host OS are still stripped.
    """
    base = os.path.basename(audio_path or "")
    stem, _ext = os.path.splitext(base)
    if not stem:
        stem = f"song_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    # Remove path separators / nulls only — preserve spaces and most punctuation.
    stem = stem.replace("/", "_").replace("\\", "_").replace("\x00", "")
    if len(stem) > 180:
        stem = stem[:180]
    return stem + ".json"


def save_song_data(
    filename,
    track_key,
    analysis,
    corrections,
    metadata=None,
    cover_observations=None,
    singer_identity=None,
    cover_bytes=None,
    cover_mime="image/jpeg",
    preserve_audio_basename=False,
):
    os.makedirs(SAVE_DIR, exist_ok=True)

    if preserve_audio_basename and track_key and not str(track_key).startswith(("http://", "https://")):
        safe = _batch_save_basename(track_key)
    else:
        safe = _sanitize_saved_name(filename) or f"song_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        if not safe.lower().endswith(".json"):
            safe += ".json"

    path = os.path.join(SAVE_DIR, safe)

    lyrics_text = ""
    if metadata:
        lyrics_text = str(metadata.get("lyrics") or "").strip()

    # Write the prepared cover art next to the JSON so saves are self-contained.
    cover_path = None
    if cover_bytes:
        stem_name = safe[:-5] if safe.lower().endswith(".json") else safe
        img_suffix = _cover_temp_suffix(cover_mime or _guess_image_mime(cover_bytes))
        cover_path = os.path.join(SAVE_DIR, stem_name + img_suffix)
        try:
            with open(cover_path, "wb") as f:
                f.write(cover_bytes)
        except Exception as e:
            print(f"  (could not write cover art to save: {e})")
            cover_path = None

    data = {
        "version": 1,
        "track_path": track_key,
        "label": track_label(track_key),
        "saved_at": datetime.datetime.now().isoformat(),
        "analysis": analysis,
        "corrections": corrections or {},
        "metadata": metadata or {},
        "lyrics": lyrics_text,
        "cover_observations": cover_observations or {},
        "singer_identity": singer_identity or "",
        "cover_art": os.path.basename(cover_path) if cover_path else "",
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    return path, cover_path


def load_song_data(filename):
    os.makedirs(SAVE_DIR, exist_ok=True)

    safe = _sanitize_saved_name(filename)
    if not safe:
        return None

    candidates = [os.path.join(SAVE_DIR, safe)]
    if not safe.lower().endswith(".json"):
        candidates.append(os.path.join(SAVE_DIR, safe + ".json"))

    path = None
    for cand in candidates:
        if os.path.exists(cand):
            path = cand
            break

    if path is None:
        target = safe[:-5].lower() if safe.lower().endswith(".json") else safe.lower()
        match = None
        try:
            for f in os.listdir(SAVE_DIR):
                if not f.lower().endswith(".json"):
                    continue
                stem_name = f[:-5].lower()
                if stem_name == target or f.lower() == safe.lower():
                    match = os.path.join(SAVE_DIR, f)
                    break
        except Exception:
            pass

        if match is None:
            return None
        path = match

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        raise RuntimeError(f"Could not read saved song file {path}: {e}")

    if not isinstance(data, dict):
        return None

    return data


def _command_remainder(text, flag):
    t = text.strip()
    if not t.lower().startswith(flag):
        return None
    rest = t[len(flag):]
    if rest and rest[0] not in ("=", " ", "\t"):
        return None
    return rest.lstrip("=").strip()


def list_audio_files_in_folder(folder):
    """Non-recursive listing of audio files in folder (stable sorted order)."""
    folder = os.path.abspath(os.path.expanduser(folder))
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Not a directory: {folder}")
    out = []
    for name in sorted(os.listdir(folder)):
        path = os.path.join(folder, name)
        if not os.path.isfile(path):
            continue
        ext = os.path.splitext(name)[1].lower()
        if ext in AUDIO_EXTENSIONS:
            out.append(path)
    return out


def run_fresh_track_analysis(
    track_path,
    *,
    audio_temp_files,
    dsp_temp_files,
    stem_temp_files,
    demucs_out_dirs,
):
    """
    Full /listen-quality analysis for one local file. Does not touch writer_history
    or session token counters. Returns a dict ready for save_song_data.
    """
    track_path = os.path.abspath(os.path.expanduser(track_path))
    if not os.path.exists(track_path):
        raise FileNotFoundError(f"File not found: {track_path}")

    metadata = {}
    cover_b64 = None
    cover_bytes_for_save = None
    cover_mime = "image/jpeg"
    cover_observations = {}
    singer_identity = ""

    if ENABLE_FILE_METADATA:
        status("Reading file metadata/tags...")
        try:
            meta, cover_bytes, cover_mime = extract_audio_metadata(track_path)
            metadata = meta or {}
            if cover_bytes:
                prepared = prepare_cover_image_for_ollama(cover_bytes, cover_mime)
                if prepared:
                    cover_bytes_for_save = prepared
                    cover_b64 = base64.b64encode(prepared).decode("utf-8")
                    cover_mime = _guess_image_mime(prepared)
        except Exception as e:
            print(f"  (metadata/cover extraction skipped: {e})")

    if (
        ENABLE_COVER_ART_DESCRIPTION
        and cover_b64
        and cover_b64 != NO_COVER_SENTINEL
    ):
        status("Describing embedded cover art...")
        try:
            obs = describe_cover_art(cover_b64)
            if obs:
                cover_observations = obs
        except Exception as e:
            print(f"  (cover description skipped: {e})")

    ext = os.path.splitext(track_path)[1].lower()
    if ext in (".wav", ".flac"):
        resolved_path = track_path
    elif track_path in audio_temp_files:
        resolved_path = audio_temp_files[track_path]
    else:
        status(f"Converting {ext} to WAV for compatibility...")
        resolved_path = convert_to_wav(track_path)
        audio_temp_files[track_path] = resolved_path

    ollama_unload_model()
    mf_model, mf_processor, _mf_device = get_music_flamingo()

    status("Listening — estimating era...")
    era_conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": ERA_ANALYSIS_PROMPT},
                {"type": "audio", "path": resolved_path},
            ],
        }
    ]
    era_result = mf_generate(mf_model, mf_processor, era_conversation, max_new_tokens=1024)

    status("Listening — running full analysis (this can take a while)...")
    main_prompt = MF_FULL_ANALYSIS_PROMPT + (MF_DEEP_MODE_ADDENDUM if DEEP_MODE else "")
    main_max_tokens = 3072 if DEEP_MODE else 2048
    mf_conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": main_prompt},
                {"type": "audio", "path": resolved_path},
            ],
        }
    ]
    first_pass = mf_generate(mf_model, mf_processor, mf_conversation, max_new_tokens=main_max_tokens)

    objective_report = ""
    vocal_objective_report = ""
    essentia_report = ""
    dsp_path = None
    if track_path in dsp_temp_files:
        dsp_path = dsp_temp_files[track_path]
    else:
        status("Preparing higher-rate WAV for signal processing...")
        try:
            dsp_path = convert_to_wav(track_path, sample_rate=22050)
            dsp_temp_files[track_path] = dsp_path
        except Exception:
            dsp_path = track_path if ext in (".wav", ".flac") else resolved_path

    if dsp_path is not None:
        if ENABLE_OBJECTIVE_AUDIO_REPORT:
            status("Measuring beat/timbre with signal processing...")
            objective_report = build_objective_audio_report(dsp_path)
        if ENABLE_VOCAL_OBJECTIVE_REPORT:
            status("Measuring vocal pitch/formant proxies...")
            vocal_objective_report = build_vocal_objective_report(dsp_path)
        if ENABLE_ESSENTIA_REPORT and ESSENTIA_AVAILABLE:
            status("Measuring tempo/key/spectral features with Essentia...")
            essentia_report = build_essentia_report(dsp_path)

    vocal_result = ""
    confirmation_result = ""
    initial_lead = ""
    initial_backing = ""
    confirm_lead = ""
    confirm_confidence = ""
    final_lead = ""
    vocal_pitch = {"median": None, "low": None, "high": None, "note": None}
    median_f0 = None
    no_clear_vocals = False

    if ENABLE_VOCAL_PASS:
        status("Listening — estimating singer profile...")
        vocal_prompt_text = VOCAL_ANALYSIS_PROMPT
        if vocal_objective_report:
            vocal_prompt_text += (
                "\n\nObjective vocal measurements (evidence only, not proof):\n"
                f"{vocal_objective_report}"
            )
        vocal_conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": vocal_prompt_text},
                    {"type": "audio", "path": resolved_path},
                ],
            }
        ]
        vocal_result = mf_generate(mf_model, mf_processor, vocal_conversation, max_new_tokens=768)
        if vocal_result:
            initial_lead, initial_backing = parse_vocal_tags(vocal_result)
        vocal_pitch = extract_vocal_pitch_summary(vocal_objective_report) if vocal_objective_report else vocal_pitch
        median_f0 = vocal_pitch.get("median")
        vocals_present_match = re.search(
            r"VOCALS PRESENT\s*[-–—:]?\s*(yes|no|uncertain|instrumental)",
            vocal_result or "",
            re.IGNORECASE,
        )
        no_clear_vocals = bool(
            vocals_present_match and vocals_present_match.group(1).lower() in ("no", "instrumental")
        )
        if ENABLE_VOCAL_CONFIRMATION_PASS and vocal_result and not no_clear_vocals:
            should_confirm = False
            if initial_lead in UNCERTAIN_YOUNG_CATEGORIES:
                should_confirm = True
            elif initial_lead in FEMALE_LEAD_CATEGORIES and (
                (median_f0 is not None and median_f0 >= VOCAL_CONFIRMATION_F0_THRESHOLD)
                or (median_f0 is None and VOCAL_CONFIRMATION_WITHOUT_F0)
            ):
                should_confirm = True
            if should_confirm:
                status("Listening — confirming lead voice category...")
                confirmation_conversation = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": VOCAL_CONFIRMATION_PROMPT},
                            {"type": "audio", "path": resolved_path},
                        ],
                    }
                ]
                confirmation_result = mf_generate(
                    mf_model, mf_processor, confirmation_conversation, max_new_tokens=256
                )
                if confirmation_result:
                    confirm_lead, confirm_confidence = parse_vocal_confirmation(confirmation_result)
        if vocal_result:
            final_lead = choose_final_vocal_lead(initial_lead, confirm_lead, confirm_confidence)

    if FAST_MODE:
        revised = first_pass
    else:
        status("Double-checking its own analysis for overconfident claims...")
        self_check_text = SELF_CHECK_PROMPT
        if vocal_result or confirmation_result:
            self_check_text += "\n\nVocal profile evidence:\n"
            if vocal_result:
                self_check_text += f"{vocal_result}\n"
            if confirmation_result:
                self_check_text += f"{confirmation_result}\n"
        objective_crosscheck_parts = []
        if essentia_report:
            objective_crosscheck_parts.append(essentia_report)
        if objective_report:
            objective_crosscheck_parts.append(objective_report)
        if objective_crosscheck_parts:
            self_check_text += (
                "\n\nIndependent signal-processing cross-checks "
                "(use only for tempo/beat, key/key strength when explicitly reported, timbre, "
                "element activity, and dynamic range; do NOT use them to change GENRE or vocal identity):\n"
                + "\n\n".join(objective_crosscheck_parts)
            )
        mf_conversation.append(
            {"role": "assistant", "content": [{"type": "text", "text": first_pass}]}
        )
        mf_conversation.append(
            {"role": "user", "content": [{"type": "text", "text": self_check_text}]}
        )
        revised = mf_generate(mf_model, mf_processor, mf_conversation, max_new_tokens=main_max_tokens)

    tag_lyrics = str((metadata or {}).get("lyrics") or "").strip()
    skip_mf_lyrics = (
        SKIP_MF_LYRICS_WHEN_TAGS_PRESENT
        and len(tag_lyrics) >= METADATA_LYRICS_MIN_CHARS_TO_SKIP_MF
    )
    if skip_mf_lyrics:
        status("Skipping MF lyrics pass (file-tag lyrics already present)...")
        revised += (
            "\n\nFULL LYRICS TRANSCRIPTION: skipped — file-tag lyrics are present "
            "and treated as authoritative. Use TRACK METADATA lyrics for quotes."
        )
    else:
        status("Transcribing full lyrics in a separate dedicated pass...")
        lyrics_conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": LYRICS_TRANSCRIPTION_PROMPT},
                    {"type": "audio", "path": resolved_path},
                ],
            }
        ]
        full_lyrics = mf_generate(
            mf_model, mf_processor, lyrics_conversation,
            max_new_tokens=1536,
            repetition_penalty=1.45,
            no_repeat_ngram_size=12,
        )
        full_lyrics = _sanitize_lyrics_transcription(full_lyrics)
        if full_lyrics and full_lyrics.strip():
            revised += f"\n\nFULL LYRICS TRANSCRIPTION (dedicated pass):\n{full_lyrics}"

    unload_music_flamingo()

    revised += f"\n\n11. ERA / RELEASE PERIOD (isolated dedicated pass):\n{era_result}"
    if vocal_result:
        revised += f"\n\nVOCAL / SINGER PROFILE (isolated dedicated pass):\n{vocal_result}"
        if confirmation_result:
            revised += f"\n\nVOCAL CONFIRMATION PASS:\n{confirmation_result}"
        f0_text = f"{median_f0} Hz" if median_f0 is not None else "unavailable"
        revised += (
            "\n\nVOCAL DECISION AUDIT (audio-only evidence for vocal age/gender):\n"
            f"- Initial isolated lead profile: {initial_lead or 'unparsed'}\n"
            f"- Confirmation pass: {confirm_lead or 'not run/unparsed'} (confidence={confirm_confidence or 'n/a'})\n"
            f"- Objective median f0: {f0_text}\n"
            f"- FINAL LEAD PROFILE: {final_lead or 'unknown'}\n"
            f"- BACKING PROFILES: {initial_backing or 'uncertain'}\n"
            "This is audio-only evidence. If a SINGER IDENTITY RESOLUTION block appears later in this analysis, use that for user-facing singer-identity claims; otherwise use FINAL LEAD PROFILE. Do not override a well-supported combined judgment with pitch impressions alone."
        )

    if ENABLE_VOCAL_OBJECTIVE_REPORT:
        pitch_lines = ["VOCAL PITCH REPORT (independent of lead age/gender category):"]
        if median_f0 is None:
            pitch_lines.append("- No reliable voiced pitch detected in the scanned portion.")
        else:
            pitch_lines.append(f"- objective voiced pitch median: {median_f0} Hz")
            if vocal_pitch.get("note"):
                pitch_lines.append(f"- approximate median note: {vocal_pitch['note']}")
            if vocal_pitch.get("low") is not None and vocal_pitch.get("high") is not None:
                pitch_lines.append(
                    f"- 5-95 percentile range: {vocal_pitch['low']}-{vocal_pitch['high']} Hz"
                )
            pitch_lines.append(
                "- Prefer median + 5–95 percentile range for the main vocal range. "
                "Do not treat absolute extremes from stem MIDI as the sung range."
            )
        revised += "\n\n" + "\n".join(pitch_lines)

    mf_bpm_val = extract_bpm_from_text(first_pass)
    revised_bpm_val = extract_bpm_from_text(revised)
    if revised_bpm_val is not None:
        mf_bpm_val = revised_bpm_val
    essentia_bpm_val = extract_essentia_bpm(essentia_report) if essentia_report else None
    objective_bpm_val = extract_objective_bpm(objective_report) if objective_report else None
    final_bpm, bpm_note = reconcile_bpm(mf_bpm_val, essentia_bpm_val, objective_bpm_val)
    if final_bpm:
        revised += (
            f"\n\nRECOMMENDED TEMPO FOR DISCUSSION: {final_bpm} BPM. "
            f"Reasoning: {bpm_note}. "
            "This is the primary tempo to report to the user. "
            "State it as a concrete figure (e.g. 'about 158 BPM'). "
            "Do not expand it into a range unless this block itself marks the value as uncertain."
        )

    measurement_parts = []
    if objective_report:
        measurement_parts.append(objective_report)
    if vocal_objective_report:
        measurement_parts.append(vocal_objective_report)
    if essentia_report:
        measurement_parts.append(essentia_report)
    if measurement_parts:
        revised += (
            "\n\n[Independent signal-processing report:\n"
            + "\n\n".join(measurement_parts)
            + "\nThis is computed directly from the audio (librosa/Essentia), not from the model above — "
            "use it only for tempo/beat, key/key strength when explicitly reported, timbre, "
            "low/mid/high element activity, dynamic range, and vocal pitch/formant proxies. "
            "Do NOT use it to infer or revise GENRE."
        )

    stem_midi_report = ""
    if ENABLE_STEM_MIDI:
        try:
            _get_omnizart()
            if track_path in stem_temp_files:
                stem_wav = stem_temp_files[track_path]
            else:
                status("Preparing stereo WAV for Demucs/MIDI...")
                stem_wav = convert_to_wav_for_stems(
                    track_path,
                    sample_rate=44100,
                    channels=2,
                    max_seconds=STEM_MIDI_MAX_SECONDS,
                )
                stem_temp_files[track_path] = stem_wav
            status("Running Demucs 6s stem separation (this can be slow)...")
            out_dir = tempfile.mkdtemp(prefix="demucs_")
            demucs_out_dirs.append(out_dir)
            stems = run_demucs_stems(stem_wav, out_dir)
            if not stems:
                stem_midi_report = "STEM MIDI REPORT unavailable: Demucs did not produce expected stems."
            else:
                status("Running Omnizart on each separated stem...")
                stem_midi_report = build_omnizart_summaries(stems)
                _release_omnizart_memory()
        except Exception as e:
            print(f"  (stem MIDI skipped/unavailable: {e})")
            stem_midi_report = f"STEM MIDI REPORT unavailable: {e}"
    stem_midi_report = stem_midi_report or "STEM MIDI REPORT not run (disabled)."
    revised += "\n\n" + stem_midi_report

    if ENABLE_SINGER_IDENTITY_RESOLUTION and (metadata or cover_observations):
        f0_text_res = f"{median_f0} Hz" if median_f0 is not None else "unavailable"
        vocal_audit_for_resolution = (
            f"Initial isolated lead profile: {initial_lead or 'unparsed'}\n"
            f"Confirmation pass: {confirm_lead or 'not run/unparsed'} (confidence={confirm_confidence or 'n/a'})\n"
            f"FINAL LEAD PROFILE: {final_lead or 'unknown'}\n"
            f"BACKING PROFILES: {initial_backing or 'uncertain'}\n"
            f"objective median f0: {f0_text_res}"
        )
        if no_clear_vocals:
            vocal_audit_for_resolution = "No clear vocals detected.\n" + vocal_audit_for_resolution
        status("Resolving singer identity from audio + metadata + cover art...")
        try:
            singer_identity = resolve_singer_identity(
                metadata,
                vocal_audit_for_resolution,
                cover_observations,
                {},
            ) or ""
        except Exception as e:
            print(f"  (singer identity skipped: {e})")
            singer_identity = ""
        cover_obs_block = (
            _format_cover_observation_block(cover_observations) if cover_observations else ""
        )
        if cover_obs_block and "COVER ART OBSERVATIONS" not in revised:
            revised += "\n\n" + cover_obs_block
        if singer_identity and "SINGER IDENTITY RESOLUTION" not in revised:
            revised += (
                f"\n\nSINGER IDENTITY RESOLUTION (combined audio + metadata + cover art; "
                f"use for who-is-singing questions):\n{singer_identity}"
            )
        resolved_tag = _parse_singer_identity(singer_identity) if singer_identity else ""
        priority_tag = resolved_tag if resolved_tag in VOCAL_LEAD_TAGS else final_lead
        if vocal_result or singer_identity:
            revised += build_vocal_priority_note(priority_tag, initial_backing or "uncertain")

    revised = _collapse_runaway_chord_repetition(revised)
    status_done("Analysis complete")

    return {
        "track_path": track_path,
        "analysis": revised,
        "corrections": {},
        "metadata": metadata,
        "cover_observations": cover_observations,
        "singer_identity": singer_identity,
        "cover_bytes": cover_bytes_for_save,
        "cover_mime": cover_mime,
        "stem_midi_report": stem_midi_report,
    }


def batch_scan_folder_to_saved_songs(
    folder,
    *,
    audio_temp_files,
    dsp_temp_files,
    stem_temp_files,
    demucs_out_dirs,
    skip_existing=True,
):
    """
    Scan every audio file in folder with the same pipeline as /listen and write
    JSON (+ cover art) into SAVE_DIR. Does not import anything into chat history.
    """
    files = list_audio_files_in_folder(folder)
    if not files:
        print(f"  No audio files found in {folder}\n")
        return

    os.makedirs(SAVE_DIR, exist_ok=True)
    print(
        f"  Batch scan: {len(files)} track(s) in {folder}\n"
        f"  Saving to {os.path.abspath(SAVE_DIR)} (not imported into chat)\n"
    )

    ok, skipped, failed = 0, 0, 0
    for i, path in enumerate(files, 1):
        save_name = _batch_save_basename(path)
        dest = os.path.join(SAVE_DIR, save_name)
        label = os.path.basename(path)
        print(f"\n  [{i}/{len(files)}] {label}")

        if skip_existing and os.path.exists(dest):
            print(f"  (skip — already saved as {save_name})")
            skipped += 1
            continue

        try:
            result = run_fresh_track_analysis(
                path,
                audio_temp_files=audio_temp_files,
                dsp_temp_files=dsp_temp_files,
                stem_temp_files=stem_temp_files,
                demucs_out_dirs=demucs_out_dirs,
            )
            out_path, cover_path = save_song_data(
                save_name,
                result["track_path"],
                result["analysis"],
                result.get("corrections") or {},
                metadata=result.get("metadata"),
                cover_observations=result.get("cover_observations"),
                singer_identity=result.get("singer_identity"),
                cover_bytes=result.get("cover_bytes"),
                cover_mime=result.get("cover_mime") or "image/jpeg",
                preserve_audio_basename=True,
            )
            print(f"  ✓ saved {os.path.basename(out_path)}"
                  + (f" + {os.path.basename(cover_path)}" if cover_path else ""))
            ok += 1
        except Exception as e:
            failed += 1
            print(f"  ✗ failed: {e}")
            try:
                unload_music_flamingo()
            except Exception:
                pass
            try:
                _release_omnizart_memory()
            except Exception:
                pass

    print(
        f"\n  Batch done — saved: {ok}, skipped: {skipped}, failed: {failed}\n"
        f"  Chat context was not modified. Use /load <name>.json to discuss a saved track.\n"
    )


def main():
    print_logo()

    check_ollama_running()

    if ENABLE_ESSENTIA_REPORT and not ESSENTIA_AVAILABLE:
        print(
            "  (Essentia is enabled but could not be imported; Essentia report will be skipped. "
            f"{ESSENTIA_IMPORT_ERROR})"
        )

    # Music Flamingo is NOT loaded here. It's large, and most turns in a
    # session (general chat, cached-track questions) never touch it, so it's
    # loaded lazily the first time a fresh audio analysis actually needs it
    # (see get_music_flamingo()/unload_music_flamingo() below), and released
    # again as soon as that analysis finishes. mf_model/mf_processor start
    # unset and are only populated inside the /listen fresh-analysis branch.
    mf_model = mf_processor = None
    mf_device = _mf_planned_device()

    # Per-track state, keyed by the raw path/URL as the user referenced it
    comprehensive_analyses = {}   # raw_path -> cached full analysis text from Music Flamingo
    track_corrections = {}        # raw_path -> {field: user-confirmed value}
    audio_temp_files = {}         # raw_path -> converted temp .wav path (cached so we don't reconvert)
    dsp_temp_files = {}
    stem_temp_files = {}          # raw_path -> stereo WAV used for Demucs/MIDI
    demucs_out_dirs = []          # temporary Demucs output directories to clean up later
    track_metadata = {}       # raw_path -> file-tag metadata dict
    track_cover_b64 = {}      # raw_path -> base64 cover art for Ollama, or None
    track_cover_observations = {}  # raw_path -> structured visual observations from cover art
    track_singer_identity = {}     # raw_path -> singer identity resolution text
    track_stem_midi_report = {}    # raw_path -> stem/MIDI evidence for that exact track
    # raw_path -> the exact writer_history message dict that carries this track's full
    # evidence block. Used (via identity + trailing-size check, see
    # _evidence_message_still_safe below) to decide whether it's safe to send a short
    # pointer instead of re-sending the full evidence on a follow-up question.
    track_evidence_message = {}
    last_writer_message = None     # most recent message sent to the writer model
    

    current_track = sys.argv[1] if len(sys.argv) > 1 else None
    last_scanned_track = None

    if current_track and not current_track.startswith(("http://", "https://")):
        if not os.path.exists(current_track):
            print(f"File not found: {current_track}")
            sys.exit(1)

    # Active chat persona (voice only). Music evidence rules stay fixed underneath.
    active_persona_text = DEFAULT_PERSONA_PROMPT
    active_persona_label = "default"
    writer_history = [{"role": "system", "content": build_writer_system_prompt(active_persona_text)}]

    print(_colorize(
        f"\nReady. Musiclyse uses {OLLAMA_MODEL} (writer) and {MF_MODEL_ID} "
        f"(listener, on {mf_device}).\n"
        f"Both models load lazily and are swapped in/out of memory as needed, "
        f"rather than staying resident for the whole session.\n"
        f"{OLLAMA_MODEL} chats directly for general questions.\n"
        f"'{LISTEN_FLAG} <question>' gets a full 10-category analysis of the current track "
        f"(cached after the first listen) and answers your question from it.\n"
        f"'{LISTEN_FLAG} <path or URL> <question>' switches tracks first.\n"
        f"'{RELISTEN_FLAG} [<path or URL>] <question>' forces a fresh re-analysis instead of using the cache.\n"
        f"'{CORRECT_FLAG} field=value[, field=value...]' records a confirmed fact for the current "
        f"track (e.g. '{CORRECT_FLAG} year=1966') that overrides the analysis from then on.\n"
        f"'{SAVE_FLAG}=filename.json' saves technical details for the most recently scanned track to {SAVE_DIR}/.\n"
        f"'{LOAD_FLAG} filename.json [question]' loads a saved song; optional question answered after load.\n"
        f"'{BATCH_FLAG} /path/to/folder' overnight-scans every audio file in that folder into {SAVE_DIR}/ "
        f"(same analysis as /listen; does NOT import into chat). Skips files already saved.\n"
        f"'{CLEAR_FLAG}' wipes chat context and resets session token counters "
        f"(track analysis cache is kept so you don't re-scan).\n"
        f"'{PERSONA_FLAG} <description>' sets a custom chat persona (taste/voice); music evidence rules stay.\n"
        f"'{PERSONA_FLAG} reset' (or default) restores the music-obsessed friend. "
        f"'{PERSONA_FLAG}' alone shows the current persona.\n"
        + (f"Starting track: {current_track}\n" if current_track else "No starting track yet — set one with /listen.\n")
        + "Type 'quit' to exit.\n",
        Ansi.YELLOW,
    ))

    try:
        while True:
            user_text = colored_input("You: ", Ansi.LIGHT_GREEN).strip()
            if user_text.lower() in ("quit", "exit"):
                break
            if not user_text:
                continue

            _compact_writer_history_in_place(writer_history)
            
            
            clear_rem = _command_remainder(user_text, CLEAR_FLAG)
            if clear_rem is not None:
                # Wipe conversation context + token counters only.
                # Track analysis caches stay so /listen on a known path stays fast.
                # Active persona is preserved.
                writer_history.clear()
                writer_history.append({
                    "role": "system",
                    "content": build_writer_system_prompt(active_persona_text),
                })
                track_evidence_message.clear()
                last_writer_message = None
                SESSION_TOKEN_USAGE["prompt"] = 0
                SESSION_TOKEN_USAGE["completion"] = 0
                SESSION_TOKEN_USAGE["total"] = 0
                SESSION_TOKEN_USAGE["last_prompt"] = 0
                SESSION_TOKEN_USAGE["last_completion"] = 0
                SESSION_TOKEN_USAGE["last_ctx"] = 0
                print(
                    "  Chat context cleared and token counters reset to zero. "
                    "Cached track analyses are still in memory "
                    f"({len(comprehensive_analyses)} track(s)). "
                    f"Persona: {active_persona_label}.\n"
                )
                continue

            persona_rem = _command_remainder(user_text, PERSONA_FLAG)
            if persona_rem is not None:
                rem = (persona_rem or "").strip()
                if not rem:
                    preview = active_persona_text.strip().replace("\n", " ")
                    if len(preview) > 160:
                        preview = preview[:157] + "..."
                    print(
                        f"  Current persona: {active_persona_label}\n"
                        f"  Preview: {preview}\n"
                        f"  Usage:\n"
                        f"    {PERSONA_FLAG} <description>   set a custom voice/taste\n"
                        f"    {PERSONA_FLAG} reset           restore default music buddy\n"
                        f"    {PERSONA_FLAG}                 show current persona\n"
                    )
                    continue

                rem_lower = rem.lower()
                if rem_lower in ("reset", "default", "clear"):
                    active_persona_text = DEFAULT_PERSONA_PROMPT
                    active_persona_label = "default"
                    label_msg = "default music-obsessed friend"
                else:
                    # Strip a leading "Roleplay as" / "You are" so expansion stays clean.
                    cleaned = rem
                    for prefix in (
                        "roleplay as ",
                        "role-play as ",
                        "role play as ",
                        "act as ",
                        "be ",
                        "you are ",
                    ):
                        if cleaned.lower().startswith(prefix):
                            cleaned = cleaned[len(prefix):].strip()
                            break

                    # Short labels → full brief with authentic-taste instructions.
                    # Longer free-form text is used as written, with a taste reminder appended.
                    if len(cleaned) < 100 and "\n" not in cleaned:
                        active_persona_text = expand_persona_label(cleaned)
                        active_persona_label = cleaned[:60]
                    else:
                        active_persona_text = (
                            cleaned
                            + "\n\nMUSICAL TASTE: React as this persona truly would. "
                            "Do not automatically like every track. Use known or plausible "
                            "taste for real people/characters when available. Description of "
                            "a song is not the same as endorsement.\n"
                            "DEPTH: Stay conversational. Do not default to analytical "
                            "breakdowns (tempo, key, stems, production essays) unless this "
                            "persona would naturally talk that way or the user asks for it."
                        )
                        active_persona_label = cleaned.strip().split("\n", 1)[0][:60] or "custom"
                    label_msg = active_persona_label

                # Replace system message in place so the new voice applies immediately.
                new_system = build_writer_system_prompt(active_persona_text)
                if writer_history and writer_history[0].get("role") == "system":
                    writer_history[0] = {"role": "system", "content": new_system}
                else:
                    writer_history.insert(0, {"role": "system", "content": new_system})

                # Soft marker in history so the model notices the switch mid-thread.
                writer_history.append({
                    "role": "user",
                    "content": (
                        f"[Persona switched to: {label_msg}. Reply fully in character from now on. "
                        "Match their speech AND their musical taste — do not fake enthusiasm for "
                        "tracks they would not care about. Keep it human and conversational: "
                        "gut reactions over analysis. Do not default to tempo/key/production "
                        "breakdowns unless this persona would naturally geek out that way or "
                        "the user asks. You may name a song accurately without pretending to "
                        "love it. Music evidence rules are unchanged.]"
                    ),
                })
                writer_history.append({
                    "role": "assistant",
                    "content": (
                        f"Understood — staying in character as {label_msg}, "
                        "including taste and how much I care about the technical stuff."
                    ),
                })
                print(f"  Persona set to: {label_msg}\n")
                continue

            batch_rem = _command_remainder(user_text, BATCH_FLAG)
            if batch_rem is not None:
                folder = batch_rem.strip().strip('"').strip("'")
                if not folder:
                    print(
                        f"  Usage: {BATCH_FLAG} /path/to/folder\n"
                        f"  Scans all audio files in that folder (non-recursive), "
                        f"saves JSON+cover into {SAVE_DIR}/, does not load them into chat.\n"
                    )
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
                continue

            debug_rem = _command_remainder(user_text, DEBUG_FLAG)
            if debug_rem is not None:
                    if last_writer_message is None:
                        print("  No writer message yet.\n")
                    else:
                        print(json.dumps(_redact_message_for_debug(last_writer_message), indent=2, default=str))
                        print(f"  cover observations present for current track: {bool(track_cover_observations.get(current_track))}")
                        print(f"  singer identity resolution present for current track: {bool(track_singer_identity.get(current_track))}\n")
                    continue
                

            save_rem = _command_remainder(user_text, SAVE_FLAG)
            if save_rem is not None:
                filename = save_rem.strip('"').strip("'")
                if last_scanned_track is None or last_scanned_track not in comprehensive_analyses:
                    print("  No scanned track to save yet — use /listen first.\n")
                    continue

                # Collect prepared cover art so it can be written into the save.
                save_cover_bytes, save_cover_mime = None, "image/jpeg"
                b64 = track_cover_b64.get(last_scanned_track)
                if b64 and b64 != NO_COVER_SENTINEL:
                    try:
                        save_cover_bytes = base64.b64decode(b64)
                        save_cover_mime = _guess_image_mime(save_cover_bytes)
                    except Exception:
                        save_cover_bytes = None
                if (save_cover_bytes is None and not last_scanned_track.startswith(("http://", "https://"))
                        and os.path.exists(last_scanned_track)):
                    try:
                        cb, cm = extract_cover_art_only(last_scanned_track)
                        prepared = prepare_cover_image_for_ollama(cb, cm) if cb else None
                        if prepared:
                            save_cover_bytes, save_cover_mime = prepared, _guess_image_mime(prepared)
                    except Exception:
                        pass

                try:
                    path, cover_path = save_song_data(
                        filename,
                        last_scanned_track,
                        comprehensive_analyses[last_scanned_track],
                        track_corrections.get(last_scanned_track, {}),
                        track_metadata.get(last_scanned_track, {}),
                        track_cover_observations.get(last_scanned_track),
                        track_singer_identity.get(last_scanned_track),
                        save_cover_bytes,
                        save_cover_mime,
                    )
                except Exception as e:
                    print(f"  Save failed: {e}\n")
                    continue

                extra = f" + cover art → {cover_path}" if cover_path else ""
                print(f"  (saved technical details for {track_label(last_scanned_track)} to {path}{extra})\n")
                continue

            load_rem = _command_remainder(user_text, LOAD_FLAG)
            if load_rem is not None:
                # Support: /load name.json   OR   /load 'name.json' what about the chorus?
                # Optional prompt after the filename is sent to the writer like /listen.
                load_rest = load_rem.strip()
                load_prompt = ""
                filename = load_rest
                try:
                    load_tokens = shlex.split(load_rest, posix=True)
                except ValueError:
                    load_tokens = load_rest.split()
                if load_tokens:
                    filename = load_tokens[0].strip('"').strip("'")
                    if len(load_tokens) > 1:
                        load_prompt = " ".join(load_tokens[1:]).strip()
                else:
                    filename = load_rest.strip('"').strip("'")
                try:
                    data = load_song_data(filename)
                except Exception as e:
                    print(f"  Load failed: {e}\n")
                    continue

                if not data:
                    print(f"  Saved song not found in {SAVE_DIR}/ for '{filename}'.\n")
                    continue

                key = data.get("track_path") or os.path.splitext(os.path.basename(data.get("label", "saved-song")))[0]
                comprehensive_analyses[key] = data["analysis"]
                track_corrections.setdefault(key, {}).update(data.get("corrections", {}) or {})

                saved_metadata = data.get("metadata") or {}
                track_cover_b64.pop(key, None)
                # Prefer the cover written into the save (works even if the original file moved).
                saved_cover_name = str(data.get("cover_art") or "").strip()
                if saved_cover_name and os.path.exists(os.path.join(SAVE_DIR, saved_cover_name)):
                    try:
                        with open(os.path.join(SAVE_DIR, saved_cover_name), "rb") as f:
                            track_cover_b64[key] = base64.b64encode(f.read()).decode("utf-8")
                    except Exception as e:
                        print(f"  (could not read saved cover art: {e})")
                

                if (
                    ENABLE_FILE_METADATA
                    and not key.startswith(("http://", "https://"))
                    and os.path.exists(key)
                ):
                    # If the saved JSON has no metadata, try to recover it from the original file.
                    if not any(
                        str(saved_metadata.get(k) or "").strip()
                        for k in ("title", "artist", "album", "year", "lyrics")
                    ):
                        try:
                            meta, _, _ = extract_audio_metadata(key)
                            saved_metadata = meta
                        except Exception as e:
                            print(f"  (metadata recovery skipped on load: {e})")

                    # Recover cover art if the original file is still available.
                    if (SEND_COVER_ART_TO_OLLAMA or ENABLE_COVER_ART_DESCRIPTION) and track_cover_b64.get(key) is None:
                        try:
                            cover_bytes, cover_mime = extract_cover_art_only(key)
                            prepared = (
                                prepare_cover_image_for_ollama(cover_bytes, cover_mime)
                                if cover_bytes else None
                            )

                            if prepared:
                                track_cover_b64[key] = base64.b64encode(prepared).decode("utf-8")
                            else:
                                track_cover_b64[key] = NO_COVER_SENTINEL

                        except Exception as e:
                            print(f"  (cover art skipped on load: {e})")

                track_metadata[key] = saved_metadata

                saved_obs = data.get("cover_observations") or {}
                saved_identity = str(data.get("singer_identity") or "").strip()

                if saved_obs:
                    track_cover_observations[key] = saved_obs
                else:
                    cover_b64_for_desc = track_cover_b64.get(key)
                    if (
                        ENABLE_COVER_ART_DESCRIPTION
                        and cover_b64_for_desc
                        and cover_b64_for_desc != NO_COVER_SENTINEL
                    ):
                        obs = describe_cover_art(cover_b64_for_desc)
                        track_cover_observations[key] = obs if obs else {}
                    else:
                        track_cover_observations.pop(key, None)

                if saved_identity:
                    track_singer_identity[key] = saved_identity
                elif ENABLE_SINGER_IDENTITY_RESOLUTION and (track_metadata.get(key) or track_cover_observations.get(key)):
                    identity_text = resolve_singer_identity(
                        track_metadata.get(key, {}),
                        _short_vocal_audit(data["analysis"]),
                        track_cover_observations.get(key),
                        track_corrections.get(key, {}),
                    )
                    if identity_text:
                        track_singer_identity[key] = identity_text.strip()
                    else:
                        track_singer_identity.pop(key, None)
                else:
                    track_singer_identity.pop(key, None)

                current_track = key
                last_scanned_track = key
                label = track_label(key)

                loaded_metadata_block = (
                    _format_metadata_block(track_metadata.get(key, {}))
                    if ENABLE_FILE_METADATA else ""
                )

                loaded_cover_images = []
                cover_b64 = track_cover_b64.get(key)
                if SEND_COVER_ART_TO_OLLAMA and cover_b64 and cover_b64 != NO_COVER_SENTINEL:
                    loaded_cover_images.append(cover_b64)

                loaded_obs_block = (
                    _format_cover_observation_block(track_cover_observations.get(key))
                    if track_cover_observations else ""
                )

                loaded_identity_block = (
                    f"\n\nSINGER IDENTITY RESOLUTION (combined audio + metadata + cover art; use for who-is-singing questions):\n{track_singer_identity[key]}"
                    if track_singer_identity.get(key) else ""
                )

                extra_loaded_context = ""
                if "COVER ART OBSERVATIONS" not in data["analysis"]:
                    extra_loaded_context += loaded_obs_block
                if "SINGER IDENTITY RESOLUTION" not in data["analysis"]:
                    extra_loaded_context += loaded_identity_block

                has_cover_context = (
                    bool(loaded_cover_images)
                    or bool(loaded_obs_block)
                    or ("COVER ART OBSERVATIONS" in data["analysis"])
                )
                loaded_cover_note = COVER_ART_CONTEXT_NOTE if has_cover_context else ""

                loaded_message = {
                    "role": "user",
                    "content": (
                        f"[Loaded saved track '{label}']\n"
                        "(background technical details restored from a previous scan; use them for discussion):\n"
                        f"{data['analysis']}\n"
                        f"{loaded_metadata_block}{extra_loaded_context}{loaded_cover_note}\n"
                        "This track is now available in the session."
                    ),
                }

                if loaded_cover_images:
                    loaded_message["images"] = loaded_cover_images[:MAX_COVER_IMAGES_PER_REQUEST]

                writer_history.append(loaded_message)
                # Register this message with the evidence-dedup tracker (see the /listen
                # branch below) so a later `/listen <question>` about this same track
                # knows full evidence is already in history and doesn't resend it.
                track_evidence_message[key] = loaded_message
                writer_history.append({
                    "role": "assistant",
                    "content": "Got it — I've loaded the saved analysis for that track and can discuss it from here on.",
                })

                print(f"  (loaded saved song '{label}' into this session)\n")

                if load_prompt:
                    # Optional trailing question, same idea as /listen path + question.
                    writer_message = {
                        "role": "user",
                        "content": (
                            f"Regarding the loaded track '{label}': {load_prompt}"
                        ),
                    }
                    last_writer_message = writer_message
                    writer_history.append(writer_message)
                    status("Writing...")
                    final_reply, usage = ollama_chat(writer_history)
                    writer_history.append({"role": "assistant", "content": final_reply})
                    status_done()
                    print(_colorize(f"\nMusiclyse: {final_reply}\n", Ansi.MAGENTA))
                    _print_token_usage(usage)
                continue

            correct_rem = _command_remainder(user_text, CORRECT_FLAG)
            if correct_rem is not None:
                if current_track is None:
                    print("  No current track to correct — use /listen with a path first.\n")
                    continue

                remainder = correct_rem
                if "=" not in remainder:
                    print(f"  Usage: {CORRECT_FLAG} field=value[, field=value...]  e.g. '{CORRECT_FLAG} year=1966' or '{CORRECT_FLAG} singer=child_gender_uncertain'\n")
                    continue

                corrections = track_corrections.setdefault(current_track, {})
                applied = []
                for part in remainder.split(","):
                    field, sep, value = part.strip().partition("=")
                    field, value = field.strip(), value.strip()
                    if sep and field and value:
                        if field.lower() in VOCAL_CORRECTION_FIELDS:
                            normalized_value = _normalize_vocal_tag(value)
                            if normalized_value in VOCAL_LEAD_TAGS or normalized_value in VOCAL_LEAD_ALIASES:
                                value = VOCAL_LEAD_ALIASES.get(normalized_value, normalized_value)
                        corrections[field] = value
                        applied.append(f"{field} = {value}")

                if not applied:
                    print("  Couldn't parse that — use field=value format.\n")
                    continue

                label = track_label(current_track)
                print(f"  (recorded correction for {label}: {', '.join(applied)})\n")

                # Tell Gemma immediately, so it updates rather than waiting for the next /listen turn
                correction_note = (
                    f"[User correction for track '{label}']: " + "; ".join(applied) + ". "
                    "This came directly from the user and is ground truth. It overrides anything "
                    "the audio-perception model said or implied about these point(s), including any "
                    "reasoning it built on top of the now-corrected fact. Treat it as fact from now on."
                )
                writer_history.append({"role": "user", "content": correction_note})
                writer_history.append(
                    {"role": "assistant", "content": f"Got it — noted: {', '.join(applied)}. I'll treat that as confirmed going forward."}
                )
                continue  # nothing further to generate for this turn

            listen_rem = None
            force_fresh = False
            relisten_rem = _command_remainder(user_text, RELISTEN_FLAG)
            if relisten_rem is not None:
                force_fresh = True
                listen_rem = relisten_rem
            else:
                listen_rem = _command_remainder(user_text, LISTEN_FLAG)

            if listen_rem is not None:
                remainder = listen_rem
                cleaned_question, audio_ref = extract_audio_reference(remainder)

                if audio_ref:
                    current_track = audio_ref
                    print(f"  (switching current track to: {track_label(current_track)})")

                if current_track is None:
                    print(
                        f"  No track set yet — include a path or URL the first time, e.g.\n"
                        f"  '{LISTEN_FLAG} /path/to/song.mp3 what genre is this?'\n"
                    )
                    continue

                extra_image_refs = []
                if cleaned_question:
                    cleaned_question, extra_image_refs = extract_image_references(cleaned_question)
                    if extra_image_refs and not OLLAMA_SUPPORTS_IMAGES:
                        print("  (Ollama model is marked as text-only; ignoring explicit image references)")
                        extra_image_refs = []
                    elif extra_image_refs:
                        print(f"  (found {len(extra_image_refs)} explicit image reference(s) in /listen question)")

                question = cleaned_question or "Give me an overview of this track."
                # If the user forced a fresh listen, also retry cover-art extraction in case it failed before.
                if force_fresh:
                    track_cover_b64.pop(current_track, None)
                # Extract file metadata/cover art from the ORIGINAL file,
                # before any WAV conversion.
                if (
                    ENABLE_FILE_METADATA
                    and current_track
                    and not current_track.startswith(("http://", "https://"))
                    and os.path.exists(current_track)
                ):
                    need_meta = current_track not in track_metadata
                    need_cover = (SEND_COVER_ART_TO_OLLAMA or ENABLE_COVER_ART_DESCRIPTION) and (
                        current_track not in track_cover_b64
                        or track_cover_b64.get(current_track) is None
                    )

                    if need_meta or need_cover:
                        status("Reading file metadata/tags before WAV conversion...")

                        meta = {}
                        cover_bytes = None
                        cover_mime = "image/jpeg"

                        try:
                            if need_meta:
                                meta, cover_bytes, cover_mime = extract_audio_metadata(current_track)
                            else:
                                cover_bytes, cover_mime = extract_cover_art_only(current_track)
                        except Exception as e:
                            print(f"  (metadata/cover extraction skipped: {e})")

                        if need_meta:
                            existing_meta = track_metadata.setdefault(current_track, {})
                            for k, v in meta.items():
                                if v and not existing_meta.get(k):
                                    existing_meta[k] = v

                        if need_cover:
                            try:
                                prepared = (
                                    prepare_cover_image_for_ollama(cover_bytes, cover_mime)
                                    if cover_bytes else None
                                )

                                if prepared:
                                    track_cover_b64[current_track] = base64.b64encode(prepared).decode("utf-8")
                                    print(f"  (embedded cover art found for {track_label(current_track)})")
                                else:
                                    # Cache a permanent "no cover" only when no raw embedded art was found.
                                    # If raw art exists but preparation failed, leave it unset so a later /listen can retry.
                                    if cover_bytes is None:
                                        track_cover_b64[current_track] = NO_COVER_SENTINEL
                                    else:
                                        track_cover_b64.pop(current_track, None)

                                    print(
                                        f"  (no usable embedded cover art found for {track_label(current_track)}; "
                                        f"raw_bytes={len(cover_bytes) if cover_bytes else 0}, mime={cover_mime})"
                                    )

                            except Exception as e:
                                # Leave it unset/None so a later /listen can retry if this was transient.
                                print(f"  (cover art skipped: {e})")

                    # Cover-art description for identity/context (must run on success, not only on failure).
                    if SEND_COVER_ART_TO_OLLAMA or ENABLE_COVER_ART_DESCRIPTION:
                        cover_b64 = track_cover_b64.get(current_track)
                        if (
                            cover_b64
                            and cover_b64 != NO_COVER_SENTINEL
                            and (force_fresh or current_track not in track_cover_observations)
                        ):
                            status("Describing embedded cover art for writer context...")
                            obs = describe_cover_art(cover_b64)
                            if obs:
                                track_cover_observations[current_track] = obs
                            else:
                                track_cover_observations.pop(current_track, None)
                        elif not cover_b64 or cover_b64 == NO_COVER_SENTINEL:
                            track_cover_observations.pop(current_track, None)

                need_analysis = force_fresh or current_track not in comprehensive_analyses
                if need_analysis:
                    # A fresh (or forced) analysis invalidates any evidence block already
                    # sitting in writer_history for this track, so the new evidence must
                    # be sent again in full on this turn.
                    track_evidence_message.pop(current_track, None)
                    if not current_track.startswith(("http://", "https://")) and not os.path.exists(current_track):
                        print(f"  File not found for fresh analysis: {current_track}\n")
                        continue

                    # Resolve to a playable path (convert non-wav/flac local files; URLs pass through)
                    if current_track.startswith(("http://", "https://")):
                        resolved_path = current_track
                    else:
                        ext = os.path.splitext(current_track)[1].lower()
                        if ext in (".wav", ".flac"):
                            resolved_path = current_track
                        elif current_track in audio_temp_files:
                            resolved_path = audio_temp_files[current_track]
                        else:
                            status(f"Converting {ext} to WAV for compatibility...")
                            resolved_path = convert_to_wav(current_track)
                            audio_temp_files[current_track] = resolved_path

                    # About to start a batch of Music Flamingo passes (era, full analysis,
                    # vocal, confirmation, lyrics). Free Gemma from Ollama's memory first —
                    # it isn't needed again until we're back to writing the final answer —
                    # then lazily load Music Flamingo (a no-op if it's already loaded from
                    # a call earlier in this same batch).
                    ollama_unload_model()
                    mf_model, mf_processor, mf_device = get_music_flamingo()

                    # ERA is run FIRST, in its own isolated conversation with no prior generated
                    # text (mood/vibe language especially) sitting in context to bias the judgment
                    # toward "sounds like modern indie."
                    status("Listening — estimating era...")
                    era_conversation = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": ERA_ANALYSIS_PROMPT},
                                {"type": "audio", "path": resolved_path},
                            ],
                        }
                    ]
                    era_result = mf_generate(mf_model, mf_processor, era_conversation, max_new_tokens=1024)

                    status("Listening — running full analysis (this can take a while)...")
                    main_prompt = MF_FULL_ANALYSIS_PROMPT + (MF_DEEP_MODE_ADDENDUM if DEEP_MODE else "")
                    main_max_tokens = 3072 if DEEP_MODE else 2048
                    mf_conversation = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": main_prompt},
                                {"type": "audio", "path": resolved_path},
                            ],
                        }
                    ]
                    first_pass = mf_generate(mf_model, mf_processor, mf_conversation, max_new_tokens=main_max_tokens)

                    objective_report = ""
                    vocal_objective_report = ""
                    essentia_report = ""
                    if not current_track.startswith(("http://", "https://")):
                        dsp_path = None
                        if current_track in dsp_temp_files:
                            dsp_path = dsp_temp_files[current_track]
                        else:
                            status("Preparing higher-rate WAV for signal processing...")
                            try:
                                dsp_path = convert_to_wav(current_track, sample_rate=22050)
                                dsp_temp_files[current_track] = dsp_path
                            except Exception:
                                ext = os.path.splitext(current_track)[1].lower()
                                if ext in (".wav", ".flac"):
                                    dsp_path = current_track
                                else:
                                    dsp_path = resolved_path

                        if dsp_path is not None:
                            if ENABLE_OBJECTIVE_AUDIO_REPORT:
                                status("Measuring beat/timbre with signal processing...")
                                objective_report = build_objective_audio_report(dsp_path)
                            if ENABLE_VOCAL_OBJECTIVE_REPORT:
                                status("Measuring vocal pitch/formant proxies...")
                                vocal_objective_report = build_vocal_objective_report(dsp_path)
                            if ENABLE_ESSENTIA_REPORT and ESSENTIA_AVAILABLE:
                                status("Measuring tempo/key/spectral features with Essentia...")
                                essentia_report = build_essentia_report(dsp_path)

                    vocal_result = ""
                    if ENABLE_VOCAL_PASS:
                        status("Listening — estimating singer profile...")
                        vocal_prompt_text = VOCAL_ANALYSIS_PROMPT
                        if vocal_objective_report:
                            vocal_prompt_text += (
                                "\n\nObjective vocal measurements (evidence only, not proof):\n"
                                f"{vocal_objective_report}"
                            )

                        vocal_conversation = [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": vocal_prompt_text},
                                    {"type": "audio", "path": resolved_path},
                                ],
                            }
                        ]
                        vocal_result = mf_generate(mf_model, mf_processor, vocal_conversation, max_new_tokens=768)

                    confirmation_result = ""
                    initial_lead = ""
                    initial_backing = ""
                    confirm_lead = ""
                    confirm_confidence = ""
                    final_lead = ""

                    vocal_pitch = extract_vocal_pitch_summary(vocal_objective_report) if vocal_objective_report else {
                        "median": None, "low": None, "high": None, "note": None
                    }
                    median_f0 = vocal_pitch["median"]

                    if vocal_result:
                        initial_lead, initial_backing = parse_vocal_tags(vocal_result)

                    vocals_present_match = re.search(
                        r"VOCALS PRESENT\s*[-–—:]?\s*(yes|no|uncertain|instrumental)",
                        vocal_result or "",
                        re.IGNORECASE,
                    )
                    no_clear_vocals = bool(
                        vocals_present_match and vocals_present_match.group(1).lower() in ("no", "instrumental")
                    )

                    if ENABLE_VOCAL_CONFIRMATION_PASS and vocal_result and not no_clear_vocals:
                        should_confirm = False

                        # Always confirm when the initial result is young/uncertain.
                        if initial_lead in UNCERTAIN_YOUNG_CATEGORIES:
                            should_confirm = True

                        # Confirm female-leaning results more aggressively, especially when pitch is high
                        # or objective f0 is unavailable. This targets the common miscategorisation case.
                        elif initial_lead in FEMALE_LEAD_CATEGORIES and (
                            (median_f0 is not None and median_f0 >= VOCAL_CONFIRMATION_F0_THRESHOLD)
                            or (median_f0 is None and VOCAL_CONFIRMATION_WITHOUT_F0)
                        ):
                            should_confirm = True

                        if should_confirm:
                            status("Listening — confirming lead voice category...")
                            confirmation_conversation = [
                                {
                                    "role": "user",
                                    "content": [
                                        {"type": "text", "text": VOCAL_CONFIRMATION_PROMPT},
                                        {"type": "audio", "path": resolved_path},
                                    ],
                                }
                            ]
                            confirmation_result = mf_generate(
                                mf_model, mf_processor, confirmation_conversation, max_new_tokens=256
                            )

                        if confirmation_result:
                            confirm_lead, confirm_confidence = parse_vocal_confirmation(confirmation_result)

                    if vocal_result:
                        final_lead = choose_final_vocal_lead(initial_lead, confirm_lead, confirm_confidence)

                    if FAST_MODE:
                        revised = first_pass
                    else:
                        status("Double-checking its own analysis for overconfident claims...")
                        self_check_text = SELF_CHECK_PROMPT
                        if vocal_result or confirmation_result:
                            self_check_text += "\n\nVocal profile evidence:\n"
                            if vocal_result:
                                self_check_text += f"{vocal_result}\n"
                            if confirmation_result:
                                self_check_text += f"{confirmation_result}\n"
                            self_check_text += (
                                "Use this ONLY to correct lead/backing voice age/gender category claims. "
                                "Do NOT use it to change GENRE, KEY, CHORD PROGRESSION, SONG STRUCTURE, ERA, or non-vocal instrumentation. "
                                "If the lead is identified as young_male, child_male_likely, child_gender_uncertain, or child_gender_uncertain, do not leave a confident female-lead claim in place."
                            )

                        if final_lead in UNCERTAIN_YOUNG_CATEGORIES:
                            self_check_text += (
                                " The lead vocal is gender-uncertain/young-uncertain according to the isolated vocal decision logic. "
                                "Remove any confident boy/girl/woman/girl/man claim from the analysis."
                            )

                        objective_crosscheck_parts = []
                        if essentia_report:
                            objective_crosscheck_parts.append(essentia_report)
                        if objective_report:
                            objective_crosscheck_parts.append(objective_report)

                        if objective_crosscheck_parts:
                            self_check_text += (
                                "\n\nIndependent signal-processing cross-checks "
                                "(use only for tempo/beat, key/key strength when explicitly reported, timbre, element activity, and dynamic range; "
                                "do NOT use them to change GENRE or vocal identity):\n"
                                + "\n\n".join(objective_crosscheck_parts)
                            )

                        mf_conversation.append(
                            {"role": "assistant", "content": [{"type": "text", "text": first_pass}]}
                        )
                        mf_conversation.append(
                            {"role": "user", "content": [{"type": "text", "text": self_check_text}]}
                        )
                        revised = mf_generate(mf_model, mf_processor, mf_conversation, max_new_tokens=main_max_tokens)

                    # Dedicated lyrics pass in an ISOLATED conversation (like ERA).
                    # Skip when authoritative file-tag lyrics are already present — the MF
                    # transcription is often noisy, duplicates tags, and is a major token cost.
                    tag_lyrics = ""
                    if ENABLE_FILE_METADATA:
                        tag_lyrics = str(
                            (track_metadata.get(current_track) or {}).get("lyrics") or ""
                        ).strip()
                    skip_mf_lyrics = (
                        SKIP_MF_LYRICS_WHEN_TAGS_PRESENT
                        and len(tag_lyrics) >= METADATA_LYRICS_MIN_CHARS_TO_SKIP_MF
                    )
                    if skip_mf_lyrics:
                        status("Skipping MF lyrics pass (file-tag lyrics already present)...")
                        revised += (
                            "\n\nFULL LYRICS TRANSCRIPTION: skipped — file-tag lyrics are present "
                            "and treated as authoritative. Use TRACK METADATA lyrics for quotes."
                        )
                    else:
                        status("Transcribing full lyrics in a separate dedicated pass...")
                        lyrics_conversation = [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": LYRICS_TRANSCRIPTION_PROMPT},
                                    {"type": "audio", "path": resolved_path},
                                ],
                            }
                        ]
                        full_lyrics = mf_generate(
                            mf_model, mf_processor, lyrics_conversation,
                            max_new_tokens=1536,
                            repetition_penalty=1.45,
                            no_repeat_ngram_size=12,
                        )
                        full_lyrics = _sanitize_lyrics_transcription(full_lyrics)
                        if full_lyrics and full_lyrics.strip():
                            revised += (
                                f"\n\nFULL LYRICS TRANSCRIPTION (dedicated pass):\n{full_lyrics}"
                            )

                    # This was the last Music Flamingo pass in this batch (stem/MIDI below
                    # uses Demucs/Omnizart, not Music Flamingo; singer-identity resolution and
                    # the final written answer use Gemma via Ollama). Free its memory now
                    # rather than holding it resident for the rest of the turn.
                    unload_music_flamingo()
                    mf_model = mf_processor = None

                    revised += f"\n\n11. ERA / RELEASE PERIOD (isolated dedicated pass):\n{era_result}"
                    if vocal_result:
                        revised += f"\n\nVOCAL / SINGER PROFILE (isolated dedicated pass):\n{vocal_result}"
                        if confirmation_result:
                            revised += f"\n\nVOCAL CONFIRMATION PASS:\n{confirmation_result}"

                        f0_text = f"{median_f0} Hz" if median_f0 is not None else "unavailable"
                        revised += (
                            "\n\nVOCAL DECISION AUDIT (audio-only evidence for vocal age/gender):\n"
                            f"- Initial isolated lead profile: {initial_lead or 'unparsed'}\n"
                            f"- Confirmation pass: {confirm_lead or 'not run/unparsed'} (confidence={confirm_confidence or 'n/a'})\n"
                            f"- Objective median f0: {f0_text}\n"
                            f"- FINAL LEAD PROFILE: {final_lead or 'unknown'}\n"
                            f"- BACKING PROFILES: {initial_backing or 'uncertain'}\n"
                            "This is audio-only evidence. If a SINGER IDENTITY RESOLUTION block appears later in this analysis, use that for user-facing singer-identity claims; otherwise use FINAL LEAD PROFILE. Do not override a well-supported combined judgment with pitch impressions alone."
                        )

                    # Always expose vocal pitch information to the writer, independent of gender/age category.
                    if ENABLE_VOCAL_OBJECTIVE_REPORT and not current_track.startswith(("http://", "https://")):
                        pitch_lines = ["VOCAL PITCH REPORT (independent of lead age/gender category):"]
                        if median_f0 is None:
                            pitch_lines.append("- No reliable voiced pitch detected in the scanned portion.")
                        else:
                            pitch_lines.append(f"- objective voiced pitch median: {median_f0} Hz")
                            if vocal_pitch.get("note"):
                                pitch_lines.append(f"- approximate median note: {vocal_pitch['note']}")
                            if vocal_pitch.get("low") is not None and vocal_pitch.get("high") is not None:
                                pitch_lines.append(
                                    f"- 5-95 percentile range: {vocal_pitch['low']}-{vocal_pitch['high']} Hz"
                                )
                        pitch_lines.append(
                            "- Prefer median + 5–95 percentile range for the main vocal range. "
                            "Do not treat absolute extremes from stem MIDI (often harmonics/octave errors) "
                            "as the sung range. Do not say the specific pitch range is missing when this block exists."
                        )
                        revised += "\n\n" + "\n".join(pitch_lines)
                        
                    # --- Reconcile BPM from MF, Essentia, and objective detector ---
                    mf_bpm_val = extract_bpm_from_text(first_pass)
                    revised_bpm_val = extract_bpm_from_text(revised)
                    if revised_bpm_val is not None:
                        mf_bpm_val = revised_bpm_val
                    essentia_bpm_val = extract_essentia_bpm(essentia_report) if essentia_report else None
                    objective_bpm_val = extract_objective_bpm(objective_report) if objective_report else None

                    final_bpm, bpm_note = reconcile_bpm(mf_bpm_val, essentia_bpm_val, objective_bpm_val)

                    if final_bpm:
                        revised += (
                            f"\n\nRECOMMENDED TEMPO FOR DISCUSSION: {final_bpm} BPM. "
                            f"Reasoning: {bpm_note}. "
                            "This is the primary tempo to report to the user. "
                            "State it as a concrete figure (e.g. 'about 158 BPM'). "
                            "Do not expand it into a range unless this block itself marks the value as uncertain."
                        )
                    # -----------------------------------------------
                    

                    # Independent cross-checks via signal processing (local files only) — these
                    # come from actual DSP analysis of the waveform, not the model's own perception
                    measurement_parts = []
                    if objective_report:
                        measurement_parts.append(objective_report)
                    if vocal_objective_report:
                        measurement_parts.append(vocal_objective_report)
                    if essentia_report:
                        measurement_parts.append(essentia_report)
                    if ENABLE_ESSENTIA_REPORT:
                        if not ESSENTIA_AVAILABLE:
                            measurement_parts.append("ESSENTIA REPORT unavailable (essentia could not be imported).")
                        elif not essentia_report:
                            measurement_parts.append("ESSENTIA REPORT unavailable (imported, but no measurements were produced for this file).")

                    if measurement_parts:
                        revised += (
                            "\n\n[Independent signal-processing report:\n"
                            + "\n\n".join(measurement_parts)
                            + "\nThis is computed directly from the audio (librosa/Essentia), not from the model above — use it only for tempo/beat, key/key strength when explicitly reported, timbre, low/mid/high element activity, dynamic range, and vocal pitch/formant proxies. Do NOT use it to infer or revise GENRE."
                        )

                    # Optional Demucs 6s + Omnizart stem MIDI report.
                    # Essentia and Music Flamingo above still used only the original track.
                    stem_midi_report = ""
                    if ENABLE_STEM_MIDI and not current_track.startswith(("http://", "https://")):
                        try:
                            _get_omnizart()  # fail fast before slow Demucs run

                            if current_track in stem_temp_files:
                                stem_wav = stem_temp_files[current_track]
                            else:
                                status("Preparing stereo WAV for Demucs/MIDI...")
                                stem_wav = convert_to_wav_for_stems(
                                    current_track,
                                    sample_rate=44100,
                                    channels=2,
                                    max_seconds=STEM_MIDI_MAX_SECONDS,
                                )
                                stem_temp_files[current_track] = stem_wav

                            status("Running Demucs 6s stem separation (this can be slow)...")
                            out_dir = tempfile.mkdtemp(prefix="demucs_")
                            demucs_out_dirs.append(out_dir)
                            stems = run_demucs_stems(stem_wav, out_dir)

                            if not stems:
                                stem_midi_report = "STEM MIDI REPORT unavailable: Demucs did not produce expected stems."
                            else:
                                status("Running Omnizart on each separated stem...")
                                stem_midi_report = build_omnizart_summaries(stems)
                                _release_omnizart_memory()

                        except Exception as e:
                            print(f"  (stem MIDI skipped/unavailable: {e})")
                            stem_midi_report = f"STEM MIDI REPORT unavailable: {e}"

                    # Always append the stem MIDI report to revised, even if unavailable,
                    # so Gemma knows the status and it's logged in the save file.
                    stem_midi_report = (
                        stem_midi_report
                        or "STEM MIDI REPORT not run (disabled or non-local track)."
                    )

                    track_stem_midi_report[current_track] = stem_midi_report

                    # Include the report in the saved/comprehensive analysis once.
                    revised += "\n\n" + stem_midi_report
                    
                    # Move singer identity resolution OUTSIDE the MIDI check, 
                    # because it should run regardless of whether MIDI succeeded.
                    singer_identity_text = ""
                    if ENABLE_SINGER_IDENTITY_RESOLUTION and (
                        track_metadata.get(current_track) or track_cover_observations.get(current_track)
                    ):
                        f0_text_res = f"{median_f0} Hz" if median_f0 is not None else "unavailable"
                        vocal_audit_for_resolution = (
                            f"Initial isolated lead profile: {initial_lead or 'unparsed'}\n"
                            f"Confirmation pass: {confirm_lead or 'not run/unparsed'} (confidence={confirm_confidence or 'n/a'})\n"
                            f"FINAL LEAD PROFILE: {final_lead or 'unknown'}\n"
                            f"BACKING PROFILES: {initial_backing or 'uncertain'}\n"
                            f"objective median f0: {f0_text_res}"
                        )

                        if no_clear_vocals:
                            vocal_audit_for_resolution = "No clear vocals detected.\n" + vocal_audit_for_resolution

                        status("Resolving singer identity from audio + metadata + cover art...")
                        singer_identity_text = resolve_singer_identity(
                            track_metadata.get(current_track, {}),
                            vocal_audit_for_resolution,
                            track_cover_observations.get(current_track),
                            track_corrections.get(current_track, {}),
                        )

                        if singer_identity_text:
                            track_singer_identity[current_track] = singer_identity_text.strip()
                        else:
                            track_singer_identity.pop(current_track, None)

                        cover_obs_block = (
                            _format_cover_observation_block(track_cover_observations.get(current_track))
                            if track_cover_observations else ""
                        )
                        if cover_obs_block and "COVER ART OBSERVATIONS" not in revised:
                            revised += "\n\n" + cover_obs_block

                        resolved_tag = _parse_singer_identity(singer_identity_text) if singer_identity_text else ""

                        if singer_identity_text and "SINGER IDENTITY RESOLUTION" not in revised:
                            revised += (
                                f"\n\nSINGER IDENTITY RESOLUTION (combined audio + metadata + cover art; use for who-is-singing questions):\n{singer_identity_text}"
                            )

                        priority_tag = resolved_tag if resolved_tag in VOCAL_LEAD_TAGS else final_lead
                        if vocal_result or singer_identity_text:
                            revised += build_vocal_priority_note(priority_tag, initial_backing or "uncertain")

                    revised = _collapse_runaway_chord_repetition(revised)
                    comprehensive_analyses[current_track] = revised
                    last_scanned_track = current_track

                    if SHOW_RAW_ANALYSIS:
                        print(f"\n  ── Music Flamingo's raw analysis for {track_label(current_track)} ──")
                        print(f"  {revised}")
                        print("  ──────────────────────────────────────────────────────────\n")
                else:
                    status_done("Using cached full analysis for this track")
                    last_scanned_track = current_track

                    if SHOW_RAW_ANALYSIS:
                        print(f"\n  ── Music Flamingo's raw analysis for {track_label(current_track)} ──")
                        print(f"  {comprehensive_analyses[current_track]}")
                        print("  ──────────────────────────────────────────────────────────\n")

                label = track_label(current_track)

                corrections = track_corrections.get(current_track)
                correction_block = ""
                if corrections:
                    correction_lines = "\n".join(f"- {k}: {v}" for k, v in corrections.items())
                    correction_block = (
                        f"\n\nCONFIRMED CORRECTIONS FOR THIS TRACK (from the user — ground truth, "
                        f"overrides the analysis above wherever they conflict):\n{correction_lines}"
                    )

                vocal_question = bool(
                    re.search(
                        r"\b(singer|singing|vocals?|voice|male|female|boy|girl|man|woman|gender)\b",
                        question,
                        re.IGNORECASE,
                    )
                )
                vocal_gate_note = ""
                if vocal_question:
                    vocal_gate_note = (
                        "\n\nFor this specific question about the singer/voice, start from any CONFIRMED CORRECTIONS. "
                        "If a SINGER IDENTITY RESOLUTION block is present in the track context, use it as the primary combined judgment; it already weighs audio, metadata, and cover art. "
                        "Only override it if user correction, distinct co-lead evidence, or unambiguous adult vocal evidence clearly contradicts it. "
                        "If no resolution block is present, combine the VOCAL DECISION AUDIT / FINAL LEAD PROFILE with file metadata, reliable artist knowledge, and cover-art cues. "
                        "Do not default a high-pitched young voice to female when the combined context supports a male solo artist."
                    )

                metadata_block = (
                    _format_metadata_block(track_metadata.get(current_track, {}))
                    if ENABLE_FILE_METADATA else ""
                )

                context_prior_note = ""
                if ENABLE_FILE_METADATA:
                    meta = track_metadata.get(current_track, {}) or {}
                    if any(str(meta.get(k) or "").strip() for k in ("title", "artist", "album", "year")):
                        year_bit = f", year tag {meta.get('year')}" if meta.get("year") else ""
                        year_lock = ""
                        if meta.get("year"):
                            year_lock = (
                                f" The release year from the file tags is {meta.get('year')}; "
                                "use that year when stating when the track or album was released. "
                                "Do not replace it with a different year from knowledge of other releases by the same artist."
                            )
                        context_prior_note = (
                            "\n\nIDENTITY/STYLE CONTEXT PRIOR: The file metadata identifies this recording as "
                            f"{meta.get('title') or 'unknown title'} by {meta.get('artist') or 'unknown artist'}"
                            + (f" from {meta.get('album')}" if meta.get("album") else "")
                            + year_bit
                            + ". If you have reliable general knowledge about this artist/title/album, use it as a prior for genre/subgenre, production style, overall album vibe, and lead-vocal expectations (including whether the act is known to be all-male/all-female or has a known lead vocalist). "
                            "Do not state uncertain trivia as fact. "
                            "Do not invent nationality or country of origin unless you are highly confident; prefer neutral wording if unsure."
                            + year_lock
                        )

                cover_images = []
                if SEND_COVER_ART_TO_OLLAMA and OLLAMA_SUPPORTS_IMAGES:
                    cover_b64 = track_cover_b64.get(current_track)
                    if cover_b64 and cover_b64 != NO_COVER_SENTINEL:
                        cover_images.append(cover_b64)
                        status_done("Attaching embedded cover art to writer")
                    else:
                        status_done("No usable embedded cover art found/attached")

                writer_images = cover_images[:]
                explicit_image_obs_block = ""

                if extra_image_refs:
                    obs_parts = []
                    for i, ref in enumerate(extra_image_refs):
                        if len(writer_images) >= MAX_IMAGES_PER_REQUEST:
                            break

                        try:
                            b64 = image_to_base64(ref)
                        except Exception as e:
                            print(f"  (skipping unreadable explicit image {ref}: {e})")
                            continue

                        writer_images.append(b64)

                        if ENABLE_IMAGE_OBSERVATIONS_FOR_GENERAL and i < MAX_IMAGES_TO_DESCRIBE:
                            obs = describe_cover_art(b64)
                            if obs:
                                obs_parts.append(
                                    f"EXPLICIT IMAGE {i + 1} OBSERVATIONS ({ref}):\n"
                                    + _format_cover_observation_block(obs)
                                )

                    explicit_image_obs_block = "\n\n".join(obs_parts)

                has_cover_context = (
                    bool(cover_images)
                    or bool(track_cover_observations.get(current_track))
                    or ("COVER ART OBSERVATIONS" in comprehensive_analyses[current_track])
                    or bool(explicit_image_obs_block)
                )
                cover_note = COVER_ART_CONTEXT_NOTE if has_cover_context else ""

                # The full evidence block (Music Flamingo analysis + Essentia report +
                # stem/MIDI note logs + singer identity resolution) is large — often the
                # single biggest contributor to prompt size. It only needs to go out in
                # full once per track per session — after that it's already sitting in
                # writer_history. But that's only safe to skip if the earlier message
                # holding it (a) hasn't been evicted by history compaction and (b) isn't
                # at risk of being dropped by this request's context-budget trimming
                # (which only guarantees the newest message survives). If either check
                # fails, fall back to sending it in full again — a pointer to evidence
                # the model doesn't actually have is worse than the extra tokens.
                prior_evidence_msg = track_evidence_message.get(current_track)
                evidence_still_safe = prior_evidence_msg is not None and _evidence_message_still_safe(
                    writer_history, prior_evidence_msg, OLLAMA_NUM_CTX
                )

                if evidence_still_safe:
                    evidence_section = (
                        "(Private notes for this track are already in the conversation — "
                        "reuse them; they have not changed unless a CONFIRMED CORRECTIONS "
                        "section appears below. Do not restate the notes unless the user asks.)"
                    )
                else:
                    evidence_section = (
                        "=== PRIVATE TRACK NOTES (for you only; do not recite as a report) ===\n"
                        + comprehensive_analyses[current_track]
                    )

                writer_user_msg = (
                    f"[We're listening to: {label}]\n\n"
                    f"{evidence_section}"
                    f"{correction_block}"
                    f"{metadata_block}"
                    f"{context_prior_note}"
                    f"{cover_note}"
                    f"{vocal_gate_note}\n\n"
                    f"User said: {question}\n\n"
                    "Reply as their music buddy in the conversation — answer what they asked, "
                    "not a full analytical write-up unless they asked for one."
                )

                if not OLLAMA_SUPPORTS_IMAGES:
                    writer_images = []
                else:
                    writer_images = [img for img in writer_images if _is_sendable_base64_image(img)]

                writer_message = {"role": "user", "content": writer_user_msg}
                if writer_images:
                    writer_message["images"] = writer_images[:MAX_WRITER_IMAGES_PER_TURN]

                last_writer_message = writer_message
                writer_history.append(writer_message)

                if not evidence_still_safe:
                    # This message now holds the full evidence block — remember it (by
                    # object identity) as the one to check against on future turns.
                    track_evidence_message[current_track] = writer_message

                status("Writing...")
                final_reply, _usage = ollama_chat(writer_history)
                writer_history.append({"role": "assistant", "content": final_reply})

            else:
                # General question / cross-track comparison — straight to Gemma.
                # Multiple image references are now supported in one message.
                cleaned_text, image_refs = extract_image_references(user_text)

                if image_refs and not OLLAMA_SUPPORTS_IMAGES:
                    print("  (Ollama model is marked as text-only; ignoring image references)")
                    image_refs = []

                if image_refs:
                    refs = image_refs[:max(MAX_WRITER_IMAGES_PER_TURN, MAX_IMAGES_TO_DESCRIBE)]
                    print(f"  (found {len(image_refs)} image reference(s), sending to {OLLAMA_MODEL}...)")
                    if len(image_refs) > len(refs):
                        print(f"  (limiting this request to the first {len(refs)} images)\n")

                    b64_images = []
                    obs_blocks = []

                    for i, ref in enumerate(refs):
                        try:
                            b64 = image_to_base64(ref)
                        except Exception as e:
                            print(f"  (skipping unreadable image {ref}: {e})")
                            continue

                        b64_images.append(b64)

                        if ENABLE_IMAGE_OBSERVATIONS_FOR_GENERAL and i < MAX_IMAGES_TO_DESCRIBE:
                            obs = describe_cover_art(b64)
                            if obs:
                                obs_blocks.append(
                                    f"IMAGE {i + 1} OBSERVATIONS ({ref}):\n"
                                    + _format_cover_observation_block(obs)
                                )

                    image_obs_block = "\n\n".join(obs_blocks)
                    base_content = cleaned_text or "Describe these images."
                    content = f"{base_content}\n\n{image_obs_block}" if image_obs_block else base_content

                    if len(b64_images) > MAX_WRITER_IMAGES_PER_TURN:
                        print(f"  (limiting Ollama request to first {MAX_WRITER_IMAGES_PER_TURN} images)")

                    writer_message = {
                        "role": "user",
                        "content": content,
                        "images": b64_images[:MAX_WRITER_IMAGES_PER_TURN],
                    }
                    last_writer_message = writer_message
                    writer_history.append(writer_message)

                    status("Writing...")
                    final_reply, _usage = ollama_chat(writer_history)
                    writer_history.append({"role": "assistant", "content": final_reply})

                else:
                    writer_message = {"role": "user", "content": user_text}
                    last_writer_message = writer_message
                    writer_history.append(writer_message)
                    status("Thinking...")
                    final_reply, _usage = ollama_chat(writer_history)
                    writer_history.append({"role": "assistant", "content": final_reply})

            status_done()
            print(_colorize(f"\nMusiclyse: {final_reply}\n", Ansi.MAGENTA))
            _print_token_usage(locals().get("_usage") or SESSION_TOKEN_USAGE)
    finally:
        for temp_path in audio_temp_files.values():
            if os.path.exists(temp_path):
                os.remove(temp_path)
        for temp_path in dsp_temp_files.values():
            if os.path.exists(temp_path):
                os.remove(temp_path)
        for temp_path in stem_temp_files.values():
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
        try:
            unload_music_flamingo()
        except Exception:
            pass

        _release_omnizart_memory()
        

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


if __name__ == "__main__":
    main()