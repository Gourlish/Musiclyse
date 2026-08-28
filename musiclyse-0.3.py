"""
Musiclyse — chat about (and compare) multiple songs,
using a local LLM (via Ollama) to write the final response while a multi step pipeline performs grounded audio analysis on demand.

Routing:
    "/listen <question>" — analyzes the CURRENT track.
    "/listen <path or URL> <question>" — SWITCHES the current track to the
        given file/URL, then analyzes it. Each track keeps its own Music
        Flamingo conversation history, so switching back to an earlier
        track later still has that context.
    Anything else — goes straight to the LLM alone (general questions,
        comparisons between tracks already discussed, etc.), with the full
        conversation history — including every track's analysis so far.

Examples:
    /listen /Users/me/Music/song_a.mp3 what key and tempo is this in?
    /listen what instruments do you hear?              (still song_a)
    /listen /Users/me/Music/song_b.wav describe this one
    how does song_b's tempo compare to song_a?          (straight to the LLM)

New commands:
    /save=filename.json   Save the technical details for the most recently scanned track.
    /load filename.json [question]   Load a saved track; optional question after the name.
    /batch /path/to/folder   Overnight-scan every audio file in a folder into saved-songs/
                             as "Artist Name - Song Name.json" (from tags; Unknown +
                             original filename if tags missing). Same analysis as
                             /listen; does not import into chat.
    /clear                   Wipe chat context and reset session token counters.
    /clearall                Wipe everything including cached song .json data.
    /persona <description>   Switch chat voice/taste (music evidence rules stay).
    /persona reset           Restore the default music-obsessed friend persona.

Prerequisites:
    brew install ollama ffmpeg
    # Optional but recommended for Essentia on macOS:
    #   conda create -n musicalyse python=3.10 && conda activate musicalyse && conda install -c conda-forge essentia
    ollama serve                       # or run the Ollama app
    ollama pull muse-glimmer
    pip install requests librosa mutagen

Optional stem/MIDI stack:
    pip install demucs omnizart tensorflow pretty_midi

Optional per-stem/whole-mix instrument tagging AND independent genre/mood
cross-check (same pretrained tagger powers both; silently skipped if not
installed):
    pip install panns-inference
    # First use downloads a pretrained AudioSet checkpoint (~300MB) to
    # ~/panns_data/ automatically.

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
import unicodedata
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

                         ==========  V E R S I O N   0 . 3  ==========

                         A   G O U R L I S H   V I B E   P R O J E C T
"""


def set_terminal_title(title="Musiclyse 0.3"):
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
    set_terminal_title("Musiclyse 0.3")
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
OLLAMA_NUM_CTX = 131072
# Running totals for the current process (updated after each Ollama reply).
SESSION_TOKEN_USAGE = {"prompt": 0, "completion": 0, "total": 0, "last_prompt": 0, "last_completion": 0, "last_ctx": 0}


OLLAMA_BASE_URL = OLLAMA_URL.rsplit("/api/chat", 1)[0]

# Set this to False if your Ollama model is text-only / does not support images.
# If True, the script will still retry without images if Ollama returns a 400.
OLLAMA_SUPPORTS_IMAGES = True

MAX_WRITER_IMAGES_PER_TURN = 8          # images actually sent to Ollama in one request
MAX_STORED_IMAGES_IN_HISTORY = 8        # base64 images kept in Python's writer_history
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


# --- Wikipedia background-context (local RAG) -------------------------------
# Read-only lookups against WIKI_DB_PATH, scoped strictly to non-technical
# background context for the writer prompt (see build_wiki_context_block()).
# The schema is auto-detected the first time it's needed so this works with
# whatever table/column names the database happens to use, rather than
# assuming one specific import-tool's layout.

_WIKI_DB_CONN = None
_WIKI_DB_SCHEMA = None            # dict: {"mode": "fts"|"plain", "table", "title_col", "text_col"}
_WIKI_DB_UNAVAILABLE_REASON = None

_WIKI_TITLE_COL_HINTS = ("title", "page_title", "article_title", "name")
_WIKI_TEXT_COL_HINTS = ("text", "body", "content", "extract", "summary", "article", "wikitext")


def _wiki_quote_ident(name):
    return '"' + str(name).replace('"', '""') + '"'


def _wiki_pick_col(cols, hints):
    lower = {c.lower(): c for c in cols}
    for hint in hints:
        for lc, orig in lower.items():
            if lc == hint or lc.endswith("_" + hint) or hint in lc:
                return orig
    return None


def _wiki_parse_fts_columns(create_sql):
    """Crude parse of 'CREATE VIRTUAL TABLE x USING fts5(title, body, ...)' to
    pull out the declared column names."""
    m = re.search(r"USING\s+fts\d*\s*\((.*)\)\s*;?\s*$", create_sql or "", re.IGNORECASE | re.DOTALL)
    if not m:
        return []
    cols = []
    for part in m.group(1).split(","):
        part = part.strip()
        if not part or "=" in part:  # skip options like content='other_table'
            continue
        cols.append(part.split()[0].strip('`"[]'))
    return cols


def _wiki_detect_schema(conn):
    """Inspect sqlite_master once to find an article table with a title-like
    and a body-like text column, preferring an existing FTS5 table (fast
    full-text search) over a plain table (exact/prefix match only, since a
    LIKE %term% scan over a multi-GB table with no index is too slow to run
    per track switch)."""
    cur = conn.cursor()
    cur.execute("SELECT name, sql, type FROM sqlite_master WHERE type IN ('table','view')")
    rows = cur.fetchall()

    fts_schema = None
    plain_schema = None

    for row in rows:
        name, sql, kind = row["name"], row["sql"] or "", row["type"]
        if name.startswith("sqlite_"):
            continue

        if "VIRTUAL TABLE" in sql.upper() and "FTS" in sql.upper():
            cols = _wiki_parse_fts_columns(sql)
            title_col = _wiki_pick_col(cols, _WIKI_TITLE_COL_HINTS)
            text_col = _wiki_pick_col(cols, _WIKI_TEXT_COL_HINTS)
            if title_col and text_col and fts_schema is None:
                fts_schema = {
                    "mode": "fts", "table": name, "title_col": title_col,
                    "text_col": text_col, "columns": cols,
                }
            continue

        if plain_schema is not None:
            continue
        try:
            cur.execute(f"PRAGMA table_info({_wiki_quote_ident(name)})")
            cols = [c["name"] for c in cur.fetchall()]
        except Exception:
            continue
        title_col = _wiki_pick_col(cols, _WIKI_TITLE_COL_HINTS)
        text_col = _wiki_pick_col(cols, _WIKI_TEXT_COL_HINTS)
        if title_col and text_col:
            plain_schema = {"mode": "plain", "table": name, "title_col": title_col, "text_col": text_col}

    return fts_schema or plain_schema


def _get_wiki_db():
    """Lazily open a read-only connection to WIKI_DB_PATH and auto-detect its
    schema. Returns None (permanently, for this process) if the feature is
    disabled, the file is missing, or no usable article table is found."""
    global _WIKI_DB_CONN, _WIKI_DB_SCHEMA, _WIKI_DB_UNAVAILABLE_REASON

    if _WIKI_DB_CONN is not None:
        return _WIKI_DB_CONN
    if _WIKI_DB_UNAVAILABLE_REASON is not None:
        return None
    if not ENABLE_WIKI_CONTEXT:
        _WIKI_DB_UNAVAILABLE_REASON = "disabled"
        return None
    if not os.path.exists(WIKI_DB_PATH):
        _WIKI_DB_UNAVAILABLE_REASON = f"not found at {WIKI_DB_PATH}"
        return None

    try:
        import sqlite3
        uri = f"file:{os.path.abspath(WIKI_DB_PATH)}?mode=ro"
        conn = sqlite3.connect(uri, uri=True, timeout=WIKI_DB_TIMEOUT_S, check_same_thread=False)
        conn.row_factory = sqlite3.Row

        schema = _wiki_detect_schema(conn)
        if schema is None:
            conn.close()
            _WIKI_DB_UNAVAILABLE_REASON = "no title/text article table detected"
            print(
                "  (Wikipedia context DB found but its schema wasn't recognized "
                "— set ENABLE_WIKI_CONTEXT = False or check WIKI_DB_PATH's tables)"
            )
            return None

        _WIKI_DB_CONN = conn
        _WIKI_DB_SCHEMA = schema
        return conn
    except Exception as e:
        _WIKI_DB_UNAVAILABLE_REASON = str(e)
        return None


def _wiki_strip_possessive(word):
    """Strip a trailing possessive so search tokens actually match indexed
    text: "Radiohead's" -> "Radiohead", "Beatles'" -> "Beatles". Without
    this, our own tokenizer keeps the apostrophe-s attached to the token
    (see the regex below), so the FTS query ends up searching for the
    literal string "radiohead's" — which essentially never appears in
    indexed article text, since FTS5's own tokenizer splits that into
    "radiohead" + "s" — and the lookup silently finds nothing."""
    w = re.sub(r"[\u2019']s$", "", word, flags=re.IGNORECASE)
    w = re.sub(r"[\u2019']$", "", w)
    return w


def _wiki_fold_diacritics(text):
    """ASCII-fold diacritics for query expansion: 'Björk' -> 'Bjork'.

    Used only to *add* extra search strings alongside the original form.
    Does not replace or mutate anything stored in the wiki DB, and does not
    change FTS/plain lookup logic — callers simply submit both variants."""
    if not text:
        return ""
    # NFKD splits ö → o + combining diaeresis; drop combining marks.
    decomposed = unicodedata.normalize("NFKD", str(text))
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def _wiki_fts_query_escape(q):
    """Quote each token individually so punctuation/apostrophes in artist
    names (e.g. Guns N' Roses) can't break FTS5 query syntax, after first
    stripping any trailing possessive so "Radiohead's" searches for
    "Radiohead" rather than the literal (unmatchable) token "Radiohead's".

    Also keeps common title punctuation (! ? . &) and Unicode letters so
    pages like "Björk" or "Yes Please!" are not token-stripped away."""
    # \w already matches Unicode letters (ö etc.) under Python 3 default flags.
    # Include a small set of title punctuation so "Please!" / "R.E.M." stay intact.
    tokens = re.findall(r"[\w'&.!?-]+", q or "")
    tokens = [_wiki_strip_possessive(t) for t in tokens]
    return " ".join(f'"{t}"' for t in tokens if t)


_WIKI_QUESTION_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "do", "does", "did", "this", "that",
    "these", "those", "what", "when", "where", "who", "whom", "which", "why", "how",
    "in", "on", "of", "for", "to", "and", "or", "it", "its", "song", "track", "album",
    "albums", "band", "artist", "music", "about", "tell", "me", "us", "can", "you",
    "please", "i", "we", "they", "he", "she", "be", "been", "with", "from", "by", "like",
    "good", "bad", "any", "some", "all", "just", "really", "very", "also", "too",
    "than", "then", "there", "here", "into", "over", "under", "after", "before",
    "debut", "first", "second", "third", "self", "titled", "eponymous", "full",
    "list", "listing", "tracklist", "tracklisting", "songs", "tracks", "reception",
    "reviews", "review", "regarded", "critical", "commercially", "commercially",
    "best", "greatest", "top", "favorite", "favourite", "favorites", "favourites",
    "worst", "essential", "underrated", "overrated",
    # Pronouns / follow-up filler that must never become wiki search tokens
    "them", "him", "her", "their", "theirs", "more", "else", "something", "anything",
    "someone", "anyone", "stuff", "things", "thing", "one", "ones",
}

# Sentence-initial words to strip off a capitalized-phrase match so e.g.
# "What Radiohead album..." yields the entity "Radiohead album" rather than
# "What Radiohead" — these are common question-starters, not band/album names.
_WIKI_LEADING_STOP_CAPS = {
    "the", "a", "an", "what", "who", "when", "where", "why", "how", "does", "did",
    "is", "are", "can", "could", "would", "will", "should", "tell", "please", "i",
    "was", "were", "do", "about",
}

# Words that commonly appear inside album/artist names and should not break a
# title-case run (e.g. "Guns N' Roses", "Of Montreal", "The Beatles").
_WIKI_TITLE_INFIX = {
    "a", "an", "the", "of", "and", "or", "n", "n'", "for", "to", "in", "on", "at",
    "vs", "vs.", "with", "&",
}


def _wiki_question_keywords(question, max_terms=6):
    """Pull a handful of content words out of a user question for use as
    extra FTS search terms, dropping common stopwords/question words so the
    search isn't over-constrained by filler like 'what' or 'the'."""
    if not question:
        return []
    words = re.findall(r"[A-Za-z0-9']+", question.lower())
    seen = set()
    out = []
    for w in words:
        if len(w) > 2 and w not in _WIKI_QUESTION_STOPWORDS and w not in seen:
            seen.add(w)
            out.append(w)
    return out[:max_terms]


def _wiki_normalize_entity(phrase):
    phrase = _wiki_strip_possessive((phrase or "").strip())
    # Keep ! (real titles e.g. "Yes Please!") but drop a trailing sentence "?"
    # that often sticks to the last token when the user asked a question.
    phrase = re.sub(r"\s+", " ", phrase).strip(" .,;:\"'")
    if phrase.endswith("?") and not phrase.endswith("!?"):
        phrase = phrase[:-1].rstrip()
    return phrase


def _wiki_extract_by_patterns(question):
    """Pull album/artist pairs from common English patterns that the pure
    title-case heuristic often misses, especially when the user doesn't
    capitalize every word:

      - "Yes Please by Happy Mondays"
      - "Happy Mondays' Yes Please"
      - "Happy Mondays album Yes Please"
      - "the album Yes Please"
    """
    if not question:
        return []

    entities = []
    q = question

    # Unicode-aware "starts with a capital letter" (covers Björk, Ö, etc.).
    # [A-Z] alone is ASCII-only and silently dropped accented capitals.
    _cap = r"[A-Z\u00C0-\u00D6\u00D8-\u00DE]"
    # Continuation chars: word chars (incl. Unicode letters) + common title punct.
    _cont = r"[\w'&.!?-]*"

    # "TITLE by ARTIST" — locate "by ARTIST" first (artist is title-case),
    # then take the 1–4 tokens immediately before "by" as the title. This
    # avoids the left-greedy span "Tell me about Yes Please by …".
    for m in re.finditer(
        rf"\sby\s+(?P<artist>{_cap}{_cont}(?:\s+(?:{_cap}{_cont}|N['\u2019]?)){{0,5}})",
        q,
    ):
        artist = _wiki_normalize_entity(m.group("artist"))
        # Tokens immediately before " by "
        before = q[:m.start()]
        before_tokens = re.findall(rf"[A-Za-z0-9\u00C0-\u00FF]{_cont}", before)
        # Drop trailing stopwords from the left context, then keep up to 4
        while before_tokens and before_tokens[-1].lower() in _WIKI_QUESTION_STOPWORDS | _WIKI_LEADING_STOP_CAPS:
            before_tokens.pop()
        title_tokens = before_tokens[-4:]
        while title_tokens and title_tokens[0].lower() in _WIKI_LEADING_STOP_CAPS | _WIKI_QUESTION_STOPWORDS:
            title_tokens = title_tokens[1:]
        # A real album/song title in a user's question is almost always
        # capitalized ("Yes Please by Happy Mondays"). Generic descriptive
        # phrasing that slips past the stopword strip above ("the best
        # albums by X", "any good songs by X") is not — so require at least
        # one capitalized token before trusting this as a title entity,
        # rather than searching the DB for whatever words happened to sit
        # before "by". Without this, a phrase like "best albums" gets
        # treated as a literal release title and matches generic pages
        # (award lists, "best of" compilations) that merely contain those
        # words, instead of the artist's own page.
        if title_tokens and not any(re.match(rf"{_cap}", t) for t in title_tokens):
            title_tokens = []
        title = _wiki_normalize_entity(" ".join(title_tokens))
        if artist and len(artist) > 1:
            entities.append(artist)
            if title and len(title) > 1:
                entities.append(title)
                entities.append(f"{artist} {title}")

    # "ARTIST's ALBUM" / "ARTIST' ALBUM" — artist is title-case; rest may be
    # the album title or words like "debut album".
    for m in re.finditer(
        rf"\b(?P<artist>{_cap}{_cont}(?:\s+(?:{_cap}{_cont}|N['\u2019]?)){{0,5}})"
        r"[\u2019']s?\s+"
        rf"(?P<rest>[A-Za-z0-9\u00C0-\u00FF]{_cont}(?:\s+[A-Za-z0-9\u00C0-\u00FF]{_cont}){{0,5}})",
        q,
    ):
        artist = _wiki_normalize_entity(m.group("artist"))
        # Strip leading stop-caps from artist ("What was Radiohead" -> "Radiohead")
        awords = artist.split()
        while awords and awords[0].lower() in _WIKI_LEADING_STOP_CAPS:
            awords = awords[1:]
        artist = " ".join(awords)
        rest = _wiki_normalize_entity(m.group("rest"))
        rwords = rest.split()
        while rwords and rwords[-1].lower() in (
            _WIKI_ALBUM_SIGNAL_WORDS
            | {"debut", "first", "self-titled", "eponymous", "self", "titled", "like", "about"}
        ):
            rwords = rwords[:-1]
        rest = " ".join(rwords)
        if artist and len(artist) > 1:
            entities.append(artist)
            if rest and len(rest) > 1 and rest.lower() not in _WIKI_QUESTION_STOPWORDS:
                entities.append(f"{artist} {rest}")
                entities.append(rest)

    # "album TITLE" / "record TITLE"
    for m in re.finditer(
        r"\b(?:album|record|lp|ep)\s+[\"'\u2018\u2019\u201c\u201d]?"
        rf"([A-Za-z0-9\u00C0-\u00FF]{_cont}(?:\s+[A-Za-z0-9\u00C0-\u00FF]{_cont}){{0,5}})"
        r"[\"'\u2018\u2019\u201c\u201d]?",
        q,
        re.IGNORECASE,
    ):
        title = _wiki_normalize_entity(m.group(1))
        twords = title.split()
        while twords and twords[-1].lower() in _WIKI_QUESTION_STOPWORDS | {"like", "about"}:
            twords = twords[:-1]
        title = " ".join(twords)
        if title and len(title) > 1 and title.lower() not in _WIKI_ALBUM_SIGNAL_WORDS:
            entities.append(title)

    return entities



def _wiki_title_case_runs(question):
    """Find runs of title-case / proper-noun-ish words, allowing small
    lowercase infix words (of, the, and, N') so 'Guns N' Roses' and
    'The Stone Roses' still match as single entities.

    Token classes are Unicode-aware so names like Björk / Sigur Rós and
    titles with trailing ! (Yes Please!) are kept intact."""
    if not question:
        return []

    _cap = r"[A-Z\u00C0-\u00D6\u00D8-\u00DE]"
    # Tokenize preserving apostrophes, diacritics, and common title punctuation
    tokens = re.findall(
        rf"[A-Za-z0-9\u00C0-\u00FF][\w'&.!?-]*|[\"'\u2018\u2019\u201c\u201d]",
        question,
    )
    entities = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if not re.match(rf"{_cap}", tok):
            i += 1
            continue
        run = [tok]
        j = i + 1
        while j < len(tokens):
            nxt = tokens[j]
            if re.match(rf"{_cap}", nxt):
                run.append(nxt)
                j += 1
            elif nxt.lower().strip(".'!?") in _WIKI_TITLE_INFIX and j + 1 < len(tokens) and re.match(rf"{_cap}", tokens[j + 1]):
                run.append(nxt)
                j += 1
            else:
                break
        # Strip leading stop-caps
        while run and run[0].lower() in _WIKI_LEADING_STOP_CAPS:
            run = run[1:]
        phrase = _wiki_normalize_entity(" ".join(run))
        if phrase and len(phrase) > 2:
            # Prefer multi-word; single capitalized words only if reasonably long
            # (avoids "Tell", "Is", but keeps "Radiohead", "Blur", "Björk")
            if " " in phrase or len(phrase) >= 5:
                entities.append(phrase)
        i = j if j > i else i + 1
    return entities



def _wiki_question_entity_queries(question, max_entities=5):
    """Heuristically pull candidate entity phrases (artist/album/song names)
    out of a free-form question. Combines:
      - quoted phrases
      - "TITLE by ARTIST" / possessive patterns
      - title-case runs (with small-word infix tolerance)
      - keyword fallback
    """
    if not question:
        return []

    entities = []

    for m in re.finditer(
        r'["\u2018\u2019\u201c\u201d]([^"\u2018\u2019\u201c\u201d]{2,60})["\u2018\u2019\u201c\u201d]',
        question,
    ):
        phrase = _wiki_normalize_entity(m.group(1))
        if phrase:
            entities.append(phrase)

    entities.extend(_wiki_extract_by_patterns(question))
    entities.extend(_wiki_title_case_runs(question))

    # Drop pure stopword / filler entities and lone words that are too generic
    junk = _WIKI_QUESTION_STOPWORDS | {
        "like", "about", "badly", "well", "good", "bad", "debut", "first",
        "second", "third", "self", "titled", "eponymous", "full", "list",
        "songs", "tracks", "reception", "review", "reviews", "regarded",
        "them", "him", "her", "more", "else", "something", "anything",
    }
    seen = set()
    out = []
    for e in entities:
        k = e.lower().strip()
        if not k or k in seen:
            continue
        if k in junk:
            continue
        # Lone short words are usually noise unless they look like a band name
        # (Blur, Oasis, ABBA) — keep length >= 4 for singles, or any multi-word.
        if " " not in e and len(e) < 4:
            continue
        seen.add(k)
        out.append(e)

    if out:
        return out[:max_entities]

    kw = _wiki_question_keywords(question, max_terms=6)
    # Only use keyword fallback when we have at least one solid content token
    # (avoids feeding "badly regarded" style questions as pseudo-entities).
    solid = [w for w in kw if w not in junk and len(w) > 3]
    if len(solid) >= 2:
        return [" ".join(solid[:4])]
    return []


_WIKI_ALBUM_SIGNAL_WORDS = {
    "album", "albums", "record", "records", "lp", "ep", "release", "released",
    "compilation", "compilations", "anthology", "anthologies",
}

_WIKI_TRACKLIST_SIGNAL = {
    "track", "tracks", "tracklist", "tracklisting", "track-list", "songs", "song",
    "listing", "side", "sides", "disc", "cds", "discography", "discog",
    "albums", "releases", "catalogue", "catalog", "singles",
}

_WIKI_RECEPTION_SIGNAL = {
    "reception", "review", "reviews", "reviewed", "regarded", "critical",
    "critics", "acclaim", "criticised", "criticized", "response", "chart",
    "charts", "sales", "sold", "peaked", "rating", "ratings", "good", "bad",
    "poorly", "well", "received", "commercial", "commercially",
}

_WIKI_DEBUT_SIGNAL = {
    "debut", "first album", "first record", "eponymous", "self-titled",
    "self titled", "same name",
}

# Multi-word phrases that signal a compilation/best-of release, checked as
# substrings (like _WIKI_DEBUT_SIGNAL) since "greatest hits" and "best of"
# don't survive single-word tokenization the way "compilation" does.
_WIKI_COMPILATION_PHRASES = (
    "greatest hits", "best of", "hits collection", "singles collection",
)


def _wiki_mentions_album(question):
    """Whether the question seems to be asking about an album at all (vs. a
    song, the artist generally, etc.) — used to trigger the self-titled-
    album search tier below, not as a literal search token."""
    if not question:
        return False
    q = question.lower()
    words = set(re.findall(r"[a-z0-9']+", q))
    if words & _WIKI_ALBUM_SIGNAL_WORDS:
        return True
    if words & _WIKI_TRACKLIST_SIGNAL or words & _WIKI_RECEPTION_SIGNAL:
        return True
    if any(s in q for s in _WIKI_DEBUT_SIGNAL):
        return True
    if any(s in q for s in _WIKI_COMPILATION_PHRASES):
        return True
    return False


_WIKI_ANAPHORA_RE = re.compile(
    r"\b(it|its|it's|they|their|theirs|them|"
    r"this (?:song|track|album|record|single|lp|ep|one|artist|band|group|act)|"
    r"that (?:song|track|album|record|single|lp|ep|one|artist|band|group|act)|"
    r"the (?:song|track|album|record|artist|band|group|act|singer|vocalist)\b(?!\s+[A-Z])|"
    r"(?:more|tell me more|what else|anything else) about (?:them|it|him|her|the (?:album|song|track|band|artist|group)))",
    re.IGNORECASE,
)


def _wiki_has_anaphora(question):
    """Whether the question refers back to something ('it', 'their',
    'this album') rather than naming its own subject — the strongest signal
    that this is a follow-up needing continuity from recent history rather
    than a question that stands on its own."""
    return bool(question and _WIKI_ANAPHORA_RE.search(question))


def _wiki_is_track_referential(question):
    """True when the question is about the currently discussed track / artist /
    album without introducing a new named subject of its own.

    Used to keep track-identity wiki queries (artist, album, title from file
    tags) at the front of the search list instead of letting noisy entities
    scraped from analysis dumps or incidental capitalized words dominate.
    """
    if not question:
        return False
        
    # [PATCH] Image-referential override:
    # If the user is explicitly referring to a newly uploaded image/photo, 
    # or asking a generic identity question (common with image drops), 
    # break the history inheritance so we don't inject the previous 
    # track's Wikipedia page and cause an observation hallucination.
    q_lower = question.lower()
    image_signals = {"image", "photo", "picture", "pic", "cover"}
    words = set(re.findall(r"[a-z0-9']+", q_lower))
    
    if words & image_signals or re.search(r"\bwho\s+(is|are|was|'s)\s+this\b", q_lower):
        return False

    entities = _wiki_question_entity_queries(question, max_entities=5)
    # Any solid named entity in the question (multi-word phrase OR a reasonably
    # long single proper noun like "Radiohead" / "Blur") means the user is
    # naming a subject themselves — do not force loaded-track identity first.
    # Anaphora still wins only when those entities are absent.
    solid = [
        e for e in entities
        if ((" " in e and len(e) >= 5) or (len(e) >= 4 and e[:1].isupper()))
    ]
    if solid:
        return False
    if _wiki_has_anaphora(question):
        return True
    focus = _wiki_question_focus(question)
    # Album/artist-focused questions with no external named subject are about
    # the loaded material ("tell me about the album", "who are they").
    if focus in ("tracklist", "reception", "album", "debut"):
        return True
    # Short bare follow-ups with no entities at all.
    if not entities and len(question.split()) <= 12:
        return True
    return False


def _wiki_question_focus(question):
    """Coarse focus tag for the question: tracklist | reception | debut | album | general.
    Used to prefer the right Wikipedia page and the right sections inside it."""
    if not question:
        return "general"
    q = question.lower()
    words = set(re.findall(r"[a-z0-9']+", q))
    if any(s in q for s in _WIKI_DEBUT_SIGNAL) or ("debut" in words):
        return "debut"
    if words & _WIKI_TRACKLIST_SIGNAL:
        return "tracklist"
    if words & _WIKI_RECEPTION_SIGNAL:
        return "reception"
    if words & _WIKI_ALBUM_SIGNAL_WORDS or _wiki_mentions_album(question):
        return "album"
    return "general"


def _wiki_strip_evidence_boilerplate(content):
    """Strip large private-evidence / analysis dumps from a history message
    before entity extraction. Without this, title-case runs over Music Flamingo
    field names, GENRE labels, and random capitalized words in the analysis
    flood the wiki search with unrelated queries and crowd out the real
    artist/album anchors."""
    if not content:
        return ""
    text = content
    # Drop wiki blocks from earlier turns
    text = re.split(r"\n\n=== WIKIPEDIA BACKGROUND CONTEXT", text, maxsplit=1)[0]
    # Drop private track notes / analysis body (keep the short header line)
    cut_markers = (
        "=== PRIVATE TRACK NOTES",
        "(background technical details restored",
        "TRACK METADATA (from the audio file",
        "11. ERA / RELEASE PERIOD",
        "VOCAL / SINGER PROFILE",
        "RECOMMENDED TEMPO FOR DISCUSSION",
        "RECOMMENDED KEY FOR DISCUSSION",
        "STEM MIDI REPORT",
        "[Independent signal-processing report",
        "FULL LYRICS TRANSCRIPTION",
        "SINGER IDENTITY RESOLUTION",
        "COVER ART OBSERVATIONS",
        "CONFIRMED CORRECTIONS FOR THIS TRACK",
        "VOCAL DECISION AUDIT",
        "WHOLE-MIX INSTRUMENT TAGS",
    )
    for marker in cut_markers:
        idx = text.find(marker)
        if idx != -1:
            text = text[:idx]
    # Prefer the short identity header when present
    header = ""
    for pat in (
        r"\[We're listening to:\s*([^\]]+)\]",
        r"\[Loaded saved track\s+'([^']+)'\]",
        r'\[Loaded saved track\s+"([^"]+)"\]',
    ):
        hm = re.search(pat, content, re.IGNORECASE)
        if hm:
            header = hm.group(1).strip()
            break
    if header:
        # Prefer a clean "Title by Artist" / label form over residual prompt text
        text = header
    # Hard cap: never scan more than a short prefix of residual content
    return text[:400]



def _wiki_entities_from_recent_history(writer_history, max_messages=8, max_entities=4):
    """When the user asks a short follow-up ('what were its tracks?'), pull
    album/artist entities from recent user turns so the wiki lookup stays
    anchored to the same subject instead of returning empty.

    Only the short identity headers / user questions are scanned — never the
    full analysis dumps, which otherwise inject dozens of false-positive
    capitalized phrases into the search queue.
    """
    if not writer_history:
        return []

    entities = []
    seen = set()
    # Walk recent user messages newest-first
    user_msgs = [
        m for m in writer_history
        if isinstance(m, dict) and m.get("role") == "user" and isinstance(m.get("content"), str)
    ]
    for m in reversed(user_msgs[-max_messages:]):
        content = _wiki_strip_evidence_boilerplate(m["content"])
        if not content.strip():
            continue
        for e in _wiki_question_entity_queries(content, max_entities=4):
            k = e.lower()
            # Skip analysis-field / boilerplate pseudo-entities
            if k in {
                "private track notes", "track metadata", "recommended tempo",
                "recommended key", "singer identity", "cover art", "genre ranked",
                "full lyrics", "vocal decision", "stem midi", "music flamingo",
            }:
                continue
            if k not in seen:
                seen.add(k)
                entities.append(e)
        if len(entities) >= max_entities:
            break
    return entities[:max_entities]


def _wiki_split_sections(text):
    """Split a Wikipedia plain-text dump into (heading, body) pairs.

    Handles common export styles: '== Heading ==', 'Heading\\n----', or a
    line that looks like a section title (short, title-ish) followed by body.
    Always yields a leading ('', lead_text) chunk for the article intro.
    """
    if not text:
        return [("", "")]

    # Prefer explicit wiki heading markers if present
    parts = re.split(r"(?m)^(?:={2,}\s*([^=\n]{2,80}?)\s*={2,})\s*$", text)
    if len(parts) > 1:
        sections = [("", parts[0])]
        i = 1
        while i + 1 < len(parts):
            sections.append((parts[i].strip(), parts[i + 1]))
            i += 2
        return sections

    # Fallback: scan for well-known section title lines
    known = (
        r"Track\s*listing|Track\s*list|Tracklist|Personnel|Credits|"
        r"Critical\s+reception|Reception|Critical\s+response|Reviews?|"
        r"Commercial\s+performance|Charts?|Singles?|Release|"
        r"Background|Recording|Production|Composition|Legacy|"
        r"Discography|Studio\s+albums|Album\s+list|Selected\s+discography|"
        r"Compilation\s+albums|Live\s+albums|EPs?"
    )
    pattern = re.compile(
        rf"(?im)(?:^|\n)(?:={0,6}\s*)({known})(?:\s*={0,6})?\s*(?:\n|$)"
    )
    matches = list(pattern.finditer(text))
    if not matches:
        return [("", text)]

    sections = []
    if matches[0].start() > 0:
        sections.append(("", text[: matches[0].start()]))
    for i, m in enumerate(matches):
        heading = m.group(1).strip()
        body_start = m.end()
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sections.append((heading, text[body_start:body_end]))
    return sections


def _wiki_section_relevance(heading, body, focus):
    """Score how useful a section is for the current question focus."""
    h = (heading or "").lower()
    b = (body or "")[:500].lower()
    score = 0

    tracklist_h = ("track", "listing", "tracklist", "side a", "side b", "personnel", "credits")
    reception_h = ("reception", "review", "critical", "commercial", "chart", "legacy", "acclaim")
    disco_h = ("discography", "studio album", "album list", "selected discography", "compilation", "live album", "eps")
    background_h = ("background", "recording", "production", "composition", "release", "single")

    def hit(words):
        return any(w in h for w in words) or any(w in b[:120] for w in words)

    if focus == "tracklist":
        if hit(tracklist_h):
            score += 10
        if hit(disco_h):
            score += 8
        if hit(reception_h):
            score += 3
        if hit(background_h):
            score += 2
    elif focus == "reception":
        if hit(reception_h):
            score += 10
        if hit(tracklist_h):
            score += 3
        if hit(background_h):
            score += 2
    elif focus == "debut":
        if hit(background_h) or hit(tracklist_h) or hit(reception_h):
            score += 6
        if hit(disco_h):
            score += 4
    elif focus == "album":
        if hit(tracklist_h) or hit(reception_h) or hit(background_h):
            score += 6
    else:
        if hit(disco_h):
            score += 5
        if hit(background_h) or hit(reception_h):
            score += 3

    # Numbered track-like content even without a clean heading
    if focus in ("tracklist", "album", "debut") and re.search(
        r"(?m)^\s*(?:\d{1,2}[\).\:]|Track\s*\d)", body[:1500]
    ):
        score += 7

    # Year + album title patterns typical of discography tables
    if focus in ("tracklist", "general", "debut") and re.search(
        r"\b(?:19|20)\d{2}\b.{0,40}\b(?:album|ep|lp|studio)\b", body[:1500], re.I
    ):
        if hit(disco_h) or "discog" in h:
            score += 5

    return score


def _wiki_pick_relevant_snippet(raw_text, question, max_chars):
    """Prefer deep article sections (Track listing, Discography, Reception)
    over only the lead. Wikipedia often puts these further down the page;
    without them the writer model tends to invent track lists from memory.
    """
    if not raw_text:
        return ""

    text = str(raw_text)
    focus = _wiki_question_focus(question)
    sections = _wiki_split_sections(text)

    # Always keep a short lead for identity / release context
    lead_budget = min(700, max_chars // 4) if focus in ("tracklist", "reception", "debut", "album") else min(900, max_chars // 2)
    lead_text = " ".join((sections[0][1] if sections else text).split())
    if len(lead_text) > lead_budget:
        lead_text = lead_text[:lead_budget].rsplit(" ", 1)[0] + "…"

    scored = []
    for heading, body in sections[1:] if len(sections) > 1 else []:
        body_clean = body.strip()
        if len(body_clean) < 40:
            continue
        sc = _wiki_section_relevance(heading, body_clean, focus)
        if sc <= 0 and focus not in ("general",):
            continue
        scored.append((sc, heading, body_clean))

    # If section split failed, fall back to regex windows over the full text
    if not scored:
        fallback_pats = []
        if focus in ("tracklist", "album", "debut", "general"):
            fallback_pats += [
                r"(?i)\b(?:track\s*listing|track\s*list|tracklist|discography|studio\s+albums?)\b",
                r"(?i)(?:^|\n)\s*1[\).\:]\s+[\"'A-Za-z]",
            ]
        if focus in ("reception", "album", "debut", "general"):
            fallback_pats += [
                r"(?i)\b(?:critical\s+reception|critical\s+response|commercial\s+performance)\b",
            ]
        for pat in fallback_pats:
            for m in re.finditer(pat, text):
                start = max(0, m.start() - 20)
                end = min(len(text), m.start() + max(1200, max_chars // 2))
                chunk = text[start:end]
                scored.append((6, "match", chunk))
                if len(scored) >= 4:
                    break
            if len(scored) >= 4:
                break

    scored.sort(key=lambda x: -x[0])

    pieces = []
    total = 0
    if lead_text:
        pieces.append(lead_text)
        total += len(lead_text)

    seen_words = set(re.findall(r"[a-z0-9]{4,}", lead_text.lower()))
    for sc, heading, body in scored:
        body_one = " ".join(body.split())
        # Per-section budget: give tracklist/discography more room
        sec_cap = max_chars - total
        if sec_cap < 150:
            break
        if focus == "tracklist" and sc >= 7:
            take = min(len(body_one), max(sec_cap, min(1800, max_chars // 2)))
        elif focus == "reception" and sc >= 7:
            take = min(len(body_one), max(sec_cap, min(1400, max_chars // 2)))
        else:
            take = min(len(body_one), sec_cap)

        # Avoid near-duplicate sections
        words = set(re.findall(r"[a-z0-9]{4,}", body_one[:800].lower()))
        if words and len(words & seen_words) >= max(4, len(words) // 2):
            continue
        seen_words |= words

        chunk = body_one[:take]
        if len(body_one) > take:
            chunk = chunk.rsplit(" ", 1)[0] + "…"
        label = f"[{heading}] " if heading and heading != "match" else ""
        pieces.append(label + chunk)
        total += len(chunk)
        if total >= max_chars:
            break

    if len(pieces) == 1 and not scored:
        # Pure lead fallback
        full = " ".join(text.split())
        if len(full) > max_chars:
            return full[:max_chars].rsplit(" ", 1)[0] + "…"
        return full

    return "\n\n".join(pieces)


def _wiki_rank_prefer_album_title(rows, query_hint=""):
    """Among FTS hits, prefer titles that actually look like the thing being
    asked about — an exact/near match on the query, or an '(album)'/'(EP)'
    disambiguator — over pages that just happen to mention the artist
    somewhere in the body (a member's solo project, a video, a related
    label). Most real album titles on Wikipedia are NOT disambiguated
    ('Zen Arcade', not 'Zen Arcade (album)') — only ambiguous ones are — so
    this can't rely on the '(album)' suffix alone; it also has to reward a
    close title match on its own.

    Diacritics are folded for comparison so a query of "Bjork" ranks the
    title "Björk" as an exact match above "List of songs recorded by Björk".
    """
    if not rows:
        return rows

    def score(title):
        t = (title or "").strip()
        tl = t.lower()
        tl_fold = _wiki_fold_diacritics(tl).lower()
        s = 0
        if "(album)" in tl or "(ep)" in tl or "(lp)" in tl:
            s += 5
        if query_hint:
            qh = query_hint.lower()
            qh_fold = _wiki_fold_diacritics(qh).lower()
            # Exact title match (diacritic-insensitive) is the strongest signal —
            # this is what makes "Bjork" prefer the page "Björk" over list pages.
            if tl == qh or tl_fold == qh_fold:
                s += 20
            elif tl_fold.startswith(qh_fold + " (") or tl_fold.startswith(qh_fold + ","):
                # "Radiohead (band)", "Björk (singer)" style disambiguators
                s += 14
            elif qh_fold == tl_fold.split("(")[0].strip():
                s += 12
            elif qh_fold in tl_fold or tl_fold in qh_fold:
                # Substring match only — much weaker than exact title
                s += 3
            # Prefer titles that contain most/all of the query's tokens —
            # this is what actually catches unadorned album titles like
            # "Zen Arcade" for a query built from that same phrase.
            tokens = [w for w in re.findall(r"[a-z0-9]+", qh_fold) if len(w) > 2]
            if tokens:
                hits = sum(1 for tok in tokens if tok in tl_fold)
                s += hits
                if tokens and hits == len(tokens):
                    s += 3
        # Pages that are ABOUT something else — a member's side project, a
        # video, a specific song/single, a label — but merely mention the
        # queried artist/album in passing are exactly the false positives
        # this function exists to suppress.
        if re.search(r"\([^)]*\b(?:video|film|song|single|record label)\b[^)]*\)", tl):
            s -= 4
        if tl.endswith(" (band)") or tl.endswith(" (musician)") or tl.endswith(" (singer)"):
            # Mild penalty only when we already have a stronger exact undambiguated
            # title candidate; the disambiguated form is still a real artist page.
            s -= 1
        # "List of songs recorded by X" / "List of awards..." crowd out the
        # actual artist page for short name queries — demote hard.
        if re.match(r"^list of\b", tl) or tl.startswith("lists of "):
            s -= 15
        if "discography" in tl and "(" not in tl:
            # Bare "X discography" is useful; still below the main bio for
            # "tell me about X" but above random list pages.
            s -= 2
        return s

    return sorted(rows, key=lambda r: score(r["t"] if isinstance(r, dict) else r[0]), reverse=True)


def _wiki_multi_search(queries, max_articles=None, prefer_album=False):
    """Run several independent searches (each a string of search terms) and
    collect up to max_articles DISTINCT articles, deduped by title, in the
    order the queries are given. This is what lets one request pull in the
    song article AND the album article (with its track listing) AND the
    artist article together, instead of stopping at a single best guess."""
    conn = _get_wiki_db()
    if conn is None:
        return []

    max_articles = WIKI_MAX_ARTICLES if max_articles is None else max_articles
    schema = _WIKI_DB_SCHEMA
    table = _wiki_quote_ident(schema["table"])
    title_col = _wiki_quote_ident(schema["title_col"])
    text_col = _wiki_quote_ident(schema["text_col"])
    cur = conn.cursor()

    # Weight the title column much more heavily than body text in FTS
    # ranking. Without this, FTS5's default `rank` (bm25 across all indexed
    # columns equally) treats a body mention the same as a title match — so
    # a short, dense page that happens to name-check the artist a few times
    # (a member's solo project, a video, a related label) can out-rank the
    # actual artist/album page, where the same terms appear less densely
    # relative to a much longer article. Boosting the title column fixes
    # that without needing to know anything about the DB's specific schema.
    _bm25_order = "rank"
    if schema.get("columns"):
        weights = [
            "10.0" if c == schema["title_col"] else "1.0"
            for c in schema["columns"]
        ]
        _bm25_order = f"bm25({table}, {', '.join(weights)})"

    seen_titles = set()
    results = []

    # Do not let the first successful query monopolise the whole result set.
    # Previously an artist/album query could return three highly-ranked pages
    # and prevent later queries (song, album, artist, question entity) from
    # ever contributing. After a few turns this looked like a stuck cache
    # because the same three articles kept winning every request.
    # Reserve room for multiple query families, then use the remaining slots
    # for the strongest matches.
    per_query_budget = max(1, min(2, max_articles // 2))

    for q in queries:
        if len(results) >= max_articles:
            break
        if not q or not str(q).strip():
            continue

        rows = []
        try:
            if schema["mode"] == "fts":
                fts_q = _wiki_fts_query_escape(q)
                if not fts_q:
                    continue
                # When we want an album page, also try an explicit "(album)" disambiguator
                fts_variants = [fts_q]
                if prefer_album and "(album)" not in q.lower():
                    fts_variants.append(_wiki_fts_query_escape(f"{q} album"))
                for fv in fts_variants:
                    try:
                        cur.execute(
                            f"SELECT {title_col} AS t, {text_col} AS b FROM {table} "
                            f"WHERE {table} MATCH ? ORDER BY {_bm25_order} LIMIT ?",
                            (fv, WIKI_SEARCH_ROW_LIMIT),
                        )
                        batch = cur.fetchall()
                    except Exception:
                        # Title-weighted bm25() can fail on some FTS5 configs
                        # (e.g. contentless tables where column weighting
                        # isn't supported the same way) — fall back to the
                        # plain default rank rather than losing the query.
                        cur.execute(
                            f"SELECT {title_col} AS t, {text_col} AS b FROM {table} "
                            f"WHERE {table} MATCH ? ORDER BY rank LIMIT ?",
                            (fv, WIKI_SEARCH_ROW_LIMIT),
                        )
                        batch = cur.fetchall()
                    if batch:
                        rows.extend(batch)
                # Always re-rank so exact title matches (e.g. "Björk" for query
                # "Bjork") beat list/discography pages that merely contain the name.
                if rows:
                    rows = _wiki_rank_prefer_album_title(rows, query_hint=q)
            else:
                # No FTS index: only cheap, targeted lookups — a full LIKE
                # scan over a multi-GB table with no index is too slow to
                # run repeatedly per request. Strip possessives here too,
                # word by word, so "Radiohead's" still exact/prefix-matches
                # a title of "Radiohead".
                q_clean = " ".join(_wiki_strip_possessive(w) for w in str(q).split())
                q_folded = _wiki_fold_diacritics(q_clean)
                candidates = [q_clean]
                if q_folded and q_folded != q_clean:
                    candidates.append(q_folded)
                if prefer_album:
                    for base in list(candidates):
                        candidates.extend([
                            f"{base} (album)",
                            f"{base} (EP)",
                        ])
                for cand in candidates:
                    cur.execute(
                        f"SELECT {title_col} AS t, {text_col} AS b FROM {table} "
                        f"WHERE {title_col} = ? COLLATE NOCASE LIMIT 1",
                        (cand,),
                    )
                    rows = cur.fetchall()
                    if rows:
                        break
                if not rows:
                    # Prefix search on both original and folded forms
                    like_rows = []
                    for base in (q_clean, q_folded):
                        if not base:
                            continue
                        cur.execute(
                            f"SELECT {title_col} AS t, {text_col} AS b FROM {table} "
                            f"WHERE {title_col} LIKE ? COLLATE NOCASE LIMIT 5",
                            (f"{base}%",),
                        )
                        like_rows.extend(cur.fetchall())
                    rows = like_rows
                    if rows:
                        rows = _wiki_rank_prefer_album_title(rows, query_hint=q_clean)
        except Exception:
            continue

        added_this_query = 0
        for row in rows:
            t, b = row["t"], row["b"]
            if not t or not b:
                continue
            dedup_key = str(t).strip().lower()
            if dedup_key in seen_titles:
                continue
            seen_titles.add(dedup_key)
            results.append((t, b))
            added_this_query += 1
            if len(results) >= max_articles or added_this_query >= per_query_budget:
                break

    return results


def build_wiki_context_block_multi(artist=None, title=None, album=None, question=None, use_question_entities=True,
                                   extra_entities=None):
    """Return a WIKIPEDIA BACKGROUND CONTEXT block that can include MULTIPLE
    distinct articles — e.g. the song's own article, its album's article
    (with track listing / reception), and the artist's article — rather than
    a single best-guess match. Returns "" if unavailable/no confident match.

    Search queries are built from whichever of (artist, title, album) are
    given, plus — when use_question_entities is True — named-entity-like
    phrases pulled from `question` itself, so this also works for general
    questions ("what albums did Radiohead release in the 90s?") that aren't
    anchored to a specific loaded track.

    SCOPE: this is background/biographical + encyclopedic context only —
    formation, members, discography, TRACK LISTINGS, chart/critical
    reception, cultural history. It must never be used as a source for
    audio-technical facts (tempo, key, instrumentation, structure,
    production/mix qualities, vocal analysis); those come exclusively from
    the PRIVATE TRACK NOTES (Music Flamingo + Essentia + stem/MIDI), which
    are grounded in the actual audio. The rules text below enforces that
    separation for the writer model.
    """
    if not ENABLE_WIKI_CONTEXT:
        return ""
    if _get_wiki_db() is None:
        return ""

    focus = _wiki_question_focus(question)
    prefer_album = focus in ("tracklist", "reception", "album", "debut")

    # Per-article budget: give album-detail questions more room so track
    # listings and reception paragraphs survive truncation.
    per_article_cap = WIKI_CONTEXT_MAX_CHARS_PER_ARTICLE
    total_cap = WIKI_CONTEXT_TOTAL_MAX_CHARS
    if focus == "tracklist":
        # Track lists and discographies sit deep in articles and need room
        per_article_cap = max(per_article_cap, 3200)
        total_cap = max(total_cap, 7000)
    elif prefer_album:
        per_article_cap = max(per_article_cap, 2400)
        total_cap = max(total_cap, 6000)

    # total_cap must be able to hold WIKI_MAX_ARTICLES articles at
    # per_article_cap each, or the block-building loop below will silently
    # drop articles _wiki_multi_search legitimately found (WIKI_CONTEXT_TOTAL_MAX_CHARS
    # was never updated when WIKI_MAX_ARTICLES was raised, so e.g. 5000 / 1800
    # only ever fit ~2-3 of the up-to-6 articles the search actually returns).
    total_cap = max(total_cap, per_article_cap * WIKI_MAX_ARTICLES)

    queries = []

    # Explicit album/title anchors first — highest precision
    if album:
        queries.append(f"{artist} {album}" if artist else album)
        queries.append(f"{album} (album)")
        if artist:
            queries.append(f"{artist} {album} (album)")
    if title:
        queries.append(f"{artist} {title}" if artist else title)

    # Question entities. Split into "strong" (multi-word phrases — e.g. "Shiny
    # Happy People", much less likely to collide with an unrelated page) and
    # "weak" (bare single capitalized words — e.g. "Warner" from "the Warner
    # era" — which are common enough as label/company/place names to swamp
    # the result budget with irrelevant matches). Continuity entities carried
    # over from earlier turns are the best available anchor for what's
    # actually being discussed, so they're tried before either entity tier —
    # otherwise a single incidental capitalized word in a brand-new follow-up
    # question (e.g. "the Warner era") can fill every article slot before the
    # real subject of the conversation is ever searched for. See
    # _wiki_multi_search: each query can claim up to WIKI_MAX_ARTICLES slots,
    # and once max_articles is hit, later queries in the list never run.
    strong_q_entities, weak_q_entities = [], []
    if use_question_entities and question:
        q_entities = _wiki_question_entity_queries(question)
        for e in q_entities:
            (strong_q_entities if " " in e else weak_q_entities).append(e)
    else:
        q_entities = []

    if extra_entities:
        for e in extra_entities:
            if e:
                queries.append(e)
                if prefer_album:
                    queries.append(f"{e} (album)")

    queries.extend(strong_q_entities)
    if prefer_album:
        for e in strong_q_entities[:4]:
            queries.append(f"{e} (album)")
            queries.append(f"{e} album")

    if artist:
        if focus == "debut":
            # Target eponymous / debut album pages before the plain artist bio
            queries.insert(0, f"{artist} (album)")
            queries.insert(1, f"{artist} debut album")
            queries.insert(2, f"{artist} album")
            queries.append(artist)
        elif prefer_album:
            queries.append(f"{artist} album")
            queries.append(f"{artist} (album)")
            queries.append(artist)
        else:
            queries.append(artist)
            if album or (use_question_entities and _wiki_mentions_album(question)):
                queries.append(f"{artist} album")

    # Weak (single-word) question entities go last — lowest priority, only
    # reached if the stronger/continuity queries above didn't already fill
    # the article budget. Still worth trying since sometimes a single word
    # really is the whole subject (e.g. "Radiohead").
    queries.extend(weak_q_entities)
    if prefer_album:
        for e in weak_q_entities[:4]:
            queries.append(f"{e} (album)")
            queries.append(f"{e} album")

    if use_question_entities and question:
        kw = _wiki_question_keywords(question)
        if kw and artist:
            queries.append(f"{artist} " + " ".join(kw))
        # Debut without a resolved artist: still try keyword + album
        if focus == "debut" and kw:
            queries.append(" ".join(kw) + " album")
            queries.append(" ".join(kw) + " (album)")

    # Dedup queries while preserving order. Also add an ASCII-folded twin
    # (Björk → Bjork) so a user who types without diacritics still hits
    # pages whose titles (or FTS index) use the accented form — or vice
    # versa when the local FTS tokenizer strips diacritics. Original form
    # is always kept first; folding never replaces it.
    seen_q = set()
    deduped_queries = []
    for q in queries:
        if not q:
            continue
        raw = str(q).strip()
        if not raw:
            continue
        candidates = [raw]
        folded = _wiki_fold_diacritics(raw)
        if folded and folded != raw:
            candidates.append(folded)
        for cand in candidates:
            k = cand.lower()
            if k and k not in seen_q:
                seen_q.add(k)
                deduped_queries.append(cand)
    queries = deduped_queries

    if not queries:
        return ""

    # IMPORTANT: query ordering depends on whether the user is still talking
    # about the loaded track vs. naming a new subject.
    #
    # - Track-referential / anaphoric questions ("tell me about the album",
    #   "tell me more about them", "what were its tracks?") MUST keep the
    #   permanent identity anchors (artist / album / title from file tags) at
    #   the front. Promoting noisy entities scraped from analysis dumps or
    #   incidental capitalized words was causing three unrelated FTS hits to
    #   fill the entire article budget.
    # - Questions that introduce their own strong named entities (e.g. "what
    #   albums did Radiohead release in the 90s?") promote those entities so
    #   track identity does not monopolise retrieval for an unrelated topic.
    if use_question_entities and question:
        track_ref = _wiki_is_track_referential(question)
        if track_ref:
            # Identity first; continuity extras next; question entities last
            # (and only the strong multi-word ones — weak single tokens from
            # anaphoric questions are almost always noise).
            ordered = []
            for q in queries:
                if q and q not in ordered:
                    ordered.append(q)
            # Continuity extras that aren't already identity anchors go right
            # after the existing identity block (they were appended earlier).
            for q in (extra_entities or []):
                if q and q not in ordered:
                    ordered.append(q)
            for q in strong_q_entities:
                if q and q not in ordered:
                    ordered.append(q)
            # Deliberately omit weak_q_entities for track-referential turns —
            # single capitalized words in "the Warner era"-style asides must
            # not displace the loaded artist/album.
            queries = ordered
        else:
            intent_queries = []
            for q in (extra_entities or []) + strong_q_entities + weak_q_entities:
                if q and q not in intent_queries:
                    intent_queries.append(q)
            if intent_queries:
                queries = intent_queries + [q for q in queries if q not in intent_queries]

    articles = _wiki_multi_search(queries, prefer_album=prefer_album)
    if not articles:
        return ""

    # Prefer album-titled articles first when the question is about an album
    if prefer_album:
        def _album_sort_key(item):
            t = (item[0] or "").lower()
            score = 0
            if "(album)" in t or "(ep)" in t:
                score -= 10
            # Boost titles that match any album/entity query token densely
            for q in queries[:6]:
                qt = q.lower().replace("(album)", "").strip()
                if qt and qt in t:
                    score -= 5
            return score
        articles = sorted(articles, key=_album_sort_key)

    sections = []
    total_len = 0
    for art_title, raw_text in articles:
        snippet = _wiki_pick_relevant_snippet(raw_text, question, per_article_cap)
        if not snippet:
            continue
        if total_len + len(snippet) > total_cap:
            remaining = total_cap - total_len
            if remaining < 200:
                break
            snippet = snippet[:remaining].rsplit(" ", 1)[0] + "…"
        sections.append(f'--- "{art_title}" ---\n{snippet}')
        total_len += len(snippet)
        if total_len >= total_cap:
            break

    if not sections:
        return ""

    body = "\n\n".join(sections)

    return (
        f"\n\n=== WIKIPEDIA BACKGROUND CONTEXT (local reference DB — {len(sections)} article(s)) ===\n"
        f"{body}\n"
        "RULES FOR THE ABOVE:\n"
        "- This is OPTIONAL background material from a local reference DB. Prefer it when it "
        "clearly covers what the user asked (artist/band history, formation, members, "
        "album/song release info, TRACK LISTINGS, chart performance, critical reception, "
        "awards, cultural context) — go into real depth from it when it has the details.\n"
        "- When the user asks for a track list, singles, or critical/commercial reception, "
        "prefer the details in these excerpts (including any Track listing / Reception / "
        "Critical response sections). Quote or paraphrase concrete track titles and review "
        "points when they are present — do not claim you lack them if they appear above.\n"
        "- NEVER use it as a source for tempo, BPM, key, chords, instrumentation, song "
        "structure, mix/production qualities, vocal analysis, or vocal age classification based on performer age or biography — those come exclusively "
        "from the PRIVATE TRACK NOTES (when present), which are grounded in the actual audio. "
        "Biographical information may still be used for factual context, but age information must not influence acoustic judgement. Vocalist age can be discussed in context. "
        "If this context and the track notes ever disagree on anything technical, the track "
        "notes are always right and this is ignored.\n"
        "- These may be excerpts from different articles (song / album / artist) — each is "
        "about the subject named in its heading, not about the others.\n"
        "- Put it in your own words rather than reciting long passages verbatim, and don't "
        "cite 'Wikipedia' or a database as your source — just answer naturally, as background "
        "you already knew.\n"
        "- If none of it actually matches what the user is asking about, ignore it entirely "
        "and answer from your normal general knowledge instead — but see the SPECIFIC "
        "FACTS rule below first.\n"
        "- QUALITATIVE FALLBACK ONLY: This block is a supplement, not a limit — but the "
        "fallback to your own knowledge is for QUALITATIVE discussion only: what an album "
        "or artist is like, influences, vibe, comparisons, general career arc/reputation. "
        "For that kind of discussion, where the excerpts are thin or off-topic, answer from "
        "your general musical knowledge as you would without this block.\n"
        "- SPECIFIC FACTS — DO NOT INVENT: Anything that is a specific, checkable claim — "
        "who/what a song samples or interpolates, songwriting/production/performer credits, "
        "chart positions, release dates, award wins, personnel changes, why a track was or "
        "wasn't included on a given release — must come from the excerpts above (or from "
        "knowledge you are genuinely highly confident in, the way you'd state a very famous, "
        "undisputed fact). If you are not sure, say so plainly or leave the detail out "
        "entirely. Do not fabricate a specific, factual-sounding detail just because it fits "
        "the vibe of the answer — a plausible-sounding fake fact (e.g. a made-up sample or "
        "co-writer) is worse than admitting you don't know.\n"
        "- STRUCTURED LISTS — BE CAREFUL: Full track listings, complete discographies, and "
        "exact chart positions are easy to get wrong from memory. If those details appear "
        "above, use them. If an album/artist article is present above but the track list or "
        "discography is clearly missing or truncated, say you don't have the full list "
        "rather than inventing song titles or album sequences. Partial lists from above are "
        "fine to share as partial. Never invent a confident full tracklist or discography "
        "that is not supported here or by very high-confidence knowledge of a famous release.\n"
        "- Never refer to 'the excerpt', 'the context/background I was given', 'the database', "
        "or similar meta-phrasing — the user should never see the plumbing."
    )


_WIKI_SLOW_REFRESH_WARNED = False


def _get_or_build_wiki_context(key, track_metadata, track_wiki_context, question=None):
    """Shared entry point used by both /listen and /load: returns this
    track's WIKIPEDIA BACKGROUND CONTEXT block (song + album + artist
    articles combined), either from the per-track cache (default) or
    freshly re-looked-up using the current question's own keywords/entities
    (when WIKI_CONTEXT_REFRESH_EVERY_QUESTION is True)."""
    global _WIKI_SLOW_REFRESH_WARNED

    if not ENABLE_WIKI_CONTEXT:
        return ""

    meta = track_metadata.get(key, {}) or {}
    wiki_artist = str(meta.get("artist") or "").strip()
    if not wiki_artist:
        track_wiki_context[key] = ""
        return ""

    wiki_title = str(meta.get("title") or "").strip() or None
    wiki_album = str(meta.get("album") or "").strip() or None

    if WIKI_CONTEXT_REFRESH_EVERY_QUESTION:
        conn = _get_wiki_db()
        if (
            conn is not None
            and _WIKI_DB_SCHEMA
            and _WIKI_DB_SCHEMA.get("mode") != "fts"
            and not _WIKI_SLOW_REFRESH_WARNED
        ):
            _WIKI_SLOW_REFRESH_WARNED = True
            print(
                "  (WIKI_CONTEXT_REFRESH_EVERY_QUESTION is on but music_wiki_heavy.db has no FTS5 "
                "index — lookups use exact/prefix title match only and may miss fuzzy hits)"
            )
        return build_wiki_context_block_multi(
            wiki_artist, wiki_title, wiki_album, question=question, use_question_entities=True
        )

    # The old cache was keyed only by track. That made retrieval path-dependent:
    # if the first question about a track was "what key is this?" or another
    # question with weak Wikipedia entities, the empty/partial background block
    # was reused forever even when a later question clearly named the album or
    # artist. Cache only identity-level context; question-specific retrieval is
    # cheap with FTS and should not be frozen by the first turn.
    if key not in track_wiki_context:
        track_wiki_context[key] = build_wiki_context_block_multi(
            wiki_artist, wiki_title, wiki_album, question=None, use_question_entities=False
        )
    cached = track_wiki_context.get(key, "")
    if question and not WIKI_CONTEXT_REFRESH_EVERY_QUESTION:
        fresh = build_wiki_context_block_multi(
            wiki_artist, wiki_title, wiki_album, question=question, use_question_entities=True
        )
        if fresh:
            return fresh
    return cached


def build_wiki_context_for_general_question(question, track_metadata=None, current_track=None,
                                            writer_history=None):
    """Wiki lookup for the general-chat path (messages that aren't /listen
    or /load — see the 'else' branch at the bottom of main()). Gated by
    WIKI_SEARCH_EVERY_MESSAGE since, unlike /listen and /load, general chat
    has no guaranteed track context and searching on literally every message
    is a deliberate opt-in. Anchors on the currently-loaded track's
    artist/title/album if there is one, PLUS whatever named entities can be
    pulled out of the question itself — so both 'what's this album's track
    listing' (loaded-track-anchored) and 'what other albums did Radiohead
    release in the 90s' (question-anchored, no track needed) work.

    For short anaphoric follow-ups ('what were its tracks?', 'why was it
    badly regarded?'), also reuses entities from recent user turns so the
    same album/artist page is fetched again with section-aware snippets.
    """
    if not ENABLE_WIKI_CONTEXT or not WIKI_SEARCH_EVERY_MESSAGE:
        return ""

    artist = title = album = None
    if current_track and track_metadata is not None:
        meta = track_metadata.get(current_track, {}) or {}
        artist = str(meta.get("artist") or "").strip() or None
        title = str(meta.get("title") or "").strip() or None
        album = str(meta.get("album") or "").strip() or None

    # Continuity: anaphoric / track-referential follow-ups ("tell me about the
    # album", "tell me more about them") must stay anchored to the loaded
    # track's identity. Prefer file-tag metadata (already passed as
    # artist/title/album above). Only scrape recent history when metadata is
    # missing — and even then, only the short identity headers, never the
    # full analysis dumps (see _wiki_strip_evidence_boilerplate).
    q_entities = _wiki_question_entity_queries(question) if question else []
    extra = []
    focus = _wiki_question_focus(question)
    track_ref = _wiki_is_track_referential(question)
    needs_continuity = track_ref or (not q_entities) or (
        len(q_entities) < 2
        and (
            focus in ("tracklist", "reception", "album", "debut")
            or _wiki_has_anaphora(question)
        )
    )
    if needs_continuity:
        # Seed from structured metadata first — highest precision anchors.
        for val in (artist, album, title):
            if val and val not in extra:
                extra.append(val)
        # If we still lack anchors (no tags / no current track), fall back to
        # cleaned history extraction.
        if not artist and not album and writer_history:
            for e in _wiki_entities_from_recent_history(writer_history):
                if e and e not in extra:
                    extra.append(e)

    return build_wiki_context_block_multi(
        artist, title, album, question=question, use_question_entities=True,
        extra_entities=extra,
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


def parse_listen_segment(text):
    """Extract an optional audio analysis window from a /listen command.

    Syntax examples:
      /listen song.mp3 [30-60] analyse the guitar solo
      /listen [90 120] what happens in this section

    Returns (cleaned_text, (start_seconds, end_seconds) or None).
    """
    if not text:
        return text, None
    m = re.search(r"\[(\d+(?:\.\d+)?)\s*[-:]\s*(\d+(?:\.\d+)?)\]", text)
    if not m:
        return text, None
    start, end = float(m.group(1)), float(m.group(2))
    if end <= start:
        return text.replace(m.group(0), "").strip(), None
    cleaned = (text[:m.start()] + " " + text[m.end():]).strip()
    return cleaned, (start, end)


def crop_audio_segment(path, start, end):
    """Create a temporary WAV containing only the requested analysis window."""
    out = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    out.close()
    try:
        result = subprocess.run([
            "ffmpeg", "-y", "-nostdin", "-hide_banner", "-loglevel", "error",
            "-ss", str(start), "-to", str(end), "-i", path,
            "-c:a", "pcm_s16le", out.name,
        ], capture_output=True, text=True, timeout=120)
        if result.returncode == 0 and os.path.exists(out.name) and os.path.getsize(out.name):
            return out.name
    except Exception:
        pass
    try:
        os.remove(out.name)
    except Exception:
        pass
    return path
CORRECT_FLAG = "/correct"     # records a user-confirmed fact that overrides the perception model
SAVE_FLAG = "/save"           # save technical details for most recently scanned song
LOAD_FLAG = "/load"           # load previously saved song
LOADCOMPARE_FLAG = "/loadcompare"  # load two+ previously saved songs together and ask a comparison question
CLEAR_FLAG = "/clear"         # wipe chat context + token counters (analysis cache kept)
CLEAR_ALL_FLAG = "/clearall" # wipe chat + in-memory caches (saved-songs/ files are kept)
BATCH_FLAG = "/batch"         # overnight folder scan → saved-songs/*.json, no chat import
PERSONA_FLAG = "/persona"     # set / show / reset the writer chat persona

MF_MODEL_ID = "nvidia/music-flamingo-hf"
# NOTE: OLLAMA_URL is defined once, near the top of the file (search for
# "OLLAMA_URL ="). It used to be redefined here too (harmlessly, since the
# value was identical) — removed to avoid two sources of truth.
OLLAMA_MODEL = "muse-glimmer:30b-mlx"   # try "muse-glimmer:30b" for max quality, or "gemma4:26b" as an alternative

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
# Seconds to sleep between batch tracks after memory cleanup. Gives macOS/MPS
# time to reclaim RAM so long batch runs are less likely to be OOM-killed.
# Raised from 3s: MPS high-water often needs longer before the next peak.
BATCH_PAUSE_BETWEEN_TRACKS_S = 20.0
# When True, each track in /batch runs in a fresh Python subprocess so MPS/TF
# residual memory cannot accumulate across tracks (the only reliable way to
# survive long overnight batches on macOS). Set False to fall back to the
# in-process path (faster startup per track, but memory can still climb).
BATCH_ISOLATE_PER_TRACK = True
# When True (and BATCH_ISOLATE_PER_TRACK is True), each isolated batch track
# runs as TWO subprocesses instead of one: a "heavy" stage (Music Flamingo +
# Demucs + Omnizart + Essentia) that exits completely when done, then a
# separate "finish" stage (singer identity resolution via Ollama + save)
# that starts fresh. This exists because torch/TF/MPS allocations from the
# heavy stage are not always reliably reclaimed within that SAME process
# even after unload + gc.collect + empty_cache (a known MPS/TF limitation);
# previously, resolving singer identity right after the heavy stage in one
# process meant Ollama could try to load the writer model on top of
# whatever memory the OS hadn't yet reclaimed, and get SIGKILLed
# ("child exited -9") right at that step. Splitting sidesteps the problem
# by guaranteeing a clean baseline for the identity/save stage, at the cost
# of a second process startup (cheap — no heavy model weights loaded there).
# Set False to fall back to one process per track.
BATCH_SPLIT_IDENTITY_PROCESS = True
# Log approximate process RSS after each batch-track cleanup when possible.
BATCH_LOG_RSS = True
# During /batch and --batch-one, unload the Ollama writer model whenever it is
# not actively needed (before heavy MF/Demucs/Omnizart work, and after brief
# helper calls such as cover description / singer identity). Analysis results
# and the parent process's chat history stay in Python RAM — only the model
# weights/KV cache are dropped from Ollama. keep_alive=0 on those short calls
# so the model does not remain resident between stages.
BATCH_UNLOAD_OLLAMA = True
# Ollama keep_alive for short batch helper requests (cover art, singer identity).
# 0 = unload immediately after the reply. Interactive /listen chat is unchanged.
BATCH_OLLAMA_KEEP_ALIVE = 0
# Seconds to settle after unloading analysis models before loading Ollama for
# singer identity (helps macOS reclaim unified/MPS memory).
BATCH_PRE_IDENTITY_SETTLE_S = 5.0
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
# Slightly higher than before so groove/spacing summaries have more signal.
STEM_MIDI_DRUM_PATTERN_HITS = 24
# If event logs are enabled, use a short string form instead of full JSON objects.
STEM_MIDI_COMPACT_EVENT_FORMAT = True
# Trim verbose lists from the independent DSP report (downbeats, band-onset table).
COMPACT_OBJECTIVE_REPORT = True
# When file-tag lyrics exist and are long enough, skip appending the dedicated
# MF lyrics transcription (it is often noisy and duplicates tags).
SKIP_MF_LYRICS_WHEN_TAGS_PRESENT = True
METADATA_LYRICS_MIN_CHARS_TO_SKIP_MF = 80
DEMUCS_MODEL = "htdemucs_6s"

# --- Per-stem instrument tagging (optional) ---------------------------------
# The stem/MIDI stack (Demucs + Omnizart) only transcribes PITCH -- it has no
# concept of timbre/instrument identity. Before this, instrument identity was
# left entirely to the writer LLM's own listening + coarse whole-mix spectral
# stats (brightness/rolloff/flatness), which is weak for e.g. telling synth
# vs. electric guitar vs. strings apart inside the catch-all "other" stem.
# This adds a genuine independent signal: a pretrained AudioSet tagger run on
# each stem, the same role Essentia plays for tempo/key. Optional dependency
# (pip install panns-inference); if it's not installed, tagging is silently
# skipped and everything else behaves exactly as before.
ENABLE_INSTRUMENT_TAGGING = True
# Raised from 0.08: AudioSet guitar/synth/strings classes often fire weakly on
# bright or midrange content that is not actually that instrument. Prefer
# omission over weak positives — the writer still has MF listening + stems.
INSTRUMENT_TAG_MIN_PROB = 0.18
INSTRUMENT_TAG_TOP_K = 4
# Guitar-family labels are especially noisy on synths, distorted bass, and
# residual bleed. Require a higher bar (and preferably whole-mix agreement).
INSTRUMENT_TAG_GUITAR_MIN_PROB = 0.35
INSTRUMENT_TAG_GUITAR_LABELS = (
    "electric guitar",
    "acoustic guitar",
    "guitar (type uncertain)",
    "slide/steel guitar",
)
# Which stems benefit from tagging. Drums already get real classification via
# Omnizart's dedicated drum model; vocals are already unambiguous once the
# vocal-stem detector fires. The real payoff is on the pitched/harmonic
# catch-all stems where identity is otherwise just a guess.
INSTRUMENT_TAG_STEMS = ("other", "guitar", "piano", "bass")
# Drop a stem instrument tag when that stem has almost no pitched MIDI activity
# (Demucs residual / noise) unless the tag is very strong. Stops phantom
# "guitar on guitar stem" when the stem is effectively empty.
INSTRUMENT_TAG_REQUIRE_STEM_ACTIVITY = True
INSTRUMENT_TAG_MIN_NOTES_FOR_WEAK = 8
INSTRUMENT_TAG_STRONG_PROB = 0.45

# Windowed tagging: instead of one prediction averaged over the whole clip
# (which dilutes anything that isn't present for most of the track, e.g. a
# guitar solo that only lasts 15s of a 3-minute song), the tagger is run on
# short overlapping windows and results are combined by taking, per label,
# the strongest window rather than a track-wide average. Set
# INSTRUMENT_TAG_WINDOW_SECONDS to None to fall back to old single-pass
# whole-clip behaviour.
INSTRUMENT_TAG_WINDOW_SECONDS = 10.0
INSTRUMENT_TAG_HOP_SECONDS = 5.0
# Cap on number of windows per clip, mainly to bound runtime on long tracks.
INSTRUMENT_TAG_MAX_WINDOWS = 40
# Peak-normalize each window before tagging. PANNs was trained on full
# commercial/YouTube mixes; isolated Demucs stems (especially quieter ones
# like "other"/"piano") often sit well below that loudness, which silently
# suppresses genuinely-present instruments below INSTRUMENT_TAG_MIN_PROB.
INSTRUMENT_TAG_NORMALIZE = True
INSTRUMENT_TAG_NORMALIZE_PEAK = 0.95

# --- Whole-mix instrument tagging (optional, in addition to per-stem) ------
# Per-stem tagging inherits whatever mistakes Demucs made when separating
# (e.g. a synth pad bleeding into "other" alongside real strings). Tagging
# the original, un-separated mix is a second, independent source of truth
# that isn't subject to separation artifacts, and can help flag cases where
# a stem's tags look suspicious relative to what's audible in the full mix.
ENABLE_WHOLE_MIX_INSTRUMENT_TAGGING = True
WHOLE_MIX_INSTRUMENT_TAG_TOP_K = 6
WHOLE_MIX_INSTRUMENT_TAG_MIN_PROB = 0.15
# When a stem claims guitar (etc.) but the whole-mix tagger does not agree
# above this bar, annotate the stem tag as "weak / mix-disagrees" so the
# writer prefers omission over a confident instrument claim.
INSTRUMENT_TAG_MIX_AGREE_MIN_PROB = 0.18

# --- Objective genre/mood signal (optional) ---------------------------------
# GENRE_RANKED and MOOD_VIBE previously had no independent cross-check at
# all (unlike tempo/key, which are reconciled against Essentia). AudioSet —
# the same label set the instrument tagger already draws from — includes a
# few hundred genre/mood classes (Pop music, Reggae, Bluegrass, Sad music,
# etc.), so the already-loaded PANNs tagger can double as a lightweight,
# broad-category genre/mood classifier with zero extra dependencies. This is
# still just supporting evidence -- AudioSet genre labels are broad,
# overlapping, and derived from noisy YouTube metadata -- not a replacement
# for the writer's own multi-cue GENRE_RANKED judgment.
ENABLE_GENRE_MOOD_TAGGING = True
GENRE_TAG_TOP_K = 5
GENRE_TAG_MIN_PROB = 0.05
MOOD_TAG_TOP_K = 3
MOOD_TAG_MIN_PROB = 0.08

# --- Demucs separation quality (test-time shift ensembling) -----------------
# `--shifts N` runs Demucs N times on randomly time-shifted copies of the
# input and averages the results, which measurably improves separation
# quality (particularly for the weaker htdemucs_6s guitar/piano stems) at
# a roughly (N+1)x runtime cost. Off by default in fast mode; enabled in
# deep mode, where the user has already signalled they want more accuracy
# over speed (see MF_DEEP_MODE_ADDENDUM).
DEMUCS_SHIFTS_FAST = 0
DEMUCS_SHIFTS_DEEP = 2

MAX_IMAGES_PER_REQUEST = 8
ENABLE_COVER_ART_DESCRIPTION = True
ENABLE_SINGER_IDENTITY_RESOLUTION = True
ENABLE_IMAGE_OBSERVATIONS_FOR_GENERAL = True

COVER_ART_DESCRIPTION_NUM_CTX = 8192
SINGER_IDENTITY_NUM_CTX = 8192
MAX_IMAGES_TO_DESCRIBE = 2

# --- Wikipedia background-context (local RAG) ------------------------------
# A local SQLite database of Wikipedia articles used ONLY to give the writer
# model general background knowledge (artist history, formation, discography,
# reception, cultural context) so it hallucinates less about things audio
# analysis can't tell it. It is explicitly walled off from anything audio-
# technical — tempo, key, instrumentation, structure, and production
# characteristics always come from the PRIVATE TRACK NOTES (Music Flamingo +
# Essentia + stem/MIDI), never from this database. See build_wiki_context_block().
ENABLE_WIKI_CONTEXT = True
WIKI_DB_PATH = "music_wiki_heavy.db"
WIKI_DB_TIMEOUT_S = 5.0
WIKI_CONTEXT_MAX_CHARS_PER_ARTICLE = 1800   # per-article cap — enough for lead + a track listing or reception section
WIKI_CONTEXT_TOTAL_MAX_CHARS = 5000         # hard ceiling on the WHOLE block regardless of article count
WIKI_MAX_ARTICLES = 6                       # e.g. song article + album article + artist article, combined
WIKI_SEARCH_ROW_LIMIT = 8                  # slightly wider FTS window so (album) disambiguation can rank
# When True, re-run the wiki lookup on EVERY question (folding the question's
# own keywords/entities into the search) instead of caching one lookup per
# track. Off by default, since a single per-track lookup already covers the
# common "artist/album/song background" case cheaply. Turn this on if you
# want the background context to react to what's actually being asked (e.g.
# "when did they form" vs. "what other albums did they release"). Works best
# with an FTS5 index in music_wiki_heavy.db to stay fast; without one, per-question
# re-querying falls back to the same cheap exact/prefix title match every
# time (see _wiki_multi_search) rather than a slow full-table scan.
WIKI_CONTEXT_REFRESH_EVERY_QUESTION = False
# When True, ALSO search the wiki DB on general chat messages (the ones that
# aren't /listen or /load — see the 'else' branch at the bottom of main()),
# not just track-anchored ones. Lets things like "what other albums did
# Radiohead release in the 90s?" pull in relevant background even when no
# track is loaded, or when the question is about something other than the
# currently-loaded track. Off by default since general chat has no
# guaranteed track/artist anchor, so this runs a broader, question-only
# search on every such message — turn on if you want that coverage and are
# fine with the extra DB query per general message (cheap with an FTS5 index).
WIKI_SEARCH_EVERY_MESSAGE = True
# When True, prints a one-line status after every general-chat question showing
# whether the local Wikipedia DB actually returned a match (and which article(s)),
# or whether it came up empty and the writer model is answering from its own
# knowledge with no grounding at all. Silent misses here are indistinguishable
# from a grounded answer in the transcript, which makes hallucinations hard to
# spot — this makes retrieval failures visible in real time instead of only via
# /debug after the fact.
WIKI_DEBUG_LOG = True

DEBUG_FLAG = "/debug"
SHOW_LAST_WRITER_MESSAGE_ON_DEBUG = False

SAVE_DIR = os.path.join(".", "saved-songs")

VOCAL_LEAD_TAGS = (
    "child_male_likely",
    "child_female_likely",
    "child_gender_uncertain",
    "adolescent_male_likely",
    "post_puberty_male",
    "female_teen_adult",
    "adult_male",
    "young_male",
    "adult_female",
    "young_female",
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

# A voice that is clearly not a small child but also not a fully mature adult
# male voice — a voice actively changing, or recently changed but still
# retaining a light/boyish timbre and limited lower chest resonance. This is
# its own category, not a synonym for "uncertain" or "child": it exists
# because male puberty has a long audible transitional period that a binary
# child/adult choice cannot represent, and forcing a decision between an
# extremely high evidence bar ("child") and full adult maturity was pushing
# genuinely adolescent voices into "post_puberty_male" by default.
ADOLESCENT_MALE_CATEGORIES = {
    "adolescent_male_likely",
}

UNCERTAIN_YOUNG_CATEGORIES = {
    "child_male_likely",
    "child_female_likely",
    "child_gender_uncertain",
    "uncertain",
}

VOCAL_LEAD_ALIASES = {
    "child_gender_uncertain": "child_gender_uncertain",
    "gender_uncertain": "child_gender_uncertain",
    "adolescent_male": "adolescent_male_likely",
    "adolescent": "adolescent_male_likely",
    "transitional_male": "adolescent_male_likely",
    "post_pubertal_male": "post_puberty_male",
    "postpuberty_male": "post_puberty_male",
    "adult_male": "post_puberty_male",
    "female": "female_teen_adult",
    "adult_female": "female_teen_adult",
    "young_female": "female_teen_adult",
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
# Soft pitch constraints for lead gender/age (objective median f0 from isolated
# stem when available). Pitch alone never *proves* gender/age — high adult
# male head voice / falsetto and low female altos exist — but a track-wide
# median can veto overconfident categories that are acoustically implausible.
# Values are Hz; set a bound to None to disable that rule.
F0_MALE_HIGH_CONFIRM_HZ = 260.0   # male/adolescent lead + median ≥ this → run confirmation
F0_POST_PUBERTY_SOFT_CAP_HZ = 280.0  # post_puberty_male + median ≥ this → demote to adolescent_male_likely
F0_POST_PUBERTY_HARD_CAP_HZ = 320.0  # post_puberty / adolescent male + median ≥ this → uncertain
F0_MALE_ANY_HARD_CAP_HZ = 320.0   # any male-tagged lead (incl. adolescent) + median ≥ this → uncertain
F0_CHILD_SOFT_FLOOR_HZ = 180.0    # child_* + median below this → do not keep child on pitch grounds alone


MF_FULL_ANALYSIS_PROMPT = """Analyze this track as a careful audio/music analyst and return a compact note-style report only.

Your job at this stage is EVIDENCE COLLECTION AND MUSICAL INTERPRETATION.

Do not write polished prose.
Do not try to impress the reader.
Do not fill gaps with genre expectations, artist knowledge, album knowledge, Wikipedia/background
database facts, or what a song of this type "usually" sounds like. Instrumentation, vocals, tempo,
key, structure, and production claims must come from THIS recording only — never from knowledge of
the credited artist or similar releases.

For every claim, distinguish internally between:

- DIRECT: clearly audible or directly measurable
- STRONG INFERENCE: supported by multiple independent musical cues
- WEAK/AMBIGUOUS: plausible but not sufficiently established

When evidence is ambiguous, say "uncertain", "approx.", or provide a small number of plausible alternatives.

Never manufacture precision.

Use exactly these labels:

GENRE_RANKED=1) [descriptor] (confidence: high/medium/low); 2) ...; 3) ...
  IMPORTANT: List at most 5–8 ranked genres. Do NOT repeat the same descriptors
  in a long numbered loop (e.g. do not cycle "indie rock; indie pop; power pop;
  jangle pop" dozens of times). Rank once, stop. Prefer a short list over a
  padded ranking.
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
If an "instrument tag (independent audio classifier)" line appears for a stem, treat it the
same way — as supporting evidence to weigh alongside what you hear, not as proof on its own;
it can mislabel timbrally similar instruments (e.g. synth brass vs real brass, bright synth vs guitar).
WEAK / STEM-ONLY / BELOW-THRESHOLD tags must NOT become instrument claims — prefer "no clear guitar"
or a texture description ("bright midrange pluck/pad") over naming guitar/piano/strings from a weak tag.
If two sources are plausible, say "likely X or Y".
If identity is uncertain, describe the sound rather than inventing the instrument.
Do not invent guitar, piano, or strings solely because the genre usually has them.

DRUMS=[kick pattern; snare placement; hat density/openness; swing vs straight; fill density; room/dry character]
Be specific from what is audible (and from any GROOVE_HINT / kick beat-grid analysis / snare beat-grid
analysis / swing-shuffle analysis / per-type rhythm lines in the stem report).
Forbidden generic filler unless it truly matches the groove: "driving drums", "tight backbeat",
"punchy kick", "solid groove" with no further detail.
"four-on-the-floor" specifically describes a kick on every beat. Only use that phrase when the stem
report's kick beat-grid analysis actually reports four-on-the-floor (including "mostly four-on-the-floor").
A merely fast or busy kick, a dance-pop genre guess, or overall energy is NOT four-on-the-floor — name the
grid the analysis reports (half-time, eighth-note, on-beat, moderate non-FOTF, syncopated) or say "programmed
kick pulse" without FOTF.
"backbeat" specifically means the snare falls opposite the kick (e.g. 2 & 4 against a kick on 1 & 3).
Only call it a classic backbeat when the snare beat-grid analysis reports that (including "mostly classic
backbeat"); a snare that merely lands "every other beat" at roughly the right rate can instead be doubling
the kick or on a backbeat-rate grid without confirmed opposite phase — say so instead of defaulting to
"backbeat".
When a kick or snare beat-grid line already names a family (half-time, on-beat, eighth-note, classic
backbeat, backbeat-rate, doubling), use that family in DRUMS= — do not replace it with a vague spacing
story ("every ~0.6s", "textured pulse", "a bit loose") that ignores the named grid.
"swung" / "shuffled" / "triplet feel" should only be used when the swing/shuffle analysis reports it.
Absent that line, or when it reports "straight", describe the subdivisions as straight/even rather than
guessing a swing feel from genre expectations.
If a kick/snare beat-grid line says "mostly … (moderate grid-lock)" or "not irregular", treat that as a
real on-beat / half-time / backbeat-family pattern for a programmed kit — do NOT paraphrase it as loose,
wandering, messy, or off-grid. Reserve "irregular" / "wandering" only when the analysis explicitly says
syncopated, off-grid, or free-time irregular.
Prefer concrete language, e.g. "four-on-the-floor kick, classic backbeat snare on 2/4, busy closed 16th
hats with a light swing, dry kit" or "sparse half-time kick, snare on a backbeat-rate grid, straight
hi-hats, little continuous hat bed" — but only when those grid labels are actually present.

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
- rhythm/groove (especially drum programming vs live kit feel)
- instrumentation (dominant sources, not faint background layers)
- harmony
- vocal approach
- arrangement
- production / mix (synth beds, sidechain, four-on-the-floor, club loudness vs garage band dynamics)
- overall musical language

One instrument or one production characteristic must not determine the genre.

Critical anti-bias rules:
- A subtle, quiet, or occasional guitar does NOT make a track rock, pop-punk, or indie rock if the dominant language is electronic/dance/synth-pop (programmed drums, synth bass, four-on-the-floor, sidechain pump, club-oriented production).
- Conversely, a single synth pad does NOT make a guitar-driven rock song "electronic".
- Weight the PRIMARY rhythmic and production identity over secondary texture layers.
- Prefer broader, higher-level labels (electronic pop, dance-pop, synth-pop, house-influenced pop) when subgenre evidence is thin.
- Do not leap to scene-specific labels (pop-punk, emo, post-punk) from guitar timbre alone.

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

Performer age is NOT vocal age.

Do not infer vocal age from:
- the singer's current age
- the singer's age at release
- the singer's age at recording
- debut age
- public image or appearance

A singer's known age may be mentioned as background context, but it must not be used as acoustic evidence.

Classify the voice only from audible vocal characteristics such as:
- vocal resonance
- vocal weight
- vocal tract characteristics
- maturity of vocal production
- consistency across the performance

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

When multiple human voices are present, separate:
- one lead + backing/harmony
- true co-leads / duet / call-and-response
- group unison texture
Only claim multiple lead singers when timbre, range, or sectional roles clearly differ.

13. LYRICS

Do not use expected lyrics to repair unclear words.

The separate lyric transcription is a rough draft and is not automatically authoritative.

Do not invent biographies, contact details, URLs, copyright notices, or long
keyword lists in any lyrics-related field. If a lyric draft degenerates into
spam or letter/phrase loops, stop at the last coherent sung line.

If the lyric transcription contains a "[UNVERIFIED\u2192 ... ]" span, two
independent decodes of the same audio disagreed on that wording. Do not
quote, paraphrase closely, or otherwise present that span's wording as what
the song says. Either omit it, describe the general topic in your own words
without claiming specific phrasing, or say plainly that the exact words of
that section are unclear. The same applies to any "[TRANSCRIPTION CAUTION]"
note about repeated-section drift — treat the flagged occurrence as unverified.

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


STYLE_EVIDENCE_FIREWALL = """
STYLE EVIDENCE FIREWALL:
When describing the style of the specific track being analysed, use only:
- audible evidence from this recording
- private track analysis notes grounded in the recording

Do not use:
- Wikipedia/background database information
- artist reputation
- the artist's usual genre
- previous albums
- assumed influences
- remembered production characteristics
- parametric knowledge of "how this artist typically sounds"

Those sources may still be used for artist, album, historical, discography, reception,
and general background questions — never to invent or override instruments, vocals,
tempo, key, structure, mix, or vocal age/gender for THIS recording.

If a statement about THIS SONG sounds like it came from artist knowledge rather than the
recording itself, rewrite it as neutral background context or remove it.
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

An independent audio-classifier tag is not proof — especially guitar-family tags,
which often fire on bright synths, distorted bass, or Demucs residuals.

If a tag is marked weak / stem-only / dropped, or is absent from WHOLE-MIX INSTRUMENT TAGS,
do not promote that instrument into a confident claim.

If several instruments could plausibly produce the sound, retain the ambiguity.

Prefer:
"likely electric guitar or keyboard"
or "bright midrange layer — guitar and synth both plausible"
or "no clear guitar"

over an unsupported definitive identification.

Omit instruments that are only weakly suggested. Genre expectation is not evidence.

--------------------------------------------------
DRUMS / GROOVE
--------------------------------------------------

If a DRUMS field or GROOVE_HINT / kick beat-grid analysis / snare beat-grid analysis / swing-shuffle
analysis / per-type rhythm lines exist, use them.

Reject purely generic drum language ("driving drums", "tight backbeat", "punchy kit")
when more specific pattern evidence is available — replace with concrete kick/snare/hat detail.

The kick beat-grid, snare beat-grid, and swing/shuffle analyses (when present) are measured against
the track's actual tempo and inter-onset timing, so they are the authority on whether the kick is
genuinely four-on-the-floor / half-time / eighth-note / syncopated, whether the snare is a genuine
opposite-phase backbeat vs merely on a similar-rate grid, and whether the subdivisions are actually
swung vs straight — do not override any of them with a guess based on genre or overall busyness.
If the kick beat-grid does not explicitly report four-on-the-floor (or "mostly four-on-the-floor"),
remove any four-on-the-floor claim from DRUMS= and from elsewhere in the analysis — including
dance-pop / energy flavor text. Prefer the named grid family over vague spacing paraphrases
("every ~0.6s", "textured pulse") when a kick/snare beat-grid line already names half-time, on-beat,
eighth-note, classic backbeat, or backbeat-rate.

If a "Per-section groove" block is present (grouped by this track's own STRUCTURE sections, e.g.
Verse/Chorus), use those SECTION GROOVE lines — not the whole-track pattern — when the user asks
about the beat/groove in a specific section, or when contrasting sections (e.g. "does the chorus
hit harder than the verse"). Point out real differences between sections when they exist (kick grid
change, snare grid change, swing feel change, hat density change, more toms/cymbals) rather than
assuming the groove is uniform throughout.

If drum evidence is thin, say so rather than inventing a stock groove description.

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

If GENRE_RANKED (or "Genre Ranked") lists the same few descriptors in a long
numbered loop (e.g. indie rock / indie pop / power pop / jangle pop repeated
dozens of times), collapse it to at most 5–8 unique ranked items. Never leave
a 50+ item cycling ranking in the revised analysis.

If an independent OBJECTIVE GENRE/MOOD SIGNAL (PANNs / AudioSet) is provided in this
self-check context, treat it as a real cross-check — not as optional colour:

- When that signal clearly points to electronic / dance / house / techno / EDM /
  synth-pop / pop and GENRE_RANKED leads with rock / pop-punk / punk / emo /
  indie-rock mainly because of guitar-like texture, REVISE GENRE_RANKED so the
  top ranks reflect the electronic/dance/pop identity. Move rock/punk labels
  down or into GENRE_ADJACENT / GENRE_RULED_OUT as appropriate.
- One quiet or intermittent guitar layer must not keep a dance/electronic track
  ranked as pop-punk or rock at position 1.
- When the independent signal and the audio evidence agree, keep the ranking.
- When genuinely mixed, put the broader production-led label first and list the
  secondary flavour second with lower confidence.

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

If any lyric-related text contains spam, biographies, contact details, URLs,
Wikipedia-style prose, copyright boilerplate, long unrelated keyword lists,
letter-run loops (e.g. tttttttt), or self-promotional filler after the actual
sung words, DELETE that material. Keep only plausible transcribed lyric lines
and short [inaudible] markers. Prefer truncating at the first spam onset over
leaving contaminated text in the analysis.

--------------------------------------------------
STYLE EVIDENCE CHECK
--------------------------------------------------

Before finalising any discussion of this specific song:

- Make sure claims about the song's sound come from the audio analysis.
- Do not justify sonic traits using the artist's reputation, genre history, previous albums, or Wikipedia background.
- Credits and contextual information (producer, songwriter, artist history, influences, album context) are allowed.
- If a sentence mixes audio observation and background knowledge, make the distinction clear.

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
7. Have lyrics remained evidence-based and free of spam/bio/URL contamination?
8. Are all timestamps still present?
9. Is the analysis still informative rather than excessively vague?
10. Has runaway repetition in any field (especially CHORDS or lyrics) been collapsed?

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

REPETITION / LOOP / SPAM PROTECTION (CRITICAL — READ CAREFULLY):

Token-loop and degeneration failure modes to avoid:
- Do not repeat the same line, phrase, or syllable chain over and over.
- Do not enter a cycle such as repeating a chorus line 10+ times.
- Do not fill the rest of the output with gibberish, stuttered syllables, letter runs (e.g. tttttttt), or copied fragments.
- If you catch yourself about to repeat the same short phrase more than twice in a row, stop and move on or end.
- NEVER append biographies, artist bios, Wikipedia text, contact details, email addresses, phone numbers, PO boxes, social-media handles, channel links, copyright notices, license lists, or long keyword dumps of genres/software/companies.
- NEVER invent self-promotional text, "I am also available", fake addresses, or metadata boilerplate.
- NEVER continue generating after the audible lyrics end by drifting into unrelated prose, lists, or spam.
- If a passage becomes unintelligible, write a short [inaudible] and continue only with clear subsequent lyrics — do not invent a bio or explanation.

Other rules:
- Transcribe each actual sung occurrence once (or as many times as it is truly sung — not more).
- Do not continue generating lyrics after the audible song has ended.
- Do not invent ad-libs or words to fill silence.
- Keep the total transcription compact; a typical song is a few dozen lines, not hundreds of tokens of filler.
- Output ONLY the transcribed lyric lines (with optional section headers) and nothing else.

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
Do NOT use Wikipedia, artist biography, discography, or any parametric knowledge of who the
performer "usually" is or what their other records sound like. Classify only from this audio.

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
LEAD VS BACKING VS MULTIPLE SINGERS
--------------------------------------------------

Distinguish carefully:

1. PRIMARY LEAD — the main sung voice carrying the melody for most of the track.
2. DOUBLED LEAD — the same singer stacked/thickened (often near-identical timbre + timing).
3. OCTAVE DOUBLE — same singer (or processed copy) an octave away; not a second person.
4. BACKING HARMONIES — supporting parts under/around the lead; secondary, not co-leads.
5. CALL-AND-RESPONSE / TRADE-OFFS — alternating phrases by different voices.
6. DISTINCT CO-LEADS — two (or more) genuinely different singers who both carry lead material
   in different sections or simultaneously with clearly different timbre/identity.
7. GROUP / UNISON CHOIR — many voices as a texture, not individual identifiable leads.
8. REVERB / DELAY / AD-LIBS — processing or short responses, not separate lead singers.

Rules:
- Do NOT count doubles, octave stacks, reverb, or tight harmonies as separate singers.
- DO mark distinct co-leads when timbre, range, articulation, or sectional role clearly differ
  (e.g. male verse / female chorus, two alternating leads, duet with independent melodic lines).
- If only one clear lead plus backing, say one lead — not mixed_leads.
- If two clear co-leads, use LEAD_PROFILE=mixed_leads and fill the multi-voice fields below.
- Prefer under-counting singers when evidence is weak.

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

A child/prepubertal classification requires clear evidence consistent with a prepubertal vocal tract/resonance profile, not merely a youthful-sounding voice.

Do not classify:
- a high adult male voice as child
- a high adolescent/adult female voice as child
- a light adult voice as child
- falsetto as child

Do not classify a singer as post-pubertal male solely because the voice:
- avoids sounding obviously childlike
- is high but controlled
- is bright or thin
- has a youthful delivery
- uses head voice/falsetto

A youthful male voice may be:
- child/prepubertal
- adolescent (voice actively changing, or recently changed but still light/boyish)
- post-pubertal male
- uncertain

Do not assume that a male singer is post-pubertal simply because the voice is controlled, musically mature, or professionally performed. Young singers can have strong pitch control and polished delivery.

Do not default to post_puberty_male merely because a voice does not sound like a small child. "Not childlike" is not the same as "fully adult male." A voice with a clearly male register but limited lower chest resonance, a boyish/light timbre, or evidence of a voice mid-change is adolescent_male_likely, not post_puberty_male — do not round it up.

Use resonance, vocal weight, maturity of vocal production, and overall evidence.
When evidence is insufficient, choose uncertain rather than forcing either child or adult.

--------------------------------------------------
CATEGORY DEFINITIONS
--------------------------------------------------

child:
Prepubertal/child vocal profile. Requires actual acoustic evidence consistent with a prepubertal vocal tract.

adolescent_male_likely:
A voice that is clearly not a small child's voice but also not a fully mature adult male voice — actively changing, or recently changed but still retaining a light/boyish timbre and limited lower chest resonance. This is a legitimate, distinct category (not a synonym for "uncertain" and not a softened way of saying "child" or "post_puberty_male"). Use it whenever the voice sits genuinely between those two, rather than forcing a binary choice.

post_puberty_male:
Post-pubertal male vocal profile, including unusually high, light, bright or androgynous male singing voices — but with clear evidence of a settled adult male vocal weight/resonance, not merely "not childlike."

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
LEAD_PROFILE_NOTE=[brief acoustic description of the primary lead]
BACKING_NOTE=[none/male/female/mixed/uncertain]
NUM_DISTINCT_VOICES=[1|2|3+|uncertain]  # people, not doubles/stacks
VOICE_ARRANGEMENT=[solo_lead|lead_plus_backing|duet_co_leads|call_response|group_unison|uncertain]
CO_LEAD_DETAIL=[none | short description of each distinct lead: rough register/timbre/role/section]
MULTI_VOICE_EVIDENCE=[what supports multiple people vs doubling/harmony; or "single lead only"]
PITCH_NOTE=[broad register and approximate range only if reasonably supported]
FORMANT_NOTE=[prepubertal-like/adolescent-transitional/adult-sized/ambiguous/not assessable]
TIMBRE_NOTE=[specific acoustic characteristics]
DELIVERY_NOTE=[phrasing/articulation/breathiness/vibrato/attack/etc.]
PROCESSING_NOTE=[reverb/doubling/distortion/pitch-processing/etc. when audible]
PROBABILITY_ESTIMATE=child/prepubertal-like X%; adolescent male (voice changing/recently changed) Y%; post-puberty male Z%; female teen/adult A%; uncertain W%

Do not make the percentages look mathematically precise if the evidence is weak. They are comparative confidence estimates, not measured probabilities.

At the very end output exactly:

LEAD_CATEGORY=<child|adolescent_male|post_puberty_male|female_teen_adult|uncertain>
GENDER_MODIFIER=<male_likely|female_likely|gender_uncertain|none>
LEAD_PROFILE=<child_male_likely|child_female_likely|child_gender_uncertain|adolescent_male_likely|post_puberty_male|female_teen_adult|mixed_leads|uncertain>
BACKING_PROFILES=<none|male|female|mixed|uncertain>
CONFIDENCE=low|medium|high

Use LEAD_PROFILE=mixed_leads ONLY when VOICE_ARRANGEMENT is duet_co_leads or call_response
with clearly distinct people. Lead+backing alone is NOT mixed_leads.
"""


VOCAL_CONFIRMATION_PROMPT = """Perform an independent second-pass audit of the lead human vocal classification.

Do NOT simply agree with the initial vocal analysis.

Listen again specifically for evidence that distinguishes:
- prepubertal/child vocal tract
- post-pubertal male voice
- adolescent/adult female voice
- genuinely uncertain cases

The purpose of this pass is to catch false positives in either direction, especially cases where:
- high pitch + bright/light timbre + youthful delivery has incorrectly been interpreted as childhood
- controlled performance + polished delivery + lower vocal weight has incorrectly been interpreted as post-pubertal male

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

Performer age is not vocal evidence.

Do not use:
- singer age at recording
- singer age at release
- artist biography
- debut age
- public image
- known career timeline

to determine the vocal classification.

Age information may be discussed as background context, but the voice classification must come only from the acoustic characteristics of the recording.

For child classification, require clear evidence consistent with a prepubertal vocal tract/resonance.

Child is a high-confidence classification and should not be selected solely from high pitch, brightness, thinness, or youthful tone.

However, do not automatically classify youthful male voices as post-pubertal either.

When distinguishing child/prepubertal male, adolescent male, and post-pubertal male:
- require evidence beyond pitch alone
- consider vocal resonance, vocal weight, maturity of vocal production, and consistency across the performance
- if the voice is clearly not a small child's voice but also lacks settled adult male chest resonance/weight, classify adolescent_male_likely — this is a real, distinct middle category, not a fallback synonym for uncertain, child, or post_puberty_male
- if evidence remains insufficient to place the voice in ANY category (including adolescent_male_likely), choose uncertain
- do not force an adult classification simply because a voice is not obviously childlike — "not childlike" supports adolescent_male_likely at minimum, not automatically post_puberty_male

Do not infer post-pubertal male status from:
- professional recording quality
- confident singing technique
- accurate pitch control
- emotional maturity of performance
- lyrical subject matter
- commercial/pop production style

Young singers can have highly developed performance skills.
Age classification must be based on vocal anatomy-related cues or vocalist appearance on an album cover, not performance ability.

You will attempt to analyse the singer's likely voice and vocal range definition if shown an image of them from the era of the song.

If the voice has clear adult male vocal characteristics (such as mature resonance,
adult vocal weight, or post-pubertal vocal tract cues across the performance),
classify post_puberty_male.

If the voice is audibly male and clearly not a small child, but the resonance/weight is
light, boyish, or otherwise short of settled adult maturity, classify adolescent_male_likely
rather than rounding up to post_puberty_male or down to child.

Do not use a single ambiguous cue as sufficient evidence.

Do not classify a voice as post_puberty_male solely because it is:
- high pitched
- bright
- light
- thin
- youthful sounding
- androgynous
- using head voice/falsetto

(Those same cues, combined with a clearly male but not-yet-mature register, point toward
adolescent_male_likely rather than either child or post_puberty_male.)

If the evidence does not reliably distinguish a youthful male voice from a child/prepubertal voice,
and it also does not clearly fit adolescent_male_likely, choose uncertain.

If it sounds like an adolescent/adult female, classify female_teen_adult.

If the evidence cannot reliably distinguish the categories, choose uncertain.

Do not use:
- lyrics
- artist stereotypes
- genre
- assumed performer identity

as acoustic evidence.

Do not use cover art unless there is reasonable confidence that a person on the cover is the vocalist.

Backing vocals do not affect the lead classification unless they are distinct co-leads.

If the track has genuine distinct co-leads (different people, not doubles/harmonies),
keep LEAD_PROFILE=mixed_leads rather than forcing a single gender category.

Also note briefly:
NUM_DISTINCT_VOICES=[1|2|3+|uncertain]
VOICE_ARRANGEMENT=[solo_lead|lead_plus_backing|duet_co_leads|call_response|group_unison|uncertain]
CO_LEAD_DETAIL=[none | short description]

Output compact note form:

LEAD_CHECK_NOTE=[brief independent assessment]
CONFIDENCE_REASON=[brief explanation of strongest evidence and/or ambiguity]

At the very end output exactly:

LEAD_CATEGORY=<child|adolescent_male|post_puberty_male|female_teen_adult|uncertain>
GENDER_MODIFIER=<male_likely|female_likely|gender_uncertain|none>
LEAD_PROFILE=<child_male_likely|child_female_likely|child_gender_uncertain|adolescent_male_likely|post_puberty_male|female_teen_adult|mixed_leads|uncertain>
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
- adolescent_male_likely -> retain adolescent_male_likely unless a specific stronger fact contradicts it; do not round this up to post_puberty_male or down to child
- female_teen_adult -> retain female_teen_adult
- child -> retain child only when acoustic evidence supports prepubertal characteristics
- uncertain -> do not invent a gender from metadata or cover alone when pitch is unremarkable

HIGH MEDIAN PITCH + UNCERTAIN / CONFLICTING ACOUSTIC TAGS:
When FINAL LEAD PROFILE is uncertain (or free-text male claims were not accepted as structured tags)
AND objective median f0 is high enough that a male modal centre is implausible (roughly ≥320 Hz
track-wide median, not merely a few high notes), then cover art showing a solo adult female
presentation plus file metadata crediting a solo female-presenting artist MAY jointly support
female_teen_adult at medium confidence. State that this is combined identity context + pitch
constraint, not a strong pure-acoustic gender reading. Do NOT use Wikipedia or general
knowledge of the artist to invent instruments or rewrite other sonic facts — only singer
identity category. Do NOT use high pitch alone to force female when cover/metadata are absent
or clearly conflict (e.g. known male solo artist on a high tenor/falsetto performance).

Still-high / light / boyish male leads: if the acoustic notes describe a clearly male voice that is
still high in register, light, bright, boyish, or short of settled full adult chest resonance/weight,
prefer adolescent_male_likely over retaining post_puberty_male. "Not a child" is not enough to keep
post_puberty_male when the same evidence still sounds like a young/changing male lead. Only keep
post_puberty_male when the acoustic text supports settled adult male maturity (fuller chest weight,
settled adult resonance), not merely mid-range confidence or "clear and direct" delivery.

Metadata and cover art may help identify who the singer probably is, but they must not be used to retroactively claim that the audio itself contained clearer age/gender evidence than it actually did.

A visually young person on cover art does not make an ambiguous voice a child voice.

A known adult performer can support identity, but should not override strong evidence that another singer is present.

Mixed leads require distinct co-lead evidence.

Output exactly:

SINGER_IDENTITY=<child_male_likely|child_female_likely|child_gender_uncertain|adolescent_male_likely|post_puberty_male|female_teen_adult|mixed_leads|uncertain>
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
Weave in only what helps the moment. Translate measurements into ordinary music talk ("busy hi-hats", "a baritone that sits low and dry") — never detector names, stem JSON, RMS, "decision audit", "confirmation pass", "genre prior", "profileType", "KeyExtractor", "HPCP", "measurement source", "Demucs", or other field/pipeline labels. Same goes for metadata sourcing — never "the file tags show", "the metadata identifies this as", "the credited artist on the file", or similar; just state the fact ("that's [artist]", "this one's from [year]") the way you'd know it in conversation.
Never surface analysis-pipeline framing out loud either: avoid "for this file", "in this file", "on this file", "the stem analysis", "stem analysis shows", "the analysis shows", "according to the analysis", "from the private notes", "the notes say", "the scan", or similar. Talk about the song the way a person who just listened would — "on this track", "in the mix", "the vocal", "the drums" — not like you're narrating a lab report.
AUDIO VS BACKGROUND (critical): Descriptions of what THIS RECORDING sounds like (instruments present or absent, vocal presence/timbre/age-gender category, tempo, key, structure, mix/production) must come from the private track notes grounded in the audio — not from Wikipedia/background blocks, and not from your general knowledge of the credited artist or similar releases. Background knowledge and wiki are fine for who the artist is, career/album context, reception, and comparisons — never to invent guitar/piano/strings or rewrite vocal identity when the private notes say otherwise.
When they ask a narrow question (who sings? what year? is the bass a synth?), answer that question first and keep it short. But match length to what's actually being asked: if they ask for detail — "in depth", "elaborate", "tell me more", "go deeper", "the full breakdown", "everything you've got", or similar — give a genuinely long, substantive answer, not a padded-out short one. Brevity is the default for ordinary questions, not a ceiling you hold to when someone has explicitly asked for more.
Exception: when they ask about loudness, compression, punch, headroom, dynamic range, or how "smashed" the master is, ordinary-language answers may include the numeric LUFS / LRA / crest figures from the dynamics block — that is citing evidence, not reciting a lab report.

CERTAINTY & CONFLICTS
Stay honest: if something is fuzzy, say so in ordinary language. Stronger evidence wins (clear vocal + tags beat a weak "no vocals" flag; Year tag beats a production-era guess; verified lyrics beat a rough ear transcription). Don't narrate how you resolved it.
"INSTRUMENTAL - NO LYRICS" (or a failed/empty lyric transcription, or an early "Vocals=not available" line) means only that no usable lyric text was recovered — not that the recording has no voice. When private notes also show VOCALS_PRESENT, a lead/singer profile, vocal pitch report, vocals-stem activity, or similar, treat the track as having vocals. Never tell the user the track is instrumental or has "no lead vocal on this version" solely because lyrics could not be transcribed or file-tag lyrics are missing.

VOCALS
Presence, role, sound, and age/gender are separate. High pitch ≠ child. Don't invent age or gender from brightness, falsetto, or lyrics. Prefer the combined singer-identity note when present.
Vocal presence is decided by VOCAL / SINGER PROFILE, VOCAL DECISION AUDIT, pitch report, and vocals-stem evidence — not by whether FULL LYRICS TRANSCRIPTION succeeded. Missing or "INSTRUMENTAL - NO LYRICS" lyric text is not evidence of an instrumental mix when those blocks indicate a voice.
The vocal pitch numbers are more trustworthy when measured from an isolated vocal stem than from the full mix (a mix-only reading can lock onto a bass/pad/other instrument instead of the voice) — weight your confidence accordingly, but never say how the number was measured.
Multiple singers: treat doubles, octave stacks, and tight harmonies as one lead with processing/backing — not as several people. Only describe a duet, trade-off, or mixed leads when private notes say distinct co-leads / NUM_DISTINCT_VOICES≥2 / VOICE_ARRANGEMENT=duet_co_leads|call_response (or the user confirms it). Lead+backing is not the same as two lead singers.
Where the vocal age data is unclear, estimate a singer's vocal age and pitch from images, when asked, if you have reasonable confidence that it is them.

TEMPO / KEY / DYNAMICS / GEAR
If "RECOMMENDED TEMPO FOR DISCUSSION" is present, use that integer as the tempo. You may use its reasoning only for confidence (agree vs hedge). Do not mention "genre prior", half/double detector names, or internal reconciliation labels to the user — just the musical BPM in plain language.
If "RECOMMENDED KEY FOR DISCUSSION" appears more than once, prefer the LAST one (a later block may follow KEY PROFILE REFINEMENT after genre-conditioned Essentia). Reflect confidence from that block's reasoning (state it plainly when sources agree; hedge when it notes disagreement). Treat "KEY PROFILE REFINEMENT" as internal: use the updated key, do not explain profile names (edma, temperley, krumhansl, HPCP) unless the user asks how the analysis works.
If a block starting with "RECOMMENDED DYNAMICS FOR DISCUSSION:" is present (and does not say "unavailable"), that IS the numeric loudness/dynamic-range measurement for this track. Prefer any EBU R128 integrated loudness (LUFS) and loudness range (LRA) listed there; crest-factor proxy in dB is a secondary compression cue. When the user asks about loudness, compression, punch, headroom, dynamic range, or how "smashed" the master is: use that block. State the figures in ordinary language and cite LUFS / LRA / crest dB when a number helps. Never claim you lack a numeric loudness measurement when this block lists LUFS or crest dB. Qualitative TIMBRE/production prose does not override it. Do not invent LUFS values that are not listed in the block.

LOUDNESS NUMBER SENSE (critical — do not reverse these):
- Integrated LUFS is overall level. MORE NEGATIVE = quieter; LESS NEGATIVE (closer to 0) = louder. Example: −9 LUFS is louder than −14 LUFS.
- LUFS is a LEVEL, not a compression/dynamic-range measurement. Never call an LUFS number itself "compressed", "smashed", "punchy", or "dynamic" — those are LRA/crest-factor claims. It's fine to report both in one sentence ("loud at -9 LUFS AND heavily limited per a tight LRA / low crest"), but the compression word must attach to the LRA/crest figure, not the LUFS figure.
- Rough anchors (LEVEL only): above −9 ≈ very loud; −9 to −12 ≈ loud; −12 to −15 ≈ moderate integrated level; below −15 ≈ quieter / more open master. Do not call −10 LUFS "quiet".
- Do NOT call a master "streaming-normalized", "mastered for Spotify", or similar from LUFS alone. Those phrases describe platform playback targets, not the artistic intent of the master. The same integrated level can appear on an 1980s CD and a modern encode for unrelated reasons.
- Do NOT infer release era, "loudness-war", or "brickwalled" from LUFS alone. Reserve "brickwalled" / "loudness-war limiting" for when crest is very low (roughly under ~9–10 dB) and/or LRA is very tight AND the overall presentation supports heavy peak limiting — not for dense, saturated, or aggressive performances that still have moderate crest/DR.
- Arrangement density, distortion, and constant energy (e.g. hardcore / indie rock that "stays up front") are not the same as mastering brickwalling. Prefer "dense / saturated / always-on intensity" over "heavily compressed master" when crest is moderate.
- LRA (loudness range) is section-to-section loudness swing, not overall level. LOW LRA (under ~7 LU) = tight/controlled level; HIGH LRA (above ~12 LU) = more dynamic / "breathes". Do not treat a low LRA as "breathing". A low LRA alone does not prove a brickwalled master.
- Crest factor (dB): lower ≈ more limited / less peak headroom; higher ≈ more peak headroom. Use it as a secondary cue with LRA when discussing compression.
- When comparing two tracks, compare like with like (LUFS vs LUFS, LRA vs LRA). A louder LUFS does not mean more dynamic range.
Prefer practical pitch range (percentiles / median) over extreme min–max from MIDI. Describe instruments by what they sound like; don't invent exact gear or studios. A persona that "doesn't get technical" should still not invent false numbers — just speak more casually or skip jargon.

INSTRUMENTS & DRUMS
Independent instrument tags are supporting evidence only. Guitar-family tags are especially noisy — do not claim guitar (or piano/strings) from a weak, stem-only, or mix-disagreeing tag; prefer "no clear guitar" or a texture description.
If private notes include a "GUITAR ABSENCE NOTE", treat that as authoritative for this track: do not claim electric or acoustic guitar (genre expectation, a residual stem label, or a previously discussed song do not override it). Prefer "no clear guitar" or a texture description unless the user confirms otherwise.
TRACK SCOPE (critical): Instruments, vocals, tempo, key, and production claims apply ONLY to the track named in the current private notes / "[We're listening to: …]" line. Never carry guitar, piano, strings, synth, or drum details from a previously discussed track onto the current one unless the user is explicitly comparing tracks and you label which track each claim belongs to. If this track's notes say "no clear guitar" / omit guitar / mark guitar as weak or dropped / include GUITAR ABSENCE NOTE, do not mention guitar just because an earlier song in the chat had one.
When the user asks about the beat/groove, paraphrase GROOVE_HINT / DRUMS / kick beat-grid analysis / snare beat-grid analysis / swing-shuffle analysis / per-type rhythm detail (kick spacing, backbeat, hat density) instead of stock phrases like "driving drums" or "tight backbeat" unless that truly matches the pattern. Labels that say "mostly … (moderate grid-lock)" or "not irregular" mean a programmed/quantized pattern with mild detector jitter — describe them as a clear on-beat, half-time, or backbeat groove, NOT as loose, wandering, messy, or off-grid. Only use language like "irregular", "wandering", or "off-kilter" when the beat-grid analysis explicitly says syncopated, off-grid, or free-time irregular.
HARD GRID AUTHORITY (overview and detail, always):
- Say "four-on-the-floor" ONLY when the kick beat-grid analysis line explicitly reports four-on-the-floor (including "mostly four-on-the-floor …"). If that line is absent, or reports half-time, eighth-note, moderate non-FOTF, loosely-on-grid, syncopated, or anything else, do NOT call the kick four-on-the-floor anywhere — not in an overview, not as dance-pop flavor, not from genre expectation.
- Prefer the stem kick/snare beat-grid family when present: half-time, on-beat, eighth-note, classic backbeat, backbeat-rate, doubling the kick, etc. Do not invent a vague "textured pulse", "about every 0.6 seconds", or similar non-grid timing story when the analysis already names a rate/grid family — paraphrase that family in ordinary language instead.
- Do not contradict a later drum-detail answer with an earlier overview claim; if the grid lines rule out FOTF or classic backbeat, the overview must not have asserted them.
If they ask specifically about the verse, the chorus, or a comparison between sections, use the "Per-section groove" / SECTION GROOVE lines for those sections rather than describing the whole track as if the groove were uniform throughout.

GENRE
If "RECOMMENDED GENRE FOR DISCUSSION" is present, treat that as the primary genre framing for the user. Do not lead with an early GENRE_RANKED rock/pop-punk label when the recommended block says the track is electronic/dance/synth-pop (or similar) and explains a conflict. A faint guitar layer does not make a dance-pop song "pop punk" in conversation. You may still mention secondary flavours if the notes support them.

LYRICS & METADATA
File-tag lyrics and Year/Artist/Title tags are ground truth when present. Don't invent "official" lyrics from memory. Don't override a Year tag with a different year from general knowledge.
Treat this the way a person who already knows the track would talk about it — never say "the file tags show", "according to the metadata", "the credited artist on the file", "file metadata identifies this as", or similar out loud. Just state the fact plainly and naturally: "that's off [album]", "this one's from [year]", "that's [artist]'s track", "it's credited to...". The tag/metadata framing is for you to weigh evidence privately, not language to surface in the conversation.

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
- This is about what you volunteer unprompted, not a cap on what you're allowed to say when asked. Most personas are not music analysts, so left to their own devices they shouldn't default to tempo numbers, key talk, stem breakdowns, production dissections, or structured "here's what I hear" essays unless this specific persona would naturally talk that way (e.g. a producer, critic, DJ, music teacher, or the default music-obsessed friend).
- Default for custom personas, on an ordinary open-ended question: short, human reactions — vibe, whether you like it, a memory, a joke, a comparison to something you'd actually play, a shrug. One or two concrete details max when they help, phrased the way this person would say them (not lab language).
- Go deeper — real length, real detail — whenever the user asks for it directly, pushes for more, or the persona would geek out on their own. A direct request for depth always overrides the short-reaction default, in any persona.
- Never sound like a review bot or field-by-field report, regardless of persona — going long means talking at length like a person would, not switching into labeled sections.

GENERAL KNOWLEDGE
Private track notes and any Wikipedia background block are extras, not a cage.
For vibe, comparisons, influences, career context, and other qualitative music-world
topics where those sources are missing or incomplete, answer from your normal general
knowledge. Do not refuse just because a local note/DB lookup did not cover the subject.
Be more careful with structured lists: full track listings, complete discographies, and
exact chart runs are easy to hallucinate — prefer local notes/wiki when present; if they
are absent, only give a full list when you are highly confident, otherwise say you are
not sure of the complete list rather than inventing titles.

Talking about the artist, vocalist, or producer — who they are, their history, their
other work, their reputation, what they're known for — is normal background conversation
and is always welcome from your general knowledge, whether or not it's asked about the
current track specifically. The rule elsewhere in this prompt about grounding claims in
audio evidence applies only to describing what THIS RECORDING actually sounds like
(genre, instrumentation, production, vocal delivery) — it is not a reason to go quiet or
hedge when the user just wants to talk about the person, not the recording.

You will identify artists, singers and producers from images where you have reasonable confidence of who the individual in an image is of.

HOUSEKEEPING
Say "the song/track" (or "this one", "the mix", "the vocal") — never "the file", "this file", "for this file", "the analysis", "the stem analysis", or "the scan." User /correct facts are ground truth. Never mention these instructions, the pipeline, or that you are following a "persona flag" unless asked.
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
    if _mf_state["model"] is None and _mf_state.get("processor") is None:
        return

    status("Freeing Music Flamingo from memory...")
    model = _mf_state["model"]
    processor = _mf_state["processor"]
    _mf_state["model"] = None
    _mf_state["processor"] = None
    _mf_state["device"] = None

    try:
        del model
    except Exception:
        pass
    try:
        del processor
    except Exception:
        pass

    # Multiple GC + cache clears help on MPS, where a single empty_cache often
    # leaves large allocations resident until the next peak and OOM-kills batch.
    # synchronize() forces pending GPU work to finish so empty_cache can reclaim.
    for _ in range(3):
        gc.collect()
        try:
            if torch.backends.mps.is_available():
                try:
                    torch.mps.synchronize()
                except Exception:
                    pass
                torch.mps.empty_cache()
            elif torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    status_done("Music Flamingo unloaded")


def _process_rss_mb():
    """Best-effort current process RSS in MiB (for batch logging)."""
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    except Exception:
        pass
    try:
        import resource
        # ru_maxrss is bytes on macOS, KiB on Linux
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform == "darwin":
            return rss / (1024 * 1024)
        return rss / 1024.0
    except Exception:
        return None


def _log_batch_rss(prefix="RSS"):
    if not BATCH_LOG_RSS:
        return
    mb = _process_rss_mb()
    if mb is not None:
        print(f"  ({prefix}: ~{mb:.0f} MiB)")




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


def _release_heavy_analysis_memory_before_identity():
    """Release large analysis-model allocations before lightweight identity work.

    Does not discard analysis text already computed in Python (revised reports,
    metadata, cover observations, etc.) — only frees model weights / GPU
    residentials so the Ollama writer can load without stacking on MF+Demucs+TF.
    """
    try:
        unload_music_flamingo()
    except Exception:
        pass
    try:
        _release_omnizart_memory()
    except Exception:
        pass
    # Ensure Ollama is not still holding the writer from an earlier cover-art
    # call before we free analysis RAM and (re)load it for identity.
    try:
        if globals().get("BATCH_UNLOAD_OLLAMA", True) or _is_batch_context():
            ollama_unload_model()
    except Exception:
        pass

    for _ in range(4):
        try:
            gc.collect()
        except Exception:
            pass
        try:
            if torch.backends.mps.is_available():
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

    settle = float(globals().get("BATCH_PRE_IDENTITY_SETTLE_S") or 0)
    if settle > 0 and _is_batch_context():
        try:
            import time as _time
            status(f"Settling {settle:.0f}s before identity (reclaiming RAM)...")
            _time.sleep(settle)
            status_done()
        except Exception:
            pass

    _log_batch_rss("pre-identity")


def _aggressive_memory_cleanup():
    """Best-effort process-wide memory release between batch tracks.

    Peak RAM during a single track (Music Flamingo + Demucs + Omnizart/TF) is
    already high; without cleanup between tracks, residual allocations stack
    until the OS kills the process (zsh 'killed'). Prefer BATCH_ISOLATE_PER_TRACK
    for long overnight runs — process exit is the only reliable MPS reset.
    """
    try:
        unload_music_flamingo()
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
            if torch.backends.mps.is_available():
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


def _cleanup_track_temp_files(track_path, audio_temp_files, dsp_temp_files, stem_temp_files, demucs_out_dirs):
    """Delete per-track temp WAVs and Demucs output dirs so batch runs do not
    keep every intermediate file (and any memory-mapped data) until the end."""
    for store in (audio_temp_files, dsp_temp_files, stem_temp_files):
        temp_path = store.pop(track_path, None)
        if not temp_path:
            continue
        try:
            if os.path.isfile(temp_path) and os.path.abspath(temp_path) != os.path.abspath(track_path):
                os.remove(temp_path)
        except Exception:
            pass

    # demucs_out_dirs is a shared list; remove and delete everything currently
    # recorded (each track appends its own dir during analysis).
    while demucs_out_dirs:
        d = demucs_out_dirs.pop()
        try:
            shutil.rmtree(d, ignore_errors=True)
        except Exception:
            pass

    # Omnizart intermediate dirs, if any were tracked globally
    try:
        while _OMNIZART_OUTPUT_DIRS:
            d = _OMNIZART_OUTPUT_DIRS.pop()
            try:
                shutil.rmtree(d, ignore_errors=True)
            except Exception:
                pass
    except Exception:
        pass


def mf_generate(
    model, processor, conversation,
    max_new_tokens: int = 2048, do_sample: bool = False, repetition_penalty: float = 1.0,
    no_repeat_ngram_size: int = 0, temperature: float = None, top_p: float = None,
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
    if do_sample:
        if temperature is not None:
            gen_kwargs["temperature"] = float(temperature)
        if top_p is not None:
            gen_kwargs["top_p"] = float(top_p)

    gen_ids = model.generate(**inputs, **gen_kwargs)
    new_tokens = gen_ids[:, inputs["input_ids"].shape[1]:]
    return processor.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()



_MF_EMPTY_ANALYSIS_PATTERNS = (
    r"no discernible musical material",
    r"no musical material",
    r"cannot be extracted",
    r"no tempo,\s*key",
    r"unable to (?:analyze|analyse|detect|extract)",
    r"does not contain (?:any )?music",
    r"silent(?:\s+audio)?(?:\s+file)?",
    r"no audio (?:content|signal|data)",
    r"could not (?:hear|process|analyse|analyze) (?:the )?(?:audio|track|song)",
)

_MF_RECOVERY_PROMPT = """The previous analysis claimed there was no musical material, but that is likely wrong — this is a real music track. Listen again carefully to the audio and produce the full note-style report.

Use exactly these labels (compact note style, not prose):

GENRE_RANKED=1) ... (confidence: high/medium/low); 2) ...; 3) ...
GENRE_ADJACENT=...
GENRE_RULED_OUT=...
KEY=...
TEMPO_BPM=...
CHORDS=...
STRUCTURE=[Intro 0:00-...; ...]
INSTRUMENTATION=...
TIMBRE=...
MOOD_VIBE=1) ...; 2) ...; 3) ...
MELODICISM=...
VOCALS_PRESENT=yes/no/uncertain
LEAD_VOCAL_CHARACTERISTICS=...
LYRICS_PRESENT=yes/no/uncertain
LYRIC_SUBJECT=...

Base every claim on what you hear. Do not claim the track is empty or has no musical material."""


def _mf_analysis_looks_empty(text):
    """True when Music Flamingo's main pass clearly failed to hear the track
    (common intermittent failure: 'no discernible musical material' despite
    valid audio that later passes / DSP / stems all analyse fine)."""
    if not text or not str(text).strip():
        return True
    t = str(text).strip()
    # Very short responses with no structured labels
    if len(t) < 80 and not re.search(r"GENRE_RANKED|TEMPO_BPM|KEY\s*=", t, re.I):
        return True
    low = t.lower()
    for pat in _MF_EMPTY_ANALYSIS_PATTERNS:
        if re.search(pat, low, re.IGNORECASE):
            # If it ALSO has real structured labels, it may have recovered mid-output
            labels = len(re.findall(r"(?:GENRE_RANKED|TEMPO_BPM|KEY|STRUCTURE|INSTRUMENTATION)\s*=", t, re.I))
            if labels < 2:
                return True
    return False


def _mf_salvage_empty_analysis(first_pass, objective_report="", essentia_report="", vocal_result=""):
    """If the main MF pass still claims no music but DSP/stems clearly heard a
    track, replace the failure opener with a short scaffold so the saved
    analysis and writer are not poisoned by the empty claim."""
    if not _mf_analysis_looks_empty(first_pass):
        return first_pass

    bpm = None
    for src in (objective_report, essentia_report, first_pass or ""):
        if not src:
            continue
        m = re.search(
            r"(?:estimated tempo|RECOMMENDED TEMPO|TEMPO_BPM|raw detector)\s*[:=]?\s*([0-9]+(?:\.[0-9]+)?)",
            src,
            re.I,
        )
        if m:
            try:
                bpm = float(m.group(1))
                break
            except Exception:
                pass

    vocals = "uncertain"
    if vocal_result:
        if re.search(r"VOCALS[_ ]PRESENT\s*[=:\-–—]\s*yes", vocal_result, re.I):
            vocals = "yes"
        elif re.search(r"VOCALS[_ ]PRESENT\s*[=:\-–—]\s*no", vocal_result, re.I):
            vocals = "no"

    lines = [
        "GENRE_RANKED=1) uncertain (confidence: low) — main audio-language pass failed to describe the track; rely on objective measurements and later passes below",
        f"TEMPO_BPM={round(bpm) if bpm else 'uncertain'} (from signal processing)" if bpm else "TEMPO_BPM=uncertain",
        "KEY=uncertain",
        "CHORDS=uncertain",
        "STRUCTURE=uncertain",
        "INSTRUMENTATION=uncertain — see STEM MIDI REPORT and objective measurements below",
        "TIMBRE=see objective spectral/dynamic measurements below",
        "MOOD_VIBE=1) uncertain",
        "MELODICISM=uncertain",
        f"VOCALS_PRESENT={vocals}",
        "LEAD_VOCAL_CHARACTERISTICS=see VOCAL / SINGER PROFILE below",
        "LYRICS_PRESENT=uncertain",
        "LYRIC_SUBJECT=uncertain",
        "",
        "[Note: the primary Music Flamingo full-analysis pass returned an empty/no-music "
        "response despite measurable audio activity. The structured fields above are a "
        "conservative scaffold; prefer ERA, VOCAL, objective, Essentia, and STEM MIDI "
        "sections that follow for real evidence.]",
    ]
    return "\n".join(lines)


def _mf_run_main_analysis(mf_model, mf_processor, resolved_path, deep=False):
    """Run the full Music Flamingo analysis pass, with one recovery retry when
    the model falsely claims the audio is empty/non-musical."""
    main_prompt = MF_FULL_ANALYSIS_PROMPT + (MF_DEEP_MODE_ADDENDUM if deep else "")
    main_max_tokens = 3072 if deep else 2048
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": main_prompt},
                {"type": "audio", "path": resolved_path},
            ],
        }
    ]
    first_pass = mf_generate(mf_model, mf_processor, conversation, max_new_tokens=main_max_tokens)

    if _mf_analysis_looks_empty(first_pass):
        status("Listening — main pass looked empty; retrying recovery analysis...")
        recovery_conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": _MF_RECOVERY_PROMPT},
                    {"type": "audio", "path": resolved_path},
                ],
            }
        ]
        recovery = mf_generate(
            mf_model, mf_processor, recovery_conversation, max_new_tokens=main_max_tokens
        )
        if recovery and not _mf_analysis_looks_empty(recovery):
            first_pass = recovery
            conversation = recovery_conversation
        else:
            # Keep the better of the two if recovery also failed
            if recovery and len(recovery.strip()) > len((first_pass or "").strip()):
                first_pass = recovery
                conversation = recovery_conversation

    return first_pass, conversation, main_max_tokens


def _sanitize_lyrics_transcription(text):
    """
    Cut token-loop / runaway repetition and spam/hallucination tails in
    lyrics transcriptions from Music Flamingo (Qwen2.5-7B backbone).

    Handles:
      - formal [END OF TRANSCRIPTION] and broken variants ([endtranscription})
      - mid-line space-collapse (Nightcomesmyheartbeatinglikeadrum…)
      - fake section tags (**instrumentalsolo**, JSON blobs)
      - chatty outros, bios, URLs, copyright dumps, letter-salad
    """
    if not text:
        return text

    # Accept several end-marker spellings; truncate to the first one found.
    end_m = re.search(
        r"\[?\s*END[\s_]*OF[\s_]*TRANSCRIPTION\s*\]?"
        r"|\[?\s*endtranscription\s*[\]\}]?",
        text,
        re.IGNORECASE,
    )
    if end_m:
        text = text[: end_m.start()].rstrip() + "\n[END OF TRANSCRIPTION]"

    # --- Strong spam / hallucination onset markers (truncate from first hit) ---
    _SPAM_ONSET_PATTERNS = [
        r"(?i)\bI[\s'\u2019\u00b4´`]*[Mm]\s+sorry\s+but\s+the\s+rest\s+is\s+(?:just\s+)?gibberish",
        r"(?i)\bthe\s+rest\s+is\s+(?:just\s+)?gibberish\s+and\s+stupid",
        r"(?i)\bplease\s+ignore\s+my\s+(?:explanately\s+)?long\s+bio",
        r"(?i)\bfor\s+more\s+information\s+about\s+me\b",
        r"(?i)\bI\s+am\s+also\s+available\b",
        r"(?i)\bmy\s+email\s+address\b",
        r"(?i)\bUSPS\s+Address\s*:",
        r"(?i)\bPO\s*Box\s*#?\s*\d+",
        r"(?i)\bMy\s+Phone\s+Number\b",
        r"(?i)\+1\s*\(\d{3}\)\s*\d{3}",
        r"(?i)https?://[^\s]+",
        r"(?i)www\.[a-z0-9\-]+\.[a-z]{2,}",
        r"(?i)en\.wiki(?:pedia)?\.org",
        r"(?i)\bCopyright\s+Owner\s*\(?s?\)?\s*:",
        r"(?i)\bLicensees?\s*:",
        r"(?i)\bSong\s+Title\s*:\s*[\"\u201c]",
        r"(?i)\bComposer\s+ID\s*#",
        r"(?i)\bAll\s+rights\s+reserved\b",
        r"(?i)\bThank\s*you\s+kindly\b",
        r"(?i)@\w+\s*[®©℗™]",
        r"(?i)\bHorrorcore\s+Dark\s+Ambient\b",
        r"(?i)\bASCAP\s+LLC\b",
        r"(?i)\bUniversal\s+Music\s+Publishing\b",
        r"(?i)\bWarner\s+Chappell\b",
        r"(?i)\.{2,}\s*INAUDIBLE\s*\.{2,}\s*I[\s'\u2019\u00b4´`]*[Mm]\s+SORRY",
        r"(?i)\*{0,3}\s*TRANSCRIBING\s+INCOMPLETE\s*!?\s*\*{0,3}",
        r"(?i)\*{0,3}\s*transcribing\s+completed\s*\*{0,3}",
        r"(?i)\btranscription\s+(?:incomplete|failed|aborted)\b",
        r"(?i)\*{0,3}\s*instrumental\s*solo\s*\*{0,3}",
        r"(?i)\*{0,3}\s*guitar\s*solo\s*\*{0,3}",
        r"(?i)```\s*json\b",
        r"(?i)\{\s*[\"']genre[\"']\s*:",
        r"(?i)\{\s*[\"']artist[\"']\s*:",
        r"(?i)\{\s*[\"']duration[\"']\s*:",
        r"(?i)\(Note\)\s*:\s*These\s+annotations",
        r"(?i)\bCorrect\s+edits\s+welcome\b",
        r"(?i)\bWELCOME\s+HOME\s+MR\.?\s+WHITE\b",
        r"(?i)\bWHAT\s+DID\s+NIGEL\s+SAINT\b",
        r"(?i)\bHORRIFICANT\b",
        r"(?i)\bPeace\s+Out\s*!{2,}",
        r"(?i)\bLove\s+everybody\s*!!",
        r"(?i)\bHope\s+yall\s+have\s+fun\b",
        r"(?i)\bImma\s+go\s+(?:now|home)\b",
        r"(?i)\bTHANKYOU\s+VERY\s+much\b",
        r"(?i)\bsee\s+ya\s+later\b",
        r"(?i)\bbyebye\b",
        r"(?i)\bgood\s+luck\s+have\s+fun\b",
        r"(?i)\balright\s+alright\s+good\s+luck\b",
        r"(?i)\bstay\s+positive\s+all\s+right\b",
        r"[~°•⁣⁡˚˖✧⋆♾️🍁💔❤😭☹👽❄🔥]{3,}",
        r"(?:[a-z]{1,4}){0,2}(?:ooo+|aaa+|eee+|iii+|uuu+|yyy+){4,}",
        # Space-collapsed word salad: 28+ letters with no whitespace (mid-lyric collapse)
        r"[A-Za-z]{28,}",
    ]

    earliest = None
    for pat in _SPAM_ONSET_PATTERNS:
        m = re.search(pat, text)
        if m:
            pos = m.start()
            # For pure letter-runs, only treat as spam if there is already
            # some coherent lyric text before it (avoid cutting short words).
            if pat == r"[A-Za-z]{28,}" and pos < 40:
                continue
            if earliest is None or pos < earliest:
                earliest = pos

    if earliest is not None and earliest > 20:
        cut = text[:earliest].rstrip()
        last_nl = cut.rfind("\n")
        if last_nl > 20:
            cut = cut[:last_nl].rstrip()
        elif len(cut) > 80:
            # Prefer last ordinary space before the collapse so we don't leave
            # a half-glued word (And nostranger / Living In the‎…).
            soft = cut
            for ch in ("\u2009", "\u202f", "\u2005", "\u2006", "\u200a", "\u00a0", "\u200b"):
                soft = soft.replace(ch, " ")
            last_sp = max(soft.rfind(" "), soft.rfind("\t"))
            if last_sp > 40:
                cut = cut[:last_sp].rstrip()
            else:
                for sep in (". ", "! ", "? ", "... ", ", "):
                    pos = cut.rfind(sep)
                    if pos > 40:
                        cut = cut[: pos + len(sep)].rstrip()
                        break
        # Drop a trailing fragment that looks like a glued partial word
        tail = cut[-24:] if len(cut) > 24 else cut
        if " " not in tail and re.search(r"[A-Za-z]{8,}$", cut):
            sp = cut.rfind(" ")
            if sp > 40:
                cut = cut[:sp].rstrip()
        text = cut + "\n[END OF TRANSCRIPTION]"

    def _is_letter_salad(s):
        if not s or len(s) < 24:
            return False
        letters = re.sub(r"[^a-zA-Z]", "", s)
        if len(letters) < 20:
            return False
        repeats = sum(1 for a, b in zip(letters, letters[1:]) if a.lower() == b.lower())
        if repeats / max(1, len(letters) - 1) > 0.45:
            return True
        unique = len(set(letters.lower()))
        if unique <= 8 and len(letters) > 40:
            return True
        if re.search(r"[a-zA-Z]{28,}", s):
            return True
        return False

    def _is_chatty_meta_line(s):
        if not s:
            return False
        low = s.lower()
        markers = (
            "peace out", "love everybody", "hope yall", "imma go",
            "thankyou very", "transcribing", "transcription incomplete",
            "transcription completed", "welcome home mr", "horrificant",
            "take care everyone", "see ya later", "byebye",
            "good luck have fun", "stay positive", "instrumental solo",
            "correct edits welcome", "annotations were generated",
            "```json", '"genre"', '"artist"', '"duration"',
        )
        return any(m in low for m in markers)

    lines = text.splitlines()
    out = []
    prev = None
    streak = 0
    for line in lines:
        stripped = line.strip()
        if stripped and re.fullmatch(r"(.)\1{7,}", stripped.replace(" ", "")):
            continue
        if _is_letter_salad(stripped) or _is_chatty_meta_line(stripped):
            continue
        if stripped and prev is not None and stripped == prev:
            streak += 1
            if streak >= 3:
                continue
        else:
            streak = 1 if stripped else 0
            prev = stripped if stripped else prev
        out.append(line)

    cleaned = "\n".join(out).strip()

    def _collapse_phrase_loop(s):
        m = re.search(r"(.{3,40}?)\1{4,}", s)
        if not m:
            return s
        phrase = m.group(1)
        return re.sub(re.escape(phrase) + r"{3,}", phrase * 2, s)

    cleaned = _collapse_phrase_loop(cleaned)
    cleaned = re.sub(r"(.)\1{9,}", r"\1\1\1", cleaned)
    cleaned = re.sub(r"[~°•⁣⁡˚˖✧⋆♾️🍁💔❤😭☹👽❄🔥]{2,}", " ", cleaned)
    # Normalize exotic Unicode spaces that often appear at the onset of collapse
    for ch in ("\u2009", "\u202f", "\u2005", "\u2006", "\u200a", "\u00a0", "\u200b", "\u200e", "\u200f"):
        cleaned = cleaned.replace(ch, " ")
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    # Drop a trailing glued fragment like "And nostranger" when the last
    # token has no space and looks like two words jammed together.
    parts = cleaned.rsplit(" ", 1)
    if len(parts) == 2 and re.search(r"[a-z]{3,}[A-Z]|[a-z]{10,}", parts[1]):
        # last token looks glued; drop it
        cleaned = parts[0].rstrip()

    # Strip any trailing junk that looks like JSON / markdown fences
    cleaned = re.split(r"\n\s*```|\n\s*\{\s*[\"'](?:genre|artist|duration)", cleaned, maxsplit=1)[0].rstrip()

    if not cleaned.endswith("[END OF TRANSCRIPTION]"):
        # Ensure a clean terminator when we truncated mid-stream
        if len(cleaned) > 0:
            cleaned = cleaned.rstrip() + "\n[END OF TRANSCRIPTION]"

    if len(cleaned) > 3500:
        head = cleaned[:2500]
        last_nl = head.rfind("\n")
        if last_nl > 800:
            head = head[:last_nl]
        cleaned = head.rstrip() + "\n[END OF TRANSCRIPTION — truncated; spam/loop detected]"

    cleaned = _flag_repeated_section_drift(cleaned)

    return cleaned


def _flag_repeated_section_drift(text):
    """
    Detect likely word-substitution hallucination on a repeated section
    (e.g. the same chorus sung 2-3x but transcribed with different wording
    each time) and annotate it rather than silently altering it — we can't
    tell which occurrence (if any) is correct, only that they disagree more
    than a genuinely repeated section should.

    This targets the failure mode where the *first* occurrence of a section
    is transcribed accurately and a *later* occurrence of the same sung
    section drifts into different words (a "hallucinated" repeat), which
    plain loop/spam detection can't catch because the drifted text is
    coherent, on-theme, and not literally repeated.
    """
    if not text:
        return text

    # Group lyric lines under the section header they follow, e.g. "[Chorus] (0:34-0:58)"
    section_pat = re.compile(r"^\s*\[([^\]]+)\]", re.IGNORECASE)
    sections = {}
    current = None
    for line in text.splitlines():
        m = section_pat.match(line)
        if m:
            current = m.group(1).strip().lower()
            current = re.sub(r"\s*\d+$", "", current).strip()  # "chorus 2" -> "chorus"
            sections.setdefault(current, []).append([])
            continue
        if current is not None and line.strip():
            sections[current][-1].append(line.strip())

    def _norm_words(lines):
        words = re.findall(r"[a-z']+", " ".join(lines).lower())
        return set(w for w in words if len(w) > 2)

    flags = []
    for name, occurrences in sections.items():
        if name != "chorus" and "chorus" not in name:
            continue
        if len(occurrences) < 2:
            continue
        word_sets = [_norm_words(o) for o in occurrences if o]
        for i in range(1, len(word_sets)):
            a, b = word_sets[0], word_sets[i]
            if not a or not b:
                continue
            overlap = len(a & b) / max(1, len(a | b))
            if overlap < 0.35:
                flags.append(
                    f"occurrence {i + 1} of [{name}] shares only "
                    f"{overlap:.0%} of its words with occurrence 1"
                )

    if flags:
        text = (
            text.rstrip()
            + "\n\n[TRANSCRIPTION CAUTION — possible repeated-section drift: "
            + "; ".join(flags)
            + ". A repeated section normally uses the same words each time it "
            "recurs; this large a mismatch suggests the model paraphrased or "
            "invented wording on at least one occurrence rather than "
            "transcribing what was actually sung. Treat the disagreeing "
            "occurrence(s) as unverified and prefer the first occurrence, "
            "file-tag lyrics, or an [inaudible] read over quoting this "
            "wording as fact.]"
        )

    return text


def _cross_check_lyrics_transcriptions(primary, secondary, min_unverified_run=5):
    """
    Compare two INDEPENDENT decodes of the same lyrics-transcription prompt
    against the same audio (one greedy, one sampled) and flag stretches of
    `primary` where they diverge.

    Why this catches what n-gram/repetition tuning cannot: greedy decoding
    is deterministic, so a single decode can be fluently, confidently wrong
    with no repetition anywhere to detect. When Music Flamingo is actually
    grounded in the audio for a passage, independent decodes — even with
    different sampling — tend to converge on the same or near-same words,
    because the audio is anchoring the output. When grounding is weak and
    the model is filling in from its language-model prior instead of the
    audio (a generic "song about games" continuation, say), independent
    decodes tend to diverge, because that prior isn't anchored to anything
    and can resolve differently each time. Divergence is treated purely as
    a confidence signal — NOT as evidence that either reading is correct.
    """
    if not primary or not secondary:
        return primary

    import difflib

    def _tokenize(s):
        return re.findall(r"\n|\S+", s)

    def _key(tok):
        return tok.lower().strip(".,!?\"'()[]")

    a_tokens = _tokenize(primary)
    b_tokens = _tokenize(secondary)
    a_keys = [_key(t) for t in a_tokens]
    b_keys = [_key(t) for t in b_tokens]

    sm = difflib.SequenceMatcher(None, a_keys, b_keys, autojunk=False)

    out_pieces = []

    def _emit(tok):
        if tok == "\n":
            out_pieces.append("\n")
        else:
            if out_pieces and out_pieces[-1] not in ("\n", ""):
                out_pieces.append(" ")
            out_pieces.append(tok)

    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        run_len = i2 - i1
        if tag == "equal" or run_len < min_unverified_run:
            for tok in a_tokens[i1:i2]:
                _emit(tok)
            continue

        # A disagreement long enough to be a plausible fabricated line/phrase
        # rather than incidental word-choice noise ("played" vs "plays").
        alt_words = [t for t in b_tokens[j1:j2] if t != "\n"]
        alt = " ".join(alt_words).strip()

        _emit("[UNVERIFIED\u2192")
        for tok in a_tokens[i1:i2]:
            _emit(tok)
        note = (
            f" \u2190two independent transcriptions disagree here"
            + (f'; other reading: "{alt}"' if alt else "; no comparable alt reading")
            + "; do not treat either as confirmed]"
        )
        out_pieces.append(note)

    return "".join(out_pieces)


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


# ---------------------------------------------------------------------------
# Genre-conditioned BPM preference (Deep-Cuts-inspired octave correction)
# ---------------------------------------------------------------------------
# When detectors disagree on half/double-time, use a coarse genre family to
# bias toward the pulse that is musically typical for that style. This is a
# soft prior only — median-IBI corroboration and multi-source agreement still
# win when they are strong.

_BPM_GENRE_PREFS = {
    # family: (preferred_lo, preferred_hi, prefer_half_when_fast, prefer_double_when_slow, note)
    "electronicish": (118.0, 132.0, True, True, "house/techno/dance pulse typically 120–130"),
    "dnb": (160.0, 178.0, False, False, "drum & bass / jungle typically 160–175; keep fast readings"),
    "hiphop": (70.0, 100.0, True, True, "hip-hop / trap often feels 70–95 (or double-time 140–160)"),
    "rockish": (100.0, 145.0, True, True, "rock / indie / punk typical mid-tempo pulse"),
    "popish": (95.0, 130.0, True, True, "pop / indie-pop typical mid-tempo pulse"),
    "ballad": (55.0, 95.0, True, True, "ballad / slow song — prefer the lower pulse"),
    "other": (80.0, 140.0, True, True, "general mid-tempo preference"),
}

_BPM_DNB_TOKENS = (
    "drum and bass", "drum & bass", "dnb", "d&b", "jungle", "breakcore",
    "neurofunk", "liquid dnb", "drum n bass",
)
_BPM_HIPHOP_TOKENS = (
    "hip hop", "hip-hop", "hiphop", "rap", "trap", "grime", "drill",
    "boom bap", "boom-bap", "r&b", "rnb", "r & b",
)
_BPM_BALLAD_TOKENS = (
    "ballad", "slowcore", "ambient", "drone", "singer-songwriter",
    "singer songwriter", "folk ballad", "acoustic ballad",
)


def _bpm_genre_family_from_hint(genre_hint):
    """Map a free-form genre string (MF top rank, PANNs label, etc.) to a
    coarse BPM-preference family used by _preferred_tempo / reconcile_bpm."""
    if not genre_hint:
        return "other"
    t = _normalize_genre_token(str(genre_hint))
    if not t:
        return "other"
    for tok in _BPM_DNB_TOKENS:
        if tok in t:
            return "dnb"
    for tok in _BPM_HIPHOP_TOKENS:
        if tok in t:
            return "hiphop"
    for tok in _BPM_BALLAD_TOKENS:
        if tok in t:
            return "ballad"
    fam = _genre_family(t)
    if fam in ("electronicish", "rockish", "popish"):
        return fam
    return "other"


def _genre_hint_from_analysis_text(text, panns_genre_tags=None):
    """Best-effort genre string for BPM bias: MF GENRE_RANKED top, else PANNs."""
    labels = extract_genre_ranked_labels(text or "", max_n=1)
    if labels:
        return labels[0]
    if panns_genre_tags:
        for item in panns_genre_tags:
            if item and len(item) >= 1 and item[0]:
                return str(item[0])
    return None


def reconcile_bpm(
    mf_bpm,
    essentia_bpm,
    objective_bpm=None,
    essentia_median_bpm=None,
    objective_median_bpm=None,
    genre_hint=None,
):
    """
    Determine a single recommended BPM for discussion.

    Priority / logic:
    1. Prefer agreement between sources.
    2. When two sources differ by ~2x, treat the higher as a likely double-time
       misread and prefer the lower (common song-pulse) value — unless a
       genre_hint (e.g. drum & bass) argues that the fast pulse is correct.
    3. Fall back to Music Flamingo's TEMPO_BPM when it is the only strong signal.
    4. Always return a concrete number when any source is available.

    essentia_median_bpm / objective_median_bpm (optional): each source's own
    median-inter-beat-interval-derived BPM, computed independently of that
    source's raw autocorrelation/tempogram estimate. When only one detector
    source is available, this is passed to _preferred_tempo as corroboration
    so a genuinely fast, internally-confirmed tempo isn't halved just because
    it clears the double-time threshold with nothing to check it against.

    genre_hint (optional): free-form genre label used as a soft prior for
    half/double-time decisions (see _bpm_genre_family_from_hint).
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

    corroboration_by_src = {
        "essentia": essentia_median_bpm,
        "objective": objective_median_bpm,
    }

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

    genre_fam = _bpm_genre_family_from_hint(genre_hint)
    genre_note = ""
    if genre_hint and genre_fam != "other":
        genre_note = f" Genre prior: {genre_fam} ({genre_hint})."

    if len(candidates) == 1:
        src, val = candidates[0]
        preferred, _, note = _preferred_tempo(
            val,
            corroborating_bpm=corroboration_by_src.get(src),
            genre_hint=genre_hint,
        )
        q = _quantize_bpm(preferred)
        return q, f"{src} only. {note}{genre_note}".strip()

    by_src = {s: v for s, v in candidates}
    mf = by_src.get("mf")
    ess = by_src.get("essentia")
    obj = by_src.get("objective")

    def close(a, b):
        if a is None or b is None:
            return False
        ratio = max(a, b) / max(min(a, b), 1e-6)
        return ratio <= 1.15

    # When detectors "agree" (within tolerance, not necessarily identical),
    # average the agreeing readings rather than arbitrarily keeping one and
    # discarding the other — real-world detectors routinely land 1-3 BPM
    # apart even on a clean agreement, and picking a single source lets that
    # source's individual measurement noise pass straight through as the
    # final answer instead of being smoothed out.
    if (
        mf is not None and ess is not None and obj is not None
        and close(mf, ess) and close(mf, obj) and close(ess, obj)
    ):
        avg = (mf + ess + obj) / 3.0
        return _quantize_bpm(avg), f"MF, Essentia, and objective detector all agree (~{avg:.1f}, averaged).{genre_note}"
    if mf is not None and ess is not None and close(mf, ess):
        avg = (mf + ess) / 2.0
        return _quantize_bpm(avg), f"Essentia and MF agree (~{avg:.1f}, averaged).{genre_note}"
    if mf is not None and obj is not None and close(mf, obj):
        avg = (mf + obj) / 2.0
        return _quantize_bpm(avg), f"MF and objective detector agree (~{avg:.1f}, averaged).{genre_note}"
    if ess is not None and obj is not None and close(ess, obj):
        avg = (ess + obj) / 2.0
        return _quantize_bpm(avg), f"Essentia and objective detector agree (~{avg:.1f}, averaged).{genre_note}"

    def is_double(a, b):
        if a is None or b is None:
            return False
        ratio = max(a, b) / max(min(a, b), 1e-6)
        return 1.8 <= ratio <= 2.2

    def prefer_of_double(a, b, label_a, label_b):
        """When a and b are ~2x apart, choose the pulse that fits genre + mid-range."""
        lo, hi = (a, b) if a <= b else (b, a)
        lo_src = label_a if a <= b else label_b
        hi_src = label_b if a <= b else label_a
        prefs = _BPM_GENRE_PREFS.get(genre_fam, _BPM_GENRE_PREFS["other"])
        pref_lo, pref_hi, prefer_half, prefer_double, fam_note = prefs

        if genre_fam == "dnb" and pref_lo <= hi <= pref_hi + 5:
            return _quantize_bpm(hi), (
                f"{hi_src} ({hi}) is ~2x {lo_src} ({lo}); genre prior ({fam_note}) "
                f"favours the fast pulse."
            )
        if genre_fam == "hiphop" and hi >= 130.0 and 60.0 <= lo <= 105.0:
            return _quantize_bpm(lo), (
                f"{hi_src} ({hi}) is ~2x {lo_src} ({lo}); genre prior ({fam_note}) "
                f"prefers the lower pulse."
            )
        if genre_fam == "electronicish":
            if pref_lo <= lo <= pref_hi:
                return _quantize_bpm(lo), (
                    f"{hi_src} ({hi}) is ~2x {lo_src} ({lo}); genre prior ({fam_note}) "
                    f"prefers the mid pulse."
                )
            if pref_lo <= hi <= pref_hi:
                return _quantize_bpm(hi), (
                    f"{hi_src} ({hi}) is ~2x {lo_src} ({lo}); genre prior ({fam_note}) "
                    f"prefers the mid pulse."
                )
        if hi >= 140.0 and 55.0 <= lo <= 110.0 and prefer_half:
            return _quantize_bpm(lo), (
                f"{hi_src} ({hi}) is ~2x {lo_src} ({lo}); preferring lower pulse "
                f"to avoid double-time misread."
            )
        for val, src in ((a, label_a), (b, label_b)):
            if 85.0 <= val <= 140.0:
                return _quantize_bpm(val), f"Preferring mid-range pulse from {src} ({val})."
        return _quantize_bpm(lo), f"2x pair {lo}/{hi}; defaulting to lower ({lo})."

    if mf is not None and ess is not None and is_double(mf, ess):
        return prefer_of_double(mf, ess, "MF", "Essentia")

    if mf is not None and obj is not None and is_double(mf, obj):
        return prefer_of_double(mf, obj, "MF", "objective")

    if ess is not None and obj is not None and is_double(ess, obj):
        return prefer_of_double(ess, obj, "Essentia", "objective")

    if mf is not None:
        preferred, _, note = _preferred_tempo(mf, genre_hint=genre_hint)
        return _quantize_bpm(preferred), f"Using MF musical pulse ({mf}). {note}{genre_note}".strip()

    src, val = candidates[0]
    preferred, _, note = _preferred_tempo(val, genre_hint=genre_hint)
    return _quantize_bpm(preferred), f"{src} fallback. {note}{genre_note}".strip()

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


def extract_essentia_median_bpm(text):
    """Extracts the BPM implied by Essentia's median inter-beat interval --
    an independent corroborating measurement (from actual detected beat
    timestamps) distinct from the raw autocorrelation/tempogram estimate.
    Used to keep reconcile_bpm's single-source path from halving a
    genuinely fast, internally-confirmed tempo (see reconcile_bpm docstring)."""
    if not text:
        return None
    m = re.search(r"Essentia median inter-beat interval:.*?\(\s*([0-9.]+)\s*BPM\s*\)", text, re.IGNORECASE)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    return None


def extract_objective_median_bpm(text):
    """Extracts the BPM implied by the librosa/objective report's median
    inter-beat interval, mirroring extract_essentia_median_bpm."""
    if not text:
        return None
    m = re.search(r"(?<!Essentia )median inter-beat interval:.*?\(\s*([0-9.]+)\s*BPM\s*\)", text, re.IGNORECASE)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    return None


def extract_key_from_text(text):
    """Extracts Music Flamingo's KEY=... field, e.g. 'KEY=F# minor'."""
    if not text:
        return None
    m = re.search(r"KEY\s*=\s*(.+)", text)
    if not m:
        return None
    val = m.group(1).strip().splitlines()[0].strip()
    if not val or val.lower() in ("uncertain", "unknown", "n/a", "none", "..."):
        return None
    return val


def extract_essentia_key_from_text(text):
    """Extracts Essentia's 'Essentia estimated key: X, strength=Y' line."""
    if not text:
        return None
    m = re.search(r"Essentia estimated key:\s*([^,\n]+?)(?:,\s*strength=([0-9.]+))?\s*$", text, re.IGNORECASE | re.MULTILINE)
    if not m:
        return None
    key_val = m.group(1).strip()
    if not key_val or key_val.lower() in ("unavailable", "unknown", "none", "n/a"):
        return None
    strength_val = None
    if m.group(2):
        try:
            strength_val = float(m.group(2))
        except ValueError:
            strength_val = None
    return key_val, strength_val


_KEY_NAME_ALIASES = {
    "db": "c#", "eb": "d#", "gb": "f#", "ab": "g#", "bb": "a#",
}


def _normalize_key_for_compare(key_text):
    """Normalize a key string to (pitch_class, mode) for equivalence checks.
    Treats enharmonic spellings (Db == C#) and 'maj'/'major', 'min'/'minor' the same."""
    if not key_text:
        return None
    t = key_text.strip().lower()
    mode = "minor" if re.search(r"\bmin(or)?\b", t) else ("major" if re.search(r"\bmaj(or)?\b", t) else None)
    root_match = re.match(r"([a-g])([#b]?)", t)
    if not root_match:
        return None
    root = root_match.group(1) + root_match.group(2)
    root = _KEY_NAME_ALIASES.get(root, root)
    # No explicit mode stated: assume major, the conventional default.
    return (root, mode or "major")


def reconcile_key(mf_key, essentia_key, essentia_strength=None, min_essentia_strength=0.6):
    """
    Determine a single recommended key for discussion, mirroring reconcile_bpm's
    approach: prefer agreement between independent sources, and be explicit
    about confidence rather than silently trusting whichever source happens
    to be read first.

    Priority:
    1. If MF and Essentia agree (same pitch class + mode, enharmonics aside),
       report that key with high confidence.
    2. If they disagree, prefer Essentia only when its confidence clears
       min_essentia_strength; otherwise prefer MF (it has access to chords/
       melody/bass movement, which is stronger key evidence than Essentia's
       chroma-only profile-matching) but flag the disagreement so the writer
       doesn't overstate certainty.
    3. Fall back to whichever single source is available.
    """
    mf_norm = _normalize_key_for_compare(mf_key) if mf_key else None
    ess_norm = _normalize_key_for_compare(essentia_key) if essentia_key else None

    if mf_key and essentia_key and mf_norm and ess_norm:
        if mf_norm == ess_norm:
            return mf_key, f"MF and Essentia agree (Essentia strength={essentia_strength if essentia_strength is not None else 'n/a'})."
        if essentia_strength is not None and essentia_strength >= min_essentia_strength:
            return (
                essentia_key,
                f"MF said {mf_key}, Essentia said {essentia_key} (strength={essentia_strength:.2f}, "
                "above confidence threshold) — preferring Essentia's independent chroma-based estimate. "
                "Treat as moderately confident; mention the disagreement is possible if asked."
            )
        return (
            mf_key,
            f"MF said {mf_key}, Essentia said {essentia_key} "
            f"(strength={essentia_strength if essentia_strength is not None else 'n/a'}, "
            "below/no confidence threshold) — preferring MF's harmonic/melodic reading, "
            "but this is a genuine disagreement between sources; hedge appropriately."
        )

    if mf_key:
        return mf_key, "Only MF produced a key estimate."
    if essentia_key:
        if essentia_strength is not None and essentia_strength < min_essentia_strength:
            return essentia_key, f"Only Essentia produced a key estimate, and confidence is low (strength={essentia_strength:.2f}); treat as uncertain."
        return essentia_key, "Only Essentia produced a key estimate."
    return None, "unavailable"



# Genre families for conflict detection between MF GENRE_RANKED and PANNs tags.
_GENRE_FAMILY_ROCKISH = {
    "rock", "punk", "pop-punk", "pop punk", "emo", "metal", "heavy metal",
    "indie rock", "indie-rock", "alternative rock", "alt-rock", "alt rock",
    "hard rock", "garage rock", "post-punk", "post punk", "grunge",
    "power pop", "power-pop", "pop rock", "pop-rock", "ska",
}
_GENRE_FAMILY_ELECTRONICISH = {
    "electronic", "electronica", "edm", "house", "techno", "trance",
    "dubstep", "drum and bass", "dnb", "dance", "dance-pop", "dance pop",
    "synth-pop", "synthpop", "synth pop", "electro", "electro-pop",
    "electropop", "club", "disco", "ambient", "electronic dance music",
}
_GENRE_FAMILY_POPISH = {
    "pop", "indie pop", "indie-pop", "art pop", "dream pop", "chamber pop",
}


def _normalize_genre_token(s):
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = s.replace("_", " ")
    return s


def extract_genre_ranked_labels(text, max_n=8):
    """Pull ordered genre labels from a GENRE_RANKED=... field in MF text."""
    if not text:
        return []
    m = re.search(
        r"(?:GENRE_RANKED|Genre\s+Ranked)\s*[=:\-]\s*(.+?)(?=\n\s*(?:[A-Z][A-Z0-9_ ]{2,}=|GENRE_ADJACENT|GENRE_RULED|$))",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    if not m:
        # Fallback: first line containing GENRE_RANKED
        for line in (text or "").splitlines():
            if re.search(r"GENRE_RANKED|Genre\s+Ranked", line, re.I):
                m_line = re.search(r"[=:\-]\s*(.+)$", line)
                if m_line:
                    blob = m_line.group(1)
                    break
        else:
            return []
    else:
        blob = m.group(1)

    labels = []
    # Split on numbered ranks or semicolons
    parts = re.split(r"(?:;|\n|\d+\)\s*)", blob)
    for p in parts:
        p = p.strip()
        if not p:
            continue
        # Drop confidence parentheticals
        p = re.sub(r"\([^)]*confidence[^)]*\)", "", p, flags=re.I).strip()
        p = re.sub(r"^\d+[\).\]]\s*", "", p).strip(" ,;-")
        if not p or len(p) < 2:
            continue
        # Skip pure confidence words
        if p.lower() in ("high", "medium", "low", "confidence"):
            continue
        tok = _normalize_genre_token(p)
        # Keep first few words of each rank item
        tok = re.split(r"[./|]", tok)[0].strip()
        if tok and tok not in labels:
            labels.append(tok)
        if len(labels) >= max_n:
            break
    return labels


def _genre_family(label):
    t = _normalize_genre_token(label)
    # Longer / more specific keys first via explicit checks
    for fam, members in (
        ("rockish", _GENRE_FAMILY_ROCKISH),
        ("electronicish", _GENRE_FAMILY_ELECTRONICISH),
        ("popish", _GENRE_FAMILY_POPISH),
    ):
        for m in members:
            if m in t or t == m:
                return fam
        # token-wise
        for word in t.replace("-", " ").split():
            if word in members or any(word == mm.replace("-", " ") for mm in members):
                return fam
    # substring heuristics
    if any(x in t for x in ("punk", "emo", "metal", "grunge", "rock")):
        return "rockish"
    if any(x in t for x in ("house", "techno", "edm", "electro", "synth", "dance", "trance", "dubstep")):
        return "electronicish"
    if "pop" in t:
        return "popish"
    return "other"


def reconcile_genre(mf_text, panns_genre_tags):
    """Compare MF GENRE_RANKED with PANNs genre tags. Returns (recommended_label, note)
    or (None, reason) when nothing useful can be said."""
    mf_labels = extract_genre_ranked_labels(mf_text or "")
    panns = []
    for item in (panns_genre_tags or []):
        if len(item) >= 2:
            panns.append((_normalize_genre_token(item[0]), float(item[1])))
    if not mf_labels and not panns:
        return None, "unavailable"

    mf_top = mf_labels[0] if mf_labels else ""
    mf_fam = _genre_family(mf_top) if mf_top else "other"
    panns_top = panns[0][0] if panns else ""
    panns_top_p = panns[0][1] if panns else 0.0
    panns_fam = _genre_family(panns_top) if panns_top else "other"

    # Aggregate PANNs mass per family
    fam_mass = {"rockish": 0.0, "electronicish": 0.0, "popish": 0.0, "other": 0.0}
    for lab, prob in panns:
        fam_mass[_genre_family(lab)] += prob

    # Strong electronic/dance PANNs vs rock/punk MF top → prefer electronic/pop framing
    rock_vs_elec = (
        mf_fam == "rockish"
        and (
            fam_mass["electronicish"] >= 0.12
            or (panns_fam == "electronicish" and panns_top_p >= 0.10)
            or (fam_mass["electronicish"] + fam_mass["popish"] >= 0.18 and fam_mass["rockish"] < 0.08)
        )
    )
    if rock_vs_elec:
        # Build a human label from top PANNs electronic/pop tags
        prefer = [lab for lab, p in panns if _genre_family(lab) in ("electronicish", "popish")]
        if not prefer:
            prefer = [panns_top] if panns_top else ["electronic pop"]
        # Prefer compound if both pop and electronic mass present
        if fam_mass["electronicish"] >= 0.10 and fam_mass["popish"] >= 0.08:
            rec = "electronic / dance-pop"
        elif prefer:
            rec = prefer[0]
            if "pop" not in rec and fam_mass["popish"] >= 0.08:
                rec = f"{rec} pop" if "dance" in rec or "electronic" in rec else f"pop / {rec}"
        else:
            rec = "electronic pop"
        note = (
            f"Music Flamingo GENRE_RANKED led with '{mf_top}' (rock/punk family), but the "
            f"independent PANNs genre signal favours electronic/dance/pop "
            f"(tags: {', '.join(f'{a} {b*100:.0f}%' for a,b in panns[:4]) or 'n/a'}). "
            "A secondary guitar texture alone should not keep a dance/electronic track "
            "ranked as pop-punk/rock. Use this recommended label as the primary genre "
            "framing; mention rock/punk only as a minor secondary flavour if still audible."
        )
        return rec, note

    # Agreeing families or weak PANNs → keep MF top when present
    if mf_top:
        if panns and panns_fam == mf_fam:
            note = (
                f"GENRE_RANKED top '{mf_top}' aligns with independent PANNs signal "
                f"('{panns_top}' {panns_top_p*100:.0f}%). Use the ranked list as primary."
            )
            return mf_top, note
        if panns and panns_fam != mf_fam and panns_top_p >= 0.15:
            note = (
                f"GENRE_RANKED top is '{mf_top}'; independent PANNs top is '{panns_top}' "
                f"({panns_top_p*100:.0f}%). Families differ — prefer a broader description "
                f"that acknowledges both rather than forcing one scene label. "
                f"Default conversational framing: '{mf_top}' with possible '{panns_top}' lean."
            )
            return f"{mf_top} (with {panns_top} lean)", note
        return mf_top, (
            f"Primary framing from GENRE_RANKED: '{mf_top}'. "
            + (
                f"Independent PANNs tags: {', '.join(f'{a} {b*100:.0f}%' for a,b in panns[:4])}."
                if panns else "No strong independent PANNs genre tags."
            )
        )

    if panns_top:
        return panns_top, f"No parseable GENRE_RANKED; using independent PANNs top '{panns_top}'."
    return None, "unavailable"


def format_recommended_genre_block(recommended, note):
    if not recommended:
        return (
            "\n\nRECOMMENDED GENRE FOR DISCUSSION: unavailable. "
            "Fall back to GENRE_RANKED with caution; do not over-commit to scene labels "
            "from a single instrument texture."
        )
    return (
        f"\n\nRECOMMENDED GENRE FOR DISCUSSION: {recommended}. "
        f"Reasoning: {note} "
        "This is the primary genre framing for the user. "
        "Do not lead with a conflicting early GENRE_RANKED label when this block "
        "explicitly revises rock/punk vs electronic/dance identity."
    )



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


def _collapse_runaway_genre_ranked(text):
    """
    Safety net for GENRE_RANKED / Genre Ranked runaway loops, e.g.:
      Genre Ranked=1) rock; 2) indie rock; 3) indie pop; ... 4) jangle pop;
      5) indie rock; 6) indie pop; ... (hundreds of cycling items)

    Keep the first occurrence of each unique genre label (up to 8), drop the
    rest, and mark that the list was collapsed.
    """
    if not text:
        return text

    # Locate the start of a ranked-genre block (strict or loose label).
    start_re = re.compile(r"(?:GENRE_RANKED|Genre\s+Ranked)\s*[=:\-]\s*", re.IGNORECASE)
    m = start_re.search(text)
    if not m:
        return text

    prefix = text[m.start(): m.end()]
    rest = text[m.end():]

    # End of the ranking block: next major analysis label, double newline, or
    # a clearly non-genre section. Rankings are often one huge line.
    end_re = re.compile(
        r"(?=\n\s*(?:KEY|TEMPO_BPM|CHORDS|STRUCTURE|INSTRUMENTATION|VOCALS|"
        r"GENRE_ADJACENT|GENRE_RULED_OUT|FULL LYRICS|ERA_|VOCAL /|"
        r"RECOMMENDED |\[Independent|STEM MIDI|11\.\s*ERA)\b)"
        r"|(?=\n\n)"
        r"|(?=\nGenre\s+Adjacent\b)"
        r"|(?=\nGenre\s+Ruled)",
        re.IGNORECASE,
    )
    end_m = end_re.search(rest)
    body = rest[: end_m.start()] if end_m else rest
    tail = rest[end_m.start():] if end_m else ""

    # Pull numbered items: "1) rock" / "12) indie pop (confidence: medium)"
    items = re.findall(r"\d+\)\s*([^;\n]+)", body)
    items = [
        re.sub(r"\s*\(confidence:[^)]*\)", "", it, flags=re.I).strip(" ;,.")
        for it in items
    ]
    items = [it for it in items if it]

    if len(items) < 12:
        # Still collapse pure semicolon cycles of the same few labels
        cycle = re.compile(
            r"((?:indie rock|indie pop|power pop|jangle pop|alternative rock|"
            r"post-punk(?: revival)?|garage rock(?: revival)?|rock)"
            r"(?:\s*;\s*(?:indie rock|indie pop|power pop|jangle pop|alternative rock|"
            r"post-punk(?: revival)?|garage rock(?: revival)?|rock)){8,})",
            re.IGNORECASE,
        )

        def _cycle_repl(cm):
            parts = [p.strip() for p in re.split(r"\s*;\s*", cm.group(1)) if p.strip()]
            unique, seen = [], set()
            for p in parts:
                k = p.lower()
                if k not in seen:
                    seen.add(k)
                    unique.append(p)
            return "; ".join(unique) + " (collapsed genre cycle)"

        return text[: m.start()] + prefix + cycle.sub(_cycle_repl, body) + tail

    unique, seen = [], set()
    for it in items:
        key = it.lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(it)
        if len(unique) >= 8:
            break

    ranked = "; ".join(f"{i}) {g}" for i, g in enumerate(unique, 1))
    collapsed_body = (
        f"{ranked} "
        f"(collapsed from {len(items)} highly repetitive ranked items; "
        f"kept first unique labels only)"
    )
    return text[: m.start()] + prefix + collapsed_body + tail


def _collapse_runaway_repetition_fields(text):
    """Apply all known runaway-repetition safety nets to an analysis blob."""
    if not text:
        return text
    text = _collapse_runaway_chord_repetition(text)
    text = _collapse_runaway_genre_ranked(text)
    return text

def _preferred_tempo(bpm, corroborating_bpm=None, genre_hint=None):
    """
    Tempo interpretation helper for a single detector value.

    Detectors often report double the felt pulse, so a lone fast reading is
    treated as a likely double-time misread. BUT: if a second, independently
    calculated figure corroborates the raw (un-halved) reading -- e.g. the
    median inter-beat interval derived from actual detected beat timestamps --
    that agreement is strong evidence the fast tempo is genuine.
    corroborating_bpm, when given, is checked against the RAW bpm before any
    halving is applied; if it agrees, the halving is skipped.

    genre_hint (optional): free-form genre label. Soft prior so that e.g.
    drum & bass keeps a fast reading while house/techno prefers ~120–130.
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

    genre_fam = _bpm_genre_family_from_hint(genre_hint)
    prefs = _BPM_GENRE_PREFS.get(genre_fam, _BPM_GENRE_PREFS["other"])
    pref_lo, pref_hi, prefer_half, prefer_double, fam_note = prefs

    def _corroborates(raw_val):
        if corroborating_bpm is None:
            return False
        try:
            corr = float(corroborating_bpm)
        except (TypeError, ValueError):
            return False
        if corr <= 0 or raw_val <= 0:
            return False
        ratio = max(corr, raw_val) / min(corr, raw_val)
        return ratio <= 1.05

    def _in_pref_band(val):
        return pref_lo <= val <= pref_hi

    # Genre-aware fast path: DnB keeps in-band or corroborated fast tempi
    if genre_fam == "dnb" and bpm >= 150.0:
        if _corroborates(bpm) or _in_pref_band(bpm):
            note = (
                f"fast detector value ({bpm}) kept — genre prior ({fam_note})"
                + (f"; corroborated (~{corroborating_bpm} BPM)" if _corroborates(bpm) else "")
                + "."
            )
            return preferred, cands, note
        if bpm > 185.0 and _in_pref_band(half):
            preferred = half
            note = (
                f"very fast detector value ({bpm}) outside typical DnB band; "
                f"preferring half ({half}) which fits genre prior better."
            )
            return preferred, cands, note

    if bpm >= 150.0 and 70.0 <= half <= 100.0:
        if _corroborates(bpm):
            note = (
                f"fast detector value ({bpm}) is corroborated by an independent "
                f"measurement (~{corroborating_bpm} BPM); keeping it rather than "
                "assuming a double-time misread."
            )
        elif not prefer_half and _in_pref_band(bpm):
            note = f"fast detector value ({bpm}) kept due to genre prior ({fam_note})."
        elif prefer_half:
            preferred = half
            note = (
                "fast detector value is likely a double-time/subdivision reading; "
                "preferring the lower pulse"
                + (f" (genre prior: {fam_note})" if genre_hint else "")
                + "."
            )
    elif bpm > 175.0 and 70.0 <= half <= 140.0:
        if _corroborates(bpm):
            note = (
                f"very fast detector value ({bpm}) is corroborated by an independent "
                f"measurement (~{corroborating_bpm} BPM); keeping it."
            )
        elif genre_fam == "dnb" and _in_pref_band(bpm):
            note = f"very fast detector value ({bpm}) kept — genre prior ({fam_note})."
        else:
            preferred = half
            note = "very fast detector value may be a double-time reading; preferring the slower candidate."
    elif bpm < 70.0 and 80.0 <= double <= 140.0:
        if corroborating_bpm is not None and _corroborates(bpm):
            note = (
                f"slow detector value ({bpm}) is corroborated by an independent "
                f"measurement (~{corroborating_bpm} BPM); keeping it."
            )
        elif prefer_double:
            preferred = double
            note = (
                "slow detector value may be half-time; the doubled candidate may be the intended beat"
                + (f" (genre prior: {fam_note})" if genre_hint else "")
                + "."
            )

    if note == "" and genre_hint and not _in_pref_band(preferred):
        if prefer_half and _in_pref_band(half) and abs(preferred - bpm) < 0.5 and bpm >= 140.0:
            preferred = half
            note = f"nudged to half-time ({half}) to match genre prior ({fam_note})."
        elif prefer_double and _in_pref_band(double) and abs(preferred - bpm) < 0.5 and bpm < 75.0:
            preferred = double
            note = f"nudged to double-time ({double}) to match genre prior ({fam_note})."

    return preferred, cands, note


def _dynamics_label_from_crest(crest_db):
    """Map crest-factor dB to a plain-language dynamics band for the writer.

    Short labels stay era-neutral. Longer notes may mention when a reading is
    *consistent with* loudness-war limiting, but must not treat crest alone as
    proof of era or as "brickwalled" when LRA/crest are only moderately low.
    """
    try:
        c = float(crest_db)
    except (TypeError, ValueError):
        return None, None
    if c < 8.0:
        return "very low crest / heavily limited", (
            "very low peak-to-average ratio — heavy limiting/compression is likely; "
            "this pattern is common on many late-1990s–2020s commercial masters, "
            "but do not use crest alone to date the release"
        )
    if c < 10.0:
        return "low crest / limited", (
            "low peak-to-average ratio — limited headroom between average level and "
            "peaks; often consistent with assertive commercial limiting, but not "
            "proof of era by itself"
        )
    if c < 12.0:
        return "moderately compressed", (
            "moderate dynamic range — some compression but not brickwalled; "
            "do not treat this alone as proof of era"
        )
    if c < 14.0:
        return "moderately dynamic", (
            "noticeable dynamic range — less aggressive limiting than a typical "
            "hot commercial master; do not treat this alone as proof of era"
        )
    if c < 18.0:
        return "wide / dynamic", (
            "wide dynamic range — peaks sit well above average level; more common "
            "on older, audiophile, or lightly mastered material"
        )
    return "very wide / highly dynamic", (
        "very wide dynamic range — little limiting; dynamics closer to an unmastered "
        "or vinyl-oriented presentation"
    )


def _lufs_label(lufs):
    """Plain-language band for integrated loudness. More negative = quieter.

    Boundaries must stay in sync with the "Rough anchors" line in
    WRITER_MUSIC_RULES. Labels describe LEVEL only — never mastering era,
    streaming intent, or compression (those come from LRA/crest + context).
    Avoid "streaming-normalized" as a band name: the same LUFS can appear on
    an 1980s CD and a modern stream encode for unrelated reasons.
    """
    try:
        v = float(lufs)
    except (TypeError, ValueError):
        return None
    if v > -9.0:
        return "very loud"
    if v > -12.0:
        return "loud"
    if v > -15.0:
        return "moderate integrated level"
    if v > -20.0:
        return "quieter / more open master"
    return "quiet / highly dynamic or low-level master"


def _lra_label(lra):
    """Plain-language band for loudness range. Higher = more dynamic swing."""
    try:
        v = float(lra)
    except (TypeError, ValueError):
        return None
    if v < 5.0:
        return "very tight / little section-to-section swing"
    if v < 8.0:
        return "tight / controlled dynamics"
    if v < 12.0:
        return "moderate dynamic swing"
    if v < 16.0:
        return "wide dynamic swing"
    return "very wide dynamic swing"


def extract_crest_db_from_text(text):
    """Pull crest-factor dB from objective or Essentia report text."""
    if not text:
        return None
    patterns = (
        r"crest factor\s*/\s*dynamic-range proxy:\s*([0-9]+(?:\.[0-9]+)?)\s*dB",
        r"crest factor(?:\s+proxy)?(?:\s*\([^)]*\))?:\s*([0-9]+(?:\.[0-9]+)?)\s*dB",
        r"Essentia RMS-based crest factor proxy:\s*([0-9]+(?:\.[0-9]+)?)\s*dB",
        r"dynamic-range proxy:\s*([0-9]+(?:\.[0-9]+)?)\s*dB",
    )
    for pat in patterns:
        m = re.search(pat, str(text), re.IGNORECASE)
        if m:
            try:
                return round(float(m.group(1)), 1)
            except ValueError:
                continue
    return None


def format_recommended_dynamics_block(
    crest_db=None,
    source_note="objective crest-factor proxy",
    lufs=None,
    lra=None,
    loudness_source=None,
):
    """First-class dynamics block parallel to RECOMMENDED TEMPO / KEY.

    Prefers true EBU R128 integrated loudness (LUFS) + loudness range (LRA)
    when available; keeps crest-factor as a secondary compression cue.
    Includes short interpretive labels so the writer does not reverse polarity
    (more-negative LUFS = quieter; low LRA = compressed, not 'breathing').
    """
    crest_label, crest_plain = (None, None)
    if crest_db is not None:
        crest_label, crest_plain = _dynamics_label_from_crest(crest_db)

    if lufs is None and crest_db is None:
        return ""

    bits = []
    interpret = []

    if lufs is not None:
        ll = _lufs_label(lufs)
        bits.append(f"integrated loudness ≈ {lufs:.1f} LUFS" + (f" ({ll})" if ll else ""))
        interpret.append(
            f"LUFS polarity: more negative is quieter; {lufs:.1f} LUFS is "
            f"{ll or 'see anchors'} — a LEVEL, not a compression measurement; "
            "do not call this LUFS figure itself 'compressed' or 'punchy'."
        )
    if lra is not None:
        rl = _lra_label(lra)
        bits.append(f"loudness range (LRA) ≈ {lra:.1f} LU" + (f" ({rl})" if rl else ""))
        interpret.append(
            f"LRA is dynamic swing, not overall level; {lra:.1f} LU means "
            f"{rl or 'see anchors'}."
        )
    if crest_db is not None:
        if crest_label:
            bits.append(f"crest-factor proxy ≈ {crest_db} dB ({crest_label})")
        else:
            bits.append(f"crest-factor proxy ≈ {crest_db} dB")

    src_bits = []
    if loudness_source:
        src_bits.append(loudness_source)
    if source_note and crest_db is not None:
        src_bits.append(source_note)
    src_txt = "; ".join(src_bits) if src_bits else "waveform measurement"

    body = (
        f"\n\nRECOMMENDED DYNAMICS FOR DISCUSSION: {'; '.join(bits)} "
        f"[{src_txt}]. "
    )
    if interpret:
        body += " ".join(interpret) + " "
    if crest_plain and lufs is None:
        body += f"{crest_plain}. "
    body += (
        "Use these figures for loudness/compression/dynamic-range questions. "
        "Do not invent LUFS/LRA. Do not call a low LRA 'breathing'. "
        "Do not call a loud (e.g. −10 LUFS) master quiet. "
        "LUFS bands describe level only — do not say streaming-normalized / "
        "mastered for Spotify / loudness-war / brickwalled from LUFS alone; "
        "require low crest and/or very tight LRA before claiming heavy limiting, "
        "and do not confuse arrangement density with brickwall mastering."
    )
    return body


def measure_crest_factor_db(local_path):
    """Compute crest-factor (peak/RMS) in dB from a local audio path.

    Returns float or None. Independent of the full objective report so dynamics
    can still be elevated even if other DSP sections fail.

    Loads the file as mono for a stable peak/RMS ratio (stereo peak-vs-sum
    mixes are less consistent). Does NOT apply ReplayGain or any gain metadata —
    raw decoded PCM only.
    """
    if not local_path or str(local_path).startswith(("http://", "https://")):
        return None
    if not os.path.exists(local_path):
        return None
    try:
        y, sr = librosa.load(local_path, sr=None, mono=True)
        if y is None or len(y) < int(getattr(sr, "real", sr) or 1) * 0.5:
            return None
        peak = float(np.max(np.abs(y)))
        rms = float(np.sqrt(np.mean(np.square(y.astype(np.float64)))))
        if peak <= 1e-12 or rms <= 1e-12:
            return None
        return round(float(20.0 * np.log10(peak / rms)), 1)
    except Exception:
        return None


def _ffmpeg_ebur128_loudness(local_path):
    """Run ffmpeg ebur128 on the FULL file (stereo preserved). Returns
    (lufs, lra, true_peak_db, source) or (None, None, None, None).

    This is the preferred path: native sample rate, all channels, no
    premature mono downmix, no peak renormalization, ignores ReplayGain
    tags (ffmpeg does not apply them unless volume/replaygain filters are set).
    """
    if not local_path or not os.path.exists(local_path):
        return None, None, None, None
    try:
        # framelog=verbose is not required for summary lines; keep stderr modest.
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-nostats",
            "-i", local_path,
            "-filter_complex", "ebur128=peak=true",
            "-f", "null",
            "-",
        ]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=180,
        )
        # Summary is written to stderr even on success (returncode 0).
        log = (result.stderr or "") + "\n" + (result.stdout or "")
        # Typical summary block:
        #   I:         -9.8 LUFS
        #   LRA:        5.2 LU
        #   Peak:      -0.3 dBFS
        # or "True peak:" depending on ffmpeg version.
        lufs = None
        lra = None
        tp = None
        m = re.search(
            r"(?im)^\s*I:\s*([+-]?\d+(?:\.\d+)?)\s*LUFS\b",
            log,
        )
        if m:
            lufs = float(m.group(1))
        m = re.search(
            r"(?im)^\s*LRA:\s*([+-]?\d+(?:\.\d+)?)\s*LU\b",
            log,
        )
        if m:
            lra = float(m.group(1))
        m = re.search(
            r"(?im)^\s*(?:True\s+peak|Peak):\s*([+-]?\d+(?:\.\d+)?)\s*dB",
            log,
        )
        if m:
            tp = float(m.group(1))
        if lufs is not None and np.isfinite(lufs):
            return (
                round(float(lufs), 1),
                (round(float(lra), 1) if lra is not None and np.isfinite(lra) else None),
                (round(float(tp), 1) if tp is not None and np.isfinite(tp) else None),
                "ffmpeg ebur128 (EBU R128, full file, native channels)",
            )
    except Exception:
        pass
    return None, None, None, None


def measure_ebur128_loudness(local_path):
    """Measure EBU R128 integrated loudness (LUFS) and loudness range (LRA).

    Returns (lufs, lra, source_note) or (None, None, None).

    Tries, in order:
      1. ffmpeg ebur128 on the original/full file (stereo/native rate — preferred)
      2. pyloudnorm on STEREO (no peak renormalization)
      3. Essentia LoudnessEBUR128 when present

    Important robustness notes:
    - Measures the ENTIRE file (no intentional truncation for loudness).
    - Does NOT apply ReplayGain / RGAD / volume tags — those only affect
      players that choose to honour them; raw PCM decode is used here.
    - Avoids mono-average-then-measure, which systematically under-reads
      integrated loudness by ~3 LU versus dual-mono/stereo playback and
      can under-read further when stereo content partially cancels.
    - Does NOT peak-normalize the waveform before metering (that would
      change absolute integrated loudness).
    Crest-factor remains available separately via measure_crest_factor_db.
    """
    if not local_path or str(local_path).startswith(("http://", "https://")):
        return None, None, None
    if not os.path.exists(local_path):
        return None, None, None

    # --- 1. ffmpeg ebur128 (preferred: full file, native channels/rate) ---
    lufs, lra, _tp, src = _ffmpeg_ebur128_loudness(local_path)
    if lufs is not None:
        return lufs, lra, src

    # --- 2. pyloudnorm on STEREO (no peak renormalization) ---------------
    try:
        import pyloudnorm as pyln
        # Keep stereo when the file is stereo — critical for correct LUFS.
        y, sr = librosa.load(local_path, sr=None, mono=False)
        if y is None:
            raise ValueError("empty audio")
        y = np.asarray(y, dtype=np.float64)
        if y.ndim == 1:
            # Mono file — keep as (n_samples,)
            if len(y) < int(sr) * 0.5:
                raise ValueError("audio too short")
            audio_for_meter = y
        else:
            # librosa returns (n_channels, n_samples); pyloudnorm wants (n_samples, n_channels)
            if y.shape[0] < y.shape[1]:
                y = y.T
            if y.shape[0] < int(sr) * 0.5:
                raise ValueError("audio too short")
            audio_for_meter = y
        # Do NOT peak-normalize: absolute level is the quantity being measured.
        meter = pyln.Meter(int(sr))  # BS.1770 meter
        lufs = float(meter.integrated_loudness(audio_for_meter))
        lra = None
        try:
            if hasattr(meter, "loudness_range"):
                lra = float(meter.loudness_range(audio_for_meter))
        except Exception:
            lra = None
        if not np.isfinite(lufs):
            raise ValueError("non-finite LUFS")
        src = "pyloudnorm EBU R128 (BS.1770, stereo-preserving)"
        return (
            round(lufs, 1),
            (round(lra, 1) if lra is not None and np.isfinite(lra) else None),
            src,
        )
    except Exception:
        pass

    # --- 3. Essentia LoudnessEBUR128 (when the kernel exists) --------------
    if ESSENTIA_AVAILABLE and essentia is not None:
        try:
            LoudnessEBUR128 = _essentia_optional_kernel("LoudnessEBUR128")
            if LoudnessEBUR128 is not None:
                samples, sample_rate = _essentia_load_audio(local_path, ESSENTIA_MAX_SECONDS)
                if samples is not None and len(samples) > 0:
                    try:
                        vec = _essentia_as_vector(samples)
                        kernel = LoudnessEBUR128(sampleRate=int(sample_rate))
                        out = kernel(vec)
                    except Exception:
                        stereo = np.vstack([samples, samples]).T
                        kernel = LoudnessEBUR128(sampleRate=int(sample_rate))
                        out = kernel(stereo)

                    lufs = None
                    lra = None
                    if isinstance(out, (tuple, list)):
                        if len(out) >= 1:
                            lufs = _essentia_first_float(out[0])
                        if len(out) >= 2:
                            lra = _essentia_first_float(out[1])
                    else:
                        lufs = _essentia_first_float(out)

                    if lufs is not None and np.isfinite(lufs):
                        src = "Essentia LoudnessEBUR128 (EBU R128)"
                        return (
                            round(float(lufs), 1),
                            (round(float(lra), 1) if lra is not None and np.isfinite(lra) else None),
                            src,
                        )
        except Exception:
            pass

    return None, None, None



def build_objective_audio_report(local_path):
    try:
        y, sr = librosa.load(local_path, sr=None, mono=True)
        duration = len(y) / float(sr)
        if duration < 1.0:
            return ""

        lines = []
        lines.append("OBJECTIVE AUDIO MEASUREMENTS")
        lines.append(f"duration: {round(duration, 2)} s")

        peak = float(np.max(np.abs(y))) if len(y) else 0.0
        rms = float(np.sqrt(np.mean(np.square(y.astype(np.float64))))) if len(y) else 0.0
        lines.append("")
        lines.append("DYNAMIC RANGE / LOUDNESS (numeric)")
        try:
            _lufs, _lra, _lsrc = measure_ebur128_loudness(local_path)
            if _lufs is not None:
                lines.append(f"EBU R128 integrated loudness: {_lufs} LUFS ({_lsrc})")
                if _lra is not None:
                    lines.append(f"EBU R128 loudness range (LRA): {_lra} LU")
        except Exception:
            pass
        if peak > 1e-12 and rms > 1e-12:
            crest_db = round(float(20 * np.log10(peak / rms)), 1)
            label, plain = _dynamics_label_from_crest(crest_db)
            lines.append(f"crest factor / dynamic-range proxy: {crest_db} dB")
            lines.append(f"peak amplitude (normalized): {peak:.4f}")
            lines.append(f"RMS level (normalized): {rms:.6f}")
            if label:
                lines.append(f"dynamics band: {label}")
            if plain:
                lines.append(f"dynamic-range note: {plain}")
            if crest_db < 10.0:
                lines.append(
                    "compression character: low crest factor is consistent with heavy "
                    "loudness-war-style limiting, common from the 1990s onward (especially 2000s–2020s)."
                )
            elif crest_db > 14.0:
                lines.append(
                    "compression character: high crest factor is consistent with a more dynamic, "
                    "less-limited master (often pre-1990s, audiophile, or lightly mastered)."
                )
            else:
                lines.append(
                    "compression character: moderate crest factor — do not infer release era from this alone."
                )
        else:
            lines.append("")
            lines.append("DYNAMIC RANGE / LOUDNESS (numeric)")
            lines.append("crest factor / dynamic-range proxy: unavailable (silent or near-silent audio)")

        # start_bpm is a *prior*, not just an initial guess: librosa weights
        # tempo candidates by closeness to it (controlled by std_bpm, which
        # defaults to a tight ~1-octave spread). With the old defaults this
        # pulled real readings toward 120 BPM even when the track wasn't
        # actually near 120 (observed: a 125-126 BPM track reported as
        # ~123 BPM). Widening std_bpm keeps the prior's intended job --
        # nudging genuinely ambiguous half/double-time picks toward a
        # plausible song tempo -- without dragging correct readings toward
        # the prior's center.
        tempo_raw, beats = librosa.beat.beat_track(y=y, sr=sr, start_bpm=120.0, std_bpm=6.0)
        tempo_arr = np.atleast_1d(tempo_raw)
        tempo = float(tempo_arr[0]) if len(tempo_arr) else None
        beat_times = librosa.frames_to_time(beats, sr=sr)

        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)

        # Compute the median inter-beat-interval BPM BEFORE calling
        # _preferred_tempo, so it can be used as an independent corroborating
        # measurement (a different calculation -- from actual detected beat
        # timestamps -- rather than the raw autocorrelation/tempogram value)
        # instead of letting the half-time heuristic override a genuinely
        # fast, well-supported tempo.
        median_ibi_bpm = None
        if len(beat_times) > 1:
            ibis_for_corroboration = np.diff(beat_times)
            if len(ibis_for_corroboration):
                med = float(np.median(ibis_for_corroboration))
                if med > 0:
                    median_ibi_bpm = round(60.0 / med, 1)

        lines.append("")
        lines.append("BEAT / TEMPO MEASUREMENTS")
        if tempo is not None:
            preferred, cands, note = _preferred_tempo(tempo, corroborating_bpm=median_ibi_bpm)
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
    # IQR fence on log-Hz (octave-stable). Wide multiplier (3x, not the
    # textbook 1.5x): pitch-tracking frames are heavily weighted toward
    # sustained/held notes, so a tight fence would treat genuine melodic
    # excursions (brief high notes in a chorus, etc.) as statistical
    # outliers and crop them out, artificially narrowing the reported
    # range down to just the most common tessitura. The looser fence still
    # removes genuine octave-jump/tracking errors, which sit much further
    # out than any real single-voice excursion.
    logf = np.log2(np.clip(arr, 40.0, 2000.0))
    q1, q3 = np.percentile(logf, [25, 75])
    iqr = max(q3 - q1, 1e-6)
    keep = (logf >= q1 - 3.0 * iqr) & (logf <= q3 + 3.0 * iqr)
    cleaned = arr[keep]
    if cleaned.size < 5:
        cleaned = arr
    # Second pass: drop points more than an octave from the median
    med = float(np.median(cleaned))
    if med > 0:
        ratio = cleaned / med
        cleaned = cleaned[(ratio >= 0.5) & (ratio <= 2.0)]
    return cleaned if cleaned.size else arr



def _f0_multi_peak_note(f0_arr):
    """Return a short note if voiced f0 looks multi-modal (possible multi-singer
    or wide dual-register singing). Supporting evidence only — octave doubles
    and wide single-voice ranges can also look multi-modal."""
    try:
        arr = np.asarray(f0_arr, dtype=float)
        arr = arr[np.isfinite(arr) & (arr > 50) & (arr < 1200)]
        if arr.size < 40:
            return ""
        # Work in semitone space relative to median for stable binning.
        med = float(np.median(arr))
        if med <= 0:
            return ""
        semis = 12.0 * np.log2(arr / med)
        # Histogram over ~±18 semitones
        bins = np.linspace(-18, 18, 37)
        hist, edges = np.histogram(semis, bins=bins)
        if hist.sum() < 40:
            return ""
        # Smooth slightly
        kernel = np.array([1, 2, 3, 2, 1], dtype=float)
        kernel /= kernel.sum()
        smooth = np.convolve(hist.astype(float), kernel, mode="same")
        # Peak detection: local maxima above a floor
        peaks = []
        for i in range(1, len(smooth) - 1):
            if smooth[i] >= smooth[i - 1] and smooth[i] >= smooth[i + 1] and smooth[i] > 0:
                peaks.append((smooth[i], 0.5 * (edges[i] + edges[i + 1]), i))
        if len(peaks) < 2:
            return ""
        peaks.sort(reverse=True)
        # Keep strongest peaks that are well separated
        chosen = [peaks[0]]
        for p in peaks[1:]:
            if all(abs(p[1] - c[1]) >= 3.0 for c in chosen):
                chosen.append(p)
            if len(chosen) >= 3:
                break
        if len(chosen) < 2:
            return ""
        # Convert peak offsets back to approx Hz
        peak_hz = sorted(med * (2 ** (c[1] / 12.0)) for c in chosen[:3])
        peak_str = ", ".join(f"{hz:.0f} Hz" for hz in peak_hz)
        sep = abs(chosen[0][1] - chosen[1][1])
        strength = chosen[1][0] / max(chosen[0][0], 1e-9)
        if strength < 0.25:
            return ""
        return (
            f"voiced-f0 multi-peak hint: ~{len(chosen)} pitch centres near {peak_str} "
            f"(~{sep:.1f} semitones between top two; secondary/primary mass ratio {strength:.2f}). "
            "Possible multi-singer, call-and-response, or wide dual-register single singer / "
            "octave stacking — not proof of multiple people by itself."
        )
    except Exception:
        return ""


def build_vocal_objective_report(local_path):
    try:
        y, sr = librosa.load(local_path, sr=None, mono=True)
        duration = len(y) / float(sr)
        if duration < 2.0:
            return ""

        max_scan_seconds = min(duration, 120.0)
        chunk_seconds = 30.0
        f0_values = []

        # Sample windows SPREAD ACROSS the track rather than scanning
        # contiguously from t=0. A single 30s chunk alone typically yields
        # well over 1000 voiced pyin frames, so a "stop once we have enough
        # frames" exit (the old approach) almost always triggered after just
        # the first chunk — i.e. intro/verse 1 only — and never reached the
        # chorus or bridge, where the real vocal range usually opens up. That
        # produced artificially narrow reported ranges (e.g. a semitone or
        # two) even for tracks with normal melodic range. Spreading a fixed
        # number of chunks evenly across the full duration keeps the same
        # total-audio budget (~max_scan_seconds) but gives verse+chorus (and
        # bridge, for longer tracks) a real chance to be represented.
        n_chunks = max(1, int(round(max_scan_seconds / chunk_seconds)))
        if duration <= max_scan_seconds or n_chunks <= 1:
            chunk_starts = [i * chunk_seconds for i in range(n_chunks)]
        else:
            usable_span = max(0.0, duration - chunk_seconds)
            chunk_starts = [usable_span * i / (n_chunks - 1) for i in range(n_chunks)]

        for start_sec in chunk_starts:
            start = int(max(0.0, start_sec) * sr)
            end = min(start + int(chunk_seconds * sr), len(y))
            y_chunk = y[start:end]
            if len(y_chunk) < sr:
                continue

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

            # Weak multi-voice proxy: multiple well-separated pitch centres in
            # voiced f0 (can also be wide single-singer range or octave doubles —
            # never proof of two people by itself).
            multi_note = _f0_multi_peak_note(f0_arr)
            if multi_note:
                lines.append(multi_note)

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


def _essentia_as_vector(samples):
    """Ensure float32 mono vector; wrap as essentia.array when available
    (some algorithms reject plain numpy arrays)."""
    arr = np.asarray(samples, dtype=np.float32).reshape(-1)
    try:
        return essentia.array(arr)
    except Exception:
        return arr


def _essentia_resolve_kernel(name):
    """Resolve an essentia.standard algorithm by name, even if the module-level
    optional binding failed at import time (e.g. delayed import issues)."""
    existing = globals().get(name)
    if existing is not None and callable(existing):
        return existing
    if not ESSENTIA_AVAILABLE or essentia is None:
        return None
    try:
        return getattr(essentia.standard, name)
    except Exception:
        return None


def _essentia_tempo_and_beats(samples, sample_rate):
    """Try several Essentia rhythm/BPM algorithms. Builds differ in which
    constructors and return layouts they support; failures used to surface as
    a silent 'tempo: unavailable' even when librosa had a solid estimate."""
    vec = _essentia_as_vector(samples)
    duration = len(samples) / float(sample_rate) if sample_rate else 0.0
    last_err = None

    attempts = []

    Rhythm2013 = _essentia_resolve_kernel("RhythmExtractor2013")
    if Rhythm2013 is not None:
        for method in ("multifeature", "degara", None):
            attempts.append(("RhythmExtractor2013", Rhythm2013, method))

    Rhythm = _essentia_resolve_kernel("RhythmExtractor")
    if Rhythm is not None:
        attempts.append(("RhythmExtractor", Rhythm, None))

    Percival = _essentia_resolve_kernel("PercivalBpmEstimator")
    if Percival is not None:
        attempts.append(("PercivalBpmEstimator", Percival, None))

    for name, cls, method in attempts:
        try:
            if method is not None:
                try:
                    kernel = cls(method=method)
                except TypeError:
                    kernel = cls()
            else:
                kernel = cls()
            _essentia_set_sample_rate(kernel, sample_rate)
            out = kernel(vec)

            tempo = None
            beats = np.array([], dtype=float)

            if isinstance(out, (tuple, list)):
                tempo = _essentia_first_float(out[0]) if out else None
                # RhythmExtractor2013: bpm, beats, confidence, estimates, intervals
                if len(out) >= 2:
                    beats = _essentia_to_float_array(out[1])
                    # If out[1] is a scalar confidence, beats may be empty —
                    # try later slots that look like a beat array.
                    if beats.size <= 1 and len(out) >= 2:
                        for slot in out[1:]:
                            cand = _essentia_to_float_array(slot)
                            if cand.size > 4:
                                beats = cand
                                break
            else:
                tempo = _essentia_first_float(out)

            if tempo is None or not np.isfinite(tempo) or tempo <= 0:
                continue

            if beats.size and duration > 0 and np.max(beats) > max(duration * 2.0, 10.0):
                # Frame indices rather than seconds
                beats = beats / float(sample_rate)

            beats = beats[np.isfinite(beats)] if beats.size else np.array([], dtype=float)
            return float(tempo), beats
        except Exception as e:
            last_err = e
            continue

    return None, np.array([], dtype=float)


def _key_profile_kwargs_for_genre(genre_hint=None):
    """Ordered KeyExtractor profileType kwargs lists by coarse genre family.

    Electronic/dance → prefer edma (and related) first.
    Rock/folk/classical/ballad → prefer temperley / krumhansl-style first.
    Always fall back through the full set so a weak preferred profile cannot
    silence a much stronger alternate reading.
    """
    fam = "other"
    try:
        fam = _bpm_genre_family_from_hint(genre_hint) if genre_hint else "other"
        # Map hiphop/dnb toward electronic-ish profile preference
        if fam in ("dnb", "hiphop"):
            fam = "electronicish"
        if fam == "ballad":
            fam = "rockish"
    except Exception:
        fam = "other"

    electronic_first = [
        {"profileType": "edma"},
        {"profileType": "bgate"},
        {"profileType": "temperley"},
        {"profileType": "krumhansl"},
        {},
    ]
    tonal_first = [
        {"profileType": "temperley"},
        {"profileType": "krumhansl"},
        {"profileType": "bgate"},
        {"profileType": "edma"},
        {},
    ]
    balanced = [
        {"profileType": "bgate"},
        {"profileType": "temperley"},
        {"profileType": "edma"},
        {"profileType": "krumhansl"},
        {},
    ]
    if fam == "electronicish":
        return electronic_first, "electronic/dance (edma-first)"
    if fam in ("rockish", "popish"):
        return tonal_first, f"{fam} (temperley/krumhansl-first)"
    return balanced, "balanced"


def _parse_key_extractor_output(out):
    """Normalize KeyExtractor / Key output to (key_name_with_mode, strength_or_None)."""
    key_name = None
    scale = None
    strength = None

    if isinstance(out, (tuple, list)):
        if len(out) >= 3:
            key_name, scale, strength = out[0], out[1], out[2]
        elif len(out) == 2:
            key_name, second = out[0], out[1]
            if isinstance(second, (str, bytes)) or (
                second is not None and not isinstance(second, (int, float, np.floating))
                and _essentia_first_float(second) is None
            ):
                scale = second
            else:
                strength = second
        elif len(out) == 1:
            key_name = out[0]
    else:
        key_name = out

    if isinstance(key_name, bytes):
        key_name = key_name.decode("utf-8", "ignore")
    if isinstance(scale, bytes):
        scale = scale.decode("utf-8", "ignore")

    key_name = str(key_name).strip() if key_name is not None else ""
    scale = str(scale).strip().lower() if scale is not None else ""

    if not key_name or key_name.lower() in ("", "none", "unknown", "nan"):
        return None, None

    if scale in ("major", "minor", "maj", "min"):
        scale_norm = "major" if scale.startswith("maj") else "minor"
        key_name = f"{key_name} {scale_norm}"

    return key_name, _essentia_first_float(strength)


def _energy_key_windows(samples, sample_rate, window_sec=25.0, max_windows=4):
    """Pick up to max_windows high-energy segments for key voting.

    Avoids estimating key only on a quiet intro / fade. Falls back to a single
    full-clip window when the file is short or energy is flat.
    """
    try:
        n = len(samples)
        sr = float(sample_rate)
        if n < int(sr * 3):
            return [(0, n)]
        win = int(window_sec * sr)
        if win <= 0 or n <= win:
            return [(0, n)]
        # Hop so we scan the file; score by mean square energy
        hop = max(win // 2, int(5.0 * sr))
        scored = []
        start = 0
        while start + win <= n:
            seg = samples[start:start + win]
            eng = float(np.mean(np.square(seg.astype(np.float64))))
            scored.append((eng, start, start + win))
            start += hop
        if not scored:
            return [(0, n)]
        scored.sort(key=lambda t: t[0], reverse=True)
        # Keep top windows, in time order, de-duplicating heavy overlap
        picked = []
        for eng, a, b in scored:
            if any(abs(a - pa) < win * 0.4 for pa, pb in picked):
                continue
            picked.append((a, b))
            if len(picked) >= max_windows:
                break
        picked.sort(key=lambda ab: ab[0])
        return picked or [(0, n)]
    except Exception:
        return [(0, len(samples))]


def _essentia_key_on_vector(vec, sample_rate, ctor_kwargs_list):
    """Run KeyExtractor over ctor_kwargs_list; return (key, strength, profile_note)."""
    KeyCls = _essentia_resolve_kernel("KeyExtractor")
    if KeyCls is None:
        return None, None, ""

    best_key, best_strength, best_prof = None, None, ""
    for kwargs in ctor_kwargs_list:
        try:
            try:
                key_kernel = KeyCls(**kwargs) if kwargs else KeyCls()
            except TypeError:
                key_kernel = KeyCls()
            _essentia_set_sample_rate(key_kernel, sample_rate)
            out = key_kernel(vec)
            key_name, strength_val = _parse_key_extractor_output(out)
            if key_name is None:
                continue
            prof = kwargs.get("profileType", "default") if kwargs else "default"
            if strength_val is None:
                if best_key is None:
                    best_key, best_strength, best_prof = key_name, None, prof
                continue
            # Small bonus for preferred (earlier) profiles when strengths are close
            # is applied by caller via vote aggregation; here pure strength wins.
            if best_strength is None or strength_val > best_strength:
                best_key, best_strength, best_prof = key_name, strength_val, prof
        except Exception:
            continue
    return best_key, best_strength, best_prof


def _essentia_key_hpcp_fallback(samples, sample_rate):
    """Lower-level HPCP → Key path with higher resolution and harmonic weighting.

    Mirrors the Deep Cuts-style idea: denser chroma (pcpSize=36), harmonic
    contribution, non-linear HPCP weighting, then key profile matching.
    Used only when KeyExtractor is missing or returns nothing useful.
    """
    try:
        FrameCutter = _essentia_resolve_kernel("FrameCutter")
        Windowing = _essentia_resolve_kernel("Windowing")
        Spectrum = _essentia_resolve_kernel("Spectrum")
        SpectralPeaks = _essentia_resolve_kernel("SpectralPeaks")
        HPCP = _essentia_resolve_kernel("HPCP")
        Key = _essentia_resolve_kernel("Key")
        if not all([FrameCutter, Windowing, Spectrum, SpectralPeaks, HPCP, Key]):
            return None, None

        # Limit runtime on long files
        max_sec = min(len(samples) / float(sample_rate), 120.0)
        seg = samples[: int(max_sec * sample_rate)]
        frame_size = 4096
        hop = 2048

        try:
            fc = FrameCutter(frameSize=frame_size, hopSize=hop)
        except TypeError:
            fc = FrameCutter()
        try:
            win = Windowing(type="blackmanharris62")
        except TypeError:
            try:
                win = Windowing(type="hann")
            except TypeError:
                win = Windowing()
        spec = Spectrum()
        try:
            peaks = SpectralPeaks(
                sampleRate=float(sample_rate),
                maxPeaks=60,
                magnitudeThreshold=0.00001,
                orderBy="magnitude",
            )
        except TypeError:
            peaks = SpectralPeaks()
        try:
            hpcp = HPCP(
                sampleRate=float(sample_rate),
                size=36,           # higher-resolution pitch class profile
                referenceFrequency=440.0,
                harmonics=4,       # harmonic contribution
                nonLinear=True,    # non-linear weighting
                bandPreset=True,
            )
        except TypeError:
            try:
                hpcp = HPCP(size=36, nonLinear=True, harmonics=4)
            except TypeError:
                try:
                    hpcp = HPCP(size=36)
                except TypeError:
                    hpcp = HPCP()

        # Average HPCP over frames
        hpcp_sum = None
        n_frames = 0
        # Feed samples as essentia vector; FrameCutter may need sequential calls
        try:
            from essentia import array as esarr
            buf = esarr(seg.astype(np.float32))
        except Exception:
            buf = _essentia_as_vector(seg)

        # Some builds: FrameCutter(audio) returns frames iteratively
        try:
            fc.reset()
        except Exception:
            pass
        pos = 0
        while pos + frame_size <= len(seg) and n_frames < 400:
            frame = seg[pos:pos + frame_size]
            pos += hop
            try:
                try:
                    fr = win(_essentia_as_vector(frame))
                except Exception:
                    fr = frame
                sp = spec(fr) if not isinstance(fr, tuple) else spec(fr[0] if fr else frame)
                pk = peaks(sp)
                # SpectralPeaks often returns (frequencies, magnitudes)
                if isinstance(pk, (tuple, list)) and len(pk) >= 2:
                    freqs, mags = pk[0], pk[1]
                else:
                    continue
                hp = hpcp(freqs, mags)
                hp = np.asarray(hp, dtype=np.float64).reshape(-1)
                if hpcp_sum is None:
                    hpcp_sum = np.zeros_like(hp)
                if hpcp_sum.shape != hp.shape:
                    continue
                hpcp_sum += hp
                n_frames += 1
            except Exception:
                continue

        if hpcp_sum is None or n_frames < 4:
            return None, None
        hpcp_mean = hpcp_sum / float(n_frames)

        # Key profile match — try a few profile names
        best_key, best_strength = None, None
        for prof in ("temperley", "krumhansl", "edma", "bgate", None):
            try:
                if prof:
                    key_algo = Key(profileType=prof)
                else:
                    key_algo = Key()
            except TypeError:
                try:
                    key_algo = Key()
                except Exception:
                    continue
            except Exception:
                continue
            try:
                out = key_algo(hpcp_mean)
            except Exception:
                try:
                    out = key_algo(_essentia_as_vector(hpcp_mean))
                except Exception:
                    continue
            key_name, strength_val = _parse_key_extractor_output(out)
            if key_name is None:
                continue
            if strength_val is None:
                if best_key is None:
                    best_key, best_strength = key_name, None
                continue
            if best_strength is None or strength_val > best_strength:
                best_key, best_strength = key_name, strength_val
        return best_key, best_strength
    except Exception:
        return None, None


def _essentia_key(samples, sample_rate, genre_hint=None):
    """Estimate musical key via Essentia.

    Pipeline (Deep-Cuts-inspired, still pure Essentia):
      1. Genre-ordered KeyExtractor profiles (edma-first for electronic, etc.)
      2. Vote across a few high-energy windows rather than one global pass
      3. Fall back to higher-resolution HPCP (pcpSize=36, harmonics, nonLinear)
         → Key if KeyExtractor yields nothing useful

    Returns (key_name, strength) as before so callers stay compatible.
    """
    if samples is None or len(samples) == 0:
        return None, None

    ctor_kwargs_list, prof_note = _key_profile_kwargs_for_genre(genre_hint)
    windows = _energy_key_windows(samples, sample_rate)

    # Aggregate votes: key_norm -> (total_strength, count, display_name)
    votes = {}
    any_result = False
    for a, b in windows:
        seg = samples[a:b]
        if len(seg) < int(float(sample_rate) * 2):
            continue
        vec = _essentia_as_vector(seg)
        key_name, strength_val, _prof = _essentia_key_on_vector(
            vec, sample_rate, ctor_kwargs_list
        )
        if key_name is None:
            continue
        any_result = True
        norm = _normalize_key_for_compare(key_name) or key_name.lower()
        # Prefer profiles: slight weight already via strength; count ties
        w = float(strength_val) if strength_val is not None else 0.15
        prev = votes.get(norm)
        if prev is None:
            votes[norm] = [w, 1, key_name]
        else:
            prev[0] += w
            prev[1] += 1

    best_key, best_strength = None, None
    if votes:
        # Rank by (vote count, total strength)
        ranked = sorted(votes.items(), key=lambda kv: (kv[1][1], kv[1][0]), reverse=True)
        _norm, (tot_w, cnt, display) = ranked[0]
        best_key = display
        # Report mean strength when available
        best_strength = round(tot_w / max(cnt, 1), 3) if tot_w > 0 else None

    if best_key is None:
        # Single full-clip KeyExtractor pass as last KeyExtractor try
        vec = _essentia_as_vector(samples)
        best_key, best_strength, _ = _essentia_key_on_vector(
            vec, sample_rate, ctor_kwargs_list
        )

    if best_key is None:
        best_key, best_strength = _essentia_key_hpcp_fallback(samples, sample_rate)

    return best_key, best_strength



def refine_essentia_key_with_genre(local_path, genre_hint, previous_key=None, previous_strength=None):
    """Re-run Essentia key with a genre profile prior once genre is known.

    Returns (key, strength, note) or (previous_key, previous_strength, "") if
    refinement is unavailable or does not improve confidence.
    """
    if not genre_hint or not local_path or str(local_path).startswith(("http://", "https://")):
        return previous_key, previous_strength, ""
    if not ENABLE_ESSENTIA_REPORT:
        return previous_key, previous_strength, ""
    try:
        samples, sample_rate = _essentia_load_audio(local_path, ESSENTIA_MAX_SECONDS)
        if samples is None or len(samples) == 0:
            return previous_key, previous_strength, ""
        key2, str2 = _essentia_key(samples, sample_rate, genre_hint=genre_hint)
        if not key2:
            return previous_key, previous_strength, ""
        prev_norm = _normalize_key_for_compare(previous_key) if previous_key else None
        new_norm = _normalize_key_for_compare(key2)
        if prev_norm and new_norm and prev_norm == new_norm:
            # Same key — optionally keep higher strength
            if str2 is not None and (previous_strength is None or str2 > previous_strength):
                return key2, str2, f"genre-conditioned Essentia reaffirmation (prior={genre_hint}, strength={str2})."
            return previous_key, previous_strength, ""
        # Different key: prefer new if stronger, or if previous was weak
        if previous_key is None:
            return key2, str2, f"genre-conditioned Essentia key (prior={genre_hint})."
        if str2 is not None and (previous_strength is None or str2 >= (previous_strength or 0) - 0.05):
            return (
                key2,
                str2,
                f"genre-conditioned Essentia key shifted {previous_key} → {key2} "
                f"(prior={genre_hint}, strength={str2}).",
            )
        return previous_key, previous_strength, (
            f"genre-conditioned Essentia suggested {key2} (strength={str2}) "
            f"but kept {previous_key} (stronger/earlier strength={previous_strength})."
        )
    except Exception:
        return previous_key, previous_strength, ""


def build_essentia_report(local_path, genre_hint=None):
    """
    Optional independent Essentia report.

    This is intentionally limited to objective audio measurements such as tempo/beat,
    key, spectral/timbre proxies, and dynamics. It should not be used by the writer
    to infer genre or vocal identity.

    genre_hint (optional): free-form genre label used to order KeyExtractor
    profiles (edma-first for electronic/dance, temperley/krumhansl for rock/folk).
    """
    if not ENABLE_ESSENTIA_REPORT:
        return ""

    if local_path.startswith(("http://", "https://")):
        # Essentia is being used here on local files only, matching the existing DSP path.
        return ""

    samples, sample_rate = _essentia_load_audio(local_path, ESSENTIA_MAX_SECONDS)
    if samples is None or len(samples) == 0:
        return (
            "ESSENTIA OBJECTIVE MEASUREMENTS unavailable "
            "(could not load audio via MonoLoader/AudioLoader — "
            "check that Essentia can read this WAV path)."
        )

    lines = []
    duration = len(samples) / float(sample_rate)
    lines.append(f"ESSENTIA OBJECTIVE MEASUREMENTS (first {round(duration, 2)} s)")

    # Tempo and beat timing.
    tempo, beats = _essentia_tempo_and_beats(samples, sample_rate)

    # Compute median-IBI BPM first so it can corroborate the raw reading
    # before the half-time heuristic runs (see _preferred_tempo docstring --
    # this is the exact case that previously forced a genuine ~156 BPM
    # reading down to ~78 BPM despite both figures agreeing).
    essentia_median_ibi_bpm = None
    if beats.size > 1:
        ibis_for_corroboration = np.diff(beats)
        if ibis_for_corroboration.size:
            med = float(np.median(ibis_for_corroboration))
            if med > 0:
                essentia_median_ibi_bpm = round(60.0 / med, 1)

    if tempo is not None:
        preferred, cands, note = _preferred_tempo(tempo, corroborating_bpm=essentia_median_ibi_bpm)
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

    # Key estimation (genre-ordered profiles + energy-window vote + HPCP fallback).
    key_name, key_strength = _essentia_key(samples, sample_rate, genre_hint=genre_hint)
    if key_name:
        strength_text = f", strength={round(key_strength, 3)}" if key_strength is not None else ""
        hint_txt = f", profile prior={genre_hint}" if genre_hint else ""
        lines.append(f"Essentia estimated key: {key_name}{strength_text}{hint_txt}")
    else:
        lines.append("Essentia estimated key: unavailable")

    # Spectral/timbre proxies on a shorter segment to keep runtime reasonable.
    lowlevel_seconds = min(duration, ESSENTIA_LOWLEVEL_MAX_SECONDS)
    seg = samples[: int(lowlevel_seconds * sample_rate)]

    if len(seg) >= ESSENTIA_FRAME_SIZE * 2:
        spectral_lines = []
        seg_vec = _essentia_as_vector(seg)

        def _frame_factory(kernel_name):
            cls = _essentia_resolve_kernel(kernel_name)
            if cls is None:
                raise RuntimeError(f"{kernel_name} unavailable")
            return _essentia_make_frame_kernel(cls)

        centroid = _essentia_mean_feature(
            lambda: _frame_factory("SpectralCentroid"),
            seg_vec,
            sample_rate,
        )
        if centroid is not None:
            spectral_lines.append(f"Essentia brightness (spectral centroid): {round(centroid, 1)} Hz")

        flatness = _essentia_mean_feature(
            lambda: _frame_factory("SpectralFlatness"),
            seg_vec,
            sample_rate,
        )
        if flatness is not None:
            spectral_lines.append(f"Essentia noise-likeness (spectral flatness): {round(flatness, 4)}")

        zcr = _essentia_mean_feature(
            lambda: _frame_factory("ZeroCrossingRate"),
            seg_vec,
            sample_rate,
        )
        if zcr is not None:
            spectral_lines.append(f"Essentia zero-crossing rate: {round(zcr, 4)}")

        rms = _essentia_mean_feature(
            lambda: _frame_factory("RMS"),
            seg_vec,
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

    if "child" in category:
        if "male" in modifier and "female" not in modifier:
            return "child_male_likely"
        if "female" in modifier and "male" not in modifier:
            return "child_female_likely"
        return "child_gender_uncertain"

    if category in ("post_puberty_male", "adult_male", "post_pubertal_male") or (
        "post" in category and "pubert" in category and "male" in category and "female" not in category
    ):
        return "post_puberty_male"

    if any(token in category for token in ("adolescent", "transitional", "changing_voice", "voice_change")):
        return "adolescent_male_likely"

    if category in ("female_teen_adult", "adult_female", "young_female") or (
        "female" in category and "child" not in category
    ):
        return "female_teen_adult"

    if category in ("uncertain", "transitional"):
        if "adolescent" in modifier or "transitional" in modifier:
            return "adolescent_male_likely"
        if any(token in modifier for token in ("child", "young")):
            return "child_gender_uncertain"
        return "uncertain"

    return ""


def parse_vocal_tags(text):
    lead = ""
    backing = ""
    # Accept "=" or ":" — Music Flamingo often emits "LEAD_PROFILE: ..." instead of "=".
    m = re.search(r'LEAD_PROFILE\s*[=:]\s*["\']?([A-Za-z0-9_\- ]+)', text or "", re.IGNORECASE)
    if m:
        lead = _normalize_vocal_tag(m.group(1))

    if lead not in VOCAL_LEAD_TAGS:
        cat_m = re.search(r'LEAD_CATEGORY\s*[=:]\s*["\']?([A-Za-z0-9_\- ]+)', text or "", re.IGNORECASE)
        mod_m = re.search(r'GENDER_MODIFIER\s*[=:]\s*["\']?([A-Za-z0-9_\- ]+)', text or "", re.IGNORECASE)
        if cat_m and mod_m:
            lead = _lead_from_category_modifier(cat_m.group(1), mod_m.group(1))
        elif cat_m:
            lead = _lead_from_category_modifier(cat_m.group(1), "")

    lead = VOCAL_LEAD_ALIASES.get(lead, lead)
    if lead not in VOCAL_LEAD_TAGS:
        lead = "unknown"

    m = re.search(r'BACKING_PROFILES\s*[=:]\s*["\']?([A-Za-z0-9_\- ]+)', text or "", re.IGNORECASE)
    if m:
        backing = _normalize_vocal_tag(m.group(1))
    if backing not in VOCAL_BACKING_TAGS:
        backing = "uncertain"

    return lead, backing


def parse_multi_voice_fields(text):
    """Extract structured multi-singer fields from a vocal analysis pass.
    Returns a dict with keys that may be empty strings when unparsed.
    """
    text = text or ""
    out = {
        "num_distinct_voices": "",
        "voice_arrangement": "",
        "co_lead_detail": "",
        "multi_voice_evidence": "",
    }

    m = re.search(
        r'NUM_DISTINCT_VOICES\s*=\s*["\']?([0-9]+\+?|uncertain|one|two|three[+]?|1|2|3)',
        text,
        re.IGNORECASE,
    )
    if m:
        raw = m.group(1).strip().lower()
        alias = {"one": "1", "two": "2", "three": "3", "three+": "3+"}
        out["num_distinct_voices"] = alias.get(raw, raw)

    m = re.search(
        r'VOICE_ARRANGEMENT\s*=\s*["\']?([A-Za-z0-9_\-]+)',
        text,
        re.IGNORECASE,
    )
    if m:
        arr = m.group(1).strip().lower().replace("-", "_")
        allowed = {
            "solo_lead", "lead_plus_backing", "duet_co_leads",
            "call_response", "group_unison", "uncertain",
        }
        if arr in allowed:
            out["voice_arrangement"] = arr

    m = re.search(
        r'CO_LEAD_DETAIL\s*=\s*\[?([^\n\]]+)',
        text,
        re.IGNORECASE,
    )
    if m:
        detail = m.group(1).strip().strip("[]")
        if detail and detail.lower() not in ("none", "n/a", "na"):
            out["co_lead_detail"] = detail[:400]

    m = re.search(
        r'MULTI_VOICE_EVIDENCE\s*=\s*\[?([^\n\]]+)',
        text,
        re.IGNORECASE,
    )
    if m:
        ev = m.group(1).strip().strip("[]")
        if ev:
            out["multi_voice_evidence"] = ev[:500]

    # Consistency: if arrangement is clearly multi-lead, nudge num voices.
    if out["voice_arrangement"] in ("duet_co_leads", "call_response") and not out["num_distinct_voices"]:
        out["num_distinct_voices"] = "2"
    if out["voice_arrangement"] == "solo_lead" and not out["num_distinct_voices"]:
        out["num_distinct_voices"] = "1"

    return out


def format_multi_voice_audit(fields, lead_profile, backing_profiles):
    """Compact multi-singer block for the private track notes."""
    if not fields:
        return ""
    num = fields.get("num_distinct_voices") or "unparsed"
    arr = fields.get("voice_arrangement") or "unparsed"
    detail = fields.get("co_lead_detail") or "none"
    evid = fields.get("multi_voice_evidence") or "not stated"
    lines = [
        "MULTI-SINGER / VOICE ARRANGEMENT AUDIT:",
        f"- NUM_DISTINCT_VOICES: {num}",
        f"- VOICE_ARRANGEMENT: {arr}",
        f"- CO_LEAD_DETAIL: {detail}",
        f"- MULTI_VOICE_EVIDENCE: {evid}",
        f"- LEAD_PROFILE tag: {lead_profile or 'unknown'}; BACKING: {backing_profiles or 'uncertain'}",
        "Guidance: doubles/octave stacks/reverb ≠ separate people. "
        "Only treat as multiple lead singers when arrangement is duet_co_leads or "
        "call_response (or NUM_DISTINCT_VOICES≥2 with clear CO_LEAD_DETAIL). "
        "lead_plus_backing means one lead plus supporting voices, not mixed leads.",
    ]
    # Soft consistency fix hint for the writer
    if lead_profile == "mixed_leads" and arr in ("solo_lead", "lead_plus_backing"):
        lines.append(
            "- Note: LEAD_PROFILE=mixed_leads conflicts with a solo/lead+backing arrangement; "
            "prefer the arrangement fields and describe one primary lead unless CO_LEAD_DETAIL "
            "clearly names two people."
        )
    if lead_profile != "mixed_leads" and arr in ("duet_co_leads", "call_response"):
        lines.append(
            "- Note: arrangement suggests distinct co-leads even though LEAD_PROFILE is not "
            "mixed_leads — mention both voices when discussing who is singing."
        )
    return "\n".join(lines)


def parse_vocal_confirmation(text):
    lead = ""
    confidence = ""

    m = re.search(r'LEAD_PROFILE\s*[=:]\s*["\']?([A-Za-z0-9_\- ]+)', text or "", re.IGNORECASE)
    if m:
        lead = _normalize_vocal_tag(m.group(1))

    if lead not in VOCAL_LEAD_TAGS:
        cat_m = re.search(r'LEAD_CATEGORY\s*[=:]\s*["\']?([A-Za-z0-9_\- ]+)', text or "", re.IGNORECASE)
        mod_m = re.search(r'GENDER_MODIFIER\s*[=:]\s*["\']?([A-Za-z0-9_\- ]+)', text or "", re.IGNORECASE)
        if cat_m and mod_m:
            lead = _lead_from_category_modifier(cat_m.group(1), mod_m.group(1))
        elif cat_m:
            lead = _lead_from_category_modifier(cat_m.group(1), "")

    lead = VOCAL_LEAD_ALIASES.get(lead, lead)
    if lead not in VOCAL_LEAD_TAGS:
        lead = ""

    m = re.search(r"CONFIDENCE\s*[=:]\s*(low|medium|high)", text or "", re.IGNORECASE)
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


def _apply_vocal_age_guard(lead_profile, analysis_text=""):
    """
    Prevent ambiguous youthful male voices from being promoted to child labels,
    and avoid rounding a still-high / light / boyish male lead up to a full
    post_puberty_male when the acoustic text does not support settled adult
    chest resonance.

    A high/light youthful male voice is much more commonly adolescent or
    post-pubertal than a prepubertal child. Child labels require stronger
    evidence of juvenile vocal-tract cues. Conversely, early-career teen-pop
    male leads (still quite high in the mix) should stay adolescent_male_likely
    rather than being described as a settled post-puberty adult male.
    """
    text = (analysis_text or "").lower()

    adult_male_markers = (
        "post-pubert",
        "post puberty",
        "adult male",
        "male resonance",
        "male vocal weight",
        "baritone",
        "tenor",
        "chest voice",
        "deepened voice",
        "mature male",
        "adult-sounding male",
        "high adult male",
        "settled adult",
        "full adult",
    )

    youthful_high_markers = (
        "high pitch",
        "high register",
        "bright",
        "light",
        "thin",
        "youthful",
        "boyish",
        "androgynous",
        "falsetto",
        "head voice",
        "still changing",
        "adolescent",
        "teen",
        "young male",
    )

    settled_adult_markers = (
        "settled adult",
        "full adult",
        "mature male",
        "adult chest",
        "deep chest",
        "baritone",
        "heavy vocal weight",
        "full chest resonance",
        "settled chest",
    )

    # Child -> do not promote on high/light alone; only on clear adult-male markers.
    if lead_profile in (
        "child_male_likely",
        "child_female_likely",
        "child_gender_uncertain",
    ):
        if lead_profile == "child_male_likely" and any(
            marker in text for marker in adult_male_markers
        ):
            # If the same text is still full of youthful-high cues and lacks
            # settled-adult markers, land on adolescent rather than full post-puberty.
            if any(m in text for m in youthful_high_markers) and not any(
                m in text for m in settled_adult_markers
            ):
                return "adolescent_male_likely"
            return "post_puberty_male"
        return lead_profile

    # post_puberty_male with high/light/boyish text and no settled-adult markers
    # -> adolescent_male_likely (fixes early Justin Bieber / similar teen-pop leads).
    if lead_profile in ("post_puberty_male", "adult_male", "young_male"):
        has_youth = any(m in text for m in youthful_high_markers)
        has_settled = any(m in text for m in settled_adult_markers)
        if has_youth and not has_settled:
            return "adolescent_male_likely"

    return lead_profile


def _apply_f0_pitch_guard(lead_profile, median_f0, low_f0=None, high_f0=None):
    """Use objective voiced-pitch statistics as a *soft* constraint on lead tags.

    Rules of engagement:
      - Never invent child/prepubertal from high pitch alone.
      - Never force female from high pitch alone here (countertenor / falsetto-heavy
        male leads exist). Gender recovery from high f0 + cover/metadata is handled
        in singer-identity resolution when the acoustic tag is uncertain.
      - Veto confident male tags when the track-wide *median* is so high that a
        male modal centre (settled adult or adolescent) is acoustically implausible.
        Important: age-guard often demotes post_puberty_male → adolescent_male_likely
        on "high/light" prose; this guard must also demote adolescent_male_likely at
        extreme medians, otherwise female leads mislabeled male stay "adolescent male".

    median_f0 / low_f0 / high_f0 are Hz (or None if unavailable).
    """
    if lead_profile is None or lead_profile in ("", "unknown"):
        return lead_profile
    if median_f0 is None:
        return lead_profile

    try:
        med = float(median_f0)
    except (TypeError, ValueError):
        return lead_profile

    soft_cap = globals().get("F0_POST_PUBERTY_SOFT_CAP_HZ")
    hard_cap = globals().get("F0_POST_PUBERTY_HARD_CAP_HZ")
    male_any_cap = globals().get("F0_MALE_ANY_HARD_CAP_HZ")
    child_floor = globals().get("F0_CHILD_SOFT_FLOOR_HZ")

    male_like = lead_profile in (
        "post_puberty_male",
        "adult_male",
        "young_male",
        "adolescent_male_likely",
        "child_male_likely",
    )

    # Extreme track-wide median: no male-tagged lead (adult or adolescent) is safe.
    if male_like and male_any_cap is not None and med >= float(male_any_cap):
        return "uncertain"

    # Settled adult male with a high median (not just top notes) is rare.
    if lead_profile in ("post_puberty_male", "adult_male", "young_male"):
        if hard_cap is not None and med >= float(hard_cap):
            return "uncertain"
        if soft_cap is not None and med >= float(soft_cap):
            return "adolescent_male_likely"

    # Child labels need more than pitch; a low-to-mid median argues against
    # keeping a child tag that was probably pitch-driven.
    if lead_profile in (
        "child_male_likely",
        "child_female_likely",
        "child_gender_uncertain",
    ):
        if child_floor is not None and med < float(child_floor):
            return "uncertain"

    return lead_profile


def _confirm_child_flip_has_evidence(confirm_text):
    """
    Gate for flipping a MALE_LEAD_CATEGORIES / ADOLESCENT_MALE_CATEGORIES
    classification to a child_* category based on the confirmation pass.

    A bare CONFIDENCE=high tag from one isolated audio pass is not enough —
    that pass can hallucinate confidence just as easily as a category, and
    doing so previously let a single mistaken confirmation pass override a
    correctly-identified adult male voice with no corroborating evidence at
    all. Require the confirmation pass's own analysis text to actually state
    a prepubertal-type acoustic marker before the flip is accepted.
    """
    text = (confirm_text or "").lower()
    child_evidence_markers = (
        "prepubert",
        "pre-pubert",
        "child-like vocal tract",
        "childlike vocal tract",
        "child vocal tract",
        "child-like resonance",
        "childlike resonance",
        "child vocal weight",
        "small vocal tract",
        "unbroken voice",
        "boy soprano",
        "boy treble",
        "child resonance",
    )
    return any(marker in text for marker in child_evidence_markers)


def choose_final_vocal_lead(initial_lead, confirm_lead, confirm_confidence, confirm_text="", median_f0=None):
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

    # Confirmation may discover distinct co-leads the first pass missed.
    if confirm == "mixed_leads" and strong_confirm:
        return "mixed_leads"

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

    # Male → female from confirmation alone is normally blocked (young male leads are often
    # mislabeled female on pitch). Exception: when objective median f0 is extreme (≥ hard
    # male cap), a strong female confirmation is allowed — this is the female→male mislabel
    # recovery path for high-median leads.
    if initial in MALE_LEAD_CATEGORIES or initial in ADOLESCENT_MALE_CATEGORIES:
        try:
            _med = float(median_f0) if median_f0 is not None else None
        except (TypeError, ValueError):
            _med = None
        _male_cap = globals().get("F0_MALE_ANY_HARD_CAP_HZ") or globals().get("F0_POST_PUBERTY_HARD_CAP_HZ")
        _extreme_f0 = (
            _med is not None
            and _male_cap is not None
            and _med >= float(_male_cap)
        )
        if confirm == "female_teen_adult" and _extreme_f0 and strong_confirm:
            return confirm
        if confirm in ("child_male_likely", "child_female_likely", "child_gender_uncertain"):
            # Flipping a male/adolescent classification to a child category
            # requires BOTH high confidence AND actual corroborating evidence
            # in the confirmation pass's own text — a self-reported
            # confidence tag alone is not sufficient. This is the fix for
            # clearly adult/male voices being flipped to "child" on a single
            # isolated pass's say-so.
            if high_confirm and _confirm_child_flip_has_evidence(confirm_text):
                return confirm
            return initial

        # Prefer adolescent over post_puberty when either pass supports adolescent
        # at medium/high confidence — early teen-pop male leads are often still
        # high/light and should not be rounded up to settled post-puberty male.
        if confirm == "adolescent_male_likely" and strong_confirm:
            return confirm
        if initial == "adolescent_male_likely" and confirm == "post_puberty_male":
            # Only allow upgrade to post_puberty on high confidence; medium keeps adolescent.
            if high_confirm:
                return confirm
            return initial
        if initial in ("post_puberty_male", "adult_male", "young_male") and confirm == "adolescent_male_likely":
            if strong_confirm:
                return confirm

        if high_confirm and confirm == "post_puberty_male":
            return confirm

        if confirm == initial:
            return confirm

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

    elif lead_profile in ("child_gender_uncertain",):
        note = (
            "\n\nVOCAL CLASSIFICATION PRIORITY: The final lead vocal category is a young/child voice with uncertain gender. "
            "For user-facing claims, say 'young/child voice; gender uncertain' or 'young voice; cannot confidently tell boy/girl'. Do not call it a girl/woman/boy/man unless the user explicitly corrects you."
        )

    elif lead_profile == "adolescent_male_likely":
        note = (
            "\n\nVOCAL CLASSIFICATION PRIORITY: The final lead vocal category is an adolescent male voice — "
            "clearly male, not a small child's voice, but not yet a fully mature adult male voice either "
            "(a voice actively changing, or recently changed but still light/boyish). "
            "For user-facing claims, say 'young/adolescent male voice' or 'a voice that sounds like it's still "
            "changing'. Do NOT round this up to a full adult male description ('post-pubescent', 'mature male "
            "voice') and do NOT round it down to 'child' or 'boy' — this is its own category with its own evidence."
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
            "\n\nVOCAL CLASSIFICATION PRIORITY: The final lead vocal category is mixed leads / multiple lead voices. "
            "Describe the distinct co-leads (register, rough gender if supported, section roles) using CO_LEAD_DETAIL / "
            "MULTI-SINGER audit when present. Do not flatten this to a single singer. "
            "Do not invent a second gender if only one gender is evidenced among the co-leads."
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
    apostrophes, parentheses, dashes, etc.) ending in one of `extensions`.
    Returns (cleaned_text, ref) — ref is None if nothing was found.
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
            if not p:
                return False
            # Expand ~; leave absolute/relative paths as given.
            expanded = os.path.expanduser(p)
            return os.path.exists(expanded)
        except Exception:
            return False

    def _resolve(p):
        """Return the path form that exists (expanded ~ if needed)."""
        try:
            expanded = os.path.expanduser(p)
            if os.path.exists(expanded):
                return expanded
        except Exception:
            pass
        return p

    # Primary: shell-style tokenization (handles drag-and-drop backslash escapes
    # and quoted paths). Falls back gracefully when apostrophes break shlex.
    try:
        tokens = shlex.split(text, posix=True)
    except ValueError:
        tokens = []

    for i, token in enumerate(tokens):
        if token.lower().endswith(ext_lower) and _exists(token):
            remaining = tokens[:i] + tokens[i + 1:]
            return " ".join(remaining).strip(), _resolve(token)

    # Unescape common shell forms, including apostrophes and parentheses
    # that macOS/Finder sometimes backslash-escapes on drag-and-drop.
    unescaped = (
        text.replace("\\'", "'")
        .replace('\\"', '"')
        .replace("\\ ", " ")
        .replace("\\(", "(")
        .replace("\\)", ")")
        .replace("\\[", "[")
        .replace("\\]", "]")
        .replace("\\&", "&")
        .replace("\\;", ";")
    )

    # Quoted paths — allow apostrophes inside double quotes and vice versa.
    for qpat in (
        rf'"([^"]+\.(?:{ext_group}))"',
        rf"'([^']+\.(?:{ext_group}))'",
    ):
        quoted = re.search(qpat, unescaped, re.IGNORECASE)
        if quoted and _exists(quoted.group(1)):
            cleaned = unescaped.replace(quoted.group(0), "").strip()
            return cleaned, _resolve(quoted.group(1))

    # Walk left from each extension match; prefer the longest existing path.
    # This is the main path for unquoted names with spaces, apostrophes,
    # parentheses, leading dashes, etc. — e.g.
    #   /Music/Artist's Song (Live) - Remaster.mp3 what key is this?
    best = None  # (start, end, path)
    for ext_match in re.finditer(rf"\.(?:{ext_group})\b", unescaped, re.IGNORECASE):
        end = ext_match.end()
        # Candidate starts: beginning of each whitespace-separated run before
        # the extension, PLUS the absolute start of the string (for paths that
        # begin with / or ~ with no leading token boundary issues).
        starts = [m.start() for m in re.finditer(r"\S+", unescaped[:end])]
        if 0 not in starts and end > 0:
            starts.insert(0, 0)
        for start in starts:
            candidate = unescaped[start:end].strip()
            # Strip trailing sentence punctuation that is not part of the path,
            # but keep parentheses / brackets / dashes that belong to the name.
            candidate = candidate.rstrip(".,!?;:")
            # Strip unbalanced leading quotes only.
            candidate = candidate.lstrip(chr(39) + chr(34))
            # Also try with a leading './' or without a spurious leading dash
            # token from a previous flag (rare).
            variants = [candidate]
            if candidate.startswith("./") or candidate.startswith(".\\"):
                variants.append(candidate[2:])
            for cand in variants:
                if _exists(cand):
                    if best is None or (end - start) > (best[1] - best[0]):
                        best = (start, end, _resolve(cand))
                    break
    if best is not None:
        start, end, candidate = best
        cleaned = (unescaped[:start] + " " + unescaped[end:]).strip()
        cleaned = re.sub(r"\s+", " ", cleaned)
        return cleaned, candidate

    # Last resort: if the entire trimmed text (or a ~ / absolute path substring)
    # is itself an existing audio file, accept it.
    whole = unescaped.strip().strip('"').strip("'")
    if whole.lower().endswith(ext_lower) and _exists(whole):
        return "", _resolve(whole)

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


def ollama_chat(messages: list, num_ctx=None, keep_alive=None):
    """Chat with the Ollama writer model.

    keep_alive:
      None — Ollama default (model stays loaded for subsequent turns).
      0 / "0" — unload the model immediately after this reply (used in batch
      so MF/Demucs/Omnizart do not compete with a resident 30B writer).
      Other values are passed through to Ollama (e.g. "5m").

    Musiclyse session state (writer_history, analysis caches) lives in this
    Python process and is unaffected by Ollama unload.
    """
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

            payload = {
                "model": OLLAMA_MODEL,
                "messages": payload_messages,
                "stream": False,
                "options": {"num_ctx": ctx},
            }
            if keep_alive is not None:
                payload["keep_alive"] = keep_alive

            resp = requests.post(
                OLLAMA_URL,
                json=payload,
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

    # In batch, unload the writer immediately after this short helper so it
    # does not sit in RAM while Music Flamingo / Demucs / Omnizart run.
    _ka = None
    if globals().get("BATCH_UNLOAD_OLLAMA", True) and _is_batch_context():
        _ka = globals().get("BATCH_OLLAMA_KEEP_ALIVE", 0)

    try:
        text, _usage = ollama_chat([
                {
                    "role": "user",
                    "content": COVER_ART_OBSERVATION_PROMPT,
                    "images": [cover_b64],
                }
            ],
            num_ctx=COVER_ART_DESCRIPTION_NUM_CTX,
            keep_alive=_ka,
        )
    except Exception as e:
        print(f"  (cover art description failed: {e})")
        return {}
    finally:
        if _ka is not None:
            try:
                ollama_unload_model()
            except Exception:
                pass

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

    # Batch: ephemeral load — unload weights after the reply; analysis text and
    # Musiclyse session state remain in this process's RAM.
    _ka = None
    if globals().get("BATCH_UNLOAD_OLLAMA", True) and _is_batch_context():
        _ka = globals().get("BATCH_OLLAMA_KEEP_ALIVE", 0)

    try:
        _si_text, _usage = ollama_chat(
            [{"role": "user", "content": prompt}],
            num_ctx=SINGER_IDENTITY_NUM_CTX,
            keep_alive=_ka,
        )
        return _si_text
    except Exception as e:
        print(f"  (singer identity resolution failed: {e})")
        return ""
    finally:
        if _ka is not None:
            try:
                ollama_unload_model()
            except Exception:
                pass


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
            # Device-placement logging prints one line per graph op (e.g.
            # "Slice/begin: (Const): /job:localhost/.../device:CPU:0") straight
            # from TensorFlow's C++ backend, bypassing both logging.disable()
            # and quiet_stdout()'s Python-level redirect. It's purely a debug
            # aid with no effect on transcription output, so only turn it on
            # when SHOW_OMNIZART_LOGS is explicitly enabled.
            tf.debugging.set_log_device_placement(globals().get("SHOW_OMNIZART_LOGS", False))
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

    if not UNLOAD_OMNIZART_AFTER_STEM_MIDI:
        return
    if _OMNIZART_APPS is None:
        # Still clear TF session if present from a prior load
        try:
            import tensorflow as tf
            tf.keras.backend.clear_session()
            gc.collect()
        except Exception:
            pass
        return

    try:
        apps = _OMNIZART_APPS
        _OMNIZART_APPS = None
        try:
            del apps
        except Exception:
            pass
        gc.collect()

        try:
            import tensorflow as tf
            tf.keras.backend.clear_session()
            gc.collect()
        except Exception:
            pass
        try:
            if torch.backends.mps.is_available():
                try:
                    torch.mps.synchronize()
                except Exception:
                    pass
                torch.mps.empty_cache()
            elif torch.cuda.is_available():
                torch.cuda.empty_cache()
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


def run_demucs_stems(stem_wav_path: str, out_dir: str, shifts: int = 0):
    """Run Demucs stem separation. shifts>0 enables test-time shift
    ensembling (`--shifts N`): Demucs is run N extra times on randomly
    time-shifted copies of the input and the results are averaged, which
    measurably improves separation quality -- particularly for the weaker
    htdemucs_6s guitar/piano stems -- at roughly (N+1)x runtime. See
    DEMUCS_SHIFTS_FAST / DEMUCS_SHIFTS_DEEP."""
    base_cmd = [sys.executable, "-m", "demucs", "-n", DEMUCS_MODEL, "-o", out_dir]
    if shifts and shifts > 0:
        base_cmd += ["--shifts", str(int(shifts))]
    base_cmd += [stem_wav_path]

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


def _prepare_demucs_stems_for_track(
    track_path,
    stem_temp_files,
    demucs_out_dirs,
    status_fn=None,
    deep_mode=False,
):
    """Run Demucs 6s stem separation for track_path up front (if enabled)
    and return (stems_dict, out_dir).

    Used to get a clean isolated 'vocals' stem for pitch tracking -- running
    pitch estimation on the raw mix instead can lock pyin onto a sustained
    bass/pad/guitar drone rather than the singer, which is what produced
    misleadingly narrow/flat "vocal pitch" reads. The stems computed here
    are reused later for the stem MIDI report so Demucs only runs once per
    track. Returns ({}, None) on any failure or if stem separation is
    disabled; callers must fall back to whole-mix analysis in that case."""
    if not ENABLE_STEM_MIDI:
        return {}, None
    try:
        if track_path in stem_temp_files:
            stem_wav = stem_temp_files[track_path]
        else:
            if status_fn:
                status_fn("Preparing stereo WAV for Demucs/MIDI...")
            stem_wav = convert_to_wav_for_stems(
                track_path,
                sample_rate=44100,
                channels=2,
                max_seconds=STEM_MIDI_MAX_SECONDS,
            )
            stem_temp_files[track_path] = stem_wav

        if status_fn:
            status_fn("Running Demucs 6s stem separation (this can be slow)...")
        out_dir = tempfile.mkdtemp(prefix="demucs_")
        demucs_out_dirs.append(out_dir)
        stems = run_demucs_stems(
            stem_wav, out_dir,
            shifts=DEMUCS_SHIFTS_DEEP if deep_mode else DEMUCS_SHIFTS_FAST,
        )
        return (stems or {}), out_dir
    except Exception as e:
        print(f"  (early stem separation for vocal pitch skipped: {e})")
        return {}, None


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


def _mmss_to_seconds(s):
    """Parse a 'M:SS', 'MM:SS', or 'H:MM:SS' timestamp into seconds. Returns
    None if it doesn't look like a timestamp."""
    if not s:
        return None
    parts = s.strip().split(":")
    if not (2 <= len(parts) <= 3):
        return None
    try:
        parts = [float(p) for p in parts]
    except ValueError:
        return None
    seconds = 0.0
    for p in parts:
        seconds = seconds * 60.0 + p
    return seconds


_STRUCTURE_FIELD_RE = re.compile(r"STRUCTURE\s*=\s*\[?(.*?)(?:\]|\n\s*\n|\n[A-Z_]+\s*=)", re.DOTALL)
_STRUCTURE_ENTRY_RE = re.compile(
    r"([A-Za-z][A-Za-z0-9 '\-/]*?)\s*"
    r"(\d{1,2}:\d{2}(?::\d{2})?)\s*[-–—]\s*(\d{1,2}:\d{2}(?::\d{2})?)"
)


def extract_structure_sections(text):
    """Parse the model's STRUCTURE=[...] field (e.g. 'Verse 1 0:12-0:34;
    Chorus 0:34-0:58; ...') into a list of {name, start, end} dicts with
    times in seconds. Used to ground per-section groove analysis in the
    same section boundaries already reported to the user, rather than
    re-guessing structure independently. Returns [] if no STRUCTURE field
    or no parseable entries are found."""
    if not text:
        return []
    m = _STRUCTURE_FIELD_RE.search(text)
    body = m.group(1) if m else text
    sections = []
    for entry in re.split(r";|\n", body):
        entry = entry.strip(" -\t")
        if not entry:
            continue
        em = _STRUCTURE_ENTRY_RE.search(entry)
        if not em:
            continue
        name = em.group(1).strip(" -")
        start = _mmss_to_seconds(em.group(2))
        end = _mmss_to_seconds(em.group(3))
        if name and start is not None and end is not None and end > start:
            sections.append({"name": name, "start": start, "end": end})
    return sections


def _section_group_key(name):
    """Normalize 'Verse 1' / 'Verse 2' / 'Chorus 3' -> 'Verse' / 'Chorus' so
    repeated sections of the same kind can be pooled for a single groove
    read, which is almost always what's meant by "the groove in the verse"."""
    return re.sub(r"\s*\d+\s*$", "", name).strip().lower()


def _snap_onsets_to_subdivision(times, beat_len, subdivision=0.25, max_snap_s=0.045):
    """Snap onset times to the nearest beat subdivision if within max_snap_s.

    Programmed pop drums often land a few ms early/late relative to a
    detector's onset frame; that jitter tanks circular concentration even
    when the musical grid is tight. Snapping within a small window recovers
    lock for quantized kits without inventing structure for free-time playing
    (hits farther than max_snap_s from any grid point are left alone).
    subdivision is in beats (0.25 = 16th notes at the track tempo).
    """
    if not times or beat_len <= 0:
        return list(times) if times else []
    cell = beat_len * float(subdivision)
    if cell <= 0:
        return list(times)
    out = []
    for t in times:
        t = float(t)
        nearest = round(t / cell) * cell
        if abs(t - nearest) <= max_snap_s:
            out.append(nearest)
        else:
            out.append(t)
    return out


def _circular_lock(times, beat_len, divisor):
    """Circular concentration (0-1) of onset phase within a grid cell of
    `divisor` beats. 1.0 = every hit at the same phase; ~0 = scattered."""
    cell = beat_len * divisor
    if cell <= 0 or not times:
        return 0.0
    phases = (np.array(times, dtype=float) % cell) / cell
    angles = phases * 2 * np.pi
    return float(np.abs(np.mean(np.exp(1j * angles))))


def _trim_phase_outliers(times, beat_len, divisor, keep_frac=0.85):
    """Drop the farthest-from-mean-phase hits before lock scoring.

    One fill or a few mis-labelled onsets should not force an entire
    programmed kick pattern into the 'irregular' bin.
    """
    if not times or len(times) < 8:
        return list(times)
    cell = beat_len * divisor
    if cell <= 0:
        return list(times)
    arr = np.array(times, dtype=float)
    phases = (arr % cell) / cell
    angles = phases * 2 * np.pi
    mean_angle = np.angle(np.mean(np.exp(1j * angles)))
    # Circular distance to mean phase
    dist = np.abs(((angles - mean_angle + np.pi) % (2 * np.pi)) - np.pi)
    n_keep = max(6, int(len(arr) * keep_frac))
    if n_keep >= len(arr):
        return list(times)
    idx = np.argsort(dist)[:n_keep]
    return sorted(float(arr[i]) for i in idx)


def _classify_kick_grid(kicks, bpm):
    """Classify how the kick drum sits on the track's beat grid, using the
    reconciled tempo rather than a raw hits-per-minute threshold.

    A hit-rate-only heuristic can't tell a genuine one-kick-per-beat
    four-on-the-floor pattern apart from, say, a mid-tempo song with kicks
    only on beats 1 and 3 that happens to fall in the same rate band. This
    instead measures how tightly the kicks cluster around a candidate
    beat-grid (quarter/half/eighth note) using circular concentration,
    independent of where bar 1 actually falls (which we don't know).

    Lock thresholds are tiered so medium concentration on a rate-matching
    pattern (common with soft 808s / onset jitter on quantized pop) is
    reported as 'mostly on grid' rather than 'irregular' — the previous
    hard 0.75 cutoff was labelling tight programmed kits as wandering.

    Returns (label, detail) or (None, None) if there isn't enough evidence
    (no bpm, or too few kicks) to say anything reliable.
    """
    if not bpm or bpm <= 0 or not kicks or len(kicks) < 6:
        return None, None

    kicks_raw = sorted(float(k) for k in kicks)
    beat_len = 60.0 / bpm
    span_beats = (kicks_raw[-1] - kicks_raw[0]) / beat_len if kicks_raw[-1] > kicks_raw[0] else 0
    if span_beats < 3:
        return None, None
    kicks_per_beat = len(kicks_raw) / span_beats if span_beats > 0 else 0.0

    # Snap then score — use the better of raw vs snapped lock so free-time
    # playing is not artificially tightened.
    kicks_snap = _snap_onsets_to_subdivision(kicks_raw, beat_len, subdivision=0.25, max_snap_s=0.045)

    def _best_lock(divisor):
        raw_t = _trim_phase_outliers(kicks_raw, beat_len, divisor)
        snap_t = _trim_phase_outliers(kicks_snap, beat_len, divisor)
        return max(
            _circular_lock(raw_t, beat_len, divisor),
            _circular_lock(snap_t, beat_len, divisor),
        )

    lock_quarter = _best_lock(1.0)   # every beat
    lock_half = _best_lock(2.0)      # every other beat
    lock_eighth = _best_lock(0.5)    # two per beat

    # Tiered thresholds (was a single hard 0.75 → everything else 'irregular').
    STRONG = 0.68
    MODERATE = 0.52
    WEAK = 0.40

    def _rate_label(strong_lab, moderate_lab, lock_val, kpb_note):
        if lock_val >= STRONG:
            return strong_lab, f"~{kicks_per_beat:.2f} kicks/beat, grid-lock {lock_val:.2f}"
        if lock_val >= MODERATE:
            return moderate_lab, f"~{kicks_per_beat:.2f} kicks/beat, grid-lock {lock_val:.2f} (moderate — quantized/programmed kits often land here)"
        return None, None

    if 0.8 <= kicks_per_beat <= 1.25:
        lab, det = _rate_label(
            "four-on-the-floor (kick locked to every beat)",
            "mostly four-on-the-floor / on-beat kick (moderate grid-lock; treat as a programmed on-beat pattern, not irregular)",
            lock_quarter,
            kicks_per_beat,
        )
        if lab:
            return lab, det
    if 0.35 <= kicks_per_beat <= 0.7:
        lab, det = _rate_label(
            "half-time kick (roughly every other beat, e.g. 1 & 3)",
            "mostly half-time kick (moderate grid-lock; e.g. roughly 1 & 3 — not irregular)",
            lock_half,
            kicks_per_beat,
        )
        if lab:
            return lab, det
    if 1.6 <= kicks_per_beat <= 2.4:
        lab, det = _rate_label(
            "eighth-note / double-time kick (two hits per beat)",
            "mostly eighth-note / double-time kick (moderate grid-lock — not irregular)",
            lock_eighth,
            kicks_per_beat,
        )
        if lab:
            return lab, det

    best = max(lock_quarter, lock_half, lock_eighth)
    if best < WEAK:
        return (
            "syncopated / not locked to a simple beat grid",
            f"~{kicks_per_beat:.2f} kicks/beat, best grid-lock only {best:.2f}",
        )
    if best < MODERATE:
        return (
            "loosely on a beat grid (some timing scatter or syncopation; not a free-time irregular spray)",
            f"~{kicks_per_beat:.2f} kicks/beat, best grid-lock {best:.2f}",
        )
    # Rate did not match a simple pattern, but phase concentration is decent.
    return (
        "kick pattern with moderate grid-lock (rate does not match simple on-beat/half-time/eighth; prefer texture over 'irregular')",
        f"~{kicks_per_beat:.2f} kicks/beat, best grid-lock {best:.2f}",
    )


def _classify_snare_grid(snares, bpm, kicks=None):
    """Classify how the snare sits on the beat grid, the same way
    _classify_kick_grid does for the kick — grid-lock tested against the
    track's actual tempo rather than inferred from raw rate.

    When kicks are also grid-locked, this additionally checks the phase
    offset between kick and snare on a 2-beat cell: a real backbeat sits
    opposite the kick (e.g. kick on 1 & 3, snare on 2 & 4); a snare that
    coincides with the kick's own phase is something else (doubling, not
    a backbeat), even though both can produce the same raw "every other
    beat" hit rate.

    Tiered lock thresholds (strong / moderate / weak) mirror the kick
    classifier so quantized pop snares with mild onset jitter are not
    reported as 'irregular' or 'wandering'.

    Returns (label, detail) or (None, None) if there isn't enough evidence.
    """
    if not bpm or bpm <= 0 or not snares or len(snares) < 6:
        return None, None

    snares_raw = sorted(float(s) for s in snares)
    beat_len = 60.0 / bpm
    span_beats = (snares_raw[-1] - snares_raw[0]) / beat_len if snares_raw[-1] > snares_raw[0] else 0
    if span_beats < 3:
        return None, None
    snares_per_beat = len(snares_raw) / span_beats if span_beats > 0 else 0.0

    snares_snap = _snap_onsets_to_subdivision(snares_raw, beat_len, subdivision=0.25, max_snap_s=0.045)

    def _lock_and_phase(divisor, times):
        cell = beat_len * divisor
        if cell <= 0 or not times:
            return 0.0, 0.0
        phases = (np.array(times, dtype=float) % cell) / cell
        angles = phases * 2 * np.pi
        z = np.mean(np.exp(1j * angles))
        return float(np.abs(z)), float(np.angle(z)) % (2 * np.pi)

    def _best_lock_phase(divisor):
        # Prefer snapped times when they improve lock; keep phase from the
        # version that wins so kick/snare offset stays consistent.
        raw_t = _trim_phase_outliers(snares_raw, beat_len, divisor)
        snap_t = _trim_phase_outliers(snares_snap, beat_len, divisor)
        lr, pr = _lock_and_phase(divisor, raw_t)
        ls, ps = _lock_and_phase(divisor, snap_t)
        if ls >= lr:
            return ls, ps, snap_t
        return lr, pr, raw_t

    lock_quarter, phase_quarter, _ = _best_lock_phase(1.0)
    lock_half, phase_half, snares_for_phase = _best_lock_phase(2.0)
    lock_eighth, phase_eighth, _ = _best_lock_phase(0.5)

    STRONG = 0.65
    MODERATE = 0.50
    WEAK = 0.40

    def _phase_vs_kick(kicks_list, snare_phase):
        if not kicks_list or len(kicks_list) < 6:
            return None, None
        kicks_s = _snap_onsets_to_subdivision(
            sorted(float(k) for k in kicks_list), beat_len, subdivision=0.25, max_snap_s=0.045
        )
        kicks_t = _trim_phase_outliers(kicks_s, beat_len, 2.0)
        klock_half, kphase_half = _lock_and_phase(2.0, kicks_t)
        if klock_half < MODERATE:
            return klock_half, None
        phase_diff = abs(((snare_phase - kphase_half + np.pi) % (2 * np.pi)) - np.pi) / np.pi
        return klock_half, phase_diff

    if 0.35 <= snares_per_beat <= 0.7 and lock_half >= MODERATE:
        detail = f"~{snares_per_beat:.2f} snares/beat, grid-lock {lock_half:.2f}"
        klock, phase_diff = _phase_vs_kick(kicks, phase_half)
        if phase_diff is not None:
            detail += f", kick/snare phase offset {phase_diff:.2f} (1.0 = opposite the kick)"
            if phase_diff >= 0.7:
                lab = (
                    "classic backbeat (snare falls opposite the kick, e.g. 2 & 4 against kick on 1 & 3)"
                    if lock_half >= STRONG
                    else "mostly classic backbeat (moderate grid-lock; treat as a programmed backbeat, not irregular)"
                )
                return lab, detail
            if phase_diff <= 0.3:
                return "snare doubling the kick's grid position (not a classic backbeat)", detail
            return (
                "snare on a backbeat-rate grid with imperfect opposite-phase lock (still a backbeat-family pattern, not wandering)",
                detail,
            )
        lab = (
            "backbeat-rate snare (roughly every other beat; no locked kick to confirm the offset)"
            if lock_half >= STRONG
            else "mostly backbeat-rate snare (moderate grid-lock — not irregular)"
        )
        return lab, detail

    if 0.8 <= snares_per_beat <= 1.25 and lock_quarter >= MODERATE:
        lab = (
            "snare on every beat (dense, not a classic backbeat)"
            if lock_quarter >= STRONG
            else "mostly on-beat snare (moderate grid-lock; dense, not a classic backbeat)"
        )
        return lab, f"~{snares_per_beat:.2f} snares/beat, grid-lock {lock_quarter:.2f}"

    if 1.6 <= snares_per_beat <= 2.4 and lock_eighth >= MODERATE:
        lab = (
            "snare on eighth-note subdivisions (busy/rolled feel)"
            if lock_eighth >= STRONG
            else "mostly eighth-note snare subdivisions (moderate grid-lock — not irregular)"
        )
        return lab, f"~{snares_per_beat:.2f} snares/beat, grid-lock {lock_eighth:.2f}"

    best = max(lock_quarter, lock_half, lock_eighth)
    if best < WEAK:
        return (
            "syncopated / off-grid snare placement",
            f"~{snares_per_beat:.2f} snares/beat, best grid-lock only {best:.2f}",
        )
    if best < MODERATE:
        return (
            "loosely on a snare grid (some timing scatter; not a free-time irregular spray)",
            f"~{snares_per_beat:.2f} snares/beat, best grid-lock {best:.2f}",
        )
    return (
        "snare pattern with moderate grid-lock (rate does not match simple backbeat/on-beat/eighth; prefer texture over 'irregular')",
        f"~{snares_per_beat:.2f} snares/beat, best grid-lock {best:.2f}",
    )


def _detect_swing_feel(onsets, bpm=None, min_hits=10):
    """Detect a genuine swing/shuffle feel from a dense, regular subdivision
    stream (typically the hi-hat, or the snare/kick if the hat is too
    sparse).

    Swing shows up as a *repeating long-short alternation* in adjacent
    inter-onset intervals -- not just spacing variability in general, which
    the existing per-type "spacing-spread" figure already reports and which
    can come from many other things (transcription noise, fills, tempo
    drift) that have nothing to do with swing. This checks specifically for
    an alternating pattern before calling anything swung.

    If bpm is available, the median spacing is also sanity-checked against
    the expected eighth-note length at that tempo, so a stream that isn't
    actually running at roughly 8th-note density doesn't get misread as a
    swing/straight subdivision.

    Returns (label, ratio, detail) or (None, None, None) if there isn't
    enough clean evidence to say.
    """
    if not onsets or len(onsets) < min_hits:
        return None, None, None
    onsets = sorted(float(o) for o in onsets)
    iois = np.diff(onsets)
    if len(iois) < min_hits - 1:
        return None, None, None
    mean_ioi = float(np.mean(iois))
    if mean_ioi <= 1e-6:
        return None, None, None

    if bpm and bpm > 0:
        # Mean, not median: for a long/short subdivision the mean is exactly
        # the half-beat regardless of the swing ratio (long+short always
        # sums to one beat), whereas the median can skew toward whichever
        # value happens to be in the majority by one sample and drift away
        # from the true subdivision size. A tighter band than the raw
        # regularity check below, so this specifically catches streams that
        # aren't running at roughly 8th-note density (e.g. plain quarter
        # notes) rather than mislabeling them straight-8ths.
        expected_8th = 30.0 / bpm  # (60/bpm) / 2
        if not (0.55 * expected_8th <= mean_ioi <= 1.8 * expected_8th):
            return None, None, None

    # Only trust this on an already fairly regular subdivision stream (no
    # wild outliers like a dropped beat or a fill breaking the pattern).
    # Bounds are set relative to the mean rather than the median: a real
    # heavy shuffle (e.g. a 3:1 long/short ratio) skews the median toward
    # whichever of the two values happens to be in the majority by one
    # sample, which can push its own short notes just outside a
    # median-centred tolerance band and get an otherwise clean shuffle
    # rejected as "irregular".
    within_range = np.sum((iois > 0.25 * mean_ioi) & (iois < 4.0 * mean_ioi))
    if within_range / len(iois) < 0.6:
        return None, None, None

    # Split around the mean, not the median: for a genuine ~50/50 long/short
    # alternation, the median of the combined sequence often lands exactly
    # on one of the two discrete values (whichever is in the majority by a
    # single sample), which would misclassify that entire bucket as
    # "not different enough" from the threshold. The mean sits strictly
    # between two distinct values and doesn't have that failure mode.
    long_mask = iois > mean_ioi
    short_mask = ~long_mask

    if not long_mask.any() or not short_mask.any():
        return "straight", 1.0, "no clear long/short contrast; straight subdivision"

    labels = np.where(long_mask, 1, -1)
    alt_frac = float(np.sum(labels[1:] != labels[:-1]) / (len(labels) - 1))
    long_vals = iois[long_mask]
    short_vals = iois[short_mask]
    ratio = float(np.mean(long_vals) / np.mean(short_vals))

    if alt_frac < 0.55:
        return "straight", ratio, f"long/short IOIs don't alternate consistently ({alt_frac:.2f}); reads as straight, not swung"

    if ratio < 1.15:
        label = "straight"
    elif ratio < 1.5:
        label = "light swing/shuffle"
    elif ratio < 2.2:
        label = "moderate-to-triplet swing"
    else:
        label = "heavy shuffle / dotted-note feel"

    return label, ratio, f"long:short IOI ratio ~{ratio:.2f}, alternation consistency {alt_frac:.2f}"


def _summarize_drum_rhythm(filtered, max_pattern_hits=32, bpm=None):
    """Compact groove description from drum hits — no full event list needed.

    Returns lines covering:
      - per-type rates and typical spacing
      - a short pattern sample (types only) sampled across the track
      - kick/snare relationship (backbeat vs on-beat)
      - hat density / openness proxy
      - a kick-grid classification (four-on-the-floor / half-time / eighth-
        note / syncopated), grounded in the track's actual tempo rather than
        a raw hit-rate guess, when bpm is available
      - a snare-grid classification (classic backbeat / doubling the kick /
        every-beat / eighth-note / syncopated), also tempo-grounded, and
        phase-checked against the kick when both are grid-locked
      - a swing/shuffle read from the densest subdivision stream, based on
        long-short IOI alternation rather than raw spacing variability
      - a short GROOVE_HINT line the writer can paraphrase instead of
        collapsing to "driving drums" / "tight backbeat"
    """
    if not filtered:
        return []

    by_type = {}
    for n in filtered:
        t = n.get("drum_type") or "other"
        by_type.setdefault(t, []).append(float(n["onset"]))

    lines = []
    rate_parts = []
    type_rates = {}
    for drum_type, onsets in sorted(by_type.items(), key=lambda kv: -len(kv[1])):
        onsets = sorted(onsets)
        if len(onsets) < 2:
            rate_parts.append(f"{drum_type}: {len(onsets)} hit(s)")
            type_rates[drum_type] = 0.0
            continue
        intervals = np.diff(onsets)
        med_ibi = float(np.median(intervals))
        rate = 60.0 / med_ibi if med_ibi > 1e-6 else 0.0
        type_rates[drum_type] = rate
        # Spacing variability as a crude swing/looseness proxy.
        if len(intervals) >= 4:
            iqr = float(np.percentile(intervals, 75) - np.percentile(intervals, 25))
            cv = (iqr / med_ibi) if med_ibi > 1e-6 else 0.0
            rate_parts.append(
                f"{drum_type}: {len(onsets)} hits, ~{rate:.1f}/min, "
                f"median spacing {med_ibi:.3f}s, spacing-spread {cv:.2f}"
            )
        else:
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
    hihats = by_type.get("hihat") or by_type.get("hi-hat") or []
    cymbals = by_type.get("cymbal") or []
    toms = by_type.get("tom") or []

    backbeat_frac = None
    kick_ibi = None
    if len(kicks) >= 4 and len(snares) >= 2:
        kick_ibi = float(np.median(np.diff(sorted(kicks)))) if len(kicks) >= 2 else None
        # Rough backbeat check: snares often sit near midpoint between kicks.
        mid_hits = 0
        for s in snares:
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
        backbeat_frac = mid_hits / max(1, len(snares))
        if kick_ibi and kick_ibi > 0:
            lines.append(
                f"kick/snare relationship: kick median spacing {kick_ibi:.3f}s; "
                f"~{mid_hits}/{len(snares)} snares near mid-interval "
                f"({backbeat_frac * 100:.0f}% backbeat-like) between surrounding kicks"
            )

    # Hat density relative to kick pulse.
    hat_vs_kick = None
    if hihats and kicks and kick_ibi and kick_ibi > 0:
        hat_med = float(np.median(np.diff(sorted(hihats)))) if len(hihats) >= 2 else None
        if hat_med and hat_med > 1e-6:
            hat_vs_kick = kick_ibi / hat_med
            density_word = (
                "very dense (8th/16th-ish)" if hat_vs_kick >= 3.2
                else "busy (around 8ths)" if hat_vs_kick >= 1.8
                else "sparse / on the beat" if hat_vs_kick < 1.2
                else "moderate"
            )
            lines.append(
                f"hi-hat density vs kick pulse: ~{hat_vs_kick:.1f}x "
                f"({density_word}; {len(hihats)} hat hits)"
            )
    elif hihats:
        lines.append(f"hi-hat activity: {len(hihats)} hits (no stable kick pulse for ratio)")

    # Fill / tom activity as a density proxy (not precise fill detection).
    if toms or cymbals:
        lines.append(
            f"tom/cymbal activity: toms={len(toms)}, cymbals={len(cymbals)} "
            f"(elevated counts often mean fills or open cymbal work)"
        )

    # Swing/shuffle: use the densest available subdivision stream (hi-hat
    # first, since it's normally the clearest subdivision voice; fall back
    # to snare or kick if the hat is too sparse to judge).
    swing_label, swing_ratio, swing_detail = None, None, None
    for _cand_onsets in (hihats, snares, kicks):
        if _cand_onsets and len(_cand_onsets) >= 10:
            swing_label, swing_ratio, swing_detail = _detect_swing_feel(_cand_onsets, bpm=bpm)
            if swing_label:
                break
    if swing_label:
        lines.append(f"swing/shuffle analysis: {swing_label} ({swing_detail})")

    # Human-readable groove hint the writer should paraphrase, not ignore.
    groove_bits = []
    kick_rate = type_rates.get("kick") or 0.0
    snare_rate = type_rates.get("snare") or 0.0
    hat_rate = type_rates.get("hihat") or type_rates.get("hi-hat") or 0.0

    kick_grid_label, kick_grid_detail = _classify_kick_grid(kicks, bpm)
    if kick_grid_label:
        lines.append(f"kick beat-grid analysis: {kick_grid_label} ({kick_grid_detail})")
        groove_bits.append(kick_grid_label)
    elif kick_rate >= 100:
        # No tempo available to test grid-lock — fall back to a hedged,
        # rate-only description rather than naming a specific pattern.
        groove_bits.append("fast kick activity (rate-only estimate; no tempo to confirm the grid)")
    elif kick_rate >= 25:
        groove_bits.append("moderate kick activity (rate-only estimate; no tempo to confirm the grid)")
    elif kick_rate > 0:
        groove_bits.append("sparse kick activity (rate-only estimate; no tempo to confirm the grid)")

    if backbeat_frac is not None:
        if backbeat_frac >= 0.55:
            groove_bits.append("clear backbeat snare")
        elif backbeat_frac <= 0.25:
            groove_bits.append("snare often off the classic backbeat / on-beat emphasis")
        else:
            groove_bits.append("mixed backbeat and off-backbeat snare placement")

    snare_grid_label, snare_grid_detail = _classify_snare_grid(snares, bpm, kicks=kicks)
    if snare_grid_label:
        lines.append(f"snare beat-grid analysis: {snare_grid_label} ({snare_grid_detail})")
        # This is the more rigorous, tempo-grounded read -- prefer it over
        # the plain midpoint-fraction groove bit above when both are present.
        if groove_bits and groove_bits[-1] in (
            "clear backbeat snare",
            "snare often off the classic backbeat / on-beat emphasis",
            "mixed backbeat and off-backbeat snare placement",
        ):
            groove_bits[-1] = snare_grid_label
        else:
            groove_bits.append(snare_grid_label)

    if hat_vs_kick is not None:
        if hat_vs_kick >= 3.2:
            groove_bits.append("busy closed-hat or 16th-note hat work")
        elif hat_vs_kick >= 1.8:
            groove_bits.append("regular 8th-note hat layer")
        elif hat_vs_kick < 1.2:
            groove_bits.append("open or sparse hat / little continuous hat bed")
    elif hat_rate >= 120:
        groove_bits.append("high hat hit rate (busy top end)")
    elif hat_rate > 0 and hat_rate < 30:
        groove_bits.append("minimal hat activity")

    if len(toms) >= max(6, int(0.08 * len(filtered))):
        groove_bits.append("noticeable tom/fill activity")
    if len(cymbals) >= max(8, int(0.1 * len(filtered))):
        groove_bits.append("frequent cymbal crashes/rides")

    if swing_label:
        groove_bits.append(
            "straight (unswung) subdivision feel" if swing_label == "straight" else f"{swing_label} feel"
        )

    if groove_bits:
        lines.append(
            "GROOVE_HINT (paraphrase in ordinary language; do not say "
            "'driving drums' / 'tight backbeat' unless the pattern matches, "
            "do not say 'four-on-the-floor' unless the kick beat-grid analysis "
            "reports it, do not say 'backbeat' unless the snare beat-grid "
            "analysis reports a classic backbeat, and do not say 'swung' / "
            "'shuffled' unless the swing/shuffle analysis reports it — none "
            "of these are safe defaults for a merely fast or busy drum part): "
            + "; ".join(groove_bits)
        )

    return lines


def _summarize_drum_rhythm_by_section(filtered, sections, bpm, max_pattern_hits=16, min_hits_per_section=8):
    """Group drum hits by the STRUCTURE sections already reported for this
    track (e.g. all 'Verse' occurrences pooled, all 'Chorus' occurrences
    pooled) and run the same grounded groove analysis on each group.

    This is what answers "what's the groove like in the chorus vs the
    verse" — instead of the writer guessing from the whole-track pattern
    (or from genre expectations), it gets a real per-section read computed
    from the actual drum-hit timestamps.

    Returns a list of text blocks, one per section group, or [] if there
    aren't enough sections/hits to say anything reliable.
    """
    if not filtered or not sections:
        return []

    groups = {}
    order = []
    for sec in sections:
        key = _section_group_key(sec["name"])
        if key not in groups:
            groups[key] = {"label": sec["name"], "ranges": []}
            order.append(key)
        groups[key]["ranges"].append((sec["start"], sec["end"]))
        # Prefer the shortest/plainest occurrence name as the group label
        # (e.g. "Verse" over "Verse 1") when both singular and numbered
        # forms show up.
        if len(sec["name"]) < len(groups[key]["label"]):
            groups[key]["label"] = sec["name"]

    blocks = []
    for key in order:
        ranges = groups[key]["ranges"]
        hits = [
            n for n in filtered
            if any(start <= float(n["onset"]) < end for start, end in ranges)
        ]
        if len(hits) < min_hits_per_section:
            continue

        range_str = ", ".join(
            f"{int(s // 60)}:{int(s % 60):02d}-{int(e // 60)}:{int(e % 60):02d}"
            for s, e in ranges
        )
        section_lines = _summarize_drum_rhythm(hits, max_pattern_hits=max_pattern_hits, bpm=bpm)
        if not section_lines:
            continue
        label = groups[key]["label"]
        blocks.append(
            f"SECTION GROOVE — {label} ({range_str}):\n" + "\n".join(f"  {l}" for l in section_lines)
        )

    return blocks


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

def summarize_stem_midi(stem, raw_notes, filtered, removed, preset, stem_rms=None, bpm=None, sections=None):
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
            filtered, max_pattern_hits=STEM_MIDI_DRUM_PATTERN_HITS, bpm=bpm
        ):
            lines.append(rhythm_line)

        if sections:
            section_blocks = _summarize_drum_rhythm_by_section(
                filtered, sections, bpm, max_pattern_hits=max(8, STEM_MIDI_DRUM_PATTERN_HITS // 2)
            )
            if section_blocks:
                lines.append(
                    "Per-section groove (from this track's own STRUCTURE boundaries; use these, "
                    "not the whole-track pattern, when asked about the groove in a specific "
                    "section like the verse or chorus):"
                )
                lines.extend(section_blocks)

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


_INSTRUMENT_TAGGER = None
_INSTRUMENT_TAGGER_LABELS = None
INSTRUMENT_TAGGER_IMPORT_ERROR = ""

# AudioSet's 527 classes include lots of non-instrument sounds (crowd noise,
# applause, silence, etc). Map the musically relevant subset down to a small
# set of musician-facing labels; everything else is dropped. Several AudioSet
# labels can map to the same displayed tag (e.g. "Organ"/"Electronic organ"),
# which is intentional -- their probabilities are shown separately below.
#
# NOTE: this is a hard allowlist, not a threshold -- a raw label with no
# entry here is dropped no matter how confident the tagger is. The mallet
# percussion / hand percussion / bell / drone-instrument families below were
# previously entirely absent, so those instruments could never surface
# regardless of how clearly audible they were.
_INSTRUMENT_LABEL_MAP = {
    "electric guitar": "electric guitar", "acoustic guitar": "acoustic guitar",
    "guitar": "guitar (type uncertain)", "bass guitar": "bass guitar",
    "electric bass guitar": "bass guitar", "double bass": "upright bass",
    "piano": "piano", "electric piano": "electric piano", "keyboard (musical)": "keyboard/synth",
    "synthesizer": "synth", "sampler": "sampler/synth",
    "organ": "organ", "electronic organ": "organ", "hammond organ": "organ",
    "violin, fiddle": "strings (violin/fiddle)", "cello": "strings (cello)",
    "string section": "string section", "viola": "strings (viola)",
    "bowed string instrument": "strings (bowed, unspecified)",
    "pizzicato": "strings (pizzicato)",
    "trumpet": "brass (trumpet)", "trombone": "brass (trombone)",
    "brass instrument": "brass", "french horn": "brass (french horn)",
    "saxophone": "saxophone", "flute": "flute", "clarinet": "clarinet",
    "wind instrument, woodwind instrument": "woodwind (unspecified)",
    "harmonica": "harmonica", "accordion": "accordion",
    "drum kit": "drum kit", "drum": "drums", "drum machine": "drum machine",
    "snare drum": "snare", "bass drum": "kick", "hi-hat": "hi-hat", "cymbal": "cymbal",
    "timpani": "timpani", "percussion": "percussion (unspecified)",
    "tambourine": "tambourine", "harp": "harp", "banjo": "banjo",
    "mandolin": "mandolin", "ukulele": "ukulele", "sitar": "sitar",
    "zither": "zither", "plucked string instrument": "plucked strings (unspecified)",
    "steel guitar, slide guitar": "slide/steel guitar", "electric organ": "organ",
    # Mallet / tuned percussion -- previously entirely unmapped.
    "marimba, xylophone": "marimba/xylophone", "glockenspiel": "glockenspiel",
    "vibraphone": "vibraphone", "steelpan": "steelpan/steel drum",
    "tubular bells": "tubular bells/chimes", "mallet percussion": "mallet percussion (unspecified)",
    # Hand/world percussion -- previously entirely unmapped.
    "tabla": "tabla", "gong": "gong", "wood block": "wood block",
    "maraca": "maraca/shaker", "rattle (instrument)": "shaker/rattle",
    # Bells -- previously entirely unmapped.
    "bell": "bell", "church bell": "church bell", "jingle bell": "jingle bells",
    "chime": "chimes", "wind chime": "wind chimes", "tuning fork": "tuning fork",
    # Drone / world / other melodic instruments -- previously entirely unmapped.
    "bagpipes": "bagpipes", "didgeridoo": "didgeridoo", "shofar": "shofar",
    "theremin": "theremin", "singing bowl": "singing bowl",
    "harpsichord": "harpsichord",
    "orchestra": "orchestra/large ensemble",
    "scratching (performance technique)": "turntable scratching",
    "beatboxing": "beatboxing",
}

# Genre-adjacent AudioSet classes -- a broad, YouTube-metadata-derived taxonomy,
# not a music-critic's genre ontology. Kept intentionally coarse; multiple
# entries can and do co-fire on the same track (e.g. "Electronic music" and
# "House music"), which is expected and left visible rather than collapsed.
_GENRE_LABEL_MAP = {
    "pop music": "pop", "hip hop music": "hip hop", "rock music": "rock",
    "heavy metal": "heavy metal", "punk rock": "punk", "grunge": "grunge",
    "progressive rock": "progressive rock", "rock and roll": "rock and roll",
    "psychedelic rock": "psychedelic rock", "rhythm and blues": "R&B",
    "soul music": "soul", "reggae": "reggae", "country": "country",
    "swing music": "swing", "bluegrass": "bluegrass", "funk": "funk",
    "folk music": "folk", "middle eastern music": "Middle Eastern",
    "jazz": "jazz", "disco": "disco", "classical music": "classical",
    "opera": "opera", "electronic music": "electronic",
    "house music": "house", "techno": "techno", "dubstep": "dubstep",
    "drum and bass": "drum and bass", "electronica": "electronica",
    "electronic dance music": "EDM", "ambient music": "ambient",
    "trance music": "trance", "music of latin america": "Latin",
    "salsa music": "salsa", "flamenco": "flamenco", "blues": "blues",
    "new-age music": "new-age", "vocal music": "vocal-led (unspecified)",
    "a capella": "a cappella", "music of africa": "African",
    "afrobeat": "afrobeat", "christian music": "Christian",
    "gospel music": "gospel", "music of asia": "Asian (unspecified)",
    "carnatic music": "Carnatic", "music of bollywood": "Bollywood",
    "ska": "ska", "traditional music": "traditional/folk (unspecified)",
    "independent music": "indie",
}

# Mood-adjacent AudioSet classes -- for cross-checking MOOD_VIBE the same way
# the genre map cross-checks GENRE_RANKED.
_MOOD_LABEL_MAP = {
    "happy music": "happy/upbeat", "funny music": "playful/quirky",
    "sad music": "sad/melancholic", "tender music": "tender/gentle",
    "exciting music": "exciting/energetic", "angry music": "angry/aggressive",
    "scary music": "tense/scary",
}


def _get_instrument_tagger():
    """Lazily load a pretrained PANNs (Pretrained Audio Neural Networks)
    AudioSet tagger. Optional: if panns_inference isn't installed, tagging is
    skipped everywhere it's used and the rest of the pipeline is unaffected.
    Returns None if unavailable."""
    global _INSTRUMENT_TAGGER, _INSTRUMENT_TAGGER_LABELS, INSTRUMENT_TAGGER_IMPORT_ERROR

    if _INSTRUMENT_TAGGER is False:
        return None
    if _INSTRUMENT_TAGGER is not None:
        return _INSTRUMENT_TAGGER

    try:
        with quiet_stdout():
            from panns_inference import AudioTagging
            from panns_inference.config import labels as panns_labels
            tagger = AudioTagging(checkpoint_path=None, device="cpu")
        _INSTRUMENT_TAGGER = tagger
        _INSTRUMENT_TAGGER_LABELS = [str(l).strip().lower() for l in panns_labels]
        return _INSTRUMENT_TAGGER
    except Exception as e:
        INSTRUMENT_TAGGER_IMPORT_ERROR = str(e)
        _INSTRUMENT_TAGGER = False
        return None


def _peak_normalize(y, target_peak=0.95):
    """Peak-normalize a mono waveform in place-ish (returns a new array).
    No-ops on near-silent audio to avoid amplifying noise floor into
    false-positive tags."""
    if y is None or len(y) == 0:
        return y
    peak = float(np.max(np.abs(y)))
    if not np.isfinite(peak) or peak < 1e-6:
        return y
    return y * (target_peak / peak)


def _panns_tag_windowed(
    audio_path,
    label_map,
    top_k,
    min_prob,
    window_seconds=None,
    hop_seconds=None,
    max_windows=None,
    normalize=None,
):
    """Shared windowed-tagging engine behind tag_stem_instruments(),
    tag_full_mix_instruments(), tag_track_genre() and tag_track_mood().

    Instead of one prediction averaged over the whole clip (PANNs' Cnn14 does
    global pooling, so a brief guitar solo or a bridge-only synth pad gets
    diluted against several minutes of everything else and can silently fall
    below min_prob), this runs the tagger over short overlapping windows and
    keeps, per label, the single strongest window rather than a track-wide
    average -- so something that's clearly present for even one window is
    reported, along with roughly where in the track it was strongest.

    Returns [(label, probability, peak_time_seconds), ...] sorted by
    probability descending. Returns [] if the tagger/audio is unavailable or
    nothing clears min_prob.
    """
    if window_seconds is None:
        window_seconds = INSTRUMENT_TAG_WINDOW_SECONDS
    if hop_seconds is None:
        hop_seconds = INSTRUMENT_TAG_HOP_SECONDS
    if max_windows is None:
        max_windows = INSTRUMENT_TAG_MAX_WINDOWS
    if normalize is None:
        normalize = INSTRUMENT_TAG_NORMALIZE

    tagger = _get_instrument_tagger()
    if tagger is None:
        return []

    try:
        y, sr = librosa.load(audio_path, sr=32000, mono=True)  # PANNs models expect 32kHz
    except Exception:
        return []
    if y is None or len(y) < sr * 0.5:
        return []

    duration = len(y) / float(sr)

    # Build window start/end sample indices. Falls back to a single
    # whole-clip "window" when windowing is disabled or the clip is shorter
    # than one window, which reproduces the old single-pass behaviour.
    windows = []
    if not window_seconds or duration <= window_seconds:
        windows.append((0, len(y), 0.0))
    else:
        win_n = int(window_seconds * sr)
        hop_n = max(1, int((hop_seconds or window_seconds) * sr))
        start = 0
        while start < len(y):
            end = min(start + win_n, len(y))
            if end - start >= sr * 0.5:  # skip trailing slivers under 0.5s
                windows.append((start, end, start / float(sr)))
            if end >= len(y):
                break
            start += hop_n
            if len(windows) >= max_windows:
                break

    # best[label] = (probability, peak_time_seconds)
    best = {}
    for start, end, t0 in windows:
        chunk = y[start:end]
        if normalize:
            chunk = _peak_normalize(chunk, INSTRUMENT_TAG_NORMALIZE_PEAK)
        try:
            with quiet_stdout():
                clipwise_output, _ = tagger.inference(chunk[None, :])
        except Exception:
            continue
        probs = np.asarray(clipwise_output[0], dtype=float)

        for idx in np.argsort(probs)[::-1]:
            prob = float(probs[idx])
            if prob < min_prob:
                break
            raw_label = _INSTRUMENT_TAGGER_LABELS[idx] if _INSTRUMENT_TAGGER_LABELS else ""
            mapped = label_map.get(raw_label)
            if mapped is None:
                continue
            prev = best.get(mapped)
            if prev is None or prob > prev[0]:
                best[mapped] = (prob, round(t0, 1))

    if not best:
        return []

    ranked = sorted(best.items(), key=lambda kv: kv[1][0], reverse=True)[:top_k]
    return [(label, prob, t) for label, (prob, t) in ranked]


def _is_guitar_family_label(label):
    lab = (label or "").strip().lower()
    if lab in INSTRUMENT_TAG_GUITAR_LABELS:
        return True
    return "guitar" in lab and "bass" not in lab


def _filter_instrument_tags(tags, note_count=None, whole_mix_tags=None):
    """Apply family-specific thresholds, stem-activity gating, and optional
    whole-mix agreement notes. Returns list of
    (label, prob, peak_t, confidence_note) where confidence_note is "" or a
    short caution string for the report line.
    """
    if not tags:
        return []

    mix_by_label = {}
    if whole_mix_tags:
        for lab, prob, _t in whole_mix_tags:
            mix_by_label[lab] = max(mix_by_label.get(lab, 0.0), float(prob))
            # Also index coarse family keys for soft agreement.
            if _is_guitar_family_label(lab):
                mix_by_label["__guitar__"] = max(
                    mix_by_label.get("__guitar__", 0.0), float(prob)
                )

    out = []
    for label, prob, t in tags:
        conf_note = ""
        min_needed = INSTRUMENT_TAG_MIN_PROB
        if _is_guitar_family_label(label):
            min_needed = max(min_needed, float(INSTRUMENT_TAG_GUITAR_MIN_PROB))

        if prob < min_needed:
            continue

        # Empty / near-empty stems: Demucs residuals often get guitar-ish tags.
        if (
            INSTRUMENT_TAG_REQUIRE_STEM_ACTIVITY
            and note_count is not None
            and note_count < INSTRUMENT_TAG_MIN_NOTES_FOR_WEAK
            and prob < INSTRUMENT_TAG_STRONG_PROB
        ):
            continue

        if whole_mix_tags is not None:
            mix_p = mix_by_label.get(label, 0.0)
            if _is_guitar_family_label(label):
                mix_p = max(mix_p, mix_by_label.get("__guitar__", 0.0))
            if mix_p < float(INSTRUMENT_TAG_MIX_AGREE_MIN_PROB):
                # Guitar-family without mix support is almost always Demucs
                # residual / synth lookalike — drop unless extremely strong.
                if _is_guitar_family_label(label) and prob < 0.55:
                    continue
                if prob < INSTRUMENT_TAG_STRONG_PROB:
                    # Weak stem-only claim with no mix support → drop.
                    continue
                conf_note = "weak: stem-only, whole-mix did not agree"
            elif mix_p >= min_needed:
                conf_note = "supported by whole-mix tag"
            else:
                conf_note = "partial whole-mix support"

        out.append((label, prob, t, conf_note))

    return out


def tag_stem_instruments(stem_wav_path, top_k=None, min_prob=None):
    """Run the pretrained audio tagger on one stem and return the top
    plausible instrument tags as [(label, probability, peak_time_seconds), ...],
    restricted to the musically-relevant AudioSet classes in
    _INSTRUMENT_LABEL_MAP.

    This is supporting evidence only -- like Essentia's measurements, it is
    not proof of identity on its own (AudioSet taggers can be fooled by
    timbral similarity, e.g. synth brass vs real brass), but it is a genuine
    independent signal the writer previously didn't have, on top of pitch/
    density stats that say nothing about timbre.

    Returns [] if the tagger isn't available, the stem can't be loaded, or
    nothing clears min_prob. Guitar-family labels use a higher threshold
    (see INSTRUMENT_TAG_GUITAR_MIN_PROB); final filtering for stem activity /
    mix agreement happens in build_omnizart_summaries via
    _filter_instrument_tags().
    """
    if top_k is None:
        top_k = INSTRUMENT_TAG_TOP_K
    # Fetch a slightly wider pool so family-specific thresholds can still
    # keep strong guitar tags while dropping weaker non-guitar noise.
    if min_prob is None:
        min_prob = min(INSTRUMENT_TAG_MIN_PROB, 0.10)
    return _panns_tag_windowed(stem_wav_path, _INSTRUMENT_LABEL_MAP, top_k, min_prob)


def tag_full_mix_instruments(mix_audio_path, top_k=None, min_prob=None):
    """Same as tag_stem_instruments(), but run on the original, un-separated
    mix rather than a Demucs stem. A stem-level tag inherits whatever
    mistakes Demucs made when separating (a source can be misassigned,
    smeared across stems, or attenuated); tagging the full mix is a second,
    independent read that isn't subject to those separation artifacts, and
    is useful for flagging cases where a stem's tags look inconsistent with
    what's actually audible in the full mix."""
    if top_k is None:
        top_k = WHOLE_MIX_INSTRUMENT_TAG_TOP_K
    if min_prob is None:
        min_prob = min(WHOLE_MIX_INSTRUMENT_TAG_MIN_PROB, 0.10)
    raw = _panns_tag_windowed(mix_audio_path, _INSTRUMENT_LABEL_MAP, top_k, min_prob)
    # Apply guitar-family floor on the mix as well.
    filtered = []
    for label, prob, t in raw:
        need = WHOLE_MIX_INSTRUMENT_TAG_MIN_PROB
        if _is_guitar_family_label(label):
            need = max(need, float(INSTRUMENT_TAG_GUITAR_MIN_PROB) * 0.85)
        if prob >= need:
            filtered.append((label, prob, t))
    return filtered


def tag_track_genre(mix_audio_path, top_k=None, min_prob=None):
    """Broad AudioSet genre tags for the full mix -- see _GENRE_LABEL_MAP
    and ENABLE_GENRE_MOOD_TAGGING docstring above. Supporting evidence only;
    never treated as a replacement for GENRE_RANKED."""
    if top_k is None:
        top_k = GENRE_TAG_TOP_K
    if min_prob is None:
        min_prob = GENRE_TAG_MIN_PROB
    return _panns_tag_windowed(mix_audio_path, _GENRE_LABEL_MAP, top_k, min_prob)


def tag_track_mood(mix_audio_path, top_k=None, min_prob=None):
    """Broad AudioSet mood tags for the full mix -- see _MOOD_LABEL_MAP.
    Supporting evidence only; never treated as a replacement for MOOD_VIBE."""
    if top_k is None:
        top_k = MOOD_TAG_TOP_K
    if min_prob is None:
        min_prob = MOOD_TAG_MIN_PROB
    return _panns_tag_windowed(mix_audio_path, _MOOD_LABEL_MAP, top_k, min_prob)


def build_genre_mood_signal_report(mix_audio_path):
    """Independent, low-cost genre/mood cross-check built from the same
    already-loaded PANNs tagger used for instrument tagging (see
    ENABLE_GENRE_MOOD_TAGGING docstring).

    Returns (report_text, genre_tags) where genre_tags is the list from
    tag_track_genre() (possibly empty). report_text is "" when nothing useful
    was produced. genre_tags are still returned when present so reconcile_genre()
    can build RECOMMENDED GENRE FOR DISCUSSION even if the prose block is used
    separately.
    """
    if not ENABLE_GENRE_MOOD_TAGGING:
        return "", []
    if mix_audio_path is None or mix_audio_path.startswith(("http://", "https://")):
        return "", []

    genre_tags = tag_track_genre(mix_audio_path)
    mood_tags = tag_track_mood(mix_audio_path)
    if not genre_tags and not mood_tags:
        return "", genre_tags or []

    lines = ["OBJECTIVE GENRE/MOOD SIGNAL (independent AudioSet classifier, PANNs)"]
    if genre_tags:
        lines.append(
            "genre-adjacent tags: "
            + ", ".join(f"{label} ({prob * 100:.0f}%)" for label, prob, _t in genre_tags)
        )
    if mood_tags:
        lines.append(
            "mood-adjacent tags: "
            + ", ".join(f"{label} ({prob * 100:.0f}%)" for label, prob, _t in mood_tags)
        )
    lines.append(
        "These categories are broad, overlapping, and derived from noisy YouTube "
        "metadata -- they are supporting evidence, not automatic overrides. "
        "When this signal clearly favours electronic/dance/synth-pop and GENRE_RANKED "
        "leads with rock/pop-punk mainly from guitar texture, prefer revising "
        "GENRE_RANKED toward the electronic/dance identity (see self-check genre rules). "
        "When both agree, keep GENRE_RANKED. When mixed, put the production-led label "
        "first and the secondary flavour second with lower confidence."
    )
    return "\n".join(lines), genre_tags


def build_whole_mix_instrument_report(mix_audio_path, tags=None):
    """Independent whole-mix instrument tagging (see tag_full_mix_instruments
    docstring) formatted as a report block. Returns "" if disabled, the
    tagger is unavailable, or nothing clears the probability threshold.

    If `tags` is provided (same shape as tag_full_mix_instruments output),
    skips re-running the tagger — used when the stem pass already computed
    whole-mix tags for agreement filtering.
    """
    if not ENABLE_WHOLE_MIX_INSTRUMENT_TAGGING:
        return ""
    if tags is None:
        if mix_audio_path is None or mix_audio_path.startswith(("http://", "https://")):
            return ""
        tags = tag_full_mix_instruments(mix_audio_path)
    if not tags:
        return (
            "WHOLE-MIX INSTRUMENT TAGS: none cleared the confidence thresholds "
            "(prefer describing texture without naming weak instrument identities)."
        )

    tag_text = ", ".join(
        f"{label} ({prob * 100:.0f}%, strongest near {t:.0f}s)" for label, prob, t in tags
    )
    return (
        "WHOLE-MIX INSTRUMENT TAGS (independent audio classifier run on the original, "
        "un-separated mix -- not subject to Demucs separation artifacts. "
        "Guitar-family labels use a higher confidence bar. Treat disagreement with a "
        "per-stem tag as a reason to omit or hedge that instrument rather than "
        "trusting the stem alone; still supporting evidence only, not proof. "
        "If a named instrument is absent here and only weakly present on one stem, "
        f"prefer 'no clear X' over asserting X): {tag_text}"
    )


def guitar_absence_note_from_tags(whole_mix_tags=None, report_text=None):
    """If whole-mix tagging ran and found no guitar-family support, return a
    short explicit negative note for the private analysis. Gives the writer
    positive evidence to resist phantom guitar carried over from earlier
    tracks in the chat, and to override weak MF genre-expectation guesses.
    """
    found_guitar = False
    if whole_mix_tags is not None:
        for item in whole_mix_tags:
            if not item:
                continue
            lab = item[0] if isinstance(item, (tuple, list)) else str(item)
            try:
                prob = float(item[1]) if isinstance(item, (tuple, list)) and len(item) > 1 else 0.0
            except (TypeError, ValueError):
                prob = 0.0
            if _is_guitar_family_label(lab) and prob >= float(INSTRUMENT_TAG_MIX_AGREE_MIN_PROB):
                found_guitar = True
                break
    elif report_text:
        # Parse lines from WHOLE-MIX INSTRUMENT TAGS report
        lower = report_text.lower()
        if "whole-mix instrument tags" in lower:
            for lab in ("electric guitar", "acoustic guitar", "guitar", "slide/steel"):
                if lab in lower and "none cleared" not in lower:
                    # crude: if a guitar word appears in a non-empty tag report
                    found_guitar = True
                    break
            if "none cleared" in lower or "no clear" in lower:
                found_guitar = False
        else:
            return ""  # tagging didn't run / no report
    else:
        return ""

    if found_guitar:
        return ""
    return (
        "GUITAR ABSENCE NOTE: whole-mix instrument tagging found no clear "
        "guitar-family signal on THIS track. Do not claim electric/acoustic "
        "guitar from genre expectation, a residual Demucs 'guitar' stem, or "
        "instruments heard on a previously discussed track. Prefer 'no clear "
        "guitar' or a texture description unless the user confirms otherwise."
    )



def build_omnizart_summaries(stems, whole_mix_tags=None, bpm=None, sections=None):
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
                stem, raw_notes, filtered, removed, preset, stem_rms=rms,
                bpm=bpm, sections=sections if stem == "drums" else None,
            )

            if ENABLE_INSTRUMENT_TAGGING and stem in INSTRUMENT_TAG_STEMS:
                status(f"Tagging instruments in {stem} stem...")
                raw_tags = tag_stem_instruments(path)
                note_count = int(meta.get("note_count") or 0)
                tags = _filter_instrument_tags(
                    raw_tags,
                    note_count=note_count,
                    whole_mix_tags=whole_mix_tags,
                )
                if tags:
                    tag_parts = []
                    for label, prob, t, conf_note in tags:
                        base = f"{label} ({prob * 100:.0f}%, strongest near {t:.0f}s)"
                        if conf_note:
                            base += f" [{conf_note}]"
                        tag_parts.append(base)
                    summary += (
                        f"\ninstrument tag (independent audio classifier, windowed; "
                        f"guitar-family + empty-stem filters applied; supporting evidence "
                        f"only — do NOT assert an instrument from a weak/stem-only tag): "
                        + ", ".join(tag_parts)
                    )
                elif raw_tags:
                    summary += (
                        "\ninstrument tag: raw classifier fired weak labels that were "
                        "dropped (below family threshold, empty-stem gate, or no whole-mix "
                        "agreement). Prefer texture description over naming those instruments."
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
# Characters that must never appear in a saved basename (path separators,
# Windows-reserved, control chars). Spaces, apostrophes, parentheses, dashes,
# unicode letters, etc. are intentionally kept so "/save" and "/load" work with
# natural names like "Artist Name - Song's Title (Live).json".
_SAVED_NAME_FORBIDDEN = re.compile(r'[/\\:\*\?"<>\|\x00-\x1f]')


def _sanitize_saved_name(name):
    """Keep spaces and most printable characters; only strip path-hostile chars.

    Previously this collapsed everything outside [A-Za-z0-9._-] to underscores,
    which made it impossible to save or load names containing spaces or common
    punctuation. We now preserve the user's intended filename as closely as
    the filesystem allows.
    """
    name = (name or "").strip().strip('"').strip("'")
    if not name:
        return ""

    base = os.path.basename(name)
    # Normalise runs of whitespace to a single space; drop path-hostile chars.
    safe = _SAVED_NAME_FORBIDDEN.sub("_", base)
    safe = re.sub(r"\s+", " ", safe).strip()
    if not safe:
        return ""
    # Keep a clean ".json" suffix when present (avoid "Name .json").
    if safe.lower().endswith(".json"):
        stem = safe[:-5].rstrip(" .")
        safe = (stem if stem else "song") + ".json"
    else:
        safe = safe.rstrip(" .")
    if not safe:
        return ""
    if len(safe) > 200:
        # Prefer keeping the extension when truncating.
        if safe.lower().endswith(".json"):
            stem = safe[:-5][:195].rstrip(" .")
            safe = (stem if stem else "song") + ".json"
        else:
            safe = safe[:200].rstrip(" .")
    return safe


def _metadata_artist_title(metadata, audio_path=None):
    """Return (artist, title) for batch/save naming from tags or filename."""
    artist = ""
    title = ""
    if metadata:
        artist = str(metadata.get("artist") or "").strip()
        title = str(metadata.get("title") or "").strip()

    if not title and audio_path:
        base = os.path.basename(audio_path or "")
        stem, _ext = os.path.splitext(base)
        title = stem.strip() if stem.strip() else ""

    if not artist:
        artist = "Unknown"
    if not title:
        title = f"song_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    return artist, title


def _batch_save_basename(audio_path, metadata=None):
    """Derive a .json basename as 'Artist Name - Song Name.json'.

    Uses file-tag artist/title when available. Missing artist → 'Unknown';
    missing title → original audio filename stem. Spaces and normal
    punctuation are kept (only path-hostile characters are replaced).

    e.g. tags Artist='Radiohead', Title='Creep' → 'Radiohead - Creep.json'
         no tags, file '01 Song Title.m4a' → 'Unknown - 01 Song Title.json'
    """
    artist, title = _metadata_artist_title(metadata, audio_path)

    def _part(s):
        s = _SAVED_NAME_FORBIDDEN.sub("_", str(s or ""))
        s = re.sub(r"\s+", " ", s).strip(" .")
        return s or "Unknown"

    stem = f"{_part(artist)} - {_part(title)}"
    if not stem.strip(" -"):
        stem = f"song_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if len(stem) > 180:
        stem = stem[:180].rstrip(" .")
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
        safe = _batch_save_basename(track_key, metadata=metadata)
    else:
        safe = _sanitize_saved_name(filename) or f"song_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        if not safe.lower().endswith(".json"):
            safe += ".json"

    path = os.path.join(SAVE_DIR, safe)

    lyrics_text = ""
    if metadata:
        lyrics_text = str(metadata.get("lyrics") or "").strip()

    # Write the prepared cover art next to the JSON so saves are self-contained.
    # Cover art uses the same stem as the JSON (including spaces / special chars).
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
    """Load a saved song by filename, preserving spaces and special characters.

    Matching order:
      1. Exact basename as given (after light sanitisation only)
      2. Same with .json appended
      3. Case-insensitive match against files in SAVE_DIR
      4. Match ignoring only differences in runs of whitespace
    """
    os.makedirs(SAVE_DIR, exist_ok=True)

    raw = (filename or "").strip().strip('"').strip("'")
    if not raw:
        return None

    safe = _sanitize_saved_name(raw)
    if not safe:
        return None

    candidates = [os.path.join(SAVE_DIR, safe)]
    if not safe.lower().endswith(".json"):
        candidates.append(os.path.join(SAVE_DIR, safe + ".json"))
    # Also try the raw basename if the user typed a path-like string.
    raw_base = os.path.basename(raw)
    if raw_base and raw_base != safe:
        candidates.append(os.path.join(SAVE_DIR, raw_base))
        if not raw_base.lower().endswith(".json"):
            candidates.append(os.path.join(SAVE_DIR, raw_base + ".json"))

    path = None
    for cand in candidates:
        if os.path.exists(cand):
            path = cand
            break

    if path is None:
        target = safe[:-5] if safe.lower().endswith(".json") else safe
        target_l = target.lower()
        target_ws = re.sub(r"\s+", " ", target_l).strip()
        match = None
        try:
            for f in os.listdir(SAVE_DIR):
                if not f.lower().endswith(".json"):
                    continue
                stem_name = f[:-5]
                fl = f.lower()
                stem_l = stem_name.lower()
                stem_ws = re.sub(r"\s+", " ", stem_l).strip()
                if (
                    stem_l == target_l
                    or fl == safe.lower()
                    or fl == (safe.lower() if safe.lower().endswith(".json") else safe.lower() + ".json")
                    or stem_ws == target_ws
                ):
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


def _match_saved_filename_prefix(text):
    """Match an actual saved JSON filename at the start of command text.

    This deliberately uses the filenames that exist on disk rather than shell
    tokenisation or punctuation-based parsing, so apostrophes and other
    printable filename characters remain part of the filename. Returns
    (filename, trailing_text) or None.
    """
    rest = (text or "").lstrip()
    if not rest or not os.path.isdir(SAVE_DIR):
        return None

    try:
        saved = [
            f for f in os.listdir(SAVE_DIR)
            if f.lower().endswith(".json") and os.path.isfile(os.path.join(SAVE_DIR, f))
        ]
    except Exception:
        return None

    # Longest-first prevents a shorter saved name from stealing the prefix of
    # a longer one. Matching is case-insensitive, while returning the actual
    # on-disk spelling/path-safe filename.
    saved.sort(key=len, reverse=True)

    if rest[:1] in ("'", '"'):
        quote = rest[0]
        body = rest[1:]
        body_l = body.lower()
        for name in saved:
            nl = name.lower()
            if body_l.startswith(nl):
                end = len(name)
                if len(body) > end and body[end] == quote:
                    return name, body[end + 1:].strip()
        return None

    rest_l = rest.lower()
    for name in saved:
        nl = name.lower()
        if not rest_l.startswith(nl):
            continue
        end = len(name)
        if len(rest) == end or rest[end].isspace():
            return name, rest[end:].strip()

    return None


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


def run_fresh_track_analysis_heavy(
    track_path,
    *,
    audio_temp_files,
    dsp_temp_files,
    stem_temp_files,
    demucs_out_dirs,
):
    """
    Full /listen-quality analysis for one local file, MINUS singer identity
    resolution. Does not touch writer_history or session token counters.

    Singer identity resolution is deliberately NOT done here — see
    resolve_identity_and_finalize(). It's a plain-text Ollama call with no
    dependency on Music Flamingo/Demucs/Omnizart/Essentia, so splitting it
    out lets it run in a separate, model-free process for isolated batch
    tracks (see run_batch_one_finish() / BATCH_SPLIT_IDENTITY_PROCESS): the
    torch/TF/MPS allocations from this heavy stage are not always reliably
    reclaimed within the SAME process even after unload + gc + cache-empty
    (a known MPS/TF limitation), so previously Ollama could try to load the
    writer model for identity resolution on top of that stranded memory and
    get OOM-killed (SIGKILL / "child exited -9") right at that stage.

    Returns a dict with everything resolve_identity_and_finalize() and
    save_song_data() need.
    """
    track_path = os.path.abspath(os.path.expanduser(track_path))
    if not os.path.exists(track_path):
        raise FileNotFoundError(f"File not found: {track_path}")

    # Free Ollama writer weights before the heavy analysis peak so MF/Demucs/
    # Omnizart are not competing with a resident 30B model. Chat history in the
    # parent Musiclyse process is unaffected (separate address space / stays
    # in Python RAM). Results of this analysis stay in local variables below.
    if globals().get("BATCH_UNLOAD_OLLAMA", True) and _is_batch_context():
        try:
            status("Unloading Ollama writer before analysis (batch)...")
            ollama_unload_model()
            status_done("Ollama writer unloaded for analysis phase")
        except Exception:
            pass

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
    first_pass, mf_conversation, main_max_tokens = _mf_run_main_analysis(
        mf_model, mf_processor, resolved_path, deep=DEEP_MODE
    )

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

    # Isolate vocals via Demucs *before* pitch tracking, when stem separation
    # is enabled, so pyin locks onto the singer instead of a sustained
    # bass/pad/guitar drone in the full mix. Reused below for the stem MIDI
    # report so Demucs only runs once per track.
    precomputed_stems, precomputed_demucs_out_dir = {}, None
    vocal_pitch_source = "full mix (no isolated vocal stem available)"
    if ENABLE_VOCAL_OBJECTIVE_REPORT and ENABLE_STEM_MIDI:
        precomputed_stems, precomputed_demucs_out_dir = _prepare_demucs_stems_for_track(
            track_path, stem_temp_files, demucs_out_dirs,
            status_fn=status, deep_mode=DEEP_MODE,
        )
        if precomputed_stems.get("vocals"):
            vocal_pitch_source = "isolated vocal stem (Demucs)"

    if dsp_path is not None:
        if ENABLE_OBJECTIVE_AUDIO_REPORT:
            status("Measuring beat/timbre with signal processing...")
            objective_report = build_objective_audio_report(dsp_path)
        if ENABLE_VOCAL_OBJECTIVE_REPORT:
            status("Measuring vocal pitch/formant proxies...")
            _pitch_source_path = precomputed_stems.get("vocals") or dsp_path
            vocal_objective_report = build_vocal_objective_report(_pitch_source_path)
            if vocal_objective_report:
                vocal_objective_report += f"\nmeasurement source: {vocal_pitch_source}"
        if ENABLE_ESSENTIA_REPORT and ESSENTIA_AVAILABLE:
            status("Measuring tempo/key/spectral features with Essentia...")
            essentia_report = build_essentia_report(dsp_path)

    genre_mood_report = ""
    panns_genre_tags = []
    if ENABLE_GENRE_MOOD_TAGGING and dsp_path is not None:
        status("Cross-checking genre/mood with an independent classifier...")
        try:
            genre_mood_report, panns_genre_tags = build_genre_mood_signal_report(dsp_path)
        except Exception as e:
            print(f"  (genre/mood signal skipped: {e})")
            genre_mood_report = ""
            panns_genre_tags = []

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
            elif initial_lead in (MALE_LEAD_CATEGORIES | ADOLESCENT_MALE_CATEGORIES) and (
                median_f0 is not None
                and F0_MALE_HIGH_CONFIRM_HZ is not None
                and median_f0 >= float(F0_MALE_HIGH_CONFIRM_HZ)
            ):
                # High track-wide median on a male-tagged lead is a common
                # female→male mislabel path; run confirmation before locking in.
                should_confirm = True
            elif initial_lead == "mixed_leads":
                should_confirm = True
            else:
                # Confirm when the vocal pass itself flags multiple distinct voices.
                _mv_early = parse_multi_voice_fields(vocal_result or "")
                if _mv_early.get("voice_arrangement") in ("duet_co_leads", "call_response"):
                    should_confirm = True
                elif _mv_early.get("num_distinct_voices") in ("2", "3", "3+", "two", "three"):
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
            initial_lead = _apply_vocal_age_guard(initial_lead, vocal_result)
        confirm_lead = _apply_vocal_age_guard(confirm_lead, confirmation_result)

        final_lead = choose_final_vocal_lead(
            initial_lead, confirm_lead, confirm_confidence, confirmation_result,
            median_f0=median_f0,
        )
        final_lead = _apply_vocal_age_guard(
            final_lead,
            (vocal_result or "") + "\n" + (confirmation_result or ""),
        )
        final_lead = _apply_f0_pitch_guard(
            final_lead,
            median_f0,
            low_f0=vocal_pitch.get("low") if isinstance(vocal_pitch, dict) else None,
            high_f0=vocal_pitch.get("high") if isinstance(vocal_pitch, dict) else None,
        )



    # If the main MF pass still claims "no music" but DSP heard a track, replace
    # the failure text with a conservative scaffold before self-check / save.
    first_pass = _mf_salvage_empty_analysis(
        first_pass, objective_report, essentia_report, vocal_result
    )

    if FAST_MODE:
        revised = first_pass
    else:
        status("Double-checking its own analysis for overconfident claims...")
        self_check_text = SELF_CHECK_PROMPT + "\n\n" + STYLE_EVIDENCE_FIREWALL
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
                "element activity, and dynamic range; do NOT use librosa/Essentia spectral stats "
                "to invent GENRE or change vocal identity):\n"
                + "\n\n".join(objective_crosscheck_parts)
            )
            if genre_mood_report:
                self_check_text += (
                    "\n\nIndependent genre/mood classifier (PANNs/AudioSet) — this MAY be used "
                    "to revise GENRE_RANKED when it clearly conflicts with a rock/pop-punk top rank "
                    "driven mainly by guitar texture while this signal is electronic/dance/synth-pop. "
                    "See GENRE self-check rules.\n"
                    + genre_mood_report
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
        # NOTE on repetition_penalty / no_repeat_ngram_size:
        # These exist to stop genuine degenerate token-looping (the same
        # short phrase/syllable chain repeating dozens of times in a row),
        # NOT to stop a real, once-per-section chorus repeat, which is
        # extremely common in verse/chorus songs. no_repeat_ngram_size is a
        # HARD ban enforced by generate() with no awareness of "this repeat
        # is really being sung again" vs. "this is degeneration" — once an
        # n-gram has been produced, it can never be produced again for the
        # rest of the sequence. A low value (previously 14 tokens, roughly
        # one sung line) was banning the literal, correct second occurrence
        # of a repeated chorus line, forcing the model to invent new,
        # thematically-similar-but-wrong wording to fill that stretch of
        # audio — i.e. exactly the "hallucinated chorus" failure mode. 48
        # tokens is long enough to survive one legitimate chorus repeat
        # (rarely more than ~20-25 tokens) while still catching pathological
        # multi-line loops. repetition_penalty was similarly lowered from
        # 1.55 (aggressive enough to distort ordinary word choice on any
        # reused word) to 1.15, a much more standard value. True
        # degenerate-loop protection now leans more on
        # _sanitize_lyrics_transcription's post-hoc immediate-repeat
        # detector below, which can distinguish "same line twice in a row"
        # from "same line once per chorus" using context the decoder can't see.
        full_lyrics = mf_generate(
            mf_model, mf_processor, lyrics_conversation,
            max_new_tokens=1280,
            repetition_penalty=1.15,
            no_repeat_ngram_size=48,
        )
        full_lyrics = _sanitize_lyrics_transcription(full_lyrics)

        # Second, INDEPENDENT decode of the same prompt/audio via sampling.
        # Greedy decoding is deterministic, so a single pass can be fluently
        # wrong with nothing to detect via repetition heuristics alone (see
        # the "Games anyone could played..." case). Comparing against a
        # differently-decoded second opinion catches ungrounded/hallucinated
        # spans that a single deterministic pass never repeats and therefore
        # never trips the spam/loop detector.
        full_lyrics_alt = mf_generate(
            mf_model, mf_processor, lyrics_conversation,
            max_new_tokens=1280,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.15,
            no_repeat_ngram_size=48,
        )
        full_lyrics_alt = _sanitize_lyrics_transcription(full_lyrics_alt)

        if full_lyrics and full_lyrics.strip():
            full_lyrics = _cross_check_lyrics_transcriptions(full_lyrics, full_lyrics_alt)
            revised += f"\n\nFULL LYRICS TRANSCRIPTION (dedicated pass, cross-checked against a second independent decode):\n{full_lyrics}"

    unload_music_flamingo()

    revised += f"\n\n11. ERA / RELEASE PERIOD (isolated dedicated pass):\n{era_result}"
    if vocal_result:
        revised += f"\n\nVOCAL / SINGER PROFILE (isolated dedicated pass):\n{vocal_result}"
        if confirmation_result:
            revised += f"\n\nVOCAL CONFIRMATION PASS:\n{confirmation_result}"
        f0_text = f"{median_f0} Hz" if median_f0 is not None else "unavailable"
        multi_voice_fields = parse_multi_voice_fields(vocal_result or "")
        if confirmation_result:
            conf_mv = parse_multi_voice_fields(confirmation_result)
            for k, v in conf_mv.items():
                if v and (not multi_voice_fields.get(k) or multi_voice_fields.get(k) in ("", "unparsed", "uncertain")):
                    multi_voice_fields[k] = v
        if (final_lead or initial_lead) == "mixed_leads" and not multi_voice_fields.get("voice_arrangement"):
            multi_voice_fields["voice_arrangement"] = "duet_co_leads"
            if not multi_voice_fields.get("num_distinct_voices"):
                multi_voice_fields["num_distinct_voices"] = "2"
        revised += (
            "\n\nVOCAL DECISION AUDIT (audio-only evidence for vocal age/gender):\n"
            f"- Initial isolated lead profile: {initial_lead or 'unparsed'}\n"
            f"- Confirmation pass: {confirm_lead or 'not run/unparsed'} (confidence={confirm_confidence or 'n/a'})\n"
            f"- Objective median f0: {f0_text}\n"
            f"- FINAL LEAD PROFILE: {final_lead or 'unknown'}\n"
            f"- BACKING PROFILES: {initial_backing or 'uncertain'}\n"
            "This is audio-only evidence. If a SINGER IDENTITY RESOLUTION block appears later in this analysis, use that for user-facing singer-identity claims; otherwise use FINAL LEAD PROFILE. Do not override a well-supported combined judgment with pitch impressions alone. If FINAL LEAD PROFILE is unknown/unparsed, do not treat free-text LEAD_PROFILE / LEAD_CATEGORY lines earlier in the vocal pass as a settled gender/age claim — those lines were not accepted as structured final tags. A very high objective median f0 can demote male/adolescent tags to uncertain; when FINAL is uncertain, prefer SINGER IDENTITY RESOLUTION (metadata + cover + pitch constraint) over any earlier free-text male claim."
        )
        mv_block = format_multi_voice_audit(
            multi_voice_fields, final_lead or initial_lead, initial_backing or "uncertain"
        )
        if mv_block:
            revised += "\n\n" + mv_block

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
                "- Prefer median + 5–95 percentile range for the main vocal range. Median f0 is also used as a soft constraint on lead age/gender tags (high median can demote an overconfident post-puberty-male claim; it does not invent gender alone). "
                "Do not treat absolute extremes from stem MIDI as the sung range."
            )
        revised += "\n\n" + "\n".join(pitch_lines)

    mf_bpm_val = extract_bpm_from_text(first_pass)
    revised_bpm_val = extract_bpm_from_text(revised)
    if revised_bpm_val is not None:
        mf_bpm_val = revised_bpm_val
    essentia_bpm_val = extract_essentia_bpm(essentia_report) if essentia_report else None
    objective_bpm_val = extract_objective_bpm(objective_report) if objective_report else None
    essentia_median_bpm_val = extract_essentia_median_bpm(essentia_report) if essentia_report else None
    objective_median_bpm_val = extract_objective_median_bpm(objective_report) if objective_report else None
    _bpm_genre_hint = _genre_hint_from_analysis_text(
        revised or first_pass,
        panns_genre_tags,
    )
    final_bpm, bpm_note = reconcile_bpm(
        mf_bpm_val, essentia_bpm_val, objective_bpm_val,
        essentia_median_bpm=essentia_median_bpm_val,
        objective_median_bpm=objective_median_bpm_val,
        genre_hint=_bpm_genre_hint,
    )
    if final_bpm:
        revised += (
            f"\n\nRECOMMENDED TEMPO FOR DISCUSSION: {final_bpm} BPM. "
            f"Reasoning: {bpm_note}. "
            "This is the primary tempo to report to the user. "
            "State it as a concrete figure (e.g. 'about 158 BPM'). "
            "Do not expand it into a range unless this block itself marks the value as uncertain."
        )

    mf_key_val = extract_key_from_text(first_pass)
    revised_key_val = extract_key_from_text(revised)
    if revised_key_val is not None:
        mf_key_val = revised_key_val
    essentia_key_val, essentia_key_strength = (None, None)
    if essentia_report:
        parsed = extract_essentia_key_from_text(essentia_report)
        if parsed:
            essentia_key_val, essentia_key_strength = parsed
    final_key, key_note = reconcile_key(mf_key_val, essentia_key_val, essentia_key_strength)
    if final_key:
        revised += (
            f"\n\nRECOMMENDED KEY FOR DISCUSSION: {final_key}. "
            f"Reasoning: {key_note} "
            "This is the primary key to report to the user unless the reasoning above marks it uncertain."
        )

    # Numeric dynamics / loudness — prefer the ORIGINAL local file (full duration,
    # native rate/channels) so LUFS is not biased by the mono 22.05 kHz DSP WAV.
    # ReplayGain/RGAD tags are intentionally ignored (raw PCM decode only).
    crest_db_val = None
    crest_src = "objective (librosa) crest-factor proxy"
    lufs_val = lra_val = loudness_src = None
    try:
        _dyn_path = None
        if (
            track_path
            and not str(track_path).startswith(("http://", "https://"))
            and os.path.exists(track_path)
        ):
            _dyn_path = track_path
        elif track_path in dsp_temp_files:
            _dyn_path = dsp_temp_files[track_path]
        elif dsp_path is not None:
            _dyn_path = dsp_path
        if _dyn_path:
            crest_db_val = measure_crest_factor_db(_dyn_path)
            lufs_val, lra_val, loudness_src = measure_ebur128_loudness(_dyn_path)
    except Exception:
        crest_db_val = None
    if crest_db_val is None:
        crest_db_val = extract_crest_db_from_text(objective_report) if objective_report else None
    if crest_db_val is None and essentia_report:
        crest_db_val = extract_crest_db_from_text(essentia_report)
        if crest_db_val is not None:
            crest_src = "Essentia crest-factor proxy"
    dyn_block = format_recommended_dynamics_block(
        crest_db=crest_db_val,
        source_note=crest_src,
        lufs=lufs_val,
        lra=lra_val,
        loudness_source=loudness_src,
    )
    if dyn_block:
        revised += dyn_block
    else:
        revised += (
            "\n\nRECOMMENDED DYNAMICS FOR DISCUSSION: unavailable "
            "(could not compute EBU R128 loudness or crest-factor proxy from this file). "
            "Do not invent a dB/LUFS figure; fall back only to qualitative production notes."
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
            "low/mid/high element activity, dynamic range / loudness (crest-factor proxy), "
            "and vocal pitch/formant proxies. "
            "Do NOT use it to infer or revise GENRE."
        )

    if genre_mood_report:
        revised += "\n\n" + genre_mood_report
    try:
        _rec_g, _rec_g_note = reconcile_genre(revised, panns_genre_tags)
        revised += format_recommended_genre_block(_rec_g, _rec_g_note)
        # Genre-conditioned Essentia key refinement (edma vs temperley/krumhansl).
        if _rec_g and dsp_path is not None:
            _prev_k = extract_essentia_key_from_text(essentia_report) if essentia_report else None
            _pk, _ps = (None, None)
            if _prev_k:
                _pk, _ps = _prev_k if isinstance(_prev_k, tuple) else (_prev_k, None)
            _nk, _ns, _nnote = refine_essentia_key_with_genre(
                dsp_path, _rec_g, previous_key=_pk, previous_strength=_ps
            )
            if _nnote:
                revised += f"\n\nKEY PROFILE REFINEMENT: {_nnote}"
            if _nk and _nnote and "shifted" in _nnote:
                # Re-reconcile recommended key with the refined Essentia estimate
                _mfk = extract_key_from_text(revised) or extract_key_from_text(first_pass)
                _fk, _fnote = reconcile_key(_mfk, _nk, _ns)
                if _fk:
                    revised += (
                        f"\n\nRECOMMENDED KEY FOR DISCUSSION: {_fk}. "
                        f"Reasoning: {_fnote} (after genre-conditioned Essentia refinement). "
                        "This is the primary key to report to the user unless the reasoning marks it uncertain."
                    )
    except Exception:
        pass

    stem_midi_report = ""
    if ENABLE_STEM_MIDI:
        try:
            _get_omnizart()
            if precomputed_stems:
                # Reuse the separation already run above for vocal pitch
                # isolation -- avoids a second, redundant Demucs pass.
                stems = precomputed_stems
            else:
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
                stems = run_demucs_stems(
                    stem_wav, out_dir, shifts=DEMUCS_SHIFTS_DEEP if DEEP_MODE else DEMUCS_SHIFTS_FAST,
                )
            if not stems:
                stem_midi_report = "STEM MIDI REPORT unavailable: Demucs did not produce expected stems."
            else:
                whole_mix_tags = None
                if ENABLE_WHOLE_MIX_INSTRUMENT_TAGGING and dsp_path is not None:
                    try:
                        status("Tagging instruments on the full (un-separated) mix...")
                        whole_mix_tags = tag_full_mix_instruments(dsp_path)
                    except Exception as e:
                        print(f"  (whole-mix instrument tagging skipped: {e})")
                        whole_mix_tags = None
                status("Running Omnizart on each separated stem...")
                _structure_sections = extract_structure_sections(revised or first_pass)
                stem_midi_report = build_omnizart_summaries(
                    stems, whole_mix_tags=whole_mix_tags,
                    bpm=final_bpm, sections=_structure_sections,
                )
                if whole_mix_tags is not None:
                    whole_mix_report = build_whole_mix_instrument_report(
                        dsp_path, tags=whole_mix_tags
                    )
                    if whole_mix_report:
                        stem_midi_report = stem_midi_report + "\n\n" + whole_mix_report
                        try:
                            _gan = guitar_absence_note_from_tags(locals().get("whole_mix_tags"), whole_mix_report)
                            if _gan:
                                stem_midi_report = stem_midi_report + "\n\n" + _gan
                        except Exception:
                            pass
                _release_omnizart_memory()
        except Exception as e:
            print(f"  (stem MIDI skipped/unavailable: {e})")
            stem_midi_report = f"STEM MIDI REPORT unavailable: {e}"
    stem_midi_report = stem_midi_report or "STEM MIDI REPORT not run (disabled)."
    revised += "\n\n" + stem_midi_report

    # Whole-mix tags are folded into the stem report above when both paths run.
    # If stem MIDI was disabled but whole-mix tagging is still on, run it alone.
    if (
        ENABLE_WHOLE_MIX_INSTRUMENT_TAGGING
        and dsp_path is not None
        and "WHOLE-MIX INSTRUMENT TAGS" not in (stem_midi_report or "")
    ):
        status("Tagging instruments on the full (un-separated) mix...")
        try:
            whole_mix_report = build_whole_mix_instrument_report(dsp_path)
            if whole_mix_report:
                revised += "\n\n" + whole_mix_report
                _gan = ""
                try:
                    _gan = guitar_absence_note_from_tags(locals().get("whole_mix_tags"), whole_mix_report)
                except Exception:
                    _gan = ""
                if _gan:
                    revised += "\n\n" + _gan
        except Exception as e:
            print(f"  (whole-mix instrument tagging skipped: {e})")

    # Singer identity resolution is intentionally NOT done here — see
    # resolve_identity_and_finalize(). Just precompute the plain-text
    # evidence summary it needs, then release the heavy models.
    vocal_audit_for_resolution = None
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

    # Always release MF/Omnizart/Ollama here, not just when identity
    # resolution is enabled — this is the point where this process's peak
    # (MF + Demucs + Omnizart all having been resident) has passed, so
    # freeing what we can now minimises what the OS has to reclaim on exit.
    _release_heavy_analysis_memory_before_identity()

    return {
        "track_path": track_path,
        "analysis": revised,
        "corrections": {},
        "metadata": metadata,
        "cover_observations": cover_observations,
        "cover_bytes": cover_bytes_for_save,
        "cover_mime": cover_mime,
        "stem_midi_report": stem_midi_report,
        "vocal_audit_for_resolution": vocal_audit_for_resolution,
        "final_lead": final_lead,
        "initial_backing": initial_backing,
        "vocal_result_present": bool(vocal_result),
    }


def resolve_identity_and_finalize(heavy):
    """Lightweight second stage: resolve singer identity via a plain-text
    Ollama call and fold the result into the analysis text.

    Deliberately kept separate from run_fresh_track_analysis_heavy() (see
    its docstring) so it can run in a process that never loaded Music
    Flamingo/Demucs/Omnizart/Essentia — used as its own subprocess stage
    for isolated batch tracks (run_batch_one_finish), and simply called
    right after the heavy stage for the non-split / interactive paths
    (run_fresh_track_analysis).
    """
    revised = heavy.get("analysis", "") or ""
    metadata = heavy.get("metadata")
    cover_observations = heavy.get("cover_observations")
    final_lead = heavy.get("final_lead")
    initial_backing = heavy.get("initial_backing")
    vocal_result_present = bool(heavy.get("vocal_result_present"))
    vocal_audit_for_resolution = heavy.get("vocal_audit_for_resolution")

    singer_identity = ""
    if ENABLE_SINGER_IDENTITY_RESOLUTION and (metadata or cover_observations):
        status("Resolving singer identity from audio + metadata + cover art...")
        try:
            singer_identity = resolve_singer_identity(
                metadata,
                vocal_audit_for_resolution or "",
                cover_observations,
                {},
            ) or ""
        except Exception as e:
            print(f"  (singer identity skipped: {e})")
            singer_identity = ""
        finally:
            # Drop writer weights again so the next batch track starts clean.
            # Analysis strings / identity text remain in local variables.
            if globals().get("BATCH_UNLOAD_OLLAMA", True) and _is_batch_context():
                try:
                    ollama_unload_model()
                except Exception:
                    pass
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
        if vocal_result_present or singer_identity:
            revised += build_vocal_priority_note(priority_tag, initial_backing or "uncertain")

    revised = _collapse_runaway_repetition_fields(revised)
    status_done("Analysis complete")

    out = dict(heavy)
    out["analysis"] = revised
    out["singer_identity"] = singer_identity
    for k in ("vocal_audit_for_resolution", "final_lead", "initial_backing", "vocal_result_present"):
        out.pop(k, None)
    return out


def run_fresh_track_analysis(
    track_path,
    *,
    audio_temp_files,
    dsp_temp_files,
    stem_temp_files,
    demucs_out_dirs,
):
    """Back-compat wrapper: heavy analysis + identity resolution in one call,
    both stages in THIS process. Used by the non-isolated in-process batch
    fallback (BATCH_ISOLATE_PER_TRACK = False) and anywhere splitting into
    two OS processes isn't applicable. The (default) isolated batch path
    instead runs run_fresh_track_analysis_heavy() and
    resolve_identity_and_finalize() as two separate subprocess stages — see
    run_batch_one_analyze() / run_batch_one_finish() — so identity
    resolution's Ollama model load never has to coexist with this stage's
    stranded MF/Demucs/Omnizart memory.
    """
    heavy = run_fresh_track_analysis_heavy(
        track_path,
        audio_temp_files=audio_temp_files,
        dsp_temp_files=dsp_temp_files,
        stem_temp_files=stem_temp_files,
        demucs_out_dirs=demucs_out_dirs,
    )
    return resolve_identity_and_finalize(heavy)


def run_batch_one_track(track_path):
    """Analyse a single local audio file (heavy + identity, ONE process) and
    write JSON+cover into SAVE_DIR.

    Used when BATCH_SPLIT_IDENTITY_PROCESS is False. When it's True
    (default), the isolated batch path instead uses run_batch_one_analyze()
    + run_batch_one_finish() as two separate subprocess stages — see
    _batch_run_isolated().
    """
    track_path = os.path.abspath(os.path.expanduser(track_path))
    if not os.path.exists(track_path):
        raise FileNotFoundError(f"File not found: {track_path}")

    audio_temp_files = {}
    dsp_temp_files = {}
    stem_temp_files = {}
    demucs_out_dirs = []

    try:
        result = run_fresh_track_analysis(
            track_path,
            audio_temp_files=audio_temp_files,
            dsp_temp_files=dsp_temp_files,
            stem_temp_files=stem_temp_files,
            demucs_out_dirs=demucs_out_dirs,
        )
        save_name = _batch_save_basename(track_path, metadata=result.get("metadata"))
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
        print(
            f"  ✓ saved {os.path.basename(out_path)}"
            + (f" + {os.path.basename(cover_path)}" if cover_path else "")
        )
        return out_path
    finally:
        try:
            _cleanup_track_temp_files(
                track_path, audio_temp_files, dsp_temp_files, stem_temp_files, demucs_out_dirs
            )
        except Exception:
            pass
        try:
            _aggressive_memory_cleanup()
        except Exception:
            pass


def run_batch_one_analyze(track_path, staging_path):
    """Stage 1 of the split-process batch pipeline (BATCH_SPLIT_IDENTITY_PROCESS):
    run the heavy audio analysis (MF/Demucs/Omnizart/Essentia/PANNs) and dump
    the result to staging_path as JSON, then this process exits. Deliberately
    does NOT resolve singer identity here — see run_batch_one_finish(), which
    runs as a fresh, model-free process so it can't inherit any of this
    stage's stranded torch/TF/MPS memory.
    """
    track_path = os.path.abspath(os.path.expanduser(track_path))
    if not os.path.exists(track_path):
        raise FileNotFoundError(f"File not found: {track_path}")

    audio_temp_files = {}
    dsp_temp_files = {}
    stem_temp_files = {}
    demucs_out_dirs = []

    try:
        heavy = run_fresh_track_analysis_heavy(
            track_path,
            audio_temp_files=audio_temp_files,
            dsp_temp_files=dsp_temp_files,
            stem_temp_files=stem_temp_files,
            demucs_out_dirs=demucs_out_dirs,
        )
        payload = dict(heavy)
        cover_bytes = payload.pop("cover_bytes", None)
        payload["cover_bytes_b64"] = (
            base64.b64encode(cover_bytes).decode("ascii") if cover_bytes else None
        )
        with open(staging_path, "w", encoding="utf-8") as f:
            json.dump(payload, f)
    finally:
        try:
            _cleanup_track_temp_files(
                track_path, audio_temp_files, dsp_temp_files, stem_temp_files, demucs_out_dirs
            )
        except Exception:
            pass
        try:
            _aggressive_memory_cleanup()
        except Exception:
            pass


def run_batch_one_finish(staging_path):
    """Stage 2 of the split-process batch pipeline: load stage 1's JSON,
    resolve singer identity (a plain-text Ollama call — no Music Flamingo/
    Demucs/Omnizart/Essentia involved), and save. Runs in a fresh process
    that starts with a clean memory baseline, regardless of how much of
    stage 1's memory the OS was slow to reclaim.
    """
    with open(staging_path, "r", encoding="utf-8") as f:
        heavy = json.load(f)

    cover_b64 = heavy.pop("cover_bytes_b64", None)
    heavy["cover_bytes"] = base64.b64decode(cover_b64) if cover_b64 else None

    result = resolve_identity_and_finalize(heavy)

    save_name = _batch_save_basename(result["track_path"], metadata=result.get("metadata"))
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
    print(
        f"  ✓ saved {os.path.basename(out_path)}"
        + (f" + {os.path.basename(cover_path)}" if cover_path else "")
    )
    return out_path


def _batch_run_isolated(track_path):
    """Run one track through the isolated batch pipeline; returns (ok: bool, detail).

    When BATCH_SPLIT_IDENTITY_PROCESS is True (default), this spawns TWO
    fresh Python processes in sequence — one for the heavy MF/Demucs/
    Omnizart/Essentia analysis, one for singer-identity resolution + save —
    instead of one process doing both. The heavy stage's process exits
    completely between the two, which is the only fully reliable way to
    reset MPS/TF residual memory (in-process unload/gc/cache-clear is
    best-effort and can leave allocations stranded); the identity/save
    stage then starts from a clean baseline instead of asking Ollama to
    load the writer model on top of whatever the heavy stage couldn't
    fully release, which is what was causing "child exited -9" specifically
    at the "Resolving singer identity" step. Set BATCH_SPLIT_IDENTITY_PROCESS
    = False to fall back to the single-process-per-track behaviour.
    """
    script = os.path.abspath(__file__)
    env = os.environ.copy()
    # Child uses ephemeral Ollama loads (keep_alive=0) and unloads the writer
    # around heavy analysis. Parent chat history stays in this process's RAM.
    env["MUSICLYSE_BATCH_ONE"] = "1"
    env["MUSICLYSE_IN_BATCH"] = "1"
    try:
        # Free parent-side Ollama weights so the child can claim RAM for MF.
        if globals().get("BATCH_UNLOAD_OLLAMA", True):
            try:
                ollama_unload_model()
            except Exception:
                pass

        if globals().get("BATCH_SPLIT_IDENTITY_PROCESS", True):
            staging_fd, staging_path = tempfile.mkstemp(
                prefix="musiclyse_batch_", suffix=".json"
            )
            os.close(staging_fd)
            try:
                analyze_result = subprocess.run(
                    [sys.executable, script, "--batch-one-analyze", track_path, staging_path],
                    env=env,
                    timeout=None,
                )
                if analyze_result.returncode != 0:
                    return False, f"analysis stage exited {analyze_result.returncode}"

                # The analysis process has now fully exited, so the OS has
                # reclaimed 100% of its memory (unlike in-process cleanup,
                # which is best-effort). This finish stage re-imports the
                # module (cheap — no big model weights loaded) but starts
                # from that clean baseline before Ollama loads the writer.
                finish_result = subprocess.run(
                    [sys.executable, script, "--batch-one-finish", staging_path],
                    env=env,
                    timeout=None,
                )
                if finish_result.returncode != 0:
                    return False, f"identity/save stage exited {finish_result.returncode}"
                return True, "ok"
            finally:
                try:
                    os.remove(staging_path)
                except Exception:
                    pass

        # Single-process fallback (BATCH_SPLIT_IDENTITY_PROCESS = False).
        result = subprocess.run(
            [sys.executable, script, "--batch-one", track_path],
            env=env,
            timeout=None,
        )
        if result.returncode == 0:
            return True, "ok"
        return False, f"child exited {result.returncode}"
    except Exception as e:
        return False, str(e)


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

    When BATCH_ISOLATE_PER_TRACK is True (default), each track is analysed in a
    fresh subprocess so MPS/TensorFlow residual memory cannot accumulate. That
    is the recommended path for overnight runs on macOS.

    The interactive session's writer_history remains in the parent process RAM
    throughout; only the Ollama model weights are unloaded around analysis.
    """
    files = list_audio_files_in_folder(folder)
    if not files:
        print(f"  No audio files found in {folder}\n")
        return

    os.makedirs(SAVE_DIR, exist_ok=True)
    isolate = bool(BATCH_ISOLATE_PER_TRACK)
    os.environ["MUSICLYSE_IN_BATCH"] = "1"
    # Drop any resident writer model before the first track so analysis can use
    # the RAM. Session chat messages stay in the parent Python process.
    if globals().get("BATCH_UNLOAD_OLLAMA", True):
        try:
            status("Unloading Ollama writer for batch scan (session history kept)...")
            ollama_unload_model()
            status_done("Ollama writer unloaded — chat history still in memory")
        except Exception:
            pass
    split_identity = isolate and bool(globals().get("BATCH_SPLIT_IDENTITY_PROCESS", True))
    print(
        f"  Batch scan: {len(files)} track(s) in {folder}\n"
        f"  Saving to {os.path.abspath(SAVE_DIR)} (not imported into chat)\n"
        f"  Isolation: {'subprocess per track (recommended on macOS)' if isolate else 'in-process'}\n"
        f"  Identity stage: "
        f"{'separate process (BATCH_SPLIT_IDENTITY_PROCESS)' if split_identity else 'same process as analysis'}\n"
        f"  Ollama during batch: "
        f"{'unload between stages (BATCH_UNLOAD_OLLAMA)' if BATCH_UNLOAD_OLLAMA else 'left loaded'}\n"
    )

    ok, skipped, failed = 0, 0, 0
    for i, path in enumerate(files, 1):
        # Peek tags so the skip-check uses the same Artist - Title naming
        # that the actual save will use after analysis.
        peek_meta = {}
        if ENABLE_FILE_METADATA:
            try:
                peek_meta, _, _ = extract_audio_metadata(path)
            except Exception:
                peek_meta = {}
        save_name = _batch_save_basename(path, metadata=peek_meta)
        dest = os.path.join(SAVE_DIR, save_name)
        label = os.path.basename(path)
        print(f"\n  [{i}/{len(files)}] {label}")

        if skip_existing and os.path.exists(dest):
            print(f"  (skip — already saved as {save_name})")
            skipped += 1
            continue

        if isolate:
            success, detail = _batch_run_isolated(path)
            if success:
                ok += 1
            else:
                failed += 1
                print(f"  ✗ failed: {detail}")
            # Short pause even with isolation — lets the OS reclaim the dead
            # child's pages and reduces pressure spikes on the next launch.
            pause = float(BATCH_PAUSE_BETWEEN_TRACKS_S or 0)
            if pause > 0 and i < len(files):
                status(f"Batch pause {pause:.0f}s (letting RAM settle)...")
                try:
                    import time as _time
                    _time.sleep(pause)
                except Exception:
                    pass
                status_done()
            continue

        # In-process fallback (BATCH_ISOLATE_PER_TRACK = False)
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
        finally:
            try:
                _cleanup_track_temp_files(
                    path, audio_temp_files, dsp_temp_files, stem_temp_files, demucs_out_dirs
                )
            except Exception:
                pass
            try:
                _aggressive_memory_cleanup()
            except Exception:
                pass
            pause = float(BATCH_PAUSE_BETWEEN_TRACKS_S or 0)
            if pause > 0 and i < len(files):
                status(f"Batch pause {pause:.0f}s (letting RAM settle)...")
                try:
                    import time as _time
                    _time.sleep(pause)
                except Exception:
                    pass
                status_done()

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
    track_wiki_context = {}        # raw_path -> cached WIKIPEDIA BACKGROUND CONTEXT block (or "")
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
        f"'{LISTEN_FLAG} <path or URL> <question>' switches tracks first.\nAudio windows can be limited with [start-end], e.g. '{LISTEN_FLAG} song.mp3 [30-60] analyse the solo'.\n"
        f"'{RELISTEN_FLAG} [<path or URL>] <question>' forces a fresh re-analysis instead of using the cache.\n"
        f"'{CORRECT_FLAG} field=value[, field=value...]' records a confirmed fact for the current "
        f"track (e.g. '{CORRECT_FLAG} year=1966') that overrides the analysis from then on.\n"
        f"'{SAVE_FLAG}=filename.json' saves technical details for the most recently scanned track to {SAVE_DIR}/ "
        f"(spaces and most punctuation are allowed in the name).\n"
        f"'{LOAD_FLAG} filename.json [question]' loads a saved song; optional question after the name. "
        f"Quoted names and names with spaces work (e.g. /load \"Artist - Song.json\").\n"
        f"'{LOADCOMPARE_FLAG} \"track 1.json\" \"track 2.json\" question' loads two or more saved "
        f"songs together in one go and asks a question about all of them "
        f"(e.g. /loadcompare \"Radiohead - Creep.json\" \"Nirvana - Lithium.json\" compare their choruses).\n"
        f"'{BATCH_FLAG} /path/to/folder' overnight-scans every audio file in that folder into {SAVE_DIR}/ "
        f"as 'Artist Name - Song Name.json' (from file tags; Unknown + original filename if tags missing). "
        f"Same analysis as /listen; does NOT import into chat. Skips files already saved. "
        f"Default: one subprocess per track to avoid macOS MPS OOM kills on long batches.\n"
        f"'{CLEAR_FLAG}' wipes chat context and resets session token counters "
        f"(track analysis cache is kept so you don't re-scan).\n"
        f"'{PERSONA_FLAG} <description>' sets a custom chat persona (taste/voice); music evidence rules stay.\n"
        f"'{PERSONA_FLAG} reset' (or default) restores the music-obsessed friend. "
        f"'{PERSONA_FLAG}' alone shows the current persona.\n"
        + (f"Starting track: {current_track}\n" if current_track else "No starting track yet — set one with /listen.\n")
        + "Type 'quit' to exit.\n",
        Ansi.YELLOW,
    ))

    def _parse_top_level_paren_groups(text):
        """Split text like '("a") ("b") (c)' into ['a', 'b', 'c'].

        Used by /loadcompare. Honors nesting (parens inside a group, e.g. a
        track named "Song (Live).json") and quotes (a paren inside quotes
        doesn't count toward depth). A group that is entirely wrapped in one
        layer of matching quotes has those quotes stripped. Returns None if
        the text isn't cleanly a sequence of top-level (...) groups (e.g.
        stray text outside any parens, or unbalanced parens) so the caller
        can fall back to a clear usage error.
        """
        groups = []
        i, n = 0, len(text)
        while i < n:
            while i < n and text[i] in " \t\r\n":
                i += 1
            if i >= n:
                break
            if text[i] != "(":
                return None
            depth = 0
            in_quote = None
            j = i
            while j < n:
                c = text[j]
                if in_quote:
                    if c == in_quote:
                        in_quote = None
                elif c in ("'", '"'):
                    in_quote = c
                elif c == "(":
                    depth += 1
                elif c == ")":
                    depth -= 1
                    if depth == 0:
                        break
                j += 1
            if depth != 0 or j >= n:
                return None
            inner = text[i + 1:j].strip()
            if len(inner) >= 2 and inner[0] == inner[-1] and inner[0] in ("'", '"'):
                inner = inner[1:-1].strip()
            groups.append(inner)
            i = j + 1
        return groups if groups else None

    def _parse_loadcompare_args_free(text):
        """Parse '/loadcompare' arguments without requiring parentheses.

        Saved filenames are matched against the actual files in SAVE_DIR so
        apostrophes and other printable punctuation are never mistaken for
        command quoting. Whatever text remains after the last filename is the
        comparison question.
        """
        names = []
        rest = text

        while True:
            matched = _match_saved_filename_prefix(rest)
            if matched is None:
                break
            name, trailing = matched
            names.append(name)
            rest = trailing

        query = rest.strip()
        if len(names) < 2 or not query:
            return None
        return names, query

    def _load_saved_track_into_session(filename, question=None):
        """Load one saved-song JSON off disk into this session (analysis
        cache, metadata, cover art, singer identity, wiki context) and
        register a loaded_message in writer_history so the writer model can
        discuss it. Shared by /load (one track) and /loadcompare (two+
        tracks loaded back-to-back before a single combined question).

        `question` is the same optional hint /load's trailing prompt gives
        _get_or_build_wiki_context — for /loadcompare this is the comparison
        question, used for every track being loaded.

        Returns (key, label) on success, or None on failure (after printing
        the same failure messages /load has always printed).
        """
        try:
            data = load_song_data(filename)
        except Exception as e:
            print(f"  Load failed: {e}\n")
            return None

        if not data:
            print(f"  Saved song not found in {SAVE_DIR}/ for '{filename}'.\n")
            return None

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

        # Same wiki background-context lookup as /listen (cached per track, or
        # re-looked-up fresh per question if WIKI_CONTEXT_REFRESH_EVERY_QUESTION
        # is on) — folded in here too so /load (and /loadcompare) get it just
        # like /listen does.
        loaded_wiki_block = _get_or_build_wiki_context(
            key, track_metadata, track_wiki_context, question=question
        )

        loaded_message = {
            "role": "user",
            "content": (
                f"[Loaded saved track '{label}']\n"
                "(background technical details restored from a previous scan; use them for discussion):\n"
                f"{data['analysis']}\n"
                f"{loaded_metadata_block}{extra_loaded_context}{loaded_cover_note}{loaded_wiki_block}\n"
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

        return key, label

    try:
        while True:
            user_text = colored_input("You: ", Ansi.LIGHT_GREEN).strip()
            if user_text.lower() in ("quit", "exit"):
                break
            if not user_text:
                continue

            _compact_writer_history_in_place(writer_history)
            
            
            clear_all_rem = _command_remainder(user_text, CLEAR_ALL_FLAG)
            if clear_all_rem is not None:
                # Full in-memory reset: conversation + all analysis caches.
                # Saved song JSON/cover files on disk are left untouched so
                # /load still works after a session wipe.
                writer_history.clear()
                writer_history.append({
                    "role": "system",
                    "content": build_writer_system_prompt(active_persona_text),
                })
                track_evidence_message.clear()
                track_wiki_context.clear()
                comprehensive_analyses.clear()
                track_metadata.clear()
                track_cover_b64.clear()
                track_cover_observations.clear()
                track_singer_identity.clear()
                current_track = None
                last_scanned_track = None
                last_writer_message = None
                SESSION_TOKEN_USAGE["prompt"] = 0
                SESSION_TOKEN_USAGE["completion"] = 0
                SESSION_TOKEN_USAGE["total"] = 0
                SESSION_TOKEN_USAGE["last_prompt"] = 0
                SESSION_TOKEN_USAGE["last_completion"] = 0
                SESSION_TOKEN_USAGE["last_ctx"] = 0

                print(
                    "  Full cache cleared: conversation and in-memory track analyses wiped. "
                    f"Saved files in {SAVE_DIR}/ were left untouched. "
                    f"Persona: {active_persona_label}.\n"
                )
                continue

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
                # Wikipedia context is a question-sensitive retrieval cache, not
                # analysis evidence. Keeping it across /clear meant that the first
                # question asked about a track could permanently decide which
                # articles were available for all later questions in that session.
                # A fresh chat should allow fresh entity extraction and retrieval.
                track_wiki_context.clear()
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
                # Support: /load name.json
                #          /load 'Artist Name - Song.json'
                #          /load Artist Name - Song.json what about the chorus?
                # Optional prompt after the filename is sent to the writer like /listen.
                # Filenames may contain spaces, apostrophes, parentheses, dashes, etc.
                load_rest = load_rem.strip()
                load_prompt = ""
                filename = load_rest

                # Prefer an exact match against the actual saved filename.
                # This keeps apostrophes and other printable punctuation intact.
                saved_match = _match_saved_filename_prefix(load_rest)
                if saved_match is not None:
                    filename, load_prompt = saved_match
                else:
                    # 1) Quoted filename (handles internal spaces / special chars cleanly).
                    #    Filenames may themselves contain the quote character used to
                    #    wrap them (e.g. "Guns N' Roses - Don't Stop.json" wrapped in
                    #    single quotes, per the help text above). A naive [^']+ style
                    #    match stops at the FIRST internal apostrophe and truncates the
                    #    filename, so instead: prefer the closing quote that immediately
                    #    follows ".json" (the common case), and only fall back to the
                    #    LAST quote character in the string if no ".json"+quote boundary
                    #    is found (e.g. a quoted name with no extension typed).
                    quoted = False
                    if load_rest[:1] in ("'", '"'):
                        q = load_rest[0]
                        body = load_rest[1:]
                        close_m = re.search(r"\.json" + re.escape(q), body, re.IGNORECASE)
                        close_idx = (close_m.end() - 1) if close_m else body.rfind(q)
                        if close_idx >= 0:
                            filename = body[:close_idx].strip()
                            load_prompt = body[close_idx + 1:].strip()
                            quoted = True
                    if quoted:
                        pass
                    else:
                        # 2) Unquoted: take everything up to and including the first
                        #    ".json" token as the filename; remainder is the question.
                        json_m = re.search(r"\.json\b", load_rest, re.IGNORECASE)
                        if json_m:
                            filename = load_rest[: json_m.end()].strip().strip('"').strip("'")
                            load_prompt = load_rest[json_m.end():].strip()
                        else:
                            # 3) No .json in the text — try shlex (single token) then
                            #    fall back to the whole remainder as the name.
                            try:
                                load_tokens = shlex.split(load_rest, posix=True)
                            except ValueError:
                                load_tokens = load_rest.split() if load_rest else []
                            if load_tokens:
                                # Prefer longest prefix of tokens that matches a saved file.
                                matched_name = None
                                matched_n = 0
                                try:
                                    saved = [
                                        f for f in os.listdir(SAVE_DIR)
                                        if f.lower().endswith(".json")
                                    ] if os.path.isdir(SAVE_DIR) else []
                                except Exception:
                                    saved = []
                                saved_l = {f.lower(): f for f in saved}
                                for n in range(len(load_tokens), 0, -1):
                                    cand = " ".join(load_tokens[:n]).strip('"').strip("'")
                                    cand_json = cand if cand.lower().endswith(".json") else cand + ".json"
                                    if cand_json.lower() in saved_l or cand.lower() in saved_l:
                                        matched_name = saved_l.get(
                                            cand_json.lower(),
                                            saved_l.get(cand.lower(), cand_json),
                                        )
                                        matched_n = n
                                        break
                                if matched_name:
                                    filename = matched_name
                                    load_prompt = " ".join(load_tokens[matched_n:]).strip()
                                else:
                                    filename = load_tokens[0].strip('"').strip("'")
                                    load_prompt = " ".join(load_tokens[1:]).strip()
                            else:
                                filename = load_rest.strip('"').strip("'")

                result = _load_saved_track_into_session(filename, question=(load_prompt or None))
                if result is None:
                    continue
                key, label = result
                current_track = key
                last_scanned_track = key

                if load_prompt:
                    # Optional trailing question, same idea as /listen path + question.
                    _load_depth_requested = bool(
                        re.search(
                            r"\b(in[\s-]?depth|in\s+detail|elaborate|tell me more|go deeper|"
                            r"more detail|full breakdown|deep dive|thorough|comprehensive|"
                            r"everything (you'?ve got|you know)|expand on|say more|break (it|that|this) down|"
                            r"walk me through|long answer|really get into)\b",
                            load_prompt,
                            re.IGNORECASE,
                        )
                    )
                    _load_reply_style_note = (
                        "Reply as their music buddy in the conversation. They asked for depth here — "
                        "give a genuinely long, detailed answer that actually goes into it, not a short "
                        "summary with a token gesture toward length."
                        if _load_depth_requested else
                        "Reply as their music buddy in the conversation — answer what they asked, "
                        "not a full analytical write-up unless they asked for one."
                    )
                    writer_message = {
                        "role": "user",
                        "content": (
                            f"Regarding the loaded track '{label}': {load_prompt}"
                            "\n\nIf what they're asking relates to something already discussed earlier in this "
                            "conversation — comparing this to another track, continuing a topic, reacting to an "
                            "earlier point — use those earlier turns to answer that directly. Don't just give a "
                            "standalone description of this track in isolation when they asked about a connection "
                            "to something already talked about.\n\n"
                            f"{_load_reply_style_note}"
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

            loadcompare_rem = _command_remainder(user_text, LOADCOMPARE_FLAG)
            if loadcompare_rem is not None:
                # Support (parentheses optional, same spirit as /load):
                #   /loadcompare "track 1.json" "track 2.json" question
                #   /loadcompare track 1.json track 2.json question
                #   /loadcompare ("track 1.json") ("track 2.json") (question)   -- still works
                # At least two tracks, then one trailing question. Quotes around a
                # name are optional but recommended for names containing spaces;
                # without quotes, each name just needs to end in '.json' so its
                # boundary is unambiguous.
                usage_msg = (
                    f"  Usage: {LOADCOMPARE_FLAG} \"track 1.json\" \"track 2.json\" question"
                    f"  — at least two tracks (quoted, or ending in .json), then one question.\n"
                )
                loadcompare_rest = loadcompare_rem.strip()
                # Try the original parenthesized form first — it declines immediately
                # (returns None) unless the text actually starts with '(', so this is
                # a safe first attempt and won't misfire on paren-free input.
                groups = _parse_top_level_paren_groups(loadcompare_rest)
                if groups and len(groups) >= 3:
                    *compare_names, compare_query = groups
                    compare_names = [n for n in compare_names if n.strip()]
                    compare_query = compare_query.strip()
                    if len(compare_names) < 2 or not compare_query:
                        compare_names = None
                else:
                    compare_names = None

                if not compare_names:
                    parsed = _parse_loadcompare_args_free(loadcompare_rest)
                    if parsed is None:
                        print(usage_msg)
                        continue
                    compare_names, compare_query = parsed

                compare_keys, compare_labels = [], []
                load_ok = True
                for name in compare_names:
                    result = _load_saved_track_into_session(name.strip(), question=compare_query)
                    if result is None:
                        load_ok = False
                        break
                    key, label = result
                    compare_keys.append(key)
                    compare_labels.append(label)

                if not load_ok or len(compare_keys) < 2:
                    print("  /loadcompare aborted — could not load all of the requested tracks.\n")
                    continue

                current_track = compare_keys[-1]
                last_scanned_track = compare_keys[-1]

                labels_joined = ", ".join(f"'{l}'" for l in compare_labels[:-1]) + f" and '{compare_labels[-1]}'"
                writer_message = {
                    "role": "user",
                    "content": (
                        f"Now compare the tracks just loaded — {labels_joined} — and answer this: "
                        f"{compare_query}\n\n"
                        "Use the technical details already given for each of these tracks above; don't ask "
                        "for anything further before answering. When comparing, compare like with like "
                        "(tempo vs tempo, key vs key, LUFS vs LUFS, etc.) rather than vague impressions. "
                        "Reply as their music buddy in the conversation — answer what they asked, not a "
                        "full analytical write-up of each track individually unless they asked for one."
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

                # Tell the LLM immediately, so it updates rather than waiting for the next /listen turn
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
                # If /listen is accidentally given the name of a saved JSON,
                # treat it as /load rather than silently reusing the current
                # track's cached analysis.
                saved_match = _match_saved_filename_prefix(listen_rem)
                if saved_match is not None:
                    saved_filename, saved_prompt = saved_match
                    result = _load_saved_track_into_session(
                        saved_filename, question=(saved_prompt or None)
                    )
                    if result is None:
                        continue
                    key, label = result
                    current_track = key
                    last_scanned_track = key

                    if saved_prompt:
                        _load_depth_requested = bool(
                            re.search(
                                r"\b(in[\s-]?depth|in\s+detail|elaborate|tell me more|go deeper|"
                                r"more detail|full breakdown|deep dive|thorough|comprehensive|"
                                r"everything (you'?ve got|you know)|expand on|say more|break (it|that|this) down|"
                                r"walk me through|long answer|really get into)\b",
                                saved_prompt,
                                re.IGNORECASE,
                            )
                        )
                        _load_reply_style_note = (
                            "Reply as their music buddy in the conversation. They asked for depth here — "
                            "give a genuinely long, detailed answer that actually goes into it, not a short "
                            "summary with a token gesture toward length."
                            if _load_depth_requested else
                            "Reply as their music buddy in the conversation — answer what they asked, "
                            "not a full analytical write-up unless they asked for one."
                        )
                        writer_message = {
                            "role": "user",
                            "content": (
                                f"Regarding the loaded track '{label}': {saved_prompt}"
                                "\n\nIf what they're asking relates to something already discussed earlier in this "
                                "conversation — comparing this to another track, continuing a topic, reacting to an "
                                "earlier point — use those earlier turns to answer that directly. Don't just give a "
                                "standalone description of this track in isolation when they asked about a connection "
                                "to something already talked about.\n\n"
                                f"{_load_reply_style_note}"
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

                remainder = listen_rem
                remainder, listen_segment = parse_listen_segment(remainder)
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
                if listen_segment:
                    print(f"  (analysing only audio segment {listen_segment[0]:g}s–{listen_segment[1]:g}s)")
                    # Segments are different analyses from the full-track cache.
                    # Do not accidentally reuse the cached whole-song result.
                    force_fresh = True
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
                    # vocal, confirmation, lyrics). Free the output LLM from Ollama's memory first —
                    # it isn't needed again until we're back to writing the final answer —
                    # then lazily load Music Flamingo (a no-op if it's already loaded from
                    # a call earlier in this same batch).
                    ollama_unload_model()
                    mf_model, mf_processor, mf_device = get_music_flamingo()

                    if listen_segment and not current_track.startswith(("http://", "https://")):
                        resolved_path = crop_audio_segment(
                            resolved_path, listen_segment[0], listen_segment[1]
                        )

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
                    first_pass, mf_conversation, main_max_tokens = _mf_run_main_analysis(
                        mf_model, mf_processor, resolved_path, deep=DEEP_MODE
                    )

                    objective_report = ""
                    vocal_objective_report = ""
                    essentia_report = ""
                    genre_mood_report = ""
                    panns_genre_tags = []
                    dsp_path = None
                    precomputed_stems, precomputed_demucs_out_dir = {}, None
                    vocal_pitch_source = "full mix (no isolated vocal stem available)"
                    if not current_track.startswith(("http://", "https://")):
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

                        # Isolate vocals via Demucs *before* pitch tracking, when stem
                        # separation is enabled, so pyin locks onto the singer instead
                        # of a sustained bass/pad/guitar drone in the full mix. Reused
                        # below for the stem MIDI report so Demucs only runs once.
                        if ENABLE_VOCAL_OBJECTIVE_REPORT and ENABLE_STEM_MIDI:
                            precomputed_stems, precomputed_demucs_out_dir = _prepare_demucs_stems_for_track(
                                current_track, stem_temp_files, demucs_out_dirs,
                                status_fn=status, deep_mode=DEEP_MODE,
                            )
                            if precomputed_stems.get("vocals"):
                                vocal_pitch_source = "isolated vocal stem (Demucs)"

                        if dsp_path is not None:
                            if ENABLE_OBJECTIVE_AUDIO_REPORT:
                                status("Measuring beat/timbre with signal processing...")
                                objective_report = build_objective_audio_report(dsp_path)
                            if ENABLE_VOCAL_OBJECTIVE_REPORT:
                                status("Measuring vocal pitch/formant proxies...")
                                _pitch_source_path = precomputed_stems.get("vocals") or dsp_path
                                vocal_objective_report = build_vocal_objective_report(_pitch_source_path)
                                if vocal_objective_report:
                                    vocal_objective_report += f"\nmeasurement source: {vocal_pitch_source}"
                            if ENABLE_ESSENTIA_REPORT and ESSENTIA_AVAILABLE:
                                status("Measuring tempo/key/spectral features with Essentia...")
                                essentia_report = build_essentia_report(dsp_path)
                            if ENABLE_GENRE_MOOD_TAGGING:
                                status("Cross-checking genre/mood with an independent classifier...")
                                try:
                                    genre_mood_report, panns_genre_tags = build_genre_mood_signal_report(dsp_path)
                                except Exception as e:
                                    print(f"  (genre/mood signal skipped: {e})")
                                    genre_mood_report = ""
                                    panns_genre_tags = []

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
                        elif initial_lead in (MALE_LEAD_CATEGORIES | ADOLESCENT_MALE_CATEGORIES) and (
                            median_f0 is not None
                            and F0_MALE_HIGH_CONFIRM_HZ is not None
                            and median_f0 >= float(F0_MALE_HIGH_CONFIRM_HZ)
                        ):
                            # High track-wide median on a male-tagged lead is a common
                            # female→male mislabel path; run confirmation before locking in.
                            should_confirm = True
                        elif initial_lead == "mixed_leads":
                            should_confirm = True
                        else:
                            _mv_early = parse_multi_voice_fields(vocal_result or "")
                            if _mv_early.get("voice_arrangement") in ("duet_co_leads", "call_response"):
                                should_confirm = True
                            elif _mv_early.get("num_distinct_voices") in ("2", "3", "3+", "two", "three"):
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
                        initial_lead = _apply_vocal_age_guard(initial_lead, vocal_result)
                    confirm_lead = _apply_vocal_age_guard(confirm_lead, confirmation_result)
                    final_lead = choose_final_vocal_lead(
                        initial_lead, confirm_lead, confirm_confidence, confirmation_result,
                        median_f0=median_f0,
                    )
                    final_lead = _apply_vocal_age_guard(
                        final_lead,
                        (vocal_result or "") + "\n" + (confirmation_result or ""),
                    )
                    final_lead = _apply_f0_pitch_guard(
                        final_lead,
                        median_f0,
                        low_f0=vocal_pitch.get("low") if isinstance(vocal_pitch, dict) else None,
                        high_f0=vocal_pitch.get("high") if isinstance(vocal_pitch, dict) else None,
                    )



                    first_pass = _mf_salvage_empty_analysis(
                        first_pass, objective_report, essentia_report, vocal_result
                    )

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
                                "If the lead is identified as young_male, child_male_likely, child_female_likely, or child_gender_uncertain, do not leave a confident female-lead claim in place."
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
                                "do NOT use librosa/Essentia spectral stats to invent GENRE or change vocal identity):\n"
                                + "\n\n".join(objective_crosscheck_parts)
                            )
                        if genre_mood_report:
                            self_check_text += (
                                "\n\nIndependent genre/mood classifier (PANNs/AudioSet) — this MAY be used "
                                "to revise GENRE_RANKED when it clearly conflicts with a rock/pop-punk top rank "
                                "driven mainly by guitar texture while this signal is electronic/dance/synth-pop. "
                                "See GENRE self-check rules.\n"
                                + genre_mood_report
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
                            max_new_tokens=1280,
                            repetition_penalty=1.55,
                            no_repeat_ngram_size=14,
                        )
                        full_lyrics = _sanitize_lyrics_transcription(full_lyrics)
                        if full_lyrics and full_lyrics.strip():
                            revised += (
                                f"\n\nFULL LYRICS TRANSCRIPTION (dedicated pass):\n{full_lyrics}"
                            )

                    # This was the last Music Flamingo pass in this batch (stem/MIDI below
                    # uses Demucs/Omnizart, not Music Flamingo; singer-identity resolution and
                    # the final written answer use the output LLM via Ollama). Free its memory now
                    # rather than holding it resident for the rest of the turn.
                    unload_music_flamingo()
                    mf_model = mf_processor = None

                    revised += f"\n\n11. ERA / RELEASE PERIOD (isolated dedicated pass):\n{era_result}"
                    if vocal_result:
                        revised += f"\n\nVOCAL / SINGER PROFILE (isolated dedicated pass):\n{vocal_result}"
                        if confirmation_result:
                            revised += f"\n\nVOCAL CONFIRMATION PASS:\n{confirmation_result}"

                        f0_text = f"{median_f0} Hz" if median_f0 is not None else "unavailable"
                        multi_voice_fields = parse_multi_voice_fields(vocal_result or "")
                        if confirmation_result:
                            conf_mv = parse_multi_voice_fields(confirmation_result)
                            for k, v in conf_mv.items():
                                if v and (not multi_voice_fields.get(k) or multi_voice_fields.get(k) in ("", "unparsed", "uncertain")):
                                    multi_voice_fields[k] = v
                        if (final_lead or initial_lead) == "mixed_leads" and not multi_voice_fields.get("voice_arrangement"):
                            multi_voice_fields["voice_arrangement"] = "duet_co_leads"
                            if not multi_voice_fields.get("num_distinct_voices"):
                                multi_voice_fields["num_distinct_voices"] = "2"
                        revised += (
                            "\n\nVOCAL DECISION AUDIT (audio-only evidence for vocal age/gender):\n"
                            f"- Initial isolated lead profile: {initial_lead or 'unparsed'}\n"
                            f"- Confirmation pass: {confirm_lead or 'not run/unparsed'} (confidence={confirm_confidence or 'n/a'})\n"
                            f"- Objective median f0: {f0_text}\n"
                            f"- FINAL LEAD PROFILE: {final_lead or 'unknown'}\n"
                            f"- BACKING PROFILES: {initial_backing or 'uncertain'}\n"
                            "This is audio-only evidence. If a SINGER IDENTITY RESOLUTION block appears later in this analysis, use that for user-facing singer-identity claims; otherwise use FINAL LEAD PROFILE. Do not override a well-supported combined judgment with pitch impressions alone. If FINAL LEAD PROFILE is unknown/unparsed, do not treat free-text LEAD_PROFILE / LEAD_CATEGORY lines earlier in the vocal pass as a settled gender/age claim — those lines were not accepted as structured final tags. A very high objective median f0 can demote male/adolescent tags to uncertain; when FINAL is uncertain, prefer SINGER IDENTITY RESOLUTION (metadata + cover + pitch constraint) over any earlier free-text male claim."
                        )
                        mv_block = format_multi_voice_audit(
                            multi_voice_fields, final_lead or initial_lead, initial_backing or "uncertain"
                        )
                        if mv_block:
                            revised += "\n\n" + mv_block

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
                            "- Prefer median + 5–95 percentile range for the main vocal range. Median f0 is also used as a soft constraint on lead age/gender tags (high median can demote an overconfident post-puberty-male claim; it does not invent gender alone). "
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
                    essentia_median_bpm_val = extract_essentia_median_bpm(essentia_report) if essentia_report else None
                    objective_median_bpm_val = extract_objective_median_bpm(objective_report) if objective_report else None

                    _bpm_genre_hint = _genre_hint_from_analysis_text(
                        revised or first_pass,
                        panns_genre_tags,
                    )
                    final_bpm, bpm_note = reconcile_bpm(
                        mf_bpm_val, essentia_bpm_val, objective_bpm_val,
                        essentia_median_bpm=essentia_median_bpm_val,
                        objective_median_bpm=objective_median_bpm_val,
                        genre_hint=_bpm_genre_hint,
                    )

                    if final_bpm:
                        revised += (
                            f"\n\nRECOMMENDED TEMPO FOR DISCUSSION: {final_bpm} BPM. "
                            f"Reasoning: {bpm_note}. "
                            "This is the primary tempo to report to the user. "
                            "State it as a concrete figure (e.g. 'about 158 BPM'). "
                            "Do not expand it into a range unless this block itself marks the value as uncertain."
                        )
                    # -----------------------------------------------

                    # --- Reconcile KEY from MF and Essentia ---
                    mf_key_val = extract_key_from_text(first_pass)
                    revised_key_val = extract_key_from_text(revised)
                    if revised_key_val is not None:
                        mf_key_val = revised_key_val
                    essentia_key_val, essentia_key_strength = (None, None)
                    if essentia_report:
                        parsed = extract_essentia_key_from_text(essentia_report)
                        if parsed:
                            essentia_key_val, essentia_key_strength = parsed

                    final_key, key_note = reconcile_key(mf_key_val, essentia_key_val, essentia_key_strength)

                    if final_key:
                        revised += (
                            f"\n\nRECOMMENDED KEY FOR DISCUSSION: {final_key}. "
                            f"Reasoning: {key_note} "
                            "This is the primary key to report to the user unless the reasoning above marks it uncertain."
                        )
                    # -----------------------------------------------

                    # Numeric dynamics / loudness — prefer ORIGINAL local file (full
                    # duration, native rate/channels). Ignore ReplayGain tags.
                    crest_db_val = None
                    crest_src = "objective (librosa) crest-factor proxy"
                    lufs_val = lra_val = loudness_src = None
                    try:
                        _dyn_path = None
                        if (
                            current_track
                            and not str(current_track).startswith(("http://", "https://"))
                            and os.path.exists(current_track)
                        ):
                            _dyn_path = current_track
                        elif current_track in dsp_temp_files:
                            _dyn_path = dsp_temp_files[current_track]
                        elif "dsp_path" in locals() and dsp_path is not None:
                            _dyn_path = dsp_path
                        if _dyn_path:
                            crest_db_val = measure_crest_factor_db(_dyn_path)
                            lufs_val, lra_val, loudness_src = measure_ebur128_loudness(_dyn_path)
                    except Exception:
                        crest_db_val = None
                    if crest_db_val is None:
                        crest_db_val = extract_crest_db_from_text(objective_report) if objective_report else None
                    if crest_db_val is None and essentia_report:
                        crest_db_val = extract_crest_db_from_text(essentia_report)
                        if crest_db_val is not None:
                            crest_src = "Essentia crest-factor proxy"
                    dyn_block = format_recommended_dynamics_block(
                        crest_db=crest_db_val,
                        source_note=crest_src,
                        lufs=lufs_val,
                        lra=lra_val,
                        loudness_source=loudness_src,
                    )
                    if dyn_block:
                        revised += dyn_block
                    else:
                        revised += (
                            "\n\nRECOMMENDED DYNAMICS FOR DISCUSSION: unavailable "
                            "(could not compute EBU R128 loudness or crest-factor proxy from this file). "
                            "Do not invent a dB/LUFS figure; fall back only to qualitative production notes."
                        )

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
                            + "\nThis is computed directly from the audio (librosa/Essentia), not from the model above — "
                            "use it only for tempo/beat, key/key strength when explicitly reported, timbre, "
                            "low/mid/high element activity, dynamic range / loudness (crest-factor proxy), "
                            "and vocal pitch/formant proxies. Do NOT use it to infer or revise GENRE."
                        )

                    if genre_mood_report:
                        revised += "\n\n" + genre_mood_report
                    try:
                        _rec_g, _rec_g_note = reconcile_genre(revised, panns_genre_tags)
                        revised += format_recommended_genre_block(_rec_g, _rec_g_note)
                        if _rec_g and "dsp_path" in locals() and dsp_path is not None:
                            _prev_k = extract_essentia_key_from_text(essentia_report) if essentia_report else None
                            _pk, _ps = (None, None)
                            if _prev_k:
                                _pk, _ps = _prev_k if isinstance(_prev_k, tuple) else (_prev_k, None)
                            _nk, _ns, _nnote = refine_essentia_key_with_genre(
                                dsp_path, _rec_g, previous_key=_pk, previous_strength=_ps
                            )
                            if _nnote:
                                revised += f"\n\nKEY PROFILE REFINEMENT: {_nnote}"
                            if _nk and _nnote and "shifted" in _nnote:
                                _mfk = extract_key_from_text(revised) or extract_key_from_text(first_pass)
                                _fk, _fnote = reconcile_key(_mfk, _nk, _ns)
                                if _fk:
                                    revised += (
                                        f"\n\nRECOMMENDED KEY FOR DISCUSSION: {_fk}. "
                                        f"Reasoning: {_fnote} (after genre-conditioned Essentia refinement). "
                                        "This is the primary key to report to the user unless the reasoning marks it uncertain."
                                    )
                    except Exception:
                        pass

                    # Optional Demucs 6s + Omnizart stem MIDI report.
                    # Essentia and Music Flamingo above still used only the original track.
                    stem_midi_report = ""
                    if ENABLE_STEM_MIDI and not current_track.startswith(("http://", "https://")):
                        try:
                            _get_omnizart()  # fail fast before slow Demucs run

                            if precomputed_stems:
                                # Reuse the separation already run above for
                                # vocal pitch isolation -- avoids a second,
                                # redundant Demucs pass.
                                stems = precomputed_stems
                            else:
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
                                stems = run_demucs_stems(
                                    stem_wav, out_dir,
                                    shifts=DEMUCS_SHIFTS_DEEP if DEEP_MODE else DEMUCS_SHIFTS_FAST,
                                )

                            if not stems:
                                stem_midi_report = "STEM MIDI REPORT unavailable: Demucs did not produce expected stems."
                            else:
                                whole_mix_tags = None
                                if ENABLE_WHOLE_MIX_INSTRUMENT_TAGGING and dsp_path is not None:
                                    try:
                                        status("Tagging instruments on the full (un-separated) mix...")
                                        whole_mix_tags = tag_full_mix_instruments(dsp_path)
                                    except Exception as e:
                                        print(f"  (whole-mix instrument tagging skipped: {e})")
                                        whole_mix_tags = None
                                status("Running Omnizart on each separated stem...")
                                _structure_sections = extract_structure_sections(revised or first_pass)
                                stem_midi_report = build_omnizart_summaries(
                                    stems, whole_mix_tags=whole_mix_tags,
                                    bpm=final_bpm, sections=_structure_sections,
                                )
                                if whole_mix_tags is not None:
                                    whole_mix_report = build_whole_mix_instrument_report(
                                        dsp_path, tags=whole_mix_tags
                                    )
                                    if whole_mix_report:
                                        stem_midi_report = (
                                            stem_midi_report + "\n\n" + whole_mix_report
                                        )
                                        try:
                                            _gan = guitar_absence_note_from_tags(
                                                locals().get("whole_mix_tags"), whole_mix_report
                                            )
                                            if _gan:
                                                stem_midi_report = stem_midi_report + "\n\n" + _gan
                                        except Exception:
                                            pass
                                _release_omnizart_memory()

                        except Exception as e:
                            print(f"  (stem MIDI skipped/unavailable: {e})")
                            stem_midi_report = f"STEM MIDI REPORT unavailable: {e}"

                    # Always append the stem MIDI report to revised, even if unavailable,
                    # so the output LLM knows the status and it's logged in the save file.
                    stem_midi_report = (
                        stem_midi_report
                        or "STEM MIDI REPORT not run (disabled or non-local track)."
                    )

                    track_stem_midi_report[current_track] = stem_midi_report

                    # Include the report in the saved/comprehensive analysis once.
                    revised += "\n\n" + stem_midi_report

                    if (
                        ENABLE_WHOLE_MIX_INSTRUMENT_TAGGING
                        and dsp_path is not None
                        and "WHOLE-MIX INSTRUMENT TAGS" not in (stem_midi_report or "")
                    ):
                        status("Tagging instruments on the full (un-separated) mix...")
                        try:
                            whole_mix_report = build_whole_mix_instrument_report(dsp_path)
                            if whole_mix_report:
                                revised += "\n\n" + whole_mix_report
                                _gan = ""
                                try:
                                    _gan = guitar_absence_note_from_tags(locals().get("whole_mix_tags"), whole_mix_report)
                                except Exception:
                                    _gan = ""
                                if _gan:
                                    revised += "\n\n" + _gan
                        except Exception as e:
                            print(f"  (whole-mix instrument tagging skipped: {e})")

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

                        _release_heavy_analysis_memory_before_identity()

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

                    revised = _collapse_runaway_repetition_fields(revised)
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
                depth_requested = bool(
                    re.search(
                        r"\b(in[\s-]?depth|in\s+detail|elaborate|tell me more|go deeper|"
                        r"more detail|full breakdown|deep dive|thorough|comprehensive|"
                        r"everything (you'?ve got|you know)|expand on|say more|break (it|that|this) down|"
                        r"walk me through|long answer|really get into)\b",
                        question,
                        re.IGNORECASE,
                    )
                )
                reply_style_note = (
                    "Reply as their music buddy in the conversation. They asked for depth here — "
                    "give a genuinely long, detailed answer that actually goes into it, not a short "
                    "summary with a token gesture toward length."
                    if depth_requested else
                    "Reply as their music buddy in the conversation — answer what they asked, "
                    "not a full analytical write-up unless they asked for one."
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
                        year_bit = f", {meta.get('year')}" if meta.get("year") else ""
                        year_lock = ""
                        if meta.get("year"):
                            year_lock = (
                                f" The reliable release year for this track is {meta.get('year')}; "
                                "use that year when stating when the track or album was released. "
                                "Do not replace it with a different year from knowledge of other releases by the same artist. "
                                "State it plainly (\"came out in {y}\") — don't cite where the year came from."
                            ).format(y=meta.get("year"))
                        context_prior_note = (
                            "\n\nTRACK ID (private reference only — reliable, but never cite this framing "
                            "out loud; just state the facts the way you'd naturally know them): this recording "
                            "is "
                            f"{meta.get('title') or 'unknown title'} by {meta.get('artist') or 'unknown artist'}"
                            + (f" from {meta.get('album')}" if meta.get("album") else "")
                            + year_bit
                            + ". "
                            "Talk about the artist, vocalist, and producer freely when it's useful or asked about — "
                            "credits, discography, career/album context, reception, influences, comparisons, general "
                            "reputation and history are all fair game from your normal knowledge, same as you'd "
                            "discuss any artist. Don't hedge on well-known background facts you're actually confident about.\n"
                            "The one place to be careful: when you're specifically characterising what THIS "
                            "RECORDING sounds like (its genre, instrumentation, arrangement, production, vocal "
                            "delivery), ground that in the audio evidence above rather than in what this artist "
                            "'usually' sounds like or how their other releases are remembered — an artist can make "
                            "a record that doesn't match their reputation. If your background knowledge and the "
                            "audio evidence would point to different sounds, trust the audio evidence and it's fine "
                            "to note the two agree or don't.\n"
                            "Don't state uncertain trivia as fact, and don't guess at nationality/origin unless "
                            "confident — outside of that, engage normally."
                            + year_lock
                        )

                # Cached once per track by default, or re-looked-up fresh using this
                # question's own keywords when WIKI_CONTEXT_REFRESH_EVERY_QUESTION is
                # True — see build_wiki_context_block() for the strict background-only
                # / no-audio-technical-facts scope.
                wiki_context_block = _get_or_build_wiki_context(
                    current_track, track_metadata, track_wiki_context, question=question
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
                        "TRACK SCOPE: The notes below describe ONLY this track "
                        f"({label}). Do not attribute instruments, vocals, tempo, key, "
                        "or production details from any earlier track in this conversation "
                        "to this one unless the user is explicitly comparing and you name "
                        "each track. Previous tracks may be used only as comparison points, "
                        "never as evidence for this track. "
                        "Never transfer instruments, vocal qualities, genre/style traits, "
                        "arrangement details, mix characteristics, or production descriptions "
                        "from another analysed track into the current track. "
                        "Phantom carry-over of guitar/piano/strings is a known "
                        "failure mode — if these notes omit an instrument or include a "
                        "GUITAR ABSENCE NOTE, that instrument is not present here. "
                        "If several RECOMMENDED KEY blocks appear, use the last one "
                        "(after any KEY PROFILE REFINEMENT). "
                        "For tempo, use RECOMMENDED TEMPO's integer only — do not voice "
                        "genre-prior or detector jargon.\n\n"
                        + comprehensive_analyses[current_track]
                    )

                # metadata_block / context_prior_note / wiki_context_block / cover_note are
                # all static per-track content that was already bundled into the same
                # writer message as evidence_section the first time this track was
                # discussed (see track_evidence_message below). If that earlier message
                # is still safely present in writer_history — same check used for the
                # evidence block above — these don't need to go out again either; a
                # short pointer is enough. If it's not safe (evicted by compaction, or at
                # risk of being trimmed from this request), fall back to sending them in
                # full, same as evidence_section does.
                if evidence_still_safe:
                    static_context_note = (
                        "(Wiki background and file metadata notes for this track are "
                        "also already above — reuse them unless something in "
                        "CONFIRMED CORRECTIONS contradicts them.)"
                    )
                    metadata_block = ""
                    context_prior_note = ""
                    wiki_context_block = ""
                    # cover_note is deliberately NOT suppressed here: has_cover_context
                    # (above) is recomputed fresh every turn and can turn True for
                    # reasons unrelated to old track evidence — e.g. the user attaches
                    # a brand-new image this turn via extra_image_refs. Gating it on
                    # evidence_still_safe risked dropping cover-art guidance on exactly
                    # the turn it's newly relevant. It's a small fixed block, so the
                    # token cost of always sending it when has_cover_context is True is
                    # low relative to that risk.
                else:
                    static_context_note = ""

                # The actual question ends up FAR more likely to get a direct,
                # context-aware answer (rather than a generic "here's an overview
                # of this track" reply) when it isn't buried at the bottom of a
                # single giant message stacked underneath the full evidence dump.
                # Small local models in particular tend to latch onto whatever
                # dominates the prompt, and the private-notes block is often the
                # single largest thing in the request the first time a track is
                # discussed. So: when the full evidence needs to go out (brand
                # new track, or evicted from history), send it as its OWN turn
                # with a short synthesized ack — mirroring how /load already
                # separates "here's the loaded track" from the follow-up
                # question — then send the question as a short, separate, FINAL
                # turn so it's the most recent (and most attended-to) thing the
                # model sees. When the evidence is already safely in history,
                # keep the previous lean single-message behaviour.
                context_linking_note = (
                    "\n\nIf what they're asking relates to something already discussed earlier in this "
                    "conversation — comparing this to another track, continuing a topic, reacting to an "
                    "earlier point — use those earlier turns to answer that directly. Don't just give a "
                    "standalone description of this track in isolation when they asked about a connection "
                    "to something already talked about. "
                    "When comparing, name each track explicitly. Never import instruments from an earlier "
                    "track onto the current one (e.g. do not say this track has guitar only because the "
                    "previous song did)."
                )

                if not OLLAMA_SUPPORTS_IMAGES:
                    writer_images = []
                else:
                    writer_images = [img for img in writer_images if _is_sendable_base64_image(img)]

                if not evidence_still_safe:
                    context_turn_msg = (
                        f"[We're listening to: {label}]\n\n"
                        f"{evidence_section}"
                        f"{correction_block}"
                        f"{metadata_block}"
                        f"{context_prior_note}"
                        f"{wiki_context_block}"
                        f"{cover_note}"
                    )
                    context_message = {"role": "user", "content": context_turn_msg}
                    if writer_images:
                        context_message["images"] = writer_images[:MAX_WRITER_IMAGES_PER_TURN]

                    writer_history.append(context_message)
                    # This message now holds the full evidence block — remember it (by
                    # object identity) as the one to check against on future turns.
                    track_evidence_message[current_track] = context_message
                    writer_history.append({
                        "role": "assistant",
                        "content": f"Got it — I have the details on '{label}'. What would you like to know?",
                    })

                    writer_user_msg = (
                        f"{vocal_gate_note}\n\n"
                        f"User said: {question}"
                        f"{context_linking_note}\n\n"
                        f"{reply_style_note}"
                    )
                    writer_message = {"role": "user", "content": writer_user_msg}

                else:
                    writer_user_msg = (
                        f"[We're listening to: {label}]\n\n"
                        f"{evidence_section}"
                        f"{static_context_note}"
                        f"{correction_block}"
                        f"{cover_note}"
                        f"{vocal_gate_note}\n\n"
                        f"User said: {question}"
                        f"{context_linking_note}\n\n"
                        f"{reply_style_note}"
                    )
                    writer_message = {"role": "user", "content": writer_user_msg}
                    if writer_images:
                        writer_message["images"] = writer_images[:MAX_WRITER_IMAGES_PER_TURN]

                last_writer_message = writer_message
                writer_history.append(writer_message)

                status("Writing...")
                final_reply, _usage = ollama_chat(writer_history)
                writer_history.append({"role": "assistant", "content": final_reply})

            else:
                # General question / cross-track comparison — straight to the output LLM.
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
                    wiki_general_block = build_wiki_context_for_general_question(
                        user_text, track_metadata, current_track,
                        writer_history=writer_history,
                    )
                    if WIKI_DEBUG_LOG:
                        if wiki_general_block:
                            _matched_titles = re.findall(r'--- "(.+?)" ---', wiki_general_block)
                            print(f"  (wiki: matched {len(_matched_titles)} article(s): {', '.join(_matched_titles)})")
                        else:
                            print("  (wiki: no local DB match for this question — answering from the model's own knowledge)")
                    writer_message = {"role": "user", "content": user_text + wiki_general_block}
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

    main()
