#!/usr/bin/env python3
"""Musiclyse SongFormer helper (0.7) -- runs SongFormer for Musiclyse.

pre62: when Musiclyse's /songformer runs it in a colour terminal (MUSICLYSE_COLOR=1),
its own messages print in Musiclyse's default yellow; otherwise exactly as before.

pre59 (Stage 2): Musiclyse's analysis runs the file command below while it settles a
song's section map, and uses SongFormer's sections to check and correct its own (its
SONGFORMER_CORRECT_STRUCTURE settings). The commands here only run SongFormer and
compare; they never change a save.

SongFormer (ASLP-lab, 2025) is an open music-structure model: it splits a song into
sections and names them (intro, verse, pre-chorus, chorus, bridge, inst, outro,
silence). This helper keeps it entirely separate from Musiclyse:

  * it runs in SongFormer's own Python environment (Python 3.11, its own packages);
  * its results go in saved-songs/songformer/<save name>.json, beside -- never inside --
    the saves; this helper never changes a save, an export or the chat;
  * each section after the first also carries "boundary_strength" (0..1, how sure
    SongFormer is of its start); Musiclyse lets a very sure boundary move its own
    a little further (SONGFORMER_STRONG_MOVES).

Commands (Musiclyse's /songformer runs the same ones):

  setup                     print the one-time setup steps
  selftest [audio file]     check the setup and time one song (a test tone if no file)
  scan [saves or folder]    run SongFormer on saved songs' audio (default: every save);
                            songs already done are skipped unless their audio changed
       --rescan             run again even where a result exists
       --device auto|mps|cpu|cuda   (auto: the Mac GPU when it works, else the CPU)
  file AUDIO --out RESULT    run SongFormer on one audio (or video) file and write its
                            sections to RESULT (Musiclyse's analysis uses this)
  compare [saves or folder] compare SongFormer's sections with Musiclyse's and write a
                            report and a check sheet (needs no SongFormer setup)
       --sheet-songs N      songs on the check sheet (default 20)
       --include NAME       put a song on the check sheet (part of its name; repeatable)

Where things are: saves default to the saved-songs folder beside this script (or the
folder Musiclyse names). SongFormer's code is expected in a folder called SongFormer
beside this script, else in ~/SongFormer/code where install_songformer.sh puts it
(--repo PATH to use another).
"""

from __future__ import annotations

import argparse
import datetime
import glob
import json
import math
import os
import re
import subprocess
import sys
import time

__version__ = "0.7"

# 0.7-pre62: when Musiclyse's /songformer runs this helper in a colour terminal it sets
# MUSICLYSE_COLOR=1, and the helper's own messages then print in Musiclyse's default
# yellow, as Musiclyse's do. Without it (run on its own, or by an analysis, whose output
# is captured) every message prints exactly as before. Results never change.
import builtins as _builtins

_MUSICLYSE_COLOR = os.environ.get("MUSICLYSE_COLOR") == "1"


def _musiclyse_color(text, stream):
    """text for stream, each plain line in Musiclyse's default yellow when MUSICLYSE_COLOR
    asks for it and the stream is a terminal; otherwise text unchanged."""
    if not _MUSICLYSE_COLOR or not (stream is sys.stdout or stream is sys.stderr):
        return text
    try:
        if not stream.isatty():
            return text
    except Exception:
        return text
    return "\n".join(("\033[33m" + part + "\033[0m") if part.strip() and "\x1b" not in part else part
                     for part in str(text).split("\n"))


def print(*values, sep=" ", end="\n", file=None, flush=False):  # noqa: A001
    """The built-in print, with Musiclyse's default colour when MUSICLYSE_COLOR asks for it."""
    if _MUSICLYSE_COLOR:
        stream = sys.stdout if file is None else file
        values = tuple(_musiclyse_color(str(value), stream) for value in values)
    _builtins.print(*values, sep=sep, end=end, file=file, flush=flush)


class _MusiclyseArgumentParser(argparse.ArgumentParser):
    """argparse's usage, help and error messages in Musiclyse's colour (MUSICLYSE_COLOR only)."""

    def _print_message(self, message, file=None):
        if message:
            message = _musiclyse_color(message, sys.stderr if file is None else file)
        super()._print_message(message, file)

HERE = os.path.dirname(os.path.abspath(__file__))
RESULT_FOLDER = "songformer"                  # inside the saves folder
RESULT_FORMAT = "musiclyse-songformer"
SAMPLE_RATE = 24000                           # what SongFormer expects
MODEL_NAME = "SongFormer (ASLP-lab), SongForm-HX-8Class output"

SETUP_TEXT = r"""
SongFormer setup (macOS; once)
------------------------------
SongFormer is installed in its own folder in your home folder, ~/SongFormer, never in
the Musiclyse folder, with its own Python packages, so Musiclyse's own stay untouched.

The easy way: in Terminal, from the Musiclyse folder, run

    bash install_songformer.sh

It downloads SongFormer's code to ~/SongFormer/code, makes its Python environment in
~/SongFormer/env (Python 3.11), installs its packages,
downloads the model files (a few GB) and runs the self-test. Running it again is safe:
finished steps are skipped. Musiclyse finds ~/SongFormer by itself -- nothing to set.

By hand instead (the same steps):

    mkdir -p ~/SongFormer && cd ~/SongFormer
    git clone https://github.com/ASLP-lab/SongFormer.git code
    cd code && git submodule update --init --recursive
    python3.11 -m venv ~/SongFormer/env
    grep -v -E '^(triton|pesq|gpustat)' requirements.txt > ~/SongFormer/requirements-mac.txt
    ~/SongFormer/env/bin/pip install -r ~/SongFormer/requirements-mac.txt
    cd src/SongFormer && ~/SongFormer/env/bin/python utils/fetch_pretrained.py
    ~/SongFormer/env/bin/python -c "from huggingface_hub import snapshot_download, hf_hub_download; snapshot_download('OpenMuQ/MuQ-large-msd-iter'); hf_hub_download('facebook/wav2vec2-conformer-rope-large-960h-ft', 'config.json')"

(The last line matters: Musiclyse keeps Hugging Face offline, so the MuQ model and the
small config file MusicFM reads must be on disk before the first scan.)

Installed somewhere else? Set SONGFORMER_PYTHON and SONGFORMER_REPO near the top of
the Musiclyse script. Then, in Musiclyse:  /songformer selftest
"""


# ---------------------------------------------------------------------------
# Saves and results (standard library only)
# ---------------------------------------------------------------------------

def default_saves_folder():
    return os.environ.get("MUSICLYSE_SONGFORMER_SAVES") or os.path.join(HERE, "saved-songs")


def collect_saves(paths):
    """Save files named by paths, folders (their *.json, not subfolders) or patterns."""
    if not paths:
        paths = [default_saves_folder()]
    found = []
    for p in paths:
        p = os.path.expanduser(p)
        if os.path.isdir(p):
            found.extend(sorted(glob.glob(os.path.join(glob.escape(p), "*.json"))))
        elif any(ch in p for ch in "*?["):
            found.extend(sorted(glob.glob(p)))
        else:
            found.append(p)
    seen, out = set(), []
    for f in found:
        a = os.path.abspath(f)
        if a not in seen:
            seen.add(a)
            out.append(f)
    return out


def load_save(path):
    """A Musiclyse save as a dict, or None when it isn't one."""
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict) or "analysis" not in data or "track_path" not in data:
        return None
    return data


def result_path(save_path):
    folder = os.path.dirname(os.path.abspath(save_path))
    stem = os.path.splitext(os.path.basename(save_path))[0]
    return os.path.join(folder, RESULT_FOLDER, stem + ".json")


def source_signature(path):
    try:
        st = os.stat(path)
        return [int(st.st_size), int(st.st_mtime_ns)]
    except OSError:
        return None


def read_result(path):
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict) or data.get("format") != RESULT_FORMAT or not isinstance(data.get("sections"), list):
        return None
    return data


def write_json_atomic(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temp = path + ".tmp"
    with open(temp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)
    os.replace(temp, path)


def mmss(seconds):
    seconds = max(0.0, float(seconds))
    m = int(seconds // 60)
    return f"{m}:{seconds - 60 * m:04.1f}"


# ---------------------------------------------------------------------------
# Audio
# ---------------------------------------------------------------------------

def decode_audio(path):
    """The file's audio as mono float32 at 24 kHz (any format ffmpeg reads, video too)."""
    import numpy as np
    result = subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-i", path, "-vn", "-ac", "1",
                             "-ar", str(SAMPLE_RATE), "-f", "f32le", "-"],
                            capture_output=True, timeout=600)
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", "replace").strip().splitlines()
        raise RuntimeError("ffmpeg couldn't read the audio" + (f": {detail[-1]}" if detail else ""))
    wav = np.frombuffer(result.stdout, dtype=np.float32).copy()
    if wav.size < SAMPLE_RATE:
        raise RuntimeError("less than a second of audio")
    return wav


# ---------------------------------------------------------------------------
# SongFormer (its own environment)
# ---------------------------------------------------------------------------

def find_repo(explicit=None):
    """SongFormer's code folder: the one named (--repo, or Musiclyse's SONGFORMER_REPO),
    else the SongFormer folder beside this script, else ~/SongFormer/code."""
    named = explicit or os.environ.get("MUSICLYSE_SONGFORMER_REPO")
    if named:
        candidates = [named]
    else:     # beside this script, else where install_songformer.sh puts it
        candidates = [os.path.join(HERE, "SongFormer"), os.path.join(os.path.expanduser("~"), "SongFormer", "code")]
    for candidate in candidates:
        candidate = os.path.abspath(os.path.expanduser(candidate))
        if os.path.isfile(os.path.join(candidate, "src", "SongFormer", "models", "SongFormer.py")):
            return candidate
    where = os.path.abspath(os.path.expanduser(named)) if named else "~/SongFormer/code"
    raise RuntimeError(f"SongFormer's code wasn't found in {where}"
                       + (" (the folder named for it)" if named else "")
                       + ". Run install_songformer.sh (or: setup)")


def add_boundary_strengths(sections, boundary_prob, frame_rate, reach=2):
    """pre59: give each section after the first a "boundary_strength" (0..1): the
    highest boundary probability SongFormer gave within `reach` frames (about a
    quarter of a second) of its start. Only adds that field; never changes the
    sections. Musiclyse uses it for SONGFORMER_STRONG_MOVES."""
    import numpy as np
    prob = np.asarray(boundary_prob, dtype=float).ravel()
    if prob.size == 0:
        return sections
    for idx, sec in enumerate(sections):
        if idx == 0:
            continue
        frame = int(round(float(sec["start"]) * float(frame_rate)))
        lo, hi = max(0, frame - reach), min(prob.size, frame + reach + 1)
        if hi > lo:
            sec["boundary_strength"] = round(float(prob[lo:hi].max()), 3)
    return sections


class SongFormerRunner:
    """SongFormer's own inference (as in its app.py), loaded once for many songs.

    Differences from app.py, none of which change its results: the device can be the
    Mac GPU (MPS) with a CPU fallback; audio arrives already decoded at 24 kHz; paths
    are absolute instead of changing the working folder; a song whose last window is
    under 1,024 samples no longer loops forever (that window is skipped, as app.py
    intends); CUDA cache calls only run on CUDA.
    """

    WIN = 420                     # seconds per long window (app.py: TIME_DUR / win_size)
    HOP = 420
    NUM_CLASSES = 128
    FRAME_RATE = 8.333            # AFTER_DOWNSAMPLING_FRAME_RATES
    DATASET_LABEL = "SongForm-HX-8Class"
    DATASET_IDS = [5]

    def __init__(self, repo, device="auto", log=print):
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")   # unsupported MPS ops run on the CPU
        self.repo = repo
        self.log = log
        src = os.path.join(repo, "src", "SongFormer")
        third = os.path.join(repo, "src", "third_party")
        self.src = src
        for path in (third, src):
            if path not in sys.path:
                sys.path.insert(0, path)
        self.musicfm_dir = os.path.join(src, "ckpts", "MusicFM")
        self.checkpoint = next((p for p in (os.path.join(src, "ckpts", "SongFormer.safetensors"),
                                            os.path.join(src, "ckpts", "SongFormer.pt")) if os.path.isfile(p)), None)
        missing = [p for p in (os.path.join(self.musicfm_dir, "msd_stats.json"),
                               os.path.join(self.musicfm_dir, "pretrained_msd.pt")) if not os.path.isfile(p)]
        if self.checkpoint is None:
            missing.append(os.path.join(src, "ckpts", "SongFormer.safetensors"))
        if missing:
            raise RuntimeError("model files missing (run install_songformer.sh again, or see: setup): "
                               + ", ".join(missing))
        import numpy as np
        import scipy
        scipy.inf = np.inf          # as app.py: msaf still uses scipy.inf
        import torch
        self.torch = torch
        self.np = np
        self.device = self._pick_device(device)
        self._load(self.device)

    def _pick_device(self, wanted):
        torch = self.torch
        wanted = (wanted or "auto").lower()
        if wanted not in ("auto", "mps", "cpu", "cuda"):
            raise RuntimeError(f"unknown device {wanted!r} (use auto, mps, cpu or cuda)")
        if wanted == "auto":
            if torch.cuda.is_available():
                return "cuda"
            if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
                return "mps"
            return "cpu"
        if wanted == "mps" and not (getattr(torch.backends, "mps", None) is not None
                                    and torch.backends.mps.is_available()):
            raise RuntimeError("the Mac GPU (MPS) isn't available to this PyTorch")
        if wanted == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA isn't available")
        return wanted

    def _load(self, device):
        torch = self.torch
        import importlib
        from omegaconf import OmegaConf
        from ema_pytorch import EMA
        from muq import MuQ
        from musicfm.model.musicfm_25hz import MusicFM25Hz
        from postprocessing.functional import postprocess_functional_structure
        from dataset.label2id import DATASET_ID_ALLOWED_LABEL_IDS, DATASET_LABEL_TO_DATASET_ID
        self._postprocess = postprocess_functional_structure
        self.hp = OmegaConf.load(os.path.join(self.src, "configs", "SongFormer.yaml"))
        try:
            self.muq = MuQ.from_pretrained("OpenMuQ/MuQ-large-msd-iter").to(device).eval()
        except Exception as exc:
            if os.environ.get("HF_HUB_OFFLINE") == "1":
                raise RuntimeError("the MuQ model isn't downloaded yet, and Musiclyse keeps Hugging Face "
                                   "offline -- run install_songformer.sh again to download it") from exc
            raise
        # MusicFM's own loader calls torch.load without weights_only; PyTorch 2.6+ changed
        # that default, so give it the old one while MusicFM loads (SongFormer pins 2.4).
        original_load = torch.load

        def compatible_load(*args, **kwargs):
            kwargs.setdefault("weights_only", False)
            return original_load(*args, **kwargs)

        torch.load = compatible_load
        try:
            self.musicfm = MusicFM25Hz(is_flash=False,
                                       stat_path=os.path.join(self.musicfm_dir, "msd_stats.json"),
                                       model_path=os.path.join(self.musicfm_dir, "pretrained_msd.pt")).to(device).eval()
        finally:
            torch.load = original_load
        Model = getattr(importlib.import_module("models.SongFormer"), "Model")
        self.msa = Model(self.hp)
        if self.checkpoint.endswith(".safetensors"):
            from safetensors.torch import load_file
            ckpt = {"model_ema": load_file(self.checkpoint, device="cpu")}
        else:
            ckpt = torch.load(self.checkpoint, map_location="cpu")
        if ckpt.get("model_ema", None) is not None:
            ema = EMA(self.msa, include_online_model=False)
            ema.load_state_dict(ckpt["model_ema"])
            self.msa.load_state_dict(ema.ema_model.state_dict())
        else:
            self.msa.load_state_dict(ckpt["model"])
        self.msa.to(device).eval()
        self.label_mask = {}
        for key, allowed in DATASET_ID_ALLOWED_LABEL_IDS.items():
            mask = self.np.ones(self.NUM_CLASSES, dtype=bool)
            mask[allowed] = False
            self.label_mask[key] = mask
        self.dataset_index = DATASET_LABEL_TO_DATASET_ID[self.DATASET_LABEL]
        self.device = device

    def to_cpu(self):
        for model in (self.muq, self.musicfm, self.msa):
            model.to("cpu")
        self.device = "cpu"

    def _free(self):
        torch = self.torch
        if self.device == "cuda":
            torch.cuda.empty_cache()
        elif self.device == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()

    def sections(self, wav):
        """[{start, end, label}] for one song's 24 kHz mono audio."""
        try:
            return self._sections(wav)
        except Exception as exc:
            if self.device != "mps":
                raise
            self.log(f"    (the Mac GPU couldn't run this ({exc.__class__.__name__}: {str(exc)[:120]}); "
                     f"switching to the CPU for the rest of this run)")
            self.to_cpu()
            return self._sections(wav)

    def _sections(self, wav):
        torch, np = self.torch, self.np
        sr, win, hop, fr = SAMPLE_RATE, self.WIN, self.HOP, self.FRAME_RATE
        audio = torch.tensor(wav).to(self.device)
        total_len = ((audio.shape[0] // sr) // win * win) + win
        total_frames = math.ceil(total_len * fr)
        logits = {"function_logits": np.zeros([total_frames, self.NUM_CLASSES]),
                  "boundary_logits": np.zeros([total_frames])}
        counts = {"function_logits": np.zeros([total_frames, self.NUM_CLASSES]),
                  "boundary_logits": np.zeros([total_frames])}
        lens = 0
        i = 0
        with torch.no_grad():
            while True:
                start_idx = i * sr
                end_idx = min((i + win) * sr, audio.shape[-1])
                if start_idx >= audio.shape[-1]:
                    break
                if end_idx - start_idx <= 1024:
                    i += hop          # app.py `continue`s here without moving on
                    continue
                seg = audio[start_idx:end_idx]
                muq_420 = self.muq(seg.unsqueeze(0), output_hidden_states=True)["hidden_states"][10]
                self._free()
                musicfm_420 = self.musicfm.get_predictions(seg.unsqueeze(0))[1][10]
                self._free()
                muq_30, musicfm_30 = [], []
                for i30 in range(i, i + hop, 30):
                    s30 = i30 * sr
                    e30 = min((i30 + 30) * sr, audio.shape[-1], (i + hop) * sr)
                    if s30 >= audio.shape[-1]:
                        break
                    if e30 - s30 <= 1024:
                        continue
                    muq_30.append(self.muq(audio[s30:e30].unsqueeze(0), output_hidden_states=True)["hidden_states"][10])
                    self._free()
                    musicfm_30.append(self.musicfm.get_predictions(audio[s30:e30].unsqueeze(0))[1][10])
                    self._free()
                if muq_30:
                    embeds = [torch.concatenate(musicfm_30, dim=1), torch.concatenate(muq_30, dim=1),
                              musicfm_420, muq_420]
                    shortest = min(x.shape[1] for x in embeds)
                    embeds = [x[:, :shortest, :] for x in embeds]
                    embd = torch.concatenate(embeds, axis=-1)
                    dataset_ids = torch.Tensor(self.DATASET_IDS).to(self.device, dtype=torch.long)
                    mask = (torch.Tensor(self.label_mask[self.dataset_index]).to(self.device, dtype=bool)
                            .unsqueeze(0).unsqueeze(0))
                    _info, chunk = self.msa.infer(input_embeddings=embd, dataset_ids=dataset_ids,
                                                  label_id_masks=mask, with_logits=True)
                    start_frame = int(i * fr)
                    end_frame = start_frame + min(math.ceil(hop * fr), chunk["boundary_logits"][0].shape[0])
                    logits["function_logits"][start_frame:end_frame, :] += \
                        chunk["function_logits"][0].detach().cpu().numpy()
                    logits["boundary_logits"][start_frame:end_frame] = chunk["boundary_logits"][0].detach().cpu().numpy()
                    counts["function_logits"][start_frame:end_frame, :] += 1
                    counts["boundary_logits"][start_frame:end_frame] += 1
                    lens += end_frame - start_frame
                i += hop
        logits["function_logits"] /= np.maximum(counts["function_logits"], 1)
        logits["boundary_logits"] /= np.maximum(counts["boundary_logits"], 1)
        logits["function_logits"] = torch.from_numpy(logits["function_logits"][:lens]).unsqueeze(0)
        logits["boundary_logits"] = torch.from_numpy(logits["boundary_logits"][:lens]).unsqueeze(0)
        output = self._postprocess(logits, self.hp)
        sections = []
        for idx in range(len(output) - 1):
            start, label = output[idx]
            end = output[idx + 1][0]
            if end > start:
                sections.append({"start": round(float(start), 2), "end": round(float(end), 2), "label": str(label)})
        try:
            # pre59: how sure SongFormer is of each section's start (it doesn't change the sections).
            add_boundary_strengths(sections, torch.sigmoid(logits["boundary_logits"][0]).numpy(), fr)
        except Exception:
            pass
        return sections


# ---------------------------------------------------------------------------
# Commands: setup, selftest, scan
# ---------------------------------------------------------------------------

def explain(exc):
    """The error, plus what to do when it's a model file missing while offline."""
    text = f"{exc.__class__.__name__}: {exc}"
    if "couldn't connect to 'https://huggingface.co'" in text or "outgoing traffic has been disabled" in text:
        text += ("\n    (A model file SongFormer needs isn't downloaded yet, and Musiclyse keeps Hugging Face "
                 "offline -- run install_songformer.sh again to fetch it.)")
    return text


def cmd_setup(_args):
    print(SETUP_TEXT)
    return 0


def cmd_selftest(args):
    print(f"  SongFormer helper {__version__}; Python {sys.version.split()[0]} ({sys.executable})")
    if sys.version_info[:2] not in ((3, 10), (3, 11)):
        print("  (SongFormer is meant for Python 3.11 (or 3.10); this is a different version -- it may still work.)")
    try:
        repo = find_repo(args.repo)
        print(f"  SongFormer code: {repo}")
        started = time.time()
        runner = SongFormerRunner(repo, args.device)
        print(f"  Models loaded on {runner.device} in {time.time() - started:.1f} s")
        if args.audio:
            wav = decode_audio(os.path.abspath(os.path.expanduser(args.audio)))
            what = os.path.basename(args.audio)
        else:
            import numpy as np
            t = np.arange(SAMPLE_RATE * 40, dtype=np.float32) / SAMPLE_RATE
            wav = (0.2 * np.sin(2 * np.pi * 220 * t) * (1 + (t > 20))).astype(np.float32)
            what = "a 40-second test tone"
        started = time.time()
        sections = runner.sections(wav)
        took = time.time() - started
    except KeyboardInterrupt:
        print("\n  Self-test stopped.")
        return 130
    except Exception as exc:
        print(f"  ✗ Self-test failed: {explain(exc)}")
        return 1
    print(f"  ✓ {what}: {len(sections)} section(s) in {took:.1f} s on {runner.device} "
          f"({len(wav) / SAMPLE_RATE / max(took, 1e-6):.1f}x real time)")
    for sec in sections[:12]:
        print(f"      {mmss(sec['start'])}–{mmss(sec['end'])}  {sec['label']}")
    return 0


def cmd_scan(args):
    saves = collect_saves(args.inputs)
    if not saves:
        print("  No saves found.")
        return 1
    try:
        repo = find_repo(args.repo)
    except RuntimeError as exc:
        print(f"  ✗ {exc}")
        return 1
    runner = None
    done = skipped = failed = 0
    missing_audio, not_saves = [], []
    print(f"  SongFormer scan: {len(saves)} save(s); results go in "
          f"{os.path.join(os.path.dirname(os.path.abspath(saves[0])), RESULT_FOLDER)}")
    try:
        for n, save_path in enumerate(saves, 1):
            name = os.path.splitext(os.path.basename(save_path))[0]
            data = load_save(save_path)
            if data is None:
                not_saves.append(name)
                continue
            source = str(data.get("track_path") or "")
            if not source or source.startswith(("http://", "https://")):
                missing_audio.append(f"{name} (no local audio file recorded)")
                continue
            source = os.path.abspath(os.path.expanduser(source))
            if not os.path.isfile(source):
                missing_audio.append(f"{name} ({source})")
                continue
            target = result_path(save_path)
            old = read_result(target)
            signature = source_signature(source)
            if old is not None and not args.rescan and old.get("source_signature") == signature:
                skipped += 1
                continue
            if runner is None:
                started = time.time()
                print("  Loading SongFormer...")
                runner = SongFormerRunner(repo, args.device)
                print(f"  Loaded on {runner.device} in {time.time() - started:.1f} s")
            try:
                started = time.time()
                wav = decode_audio(source)
                sections = runner.sections(wav)
                took = time.time() - started
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                failed += 1
                print(f"  [{n}/{len(saves)}] ✗ {name}: {exc.__class__.__name__}: {str(exc)[:200]}")
                continue
            write_json_atomic(target, {
                "format": RESULT_FORMAT, "version": 1, "helper": __version__, "model": MODEL_NAME,
                "save": os.path.basename(save_path), "source": source, "source_signature": signature,
                "duration": round(len(wav) / SAMPLE_RATE, 2), "device": runner.device,
                "seconds": round(took, 2), "created": datetime.datetime.now().isoformat(timespec="seconds"),
                "sections": sections})
            done += 1
            print(f"  [{n}/{len(saves)}] {name}: {len(sections)} sections in {took:.1f} s")
    except KeyboardInterrupt:
        print(f"\n  Stopped. {done} song(s) finished in this run are kept; run the same command to carry on.")
        return 130
    except Exception as exc:
        print(f"  ✗ SongFormer couldn't start: {explain(exc)}")
        return 1
    print(f"\n  SongFormer scan done -- scanned: {done}, already done: {skipped}, failed: {failed}"
          + (f", audio not found: {len(missing_audio)}" if missing_audio else ""))
    for line in missing_audio[:30]:
        print(f"    audio not found: {line}")
    if len(missing_audio) > 30:
        print(f"    ... and {len(missing_audio) - 30} more")
    if not_saves:
        print(f"  ({len(not_saves)} file(s) skipped: not Musiclyse saves)")
    print("  Nothing in Musiclyse's saves, exports or chat changed. The comparison: compare (in Musiclyse, /songformer compare).")
    return 1 if failed else 0


def cmd_file(args):
    """pre59: SongFormer's sections for one audio file, written to --out (the same
    result format as scan, without a save name). Musiclyse's analysis runs this while
    it settles the section map; exit code 0 only when the result was written."""
    source = os.path.abspath(os.path.expanduser(args.audio))
    if not os.path.isfile(source):
        print(f"  ✗ audio not found: {source}")
        return 1
    try:
        repo = find_repo(args.repo)
    except RuntimeError as exc:
        print(f"  ✗ {exc}")
        return 1
    try:
        started = time.time()
        runner = SongFormerRunner(repo, args.device, log=lambda *a, **k: None)
        loaded = time.time() - started
        started = time.time()
        wav = decode_audio(source)
        sections = runner.sections(wav)
        took = time.time() - started
    except KeyboardInterrupt:
        raise
    except Exception as exc:
        print(f"  ✗ SongFormer couldn't run: {explain(exc)}")
        return 1
    write_json_atomic(os.path.abspath(os.path.expanduser(args.out)), {
        "format": RESULT_FORMAT, "version": 1, "helper": __version__, "model": MODEL_NAME,
        "save": "", "source": source, "source_signature": source_signature(source),
        "duration": round(len(wav) / SAMPLE_RATE, 2), "device": runner.device,
        "seconds": round(took, 2), "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "sections": sections})
    print(f"  SongFormer: {len(sections)} sections on {runner.device} "
          f"(loaded in {loaded:.1f} s, ran in {took:.1f} s)")
    return 0


# ---------------------------------------------------------------------------
# compare (standard library only)
# ---------------------------------------------------------------------------

VOCAL_NAMES = {"verse", "chorus", "pre-chorus", "post-chorus"}
DANCE_STYLE_WORDS = ("house", "techno", "trance", "drum n bass", "drum and bass", "dubstep", "electro",
                     "breakbeat", "breaks", "hardstyle", "hardcore", "jungle", "garage", "big beat",
                     "edm", "eurodance", "electro house", "progressive", "gabber", "happy hardcore")


def base_name(name):
    """'Chorus 2' -> 'chorus'; 'Section 3' -> 'section'."""
    text = re.sub(r"\s*\d+\s*$", "", str(name or "")).strip().lower()
    return re.sub(r"\s+", " ", text)


def musiclyse_category(name):
    b = base_name(name)
    table = {"intro": "intro", "verse": "verse", "pre-chorus": "pre-chorus", "prechorus": "pre-chorus",
             "chorus": "chorus", "post-chorus": "chorus", "refrain": "chorus", "hook": "chorus",
             "bridge": "bridge", "middle eight": "bridge", "instrumental": "inst", "solo": "inst",
             "break": "inst", "interlude": "inst", "outro": "outro", "coda": "outro", "silence": "silence"}
    for key, value in table.items():
        if b == key or b.startswith(key + " "):
            return value
    return None          # "Section N" and anything unusual: no category


def songformer_category(label):
    b = str(label or "").strip().lower()
    return {"prechorus": "pre-chorus", "pre-chorus": "pre-chorus", "intro": "intro", "verse": "verse",
            "chorus": "chorus", "bridge": "bridge", "inst": "inst", "outro": "outro",
            "silence": "silence"}.get(b, b or None)


def boundaries(sections, duration):
    """Section starts, leaving out the very start and anything in the last half-second."""
    out = []
    for sec in sections:
        try:
            t = float(sec["start"])
        except (KeyError, TypeError, ValueError):
            continue
        if t > 0.5 and (duration is None or t < duration - 0.5):
            out.append(t)
    return sorted(set(round(t, 2) for t in out))


def match(a, b, tolerance):
    """One-to-one matches between two boundary lists within `tolerance`: [(a_t, b_t)]."""
    pairs = sorted((abs(x - y), x, y) for x in a for y in b if abs(x - y) <= tolerance)
    used_a, used_b, out = set(), set(), []
    for _d, x, y in pairs:
        if x in used_a or y in used_b:
            continue
        used_a.add(x)
        used_b.add(y)
        out.append((x, y))
    return out


def section_at(sections, t, key):
    for sec in sections:
        try:
            if float(sec["start"]) <= t < float(sec["end"]):
                return sec.get(key)
        except (KeyError, TypeError, ValueError):
            continue
    return None


def track_groups(data, m_sections):
    groups = []
    # By category, so "Verse Part 3" or "Chorus 2" count as vocal names too.
    categories = {musiclyse_category(s.get("name")) for s in m_sections}
    groups.append("vocal" if categories & {"verse", "chorus", "pre-chorus"} else "instrumental")
    styles = data.get("discogs_styles") or []
    try:
        top = str(styles[0][0]).lower() if styles else ""
    except (TypeError, IndexError):
        top = ""
    style = top.split("---")[-1]
    if style and any(word in style for word in DANCE_STYLE_WORDS):
        groups.append("dance/electronic")
    return groups


def compare_song(data, result):
    m = [s for s in (data.get("structure_sections") or []) if isinstance(s, dict)]
    s = [x for x in result.get("sections") or [] if isinstance(x, dict)]
    if not m or not s:
        return None
    try:
        duration = max(float(x["end"]) for x in m + s)
    except (KeyError, TypeError, ValueError):
        duration = None
    try:
        bpm = float(data.get("bpm") or 0)
    except (TypeError, ValueError):
        bpm = 0
    bar = 240.0 / bpm if 40 <= bpm <= 260 else 2.0
    mb, sb = boundaries(m, duration), boundaries(s, duration)
    stats = {"m_count": len(mb), "s_count": len(sb), "bar": round(bar, 2)}
    for key, tol in (("half", 0.5), ("bar", bar), ("three", 3.0)):
        hits = match(mb, sb, tol)
        stats[key] = len(hits)
    near = match(mb, sb, 3.0)
    offsets = sorted(y - x for x, y in near)
    stats["median_offset"] = offsets[len(offsets) // 2] if offsets else None
    # Names: compare every quarter-second where Musiclyse's name has a category.
    agree = total = 0
    confusion = {}
    if duration:
        steps = int(duration / 0.25)
        for k in range(steps):
            t = k * 0.25 + 0.125
            mc = musiclyse_category(section_at(m, t, "name"))
            sc = songformer_category(section_at(s, t, "label"))
            if mc is None or sc is None:
                continue
            total += 1
            if mc == sc or {mc, sc} == {"verse", "pre-chorus"}:
                agree += 1
            else:
                confusion[(mc, sc)] = confusion.get((mc, sc), 0) + 1
    stats["label_total"] = total
    stats["label_agree"] = agree
    stats["confusion"] = sorted(((v * 0.25, k) for k, v in confusion.items()), reverse=True)
    # Disagreements for the check sheet.
    rows = []
    matched_m = {x for x, _y in near}
    matched_s = {y for _x, y in near}
    for x, y in near:
        if abs(y - x) > 0.5:
            rows.append((min(x, y), "boundaries differ in time",
                         f"boundary at {mmss(x)} ({section_at(m, x - 0.05, 'name')} → {section_at(m, x + 0.05, 'name')})",
                         f"boundary at {mmss(y)} ({section_at(s, y - 0.05, 'label')} → {section_at(s, y + 0.05, 'label')})"))
    for x in mb:
        if x not in matched_m:
            rows.append((x, "only Musiclyse has a boundary here",
                         f"{section_at(m, x - 0.05, 'name')} → {section_at(m, x + 0.05, 'name')} at {mmss(x)}",
                         f"no boundary within 3 s (inside {section_at(s, x, 'label')})"))
    for y in sb:
        if y not in matched_s:
            rows.append((y, "only SongFormer has a boundary here",
                         f"no boundary within 3 s (inside {section_at(m, y, 'name')})",
                         f"{section_at(s, y - 0.05, 'label')} → {section_at(s, y + 0.05, 'label')} at {mmss(y)}"))
    # Names that disagree for 4 s or more in a row.
    if duration:
        run_start, run_pair = None, None
        steps = int(duration / 0.25)
        for k in range(steps + 1):
            t = k * 0.25 + 0.125
            if k < steps:
                mname = section_at(m, t, "name")
                mc = musiclyse_category(mname)
                sl = section_at(s, t, "label")
                sc = songformer_category(sl)
                differ = (mc is not None and sc is not None and mc != sc and {mc, sc} != {"verse", "pre-chorus"})
                pair = (mname, sl) if differ else None
            else:
                pair = None
            if pair != run_pair:
                if run_pair is not None and t - run_start >= 4.0:
                    rows.append((run_start, "names differ",
                                 f"{run_pair[0]} ({mmss(run_start)}–{mmss(t - 0.125)})",
                                 f"{run_pair[1]}"))
                run_start, run_pair = (t - 0.125, pair) if pair is not None else (None, None)
    stats["rows"] = sorted(rows)
    return stats


def pct(a, b):
    return f"{100.0 * a / b:5.1f}%" if b else "   – "


def cmd_compare(args):
    saves = collect_saves(args.inputs)
    songs = []
    no_result = 0
    for save_path in saves:
        data = load_save(save_path)
        if data is None:
            continue
        result = read_result(result_path(save_path))
        if result is None:
            no_result += 1
            continue
        stats = compare_song(data, result)
        if stats is None:
            continue
        name = os.path.splitext(os.path.basename(save_path))[0]
        m_sections = [s for s in (data.get("structure_sections") or []) if isinstance(s, dict)]
        stats.update(name=name, groups=track_groups(data, m_sections))
        songs.append(stats)
    if not songs:
        print("  Nothing to compare yet: scan some songs first (scan)." if not no_result
              else f"  None of the {no_result} save(s) has a SongFormer result yet: run scan first.")
        return 1
    folder = os.path.join(os.path.dirname(os.path.abspath(saves[0])), RESULT_FOLDER)
    stamp = datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S")
    report_path = os.path.join(folder, f"comparison {stamp}.txt")
    sheet_path = os.path.join(folder, f"check sheet {stamp}.csv")

    def summary(group_songs):
        mb = sum(x["m_count"] for x in group_songs)
        sb = sum(x["s_count"] for x in group_songs)
        lines = []
        for key, label in (("half", "within 0.5 s"), ("bar", "within a bar"), ("three", "within 3 s")):
            hit = sum(x[key] for x in group_songs)
            lines.append(f"    {label:13}  {pct(hit, mb)} of Musiclyse's boundaries   {pct(hit, sb)} of SongFormer's")
        offs = sorted(x["median_offset"] for x in group_songs if x["median_offset"] is not None)
        if offs:
            lines.append(f"    typical offset (SongFormer minus Musiclyse, matched within 3 s): "
                         f"{offs[len(offs) // 2]:+.2f} s")
        lt = sum(x["label_total"] for x in group_songs)
        la = sum(x["label_agree"] for x in group_songs)
        lines.append(f"    section names agree for {pct(la, lt).strip()} of the time where Musiclyse uses a "
                     f"pop name (Section N parts aren't counted)")
        return lines

    out = [f"Musiclyse x SongFormer comparison ({stamp}; helper {__version__})",
           "=" * 72,
           "Agreement, not accuracy: neither is assumed right. Where they disagree, the check sheet",
           "lists the spots to listen to.",
           "",
           f"Songs compared: {len(songs)}" + (f" ({no_result} save(s) have no SongFormer result)" if no_result else ""),
           f"Boundaries: Musiclyse {sum(x['m_count'] for x in songs)}, SongFormer {sum(x['s_count'] for x in songs)}",
           ""]
    for title, pick in (("All songs", lambda x: True),
                        ("Songs with vocal section names", lambda x: "vocal" in x["groups"]),
                        ("Instrumental (no Verse/Chorus names)", lambda x: "instrumental" in x["groups"]),
                        ("Dance/electronic (by top style)", lambda x: "dance/electronic" in x["groups"])):
        group = [x for x in songs if pick(x)]
        if group:
            out.append(f"{title} -- {len(group)} song(s)")
            out.extend(summary(group))
            out.append("")
    confusion = {}
    for x in songs:
        for seconds, pair in x["confusion"]:
            confusion[pair] = confusion.get(pair, 0) + seconds
    if confusion:
        out.append("Where the names differ most (Musiclyse name -> SongFormer name, total time):")
        for pair, seconds in sorted(confusion.items(), key=lambda kv: -kv[1])[:10]:
            out.append(f"    {pair[0]:>10} -> {pair[1]:<10} {mmss(seconds)}")
        out.append("")
    out.append("Songs, least boundary agreement first (within 3 s; Musiclyse / SongFormer counts):")
    for x in sorted(songs, key=lambda x: (x["three"] / max(1, max(x["m_count"], x["s_count"])), x["name"])):
        lab = pct(x["label_agree"], x["label_total"]).strip()
        out.append(f"    {pct(x['three'], max(1, max(x['m_count'], x['s_count'])))}  "
                   f"{x['m_count']:>2}/{x['s_count']:<2}  names {lab:>6}  {x['name']}  "
                   f"[{', '.join(x['groups'])}]")
    os.makedirs(folder, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(out) + "\n")

    # Check sheet: songs picked by criteria (plus any named with --include).
    wanted = max(1, int(args.sheet_songs))
    chosen = []

    def add(candidates, limit):
        for x in candidates:
            if len([c for c in chosen if c in candidates]) >= limit or len(chosen) >= wanted:
                break
            if x not in chosen and x["rows"]:
                chosen.append(x)

    for text in args.include or []:
        folded = text.casefold()
        for x in songs:
            if folded in x["name"].casefold() and x not in chosen:
                chosen.append(x)
    by_disagreement = sorted(songs, key=lambda x: -len(x["rows"]))
    add([x for x in by_disagreement if "instrumental" in x["groups"]], max(1, wanted * 3 // 10))
    add([x for x in by_disagreement if "dance/electronic" in x["groups"]], max(1, wanted * 2 // 10))
    add([x for x in by_disagreement if "vocal" in x["groups"]], wanted)
    add(by_disagreement, wanted)
    import csv
    with open(sheet_path, "w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["Song", "Time", "What differs", "Musiclyse", "SongFormer",
                         "Which is right? (M / S / both / neither)", "Notes"])
        for x in chosen:
            for t, kind, mtext, stext in x["rows"]:
                writer.writerow([x["name"], mmss(t), kind, mtext, stext, "", ""])
    print(f"  Compared {len(songs)} song(s).")
    print(f"  Report:      {report_path}")
    print(f"  Check sheet: {sheet_path} ({len(chosen)} song(s); open it in Numbers or Excel, listen at each "
          f"time and fill in the 'Which is right?' column)")
    return 0


# ---------------------------------------------------------------------------

def main(argv=None):
    ap = (_MusiclyseArgumentParser if _MUSICLYSE_COLOR else argparse.ArgumentParser)(   # pre62 colour
        prog="musiclyse-songformer",
        description="Run SongFormer for Musiclyse (this helper never changes a save).")
    ap.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = ap.add_subparsers(dest="command")
    sub.add_parser("setup", help="print the one-time setup steps")
    p = sub.add_parser("selftest", help="check the setup and time one song")
    p.add_argument("audio", nargs="?", help="an audio file (default: a test tone)")
    p.add_argument("--device", default=os.environ.get("MUSICLYSE_SONGFORMER_DEVICE", "auto"),
                   choices=("auto", "mps", "cpu", "cuda"))
    p.add_argument("--repo", help="SongFormer's code folder (default: SongFormer beside this script)")
    p = sub.add_parser("scan", help="run SongFormer on saved songs' audio")
    p.add_argument("inputs", nargs="*", help="saves, folders or patterns (default: every save)")
    p.add_argument("--rescan", action="store_true", help="run again even where a result exists")
    p.add_argument("--device", default=os.environ.get("MUSICLYSE_SONGFORMER_DEVICE", "auto"),
                   choices=("auto", "mps", "cpu", "cuda"))
    p.add_argument("--repo", help="SongFormer's code folder (default: SongFormer beside this script)")
    p = sub.add_parser("file", help="run SongFormer on one audio file (Musiclyse's analysis uses this)")
    p.add_argument("audio", help="an audio or video file")
    p.add_argument("--out", required=True, help="where to write the result (JSON)")
    p.add_argument("--device", default=os.environ.get("MUSICLYSE_SONGFORMER_DEVICE", "auto"),
                   choices=("auto", "mps", "cpu", "cuda"))
    p.add_argument("--repo", help="SongFormer's code folder (default: SongFormer beside this script)")
    p = sub.add_parser("compare", help="compare SongFormer's sections with Musiclyse's")
    p.add_argument("inputs", nargs="*", help="saves, folders or patterns (default: every save)")
    p.add_argument("--sheet-songs", type=int, default=20, help="songs on the check sheet (default 20)")
    p.add_argument("--include", action="append", help="put a song on the check sheet (part of its name)")
    args = ap.parse_args(argv)
    if not args.command:
        ap.print_help()
        return 0
    return {"setup": cmd_setup, "selftest": cmd_selftest, "scan": cmd_scan, "file": cmd_file,
            "compare": cmd_compare}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
