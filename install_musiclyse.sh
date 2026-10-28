#!/usr/bin/env bash
# =============================================================================
# Musiclyse 0.5-moss (dev46) installer -- macOS / Apple Silicon, Python 3.11
# =============================================================================
# Run it from inside an activated Python 3.11 virtual environment, in the
# folder that holds the Musiclyse script and requirements.txt:
#
#     python3.11 -m venv venv
#     source venv/bin/activate
#     bash install_musiclyse.sh            # asks before each large download
#     bash install_musiclyse.sh --yes      # no questions
#
# Every step checks first and skips what is already there: Python packages,
# ffmpeg, Ollama and its model, the MOSS-Music code, and every model download.
# Running it again is safe. Nothing outside the venv, the Hugging Face / torch
# caches, ~/panns_data and ~/MOSS-Music (or MOSS_MUSIC_CODE_DIR) is changed,
# apart from Homebrew installs of ffmpeg / Ollama, which it asks about first.
#
# Options:
#   --yes            answer yes to every question
#   --skip-models    install the software only; no model downloads
#   --no-essentia    don't try the optional Essentia package
# =============================================================================

# (no "set -u": macOS ships bash 3.2, where an empty array trips it)

OLLAMA_MODEL="muse-glimmer:30b-mlx"          # must match OLLAMA_MODEL in the script
MOSS_REPO="https://github.com/OpenMOSS/MOSS-Music.git"
MOSS_COMMIT="ad107c7ddaa06de168a0dfbc18d3e1e6a40c0e5e"

ASSUME_YES=0; SKIP_MODELS=0; TRY_ESSENTIA=1
for a in "$@"; do
  case "$a" in
    --yes|-y) ASSUME_YES=1 ;;
    --skip-models) SKIP_MODELS=1 ;;
    --no-essentia) TRY_ESSENTIA=0 ;;
    -h|--help) sed -n '2,24p' "$0"; exit 0 ;;
    *) echo "Unknown option: $a (try --help)"; exit 2 ;;
  esac
done

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
WARNINGS=()

say()  { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
ok()   { printf '    \033[32mok\033[0m  %s\n' "$*"; }
skip() { printf '    \033[36mskip\033[0m %s\n' "$*"; }
warn() { printf '    \033[33m!!\033[0m  %s\n' "$*"; WARNINGS+=("$*"); }
die()  { printf '\n\033[31mStopped: %s\033[0m\n' "$*"; exit 1; }
ask() {   # ask "question" -> 0 for yes
  [ "$ASSUME_YES" = 1 ] && return 0
  local r; read -r -p "    $1 [y/N] " r; [[ "$r" =~ ^[Yy] ]]
}

# --- 1. The environment ------------------------------------------------------
say "Checking the Python environment"
[ -n "${VIRTUAL_ENV:-}" ] || die "no virtual environment is active. Run:  python3.11 -m venv venv && source venv/bin/activate"
PY="$(command -v python)"
PYV="$("$PY" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
[ "$PYV" = "3.11" ] || die "this venv runs Python $PYV; Musiclyse's pinned packages need Python 3.11."
ok "Python $PYV in $VIRTUAL_ENV"
[ -f requirements.txt ] || die "requirements.txt is not next to this script ($HERE)."
SCRIPT="$(ls -t musiclyse*.py 2>/dev/null | grep -v -e style_setup -e export | head -1)"
[ -n "$SCRIPT" ] && ok "Musiclyse script: $SCRIPT" || warn "no musiclyse*.py found here -- the final check and the MAEST self-test are skipped"
[ "$(uname -s)" = "Darwin" ] || warn "this installer is written for macOS; tensorflow-macos / tensorflow-metal will not install elsewhere"

have_pkg() { "$PY" -m pip show "$1" >/dev/null 2>&1; }
# The same compatibility patches Musiclyse applies before importing madmom and
# omnizart (old collections / numpy names), so an import check here tests what
# Musiclyse will actually see.
COMPAT_PY='import collections, collections.abc, warnings
warnings.simplefilter("ignore")
for _n in ("MutableSequence", "MutableMapping", "Mapping"):
    if not hasattr(collections, _n): setattr(collections, _n, getattr(collections.abc, _n))
import numpy as np
for _n, _t in (("float", float), ("int", int), ("complex", complex), ("bool", bool)):
    if not hasattr(np, _n): setattr(np, _n, _t)
'
py_imports() { "$PY" -W ignore -c "$COMPAT_PY
import $1" >/dev/null 2>&1; }
pkg_ver()  { "$PY" -m pip show "$1" 2>/dev/null | awk '/^Version:/{print $2}'; }

# --- 2. Python packages, in the order madmom needs on Python 3.11 ------------
say "Installing Python packages"
"$PY" -m pip install --upgrade pip wheel "setuptools<81" -q || die "could not upgrade pip/setuptools/wheel"
ok "pip, wheel, setuptools (<81: madmom still needs pkg_resources)"

if [ "$(pkg_ver madmom)" = "0.16.1" ] && py_imports madmom; then
  skip "madmom 0.16.1 already built and importable"
else
  "$PY" -m pip install "Cython==3.0.1" "numpy==1.26.4" "scipy==1.15.3" "mido==1.3.3" -q \
    || die "could not install the madmom build prerequisites"
  ok "Cython 3.0.1, numpy 1.26.4, scipy, mido (madmom build prerequisites)"
  "$PY" -m pip install "madmom==0.16.1" --no-build-isolation -q \
    || die "madmom 0.16.1 failed to build (it needs the Cython / numpy pinned just above)"
  ok "madmom 0.16.1 (built without build isolation)"
fi

"$PY" -m pip install -r requirements.txt -q || die "pip install -r requirements.txt failed (see the pip output above)"
ok "requirements.txt"
py_imports madmom || warn "madmom does not import after the requirements step -- rerun this script"

if [ "$TRY_ESSENTIA" = 1 ]; then
  if py_imports essentia; then
    skip "Essentia already installed"
  elif "$PY" -m pip install essentia -q >/dev/null 2>&1 && py_imports essentia; then
    ok "Essentia (optional)"
  else
    warn "Essentia (optional) did not install with pip; Musiclyse works without it. Do NOT install essentia-tensorflow."
  fi
fi

# --- 3. System tools: ffmpeg and Ollama ---------------------------------------
say "Checking ffmpeg and Ollama"
if command -v ffmpeg >/dev/null 2>&1 && command -v ffprobe >/dev/null 2>&1; then
  skip "ffmpeg $(ffmpeg -version 2>/dev/null | head -1 | awk '{print $3}')"
elif command -v brew >/dev/null 2>&1 && ask "ffmpeg is missing. Install it with Homebrew (brew install ffmpeg)?"; then
  brew install ffmpeg && ok "ffmpeg" || warn "brew install ffmpeg failed"
else
  warn "ffmpeg is missing -- install it (brew install ffmpeg); Musiclyse needs it to read audio files"
fi

if command -v ollama >/dev/null 2>&1; then
  skip "Ollama $(ollama --version 2>/dev/null | awk '{print $NF}')"
elif command -v brew >/dev/null 2>&1 && ask "Ollama is missing. Install it with Homebrew (brew install ollama)?"; then
  brew install ollama && ok "Ollama" || warn "brew install ollama failed -- get it from https://ollama.com"
else
  warn "Ollama is missing -- install it from https://ollama.com (Musiclyse's chat model runs in it)"
fi

# --- 4. The MOSS-Music code (imported directly by the script) ----------------
say "Checking the MOSS-Music code"
MOSS_DIR=""
for c in "${MOSS_MUSIC_CODE_DIR:-}" "$HERE/MOSS-Music" "$HOME/MOSS-Music"; do
  if [ -n "$c" ] && [ -f "$c/src/modeling_moss_music.py" ]; then MOSS_DIR="$c"; break; fi
done
if [ -n "$MOSS_DIR" ]; then
  have="$(git -C "$MOSS_DIR" rev-parse HEAD 2>/dev/null || echo unknown)"
  if [ "$have" = "$MOSS_COMMIT" ]; then
    skip "MOSS-Music code in $MOSS_DIR (tested commit)"
  else
    warn "MOSS-Music code in $MOSS_DIR is at ${have:0:8}, not the tested ${MOSS_COMMIT:0:8} -- left as it is (git -C \"$MOSS_DIR\" checkout $MOSS_COMMIT switches it)"
  fi
elif command -v git >/dev/null 2>&1; then
  git clone -q "$MOSS_REPO" "$HOME/MOSS-Music" && git -C "$HOME/MOSS-Music" checkout -q "$MOSS_COMMIT" \
    && ok "MOSS-Music code cloned to ~/MOSS-Music at the tested commit" \
    || warn "could not clone $MOSS_REPO -- clone it to ~/MOSS-Music and check out $MOSS_COMMIT"
else
  warn "git is missing -- install the Xcode command line tools (xcode-select --install), then rerun"
fi

# --- 5. Models ---------------------------------------------------------------
if [ "$SKIP_MODELS" = 1 ]; then
  say "Skipping model downloads (--skip-models)"
else
  say "Checking the models"

  # Ollama chat model
  if command -v ollama >/dev/null 2>&1; then
    STARTED_OLLAMA=0
    if ! curl -s -m 3 http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
      (ollama serve >/dev/null 2>&1 &) ; STARTED_OLLAMA=1
      for _ in $(seq 1 20); do curl -s -m 2 http://127.0.0.1:11434/api/tags >/dev/null 2>&1 && break; sleep 1; done
    fi
    if ollama list 2>/dev/null | awk 'NR>1{print $1}' | grep -qx "$OLLAMA_MODEL"; then
      skip "Ollama model $OLLAMA_MODEL"
    elif ask "Download the chat model $OLLAMA_MODEL with Ollama (roughly 17 GB)?"; then
      ollama pull "$OLLAMA_MODEL" && ok "Ollama model $OLLAMA_MODEL" || warn "ollama pull $OLLAMA_MODEL failed"
    else
      warn "chat model not downloaded -- later: ollama pull $OLLAMA_MODEL"
    fi
    [ "$STARTED_OLLAMA" = 1 ] && warn "Ollama was started in the background for this install; open the Ollama app (or run 'ollama serve') before using Musiclyse"
  fi

  rm -f "${TMPDIR:-/tmp}/musiclyse_need_omnizart"
  # Hugging Face, Demucs, PANNs and Omnizart models (Musiclyse runs with
  # Hugging Face offline, so these must be on disk before the first scan).
  ASSUME_YES="$ASSUME_YES" MUSICLYSE_SCRIPT="${SCRIPT:+$HERE/$SCRIPT}" PYTHONWARNINGS=ignore "$PY" - <<'PYEOF'
import os, sys, subprocess, urllib.request, warnings
warnings.simplefilter("ignore")
os.environ.pop("HF_HUB_OFFLINE", None); os.environ.pop("TRANSFORMERS_OFFLINE", None)
yes = os.environ.get("ASSUME_YES") == "1"
problems = []

def ask(q):
    """Asks on the terminal itself: this program's standard input is the
    script text, so input() would read the end of it and answer "no"."""
    if yes:
        return True
    try:
        # separate handles: a terminal can't be opened read+write in text mode
        with open("/dev/tty", "w") as out, open("/dev/tty") as inp:
            out.write(f"    {q} [y/N] ")
            out.flush()
            return inp.readline().strip().lower().startswith("y")
    except Exception:
        print(f"    (could not ask: {q} -- answering no; rerun with --yes to accept)")
        return False

def ok(m):   print(f"    \033[32mok\033[0m  {m}")
def skip(m): print(f"    \033[36mskip\033[0m {m}")
def warn(m): print(f"    \033[33m!!\033[0m  {m}"); problems.append(m)

# -- Hugging Face models
try:
    from huggingface_hub import snapshot_download, try_to_load_from_cache
except Exception as e:
    snapshot_download = None
    warn(f"huggingface_hub not importable ({e}); Hugging Face models not checked")


def hf_cached(repo):
    """True when the cache holds what from_pretrained loads offline: the config
    and the weights (one file or a sharded index). A full snapshot is not needed."""
    def has(name):
        try:
            p = try_to_load_from_cache(repo, name)
            return isinstance(p, str) and os.path.isfile(p)
        except Exception:
            return False
    weights = ("model.safetensors", "model.safetensors.index.json", "pytorch_model.bin",
               "pytorch_model.bin.index.json")
    return has("config.json") and any(has(w) for w in weights)

HF = [("OpenMOSS-Team/MOSS-Music-8B-Instruct", "the listening model, MOSS-Music", "about 18 GB"),
      ("openai/whisper-large-v3", "Whisper (lyrics and word timing)", "about 3 GB"),
      ("laion/larger_clap_music_and_speech", "CLAP (sonic similarity)", "about 0.8 GB")]
for repo, what, size in HF if snapshot_download else []:
    if hf_cached(repo):
        skip(f"{what} ({repo})")
        continue
    if ask(f"Download {what}, {repo} ({size})?"):
        try:
            snapshot_download(repo)
            ok(f"{what} ({repo})")
        except Exception as e:
            warn(f"{repo} download failed: {str(e)[:160]}")
    else:
        warn(f"{repo} not downloaded -- Musiclyse keeps Hugging Face offline, so rerun this script before scanning")

# -- MAEST (Discogs styles): its own setup script downloads and self-tests it
here = os.getcwd()
setup = os.path.join(here, "musiclyse_style_setup.py")
maest = "mtg-upf/discogs-maest-30s-pw-73e-ts"
have_maest = False
if snapshot_download:
    have_maest = hf_cached(maest)
if have_maest:
    skip(f"MAEST style model ({maest})")
elif os.path.isfile(setup):
    if ask("Download the MAEST style model (about 350 MB) with musiclyse_style_setup.py?"):
        r = subprocess.run([sys.executable, setup])
        ok("MAEST style model") if r.returncode == 0 else warn("musiclyse_style_setup.py reported a problem (see above)")
    else:
        warn("MAEST not downloaded -- later: python musiclyse_style_setup.py")
else:
    warn("musiclyse_style_setup.py is not in this folder -- the finer style tags (MAEST) stay off until it is run")

# -- Demucs htdemucs_6s (stem separation). Its weights are one checkpoint,
#    5c90dfd2, in the Hugging Face cache (demucs 4.1+) or torch's hub folder.
try:
    import torch
    hub = os.path.join(torch.hub.get_dir(), "checkpoints")
    name = "htdemucs_6s"
    try:   # the model the Musiclyse script actually uses (DEMUCS_MODEL)
        scr = os.environ.get("MUSICLYSE_SCRIPT") or ""
        import re as _re
        m_ = _re.search(r'(?m)^DEMUCS_MODEL\s*=\s*["\']([\w\-]+)["\']', open(scr, encoding="utf-8").read()) if scr else None
        name = m_.group(1) if m_ else name
    except Exception:
        pass
    # demucs 4.1+ keeps the model in the Hugging Face cache (adefossez/HTDemucs-6s:
    # htdemucs_6s.yaml + <sig>.safetensors); older versions use torch's hub folder.
    sigs = ["5c90dfd2"] if name == "htdemucs_6s" else []
    try:
        import yaml, demucs
        with open(os.path.join(os.path.dirname(demucs.__file__), "remote", f"{name}.yaml")) as fh:
            sigs = list(yaml.safe_load(fh).get("models") or sigs)
    except Exception:
        pass

    def in_hf_cache():
        try:
            from huggingface_hub import try_to_load_from_cache
            if name == "htdemucs":
                rn = "HTDemucs"
            elif name.startswith("htdemucs_"):
                rn = "HTDemucs-" + name[len("htdemucs_"):]
            else:
                rn = "Demucs-" + name
            repo = f"adefossez/{rn}"
            files = [f"{name}.yaml"] + [f"{sg}.safetensors" for sg in sigs]
            return bool(sigs) and all(isinstance(try_to_load_from_cache(repo, f), str) for f in files)
        except Exception:
            return False

    in_torch_hub = bool(sigs) and os.path.isdir(hub) and all(
        any(f.startswith(sg) and f.endswith(".th") for f in os.listdir(hub)) for sg in sigs)
    have = in_hf_cache() or in_torch_hub
    if have:
        skip(f"Demucs {name}")
    elif ask(f"Get the Demucs {name} stem model (about 0.2 GB; quick if it is already on disk)?"):
        from demucs.pretrained import get_model
        get_model(name)
        ok(f"Demucs {name}")
    else:
        warn("Demucs model not downloaded -- it downloads on the first scan instead")
except Exception as e:
    warn(f"Demucs check failed: {str(e)[:160]}")

# -- PANNs (instrument tags, genre/mood cross-check). panns_inference fetches
#    these with wget, which macOS doesn't have, so they are fetched here.
panns_dir = os.path.expanduser("~/panns_data")
PANNS = [("Cnn14_mAP=0.431.pth", "https://zenodo.org/record/3987831/files/Cnn14_mAP%3D0.431.pth?download=1", 300e6),
         ("class_labels_indices.csv",
          "http://storage.googleapis.com/us_audio_set/youtube_corpus/v1/csv/class_labels_indices.csv", 1e4)]
for name, url, min_size in PANNS:
    p = os.path.join(panns_dir, name)
    if os.path.isfile(p) and os.path.getsize(p) >= min_size:
        skip(f"PANNs {name}")
        continue
    if name.endswith(".pth") and not ask("Download the PANNs tagger checkpoint (about 330 MB)?"):
        warn("PANNs checkpoint not downloaded -- instrument tags stay off until it is")
        continue
    try:
        os.makedirs(panns_dir, exist_ok=True)
        tmp = p + ".part"
        urllib.request.urlretrieve(url, tmp)
        if os.path.getsize(tmp) < min_size:
            raise RuntimeError("file came back too small")
        os.replace(tmp, p)
        ok(f"PANNs {name}")
    except Exception as e:
        warn(f"PANNs {name} download failed ({str(e)[:120]}) -- get it from {url} into {panns_dir}")

# -- Omnizart checkpoints (note transcription of the stems). The pip package
#    ships each model's graph (saved_model.pb) but not its weights
#    (variables/variables.data-*); those come from `omnizart download-checkpoints`.
#    Without them every stem transcribes to nothing, silently.
try:
    import importlib.util
    spec = importlib.util.find_spec("omnizart")
    root = os.path.dirname(spec.origin) if spec and spec.origin else None
    if not root:
        raise RuntimeError("omnizart is not installed")
    need = [os.path.join(root, "checkpoints", "drum", "drum_keras"),
            os.path.join(root, "checkpoints", "music", "music_piano"),
            os.path.join(root, "checkpoints", "vocal", "vocal_semi")]

    # Exact sizes of the release files, so a cut-off download is not taken as complete.
    full_size = {"drum_keras": 31090686, "music_piano": 50407445, "vocal_semi": 114617623}

    def weights_ok(d):
        f = os.path.join(d, "variables", "variables.data-00000-of-00001")
        return os.path.isfile(f) and os.path.getsize(f) >= full_size.get(os.path.basename(d), 1_000_000)
    missing = [os.path.relpath(d, root) for d in need if not weights_ok(d)]
    if not missing:
        skip("Omnizart checkpoints")
    else:
        # Not downloaded here: the user runs `omnizart download-checkpoints`
        # once, at the end (it fetches all models from Omnizart's GitHub release).
        open(os.path.join(os.environ.get("TMPDIR", "/tmp"), "musiclyse_need_omnizart"), "w").close()
        print("    \033[33m!!\033[0m  Omnizart checkpoints are missing (" + ", ".join(missing) + ") -- see the last step below")
except Exception as e:
    warn(f"Omnizart check failed: {str(e)[:160]}")

# lv-chordia ships its own model inside the package: nothing to download.
sys.exit(1 if problems else 0)
PYEOF
  [ $? -eq 0 ] || WARNINGS+=("some model steps need attention (see the !! lines above)")
fi

# --- 6. Final check -----------------------------------------------------------
say "Checking that everything imports"
PYTHONWARNINGS=ignore "$PY" - <<'PYEOF' || WARNINGS+=("some packages do not import (see above)")
import importlib, sys, warnings
warnings.simplefilter("ignore")
# Musiclyse's own compatibility patches for madmom / omnizart
import collections, collections.abc
for _n in ("MutableSequence", "MutableMapping", "Mapping"):
    if not hasattr(collections, _n):
        setattr(collections, _n, getattr(collections.abc, _n))
import numpy as np
for _n, _t in (("float", float), ("int", int), ("complex", complex), ("bool", bool)):
    if not hasattr(np, _n):
        setattr(np, _n, _t)
bad = []
for mod in ("requests", "numpy", "scipy", "librosa", "mutagen", "torch", "torchaudio", "transformers",
            "accelerate", "huggingface_hub", "demucs", "madmom", "omnizart", "pretty_midi", "tensorflow",
            "panns_inference", "pyloudnorm", "soundfile", "psutil", "lv_chordia"):
    try:
        importlib.import_module(mod)
    except Exception as e:
        bad.append(f"{mod}: {str(e).splitlines()[0][:120] if str(e) else type(e).__name__}")
for b in bad:
    print(f"    \033[33m!!\033[0m  {b}")
print("    \033[32mok\033[0m  all core packages import" if not bad else "")
sys.exit(1 if bad else 0)
PYEOF

OMZ_FLAG="${TMPDIR:-/tmp}/musiclyse_need_omnizart"
NEED_OMZ=0
if [ -f "$OMZ_FLAG" ]; then
  rm -f "$OMZ_FLAG"; NEED_OMZ=1
  say "One more step: the Omnizart checkpoints"
  echo "    Omnizart's model weights are not installed yet. Without them Musiclyse transcribes no notes"
  echo "    or drum hits. With this venv still active, run:"
  echo ""
  echo "        omnizart download-checkpoints"
  echo ""
  echo "    It downloads about 850 MB and takes a few minutes. Afterwards you can rerun this installer"
  echo "    to confirm (optional; it should say: skip Omnizart checkpoints)."
  echo "    If it stops with CERTIFICATE_VERIFY_FAILED, your Python is the python.org build: double-click"
  echo "    'Install Certificates.command' in /Applications/Python 3.11, then run the command again."
  WARNINGS+=("Omnizart checkpoints: run 'omnizart download-checkpoints'")
fi

say "Done"
if [ ${#WARNINGS[@]} -eq 0 ]; then
  echo "    Everything is in place. Start Musiclyse with:  python ${SCRIPT:-musiclyse.py}"
else
  echo "    Finished with ${#WARNINGS[@]} thing(s) to look at:"
  for w in "${WARNINGS[@]}"; do echo "     - $w"; done
  if [ "$NEED_OMZ" = 1 ] && [ ${#WARNINGS[@]} -eq 1 ]; then
    echo "    After running it, you can rerun this script to confirm (optional)."
  else
    echo "    Rerun this script after fixing them; finished steps are skipped."
  fi
fi
