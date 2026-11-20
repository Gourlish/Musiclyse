# MUSICLYSE FILE pipeline-stems 0.8
# ==============================================================================
#  MUSICLYSE 0.8: STEMS AND NOTES
# ==============================================================================
# This file holds stem separation (Demucs), the notes transcribed from the stems (Omnizart) and
# the stem summaries. It is part of musiclyse-0.8.py, which loads it whole at start-up, after
# its own code, into its own namespace (see "THE MUSICLYSE FILES" there); it is not run on its
# own.
#
#  * New code for these areas goes in this file. Helpers that several files use go in
#    musiclyse-0.8.py.
#  * Settings go only in musiclyse-0.8-settings.py.
#  * Never import another Musiclyse file: all of them share one namespace, so a name defined
#    in any of them can be used directly. Define each function and class in one file only.
#  * At the top level (outside functions), use only names from the main script or from a file
#    loaded before this one; inside functions, any name can be used.
#  * CLAUDE.md, beside these files, lists what each file holds.
# ==============================================================================
# Stem pairs Demucs routinely bleeds between; agreement across only these,
# with no native anchor, is reported as possible double-counting.
BLEED_PRONE_STEM_PAIRS = frozenset({
    frozenset(("bass", "guitar")),
    frozenset(("guitar", "other")),
    frozenset(("bass", "other")),
    frozenset(("piano", "other")),
    frozenset(("guitar", "piano")),
})

# 2) EXTENT CONFLICT. Presence and extent are different claims. When the
#    whole-mix pass DID detect a label but placed it in only a couple of
#    windows, while the stems claim it runs through the track, the whole-mix
#    read is the one measured without separation artifacts -- so the stems are
#    probably catching a similar timbre (a pad read as organ) rather than the
#    instrument playing throughout. Report the conflict instead of asserting
#    an ongoing part. Requires whole-mix persistence stats, which is why these
#    must survive the probability-matrix eviction (see PANNS_TAG_STATS_CACHE_SIZE).
# (CONVERGENCE_EXTENT_CONFLICT_ENABLED moved to the settings file beside this script.)

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
# (ENABLE_GENRE_MOOD_TAGGING moved to the settings file beside this script.)
# (GENRE_TAG_TOP_K moved to the settings file beside this script.)
# (GENRE_TAG_MIN_PROB moved to the settings file beside this script.)
# (MOOD_TAG_TOP_K moved to the settings file beside this script.)
# (MOOD_TAG_MIN_PROB moved to the settings file beside this script.)

# --- Demucs separation quality (test-time shift ensembling) -----------------
# `--shifts N` runs Demucs N times on randomly time-shifted copies of the
# input and averages the results, which measurably improves separation
# quality (particularly for the weaker htdemucs_6s guitar/piano stems) at
# a roughly (N+1)x runtime cost. Off by default in fast mode; enabled in
# deep mode, where the user has already signalled they want more accuracy
# over speed (see MOSS_DEEP_MODE_ADDENDUM).
# (DEMUCS_SHIFTS_FAST moved to the settings file beside this script.)
# (DEMUCS_SHIFTS_DEEP moved to the settings file beside this script.)

# (MAX_IMAGES_PER_REQUEST moved to the settings file beside this script.)
# (ENABLE_COVER_ART_DESCRIPTION moved to the settings file beside this script.)
# (ENABLE_SINGER_IDENTITY_RESOLUTION moved to the settings file beside this script.)
# (ENABLE_IMAGE_OBSERVATIONS_FOR_GENERAL moved to the settings file beside this script.)

# Rewrite internal vocal-category vocabulary ("post-puberty male", raw
# snake_case tags) into ordinary language in the writer's user-facing replies.
# The writer prompt already forbids this vocabulary, but that is only an
# instruction; smaller local models leak the tags anyway because the tags are
# sitting right there in the private notes. Set False to see raw model output.
# (SANITIZE_VOCAL_CATEGORY_LANGUAGE moved to the settings file beside this script.)

# (COVER_ART_DESCRIPTION_NUM_CTX moved to the settings file beside this script.)
# (SINGER_IDENTITY_NUM_CTX moved to the settings file beside this script.)
# (MAX_IMAGES_TO_DESCRIBE moved to the settings file beside this script.)



def convert_to_wav_for_stems(input_path: str, sample_rate: int = 44100, channels: int = 2, max_seconds=None) -> str:
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    cmd = ["ffmpeg", "-nostdin", "-y", "-i", input_path, "-ar", str(sample_rate), "-ac", str(channels)]
    if max_seconds is not None and max_seconds > 0:
        cmd += ["-t", str(max_seconds)]
    cmd.append(tmp.name)
    result = subprocess.run(cmd, capture_output=True, text=True, stdin=subprocess.DEVNULL)
    if result.returncode != 0:
        _remove_quietly(tmp.name)
        raise RuntimeError(f"ffmpeg stem-WAV conversion failed:\n{result.stderr}")
    return tmp.name


_MIDI_JSON_LOG_RE = re.compile(r"((?:note|hit) events \((?:MIDI|rhythm)-JSON\):\s*)(\[.*\])")
# The formats actually emitted today. STEM_MIDI_COMPACT_EVENT_FORMAT and
# FULL_MIDI_TRANSCRIPTION mean the JSON form above is never produced, so the
# original pattern matched nothing and this whole function was a no-op.
_MIDI_FULL_LIST_RE = re.compile(r"(full (note|hit) list \((\d+) (notes|hits)\):\s*)(.+)$")
_MIDI_COMPACT_LOG_RE = re.compile(r"((note|hit) events \(compact\):\s*)(.+)$")

# Substrings that indicate a line may carry an event log worth compressing.
# An already-sampled list, so a log can be compressed AGAIN when one pass was
# not enough. Without this the sampling rewrite was a one-shot: the new label
# ("note list sample (40 of 1893 notes...)") matches none of the patterns
# above, so a second pass returned the text untouched and anything still over
# budget fell through to blind head/tail truncation, which removes the MIDDLE
# of the analysis -- where the per-section evidence lives.
_MIDI_SAMPLE_LIST_RE = re.compile(
    r"((note|hit) list sample \((\d+) of (\d+) (notes|hits)[^)]*\):\s*)(.+)$"
)
_MIDI_LOG_MARKERS = ("events (", "full note list (", "full hit list (",
                     "note list sample (", "hit list sample (")


def _compress_stem_midi_event_logs(text, max_events_per_log=30):
    """Reduce huge stem/MIDI event logs before sending them to Ollama.

    Samples evenly rather than head-truncating, so coverage spans the whole
    track instead of stopping partway through.

    Handles three serialisations:
      * "note/hit events (MIDI-JSON|rhythm-JSON): [...]"  - JSON array
      * "note/hit events (compact): C4@1.2-1.5, ..."      - comma-separated
      * "full note/hit list (N notes|hits): C4@1.2-1.5 (amp=0.4), ..."

    Only the first was previously matched, and it is the one format the current
    config never emits — so oversized note lists reached _truncate_text_for_ollama
    intact and had their MIDDLE removed, losing an entire span of the track.
    Even sampling is strictly better: it keeps the track's shape end to end.

    When a full list is sampled its label is REWRITTEN to say so. That matters:
    WRITER_MUSIC_RULES tells the model a "full note list" covers every
    transcribed note and is not a sample, so leaving the original label on
    reduced data would make the model trust it as complete.
    """
    if not text:
        return text
    if not any(marker in text for marker in _MIDI_LOG_MARKERS):
        return text

    out_lines = []
    for line in str(text).splitlines():
        m = _MIDI_JSON_LOG_RE.search(line)
        if m:
            try:
                arr = json.loads(m.group(2))
                if isinstance(arr, list) and len(arr) > max_events_per_log:
                    arr = _sample_list_evenly(arr, max_events_per_log)
                line = line[:m.start()] + m.group(1) + json.dumps(arr) + line[m.end():]
            except Exception:
                pass
            out_lines.append(line)
            continue

        m = _MIDI_FULL_LIST_RE.search(line)
        if m:
            try:
                kind, total_str, unit, body = m.group(2), m.group(3), m.group(4), m.group(5)
                events = [e for e in (x.strip() for x in body.split(",")) if e]
                if len(events) > max_events_per_log:
                    kept = _sample_list_evenly(events, max_events_per_log)
                    label = (
                        f"{kind} list sample ({len(kept)} of {total_str} {unit}, "
                        "evenly spaced across the track — NOT a complete list): "
                    )
                    line = line[:m.start()] + label + ", ".join(kept)
            except Exception:
                pass
            out_lines.append(line)
            continue

        m = _MIDI_SAMPLE_LIST_RE.search(line)
        if m:
            try:
                kind, orig_total, unit, body = m.group(2), m.group(4), m.group(5), m.group(6)
                events = [e for e in (x.strip() for x in body.split(",")) if e]
                if len(events) > max_events_per_log:
                    kept = _sample_list_evenly(events, max_events_per_log)
                    # Keep the ORIGINAL total in the label, not the count of
                    # the previous sample, so the writer is never told a track
                    # has fewer notes than it does.
                    label = (
                        f"{kind} list sample ({len(kept)} of {orig_total} {unit}, "
                        "evenly spaced across the track — NOT a complete list): "
                    )
                    line = line[:m.start()] + label + ", ".join(kept)
            except Exception:
                pass
            out_lines.append(line)
            continue

        m = _MIDI_COMPACT_LOG_RE.search(line)
        if m:
            try:
                body = m.group(3)
                events = [e for e in (x.strip() for x in body.split(",")) if e]
                if len(events) > max_events_per_log:
                    kept = _sample_list_evenly(events, max_events_per_log)
                    line = line[:m.start()] + m.group(1) + ", ".join(kept)
            except Exception:
                pass
            out_lines.append(line)
            continue

        out_lines.append(line)

    return "\n".join(out_lines)


# --- Stem separation + Omnizart MIDI helpers ---------------------------------
_OMNIZART_APPS = None
OMNIZART_IMPORT_ERROR = ""

STEM_MIDI_STEMS = ("vocals", "bass", "guitar", "piano", "other", "drums")

# Pitched-stem filters run after Omnizart. Duration and relative stem level
# are multiplied by STEM_NOTE_FILTER_PRESET_WEIGHT, then clamped to the
# global floors. The comments below give the resulting default cutoffs.
# The onset threshold is used by the onset-band fallback for other and drums.
# Monophonic controls melody reporting for vocals and bass.
_STEM_MIDI_PRESETS_PRE6 = {
    "vocals": {
        "min_frequency": 80,
        "max_frequency": 1200,
        "minimum_note_length_ms": 75,   # 45 ms, to drop brief pitch fragments
        "min_amplitude": 0.10,          # 0.060 of the stem reference level
        "monophonic": True,
    },
    "bass": {
        "min_frequency": 27,
        "max_frequency": 300,
        "minimum_note_length_ms": 90,   # 54 ms, to drop short rumbles and bleed
        "min_amplitude": 0.12,          # 0.072 of the stem reference level
        "monophonic": True,
    },
    "guitar": {
        "min_frequency": 65,
        "max_frequency": 2000,
        "minimum_note_length_ms": 50,   # 30 ms, to retain quick picked notes
        "min_amplitude": 0.08,          # 0.048 of the stem reference level
        "monophonic": False,
    },
    "piano": {
        "min_frequency": 27,
        "max_frequency": 4200,
        "minimum_note_length_ms": 45,   # 27 ms, to retain ornaments and staccato
        "empty_rms_threshold": 1e-8,
        "min_level_reference_rms": 0.03,  # ~ -30.5 dBFS
        "min_abs_rms": 0.002,           # ~ -54 dBFS per note, to reject quiet bleed
        "min_amplitude": 0.07,          # 0.042 of the stem reference level
        "monophonic": False,
    },
    "other": {
        "min_frequency": 80,
        "max_frequency": 4000,
        "onset_threshold": 0.35,
        "onset_min_amplitude": 0.05,
        "min_level_reference_rms": 0.03,  # ~ -30.5 dBFS
        "minimum_note_length_ms": 85,   # 51 ms, to drop short mixed-stem artifacts
        "min_amplitude": 0.12,          # 0.072 of the stem reference level
        "monophonic": False,
    },
    "drums": {
        "onset_threshold": 0.10,
        "min_amplitude": 0.05,
    },
}
_STEM_MIDI_PRESETS_07 = {
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
        # Only skip numerical silence; whole-track loudness cannot establish
        # piano absence (quiet or brief passages must reach the transcriber).
        "empty_rms_threshold": 1e-8,
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
STEM_MIDI_PRESETS = _STEM_MIDI_PRESETS_PRE6 if STEM_MIDI_PRE6_TUNING else _STEM_MIDI_PRESETS_07


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
    disabled; callers must fall back to whole-mix analysis in that case.

    Gated on ENABLE_STEM_MIDI OR ENABLE_VOCAL_STEM_ISOLATION (not just the
    former) so that disabling instrument MIDI transcription doesn't also
    silently disable vocal isolation for pitch tracking -- the two are
    independent concerns that happened to share one Demucs call."""
    if not (ENABLE_STEM_MIDI or ENABLE_VOCAL_STEM_ISOLATION):
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


# (DEBUG_STEM_MIDI moved to the settings file beside this script.)
# False (default): silence Omnizart/TensorFlow/Keras progress bars and
# per-stem console chatter during stem MIDI. Set True only when debugging
# transcription failures.
# (SHOW_OMNIZART_LOGS moved to the settings file beside this script.)


def _pretty_midi_to_note_dicts(midi):
    """Convert an Omnizart pretty_midi.PrettyMIDI result (pitched stems:
    vocals/bass/guitar/piano/other) into this file's internal note-dict
    format: onset/offset in seconds, frequency in Hz, amplitude 0..1."""
    if midi is None:
        return []

    out = []
    skipped_placeholders = 0
    for instrument in midi.instruments:
        # Omnizart's vocal model (vocal/inference.py, infer_midi) writes every
        # onset it could not assign a pitch to as a fixed MIDI 77 (F5) note on a
        # separate is_drum track named "Missing Notes". Those are placeholders,
        # not sung notes: read as pitch they produced a phantom F5 "highest
        # note", impossible 2-octave leaps, and runs of F5 in sections where the
        # singer had stopped (which also made the vocal stem look active).
        if getattr(instrument, "is_drum", False) or (
            str(getattr(instrument, "name", "") or "").strip().lower() == "missing notes"
        ):
            skipped_placeholders += len(instrument.notes)
            continue
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
            # Omnizart's velocities are not measurements: the vocal model writes
            # a constant 80 and the music model squeezes its activation into
            # 110-127 (inverted). The model value is kept for reference only;
            # "amplitude" is replaced with a level measured from the stem audio
            # by _attach_measured_amplitudes() before any filtering/analysis.
            amp = max(0.0, min(1.0, float(note.velocity) / 127.0))
            out.append({
                "onset": start,
                "offset": end,
                "frequency": freq,
                "amplitude": amp,
                "model_velocity": int(note.velocity),
                "duration": end - start,
            })

    if DEBUG_STEM_MIDI:
        print(f"  (stem MIDI debug: Omnizart returned {len(out)} pitched note(s); "
              f"skipped {skipped_placeholders} unpitched placeholder note(s))")

    return sorted(out, key=lambda x: x["onset"])


# The only pitches Omnizart's three-class drum model ever writes.
_OMNIZART_DRUM_CLASS_PITCHES = frozenset({35, 38, 42})


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
        track_name = getattr(instrument, "name", "") or ""
        name_guess = _guess_drum_type_from_name(track_name)
        # Kept alongside the coarse guess so the finer subtype can be read off
        # the track name too, not only off GM pitch numbers.
        subtype_guess = _guess_drum_subtype_from_name(track_name)
        for note in instrument.notes:
            start = float(note.start)
            end = float(note.end)
            if end <= start:
                end = start + 0.05
            pitch = int(note.pitch)
            raw.append((start, end, pitch, int(note.velocity), name_guess, subtype_guess))
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

    # Omnizart's drum model (drum/inference.py) has exactly three outputs and
    # writes them as fixed GM pitches: 35 (kick), 38 (snare), 42 (hi-hat, which
    # also absorbs every cymbal). Those pitches are class IDs, not instrument
    # identifications, so mapping them through GM_DRUM_PITCH_SUBTYPE reported
    # "acoustic bass drum / acoustic snare / closed hi-hat" and "hi-hat closed
    # throughout" for every track ever scanned. Only attach a finer subtype when
    # the transcription actually distinguishes more than those three classes.
    three_class_only = bool(distinct_pitches) and distinct_pitches <= _OMNIZART_DRUM_CLASS_PITCHES

    out = []
    for start, end, pitch, velocity, name_guess, subtype_guess in raw:
        drum_type = name_guess or GM_DRUM_PITCH_TYPE.get(pitch) or pitch_rank_type.get(pitch, "other")
        # Finer class, when the pitch or track name identifies one. Carried as
        # a separate field so every existing consumer that groups on
        # drum_type keeps seeing exactly the buckets it saw before.
        subtype = subtype_guess or (None if three_class_only else GM_DRUM_PITCH_SUBTYPE.get(pitch))
        # Model velocity is a constant (Omnizart writes 100 for every hit); it is
        # kept for reference and replaced by a level measured from the drums
        # stem in _attach_measured_amplitudes().
        amp = max(0.0, min(1.0, float(velocity) / 127.0))
        entry = {
            "onset": start,
            "offset": end,
            "duration": max(end - start, 0.01),
            "amplitude": amp,
            "model_velocity": int(velocity),
            "drum_type": drum_type,
        }
        if subtype:
            entry["drum_subtype"] = subtype
        out.append(entry)

    if DEBUG_STEM_MIDI:
        print(f"  (stem MIDI debug: Omnizart drum model returned {len(out)} hit(s); "
              f"distinct raw pitches={sorted_pitches})")

    return sorted(out, key=lambda x: x["onset"])


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
    min_amplitude = float(preset.get("onset_min_amplitude", preset.get("min_amplitude", 0.015)))
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
            # Preserve band separation on older APIs too; never assign the
            # same full-band envelope to kick, snare AND hat.
            band_y = _band_filter(y, sr, fmin, fmax)
            env = librosa.onset.onset_strength(
                y=band_y, sr=sr, hop_length=hop_length, aggregate=np.mean,
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

    # Retain the legacy selection as provenance ONLY, for stable structure
    # and tempo evidence. All simultaneous bands still reach the drum report.
    legacy = []
    ordered = sorted(all_notes, key=lambda n: n["onset"])
    for n in ordered:
        if legacy and n["onset"] - legacy[-1]["onset"] < 0.02:
            if n["amplitude"] > legacy[-1]["amplitude"]:
                legacy[-1] = n
        else:
            legacy.append(n)
    keep_ids = {id(n) for n in legacy}
    # Backtracking may return the exact same onset twice in one band.
    # Only deduplicate that event, never a simultaneous different band.
    distinct = {}
    for n in ordered:
        n["structure_legacy_keep"] = id(n) in keep_ids
        key = (n["frequency"], n["onset"])
        old = distinct.get(key)
        if old is None or n["amplitude"] > old["amplitude"]:
            distinct[key] = n
        elif n["structure_legacy_keep"]:
            old["structure_legacy_keep"] = True
    return sorted(distinct.values(), key=lambda n: n["onset"])


_OMNIZART_OUTPUT_DIRS = []  # temp dirs Omnizart writes its .mid side-output into; cleaned up in main()'s finally block
_OMNIZART_MODEL_USED = {}   # stem -> Omnizart model/mode that produced its notes (for the report)

_OMNIZART_PREFETCH = {"proc": None, "out": None, "results": None, "paths": {}, "t0": None}


def _start_omnizart_prefetch(stems):
    """dev55: transcribe the non-silent stems in a separate process (see
    PARALLEL_ANALYSIS / OMNIZART_PREFETCH_PROCESS). Returns True when started."""
    if not (PARALLEL_ANALYSIS and OMNIZART_PREFETCH_PROCESS and ENABLE_STEM_MIDI and stems):
        return False
    try:
        script = os.path.abspath(__file__)
    except Exception:
        return False
    if not os.path.exists(script):
        return False
    todo = []
    for stem in STEM_MIDI_STEMS:
        path = stems.get(stem)
        if not path or not os.path.exists(path):
            continue
        try:
            preset = STEM_MIDI_PRESETS[stem]
            empty_rms = float(preset.get("empty_rms_threshold", 0) or 0)
            if empty_rms > 0:
                rms, _ = _stem_rms_by_section(path, None)
                if rms < empty_rms:
                    continue        # summaries skip it too
        except Exception:
            pass
        todo.append({"stem": stem, "path": os.path.abspath(path)})
    if not todo:
        return False
    try:
        tmpd = tempfile.mkdtemp(prefix="omnizart_prefetch_")
        _OMNIZART_OUTPUT_DIRS.append(tmpd)
        jin = os.path.join(tmpd, "in.json")
        jout = os.path.join(tmpd, "out.json")
        with open(jin, "w", encoding="utf-8") as f:
            json.dump({"stems": todo}, f)
        env = dict(os.environ)
        env["MUSICLYSE_OMNIZART_CHILD"] = "1"
        proc = subprocess.Popen(
            [sys.executable, script, "--omnizart-prefetch", jin, jout],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env,
            cwd=os.getcwd(),
        )
    except Exception:
        return False
    _OMNIZART_PREFETCH.update(proc=proc, out=jout, results=None, t0=time.perf_counter(),
                              paths={(d["path"], d["stem"]) for d in todo})
    return True


def _omnizart_prefetch_results():
    """Wait for the prefetch process (once) and return its results dict
    {(path, stem): (notes, model_used)} -- empty when it failed."""
    st = _OMNIZART_PREFETCH
    if st.get("results") is not None:
        return st["results"]
    proc = st.get("proc")
    res = {}
    if proc is not None:
        try:
            proc.wait(timeout=float(OMNIZART_PREFETCH_TIMEOUT_S))
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
        try:
            if st.get("t0") is not None:
                _STAGE_LOG.append(("[background, overlapped] Omnizart stem transcription (separate process)",
                                   time.perf_counter() - st["t0"]))
        except Exception:
            pass
        try:
            if proc.returncode == 0 and st.get("out") and os.path.exists(st["out"]):
                with open(st["out"], "r", encoding="utf-8") as f:
                    data = json.load(f)
                for item in data.get("results") or []:
                    res[(item["path"], item["stem"])] = (item.get("notes") or [], item.get("model_used"))
        except Exception:
            res = {}
    st["results"] = res
    st["proc"] = None
    try:
        if st.get("out"):
            shutil.rmtree(os.path.dirname(st["out"]), ignore_errors=True)
    except Exception:
        pass
    return res


def _clear_omnizart_prefetch():
    proc = _OMNIZART_PREFETCH.get("proc")
    if proc is not None:
        try:
            if proc.poll() is None:
                proc.kill()
        except Exception:
            pass
    _OMNIZART_PREFETCH.update(proc=None, out=None, results=None, paths={}, t0=None)


def _omnizart_prefetch_child(in_json, out_json):
    """Entry for `--omnizart-prefetch in.json out.json` (see
    _start_omnizart_prefetch): the same omnizart_transcribe() as the in-order
    run, one stem after another, results written as JSON."""
    with open(in_json, "r", encoding="utf-8") as f:
        todo = (json.load(f) or {}).get("stems") or []
    results = []
    for item in todo:
        stem, path = item["stem"], item["path"]
        try:
            notes = _omnizart_transcribe_impl(path, stem, STEM_MIDI_PRESETS.get(stem, {}))
        except Exception:
            notes = []
        results.append({"stem": stem, "path": path, "notes": notes or [],
                        "model_used": _OMNIZART_MODEL_USED.get(stem)})
    tmp = out_json + ".part"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"results": results}, f, default=float)
    os.replace(tmp, out_json)
    for d in list(_OMNIZART_OUTPUT_DIRS):      # this process's own scratch dirs
        try:
            shutil.rmtree(d, ignore_errors=True)
        except Exception:
            pass


def omnizart_transcribe(stem_path: str, stem: str, preset: dict):
    """Transcribe one separated stem with Omnizart (list of note dicts, or []
    on failure). dev55: uses the background process's result for this stem
    when it produced one; otherwise transcribes here, exactly as before."""
    key = (os.path.abspath(str(stem_path)), stem)
    if key in (_OMNIZART_PREFETCH.get("paths") or ()):
        got = _omnizart_prefetch_results().get(key)
        if got is not None and got[0]:
            if got[1] is not None:
                _OMNIZART_MODEL_USED[stem] = got[1]
            return [dict(n) for n in got[0]]
    return _omnizart_transcribe_impl(stem_path, stem, preset)


def _omnizart_transcribe_impl(stem_path: str, stem: str, preset: dict):
    """
    Transcribe one separated stem with Omnizart.
    Returns a list of note dicts (or empty list on any failure).
    """
    import pretty_midi
    import tempfile

    try:
        apps = _get_omnizart()
    except Exception:
        return []

    app_key = OMNIZART_STEM_APP.get(stem, "music")
    try:
        app = apps[app_key]
    except Exception:
        return []

    out_dir = tempfile.mkdtemp(prefix="omnizart_")
    _OMNIZART_OUTPUT_DIRS.append(out_dir)

    def _load_midi_candidate(candidate):
        if candidate is None:
            return None
        if isinstance(candidate, pretty_midi.PrettyMIDI):
            return candidate
        if isinstance(candidate, (str, os.PathLike)):
            candidate = os.fspath(candidate)
            if os.path.isfile(candidate):
                try:
                    return pretty_midi.PrettyMIDI(candidate)
                except Exception:
                    return None
        return None

    # Run Omnizart. The music app defaults to its PIANO checkpoint when no
    # model is named; OMNIZART_MUSIC_MODEL_BY_STEM can name another mode
    # (e.g. "Pop") per stem. A named mode that fails to load falls back to the
    # default call, so a missing checkpoint never costs the stem its notes.
    result = None
    model_mode = None
    if app_key == "music":
        model_mode = (globals().get("OMNIZART_MUSIC_MODEL_BY_STEM") or {}).get(stem)
    attempts = []
    if model_mode:
        attempts.append({"model_path": model_mode, "output": out_dir})
    attempts += [{"output": out_dir}, {}]
    used_mode = None
    with quiet_stdout():
        for kwargs in attempts:
            try:
                result = app.transcribe(stem_path, **kwargs)
                used_mode = kwargs.get("model_path")
                break
            except TypeError:
                continue
            except Exception:
                if kwargs.get("model_path"):
                    continue
                return []
    _OMNIZART_MODEL_USED[stem] = used_mode or ("default (Piano)" if app_key == "music" else app_key)

    # Try to get the MIDI object
    midi_obj = _load_midi_candidate(result)

    if midi_obj is None:
        # Search the output directory for the side-output Omnizart wrote.
        # Prefer the newest MIDI file: Omnizart may leave auxiliary files.
        candidates = []
        for root, _, files in os.walk(out_dir):
            for filename in files:
                if filename.lower().endswith((".mid", ".midi")):
                    candidates.append(os.path.join(root, filename))
        candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        for candidate in candidates:
            midi_obj = _load_midi_candidate(candidate)
            if midi_obj is not None:
                break

    if midi_obj is None:
        return []

    try:
        if stem == "drums":
            notes = _drum_pretty_midi_to_note_dicts(midi_obj)
        else:
            notes = _pretty_midi_to_note_dicts(midi_obj)
        return notes if notes is not None else []
    except Exception:
        return []


def _attach_measured_amplitudes(notes, stem_path, stem):
    """Give every note/hit an amplitude MEASURED from the separated stem.

    Omnizart does not estimate loudness: drums are always velocity 100, vocals
    always 80, and the music model maps its activation into 110-127 (inverted).
    Everything that read "amplitude" -- the velocity-spread / programmed-vs-
    played line, the per-stem amplitude floor that is meant to drop low-level
    bleed, and the saved note data -- was therefore reading a constant.

    Level = peak band-limited RMS in a short window after the onset (50 ms for
    drum hits, the note itself up to 1 s for pitched notes), relative to the 99th-percentile
    frame level of the same band in the same stem. Piano and other use a
    minimum reference level so quiet separation bleed cannot scale itself to
    full amplitude. Drum hits are measured inside
    their own drum's frequency band so a kick is not credited with the hat on
    top of it. The model value stays in "model_velocity". Notes are modified in
    place and also returned. On any failure the notes are returned untouched.
    """
    if not notes or not stem_path or not os.path.exists(str(stem_path)):
        return notes
    try:
        y, sr = librosa.load(str(stem_path), sr=None, mono=True)
    except Exception:
        return notes
    if y is None or y.size < 4096:
        return notes
    hop = 256
    is_drums = (stem == "drums")
    min_ref = STEM_MIDI_PRESETS.get(stem, {}).get("min_level_reference_rms", 0.0)
    cache = {}
    # Absolute (time-domain) level as well: a stem that is separation residue
    # throughout is loud RELATIVE to itself, so only an absolute floor can tell
    # its "notes" from playing (see STEM_NOTE_MIN_ABS_RMS). Drums never go
    # through that filter, so only pitched stems need it.
    abs_env = _rms_envelope(y, frame=2048, hop=hop) if not is_drums else np.zeros(0)

    def _env(key):
        if key not in cache:
            if is_drums:
                lo, hi = _DRUM_ANALYSIS_BANDS.get(key, (30.0, 16000.0))
                env = _rms_envelope(_band_filter(y, sr, lo, hi), frame=1024, hop=hop)
            else:
                env = abs_env
            nz = env[env > 0]
            ref = float(np.percentile(nz, 99)) if nz.size else 0.0
            cache[key] = (env, max(ref, min_ref))
        return cache[key]

    for n in notes:
        try:
            onset = float(n["onset"])
        except (KeyError, TypeError, ValueError):
            continue
        key = (n.get("drum_type") or "other") if is_drums else "_all"
        env, ref = _env(key)
        if ref <= 0 or env.size == 0:
            continue
        if is_drums:
            win = 0.05
        else:
            # Over the note itself (up to 1 s), not just its first instants:
            # a slow-attack note (string swell, pad) is still rising 150 ms in
            # and would read far quieter than it gets.
            win = min(max(float(n.get("duration") or 0.0), 0.03), 1.0)
        a = int(max(0.0, onset) * sr / hop)
        b = max(a + 1, int((onset + win) * sr / hop) + 1)
        if a >= env.size:
            continue
        lvl = float(np.max(env[a:min(b, env.size)]))
        n["amplitude"] = round(float(min(1.0, lvl / ref)), 4)
        n["amplitude_source"] = "stem audio"
        if a < abs_env.size:
            n["abs_rms"] = round(float(np.max(abs_env[a:min(b, abs_env.size)])), 6)
    return notes


def _stabilize_vocal_midi(notes, stem_path, preset):
    """Correct adjacent-semitone errors only where the vocal contour settles."""
    if not notes:
        return notes, 0, 0

    sr, frame, hop = 11025, 2048, 128
    y, _ = librosa.load(str(stem_path), sr=sr, mono=True)
    if len(y) < frame:
        return notes, 0, 0

    # Analyze overlapping windows in bounded chunks. Only keep frames whose
    # centers belong to each chunk, so joins never double-count pitch evidence.
    chunk = 20 * sr
    pad = (sr // (4 * hop)) * hop
    chunk = (chunk // hop) * hop
    times, contour = [], []
    for start in range(0, len(y), chunk):
        left, stop = max(0, start - pad), min(len(y), start + chunk)
        right = min(len(y), stop + pad)
        if right - left < frame:
            continue
        f0, _, prob = librosa.pyin(
            y[left:right], sr=sr, fmin=65.0, fmax=1200.0,
            frame_length=frame, hop_length=hop,
        )
        centers = left + np.arange(len(f0)) * hop
        keep = (centers >= start) & (centers < stop)
        f0 = np.where(np.isfinite(f0) & (prob >= 0.65), f0, np.nan)
        times.extend(centers[keep] / sr)
        contour.extend(librosa.hz_to_midi(f0[keep]))

    times, contour = np.asarray(times), np.asarray(contour)
    out, corrected, merged = [], 0, 0
    for note in notes:
        start, end = note["onset"], note["offset"]
        duration = end - start
        # Ignore attack/release bends while retaining most of short notes.
        edge = min(0.045, 0.18 * duration)
        a, z = times.searchsorted([start + edge, end - edge])
        core = contour[a:z]
        voiced = core[np.isfinite(core)]
        if len(voiced) >= max(5, 0.7 * len(core)):
            original = int(round(float(librosa.hz_to_midi(note["frequency"]))))
            target = int(round(float(np.median(voiced))))
            target_hz = float(librosa.midi_to_hz(target))
            in_range = preset["min_frequency"] <= target_hz <= preset["max_frequency"]
            if in_range and abs(target - original) == 1 and np.mean(np.abs(voiced - target) <= 0.45) >= 0.75:
                note = dict(note, frequency=target_hz,
                            raw_midi_pitch=original)
                corrected += 1

        # Join only touching fragments when a short piece was pitch-corrected.
        # Preserve repeated syllables and distinct, well-supported notes.
        if out:
            prev = out[-1]
            gap = start - prev["offset"]
            short_correction = any(
                "raw_midi_pitch" in n and n["duration"] <= 0.14
                for n in (prev, note)
            )
            if (short_correction and -0.025 <= gap <= 0.025
                    and abs(prev["frequency"] - note["frequency"]) < 0.1):
                offset = max(prev["offset"], end)
                out[-1] = dict(prev, offset=offset, duration=offset - prev["onset"],
                               amplitude=max(prev.get("amplitude", 0), note.get("amplitude", 0)))
                merged += 1
                continue
        out.append(note)
    return out, corrected, merged


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

    # Per-stem duration/amplitude floors, scaled down from the raw preset
    # values (see STEM_NOTE_FILTER_PRESET_WEIGHT) so instruments keep some
    # differentiation -- e.g. bass gets a longer floor than guitar/piano
    # to reject short artifacts -- without reintroducing the over-filtering
    # that came from applying the full preset values. Never goes below the
    # global floors that were found to work well on their own.
    preset_min_duration_s = float(preset.get("minimum_note_length_ms", 0) or 0) / 1000.0
    preset_min_amplitude = float(preset.get("min_amplitude", 0) or 0)
    min_duration_s = max(
        STEM_NOTE_MIN_DURATION_SEC_FLOOR,
        preset_min_duration_s * STEM_NOTE_FILTER_PRESET_WEIGHT,
    )
    min_amplitude = max(
        STEM_NOTE_MIN_AMPLITUDE_FLOOR,
        preset_min_amplitude * STEM_NOTE_FILTER_PRESET_WEIGHT,
    )
    min_abs_rms = max(STEM_NOTE_MIN_ABS_RMS, float(preset.get("min_abs_rms", 0)))

    filtered = []
    removed = 0

    for n in notes:
        # 1. Frequency Range Check (Keep this to remove sub-bass noise/overtones)
        if not (min_freq <= n["frequency"] <= max_freq):
            removed += 1
            continue
        
        # 2. Duration Check: per-stem floor (scaled down from preset; see above),
        # still permissive enough to catch staccato/fast runs.
        if n["duration"] < min_duration_s:
            removed += 1
            continue
            
        # 3. Amplitude Check: per-stem floor (scaled down from preset; see above),
        # still permissive enough to keep quiet vocals and soft piano notes.
        # "amplitude" is measured from the stem audio (see
        # _attach_measured_amplitudes); the transcriber's own velocities were
        # constants, so before that this check could never remove anything.
        if n.get("amplitude", 1.0) < min_amplitude:       
            removed += 1
            continue

        # 4. Absolute level: drop "notes" in passages where the stem is
        # effectively silent (separation residue), however loud they are
        # relative to the rest of a residue-only stem.
        if n.get("abs_rms") is not None and float(n["abs_rms"]) < min_abs_rms:
            removed += 1
            continue
            
        filtered.append(n)

    return sorted(filtered, key=lambda x: x["onset"]), removed


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
      
        
def _notes_to_full_log(notes):
    """Return a compact string of all filtered notes: 'note_name@onset-offset (amp=...)', separated by commas."""
    if not notes:
        return ""
    parts = []
    for n in notes:
        try:
            name = librosa.hz_to_note(float(n["frequency"]))
        except Exception:
            name = f"{float(n['frequency']):.1f}Hz"
        onset = round(float(n["onset"]), 2)
        offset = round(float(n["offset"]), 2)
        amp = round(float(n.get("amplitude", 1.0)), 3)
        parts.append(f"{name}@{onset}-{offset} (amp={amp})")
    return ", ".join(parts)
    
    
def _stem_notes_to_json(stem_notes, max_per_stem=None):
    """Serialise raw per-stem transcribed MIDI for the saved song .json.

    Compact keys keep the file to a sane size while remaining fully
    self-describing:
      t   onset seconds
      d   duration seconds
      m   MIDI note number
      n   note name
      v   normalised amplitude / velocity proxy
      k   drum type (percussive stems only)

    This is what makes a saved song reusable: without the note events, a
    reloaded track could only ever be discussed through whatever prose summary
    happened to be written at analysis time, and any melody question asked
    later had nothing underneath it. With FULL_MIDI_TRANSCRIPTION on, every
    detected note in every stem is stored.
    """
    if not stem_notes:
        return {}
    out = {}
    for stem, notes in (stem_notes or {}).items():
        if not notes:
            continue
        seq = _notes_sorted(notes)
        if max_per_stem and len(seq) > max_per_stem:
            seq = _evenly_sample_notes(seq, max_per_stem)
        rows = []
        for n in seq:
            try:
                onset = round(float(n.get("onset", 0.0)), 3)
                freq = float(n.get("frequency", 0.0))
            except Exception:
                continue
            dur = n.get("duration")
            if dur is None:
                try:
                    dur = float(n.get("offset", onset)) - onset
                except Exception:
                    dur = 0.0
            row = {
                "t": onset,
                "d": round(float(dur or 0.0), 3),
                "v": round(float(n.get("amplitude", 1.0) or 1.0), 3),
            }
            drum_type = n.get("drum_type")
            if drum_type:
                row["k"] = str(drum_type)
                if "structure_legacy_keep" in n:
                    row["sl"] = bool(n["structure_legacy_keep"])
            else:
                midi_num = _note_midi_number(freq)
                if midi_num is not None:
                    row["m"] = midi_num
                    row["n"] = _hz_to_note_name(freq)
                else:
                    row["hz"] = round(freq, 2)
            rows.append(row)
        if rows:
            out[stem] = rows
    return out


def _json_notes_to_stem_notes(saved):
    """Inverse of _stem_notes_to_json: rebuild the internal note dicts from a
    saved song file so a reloaded track supports the same melodic analysis as a
    freshly scanned one."""
    if not saved or not isinstance(saved, dict):
        return {}
    out = {}
    for stem, rows in saved.items():
        if not isinstance(rows, list):
            continue
        notes = []
        for r in rows:
            if not isinstance(r, dict):
                continue
            try:
                onset = float(r.get("t", 0.0))
                dur = float(r.get("d", 0.0))
            except Exception:
                continue
            if "m" in r:
                try:
                    freq = 440.0 * (2.0 ** ((float(r["m"]) - 69) / 12.0))
                except Exception:
                    continue
            elif "hz" in r:
                try:
                    freq = float(r["hz"])
                except Exception:
                    continue
            else:
                freq = 0.0
            note = {
                "onset": onset,
                "offset": onset + dur,
                "duration": dur,
                "frequency": freq,
                "amplitude": float(r.get("v", 1.0) or 1.0),
            }
            if STRUCTURE_IGNORE_SILENT_DRUM_HITS and r.get("k") and "v" in r:
                try:                            # 0.7.1-pre13: a saved level of 0 stays 0 for the section map
                    if float(r["v"]) == 0.0:
                        note["level"] = 0.0
                except Exception:
                    pass
            if r.get("k"):
                note["drum_type"] = r["k"]
                if isinstance(r.get("sl"), bool):
                    note["structure_legacy_keep"] = r["sl"]
            notes.append(note)
        if notes:
            out[stem] = notes
    return out


def _confirmed_vocal_onset(notes, min_notes=5, window_seconds=3.0, max_range_semitones=12,
                            max_gap_seconds=1.0, min_distinct_pitches=3):
    """The literal earliest onset in a transcribed vocal stem is not always
    the true entrance. A sustained harmonic or percussive element bleeding
    into the vocal stem from elsewhere in the mix (a ringing cymbal, a guitar
    harmonic) can register as one or more notes before the real voice ever
    starts, and it does not look like a sung phrase once you look at what
    surrounds it: either the same pitch repeated many times with no melodic
    movement, or a scatter of notes leaping by huge, non-melodic intervals.
    A genuine sung entrance instead shows several notes close together in
    time, moving in a fairly narrow pitch range with real (but not wild)
    variation -- confirmed against a real case where the reported onset
    (an isolated note followed by a two-octave leap) was ~16 seconds early.

    Walks the notes in onset order and returns the first one that begins a
    window meeting all of: enough notes close together in time (no gap
    bigger than max_gap_seconds), a tight overall pitch range (a real
    melodic phrase does not span more than an octave in a few seconds), and
    genuine pitch movement (not one repeated tone). Falls back to the literal
    earliest onset if nothing in the stem satisfies this -- reporting a
    possibly-noisy true earliest note is better than reporting nothing, and
    this matched the literal-earliest onset exactly on every other track
    checked, so it only moves the answer when the leading note actually
    looks unlike a sung phrase.
    """
    try:
        ordered = sorted(notes, key=lambda n: n["onset"])
        for i in range(len(ordered)):
            t0 = ordered[i]["onset"]
            win = [n for n in ordered[i:] if n["onset"] < t0 + window_seconds]
            if len(win) < min_notes:
                continue
            gaps = [win[j + 1]["onset"] - win[j]["onset"] for j in range(len(win) - 1)]
            if gaps and max(gaps) > max_gap_seconds:
                continue
            midi_pitches = [
                69 + 12 * math.log2(max(1.0, float(n.get("frequency", 0) or 0)) / 440.0)
                for n in win
            ]
            pitch_range = max(midi_pitches) - min(midi_pitches)
            if pitch_range > max_range_semitones:
                continue
            if len({round(p) for p in midi_pitches}) < min_distinct_pitches:
                continue  # a single held/repeated tone, not a sung phrase
            return float(ordered[i]["onset"])
    except Exception:
        pass
    return float(min(n["onset"] for n in notes)) if notes else 0.0


def summarize_stem_midi(stem, raw_notes, filtered, removed, preset, stem_rms=None, bpm=None,
                         sections=None, section_rms=None, key=None, stem_path=None, joined_fragments=0):
    """Build a compact per-stem summary. Returns (text, meta_dict) where meta
    holds activity stats used for the cross-stem prominence ranking.

    `stem_path` (optional): the separated stem's own audio file. Required for
    the drum production/timbre analysis, which measures how the kit SOUNDS
    rather than what it plays. That analysis referenced an undefined `stems`
    name, so every call raised NameError into a bare except and the whole
    drum-sound block -- kick weight, snare decay and room, hat character, bus
    compression, programmed-vs-played -- was silently dropped from every scan
    ever produced. Nothing described drum tone, so tone questions were answered
    from genre habit.

    `stem_rms` is now the WHOLE-TRACK energy reading (see
    _stem_rms_by_section) -- it used to silently only cover the first ~60s.
    `section_rms` (optional): per-section RMS list from _stem_rms_by_section,
    used to render an INTENSITY BY SECTION block for every stem, not just
    drums. `key` (optional): track key string, used for scale-degree framing
    in melodic-line summaries.
    """
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
        lines.append(f"stem energy (RMS, whole track): {float(stem_rms):.5f}")

    intensity_block = _format_intensity_by_section(stem, section_rms) if section_rms else ""
    if intensity_block:
        lines.append(intensity_block)

    if not raw_notes:
        lines.append("No reliable MIDI notes detected.")
        meta["silent"] = True
        return "\n".join(lines), meta

    # raw_notes already contains the removed notes (filtered = raw - removed).
    # Adding `removed` again understated the removal ratio.
    total_raw = len(raw_notes)
    removed_ratio = removed / total_raw if total_raw > 0 else 0.0

    quality = "medium"
    if len(filtered) < 10 or removed_ratio > 0.6:
        quality = "low"
    elif len(filtered) >= 30 and removed_ratio < 0.25:
        quality = "high"

    # This rating is how many notes survived the range/length/level filters
    # and how many there are -- it says nothing about whether the transcriber
    # got the notes right, so it must not be worded as transcription confidence.
    lines.append(
        f"note count after filtering: {len(filtered) + joined_fragments} (raw {len(raw_notes)}, filtered out {removed}); "
        f"filter retention: {quality} (how much of the transcription survived the range/"
        "length/level filters -- not a measure of transcription accuracy)"
    )

    if stem == "drums" and any(n.get("detected_from") == "onset-band fallback" for n in (raw_notes or [])):
        lines.append(
            "NOTE: Omnizart returned no drum hits for this track, so the hits below come from a "
            "band-split onset detector. 'kick'/'snare'/'hihat' here mean low/mid/high-band hits "
            "(mid can include toms); treat the kit labels and grid families as approximate."
        )
    if stem == "drums":
        lines.append(
            "Drum/percussion stem: Omnizart drum transcription (not pitched MIDI). "
            "Use hit-type breakdown, per-type spacing, and the pattern sample below "
            "for rhythm/groove discussion. IMPORTANT LIMIT: this transcription model "
            "resolves only three classes -- kick, snare and hi-hat. Toms, crashes, "
            "rides, shakers, claps and other percussion are NOT detectable here and "
            "will be silently folded into those three or missed entirely, so the "
            "breakdown below is a partial view of the kit, not a complete inventory. "
            "Never state or imply that a track has no toms, no cymbals or no "
            "percussion on the strength of this list, and do not present the three "
            "classes as the full kit; if the whole-mix or drum-stem instrument tags "
            "name cymbals, tambourine or other percussion, believe those instead."
        )

    if not filtered:
        lines.append("No notes remained after filtering.")
        meta["silent"] = True
        return "\n".join(lines), meta

    if stem == "vocals":
        # Measured directly from the transcribed vocal stem's note onsets --
        # not a guess. Surfaced as its own explicit line (rather than left
        # buried in the full note list) because the listening model has no
        # reliable temporal grounding and defaults to assuming an
        # instrumental intro unless told otherwise, even when the vocal
        # stem shows a note starting within the first few seconds.
        _first_vocal_onset = _confirmed_vocal_onset(filtered)
        lines.append(
            f"EARLIEST VOCAL ONSET: {_seconds_to_mmss(_first_vocal_onset)} "
            f"({_first_vocal_onset:.1f}s) -- measured directly from the transcribed vocal "
            "stem, not an impression. If this is only a few seconds into the track, vocals "
            "are present at or near the very start -- do NOT describe the opening/intro as "
            "purely instrumental, or say vocals 'enter later' / 'after an instrumental intro', "
            "when this measured value is small. Trust this measurement over any general "
            "assumption that songs typically open with an instrumental intro before the voice. "
            "Exception: when the vocal pass reports VOCALS_PRESENT=no, this onset is separation "
            "residue (instruments leaking into the vocals stem), not a voice."
        )

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

        # When the kit enters, drops out, and changes shape. Independent of the
        # section map, so it still answers "when do the drums come in" on
        # tracks whose structure is unlabelled or generically named.
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

        # Recover cymbal onsets from the audio and add them as an extra
        # drum_type. Everything downstream (grids, arrangement timeline,
        # section grooves) then gets a cymbal row for free, and the
        # transcribed kick/snare/hat data is left exactly as measured.
        try:
            _cym = detect_cymbal_onsets(
                stem_path, _structure_stem_evidence({"drums": filtered})["drums"])
            if _cym:
                filtered = list(filtered) + [
                    {"onset": t, "duration": 0.25, "amplitude": 0.5,
                     "drum_type": "cymbal", "detected_from": "audio"}
                    for t in _cym
                ]
                filtered.sort(key=lambda n: float(n.get("onset") or 0.0))
                # Hand the augmented list back so section grids, the arrangement
                # timeline and the saved MIDI all include the cymbal row. It was
                # previously local to this function only.
                meta["augmented_notes"] = filtered
                lines.append(
                    f"CYMBALS (detected from the drums-stem audio, not from the note "
                    f"transcription, which cannot represent them): {len(_cym)} cymbal "
                    "hits located by high-band sustain. These appear as a 'cymbals' row "
                    "in the beat grids and arrangement timeline below. Treat their "
                    "TIMING as measured but their exact type (crash vs ride vs open "
                    "hat) as undetermined. KNOWN LIMIT: this finds ACCENTED cymbals -- "
                    "crashes marking a section or an accent -- because it works by "
                    "spotting a hit whose top end rings on above the track's usual "
                    "level. A ride or hat pattern played continuously RAISES that usual "
                    "level, so a steady ride is generally NOT captured here. A missing "
                    "cymbals row therefore means no accented cymbal was found, never "
                    "that the track has no cymbals; if the ADDITIONAL PERCUSSION block "
                    "reports sustained cymbals, believe that for presence and use this "
                    "row only for placement."
                )
        except Exception as e:
            print(f"  (cymbal onset detection skipped: {e})")

        # Kit entry/exit timeline, built after cymbal detection so the
        # 'cymbals' row the note above promises is really there.
        try:
            lines.extend(_summarize_drum_entry_timeline(filtered, bpm=bpm))
        except Exception:
            pass

        # One representative grid for the track as a whole, so a beat diagram
        # can still be given when no section map exists.
        try:
            _grid = build_beat_grid_diagram(filtered, bpm, label="representative bars")
            if _grid:
                lines.append(_grid)
        except Exception:
            pass

        # Kit pieces the three-class transcription cannot see at all.
        try:
            _extras = _format_drum_extras(_drum_extras_probe(stem_path))
            if _extras:
                lines.append(_extras)
        except Exception:
            pass

        # How the kit SOUNDS, measured from the separated drums audio itself.
        # The rest of this block covers what is played; nothing described the
        # tone, so drum production was previously answered from genre habit.
        try:
            _drum_sound = build_drum_production_report(
                stem_path, filtered, bpm=bpm
            )
            if _drum_sound:
                lines.append("")
                lines.append(_drum_sound)
        except Exception as e:
            # Loud on purpose: this block failed silently for a long time.
            print(f"  (drum sound analysis skipped: {e})")

        if quality == "low" or removed_ratio > 0.5:
            lines.append("Caution: this stem contains likely spurious/erratic onsets; treat only broad rhythmic tendencies as reliable.")

        # --- Drum event inclusion ---
        if FULL_MIDI_TRANSCRIPTION:
            # True full transcription: every filtered hit, not an evenly
            # sampled subset (that would just be a bigger "limited" mode).
            log = _drum_events_to_full_log(filtered)
            if log:
                lines.append(
                    f"full hit list ({len(filtered)} hits): {log}"
                )
        elif STEM_MIDI_INCLUDE_EVENT_LOGS:
            event_log = _drum_events_to_event_log(filtered, max_notes=STEM_MIDI_EVENT_LOG_MAX_NOTES)
            if event_log:
                label = "hit events (compact)" if STEM_MIDI_COMPACT_EVENT_FORMAT else "hit events (rhythm-JSON)"
                lines.append(f"{label}: {event_log}")

        return "\n".join(lines), meta
        

    # --- Pitched stems (vocals/bass/guitar/piano/other) ---
    if stem == "other" and (
        all(n.get("detected_from") == "onset-band fallback" for n in filtered)
        or not any(
            preset.get("min_frequency", 0) <= n["frequency"] <= preset.get("max_frequency", 100000)
            for n in filtered
        )
    ):
        # Onset-fallback mode: report hit density only, skip pitch stats. The
        # fallback's "frequency" is a band proxy, not a pitch, so this must key
        # on the explicit flag -- the pitch-range test alone never fired.
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
        # --- Note event inclusion ---
        if FULL_MIDI_TRANSCRIPTION:
            # True full transcription: every filtered event, not a capped sample.
            log = _notes_to_full_log(filtered)
            if log:
                lines.append(
                    f"full note list ({len(filtered)} notes): {log}"
                )
        elif STEM_MIDI_INCLUDE_EVENT_LOGS:
            event_log = _notes_to_event_log(filtered, max_notes=STEM_MIDI_EVENT_LOG_MAX_NOTES)
            if event_log:
                label = "note events (compact)" if STEM_MIDI_COMPACT_EVENT_FORMAT else "note events (MIDI-JSON)"
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
        parsed_key = _parse_key_root_quality(key)
        chord_parts = []
        for label, count in chords:
            part = f"{label} x{count}"
            if parsed_key:
                m = re.match(r"([A-G]#?)\s+(major|minor)", label)
                if m:
                    letter = m.group(1)
                    root_pc = _NOTE_LETTER_PC.get(letter[0].upper())
                    if root_pc is not None:
                        if len(letter) > 1 and letter[1] == "#":
                            root_pc += 1
                        deg = _scale_degree_label(root_pc % 12, parsed_key[0], parsed_key[1])
                        part += f" [chord root = scale degree {deg}]"
            chord_parts.append(part)
        lines.append("common harmonic clusters: " + "; ".join(chord_parts))
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
        mel = _summarize_melody(line_notes, max_notes=melody_budget, key_text=key, bpm=bpm)
        if mel:
            # Tag stem so the writer can talk about "the vocal line" vs "the bass line"
            lines.append(f"{stem} line: {mel}")
    else:
        low_notes = _lowest_line(filtered)
        mel = _summarize_melody(low_notes, max_notes=melody_budget, key_text=key, bpm=bpm)
        if mel:
            label = (
                f"{stem} lowest-line contour (supporting/inner line, not necessarily the tune)"
                if FULL_MIDI_TRANSCRIPTION
                else "lowest-line contour (not necessarily melody)"
            )
            lines.append(f"{label}: {mel}")

        # Full MIDI only: also sample a higher line so poly stems aren't only "bass of the chord"
        if FULL_MIDI_TRANSCRIPTION and len(filtered) >= 8:
            try:
                # Highest note in small time buckets ≈ upper voice sketch
                high_notes = sorted(filtered, key=lambda n: (-float(n["frequency"]), float(n["onset"])))
                # Re-sort chronologically after taking a frequency-biased sample
                high_sample = _evenly_sample_notes(
                    sorted(high_notes[: max(24, len(filtered) // 3)], key=lambda n: float(n["onset"])),
                    max(8, melody_budget // 2),
                )
                high_mel = _summarize_melody(high_sample, max_notes=max(8, melody_budget // 2),
                                             key_text=key, bpm=bpm)
                if high_mel:
                    lines.append(
                        f"{stem} upper-line sketch (approx top voice): "
                        + high_mel.replace("melodic line", "line", 1)
                    )
            except Exception:
                pass

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

    # --- Note/hit event inclusion ---
    if FULL_MIDI_TRANSCRIPTION:
        # True full transcription: every filtered event, not a capped sample.
        # (For monophonic stems we still select just the monophonic line --
        # that's picking the right notes, not limiting how many of them we
        # report -- and then report all of it.)
        if stem == "drums":
            log = _drum_events_to_full_log(filtered)
            if log:
                lines.append(
                    f"full hit list ({len(filtered)} hits): {log}"
                )
        else:
            if preset.get("monophonic"):
                pub_src = _select_monophonic_line(filtered) or filtered
            else:
                pub_src = filtered
            log = _notes_to_full_log(pub_src)
            if log:
                lines.append(
                    f"full note list ({len(pub_src)} notes): {log}"
                )
    elif STEM_MIDI_INCLUDE_EVENT_LOGS:
        # Original sampled event log (unchanged)
        if stem == "drums":
            event_log = _drum_events_to_event_log(filtered, max_notes=STEM_MIDI_EVENT_LOG_MAX_NOTES)
        else:
            event_log = _notes_to_event_log(filtered, max_notes=STEM_MIDI_EVENT_LOG_MAX_NOTES)
        if event_log:
            label = "hit events (compact)" if STEM_MIDI_COMPACT_EVENT_FORMAT else "hit events (rhythm-JSON)"
            lines.append(f"{label}: {event_log}")

    return "\n".join(lines), meta


def _drum_stem_bleed_db(activity_metas):
    """The drums stem's level against all stems combined (dB, negative), when
    it is at least DRUM_STEM_BLEED_BELOW_TOTAL_DB down -- far quieter than any
    drum part: on every band recording tested it sits 1-10 dB below, on the
    acoustic recordings 21 and 54 dB below. None otherwise, or when off."""
    if not DRUM_BLEED_NOTE or not activity_metas:
        return None
    rms = {m.get("stem"): float(m.get("rms") or 0.0) for m in activity_metas if m.get("stem")}
    if "drums" not in rms:
        return None
    total = float(np.sqrt(sum(v * v for v in rms.values()))) or 0.0
    if total <= 0:
        return None
    rel = float(20.0 * np.log10(max(rms["drums"], 1e-9) / total))
    return rel if rel <= -float(DRUM_STEM_BLEED_BELOW_TOTAL_DB) else None


def build_omnizart_summaries(stems, whole_mix_tags=None, bpm=None, sections=None,
                             key=None, notes_out=None, mix_audio_path=None,
                             activity_out=None, convergence_out=None,
                             instrumental=False, listening_text=None):
    """`mix_audio_path` (optional): path to the original un-separated mix.
    Threaded through to _filter_instrument_tags so guitar-family stem tags can
    be checked against the raw whole-mix probabilities rather than only
    against the top-k truncated whole-mix tag list.

    `convergence_out` (optional dict): populated in place with the cross-stem
    instrument agreement map (label -> [stem, ...]) so callers can use "several
    stems independently found this instrument" as evidence, not just print it.

    `activity_out` (optional list): populated in place with the per-stem
    activity metadata (stem, rms, note_count, density, silent). The Demucs
    guitar stem's energy is independent evidence about whether a guitar is
    present, and the caller needs it to decide whether a missing guitar tag
    means absence or just a detector miss.

    `key` (optional): the track's RECOMMENDED KEY FOR DISCUSSION string,
    e.g. "C major" -- threaded through to melodic-line summaries so they can
    report real scale-degree emphasis instead of just bare note names.

    `notes_out` (optional dict): populated in place with the filtered note
    events per stem. The raw transcription used to be discarded the moment
    this function returned its prose report, so nothing downstream -- the
    saved .json, the melodic analysis block, a reloaded track -- could ever
    see an actual note again. Passing a dict here keeps the measurements
    available for the rest of the pipeline without changing the return type
    for existing callers."""
    if not stems:
        return "STEM MIDI REPORT unavailable: no Demucs stems found."

    lines = []
    activity_metas = []
    _drum_line_idx = None
    stem_notes = {}   # will store filtered notes per stem
    # label -> [(stem, prob, coverage), ...], for cross-stem agreement below.
    per_stem_tags = {}
    # stem -> (hits moved onto real transients, total hits), drums only.
    _drum_onsets_refined = {}

    for stem in STEM_MIDI_STEMS:
        path = stems.get(stem)
        if not path or not os.path.exists(path):
            continue

        preset = STEM_MIDI_PRESETS[stem]
        # Always measure stem energy for prominence ranking (cheap vs transcription).
        # This now reads the WHOLE stem and, when structure sections are
        # available, also returns a per-section RMS breakdown -- see
        # _stem_rms_by_section. One file load covers both.
        rms, section_rms_list = _stem_rms_by_section(path, sections)
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
            try:
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
                        for _n in raw_notes or []:
                            _f = float(_n.get("frequency", 300.0))
                            _n["drum_type"] = (
                                "kick" if _f < 150 else ("snare" if _f < 800 else "hihat")
                            )
                            _n["detected_from"] = "onset-band fallback"
                    else:
                        if SHOW_OMNIZART_LOGS:
                            print(
                                f"  (Omnizart drum model returned "
                                f"{len(raw_notes)} classified hit(s))"
                            )
                        # Omnizart places every hit on its own 32nd-note beat
                        # grid; move each one onto the real transient.
                        raw_notes, _n_moved = _refine_drum_onsets_from_audio(raw_notes, path)
                        raw_notes = sorted(raw_notes, key=lambda n: float(n["onset"]))
                        _drum_onsets_refined[stem] = (_n_moved, len(raw_notes))

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
                        # Flag these explicitly. Their "frequency" is a band
                        # proxy (80 / 300 / 8000 Hz), and the old test for this
                        # path -- "no note inside the stem's pitch range" --
                        # never fired because 80 and 300 Hz ARE inside it, so
                        # onset hits were reported as real D#2 / D4 pitches.
                        for _n in raw_notes or []:
                            _n["detected_from"] = "onset-band fallback"

                else:
                    # vocals -> Omnizart vocal model
                    # bass/guitar/piano -> Omnizart music model
                    raw_notes = omnizart_transcribe(
                        path,
                        stem,
                        preset
                    )
            except Exception as e:
                print(f"  ⚠ {stem} stem MIDI transcription failed: {e}")
                raw_notes = []

            # Ensure raw_notes is a list (some stems may return None on failure)
            if raw_notes is None:
                raw_notes = []

            # Replace the transcriber's placeholder velocities with levels
            # measured from this stem (the onset-band fallback already measures
            # its own amplitudes from the audio envelope).
            if raw_notes and not any(
                n.get("detected_from") == "onset-band fallback" for n in raw_notes
            ):
                try:
                    _attach_measured_amplitudes(raw_notes, path, stem)
                except Exception as _amp_err:
                    print(f"  (measured note levels skipped for {stem}: {_amp_err})")

            _is_onset_fallback = bool(raw_notes) and all(
                n.get("detected_from") == "onset-band fallback" for n in raw_notes
            )

            # --- FILTERING LOGIC ---
            if stem == "drums":
                filtered = [n for n in raw_notes if n.get("onset", 0) >= 0]
                removed = len(raw_notes) - len(filtered)
                stem_notes[stem] = filtered
            elif stem == "other" and _is_onset_fallback:
                # Onset-fallback path: skip pitch-range filter, keep only valid onsets.
                filtered = [n for n in raw_notes if n.get("onset", 0) > 0 and n.get("duration", 0) > 0]
                removed = len(raw_notes) - len(filtered)
                stem_notes[stem] = filtered
            else:
                filtered, removed = _filter_transcribed_notes(raw_notes, preset)
                stem_notes[stem] = filtered

            corrected = merged = 0
            if stem == "vocals" and not instrumental and STEM_MIDI_STABILIZE_VOCAL_PITCH:
                try:
                    filtered, corrected, merged = _stabilize_vocal_midi(filtered, path, preset)
                    stem_notes[stem] = filtered
                except Exception as exc:
                    if DEBUG_STEM_MIDI:
                        print(f"  (vocal MIDI pitch stabilization skipped: {exc})")

            summary, meta = summarize_stem_midi(
                stem, raw_notes, filtered, removed, preset, stem_rms=rms,
                bpm=bpm, sections=sections, section_rms=section_rms_list, key=key,
                stem_path=path, joined_fragments=merged,
            )
            if corrected:
                summary += (
                    f"\nvocal MIDI pitch stabilization: {corrected} adjacent-semitone "
                    f"corrections supported by the vocal contour, {merged} short fragments joined."
                )
            if meta.get("augmented_notes") is not None:
                stem_notes[stem] = meta.pop("augmented_notes")

            if stem == "vocals" and instrumental:
                # Measured instrumental: this stem is residue. Keep the report
                # line for transparency, but no vocal onset claim, and keep its
                # "notes" out of every melody / arrangement / structure block.
                summary = "\n".join(
                    l for l in summary.splitlines() if not l.startswith("EARLIEST VOCAL ONSET:")
                )
                summary += (
                    "\nNOTE: the recording is instrumental (see VOCAL PRESENCE ARBITRATION). "
                    "Everything this stem holds is separation residue from other instruments, "
                    "not a voice; its notes are excluded from the melody, arrangement and "
                    "structure analysis."
                )
                stem_notes.pop("vocals", None)
                meta["silent"] = True

            if stem in _drum_onsets_refined:
                _mv, _tot = _drum_onsets_refined[stem]
                summary += (
                    f"\nhit timing: the drum transcriber places every hit on its own "
                    f"32nd-note beat grid; {_mv} of {_tot} hits were moved onto the actual "
                    "transients in the drums audio, so the grid, swing and timing reads above "
                    "describe the playing rather than that grid."
                )
            _model_used = _OMNIZART_MODEL_USED.get(stem)
            if stem in ("bass", "guitar", "other") and raw_notes and not _is_onset_fallback \
                    and str(_model_used or "").startswith("default (Piano)"):
                summary += (
                    "\ntranscription model: Omnizart's default piano-trained model (no "
                    "instrument-specific model is configured for this stem), so note-level "
                    "detail for non-piano instruments is approximate -- prefer contour, range "
                    "and density over exact pitches here."
                )

            accepted_labels = None
            if ENABLE_INSTRUMENT_TAGGING and stem in INSTRUMENT_TAG_STEMS:
                status(f"Tagging instruments in {stem} stem...")
                raw_tags = tag_stem_instruments(path)
                note_count = int(meta.get("note_count") or 0)
                tags = _filter_instrument_tags(
                    raw_tags,
                    note_count=note_count,
                    whole_mix_tags=whole_mix_tags,
                    audio_path=path,
                    mix_audio_path=mix_audio_path,
                )
                accepted_labels = [t[0] for t in (tags or [])]
                # The per-section presence block reads the raw window
                # probabilities; it must only report labels that survived
                # these filters for this stem.
                _ACCEPTED_STEM_TAGS[os.path.abspath(str(path))] = set(accepted_labels)
                # ...and must keep the "weak" caveat those labels carry here.
                _WEAK_STEM_TAGS[os.path.abspath(str(path))] = {
                    t[0] for t in (tags or []) if "weak" in str(t[3] or "").lower()
                }
                if tags:
                    tag_parts = []
                    for label, prob, t, conf_note in tags:
                        base = f"{label} ({prob * 100:.0f}%, strongest near {t:.0f}s)"
                        if conf_note:
                            base += f" [{conf_note}]"
                        tag_parts.append(base)
                        # Record for the cross-stem agreement pass.
                        try:
                            _st = _panns_tag_stats(path, label) or {}
                            per_stem_tags.setdefault(label, []).append(
                                (stem, float(prob), float(_st.get("coverage") or 0.0))
                            )
                        except Exception:
                            pass
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

            _identity = _stem_identity_note(stem, accepted_labels, meta)
            if _identity:
                summary += "\n" + _identity

            activity_metas.append(meta)
        except Exception as e:
            summary = f"[{stem.upper()} stem] MIDI transcription failed: {e}"
            print(f"  \u26a0 {stem} stem MIDI transcription failed: {e}")
            activity_metas.append({
                "stem": stem,
                "rms": rms,
                "note_count": 0,
                "density": 0.0,
                "is_drums": stem == "drums",
                "silent": True,
            })
            
            stem_notes[stem] = []

        if stem == "drums":
            _drum_line_idx = len(lines)
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

    _low_note = _bass_claim_note(listening_text, activity_metas)
    if _low_note:
        lines.append(_low_note)

    # DRUM_BLEED_NOTE: a near-silent drums stem holds bleed (strummed guitar,
    # hand percussion of other parts), not a kit -- its "hits", grids and
    # timeline are replaced by a note saying so.
    _bleed_db = _drum_stem_bleed_db(activity_metas)
    if _bleed_db is not None and _drum_line_idx is not None:
        lines[_drum_line_idx] = _drum_bleed_note(listening_text, _bleed_db)

    # Cross-stem agreement: the same instrument found independently on more
    # than one isolated stem. Built after every stem has been tagged.
    _convergence = {}
    try:
        _conv_block, _convergence = _build_cross_stem_convergence(
            per_stem_tags, whole_mix_tags=whole_mix_tags,
            mix_audio_path=mix_audio_path,
        )
        if _conv_block:
            lines.append(_conv_block)
    except Exception:
        _convergence = {}
    if convergence_out is not None and isinstance(convergence_out, dict):
        try:
            convergence_out.update(_convergence)
        except Exception:
            pass

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
        "for the whole track. Per-stem energy (RMS) is measured over the WHOLE track, and "
        "when structure sections are available, each stem also gets an INTENSITY BY "
        "SECTION block showing how that instrument's presence rises and falls across the "
        "song's own verse/chorus/etc. sections — use that (not just the single whole-track "
        "prominence figure) whenever the user asks how an instrument changes across the "
        "song, builds into the chorus, drops out, etc. "
        "Melodic-line summaries also report real step/leap interval movement and, when the "
        "track's key is known, which scale degrees (relative to that key) the line "
        "emphasizes — both computed directly from the transcribed notes, not invented; use "
        "them for genuine music-theory framing (e.g. 'mostly stepwise, centers on the tonic "
        "and dominant') instead of guessing at harmony from note names alone. "
        "Stem names are separation buckets, NOT instrument identifications: the separator "
        "has no strings, synth, brass or woodwind stem, so those parts land in 'other' -- or "
        "in 'bass' when they sit low. Where a stem carries a STEM IDENTITY NOTE, that note "
        "says what the part actually is. Note levels ('amp') are measured from each stem's "
        "audio relative to that stem's loudest moments. "
        + event_note
        + "Use it for casual melody/harmony/composition/rhythm discussion; be cautious "
        "with erratic short notes and low-confidence stems.\n"
    )

    # Build the melody‑per‑section block from the filtered notes and the
    # structural sections, then append it to the report.
    if sections and stem_notes:
        melody_block = _build_melody_per_section(sections, stem_notes, bpm=bpm, key=key)
        if melody_block:
            lines.append("\n" + melody_block)

    # Independent section-change candidates from MIDI note-density jumps.
    # These catch arrangement shifts that keep the same full-mix timbre
    # (e.g. denser chorus line) and feed the structure snap pipeline via
    # _collect_structure_boundaries().
    try:
        _dur_for_midi = None
        if sections:
            try:
                _dur_for_midi = max(float(s.get("end") or 0) for s in sections)
            except Exception:
                _dur_for_midi = None
        _midi_bounds = _midi_activity_boundary_candidates(
            stem_notes, duration=_dur_for_midi
        )
        if _midi_bounds:
            lines.append(
                "MIDI ACTIVITY BOUNDARY CANDIDATES (objective, per-stem note-density "
                "change points — independent of full-mix timbre novelty):\n"
                "candidate change points: "
                + ", ".join(_seconds_to_mmss(t) for t in _midi_bounds)
            )
    except Exception:
        pass

    if notes_out is not None and isinstance(notes_out, dict):
        notes_out.update(stem_notes)

    # Per-stem energy/density, exported for callers that need to cross-check a
    # tagger result against what the separated stems actually contain.
    if activity_out is not None and isinstance(activity_out, list):
        try:
            activity_out.extend(activity_metas)
        except Exception:
            pass

    # Return the full report, now including melody per section.
    return header + "\n\n".join(lines)
