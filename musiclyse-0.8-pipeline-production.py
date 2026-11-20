# MUSICLYSE FILE pipeline-production 0.8
# ==============================================================================
#  MUSICLYSE 0.8: PRODUCTION
# ==============================================================================
# This file holds loudness, dynamics, stereo field, clipping and the other production
# measurements, and the Essentia measurements. It is part of musiclyse-0.8.py, which loads it
# whole at start-up, after its own code, into its own namespace (see "THE MUSICLYSE FILES"
# there); it is not run on its own.
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

def _dynamics_label_from_crest(crest_db):
    """Map crest-factor dB to a plain-language band for the writer.

    Crest factor is PEAK-TO-AVERAGE: how far the transients stand above the
    average level, i.e. how much peak limiting there is. It is not loudness
    range. The old labels called a high crest a "very wide dynamic range ...
    closer to an unmastered or vinyl-oriented presentation", which sat right
    next to a tight LRA in every report for a punchy, steady-level mix (spiky
    drums over a constant level give a high crest AND a low LRA) and read as a
    contradiction. Labels now describe peak limiting / transient headroom only;
    section-to-section dynamics come from LRA.

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
        return "moderate crest / some peak control", (
            "moderate peak-to-average ratio — some peak limiting but not brickwalled; "
            "do not treat this alone as proof of era"
        )
    if c < 14.0:
        return "moderately high crest / light peak limiting", (
            "noticeable transient headroom — less aggressive peak limiting than a typical "
            "hot commercial master; do not treat this alone as proof of era"
        )
    if c < 18.0:
        return "high crest / transients well above the average level", (
            "peaks sit well above the average level — light peak limiting. This is "
            "transient headroom, not section-to-section dynamics (use LRA for that)"
        )
    return "very high crest / little peak limiting", (
        "transients stand far above the average level — little or no peak limiting. "
        "This is transient headroom, not loudness range: a very high crest alongside a "
        "tight LRA is normal (punchy hits over a steady level) and does not make the "
        "track 'very dynamic'"
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
        interpret.append(
            "Crest factor is peak-to-average (how much the transients are limited), not "
            "dynamic range; section-to-section dynamics come from LRA, so a high crest with "
            "a tight LRA means punchy peaks over a steady level, not a contradiction."
        )

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

    # A loud, tightly-compressed measurement of THIS AUDIO FILE is not the
    # same claim as "this recording is recent". Observed case: a genuinely
    # 1990 track measured at -11.5 LUFS / 2.0 LU LRA -- a loudness/dynamics
    # profile essentially never produced by an untouched pre-2000s master,
    # and much more consistent with a modern remaster or reissue -- and the
    # dedicated era-estimation pass concluded "2010s-2020s" from exactly this
    # kind of loud/polished production character, despite its own prompt
    # explicitly warning against treating "polished", "compressed" and "wide
    # stereo" as decisive era evidence. The measurement genuinely reflects
    # THIS FILE's mastering; it says nothing about when the song was
    # originally recorded if the file has since been remastered, which is
    # common for reissued or streaming-service catalogue material. Attach the
    # caveat here, next to the numbers themselves, so it travels with the
    # measurement into every context that reads this block -- including era
    # questions -- rather than depending on the era pass remembering to say
    # it (which it was explicitly asked to, and still did not).
    if lufs is not None and lra is not None and lufs >= -13.0 and lra <= 3.5:
        body += (
            " ERA CAVEAT: this loud (" + f"{lufs:.1f} LUFS" + "), tightly-compressed "
            "(" + f"{lra:.1f} LU LRA" + ") profile is completely ORDINARY for anything "
            "mastered from roughly the early 2000s onward and is not unusual or "
            "noteworthy on its own — loudness has not meaningfully changed since then, "
            "so this reading alone cannot tell a 2002 recording from a brand-new one. "
            "Do NOT say or imply the track has been remastered; you have a loudness "
            "number, not evidence of a specific mastering event. This is only worth "
            "mentioning at all if OTHER evidence (genre, playing style, period-typical "
            "instrumentation) already points to an era before roughly the mid-to-late "
            "1990s, in which case treat it as an open question, not a settled claim. "
            "Otherwise say nothing about mastering age and reason from genre/style/"
            "performance cues instead."
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


_EBUR128_CACHE = {}


def _ffmpeg_ebur128_loudness__pre6(local_path):
    """Run ffmpeg ebur128 on the FULL file (stereo preserved). Returns
    (lufs, lra, true_peak_db, source) or (None, None, None, None).

    This is the preferred path: native sample rate, all channels, no
    premature mono downmix, no peak renormalization, ignores ReplayGain
    tags (ffmpeg does not apply them unless volume/replaygain filters are set).

    Results are cached per (path, size, mtime): the measurement is a full-file
    decode, and the true-peak figure it already produces is now also read by
    the production analysis, which would otherwise pay for a second identical
    ffmpeg run.
    """
    if not local_path or not os.path.exists(local_path):
        return None, None, None, None

    cache_key = None
    try:
        _st = os.stat(local_path)
        cache_key = (os.path.abspath(local_path), int(_st.st_size), int(_st.st_mtime_ns))
        if cache_key in _EBUR128_CACHE:
            return _EBUR128_CACHE[cache_key]
    except Exception:
        cache_key = None

    try:
        # framelog=verbose is not required for summary lines; keep stderr modest.
        cmd = [
            "ffmpeg",
            "-nostdin",          # dev2: never read the terminal (see convert_to_wav)
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
            stdin=subprocess.DEVNULL,
        )
        # A failed decode can still print a summary of its partial audio.
        # Do not label that as a full-file measurement or cache a failed run.
        if result.returncode != 0:
            return None, None, None, None
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
            out = (
                round(float(lufs), 1),
                (round(float(lra), 1) if lra is not None and np.isfinite(lra) else None),
                (round(float(tp), 1) if tp is not None and np.isfinite(tp) else None),
                "ffmpeg ebur128 (EBU R128, full file, native channels)",
            )
            if cache_key is not None:
                try:
                    if len(_EBUR128_CACHE) > 8:
                        _EBUR128_CACHE.clear()
                    _EBUR128_CACHE[cache_key] = out
                except Exception:
                    pass
            return out
    except Exception:
        pass
    # Transient failures must remain retryable in this process.
    return None, None, None, None


def _ffmpeg_ebur128_loudness__07(local_path):
    """Run ffmpeg ebur128 on the FULL file (stereo preserved). Returns
    (lufs, lra, true_peak_db, source) or (None, None, None, None).

    This is the preferred path: native sample rate, all channels, no
    premature mono downmix, no peak renormalization, ignores ReplayGain
    tags (ffmpeg does not apply them unless volume/replaygain filters are set).

    Results are cached per (path, size, mtime): the measurement is a full-file
    decode, and the true-peak figure it already produces is now also read by
    the production analysis, which would otherwise pay for a second identical
    ffmpeg run.
    """
    if not local_path or not os.path.exists(local_path):
        return None, None, None, None

    cache_key = None
    try:
        _st = os.stat(local_path)
        cache_key = (os.path.abspath(local_path), int(_st.st_size), int(_st.st_mtime))
        if cache_key in _EBUR128_CACHE:
            return _EBUR128_CACHE[cache_key]
    except Exception:
        cache_key = None

    try:
        # framelog=verbose is not required for summary lines; keep stderr modest.
        cmd = [
            "ffmpeg",
            "-nostdin",          # dev2: never read the terminal (see convert_to_wav)
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
            stdin=subprocess.DEVNULL,
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
            out = (
                round(float(lufs), 1),
                (round(float(lra), 1) if lra is not None and np.isfinite(lra) else None),
                (round(float(tp), 1) if tp is not None and np.isfinite(tp) else None),
                "ffmpeg ebur128 (EBU R128, full file, native channels)",
            )
            if cache_key is not None:
                try:
                    if len(_EBUR128_CACHE) > 8:
                        _EBUR128_CACHE.clear()
                    _EBUR128_CACHE[cache_key] = out
                except Exception:
                    pass
            return out
    except Exception:
        pass
    if cache_key is not None:
        try:
            if len(_EBUR128_CACHE) > 8:
                _EBUR128_CACHE.clear()
            _EBUR128_CACHE[cache_key] = (None, None, None, None)
        except Exception:
            pass
    return None, None, None, None


def _ffmpeg_ebur128_loudness(*args, **kwargs):
    """AUDIO_CACHE_FIXES picks the 0.7.1-pre6 version; False gives 0.7's."""
    return (_ffmpeg_ebur128_loudness__pre6 if AUDIO_CACHE_FIXES else _ffmpeg_ebur128_loudness__07)(*args, **kwargs)


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
_PROD_TAGS_NONE = ("production-character tags (audio classifier, supporting evidence only): "
                   "none above the detection threshold.")


def _fixed_blocks_production_tags(t):
    i = t.find("\n\nPRODUCTION / MIX MEASUREMENTS")
    if i < 0 or "\nPRODUCTION-CHARACTER TAGS\n" in t:
        return t
    # walk the production report's own sub-sections; the tags go after the last one
    end = t.find("\n\n", i + 2)
    while end >= 0:
        nxt = t[end + 2:t.find("\n", end + 2) if t.find("\n", end + 2) >= 0 else len(t)]
        if nxt.strip() in _PROD_SECTION_TITLES:
            e2 = t.find("\n\n", end + 2)
            if e2 < 0:
                end = len(t)
                break
            end = e2
            continue
        break
    if end < 0:
        end = len(t)
    return t[:end] + "\n\nPRODUCTION-CHARACTER TAGS\n" + _PROD_TAGS_NONE + t[end:]
# ==============================================================================
# PRODUCTION / MIX MEASUREMENTS
# ==============================================================================
# TIMBRE asks the writer for "stereo image, ambience, reverb, saturation/
# distortion, transient character", the genre prompt asks it to weigh
# "sidechain pump" and "club loudness vs garage band dynamics", and the era
# prompt asks about "stereo image and panning conventions" and "room/plate/
# digital reverb characteristics" -- but nothing in the pipeline measured any
# of that. Everything upstream loads the file with mono=True, so stereo was
# structurally invisible, and the only production numbers available were
# loudness/crest plus first-60s spectral stats. That left those fields to be
# filled from the writer's impression, or from genre expectation.
#
# These measurements are deliberately conservative: each is reported as a
# number with a plain-language reading, and each runs in its own try/except so
# a failure degrades that one line rather than the report.

# (PRODUCTION_ANALYSIS_ENABLED moved to the settings file beside this script.)
# Excerpts are spread evenly across the whole file rather than analysing one
# contiguous chunk, so an atypical intro cannot define the track's profile.
# (PRODUCTION_EXCERPT_COUNT moved to the settings file beside this script.)
# (PRODUCTION_EXCERPT_SECONDS moved to the settings file beside this script.)
# Sample rate for the continuous envelope analyses (pumping, ambience decay).
# The features live well below 5 kHz, so decoding at full rate would cost
# memory for no accuracy.
# (PRODUCTION_ENVELOPE_SR moved to the settings file beside this script.)


def _prod_excerpt_offsets(duration, count=None, seconds_each=None):
    """Evenly spaced excerpt start times covering the whole track, skipping
    the very start and very end (fades and silence would skew every
    measurement that follows)."""
    count = int(count or PRODUCTION_EXCERPT_COUNT)
    seconds_each = float(seconds_each or PRODUCTION_EXCERPT_SECONDS)
    if not duration or duration <= 0:
        return []
    if duration <= seconds_each * 1.5:
        return [0.0]
    edge = min(2.0, duration * 0.02)
    first = edge
    last = max(first, duration - seconds_each - edge)
    if last <= first:
        return [first]
    count = max(1, min(count, int((last - first) / max(1.0, seconds_each * 0.5)) + 1))
    if count == 1:
        return [(first + last) / 2.0]
    return [first + (last - first) * i / float(count - 1) for i in range(count)]


def _prod_load_excerpts(local_path, duration, mono=False):
    """Load evenly spaced excerpts at NATIVE sample rate (needed for the
    high-frequency cutoff measurement) preserving channels (needed for the
    stereo field measurement). Returns (list_of_arrays, sr) where each array
    is (channels, n) for stereo or (n,) for mono."""
    offsets = _prod_excerpt_offsets(duration)
    chunks = []
    sr = None
    for off in offsets:
        try:
            y, this_sr = librosa.load(
                local_path, sr=None, mono=mono,
                offset=float(off), duration=float(PRODUCTION_EXCERPT_SECONDS),
            )
        except Exception:
            continue
        if y is None or np.size(y) == 0:
            continue
        sr = this_sr
        chunks.append(np.asarray(y, dtype=np.float32))
    return chunks, sr


def _prod_split_lr(chunk):
    """Return (left, right) or (mono, None) from a librosa-shaped array."""
    arr = np.asarray(chunk)
    if arr.ndim == 1:
        return arr, None
    if arr.shape[0] == 1:
        return arr[0], None
    if arr.shape[0] >= 2:
        return arr[0], arr[1]
    return arr.reshape(-1), None


def _prod_stereo_field(chunks, sr):
    """Stereo image measurements: correlation, mid/side balance overall and by
    band, and mono-fold-down loss.

    Nothing else in the pipeline sees stereo at all, so width, panning
    convention and mono compatibility previously had no evidence behind them.
    Per-band width matters especially at the low end: a centred bass is a
    near-universal modern mastering convention, so an uncorrelated low end is
    a genuine signal about era or how finished the master is.
    """
    lines = []
    lefts, rights = [], []
    for c in chunks:
        L, R = _prod_split_lr(c)
        if R is None:
            continue
        n = min(len(L), len(R))
        if n > 0:
            lefts.append(L[:n])
            rights.append(R[:n])

    if not lefts:
        lines.append("channel format: mono (single channel) — no stereo image to assess")
        return lines

    L = np.concatenate(lefts).astype(np.float64)
    R = np.concatenate(rights).astype(np.float64)
    if len(L) < 1000:
        return lines

    mid = (L + R) * 0.5
    side = (L - R) * 0.5
    mid_e = float(np.mean(mid ** 2))
    side_e = float(np.mean(side ** 2))

    if mid_e <= 1e-12 and side_e <= 1e-12:
        lines.append("stereo field: unavailable (silent excerpts)")
        return lines

    # A file can be stereo-format but dual-mono in content.
    if side_e <= mid_e * 1e-7:
        lines.append(
            "channel format: stereo file carrying identical channels (dual mono) "
            "— no true stereo image"
        )
        return lines

    denom = float(np.sqrt(np.mean(L ** 2) * np.mean(R ** 2)))
    corr = float(np.mean(L * R) / denom) if denom > 1e-12 else 0.0
    corr = max(-1.0, min(1.0, corr))
    width_db = 10.0 * np.log10(max(side_e, 1e-20) / max(mid_e, 1e-20))

    lines.append(f"stereo correlation (L/R): {round(corr, 3)} (1.0=mono, 0=uncorrelated, <0=out of phase)")
    lines.append(f"side/mid energy ratio: {round(float(width_db), 1)} dB")

    if corr >= 0.95:
        desc = "very narrow / essentially centred"
    elif corr >= 0.75:
        desc = "narrow-to-moderate width"
    elif corr >= 0.4:
        desc = "conventional stereo width"
    elif corr >= 0.05:
        desc = "wide stereo image"
    else:
        desc = "very wide / heavily decorrelated"
    lines.append(f"stereo width reading: {desc}")

    if corr < 0.0:
        lines.append(
            "phase note: net negative L/R correlation — significant out-of-phase content "
            "(wide stereo processing, or a phase problem); would partially cancel in mono."
        )

    # Mono fold-down loss: how much energy disappears when summed to mono.
    stereo_e = float(np.mean((L ** 2 + R ** 2) * 0.5))
    mono_e = float(np.mean(mid ** 2))
    if stereo_e > 1e-12:
        fold = 10.0 * np.log10(max(mono_e, 1e-20) / stereo_e)
        lines.append(f"mono fold-down level change: {round(float(fold), 1)} dB")
        if fold < -3.0:
            lines.append(
                "mono compatibility: substantial cancellation when summed to mono "
                "(heavy widening or phase issues)."
            )

    # Per-band width. Bass centring is the informative one.
    try:
        n_fft = 2048
        Sm = np.abs(librosa.stft(mid.astype(np.float32), n_fft=n_fft, hop_length=1024))
        Ss = np.abs(librosa.stft(side.astype(np.float32), n_fft=n_fft, hop_length=1024))
        freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
        bands = [("low (20-120 Hz)", 20, 120), ("low-mid (120-800 Hz)", 120, 800),
                 ("mid (800-4k Hz)", 800, 4000), ("high (4k-16k Hz)", 4000, 16000)]
        parts = []
        low_ratio = None
        for name, lo, hi in bands:
            mask = (freqs >= lo) & (freqs < hi)
            if not np.any(mask):
                continue
            m_e = float(np.sum(Sm[mask] ** 2))
            s_e = float(np.sum(Ss[mask] ** 2))
            if m_e <= 1e-12:
                continue
            ratio_db = 10.0 * np.log10(max(s_e, 1e-20) / m_e)
            parts.append(f"{name}: {round(float(ratio_db), 1)} dB")
            if lo == 20:
                low_ratio = ratio_db
        if parts:
            lines.append("stereo width by band (side/mid, higher = wider): " + "; ".join(parts))
        if low_ratio is not None:
            if low_ratio < -18.0:
                lines.append(
                    "low-end imaging: bass is strongly centred — the standard modern "
                    "mastering convention (elliptical/mono-below filtering)."
                )
            elif low_ratio > -8.0:
                lines.append(
                    "low-end imaging: unusually wide/uncorrelated low end — more typical of "
                    "older, unmastered, or deliberately unconventional productions."
                )
    except Exception:
        pass

    return lines


def _full_file_clip_runs(path, threshold=0.9995, min_run=4, block_frames=262144):
    """Count flat-topped clipping over the WHOLE file, streaming the decode.

    Returns (clipped_runs, clipped_samples, total_samples) or None when the file
    cannot be decoded this way. Runs are counted per channel and may span the
    decoder's block boundaries. Uses ffmpeg so every container the rest of the
    pipeline accepts works, without holding the whole file in memory.
    """
    if not path or str(path).startswith(("http://", "https://")) or not os.path.exists(str(path)):
        return None
    try:
        pr = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a:0",
             "-show_entries", "stream=channels", "-of", "default=nw=1:nk=1", str(path)],
            capture_output=True, text=True, timeout=60,
        )
        ch = int((pr.stdout or "").strip().splitlines()[0])
        if ch < 1:
            return None
    except Exception:
        return None
    try:
        proc = subprocess.Popen(
            ["ffmpeg", "-v", "error", "-nostdin", "-i", str(path), "-map", "0:a:0",
             "-f", "f32le", "-acodec", "pcm_f32le", "-"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        )
    except Exception:
        return None
    runs = 0
    clipped = 0
    total = 0
    carry = [0] * ch            # length of a hot run still open at the block edge
    leftover = b""
    frame_bytes = 4 * ch
    try:
        while True:
            data = proc.stdout.read(block_frames * frame_bytes)
            if not data:
                break
            data = leftover + data
            usable = (len(data) // frame_bytes) * frame_bytes
            leftover = data[usable:]
            if usable == 0:
                continue
            arr = np.frombuffer(data[:usable], dtype=np.float32).reshape(-1, ch)
            total += arr.size
            hot_all = np.abs(arr) >= float(threshold)
            for c in range(ch):
                hot = hot_all[:, c]
                if not np.any(hot):
                    if carry[c] >= min_run:
                        runs += 1
                        clipped += carry[c]
                    carry[c] = 0
                    continue
                edges = np.diff(np.concatenate(([0], hot.view(np.int8), [0])))
                starts = np.flatnonzero(edges == 1)
                ends = np.flatnonzero(edges == -1)
                lengths = (ends - starts).tolist()
                if starts[0] == 0:
                    lengths[0] += carry[c]          # continues a run from the last block
                elif carry[c] >= min_run:
                    runs += 1
                    clipped += carry[c]
                carry[c] = 0
                if ends[-1] == hot.size:            # still open at this block's end
                    carry[c] = lengths.pop()
                for L in lengths:
                    if L >= min_run:
                        runs += 1
                        clipped += L
        for c in range(ch):
            if carry[c] >= min_run:
                runs += 1
                clipped += carry[c]
        proc.wait(timeout=60)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass
        return None
    if total == 0:
        return None
    return runs, clipped, total


def _prod_peaks_and_clipping(local_path, chunks):
    """True peak, peak-to-loudness ratio, and digital clipping estimate.

    ffmpeg already computes a true-peak figure during the loudness pass but it
    was parsed and then discarded; it is the single most direct indicator of
    how hard a master was pushed. Clipped-sample runs are counted separately,
    since intersample overs and actual flat-topped clipping are different
    production facts.
    """
    lines = []
    lufs = tp = None
    try:
        lufs, _lra, tp, _src = _ffmpeg_ebur128_loudness(local_path)
    except Exception:
        pass

    if tp is not None:
        lines.append(f"true peak: {round(float(tp), 1)} dBTP")
        if tp > 0.0:
            lines.append(
                "true-peak note: exceeds 0 dBTP — intersample overs, typical of a loud "
                "master pushed to (or past) the ceiling; can distort on lossy playback."
            )
        elif tp > -0.3:
            lines.append("true-peak note: mastered right up against the ceiling.")

    if tp is not None and lufs is not None:
        plr = float(tp) - float(lufs)
        lines.append(f"peak-to-loudness ratio (PLR): {round(plr, 1)} dB")
        if plr < 8.0:
            lines.append(
                "PLR reading: very low — heavily limited/compressed master "
                "(loudness-maximised)."
            )
        elif plr < 12.0:
            lines.append("PLR reading: moderate — conventional contemporary master.")
        else:
            lines.append("PLR reading: high — dynamic, lightly limited master.")

    # Flat-topped clipping: consecutive samples pinned near full scale. Scanned
    # over the whole file -- the excerpts below cover only about a third of a
    # typical track, so "none" from them meant little (on one test master the
    # excerpts found 2 of 5 clipped runs).
    full = None
    try:
        full = _full_file_clip_runs(local_path)
    except Exception:
        full = None
    if full is not None:
        clipped_runs, clipped_samples, total = full
        if clipped_runs > 0:
            rate = clipped_samples / float(total) * 1e6
            lines.append(
                f"clipped-sample runs detected (whole file): {clipped_runs} "
                f"({round(rate, 1)} clipped samples per million)"
            )
            lines.append(
                "clipping note: flat-topped samples at full scale — the master is "
                "clipped, either deliberately for loudness or from a gain-staging error."
            )
        else:
            lines.append("clipped-sample runs detected (whole file): none")
        return lines

    try:
        total = 0
        clipped_runs = 0
        clipped_samples = 0
        for c in chunks:
            L, R = _prod_split_lr(c)
            for ch in (L, R):
                if ch is None or len(ch) == 0:
                    continue
                a = np.abs(np.asarray(ch, dtype=np.float32))
                total += a.size
                hot = a >= 0.9995
                if not np.any(hot):
                    continue
                # Count runs of >= 4 consecutive at-ceiling samples.
                idx = np.flatnonzero(
                    np.diff(np.concatenate(([0], hot.view(np.int8), [0]))) != 0
                )
                for s, e in zip(idx[0::2], idx[1::2]):
                    if e - s >= 4:
                        clipped_runs += 1
                        clipped_samples += int(e - s)
        if total > 0 and clipped_runs > 0:
            rate = clipped_samples / float(total) * 1e6
            lines.append(
                f"clipped-sample runs detected (sampled excerpts only): {clipped_runs} "
                f"({round(rate, 1)} clipped samples per million)"
            )
            lines.append(
                "clipping note: flat-topped samples at full scale — the master is "
                "clipped, either deliberately for loudness or from a gain-staging error."
            )
        elif total > 0:
            lines.append(
                "clipped-sample runs detected: none in the sampled excerpts (the whole "
                "file could not be scanned, so clipping elsewhere is not ruled out)"
            )
    except Exception:
        pass

    return lines


_LOSSY_AUDIO_EXTS = {".mp3", ".aac", ".ogg", ".oga", ".opus", ".wma", ".mp2", ".mpc", ".webm"}
_LOSSLESS_AUDIO_EXTS = {".wav", ".wave", ".flac", ".aif", ".aiff", ".aifc", ".w64", ".ape", ".wv"}


def _prod_hf_cutoff(chunks, sr, source_path=None):
    """Highest frequency carrying real energy, plus whether it ends in a sharp
    shelf. A brickwall well below Nyquist is the fingerprint of lossy encoding
    (or an old/band-limited source), which is real provenance information: it
    tells the writer this file may not be the full-quality master, and it is a
    genuine era/format clue that nothing else in the pipeline provides.

    Detection scans for the steepest sustained drop rather than thresholding
    against a fixed level, because a fixed threshold is defeated by spectral
    tilt: a bass-heavy mix has so little midrange reference energy that a
    dither/noise floor at 20 kHz can sit within any reasonable margin of it.
    A brickwall is a *shape* — a cliff — so the cliff is what is measured.
    """
    lines = []
    if not chunks or not sr:
        return lines
    nyq = sr / 2.0
    try:
        n_fft = 4096
        spec_sum = None
        frames = 0
        for c in chunks:
            L, R = _prod_split_lr(c)
            mono = L if R is None else (L[: min(len(L), len(R))] + R[: min(len(L), len(R))]) * 0.5
            if len(mono) < n_fft:
                continue
            S = np.abs(librosa.stft(np.asarray(mono, dtype=np.float32),
                                    n_fft=n_fft, hop_length=n_fft // 2))
            if S.size == 0:
                continue
            spec_sum = np.sum(S ** 2, axis=1) if spec_sum is None else spec_sum + np.sum(S ** 2, axis=1)
            frames += S.shape[1]
        if spec_sum is None or frames == 0:
            return lines

        freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
        db = 10.0 * np.log10(np.maximum(spec_sum / float(frames), 1e-20))

        # Smooth over ~250 Hz so individual partials don't read as cliffs.
        bin_hz = float(freqs[1] - freqs[0]) if len(freqs) > 1 else 1.0
        w = max(1, int(250.0 / max(bin_hz, 1e-6)))
        if w % 2 == 0:
            w += 1
        pad = w // 2
        sm = np.convolve(np.pad(db, (pad, pad), mode="edge"),
                         np.ones(w) / w, mode="valid")[: len(db)]

        peak_db = float(np.max(sm))

        # Scan for the steepest 1.2 kHz-either-side drop above 5 kHz.
        best_f, best_drop = None, 0.0
        lo_i = int(np.searchsorted(freqs, 5000.0))
        hi_i = int(np.searchsorted(freqs, nyq * 0.985))
        step = max(1, (hi_i - lo_i) // 240)
        for i in range(lo_i, hi_i, step):
            f = float(freqs[i])
            below = (freqs >= f - 1200) & (freqs < f)
            above = (freqs > f) & (freqs <= f + 1200)
            if not np.any(below) or not np.any(above):
                continue
            drop = float(np.median(sm[below]) - np.median(sm[above]))
            if drop > best_drop:
                best_drop, best_f = drop, f

        sharp = best_f is not None and best_drop >= 18.0

        if sharp:
            # Corner frequency: last point still within 10 dB of the shelf's
            # pre-drop level, which lands on the knee rather than mid-slope.
            pre = (freqs >= best_f - 1200) & (freqs < best_f)
            pre_level = float(np.median(sm[pre])) if np.any(pre) else peak_db
            within = np.flatnonzero((sm >= pre_level - 10.0) & (freqs <= best_f + 1200))
            cutoff = float(freqs[within[-1]]) if len(within) else best_f
        else:
            floor = peak_db - 65.0
            above_floor = np.flatnonzero(sm >= floor)
            cutoff = float(freqs[above_floor[-1]]) if len(above_floor) else nyq

        lines.append(
            f"upper frequency limit: ~{int(round(cutoff / 100.0) * 100)} Hz "
            f"(file Nyquist {int(nyq)} Hz)"
        )
        if sharp:
            lines.append(
                f"cutoff character: sharp shelf (~{int(round(best_drop))} dB drop) — a "
                "brickwall lowpass rather than a natural rolloff"
            )

        # The container tells whether the file ITSELF is a lossy encode. A
        # lossless file (WAV/FLAC/AIFF) with a sharp shelf got its band limit
        # upstream -- an earlier lossy encode, a band-limited generator or
        # synth (AI music generators commonly produce exactly this), or
        # resampling -- and an AI-generated test track, a WAV straight from its generator, was
        # told it had "very probably been through lossy compression". m4a is
        # left as unknown: it can hold AAC or lossless ALAC.
        _ext = os.path.splitext(str(source_path or ""))[1].lower()
        _container = ("lossy" if _ext in _LOSSY_AUDIO_EXTS
                      else "lossless" if _ext in _LOSSLESS_AUDIO_EXTS else "")
        _fmt = _ext[1:].upper() if _ext else ""

        # Only interpret as a codec fingerprint when the ceiling is genuinely
        # below what the container could have carried.
        if cutoff < nyq * 0.93:
            if cutoff >= 19000:
                # A sharp shelf at 19-20 kHz is just as typical of a resampler's
                # anti-alias filter, a DAW/mastering low-pass or a soft-synth's
                # band limit as of a 256-320 kbps encode, so it proves nothing.
                note = ("high-bitrate lossy encoding (~256 kbps+) OR a sample-rate-conversion / "
                        "mastering / instrument low-pass -- a ceiling this high is NOT proof of "
                        "lossy compression")
                if _container == "lossy" and sharp:
                    # ...unless the file IS a lossy encode, when its own
                    # encoder's low-pass is the obvious explanation.
                    note = (f"high-bitrate lossy encoding (~256 kbps+) -- this file is itself a "
                            f"lossy {_fmt}, so the shelf is most likely its own encoder's low-pass")
                source_like = ""
            elif cutoff >= 16500:
                note = "consistent with mid-bitrate lossy encoding (roughly 160-192 kbps)"
                source_like = "a mid-bitrate (roughly 160-192 kbps) lossy encode"
            elif cutoff >= 14500:
                note = "consistent with ~128 kbps lossy encoding"
                source_like = "a ~128 kbps lossy encode"
            elif cutoff >= 10000:
                note = ("strongly band-limited — low-bitrate encoding, an analogue/broadcast "
                        "source, or deliberate lo-fi")
                source_like = ("low-bitrate encoding, an analogue/broadcast source or "
                               "deliberate lo-fi")
            else:
                note = ("very band-limited — a low-quality, very old, or deliberately "
                        "degraded source")
                source_like = "a low-quality, very old or deliberately degraded source"
            if _container == "lossless" and sharp and cutoff < 19000:
                note = (f"band-limited source -- the shelf sits where {source_like} "
                        f"would leave it, but this file is lossless ({_fmt}), so the limit was "
                        "introduced before the file was written")
            lines.append(f"bandwidth reading: {note}")
            if sharp and _container == "lossy":
                lines.append(
                    f"provenance caution: this file is itself a lossy {_fmt} encode, so the shelf is "
                    + ("most likely its encoder's low-pass at a low bitrate (or the source was "
                       "already band-limited)" if cutoff < 14500 else "its encoder's low-pass")
                    + " and the very top end is not the original master. Do not "
                    "read the missing top octave as a mixing or era choice."
                )
            elif sharp and _container == "lossless" and cutoff < 19000:
                lines.append(
                    f"provenance caution: the file itself is lossless ({_fmt}) but its spectrum "
                    "stops sharply, so the band limit came from upstream -- an earlier lossy "
                    "encode, a band-limited synthesis or source, or resampling from a lower "
                    "rate. Do not read the missing top "
                    "octave as a mixing or era choice, and do not call the file itself lossy or "
                    "transcoded."
                )
            elif sharp and cutoff < 19000:
                lines.append(
                    "provenance caution: this file has very probably been through lossy "
                    "compression at some point, so the very top end is not the original master. "
                    "Do not read the missing top octave as a mixing or era choice by itself."
                )
            elif sharp:
                lines.append(
                    "provenance note: the top end above the shelf is inaudible to most listeners; "
                    "do not describe the recording as lossy, degraded or transcoded on this "
                    "evidence alone."
                )
        else:
            lines.append(
                "bandwidth reading: full-band up to the file's own limit — no evidence of "
                "aggressive lossy band-limiting"
            )
    except Exception:
        pass
    return lines


def _prod_pumping(y, sr, beat_times=None):
    """Beat-synchronous amplitude modulation: objective evidence for
    sidechain-style pumping.

    The genre prompt explicitly asks the writer to weigh "sidechain pump" when
    choosing between electronic/dance and band-based genres, but nothing
    measured it, so that call rested on impression alone.

    Crucially this is measured in a MID BAND (roughly 300 Hz - 5 kHz), not on
    the full-mix envelope. A loud kick raises full-mix level at every beat, so
    full-mix ducking depth is nearly identical for a sidechained track and for
    one with a merely hard-hitting backbeat (measured: 11.6 dB vs 11.8 dB —
    indistinguishable). A kick contributes little above 300 Hz, so it cannot
    make a pad's midrange duck; only a compressor can. In the mid band the same
    two cases separate cleanly (11.1 dB vs 0.7 dB).

    The reliable, high-value output is the NEGATIVE: when the sustained bed
    plainly does not duck, the writer can say so with confidence. A positive
    reading is reported as consistent-with rather than proof, because a long
    reverb tail re-excited every beat produces a similar mid-band envelope.
    """
    lines = []
    try:
        hop = 512
        frame_rate = sr / float(hop)

        # Beat period, preferring measured beats over a raw tempo estimate.
        beat_period = None
        if beat_times is not None and len(beat_times) > 2:
            d = np.diff(np.asarray(beat_times, dtype=float))
            d = d[(d > 0.15) & (d < 2.5)]
            if len(d):
                beat_period = float(np.median(d))
        if not beat_period:
            return lines

        S = np.abs(librosa.stft(np.asarray(y, dtype=np.float32), n_fft=2048, hop_length=hop))
        if S.size == 0 or S.shape[1] < 32:
            return lines
        freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
        mask = (freqs >= 300.0) & (freqs < min(5000.0, sr / 2.0 - 1.0))
        if not np.any(mask):
            return lines
        band = np.sqrt(np.mean(S[mask] ** 2, axis=0))
        if band.size < 32:
            return lines

        # Floor relative to the band's own level, so near-silent passages
        # cannot produce absurd dB swings.
        med = float(np.median(band))
        floor = max(med * 1e-3, 1e-10)
        band_db = 20.0 * np.log10(np.maximum(band, floor))
        times = librosa.frames_to_time(np.arange(len(band_db)), sr=sr, hop_length=hop)

        # Guard: is there a sustained bed to duck at all? On a percussion-only
        # track the mid band is pure transients, so "ducking" between hits is
        # just silence and means nothing about compression.
        spread = float(np.percentile(band_db, 90) - np.percentile(band_db, 50))
        if spread > 20.0:
            lines.append(
                "sidechain assessment: not assessable — the 300 Hz-5 kHz range is almost "
                "entirely transient, with no sustained bed that could be ducked."
            )
            return lines

        # Detrend so slow arrangement dynamics don't read as beat-rate pumping.
        win = max(3, int(frame_rate * 4.0))
        if win % 2 == 0:
            win += 1
        pad = win // 2
        trend = np.convolve(np.pad(band_db, (pad, pad), mode="edge"),
                            np.ones(win) / win, mode="valid")[: len(band_db)]
        detr = band_db - trend
        detr = detr - float(np.mean(detr))
        if float(np.std(detr)) < 1e-6:
            return lines

        ac_full = np.correlate(detr, detr, mode="full")[len(detr) - 1:]
        if ac_full[0] <= 0:
            return lines
        ac = ac_full / ac_full[0]
        lag = int(round(beat_period * frame_rate))
        if lag < 2 or lag >= len(ac):
            return lines
        span = max(1, int(lag * 0.12))
        beat_ac = float(np.max(ac[max(1, lag - span):min(len(ac), lag + span + 1)]))

        depths = []
        bt = np.asarray(beat_times, dtype=float)
        for i in range(len(bt) - 1):
            s, e = bt[i], bt[i + 1]
            if e - s <= 0.05:
                continue
            seg = band_db[(times >= s) & (times < e)]
            if seg.size >= 4:
                depths.append(float(np.max(seg) - np.min(seg)))
        if not depths:
            return lines
        med_depth = float(np.median(depths))

        lines.append(
            f"beat-locked modulation of the sustained bed (300 Hz-5 kHz): "
            f"{round(beat_ac, 3)} (0=none, 1=perfectly periodic)"
        )
        lines.append(f"median per-beat ducking depth in that band: {round(med_depth, 1)} dB")

        if med_depth < 3.0 or beat_ac < 0.15:
            lines.append(
                "sidechain reading: the sustained bed does NOT duck with the beat — this track "
                "is not sidechained/pumping. Do not describe it as pumping, and do not use "
                "pumping as an argument for an electronic/dance genre label."
            )
        elif med_depth >= 6.0 and beat_ac >= 0.30:
            # Beat-locked level movement has two opposite causes. Sidechain
            # ducking pulls the bed DOWN just after each beat and lets it swell
            # back; parts that play ON the beat (strummed or stabbed chords,
            # picked riffs) push the level UP at each beat and let it decay.
            # The depth/periodicity numbers above cannot tell them apart -- a
            # strummed-guitar rock track read as "sidechain, a dance/electronic
            # signature". Where in the beat the harmonic level peaks can.
            shape, fold_db = None, None
            try:
                shape, fold_db = _beat_phase_shape(y, sr, beat_times)
            except Exception:
                shape, fold_db = None, None
            if shape == "weak":
                lines.append(
                    f"sidechain reading: the per-beat swings average out to only {fold_db:.1f} dB "
                    "when folded onto the beat -- they are not a consistent duck-and-recover, so "
                    "this is NOT sidechain pumping (playing dynamics / on-beat parts). Do not use it "
                    "as an argument for a dance/electronic label."
                )
            elif shape == "attack":
                lines.append(
                    "sidechain reading: the beat-locked movement comes from parts that play ON "
                    "the beat (strummed or stabbed chords, picked figures) -- the level jumps at "
                    "each beat and decays, rather than ducking and recovering. This is NOT "
                    "sidechain pumping; do not use it as an argument for a dance/electronic label."
                )
            elif shape == "duck":
                lines.append(
                    "sidechain reading: the sustained bed ducks strongly in lockstep with the beat "
                    "and recovers between beats — consistent with sidechain compression, a "
                    "dance/electronic production signature. Caveat: a long reverb tail "
                    "re-triggered each beat can produce a similar envelope, so check this against "
                    "the ambience measurement below."
                )
            else:
                lines.append(
                    "sidechain reading: strong beat-locked level movement, but its shape is "
                    "ambiguous (neither a clear duck-and-recover nor an on-beat attack) -- could "
                    "be sidechain, rhythmic gating or simply the arrangement. Do not assert pumping."
                )
        else:
            lines.append(
                "sidechain reading: mild beat-locked movement in the sustained bed — could be "
                "light sidechain, rhythmic gating, or simply arrangement. Not strong enough to "
                "assert pumping."
            )
    except Exception:
        pass
    return lines


def _prod_ambience(y, sr):
    """Reverb/ambience proxy from post-transient energy decay.

    TIMBRE and the era prompt both ask about reverb and room character with no
    measurement behind them. Decay rate after transients is a legitimate proxy:
    a dry, close-miked or heavily gated production decays fast, a plate/hall or
    washy production decays slowly. Reported as a proxy, not an RT60 claim.

    Two readings are made. On the full mix, anything that keeps sounding after
    a hit -- held chords, pads, strings -- reads as "decay", so a dense or
    sustained arrangement came out "very washy/reverb-heavy" whatever its
    reverb (and often pinned at the 6 s ceiling). The same measurement on the
    percussive component (harmonic/percussive separation) follows the tails of
    the hits themselves, which is much closer to the room/reverb on them; the
    ambience reading now comes from that one, and the full-mix figure is
    reported as sustain.
    """
    lines = []
    ceiling = 6.0

    def _decay(sig, s_r):
        hop = 256
        rms = librosa.feature.rms(y=sig, frame_length=1024, hop_length=hop)[0]
        if rms.size < 64:
            return None, None, 0
        db = 20.0 * np.log10(np.maximum(rms, 1e-8))
        frame_rate = s_r / float(hop)
        onset_frames = librosa.onset.onset_detect(y=sig, sr=s_r, hop_length=hop, backtrack=False)
        if len(onset_frames) < 4:
            return None, None, 0
        slopes, late_ratios = [], []
        min_gap = int(frame_rate * 0.28)
        skip = int(frame_rate * 0.04)
        for i, f in enumerate(onset_frames):
            nxt = onset_frames[i + 1] if i + 1 < len(onset_frames) else len(db)
            if nxt - f < min_gap:
                continue  # next hit masks the tail
            a = int(f + skip)
            b = int(min(f + frame_rate * 0.30, nxt - 1, len(db) - 1))
            if b - a < 6:
                continue
            seg = db[a:b]
            t = np.arange(len(seg)) / frame_rate
            slope = float(np.polyfit(t, seg, 1)[0])  # dB per second
            if slope < -1.0:
                slopes.append(slope)
            peak = float(np.max(db[int(f):int(f + max(2, skip)) + 1]))
            late_i = int(f + frame_rate * 0.18)
            if late_i < len(db):
                late_ratios.append(float(db[late_i] - peak))
        if not slopes:
            return None, None, 0
        med_slope = float(np.median(slopes))
        late = float(np.median(late_ratios)) if late_ratios else None
        return med_slope, late, len(slopes)

    def _fmt_decay(med_slope):
        d = 60.0 / abs(med_slope) if med_slope < -1e-6 else None
        if d is None:
            return None, "n/a"
        if d >= ceiling:
            return ceiling, f">= {ceiling:.0f} s (at the measurement ceiling)"
        d = max(0.05, d)
        return d, f"{round(d, 2)} s"

    def _reading(d):
        if d < 0.35:
            return "dry/tight — close, damped, or gated production"
        if d < 0.8:
            return "moderate room/plate ambience"
        if d < 1.6:
            return "noticeably reverberant — large room/hall/plate feel"
        return "very washy/reverb-heavy, long tails"

    try:
        mix_slope, mix_late, mix_n = _decay(y, sr)
        if mix_slope is None:
            return lines
        mix_d, mix_txt = _fmt_decay(mix_slope)

        perc_slope = perc_late = None
        perc_n = 0
        try:
            # Separation at 11.025 kHz: tails of hits are broadband, so this
            # loses nothing the decay estimate uses and halves the cost.
            sr_p = 11025 if sr > 11025 else sr
            y_p = np.asarray(y, dtype=np.float32)
            if sr_p != sr:
                y_p = librosa.resample(y_p, orig_sr=sr, target_sr=sr_p)
            S = librosa.stft(y_p, n_fft=1024, hop_length=256)
            _H, P = librosa.decompose.hpss(S, margin=2.0)
            y_perc = librosa.istft(P, hop_length=256, length=len(y_p))
            perc_slope, perc_late, perc_n = _decay(y_perc, sr_p)
        except Exception:
            perc_slope = None

        lines.append(f"post-transient decay rate, full mix: {round(mix_slope, 1)} dB/s "
                     f"(median over {mix_n} isolated hits)")
        lines.append(f"full-mix sustain after hits (extrapolated to -60 dB): {mix_txt}")
        if mix_late is not None:
            lines.append(
                f"energy remaining 180 ms after a hit: {round(mix_late, 1)} dB relative to the hit"
            )
        if perc_slope is not None:
            perc_d, perc_txt = _fmt_decay(perc_slope)
            lines.append(
                f"decay of the hits themselves (percussive component, extrapolated to -60 dB): "
                f"{perc_txt} (median over {perc_n} isolated hits)"
            )
            if perc_d is not None:
                lines.append(f"ambience reading: {_reading(perc_d)}")
                if mix_d is not None and mix_d >= 1.6 and perc_d < 0.8:
                    lines.append(
                        "sustain note: the long full-mix sustain comes from instruments that keep "
                        "sounding (held chords, pads, strings), not from reverb on the hits — do "
                        "not describe the production as reverb-heavy on the strength of it."
                    )
        elif mix_d is not None:
            lines.append(
                f"ambience reading (full mix only, so sustained instruments count as decay): "
                f"{_reading(mix_d)}"
            )
        lines.append(
            "NOTE: these are proxies, not an RT60 — dense arrangements mask true tails, and the "
            "full-mix figure includes anything that sustains."
        )
    except Exception:
        pass
    return lines


def _prod_fullband_balance(chunks, sr):
    """Band energy share averaged over excerpts spanning the WHOLE track.

    The existing TIMBRE block measures only the first 60 seconds. Intros are
    routinely unrepresentative — no drums, no bass, filtered pads, a solo
    voice — so first-60s band shares can describe a section that never
    returns. This does not replace that block; it sits alongside it so the
    two can be compared, and reports how much the balance actually moves
    across the track.
    """
    lines = []
    if not chunks or not sr:
        return lines
    try:
        n_fft = 2048
        bands = [("sub-bass", 20, 60), ("bass", 60, 250), ("low-mid", 250, 800),
                 ("mid", 800, 4000), ("high", 4000, 16000)]
        per_excerpt = []
        centroids = []
        for c in chunks:
            L, R = _prod_split_lr(c)
            mono = L if R is None else (L[: min(len(L), len(R))] + R[: min(len(L), len(R))]) * 0.5
            mono = np.asarray(mono, dtype=np.float32)
            if len(mono) < n_fft:
                continue
            S = np.abs(librosa.stft(mono, n_fft=n_fft, hop_length=n_fft // 2))
            freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
            tot = float(np.sum(S ** 2))
            if tot <= 0:
                continue
            shares = []
            for _name, lo, hi in bands:
                mask = (freqs >= lo) & (freqs < hi)
                shares.append(float(np.sum(S[mask] ** 2)) / tot * 100.0 if np.any(mask) else 0.0)
            per_excerpt.append(shares)
            try:
                centroids.append(float(np.mean(librosa.feature.spectral_centroid(S=S, sr=sr))))
            except Exception:
                pass

        if not per_excerpt:
            return lines
        arr = np.asarray(per_excerpt, dtype=float)
        means = arr.mean(axis=0)
        lines.append(
            f"band energy share, whole-track average over {len(per_excerpt)} excerpts "
            "(compare with the first-60s figures above):"
        )
        for (name, lo, hi), m in zip(bands, means):
            # Label the band by what this sample rate can actually hold: on the
            # 22.05 kHz working copy "high (4000-16000 Hz)" really ends at ~11 kHz.
            lines.append(f"  {name} ({lo}-{min(int(hi), int(sr // 2))} Hz): {round(float(m), 1)}%")
        if centroids:
            lines.append(
                f"whole-track brightness (mean spectral centroid): "
                f"{round(float(np.mean(centroids)), 1)} Hz"
            )
        if arr.shape[0] >= 3:
            spread = float(np.mean(arr.std(axis=0)))
            lines.append(f"section-to-section spectral variation: {round(spread, 1)} percentage points")
            if spread < 1.5:
                lines.append(
                    "arrangement reading: spectral balance barely changes across the track — "
                    "a consistent, static arrangement/mix."
                )
            elif spread > 5.0:
                lines.append(
                    "arrangement reading: spectral balance shifts a lot across the track — "
                    "sections differ substantially in instrumentation or density."
                )
    except Exception:
        pass
    return lines


def _prod_effect_tags(mix_audio_path):
    """Production-character tags (audible distortion, reverb, echo, noise
    floor) from the same already-cached PANNs windows the instrument tagger
    uses, so this costs no extra inference."""
    lines = []
    if not ENABLE_INSTRUMENT_TAGGING:
        return lines
    if not mix_audio_path or str(mix_audio_path).startswith(("http://", "https://")):
        return lines
    try:
        tags = _panns_tag_windowed(mix_audio_path, _PRODUCTION_LABEL_MAP, 5, 0.15)
        if not tags:
            return lines
        parts = []
        for label, prob, t in tags:
            part = f"{label} ({prob * 100:.0f}%"
            cov = _tag_coverage_note(_panns_tag_stats(mix_audio_path, label))
            if cov:
                part += f", {cov}"
            parts.append(part + ")")
        lines.append(
            "production-character tags (audio classifier, supporting evidence only): "
            + ", ".join(parts)
        )
    except Exception:
        pass
    return lines


def build_production_analysis_report(local_path, y=None, sr=None, duration=None,
                                     beat_times=None, original_path=None):
    """Objective mix/production measurements: stereo image, true peak and
    clipping, bandwidth/provenance, sidechain-style pumping, ambience decay,
    and whole-track spectral balance.

    `y`/`sr` are the mono waveform the caller has already decoded (reused for
    the envelope analyses so the file is not decoded twice); stereo excerpts
    are loaded separately because every existing load path is mono-only.
    Returns "" when nothing could be measured. Each measurement is independent
    and failure-isolated.
    """
    if not PRODUCTION_ANALYSIS_ENABLED:
        return ""
    if not local_path or str(local_path).startswith(("http://", "https://")):
        return ""

    if duration is None and y is not None and sr:
        try:
            duration = len(y) / float(sr)
        except Exception:
            duration = None
    if not duration or duration < 3.0:
        return ""

    chunks, ex_sr = [], None
    try:
        chunks, ex_sr = _prod_load_excerpts(local_path, duration, mono=False)
    except Exception:
        chunks, ex_sr = [], None

    # `local_path` is the pipeline's 22.05 kHz working WAV, and convert_to_wav()
    # builds that with ffmpeg "-ac 1" — so it is ALWAYS single-channel and the
    # excerpts above can never carry a second channel. _prod_stereo_field() then
    # saw R is None for every chunk and reported "mono (single channel)" for
    # every track, regardless of the real source. Load a separate set of
    # excerpts from the untouched original file for that one measurement.
    #
    # Deliberately scoped to the stereo lines only: every other measurement
    # below still reads `chunks`/`ex_sr` exactly as before, so their numbers do
    # not move. Falls back to the original chunks whenever the source file is
    # missing, remote, or unreadable, which reproduces the previous behaviour.
    stereo_chunks, stereo_sr = [], None
    _stereo_src_path = None   # the file the stereo excerpts came from, if it is the original
    try:
        if (
            original_path
            and isinstance(original_path, str)
            and not str(original_path).startswith(("http://", "https://"))
            and os.path.exists(original_path)
        ):
            stereo_chunks, stereo_sr = _prod_load_excerpts(
                original_path, duration, mono=False
            )
            if stereo_chunks and stereo_sr:
                _stereo_src_path = original_path
    except Exception:
        stereo_chunks, stereo_sr = [], None
    if not stereo_chunks or not stereo_sr:
        # The pipeline's own working WAV: its container says nothing about the
        # source, so the bandwidth reading must not treat it as "lossless".
        stereo_chunks, stereo_sr = chunks, ex_sr
        _stereo_src_path = None

    sections = []

    stereo_lines = []
    try:
        if stereo_chunks and stereo_sr:
            stereo_lines = _prod_stereo_field(stereo_chunks, stereo_sr)
    except Exception:
        pass
    if stereo_lines:
        sections.append(("STEREO IMAGE / MIX WIDTH (objective)", stereo_lines))

    peak_lines = []
    try:
        # True peak / PLR / clipping describe the MASTER, so they are read from
        # the untouched source (stereo, native rate) when available. On the
        # resampled mono working copy, resampling moves peaks and softens the
        # flat tops that clipping detection looks for.
        _peak_src = local_path
        if (
            original_path
            and isinstance(original_path, str)
            and not str(original_path).startswith(("http://", "https://"))
            and os.path.exists(original_path)
            and stereo_chunks
        ):
            _peak_src = original_path
        peak_lines = _prod_peaks_and_clipping(
            _peak_src, stereo_chunks if _peak_src != local_path else chunks
        )
    except Exception:
        pass
    if peak_lines:
        sections.append(("MASTERING: TRUE PEAK / CLIPPING (objective)", peak_lines))

    bw_lines = []
    try:
        # Must use the NATIVE-rate excerpts. `chunks` comes from the pipeline's
        # 22.05 kHz working WAV, so its Nyquist is ~11 kHz — this measurement
        # could never observe a cutoff above that, and reported "full-band up to
        # the file's own limit" for everything, including 128 kbps MP3s with a
        # 16 kHz brickwall. stereo_chunks is loaded from the untouched source at
        # its own sample rate, which is exactly what this needs. _prod_hf_cutoff
        # folds L/R to mono itself, so stereo input is fine.
        #
        # Scoped to this measurement and the stereo one. _prod_peaks_and_clipping
        # and _prod_fullband_balance still read `chunks`/`ex_sr` unchanged, so
        # their numbers do not move.
        if stereo_chunks and stereo_sr:
            bw_lines = _prod_hf_cutoff(stereo_chunks, stereo_sr, source_path=_stereo_src_path)
    except Exception:
        pass
    if bw_lines:
        sections.append(("BANDWIDTH / SOURCE PROVENANCE (objective)", bw_lines))

    env_y, env_sr = None, None
    try:
        if y is not None and sr:
            env_sr = int(sr)
            env_y = np.asarray(y, dtype=np.float32)
            # Envelope features live below ~5 kHz; downsample to keep this cheap.
            if env_sr > PRODUCTION_ENVELOPE_SR * 1.5:
                env_y = librosa.resample(env_y, orig_sr=env_sr, target_sr=PRODUCTION_ENVELOPE_SR)
                env_sr = PRODUCTION_ENVELOPE_SR
    except Exception:
        env_y, env_sr = None, None

    if env_y is not None and env_sr:
        pump_lines = []
        try:
            pump_lines = _prod_pumping(env_y, env_sr, beat_times=beat_times)
        except Exception:
            pass
        if pump_lines:
            sections.append(("BEAT-SYNCHRONOUS DYNAMICS / PUMPING (objective)", pump_lines))

        amb_lines = []
        try:
            amb_lines = _prod_ambience(env_y, env_sr)
        except Exception:
            pass
        if amb_lines:
            sections.append(("AMBIENCE / REVERB CHARACTER (objective proxy)", amb_lines))

    bal_lines = []
    try:
        if chunks and ex_sr:
            bal_lines = _prod_fullband_balance(chunks, ex_sr)
    except Exception:
        pass
    if bal_lines:
        sections.append(("WHOLE-TRACK SPECTRAL BALANCE (objective)", bal_lines))

    fx_lines = []
    try:
        fx_lines = _prod_effect_tags(local_path)
    except Exception:
        pass
    if fx_lines:
        sections.append(("PRODUCTION-CHARACTER TAGS", fx_lines))

    # Free the excerpt buffers promptly; the batch path is RSS-sensitive.
    # stereo_chunks is native-rate/stereo and so can be several times larger
    # than chunks — it must be released here too. (In the fallback case both
    # names point at the same list, which del handles fine.)
    try:
        del chunks
        del stereo_chunks
        del env_y
    except Exception:
        pass

    if not sections:
        return ""

    out = [
        "PRODUCTION / MIX MEASUREMENTS (objective, measured from THIS file)",
        "These are measurements, not impressions. When describing stereo width, reverb, "
        "loudness/limiting, sidechain pumping or bandwidth, use these figures rather than "
        "inferring from genre or era. They describe the mix and master only — they say "
        "nothing about which instruments are playing.",
    ]
    for title, body in sections:
        out.append("")
        out.append(title)
        out.extend(body)
    return "\n".join(out)


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
    except Exception:
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
            lambda: _frame_factory("SpectralCentroidTime"),
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

# Production / signal-character AudioSet classes. These are NOT instruments,
# so they are kept out of _INSTRUMENT_LABEL_MAP (where they would compete for
# the instrument top-k and get reported as if they were parts in the
# arrangement). They feed the production analysis block instead: audible
# distortion, obvious reverb or echo, and noise-floor character are the kind
# of mix evidence TIMBRE and the era estimate ask for but nothing measured.
# Labels absent from this checkpoint's AudioSet vocabulary are simply never
# matched, so listing an uncertain name here is harmless.
_PRODUCTION_LABEL_MAP = {
    "distortion": "audible distortion/saturation",
    "reverberation": "audible reverb",
    "echo": "audible echo/delay",
    "chorus effect": "chorus/modulation effect",
    "effects unit": "outboard/effects processing",
    "wah-wah": "wah-wah",
    "sound effect": "sound effects/production FX",
    "noise": "broadband noise",
    "white noise": "white noise",
    "pink noise": "pink noise",
    "static": "static/crackle",
    "hum": "hum",
    "mains hum": "mains hum (50/60 Hz)",
    "sine wave": "pure tone/sine",
    "vibration": "low-frequency rumble",
}
