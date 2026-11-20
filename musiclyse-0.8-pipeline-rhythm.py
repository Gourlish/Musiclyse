# MUSICLYSE FILE pipeline-rhythm 0.8
# ==============================================================================
#  MUSICLYSE 0.8: TEMPO, BEAT GRID, DRUMS AND GROOVE
# ==============================================================================
# This file holds tempo, the beat and downbeat grid, metre, drums, swing and groove. It is part
# of musiclyse-0.8.py, which loads it whole at start-up, after its own code, into its own
# namespace (see "THE MUSICLYSE FILES" there); it is not run on its own.
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
# --- Percussion beyond the three transcribed classes ------------------------
# Omnizart's drum model resolves kick, snare and hi-hat and nothing else, so a
# crash, a ride, a tom fill, a tambourine or a shaker is either folded into
# those three or missed outright. That is a hard ceiling on the transcription
# and it cannot be raised without a different model -- but it is NOT a ceiling
# on what the pipeline can know, because the AudioSet tagger has classes for
# exactly the pieces the transcription is blind to. Running it over the
# isolated, peak-normalised drums stem and reading the raw per-window
# probabilities (no top-k, same approach as the guitar probe) recovers the
# missing kit vocabulary, with real coverage figures so extent can be stated
# honestly rather than guessed.
# (DRUM_EXTRAS_PROBE_ENABLED moved to the settings file beside this script.)
_DRUM_EXTRA_PROBE_LABELS = (
    "cymbal", "tambourine", "timpani", "drum roll", "percussion (unspecified)",
    "rimshot", "cowbell", "wood block", "maraca/shaker", "conga", "bongo",
    "tabla", "djembe", "castanets", "triangle", "steelpan/steel drum",
    "vibraphone", "marimba/xylophone", "glockenspiel", "chime/wind chime",
    "gong", "drum machine",
)
# A window counts as containing the instrument at/above this.
# (DRUM_EXTRAS_FIRE_PROB moved to the settings file beside this script.)
# Report it when it either peaks this high...
# (DRUM_EXTRAS_REPORT_PEAK moved to the settings file beside this script.)
# ...or persists across at least this share of windows at the fire threshold.
# (DRUM_EXTRAS_REPORT_COVERAGE moved to the settings file beside this script.)

# --- Per-section instrument identity ----------------------------------------
# Instrument TAGS were whole-track only: a label either made the list for the
# whole song or it did not. Energy per section (INTENSITY BY SECTION) says how
# LOUD a stem is in each part, and melody per section says what it plays, but
# nothing said WHICH INSTRUMENT is audible WHERE -- so "when does the organ
# come in", "is there guitar in the bridge", "what's playing in the intro"
# had no evidence behind them and got answered from arrangement habit. The
# windowed probabilities already carry timestamps, so this is a regrouping of
# data the tagger has always produced rather than new inference.
# (INSTRUMENT_SECTION_PRESENCE_ENABLED moved to the settings file beside this script.)
# (INSTRUMENT_SECTION_MIN_PROB moved to the settings file beside this script.)
# (INSTRUMENT_SECTION_MAX_LABELS moved to the settings file beside this script.)

# --- Beat grid diagrams -----------------------------------------------------
# A drum pattern is far clearer as a grid than as prose, but an LLM asked to
# draw one from a list of onset timestamps will mangle both the rhythm and the
# column alignment. So the grid is rendered here, from the measured hits, and
# the writer is told to reproduce it verbatim rather than redraw it.
# (BEAT_GRID_ENABLED moved to the settings file beside this script.)
# (BEAT_GRID_BARS moved to the settings file beside this script.)
# (BEAT_GRID_SUBDIV moved to the settings file beside this script.)
# (BEAT_GRID_MIN_HITS moved to the settings file beside this script.)

# --- Cymbal detection from audio (the transcription cannot do it) -----------
# Omnizart resolves kick/snare/hi-hat and nothing else, so a crash or ride is
# either missed or folded into the hi-hat row. The percussion PROBE can say a
# cymbal exists somewhere in a window, but not WHEN, so it cannot fill a grid.
# Cymbals are, however, easy to separate from hats acoustically: both are
# bright, but a closed hat dies in tens of milliseconds while a crash or ride
# rings for hundreds. Measuring how long the high band sustains after each hit
# recovers cymbal ONSETS, which is what a grid row needs.
# Deliberately conservative -- a false cymbal row is worse than a missing one,
# since the grid is presented as measured fact.
# (CYMBAL_ONSET_DETECT_ENABLED moved to the settings file beside this script.)
# (CYMBAL_BAND_HZ moved to the settings file beside this script.)
# The high band must stay this many times above the track's own median
# between-hits floor to count as still ringing.
# (CYMBAL_FLOOR_RATIO moved to the settings file beside this script.)
# ...and that sustained level must be a real fraction of the hit's own peak,
# so a quiet hit sitting inside an earlier cymbal's wash is not counted again.
# (CYMBAL_MIN_SUSTAIN_RATIO moved to the settings file beside this script.)
# A struck cymbal must spike this many times above the level just before it.
# Separates a genuine hit from the tail of a previous crash.
# (CYMBAL_MIN_ATTACK_RATIO moved to the settings file beside this script.)
# Consecutive detections closer than this are one cymbal ringing, not several.
# (CYMBAL_MERGE_GAP_S moved to the settings file beside this script.)
# Give up unless the drums stem yields at least this many, to avoid drawing a
# row from two ambiguous detections.
# (CYMBAL_MIN_DETECTIONS moved to the settings file beside this script.)

# --- Arrangement change points for structure boundaries ---------------------
# Acoustic novelty is weakest exactly where pop songs change section: a chorus
# entering on the same pad, key and tempo as the verse produces almost no
# timbral novelty, so the boundary gets placed by whichever nearby spectral
# wobble happened to score highest. But the ARRANGEMENT changes audibly at
# that moment -- the snare arrives, the bass drops out, a pad enters -- and
# that is now measurable, because every stem is transcribed with timestamps.
#
# Measured against a real track, every one of its nine section boundaries had
# an instrument entering or leaving within four seconds. The catch is
# selectivity: isolated single-stem events (a pad swell, a transcription gap)
# outnumber real section changes, while genuine boundaries tend to move
# SEVERAL stems at once. So corroboration count is what separates signal from
# noise here, and single-stem events are deliberately ranked below bar lines.
# (ARRANGEMENT_CHANGE_ENABLED moved to the settings file beside this script.)
# Silence longer than this ends an instrument's playing span (also scaled by
# tempo at the call site, as in the drum timeline).
# (ARRANGEMENT_SPAN_GAP_S moved to the settings file beside this script.)
# A span shorter than this many bars is a burst or a transcription artifact,
# not an instrument entering the arrangement.
# (ARRANGEMENT_MIN_SPAN_BARS moved to the settings file beside this script.)
# Entries/exits within this many bars of each other are one arrangement change.
# (ARRANGEMENT_CLUSTER_BARS moved to the settings file beside this script.)
# At least this many distinct stems must move together for a change point to
# be trusted as strongly as an acoustic novelty peak.
# (ARRANGEMENT_STRONG_MIN_STEMS moved to the settings file beside this script.)

# --- Low-power whole-mix pass (disagreement needs power to mean anything) ---
# The mix-agreement gate assumes the whole-mix pass is a competent second
# opinion, so "the mix did not agree" is read as evidence against a stem tag.
# On some masters the pass simply fails: it returns one or two labels barely
# over threshold and identifies nothing else in the track. A pass like that has
# no discriminating power, so its disagreement carries no information -- yet it
# was still vetoing every stem tag, which is the same absence-of-evidence
# error the guitar note used to make, one level up. When the mix pass looks
# this weak the gate is suspended: stem tags are annotated as un-corroborated
# rather than downgraded, and the drop rules do not fire.
# (MIX_PASS_LOW_POWER_MAX_TAGS moved to the settings file beside this script.)
# (MIX_PASS_LOW_POWER_MAX_TOP_PROB moved to the settings file beside this script.)



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
    """Map a free-form genre string (listening-model top rank, PANNs label, etc.) to a
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


def reconcile_bpm(
    moss_bpm,
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
    3. Fall back to the listening model's TEMPO_BPM when it is the only strong signal.
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
    if moss_bpm is not None:
        try:
            candidates.append(("MOSS-Music", float(moss_bpm)))
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
        if GENRE_DEV55 and GENRE_DEV56:
            # dev56: say what this prior is -- read on one post-punk structure test scan's notes as
            # the track's genre ("Genre prior: electronicish (electronic)").
            genre_note = (f" Genre prior (the listening model's first impression, used only to choose "
                          f"between half- and double-time readings; not the track's genre): "
                          f"{genre_fam} ({genre_hint}).")

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
    moss = by_src.get("MOSS-Music")
    ess = by_src.get("essentia")
    obj = by_src.get("objective")

    def close(a, b):
        if a is None or b is None:
            return False
        ratio = max(a, b) / max(min(a, b), 1e-6)
        return ratio <= 1.04

    # When detectors "agree" (within tolerance, not necessarily identical),
    # average the agreeing readings rather than arbitrarily keeping one and
    # discarding the other — real-world detectors routinely land 1-3 BPM
    # apart even on a clean agreement, and picking a single source lets that
    # source's individual measurement noise pass straight through as the
    # final answer instead of being smoothed out.
    if (
        moss is not None and ess is not None and obj is not None
        and close(moss, ess) and close(moss, obj) and close(ess, obj)
    ):
        avg = (moss + ess + obj) / 3.0
        return _quantize_bpm(avg), f"MOSS-Music, Essentia, and objective detector all agree (~{avg:.1f}, averaged).{genre_note}"
    if moss is not None and ess is not None and close(moss, ess):
        avg = (moss + ess) / 2.0
        return _quantize_bpm(avg), f"Essentia and MOSS-Music agree (~{avg:.1f}, averaged).{genre_note}"
    if moss is not None and obj is not None and close(moss, obj):
        avg = (moss + obj) / 2.0
        return _quantize_bpm(avg), f"MOSS-Music and objective detector agree (~{avg:.1f}, averaged).{genre_note}"
    _dsp_pair_vs_moss_double = None
    if ess is not None and obj is not None and close(ess, obj):
        avg = (ess + obj) / 2.0
        # librosa and Essentia share the same onset-periodicity blind spot, so
        # their agreement is weak evidence about the half/double-time LEVEL.
        # When the listening model hears exactly the other octave AND the genre carries a strong
        # tempo convention, let the octave decision below weigh all three
        # instead of returning the detectors' shared reading unexamined.
        # (Drum & bass read at 87 BPM, house at 62 and hip-hop at 170 were all
        # reported that way even with the listening model and the genre prior pointing to the
        # musical pulse.) Genres without a strong convention keep the
        # detectors' answer exactly as before.
        _moss_ratio = (max(moss, avg) / max(min(moss, avg), 1e-6)) if moss else None
        _band = _BPM_GENRE_PREFS.get(genre_fam) if genre_fam in (
            "dnb", "hiphop", "electronicish", "ballad", "rockish", "popish") else None
        # Only when the listening model's reading is inside the genre's typical tempo range and
        # the detectors' shared reading is outside it. Anything less specific
        # (both outside the range, or both inside) keeps the detectors' answer.
        if (
            _band is not None
            and _moss_ratio is not None
            and 1.8 <= _moss_ratio <= 2.2
            and _band[0] <= moss <= _band[1]
            and not (_band[0] <= avg <= _band[1])
        ):
            _dsp_pair_vs_moss_double = avg
        else:
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
        # Rock and pop have a typical tempo band too (see _BPM_GENRE_PREFS), and
        # it was never consulted: the generic "hi >= 140 -> take the lower
        # pulse" rule below fired first, so a 144 BPM rock track with a 72 BPM
        # reading was always reported at 72 -- outside the rock band, and with
        # every downstream groove label (backbeat -> "snare on every beat",
        # 8th-note hats -> "16th-note hats") reinterpreted at half speed. When
        # exactly one of the two readings sits in the genre's band, use it.
        if genre_fam in ("rockish", "popish"):
            hi_in = pref_lo <= hi <= pref_hi
            lo_in = pref_lo <= lo <= pref_hi
            if hi_in and not lo_in:
                return _quantize_bpm(hi), (
                    f"{hi_src} ({hi}) is ~2x {lo_src} ({lo}); genre prior ({fam_note}) "
                    f"favours {hi}, which is inside that range."
                )
            if lo_in and not hi_in:
                return _quantize_bpm(lo), (
                    f"{hi_src} ({hi}) is ~2x {lo_src} ({lo}); genre prior ({fam_note}) "
                    f"favours {lo}, which is inside that range."
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

    if (TEMPO_MEASURED_OCTAVE_PAIR and ess is not None and obj is not None and _dsp_pair_vs_moss_double is None
            and ess > 0 and obj > 0
            and abs(max(ess, obj) / min(ess, obj) - 2.0) <= float(TEMPO_MEASURED_OCTAVE_PAIR_TOL)):
        # dev39: see TEMPO_MEASURED_OCTAVE_PAIR -- the two measurements are an
        # exact octave apart and the listening model's figure matches neither
        # (an agreement would have been taken above), so the measurements
        # decide between themselves.
        _q, _n = prefer_of_double(ess, obj, "Essentia", "objective")
        if moss is not None:
            _n = (f"{_n} MOSS-Music's figure ({moss}) is set aside: it matches neither measurement, "
                  f"which agree with each other an octave apart.")
        return _q, _n

    if _dsp_pair_vs_moss_double is not None:
        _fam_note = _BPM_GENRE_PREFS.get(genre_fam, ("", "", "", "", ""))[4]
        return _quantize_bpm(moss), (
            f"Essentia and objective detector agree (~{_dsp_pair_vs_moss_double:.1f}) but MOSS-Music "
            f"hears the other half/double-time reading (~{moss}), which is the one that fits "
            f"the genre's usual range ({_fam_note}); using MOSS-Music's pulse.{genre_note}"
        )

    if moss is not None and ess is not None and is_double(moss, ess):
        return prefer_of_double(moss, ess, "MOSS-Music", "Essentia")

    if moss is not None and obj is not None and is_double(moss, obj):
        return prefer_of_double(moss, obj, "MOSS-Music", "objective")

    if ess is not None and obj is not None and is_double(ess, obj):
        return prefer_of_double(ess, obj, "Essentia", "objective")

    if moss is not None:
        _near = [(abs(v - moss) / max(moss, 1e-6), s_, v) for s_, v in (("Essentia", ess), ("objective", obj))
                 if v is not None and abs(v - moss) / max(moss, 1e-6) <= 0.10]
        if _near:
            _near.sort()
            _, _src, _val = _near[0]
            return _quantize_bpm(_val), (
                f"MOSS-Music (~{moss}) and {_src} ({_val}) roughly agree; using the measured {_src} value."
                f"{genre_note}"
            )
    if moss is not None:
        # With multiple disagreeing sources, do not let a genre label silently
        # convert one detector's value into a different pulse. Genre is a tie-break
        # only for an otherwise ambiguous single-source result or an explicit 2x pair.
        preferred, _, note = _preferred_tempo(moss, genre_hint=None)
        return _quantize_bpm(preferred), f"Using MOSS-Music musical pulse ({moss}) without genre override because detectors disagree.{genre_note}".strip()

    src, val = candidates[0]
    preferred, _, note = _preferred_tempo(val, genre_hint=None)
    return _quantize_bpm(preferred), f"{src} fallback. {note}{genre_note}".strip()

# dev19 tempo checks (see TEMPO_LISTENING_CHECK / TEMPO_SNARE_TIEBREAK).
_TEMPO_ECHO_RE = re.compile(
    r"(?im)^\W*TEMPO[_ ]?BPM\**\s*[=:]\s*\[?\s*(?:~|\u2248)?\s*\d+(?:\.\d+)?\s*\(?\s*"
    r"best estimate of the musical pulse")


def _tempo_quantize(val):
    """reconcile_bpm's rounding: nearest integer, snapped to a common published
    tempo within 1.5 BPM."""
    v = float(val)
    base = int(round(v))
    common = (
        60, 64, 66, 70, 72, 74, 76, 80, 84, 88, 90, 92, 96, 98, 100,
        104, 105, 108, 110, 112, 115, 116, 118, 120, 122, 124, 125,
        126, 128, 130, 132, 135, 136, 138, 140, 144, 145, 148, 150,
        152, 155, 160, 165, 168, 170, 174, 175, 180, 190, 200,
    )
    best, best_dist = base, abs(v - base)
    for c in common:
        d = abs(v - c)
        if d <= 1.5 and d < best_dist:
            best, best_dist = c, d
    return int(best)


def _moss_tempo_reading_check(moss_bpm, source_text):
    """(moss_bpm, echoed): the listening model's TEMPO_BPM is not a reading
    when its field goes on with the prompt's own instruction ("120.0 (best
    estimate of the musical pulse in BPM. Consider ...") -- MOSS-Music wrote
    that on the boy-band test track (108 BPM), and its 120 decided the tempo."""
    if (TEMPO_LISTENING_CHECK and moss_bpm is not None
            and _TEMPO_ECHO_RE.search(str(source_text or ""))):
        return None, True
    return moss_bpm, False


def _moss_tempo_copies_detector(moss_bpm, *detector_values):
    """dev46 TEMPO_COPY_CHECK: the detector value the listening model's
    TEMPO_BPM repeats to the decimal (a non-integer reading it was shown), or
    None. Never raises."""
    try:
        if not TEMPO_COPY_CHECK or moss_bpm is None:
            return None
        m = float(moss_bpm)
        for v in detector_values:
            if v is None:
                continue
            v = float(v)
            if abs(m - v) <= 0.06 and abs(v - round(v)) >= 0.05:
                return v
    except Exception:
        pass
    return None


def _tempo_without_copy(final_bpm, bpm_note, moss, copied, ess, obj, ess_med, obj_med, genre_hint, echoed=False):
    """dev46 TEMPO_COPY_CHECK: (bpm, note) decided again without the listening
    model's copied TEMPO_BPM. Used only when that moves the tempo by more than
    10% (a half/double-time decision); otherwise the original stands. Never raises."""
    try:
        rb, rn = reconcile_bpm(None, ess, obj, essentia_median_bpm=None, objective_median_bpm=None,
                               genre_hint=genre_hint)
        if not rb:
            return final_bpm, bpm_note
        rb, rn = _tempo_listening_check(rb, rn, None, ess, obj, ess_med, obj_med, echoed)
        if rb and final_bpm and abs(float(rb) - float(final_bpm)) / float(final_bpm) > 0.10:
            return rb, (str(rn or "").rstrip().rstrip(".") + ". " + _TEMPO_COPY_NOTE.format(m=moss, v=copied))
    except Exception:
        pass
    return final_bpm, bpm_note


_TEMPO_COPY_NOTE = ("TEMPO CHECK: MOSS-Music's TEMPO_BPM ({m:g}) only repeated a detector reading it had been "
                    "shown ({v:g}), so it was not counted as a separate opinion.")


def _tempo_listening_check(final_bpm, bpm_note, moss, ess, obj, ess_med, obj_med, echoed=False):
    """(final_bpm, bpm_note) after the dev19 listening-model checks.

    * When reconcile_bpm fell back to the listening model's pulse because the
      detectors disagree, a detector whose raw estimate matches its own median
      beat spacing (within TEMPO_SELF_CONSISTENT_TOL) is used instead, unless
      the listening model's pulse is within 10% of that reading or of its
      half / double. MOSS-Music's 127 decided the teen-pop test track (146 BPM; Essentia 146,
      its beat spacing 147.7; the other detector 91.3 against 95.7).
    * A listening-model tempo that only repeated the prompt is noted."""
    try:
        if not TEMPO_LISTENING_CHECK or final_bpm is None:
            return final_bpm, bpm_note
        note = str(bpm_note or "")
        if echoed:
            note = (note.rstrip().rstrip(".") + ". TEMPO CHECK: MOSS-Music's TEMPO_BPM repeated the "
                    "prompt's instruction text instead of giving a reading, so it was not used.")
        if moss is not None and note.startswith("Using MOSS-Music musical pulse"):
            m = float(moss)
            for src, raw, med in (("Essentia", ess, ess_med), ("the objective detector", obj, obj_med)):
                if raw is None or med is None:
                    continue
                raw, med = float(raw), float(med)
                if raw <= 0 or abs(raw - med) / raw > float(TEMPO_SELF_CONSISTENT_TOL):
                    continue
                if any(abs(m - r) / r <= 0.10 for r in (raw, raw * 2.0, raw / 2.0)):
                    continue
                q = _tempo_quantize(raw)
                genre = ""
                gm = re.search(r"\s*(Genre prior[^\n]*)$", note)
                if gm:
                    genre = " " + gm.group(1).strip()
                return q, (
                    f"TEMPO CHECK: MOSS-Music's pulse ({m:g}) matches neither detector, and {src}'s reading "
                    f"({raw:g}) matches its own beat spacing ({med:.1f} BPM), so {q} is used. Treat it as "
                    f"a measured reading; the detectors did not all give the same figure.{genre}")
        return final_bpm, note
    except Exception:
        return final_bpm, bpm_note


_TEMPO_BLOCK_RE = re.compile(
    r"RECOMMENDED TEMPO FOR DISCUSSION: (?P<bpm>\d+) BPM\. Reasoning: (?P<why>[^\n]*?) This is the primary "
    r"tempo to report to the user\. State it as a concrete figure \(e\.g\. 'about \d+ BPM'\)\. Do not expand "
    r"it into a range unless this block itself marks the value as uncertain\.")
_TEMPO_DEFAULT_LOWER_RE = re.compile(
    r"^(?P<hs>.+?) \((?P<hi>[\d.]+)\) is ~2x (?P<ls>.+?) \((?P<lo>[\d.]+)\); preferring lower pulse to avoid "
    r"double-time misread\.")


def _snare_backbeat_bpm(drum_notes):
    """(bpm, share, intervals) from the snare hits of the drums stem when one
    spacing dominates, read as a backbeat (snare on 2 and 4: two beats); else
    None. Takes scan-time note dicts (onset, drum_type) or saved ones (t, k)."""
    t = []
    for n in drum_notes or []:
        try:
            k = n.get("drum_type") if "drum_type" in n else n.get("k")
            if k != "snare":
                continue
            t.append(float(n["onset"] if "onset" in n else n["t"]))
        except Exception:
            continue
    t.sort()
    iois = [b - a for a, b in zip(t, t[1:]) if 0.25 < b - a < 2.5]
    if len(iois) < int(TEMPO_SNARE_MIN_INTERVALS):
        return None
    bins = {}
    for i in iois:
        k = int(round(i / 0.02))
        bins[k] = bins.get(k, 0) + 1
    best = max(bins, key=lambda k: (bins[k] + bins.get(k - 1, 0) + bins.get(k + 1, 0), -k))
    share = (bins[best] + bins.get(best - 1, 0) + bins.get(best + 1, 0)) / float(len(iois))
    if share < float(TEMPO_SNARE_MIN_SHARE):
        return None
    sel = [i for i in iois if abs(i / 0.02 - best) <= 1.5]
    ioi = sum(sel) / len(sel)
    return 120.0 / ioi, share, ioi


def _tempo_snare_tiebreak(text, final_bpm, drum_notes):
    """dev19 TEMPO_SNARE_TIEBREAK, then (when it does not fire) the dev20
    TEMPO_OCTAVE_PAIR_CHECK. (text, bpm) or None."""
    drum_notes = (_structure_stem_evidence({"drums": drum_notes}) or {}).get("drums")
    r = _tempo_snare_tiebreak_core(text, final_bpm, drum_notes)
    if r is None and TEMPO_OCTAVE_PAIR_CHECK:
        r = _tempo_octave_pair_check(text, final_bpm, drum_notes)
    if r is None and (TEMPO_HALFTIME_CHECK or TEMPO_HALFTIME_BOTH_READINGS):
        r = _tempo_halftime_check(text, final_bpm, drum_notes)       # pre60
    return r


_TEMPO_MOSS_DOUBLE_RE = re.compile(
    r"^(?P<hs>.+?) \((?P<hi>[\d.]+)\) is ~2x MOSS-Music \((?P<lo>[\d.]+)\);")


def _tempo_octave_pair_check(text, final_bpm, drum_notes):
    """TEMPO_OCTAVE_PAIR_CHECK (see the setting): (text, bpm) with the
    RECOMMENDED TEMPO block rewritten, or None."""
    try:
        if not final_bpm or not drum_notes:
            return None
        t = str(text or "")
        bm = None
        for m_ in _TEMPO_BLOCK_RE.finditer(t):
            bm = m_
        if not bm or int(bm.group("bpm")) != int(final_bpm) or "TEMPO CHECK" in bm.group("why"):
            return None
        dm = _TEMPO_MOSS_DOUBLE_RE.match(bm.group("why"))
        if not dm:
            return None
        hi, moss = float(dm.group("hi")), float(dm.group("lo"))
        if int(final_bpm) != int(_tempo_quantize(moss)):
            return None
        em = re.search(r"Essentia estimated tempo \(raw detector\):\s*([\d.]+)\s*BPM", t)
        if not em:
            return None
        ess = float(em.group(1))
        lo_d, hi_d = min(ess, hi), max(ess, hi)
        if lo_d <= 0 or abs(hi_d / lo_d - 2.0) > 0.06:
            return None
        tol = float(TEMPO_OCTAVE_PAIR_TOL)
        if any(abs(moss - r) / r <= tol for r in (lo_d, hi_d, lo_d / 2.0, hi_d * 2.0)):
            return None
        cand = min((lo_d, hi_d), key=lambda r: abs(math.log(moss / r)))
        sb = _snare_backbeat_bpm(drum_notes)
        if not sb:
            return None
        sbpm, share, ioi = sb
        if abs(sbpm - cand) / cand > float(TEMPO_SNARE_TOL):
            return None
        q = _tempo_quantize(cand)
        if TEMPO_SNARE_HALFTIME_GUARD and q > int(final_bpm):
            ht = _drum_halftime_at(drum_notes, q, TEMPO_SNARE_HALFTIME_GUARD_SHARE)   # pre60
            if ht:
                return _tempo_guard_note(t, bm, int(final_bpm), q, ioi, ht), int(final_bpm)

        why = (bm.group("why").rstrip() + f" TEMPO CHECK: MOSS-Music's pulse ({moss:g}) is not either detector's "
               f"reading or its half or double, while the detectors agree with each other an octave apart "
               f"({lo_d:g} and {hi_d:g}); the snare on the separated drums lands every {ioi:.2f} s ({share * 100:.0f}% "
               f"of its gaps), a backbeat on 2 and 4 at {sbpm:.0f} BPM, so {cand:g} is the pulse: {q} BPM.")
        return t[:bm.start()] + _tempo_block_line(q, why) + t[bm.end():], q
    except Exception:
        return None


def _tempo_snare_tiebreak_core(text, final_bpm, drum_notes):
    """dev19 TEMPO_SNARE_TIEBREAK: (text, bpm) with the RECOMMENDED TEMPO block
    rewritten when the tempo was the lower of two readings ~2x apart only by
    the "prefer the lower pulse" default, and the snare on the separated drums
    falls every two beats of the HIGHER reading (within TEMPO_SNARE_TOL) and not
    of the lower; else None. the alt-rock test track (156 BPM) was reported at 78:
    Essentia 156 against the other detector's 78.2, both outside the rock
    range, and no listening-model reading to break the tie; its snare lands
    every 0.77 s."""
    try:
        if not TEMPO_SNARE_TIEBREAK or not final_bpm or not drum_notes:
            return None
        t = str(text or "")
        bm = None
        for m_ in _TEMPO_BLOCK_RE.finditer(t):
            bm = m_
        if not bm or int(bm.group("bpm")) != int(final_bpm):
            return None
        dm = _TEMPO_DEFAULT_LOWER_RE.match(bm.group("why"))
        if not dm:
            return None
        hi, lo = float(dm.group("hi")), float(dm.group("lo"))
        if hi < lo:
            return None
        sb = _snare_backbeat_bpm(drum_notes)
        if not sb:
            return None
        sbpm, share, ioi = sb
        tol = float(TEMPO_SNARE_TOL)
        if abs(sbpm - hi) / hi > tol or abs(sbpm - lo) / lo <= tol:
            return None
        q = _tempo_quantize(hi)
        if TEMPO_SNARE_HALFTIME_GUARD:
            ht = _drum_halftime_at(drum_notes, q, TEMPO_SNARE_HALFTIME_GUARD_SHARE)   # pre60
            if ht:
                return _tempo_guard_note(t, bm, int(final_bpm), q, ioi, ht), int(final_bpm)

        why = (bm.group("why").rstrip() + f" TEMPO CHECK: the snare on the separated drums lands every "
               f"{ioi:.2f} s ({share * 100:.0f}% of its gaps), a backbeat on 2 and 4 at {sbpm:.0f} BPM, so "
               f"the faster reading, {hi:g}, is the pulse: {q} BPM.")
        block = (f"RECOMMENDED TEMPO FOR DISCUSSION: {q} BPM. Reasoning: {why} This is the primary tempo to "
                 "report to the user. State it as a concrete figure (e.g. 'about "
                 f"{q} BPM'). Do not expand it into a range unless this block itself marks the value as uncertain.")
        return t[:bm.start()] + block + t[bm.end():], q
    except Exception:
        return None


def _tempo_drum_lists(drum_notes):
    """pre60: (kicks, snares, hi-hats) onset lists from scan-time note dicts
    (onset, drum_type) or saved ones (t, k), as _summarize_drum_rhythm groups them."""
    by = {}
    for n in drum_notes or []:
        try:
            k = n.get("drum_type") if "drum_type" in n else n.get("k")
            by.setdefault(k or "other", []).append(float(n["onset"] if "onset" in n else n["t"]))
        except Exception:
            continue
    return (sorted(by.get("kick") or []), sorted(by.get("snare") or []),
            sorted(by.get("hihat") or by.get("hi-hat") or []))


def _drum_halftime_share(drum_notes, bpm):
    """pre60: (share, stretches). The song is cut into TEMPO_HALFTIME_WINDOW_BEATS-beat
    stretches at bpm; of those whose snare _classify_snare_grid can name (on the
    tempo-steadied timeline the drum report uses), the share it names a half-time
    backbeat -- one main snare per bar, which is an ordinary 2-and-4 backbeat at half
    the tempo. (0.0, 0) when there is too little to judge."""
    try:
        if not bpm or float(bpm) <= 0:
            return 0.0, 0
        kicks, snares, hats = _tempo_drum_lists(drum_notes)
        if len(snares) < 20:
            return 0.0, 0
        fit = _refine_bpm_from_onsets(kicks + snares + hats, float(bpm))
        if not fit or float(fit) <= 0:
            return 0.0, 0
        try:
            warp = _tempo_warp_onsets(kicks + snares + hats, fit)
        except Exception:
            warp = None
        if warp is not None:
            kicks = sorted(warp(t) for t in kicks)
            snares = sorted(warp(t) for t in snares)
        span = float(TEMPO_HALFTIME_WINDOW_BEATS) * 60.0 / float(fit)
        t, end = snares[0], snares[-1]
        named = half = 0
        while t < end:
            s_ = [x for x in snares if t <= x < t + span]
            if len(s_) >= 6:
                lab, _ = _classify_snare_grid(s_, fit, kicks=[x for x in kicks if t <= x < t + span])
                if lab:
                    named += 1
                    half += 1 if str(lab).startswith("half-time backbeat") else 0
            t += span
        return (half / float(named) if named else 0.0), named
    except Exception:
        return 0.0, 0


def _drum_halftime_at(drum_notes, bpm, need=None):
    """pre60: (share, stretches) when the drums play a half-time backbeat at bpm
    (TEMPO_HALFTIME_SHARE -- or need -- of TEMPO_HALFTIME_MIN_WINDOWS stretches or
    more), else None."""
    share, n = _drum_halftime_share(drum_notes, bpm)
    if n >= int(TEMPO_HALFTIME_MIN_WINDOWS) and share >= float(TEMPO_HALFTIME_SHARE if need is None else need):
        return share, n
    return None


_TEMPO_AGREE_RE = re.compile(r"^Essentia and objective detector agree \(~(?P<avg>[\d.]+), averaged\)\.")
# pre60 TEMPO_HALFTIME_LISTENING_PAIR: the listening model and one detector agree;
# the group names the detector that was left out.
_TEMPO_MOSS_AGREE_RE = re.compile(
    r"^(?:MOSS-Music and objective detector agree|Essentia and MOSS-Music agree) \(~(?P<avg>[\d.]+), averaged\)\.")
_TEMPO_BOTH_MARK = "can also be counted at half that speed"
_TEMPO_GENRE_FAM_RE = re.compile(r"Genre prior(?: \([^)]*\))?: (?P<fam>[a-z]+) \(")
_TEMPO_FAM_WORDS = {
    "electronicish": "dance and electronic music", "hiphop": "hip-hop", "rockish": "rock",
    "popish": "pop", "ballad": "a ballad", "other": "most music",
}


def _tempo_halftime_check(text, final_bpm, drum_notes):
    """pre60 TEMPO_HALFTIME_CHECK (see the setting): (text, bpm) with the
    RECOMMENDED TEMPO block rewritten at half the detectors' shared reading, or None."""
    try:
        if not (TEMPO_HALFTIME_CHECK or TEMPO_HALFTIME_BOTH_READINGS) or not final_bpm or not drum_notes:
            return None
        t = str(text or "")
        bm = None
        for m_ in _TEMPO_BLOCK_RE.finditer(t):
            bm = m_
        if not bm or int(bm.group("bpm")) != int(final_bpm):
            return None
        why = bm.group("why")
        am = _TEMPO_AGREE_RE.match(why)
        pair = None                                  # pre60: the left-out detector's reading
        if not am and TEMPO_HALFTIME_CHECK and TEMPO_HALFTIME_LISTENING_PAIR:
            am = _TEMPO_MOSS_AGREE_RE.match(why)
            if am:
                src = (r"Essentia estimated tempo \(raw detector\):\s*([\d.]+)\s*BPM"
                       if why.startswith("MOSS-Music and objective")
                       else r"(?<!Essentia )estimated tempo \(raw detector\):\s*([\d.]+)\s*BPM")
                om = re.search(src, t)
                pair = float(om.group(1)) if om else None
                if not pair or abs(float(am.group("avg")) / pair - 2.0) > 0.06:
                    return None
        if not am:
            return None
        avg = float(am.group("avg"))
        if int(_tempo_quantize(avg)) != int(final_bpm):
            return None
        both = False
        if avg < float(TEMPO_HALFTIME_MIN_BPM):
            # pre60 TEMPO_HALFTIME_BOTH_READINGS: just under the floor -- name both readings
            if (pair is not None or not TEMPO_HALFTIME_BOTH_READINGS
                    or avg < float(TEMPO_HALFTIME_BOTH_MIN_BPM) or _TEMPO_BOTH_MARK in why):
                return None
            both = True
        elif not TEMPO_HALFTIME_CHECK:
            return None
        gm = _TEMPO_GENRE_FAM_RE.search(why)
        fam = gm.group("fam") if gm and gm.group("fam") in _BPM_GENRE_PREFS else "other"
        if fam == "dnb" or avg <= float(_BPM_GENRE_PREFS[fam][1]):
            return None
        ht = _drum_halftime_at(drum_notes, int(final_bpm))
        if not ht:
            return None
        share, n = ht
        half = avg / 2.0
        q = _tempo_quantize(half)
        drums = (f"the separated drums play one main snare per bar at it (a half-time backbeat) in "
                 f"{int(round(share * n))} of the song's {n} stretches of {int(TEMPO_HALFTIME_WINDOW_BEATS)} beats")
        fam_w = _TEMPO_FAM_WORDS.get(fam, "most music")
        if both:
            why = (why.rstrip() + f" TEMPO CHECK: that is above the usual range for {fam_w}, and {drums}, so "
                   f"the song {_TEMPO_BOTH_MARK}, about {q} BPM, with that backbeat as the beat; a double-time "
                   f"reading is still possible.")
            return t[:bm.start()] + _tempo_block_line(int(final_bpm), why) + t[bm.end():], int(final_bpm)
        heard = ("" if pair is None else
                 f" ({'Essentia' if why.startswith('MOSS-Music and objective') else 'the objective detector'} "
                 f"measured {pair:g}, half of it)")
        why = (why.rstrip() + f" TEMPO CHECK: that is a fast reading{heard}, above the usual range for "
               f"{fam_w}, and {drums} -- an ordinary 2-and-4 backbeat at half that speed -- so the pulse is "
               f"{half:.1f}: {q} BPM.")
        return t[:bm.start()] + _tempo_block_line(q, why) + t[bm.end():], q
    except Exception:
        return None


def _tempo_guard_note(t, bm, bpm, faster, ioi, ht):
    """pre60 TEMPO_SNARE_HALFTIME_GUARD: the analysis with a TEMPO CHECK note in its
    RECOMMENDED TEMPO block saying why the faster reading was not used (the tempo
    stays bpm). The note also keeps the exporter's own copy of the dev19 checks,
    which leaves a block with a TEMPO CHECK alone, from making that move itself."""
    share, n = ht
    if "TEMPO CHECK: the snare's most common spacing" in bm.group("why"):
        return t                                   # already said (the check run a second time)
    why = (bm.group("why").rstrip() + f" TEMPO CHECK: the snare's most common spacing ({ioi:.2f} s) would "
           f"fit a backbeat on 2 and 4 at {faster} BPM, but at {faster} BPM the separated drums play one main "
           f"snare per bar (a half-time backbeat) in {int(round(share * n))} of the song's {n} stretches of "
           f"{int(TEMPO_HALFTIME_WINDOW_BEATS)} beats, so the slower reading stands: {bpm} BPM.")
    return t[:bm.start()] + _tempo_block_line(bpm, why) + t[bm.end():]


_TEMPO_SNARE_FLIP_RE = re.compile(
    r" TEMPO CHECK: the snare on the separated drums lands every [\d.]+ s \(\d+% of its gaps\), a "
    r"backbeat on 2 and 4 at \d+ BPM, so the faster reading, [\d.]+, is the pulse: (?P<q>\d+) BPM\.")


def _tempo_saved_pre60(text, old, stem_drums):
    """pre60, for a save whose tempo note already has a TEMPO CHECK: (text, bpm)
    after TEMPO_SNARE_HALFTIME_GUARD (a dev19 snare tie-break that the guard now
    stops is undone) or TEMPO_HALFTIME_CHECK, or None. Only the RECOMMENDED TEMPO
    block changes."""
    try:
        t = str(text or "")
        bm = None
        for m_ in _TEMPO_BLOCK_RE.finditer(t):
            bm = m_
        if not bm or int(bm.group("bpm")) != int(old):
            return None
        drums = (_structure_stem_evidence({"drums": stem_drums}) or {}).get("drums")
        if not drums:
            return None
        why = bm.group("why")
        if TEMPO_SNARE_HALFTIME_GUARD and TEMPO_SNARE_TIEBREAK:
            fm = _TEMPO_SNARE_FLIP_RE.search(why)
            dm = _TEMPO_DEFAULT_LOWER_RE.match(why)
            if fm and dm and int(fm.group("q")) == int(old) and not why[fm.end():].strip():
                lo = float(dm.group("lo"))
                if lo < float(dm.group("hi")) and _drum_halftime_at(drums, int(old), TEMPO_SNARE_HALFTIME_GUARD_SHARE):
                    q = _tempo_quantize(lo)
                    return t[:bm.start()] + _tempo_block_line(q, why[:fm.start()].rstrip()) + t[bm.end():], q
        return _tempo_halftime_check(t, old, drums)
    except Exception:
        return None


def _tempo_block_line(bpm, why):
    return (f"RECOMMENDED TEMPO FOR DISCUSSION: {bpm} BPM. Reasoning: {why} This is the primary tempo to "
            f"report to the user. State it as a concrete figure (e.g. 'about {bpm} BPM'). Do not expand it "
            "into a range unless this block itself marks the value as uncertain.")


def _tempo_saved_view(text, stem_midi=None):
    """(text, new_bpm or None): the dev19 tempo checks applied to an analysis
    saved before dev19, from the detector readings, the reasoning and the
    saved drum hits it keeps. Only the RECOMMENDED TEMPO block changes; it
    says that beat-counting figures elsewhere were worked out at the old
    tempo."""
    t = str(text or "")
    try:
        if not (TEMPO_LISTENING_CHECK or TEMPO_SNARE_TIEBREAK or TEMPO_HALFTIME_CHECK):
            return t, None
        bm = None
        for m_ in _TEMPO_BLOCK_RE.finditer(t):
            bm = m_
        if bm and "TEMPO CHECK" in bm.group("why") and (TEMPO_HALFTIME_CHECK or TEMPO_SNARE_HALFTIME_GUARD):
            # pre60: a save whose tempo already had a check -- only the pre60 checks
            old = int(bm.group("bpm"))
            r = _tempo_saved_pre60(t, old, ((stem_midi or {}).get("drums") if isinstance(stem_midi, dict) else None))
            if not r or int(r[1]) == old:
                return t, None
            return _tempo_saved_note(r[0], old, int(r[1])), int(r[1])
        if not (TEMPO_LISTENING_CHECK or TEMPO_SNARE_TIEBREAK):
            # pre60: only TEMPO_HALFTIME_CHECK of the saved-file checks is on
            if not bm:
                return t, None
            old = int(bm.group("bpm"))
            r = _tempo_halftime_check(t, old, (_structure_stem_evidence(
                {"drums": (stem_midi or {}).get("drums") if isinstance(stem_midi, dict) else None}) or {}).get("drums"))
            if not r or int(r[1]) == old:
                return t, None
            return _tempo_saved_note(r[0], old, int(r[1])), int(r[1])
        if not bm or "TEMPO CHECK" in bm.group("why"):
            return t, None
        old, why = int(bm.group("bpm")), bm.group("why")
        num = lambda rx: (lambda m: float(m.group(1)) if m else None)(re.search(rx, t))
        ess = num(r"Essentia estimated tempo \(raw detector\):\s*([\d.]+)\s*BPM")
        obj = num(r"(?<!Essentia )estimated tempo \(raw detector\):\s*([\d.]+)\s*BPM")
        ess_med = extract_essentia_median_bpm(t)
        obj_med = extract_objective_median_bpm(t)
        nb, nn = old, why
        mm = re.search(r"MOSS-Music(?: musical pulse)? \(~?([\d.]+)\)", why)
        em = _TEMPO_ECHO_RE.search(t)
        if TEMPO_LISTENING_CHECK and mm and em:
            ev = re.search(r"[=:]\s*\[?\s*(?:~|\u2248)?\s*(\d+(?:\.\d+)?)", em.group(0))
            if ev and abs(float(ev.group(1)) - float(mm.group(1))) < 0.05:
                rb, rn = reconcile_bpm(None, ess, obj, essentia_median_bpm=None, objective_median_bpm=None,
                                       genre_hint=_genre_hint_from_analysis_text(t))
                if rb:
                    nb, nn = _tempo_listening_check(rb, rn, None, ess, obj, ess_med, obj_med, True)
        elif TEMPO_LISTENING_CHECK and why.startswith("Using MOSS-Music musical pulse") and mm:
            nb, nn = _tempo_listening_check(old, why, float(mm.group(1)), ess, obj, ess_med, obj_med)
        if TEMPO_COPY_CHECK and nb == old and "MOSS-Music" in why:
            # dev46: the listening model's TEMPO_BPM was a copy of a detector reading
            _tm = re.search(r"TEMPO_BPM\s*=[^\n]*", t)
            _mv = extract_bpm_from_text(_tm.group(0)) if _tm else None
            _cv = _moss_tempo_copies_detector(_mv, ess, obj, ess_med, obj_med)
            if _cv is not None:
                rb, rn = reconcile_bpm(None, ess, obj, essentia_median_bpm=None, objective_median_bpm=None,
                                       genre_hint=_genre_hint_from_analysis_text(t))
                if rb:
                    rb, rn = _tempo_listening_check(rb, rn, None, ess, obj, ess_med, obj_med)
                    if rb and abs(rb - old) / float(old) > 0.10:
                        nb, nn = rb, (str(rn).rstrip().rstrip(".") + ". " + _TEMPO_COPY_NOTE.format(m=_mv, v=_cv))
        nn = _sentence(nn) if nn != why else nn
        out = t[:bm.start()] + _tempo_block_line(nb, nn) + t[bm.end():] if nb != old else t
        tb = _tempo_snare_tiebreak(out, nb, ((stem_midi or {}).get("drums") if isinstance(stem_midi, dict) else None))
        if tb:
            out, nb = tb
        if nb == old:
            return t, None
        return _tempo_saved_note(out, old, nb), nb
    except Exception:
        return t, None


def _tempo_saved_note(out, old, nb):
    """The saved-file tempo note: beat-counting figures elsewhere were worked out at old."""
    return out.replace(
        " This is the primary tempo to report to the user. State it as a concrete figure (e.g. 'about "
        f"{nb} BPM').",
        f" Figures elsewhere in these notes that count beats or bars were worked out at {old} BPM, the "
        "tempo this file was first given. This is the primary tempo to report to the user. State it as a "
        f"concrete figure (e.g. 'about {nb} BPM').", 1)


def extract_bpm_from_text(text):
    """Extracts the first numeric BPM value from text."""
    if not text:
        return None
    # Look for TEMPO_BPM= or estimated tempo ... BPM
    m = re.search(r"\bTEMPO(?:_BPM)?\s*[=:]\s*\[?\s*(?:~|approx\.?|about|around|c\.)?\s*([0-9]+(?:\.[0-9]+)?)", text, re.IGNORECASE)
    if not m:
        # Tolerant fallback for the looser spellings the model sometimes emits
        # instead of the requested label ("**TEMPO_BPM**=96", "Tempo BPM: 120",
        # "TEMPO_BPM=approximately 124"). Without it the model's own tempo was
        # silently dropped from tempo reconciliation.
        m = re.search(
            r"\bTEMPO(?:[_ ]?BPM)?\**\s*[=:]\s*\**\s*\[?\s*"
            r"(?:~|\u2248|approx(?:imately|\.)?|about|around|roughly|circa|c\.)?\s*"
            r"([0-9]+(?:\.[0-9]+)?)",
            text, re.IGNORECASE,
        )
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
    drum & bass keeps a fast reading while house/techno prefers ~120-130.
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
                f"fast detector value ({bpm:.1f}) kept — genre prior ({fam_note})"
                + (f"; corroborated (~{corroborating_bpm} BPM)" if _corroborates(bpm) else "")
                + "."
            )
            return preferred, cands, note
        if bpm > 185.0 and _in_pref_band(half):
            preferred = half
            note = (
                f"very fast detector value ({bpm:.1f}) outside typical DnB band; "
                f"preferring half ({half}) which fits genre prior better."
            )
            return preferred, cands, note

    if bpm >= 150.0 and 70.0 <= half <= 100.0:
        if _corroborates(bpm):
            note = (
                f"fast detector value ({bpm:.1f}) matches the tracker's own median beat "
                f"spacing (~{corroborating_bpm} BPM); keeping it, though that is the same "
                "tracker checking itself, so a double-time reading is still possible."
            )
        elif not prefer_half and _in_pref_band(bpm):
            note = f"fast detector value ({bpm:.1f}) kept due to genre prior ({fam_note})."
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
                f"very fast detector value ({bpm:.1f}) matches the tracker's own median beat "
                f"spacing (~{corroborating_bpm} BPM); keeping it, though that is the same "
                "tracker checking itself, so a double-time reading is still possible."
            )
        elif genre_fam == "dnb" and _in_pref_band(bpm):
            note = f"very fast detector value ({bpm:.1f}) kept — genre prior ({fam_note})."
        else:
            preferred = half
            note = "very fast detector value may be a double-time reading; preferring the slower candidate."
    elif bpm < 70.0 and 80.0 <= double <= 140.0:
        if corroborating_bpm is not None and _corroborates(bpm):
            note = (
                f"slow detector value ({bpm:.1f}) matches the tracker's own median beat "
                f"spacing (~{corroborating_bpm} BPM); keeping it, though that is the same "
                "tracker checking itself, so a half-time reading is still possible."
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


def _estimate_meter_from_beats(beat_times, accents):
    """Estimate common meter from an already-detected beat sequence.

    The beat detector supplies the pulse; this helper only chooses the recurring
    accent cycle, so callers do not have to run beat tracking a second time.
    Returns (downbeat_times, beats_per_bar) or (beat_times, 1) when inconclusive.
    """
    try:
        beat_times = np.asarray(beat_times, dtype=float)
        accents = np.asarray(accents, dtype=float)
        if beat_times.size < 4 or accents.size != beat_times.size:
            return beat_times.tolist(), 1
        # Score each (meter, phase) with a two-sample t statistic rather than a
        # raw mean difference. The old score ignored how many beats fell in the
        # "downbeat" set, so long cycles (8, 9, 12) -- which have few downbeats
        # and many phases to choose from -- won on noise alone: on synthetic
        # 4/4 accents it picked 12 or 8 about 75% of the time. The t statistic
        # accounts for sample size, and a mild prior toward common meters stops
        # a longer cycle from winning on a marginal difference.
        n_total = int(beat_times.size)
        all_idx = np.arange(n_total)
        best_by_bpb = {}
        for bpb in (4, 3, 6, 2, 8, 5, 7, 9, 12):
            if n_total < bpb * 3:
                continue
            for phase in range(bpb):
                in_mask = (all_idx % bpb) == phase
                n_in = int(np.count_nonzero(in_mask))
                n_out = n_total - n_in
                if n_in < 2 or n_out < 2:
                    continue
                a_in = accents[in_mask]
                a_out = accents[~in_mask]
                pooled = np.sqrt(
                    ((n_in - 1) * float(np.var(a_in, ddof=1))
                     + (n_out - 1) * float(np.var(a_out, ddof=1)))
                    / float(max(1, n_in + n_out - 2))
                ) + 1e-6
                se = pooled * np.sqrt(1.0 / n_in + 1.0 / n_out)
                t = (float(np.mean(a_in)) - float(np.mean(a_out))) / se
                prev = best_by_bpb.get(bpb)
                if prev is None or t > prev[0]:
                    best_by_bpb[bpb] = (t, phase)
        if not best_by_bpb:
            return beat_times.tolist(), 1

        # Prior toward the meters popular music actually uses at the tracked
        # pulse. A less common cycle must beat 4/4 or 3/4 by a clear margin.
        prior = {4: 1.0, 3: 1.0, 2: 0.8, 6: 0.8, 5: 0.7, 7: 0.7, 8: 0.7, 9: 0.65, 12: 0.65}
        scored = []
        for bpb, (t, phase) in best_by_bpb.items():
            w = prior.get(bpb, 0.6)
            eff = t * w if t > 0 else t / max(w, 1e-6)
            scored.append((eff, 1 if bpb == 4 else 0, bpb, phase, t))
        scored.sort(reverse=True)
        _eff, _pref4, bpb, phase, t_best = scored[0]

        # A 2-beat cycle and a 4-beat cycle are hard to tell apart when beat 3
        # carries a secondary accent. Keep 4 when the would-be downbeat is
        # itself clearly stronger than the beat half a bar later.
        if bpb == 2 and 4 in best_by_bpb and n_total >= 16:
            best4 = None
            for p4 in (phase, phase + 2):
                a_down = accents[(all_idx % 4) == p4]
                a_mid = accents[(all_idx % 4) == ((p4 + 2) % 4)]
                if a_down.size >= 3 and a_mid.size >= 3:
                    sd = np.sqrt((float(np.var(a_down, ddof=1)) + float(np.var(a_mid, ddof=1))) / 2.0) + 1e-6
                    t4 = (float(np.mean(a_down)) - float(np.mean(a_mid))) / (
                        sd * np.sqrt(1.0 / a_down.size + 1.0 / a_mid.size))
                    if best4 is None or t4 > best4[0]:
                        best4 = (t4, p4)
            if best4 is not None and best4[0] >= 2.5:
                bpb, phase = 4, best4[1]

        # No meaningful accent cycle at all: fall back to the common 4-beat bar
        # (already the downstream default) instead of asserting an odd meter.
        # The threshold allows for having searched many (meter, phase) pairs.
        if t_best < 3.0 and 4 in best_by_bpb:
            bpb, phase = 4, best_by_bpb[4][1]
        return beat_times[phase::bpb].tolist(), int(bpb)
    except Exception:
        return beat_times.tolist(), 1


def _downbeat_accent_curve(y, sr, beat_times, onset_env=None, hop_length=512):
    """Per-beat evidence that a beat is the FIRST beat of a bar.

    Plain broadband onset strength was used before, and it favours the snare:
    a backbeat on 2 and 4 has more high-frequency flux than the kick on 1 and
    3, so the bar phase routinely locked onto the backbeat (and often reported
    a 2-beat meter). Every section boundary is then snapped onto the wrong
    beat of the bar. Downbeats are better marked by low-frequency onsets (kick,
    bass entries) and by harmonic change (chords usually change on beat 1), so
    those lead here, with broadband onsets kept as a minor term.

    Returns an array aligned with beat_times. Falls back to the plain onset
    envelope if any of the extra features cannot be computed.
    """
    beat_times = np.asarray(beat_times, dtype=float)
    if onset_env is None:
        onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    frame_times = librosa.frames_to_time(np.arange(len(onset_env)), sr=sr, hop_length=hop_length)
    full = np.interp(beat_times, frame_times, onset_env)

    def _z(a):
        a = np.asarray(a, dtype=float)
        sd = float(np.std(a))
        return (a - float(np.mean(a))) / sd if sd > 1e-9 else np.zeros_like(a)

    parts = [(0.15, _z(full))]
    try:
        low_env = librosa.onset.onset_strength(
            y=y, sr=sr, hop_length=hop_length, fmax=200.0, n_mels=24
        )
        low_t = librosa.frames_to_time(np.arange(len(low_env)), sr=sr, hop_length=hop_length)
        # Peak in a short window around each beat: a kick's low-band flux can
        # trail the broadband onset the beat tracker locked to.
        win = 0.05
        low = np.array([
            float(np.max(low_env[(low_t >= t - win) & (low_t <= t + win)]))
            if np.any((low_t >= t - win) & (low_t <= t + win)) else 0.0
            for t in beat_times
        ])
        parts.append((0.5, _z(low)))
    except Exception:
        pass
    try:
        chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop_length)
        bf = librosa.time_to_frames(beat_times, sr=sr, hop_length=hop_length)
        bf = np.clip(bf, 0, chroma.shape[1] - 1)
        # Mean chroma of the interval AFTER each beat vs the interval BEFORE it,
        # built explicitly per beat. This used to go through librosa.util.sync
        # and assume column k+1 starts at beat k -- but beat_track(trim=False)
        # routinely places the first beat at frame 0, sync then emits one column
        # fewer, the length check failed, and the harmonic-change term was
        # silently dropped on most tracks. Without it the accent curve cannot
        # tell beat 1 from beat 3 (kicks land on both), which is what kept
        # ordinary 4/4 songs being reported as 2/4 with half-bar "downbeats".
        n_frames = chroma.shape[1]
        n_beats = int(beat_times.size)
        typical = int(np.median(np.diff(bf))) if n_beats > 1 else 8
        typical = max(1, typical)

        def _seg_mean(a, b):
            a = int(max(0, min(a, n_frames - 1)))
            b = int(max(a + 1, min(b, n_frames)))
            v = np.mean(chroma[:, a:b], axis=1)
            nv = float(np.linalg.norm(v))
            return v / nv if nv > 1e-9 else None

        change = np.zeros(n_beats)
        valid = np.zeros(n_beats, dtype=bool)
        for k in range(n_beats):
            nxt = int(bf[k + 1]) if k + 1 < n_beats else int(bf[k]) + typical
            prv = int(bf[k - 1]) if k > 0 else int(bf[k]) - typical
            if prv < 0 or int(bf[k]) <= prv:
                continue
            after = _seg_mean(int(bf[k]), nxt)
            before = _seg_mean(prv, int(bf[k]))
            if after is None or before is None:
                continue
            change[k] = 1.0 - float(np.dot(after, before))
            valid[k] = True
        if np.count_nonzero(valid) >= max(4, n_beats // 2):
            if not np.all(valid):
                change[~valid] = float(np.median(change[valid]))
            parts.append((0.35, _z(change)))
    except Exception:
        pass
    if len(parts) == 1:
        return full
    total_w = sum(w for w, _ in parts)
    return sum(w * a for w, a in parts) / total_w


def _legacy_beat_and_downbeat_grid(y, sr, bpm_hint=None):
    """Previous grid estimator (512-sample hop, frame-quantised tempo). Kept as
    the fallback for _beat_and_downbeat_grid.

    Estimate the metrical grid: beat times, bar (downbeat) times, beat length.

    Section boundaries in almost all popular music land on a BAR LINE, not at an
    arbitrary frame index. Everything downstream in this pipeline used to report
    boundaries straight off a 0.25 s novelty frame grid, which is why timestamps
    were consistently a fraction of a bar out even when the change point itself
    was detected correctly. Snapping to this grid is what turns a roughly-right
    detection into an accurate timestamp.

    Downbeats are estimated without a dedicated downbeat tracker (which we don't
    have here) by testing every possible bar phase against a spectral-flux
    accent curve and keeping the phase whose beats carry the most accent. That
    is a well-behaved proxy: bar-initial beats are, on average, the most
    strongly accented ones.

    Returns (beat_times, downbeat_times, beat_len_s, beats_per_bar) or
    (None, None, None, None) when the track is too short / too weak to grid.
    """
    try:
        if y is None or len(y) < sr * 4:
            return (None, None, None, None)

        onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=512)

        kwargs = {"onset_envelope": onset_env, "sr": sr, "hop_length": 512,
                  "trim": False, "units": "time"}
        if bpm_hint:
            try:
                bh = float(bpm_hint)
                # A hint is useful only as a weak lock. The beat tracker itself
                # must remain free to discover a different pulse; a reconciled
                # BPM is not allowed to become a structural prior.
                if 40.0 <= bh <= 220.0:
                    kwargs["start_bpm"] = bh
            except Exception:
                pass

        tempo, beat_times = librosa.beat.beat_track(y=y, **kwargs)
        beat_times = np.asarray(beat_times, dtype=float)
        if beat_times.size < 8:
            return (None, None, None, None)

        # Median inter-beat interval is far more robust than the reported tempo.
        beat_len = float(np.median(np.diff(beat_times)))
        if not (0.2 <= beat_len <= 2.0):
            return (None, None, None, None)

        accents = _downbeat_accent_curve(y, sr, beat_times, onset_env=onset_env, hop_length=512)

        downbeats, bpb = _estimate_meter_from_beats(beat_times, accents)
        return (beat_times.tolist(), downbeats, beat_len, bpb)
    except Exception:
        return (None, None, None, None)


def _beat_and_downbeat_grid(y, sr, bpm_hint=None):
    """Estimate the metrical grid: beat times, bar (downbeat) times, beat
    length and beats per bar.

    Uses _sx_beat_grid: a sub-frame tempo estimate and a kick-anchored onset
    envelope, so the grid does not drift (the previous estimator snapped the
    tempo to whole onset frames -- 120 BPM read as 117.45 BPM -- and every
    bar line after the first few drifted with it). Downbeats are then chosen
    from the accent curve exactly as before. Falls back to the previous
    estimator if anything fails.

    Returns (beat_times, downbeat_times, beat_len_s, beats_per_bar) or
    (None, None, None, None) when the track is too short / too weak to grid.
    """
    if not STRUCTURE_V2_ENABLED:
        return _legacy_beat_and_downbeat_grid(y, sr, bpm_hint=bpm_hint)
    try:
        if y is None or len(y) < sr * 4:
            return (None, None, None, None)
        beats, _bpm = _sx_beat_grid(y, sr, bpm_hint=bpm_hint)
        if not beats or len(beats) < 8:
            return _legacy_beat_and_downbeat_grid(y, sr, bpm_hint=bpm_hint)
        beat_times = np.asarray(beats, dtype=float)
        beat_len = float(np.median(np.diff(beat_times)))
        if not (0.2 <= beat_len <= 2.0):
            return _legacy_beat_and_downbeat_grid(y, sr, bpm_hint=bpm_hint)
        accents = _downbeat_accent_curve(y, sr, beat_times)
        downbeats, bpb = _estimate_meter_from_beats(beat_times, accents)
        return (beat_times.tolist(), downbeats, beat_len, bpb)
    except Exception:
        return _legacy_beat_and_downbeat_grid(y, sr, bpm_hint=bpm_hint)


def _snap_time_to_grid(t, grid, max_shift_s):
    """Move a time to the nearest grid point if one is within max_shift_s."""
    if not grid:
        return float(t)
    arr = np.asarray(grid, dtype=float)
    i = int(np.argmin(np.abs(arr - float(t))))
    if abs(arr[i] - float(t)) <= float(max_shift_s):
        return float(arr[i])
    return float(t)


def _beat_phase_shape(y, sr, beat_times, bins=24):
    """Fold the HARMONIC midrange (300 Hz-5 kHz) level onto the beat and read
    the shape of the average beat.

    Returns (shape, folded_range_db) where shape is
      "duck"   -- sharp drop, slow recovery: what a sidechain compressor does;
      "attack" -- sharp rise, slow decay: parts played ON the beat (strums,
                  stabs, picked figures);
      "weak"   -- the per-beat swings do not add up to a consistent shape
                  (folded range below BEAT_FOLD_MIN_CONSISTENT_DB), so they
                  are playing dynamics, not a compressor;
      None     -- not measurable, or neither shape is clear.

    The shape test uses only the ratio of the steepest rise to the steepest
    drop of the folded profile, so it does not depend on where the beat tracker
    put the beat (checked at +/-40 ms offsets). Percussive hits are removed
    first (soft HPSS at 11.025 kHz) so a snare cannot pose as an attack.
    """
    bt = np.asarray(beat_times if beat_times is not None else [], dtype=float)
    if bt.size < 8:
        return None, None
    # A beat grid found at half speed folds two real beats into one, which
    # blurs the shape. Subdivide until the fold period is a plausible beat.
    for _ in range(3):
        if float(np.median(np.diff(bt))) <= 0.75:
            break
        bt = np.sort(np.concatenate([bt, (bt[:-1] + bt[1:]) / 2.0]))
    sr_p = 11025 if sr > 11025 else sr
    yp = np.asarray(y, dtype=np.float32)
    if sr_p != sr:
        yp = librosa.resample(yp, orig_sr=sr, target_sr=sr_p)
    hop = 256   # ~23 ms
    f = librosa.fft_frequencies(sr=sr_p, n_fft=1024)
    m = (f >= 300.0) & (f < min(5000.0, sr_p / 2.0 - 1.0))
    # HPSS in 60 s blocks (+2 s context each side, far wider than the median
    # kernel) so peak memory stays ~100 MB whatever the track length.
    blk, ctx = 60 * sr_p, 2 * sr_p
    parts = []
    for a in range(0, len(yp), blk):
        lo, hi = max(0, a - ctx), min(len(yp), a + blk + ctx)
        Sm = np.abs(librosa.stft(yp[lo:hi], n_fft=1024, hop_length=hop, center=True))
        H = librosa.decompose.hpss(Sm, margin=1.0)[0]
        e = np.sqrt(np.mean(H[m] ** 2, axis=0))
        f0 = (a - lo) // hop
        n_keep = int(np.ceil(min(blk, len(yp) - a) / float(hop)))
        parts.append(e[f0:f0 + n_keep])
        del Sm, H
    band = np.concatenate(parts) if parts else np.zeros(0)
    if band.size < 32:
        return None, None
    db = 20.0 * np.log10(np.maximum(band, float(np.median(band)) * 1e-3 + 1e-12))
    fr = sr_p / float(hop)
    win = max(3, int(fr * 4.0)) | 1
    trend = np.convolve(np.pad(db, (win // 2, win // 2), mode="edge"), np.ones(win) / win, mode="valid")[: len(db)]
    detr = db - trend
    t = np.arange(len(detr)) / fr
    # Sample each beat at fixed phase points (interpolated), so the phase bins
    # never go empty when a bin is shorter than a frame.
    frac = (np.arange(bins) + 0.5) / float(bins)
    rows = []
    for a, b in zip(bt[:-1], bt[1:]):
        if b - a <= 0.1 or b > t[-1]:
            continue
        rows.append(np.interp(a + frac * (b - a), t, detr))
    if len(rows) < 8:
        return None, None
    prof = np.mean(rows, axis=0)
    rng = float(prof.max() - prof.min())
    if rng < BEAT_FOLD_MIN_CONSISTENT_DB:
        return "weak", rng
    q = np.convolve(np.concatenate([prof[-1:], prof, prof[:1]]), np.ones(3) / 3.0, mode="valid")
    d = np.diff(np.concatenate([q, q[:1]]))
    rise, drop = float(d.max()), float(-d.min())
    if drop <= 1e-9:
        return "attack", rng
    ratio = rise / drop
    if ratio <= 0.75:
        return "duck", rng
    if ratio >= 1.3:
        return "attack", rng
    return None, rng


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
    # Auxiliary percussion. These GM pitches were previously unmapped, so any
    # tambourine, cowbell, conga or shaker Omnizart placed on them fell through
    # to the three-bucket rank guess and was reported as a kick, snare or hat.
    # They get their own primary bucket rather than being forced into the kit,
    # which keeps the kick/snare/hihat groove statistics clean.
    54: "percussion", 56: "percussion", 58: "percussion",
    60: "percussion", 61: "percussion", 62: "percussion", 63: "percussion",
    64: "percussion", 65: "percussion", 66: "percussion", 67: "percussion",
    68: "percussion", 69: "percussion", 70: "percussion", 71: "percussion",
    72: "percussion", 73: "percussion", 74: "percussion", 75: "percussion",
    76: "percussion", 77: "percussion", 78: "percussion", 79: "percussion",
    80: "percussion", 81: "percussion",
}

# Specific instrument behind each GM pitch. Reported ALONGSIDE the primary
# drum_type, never instead of it: everything downstream groups on the coarse
# kick/snare/hihat/tom/cymbal buckets, so replacing those with finer labels
# would silently empty the groove analysis. The distinctions here are the ones
# a listener actually notices and asks about -- an open hat versus a closed
# hat, a ride versus a crash, a clap versus a real snare, which tom is being
# hit -- and previously all of them were flattened away before the report was
# written.
GM_DRUM_PITCH_SUBTYPE = {
    35: "acoustic bass drum", 36: "kick drum (electric/808)",
    37: "side stick / cross stick", 38: "acoustic snare",
    39: "hand clap", 40: "electric snare",
    41: "low floor tom", 43: "high floor tom", 45: "low tom",
    47: "low-mid tom", 48: "hi-mid tom", 50: "high tom",
    42: "closed hi-hat", 44: "pedal hi-hat", 46: "open hi-hat",
    49: "crash cymbal", 51: "ride cymbal", 52: "china cymbal",
    53: "ride bell", 55: "splash cymbal", 57: "crash cymbal 2",
    59: "ride cymbal 2",
    54: "tambourine", 56: "cowbell", 58: "vibraslap",
    60: "high bongo", 61: "low bongo", 62: "muted high conga",
    63: "open high conga", 64: "low conga", 65: "high timbale",
    66: "low timbale", 67: "high agogo", 68: "low agogo", 69: "cabasa",
    70: "maracas", 71: "short whistle", 72: "long whistle",
    73: "short guiro", 74: "long guiro", 75: "claves",
    76: "high wood block", 77: "low wood block", 78: "muted cuica",
    79: "open cuica", 80: "muted triangle", 81: "open triangle",
}


def _guess_drum_subtype_from_name(name):
    """Specific drum class from an instrument track name, for the Omnizart
    builds that name tracks instead of using GM pitches."""
    if not name:
        return None
    n = name.lower()
    for needle, label in (
        ("open hi", "open hi-hat"), ("openhat", "open hi-hat"),
        ("pedal hi", "pedal hi-hat"),
        ("closed hi", "closed hi-hat"), ("closedhat", "closed hi-hat"),
        ("hihat", "hi-hat"), ("hi-hat", "hi-hat"), ("hi hat", "hi-hat"),
        ("ride bell", "ride bell"), ("ride", "ride cymbal"),
        ("crash", "crash cymbal"), ("china", "china cymbal"),
        ("splash", "splash cymbal"), ("cymbal", "cymbal"),
        ("clap", "hand clap"), ("rim", "side stick / cross stick"),
        ("side stick", "side stick / cross stick"),
        ("floor tom", "floor tom"), ("high tom", "high tom"),
        ("mid tom", "mid tom"), ("low tom", "low tom"), ("tom", "tom"),
        ("808", "kick drum (electric/808)"),
        ("bass drum", "acoustic bass drum"), ("kick", "kick drum"),
        ("snare", "snare drum"),
        ("tambourine", "tambourine"), ("cowbell", "cowbell"),
        ("shaker", "shaker"), ("conga", "conga"), ("bongo", "bongo"),
        ("timbale", "timbale"), ("triangle", "triangle"),
        ("woodblock", "wood block"), ("wood block", "wood block"),
        ("clave", "claves"), ("maraca", "maracas"), ("cabasa", "cabasa"),
        ("guiro", "guiro"), ("percussion", "percussion"),
    ):
        if needle in n:
            return label
    return None


def _guess_drum_type_from_name(name):
    """Some Omnizart versions emit one pretty_midi.Instrument per drum
    class (named e.g. 'Bass Drum'/'Snare'/'Hi-Hat') rather than encoding
    the class purely via note pitch. Check the instrument name first so
    we don't miss the class label in that case."""
    if not name:
        return None
    n = name.lower()
    if "kick" in n or "bass drum" in n or "808" in n:
        return "kick"
    if "snare" in n or "clap" in n or "rim" in n or "side stick" in n:
        return "snare"
    if "hihat" in n or "hi-hat" in n or "hi hat" in n:
        return "hihat"
    if "tom" in n:
        return "tom"
    if "cymbal" in n or "crash" in n or "ride" in n or "china" in n or "splash" in n:
        return "cymbal"
    if any(k in n for k in (
        "tambourine", "cowbell", "shaker", "conga", "bongo", "timbale",
        "triangle", "woodblock", "wood block", "clave", "maraca", "cabasa",
        "guiro", "percussion",
    )):
        return "percussion"
    return None

def _drum_band_label(freq_hz):
    """Very rough register guess from spectral centroid — not a real pitch."""
    if freq_hz < 150:
        return "low (kick-like)"
    if freq_hz < 800:
        return "mid (snare/tom-like)"
    return "high (hihat/cymbal-like)"


def _refine_drum_onsets_from_audio(hits, stem_path, max_shift_s=0.045):
    """Move Omnizart drum hits from its own beat grid onto the real transients.

    Omnizart's drum model does not output onset times. It tracks beats with
    madmom, divides each beat into eight "mini-beats" (32nd notes), classifies
    each mini-beat, and writes every hit exactly ON a mini-beat
    (drum/inference.py, feature/beat_for_drum.py). Every hit time the groove
    code saw was therefore quantised to another beat tracker's grid, so
    grid-lock, swing and micro-timing were measuring madmom, not the drummer
    (a 2:1 swing cannot even be represented on a 32nd grid).

    For each hit, look within +/- max_shift_s for the strongest onset in that
    drum's own frequency band in the drums stem and move the hit there. A hit
    with no clear transient nearby keeps its grid time. The grid time is kept
    in "grid_onset". Returns (hits, n_moved). Never raises.
    """
    if not hits or not stem_path or not os.path.exists(str(stem_path)):
        return hits, 0
    try:
        y, sr = librosa.load(str(stem_path), sr=None, mono=True)
    except Exception:
        return hits, 0
    if y is None or y.size < 4096:
        return hits, 0
    hop = 64
    flux_cache = {}

    def _flux(dtype):
        """Onset function for one drum: rise of the band's log envelope."""
        if dtype not in flux_cache:
            lo, hi = _DRUM_ANALYSIS_BANDS.get(dtype, (30.0, 16000.0))
            env = _rms_envelope(_band_filter(y, sr, lo, hi), frame=256, hop=hop)
            loge = np.log1p(1000.0 * env / (float(np.max(env)) + 1e-12))
            fl = np.concatenate([[0.0], np.maximum(0.0, np.diff(loge))])
            base = float(np.median(fl)) + 1e-9
            flux_cache[dtype] = (fl, base)
        return flux_cache[dtype]

    try:
        from scipy.signal import hilbert as _hilbert
    except Exception:
        _hilbert = None

    def _attack_time(dtype, t_coarse):
        """Pin the attack in the time domain. Spectral flux finds the right
        transient but, being frame-based, leads the true onset by several ms
        (the analysis window starts filling before the frame centre). The
        band-limited envelope's 50%-of-peak crossing does not have that bias."""
        if _hilbert is None:
            return None
        lo, hi = _DRUM_ANALYSIS_BANDS.get(dtype, (30.0, 16000.0))
        a = int(max(0.0, t_coarse - 0.015) * sr)
        b = int((t_coarse + 0.040) * sr)
        if b - a < 64 or a >= y.size:
            return None
        xb = _drum_band_limit(y[a:min(b, y.size)], sr, lo, hi)
        env = np.abs(_hilbert(xb))
        w = max(1, int(sr * 0.0005))
        if w > 1:
            env = np.convolve(env, np.ones(w) / w, mode="same")
        m = int(np.argmax(env))
        if env[m] <= 0:
            return None
        below = np.flatnonzero(env[:m + 1] < 0.5 * env[m])
        i = int(below[-1]) + 1 if below.size else 0
        return (a + i) / float(sr)

    moved = 0
    for h in hits:
        dtype = h.get("drum_type")
        if dtype not in _DRUM_ANALYSIS_BANDS:
            continue
        try:
            t = float(h["onset"])
        except (KeyError, TypeError, ValueError):
            continue
        fl, base = _flux(dtype)
        a = int(max(0.0, t - max_shift_s) * sr / hop)
        b = int((t + max_shift_s) * sr / hop) + 1
        if a >= fl.size or b <= a:
            continue
        seg = fl[a:min(b, fl.size)]
        k = int(np.argmax(seg))
        if float(seg[k]) < 4.0 * base:
            continue                      # no clear transient: keep grid time
        coarse = (a + k) * hop / float(sr)
        fine = _attack_time(dtype, coarse)
        new_t = fine if (fine is not None and abs(fine - t) <= max_shift_s + 0.02) else coarse
        new_t = max(0.0, new_t)
        if abs(new_t - t) > 1e-4:
            h.setdefault("grid_onset", t)
            dur = float(h.get("offset", t) or t) - t
            h["onset"] = round(new_t, 4)
            h["offset"] = round(new_t + max(dur, 0.01), 4)
            moved += 1
        h["onset_source"] = "drum-stem transient"
    return hits, moved


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


def _active_segments(times, beat_len, gap_beats=8.0):
    """Split sorted onset times into stretches where the part is actually
    playing, breaking wherever it stops for longer than `gap_beats`.

    Per-section groove reads pool every occurrence of a section (Verse 1 and
    Verse 2, which sit a minute apart), and a whole-track read spans drum
    breaks. Measuring hit RATE over first-to-last hit counted all of that
    silence as beats with no hits: a clean half-time kick pooled over two
    verses came out at ~0.33 kicks per beat and was reported as "rate does
    not match a simple pattern", and a four-on-the-floor kick lost its
    beat coverage the same way.
    """
    arr = np.asarray(sorted(float(t) for t in (times if times is not None else [])), dtype=float)
    if arr.size == 0 or not beat_len or beat_len <= 0:
        return []
    gap = float(gap_beats) * float(beat_len)
    segs = []
    start = prev = float(arr[0])
    for t in arr[1:]:
        t = float(t)
        if t - prev > gap:
            segs.append((start, prev))
            start = t
        prev = t
    segs.append((start, prev))
    return segs


def _active_rate_per_beat(times, beat_len, gap_beats=8.0):
    """(hits per beat, active span in beats) over the playing stretches only.

    For a single contiguous stretch this is exactly count / span, the value
    the classifiers always used; each extra stretch removes one count so a
    pooled read is not biased upward by its segment boundaries.
    """
    segs = _active_segments(times, beat_len, gap_beats=gap_beats)
    if not segs:
        return 0.0, 0.0
    span_beats = sum(e - s for s, e in segs) / float(beat_len)
    if span_beats <= 0:
        return 0.0, 0.0
    count = len(times) - (len(segs) - 1)
    return max(0.0, count / span_beats), span_beats


def _kick_beat_coverage(kicks, beat_len, tol_frac=0.10):
    """Fraction of beats in the span that actually carry a kick, plus the
    fraction of kicks that land on a beat.

    "Four on the floor" means a kick on EVERY beat. The grid-lock score alone
    cannot express that: it measures how tightly kick phases cluster, not how
    many beats get hit. A boom-bap or dembow pattern (kick on 1, the 'a' of 1,
    the '&' of 2, the '&' of 3) averages almost exactly one kick per beat and
    clusters into two phase groups, which scores as moderate lock at a
    four-on-the-floor hit rate -- so rate+lock alone labelled several of the
    most common non-four-on-the-floor grooves in popular music as
    four-on-the-floor. Coverage separates them cleanly: a real
    four-on-the-floor covers ~100% of beats, those grooves cover ~25-50%.

    The beat grid's absolute phase is unknown here (only bpm is passed in), so
    the offset that maximises coverage is searched for and used -- consistent
    with this module's existing assumption that we don't know where bar 1 is.
    A consequence worth knowing: a kick landing evenly on every off-beat scores
    as full coverage, because with only a tempo to go on there is no way to
    tell "kick on every beat" from "kick on every off-beat" -- and a listener
    would hear an evenly spaced kick as the pulse either way.

    The tolerance must stay below half a 16th-note (0.125 beat) or a single
    grid point can sit between two adjacent 16th positions and match both,
    which is exactly how a boom-bap pattern scored 0.76 coverage instead of
    its true 0.5.

    Returns (beat_coverage, on_beat_fraction, n_beats).
    """
    if not kicks or not beat_len or beat_len <= 0:
        return 0.0, 0.0, 0
    arr = np.asarray(sorted(float(k) for k in kicks), dtype=float)
    if arr.size < 2:
        return 0.0, 0.0, 0
    t0, t1 = arr[0], arr[-1]
    n_beats = int(round((t1 - t0) / beat_len)) + 1
    if n_beats < 4:
        return 0.0, 0.0, n_beats

    tol = beat_len * float(tol_frac)
    # Only beats inside stretches where the kick is actually playing count as
    # "should carry a kick" -- a drum break or the gap between two pooled
    # verses is not a run of missing four-on-the-floor kicks.
    segs = _active_segments(arr, beat_len)
    if len(segs) > 1:
        seg_lo = np.asarray([a for a, _b in segs], dtype=float) - tol
        seg_hi = np.asarray([b for _a, b in segs], dtype=float) + tol
    else:
        seg_lo = seg_hi = None
    best = (0.0, 0.0)
    # 24 candidate grid phases is finer than the tolerance window, so the true
    # alignment is always within reach of one of them.
    for step in range(24):
        offset = (step / 24.0) * beat_len
        grid = t0 + offset + np.arange(n_beats) * beat_len
        grid = grid[(grid >= t0 - tol) & (grid <= t1 + tol)]
        if seg_lo is not None and grid.size:
            k = np.searchsorted(seg_lo, grid, side="right") - 1
            inside = (k >= 0) & (grid <= seg_hi[np.clip(k, 0, len(seg_hi) - 1)])
            grid = grid[inside]
        if grid.size == 0:
            continue
        idx = np.searchsorted(arr, grid)
        left = np.clip(idx - 1, 0, arr.size - 1)
        right = np.clip(idx, 0, arr.size - 1)
        dist = np.minimum(np.abs(arr[left] - grid), np.abs(arr[right] - grid))
        covered = float(np.count_nonzero(dist <= tol)) / float(grid.size)

        gidx = np.searchsorted(grid, arr)
        gl = np.clip(gidx - 1, 0, grid.size - 1)
        gr = np.clip(gidx, 0, grid.size - 1)
        kdist = np.minimum(np.abs(grid[gl] - arr), np.abs(grid[gr] - arr))
        on_beat = float(np.count_nonzero(kdist <= tol)) / float(arr.size)

        if covered > best[0]:
            best = (covered, on_beat)
    return best[0], best[1], n_beats


def _local_grid_windows(times, beat_len, win_beats=16.0, gap_beats=8.0, min_seg_hits=3):
    """Split hits into local windows (inside the stretches where the part is
    actually playing) for the drift-tolerant grid measures below.

    Yields (window_start, window_end, hits_in_window). A short tail is merged
    into the window before it so no window is too short to judge. Stretches
    of fewer than `min_seg_hits` hits (an isolated accent between long gaps)
    are skipped: a lone hit trivially "covers" a one-point grid.
    """
    arr = np.asarray(sorted(float(t) for t in (times if times is not None else [])), dtype=float)
    if arr.size == 0 or not beat_len or beat_len <= 0:
        return []
    W = float(win_beats) * float(beat_len)
    out = []
    for s, e in _active_segments(arr, beat_len, gap_beats=gap_beats):
        seg_hits = arr[(arr >= s) & (arr <= e)]
        if seg_hits.size < int(min_seg_hits):
            continue
        edges = [s]
        while edges[-1] + W < e:
            edges.append(edges[-1] + W)
        if len(edges) > 1 and (e - edges[-1]) < 0.5 * W:
            edges.pop()
        edges.append(e + 1e-6)
        for a, b in zip(edges[:-1], edges[1:]):
            m = arr[(arr >= a) & (arr < b)]
            out.append((float(a), float(b), m))
    return out


def _fit_window_grid(m, a, b, period, tol, n_off):
    """Best-fitting grid (one point every `period` s) for the hits `m` inside
    the window [a, b).

    Grid points are counted across the WHOLE window, not just between its
    first and last hit, so a window cannot score well by trimming away the
    stretch it fails to cover. Among offsets that tie on coverage the centre
    of the tied run is used, and the grid is then shifted onto the mean
    position of the hits it caught -- otherwise the first tied offset (the
    early edge of the tolerance) was used and every phase measured from it
    came out ~0.1 beat late.

    Returns (grid, covered, on_grid_mask) or None.
    """
    if m.size == 0 or period <= 0 or b <= a:
        return None
    n_pts = int(np.ceil((b - a) / period)) + 1
    scores = []
    for i in range(n_off):
        off = (i / float(n_off)) * period
        grid = a + off + np.arange(n_pts) * period
        grid = grid[(grid >= a) & (grid < b)]
        if grid.size == 0:
            scores.append((-1, -1))
            continue
        idx = np.searchsorted(m, grid)
        lo = np.clip(idx - 1, 0, m.size - 1)
        hi = np.clip(idx, 0, m.size - 1)
        dist = np.minimum(np.abs(m[lo] - grid), np.abs(m[hi] - grid))
        gidx = np.searchsorted(grid, m)
        gl = np.clip(gidx - 1, 0, grid.size - 1)
        gr = np.clip(gidx, 0, grid.size - 1)
        hd = np.minimum(np.abs(grid[gl] - m), np.abs(grid[gr] - m))
        scores.append((int(np.count_nonzero(dist <= tol)), int(np.count_nonzero(hd <= tol))))
    best_key = max(scores)
    if best_key[0] <= 0:
        return None
    tied = np.asarray([i for i, k in enumerate(scores) if k == best_key], dtype=float)
    ang = 2.0 * np.pi * tied / float(n_off)
    centre = (float(np.angle(np.mean(np.exp(1j * ang)))) % (2.0 * np.pi)) / (2.0 * np.pi)

    def _evaluate(off):
        grid = a + off + np.arange(-1, n_pts + 1) * period
        # Shift onto the hits actually caught.
        gidx = np.searchsorted(grid, m)
        gl = np.clip(gidx - 1, 0, grid.size - 1)
        gr = np.clip(gidx, 0, grid.size - 1)
        near = np.where(np.abs(grid[gl] - m) <= np.abs(grid[gr] - m), grid[gl], grid[gr])
        resid = m - near
        on = np.abs(resid) <= tol
        if np.any(on):
            grid = grid + float(np.mean(resid[on]))
        grid_in = grid[(grid >= a - tol) & (grid < b)]
        if grid_in.size == 0:
            return None
        idx = np.searchsorted(m, grid_in)
        lo = np.clip(idx - 1, 0, m.size - 1)
        hi = np.clip(idx, 0, m.size - 1)
        dist = np.minimum(np.abs(m[lo] - grid_in), np.abs(m[hi] - grid_in))
        covered = int(np.count_nonzero(dist <= tol))
        gidx = np.searchsorted(grid_in, m)
        gl = np.clip(gidx - 1, 0, grid_in.size - 1)
        gr = np.clip(gidx, 0, grid_in.size - 1)
        hd = np.minimum(np.abs(grid_in[gl] - m), np.abs(grid_in[gr] - m))
        return grid_in, covered, hd <= tol

    res = _evaluate(centre * period)
    if res is None or res[1] < best_key[0]:
        # Tied offsets split into separate runs (their centre is not itself
        # a good fit): fall back to the first tied offset, still shifted
        # onto the caught hits.
        alt = _evaluate((float(tied[0]) / float(n_off)) * period)
        if alt is not None and (res is None or alt[1] > res[1]):
            res = alt
    return res


def _local_cell_coverage(times, beat_len, period_beats=1.0, win_beats=16.0,
                         tol_frac=0.12, steps_per_beat=24, collect_on_grid=None):
    """Coverage of a grid with one point every `period_beats` beats, with the
    grid's phase fitted separately in each local window.

    _kick_beat_coverage fits ONE phase to the whole stretch at a constant
    tempo. Real recordings -- an unclicked band, and in particular AI-generated
    audio, whose timing wanders by tens of milliseconds and can jump between
    generation chunks -- drift off any single grid within half a minute. A kick
    landing on every beat to within 10 ms (an AI-generated test track: kick spacing 0.500 s,
    spread 0.01) then scored ~0.69 coverage over the track and only ~0.53 over
    its pooled verses, and was reported as "syncopated, NOT four-on-the-floor".
    Fitting the phase per window of `win_beats` beats removes the drift while
    keeping the test itself: a syncopated one-kick-per-beat pattern (boom-bap,
    dembow) still covers at most ~half of the beats in every window.

    Returns (coverage, on_grid_fraction, n_grid_points). `coverage` is the
    share of grid points that carry a hit; `on_grid_fraction` the share of ALL
    the hits passed in that sit on the fitted grids. When `collect_on_grid` is
    a list, the on-grid hit times are appended to it.
    """
    try:
        beat_len = float(beat_len)
        period = float(period_beats) * beat_len
    except (TypeError, ValueError):
        return 0.0, 0.0, 0
    if not times or beat_len <= 0 or period <= 0:
        return 0.0, 0.0, 0
    tol = float(tol_frac) * beat_len
    gap_beats = max(8.0, 2.5 * float(period_beats))
    n_off = max(8, int(round(steps_per_beat * float(period_beats))))
    covered_tot = 0
    grid_tot = 0
    on_tot = 0
    for a, b, m in _local_grid_windows(times, beat_len, win_beats=max(win_beats, 4.0 * period_beats),
                                       gap_beats=gap_beats):
        fit = _fit_window_grid(m, a, b, period, tol, n_off)
        if fit is None:
            # Hits but no grid point caught anything: the grid is uncovered.
            grid_tot += max(1, int(np.floor((b - a) / period)))
            continue
        grid, cov, on_mask = fit
        covered_tot += cov
        grid_tot += int(grid.size)
        on_tot += int(np.count_nonzero(on_mask))
        if collect_on_grid is not None:
            collect_on_grid.extend(float(t) for t in m[on_mask])
    if grid_tot == 0:
        return 0.0, 0.0, 0
    return covered_tot / float(grid_tot), on_tot / float(max(1, len(times))), grid_tot


def _local_circular_lock(times, beat_len, divisor, win_beats=16.0, min_hits=4):
    """Grid-lock (circular phase concentration) measured window by window.

    The same drift that defeats a single whole-track phase also lowers the
    whole-track lock: phases that are tight locally smear into a broad
    distribution once the grid has wandered. Each window's resultant is
    debiased for its hit count (a handful of random hits always clusters a
    little), then the windows are pooled weighted by hits.
    """
    cell = float(beat_len) * float(divisor)
    if not times or cell <= 0:
        return 0.0
    num = 0.0
    den = 0.0
    for _a, _b, m in _local_grid_windows(times, beat_len, win_beats=win_beats):
        n = int(m.size)
        if n < int(min_hits):
            continue
        ang = 2.0 * np.pi * (m % cell) / cell
        r2 = float(np.abs(np.mean(np.exp(1j * ang)))) ** 2
        r2u = max(0.0, (n * r2 - 1.0) / (n - 1.0))
        num += n * r2u
        den += n
    if den <= 0:
        return 0.0
    return float(np.sqrt(num / den))


def _local_relative_phase(ref_times, other_times, beat_len, period_beats, win_beats=16.0, tol_frac=0.12):
    """Where `other_times` sit inside a `period_beats` cycle anchored on the
    locally fitted grid of `ref_times` (e.g. kicks relative to the snare).

    Returns (mean_position_in_beats, concentration) -- the circular mean of
    the other part's position within the cycle (0 = with the reference,
    period/2 = opposite it) and how tightly it clusters there (0-1), or
    (None, 0.0) when there is not enough to say.
    """
    period = float(period_beats) * float(beat_len)
    if not ref_times or not other_times or period <= 0:
        return None, 0.0
    oth = np.asarray(sorted(float(t) for t in other_times), dtype=float)
    tol = float(tol_frac) * float(beat_len)
    n_off = max(8, int(round(24 * float(period_beats))))
    zs = []
    for a, b, m in _local_grid_windows(ref_times, beat_len, win_beats=max(win_beats, 4.0 * period_beats),
                                       gap_beats=max(8.0, 2.5 * float(period_beats))):
        fit = _fit_window_grid(m, a, b, period, tol, n_off)
        if fit is None:
            continue
        grid = fit[0]
        o = oth[(oth >= a) & (oth < b)]
        if o.size == 0:
            continue
        anchor = float(grid[0])
        ang = 2.0 * np.pi * ((o - anchor) % period) / period
        zs.extend(np.exp(1j * ang).tolist())
    if len(zs) < 4:
        return None, 0.0
    z = np.mean(np.asarray(zs))
    pos = (float(np.angle(z)) % (2.0 * np.pi)) / (2.0 * np.pi) * float(period_beats)
    return pos, float(np.abs(z))


def _local_offset_hit_rate(ref_times, other_times, beat_len, period_beats, offset_beats,
                           win_beats=32.0, tol_frac=0.12):
    """Share of the reference part's locally fitted grid points (one every
    `period_beats`) that have an `other_times` hit `offset_beats` later (a
    negative offset looks earlier). E.g. with the snare as reference and a
    4-beat period: is there a kick two beats away from the snare, bar after
    bar? Returns (rate, n_points) or (0.0, 0).
    """
    period = float(period_beats) * float(beat_len)
    if not ref_times or not other_times or period <= 0:
        return 0.0, 0
    oth = np.asarray(sorted(float(t) for t in other_times), dtype=float)
    tol = float(tol_frac) * float(beat_len)
    n_off = max(8, int(round(24 * float(period_beats))))
    hit = 0
    tot = 0
    for a, b, m in _local_grid_windows(ref_times, beat_len, win_beats=max(win_beats, 4.0 * period_beats),
                                       gap_beats=max(8.0, 2.5 * float(period_beats))):
        fit = _fit_window_grid(m, a, b, period, tol, n_off)
        if fit is None:
            continue
        targets = fit[0] + float(offset_beats) * float(beat_len)
        idx = np.searchsorted(oth, targets)
        lo = np.clip(idx - 1, 0, oth.size - 1)
        hi = np.clip(idx, 0, oth.size - 1)
        dist = np.minimum(np.abs(oth[lo] - targets), np.abs(oth[hi] - targets))
        hit += int(np.count_nonzero(dist <= tol))
        tot += int(targets.size)
    if tot == 0:
        return 0.0, 0
    return hit / float(tot), tot


def _refine_bpm_from_onsets(onsets, bpm, max_dev=0.03, min_onsets=12):
    """Return the tempo that the drum onsets themselves sit on, near `bpm`.

    The reported BPM is a rounded integer (and the reconciliation that picks
    it can land a percent or so off), while grid-lock is measured as the
    circular spread of onset phases across the WHOLE track or section. A
    0.3% error drifts the phase by more than a beat over three minutes, so a
    perfectly quantised four-on-the-floor kick scored as "syncopated, NOT
    four-on-the-floor" and a clean backbeat as merely "moderate". Fitting the
    16th-note cell length to the onsets removes that drift before any grid
    judgement is made. The nominal tempo is kept unless the fit is clearly
    better, and the search never leaves +/- max_dev, so this can only correct
    a rounding/estimation error -- never change the pulse level.
    """
    try:
        bpm = float(bpm)
    except (TypeError, ValueError):
        return bpm
    if not bpm or bpm <= 0:
        return bpm
    arr = np.asarray(sorted(float(t) for t in (onsets if onsets is not None else [])), dtype=float)
    if arr.size < int(min_onsets):
        return bpm
    cell0 = 60.0 / bpm / 4.0
    rel = arr - float(np.median(arr))
    span = float(arr[-1] - arr[0])
    if span <= cell0 * 8:
        return bpm

    def _r(cells):
        cells = np.asarray(cells, dtype=float).reshape(-1, 1)
        ang = 2.0 * np.pi * (rel.reshape(1, -1) / cells)
        return np.hypot(np.mean(np.sin(ang), axis=1), np.mean(np.cos(ang), axis=1))

    n_cells = span / cell0
    # Step fine enough that the main lobe of the fit (width ~1/n_cells) is
    # always sampled, however long the stretch of music is.
    step = min(0.00025, 0.25 / max(1.0, n_cells))
    n_steps = int(min(6001, max(61, np.ceil(2 * max_dev / step) + 1)))
    scales = np.linspace(1.0 - max_dev, 1.0 + max_dev, n_steps)
    try:
        r = _r(cell0 * scales)
    except MemoryError:
        return bpm
    i = int(np.argmax(r))
    best_scale, best_r = float(scales[i]), float(r[i])
    fine = np.linspace(best_scale - step, best_scale + step, 41)
    rf = _r(cell0 * fine)
    j = int(np.argmax(rf))
    if float(rf[j]) > best_r:
        best_scale, best_r = float(fine[j]), float(rf[j])
    r0 = float(_r([cell0])[0])
    if best_r < 0.3 or best_r < r0 + 0.03:
        return bpm
    return bpm / best_scale


def _cell_resultant(times, cell):
    """Circular resultant length of onset phases on a grid of `cell` seconds."""
    arr = np.asarray(times, dtype=float)
    if arr.size == 0 or cell <= 0:
        return 0.0
    ang = 2.0 * np.pi * (arr - float(np.median(arr))) / float(cell)
    return float(np.hypot(np.mean(np.sin(ang)), np.mean(np.cos(ang))))


def _tempo_warp_onsets(onsets, bpm, win_beats=32.0, hop_beats=8.0, max_dev=0.025):
    """Map onset times onto a steady-tempo timeline when the drummer drifts.

    Returns a function t -> warped t, or None when no warp is warranted.

    An unclicked live take wanders a percent or two around its average tempo.
    Grid-lock is measured as phase against ONE tempo across the whole track,
    so that wander alone made a live four-on-the-floor read as "syncopated,
    NOT four-on-the-floor" and a live backbeat as "loosely on a grid". Here
    the local tempo is fitted in overlapping windows (tempo only -- no phase,
    so phase stays continuous) and integrated into a beat-position clock.

    Used only when it clearly tightens the whole-track fit, so clicked and
    programmed tracks, whose tempo is already constant, are left exactly as
    they were.
    """
    try:
        bpm = float(bpm)
    except (TypeError, ValueError):
        return None
    arr = np.asarray(sorted(float(t) for t in (onsets if onsets is not None else [])), dtype=float)
    if not bpm or bpm <= 0 or arr.size < 48:
        return None
    beat = 60.0 / bpm
    cell = beat / 4.0
    t0, t1 = float(arr[0]), float(arr[-1])
    if (t1 - t0) < beat * win_beats * 2.0:
        return None

    win = beat * win_beats
    centers, scales = [], []
    c = t0 + win / 2.0
    scale_grid = np.linspace(1.0 - max_dev, 1.0 + max_dev, 201)
    while c <= t1 - win / 2.0 + 1e-9:
        seg = arr[(arr >= c - win / 2.0) & (arr <= c + win / 2.0)]
        if seg.size >= 16:
            rel = (seg - c).reshape(1, -1)
            cells = (cell * scale_grid).reshape(-1, 1)
            ang = 2.0 * np.pi * rel / cells
            r = np.hypot(np.mean(np.sin(ang), axis=1), np.mean(np.cos(ang), axis=1))
            i = int(np.argmax(r))
            if float(r[i]) >= 0.35:
                centers.append(c)
                scales.append(float(scale_grid[i]))
        c += beat * hop_beats
    if len(scales) < 3:
        return None
    scales = np.asarray(scales, dtype=float)
    # Median-of-3 smoothing so one noisy window cannot kink the clock.
    sm = scales.copy()
    for i in range(1, len(scales) - 1):
        sm[i] = float(np.median(scales[i - 1:i + 2]))
    if float(np.max(np.abs(sm - 1.0))) < 0.002:
        return None  # already steady; nothing to correct
    centers = np.asarray(centers, dtype=float)

    grid_t = np.linspace(t0 - beat, t1 + beat, int(max(64, (t1 - t0) / cell)) + 1)
    local_beat = beat * np.interp(grid_t, centers, sm)
    rate = 1.0 / local_beat                     # beats per second
    dt = np.diff(grid_t)
    beats_pos = np.concatenate([[0.0], np.cumsum(0.5 * (rate[1:] + rate[:-1]) * dt)])

    def warp(t):
        return float(grid_t[0] + beat * np.interp(float(t), grid_t, beats_pos))

    warped = np.asarray([warp(t) for t in arr])
    r_raw = _cell_resultant(arr, cell)
    r_warp = _cell_resultant(warped, cell)
    if r_warp < r_raw + 0.05:
        return None
    return warp


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

    Rate and lock are necessary but NOT sufficient for a four-on-the-floor
    claim: both are satisfied by common syncopated grooves that put roughly
    one kick per beat somewhere other than on the beats. Beat COVERAGE (are
    the beats themselves actually being hit?) is therefore required as well
    before the phrase is used — see _kick_beat_coverage.

    Returns (label, detail) or (None, None) if there isn't enough evidence
    (no bpm, or too few kicks) to say anything reliable.
    """
    if not bpm or bpm <= 0 or not kicks or len(kicks) < 6:
        return None, None

    kicks_raw = sorted(float(k) for k in kicks)
    beat_len = 60.0 / bpm
    kicks_per_beat, span_beats = _active_rate_per_beat(kicks_raw, beat_len)
    if span_beats < 3:
        return None, None

    # Snap then score — use the better of raw vs snapped lock so free-time
    # playing is not artificially tightened.
    kicks_snap = _snap_onsets_to_subdivision(kicks_raw, beat_len, subdivision=0.25, max_snap_s=0.045)

    def _best_lock(divisor):
        raw_t = _trim_phase_outliers(kicks_raw, beat_len, divisor)
        snap_t = _trim_phase_outliers(kicks_snap, beat_len, divisor)
        return max(
            _circular_lock(raw_t, beat_len, divisor),
            _circular_lock(snap_t, beat_len, divisor),
            # Window-by-window lock, so timing drift over the track (live
            # playing, AI-generated audio) is not mistaken for looseness.
            _local_circular_lock(kicks_raw, beat_len, divisor),
        )

    lock_quarter = _best_lock(1.0)   # every beat
    lock_half = _best_lock(2.0)      # every other beat
    lock_eighth = _best_lock(0.5)    # two per beat

    coverage, on_beat_frac, _n_beats = _kick_beat_coverage(kicks_raw, beat_len)
    # The same coverage test with its grid phase fitted locally (see
    # _local_cell_coverage): drift moved a perfectly regular kick off the
    # single whole-track grid and it was reported as NOT four-on-the-floor.
    _lcov, _lon, _ln = _local_cell_coverage(kicks_raw, beat_len, 1.0)
    if _ln and _lcov > coverage:
        coverage, on_beat_frac = _lcov, _lon

    # Tiered thresholds (was a single hard 0.75 → everything else 'irregular').
    STRONG = 0.68
    MODERATE = 0.52
    WEAK = 0.40
    # A kick on every beat is what the phrase means, so demand most beats be hit.
    FOTF_COVERAGE_STRONG = 0.85
    FOTF_COVERAGE_MODERATE = 0.70

    def _rate_label(strong_lab, moderate_lab, lock_val, kpb_note):
        if lock_val >= STRONG:
            return strong_lab, f"~{kicks_per_beat:.2f} kicks/beat, grid-lock {lock_val:.2f}"
        if lock_val >= MODERATE:
            return moderate_lab, f"~{kicks_per_beat:.2f} kicks/beat, grid-lock {lock_val:.2f} (moderate — quantized/programmed kits often land here)"
        return None, None

    if 0.8 <= kicks_per_beat <= 1.25:
        cov_note = f", beat coverage {coverage:.2f}"
        if lock_quarter >= STRONG and coverage >= FOTF_COVERAGE_STRONG:
            return (
                "four-on-the-floor (kick locked to every beat)",
                f"~{kicks_per_beat:.2f} kicks/beat, grid-lock {lock_quarter:.2f}{cov_note}",
            )
        if lock_quarter >= MODERATE and coverage >= FOTF_COVERAGE_MODERATE:
            return (
                "mostly four-on-the-floor / on-beat kick (moderate grid-lock; treat as a programmed on-beat pattern, not irregular)",
                f"~{kicks_per_beat:.2f} kicks/beat, grid-lock {lock_quarter:.2f}{cov_note} "
                "(moderate — quantized/programmed kits often land here)",
            )
        if coverage < FOTF_COVERAGE_MODERATE:
            # Roughly one kick per beat on average, but the beats themselves are
            # not being hit — a syncopated groove (boom-bap, dembow, funk,
            # shuffle) rather than four-on-the-floor. Say so explicitly, because
            # this is precisely the case that used to be mislabelled.
            return (
                "syncopated kick at roughly one hit per beat — NOT four-on-the-floor "
                "(the kicks fall between the beats as often as on them)",
                f"~{kicks_per_beat:.2f} kicks/beat, grid-lock {lock_quarter:.2f}{cov_note}, "
                f"only {on_beat_frac:.2f} of kicks land on a beat — do not call this "
                "four-on-the-floor",
            )
    if 0.35 <= kicks_per_beat <= 0.7:
        # Worded as what it is -- a kick on every other beat. Calling it a
        # "half-time kick" invited the writer to call an ordinary rock beat
        # (kick 1 & 3, snare 2 & 4) a half-time feel; half-time is a property
        # of the SNARE (on 3 only), which the snare analysis reports itself.
        lab, det = _rate_label(
            "kick on every other beat (e.g. 1 & 3)",
            "kick mostly on every other beat (moderate grid-lock; e.g. roughly 1 & 3 — not irregular)",
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
        "kick pattern with moderate grid-lock (rate does not match a simple on-beat / every-other-beat / eighth-note pattern; prefer texture over 'irregular')",
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
    snares_per_beat, span_beats = _active_rate_per_beat(snares_raw, beat_len)
    if span_beats < 3:
        return None, None

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

    # ---- Coverage-led read -------------------------------------------------
    # Which grid does the snare actually fill -- every 8th, every beat, every
    # other beat, or once every four beats? Measured as the share of grid
    # positions that carry a snare, with the grid's phase fitted window by
    # window so tempo drift and small timing jumps cannot hide a steady
    # pattern (see _local_cell_coverage). This is checked before the raw
    # hit-rate tests below because the rate counts fills, flams and ghost
    # notes too: a half-time rock snare (once per bar, on 3) plus its fills
    # averages ~0.45 snares per beat -- right in the backbeat band -- and was
    # reported as "rate does not match simple backbeat" or "doubling the kick",
    # while no half-time class existed at all.
    COV = 0.65
    # Most of the snare hits must belong to the pattern being named, so a
    # one-bar roll or a handful of isolated accents cannot carry a label.
    ON_MIN = 0.40
    cov_e, on_e, n_e = _local_cell_coverage(snares_raw, beat_len, 0.5)
    cov_1, on_1, n_1 = _local_cell_coverage(snares_raw, beat_len, 1.0)
    main_2 = []
    cov_2, on_2, n_2 = _local_cell_coverage(snares_raw, beat_len, 2.0, collect_on_grid=main_2)
    cov_4, on_4, n_4 = _local_cell_coverage(snares_raw, beat_len, 4.0, win_beats=32.0)
    k_list = sorted(float(k) for k in kicks) if kicks else []
    kick_every_beat = False
    if len(k_list) >= 6:
        try:
            _k_rate, _k_span = _active_rate_per_beat(k_list, beat_len)
            _k_cov = max(
                _local_cell_coverage(k_list, beat_len, 1.0)[0],
                _kick_beat_coverage(k_list, beat_len)[0],
            )
            kick_every_beat = 0.8 <= _k_rate <= 1.25 and _k_cov >= 0.70
        except Exception:
            kick_every_beat = False

    if n_e >= 8 and cov_e >= COV and on_e >= ON_MIN and snares_per_beat >= 1.5:
        return (
            "snare on eighth-note subdivisions (busy/rolled feel)",
            f"~{snares_per_beat:.2f} snares/beat, snare on {cov_e:.0%} of eighth-note positions",
        )
    if n_1 >= 8 and cov_1 >= COV and on_1 >= ON_MIN and snares_per_beat >= 0.8:
        return (
            "snare on every beat (dense, not a classic backbeat)",
            f"~{snares_per_beat:.2f} snares/beat, snare on {cov_1:.0%} of beats",
        )
    if n_2 >= 6 and cov_2 >= COV and on_2 >= ON_MIN:
        detail = (
            f"~{snares_per_beat:.2f} snares/beat, snare on {cov_2:.0%} of every-other-beat "
            f"positions, grid-lock {max(lock_half, _local_circular_lock(snares_raw, beat_len, 2.0)):.2f}"
        )
        if kick_every_beat:
            # Every-other-beat periodicity alone is not a backbeat: a dembow
            # or breakbeat snare also repeats every two beats but lands
            # BETWEEN the kicks. Check where the main (on-grid) snares sit
            # against the kick's beats; rolls and fills are left out of that,
            # and only a clear off-beat placement gets the syncopated label.
            s_pos, s_conc = _local_relative_phase(k_list, main_2 or snares_raw, beat_len, 1.0)
            if s_pos is not None and s_conc >= 0.5 and min(s_pos, 1.0 - s_pos) >= 0.25:
                return (
                    "syncopated snare/clap repeating every two beats but landing between the "
                    "kicks, over a kick on every beat (e.g. a dembow-style pattern — not a 2-and-4 backbeat)",
                    detail + ", kick on every beat, snare off the beat",
                )
            return (
                "snare/clap on alternate beats over a kick on every beat "
                "(the usual four-on-the-floor backbeat pairing)",
                detail + ", kick on every beat",
            )
        pos, conc = _local_relative_phase(snares_raw, k_list, beat_len, 2.0)
        if pos is not None and conc >= 0.30:
            phase_diff = min(pos, 2.0 - pos)  # 0 = with the snare, 1 = opposite
            detail += f", kick/snare phase offset {phase_diff:.2f} (1.0 = opposite the kick)"
            if phase_diff >= 0.7:
                return (
                    "classic backbeat (snare falls opposite the kick, e.g. 2 & 4 against kick on 1 & 3)"
                    if cov_2 >= 0.80 else
                    "mostly classic backbeat (snare on most 2s and 4s, opposite the kick — not irregular)",
                    detail,
                )
            if phase_diff <= 0.3:
                return "snare doubling the kick's grid position (not a classic backbeat)", detail
            return (
                "snare on a backbeat-rate grid with imperfect opposite-phase lock (still a backbeat-family pattern, not wandering)",
                detail,
            )
        return (
            "backbeat-rate snare (on every other beat; the kick moves around too much to confirm it sits opposite)",
            detail,
        )
    if n_4 >= 4 and cov_4 >= COV and cov_2 < COV and on_4 >= ON_MIN:
        half_bpm = float(bpm) / 2.0
        detail = (
            f"~{snares_per_beat:.2f} snares/beat including fills, main snare on {cov_4:.0%} of "
            f"four-beat cycles but only {cov_2:.0%} of every-other-beat positions"
        )
        if len(k_list) >= 6:
            # Half-time puts the kick on 1, two beats from the snare on 3. A
            # 2-and-4 backbeat whose every other snare the transcription
            # missed also leaves one snare per bar, but its kicks sit one
            # beat either side of that snare (on 1 and 3) and the position
            # opposite it is an empty snare slot.
            opp, _n_opp = _local_offset_hit_rate(snares_raw, k_list, beat_len, 4.0, 2.0)
            before, _n_b = _local_offset_hit_rate(snares_raw, k_list, beat_len, 4.0, -1.0)
            after, _n_a = _local_offset_hit_rate(snares_raw, k_list, beat_len, 4.0, 1.0)
            if opp >= 0.45:
                detail += f", kick two beats from the snare (beat 1) in {opp:.0%} of bars"
            elif opp < 0.25 and before >= 0.5 and after >= 0.5:
                return (
                    "snare once every four beats (one main snare per bar; the kicks sit one beat "
                    "either side of it rather than opposite, so this may be a 2-and-4 backbeat with "
                    "missed snares — not confirmed as a half-time feel)",
                    detail + f", kick one beat before/after the snare in {before:.0%}/{after:.0%} of bars, "
                    f"opposite it in only {opp:.0%}",
                )
        return (
            "half-time backbeat (snare once per bar, on beat 3 — a half-time feel at this tempo; "
            f"counted at half speed, ~{half_bpm:.0f} BPM, it is an ordinary 2-and-4 backbeat)",
            detail,
        )

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
        # A kick on EVERY beat has no single phase on a 2-beat cell, so the
        # opposite-phase test above cannot run -- but that is the kick being
        # locked, not the kick being absent. Say so rather than claiming there
        # is no locked kick: snare/clap on alternate beats over an every-beat
        # kick is the standard four-on-the-floor backbeat.
        if kicks and len(kicks) >= 6:
            try:
                kicks_q = _snap_onsets_to_subdivision(
                    sorted(float(k) for k in kicks), beat_len, subdivision=0.25, max_snap_s=0.045
                )
                klock_q, _kp = _lock_and_phase(1.0, _trim_phase_outliers(kicks_q, beat_len, 1.0))
                k_per_beat, _k_span = _active_rate_per_beat(sorted(kicks_q), beat_len)
                # Same beat-coverage test the kick classifier uses before it
                # will say four-on-the-floor. Without it a syncopated kick at
                # about one hit per beat (boom-bap, dembow) was described here
                # as "a kick on every beat" while the kick analysis two lines
                # up said "NOT four-on-the-floor" for the same hits.
                k_cov, _k_on, _k_n = _kick_beat_coverage(sorted(kicks_q), beat_len)
                k_cov = max(k_cov, _local_cell_coverage(sorted(kicks_q), beat_len, 1.0)[0])
            except Exception:
                klock_q, k_per_beat, k_cov = 0.0, 0.0, 0.0
            if klock_q >= MODERATE and 0.8 <= k_per_beat <= 1.25 and k_cov >= 0.70:
                return (
                    "snare/clap on alternate beats over a kick on every beat "
                    "(the usual four-on-the-floor backbeat pairing)",
                    detail + f", kick locked to every beat (grid-lock {klock_q:.2f})",
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
    # Drop the gaps where the part stops (breakdowns, section changes) before
    # measuring the subdivision. A handful of multi-second gaps inflated the
    # mean enough to push a real 8th-note stream out of its tempo band, and
    # each gap also counted as a spurious "long" note in the alternation test.
    med_ioi = float(np.median(iois))
    if med_ioi > 0:
        core = iois[iois <= 4.0 * med_ioi]
        if core.size >= max(3, min_hits - 1):
            iois = core
    mean_ioi = float(np.mean(iois))
    if mean_ioi <= 1e-6:
        return None, None, None

    level = ""
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
        if 0.55 * expected_8th <= mean_ioi <= 1.8 * expected_8th:
            level = ""
        elif 0.55 * (expected_8th / 2.0) <= mean_ioi < 0.55 * expected_8th:
            # A 16th-note stream (busy hats in pop, hip-hop, funk). This used to
            # be rejected outright, so the most common modern hat pattern never
            # got a straight/swung read at all -- and swung 16ths, the defining
            # feel of a lot of hip-hop and neo-soul, could not be reported.
            level = "16th-note "
        else:
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

    sub_note = f" ({level.strip()} subdivision)" if level else ""
    if alt_frac < 0.55:
        return "straight", ratio, f"long/short IOIs don't alternate consistently ({alt_frac:.2f}); reads as straight, not swung{sub_note}"

    if ratio < 1.15:
        label = "straight"
    elif ratio < 1.5:
        label = f"light {level}swing/shuffle"
    elif ratio < 2.2:
        label = f"moderate-to-triplet {level}swing"
    else:
        label = f"heavy {level}shuffle / dotted-note feel"

    return label, ratio, f"long:short IOI ratio ~{ratio:.2f}, alternation consistency {alt_frac:.2f}{sub_note}"


def _drum_band_limit(seg, sr, lo, hi):
    """Zero-phase FFT band-limit. numpy-only, so this adds no dependency.

    Essential rather than cosmetic: a drums stem is a MIX of the whole kit, so
    the window around a kick also contains whatever snare, hat or cymbal
    happens to land nearby. Measured broadband, a kick's 'decay' is frequently
    just the snare on the next beat -- in testing, the loudest point inside the
    first kick's own window was the snare 502 ms later, so every kick decay and
    every consistency score was actually describing a different drum. Measuring
    each drum inside its own frequency band removes that.
    """
    seg = np.asarray(seg, dtype=np.float64)
    if seg.size < 32:
        return seg
    spec = np.fft.rfft(seg)
    freqs = np.fft.rfftfreq(seg.size, 1.0 / sr)
    mask = np.ones(spec.shape, dtype=bool)
    if lo is not None:
        mask &= (freqs >= lo)
    if hi is not None:
        mask &= (freqs <= hi)
    spec = spec * mask
    return np.fft.irfft(spec, n=seg.size)


# Frequency ranges each drum is measured in. Chosen to keep the neighbours out:
# the kick band stops below a typical snare fundamental (~180-200 Hz), and the
# hat band starts above where snare body lives.
_DRUM_ANALYSIS_BANDS = {
    "kick": (30.0, 150.0),
    "snare": (150.0, 6000.0),
    "hihat": (6000.0, 16000.0),
    "tom": (60.0, 1200.0),
    "cymbal": (3000.0, 16000.0),
}


# Which drums put energy into each drum's own analysis band (see
# _DRUM_ANALYSIS_BANDS). A tone/decay window only has to stop at these.
_DRUM_WINDOW_CONFLICTS = {
    "kick": ("kick", "snare", "tom"),
    "snare": ("snare", "kick", "tom", "cymbal"),
    "hihat": ("hihat", "cymbal", "snare"),
}


def _drum_hit_windows(y, sr, onsets, pre=0.005, post=0.25, all_onsets=None,
                      band=None, min_gap=0.03):
    """Slice the drums stem around each transcribed hit.

    `all_onsets`: every hit in the stem regardless of type. Each window is
    truncated before the next hit of ANY drum so a neighbouring hit cannot be
    mistaken for the tail of this one.
    `band`: (lo, hi) to band-limit the slice to this drum's own range.
    """
    out = []
    n = len(y)
    nxt = np.asarray(sorted(all_onsets), dtype=float) if all_onsets else None
    for t in onsets:
        t = float(t)
        a = int(max(0, (t - pre) * sr))
        limit = t + post
        if nxt is not None and nxt.size:
            i = int(np.searchsorted(nxt, t + min_gap))
            if i < nxt.size:
                limit = min(limit, float(nxt[i]))
        b = int(min(n, limit * sr))
        if b - a > int(sr * 0.02):
            seg = y[a:b]
            if band is not None:
                try:
                    seg = _drum_band_limit(seg, sr, band[0], band[1])
                except Exception:
                    pass
            out.append(seg)
    return out


def _drum_decay_ms(seg, sr, drop_db=20.0):
    """Time from a hit's peak until it falls drop_db below that peak.
    Separates a tight gated snare or short acoustic kick from a long 808 tail."""
    if seg is None or len(seg) < 16:
        return None
    seg = np.asarray(seg, dtype=np.float64)
    # Amplitude envelope. This used to be |x| smoothed over 2 ms, which on a
    # low-pitched hit is not an envelope at all: a 60-80 Hz kick has a zero
    # crossing every ~7 ms, the 2 ms average dips deep at each one, and the
    # "-20 dB point" was simply the first dip once the hit had decayed a few
    # dB -- a kick with a 276 ms decay measured ~90 ms. The analytic-signal
    # magnitude follows the true envelope regardless of pitch.
    try:
        from scipy.signal import hilbert as _hilbert
        env = np.abs(_hilbert(seg))
    except Exception:
        env = np.abs(seg)
    win = max(1, int(sr * 0.003))
    if win > 1:
        env = np.convolve(env, np.ones(win) / win, mode="same")
    pk = int(np.argmax(env))
    peak = float(env[pk])
    if peak <= 1e-8:
        return None
    thresh = peak * (10.0 ** (-drop_db / 20.0))
    tail = env[pk:]
    below = np.flatnonzero(tail <= thresh)
    if below.size:
        return float(below[0]) / sr * 1000.0
    # The window ended before the hit fell drop_db. Discarding those hits (the
    # old behaviour) kept only the fastest-decaying ones and biased the median
    # short. Extrapolate the log-linear decay instead, when there is enough
    # tail to fit and it is clearly falling.
    if tail.size < int(sr * 0.03):
        return None
    t = np.arange(tail.size) / float(sr)
    db = 20.0 * np.log10(np.maximum(tail, peak * 1e-6) / peak)
    try:
        slope = float(np.polyfit(t, db, 1)[0])          # dB per second
    except Exception:
        return None
    if slope >= -3.0 or float(db[-1]) > -3.0:
        return None
    return float(min(2000.0, (-drop_db / slope) * 1000.0))


def _drum_centroid_hz(seg, sr):
    """Spectral centroid of one hit. n_fft is sized to the slice: hit windows
    are routinely shorter than librosa's 2048 default, which otherwise emits a
    warning per hit and floods the console during a scan."""
    if seg is None or len(seg) < 64:
        return None
    try:
        seg = np.asarray(seg, dtype=np.float32)
        n_fft = int(min(2048, 2 ** int(np.floor(np.log2(max(64, len(seg)))))))
        return float(np.mean(librosa.feature.spectral_centroid(
            y=seg, sr=sr, n_fft=n_fft, hop_length=max(1, n_fft // 4))))
    except Exception:
        return None


def _drum_fundamental_hz(seg, sr, fmax=250.0):
    """Dominant low-frequency partial of a kick hit -- the difference between
    a sub-heavy 808 (~40-55 Hz) and a punchy acoustic kick (~60-100 Hz)."""
    if seg is None or len(seg) < 256:
        return None
    try:
        n_fft = 4096
        seg = np.asarray(seg, dtype=np.float32)
        if len(seg) < n_fft:
            seg = np.pad(seg, (0, n_fft - len(seg)))
        spec = np.abs(np.fft.rfft(seg[:n_fft] * np.hanning(n_fft)))
        freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
        mask = (freqs >= 25.0) & (freqs <= fmax)
        if not np.any(mask) or float(np.max(spec[mask])) <= 0:
            return None
        return float(freqs[mask][int(np.argmax(spec[mask]))])
    except Exception:
        return None


def _isolated_onsets(onsets, conflicting_onsets, guard=0.045):
    """Hits with no CONFLICTING drum landing within `guard` seconds either side.

    Consistency is only meaningful on hits that aren't overlapping something
    else in the same frequency range: a kick that always coincides with a
    snare carries that snare's variation into its own score.

    Only genuinely conflicting drums are passed in, not the whole kit.
    Requiring total isolation is useless in practice -- a hi-hat lands on
    almost every kick in most music, so a strict test finds zero usable hits
    and the measurement silently disappears. A hat sits far above the band the
    kick is measured in, so it does not conflict; a snare or tom does.
    """
    if not conflicting_onsets:
        return list(onsets)
    arr = np.asarray(sorted(conflicting_onsets), dtype=float)
    out = []
    for t in onsets:
        t = float(t)
        lo = np.searchsorted(arr, t - guard, side="left")
        hi = np.searchsorted(arr, t + guard, side="right")
        # No conflicting drum anywhere in the guard window.
        if hi - lo == 0:
            out.append(t)
    return out


def _drum_hit_consistency(segs, sr, max_hits=60):
    """How alike successive hits of the same drum are.

    Programmed/sampled drums retrigger the same waveform, so their spectra are
    near-identical hit to hit; a human hitting a real drum varies. Returns the
    mean pairwise cosine similarity of consecutive hits' log-magnitude spectra
    (1.0 = identical). This is the closest thing to an objective
    programmed-vs-played signal available from the audio, and nothing in the
    pipeline measured it -- 'programmed drums' was previously a genre-driven
    assumption.

    Feed this WIDE-band, isolated hits. Restricted to a kick's 30-150 Hz band
    the metric barely discriminates, because the low fundamental of a real
    played kick is nearly as repeatable as a sampled one -- what actually
    varies hit to hit is the attack/beater content higher up. Measured that
    way a live kit scored 0.98 and was reported as 'retriggered sample',
    which is exactly the wrong answer.
    """
    segs = [s for s in segs if s is not None and len(s) >= 512][:max_hits]
    if len(segs) < 4:
        return None
    n_fft = 1024
    specs = []
    for s in segs:
        s = np.asarray(s[:n_fft], dtype=np.float32)
        if len(s) < n_fft:
            s = np.pad(s, (0, n_fft - len(s)))
        mag = np.abs(np.fft.rfft(s * np.hanning(n_fft)))
        mag = np.log1p(mag)
        nrm = float(np.linalg.norm(mag))
        if nrm > 1e-9:
            specs.append(mag / nrm)
    if len(specs) < 4:
        return None
    sims = [float(np.dot(specs[i], specs[i + 1])) for i in range(len(specs) - 1)]
    return float(np.median(sims))


def _consistency_note(cons):
    """Wording for a hit-consistency score.

    Only the extremes are asserted. The middle of this range is deliberately
    left ambiguous: a tightly played, well-recorded kit and a sample-replaced
    one both sit around 0.98-0.99, and the absolute thresholds here have been
    validated on synthetic kits rather than on a corpus of real recordings.
    Claiming "programmed" from a mid-range score would turn a genuine
    measurement back into the guess it was meant to replace.
    """
    if cons is None:
        return ""
    if cons >= 0.995:
        return ("hits are essentially identical — a retriggered sample; "
                "safe to call this programmed/sequenced")
    if cons >= 0.97:
        return ("hits are very repeatable — consistent with programming or "
                "sample-replacement, but also with a tightly played, closely "
                "miked kit; do not assert programmed on this number alone")
    if cons >= 0.93:
        return "hits vary noticeably — played, or programmed with round-robin/humanisation"
    return "hits vary a lot — played, or heavily processed/varied per hit"


def build_drum_production_report(drums_stem_path, hits, bpm=None):
    """Sonic/production character of the kit, measured from the drums stem.

    The pipeline already described WHAT the drummer plays (rates, grid, swing)
    but nothing about how the kit SOUNDS -- whether the kick is a sub-heavy
    808 or a tight acoustic thud, whether the snare is gated or roomy, how
    hard the bus is compressed, whether the hits are retriggered samples or
    played. Those are exactly the production questions TIMBRE and the era/genre
    prompts ask about, and they were being answered from genre expectation.

    Returns "" when it cannot measure anything. Every measurement is
    independent and failure-isolated.
    """
    if not drums_stem_path or not hits:
        return ""
    if not os.path.exists(str(drums_stem_path)):
        return ""

    try:
        y, sr = librosa.load(drums_stem_path, sr=None, mono=True)
    except Exception:
        return ""
    if y is None or len(y) < 1000:
        return ""

    by_type = {}
    for h in hits:
        t = h.get("drum_type") or "other"
        by_type.setdefault(t, []).append(float(h["onset"]))

    # Every hit in the stem, used to stop each analysis window before the next
    # drum lands (see _drum_hit_windows).
    all_onsets = sorted(float(h["onset"]) for h in hits if h.get("onset") is not None)

    # Tone/decay windows are band-limited, so they only need to stop at hits
    # that put energy into the SAME band. Truncating a kick's 30-150 Hz window
    # at the next hi-hat (often 100-200 ms later) cut almost every kick off
    # before it could decay, which is half of why kick decays read so short.
    def _conflict_onsets(kind):
        kinds = _DRUM_WINDOW_CONFLICTS.get(kind)
        if not kinds:
            return all_onsets
        return sorted(t for k in kinds for t in (by_type.get(k) or []))

    lines = []

    # --- Kick character -----------------------------------------------------
    kicks = sorted(by_type.get("kick") or [])
    if len(kicks) >= 4:
        segs = _drum_hit_windows(
            y, sr, kicks[:80], post=0.6, all_onsets=_conflict_onsets("kick"),
            band=_DRUM_ANALYSIS_BANDS["kick"],
        )
        decays = [d for d in (_drum_decay_ms(s, sr) for s in segs) if d]
        fundamentals = [f for f in (_drum_fundamental_hz(s, sr) for s in segs) if f]
        parts = []
        if fundamentals:
            f0 = float(np.median(fundamentals))
            parts.append(f"fundamental ~{f0:.0f} Hz")
            if f0 < 50:
                parts.append("sub-heavy (808/synth-bass-drum territory)")
            elif f0 < 70:
                parts.append("deep, weighted low end")
            elif f0 < 110:
                parts.append("punchy, midweight kick")
            else:
                parts.append("light/high-tuned kick")
        if decays:
            d = float(np.median(decays))
            parts.append(f"-20 dB decay ~{d:.0f} ms")
            if d > 300:
                parts.append("long sustained tail (808-style)")
            elif d > 120:
                parts.append("moderate tail")
            else:
                parts.append("tight/damped")
        if parts:
            lines.append("kick sound: " + "; ".join(parts))
        # Consistency uses isolated hits over a wide band: the attack content is
        # where a played kit differs from a retriggered sample, and it lives
        # well above the 30-150 Hz band used for tone measurements above.
        kick_conflicts = sorted(
            (by_type.get("snare") or []) + (by_type.get("tom") or [])
            + (by_type.get("cymbal") or [])
        )
        iso = _isolated_onsets(kicks, kick_conflicts)
        if len(iso) >= 4:
            cons = _drum_hit_consistency(
                _drum_hit_windows(y, sr, iso[:80], post=0.25,
                                  all_onsets=all_onsets, band=(30.0, 2500.0)),
                sr,
            )
            if cons is not None:
                lines.append(
                    f"kick hit-to-hit consistency: {cons:.3f} "
                    f"({_consistency_note(cons)}; measured on {len(iso)} isolated hits)"
                )

    # --- Snare / backbeat character ----------------------------------------
    snares = sorted(by_type.get("snare") or [])
    if len(snares) >= 4:
        segs = _drum_hit_windows(
            y, sr, snares[:80], post=0.5, all_onsets=_conflict_onsets("snare"),
            band=_DRUM_ANALYSIS_BANDS["snare"],
        )
        decays = [d for d in (_drum_decay_ms(s, sr) for s in segs) if d]
        cents = [c for c in (_drum_centroid_hz(s, sr) for s in segs) if c]
        parts = []
        if cents:
            c = float(np.median(cents))
            parts.append(f"brightness ~{c:.0f} Hz centroid")
            parts.append("bright//crisp or clap-like" if c > 3500 else
                         "midrange-weighted, bodied" if c > 1800 else "dark/thick")
        if decays:
            d = float(np.median(decays))
            parts.append(f"-20 dB decay ~{d:.0f} ms")
            if d < 80:
                parts.append("very tight — gated or heavily damped")
            elif d < 200:
                parts.append("controlled, modern-dry")
            else:
                parts.append("open with audible tail/room")
        if parts:
            lines.append("snare/backbeat sound: " + "; ".join(parts))
        # The kick sits below this band, so it is not a conflict for the snare.
        snare_conflicts = sorted(
            (by_type.get("tom") or []) + (by_type.get("cymbal") or [])
        )
        iso_s = _isolated_onsets(snares, snare_conflicts)
        if len(iso_s) >= 4:
            cons = _drum_hit_consistency(
                _drum_hit_windows(y, sr, iso_s[:80], post=0.25,
                                  all_onsets=all_onsets, band=(150.0, 9000.0)),
                sr,
            )
            if cons is not None:
                lines.append(
                    f"snare hit-to-hit consistency: {cons:.3f} "
                    f"({_consistency_note(cons)}; measured on {len(iso_s)} isolated hits)"
                )

    # --- Hats ---------------------------------------------------------------
    hats = sorted(by_type.get("hihat") or [])
    if len(hats) >= 6:
        segs = _drum_hit_windows(
            y, sr, hats[:80], post=0.15, all_onsets=_conflict_onsets("hihat"),
            band=_DRUM_ANALYSIS_BANDS["hihat"],
        )
        cents = [c for c in (_drum_centroid_hz(s, sr) for s in segs) if c]
        decays = [d for d in (_drum_decay_ms(s, sr, drop_db=15.0) for s in segs) if d]
        parts = []
        if cents:
            c = float(np.median(cents))
            parts.append(f"brightness ~{c:.0f} Hz centroid")
            parts.append("very bright/thin" if c > 8000 else
                         "bright" if c > 5500 else "darker/filtered")
        if decays:
            parts.append(f"typical ring ~{float(np.median(decays)):.0f} ms")
        if parts:
            lines.append("hi-hat sound: " + "; ".join(parts))

    # --- Whole-kit dynamics -------------------------------------------------
    # Hit levels must be MEASURED from the audio (amplitude_source set by
    # _attach_measured_amplitudes, or the onset fallback's own envelope). The
    # drum transcriber writes velocity 100 for every hit, so the old line --
    # computed from that constant -- reported "0.00, machine-like/quantized
    # dynamics" for every track, played or programmed. Spread is taken per
    # drum and then combined, because pooling kicks with hi-hats measures the
    # level difference between drums, not how evenly each one is played.
    try:
        per_type = {}
        for h in hits:
            src = h.get("amplitude_source")
            if src != "stem audio" and h.get("detected_from") != "onset-band fallback":
                continue
            dt = h.get("drum_type")
            if dt in ("kick", "snare", "hihat") and h.get("amplitude") is not None:
                per_type.setdefault(dt, []).append(float(h["amplitude"]))
        spreads = {
            dt: float(np.percentile(v, 90) - np.percentile(v, 10))
            for dt, v in per_type.items() if len(v) >= 8
        }
        if spreads:
            spread = float(np.median(list(spreads.values())))
            detail = ", ".join(
                f"{ {'hihat': 'hi-hat'}.get(dt, dt)} {v:.2f}" for dt, v in sorted(spreads.items())
            )
            lines.append(
                f"hit-level spread measured from the drums audio (90th-10th pct, per drum): "
                f"{detail} "
                + ("— nearly uniform; machine-like/quantized dynamics"
                   if spread < 0.12 else
                   "— some accenting, but controlled" if spread < 0.30 else
                   "— wide dynamic accenting, human-feeling")
            )
    except Exception:
        pass

    # Drum-bus crest: how hard the kit itself is compressed, independent of the
    # whole-mix loudness figures (which the bass and synths dominate). RMS is
    # taken over the passages where the kit is actually playing: averaging in
    # drum-free intros and breakdowns lowered the RMS and inflated the crest,
    # pushing every arrangement with a drum-less section toward "open, lightly
    # compressed".
    try:
        peak = float(np.max(np.abs(y)))
        yy = y.astype(np.float64)
        fr = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0].astype(np.float64)
        active = fr >= (float(np.max(fr)) * 10.0 ** (-35.0 / 20.0)) if fr.size else fr
        if fr.size and np.any(active):
            rms = float(np.sqrt(np.mean(np.square(fr[active]))))
        else:
            rms = float(np.sqrt(np.mean(np.square(yy))))
        if peak > 1e-6 and rms > 1e-9:
            crest = 20.0 * np.log10(peak / rms)
            lines.append(
                f"drum-bus peak-to-RMS (while the kit is playing): {crest:.1f} dB "
                + ("— very squashed; heavy bus compression/limiting on the kit"
                   if crest < 10 else
                   "— moderately compressed kit" if crest < 15 else
                   "— open, lightly compressed kit with intact transients")
            )
    except Exception:
        pass

    if not lines:
        return ""

    return (
        "DRUM SOUND / PRODUCTION (objective, measured from the separated drums stem):\n"
        "These describe how the kit SOUNDS, not what it plays. Use them for questions about "
        "drum tone, punch, room, and whether the kit is programmed or played. Note the stem "
        "comes from source separation, so faint bleed from other instruments is possible — "
        "treat these as strong indicators rather than absolute values.\n"
        + "\n".join(lines)
    )


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

    # Grid judgements below measure phase against the tempo; fit that tempo to
    # the kit itself first so rounding in the reported BPM cannot masquerade as
    # syncopation (see _refine_bpm_from_onsets). Only the transcribed kit
    # pieces are used -- audio-detected cymbals and percussion are sparser and
    # less precisely timed.
    if bpm:
        _fit_onsets = []
        for _k in ("kick", "snare", "hihat", "hi-hat"):
            _fit_onsets.extend(by_type.get(_k) or [])
        bpm = _refine_bpm_from_onsets(_fit_onsets, bpm)

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
    # The kick pulse does not depend on the snare. It used to be measured only
    # when snares were present too, so a snare-less passage (kick + hats verse)
    # lost its hat-density figure and was told there was "no stable kick pulse".
    if len(kicks) >= 4:
        kick_ibi = float(np.median(np.diff(sorted(kicks))))
        if not (kick_ibi > 0):
            kick_ibi = None
    if len(kicks) >= 4 and len(snares) >= 2:
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
    # Hat subdivision measured against the BEAT when the tempo is known. The
    # kick-relative ratio alone mislabels the most common pattern in pop and
    # rock: 8th-note hats over a half-time kick (kicks on 1 and 3) come out as
    # "4x the kick pulse" and were reported as 16th-note hat work.
    hat_per_beat = None
    hat_med = None
    if len(hihats) >= 2:
        _hd = np.diff(sorted(hihats))
        _hd = _hd[_hd > 1e-6]
        if _hd.size:
            hat_med = float(np.median(_hd))
    if hat_med and bpm:
        try:
            hat_per_beat = (60.0 / float(bpm)) / hat_med
        except (TypeError, ValueError, ZeroDivisionError):
            hat_per_beat = None
    if hihats and kicks and kick_ibi and kick_ibi > 0:
        if hat_med and hat_med > 1e-6:
            hat_vs_kick = kick_ibi / hat_med
            if hat_per_beat is not None:
                density_word = (
                    "very dense (16th-note hats)" if hat_per_beat >= 3.2
                    else "busy (8th-note hats)" if hat_per_beat >= 1.6
                    else "quarter-note hats, one per beat" if hat_per_beat >= 0.8 and hat_per_beat < 1.2
                    else "sparse / on the beat" if hat_per_beat < 0.8
                    else "moderate"
                )
            else:
                density_word = (
                    "very dense (8th/16th-ish)" if hat_vs_kick >= 3.2
                    else "busy (around 8ths)" if hat_vs_kick >= 1.8
                    else "sparse / on the beat" if hat_vs_kick < 1.2
                    else "moderate"
                )
            per_beat_txt = (
                f", ~{hat_per_beat:.1f} per beat" if hat_per_beat is not None else ""
            )
            lines.append(
                f"hi-hat density vs kick pulse: ~{hat_vs_kick:.1f}x{per_beat_txt} "
                f"({density_word}; {len(hihats)} hat hits)"
            )
    elif hihats:
        if hat_per_beat is not None:
            lines.append(
                f"hi-hat activity: {len(hihats)} hits, ~{hat_per_beat:.1f} per beat "
                "(no stable kick pulse for a kick-relative ratio)"
            )
        else:
            lines.append(f"hi-hat activity: {len(hihats)} hits (no stable kick pulse for ratio)")

    # Fill / tom activity as a density proxy (not precise fill detection).
    if toms or cymbals:
        lines.append(
            f"tom/cymbal activity: toms={len(toms)}, cymbals={len(cymbals)} "
            f"(elevated counts often mean fills or open cymbal work)"
        )

    # Finer kit breakdown. The buckets above collapse a lot of detail that a
    # listener plainly hears -- open vs closed hats, ride vs crash, a clap
    # standing in for a snare, which tom -- so where the transcription
    # identified the specific instrument, report it. This is additive: the
    # groove statistics above still use the coarse buckets.
    subtype_counts = {}
    for n in filtered:
        st = n.get("drum_subtype")
        if st:
            subtype_counts[st] = subtype_counts.get(st, 0) + 1
    if subtype_counts:
        ranked_sub = sorted(subtype_counts.items(), key=lambda kv: -kv[1])
        lines.append(
            "kit detail (specific instruments identified in the transcription): "
            + ", ".join(f"{name} x{count}" for name, count in ranked_sub[:12])
        )
        open_hats = subtype_counts.get("open hi-hat", 0)
        closed_hats = subtype_counts.get("closed hi-hat", 0)
        if open_hats and closed_hats:
            lines.append(
                f"hi-hat articulation: {closed_hats} closed vs {open_hats} open "
                f"({open_hats / float(open_hats + closed_hats) * 100:.0f}% open) "
                "-- open hats on the off-beats are a house/dance signature; "
                "sparse opens usually mark section ends or accents"
            )
        elif closed_hats and not open_hats:
            lines.append("hi-hat articulation: closed throughout (no open hats transcribed)")
        claps = subtype_counts.get("hand clap", 0)
        real_snare = subtype_counts.get("acoustic snare", 0) + subtype_counts.get("electric snare", 0)
        if claps and claps >= real_snare:
            lines.append(
                f"backbeat voice: hand claps ({claps}) rather than (or layered over) a struck "
                f"snare ({real_snare}) -- describe the backbeat as claps, not as a snare"
            )
        rides = subtype_counts.get("ride cymbal", 0) + subtype_counts.get("ride cymbal 2", 0) \
            + subtype_counts.get("ride bell", 0)
        crashes = subtype_counts.get("crash cymbal", 0) + subtype_counts.get("crash cymbal 2", 0)
        if rides or crashes:
            lines.append(
                f"cymbal voices: ride/bell hits={rides}, crash hits={crashes} "
                "(a sustained ride pattern is a different groove from occasional crashes)"
            )
        aux = {k: v for k, v in subtype_counts.items() if k in (
            "tambourine", "cowbell", "shaker", "claves", "conga", "bongo",
            "high bongo", "low bongo", "muted high conga", "open high conga",
            "low conga", "high timbale", "low timbale", "cabasa", "maracas",
            "triangle", "open triangle", "muted triangle", "wood block",
            "high wood block", "low wood block", "guiro",
        )}
        if aux:
            lines.append(
                "auxiliary percussion present: "
                + ", ".join(f"{k} x{v}" for k, v in sorted(aux.items(), key=lambda kv: -kv[1]))
            )

    # Grid-sensitive judgements (swing, kick grid, snare grid) are made on a
    # tempo-steadied timeline when the playing drifts. The rate/spacing lines
    # above keep the real timestamps. See _tempo_warp_onsets.
    g_kicks, g_snares, g_hihats = kicks, snares, hihats
    if bpm:
        try:
            _warp = _tempo_warp_onsets(list(kicks) + list(snares) + list(hihats), bpm)
        except Exception:
            _warp = None
        if _warp is not None:
            g_kicks = sorted(_warp(t) for t in kicks)
            g_snares = sorted(_warp(t) for t in snares)
            g_hihats = sorted(_warp(t) for t in hihats)

    # Swing/shuffle: use the densest available subdivision stream (hi-hat
    # first, since it's normally the clearest subdivision voice; fall back
    # to snare or kick if the hat is too sparse to judge).
    swing_label, swing_ratio, swing_detail = None, None, None
    for _cand_onsets in (g_hihats, g_snares, g_kicks):
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

    kick_grid_label, kick_grid_detail = _classify_kick_grid(g_kicks, bpm)
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

    snare_grid_label, snare_grid_detail = _classify_snare_grid(g_snares, bpm, kicks=g_kicks)
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

    if hat_per_beat is not None and len(hihats) >= 8:
        if hat_per_beat >= 3.2:
            groove_bits.append("busy closed-hat or 16th-note hat work")
        elif hat_per_beat >= 1.6:
            groove_bits.append("regular 8th-note hat layer")
        elif 0.8 <= hat_per_beat < 1.2:
            groove_bits.append("quarter-note hat pulse (one hat per beat)")
        elif hat_per_beat < 0.8:
            groove_bits.append("open or sparse hat / little continuous hat bed")
    elif hat_vs_kick is not None:
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


def _fmt_mmss(t):
    t = max(0.0, float(t))
    return f"{int(t // 60)}:{int(t % 60):02d}"


def _drum_active_spans(onsets, max_gap):
    """Group one drum type's onsets into continuous playing spans.

    A span breaks when the instrument stops for longer than `max_gap`, which is
    set to a few bars so ordinary pattern spacing (a snare on 2 and 4) never
    splits a span, but a real drop-out does.
    """
    spans = []
    if not onsets:
        return spans
    ordered = sorted(float(o) for o in onsets)
    start = prev = ordered[0]
    for t in ordered[1:]:
        if t - prev > max_gap:
            spans.append((start, prev))
            start = t
        prev = t
    spans.append((start, prev))
    return spans


def _describe_kit_config(active):
    """Plain-language name for a set of simultaneously-playing drum types."""
    if not active:
        return "no drums"
    pretty = {"kick": "kick", "snare": "snare", "hihat": "hi-hat",
              "tom": "toms", "cymbal": "cymbals", "percussion": "percussion"}
    order = ["kick", "snare", "hihat", "tom", "cymbal", "percussion", "other"]
    names = [pretty.get(a, a) for a in order if a in active]
    names += [pretty.get(a, a) for a in sorted(active) if a not in order]
    label = " + ".join(names)
    core = {"kick", "snare", "hihat"}
    if core.issubset(active):
        label = "full kit (" + label + ")"
    elif active & core and "snare" not in active:
        label += " (no snare)"
    return label


def _summarize_drum_entry_timeline(filtered, bpm=None, total_duration=None,
                                   min_segment_s=2.0):
    """Say WHEN each part of the kit is playing, not just what the kit does on
    average across the track.

    Every other drum statistic in this report is a whole-track or whole-section
    aggregate, so a track that starts with no drums, brings in a kick, and only
    adds the snare later reads as one averaged pattern -- there was no evidence
    anywhere that the arrangement changed, and questions like "what are the
    drums doing in the first half of the verse" could only be answered from
    that average. This walks the actual onset times and reports the kit
    configuration over time, including silence before the drums enter.

    Returns a list of text lines, or [] when there is nothing useful to say.
    """
    if not filtered:
        return []
    by_type = {}
    for n in filtered:
        t = n.get("drum_type") or "other"
        try:
            by_type.setdefault(t, []).append(float(n["onset"]))
        except (TypeError, ValueError, KeyError):
            continue
    if not by_type:
        return []

    beat_len = (60.0 / float(bpm)) if bpm and float(bpm) > 0 else 0.5
    # A few bars of silence is a real drop-out; anything shorter is just the
    # instrument's own spacing (or a fill) and must not split a span.
    max_gap = max(3.0, 8.0 * beat_len)

    spans_by_type = {t: _drum_active_spans(o, max_gap) for t, o in by_type.items()}

    first_seen = {t: min(o) for t, o in by_type.items() if o}
    last_seen = {t: max(o) for t, o in by_type.items() if o}
    track_end = float(total_duration) if total_duration else max(last_seen.values())
    track_end = max(track_end, max(last_seen.values()))

    # Change points: every moment the set of playing instruments could differ.
    points = {0.0, track_end}
    for spans in spans_by_type.values():
        for a, b in spans:
            points.add(a)
            points.add(b)
    edges = sorted(p for p in points if 0.0 <= p <= track_end)

    segments = []
    for i in range(len(edges) - 1):
        a, b = edges[i], edges[i + 1]
        if b - a <= 0.01:
            continue
        mid = (a + b) / 2.0
        active = frozenset(
            t for t, spans in spans_by_type.items()
            if any(s <= mid <= e for s, e in spans)
        )
        if segments and segments[-1][2] == active:
            segments[-1] = (segments[-1][0], b, active)
        else:
            segments.append((a, b, active))

    # Drop segments too short to be a real arrangement change, folding them
    # into whichever neighbour they interrupt, then re-merge.
    if len(segments) > 1:
        kept = []
        for seg in segments:
            if seg[1] - seg[0] < float(min_segment_s) and kept:
                kept[-1] = (kept[-1][0], seg[1], kept[-1][2])
            else:
                kept.append(seg)
        # A short FIRST segment has no earlier neighbour to fold into, so it
        # used to survive as its own line ("0:00-0:00: kick + hi-hat (no
        # snare)") -- usually just the first hit of each drum landing a few
        # milliseconds apart. Fold it forward into the segment that follows.
        if len(kept) >= 2 and kept[0][1] - kept[0][0] < float(min_segment_s):
            kept[1] = (kept[0][0], kept[1][1], kept[1][2])
            kept = kept[1:]
        merged = []
        for seg in kept:
            if merged and merged[-1][2] == seg[2]:
                merged[-1] = (merged[-1][0], seg[1], merged[-1][2])
            else:
                merged.append(list(seg))
        segments = [tuple(x) for x in merged]

    if not segments:
        return []

    lines = ["DRUM ARRANGEMENT TIMELINE (kit configuration over time, computed from the "
             "actual hit timestamps — use this, not the whole-track pattern, when asked "
             "what the drums are doing at a particular moment, when they come in, or "
             "whether they drop out; a segment listed as 'no drums' means no drum hits "
             "were transcribed in that span at all):"]
    for a, b, active in segments:
        lines.append(
            f"  {_fmt_mmss(a)}-{_fmt_mmss(b)}: {_describe_kit_config(set(active))}"
        )

    # Plain type names here -- _describe_kit_config() appends "(no snare)" to a
    # partial kit, which is right for a segment but nonsense for a single type.
    _pretty = {"kick": "kick", "snare": "snare", "hihat": "hi-hat", "tom": "toms",
               "cymbal": "cymbals", "percussion": "percussion"}
    entries = []
    for t in sorted(first_seen, key=lambda k: first_seen[k]):
        entries.append(f"{_pretty.get(t, t)} from {_fmt_mmss(first_seen[t])}")
    if entries:
        lines.append("  first entry per drum type: " + "; ".join(entries))
    return lines


def detect_cymbal_onsets(drums_stem_path, hits, max_hits=600):
    """Recover cymbal hit TIMES from the drums stem audio.

    The note transcription cannot represent cymbals, and the percussion probe
    only says one exists somewhere in a 10-second window -- too coarse for a
    grid, which needs onsets.

    The discriminator is the high-band FLOOR BETWEEN hits, not the decay of any
    single hit. Measuring one hit's decay does not work here: a cymbal almost
    always rings underneath a busy hi-hat pattern, and any per-hit window has
    to stop at the next hit a few tens of milliseconds later, so the ring is
    never inside the window. But a crash or ride raises the high-frequency
    floor for hundreds of milliseconds, whereas closed hats are spikes with
    near-silence between them. So: after each hit, look at the QUIETEST moment
    in the following window. If the top end never falls back to the track's
    usual between-hits level, something up there is still ringing.

    This ADDS a row; it never edits the transcribed kick/snare/hat data, and a
    hit can be both (a crash landing with a kick is ordinary).

    Returns a sorted list of onset times, or [] when nothing is confident.
    """
    if not CYMBAL_ONSET_DETECT_ENABLED or not drums_stem_path or not hits:
        return []
    if not os.path.exists(str(drums_stem_path)):
        return []
    try:
        y, sr = librosa.load(str(drums_stem_path), sr=None, mono=True)
    except Exception:
        return []
    if y is None or not len(y):
        return []

    onsets = []
    for h in hits:
        try:
            v = h.get("onset", h.get("t"))
            if v is not None:
                onsets.append(round(float(v), 3))
        except (TypeError, ValueError):
            continue
    onsets = sorted(set(onsets))
    if not onsets:
        return []
    if len(onsets) > max_hits:
        step = len(onsets) / float(max_hits)
        onsets = [onsets[int(i * step)] for i in range(max_hits)]

    lo, hi = CYMBAL_BAND_HZ
    try:
        hf = _drum_band_limit(y, sr, lo, hi)
        if hf is None or not len(hf):
            return []
        env = np.abs(np.asarray(hf, dtype=np.float64))
        win = max(1, int(sr * 0.005))
        if win > 1:
            env = np.convolve(env, np.ones(win) / win, mode="same")
    except Exception:
        return []

    n = len(env)
    if n < sr // 2:
        return []

    def _floor_after(t):
        a = int((t + 0.15) * sr)
        b = int((t + 0.45) * sr)
        if a >= n:
            return None
        return float(np.min(env[a:min(b, n)])) if min(b, n) > a else None

    floors, peaks = [], []
    for t in onsets:
        f = _floor_after(t)
        a = int(max(0.0, t) * sr)
        b = int((t + 0.05) * sr)
        pk = float(np.max(env[a:min(b, n)])) if min(b, n) > a else 0.0
        floors.append(f)
        peaks.append(pk)

    valid = [f for f in floors if f is not None]
    if len(valid) < 8:
        return []
    # The track's own idea of "top end has decayed away", so this adapts to
    # both a dry programmed kit and a washy live one.
    baseline = float(np.median(valid))
    if baseline <= 0:
        baseline = float(np.mean(valid)) or 1e-9

    found = []
    for t, f, pk in zip(onsets, floors, peaks):
        if f is None or pk <= 0:
            continue
        # Still ringing well above the track's usual between-hits floor...
        if f < baseline * float(CYMBAL_FLOOR_RATIO):
            continue
        # ...and the sustained energy is a real fraction of the hit itself, so
        # a quiet hit sitting inside another cymbal's wash is not double-counted.
        if (f / pk) < float(CYMBAL_MIN_SUSTAIN_RATIO):
            continue
        # ...and there must be a fresh ATTACK here. A crash rings for over a
        # second, so the two or three hits that follow it all sit on a raised
        # high-frequency floor and pass the tests above on the strength of the
        # FIRST cymbal's tail -- one crash would be drawn as four. A struck
        # cymbal spikes above the level immediately preceding it; a decaying
        # tail does not. A time-based merge cannot separate these, because a
        # ride played on every beat has exactly the spacing of a crash tail.
        a0 = int(max(0.0, t - 0.06) * sr)
        a1 = int(max(0.0, t - 0.005) * sr)
        prev = float(np.max(env[a0:a1])) if a1 > a0 else 0.0
        if prev > 0 and (pk / prev) < float(CYMBAL_MIN_ATTACK_RATIO):
            continue
        # ...and this hit must actually RAISE the sustained top end. A hat or
        # snare landing on an earlier crash's tail passes the attack test (it
        # is a fresh transient) and the floor test (the tail is still ringing),
        # so each crash was also reported a beat or two later. Comparing the
        # floor after the hit with the floor just before it separates the two:
        # a new cymbal lifts it, a hit on a decaying tail does not.
        b0 = int(max(0.0, t - 0.35) * sr)
        b1 = int(max(0.0, t - 0.03) * sr)
        floor_before = float(np.min(env[b0:b1])) if b1 > b0 else 0.0
        if floor_before > 0 and f < floor_before * 1.5:
            continue
        found.append(float(t))

    if len(found) < int(CYMBAL_MIN_DETECTIONS):
        return []
    # Collapse a ring detected across several consecutive hits down to its
    # start: one crash, not one per hit it happens to cover.
    merged = [found[0]]
    for t in found[1:]:
        if t - merged[-1] >= float(CYMBAL_MERGE_GAP_S):
            merged.append(t)
    return merged


def _refine_local_grid(onsets, cell_len, ref=0.0, max_dev=0.03, steps=241):
    """Fit the grid's cell length and phase to a local group of onsets.

    Returns (cell_len, phase) where grid points are phase + k * cell_len.

    The cell length is searched within +/- max_dev of the nominal value and
    scored by how tightly the onsets cluster at one sub-cell offset (circular
    resultant length). Offsets are measured relative to `ref`, a time near the
    onsets, so the fit is not dominated by how far the window sits from 0:00.
    The nominal cell is kept unless another length fits clearly better, so a
    well-tempo'd track gets exactly the grid it always did.
    """
    arr = np.asarray(onsets if onsets is not None else [], dtype=float)
    if arr.size == 0 or cell_len <= 0:
        return cell_len, 0.0
    rel = arr - float(ref)

    def _fit(c):
        ang = 2.0 * np.pi * np.mod(rel, c) / c
        s, co = float(np.mean(np.sin(ang))), float(np.mean(np.cos(ang)))
        return float(np.hypot(s, co)), float(np.arctan2(s, co))

    r0, a0 = _fit(cell_len)
    best_c, best_r, best_a = cell_len, r0, a0
    if arr.size >= 8:
        for scale in np.linspace(1.0 - max_dev, 1.0 + max_dev, int(steps)):
            c = cell_len * float(scale)
            r, a = _fit(c)
            if r > best_r:
                best_c, best_r, best_a = c, r, a
        # Require a real improvement before abandoning the nominal tempo.
        if best_r < r0 + 0.03:
            best_c, best_r, best_a = cell_len, r0, a0
    offset = (best_a / (2.0 * np.pi)) * best_c % best_c
    return float(best_c), float((float(ref) + offset) % best_c)


def _grid_alignment(by_type, cell_index, subdiv, anchor_cell=None):
    """Find which cells are beats, and which beat is beat 1.

    Both are decided from the hits themselves, in CELL INDEX space. Phase only
    fixes the sub-cell offset; it says nothing about which cells carry the
    pulse, so anchoring the bar to phase alone leaves the whole diagram free to
    sit a subdivision or a beat out -- rhythmically correct but starting in the
    wrong place, which for a reader is simply a wrong pattern.

    Returns (beat_residue, downbeat_offset):
      beat_residue    -- cell indices where c %% subdiv == this are on a beat
      downbeat_offset -- which of the four beats in the bar is beat 1
    """
    kicks = [cell_index(t) for t in (by_type.get("kick") or [])]
    snares = [cell_index(t) for t in (by_type.get("snare") or [])]
    cymbal_cells = [cell_index(t) for t in (by_type.get("cymbal") or [])]
    every = [cell_index(t) for ts in by_type.values() for t in ts]

    # Kicks define the pulse when present: they are the most reliably
    # on-the-beat part of a kit. Fall back to all hits for kick-less passages.
    # Backbeat snares are just as reliably on the beat, and settle the common
    # case where kicks alone tie between two residues (drum & bass kicks on 1
    # and the "and" of 3, boom-bap kicks pushed onto off-16ths). With kicks
    # only, that tie went to whichever residue was counted first, so the whole
    # diagram could be drawn an 8th or a 16th out. All other hits add a small
    # weight as a final tie-break.
    res_src = kicks if kicks else every
    res_counts = [0.0] * subdiv
    for c in res_src:
        res_counts[c % subdiv] += 1.0
    if kicks:
        for c in snares:
            res_counts[c % subdiv] += 1.0
        for c in every:
            res_counts[c % subdiv] += 0.01
    beat_residue = int(np.argmax(res_counts)) if any(res_counts) else 0

    def _beat_no(c):
        return ((c - beat_residue) // subdiv) % 4

    kh, sh = [0] * 4, [0] * 4
    for c in kicks:
        if c % subdiv == beat_residue:
            kh[_beat_no(c)] += 1
    for c in snares:
        if c % subdiv == beat_residue:
            sh[_beat_no(c)] += 1
    if not any(kh) and not any(sh):
        return beat_residue, 0

    # Accented cymbals (crashes) mark bar starts far more often than any other
    # beat. Only a light weight: they are audio detections, not transcribed hits.
    ch = [0] * 4
    for c in (cymbal_cells or []):
        if c % subdiv == beat_residue:
            ch[_beat_no(c)] += 1

    # Which beat the window's own start lands on. A section boundary is
    # normally a bar line, so when the kit alone is ambiguous -- kicks on 1 and
    # 3 score identically for offsets 0 and 2, which is the most common pattern
    # in popular music -- the window start decides. Previously the tie always
    # went to whichever offset happened to be tried first, so a grid could
    # quietly start on beat 3 and show the pattern half a bar rotated.
    anchor_beat = None
    if anchor_cell is not None:
        try:
            ac = int(anchor_cell)
            delta = (beat_residue - ac) % subdiv
            if delta > subdiv / 2.0:
                delta -= subdiv
            anchor_beat = _beat_no(ac + delta)
        except Exception:
            anchor_beat = None

    # A half-time snare lands once per bar, on beat 3. Scored as a 2-and-4
    # backbeat it fits beat 2 or beat 4 equally well and never beat 3, so the
    # diagram drew the snare on "2" while the snare analysis (correctly) said
    # "half-time, on 3". Recognise it: one beat position carries most of the
    # snares and the position two beats away carries almost none.
    s_tot = float(sum(sh))
    half_time = False
    if s_tot >= 3:
        s_top = int(np.argmax(sh))
        half_time = (sh[s_top] >= 0.6 * s_tot and sh[(s_top + 2) % 4] <= 0.25 * sh[s_top])

    raw = []
    for off in range(4):
        # Beat 1 usually carries a kick, and the backbeat snares sit opposite.
        kick_score = kh[off] + kh[(off + 2) % 4] * 0.5
        if half_time:
            snare_score = sh[(off + 2) % 4] * 2.0
        else:
            snare_score = sh[(off + 1) % 4] + sh[(off + 3) % 4]
        raw.append(kick_score * 2.0 + snare_score + ch[off] * 0.5)

    top = max(raw) if raw else 0.0
    bonus = 0.25 + 0.03 * max(0.0, top)
    best, best_score = 0, None
    for off in range(4):
        score = raw[off]
        if anchor_beat is not None and off == anchor_beat:
            # Tie-break only: large enough to settle an exact or near tie,
            # far too small to overturn a clear kick/snare pattern.
            score += bonus
        if best_score is None or score > best_score:
            best, best_score = off, score
    return beat_residue, best


_BEAT_GRID_ROW_ORDER = ("kick", "snare", "hihat", "tom", "cymbal", "percussion", "other")
_BEAT_GRID_ROW_NAMES = {
    "kick": "kick", "snare": "snare", "hihat": "hi-hat", "tom": "toms",
    "cymbal": "cymbals", "percussion": "perc", "other": "other",
}


def build_beat_grid_diagram(hits, bpm, start=None, end=None, bars=None,
                            subdiv=None, label=""):
    """Render the measured drum pattern as a fixed-width grid.

    Every column is exactly two characters wide and the count row uses the
    same spacing, so the beat markers line up with the cells beneath them in
    any monospace font. Alignment is the whole point of the diagram: a grid
    whose columns drift is worse than no grid, because it reads as a rhythm
    that was never played.

    Cells are filled from the real onsets, quantised to the nearest
    subdivision with a half-cell tolerance. An 'x' means at least one hit of
    that type landed in that cell.

    Returns "" when there is not enough to draw honestly.
    """
    if not BEAT_GRID_ENABLED or not hits or not bpm:
        return ""
    try:
        beat_len = 60.0 / float(bpm)
    except (TypeError, ValueError, ZeroDivisionError):
        return ""
    if beat_len <= 0:
        return ""
    bars = int(bars or BEAT_GRID_BARS)
    subdiv = int(subdiv or BEAT_GRID_SUBDIV)
    if bars < 1 or subdiv < 1:
        return ""

    window = [h for h in hits
              if (start is None or float(h.get("onset", -1)) >= float(start))
              and (end is None or float(h.get("onset", -1)) < float(end))]
    if len(window) < BEAT_GRID_MIN_HITS:
        return ""

    by_type = {}
    for h in window:
        try:
            by_type.setdefault(h.get("drum_type") or "other", []).append(float(h["onset"]))
        except (TypeError, ValueError, KeyError):
            continue
    if not by_type:
        return ""

    all_onsets = sorted(t for ts in by_type.values() for t in ts)
    cell_len = beat_len / float(subdiv)
    bar_len = beat_len * 4.0

    # Where the drawn bars will sit: one bar into the window when there is room,
    # so the diagram shows the settled pattern rather than a pickup or fill.
    first = all_onsets[0]
    span = (end if end is not None else all_onsets[-1]) - first
    lead_in = beat_len * 4.0 if span > beat_len * 4.0 * (bars + 1) else 0.0
    anchor_t = first + lead_in

    # LOCAL tempo and phase. The reported BPM is a rounded integer (and can be
    # a percent or so off the true tempo), and the phase used to be a circular
    # mean over EVERY hit in the window -- a whole section, or for the
    # representative grid the whole song. A 0.4 BPM rounding error drifts the
    # grid by a full 16th every ~15 s, so on anything longer than a short
    # section the averaged phase described no particular moment and the
    # rhythm came out shifted by one or two cells (kicks on the "e", snares on
    # the "&"). The grid is now fitted to the hits around the bars actually
    # drawn, with the cell length allowed to flex by up to 3%.
    loc_lo = anchor_t - bar_len
    loc_hi = anchor_t + (bars + 2) * bar_len
    local = [t for t in all_onsets if loc_lo <= t <= loc_hi]
    if len(local) < BEAT_GRID_MIN_HITS:
        local = all_onsets
    cell_len, phase = _refine_local_grid(local, cell_len, ref=anchor_t)

    def _cell_index(t):
        return int(round((float(t) - phase) / cell_len))

    # Beat/downbeat voting uses the same neighbourhood, so that tempo drift
    # far from the drawn bars cannot out-vote the hits actually shown.
    vote_lo = anchor_t - 2 * bar_len
    vote_hi = anchor_t + (bars + 4) * bar_len
    local_by_type = {}
    for dtype, ts in by_type.items():
        near = [t for t in ts if vote_lo <= t <= vote_hi]
        if near:
            local_by_type[dtype] = near
    if sum(len(v) for v in local_by_type.values()) < BEAT_GRID_MIN_HITS:
        local_by_type = by_type

    anchor_ref = float(start) if start is not None else first
    beat_res, down_off = _grid_alignment(
        local_by_type, _cell_index, subdiv, anchor_cell=_cell_index(anchor_ref)
    )

    # Walk forward in CELL space to the first cell that is both on a beat and
    # is beat 1 of a bar, so the grid can never start mid-beat or mid-bar.
    c0 = _cell_index(anchor_t)
    for _ in range(subdiv * 4 + 1):
        if c0 % subdiv == beat_res and ((c0 - beat_res) // subdiv) % 4 == down_off:
            break
        c0 += 1
    grid_start = phase + c0 * cell_len

    cells = bars * 4 * subdiv
    tol = cell_len / 2.0

    # Keep the drawn bars inside the material. Walking forward to a downbeat
    # (or the lead-in) could push the last bar past the end of a short section,
    # leaving trailing cells empty -- which reads as the kit stopping, a rhythm
    # that was never played. Step back a bar at a time while that happens and
    # there is earlier material to show instead.
    data_end = all_onsets[-1] + beat_len
    if end is not None:
        data_end = min(data_end, float(end))
    bar_cells = 4 * subdiv
    while (
        grid_start + cells * cell_len > data_end + tol
        and grid_start - bar_cells * cell_len >= first - tol
    ):
        grid_start -= bar_cells * cell_len

    grid_end = grid_start + cells * cell_len

    rows = {}
    for dtype, onsets in by_type.items():
        row = ["."] * cells
        used = False
        for t in onsets:
            if t < grid_start - tol or t >= grid_end + tol:
                continue
            idx = int(round((t - grid_start) / cell_len))
            if 0 <= idx < cells:
                row[idx] = "x"
                used = True
        if used:
            rows[dtype] = row
    if not rows:
        return ""

    ordered = [d for d in _BEAT_GRID_ROW_ORDER if d in rows]
    ordered += [d for d in sorted(rows) if d not in _BEAT_GRID_ROW_ORDER]

    # Includes len("count") so the header/count row's own label fits within
    # the same column width as every data row — see the header-row rename
    # below for why this label exists at all.
    name_w = max(
        [len(_BEAT_GRID_ROW_NAMES.get(d, d)) for d in ordered] + [len("count")]
    )

    # Count row. One token per cell, every token padded to the same 2 chars as
    # a grid cell, so column N of the count sits above column N of every row.
    marks_4 = ["1", "e", "&", "a"]
    counts = []
    for c in range(cells):
        beat = (c // subdiv) % 4 + 1
        sub = c % subdiv
        if sub == 0:
            counts.append(str(beat))
        elif subdiv == 4:
            counts.append(marks_4[sub])
        elif subdiv == 2:
            counts.append("&")
        elif subdiv == 3:
            counts.append("ea"[sub - 1] if sub <= 2 else ".")
        else:
            counts.append(".")

    def _render(tokens):
        out = []
        for c, tok in enumerate(tokens):
            out.append(tok)
            # Bar separator sits between bars, never inside one.
            if (c + 1) % (4 * subdiv) == 0 and (c + 1) < cells:
                out.append("|")
        return " ".join(out)

    res = {4: "16th", 2: "8th", 3: "triplet 8th", 1: "quarter"}.get(subdiv, f"1/{subdiv} beat")
    head = f"BEAT GRID{(' — ' + label) if label else ''}"
    meta = (
        f"{bars} bar{'s' if bars != 1 else ''} from {_fmt_mmss(grid_start)}, "
        f"{float(bpm):.0f} BPM, {res} resolution, 4-beat bar proxy"
    )
    out = [
        f"{head} ({meta}). Fixed-width: reproduce EXACTLY as-is inside a code block, "
        "never redraw or re-space it. The 'count' row is the beat position (1 e & a "
        "2 e & a...), not a drum -- do not read it as a percussion part. 'x' = a "
        "transcribed hit in that cell, '.' = no hit, "
        "'|' = bar line. Kick, snare and hi-hat come from the note transcription; "
        "a 'cymbals' row, when present, is detected from the stem audio instead "
        "and its timing is measured the same way. Toms, shakers and hand "
        "percussion still cannot be gridded — if the notes list them elsewhere, "
        "mention them separately rather than adding a row.",
        f"{'count':<{name_w}} : {_render(counts)}",
    ]
    for d in ordered:
        out.append(f"{_BEAT_GRID_ROW_NAMES.get(d, d):<{name_w}} : {_render(rows[d])}")
    return "\n".join(out)


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
        if key == "chorus" and re.fullmatch(r"Chorus [A-C]\d*", str(sec["name"])):
            groups[key]["label"] = "Chorus"      # pre60: the chorus groups pooled under one label
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
        block = (
            f"SECTION GROOVE — {label} ({range_str}):\n"
            + "\n".join(f"  {l}" for l in section_lines)
        )
        # A grid drawn from this section's own hits. Prose can say "syncopated
        # kick, busy hats"; only the grid says exactly where they land, and it
        # is drawn here rather than left to the writer because reproducing a
        # rhythm as aligned columns from a list of timestamps is not something
        # a language model does reliably.
        try:
            grid = ""
            # Try occurrences in chronological order (ranges[0] first, exactly
            # as before) and fall back to a later occurrence only if an
            # earlier one has nothing to draw from -- e.g. the first "Verse"
            # occurrence can be an intro stretch before the drums have
            # entered yet, which would otherwise silently drop the grid for
            # the whole section even though later occurrences are full of
            # hits.
            for range_start, range_end in ranges:
                grid = build_beat_grid_diagram(
                    filtered, bpm, start=range_start, end=range_end, label=label
                )
                if grid:
                    break
            if grid:
                block += "\n" + "\n".join(f"  {l}" for l in grid.split("\n"))
        except Exception:
            pass
        blocks.append(block)

    return blocks


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

def _drum_events_to_full_log(notes):
    """Return a compact string of all drum hits: 'drum_type@onset', separated by commas."""
    if not notes:
        return ""
    parts = []
    for n in notes:
        drum_type = n.get("drum_type") or _drum_band_label(float(n.get("frequency", 300.0)))
        onset = round(float(n["onset"]), 2)
        parts.append(f"{drum_type}@{onset}")
    return ", ".join(parts)


def _drum_extras_probe(drums_stem_path):
    """Find kit pieces the drum transcription structurally cannot detect.

    Reads raw per-window probabilities for cymbals, tambourine, shakers, toms'
    tuned cousins and other percussion straight from the cached PANNs matrix
    for the DRUMS stem, with no top-k truncation. Returns a list of
    (label, peak, coverage, hits, n_windows) sorted strongest first, or [].

    This is deliberately additive: it never touches drum_type, the groove
    statistics or the hit counts, all of which stay exactly as measured. It
    only supplies vocabulary for instruments that were previously invisible.
    """
    if not DRUM_EXTRAS_PROBE_ENABLED or not drums_stem_path:
        return []
    try:
        probs, times = _panns_window_probs(
            drums_stem_path, INSTRUMENT_TAG_WINDOW_SECONDS,
            INSTRUMENT_TAG_HOP_SECONDS, INSTRUMENT_TAG_MAX_WINDOWS,
            INSTRUMENT_TAG_NORMALIZE,
        )
    except Exception:
        return []
    if probs is None or not times:
        return []
    try:
        groups = _panns_label_groups(_INSTRUMENT_LABEL_MAP)
    except Exception:
        return []
    if not groups:
        return []

    n_windows = int(probs.shape[0])
    if not n_windows:
        return []

    out = []
    for mapped, cols in groups.items():
        if mapped not in _DRUM_EXTRA_PROBE_LABELS:
            continue
        try:
            per_window = probs[:, cols].max(axis=1).astype(float)
        except Exception:
            continue
        peak = float(per_window.max())
        hits = int(np.count_nonzero(per_window >= float(DRUM_EXTRAS_FIRE_PROB)))
        coverage = hits / float(n_windows)
        if peak >= float(DRUM_EXTRAS_REPORT_PEAK) or (
            coverage >= float(DRUM_EXTRAS_REPORT_COVERAGE)
            and peak >= float(DRUM_EXTRAS_FIRE_PROB)
        ):
            out.append((mapped, peak, coverage, hits, n_windows))
    out.sort(key=lambda r: r[1], reverse=True)
    return out


_TUNED_PERC_LABELS = (
    "glockenspiel", "bell", "church bell", "jingle bells", "chimes", "wind chimes", "tubular bells/chimes",
    "vibraphone", "marimba/xylophone", "mallet percussion (unspecified)", "steelpan/steel drum",
)


def _tuned_percussion_probe(mix_path, stems):
    """0.7-pre5 TUNED_PERCUSSION_PROBE: the instrument tagger's bell, chime,
    glockenspiel and mallet readings on the mix and every stem, from the same
    cached window probabilities the instrument tags use (same settings).
    Saved only, for calibration: {"labels": [...], "sources": {source:
    {label: [peak, peak time s, coverage, windows]}}} with readings of at
    least 5% peak, or None."""
    if not TUNED_PERCUSSION_PROBE:
        return None
    try:
        groups = _panns_label_groups(_INSTRUMENT_LABEL_MAP)
    except Exception:
        return None
    groups = {k: v for k, v in (groups or {}).items() if k in _TUNED_PERC_LABELS}
    if not groups:
        return None
    paths = [("mix", mix_path)] + [(k, (stems or {}).get(k)) for k in INSTRUMENT_TAG_STEMS]
    out = {"labels": sorted(groups), "fire_prob": float(DRUM_EXTRAS_FIRE_PROB), "sources": {}}
    for name, path in paths:
        if not path or str(path).startswith(("http://", "https://")) or not os.path.exists(str(path)):
            continue
        try:
            probs, times = _panns_window_probs(path, INSTRUMENT_TAG_WINDOW_SECONDS, INSTRUMENT_TAG_HOP_SECONDS,
                                               INSTRUMENT_TAG_MAX_WINDOWS, INSTRUMENT_TAG_NORMALIZE)
        except Exception:
            continue
        if probs is None or not times:
            continue
        n = int(probs.shape[0])
        src = {}
        for lab, cols in groups.items():
            try:
                pw = probs[:, cols].max(axis=1).astype(float)
            except Exception:
                continue
            k = int(np.argmax(pw))
            peak = float(pw[k])
            if peak >= 0.05:
                src[lab] = [round(peak, 3), round(float(times[k]), 1),
                            round(float(np.count_nonzero(pw >= float(DRUM_EXTRAS_FIRE_PROB))) / n, 3), n]
        out["sources"][name] = src
    return out if out["sources"] else None


def _tuned_percussion_for_save(mix_path, stems):
    try:
        return _tuned_percussion_probe(mix_path, stems)
    except Exception:
        return None


def _format_drum_extras(extras):
    """Turn _drum_extras_probe output into a report block."""
    if not extras:
        return ""
    parts = []
    for label, peak, coverage, hits, n_windows in extras:
        note = _tag_coverage_note({"hits": hits, "n_windows": n_windows})
        parts.append(
            f"{label} ({peak * 100:.0f}%, {note or f'{hits}/{n_windows} windows'})"
        )
    return (
        "ADDITIONAL PERCUSSION DETECTED (independent audio classifier run on the "
        "separated drums stem, NOT the note transcription). The transcription above "
        "resolves only kick, snare and hi-hat; these are kit pieces it is structurally "
        "incapable of detecting, so this is the ONLY evidence in the notes about them. "
        "Treat a sustained entry here as a real part of the kit and feel free to name it "
        "(crash/ride when cymbals are listed, tambourine, shaker and so on), while "
        "keeping in mind these are classifier tags rather than counted hits -- describe "
        "presence and rough extent, not exact hit counts or placement, and prefer the "
        "coverage note over inventing a pattern: "
        + "; ".join(parts)
    )


_DRUM_CLAIM_RE = re.compile(
    r"(?i)\b(?:drums?|drum\s*kit|kick|snare|hi-?hats?|backbeat|four-on-the-floor|percussion)\b")


def _drum_bleed_note(listening_text, rel_db):
    """The [DRUMS stem] block for a near-silent drums stem."""
    said = ""
    if listening_text and _DRUM_CLAIM_RE.search(str(listening_text)):
        said = (" Where the listening pass mentions drums, a kick, a snare or hi-hats, that is "
                "contradicted by this measurement.")
    return (
        f"[DRUMS stem]\nDRUM STEM NOTE: the separated drums stem sits {abs(rel_db):.0f} dB below the "
        "combined stems -- there is no drum kit on this recording. What the drum transcriber "
        "picked up there is bleed from other parts (strummed or picked strings, hand slaps on "
        "an instrument body, consonants in the voice), so its hit counts, grooves, beat grids "
        "and drum timeline are left out of these notes." + said + " Do not describe drums, a "
        "beat or a groove played on a kit unless the user says there are drums; describe the "
        "rhythm as carried by the instruments that are there."
    )


_FULL_BAND_TEMPO_HEAD = "FULL-BANDWIDTH TEMPO CHECK:"


def _tempo_multiple(hi, lo, tol=0.04):
    """1, 2 or 4 when `hi` is that multiple of `lo` (within tol), else None."""
    if not hi or not lo:
        return None
    for k in (1, 2, 4):
        if abs(hi / (lo * k) - 1.0) <= tol:
            return k
    return None


def _full_band_tempo_decision(lows, obj_hi, ess_hi, hf_share):
    """The full-bandwidth tempo to use, or None (see FULL_BAND_TEMPO_CHECK)."""
    lows = [float(v) for v in lows if v]
    if not lows or max(lows) >= FULL_BAND_TEMPO_SLOW_BPM:
        return None
    if hf_share is None or hf_share < FULL_BAND_TEMPO_MIN_HF_SHARE:
        return None
    his = [float(v) for v in (obj_hi, ess_hi) if v]
    if not his or (ess_hi is not None and obj_hi is not None and abs(obj_hi / ess_hi - 1.0) > 0.04):
        return None
    h = sum(his) / len(his)
    if not (FULL_BAND_TEMPO_SLOW_BPM <= h <= 200.0):
        return None
    ks = [_tempo_multiple(h, l) for l in lows]
    if any(k is None for k in ks) or max(ks) < 2:
        return None
    return h


def _full_band_tempo_check(out, source_path, dsp_path):
    """See FULL_BAND_TEMPO_CHECK. Rewrites out["objective_report"] and
    out["essentia_report"] in place when it applies. Never raises out."""
    obj = extract_objective_bpm(out.get("objective_report"))
    ess = extract_essentia_bpm(out.get("essentia_report"))
    lows = [v for v in (obj, ess) if v]
    if not lows or max(lows) >= FULL_BAND_TEMPO_SLOW_BPM or not source_path or not os.path.exists(str(source_path)):
        return False
    status("Tempo reads slow -- re-measuring on the full-bandwidth audio...")
    hi_path = convert_to_wav(str(source_path), sample_rate=44100)
    try:
        y, sr = librosa.load(hi_path, sr=None, mono=True)
        seg = y[: int(sr * 90)]
        # Magnitude share (what an onset envelope responds to), not power.
        S_ = np.abs(librosa.stft(seg, n_fft=2048))
        fr = librosa.fft_frequencies(sr=sr, n_fft=2048)
        tot = float(S_.sum())
        hf = float(S_[fr > 11025.0].sum()) / tot if tot > 0 else 0.0
        obj_hi = float(np.atleast_1d(librosa.feature.tempo(y=y, sr=sr, start_bpm=120.0, std_bpm=30.0))[0])
        ess_hi = None
        if ESSENTIA_AVAILABLE:
            try:
                _es, _esr = _essentia_load_audio(hi_path, ESSENTIA_MAX_SECONDS)
                ess_hi, _b = _essentia_tempo_and_beats(_es, _esr) if _es is not None and len(_es) else (None, None)
                ess_hi = float(ess_hi) if ess_hi else None
            except Exception:
                ess_hi = None
        h = _full_band_tempo_decision(lows, obj_hi, ess_hi, hf)
        if h is None:
            status_done(f"Full-bandwidth tempo check: no change ({obj_hi:.0f}/{ess_hi or 0:.0f} BPM, "
                        f"{100 * hf:.1f}% of energy above 11 kHz)")
            return False
        new_obj = build_objective_audio_report(hi_path, original_path=source_path)
        new_ess = build_essentia_report(hi_path) if out.get("essentia_report") else ""
        nobj = extract_objective_bpm(new_obj)
        if not nobj or _tempo_multiple(nobj, h) != 1:
            status_done("Full-bandwidth tempo check: rebuilt reading did not confirm; no change")
            return False
        low_s = " and ".join(f"{v:g}" for v in lows)
        detector_note = ("both detectors put" if ess_hi is not None else
                         "the available librosa detector puts")
        note = (f"{_FULL_BAND_TEMPO_HEAD} the usual 22 kHz analysis copy read the tempo as {low_s} BPM, "
                f"but {100 * hf:.0f}% of this track's energy (hi-hats and other bright percussion) is above "
                f"11 kHz, which that copy drops. Measured again on the full-bandwidth audio, {detector_note} "
                f"the beat at about {h:.0f} BPM -- the slow readings were the same pulse counted every "
                "2 or 4 beats. The beat and tempo figures below come from the full-bandwidth audio.\n")
        out["objective_report"] = note + new_obj
        if new_ess:
            out["essentia_report"] = new_ess
        status_done(f"Full-bandwidth tempo check: {low_s} -> {h:.0f} BPM")
        return True
    finally:
        try:
            os.remove(hi_path)
        except OSError:
            pass
