# MUSICLYSE FILE pipeline-vocals 0.8
# ==============================================================================
#  MUSICLYSE 0.8: VOCALS
# ==============================================================================
# This file holds vocals: whether there is singing, vocal pitch and voice-type measurements,
# vocal tags and the vocal checks. It is part of musiclyse-0.8.py, which loads it whole at
# start-up, after its own code, into its own namespace (see "THE MUSICLYSE FILES" there); it is
# not run on its own.
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
# (SAVE_DIR moved to the settings file beside this script.)

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
    # "gender_uncertain" used to map to child_gender_uncertain. It says
    # nothing about age: an AI-generated test track's vocal pass wrote "LEAD PROFILE: gender
    # uncertain" for an adult-sounding pop singer and the alias turned that
    # into a CHILD lead, which then primed the confirmation pass into
    # agreeing. Left unmapped, so the LEAD_CATEGORY line decides instead.
    "adolescent_male": "adolescent_male_likely",
    "adolescent": "adolescent_male_likely",
    "transitional_male": "adolescent_male_likely",
    "post_pubertal_male": "post_puberty_male",
    "postpuberty_male": "post_puberty_male",
    "adult_male": "post_puberty_male",
    "female": "female_teen_adult",
    "adult_female": "female_teen_adult",
    "young_female": "female_teen_adult",
    # --- hedged spellings -------------------------------------------------
    # The canonical taxonomy itself uses "_likely" endings
    # (child_male_likely, adolescent_male_likely), so the vocal model
    # routinely generalises the pattern to categories that never had one.
    # None of these were recognised, so they resolved to "unknown", which
    # silently disabled BOTH age guards (they only act on known male/child
    # tags) and handed the entire decision to the identity pass.
    "adult_male_likely": "post_puberty_male",
    "post_puberty_male_likely": "post_puberty_male",
    "adult_female_likely": "female_teen_adult",
    "female_teen_adult_likely": "female_teen_adult",
    "male_likely": "post_puberty_male",
    "female_likely": "female_teen_adult",
    "likely_male": "post_puberty_male",
    "likely_female": "female_teen_adult",
    "young_male_likely": "adolescent_male_likely",
    "young_female_likely": "female_teen_adult",
    "teen_male_likely": "adolescent_male_likely",
    "teenage_male": "adolescent_male_likely",
    "child_likely": "child_gender_uncertain",
    # "female" was aliased but "male" was not -- a latent asymmetry that sent
    # a bare male lead to "unknown" while a bare female lead resolved cleanly.
    "male": "post_puberty_male",
}

VOCAL_CORRECTION_FIELDS = {
    "singer",
    "lead_singer",
    "lead_vocal",
    "vocal_lead",
    "voice",
}


def _vocal_coverage_by_span(stem_notes, spans):
    """Fraction of each span covered by transcribed vocal notes.

    Returns a list parallel to `spans`, or None when there is no usable vocal
    stem — in which case callers must fall back to their previous behaviour
    rather than assuming silence, since "no vocal data" and "no singing" are
    very different claims.

    Accepts both note shapes used in this file: the in-memory form
    {"onset": s, "offset": s} and the saved compact form {"t": s, "d": s}.
    """
    if not stem_notes or not isinstance(stem_notes, dict) or not spans:
        return None

    notes = None
    for key in ("vocals", "vocal", "voice"):
        if stem_notes.get(key):
            notes = stem_notes[key]
            break
    if not notes:
        return None

    intervals = []
    for n in notes:
        if not isinstance(n, dict):
            continue
        try:
            if "onset" in n:
                start = float(n.get("onset", 0.0))
                end = float(n.get("offset", start))
            else:
                start = float(n.get("t", 0.0))
                end = start + float(n.get("d", 0.0))
            if end > start:
                intervals.append((start, end))
        except Exception:
            continue
    if not intervals:
        return None

    intervals.sort()
    # Merge overlaps so simultaneous/overlapping notes are not double-counted;
    # coverage must be a fraction of TIME, not a sum of note durations.
    merged = [list(intervals[0])]
    for s, e in intervals[1:]:
        if s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])

    out = []
    for (a, b) in spans:
        try:
            a = float(a)
            b = float(b)
            width = b - a
            if width <= 0:
                out.append(0.0)
                continue
            covered = 0.0
            for s, e in merged:
                if e <= a:
                    continue
                if s >= b:
                    break
                covered += min(b, e) - max(a, s)
            out.append(max(0.0, min(1.0, covered / width)))
        except Exception:
            out.append(0.0)
    return out


_LVC_NO_VOICE_RE = re.compile(
    r"(?i)^\s*(?:none|no\b|n/?a\b|not applicable|no (?:lead )?vocals?|instrumental)")


def _drop_no_vocal_sentences(text):
    """`text` without its sentences that say the recording has no voice
    (_INSTRUMENTAL_PROSE_RE: "absence of vocals", "there are no vocals",
    "an instrumental track" ...)."""
    t = str(text or "")
    if not _INSTRUMENTAL_PROSE_RE.search(t):
        return t
    parts = re.split(r"(?<=[.!?])\s+", t)
    return " ".join(x for x in parts if not _INSTRUMENTAL_PROSE_RE.search(x)).strip()


def _report_unsettled_vocal_fields(head, fields, tail):
    """(head, fields) with the report's VOCALS_PRESENT set to "uncertain" when
    it is neither yes nor no ("uncertain (vocal texture present but lyrics
    unclear)", "high (lead and backing vocals present throughout)", "mostly
    yes") and the vocal measurements elsewhere in the notes settle it, so the
    fill below writes the settled answer. When the voice is settled as
    present, a LEAD_VOCAL_CHARACTERISTICS of "none (no lead vocal)" is set to
    "uncertain" the same way, to be filled from the model's own description
    or left uncertain -- never left saying there is no voice."""
    try:
        # A run of fields with one answer pasted into each (_moss_copied_fields)
        # is not an answer for any of them.
        _items = [(k, v_) for k, v_ in fields.items() if k not in ("REPORT_NOTE",)]
        _copied = _moss_copied_fields(_items) | _moss_no_content_fields(_items)
        for _n in _copied:
            if _n == "DESCRIPTION":
                head = re.sub(r"(?m)^DESCRIPTION=.*$", "DESCRIPTION=(none given)", head, count=1)
                fields = dict(fields, DESCRIPTION="(none given)")
            elif _n != "STRUCTURE":
                head = re.sub(r"(?m)^" + re.escape(_n) + r"=.*$", _n + "=uncertain", head, count=1)
                fields = dict(fields, **{_n: "uncertain"})
        _meas = _fill_measured(tail)
        # MELODIC_RANGE / MELODIC_CONTOUR: the measured vocal range and the
        # per-section vocal contour, whenever they exist, as in every scan
        # the previous listening model made (it left both out). MOSS-Music
        # writes both, mostly by repeating the prompt's examples ("C4-G5",
        # "stepwise ascending in verse, leap up in chorus" on six of
        # eighteen scans); its answers are kept only where nothing was measured.
        for _n in ("MELODIC_RANGE", "MELODIC_CONTOUR"):
            _cur = str(fields.get(_n, "")).strip()
            if _meas.get(_n) and _cur and _cur != "uncertain" and _cur != _meas[_n]:
                head = re.sub(r"(?m)^" + _n + r"=.*$", _n + "=uncertain", head, count=1)
                fields = dict(fields, **{_n: "uncertain"})
        # KEY / TEMPO_BPM: the script's decided key and tempo (RECOMMENDED KEY
        # / TEMPO FOR DISCUSSION) when the report's own value disagrees.
        # MOSS-Music writes its own here -- "Eb major (confidence: high)" or
        # "Ab minor" on most tracks, or the prompt's instruction text ("Am
        # (best-supported tonal centre ...") -- which contradicted the decided
        # values and was repeated in DESCRIPTION.
        _k = str(fields.get("KEY", "")).strip()
        if _meas.get("KEY") and _k and _k != "uncertain":
            _kn = _normalize_key_for_compare(_k) or _normalize_key_for_compare(
                (re.search(r"\b([A-G](?:#|b|♯|♭)?m?(?:\s*(?:major|minor))?)\b", _k) or [None, ""])[1])
            if _kn != _normalize_key_for_compare(_meas["KEY"]):
                head = re.sub(r"(?m)^KEY=.*$", "KEY=uncertain", head, count=1)
                fields = dict(fields, KEY="uncertain")
        _tv = str(fields.get("TEMPO_BPM", "")).strip()
        if _meas.get("TEMPO_BPM") and _tv and _tv != "uncertain":
            _tm = re.search(r"\d+(?:\.\d+)?", _tv)
            try:
                if not _tm or abs(float(_tm.group(0)) - float(_meas["TEMPO_BPM"])) > 2.0:
                    head = re.sub(r"(?m)^TEMPO_BPM=.*$", "TEMPO_BPM=uncertain", head, count=1)
                    fields = dict(fields, TEMPO_BPM="uncertain")
            except ValueError:
                pass
        # DRUM STEM NOTE: there is no kit, whatever the report's DRUMS and
        # INSTRUMENTATION say (MOSS-Music described "soft kick ... brushed
        # snare" on an acoustic guitar track).
        if "DRUM STEM NOTE:" in tail:
            _dr = str(fields.get("DRUMS", "")).strip()
            if _dr and not re.match(r"(?i)^none\b", _dr):
                head = re.sub(r"(?m)^DRUMS=.*$", "DRUMS=none (no drum kit; see DRUM STEM NOTE)", head, count=1)
                fields = dict(fields, DRUMS="none (no drum kit; see DRUM STEM NOTE)")
            _ins = str(fields.get("INSTRUMENTATION", ""))
            if _ins and "\n" not in _ins:
                _items = [x for x in _ins.split(";") if x.strip()]
                _keep = [x for x in _items if not re.match(
                    r"(?i)^\s*(?:\(?\d+\)?\s*)?(?:drums?|drum\s*kit|percussion|kick|snare|hi-?hats?)\b", x)]
                if _keep and len(_keep) < len(_items):
                    _new = ";".join(_keep).strip()
                    head = re.sub(r"(?m)^INSTRUMENTATION=.*$", lambda _m: "INSTRUMENTATION=" + _new, head, count=1)
                    fields = dict(fields, INSTRUMENTATION=_new)
        # GUITAR CHECK: acoustic guitar throughout and no electric reading
        # anywhere -- an "electric guitar" item is the listening model's guess.
        if ACOUSTIC_GUITAR_CHECK and _acoustic_only_guitar(tail):
            _ins = str(fields.get("INSTRUMENTATION", ""))
            if _ins and "\n" not in _ins:
                _items = [x for x in _ins.split(";") if x.strip()]
                _el = re.compile(r"(?i)^\s*(?:\(?\d+\)?\s*)?electric[ -]?guitars?\b")
                _keep = [(" " if i else "") + _SEMI_ACOUSTIC_ITEM if _el.match(x) else x
                         for i, x in enumerate(_items)]
                if _keep != _items:
                    _new = ";".join(_keep).strip()
                    head = re.sub(r"(?m)^INSTRUMENTATION=.*$", lambda _m: "INSTRUMENTATION=" + _new, head, count=1)
                    fields = dict(fields, INSTRUMENTATION=_new)
        vp_meas = _meas.get("VOCALS_PRESENT")
        v = str(fields.get("VOCALS_PRESENT", "")).strip()
        if vp_meas and v and v != "uncertain" and not re.match(r"(?i)^(?:yes|no)\b", v):
            head = re.sub(r"(?m)^VOCALS_PRESENT=.*$", "VOCALS_PRESENT=uncertain", head, count=1)
            fields = dict(fields, VOCALS_PRESENT="uncertain")
            v = "uncertain"
        voiced = vp_meas == "yes" or (not vp_meas and re.match(r"(?i)^yes\b", v))
        lvc = str(fields.get("LEAD_VOCAL_CHARACTERISTICS", "")).strip()
        if voiced and lvc and lvc != "uncertain" and (_LVC_NO_VOICE_RE.match(lvc)
                                                      or _INSTRUMENTAL_PROSE_RE.search(lvc)):
            head = re.sub(r"(?m)^LEAD_VOCAL_CHARACTERISTICS=.*$", "LEAD_VOCAL_CHARACTERISTICS=uncertain",
                          head, count=1)
            fields = dict(fields, LEAD_VOCAL_CHARACTERISTICS="uncertain")
    except Exception:
        pass
    return head, fields


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
    # Second pass: drop points more than an octave from the median, but only
    # when that's clearly a small minority of stray frames (isolated
    # octave-tracking/harmonic errors). A genuinely wide-ranged performance --
    # a chest-voice verse against a falsetto/whistle chorus, for example --
    # can legitimately put a large share of frames beyond one octave from the
    # median; that's real singing, not tracking noise, and this pass used to
    # discard it unconditionally, directly undercutting the wider IQR fence
    # above (whose whole point was to preserve genuine excursions). Requiring
    # the within-octave frames to be a clear majority keeps the original
    # behavior for the common single-register case (where few or no frames
    # sit outside one octave, so nothing changes) while no longer amputating
    # the top/bottom of a real multi-register range.
    med = float(np.median(cleaned))
    if med > 0:
        ratio = cleaned / med
        within_octave = (ratio >= 0.5) & (ratio <= 2.0)
        if within_octave.sum() >= 0.85 * cleaned.size:
            cleaned = cleaned[within_octave]
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
        frames_scanned = 0

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
            frames_scanned += int(len(f0))
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
            if frames_scanned > 0:
                # How much of the scanned audio was confidently pitched at all.
                # A real vocal part is voiced for a large share of the time;
                # separation residue on an instrumental is not.
                lines.append(
                    f"voiced frames: {round(100.0 * len(f0_values) / frames_scanned, 1)}% "
                    "of scanned audio"
                )
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

                # Formant-like peaks are only meaningful when a voice was found.
                if peak_ref > 0 and f0_values:
                    # Distinct spectral PEAKS at least ~250 Hz apart. Taking the
                    # four largest bins reported neighbouring FFT bins of one
                    # peak (e.g. 322 Hz and 333 Hz) as separate "formants".
                    try:
                        from scipy.signal import find_peaks as _find_peaks
                        bin_hz = float(freqs[1] - freqs[0]) if len(freqs) > 1 else 1.0
                        pk, _ = _find_peaks(valid_mean, distance=max(1, int(250.0 / max(bin_hz, 1e-6))))
                        pk = [int(i) for i in pk if valid_formant_band[int(i)] and valid_mean[int(i)] > 0]
                        top_idx = sorted(pk, key=lambda i: valid_mean[i], reverse=True)[:4]
                    except Exception:
                        top_idx = []
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
# ---------------------------------------------------------------------------


def _normalize_vocal_tag(value):
    value = (value or "").lower().strip()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_")


# 0.7.1-pre12 VOCALS_FLAG_FALLBACK_STRICT (F-26)
def _vocals_absence_is_qualified(low):
    """True when the reply's "no vocals" wording is about part of the song or one kind of
    voice, or the reply also describes a singer."""
    if re.search(r"\bno (?:clear )?vocals?(?: detected)?\s+(?:harmon|in the|during|until|before|after|on the|"
                 r"for the|at the|backing|layers?|doubl|effects?)", low):
        return True
    if re.search(r"\bno (?:clear )?vocal (?:harmon|backing|layer|doubl|effect|processing)", low):
        return True
    return bool(re.search(r"\b(?:lead (?:vocal|vocalist|singer|voice)s?|(?:male|female) (?:lead|vocal|voice|singer)s?|"
                          r"vocalist|sings|sung)\b", low))



def _parse_vocals_present_flag(text):
    """Parse VOCALS_PRESENT from listening-model output. Accepts underscore or space forms
    and common synonyms. Returns 'yes'|'no'|'uncertain'|'' (unparsed)."""
    if not text:
        return ""
    text = _strip_md_emphasis(text)
    m = re.search(
        # Also accepts the "VOCAL PRESENCE: yes" spelling the listening model often
        # uses, which this regex used to miss entirely (flag left unparsed).
        r"\bVOCALS?[_\s]?PRESEN(?:T|CE)\s*[=:\-–—]?\s*(yes|no|uncertain|instrumental|true|false)",
        text,
        re.IGNORECASE,
    )
    if not m:
        # Free-text fallbacks common in failed short-clip passes
        low = text.lower()
        # "instrumental" only counts when it describes the whole track: "an
        # instrumental intro / break / solo / backing" says nothing about
        # whether there is a singer (the rock test track's description pass
        # opens "an instrumental intro establishes the guitar motif").
        if re.search(
            r"\b(?:no clear(?:ly)? (?:singable )?lead|no (?:clear )?vocals?(?: detected)?|"
            r"instrumental(?![\s\-]+(?:only[\s\-]+)?(?:intro|introduction|opening|break|breakdown|"
            r"bridge|section|passage|solo|outro|coda|interlude|hook|riff|motif|part|backing|"
            r"backdrop|bed|arrangement|accompaniment|layer|element|line|tag|fill|build|"
            r"lead-in|verse|chorus|stretch|texture|groove|figure|theme|melod(?:y|ies)|"
            r"segment|moment|space|gap|pause|swell|lick)(?:e?s)?\b)(?:\s+only)?|"
            r"no human voice)\b",
            low,
        ):
            if not re.search(r"\bVOCALS[_\s]?PRESENT\s*[=:\-–—]?\s*yes", text, re.I):
                if VOCALS_FLAG_FALLBACK_STRICT and _vocals_absence_is_qualified(low):
                    return ""                 # 0.7.1-pre12: "no vocal harmonies", "no vocals in the intro"
                return "no"
        return ""
    val = m.group(1).strip().lower()
    if val in ("yes", "true"):
        return "yes"
    if val in ("no", "instrumental", "false"):
        return "no"
    return "uncertain"


def _vocal_report_pitch_is_voice_evidence(vocal_objective_report):
    """Can this report's pitch reading stand as evidence of a human voice?

    No when it was measured on the full mix (pyin there tracks whatever is
    loudest and pitched -- a pad, a bass, a guitar), or when only a sliver of
    the scanned audio was voiced. Reports without either line (older saved
    analyses) keep the previous behaviour.
    """
    rep = str(vocal_objective_report or "").lower()
    if "measurement source: full mix" in rep:
        return False
    m = re.search(r"voiced frames:\s*([0-9]+(?:\.[0-9]+)?)\s*%", rep)
    if m:
        try:
            if float(m.group(1)) / 100.0 < float(VOCAL_PITCH_EVIDENCE_MIN_VOICED_FRACTION):
                return False
        except ValueError:
            pass
    return True


def _objective_evidence_of_vocals(median_f0=None, vocal_objective_report="", stem_midi_report=""):
    """True when DSP/stems independently suggest a human voice even if the listening
    model's vocal pass said no vocals / instrumental.

    A pitch reading only counts when it came from the isolated vocal stem and
    enough of the audio was voiced (see _vocal_report_pitch_is_voice_evidence).
    Previously ANY median f0 between 70 and 1100 Hz -- i.e. any pitched sound,
    including a synth pad measured on the full mix -- counted as a voice, and so
    did any line mentioning notes in the vocals stem, which on an instrumental
    is just whatever the separator routed there. Neither is evidence of a singer.
    """
    pitch_ok = _vocal_report_pitch_is_voice_evidence(vocal_objective_report)
    if pitch_ok:
        try:
            if median_f0 is not None:
                f0 = float(median_f0)
                # Human sung range (rough); exclude obvious noise floors
                if 70.0 <= f0 <= 1100.0:
                    return True
        except (TypeError, ValueError):
            pass
        rep = (vocal_objective_report or "").lower()
        if re.search(r"voiced pitch median:\s*[1-9]", rep):
            return True
    return False


# A sentence in the main listening pass that states the track has no vocal.
# Deliberately narrow: "an instrumental intro/break/solo" must NOT match.
_INSTRUMENTAL_PROSE_RE = re.compile(
    r"(?i)\b(?:"
    r"(?:this|the)\s+(?:piece|track|song|recording|composition|tune|cut)\s+is\s+"
    r"(?:an?\s+|entirely\s+|purely\s+|fully\s+|wholly\s+)*instrumental\b"
    r"|(?:an?|purely|entirely|fully)\s+instrumental\s+(?:[a-z\-]+\s+){0,2}"
    r"(?:piece|track|song|recording|composition|number|work)\b"
    r"|there\s+(?:are|is)\s+no\s+(?:lead\s+)?vocals?\b"
    r"|no\s+(?:lead\s+)?vocals?\s+(?:are\s+|is\s+)?(?:present|audible|detected)"
    r"|(?:has|contains|features)\s+no\s+(?:lead\s+)?vocals?\b"
    r"|without\s+(?:any\s+)?(?:lead\s+)?vocals?\b"
    r"|absence\s+of\s+(?:vocals?|lyrics|a\s+vocal|singing)"
    r")"
)

VOCAL_ARBITRATION_MARKER = "VOCALS_PRESENT=no  # from vocal-presence arbitration"


def _main_pass_says_instrumental(text):
    """True when the main listening pass states the recording has no vocal --
    an explicit VOCALS_PRESENT=no field, or a clear prose statement."""
    t = _strip_md_emphasis(str(text or ""))
    # A field the model pasted into a run of fields is not an answer (see
    # _moss_copied_fields): on the post-punk test track and another test track "none (...)" had been
    # copied into VOCALS_PRESENT and every field around it.
    t = _moss_drop_copied_field_lines(t)
    m = re.search(
        r"\bVOCALS?[_\s]?PRESEN(?:T|CE)\s*[=:\-–—]?\s*(yes|no|uncertain|instrumental|true|false)",
        t, re.IGNORECASE,
    )
    if m:
        # An answer limited to a stretch of time ("none (0.00s-13.92s)": no
        # voice in the intro) is not a claim about the whole recording.
        rest = t[m.end():].split("\n", 1)[0]
        if _VOCAL_ANSWER_TIME_SPAN_RE.search(rest):
            return False
        return m.group(1).lower() in ("no", "instrumental", "false")
    return bool(_INSTRUMENTAL_PROSE_RE.search(t))


_VOCAL_ANSWER_TIME_SPAN_RE = re.compile(
    r"\d+(?:\.\d+)?\s*s\s*[-–—]\s*\d+(?:\.\d+)?\s*s\b|\b\d{1,2}:\d{2}\b")


def _voiced_fraction_from_report(vocal_objective_report):
    m = re.search(r"voiced frames:\s*([0-9]+(?:\.[0-9]+)?)\s*%", str(vocal_objective_report or ""), re.I)
    try:
        return float(m.group(1)) / 100.0 if m else None
    except ValueError:
        return None


def _stem_level_vs_mix_db(stem_path, mix_path):
    """RMS of a stem relative to the mix, in dB (negative = below). None on
    any failure. Streams both files so memory stays flat."""
    def _rms(path):
        tot, n = 0.0, 0
        try:
            import soundfile as _sf
            for block in _sf.blocks(str(path), blocksize=262144, always_2d=True):
                b = np.asarray(block, dtype=np.float64).mean(axis=1)
                tot += float(np.dot(b, b)); n += b.size
        except Exception:
            try:
                y, _sr = librosa.load(str(path), sr=None, mono=True)
                tot, n = float(np.dot(y.astype(np.float64), y.astype(np.float64))), int(y.size)
            except Exception:
                return None
        return (tot / n) ** 0.5 if n else None
    if not stem_path or not mix_path:
        return None
    a, b = _rms(stem_path), _rms(mix_path)
    if not a or not b:
        return None
    return float(20.0 * np.log10(a / b))


# A person, not an instrument that "sings": "the trumpet sings" never counts.
_HEARD_VOICE_RE = re.compile(
    r"(?i)\b(?:vocals?|vocalist|singer|rapper|crooner|"
    r"baritone|tenor|soprano|mezzo(?:-soprano)?|contralto)\b")
# The words, not "lyrical phrasing" on a saxophone.
_HEARD_LYRICS_RE = re.compile(
    r"(?i)\blyrics?\b|\blyrical\s+(?:content|themes?|subject|focus|narrative|imagery)\b")
_HEARD_NEGATION_RE = re.compile(
    r"(?i)\b(?:no|not|without|absence|absent|lacks?|lacking|instrumental|wordless|devoid)\b|n't\b")
_HEARD_QUOTE_RE = re.compile(r"[“\"]([^”\"\n]{6,160})[”\"]")


def _other_pass_heard_singing(other_text, vocal_result):
    """dev73: True when a SEPARATE listening pass (the era pass's free
    description) plainly heard a singer: the dedicated vocal pass answered
    that a voice is present, and the other pass has a sentence naming a
    singer and a sentence about the lyrics (neither negated) and quotes at
    least two sung lines of three or more words.

    One half-spoken big-band test scan is sung, half-spoken, over a
    big band. The main pass called it instrumental and the vocal stem was
    confidently pitched in only 1.5% of the audio (a near-spoken delivery
    under horns), so the arbitration declared it instrumental -- while the
    era pass described "a male baritone" and quoted four lines it had heard
    (misheard, so not copied from the file's lyric tag, which no listening
    pass is given). Audio only: tags, lyrics files and identity play no part."""
    t = _strip_md_emphasis(str(other_text or ""))
    if not t or _parse_vocals_present_flag(vocal_result or "") != "yes":
        return False
    if _INSTRUMENTAL_PROSE_RE.search(t):     # it also calls the track instrumental
        return False
    sents = re.split(r"(?<=[.!?])\s+", t)
    voice = any(_HEARD_VOICE_RE.search(s) and not _HEARD_NEGATION_RE.search(s) for s in sents)
    lyrics = any(_HEARD_LYRICS_RE.search(s) and not _HEARD_NEGATION_RE.search(s) for s in sents)
    quotes = [q for q in _HEARD_QUOTE_RE.findall(t) if len(q.split()) >= 3]
    return bool(voice and lyrics and len(quotes) >= 2)


def _arbitrate_vocal_presence(main_text, vocal_result, vocal_objective_report,
                              vocal_stem_path=None, mix_path=None, other_text=""):
    """Decide whether the recording has a voice at all, before anything builds
    a singer profile on top of the answer.

    The main listening pass and the dedicated vocal pass were never reconciled:
    on an instrumental the main pass said "the piece is instrumental" while the
    vocal pass -- asked only about voices -- answered "VOCAL PRESENCE: yes,
    child, high", and that answer then fed the f0 duet upgrade and singer
    identity ("mixed_leads"). This arbitration only ever turns a vocal claim
    OFF, and only with an isolated vocal stem to measure:

      - the main pass says instrumental AND the vocal stem is weak (voiced in
        under VOCAL_PITCH_EVIDENCE_MIN_VOICED_FRACTION of the audio, no usable
        pitch reading, or >= VOCAL_ARBITRATION_STEM_BELOW_MIX_DB below the mix); or
      - the vocal stem is >= VOCAL_ARBITRATION_STRONG_BELOW_MIX_DB below the mix
        and almost never voiced, whatever the listening passes said.

    Returns a dict: instrumental (bool), reasons (list), level_db, voiced.
    """
    out = {"instrumental": False, "reasons": [], "level_db": None, "voiced": None,
           "main_says_instrumental": False}
    if not vocal_stem_path or not os.path.exists(str(vocal_stem_path)):
        return out          # nothing measured: leave the listening passes alone
    main_instr = _main_pass_says_instrumental(main_text)
    voiced = _voiced_fraction_from_report(vocal_objective_report)
    pitch_ok = (
        _vocal_report_pitch_is_voice_evidence(vocal_objective_report)
        and bool(re.search(r"voiced pitch median:\s*[1-9]", str(vocal_objective_report or ""), re.I))
    )
    level_db = _stem_level_vs_mix_db(vocal_stem_path, mix_path) if mix_path else None
    out.update(level_db=level_db, voiced=voiced, main_says_instrumental=main_instr)

    weak = []
    if voiced is not None and voiced < float(VOCAL_PITCH_EVIDENCE_MIN_VOICED_FRACTION):
        weak.append(f"the vocal stem is confidently pitched in only {voiced * 100:.1f}% of the scanned audio")
    elif not pitch_ok:
        weak.append("no usable voiced pitch in the vocal stem")
    _below = float(VOCAL_ARBITRATION_STEM_BELOW_MIX_DB)
    if FAINT_VOCAL_TEXTURE_CHECK:
        _below = min(_below, float(VOCAL_ARBITRATION_FAINT_BELOW_MIX_DB))
    if level_db is not None and level_db <= -_below:
        weak.append(f"the vocal stem sits {abs(level_db):.0f} dB below the mix")
    strong = (
        level_db is not None and level_db <= -float(VOCAL_ARBITRATION_STRONG_BELOW_MIX_DB)
        and (voiced is None or voiced < float(VOCAL_ARBITRATION_STRONG_MAX_VOICED))
    )
    # dev73: another listening pass heard a singer and quoted the lyrics, so
    # the main pass's "instrumental" is contradicted and cannot decide on its
    # own; only the measured "strong" case below still can.
    if main_instr and VOCAL_ARBITRATION_HEARD_LYRICS_GUARD and other_text:
        try:
            if _other_pass_heard_singing(other_text, vocal_result):
                out["other_pass_heard_singing"] = True
                main_instr = False
        except Exception:
            pass
    # The main pass's "instrumental" against the dedicated vocal pass's "yes"
    # cannot be settled by a low pitched-frame share alone when the vocal stem
    # is loud: real voices in these scans sit 2-9 dB below the mix and are
    # confidently pitched in as little as 0.2-6% of frames (rough or
    # consonant-heavy singing under a band), while separation residue on an
    # instrumental sits far lower. MOSS-Music's main pass has called sung
    # tracks instrumental ("Despite the absence of vocals" on a duo test track,
    # while its own melody notes described "vocal lines ... catchy and
    # singable"). Only the measured "strong" case can still decide then.
    if (main_instr and VOCAL_ARBITRATION_LOUD_STEM_GUARD and level_db is not None
            and level_db > -float(VOCAL_ARBITRATION_LOUD_STEM_DB)
            and _parse_vocals_present_flag(vocal_result or "") == "yes"):
        out["loud_stem_guard"] = True
        main_instr = False
    if (main_instr and weak) or strong:
        out["instrumental"] = True
        if main_instr:
            out["reasons"].append("the main listening pass describes the track as instrumental")
        out["reasons"].extend(weak)
    return out


def _faint_vocal_texture_reasons(arb, lyrics_text):
    """Reasons to rule a track instrumental after the lyrics pass: the vocal
    stem sits VOCAL_ARBITRATION_FAINT_BELOW_MIX_DB+ below the mix and Whisper
    transcribed fewer than FAINT_VOCAL_MAX_WHISPER_WORDS words. Else None.
    Only called when Whisper ran and there are no tagged lyrics."""
    if not FAINT_VOCAL_TEXTURE_CHECK or not isinstance(arb, dict):
        return None
    level = arb.get("level_db")
    if level is None or level > -float(VOCAL_ARBITRATION_FAINT_BELOW_MIX_DB):
        return None
    words = len(re.findall(r"[A-Za-z\u00C0-\u024F']+", str(lyrics_text or "")))
    if words >= int(FAINT_VOCAL_MAX_WHISPER_WORDS):
        return None
    return [f"the vocal stem sits {abs(level):.0f} dB below the mix -- far quieter than a sung lead",
            f"the lyrics pass could transcribe almost nothing from the track ({words} word{'s' if words != 1 else ''})"]


def _vocal_arbitration_block(arb, vocal_result=""):
    """Evidence text for an instrumental verdict. When the vocal pass claimed a
    voice, its output is kept only as a marked false positive; when it agreed
    (VOCALS_PRESENT=no) it is shown as agreeing."""
    reasons = "; ".join(arb.get("reasons") or []) or "the measurements show no voice"
    vr = _collapse_repeated_line_blocks(vocal_result or "").strip()
    agreed = bool(vr) and _parse_vocals_present_flag(vr) == "no"
    if agreed:
        tail = ("The dedicated vocal pass agrees. Treat this recording as instrumental: do not "
                "describe a singer, a vocal range, lyrics or a duet.")
    else:
        tail = ("The dedicated vocal pass's 'vocal present' answer, any singer profile, age/gender, "
                "pitch or range figures, multi-singer reading and audio-transcribed lyrics are a "
                "false positive and are SUPERSEDED. Treat this recording as instrumental: do not "
                "describe a singer, a vocal range, lyrics or a duet.")
    residue = ("Whatever the vocals stem holds is separation residue (instrument bleed). " if not arb.get("texture")
               else "Faint vocal-like sounds (a sample, or a chopped or processed voice used as texture) may "
                    "be audible in the background, but there is no sung lead: mention them, if at all, only "
                    "as a background texture. ")
    txt = (
        "\n\nVOCAL PRESENCE ARBITRATION: INSTRUMENTAL. " + reasons[:1].upper() + reasons[1:] + ". "
        + residue + tail
        + "\n" + VOCAL_ARBITRATION_MARKER
    )
    if vr:
        label = ("[agrees: no voice]" if agreed else
                 "[SUPERSEDED — listening-model false positive; see VOCAL PRESENCE ARBITRATION]")
        txt += "\n\nVOCAL / SINGER PROFILE (isolated dedicated pass) " + label + ":\n" + vr
    return txt


def _stem_onset_vocal_salvage(vocal_result, revised, stem_midi_report):
    """Stem notes cannot establish human vocal presence: instruments leak in.
    Kept as a no-op for existing callers; actual voices use listening evidence.
    """
    return ""


def _salvage_false_no_vocals(
    vocal_result,
    no_clear_vocals,
    median_f0=None,
    vocal_objective_report="",
    stem_midi_report="",
    initial_lead="",
):
    """If the listening model claimed no vocals but objective pitch/stem evidence says otherwise,
    clear the no_clear_vocals flag and return a salvage note + adjusted flags.

    Returns (no_clear_vocals, salvage_note, forced_present_yes).
    """
    present = _parse_vocals_present_flag(vocal_result or "")
    if present == "yes":
        return False, "", True
    objective = _objective_evidence_of_vocals(
        median_f0=median_f0,
        vocal_objective_report=vocal_objective_report,
        stem_midi_report=stem_midi_report,
    )
    # Also salvage when a structured LEAD_PROFILE was parsed despite a free-text "no"
    lead_ok = initial_lead in VOCAL_LEAD_TAGS and initial_lead not in ("", "unknown")
    if no_clear_vocals or present == "no":
        if objective or lead_ok:
            note = (
                "VOCAL PRESENCE SALVAGE: MOSS-Music's vocal pass reported no/unclear vocals, "
                "but independent evidence indicates a human voice is present"
            )
            if median_f0 is not None:
                note += f" (objective voiced pitch median ~{median_f0} Hz)"
            if lead_ok:
                note += f" (structured LEAD_PROFILE={initial_lead})"
            note += (
                ". Treat VOCALS_PRESENT=yes. Demographic age/gender may still be uncertain. "
                "Do NOT tell the user there is no singable lead or that the track is instrumental "
                "solely because lyric transcription failed or the clip is short."
            )
            return False, note, True
    return bool(no_clear_vocals), "", present == "yes"


def _discard_vocal_choice_lists(text):
    text = _strip_md_emphasis(text)
    return "\n".join(line for line in text.splitlines()
                     if not (re.match(r"\s*(?:LEAD_PROFILE|LEAD_CATEGORY|GENDER_MODIFIER)\s*[=:]", line, re.I)
                             and "|" in line.split("=", 1)[-1].split(":", 1)[-1]))


def parse_vocal_tags(text):
    text = _discard_vocal_choice_lists(text)
    lead = ""
    backing = ""
    # Accept "=" or ":" — the listening model often emits "LEAD_PROFILE: ..." instead of "=".
    m = re.search(r'LEAD_PROFILE\s*[=:]\s*["\'<\[]?\s*([A-Za-z0-9_\- ]+)', text or "", re.IGNORECASE)
    if m:
        lead = _resolve_lead_tag_variant(m.group(1))

    if lead not in VOCAL_LEAD_TAGS:
        cat_m = re.search(r'LEAD_CATEGORY\s*[=:]\s*["\'<\[]?\s*([A-Za-z0-9_\- ]+)', text or "", re.IGNORECASE)
        mod_m = re.search(r'GENDER_MODIFIER\s*[=:]\s*["\'<\[]?\s*([A-Za-z0-9_\- ]+)', text or "", re.IGNORECASE)
        if cat_m and mod_m:
            lead = _lead_from_category_modifier(cat_m.group(1), mod_m.group(1))
        elif cat_m:
            lead = _lead_from_category_modifier(cat_m.group(1), "")
        if lead not in VOCAL_LEAD_TAGS and cat_m:
            # The same hedge-spelling problem appears on LEAD_CATEGORY.
            lead = _resolve_lead_tag_variant(cat_m.group(1)) or lead

    lead = VOCAL_LEAD_ALIASES.get(lead, lead)
    if lead not in VOCAL_LEAD_TAGS:
        lead = "unknown"

    m = re.search(r'BACKING_PROFILES\s*[=:]\s*["\'<\[]?\s*([A-Za-z0-9_\- ]+)', text or "", re.IGNORECASE)
    if m:
        backing = _normalize_vocal_tag(m.group(1))
    if backing not in VOCAL_BACKING_TAGS:
        backing = "uncertain"

    return lead, backing


def parse_multi_voice_fields(text):
    """Extract structured multi-singer fields from a vocal analysis pass.
    Returns a dict with keys that may be empty strings when unparsed.
    """
    text = _strip_md_emphasis(text or "")
    out = {
        "num_distinct_voices": "",
        "voice_arrangement": "",
        "co_lead_detail": "",
        "multi_voice_evidence": "",
    }

    m = re.search(
        r'NUM_DISTINCT_VOICES\s*[=:]\s*["\'<\[]?\s*([0-9]+\+?|uncertain|one|two|three[+]?|1|2|3)',
        text,
        re.IGNORECASE,
    )
    if m:
        raw = m.group(1).strip().lower()
        alias = {"one": "1", "two": "2", "three": "3", "three+": "3+"}
        out["num_distinct_voices"] = alias.get(raw, raw)

    m = re.search(
        r'VOICE_ARRANGEMENT\s*[=:]\s*["\'<\[]?\s*([A-Za-z0-9_\-]+)',
        text,
        re.IGNORECASE,
    )
    if m:
        arr = m.group(1).strip().lower().replace("-", "_")
        if CLIP_ANSWER_TOLERANT:                                  # 0.7.1-pre12 (F-28)
            m2 = re.search(r'VOICE_ARRANGEMENT\s*[=:]\s*["\'<\[]?\s*([^\n"\'>\](]*)', text, re.IGNORECASE)
            full = re.sub(r"[\s\-]+", "_", (m2.group(1) if m2 else arr).strip().lower()).strip("_.,;")
            arr = (_clip_value_pick(full, ("solo_lead", "lead_plus_backing", "duet_co_leads", "call_response",
                                           "group_unison", "uncertain"),
                                    dict(_CLIP_ARRANGEMENT_SYNONYMS, call_and_response="call_response"))
                   or arr)
        allowed = {
            "solo_lead", "lead_plus_backing", "duet_co_leads",
            "call_response", "group_unison", "uncertain",
        }
        if arr in allowed:
            out["voice_arrangement"] = arr

    m = re.search(
        r'CO_LEAD_DETAIL\s*[=:]\s*\[?([^\n\]]+)',
        text,
        re.IGNORECASE,
    )
    if m:
        detail = m.group(1).strip().strip("[]")
        if detail and detail.lower() not in ("none", "n/a", "na"):
            out["co_lead_detail"] = detail[:400]

    m = re.search(
        r'MULTI_VOICE_EVIDENCE\s*[=:]\s*\[?([^\n\]]+)',
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
    text = _discard_vocal_choice_lists(text)
    lead = ""
    confidence = ""

    m = re.search(r'LEAD_PROFILE\s*[=:]\s*["\'<\[]?\s*([A-Za-z0-9_\- ]+)', text or "", re.IGNORECASE)
    if m:
        lead = _resolve_lead_tag_variant(m.group(1))

    if lead not in VOCAL_LEAD_TAGS:
        cat_m = re.search(r'LEAD_CATEGORY\s*[=:]\s*["\'<\[]?\s*([A-Za-z0-9_\- ]+)', text or "", re.IGNORECASE)
        mod_m = re.search(r'GENDER_MODIFIER\s*[=:]\s*["\'<\[]?\s*([A-Za-z0-9_\- ]+)', text or "", re.IGNORECASE)
        if cat_m and mod_m:
            lead = _lead_from_category_modifier(cat_m.group(1), mod_m.group(1))
        elif cat_m:
            lead = _lead_from_category_modifier(cat_m.group(1), "")
        if lead not in VOCAL_LEAD_TAGS and cat_m:
            # The same hedge-spelling problem appears on LEAD_CATEGORY.
            lead = _resolve_lead_tag_variant(cat_m.group(1)) or lead

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
    band = None

    # Prefer the "practical sung range (10-90%)" line -- build_vocal_objective_report
    # computes this specifically because it's less skewed by stray high/low
    # frames than the wider 5-95 band, and its own text says to prefer it over
    # extremes. Previously this function only ever looked for the 5-95 line,
    # so that preferred, tighter statistic never actually reached the final
    # VOCAL PITCH REPORT even though it was already being computed. Fall back
    # to the 5-95 line for robustness (e.g. older/cached report text) if the
    # practical line isn't found.
    m = re.search(
        r"practical sung range \(10-90%\):\s*([0-9]+(?:\.[0-9]+)?)-([0-9]+(?:\.[0-9]+)?)\s*Hz",
        text or "",
    )
    if m:
        band = "10-90"
    else:
        m = re.search(
            r"voiced pitch 5-95 percentile range:\s*([0-9]+(?:\.[0-9]+)?)-([0-9]+(?:\.[0-9]+)?)\s*Hz",
            text or "",
        )
        if m:
            band = "5-95"
    if m:
        try:
            low = float(m.group(1))
            high = float(m.group(2))
        except Exception:
            low = None
            high = None
            band = None

    m = re.search(r"approx median note:\s*(\S+)", text or "")
    if m:
        note = m.group(1).strip()

    return {"median": median, "low": low, "high": high, "note": note, "band": band}


_FORMANT_NOTE_TEMPLATE_OPTIONS = "prepubertal-like/adolescent-transitional/adult-sized/ambiguous/not assessable"


def _vocal_text_without_template_labels(text):
    """A vocal pass's reply without the wording it copies from its own output
    template, for the keyword scan in _apply_vocal_age_guard.

    The PROBABILITY_ESTIMATE line repeats the template's category names
    ("adolescent male (voice changing/recently changed) Y%; female teen/adult
    A%") whatever the numbers are, and a FORMANT_NOTE copied whole lists
    "adolescent-transitional" among the options. Read as prose, those labels
    are "explicit voice-change evidence" in every reply that fills the line
    in, which demoted adult male voices to adolescent_male_likely. The
    previous listening model wrote "PROBABILITY ESTIMATE: not available", so
    this never came up; MOSS-Music always fills the line. Everything else in
    the reply is scanned as before."""
    t = str(text or "")
    t = re.sub(r"(?im)^[ \t*_-]*PROBABILITY[ _]ESTIMATE[ \t]*[:=][^\n]*$", "", t)
    return t.replace(_FORMANT_NOTE_TEMPLATE_OPTIONS, "")


# 0.7.1-pre12 AGE_GUARD_LEAD_FIELDS_ONLY (F-23): the age guard reads what is said about the
# lead voice, not the backing or second-voice fields, and "teen/adult" is the template's
# female label, not a teenage reading.
_AGE_GUARD_OTHER_VOICE_FIELDS_RE = re.compile(
    r"(?im)^[ \t*_-]*(?:BACKING[A-Z_]*|CO_LEAD_DETAIL|MULTI_VOICE_EVIDENCE|NUM_DISTINCT_VOICES|VOICE_ARRANGEMENT)"
    r"[ \t]*[:=][^\n]*$")


def _mark_superseded_vocal_tags(raw_text, final_lead):
    """Annotate stale lead-profile tags inside a raw vocal pass before it is
    embedded verbatim in the analysis.

    The raw VOCAL / SINGER PROFILE and VOCAL CONFIRMATION PASS text is stored
    unmodified so its reasoning stays auditable, but it carries the pass's own
    structured guess (e.g. "LEAD_PROFILE=post_puberty_male"). When the guards or
    the confirmation merge later settle on a different category, that literal
    token is still sitting in the notes contradicting the settled answer, and
    the writer reads it as current -- which is how a reply hedges correctly
    ("youthful") early and then reverts to the stale tag further down.

    Values that already agree with the final decision are left untouched.
    """
    if not raw_text or not final_lead:
        return raw_text or ""

    final_norm = _normalize_vocal_tag(final_lead)
    final_norm = VOCAL_LEAD_ALIASES.get(final_norm, final_norm)
    if not final_norm or final_norm in ("", "unknown"):
        return raw_text

    def _annotate(m):
        label, value = m.group(1), m.group(2).rstrip()
        if not value:
            return m.group(0)
        if raw_text[m.end():m.end() + 16].lstrip().startswith("[SUPERSEDED"):
            return m.group(0)
        norm = _resolve_lead_tag_variant(value)
        if not norm or norm == final_norm:
            return m.group(0)
        return (
            f"{label}={value}  [SUPERSEDED — this was this pass's own provisional "
            f"guess and was overruled by later evidence. The settled category for "
            f"this track is {final_norm}. Do not use this line, and do not describe "
            f"the singer using this superseded category.]"
        )

    try:
        return re.sub(
            r"\b(LEAD_PROFILE|LEAD_CATEGORY)\s*[=:]\s*([A-Za-z0-9_\- ]+)",
            _annotate,
            raw_text,
        )
    except Exception:
        return raw_text


def _voice_type_map_lines(median_f0, low_f0=None, high_f0=None):
    """Where the measured vocal line sits, expressed as voice-type words for
    BOTH a male and a female singer, computed from pitch alone.

    Kept gender-neutral on purpose so the pitch report is identical with or
    without tags; the writer picks the column that matches the settled
    gender. Boundaries are approximate tessitura centres for a sung lead
    (median of the voiced line), not strict classical ranges.
    """
    try:
        med = float(median_f0)
    except (TypeError, ValueError):
        return []
    if med <= 0:
        return []

    # Bands are for the MEDIAN of a sung popular-music lead, which sits well
    # above classical tessitura centres. They used to put anything from 250 Hz
    # (B3) up in "high tenor / countertenor (head voice or falsetto)" and from
    # 330 Hz (E4) up in "treble / boy soprano", and to flag "unusually high for
    # an adult man" from 250 Hz -- so an ordinary male pop lead centred on C4
    # and a belting rock tenor centred on E4 were both described as falsetto.
    # Chest/mixed-voice male leads routinely centre anywhere up to ~F4.
    if med < 130:
        male = "bass range"
    elif med < 175:
        male = "baritone range"
    elif med < 280:
        male = "tenor range (the usual home of male pop/rock leads)"
    elif med < 350:
        male = "high tenor range (typical of belted rock/pop leads in chest or mixed voice)"
    elif med < 440:
        male = "very high for a man (upper-tenor belting, head voice or falsetto)"
    else:
        male = "falsetto / countertenor territory for an adult man (treble range for a boy)"

    if med < 185:
        female = "below contralto — unusually deep for a woman, tenor-like"
    elif med < 250:
        female = "contralto range (low for a woman)"
    elif med < 330:
        female = "alto / low mezzo range (typical of female pop leads in chest voice)"
    elif med < 440:
        female = "mezzo-soprano range"
    else:
        female = "soprano range"

    lines = [
        "VOICE-TYPE MAP (from measured pitch only; approximate tessitura of the sung line):",
        f"- if the singer is male (any age): {male}",
        f"- if the singer is female: {female}",
    ]
    notes = []
    if med >= 440:
        notes.append("remarkably high for an adult man (falsetto-led)")
    elif med >= 350:
        notes.append("unusually high for an adult man")
    if med < 185:
        notes.append("unusually deep for a woman")
    if med < 200:
        notes.append("deep for a boy whose voice hasn't fully broken")
    if notes:
        lines.append("- worth remarking on if it matches the settled gender/age: " + "; ".join(notes))
    try:
        if low_f0 and high_f0:
            lines.append(
                f"- practical range in notes: {_hz_to_note_name(float(low_f0))} to "
                f"{_hz_to_note_name(float(high_f0))}"
            )
    except Exception:
        pass
    return lines


_SETTLED_ADULT_MALE_CUES = (
    "adult male", "male resonance", "male vocal weight", "mature male",
    "settled adult", "full adult", "full chest", "deep chest", "man's voice",
    "adult chest", "adult-sounding male", "grown man",
)
_HIGH_ADULT_MALE_REGISTER_CUES = (
    "falsetto", "head voice", "countertenor", "high tenor", "tenore contraltino",
)


def _apply_f0_pitch_guard(lead_profile, median_f0, low_f0=None, high_f0=None,
                          analysis_text="", confirmed_adult_male=False):
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

    _txt = (analysis_text or "").lower()
    adult_high_male = (
        confirmed_adult_male
        and lead_profile in ("post_puberty_male", "adult_male", "young_male")
        and _marker_present_unnegated(_txt, _SETTLED_ADULT_MALE_CUES)
        and _marker_present_unnegated(_txt, _HIGH_ADULT_MALE_REGISTER_CUES)
    )
    if adult_high_male:
        return lead_profile

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


# dev43: single singers called "two co-leads ... about an octave apart". The
# pitch histogram upgrade (below) is meant to run only when neither listening
# pass said how many singers there are, but the vocal pass writes that in
# forms the field parser does not read ("NUM_DISTINCT VOICES: 1", "VOICE
# ARRANGEMENT: solo lead", "single lead voice with occasional backing
# harmonies"), so the upgrade ran on single-singer tracks; and two centres an
# octave apart are what octave doubling, or the pitch tracker jumping an
# octave on one voice, produce.
_SINGLE_VOICE_RES = (
    r"NUM[_ ]DISTINCT[_ ]VOICES\s*[=:]\s*[\"'<\[]?\s*(?:1|one)\b",
    r"VOICE[_ ]ARRANGEMENT\s*[=:]\s*[\"'<\[]?\s*(?:solo[_ ]lead|lead[_ ]plus[_ ]backing|single[_ ]lead)\b",
    r"MULTI[_ \-]VOICE[_ ]EVIDENCE\s*[=:]\s*[\"'<\[]?\s*single lead",
    r"\bsingle lead (?:voice|vocal|vocalist|singer)\b",
    r"\bsingle (?:vocalist|singer)\b",
    r"\bone lead (?:vocalist|singer)\b",
)
F0_OCTAVE_BAND_SEMITONES = (11.0, 13.0)


def _vocal_pass_says_single(text):
    """True when a listening pass's own text reports one singer, in any of the
    spellings it uses."""
    t = _strip_md_emphasis(str(text or ""))
    return any(re.search(p, t, re.IGNORECASE) for p in _SINGLE_VOICE_RES)


def _f0_hint_separation(report):
    m = re.search(r"voiced-f0 multi-peak hint:[^\n]*?~(\d+(?:\.\d+)?)\s+semitones between top two",
                  str(report or ""))
    return float(m.group(1)) if m else None


def _f0_sep_is_octave(sep):
    lo, hi = F0_OCTAVE_BAND_SEMITONES
    return sep is not None and lo <= sep <= hi

# 0.7.1-pre12 (F-28): near-miss arrangement values.
_CLIP_ARRANGEMENT_SYNONYMS = {"solo": "solo_lead", "single_lead": "solo_lead", "lead_with_backing": "lead_plus_backing",
                              "lead_and_backing": "lead_plus_backing", "duet": "duet_co_leads",
                              "co_leads": "duet_co_leads", "co_lead": "duet_co_leads", "unison": "group_unison"}


def _instrumental_presence_answer(reply):
    """Only exact, unique, confident absence is an instrumental vote."""
    text = _strip_md_emphasis(str(reply or ""))
    fields = {}
    for key in ("HUMAN_VOICE", "CONFIDENCE"):
        values = re.findall(r"(?mi)^\s*" + key + r"\s*[=:]\s*([^\n]*)", text)
        if len(values) != 1:
            return False
        fields[key] = values[0].strip().lower()
    return fields == {"HUMAN_VOICE": "no", "CONFIDENCE": "high"}


def _instrumental_full_coverage(mix, rate, ask, diagnostic=None):
    """Full coverage, with one shorter-window retry for ambiguous answers.
    A positive voice answer vetoes absence. All negative votes remain high-confidence.
    """
    import math
    log = diagnostic if diagnostic is not None else {}
    log.update(outcome="unresolved", detail="verification unavailable")
    if not INSTRUMENTAL_FULL_COVERAGE_CHECK or rate <= 0 or len(mix) < rate * 6:
        return None
    duration = len(mix) / float(rate)
    count = max(1, int(math.ceil(max(0., duration - 20.) / 19.)) + 1)
    limit = max(0, int(INSTRUMENTAL_PRESENCE_MAX_CLIPS))
    if count > limit:
        log["detail"] = "full coverage exceeds configured call limit"
        return None
    calls = 0
    def check(a, b):
        nonlocal calls
        audio = mix[int(a * rate):min(len(mix), int(b * rate))]
        if not len(audio) or not np.isfinite(audio).all():
            return "uncertain"
        if float(np.max(np.abs(audio))) < 1e-5:
            return "no"  # Only digital/near silence; not low vocal-stem energy.
        if calls >= limit:
            log["detail"] = "verification call limit reached"
            return "uncertain"
        calls += 1
        reply = ask(audio, VOCAL_PRESENCE_VERIFY_PROMPT)
        log.update(detail=f"{_mmss(a)}-{_mmss(b)}: " + " ".join(str(reply).split())[:240],
                   calls=calls)
        if _instrumental_presence_answer(reply):
            return "no"
        values = re.findall(r"(?mi)^\s*HUMAN_VOICE\s*[=:]\s*([^\n]*)",
                            _strip_md_emphasis(str(reply or "")))
        # Even a low-confidence explicit yes blocks an absence verdict.
        if len(values) == 1 and values[0].strip().lower() == "yes":
            confidence = re.findall(r"(?mi)^\s*CONFIDENCE\s*[=:]\s*([^\n]*)",
                                    _strip_md_emphasis(str(reply or "")))
            log["positive_confidence"] = confidence[0].strip().lower() if len(confidence) == 1 else ""
            return "yes"
        return "uncertain"
    def covered(a, b, depth=0):
        verdict = check(a, b)
        if verdict == "yes":
            log["outcome"] = "voice_detected"
            # A positive vote still vetoes absence. Require a second, explicit
            # human-articulation check before it can unlock singer identification.
            if calls < limit and log.get("positive_confidence") == "high":
                corroboration = ask(mix[int(a * rate):min(len(mix), int(b * rate))],
                    "Listen only to this audio clip. Distinguish a real human voice from guitar, synth, "
                    "feedback, percussion and separation artefacts. Speaking, chanting, rapping and "
                    "wordless singing count; pitch or a voice-like tone alone does not. Do not infer "
                    "a singer from the song or artist. Return exactly three fields on separate lines: "
                    "HUMAN_VOICE=yes|no|uncertain; CONFIDENCE=high|medium|low; "
                    "HUMAN_CUES=speech|singing|chanting|rapping|wordless_singing|none|uncertain. "
                    "Choose one value per field; use uncertain if human articulation is not clear.")
                log["calls"] = calls + 1
                fields = {}
                for key in ("HUMAN_VOICE", "CONFIDENCE", "HUMAN_CUES"):
                    vals = re.findall(r"(?mi)^\s*" + key + r"\s*[=:]\s*([^\n]*)",
                                      _strip_md_emphasis(str(corroboration or "")))
                    fields[key] = vals[0].strip().lower() if len(vals) == 1 else ""
                if (fields["HUMAN_VOICE"] == "yes" and fields["CONFIDENCE"] == "high"
                        and fields["HUMAN_CUES"] in ("speech", "singing", "chanting", "rapping", "wordless_singing")):
                    log["outcome"] = "voice_confirmed"
                log["detail"] += "; corroboration: " + " ".join(str(corroboration or "").split())[:240]
            return False
        if verdict == "no":
            return True
        # A second subdivision covers ambiguous half-windows without lowering
        # confidence or interpreting explanatory prose as a negative vote.
        if (depth >= (2 if INSTRUMENTAL_EXTRA_PRESENCE_RETRY else 1)
                or b - a < (8 if INSTRUMENTAL_EXTRA_PRESENCE_RETRY else 12)
                or calls + 2 > limit):
            return False
        mid = (a + b) / 2.
        return covered(a, mid + .5, depth + 1) and covered(mid - .5, b, depth + 1)
    for index in range(count):
        a, b = index * 19., min(duration, index * 19. + 20.)
        if not covered(a, b):
            return None
    log.update(outcome="instrumental", calls=calls)
    return {"instrumental": True, "reasons": [
        f"a dedicated human-voice presence check found no human voice with high confidence "
        f"throughout the full {duration:.1f}-second mix (silent windows excluded directly)"],
        "presence_coverage_seconds": duration, "presence_windows": count}


def _uncorroborated_presence_report(text):
    """Reject a lone presence vote contradicted by its targeted corroboration."""
    t = str(text or "")
    check = re.search(r"Section voice check[^\n]*inconclusive -- no reliable sampled lead voices[^\n]*", t)
    if not check or "presence verification voice_detected:" not in check.group(0):
        return False
    line = check.group(0)
    # v13 recorded the corroboration reply. An uncertain/negative reply does
    # not become positive evidence just because the whole-track pass said yes.
    failed = re.search(r"corroboration:\s*HUMAN_VOICE=(?:uncertain|no)\b", line)
    old_uncertain = (re.search(r"(?mi)^- FINAL LEAD PROFILE:\s*(?:unknown|unparsed|uncertain)\s*$", t)
        and re.search(r"VOCAL / SINGER PROFILE[^\n]*\nVOCALS_PRESENT=uncertain\b", t))
    return bool(failed or old_uncertain)


def _repair_confirmed_vocal_description(text):
    t = str(text or "")
    if ("VOCAL PRESENCE VERIFICATION: UNRESOLVED" in t
            or "VOCALS_PRESENT=no  # from vocal-presence arbitration" in t):
        return text
    settled = re.search(r"(?mi)^- FINAL LEAD PROFILE:\s*(post_puberty_male|female_teen_adult|adolescent_male|child_male|child_female|mixed_leads)\b", t)
    if not settled:
        return text
    profile = re.search(r"VOCAL / SINGER PROFILE[^\n]*\n([\s\S]*?)(?:\n\nVOCAL DECISION AUDIT|\Z)", t)
    if not profile or not re.search(r"(?mi)^VOCALS_PRESENT\s*[=:]\s*yes\s*$", profile.group(1)):
        return text
    denial = (r"(?i)\bno\s+(?:(?:discernible|clear|audible|human|identifiable|obvious)\s+){0,3}"
              r"(?:vocals?|voices?|singing|speech)\b|\bwithout\s+(?:any\s+)?(?:vocals?|voices?|singing)\b"
              r"|\b(?:purely|entirely|wholly)\s+instrumental\b")
    def clean(m):
        kept = [x for x in re.split(r"(?<=[.!?])\s+", m.group(1)) if not re.search(denial, x)]
        return "DESCRIPTION=" + (" ".join(kept).strip() or "Human vocals are present; see the dedicated vocal profile.")
    return re.sub(r"(?m)^DESCRIPTION=([^\n]*)", clean, t)


def _normalize_unresolved_vocal_report(text):
    """Do not publish positive vocal fields when verification remains open."""
    t = str(_repair_confirmed_vocal_description(text) or "")
    if _uncorroborated_presence_report(t) and "VOCAL PRESENCE VERIFICATION: UNRESOLVED" not in t:
        t += "\n\nVOCAL PRESENCE VERIFICATION: UNRESOLVED. An isolated positive presence vote was not corroborated; artist membership is not evidence of a singer on this recording."
    # Repair older saved reports whose sole positive override was stem notes.
    if ("VOCAL PRESENCE SALVAGE (from vocal-stem transcription)" in t
            and re.search(r"(?mi)^- FINAL LEAD PROFILE:\s*(?:unknown|unparsed|uncertain)\s*$", t)):
        t = re.sub(r"(?m)^VOCAL PRESENCE SALVAGE \(from vocal-stem transcription\)[^\n]*(?:\nVOCALS_PRESENT=[^\n]*)?", "", t)
        if "VOCAL PRESENCE VERIFICATION: UNRESOLVED" not in t:
            t += "\n\nVOCAL PRESENCE VERIFICATION: UNRESOLVED. Stem-note onset alone did not establish a human voice; earlier singer claims are unverified."
    if "VOCAL PRESENCE VERIFICATION: UNRESOLVED" not in t:
        return t
    # Previously generated lyrics or identity assertions cannot survive a
    # failed presence decision when an older report is loaded or exported.
    t = re.sub(r"(?ms)^LYRICS HEARD IN AUDIO[^\n]*\n.*?\[END OF TRANSCRIPTION\]\s*", "", t)
    t = re.sub(r"(?ms)^SINGER IDENTITY RESOLUTION[^\n]*\n.*?(?=\n\n[A-Z][A-Z /()_-]{5,}|\Z)", "", t)
    t = re.sub(r"(?mi)^(VOCALS_PRESENT\s*[=:]\s*)yes\b", r"\1uncertain", t)
    t = re.sub(r"(?mi)^LEAD_VOCAL_CHARACTERISTICS=.*$",
               "LEAD_VOCAL_CHARACTERISTICS=unverified; human vocal presence is unresolved", t)
    t = re.sub(r"(?mi)^MELODIC_RANGE=.*$",
               "MELODIC_RANGE=not established as a vocal range; vocal presence is unresolved", t)
    t = re.sub(r"(?mi)^EARLIEST VOCAL ONSET:.*$",
               "UNVERIFIED STEM ONSET: notes in a separated stem do not establish a human voice or vocal entrance.", t)
    # Remove unsupported vocal claims only from the user-facing description;
    # retain the listening pass and verification diagnostics elsewhere.
    def description(m):
        sentences = re.split(r"(?<=[.!?])\s+", m.group(1))
        kept = [x for x in sentences if not re.search(r"(?i)\b(?:vocal(?:s|ist|ists)?|sing(?:er|ers|ing)?|lyrics?|voice|voices)\b", x)]
        return "DESCRIPTION=" + (" ".join(kept).strip() or "Vocal presence is unresolved.")
    t = re.sub(r"(?m)^DESCRIPTION=([^\n]*)", description, t)
    # Keep the precedence notice independent of DESCRIPTION, whose text is
    # rearranged by the report formatter.
    notice = ("VOCAL PRESENCE VERIFICATION: UNRESOLVED. Human vocal presence was not established; "
              "do not assert a singer, vocal range, vocal onset or reliable audio-transcribed lyrics. "
              "Earlier listening impressions and stem notes are unverified. This is not an instrumental verdict.")
    if notice not in t:
        t += "\n\n" + notice
    return t


VOCAL_PRESENCE_UNRESOLVED_MARKER = "VOCAL PRESENCE VERIFICATION: UNRESOLVED"
VOCAL_PRESENCE_UNRESOLVED_NOTE = (
    "\n\n" + VOCAL_PRESENCE_UNRESOLVED_MARKER + ". No sampled foreground lead was confirmed, "
    "and the dedicated check could not settle whether any human voice is audible. "
    "Do not assert a singer, singer identity, vocal range or reliable audio-transcribed lyrics. "
    "Earlier positive vocal descriptions are unverified. Do not call the track instrumental "
    "unless a separate VOCAL PRESENCE ARBITRATION verdict establishes it. Artist or group "
    "membership cannot establish vocal presence on this recording. This warning takes precedence "
    "over generic singer claims elsewhere in these notes.")


def _reference_vocal_correction(candidate, background, metadata):
    """Require verbatim, positive credits; silence about guests is insufficient."""
    def field(key):
        hits = re.findall(r"(?mi)^" + key + r"\s*=\s*([^\n]+)", str(candidate or ""))
        return hits[0].strip() if len(hits) == 1 else ""
    norm = lambda x: re.sub(r"\s+", " ", str(x)).strip().casefold()
    title = norm((metadata or {}).get("title", ""))
    scope, name = field("REFERENCE_SCOPE"), field("REFERENCE_SINGER")
    credit, identity = field("REFERENCE_TRACK_SOURCE"), field("REFERENCE_IDENTITY_SOURCE")
    bg = norm(background)
    if scope not in ("single_male", "all_male_group") or not title or not name:
        return None
    if any(not q or len(q) > 1800 or norm(q) not in bg for q in (credit, identity)):
        return None
    roster = (scope == "single_male" and _reference_single_credit_roster(background, metadata, name, credit))
    if (not roster and title not in norm(credit)) or norm(name) not in norm(credit) or norm(name) not in norm(identity):
        return None
    if not roster and not re.search(r"(?i)\b(?:lead vocals?|lead singer|sole vocalist|only vocalist)\b", credit):
        return None
    evidence = norm(credit + " " + identity).replace(title, "[title]").replace(norm(name), "[name]")
    if re.search(r"\b(?:not|no|possibly|unconfirmed|rumou?r|female|duet|co[- ]lead|guest vocalist)\b", evidence):
        return None
    if scope == "single_male":
        if not roster and not re.search(r"\b(?:sole|only|single) (?:lead )?(?:singer|vocalist)\b", evidence):
            return None
        if not re.search(r"\b(?:male|man|his|he)\b", norm(identity)):
            return None
        count = "one"
    else:
        if not re.search(r"\b(?:all[- ]male|boy ?band|male vocal group)\b", evidence):
            return None
        count = "several"
    tag = field("SINGER_IDENTITY")
    if tag not in ("post_puberty_male", "adolescent_male_likely", "child_male_likely"):
        return None
    return ("SINGER_IDENTITY=" + tag + "\nCONFIDENCE=medium\nLEAD_SINGERS=" + count
            + "\nREFERENCE_VOCAL_CORRECTION=yes\nREASONING=Positive reference credits constrain singer identity and count; "
            "the contradictory sampled gender impressions do not establish a female co-lead. "
            "Acoustic measurements remain unchanged.\nREFERENCE_TRACK_SOURCE=" + credit
            + "\nREFERENCE_IDENTITY_SOURCE=" + identity)


def _vocal_presence_still_unresolved(vocal_result, svc, final_lead=None):
    svc = svc or {}
    # A whole-track yes cannot overrule failed targeted corroboration when
    # the section checks found no reliable lead anywhere. Accepted mixed
    # leads and successful presence arbitration retain their existing path.
    if (not svc.get("mixed") and not svc.get("presence_arbitration")
            and (svc.get("presence_diagnostic") or {}).get("outcome") == "voice_detected"
            and "no reliable sampled lead voices" in str(svc.get("summary") or "")):
        return True
    flag = _parse_vocals_present_flag(vocal_result or "")
    if flag in ("yes", "no"):
        return False
    if not flag and final_lead in ("post_puberty_male", "female_teen_adult", "mixed_leads"):
        return False  # Preserve an accepted legacy lead when only the flag is missing.
    svc = svc or {}
    if svc.get("mixed") or svc.get("presence_arbitration"):
        return False
    if (svc.get("presence_diagnostic") or {}).get("outcome") == "voice_confirmed":
        return False
    # Accepted lead samples are positive human-vocal evidence too.
    if re.search(r"; samples:[^\n]*\b(?:male|female) \(high; (?:solo_lead|lead_plus_backing)\)", str(svc.get("summary") or "")):
        return False
    return True


def choose_final_vocal_lead(initial_lead, confirm_lead, confirm_confidence, confirm_text="", median_f0=None, multi_voice_fields=None):
    initial = initial_lead if initial_lead in VOCAL_LEAD_TAGS else "unknown"
    confirm = confirm_lead if confirm_lead in VOCAL_LEAD_TAGS else ""

    if not confirm:
        return initial
        
    # If multi-voice fields clearly indicate multiple distinct leads, override to mixed_leads
    if multi_voice_fields:
        _arr = multi_voice_fields.get("voice_arrangement")
        if _arr in ("duet_co_leads", "call_response") or (
            multi_voice_fields.get("num_distinct_voices") in ("2", "3", "3+")
            and _arr not in ("solo_lead", "lead_plus_backing", "group_unison")
        ):
            return "mixed_leads"

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
        # Its own threshold (the value the male caps used to have), so raising
        # those caps for adult rock/pop tenors does not also narrow this
        # female-mislabel recovery path, which needs a strong female
        # confirmation as well.
        _male_cap = globals().get("F0_FEMALE_RECOVERY_MIN_HZ") or globals().get("F0_MALE_ANY_HARD_CAP_HZ")
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


_VFT_NOUN = r"(?:vocal(?:s|ist|ly)?|singer|singing|sings|sung|voice|lead|led|crooner|frontman|frontwoman|rapper)"
_VFT_FEMALE = r"(?:female|woman|women|girl|girls|soprano|mezzo(?:[- ]soprano)?|contralto|alto)"


def _free_form_vocal_lean(texts):
    """dev55: what the listening model's free descriptions say about the lead
    voice's gender. Returns dict(lean=None|'male'|'female'|'both',
    male=n, female=n, teen=bool, quotes=[...]). Only sentences that are about
    a voice (vocal/singer/lead...) count; hedged sentences ("uncertain",
    "unclear"), negations ("no female vocals"), structured tag lines and
    saxophone ranges are ignored."""
    male = female = 0
    teen = False
    quotes = []
    for txt in texts or []:
        if not txt:
            continue
        for sent in re.split(r"(?<=[.!?\n])\s+", str(txt)):
            s_ = sent.strip()
            if not s_ or len(s_) > 400:
                continue
            low = s_.lower()
            if not re.search(r"\b" + _VFT_NOUN + r"\b", low):
                continue
            if re.search(r"\b(?:uncertain|unclear|cannot|can't|unsure|ambiguous|androgynous|not (?:clear|possible|determinable))\b", low):
                continue
            if re.search(r"\b(?:no|not|without|absence of|lacks?)\s+(?:a\s+|any\s+)?(?:male|female|man|woman|boy|girl)\b", low):
                continue
            if re.match(r"^\W*(?:lead_category|gender_modifier|lead_profile|backing_profiles?|backing_note|"
                        r"vocal[ _]demographics|probability_estimate|co_lead_detail|backing)\b", low):
                continue
            if re.search(r"\b(?:alto|tenor|baritone|soprano)\s+sax", low):
                continue
            f_ = bool(re.search(r"\b" + _VFT_FEMALE + r"\b", low))
            m_ = bool(re.search(r"\b(?:male|man|men|boy|baritone|tenor)\b", low)) and not re.search(
                r"\b(?:female|woman|women|girl)\b", low)
            if m_ and not f_:
                male += 1
            elif f_ and not m_:
                female += 1
            if m_ or f_:
                quotes.append(s_)
                if re.search(r"\b(?:teen|teens|teenage|teenager|adolescent)\b", low):
                    teen = True
    lean = None
    if male and not female:
        lean = "male"
    elif female and not male:
        lean = "female"
    elif male and female:
        lean = "both"
    return {"lean": lean, "male": male, "female": female, "teen": teen, "quotes": quotes}


def _apply_free_form_vocal_lean(final_lead, info, median_f0=None):
    """dev55: lead profile from the free descriptions, only when the
    structured passes settled nothing (unknown / uncertain / empty)."""
    if not VOCAL_FREE_TEXT_GENDER or not info or final_lead not in ("", "unknown", "uncertain", None):
        return final_lead
    lean = info.get("lean")
    if info.get("vetoed"):
        return final_lead
    if lean == "female":
        return "female_teen_adult"
    if lean == "male":
        cap = float(F0_MALE_ANY_HARD_CAP_HZ) if F0_MALE_ANY_HARD_CAP_HZ is not None else None
        if info.get("teen"):
            return "adolescent_male_likely"
        if median_f0 is None or cap is None or float(median_f0) < cap:
            return "post_puberty_male"
    return final_lead


def _free_form_vocal_lean_line(info):
    if not info or not info.get("quotes"):
        return ""
    qs = "; ".join('"%s"' % q.replace("\n", " ")[:160] for q in info["quotes"][:3])
    lean = {"male": "a male lead", "female": "a female lead", "both": "both male and female voices"}.get(
        info.get("lean"), "no consistent gender")
    return (f"LISTENING-MODEL VOICE DESCRIPTIONS (free text, audio-based): {info.get('male', 0)} describe a male "
            f"voice, {info.get('female', 0)} a female voice -> {lean}"
            + (" (described as teenage)" if info.get("teen") else "") + f". Quotes: {qs}"
            + (f" -- NOT used for FINAL LEAD PROFILE: {info['vetoed']}; the two disagree, so the "
               "audio-only gender is left open" if info.get("vetoed") else "")
            + (f". {_VOICE_UNBROKEN_MARK}: heard as a woman's voice in those descriptions, while "
               f"the dedicated vocal pass rated it child-like or a teenage boy's at "
               f"{info['unbroken'][0]:.2f} together (adult man {info['unbroken'][1]:.2f}, woman "
               f"{info['unbroken'][2]:.2f}) -- the usual signature of a boy whose voice has not "
               "dropped yet" if info.get("unbroken") else ""))


def _free_text_vocal_step(moss_model, moss_processor, audio_path, first_pass, era_result,
                          vocal_result, final_lead, median_f0, no_clear_vocals):
    """dev55: run the short voice question and collect the free descriptions;
    returns (final_lead, info). Never raises."""
    if not (VOCAL_FREE_TEXT_GENDER and vocal_result and not no_clear_vocals):
        return final_lead, None
    _vq = ""
    if VOCAL_QUICK_PASS and moss_model is not None and moss_processor is not None:
        try:
            status("Listening — describing the lead voice in plain words...")
            _vq = moss_generate(moss_model, moss_processor, [{
                "role": "user",
                "content": [
                    {"type": "text", "text": VOCAL_QUICK_PROMPT},
                    {"type": "audio", "path": audio_path},
                ],
            }], max_new_tokens=96) or ""
        except Exception:
            _vq = ""
    try:
        _fp_desc = re.split(r"\n\s*(?:LYRICS|13\.\s*LYRICS)", first_pass or "")[0]
        info = _free_form_vocal_lean([_fp_desc, era_result or "", _vq])
        info["quick"] = (_vq or "").strip()
        _veto = _vocal_probability_veto(info.get("lean"), vocal_result)
        if _veto:
            info["vetoed"] = _veto
            if VOCAL_UNBROKEN_SIGNATURE and info.get("lean") == "female":
                _ym = _vocal_probability_young_mass(vocal_result)
                if _ym and _ym[0] >= 0.5:
                    info["unbroken"] = _ym
        before = final_lead
        final_lead = _apply_free_form_vocal_lean(final_lead, info, median_f0)
        if final_lead != before:
            info["applied"] = final_lead
        return final_lead, info
    except Exception:
        return final_lead, None


def _vocal_note_pitch_line(stem_notes):
    """dev55: duration-weighted pitch summary of the vocal stem's transcribed
    notes, or "" when there are too few (fewer than 40 notes / 15 s sung)."""
    if not VOCAL_NOTE_PITCH_LINE or not isinstance(stem_notes, dict):
        return ""
    notes = None
    for key in ("vocals", "vocal", "voice"):
        if stem_notes.get(key):
            notes = stem_notes[key]
            break
    if not notes:
        return ""
    mids, durs = [], []
    for n in notes:
        if not isinstance(n, dict):
            continue
        m = _sx_note_midi(n)
        sp = _sx_note_span(n)
        if m is None or not sp:
            continue
        d = float(sp[1] - sp[0])
        if d <= 0 or not (36 <= m <= 96):
            continue
        mids.append(float(m))
        durs.append(d)
    if len(mids) < 40 or sum(durs) < 15.0:
        return ""
    m_ = np.asarray(mids)
    w_ = np.asarray(durs)
    o = np.argsort(m_)
    cw = np.cumsum(w_[o]) / float(w_.sum())

    def _q(p):
        return float(m_[o][min(len(o) - 1, int(np.searchsorted(cw, p)))])

    def _hz(mm):
        return 440.0 * 2.0 ** ((mm - 69.0) / 12.0)

    def _nm(mm):
        try:
            return librosa.midi_to_note(int(round(mm)))
        except Exception:
            return ""
    med, lo, hi = _q(0.5), _q(0.1), _q(0.9)
    return (f"VOCAL NOTE PITCH (second reading, from the vocal stem's {len(mids)} transcribed notes, "
            f"{sum(durs):.0f} s sung, duration-weighted): median {_hz(med):.0f} Hz ({_nm(med)}); "
            f"10-90%: {_hz(lo):.0f}-{_hz(hi):.0f} Hz ({_nm(lo)} to {_nm(hi)}). Pitch is not gender or age; "
            "use it, like the pyin reading, only as a range/register description and a soft constraint.")


def _pyin_reading_confidence_note(vocal_objective_report):
    """dev55: a low-confidence label for a pyin reading that rests on very few
    voiced frames or has collapsed to one pitch, else ""."""
    rep = str(vocal_objective_report or "")
    if not VOCAL_NOTE_PITCH_LINE or not rep:
        return ""
    reasons = []
    vf = _voiced_fraction_from_report(rep)
    if vf is not None and vf < 0.05:
        reasons.append(f"only {vf * 100:.1f}% of the scanned frames were confidently voiced")
    m = re.search(r"(?:practical sung range \(10-90%\)|voiced pitch 5-95 percentile range):\s*([0-9.]+)-([0-9.]+)\s*Hz", rep)
    if m:
        try:
            a, b = float(m.group(1)), float(m.group(2))
            if a > 0 and b / a < 2.0 ** (1.0 / 12.0):
                reasons.append(f"its range collapsed to {a:.0f}-{b:.0f} Hz (under a semitone)")
        except ValueError:
            pass
    if not reasons:
        return ""
    return ("pyin pitch reading confidence: LOW (" + "; ".join(reasons)
            + ") -- prefer VOCAL NOTE PITCH when present.")


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


def _backing_guidance(backing_profiles):
    full = build_vocal_priority_note("unknown", backing_profiles)
    for start in (" If female backing", " If male backing", " Mixed backing"):
        i = full.find(start)
        if i != -1:
            return full[i:]
    return ""


def build_acoustic_vocal_note(acoustic_lead, backing_profiles, identity_tag=None, unbroken=False):
    """How the lead voice SOUNDS, from the audio passes only.

    Emitted for every track, with or without tags. Previously the only vocal
    wording directive was produced inside the identity-resolution branch, so
    a tagged file got a strong "say 'male voice'" instruction keyed to the
    identity result while an untagged copy of the same audio got none -- and
    the writer's description of pitch and register changed with the tags.
    This note is keyed to the acoustic FINAL LEAD PROFILE and is therefore
    identical for the same audio regardless of metadata.
    """
    if (unbroken and VOCAL_UNBROKEN_SIGNATURE and acoustic_lead in ("", "unknown", "uncertain", None)):
        return _unbroken_voice_note(identity_tag, backing_profiles)
    if _identity_overrides_gender(identity_tag, acoustic_lead):
        # The audio passes' gender label is known to be wrong for this track
        # (typically an unbroken boy's voice read as female). Don't hand the
        # writer a sentence telling it to say "female voice" at all.
        return (
            "\n\nACOUSTIC VOCAL READING (audio only): the audio passes' gender label for "
            f"this lead ({acoustic_lead}) is overridden by the SINGER IDENTITY note below — "
            "a high voice that has not fully deepened is routinely misread by those passes. "
            "Everything they MEASURED still stands: use the VOCAL PITCH REPORT and VOCAL / "
            "SINGER PROFILE for pitch figures, range, register, timbre, brightness and weight, "
            "exactly as reported. Only the gender and age words come from the identity."
            + _backing_guidance(backing_profiles)
        )
    note = build_vocal_priority_note(acoustic_lead, backing_profiles)
    note = note.replace(
        "VOCAL CLASSIFICATION PRIORITY:",
        "ACOUSTIC VOCAL READING (audio only; identical with or without file tags; its gender/age "
        "wording is only the default when no SINGER IDENTITY note follows):",
    ).replace(
        "VOCAL PROFILE PRIORITY:",
        "ACOUSTIC VOCAL READING (audio only; identical with or without file tags; its gender/age "
        "wording is only the default when no SINGER IDENTITY note follows):",
    )
    # The per-category wording directives inside build_vocal_priority_note
    # ("say 'female voice'", "do not call it a young male voice") must not
    # overrule an identity result that names a different gender.
    note = note.replace(
        " Do not describe this as child-like, boy/girl uncertain, or a young male voice unless the user explicitly corrects it.",
        "",
    )
    return note + (
        " Use this reading, the VOCAL PITCH REPORT and the VOCAL / SINGER PROFILE for the "
        "measured SOUND: pitch figures, range, register, timbre and weight. File tags, the "
        "artist's name, Wikipedia and the cover art must NOT change those. Gender and age "
        "words (she/he, woman/boy/man, young/adult) are different: if a SINGER IDENTITY note "
        "is present it decides them, because the audio alone cannot tell an unbroken boy's "
        "voice from a woman's. With no identity note, use this reading's category."
    )


def _measured_lead_pitch(text):
    """{'median_hz', 'median_note', 'lo_hz', 'hi_hz', 'lo_note', 'hi_note',
    'notes_median_hz', 'notes_median_note', 'notes_lo_hz', 'notes_hi_hz'} from
    the VOCAL PITCH REPORT and the VOCAL NOTE PITCH line, whichever exist."""
    t = str(text or "")
    out = {}
    m = re.search(r"objective voiced pitch median:\s*([\d.]+)\s*Hz", t)
    if m:
        out["median_hz"] = float(m.group(1))
    m = re.search(r"approximate median note:\s*([A-G][#♯♭b]?\d)", t)
    if m:
        out["median_note"] = m.group(1)
    m = re.search(r"practical sung range \(10th-90th percentile\):\s*([\d.]+)-([\d.]+)\s*Hz", t)
    if m:
        out["lo_hz"], out["hi_hz"] = float(m.group(1)), float(m.group(2))
    m = re.search(r"practical range in notes:\s*([A-G][#♯♭b]?\d)\s+to\s+([A-G][#♯♭b]?\d)", t)
    if m:
        out["lo_note"], out["hi_note"] = m.group(1), m.group(2)
    m = re.search(r"VOCAL NOTE PITCH \(second reading[^)]*\):\s*median\s*([\d.]+)\s*Hz\s*\(([^)]+)\);\s*"
                  r"10-90%:\s*([\d.]+)-([\d.]+)\s*Hz", t)
    if m:
        out["notes_median_hz"], out["notes_median_note"] = float(m.group(1)), m.group(2)
        out["notes_lo_hz"], out["notes_hi_hz"] = float(m.group(3)), float(m.group(4))
    return out


# dev71: a pitch tracker reading an octave above the sung line (see VOCAL_PITCH_OCTAVE_CHECK).
_OCTAVE_CHECK_TAG = "OCTAVE CHECK:"


def _pitch_octave_error(text):
    """dev71: (pyin_median, corrected_median, notes_lo, notes_hi) when the
    pitch tracker's median sits an octave above the vocal stem's
    note-by-note median and the tracker itself also saw the lower octave;
    else None."""
    t = str(text or "")
    if not VOCAL_PITCH_OCTAVE_CHECK:
        return None
    p = re.search(r"objective voiced pitch median:\s*([\d.]+)\s*Hz", t)
    nm = re.search(r"VOCAL NOTE PITCH \(second reading[^)]*\):\s*median\s*([\d.]+)\s*Hz\s*\([^)]*\);\s*"
                   r"10-90%:\s*([\d.]+)-([\d.]+)\s*Hz", t)
    if not (p and nm):
        return None
    pyin, nmed = float(p.group(1)), float(nm.group(1))
    if pyin <= 0 or nmed <= 0:
        return None
    d = 12.0 * math.log2(pyin / nmed)
    if not (12.0 - float(VOCAL_PITCH_OCTAVE_TOL_ST) <= d <= 12.0 + float(VOCAL_PITCH_OCTAVE_TOL_ST)):
        return None
    # The tracker must itself show the lower octave: two pitch centres an
    # octave apart, or its own 10th percentile down at the note reading.
    seen_low = False
    mp = re.search(r"multi-peak hint: ~\d+ pitch centres near ([^(]*)\(~([\d.]+) semitones between top two", t)
    if mp:
        try:
            if abs(float(mp.group(2)) - 12.0) <= 1.0:
                seen_low = True
        except Exception:
            pass
    lo = re.search(r"practical sung range \(10th-90th percentile\):\s*([\d.]+)-([\d.]+)\s*Hz", t)
    if lo and abs(12.0 * math.log2(float(lo.group(1)) / nmed)) <= 2.0:
        seen_low = True
    if not seen_low:
        return None
    return pyin, pyin / 2.0, float(nm.group(2)), float(nm.group(3))


def _pitch_octave_raw_figure(text):
    """dev72: the tracker's median exactly as written ("200.0", "659.3")."""
    p = re.search(r"objective voiced pitch median:\s*([\d.]+)\s*Hz", str(text or ""))
    return p.group(1) if p else None


def _pitch_octave_check(text):
    """dev71: rewrite the VOCAL PITCH REPORT (median, note, range, voice-type
    map) an octave down when _pitch_octave_error finds the tracker an octave
    high, and flag the raw figure in the vocal measurements. Idempotent."""
    t = str(text or "")
    if _OCTAVE_CHECK_TAG in t:
        return t
    oe = _pitch_octave_error(t)
    if not oe:
        return t
    pyin, med, lo, hi = oe
    _raw_fig = _pitch_octave_raw_figure(t) or ("%g" % pyin)
    try:
        i = t.find("VOCAL PITCH REPORT (independent")
        if i < 0:
            return t
        j = t.find("\n\n", i)
        j = len(t) if j < 0 else j
        block = t[i:j]
        lines = block.split("\n")
        out = [lines[0],
               f"- objective voiced pitch median: {med:.1f} Hz",
               f"- approximate median note: {_hz_to_note_name(med)}",
               f"- practical sung range (10th-90th percentile): {lo:.1f}-{hi:.1f} Hz",
               f"- {_OCTAVE_CHECK_TAG} the pitch tracker read a median of {pyin:.1f} Hz "
               f"({_hz_to_note_name(pyin)}), exactly an octave above the vocal stem's note-by-note "
               f"reading ({_hz_to_note_name(med)}), and it also found the lower octave -- it was "
               "following the voice's first overtone for much of the song. The figures in this "
               "block are the sung line's: the median an octave down, the range from the note "
               "reading. Do not describe the voice as a soprano, treble or falsetto from the "
               "tracker's figure."]
        out.extend(_voice_type_map_lines(med, lo, hi))
        keep = [ln for ln in lines[1:] if ln.startswith("- Prefer median")]
        out.extend(keep)
        t = t[:i] + "\n".join(out) + t[j:]
        # Every other place that quotes the tracker's median (the vocal
        # measurements, the decision audit) is flagged, not rewritten.
        _tag = (f" [{_OCTAVE_CHECK_TAG} an octave above the sung line, which centres near "
                f"{med:.0f} Hz ({_hz_to_note_name(med)}); see VOCAL PITCH REPORT]")
        _num = r"(?:" + re.escape(_raw_fig) + r"|" + re.escape("%g" % pyin) + r"(?:\.0+)?)"
        _pat = re.compile(r"(?im)^([^\n]*(?:median f0|voiced pitch median)[^\n]*?\b"
                          + _num + r"\s*Hz[^\n]*)$")
        _hi_note = _hz_to_note_name(pyin)
        _pat2 = re.compile(r"(?m)^(approx median note:\s*" + re.escape(_hi_note) + r")\s*$")
        k = t.find("VOCAL PITCH REPORT (independent")
        k_end = t.find("\n\n", k) if k >= 0 else -1
        head, mid, tail = (t[:k], t[k:k_end], t[k_end:]) if (k >= 0 and k_end >= 0) else (t, "", "")
        fix = lambda x: _pat2.sub(lambda m: m.group(1) + _tag, _pat.sub(lambda m: m.group(1) + _tag, x))
        t = fix(head) + mid + fix(tail)
        # the report field filled from the old figures (dev66)
        t = re.sub(r"(?m)^MELODIC_RANGE=lead vocal [^\n]*\(median " + re.escape(_hi_note) + r"\)$",
                   lambda m: (f"MELODIC_RANGE=lead vocal {_hz_to_note_name(lo)}-{_hz_to_note_name(hi)} "
                              f"(median {_hz_to_note_name(med)})"), t, count=1)
        return t
    except Exception:
        return text


def _pitch_octave_audit_line(oct_err):
    """dev71: one line for the identity step's vocal evidence when the
    objective median f0 there is an octave too high (oct_err from
    _pitch_octave_error)."""
    if not oct_err:
        return ""
    pyin, med, _lo, _hi = oct_err
    return (f"{_OCTAVE_CHECK_TAG} the objective median f0 above ({pyin:.1f} Hz) is an octave too high "
            f"(the pitch tracker followed an overtone); the sung line centres near {med:.0f} Hz "
            f"({_hz_to_note_name(med)}), as the note reading shows.")


_STUCK_CHECK_TAG = "PITCH CHECK:"
_SEMITONE = 2.0 ** (1.0 / 12.0)


def _pitch_stuck_error(text):
    """dev73: (pyin_median, pyin_lo, pyin_hi, note_median, note_lo, note_hi)
    when the pitch tracker's sung range has collapsed to under a semitone
    while the vocal stem's note-by-note reading spans three semitones or
    more; else None. One stuck-pitch test scan read "194.9-197.1 Hz (G3 to G3)"
    from 5.9% voiced frames -- the tracker held one sustained tone -- while
    524 transcribed vocal notes over 143 s gave E3 to E4 around B3."""
    t = str(text or "")
    if not VOCAL_PITCH_STUCK_CHECK or _OCTAVE_CHECK_TAG in t:
        return None
    p = re.search(r"objective voiced pitch median:\s*([\d.]+)\s*Hz", t)
    r = re.search(r"practical sung range \(10th-90th percentile\):\s*([\d.]+)-([\d.]+)\s*Hz", t)
    nm = re.search(r"VOCAL NOTE PITCH \(second reading[^)]*\):\s*median\s*([\d.]+)\s*Hz\s*\([^)]*\);\s*"
                   r"10-90%:\s*([\d.]+)-([\d.]+)\s*Hz", t)
    if not (p and r and nm):
        return None
    try:
        pyin, lo, hi = float(p.group(1)), float(r.group(1)), float(r.group(2))
        nmed, nlo, nhi = float(nm.group(1)), float(nm.group(2)), float(nm.group(3))
    except ValueError:
        return None
    if min(pyin, lo, hi, nmed, nlo, nhi) <= 0:
        return None
    if hi / lo >= _SEMITONE or nhi / nlo < _SEMITONE ** 3:
        return None
    return pyin, lo, hi, nmed, nlo, nhi


def _pitch_stuck_check(text):
    """dev73: rewrite the VOCAL PITCH REPORT from the note-by-note reading
    when _pitch_stuck_error finds the tracker's range collapsed, and flag the
    collapsed figures where the vocal measurements quote them. Idempotent."""
    t = str(text or "")
    if _STUCK_CHECK_TAG in t:
        return t
    se = _pitch_stuck_error(t)
    if not se:
        return t
    pyin, lo, hi, med, nlo, nhi = se
    try:
        i = t.find("VOCAL PITCH REPORT (independent")
        if i < 0:
            return t
        j = t.find("\n\n", i)
        j = len(t) if j < 0 else j
        lines = t[i:j].split("\n")
        raw_med = _pitch_octave_raw_figure(t) or ("%g" % pyin)
        out = [lines[0],
               f"- objective voiced pitch median: {med:.1f} Hz",
               f"- approximate median note: {_hz_to_note_name(med)}",
               f"- practical sung range (10th-90th percentile): {nlo:.1f}-{nhi:.1f} Hz",
               f"- {_STUCK_CHECK_TAG} the pitch tracker's own range collapsed to {lo:.1f}-{hi:.1f} Hz "
               f"(under a semitone, median {raw_med} Hz, {_hz_to_note_name(pyin)}): it held one "
               "sustained tone for the few frames it could follow, which no sung line does. The "
               "figures in this block are the vocal stem's note-by-note reading (see VOCAL NOTE "
               "PITCH). Do not describe the voice as staying on one note."]
        out.extend(_voice_type_map_lines(med, nlo, nhi))
        out.extend(ln for ln in lines[1:] if ln.startswith("- Prefer median"))
        t = t[:i] + "\n".join(out) + t[j:]
        # Elsewhere the collapsed figures are flagged, not rewritten.
        _tag = (f" [{_STUCK_CHECK_TAG} a collapsed reading; the sung line spans "
                f"{_hz_to_note_name(nlo)}-{_hz_to_note_name(nhi)} around {_hz_to_note_name(med)}; "
                "see VOCAL PITCH REPORT]")
        _rng = r"%s\s*-\s*%s\s*Hz" % (re.escape("%.1f" % lo), re.escape("%.1f" % hi))
        _num = r"(?:" + re.escape(raw_med) + r"|" + re.escape("%g" % pyin) + r"(?:\.0+)?)"
        _pat = re.compile(r"(?im)^([^\n]*(?:(?:median f0|voiced pitch median)[^\n]*?\b" + _num
                          + r"\s*Hz|" + _rng + r")[^\n]*)$")
        k = t.find("VOCAL PITCH REPORT (independent")
        k_end = t.find("\n\n", k) if k >= 0 else -1
        head, mid, tail = (t[:k], t[k:k_end], t[k_end:]) if (k >= 0 and k_end >= 0) else (t, "", "")
        fix = lambda x: _pat.sub(lambda m: m.group(1) + _tag, x)
        t = fix(head) + mid + fix(tail)
        # the report field filled from the collapsed figures (dev66)
        t = re.sub(r"(?m)^MELODIC_RANGE=lead vocal [^\n]*\(median " + re.escape(_hz_to_note_name(pyin))
                   + r"\)$",
                   lambda m: (f"MELODIC_RANGE=lead vocal {_hz_to_note_name(nlo)}-{_hz_to_note_name(nhi)} "
                              f"(median {_hz_to_note_name(med)})"), t, count=1)
        return t
    except Exception:
        return text


_LOWCONF_CHECK_TAG = "PITCH CHECK (few voiced frames):"


def _pitch_low_confidence_error(text):
    """(pyin_median, note_median, note_lo, note_hi) when the pitch tracker's
    reading is marked LOW confidence and its median sits at least
    VOCAL_PITCH_LOW_CONFIDENCE_MIN_ST semitones from the vocal stem's
    note-by-note reading; else None. The octave and collapsed-range cases
    have their own checks and are left to them. the post-punk test track's tracker read
    a median of 622 Hz (D#5) from 0.4% voiced frames, so the voice-type map
    called the post-punk singer's baritone "falsetto / countertenor territory", while
    271 transcribed vocal notes centred on A#3."""
    t = str(text or "")
    if (not VOCAL_PITCH_LOW_CONFIDENCE_CHECK or _OCTAVE_CHECK_TAG in t or _STUCK_CHECK_TAG in t
            or "pyin pitch reading confidence: LOW" not in t):
        return None
    p = re.search(r"objective voiced pitch median:\s*([\d.]+)\s*Hz", t)
    nm = re.search(r"VOCAL NOTE PITCH \(second reading[^)]*\):\s*median\s*([\d.]+)\s*Hz\s*\([^)]*\);\s*"
                   r"10-90%:\s*([\d.]+)-([\d.]+)\s*Hz", t)
    if not (p and nm):
        return None
    try:
        pyin = float(p.group(1))
        nmed, nlo, nhi = float(nm.group(1)), float(nm.group(2)), float(nm.group(3))
    except ValueError:
        return None
    if min(pyin, nmed, nlo, nhi) <= 0:
        return None
    _tol = PITCH_CHECK_ROUNDING_TOL_ST if PITCH_CHECK_ROUNDED_FIGURES else 0.0
    if abs(12.0 * math.log2(pyin / nmed)) < float(VOCAL_PITCH_LOW_CONFIDENCE_MIN_ST) - _tol:
        return None
    return pyin, nmed, nlo, nhi


def _pitch_low_confidence_check(text):
    """Rewrite the VOCAL PITCH REPORT (median, note, range, voice-type map)
    from the note-by-note reading when _pitch_low_confidence_error applies,
    and flag the tracker's median where the vocal measurements quote it.
    Idempotent. The measured age/gender guards are not affected."""
    t = str(text or "")
    if _LOWCONF_CHECK_TAG in t:
        return t
    le = _pitch_low_confidence_error(t)
    if not le:
        return t
    pyin, med, nlo, nhi = le
    try:
        i = t.find("VOCAL PITCH REPORT (independent")
        if i < 0:
            return t
        j = t.find("\n\n", i)
        j = len(t) if j < 0 else j
        lines = t[i:j].split("\n")
        raw_med = _pitch_octave_raw_figure(t) or ("%g" % pyin)
        out = [lines[0],
               f"- objective voiced pitch median: {med:.1f} Hz",
               f"- approximate median note: {_hz_to_note_name(med)}",
               f"- practical sung range (10th-90th percentile): {nlo:.1f}-{nhi:.1f} Hz",
               f"- {_LOWCONF_CHECK_TAG} the pitch tracker read a median of {raw_med} Hz "
               f"({_hz_to_note_name(pyin)}) from too few confidently voiced frames to trust (its reading "
               "is marked LOW confidence), far from the vocal stem's note-by-note reading. The figures "
               "in this block are the note reading's (see VOCAL NOTE PITCH). Do not describe the voice's "
               "register from the tracker's figure."]
        out.extend(_voice_type_map_lines(med, nlo, nhi))
        out.extend(ln for ln in lines[1:] if ln.startswith("- Prefer median"))
        t = t[:i] + "\n".join(out) + t[j:]
        _tag = (f" [{_LOWCONF_CHECK_TAG} a low-confidence reading; the sung line centres near "
                f"{med:.0f} Hz ({_hz_to_note_name(med)}); see VOCAL PITCH REPORT]")
        _num = r"(?:" + re.escape(raw_med) + r"|" + re.escape("%g" % pyin) + r"(?:\.0+)?)"
        _pat = re.compile(r"(?im)^([^\n]*(?:median f0|voiced pitch median)[^\n]*?\b" + _num
                          + r"\s*Hz[^\n]*)$")
        k = t.find("VOCAL PITCH REPORT (independent")
        k_end = t.find("\n\n", k) if k >= 0 else -1
        head, mid, tail = (t[:k], t[k:k_end], t[k_end:]) if (k >= 0 and k_end >= 0) else (t, "", "")
        fix = lambda x: _pat.sub(lambda m: m.group(1) + _tag, x)
        t = fix(head) + mid + fix(tail)
        # the report field filled from the tracker's figures (dev66)
        t = re.sub(r"(?m)^MELODIC_RANGE=lead vocal [^\n]*\(median " + re.escape(_hz_to_note_name(pyin))
                   + r"\)$",
                   lambda m: (f"MELODIC_RANGE=lead vocal {_hz_to_note_name(nlo)}-{_hz_to_note_name(nhi)} "
                              f"(median {_hz_to_note_name(med)})"), t, count=1)
        return t
    except Exception:
        return text


def _vocal_pitch_into(out, dsp_path):
    status("Measuring vocal pitch/formant proxies...")
    src = "isolated vocal stem (Demucs)" if (out.get("precomputed_stems") or {}).get("vocals") else \
        "full mix (no isolated vocal stem available)"
    _pitch_source_path = (out.get("precomputed_stems") or {}).get("vocals") or dsp_path
    rep = build_vocal_objective_report(_pitch_source_path)
    if rep:
        rep += f"\nmeasurement source: {src}"
    out["vocal_objective_report"] = rep
