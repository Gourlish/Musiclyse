# MUSICLYSE FILE pipeline-harmony 0.8
# ==============================================================================
#  MUSICLYSE 0.8: KEY, CHORDS AND MELODY
# ==============================================================================
# This file holds key, chords and chord progressions, and the melody (also its per-section
# blocks). It is part of musiclyse-0.8.py, which loads it whole at start-up, after its own code,
# into its own namespace (see "THE MUSICLYSE FILES" there); it is not run on its own.
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


def extract_key_from_text(text):
    """Extracts the listening model's KEY=... field, e.g. 'KEY=F# minor'."""
    if not text:
        return None
    m = (re.search(r"(?<![A-Za-z_])KEY\s*=\s*(.+)", text)
         or re.search(r"(?im)^[\s*\-\d.)]*(?:musical\s+)?key\s*[=:]\s*(.+)", text)
         # Markdown-bolded label ("**KEY**= A major"), which the two patterns
         # above miss because of the asterisks between the label and "=".
         or re.search(r"(?im)^[\s*\-\d.)]*(?:musical\s+)?key\*+\s*[=:]\s*(.+)", text))
    if not m:
        return None
    val = m.group(1).strip().splitlines()[0].strip().strip("*").strip().strip("[]").strip()
    if not val or val.lower() in ("uncertain", "unknown", "n/a", "none", "..."):
        return None
    return _clean_key_text(val)


_KEY_TEXT_RE = re.compile(
    r"^\s*([A-Ga-g])\s*(#|\u266f|\u266d|-?sharp|-?flat|b(?=$|[^a-z]|m(?:$|[^a-z]|in|aj)))?\s*"
    r"(major|minor|maj\b|min\b|m\b|dorian|phrygian|lydian|mixolydian|aeolian|ionian|locrian)?",
    re.IGNORECASE,
)


def _clean_key_text(val):
    """Reduce a free-text key field to just the key ("C major", "F# minor").

    The listening model sometimes decorates the field -- observed:
    "C major (as supplied)." -- and the decoration was carried verbatim into
    RECOMMENDED KEY FOR DISCUSSION. Parenthetical remarks and anything after
    the key name are dropped; text that does not start with a key is
    returned with only its parentheticals and trailing punctuation removed.
    """
    if not val:
        return val
    raw = str(val)
    m = _KEY_TEXT_RE.match(raw)
    if not m:
        cleaned = re.sub(r"\s*\([^)]*\)", "", raw).strip().rstrip(".;,: ")
        return cleaned or raw.strip()
    root = m.group(1).upper()
    acc = (m.group(2) or "").lower()
    if acc in ("#", "\u266f", "sharp", "-sharp"):
        root += "#"
    elif acc in ("b", "\u266d", "flat", "-flat"):
        root += "b"
    mode = (m.group(3) or "").lower()
    mode = {"maj": "major", "min": "minor", "m": "minor"}.get(mode, mode)
    if not mode:
        # "Am", "C#m" written without a space are caught above; a bare root
        # keeps whatever qualifier followed it, minus decoration.
        rest = re.sub(r"\s*\([^)]*\)", "", raw[m.end():]).strip().rstrip(".;,: ")
        return (root + (" " + rest if rest and len(rest) <= 20 else "")).strip()
    return f"{root} {mode}"


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
    "cb": "b", "b#": "c", "db": "c#", "d#": "d#", "eb": "d#",
    "e#": "f", "fb": "e", "gb": "f#", "g#": "g#", "ab": "g#",
    "a#": "a#", "bb": "a#",
}


def _normalize_key_for_compare(key_text):
    """Normalize a key string to (pitch_class, mode) for equivalence checks.
    Treats enharmonic spellings (Db == C#) and 'maj'/'major', 'min'/'minor' the same."""
    if not key_text:
        return None
    t = key_text.strip().lower()
    mode = "minor" if re.search(r"\bmin(or)?\b", t) else ("major" if re.search(r"\bmaj(or)?\b", t) else None)
    t = re.sub(r"^([a-g])[\s\-]*(?:flat|\u266d)", r"\1b", t)
    t = re.sub(r"^([a-g])[\s\-]*(?:sharp|\u266f)", r"\1#", t)
    root_match = re.match(r"([a-g])(#|b(?=$|[^a-z]|m(?:$|[^a-z]|in|aj)))?(?=$|[^a-z]|m(?:$|[^a-z]|in|aj))", t)
    if not root_match:
        return None
    root = root_match.group(1) + (root_match.group(2) or "")
    if mode is None:
        _after = t[root_match.end():]
        if re.match(r"maj", _after):
            mode = "major"
        elif re.match(r"m(?:$|[^a-z]|in)", _after):
            mode = "minor"
    root = _KEY_NAME_ALIASES.get(root, root)
    # Preserve an omitted mode as unknown. Silently assuming major makes a modal
    # or underspecified listening-model result look like a confident major-key disagreement.
    return (root, mode)


_KEY_MODE_NAME_RE = re.compile(r"(?i)\b(?:maj(?:or)?|min(?:or)?|ionian|dorian|phrygian|lydian|mixolydian|aeolian|"
                               r"locrian|modal|mode|blues|pentatonic)\b|\b[A-G][#b]?m\b")


def reconcile_key(moss_key, essentia_key, essentia_strength=None, min_essentia_strength=0.6):
    """
    Determine a single recommended key for discussion, mirroring reconcile_bpm's
    approach: prefer agreement between independent sources, and be explicit
    about confidence rather than silently trusting whichever source happens
    to be read first.

    Priority:
    1. If the listening model and Essentia agree (same pitch class + mode, enharmonics aside),
       report that key with high confidence.
    2. If they disagree, prefer Essentia only when its confidence clears
       min_essentia_strength; otherwise prefer the listening model (it has access to chords/
       melody/bass movement, which is stronger key evidence than Essentia's
       chroma-only profile-matching) but flag the disagreement so the writer
       doesn't overstate certainty.
    3. Fall back to whichever single source is available.
    """
    moss_hedged = False
    if moss_key and _normalize_key_for_compare(moss_key) is None:
        _km = re.search(
            r"\b([A-G](?:#|b|\s*-?\s*(?:sharp|flat))?)\s*(major|minor|maj|min)\b",
            str(moss_key), re.IGNORECASE,
        )
        moss_key = f"{_km.group(1)} {_km.group(2)}" if _km else None
        moss_hedged = moss_key is not None
    moss_norm = _normalize_key_for_compare(moss_key) if moss_key else None
    ess_norm = _normalize_key_for_compare(essentia_key) if essentia_key else None
    if moss_hedged and ess_norm and moss_norm and moss_norm[0] != ess_norm[0]:
        # The listening model itself was unsure; don't let a hedge outvote a measurement.
        return essentia_key, (
            f"MOSS-Music was uncertain (leaned {moss_key}); Essentia measured {essentia_key} "
            f"(strength={essentia_strength if essentia_strength is not None else 'n/a'}). "
            "Treat as moderately confident and hedge."
        )

    if moss_key and essentia_key and moss_norm and ess_norm:
        if moss_norm == ess_norm or (moss_norm[0] == ess_norm[0] and (moss_norm[1] is None or ess_norm[1] is None)):
            if (KEY_MODE_FROM_ESSENTIA and moss_norm[1] is None and ess_norm[1] is not None
                    and not _KEY_MODE_NAME_RE.search(str(moss_key))):
                return essentia_key, (
                    f"MOSS-Music and Essentia agree on the tonal center; MOSS-Music gave '{moss_key}' with no "
                    f"mode, so Essentia's {essentia_key} is used (Essentia strength="
                    f"{essentia_strength if essentia_strength is not None else 'n/a'}).")
            mode_note = " with unspecified mode" if moss_norm[1] != ess_norm[1] else ""
            return moss_key, f"MOSS-Music and Essentia agree on tonal center{mode_note} (Essentia strength={essentia_strength if essentia_strength is not None else 'n/a'})."
        if essentia_strength is not None and essentia_strength >= min_essentia_strength:
            return (
                essentia_key,
                f"MOSS-Music said {moss_key}, Essentia said {essentia_key} (strength={essentia_strength:.2f}, "
                "above confidence threshold) — preferring Essentia's independent chroma-based estimate. "
                "Treat as moderately confident; mention the disagreement is possible if asked."
            )
        return (
            moss_key,
            f"MOSS-Music said {moss_key}, Essentia said {essentia_key} "
            f"(strength={essentia_strength if essentia_strength is not None else 'n/a'}, "
            "below/no confidence threshold) — preferring MOSS-Music's harmonic/melodic reading, "
            "but this is a genuine disagreement between sources; hedge appropriately."
        )

    if moss_key:
        return moss_key, "Only MOSS-Music produced a key estimate."
    if essentia_key:
        if essentia_strength is not None and essentia_strength < min_essentia_strength:
            return essentia_key, f"Only Essentia produced a key estimate, and confidence is low (strength={essentia_strength:.2f}); treat as uncertain."
        return essentia_key, "Only Essentia produced a key estimate."
    return None, "unavailable"

def _build_melody_per_section(sections, stem_notes, bpm=None, key=None):
    """Per-section melodic detail for every pitched stem, from raw MIDI.

    One block per stem, one line per section, plus a whole-line summary that
    names the recurring figures. Every statement here is arithmetic on the
    transcribed note events for THIS recording -- nothing is drawn from
    Wikipedia, file tags, or model knowledge of the song.

    Sections are matched by note ONSET falling inside the section window, with
    a small tolerance at the section head so a pickup note that starts a beat
    early is credited to the phrase it belongs to rather than to the previous
    section.
    """
    if not sections or not stem_notes:
        return ""

    preferred = ["vocals", "guitar", "piano", "bass", "other"]
    available = [s for s in preferred if stem_notes.get(s)]
    if not available:
        return ""

    if FULL_MIDI_TRANSCRIPTION:
        available = available[:4]
    else:
        available = available[:1]

    beat = (60.0 / float(bpm)) if bpm and float(bpm) > 0 else 0.5
    pickup = min(0.75, beat * 1.0)

    blocks = []
    for chosen_stem in available:
        notes = _notes_sorted(stem_notes.get(chosen_stem) or [])
        if not notes:
            continue

        lines = [
            "MELODY PER SECTION (%s stem) "
            "[computed from this recording's transcribed notes]:" % chosen_stem
        ]
        for sec in sections:
            try:
                start = float(sec["start"])
                end = float(sec["end"])
            except Exception:
                continue
            sec_notes = [
                n for n in notes
                if (start - pickup) <= float(n.get("onset", -1)) < end
            ]
            if len(sec_notes) < 2:
                lines.append(
                    "%s (%s-%s): no sustained melodic content detected"
                    % (sec["name"], _seconds_to_mmss(start), _seconds_to_mmss(end))
                )
                continue

            max_n = 14 if FULL_MIDI_TRANSCRIPTION else 8
            summary = _summarize_melody(
                sec_notes, max_notes=max_n, key_text=key, bpm=bpm,
                include_motifs=False,
            )
            if not summary:
                lines.append(
                    "%s (%s-%s): melody unclear"
                    % (sec["name"], _seconds_to_mmss(start), _seconds_to_mmss(end))
                )
                continue
            summary = summary.replace("melodic line", "melody", 1)

            # Phrase-level shape: how the line is actually broken up by rests.
            phrases = _split_melodic_phrases(sec_notes, bpm=bpm, max_phrases=4)
            if len(phrases) >= 2:
                shapes = [_contour_shape(p) for p in phrases]
                shapes = [s for s in shapes if s and s != "short line"]
                if shapes:
                    summary += "; %d phrases (%s)" % (
                        len(phrases), "; then ".join(shapes[:3])
                    )

            lines.append(
                "%s (%s-%s): %s"
                % (sec["name"], _seconds_to_mmss(start), _seconds_to_mmss(end), summary)
            )

        # Whole-line motif summary, so a recurring hook that spans sections is
        # still named even though the per-section lines are computed separately.
        motifs = _extract_motifs(notes, max_motifs=2)
        if motifs:
            phrases = [_motif_phrase(m) for m in motifs]
            phrases = [p for p in phrases if p]
            if phrases:
                lines.append(
                    "Recurring figures across the whole %s line: %s"
                    % (chosen_stem, " | ".join(phrases))
                )

        # Where the line sits highest -- usually the emotional peak.
        try:
            peak_note = max(
                notes, key=lambda n: float(n.get("frequency", 0.0))
            )
            lines.append(
                "Highest point of the %s line: %s at %s"
                % (chosen_stem,
                   _hz_to_note_name(float(peak_note["frequency"])),
                   _seconds_to_mmss(float(peak_note.get("onset", 0.0))))
            )
        except Exception:
            pass

        blocks.append("\n".join(lines))

    return "\n\n".join(blocks)


_CHD_PC = {"C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3, "E": 4, "F": 5, "F#": 6, "Gb": 6, "G": 7,
           "G#": 8, "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11, "Cb": 11, "Fb": 4, "E#": 5, "B#": 0}
_CHD_SHARP = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
_CHD_FLAT = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]
_CHD_MODES = {"major": (0, 2, 4, 5, 7, 9, 11), "ionian": (0, 2, 4, 5, 7, 9, 11), "minor": (0, 2, 3, 5, 7, 8, 10),
              "aeolian": (0, 2, 3, 5, 7, 8, 10), "dorian": (0, 2, 3, 5, 7, 9, 10), "phrygian": (0, 1, 3, 5, 7, 8, 10),
              "lydian": (0, 2, 4, 6, 7, 9, 11), "mixolydian": (0, 2, 4, 5, 7, 9, 10), "locrian": (0, 1, 3, 5, 6, 8, 10)}
_CHD_REL = {"major": 0, "ionian": 0, "minor": 9, "aeolian": 9, "dorian": 2, "phrygian": 4, "lydian": 5,
            "mixolydian": 7, "locrian": 11}
_CHD_TOKEN_RE = re.compile(r"^([A-G])(#|b|\u266f|\u266d)?((?:maj|min|m|dim|aug|sus|add|M)?[0-9]*(?:b5|#5|b9|#9|#11|b13)?"
                           r"(?:sus[24])?(?:add[0-9]+)?)(?:/([A-G](?:#|b|\u266f|\u266d)?))?$")
_CHORDS_MEASURED_MARK = "(measured from the separated instruments"
_CHORDS_NONE_MEASURED = "no clear chords measured"
_CHORDS_HEARD_LABEL = "heard by the listening model:"


def _chd_key(text):
    """(root pc, mode, display, flats?) of RECOMMENDED KEY FOR DISCUSSION, or None."""
    m = re.search(r"RECOMMENDED KEY FOR DISCUSSION: ([A-G](?:#|b|\u266f|\u266d)?)\s+(major|minor|ionian|dorian|phrygian|"
                  r"lydian|mixolydian|aeolian|locrian)\b", str(text or ""), re.I)
    if not m:
        return None
    r = m.group(1).replace("\u266f", "#").replace("\u266d", "b")
    mode = m.group(2).lower()
    if r not in _CHD_PC:
        return None
    if "#" in r:
        flats = False
    elif r.endswith("b"):
        flats = True
    else:
        flats = ((_CHD_PC[r] - _CHD_REL[mode]) % 12) in (5, 10, 3, 8, 1, 6)
    disp = f"{r} {mode}" if mode in ("major", "minor") else f"{r} {mode.capitalize()}"
    return _CHD_PC[r], mode, disp, flats


def _chd_measured(text, key):
    """Measured chords from the guitar / piano / other stems' common harmonic
    clusters: triads, and root-fifth dyads as power chords (X5) on roots with
    no triad; most common first, at least 5 hits and 15% of the top count,
    at most six."""
    t = str(text or "")
    counts = {}
    for stem in ("GUITAR", "PIANO", "OTHER"):
        i = t.find(f"[{stem} stem]")
        if i < 0:
            continue
        j = t.find("\n[", i + 5)
        m = re.search(r"common harmonic clusters: ([^\n]+)", t[i:j if j > 0 else len(t)])
        if not m:
            continue
        for it in m.group(1).split(";"):
            it = it.strip()
            mm = re.match(r"([A-G]#?) (major|minor) x(\d+)", it)
            if mm:
                k = (_CHD_PC[mm.group(1)], "m" if mm.group(2) == "minor" else "")
                counts[k] = counts.get(k, 0) + int(mm.group(3))
                continue
            mm = re.match(r"([A-G]#?)\+([A-G]#?) x(\d+)$", it)
            if mm:
                a, b = _CHD_PC[mm.group(1)], _CHD_PC[mm.group(2)]
                root = a if (b - a) % 12 == 7 else (b if (a - b) % 12 == 7 else None)
                if root is not None:
                    counts[(root, "5")] = counts.get((root, "5"), 0) + int(mm.group(3))
    triad_roots = {r for (r, q) in counts if q in ("", "m")}
    items = sorted(((c, k) for k, c in counts.items() if not (k[1] == "5" and k[0] in triad_roots)),
                   key=lambda x: -x[0])
    if not items:
        return []
    top = items[0][0]
    names = _CHD_FLAT if (key and key[3]) else _CHD_SHARP
    return [names[k[0]] + k[1] for c, k in items if c >= max(5, 0.15 * top)][:6]


def _chd_heard(value):
    """The listening model's chords from its CHORDS answer, each once, in
    order of first appearance (section labels, timestamps and prose dropped)."""
    v = str(value or "").strip()
    if not v or re.match(r"(?i)^(uncertain|none|n/?a|unknown)\b", v):
        return []
    toks = re.split(r"[\s,;:()\[\]\u2192>\-\u2010-\u2015|]+", v)
    prose = sum(1 for x in toks if re.match(r"^[a-z]{3,}$", x)) > 8
    out = []
    for x in toks:
        x = x.strip(".")
        if not _CHD_TOKEN_RE.match(x) or (prose and len(x) == 1):
            continue
        c = x.replace("\u266f", "#").replace("\u266d", "b")
        if c not in out:
            out.append(c)
    return out


def _chd_fits(chord, key):
    """True when the chord belongs to the key (diatonic root and triad quality;
    also bVII major in a major key and V major in a minor key)."""
    if not key:
        return True
    root, mode = key[0], key[1]
    scale = [(root + i) % 12 for i in _CHD_MODES[mode]]
    m = _CHD_TOKEN_RE.match(chord)
    if not m:
        return True
    pc = _CHD_PC.get(m.group(1) + (m.group(2) or "").replace("\u266f", "#").replace("\u266d", "b"))
    if pc is None:
        return True
    q = m.group(3) or ""
    if re.match(r"^(5|sus)", q):
        qual = None
    elif "dim" in q or "m7b5" in q:
        qual = "dim"
    elif re.match(r"^(m(?!aj)|min)", q):
        qual = "m"
    else:
        qual = ""
    if pc in scale:
        if qual is None:
            return True
        d = scale.index(pc)
        third = (scale[(d + 2) % 7] - pc) % 12
        fifth = (scale[(d + 4) % 7] - pc) % 12
        want = "dim" if (third == 3 and fifth == 6) else ("m" if third == 3 else "")
        return qual == want
    if mode in ("major", "ionian") and pc == (root + 10) % 12 and qual in ("", None):
        return True
    if mode in ("minor", "aeolian") and pc == (root + 7) % 12 and qual in ("", None):
        return True
    return False


def _chd_outside_note(chords, key):
    out = [c for c in chords if not _chd_fits(c, key)]
    if not key or not out:
        return ""
    if len(out) * 2 > len(chords):
        return f" (most of these are outside the measured key of {key[2]}; not reliable)"
    return f" ({', '.join(out)} outside the measured key of {key[2]})"


def _chords_line_value(old_value, text):
    """See CHORDS_MEASURED_LINE. The new CHORDS value, or old_value when it
    is already in this format."""
    old = str(old_value or "")
    if _CHORDS_HEARD_LABEL in old:
        return old_value
    key = _chd_key(text)
    meas = _chd_measured(text, key)
    heard = _chd_heard(old)
    ms = ((", ".join(meas) + f" {_CHORDS_MEASURED_MARK}, most common first)" + _chd_outside_note(meas, key))
          if meas else _CHORDS_NONE_MEASURED)
    if heard:
        shown = heard[:8]
        hs = (f"{_CHORDS_HEARD_LABEL} " + ", ".join(shown) + (", …" if len(heard) > 8 else "")
              + _chd_outside_note(shown, key))
    else:
        hs = f"{_CHORDS_HEARD_LABEL} none given"
    return ms + "; " + hs


# --- dev45: chord progressions (lv-chordia) and the melody in the key ---------
# See ENABLE_CHORD_PROGRESSION and MELODY_IN_KEY_BY_SECTION.
_CHORD_REC_LOCK = threading.Lock()
_CHORD_REC_STATE = {"unavailable": None}
_CHORD_RESULTS = collections.OrderedDict()      # track path -> segments, for the save
_CP_HEAD = "CHORD PROGRESSION BY SECTION"
_MK_HEAD = "MELODY IN THE KEY BY SECTION"
_CP_QUALITY = {"maj": "", "min": "m", "7": "7", "maj7": "maj7", "min7": "m7", "dim": "dim", "aug": "aug",
               "sus2": "sus2", "sus4": "sus4", "sus4(b7)": "7sus4", "9": "9", "maj9": "maj9", "min9": "m9",
               "11": "11", "13": "13", "dim7": "dim7", "hdim7": "m7b5", "maj6": "6", "min6": "m6"}
_CP_MINORISH = {"min", "min7", "min9", "dim", "dim7", "hdim7", "min6"}
_CP_BASS_INTERVAL = {"2": 2, "b3": 3, "3": 4, "4": 5, "b5": 6, "5": 7, "#5": 8, "b6": 8, "6": 9, "bb7": 9,
                     "b7": 10, "7": 11, "9": 2}
_CP_ROMAN = {0: "I", 1: "bII", 2: "II", 3: "bIII", 4: "III", 5: "IV", 6: "#IV", 7: "V", 8: "bVI", 9: "VI",
             10: "bVII", 11: "VII"}
_CP_DIATONIC = {"major": {0: "maj", 2: "min", 4: "min", 5: "maj", 7: "maj", 9: "min", 11: "dim"},
                "minor": {0: "min", 2: "dim", 3: "maj", 5: "min", 7: "min", 8: "maj", 10: "maj"}}
_CP_ALSO_IN_KEY = {"minor": {7: "maj", 11: "dim"}}        # the harmonic-minor V and vii
_CP_SHARP = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
_CP_FLAT = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]
_CP_FLAT_KEYS = {("major", 5), ("major", 10), ("major", 3), ("major", 8), ("major", 1), ("major", 6),
                 ("minor", 2), ("minor", 7), ("minor", 0), ("minor", 5), ("minor", 10), ("minor", 3)}
_CP_ROOT_PC = {"C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3, "E": 4, "F": 5, "F#": 6, "Gb": 6, "G": 7,
               "G#": 8, "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11, "Cb": 11, "Fb": 4, "E#": 5, "B#": 0}
_MK_DEGREE = {0: "1", 1: "b2", 2: "2", 3: "b3", 4: "3", 5: "4", 6: "#4", 7: "5", 8: "b6", 9: "6", 10: "b7",
              11: "7"}


def chord_segments_for(audio_path):
    """[[start, end, label]] from lv-chordia ("A:maj", "D:maj/3", "N"), or [] when
    disabled, not installed or failed. CPU only; never raises. A first failure to
    import or load is not retried in this session."""
    if not ENABLE_CHORD_PROGRESSION or not audio_path or str(audio_path).startswith(("http://", "https://")):
        return []
    if _CHORD_REC_STATE["unavailable"]:
        return []
    try:
        with _CHORD_REC_LOCK:
            try:
                import importlib
                _lvc = importlib.import_module("lv_chordia.chord_recognition")   # the module, not the function
            except Exception as e:
                _CHORD_REC_STATE["unavailable"] = str(e) or type(e).__name__
                print("  (chord progressions skipped for this session -- install the chord model once with "
                      f"'pip install lv-chordia': {_CHORD_REC_STATE['unavailable'][:160]})")
                return []
            _lvc.print = lambda *a, **k: None        # its per-model progress lines
            res = _lvc.chord_recognition(os.path.abspath(str(audio_path)), CHORD_PROGRESSION_DICTIONARY)
        out = []
        for x in res or []:
            try:
                s, e, lab = float(x["start_time"]), float(x["end_time"]), str(x["chord"])
            except Exception:
                continue
            if e > s:
                out.append([round(s, 2), round(e, 2), lab])
        return out
    except Exception as e:
        print(f"  (chord progressions skipped: {str(e).strip()[:160] or type(e).__name__})")
        return []


def _chord_remember(track_path, segs):
    if not segs or not track_path:
        return
    _CHORD_RESULTS[str(track_path)] = segs
    _CHORD_RESULTS.move_to_end(str(track_path))
    while len(_CHORD_RESULTS) > 16:
        _CHORD_RESULTS.popitem(last=False)


def _chord_segments_for_save(segs, track_key=None):
    """The saved form ([[start, end, label]]), from `segs` or the remembered result."""
    src = segs if segs else (_CHORD_RESULTS.get(str(track_key)) if track_key else None)
    out = []
    for it in src or []:
        try:
            out.append([round(float(it[0]), 2), round(float(it[1]), 2), str(it[2])])
        except Exception:
            continue
    return out


def _cp_parse(label):
    """(root pc, quality, bass pc or None) from "A:maj/3"; None for no chord."""
    lab = str(label or "").strip()
    if not lab or lab in ("N", "X") or ":" not in lab:
        return None
    root, _, rest = lab.partition(":")
    pc = _CP_ROOT_PC.get(root)
    if pc is None:
        return None
    qual, _, bass = rest.partition("/")
    if qual not in _CP_QUALITY:
        return None
    bpc = None
    if bass:
        iv = _CP_BASS_INTERVAL.get(bass)
        if iv is not None and iv != 0:
            bpc = (pc + iv) % 12
    return (pc, qual, bpc)


def _cp_key(key_text):
    k = _parse_key_root_quality(key_text) if key_text else None
    return k  # (root pc, "major"/"minor") or None


def _cp_name(ch, key):
    names = _CP_FLAT if key and (key[1], key[0]) in _CP_FLAT_KEYS else _CP_SHARP
    pc, qual, bpc = ch
    s = names[pc] + _CP_QUALITY.get(qual, "")
    return s + ("/" + names[bpc] if bpc is not None else "")


def _cp_roman(ch, key):
    pc, qual, _b = ch
    iv = (pc - key[0]) % 12
    r = _CP_ROMAN[iv]
    head = r.lstrip("b#")
    acc = r[: len(r) - len(head)]
    if qual in _CP_MINORISH:
        head = head.lower()
    suf = {"dim": "o", "dim7": "o7", "hdim7": "ø7", "aug": "+", "7": "7",
           "maj7": "maj7", "min7": "7", "9": "9", "maj9": "maj9", "min9": "9", "11": "11", "13": "13",
           "sus2": "sus2", "sus4": "sus4", "sus4(b7)": "7sus4", "maj6": "6", "min6": "6"}.get(qual, "")
    return acc + head + suf


def _cp_in_key(ch, key):
    pc, qual, _b = ch
    iv = (pc - key[0]) % 12
    want = _CP_DIATONIC[key[1]].get(iv) or _CP_ALSO_IN_KEY.get(key[1], {}).get(iv)
    if want is None:
        return False
    if qual.startswith("sus") or qual in ("aug",):
        return True
    fam = "dim" if qual in ("dim", "dim7", "hdim7") else ("min" if qual in _CP_MINORISH else "maj")
    if want == "dim":
        return fam in ("dim", "min")
    return fam == want or (key[1] == "minor" and iv in _CP_ALSO_IN_KEY["minor"])


def _cp_loop(seq):
    """(pattern, repeats) for the shortest chord cycle covering 80%+ of the
    sequence at least twice, else None."""
    n = len(seq)
    for k in range(1, min(8, n // 2) + 1):
        best = None
        for off in range(k):
            pat = seq[off:off + k]
            if len(pat) < k:
                continue
            hits = sum(1 for i in range(n) if seq[i] == pat[(i - off) % k])
            if best is None or hits > best[0]:
                best = (hits, pat, off)
        if best and best[0] / float(n) >= 0.8 and n / float(k) >= 2:
            pat, off = best[1], best[2]
            if off:          # show the cycle from its first full statement
                pat = seq[off:off + k]
            return pat, int(round(n / float(k)))
    return None


def _cp_rhythm(durs, bar):
    if not durs or not bar:
        return ""
    r = float(np.median(durs)) / bar
    if r < 0.4:
        return "the chords change every beat or two"
    if r < 0.75:
        return "about two chords per bar"
    if r < 1.5:
        return "about one chord per bar"
    if r < 3.0:
        return "about one chord every two bars"
    return "long, held chords"


_CP_TIMES = {2: "twice", 3: "three times", 4: "four times", 5: "five times", 6: "six times", 7: "seven times",
             8: "eight times"}


def build_chord_progression_block(segments, sections, key_text=None, bpm=None):
    """See ENABLE_CHORD_PROGRESSION. "" when there is nothing to say. Never raises."""
    try:
        if not segments or not sections:
            return ""
        key = _cp_key(key_text)
        segs = []
        for it in segments:
            try:
                ch = _cp_parse(it[2])
                if ch:
                    segs.append((float(it[0]), float(it[1]), ch))
            except Exception:
                continue
        if len(segs) < 2:
            return ""
        beat = 60.0 / float(bpm) if bpm and float(bpm) > 0 else 0.5
        bar = 4 * beat
        lines, loops = [], []
        for sec in sections:
            try:
                a, b = float(sec["start"]), float(sec["end"])
            except Exception:
                continue
            name = str(sec.get("name") or "").strip() or "Section"
            if b - a < 3.0 or name.lower().startswith("silence"):
                continue
            seq = _cp_section_seq(segs, a, b, beat)
            if not seq:
                lines.append(f"  {name} ({_seconds_to_mmss(a)}-{_seconds_to_mmss(b)}): no clear chords")
                continue
            chords = [c for c, _d in seq]
            durs = [d for _c, d in seq]
            lp = _cp_loop(chords)
            shown = lp[0] if lp else chords[:8]
            names = " - ".join(_cp_name(c, key) for c in shown)
            if lp and len(lp[0]) == 1:
                desc = f"stays on {names}"
            elif lp:
                desc = f"{names}, {_CP_TIMES.get(lp[1], str(lp[1]) + ' times')}"
            else:
                desc = names + (" ..." if len(chords) > 8 else "")
            if key:
                desc += " (" + " - ".join(_cp_roman(c, key) for c in shown) + (" ..." if not lp and len(chords) > 8
                                                                              else "") + ")"
            parts = [desc]
            rh = _cp_rhythm(durs, bar)
            if rh and len(chords) > 1:
                parts.append(rh)
            if key:
                out = []
                for c in dict.fromkeys(chords):
                    if not _cp_in_key(c, key):
                        out.append(f"{_cp_name(c, key)} ({_cp_roman(c, key)})")
                if out:
                    tot = sum(durs)
                    od = sum(d for c, d in seq if not _cp_in_key(c, key))
                    if tot and od / tot >= 0.6:
                        parts.append("most chords here lie outside the key (a key change or a modal passage)")
                    else:
                        parts.append("outside the key: " + ", ".join(out[:4]))
            if lp and len(lp[0]) > 1:
                loops.append(tuple(lp[0]))
            lines.append(f"  {name} ({_seconds_to_mmss(a)}-{_seconds_to_mmss(b)}): " + "; ".join(parts))
        if not lines:
            return ""
        head = (f"{_CP_HEAD} (measured with a chord-recognition model on the full mix, in TIME ORDER -- unlike the "
                "ranked chord lists elsewhere, this IS the progression; repeats of one cycle are folded into "
                "'..., N times'" + (f"; Roman numerals are against the recommended key, {key_text}" if key else "")
                + "; chord names can be off on dense or distorted mixes, so treat extensions such as 7ths and "
                "sus chords as likely rather than certain):")
        summary = ""
        if loops:
            canon = {}
            for lp in loops:
                rots = [lp[i:] + lp[:i] for i in range(len(lp))]
                c = min(rots, key=lambda r: [x[0] for x in r])
                canon.setdefault(c, []).append(lp)
            best = max(canon.items(), key=lambda kv: len(kv[1]))
            if len(best[1]) >= 2:
                lp = max(set(best[1]), key=best[1].count)
                summary = ("\n  Most common cycle: " + " - ".join(_cp_name(c, key) for c in lp)
                           + (" (" + " - ".join(_cp_roman(c, key) for c in lp) + ")" if key else "")
                           + f", in {len(best[1])} sections")
        return head + summary + "\n" + "\n".join(lines)
    except Exception:
        return ""


def _mk_phrases(lead):
    """Phrases (lists of (onset, midi)) from (onset, dur, midi) notes: very short notes
    dropped, a gap of 0.45 s or more starts a new phrase."""
    notes = sorted((on, d, m) for on, d, m in lead if d >= 0.08)
    out, cur, last_end = [], [], None
    for on, d, m in notes:
        if cur and last_end is not None and on - last_end >= 0.45:
            out.append(cur)
            cur = []
        cur.append((on, int(round(m))))
        last_end = on + d
    if cur:
        out.append(cur)
    return out


_MK_SCALE = {"major": {0, 2, 4, 5, 7, 9, 11}, "minor": {0, 2, 3, 5, 7, 8, 10}}
_MK_END_WORDS = {"1": "1 (the tonic: settled, finished-sounding)", "5": "5 (the fifth: open, unresolved)",
                 "3": "3 (the third: warm but not final)", "b3": "b3 (the minor third: unsettled)"}


def _build_melody_in_key_by_section(sections, stem_notes, key=None, instrumental=False, context_text=""):
    """See MELODY_IN_KEY_BY_SECTION. "" when there is nothing to say. Never raises.
    Only statements that survive transcription slips are made: the degrees the
    held notes centre on, where phrases come to rest and the highest note."""
    if not MELODY_IN_KEY_BY_SECTION or not sections or not stem_notes or not key:
        return ""
    try:
        kq = _parse_key_root_quality(key)
        if not kq:
            return ""
        tonic = kq[0]
        scale = _MK_SCALE[kq[1]]
        pitched = {s_: _hm_pitched(stem_notes.get(s_)) for s_ in ("vocals", "bass") + _HM_ACC_STEMS}
        vocal_absent = _hm_vocal_absent(context_text)
        lines = []
        for sec in sections:
            try:
                a, b = float(sec["start"]), float(sec["end"])
            except Exception:
                continue
            name = str(sec.get("name") or "").strip() or "Section"
            if b - a < 3.0 or name.lower().startswith("silence"):
                continue
            lead_stem, lead = _hm_lead(pitched, a, b, instrumental, vocal_absent)
            if not lead:
                continue
            # Short notes, and short notes outside the key above all, are mostly transcription slips
            # (on the scanned library 69% of notes under 0.15 s fit the key, 90% of held ones).
            clean = [x for x in lead if x[1] >= 0.15 and (x[1] >= 0.25 or (int(round(x[2])) - tonic) % 12 in scale)]
            held = [x for x in clean if x[1] >= 0.25]
            if len(held) < 6:
                continue
            parts = ["the voice" if lead_stem == "vocals" else f"the {lead_stem} stem"]
            w = {}
            for on, d, m in held:
                dg = _MK_DEGREE[(int(round(m)) - tonic) % 12]
                w[dg] = w.get(dg, 0.0) + d
            tot = sum(w.values())
            fit = sum(d for _on, d, m in held if (int(round(m)) - tonic) % 12 in scale) / tot if tot else 0.0
            if fit < MELODY_IN_KEY_MIN_FIT:
                lines.append(f"  {name} ({_seconds_to_mmss(a)}-{_seconds_to_mmss(b)}): {parts[0]}; its held notes "
                             "do not sit clearly in the key here (a passing modulation, or an unreliable "
                             "transcription) -- describe this part by its contour only")
                continue
            top = [dg for dg, v in sorted(w.items(), key=lambda kv: -kv[1])[:3] if v / tot >= 0.15][:2]
            if top:
                parts.append("its held notes centre on " + " and ".join(top))
            ends = []
            for p in _mk_phrases(clean):
                if len(p) >= 3:
                    ends.append(_MK_DEGREE[(p[-1][1] - tonic) % 12])
            if len(ends) >= 3:
                tdg, cnt = max(((e, ends.count(e)) for e in set(ends)), key=lambda x: (x[1], x[0] == "1"))
                if cnt / float(len(ends)) >= 0.4:
                    parts.append("phrases mostly end on " + _MK_END_WORDS.get(tdg, tdg))
                elif ends.count("1") / float(len(ends)) < 0.15:
                    parts.append("phrases rarely come to rest on the key note")
            hi = max(held, key=lambda x: x[2])
            parts.append(f"peaks on {_MK_DEGREE[(int(round(hi[2])) - tonic) % 12]} ({_hm_name(hi[2])})")
            lines.append(f"  {name} ({_seconds_to_mmss(a)}-{_seconds_to_mmss(b)}): " + "; ".join(parts))
        if not lines or sum(1 for ln in lines if "do not sit clearly" not in ln) < max(1, len(lines) // 2):
            return ""          # most parts do not fit the key: the key or the transcription is in doubt
        root_name = (_CP_FLAT if (kq[1], kq[0]) in _CP_FLAT_KEYS else _CP_SHARP)[tonic]
        head = (f"{_MK_HEAD} (the lead line against the key, {key}: 1 = {root_name}; degrees are numbered against "
                "the major scale, so b3, b6 and b7 are the minor-key notes; from the longer, held notes of this "
                "recording's transcription -- short notes, often transcription slips, are left out -- so these are "
                "the line's resting points and peaks, not a note-by-note score):")
        return head + "\n" + "\n".join(lines)
    except Exception:
        return ""


# dev45 MELODICISM_MEASURED: how melodic the lead line is, from two measured
# things that separate sung from rhythm-led delivery on the scanned library:
# the word rate of the timed lyric lines and the share of long, held notes in
# the vocal transcription (library means/spreads below, from 185 scans).
_ML_HEAD = "MELODICISM BY SECTION (measured)"
_ML_WPS_MEAN, _ML_WPS_SD = 2.046, 0.699
_ML_HELD_MEAN, _ML_HELD_SD = 0.1585, 0.0616


def _ml_chunk_times(c):
    try:
        if isinstance(c, dict):
            if "timestamp" in c and c["timestamp"]:
                a, b = c["timestamp"][0], c["timestamp"][1]
            else:
                a, b = c.get("start"), c.get("end")
            return float(a), float(b), str(c.get("text") or "")
    except Exception:
        return None
    return None


def _ml_wps(chunks, a=None, b=None):
    """(words per second, seconds of timed lyrics) for lines whose middle is in [a, b)."""
    words = secs = 0.0
    for c in chunks or []:
        t = _ml_chunk_times(c)
        if not t:
            continue
        s, e, txt = t
        if e <= s or e - s > 12:
            continue
        mid = (s + e) / 2.0
        if a is not None and not (a <= mid < b):
            continue
        words += len(re.findall(r"[^\W\d_]+(?:'[^\W\d_]+)?", txt))
        secs += e - s
    return (words / secs if secs else None), secs


def _ml_range_words(st):
    if st <= 4:
        return "a narrow range (a few neighbouring notes)"
    if st <= 7:
        return "a moderate range (about a fourth to a fifth)"
    if st <= 11:
        return "a wide range (most of an octave)"
    return "a very wide range (an octave or more)"


def _ml_held_words(h):
    if h < 0.10:
        return "few held notes (short, clipped syllables)"
    if h <= 0.20:
        return "some held notes"
    return "many long, held notes"


def _ml_rate_words(w):
    if w >= 2.8:
        return "a fast, rap-like word rate"
    if w >= 2.2:
        return "a brisk word rate"
    if w >= 1.6:
        return "a moderate word rate"
    return "an unhurried word rate"


def _ml_band(score):
    if score <= -1.5:
        return "mostly rhythm-led: a fast, rapped, chanted or spoken delivery with few held notes"
    if score <= -0.5:
        return "partly melodic: a brisk, wordy delivery with some sung, held lines"
    if score <= 0.8:
        return "melodic: a sung line at a moderate pace"
    return "strongly melodic: unhurried singing with many long, held notes"


def _ml_stats(notes):
    v = [x for x in notes if x[1] >= 0.08]
    if len(v) < 12:
        return None
    r = np.round([x[2] for x in v]).astype(int)
    d = np.array([x[1] for x in v])
    return {"range": float(np.percentile(r, 90) - np.percentile(r, 10)), "held": float(np.mean(d >= 0.4)),
            "notes": len(v)}


def _build_melodicism_block(sections, stem_notes, lyric_chunks=None, instrumental=False, context_text=""):
    """See MELODICISM_MEASURED. "" when there is nothing to say. Never raises."""
    if not MELODICISM_MEASURED or not stem_notes or instrumental:
        return ""
    try:
        voc = _hm_pitched(stem_notes.get("vocals"))
        whole = _ml_stats(voc)
        if not whole or whole["notes"] < 40:
            return ""
        wps, wsecs = _ml_wps(lyric_chunks)
        if wps is not None and wsecs < 20:
            wps = None
        parts = [_ml_range_words(whole["range"]), _ml_held_words(whole["held"])]
        if wps is not None:
            parts.append(_ml_rate_words(wps))
            score = -(wps - _ML_WPS_MEAN) / _ML_WPS_SD + (whole["held"] - _ML_HELD_MEAN) / _ML_HELD_SD
            overall = f"Overall: {_ml_band(score)} -- " + ", ".join(parts) + "."
        else:
            overall = ("Overall: " + ", ".join(parts) + " (no timed lyrics, so the word rate -- the clearest sign of "
                       "sung versus rapped delivery -- is unknown; do not judge how melodic it is from this alone).")
        lines, ranked = [], []
        vocal_absent = _hm_vocal_absent(context_text)
        for sec in sections or []:
            try:
                a, b = float(sec["start"]), float(sec["end"])
            except Exception:
                continue
            name = str(sec.get("name") or "").strip() or "Section"
            if b - a < 6.0 or name.lower().startswith("silence"):
                continue
            if f"{_seconds_to_mmss(a)}-{_seconds_to_mmss(b)}" in vocal_absent:
                continue
            st = _ml_stats([x for x in voc if a <= x[0] < b])
            if not st:
                continue
            sp = [_ml_range_words(st["range"]).split(" (")[0], _ml_held_words(st["held"]).split(" (")[0]]
            sw, ss = _ml_wps(lyric_chunks, a, b)
            if sw is not None and ss >= 4:
                sp.append(_ml_rate_words(sw))
                sc = -(sw - _ML_WPS_MEAN) / _ML_WPS_SD + (st["held"] - _ML_HELD_MEAN) / _ML_HELD_SD
                ranked.append((sc, name))
                sp.insert(0, _ml_band(sc).split(":")[0])
            lines.append(f"  {name} ({_seconds_to_mmss(a)}-{_seconds_to_mmss(b)}): " + ", ".join(sp))
        contrast = ""
        if len(ranked) >= 2:
            hi, lo = max(ranked), min(ranked)
            if hi[0] - lo[0] >= 1.0:
                contrast = (f"\n  Most melodic part: {hi[1]}; most rhythm-led part: {lo[1]}.")
        head = (f"{_ML_HEAD} (from the lead vocal's transcribed notes -- their range and how many are long, held "
                "notes -- and the word rate of the timed lyric lines; together these separated sung from rapped or "
                "chanted songs on the scanned library: 9 of 11 rap tracks fall in the rhythm-led band and none of "
                "9 ballads do. Judged on the voice only; a song can still be melodic in its instruments):")
        return head + "\n  " + overall + contrast + ("\n" + "\n".join(lines) if lines else "")
    except Exception:
        return ""


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
    """Optional *corroboration* pass for key estimation.

    Genre is never allowed to rewrite the primary key by itself. The returned
    candidate is informational so callers can expose disagreement without
    introducing circular genre -> key -> genre reasoning.
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
        if previous_key and _normalize_key_for_compare(previous_key) and _normalize_key_for_compare(key2) == _normalize_key_for_compare(previous_key):
            return previous_key, previous_strength, (
                f"genre-conditioned Essentia reaffirmed {key2} (prior={genre_hint}, strength={str2})."
            )
        return previous_key, previous_strength, (
            f"genre-conditioned Essentia independently suggested {key2} "
            f"(prior={genre_hint}, strength={str2}); primary key was not changed by genre."
        )
    except Exception:
        return previous_key, previous_strength, ""

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


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


def _note_midi_number(freq):
    """Continuous-to-integer MIDI note number from a frequency in Hz."""
    try:
        f = float(freq)
        if f <= 0:
            return None
        return int(round(69 + 12 * math.log2(f / 440.0)))
    except Exception:
        return None


def _notes_sorted(notes):
    """Notes in time order, skipping any whose onset isn't a usable number.

    The onset is parsed once and carried through to the sort key. Previously it
    was parsed, discarded, and then parsed again inside the sort lambda -- which
    left the sort able to raise on input the filter above it had just approved,
    and made the filtering look accidental rather than deliberate.
    """
    out = []
    for n in (notes or []):
        if not isinstance(n, dict):
            continue
        try:
            onset = float(n.get("onset", 0.0))
        except (TypeError, ValueError):
            continue
        out.append((onset, n))
    out.sort(key=lambda pair: pair[0])
    return [n for _onset, n in out]


def _split_melodic_phrases(notes, bpm=None, max_phrases=8):
    """Split a line into phrases at rests, the way a singer actually breathes.

    A melody is not one undifferentiated stream of pitches; it is a sequence of
    phrases separated by rests. Splitting on rest length is what makes it
    possible to say "the verse phrase rises then falls back to the tonic"
    instead of only being able to report an average contour over the whole
    section. Gap threshold scales with tempo so it means the same musical
    amount of silence at any BPM.
    """
    seq = _notes_sorted(notes)
    if not seq:
        return []
    beat = (60.0 / float(bpm)) if bpm and float(bpm) > 0 else 0.5
    # A rest of about a beat is a phrase break to a listener. 1.5 beats was
    # wide enough that ordinary phrase gaps fell under it and every line came
    # back as a single undifferentiated phrase.
    gap_threshold = max(0.35, beat * 0.9)

    phrases = []
    current = [seq[0]]
    for prev, cur in zip(seq, seq[1:]):
        try:
            prev_end = float(prev.get("offset", prev.get("onset", 0.0)))
            cur_start = float(cur.get("onset", 0.0))
        except Exception:
            current.append(cur)
            continue
        if cur_start - prev_end >= gap_threshold:
            phrases.append(current)
            current = [cur]
        else:
            current.append(cur)
    phrases.append(current)

    phrases = [p for p in phrases if len(p) >= 2]
    return phrases[:max_phrases]


def _interval_sequence(notes):
    """Semitone intervals between consecutive notes -- the shape of the line,
    independent of what key it happens to be in."""
    seq = _notes_sorted(notes)
    pitches = []
    for n in seq:
        m = _note_midi_number(n.get("frequency"))
        if m is not None:
            pitches.append(m)
    return [b - a for a, b in zip(pitches, pitches[1:])], pitches


def _extract_motifs(notes, min_len=3, max_len=6, max_motifs=3):
    """Find the actual repeated melodic figures in a line.

    Motifs are matched on INTERVAL patterns rather than absolute pitches, so a
    figure still counts as the same figure when it is transposed -- which is
    how riffs and hooks normally recur. Returns a list of
    {"intervals", "count", "first_onset", "notes", "span"} ordered by how
    strongly each figure characterises the line.

    This exists because contour + range + step/leap ratio can all be reported
    correctly and still leave the actual identifiable tune undescribed. The
    repeated figure is usually the thing a listener would call "the melody".
    """
    intervals, pitches = _interval_sequence(notes)
    if len(intervals) < min_len:
        return []
    seq = _notes_sorted(notes)

    found = {}
    n = len(intervals)
    for length in range(min_len, min(max_len, n) + 1):
        seen = {}
        for i in range(n - length + 1):
            pat = tuple(intervals[i:i + length])
            # A run of identical pitches is not a motif.
            if all(v == 0 for v in pat):
                continue
            seen.setdefault(pat, []).append(i)
        for pat, positions in seen.items():
            # Only count non-overlapping occurrences.
            kept = []
            for p in positions:
                if not kept or p - kept[-1] >= length:
                    kept.append(p)
            if len(kept) >= 2:
                found[pat] = kept

    if not found:
        return []

    # Prefer longer figures that recur often; a 5-note figure heard 3 times is
    # more characteristic than a 3-note fragment heard 4 times.
    scored = sorted(
        found.items(),
        key=lambda kv: (len(kv[0]) * len(kv[1]), len(kv[0])),
        reverse=True,
    )

    results = []
    claimed = set()
    for pat, positions in scored:
        span = set()
        for p in positions:
            span.update(range(p, p + len(pat) + 1))
        # Skip figures that are just sub-patterns of one already reported.
        if len(span & claimed) > 0.6 * len(span):
            continue
        claimed.update(span)
        first = positions[0]
        motif_notes = seq[first:first + len(pat) + 1]
        results.append({
            "intervals": list(pat),
            "count": len(positions),
            "first_onset": float(motif_notes[0].get("onset", 0.0)) if motif_notes else 0.0,
            "notes": motif_notes,
            "span": len(pat) + 1,
        })
        if len(results) >= max_motifs:
            break
    return results


def _describe_intervals(intervals):
    """Turn an interval pattern into plain musical language."""
    names = {
        0: "repeat", 1: "semitone up", -1: "semitone down",
        2: "whole step up", -2: "whole step down",
        3: "minor 3rd up", -3: "minor 3rd down",
        4: "major 3rd up", -4: "major 3rd down",
        5: "4th up", -5: "4th down",
        7: "5th up", -7: "5th down",
        12: "octave up", -12: "octave down",
    }
    parts = []
    for iv in intervals:
        if iv in names:
            parts.append(names[iv])
        else:
            parts.append("%d semitones %s" % (abs(iv), "up" if iv > 0 else "down"))
    return ", ".join(parts)


def _motif_phrase(motif):
    """Render one motif as a short quotable figure, e.g.
    'G3-A3-C4 (up a whole step, then a minor 3rd), heard 4x from 0:31'."""
    seq = motif.get("notes") or []
    if not seq:
        return ""
    try:
        names = [librosa.hz_to_note(float(n["frequency"]), unicode=False) for n in seq]
    except Exception:
        try:
            names = [librosa.hz_to_note(float(n["frequency"])) for n in seq]
        except Exception:
            return ""
    shape = _describe_intervals(motif.get("intervals") or [])
    return "%s (%s), recurs %dx from %s" % (
        "-".join(names), shape, int(motif.get("count") or 0),
        _seconds_to_mmss(float(motif.get("first_onset") or 0.0)),
    )


def _rhythmic_profile(notes, bpm=None):
    """Note density and typical note length, in musical units when BPM is known."""
    seq = _notes_sorted(notes)
    if len(seq) < 3:
        return ""
    try:
        span = float(seq[-1].get("offset", seq[-1].get("onset", 0.0))) - float(
            seq[0].get("onset", 0.0)
        )
    except Exception:
        return ""
    if span <= 0.5:
        return ""
    density = len(seq) / span
    durations = []
    for n in seq:
        try:
            d = float(n.get("duration",
                            float(n.get("offset", 0.0)) - float(n.get("onset", 0.0))))
            if d > 0:
                durations.append(d)
        except Exception:
            continue
    if not durations:
        return ""
    med_dur = float(np.median(durations))

    desc = "note density %.1f notes/sec" % density
    if bpm and float(bpm) > 0:
        beat = 60.0 / float(bpm)
        in_beats = med_dur / beat
        if in_beats >= 1.75:
            unit = "half notes or longer"
        elif in_beats >= 0.85:
            unit = "quarter notes"
        elif in_beats >= 0.42:
            unit = "eighth notes"
        elif in_beats >= 0.21:
            unit = "sixteenth notes"
        else:
            unit = "faster than sixteenths"
        desc += "; typical note length around %s" % unit
    else:
        desc += "; median note length %.2fs" % med_dur

    sustained = sum(1 for d in durations if d >= 0.6)
    if sustained / len(durations) > 0.4:
        desc += "; many sustained notes"
    return desc


def _contour_shape(notes):
    """Describe the arc of a line from its actual pitch trajectory.

    Reported over thirds of the line rather than a single first-half /
    second-half median comparison, so an arch (rise then fall) and a steady
    climb are no longer described identically.
    """
    seq = _notes_sorted(notes)
    pitches = [_note_midi_number(n.get("frequency")) for n in seq]
    pitches = [p for p in pitches if p is not None]
    if len(pitches) < 4:
        return "short line"
    arr = np.array(pitches, dtype=float)
    third = max(1, len(arr) // 3)
    a = float(np.median(arr[:third]))
    b = float(np.median(arr[third:2 * third]))
    c = float(np.median(arr[2 * third:]))

    up_ab, up_bc = (b - a) > 1.0, (c - b) > 1.0
    dn_ab, dn_bc = (a - b) > 1.0, (b - c) > 1.0

    if up_ab and up_bc:
        return "climbs steadily"
    if dn_ab and dn_bc:
        return "falls steadily"
    if up_ab and dn_bc:
        return "arches up then falls back"
    if dn_ab and up_bc:
        return "dips then rises"
    if abs(c - a) <= 1.0 and float(np.std(arr)) < 2.0:
        return "stays around one pitch centre"
    return "moves up and down without a single overall direction"


# --- Melody theory helpers ---------------------------------------------
# These turn raw transcribed notes into the kind of vocabulary a musician
# would actually use (steps vs leaps, scale-degree/tonal-center emphasis)
# WITHOUT inventing anything: every claim here is derived arithmetically
# from the same note events already in evidence, never guessed.

_MAJOR_SCALE_DEGREES = {
    0: "1 (tonic)", 1: "b2", 2: "2", 3: "b3 (blue/minor 3rd)", 4: "3",
    5: "4", 6: "#4/b5", 7: "5 (dominant)", 8: "b6", 9: "6",
    10: "b7", 11: "7 (leading tone)",
}
_MINOR_SCALE_DEGREES = {
    0: "1 (tonic)", 1: "b2", 2: "2", 3: "b3", 4: "3 (major 3rd, borrowed)",
    5: "4", 6: "#4/b5", 7: "5 (dominant)", 8: "b6", 9: "6 (natural 6, borrowed)",
    10: "b7", 11: "7 (leading tone, borrowed)",
}
_NOTE_LETTER_PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}

# --- Modal inflection detection --------------------------------------------
# Essentia's KeyExtractor (and therefore _parse_key_root_quality) only ever
# emits major/minor, so a Dorian track lands on "minor" and a Mixolydian one on
# "major". The transcribed notes usually still SAY which mode it is: what
# separates Dorian from natural minor is a natural 6th where the b6 would be,
# and Mixolydian from major is a b7 where the leading tone would be.
#
# This is deliberately a characteristic-degree test rather than full seven-note
# template matching. Templates score badly here because all these modes share
# six of seven notes, so the shared tones swamp the one degree that actually
# decides it. Comparing the deciding pair directly is both more reliable and
# easier to state as evidence.
#
# Nothing else reads this: _parse_key_root_quality, reconcile_key, the
# RECOMMENDED KEY block and the saved JSON schema are all untouched, and the
# detector returns "" whenever the evidence is thin or ambiguous, in which case
# output is byte-identical to before.
#
# quality -> (mode, parent name, characteristic semitone, contrasting semitone,
#             plain-language name for the characteristic degree)
_MODAL_INFLECTION_TESTS = {
    "minor": [
        ("Dorian", "natural minor", 9, 8, "natural 6th"),
        ("Phrygian", "natural minor", 1, 2, "flat 2nd"),
    ],
    "major": [
        ("Mixolydian", "major", 10, 11, "flat 7th"),
        ("Lydian", "major", 6, 5, "sharp 4th"),
    ],
}

# Conservative gates: a wrong modal claim is far worse than a missing one, so
# the detector stays silent unless the deciding degree is unambiguous.
# (MODAL_DETECTION_MIN_NOTES moved to the settings file beside this script.)
# (MODAL_DETECTION_MIN_SHARE moved to the settings file beside this script.)
# (MODAL_DETECTION_MIN_RATIO moved to the settings file beside this script.)


_FLAT_NOTE_NAMES = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]


def _pc_duration_shares(notes):
    """(12 duration shares, note count) for a stem's note dicts."""
    h = [0.0] * 12
    n = 0
    for x in notes or []:
        try:
            pc = _hz_to_pitch_class(x.get("frequency"))
            d = float(x.get("offset", 0.0)) - float(x.get("onset", 0.0))
        except Exception:
            continue
        if pc is None or not (d > 0):
            continue
        h[pc % 12] += d
        n += 1
    t = sum(h)
    return ([v / t for v in h] if t > 0 else h), n


def _key_centre_reading(stem_notes, key_text):
    """Mixolydian on the fifth, when a MAJOR key's music centres on its fifth.

    Essentia matches the notes against major and minor templates only, so a
    song in D with a C chord (D Mixolydian: D E F# G A B C) is named G major,
    the major key with exactly those notes. The transcribed notes still show
    where the music sits. All of these must hold:

      - the low part (bass stem, KEY_CENTRE_MIN_NOTES notes or more) and the
        vocal line (the same) both use the fifth more than any other note,
        each KEY_CENTRE_MIN_RATIO times as much as the reported root;
      - the chord parts (guitar, piano, other) use the fifth at least as much
        as the root;
      - the fifth's own flat 7th (the reported key's 4th) carries at least
        KEY_CENTRE_FLAT7_MIN_SHARE of the chord parts and KEY_CENTRE_FLAT7_RATIO
        times its leading tone, so it is not simply the key a fifth up.

    Of 36 saved scans this fires on one (an acoustic country song in D, named
    G major); songs whose bass alone sits on the fifth (A1, the dance-pop test track) are kept
    by the vocal test. Returns None, or a dict with the new key and evidence."""
    if not KEY_CENTRE_CHECK or not stem_notes or not key_text:
        return None
    if not re.search(r"\bmajor\b", str(key_text), re.I):
        return None
    parsed = _parse_key_root_quality(key_text)
    if not parsed or parsed[1] != "major":
        return None
    root = parsed[0]
    fifth = (root + 7) % 12
    flat7, lead = (root + 5) % 12, (root + 6) % 12
    bass, nb = _pc_duration_shares(stem_notes.get("bass"))
    voc, nv = _pc_duration_shares(stem_notes.get("vocals"))
    if nb < KEY_CENTRE_MIN_NOTES or nv < KEY_CENTRE_MIN_NOTES:
        return None
    chord = [0.0] * 12
    for st in ("guitar", "piano", "other"):
        sh, n_ = _pc_duration_shares(stem_notes.get(st))
        if n_:
            chord = [a + b for a, b in zip(chord, sh)]
    tc = sum(chord)
    if tc <= 0:
        return None
    chord = [v / tc for v in chord]
    r = float(KEY_CENTRE_MIN_RATIO)
    for sh in (bass, voc):
        if max(range(12), key=lambda i: sh[i]) != fifth or sh[fifth] < r * sh[root]:
            return None
    if chord[fifth] < chord[root]:
        return None
    if chord[flat7] < float(KEY_CENTRE_FLAT7_MIN_SHARE) or chord[flat7] < float(KEY_CENTRE_FLAT7_RATIO) * chord[lead]:
        return None
    names = _FLAT_NOTE_NAMES if re.match(r"\s*[A-G](?:b|\u266d|\s*flat)", str(key_text), re.I) else NOTE_NAMES
    c, k, f7, ld = names[fifth], names[root], names[flat7], names[lead]
    parent = f"{k} major"
    return {
        "key": f"{c} Mixolydian",
        "display": f"{c} Mixolydian (the notes of {parent}, centred on {c})",
        "note": (
            f"KEY CENTRE CHECK: the key detector named {parent}, whose notes {c} Mixolydian shares; "
            f"it can only choose between major and minor keys. The music centres on {c}: it is the "
            f"most-used note in the low part ({bass[fifth] * 100:.0f}% against {bass[root] * 100:.0f}% "
            f"for {k}) and in the vocal line ({voc[fifth] * 100:.0f}% against {voc[root] * 100:.0f}%), "
            f"the chord parts use it at least as much as {k}, and {f7} natural is played "
            f"({chord[flat7] * 100:.0f}% of the chord parts) where {c} major would have {ld} "
            f"({chord[lead] * 100:.0f}%). So the song is in {c} with a flattened seventh ({f7}): "
            f"{c} Mixolydian. In plain terms say it is in {c}; mention that it uses {f7} rather than "
            f"{ld} (the {k} major scale) when the scale comes up. Scale degrees in the melody blocks "
            f"count from {c}; chord-root degrees in the stem report above count from {k}. "
            f"The detector's own reading follows: "
        ),
    }


def _apply_key_centre_check(revised, final_key, stem_notes):
    """(revised, final_key) with the RECOMMENDED KEY block rewritten when
    _key_centre_reading finds the music centred on the key's fifth; else
    both unchanged."""
    try:
        kc = _key_centre_reading(stem_notes, final_key)
        if not kc:
            return revised, final_key
        pat = re.compile(r"(?m)^RECOMMENDED KEY FOR DISCUSSION: " + re.escape(str(final_key)) + r"\. Reasoning: ")
        if not pat.search(revised):
            return revised, final_key
        revised = pat.sub(lambda _m: "RECOMMENDED KEY FOR DISCUSSION: " + kc["display"] + ". Reasoning: "
                          + kc["note"], revised, count=1)
        return revised, kc["key"]
    except Exception:
        return revised, final_key


# Krumhansl-Kessler major / minor key profiles, tonic first.
_KK_MAJOR_PROFILE = (6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88)
_KK_MINOR_PROFILE = (6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17)
_KEY_NOTE_STEMS = ("bass", "guitar", "piano", "other", "vocals")


def _profile_correlation(a, b):
    ma, mb = sum(a) / 12.0, sum(b) / 12.0
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    da = math.sqrt(sum((x - ma) ** 2 for x in a))
    db = math.sqrt(sum((y - mb) ** 2 for y in b))
    return num / (da * db) if da > 0 and db > 0 else 0.0


def _key_note_fits(stem_notes):
    """({(root pc, 'major'|'minor'): fit}, note count): how well each of the
    24 major/minor keys fits the transcribed notes of the pitched stems
    (duration-weighted pitch classes against the Krumhansl-Kessler profiles,
    Pearson correlation)."""
    hist = [0.0] * 12
    n = 0
    for st in _KEY_NOTE_STEMS:
        sh, k = _pc_duration_shares((stem_notes or {}).get(st))
        if k:
            hist = [a + b for a, b in zip(hist, sh)]
            n += k
    if n <= 0 or sum(hist) <= 0:
        return {}, n
    fits = {}
    for r in range(12):
        for q, prof in (("major", _KK_MAJOR_PROFILE), ("minor", _KK_MINOR_PROFILE)):
            fits[(r, q)] = _profile_correlation(hist, [prof[(i - r) % 12] for i in range(12)])
    return fits, n


_KEY_NOTE_REASON_RE = re.compile(
    r"(?m)^RECOMMENDED KEY FOR DISCUSSION: (?P<final>[^\n]+?)\. Reasoning: "
    r"(?P<lm>MOSS-Music|the listening model|MF) said (?P<x>[^,\n]+?), Essentia said (?P<y>[^(\n]+?) "
    r"\(strength=(?P<s>[\d.]+|n/a), (?P<side>above|below/no) confidence threshold\) — "
    r"[^\n]*?(?:mention the disagreement is possible if asked\.|hedge appropriately\.)")


def _key_note_reading(text, final_key, stem_notes):
    """dev18 KEY NOTE CHECK. When the RECOMMENDED KEY block records a
    listening-model key and an Essentia key that disagree, and the
    transcribed notes fit Essentia's key well (KEY_NOTE_GOOD_FIT, within
    KEY_NOTE_NEAR_BEST of the best of the 24 keys) and the listening model's
    key badly (below KEY_NOTE_POOR_FIT, KEY_NOTE_MARGIN or more behind),
    returns {'span', 'key', 'line'}: the block's text to replace and the key
    to report. Otherwise None, and the block stays as it is."""
    if not KEY_NOTE_CHECK or not stem_notes or not final_key:
        return None
    m = _KEY_NOTE_REASON_RE.search(str(text or ""))
    if not m or m.group("final").strip() != str(final_key).strip():
        return None
    x, y = m.group("x").strip(), m.group("y").strip()
    px, py = _parse_key_root_quality(x), _parse_key_root_quality(y)
    if not px or not py or px == py:
        return None
    if not re.search(r"(?i)\b(?:major|minor)\b", x) or not re.search(r"(?i)\b(?:major|minor)\b", y):
        return None
    fits, n = _key_note_fits(stem_notes)
    if not fits or n < KEY_NOTE_MIN_NOTES:
        return None
    fx, fy, best = fits[px], fits[py], max(fits.values())
    if not (fx < KEY_NOTE_POOR_FIT and fy >= KEY_NOTE_GOOD_FIT and fy - fx >= KEY_NOTE_MARGIN
            and fy >= best - KEY_NOTE_NEAR_BEST):
        return None
    s = m.group("s")
    lm = m.group("lm")
    if lm == "MF":
        lm = "the listening model"
    lm_cap = lm[:1].upper() + lm[1:]
    tail = ("KEY NOTE CHECK: the transcribed notes of every pitched part fit "
            f"{y} (fit {fy:.2f} of 1) and do not fit {x} (fit {fx:.2f}).")
    if m.group("side") == "above":
        line = (f"RECOMMENDED KEY FOR DISCUSSION: {final_key}. Reasoning: Essentia measured {y} "
                f"(strength={s}) and the transcribed notes agree. {lm_cap}'s own answer, {x}, is set "
                f"aside: {tail} Treat {y} as a clear reading; do not offer {x} as an alternative.")
        return {"span": m.span(), "key": final_key, "line": line}
    line = (f"RECOMMENDED KEY FOR DISCUSSION: {y}. Reasoning: {lm} said {x} and Essentia said {y} "
            f"(strength={s}, under its confidence threshold), so {x} was first preferred. {tail} "
            f"So {y} is used, and {x} is set aside. Treat {y} as moderately confident. Chord-root "
            f"degrees in the stem report above count from {x}; the melody blocks count from {y}.")
    return {"span": m.span(), "key": y, "line": line}


_KEY_FIT_LINE_RE = re.compile(r"(?m)^RECOMMENDED KEY FOR DISCUSSION: (?P<final>[^\n]+?)\. Reasoning: [^\n]*")
_SCALE_STEPS = {"major": (0, 2, 4, 5, 7, 9, 11), "minor": (0, 2, 3, 5, 7, 8, 10)}


def _key_fit_reading(text, final_key, stem_notes):
    """KEY_FIT_CHECK (see the setting): {'span', 'key', 'line'} when the
    transcribed notes clearly fit a neighbouring key (one scale note apart)
    whose tonic the bass sits on most; else None."""
    if not KEY_FIT_CHECK or not stem_notes or not final_key:
        return None
    t = str(text or "")
    m = _KEY_FIT_LINE_RE.search(t)
    if not m or m.group("final").strip() != str(final_key).strip():
        return None
    if re.search(r"KEY (?:FIT|CENTRE) CHECK|Mixolydian|Dorian|Lydian|Phrygian", m.group(0)):
        return None
    po = _parse_key_root_quality(final_key)
    if not po or not re.search(r"(?i)\b(?:major|minor)\b", str(final_key)):
        return None
    fits, n = _key_note_fits(stem_notes)
    if not fits or n < KEY_NOTE_MIN_NOTES:
        return None
    best = max(fits, key=lambda k: fits[k])
    fo, fb = fits[po], fits[best]
    if best == po or fb < KEY_NOTE_GOOD_FIT or fb - fo < float(KEY_FIT_MARGIN):
        return None
    so = {(po[0] + i) % 12 for i in _SCALE_STEPS[po[1]]}
    sb = {(best[0] + i) % 12 for i in _SCALE_STEPS[best[1]]}
    if len(so - sb) > 1:
        return None
    bass, nb = _pc_duration_shares(stem_notes.get("bass"))
    if nb < KEY_CENTRE_MIN_NOTES or max(range(12), key=lambda i: bass[i]) != best[0]:
        return None
    names = _FLAT_NOTE_NAMES if re.match(r"\s*[A-G](?:b|\u266d)", str(final_key)) else NOTE_NAMES
    new = f"{names[best[0]]} {best[1]}"
    sm = re.search(r"strength=([\d.]+)", m.group(0))
    det = f"named {final_key}" + (f" (strength={sm.group(1)})" if sm else "")
    line = (f"RECOMMENDED KEY FOR DISCUSSION: {new}. Reasoning: KEY FIT CHECK: the key detector {det}, "
            f"but it matches the notes against major and minor templates only, and {final_key} shares all "
            f"but one of its notes with {new}. The transcribed notes of every pitched part fit {new} better "
            f"(fit {fb:.2f} of 1, against {fo:.2f} for {final_key}), and the bass sits on {names[best[0]]} "
            f"more than on any other note ({bass[best[0]] * 100:.0f}% of its note time). So the song is in "
            f"{new}; treat it as moderately confident and do not present {final_key} as the key. Chord-root "
            f"degrees in the stem report above, and chords marked outside the measured key, count from "
            f"{final_key}. This is the primary key to report to the user unless the reasoning above marks it "
            f"uncertain.")
    return {"span": m.span(), "key": new, "line": line}


def _apply_key_fit_check(revised, final_key, stem_notes):
    """(revised, final_key) with the RECOMMENDED KEY line rewritten when
    _key_fit_reading finds a better-fitting neighbouring key; else unchanged."""
    try:
        kf = _key_fit_reading(revised, final_key, stem_notes)
        if not kf:
            return revised, final_key
        a, b = kf["span"]
        return revised[:a] + kf["line"] + revised[b:], kf["key"]
    except Exception:
        return revised, final_key


def _key_scale_pooled_shares(stem_notes):
    """Per pitch class, the mean of each pitched stem's duration share (every
    stem with 30+ notes counts equally, so one busy part cannot decide)."""
    tot, m = [0.0] * 12, 0
    for st in _KEY_NOTE_STEMS:
        sh, n_ = _pc_duration_shares((stem_notes or {}).get(st))
        if n_ >= 30:
            tot = [a + b for a, b in zip(tot, sh)]
            m += 1
    return ([x / m for x in tot] if m else None), m


def _key_scale_reading(text, final_key, stem_notes):
    """dev26 KEY_SCALE_CHECK (see the setting): {'span', 'key', 'line'} when
    the reported key's own scale is contradicted by the transcribed notes and
    a key a fifth away (one sharp or flat apart) fits them; else None."""
    if not KEY_SCALE_CHECK or not stem_notes or not final_key:
        return None
    t = str(text or "")
    m = _KEY_FIT_LINE_RE.search(t)
    if not m or m.group("final").strip() != str(final_key).strip():
        return None
    if re.search(r"KEY (?:CENTRE|SCALE) CHECK|Mixolydian|Dorian|Lydian|Phrygian|Locrian", m.group(0)):
        return None
    po = _parse_key_root_quality(final_key)
    if not po or not re.search(r"(?i)\b(?:major|minor)\b", str(final_key)):
        return None
    fits, n = _key_note_fits(stem_notes)
    if not fits or n < KEY_NOTE_MIN_NOTES:
        return None
    h, nst = _key_scale_pooled_shares(stem_notes)
    if not h or nst < 2:
        return None
    steps = (0, 2, 4, 5, 7, 9, 11)
    scale = lambda r_: {(r_ + i) % 12 for i in steps}
    s0 = po[0] if po[1] == "major" else (po[0] + 3) % 12       # its key signature (relative major)
    hit = None
    for step in (7, 5):     # one sharp more (the 4th raised) / one flat more (the 7th lowered)
        s1 = (s0 + step) % 12
        x = next(iter(scale(s0) - scale(s1)))     # the reported key's own note
        y = next(iter(scale(s1) - scale(s0)))     # the neighbouring key's note instead
        if h[y] >= float(KEY_SCALE_PRESENT_MIN) and h[x] <= float(KEY_SCALE_ABSENT_RATIO) * h[y]:
            if hit is None or h[x] / max(1e-9, h[y]) < hit[4]:
                hit = (step, s1, x, y, h[x] / max(1e-9, h[y]))
    if hit is None:
        return None
    step, s1, x, y, _r = hit
    pick = max(((s1, "major"), ((s1 + 9) % 12, "minor")), key=lambda c: fits[c])
    s2 = (s1 + step) % 12
    p2 = max(((s2, "major"), ((s2 + 9) % 12, "minor")), key=lambda c: fits[c])
    if fits[p2] >= fits[pick] + float(KEY_SCALE_NEXT_MARGIN):
        pick = p2
    if pick == po or fits[pick] < float(KEY_NOTE_GOOD_FIT):
        return None
    # The old tonic still rules the bass and the chord parts: the song keeps
    # its centre and borrows the note (a Mixolydian / Dorian colour), so the
    # reported key stays.
    bass, nb = _pc_duration_shares(stem_notes.get("bass"))
    chord = [0.0] * 12
    for st in ("guitar", "piano", "other"):
        sh, n_ = _pc_duration_shares(stem_notes.get(st))
        if n_:
            chord = [a + b for a, b in zip(chord, sh)]
    g = float(KEY_SCALE_GUARD_RATIO)
    if nb and bass[po[0]] >= g * bass[pick[0]] and chord[po[0]] >= g * chord[pick[0]]:
        return None
    _flat_sig = (pick[0] if pick[1] == "major" else (pick[0] + 3) % 12) in (5, 10, 3, 8, 1)
    names = (_FLAT_NOTE_NAMES if (re.match(r"\s*[A-G](?:b|\u266d)", str(final_key)) or _flat_sig)
             else NOTE_NAMES)
    new = f"{names[pick[0]]} {pick[1]}"
    sm = re.search(r"strength=([\d.]+)", m.group(0))
    det = f"named {final_key}" + (f" (strength={sm.group(1)})" if sm else "")
    line = (f"RECOMMENDED KEY FOR DISCUSSION: {new}. Reasoning: KEY SCALE CHECK: the key detector {det}, "
            f"but {final_key}'s own {names[x]} is all but absent from the transcribed notes of the pitched parts "
            f"({h[x] * 100:.1f}% of their note time) while {names[y]}, which {final_key} does not have, is clearly "
            f"played ({h[y] * 100:.1f}%). That is the scale of a key a fifth away, and of the keys with that scale "
            f"the notes fit {new} best (fit {fits[pick]:.2f} of 1; {final_key}'s {fits[po]:.2f} comes from the notes the "
            f"two keys share). So the song is in "
            f"{new}; treat it as moderately confident and do not present {final_key} as the key. Chord-root "
            f"degrees in the stem report above, and chords marked outside the measured key, count from "
            f"{final_key}. This is the primary key to report to the user unless the reasoning above marks it "
            f"uncertain.")
    return {"span": m.span(), "key": new, "line": line}


def _key_change_reading(stem_notes, duration=None):
    """dev28 KEY_CHANGE_CHECK: (key before, key after, time, semitones) when
    the transcribed notes show the song moving to a new key for its last
    part -- the notes before some point fit one key (KEY_CHANGE_MIN_FIT or
    more) and the notes after it fit that same key moved by 1 to 4 semitones
    up or 1 to 2 down (as well), each side holding KEY_NOTE_MIN_NOTES/2 notes
    or more, the change in the second half of the song and at least
    KEY_CHANGE_MIN_TAIL_S before its end, and the notes of the whole song
    fitting no single key as well. Else None."""
    if not stem_notes:
        return None
    spans = []
    for st in _KEY_NOTE_STEMS:
        for n_ in (stem_notes or {}).get(st) or []:
            sp = _sx_note_span(n_) if isinstance(n_, dict) else None
            if sp:
                spans.append((float(sp[0]), st, n_))
    if len(spans) < KEY_NOTE_MIN_NOTES:
        return None
    end = float(duration or max(t for t, _s, _n in spans))
    whole, _n = _key_note_fits(stem_notes)
    if not whole:
        return None
    w_best = max(whole.values())

    def _fits(t0, t1):
        sub = {}
        for t, st, n_ in spans:
            if t0 <= t < t1:
                sub.setdefault(st, []).append(n_)
        return _key_note_fits(sub)
    best = None
    t = 0.5 * end
    while t <= end - float(KEY_CHANGE_MIN_TAIL_S):
        fa, na = _fits(0.0, t)
        fb, nb_ = _fits(t, end + 1.0)
        if fa and fb and na >= KEY_NOTE_MIN_NOTES // 2 and nb_ >= KEY_NOTE_MIN_NOTES // 2:
            ka = max(fa, key=lambda k: fa[k])
            kb = max(fb, key=lambda k: fb[k])
            shift = (kb[0] - ka[0]) % 12
            if kb[1] == ka[1] and shift in (1, 2, 3, 4, 10, 11):
                sc = min(fa[ka], fb[kb])
                if best is None or sc > best[0]:
                    best = (sc, ka, kb, t, shift if shift <= 4 else shift - 12)
        t += 2.0
    if best is None:
        return None
    sc, ka, kb, t, k = best
    if sc < float(KEY_CHANGE_MIN_FIT) or sc - w_best < float(KEY_CHANGE_MIN_GAIN):
        return None
    return ka, kb, t, k


def _apply_key_change_check(revised, final_key, stem_notes):
    """dev28 KEY_CHANGE_CHECK: (revised, final_key) with the RECOMMENDED KEY
    line rewritten to the key the song starts in, naming the key change,
    when _key_change_reading finds one; else unchanged."""
    try:
        if not KEY_CHANGE_CHECK or not final_key:
            return revised, final_key
        m = _KEY_FIT_LINE_RE.search(str(revised or ""))
        if not m or m.group("final").strip() != str(final_key).strip() or "KEY CHANGE CHECK" in m.group(0):
            return revised, final_key
        kc = _key_change_reading(stem_notes)
        if not kc:
            return revised, final_key
        ka, kb, t, k = kc

        def _nm(key):
            sig = key[0] if key[1] == "major" else (key[0] + 3) % 12
            names = _FLAT_NOTE_NAMES if sig in (5, 10, 3, 8, 1) else NOTE_NAMES
            return f"{names[key[0]]} {key[1]}"
        a_, b_ = _nm(ka), _nm(kb)
        way = "up" if k > 0 else "down"
        n_st = abs(int(k))
        step = {1: "a semitone", 2: "a whole tone", 3: "three semitones", 4: "four semitones"}.get(n_st, f"{n_st} semitones")
        mm = f"{int(t // 60)}:{int(t % 60):02d}"
        prev = f"named {final_key}" if final_key != a_ else f"also named {final_key}"
        line = (f"RECOMMENDED KEY FOR DISCUSSION: {a_}. Reasoning: KEY CHANGE CHECK: the song changes key -- "
                f"it is in {a_} and moves {way} {step} to {b_} at about {mm} for its last part (the transcribed "
                f"notes before that point fit {a_}, those after it fit {b_}, each better than any single key fits "
                f"the whole song). The key detector, which measures the whole song at once, {prev}. Report {a_} as "
                f"the song's key and say that it modulates {way} to {b_} near the end; treat it as moderately "
                f"confident. Chord-root degrees in the stem report above count from {final_key}. This is the "
                f"primary key to report to the user unless the reasoning above marks it uncertain.")
        a0, b0 = m.span()
        return revised[:a0] + line + revised[b0:], a_
    except Exception:
        return revised, final_key


def _apply_key_scale_check(revised, final_key, stem_notes):
    """(revised, final_key) with the RECOMMENDED KEY line rewritten when
    _key_scale_reading finds the reported key's scale contradicted; else
    unchanged."""
    try:
        ks = _key_scale_reading(revised, final_key, stem_notes)
        if not ks:
            return revised, final_key
        a, b = ks["span"]
        return revised[:a] + ks["line"] + revised[b:], ks["key"]
    except Exception:
        return revised, final_key


def _apply_key_note_check(revised, final_key, stem_notes):
    """(revised, final_key) with the RECOMMENDED KEY block rewritten when
    _key_note_reading sets the listening model's key aside; else both
    unchanged."""
    try:
        kn = _key_note_reading(revised, final_key, stem_notes)
        if not kn:
            return revised, final_key
        a, b = kn["span"]
        return revised[:a] + kn["line"] + revised[b:], kn["key"]
    except Exception:
        return revised, final_key


def _detect_modal_inflection(notes, key_text):
    """Report a diatonic mode when the transcribed notes clearly show one.

    Weights each pitch class by note duration (a long natural 6th is stronger
    evidence than a passing one), then compares the single degree that
    distinguishes the mode from its parent major/minor. Returns "" unless the
    evidence is unambiguous, so this can only ever ADD a line.
    """
    try:
        parsed = _parse_key_root_quality(key_text)
        if not parsed or not notes:
            return ""
        root_pc, quality = parsed

        weights = {}
        total = 0.0
        counted = 0
        for n in notes:
            try:
                pc = _hz_to_pitch_class(n.get("frequency"))
            except Exception:
                pc = None
            if pc is None:
                continue
            try:
                w = float(n.get("offset", 0.0)) - float(n.get("onset", 0.0))
            except Exception:
                w = 0.0
            if not (w > 0):
                w = 1.0  # durations missing/degenerate: fall back to counting
            deg = (pc - root_pc) % 12
            weights[deg] = weights.get(deg, 0.0) + w
            total += w
            counted += 1

        if counted < MODAL_DETECTION_MIN_NOTES or total <= 0:
            return ""

        findings = []
        for mode, parent, char_deg, rival_deg, label in _MODAL_INFLECTION_TESTS.get(quality, []):
            c = weights.get(char_deg, 0.0)
            o = weights.get(rival_deg, 0.0)
            if (c + o) / total < MODAL_DETECTION_MIN_SHARE:
                continue
            if c <= o * MODAL_DETECTION_MIN_RATIO:
                continue
            findings.append((c, mode, parent, label, o))

        if not findings:
            return ""

        findings.sort(key=lambda f: -f[0])
        c, mode, parent, label, o = findings[0]
        root_name = NOTE_NAMES[root_pc % 12]
        out = (
            f"modal reading: the notes fit {root_name} {mode} rather than plain "
            f"{root_name} {parent} — the {label} carries {c / total * 100:.0f}% "
            "of the note weight"
        )
        if o <= 0:
            out += ", and the contrasting degree never appears"
        else:
            out += f" against {o / total * 100:.0f}% for the {parent} form"
        return out
    except Exception:
        return ""


def _parse_key_root_quality(key_text):
    """Parse a 'C major' / 'A minor' / 'F# Major' style key string into
    (root_pitch_class, quality). Returns None if it can't be parsed --
    callers must treat that as "no key available", not fall back to a guess."""
    if not key_text:
        return None
    m = re.match(
        r"\s*([A-Ga-g])[\s\-]*(#|b(?=$|[^a-z]|m(?:$|[^a-z]|in|aj))|sharp|flat|\u266f|\u266d)?\s*(major|minor|maj|min|m(?![a-z]))?(?![a-z])",
        str(key_text),
        re.IGNORECASE,
    )
    if not m or not m.group(1):
        return None
    base = _NOTE_LETTER_PC.get(m.group(1).upper())
    if base is None:
        return None
    accidental = (m.group(2) or "").lower()
    if accidental in ("#", "sharp", "\u266f"):
        base += 1
    elif accidental in ("b", "flat", "\u266d"):
        base -= 1
    quality_raw = (m.group(3) or "major").lower()
    quality = "minor" if (quality_raw.startswith("min") or quality_raw == "m") else "major"
    if not m.group(3):
        # A modal key ("D Dorian", "A Aeolian") used to fall through to major,
        # so scale degrees were labelled against a major scale for modes
        # whose third is minor. Map each mode onto its major/minor family.
        _rest = str(key_text)[m.end():].lower()
        if re.search(r"\b(dorian|phrygian|aeolian|locrian)\b", _rest):
            quality = "minor"
    return (base % 12, quality)


def _scale_degree_label(pitch_class, root_pc, quality):
    interval = (pitch_class - root_pc) % 12
    table = _MINOR_SCALE_DEGREES if quality == "minor" else _MAJOR_SCALE_DEGREES
    return table.get(interval, f"chromatic ({interval} semitones from tonic)")


def _scale_degree_summary(notes, key_text):
    """Which scale degrees (relative to the track's own recommended key)
    this note set actually emphasizes, e.g. 'centers on 1 and 5 (tonic and
    dominant)'. Grounded entirely in the transcribed pitch classes -- this
    is descriptive statistics on real notes, not inferred harmony."""
    parsed = _parse_key_root_quality(key_text)
    if not parsed or not notes:
        return ""
    root_pc, quality = parsed
    counts = {}
    for n in notes:
        pc = _hz_to_pitch_class(n["frequency"])
        if pc is None:
            continue
        deg = _scale_degree_label(pc, root_pc, quality)
        counts[deg] = counts.get(deg, 0) + 1
    total = sum(counts.values())
    if total < 4:
        return ""
    top = sorted(counts.items(), key=lambda kv: -kv[1])[:3]
    chromatic_ct = sum(c for d, c in counts.items() if d.startswith("chromatic"))
    desc = "emphasizes scale degree" + ("s " if len(top) > 1 else " ") + ", ".join(
        f"{d}" for d, _c in top
    )
    if chromatic_ct / total > 0.3:
        desc += "; notable chromatic content outside the estimated key"
    return desc


def _melodic_interval_profile(notes):
    """Characterize consecutive-note movement (step vs leap, direction bias)
    from the ACTUAL transcribed interval sequence -- computed once here so
    the writer never has to eyeball a note-name string and guess."""
    if not notes or len(notes) < 3:
        return ""
    ordered = sorted(notes, key=lambda n: n["onset"])
    steps = leaps = same = up = down = 0
    leap_max = 0
    for a, b in zip(ordered, ordered[1:]):
        fa, fb = float(a.get("frequency") or 0), float(b.get("frequency") or 0)
        if fa <= 0 or fb <= 0:
            continue
        semis = int(round(12 * np.log2(fb / fa)))
        absv = abs(semis)
        if absv == 0:
            same += 1
        elif absv <= 2:
            steps += 1
        else:
            leaps += 1
            leap_max = max(leap_max, absv)
        if semis > 0:
            up += 1
        elif semis < 0:
            down += 1
    total = steps + leaps + same
    if total < 3:
        return ""
    step_pct = steps / total * 100
    leap_pct = leaps / total * 100
    if step_pct >= 65:
        motion = "mostly stepwise motion"
    elif leap_pct >= 50:
        motion = "leap-heavy, wide-interval motion"
    else:
        motion = "a mix of stepwise motion and leaps"
    desc = f"melodic motion: {motion} ({step_pct:.0f}% steps, {leap_pct:.0f}% leaps"
    if leap_max >= 5:
        desc += f", widest leap ~{leap_max} semitones"
    desc += ")"
    if up + down >= 4:
        if up > down * 1.5:
            desc += "; more upward movement than downward"
        elif down > up * 1.5:
            desc += "; more downward movement than upward"
    return desc


def _summarize_melody(notes, max_notes=None, key_text=None, bpm=None,
                      include_motifs=True):
    """Describe a melodic line from the real transcribed notes.

    Reports, all computed arithmetically from the note events themselves:
    the opening figure as heard (contiguous, not a sample scattered across the
    line), the arc of the line, practical and absolute pitch range, step/leap
    motion, scale-degree emphasis against this track's own measured key,
    rhythmic density, and any repeated motif.

    The note sequence shown is now the START of the line rather than notes
    sampled evenly across it. An evenly-sampled sequence is not a thing anyone
    ever heard -- printing every fifth note of a phrase produces a series of
    pitches in an order that never occurred, which is exactly the kind of
    detail a language model then repeats back as if it were the tune.
    """
    if not notes:
        return ""

    if max_notes is None:
        max_notes = STEM_MIDI_MELODY_LINE_NOTES

    seq_notes = _notes_sorted(notes)
    freqs = np.array(
        [float(n["frequency"]) for n in seq_notes if float(n.get("frequency", 0)) > 0],
        dtype=float,
    )
    if freqs.size == 0:
        return ""

    # Contiguous opening figure -- an order that was actually played.
    head = seq_notes[:max_notes]
    seq = [_hz_to_note_name(float(n["frequency"])) for n in head]

    direction = _contour_shape(seq_notes)

    lo_abs = _hz_to_note_name(float(np.min(freqs)))
    hi_abs = _hz_to_note_name(float(np.max(freqs)))
    # Practical range trims transcription outliers at both ends.
    if freqs.size >= 8:
        lo_p = _hz_to_note_name(float(np.percentile(freqs, 5)))
        hi_p = _hz_to_note_name(float(np.percentile(freqs, 95)))
        range_note = "%s-%s (practical), %s-%s (absolute)" % (lo_p, hi_p, lo_abs, hi_abs)
    else:
        range_note = "%s-%s" % (lo_abs, hi_abs)

    if len(seq_notes) > len(seq):
        lead = "melodic line (%d notes total; opens %s)" % (len(seq_notes), " ".join(seq))
    else:
        lead = "melodic line (%d notes): %s" % (len(seq), " ".join(seq))

    out = "%s; contour: %s; pitch range: %s" % (lead, direction, range_note)

    interval_desc = _melodic_interval_profile(seq_notes)
    if interval_desc:
        out += "; " + interval_desc
    degree_desc = _scale_degree_summary(seq_notes, key_text)
    if degree_desc:
        out += "; " + degree_desc
    modal_desc = _detect_modal_inflection(seq_notes, key_text)
    if modal_desc:
        out += "; " + modal_desc
    rhythm_desc = _rhythmic_profile(seq_notes, bpm=bpm)
    if rhythm_desc:
        out += "; " + rhythm_desc

    if include_motifs:
        motifs = _extract_motifs(seq_notes)
        if motifs:
            phrase = _motif_phrase(motifs[0])
            if phrase:
                out += "; repeated figure: " + phrase
    return out


_HM_HEAD = "HARMONY AND MELODY BY SECTION"
_HM_ACC_STEMS = ("guitar", "piano", "other")


def _hm_midi(n):
    try:
        f = float(n.get("frequency") or 0.0)
        return 69.0 + 12.0 * math.log2(f / 440.0) if f > 0 else None
    except Exception:
        return None


def _hm_pitched(notes):
    """Notes with a pitch and no drum type (drum hits never count)."""
    out = []
    for n in notes or []:
        if n.get("drum_type"):
            continue
        m = _hm_midi(n)
        if m is None:
            continue
        try:
            on = float(n.get("onset"))
        except Exception:
            continue
        d = n.get("duration")
        if d is None:
            try:
                d = float(n.get("offset", on)) - on
            except Exception:
                d = 0.0
        out.append((on, max(0.05, float(d or 0.0)), m))
    return out


def _hm_avg_poly(notes):
    """Average number of notes sounding at each note onset (1 = a single line)."""
    if not notes:
        return 0.0
    tot = 0
    for on, _d, _m in notes:
        tot += sum(1 for o2, d2, _ in notes if o2 <= on < o2 + d2)
    return tot / float(len(notes))


def _hm_section_chords(stem_notes, a, b, key):
    counts = {}
    for stem in _HM_ACC_STEMS:
        sec = [n for n in (stem_notes.get(stem) or []) if not n.get("drum_type")
               and a <= float(n.get("onset", -1) or -1) < b]
        if len(sec) < 4:
            continue
        try:
            found = _detect_common_chords(sec, max_count=8)
        except Exception:
            continue
        for label, cnt in found:
            mm = re.match(r"([A-G]#?) (major|minor)$", label)
            if mm:
                k = (_CHD_PC[mm.group(1)], "m" if mm.group(2) == "minor" else "")
                counts[k] = counts.get(k, 0) + cnt
                continue
            mm = re.match(r"([A-G]#?)\+([A-G]#?)$", label)
            if mm:
                x, y = _CHD_PC[mm.group(1)], _CHD_PC[mm.group(2)]
                root = x if (y - x) % 12 == 7 else (y if (x - y) % 12 == 7 else None)
                if root is not None:
                    counts[(root, "5")] = counts.get((root, "5"), 0) + cnt
    triad_roots = {r for (r, q) in counts if q in ("", "m")}
    items = sorted(((c, k) for k, c in counts.items() if not (k[1] == "5" and k[0] in triad_roots)),
                   key=lambda x: -x[0])
    if not items:
        return []
    top = items[0][0]
    names = _CHD_FLAT if (key and key[3]) else _CHD_SHARP
    return [names[k[0]] + k[1] for c, k in items if c >= max(2, 0.25 * top)][:4]


def _hm_vocal_absent(text):
    """Section time ranges ("m:ss-m:ss") where INTENSITY BY SECTION (vocals
    stem) says the voice is not playing or quiet / mostly absent."""
    t = str(text or "")
    i = t.find("INTENSITY BY SECTION (vocals stem")
    if i < 0:
        return set()
    e = t.find("\n\n", i)
    out = set()
    for m in re.finditer(r"(?m)^\s+.+? \((\d+:\d\d)-(\d+:\d\d)\): (not playing|quiet)", t[i:e if e >= 0 else len(t)]):
        out.add(f"{m.group(1)}-{m.group(2)}")
    return out


def _hm_lead(pitched, a, b, instrumental, vocal_absent=None):
    """(stem, notes) carrying the lead line in [a, b): the voice when it has
    8+ notes there, else the most melody-like single-note part among guitar,
    piano and other (8+ notes, near-single-line, highest), else (None, [])."""
    if not instrumental and f"{_seconds_to_mmss(a)}-{_seconds_to_mmss(b)}" not in (vocal_absent or set()):
        v = [x for x in pitched.get("vocals", []) if a <= x[0] < b]
        if len(v) >= 8:
            return "vocals", v
    best = None
    for stem in _HM_ACC_STEMS:
        sec = [x for x in pitched.get(stem, []) if a <= x[0] < b]
        if len(sec) < 8:
            continue
        if _hm_avg_poly(sec) > 1.6:
            continue
        med = float(np.median([x[2] for x in sec]))
        if best is None or med > best[0]:
            best = (med, stem, sec)
    return (best[1], best[2]) if best else (None, [])


def _hm_name(m):
    try:
        return _hz_to_note_name(440.0 * (2.0 ** ((float(m) - 69.0) / 12.0)))
    except Exception:
        return "?"


def _hm_octaves_above_bass(d):
    """A lead-to-bass gap in semitones, in words (nearest half octave)."""
    try:
        d = float(d)
    except Exception:
        return ""
    if d < 0:
        return "below the bass"
    if d < 9:
        return "less than an octave above the bass"
    halves = int(round(d / 6.0))
    whole, half = divmod(halves, 2)
    if whole == 0:
        return "about half an octave above the bass"
    words = {1: "an octave", 2: "two octaves", 3: "three octaves", 4: "four octaves"}
    if half:
        n = {1: "one", 2: "two", 3: "three", 4: "four"}.get(whole, str(whole))
        return f"about {n} and a half octaves above the bass"
    return f"about {words.get(whole, str(whole) + ' octaves')} above the bass"


def _build_harmony_melody_by_section(sections, stem_notes, key=None, instrumental=False, context_text=""):
    """See HARMONY_MELODY_BY_SECTION. "" when there is nothing to say."""
    if not HARMONY_MELODY_BY_SECTION or not sections or not stem_notes:
        return ""
    try:
        kk = _chd_key(f"RECOMMENDED KEY FOR DISCUSSION: {key}." if key else "")
        pitched = {s_: _hm_pitched(stem_notes.get(s_)) for s_ in ("vocals", "bass") + _HM_ACC_STEMS}
        vocal_absent = _hm_vocal_absent(context_text)
        lines = []
        for sec in sections:
            try:
                a, b = float(sec["start"]), float(sec["end"])
            except Exception:
                continue
            name = str(sec.get("name") or "").strip() or "Section"
            if b - a < 3.0 or name.lower().startswith("silence"):
                continue
            chords = _hm_section_chords(stem_notes, a, b, kk)
            parts = ["chords " + (", ".join(chords) + _chd_outside_note(chords, kk) if chords else "none clear")]
            lead_stem, lead = _hm_lead(pitched, a, b, instrumental, vocal_absent)
            if not lead:
                parts.append("no clear lead line")
            else:
                lm = [x[2] for x in lead]
                lo, hi = np.percentile(lm, 10), np.percentile(lm, 90)
                med = float(np.median(lm))
                who = "the voice" if lead_stem == "vocals" else f"the {lead_stem} stem"
                parts.append(f"lead line: {who}, {_hm_name(lo)}-{_hm_name(hi)}")
                acc = [x for st in _HM_ACC_STEMS if st != lead_stem for x in pitched.get(st, []) if a <= x[0] < b]
                if len(acc) >= 8:
                    d = med - float(np.median([x[2] for x in acc]))
                    rel = ("well above the chords" if d >= 7 else "just above the chords" if d >= 2
                           else "inside the chords' range" if d > -2 else "below the chords")
                    parts.append(rel if HM_PLAIN_FIGURES else f"{rel} ({d:+.0f} semitones)")
                bass = [x for x in pitched.get("bass", []) if a <= x[0] < b]
                if len(bass) >= 8 and lead_stem != "bass":
                    _db = med - float(np.median([x[2] for x in bass]))
                    parts.append(_hm_octaves_above_bass(_db) if HM_PLAIN_FIGURES
                                 else f"{_db:.0f} semitones above the bass")
                support = [x for st in ("bass",) + _HM_ACC_STEMS if st != lead_stem for x in pitched.get(st, [])
                           if a <= x[0] < b]
                hits = tot = 0
                for on, dur, m in lead:
                    pcs = {int(round(y[2])) % 12 for y in support if y[0] <= on + 0.05 < y[0] + y[1]}
                    if not pcs:
                        continue
                    tot += 1
                    hits += int(int(round(m)) % 12 in pcs)
                if tot >= 8:
                    share = hits / float(tot)
                    desc = ("mostly on notes the accompaniment is playing (sits inside the harmony)" if share >= 0.6
                            else "a mix of notes the accompaniment is playing and notes it is not "
                                 "(passing or colour notes)" if share >= 0.35
                            else "mostly on notes the accompaniment is not playing at that moment "
                                 "(moves independently of the chords, or adds tension)")
                    parts.append(desc if HM_PLAIN_FIGURES
                                 else f"{share * 100:.0f}% of its notes match the accompaniment -- {desc}")
            lines.append(f"  {name} ({_seconds_to_mmss(a)}-{_seconds_to_mmss(b)}): " + "; ".join(parts))
        if not lines:
            return ""
        return (f"{_HM_HEAD} (measured from this recording's transcribed notes; drums are never counted; chords "
                "are the ones heard most in each section, ranked, not an exact progression; the lead line is the "
                "voice where it sings, otherwise the most melody-like single-note part -- never the bass; "
                "transcription is probabilistic, so treat these as tendencies):\n" + "\n".join(lines))
    except Exception:
        return ""


def build_melody_understanding_block(sections, stem_notes, bpm=None, key=None):
    """Top-level melodic evidence block assembled purely from raw stem MIDI.

    Header states the provenance explicitly so the writing model treats it as
    measurement of this recording rather than as something it could substitute
    background knowledge for.
    """
    if not stem_notes:
        return ""
    body = _build_melody_per_section(sections or [], stem_notes, bpm=bpm, key=key)
    if not body:
        return ""
    header = (
        "=== MELODIC ANALYSIS (measured from this recording's raw stem MIDI) ===\n"
        "Every pitch, interval, contour, motif and range below was computed "
        "arithmetically from the transcribed note events of this specific "
        "recording. It is not derived from the reference database, file tags, "
        "or any prior knowledge of the song, and must not be replaced or "
        "'corrected' by those sources. Transcription is probabilistic, so treat "
        "isolated short notes with caution -- but the contour, range, motion and "
        "recurring-figure statements are direct measurements.\n"
    )
    return header + body
