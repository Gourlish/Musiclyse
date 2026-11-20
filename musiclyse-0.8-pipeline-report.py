# MUSICLYSE FILE pipeline-report 0.8
# ==============================================================================
#  MUSICLYSE 0.8: THE WRITTEN REPORT
# ==============================================================================
# This file holds the written report: tidying and checking the description, filling gaps from
# measurements, and the objective audio report. It is part of musiclyse-0.8.py, which loads it
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


_STATED_DURATION_RE = re.compile(
    r"\b((?:total\s+)?(?:duration|length|running time|runtime)\s+of\s+the\s+"
    r"(?:piece|track|song|recording|composition)\s+is\s+(?:approximately\s+|about\s+|roughly\s+|around\s+)?)"
    r"(\d+(?:\.\d+)?)\s*(seconds|secs?|s)\b"
    r"(?!\s+(?:longer|shorter|more|less|over|under|before|after|into|in\b))",
    re.IGNORECASE,
)


def _correct_stated_duration(text, total_duration, tol_s=3.0):
    """Replace a wrong track length stated in the listening model's prose.

    The listening model sometimes opens with "The duration of the piece is N seconds" and gets
    N wrong (an AI-generated test track: "266.00 seconds" for a 256.0 s file). The writer quotes
    such sentences, so the measured length is substituted and the change is
    marked. Only the "duration/length of the track is N seconds" phrasing is
    touched, and only when it is off by more than `tol_s`.
    """
    try:
        dur = float(total_duration)
    except (TypeError, ValueError):
        return text
    if not text or dur <= 0:
        return text

    def _fix(m):
        try:
            stated = float(m.group(2))
        except ValueError:
            return m.group(0)
        if abs(stated - dur) <= float(tol_s):
            return m.group(0)
        if not (0.5 * dur <= stated <= 1.5 * dur):
            # Too far off to be a misreading of the track length -- some
            # other quantity; leave the sentence alone.
            return m.group(0)
        return (
            f"{m.group(1)}{dur:.1f} {m.group(3)} [measured from the file; the listening "
            f"model's own figure of {m.group(2)} was wrong]"
        )

    try:
        return _STATED_DURATION_RE.sub(_fix, text)
    except Exception:
        return text


def _resolve_total_duration(track_path=None, dsp_path=None, objective_report=None):
    """Best-effort track duration in seconds, probed from whichever source
    is actually available.

    This used to come ONLY from a regex match on the objective audio
    report's "duration: X s" line. That meant total_duration silently
    stayed None whenever that report was empty -- ENABLE_OBJECTIVE_AUDIO_REPORT
    off, dsp_path conversion having failed for this file, or any exception
    inside build_objective_audio_report before that line was written -- and
    losing total_duration took down the ENTIRE structure-block pipeline with
    it, including the last-resort "coarse evenly-spaced windows" fallback in
    format_recommended_structure_block that specifically exists to guarantee
    the writer always has SOMETHING timestamped to work with. That single
    point of failure is why the writer could end up saying it has no section
    map at all for a track. Duration is now probed independently through
    several cheap, robust sources so no single failure can take it out.
    """
    # 1. Already-computed objective report text, if any (cheapest, no I/O).
    if objective_report:
        m = re.search(r"duration:\s*([0-9.]+)\s*s", objective_report)
        if m:
            try:
                val = float(m.group(1))
                if val > 0:
                    return val
            except Exception:
                pass

    candidates = [p for p in (dsp_path, track_path) if p and isinstance(p, str)]

    # 2. File tags (mutagen) -- near-instant, no audio decode.
    if METADATA_AVAILABLE:
        for p in candidates:
            if p.startswith(("http://", "https://")) or not os.path.exists(p):
                continue
            try:
                mf = MutagenFile(p)
                length = getattr(getattr(mf, "info", None), "length", None)
                if length and float(length) > 0:
                    return float(length)
            except Exception:
                continue

    # 3. librosa's duration probe -- uses the file header/stream length via
    # soundfile where possible, so it stays fast even on long files.
    for p in candidates:
        if p.startswith(("http://", "https://")) or not os.path.exists(p):
            continue
        try:
            dur = float(librosa.get_duration(path=p))
            if dur > 0:
                return dur
        except TypeError:
            try:
                dur = float(librosa.get_duration(filename=p))
                if dur > 0:
                    return dur
            except Exception:
                continue
        except Exception:
            continue

    return None


def _report_line_repairs(text):
    """CHORDS and INSTRUMENTATION in a saved scan's report, in the current
    format (load time; idempotent)."""
    t = str(text or "")
    try:
        if CHORDS_MEASURED_LINE:
            m = re.search(r"(?m)^(CHORDS=)([^\n]*)$", t)
            if m:
                nv = _chords_line_value(m.group(2), t)
                if nv != m.group(2):
                    t = t[:m.start()] + m.group(1) + nv + t[m.end():]
        if INSTRUMENTATION_COLLAPSE_SPANS:
            m = re.search(r"(?m)^(INSTRUMENTATION=)([^\n]*)$", t)
            if m:
                nv = _instrumentation_collapse_spans(m.group(2))
                if nv != m.group(2):
                    t = t[:m.start()] + m.group(1) + nv + t[m.end():]
    except Exception:
        return str(text or "")
    return t


# dev66: fill the report fields the listening model left out (see REPORT_FILL_FROM_EVIDENCE).
_FILL_TOPICS = (
    # field, words (a sentence's score is how many of these it contains)
    ("LYRIC_SUBJECT", r"lyric|lyrics|lyrical|lyrically|theme|themes|thematic|subject|storytelling|narrat|chorus chants|hook “|refrain"),
    ("LEAD_VOCAL_CHARACTERISTICS", r"vocal|vocals|vocalist|singer|voice|tenor|baritone|alto|soprano|falsetto|sung|singing|croon|rasp"),
    ("DRUMS", r"drum|drums|kick|snare|hi-hat|hi‑hat|hihat|hats|cymbal|percussion|backbeat|4/4|groove|fills"),
    ("CHORDS", r"chord|chords|progression|harmonic|harmony|voicings|cadence"),
    ("INSTRUMENTATION", r"guitar|guitars|bass|synth|synths|synthesizer|keyboard|keyboards|piano|organ|saxophone|sax|horn|horns|brass|strings|violin|cello|pad|pads|instrument|instrumentation|arrangement|accordion|harmonica|trumpet|flute"),
    ("TIMBRE", r"production|produced|mix|mixed|stereo|reverb|delay|compression|compressed|eq|polished|fidelity|sheen|texture|textures|timbre|tone|tones|bright|warm|crisp|gritty|lo-fi|clean"),
    ("MELODICISM", r"melody|melodies|melodic|hook|hooks|catchy|riff|riffs|motif"),
    ("MELODIC_CONTOUR", r"contour|rises|rising|falls|falling|ascend|ascending|descend|descending|arch|arching|stepwise|leaps"),
    ("MOOD_VIBE", r"mood|atmosphere|vibe|feel|feeling|exudes|energy|energetic|melanchol|upbeat|wistful|emotional|uplifting|dark|brooding|joyful|rebellious|introspective|reflective|bittersweet|sorrow|sorrowful|sad|sadness|longing|nostalgi|hopeful|tense|tension-filled|playful|dreamy|calm|somber|sombre|yearning|defiant|euphoric|aggressive|mournful|tender"),
    # a sentence about the song's sections is not used (STRUCTURE is the measured map)
    ("_SECTIONS", r"intro|verse|verses|chorus|choruses|pre-chorus|pre\u2011chorus|bridge|outro|structur|section|sections|solo|breakdown|fade-out|fade\u2011out"),
)
_FILL_HEADINGS = (
    ("_SECTIONS", r"structure|section|\bform\b"),
    ("LYRIC_SUBJECT", r"lyric"), ("LEAD_VOCAL_CHARACTERISTICS", r"vocal"), ("DRUMS", r"drum|rhythm section"),
    ("INSTRUMENTATION", r"instrument|arrangement"), ("TIMBRE", r"production|sound"), ("MOOD_VIBE", r"mood|aesthetic|vibe"),
    ("MELODICISM", r"melod"), ("CHORDS", r"harmon|chord"),
    ("_SECTIONS", r"intro|verse|chorus|bridge|outro|solo|structure|section|build"),
)
# Sentences that date the track, name a genre, or give a key or tempo are left
# out: those have their own measured blocks and checks.
_FILL_SKIP_RE = re.compile(
    r"\b(?:1[89]|20)\d0['’]?s\b|\b[2-9]0['’]?s\b|\bmodern\b|\bcontemporary\b|\bvintage\b|\bretro\b|"
    r"\bera\b|\bdecade|\bgenre|\bsub-?genre|\bBPM\b|\btempo\b|\b[A-G][#♯♭b]?\s+(?:major|minor)\b|"
    r"\bduration of the piece\b|\bSTRUCTURE=|\b\d{1,2}:\d{2}\b|"
    r"^(?:The|This) (?:track|song|piece|recording|tune|number) is an? ", re.I)
# "Indie Pop / Synth-Pop", "Classic Rock / Hard Rock": genre labels (case matters here,
# so "baritone/tenor" is not one)
_FILL_GENRE_PAIR_RE = re.compile(r"\b[A-Z][\w\u2011-]+ ?/ ?[A-Z][\w\u2011-]+")
# ... and statements of absence ("No synths ... are evident"): not a description.
_FILL_NEGATIVE_RE = re.compile(
    r"^\s*(?:No|Not|Nor|Without|There (?:is|are) no)\b|\bno (?:additional|other|obvious|audible)\b|"
    r"\b(?:is|are) (?:not|absent)\b|\bnot (?:evident|present|audible)\b", re.I)


_FILL_TRAIL_SPLIT_RE = re.compile(
    r",\s+(?=(?:channeling|channelling|while|blending|recalling|evoking|reminiscent|typical|"
    r"situating|placing|harking|harkening|in the (?:style|vein|tradition))\b)", re.I)


def _fill_clean(sent):
    """The sentence, or its opening clause when only a trailing clause dates
    the track or names a genre ("... an uplifting atmosphere, channeling the
    spirit of 1970s classic rock"); None when it cannot be used."""
    if _FILL_NEGATIVE_RE.search(sent) or _FILL_GENRE_PAIR_RE.search(sent):
        return None
    if not _FILL_SKIP_RE.search(sent):
        return sent
    lead = _FILL_TRAIL_SPLIT_RE.split(sent, maxsplit=1)[0].strip()
    if lead != sent and len(lead) >= 25 and not _FILL_SKIP_RE.search(lead):
        return lead.rstrip(",;:") + "."
    return None


def _fill_clause(sent, pat):
    """The part of a long sentence that is about one field (its list item
    with the most of the field's words), or None."""
    pieces = []
    for p in re.split(r"(?:,\s+(?:and\s+|while\s+)?|:\s+)(?=(?:a|an|the)\b)", sent):
        pieces += re.split(r"\s+(?:that\s+\w+\s+(?:in\s+)?with|alongside|against)\s+(?=(?:a|an|the)\b)", p)
    if len(pieces) < 2:
        return None
    best, best_sc = None, 0
    for p in pieces:
        sc = len(re.findall(r"\b(?:" + pat + r")", p.lower()))
        if sc > best_sc:
            best, best_sc = p.strip(), sc
    if best is None or best_sc < 1:
        return None
    best = best[:1].upper() + best[1:]
    return best if best.endswith((".", "!", "?")) else best.rstrip(",;:") + "."


def _fill_sentences(prose):
    """(heading-topic, sentence) units from the listening model's prose. A
    heading ("Instrumentation & production:", "Vocal Characteristics -- ...",
    "- Drums: ...") gives its topic to the sentences under / after it."""
    out = []
    topic = None
    for raw in str(prose or "").split("\n"):
        ln = raw.strip()
        if not ln:
            continue
        ln = re.sub(r"\s*\[(?:measured|the listening model)[^\]]*\]", "", ln)
        if re.match(r"^[-\u2022* ]*[A-Za-z][A-Za-z_ ]{1,30}=", ln):
            continue  # a FIELD=value line (CONFIDENCE=, EVIDENCE=, Genre_Related= ...)
        bullet = ln[:1] in "-\u2022*"
        body = ln.lstrip("-\u2022* ").strip()
        unit_topic = topic
        hm = re.match(r"^([A-Z][A-Za-z &/,\u2011-]{2,40}?)\s*(?::|\u2013|\u2014| - )\s*(.*)$", body)
        if hm and len(hm.group(1).split()) <= 5:
            t = None
            for f, pat in _FILL_HEADINGS:
                if re.search(pat, hm.group(1).lower()):
                    t = f
                    break
            if bullet:
                unit_topic = t or topic
                if t is not None or not hm.group(2).strip():
                    body = hm.group(2).strip()
            else:
                topic = unit_topic = t
                body = hm.group(2).strip()
            if not body:
                continue
        for sent in re.split(r"(?<=[.!?])\s+(?=[A-Z\u201c\"])", body):
            sent = sent.strip()
            if len(sent) >= 12 and not sent.endswith(":"):
                out.append((unit_topic, sent[:1].upper() + sent[1:], sent))
    return out


def _fill_from_prose(prose, wanted, ident=None, moved=None, already=""):
    """{field: text} for the wanted fields, from the listening model's own
    sentences: each sentence goes to the field whose words it uses most (its
    heading counts double), at most three sentences / ~420 characters each,
    and no sentence is used twice (dev67). A field still empty may then take
    an unused sentence, or the part of a used one that is about it ("a
    powerful drum kit featuring punchy kick, snappy snare ..." inside the
    instrumentation sentence). `moved`, when given, collects the original
    text of every sentence copied up whole."""
    units = []
    for topic, s, raw in _fill_sentences(prose):
        c = _fill_clean(s)
        if not c:
            continue
        whole = c == s
        s = c
        low = s.lower()
        best, best_score, scores = None, 0, []
        for f, pat in _FILL_TOPICS:
            sc = len(re.findall(r"\b(?:" + pat + r")", low))
            if topic == f:
                sc += 2
            scores.append(sc)
            if sc > best_score:
                best, best_score = f, sc
        if best_score == 1 and scores.count(1) > 1:
            best = None  # one word for each of several fields: not clearly about any
        if scores[-1] >= 3 or scores[-1] >= 2 and scores[-1] >= max(scores[:-1]):
            best = "_SECTIONS"  # walks through the song's sections: not used
        if ident and (best == "LEAD_VOCAL_CHARACTERISTICS" or best is None or
                      re.search(r"\b(?:" + dict(_FILL_TOPICS)["LEAD_VOCAL_CHARACTERISTICS"] + r")", low)) \
                and _fill_identity_conflict(s, ident):
            continue
        if any(u[0] == s for u in units):
            continue  # the same sentence again (report prose and era pass)
        if already and s in already:
            continue  # already in one of the report's fields
        if s[:1] in "\u201c\"" and best != "LYRIC_SUBJECT":
            continue  # a quoted lyric line describes the lyrics, nothing else
        units.append((s, best, raw if whole else None))
    got, used = {}, set()
    for s, best, raw in units:
        if best is None or best not in wanted:
            continue
        cur = got.setdefault(best, [])
        words = set(re.findall(r"[a-z]{4,}", s.lower()))
        if any(len(words & set(re.findall(r"[a-z]{4,}", x.lower()))) > 0.6 * max(1, len(words)) for x in cur):
            continue  # says the same as a sentence already there
        if len(cur) < 3 and sum(len(x) for x in cur) + len(s) <= 420 or not cur:
            cur.append(s)
            used.add(s)
            if moved is not None and raw:
                moved.add(raw)
    for f, pat in _FILL_TOPICS:
        if f not in wanted or got.get(f):
            continue
        for s, best, raw in units:
            n_words = len(re.findall(r"\b(?:" + pat + r")", s.lower()))
            # a sentence no field kept needs one of this field's words; one
            # already used elsewhere lends only its part about this field
            if best == "_SECTIONS" or n_words < (1 if s not in used else 2):
                continue
            if s[:1] in "\u201c\"" and f != "LYRIC_SUBJECT":
                continue
            if s in used or len(s) > 200:
                c = _fill_clause(s, pat)
                if not c or c in used:
                    continue
            else:
                c = s
            got[f] = [c]
            used.add(c)
            if moved is not None and raw and c == s:
                moved.add(raw)
            break
    return {f: " ".join(v) for f, v in got.items() if v and not f.startswith("_")}


def _fill_era_prose(text):
    m = re.search(r"(?ms)^11\. ERA / RELEASE PERIOD \([^\n]*\):\n(.*?)(?=\n\n(?:VOCAL / SINGER PROFILE|[A-Z][A-Z /&-]{6,}[:(])|\Z)", text)
    return m.group(1) if m else ""


def _note_reading_range(text):
    """'lead vocal A3-G4 (median D4)' from a VOCAL NOTE PITCH line (the vocal
    stem's transcribed notes), or None."""
    m = re.search(r"VOCAL NOTE PITCH[^\n]*?median \d+ Hz \(([A-G][#♯♭b]?\d)\); 10-90%: [0-9]+-[0-9]+ Hz "
                  r"\(([A-G][#♯♭b]?\d) to ([A-G][#♯♭b]?\d)\)", str(text or ""))
    if not m or m.group(2) == m.group(3):
        return None
    return f"lead vocal {m.group(2)}-{m.group(3)} (median {m.group(1)})"


def _fill_measured(text):
    """{field: value} measured elsewhere in the saved analysis."""
    got = {}
    m = re.search(r"RECOMMENDED KEY FOR DISCUSSION:\s*([^\n]+?)\.\s+Reasoning", text)
    if m and not re.search(r"uncertain|unknown|ambiguous", m.group(1), re.I):
        got["KEY"] = m.group(1).strip()
    m = re.search(r"RECOMMENDED TEMPO FOR DISCUSSION:\s*(?:about\s+|~)?(\d+(?:\.\d+)?)\s*BPM\.", text)
    if m:
        got["TEMPO_BPM"] = m.group(1)
    vp = re.findall(r"(?m)^VOCALS_PRESENT=(yes|no)\b[^\n]*#\s*from\b", text)
    if not vp:
        vp = re.findall(r"(?m)^VOCAL PRESENCE:\s*(yes|no)\b", text)
    if not vp and VOCAL_ARBITRATION_MARKER not in text:
        # The dedicated vocal pass's own answer (MOSS-Music writes it as a
        # field), when no arbitration overruled it.
        m = re.search(r"(?m)^VOCAL / SINGER PROFILE \(isolated dedicated pass\):\n"
                      r"(?:[^\n]*\n){0,3}?VOCALS_PRESENT=(yes|no)\b", text)
        if m:
            vp = [m.group(1)]
    if vp:
        got["VOCALS_PRESENT"] = vp[-1]
    if got.get("VOCALS_PRESENT") != "no" and not re.search(r"(?m)^VOCALS_PRESENT=no\b", text):
        m = re.search(r"(?ms)^LYRICS HEARD IN AUDIO[^\n]*:\n(.*?)(?:\n\n|\Z)", text)
        words = re.findall(r"[A-Za-z']+", re.sub(r"\[[^\]]*\]", " ", m.group(1))) if m else []
        if len(words) >= 20 or re.search(r"(?m)^LYRICS: reliable lyrics are present", text):
            got["LYRICS_PRESENT"] = "yes"
        m = re.search(r"VOCAL PITCH REPORT[^\n]*\n(?:[^\n]*\n){0,12}?\s*- practical range in notes:\s*([A-G][#♯♭b]?\d\s+to\s+[A-G][#♯♭b]?\d)", text)
        _nr = _note_reading_range(text) if re.search(r"pyin pitch reading confidence: LOW", text) else None
        if _nr:
            # The pyin reading is marked LOW confidence and the notes say to
            # prefer the note reading: take the range from that.
            got["MELODIC_RANGE"] = _nr
            m = None
        if m and len(set(m.group(1).split(" to "))) == 2:
            med = re.search(r"VOCAL PITCH REPORT[^\n]*\n(?:[^\n]*\n){0,4}?\s*- approximate median note:\s*([A-G][#♯♭b]?\d)", text)
            got["MELODIC_RANGE"] = ("lead vocal " + m.group(1).replace(" to ", "-")
                                    + (f" (median {med.group(1)})" if med else ""))
        m = re.search(r"(?ms)^MELODY PER SECTION \(vocals stem\)[^\n]*\n(.*?)(?:\n\n|\n(?=[A-Z][A-Z /]{3,}[(:])|\Z)", text)
        if m:
            groups = {}
            for sec, contour in re.findall(r"(?m)^\s*([^\n(]+?) \([0-9:]+-[0-9:]+\): melody[^\n]*?; contour: ([^;\n]+)", m.group(1)):
                groups.setdefault(contour.strip(), []).append(sec.strip())
            if groups:
                got["MELODIC_CONTOUR"] = "vocal stem, per section: " + "; ".join(
                    f"{c} ({', '.join(v)})" for c, v in groups.items())
    return got


def _report_fill_from_evidence(out, structure_given=False):
    """dev66: see REPORT_FILL_FROM_EVIDENCE. `out` is a laid-out analysis
    (_moss_canonical_report); fields that are still "uncertain" are filled and
    REPORT_NOTE says where every filled value came from."""
    if not REPORT_FILL_FROM_EVIDENCE:
        return out
    try:
        cut = _moss_report_head_end(out)
        head, tail = out[:cut], out[cut:]
        fields, _ = _moss_parse_report(head)
        head, fields = _report_unsettled_vocal_fields(head, fields, tail)
        missing = [n for n in _MOSS_CANONICAL_FIELDS
                   if n != "STRUCTURE" and str(fields.get(n, "")).strip() in ("uncertain", "")]
        desc = str(fields.get("DESCRIPTION", "")).strip()
        era = _fill_era_prose(tail)
        filled_model, filled_measured = {}, {}
        moved = set()
        if missing:
            meas = _fill_measured(tail)
            for n in ("KEY", "TEMPO_BPM", "VOCALS_PRESENT", "LYRICS_PRESENT", "MELODIC_RANGE"):
                if n in missing and meas.get(n):
                    filled_measured[n] = meas[n]
            if "MELODIC_CONTOUR" in missing and meas.get("MELODIC_CONTOUR"):
                filled_measured["MELODIC_CONTOUR"] = meas["MELODIC_CONTOUR"]
            wanted = [n for n in missing if n not in filled_measured
                      and n not in ("GENRE_RANKED", "GENRE_ADJACENT", "GENRE_RULED_OUT", "KEY", "TEMPO_BPM",
                                    "VOCALS_PRESENT", "LYRICS_PRESENT")]
            vp = str(filled_measured.get("VOCALS_PRESENT") or fields.get("VOCALS_PRESENT") or "").strip().lower()
            if vp.startswith("no"):
                # decided instrumental: no singer or lyric description is copied up
                wanted = [n for n in wanted if n not in ("LEAD_VOCAL_CHARACTERISTICS", "LYRIC_SUBJECT")]
            if wanted:
                d2 = desc if desc and desc != "(none given)" else ""
                d2 = re.sub(r"\s+-\s+(?=[A-Z])", "\n- ", d2)
                d2 = re.sub(r"(?<=[.!?\u201d\"])\s+(?=[A-Z][A-Za-z &/\u2011-]{2,40}?\s?[\u2013\u2014:]\s)", "\n", d2)
                src = d2 + "\n" + era
                if vp.startswith("yes"):
                    # a voice is settled as present: no "absence of vocals"
                    # sentence may be copied into a vocal field
                    src = _drop_no_vocal_sentences(src)
                _have = "\n".join(v for k, v in fields.items() if k not in ("DESCRIPTION", "REPORT_NOTE"))
                filled_model = _fill_from_prose(src, set(wanted), ident=_fill_identity(tail), moved=moved,
                                                already=_have)
                if ("LEAD_VOCAL_CHARACTERISTICS" in wanted and not filled_model.get("LEAD_VOCAL_CHARACTERISTICS")
                        and vp.startswith("yes") and VOCAL_ARBITRATION_MARKER not in tail):
                    # The dedicated vocal pass's own description of the voice
                    # (MOSS-Music writes it as fields) when the report gave none.
                    _vb = re.search(r"(?ms)^VOCAL / SINGER PROFILE \(isolated dedicated pass\):\n(.*?)(?:\n\n|\Z)", tail)
                    if _vb:
                        _vparts = []
                        for _f in ("LEAD_PROFILE_NOTE", "TIMBRE_NOTE", "DELIVERY_NOTE"):
                            _m = re.search(r"(?m)^" + _f + r"=(.+)$", _vb.group(1))
                            _v = _m.group(1).strip() if _m else ""
                            if _v and _v.lower() not in ("none", "uncertain", "n/a") and \
                                    not _INSTRUMENTAL_PROSE_RE.search(_v):
                                _vparts.append(_v)
                        if _vparts:
                            filled_model["LEAD_VOCAL_CHARACTERISTICS"] = "; ".join(_vparts)
        new_desc = None
        in_fields = " ".join(list(filled_model.values()) + [v for k, v in fields.items()
                                                             if k not in ("DESCRIPTION", "REPORT_NOTE")])
        if desc in ("", "(none given)") and era:
            ident = _fill_identity(tail)
            sents = []
            for _t, s0, _r in _fill_sentences(era):
                c = _fill_clean(s0)
                if not c or _fill_identity_conflict(c, ident) or c in in_fields or c[:1] in "\u201c\"":
                    continue  # dev67: a sentence already in a field (or a quoted lyric line) is not repeated here
                if len(re.findall(r"\b(?:" + dict(_FILL_TOPICS)["_SECTIONS"] + r")", c.lower())) >= 2:
                    continue
                sents.append(c)
            if sents:
                new_desc = " ".join(sents[:2])
        elif desc and desc != "(none given)" and moved and REPORT_FIXED_BLOCKS:
            # dev67: sentences of the report's own prose that now sit in a
            # field are moved there, not repeated in DESCRIPTION.
            d3 = desc
            for raw in sorted(moved, key=len, reverse=True):
                d3 = d3.replace(raw, " ")
            _hd = r"[A-Z][A-Za-z &/\u2011-]{2,40}?\s?[\u2013\u2014:]"
            for _ in range(3):
                d3 = re.sub(r"(?:(?<=\s)|^)" + _hd + r"\s*(?=(?:" + _hd + r"\s)|$)", " ", d3)
                d3 = re.sub(r"(?:(?<=\s)|^)-\s+(?=-\s|$)", " ", d3)
            d3 = re.sub(r"\s{2,}", " ", d3).strip()
            if d3 != desc:
                new_desc = d3 if len(re.findall(r"[A-Za-z]", d3)) >= 12 else \
                    "(the listening model's description is in the fields above)"
        # A voice settled as present: the model's own sentences saying there
        # is none ("Despite the absence of vocals in this recording ...") are
        # left out of DESCRIPTION too.
        _vp_now = str(filled_measured.get("VOCALS_PRESENT") or fields.get("VOCALS_PRESENT") or "").strip().lower()
        if _vp_now.startswith("yes"):
            _dbase = new_desc if new_desc is not None else desc
            if _dbase and _dbase != "(none given)":
                _dscr = _drop_no_vocal_sentences(_dbase)
                if _dscr != _dbase:
                    new_desc = _dscr or "(none given)"
        old_note = re.search(r"(?m)^REPORT_NOTE=(.*)$", head)
        struct_in_note = bool(old_note and structure_given and re.search(
            r"(?:first pass[^;]*|not given[^;]*)\bSTRUCTURE\b", old_note.group(1)))
        if not (filled_model or filled_measured or new_desc or struct_in_note):
            # CHORDS / INSTRUMENTATION in the current format (the same repair
            # as at load time, so saved and loaded scans read alike).
            return _report_line_repairs(out)

        def _sub(name, value):
            nonlocal head
            head = re.sub(r"(?m)^" + name + r"=uncertain$", lambda _m: f"{name}={value}", head, count=1)

        for n, v in filled_measured.items():
            if v:
                _sub(n, v)
        for n, v in filled_model.items():
            if v:
                _sub(n, v.replace("\n", " "))
        if new_desc:
            head = re.sub(r"(?m)^DESCRIPTION=.*$", lambda _m: "DESCRIPTION=" + new_desc, head, count=1)
        # REPORT_NOTE: where each value came from.
        old = re.search(r"(?m)^REPORT_NOTE=(.*)$", head)
        first = []
        if old:
            m1 = re.search(r"from the listening model's first pass \([^)]*\): ([^;]*)", old.group(1))
            if m1:
                first = [x.strip() for x in m1.group(1).split(",") if x.strip() and x.strip() != "STRUCTURE"]
        prev_model, prev_meas, prev_desc = [], [], []
        if old:
            mm = re.search(r"from the listening model's own description further down \(its report left them out\): ([^;]*)", old.group(1))
            prev_model = [x.strip() for x in mm.group(1).split(",")] if mm else []
            # a field reset to uncertain above no longer comes from there
            prev_model = [x for x in prev_model
                          if x in filled_model or str(fields.get(x, "")).strip() != "uncertain"]
            mm = re.search(r"measured, not the listening model's words: ([^;]*)", old.group(1))
            prev_meas = [x.strip() for x in mm.group(1).split(",")] if mm else []
            prev_desc = [p for p in old.group(1).split("; ") if p.startswith("DESCRIPTION ")]
        filled_model = dict(filled_model)
        for k in prev_model:
            filled_model.setdefault(k, "")
        filled_measured = dict(filled_measured)
        for k in prev_meas:
            filled_measured.setdefault(k, "")
        done = set(filled_model) | set(filled_measured)
        still = [n for n in missing if n not in done]
        parts = []
        if first:
            parts.append("from the listening model's first pass (its self-check reply left them out): " + ", ".join(first))
        if filled_model:
            parts.append("from the listening model's own description further down (its report left them out): "
                         + ", ".join(n for n in _MOSS_CANONICAL_FIELDS if n in filled_model))
        if not new_desc and prev_desc:
            parts.extend(prev_desc)
        elif new_desc and desc in ("", "(none given)"):
            parts.append("DESCRIPTION from the listening model's era-pass description")
        elif new_desc:
            parts.append("DESCRIPTION keeps the rest of the report's own prose (sentences moved into the fields "
                         "above are not repeated)")
        if filled_measured:
            parts.append("measured, not the listening model's words: "
                         + ", ".join(n for n in _MOSS_CANONICAL_FIELDS if n in filled_measured))
        if structure_given:
            parts.append("STRUCTURE is the measured section map")
        if still:
            parts.append("not given by the listening model: " + ", ".join(still))
        if not (first or filled_model or filled_measured or new_desc or still):
            parts.insert(0, "all fields from the listening model's self-check")
        if old:
            head = head[:old.start(1)] + "; ".join(parts) + head[old.end(1):]
        return _report_line_repairs(head + tail)
    except Exception:
        return out


_DESC_TIDY_NOTE = ("DESCRIPTION leaves out the listening model's own section times, chord lists, "
                   "field fragments and key or tempo statements that disagree with KEY / TEMPO_BPM")
_ENUM_PREFIX_RE = re.compile(r"^\s*(?:\(?\d{1,2}[).]\s*)+")
_DESC_TIME_RE = re.compile(r"\b\d{1,2}:\d{2}\b")
_DESC_HEADER_RE = re.compile(r"^[A-Z][A-Za-z]+(?:\s+(?:&|and)?\s*[A-Za-z]+){0,3}\s*:\s")
_DESC_FIELD_RE = re.compile(r"^\s*[A-Za-z][A-Za-z_]{2,30}\s*=")
_KEY_NAME_RE = r"([A-G])[\s  ]*([#b♯♭]?)(?:[\s  ]*(major|minor))?"
_DESC_KEY_CLAIM_RE = re.compile(
    r"(?i:rooted in|in the key of|key of|diatonic to|cent(?:re|er)d? (?:on|in|around)|"
    r"tonal cent(?:re|er)(?: of| on| in| is)?|tonic)\s*\(?(?:the\s+)?" + _KEY_NAME_RE + r"(?![a-z0-9])")
_PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


def _key_pc_mode(letter, acc, mode):
    pc = _PC[letter.upper()] + (1 if acc in ("#", "♯") else -1 if acc in ("b", "♭") else 0)
    return pc % 12, (mode or "").lower()


def _first_key(text):
    m = re.search(r"\b" + _KEY_NAME_RE + r"(?![a-z0-9])", str(text or ""))
    return _key_pc_mode(m.group(1), m.group(2), m.group(3)) if m and m.group(3) else None


def _first_bpm(text):
    m = re.search(r"(\d{2,3}(?:\.\d+)?)", str(text or ""))
    return float(m.group(1)) if m else None


def _tidy_description(desc, keys, tempos):
    """dev73: DESCRIPTION without what contradicts or duplicates the
    measured fields: the listening model's own section walkthrough with its
    own times (the measured map is STRUCTURE), chord-arrow lists, field-like
    fragments ("Genre_Related=..."), and sentences naming a key or a tempo
    that agrees with neither the report's KEY / TEMPO_BPM nor the
    recommended figure. Returns (text, removed_anything)."""
    units = [u for u in re.split(r"(?<=[.!?])\s+|\s+(?=-\s+[A-Z])", str(desc or "").strip()) if u.strip()]
    keep, decided, removed, walk = [], [], False, False
    for u in units:
        s = u.strip()
        drop = False
        if walk and (s.startswith("- ") or not _DESC_HEADER_RE.match(s)):
            drop = True              # rest of a section walkthrough
        else:
            walk = False
        if not drop and _DESC_TIME_RE.search(s):
            drop = True
            walk = bool(_DESC_SECTION_TIME_RE.search(s))
        if not drop and (len(re.findall("→|->", s)) >= 6 or re.match(r"(?i)chord progression\b", s)
                         or "STRUCTURE=[" in s or _DESC_FIELD_RE.match(s)):
            drop = True
        if not drop and keys:
            for m in _DESC_KEY_CLAIM_RE.finditer(s):
                pc, mode = _key_pc_mode(m.group(1), m.group(2), m.group(3))
                if not any(pc == k_pc and (not mode or mode == k_mode) for k_pc, k_mode in keys):
                    drop = True
                    break
        if not drop and tempos:
            for m in re.finditer(r"(\d{2,3}(?:\.\d+)?)\s*BPM", s):
                b = float(m.group(1))
                if not any(abs(b - x * f) <= 4.0 for x in tempos for f in (0.5, 1.0, 2.0)):
                    drop = True
                    break
        if drop:
            removed = True
        decided.append((s, drop))
    # a heading left with nothing under it ("Song structure & dynamics:")
    for i, (s, drop) in enumerate(decided):
        if drop:
            continue
        if re.fullmatch(r"[A-Z][^.!?:]{0,60}:", s) and (i + 1 >= len(decided) or decided[i + 1][1]):
            removed = True
            continue
        keep.append(s)
    return " ".join(keep).strip(), removed


def _saved_report_tidy(text):
    """dev73: in a saved analysis's report, (1) GENRE_RANKED written once per
    item -- a numbered "Genre: 1) a; 2) b" line had been numbered again
    ("1) 1) a; 2) 2) b"); the labels are unchanged, as the genre code
    already read past the extra numbers -- and (2) DESCRIPTION tidied by
    _tidy_description, with a REPORT_NOTE line saying so. Idempotent."""
    if not REPORT_TIDY_SAVED:
        return text
    t = str(text or "")
    try:
        cut = _moss_report_head_end(t)
        head, tail = t[:cut], t[cut:]
        if not re.search(r"(?m)^REPORT_NOTE=", head):
            return text

        def _gr(m):
            items = [x.strip() for x in m.group(1).split(";")]
            if not any(re.match(r"^\s*(?:\(?\d{1,2}[).]\s*){2}", x) for x in items):
                return m.group(0)
            items = [_ENUM_PREFIX_RE.sub("", x).strip() for x in items]
            items = [x for x in items if x]
            return "GENRE_RANKED=" + "; ".join(f"{i + 1}) {x}" for i, x in enumerate(items))
        head = re.sub(r"(?m)^GENRE_RANKED=(.*)$", _gr, head, count=1)

        dm = re.search(r"(?m)^DESCRIPTION=(.*)$", head)
        if dm and dm.group(1).strip() not in ("", "(none given)"):
            keys = []
            kf = re.search(r"(?m)^KEY=(.*)$", head)
            rk = re.search(r"RECOMMENDED KEY FOR DISCUSSION:\s*([^\n.]*)", tail)
            for src in ((kf.group(1) if kf else ""), (rk.group(1) if rk else "")):
                k = _first_key(src)
                if k and k not in keys:
                    keys.append(k)
            tempos = []
            tf = re.search(r"(?m)^TEMPO_BPM=(.*)$", head)
            rt = re.search(r"RECOMMENDED TEMPO FOR DISCUSSION:\s*([^\n]*)", tail)
            for src in ((tf.group(1) if tf else ""), (rt.group(1) if rt else "")):
                b = _first_bpm(src)
                if b and b not in tempos:
                    tempos.append(b)
            new, removed = _tidy_description(dm.group(1), keys, tempos)
            if removed:
                head = head[:dm.start()] + "DESCRIPTION=" + (new or "(none given)") + head[dm.end():]
                nm = re.search(r"(?m)^REPORT_NOTE=(.*)$", head)
                if nm and _DESC_TIDY_NOTE not in nm.group(1):
                    head = head[:nm.end()] + "; " + _DESC_TIDY_NOTE + head[nm.end():]
        return head + tail
    except Exception:
        return text


def _saved_report_tidy_then_fill(text, structure_sections=None):
    """dev73: _saved_report_tidy, and when that left DESCRIPTION empty, the
    dev66 fill once more so the description it would give an empty field on
    the next save is given now (saving stays idempotent)."""
    t = _saved_report_tidy(text)
    if t == text or not REPORT_FILL_FROM_EVIDENCE:
        return t
    if re.search(r"(?m)^DESCRIPTION=\(none given\)$", t[:_moss_report_head_end(t)]):
        t = _saved_report_tidy(_report_fill_from_evidence(t, structure_given=bool(structure_sections)))
    return t


_ERA_DRUM_NOTE_HEAD = "ERA NOTE (drums):"


def _era_drum_evidence_check(text):
    """With a DRUM STEM NOTE (no kit on the recording), flag the era pass's
    drum observations as contradicted, right after its EVIDENCE line. The era
    pass hears the mix only and described "vintage drum machines" and "analog
    drum recording" on acoustic recordings with no drums. Idempotent; text
    without a DRUM STEM NOTE is returned unchanged."""
    t = str(text or "")
    if "DRUM STEM NOTE:" not in t or _ERA_DRUM_NOTE_HEAD in t:
        return t
    i = t.find("ERA / RELEASE PERIOD (isolated dedicated pass")
    if i < 0:
        return t
    m = re.compile(r"(?m)^EVIDENCE=([^\n]*)$").search(t, i)
    if not m or not _DRUM_CLAIM_RE.search(m.group(1)) and not re.search(r"(?i)drum machine", m.group(1)):
        return t
    note = ("\n" + _ERA_DRUM_NOTE_HEAD + " the era pass's drum observations above are contradicted -- "
            "there is no drum kit on this recording (see DRUM STEM NOTE). Leave them out of any era "
            "reasoning; date it from the other evidence.")
    return t[:m.end()] + note + t[m.end():]


def build_objective_audio_report(local_path, original_path=None):
    """`local_path` is the mono 22.05 kHz working WAV used for every
    measurement. `original_path`, when given, is the untouched source file and
    is used ONLY for the stereo-image measurement, which cannot work on a
    single-channel downmix. Optional and defaulting to None so any existing
    caller behaves exactly as before."""
    try:
        y, sr = librosa.load(local_path, sr=None, mono=True)
        duration = len(y) / float(sr)
        if duration < 1.0:
            return ""

        lines = []
        lines.append("OBJECTIVE AUDIO MEASUREMENTS")
        lines.append(f"duration: {round(duration, 2)} s")

        # Structural boundary detection is moved up here, right after duration,
        # and computed before any of the riskier downstream measurements
        # (LUFS, beat tracking, spectral peaks, band onset density). Those
        # later blocks can throw on unusual audio; computing/recording the
        # boundary candidates first means they're already captured into
        # `lines` and won't be lost even if something later fails (see the
        # partial-report fallback at the bottom of this function).
        try:
            _boundary_times = _structural_novelty_boundaries(y, sr)
        except Exception:
            _boundary_times = []
        if _boundary_times:
            lines.append("")
            lines.append(
                "STRUCTURAL BOUNDARY CANDIDATES (objective, timbre+harmony+energy self-similarity "
                "novelty -- a raw first-pass cross-check, NOT the section map and NOT section names/labels)"
            )
            lines.append(
                "candidate change points: "
                + ", ".join(_seconds_to_mmss(t) for t in _boundary_times)
            )
            lines.append(
                "WARNING: The raw list on the line above is unmerged DSP peaks. Do not read that "
                "raw list out as the song's structure, and do not quote it as section boundaries. "
                "ALWAYS use the RECOMMENDED STRUCTURE FOR DISCUSSION block for section names and "
                "timestamps. NOTE: if that block appears with the heading '(from objective change "
                "points)', it was built by merging and validating these very peaks against the "
                "track duration — it is a finished section map and IS authoritative. Otherwise the "
                "section map is built separately, bar by bar, from drum fills, vocal entries, "
                "repetition and lyric timing, so some peaks here fall inside sections and some real "
                "section changes are missing; where the two disagree, the RECOMMENDED STRUCTURE is "
                "right. This warning applies only to the raw list here, never to the RECOMMENDED "
                "STRUCTURE block."
            )

        peak = float(np.max(np.abs(y))) if len(y) else 0.0
        rms = float(np.sqrt(np.mean(np.square(y.astype(np.float64))))) if len(y) else 0.0
        lines.append("")
        lines.append("DYNAMIC RANGE / LOUDNESS (numeric)")
        try:
            # Measure loudness on the untouched source when it is available.
            # local_path is the mono 22.05 kHz working copy, and a mono downmix
            # reads roughly 3 LU quieter than the stereo master under EBU R128
            # -- which made this block disagree with RECOMMENDED DYNAMICS (which
            # already uses the original) and skewed the loud-master era caveat
            # that is parsed back out of this text.
            _loud_path = local_path
            if (
                original_path
                and isinstance(original_path, str)
                and not original_path.startswith(("http://", "https://"))
                and os.path.exists(original_path)
            ):
                _loud_path = original_path
            _lufs, _lra, _lsrc = measure_ebur128_loudness(_loud_path)
            if _lufs is None and _loud_path != local_path:
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
                lines.append(f"crest band (peak-to-average / peak limiting, not loudness range): {label}")
            if plain:
                lines.append(f"dynamic-range note: {plain}")
            if crest_db < 10.0:
                lines.append(
                    "compression character: low crest factor is consistent with heavy "
                    "limiting/compression; do not infer release era from crest factor alone."
                )
            elif crest_db >= 14.0:
                # Same boundary as the crest band above (_dynamics_label_from_crest
                # calls 14.0 dB "high"); with "> 14.0" a 14.0 dB track was "high
                # crest" on one line and "moderate crest factor" on the next.
                lines.append(
                    "compression character: high crest factor is consistent with a more dynamic, "
                    "less-limited master; do not infer release era from crest factor alone."
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
        try:
            # Old librosa: beat_track() accepted std_bpm directly.
            tempo_raw, beats = librosa.beat.beat_track(y=y, sr=sr, start_bpm=120.0, std_bpm=30.0)
        except TypeError:
            # Newer librosa removed std_bpm from beat_track() (it now wants
            # a scipy.stats `prior` distribution instead). librosa.feature.tempo()
            # still exposes std_bpm and uses the exact same log-normal prior
            # math beat_track() used to apply internally, so compute the
            # tempo there first (preserving the widened-prior tuning) and
            # feed the fixed tempo into beat_track() for beat positions.
            tempo_raw = librosa.feature.tempo(y=y, sr=sr, start_bpm=120.0, std_bpm=30.0)
            _tempo_val = float(np.atleast_1d(tempo_raw)[0]) if np.size(tempo_raw) else 120.0
            _, beats = librosa.beat.beat_track(y=y, sr=sr, bpm=_tempo_val)
            tempo_raw = np.array([_tempo_val])
        tempo_arr = np.atleast_1d(tempo_raw)
        tempo = float(tempo_arr[0]) if len(tempo_arr) else None
        # librosa reports tempo in whole onset frames (60 * sr/512 / k), a
        # coarse ladder: 120 BPM reads as 117.5, ~158 as 161.5. Refine within
        # +/-6 % (same metrical level) from the onset envelope's periodicity.
        beat_times = librosa.frames_to_time(beats, sr=sr)
        if tempo:
            try:
                _env_ref = librosa.onset.onset_strength(y=y, sr=sr, hop_length=512)
                _tempo_ref = _sx_refine_tempo(_env_ref, sr / 512.0, tempo)
                if _tempo_ref and abs(_tempo_ref - tempo) / tempo <= 0.06:
                    tempo = float(_tempo_ref)
                    # Re-track at the refined tempo (same metrical level) on a
                    # finer hop and a kick-anchored envelope, so beat spacing,
                    # meter and the production measurements below use beats
                    # that do not drift against the music.
                    _b2 = _sx_track_beats_at(y, sr, float(tempo))
                    if _b2 is not None and _b2.size >= 8:
                        _med = 60.0 / float(np.median(np.diff(_b2)))
                        if abs(_med - tempo) / tempo <= 0.03:
                            beat_times = _b2
            except Exception:
                pass

        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
        frame_times = librosa.frames_to_time(np.arange(len(onset_env)), sr=sr)
        if len(beat_times):
            try:
                beat_accents = _downbeat_accent_curve(y, sr, beat_times, onset_env=onset_env, hop_length=512)
            except Exception:
                beat_accents = np.interp(beat_times, frame_times, onset_env)
        else:
            beat_accents = np.array([])
        downbeat_times, beats_per_bar = _estimate_meter_from_beats(beat_times, beat_accents)

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
                meter = int(beats_per_bar or 1)
                bar_times = np.asarray(downbeat_times, dtype=float)
                if len(bar_times) > 1:
                    bar_durs = np.diff(bar_times)
                    meter_label = {2: "2/4", 3: "3/4", 4: "4/4", 5: "5/4", 6: "6/8", 7: "7/4", 8: "8-beat cycle (usually 4/4 with a two-bar accent pattern, not literally 8/8)", 9: "9/8", 12: "12/8"}.get(meter, f"{meter}-beat meter")
                    bar_note = ""
                    # The bars are counted in the tracker's own beats. When that
                    # pulse is very fast it is usually double the felt tempo (the
                    # dance-pop test track: 240 tracker BPM, 120 felt), and "4/4,
                    # 1.0 s bars" then describes half-bars. Say so, so the bar
                    # length is not read against the RECOMMENDED TEMPO.
                    try:
                        _pulse = 60.0 / float(median_ibis) if median_ibis else None
                    except (TypeError, ValueError, ZeroDivisionError):
                        _pulse = None
                    if _pulse and _pulse >= 175.0 and meter in (2, 4):
                        bar_note = (
                            f"; counted in the detector's own ~{_pulse:.0f}-BPM beats — if the felt "
                            f"tempo is half that (~{_pulse / 2.0:.0f} BPM), a 4/4 bar lasts "
                            f"~{480.0 / _pulse:.1f} s"
                        )
                    lines.append(f"meter estimate: {meter_label} ({len(bar_times)} bars, median bar length {round(float(np.median(bar_durs)), 3)} s{bar_note})")

                if not COMPACT_OBJECTIVE_REPORT and len(downbeat_times):
                    shown = ", ".join(f"{round(float(t), 2)}" for t in downbeat_times[:16])
                    suffix = "..." if len(downbeat_times) > 16 else ""
                    lines.append(
                        f"candidate downbeats (accent-cycle meter estimate): {shown}{suffix}"
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
                            lines.append(f"  {name} ({lo}-{min(int(hi), int(sr // 2))} Hz): {round(band_power * 100.0, 1)}%")

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

        # Production/mix measurements (stereo image, true peak/clipping,
        # bandwidth, pumping, ambience, whole-track balance). Appended last and
        # fully failure-isolated so it can never cost the report anything that
        # was already computed above. `y`/`beat_times` are handed over so the
        # file is not decoded and beat-tracked a second time.
        try:
            _prod_block = build_production_analysis_report(
                local_path,
                y=y,
                sr=sr,
                duration=duration,
                beat_times=(beat_times if len(beat_times) else None),
                original_path=original_path,
            )
        except Exception as _pe:
            _prod_block = ""
            try:
                print(f"  (production analysis skipped -- {type(_pe).__name__}: {_pe})")
            except Exception:
                pass
        if _prod_block:
            lines.append("")
            lines.append(_prod_block)

        return "\n".join(lines)
    except Exception as _e:
        # Previously this silently returned "" on ANY exception anywhere in
        # this ~270-line function, discarding even successfully-computed
        # data (duration, structural boundary candidates, etc.) with zero
        # visibility into why. Now: report the failure so it's actually
        # diagnosable, and salvage whatever was already computed rather than
        # throwing away a partial report because one unrelated measurement
        # (e.g. spectral peaks, band onset density) failed.
        try:
            print(f"  (objective audio report: partial/failed -- {type(_e).__name__}: {_e})")
        except Exception:
            pass
        try:
            return "\n".join(lines) if lines else ""
        except Exception:
            return ""
