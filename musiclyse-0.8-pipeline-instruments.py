# MUSICLYSE FILE pipeline-instruments 0.8
# ==============================================================================
#  MUSICLYSE 0.8: INSTRUMENTS
# ==============================================================================
# This file holds instruments: the PANNs and MTG-Jamendo taggers, per-stem tags and the
# instrument checks. It is part of musiclyse-0.8.py, which loads it whole at start-up, after its
# own code, into its own namespace (see "THE MUSICLYSE FILES" there); it is not run on its own.
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
# (ENABLE_INSTRUMENT_TAGGING moved to the settings file beside this script.)
# Raised from 0.08: AudioSet guitar/synth/strings classes often fire weakly on
# bright or midrange content that is not actually that instrument. Prefer
# omission over weak positives — the writer still has the listening model + stems.
# (INSTRUMENT_TAG_MIN_PROB moved to the settings file beside this script.)
# (INSTRUMENT_TAG_TOP_K moved to the settings file beside this script.)
# Guitar-family labels are especially noisy on synths, distorted bass, and
# residual bleed. Require a higher bar (and preferably whole-mix agreement).
# (INSTRUMENT_TAG_GUITAR_MIN_PROB moved to the settings file beside this script.)
INSTRUMENT_TAG_GUITAR_LABELS = (
    "electric guitar",
    "acoustic guitar",
    "guitar (type uncertain)",
    "slide/steel guitar",
)
# Which stems benefit from tagging. Vocals are already unambiguous once the
# vocal-stem detector fires. The big payoff is on the pitched/harmonic
# catch-all stems where identity is otherwise just a guess -- and on drums,
# which was previously excluded on the grounds that Omnizart's drum model
# already classifies it. That model only resolves kick, snare and hi-hat, so
# excluding drums meant nothing in the pipeline could ever report a crash, a
# ride, a tambourine or a shaker, no matter how prominent. The tagger's label
# map covers cymbal, tambourine, timpani, drum machine and drum roll, so it
# supplies exactly the kit detail the transcription cannot.
# (INSTRUMENT_TAG_STEMS moved to the settings file beside this script.)
# Drop a stem instrument tag when that stem has almost no pitched MIDI activity
# (Demucs residual / noise) unless the tag is very strong. Stops phantom
# "guitar on guitar stem" when the stem is effectively empty.
# (INSTRUMENT_TAG_REQUIRE_STEM_ACTIVITY moved to the settings file beside this script.)
# (INSTRUMENT_TAG_MIN_NOTES_FOR_WEAK moved to the settings file beside this script.)
# (INSTRUMENT_TAG_STRONG_PROB moved to the settings file beside this script.)

# Windowed tagging: instead of one prediction averaged over the whole clip
# (which dilutes anything that isn't present for most of the track, e.g. a
# guitar solo that only lasts 15s of a 3-minute song), the tagger is run on
# short overlapping windows and results are combined by taking, per label,
# the strongest window rather than a track-wide average. Set
# INSTRUMENT_TAG_WINDOW_SECONDS to None to fall back to old single-pass
# whole-clip behaviour.
# (INSTRUMENT_TAG_WINDOW_SECONDS moved to the settings file beside this script.)
# (INSTRUMENT_TAG_HOP_SECONDS moved to the settings file beside this script.)
# Cap on number of windows per clip, mainly to bound runtime on long tracks.
# Raised from 40: whole-mix instrument/genre/mood passes now share one cached
# set of inferences instead of re-running three times, so a larger budget
# costs less wall-clock than the old 40 did, and more windows means finer
# temporal resolution plus more reliable persistence counts.
# (INSTRUMENT_TAG_MAX_WINDOWS moved to the settings file beside this script.)
# Peak-normalize each window before tagging. PANNs was trained on full
# commercial/YouTube mixes; isolated Demucs stems (especially quieter ones
# like "other"/"piano") often sit well below that loudness, which silently
# suppresses genuinely-present instruments below INSTRUMENT_TAG_MIN_PROB.
# (INSTRUMENT_TAG_NORMALIZE moved to the settings file beside this script.)
# (INSTRUMENT_TAG_NORMALIZE_PEAK moved to the settings file beside this script.)

# --- Window coverage, persistence gating, and inference reuse ---------------
# COVERAGE FIX: windows used to be laid out strictly left-to-right from t=0 at
# a fixed hop until INSTRUMENT_TAG_MAX_WINDOWS was hit, which meant anything
# past ~(MAX_WINDOWS * HOP) seconds was never looked at at all -- on a 6-minute
# track the final 43% of the song was invisible to the tagger, so a string
# section that only enters in the last chorus, or an outro guitar solo, simply
# could not be detected. When the hop-spaced layout would exceed the window
# budget, starts are now distributed evenly across the WHOLE file instead of
# truncating at the start, so coverage is always end-to-end.
# (INSTRUMENT_TAG_EVEN_COVERAGE moved to the settings file beside this script.)

# PERSISTENCE GATING: results were combined by taking, per label, the single
# strongest window. That is the right call for detecting brief events (a 15s
# solo), but it also means one fluke window scores identically to an
# instrument audible throughout -- a classic false-positive source, because
# over 40 windows the chance that *some* window spuriously spikes is high.
# The tagger now also records in how many windows each label actually fired,
# so a one-window-only claim can be held to a higher bar while a sustained
# one is trusted at the normal bar. Set INSTRUMENT_TAG_MIN_WINDOW_HITS = 1 to
# restore the previous peak-only behaviour.
# (INSTRUMENT_TAG_MIN_WINDOW_HITS moved to the settings file beside this script.)
# A label seen in only one window must be this strong to survive.
# (INSTRUMENT_TAG_SINGLE_WINDOW_STRONG_PROB moved to the settings file beside this script.)
# Fraction of windows a label must fire in to count as "sustained" (reported
# to the writer, and allowed to pass mix-disagreement on its own merits).
# (INSTRUMENT_TAG_SUSTAINED_COVERAGE moved to the settings file beside this script.)
# Sustained-but-moderate tags are real instruments the peak-only rule used to
# under-rate; a label present in at least this share of windows is accepted
# slightly below its family threshold.
# (INSTRUMENT_TAG_SUSTAINED_PROB_RELIEF moved to the settings file beside this script.)

# PANNs inference over one file used to be repeated once per label map --
# whole-mix instruments, genre and mood each re-decoded the same audio and
# re-ran the same Cnn14 forward passes over the same windows, three times.
# Raw per-window probabilities are now cached and re-used across label maps,
# which makes the extra passes essentially free and buys back enough runtime
# to afford finer temporal resolution below.
# (PANNS_WINDOW_CACHE_SIZE moved to the settings file beside this script.)
# Persistence STATS (how many windows each label fired in) used to be evicted
# together with the probability matrices above, which is wrong: the memory
# pressure the LRU exists for is entirely in the (n_windows x 527) float32
# matrices, while the stats are a handful of floats per label. With a cache of
# 3 the whole-mix entry was always popped by the time the four per-stem passes
# had run, so the whole-mix tag report silently lost its window counts -- the
# sustained/intermittent/isolated distinction the writer prompt leans on -- and
# any later lookup on the mix had to re-run inference. Stats are now bounded
# separately and generously.
# (PANNS_TAG_STATS_CACHE_SIZE moved to the settings file beside this script.)

# --- Whole-mix instrument tagging (optional, in addition to per-stem) ------
# Per-stem tagging inherits whatever mistakes Demucs made when separating
# (e.g. a synth pad bleeding into "other" alongside real strings). Tagging
# the original, un-separated mix is a second, independent source of truth
# that isn't subject to separation artifacts, and can help flag cases where
# a stem's tags look suspicious relative to what's audible in the full mix.
# (ENABLE_WHOLE_MIX_INSTRUMENT_TAGGING moved to the settings file beside this script.)
# (WHOLE_MIX_INSTRUMENT_TAG_TOP_K moved to the settings file beside this script.)
# (WHOLE_MIX_INSTRUMENT_TAG_MIN_PROB moved to the settings file beside this script.)
# When a stem claims guitar (etc.) but the whole-mix tagger does not agree
# above this bar, annotate the stem tag as "weak / mix-disagrees" so the
# writer prefers omission over a confident instrument claim.
# (INSTRUMENT_TAG_MIX_AGREE_MIN_PROB moved to the settings file beside this script.)

# --- Guitar-family presence probe (absence must be EARNED, not assumed) -----
# The thresholds above are deliberately strict because a false guitar claim is
# expensive. But "did not clear a strict reporting bar" is NOT the same fact as
# "there is no guitar", and the pipeline used to conflate the two: any track
# whose guitar tag failed the 0.35 family bar -- or that simply got squeezed
# out of the whole-mix top-k by six drum/bass/synth labels -- produced a hard
# GUITAR ABSENCE NOTE, which the writer then treated as authoritative. On
# dense, brick-walled modern pop-rock masters (where PANNs routinely scores a
# loud overdriven rhythm guitar in the 0.15-0.30 band, and often reads it as
# synth) that turns a detector limitation into a confident musicological
# error, and then into a wrong genre call, since "no guitar" gets used as an
# argument against a rock/pop-rock reading.
#
# These constants define a SEPARATE, lower-stakes question answered from the
# raw per-window probabilities directly (bypassing top-k and the reporting
# thresholds entirely): is there measurable guitar-family energy anywhere in
# this mix? A confident absence claim now requires positive evidence of
# absence -- low peak AND low persistence AND an inactive Demucs guitar stem.
# Anything in between produces an explicit "inconclusive" note instead.
# (GUITAR_PROBE_ENABLED moved to the settings file beside this script.)
# A window counts as "fired" for persistence purposes at/above this.
# (GUITAR_PROBE_FIRE_PROB moved to the settings file beside this script.)
# Peak at/above this = treat guitar-family as present (enough to reinstate a
# whole-mix tag that top-k truncation dropped, and to block an absence claim).
# (GUITAR_PROBE_PRESENT_PEAK moved to the settings file beside this script.)
# ...or a lower peak that persists across at least this share of windows.
# (GUITAR_PROBE_PRESENT_COVERAGE moved to the settings file beside this script.)
# (GUITAR_PROBE_PRESENT_COVERAGE_PEAK moved to the settings file beside this script.)
# Confident absence requires the peak to sit below this across the whole track.
# (GUITAR_PROBE_ABSENT_PEAK moved to the settings file beside this script.)
# (GUITAR_PROBE_ABSENT_COVERAGE moved to the settings file beside this script.)
# Demucs guitar-stem cross-check: htdemucs_6s has a dedicated guitar stem, so
# a genuinely guitar-driven track should leave real energy and real transcribed
# notes there even when the tagger misses it. These are the bars above which
# the stem counts as corroborating evidence against an absence claim.
# (GUITAR_STEM_ACTIVE_MIN_RMS moved to the settings file beside this script.)
# (GUITAR_STEM_ACTIVE_MIN_NOTES moved to the settings file beside this script.)

# --- Cross-stem convergence -------------------------------------------------
# Demucs splits one instrument across several stems (a loud rhythm guitar
# leaves energy in "guitar", "bass" and "other"), so the same instrument being
# tagged independently on multiple stems, each with sustained coverage, is real
# corroboration -- arguably better than the whole-mix pass, since each stem is
# isolated and peak-normalised rather than buried in a limited master. This
# counted for nothing before: three stems could agree on guitar at 64-82% and
# all three would still be marked "stem-only, whole-mix did not agree".
# (CROSS_STEM_CONVERGENCE_MIN_STEMS moved to the settings file beside this script.)
# (CROSS_STEM_CONVERGENCE_MIN_PROB moved to the settings file beside this script.)
# (CROSS_STEM_CONVERGENCE_MIN_SUSTAINED_STEMS moved to the settings file beside this script.)

# Cross-stem agreement is NOT fully independent evidence. All six stems come
# from one separation of one file, so a single timbral confusion propagates
# into every stem it bleeds into, and two stems "agreeing" can be one source
# counted twice. Two guards below.
#
# 1) ANCHORING. Demucs has a dedicated stem for some families, so a guitar tag
#    on the guitar stem, or a bass tag on the bass stem, is that stem's own
#    source rather than bleed. An agreement containing such a native hit is
#    anchored and trustworthy; an agreement made only of non-native stems
#    (e.g. organ found on "other" and "guitar") has no anchor and may be a
#    single mis-identified source appearing twice.
_STEM_NATIVE_FAMILIES = {
    "guitar": ("guitar (type uncertain)", "electric guitar", "acoustic guitar",
               "slide/steel guitar", "strummed chords (guitar family)",
               "guitar technique (tapping)"),
    "bass": ("bass guitar", "upright bass"),
    "piano": ("piano", "electric piano"),
    "vocals": ("choir/vocal ensemble", "chant", "humming", "whistling"),
    "drums": ("drum kit", "drums", "drum machine", "snare", "kick", "hi-hat",
              "cymbal", "percussion (unspecified)", "tambourine", "rimshot",
              "drum roll", "cowbell"),
}


def _instrument_activity_spans(onsets, gap_s, min_len_s):
    """Continuous playing spans for one instrument, with bursts discarded.

    A span breaks after `gap_s` of silence. Spans shorter than `min_len_s` are
    dropped: a two-note blip is not an instrument entering the arrangement, and
    including them was the single largest source of false change points.
    """
    o = sorted(float(t) for t in (onsets or []))
    if not o:
        return []
    out = []
    start = prev = o[0]
    for t in o[1:]:
        if t - prev > gap_s:
            out.append((start, prev))
            start = t
        prev = t
    out.append((start, prev))
    return [(a, b) for a, b in out if (b - a) >= min_len_s]


def _stem_prominence_ranks(text):
    """{stem: (rank, band)} from the STEM ACTIVITY / PROMINENCE block, or {}."""
    t = str(text or "")
    i = t.find("STEM ACTIVITY / PROMINENCE (ranked")
    if i < 0:
        return {}
    e = t.find("\n\n", i)
    ranks = {}
    for m in re.finditer(r"(?m)^(\d+)\) (\w+): ([a-z /]+?) \(RMS", t[i:e if e >= 0 else len(t)]):
        ranks.setdefault(m.group(2), (int(m.group(1)), m.group(3).strip()))
    return ranks


def _guitar_stem_reading(text):
    """How the separated guitar stem sits in the measured balance:
    (leads, detail). leads is True when the guitar stem is an active part
    ranked above the drums or the bass stem, False when it is quiet or ranked
    below both, None when no stem ranking is in the notes."""
    ranks = _stem_prominence_ranks(text)
    if not ranks:
        return None, ""
    n = len(ranks)
    g = ranks.get("guitar")
    if not g:
        return False, "there is no separated guitar part in the stem balance"
    if g[1] in ("silent/negligible", "low / background"):
        return False, f"the separated guitar stem is {g[1]} (#{g[0]} of {n} stems by prominence)"
    rhythm = [(s, ranks[s][0]) for s in ("drums", "bass") if s in ranks]
    if not rhythm:
        return None, ""
    if any(g[0] < r for _s, r in rhythm):
        return True, f"the separated guitar stem is {g[1]} and ranks #{g[0]} of {n} stems by prominence"
    above = [s for s, (r, _b) in sorted(ranks.items(), key=lambda kv: kv[1][0]) if r < g[0]]
    above_s = (", ".join(above[:-1]) + " and " + above[-1]) if len(above) > 1 else "".join(above)
    return False, (f"the separated guitar stem ranks #{g[0]} of {n} stems by prominence, "
                   f"below the {above_s}")


# dev60: see INSTRUMENT_REED_TAG_CHECK.
_REED_TAGS = (("accordion", "an accordion"), ("harmonium", "a harmonium"), ("bandoneon", "a bandoneon"),
              ("concertina", "a concertina"), ("harmonica", "a harmonica"), ("melodica", "a melodica"))
_REED_PLAUSIBLE_GENRES = re.compile(
    r"\b(?:folk|country|blues|bluegrass|americana|cajun|zydeco|tango|polka|world|celtic|irish|"
    r"scottish|latin|cumbia|norte\w*|tex-?mex|vallenato|musette|chanson|klezmer|forr[oó]|"
    r"tejano|conjunto|sea shanty|shanty|gypsy|balkan|french|italian|ranchera|singer-songwriter|"
    r"roots|skiffle|harmonica)\b", re.IGNORECASE)
_INSTR_CHECK_HEAD = "INSTRUMENT CHECK (read before the instrument tags below):"


def _listening_model_spans(text):
    """dev60: the listening model's own words in the notes -- its opening
    report and its era, vocal and confirmation passes -- not the tagger
    lines or measurements."""
    t = str(text or "")
    # Check lines added on top of the notes (GENRE CHECK, ARRANGEMENT CHECK,
    # ...) are not the listening model's words.
    while re.match(r"[A-Z][A-Z /-]+ CHECK \(", t) or t.startswith("CHECKS (corrections"):
        k = t.find("\n\n")
        if k < 0:
            break
        t = t[k + 2:]
    cut = len(t)
    for mk in ("\n\nLYRICS", "[Independent signal-processing report", "\nRECOMMENDED TEMPO FOR DISCUSSION",
               "\n\nVOCAL DECISION AUDIT"):
        k = t.find(mk)
        if 0 <= k < cut:
            cut = k
    spans = [t[:cut]]
    for head in ("11. ERA / RELEASE PERIOD", "VOCAL / SINGER PROFILE", "VOCAL CONFIRMATION PASS"):
        k = t.find(head)
        if k >= 0:
            e = t.find("\n\n", k + len(head) + 2)
            spans.append(t[k:e if e >= 0 else len(t)])
    return "\n".join(spans)


def _reed_tag_check_note(text):
    """dev60: an INSTRUMENT CHECK line for unconfirmed free-reed tags, or ""."""
    if not INSTRUMENT_REED_TAG_CHECK:
        return ""
    t = str(text or "")
    tag_lines = [ln for ln in t.splitlines()
                 if "instrument tag (" in ln or re.match(r"\s+[a-z][\w /()-]* \((?:\w+) stem\):", ln)]
    if not tag_lines:
        return ""
    tagged = "\n".join(tag_lines).lower()
    own = _listening_model_spans(t).lower()
    genre_txt = " ".join(m.group(1) for m in _RECOMMENDED_GENRE_RE.finditer(t))
    genre_txt += " " + " ".join(extract_genre_ranked_labels(t, max_n=5))
    if _REED_PLAUSIBLE_GENRES.search(genre_txt):
        return ""
    found = []
    for word, art in _REED_TAGS:
        if re.search(r"\b" + word + r"\b", tagged) and not re.search(r"\b" + word, own):
            found.append((word, art))
    if not found:
        return ""
    strong = []
    if INSTRUMENT_REED_STRONG_LEAD:
        for w, a in found:
            info = _strong_reed_reading(t, w) if w == "harmonica" else None
            if info:
                strong.append((w, a, info))
        found = [(w, a) for w, a in found if w not in {x[0] for x in strong}]
    lead_s = ""
    if strong:
        w, _a, (pct, n_on, n_all, stem) = strong[0]
        lead_s = (f"the instrument tagger hears a steady reedy lead{' on the ' + stem + ' stem' if stem else ''}: "
                  f"'{w}' at {pct}% in {n_on} of {n_all} windows -- too strong and too sustained to dismiss, "
                  "although the listening model does not mention it. Treat it as a real lead or riff "
                  "instrument with a reedy tone: a harmonica, or a horn such as a saxophone or trumpet (that "
                  "classifier also calls a saxophone 'harmonica'). Describe it by that sound and role, as a "
                  "harmonica-like or brassy lead, without asserting which; name it exactly only if the user "
                  "confirms it. A lead line the descriptions give to another instrument may be this part.")
        if not found:
            return f"{_INSTR_CHECK_HEAD} {lead_s}"
    stems = sorted(set(re.findall(r"(?m)^\s+(?:" + "|".join(w for w, _a in found)
                                  + r")\b[^\n(]*\((\w+) stem\):", t)))
    where = (f"heard on the {' and '.join(stems)} stem{'s' if len(stems) > 1 else ''}" if stems
             else "from the separated stems")
    names = [f"'{w}'" for w, _a in found]
    names_s = names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]
    arts = [a for _w, a in found]
    arts_s = arts[0] if len(arts) == 1 else ", ".join(arts[:-1]) + " or " + arts[-1]
    likely = []
    for pat, label in ((r"\bsax(?:ophone)?s?\b", "saxophone"), (r"\bsynth(?:esizer)?s?\b|\bsynth[- ]", "synths"),
                       (r"\b(?:pads?|synth pads?)\b", "pads"), (r"\borgans?\b", "organ"),
                       (r"\bkeyboards?\b|\bkeys\b", "keyboards"), (r"\bstrings?\b", "strings"),
                       (r"\bhorns?\b|\bbrass\b", "horns")):
        if re.search(pat, own):
            likely.append(label)
    likely_s = ""
    if likely:
        likely_s = (" The listening model itself names " + (likely[0] if len(likely) == 1 else
                    ", ".join(likely[:-1]) + " and " + likely[-1]) + " on this track -- the likelier source.")
    describe = "(a reedy synth pad, sustained keyboard chords, a sax or lead line)"
    poly = _other_stem_polyphony(t) if INSTRUMENT_REED_STRONG_LEAD and stems == ["other"] else None
    if poly is not None and poly >= REED_PAD_MIN_POLYPHONY and not re.search(
            r"\bsax(?:ophone)?s?\b|\bhorns?\b|\bbrass\b|\btrumpets?\b|\btrombones?\b", own):
        describe = (f"(most likely a synth pad or sustained keyboard chords: the other stem it was heard on "
                    f"plays chords of about {poly:g} notes at once)")
    body = (
        f"the instrument tagger's {names_s} label{'s' if len(found) > 1 else ''} "
        f"({where}) {'are' if len(found) > 1 else 'is'} not confirmed by anything else: the listening "
        f"model never mentions {arts_s}, and that classifier routinely gives these names to sustained, "
        "reedy synth pads, organ-like synths and reed or horn leads such as saxophone." + likely_s
        + f" Do not say the track has {arts_s}; describe that part by its sound and role {describe}. "
        "Name the instrument only if the user confirms it."
    )
    if lead_s:
        return f"{_INSTR_CHECK_HEAD} {lead_s} Separately, {body}"
    return f"{_INSTR_CHECK_HEAD} {body}"


def _strong_reed_reading(text, word):
    """(percent, windows on, windows in all, stem) for a strong, steady,
    not-weak tagger reading of `word` in the per-stem tag lines, else None."""
    for blk in re.finditer(r"(?m)^\[(\w+) stem\]\n", text):
        e = text.find("\n\n", blk.end())
        seg = text[blk.end():e if e >= 0 else len(text)]
        for ln in seg.splitlines():
            if not ln.lstrip().startswith("instrument tag ("):
                continue
            for m in re.finditer(r"\b" + word + r" \((\d+)%[^)]*\) \[([^\]]*)\]", ln):
                if m.group(2).lstrip().startswith("weak"):
                    continue
                w = re.search(r"(\d+)/(\d+) windows", m.group(2))
                if not w:
                    continue
                pct, n_on, n_all = int(m.group(1)), int(w.group(1)), int(w.group(2))
                if (pct >= REED_STRONG_LEAD_MIN_PCT and n_all
                        and n_on >= REED_STRONG_LEAD_MIN_WINDOW_SHARE * n_all):
                    return pct, n_on, n_all, blk.group(1).lower()
    return None


def _other_stem_polyphony(text):
    t = str(text or "")
    j = t.find("[OTHER stem]")
    if j < 0:
        return None
    k = t.find("\n\n", j)
    m = re.search(r"average simultaneous notes: (\d+(?:\.\d+)?)", t[j:k if k >= 0 else len(t)])
    return float(m.group(1)) if m else None


_SEMI_ACOUSTIC_ITEM = ("possible second guitar (clean, lightly amplified or semi-acoustic in tone; the "
                       "guitar stem reads acoustic throughout and nothing reads a distorted electric guitar)")
_GUITAR_CHECK_HEAD = "GUITAR CHECK (measured from the separated stems; read before the descriptions below):"
_ELECTRIC_GUITAR_RE = re.compile(r"(?i)\belectric[ -]?guitars?\b")
_ELECTRIC_NEGATED_RE = re.compile(r"(?i)\b(?:no|without|absence of|not)\s+(?:\w+\s+){0,2}$")


def _acoustic_only_guitar(text):
    """True when the tagger reads the guitar stem as acoustic guitar in every
    section and no section, stem or whole-mix tag reads an electric guitar."""
    t = str(text or "")
    if not re.search(r"(?m)^\s+acoustic guitar \(guitar stem\): throughout\b", t):
        return False
    if re.search(r"(?m)^\s+electric guitar\b[^\n]*\(\w+ stem\):", t):
        return False
    if re.search(r"(?m)^instrument tag \([^\n]*\belectric guitar \(\d+%", t):
        return False
    i = t.find("WHOLE-MIX INSTRUMENT TAGS")
    if i >= 0:
        e = t.find("\n\n", i)
        if "electric guitar" in t[i:e if e >= 0 else len(t)].split("\n", 1)[-1]:
            return False
    return True


def _asserts_electric_guitar(text):
    s = str(text or "")
    return any(not _ELECTRIC_NEGATED_RE.search(s[max(0, m.start() - 40):m.start()])
               for m in _ELECTRIC_GUITAR_RE.finditer(s))


def _acoustic_guitar_check_note(text):
    """A GUITAR CHECK line when the listening model names an electric guitar
    on a track whose guitar stem reads acoustic throughout, else ""."""
    if not ACOUSTIC_GUITAR_CHECK:
        return ""
    t = str(text or "")
    if not _acoustic_only_guitar(t) or not _asserts_electric_guitar(_listening_model_spans(t)):
        return ""
    return (
        f"{_GUITAR_CHECK_HEAD} the instrument tagger reads the separated guitar stem as acoustic guitar "
        "in every section, and no section, stem or whole-mix tag reads an electric guitar. Any "
        "electric guitar the descriptions mention is at most a clean, lightly amplified or "
        "semi-acoustic-sounding guitar alongside the acoustic one: describe it that way, never as "
        "a distorted, overdriven or rock electric guitar, and keep the acoustic guitar as the main "
        "guitar sound."
    )


_INSTR_SPAN_ITEM_RE = re.compile(r"^\s*(?P<name>[^():;\n]+?)\s*\(\s*\d+(?:\.\d+)?\s*s\s*[\u2010-\u2015\-]\s*\d+(?:\.\d+)?\s*s\s*\)"
                                 r"\s*(?::\s*(?P<desc>.*))?$")
_INSTR_JOINED = {"electricguitar": "electric guitar", "acousticguitar": "acoustic guitar",
                 "electricbass": "electric bass", "basguitar": "bass guitar", "bassguitar": "bass guitar"}


def _instrumentation_collapse_spans(value):
    """See INSTRUMENTATION_COLLAPSE_SPANS. A list in which most items carry a
    time span, with each instrument once; anything else unchanged."""
    v = str(value or "")
    if not INSTRUMENTATION_COLLAPSE_SPANS or "\n" in v:
        return value
    items = [x for x in v.split(";") if x.strip()]
    ms = [_INSTR_SPAN_ITEM_RE.match(x) for x in items]
    if sum(1 for m in ms if m) < 2 or sum(1 for m in ms if m) * 2 < len(items):
        return value
    order, desc, extra = [], {}, []
    for x, m in zip(items, ms):
        if m:
            nm = m.group("name").strip()
            k = re.sub(r"\s+", "", nm.lower())
            if k not in desc:
                order.append((k, _INSTR_JOINED.get(k, nm)))
                desc[k] = (m.group("desc") or "").strip()
            elif not desc[k] and m.group("desc"):
                desc[k] = m.group("desc").strip()
        else:
            k0 = re.sub(r"[\s(]+", "", re.split(r"\(", x, 1)[0].lower())
            if k0 and (k0 in desc or (len(k0) >= 3 and any(k.startswith(k0) for k in desc))):
                continue            # a repeat cut off mid-item ("drums (25.", "synthes")
            extra.append(x.strip())
    out = [(f"{nm}: {desc[k]}" if desc[k] else nm) for k, nm in order] + extra
    return "; ".join(out)


def _saved_analysis_with_instrument_check(analysis):
    """dev60: the analysis with its INSTRUMENT CHECK line at the top, as the
    writer is shown it, or unchanged when there is none."""
    t = str(analysis or "")
    if not INSTRUMENT_REED_TAG_CHECK or not t or _INSTR_CHECK_HEAD in t:
        return analysis
    try:
        note = _reed_tag_check_note(t)
        out = (note + "\n\n" + t) if note else t
        gc = _acoustic_guitar_check_note(t) if _GUITAR_CHECK_HEAD not in t else ""
        if gc:
            out = gc + "\n\n" + out
        return out if (note or gc) else analysis
    except Exception:
        return analysis


_INSTRUMENT_TAGGER = None
_INSTRUMENT_TAGGER_LABELS = None
INSTRUMENT_TAGGER_IMPORT_ERROR = ""

# Raw per-window AudioSet probability matrices, keyed by file + window params
# (see _panns_cache_key). Lets the whole-mix instrument, genre, mood and
# production-effect passes share one set of network forward passes instead of
# each re-decoding and re-inferring the same audio.
_PANNS_WINDOW_CACHE = {}
_PANNS_WINDOW_CACHE_ORDER = []
# Separate insertion order for the (tiny) persistence-stats table, so it can
# outlive the probability matrices it was derived from.
_PANNS_TAG_STATS_ORDER = []
# Per-label persistence stats (hits / coverage / median prob) recorded by the
# most recent tagging pass. Held here rather than widened into the returned
# tuples so existing 3-tuple unpacking is untouched.
_PANNS_TAG_STATS = {}
# working-copy path (abs) -> original file to decode instead, for tagging.
_PANNS_SOURCE_OVERRIDE = {}


def register_panns_source(working_path, original_path):
    """Tag `original_path`'s audio whenever `working_path` is passed to the
    PANNs passes (cache keys and stats lookups keep using working_path)."""
    try:
        if (working_path and original_path
                and not str(original_path).startswith(("http://", "https://"))
                and os.path.exists(str(original_path))):
            _PANNS_SOURCE_OVERRIDE[os.path.abspath(str(working_path))] = str(original_path)
    except Exception:
        pass

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
    # Guitar performance techniques -- AudioSet scores these separately from
    # the instrument itself, and they are strong positive evidence that a real
    # played guitar is present rather than a synth patch that merely sounds
    # guitar-ish, which is exactly the confusion the guitar threshold exists
    # to guard against.
    "strum": "strummed chords (guitar family)",
    "tapping (guitar technique)": "guitar technique (tapping)",
    # Kit detail the drum map was missing.
    "rimshot": "rimshot",
    "drum roll": "drum roll",
    "cowbell": "cowbell",
    # Vocal-as-arrangement-texture only. Deliberately excludes singing /
    # male singing / female singing / child singing: lead-vocal identity,
    # gender and age are decided by the dedicated vocal pipeline (F0 analysis
    # + vocal audit + identity resolution), and duplicating those judgements
    # from a coarse AudioSet class would create a second, weaker opinion that
    # could contradict it. These four are arrangement details that pipeline
    # does not cover.
    "choir": "choir/vocal ensemble",
    "chant": "chant",
    "humming": "humming",
    "whistling": "whistling",
}


_PANNS_LOCK = threading.RLock()   # dev55: one tagger, used from background work too


def _get_instrument_tagger():
    """Lazily load a pretrained PANNs (Pretrained Audio Neural Networks)
    AudioSet tagger. Optional: if panns_inference isn't installed, tagging is
    skipped everywhere it's used and the rest of the pipeline is unaffected.
    Returns None if unavailable."""
    with _PANNS_LOCK:
        return _get_instrument_tagger_locked()


def _get_instrument_tagger_locked():
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


def _panns_cache_key__pre6(audio_path, window_seconds, hop_seconds, max_windows, normalize):
    """Identity for a set of raw per-window PANNs probabilities: the file
    (path + size + mtime, so a re-encoded/replaced temp file at the same path
    never returns stale rows) plus the windowing parameters."""
    try:
        st = os.stat(audio_path)
        sig = (int(st.st_size), int(st.st_mtime_ns))
    except Exception:
        sig = (None, None)
    return (
        os.path.abspath(str(audio_path)),
        sig[0],
        sig[1],
        round(float(window_seconds or 0.0), 3),
        round(float(hop_seconds or 0.0), 3),
        int(max_windows or 0),
        bool(normalize),
    )


def _panns_cache_key__07(audio_path, window_seconds, hop_seconds, max_windows, normalize):
    """Identity for a set of raw per-window PANNs probabilities: the file
    (path + size + mtime, so a re-encoded/replaced temp file at the same path
    never returns stale rows) plus the windowing parameters."""
    try:
        st = os.stat(audio_path)
        sig = (int(st.st_size), int(st.st_mtime))
    except Exception:
        sig = (None, None)
    return (
        os.path.abspath(str(audio_path)),
        sig[0],
        sig[1],
        round(float(window_seconds or 0.0), 3),
        round(float(hop_seconds or 0.0), 3),
        int(max_windows or 0),
        bool(normalize),
    )


def _panns_cache_key(*args, **kwargs):
    """AUDIO_CACHE_FIXES picks the 0.7.1-pre6 version; False gives 0.7's."""
    return (_panns_cache_key__pre6 if AUDIO_CACHE_FIXES else _panns_cache_key__07)(*args, **kwargs)


def _panns_window_starts(n_samples, sr, window_seconds, hop_seconds, max_windows):
    """Window start/end sample indices covering the WHOLE clip.

    Previously windows marched left-to-right at a fixed hop and stopped dead
    once max_windows was reached, so everything after roughly
    (max_windows * hop) seconds was never analysed -- 43% of a 6-minute track
    went unseen, making late-entering instruments undetectable. When the
    hop-spaced layout would overrun the budget, starts are spread evenly
    across the entire file instead, trading a slightly larger effective hop
    for complete end-to-end coverage.

    Returns [(start_sample, end_sample, start_seconds), ...].
    """
    if n_samples <= 0 or sr <= 0:
        return []

    duration = n_samples / float(sr)
    if not window_seconds or duration <= window_seconds:
        return [(0, n_samples, 0.0)]

    win_n = int(window_seconds * sr)
    hop_n = max(1, int((hop_seconds or window_seconds) * sr))
    max_windows = int(max_windows or 0) or 1
    last_start = max(0, n_samples - win_n)

    needed = int(last_start // hop_n) + 1
    if INSTRUMENT_TAG_EVEN_COVERAGE and needed > max_windows:
        # Budget-constrained: spread evenly over the whole file.
        starts = [
            int(round(last_start * i / float(max_windows - 1)))
            for i in range(max_windows)
        ] if max_windows > 1 else [0]
    else:
        starts = list(range(0, last_start + 1, hop_n))
        if starts and starts[-1] < last_start:
            starts.append(last_start)
        starts = starts[:max_windows]

    windows = []
    seen = set()
    for s in starts:
        s = max(0, min(int(s), last_start))
        if s in seen:
            continue
        seen.add(s)
        e = min(s + win_n, n_samples)
        if e - s >= sr * 0.5:  # skip slivers under 0.5s
            windows.append((s, e, s / float(sr)))
    return windows


def _panns_window_probs(
    audio_path,
    window_seconds,
    hop_seconds,
    max_windows,
    normalize,
):
    """Raw per-window AudioSet probabilities for one file, cached.

    Returns (probs, times) where probs is an (n_windows, n_labels) float32
    array and times is a list of window start seconds -- or (None, None).

    The whole-mix instrument, genre and mood passes all tag the SAME audio
    over the SAME windows and differ only in which labels they keep, so they
    used to decode the file and run the network three separate times. Caching
    the raw probability matrix collapses that to one sweep; later passes are
    a dictionary lookup.
    """
    key = _panns_cache_key(audio_path, window_seconds, hop_seconds, max_windows, normalize)
    cached = _PANNS_WINDOW_CACHE.get(key)
    if cached is not None:
        return cached

    tagger = _get_instrument_tagger()
    if tagger is None:
        return None, None

    # The pipeline's working copy is 22.05 kHz mono, which leaves the top of
    # PANNs' 32 kHz / 14 kHz-mel input empty. When the caller registered the
    # untouched original for that working path, decode the original instead.
    # The cache key (and therefore every stats lookup) still uses audio_path.
    load_path = _PANNS_SOURCE_OVERRIDE.get(os.path.abspath(str(audio_path)), audio_path)
    try:
        y, sr = librosa.load(load_path, sr=32000, mono=True)  # PANNs models expect 32kHz
    except Exception:
        try:
            y, sr = librosa.load(audio_path, sr=32000, mono=True)
        except Exception:
            return None, None
    if y is None or len(y) < sr * 0.5:
        return None, None

    windows = _panns_window_starts(len(y), sr, window_seconds, hop_seconds, max_windows)
    if not windows:
        return None, None

    # Each window is peak-normalised before tagging, which is right for a quiet
    # but real part -- and wrong for a window holding nothing but separation
    # residue: blown up to full scale, that residue is exactly what produced
    # "didgeridoo" on a bass stem in the opening seconds, "gong" in a final
    # section where every stem was silent, and "flute" on an 'other' stem that
    # was quiet/absent. Windows far below this file's loudest window (or below
    # an absolute floor) are recorded as all-zero probabilities instead.
    levels = []
    for start, end, _t0 in windows:
        seg = y[start:end]
        levels.append(float(np.sqrt(np.mean(np.square(seg.astype(np.float64))))) if seg.size else 0.0)
    loudest = max(levels) if levels else 0.0
    quiet_floor = max(float(PANNS_QUIET_WINDOW_ABS_RMS),
                      loudest * 10.0 ** (-float(PANNS_QUIET_WINDOW_REL_DB) / 20.0))

    rows = []
    times = []
    n_labels = len(_INSTRUMENT_TAGGER_LABELS or [])
    # dev55: windows are tagged in batches (PANNS_BATCH_WINDOWS); the rows,
    # their order and the per-window failure handling are as before.
    out_rows = [None] * len(windows)
    pending = []
    for wi, ((start, end, t0), lvl) in enumerate(zip(windows, levels)):
        chunk = y[start:end]
        if lvl < quiet_floor and n_labels:
            out_rows[wi] = np.zeros(n_labels, dtype=np.float32)
            continue
        if normalize:
            chunk = _peak_normalize(chunk, INSTRUMENT_TAG_NORMALIZE_PEAK)
        pending.append((wi, chunk))
    bsz = max(1, int(PANNS_BATCH_WINDOWS or 1))
    k = 0
    while k < len(pending):
        group = [pending[k]]
        while (len(group) < bsz and k + len(group) < len(pending)
               and len(pending[k + len(group)][1]) == len(group[0][1])):
            group.append(pending[k + len(group)])
        k += len(group)
        done_batch = False
        if len(group) > 1:
            try:
                with _PANNS_LOCK, quiet_stdout():
                    clipwise_output, _ = tagger.inference(np.stack([c for _wi, c in group], axis=0))
                for j, (wi, _c) in enumerate(group):
                    out_rows[wi] = np.asarray(clipwise_output[j], dtype=np.float32)
                done_batch = True
            except Exception:
                done_batch = False
        if not done_batch:
            for wi, chunk in group:
                try:
                    with _PANNS_LOCK, quiet_stdout():
                        clipwise_output, _ = tagger.inference(chunk[None, :])
                except Exception:
                    continue
                out_rows[wi] = np.asarray(clipwise_output[0], dtype=np.float32)
    for (start, end, t0), row in zip(windows, out_rows):
        if row is None:
            continue
        rows.append(row)
        times.append(round(float(t0), 1))

    if not rows:
        return None, None

    probs = np.vstack(rows)
    try:
        _PANNS_WINDOW_CACHE[key] = (probs, times)
        _PANNS_WINDOW_CACHE_ORDER.append(key)
        # Bound memory: a few (n_windows x 527) float32 matrices only, but the
        # batch path is deliberately frugal about RSS. Persistence stats are
        # NOT dropped here -- they are tiny, they are still needed after the
        # matrix is gone (the whole-mix report is built after all four stem
        # passes have pushed the mix out of this cache), and discarding them
        # was losing the window counts from every whole-mix tag line.
        while len(_PANNS_WINDOW_CACHE_ORDER) > max(1, int(PANNS_WINDOW_CACHE_SIZE)):
            old = _PANNS_WINDOW_CACHE_ORDER.pop(0)
            _PANNS_WINDOW_CACHE.pop(old, None)
    except Exception:
        pass
    return probs, times


def _panns_label_groups(label_map):
    """Map each displayed label to the raw AudioSet column indices that feed
    it (several raw labels can share one displayed label, e.g. Organ /
    Electronic organ / Hammond organ). Computed once per label map."""
    groups = {}
    if not _INSTRUMENT_TAGGER_LABELS:
        return groups
    for idx, raw in enumerate(_INSTRUMENT_TAGGER_LABELS):
        mapped = label_map.get(raw)
        if mapped is None:
            continue
        groups.setdefault(mapped, []).append(idx)
    return groups


def _panns_tag_stats(audio_path, label, window_seconds=None, hop_seconds=None,
                     max_windows=None, normalize=None):
    """Persistence statistics for one displayed label on one file, recorded by
    the most recent _panns_tag_windowed() pass. Returns a dict with keys
    hits / n_windows / coverage / peak_prob / peak_time / median_prob, or None.

    Kept in this side table rather than widened into the returned tuples so
    every existing `for label, prob, t in tags` unpack keeps working exactly
    as before.
    """
    if window_seconds is None:
        window_seconds = INSTRUMENT_TAG_WINDOW_SECONDS
    if hop_seconds is None:
        hop_seconds = INSTRUMENT_TAG_HOP_SECONDS
    if max_windows is None:
        max_windows = INSTRUMENT_TAG_MAX_WINDOWS
    if normalize is None:
        normalize = INSTRUMENT_TAG_NORMALIZE
    try:
        key = _panns_cache_key(audio_path, window_seconds, hop_seconds, max_windows, normalize)
    except Exception:
        return None
    return (_PANNS_TAG_STATS.get(key) or {}).get(label)


def _tag_coverage_note(stats):
    """Short human-readable persistence note for a tag line, e.g.
    'sustained: 24/56 windows' or 'isolated: 1/56 windows'. Tells the writer
    the difference between an instrument that plays throughout and a single
    momentary spike, which the peak-only score alone cannot express."""
    if not stats:
        return ""
    hits = int(stats.get("hits") or 0)
    n = int(stats.get("n_windows") or 0)
    if hits <= 0 or n <= 0:
        return ""
    cov = hits / float(n)
    if hits == 1:
        word = "isolated"
    elif cov >= INSTRUMENT_TAG_SUSTAINED_COVERAGE:
        word = "sustained"
    else:
        word = "intermittent"
    return f"{word}: {hits}/{n} windows"


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

    probs, times = _panns_window_probs(
        audio_path, window_seconds, hop_seconds, max_windows, normalize
    )
    if probs is None or not times:
        return []

    groups = _panns_label_groups(label_map)
    if not groups:
        return []

    n_windows = int(probs.shape[0])
    stats_key = _panns_cache_key(audio_path, window_seconds, hop_seconds, max_windows, normalize)
    if stats_key not in _PANNS_TAG_STATS:
        _PANNS_TAG_STATS_ORDER.append(stats_key)
        while len(_PANNS_TAG_STATS_ORDER) > max(1, int(PANNS_TAG_STATS_CACHE_SIZE)):
            _PANNS_TAG_STATS.pop(_PANNS_TAG_STATS_ORDER.pop(0), None)
    stats_for_file = _PANNS_TAG_STATS.setdefault(stats_key, {})

    # best[label] = (effective_probability, peak_time_seconds)
    # A peak remains useful for transient instruments, but track-level claims
    # now require either persistence or a substantially stronger one-window hit.
    best = {}
    for mapped, idxs in groups.items():
        # Per window, a displayed label scores the strongest of the raw
        # AudioSet columns that feed it.
        per_window = probs[:, idxs].max(axis=1).astype(float)
        peak_i = int(np.argmax(per_window))
        peak_prob = float(per_window[peak_i])
        if peak_prob < min_prob:
            continue

        fired = per_window >= float(min_prob)
        hits = int(np.count_nonzero(fired))
        median_prob = float(np.median(per_window[fired])) if hits else 0.0
        coverage = (hits / float(n_windows)) if n_windows else 0.0
        persistent = hits >= int(INSTRUMENT_TAG_MIN_WINDOW_HITS)
        single_strong = (hits == 1 and peak_prob >= float(INSTRUMENT_TAG_SINGLE_WINDOW_STRONG_PROB))
        sustained = coverage >= float(INSTRUMENT_TAG_SUSTAINED_COVERAGE)
        # Blend peak + persistent evidence. Sustained material keeps good recall;
        # isolated material must clear the explicit strong-single threshold.
        if not (persistent or single_strong or sustained):
            continue
        effective_prob = max(
            peak_prob * 0.65 + median_prob * 0.35,
            peak_prob if single_strong else 0.0,
        )
        best[mapped] = (effective_prob, times[peak_i] if peak_i < len(times) else 0.0)
        stats_for_file[mapped] = {
            "hits": hits,
            "n_windows": n_windows,
            "coverage": coverage,
            "peak_prob": peak_prob,
            "peak_time": times[peak_i] if peak_i < len(times) else 0.0,
            "median_prob": median_prob,
            "mean_prob": float(np.mean(per_window)),
        }

    if not best:
        return []

    ranked = sorted(best.items(), key=lambda kv: kv[1][0], reverse=True)[:top_k]
    return [(label, prob, t) for label, (prob, t) in ranked]


def _is_guitar_family_label(label):
    lab = (label or "").strip().lower()
    if lab in INSTRUMENT_TAG_GUITAR_LABELS:
        return True
    return "guitar" in lab and "bass" not in lab


def _guitar_family_probe(audio_path, window_seconds=None, hop_seconds=None,
                         max_windows=None, normalize=None):
    """Measure guitar-family energy straight from the raw per-window PANNs
    probabilities, with no top-k truncation and no reporting thresholds.

    Why this exists separately from the tag pipeline: _panns_tag_windowed()
    ranks every displayed label by peak probability and keeps only the top
    `top_k`. The whole-mix pass keeps 6, and the instrument label map contains
    a large drum family (drum kit, drums, snare, kick, hi-hat, cymbal) plus
    bass and synth, all of which score high on a loud commercial pop-rock
    master. A prominent electric guitar can therefore be ranked seventh and
    silently discarded before the guitar-family threshold is ever applied --
    after which the pipeline concluded "no guitar" and said so in the notes.

    This probe answers a narrower question the tag list cannot: how much
    guitar-family evidence is actually in the audio, including values far
    below any reporting bar, so an ABSENCE claim can be made on positive
    evidence rather than on a failure to clear a strict threshold.

    Reads the same cached probability matrix the tagging passes populate, so
    when tagging has already run this costs one dictionary lookup and a numpy
    reduction -- no extra decode, no extra inference.

    Returns a dict:
        {"peak": float, "coverage": float, "hits": int, "n_windows": int,
         "peak_time": float, "best_label": str, "per_label": {label: peak},
         "present": bool, "absent": bool}
    or None when the tagger/audio is unavailable.
    """
    if not GUITAR_PROBE_ENABLED:
        return None
    if not audio_path or str(audio_path).startswith(("http://", "https://")):
        return None
    if window_seconds is None:
        window_seconds = INSTRUMENT_TAG_WINDOW_SECONDS
    if hop_seconds is None:
        hop_seconds = INSTRUMENT_TAG_HOP_SECONDS
    if max_windows is None:
        max_windows = INSTRUMENT_TAG_MAX_WINDOWS
    if normalize is None:
        normalize = INSTRUMENT_TAG_NORMALIZE

    try:
        probs, times = _panns_window_probs(
            audio_path, window_seconds, hop_seconds, max_windows, normalize
        )
    except Exception:
        return None
    if probs is None or not times:
        return None

    try:
        groups = _panns_label_groups(_INSTRUMENT_LABEL_MAP)
    except Exception:
        return None
    if not groups:
        return None

    # Every displayed label in the guitar family, including the performance
    # techniques ("strum", "tapping") -- AudioSet scores those separately and
    # they are strong evidence that a real played guitar is present rather
    # than a synth patch that merely sounds guitar-ish.
    idxs = []
    per_label = {}
    for mapped, cols in groups.items():
        if not _is_guitar_family_label(mapped):
            continue
        idxs.extend(cols)
        try:
            per_label[mapped] = float(probs[:, cols].max())
        except Exception:
            continue
    if not idxs:
        return None

    try:
        per_window = probs[:, sorted(set(idxs))].max(axis=1).astype(float)
    except Exception:
        return None
    if per_window.size == 0:
        return None

    n_windows = int(per_window.shape[0])
    peak_i = int(np.argmax(per_window))
    peak = float(per_window[peak_i])
    hits = int(np.count_nonzero(per_window >= float(GUITAR_PROBE_FIRE_PROB)))
    coverage = (hits / float(n_windows)) if n_windows else 0.0
    best_label = ""
    if per_label:
        best_label = max(per_label.items(), key=lambda kv: kv[1])[0]

    present = (
        peak >= float(GUITAR_PROBE_PRESENT_PEAK)
        or (
            coverage >= float(GUITAR_PROBE_PRESENT_COVERAGE)
            and peak >= float(GUITAR_PROBE_PRESENT_COVERAGE_PEAK)
        )
    )
    absent = (
        peak < float(GUITAR_PROBE_ABSENT_PEAK)
        and coverage <= float(GUITAR_PROBE_ABSENT_COVERAGE)
    )

    return {
        "peak": peak,
        "coverage": coverage,
        "hits": hits,
        "n_windows": n_windows,
        "peak_time": float(times[peak_i]) if peak_i < len(times) else 0.0,
        "best_label": best_label,
        "per_label": per_label,
        "present": bool(present),
        "absent": bool(absent),
    }


def build_instrument_presence_by_section(stems, sections):
    """Which instrument is audible in WHICH section, not just in the track.

    Instrument tags were whole-track: a label either made the list for the
    whole song or it did not. So "when does the organ come in", "is there
    guitar in the bridge", "what's actually playing in the intro" had no
    evidence behind them at all, and got answered from arrangement habit --
    the same failure mode as the drums, one instrument family over.

    The windowed tagger already produces timestamped probabilities; this
    regroups them by section instead of collapsing them to a track-wide peak.
    No extra inference: the probability matrices are cached from the per-stem
    tagging pass that has already run.

    Reported per instrument rather than per section, because "the organ is in
    the bridge only" is the answer to the question people actually ask, and it
    stays short on a track with ten sections and five tagged stems.
    """
    if not INSTRUMENT_SECTION_PRESENCE_ENABLED or not stems or not sections:
        return ""
    try:
        groups = _panns_label_groups(_INSTRUMENT_LABEL_MAP)
    except Exception:
        return ""
    if not groups:
        return ""

    win = float(INSTRUMENT_TAG_WINDOW_SECONDS or 10.0)
    found = {}   # (label, stem) -> [section names]
    boundary_ambiguous = set()  # (label, stem, section name)
    weak = set()  # (label, stem) pairs that are weak, stem-only detections
    for stem in INSTRUMENT_TAG_STEMS:
        path = (stems or {}).get(stem)
        if not path or not os.path.exists(str(path)):
            continue
        # Only labels that survived the per-stem filters (family thresholds,
        # persistence, empty-stem gate, whole-mix agreement) may be placed in a
        # section. This block used to read the raw probabilities with none of
        # those checks, which is how one-window misfires ("didgeridoo", "gong",
        # "flute") reached the section map. A stem that was never tagged (it
        # was silent, or tagging is off) has nothing verified to place.
        allowed = _ACCEPTED_STEM_TAGS.get(os.path.abspath(str(path)))
        if not allowed:
            continue
        for _lab in _WEAK_STEM_TAGS.get(os.path.abspath(str(path))) or ():
            weak.add((_lab, stem))
        try:
            probs, times = _panns_window_probs(
                path, INSTRUMENT_TAG_WINDOW_SECONDS, INSTRUMENT_TAG_HOP_SECONDS,
                INSTRUMENT_TAG_MAX_WINDOWS, INSTRUMENT_TAG_NORMALIZE,
            )
        except Exception:
            continue
        if probs is None or not times:
            continue
        centers = [float(t) + win / 2.0 for t in times]
        for sec in sections:
            try:
                a, b = float(sec["start"]), float(sec["end"])
            except Exception:
                continue
            idx = [i for i, c in enumerate(centers) if a <= c < b]
            if not idx:
                # Short section: fall back to whichever window covers it.
                idx = [i for i, t in enumerate(times) if float(t) <= a < float(t) + win]
            if not idx:
                continue
            for mapped, cols in groups.items():
                if mapped not in allowed:
                    continue
                try:
                    peak = float(probs[np.ix_(idx, cols)].max())
                except Exception:
                    continue
                if peak >= float(INSTRUMENT_SECTION_MIN_PROB):
                    sec_name = sec.get("name") or "?"
                    found.setdefault((mapped, stem), []).append(sec_name)
                    # A hit in a 10-second window straddling a boundary may
                    # belong entirely to the neighbouring section. Retain the
                    # detection, but disclose its temporal ambiguity.
                    interior_support = any(
                        float(times[i]) >= a and float(times[i]) + win <= b
                        and float(probs[i, cols].max()) >= float(INSTRUMENT_SECTION_MIN_PROB)
                        for i in idx)
                    if not interior_support:
                        boundary_ambiguous.add((mapped, stem, sec_name))
    if not found:
        return ""

    all_names = [sec.get("name") or "?" for sec in sections]
    total = len(all_names)
    rows = []
    for (label, stem), names in found.items():
        uniq = [n for i, n in enumerate(all_names) if n in set(names)]
        if len(uniq) >= total:
            where = "throughout"
        elif len(uniq) == 1:
            where = f"{uniq[0]} only"
        else:
            where = ", ".join(uniq)
        if any((label, stem, name) in boundary_ambiguous for name in uniq):
            where = ("windows spanning all listed sections" if where == "throughout"
                     else where.removesuffix(" only"))
            where += " [location uncertain: supporting windows overlap section boundaries]"
        rows.append((len(uniq), label, stem, where))
    rows.sort(key=lambda r: (r[0], r[1]))
    rows = rows[: max(1, int(INSTRUMENT_SECTION_MAX_LABELS)) * 3]

    lines = [
        "INSTRUMENT PRESENCE BY SECTION (independent classifier, per separated "
        "stem, grouped by this track's own sections -- use this when asked which "
        "instruments play in a particular section, when something enters or drops "
        "out, or what is playing at a given point. It reports WHERE an instrument "
        "is detected, not how loud it is there: for loudness use INTENSITY BY "
        "SECTION, and for what it plays use MELODY PER SECTION. These are "
        "classifier detections on separated stems, so treat a single-section "
        "entry as a possible localised appearance, not proof of exclusive presence; stay cautious about exact "
        "entry/exit moments):"
    ]
    for _n, label, stem, where in rows:
        # Weak tags keep their caveat here: an AI-generated test track's bass stem was listed as
        # "bass guitar", "guitar" and "plucked strings" throughout, with nothing
        # to say all three were stem-only detections the whole-mix pass did
        # not confirm.
        caveat = (" [weak: this stem only; the whole-mix pass did not confirm it -- "
                  "a timbre in this register, not a confirmed instrument]"
                  if (label, stem) in weak else "")
        lines.append(f"  {label} ({stem} stem): {where}{caveat}")
    return "\n".join(lines)


def _mix_pass_is_low_power(whole_mix_tags):
    """Does the whole-mix tagging pass have enough discriminating power for its
    DISAGREEMENT to mean anything?

    Returns (is_low_power, detail_string). A pass that identified one or two
    labels barely over threshold has not really analysed the track, so "guitar
    is not in that list" says nothing about guitar. Distinguishing this from a
    confident pass that genuinely found no guitar is the whole point: only the
    latter is evidence.
    """
    if whole_mix_tags is None:
        return False, ""
    try:
        n = len(whole_mix_tags)
    except Exception:
        return False, ""
    if n == 0:
        return True, "whole-mix pass returned no tags at all"
    try:
        top = max(float(t[1]) for t in whole_mix_tags if t and len(t) > 1)
    except (TypeError, ValueError):
        return False, ""
    if n <= int(MIX_PASS_LOW_POWER_MAX_TAGS) and top < float(MIX_PASS_LOW_POWER_MAX_TOP_PROB):
        return True, (
            f"whole-mix pass identified only {n} label"
            f"{'' if n == 1 else 's'}, top confidence {top * 100:.0f}%"
        )
    return False, ""


def _label_is_native_to_stem(label, stem):
    """Is this instrument the stem's own source rather than bleed into it?

    Demucs gives guitar, bass, piano, vocals and drums their own stems, so a
    guitar tag on the guitar stem is that stem doing its job. A guitar tag on
    the "other" stem is bleed or a lookalike. Agreements that contain a native
    hit are anchored to a real separated source; agreements made entirely of
    non-native stems may be one mis-identified source counted several times.
    """
    fams = _STEM_NATIVE_FAMILIES.get((stem or "").strip().lower())
    if not fams:
        return False
    lab = (label or "").strip().lower()
    if lab in fams:
        return True
    # Guitar family is matched by shape as well as by name, so a guitar-family
    # label not spelled exactly as listed still anchors to the guitar stem.
    if (stem or "").strip().lower() == "guitar":
        return _is_guitar_family_label(lab)
    return False


def _build_cross_stem_convergence(per_stem_tags, whole_mix_tags=None,
                                  mix_audio_path=None):
    """Find instruments that several independently-tagged stems agree on, and
    say honestly how much that agreement is worth.

    `per_stem_tags` maps displayed label -> [(stem, prob, coverage), ...].
    Returns (block_text, converged) where converged maps label -> list of stem
    names for agreements that survived both guards, so callers can use the
    agreement as evidence as well as report it.

    Two things this deliberately does NOT do. It does not treat agreement as
    proof of an ongoing part: presence and extent are separate claims, and the
    extent question is answered by whole-mix persistence and the per-section
    intensity blocks. And it does not treat stems as independent observers:
    they share one separation, so bleed-prone agreement with no native anchor
    is flagged rather than counted twice.
    """
    mix_cov = {}
    if CONVERGENCE_EXTENT_CONFLICT_ENABLED and mix_audio_path:
        try:
            for lab, _p, _t in (whole_mix_tags or []):
                st = _panns_tag_stats(mix_audio_path, lab)
                if st:
                    mix_cov[lab] = (
                        float(st.get("coverage") or 0.0),
                        int(st.get("hits") or 0),
                        int(st.get("n_windows") or 0),
                        float(st.get("peak_time") or 0.0),
                    )
        except Exception:
            mix_cov = {}

    converged = {}
    lines = []
    for label, entries in sorted(per_stem_tags.items()):
        strong = [e for e in entries if float(e[1]) >= float(CROSS_STEM_CONVERGENCE_MIN_PROB)]
        if len(strong) < int(CROSS_STEM_CONVERGENCE_MIN_STEMS):
            continue
        sustained = [
            e for e in strong
            if float(e[2] or 0.0) >= float(INSTRUMENT_TAG_SUSTAINED_COVERAGE)
        ]
        if len(sustained) < int(CROSS_STEM_CONVERGENCE_MIN_SUSTAINED_STEMS):
            continue

        stems_hit = [e[0] for e in sustained]
        anchored = any(_label_is_native_to_stem(label, st) for st in stems_hit)

        # Unanchored agreement across a pair Demucs is known to smear between
        # is weak: it is consistent with one source appearing in two stems.
        bleed_only = False
        if not anchored and len(stems_hit) >= 2:
            pairs = [
                frozenset((a, b))
                for i, a in enumerate(stems_hit) for b in stems_hit[i + 1:]
                if a != b
            ]
            bleed_only = bool(pairs) and all(p in BLEED_PRONE_STEM_PAIRS for p in pairs)

        # Extent conflict: whole mix found it, but only briefly.
        #
        # Only meaningful when the agreement is UNANCHORED. The whole-mix pass
        # is weak at persistence as well as at identification -- in a dense
        # master an instrument playing throughout still only clears threshold
        # in a handful of windows -- so low whole-mix coverage is not by itself
        # evidence of brief extent. When a native stem carries the instrument
        # sustained, that stem is the better extent witness (isolated and
        # peak-normalised, which is the reason it exists), and firing this rule
        # there would downgrade a genuinely constant part into "a brief detail".
        # With no native anchor there is no such witness, and the whole-mix
        # reading becomes the more trustworthy of the two.
        cov = mix_cov.get(label)
        extent_conflict = bool(
            cov
            and not anchored
            and cov[0] < float(INSTRUMENT_TAG_SUSTAINED_COVERAGE)
        )

        detail = ", ".join(
            f"{stem} stem {prob * 100:.0f}%" for stem, prob, _c in sorted(
                sustained, key=lambda e: e[1], reverse=True
            )
        )
        line = f"- {label}: agreed by {len(sustained)} stems ({detail})"

        cautions = []
        if bleed_only:
            cautions.append(
                "UNANCHORED -- none of these stems is the dedicated stem for this "
                "instrument, and Demucs commonly smears sources between them, so this "
                "may be one mis-identified source appearing twice rather than two "
                "independent detections. Do not upgrade it to a confident identification"
            )
        if extent_conflict:
            _c, _h, _n, _pt = cov
            cautions.append(
                f"EXTENT CONFLICT -- the whole-mix pass, which has no separation "
                f"artifacts, detected this in only {_h}/{_n} windows (strongest near "
                f"{_pt:.0f}s), not across the track. Treat it as a brief or localised "
                f"detail around that point, NOT as an ongoing layer; the sustained stem "
                f"reading is likely a similar timbre bleeding through"
            )
        if cautions:
            line += "\n    [" + "; ".join(cautions) + "]"
        else:
            converged[label] = stems_hit
        lines.append(line)

    if not lines:
        return "", converged

    return (
        "CROSS-STEM INSTRUMENT AGREEMENT (the same instrument identified independently "
        "on more than one separated stem, each sustained across that stem). Demucs "
        "routinely splits one loud instrument across several stems, so repeated "
        "agreement between isolated, peak-normalised stems is real evidence that a "
        "timbre of this family is PRESENT somewhere in the track -- often more "
        "reliable than the whole-mix pass, where everything competes for a few label "
        "slots. But it is evidence of presence only, NOT of extent: these stems all "
        "come from one separation of one file, so they are not fully independent "
        "observers. How much of the track an instrument actually plays is answered by "
        "the whole-mix window counts and the INTENSITY BY SECTION blocks, and how loud "
        "it sits by the STEM ACTIVITY / PROMINENCE ranking -- never by this list. Read "
        "any bracketed caution below as overriding the agreement above it:\n"
        + "\n".join(lines)
    ), converged


def _guitar_stem_is_active(stem_notes=None, activity_metas=None):
    """Corroborating evidence from the dedicated Demucs guitar stem.

    htdemucs_6s separates guitar into its own stem, so a guitar-driven track
    should leave real energy and real transcribed notes there even when the
    AudioSet tagger misses the instrument in the full mix. Returns
    (is_active, detail_string). Both inputs are optional; missing data simply
    yields (False, "") and the caller falls back to tag evidence alone.
    """
    note_count = None
    rms = None
    try:
        if stem_notes and isinstance(stem_notes, dict):
            gn = stem_notes.get("guitar")
            if gn is not None:
                note_count = len(gn)
    except Exception:
        note_count = None
    try:
        for m in (activity_metas or []):
            if (m or {}).get("stem") == "guitar":
                if m.get("silent"):
                    rms = 0.0
                else:
                    rms = float(m.get("rms") or 0.0)
                if note_count is None:
                    note_count = int(m.get("note_count") or 0)
                break
    except Exception:
        pass

    if note_count is None and rms is None:
        return False, ""

    active = False
    if rms is not None and rms >= float(GUITAR_STEM_ACTIVE_MIN_RMS):
        active = True
    # A note count alone is not a part: Omnizart transcribes notes from
    # separation residue too (an AI-generated test track's guitar stem: 149 notes at RMS 0.002,
    # i.e. -53 dBFS, quieter than its piano stem, which the report already
    # calls effectively silent). Require at least a fifth of the RMS bar.
    if (note_count is not None and note_count >= int(GUITAR_STEM_ACTIVE_MIN_NOTES)
            and (rms is None or rms >= 0.2 * float(GUITAR_STEM_ACTIVE_MIN_RMS))):
        active = True

    bits = []
    if rms is not None:
        bits.append(f"RMS≈{rms:.5f}")
    if note_count is not None:
        bits.append(f"{note_count} transcribed notes")
    return active, ", ".join(bits)


def _instrument_mix_support(mix_audio_path, label):
    """A label omitted by top-k is not absent. Reuse cached, persistence-gated
    stats from the mix pass; do not run an additional model or lower thresholds.
    """
    if not mix_audio_path:
        return 0.0
    stats = _panns_tag_stats(mix_audio_path, label)
    if not stats:
        return 0.0
    peak = float(stats.get("peak_prob") or 0.0)
    median = float(stats.get("median_prob") or 0.0)
    single = (int(stats.get("hits") or 0) == 1
              and peak >= float(INSTRUMENT_TAG_SINGLE_WINDOW_STRONG_PROB))
    return max(.65 * peak + .35 * median, peak if single else 0.0)


def _filter_instrument_tags(tags, note_count=None, whole_mix_tags=None,
                            audio_path=None, mix_audio_path=None):
    """Apply family-specific thresholds, stem-activity gating, persistence
    gating, and optional whole-mix agreement notes. Returns list of
    (label, prob, peak_t, confidence_note) where confidence_note is "" or a
    short caution string for the report line.

    `audio_path` / `mix_audio_path` are optional: when supplied, per-label
    persistence statistics (how many windows the label actually fired in) are
    consulted so a single fluke window can be held to a higher bar than an
    instrument that is audible throughout. Both default to None, in which case
    behaviour is exactly the previous peak-probability-only logic.
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

    # A whole-mix pass that barely identified anything cannot meaningfully
    # disagree with a stem tag; treating its silence as a veto suppresses real
    # instruments (three stems agreeing on guitar, all three marked "whole-mix
    # did not agree", because the mix pass returned one label at 21%).
    mix_low_power, mix_low_power_detail = _mix_pass_is_low_power(whole_mix_tags)

    # The whole-mix tag list is top-k truncated and threshold-filtered, so
    # "guitar is not in this list" is weaker evidence than it looks. When the
    # original mix audio is available, read guitar-family support straight
    # from the raw per-window probabilities instead, so a stem guitar tag is
    # not discarded merely because the mix list had no room for it.
    if mix_audio_path:
        try:
            probe = _guitar_family_probe(mix_audio_path)
            if probe:
                mix_by_label["__guitar__"] = max(
                    mix_by_label.get("__guitar__", 0.0), float(probe.get("peak") or 0.0)
                )
        except Exception:
            pass

    out = []
    for label, prob, t in tags:
        conf_note = ""
        min_needed = INSTRUMENT_TAG_MIN_PROB
        if _is_guitar_family_label(label):
            min_needed = max(min_needed, float(INSTRUMENT_TAG_GUITAR_MIN_PROB))

        stats = _panns_tag_stats(audio_path, label) if audio_path else None
        coverage = float(stats.get("coverage") or 0.0) if stats else 0.0
        hits = int(stats.get("hits") or 0) if stats else 0
        sustained = bool(stats) and coverage >= float(INSTRUMENT_TAG_SUSTAINED_COVERAGE)

        # Persistence gate. The score is the single strongest window, which is
        # what makes brief solos detectable — but it also means one spurious
        # window reads exactly like a part that plays all the way through.
        # Across ~56 windows some label spiking once by chance is likely, so a
        # one-window claim has to clear a higher bar.
        if stats and hits > 0:
            if hits < int(INSTRUMENT_TAG_MIN_WINDOW_HITS) and prob < float(
                INSTRUMENT_TAG_SINGLE_WINDOW_STRONG_PROB
            ):
                continue
            # Conversely, an instrument audible across much of the track is
            # genuinely present even at a moderate peak score; the peak-only
            # rule used to discard these.
            if sustained:
                min_needed *= float(INSTRUMENT_TAG_SUSTAINED_PROB_RELIEF)

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

        if whole_mix_tags is not None and not mix_low_power:
            mix_p = mix_by_label.get(label, 0.0)
            try:
                mix_p = max(mix_p, _instrument_mix_support(mix_audio_path, label))
            except (TypeError, ValueError):
                pass
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
        elif mix_low_power:
            # The mix pass failed on this master, so it cannot corroborate OR
            # contradict. Say exactly that rather than reporting the stem tag
            # as contradicted, and let it through on its own merits.
            conf_note = f"whole-mix cross-check inconclusive ({mix_low_power_detail})"

        # Persistence is reported alongside the confidence note: "sustained"
        # vs "isolated" is exactly the distinction between a rhythm part and a
        # one-off fill, and the peak score alone cannot express it.
        cov_note = _tag_coverage_note(stats)
        if cov_note:
            conf_note = f"{conf_note}; {cov_note}" if conf_note else cov_note

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

        # Same persistence reasoning as the per-stem path: a label that fired
        # in exactly one window out of dozens is far more likely to be a
        # momentary misfire than an instrument in the arrangement, so it has
        # to be strong; one that persists is trusted a little lower.
        stats = _panns_tag_stats(mix_audio_path, label)
        if stats and int(stats.get("hits") or 0) > 0:
            hits = int(stats.get("hits") or 0)
            coverage = float(stats.get("coverage") or 0.0)
            if hits < int(INSTRUMENT_TAG_MIN_WINDOW_HITS) and prob < float(
                INSTRUMENT_TAG_SINGLE_WINDOW_STRONG_PROB
            ):
                continue
            if coverage >= float(INSTRUMENT_TAG_SUSTAINED_COVERAGE):
                need *= float(INSTRUMENT_TAG_SUSTAINED_PROB_RELIEF)

        if prob >= need:
            filtered.append((label, prob, t))

    # TOP-K RESCUE. _panns_tag_windowed() truncates to the top
    # WHOLE_MIX_INSTRUMENT_TAG_TOP_K labels by peak probability BEFORE any
    # family threshold is applied. The instrument map holds a large drum
    # family plus bass and synth, all of which score high on a loud
    # commercial master, so on a dense pop-rock mix a real, prominent
    # electric guitar can be ranked outside the top 6 and vanish before it is
    # ever tested against its own bar -- after which the pipeline reported
    # "no clear guitar-family signal" on a track built around guitar riffs.
    # Consult the raw probabilities directly and put the guitar back if it is
    # genuinely there, flagged honestly as a low-confidence detection.
    try:
        if not any(_is_guitar_family_label(lab) for lab, _p, _t in filtered):
            probe = _guitar_family_probe(mix_audio_path)
            if probe and probe.get("present") and probe.get("best_label"):
                filtered.append(
                    (probe["best_label"], float(probe["peak"]), float(probe["peak_time"]))
                )
                filtered.sort(key=lambda kv: kv[1], reverse=True)
    except Exception:
        pass
    return filtered


# =============================================================================
# dev3 (Stage 1) MTG-JAMENDO INSTRUMENT TAGS -- a second, independent instrument
# classifier, saved for later use and NOT shown to the writer model.
# =============================================================================
# MTG's Discogs-EffNet embedding model plus its MTG-Jamendo instrument head (40
# instruments, finer on electronic sounds: synthesizer, pad, electric piano,
# Rhodes, organ, drum machine...), run through onnxruntime on the CPU -- no
# TensorFlow. The mel input is Essentia's own TensorflowInputMusiCNN (16 kHz,
# 512/256, 96 bands), framed as Essentia's EffNet predictor frames it.
# Run on the full mix and on each separated stem, in ~2 s patches every ~1 s;
# near-silent patches are left out. The result is saved in the song's .json as
# "jamendo_instruments" (per-source and per-section summaries) and is not added
# to the analysis text, so the writer, the genre/identity steps and every other
# decision are unchanged. Needs a one-time download:
#     python musiclyse_jamendo_setup.py
# Until then it is skipped with a one-line note. Model licence: CC BY-NC-SA 4.0
# (non-commercial), like MAEST. Never raises out of the scan.
_JAMENDO_LOCK = threading.Lock()
_JAMENDO_STATE = {"unavailable": None}
_JAMENDO_RESULTS = collections.OrderedDict()   # track path -> {"labels": [...], "sources": {...}}
_JAMENDO_SR = 16000
_JAMENDO_FRAME = 512
_JAMENDO_HOP = 256
_JAMENDO_PATCH = 128            # frames per patch (~2.05 s)
_JAMENDO_PATCH_HOP = 62         # frames between patches (~0.99 s)
_JAMENDO_EMBED_DIM = 1280
_JAMENDO_N_LABELS = 40
_JAMENDO_FORMAT = 1


class _JamendoModelMissing(RuntimeError):
    """The MTG-Jamendo models could not be loaded (not downloaded, damaged, or
    onnxruntime / Essentia missing): skipped for the rest of the session."""


def _jamendo_dir():
    d = str(JAMENDO_MODEL_DIR or "").strip()
    if d:
        return os.path.abspath(os.path.expanduser(d))
    # 0.6.2-dev2: one shared folder in the home folder (like ~/panns_data); the
    # 0.6.2-dev1 location next to the script is still used if the models are
    # only there.
    shared = os.path.join(os.path.expanduser("~"), "mtg_jamendo_data")
    old = os.path.join(os.path.dirname(os.path.abspath(__file__)), "musiclyse-models", "mtg-jamendo")
    if not os.path.isfile(os.path.join(shared, JAMENDO_HEAD_FILE)) and os.path.isfile(os.path.join(old, JAMENDO_HEAD_FILE)):
        return old
    return shared


def _jamendo_io(sess, want_last_dim):
    """(input name, input shape, output name) of an ONNX session, picking the
    output whose last dimension is `want_last_dim`."""
    ins = sess.get_inputs()
    if len(ins) != 1:
        raise RuntimeError(f"expected one model input, found {len(ins)}")
    outs = [o for o in sess.get_outputs() if o.shape and o.shape[-1] == want_last_dim]
    if not outs:
        raise RuntimeError(f"no model output of size {want_last_dim}")
    return ins[0].name, list(ins[0].shape), outs[0].name


def _jamendo_load():
    """(embed session, head session, labels, io) from JAMENDO_MODEL_DIR."""
    try:
        import onnxruntime as ort
    except Exception as e:
        raise _JamendoModelMissing(f"onnxruntime is not installed ({e})")
    if not ESSENTIA_AVAILABLE:
        raise _JamendoModelMissing("Essentia is not available for the mel input")
    try:
        import essentia.standard as _es
        _es.TensorflowInputMusiCNN
    except Exception as e:
        raise _JamendoModelMissing(f"this Essentia lacks TensorflowInputMusiCNN ({e})")
    d = _jamendo_dir()
    paths = [os.path.join(d, f) for f in (JAMENDO_EMBED_FILE, JAMENDO_HEAD_FILE, JAMENDO_LABELS_FILE)]
    missing = [os.path.basename(p) for p in paths if not os.path.isfile(p)]
    if missing:
        raise _JamendoModelMissing(f"not downloaded ({', '.join(missing)} missing in {d})")
    with open(paths[2], "r", encoding="utf-8") as f:
        meta = json.load(f)
    labels = [str(x) for x in (meta.get("classes") or [])] if isinstance(meta, dict) else []
    if len(labels) != _JAMENDO_N_LABELS:
        raise _JamendoModelMissing(f"the labels file lists {len(labels)} classes, not {_JAMENDO_N_LABELS}")
    so = ort.SessionOptions()
    so.intra_op_num_threads = max(1, int(JAMENDO_THREADS or 1))
    so.inter_op_num_threads = 1
    try:
        emb = ort.InferenceSession(paths[0], sess_options=so, providers=["CPUExecutionProvider"])
        head = ort.InferenceSession(paths[1], sess_options=so, providers=["CPUExecutionProvider"])
        io = {"emb": _jamendo_io(emb, _JAMENDO_EMBED_DIM), "head": _jamendo_io(head, _JAMENDO_N_LABELS)}
    except Exception as e:
        raise _JamendoModelMissing(f"the model files could not be opened ({e})")
    return emb, head, labels, io


def _jamendo_mel(path):
    """(mel [frames, 96] float32, audio 16 kHz mono float32) via Essentia."""
    import essentia.standard as _es
    audio = _es.MonoLoader(filename=str(path), sampleRate=_JAMENDO_SR, resampleQuality=4)()
    audio = np.asarray(audio, dtype=np.float32)
    if audio.size < _JAMENDO_FRAME:
        return np.zeros((0, 96), dtype=np.float32), audio
    fe = _es.TensorflowInputMusiCNN()
    frames = [fe(fr) for fr in _es.FrameGenerator(audio, frameSize=_JAMENDO_FRAME, hopSize=_JAMENDO_HOP,
                                                    startFromZero=True, validFrameThresholdRatio=1)]
    if not frames:
        return np.zeros((0, 96), dtype=np.float32), audio
    return np.asarray(frames, dtype=np.float32), audio


def _jamendo_patches(mel):
    """[patches, 128, 96] and each patch's centre time in seconds."""
    n = (mel.shape[0] - _JAMENDO_PATCH) // _JAMENDO_PATCH_HOP + 1 if mel.shape[0] >= _JAMENDO_PATCH else 0
    if n <= 0:
        return np.zeros((0, _JAMENDO_PATCH, 96), dtype=np.float32), []
    starts = [i * _JAMENDO_PATCH_HOP for i in range(n)]
    patches = np.stack([mel[s:s + _JAMENDO_PATCH] for s in starts]).astype(np.float32)
    times = [(s + _JAMENDO_PATCH / 2.0) * _JAMENDO_HOP / _JAMENDO_SR for s in starts]
    return patches, times


def _jamendo_shape_batch(x, shape):
    """Fit [n, 128, 96] to the model's declared input layout."""
    rank = len(shape)
    if rank == 3:
        return x
    if rank == 4:
        if shape[1] == 1:
            return x[:, None, :, :]
        if shape[3] == 1:
            return x[:, :, :, None]
    raise RuntimeError(f"unexpected model input shape {shape}")


def _jamendo_run(sess, io, x):
    """Run `x` ([n, ...]) through a session, honouring a fixed batch size."""
    name, shape, out = io
    fixed = shape[0] if isinstance(shape[0], int) and shape[0] > 0 else None
    step = fixed or 64
    res = []
    for i in range(0, x.shape[0], step):
        chunk = x[i:i + step]
        k = chunk.shape[0]
        if fixed and k < fixed:
            chunk = np.concatenate([chunk, np.zeros((fixed - k,) + chunk.shape[1:], dtype=chunk.dtype)])
        y = sess.run([out], {name: chunk})[0]
        res.append(np.asarray(y, dtype=np.float32).reshape(chunk.shape[0], -1)[:k])
    return np.concatenate(res) if res else np.zeros((0, 0), dtype=np.float32)


def _jamendo_tag_audio(path, models):
    """{"times": [...], "active": [bool], "probs": [patches, 40]} or None."""
    emb, head, labels, io = models
    mel, audio = _jamendo_mel(path)
    patches, times = _jamendo_patches(mel)
    if not times:
        return None
    # near-silent patches (a stem with nothing in it) are left out
    rms = []
    for t in times:
        a = int(max(0, (t - _JAMENDO_PATCH * _JAMENDO_HOP / _JAMENDO_SR / 2) * _JAMENDO_SR))
        b = int(a + _JAMENDO_PATCH * _JAMENDO_HOP)
        seg = audio[a:b]
        rms.append(float(np.sqrt(np.mean(seg.astype(np.float64) ** 2))) if seg.size else 0.0)
    peak = max(rms) if rms else 0.0
    floor = max(1e-4, peak * 10 ** (-float(JAMENDO_QUIET_REL_DB) / 20.0))
    active = [r >= floor for r in rms]
    if not any(active):
        return {"times": times, "active": active, "probs": np.zeros((len(times), _JAMENDO_N_LABELS), np.float32)}
    x = _jamendo_shape_batch(patches, io["emb"][1])
    e = _jamendo_run(emb, io["emb"], x)
    p = _jamendo_run(head, io["head"], e.reshape(e.shape[0], -1))
    if p.size and (float(np.min(p)) < 0.0 or float(np.max(p)) > 1.0):
        p = 1.0 / (1.0 + np.exp(-p))           # a head exported without its sigmoid
    return {"times": times, "active": active, "probs": p}


def _jamendo_summary(res, labels):
    """Per-label [max, time of max, mean, coverage] over the active patches."""
    act = [i for i, a in enumerate(res["active"]) if a]
    out = {"patches": len(res["times"]), "active_patches": len(act)}
    if not act:
        out["labels"] = {}
        return out
    P = res["probs"][act]
    T = [res["times"][i] for i in act]
    lab = {}
    for j, name in enumerate(labels):
        col = P[:, j]
        k = int(np.argmax(col))
        lab[name] = [round(float(col[k]), 3), round(float(T[k]), 1), round(float(np.mean(col)), 3),
                     round(float(np.mean(col >= float(JAMENDO_COVERAGE_PROB))), 3)]
    out["labels"] = lab
    return out


def _jamendo_remember(track_path, entry):
    if not track_path or not entry:
        return
    _JAMENDO_RESULTS[str(track_path)] = entry
    _JAMENDO_RESULTS.move_to_end(str(track_path))
    while len(_JAMENDO_RESULTS) > 16:
        _JAMENDO_RESULTS.popitem(last=False)


def _jamendo_models():
    """Loaded models, or None (with a one-line note, once per session)."""
    if _JAMENDO_STATE["unavailable"]:
        return None
    try:
        return _jamendo_load()
    except Exception as e:
        msg = str(e).strip().splitlines()[0][:200] if str(e).strip() else type(e).__name__
        _JAMENDO_STATE["unavailable"] = msg
        print(f"  (MTG-Jamendo instrument tags skipped for this session -- run musiclyse_jamendo_setup.py "
              f"once to download (or repair) the models: {msg})")
        return None


_JAMENDO_STEM_ORDER = ("vocals", "drums", "bass", "guitar", "piano", "other")


def jamendo_tag_track(track_path, mix_path, stems=None):
    """Tag the full mix (and any separated stems) and keep the result for the
    save. Never raises; does nothing when switched off."""
    if not ENABLE_JAMENDO_INSTRUMENTS or not track_path or not mix_path:
        return
    if str(track_path).startswith(("http://", "https://")):
        return
    try:
        with _JAMENDO_LOCK:
            models = _jamendo_models()
            if models is None:
                return
            status("Tagging instruments with a second classifier (MTG-Jamendo)...")
            entry = {"labels": list(models[2]), "sources": {}, "raw": {}}
            res = _jamendo_tag_audio(mix_path, models)
            if res is not None:
                entry["sources"]["mix"] = _jamendo_summary(res, models[2])
                entry["raw"]["mix"] = res
            _jamendo_tag_stems_into(entry, stems, models)
            del models
            _jamendo_remember(track_path, entry)
            status_done("MTG-Jamendo instruments: " + (_jamendo_top_words(entry) or "none clear"))
    except Exception as e:
        msg = str(e).strip().splitlines()[0][:200] if str(e).strip() else type(e).__name__
        print(f"  (MTG-Jamendo instrument tags skipped: {msg})")


def jamendo_add_stems(track_path, stems):
    """Add stems separated later in the scan to an existing mix result."""
    if not ENABLE_JAMENDO_INSTRUMENTS or not JAMENDO_STEMS or not stems or not track_path:
        return
    entry = _JAMENDO_RESULTS.get(str(track_path))
    if not entry or any(k != "mix" for k in entry["sources"]):
        return
    try:
        with _JAMENDO_LOCK:
            models = _jamendo_models()
            if models is None:
                return
            status("Tagging the separated parts with the second instrument classifier...")
            _jamendo_tag_stems_into(entry, stems, models)
            del models
    except Exception as e:
        msg = str(e).strip().splitlines()[0][:200] if str(e).strip() else type(e).__name__
        print(f"  (MTG-Jamendo stem tags skipped: {msg})")


def _jamendo_tag_stems_into(entry, stems, models):
    if not JAMENDO_STEMS or not isinstance(stems, dict):
        return
    for st in sorted(stems, key=lambda s: (_JAMENDO_STEM_ORDER.index(s) if s in _JAMENDO_STEM_ORDER else 99, s)):
        p = stems.get(st)
        if not p or not os.path.isfile(str(p)):
            continue
        try:
            res = _jamendo_tag_audio(p, models)
        except Exception as e:
            print(f"  (MTG-Jamendo tags for the {st} stem skipped: {str(e).strip()[:160]})")
            continue
        if res is not None:
            entry["sources"][str(st)] = _jamendo_summary(res, models[2])
            entry["raw"][str(st)] = res


def _jamendo_top_words(entry):
    lab = ((entry.get("sources") or {}).get("mix") or {}).get("labels") or {}
    top = sorted(((v[0], k) for k, v in lab.items() if v and v[0] >= 0.5), reverse=True)[:4]
    return ", ".join(k for _p, k in top)


def _jamendo_for_save_safe(track_key, structure_sections=None):
    """_jamendo_for_save, never raising (for the analysis result handed to a
    separate save process)."""
    try:
        return _jamendo_for_save(track_key, structure_sections)
    except Exception:
        return None


def _jamendo_for_save(track_key, structure_sections=None):
    """The saved "jamendo_instruments" block for this track, or None."""
    entry = _JAMENDO_RESULTS.get(str(track_key)) if track_key else None
    if not entry or not entry.get("sources"):
        return None
    secs_out = []
    for s in structure_sections or []:
        try:
            a, z = float(s.get("start", 0.0)), float(s.get("end", 0.0))
        except Exception:
            continue
        if z - a < 1.0 or str(s.get("name") or "").lower().startswith("silence"):
            continue
        per = {}
        for src, res in (entry.get("raw") or {}).items():
            idx = [i for i, t in enumerate(res["times"]) if a <= t < z and res["active"][i]]
            if not idx:
                continue
            P = res["probs"][idx]
            rows = []
            for j, name in enumerate(entry["labels"]):
                mx, mn = float(np.max(P[:, j])), float(np.mean(P[:, j]))
                if mx >= float(JAMENDO_SECTION_MIN_PROB):
                    rows.append([name, round(mx, 3), round(mn, 3)])
            if rows:
                per[src] = sorted(rows, key=lambda r: -r[1])
        if per:
            secs_out.append({"name": s.get("name"), "start": round(a, 2), "end": round(z, 2), "sources": per})
    return {
        "format": _JAMENDO_FORMAT,
        "models": {"embedding": os.path.splitext(JAMENDO_EMBED_FILE)[0],
                   "head": os.path.splitext(JAMENDO_HEAD_FILE)[0]},
        "patch_seconds": round(_JAMENDO_PATCH * _JAMENDO_HOP / _JAMENDO_SR, 3),
        "hop_seconds": round(_JAMENDO_PATCH_HOP * _JAMENDO_HOP / _JAMENDO_SR, 3),
        "coverage_prob": float(JAMENDO_COVERAGE_PROB),
        "label_fields": "max, time_of_max_s, mean, coverage",
        "sources": entry["sources"],
        "sections": secs_out,
    }


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
            "(prefer describing texture without naming weak instrument identities). "
            "This means the classifier was not confident about any instrument on this "
            "track -- it does NOT mean the track is instrumentally empty or that any "
            "particular instrument is missing. Do not turn this into a negative claim."
        )

    tag_parts = []
    for label, prob, t in tags:
        part = f"{label} ({prob * 100:.0f}%, strongest near {t:.0f}s"
        cov_note = _tag_coverage_note(_panns_tag_stats(mix_audio_path, label))
        if cov_note:
            part += f", {cov_note}"
        tag_parts.append(part + ")")
    tag_text = ", ".join(tag_parts)
    return (
        "WHOLE-MIX INSTRUMENT TAGS (independent audio classifier run on the original, "
        "un-separated mix -- not subject to Demucs separation artifacts. "
        "Guitar-family labels use a higher confidence bar. Treat disagreement with a "
        "per-stem tag as a reason to omit or hedge that instrument rather than "
        "trusting the stem alone; still supporting evidence only, not proof. "
        "If a named instrument is absent here and only weakly present on one stem, "
        "prefer 'no clear X' over asserting X -- but note that this list is capped at a "
        "fixed number of labels and uses strict per-family thresholds, so a label's "
        "absence from it is NOT positive evidence that the instrument is absent from "
        "the recording, and must never be reported as 'the track has no X'. Only an "
        "explicit absence note elsewhere in these notes carries that meaning. The "
        "window counts say how much of the "
        "track each label was actually detected in: 'sustained' means an ongoing part "
        "in the arrangement, while 'isolated' means a single brief moment -- describe "
        f"an isolated tag as a passing detail, never as a main instrument): {tag_text}"
    )


def guitar_absence_note_from_tags(whole_mix_tags=None, report_text=None,
                                  mix_audio_path=None, stem_notes=None,
                                  activity_metas=None, cross_stem_agreement=None):
    """Report what is actually known about guitar on this track: present,
    absent, or undetermined -- and never silently turn the third case into the
    second.

    The original version emitted a hard GUITAR ABSENCE NOTE whenever the
    whole-mix tag list contained no guitar-family label above
    INSTRUMENT_TAG_MIX_AGREE_MIN_PROB. That list is top-k truncated and sits
    behind a deliberately strict 0.35 family threshold, so it goes empty for
    two very different reasons: because there is no guitar, or because a real
    guitar scored 0.2 on a dense limited master, or read as synth, or lost its
    place in the top 6 to the drum family. The note did not distinguish them,
    and the writer prompt treats it as authoritative -- so a detector miss
    became a confident claim of absence, and then an argument against a
    rock/pop-rock genre reading. That is the failure this rewrite fixes.

    Absence is now asserted only on positive evidence: guitar-family energy
    measurably low across the whole mix (raw per-window probabilities, no
    top-k, no reporting threshold) AND a dedicated Demucs guitar stem with no
    real activity. When the two disagree, or when the evidence is merely
    inconclusive, an explicit uncertainty note is emitted instead, which tells
    the writer to describe the texture neutrally and forbids using "no guitar"
    as a genre argument.

    `mix_audio_path`, `stem_notes`, `activity_metas` and `cross_stem_agreement`
    are optional; without them the function degrades to the tag-list evidence it
    always had, but still hedges rather than asserting absence outright.
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
        # Parse lines from WHOLE-MIX INSTRUMENT TAGS report.
        lower = report_text.lower()
        if "whole-mix instrument tags" in lower:
            # Only the tag list itself is evidence. The block's own preamble
            # contains the advisory phrase "prefer 'no clear X'", so scanning
            # the whole string for "no clear" used to zero out a positive
            # detection on every single track that reached this branch.
            tail = report_text.split("):", 1)[-1].lower() if "):" in report_text else lower
            if "none cleared" in lower:
                found_guitar = False
            else:
                for lab in ("electric guitar", "acoustic guitar", "guitar", "slide/steel"):
                    if lab in tail:
                        found_guitar = True
                        break
        else:
            return ""  # tagging didn't run / no report
    else:
        return ""

    if found_guitar:
        return ""

    # --- No guitar in the reported tags. Establish WHY before claiming it. ---
    probe = None
    try:
        probe = _guitar_family_probe(mix_audio_path) if mix_audio_path else None
    except Exception:
        probe = None

    stem_active, stem_detail = _guitar_stem_is_active(
        stem_notes=stem_notes, activity_metas=activity_metas
    )

    # Several isolated stems independently identifying guitar is corroboration
    # in its own right, and settles the question against a mix pass that found
    # nothing. Without this, three stems agreeing at 64-82% counted for zero.
    conv_stems = []
    try:
        for lab, stems_list in (cross_stem_agreement or {}).items():
            if _is_guitar_family_label(lab):
                for st in (stems_list or []):
                    if st not in conv_stems:
                        conv_stems.append(st)
    except Exception:
        conv_stems = []

    mix_low_power, mix_low_power_detail = _mix_pass_is_low_power(whole_mix_tags)

    probe_bits = ""
    if probe:
        probe_bits = (
            f"peak guitar-family confidence {probe['peak'] * 100:.0f}% "
            f"across the whole mix, fired in {probe['hits']}/{probe['n_windows']} windows"
        )

    # Case 1: the raw probabilities agree there is essentially nothing there,
    # and the dedicated guitar stem is empty too. This is a real absence.
    if probe and probe.get("absent") and not stem_active and not conv_stems and not mix_low_power:
        detail = f" ({probe_bits}"
        detail += f"; separated guitar stem inactive: {stem_detail})" if stem_detail else ")"
        return (
            "GUITAR ABSENCE NOTE: whole-mix instrument tagging found no clear "
            "guitar-family signal on THIS track" + detail + ", and the dedicated "
            "Demucs guitar stem carries no real activity. Two independent checks "
            "agree, so treat this as genuine absence: do not claim electric/acoustic "
            "guitar from genre expectation, a residual Demucs 'guitar' stem, or "
            "instruments heard on a previously discussed track. Prefer 'no clear "
            "guitar' or a texture description unless the user confirms otherwise."
        )

    # Case 2: the tag list is empty but something is still there -- the raw
    # probabilities show guitar-family energy, or the guitar stem is carrying
    # a real part, or both. This is the case that used to be misreported as
    # absence, and it is where guitar-driven tracks were losing their guitars.
    reasons = []
    if probe and (probe.get("present") or not probe.get("absent")):
        reasons.append(
            "the raw classifier still shows measurable guitar-family energy in the mix"
            + (f" ({probe_bits})" if probe_bits else "")
        )
    if stem_active:
        reasons.append(
            "the dedicated Demucs guitar stem is carrying a real part"
            + (f" ({stem_detail})" if stem_detail else "")
        )
    if conv_stems:
        reasons.append(
            "guitar was identified independently on "
            + f"{len(conv_stems)} separate stems ({', '.join(conv_stems)}), each "
            "sustained across the track"
        )
    if mix_low_power and not reasons:
        reasons.append(
            "the whole-mix pass had no discriminating power on this master "
            f"({mix_low_power_detail}), so its silence about guitar is not evidence"
        )
    if reasons:
        return (
            "GUITAR EVIDENCE INCONCLUSIVE (do NOT report this as 'no guitar'): the "
            "whole-mix tagger did not produce a confident guitar-family tag on this "
            "track, but " + ", and ".join(reasons) + ". This "
            "classifier is known to under-report guitar on dense, heavily limited "
            "commercial masters and to score overdriven or heavily processed electric "
            "guitar as synth/keyboard, so a missing tag here is weak evidence of "
            "absence and no evidence at all about the arrangement. Do NOT state that "
            "the track has no guitar, and do NOT use absence of guitar as an argument "
            "for or against any genre label (a pop-rock or rock reading must stand or "
            "fall on the rest of the evidence). Describe the midrange layer neutrally "
            "instead -- by its measured texture, brightness, register and rhythmic role, "
            "as 'guitar or a guitar-like synth' rather than committing to either -- and "
            "say the instrument tagging was inconclusive if the user asks directly."
            + (
                " NOTE: with this many independent stems agreeing, guitar is best "
                "treated as PRESENT in the arrangement; what remains genuinely "
                "uncertain is only its prominence and whether it is guitar or a "
                "guitar-like synth patch. Use the STEM ACTIVITY / PROMINENCE "
                "ranking for how loud it sits."
                if len(conv_stems) >= int(CROSS_STEM_CONVERGENCE_MIN_STEMS) else ""
            )
        )

    # Case 3: no probe available (tagger/audio missing) and no stem evidence.
    # Nothing was measured, so nothing is claimed either way.
    return (
        "GUITAR EVIDENCE UNAVAILABLE: whole-mix instrument tagging reported no "
        "guitar-family tag on this track, but no independent cross-check could be "
        "run, so this is not evidence of absence. Do not assert either that the "
        "track has guitar or that it has none; describe the texture from what was "
        "measured and treat instrument identity here as undetermined."
    )


# abs stem path -> set of instrument labels that passed _filter_instrument_tags
# for that stem (populated by build_omnizart_summaries, read by
# build_instrument_presence_by_section).
_ACCEPTED_STEM_TAGS = {}
_WEAK_STEM_TAGS = {}

# What each htdemucs_6s stem actually collects. The stem names are separation
# buckets, not instrument identifications: there is no strings, synth, brass or
# woodwind stem, so those parts land in "other" -- or in "bass" when they sit low.
_STEM_BUCKET_DESCRIPTION = {
    "bass": "any low-register pitched sound (bass guitar, but equally synth bass, low "
            "strings or pads, organ pedals or low piano)",
    "guitar": "guitar-like mid-register parts (and sometimes other plucked or "
              "distorted lead sounds)",
    "piano": "piano-like keyboard parts (and sometimes other keys or mallets)",
}
_STEM_EXPECTED_LABELS = {
    "bass": {"bass guitar", "upright bass"},
    "piano": {"piano", "electric piano", "keyboard/synth", "organ", "harpsichord"},
}


_BASS_CLAIM_RE = re.compile(r"(?i)\b(?:bass\s*guitar|bass\s*line|bassline|electric\s+bass|bass\s+part|the\s+bass\b)")


def _bass_claim_note(listening_text, activity_metas):
    """Flag a listening-model bass claim that the separated stems contradict.

    The listening model's prose described "a driving, melodic bass line ... the bass
    guitar" on a track with no bass (the low part was overdubbed synth strings,
    which the separator did not put in the bass stem at all). When the bass
    stem is far below the other stems, say so next to the stem evidence.
    """
    if not listening_text or not activity_metas:
        return ""
    if not _BASS_CLAIM_RE.search(str(listening_text)):
        return ""
    rms = {m.get("stem"): float(m.get("rms") or 0.0) for m in activity_metas if m.get("stem")}
    if "bass" not in rms:
        return ""
    total = float(np.sqrt(sum(v * v for v in rms.values()))) or 0.0
    if total <= 0:
        return ""
    bass = rms.get("bass", 0.0)
    rel = 20.0 * np.log10(max(bass, 1e-9) / total)
    if rel > -float(BASS_STEM_ABSENT_BELOW_TOTAL_DB):
        return ""
    return (
        f"LOW-END NOTE: the listening pass describes a bass guitar / bass line, but the separated "
        f"bass stem sits {abs(rel):.0f} dB below the combined stems -- there is no substantial "
        "separate bass part. Whatever low-register material the listening pass heard belongs to "
        "other instruments (for example a low synth or string part, or the guitar's low strings). "
        "Do not describe a bass guitar unless the user says there is one."
    )
