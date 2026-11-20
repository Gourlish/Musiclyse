# MUSICLYSE FILE pipeline-identity 0.8
# ==============================================================================
#  MUSICLYSE 0.8: SINGER IDENTITY
# ==============================================================================
# This file holds singer identity: who sings lead, singer age and voice checks, the
# section-by-section voice checks, and the identity prompts, notes and singer lookup. It is part
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

FEMALE_LEAD_CATEGORIES = {
    "adult_female",
    "young_female",
    "female_teen_adult",
}


SINGER_IDENTITY_RESOLUTION_PROMPT = """Resolve the likely lead singer category for this track by combining the supplied evidence.

This is an evidence-reconciliation task, not a creative inference task.

EVIDENCE PRIORITY:

1. User-confirmed corrections = ground truth.
2. Distinct co-lead evidence = strong.
3. Reliable track metadata / known artist identity = primary evidence for the singer's
   gender, and (with the release year) for their age at the time of this recording.
4. Explicit vocal acoustic analysis = primary evidence for how mature the voice sounds,
   and for gender only when identity evidence is absent or ambiguous.
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
- female_teen_adult -> retain female_teen_adult UNLESS the credited solo artist is documented
  as male (see UNBROKEN MALE VOICES below)
- child -> retain child only when acoustic evidence supports prepubertal characteristics
- uncertain -> do not invent a gender from metadata or cover alone when pitch is unremarkable

KNOWN ARTIST IDENTITY (use this — it is evidence, not an assumption):
Identifying WHO is singing is a different question from reading age/gender out of a
waveform, and the cautions below are aimed only at the second. When the track metadata
names an artist and that artist is documented (in the supplied background, or reliably
known to you) as a man or a woman, that documented fact is legitimate identity evidence
and SHOULD set the singer category. The same applies when the cover art shows a single
performer and there is reasonable confidence that person is the credited artist.
- A named, documented solo male artist -> post_puberty_male (or adult_male).
- A named, documented solo female artist -> female_teen_adult (or adult_female).
- Report CONFIDENCE=medium and say the basis was artist identity rather than a pure
  acoustic reading. That is an honest, useful answer.
- Do NOT answer "uncertain" simply because the acoustics were unremarkable. If the
  identity evidence is clear, use it; unremarkable acoustics are not a contradiction.
- Only decline when identity evidence genuinely conflicts with strong acoustic evidence
  that the acoustics CAN settle -- e.g. a group/duo where the lead is unclear, or a
  "feat." credit where a second singer may be carrying the lead. Say what the conflict is.
- UNBROKEN MALE VOICES: a boy whose voice has not broken or has only partly deepened sits
  in the same pitch range as a woman's voice, and audio classifiers routinely label it
  female_teen_adult. A female acoustic read is therefore NOT a real conflict with a
  documented solo male artist. When the credited solo artist is documented as male and
  there is no featured female singer credited, set the category from identity:
  child_male_likely or adolescent_male_likely when this recording falls in his childhood
  or teenage years (compare the release year with his birth year), otherwise
  post_puberty_male. Say plainly that the acoustic read was female-sounding pitch.
- For bands, collectives and features, the credited act's name may not tell you who sang
  lead. Prefer the acoustic reading there, and say why.
- "Documented as a man" is about sex/gender, not adult status — it does NOT by itself mean
  post_puberty_male. Check whether the documented facts point to a specific adolescent/child
  era for THIS recording (a child-star or teen-pop debut, a track released before the artist's
  voice was documented as having changed, an era publicly known for a still-high/unbroken or
  changing voice). When they do, that era-specific fact is the identity evidence to use — apply
  adolescent_male_likely (or child_male_likely) for that track, not post_puberty_male, even
  though the same person is a post-puberty adult on later recordings or today. Only use
  post_puberty_male when the documented era for this specific recording is adult or the voice
  is not otherwise documented as still changing.
This permits ONLY the singer category. It never licenses claims about instrumentation,
arrangement or any other sonic detail — those come from the audio measurements alone.

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


def _release_heavy_analysis_memory_before_identity():
    """Release large analysis-model allocations before lightweight identity work.

    Does not discard analysis text already computed in Python (revised reports,
    metadata, cover observations, etc.) — only frees model weights / GPU
    residentials so the Ollama writer can load without stacking on MOSS-Music+Demucs+TF.
    """
    try:
        unload_moss_music()
    except Exception:
        pass
    try:
        unload_whisper()
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


def _fill_identity(text):
    """(gender, age) of the saved SINGER_IDENTITY decision, or (None, None)."""
    m = re.findall(r"(?m)^SINGER_IDENTITY=(\S+)", str(text or ""))
    if not m:
        return None, None
    c = m[-1].lower()
    if "mixed" in c or "uncertain" in c or "unknown" in c:
        return None, None
    g = "female" if "female" in c else ("male" if "male" in c else None)
    a = ("young" if re.search(r"adolescent|pre_?puberty|child|boy|girl", c)
         else ("adult" if re.search(r"post_?puberty|adult", c) else None))
    return g, a


def _fill_identity_conflict(sent, ident):
    """True when a sentence about the voice contradicts the saved singer
    decision (a "female" voice for a male singer, "adult" for an unbroken
    voice ...): the decision wins, so such a sentence is not copied up."""
    g, a = ident or (None, None)
    low = sent.lower()
    fem = re.search(r"\b(?:female|woman|women|girl|girls|she|her|hers|herself)\b", low)
    mal = re.search(r"\b(?:male|man|men|boy|boys|he|him|his|himself)\b", low)
    if g == "male" and fem or g == "female" and mal:
        return True
    if a == "young" and re.search(r"\b(?:adult|man|woman|men|women|mature)\b", low):
        return True
    if a == "adult" and re.search(r"\b(?:child|children|kid|boy|girl|prepubescent|pre-pubescent|prepubertal)\b", low):
        return True
    return False


def _resolve_lead_tag_variant(tag):
    """Map near-miss lead-tag spellings onto the canonical taxonomy.

    Explicit aliases cover the common cases; this strips hedge affixes
    ("_likely", "likely_", "_probable", "possible_") so a spelling nobody
    anticipated still lands on a real category instead of "unknown".

    Returning "unknown" is not a safe default here: both age guards ignore it,
    so an unparsed tag silently switches off the entire acoustic age-correction
    layer and lets the identity pass decide alone.
    """
    tag = _normalize_vocal_tag(tag)
    if not tag:
        return ""
    if tag in VOCAL_LEAD_TAGS:
        return tag
    if tag in VOCAL_LEAD_ALIASES:
        return VOCAL_LEAD_ALIASES[tag]
    if (ADOLESCENT_TAG_NEEDS_NOT_FEMALE and re.search(r"female|girl|wom[ae]n", tag) and "child" not in tag
            and re.search(r"adolescen|transitional|chang", tag)
            and not re.search(r"(?<!fe)male|boy", tag)):
        return "female_teen_adult"       # 0.7.1-pre12: a female capture is never a male tag

    # Trailing words on the same line ("post_puberty_male (medium confidence)",
    # "female_teen_adult likely") used to collapse the whole capture into an
    # unknown tag, which silently disabled both age guards. Accept the
    # longest canonical tag/alias the capture starts with.
    _both = "female" in tag and re.search(r"(?<!fe)male", tag)
    for known in ([] if _both else sorted(set(VOCAL_LEAD_TAGS) | set(VOCAL_LEAD_ALIASES), key=len, reverse=True)):
        if tag.startswith(known + "_") and known not in ("unknown",):
            return VOCAL_LEAD_ALIASES.get(known, known) if known not in VOCAL_LEAD_TAGS else known

    stripped = re.sub(r"^(?:likely|probably|probable|possible|possibly|sounds_like)_", "", tag)
    stripped = re.sub(r"_(?:likely|probable|probably|possible|possibly)$", "", stripped)
    if stripped != tag:
        if stripped in VOCAL_LEAD_TAGS:
            return stripped
        if stripped in VOCAL_LEAD_ALIASES:
            return VOCAL_LEAD_ALIASES[stripped]
    return ""


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

    if (ADOLESCENT_TAG_NEEDS_NOT_FEMALE
            and any(token in category for token in ("adolescent", "transitional", "changing_voice", "voice_change"))
            and re.search(r"female|girl|wom[ae]n", category + " " + modifier)
            and not re.search(r"(?<!fe)male|boy", category + " " + modifier)):
        return "female_teen_adult"       # 0.7.1-pre12: "adolescent" + female is a girl's voice
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


def _marker_present_unnegated(text, markers):
    """Like _any_marker_present, but ignores markers that are negated or belong
    to an explicitly rejected alternative.

    Deliberately a SEPARATE function rather than a change to
    _any_marker_present: that helper is also used by the genre/style detection
    path, which wants plain substring/word matching and must not start skipping
    negated mentions.

    The vocal passes routinely write things like "an adult male voice, not a
    teen voice" or "the voice is not still changing". Plain keyword scanning
    reads those as adolescent evidence and inverts the conclusion.
    """
    if not text:
        return False

    negators = (
        "not", "no", "never", "nothing", "isn't", "isnt", "aren't", "arent",
        "doesn't", "doesnt", "don't", "dont", "wasn't", "wasnt", "hardly",
        "rather than", "instead of", "far from", "without", "no longer",
        "rejected", "ruled out", "unlike", "n't",
    )

    for marker in markers:
        for m in re.finditer(r"\b" + re.escape(marker), text):
            # Look back a short window for a negation cue. Stop at sentence
            # boundaries so a negation in a previous sentence doesn't leak.
            window = text[max(0, m.start() - 45):m.start()]
            window = re.split(r"[.;!?]", window)[-1]
            if any(re.search(r"\b" + re.escape(n), window) for n in negators):
                continue
            # Rejection cues can also FOLLOW the marker ("a young male reading
            # was ruled out"), so check a short forward window in the same
            # clause too.
            ahead = text[m.end():m.end() + 45]
            ahead = re.split(r"[.;!?]", ahead)[0]
            if re.search(r"\b(?:was |were |is |are )?(?:ruled out|rejected|discounted|dismissed)", ahead):
                continue
            return True
    return False
_ADULT_SIZED_MARKERS = ("adult-sized", "adult sized")


def _age_guard_lead_text(text):
    t = _AGE_GUARD_OTHER_VOICE_FIELDS_RE.sub("", str(text or ""))
    return re.sub(r"(?i)\bteen\s*/\s*adult\b", "adult", t)


def _apply_vocal_age_guard(lead_profile, analysis_text=""):
    """
    Nudge a post_puberty_male classification to adolescent_male_likely ONLY when
    the analysis text contains explicit adolescent / voice-change evidence, and
    never when there is settled adult-male evidence.

    CRITICAL: this must NOT fire on generic timbre or pitch adjectives
    ('bright', 'light', 'thin', 'high register', 'falsetto', etc.). The vocal
    prompts are explicit that pitch, brightness, thinness, and register are NOT
    age evidence. Using those words here was systematically demoting correctly
    identified adult male voices to adolescent on the strength of ordinary
    timbre description — the exact error the prompts warn against.

    'youthful' is deliberately back in the marker list below despite that
    history. It was dropped in the same pass as the pure-timbre words, but it
    isn't one: "bright"/"thin"/"light" describe acoustic PROPERTIES a mature
    voice can have too, while "youthful" is already an AGE JUDGEMENT the
    analysis chose to make (the same category as "boyish", which was kept) —
    removing it was collateral damage from fixing the timbre-word bug, not a
    deliberate tightening, and it left the gate narrow enough that the listening model's more
    natural age-descriptive language (which reads as "youthful-sounding"
    rather than the clinical "voice still changing") stopped being caught at
    all, under-firing on genuinely adolescent voices instead of over-firing on
    adult ones.
    """
    text = _vocal_text_without_template_labels(analysis_text)
    if AGE_GUARD_LEAD_FIELDS_ONLY:                                # 0.7.1-pre12 (F-23)
        text = _age_guard_lead_text(text)
    text = text.lower()

    # Explicit adolescent / voice-change evidence ONLY. No pure-timbre words —
    # but "youthful" and "boyish" are age judgements, not timbre properties,
    # so they stay.
    adolescent_markers = (
        "still changing", "voice changing", "voice change", "mid-change",
        "in transition", "transitional voice", "changing voice",
        "adolescent", "teen", "young male", "boyish",
        "youthful voice", "youthful-sounding voice", "youthful timbre",
        "youthful tone", "youthful male", "youthful-sounding male",
        "not yet settled", "not fully settled", "recently changed",
        "recently broken", "voice recently",
    )
    # Explicit statements that the voice has not finished changing. These are
    # direct age evidence -- the strongest this analysis can produce -- so they
    # override the settled-adult veto below rather than being cancelled by it.
    # Without this, a single incidental register word ("light tenor") was enough
    # to discard an explicit "the voice is still changing" reading.
    decisive_adolescent_markers = (
        "still changing", "voice changing", "changing voice", "voice change",
        "mid-change", "in transition", "transitional voice",
        "not yet settled", "not fully settled", "recently changed",
        "recently broken", "unbroken",
    )
    # Any settled adult-male evidence blocks a demotion.
    #
    # NOTE: this list must contain only MATURITY evidence. "tenor" and "chest
    # voice" were removed because they are tessitura/register labels, not age
    # markers -- adolescent male leads are routinely described as light tenors
    # singing in chest voice, and including them here vetoed almost every
    # correct demotion (the same pure-register error the docstring above warns
    # against, applied to the blocking side instead of the firing side).
    # "baritone" is kept: a settled baritone tessitura does imply a changed voice.
    adult_male_markers = (
        "post-pubert", "post puberty", "adult male", "male resonance",
        "male vocal weight", "baritone",
        "deepened voice", "mature male", "adult-sounding male",
        "settled adult", "full adult", "full chest", "deep chest",
        "heavy vocal weight", "settled chest", "grown", "man's voice",
        "man voice", "adult chest",
    )

    # Settled adult FEMALE evidence, so the child_female / child_gender_uncertain
    # branches below have a destination too. Previously only child_male_likely
    # was ever revisited, so an erroneous child-female read survived evidence
    # that would have overturned the male equivalent.
    adult_female_markers = (
        "adult female", "mature female", "adult woman", "grown woman",
        "woman's voice", "woman voice", "settled female", "female resonance",
        "adult-sounding female", "full female",
    )

    # Child labels: only move off 'child' when there is explicit adult
    # (or adolescent) evidence. Handled symmetrically across all three
    # child categories.
    if lead_profile in ("child_male_likely", "child_female_likely", "child_gender_uncertain"):
        has_adult_male = _marker_present_unnegated(text, adult_male_markers)
        has_adult_female = _marker_present_unnegated(text, adult_female_markers)
        has_adolescent = _marker_present_unnegated(text, adolescent_markers)

        if lead_profile == "child_male_likely" and has_adult_male:
            if has_adolescent:
                return "adolescent_male_likely"
            return "post_puberty_male"

        if lead_profile == "child_female_likely" and has_adult_female:
            return "female_teen_adult"

        if lead_profile == "child_gender_uncertain":
            # Only act when the evidence points one way; conflicting adult
            # male+female cues keep the conservative child tag.
            if has_adult_male and not has_adult_female:
                return "adolescent_male_likely" if has_adolescent else "post_puberty_male"
            if has_adult_female and not has_adult_male:
                return "female_teen_adult"

        return lead_profile

    # Adult male -> adolescent ONLY on explicit adolescent/changing evidence
    # with no settled-adult evidence present -- unless the analysis explicitly
    # states the voice is still changing, which outranks the veto.
    if lead_profile in ("post_puberty_male", "adult_male", "young_male"):
        has_adolescent = _marker_present_unnegated(text, adolescent_markers)
        # "adult-sized" is the vocal template's own FORMANT_NOTE answer for a
        # grown vocal tract, the counterpart of "adolescent-transitional"
        # (which already counts through "adolescent"), so it vetoes a
        # demotion here. Not used in the child branch above: it says nothing
        # about gender, so it cannot turn child_gender_uncertain male.
        has_settled = _marker_present_unnegated(text, adult_male_markers + _ADULT_SIZED_MARKERS)
        has_decisive = _marker_present_unnegated(text, decisive_adolescent_markers)
        if has_decisive or (has_adolescent and not has_settled):
            return "adolescent_male_likely"
    return lead_profile

def _refine_multi_singer_from_f0(vocal_objective_report, min_sep=3.0, min_ratio=0.2):
    """
    Parse the vocal objective report for the 'voiced-f0 multi-peak hint' line.
    If present, return a dict with suggestions for multi-voice fields.

    `min_sep`/`min_ratio` are now parameters, not fixed constants, because this
    function is used two ways with two very different risk profiles:

    1. REFINEMENT of a classification the listening model (or the confirmation pass) has already
       called mixed_leads -- adding detail to something already established.
       Called with the original, looser defaults (3 semitones, 0.2 ratio),
       exactly as before, so nothing about that path changes.

    2. INDEPENDENT TRIGGER, upgrading a SINGLE-voice classification to
       mixed_leads on this evidence alone, when the listening model's own pass reported only
       one voice — the case a male-verse/female-chorus track hits when the listening model's
       one holistic listen just doesn't notice or report the second voice
       (a small audio-language model missing a secondary register across
       sections is a common failure, not a corner case). Overriding the listening model's own
       classification is a bigger claim than adding detail to one it has already
       made, so the caller uses a stricter bar for this case -- see
       INDEPENDENT_MULTI_SINGER_MIN_SEP_SEMITONES /
       _MIN_RATIO below, tuned toward a real male/female split (typically
       7-12+ semitones apart) rather than the vibrato or dual-register
       artefacts a single voice can produce.
    """
    # A pitch histogram is only evidence about SINGERS when it was measured on
    # a real voice: from the isolated stem, and voiced for a meaningful share
    # of the audio. Two pitch centres in vocal-stem residue (instrument bleed)
    # turned an instrumental into "mixed_leads / duet_co_leads".
    if not _vocal_report_pitch_is_voice_evidence(vocal_objective_report):
        return {}
    if not vocal_objective_report:
        return {}
    lines = vocal_objective_report.splitlines()
    multi_hint = None
    for line in lines:
        if "voiced-f0 multi-peak hint" in line:
            multi_hint = line.strip()
            break
    if not multi_hint:
        return {}
    import re
    peaks_match = re.search(r'~(\d+)\s+pitch centres?', multi_hint)
    num_peaks = int(peaks_match.group(1)) if peaks_match else 0
    if num_peaks < 2:
        return {}
    sep_match = re.search(r'~(\d+(?:\.\d+)?)\s+semitones between top two', multi_hint)
    sep = float(sep_match.group(1)) if sep_match else 0.0
    ratio_match = re.search(r'secondary/primary mass ratio ([\d.]+)', multi_hint)
    ratio = float(ratio_match.group(1)) if ratio_match else 0.0
    if sep < float(min_sep) or ratio < float(min_ratio):
        return {}
    return {
        "num_distinct_voices": "2",
        "voice_arrangement": "duet_co_leads",
        "co_lead_detail": f"two distinct pitch centres (~{sep:.1f} semitones apart) from objective f0 analysis; likely two singers or a wide dual-register single voice",
        "multi_voice_evidence": multi_hint
    }

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
    if CHILD_FLIP_EVIDENCE_PROSE_ONLY:                            # 0.7.1-pre12 (F-22)
        text = _vocal_text_without_template_labels(confirm_text or "").lower()
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
    if CHILD_FLIP_EVIDENCE_PROSE_ONLY:                            # template labels and negations don't count
        return _marker_present_unnegated(text, child_evidence_markers)
    return any(marker in text for marker in child_evidence_markers)


def _apply_independent_f0_multi_singer(final_lead, initial_lead, vocal_objective_report,
                                       multi_voice_fields, vocal_text=None):
    """Consult the objective f0 histogram when the listening model reported a single voice.

    The two analysis paths used to disagree: the batch path upgraded to
    mixed_leads on the histogram alone, while the interactive path kept it as
    evidence but still wrote "[Upgraded from a single-voice listening-model
    classification...]" into the notes -- telling the writer a duet had been
    confirmed when it had not.

    Rule now used by both: upgrade only when neither vocal pass explicitly
    reported a single singer. When a pass did say one voice, the histogram is
    recorded honestly as supporting evidence (one singer's chest/head
    registers, octave doubling and harmonies all produce two pitch centres).
    """
    if not (ENABLE_INDEPENDENT_F0_MULTI_SINGER_TRIGGER and vocal_objective_report):
        return final_lead
    if (final_lead or initial_lead) in ("mixed_leads", ""):
        return final_lead
    try:
        f0_independent = _refine_multi_singer_from_f0(
            vocal_objective_report,
            min_sep=INDEPENDENT_MULTI_SINGER_MIN_SEP_SEMITONES,
            min_ratio=INDEPENDENT_MULTI_SINGER_MIN_RATIO,
        )
    except Exception:
        f0_independent = {}
    if f0_independent.get("num_distinct_voices") not in ("2", "3+"):
        return final_lead

    if VOCAL_F0_UPGRADE_GUARDS and _f0_sep_is_octave(_f0_hint_separation(vocal_objective_report)):
        multi_voice_fields["multi_voice_evidence"] = (
            (multi_voice_fields.get("multi_voice_evidence") or "")
            + " [Objective f0 histogram shows two centres about an octave apart. NOT upgraded: "
            "that is the signature of octave doubling or of the pitch tracker jumping an octave "
            "on one voice, not of two singers.]"
        ).strip()
        return final_lead
    said_single = (
        multi_voice_fields.get("num_distinct_voices") == "1"
        or multi_voice_fields.get("voice_arrangement") in ("solo_lead", "lead_plus_backing")
        or (VOCAL_F0_UPGRADE_GUARDS and _vocal_pass_says_single(vocal_text))
    )
    prior_evidence = multi_voice_fields.get("multi_voice_evidence") or ""
    if said_single:
        multi_voice_fields["multi_voice_evidence"] = (
            prior_evidence
            + " [Objective f0 histogram shows two well-separated pitch centres, but the "
            "listening passes reported a single lead. NOT upgraded: this pattern is also "
            "produced by one singer moving between registers, octave doubling or "
            "harmonies. Mention a second voice only if other evidence supports it.]"
        ).strip()
        return final_lead

    for k, v in f0_independent.items():
        if v:
            multi_voice_fields[k] = v
    multi_voice_fields["multi_voice_evidence"] = (
        prior_evidence
        + " [Upgraded to mixed leads on independent objective f0 evidence: the listening "
        "passes did not specify the number of voices, and the measured pitch histogram "
        "shows two well-separated, substantial centres consistent with two singers.]"
    ).strip()
    return "mixed_leads"


_SECTION_VOICE_HEAD = "SECTION VOICE CHECK"


def _adult_lead_gender(tag):
    t = str(tag or "")
    if t == "female_teen_adult":
        return "female"
    if t in ("post_puberty_male", "adult_male"):
        return "male"
    return None


def _section_voice_gate(final_lead, initial_lead, confirm_lead, vocal_objective_report):
    """The lead's gender when the section voice check should look further,
    else None: a settled adult lead both vocal passes agree on, and two
    pitch centres 4+ semitones apart that are not an octave."""
    if not SECTION_VOICE_CHECK:
        return None
    g = _adult_lead_gender(final_lead)
    if not g:
        return None
    for other in (initial_lead, confirm_lead):
        if other and other not in ("unknown", "unparsed") and _adult_lead_gender(other) != g:
            return None
    sep = _f0_hint_separation(vocal_objective_report)
    if sep is None or sep < SECTION_VOICE_MIN_SHIFT_ST or _f0_sep_is_octave(sep):
        return None
    return g


def _register_regions(times, midi, duration, gender, win=8.0, hop=4.0):
    """(lead median, [(start, end, median)], control (start, end) or None)
    from voiced pitch frames: stretches of SECTION_VOICE_MIN_REGION_S+ whose
    8-second windows sit SECTION_VOICE_MIN_SHIFT_ST+ below a female lead (at
    or below E4) or above a male lead (at or above C4)."""
    t = np.asarray(times, dtype=float)
    m = np.asarray(midi, dtype=float)
    if m.size < 200 or not duration:
        return None
    lead = float(np.median(m))
    fps = m.size / max(1e-9, float(duration))
    rows = []
    s0 = 0.0
    while s0 + win <= duration + 0.01:
        sel = (t >= s0) & (t < s0 + win)
        if int(sel.sum()) >= 0.15 * win * max(fps, 1.0):
            rows.append((s0, float(np.median(m[sel])), int(sel.sum())))
        s0 += hop

    def odd(med):
        if gender == "female":
            return med <= lead - SECTION_VOICE_MIN_SHIFT_ST and med <= 64.0     # E4
        return med >= lead + SECTION_VOICE_MIN_SHIFT_ST and med >= 60.0         # C4
    regs, cur = [], None
    for s_, med, _n in rows:
        if odd(med):
            if cur and s_ <= cur[1]:
                cur[1] = s_ + win
                cur[2].append(med)
            else:
                if cur:
                    regs.append(cur)
                cur = [s_, s_ + win, [med]]
        elif cur:
            regs.append(cur)
            cur = None
    if cur:
        regs.append(cur)
    regs = [(a, b, float(np.median(ms))) for a, b, ms in regs if b - a >= SECTION_VOICE_MIN_REGION_S]
    typical = [(n, s_) for s_, med, n in rows if abs(med - lead) <= 2.0
               and not any(a - win < s_ < b for a, b, _m in regs)]
    control = None
    if typical:
        _n, cs = max(typical)
        control = (cs, cs + win)
    return lead, regs, control


def _section_voice_clip_answer(reply):
    """{"lead", "conf", "g"} from the standard singer-confirmation answer for
    one clip; g is the voice's gender (boys and children count as male), or
    None for an uncertain, mixed or unparsed answer."""
    lead, conf = parse_vocal_confirmation(reply or "")
    g = _lead_gender(lead) if lead and lead not in ("mixed_leads", "uncertain", "unknown") else None
    return {"lead": lead or "", "conf": (conf or "").lower(), "g": g}


def _section_voice_decide(gender, regions, answers, control=None):
    """True when the clip answers show a second lead of the other gender:
    the main-lead clip heard as the lead's gender (medium or high
    confidence), and 2+ stretches heard as the other gender (and at least
    half of those asked), or a single stretch of
    SECTION_VOICE_SINGLE_REGION_MIN_S+ heard so with high confidence."""
    other = "male" if gender == "female" else "female"
    if not control or control.get("g") != gender or control.get("conf") not in ("medium", "high"):
        return False
    # A high male voice (falsetto, a boy-band tenor) is the likelier thing to
    # be heard as a woman, so a female stretch in a male-led track needs high
    # confidence; a male stretch in a female-led one needs medium or high.
    confs = ("medium", "high") if gender == "female" else ("high",)
    ok = [a.get("g") == other and a.get("conf") in confs for a in answers]
    n_ok = sum(ok)
    if n_ok >= 2 and n_ok * 2 >= len(answers):
        return True
    if (len(answers) == 1 and n_ok == 1 and answers[0].get("conf") == "high" and regions
            and regions[0][1] - regions[0][0] >= SECTION_VOICE_SINGLE_REGION_MIN_S):
        return True
    return False


def _mmss(x):
    if CLIP_TIMES_LIKE_SECTIONS:                 # 0.7.1-pre12 (F-15): as the section map prints it
        return _seconds_to_mmss(x)
    return f"{int(x // 60)}:{int(x % 60):02d}"


def _section_voice_check_legacy(moss_model, moss_processor, vocal_stem_path, final_lead, initial_lead,
                         confirm_lead, vocal_objective_report, mix_path=None):
    """See SECTION_VOICE_CHECK. Returns None when it did not run, else a dict
    with "mixed" (bool) and, when mixed, "detail", "evidence" and "block".
    Never raises."""
    try:
        gender = _section_voice_gate(final_lead, initial_lead, confirm_lead, vocal_objective_report)
        if not gender or moss_model is None or not vocal_stem_path or not os.path.exists(str(vocal_stem_path)):
            return None
        status("Section voice check: looking for a second lead voice of the other gender...")
        sr = 8000
        y, _ = librosa.load(str(vocal_stem_path), sr=sr, mono=True)
        duration = len(y) / float(sr)
        f0, _vflag, vprob = librosa.pyin(y, fmin=65.0, fmax=1000.0, sr=sr, frame_length=1024, hop_length=256)
        times = librosa.times_like(f0, sr=sr, hop_length=256)
        ok = np.isfinite(f0) & (np.nan_to_num(vprob) > 0.65)
        if int(ok.sum()) < 200:
            status_done("Section voice check: not enough voiced audio")
            return {"mixed": False, "summary": "inconclusive -- not enough voiced audio to check for another lead"}
        midi = 69.0 + 12.0 * np.log2(f0[ok] / 440.0)
        rr = _register_regions(times[ok], midi, duration, gender)
        if not rr:
            status_done("Section voice check: not enough voiced audio")
            return {"mixed": False, "summary": "inconclusive -- not enough voiced audio to check for another lead"}
        lead_m, regions, control = rr
        enough = len(regions) >= 2 or (regions and regions[0][1] - regions[0][0] >= SECTION_VOICE_SINGLE_REGION_MIN_S)
        if not enough or not control:
            status_done("Section voice check: no qualifying separate-register stretch; another lead not confirmed")
            return {"mixed": False, "summary": "inconclusive -- no qualifying separate-register stretch; another lead not confirmed"}
        asked = sorted(regions, key=lambda r: (-(r[1] - r[0]), r[0]))[:SECTION_VOICE_MAX_REGIONS]
        asked.sort(key=lambda r: r[0])
        # Clips of the full mix (what the listening model is used to hearing;
        # on an isolated vocal stem it answered "unclear" to everything), each
        # asked the scan's own singer-confirmation question on its own.
        src = mix_path if (mix_path and os.path.exists(str(mix_path))) else vocal_stem_path
        y16, sr16 = librosa.load(str(src), sr=16000, mono=True)

        def ask(a, b):
            mid = 0.5 * (a + b)
            half = min(SECTION_VOICE_CLIP_S, b - a) / 2.0
            st_ = max(0, int((mid - half) * sr16))
            audio = y16[st_:int((mid + half) * sr16)]
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp.close()
            try:
                import soundfile as _sf
                _sf.write(tmp.name, audio, sr16)
                reply = moss_generate(moss_model, moss_processor, [{
                    "role": "user",
                    "content": [{"type": "text", "text": VOCAL_CONFIRMATION_PROMPT},
                                {"type": "audio", "path": tmp.name}],
                }], max_new_tokens=256) or ""
            finally:
                try:
                    os.remove(tmp.name)
                except OSError:
                    pass
            return _section_voice_clip_answer(reply)
        c0, c1 = control
        ctrl_ans = ask(max(0.0, c0 - 2.0), min(duration, c1 + 2.0))
        answers = [ask(a, b) for a, b, _m in asked]
        mixed = _section_voice_decide(gender, asked, answers, ctrl_ans)
        if not mixed:
            _ans_s = ("main " + f"{_mmss(c0)}-{_mmss(c1)}: {ctrl_ans.get('lead') or '?'} "
                      f"({ctrl_ans.get('conf') or '?'}); " + "; ".join(
                          f"{_mmss(a)}-{_mmss(b)}: {x.get('lead') or '?'} ({x.get('conf') or '?'})"
                          for (a, b, _m), x in zip(asked, answers)))
            status_done("Section voice check: another lead not confirmed (" + _ans_s + ")")
            return {"mixed": False, "summary": "inconclusive -- another lead not confirmed (" + _ans_s + ")"}
        other = "male" if gender == "female" else "female"
        conf_spans = [(a, b) for (a, b, _m), x in zip(asked, answers) if x.get("g") == other]
        all_spans = [(a, b) for a, b, _m in regions]
        spans_s = ", ".join(f"{_mmss(a)}-{_mmss(b)}" for a, b in conf_spans)
        more = [f"{_mmss(a)}-{_mmss(b)}" for a, b in all_spans if (a, b) not in conf_spans]
        detail = (f"a {other} lead in {spans_s}" + (f" (and likely {', '.join(more)})" if more else "")
                  + f"; a {gender} lead elsewhere")
        evidence = (f"Section voice check: asked about each stretch on its own, the listening model heard a "
                    f"{other} singer in {spans_s} and a {gender} one in the main lead at "
                    f"{_mmss(control[0])}-{_mmss(control[1])}.")
        block = (
            f"\n\n{_SECTION_VOICE_HEAD} (measured on the separated vocal track, then heard by the listening "
            "model; read with the vocal blocks above):\n"
            f"- The lead is {gender} for most of the track, but the voice sits "
            f"{SECTION_VOICE_MIN_SHIFT_ST:g}+ semitones {'lower' if gender == 'female' else 'higher'} in "
            + ", ".join(f"{_mmss(a)}-{_mmss(b)}" for a, b in all_spans) + ".\n"
            f"- Asked about each stretch on its own, the listening model heard a {other} singer in "
            f"{spans_s}, and a {gender} singer in the main lead ({_mmss(control[0])}-{_mmss(control[1])}).\n"
            f"- FINAL LEAD PROFILE is therefore mixed_leads: a {gender} lead and a {other} lead who takes "
            "those stretches. When asked who sings, describe both voices and where each one leads; the "
            "whole-track vocal passes above heard only the main voice. Pitch alone did not decide this."
        )
        status_done(f"Section voice check: a {other} lead in {spans_s} -> mixed leads")
        return {"mixed": True, "detail": detail, "evidence": evidence, "block": block,
                "summary": f"checked -- a {other} lead in {spans_s} (see SECTION VOICE CHECK)"}
    except Exception as e:
        print(f"  (section voice check skipped: {e})")
        return None


def _section_voice_conflict(final_lead, initial_lead, confirm_lead):
    """Only opposing recognised voice genders trigger recovery, never age alone.
    A contradiction is a reason to listen again, not proof of two people.
    """
    if not SECTION_VOICE_CHECK or not SECTION_VOICE_CONFLICT_RECOVERY:
        return False
    if final_lead == "mixed_leads":
        return False
    genders = {_lead_gender(t) for t in (initial_lead, confirm_lead)
               if t and t not in ("unknown", "unparsed", "uncertain", "mixed_leads")}
    return "male" in genders and "female" in genders


def _section_voice_coverage_windows(duration, section_hints=None):
    """Bounded disjoint listening windows, with no pitch/voicing prerequisite.
    Start inside measured sections where possible; fill gaps across the timeline.
    Section labels are never used, and each clip must independently prove a lead.
    """
    import math
    limit = max(0, int(SECTION_VOICE_RECOVERY_MAX_CLIPS))
    width = float(SECTION_VOICE_RECOVERY_CLIP_S)
    if not math.isfinite(duration) or not math.isfinite(width) or duration < 6 or width < 6 or not limit:
        return []
    candidates = []
    for section in section_hints or []:
        try:
            a, b = float(section["start"]), float(section["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if not math.isfinite(a) or not math.isfinite(b):
            continue
        a, b = max(0.0, a), min(duration, b)
        if b - a >= 6:
            span = min(width, b - a)
            mid = (a + b) / 2
            candidates.append((mid - span / 2, mid + span / 2, 0.0))
    candidates.sort()
    # Spread a capped selection over the whole song, rather than keeping only
    # the first sections. Tiny/invalid/overlapping sections cannot add votes.
    if len(candidates) > limit:
        indices = ([len(candidates) // 2] if limit == 1 else
                   [round(i * (len(candidates) - 1) / (limit - 1)) for i in range(limit)])
        candidates = [candidates[i] for i in indices]
    chosen = []
    def add(a, b):
        if len(chosen) < limit and b - a >= 6 and not any(a < z and b > x for x, z, _ in chosen):
            chosen.append((a, b, 0.0))
    for a, b, _ in candidates:
        add(a, b)
    # No section map is required. Grid centres also cover unrepresented parts
    # of long sections; failed pitch tracking can never suppress this fallback.
    for i in range(limit):
        span = min(width, duration)
        mid = (i + .5) * duration / limit
        a = max(0.0, min(duration - span, mid - span / 2))
        add(a, a + span)
    return sorted(chosen)


def _section_voice_gap_windows(duration, regions, answers):
    """Sample unused audio, prioritising neighbours of a rare confident lead.
    Never overlap an existing vote or use section names, artist or pitch as evidence.
    """
    import math
    limit = max(0, int(SECTION_VOICE_RECOVERY_GAP_CLIPS))
    if not limit or not math.isfinite(duration) or duration < 6:
        return []
    counts = {"male": 0, "female": 0}
    anchors = []
    for (a, b, _), x in zip(regions, answers):
        g = x.get("g")
        if (g in counts and x.get("present") and x.get("conf") == "high"
                and not x.get("conflict")
                and x.get("arrangement") in ("solo_lead", "lead_plus_backing")):
            counts[g] += 1
            anchors.append((a, b, g))
    # This is not a search for a second singer on otherwise consistent tracks.
    if not all(counts.values()) or min(counts.values()) >= 2:
        return []
    rare = [(a, b) for a, b, g in anchors if counts[g] < 2]
    occupied = sorted((max(0., a), min(duration, b)) for a, b, _ in regions)
    gaps, end = [], 0.
    for a, b in occupied:
        if a - end >= 6:
            gaps.append((end, a))
        end = max(end, b)
    if duration - end >= 6:
        gaps.append((end, duration))
    candidates = []
    for a, b in gaps:
        # Tile a gap with nonoverlapping clips, including its far end.
        count = int((b - a) // 6)
        for i in range(count):
            start = a if count == 1 else a + i * (b - a - 6) / (count - 1)
            candidates.append((start, start + 6, 0.))
    def distance(window):
        a, b, _ = window
        return min(max(x - b, a - z, 0.) for x, z in rare), a
    return sorted(sorted(candidates, key=distance)[:limit])



# 0.7.1-pre12 CLIP_ANSWER_TOLERANT: a clip answer's value read past formatting only --
# a bullet, case, "different lead" for different_lead, a trailing full stop, a note in brackets.
def _clip_field_values(text, key, tolerant):
    if not tolerant:
        return None
    vals = re.findall(r"(?mi)^\s*(?:[-*\u2022]|\d+[.)])?\s*" + key + r"\s*[=:]\s*([^\n]*)", text)
    out = []
    for v in vals:
        v = v.split("(")[0].split("[")[0].strip().lower()
        v = re.sub(r"[\s\-]+", "_", v).strip("_\"'<>`")
        out.append(v)
    return out


def _clip_value_pick(value, allowed, synonyms=None):
    for a in sorted(allowed, key=len, reverse=True):
        if re.match(re.escape(a) + r"(?![a-z0-9_])", value):
            return a
    for a, canon in (synonyms or {}).items():
        if re.match(re.escape(a) + r"(?![a-z0-9_])", value):
            return canon
    return ""


def _section_voice_recovery_answer(reply):
    text = _strip_md_emphasis(reply)
    fields = {}
    allowed = {"VOCALS_PRESENT": {"yes", "no", "uncertain"},
               "LEAD_GENDER": {"male", "female", "mixed", "uncertain"},
               "VOICE_ARRANGEMENT": {"solo_lead", "lead_plus_backing", "duet_co_leads", "group_unison", "uncertain"},
               "CONFIDENCE": {"high", "medium", "low"}}
    for key, values in allowed.items():
        tol = _clip_field_values(text, key, CLIP_ANSWER_TOLERANT)
        if tol is not None:                                       # 0.7.1-pre12
            value = _clip_value_pick(tol[0], values, _CLIP_ARRANGEMENT_SYNONYMS if key == "VOICE_ARRANGEMENT"
                                     else None) if len(tol) == 1 else ""
            fields[key] = value or "uncertain"
            continue
        found = re.findall(r"(?mi)^\s*" + key + r"\s*[=:]\s*([^\n]*)", text)
        value = found[0].strip().lower() if len(found) == 1 else ""
        fields[key] = value if value in values else "uncertain"
    g = fields["LEAD_GENDER"]
    present = fields["VOCALS_PRESENT"] == "yes"
    return {"g": g if present and g in ("male", "female") else None,
            "lead": g if present else "no clear foreground lead",
            "conf": fields["CONFIDENCE"], "present": present,
            "conflict": g == "mixed", "arrangement": fields["VOICE_ARRANGEMENT"]}


def _section_voice_recovery_groups(regions, answers):
    """Repeated high-confidence opposite classifications, in disjoint clips.
    Empty/low-confidence, contradictory, mixed and backing-only answers do not vote.
    """
    groups = {"male": [], "female": []}
    if len(regions) != len(answers):
        return {}
    used = []
    for (a, b, _m), answer in zip(regions, answers):
        g = answer.get("g")
        if (g not in groups or not answer.get("present") or answer.get("conf") != "high" or answer.get("conflict")
                or answer.get("arrangement") not in ("solo_lead", "lead_plus_backing")
                or b - a < 6 or any(a < z and b > x for x, z in used)):
            continue
        groups[g].append((float(a), float(b)))
        used.append((a, b))
    return groups if all(len(groups[g]) >= 2 for g in groups) else {}


def _section_voice_pair_confirmed(reply):
    # Require an exact field, a stated lead change (not backing/harmony), and
    # high confidence. Never accept a phrase merely containing "different".
    fields = {}
    allowed = {"LEAD_CHANGE": ("same_lead", "different_lead", "uncertain"), "CONFIDENCE": ("high", "medium", "low")}
    for key in ("LEAD_CHANGE", "CONFIDENCE"):
        values = _clip_field_values(_strip_md_emphasis(reply), key, CLIP_ANSWER_TOLERANT)
        if values is not None:                                    # 0.7.1-pre12
            values = [_clip_value_pick(v, allowed[key]) for v in values]
        else:
            values = re.findall(r"(?mi)^\s*" + key + r"\s*[=:]\s*([a-z_]+)\s*$", _strip_md_emphasis(reply))
        if len(values) != 1:
            return False
        fields[key] = values[0].lower()
    return fields == {"LEAD_CHANGE": "different_lead", "CONFIDENCE": "high"}


def _identity_pitch_median(median, report, initial, confirmed, confidence):
    """A rejected tracker reading cannot veto two agreeing adult-male passes.
    Keep the raw measurement for reporting and all section-voice checks.
    """
    if (initial == confirmed == "post_puberty_male" and confidence == "high"
            and _pyin_reading_confidence_note(report)):
        return None
    return median


def _audio_only_singer_identity(audit):
    """Record the accepted acoustic category without inventing a named person."""
    t = str(audit or "")
    if ("VOCAL PRESENCE VERIFICATION: UNRESOLVED" in t
            or "No clear vocals detected." in t):
        return ""
    m = re.search(r"(?mi)^-?\s*FINAL LEAD PROFILE:\s*([a-z_]+)\s*$", t)
    if not m or m.group(1) not in ("post_puberty_male", "female_teen_adult", "mixed_leads",
                                  "adolescent_male_likely", "child_male_likely", "child_female_likely"):
        return ""
    tag = m.group(1)
    count = "several" if tag == "mixed_leads" else "uncertain"
    return (f"SINGER_IDENTITY={tag}\n"
            "REASONING=Accepted audio vocal profile; no artist metadata or cover evidence is available. "
            "This describes the audible voice category, not a named performer or documented singer count.\n"
            "CONFIDENCE=medium\n"
            f"LEAD_SINGERS={count}\nIDENTITY_SOURCE=audio_only")


def _section_voice_subdivide_minority(regions, answers, clip, ask):
    """Recheck one rare lead as two disjoint clips; never count its parent too.
    This validates a brief appearance without demanding two separate verses.
    """
    eligible = {"male": [], "female": []}
    for i, ((a, b, _), x) in enumerate(zip(regions, answers)):
        if (x.get("g") in eligible and x.get("present") and x.get("conf") == "high"
                and not x.get("conflict")
                and x.get("arrangement") in ("solo_lead", "lead_plus_backing")):
            eligible[x["g"]].append(i)
    rare = [g for g in eligible if len(eligible[g]) == 1
            and len(eligible["female" if g == "male" else "male"]) >= 2]
    if len(rare) != 1:
        return regions, answers, ""
    g = rare[0]
    index = eligible[g][0]
    a, b, m = regions[index]
    if b - a < 12:
        return regions, answers, "minority sample too short for two disjoint checks"
    mid = (a + b) / 2.
    halves = [(a, mid, m), (mid, b, m)]
    checked = []
    for x, z, _ in halves:
        answer = _section_voice_recovery_answer(ask(clip(x, z), SECTION_VOICE_CLIP_PROMPT))
        checked.append(answer)
    if not all(x.get("g") == g and x.get("present") and x.get("conf") == "high"
               and not x.get("conflict") and x.get("arrangement") in ("solo_lead", "lead_plus_backing")
               for x in checked):
        return regions, answers, "minority subdivision did not corroborate both halves"
    return (regions[:index] + halves + regions[index + 1:],
            answers[:index] + checked + answers[index + 1:],
            "minority lead corroborated in two disjoint halves; parent sample replaced")


def _section_voice_context_recheck(regions, answers, clip, stem_clip, ask, duration, section_hints=None):
    """Corroborate a rare voice with longer context, without weakening votes."""
    def eligible(x):
        return (x.get("present") and x.get("conf") == "high" and not x.get("conflict")
                and x.get("arrangement") in ("solo_lead", "lead_plus_backing"))
    counts = {g: sum(eligible(x) and x.get("g") == g for x in answers) for g in ("male", "female")}
    if min(counts.values()) != 1 or max(counts.values()) < 2:
        return regions, answers, ""
    minority = min(counts, key=counts.get)
    i = next(i for i, x in enumerate(answers) if eligible(x) and x.get("g") == minority)
    a, b, m = regions[i]
    # Two disjoint eight-to-twelve-second checks retain phrase context.
    # Each must independently agree in both the mix and vocal stem.
    lo, hi = a, min(duration, a + 24.)
    for sec in section_hints or []:
        start, end = float(sec["start"]), float(sec["end"])
        overlap = max(0., min(b, end) - max(a, start))
        if start <= (a + b) / 2. < end and overlap >= .75 * (b - a):
            length = min(24., end - start)
            lo = min(max(start, a), end - length)
            hi = lo + length
            break
    if hi - lo < 16.:
        return regions, answers, "minority context too short for two eight-second checks"
    mid = (lo + hi) / 2.
    halves = [(lo, mid, m), (mid, hi, m)]
    checked = []
    for x, z, _ in halves:
        pair = [_section_voice_recovery_answer(ask(audio, SECTION_VOICE_CLIP_PROMPT))
                for audio in (clip(x, z), stem_clip(x, z))]
        if not all(eligible(v) and v.get("g") == minority for v in pair):
            return regions, answers, "longer-context mix/stem checks did not corroborate the minority lead"
        checked.append(pair[0])
    # Replace every overlapping old sample: no parent or contrary overlapping
    # observation is counted alongside the new evidence.
    kept = [(r, v) for r, v in zip(regions, answers) if r[1] <= lo or r[0] >= hi]
    result = sorted(kept + list(zip(halves, checked)), key=lambda item: item[0])
    rr, aa = [r for r, v in result], [v for r, v in result]
    if not _section_voice_recovery_groups(rr, aa):
        return regions, answers, "longer context lacked disjoint majority control samples"
    return rr, aa, "minority lead corroborated in mix and stem on two disjoint longer-context clips"


def _section_voice_short_recheck(regions, answers, clip, stem_clip, ask):
    eligible = lambda x: (x.get("present") and x.get("conf") == "high" and not x.get("conflict")
                          and x.get("arrangement") in ("solo_lead", "lead_plus_backing"))
    counts = {g: sum(eligible(x) and x.get("g") == g for x in answers) for g in ("male", "female")}
    if max(counts.values()) < 2:
        return regions, answers, ""
    majority = max(counts, key=counts.get)
    minority = "female" if majority == "male" else "male"
    # Examine short or uncertain passages first; no successful mixed result reaches here.
    candidates = [i for i, ((a,b,_), x) in enumerate(zip(regions, answers))
                  if b-a >= 12 and x.get("present") and (x.get("g") == minority or not eligible(x))]
    candidates.sort(key=lambda i: (not (eligible(answers[i]) and answers[i].get("g") == minority), i))
    for i in candidates[:3]:
        a,b,m = regions[i]; mid=(a+b)/2
        halves=[(a,mid,m),(mid,b,m)]; checked=[]
        for lo,hi,_ in halves:
            pair = [_section_voice_recovery_answer(ask(audio, SECTION_VOICE_CLIP_PROMPT))
                    for audio in (clip(lo,hi), stem_clip(lo,hi))]
            if not all(eligible(x) and x.get("g") == minority for x in pair):
                break
            checked.append(pair[0])
        if len(checked) == 2:
            return (regions[:i]+halves+regions[i+1:], answers[:i]+checked+answers[i+1:],
                    "short minority passage corroborated in mix and vocal stem on two disjoint halves")
    return regions, answers, "short passage mix/stem recheck did not confirm another lead"


def _harmony_clips_one_lead(answers, pair_reply, had_conflict, whole_track_lead):
    """0.7-pre5 SECTION_VOICE_HARMONY_CLIPS: a short reason when sampled clips
    that disagree on the singer's sex still show ONE lead, or "". All of:
    the whole-track passes agree (no conflict) on a sex that matches the
    majority of the confident clip readings; a direct comparison heard the
    same singer with high confidence; the other-sex readings are a small
    minority (at most a quarter, at most three) and every one of them had
    backing harmonies; and no more than one clip heard co-leads."""
    if not SECTION_VOICE_HARMONY_CLIPS or had_conflict:
        return ""
    wt = str(whole_track_lead or "").lower()
    whole = "male" if wt in ("post_puberty_male", "adult_male", "young_male") else (
        "female" if wt in ("female_teen_adult", "adult_female", "young_female") else "")
    if not whole:
        return ""
    fields = {}
    for key in ("LEAD_CHANGE", "CONFIDENCE"):
        values = re.findall(r"(?mi)^\s*" + key + r"\s*[=:]\s*([a-z_]+)", _strip_md_emphasis(str(pair_reply or "")))
        if len(values) != 1:
            return ""
        fields[key] = values[0].lower()
    if fields != {"LEAD_CHANGE": "same_lead", "CONFIDENCE": "high"}:
        return ""
    votes = [x for x in answers if x.get("present") and x.get("conf") == "high" and not x.get("conflict")
             and x.get("arrangement") in ("solo_lead", "lead_plus_backing") and x.get("g") in ("male", "female")]
    if sum(1 for x in answers if x.get("arrangement") == "duet_co_leads" and x.get("conf") == "high") > 1:
        return ""
    major = [x for x in votes if x["g"] == whole]
    minor = [x for x in votes if x["g"] != whole]
    if not minor or len(major) < 3 or len(minor) > 3 or len(minor) * 4 > len(votes):
        return ""
    if any(x.get("arrangement") != "lead_plus_backing" for x in minor):
        return ""
    if HARMONY_CLIPS_SCATTERED_ONLY:
        # 0.7-pre8: two other-sex readings back to back are a passage sung by
        # someone else (a second singer's verse), not harmonies heard in passing.
        _other = [x["g"] != whole for x in votes]
        if any(a and b for a, b in zip(_other, _other[1:])):
            return ""
    return (f"{len(minor)} of {len(votes)} confident clips heard a {'female' if whole == 'male' else 'male'} "
            "voice, all with backing harmonies, and a direct comparison heard the same singer")


def _section_voice_recovery_check(moss_model, moss_processor, vocal_stem_path, mix_path,
                                  section_hints=None, had_conflict=False, whole_track_lead=None):
    """Listen across the full mix; pitch and vocal-stem availability are optional.
    Clear foreground singing, repeated clip readings and pair checks are required.
    """
    prefix = ("inconclusive -- conflicting whole-track voices" if had_conflict else
              "inconclusive -- no reliable sampled lead voices")
    failure = {"mixed": False, "summary": prefix + "; another lead not confirmed"}
    if moss_model is None or not mix_path or not os.path.exists(str(mix_path)):
        failure["summary"] += " (listening model or full-mix audio unavailable)"
        return failure
    status("Section voice check: listening across the track...")
    mix, rate = librosa.load(str(mix_path), sr=16000, mono=True)
    regions = _section_voice_coverage_windows(len(mix) / float(rate), section_hints)
    if not regions:
        failure["summary"] += " (no usable listening windows)"
        return failure

    def clip(a, b):
        return mix[max(0, int(a * rate)):min(len(mix), int(b * rate))]

    def ask(audio, prompt, budget=256):
        import soundfile as sf
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.close()
        try:
            sf.write(tmp.name, audio, rate)
            return moss_generate(moss_model, moss_processor, [{"role": "user", "content": [
                {"type": "text", "text": prompt}, {"type": "audio", "path": tmp.name}]}],
                max_new_tokens=budget) or ""
        finally:
            try:
                os.remove(tmp.name)
            except OSError:
                pass

    answers = []
    for a, b, _m in regions:
        audio = clip(a, b)
        # Digital/near silence needs no model call. Music without voice is
        # handled by the prompt's explicit foreground-vocal presence field.
        if not len(audio) or float(np.max(np.abs(audio))) < 1e-5:
            answer = _section_voice_recovery_answer("VOCALS_PRESENT=no")
        else:
            answer = _section_voice_recovery_answer(ask(audio, SECTION_VOICE_CLIP_PROMPT))
        answers.append(answer)
    presence_diagnostic = {}
    # Lead absence is only a trigger, never evidence that all vocals are absent.
    # Any sampled vocal presence (including uncertain-gender/group voices) vetoes it.
    if (INSTRUMENTAL_FULL_COVERAGE_CHECK and len(answers) >= 4
            and all(not x.get("present") for x in answers)):
        try:
            presence = _instrumental_full_coverage(mix, rate, ask, presence_diagnostic)
        except Exception as exc:
            presence = None
            presence_diagnostic.update(outcome="unresolved", detail="verification failed: " + type(exc).__name__)
        if presence:
            return {"mixed": False, "summary": "checked -- full-mix presence verification: instrumental",
                    "presence_arbitration": presence}
    # Preserve every already-successful v2/v3 decision and all original votes.
    # Only a shortage of repeated, contradictory readings enables extra sampling.
    if not _section_voice_recovery_groups(regions, answers):
        extra = _section_voice_gap_windows(len(mix) / float(rate), regions, answers)
        for a, b, m in extra:
            audio = clip(a, b)
            if not len(audio) or float(np.max(np.abs(audio))) < 1e-5:
                answer = _section_voice_recovery_answer("VOCALS_PRESENT=no")
            else:
                answer = _section_voice_recovery_answer(ask(audio, SECTION_VOICE_CLIP_PROMPT))
            regions.append((a, b, m))
            answers.append(answer)
        ordered = sorted(zip(regions, answers), key=lambda item: item[0])
        regions = [item[0] for item in ordered]
        answers = [item[1] for item in ordered]
    subdivision_log = ""
    if not _section_voice_recovery_groups(regions, answers):
        regions, answers, subdivision_log = _section_voice_subdivide_minority(regions, answers, clip, ask)
    if (not _section_voice_recovery_groups(regions, answers)
            and (had_conflict or "minority subdivision did not corroborate" in subdivision_log)
            and vocal_stem_path and os.path.exists(str(vocal_stem_path))):
        try:
            stem, stem_rate = librosa.load(str(vocal_stem_path), sr=rate, mono=True)
            if abs(len(stem) / float(stem_rate) - len(mix) / float(rate)) <= 0.25:
                def stem_clip(a, b):
                    return stem[max(0, int(a * rate)):min(len(stem), int(b * rate))]
                regions, answers, short_log = _section_voice_short_recheck(regions, answers, clip, stem_clip, ask)
                subdivision_log = (subdivision_log + "; " + short_log).strip("; ")
                if not _section_voice_recovery_groups(regions, answers):
                    regions, answers, context_log = _section_voice_context_recheck(
                        regions, answers, clip, stem_clip, ask, len(mix) / float(rate), section_hints)
                    subdivision_log = (subdivision_log + "; " + context_log).strip("; ")
        except Exception:
            pass  # Optional recovery cannot erase existing evidence.
    heard = {x.get("g") for x in answers if x.get("present") and x.get("conf") == "high"
             and x.get("arrangement") in ("solo_lead", "lead_plus_backing") and x.get("g")}
    if len(heard) > 1:
        prefix = "inconclusive -- sampled lead voices disagree"
    elif heard and not had_conflict:
        prefix = "sampled -- no different lead confirmed"
    sample_log = "; samples: " + "; ".join(
        f"{_mmss(a)}-{_mmss(b)} {x.get('lead') or 'uncertain'} ({x.get('conf') or 'unknown confidence'}; {x.get('arrangement') or 'unknown arrangement'})"
        for (a, b, _), x in zip(regions, answers))
    failure["summary"] = prefix + sample_log
    if subdivision_log:
        failure["summary"] += "; " + subdivision_log
    if presence_diagnostic:
        failure["summary"] += "; presence verification " + presence_diagnostic.get("outcome", "unresolved") + ": " + presence_diagnostic.get("detail", "")
        failure["presence_diagnostic"] = presence_diagnostic
        failure["presence_unresolved"] = presence_diagnostic.get("outcome") == "unresolved"
    groups = _section_voice_recovery_groups(regions, answers)
    if not groups:
        return failure
    # Two comparisons must corroborate the clip readings. Opposite apparent
    # genders/pitches alone never establish different people.
    prompt = SECTION_VOICE_PAIR_PROMPT
    for index in range(2):
        a, b = groups["female"][index]
        c, d = groups["male"][index]
        pair = np.concatenate((clip(a, b), np.zeros(rate, dtype=mix.dtype), clip(c, d)))
        pair_reply = ask(pair, prompt)
        if not _section_voice_pair_confirmed(pair_reply):
            diagnostic = " ".join(str(pair_reply).split())[:300] or "empty answer"
            failure["summary"] += f"; direct comparison {index + 1} did not confirm a different lead: {diagnostic}"
            _one = _harmony_clips_one_lead(answers, pair_reply, had_conflict, whole_track_lead)
            if _one and failure["summary"].startswith("inconclusive -- sampled lead voices disagree"):
                failure["summary"] = ("sampled -- no different lead confirmed (" + _one + ")"
                                      + failure["summary"][len("inconclusive -- sampled lead voices disagree"):])
            return failure
    female, male = groups["female"], groups["male"]
    def spans(items):
        return ", ".join(f"{_mmss(a)}-{_mmss(b)}" for a, b in items)
    detail = f"female lead heard in sampled clips {spans(female)}; male lead heard in sampled clips {spans(male)}"
    evidence = "Repeated voiced clips and two direct comparisons confirmed different foreground lead voices. " + detail
    # Keep the legacy sentence readable by existing Musiclyse/load/export code;
    # explicitly restrict its interpretation to the sampled intervals.
    block = ("\n\nSECTION VOICE CHECK (measured clip times, then heard by the listening model):\n"
             "- SAMPLED_LEADS_ONLY=yes\n"
             f"- FEMALE_LEAD_SAMPLES={spans(female)}\n"
             f"- MALE_LEAD_SAMPLES={spans(male)}\n"
             f"- Asked about each stretch on its own, the listening model heard a male singer in {spans(male)}, "
             f"and a female singer in the main lead ({_mmss(female[0][0])}-{_mmss(female[0][1])}).\n"
             "- Two direct clip comparisons confirmed different foreground leads, rather than a register "
             "change, doubling or backing harmonies. FINAL LEAD PROFILE=mixed_leads.\n"
             "- These times are sampled listening windows, NOT singer-change or section boundaries. "
             "Neither singer is established as the majority lead. Do not assign untested passages to either "
             "voice or describe the whole-track pitch range as one person's range. Earlier whole-track "
             "single-lead claims are superseded. The singers' ages are not established by these clips.")
    return {"mixed": True, "detail": detail, "evidence": evidence, "block": block,
            "summary": "confirmed -- different leads in sampled clips (see SECTION VOICE CHECK)"}


def _section_voice_check(moss_model, moss_processor, vocal_stem_path, final_lead, initial_lead,
                         confirm_lead, vocal_objective_report, mix_path=None, section_hints=None,
                         vocals_present=""):
    conflict = _section_voice_conflict(final_lead, initial_lead, confirm_lead)
    coverage = (SECTION_VOICE_CHECK and SECTION_VOICE_COVERAGE_CHECK
                and (vocals_present in ("yes", "uncertain")
                     or (not vocals_present and final_lead in (None, "", "unknown", "unparsed", "uncertain")))
                and final_lead != "mixed_leads")
    # An explicit no-vocals decision must never trigger recovery.
    if vocals_present == "no":
        return None
    legacy = None
    if not conflict:
        legacy = _section_voice_check_legacy(moss_model, moss_processor, vocal_stem_path, final_lead,
                                             initial_lead, confirm_lead, vocal_objective_report, mix_path)
        if not coverage or (legacy and legacy.get("mixed")):
            return legacy
    try:
        result = _section_voice_recovery_check(moss_model, moss_processor, vocal_stem_path, mix_path,
                                               section_hints, conflict, whole_track_lead=final_lead)
        status_done("Section voice check: " + result["summary"].split("; samples:")[0])
        return result
    except Exception as exc:
        status_done("Section voice check: coverage check unavailable")
        reason = "conflicting whole-track voices" if conflict else "no reliable sampled lead voices"
        return {"mixed": False, "summary": "inconclusive -- " + reason
                + "; recovery failed (" + type(exc).__name__ + ")"}


_SV_BLOCK_RE = re.compile(r"SECTION VOICE CHECK \(measured[^\n]*\n[\s\S]*?heard a (male|female) singer in ([^\n]*?), and a "
                          r"(male|female) singer in the main lead")
_SV_NOT_UPGRADED_RE = re.compile(r"\s*\[Objective f0 histogram shows two well-separated pitch centres, but the listening "
                                 r"passes reported a single lead\. NOT upgraded:[^\]]*\]")


def _unbroken_colead_prompt_note__pre6(vocal_text, background):
    """Target identity resolution only; leave adult co-lead acoustic checks alone."""
    if not PRESERVE_UNBROKEN_COLEAD_IDENTITY:
        return ""
    young = re.search(r"(?mi)(?:Initial isolated lead profile:|FINAL LEAD PROFILE:|SINGER_IDENTITY=)\s*"
                      r"(?:child_male_likely|adolescent_male_likely)\b", str(vocal_text or ""))
    unbroken = re.search(r"(?i)\bunbroken\b|\bboy[- ]soprano\b|\bvoice (?:that )?(?:had|has) not (?:yet )?broken\b",
                         str(background or ""))
    acoustic_mark = "VOICE MATURITY: the voice sounds UNBROKEN" in str(vocal_text or "")
    if not (young or unbroken or acoustic_mark):
        return ""
    return (
        "\n\nUNBROKEN VOICE WITH CO-LEADS (identity, not a new acoustic decision):\n"
        "Resolve each credited singer separately. An unbroken male voice can be heard as female "
        "in the same track as a real female co-lead. Similar pitch, female-sounding clips or the "
        "presence of a female guest do not change a documented male singer into a woman. Conversely, "
        "a male artist credit does not make a documented female co-lead male. Preserve both supported "
        "identities and any supported unbroken-voice evidence; do not assign identities to sampled "
        "windows merely from their perceived gender. A youthful age alone does not prove an unbroken voice. "
        "Group membership, a feature credit or hypothetical wording alone does not prove either lead.\n"
        "ONLY when THIS recording has both a documented unbroken male lead and a documented female "
        "co-lead, use SINGER_IDENTITY=mixed_leads and LEAD_SINGERS=several. In addition to the normal "
        "fields, append all six fields below, with concrete track/recording-era evidence for each basis. "
        "Omit these fields if either performer, role or voice maturity is uncertain. Never copy the "
        "placeholders or infer a fact just to fill them:\n"
        "UNBROKEN_MALE_LEAD_CONFIRMED=yes\nUNBROKEN_MALE_LEAD_NAME=<verified performer name>\n"
        "UNBROKEN_MALE_LEAD_BASIS=<supported male lead identity and unbroken voice at this recording>\n"
        "FEMALE_COLEAD_CONFIRMED=yes\nFEMALE_COLEAD_NAME=<verified different performer name>\n"
        "FEMALE_COLEAD_BASIS=<supported female co-lead identity for this recording>\n"
        "These are reference-based identity findings, not audio-only evidence or inferred changeover times."
    )


def _unbroken_colead_prompt_note__07(vocal_text, background):
    """Target identity resolution only; leave adult co-lead acoustic checks alone."""
    if not PRESERVE_UNBROKEN_COLEAD_IDENTITY:
        return ""
    young = re.search(r"(?mi)(?:Initial isolated lead profile:|FINAL LEAD PROFILE:|SINGER_IDENTITY=)\s*"
                      r"(?:child_male_likely|adolescent_male_likely)\b", str(vocal_text or ""))
    unbroken = re.search(r"(?i)\bunbroken\b|\bboy[- ]soprano\b|\bvoice (?:had|has) not (?:yet )?broken\b",
                         str(background or ""))
    acoustic_mark = "VOICE MATURITY: the voice sounds UNBROKEN" in str(vocal_text or "")
    if not (young or unbroken or acoustic_mark):
        return ""
    return (
        "\n\nUNBROKEN VOICE WITH CO-LEADS (identity, not a new acoustic decision):\n"
        "Resolve each credited singer separately. An unbroken male voice can be heard as female "
        "in the same track as a real female co-lead. Similar pitch, female-sounding clips or the "
        "presence of a female guest do not change a documented male singer into a woman. Conversely, "
        "a male artist credit does not make a documented female co-lead male. Preserve both supported "
        "identities and any supported unbroken-voice evidence; do not assign identities to sampled "
        "windows merely from their perceived gender. A youthful age alone does not prove an unbroken voice. "
        "Group membership, a feature credit or hypothetical wording alone does not prove either lead.\n"
        "ONLY when THIS recording has both a documented unbroken male lead and a documented female "
        "co-lead, use SINGER_IDENTITY=mixed_leads and LEAD_SINGERS=several. In addition to the normal "
        "fields, append all six fields below, with concrete track/recording-era evidence for each basis. "
        "Omit these fields if either performer, role or voice maturity is uncertain. Never copy the "
        "placeholders or infer a fact just to fill them:\n"
        "UNBROKEN_MALE_LEAD_CONFIRMED=yes\nUNBROKEN_MALE_LEAD_NAME=<verified performer name>\n"
        "UNBROKEN_MALE_LEAD_BASIS=<supported male lead identity and unbroken voice at this recording>\n"
        "FEMALE_COLEAD_CONFIRMED=yes\nFEMALE_COLEAD_NAME=<verified different performer name>\n"
        "FEMALE_COLEAD_BASIS=<supported female co-lead identity for this recording>\n"
        "These are reference-based identity findings, not audio-only evidence or inferred changeover times."
    )


def _unbroken_colead_prompt_note(*args, **kwargs):
    """SINGER_IDENTITY_PRE6 picks the 0.7.1-pre6 version; False gives 0.7's."""
    return (_unbroken_colead_prompt_note__pre6 if SINGER_IDENTITY_PRE6 else _unbroken_colead_prompt_note__07)(*args, **kwargs)


def _unbroken_colead_identity__pre6(identity_text):
    """Retain explicitly grounded per-performer findings, never infer from a pitch tag.
    No fields -> exact no-op. Keep the acoustic pipeline and its sample map unchanged.
    """
    if not PRESERVE_UNBROKEN_COLEAD_IDENTITY:
        return identity_text
    text = str(identity_text or "")
    def field(key):
        hits = re.findall(r"(?mi)^" + key + r"\s*[=:]\s*([^\n]*)", text)
        return hits[0].strip() if len(hits) == 1 else ""
    if any(field(k).lower() != "yes" for k in ("UNBROKEN_MALE_LEAD_CONFIRMED", "FEMALE_COLEAD_CONFIRMED")):
        return identity_text
    keys = ("UNBROKEN_MALE_LEAD_NAME", "FEMALE_COLEAD_NAME", "UNBROKEN_MALE_LEAD_BASIS", "FEMALE_COLEAD_BASIS")
    values = [field(k) for k in keys]
    if any(not v or re.search(r"[<>|]|\b(?:unknown|uncertain|unverified|placeholder)\b", v, re.I) for v in values):
        return identity_text
    if values[0].casefold() == values[1].casefold():
        return identity_text
    if any(name.casefold() not in basis.casefold() for name, basis in zip(values[:2], values[2:])):
        return identity_text
    if not re.search(r"(?i)\bunbroken\b|\bvoice (?:that )?(?:had|has) not (?:yet )?broken\b", values[2]):
        return identity_text
    if any(re.search(r"(?i)\b(?:no evidence|not confirmed|not established|hypothetical|possibly)\b", v) for v in values[2:]):
        return identity_text
    if re.search(r"(?i)\b(?:not|never|without)\b.{0,30}\bunbroken\b", values[2]):
        return identity_text
    # Do not manufacture a valid identity result from an otherwise malformed reply.
    if not re.search(r"(?m)^SINGER_IDENTITY\s*[=:]\s*\S+", text):
        return identity_text
    out = re.sub(r"(?m)^(SINGER_IDENTITY\s*[=:]\s*)\S+", r"\g<1>mixed_leads", text, count=1)
    if re.search(r"(?m)^LEAD_SINGERS\s*[=:]", out):
        out = re.sub(r"(?m)^(LEAD_SINGERS\s*[=:]\s*)\S+", r"\g<1>several", out, count=1)
    else:
        out += "\nLEAD_SINGERS=several"
    if "UNBROKEN_MALE_IDENTITY_PRESERVED=yes" not in out:
        out = out.rstrip() + "\nUNBROKEN_MALE_IDENTITY_PRESERVED=yes\n"
    if "COLEAD_IDENTITY_SCOPE=" not in out:
        out += ("COLEAD_IDENTITY_SCOPE=The identity findings retain an unbroken male lead AND a female "
                "co-lead. Female-sounding samples do not establish which person sang them; do not "
                "relabel both people female or assign unverified passage identities.\n")
    return out


def _unbroken_colead_identity__07(identity_text):
    """Retain explicitly grounded per-performer findings, never infer from a pitch tag.
    No fields -> exact no-op. Keep the acoustic pipeline and its sample map unchanged.
    """
    if not PRESERVE_UNBROKEN_COLEAD_IDENTITY:
        return identity_text
    text = str(identity_text or "")
    def field(key):
        hits = re.findall(r"(?mi)^" + key + r"\s*[=:]\s*([^\n]*)", text)
        return hits[0].strip() if len(hits) == 1 else ""
    if any(field(k).lower() != "yes" for k in ("UNBROKEN_MALE_LEAD_CONFIRMED", "FEMALE_COLEAD_CONFIRMED")):
        return identity_text
    keys = ("UNBROKEN_MALE_LEAD_NAME", "FEMALE_COLEAD_NAME", "UNBROKEN_MALE_LEAD_BASIS", "FEMALE_COLEAD_BASIS")
    values = [field(k) for k in keys]
    if any(not v or re.search(r"[<>|]|\b(?:unknown|uncertain|unverified|placeholder)\b", v, re.I) for v in values):
        return identity_text
    if values[0].casefold() == values[1].casefold():
        return identity_text
    if any(name.casefold() not in basis.casefold() for name, basis in zip(values[:2], values[2:])):
        return identity_text
    if not re.search(r"(?i)\bunbroken\b|\bvoice (?:had|has) not (?:yet )?broken\b", values[2]):
        return identity_text
    if any(re.search(r"(?i)\b(?:no evidence|not confirmed|not established|hypothetical|possibly)\b", v) for v in values[2:]):
        return identity_text
    if re.search(r"(?i)\b(?:not|never|without)\b.{0,30}\bunbroken\b", values[2]):
        return identity_text
    # Do not manufacture a valid identity result from an otherwise malformed reply.
    if not re.search(r"(?m)^SINGER_IDENTITY\s*[=:]\s*\S+", text):
        return identity_text
    out = re.sub(r"(?m)^(SINGER_IDENTITY\s*[=:]\s*)\S+", r"\g<1>mixed_leads", text, count=1)
    if re.search(r"(?m)^LEAD_SINGERS\s*[=:]", out):
        out = re.sub(r"(?m)^(LEAD_SINGERS\s*[=:]\s*)\S+", r"\g<1>several", out, count=1)
    else:
        out += "\nLEAD_SINGERS=several"
    if "UNBROKEN_MALE_IDENTITY_PRESERVED=yes" not in out:
        out = out.rstrip() + "\nUNBROKEN_MALE_IDENTITY_PRESERVED=yes\n"
    if "COLEAD_IDENTITY_SCOPE=" not in out:
        out += ("COLEAD_IDENTITY_SCOPE=The identity findings retain an unbroken male lead AND a female "
                "co-lead. Female-sounding samples do not establish which person sang them; do not "
                "relabel both people female or assign unverified passage identities.\n")
    return out


def _unbroken_colead_identity(*args, **kwargs):
    """SINGER_IDENTITY_PRE6 picks the 0.7.1-pre6 version; False gives 0.7's."""
    return (_unbroken_colead_identity__pre6 if SINGER_IDENTITY_PRE6 else _unbroken_colead_identity__07)(*args, **kwargs)


def _identity_colead_recheck_needed(identity_text, background):
    if not PRESERVE_UNBROKEN_COLEAD_IDENTITY:
        return False
    text = str(identity_text or "")
    # A complete supported mixed result requires no extra call.
    if re.search(r"(?m)^SINGER_IDENTITY=mixed_leads\b", text):
        return False
    young = re.search(r"(?m)^SINGER_IDENTITY=(?:child_male_likely|adolescent_male_likely)\b", text)
    young = young or re.search(r"(?i)\bunbroken\b|before his voice broke", text)
    return bool(young and re.search(r"(?i)\b(?:co[- ]lead|duet|shared lead vocals)\b", str(background or "")))


def _identity_colead_source_checked__pre6(candidate, background, metadata):
    """Accept a repair only with explicit per-performer fields and source excerpts.
    Never promote a group member, backing singer or pitch impression to co-lead.
    """
    fixed = _unbroken_colead_identity(candidate)
    if "UNBROKEN_MALE_IDENTITY_PRESERVED=yes" not in str(fixed):
        return None
    def field(key):
        hits = re.findall(r"(?mi)^" + key + r"\s*=\s*([^\n]+)", str(candidate or ""))
        return hits[0].strip() if len(hits) == 1 else ""
    norm = lambda x: re.sub(r"\s+", " ", str(x)).strip().casefold()
    bg = norm(background)
    title = norm((metadata or {}).get("title", ""))
    for prefix, role in (("MALE", "UNBROKEN_MALE_LEAD_NAME"), ("FEMALE", "FEMALE_COLEAD_NAME")):
        quote, name = field(prefix + "_IDENTITY_SOURCE"), field(role)
        if not quote or not name or len(quote) > 800 or norm(quote) not in bg or norm(name) not in norm(quote):
            return None
        evidence = re.sub(re.escape(name), "[name]", quote, flags=re.I)
        if title:
            evidence = re.sub(re.escape(str((metadata or {}).get("title") or "")), "[title]", evidence, flags=re.I)
        if not _identity_asserted_text(evidence):
            return None
        if prefix == "MALE" and not re.search(r"(?i)\bunbroken\b|voice (?:that )?(?:had|has) not (?:yet )?broken|before his voice broke", _identity_asserted_text(evidence)):
            return None
        if prefix == "FEMALE":
            if (not title or title not in norm(quote)
                    or not re.search(r"(?i)\b(?:co[- ]lead|duet|shared lead vocals)\b", quote)
                    or re.search(r"(?i)\b(?:not|no|rumou?r|unconfirmed|possibly|backing only)\b", norm(quote).replace(title, "[title]").replace(norm(name), "[name]"))):
                return None
    return fixed


def _identity_colead_source_checked__07(candidate, background, metadata):
    """Accept a repair only with explicit per-performer fields and source excerpts.
    Never promote a group member, backing singer or pitch impression to co-lead.
    """
    fixed = _unbroken_colead_identity(candidate)
    if "UNBROKEN_MALE_IDENTITY_PRESERVED=yes" not in str(fixed):
        return None
    def field(key):
        hits = re.findall(r"(?mi)^" + key + r"\s*=\s*([^\n]+)", str(candidate or ""))
        return hits[0].strip() if len(hits) == 1 else ""
    norm = lambda x: re.sub(r"\s+", " ", str(x)).strip().casefold()
    bg = norm(background)
    title = norm((metadata or {}).get("title", ""))
    for prefix, role in (("MALE", "UNBROKEN_MALE_LEAD_NAME"), ("FEMALE", "FEMALE_COLEAD_NAME")):
        quote, name = field(prefix + "_IDENTITY_SOURCE"), field(role)
        if not quote or not name or len(quote) > 800 or norm(quote) not in bg or norm(name) not in norm(quote):
            return None
        if prefix == "MALE" and not re.search(r"(?i)\bunbroken\b|voice (?:had|has) not (?:yet )?broken|before his voice broke", quote):
            return None
        if prefix == "FEMALE":
            if (not title or title not in norm(quote)
                    or not re.search(r"(?i)\b(?:co[- ]lead|duet|shared lead vocals)\b", quote)
                    or re.search(r"(?i)\b(?:not|no|rumou?r|unconfirmed|possibly|backing only)\b", norm(quote).replace(title, "[title]").replace(norm(name), "[name]"))):
                return None
    return fixed


def _identity_colead_source_checked(*args, **kwargs):
    """SINGER_IDENTITY_PRE6 picks the 0.7.1-pre6 version; False gives 0.7's."""
    return (_identity_colead_source_checked__pre6 if SINGER_IDENTITY_PRE6 else _identity_colead_source_checked__07)(*args, **kwargs)


def _reference_single_credit_roster(background, metadata, name, credit):
    """Positive, complete track personnel list; never infer exclusivity from silence."""
    norm = lambda x: re.sub(r"\s+", " ", str(x)).strip().casefold()
    title = norm((metadata or {}).get("title", ""))
    artist = norm((metadata or {}).get("artist", ""))
    if not title or not artist or not name or not credit:
        return False
    articles = re.split(r'(?m)^--- "([^"\n]+)" ---\s*\n', str(background or ""))
    for i in range(1, len(articles), 2):
        article_title, body = articles[i], articles[i+1]
        # Require the actual track article, rather than band/album credits.
        if norm(article_title) not in (title, title + " (song)") or artist not in norm(body):
            continue
        if re.search(r"(?i)\b(?:co[- ]lead|duet|guest vocal(?:ist|s)?|additional vocal(?:ist|s)?)\b", body):
            continue
        headings = list(re.finditer(r"(?mi)^(?:={2,6}\s*)?(Personnel|Credits)(?:\s*={2,6})?\s*$", body))
        for h in headings:
            tail = body[h.end():]
            # A following non-personnel heading establishes that the list was
            # not cut off by a reference snippet's length cap.
            end = re.search(r"(?mi)^(?:={2,6}\s*)?(Charts|Chart performance|Certifications|References|External links|Notes|Release history|Reception)(?:\s*={2,6})?\s*$", tail)
            if not end:
                continue
            block = tail[:end.start()].strip()
            if not block or re.search(r"\.\.\.|…|(?i:additional personnel|guest|co[- ]lead)", block):
                continue
            vocal_names = []; valid = True
            for line in block.splitlines():
                if not line.strip():
                    continue
                row = re.match(r"^\s*[*#•-]?\s*([^–—\n]+?)\s+[–—-]\s+(.+?)\s*$", line)
                if not row:
                    valid = False
                    break
                performer, roles = row.groups()
                if re.search(r"(?i)\b(?:vocals?|singer|singing)\b", roles):
                    # Even another backing singer vetoes this conservative path.
                    vocal_names.append(norm(performer))
                    if re.search(r"(?i)\bbacking\b|\bbackground\b", roles):
                        valid = False
                if re.search(r"(?i)\b(?:not|uncertain|possibly|uncredited)\b", roles):
                    valid = False
            if (valid and vocal_names == [norm(name)] and norm(credit) in norm(block)
                    and norm(name) in norm(credit) and re.search(r"(?i)\bvocals?\b", credit)):
                return True
    return False


def _section_voice_identity_line(svc):
    """The line the singer-identity step is shown when the section voice
    check confirmed a second lead of the other gender, else ""."""
    if not svc or not svc.get("mixed"):
        return ""
    if "SAMPLED_LEADS_ONLY=yes" in svc.get("block", ""):
        return (svc.get("evidence", "") + " These acoustic impressions suggest mixed leads but can confuse register changes with another person. "
                "Check positive reference credits before settling singer identity. Neither voice is established as the majority lead. "
                "Only the sampled intervals were checked; do not infer ages from these clips.")
    return ("SECTION VOICE CHECK (measured, then heard by the listening model): " + (svc.get("evidence") or "")
            + " These acoustic impressions suggest mixed leads; reference credits may identify a register-change false positive. "
            "Preserve genuine documented co-leads and do not infer exclusive credits from missing information.")


def _qualify_unconfirmed_leads__pre6(identity_text, analysis_text):
    """Do not publish a single-lead identity when sampled genders disagree."""
    t = str(identity_text or "")
    if (not t or "REFERENCE_VOCAL_CORRECTION=yes" in t
            or "UNBROKEN_MALE_IDENTITY_PRESERVED=yes" in t
            or re.search(r"(?mi)^SINGER_IDENTITY=mixed_leads\s*$", t)):
        return identity_text
    if (re.search(r"(?mi)^YOUNG_SOLO_MALE_CLIPS=applied: documented adult\s*$", t)
            and re.search(r"(?mi)^SINGER_IDENTITY=post_puberty_male\s*$", t)
            and re.search(r"(?mi)^LEAD_SINGERS=one\s*$", t)):
        return identity_text
    line = re.search(r"Section voice check[^\n]*inconclusive -- sampled lead voices disagree[^\n]*", str(analysis_text or ""))
    if not line:
        return identity_text
    evidence = line.group(0)
    if not all(re.search(r"\b" + g + r" \(high; (?:solo_lead|lead_plus_backing)\)", evidence) for g in ("male", "female")):
        return identity_text
    if not re.search(r"(?mi)^SINGER_IDENTITY=(?:female_teen_adult|post_puberty_male|uncertain)\s*$", t):
        return identity_text
    return ("SINGER_IDENTITY=uncertain\n"
            "REASONING=High-confidence sampled clips suggest both male and female foreground leads, "
            "but corroboration did not establish distinct singers. The whole-track lead profile is provisional; "
            "neither a single lead nor mixed leads is established.\nCONFIDENCE=low\nLEAD_SINGERS=uncertain")


def _qualify_unconfirmed_leads__07(identity_text, analysis_text):
    """Do not publish a single-lead identity when sampled genders disagree."""
    t = str(identity_text or "")
    if (not t or "REFERENCE_VOCAL_CORRECTION=yes" in t
            or "UNBROKEN_MALE_IDENTITY_PRESERVED=yes" in t
            or re.search(r"(?mi)^SINGER_IDENTITY=mixed_leads\s*$", t)):
        return identity_text
    line = re.search(r"Section voice check[^\n]*inconclusive -- sampled lead voices disagree[^\n]*", str(analysis_text or ""))
    if not line:
        return identity_text
    evidence = line.group(0)
    if not all(re.search(r"\b" + g + r" \(high; (?:solo_lead|lead_plus_backing)\)", evidence) for g in ("male", "female")):
        return identity_text
    if not re.search(r"(?mi)^SINGER_IDENTITY=(?:female_teen_adult|post_puberty_male|uncertain)\s*$", t):
        return identity_text
    return ("SINGER_IDENTITY=uncertain\n"
            "REASONING=High-confidence sampled clips suggest both male and female foreground leads, "
            "but corroboration did not establish distinct singers. The whole-track lead profile is provisional; "
            "neither a single lead nor mixed leads is established.\nCONFIDENCE=low\nLEAD_SINGERS=uncertain")


def _qualify_unconfirmed_leads(*args, **kwargs):
    """SINGER_IDENTITY_PRE6 picks the 0.7.1-pre6 version; False gives 0.7's."""
    return (_qualify_unconfirmed_leads__pre6 if SINGER_IDENTITY_PRE6 else _qualify_unconfirmed_leads__07)(*args, **kwargs)


def _reconcile_identity_with_section_voice(identity_text, analysis_text):
    """When the notes carry a SECTION VOICE CHECK block (a second lead of the
    other gender, confirmed) but the singer-identity answer names one
    voice, set it to mixed_leads / LEAD_SINGERS=several and say why. Any
    other identity text is returned unchanged."""
    t = str(identity_text or "")
    try:
        if not SECTION_VOICE_CHECK or not t or "REFERENCE_VOCAL_CORRECTION=yes" in t:
            return identity_text
        m = _SV_BLOCK_RE.search(str(analysis_text or ""))
        if not m:
            return _qualify_unconfirmed_leads(identity_text, analysis_text)
        tag = _parse_singer_identity(t)
        if not tag or tag == "mixed_leads":
            return identity_text
        other, spans, main = m.group(1), m.group(2).strip(), m.group(3)
        preserved_young = (PRESERVE_UNBROKEN_COLEAD_IDENTITY
                           and tag in ("child_male_likely", "adolescent_male_likely")
                           and _identity_reason_says_unbroken(t))
        out = re.sub(r"(?m)^(SINGER_IDENTITY\s*[=:]\s*)\S+", lambda _m: _m.group(1) + "mixed_leads", t, count=1)
        if re.search(r"(?m)^LEAD_SINGERS\s*[=:]", out):
            out = re.sub(r"(?m)^(LEAD_SINGERS\s*[=:]\s*)\S+", lambda _m: _m.group(1) + "several", out, count=1)
        else:
            out = out.rstrip() + "\nLEAD_SINGERS=several"
        note = (f" [Set to mixed_leads by the section voice check: the listening model heard a {other} lead in "
                f"{spans} and a {main} lead elsewhere; '{tag}' above describes the main lead only.]")
        if "SAMPLED_LEADS_ONLY=yes" in str(analysis_text or ""):
            note = " [Set to mixed_leads by confirmed sampled lead voices; no majority lead or untested section assignments are established.]"
        if re.search(r"(?m)^REASONING\s*[=:]", out):
            out = re.sub(r"(?m)^(REASONING\s*[=:][^\n]*)", lambda _m: _m.group(1) + note, out, count=1)
        else:
            out = out.rstrip() + "\nREASONING=" + note.strip()
        if preserved_young and "UNBROKEN_MALE_IDENTITY_PRESERVED=yes" not in out:
            out = out.rstrip() + "\nUNBROKEN_MALE_IDENTITY_PRESERVED=yes\n"
            out += ("COLEAD_IDENTITY_SCOPE=A supported unbroken male identity was retained before the "
                    "track-wide tag became mixed_leads. Female-sounding samples are impressions, not "
                    "proof of female identity. Preserve any separately documented female co-lead; "
                    "do not assign the male identity to every sample or invent passage identities.\n")
        return out
    except Exception:
        return identity_text


def _section_voice_audit_line(svc, final_lead, initial_lead, confirm_lead, vocal_objective_report):
    """One line for the VOCAL DECISION AUDIT, on every track that has one:
    what the section voice check did, or why it did not run."""
    try:
        if svc and svc.get("summary"):
            why = svc["summary"]
        elif not SECTION_VOICE_CHECK:
            why = "not run -- switched off"
        else:
            g = _adult_lead_gender(final_lead)
            sep = _f0_hint_separation(vocal_objective_report)
            if not g:
                why = "not run -- the lead is not a settled adult voice (young, child, uncertain or mixed)"
            elif any(o and o not in ("unknown", "unparsed") and _adult_lead_gender(o) != g
                     for o in (initial_lead, confirm_lead)):
                why = "not run -- the vocal passes disagree on the lead's gender or age"
            elif sep is None:
                why = "not run -- no second vocal pitch centre"
            elif _f0_sep_is_octave(sep):
                why = "not run -- the two vocal pitch centres are an octave apart"
            elif sep < SECTION_VOICE_MIN_SHIFT_ST:
                why = f"not run -- the vocal pitch centres are under {SECTION_VOICE_MIN_SHIFT_ST:g} semitones apart"
            else:
                why = "not run -- no separated vocal track to measure"
        return f"- Section voice check (a second lead of the other gender): {why}\n"
    except Exception:
        return ""


def _apply_section_voice_check(final_lead, multi_voice_fields, svc):
    """mixed_leads and the co-lead fields when the section voice check found
    a second lead of the other gender; else final_lead unchanged."""
    if not svc or not svc.get("mixed") or multi_voice_fields is None:
        return final_lead
    multi_voice_fields["num_distinct_voices"] = "2"
    multi_voice_fields["voice_arrangement"] = "duet_co_leads"
    multi_voice_fields["co_lead_detail"] = svc.get("detail") or ""
    _prev = _SV_NOT_UPGRADED_RE.sub("", multi_voice_fields.get("multi_voice_evidence") or "")
    if _prev.strip().lower() == "none":
        _prev = ""
    multi_voice_fields["multi_voice_evidence"] = ((_prev + " [" + (svc.get("evidence") or "") + "]").strip())
    return "mixed_leads"


def _reconcile_identity_with_acoustic_age(resolved_tag, final_lead):
    """Decide the user-facing lead category from the identity pass + acoustics.

    The identity pass is the right authority for WHO is singing (and therefore
    gender, which artist identity genuinely settles). It is NOT an authority on
    vocal maturity: it is allowed to map "a named, documented solo male artist"
    straight to post_puberty_male, which silently overwrites a correct
    adolescent acoustic read whenever the artist is an adult today. That is the
    early-teen-recording failure -- the artist is a grown man now, so the
    identity pass upgrades a voice that had not finished changing when the
    record was made.

    So: identity wins on gender and on everything else, but it may not push a
    male acoustic read UP the maturity scale. Demotions and gender corrections
    are unaffected.
    """
    if resolved_tag not in VOCAL_LEAD_TAGS:
        return final_lead
    if resolved_tag == final_lead:
        return resolved_tag

    acoustic_young_male = final_lead in ("adolescent_male_likely",)
    identity_adult_male = resolved_tag in ("post_puberty_male", "adult_male", "young_male")
    if acoustic_young_male and identity_adult_male:
        # Same gender, higher maturity, sourced from identity rather than audio.
        return final_lead

    return resolved_tag


# 0.7.1-pre12 PROBABILITY_LINE_ANY_FORMAT (F-03): one reader for the PROBABILITY_ESTIMATE line,
# in percentages or 0-1 decimals, label before or after the number, "=" or ":" after the name.
def _probability_line_shares(text):
    """{"female", "adult_male", "adolescent", "child"} -> fraction, from the first
    PROBABILITY_ESTIMATE line; None when there is none or nothing could be read."""
    t = re.sub(r"[\u2010-\u2015]", "-", _strip_md_emphasis(str(text or "")))
    m = re.search(r"(?im)^[ \t*_-]*PROBABILITY[ _]ESTIMATE[ \t]*[:=][ \t]*(.+)$", t)
    if not m:
        return None
    line = m.group(1)
    pct = "%" in line
    out = {}
    for part in re.split(r"[;,]\s*(?![^()]*\))", line):
        if pct:
            n = re.search(r"~?\s*(\d{1,3}(?:\.\d+)?)\s*%", part)
            val = float(n.group(1)) / 100.0 if n else None
        else:
            n = re.search(r"(?<![\w.])(0?\.\d+|1(?:\.0+)?|0)(?![\w.])", part)
            val = float(n.group(1)) if n else None
        if val is None or not 0.0 <= val <= 1.0:
            continue
        lab = (part[:n.start()] + " " + part[n.end():]).lower()
        if re.search(r"female|wom[ae]n|girl", lab):
            key = "female"
        elif re.search(r"child|prepubert", lab):
            key = "child"
        elif re.search(r"adolescen|changing|transitional|teen(?:age)?\s*male|\bboy", lab):
            key = "adolescent"
        elif re.search(r"\bmale|\bm[ae]n\b|post[- ]?pubert", lab):
            key = "adult_male"
        else:
            continue
        out[key] = out.get(key, 0.0) + val
    return out or None


def _vocal_probability_masses(vocal_text):
    """dev57: (male, female) probability mass from the vocal pass's
    "PROBABILITY ESTIMATE:" line (adolescent + post-puberty male; female), or
    None when there is no such line."""
    if PROBABILITY_LINE_ANY_FORMAT:                               # 0.7.1-pre12
        sh = _probability_line_shares(vocal_text)
        return (sh.get("adult_male", 0.0) + sh.get("adolescent", 0.0), sh.get("female", 0.0)) if sh else None
    t = re.sub(r"[\u2010-\u2015]", "-", _strip_md_emphasis(str(vocal_text or "")))
    m = re.search(r"(?im)^[ \t*_-]*PROBABILITY[ _]ESTIMATE[ \t]*[:=][ \t]*(.+)$", t)
    if not m:
        return None
    male = female = 0.0
    seen = False
    for part in re.split(r"[;,]\s*", m.group(1)):
        mm = re.match(r"\s*(.+?)[\s:=]+([01](?:\.\d+)?)\s*$", part)
        if not mm:
            continue
        lab, val = mm.group(1).lower(), float(mm.group(2))
        if "female" in lab:
            female += val
            seen = True
        elif "male" in lab and "child" not in lab:
            male += val
            seen = True
    return (male, female) if seen else None


def _vocal_probability_young_mass(vocal_text):
    """dev59: (young, adult_male, female) from the vocal pass's PROBABILITY
    ESTIMATE line: child-like + adolescent male; post-puberty / adult male;
    female. None when there is no such line."""
    if PROBABILITY_LINE_ANY_FORMAT:                               # 0.7.1-pre12
        sh = _probability_line_shares(vocal_text)
        return ((sh.get("child", 0.0) + sh.get("adolescent", 0.0), sh.get("adult_male", 0.0), sh.get("female", 0.0))
                if sh else None)
    t = re.sub(r"[\u2010-\u2015]", "-", _strip_md_emphasis(str(vocal_text or "")))
    m = re.search(r"(?im)^[ \t*_-]*PROBABILITY[ _]ESTIMATE[ \t]*[:=][ \t]*(.+)$", t)
    if not m:
        return None
    young = adult = female = 0.0
    seen = False
    for part in re.split(r"[;,]\s*", m.group(1)):
        mm = re.match(r"\s*(.+?)[\s:=]+([01](?:\.\d+)?)\s*$", part)
        if not mm:
            continue
        lab, val = mm.group(1).lower(), float(mm.group(2))
        if "female" in lab:
            female += val
        elif "child" in lab or "prepubert" in lab or "adolescent" in lab or "changing" in lab:
            young += val
        elif "male" in lab:
            adult += val
        else:
            continue
        seen = True
    return (young, adult, female) if seen else None


_VOICE_UNBROKEN_MARK = "VOICE MATURITY: the voice sounds UNBROKEN"


def _vocal_probability_veto(lean, vocal_text):
    """dev57: see VOCAL_PROBABILITY_VETO. A short reason when the vocal pass's
    probability line clearly favours the other gender, else ""."""
    if not VOCAL_PROBABILITY_VETO or lean not in ("male", "female"):
        return ""
    pm = _vocal_probability_masses(vocal_text)
    if not pm:
        return ""
    male, female = pm
    mine, other = (female, male) if lean == "female" else (male, female)
    if other >= 0.4 and mine <= 0.15 and other >= 3.0 * max(mine, 1e-9):
        other_name = "male" if lean == "female" else "female"
        return (f"the dedicated vocal pass's probability estimate favours a {other_name} voice "
                f"({other:.2f} against {mine:.2f})")
    return ""


def _lead_gender(tag):
    if tag in ("child_male_likely", "adolescent_male_likely", "post_puberty_male",
               "adult_male", "young_male"):
        return "male"
    if tag in FEMALE_LEAD_CATEGORIES or tag == "child_female_likely":
        return "female"
    return None


def _identity_overrides_gender(identity_tag, acoustic_lead):
    gi, ga = _lead_gender(identity_tag), _lead_gender(acoustic_lead)
    return bool(gi and ga and gi != ga)


def _unbroken_voice_note(identity_tag, backing_profiles):
    """dev59: ACOUSTIC VOCAL READING for a voice recorded as sounding
    unbroken (see VOCAL_UNBROKEN_SIGNATURE)."""
    g = _lead_gender(identity_tag)
    if g == "male":
        who = (
            "With the SINGER IDENTITY below (male), this is a boy's voice that has not dropped yet "
            "-- describe it that way (light, high, boyish; it had not broken), never as a woman's. "
            "If he was already in his mid-teens on this recording, say so: most boys' voices have "
            "dropped by about 15 or 16, so a voice still unbroken at that age is notable, and "
            "worth remarking on when the voice comes up."
        )
    elif g == "female":
        who = "With the SINGER IDENTITY below (female), it is simply a woman's voice."
    else:
        who = ("Without an identity to settle it, say it sounds young and high -- it could be a boy "
               "whose voice has not dropped or a woman -- and do not assign a gender.")
    return (
        "\n\nACOUSTIC VOCAL READING (audio only): the lead voice sounds UNBROKEN -- the listening "
        "model heard it as a woman's voice, while its dedicated vocal pass rated it most likely a "
        "child's or a teenage boy's. " + who + " Everything measured still stands: use the VOCAL "
        "PITCH REPORT and VOCAL / SINGER PROFILE for pitch figures, range, register, timbre, "
        "brightness and weight, exactly as reported." + _backing_guidance(backing_profiles)
    )


# dev69: a young singer whose voice had already deepened (see VOCAL_DEEPENED_SIGNATURE).
_YOUNG_MALE_TAGS = ("child_male_likely", "adolescent_male_likely")
_ADULT_MALE_ACOUSTIC = ("post_puberty_male", "adult_male")


_UV_SAYS_UNBROKEN_RE = re.compile(
    r"(?i)\bunbroken\b|\bfemale[- ]sounding\b|\b(?:read|reads|heard|sounds?)\s+(?:as\s+)?(?:an?\s+)?"
    r"(?:high\s*/\s*)?(?:female|woman)\b")
_UV_SAYS_DEEPENED_RE = re.compile(
    r"(?i)\b(?:not|no longer|never|isn't|wasn't)\b[^.\n]{0,25}\bunbroken\b|"
    r"\balready\s+(?:broken|deepened|dropped)\b|\bhad\s+(?:already\s+)?(?:broken|deepened|dropped)\b")


def _identity_reason_says_unbroken(reason):
    """True when an identity's REASONING records a female-sounding or
    unbroken voice and does not say the voice had already deepened."""
    r = str(reason or "")
    return bool(_UV_SAYS_UNBROKEN_RE.search(r)) and not _UV_SAYS_DEEPENED_RE.search(r)


def _identity_resolution_reason(text):
    """The REASONING line of the SINGER IDENTITY RESOLUTION block in an
    analysis, or ""."""
    t = str(text or "")
    i = t.rfind("SINGER IDENTITY RESOLUTION")
    if i < 0:
        return ""
    m = re.search(r"REASONING\s*[=:]\s*([^\n]+)", t[i:i + 3000])
    return m.group(1) if m else ""


def _young_male_unbroken_identity_text(acoustic_lead):
    """UNBROKEN_VOICE_IDENTITY_CHECK: the SINGER IDENTITY note's sentence for a
    boy or teenager whose identity reasoning records a female-sounding or
    unbroken voice although the audio passes' final profile is a man's."""
    return (
        f"The audio passes' final profile reads {acoustic_lead} (a man's voice), but the identity "
        "reasoning records that the voice was heard as female-sounding: this is his voice before it "
        "broke, or while it was still changing, sitting where women's and boys' voices usually do. "
        "Describe it as a boy's high, light voice -- not a woman's and not a grown man's deepened "
        "voice. Keep the measured pitch, range and timbre exactly as reported; this singer is male, "
        "never 'she' or a woman."
    )


def _young_male_adult_voice_text(acoustic_lead, vocal_text):
    """dev69: the SINGER IDENTITY note's sentence when the identity puts a
    boy or teenager on the recording but the audio heard a man's voice."""
    p = _measured_lead_pitch(vocal_text)
    meds = [v for v in (p.get("median_hz"), p.get("notes_median_hz")) if v]
    low = bool(meds) and min(meds) < float(VOCAL_DEEPENED_MAX_MEDIAN_HZ)
    if low:
        _r = []
        if p.get("median_hz"):
            _r.append(f"{p['median_hz']:.0f} Hz ({p.get('median_note', '?')})")
        if p.get("notes_median_hz"):
            _r.append(f"{p['notes_median_hz']:.0f} Hz ({p.get('notes_median_note', '?')}) note by note")
        mp = " and ".join(_r)
        return (
            f"The audio heard a man's voice here: the vocal passes read it as {acoustic_lead}, and "
            f"the measured median pitch is {mp}, within an adult male singer's range. So his "
            "voice had already deepened (broken, or largely so) by this recording, even though he "
            "was young. Describe it as his deepened voice -- lower and fuller than a boy's "
            "unbroken voice -- not as high, light or boyish. If his earlier recordings, where the "
            "voice had not dropped, come up, that change is the thing to point out. Keep the "
            "measured pitch, range and timbre exactly as reported; this singer is male, never "
            "'she' or a woman."
        )
    return (
        f"The audio passes read it as {acoustic_lead} -- a man's voice -- while the identity puts "
        "him younger, so his voice may already have been deepening on this recording. Use this "
        "identity for gender and age, keep the measured pitch, range and timbre from the audio, "
        "and do not describe the voice as unbroken, high or light unless the measured pitch says so."
    )


def _confirm_flip_reading(text):
    """dev18: (first reading, confirmation reading, median Hz, median note)
    when the VOCAL DECISION AUDIT shows the confirmation pass alone turning
    the first vocal pass's female reading into a man's voice, and the lead's
    measured median pitch (both readings, when both exist) is at or above
    VOCAL_DEEPENED_MAX_MEDIAN_HZ; else None."""
    t = str(text or "")
    mi = re.search(r"(?m)^- Initial isolated lead profile:\s*([a-z_]+)", t)
    mc = re.search(r"(?m)^- Confirmation pass:\s*([a-z_]+)\s*\(confidence=(\w+)\)", t)
    mf = re.search(r"(?m)^- FINAL LEAD PROFILE:\s*([a-z_]+)", t)
    if not (mi and mc and mf):
        return None
    ini, con, fin = mi.group(1), mc.group(1), mf.group(1)
    if ini not in FEMALE_LEAD_CATEGORIES:
        return None
    if con not in _ADULT_MALE_ACOUSTIC or fin != con:
        return None
    p = _measured_lead_pitch(t)
    meds = [(v, nt) for v, nt in ((p.get("median_hz"), p.get("median_note")),
                                  (p.get("notes_median_hz"), p.get("notes_median_note"))) if v]
    if not meds:
        return None
    med, note = min(meds)
    if med < float(VOCAL_DEEPENED_MAX_MEDIAN_HZ):
        return None
    return ini, con, med, note or "?"


def _young_male_confirm_flip_text(acoustic_lead, vocal_text):
    """dev18: the SINGER IDENTITY note's sentence for a boy or teenager whose
    voice the first vocal pass heard as a woman's and only the confirmation
    pass as a man's, at a pitch above where a deepened voice sits."""
    ini, con, med, note = _confirm_flip_reading(vocal_text)
    return (
        f"The first vocal pass heard a woman's voice ({ini}); only the confirmation pass heard a "
        f"man's ({con}), and the measured median pitch is {med:.0f} Hz ({note}), above where a "
        "deepened male voice usually sits. For a boy or teenager that is his voice before it "
        "dropped: the lead voice sounds unbroken here, high and young, sitting where women's and "
        "boys' voices usually do. Call it his voice, use the male column of the VOICE-TYPE MAP for "
        "how high or deep it sits, and keep the measured pitch, range and timbre exactly as "
        "reported. Do not call it a grown man's voice. Never call this singer 'she' or a woman."
    )


def build_singer_identity_note__pre6(identity_tag, acoustic_lead, unbroken=False, vocal_text=""):
    """Who is singing (the person), which MAY use tags, cover art and
    background knowledge. Kept separate from the acoustic note so identity
    evidence can settle gender/age of the PERSON without rewriting the
    description of the SOUND."""
    if identity_tag not in VOCAL_LEAD_TAGS or identity_tag in ("unknown", ""):
        return ""
    if (SECTION_VOICE_CHECK and acoustic_lead == "mixed_leads" and identity_tag != "mixed_leads"
            and _SECTION_VOICE_HEAD + " (" in str(vocal_text or "")):
        # The section voice check heard a second lead of the other gender in
        # some stretches; a one-person identity names the main lead only.
        return (
            "\n\nSINGER IDENTITY (who is singing; combines audio, file tags, cover art and "
            f"background knowledge): {identity_tag}, for the main lead. The SECTION VOICE CHECK above "
            "heard a second lead singer of the other gender in the stretches it names, so this is a "
            "mixed-voice track: use this identity for the main lead only, and describe the second "
            "voice where it leads."
        )
    maturity = re.search(r"(?mi)^YOUNG_VOICE_MATURITY=(unbroken|partly_deepened|deepened)\s*$", str(vocal_text or ""))
    if identity_tag in ("adolescent_male_likely", "child_male_likely") and maturity:
        word = {"unbroken": "his voice sounds unbroken on this recording",
                "partly_deepened": "his voice hadn't fully deepened on this recording",
                "deepened": "his voice sounds male and had deepened on this recording"}[maturity.group(1)]
        return ("\n\nSINGER IDENTITY (identity and acoustic maturity kept separate): " + identity_tag
                + ". From the accepted single-lead acoustic evidence, " + word
                + ". Preserve the young male identity and measured pitch/range/timbre. These are audible "
                  "maturity descriptions, not a medical finding or a numeric age. Do not infer another "
                  "singer from register changes; preserve separately confirmed or documented co-leads.")
    same = identity_tag == acoustic_lead
    txt = (
        "\n\nSINGER IDENTITY (who is singing; combines audio, file tags, cover art and "
        f"background knowledge): {identity_tag}. This decides the singer's gender and age "
        "and therefore the pronouns and the kind of voice you name (a boy's voice, a young "
        "man's, a woman's). "
    )
    if same:
        txt += "It agrees with the acoustic reading."
        return txt
    male_id = identity_tag in ("child_male_likely", "adolescent_male_likely",
                               "post_puberty_male", "adult_male", "young_male")
    female_ac = acoustic_lead in FEMALE_LEAD_CATEGORIES or acoustic_lead in (
        "child_female_likely",)
    if (male_id and unbroken and VOCAL_UNBROKEN_SIGNATURE
            and acoustic_lead in ("", "unknown", "uncertain", None)):
        txt += (
            "The audio says his voice sounds unbroken (see the ACOUSTIC VOCAL READING): it is his "
            "voice before it dropped, sitting where women's and boys' voices usually do. Call it "
            "his voice, use the male column of the VOICE-TYPE MAP for how high or deep it sits, "
            "and keep the measured pitch, range and timbre exactly as reported. Never call this "
            "singer 'she' or a woman."
        )
        return txt
    if male_id and female_ac:
        txt += (
            f"The audio passes read the voice as female ({acoustic_lead}), but the singer is "
            "male: treat that as his pitch sitting where female voices usually do, not as a "
            "woman singing. Call it his voice, use the male column of the VOICE-TYPE MAP for "
            "how high or deep it sits, and keep the measured pitch, range and timbre exactly "
            "as reported. Never call this singer 'she' or a woman."
        )
    elif (VOCAL_DEEPENED_SIGNATURE and identity_tag in _YOUNG_MALE_TAGS
          and acoustic_lead in _ADULT_MALE_ACOUSTIC):
        # dev69: the mirror of the unbroken voice -- a boy or teenager whose
        # voice the audio heard as a man's. The general sentence below gave
        # "the voice still sits high and light" as its example, the opposite.
        # dev18: unless only the confirmation pass heard a man, at a high
        # measured pitch (YOUNG_VOICE_CONFIRM_FLIP).
        if (UNBROKEN_VOICE_IDENTITY_CHECK
                and _identity_reason_says_unbroken(_identity_resolution_reason(vocal_text))):
            txt += _young_male_unbroken_identity_text(acoustic_lead)
        elif YOUNG_VOICE_CONFIRM_FLIP and _confirm_flip_reading(vocal_text):
            txt += _young_male_confirm_flip_text(acoustic_lead, vocal_text)
        else:
            txt += _young_male_adult_voice_text(acoustic_lead, vocal_text)
    else:
        txt += (
            f"The audio passes read it as {acoustic_lead or 'unknown'}. Use this identity for "
            "gender and age; keep the measured pitch, range and timbre from the audio, and "
            "mention the difference only if it is interesting (e.g. 'he was an adult here, but "
            "the voice still sits high and light')."
        )
    return txt


def build_singer_identity_note__07(identity_tag, acoustic_lead, unbroken=False, vocal_text=""):
    """Who is singing (the person), which MAY use tags, cover art and
    background knowledge. Kept separate from the acoustic note so identity
    evidence can settle gender/age of the PERSON without rewriting the
    description of the SOUND."""
    if identity_tag not in VOCAL_LEAD_TAGS or identity_tag in ("unknown", ""):
        return ""
    if (SECTION_VOICE_CHECK and acoustic_lead == "mixed_leads" and identity_tag != "mixed_leads"
            and _SECTION_VOICE_HEAD + " (" in str(vocal_text or "")):
        # The section voice check heard a second lead of the other gender in
        # some stretches; a one-person identity names the main lead only.
        return (
            "\n\nSINGER IDENTITY (who is singing; combines audio, file tags, cover art and "
            f"background knowledge): {identity_tag}, for the main lead. The SECTION VOICE CHECK above "
            "heard a second lead singer of the other gender in the stretches it names, so this is a "
            "mixed-voice track: use this identity for the main lead only, and describe the second "
            "voice where it leads."
        )
    same = identity_tag == acoustic_lead
    txt = (
        "\n\nSINGER IDENTITY (who is singing; combines audio, file tags, cover art and "
        f"background knowledge): {identity_tag}. This decides the singer's gender and age "
        "and therefore the pronouns and the kind of voice you name (a boy's voice, a young "
        "man's, a woman's). "
    )
    if same:
        txt += "It agrees with the acoustic reading."
        return txt
    male_id = identity_tag in ("child_male_likely", "adolescent_male_likely",
                               "post_puberty_male", "adult_male", "young_male")
    female_ac = acoustic_lead in FEMALE_LEAD_CATEGORIES or acoustic_lead in (
        "child_female_likely",)
    if (male_id and unbroken and VOCAL_UNBROKEN_SIGNATURE
            and acoustic_lead in ("", "unknown", "uncertain", None)):
        txt += (
            "The audio says his voice sounds unbroken (see the ACOUSTIC VOCAL READING): it is his "
            "voice before it dropped, sitting where women's and boys' voices usually do. Call it "
            "his voice, use the male column of the VOICE-TYPE MAP for how high or deep it sits, "
            "and keep the measured pitch, range and timbre exactly as reported. Never call this "
            "singer 'she' or a woman."
        )
        return txt
    if male_id and female_ac:
        txt += (
            f"The audio passes read the voice as female ({acoustic_lead}), but the singer is "
            "male: treat that as his pitch sitting where female voices usually do, not as a "
            "woman singing. Call it his voice, use the male column of the VOICE-TYPE MAP for "
            "how high or deep it sits, and keep the measured pitch, range and timbre exactly "
            "as reported. Never call this singer 'she' or a woman."
        )
    elif (VOCAL_DEEPENED_SIGNATURE and identity_tag in _YOUNG_MALE_TAGS
          and acoustic_lead in _ADULT_MALE_ACOUSTIC):
        # dev69: the mirror of the unbroken voice -- a boy or teenager whose
        # voice the audio heard as a man's. The general sentence below gave
        # "the voice still sits high and light" as its example, the opposite.
        # dev18: unless only the confirmation pass heard a man, at a high
        # measured pitch (YOUNG_VOICE_CONFIRM_FLIP).
        if (UNBROKEN_VOICE_IDENTITY_CHECK
                and _identity_reason_says_unbroken(_identity_resolution_reason(vocal_text))):
            txt += _young_male_unbroken_identity_text(acoustic_lead)
        elif YOUNG_VOICE_CONFIRM_FLIP and _confirm_flip_reading(vocal_text):
            txt += _young_male_confirm_flip_text(acoustic_lead, vocal_text)
        else:
            txt += _young_male_adult_voice_text(acoustic_lead, vocal_text)
    else:
        txt += (
            f"The audio passes read it as {acoustic_lead or 'unknown'}. Use this identity for "
            "gender and age; keep the measured pitch, range and timbre from the audio, and "
            "mention the difference only if it is interesting (e.g. 'he was an adult here, but "
            "the voice still sits high and light')."
        )
    return txt


def build_singer_identity_note(*args, **kwargs):
    """SINGER_IDENTITY_PRE6 picks the 0.7.1-pre6 version; False gives 0.7's."""
    return (build_singer_identity_note__pre6 if SINGER_IDENTITY_PRE6 else build_singer_identity_note__07)(*args, **kwargs)


# 0.7.1-pre6: small, deterministic evidence guards shared by Musiclyse and the exporter.
# 0.7.1-pre6: a negated noun phrase is removed on its own: "with no co-lead indicated", "no second lead singer".
_IDENTITY_NEGATED_NOUN_RE = re.compile(
    r"(?i),?\s*\b(?:with\s+)?(?:no|without)\s+(?:unbroken|co[- ]lead|duet|shared|additional|second|"
    r"other|guest|lead|vocal|singer)\w*(?:[\s-]+(?:lead|leads|vocals?|singers?|voices?|parts?|duet)){0,2}"
    r"(?:\s+(?:indicated|found|present|detected|evident|identified|heard|credited))?\b")


def _identity_asserted_text(text):
    """Exclude denied/speculative clauses from fact extraction, not from display.

    Keep explicit unbroken-voice wording and 'not only ... but also ...'.
    Original reasoning remains available to the user, including qualifications.
    """
    source = str(text or "")
    # Preserve a positive identity before a separately denied alternative, or
    # an unknown age. A qualification of the identity itself still rejects it.
    source = re.sub(r",\s*(?=not\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b)", "; ", source)
    source = re.sub(r"(?i),\s*(?=(?:but\s+)?(?:his|her|their|the|exact)\s+"
                    r"(?:exact\s+)?(?:age|birth\s+date|date of birth)\b)", "; ", source)
    parts = re.split(r"(?<![A-Z])(?<!\bJr)(?<!\bSr)\.(?=\s|$)|[;!?\n]", source)
    kept = []
    for part in parts:
        part = _IDENTITY_NEGATED_NOUN_RE.sub("", part)
        check = re.sub(r"(?i)\bnot only\b", "", part)
        check = re.sub(r"(?i)\b(?:has|had|have) not (?:yet )?(?:broken|changed|dropped|deepened)\b", "", check)
        check = re.sub(r"(?i)\b(?:hasn't|hadn't) (?:yet )?(?:broken|changed|dropped|deepened)\b", "", check)
        if re.search(r"(?i)\b(?:not|never|isn't|wasn't|isn’t|wasn’t|cannot|can't|possibly|perhaps|"
                     r"probably|presumably|apparently|likely|might|could|unconfirmed|unverified|"
                     r"uncertain|unknown|rumou?r(?:ed)?)\b|\bno evidence\b|\bmay (?:be|have)\b|"
                     r"\b(?:no|without) (?:unbroken|co[- ]lead|duet|shared|additional|second|other|guest|lead|vocal|singer)\b", check):
            continue
        kept.append(part.strip())
    return ". ".join(p for p in kept if p)


def _identity_reason_text(text):
    """Only the REASONING field, including wrapped continuation lines."""
    if REASONING_STOPS_AT_KNOWN_FIELDS:
        m = re.search(r"(?ims)^\s*REASONING\s*[=:]\s*(.*?)(?=^\s*(?-i:[A-Z][A-Z_]{3,})\s*[=:]|\Z)", str(text or ""))
        return m.group(1).strip() if m else ""
    m = re.search(r"(?ims)^\s*REASONING\s*[=:]\s*(.*?)(?=^\s*[A-Z_]+\s*[=:]|\Z)", str(text or ""))
    return m.group(1).strip() if m else ""


def _identity_shared_leads(reason, names=()):
    """Positive shared-lead evidence, not membership or an untyped feature."""
    for clause in re.split(r"[;!\n]|(?<![A-Z])\.(?=\s|$)", str(reason or "")):
        asserted = _identity_asserted_text(clause)
        if not asserted:
            continue
        shared = re.search(r"(?i)\b(?:share[sd]?|sharing|trade[sd]?|trading|alternate[sd]?|alternating)\s+"
                           r"(?:the\s+)?(?:lead(?:\s+vocals?)?|vocals?)\b|"
                           r"\b(?:duet|co[- ]lead(?:s| vocals?)?|two lead singers|several lead singers)\b", asserted)
        if shared and (not names or any(re.search(r"(?i)(?<!\w)" + re.escape(str(n)) + r"(?!\w)", asserted) for n in names)):
            return True
    asserted = _identity_asserted_text(reason)
    person = r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3}"
    performers = []
    for m in re.finditer(r"(" + person + r"(?:\s+(?:and|&)\s+" + person + r")+)"
                         r"\s+(?i:(?:both\s+)?(?:sing|sang|perform)\s+(?:the\s+)?lead(?:\s+vocals?)?\b)", asserted):
        performers.extend(re.split(r"\s+(?:and|&)\s+", m.group(1)))
    for m in re.finditer(r"(" + person + r")\s+(?i:(?:sings|sang|performs)\s+(?:the\s+)?lead(?:\s+vocals?)?\b)", asserted):
        performers.append(m.group(1))
    unique = {p.casefold() for p in performers}
    if len(unique) >= 2 and (not names or any(str(n).casefold() in unique for n in names)):
        return True
    return False


def _identity_saved_unbroken_lead(identity_text, metadata):
    """Recover an explicit named young male voice finding, never from pitch."""
    t = str(identity_text or "")
    if not re.search(r"(?mi)^SINGER_IDENTITY=mixed_leads\s*$", t):
        return ""
    artist = str((metadata or {}).get("artist") or "")
    main = re.split(r"(?i)\s+(?:feat[.․]?|ft[.․]?|featuring)\s+", artist)[0].strip()
    if not main or len(main.split()) < 2:
        return ""
    reason = _identity_reason_text(t)
    # Ignore synthetic reconciliation brackets and rejected review responses.
    reason = re.sub(r"\[[^]]*\]", "", reason)
    if not re.search(re.escape(main), reason, re.I):
        return ""
    alias = r"(?:" + re.escape(main) + "|" + re.escape(main.split()[-1]) + r")"
    for clause in re.split(r"[;\n]", reason):
        if not _identity_asserted_text(clause):
            continue
        if re.search(r"(?i)\b" + alias + r"['’]s\s+unbroken voice\s+at age\s+(?:[89]|1[0-7])\b", clause):
            return main
    return ""


def _identity_young_voice_maturity(identity_text, evidence_text, metadata=None):
    """Acoustic maturity wording for an accepted single young male lead.

    Identity/age are supplied separately; these cues never establish age or
    turn a confirmed co-lead into one person's changing voice.
    """
    t, ev = str(identity_text or ""), str(evidence_text or "")
    if not re.search(r"(?mi)^SINGER_IDENTITY=(?:adolescent_male_likely|child_male_likely)\s*$", t):
        return ""
    if not re.search(r"(?mi)^LEAD_SINGERS=one\s*$", t):
        return ""
    credit = " ".join(str((metadata or {}).get(k) or "") for k in ("artist", "title"))
    if re.search(r"(?i)\b(?:feat[.․]?|ft[.․]?|featuring)\s", credit):
        return ""
    if re.search(r"SECTION VOICE CHECK \(measured", ev):
        return ""  # A confirmed passage map is about distinct leads.
    lines = list(dict.fromkeys(l for l in ev.splitlines() if "Section voice check" in l))
    samples = re.findall(r"\b(male|female) \(high; (?:solo_lead|lead_plus_backing)\)", "\n".join(lines))
    if samples:
        male, female = samples.count("male"), samples.count("female")
        if male and female:
            # Do not turn unconfirmed *identity/count* evidence into certainty.
            return "partly_deepened"
        if len(samples) >= 2:
            return "unbroken" if female else "deepened"
        return ""
    ini = re.search(r"(?mi)^-?\s*Initial isolated lead profile:\s*([a-z_]+)", ev)
    con = re.search(r"(?mi)^-?\s*Confirmation pass:\s*([a-z_]+)", ev)
    def sex(m):
        if not m:
            return ""
        tag = m.group(1)
        return "female" if tag in ("female_teen_adult", "adult_female", "young_female", "child_female_likely") else (
               "male" if tag in ("post_puberty_male", "adult_male") else "")
    a, b = sex(ini), sex(con)
    if a and b:
        if a != b:
            return "partly_deepened"
        return "unbroken" if a == "female" else "deepened"
    if "VOICE MATURITY: the voice sounds UNBROKEN" in ev:
        return "unbroken"
    return ""


def _identity_override_confidence(text):
    """A rule-based reconciliation cannot inherit high model confidence."""
    if re.search(r"(?mi)^CONFIDENCE\s*[=:]\s*high\s*$", text):
        return re.sub(r"(?mi)^(CONFIDENCE\s*[=:]\s*)high\s*$", r"\g<1>medium", text)
    if not re.search(r"(?mi)^CONFIDENCE\s*[=:]", text):
        return text.rstrip() + "\nCONFIDENCE=medium\n"
    return text


def _identity_secondary_role(text, position):
    prefix = str(text or "")[max(0, position - 50):position]
    return bool(re.search(r"(?i)\b(?:former|original|founding|previous|earlier|later|ex[- ]|"
                          r"guest|featured|backing|background|second|additional|co[- ]lead)"
                          r"(?:\s+(?:lead|male|female))?\s*$|\bco[- ]$", prefix))


def _identity_recording_year(metadata):
    for key in ("original_year", "originalyear", "original_date", "year", "date"):
        m = re.search(r"\b((?:18|19|20)\d\d)\b", str((metadata or {}).get(key) or ""))
        if m:
            return int(m.group(1))
    return None


def _identity_singer_clauses(text, name):
    """Conservatively bind adjacent he/his clauses to the named lead singer.

    Inspect raw clauses before filtering so an intervening person or an
    uncertain claim cannot disappear and leave a misleading antecedent.
    """
    name = re.sub(r"\s+", " ", str(name or "")).strip()
    if not name:
        return []
    person = r"(?<!\w)" + re.escape(name).replace(r"\ ", r"\s+") + r"(?!\w)"
    active = False
    out = []
    for part in re.split(r"(?<![A-Z])(?<!\bJr)(?<!\bSr)\.(?=\s|$)|[;!?\n]", str(text or "")):
        part = part.strip()
        if not part:
            continue
        clean = _identity_asserted_text(part)
        if not clean:
            active = False
            continue
        rest = re.sub(person, "", part, flags=re.I)
        # These roles and other proper names make the pronoun ambiguous.
        named_subject = bool(re.match(r"(?i)^" + person + r"\s+(?:is|was)\b", part)
                             or re.search(r"(?i)\b(?:singer|vocalist) is\s+" + person, part))
        if not named_subject and re.search(r"(?i)\b(?:producer|drummer|guitarist|bassist|manager|father|brother|son|"
                     r"friend|engineer|mother|sister|daughter|husband|wife|pianist|keyboardist|composer|songwriter|director|actor|actress|another|other singer|former singer|backing singer)\b", rest):
            active = False
            continue
        names = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b", rest)
        other_subject = re.search(r"\b(?!He\b|It\b|This\b|That\b)([A-Z][a-z]+)\s+(?:was|is|had|has|sings|sang)\b", rest)
        if names or other_subject:
            active = False
            continue
        if re.search(person, part, re.I) or re.search(r"(?i)\bthe (?:lead )?(?:singer|vocalist)\b", part):
            active = True
        if active:
            out.append(clean)
    return out


def _identity_unbroken_recording_evidence(reason, name):
    """Positive, singer-bound evidence that THIS recording predates voice change.

    A female-sounding read, an era label, artwork or a high pitch is insufficient.
    """
    clauses = _identity_singer_clauses(reason, name)
    # Contradictory maturity assertions must not be silently discarded.
    if re.search(r"(?i)\b(?:not|never|no longer|isn't|wasn't)\b[^.;\n]{0,25}\bunbroken\b|"
                 r"\b(?:voice|it)\s+(?:had|has)\s+(?:already\s+)?(?:broken|changed|dropped|deepened)\b|"
                 r"\balready\s+(?:broken|changed|dropped|deepened)\b", str(reason or "")):
        return False
    for clause in clauses:
        current = re.search(r"(?i)\b(?:this|the) (?:recording|track|song|release)\b|"
                            r"\bat (?:the|that) time\b|\brecorded\b", clause)
        unbroken = re.search(r"(?i)\bunbroken\b|\b(?:voice|it)\s+(?:had|has) not (?:yet )?"
                             r"(?:broken|changed|dropped|deepened)\b|"
                             r"\bbefore his voice (?:broke|changed|dropped|deepened)\b", clause)
        if current and unbroken:
            return True
    return False


def _identity_person_age(reason, background, year, name):
    """Approximate age from a named person's birth year or a bound stated age.

    An unrelated person's age, artwork or a missing adult biography is not
    age evidence. No bare pronoun is resolved across intervening people.
    """
    name = re.sub(r"\s+", " ", str(name or "")).strip()
    if not name:
        return None
    nq = re.escape(name).replace(r"\ ", r"\s+")
    person = r"(?<!\w)" + nq + r"(?!\w)"
    r = _identity_asserted_text(reason)
    # A stated age must identify the singer, not just occur somewhere in prose.
    subject = r"(?:" + person + r"|\bthe (?:lead )?(?:singer|vocalist))"
    for m in re.finditer(r"(?i)" + subject + r"(?:,\s*who)?\s+(?:was|is|aged|age)\s+(?:about\s+|around\s+|~)?(\d{1,2})\b"
                         r"(?!\s*(?:%|:|th|st|nd|rd))", r):
        age = int(m.group(1))
        if 5 <= age <= 99:
            return age
    for clause in _identity_singer_clauses(reason, name):
        m = re.search(r"(?i)(?:" + person + r"|\bthe (?:lead )?(?:singer|vocalist)|\bhe)"
                      r"\s*(?:,\s*|\s+)(?:(?:was|is)\s+)?(?:an?\s+)?"
                      r"(\d{1,2})[- ]year[- ]old\b", clause)
        if m and not re.search(r"(?i)\b(?:now|today|currently|present[- ]day)\b", clause):
            if 5 <= int(m.group(1)) <= 99:
                return int(m.group(1))
    # A nearby pronoun is usable only while its named singer remains the
    # unambiguous subject. Never infer its antecedent after filtering others out.
    for clause in _identity_singer_clauses(reason, name):
        if re.search(r"(?i)\b(?:now|today|currently|present[- ]day)\b", clause):
            continue
        m = re.search(r"(?i)\bhe\s+(?:was|is|aged)\s+(?:about\s+|around\s+|~)?(\d{1,2})\b"
                      r"(?!\s*(?:%|:|th|st|nd|rd))", clause)
        if m and 5 <= int(m.group(1)) <= 99:
            return int(m.group(1))
    try:
        y = int(str(year or "")[:4])
    except ValueError:
        return None
    # Inspect each clause independently: do not cross another named subject.
    for source in (r, _identity_asserted_text(background)):
        patterns = (person + r"\s*\(\s*(?:born|b\.)\s+[^)]{0,40}?\b((?:18|19|20)\d\d)\b",
                    person + r"\s+(?:was\s+)?born\s+(?:on\s+|in\s+)?[^.;\n]{0,30}?\b((?:18|19|20)\d\d)\b",
                    person + r"\s*\(\s*((?:18|19|20)\d\d)\s*[–-]")
        for pattern in patterns:
            for m in re.finditer(pattern, source, re.I):
                age = y - int(m.group(1))
                if 5 <= age <= 99:
                    return age
    if DOCUMENTED_ADULT_NAME_FORMS:
        age = _identity_person_age_more_forms(r, background, y, name)
        if age is not None:
            return age
    return None


def _identity_person_age_more_forms(reason, background, y, name):
    """More ways a biography gives this person's birth year: with a middle name
    ("First Middle Last (born 1985)"), with a birthplace before the year ("Name was born in
    Town, Region, on 1 March 1985"), or by surname alone ("Last was born in 1985") in a
    sentence that names no other full name. None when none applies."""
    parts = name.split()
    year = r"\b((?:18|19|20)\d\d)\b"
    born = r"(?:\s*\(\s*(?:born|b\.)\s+[^)]{0,60}?" + year + r"|\s+(?:was\s+)?born\b[^.;\n]{0,80}?" + year + r")"
    subjects = [r"(?<!\w)" + re.escape(name).replace(r"\ ", r"\s+") + r"(?!\w)"]
    if len(parts) >= 2:
        subjects.append(r"(?<!\w)" + re.escape(parts[0]) + r"(?:\s+[A-Z][\w'’.-]*){1,3}\s+"
                        + re.escape(parts[-1]) + r"(?!\w)")
    for source in (_identity_asserted_text(reason), _identity_asserted_text(background)):
        for sentence in re.split(r"(?<![A-Z])\.(?=\s|$)|[;\n]", source):
            for subj in subjects:
                m = re.search(subj + born, sentence)
                if m:
                    age = y - int(m.group(1) or m.group(2))
                    if 5 <= age <= 99:
                        return age
            if len(parts) >= 2 and len(parts[-1]) >= 3:
                others = [x for x in re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b", sentence)
                          if x.casefold() != name.casefold()]
                if not others:
                    m = re.search(r"(?<!\w)" + re.escape(parts[-1]) + r"(?!\w)" + born, sentence)
                    if m:
                        age = y - int(m.group(1) or m.group(2))
                        if 5 <= age <= 99:
                            return age
    return None


def _parse_singer_identity(text):
    text = _strip_md_emphasis(text)
    m = re.search(r"SINGER_IDENTITY\s*[:=]\s*[\"'<\[]?\s*([A-Za-z0-9_\- ]+)", text or "")
    if not m:
        return ""

    tag = _resolve_lead_tag_variant(m.group(1)) or _normalize_vocal_tag(m.group(1))
    tag = VOCAL_LEAD_ALIASES.get(tag, tag)

    simple_aliases = {
        "young_male": "uncertain",
        "male_lead": "post_puberty_male",
        "adult_female": "female_teen_adult",
    }
    tag = simple_aliases.get(tag, tag)

    return tag if tag in VOCAL_LEAD_TAGS else ""


# dev40: lead vocal shared by several singers (a vocal group, duo, featured
# artist). Appended to the identity instructions when SINGER_IDENTITY_GROUP_LEADS.
# dev55: identity guidance for the free-text voice descriptions, bands with no
# documented singer, and mixed-gender groups with one documented lead. Appended
# to the identity instructions when VOCAL_FREE_TEXT_GENDER is on.
SINGER_IDENTITY_DEV55_NOTE = """

LISTENING-MODEL VOICE DESCRIPTIONS (when that line appears in the vocal evidence):
It quotes what the listening model said, in its own words, about the voice it heard
("led by a male tenor", "a female lead in a soprano range"). It is acoustic evidence of
GENDER. When FINAL LEAD PROFILE is unknown or uncertain and these descriptions name one
gender consistently, use that gender -- above all for bands and acts with no documented
lead singer, where nothing else can settle it. It says nothing reliable about age beyond an
explicit "teenage"; set age from the identity evidence (or post_puberty_male / female_teen_adult
when there is none). It never overrides user corrections or a documented solo artist.

BANDS WITHOUT A DOCUMENTED SINGER: when the cover art shows the band and every pictured
member is male (or every one female), and nothing credits a singer of the other gender, the
lead is very probably that gender (CONFIDENCE=medium); agree with it when the vocal evidence
does not contradict it.

MIXED-GENDER GROUPS: a group with both male and female members is NOT mixed_leads by
membership alone. When the background names a lead singer, or the vocal evidence describes one
lead gender (with the other gender only as backing, chants or ad-libs), use the lead singer's
category and LEAD_SINGERS=one; mixed_leads is only for leads that are genuinely traded
between a man and a woman.
"""

# dev59: see VOCAL_UNBROKEN_SIGNATURE. Added only when the vocal evidence has
# the line, so every other track's identity prompt is unchanged.
SINGER_IDENTITY_UNBROKEN_NOTE = """

VOICE MATURITY (when the vocal evidence says the voice sounds UNBROKEN):
That line is acoustic evidence of a voice that had not dropped: the listening model heard a
woman, while its dedicated vocal pass rated the voice a child's or a teenage boy's. It counts
exactly like the female acoustic read under UNBROKEN MALE VOICES. With a documented male solo
artist, he is male; set the age category from his age at this recording (child_male_likely for
a child, adolescent_male_likely for a teenager -- an unbroken voice does not make a teenager a
child). In REASONING say plainly that his voice had not broken yet on this recording, and when
he was 15 or older, that this is later than usual. With a documented female artist it is simply
her voice.
"""

# dev46: see IDENTITY_FLIP_NOTE. Added only when the vocal evidence shows that
# pattern, so every other track's identity prompt is unchanged.
SINGER_IDENTITY_FLIP_NOTE = """

AUDIO PASSES DISAGREE ON GENDER (this track):
The first vocal pass and the listening model's own descriptions of the voice heard a female
voice; only the confirmation pass heard an adult man, and that one pass set FINAL LEAD PROFILE
to a male category. On the scanned library this flip has come from women and from boys whose
voices had not broken, not from adult men, and the audio alone cannot tell a woman's voice
from an unbroken boy's. So FINAL LEAD PROFILE is not a settled acoustic reading here and must
not decide the answer -- for a band, group or collective too, where the rule to prefer the
acoustic reading does not apply because the acoustic passes themselves disagree.
Decide gender and age from the identity evidence instead: the credited artist, the cover art,
the background, and reliable knowledge of who sings lead on this recording.
- A woman, or a group whose lead singers are women -> female_teen_adult.
- A boy or teenager whose voice had not broken -> follow UNBROKEN MALE VOICES
  (adolescent_male_likely or child_male_likely).
- post_puberty_male only when that identity evidence points to an adult man.
- With no identity evidence either way, the acoustic evidence leans to a female or unbroken
  voice, not to an adult man; say in REASONING that the passes disagreed.
"""


def _identity_flip_pattern(vocal_audit_text):
    """dev46 IDENTITY_FLIP_NOTE: True when the vocal evidence given to the
    identity step shows the first pass hearing a female voice, the
    confirmation pass a male one that became FINAL LEAD PROFILE, and the
    listening model's free descriptions only a female voice. Never raises."""
    try:
        t = str(vocal_audit_text or "")
        ini = re.search(r"(?m)^-?\s*Initial isolated lead profile:\s*([a-z_]+)", t)
        con = re.search(r"(?m)^-?\s*Confirmation pass:\s*([a-z_]+)", t)
        fin = re.search(r"(?m)^-?\s*FINAL LEAD PROFILE:\s*([a-z_]+)", t)
        ft = re.search(r"LISTENING-MODEL VOICE DESCRIPTIONS[^:]*:\s*(\d+) describe a male voice, (\d+) a female voice", t)
        if not (ini and con and fin and ft):
            return False
        return (ini.group(1) in FEMALE_LEAD_CATEGORIES
                and con.group(1) in MALE_LEAD_CATEGORIES
                and fin.group(1) in MALE_LEAD_CATEGORIES
                and int(ft.group(1)) == 0 and int(ft.group(2)) >= 1)
    except Exception:
        return False


_SINGER_IDENTITY_GROUP_PROMPT_PRE6 = """
Also output, on its own line after CONFIDENCE:

LEAD_SINGERS=<one|several|uncertain>

several = the lead vocal on this recording is shared by two or more singers: the credited
act is documented (in the supplied background, or reliably known to you) as a vocal group,
girl group, boy band or duo whose members share lead vocals on THIS recording, a featured
performer is explicitly credited with co-lead vocals, or the vocal evidence confirms distinct co-leads. The audio passes often hear similar voices
in a group as one voice, so a documented group with shared leads is enough.
one = a solo artist, or a band documented as having a single lead singer (a frontman with
backing singers is one).
uncertain = neither is clear.
For a group whose lead singers are all female (or all male), SINGER_IDENTITY is that shared
category, not mixed_leads; mixed_leads is for distinct male and female co-leads.
"""
_SINGER_IDENTITY_GROUP_PROMPT_07 = """
Also output, on its own line after CONFIDENCE:

LEAD_SINGERS=<one|several|uncertain>

several = the lead vocal on this recording is shared by two or more singers: the credited
act is documented (in the supplied background, or reliably known to you) as a vocal group,
girl group, boy band or duo whose members share lead vocals, a featured artist is credited,
or the vocal evidence reports distinct co-leads. The audio passes often hear similar voices
in a group as one voice, so a documented group with shared leads is enough.
one = a solo artist, or a band documented as having a single lead singer (a frontman with
backing singers is one).
uncertain = neither is clear.
For a group whose lead singers are all female (or all male), SINGER_IDENTITY is that shared
category, not mixed_leads; mixed_leads is for distinct male and female co-leads.
"""
SINGER_IDENTITY_GROUP_PROMPT = _SINGER_IDENTITY_GROUP_PROMPT_PRE6 if IDENTITY_SHARED_LEAD_EVIDENCE else _SINGER_IDENTITY_GROUP_PROMPT_07

# 0.7.1-pre8 MULTI_LEAD_BAND_RULE: bands with more than one lead singer who take different songs,
# and the names of a group's lead singers.
SINGER_IDENTITY_MULTI_LEAD_NOTE = """
BANDS WITH MORE THAN ONE LEAD SINGER: some bands have two or more lead singers who usually take
different songs, each song sung mainly by one of them. Being in such a band does not make this
recording LEAD_SINGERS=several -- several is only for a lead shared WITHIN this recording. (Vocal
groups, boy bands and girl groups whose members share the lead lines of a song are still several.)
- If it is documented who sings lead on THIS song (a song credit or the background above, or your
  reliable knowledge of this particular song), use LEAD_SINGERS=one and say in REASONING
  "<Name> sings lead on this song".
- If it is not documented for this song, do not guess one of them: use LEAD_SINGERS=one, do not
  name any of them as the singer in REASONING, and add a line
  LEAD_SINGER_CANDIDATES=<Name 1>; <Name 2> (the band's lead singers).
For a group whose members share the lead on this recording (LEAD_SINGERS=several), you may add a
line LEAD_SINGER_NAMES=<Name 1>; <Name 2>; ... naming the members who sing lead, only when the
background or your reliable knowledge documents them. Otherwise leave that line out.
"""

# 0.7.1-pre10 NAME_GENDER_NOTE: an undocumented artist's first name as gender evidence.
SINGER_IDENTITY_NAME_GENDER_NOTE = """
ARTIST NAME AS GENDER EVIDENCE: when the artist is not documented in the background, not reliably
known to you, and the cover shows no single performer, a clearly gendered first name may suggest
the singer's gender, but only as weak evidence -- never a unisex name (Alex, Sam, Charlie, Robin,
Kim ...), a name whose gender differs by country (Andrea, Nicola, Jean ...), or a stage name or
one-word name. When the first name is your only evidence of gender, say so in REASONING and add the
line GENDER_BASIS=name_only.
"""


# 0.7.1-pre11 YOUNG_LEAD_GROUP_CHECK: a group's mixed clips are not one young voice.
def _cover_shows_group(cover_obs):
    """True when the cover-art observations show three or more people."""
    cov = cover_obs if isinstance(cover_obs, dict) else {}
    raw = str(cov.get("raw") or "")
    people = str(cov.get("people") or (re.search(r"(?m)^PEOPLE=(.*)$", raw) or [None, ""])[1]).strip().lower()
    if COVER_FIELDS_TOLERANT:                     # 0.7.1-pre12
        people = people.strip("*_`\"' ")
        return bool(re.match(r"^(?:three|four|five|six|seven|eight|nine|ten|several|many|multiple|three_or_more|"
                             r"(?:a\s+)?(?:group|band)|[3-9]|\d\d)\b", people))
    return bool(re.match(r"^(?:three|four|five|six|several|many|multiple|three_or_more|[3-9]|\d\d)\b", people))


def _qualify_young_lead_in_group(identity_text, evidence_text, cover_obs):
    """A single young-male lead (adolescent/child) is set to uncertain, as 0.7 does for an adult
    or female lead, when confident sampled clips heard both a male and a female lead voice and
    the cover shows a group. A solo singer's mixed clips (a voice that hadn't fully deepened)
    are left alone. Otherwise identity_text unchanged."""
    t = str(identity_text or "")
    if (not t or "REFERENCE_VOCAL_CORRECTION=yes" in t or "UNBROKEN_MALE_IDENTITY_PRESERVED=yes" in t
            or not re.search(r"(?mi)^SINGER_IDENTITY=(?:adolescent_male_likely|child_male_likely)\s*$", t)
            or not re.search(r"(?mi)^LEAD_SINGERS=one\s*$", t) or not _cover_shows_group(cover_obs)):
        return identity_text
    line = re.search(r"Section voice check[^\n]*inconclusive -- sampled lead voices disagree[^\n]*",
                     str(evidence_text or ""))
    if not line or not all(re.search(r"\b" + g + r" \(high; (?:solo_lead|lead_plus_backing)\)", line.group(0))
                           for g in ("male", "female")):
        return identity_text
    return ("SINGER_IDENTITY=uncertain\n"
            "REASONING=High-confidence sampled clips suggest both male and female foreground leads, "
            "but corroboration did not establish distinct singers. The whole-track lead profile is provisional; "
            "neither a single lead nor mixed leads is established.\nCONFIDENCE=low\nLEAD_SINGERS=uncertain\n"
            "YOUNG_LEAD_GROUP_CHECK=applied")


# 0.7.1-pre12 IDENTITY_CANONICAL_REPLY (audit F-02): the identity reply's known fields in
# exact KEY=value form, so every rule reads what the tolerant parser already reads.
_IDENTITY_REPLY_KEYS = {
    "SINGER_IDENTITY", "REASONING", "CONFIDENCE", "LEAD_SINGERS", "LEAD_SINGER_CANDIDATES",
    "LEAD_SINGER_NAMES", "GENDER_BASIS", "UNBROKEN_MALE_LEAD_NAME", "UNBROKEN_MALE_LEAD_CONFIRMED",
    "FEMALE_COLEAD_NAME", "FEMALE_COLEAD_CONFIRMED", "MALE_IDENTITY_SOURCE", "FEMALE_IDENTITY_SOURCE",
}
_IDENTITY_CANON_TAGS = (set(VOCAL_LEAD_TAGS) - {"adult_male", "adult_female"}) | {"uncertain", "mixed_leads", "unknown"}


def _canonical_identity_reply(text):
    """Each known identity field rewritten as KEY=value (no bullets, bold, spaces or colon),
    with SINGER_IDENTITY / LEAD_SINGERS values put in the canonical spelling the program's own
    parsers return. Lines that are already canonical, and all other lines, are unchanged."""
    out = []
    for line in str(text or "").split("\n"):
        plain = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", _strip_md_emphasis(line))
        m = re.match(r"^\s*([A-Za-z][A-Za-z_ ]{2,40}?)\s*[:=]\s*(.*)$", plain)
        key = m.group(1).strip().replace(" ", "_").upper() if m else ""
        if key not in _IDENTITY_REPLY_KEYS:
            out.append(line)
            continue
        value = m.group(2).strip()
        if line.startswith(key + "="):
            value = line[len(key) + 1:].strip()   # already canonical: keep the text exactly
        if key == "SINGER_IDENTITY" and value.lower() not in _IDENTITY_CANON_TAGS:
            value = _parse_singer_identity("SINGER_IDENTITY=" + value) or value
        elif key == "LEAD_SINGERS" and value.lower() not in ("one", "several", "uncertain"):
            value = _parse_identity_lead_singers("LEAD_SINGERS=" + value) or value
        new = f"{key}={value}"
        out.append(line if line.strip() == new else new)
    return "\n".join(out)

SINGER_GROUP_NOTE = (
    "SINGER COUNT: LEAD_SINGERS=several above means the lead vocal is shared by several "
    "singers (a vocal group, a duo or a featured artist). The audio passes' "
    "NUM_DISTINCT_VOICES / 'single lead' reading cannot tell similar voices apart, so do not "
    "describe one lone singer: say several singers share the lead, without guessing who sings "
    "which line."
)
_IDENTITY_HEADING_RE = re.compile(r"(?m)^SINGER IDENTITY RESOLUTION \(")
# Closes a block written by _format_identity_resolution_block, so a blind
# question removes exactly that block and nothing after it.
_IDENTITY_BLOCK_END = "[end of singer identity]"


# dev43 wording swapped into SINGER_IDENTITY_GROUP_PROMPT when IDENTITY_SOLO_STAYS_ONE.
_IDENTITY_GROUP_OLD_PRE6 = (
    "girl group, boy band or duo whose members share lead vocals on THIS recording, a featured\n"
    "performer is explicitly credited with co-lead vocals, or the vocal evidence confirms distinct co-leads. The audio passes often hear similar voices\n"
    "in a group as one voice, so a documented group with shared leads is enough.\n"
)
_IDENTITY_GROUP_OLD_07 = 'girl group, boy band or duo whose members share lead vocals, a featured artist is credited,\nor the vocal evidence reports distinct co-leads. The audio passes often hear similar voices\nin a group as one voice, so a documented group with shared leads is enough.\n'
_IDENTITY_GROUP_OLD = _IDENTITY_GROUP_OLD_PRE6 if IDENTITY_SHARED_LEAD_EVIDENCE else _IDENTITY_GROUP_OLD_07
_IDENTITY_GROUP_NEW_PRE6 = _IDENTITY_GROUP_OLD_PRE6 + (
    "For a documented solo artist, pitch centres, octave doubling or register changes alone\n"
    "do not establish another singer. Preserve positive, recording-specific co-lead evidence.\n"
)
_IDENTITY_GROUP_NEW_07 = "girl group, boy band or duo whose members share lead vocals, or a featured artist is\ncredited. The audio passes often hear similar voices in a group as one voice, so a\ndocumented group with shared leads is enough. The vocal evidence alone never makes it\nseveral: a documented solo artist with no credited featured singer is one, even when the\nvocal evidence says mixed_leads or shows two pitch centres -- one singer's chest and head\nvoice, octave doubling and harmonies produce those.\n"
_IDENTITY_GROUP_NEW = _IDENTITY_GROUP_NEW_PRE6 if IDENTITY_SHARED_LEAD_EVIDENCE else _IDENTITY_GROUP_NEW_07


def _parse_identity_lead_singers(text):
    m = re.search(r"LEAD_SINGERS\s*[=:]\s*[\"'<\[]?\s*(one|several|uncertain|single|multiple|two)\b",
                  _strip_md_emphasis(text or ""), re.IGNORECASE)
    if not m:
        return ""
    v = m.group(1).lower()
    return {"single": "one", "multiple": "several", "two": "several"}.get(v, v)


def _has_identity_resolution_block(text):
    """True when the SINGER IDENTITY RESOLUTION block itself (its heading at a
    line start) is present -- not merely a sentence that names it."""
    return bool(_IDENTITY_HEADING_RE.search(str(text or "")))


# dev19 TAG_FEATURE_SHARED_LEAD: see the setting.
_TAG_FEAT_RE = re.compile(
    r"(?i)(?:^|[\s(\[])(?:feat(?:\.|\u2024)?|ft(?:\.|\u2024)|featuring)\s+(.+?)\s*(?:[)\]]|$|\s[-\u2013\u2014]\s)")

# 0.7.1-pre12 FEATURE_CREDIT_KEEP_ACT_NAMES: "X and the Y" / "X & the Y" is one act,
# and a title's "(with X)" credits a guest singer unless it names an orchestra, choir, mix...
_FEAT_WITH_RE = re.compile(r"(?i)[(\[]\s*with\s+([^)\]]+?)\s*[)\]]")
_FEAT_WITH_NOT_SINGER_RE = re.compile(
    r"(?i)\b(?:orchestra|symphony|philharmonic|sinfonia|strings?|choir|chorus|choral|ensemble|quartet|"
    r"trio|band|remix|mix|version|edit|live|acoustic|demo|instrumental|reprise|piano|guitar|horns?|love|you|me)\b")


def _split_feature_credit(credit):
    if not FEATURE_CREDIT_KEEP_ACT_NAMES:
        return re.split(r"\s*(?:,|&|\band\b|\bx\b|/)\s*", credit)
    return re.split(r"\s*(?:,|&(?!\s*[Tt]he\b)|\band\b(?!\s+[Tt]he\b)|\bx\b|/)\s*", credit)


def _feature_credit_strings(key, value):
    """The credit texts in one tag: feat./ft./featuring, and (pre12) a title's "(with X)"."""
    out = [m.group(1) for m in _TAG_FEAT_RE.finditer(value)]
    if FEATURE_CREDIT_KEEP_ACT_NAMES and key == "title":
        out += [m.group(1) for m in _FEAT_WITH_RE.finditer(value)
                if not _FEAT_WITH_NOT_SINGER_RE.search(m.group(1))
                and re.search(r"\b[A-Z][\w'’.-]+", m.group(1))]
    return out


def _tag_featured_singers(metadata):
    """Featured singers credited in the file's artist / title tags
    ("Band X feat. Singer Y", "Song (feat. X & Y)")."""
    names = []
    for k in ("artist", "title", "album_artist", "albumartist"):
        v = str((metadata or {}).get(k) or "") if isinstance(metadata, dict) else ""
        for credit in _feature_credit_strings(k, v):
            for nm in _split_feature_credit(credit):
                nm = nm.strip(" .\u2024")
                if nm and len(nm) >= 2 and nm.lower() not in [n.lower() for n in names]:
                    names.append(nm)
    return names


def _feature_credit_identity(identity_text, metadata, cover=None):
    """The identity's lead-singer count after the dev19 featured-singer check
    and the dev20 vocal-group check (both settings; each False skips it)."""
    identity_text = _unbroken_colead_identity(identity_text)
    out = _group_lead_identity(_feature_credit_identity_tags(identity_text, metadata), metadata)
    return _unbroken_voice_identity(out, metadata, cover) if UNBROKEN_VOICE_IDENTITY_CHECK else out


_UV_FEMALE_READ_RE = re.compile(
    r"(?i)\bfemale[- ]sounding\b|\b(?:read|reads|heard|labell?ed|classified|described|reports?|reported|"
    r"sounds?|profiled?)\b(?:\s+\S+){0,6}?\s+(?:as\s+)?(?:an?\s+)?(?:adult\s+|teen(?:age)?\s+|young\s+)?"
    r"(?:female|woman|women)(?:'s)?\b|\b(?:female|woman)(?:_teen_adult|\s+teen|\s+adult)?\s+"
    r"(?:read|reading|lead|voice|vocal|pitch|profile)\b|\b(?:adult_female|female_teen_adult)\b")
_UV_BORN_RE = re.compile(r"(?i)\bborn\b(?:\s+(?:in|on))?(?:\s+[A-Z][a-z]+)?(?:\s+\d{1,2},?)?\s+((?:19|20)\d\d)\b")
_UV_AGE_RE = re.compile(
    r"(?i)\b(\d{1,2})[- ]year[- ]old\b|\baged\s+(\d{1,2})\b|"
    r"\b(?:was|he was|just|only)\s+(?:about\s+|around\s+|roughly\s+|approximately\s+)?(\d{1,2})\b"
    r"(?=\s*(?:years?\b|at\b|when\b|on\b|,|;|\.|\)|$))")
_UV_ERA_RE = re.compile(
    r"(?i)\b(?:child[- ]star|boy soprano|teen[- ]pop debut|teen idol|pre-?teen|"
    r"(?:voice|it)\s+had\s+not\s+(?:yet\s+)?(?:broken|changed|dropped|deepened)|"
    r"before\s+his\s+voice\s+(?:broke|changed|dropped|deepened)|unbroken)\b")


def _cover_age_of_credited_artist(cover, metadata):
    """(age, basis) from the cover-art observations, or (None, "").

    Only when the cover shows exactly one person, presented as male, in a
    photograph-like portrait (not a baby, cherub, cartoon, drawing or statue),
    and its text names the credited artist. A child presentation without
    "teen" counts as 11, any teenage presentation as 15; an adult-only one
    gives no age, so nothing changes."""
    if not cover:
        return None, ""
    if isinstance(cover, dict):
        get = lambda k: str(cover.get(k) or "")
        people, cues, logos = get("people"), get("person_cues"), get("text_logos")
        raw = get("raw")
    else:
        raw = str(cover)
        people = cues = logos = ""
    if raw and not (people and cues):
        f = dict(re.findall(r"(?m)^([A-Z_]+)=(.*)$", raw))
        people = people or f.get("PEOPLE", "")
        cues = cues or f.get("PERSON_CUES", "")
        logos = logos or f.get("TEXT_LOGOS", "")
    if people.strip().lower() != "one" or not cues:
        return None, ""
    c = cues.lower()
    if not re.search(r"\b(?:male-presenting|male|boy|young man)\b", c) or re.search(r"\bfemale|\bgirl|\bwoman", c):
        return None, ""
    if re.search(r"\b(?:infant|toddler|baby|cherub|angel|cartoon|illustrat\w*|drawing|drawn|painted|statue|doll|mascot)\b", c):
        return None, ""
    md = metadata if isinstance(metadata, dict) else {}
    artist = re.split(r"(?i)\s+(?:feat\.?|ft\.|featuring|&|and)\s", str(md.get("artist") or "").strip())[0]
    norm = lambda x: re.sub(r"[^a-z0-9]+", "", x.lower())
    if len(norm(artist)) < 3 or norm(artist) not in norm(logos):
        return None, ""
    if re.search(r"\bteen|adolescen", c):
        return 15, "the cover art shows him with a teenage presentation"
    if re.search(r"\bchild|\bpre-?teen|\bboy\b", c):
        return 11, "the cover art shows him with a child presentation"
    return None, ""


def _identity_keep_young_male(new_out, old_out, documented_adult):
    """0.7.1-pre6: the stricter identity rules never take away a young-male finding 0.7's
    rules made (a boy heard as female-sounding must not end up an adult male), unless
    this singer is documented as an adult at the recording."""
    def tag(t):
        m = re.search(r"(?mi)^SINGER_IDENTITY=(\S+)", str(t or ""))
        return m.group(1) if m else ""
    if tag(old_out) in _YOUNG_MALE_TAGS and tag(new_out) not in _YOUNG_MALE_TAGS and not documented_adult():
        return old_out
    return new_out


def _unbroken_voice_identity__pre6_rules(identity_text, metadata, cover=None):
    """UNBROKEN_VOICE_IDENTITY_CHECK (see the setting): post_puberty_male or
    adult_male becomes adolescent_male_likely / child_male_likely when the
    identity's own reasoning records a female-sounding read and puts the
    singer at 16 or younger for this release. Anything else is returned
    unchanged."""
    t = str(identity_text or "")
    m = re.search(r"(?m)^SINGER_IDENTITY=(\S+)", t)
    if not m or m.group(1) not in ("post_puberty_male", "adult_male") or "UNBROKEN VOICE CHECK" in t or "AGE IDENTITY CHECK" in t:
        return identity_text
    if re.search(r"(?m)^LEAD_SINGERS=several\b", t):
        return identity_text
    md = metadata if isinstance(metadata, dict) else {}
    credit = " ".join(str(md.get(k) or "") for k in ("artist", "album_artist", "title"))
    if re.search(r"(?i)\b(?:feat\.?|ft\.|featuring)\s", credit):
        return identity_text
    rm = re.search(r"(?ms)^REASONING=(.*?)(?=^[A-Z_]+=|\Z)", t)
    reasoning = rm.group(1) if rm else t
    artist = str(md.get("artist") or "").strip()
    unbroken_at_recording = _identity_unbroken_recording_evidence(reasoning, artist)
    if not _UV_FEMALE_READ_RE.search(reasoning) and not unbroken_at_recording:
        return identity_text
    age = _identity_person_age(reasoning, "", _identity_recording_year(md), artist)
    if age is None and re.search(r"(?i)\bsolo\b", reasoning):
        age = _identity_person_age(reasoning, _artist_background_for_identity(md),
                                   _identity_recording_year(md), artist)
    if age is None and not unbroken_at_recording:
        return identity_text
    if age is not None and (age > 17 or (age == 17 and not unbroken_at_recording)):
        return identity_text
    new_cat = "child_male_likely" if age is not None and age <= 11 else "adolescent_male_likely"
    out = re.sub(r"(?m)^SINGER_IDENTITY=\S+", "SINGER_IDENTITY=" + new_cat, t, count=1)
    out = _identity_override_confidence(out)
    if unbroken_at_recording:
        note = (" UNBROKEN VOICE CHECK: corrected the earlier %s classification to %s because "
                "the reasoning explicitly places this singer's unbroken voice at this recording. "
                "%s No age was inferred from artwork, high pitch or a female-sounding read alone." % (
                    m.group(1), new_cat,
                    "The exact age is not established; the category is provisional." if age is None
                    else "The person-bound age is about %d." % age))
    else:
        note = (" AGE IDENTITY CHECK: corrected the earlier %s classification to %s from "
                "person-bound age evidence (about %d) and the female-sounding read. "
                "Age alone does not establish whether the voice had broken." % (m.group(1), new_cat, age))
    rm = re.search(r"(?ms)^REASONING=(.*?)(?=^[A-Z_]+=|\Z)", out)
    if rm:
        end = rm.end(1)
        body = out[:end].rstrip("\n")
        out = body + note + "\n" + out[end:].lstrip("\n") if out[end:].strip() else body + note + "\n"
    else:
        out = out.rstrip("\n") + "\nREASONING=" + note.strip() + "\n"
    return out


def _unbroken_voice_identity__pre6(identity_text, metadata, cover=None):
    new = _unbroken_voice_identity__pre6_rules(identity_text, metadata, cover)
    old = _unbroken_voice_identity__07(identity_text, metadata, cover)
    def adult():
        md = metadata or {}
        try:
            bg = _artist_background_for_identity(md)
        except Exception:
            bg = ""
        return bool(_documented_adult__pre6(identity_text, bg, _identity_recording_year(md),
                                            str(md.get("artist") or "")))
    return _identity_keep_young_male(new, old, adult)



def _unbroken_voice_identity__07(identity_text, metadata, cover=None):
    """UNBROKEN_VOICE_IDENTITY_CHECK (see the setting): post_puberty_male or
    adult_male becomes adolescent_male_likely / child_male_likely when the
    identity's own reasoning records a female-sounding read and puts the
    singer at 16 or younger for this release. Anything else is returned
    unchanged."""
    t = str(identity_text or "")
    m = re.search(r"(?m)^SINGER_IDENTITY=(\S+)", t)
    if not m or m.group(1) not in ("post_puberty_male", "adult_male") or "UNBROKEN VOICE CHECK" in t:
        return identity_text
    if re.search(r"(?m)^LEAD_SINGERS=several\b", t):
        return identity_text
    md = metadata if isinstance(metadata, dict) else {}
    credit = " ".join(str(md.get(k) or "") for k in ("artist", "album_artist", "title"))
    if re.search(r"(?i)\b(?:feat\.?|ft\.|featuring)\s", credit):
        return identity_text
    rm = re.search(r"(?ms)^REASONING=(.*?)(?=^[A-Z_]+=|\Z)", t)
    reasoning = rm.group(1) if rm else t
    if not _UV_FEMALE_READ_RE.search(reasoning):
        return identity_text
    age, basis = None, ""
    bm = _UV_BORN_RE.search(reasoning)
    rel = None
    for k in ("original_year", "originalyear", "year", "date", "original_date"):
        ym = re.search(r"(?:19|20)\d\d", str(md.get(k) or ""))
        if ym:
            rel = int(ym.group(0))
            break
    if bm:
        born = int(bm.group(1))
        if rel is None:
            later = [int(y) for y in re.findall(r"\b((?:19|20)\d\d)\b", reasoning[bm.end():]) if int(y) > born]
            rel = later[0] if later else None
        if rel is not None and 0 < rel - born < 40:
            age, basis = rel - born, "born %d, released %d" % (born, rel)
    if age is None:
        for am in _UV_AGE_RE.finditer(reasoning):
            v = next(int(g) for g in am.groups() if g)
            if 5 <= v <= 30:
                age, basis = v, "stated age %d" % v
                break
    if age is None and re.search(r"(?i)\bsolo\b", reasoning):
        # The reasoning credits a solo male but gives no age: take his birth
        # year from the same local background the identity pass was given.
        bg = _artist_background_for_identity(md)
        bm = _UV_BORN_RE.search(bg or "")
        if bm and rel is not None and 0 < rel - int(bm.group(1)) < 40:
            age, basis = rel - int(bm.group(1)), "born %s per the artist background, released %d" % (bm.group(1), rel)
    if age is None and re.search(r"(?i)\bsolo\b", reasoning):
        # Still no age (no background on the artist): the cover art, when it
        # shows one real person and carries the credited artist's name, so
        # that person is reasonably taken to be him.
        age, basis = _cover_age_of_credited_artist(cover, md)
    if age is None and _UV_ERA_RE.search(reasoning):
        age, basis = 15, "the reasoning places this recording before his voice broke"
    if age is None or age > 16:
        return identity_text
    new_cat = "child_male_likely" if age <= 11 else "adolescent_male_likely"
    out = re.sub(r"(?m)^SINGER_IDENTITY=\S+", "SINGER_IDENTITY=" + new_cat, t, count=1)
    note = (" UNBROKEN VOICE CHECK: %s was kept although the listening model heard a female-sounding lead and "
            "the singer was %s at this release (%s); a credited male of that age with a female-sounding read is "
            "an unbroken or still-changing voice, so the category is %s." % (
                m.group(1), "about %d" % age if not basis.startswith("the reasoning") else "a boy",
                basis, new_cat))
    rm = re.search(r"(?ms)^REASONING=(.*?)(?=^[A-Z_]+=|\Z)", out)
    if rm:
        end = rm.end(1)
        body = out[:end].rstrip("\n")
        out = body + note + "\n" + out[end:].lstrip("\n") if out[end:].strip() else body + note + "\n"
    else:
        out = out.rstrip("\n") + "\nREASONING=" + note.strip() + "\n"
    return out


def _unbroken_voice_identity(*args, **kwargs):
    """SINGER_IDENTITY_PRE6 picks the 0.7.1-pre6 version; False gives 0.7's."""
    return (_unbroken_voice_identity__pre6 if SINGER_IDENTITY_PRE6 else _unbroken_voice_identity__07)(*args, **kwargs)


# dev20 GROUP_LEAD_STAYS_SEVERAL: see the setting.
_VOCAL_GROUP_RE = re.compile(
    r"(?i)\b(?:boy[- ]?band|girl[- ]?group|vocal group|harmony group|brothers|sisters|siblings|"
    r"brother (?:trio|duo|act|band|group)|sister (?:trio|duo|act|band|group)|family (?:band|group|act)|duo)\b")
_SOLO_ACT_RE = re.compile(r"(?i)\bsolo (?:artist|act|singer|performer|male artist|female artist)\b")
_LEAD_ROLE_RE = re.compile(
    r"\b(?:vocalist|singer|frontman|frontwoman|front-man|front-woman|(?:lead )?vocals (?:by|from))\b", re.I)
_LEAD_CLAUSE_END_RE = re.compile(r"(?<![A-Z])(?<!\bJr)(?<!\bSr)(?<!\bSt)\.(?=\s|$)|[;!?\n]|\s[\u2013\u2014-]\s")
_LEAD_NAME_TOKEN = (r"(?:[A-Z][a-z]+(?:[-'\u2019][A-Z]?[a-z]+)*|Mc[A-Z][a-z]+|Mac[A-Z][a-z]+|"
                    r"O['\u2019][A-Z][a-z]+|[A-Z]\.)")
_LEAD_NAME_RUN_RE = re.compile(
    r"(?<![\w\u2019'-])" + _LEAD_NAME_TOKEN + r"(?:\s+(?:(?:de|van|von|la|le|del|der|da|di|du|dos|bin|al)\s+)?"
    + _LEAD_NAME_TOKEN + r"){0,3}(?![\w\u2019'-])")
_LEAD_NOT_PERSON = {
    "the", "track", "song", "album", "single", "record", "recording", "release", "metadata", "background",
    "vocal", "vocals", "acoustic", "final", "identity", "wikipedia", "listening", "music", "band", "group",
    "male", "female", "documented", "credited", "solo", "artist", "lead", "singer", "vocalist", "frontman",
    "american", "british", "english", "canadian", "irish", "scottish", "welsh", "australian", "swedish",
    "january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
    "november", "december", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    "both", "his", "her", "their", "its", "it", "this", "that", "these", "those", "he", "she", "they",
    "a", "an", "and", "or", "but", "with", "for", "of", "on", "in", "at", "by", "as", "is", "was",
    "acoustically", "evidence", "confirmation", "cover", "art", "pass", "profile", "tag",
    "tags", "feature", "featured", "check", "era", "line-up", "lineup", "debut", "boy", "girl",
}


def _lead_fold(s):
    return re.sub(r"[^a-z0-9]+", "", unicodedata.normalize("NFKD", str(s or "")).lower())


def _named_lead_singer__pre6(reason, act_names=()):
    """The person the identity reasoning names as the singer: a capitalised
    personal name after "vocalist" / "singer" / "frontman" / "frontwoman" /
    "vocals by" in the same clause (words in between allowed), or a name
    followed by "sings lead" / "on lead vocals". Band names, other
    capitalised words, possessives and the act's own name are skipped. ''
    when none. The same rule as musiclyse_export 2.2.3."""
    r = _identity_asserted_text(reason)
    acts = [_lead_fold(a) for a in act_names if _lead_fold(a)]

    def ok(run):
        words = run.split()
        if not words or any(w.lower().strip(".,'\u2019") in _LEAD_NOT_PERSON for w in words):
            return False
        f = _lead_fold(run)
        if any(f == a or f in a for a in acts):
            return False
        return not all(re.fullmatch(r"[A-Z]\.", w) for w in words)

    for m in _LEAD_ROLE_RE.finditer(r):
        if _identity_secondary_role(r, m.start()):
            continue
        rest = r[m.end():]
        stop = _LEAD_CLAUSE_END_RE.search(rest)
        clause = rest[:stop.start()] if stop else rest
        for k, nm in enumerate(_LEAD_NAME_RUN_RE.finditer(clause)):
            run = nm.group(0).strip()
            if run.endswith(".") and not re.search(r"\b[A-Z]\.$", run):
                run = run[:-1]
            if re.search(r"['\u2019]s$", run):
                continue
            if len(run.split()) == 1:
                lead_in = clause[:nm.start()].strip(" ,:")
                if not (k == 0 and re.fullmatch(r"(?i)(?:(?:is|was|,)\s*)?(?:named\s+)?", lead_in)):
                    continue
            if ok(run):
                return run
    for m in re.finditer(r"(" + _LEAD_NAME_TOKEN + r"(?:\s+" + _LEAD_NAME_TOKEN + r"){1,3})\s+(?:sings|sang|singing|is on|on) "
                         r"lead\b", r):
        if ok(m.group(1)):
            return m.group(1)
    return ""


def _named_lead_singer__07(reason, act_names=()):
    """The person the identity reasoning names as the singer: a capitalised
    personal name after "vocalist" / "singer" / "frontman" / "frontwoman" /
    "vocals by" in the same clause (words in between allowed), or a name
    followed by "sings lead" / "on lead vocals". Band names, other
    capitalised words, possessives and the act's own name are skipped. ''
    when none. The same rule as musiclyse_export 2.2.3."""
    r = str(reason or "")
    acts = [_lead_fold(a) for a in act_names if _lead_fold(a)]

    def ok(run):
        words = run.split()
        if not words or any(w.lower().strip(".,'\u2019") in _LEAD_NOT_PERSON for w in words):
            return False
        f = _lead_fold(run)
        if any(f == a or f in a for a in acts):
            return False
        return not all(re.fullmatch(r"[A-Z]\.", w) for w in words)

    for m in _LEAD_ROLE_RE.finditer(r):
        rest = r[m.end():]
        stop = _LEAD_CLAUSE_END_RE.search(rest)
        clause = rest[:stop.start()] if stop else rest
        for k, nm in enumerate(_LEAD_NAME_RUN_RE.finditer(clause)):
            run = nm.group(0).strip()
            if run.endswith(".") and not re.search(r"\b[A-Z]\.$", run):
                run = run[:-1]
            if re.search(r"['\u2019]s$", run):
                continue
            if len(run.split()) == 1:
                lead_in = clause[:nm.start()].strip(" ,:")
                if not (k == 0 and re.fullmatch(r"(?i)(?:(?:is|was|,)\s*)?(?:named\s+)?", lead_in)):
                    continue
            if ok(run):
                return run
    for m in re.finditer(r"(" + _LEAD_NAME_TOKEN + r"(?:\s+" + _LEAD_NAME_TOKEN + r"){1,3})\s+(?:sings|sang|singing|is on|on) "
                         r"lead\b", r):
        if ok(m.group(1)):
            return m.group(1)
    return ""


def _named_lead_singer(*args, **kwargs):
    """SINGER_IDENTITY_PRE6 picks the 0.7.1-pre6 version; False gives 0.7's."""
    return (_named_lead_singer__pre6 if SINGER_IDENTITY_PRE6 else _named_lead_singer__07)(*args, **kwargs)


def _group_lead_identity__pre6(identity_text, metadata):
    """dev20 GROUP_LEAD_STAYS_SEVERAL: an identity that says LEAD_SINGERS=one
    while its own reasoning documents the act as a vocal group (boy band, girl
    group, brothers, sisters, a duo ...) and names no single lead singer is
    set to several -- the identity prompt's own rule (a documented group with
    shared leads is enough). The identity model rewords its reasoning on every
    run: on the sibling-band test track dev17 said several and dev19 one from the same facts
    (two different wordings of the same fact). A band with a named
    frontman, or a solo artist, is left as it is."""
    t = str(identity_text or "")
    try:
        if (not GROUP_LEAD_STAYS_SEVERAL or not t or "GROUP LEAD CHECK" in t
                or _parse_identity_lead_singers(t) != "one"):
            return identity_text
        reason = _identity_reason_text(t)
        if not _identity_shared_leads(reason) or not _VOCAL_GROUP_RE.search(reason) or _SOLO_ACT_RE.search(reason):
            return identity_text
        md = metadata if isinstance(metadata, dict) else {}
        artist = str(md.get("artist") or "").strip()
        acts = [artist, re.split(r"(?i)\s+(?:feat(?:\.|\u2024)?|ft(?:\.|\u2024)|featuring|with)\s", artist)[0],
                str(md.get("title") or ""), str(md.get("album") or "")]
        if _named_lead_singer(reason, [a for a in acts if a]):
            return identity_text
        t2 = re.sub(r"(LEAD_SINGERS\s*[=:]\s*[\"'<\[]?\s*)(one|single)\b", r"\g<1>several", t, count=1,
                    flags=re.IGNORECASE)
        t2 = _identity_override_confidence(t2)
        who = artist or "the act"
        add = (f" GROUP LEAD CHECK: the reasoning documents {who} as a vocal group and names no single lead "
               "singer, so the lead is shared (a documented group with shared leads is enough), even though "
               "the audio passes heard one voice.")
        m = re.search(r"(?m)^(\W*REASONING\s*[=:].*)$", t2)
        return t2[:m.end()] + add + t2[m.end():] if m else t2.rstrip() + "\n" + add.strip()
    except Exception:
        return identity_text


def _group_lead_identity__07(identity_text, metadata):
    """dev20 GROUP_LEAD_STAYS_SEVERAL: an identity that says LEAD_SINGERS=one
    while its own reasoning documents the act as a vocal group (boy band, girl
    group, brothers, sisters, a duo ...) and names no single lead singer is
    set to several -- the identity prompt's own rule (a documented group with
    shared leads is enough). The identity model rewords its reasoning on every
    run: on the sibling-band test track dev17 said several and dev19 one from the same facts
    (two different wordings of the same fact). A band with a named
    frontman, or a solo artist, is left as it is."""
    t = str(identity_text or "")
    try:
        if (not GROUP_LEAD_STAYS_SEVERAL or not t or "GROUP LEAD CHECK" in t
                or _parse_identity_lead_singers(t) != "one"):
            return identity_text
        mr = re.search(r"(?m)^\W*REASONING\s*[=:]\s*(.*)$", t)
        reason = mr.group(1) if mr else ""
        if not _VOCAL_GROUP_RE.search(reason) or _SOLO_ACT_RE.search(reason):
            return identity_text
        md = metadata if isinstance(metadata, dict) else {}
        artist = str(md.get("artist") or "").strip()
        acts = [artist, re.split(r"(?i)\s+(?:feat(?:\.|\u2024)?|ft(?:\.|\u2024)|featuring|with)\s", artist)[0],
                str(md.get("title") or ""), str(md.get("album") or "")]
        if _named_lead_singer(reason, [a for a in acts if a]):
            return identity_text
        t2 = re.sub(r"(LEAD_SINGERS\s*[=:]\s*[\"'<\[]?\s*)(one|single)\b", r"\g<1>several", t, count=1,
                    flags=re.IGNORECASE)
        who = artist or "the act"
        add = (f" GROUP LEAD CHECK: the reasoning documents {who} as a vocal group and names no single lead "
               "singer, so the lead is shared (a documented group with shared leads is enough), even though "
               "the audio passes heard one voice.")
        m = re.search(r"(?m)^(\W*REASONING\s*[=:].*)$", t2)
        return t2[:m.end()] + add + t2[m.end():] if m else t2.rstrip() + "\n" + add.strip()
    except Exception:
        return identity_text


def _group_lead_identity(*args, **kwargs):
    """IDENTITY_SHARED_LEAD_EVIDENCE picks the 0.7.1-pre6 version; False gives 0.7's."""
    return (_group_lead_identity__pre6 if IDENTITY_SHARED_LEAD_EVIDENCE else _group_lead_identity__07)(*args, **kwargs)


def _feature_credit_identity_tags__pre6(identity_text, metadata):
    """dev19: a singer identity that says LEAD_SINGERS=one while the file's
    tags credit a featured singer is set to several, with the reason, as the
    identity prompt asks. The identity model followed MOSS-Music's confident
    "single male lead throughout" over the "feat. Singer Y" credit
    on the duet test track, so her verse was never mentioned. The singer
    category (SINGER_IDENTITY) is left as it is: it names the main lead."""
    t = str(identity_text or "")
    try:
        if (not TAG_FEATURE_SHARED_LEAD or not t or "TAG FEATURE CHECK" in t
                or _parse_identity_lead_singers(t) != "one"):
            return identity_text
        names = [name for name in _tag_featured_singers(metadata)
                 if _identity_shared_leads(_identity_reason_text(t), [name])]
        if not names:
            return identity_text
        t2 = re.sub(r"(LEAD_SINGERS\s*[=:]\s*[\"'<\[]?\s*)(one|single)\b", r"\g<1>several", t, count=1,
                    flags=re.IGNORECASE)
        t2 = _identity_override_confidence(t2)
        add = (f" TAG FEATURE CHECK: a {' and '.join(names)} feature is credited in the file's tags, and the reasoning explicitly documents shared lead vocals.")
        m = re.search(r"(?m)^(\W*REASONING\s*[=:].*)$", t2)
        return t2[:m.end()] + add + t2[m.end():] if m else t2.rstrip() + "\n" + add.strip()
    except Exception:
        return identity_text


def _feature_credit_identity_tags__07(identity_text, metadata):
    """dev19: a singer identity that says LEAD_SINGERS=one while the file's
    tags credit a featured singer is set to several, with the reason, as the
    identity prompt asks. The identity model followed MOSS-Music's confident
    "single male lead throughout" over the "feat. Singer Y" credit
    on the duet test track, so her verse was never mentioned. The singer
    category (SINGER_IDENTITY) is left as it is: it names the main lead."""
    t = str(identity_text or "")
    try:
        if (not TAG_FEATURE_SHARED_LEAD or not t or "TAG FEATURE CHECK" in t
                or _parse_identity_lead_singers(t) != "one"):
            return identity_text
        names = _tag_featured_singers(metadata)
        if not names:
            return identity_text
        t2 = re.sub(r"(LEAD_SINGERS\s*[=:]\s*[\"'<\[]?\s*)(one|single)\b", r"\g<1>several", t, count=1,
                    flags=re.IGNORECASE)
        add = (f" TAG FEATURE CHECK: a {' and '.join(names)} feature is credited in the file's tags, so the "
               "lead is shared, even though the audio passes heard one voice.")
        m = re.search(r"(?m)^(\W*REASONING\s*[=:].*)$", t2)
        return t2[:m.end()] + add + t2[m.end():] if m else t2.rstrip() + "\n" + add.strip()
    except Exception:
        return identity_text


def _feature_credit_identity_tags(*args, **kwargs):
    """IDENTITY_SHARED_LEAD_EVIDENCE picks the 0.7.1-pre6 version; False gives 0.7's."""
    return (_feature_credit_identity_tags__pre6 if IDENTITY_SHARED_LEAD_EVIDENCE else _feature_credit_identity_tags__07)(*args, **kwargs)


def _format_identity_resolution_block(identity_text):
    # Blank lines inside the model's answer are closed up so the block stays
    # one block (the blind-question filter removes it heading to heading).
    body = re.sub(r"\n\s*\n+", "\n", str(identity_text or "").strip())
    out = ("\n\nSINGER IDENTITY RESOLUTION (combined audio + metadata + cover art; use for "
           "who-is-singing questions about the PERSON, never to re-describe the sound):\n" + body)
    if SINGER_IDENTITY_GROUP_LEADS and _parse_identity_lead_singers(body) == "several":
        out += "\n" + SINGER_GROUP_NOTE
    return out + "\n" + _IDENTITY_BLOCK_END


def _identity_ctx_for_prompt(prompt, ctx):
    """Smallest context (doubling from ctx, at most 32768) whose per-message
    cap in _prepare_ollama_messages holds the whole identity prompt, so the
    middle -- the track's tags, vocal evidence and cover notes -- is never cut."""
    ctx = int(ctx)
    while ctx < 32768:
        cap = min(MAX_MESSAGE_CHARS_FOR_OLLAMA,
                  max(4000, int(ctx * HISTORY_CHAR_BUDGET_FACTOR) // 3))
        if len(prompt) <= cap:
            break
        ctx *= 2
    return ctx


def _identity_background_by_artist_and_title(metadata, artist):
    """Background for the identity pass: the act's own article(s) first, then
    the song's. Album look-ups are left out -- they fuzzy-matched other acts'
    records ("Album P" -> "Album Q", "Album R" -> "Album S")
    -- and a single-word act name is otherwise searched last, after every
    song-title query has filled the article budget."""
    act = _wiki_clean_artist_for_search(artist) or artist
    title = str((metadata or {}).get("title") or "").strip() or None
    parts = [build_wiki_context_block_multi(artist=act, use_question_entities=False) or ""]
    if title:
        parts.append(build_wiki_context_block_multi(artist=act, title=title,
                                                    use_question_entities=False) or "")
    return "\n\n".join(p for p in parts if p.strip())


def _singer_identity_prompt_fitted(instructions, corr_block, meta_block, vocal_audit_text,
                                   cover_block, wiki_text):
    """Identity prompt with the background first and this track's own evidence
    and the answer format last, where a long background cannot push them out."""
    parts = [instructions.rstrip()]
    if wiki_text:
        if len(wiki_text) > 7000:
            wiki_text = wiki_text[:7000] + " ..."
        parts.append(
            "BACKGROUND ON THE CREDITED ARTIST (reference material, looked up by name -- it can "
            "include articles about other people or records with similar names; use an article "
            "only when it is clearly about the credited artist or this song):\n"
            + wiki_text
            + "\n\nUse this ONLY to establish who the artist is, whether they are a man or a woman "
              "or a group (and who in it sings lead) where stated or unambiguous, and -- if this "
              "recording's era is identifiable from the track metadata -- whether that era is "
              "documented as a child/adolescent period for them. Take nothing else from it."
        )
    parts.append(f"USER CORRECTIONS (ground truth):\n{corr_block}")
    parts.append(f"TRACK METADATA:\n{meta_block}")
    parts.append(f"VOCAL EVIDENCE:\n{(vocal_audit_text or '').strip() or 'No vocal evidence available.'}")
    parts.append(f"COVER ART OBSERVATIONS:\n{cover_block}")
    parts.append(
        "Now answer for THIS track (the one in TRACK METADATA above) in exactly the output "
        "format given in the instructions: SINGER_IDENTITY=..., REASONING=..., CONFIDENCE=..."
        + (", LEAD_SINGERS=..." if SINGER_IDENTITY_GROUP_LEADS else "") + "."
    )
    if "UNBROKEN VOICE WITH CO-LEADS" in instructions:
        parts.append("Also follow the conditional per-performer output instructions above when both "
                     "lead identities and the unbroken voice are supported; otherwise omit those extra fields.")
    return "\n\n".join(parts)


def _artist_background_for_identity(metadata):
    """Local reference-DB background on the credited artist, for the singer
    identity pass only.

    Looked up on demand from the track metadata because the finalize stage has
    no wiki block of its own -- the writer's copy is assembled later, in the
    chat layer. Failure is silent and returns "": background is a helpful extra
    signal here, never a requirement, and identity resolution must still run
    normally when the artist isn't in the database.
    """
    try:
        artist = str((metadata or {}).get("artist") or "").strip()
        if not artist:
            return ""
        if SINGER_IDENTITY_FIT_PROMPT:
            return _identity_background_by_artist_and_title(metadata, artist)
        return build_wiki_context_block_multi(
            artist=artist,
            title=str((metadata or {}).get("title") or "").strip() or None,
            album=str((metadata or {}).get("album") or "").strip() or None,
            use_question_entities=False,
        ) or ""
    except Exception:
        return ""


# dev58: see IDENTITY_FROM_ARTIST_WHEN_OPEN.
_IDENTITY_UNCERTAIN_OLD = (
    "- uncertain -> do not invent a gender from metadata or cover alone when pitch is unremarkable"
)
_IDENTITY_UNCERTAIN_NEW = (
    "- uncertain / unknown (including a reading left open because the listening passes disagreed) ->\n"
    "  when the credited artist is a documented solo performer, or the background names who sings\n"
    "  lead, set the gender (and the age at this recording) from that documented identity -- see\n"
    "  KNOWN ARTIST IDENTITY; that is exactly what identity evidence is for. Otherwise (a band or\n"
    "  act with no documented lead singer) do not invent a gender from metadata or cover alone when\n"
    "  pitch is unremarkable; use the LISTENING-MODEL VOICE DESCRIPTIONS rules when they appear.\n"
    "  Identity settles only who sings, their gender and age -- never the style or genre of the\n"
    "  recording."
)


# 0.7-pre7: teenage words only -- "young" and "youthful" also describe a man in
# his twenties ("young adult"), so on their own they no longer count.
_YOUNG_COVER_RE = re.compile(r"(?i)\b(?:teen|teens|teenage|teenager|teenaged|boy|boys|child|kid|kids|youth|"
                             r"adolescent|schoolboy|pre-?teen)\b")


def _documented_adult__pre6(identity_text, background, year, name=""):
    """Adult evidence explicitly bound to this singer."""
    age = _identity_person_age(_identity_reason_text(identity_text) or identity_text, background, year, name)
    return f"documented age about {age}" if age is not None and age >= 18 else None


def _documented_adult__07(identity_text, background, year, name=""):
    """0.7-pre5/pre6: the birth year (or stated age) that documents the singer as
    an adult at the time, or None. A stated age of 18+ in the identity
    reasoning counts; in the background, only a birth date right next to the
    singer's own name (full name, or surname of 3+ letters) counts -- the
    background can include articles about other people with similar names."""
    for m in re.finditer(r"(?i)\b(?:was|aged?|age)\s+~?(\d{1,2})\b", str(identity_text or "")):
        if int(m.group(1)) >= 18:
            return "stated age " + m.group(1)
    try:
        y = int(str(year or "")[:4])
    except ValueError:
        y = None
    bg, nm = str(background or ""), re.sub(r"\s+", " ", str(name or "")).strip()
    if not (y and bg and nm):
        return None
    keys = [nm]
    last = nm.split(" ")[-1]
    if len(last) >= 3 and last.lower() != nm.lower():
        keys.append(last)
    for k in keys:
        kq = re.escape(k).replace(r"\ ", r"\s+")
        for rx in (kq + r"(?:\s+\w+){0,3}\s*\(\s*(?:born|b\.)\s[^)]{0,40}?\b(19\d\d|20[0-2]\d)\b",
                   kq + r"[^.\n]{0,40}?\bwas born\b[^.\n]{0,40}?\b(19\d\d|20[0-2]\d)\b",
                   kq + r"\s*\(\s*(19\d\d|20[0-2]\d)\s*[–-]"):
            for m in re.finditer(r"(?i)\b" + rx, bg):
                if y - int(m.group(1)) >= 18:
                    return "born " + m.group(1)
    return None


def _documented_adult(*args, **kwargs):
    """SINGER_IDENTITY_PRE6 picks the 0.7.1-pre6 version; False gives 0.7's."""
    return (_documented_adult__pre6 if SINGER_IDENTITY_PRE6 else _documented_adult__07)(*args, **kwargs)


def _young_male_voice_check__pre6_rules(identity_text, evidence_text, cover_obs, metadata, background=None):
    """0.7-pre5 YOUNG_MALE_VOICE_CHECK: identity_text with post_puberty_male
    changed to adolescent_male_likely when a credited solo male singer was
    heard as female-sounding by the whole-track passes, the cover shows one
    young person, and nothing documents an adult or a falsetto/head voice.
    Otherwise identity_text unchanged. background: a callable returning the
    reference background text (only looked up when everything else holds)."""
    t = str(identity_text or "")
    if not YOUNG_MALE_VOICE_CHECK or not re.search(r"(?mi)^SINGER_IDENTITY=post_puberty_male\s*$", t):
        return identity_text
    if not re.search(r"(?mi)^LEAD_SINGERS=one\s*$", t) or "REFERENCE_VOCAL_CORRECTION=yes" in t:
        return identity_text
    artist = str((metadata or {}).get("artist") or "")
    if re.search(r"(?i)\b(?:feat\.?|featuring|ft\.?|with|and|&|x|vs\.?)\s", artist + " "):
        return identity_text
    ev = str(evidence_text or "")
    if not re.search(r"(?mi)Initial isolated lead profile:\s*(?:female_teen_adult|adult_female|young_female)\b", ev):
        return identity_text
    pf = [int(x) for x in re.findall(r"(?i)female teen/adult\s+(\d{1,3})%", ev)]
    if not pf or max(pf) < 90:
        return identity_text
    if re.search(r"(?i)\bfalsetto\b|\bhead[- ]voice\b|\bcountertenor\b", ev):
        return identity_text
    cov = cover_obs if isinstance(cover_obs, dict) else {}
    if str(cov.get("people") or "").strip().lower() not in ("one", "1"):
        return identity_text
    if not _YOUNG_COVER_RE.search(str(cov.get("person_cues") or "")):
        return identity_text
    bg = ""
    try:
        bg = background() if callable(background) else (background or "")
    except Exception:
        bg = ""
    _adult = _documented_adult(t, bg, _identity_recording_year(metadata), artist)
    if _adult:
        # 0.7-pre6: say why, so a save shows the check ran and what stopped it.
        return t.rstrip("\n") + f"\nYOUNG_MALE_VOICE_CHECK=not applied: documented adult ({_adult})\n"
    age = _identity_person_age(_identity_reason_text(t), bg, _identity_recording_year(metadata), artist)
    provisional = age is None
    if provisional:
        # Restore 0.7's acoustic/portrait fallback without inventing a numeric
        # age or accepting unrelated people, cartoons or an adult-only cover.
        cover_age, _ = _cover_age_of_credited_artist(cov, metadata)
        if cover_age is None or not _identity_asserted_text(str(cov.get("person_cues") or "")):
            return identity_text
        reason = _identity_asserted_text(_identity_reason_text(t))
        if not re.search(r"(?i)\bsolo\b", reason):
            return identity_text
    elif not 12 <= age <= 17:
        return identity_text
    out = re.sub(r"(?mi)^SINGER_IDENTITY=post_puberty_male\s*$", "SINGER_IDENTITY=adolescent_male_likely", t, count=1)
    out = _identity_override_confidence(out)
    note = (" [Provisional young-male voice check: the credited solo male has a strongly female-sounding "
            "initial audio reading and an artist-named cover portrait with teenage cues. The acoustic "
            "and identity evidence is compatible with an unbroken or still-changing male voice; "
            "the exact age and biological puberty status are not established.]") if provisional else (" [Set to adolescent_male_likely by the young-male voice check: a credited solo male singer whose "
            "whole-track listening passes heard a female-sounding voice has a documented teenage age at this recording; "
            "age alone does not establish whether his voice had broken.]")
    if re.search(r"(?m)^REASONING=", out):
        out = re.sub(r"(?m)^(REASONING=.*)$", lambda m: m.group(1) + note, out, count=1)
    else:
        out += "\nREASONING=" + note.strip()
    return out.rstrip("\n") + "\nYOUNG_MALE_VOICE_CHECK=applied\n"


# 0.7.1-pre9 (FEMALE_READ_ANY_WORDING): the whole-track pass's female share, however it is
# worded ("female teen/adult 85%", "adult female 95%", "young adult female 90%").
_FEMALE_READ_PCT_RE = re.compile(
    r"(?i)\b(?:(?:young\s+)?adult\s+female|young\s+female|teen(?:age)?\s+female|"
    r"female(?:\s+teen/adult|\s+teen|\s+adult)?)\s*[:=]?\s*(?:[A-Z]%\s*\(\s*)?(\d{1,3})\s*%")


def _female_read_evidence(evidence_text):
    """evidence_text, plus one normalised "female teen/adult N%" line when the whole-track
    pass gave its female share in other words, so the young-male checks can read it.
    Unchanged when the usual wording is present or no share is given."""
    ev = str(evidence_text or "")
    if re.search(r"(?i)female teen/adult\s+\d{1,3}%", ev):
        return evidence_text
    pcts = []
    for line in re.findall(r"(?m)PROBABILITY_ESTIMATE\s*=\s*([^\n]*)", ev):
        pcts += [int(x) for x in _FEMALE_READ_PCT_RE.findall(line) if 0 <= int(x) <= 100]
    if not pcts and PROBABILITY_LINE_ANY_FORMAT:                  # 0.7.1-pre12: any layout of the line
        for line in re.findall(r"(?im)^[ \t*_-]*PROBABILITY[ _]ESTIMATE[ \t]*[:=][^\n]*", _strip_md_emphasis(ev)):
            sh = _probability_line_shares(line)
            if sh and "female" in sh:
                pcts.append(int(round(sh["female"] * 100)))
    if not pcts:
        return evidence_text
    return ev + f"\n- Whole-track female share (normalised): female teen/adult {max(pcts)}%\n"


# 0.7.1-pre12 (YOUNG_MALE_FALSETTO_HEARD_ONLY): wording Musiclyse itself writes into every
# high-pitched analysis, which must not count as the listening model having heard falsetto.
_PROGRAM_FALSETTO_PHRASES = (
    "upper-tenor belting, head voice or falsetto",
    "falsetto / countertenor territory for an adult man",
    "(falsetto-led)",
    "Do not describe the voice as a soprano, treble or falsetto",
)
_NEGATED_FALSETTO_RE = re.compile(
    r"(?i)\b(?:no|not|without|nor|never|lacks?|lacking|absence of)\b[^.;\n]{0,60}?"
    r"\b(?:falsetto|head[- ]voice|countertenor)\b")
# 0.7.1-pre12 (YOUNG_MALE_MULTI_CREDIT_STOP): any credit naming more than one act.
_MULTI_CREDIT_RE = re.compile(r"(?i)\b(?:feat\.?|featuring|ft\.?|with|and|x|vs\.?)\s|[&+/,;]")
_COVER_MALE_WORD_RE = re.compile(r"(?i)\b(?:male|man|men|boy|boys|guy|guys)\b")


def _falsetto_heard_evidence(evidence_text):
    """evidence_text without Musiclyse's own falsetto/head-voice wording and without negated
    mentions ("no falsetto"), so only a listening pass that heard one stops the check."""
    ev = str(evidence_text or "")
    for phrase in _PROGRAM_FALSETTO_PHRASES:
        ev = ev.replace(phrase, " ")
    return _NEGATED_FALSETTO_RE.sub(" ", ev)


def _cover_people_cues(cover_obs):
    """(people, person_cues) from cover-art observations, as dict fields or the raw reply."""
    cov = cover_obs if isinstance(cover_obs, dict) else {}
    raw = str(cov.get("raw") or (cover_obs if isinstance(cover_obs, str) else ""))
    people = str(cov.get("people") or (re.search(r"(?mi)^\W*PEOPLE\W*[:=]\s*(.*)$", raw) or [None, ""])[1])
    cues = str(cov.get("person_cues") or (re.search(r"(?mi)^\W*PERSON_CUES\W*[:=]\s*(.*)$", raw) or [None, ""])[1])
    return people.strip().lower(), cues


def _young_male_lacks_male_evidence(identity_text, evidence_text, cover_obs):
    """0.7.1-pre12 YOUNG_MALE_NEEDS_MALE_EVIDENCE: True when a young-male result would rest on
    no male evidence: the cover shows a female-presenting person (and no male one), or the
    identity says its gender came from the name alone and no whole-track pass heard a male."""
    _people, cues = _cover_people_cues(cover_obs)
    if _COVER_FEMALE_RE.search(cues) and not _COVER_MALE_WORD_RE.search(cues):
        return True
    if re.search(r"(?mi)^\W*GENDER_BASIS\s*[:=]\s*name[_ ]only\b", str(identity_text or "")):
        heard_male = re.search(
            r"(?mi)(?:Initial isolated lead profile|Confirmation pass):\s*"
            r"(?:post_puberty_male|adult_male|adolescent_male_likely)\b", str(evidence_text or ""))
        if not heard_male:
            return True
    return False


def _young_male_voice_check__pre6(identity_text, evidence_text, cover_obs, metadata, background=None):
    if YOUNG_MALE_MULTI_CREDIT_STOP and _MULTI_CREDIT_RE.search(str((metadata or {}).get("artist") or "")):
        return identity_text          # 0.7.1-pre12: a duo or multi-act credit is not one young male
    if FEMALE_READ_ANY_WORDING:
        evidence_text = _female_read_evidence(evidence_text)
    if YOUNG_MALE_FALSETTO_HEARD_ONLY:
        evidence_text = _falsetto_heard_evidence(evidence_text)
    _memo = {}

    def bg():
        if "v" not in _memo:
            try:
                _memo["v"] = background() if callable(background) else (background or "")
            except Exception:
                _memo["v"] = ""
        return _memo["v"]
    new = _young_male_voice_check__pre6_rules(identity_text, evidence_text, cover_obs, metadata, bg)
    if "YOUNG_MALE_VOICE_CHECK=not applied: documented adult" in str(new or ""):
        return new
    old = _young_male_voice_check__07(identity_text, evidence_text, cover_obs, metadata, bg)
    out = _identity_keep_young_male(new, old, lambda: False)
    if ((YOUNG_MALE_CLIP_CORROBORATION or YOUNG_MALE_NO_COVER) and str(out or "") == str(identity_text or "")
            and "YOUNG_MALE_VOICE_CHECK=" not in str(out or "")):
        out = _young_male_clip_corroboration(identity_text, evidence_text, cover_obs, metadata, bg)
    if (YOUNG_MALE_NEEDS_MALE_EVIDENCE and str(out or "") != str(identity_text or "")
            and re.search(r"(?mi)^SINGER_IDENTITY=(?:adolescent_male_likely|child_male_likely)\s*$", str(out or ""))
            and _young_male_lacks_male_evidence(identity_text, evidence_text, cover_obs)):
        return identity_text          # 0.7.1-pre12: no male evidence behind a young-male result
    return out


def _young_male_clip_corroboration(identity_text, evidence_text, cover_obs, metadata, background):
    """The young-male voice check for cases its own rule misses, when the sampled clips back
    up a female-sounding whole-track pass on a credited solo male:
    - 0.7.1-pre7 YOUNG_MALE_CLIP_CORROBORATION: a pass heard as female at 75-89% (the check
      needs 90%), at least 4 confident clips, at least 80% of them female-sounding, and one
      person on the cover with teenage cues;
    - 0.7.1-pre9 YOUNG_MALE_NO_COVER: no cover art, or none showing a person (common for
      lesser-known artists), on stronger evidence: a pass heard as female at 90% or more, at
      least 6 confident clips, at least 90% of them female-sounding, and a median pitch of
      250 Hz or more. A cover showing an adult, or several people, never qualifies.
    Both also accept an adult_male identity (0.7.1-pre9), and both stop for falsetto or head
    voice, a feature/duo credit or a documented adult. Otherwise identity_text unchanged."""
    t = str(identity_text or "")
    _tags = ("post_puberty_male", "adult_male") if YOUNG_MALE_NO_COVER else ("post_puberty_male",)
    m = re.search(r"(?mi)^SINGER_IDENTITY=(\S+)\s*$", t)
    if not YOUNG_MALE_VOICE_CHECK or not m or m.group(1) not in _tags:
        return identity_text
    if not re.search(r"(?mi)^LEAD_SINGERS=one\s*$", t) or "REFERENCE_VOCAL_CORRECTION=yes" in t:
        return identity_text
    artist = str((metadata or {}).get("artist") or "")
    if re.search(r"(?i)\b(?:feat\.?|featuring|ft\.?|with|and|&|x|vs\.?)\s", artist + " "):
        return identity_text
    ev = str(evidence_text or "")
    if not re.search(r"(?mi)Initial isolated lead profile:\s*(?:female_teen_adult|adult_female|young_female)\b", ev):
        return identity_text
    pf = [int(x) for x in re.findall(r"(?i)female teen/adult\s+(\d{1,3})%", ev)]
    if not pf:
        return identity_text
    if re.search(r"(?i)\bfalsetto\b|\bhead[- ]voice\b|\bcountertenor\b", ev):
        return identity_text
    line = next((l for l in ev.splitlines() if "Section voice check" in l), "")
    clips = re.findall(r"\b(male|female) \(high;", line)
    if YOUNG_MALE_CLIPS_LEAD_ONLY:
        # 0.7.1-pre12: clips heard as a duet or group are not evidence of one young voice.
        if len(re.findall(r"\(high;\s*duet_co_leads\)", line)) >= 2:
            return identity_text
        clips = re.findall(r"\b(male|female) \(high;\s*(?:solo_lead|lead_plus_backing)\)", line)
    female = clips.count("female")
    cov = cover_obs if isinstance(cover_obs, dict) else {}
    raw = str(cov.get("raw") or "")
    people = str(cov.get("people") or (re.search(r"(?m)^PEOPLE=(.*)$", raw) or [None, ""])[1]).strip().lower()
    cues = str(cov.get("person_cues") or (re.search(r"(?m)^PERSON_CUES=(.*)$", raw) or [None, ""])[1])
    if people in ("one", "1"):
        if not (YOUNG_MALE_CLIP_CORROBORATION and _YOUNG_COVER_RE.search(cues)):
            return identity_text          # one person on the cover, not shown as a teenager
        if max(pf) >= 90 and m.group(1) == "post_puberty_male":
            return identity_text          # the check's own rule already decided this case
        if not (75 <= max(pf) and len(clips) >= 4 and female >= 0.8 * len(clips)):
            return identity_text
        how = "with a young person on the cover"
    elif people in ("", "none", "0", "zero", "no", "nobody", "unknown") and YOUNG_MALE_NO_COVER:
        median = re.search(r"(?i)Objective median f0:\s*([\d.]+)\s*Hz", ev)
        if not (max(pf) >= 90 and len(clips) >= 6 and female >= 0.9 * len(clips)
                and median and float(median.group(1)) >= 250):
            return identity_text
        # 0.7.1-pre10 (YOUNG_MALE_NO_COVER_MALE_READ): with no picture, "male" may rest on the
        # artist's name alone, so a whole-track pass must also have heard a male voice.
        if YOUNG_MALE_NO_COVER_MALE_READ and not re.search(
                r"(?mi)Confirmation pass:\s*(?:post_puberty_male|adult_male|adolescent_male_likely)\b", ev):
            return identity_text
        how = "with no picture of the singer to go on"
    else:
        return identity_text              # several people on the cover: not a solo portrait
    try:
        bg = background() if callable(background) else (background or "")
    except Exception:
        bg = ""
    _adult = _documented_adult__pre6(t, bg, _identity_recording_year(metadata), artist)
    if not _adult and not STATED_AGE_GUARDED:
        # 0.7's reader also takes "was 95%" as an age; 0.7.1-pre12 relies on the guarded one.
        _adult = _documented_adult__07(t, bg, (metadata or {}).get("year"), artist)
    if _adult:
        return t.rstrip("\n") + f"\nYOUNG_MALE_VOICE_CHECK=not applied: documented adult ({_adult})\n"
    out = re.sub(r"(?mi)^SINGER_IDENTITY=\S+\s*$", "SINGER_IDENTITY=adolescent_male_likely", t, count=1)
    out = _identity_override_confidence(out)
    note = (f" [Set to adolescent_male_likely by the young-male voice check: a credited solo male singer whose "
            f"whole-track listening pass heard a female-sounding voice ({max(pf)}%), as did {female} of the "
            f"{len(clips)} confidently sampled clips, {how} and nothing documenting an adult, is a young "
            "male whose voice had not broken or not yet fully deepened.]")
    if re.search(r"(?m)^REASONING=", out):
        out = re.sub(r"(?m)^(REASONING=.*)$", lambda mm: mm.group(1) + note, out, count=1)
    else:
        out += "\nREASONING=" + note.strip()
    return out.rstrip("\n") + "\nYOUNG_MALE_VOICE_CHECK=applied\n"


def _young_male_voice_check__07(identity_text, evidence_text, cover_obs, metadata, background=None):
    """0.7-pre5 YOUNG_MALE_VOICE_CHECK: identity_text with post_puberty_male
    changed to adolescent_male_likely when a credited solo male singer was
    heard as female-sounding by the whole-track passes, the cover shows one
    young person, and nothing documents an adult or a falsetto/head voice.
    Otherwise identity_text unchanged. background: a callable returning the
    reference background text (only looked up when everything else holds)."""
    t = str(identity_text or "")
    if not YOUNG_MALE_VOICE_CHECK or not re.search(r"(?mi)^SINGER_IDENTITY=post_puberty_male\s*$", t):
        return identity_text
    if not re.search(r"(?mi)^LEAD_SINGERS=one\s*$", t) or "REFERENCE_VOCAL_CORRECTION=yes" in t:
        return identity_text
    artist = str((metadata or {}).get("artist") or "")
    if re.search(r"(?i)\b(?:feat\.?|featuring|ft\.?|with|and|&|x|vs\.?)\s", artist + " "):
        return identity_text
    ev = str(evidence_text or "")
    if not re.search(r"(?mi)Initial isolated lead profile:\s*(?:female_teen_adult|adult_female|young_female)\b", ev):
        return identity_text
    pf = [int(x) for x in re.findall(r"(?i)female teen/adult\s+(\d{1,3})%", ev)]
    if not pf or max(pf) < 90:
        return identity_text
    if re.search(r"(?i)\bfalsetto\b|\bhead[- ]voice\b|\bcountertenor\b", ev):
        return identity_text
    cov = cover_obs if isinstance(cover_obs, dict) else {}
    if str(cov.get("people") or "").strip().lower() not in ("one", "1"):
        return identity_text
    if not _YOUNG_COVER_RE.search(str(cov.get("person_cues") or "")):
        return identity_text
    bg = ""
    try:
        bg = background() if callable(background) else (background or "")
    except Exception:
        bg = ""
    _adult = _documented_adult(t, bg, (metadata or {}).get("year"), artist)
    if _adult:
        # 0.7-pre6: say why, so a save shows the check ran and what stopped it.
        return t.rstrip("\n") + f"\nYOUNG_MALE_VOICE_CHECK=not applied: documented adult ({_adult})\n"
    out = re.sub(r"(?mi)^SINGER_IDENTITY=post_puberty_male\s*$", "SINGER_IDENTITY=adolescent_male_likely", t, count=1)
    note = (" [Set to adolescent_male_likely by the young-male voice check: a credited solo male singer whose "
            "whole-track listening passes heard a female-sounding voice, with a young person on the cover and "
            "nothing documenting an adult, is a young male whose voice had not fully deepened.]")
    if re.search(r"(?m)^REASONING=", out):
        out = re.sub(r"(?m)^(REASONING=.*)$", lambda m: m.group(1) + note, out, count=1)
    else:
        out += "\nREASONING=" + note.strip()
    return out.rstrip("\n") + "\nYOUNG_MALE_VOICE_CHECK=applied\n"


# 0.7.1-pre13 SINGER_DETAIL_LOOKUP: after the identity is settled, a separate short request asks
# for documented facts only -- who sings lead on this recording, their sex and birth year, the
# recording's original year, and (0.7.1-pre14) a group's members with their sexes. It cannot change the identity
# decision; the answers are kept as SINGER_DETAIL_* lines for the export and the chat.
SINGER_DETAIL_PROMPT = """You are filling in documented facts about who sings on one recording. Do not judge the sound.

RECORDING: artist "{artist}", title "{title}", album "{album}", year {year}

REFERENCE BACKGROUND (may be empty):
{background}

Answer with exactly these lines and nothing else:
LEAD_SINGER_NAMES=<the person or people documented as singing lead on THIS recording, separated by ";", or unknown>
LEAD_SINGER_SEXES=<male or female for each name, in the same order, separated by ";", or unknown>
LEAD_SINGER_BIRTH_YEARS=<the documented birth year of each name, in the same order, separated by ";", or unknown>
ORIGINAL_YEAR=<the year THIS recording was first released -- not a later compilation, reissue or remaster -- or unknown>
ACT_TYPE=<solo, group or unknown: whether the credited act is one person, or a band, group or duo>
GROUP_MEMBERS=<for a group or duo: its documented members at the time of this recording, each written as Name (male) or Name (female), separated by ";"; for a solo act or when not documented: unknown>
CONFIDENCE=<high, medium or low>

Rules: give only facts you are sure are documented, and write unknown rather than guess. Never name a
featured or guest singer as the lead. For a solo act the lead singer is the credited artist."""

_SINGER_DETAIL_NAME_RE = re.compile(r"^[^\W\d_][\w'’.\- ]{1,58}$")
_SINGER_DETAIL_DIAG = {}     # 0.7.1-pre19 SINGER_DETAIL_DIAGNOSTICS: the last lookup's record, until it is saved


def _singer_detail_field(text, key):
    m = re.search(r"(?mi)^[ \t*_-]*" + key + r"[ \t*_]*[=:][ \t]*(.+)$", str(text or ""))
    v = m.group(1).strip().strip("*_`\"' ") if m else ""
    return "" if re.fullmatch(r"(?i)unknown|none|n/?a|not documented|-*", v or "") else v


def _parse_singer_detail(text, background, artist, year=None):
    """{names, sexes, born, recorded, act, mixed_members, source} from the lookup reply,
    keeping only what passes the checks; {} when nothing usable.
    0.7.1-pre14: a group is only recorded as having men and women when the reply names at
    least one member of each sex and every such name appears in the reference background;
    the original recording year is kept only when it is not later than the file's year."""
    conf = _singer_detail_field(text, "CONFIDENCE").lower()
    if conf not in ("high", "medium"):
        return {}
    bg = str(background or "")
    bgf = bg.casefold()
    trust_model = conf == "high"

    def documented(s):
        return (bool(bg) and s.casefold() in bgf) or (not bg and trust_model)

    try:
        tag_year = int(str(year or "")[:4])
    except Exception:
        tag_year = None
    rec = _singer_detail_field(text, "ORIGINAL_YEAR")
    rec = rec if re.fullmatch(r"(?:19|20)\d\d", rec or "") else ""
    if rec and ((tag_year and int(rec) > tag_year) or not ((bg and rec in bg) or trust_model)):
        rec = ""
    ref_year = int(rec) if rec else tag_year
    raw_names = [x.strip().strip(".") for x in re.split(r"\s*;\s*", _singer_detail_field(text, "LEAD_SINGER_NAMES")) if x.strip()]
    raw_sexes = [x.strip().lower() for x in re.split(r"\s*;\s*", _singer_detail_field(text, "LEAD_SINGER_SEXES"))]
    raw_born = [x.strip() for x in re.split(r"\s*;\s*", _singer_detail_field(text, "LEAD_SINGER_BIRTH_YEARS"))]
    names, sexes, born = [], [], []
    for i, n in enumerate(raw_names[:4]):
        if not _SINGER_DETAIL_NAME_RE.match(n) or not documented(n):
            continue
        s = raw_sexes[i] if i < len(raw_sexes) and raw_sexes[i] in ("male", "female") else ""
        b = raw_born[i] if i < len(raw_born) else ""
        b = b if re.fullmatch(r"(?:19|20)\d\d", b or "") and (not bg or b in bg or trust_model) else ""
        if b and ref_year and not (3 <= ref_year - int(b) <= 95):
            b = ""
        names.append(n); sexes.append(s); born.append(b)
    act = _singer_detail_field(text, "ACT_TYPE").lower()
    act = act if act in ("solo", "group") else ""
    members = []
    if act == "group" and bg:
        for part in re.split(r"\s*;\s*", _singer_detail_field(text, "GROUP_MEMBERS")):
            m = re.fullmatch(r"(.+?)\s*\(\s*(male|female)\s*\)", part.strip(), re.I)
            if m and _SINGER_DETAIL_NAME_RE.match(m.group(1).strip()) and m.group(1).strip().casefold() in bgf:
                members.append((m.group(1).strip(), m.group(2).lower()))
    mixed = members if {x[1] for x in members} == {"male", "female"} else []
    if not (names or act):
        return {}
    return {"names": names, "sexes": sexes, "born": born, "recorded": rec, "act": act,
            "mixed_members": mixed, "source": "background" if bg else "model knowledge"}


def _singer_detail_lines(detail):
    if not detail:
        return ""
    out = []
    if detail.get("names"):
        out.append("SINGER_DETAIL_NAMES=" + "; ".join(detail["names"]))
        out.append("SINGER_DETAIL_SEXES=" + "; ".join(x or "unknown" for x in detail["sexes"]))
        out.append("SINGER_DETAIL_BORN=" + "; ".join(x or "unknown" for x in detail["born"]))
    if detail.get("recorded"):
        out.append("SINGER_DETAIL_RECORDED=" + detail["recorded"])
    if detail.get("act"):
        out.append("SINGER_DETAIL_ACT=" + detail["act"])
    if detail.get("mixed_members"):
        out.append("SINGER_DETAIL_MIXED_MEMBERS=" + "; ".join(f"{n} ({x})" for n, x in detail["mixed_members"]))
    out.append("SINGER_DETAIL_SOURCE=" + detail.get("source", ""))
    return "\n".join(out) + "\n"


def _singer_detail_background_text(background, artist, title):
    """0.7.1-pre20 SINGER_DETAIL_BACKGROUND_SONG_FIRST: the singer lookup's copy of the background,
    as plain articles -- the song's own article first (when its opening names the credited act),
    then the act's own article, then the rest in their original order, each article once and
    without the chat's notes around them. "" when no article can be read out of it."""
    arts, seen = [], set()
    for block in re.split(r"(?m)^=== WIKIPEDIA BACKGROUND CONTEXT\b.*$", str(background or "")):
        block = block.split("\nRULES FOR THE ABOVE:")[0]
        heads = list(re.finditer(r'(?m)^--- "(.*)" ---$', block))
        for k, m in enumerate(heads):
            body = block[m.end():heads[k + 1].start() if k + 1 < len(heads) else len(block)].strip()
            name = m.group(1).strip()
            if body and name and name.casefold() not in seen:
                seen.add(name.casefold())
                arts.append((name, body))
    if not arts:
        return ""

    def norm(s):
        return " ".join(re.sub(r"[^\w]+", " ", str(s or "").casefold()).split())

    def bare(s):                                  # without a trailing "(...)" qualifier
        return norm(re.sub(r"\s*\([^()]*\)\s*$", "", str(s or "")))

    def no_the(s):
        return s[4:] if s.startswith("the ") else s

    act = norm(_wiki_clean_artist_for_search(artist) or artist)
    t = str(title or "").strip()
    song_names = {x for x in (norm(t), bare(t), bare(t.split(" - ")[0])) if x}

    def rank(item):
        name, body = item
        qual = name.rstrip()[name.rfind("(") + 1:-1].casefold() if name.rstrip().endswith(")") and "(" in name else ""
        if (act and song_names & {norm(name), bare(name)}
                and f" {act} " in f" {norm(name + ' ' + body[:600])} "):
            return (0, 0 if ("song" in qual or "single" in qual) else 1 if not qual else 2)
        if act and no_the(bare(name)) == no_the(act):
            return (1, 0)
        return (2, 0)

    return "\n\n".join(f'--- "{n}" ---\n{b}' for n, b in sorted(arts, key=rank))


def _singer_detail_add(identity_text, metadata, background=None):
    """identity_text with SINGER_DETAIL_* lines added from the lookup; unchanged when the
    lookup is off, fails or finds nothing documented.
    0.7.1-pre19 SINGER_DETAIL_DIAGNOSTICS: also keeps a record of the requests for the save."""
    t = str(identity_text or "")
    if SINGER_DETAIL_DIAGNOSTICS:                 # 0.7.1-pre19: a lookup that doesn't run leaves no record
        _SINGER_DETAIL_DIAG.clear()
    if not SINGER_DETAIL_LOOKUP or not t.strip() or "SINGER_DETAIL_" in t:
        return identity_text
    md = metadata if isinstance(metadata, dict) else {}
    artist = str(md.get("artist") or "").strip()
    if not artist:
        return identity_text
    try:
        bg = background() if callable(background) else (background or "")
    except Exception:
        bg = ""
    year = _identity_recording_year(md)
    _bg_text = str(bg or "").strip()[:6000]
    if SINGER_DETAIL_BACKGROUND_SONG_FIRST and _bg_text:      # 0.7.1-pre20: song article first, no chat notes
        _plain = _singer_detail_background_text(bg, artist, str(md.get("title") or "").strip())
        if _plain:
            _bg_text = _plain if len(_plain) <= 6000 else _plain[:6000].rsplit(" ", 1)[0] + " ..."
    prompt = SINGER_DETAIL_PROMPT.format(
        artist=artist, title=str(md.get("title") or "").strip(), album=str(md.get("album") or "").strip(),
        year=year or "unknown", background=(_bg_text or "(none)"))
    _ka = None
    if globals().get("BATCH_UNLOAD_OLLAMA", True) and _is_batch_context():
        _ka = globals().get("BATCH_OLLAMA_KEEP_ALIVE", 0)
        if SINGER_IDENTITY_REUSE_BATCH_MODEL and _ka in (0, "0", "0s"):
            _ka = "5m"
    _ctx = SINGER_IDENTITY_NUM_CTX if _is_batch_context() else None
    if SINGER_IDENTITY_FIT_PROMPT and _ctx is not None:
        _ctx = _identity_ctx_for_prompt(prompt, _ctx)
    _diag = None
    if SINGER_DETAIL_DIAGNOSTICS:                 # 0.7.1-pre19
        _bg_sent = _bg_text                       # 0.7.1-pre20: the text the lookup was given
        _diag = {"artist": artist, "title": str(md.get("title") or "").strip(),
                 "background_chars": len(_bg_sent), "background": _bg_sent, "requests": []}
        _t0 = time.time()
    _first_usage = None
    try:
        reply, _first_usage = ollama_chat([{"role": "user", "content": prompt}], num_ctx=_ctx, keep_alive=_ka,
                                          options=_scan_llm_options())
    except Exception as _e:
        reply = None
        if _diag is not None:
            _diag["requests"].append({"request": "first", "error": f"{type(_e).__name__}: {_e}"[:500],
                                      "seconds": round(time.time() - _t0, 1)})
    if _diag is not None and reply is not None:
        _u = _first_usage if isinstance(_first_usage, dict) else {}
        _c = _u.get("ctx")
        _diag["requests"].append({"request": "first", "reply": reply, "prompt_tokens": _u.get("prompt"),
                                  "reply_tokens": _u.get("completion"),
                                  "reply_allowance": _ollama_output_budget(_c, 16384) if _c else None,
                                  "context": _c, "seconds": round(time.time() - _t0, 1)})
    _retry_note = None
    if SINGER_DETAIL_RETRY_WITHOUT_THINKING and reply is not None and not str(reply).strip():   # 0.7.1-pre17
        _rec = {"request": "thinking off"} if _diag is not None else None
        _t1 = time.time() if _rec is not None else None
        _again, _retry_note = _singer_detail_reply_without_thinking(prompt, _ctx, _ka, record=_rec)
        if _rec is not None:
            _rec.update(note=_retry_note or "", seconds=round(time.time() - _t1, 1))
            _diag["requests"].append(_rec)
        if _again:
            reply = _again
    detail = _parse_singer_detail(reply, bg, artist, year) if reply is not None else {}
    lines = _singer_detail_lines(detail)
    if SINGER_DETAIL_STATUS_LINE:                 # 0.7.1-pre15: why nothing (or only part) was added
        lines += "SINGER_DETAIL_STATUS=" + _singer_detail_status(reply, detail, bg)
        if _retry_note is not None:
            lines += "; asked again with thinking off" + (": " + _retry_note if _retry_note else "")
        if _diag is not None and _diag["requests"]:                 # 0.7.1-pre19
            _f = _diag["requests"][0]
            if _f.get("reply") == "" and _f.get("reply_tokens") is not None and _f.get("reply_allowance"):
                lines += (f"; the first answer was empty after {_f['reply_tokens']} of its "
                          f"{_f['reply_allowance']} reply tokens")
        lines += "\n"
    if _diag is not None:                         # 0.7.1-pre19: kept until save_song_data writes it
        _diag["status"] = next((ln[len("SINGER_DETAIL_STATUS="):] for ln in lines.splitlines()
                                if ln.startswith("SINGER_DETAIL_STATUS=")), "")
        _diag["_at"] = time.time()
        _SINGER_DETAIL_DIAG.clear()
        _SINGER_DETAIL_DIAG.update(_diag)
    return (t.rstrip("\n") + "\n" + lines) if lines else identity_text


def _singer_detail_reply_without_thinking(prompt, num_ctx, keep_alive, record=None):
    """0.7.1-pre17 SINGER_DETAIL_RETRY_WITHOUT_THINKING: the lookup asked once more with the
    model's thinking turned off, after its first reply came back empty (a model that thinks
    before answering can use up its whole reply allowance thinking). The prompt, context size,
    keep-alive and sampling are the first request's; the reply allowance is shorter, since the
    answer is a few lines. Returns (reply, note): reply is "" when nothing usable came back,
    and note says why ("" when a reply came back).
    0.7.1-pre19 SINGER_DETAIL_DIAGNOSTICS: record (a dict, or None) is filled in with what came back."""
    try:
        ctx = int(num_ctx if num_ctx is not None else OLLAMA_NUM_CTX)
        out_tokens = _ollama_output_budget(ctx, 16384)
        messages = [{"role": "user", "content": prompt}]
        _compact_writer_history_in_place(messages)
        payload = {
            "model": OLLAMA_MODEL,
            "messages": _prepare_ollama_messages(messages, ctx, keep_images=True, max_output_tokens=out_tokens),
            "stream": False,
            "think": False,
            "options": {"num_ctx": ctx, "num_predict": min(2048, out_tokens)},
        }
        if keep_alive is not None:
            payload["keep_alive"] = keep_alive
        options = _scan_llm_options()
        if options:
            payload["options"].update(options)
        if _TEST_SEED is not None:
            payload["options"]["seed"] = _TEST_SEED
        resp = requests.post(OLLAMA_URL, json=payload, timeout=900)
        if record is not None:                    # 0.7.1-pre19
            record["http_status"] = resp.status_code
        if resp.status_code >= 400:
            if record is not None:
                record["error"] = str(resp.text or "")[:500]
            # A model or Ollama version that refuses the option is not asked a third time:
            # without it the request would be the first one again.
            return "", ("thinking can't be turned off for this model" if "think" in str(resp.text or "").lower()
                        else "request failed")
        data = resp.json()
        reply = str((data.get("message") or {}).get("content") or "").strip()
        prompt_n = int(data.get("prompt_eval_count") or 0)
        comp_n = int(data.get("eval_count") or 0)
        if record is not None:                    # 0.7.1-pre19
            record.update(reply=reply, done_reason=str(data.get("done_reason") or ""),
                          thinking_chars=len(str((data.get("message") or {}).get("thinking") or "")),
                          prompt_tokens=prompt_n, reply_tokens=comp_n,
                          reply_allowance=min(2048, out_tokens), context=ctx)
        SESSION_TOKEN_USAGE["prompt"] = int(SESSION_TOKEN_USAGE.get("prompt") or 0) + prompt_n
        SESSION_TOKEN_USAGE["completion"] = int(SESSION_TOKEN_USAGE.get("completion") or 0) + comp_n
        SESSION_TOKEN_USAGE["total"] = (int(SESSION_TOKEN_USAGE.get("prompt") or 0)
                                        + int(SESSION_TOKEN_USAGE.get("completion") or 0))
        SESSION_TOKEN_USAGE["last_prompt"] = prompt_n
        SESSION_TOKEN_USAGE["last_completion"] = comp_n
        SESSION_TOKEN_USAGE["last_ctx"] = ctx
        if data.get("done_reason") == "length":
            return "", "cut off"
        return reply, ("" if reply else "also empty")
    except Exception as _e:
        if record is not None:                    # 0.7.1-pre19
            record["error"] = f"{type(_e).__name__}: {_e}"[:500]
        return "", "request failed"


def _singer_detail_status(reply, detail, background):
    """0.7.1-pre15: one short reason for what the lookup added, for checking scans."""
    if reply is None:
        return "request failed"
    if not str(reply).strip():
        return "empty reply"
    conf = _singer_detail_field(reply, "CONFIDENCE").lower()
    if conf == "low":
        return "low confidence"
    if conf not in ("high", "medium"):
        return "no usable reply"
    if not _singer_detail_field(reply, "LEAD_SINGER_NAMES"):
        return "no documented lead singer"
    if not (detail or {}).get("names"):
        return ("lead singer named but not found in the background" if background
                else "lead singer named without high confidence")
    return "ok"


# 0.7.1-pre13 FEATURED_SINGER_NOT_LEAD: a female identity that credits a documented featured
# singer as the lead, on a solo credit whose cover shows one young male and no female, is the
# credited boy's unbroken voice; the featured singer is kept as a guest.
_FSNL_FEMALE_TAG_RE = re.compile(r"(?mi)^SINGER_IDENTITY\s*=\s*(?:female_teen_adult|adult_female|young_female)\s*$")
_FSNL_MULTI_CREDIT_RE = re.compile(r"(?i)\b(?:feat\.?|featuring|ft\.?|with|and|x|vs\.?)\s|[&+/,;]")
_FSNL_YOUNG_RE = re.compile(r"(?i)\b(?:teen|teens|teenage|teenager|teenaged|boy|child|kid|youth|young|juvenile|"
                            r"pre-?teen|adolescent)\b")
_FSNL_MALE_RE = re.compile(r"(?i)\b(?:male|man|boy|guy|he|his|him)\b")
_FSNL_FEMALE_RE = re.compile(r"(?i)\b(?:female|woman|women|girl|girls|lady|she|her|females)\b")
_FSNL_NAME = r"[A-Z][\w'’.-]+(?:\s+(?:[A-Z][\w'’.-]+|de|van|von|la|le|del|da))*\s+[A-Z][\w'’.-]+"
_FSNL_GUEST_RES = (
    re.compile(r"\b(?:featur\w*|guest(?:\s+vocal\w*|\s+singer)?|duet\w*(?:\s+partner)?|joined\s+by|"
               r"vocals?\s+(?:from|by))\s+(?:(?:of|by|from|with|singer|vocalist|artist)\s+)*(" + _FSNL_NAME + r")"),
    re.compile(r"(" + _FSNL_NAME + r")(?:'s|’s)?\s+(?:is\s+|was\s+|as\s+(?:a|the)\s+)?(?:featured|guest)\b"),
)


def _fsnl_guest_name(reason, artist):
    """The featured or guest singer the reasoning names, when that person is not the credited act."""
    a = re.sub(r"\W+", " ", str(artist or "")).strip().casefold()
    a_words = set(a.split())
    for rx in _FSNL_GUEST_RES:
        for m in rx.finditer(str(reason or "")):
            name = re.sub(r"(?:'s|’s)$", "", m.group(1).strip(" .,'’")).strip(" .,'’")
            n = re.sub(r"\W+", " ", name).strip().casefold()
            if not n or n == a or set(n.split()) & a_words:
                continue
            if re.search(r"(?i)\b(?:acoustic|listening|vocal|evidence|lead|track|song|single|album|cover|"
                         r"the|this|male|female)\b", name):
                continue
            return name
    return ""


def _fsnl_one_female_voice_heard(evidence_text):
    """0.7.1-pre15: the listening passes heard one voice, female-sounding, and the sampled clips
    (when they ran) confirmed no different lead."""
    ev = str(evidence_text or "")
    if not re.search(r"FINAL LEAD PROFILE:\s*(?:female_teen_adult|adult_female|young_female)\b", ev):
        return False
    if not re.search(r"NUM_DISTINCT_VOICES\s*[=:]\s*1\b", ev) or re.search(r"NUM_DISTINCT_VOICES\s*[=:]\s*[2-9]", ev):
        return False
    if re.search(r"(?i)VOICE_ARRANGEMENT\s*[=:]\s*(?:duet|call)", ev):
        return False
    for line in re.findall(r"Section voice check[^\n]*", ev):
        if "no different lead confirmed" not in line:
            return False
    return True


# 0.8-pre31 FEATURED_SINGER_SAFEGUARDS: FEATURED_SINGER_NOT_LEAD is for a female-sounding lead that the
# reasoning gives to a documented featured singer. It is held back when the guest is given another part
# right after their name, or when anything says the credited artist is a woman.
_FSNL_ROLE_AFTER = re.compile(
    r"(?i)^[\s,(]{0,3}(?:who\s+|which\s+|that\s+)?(?:(?:on|playing|plays|as|for)\s+(?:the\s+|a\s+|an\s+|his\s+|her\s+)?"
    r"(?:guitars?|bass|drums|percussion|piano|keys|keyboards?|synth\w*|sax\w*|trumpets?|horns?|violins?|strings|"
    r"cello|harmonica|organ|production|producer|programming|mixing|backing\s+vocals?|backup\s+vocals?|"
    r"harmon(?:y|ies)\b(?:\s+vocals?)?|rap\w*|(?:a\s+)?(?:rap\s+)?verse)"
    r"|(?:sings|provides|adds|contributes|sang|provided|added|contributed)\s+(?:the\s+|some\s+)?"
    r"(?:backing|backup|harmon\w*|a\s+rap|rap)"
    r"|produc\w*|remix\w*|wrote|written|co-?wr\w*|songwriter|rapp\w*|raps)\b")
_FSNL_FEM_NOUN = (r"(?:woman|women|female|girl|lady|actress|songstress|chanteuse|diva|sister|daughter|mother|"
                  r"wife|queen)")


def _fsnl_words(name):
    """The words of a name, casefolded, three letters or more."""
    return [w for w in re.findall(r"[^\W_]+", str(name or "").casefold()) if len(w) >= 3]


def _fsnl_guest_other_role(reason, guest):
    """True when every mention of the guest in the reasoning gives them a part other than singing
    the lead right after their name (on guitar, producer, backing vocals, a rap verse ...)."""
    g = str(guest or "")
    if not g:
        return False
    rest = [str(reason)[m.end():] for m in re.finditer(re.escape(g), str(reason or ""))]
    return bool(rest) and all(_FSNL_ROLE_AFTER.match(r) for r in rest)


def _fsnl_artist_female(identity_text, reason, artist, background="", guest=""):
    """Why the credited artist counts as a woman ("lookup", "reasoning", "background"), or "".
    The singer lookup's lines when one of its names is the artist's; a plain statement about the
    artist in the reasoning ("X is a female singer", "X, a young woman", "X ... and she sings
    lead", with no guest named in between); the artist's own Wikipedia article (its title is the
    artist's name) using she/her and hardly ever he/his. A "she" about the guest doesn't count."""
    t = str(identity_text or "")
    aw = set(_fsnl_words(artist))
    gw = set(_fsnl_words(guest))
    n = re.search(r"(?mi)^SINGER_DETAIL_NAMES\s*=\s*(.+)$", t)
    s = re.search(r"(?mi)^SINGER_DETAIL_SEXES\s*=\s*(.+)$", t)
    if n and s and aw:
        sexes = set()
        for name, sex in zip([x.strip() for x in n.group(1).split(";")],
                             [x.strip().lower() for x in s.group(1).split(";")]):
            if aw & set(_fsnl_words(name)):
                sexes.add(sex)
        if "female" in sexes and "male" not in sexes:
            return "lookup"
    a = re.escape(str(artist or "").strip())
    if a:
        r = str(reason or "")
        if re.search(r"(?i)\b" + a + r"\b(?:'s|’s)?\s*(?:is|was|,)\s+(?:a|an|the)\s+"
                     r"(?:(?!male\b|man\b|boy\b|who\b|whose\b|with\b|that\b|and\b)[\w'’-]+\s+){0,4}"
                     + _FSNL_FEM_NOUN + r"\b", r):
            return "reasoning"
        for m in re.finditer(r"(?i)(?:^|[.;!?]\s+)(" + a + r"\b[^.;:!?\n]{0,80}?\b(?:and|,)\s+(?:she|herself)\b)",
                             r, re.M):
            if not (gw & set(_fsnl_words(m.group(1)))):
                return "reasoning"
        if re.search(r"(?i)\b" + a + r"\s+(?:herself|\(she\b|\(her\b)", r):
            return "reasoning"
    ac = str(artist or "").casefold().strip()
    for title, body in re.findall(r'(?s)--- "([^"\n]+)" ---\n(.*?)(?=\n--- "|\Z)', str(background or "")):
        tc = title.casefold().strip()
        if ac and (tc == ac or tc.startswith(ac + " (")):
            lead = body[:1500]
            she = len(re.findall(r"(?i)\b(?:she|her|hers|herself)\b", lead))
            he = len(re.findall(r"(?i)\b(?:he|him|his|himself)\b", lead))
            if she >= 2 and she > 2 * he:
                return "background"
    return ""


def _featured_singer_not_lead(identity_text, cover_obs, metadata, background=None, add_maturity=False,
                              evidence_text=""):
    t = str(identity_text or "")
    if "REFERENCE_VOCAL_CORRECTION=yes" in t:
        return identity_text
    female_one = bool(_FSNL_FEMALE_TAG_RE.search(t)) and not re.search(r"(?mi)^LEAD_SINGERS\s*=\s*(?!one\s*$)\S", t)
    # 0.7.1-pre15 FEATURED_SINGER_ONE_VOICE: a shared-lead answer resting on a documented featured
    # singer, when the listening passes heard one female-sounding voice and confirmed no second lead.
    shared_one_voice = (not female_one and FEATURED_SINGER_ONE_VOICE
                        and bool(re.search(r"(?mi)^SINGER_IDENTITY\s*=\s*mixed_leads\s*$", t))
                        and not re.search(r"(?mi)^(?:UNBROKEN_MALE_LEAD|FEMALE_COLEAD)\w*\s*=", t)
                        and "confirmed sampled lead voices" not in t
                        and _fsnl_one_female_voice_heard(evidence_text))
    if not (female_one or shared_one_voice):
        return identity_text
    md = metadata if isinstance(metadata, dict) else {}
    artist = str(md.get("artist") or "").strip()
    if not artist or _FSNL_MULTI_CREDIT_RE.search(artist + " "):
        return identity_text
    cov = cover_obs if isinstance(cover_obs, dict) else {}
    people = str(cov.get("people") or "").strip().strip("*_ ").lower()
    cues = str(cov.get("person_cues") or "")
    if people not in ("one", "1") or not _FSNL_YOUNG_RE.search(cues) or not _FSNL_MALE_RE.search(cues) \
            or _FSNL_FEMALE_RE.search(cues):
        return identity_text
    reason = _identity_reason_text(t)
    guest = _fsnl_guest_name(reason, artist)
    if not guest:
        return identity_text
    if FEATURED_SINGER_SAFEGUARDS:                   # 0.8-pre31: the guest sings the lead; the artist isn't a woman
        if _fsnl_guest_other_role(reason, guest):
            return identity_text
        try:
            background = background() if callable(background) else (background or "")
        except Exception:
            background = ""
        if _fsnl_artist_female(t, reason, artist, background, guest):
            return identity_text
    bg = ""
    try:
        bg = background() if callable(background) else (background or "")
    except Exception:
        bg = ""
    year = _identity_recording_year(md)
    age = _identity_person_age(reason, bg, year, artist)
    if age is not None and age >= 18:
        return identity_text          # a documented adult: not this rule's case
    tag = "child_male_likely" if (age is not None and age < 12) else "adolescent_male_likely"
    conf = "low" if re.search(r"(?mi)^CONFIDENCE\s*=\s*low\s*$", t) else "medium"
    age_txt = f", about {age} at the time" if age is not None else ""
    out = (f"SINGER_IDENTITY={tag}\n"
           f"REASONING=The credited solo artist is {artist}{age_txt}, and the cover shows one young boy, so the "
           f"female-sounding lead is most likely his unbroken voice"
           f"{' (the listening passes heard one voice and confirmed no second lead)' if shared_one_voice else ''}. "
           f"{guest} is documented as a featured singer "
           f"on this track; which lines {guest} sings isn't established.\n"
           f"CONFIDENCE={conf}\nLEAD_SINGERS=one\nFEATURED_SINGER={guest}\nFEATURED_SINGER_NOT_LEAD=applied\n")
    if add_maturity:
        out += "YOUNG_VOICE_MATURITY=unbroken\n"
    kept = re.findall(r"(?m)^SINGER_DETAIL_\w+=.*$", t)      # 0.7.1-pre15: a scan's lookup lines stay
    if kept:
        out += "\n".join(kept) + "\n"
    return out


def _young_male_voice_check(*args, **kwargs):
    """SINGER_IDENTITY_PRE6 picks the 0.7.1-pre6 version; False gives 0.7's."""
    return (_young_male_voice_check__pre6 if SINGER_IDENTITY_PRE6 else _young_male_voice_check__07)(*args, **kwargs)


_COVER_MALE_RE = re.compile(r"(?i)\b(?:male|man|men|boy|boys|guy|he|his|him)\b")
_COVER_FEMALE_RE = re.compile(r"(?i)\b(?:female|woman|women|girl|girls|lady|she|her)\b")
_GROUPISH_RE = re.compile(r"(?i)\b(?:band|group|duo|trio|quartet|quintet|members?|duet|siblings|"
                          r"featuring|feat\.?|co-?leads?|second (?:singer|vocalist|lead))\b")


def _young_solo_male_clips__pre6_rules(answer_text, identity_text, cover_obs, metadata, background=None):
    """0.7-pre9 YOUNG_SOLO_MALE_CLIPS: identity_text is the identity after the
    unconfirmed-leads rule; answer_text the identity model's own answer before it.
    When that rule set "uncertain" over a male or uncertain answer, on a one-act
    credit whose cover shows one teenage boy and nothing group-like, return one
    young male lead (or one adult male lead for a documented adult). Otherwise
    identity_text unchanged."""
    t = str(identity_text or "")
    if (not YOUNG_SOLO_MALE_CLIPS
            or "corroboration did not establish distinct singers" not in t
            or not re.search(r"(?mi)^SINGER_IDENTITY=uncertain\s*$", t)):
        return identity_text
    a = str(answer_text or "")
    if not re.search(r"(?mi)^SINGER_IDENTITY=(?:post_puberty_male|adolescent_male_likely|uncertain)\s*$", a):
        return identity_text
    if "REFERENCE_VOCAL_CORRECTION=yes" in a:
        return identity_text
    artist = str((metadata or {}).get("artist") or "").strip()
    if not artist or re.search(r"(?i)\b(?:feat\.?|featuring|ft\.?|with|and|x|vs\.?)\s|[&+/,]", artist + " "):
        return identity_text
    cov = cover_obs if isinstance(cover_obs, dict) else {}
    if str(cov.get("people") or "").strip().lower() not in ("one", "1"):
        return identity_text
    cues = str(cov.get("person_cues") or "")
    if not (_YOUNG_COVER_RE.search(cues) and _COVER_MALE_RE.search(cues)) or _COVER_FEMALE_RE.search(cues):
        return identity_text
    _r = re.search(r"(?m)^REASONING=(.*)$", a)
    if _GROUPISH_RE.search(_r.group(1) if _r else ""):
        return identity_text
    bg = ""
    try:
        bg = background() if callable(background) else (background or "")
    except Exception:
        bg = ""
    if _GROUPISH_RE.search(str(bg or "")[:600]):
        return identity_text
    _adult = _documented_adult(a, bg, _identity_recording_year(metadata), artist)
    age = _identity_person_age(_identity_reason_text(a), bg, _identity_recording_year(metadata), artist)
    if age is None or age < 12:
        return identity_text
    confidence = "low" if re.search(r"(?mi)^CONFIDENCE=low\s*$", a + "\n" + t) else "medium"
    why = (f"The credited solo artist is {artist}; the cover shows one teenage boy, and nothing points to a "
           "group, duo or featured singer. The sampled clips heard both male- and female-sounding voices but "
           "no second singer was confirmed")
    if _adult:
        return ("SINGER_IDENTITY=post_puberty_male\nREASONING=" + why + f"; he is documented as an adult ({_adult}), "
                f"so this is one adult male lead.\nCONFIDENCE={confidence}\nLEAD_SINGERS=one\n"
                "YOUNG_SOLO_MALE_CLIPS=applied: documented adult\n")
    return ("SINGER_IDENTITY=adolescent_male_likely\nREASONING=" + why + f"; the singer has a documented teenage age at this recording. Voice maturity remains a separate acoustic finding.\nCONFIDENCE={confidence}\nLEAD_SINGERS=one\n"
            "YOUNG_SOLO_MALE_CLIPS=applied\n")


def _young_solo_male_clips__pre6(answer_text, identity_text, cover_obs, metadata, background=None):
    _memo = {}

    def bg():
        if "v" not in _memo:
            try:
                _memo["v"] = background() if callable(background) else (background or "")
            except Exception:
                _memo["v"] = ""
        return _memo["v"]
    new = _young_solo_male_clips__pre6_rules(answer_text, identity_text, cover_obs, metadata, bg)
    if "YOUNG_SOLO_MALE_CLIPS=applied: documented adult" in str(new or ""):
        return new
    old = _young_solo_male_clips__07(answer_text, identity_text, cover_obs, metadata, bg)
    return _identity_keep_young_male(new, old, lambda: False)



def _young_solo_male_clips__07(answer_text, identity_text, cover_obs, metadata, background=None):
    """0.7-pre9 YOUNG_SOLO_MALE_CLIPS: identity_text is the identity after the
    unconfirmed-leads rule; answer_text the identity model's own answer before it.
    When that rule set "uncertain" over a male or uncertain answer, on a one-act
    credit whose cover shows one teenage boy and nothing group-like, return one
    young male lead (or one adult male lead for a documented adult). Otherwise
    identity_text unchanged."""
    t = str(identity_text or "")
    if (not YOUNG_SOLO_MALE_CLIPS
            or "corroboration did not establish distinct singers" not in t
            or not re.search(r"(?mi)^SINGER_IDENTITY=uncertain\s*$", t)):
        return identity_text
    a = str(answer_text or "")
    if not re.search(r"(?mi)^SINGER_IDENTITY=(?:post_puberty_male|adolescent_male_likely|uncertain)\s*$", a):
        return identity_text
    if "REFERENCE_VOCAL_CORRECTION=yes" in a:
        return identity_text
    artist = str((metadata or {}).get("artist") or "").strip()
    if not artist or re.search(r"(?i)\b(?:feat\.?|featuring|ft\.?|with|and|x|vs\.?)\s|[&+/,]", artist + " "):
        return identity_text
    cov = cover_obs if isinstance(cover_obs, dict) else {}
    if str(cov.get("people") or "").strip().lower() not in ("one", "1"):
        return identity_text
    cues = str(cov.get("person_cues") or "")
    if not (_YOUNG_COVER_RE.search(cues) and _COVER_MALE_RE.search(cues)) or _COVER_FEMALE_RE.search(cues):
        return identity_text
    _r = re.search(r"(?m)^REASONING=(.*)$", a)
    if _GROUPISH_RE.search(_r.group(1) if _r else ""):
        return identity_text
    bg = ""
    try:
        bg = background() if callable(background) else (background or "")
    except Exception:
        bg = ""
    if _GROUPISH_RE.search(str(bg or "")[:600]):
        return identity_text
    _adult = _documented_adult(a, bg, (metadata or {}).get("year"), artist)
    why = (f"The credited solo artist is {artist}; the cover shows one teenage boy, and nothing points to a "
           "group, duo or featured singer. The sampled clips heard both male- and female-sounding voices but "
           "no second singer was confirmed")
    if _adult:
        return ("SINGER_IDENTITY=post_puberty_male\nREASONING=" + why + f"; he is documented as an adult ({_adult}), "
                "so this is one adult male lead.\nCONFIDENCE=medium\nLEAD_SINGERS=one\n"
                "YOUNG_SOLO_MALE_CLIPS=applied: documented adult\n")
    return ("SINGER_IDENTITY=adolescent_male_likely\nREASONING=" + why + ", which fits one young male voice that "
            "hadn't fully deepened rather than two singers.\nCONFIDENCE=medium\nLEAD_SINGERS=one\n"
            "YOUNG_SOLO_MALE_CLIPS=applied\n")


def _young_solo_male_clips(*args, **kwargs):
    """SINGER_IDENTITY_PRE6 picks the 0.7.1-pre6 version; False gives 0.7's."""
    return (_young_solo_male_clips__pre6 if SINGER_IDENTITY_PRE6 else _young_solo_male_clips__07)(*args, **kwargs)


def resolve_singer_identity__pre6(metadata, vocal_audit_text, cover_obs, corrections=None,
                            wiki_context=None):
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

    if not _has_real_metadata(metadata) and not cover_obs and not _IDENTITY_USER_PHOTO:   # 0.8-pre34
        return _audio_only_singer_identity(vocal_audit_text)

    meta_lines = []
    _guessed = str((metadata or {}).get("_from_filename") or "") if IDENTITY_FILENAME_NOTE else ""
    for key, label in (("title", "Title"), ("artist", "Artist"), ("album", "Album"), ("year", "Year")):
        val = str((metadata or {}).get(key) or "").strip()
        if val:
            # pre53: a value guessed from the filename is a guess, not a tag.
            meta_lines.append(f"{label}: {val}" + (" (guessed from the filename, not a tag -- may be wrong)"
                                                    if key in _guessed else ""))

    meta_block = "\n".join(meta_lines) if meta_lines else "No file metadata available."

    corr_lines = []
    for k, v in (corrections or {}).items():
        corr_lines.append(f"- {k}: {v}")
    corr_block = "\n".join(corr_lines) if corr_lines else "None"

    cover_block = (_format_cover_observation_block(cover_obs) if cover_obs else "No cover art observations available."
                   ) + _identity_photo_block()      # 0.8-pre34 --photo ("" without one)

    _si_base = SINGER_IDENTITY_RESOLUTION_PROMPT
    if IDENTITY_FROM_ARTIST_WHEN_OPEN:
        _si_base = _si_base.replace(_IDENTITY_UNCERTAIN_OLD, _IDENTITY_UNCERTAIN_NEW, 1)
    _si_instructions = _si_base + (
        (SINGER_IDENTITY_GROUP_PROMPT.replace(_IDENTITY_GROUP_OLD, _IDENTITY_GROUP_NEW)
         if IDENTITY_SOLO_STAYS_ONE else SINGER_IDENTITY_GROUP_PROMPT)
        if SINGER_IDENTITY_GROUP_LEADS else "")
    if MULTI_LEAD_BAND_RULE and SINGER_IDENTITY_GROUP_LEADS:      # 0.7.1-pre8
        _si_instructions += SINGER_IDENTITY_MULTI_LEAD_NOTE
    if NAME_GENDER_NOTE:                                           # 0.7.1-pre10
        _si_instructions += SINGER_IDENTITY_NAME_GENDER_NOTE
    if VOCAL_FREE_TEXT_GENDER:
        _si_instructions += SINGER_IDENTITY_DEV55_NOTE
    if VOCAL_UNBROKEN_SIGNATURE and _VOICE_UNBROKEN_MARK in str(vocal_audit_text or ""):
        _si_instructions += SINGER_IDENTITY_UNBROKEN_NOTE
    if IDENTITY_FLIP_NOTE and _identity_flip_pattern(vocal_audit_text):      # dev46
        _si_instructions += SINGER_IDENTITY_FLIP_NOTE
    _si_instructions += "\nA feature credit or group membership alone does not establish shared lead vocals. Require recording-specific vocal evidence; featured instrumentalists and backing-only performers are not co-leads. Cover art never establishes the singer's age or proves the pictured person is singing. Preserve uncertainty and distinguish a named identity from an acoustic category.\n"
    _si_instructions += _unbroken_colead_prompt_note(vocal_audit_text, wiki_context)
    prompt = (
        _si_instructions
        + f"\n\nUSER CORRECTIONS (ground truth):\n{corr_block}"
        + f"\n\nTRACK METADATA:\n{meta_block}"
        + f"\n\nVOCAL EVIDENCE:\n{(vocal_audit_text or '').strip() or 'No vocal evidence available.'}"
        + f"\n\nCOVER ART OBSERVATIONS:\n{cover_block}"
    )

    # Background on the credited artist, when available. Without it this pass
    # could only ever see a name with nothing attached, so a documented solo
    # artist's gender -- often the clearest signal present -- was invisible to
    # it and the answer fell back to "uncertain".
    wiki_text = str(wiki_context or "").strip()
    if SINGER_IDENTITY_FIT_PROMPT:
        prompt = _singer_identity_prompt_fitted(
            _si_instructions, corr_block, meta_block, vocal_audit_text, cover_block, wiki_text)
    elif wiki_text:
        if len(wiki_text) > 4000:
            wiki_text = wiki_text[:4000] + " ..."
        prompt += (
            "\n\nBACKGROUND ON THE CREDITED ARTIST (reference material):\n"
            + wiki_text
            + "\n\nUse this ONLY to establish who the artist is, where stated or "
              "unambiguous whether they are a man or a woman, and — if this specific "
              "recording's era is identifiable from the track metadata/title/album — "
              "whether that era is documented as a child/adolescent period for them "
              "(e.g. a child-star or teen-pop debut, a voice not yet documented as "
              "changed) rather than their adult or current-day status. Take nothing "
              "else from it."
        )

    # Batch: ephemeral load — unload weights after the reply; analysis text and
    # Musiclyse session state remain in this process's RAM.
    _ka = None
    if globals().get("BATCH_UNLOAD_OLLAMA", True) and _is_batch_context():
        _ka = globals().get("BATCH_OLLAMA_KEEP_ALIVE", 0)
        if SINGER_IDENTITY_REUSE_BATCH_MODEL and _ka in (0, "0", "0s"):
            _ka = "5m"

    # Same reasoning as describe_cover_art()'s _cover_art_ctx above: a
    # different num_ctx than the resident interactive instance forces a full
    # reload and cache loss. This call also runs mid-conversation on track
    # load, so only use the small dedicated context in batch, where the model
    # is unloaded right after anyway.
    _singer_identity_ctx = SINGER_IDENTITY_NUM_CTX if _is_batch_context() else None
    if SINGER_IDENTITY_FIT_PROMPT and _singer_identity_ctx is not None:
        _singer_identity_ctx = _identity_ctx_for_prompt(prompt, _singer_identity_ctx)

    try:
        _si_text, _usage = ollama_chat(
            [{"role": "user", "content": prompt}],
            num_ctx=_singer_identity_ctx,
            keep_alive=_ka,
            options=_scan_llm_options(),
        )
        _si_text = _unbroken_colead_identity(_si_text)
        if _identity_colead_recheck_needed(_si_text, wiki_text):
            # Same already-retrieved reference context; no new browsing or audio decisions.
            try:
                recheck = (prompt + _unbroken_colead_prompt_note(_si_text, wiki_text) + "\n\nThe initial result below identifies an unbroken male but may have "
                    "omitted a documented female co-lead. Recheck THIS recording only. Preserve the male "
                    "identity; do not infer a female lead from a female-sounding acoustic tag or group membership. "
                    "If both leads are documented, output the six UNBROKEN_MALE_LEAD / FEMALE_COLEAD fields "
                    "specified above, SINGER_IDENTITY=mixed_leads and LEAD_SINGERS=several. Also output "
                    "MALE_IDENTITY_SOURCE= and FEMALE_IDENTITY_SOURCE= as exact single-line excerpts from "
                    "BACKGROUND, naming each performer. The female excerpt must name this track and explicitly "
                    "document co-lead or duet participation. If unsupported, retain the initial result.\n"
                    "INITIAL RESULT:\n" + _si_text)
                _recheck_ctx = (_identity_ctx_for_prompt(recheck, _singer_identity_ctx)
                                if SINGER_IDENTITY_FIT_PROMPT and _singer_identity_ctx is not None
                                else _singer_identity_ctx)
                candidate, _ = ollama_chat([{"role": "user", "content": recheck}],
                                           num_ctx=_recheck_ctx, keep_alive=_ka,
                                           options=_scan_llm_options())
                checked = _identity_colead_source_checked(candidate, wiki_text, metadata)
                if checked:
                    _si_text = checked
            except Exception:
                pass  # A failed optional recheck never erases the original identity.
        if (wiki_text and "UNBROKEN_MALE_IDENTITY_PRESERVED=yes" not in str(_si_text)
                and not re.search(r"(?i)\b(?:feat\.?|featuring|ft\.?|with)\s+", str((metadata or {}).get("artist", "")) + " " + str((metadata or {}).get("title", "")))
                and re.search(r"FINAL LEAD PROFILE\s*[:=]\s*mixed_leads", str(vocal_audit_text or ""))):
            try:
                review = (prompt + "\n\nREFERENCE CROSS-CHECK: acoustic register changes can falsely imply female co-leads. "
                    "Only correct this when positive reference evidence explicitly establishes a sole male singer "
                    "or an all-male vocal group for this recording. Absence of female credits is insufficient. "
                    "Preserve documented guests, female co-leads, mixed groups, and uncertain cases. "
                    "Do not infer age from pitch. If fully supported output SINGER_IDENTITY, REFERENCE_SCOPE="
                    "single_male or all_male_group, REFERENCE_SINGER (person or group name), "
                    "REFERENCE_TRACK_SOURCE (exact single-line excerpt naming this track and lead vocal credit), "
                    "REFERENCE_IDENTITY_SOURCE (exact excerpt naming the person/group and explicitly supporting "
                    "sole/only singer and male identity, or all-male lineup). Otherwise output NO_CORRECTION. "
                    "Never invent or paraphrase excerpts. Alternative single_male support: a COMPLETE Personnel or Credits "
                    "list in THIS TRACK'S article can establish a single credited vocalist without the words sole/only. "
                    "The list must have a following non-personnel section heading and exactly one vocal performer; "
                    "another lead, backing singer, guest or truncated/ambiguous credits disqualify this alternative. "
                    "For this case REFERENCE_TRACK_SOURCE is the exact vocalist credit row, and REFERENCE_IDENTITY_SOURCE "
                    "still independently supports that named person's male identity. Artist/album lists do not qualify. "
                    "Do not follow an acoustic instruction to force mixed_leads when verified identity credits contradict it. "
                    "Otherwise retain the original identity.\nINITIAL IDENTITY:\n" + _si_text)
                ctx = (_identity_ctx_for_prompt(review, _singer_identity_ctx)
                       if SINGER_IDENTITY_FIT_PROMPT and _singer_identity_ctx is not None else _singer_identity_ctx)
                candidate, _ = ollama_chat([{"role": "user", "content": review}], num_ctx=ctx, keep_alive=_ka,
                                           options=_scan_llm_options())
                checked = _reference_vocal_correction(candidate, wiki_text, metadata)
                if checked:
                    _si_text = checked + "\nREFERENCE_IDENTITY_CHECK=accepted"
                else:
                    _si_text += "\nREFERENCE_IDENTITY_CHECK=not accepted; original identity retained"
                    _si_text += "\nREFERENCE_REVIEW_RESPONSE_JSON=" + json.dumps(str(candidate or "")[:6000], ensure_ascii=False)
            except Exception as exc:
                _si_text += "\nREFERENCE_IDENTITY_CHECK=review failed; original identity retained (" + type(exc).__name__ + ")"
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


def resolve_singer_identity__07(metadata, vocal_audit_text, cover_obs, corrections=None,
                            wiki_context=None):
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

    if not _has_real_metadata(metadata) and not cover_obs and not _IDENTITY_USER_PHOTO:   # 0.8-pre34
        return _audio_only_singer_identity(vocal_audit_text)

    meta_lines = []
    _guessed = str((metadata or {}).get("_from_filename") or "") if IDENTITY_FILENAME_NOTE else ""
    for key, label in (("title", "Title"), ("artist", "Artist"), ("album", "Album"), ("year", "Year")):
        val = str((metadata or {}).get(key) or "").strip()
        if val:
            # pre53: a value guessed from the filename is a guess, not a tag.
            meta_lines.append(f"{label}: {val}" + (" (guessed from the filename, not a tag -- may be wrong)"
                                                    if key in _guessed else ""))

    meta_block = "\n".join(meta_lines) if meta_lines else "No file metadata available."

    corr_lines = []
    for k, v in (corrections or {}).items():
        corr_lines.append(f"- {k}: {v}")
    corr_block = "\n".join(corr_lines) if corr_lines else "None"

    cover_block = (_format_cover_observation_block(cover_obs) if cover_obs else "No cover art observations available."
                   ) + _identity_photo_block()      # 0.8-pre34 --photo ("" without one)

    _si_base = SINGER_IDENTITY_RESOLUTION_PROMPT
    if IDENTITY_FROM_ARTIST_WHEN_OPEN:
        _si_base = _si_base.replace(_IDENTITY_UNCERTAIN_OLD, _IDENTITY_UNCERTAIN_NEW, 1)
    _si_instructions = _si_base + (
        (SINGER_IDENTITY_GROUP_PROMPT.replace(_IDENTITY_GROUP_OLD, _IDENTITY_GROUP_NEW)
         if IDENTITY_SOLO_STAYS_ONE else SINGER_IDENTITY_GROUP_PROMPT)
        if SINGER_IDENTITY_GROUP_LEADS else "")
    if MULTI_LEAD_BAND_RULE and SINGER_IDENTITY_GROUP_LEADS:      # 0.7.1-pre8
        _si_instructions += SINGER_IDENTITY_MULTI_LEAD_NOTE
    if NAME_GENDER_NOTE:                                           # 0.7.1-pre10
        _si_instructions += SINGER_IDENTITY_NAME_GENDER_NOTE
    if VOCAL_FREE_TEXT_GENDER:
        _si_instructions += SINGER_IDENTITY_DEV55_NOTE
    if VOCAL_UNBROKEN_SIGNATURE and _VOICE_UNBROKEN_MARK in str(vocal_audit_text or ""):
        _si_instructions += SINGER_IDENTITY_UNBROKEN_NOTE
    if IDENTITY_FLIP_NOTE and _identity_flip_pattern(vocal_audit_text):      # dev46
        _si_instructions += SINGER_IDENTITY_FLIP_NOTE
    _si_instructions += _unbroken_colead_prompt_note(vocal_audit_text, wiki_context)
    prompt = (
        _si_instructions
        + f"\n\nUSER CORRECTIONS (ground truth):\n{corr_block}"
        + f"\n\nTRACK METADATA:\n{meta_block}"
        + f"\n\nVOCAL EVIDENCE:\n{(vocal_audit_text or '').strip() or 'No vocal evidence available.'}"
        + f"\n\nCOVER ART OBSERVATIONS:\n{cover_block}"
    )

    # Background on the credited artist, when available. Without it this pass
    # could only ever see a name with nothing attached, so a documented solo
    # artist's gender -- often the clearest signal present -- was invisible to
    # it and the answer fell back to "uncertain".
    wiki_text = str(wiki_context or "").strip()
    if SINGER_IDENTITY_FIT_PROMPT:
        prompt = _singer_identity_prompt_fitted(
            _si_instructions, corr_block, meta_block, vocal_audit_text, cover_block, wiki_text)
    elif wiki_text:
        if len(wiki_text) > 4000:
            wiki_text = wiki_text[:4000] + " ..."
        prompt += (
            "\n\nBACKGROUND ON THE CREDITED ARTIST (reference material):\n"
            + wiki_text
            + "\n\nUse this ONLY to establish who the artist is, where stated or "
              "unambiguous whether they are a man or a woman, and — if this specific "
              "recording's era is identifiable from the track metadata/title/album — "
              "whether that era is documented as a child/adolescent period for them "
              "(e.g. a child-star or teen-pop debut, a voice not yet documented as "
              "changed) rather than their adult or current-day status. Take nothing "
              "else from it."
        )

    # Batch: ephemeral load — unload weights after the reply; analysis text and
    # Musiclyse session state remain in this process's RAM.
    _ka = None
    if globals().get("BATCH_UNLOAD_OLLAMA", True) and _is_batch_context():
        _ka = globals().get("BATCH_OLLAMA_KEEP_ALIVE", 0)
        if SINGER_IDENTITY_REUSE_BATCH_MODEL and _ka in (0, "0", "0s"):
            _ka = "5m"

    # Same reasoning as describe_cover_art()'s _cover_art_ctx above: a
    # different num_ctx than the resident interactive instance forces a full
    # reload and cache loss. This call also runs mid-conversation on track
    # load, so only use the small dedicated context in batch, where the model
    # is unloaded right after anyway.
    _singer_identity_ctx = SINGER_IDENTITY_NUM_CTX if _is_batch_context() else None
    if SINGER_IDENTITY_FIT_PROMPT and _singer_identity_ctx is not None:
        _singer_identity_ctx = _identity_ctx_for_prompt(prompt, _singer_identity_ctx)

    try:
        _si_text, _usage = ollama_chat(
            [{"role": "user", "content": prompt}],
            num_ctx=_singer_identity_ctx,
            keep_alive=_ka,
            options=_scan_llm_options(),
        )
        _si_text = _unbroken_colead_identity(_si_text)
        if _identity_colead_recheck_needed(_si_text, wiki_text):
            # Same already-retrieved reference context; no new browsing or audio decisions.
            try:
                recheck = (prompt + _unbroken_colead_prompt_note(_si_text, wiki_text) + "\n\nThe initial result below identifies an unbroken male but may have "
                    "omitted a documented female co-lead. Recheck THIS recording only. Preserve the male "
                    "identity; do not infer a female lead from a female-sounding acoustic tag or group membership. "
                    "If both leads are documented, output the six UNBROKEN_MALE_LEAD / FEMALE_COLEAD fields "
                    "specified above, SINGER_IDENTITY=mixed_leads and LEAD_SINGERS=several. Also output "
                    "MALE_IDENTITY_SOURCE= and FEMALE_IDENTITY_SOURCE= as exact single-line excerpts from "
                    "BACKGROUND, naming each performer. The female excerpt must name this track and explicitly "
                    "document co-lead or duet participation. If unsupported, retain the initial result.\n"
                    "INITIAL RESULT:\n" + _si_text)
                _recheck_ctx = (_identity_ctx_for_prompt(recheck, _singer_identity_ctx)
                                if SINGER_IDENTITY_FIT_PROMPT and _singer_identity_ctx is not None
                                else _singer_identity_ctx)
                candidate, _ = ollama_chat([{"role": "user", "content": recheck}],
                                           num_ctx=_recheck_ctx, keep_alive=_ka,
                                           options=_scan_llm_options())
                checked = _identity_colead_source_checked(candidate, wiki_text, metadata)
                if checked:
                    _si_text = checked
            except Exception:
                pass  # A failed optional recheck never erases the original identity.
        if (wiki_text and "UNBROKEN_MALE_IDENTITY_PRESERVED=yes" not in str(_si_text)
                and not re.search(r"(?i)\b(?:feat\.?|featuring|ft\.?|with)\s+", str((metadata or {}).get("artist", "")) + " " + str((metadata or {}).get("title", "")))
                and re.search(r"FINAL LEAD PROFILE\s*[:=]\s*mixed_leads", str(vocal_audit_text or ""))):
            try:
                review = (prompt + "\n\nREFERENCE CROSS-CHECK: acoustic register changes can falsely imply female co-leads. "
                    "Only correct this when positive reference evidence explicitly establishes a sole male singer "
                    "or an all-male vocal group for this recording. Absence of female credits is insufficient. "
                    "Preserve documented guests, female co-leads, mixed groups, and uncertain cases. "
                    "Do not infer age from pitch. If fully supported output SINGER_IDENTITY, REFERENCE_SCOPE="
                    "single_male or all_male_group, REFERENCE_SINGER (person or group name), "
                    "REFERENCE_TRACK_SOURCE (exact single-line excerpt naming this track and lead vocal credit), "
                    "REFERENCE_IDENTITY_SOURCE (exact excerpt naming the person/group and explicitly supporting "
                    "sole/only singer and male identity, or all-male lineup). Otherwise output NO_CORRECTION. "
                    "Never invent or paraphrase excerpts. Alternative single_male support: a COMPLETE Personnel or Credits "
                    "list in THIS TRACK'S article can establish a single credited vocalist without the words sole/only. "
                    "The list must have a following non-personnel section heading and exactly one vocal performer; "
                    "another lead, backing singer, guest or truncated/ambiguous credits disqualify this alternative. "
                    "For this case REFERENCE_TRACK_SOURCE is the exact vocalist credit row, and REFERENCE_IDENTITY_SOURCE "
                    "still independently supports that named person's male identity. Artist/album lists do not qualify. "
                    "Do not follow an acoustic instruction to force mixed_leads when verified identity credits contradict it. "
                    "Otherwise retain the original identity.\nINITIAL IDENTITY:\n" + _si_text)
                ctx = (_identity_ctx_for_prompt(review, _singer_identity_ctx)
                       if SINGER_IDENTITY_FIT_PROMPT and _singer_identity_ctx is not None else _singer_identity_ctx)
                candidate, _ = ollama_chat([{"role": "user", "content": review}], num_ctx=ctx, keep_alive=_ka,
                                           options=_scan_llm_options())
                checked = _reference_vocal_correction(candidate, wiki_text, metadata)
                if checked:
                    _si_text = checked + "\nREFERENCE_IDENTITY_CHECK=accepted"
                else:
                    _si_text += "\nREFERENCE_IDENTITY_CHECK=not accepted; original identity retained"
                    _si_text += "\nREFERENCE_REVIEW_RESPONSE_JSON=" + json.dumps(str(candidate or "")[:6000], ensure_ascii=False)
            except Exception as exc:
                _si_text += "\nREFERENCE_IDENTITY_CHECK=review failed; original identity retained (" + type(exc).__name__ + ")"
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


def resolve_singer_identity(*args, **kwargs):
    """IDENTITY_SHARED_LEAD_EVIDENCE picks the 0.7.1-pre6 version; False gives 0.7's."""
    return (resolve_singer_identity__pre6 if IDENTITY_SHARED_LEAD_EVIDENCE else resolve_singer_identity__07)(*args, **kwargs)


def _stem_identity_note(stem, accepted_labels, meta=None):
    """Say what is actually in a stem when it is not what its name implies.

    A stem's name was being used as the instrument's identity -- every report
    line, the prominence ranking and the arrangement changes said "bass" for
    whatever Demucs routed to the bass stem, and a 'bass guitar' tag on it was
    automatically treated as confirmed. Returns "" when the stem is silent,
    has no description, or its tags agree with its name.
    """
    if stem not in _STEM_BUCKET_DESCRIPTION:
        return ""
    if meta and meta.get("silent"):
        return ""
    labels = list(accepted_labels or [])
    if stem == "guitar":
        expected_hit = any(_is_guitar_family_label(l) for l in labels)
        expected_name = "a guitar"
    else:
        expected_hit = any(l in _STEM_EXPECTED_LABELS.get(stem, set()) for l in labels)
        expected_name = {"bass": "a bass guitar", "piano": "a piano"}.get(stem, stem)
    if expected_hit:
        return ""
    if labels:
        return (
            f"STEM IDENTITY NOTE: this is the separation stem named '{stem}', which collects "
            f"{_STEM_BUCKET_DESCRIPTION[stem]}. The instrument classifier does not identify "
            f"{expected_name} here; it hears {', '.join(labels)}. Describe this part by what "
            f"was identified (or by its register and texture), not as {expected_name}."
        )
    if accepted_labels is None:
        return ""
    return (
        f"STEM IDENTITY NOTE: this is the separation stem named '{stem}', which collects "
        f"{_STEM_BUCKET_DESCRIPTION[stem]}. Nothing here confirms which instrument it is, so "
        f"describe it by register and texture (e.g. 'the low part') rather than as "
        f"{expected_name}, unless other evidence or the user names it."
    )


def _video_rule_cover(cover_observations, heavy, identity_text="", metadata=None):
    """What the singer identity's rule steps read as the cover. A real cover is always used as it is.
    For a music video without one: the people the video shows singing (_video_performers), as
    PEOPLE / PERSON_CUES, but only when someone is seen singing -- and never for an artist the
    singer lookup, the identity reasoning or the artist's own article describes as a woman."""
    if cover_observations:
        return cover_observations
    _VIDEO_PERFORMERS_LAST.clear()
    path = str((heavy or {}).get("track_path") or "")
    if not (path and _is_video_input(path)):
        return cover_observations
    md = metadata if isinstance(metadata, dict) else {}
    artist = str(md.get("artist") or "").strip()
    try:
        background = _artist_background_for_identity(md) if artist else ""
        if artist and _fsnl_artist_female(identity_text, _identity_reason_text(identity_text), artist, background):
            return cover_observations
    except Exception:
        pass
    try:
        obs = _video_performers(heavy)
    except Exception as e:
        print(f"  (music video performers skipped: {e})")
        obs = {}
    if not obs:
        return cover_observations
    _VIDEO_PERFORMERS_LAST.update(obs)
    _VIDEO_PERFORMERS_LAST.update(_source=path, _at=time.time())
    sung = str(obs.get("singing_frames") or "").strip().lower()
    if (obs.get("people") in ("one", "two", "three_or_more") and re.search(r"\d", sung)
            and not sung.startswith("none")):
        return {"people": obs["people"], "person_cues": obs.get("person_cues", ""),
                "confidence": obs.get("confidence", ""), "source": "music video: people seen singing"}
    return cover_observations


def _identity_step(current, fn, *args, **kwargs):
    """0.8-pre31 IDENTITY_STEPS_SEPARATE: one rule step of the singer identity. When it fails, the
    identity is left as it was (`current`), a note says so, and the steps after it still run.
    False: the step runs as before -- a failure ends the rule steps (IDENTITY_KEEP_LAST_GOOD)."""
    if not IDENTITY_STEPS_SEPARATE:
        return fn(*args, **kwargs)
    try:
        return fn(*args, **kwargs)
    except Exception as e:
        print(f"  (singer identity: one rule step was skipped -- {type(e).__name__}: {str(e)[:200]})")
        return current


# 0.8-pre34 SCAN_IMAGE_OPTIONS: --photo, a picture of the artist for the singer identity only.
_IDENTITY_USER_PHOTO = {}        # the photo's description while the identity question is asked
_SCAN_PHOTO_SEEN = {}            # image path -> its description, once per process


def _scan_user_photo_observations():
    """0.8-pre34: the chat model's description of the --photo image, as for a cover ({} without one)."""
    path = os.environ.get("MUSICLYSE_USER_PHOTO", "") if SCAN_IMAGE_OPTIONS else ""
    if not path:
        return {}
    if path not in _SCAN_PHOTO_SEEN:
        obs = {}
        try:
            with open(path, "rb") as f:
                raw = f.read()
            prepared = prepare_cover_image_for_ollama(raw, _guess_image_mime(raw))
            if prepared:
                status("Describing the photo of the artist...")
                obs = describe_cover_art(base64.b64encode(prepared).decode("utf-8")) or {}
        except Exception as e:
            print(f"  (--photo image skipped: {e})")
        _SCAN_PHOTO_SEEN[path] = obs
    return _SCAN_PHOTO_SEEN[path]


def _identity_photo_block():
    """0.8-pre34: the photo for the identity question, after the cover observations ("" without one)."""
    if not _IDENTITY_USER_PHOTO:
        return ""
    lines = _format_cover_observation_block(_IDENTITY_USER_PHOTO).split("\n")[1:]
    if SCAN_PHOTO_IS_LEAD:
        # 0.8-pre35: for a group, the photo shows its lead singer; it never makes the group a solo act
        return ("\n\nPHOTO OF THE ARTIST (a picture the user supplied of who performs this track -- for a "
                "group or duo, its lead singer -- not the cover; visual evidence, not direct audio proof):\n"
                + "\n".join(lines)
                + "\nUse it as identity evidence for the lead singer, as you would the cover art. A photo of "
                  "one person doesn't make a documented group or duo a solo act. In REASONING call it the "
                  "photo of the artist.")
    return ("\n\nPHOTO OF THE ARTIST (a picture the user supplied of who performs this track -- not the cover; "
            "visual evidence, not direct audio proof):\n" + "\n".join(lines)
            + "\nUse it as identity evidence, as you would the cover art. In REASONING call it the photo of "
              "the artist.")


def _resolve_singer_identity_with_photo(*args, **kwargs):
    """0.8-pre34: resolve_singer_identity, with the --photo description when the scan has one."""
    photo = _scan_user_photo_observations() if _scan_user_photo_given() else {}
    if not photo:
        return resolve_singer_identity(*args, **kwargs)
    _IDENTITY_USER_PHOTO.clear()
    _IDENTITY_USER_PHOTO.update(photo)
    try:
        return resolve_singer_identity(*args, **kwargs)
    finally:
        _IDENTITY_USER_PHOTO.clear()


def _photo_rule_cover(cover_observations):
    """0.8-pre34: what the identity's rule steps read as the cover -- the --photo description when the
    scan has one and the cover shows nobody (or there is no cover); otherwise the cover as before."""
    photo = _scan_user_photo_observations() if _scan_user_photo_given() else {}
    if not photo:
        return cover_observations
    if SCAN_PHOTO_IS_LEAD:
        # 0.8-pre35: the photo describes the lead; the people count stays the larger of the cover's
        # and the photo's, so a group or duo on the cover is still read as one by the rules
        out = dict(photo, source="photo of the artist supplied by the user")
        rank = {"none": 0, "one": 1, "two": 2, "three_or_more": 3}
        cov = cover_observations if isinstance(cover_observations, dict) else {}
        cover_people = "three_or_more" if _cover_shows_group(cov) else _cover_people_count(cov.get("people"))
        photo_people = "three_or_more" if _cover_shows_group(photo) else _cover_people_count(photo.get("people"))
        if rank.get(cover_people, 0) > rank.get(photo_people, 0):
            out["people"] = cover_people
        return out
    if cover_observations and _cover_people_count((cover_observations or {}).get("people")) in (
            "one", "two", "three_or_more"):
        return cover_observations
    return dict(photo, source="photo of the artist supplied by the user")


# 0.8-pre35 IDENTITY_RULE_SOURCE_WORDING: the rules' fixed sentences about "the cover", for evidence
# that came from the --photo image or from the music video's people seen singing.
_RULE_COVER_PHRASES = (
    ("the cover art shows him with", "{src} shows him with"),
    ("with a young person on the cover", "with a young person in {src}"),
    ("the cover shows one young boy", "{src} shows one young boy"),
    ("the cover shows one teenage boy", "{src} shows one teenage boy"),
)


def _rule_cover_wording(text, rule_cover, answer=""):
    """The identity text with the rules' "cover" sentences naming where the evidence came from, when
    the rules read the --photo image or the music video; otherwise unchanged. Phrases already in the
    model's own answer (`answer`) are left as they are."""
    if not IDENTITY_RULE_SOURCE_WORDING or not text or not isinstance(rule_cover, dict):
        return text
    source = str(rule_cover.get("source") or "")
    if source.startswith("photo of the artist"):
        src = "the photo of the artist"
    elif source.startswith("music video"):
        src = "the music video"
    else:
        return text
    for old, new in _RULE_COVER_PHRASES:
        if old in text and old not in str(answer or ""):
            text = text.replace(old, new.format(src=src))
    return text


def resolve_identity_and_finalize__pre6(heavy):
    """Lightweight second stage: resolve singer identity, then append the
    RECOMMENDED STRUCTURE block at the absolute end of the analysis so it
    survives context truncation and is always available for structure questions.
    """
    revised = heavy.get("analysis", "") or ""
    metadata = heavy.get("metadata")
    cover_observations = heavy.get("cover_observations")
    final_lead = heavy.get("final_lead")
    initial_backing = heavy.get("initial_backing")
    vocal_result_present = bool(heavy.get("vocal_result_present"))
    vocal_audit_for_resolution = heavy.get("vocal_audit_for_resolution")
    objective_report = heavy.get("_objective_report", "")   # now passed through

    singer_identity = ""
    # No singer to identify on a measured instrumental or unresolved presence.
    # Untagged files use the accepted acoustic category without a model lookup.
    if (ENABLE_SINGER_IDENTITY_RESOLUTION and VOCAL_PRESENCE_UNRESOLVED_MARKER not in revised and not heavy.get("instrumental")
            and (_has_real_metadata(metadata) or cover_observations or _scan_user_photo_given()
                 or _audio_only_singer_identity(vocal_audit_for_resolution))):
        status("Resolving singer identity from audio + metadata + cover art...")
        try:
            singer_identity = _resolve_singer_identity_with_photo(   # 0.8-pre34
                metadata,
                vocal_audit_for_resolution or "",
                cover_observations,
                {},
                wiki_context=_artist_background_for_identity(metadata),
            ) or ""
            if IDENTITY_CANONICAL_REPLY:                                  # 0.7.1-pre12
                singer_identity = _canonical_identity_reply(singer_identity)
            _si_last_good = singer_identity
            _identity_answer = singer_identity
            # 0.8-pre31 VIDEO_PERFORMERS: what the rule steps read as the cover (the real cover when
            # there is one); the identity question above never sees it.
            _rc_cover = _photo_rule_cover(cover_observations)            # 0.8-pre34 --photo
            _rc = (_video_rule_cover(_rc_cover, heavy, _identity_answer, metadata)
                   if VIDEO_PERFORMERS else _rc_cover)
            _st = _identity_step                                       # 0.8-pre31 IDENTITY_STEPS_SEPARATE
            singer_identity = _st(singer_identity, _reconcile_identity_with_section_voice,
                                  singer_identity, revised) or ""
            singer_identity = _st(singer_identity, _young_solo_male_clips,
                _identity_answer, singer_identity, _rc, metadata,
                background=lambda: _artist_background_for_identity(metadata)) or ""
            singer_identity = _st(singer_identity, _feature_credit_identity, singer_identity, metadata, _rc) or ""
            singer_identity = _st(singer_identity, _young_male_voice_check,
                singer_identity, (vocal_audit_for_resolution or "") + "\n" + str(revised or ""),
                _rc, metadata,
                background=lambda: _artist_background_for_identity(metadata)) or ""
            if FEATURED_SINGER_NOT_LEAD:                               # 0.7.1-pre13
                singer_identity = _st(singer_identity, _featured_singer_not_lead,
                    singer_identity, _rc, metadata,
                    background=lambda: _artist_background_for_identity(metadata),
                    evidence_text=(vocal_audit_for_resolution or "") + "\n" + str(revised or "")) or singer_identity
            if YOUNG_LEAD_GROUP_CHECK:                                 # 0.7.1-pre11
                singer_identity = _st(singer_identity, _qualify_young_lead_in_group,
                    singer_identity, (vocal_audit_for_resolution or "") + "\n" + str(revised or ""),
                    _rc) or ""
            _saved_young = _st(False, _identity_saved_unbroken_lead, singer_identity, metadata)
            if _saved_young and "UNBROKEN_MALE_IDENTITY_PRESERVED=yes" not in singer_identity:
                singer_identity = singer_identity.rstrip() + "\nUNBROKEN_MALE_IDENTITY_PRESERVED=yes\n"
            _maturity = _st("", _identity_young_voice_maturity,
                singer_identity, (vocal_audit_for_resolution or "") + "\n" + revised, metadata)
            if _maturity and YOUNG_LEAD_GROUP_CHECK and _cover_shows_group(_rc):
                _maturity = ""                                          # 0.7.1-pre11: a group's voices
            if _maturity:
                singer_identity = singer_identity.rstrip() + "\nYOUNG_VOICE_MATURITY=" + _maturity + "\n"
            if SINGER_DETAIL_LOOKUP:                                   # 0.7.1-pre13
                singer_identity = _st(singer_identity, _singer_detail_add,
                    singer_identity, metadata, background=lambda: _artist_background_for_identity(metadata))
            singer_identity = _st(singer_identity, _rule_cover_wording,   # 0.8-pre35
                                  singer_identity, _rc, _identity_answer)
        except Exception as e:
            print(f"  (singer identity skipped: {e})")
            # 0.7.1-pre12 IDENTITY_KEEP_LAST_GOOD: a failing rule step keeps the model's answer.
            singer_identity = (locals().get("_si_last_good", "") or "") if IDENTITY_KEEP_LAST_GOOD else ""
        finally:
            if globals().get("BATCH_UNLOAD_OLLAMA", True) and _is_batch_context():
                try:
                    ollama_unload_model()
                except Exception:
                    pass
        cover_obs_block = (
            _format_cover_observation_block(cover_observations) if cover_observations else ""
        )
        if cover_obs_block and not _has_cover_observations(revised):
            revised += "\n\n" + cover_obs_block
        if SINGER_IDENTITY_TO_WRITER:
            if (singer_identity and _parse_singer_identity(singer_identity)
                    and not _has_identity_resolution_block(revised)):
                revised += _format_identity_resolution_block(singer_identity)
        elif singer_identity and "SINGER IDENTITY RESOLUTION" not in revised:
            revised += (
                f"\n\nSINGER IDENTITY RESOLUTION (combined audio + metadata + cover art; "
                f"use for who-is-singing questions about the PERSON, never to re-describe the sound):\n{singer_identity}"
            )
        resolved_tag = _parse_singer_identity(singer_identity) if singer_identity else ""
        priority_tag = _reconcile_identity_with_acoustic_age(resolved_tag, final_lead)
    # Acoustic vocal wording is emitted whether or not tags/cover exist, so the
    # same audio always gets the same measured description of the voice.
    _id_tag = (
        _parse_singer_identity(singer_identity)
        if (ENABLE_SINGER_IDENTITY_RESOLUTION and singer_identity) else ""
    )
    if _identity_overrides_gender(_id_tag, final_lead):
        # Mark the passes' own female/male tag as overruled so the writer
        # doesn't pick it back up from the raw vocal-profile text.
        revised = _mark_superseded_vocal_tags(revised, _id_tag)
    _unbroken = _VOICE_UNBROKEN_MARK in revised
    if vocal_result_present:
        revised += build_acoustic_vocal_note(
            final_lead or "unknown", initial_backing or "uncertain", identity_tag=_id_tag,
            unbroken=_unbroken,
        )
    if _id_tag:
        revised += build_singer_identity_note(_id_tag, final_lead, unbroken=_unbroken, vocal_text=revised)

    # ------------------------------------------------------------
    # FINAL STEP: force the structure block to be the very last thing
    # ------------------------------------------------------------
    # 1. Remove any existing RECOMMENDED STRUCTURE block (so we don't duplicate).
    # Matches all three header variants format_recommended_structure_block can
    # emit: "...DISCUSSION:", "...DISCUSSION (from objective change points):",
    # and "...DISCUSSION (coarse, evenly-spaced windows...):". The old pattern
    # only matched the first, so once the coarse fallback started actually
    # firing (see _resolve_total_duration) it would have been left in place
    # and duplicated below instead of replaced.
    revised = re.sub(
        r"\n\nRECOMMENDED STRUCTURE FOR DISCUSSION(?:\s*\([^)]*\))?:.*?(?=\n\n[A-Z]|\Z)",
        "",
        revised,
        flags=re.DOTALL,
    )

    # 2. Build a fresh structure block using the OBJECTIVE-FIRST pipeline
    track_path = heavy.get("track_path")
    total_duration = heavy.get("_video_audio_duration") or _resolve_total_duration(
        track_path=track_path, objective_report=objective_report
    )

    # The section map was already resolved objectively during the heavy stage,
    # while the audio model was still loaded (see run_fresh_track_analysis_heavy).
    # It is reused verbatim here.
    #
    # This block used to re-run the whole cut/cluster/label pipeline from
    # scratch at this point, which meant reloading the listening model immediately
    # after _release_heavy_analysis_memory_before_identity() had deliberately
    # freed it — paying a full model load to recompute a result the run already
    # had, and producing a section map that disagreed with the one the melody
    # and intensity blocks had already been keyed to.
    structure_sections = list(heavy.get("structure_sections") or [])
    boundary_seconds = []
    _lyric_groups_for_structure = None

    if structure_sections:
        boundary_seconds = [float(s["start"]) for s in structure_sections[1:]]
        revised = _overwrite_structure_field(revised, structure_sections)
        revised = _correct_stated_duration(revised, structure_sections[-1].get("end"))

    # Fall back only if the heavy stage genuinely produced nothing.
    if not structure_sections:
        structure_sections, boundary_seconds, _lyric_groups_for_structure = _finalize_structure_sections(
            revised,
            objective_report=objective_report,
            stem_midi_report=locals().get("stem_midi_report") or "",
            lyric_chunks=locals().get("_lyric_chunks"),
            total_duration=total_duration,
        )

    structure_block = format_recommended_structure_block(
        structure_sections,
        objective_boundaries=boundary_seconds,
        total_duration=total_duration,
        lyric_groups=_lyric_groups_for_structure,
        objective_report=objective_report,
    )
    if structure_block:
        revised = revised.rstrip() + "\n\n" + structure_block

    revised = _collapse_runaway_repetition_fields(revised)

    # Sonic embedding is computed HERE, at the end of the analysis, rather than
    # only inside save_song_data(). Previously a track only ever acquired an
    # embedding if it was saved, so a freshly analysed track could not take
    # part in a sonic-similarity comparison until it had been written to disk
    # and read back. This runs at the same point in the same process the save
    # would have run it (right after identity resolution, with the heavy
    # listening models already released), so peak memory is unchanged; the
    # result is cached and handed to save_song_data(), which reuses it instead
    # of recomputing.
    clap_vector = None
    try:
        _clap_src = heavy.get("track_path")
        if ENABLE_CLAP_EMBEDDING and _clap_src and not str(_clap_src).startswith(("http://", "https://")):
            status("Computing sonic-similarity embedding...")
            # dev55: computed in the background during the heavy stage (the
            # batch finish stage runs in another process, so it is handed over).
            clap_vector = heavy.get("clap_prefetched") or compute_clap_embedding(_clap_src)
            status_done(
                "Sonic embedding ready" if clap_vector is not None
                else "Sonic embedding unavailable"
            )
    except Exception as e:
        print(f"  (CLAP embedding skipped: {e})")
        clap_vector = None

    status_done("Analysis complete")

    out = dict(heavy)
    out["analysis"] = _normalize_unresolved_vocal_report(revised)
    try:
        out["analysis"] = _apply_moss_genre_clarifications(
            out["analysis"], out.pop("_genre_clarifications", {}),
            out.pop("_genre_clarification_panns", []))
    except Exception as e:
        print(f"  (genre clarification not applied: {type(e).__name__}: {e})")
    out["analysis"] = _deduplicate_final_genre_ranking(out["analysis"])
    out["singer_identity"] = singer_identity
    out["structure_sections"] = structure_sections
    out["clap_embedding"] = clap_vector or []
    # clean up temporary keys
    for k in ("_objective_report", "vocal_audit_for_resolution", "final_lead",
              "initial_backing", "vocal_result_present", "instrumental", "clap_prefetched",
              "moss_first_pass", "_video_audio_duration"):
        out.pop(k, None)
    return out


def resolve_identity_and_finalize__07(heavy):
    """Lightweight second stage: resolve singer identity, then append the
    RECOMMENDED STRUCTURE block at the absolute end of the analysis so it
    survives context truncation and is always available for structure questions.
    """
    revised = heavy.get("analysis", "") or ""
    metadata = heavy.get("metadata")
    cover_observations = heavy.get("cover_observations")
    final_lead = heavy.get("final_lead")
    initial_backing = heavy.get("initial_backing")
    vocal_result_present = bool(heavy.get("vocal_result_present"))
    vocal_audit_for_resolution = heavy.get("vocal_audit_for_resolution")
    objective_report = heavy.get("_objective_report", "")   # now passed through

    singer_identity = ""
    # No singer to identify on a measured instrumental or unresolved presence.
    # Untagged files use the accepted acoustic category without a model lookup.
    if (ENABLE_SINGER_IDENTITY_RESOLUTION and VOCAL_PRESENCE_UNRESOLVED_MARKER not in revised and not heavy.get("instrumental")
            and (_has_real_metadata(metadata) or cover_observations or _scan_user_photo_given()
                 or _audio_only_singer_identity(vocal_audit_for_resolution))):
        status("Resolving singer identity from audio + metadata + cover art...")
        try:
            singer_identity = _resolve_singer_identity_with_photo(   # 0.8-pre34
                metadata,
                vocal_audit_for_resolution or "",
                cover_observations,
                {},
                wiki_context=_artist_background_for_identity(metadata),
            ) or ""
            if IDENTITY_CANONICAL_REPLY:                                  # 0.7.1-pre12
                singer_identity = _canonical_identity_reply(singer_identity)
            _si_last_good = singer_identity
            _identity_answer = singer_identity
            singer_identity = _reconcile_identity_with_section_voice(singer_identity, revised) or ""
            singer_identity = _young_solo_male_clips(
                _identity_answer, singer_identity, cover_observations, metadata,
                background=lambda: _artist_background_for_identity(metadata)) or ""
            singer_identity = _feature_credit_identity(singer_identity, metadata, cover_observations) or ""
            singer_identity = _young_male_voice_check(
                singer_identity, (vocal_audit_for_resolution or "") + "\n" + str(revised or ""),
                cover_observations, metadata,
                background=lambda: _artist_background_for_identity(metadata)) or ""
        except Exception as e:
            print(f"  (singer identity skipped: {e})")
            # 0.7.1-pre12 IDENTITY_KEEP_LAST_GOOD: a failing rule step keeps the model's answer.
            singer_identity = (locals().get("_si_last_good", "") or "") if IDENTITY_KEEP_LAST_GOOD else ""
        finally:
            if globals().get("BATCH_UNLOAD_OLLAMA", True) and _is_batch_context():
                try:
                    ollama_unload_model()
                except Exception:
                    pass
        cover_obs_block = (
            _format_cover_observation_block(cover_observations) if cover_observations else ""
        )
        if cover_obs_block and not _has_cover_observations(revised):
            revised += "\n\n" + cover_obs_block
        if SINGER_IDENTITY_TO_WRITER:
            if (singer_identity and _parse_singer_identity(singer_identity)
                    and not _has_identity_resolution_block(revised)):
                revised += _format_identity_resolution_block(singer_identity)
        elif singer_identity and "SINGER IDENTITY RESOLUTION" not in revised:
            revised += (
                f"\n\nSINGER IDENTITY RESOLUTION (combined audio + metadata + cover art; "
                f"use for who-is-singing questions about the PERSON, never to re-describe the sound):\n{singer_identity}"
            )
        resolved_tag = _parse_singer_identity(singer_identity) if singer_identity else ""
        priority_tag = _reconcile_identity_with_acoustic_age(resolved_tag, final_lead)
    # Acoustic vocal wording is emitted whether or not tags/cover exist, so the
    # same audio always gets the same measured description of the voice.
    _id_tag = (
        _parse_singer_identity(singer_identity)
        if (ENABLE_SINGER_IDENTITY_RESOLUTION and singer_identity) else ""
    )
    if _identity_overrides_gender(_id_tag, final_lead):
        # Mark the passes' own female/male tag as overruled so the writer
        # doesn't pick it back up from the raw vocal-profile text.
        revised = _mark_superseded_vocal_tags(revised, _id_tag)
    _unbroken = _VOICE_UNBROKEN_MARK in revised
    if vocal_result_present:
        revised += build_acoustic_vocal_note(
            final_lead or "unknown", initial_backing or "uncertain", identity_tag=_id_tag,
            unbroken=_unbroken,
        )
    if _id_tag:
        revised += build_singer_identity_note(_id_tag, final_lead, unbroken=_unbroken, vocal_text=revised)

    # ------------------------------------------------------------
    # FINAL STEP: force the structure block to be the very last thing
    # ------------------------------------------------------------
    # 1. Remove any existing RECOMMENDED STRUCTURE block (so we don't duplicate).
    # Matches all three header variants format_recommended_structure_block can
    # emit: "...DISCUSSION:", "...DISCUSSION (from objective change points):",
    # and "...DISCUSSION (coarse, evenly-spaced windows...):". The old pattern
    # only matched the first, so once the coarse fallback started actually
    # firing (see _resolve_total_duration) it would have been left in place
    # and duplicated below instead of replaced.
    revised = re.sub(
        r"\n\nRECOMMENDED STRUCTURE FOR DISCUSSION(?:\s*\([^)]*\))?:.*?(?=\n\n[A-Z]|\Z)",
        "",
        revised,
        flags=re.DOTALL,
    )

    # 2. Build a fresh structure block using the OBJECTIVE-FIRST pipeline
    track_path = heavy.get("track_path")
    total_duration = heavy.get("_video_audio_duration") or _resolve_total_duration(
        track_path=track_path, objective_report=objective_report
    )

    # The section map was already resolved objectively during the heavy stage,
    # while the audio model was still loaded (see run_fresh_track_analysis_heavy).
    # It is reused verbatim here.
    #
    # This block used to re-run the whole cut/cluster/label pipeline from
    # scratch at this point, which meant reloading the listening model immediately
    # after _release_heavy_analysis_memory_before_identity() had deliberately
    # freed it — paying a full model load to recompute a result the run already
    # had, and producing a section map that disagreed with the one the melody
    # and intensity blocks had already been keyed to.
    structure_sections = list(heavy.get("structure_sections") or [])
    boundary_seconds = []
    _lyric_groups_for_structure = None

    if structure_sections:
        boundary_seconds = [float(s["start"]) for s in structure_sections[1:]]
        revised = _overwrite_structure_field(revised, structure_sections)
        revised = _correct_stated_duration(revised, structure_sections[-1].get("end"))

    # Fall back only if the heavy stage genuinely produced nothing.
    if not structure_sections:
        structure_sections, boundary_seconds, _lyric_groups_for_structure = _finalize_structure_sections(
            revised,
            objective_report=objective_report,
            stem_midi_report=locals().get("stem_midi_report") or "",
            lyric_chunks=locals().get("_lyric_chunks"),
            total_duration=total_duration,
        )

    structure_block = format_recommended_structure_block(
        structure_sections,
        objective_boundaries=boundary_seconds,
        total_duration=total_duration,
        lyric_groups=_lyric_groups_for_structure,
        objective_report=objective_report,
    )
    if structure_block:
        revised = revised.rstrip() + "\n\n" + structure_block

    revised = _collapse_runaway_repetition_fields(revised)

    # Sonic embedding is computed HERE, at the end of the analysis, rather than
    # only inside save_song_data(). Previously a track only ever acquired an
    # embedding if it was saved, so a freshly analysed track could not take
    # part in a sonic-similarity comparison until it had been written to disk
    # and read back. This runs at the same point in the same process the save
    # would have run it (right after identity resolution, with the heavy
    # listening models already released), so peak memory is unchanged; the
    # result is cached and handed to save_song_data(), which reuses it instead
    # of recomputing.
    clap_vector = None
    try:
        _clap_src = heavy.get("track_path")
        if ENABLE_CLAP_EMBEDDING and _clap_src and not str(_clap_src).startswith(("http://", "https://")):
            status("Computing sonic-similarity embedding...")
            # dev55: computed in the background during the heavy stage (the
            # batch finish stage runs in another process, so it is handed over).
            clap_vector = heavy.get("clap_prefetched") or compute_clap_embedding(_clap_src)
            status_done(
                "Sonic embedding ready" if clap_vector is not None
                else "Sonic embedding unavailable"
            )
    except Exception as e:
        print(f"  (CLAP embedding skipped: {e})")
        clap_vector = None

    status_done("Analysis complete")

    out = dict(heavy)
    out["analysis"] = _normalize_unresolved_vocal_report(revised)
    try:
        out["analysis"] = _apply_moss_genre_clarifications(
            out["analysis"], out.pop("_genre_clarifications", {}),
            out.pop("_genre_clarification_panns", []))
    except Exception as e:
        print(f"  (genre clarification not applied: {type(e).__name__}: {e})")
    out["analysis"] = _deduplicate_final_genre_ranking(out["analysis"])
    out["singer_identity"] = singer_identity
    out["structure_sections"] = structure_sections
    out["clap_embedding"] = clap_vector or []
    # clean up temporary keys
    for k in ("_objective_report", "vocal_audit_for_resolution", "final_lead",
              "initial_backing", "vocal_result_present", "instrumental", "clap_prefetched",
              "moss_first_pass", "_video_audio_duration"):
        out.pop(k, None)
    return out


def resolve_identity_and_finalize(*args, **kwargs):
    """SINGER_IDENTITY_PRE6 picks the 0.7.1-pre6 version; False gives 0.7's."""
    return (resolve_identity_and_finalize__pre6 if SINGER_IDENTITY_PRE6 else resolve_identity_and_finalize__07)(*args, **kwargs)
