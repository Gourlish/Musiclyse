# MUSICLYSE FILE pipeline-listening 0.8
# ==============================================================================
#  MUSICLYSE 0.8: THE LISTENING MODEL
# ==============================================================================
# This file holds the listening model (MOSS-Music): loading it, every prompt sent to it, its
# passes and the clean-up of its answers. It is part of musiclyse-0.8.py, which loads it whole
# at start-up, after its own code, into its own namespace (see "THE MUSICLYSE FILES" there); it
# is not run on its own.
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

# (VOCAL_CONFIRMATION_F0_THRESHOLD moved to the settings file beside this script.)
# (VOCAL_CONFIRMATION_WITHOUT_F0 moved to the settings file beside this script.)
# Soft pitch constraints for lead gender/age (objective median f0 from isolated
# stem when available). Pitch alone never *proves* gender/age — high adult
# male head voice / falsetto and low female altos exist — but a track-wide
# median can veto overconfident categories that are acoustically implausible.
# Values are Hz; set a bound to None to disable that rule.
# (F0_MALE_HIGH_CONFIRM_HZ moved to the settings file beside this script.)
# (F0_POST_PUBERTY_SOFT_CAP_HZ moved to the settings file beside this script.)
# (F0_POST_PUBERTY_HARD_CAP_HZ moved to the settings file beside this script.)
# (F0_MALE_ANY_HARD_CAP_HZ moved to the settings file beside this script.)
# (F0_CHILD_SOFT_FLOOR_HZ moved to the settings file beside this script.)


MOSS_FULL_ANALYSIS_PROMPT = """Analyze this track as a careful audio/music analyst and return a compact note-style report only.

Your job at this stage is EVIDENCE COLLECTION AND MUSICAL INTERPRETATION.

Do not write polished prose.
Do not try to impress the reader.
Do not fill gaps with genre expectations, artist knowledge, album knowledge, Wikipedia/background
database facts, or what a song of this type "usually" sounds like. Instrumentation, vocals, tempo,
key, structure, and production claims must come from THIS recording only — never from knowledge of
the credited artist or similar releases.

For every claim, distinguish internally between:

- DIRECT: clearly audible or directly measurable
- STRONG INFERENCE: supported by multiple independent musical cues
- WEAK/AMBIGUOUS: plausible but not sufficiently established

When evidence is ambiguous, say "uncertain", "approx.", or provide a small number of plausible alternatives.

Never manufacture precision.

Use exactly these labels:

GENRE_RANKED=1) [descriptor] (confidence: high/medium/low); 2) ...; 3) ...
  Rank only distinct genres supported by the recording, up to eight. Stop
  after the last supported label. If only a broad genre is supported, give
  that one label. If the genre is unclear, say uncertain.
GENRE_ADJACENT=[short descriptors separated by semicolons]
GENRE_RULED_OUT=[categories clearly unsupported, if any]

Use unambiguous genre names in all three genre fields. Never use "indie",
"alternative" or "garage" as a complete genre label: qualify the intended
musical style using evidence from the recording. Do not assume a rock meaning
for an ambiguous term. If the specific style is unsupported, use a supported
broad genre such as rock or pop, or say uncertain. This naming rule does not
add candidates or change their order; rank solely by the audio evidence.

KEY=[best-supported tonal centre plus major/minor/mode; base this on recurring pitch/chord/harmonic evidence. Consider bass movement, chord resolutions, melodic resting points, phrase endings and recurring pitch classes. Do not infer key from genre. If harmonic evidence is weak, modal, static or ambiguous, say "uncertain" or provide two close candidates.]

TEMPO_BPM=[best estimate of the musical pulse in BPM. Consider beat tracking, bar structure, kick/snare relationship and perceived pulse. If half-time/double-time interpretations are plausible, mention the alternative only when musically meaningful.]

CHORDS=[compact section-by-section harmonic description. Use exact chord names only when reasonably supported. If exact voicings are uncertain, use functional descriptions. Keep this field SHORT. Summarise recurring patterns (e.g. "A6–D loop throughout verses/choruses") rather than listing the same progression dozens of times. Never repeat the same chord sequence more than twice in a row.]

STRUCTURE=[Intro 0:00-0:12; Verse 1 0:12-0:34; Chorus 0:34-0:58; ...]
Every section MUST have a timestamp.
Use approximate boundaries where necessary.
Use consistent names for recurring sections.
Do not invent a bridge/pre-chorus/outro merely because the arrangement changes.

IMPORTANT STRUCTURE RULES:
- For instrumental or non‑standard songs, label sections by musical function (e.g. "Intro", "Main Theme", "B Section", "Development", "Outro") or sequentially (e.g. "Section 1", "Section 2") if no clear repeats exist.
- If the piece is through‑composed with no repeated sections, number the sections in order and do not force a verse‑chorus structure.
- Timestamps must be precise (in `M:SS` or `MM:SS` format) and must be monotonically increasing, covering the entire duration from start to end. There should be no gaps between sections unless a true silence or transition is audible.
- Do not leave any section without a timestamp.
- TIMING TIP: If an "OBJECTIVE AUDIO MEASUREMENTS" report is provided alongside this audio and it lists "STRUCTURAL BOUNDARY CANDIDATES" (measured timbre/harmony change points), use the closest listed timestamp for a section boundary you hear near it, instead of estimating that one boundary yourself. For any boundary with no nearby candidate listed, give your own best M:SS estimate as usual. Every section boundary must still be a concrete M:SS timestamp in the STRUCTURE field — never omit one, hedge it into prose, or drop it for being imprecise.

INSTRUMENTATION=[source 1; source 2; ...]
For each identifiable source describe:
- likely instrument/source
- acoustic/electric/electronic character where relevant
- musical role
- distinctive technique or texture where audible

A stem label, spectral peak or MIDI transcription is NOT proof of an instrument identity.
If an "instrument tag (independent audio classifier)" line appears for a stem, treat it the
same way — as supporting evidence to weigh alongside what you hear, not as proof on its own;
it can mislabel timbrally similar instruments (e.g. synth brass vs real brass, bright synth vs guitar).
WEAK / STEM-ONLY / BELOW-THRESHOLD tags must NOT become instrument claims — prefer "no clear guitar"
or a texture description ("bright midrange pluck/pad") over naming guitar/piano/strings from a weak tag.
If two sources are plausible, say "likely X or Y".
If identity is uncertain, describe the sound rather than inventing the instrument.
Do not invent guitar, piano, or strings solely because the genre usually has them.

DRUMS=[kick pattern; snare placement; hat density/openness; swing vs straight; fill density; room/dry character]
Be specific from what is audible (and from any GROOVE_HINT / kick beat-grid analysis / snare beat-grid
analysis / swing-shuffle analysis / per-type rhythm lines in the stem report).
When asked what the drums do at a particular point — a named section, a timestamp, a range, when they
come in, whether they drop out — answer from the DRUM ARRANGEMENT TIMELINE and the per-section groove
blocks, never from the whole-track pattern sample or the whole-track per-type rates, which are averages
across the entire song and say nothing about any one moment. The timeline is computed from the actual
hit timestamps, so it is authoritative about when each part of the kit is and is not playing; a span it
marks "no drums" really has no transcribed hits, and a span listing a partial kit really is missing the
others there. Do not smooth those changes into a single description of "the groove".
The drum transcription resolves ONLY kick, snare and hi-hat. Toms, crashes, rides, shakers, claps and
other percussion cannot be detected by it, so never state or imply that a track lacks them because they
are absent from the hit breakdown, and do not present those three as the complete kit. An "ADDITIONAL
PERCUSSION DETECTED" block, when present, is the only evidence in these notes about those pieces — name
what it lists (cymbals as crash or ride, tambourine, shaker, and so on) as real parts of the kit, using
its coverage note for how much of the track they cover, but do not invent hit counts or placements for
them, since they are classifier detections rather than counted hits.
A BEAT GRID block, when present, is a pre-aligned fixed-width diagram of the measured pattern. If asked for the beat in diagram form, reproduce it verbatim in a code block rather than drawing your own — re-typing it breaks the column alignment and misstates the rhythm.
For how the kit SOUNDS rather than what it plays — kick weight and tuning, snare decay and room, hat
character, bus compression, whether hits read as programmed samples or a played kit — use the drum-sound
lines (kick sound / snare-backbeat sound / hi-hat sound / kit character / hit-to-hit consistency). Those
are measured from the separated drums audio, so they are the authority on drum tone and production;
never answer a question about how the drums sound from genre expectation when they are present, and do
not contradict them (do not call a kit roomy when the decay figures say gated and dry).
Forbidden generic filler unless it truly matches the groove: "driving drums", "tight backbeat",
"punchy kick", "solid groove" with no further detail.
"four-on-the-floor" specifically describes a kick on every beat. Only use that phrase when the stem
report's kick beat-grid analysis actually reports four-on-the-floor (including "mostly four-on-the-floor").
A merely fast or busy kick, a dance-pop genre guess, or overall energy is NOT four-on-the-floor — name the
grid the analysis reports (every other beat, eighth-note, on-beat, moderate non-FOTF, syncopated) or say
"programmed kick pulse" without FOTF.
"backbeat" specifically means the snare falls opposite the kick (e.g. 2 & 4 against a kick on 1 & 3).
Only call it a classic backbeat when the snare beat-grid analysis reports that (including "mostly classic
backbeat"); a snare that merely lands "every other beat" at roughly the right rate can instead be doubling
the kick or on a backbeat-rate grid without confirmed opposite phase — say so instead of defaulting to
"backbeat". A "half-time backbeat" (snare once per bar, on beat 3) is a half-time feel: describe it as
half-time, not as a 2-and-4 backbeat at the stated tempo.
When a kick or snare beat-grid line already names a family (every-other-beat or on-beat kick, eighth-note,
classic backbeat, half-time backbeat, backbeat-rate, doubling), use that family in DRUMS= — do not replace it with a vague spacing
story ("every ~0.6s", "textured pulse") that ignores the named grid.
"swung" / "shuffled" / "triplet feel" should only be used when the swing/shuffle analysis reports it.
Absent that line, or when it reports "straight", describe the subdivisions as straight/even rather than
guessing a swing feel from genre expectations.
If a kick/snare beat-grid line says "mostly … (moderate grid-lock)" or "not irregular", treat that as a
real on-beat / every-other-beat / backbeat-family pattern for a programmed kit — do NOT paraphrase it as loose,
wandering, messy, or off-grid. Reserve "irregular" / "wandering" only when the analysis explicitly says
syncopated, off-grid, or free-time irregular.
Prefer concrete language, e.g. "four-on-the-floor kick, classic backbeat snare on 2/4, busy closed 16th
hats with a light swing, dry kit" or "sparse kick on every other beat, snare on a backbeat-rate grid,
straight hi-hats, little continuous hat bed" — but only when those grid labels are actually present.

TIMBRE=[compact description of audible texture and production: brightness/darkness, density, stereo image, ambience, reverb, saturation/distortion, transient character, layering, separation, dynamics, etc.]

MOOD_VIBE=1) [mood]; 2) [mood]; 3) [mood]

MELODICISM=[melodic density, contour, repetition, hook strength, vocal/instrumental range impression, phrase structure]
MELODIC_CONTOUR=[brief description, in your own words, of how the main melody moves in this track]
MELODIC_RANGE=[approx. lowest and highest note of the main melody in this track, as note names]

VOCALS_PRESENT=yes/no/uncertain
  (yes if a person sings, raps or speaks words anywhere in the track, even only in some sections or with
  hard-to-make-out words; no only if there is no human voice at all; uncertain only if you cannot tell
  whether a sound is a human voice. Long instrumental passages do not make it uncertain.)

LEAD_VOCAL_CHARACTERISTICS=[acoustic description only: register, pitch behaviour, timbre, vocal weight, resonance, articulation, delivery, vibrato, breathiness, processing, etc.]

LYRICS_PRESENT=yes/no/uncertain

LYRIC_SUBJECT=[brief subject/theme/tone based only on clearly intelligible lyrical material]

DESCRIPTION=[a few plain sentences describing the track as a listener hears it: its sound, instruments, voice and mood]

ACCURACY RULES:

1. AUDIO FIRST

Describe what is actually supported by this recording.

Do not substitute:
- artist expectations
- genre conventions
- release-year assumptions
- cover-art assumptions
- remembered lyrics
- expected instrumentation

for audio evidence.

2. OBSERVATION BEFORE INTERPRETATION

Prefer:

"bright, layered electric-guitar texture"

over:

"characteristic early-1990s college-rock guitars"

The first is an observation.
The second is interpretation and requires broader evidence.

3. DO NOT COLLAPSE AMBIGUITY

If evidence supports X or Y, report X/Y rather than choosing the answer that sounds most plausible.

4. DO NOT CREATE FALSE PRECISION

Do not invent:
- exact chords
- exact timestamps
- exact BPM
- exact instrument identities
- exact vocal ranges
- exact structural boundaries

when the recording does not support that precision.

5. GENRE

Genre should be based on several independent musical characteristics:
- rhythm/groove (especially drum programming vs live kit feel)
- instrumentation (dominant sources, not faint background layers)
- harmony
- vocal approach
- arrangement
- production / mix (synth beds, sidechain, four-on-the-floor, club loudness vs garage band dynamics)
- overall musical language

One instrument or one production characteristic must not determine the genre.

Critical anti-bias rules:
- A subtle, quiet, or occasional guitar does NOT make a track rock, pop-punk, or indie rock if the dominant language is electronic/dance/synth-pop (programmed drums, synth bass, four-on-the-floor, sidechain pump, club-oriented production).
- Conversely, a single synth pad does NOT make a guitar-driven rock song "electronic".
- Weight the PRIMARY rhythmic and production identity over secondary texture layers.
- Prefer broader, higher-level labels (electronic pop, dance-pop, synth-pop, house-influenced pop) when subgenre evidence is thin.
- Do not leap to scene-specific labels (pop-punk, emo, post-punk) from guitar timbre alone.

Genre is a description of sonic similarity, not proof of release era, scene membership, influence, or artist identity.

6. ERA

Do not infer a decade merely from:
- guitar style
- brightness
- polish
- dynamic range
- general nostalgia
- genre

If production-era clues are weak, report a broad range or uncertainty.

7. KEY

A detector estimate is evidence, not truth.

Use:
- bass
- chord movement
- melody
- cadences
- phrase endings
- recurring pitch classes

If major/minor or relative-key ambiguity remains, preserve it.

8. TEMPO

Distinguish musical pulse from subdivisions and half/double-time interpretations.

Do not report a detector number merely because it is precise.

9. STRUCTURE

Structure should follow meaningful changes in:
- arrangement
- harmony
- melody
- lyrics
- rhythm
- energy

Do not force every contrasting passage into a named pop-song section.

For instrumental or through‑composed music, use descriptive names like "Theme A", "Section 1", "Development", "Bridge", etc., based on what actually changes.

10. INSTRUMENTS

Spectral evidence may identify characteristics such as brightness, density or frequency range, but does not by itself prove instrument identity.

Describe uncertain sources conservatively.

11. VOCALS

Pitch is NOT age.

Brightness is NOT age.

Thinness is NOT age.

Performer age is NOT vocal age.

Do not infer vocal age from:
- the singer's current age
- the singer's age at release
- the singer's age at recording
- debut age
- public image or appearance

A singer's known age may be mentioned as background context, but it must not be used as acoustic evidence.

Classify the voice only from audible vocal characteristics such as:
- vocal resonance
- vocal weight
- vocal tract characteristics
- maturity of vocal production
- consistency across the performance

High register is NOT age.

Youthful-sounding delivery is NOT age.

Falsetto/head voice is NOT age.

Androgynous tone is NOT age.

A child/prepubertal classification requires actual acoustic evidence consistent with a prepubertal vocal tract/resonance profile.

Do not classify a high adult male voice as a child.

Do not classify a high adolescent/adult female voice as a child.

12. VOCAL IDENTITY

Do not identify the singer from lyrics.

Do not infer gender from lyrical subject matter.

Do not infer age/gender from artist stereotypes.

Do not count doubled vocals, reverb, octave effects or harmonies as separate singers unless distinct voices are genuinely established.

When multiple human voices are present, separate:
- one lead + backing/harmony
- true co-leads / duet / call-and-response
- group unison texture
Only claim multiple lead singers when timbre, range, or sectional roles clearly differ.

13. LYRICS

Do not use expected lyrics to repair unclear words.

The separate lyric transcription is a rough draft and is not automatically authoritative.

Do not invent biographies, contact details, URLs, copyright notices, or long
keyword lists in any lyrics-related field. If a lyric draft degenerates into
spam or letter/phrase loops, stop at the last coherent sung line.

If the lyric transcription contains a "[UNVERIFIED\u2192 ... ]" span, two
independent decodes of the same audio disagreed on that wording. Do not
quote, paraphrase closely, or otherwise present that span's wording as what
the song says. Either omit it, describe the general topic in your own words
without claiming specific phrasing, or say plainly that the exact words of
that section are unclear. The same applies to any "[TRANSCRIPTION CAUTION]"
note about repeated-section drift — treat the flagged occurrence as unverified.

14. CONFIDENCE

Confidence represents evidence quality, not how easy the answer feels.

A plausible interpretation may still have low confidence.

15. NEGATIVES

Do not use GENRE_RULED_OUT merely because a category is not obvious.

Only rule something out when the recording provides meaningful evidence against it.

16. INTERNAL MEASUREMENTS

Objective measurements, stems and MIDI are evidence used to improve accuracy.

Do not optimise this pass for producing numbers that sound impressive.

The measurements should help establish musical conclusions, not replace them.

Return only the complete compact note-style analysis."""
 

MOSS_DEEP_MODE_ADDENDUM = """
Additionally add these compact fields:

PRODUCTION_NOTES=[specific audible production/arrangement choices with approximate timing where useful; distinguish clearly audible choices from interpretation]

REFERENCE_POINTS=[only concrete sonic similarities or useful comparison points; describe the shared musical characteristic rather than asserting influence, scene membership, or historical connection]

DEEP-MODE RULES:

- Additional detail must increase accuracy, not merely increase the amount of text.
- Prefer a well-supported broad observation to an impressive but speculative technical claim.
- Do not convert spectral, MIDI or stem evidence into unsupported instrument identities.
- Do not infer recording technology, microphones, consoles, tape/digital format, mixing equipment or studio practices unless directly supported.
"""


STYLE_EVIDENCE_FIREWALL = """
STYLE EVIDENCE FIREWALL:
When describing the style of the specific track being analysed, use only:
- audible evidence from this recording
- private track analysis notes grounded in the recording

Do not use:
- Wikipedia/background database information
- artist reputation
- the artist's usual genre
- previous albums
- assumed influences
- remembered production characteristics
- parametric knowledge of "how this artist typically sounds"

Those sources may still be used for artist, album, historical, discography, reception,
and general background questions — never to invent or override instruments, vocals,
tempo, key, structure, mix, or vocal age/gender for THIS recording.

If a statement about THIS SONG sounds like it came from artist knowledge rather than the
recording itself, rewrite it as neutral background context or remove it.
"""

SELF_CHECK_PROMPT = """Review the compact note-style analysis you just produced field by field.

This is an ACCURACY AUDIT, not a request to rewrite the analysis into prose.

Keep the same compact labels and format.

Your goals are to detect:

- unsupported claims
- false precision
- contradictions
- accidental hallucinations
- incorrect instrument identification
- incorrect tempo interpretation
- unsupported genre/era assumptions
- lyric hallucination
- vocal age/gender errors
- conclusions that are weaker than the available evidence

Do not remove useful confident detail merely because it is technical.

GENERAL CALIBRATION:

For each claim ask:

1. Is it directly supported?
2. If not, is it supported by multiple independent cues?
3. Is the wording more specific than the evidence?
4. Did an inference become stated as fact?
5. Did a numerical measurement get treated as a musical conclusion without checking it?
6. Did two fields contradict each other?
7. Did I use genre, artist identity, artwork or release-year expectations as evidence?
8. Did I infer vocal age from pitch or timbre?

TEMPO INTERPRETATION — IMPORTANT:

When the analysis provides both a base tempo and a 2x/half-time alternative, do not automatically choose the larger BPM.

Determine which value represents the song's PRIMARY MUSICAL PULSE.

Use:
- the apparent kick/snare cycle
- phrase rhythm
- harmonic rhythm
- how a listener naturally counts the beat
- the relationship between beat and bar
- the perceived pace of the song

A 2x BPM is often a subdivision/double-time representation of the same underlying pulse.

If the evidence does not clearly establish the faster interpretation, prefer the lower/base pulse and describe the alternative internally rather than presenting it as the main tempo.

Never allow a possible double-time detector result to transform a relaxed/mid-tempo song into a fast song without strong supporting evidence.

If TEMPO_BPM contains a value near exactly 2x another plausible value, explicitly test whether the larger value is merely a double-time interpretation.

Do not accept the 2x value as the primary tempo solely because it has a confident numerical estimate.

CORRECTION PRINCIPLE:

When appropriate:

exact -> approximate
specific -> broader
certain -> likely
likely -> possible
unsupported -> uncertain

But do NOT automatically weaken a claim if another independent source supports it.

--------------------------------------------------
CONFLICT RESOLUTION
--------------------------------------------------

When two analysis fields disagree, do NOT simply report the contradiction.

First determine which evidence is stronger.

Examples:

- A generic "no vocals detected" flag is weaker than a coherent pitched vocal melody plus a plausible vocal-stem transcription.
- A single automatic key estimate is weaker than consistent bass, chord, melodic and cadence evidence.
- An isolated stem label is weaker than repeated audible characteristics across the full mix and other evidence.
- A rough lyric transcription is weaker than clearly supplied verified lyrics.
- A cover-art impression is weaker than confirmed metadata for release identity.
- A general genre prediction is weaker than multiple track-specific musical observations.

If stronger evidence resolves the conflict:
use the resolved conclusion.

If it does not:
retain the uncertainty.

Do NOT explain the conflict by mentioning internal pipeline stages, model names or missing blocks.

--------------------------------------------------
TEMPO
--------------------------------------------------

Prefer the musical pulse.

Check:
- beat spacing
- bar structure
- kick/snare relationship
- repeated rhythmic pattern
- perceived groove

If half/double-time is plausible, choose the interpretation that best matches the musical pulse.

Do not report several nearly identical detector values as meaningful disagreement.

--------------------------------------------------
KEY / HARMONY
--------------------------------------------------

Recheck tonal centre using:
- bass
- chord movement
- melody
- phrase endings
- cadences
- recurring pitch classes

Do not infer key from genre.

Do not mistake a bass note for the complete chord.

If harmonic evidence remains genuinely ambiguous, preserve the ambiguity.

--------------------------------------------------
CHORDS
--------------------------------------------------

Do not invent exact chord names from vague harmonic colour.

Use broad harmonic descriptions when necessary.

Do not assume every detected pitch belongs to a chord.

--------------------------------------------------
STRUCTURE — HIGH PRIORITY
--------------------------------------------------

Re‑examine the STRUCTURE field carefully.

Check that:
- Every section has a timestamp (start and end) in M:SS format.
- Timestamps are in chronological order and do not overlap.
- The sections cover the entire track from start to end without gaps (unless a true silence or transition is audible, in which case a section like "Interlude" or "Break" may be inserted).
- Section names are consistent: if you have "Verse 1" and "Verse 2", they should be similar in function; if they differ greatly, they may actually be different sections.
- You have not invented a bridge, pre‑chorus, or outro unless there is a clear musical change that distinguishes it.
- For instrumental or through‑composed music, you have used neutral labels like "Section A", "Theme 1", "Development", etc., and have not forced a verse‑chorus structure.
- If the song has no clear repeated sections, you have numbered sections sequentially (e.g., "Section 1", "Section 2") rather than inventing pop labels.
- If a boundary is uncertain, you have indicated it as approximate (e.g., "Verse 1 0:12-0:34 (approx.)").

If any of these conditions are violated, revise the STRUCTURE field to correct them. Do not leave structural ambiguities unresolved.

The CORRECTION PRINCIPLE above ("exact -> approximate") governs wording confidence, not the STRUCTURE field's timestamps themselves: every section must still keep a concrete M:SS start and end. If a boundary is genuinely uncertain, mark it "(approx.)" next to its timestamp rather than removing the timestamp or describing timing only in words. If the accompanying OBJECTIVE AUDIO MEASUREMENTS report lists STRUCTURAL BOUNDARY CANDIDATES, use the closest one for any boundary you're unsure of instead of vaguing it out.

--------------------------------------------------
GENERATION QUALITY / REPETITION
--------------------------------------------------

If any field (especially CHORDS) contains the same short sequence repeated many times, collapse it to a compact summary.
Example of failure: "A6 → D → A6 → D → A6 → D → ..." repeated for hundreds of characters.
Correct form: "A6–D loop throughout most of the track" or a short section-by-section summary.

Do not allow runaway repetition to remain in the revised analysis.

--------------------------------------------------
FALSE INSTRUMENTAL / MISSED VOCALS
--------------------------------------------------

If the full mix contains a coherent human-sung melody (formants, syllabic contour, sustained pitched phrases that sound vocal), do not leave the analysis claiming the track is instrumental or VOCALS_PRESENT=no/uncertain solely because lyrics are hard to parse.
Prefer VOCALS_PRESENT=yes + uncertain demographics when voice is present but age/gender is unclear.

--------------------------------------------------
INSTRUMENTATION
--------------------------------------------------

A stem label is not proof.

A spectral peak is not proof.

A MIDI transcription is not proof.

An independent audio-classifier tag is not proof — especially guitar-family tags,
which often fire on bright synths, distorted bass, or Demucs residuals.

If a tag is marked weak / stem-only / dropped, or is absent from WHOLE-MIX INSTRUMENT TAGS,
do not promote that instrument into a confident claim.

If several instruments could plausibly produce the sound, retain the ambiguity.

Prefer:
"likely electric guitar or keyboard"
or "bright midrange layer — guitar and synth both plausible"
or "no clear guitar"

over an unsupported definitive identification.

Omit instruments that are only weakly suggested. Genre expectation is not evidence.

--------------------------------------------------
DRUMS / GROOVE
--------------------------------------------------

If a DRUMS field or GROOVE_HINT / kick beat-grid analysis / snare beat-grid analysis / swing-shuffle
analysis / per-type rhythm lines exist, use them.

Reject purely generic drum language ("driving drums", "tight backbeat", "punchy kit")
when more specific pattern evidence is available — replace with concrete kick/snare/hat detail.

The kick beat-grid, snare beat-grid, and swing/shuffle analyses (when present) are measured against
the track's actual tempo and inter-onset timing, so they are the authority on whether the kick is
genuinely four-on-the-floor / every other beat / eighth-note / syncopated, whether the snare is a genuine
opposite-phase backbeat vs merely on a similar-rate grid, and whether the subdivisions are actually
swung vs straight — do not override any of them with a guess based on genre or overall busyness.
If the kick beat-grid does not explicitly report four-on-the-floor (or "mostly four-on-the-floor"),
remove any four-on-the-floor claim from DRUMS= and from elsewhere in the analysis — including
dance-pop / energy flavor text. Prefer the named grid family over vague spacing paraphrases
("every ~0.6s", "textured pulse") when a kick/snare beat-grid line already names every-other-beat, half-time backbeat, on-beat,
eighth-note, classic backbeat, or backbeat-rate.

If a "Per-section groove" block is present (grouped by this track's own STRUCTURE sections, e.g.
Verse/Chorus), use those SECTION GROOVE lines — not the whole-track pattern — when the user asks
about the beat/groove in a specific section, or when contrasting sections (e.g. "does the chorus
hit harder than the verse"). Point out real differences between sections when they exist (kick grid
change, snare grid change, swing feel change, hat density change, more toms/cymbals) rather than
assuming the groove is uniform throughout.

If drum evidence is thin, say so rather than inventing a stock groove description.

--------------------------------------------------
VOCALS — HIGH PRIORITY
--------------------------------------------------

Pitch is not age.

Brightness is not age.

High register is not age.

Thinness is not age.

Falsetto/head voice is not age.

Youthful delivery is not age.

Androgynous tone is not age.

Child/prepubertal classification requires actual evidence consistent with a prepubertal vocal tract/resonance profile.

A high adult male voice must not be converted into a child classification.

A high adolescent/adult female voice must not be converted into a child classification.

If the evidence cannot reliably distinguish:
choose uncertain.

Do not use:
- child-like
- boyish
- girlish
- young voice
- juvenile
- youthful voice

as substitutes for an actual classification.

Do not infer gender from lyrics.

Do not count harmonies/doubling as co-leads.

--------------------------------------------------
GENRE
--------------------------------------------------

Keep GENRE_RANKED broad when evidence is mixed.

Do not introduce a new genre merely because it sounds stylistically attractive.

Do not use artist identity, cover art or release year as proof of genre.

In GENRE_RANKED (or "Genre Ranked"), keep only distinct labels supported
by the recording, up to eight. Remove repeated entries and stop when the
evidence runs out. A single supported broad label is enough.

Use unambiguous genre names in all three genre fields. Never use "indie",
"alternative" or "garage" as a complete genre label: qualify the intended
musical style using evidence from the recording. Do not assume a rock meaning
for an ambiguous term. If the specific style is unsupported, use a supported
broad genre such as rock or pop, or say uncertain. This naming rule does not
add candidates or change their order; rank solely by the audio evidence.

If an independent OBJECTIVE GENRE/MOOD SIGNAL (PANNs / AudioSet) is provided in this
self-check context, treat it as a real cross-check — not as optional colour:

- When that signal clearly points to electronic / dance / house / techno / EDM /
  synth-pop / pop and GENRE_RANKED leads with rock / pop-punk / punk / emo /
  indie-rock mainly because of guitar-like texture, REVISE GENRE_RANKED so the
  top ranks reflect the electronic/dance/pop identity. Move rock/punk labels
  down or into GENRE_ADJACENT / GENRE_RULED_OUT as appropriate.
- One quiet or intermittent guitar layer must not keep a dance/electronic track
  ranked as pop-punk or rock at position 1.
- When the independent signal and the audio evidence agree, keep the ranking.
- When genuinely mixed, put the broader production-led label first and list the
  secondary flavour second with lower confidence.

--------------------------------------------------
ERA
--------------------------------------------------

Do not call something "80s", "90s", "modern", etc. merely because it feels that way.

Use concrete production evidence.

If the evidence cannot meaningfully distinguish eras, preserve that limitation.

--------------------------------------------------
PRODUCTION ERA SANITY CHECK
--------------------------------------------------

Do not infer a recording era solely from:
- stereo width
- clean separation
- brightness
- high fidelity
- dynamic range
- low compression

These characteristics are not era-specific.

If the analysis describes such characteristics, retain them as production
observations but do not convert them into a "modern", "contemporary" or
otherwise specific era claim without independent supporting evidence.

--------------------------------------------------
LYRICS
--------------------------------------------------

Do not repair unclear lyrics based on rhyme, semantics or remembered lyrics.

Do not manufacture quotations.

If any lyric-related text contains spam, biographies, contact details, URLs,
Wikipedia-style prose, copyright boilerplate, long unrelated keyword lists,
letter-run loops (e.g. tttttttt), or self-promotional filler after the actual
sung words, DELETE that material. Keep only plausible transcribed lyric lines
and short [inaudible] markers. Prefer truncating at the first spam onset over
leaving contaminated text in the analysis.

--------------------------------------------------
STYLE EVIDENCE CHECK
--------------------------------------------------

Before finalising any discussion of this specific song:

- Make sure claims about the song's sound come from the audio analysis.
- Do not justify sonic traits using the artist's reputation, genre history, previous albums, or Wikipedia background.
- Credits and contextual information (producer, songwriter, artist history, influences, album context) are allowed.
- If a sentence mixes audio observation and background knowledge, make the distinction clear.

--------------------------------------------------
FINAL AUDIT
--------------------------------------------------

Before returning the revised analysis, check:

1. Have contradictions been resolved where possible?
2. Have genuine uncertainties been preserved?
3. Have false specifics been removed?
4. Has vocal age been kept separate from pitch?
5. Have instrument identities been properly calibrated?
6. Have genre and era been kept separate from artist/context assumptions?
7. Have lyrics remained evidence-based and free of spam/bio/URL contamination?
8. Are all timestamps still present?
9. Is the analysis still informative rather than excessively vague?
10. Has runaway repetition in any field (especially CHORDS or lyrics) been collapsed?
11. Has the STRUCTURE field been verified for completeness, consistency, and appropriate labelling (no forced pop sections for instrumental / through‑composed music)?

Return only the revised compact note-style analysis."""
 

ERA_ANALYSIS_PROMPT_LONG = """Analyze the recording for production-era clues.

Your task is to estimate the likely recording/release era from the AUDIO itself, not to identify the artist or use genre stereotypes.

Focus on concrete audible production characteristics such as:
- recording noise/floor
- stereo image and panning conventions
- room/plate/digital reverb characteristics
- dynamic range and compression behaviour
- limiting/mastering character
- transient shaping
- frequency balance
- tape-like saturation or degradation
- audible digital processing
- pitch-correction artifacts
- drum-machine or sequencing characteristics
- editing/arrangement conventions when genuinely audible

Do NOT treat these as decisive by themselves:
- bright production
- polished production
- guitar-based music
- synth presence
- "indie" feel
- general nostalgia
- genre
- clean separation
- wide stereo
- high fidelity
- open dynamics

Modern productions can imitate older production.
Older recordings can be remastered.
Clean, bright, well-separated mixes exist across many decades (including the 1980s).
Therefore, production style alone may only establish a broad range.

First identify the strongest concrete markers.
Then state what those markers can and cannot establish.

If the evidence does not meaningfully distinguish decades, say uncertain rather than forcing a decade.
When markers are generic, give a broad range or say uncertain — but choose that range from THIS recording's own evidence. Do not reach for a default vintage-sounding span. A wide range that happens to exclude the present is still a decade claim, and it is the wrong one for most material.

RECENT PRODUCTION IS AN ANSWER, NOT A LAST RESORT.
The markers below genuinely do indicate recent production when you can actually hear them, and you should say so with normal confidence rather than retreating to a vintage range:
- heavy bus limiting and very high average loudness with squashed transients (loudness-war and streaming-era mastering)
- audible pitch correction, especially hard/quantised correction or deliberate tuning artefacts
- sidechain ducking pumping the mix against the kick
- programmed hi-hat rolls with fine subdivisions and velocity ramps, trap-style
- obviously sample-based or software-instrument drums with no room bleed
- surgical, artefact-free source separation and edits that analogue workflows could not produce
Absence of tape noise, tape saturation, or analogue degradation is also evidence, not neutral.

Never use the words "modern", "contemporary", or "current" solely because the mix is clean, bright, wide, or dynamic — but DO conclude recent production when the concrete markers above are present. The caution against those words is about weak evidence, not a ban on the conclusion.

If a SUPPLEMENTARY MEASURED EVIDENCE block appears below, it contains real measurements taken from this specific file (loudness, dynamic range, frequency bandwidth, and/or file-tag clues) that a single listen cannot reliably judge by ear. Weigh it as genuine evidence alongside what you hear — but exactly as hedged as it is presented: none of those numbers alone fixes a decade, and file-tag clues (encoder strings, ISRC) date the file or its registration, not necessarily the recording. If no such block appears, reason from listening alone as before.

Output exactly:

ERA_ESTIMATE=[likely decade or broad range, or uncertain]
CONFIDENCE=low|medium|high
EVIDENCE=[specific audible production markers]
MISSING_MARKERS=[important evidence that is unavailable or inconclusive]
"""


VOCAL_ANALYSIS_PROMPT_LONG = """Listen specifically to the human voice(s) in this track, if any.

Your task is a careful acoustic vocal profile.

Do NOT write a biography.
Do NOT identify the singer from lyrics.
Do NOT infer age from pitch alone.
Do NOT let genre, cover art or artist expectations determine the acoustic classification.
Do NOT use Wikipedia, artist biography, discography, or any parametric knowledge of who the
performer "usually" is or what their other records sound like. Classify only from this audio.

--------------------------------------------------
VOCAL DETECTION
--------------------------------------------------

First determine whether a human vocal is actually present.

Consider:
- intelligible sung speech
- sustained pitched phrases with syllabic contour
- repeated melodic patterns that sound sung rather than purely instrumental
- vocal formants and vowel/consonant structure
- coherent activity consistent with a human voice in the mix

Prefer VOCALS_PRESENT=yes whenever a coherent human-sung melody is audible in the full mix, even if individual words are hard to parse.
A generic "no vocals detected" result must not override clear evidence of a coherent sung melody.
Do not classify the track as instrumental / no vocals merely because the lyrics are hard to understand, the voice is processed, the clip is short (e.g. under ~30s), or a lyric transcription would fail.
Short verse snippets still count as VOCALS_PRESENT=yes when a singable lead is audible — report demographic uncertainty separately, never as absence of voice.

Conversely, isolated pitched noise, pure instrumental melodies, harmonies, or bleed should not automatically be classified as a lead vocal.

If evidence conflicts, report the conflict internally and determine which evidence is stronger.

VOCAL PRESENCE AND VOCAL DEMOGRAPHICS ARE SEPARATE QUESTIONS:

Do not treat uncertainty about singer age, sex or gender as uncertainty about whether a vocal is present.

These are independent classifications:

1. VOCAL PRESENCE
   Is there a human voice in the recording?

2. VOCAL ROLE
   Is it functioning as a lead, backing vocal, harmony, chant, spoken voice, etc.?

3. ACOUSTIC CHARACTER
   What can reliably be heard about its register, timbre, resonance, articulation,
   phrasing and processing?

4. DEMOGRAPHIC CLASSIFICATION
   Can the voice be reliably classified as child/prepubertal, post-pubertal male,
   female adolescent/adult, etc.?

It is entirely valid to conclude:

VOCALS_PRESENT=yes
LEAD_VOCAL=yes
LEAD_CATEGORY=uncertain

Do NOT convert an uncertain demographic classification into "no vocals".

Do NOT convert weak lyric detection into "no vocals".

Do NOT convert an unreliable vocal stem into "no vocals" when a coherent human vocal
is audible in the full mix.

--------------------------------------------------
LEAD VS BACKING VS MULTIPLE SINGERS
--------------------------------------------------

Distinguish carefully:

1. PRIMARY LEAD — the main sung voice carrying the melody for most of the track.
2. DOUBLED LEAD — the same singer stacked/thickened (often near-identical timbre + timing).
3. OCTAVE DOUBLE — same singer (or processed copy) an octave away; not a second person.
4. BACKING HARMONIES — supporting parts under/around the lead; secondary, not co-leads.
5. CALL-AND-RESPONSE / TRADE-OFFS — alternating phrases by different voices.
6. DISTINCT CO-LEADS — two (or more) genuinely different singers who both carry lead material
   in different sections or simultaneously with clearly different timbre/identity.
7. GROUP / UNISON CHOIR — many voices as a texture, not individual identifiable leads.
8. REVERB / DELAY / AD-LIBS — processing or short responses, not separate lead singers.

Rules:
- Do NOT count doubles, octave stacks, reverb, or tight harmonies as separate singers.
- DO mark distinct co-leads when timbre, range, articulation, or sectional role clearly differ
  (e.g. male verse / female chorus, two alternating leads, duet with independent melodic lines).
- If only one clear lead plus backing, say one lead — not mixed_leads.
- If two clear co-leads, use LEAD_PROFILE=mixed_leads and fill the multi-voice fields below.
- Prefer under-counting singers when evidence is weak.

--------------------------------------------------
MULTIPLE SINGERS DETECTION (new detailed section)
--------------------------------------------------

Pay special attention to the possibility of more than one lead singer. This is a common failure
mode. You MUST output the following structured fields whenever any vocal is present:

NUM_DISTINCT_VOICES=[1|2|3+|uncertain]   # actual people, not doubles/stacks
VOICE_ARRANGEMENT=[solo_lead|lead_plus_backing|duet_co_leads|call_response|group_unison|uncertain]
CO_LEAD_DETAIL=[none | short description of each distinct lead: rough register/timbre/role/section]
MULTI_VOICE_EVIDENCE=[what supports multiple people vs doubling/harmony; or "single lead only"]

To determine if multiple leads exist, listen for:
- Different pitch ranges that are consistently assigned to different voices (e.g. one voice always
  lower, one higher).
- Different timbre (bright vs dark, breathy vs clear, nasal vs round) that cannot be explained by
  processing or section changes.
- Different articulation or accent (one voice might have a more pronounced vibrato, or different
  pronunciation).
- Clear alternation between two voices in call‑and‑response or verse/chorus swaps.
- Simultaneous singing of different melodic lines (not just harmonies) that are both prominent.

If you detect a duet or call‑and‑response, set VOICE_ARRANGEMENT accordingly and describe each
voice in CO_LEAD_DETAIL with approximate time markers (e.g. "Verse 1: male tenor, Verse 2: female
alto", or "Chorus: both sing together but with different timbres").

If the evidence is weak, prefer uncertain and do not invent multiple singers.

--------------------------------------------------
VOCAL AGE
--------------------------------------------------

CRITICAL:

Pitch is NOT age.

Brightness is NOT age.

Thinness is NOT age.

High register is NOT age.

Falsetto/head voice is NOT age.

Youthful-sounding delivery is NOT age.

Androgynous tone is NOT age.

A child/prepubertal classification requires clear evidence consistent with a prepubertal vocal tract/resonance profile, not merely a youthful-sounding voice.

Do not classify:
- a high adult male voice as child
- a high adolescent/adult female voice as child
- a light adult voice as child
- falsetto as child

Do not classify a singer as post-pubertal male solely because the voice:
- avoids sounding obviously childlike
- is high but controlled
- is bright or thin
- has a youthful delivery
- uses head voice/falsetto

A youthful male voice may be:
- child/prepubertal
- adolescent (voice actively changing, or recently changed but still light/boyish)
- post-pubertal male
- uncertain

Do not assume that a male singer is post-pubertal simply because the voice is controlled, musically mature, or professionally performed. Young singers can have strong pitch control and polished delivery.

Do not default to post_puberty_male merely because a voice does not sound like a small child. "Not childlike" is not the same as "fully adult male." A voice with a clearly male register but limited lower chest resonance, a boyish/light timbre, or evidence of a voice mid-change is adolescent_male_likely, not post_puberty_male — do not round it up.

Use resonance, vocal weight, maturity of vocal production, and overall evidence.
When evidence is insufficient, choose uncertain rather than forcing either child or adult.

--------------------------------------------------
CATEGORY DEFINITIONS
--------------------------------------------------

child:
Prepubertal/child vocal profile. Requires actual acoustic evidence consistent with a prepubertal vocal tract.

adolescent_male_likely:
A voice that is clearly not a small child's voice but also not a fully mature adult male voice — actively changing, or recently changed but still retaining a light/boyish timbre and limited lower chest resonance. This is a legitimate, distinct category (not a synonym for "uncertain" and not a softened way of saying "child" or "post_puberty_male"). Use it whenever the voice sits genuinely between those two, rather than forcing a binary choice.

post_puberty_male:
Post-pubertal male vocal profile, including unusually high, light, bright or androgynous male singing voices — but with clear evidence of a settled adult male vocal weight/resonance, not merely "not childlike."

female_teen_adult:
Female adolescent/adult vocal profile.

uncertain:
Evidence is insufficient to distinguish the categories reliably.

mixed_leads:
Distinct male/female or otherwise distinct co-leads, not merely harmonies.

--------------------------------------------------
OUTPUT
--------------------------------------------------

VOCALS_PRESENT=yes/no/uncertain
LEAD_PROFILE_NOTE=[brief acoustic description of the primary lead]
BACKING_NOTE=[none/male/female/mixed/uncertain]
NUM_DISTINCT_VOICES=[1|2|3+|uncertain]  # people, not doubles/stacks
VOICE_ARRANGEMENT=[solo_lead|lead_plus_backing|duet_co_leads|call_response|group_unison|uncertain]
CO_LEAD_DETAIL=[none | short description of each distinct lead: rough register/timbre/role/section]
MULTI_VOICE_EVIDENCE=[what supports multiple people vs doubling/harmony; or "single lead only"]
PITCH_NOTE=[broad register and approximate range only if reasonably supported]
FORMANT_NOTE=[prepubertal-like/adolescent-transitional/adult-sized/ambiguous/not assessable]
TIMBRE_NOTE=[specific acoustic characteristics]
DELIVERY_NOTE=[phrasing/articulation/breathiness/vibrato/attack/etc.]
PROCESSING_NOTE=[reverb/doubling/distortion/pitch-processing/etc. when audible]
PROBABILITY_ESTIMATE=child/prepubertal-like X%; adolescent male (voice changing/recently changed) Y%; post-puberty male Z%; female teen/adult A%; uncertain W%

Do not make the percentages look mathematically precise if the evidence is weak. They are comparative confidence estimates, not measured probabilities.

At the very end output exactly:

LEAD_CATEGORY=<child|adolescent_male|post_puberty_male|female_teen_adult|uncertain>
GENDER_MODIFIER=<male_likely|female_likely|gender_uncertain|none>
LEAD_PROFILE=<child_male_likely|child_female_likely|child_gender_uncertain|adolescent_male_likely|post_puberty_male|female_teen_adult|mixed_leads|uncertain>
BACKING_PROFILES=<none|male|female|mixed|uncertain>
CONFIDENCE=low|medium|high

Use LEAD_PROFILE=mixed_leads ONLY when VOICE_ARRANGEMENT is duet_co_leads or call_response
with clearly distinct people. Lead+backing alone is NOT mixed_leads.
"""


# dev55: short free-form question (see VOCAL_FREE_TEXT_GENDER). The long
# structured prompts above are answered with their own template wording.
VOCAL_QUICK_PROMPT = """Listen to the lead singer in this song. In one or two plain sentences, say whether the lead singer is a man, a woman or a child, roughly how old the voice sounds (child, teenager or adult), and whether more than one person sings lead."""


VOCAL_CONFIRMATION_PROMPT_LONG = """Perform an independent second-pass audit of the lead human vocal classification and check for multiple lead singers.

Do NOT simply agree with the initial vocal analysis.

Listen again specifically for evidence that distinguishes:
- prepubertal/child vocal tract
- post-pubertal male voice
- adolescent/adult female voice
- genuinely uncertain cases
- **whether there are multiple distinct lead singers**

The purpose of this pass is to catch false positives in either direction, especially cases where:
- high pitch + bright/light timbre + youthful delivery has incorrectly been interpreted as childhood
- controlled performance + polished delivery + lower vocal weight has incorrectly been interpreted as post-pubertal male
- a duet or call‑and‑response has been overlooked

CRITICAL RULE:

Pitch is not age.

Do not classify a singer as child/prepubertal solely because the voice:
- is high
- sounds young
- sounds bright
- sounds thin
- has a light vocal weight
- uses head voice/falsetto
- sounds androgynous
- has an innocent or playful delivery

Performer age is not vocal evidence.

Do not use:
- singer age at recording
- singer age at release
- artist biography
- debut age
- public image
- known career timeline

to determine the vocal classification.

Age information may be discussed as background context, but the voice classification must come only from the acoustic characteristics of the recording.

For child classification, require clear evidence consistent with a prepubertal vocal tract/resonance.

Child is a high-confidence classification and should not be selected solely from high pitch, brightness, thinness, or youthful tone.

However, do not automatically classify youthful male voices as post-pubertal either.

When distinguishing child/prepubertal male, adolescent male, and post-pubertal male:
- require evidence beyond pitch alone
- consider vocal resonance, vocal weight, maturity of vocal production, and consistency across the performance
- if the voice is clearly not a small child's voice but also lacks settled adult male chest resonance/weight, classify adolescent_male_likely — this is a real, distinct middle category, not a fallback synonym for uncertain, child, or post_puberty_male
- if evidence remains insufficient to place the voice in ANY category (including adolescent_male_likely), choose uncertain
- do not force an adult classification simply because a voice is not obviously childlike — "not childlike" supports adolescent_male_likely at minimum, not automatically post_puberty_male

Do not infer post-pubertal male status from:
- professional recording quality
- confident singing technique
- accurate pitch control
- emotional maturity of performance
- lyrical subject matter
- commercial/pop production style

Young singers can have highly developed performance skills.
Age classification must be based on vocal anatomy-related cues in the audio, not performance ability or appearance.

If the voice has clear adult male vocal characteristics (such as mature resonance,
adult vocal weight, or post-pubertal vocal tract cues across the performance),
classify post_puberty_male.

If the voice is audibly male and clearly not a small child, but the resonance/weight is
light, boyish, or otherwise short of settled adult maturity, classify adolescent_male_likely
rather than rounding up to post_puberty_male or down to child.

Do not use a single ambiguous cue as sufficient evidence.

Do not classify a voice as post_puberty_male solely because it is:
- high pitched
- bright
- light
- thin
- youthful sounding
- androgynous
- using head voice/falsetto

(Those same cues, combined with a clearly male but not-yet-mature register, point toward
adolescent_male_likely rather than either child or post_puberty_male.)

If the evidence does not reliably distinguish a youthful male voice from a child/prepubertal voice,
and it also does not clearly fit adolescent_male_likely, choose uncertain.

If it sounds like an adolescent/adult female, classify female_teen_adult.

If the evidence cannot reliably distinguish the categories, choose uncertain.

Do not use:
- lyrics
- artist stereotypes
- genre
- assumed performer identity

as acoustic evidence.

Do not use cover art unless there is reasonable confidence that a person on the cover is the vocalist.

Backing vocals do not affect the lead classification unless they are distinct co-leads.

If the track has genuine distinct co-leads (different people, not doubles/harmonies),
keep LEAD_PROFILE=mixed_leads rather than forcing a single gender category.

Also note briefly:
NUM_DISTINCT_VOICES=[1|2|3+|uncertain]
VOICE_ARRANGEMENT=[solo_lead|lead_plus_backing|duet_co_leads|call_response|group_unison|uncertain]
CO_LEAD_DETAIL=[none | short description]

Output compact note form:

LEAD_CHECK_NOTE=[brief independent assessment]
CONFIDENCE_REASON=[brief explanation of strongest evidence and/or ambiguity]

At the very end output exactly:

LEAD_CATEGORY=<child|adolescent_male|post_puberty_male|female_teen_adult|uncertain>
GENDER_MODIFIER=<male_likely|female_likely|gender_uncertain|none>
LEAD_PROFILE=<child_male_likely|child_female_likely|child_gender_uncertain|adolescent_male_likely|post_puberty_male|female_teen_adult|mixed_leads|uncertain>
CONFIDENCE=low|medium|high
"""


# Short versions of the era, vocal and confirmation prompts (MOSS_SHORT_PROMPTS).
# Same questions and the same output fields, in the same order, as the long
# versions above, so every parser and rule reads the replies as before. The
# long versions spend most of their length warning against adult/child
# mistakes; MOSS-Music read that weight as a hint and rated a 31-year-old
# rock tenor "child"/"adolescent", and dated a 1978 record 2010s-2020s
# ("modern loudness"), which it cannot judge by ear.
_VOCAL_AGE_GUIDE = """
AGE AND SEX OF THE LEAD VOICE
Judge from the weight, resonance and maturity of the voice, not from how high it sings. Many adult men sing rock and pop leads high, in chest voice, mixed voice or falsetto, and a sung median between about 250 and 350 Hz fits both a high adult male lead and a woman singing low, so pitch alone settles neither age nor sex. A high, bright, thin, nasal or energetic voice is not evidence of youth. Equally, skilful or polished singing is not evidence of adulthood: young singers can be very accomplished.

Categories:
- post_puberty_male: a man's voice, with the settled weight of an adult male voice, at any pitch.
- adolescent_male_likely: a boy whose voice is breaking or has only just broken: cracks or an unsteady register, and little chest weight even on low notes.
- child: a child's unbroken voice before puberty, with a child's small resonance. Needs clear evidence, not just a high pitch.
- female_teen_adult: a woman's or a teenage girl's voice.
- mixed_leads: two or more different people singing lead.
- uncertain: the voice cannot be placed.
Choose the category the voice most plainly fits, and say in plain words whether it sounds like a man, a woman, a boy or a girl.
"""


ERA_ANALYSIS_PROMPT_SHORT = """Estimate when this recording was made, from its production sound alone.

Do not identify the artist or song, and do not date it from the genre or from how polished it sounds.

Listen to how it was made: background noise (tape hiss or none), the kind of reverb, how the drums were made and recorded, what kind of synthesizers or keyboards there are, how the voice is processed, and how the parts were edited together. Name only production markers you can actually hear in this recording, each with roughly where in the track it is. Do not list a marker you cannot point to.

Loudness and dynamic range cannot be judged reliably by ear, and older records are often remastered louder and cleaner. Do not treat loudness, compression, clarity, stereo width or polish as evidence of recent production unless a SUPPLEMENTARY MEASURED EVIDENCE block appears below; if it does, those are real measurements from this file, to be weighed exactly as hedged as they are presented (file-tag clues there date the file, not necessarily the recording).

A recent decade is a valid answer when you hear production that only recent tools make, and an older decade is a valid answer when you hear older production. Do not fall back on a vintage-sounding range by default either: a range that leaves out the present is still a claim. If what you hear does not separate decades, give a broad range or say uncertain, with low confidence. Use high confidence only when several specific markers you can point to agree.

Output exactly:

ERA_ESTIMATE=[likely decade or broad range, or uncertain]
CONFIDENCE=low|medium|high
EVIDENCE=[specific audible production markers]
MISSING_MARKERS=[important evidence that is unavailable or inconclusive]
"""


VOCAL_ANALYSIS_PROMPT_SHORT = """Listen to the human voice(s) in this track and describe the lead singer from the audio alone.

Do not use the artist's name, biography, lyrics, genre or cover art, and do not guess who the singer is.

VOCALS PRESENT
Answer yes if a person sings, raps or speaks words anywhere in the track, even only in some sections, even if the words are hard to make out or the voice is processed. Answer no only if there is no human voice at all. Use uncertain only if you cannot tell whether a sound is a human voice. Long instrumental passages do not make the answer uncertain: one sung verse is enough for yes. Not knowing the singer's age or sex is not doubt about whether there is a voice.

NUMBER OF SINGERS
Count people, not vocal parts. Double-tracking, octave doubles, harmonies, reverb and ad-libs come from the same singer. Report two or more lead singers only when clearly different people carry the melody (for example a man in the verses and a woman in the chorus, or a duet).
""" + _VOCAL_AGE_GUIDE + """
Any measured pitch figures listed below describe the range of the singing only.

OUTPUT

VOCALS_PRESENT=yes/no/uncertain
LEAD_PROFILE_NOTE=[brief acoustic description of the primary lead, saying whether it sounds like a man, a woman, a boy or a girl]
BACKING_NOTE=[none/male/female/mixed/uncertain]
NUM_DISTINCT_VOICES=[1|2|3+|uncertain]  # people, not doubles/stacks
VOICE_ARRANGEMENT=[solo_lead|lead_plus_backing|duet_co_leads|call_response|group_unison|uncertain]
CO_LEAD_DETAIL=[none | short description of each distinct lead: rough register/timbre/role/section]
MULTI_VOICE_EVIDENCE=[what supports multiple people vs doubling/harmony; or "single lead only"]
PITCH_NOTE=[broad register and approximate range only if reasonably supported]
FORMANT_NOTE=[prepubertal-like/adolescent-transitional/adult-sized/ambiguous/not assessable]
TIMBRE_NOTE=[specific acoustic characteristics]
DELIVERY_NOTE=[phrasing/articulation/breathiness/vibrato/attack/etc.]
PROCESSING_NOTE=[reverb/doubling/distortion/pitch-processing/etc. when audible]
PROBABILITY_ESTIMATE=post-puberty male Z%; female teen/adult A%; adolescent male (voice changing/recently changed) Y%; child/prepubertal-like X%; uncertain W%

The percentages are comparative confidence estimates, not measured probabilities.

At the very end output exactly:

LEAD_CATEGORY=<child|adolescent_male|post_puberty_male|female_teen_adult|uncertain>
GENDER_MODIFIER=<male_likely|female_likely|gender_uncertain|none>
LEAD_PROFILE=<child_male_likely|child_female_likely|child_gender_uncertain|adolescent_male_likely|post_puberty_male|female_teen_adult|mixed_leads|uncertain>
BACKING_PROFILES=<none|male|female|mixed|uncertain>
CONFIDENCE=low|medium|high

Use LEAD_PROFILE=mixed_leads ONLY when VOICE_ARRANGEMENT is duet_co_leads or call_response
with clearly distinct people. Lead+backing alone is NOT mixed_leads.
"""


VOCAL_CONFIRMATION_PROMPT_SHORT = """Second, independent check of the lead singer. Listen again and decide for yourself how old the lead voice sounds, whether it is male or female, and whether more than one person sings lead. Do not assume an earlier answer was right.

Use only what you hear. Do not use the artist's name, biography, lyrics, genre or cover art. Backing vocals, doubles and harmonies do not count as extra lead singers.
""" + _VOCAL_AGE_GUIDE + """
Output exactly:

NUM_DISTINCT_VOICES=[1|2|3+|uncertain]
VOICE_ARRANGEMENT=[solo_lead|lead_plus_backing|duet_co_leads|call_response|group_unison|uncertain]
CO_LEAD_DETAIL=[none | short description]
LEAD_CHECK_NOTE=[brief independent assessment, saying whether it sounds like a man, a woman, a boy or a girl]
CONFIDENCE_REASON=[brief explanation of strongest evidence and/or ambiguity]
LEAD_CATEGORY=<child|adolescent_male|post_puberty_male|female_teen_adult|uncertain>
GENDER_MODIFIER=<male_likely|female_likely|gender_uncertain|none>
LEAD_PROFILE=<child_male_likely|child_female_likely|child_gender_uncertain|adolescent_male_likely|post_puberty_male|female_teen_adult|mixed_leads|uncertain>
CONFIDENCE=low|medium|high
"""


# ---- 0.7-pre3/pre4: listening details (saved only; see ENABLE_LISTENING_DETAILS) ----
_LISTENING_DETAILS_HEAD = "Describe how this recording is played and produced, from the audio alone."
_LISTENING_SECTION_HEAD = "This is one part of a longer recording. Describe what you hear in this excerpt, from the audio alone."
_LISTENING_NO_VOICE = "There is no singing in this recording, so do not describe a voice."
_LISTENING_RULES = (
    "Do not identify the artist or the song, and do not use what you know about them. Do not describe "
    "the singer's sex, age or identity; say \"backing vocals\" or \"harmonies\" without saying who "
    "sings them. Only describe what you can actually hear; if you can't tell, write \"unclear\". Keep "
    "each field short."
)


def _listening_rules(instrumental):
    # No singer sentence on an instrumental track: talking about singers there
    # invites the model to hear one.
    if instrumental:
        return _LISTENING_RULES.replace(
            " Do not describe the singer's sex, age or identity; say \"backing vocals\" or \"harmonies\" "
            "without saying who sings them.", "")
    return _LISTENING_RULES


def _listening_details_prompt_narrow(instrumental):
    """0.7-pre5 LISTENING_DETAILS_NARROW: how the voice is delivered and
    produced, effects, and how the parts are played -- no instrument lists.
    Voice fields first and "unclear" rather than "none", so an early "none"
    can't set the pattern for every field."""
    fields = []
    if not instrumental:
        fields += [
            "VOCAL_DELIVERY=[how the words are delivered: belted, breathy, falsetto, spoken, rapped, shouted, "
            "crooned; ad-libs or call-and-response]",
            "VOCAL_PRODUCTION=[how the voice is recorded and treated: dry or reverberant, echo, doubled or "
            "stacked, harmonies, backing vocals, vocoder or heavy pitch effects, filtered or telephone sound]",
        ]
    fields += [
        "EFFECTS=[effects you can clearly hear on the instruments: reverb type (room, hall, plate, spring), "
        "echo or delay, chorus, flanger, phaser, tremolo, distortion, filter sweeps; or \"none clear\"]",
        "GUITAR_PLAYING=[if a guitar is clearly playing, how: strummed, picked, fingerpicked, arpeggiated, "
        "palm-muted, slide; clean or distorted; otherwise \"unclear\"]",
        "BASS_PLAYING=[if a bass is clearly playing, how: fingered, picked, slapped, synth bass, sub-bass; "
        "otherwise \"unclear\"]",
        "DRUM_PLAYING=[if drums are clearly playing: a live kit or programmed, and how: sticks or brushes, "
        "straight or swung; otherwise \"unclear\"]",
        "KEYS_SOUND=[if keyboards or synths are clearly playing, what they sound like: piano, electric piano, "
        "organ, warm pads, bright or bell-like, plucky, buzzy; otherwise \"unclear\"]",
    ]
    return (_LISTENING_DETAILS_HEAD + (" " + _LISTENING_NO_VOICE if instrumental else "") + "\n\n"
            + _listening_rules(instrumental) + "\n\nOutput exactly:\n\n" + "\n".join(fields) + "\n")


def _listening_section_prompt_narrow(instrumental):
    fields = []
    if not instrumental:
        fields.append("VOICE=[if someone sings or speaks here, how: belted, soft, breathy, falsetto, "
                      "harmonies, spoken, rapped; otherwise \"no voice\"]")
    fields.append("STANDS_OUT=[anything notable here: a riff, a solo, an effect, a change in the beat"
                  + ("" if instrumental else " or the voice") + "; or \"nothing in particular\"]")
    return (_LISTENING_SECTION_HEAD + (" " + _LISTENING_NO_VOICE if instrumental else "") + "\n\n"
            + _listening_rules(instrumental).replace("Keep each field short.", "One short sentence per field.")
            + "\n\nOutput exactly:\n\n" + "\n".join(fields) + "\n")


_LISTEN_BLANK = {"", "none", "none clear", "unclear", "n/a", "na", "no", "nothing", "nothing in particular",
                 "none in particular", "no voice", "not applicable"}


def _listening_blank_excerpt(reply):
    """0.7.1-pre28 LISTENING_KEEP_BLANK_EXCERPT: the first 240 characters of a reply judged blank, on one
    line, with sex and age words taken out as in the saved fields (_listening_scrub)."""
    try:
        text = _listening_scrub({"VOICE": str(reply or "")})[0].get("VOICE", "")
    except Exception:
        return ""
    text = re.sub(r"\s+", " ", text).strip()
    return text[:240] + ("..." if len(text) > 240 else "")


def _listening_blank(fields):
    """0.7-pre5 LISTENING_DETAILS_DROP_BLANK: True when a reply says nothing."""
    vals = [re.sub(r"[\s.\"']+", " ", str(v)).strip().lower() for v in (fields or {}).values()]
    return not vals or all(v in _LISTEN_BLANK for v in vals)


def _listening_details_prompt(instrumental):
    # pre4: no solos or timed "notable moments" here -- on a whole song the
    # model can't place them (it invented times and missed a long solo); the
    # 30-second windows say what happens when. A drums field instead.
    fields = [
        "GUITARS=[guitar parts and how they're played: clean, crunchy or distorted; strummed, picked, "
        "arpeggiated, palm-muted, slide; or \"none\"]",
        "BASS=[how the bass is played: fingered, picked, slapped, synth bass, sub-bass, upright; or \"unclear\"]",
        "DRUMS=[a live drum kit, a drum machine or programmed beat, hand percussion, or none; and how it's played]",
        "KEYS_SYNTHS=[keyboards and synths: piano, electric piano, organ, pads, leads, arpeggios; or \"none\"]",
        "OTHER_INSTRUMENTS=[anything else clearly audible: strings, brass, woodwind, recorder or flute, "
        "saxophone, percussion, samples; or \"none\"]",
        "EFFECTS=[audible effects on the instruments: reverb type (room, hall, plate, spring), echo or "
        "delay, chorus, flanger, phaser, tremolo, distortion, filter sweeps; or \"none clear\"]",
    ]
    if not instrumental:
        fields += [
            "VOCAL_PRODUCTION=[how the voice is recorded and treated: dry or reverberant, echo, doubled or "
            "stacked, harmonies, backing vocals, vocoder or heavy pitch effects, filtered or telephone sound]",
            "VOCAL_DELIVERY=[how the words are delivered: belted, breathy, falsetto, spoken, rapped, shouted, "
            "crooned; ad-libs or call-and-response]",
        ]
    return (_LISTENING_DETAILS_HEAD + (" " + _LISTENING_NO_VOICE if instrumental else "") + "\n\n"
            + _listening_rules(instrumental) + "\n\nOutput exactly:\n\n" + "\n".join(fields) + "\n")


def _listening_section_prompt(instrumental):
    fields = ["INSTRUMENTS=[what is playing here and how]"]
    if not instrumental:
        fields.append("VOICE=[if someone sings or speaks here, how: belted, soft, breathy, falsetto, "
                      "harmonies, spoken, rapped; otherwise \"no voice\"]")
    fields += ["ENERGY=[how full it sounds here: sparse, building, full, peak or dropping away]",
               "STANDS_OUT=[anything notable here: a riff, a solo, an effect, a change in the beat"
               + ("" if instrumental else " or the voice") + "; or \"nothing in particular\"]"]
    return (_LISTENING_SECTION_HEAD + (" " + _LISTENING_NO_VOICE if instrumental else "") + "\n\n"
            + _listening_rules(instrumental).replace("Keep each field short.", "One short sentence per field.")
            + "\n\nOutput exactly:\n\n" + "\n".join(fields) + "\n")


def _listening_fields(text, labels):
    """{LABEL: value} from a LABEL=value reply (values cleaned of brackets)."""
    out = {}
    t = str(text or "")
    _alts = "|".join(re.escape(x) for x in labels)
    for i, lab in enumerate(labels):
        if LISTENING_FIELDS_INLINE_SPLIT:
            # pre53: "VOICE=...; STANDS_OUT=..." on one line holds two fields.
            m = re.search(r"(?:^|;)[ \t]*" + re.escape(lab) + r"\s*=\s*(.*?)(?=(?:^|;)[ \t]*(?:"
                          + _alts + r")\s*=|\Z)", t, re.S | re.M)
        else:
            m = re.search(r"(?m)^\s*" + re.escape(lab) + r"\s*=\s*(.*?)(?=^\s*(?:"
                          + _alts + r")\s*=|\Z)", t, re.S)
        if m:
            v = re.sub(r"\s+", " ", m.group(1)).strip().strip("[]").strip()
            if v:
                out[lab] = v
    return out


# pre4: the listening model ignored "don't describe the singer's sex or age"
# (it called an unbroken male voice "female" in most clips), so those words
# are taken out of what is saved rather than trusted to the instruction.
# Sex words come out of every field; age and voice-type words only out of the
# voice fields (so "alto saxophone" or "old-school drum machine" stay).
_LISTEN_SEX_SUBS = (
    (r"\b(?:a|an|the)\s+(?:young\s+|adult\s+|teenage\s+)?(?:woman|man|girl|boy|lady|guy)\b", "a singer"),
    (r"\b(?:women|men|girls|boys|ladies|guys)\b", "singers"),
    (r"\b(?:woman|man|girl|boy|lady|guy)(?='s)\b", "singer"),
    (r"\b(?:woman|man|girl|boy|lady|guy)\b", "singer"),
    (r"\b(?:fe)?male(?:-\w+)?\b", ""),
    (r"\b(?:feminine|masculine|girlish|boyish|womanly|manly)\b", ""),
    (r"\b(?:she|he)\b", "the singer"),
    (r"\b(?:her|his)\b", "the"),
)
_LISTEN_AGE_VOICE_SUBS = (
    (r"\b(?:child|childlike|childish|kid|teen|teenage|teenager|adolescent|young|youthful|adult|mature|"
     r"elderly|old|aged|older|younger)(?:-\w+)?\b", ""),
    (r"\b(?:counter)?tenor\b|\bbaritone\b|\bbass-baritone\b|\b(?:mezzo-)?soprano\b|\bmezzo\b|\bcontralto\b|"
     r"\balto\b|\btreble\b", ""),
)
_LISTEN_VOICE_FIELDS = ("VOICE", "VOCAL_PRODUCTION", "VOCAL_DELIVERY")


def _listening_scrub(fields):
    """Take sex/age words out of a {LABEL: value} dict. Returns (fields, n_removed)."""
    out, removed = {}, 0
    for k, v in (fields or {}).items():
        t = str(v)
        subs = _LISTEN_SEX_SUBS + (_LISTEN_AGE_VOICE_SUBS if k in _LISTEN_VOICE_FIELDS else ())
        for rx, rep in subs:
            t, n = re.subn(rx, rep, t, flags=re.I)
            removed += n
        t = re.sub(r"\s+", " ", t)
        t = re.sub(r"\s+([,;.])", r"\1", t)
        t = re.sub(r"([,;])\s*(?=[,;.]|$)", "", t)
        t = re.sub(r"^\s*[,;]\s*", "", t)
        t = re.sub(r"\b(a|an)\s+(?=[,;.]|$)", "", t, flags=re.I)
        t = re.sub(r"\b([Aa])n?\s+(?=([A-Za-z]))",
                   lambda m: m.group(1) + ("n " if m.group(2).lower() in "aeiou" else " "), t)
        t = re.sub(r"\bthe singer singer\b", "the singer", t)
        t = re.sub(r"\bsinger voice\b", "voice", t)
        t = re.sub(r"\s+", " ", t).strip(" ,;")
        if t:
            out[k] = t
    return out, removed


def _listening_windows(duration):
    """pre4: even windows across the whole track (LISTENING_DETAILS_WINDOW_S
    each, at most LISTENING_DETAILS_MAX_WINDOWS, spread evenly on longer
    tracks). A short tail joins the window before it."""
    w = max(5.0, float(LISTENING_DETAILS_WINDOW_S))
    dur = float(duration or 0)
    if dur < LISTENING_DETAILS_CLIP_MIN_S:
        return []
    edges = []
    t = 0.0
    while t < dur:
        edges.append([t, min(dur, t + w)])
        t += w
    if len(edges) > 1 and edges[-1][1] - edges[-1][0] < LISTENING_DETAILS_CLIP_MIN_S:
        tail = edges.pop()
        edges[-1][1] = tail[1]
    return _sample_list_evenly(edges, int(LISTENING_DETAILS_MAX_WINDOWS))


def _listening_section_label(sections, a, b):
    """Name of the first-map section that overlaps [a, b] most (a guide only)."""
    best, best_o = "", 0.0
    for sec in sections or []:
        try:
            o = min(b, float(sec.get("end"))) - max(a, float(sec.get("start")))
        except Exception:
            continue
        if o > best_o:
            best, best_o = str(sec.get("name") or sec.get("label") or ""), o
    return best


def _listening_details_pass(moss_model, moss_processor, mix_path, sections, instrumental):
    """0.7-pre3 Stage 1 (pre4 revision): the whole-track listening details and
    a description of each 30-second window across the song. Runs after every
    existing listening pass; reads nothing back into the analysis. Returns the
    dict saved as "listening_details", or None."""
    if moss_model is None or not mix_path or not os.path.exists(str(mix_path)):
        return None
    narrow = bool(LISTENING_DETAILS_NARROW)
    result = {"model": MOSS_MODEL_ID, "version": 3 if narrow else 2, "instrumental": bool(instrumental),
              "note": "listening-model impressions, saved only; not used by the analysis. Words about the "
                      "singer's sex or age are removed (voice_words_removed counts them)"}
    if narrow:
        result["note"] += ("; which instruments play comes from the measured instrument tags, not from these "
                           "fields")
    try:
        status("Listening — describing how it's played and produced...")
        prompt = (_listening_details_prompt_narrow if narrow else _listening_details_prompt)(instrumental)
        reply = moss_generate(moss_model, moss_processor, [{"role": "user", "content": [
            {"type": "text", "text": prompt}, {"type": "audio", "path": str(mix_path)}]}],
            max_new_tokens=900) or ""
        track, n = _listening_scrub(_listening_fields(reply, _moss_template_labels(prompt)))
        if LISTENING_DETAILS_DROP_BLANK and _listening_blank(track):
            result["track_dropped"] = "blank reply"
            if LISTENING_KEEP_BLANK_EXCERPT:
                result["track_blank_reply"] = _listening_blank_excerpt(reply)     # 0.7.1-pre28
        else:
            result["track"] = track
        if n:
            result["track_voice_words_removed"] = n
    except Exception as e:
        result["track_error"] = f"{type(e).__name__}: {e}"[:200]
    if LISTENING_DETAILS_SECTIONS:
        try:
            import soundfile as _sf
            mix, rate = librosa.load(str(mix_path), sr=16000, mono=True)
            windows = _listening_windows(len(mix) / float(rate))
            prompt = (_listening_section_prompt_narrow if narrow else _listening_section_prompt)(instrumental)
            labels = _moss_template_labels(prompt)
            out = []
            for n_, (a, b) in enumerate(windows, 1):
                status(f"Listening — part {n_} of {len(windows)}...")
                audio = mix[max(0, int(a * rate)):min(len(mix), int(b * rate))]
                entry = {"start": round(a, 2), "end": round(b, 2),
                         "first_map_section": _listening_section_label(sections, a, b)}
                if not len(audio) or float(np.max(np.abs(audio))) < 1e-5:
                    entry["silent"] = True
                    out.append(entry)
                    continue
                tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                tmp.close()
                try:
                    _sf.write(tmp.name, audio, rate)
                    reply = moss_generate(moss_model, moss_processor, [{"role": "user", "content": [
                        {"type": "text", "text": prompt}, {"type": "audio", "path": tmp.name}]}],
                        max_new_tokens=320) or ""
                    fields, n = _listening_scrub(_listening_fields(reply, labels))
                    if LISTENING_DETAILS_DROP_BLANK and narrow and _listening_blank(fields):
                        entry["blank"] = True
                        if LISTENING_KEEP_BLANK_EXCERPT:
                            entry["blank_reply"] = _listening_blank_excerpt(reply)     # 0.7.1-pre28
                    else:
                        entry.update(fields)
                    if n:
                        entry["voice_words_removed"] = n
                except Exception as e:
                    entry["error"] = f"{type(e).__name__}: {e}"[:200]
                finally:
                    try:
                        os.remove(tmp.name)
                    except OSError:
                        pass
                out.append(entry)
            result["windows"] = out
            result["windows_note"] = ("even windows across the whole track; first_map_section is the scan's "
                                      "first, rough section map, a guide only -- match to the saved sections "
                                      "by time")
        except Exception as e:
            result["windows_error"] = f"{type(e).__name__}: {e}"[:200]
    status_done()
    return result if (result.get("track") or result.get("windows")) else None


SECTION_VOICE_CLIP_PROMPT = """Inspect only this short audio excerpt, not the artist or the rest of the song.
First decide whether a foreground human LEAD is clearly singing, rapping, speaking or chanting here.
Spoken-style, monotone and heavily processed human delivery counts without a sung melody.
An instrumental melody, faint backing vocal or unclear texture is not a foreground lead.
If no lead is clear, use uncertain for its gender and arrangement. Do not guess from pitch.
For a clear lead, describe the voice heard in this clip without inferring the singer's age.
Use ONE value per field. VOCALS_PRESENT: yes, no, or uncertain. LEAD_GENDER: male,
female, mixed, or uncertain. VOICE_ARRANGEMENT: solo_lead, lead_plus_backing,
duet_co_leads, group_unison, or uncertain. CONFIDENCE: high, medium, or low.
Output:
VOCALS_PRESENT=
LEAD_GENDER=
VOICE_ARRANGEMENT=
CONFIDENCE=
REASON=
"""
VOCAL_PRESENCE_VERIFY_PROMPT = """Listen only to this audio excerpt. Determine whether ANY human voice is audible:
singing, speech, rap, wordless humming, choir, backing voices, distant or heavily
processed vocals all count, even without intelligible words or a foreground lead.
Do not infer a voice from guitar/synth melodies, formant-like effects or a file name.
Do not transcribe lyrics or identify a singer. If unsure, answer uncertain.
Answer no with high confidence only when the audible content is entirely instruments,
environmental sounds or silence and you can exclude human vocal sounds.
Use one value per field. HUMAN_VOICE: yes, no, uncertain.
CONFIDENCE: high, medium, low.
Output:
HUMAN_VOICE=
CONFIDENCE=
REASON=
"""

SECTION_VOICE_PAIR_PROMPT = """This audio contains two excerpts from one recording separated by one second of silence.
Compare the foreground LEAD vocalist in excerpt A with excerpt B. Decide whether it is
the same singer changing register/delivery, clearly a different lead singer, or uncertain.
Ignore backing harmonies, doubling, pitch alone and changes of production. Different pitch
or apparent gender alone does not prove different people. If either excerpt lacks a clear
foreground lead, answer uncertain. Use ONE value per field. LEAD_CHANGE: same_lead,
different_lead, or uncertain. CONFIDENCE: high, medium, or low.
Output:
LEAD_CHANGE=
CONFIDENCE=
REASON=
"""


if MOSS_SHORT_PROMPTS:
    ERA_ANALYSIS_PROMPT = ERA_ANALYSIS_PROMPT_SHORT
    VOCAL_ANALYSIS_PROMPT = VOCAL_ANALYSIS_PROMPT_SHORT
    VOCAL_CONFIRMATION_PROMPT = VOCAL_CONFIRMATION_PROMPT_SHORT
else:
    ERA_ANALYSIS_PROMPT = ERA_ANALYSIS_PROMPT_LONG
    VOCAL_ANALYSIS_PROMPT = VOCAL_ANALYSIS_PROMPT_LONG
    VOCAL_CONFIRMATION_PROMPT = VOCAL_CONFIRMATION_PROMPT_LONG


# (DEBUG_MODEL_LOADING moved to the settings file beside this script.)


def _moss_planned_device():
    """Which device MOSS-Music will load onto, without loading it.
    Safe to call at startup / for display purposes."""
    return "mps" if torch.backends.mps.is_available() else "cpu"


_MOSS_CODE_CLONE_HINT = (
    "MOSS-Music's model code was not found. It comes from OpenMOSS's GitHub repo "
    "(the copy on Hugging Face does not load with transformers). Run once:\n"
    "    git clone https://github.com/OpenMOSS/MOSS-Music.git ~/MOSS-Music\n"
    "    git -C ~/MOSS-Music checkout " + "ad107c7ddaa06de168a0dfbc18d3e1e6a40c0e5e" + "\n"
    "or set MOSS_MUSIC_CODE_DIR to wherever you cloned it."
)


def _moss_code_dir():
    """The OpenMOSS/MOSS-Music clone: MOSS_MUSIC_CODE_DIR, the environment
    variable of the same name, a MOSS-Music folder next to this script, or
    ~/MOSS-Music -- the first that holds src/modeling_moss_music.py."""
    here = os.path.dirname(os.path.abspath(__file__))
    for c in (MOSS_MUSIC_CODE_DIR, os.environ.get("MOSS_MUSIC_CODE_DIR"),
              os.path.join(here, "MOSS-Music"), os.path.expanduser("~/MOSS-Music")):
        if c and os.path.isfile(os.path.join(os.path.expanduser(c), "src", "modeling_moss_music.py")):
            return os.path.abspath(os.path.expanduser(c))
    return None


def _moss_code_classes():
    """(MossMusicProcessor, MossMusicModel) from the MOSS-Music clone -- the
    same code OpenMOSS's own infer.py imports. Its modules import each other
    as the package "src", so the clone is put on the path just long enough
    to import them."""
    import importlib
    d = _moss_code_dir()
    if d is None:
        raise RuntimeError(_MOSS_CODE_CLONE_HINT)
    pkg = sys.modules.get("src")
    if pkg is not None:
        paths = [os.path.abspath(p) for p in (getattr(pkg, "__path__", None) or [])]
        if not any(p.startswith(d) for p in paths):
            raise RuntimeError(
                "Another module named 'src' is already imported "
                f"({paths or getattr(pkg, '__file__', '?')}), so MOSS-Music's code in {d} "
                "cannot be loaded alongside it.")
    added = d not in sys.path
    if added:
        sys.path.insert(0, d)
    try:
        proc_mod = importlib.import_module("src.processing_moss_music")
        model_mod = importlib.import_module("src.modeling_moss_music")
    finally:
        if added:
            try:
                sys.path.remove(d)
            except ValueError:
                pass
    return proc_mod.MossMusicProcessor, model_mod.MossMusicModel


def _moss_load_tokenizer(rev):
    """MOSS-Music's Qwen2 tokenizer, loaded by its own class so nothing from
    the model repo's custom code runs (AutoTokenizer would read config.json,
    find custom code there and stop to ask "Do you wish to run the custom
    code? [y/N]" -- which is what MOSS-Music's own processor loader does)."""
    try:
        from transformers import Qwen2Tokenizer
        return Qwen2Tokenizer.from_pretrained(MOSS_MODEL_ID, **rev)
    except Exception as e:
        if DEBUG_MODEL_LOADING:
            print(f"  (Qwen2Tokenizer load failed ({type(e).__name__}: {e}); using AutoTokenizer)")
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(MOSS_MODEL_ID, use_fast=False, trust_remote_code=True, **rev)


def _moss_transformers_compat(model):
    """MOSS-Music's own model code was written for transformers 4.57 (its
    package pins 4.57.1), while transformers 5.x -- which this script used
    for the previous listening model -- changed two things it relies on.
    Each of fixes 1-2 checks for the new behaviour first and does nothing on
    a transformers version that still has the old one (checked on 4.57.1,
    5.0, 5.10 and 5.20). Fix 3 saves memory on every version:

    1. Whisper encoder layers (used inside MOSS-Music's audio encoder) now
       return a tensor rather than a tuple and no longer take
       layer_head_mask; MOSS-Music's encoder indexes [0] into the result.
       Such layers are wrapped to accept those arguments and return a tuple.
    2. generate() no longer passes cache_position, which MOSS-Music's
       prepare_inputs_for_generation uses to tell the first decoding step
       (audio + whole prompt) from the later cached steps (last token only).
       When it is missing it is rebuilt from the cache length.
    3. Only the last prompt position is turned into scores (see below).
    Returns a short description of what was adapted (for DEBUG_MODEL_LOADING)."""
    import inspect as _inspect
    done = []

    # 0. The audio encoder's sinusoidal position table (inv_timescales) is a
    #    computed, non-saved buffer. transformers 5.x builds the model on the
    #    "meta" device before loading weights, and this custom buffer is never
    #    filled in afterwards: it is left as uninitialised memory (all zeros on
    #    5.15, huge random values / NaN on 5.20). The encoder then cannot tell
    #    where in the track each audio frame is, and the model's replies
    #    degrade (invented details, runaway loops, early stops). It is
    #    recomputed here with MOSS-Music's own formula -- on transformers 4.57
    #    this rewrites the identical values.
    fixed = 0
    for mod in model.modules():
        buf = getattr(mod, "inv_timescales", None)
        if not torch.is_tensor(buf) or buf.dim() != 1 or buf.numel() < 2:
            continue
        half = int(buf.numel())                 # embedding_dim // 2
        inc = math.log(10000.0) / (half - 1)
        good = torch.exp(-inc * torch.arange(half, dtype=torch.float32))
        with torch.no_grad():
            mod.inv_timescales = good.to(device=buf.device)
        fixed += 1
    if fixed:
        done.append(f"audio position table recomputed ({fixed})")

    enc_layers = getattr(getattr(model, "audio_encoder", None), "layers", None) or []
    wrapped = 0
    for layer in enc_layers:
        f = layer.forward
        try:
            params = _inspect.signature(f).parameters
        except (TypeError, ValueError):
            continue
        if "layer_head_mask" in params:
            continue                      # transformers 4.x layer: nothing to do

        def _fwd(hidden_states, attention_mask=None, *args, _f=f, **kwargs):
            kwargs.pop("layer_head_mask", None)
            kwargs.pop("output_attentions", None)
            out = _f(hidden_states, attention_mask, **kwargs)
            return (out,) if torch.is_tensor(out) else out
        layer.forward = _fwd
        wrapped += 1
    if wrapped:
        done.append(f"audio-encoder layer outputs ({wrapped} layers)")

    orig_prep = model.prepare_inputs_for_generation

    # The parameter must be called **kwargs: generate() reads this signature
    # to decide which extra inputs (audio_data, ...) the model accepts.
    def _prep(input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None,
              cache_position=None, **kwargs):
        if cache_position is None and past_key_values is not None:
            try:
                past = int(past_key_values.get_seq_length())
            except Exception:
                past = 0
            if 0 < past < int(input_ids.shape[1]):
                cache_position = torch.arange(past, int(input_ids.shape[1]), device=input_ids.device)
        return orig_prep(input_ids, past_key_values=past_key_values, attention_mask=attention_mask,
                         inputs_embeds=inputs_embeds, cache_position=cache_position, **kwargs)
    model.prepare_inputs_for_generation = _prep
    done.append("decoding-step detection")

    # 3. Memory: MOSS-Music's forward() turns EVERY prompt position into a
    #    151,936-wide row of scores, although generate() only ever reads the
    #    last one. On a 12-20k-token listening prompt that is a 3-6 GB
    #    spike at the start of each pass. The previous listening model (a
    #    built-in transformers class) only scored the last position, so this
    #    does the same while generating (no gradients). The scores generate()
    #    reads are the same apart from last-digit bf16 rounding.
    head = getattr(model, "lm_head", None)
    if head is not None:
        orig_head = head.forward

        def _last_position_head(hidden_states, *args, _f=orig_head, **kwargs):
            if (not torch.is_grad_enabled() and torch.is_tensor(hidden_states)
                    and hidden_states.dim() == 3 and hidden_states.shape[1] > 1):
                hidden_states = hidden_states[:, -1:, :]
            return _f(hidden_states, *args, **kwargs)
        head.forward = _last_position_head
        done.append("last-position scores only")
    return ", ".join(done)


def _load_moss_music_impl():
    """Do the actual (slow, RAM/VRAM-heavy) work of loading MOSS-Music's
    weights. Callers should go through get_moss_music() instead, which
    caches the result so this only runs when the model isn't already resident."""
    if DEBUG_MODEL_LOADING:
        print(f"Loading {MOSS_MODEL_ID} ...")
    else:
        status(f"Loading {MOSS_MODEL_ID}...")
    rev = {"revision": MOSS_MODEL_REVISION} if MOSS_MODEL_REVISION else {}
    # Code from the MOSS-Music GitHub clone (as OpenMOSS's infer.py does),
    # weights and tokenizer from Hugging Face. No custom code from the model
    # repo is run, so transformers never asks about trust_remote_code.
    proc_cls, model_cls = _moss_code_classes()
    # The processor is built from its tokenizer, as its from_pretrained()
    # does -- but with the time markers on, which that method leaves off.
    processor = proc_cls(_moss_load_tokenizer(rev), enable_time_marker=bool(MOSS_ENABLE_TIME_MARKERS))
    device = _moss_planned_device()
    model = model_cls.from_pretrained(
        MOSS_MODEL_ID,
        device_map=device,
        torch_dtype=MOSS_TORCH_DTYPE,
        **rev,
    )
    model.eval()
    _compat = _moss_transformers_compat(model)
    if DEBUG_MODEL_LOADING:
        print(f"  (MOSS-Music runtime adaptations: {_compat})")
        print(f"MOSS-Music loaded on device: {device}")
    else:
        status_done(f"MOSS-Music loaded on device: {device}")
    return model, processor, device


# --- Lazy loading / swapping for MOSS-Music --------------------------------
# MOSS-Music's weights are large and only actually needed while doing
# fresh audio analysis (era/full/vocal/confirmation/lyrics passes). Rather
# than holding them in RAM/VRAM for the entire session, we load them on
# first use and explicitly release them again as soon as that batch of
# analysis passes finishes, so the memory is available to other things
# (like the Ollama-served writer model) the rest of the time. This never
# touches conversation/chat history — that lives in plain Python lists in
# main() and is unaffected by whether any model is currently loaded.
_moss_state = {"model": None, "processor": None, "device": None}


def get_moss_music():
    """Return (model, processor, device), loading MOSS-Music into
    RAM/VRAM on first call and reusing the cached instance thereafter.
    Call unload_moss_music() when done with a batch of analysis passes
    to free that memory again."""
    if _moss_state["model"] is None:
        model, processor, device = _load_moss_music_impl()
        _moss_state["model"] = model
        _moss_state["processor"] = processor
        _moss_state["device"] = device
    return _moss_state["model"], _moss_state["processor"], _moss_state["device"]


def unload_moss_music():
    """Drop MOSS-Music's weights and free the RAM/VRAM they occupied.
    Safe to call even if it was never loaded (no-op). The next call to
    get_moss_music() will transparently reload it."""
    if _moss_state["model"] is None and _moss_state.get("processor") is None:
        return

    status("Freeing MOSS-Music from memory...")
    model = _moss_state["model"]
    processor = _moss_state["processor"]
    _moss_state["model"] = None
    _moss_state["processor"] = None
    _moss_state["device"] = None
    _MOSS_AUDIO_CACHE.clear()

    try:
        del model
    except Exception:
        pass
    try:
        del processor
    except Exception:
        pass

    # Multiple GC + cache clears help on MPS, where a single empty_cache often
    # leaves large allocations resident until the next peak and OOM-kills batch.
    # synchronize() forces pending GPU work to finish so empty_cache can reclaim.
    for _ in range(3):
        gc.collect()
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

    status_done("MOSS-Music unloaded" + _whisper_memory_suffix())


def _loop_block_at_end(toks, max_period=48, min_span=32, min_reps=8):
    """Length of a block of tokens repeated back to back at the end of toks --
    at least min_reps times and at least min_span tokens in all -- or 0.
    A healthy report never does this; a runaway list ("B – C#m7 – A6 – B –
    C#m7 – A6 ..." 90 times) does."""
    L = len(toks)
    for p in range(1, max_period + 1):
        reps = max(min_reps, -(-min_span // p))
        need = p * reps
        if L < need:
            if p * min_reps > L:
                break
            continue
        if toks[-1] != toks[-1 - p]:
            continue
        ok = True
        for k in range(1, need - p + 1):
            if toks[-k] != toks[-k - p]:
                ok = False
                break
        if ok:
            return p
    return 0


class _MossLoopGuard(_HFLogitsProcessor):
    """Ends a runaway line in the listening model's report.

    Decoding is greedy, so once the model starts repeating a short block it
    repeats it until the token budget runs out: the rock test track's report
    listed the same three chords 271 times and never reached its
    instrumentation, timbre or vocal fields. This watches the current line
    and, only when a block has repeated back to back (8+ times, 32+ tokens),
    makes the next token a line break so the report moves on to its next
    field; after three cuts on consecutive lines (or eight in all) it ends
    the reply. Until a loop is found it
    leaves every score untouched, so a report that never loops comes out
    exactly as before, token for token.
    """

    def __init__(self, tokenizer, prompt_len, numbered=False, eos_id=None):
        self.tok = tokenizer
        self.prompt_len = int(prompt_len)
        # dev61: also catch a numbered list looping with changing numbers
        # ("6) rock; 7) pop; 8) power pop; ..."), see MOSS_CONSISTENT_REPORT.
        self.numbered = bool(numbered)
        self._digit_cache = {}
        self.toks = []
        self.line_start = 0
        self.fires = 0
        self.streak = 0
        self.nlines = 0
        self.last_cut_line = -10
        self._nl_cache = {}
        self.nl = None
        self.eos = None
        try:
            ids = tokenizer("\n", add_special_tokens=False)["input_ids"]
            if isinstance(ids, (list, tuple)) and len(ids) == 1:
                self.nl = int(ids[0])
        except Exception:
            self.nl = None
        try:
            eos = getattr(tokenizer, "eos_token_id", None) if eos_id is None else eos_id
            self.eos = int(eos[0] if isinstance(eos, (list, tuple)) else eos) if eos is not None else None
        except Exception:
            self.eos = None

    def _is_digits(self, t):
        v = self._digit_cache.get(t)
        if v is None:
            try:
                v = self.tok.decode([t]).strip().isdigit()
            except Exception:
                v = False
            self._digit_cache[t] = v
        return v

    def _numbered_loop(self, line):
        """dev61: True when the line ends in a numbered list repeating its
        items with only the numbers changing. Runs of digit tokens count as one
        placeholder; the repeating block must itself hold a list number ("N)"
        or "N."), so chord names with digits are never caught by this."""
        if len(line) < 48:
            return False
        norm, src = [], []
        for t in line:
            if self._is_digits(t):
                if norm and norm[-1] == -1:
                    continue
                norm.append(-1)
            else:
                norm.append(t)
            src.append(t)
        p = _loop_block_at_end(norm, max_period=160, min_span=48, min_reps=3)
        if not p:
            return False
        block = norm[-p:]
        for i in range(len(block) - 1):
            if block[i] == -1:
                try:
                    nxt = self.tok.decode([block[i + 1]]).lstrip()
                except Exception:
                    nxt = ""
                if nxt[:1] in (")", "."):
                    return True
        return False

    def _has_nl(self, t):
        v = self._nl_cache.get(t)
        if v is None:
            try:
                v = "\n" in self.tok.decode([t])
            except Exception:
                v = False
            self._nl_cache[t] = v
        return v

    def __call__(self, input_ids, scores):
        try:
            if self.nl is None:
                return scores
            n = int(input_ids.shape[1]) - self.prompt_len
            while len(self.toks) < n:
                t = int(input_ids[0, self.prompt_len + len(self.toks)])
                self.toks.append(t)
                if self._has_nl(t):
                    self.line_start = len(self.toks)
                    self.nlines += 1
            if (_loop_block_at_end(self.toks[self.line_start:])
                    or (self.numbered and self._numbered_loop(self.toks[self.line_start:]))):
                self.fires += 1
                # Ending the reply is only for a model that falls straight
                # back into a loop on line after line.
                self.streak = self.streak + 1 if self.nlines == self.last_cut_line + 1 else 1
                self.last_cut_line = self.nlines
                scores[:] = float("-inf")
                if (self.streak >= 3 or self.fires >= 8) and self.eos is not None:
                    scores[:, self.eos] = 0.0
                else:
                    scores[:, self.nl] = 0.0
            return scores
        except Exception:
            return scores


# --- MOSS-Music prompt / audio adapter ---------------------------------------
# The listening passes build their requests as chat conversations (a list of
# {"role", "content": [{"type": "text"|"audio", ...}]} messages), with system
# messages for the empty-pass retry and an assistant + user turn for the
# self-check. MOSS-Music's processor has no chat-template method: it takes
# ONE prompt string (ChatML, with an <|audio_bos|><|AUDIO|><|audio_eos|>
# placeholder per clip) plus the decoded audio arrays. The helpers below turn
# the same conversations into that prompt -- audio first in its turn, the
# turn's text after it, as MOSS-Music's own default prompt does -- so every
# caller, retry and follow-up works unchanged.
_MOSS_DEFAULT_SYSTEM = "You are a helpful assistant."   # MOSS-Music's own default
_MOSS_AUDIO_SPAN = "<|audio_bos|><|AUDIO|><|audio_eos|>"
_MOSS_AUDIO_TOKEN_ID_DEFAULT = 151654
_MOSS_AUDIO_CACHE = {}        # decoded clips, most recent last (see _moss_load_audio)
_MOSS_AUDIO_CACHE_MAX = 2
_MOSS_URL_AUDIO = {}          # URL -> local 16 kHz mono WAV made by ffmpeg


class _MossProgress(_HFLogitsProcessor):
    """Status-line progress for one MOSS-Music generation (MOSS_SHOW_PROGRESS).
    Reads the step count only; never changes a score."""

    def __init__(self, prompt_len, max_new_tokens):
        self.prompt_len = int(prompt_len)
        self.max_new = int(max_new_tokens or 0)
        self.t0 = time.perf_counter()
        self.t_first = None
        self.last = 0.0

    def __call__(self, input_ids, scores):
        try:
            now = time.perf_counter()
            if self.t_first is None:
                self.t_first = now         # first call = the prompt and audio have been read
            if now - self.last >= 1.0:
                self.last = now
                n = int(input_ids.shape[1]) - self.prompt_len
                rate = n / (now - self.t_first) if n > 0 and now > self.t_first else 0.0
                _status_suffix(f"read in {self.t_first - self.t0:.0f}s \u00b7 {n}/{self.max_new} tokens"
                               + (f" \u00b7 {rate:.1f}/s" if rate else ""))
        except Exception:
            pass
        return scores


# 0.7-pre64 BATCH_STALL_RETRY: a batch child reports, through the file the batch names
# in MUSICLYSE_STALL_FILE, when the listening model is generating and that it is still
# producing tokens. Outside a batch nothing is written. Never changes a score.
_STALL_FILE = os.environ.get("MUSICLYSE_STALL_FILE") or ""
_STALL_LAST = [0.0]


def _stall_mark(state, force=False):
    if not _STALL_FILE:
        return
    now = time.time()
    if not force and now - _STALL_LAST[0] < 2.0:
        return
    _STALL_LAST[0] = now
    try:
        temp = _STALL_FILE + ".tmp"
        with open(temp, "w", encoding="utf-8") as fh:
            fh.write(f"{state} {now:.3f}")
        os.replace(temp, _STALL_FILE)
    except OSError:
        pass


class _MossStallBeat(_HFLogitsProcessor):
    """Marks each generated token for the batch's stall watchdog. Reads nothing; never
    changes a score."""

    def __call__(self, input_ids, scores):
        _stall_mark("gen")
        return scores


def _moss_mps_trim():
    """Release PyTorch's cached-but-unused GPU blocks (Apple GPUs, main
    thread only). Never raises."""
    try:
        if torch.backends.mps.is_available() and _gpu_cache_clear_ok():
            torch.mps.empty_cache()
    except Exception:
        pass


class _MossMpsTrim(_HFLogitsProcessor):
    """Calls _moss_mps_trim() every `every` generated tokens
    (MOSS_MPS_CACHE_TRIM_EVERY). Reads the step count only."""

    def __init__(self, prompt_len, every):
        self.prompt_len = int(prompt_len)
        self.every = max(1, int(every))
        self.last = 0

    def __call__(self, input_ids, scores):
        try:
            n = int(input_ids.shape[1]) - self.prompt_len
            if n - self.last >= self.every:
                self.last = n
                _moss_mps_trim()
        except Exception:
            pass
        return scores


def _moss_tokenizer(processor):
    """The text tokenizer behind MOSS-Music's processor."""
    for attr in ("tokenizer", "_base_tokenizer"):
        tok = getattr(processor, attr, None)
        if tok is not None:
            return tok
    return processor


def _moss_sample_rate(processor):
    cfg = getattr(processor, "config", None) or getattr(processor, "mel_config", None)
    try:
        sr = int(getattr(cfg, "mel_sr", 0) or 0)
    except Exception:
        sr = 0
    return sr if sr > 0 else 16000


def _moss_load_audio(ref, sample_rate):
    """Decode one clip for MOSS-Music: mono float32 at the processor's rate,
    cut at MOSS_MAX_AUDIO_SECONDS. URLs are fetched once with ffmpeg (the
    processor reads arrays, not paths or URLs). The last couple of clips are
    kept decoded, because one track goes through several passes in a row."""
    path = str(ref)
    if path.startswith(("http://", "https://")):
        local = _MOSS_URL_AUDIO.get(path)
        if not local or not os.path.exists(local):
            local = convert_to_wav(path, sample_rate=sample_rate)
            _MOSS_URL_AUDIO[path] = local
            _MOSS_RETRY_TEMP_FILES.append(local)   # removed with the other temp copies
        path = local
    cap = float(MOSS_MAX_AUDIO_SECONDS) if MOSS_MAX_AUDIO_SECONDS else None
    try:
        st = os.stat(path)
        key = (os.path.abspath(path), st.st_mtime_ns, st.st_size, int(sample_rate), cap)
    except OSError:
        key = None
    if key is not None and key in _MOSS_AUDIO_CACHE:
        audio = _MOSS_AUDIO_CACHE.pop(key)
        _MOSS_AUDIO_CACHE[key] = audio
        return audio
    audio, _sr = librosa.load(path, sr=int(sample_rate), mono=True,
                              duration=cap if cap and cap > 0 else None)
    audio = np.ascontiguousarray(audio, dtype=np.float32)
    if audio.size == 0:
        raise ValueError(f"no audio samples could be read from {os.path.basename(path)}")
    if key is not None:
        _MOSS_AUDIO_CACHE[key] = audio
        while len(_MOSS_AUDIO_CACHE) > _MOSS_AUDIO_CACHE_MAX:
            _MOSS_AUDIO_CACHE.pop(next(iter(_MOSS_AUDIO_CACHE)))
    return audio


def _moss_build_prompt(conversation):
    """(ChatML prompt string, [audio refs in order]) for a conversation."""
    msgs = [m for m in (conversation or []) if isinstance(m, dict)]
    parts, audio_refs = [], []
    if not msgs or str(msgs[0].get("role") or "") != "system":
        parts.append(f"<|im_start|>system\n{_MOSS_DEFAULT_SYSTEM}<|im_end|>\n")
    for m in msgs:
        role = str(m.get("role") or "user")
        content = m.get("content")
        items = [{"type": "text", "text": content}] if isinstance(content, str) else list(content or [])
        texts, n_audio = [], 0
        for it in items:
            if not isinstance(it, dict):
                texts.append(str(it))
                continue
            if it.get("type") == "audio":
                ref = it.get("path") or it.get("audio") or it.get("url")
                if ref is not None:
                    audio_refs.append(ref)
                    n_audio += 1
            elif it.get("text") is not None:
                texts.append(str(it.get("text")))
        body = (_MOSS_AUDIO_SPAN + "\n") * n_audio + "\n".join(texts)
        parts.append(f"<|im_start|>{role}\n{body}<|im_end|>\n")
    parts.append("<|im_start|>assistant\n")
    return "".join(parts), audio_refs


def _moss_stop_token_ids(model, tok):
    """Token ids that end a reply: <|im_end|> first (the ChatML end of turn),
    then whatever the model's generation config lists."""
    ids = []
    try:
        im_end = tok.convert_tokens_to_ids("<|im_end|>")
        unk = getattr(tok, "unk_token_id", None)
        if isinstance(im_end, int) and im_end >= 0 and im_end != unk:
            ids.append(im_end)
    except Exception:
        pass
    try:
        e = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
        for x in (e if isinstance(e, (list, tuple)) else [e]):
            if x is not None and int(x) not in ids:
                ids.append(int(x))
    except Exception:
        pass
    return ids


def _moss_strip_think(text):
    """Drop a reasoning block if one appears (the Instruct model should not
    write one, but its Qwen3 base can)."""
    t = str(text or "")
    t = re.sub(r"^\s*<think>.*?</think>\s*", "", t, flags=re.S)
    t = re.sub(r"^\s*</think>\s*", "", t)
    return t


def _moss_prepare_inputs(model, processor, conversation):
    """Prompt + audio -> model inputs on the model's device."""
    prompt, audio_refs = _moss_build_prompt(conversation)
    tok = _moss_tokenizer(processor)
    if audio_refs:
        sr = _moss_sample_rate(processor)
        audios = [_moss_load_audio(r, sr) for r in audio_refs]
        raw = processor(text=prompt, audios=audios, return_tensors="pt")
    else:
        # Text-only (no listening pass sends this today): tokenize directly,
        # because the processor wraps a span-less prompt in its own template.
        raw = tok(prompt, add_special_tokens=False, return_tensors="pt")
    inputs = {}
    for k, v in dict(raw).items():
        inputs[k] = v.to(model.device) if hasattr(v, "to") else v
    if inputs.get("audio_data") is not None:
        inputs["audio_data"] = inputs["audio_data"].to(model.dtype)
    return inputs


class _MossDecoder(_HFLogitsProcessor):
    """The final word choice for one MOSS-Music reply (runs after the loop
    guard and any other score processor):

    * guided fields (MOSS_GUIDED_FIELDS): at the start of every line, while
      template fields remain, the next field name ("LABEL=") is written in
      the template's order; the model writes only the values. An end-of-reply
      before the last field becomes a line break; a value longer than
      MOSS_FIELD_MAX_TOKENS is ended with a line break; after the last field's
      line the reply ends.
    * the word itself: greedy (highest score) or sampled on the CPU with
      temperature / top-k / top-p from a seeded generator (MOSS_DECODING).

    It returns scores with only the chosen token left, so generate() runs with
    do_sample=False and simply takes that token."""

    def __init__(self, tok, prompt_len, stop_ids, fields=None, prefill="", sample=False,
                 temperature=0.7, top_p=0.8, top_k=20, seed=1234, field_max_tokens=300,
                 max_new_tokens=None, no_copy=False):
        self.tok = tok
        self.prompt_len = int(prompt_len)
        self.stop = set(int(x) for x in (stop_ids or []))
        self.eos = int(stop_ids[0]) if stop_ids else None
        self.nl = None
        try:
            ids = tok("\n", add_special_tokens=False)["input_ids"]
            ids = ids.tolist()[0] if hasattr(ids, "tolist") else ids
            if isinstance(ids, (list, tuple)) and len(ids) == 1:
                self.nl = int(ids[0])
        except Exception:
            self.nl = None
        self.fields = list(fields or []) if self.nl is not None else []
        self.idx = 0
        if self.fields and prefill:
            for i, f in enumerate(self.fields):
                if re.search(r"(?m)^[ \t]*" + re.escape(f) + r"[ \t]*=", prefill):
                    self.idx = i + 1
        self.at_line_start = (not prefill) or str(prefill).endswith("\n")
        self.line_tokens = 0
        self.queue = []
        self.sample = bool(sample)
        self.temperature = float(temperature or 1.0)
        self.top_p = float(top_p) if top_p is not None else 1.0
        self.top_k = int(top_k) if top_k else 0
        self.gen = torch.Generator(device="cpu")
        self.gen.manual_seed(int(seed))
        self.field_max = int(field_max_tokens or 0)
        self.max_new = int(max_new_tokens) if max_new_tokens else None
        self.n_seen = 0
        self.labels_written = 0
        self.ends_turned_into_newlines = 0
        self._nl_cache = {}
        # MOSS_NO_FIELD_COPYING: token ids of each finished field value, the
        # value being written now, and label tokens still to pass by.
        self.no_copy = bool(no_copy) and bool(self.fields)
        self.values = []
        self.cur = []
        self.skip = 0
        self.copies_blocked = 0

    def _has_nl(self, t):
        v = self._nl_cache.get(t)
        if v is None:
            try:
                v = "\n" in self.tok.decode([t])
            except Exception:
                v = False
            self._nl_cache[t] = v
        return v

    def _pick(self, scores):
        row = scores[0]
        if not self.sample:
            return int(row.argmax())
        logits = row.detach().float().cpu()
        finite = torch.isfinite(logits)
        n_ok = int(finite.sum())
        if n_ok <= 1:
            return int(row.argmax())
        # Candidates: the top-k scores (all finite ones when top_k is off),
        # highest first -- only these can be drawn, so only these are sorted.
        k = self.top_k if (self.top_k and self.top_k < n_ok) else n_ok
        vals, idx = torch.topk(logits, k)
        probs = torch.softmax(vals / max(self.temperature, 1e-5), dim=-1)
        if self.top_p < 1.0:
            cum = probs.cumsum(0)
            probs = probs.masked_fill((cum - probs) > self.top_p, 0.0)   # always keeps the first
        total = float(probs.sum())
        if not total > 0:
            return int(row.argmax())
        return int(idx[int(torch.multinomial(probs / total, 1, generator=self.gen))])

    def _value_cap(self):
        """Tokens the current field's value may use: MOSS_FIELD_MAX_TOKENS, but
        never more than its fair share of what is left of the token budget,
        so the last fields (in the vocal pass, the final LEAD_CATEGORY /
        LEAD_PROFILE tags the parsers rely on) are always reached."""
        cap = self.field_max
        if self.max_new and self.fields:
            to_come = max(0, len(self.fields) - self.idx)          # labels still to write
            left = to_come + 1                                      # this value + those fields
            budget = self.max_new - self.n_seen + self.line_tokens - 6 * to_come
            cap = min(cap, max(16, budget // left))
        return cap

    # A value is a copy once its first _COPY_MIN_MATCH tokens repeat the start
    # of an earlier value of at least _COPY_MIN_VALUE tokens.
    _COPY_MIN_MATCH = 4
    _COPY_MIN_VALUE = 6

    def _copy_continuations(self):
        """Tokens that would continue copying an earlier field's value into
        the value being written (MOSS_NO_FIELD_COPYING), else an empty set."""
        k = len(self.cur)
        if k < self._COPY_MIN_MATCH or not self.values or self.at_line_start:
            return set()
        out = set()
        for v in self.values:
            if len(v) >= self._COPY_MIN_VALUE and len(v) > k and v[:k] == self.cur:
                out.add(int(v[k]))
        if self.nl is not None:
            out.discard(self.nl)
        return out

    def __call__(self, input_ids, scores):
        try:
            n = int(input_ids.shape[1]) - self.prompt_len
            while self.n_seen < n:
                t = int(input_ids[0, self.prompt_len + self.n_seen])
                self.n_seen += 1
                if self._has_nl(t):
                    self.at_line_start, self.line_tokens = True, 0
                else:
                    self.at_line_start = False
                    self.line_tokens += 1
                if self.no_copy:
                    if self.skip > 0:
                        self.skip -= 1
                    elif self._has_nl(t):
                        if self.cur:
                            self.values.append(self.cur)
                        self.cur = []
                    else:
                        self.cur.append(t)
            choice = None
            if self.queue:
                choice = self.queue.pop(0)
            elif self.fields and self.at_line_start:
                if self.idx < len(self.fields):
                    ids = self.tok.encode(self.fields[self.idx] + "=", add_special_tokens=False)
                    self.idx += 1
                    self.labels_written += 1
                    choice, self.queue = int(ids[0]), [int(x) for x in ids[1:]]
                    if self.no_copy:
                        if self.cur:
                            self.values.append(self.cur)
                        self.cur = []
                        self.skip += len(ids)
                elif self.eos is not None and (self.n_seen > 0 or self.idx > 0):
                    choice = self.eos
            if choice is None:
                if self.fields and self.field_max and self.line_tokens >= self._value_cap():
                    choice = self.nl
                else:
                    banned = self._copy_continuations() if self.no_copy else ()
                    if banned:
                        scores = scores.clone()
                        scores[:, list(banned)] = float("-inf")
                        self.copies_blocked += 1
                    choice = self._pick(scores)
                    if choice in self.stop and self.fields and self.idx < len(self.fields):
                        choice = self.nl
                        self.ends_turned_into_newlines += 1
            out = torch.full_like(scores, float("-inf"))
            out[:, choice] = 0.0
            return out
        except Exception:
            return scores


def _moss_template_labels(prompt, after=None):
    """Field names (LABEL=...) in a prompt's output template, in order, each
    once. `after`: only look past the last occurrence of this heading."""
    t = str(prompt or "")
    if after and after in t:
        t = t[t.rfind(after):]
    out = []
    for lab in re.findall(r"(?m)^([A-Z][A-Z0-9_]{2,})=", t):
        if lab not in out:
            out.append(lab)
    return out


def _moss_template_fields(conversation):
    """The output template a conversation asks for, or [] for free text.
    Recognised by which of the script's own prompts it carries: the last user
    message decides (a self-check follows the main report in one
    conversation), then a system message (the empty-pass retry)."""
    def texts(m):
        c = m.get("content") if isinstance(m, dict) else None
        if isinstance(c, str):
            return c
        return "\n".join(str(x.get("text") or "") for x in (c or []) if isinstance(x, dict))
    msgs = [m for m in (conversation or []) if isinstance(m, dict)]
    last_user = next((texts(m) for m in reversed(msgs) if m.get("role") == "user"), "")
    system = "\n".join(texts(m) for m in msgs if m.get("role") == "system")
    main = list(_MOSS_CANONICAL_FIELDS)
    deep = _moss_template_labels(MOSS_DEEP_MODE_ADDENDUM)

    def main_fields(t):
        # DESCRIPTION last: the free description the report's DESCRIPTION
        # field and the fill-from-description step are built from (with the
        # reply ending after the last field, there was otherwise none).
        return (main + ([f for f in deep if f not in main] if MOSS_DEEP_MODE_ADDENDUM.strip() in t else [])
                + ["DESCRIPTION"])
    if SELF_CHECK_PROMPT.strip()[:400] in last_user:
        return main_fields(last_user + system)
    if VOCAL_PRESENCE_VERIFY_PROMPT.strip()[:400] in last_user:
        return _moss_template_labels(VOCAL_PRESENCE_VERIFY_PROMPT)
    if SECTION_VOICE_CLIP_PROMPT.strip()[:400] in last_user:
        return _moss_template_labels(SECTION_VOICE_CLIP_PROMPT)
    if SECTION_VOICE_PAIR_PROMPT.strip()[:400] in last_user:
        return _moss_template_labels(SECTION_VOICE_PAIR_PROMPT)
    if VOCAL_CONFIRMATION_PROMPT.strip()[:400] in last_user:
        return _moss_template_labels(VOCAL_CONFIRMATION_PROMPT)
    if VOCAL_ANALYSIS_PROMPT.strip()[:400] in last_user:
        return _moss_template_labels(VOCAL_ANALYSIS_PROMPT, after="OUTPUT\n")
    if ERA_ANALYSIS_PROMPT.strip()[:400] in last_user:
        return _moss_template_labels(ERA_ANALYSIS_PROMPT)
    if _MOSS_RECOVERY_PROMPT.strip()[:400] in last_user:
        return _moss_template_labels(_MOSS_RECOVERY_PROMPT)
    if MOSS_FULL_ANALYSIS_PROMPT.strip()[:400] in (last_user + "\n" + system):
        return main_fields(last_user + system)
    # 0.7-pre3: the listening-details passes (their fields are in the prompt sent).
    if last_user.startswith((_LISTENING_DETAILS_HEAD, _LISTENING_SECTION_HEAD)):
        return _moss_template_labels(last_user)
    return []


def moss_generate(
    model, processor, conversation,
    max_new_tokens: int = 2048, do_sample: bool = None, repetition_penalty: float = 1.0,
    no_repeat_ngram_size: int = 0, temperature: float = None, top_p: float = None,
    loop_guard: bool = True, prefill: str = None, top_k: int = None, fields=None,
):
    # do_sample / temperature / top_p / top_k left as None use MOSS_DECODING
    # and its settings. fields=None picks the output template from the
    # conversation (MOSS_GUIDED_FIELDS); fields=False or [] turns it off.
    # loop_guard defaults to on for every MOSS-Music pass (it used to be on
    # for the main report only): it changes nothing unless a line repeats a
    # block 8+ times back to back, and otherwise a runaway line -- e.g. a
    # field label looping "DETECTED_DETECTED_..." in the vocal pass -- runs to
    # the token limit, losing the remaining fields and minutes of time.
    inputs = _moss_prepare_inputs(model, processor, conversation)
    if prefill:
        # dev60: start the reply with this text (see MOSS_CONSISTENT_REPORT).
        inputs = _moss_append_prefill(inputs, processor, prefill)
    if inputs.get("audio_data") is not None and inputs.get("audio_input_mask") is None:
        # Which prompt positions the encoded audio is written into.
        inputs["audio_input_mask"] = inputs["input_ids"] == int(
            getattr(processor, "audio_token_id", None) or _MOSS_AUDIO_TOKEN_ID_DEFAULT)

    tok = _moss_tokenizer(processor)
    stop_ids = _moss_stop_token_ids(model, tok)
    if do_sample is None:
        do_sample = str(MOSS_DECODING).strip().lower() == "sampled"
    if fields is None:
        fields = _moss_template_fields(conversation) if MOSS_GUIDED_FIELDS else []
    fields = list(fields or [])
    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": False,          # the word is chosen by _MossDecoder (on the CPU when sampling)
        "repetition_penalty": repetition_penalty,
        "use_cache": True,
    }
    if stop_ids:
        gen_kwargs["eos_token_id"] = stop_ids
        pad = getattr(tok, "pad_token_id", None)
        gen_kwargs["pad_token_id"] = int(pad) if pad is not None else stop_ids[0]
    if no_repeat_ngram_size and no_repeat_ngram_size > 0:
        gen_kwargs["no_repeat_ngram_size"] = int(no_repeat_ngram_size)
    decoder = _MossDecoder(
        tok, inputs["input_ids"].shape[1], stop_ids, fields=fields, prefill=prefill or "",
        sample=bool(do_sample),
        temperature=MOSS_TEMPERATURE if temperature is None else temperature,
        top_p=MOSS_TOP_P if top_p is None else top_p,
        top_k=MOSS_TOP_K if top_k is None else top_k,
        seed=MOSS_SEED, field_max_tokens=MOSS_FIELD_MAX_TOKENS, max_new_tokens=max_new_tokens,
        no_copy=bool(MOSS_NO_FIELD_COPYING and "GENRE_RANKED" in fields),
    )

    guard = None
    procs = []
    if loop_guard and MOSS_LOOP_GUARD:
        try:
            guard = _MossLoopGuard(tok, inputs["input_ids"].shape[1],
                                   numbered=bool(MOSS_CONSISTENT_REPORT),
                                   eos_id=stop_ids[0] if stop_ids else None)
            if guard.nl is not None:
                procs.append(guard)
            else:
                guard = None
        except Exception:
            guard = None
    if MOSS_SHOW_PROGRESS:
        procs.append(_MossProgress(inputs["input_ids"].shape[1], max_new_tokens))
    _trim_every = int(MOSS_MPS_CACHE_TRIM_EVERY or 0)
    if _trim_every > 0 and str(getattr(model, "device", "")).startswith("mps"):
        procs.append(_MossMpsTrim(inputs["input_ids"].shape[1], _trim_every))
    if _STALL_FILE:
        procs.append(_MossStallBeat())   # pre64: batch stall watchdog (reads nothing)
    procs.append(decoder)            # last: makes the final choice
    if procs:
        try:
            from transformers import LogitsProcessorList
            gen_kwargs["logits_processor"] = LogitsProcessorList(procs)
        except Exception:
            guard = None

    _stall_mark("gen", force=True)
    try:
        with torch.inference_mode():
            gen_ids = model.generate(**inputs, **gen_kwargs)
    finally:
        _stall_mark("idle", force=True)
    new_tokens = gen_ids[:, inputs["input_ids"].shape[1]:]
    if _trim_every > 0:
        del inputs
        _moss_mps_trim()
    if guard is not None and guard.fires:
        print(f"  (the listening model started repeating itself; cut {guard.fires} runaway "
              f"line{'s' if guard.fires != 1 else ''} so its report could continue)")
    if getattr(decoder, "copies_blocked", 0):
        print("  (the listening model started copying one field's answer into the next; "
              "it was made to write its own answer for each field)")
    try:
        out = processor.batch_decode(new_tokens, skip_special_tokens=True)[0]
    except Exception:
        out = tok.batch_decode(new_tokens, skip_special_tokens=True)[0]
    out = _moss_strip_think(out)
    if prefill:
        return (prefill + out).strip()
    return out.strip()


def _moss_append_prefill(inputs, processor, prefill):
    """dev60: append the prefill text's tokens to the prompt (input_ids,
    attention_mask and, when present, the audio-position mask), so the model
    continues from it."""
    import torch as _t
    tok = _moss_tokenizer(processor)
    ids = tok(prefill, add_special_tokens=False, return_tensors="pt")["input_ids"]
    ids = ids.to(inputs["input_ids"].device, dtype=inputs["input_ids"].dtype)
    inputs["input_ids"] = _t.cat([inputs["input_ids"], ids], dim=1)
    if "attention_mask" in inputs and inputs["attention_mask"] is not None:
        am = inputs["attention_mask"]
        inputs["attention_mask"] = _t.cat([am, _t.ones((am.shape[0], ids.shape[1]), dtype=am.dtype,
                                                        device=am.device)], dim=1)
    if inputs.get("audio_input_mask") is not None:
        aim = inputs["audio_input_mask"]
        inputs["audio_input_mask"] = _t.cat([aim, _t.zeros((aim.shape[0], ids.shape[1]), dtype=aim.dtype,
                                                            device=aim.device)], dim=1)
    return inputs


_MOSS_FIELDS_PREFILL = "GENRE_RANKED=1)"


def _moss_self_check_generate(moss_model, moss_processor, moss_conversation, max_new_tokens):
    """dev60: the self-check reply, in the field format and with the
    runaway-line guard when MOSS_CONSISTENT_REPORT is on (falling back to the
    plain call when even that reply has no fields); exactly the old call
    otherwise."""
    if not MOSS_CONSISTENT_REPORT:
        return moss_generate(moss_model, moss_processor, moss_conversation, max_new_tokens=max_new_tokens)
    first = ""
    for m in reversed(moss_conversation or []):
        if isinstance(m, dict) and m.get("role") == "assistant":
            c = m.get("content")
            if isinstance(c, list):
                first = "\n".join(x.get("text", "") for x in c if isinstance(x, dict))
            else:
                first = str(c or "")
            break
    out = ""
    try:
        out = moss_generate(moss_model, moss_processor, moss_conversation, max_new_tokens=max_new_tokens,
                          prefill=_MOSS_FIELDS_PREFILL, loop_guard=True)
    except Exception as e:
        print(f"  (field-format self-check unavailable ({type(e).__name__}); using the plain one)")
        out = ""
    if not out or not extract_genre_ranked_labels(_moss_normalize_report_fields(out)):
        if out:
            print("  (the listening model's self-check did not keep the field format; asking again "
                  "without the field prompt)")
        out = moss_generate(moss_model, moss_processor, moss_conversation, max_new_tokens=max_new_tokens,
                          loop_guard=True)
    return _moss_canonical_report(out, first)


# dev63: the fixed report layout (MOSS_CONSISTENT_REPORT): the fields the main
# analysis prompt asks for, in its order.
_MOSS_CANONICAL_FIELDS = ("GENRE_RANKED", "GENRE_ADJACENT", "GENRE_RULED_OUT", "KEY", "TEMPO_BPM", "CHORDS",
                        "STRUCTURE", "INSTRUMENTATION", "DRUMS", "TIMBRE", "MOOD_VIBE", "MELODICISM",
                        "MELODIC_CONTOUR", "MELODIC_RANGE", "VOCALS_PRESENT", "LEAD_VOCAL_CHARACTERISTICS",
                        "LYRICS_PRESENT", "LYRIC_SUBJECT")
_MOSS_FIELD_LINE_RE = re.compile(r"^([A-Z][A-Z0-9_]{2,})=(.*)$")


def _moss_parse_report(text):
    """dev63: (fields, prose) from the listening model's report head:
    fields is an ordered dict NAME -> value (multi-line values such as
    STRUCTURE=[...] kept whole; the first occurrence of a name wins), prose
    the lines that belong to no field."""
    fields, prose = {}, []
    cur = None
    open_br = 0
    for ln in str(text or "").split("\n"):
        m = _MOSS_FIELD_LINE_RE.match(ln)
        if m and open_br <= 0:
            name, val = m.group(1), m.group(2)
            cur = name if name not in fields else None
            if cur:
                fields[cur] = val
            open_br = val.count("[") - val.count("]")
            continue
        if cur and (open_br > 0 or ln[:1] in (" ", "\t", "]") or ln.strip().startswith("]")):
            fields[cur] += "\n" + ln
            open_br += ln.count("[") - ln.count("]")
            continue
        if open_br > 0 and cur is None:
            open_br += ln.count("[") - ln.count("]")
            continue
        cur = None
        open_br = 0
        if ln.strip() and not ln.strip().startswith("(fields above from the listening model"):
            prose.append(ln.strip())
    return fields, prose


_MOSS_COPY_RUN_MIN = 3        # fields in a row ...
_MOSS_COPY_KEY_CHARS = 12     # ... whose values start with the same 12+ characters


def _moss_copied_fields(items):
    """Names of report fields whose value is a copy of the field before or
    after it: runs of _MOSS_COPY_RUN_MIN or more consecutive fields whose
    values share their first _MOSS_COPY_KEY_CHARS characters (case and
    spacing aside). `items` is [(name, value), ...] in reply order. Short
    values ("yes", "uncertain") never count. This is the copy loop
    MOSS_NO_FIELD_COPYING prevents in new scans ("none (0.00s-13.92s)" in
    every field from INSTRUMENTATION on); such a run is not twelve answers,
    it is one answer pasted twelve times."""
    keys = []
    for name, value in items or []:
        v = re.sub(r"\s+", " ", str(value or "")).strip().lower()
        keys.append((name, v[:_MOSS_COPY_KEY_CHARS] if len(v) >= _MOSS_COPY_KEY_CHARS else None))
    out = set()
    i = 0
    while i < len(keys):
        j = i
        while j + 1 < len(keys) and keys[i][1] is not None and keys[j + 1][1] == keys[i][1]:
            j += 1
        if keys[i][1] is not None and j - i + 1 >= _MOSS_COPY_RUN_MIN:
            out.update(n for n, _k in keys[i:j + 1])
        i = j + 1
    return out


# A field answer saying the model heard nothing at all ("none (no audio
# detected)", "none (no instruments detected)", "no audible content"). One
# such answer can be honest ("DRUMS=none (no drums)" does not match); three
# or more in one reply are the model losing the audio, the same failure as a
# copied run, written in different words -- which is what MOSS-Music did on
# a folk test track once it could no longer paste one answer everywhere.
_MOSS_NO_CONTENT_RE = re.compile(
    r"(?i)\bno (?:audio|audible (?:content|sound|music)|sound|music|musical content|content|"
    r"instruments?|harmonic (?:progression|content)|melod(?:y|ic content)|lyrics?|vocals?)\b"
    r"[^\n]{0,40}?\b(?:detected|present|audible|discernible|identified|heard)\b"
    r"|\bno audio\b|\bno audible content\b")
_MOSS_NO_CONTENT_MIN = 3


def _moss_no_content_fields(items):
    """Names of fields answering "nothing detected" (_MOSS_NO_CONTENT_RE),
    when _MOSS_NO_CONTENT_MIN or more do so in one reply; else empty."""
    hits = {n for n, v in (items or []) if _MOSS_NO_CONTENT_RE.search(str(v or ""))}
    return hits if len(hits) >= _MOSS_NO_CONTENT_MIN else set()


def _moss_drop_copied_field_lines(text):
    """The listening model's report without the lines of a copied-field run
    (see _moss_copied_fields); everything after the report is kept."""
    t = str(text or "")
    cut = _moss_report_head_end(t)
    head, tail = t[:cut], t[cut:]
    items = re.findall(r"(?m)^([A-Z][A-Z0-9_]{2,})=(.*)$", head)
    bad = _moss_copied_fields(items) | _moss_no_content_fields(items)
    if not bad:
        return t
    head = "\n".join(ln for ln in head.split("\n")
                     if not (re.match(r"^([A-Z][A-Z0-9_]{2,})=", ln)
                             and re.match(r"^([A-Z][A-Z0-9_]{2,})=", ln).group(1) in bad))
    return head + tail


def _moss_canonical_report(reply, first_pass):
    """dev63: the listening model's self-check reply in one fixed layout:
    every field of _MOSS_CANONICAL_FIELDS in that order -- from the reply, else
    from the first pass, else "uncertain" -- then any other fields it gave,
    then DESCRIPTION= (its prose, or "(none given)"), then REPORT_NOTE= saying
    which fields came from the first pass or were not given. Field values are
    the model's own (repeats dropped, see _moss_tidy_report_lists); everything
    after the report (lyrics, measurements ...) is kept as it was."""
    r = _moss_normalize_report_fields(str(reply or ""))
    f = _moss_normalize_report_fields(str(first_pass or ""))
    cut = _moss_report_head_end(r)
    head, tail = r[:cut], r[cut:]
    rf, rprose = _moss_parse_report(head)
    ff, _fprose = _moss_parse_report(f[:_moss_report_head_end(f)])
    # A copied-field run (one answer pasted into several fields) counts as
    # not given: the first pass's own value is used instead, as for a field
    # the reply left out.
    copied_r = _moss_copied_fields(list(rf.items())) | _moss_no_content_fields(list(rf.items()))
    copied_f = _moss_copied_fields(list(ff.items())) | _moss_no_content_fields(list(ff.items()))
    rf = {k: ("" if k in copied_r else v) for k, v in rf.items()}
    ff = {k: ("" if k in copied_f else v) for k, v in ff.items()}

    def _given(v):
        return bool(str(v or "").strip())

    lines, from_first, missing = [], [], []
    for n in _MOSS_CANONICAL_FIELDS:
        if _given(rf.get(n)):
            lines.append(f"{n}={rf[n]}")
        elif _given(ff.get(n)):
            lines.append(f"{n}={ff[n]}")
            from_first.append(n)
        else:
            # STRUCTURE in brackets, so _overwrite_structure_field always
            # finds this slot for the measured section map.
            lines.append(f"{n}=[uncertain]" if n == "STRUCTURE" else f"{n}=uncertain")
            missing.append(n)
    for n, v in rf.items():
        if n not in _MOSS_CANONICAL_FIELDS and n not in ("DESCRIPTION", "REPORT_NOTE") and _given(v):
            lines.append(f"{n}={v}")
    desc = " ".join(x for x in ([rf.get("DESCRIPTION", "").strip()] + rprose) if x).strip()
    if not desc and "DESCRIPTION" in copied_r and _given(ff.get("DESCRIPTION")):
        desc = str(ff["DESCRIPTION"]).strip()
    lines.append("DESCRIPTION=" + (desc if desc else "(none given)"))
    note = []
    if copied_r:
        note.append("the self-check reply pasted one answer (or a 'nothing detected' answer) into several fields ("
                    + ", ".join(n for n in list(_MOSS_CANONICAL_FIELDS) + ["DESCRIPTION"] if n in copied_r)
                    + "); those are taken as not given")
    if from_first:
        note.append("from the listening model's first pass (its self-check reply left them out): "
                    + ", ".join(from_first))
    if missing:
        note.append("not given by the listening model: " + ", ".join(missing))
    lines.append("REPORT_NOTE=" + ("; ".join(note) if note else "all fields from the listening model's self-check"))
    return _moss_tidy_report_lists("\n".join(lines)) + ("\n\n" + tail.lstrip("\n") if tail.strip() else "")


def _moss_list_key(item):
    return re.sub(r"[\s\-\u2010-\u2015]+", " ", re.sub(r"\s*\(confidence:[^)]*\)", "", item,
                                                       flags=re.I)).strip(" ;,.").lower()


def _moss_tidy_report_lists(text):
    """dev62: in the listening model's own report, drop exact repeats from
    its list fields -- GENRE_RANKED (renumbered), GENRE_ADJACENT,
    GENRE_RULED_OUT, INSTRUMENTATION -- and shorten a CHORDS line whose tail
    repeats the same short cycle three or more times to one cycle marked as
    repeating. A loop the guard cut early leaves exactly these repeats. No
    item is added or reworded (MOSS_CONSISTENT_REPORT only)."""
    t = str(text or "")
    if not t:
        return t
    cut = _moss_report_head_end(t)
    head, tail = t[:cut], t[cut:]
    out = []
    for ln in head.split("\n"):
        m = re.match(r"^(GENRE_RANKED|GENRE_ADJACENT|GENRE_RULED_OUT|INSTRUMENTATION|CHORDS)=(.*)$", ln)
        if not m:
            out.append(ln)
            continue
        name, body = m.group(1), m.group(2)
        note = ""
        nm = re.search(r"\s*(\((?:collapsed|recurring)[^)]*\))\s*$", body)
        if nm:
            note, body = " " + nm.group(1), body[:nm.start()]
        if name == "GENRE_RANKED":
            items = [it.strip(" ;") for it in re.findall(r"\d+\)\s*([^;\n]+)", body)]
            items = [it for it in items if it]
            seen, keep = set(), []
            for it in items:
                k = _moss_list_key(it)
                if k and k not in seen:
                    seen.add(k)
                    keep.append(it)
            if items and len(keep) < len(items):
                body = "; ".join(f"{i + 1}) {it}" for i, it in enumerate(keep))
        elif name == "CHORDS":
            toks = [x.strip() for x in re.split(r"\s+[\u2013\u2014\-\u2192>]+\s+", body.strip()) if x.strip()]
            done = False
            for p_ in range(1, 7):
                if done or len(toks) < 3 * p_:
                    continue
                cyc = toks[-p_:]
                reps = 1
                while len(toks) >= (reps + 1) * p_ and toks[-(reps + 1) * p_:-reps * p_] == cyc:
                    reps += 1
                if reps >= 3 and len(set(cyc)) == len(cyc) and (p_ > 1 or reps >= 4):
                    keep = toks[:len(toks) - (reps - 1) * p_]
                    _dash = " \u2013 "
                    body = _dash.join(keep) + " (" + _dash.join(cyc) + " repeating)"
                    done = True
        else:
            items = [x.strip() for x in body.split(";")]
            seen, keep = set(), []
            for it in items:
                k = _moss_list_key(it)
                if not it or k in seen:
                    continue
                seen.add(k)
                keep.append(it)
            if len(keep) < len([x for x in items if x]):
                body = "; ".join(keep)
        out.append(f"{name}={body}{note}")
    return "\n".join(out) + tail


# dev61: the report's field names, one spelling each (MOSS_CONSISTENT_REPORT).
_MOSS_FIELD_ALIASES = (
    (r"genre[ _\-\u2010-\u2015]*ranked", "GENRE_RANKED"),
    (r"genre[ _\-\u2010-\u2015]*adjacent", "GENRE_ADJACENT"),
    (r"genre[ _\-\u2010-\u2015]*(?:ruled|routed|rule)[ _\-\u2010-\u2015]*out", "GENRE_RULED_OUT"),
    (r"genre[ _\-\u2010-\u2015]*routed", "GENRE_ROUTED"),
    (r"key", "KEY"),
    (r"tempo(?:[ _]?bpm)?", "TEMPO_BPM"),
    (r"time[ _]signature|meter", "TIME_SIGNATURE"),
    (r"chords?", "CHORDS"),
    (r"structure", "STRUCTURE"),
    (r"instrumentation|instruments", "INSTRUMENTATION"),
)
_MOSS_REPORT_FIELD_NAMES = ("GENRE_RANKED", "GENRE_ADJACENT", "GENRE_RULED_OUT", "KEY", "TEMPO_BPM",
                          "TIME_SIGNATURE", "CHORDS", "STRUCTURE", "INSTRUMENTATION")


def _moss_report_head_end(text):
    t = str(text or "")
    cut = len(t)
    for mk in ("\n\nLYRICS", "\n\n11. ERA / RELEASE PERIOD", "\n\nVOCAL / SINGER PROFILE",
               "[Independent signal-processing report", "\n\nRECOMMENDED "):
        k = t.find(mk)
        if 0 <= k < cut:
            cut = k
    return cut


def _moss_normalize_report_fields(text):
    """dev61: in the listening model's own report (the text before the
    pipeline's blocks), spell field names one way ("Genre Ranked=" and
    "Genre_Ranked=" become "GENRE_RANKED=", "Key:" "KEY="), turn a plain
    "Genre: a / b / c" line into a ranked list, and drop a caption-style
    "The duration of the piece is N seconds" opener (the measured duration is
    in the report). Values are not changed."""
    t = str(text or "")
    if not t:
        return t
    cut = _moss_report_head_end(t)
    head, tail = t[:cut], t[cut:]
    head = re.sub(r"^\s*The (?:duration|length) of the (?:piece|track|song|recording) is [^\n]*?\n",
                  "", head, count=1, flags=re.IGNORECASE)
    lines = head.split("\n")
    out = []
    for ln in lines:
        m = re.match(r"^([ \t*_-]*)([A-Za-z][A-Za-z _\-\u2010-\u2015]{1,30}?)[ \t*_]*[:=][ \t]*(.*)$", ln)
        if m:
            name = m.group(2).strip()
            canon = None
            for pat, cn in _MOSS_FIELD_ALIASES:
                if re.fullmatch(pat, name, flags=re.IGNORECASE):
                    canon = cn
                    break
            if canon is None and re.fullmatch(r"genres?", name, flags=re.IGNORECASE):
                pl = _genre_plain_list_line(name + ": " + m.group(3))
                if pl:
                    out.append(pl)
                    continue
            if canon:
                out.append(f"{canon}={m.group(3)}")
                continue
        out.append(ln)
    return "\n".join(out).lstrip("\n") + tail


_MOSS_EMPTY_ANALYSIS_PATTERNS = (
    r"no discernible musical material",
    r"no musical material",
    r"no tempo,\s*key",
    r"(?:music|musical|information|attributes|features|details|elements|properties) (?:can ?not|could not) be (?:extracted|determined)",
    r"unable to (?:analy[sz]e|process|hear|extract anything from) (?:the |this )?(?:audio|track|song|file|recording|clip)",
    r"does not contain (?:any )?(?:music|audio)",
    r"(?:audio|track|file|recording|clip) (?:is|appears(?: to be)?|seems(?: to be)?) (?:entirely |completely |totally )?(?:silent|empty|blank)",
    r"no audio (?:content|signal|data)",
    r"could not (?:hear|process|analy[sz]e) (?:the |this )?(?:audio|track|song|file|recording)",
    # Observed on a real scan: the listening model returned only this one-line non-answer for
    # the entire main pass (no GENRE_RANKED, no KEY, no TEMPO_BPM, nothing),
    # and it slipped through every pattern above because it says "audible or
    # textual information" rather than "music". The whole downstream genre/
    # key/tempo/instrumentation reconciliation then had no listening-model signal at all
    # for that track and silently degraded to a single raw PANNs tag with no
    # retry ever attempted.
    r"(?:provided )?(?:metadata|input|file) does not contain (?:any )?(?:audible|useful|sufficient|textual)",
)

_MOSS_RECOVERY_PROMPT = """The previous analysis claimed there was no musical material, but that is likely wrong — this is a real music track. Listen again carefully to the audio and produce the full note-style report.

Use exactly these labels (compact note style, not prose):

GENRE_RANKED=1) ... (confidence: high/medium/low); 2) ...; 3) ...
GENRE_ADJACENT=...
GENRE_RULED_OUT=...
KEY=...
TEMPO_BPM=...
CHORDS=...
STRUCTURE=[Intro 0:00-...; ...]
INSTRUMENTATION=...
TIMBRE=...
MOOD_VIBE=1) ...; 2) ...; 3) ...
MELODICISM=...
VOCALS_PRESENT=yes/no/uncertain
LEAD_VOCAL_CHARACTERISTICS=...
LYRICS_PRESENT=yes/no/uncertain
LYRIC_SUBJECT=...

Base every claim on what you hear. Do not claim the track is empty or has no musical material."""


def _moss_empty_reason(text):
    """Why the listening model's main pass looks like it failed, or "" if it didn't.
    (Common intermittent failure: 'no discernible musical material' despite
    valid audio that later passes / DSP / stems all analyse fine.)"""
    if not text or not str(text).strip():
        return "the model returned no text at all"
    t = _strip_md_emphasis(str(text).strip())
    if len(t) < 80 and not re.search(r"GENRE_RANKED|TEMPO_BPM|KEY\s*=", t, re.I):
        return f"reply was only {len(t)} characters with no analysis fields"
    low = t.lower()
    for pat in _MOSS_EMPTY_ANALYSIS_PATTERNS:
        m = re.search(pat, low, re.IGNORECASE)
        if m:
            # If it ALSO has real structured labels, it may have recovered mid-output
            labels = len(re.findall(r"(?:GENRE_RANKED|TEMPO_BPM|KEY|STRUCTURE|INSTRUMENTATION)\s*=", t, re.I))
            if labels < 2:
                return f'reply said "{m.group(0)}" and had fewer than 2 analysis fields'
    return ""


_MOSS_DESCRIPTIVE_FIELDS = (
    "INSTRUMENTATION", "TIMBRE", "MOOD_VIBE", "MELODICISM", "VOCALS_PRESENT",
    "LEAD_VOCAL_CHARACTERISTICS", "LYRIC_SUBJECT", "STRUCTURE",
)


def _moss_descriptive_field_count(text):
    """How many of the main report's descriptive fields are present (any
    case; underscores may be spaces), e.g. "Instrumentation=" or "MOOD VIBE:"."""
    t = _strip_md_emphasis(str(text or ""))
    n = 0
    for f in _MOSS_DESCRIPTIVE_FIELDS:
        # Allow a list marker or number before the label ("- Instrumentation:",
        # "1. MOOD_VIBE="), so a complete report never triggers the extra pass.
        pat = r"(?:^|\n)[ \t>*\-•\d.)]*" + f.replace("_", r"[_ ]?") + r"\s*[=:]"
        if re.search(pat, t, re.IGNORECASE):
            n += 1
    return n


# Fields the structured (first) report owns. When the extra description pass
# writes its own labelled copy of one of these, it is dropped so it cannot
# override the main report's value: extract_key_from_text takes the first
# upper-case "KEY=" anywhere in the text and extract_bpm_from_text the first
# "TEMPO=" / "TEMPO_BPM:" anywhere, so an appended "KEY=C# minor" or
# "TEMPO=120" could beat the main report's "Key=E major" / "Tempo BPM: 79".
# Spelling variants of one field share one group.
_MOSS_CORE_FIELD_GROUPS = (
    ("GENRE_RANKED", r"GENRE[ _]?RANKED"),
    ("GENRE_ADJACENT", r"GENRE[ _]?ADJACENT"),
    ("GENRE_RULED_OUT", r"GENRE[ _]?RULED?[ _]?OUT"),
    ("KEY", r"(?:MUSICAL[ _])?KEY"),
    ("TEMPO", r"TEMPO(?:[ _]?BPM)?|BPM"),
    ("CHORDS", r"CHORDS"),
)
_MOSS_FIELD_LINE_PREFIX = r"^[ \t>*\-\u2022\d.)]*"


def _moss_core_fields_present(text):
    """Groups from _MOSS_CORE_FIELD_GROUPS that `text` has as a labelled field
    (at a line start, with = or :, or anywhere as LABEL=)."""
    t = _strip_md_emphasis(str(text or ""))
    have = set()
    for name, pat in _MOSS_CORE_FIELD_GROUPS:
        if (re.search(r"(?im)" + _MOSS_FIELD_LINE_PREFIX + r"(?:" + pat + r")\**\s*[=:]", t)
                or re.search(r"(?i)(?<![A-Za-z_])(?:" + pat + r")\**\s*=", t)):
            have.add(name)
    return have


def _moss_drop_duplicate_core_fields(desc, main_text):
    """Remove from `desc` every line that labels a core field (genre lists,
    key, tempo, chords) the main report already has -- at the start of the
    line, or inline as LABEL= (or, for tempo, "tempo: <number>"). Other
    prose is kept."""
    if not desc:
        return desc
    have = _moss_core_fields_present(main_text)
    if not have:
        return desc
    drops = []
    for name, pat in _MOSS_CORE_FIELD_GROUPS:
        if name not in have:
            continue
        drops.append(re.compile(r"(?i)" + _MOSS_FIELD_LINE_PREFIX + r"(?:" + pat + r")\**\s*[=:]"))
        drops.append(re.compile(r"(?i)(?<![A-Za-z_])(?:" + pat + r")\**\s*="))
        if name == "TEMPO":
            drops.append(re.compile(
                r"(?i)\bTEMPO(?:[ _]?BPM)?\**\s*[=:]\s*\**\s*\[?\s*"
                r"(?:~|\u2248|approx(?:imately|\.)?|about|around|roughly|circa|c\.)?\s*[0-9]"
            ))
    kept = []
    for line in str(desc).splitlines():
        plain = _strip_md_emphasis(line)
        if any(rx.search(plain) for rx in drops):
            continue
        kept.append(line)
    return "\n".join(kept)


# dev40: The listening model sometimes answers the main pass as if it had been
# given only text -- "No explicit genre ... is discernible from the provided
# data. Instrumentation and production details are not available ... without
# additional metadata the precise genre ... remain undefined" (the that arena-rock test artist
# test track). None of the "no music" patterns above match that wording, so
# the retries (system-message prompt, recovery prompt, loudness-normalised
# mono audio -- the one meant for quiet, dynamic masters like that one) never
# ran, and the genre fell to a single raw PANNs tag. Two or more such phrases
# in a reply with fewer than two analysis fields now count as a failed pass
# for the retries only; the salvage scaffold is unchanged.
_MOSS_NON_ANSWER_PATTERNS = (
    r"discernible from the (?:provided|supplied|given) (?:data|information|input|metadata|text)",
    r"(?:details|information|data) (?:is|are) not (?:available|provided|given|supplied)",
    r"\b(?:are|is|were|was) not provided\b",
    r"no specific (?:description|analysis|details?) can be (?:given|provided|made)",
    r"without (?:additional|further|more|any) (?:metadata|information|data|context)",
    r"\bremains? (?:undefined|unknown|undetermined)\b",
    r"(?:can ?not|could not|can't) be (?:determined|identified|inferred|assessed) from the "
    r"(?:provided|supplied|given|available) (?:data|information|input|metadata|text)",
)


def _moss_non_answer_reason(text):
    """Why the main pass reads as a text-only non-answer, or ""."""
    if not MOSS_RETRY_ON_NON_ANSWER or not text or not str(text).strip():
        return ""
    t = _strip_md_emphasis(str(text))
    low = t.lower()
    hits = sum(1 for p in _MOSS_NON_ANSWER_PATTERNS if re.search(p, low))
    if hits < 2:
        return ""
    labels = len(re.findall(r"(?:GENRE_RANKED|TEMPO_BPM|KEY|STRUCTURE|INSTRUMENTATION)\s*=", t, re.I))
    if labels >= 2:
        return ""
    return (f"reply described the audio as unavailable ({hits} phrases such as 'not provided') "
            "and had fewer than 2 analysis fields")


def _moss_main_pass_failed(text):
    return _moss_analysis_looks_empty(text) or bool(_moss_non_answer_reason(text))


def _moss_analysis_looks_empty(text):

    return bool(_moss_empty_reason(text))


def _moss_salvage_empty_analysis(first_pass, objective_report="", essentia_report="", vocal_result=""):
    """If the main listening-model pass still claims no music but DSP/stems clearly heard a
    track, replace the failure opener with a short scaffold so the saved
    analysis and writer are not poisoned by the empty claim."""
    if not _moss_analysis_looks_empty(first_pass):
        return first_pass

    bpm = None
    for src in (objective_report, essentia_report, first_pass or ""):
        if not src:
            continue
        m = re.search(
            r"(?:estimated tempo|RECOMMENDED TEMPO|TEMPO_BPM|raw detector)\s*[:=]?\s*([0-9]+(?:\.[0-9]+)?)",
            src,
            re.I,
        )
        if m:
            try:
                bpm = float(m.group(1))
                break
            except Exception:
                pass

    vocals = "uncertain"
    if vocal_result:
        if re.search(r"VOCALS[_ ]PRESENT\s*[=:\-–—]\s*yes", vocal_result, re.I):
            vocals = "yes"
        elif re.search(r"VOCALS[_ ]PRESENT\s*[=:\-–—]\s*no", vocal_result, re.I):
            vocals = "no"

    lines = [
        "GENRE_RANKED=1) uncertain (confidence: low) — main audio-language pass failed to describe the track; rely on objective measurements and later passes below",
        f"TEMPO_BPM={round(bpm) if bpm else 'uncertain'} (from signal processing)" if bpm else "TEMPO_BPM=uncertain",
        "KEY=uncertain",
        "CHORDS=uncertain",
        "STRUCTURE=uncertain",
        "INSTRUMENTATION=uncertain — see STEM MIDI REPORT and objective measurements below",
        "TIMBRE=see objective spectral/dynamic measurements below",
        "MOOD_VIBE=1) uncertain",
        "MELODICISM=uncertain",
        f"VOCALS_PRESENT={vocals}",
        "LEAD_VOCAL_CHARACTERISTICS=see VOCAL / SINGER PROFILE below",
        "LYRICS_PRESENT=uncertain",
        "LYRIC_SUBJECT=uncertain",
        "",
        "[Note: the primary MOSS-Music full-analysis pass returned an empty/no-music "
        "response despite measurable audio activity. The structured fields above are a "
        "conservative scaffold; prefer ERA, VOCAL, objective, Essentia, and STEM MIDI "
        "sections that follow for real evidence.]",
    ]
    return "\n".join(lines)


def _moss_run_main_analysis(moss_model, moss_processor, resolved_path, deep=False):
    """Run the full listening-model analysis pass, with one recovery retry when
    the model falsely claims the audio is empty/non-musical."""
    clip_note = _moss_short_clip_note(resolved_path)
    main_prompt = clip_note + MOSS_FULL_ANALYSIS_PROMPT + (MOSS_DEEP_MODE_ADDENDUM if deep else "")
    main_max_tokens = 3072 if deep else 2048
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": main_prompt},
                {"type": "audio", "path": resolved_path},
            ],
        }
    ]
    first_pass = moss_generate(moss_model, moss_processor, conversation, max_new_tokens=main_max_tokens, loop_guard=True)

    if _moss_main_pass_failed(first_pass):
        _why = _moss_empty_reason(first_pass) or _moss_non_answer_reason(first_pass)
        _preview = " ".join(str(first_pass or "").split())[:200]
        status_done(f"Main pass looked empty ({_why}). Reply began: \"{_preview}\"")
        # First retry: the SAME full instructions, moved into a system
        # message, with a short user request next to the audio. Decoding is
        # greedy, so the same prompt and audio fail the same way every run
        # (the rock test track did so in every version since dev31, answering
        # as if it had only been given text). The short recovery prompt below
        # then "succeeded" with a free-form caption that had none of the
        # requested fields -- no GENRE_RANKED, KEY, TEMPO_BPM, CHORDS or
        # STRUCTURE -- so the track lost the listening model's genre ranking, key
        # and tempo reads. (Merely reordering the user turn's items changes
        # nothing: the model's chat template always puts the audio first and
        # joins the text.) The old recovery still runs if this fails too.
        if MOSS_EMPTY_RETRY_SYSTEM_PROMPT:
            status("Listening — main pass looked empty; retrying with the instructions as a system message...")
            sys_conversation = [
                {"role": "system", "content": [{"type": "text", "text": main_prompt}]},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": (
                            "Analyze this track and return the complete compact note-style "
                            "report, using exactly the labels and rules given in the system "
                            "instructions."
                        )},
                        {"type": "audio", "path": resolved_path},
                    ],
                },
            ]
            sys_pass = ""
            try:
                sys_pass = moss_generate(moss_model, moss_processor, sys_conversation, max_new_tokens=main_max_tokens, loop_guard=True)
            except Exception as e:
                status_done(f"System-message retry could not run ({e}); trying the short recovery prompt")
            else:
                if sys_pass and not _moss_main_pass_failed(sys_pass):
                    status_done("System-message retry succeeded (full analysis prompt)")
                    # The retry can stop early: on the rock test track it gave
                    # genre, key and tempo, then looped on the chord list and
                    # never reached INSTRUMENTATION / TIMBRE / MOOD / VOCALS.
                    # The short recovery prompt reliably produces a free-form
                    # description of exactly those things, so add it rather
                    # than lose them.
                    if MOSS_EMPTY_RETRY_ADD_DESCRIPTION and _moss_descriptive_field_count(sys_pass) <= 2:
                        status("Listening — that report stopped early; adding a free-form description...")
                        desc = ""
                        try:
                            desc = moss_generate(
                                moss_model, moss_processor,
                                [{
                                    "role": "user",
                                    "content": [
                                        {"type": "text", "text": clip_note + _MOSS_RECOVERY_PROMPT},
                                        {"type": "audio", "path": resolved_path},
                                    ],
                                }],
                                max_new_tokens=main_max_tokens, loop_guard=True,
                            )
                        except Exception as e:
                            print(f"  (description pass failed: {e})")
                            desc = ""
                        if desc and not _moss_analysis_looks_empty(desc):
                            desc = _moss_drop_duplicate_core_fields(desc, sys_pass)
                        if desc and desc.strip() and not _moss_analysis_looks_empty(desc):
                            sys_pass = (
                                sys_pass.rstrip()
                                + "\n\nFREE-FORM DESCRIPTION (a second listening pass, added because "
                                "the structured report above stopped before its instrumentation, "
                                "timbre and vocal fields):\n"
                                + desc.strip()
                            )
                            status_done("Added a free-form description of instrumentation, timbre and vocals")
                    return sys_pass, sys_conversation, main_max_tokens
                status_done("System-message retry also looked empty; trying the short recovery prompt")
        status("Listening — main pass looked empty; retrying recovery analysis...")
        recovery_conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": clip_note + _MOSS_RECOVERY_PROMPT},
                    {"type": "audio", "path": resolved_path},
                ],
            }
        ]
        recovery = moss_generate(
            moss_model, moss_processor, recovery_conversation, max_new_tokens=main_max_tokens, loop_guard=True
        )
        if recovery and not _moss_main_pass_failed(recovery):
            first_pass = recovery
            conversation = recovery_conversation
            status_done("Recovery analysis succeeded")
        else:
            # Keep the better of the two if recovery also failed
            if recovery and len(recovery.strip()) > len((first_pass or "").strip()):
                first_pass = recovery
                conversation = recovery_conversation

            # Second retry on a re-prepared copy of the audio. A shorter prompt
            # doesn't help when the model isn't getting usable audio features
            # in the first place -- very quiet or very dynamic masters,
            # high-rate / multichannel WAV/FLAC passed through unconverted, or
            # tracks longer than the model's audio window. Loudness-normalise,
            # downmix to 16 kHz mono and cap the length, then ask again.
            norm_path = _moss_prepare_retry_audio(resolved_path)
            if norm_path:
                status_done("Recovery prompt also looked empty; retrying on loudness-normalised mono audio")
                status("Listening — retrying on normalised audio...")
                norm_conversation = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": main_prompt},
                            {"type": "audio", "path": norm_path},
                        ],
                    }
                ]
                try:
                    norm_pass = moss_generate(
                        moss_model, moss_processor, norm_conversation, max_new_tokens=main_max_tokens, loop_guard=True
                    )
                except Exception as e:
                    print(f"  (normalised-audio retry failed: {e})")
                    norm_pass = ""
                if norm_pass and not _moss_main_pass_failed(norm_pass):
                    first_pass = norm_pass
                    # The self-check pass continues this conversation, so it
                    # must reference audio that still exists: keep the file.
                    conversation = norm_conversation
                    _MOSS_RETRY_TEMP_FILES.append(norm_path)
                    status_done("Normalised-audio retry succeeded")
                else:
                    try:
                        os.remove(norm_path)
                    except Exception:
                        pass
                    status_done(
                        "All main-pass retries looked empty; falling back to the "
                        "signal-processing and stem measurements for this track"
                    )

    return first_pass, conversation, main_max_tokens


_MOSS_RETRY_TEMP_FILES = []
_MOSS_MAX_AUDIO_SECONDS = MOSS_MAX_AUDIO_SECONDS or 600
_MOSS_SHORT_CLIP_SECONDS = 30


def _moss_short_clip_note(path):
    """Prompt preface for clips too short to hold a whole song.

    On a 10-20 s clip the full-analysis prompt (genre ranking, chords, a
    timestamped STRUCTURE, lyric subject...) asks for things a clip can't
    show, and the model tends to answer by declaring everything
    "not discernible". Telling it up front what it has fixes that.
    """
    try:
        dur = _resolve_total_duration(track_path=path)
    except Exception:
        dur = None
    if not dur or dur >= _MOSS_SHORT_CLIP_SECONDS:
        return ""
    secs = max(1, int(round(dur)))
    return (
        f"NOTE: this audio is a short clip of about {secs} seconds, not a full song. "
        "It still contains music: analyse everything that is audible in it — genre, "
        "instruments, key, tempo, chords, timbre, mood and vocals — as confidently as "
        "the clip allows. Don't treat the lack of a full song structure as a lack of "
        "music; STRUCTURE may be a single section (e.g. [Section 0:00-0:"
        f"{secs:02d}]), and use low confidence rather than 'not discernible' where "
        "the clip is thin.\n\n"
    )


def _moss_prepare_retry_audio(path):
    """16 kHz mono, EBU R128-normalised, length-capped copy for a retry."""
    if not path or str(path).startswith(("http://", "https://")) or not os.path.exists(path):
        return None
    out = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    out.close()
    try:
        r = subprocess.run(
            ["ffmpeg", "-y", "-nostdin", "-hide_banner", "-loglevel", "error",
             "-i", path, "-t", str(_MOSS_MAX_AUDIO_SECONDS),
             "-af", "loudnorm=I=-16:TP=-1.5:LRA=11",
             "-ar", "16000", "-ac", "1", out.name],
            capture_output=True, text=True, timeout=300,
        )
        if r.returncode == 0 and os.path.getsize(out.name) > 1000:
            return out.name
    except Exception:
        pass
    try:
        os.remove(out.name)
    except Exception:
        pass
    return None


def _collapse_runaway_chord_repetition(text):
    """
    Safety net: if the listening model emits a CHORDS field that is the same short
    progression repeated dozens of times, collapse it to a compact summary so
    the writer context and saved JSON stay usable.
    """
    if not text:
        return text

    pattern = re.compile(
        r"(CHORDS\s*[=:\-]\s*)(.*?)(?=\n(?:[A-Z][A-Z0-9_ ]{2,}[=:\-]|\n[A-Z][A-Z0-9_ ]{2,}[=:\-]|\Z))",
        re.IGNORECASE | re.DOTALL,
    )

    def _repl(m):
        prefix, body = m.group(1), m.group(2)
        tokens = re.split(r"\s*[→\->,]+\s*", body.strip())
        tokens = [t.strip() for t in tokens if t.strip()]
        if len(tokens) < 12:
            return m.group(0)
        unique = []
        for t in tokens:
            if t not in unique:
                unique.append(t)
            if len(unique) > 6:
                break
        if len(unique) <= 4 and len(tokens) >= 12:
            summary = " → ".join(unique)
            return (
                f"{prefix}{summary} (recurring loop; collapsed from highly repetitive listing)"
            )
        return m.group(0)

    text = pattern.sub(_repl, text)

    # Second pass, one CHORDS line at a time, for loops the pass above misses.
    # Observed on the rock test track: "Emaj7 – B – C#m7 – A6 – B – ..." for
    # 271 chords, written with en dashes (not split above) and ending in a
    # chord cut off mid-name, so it never collapsed. This pass only fires on
    # a runaway loop -- at least 32 chords, the loop repeating at least six
    # times, with a period of 1-6 matching at 95% or more of positions -- so
    # a real progression list (verse / chorus / bridge, a 12-bar blues, a
    # song-length list with a bridge) is left alone. It never reaches past
    # its own line, and fields written after the chords on the same line
    # ("; INSTRUMENTATION=...") are kept.
    def _loop_summary(tokens):
        n = len(tokens)
        if n < 32:
            return None
        for p in range(1, 7):
            if n < 6 * p:
                break
            pairs = n - p
            same = sum(1 for i in range(pairs) if tokens[i] == tokens[i + p])
            if same < 0.95 * pairs:
                continue
            loop = []
            for j in range(p):
                col = tokens[j::p]
                loop.append(max(col, key=col.count))
            # Where the loop first runs cleanly; anything before it is an
            # intro, shown as "X, Y, then ..." (long intros abbreviated).
            start = 0
            for k in range(0, n - p + 1):
                if all(tokens[k + i] == loop[(k + i) % p] for i in range(p)):
                    start = k
                    break
            lead = tokens[:start]
            if len(lead) > 8:
                lead = lead[:8] + ["…"]
            order = [loop[(start + i) % p] for i in range(p)]
            return (", ".join(lead) + ", then " if lead else "") + " → ".join(order)
        return None

    def _line_repl(m):
        prefix, body = m.group(1), m.group(2)
        tail = ""
        nxt = re.search(r"(?:;|\s)\s*\**[A-Za-z][A-Za-z_ ]{2,}\**\s*=|;\s*\**[A-Za-z][A-Za-z_ ]{2,}\**\s*:", body)
        if nxt:
            body, tail = body[: nxt.start()], body[nxt.start():]
        tokens = re.split(r"\s*[→\->,\u2010-\u2015]+\s*", body.strip())
        tokens = [t.strip() for t in tokens if t.strip()]
        summary = _loop_summary(tokens)
        if not summary:
            return m.group(0)
        return f"{prefix}{summary} (recurring loop; collapsed from highly repetitive listing){tail}"

    # The label must open the line (after any list marker / bold), as the
    # listening model writes its fields, so prose such as "the chords - ..." is never
    # touched.
    line_pattern = re.compile(
        r"^([ \t>\-\u2022\d.)]*\**[ \t]*CHORDS\**[ \t]*[=:\-][ \t]*\**[ \t]*)([^\n]*)",
        re.IGNORECASE | re.MULTILINE,
    )
    return line_pattern.sub(_line_repl, text)


def _collapse_runaway_genre_ranked(text):
    """
    Safety net for GENRE_RANKED / Genre Ranked runaway loops, e.g.:
      Genre Ranked=1) rock; 2) indie rock; 3) indie pop; ... 4) jangle pop;
      5) indie rock; 6) indie pop; ... (hundreds of cycling items)

    Keep the first occurrence of each unique genre label (up to 8), drop the
    rest, and mark that the list was collapsed.
    """
    if not text:
        return text

    # Locate the start of a ranked-genre block (strict or loose label).
    start_re = re.compile(r"(?:GENRE_RANKED|Genre\s+Ranked)\s*[=:\-]\s*", re.IGNORECASE)
    m = start_re.search(text)
    if not m:
        return text

    prefix = text[m.start(): m.end()]
    rest = text[m.end():]

    # End of the ranking block: next major analysis label, double newline, or
    # a clearly non-genre section. Rankings are often one huge line.
    end_re = re.compile(
        r"(?=\n\s*(?:KEY|TEMPO_BPM|CHORDS|STRUCTURE|INSTRUMENTATION|VOCALS|"
        r"GENRE_ADJACENT|GENRE_RULED_OUT|FULL LYRICS|ERA_|VOCAL /|"
        r"RECOMMENDED |\[Independent|STEM MIDI|11\.\s*ERA)\b)"
        r"|(?=\n\n)"
        r"|(?=\nGenre\s+Adjacent\b)"
        r"|(?=\nGenre\s+Ruled)",
        re.IGNORECASE,
    )
    end_m = end_re.search(rest)
    body = rest[: end_m.start()] if end_m else rest
    tail = rest[end_m.start():] if end_m else ""

    # Pull numbered items: "1) rock" / "12) indie pop (confidence: medium)"
    items = re.findall(r"\d+\)\s*([^;\n]+)", body)
    items = [
        re.sub(r"\s*\(confidence:[^)]*\)", "", it, flags=re.I).strip(" ;,.")
        for it in items
    ]
    items = [it for it in items if it]

    if len(items) < 12:
        # A short numbered cycle can evade the long-loop guard. Match only a
        # complete ranking line, leaving adjacent report fields untouched.
        line, separator, remainder = body.partition("\n")
        entries = line.strip().rstrip(";").split(";")
        parsed = [re.fullmatch(r"\s*\d+\)\s*(.+?)\s*", entry) for entry in entries]
        if (len(entries) > 1 and all(parsed)
                and not text[text.rfind("\n", 0, m.start()) + 1:m.start()].strip()):
            unique_entries = {}
            for entry in parsed:
                label = entry.group(1)
                key = re.sub(r"\s*\(confidence:[^)]*\)\s*$", "", label, flags=re.I)
                unique_entries.setdefault(key.strip().casefold(), label)
            if len(unique_entries) < len(entries):
                ranked = "; ".join(f"{i}) {label}" for i, label in enumerate(unique_entries.values(), 1))
                return text[:m.start()] + prefix + ranked + separator + remainder + tail

        # Still collapse pure semicolon cycles of the same few labels
        cycle = re.compile(
            r"((?:indie rock|indie pop|power pop|jangle pop|alternative rock|"
            r"post-punk(?: revival)?|garage rock(?: revival)?|rock)"
            r"(?:\s*;\s*(?:indie rock|indie pop|power pop|jangle pop|alternative rock|"
            r"post-punk(?: revival)?|garage rock(?: revival)?|rock)){8,})",
            re.IGNORECASE,
        )

        def _cycle_repl(cm):
            parts = [p.strip() for p in re.split(r"\s*;\s*", cm.group(1)) if p.strip()]
            unique, seen = [], set()
            for p in parts:
                k = p.lower()
                if k not in seen:
                    seen.add(k)
                    unique.append(p)
            return "; ".join(unique) + " (collapsed genre cycle)"

        return text[: m.start()] + prefix + cycle.sub(_cycle_repl, body) + tail

    unique, seen = [], set()
    for it in items:
        key = it.lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(it)
        if len(unique) >= 8:
            break

    ranked = "; ".join(f"{i}) {g}" for i, g in enumerate(unique, 1))
    collapsed_body = (
        f"{ranked} "
        f"(collapsed from {len(items)} highly repetitive ranked items; "
        f"kept first unique labels only)"
    )
    return text[: m.start()] + prefix + collapsed_body + tail


def _collapse_runaway_repetition_fields(text):
    """Apply all known runaway-repetition safety nets to an analysis blob."""
    if not text:
        return text
    text = _collapse_runaway_chord_repetition(text)
    text = _collapse_runaway_genre_ranked(text)
    return text


_MOSS_FIRST_PASS_BY_TRACK = {}


def _remember_moss_first_pass(track, first_pass):
    """dev64: keep the listening model's first pass for the save-time layout
    (REPORT_FIXED_LAYOUT). Never raises."""
    try:
        if track is not None and first_pass:
            _MOSS_FIRST_PASS_BY_TRACK[str(track)] = str(first_pass)
    except Exception:
        pass


def _collapse_repeated_line_blocks(text):
    """Remove back-to-back repeats of the same block of lines. A generation
    that loops (the vocal pass was observed writing its whole answer four
    times) otherwise reaches the notes verbatim, repeated."""
    prev = str(text or "")
    # Largest blocks are tried first, so four copies can first collapse to two;
    # repeat until nothing changes.
    for _ in range(6):
        cur = _collapse_repeated_line_blocks_once(prev)
        if cur == prev:
            break
        prev = cur
    return prev


def _collapse_repeated_line_blocks_once(text):
    lines = str(text or "").split("\n")
    if len(lines) < 4:
        return str(text or "")

    # Compared loosely: a looping generation rarely repeats itself byte for
    # byte (an AI-generated test track's vocal pass wrote "LEAD_CATEGORY" in one copy and
    # "LEAD CATEGORY" in the next, three times over), so exact matching let
    # every copy through.
    def _k(line):
        return re.sub(r"[\s_]+", " ", str(line).strip().lower()).strip(" .:")

    keys = [_k(l) for l in lines]

    import difflib

    _num_re = re.compile(r"\d+(?:\.\d+)?")
    _tag_re = re.compile(r"^([^=:]{1,60})[=:]\s*(.*)$")

    def _eq(x, y):
        # Long lines may differ by a reworded word between copies ("pitch/
        # formant proxies" vs "... prefixes" in the rock track's looping vocal
        # pass). Never treated as the same: short lines that differ at all,
        # lines whose numbers differ (timestamps, Hz, counts), and "TAG=value"
        # / "Label: value" lines whose short value differs (male vs female,
        # 1 vs 2 voices) -- those are different content, not a loop.
        if x == y:
            return True
        if len(x) < 20 or len(y) < 20:
            return False
        if _num_re.findall(x) != _num_re.findall(y):
            return False
        mx, my = _tag_re.match(x), _tag_re.match(y)
        if mx and my:
            vx, vy = mx.group(2).strip(), my.group(2).strip()
            if (vx or vy) and (len(vx) <= 40 or len(vy) <= 40) and vx != vy:
                return False
        sm = difflib.SequenceMatcher(None, x, y, autojunk=False)
        if sm.real_quick_ratio() < 0.9 or sm.quick_ratio() < 0.9:
            return False
        return sm.ratio() >= 0.9

    def _same_block(j0, bk):
        if j0 + len(bk) > n:
            return False
        return all(_eq(keys[j0 + q], bk[q]) for q in range(len(bk)))

    out = []
    i = 0
    n = len(lines)
    while i < n:
        collapsed = False
        for size in range(min(40, (n - i) // 2), 1, -1):
            bkey = keys[i:i + size]
            if not any(bkey):
                continue
            j = i + size
            reps = 0
            while _same_block(j, bkey):
                j += size
                reps += 1
            if reps:
                # Keep the copy whose lines are written as the canonical
                # underscore tags (LEAD_CATEGORY, not "LEAD CATEGORY"): the
                # tag parser only reads those, so dropping that copy in
                # favour of a looser one would lose the tags.
                def _tag_lines(k0):
                    return sum(
                        1 for ln in lines[k0:k0 + size]
                        if re.match(r"\s*\**[A-Z]+_[A-Z_]+\**\s*[=:]", ln)
                    )
                best_k = max(range(reps + 1), key=lambda c: (_tag_lines(i + c * size), -c))
                keep_from = i + best_k * size
                # A final copy cut off part-way (the generation hit its
                # length limit) is dropped too: whole lines that match the
                # block's start, ending in one that is a prefix of the next.
                t = 0
                while t < size and j + t < n and keys[j + t] and _eq(keys[j + t], bkey[t]):
                    t += 1
                if t < size and j + t < n and keys[j + t] and bkey[t].startswith(keys[j + t]) \
                        and j + t + 1 >= n:
                    t += 1
                    j += t
                elif t and j + t >= n:
                    j += t
                out.extend(lines[keep_from:keep_from + size])
                i = j
                collapsed = True
                break
        if not collapsed:
            out.append(lines[i])
            i += 1
    return "\n".join(out)
