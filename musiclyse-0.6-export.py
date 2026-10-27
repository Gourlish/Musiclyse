# Musiclyse export dev2 (2.3.4): Dev3 compatibility and conditional custom-persona awareness.
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
musiclyse_export.py — turn Musiclyse saved-song .json files into plain-text
listening notes that a stock local LLM (Gemma, Llama, Qwen, Mistral... in
LM Studio, Ollama, Open WebUI, etc.) can use with NO custom system prompt.

Why this is needed
------------------
A saved .json's "analysis" field is written for Musiclyse's own writer model:
it is 100-200k characters long, full of instructions aimed at that model
("Do NOT say four-on-the-floor unless..."), internal tag vocabulary
(post_puberty_male, GENRE_RANKED, PANNs, Demucs...), thousands of raw MIDI
events, and blocks that only make sense alongside Musiclyse's 37k-character
system prompt. A stock model given that raw file will run out of context,
recite internal labels, or follow instructions that no longer apply.

This script extracts the measured facts, rewrites them in plain language,
drops the internal plumbing, and puts a short "how to read these notes"
guide at the top of the file itself — so the guidance travels with the
data and no system prompt is required.

Standard library only. Does not import Musiclyse or any audio library.

Usage
-----
    python musiclyse_export.py saved-songs/                    # every .json in the folder
    python musiclyse_export.py "saved-songs/Artist - Song.json"
    python musiclyse_export.py saved-songs/*.json --level brief
    python musiclyse_export.py a.json --level high                   # full depth, compressed MIDI
    python musiclyse_export.py a.json --level full                   # full depth, timed MIDI events
    python musiclyse_export.py a.json b.json --combine compare.txt   # one file for comparisons
    python musiclyse_export.py a.json --max-tokens 6000              # fit a small context window
    python musiclyse_export.py a.json --stdout | ollama run gemma3 "What's the groove in the chorus?"
    python musiclyse_export.py untagged.json --wiki-artist "Band R"   # background for an untagged file

Compatibility
-------------
Built for saves from Musiclyse 0.5-moss-dev46 and still reads every earlier
0.5-moss (MOSS-Music-8B-Instruct), 0.5 (Music Flamingo) and 0.4 save (see
"What 2.3 changed" and the older notes below). Same flags, same Wikipedia
lookup, same casual listening-notes voice.

What 2.3.3 changed
------------------
- High-level MIDI compression retains every individual onset, along with per-note
  duration/level information where present; run notation and its explanation agree.
- High/full exports include the saved beat-grid timestamps and exact drum-hit
  timestamps. The MIDI completeness flag is surfaced so sampled/unknown coverage
  is not mistaken for a complete transcription.
- If the prose chord-progression block is absent but top-level chord segments exist,
  the exporter reconstructs a cautious section-by-section sequence from those segments.
- Token trimming can continue through high-priority sections as a last resort instead
  of silently stopping over budget; very small budgets that cannot fit the fixed guide
  now generate a warning. Duplicate input basenames under a shared --out-dir are disambiguated.
- --raw-notes now works at --level brief too; it adds only the explicitly requested raw
  event list and otherwise leaves the brief export unchanged.

What 2.3.2 changed
------------------
- Vocal credits: multi-line singer-identity reasoning is read in full, and
  explicit guest/co-vocalist names are retained. Those names are now shown even
  when a section-voice check takes the alternate "Who's singing" path. The
  comparison table also shows them. Existing pitch-only false-positive checks
  remain in force; no second singer is inferred from pitch alone.

What 2.3 changed (Musiclyse 0.5-moss-dev42 to dev46)
---------------------------------------------------
Each change has a setting near the top of this file; False gives 2.2.3e's
output for that part.
- Genre (GENRE_VIEW_DEV44): Musiclyse dev44's own genre view, copied unchanged,
  so the dossier names the genre the program itself would: the measured styles
  can narrow a broad label (rock -> hard rock) or replace a contradicted one
  (Dancing Queen: disco, not indie pop / synth-pop); a confident period-bound
  style corrects a "2010s-2020s" era guess; K-pop, J-pop, Europop and other
  regional labels need the local language or a local artist; World /
  International labels are never used. The artist's country comes from the
  Wikipedia database when the background lookup is on (read-only; the
  background articles are chosen exactly as before).
- Styles (STYLES_IN_DOSSIER): "The sound" names the more specific styles heard
  (no percentages, no model names); a style of another family is an
  alternative reading only, and "is it X?" is at least "partly, yes" when X is
  one of them.
- Chords (CHORD_PROGRESSION_IN_DOSSIER): the measured chord progression (dev45)
  in "How it unfolds", "Melody and harmony" and a section-by-section detail
  with Roman numerals; high and full also list every chord change with its
  time. The transcription's "most common chords" lists, which often disagreed
  with the real progression, are left out when it is there.
- Melody (MELODY_IN_KEY_IN_DOSSIER, MELODICISM_IN_DOSSIER): how melodic the
  singing is (overall and part by part) and how the tune sits in the key
  (resting notes, phrase endings, peaks).
- Drum grids (GROOVE_GRIDS_BY_PART): one grid for each part whose groove
  differs (up to 4) instead of a single grid.
- Singer (IDENTITY_FLIP_GUARD): Musiclyse dev46's fix for saves made before it
  -- when only the confirmation pass heard a man against the first pass and
  the voice descriptions, and the identity simply followed it (no singer
  named), the singer's gender is not stated (Dancing Queen had "a man's voice").
- Wording (VARIED_WORDING): "How it unfolds" says a repeated groove, tune shape
  or instrument list once, then only what changes; stretches with the same kit
  are merged ("the full kit, then the full kit from 0:33"); the drum sound reads
  as words ("tight and damped", not "a tight, damped, a bright, , crisp"). The
  guide asks for fresh wording, not the notes' phrases or stock review lines.
- Sound vs knowledge (SOUND_BEFORE_KNOWLEDGE): the guide keeps how the recording
  sounds to what was heard; what is known about the artist, album or year is
  casual context beside it, never a description of the sound, and anything the
  background doesn't cover is answered from the model's own knowledge.
- Tempo (TEMPO_COPY_CHECK): Musiclyse dev46's check for saves made before it -- a
  listening-model tempo that only copied a detector reading no longer decides a
  half/double-time choice (Dancing Queen: 100 BPM, not 200).
- A by-ear voice description saying a sung track "may be instrumental" is left out.
- Fixed: report values like "none (instrumentation undetected)" were printed;
  a shared lead with no settled gender read "A single lead voice"; the raw
  style-model and chord-segment lists were dumped under "About the file".

What 2.2.3e changed
------------------
- Output layout: under llm-ready (or --out-dir), files go in a subfolder named
  {level}-wiki or {level}-no-wiki according to the export level and whether
  Wikipedia background articles were actually included (e.g. high-wiki,
  standard-no-wiki). --stdout and an explicit path in --combine are unchanged.
- Levels: --level extreme is removed; --level full is the top tier (timed MIDI
  event lists). --level high keeps compressed MIDI. brief and standard unchanged.

What 2.2.3d changed
------------------
- Levels: a new --level high sits between standard and full: same measurement
  depth as full, but pitched MIDI is run-length compressed with onset timestamps
  kept. --level full carries the full timed MIDI event list. brief and standard
  are unchanged. --raw-notes still dumps every note and suppresses the high/full
  MIDI section.

What 2.2.3c changed
------------------
- Full exports now include the timed, pitched MIDI events for vocal and
  instrument parts by default, in addition to the drum event detail already
  present. Standard exports are unchanged. --raw-notes retains its existing
  behaviour and does not create a duplicate MIDI section in full mode.

What 2.2.3b changed
------------------
- Guide: when a claim about the song conflicts with what was heard (wrong
  instrument, era, singer, structure, etc.), push back gently from the
  evidence rather than agreeing for agreement's sake. Themes and opinions stay
  free; measured facts do not bend to the user.

What 2.2.3a changed
------------------
- Unbroken male lead prose no longer says the voice "sits where women's voices
  sit" or "can pass for a woman's". Who's singing states high/light/boyish
  weight only. Easy-to-get-wrong tells the model to describe the sound from
  timbre, delivery and range — not a stock "sounds like a girl" line. No other
  behaviour changes.

What 2.2.3 changed (Musiclyse 0.5-moss-dev20)
---------------------------------------------
Each change has a setting near the top of this file; False gives 2.2.2's output.
- Singer names (SINGER_NAME_ANY_WORDING): a person's name after "vocalist",
  "singer", "frontman", "frontwoman" or "vocals by" anywhere in the same
  clause ("Band V's documented lead vocalist for the 1983-era line-up is
  John Doe"), skipping band names, possessives and the act's own name.
  A band or vocal group with no singer named is "Band H's singer", never
  "Band H sings".
- Shared lead for a documented vocal group (GROUP_LEAD_STAYS_SEVERAL, dev20):
  an identity that says one lead while documenting a boy band, girl group,
  brothers, sisters or a duo, with no single lead named, is a shared lead
  (the sibling-band test track: several in dev17, one in dev19).
- Stock genre answers (MOSS_LOW_TRUST_FIELDS, dev20): a genre resting on
  MOSS-Music's stock list ("indie rock; power pop; jangle pop", its prompt's own
  example) falls back to the family the independent tags support -- "rock",
  "rock with a punk edge", "pop with a rock edge" -- and is no longer called
  "two independent readings agree"; stock labels and non-genre words
  ("layered production", "urban") are not listed as close neighbours. The
  era warning the stock label used to trigger is kept.
- Synth-rock (GENRE_SYNTH_ROCK, dev20): the synth-rock test track is "synth-rock" -- a rock band
  with a synthesizer riff as the lead instrument -- with synth-pop and new
  wave as fair comparisons, instead of "post-punk / new wave ... don't call it
  synth-pop or new wave". the post-punk test track keeps post-punk / new wave.
- Pitch (PITCH_CHECK_ROUNDED_FIGURES, dev20): the low-confidence pitch check
  allows for figures rounded to whole hertz (the synth-rock test track: "high tenor range" from the
  notes' E4, not "very high for a man" from a 4%-voiced G4).
- By-ear pitch (BY_EAR_PITCH_FOLLOWS_CHECK): when that check sets the
  tracker's median aside, a by-ear clause quoting the tracker's figures
  ("centered around G4", "~B3 to A4 (251.3–445.1 Hz)") is left out of "How it
  sounds" instead of contradicting "Who's singing".

What 2.2.2 changed (Musiclyse 0.5-moss-dev19)
---------------------------------------------
- Tempo: dev19's tempo checks, recomputed for older saves. A MOSS-Music
  TEMPO_BPM that only repeats the prompt is not used; a MOSS-Music pulse
  matching neither detector gives way to the detector that matches its own
  beat spacing; two readings ~2x apart settled only by "prefer the lower pulse"
  go to the higher one when the saved snare hits land every two of its beats.
  On a save whose tempo changes this way, the beat patterns (counted at the
  old tempo) are not spelled out.
- A featured singer in the artist or title tag means a shared lead (dev19);
  "Band X feat. Singer Y" with no singer named comes out as
  "Band X's singer and Singer Y".
- The singer is named from more identity wordings ("lead vocalist John Doe", "frontman John Doe", "documented vocalist is John Doe"), and
  a band with no singer named is "Band Z's singer", not "Band Z sings".
- "What it isn't" leaves out genres that fuse with the decided one (folk,
  country, blues next to rock, and so on).
- An era guess starting before the 1990s on a loud, limited master says the
  master fits only the latest part of that range.
- A lyrics tag that holds no lyrics (under 12 words, or one line naming the
  artist or title) is not printed as the published lyrics.
- Another report field pasted into a by-ear value ("... phrase STRUCTURE=[
  Intro | ...") is cut off.

What 2.2.1 changed (Musiclyse 0.5-moss-dev18)
---------------------------------------------
- Unbroken boy's voices: a young male singer whose first vocal pass heard a
  woman and only the confirmation pass a man, at a measured median pitch of
  300 Hz or more, is described as his unbroken voice (dev18
  YOUNG_VOICE_CONFIRM_FLIP, recomputed for dev17 saves). dev17's the teen-pop singer
  the teen-pop test track had come out as "his voice had already deepened" beside a
  "mezzo-soprano" by-ear line. dev17's hedged "may already have been
  deepening" sentence is no longer turned into "had already deepened".
- By-ear wording for a male lead: female voice types (soprano, mezzo-soprano,
  alto, contralto -- not an alto sax) and she / her are rewritten as they
  already were for "female" / "woman"; quoted lyrics are left alone (they had
  been changed: "<lyric line>" became "<lyric line, misheard>").
- A LEAD_VOCAL_CHARACTERISTICS sentence saying there are no vocals is left out
  of a sung track's voice section.
- Key: dev18's KEY NOTE CHECK, recomputed from the saved notes for older
  saves. A listening-model key the transcribed notes don't fit (MOSS-Music's
  stock "Eb major" / "Eb minor" / "Ab minor") is set aside, so a key Essentia
  and the notes agree on reads as a clear one instead of "methods disagreed";
  where that stock answer had won over a weak Essentia key the notes fit,
  Essentia's key is used.
- Singer age from "(b. 1994)" / "was ~15-16 at the release" identity notes.
- --max-tokens drops the by-ear melody, tune-shape and chords-by-ear lines
  before dropping whole sections.

What 2.2 changed (Musiclyse 0.5-moss-dev17)
-------------------------------------------
Everything dev17 analyses that 0.5 did not now reaches the dossier, in the same
plain words, and dev17's load-time checks are recomputed from the saved text so
older saves get them too.
- MOSS-Music: its name in the notes is rewritten like the old model's; the
  genre-check readings ("MOSS-Music GENRE_RANKED led with ...", "MOSS-Music
  ranked ...", "MOSS-Music's ranked list ...") are recognised; structure-replay
  hints are read from "moss_hints"; "(confidence: high)" is taken off genre labels.
- SECTION VOICE CHECK: a second lead of the other gender in some stretches
  (a vocal-group test track's male verses) -- "Who's singing" names both voices and where each one
  leads, "How it unfolds" marks where the second voice takes over, "The voice"
  separates the two, and a one-voice identity is set to mixed leads as dev17 does.
- HARMONY AND MELODY BY SECTION: the chords each part of the song uses most, which
  part carries the tune, where it sits against the chords and the bass, and how
  much of it lands on the chord notes -- summarised under "Melody and harmony",
  quoted per section in "How it unfolds", and listed in a new detail section.
  Saves made before dev17 get the block rebuilt from their stored notes.
- DRUM STEM NOTE / ERA NOTE (drums): no drum kit on the recording -- said once in
  the nutshell, the groove and "Easy to get wrong"; drum words in the by-ear era
  evidence, instrument list and description are left out.
- FULL-BANDWIDTH TEMPO CHECK: the tempo note says the slower readings counted the
  same pulse every 2 or 4 beats.
- Faint vocal texture: an instrumental whose background may hold a sample or a
  processed voice says so (texture only, never a singer).
- KEY CENTRE CHECK (e.g. D Mixolydian, the notes of G major) and the key-quality
  repair ("D aug" takes Essentia's mode).
- INSTRUMENT CHECK, dev17 readings: a strong, steady 'harmonica' is kept as a
  harmonica-like or brassy lead; an accordion tag on a chordal part is a synth pad.
  Fixed: dev17's longer wording (an extra colon) stopped the accordion veto working.
- GUITAR CHECK, GENRE ERA CHECK and the synth-denial ARRANGEMENT CHECK, computed
  as dev17 does when it loads a save.
- Low-confidence pitch: a tracker median 3+ semitones from the note reading is
  replaced by it (dev17's PITCH CHECK), so the voice type is right on older saves.
- The report's DESCRIPTION (MOSS-Music writes it last) appears as the "overall
  impression" by ear, minus its own key / tempo figures and any beat or drums the
  measurements contradict. Empty answers ("uncertain", "none (no audio detected)",
  "(none given)") are left out instead of printed. The CHORDS line is split into
  the measured chords and the chords heard by ear. A measured MELODIC_CONTOUR is
  no longer labelled "by ear".
- Smaller fixes: "conventional stereo stereo image"; a key the methods disagreed
  on was called "a clear, confident read"; any tenor was called "high for a male
  lead"; a duo name took "sings"; an instrument tagged on two parts was named
  twice in a row; the section-name caution was printed twice; "© transcript ..."
  speech-recognition credits are dropped from heard lyrics; the comparison table
  shows the two voices and "no drum kit".
- Fixed: once an instrument tag was vetoed, the part's other tags lost their
  "sustained" / whole-mix notes, so a steady but quieter tag (a dance-pop test track's
  'synth') dropped out of the dossier. Tags now keep their notes.
- --max-tokens: the lines 2.2 adds (chords in "How it unfolds", the overall
  impression, the harmony summary) are given up first, before any section 2.1
  would have kept. Only the new "Easy to get wrong" lines can still push the
  lowest section out at a budget that was already almost full.

What 2.1.6 changed
------------------
- Meaning/theme speculation allowed: the guide no longer reads as "never guess anything." Measured
  facts stay strict (don't invent instruments, times, credits); readings of what the song is about
  are encouraged, marked as your take when nothing definitive is in the file or your knowledge.
  Lyrics intros updated the same way. Casual voice and anti-citation rules unchanged.

What 2.1.5 changed
------------------
- Hard anti-RAG voice rule (top + footer) aimed at Gemma-style "Based on the provided text"
  openers: the dossier is framed as memory of a song you already know, with an explicit ban on
  "provided text", "the text says", "according to the notes", and similar. Guide shortened;
  measurement content and casual register unchanged.

What 2.1.4 changed
------------------
- Stronger anti-report / anti-citation voice for stock chat models (Gemma, etc.): the guide now
  frames the dossier as memory of a record you know, forbids openings like "based on", "according
  to", "from the notes", "the document", and the footer repeats that. Comparison wording no longer
  says "the notes". Casual friend register unchanged; no measurement regressions.

What 2.1.3 changed
------------------
- Comparison clarity for stock chat models: single- and multi-track guides tell the model to
  resolve "it" / "this" / "the other one" from context and the titles in the notes, and never to
  claim the songs weren't specified when those titles are present. Combined files use song titles
  (not only "Track 1") in the side-by-side table and a multi-track footer. Casual voice unchanged.

What 2.1.2 changed
------------------
- Cover-art descriptions omitted by default. Txt dossiers do not ship with a
  referenced image (unlike the .json + cover file pair), so "The cover" is left
  out unless you pass --cover. --copy-cover is unchanged (still copies the image
  file when you want it for a vision model).

What 2.1.1 changed
------------------
- Broken vs unbroken male voice (dev59 UNBROKEN / dev69 DEEPENED): identity
  prose such as "consistent with an unbroken male voice" or "voice had already
  deepened" is recognised, not only the exact "VOICE MATURITY: UNBROKEN" mark.
  An adolescent-male identity whose acoustic reading is post_puberty_male is
  described as already deepened (lower, fuller — not high, light or boyish);
  the unbroken case keeps the high/boyish wording. Stops two teen-male scans
  from both reading "light and still changing".
- Birth year in identity reasoning accepts "born December 1987" as well as
  "born 1987".

What 2.1 changed (Musiclyse 0.4-dev60 through dev76)
----------------------------------------------------
Compatible with the fixed-layout reports, CHECKS block and instrument-tag
corrections that Musiclyse writes from dev60 onward, without changing the
casual listening-notes voice or the Wikipedia lookup.
- INSTRUMENT CHECK (dev60+): when the analysis says an accordion / harmonica /
  reed tag is a false positive for a synth pad or horn, those labels are dropped
  from the dossier so a stock model never names an accordion that is not there.
- VOCAL DEMOGRAPHICS runaway loops (a repeated "no overt adult-male contrast"
  string some saves carry) are truncated so "The voice" stays readable.
- Shared / featured leads: LEAD_SINGERS=several plus a named feature in the
  identity reasoning (e.g. Singer Y on a Band X track) is written
  as a shared lead with the credited names, not "nothing here says who".
- Named solo lead: "documented lead vocalist X" / "lead vocalist is X" in the
  identity reasoning is recognised the same way "credited solo artist is X" was.
- Full mode still carries every measured block; standard keeps the same breadth
  as 2.0 (nutshell through exact figures and per-section detail).
- Wikipedia lookup algorithm unchanged. When background articles are present, a
  short HTML comment names the CC BY-SA 4.0 licence for that section only — it is
  not part of the listening notes and is not meant to be read aloud by a model.
- Guide and casual "friend who just listened" register unchanged (dev58 voice).

What 2.0 changed (Musiclyse 0.4-dev59)
--------------------------------------
The dossier is now written as listening notes rather than a report, so a stock
model talks about the song the way a friend who has just played it would, with
the music and the artist's story in the same breath.
- Layout: "The track in a nutshell" (a few plain sentences), "Who's singing",
  "Easy to get wrong" (the handful of conflicts the analysis settled, said once),
  "How it unfolds" (the song from start to finish: who comes in and drops out,
  what the drums do, the beat changes, where each lyric line is sung, how the
  melody moves and where it peaks), then the sound, groove, voice, melody and
  harmony, era, lyrics, exact figures and per-section detail. Evidence is
  reconciled here and only the outcome is written: no "listening model",
  "classifier", "stem" or "measured" labels in the main sections.
- Lyrics placed in time: each timed transcription chunk is matched, word order
  and all, to the file's own lyric lines, so "How it unfolds" quotes the real
  line where it is sung ("the pre-chorus arrives around 0:30").
- The guide is shorter and follows dev58's own chat voice (one friendly take,
  no citing, sound and story woven together, numbers still exact). No track
  names or numbers appear in it, so nothing in it can be repeated as a fact.
- Fixed: the guide never contained the line the Wikipedia guidance replaces, so
  with background articles the "this is your own knowledge" note was missing.
- Fixed: section times now print exactly as Musiclyse prints them (truncated,
  not rounded: Verse 1 at 0:08, not 0:09).
- Fixed: listening-model fields written in any case ("Genre_Ranked=", "Key=",
  "Tempo_BPM=", "VOCAL PRESENCE: yes", "NUM_DISTINCT VOICES: 1") are read; a
  dev59 save's raw header was being exported as the "overall impression".
- dev55-dev59 evidence now used: the free-text voice descriptions and dev57's
  probability veto (a "female" reading the vocal pass contradicts is not used),
  dev59's UNBROKEN voice signature (with a male singer identity: a boy's voice
  that hasn't broken, and at 15+ "later than usual"), the singer's name and age
  from the identity step, the second vocal-pitch reading (VOCAL NOTE PITCH) and
  dev55's low-confidence pyin flag, dev56's genre check in either direction
  (rock -> pop, and electronic -> rock / post-punk / new wave), "(with X lean)"
  genres said as words, and dev55's era pass that gave no estimate (its heading
  changed and the whole era pass used to be lost).
- Era by ear: clauses the measurements contradict (pumping when none was
  measured, trap programming over a four-on-the-floor kick, hi-hat rolls over
  plain eighth-note hats) are left out.
- Smaller: weak mood tags dropped, the cover model's raw reply no longer
  repeated, the beat diagram comes from the longest-running part (usually the
  chorus) rather than the intro, --no-lyrics also keeps lyric quotes out of
  "How it unfolds", and the comparison table uses the same plain wording.

What 1.3.2 changed
------------------
- Wikipedia lookup: the search itself is unchanged, but each album / song article it
  finds is now checked against the track. An album is kept only if the song is in its
  track listing (an album named in the file's tags is also kept when the article text
  has no track listing at all). A song article is kept only if it is the article about
  this very song by this artist. The artist, member and discography articles are
  unaffected. --wiki-debug shows what was kept and how many were left out.
- Guide: shorter and positively worded (no instructions that mention citations, sources or
  what not to say), relaxed and conversational, asking for full multi-paragraph answers.

What 1.3 changed (Musiclyse 0.4-dev53)
--------------------------------------
- Everything in a save now reaches the dossier: the listening model's genre
  neighbours / ruled-out styles / melody fields / any new field, all vocal-pass
  and second-listen fields, the singer identity's LEAD_SINGERS (a group sharing
  the lead), the era pass's missing markers, every production reading, the beat /
  meter / tone / loudness / voice / second-opinion measurements, each separated
  part's own report (range, density, chords), the lyric repetition map, the
  dev44 structure-replay data (timed lyric lines, the listening model's names for
  unsung parts, beat count), every cover-art field, extra file tags, file
  details and any field a later Musiclyse adds. --raw-notes lists every note.
- dev53's checks on saved scans, computed from the saved text with dev53's own
  rules: GENRE CHECK (a rock/punk first impression the genre check replaced),
  ARRANGEMENT CHECK (a prominent chordal part in the 'other' stem), VOICE COUNT
  CHECK (co-leads read from the pitch histogram alone), LOUDNESS CONTEXT, the
  corrected ERA CAVEAT, the measured era clues and decade guide, and the
  section-name caveat ("Silence", swapped verse/chorus).
- The guide now keeps style, instrumentation, vocal pitch and era-of-sound to
  the recording's own data (artist, year and background are context only),
  treats background articles as the model's own knowledge, and asks for a
  listener's voice rather than a report. An Era section is always included.
- --max-tokens counts the guide too; minor hardening against malformed notes.
- The Wikipedia lookup is unchanged.

Encyclopedia background (1.2)
-----------------------------
If music_wiki_heavy.db (the Musiclyse Wikipedia database) sits next to this
script, each export also gets the full encyclopedia articles for the track's
artist, their albums and singles, and the band's members, written so a model
uses them as its own knowledge. The lookup starts from the artist tag, finds
the artist's article, then follows that article's discography and members
lists (plus a separate "<Artist> discography" article if there is one); every
album, single and person is checked against the artist before it is added.
Articles whose infobox was flattened into "key: value" text at the start (as in
music_wiki_heavy.db) are read field by field: the infobox's member lists and an
album's "artist:" field are used directly, and the infobox is shown as a short
"Quick facts" list. After converting, the script lists the articles it added
for each file.

    --no-wiki          leave the background out
    --wiki-db PATH     use a database somewhere else
    --wiki-scope core  only the artist, this track's album and song, and members
    --wiki-artist NAME artist to look up for files without an artist tag
    --wiki-debug       show every member name found and why it was or wasn't added

Also fixed in 1.2: titles, names and lyrics from the file are no longer run
through the internal-term rewriting (a song called "a ballad title" came out
as "Careless speech recognition").

What 1.1 changed
----------------
Bugs fixed:
- Vocal presence was read from any "VOCALS_PRESENT=yes/no" in the text, which
  also matched Musiclyse's explanatory line "(if VOCALS_PRESENT=no, this onset is
  separation residue)"; the vocal pass's "VOCAL PRESENCE: yes" form was missed.
- On instrumentals, separation bleed in the vocals part was still exported as a
  melody line, an intensity column, transcribed notes, change points and
  speech-recognition "lyrics" ("Thanks for watching", repeated syllables).
- The structure note called section names "reliable", and name checks were
  dropped. Saves without structure_sections lost the whole section map
  (the block parser stopped at the "(Use these EXACT ...)" line).
- The listening model's description was lost entirely when it was written as
  prose rather than LABEL= fields.
- Era EVIDENCE swallowed the "MISSING MARKERS=" and following lines.
- dev30's renamed drum lines (hit-level spread, drum-bus while playing) were
  not found; groove-by-section printed "kick kick pattern ...".
- Intensity cells were matched by row position, not section name.
- --combine with --max-tokens ignored the guide and comparison table and could
  overshoot the budget; --copy-cover did nothing with --combine.

Levels
------
    brief     ~3k tokens.  Nutshell, singer, pitfalls, a short walk through the song,
              sound, groove, era, lyrics, exact figures.
    standard  ~6-8k tokens (plus any background). Everything above in full, plus the
              voice, melody and harmony, drums and loudness section by section. (default)
    high      same measurement depth as full, but pitched MIDI is compressed
              (run-length of repeated pitches with onset timestamps) so the dossier
              is smaller without losing discussion capability for melody, intervals
              or timing. Still carries section motion summaries and timestamps.
    full      every measurement as Musiclyse wrote it, each separated part's
              transcription summary, lyric timing and repeat map, file details, and
              the full timed pitched MIDI event list for vocals and instruments.
              Add --raw-notes for every transcribed note (all levels).

Outputs land in llm-ready/{level}-wiki or llm-ready/{level}-no-wiki (or under
--out-dir) depending on the level and whether Wikipedia background was included.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
import re
import shutil
import sys
import unicodedata  # noqa: F401  (used by the Wikipedia background code)

__version__ = "2.3.7"

# -----------------------------------------------------------------------------
# 2.2.3 settings. Each one switches a 2.2.3 change; False gives 2.2.2's output.
# -----------------------------------------------------------------------------
# The singer named from any wording that puts a person's name after
# "vocalist" / "singer" / "frontman" / "frontwoman" in the same clause, words
# in between allowed ("lead vocalist for the 1983-era line-up is Firstname
# Lastname"); a band with no singer named is "Band H's singer", not "Band H sings".
SINGER_NAME_ANY_WORDING = True
# Musiclyse dev20 GROUP_LEAD_STAYS_SEVERAL, recomputed for older saves: an
# identity that says one lead while its reasoning documents a vocal group (boy
# band, girl group, brothers, sisters, a duo ...) and names no single lead
# singer means a shared lead (the sibling-band test track: several in dev17, one in dev19).
GROUP_LEAD_STAYS_SEVERAL = True
# Musiclyse dev20 MOSS_LOW_TRUST_FIELDS, recomputed for older saves: a genre
# decision resting on MOSS-Music's stock genre list ("indie rock; power pop;
# jangle pop", the example its own prompt gives) falls back to the family the
# PANNs tags support ("rock", "pop", with a lean only when other evidence names
# one), and those stock labels and non-genre words ("layered production") are
# not listed as close neighbours.
MOSS_LOW_TRUST_FIELDS = True
# Musiclyse dev20 GENRE_SYNTH_ROCK, recomputed for older saves: the dev56
# "post-punk / new wave" reading of a synth-pop / new wave first impression is
# "synth-rock" (rock with a prominent synth lead) when the separated keys/synth
# part leads the band, and synth-pop / new wave stay fair comparisons instead of
# labels to avoid (the synth-rock test track; the post-punk test track keeps post-punk / new wave).
GENRE_SYNTH_ROCK = True
# Musiclyse dev20 PITCH_CHECK_ROUNDED_FIGURES: the low-confidence pitch check
# allows for the saved figures being rounded to whole hertz (the synth-rock test track: tracker G4
# from 4% voiced frames against the notes' E4, 2.98 semitones as written).
PITCH_CHECK_ROUNDED_FIGURES = True
PITCH_CHECK_ROUNDING_TOL_ST = 0.1
# 2.2.3: when that check sets the pitch tracker's median aside, a by-ear pitch
# clause quoting it ("range spans from B3 to A4, centered around G4" -- the
# vocal pass was shown the tracker's figures) is left out of "How it sounds",
# so it no longer contradicts "Who's singing".
BY_EAR_PITCH_FOLLOWS_CHECK = True
# Analyser UNBROKEN VOICE CHECK: when the saved identity carries that note (a
# credited boy or teenager the listening model heard as female-sounding, kept
# as a man by the identity model), or a young male identity's reasoning itself
# records a female-sounding or unbroken voice, the voice is described as unbroken, not as
# already deepened or deepening from the audio-only male profile. False
# restores the earlier behaviour.
UNBROKEN_VOICE_IDENTITY_CHECK = True
# Analyser KEY FIT CHECK, for saves made before it: a reported key one scale
# note away from the key the saved notes fit clearly better (by KEY_FIT_MARGIN),
# whose tonic the bass sits on most, is replaced by that key. False restores
# the earlier behaviour.
KEY_FIT_CHECK = True
KEY_FIT_MARGIN = 0.05
# Analyser TEMPO OCTAVE PAIR CHECK, for saves made before it: detectors an
# octave apart, a listening-model pulse that is neither of them (nor half or
# double) kept as "~2x", and a snare backbeat at the detector reading nearest
# that pulse: the detector reading is used. False restores the earlier behaviour.
TEMPO_OCTAVE_PAIR_CHECK = True
TEMPO_OCTAVE_PAIR_TOL = 0.05

# -----------------------------------------------------------------------------
# 2.3 settings (Musiclyse 0.5-moss-dev42 to dev46). False gives 2.2.3e's output
# for that part.
# -----------------------------------------------------------------------------
# Musiclyse dev44's genre view, with its own settings: the measured styles can
# narrow or replace the recommended genre, a confident period-bound style
# corrects a "2010s-2020s" era guess, and regional labels (K-pop, Europop...)
# need the local language or a local artist; World / International labels are
# never used.
GENRE_VIEW_DEV44 = True
GENRE_STYLE_LED = True
GENRE_STYLE_REFINE_MIN = 0.35
GENRE_STYLE_REPLACE_MIN = 0.30
ERA_STYLE_CHECK = True
ERA_STYLE_MIN = 0.25
ERA_STYLE_MAX_SPAN = 25
REGIONAL_GENRE_GATE = True
# The more specific styles heard (dev42 DISCOGS STYLE SIGNAL), named in "The
# sound" without their percentages.
STYLES_IN_DOSSIER = True
# dev45 blocks: the measured chord progression, the melody against the key and
# how melodic the voice is, section by section.
CHORD_PROGRESSION_IN_DOSSIER = True
MELODY_IN_KEY_IN_DOSSIER = True
MELODICISM_IN_DOSSIER = True
# One drum grid for each part whose groove differs (up to GROOVE_GRIDS_MAX),
# instead of a single grid, in "The groove" (standard and above).
GROOVE_GRIDS_BY_PART = True
GROOVE_GRIDS_MAX = 4
# Musiclyse dev46 IDENTITY_FLIP_NOTE, for saves made before it: when the first
# vocal pass and the voice descriptions heard a woman, only the confirmation
# pass a man, and the saved identity simply followed that male reading (no
# singer named), the dossier does not state the singer's gender.
IDENTITY_FLIP_GUARD = True
# Musiclyse dev46 TEMPO_COPY_CHECK, for saves made before it: a listening-model
# tempo that only copied a detector reading is not a second opinion, so a
# half/double-time choice it decided is made again from the detectors
# (Dancing Queen: 100 BPM, not 200).
TEMPO_COPY_CHECK = True
# Wording: "How it unfolds" says a repeated groove, tune shape or instrument
# list once instead of in the same words for every part, and the guide asks for
# fresh wording rather than the notes' phrases and stock music-review lines.
VARIED_WORDING = True
# Guide: the sound is described from what was heard only; what is known about
# the artist, album or year is casual context beside it, never a description of
# the sound.
SOUND_BEFORE_KNOWLEDGE = True

# Rough chars-per-token for this kind of text (numbers, note names, tables).
# Deliberately a little pessimistic so a file reported as fitting really fits.
CHARS_PER_TOKEN = 3.5

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
STEM_ORDER = ["vocals", "drums", "bass", "guitar", "piano", "other"]


# =============================================================================
# Small helpers
# =============================================================================

def est_tokens(text):
    return int(len(text or "") / CHARS_PER_TOKEN) + 1


def mmss(sec):
    """M:SS exactly as Musiclyse prints it (_seconds_to_mmss): truncated like a
    player's elapsed-time counter, with 0.2 s of beat-grid slack. Rounding
    (as before 2.0) put Verse 1 at 0:09 where Musiclyse's own map says 0:08."""
    try:
        sec = max(0.0, float(sec))
    except (TypeError, ValueError):
        return "?"
    total = int(math.floor(sec + 0.2))
    m, s = divmod(total, 60)
    return f"{m}:{s:02d}"


def midi_name(m):
    m = int(round(float(m)))
    return f"{NOTE_NAMES[m % 12]}{m // 12 - 1}"


def strip_md(text):
    return re.sub(r"[*`]+", "", text or "")


def clip(text, n):
    text = (text or "").strip()
    if len(text) <= n:
        return text
    cut = text[:n].rsplit(" ", 1)[0]
    return cut.rstrip(" ,;:") + " …"


def between(text, start, stops=(), max_chars=8000):
    """Text after the first `start` marker, up to the earliest of `stops`."""
    i = text.find(start)
    if i < 0:
        return ""
    rest = text[i + len(start):]
    end = len(rest)
    for s in stops:
        j = rest.find(s)
        if 0 <= j < end:
            end = j
    return rest[:min(end, max_chars)]


def first_line_value(text, prefix, indented_ok=False):
    """Value of the first line that starts with `prefix` (unindented by default)."""
    for line in (text or "").splitlines():
        cand = line.strip() if indented_ok else line
        if cand.startswith(prefix):
            return cand[len(prefix):].strip()
    return ""


def drop_parens(text, words):
    """Remove balanced (...) groups that contain any of `words`.

    Used to strip detector internals ("grid-lock 0.71", "~0.98 kicks/beat")
    from otherwise useful plain-language labels.
    """
    s = text or ""
    out = []
    i, n = 0, len(s)
    while i < n:
        if s[i] == "(":
            depth, j = 0, i
            while j < n:
                if s[j] == "(":
                    depth += 1
                elif s[j] == ")":
                    depth -= 1
                    if depth == 0:
                        break
                j += 1
            if j >= n:              # unbalanced: keep the rest verbatim
                out.append(s[i:])
                break
            group = s[i:j + 1]
            if any(w in group for w in words):
                while out and out[-1] == " ":
                    out.pop()
            else:
                out.append(group)
            i = j + 1
            continue
        out.append(s[i])
        i += 1
    return re.sub(r"\s{2,}", " ", "".join(out)).strip()


_DETECTOR_WORDS = (
    "grid-lock", "kicks/beat", "snares/beat", "IOI", "alternation", "treat as",
    "not irregular", "phase offset", "contrast", "beat coverage", "rate-only",
    "do not call", "no locked kick", "not a free-time", "prefer texture",
)


def clean_label(text):
    return drop_parens(text, _DETECTOR_WORDS).rstrip(" ;,.")


def _clean_hats(text):
    """'~4.0x, ~4.0 per beat (very dense (16th-note hats); 812 hat hits)' -> 'very dense (16th-note hats)'."""
    m = re.search(r"\((.*);\s*\d+ hat hits\)\s*$", text or "")
    return m.group(1).strip() if m else clean_label(text)


# =============================================================================
# Plain-language mappings
# =============================================================================

VOCAL_TAG_PLAIN = {
    "child_male_likely": "a young boy's voice (not yet broken)",
    "child_female_likely": "a young girl's voice",
    "child_gender_uncertain": "a child's voice (boy or girl unclear)",
    "adolescent_male_likely": "a young male voice that still sounds like it's changing",
    "young_male": "a young male voice",
    "post_puberty_male": "an adult male voice",
    "adult_male": "an adult male voice",
    "female_teen_adult": "a female voice",
    "adult_female": "a female voice",
    "young_female": "a young female voice",
    "mixed_leads": "two or more distinct lead singers",
    "uncertain": "a lead voice whose age/gender can't be pinned down from the audio",
    "unknown": "a lead voice whose age/gender wasn't determined",
}
# Short forms used when rewriting free text that leaked a tag.
VOCAL_TAG_SHORT = {
    "adolescent_male_likely": "young, still-changing male",
    "post_puberty_male": "adult male",
    "child_male_likely": "young male",
    "child_female_likely": "young female",
    "child_gender_uncertain": "child",
    "female_teen_adult": "female",
    "mixed_leads": "two lead singers",
    "adult_male": "adult male",
    "adult_female": "adult female",
}
MALE_TAGS = {"child_male_likely", "adolescent_male_likely", "young_male",
             "post_puberty_male", "adult_male"}
FEMALE_TAGS = {"child_female_likely", "female_teen_adult", "adult_female", "young_female"}

# Internal tool names -> neutral wording, so a stock model never tells the
# user "Demucs says..." or "according to PANNs...".
TERM_MAP = [
    (r"\bMOSS[- ]Music(?:-8B-Instruct)?\b", "the listening model"),
    (r"\bMOSS\b", "the listening model"),
    (r"\bMusic Flamingo\b", "the listening model"),
    (r"\bMF(?:'s)?\b", "the listening model"),
    (r"\bPANNs(?:/AudioSet)?\b", "the audio classifier"),
    (r"\bAudioSet\b", "audio-classifier"),
    (r"\bEssentia\b", "signal analysis"),
    (r"\blibrosa\b", "signal analysis"),
    (r"\bDemucs\b", "stem separation"),
    (r"\bOmnizart\b", "note transcription"),
    (r"\bWhisper\b", "speech recognition"),
    (r"\bpyin\b", "pitch tracking"),
    (r"\bGENRE_RANKED\b", "the genre ranking"),
]

INTENSITY_SHORT = {
    "not playing (below audibility floor)": "not playing",
    "peak intensity": "peak",
    "prominent": "prominent",
    "moderate": "moderate",
    "present but restrained": "restrained",
    "quiet / mostly absent": "quiet/absent",
    "steady": "steady",
}

PROD_TITLES = [
    ("STEREO IMAGE / MIX WIDTH", "Stereo image"),
    ("MASTERING: TRUE PEAK / CLIPPING", "Mastering peaks & clipping"),
    ("BANDWIDTH / SOURCE PROVENANCE", "Bandwidth / file provenance"),
    ("BEAT-SYNCHRONOUS DYNAMICS / PUMPING", "Sidechain / pumping"),
    ("AMBIENCE / REVERB CHARACTER", "Ambience / reverb"),
    ("WHOLE-TRACK SPECTRAL BALANCE", "Spectral balance (whole track)"),
    ("PRODUCTION-CHARACTER TAGS", "Production-character tags"),
]
PROD_STANDARD_KEEP = (
    "reading", "note:", "caution", "assessment", "true peak:", "upper frequency limit",
    "channel format", "cutoff character", "decay proxy", "clipped-sample runs",
    "low-end imaging", "mono compatibility", "production-character tags",
    "whole-track brightness",
)

# Stock ambience-reading labels from Musiclyse's decay proxy (_prod_ambience /
# _reading). Left as measurement jargon they get repeated by stock chat models
# ("washy reverb") in every answer. Plain wording only — underlying levels and
# the numeric decay lines are unchanged.
AMBIENCE_PLAIN = (
    ("very washy/reverb-heavy, long tails",
     "plenty of reverb, with long tails"),
    ("very washy/reverb-heavy",
     "plenty of reverb, with long tails"),
    ("noticeably reverberant — large room/hall/plate feel",
     "noticeable reverb — a large-room, hall or plate character"),
    ("noticeably reverberant",
     "noticeable reverb — a large-room, hall or plate character"),
    ("moderate room/plate ambience",
     "a moderate sense of room or plate"),
    ("dry/tight — close, damped, or gated production",
     "dry and close, with little sense of room"),
    ("dry/tight",
     "dry and close, with little sense of room"),
)

# Stock stereo-width labels from Musiclyse's _prod_stereo_field. The old map
# only covered a few keys and missed the strings the analyser actually writes
# ("conventional stereo width", "very wide / heavily decorrelated", …).
STEREO_WIDTH_PLAIN = (
    ("very narrow / essentially centred", "a very narrow, centred image"),
    ("very narrow", "a very narrow, centred image"),
    ("narrow-to-moderate width", "a fairly narrow stereo image"),
    ("conventional stereo width", "a normal stereo image"),
    ("wide stereo image", "a wide stereo image"),
    ("very wide / heavily decorrelated", "a very wide, open stereo image"),
    ("very wide", "a very wide, open stereo image"),
    # Older / alternate wordings still seen in some saves:
    ("moderate width", "a moderately wide stereo image"),
    ("near-mono", "a near-mono image"),
    ("wide", "a wide stereo image"),
)


def plain_ambience(text):
    """Rewrite a stock ambience-reading label into plain listening language.

    Returns the rewritten phrase, or None when the text is empty / not a known
    stock label (callers keep the original wording in that case).
    """
    t = (text or "").strip()
    if not t:
        return None
    low = t.lower()
    head = re.split(r"\s*[—;(]\s*", t)[0].strip()
    head_low = head.lower()
    for stock, plain in AMBIENCE_PLAIN:
        s = stock.lower()
        if low == s or low.startswith(s) or head_low == s or head_low.startswith(s):
            return plain
    return None


def plain_stereo_width(text):
    """Rewrite a stock stereo-width reading into plain listening language."""
    t = (text or "").strip()
    if not t:
        return None
    head = re.split(r"\s*[—;(]\s*", t)[0].strip()
    low = head.lower()
    for stock, plain in STEREO_WIDTH_PLAIN:
        s = stock.lower()
        if low == s or low.startswith(s):
            return plain
    return None


MF_FIELDS = [
    "GENRE_RANKED", "GENRE_ADJACENT", "GENRE_RULED_OUT", "KEY", "TEMPO_BPM",
    "CHORDS", "INSTRUMENTATION", "DRUMS", "TIMBRE", "MOOD_VIBE", "MELODICISM",
    "MELODIC_CONTOUR", "MELODIC_RANGE", "VOCALS_PRESENT",
    "LEAD_VOCAL_CHARACTERISTICS", "LYRICS_PRESENT", "LYRIC_SUBJECT",
    "PRODUCTION_NOTES", "REFERENCE_POINTS",
]

DRUM_KIT_PREFIXES = [
    ("hit breakdown by drum type:", "Hits by type"),
    ("kit detail (specific instruments identified in the transcription):", "Kit detail"),
    ("hi-hat articulation:", "Hi-hat articulation"),
    ("backbeat voice:", "Backbeat voice"),
    ("cymbal voices:", "Cymbals"),
    ("auxiliary percussion present:", "Auxiliary percussion"),
]
DRUM_SOUND_PREFIXES = [
    ("kick sound:", "Kick"),
    ("kick hit-to-hit consistency:", "Kick consistency"),
    ("snare/backbeat sound:", "Snare / backbeat"),
    ("snare hit-to-hit consistency:", "Snare consistency"),
    ("hi-hat sound:", "Hi-hats"),
    ("hit-velocity spread (90th-10th pct):", "Velocity spread"),            # Musiclyse <= dev29
    ("hit-level spread measured from the drums audio (90th-10th pct, per drum):", "Hit-level spread"),
    ("drum-bus peak-to-RMS:", "Drum bus"),                                    # Musiclyse <= dev29
    ("drum-bus peak-to-RMS (while the kit is playing):", "Drum bus"),
]


# =============================================================================
# Text parsing
# =============================================================================

# A field label at the start of a line. Musiclyse's listening model writes its
# labels in any case, with spaces or underscores, after a list marker or number
# ("Genre_Ranked=", "Key= F major", "1. MOOD_VIBE=", "VOCAL PRESENCE: yes",
# "NUM_DISTINCT VOICES: 1"). Before 2.0 only upper-case, space-free labels were
# read, so a dev59 save's "Genre_Ranked=", "Key=" and "Tempo_BPM=" were missed
# and the raw header lines were exported as the "overall impression".
_LABEL_RE = re.compile(
    r"^\s*(?:[-•*>]\s*|\d{1,2}[.)]\s+)?([A-Za-z][A-Za-z0-9_]*(?: [A-Za-z][A-Za-z0-9_]*){0,3})\s*([=:])\s?(.*)$")
# A multi-word all-caps label or header ("MISSING MARKERS=...", "SUPPLEMENTARY
# MEASURED EVIDENCE (...)") also ends the previous field. Without this the era
# EVIDENCE value swallowed the two lines after it.
_CAPS_HDR_RE = re.compile(r"^\s*[A-Z][A-Z0-9_]+(?:[ /&-]+[A-Z][A-Z0-9_]+)+\s*(?:[=:(]|$)")

# Spelling variants Musiclyse itself treats as one field.
LABEL_ALIASES = {
    "GENRE_RULE_OUT": "GENRE_RULED_OUT", "GENRE_RULEDOUT": "GENRE_RULED_OUT",
    "GENRERANKED": "GENRE_RANKED", "GENRE": "GENRE_RANKED",
    "TEMPO": "TEMPO_BPM", "BPM": "TEMPO_BPM", "MUSICAL_KEY": "KEY",
    "MOOD": "MOOD_VIBE", "MOODVIBE": "MOOD_VIBE", "VOCALS": "VOCALS_DESCRIPTION",
    "NUM_DISTINCT_VOICES": "NUM_DISTINCT_VOICES", "CO_LEAD_DETAIL": "CO_LEAD_DETAIL",
    "MULTI_VOICE_EVIDENCE": "MULTI_VOICE_EVIDENCE",
}


def _label_of(m):
    """Canonical label of a _LABEL_RE match, or None when the line is prose
    ("Note: ...", "Reasoning: ..."): a mixed-case label counts only with '='."""
    raw, sep = m.group(1), m.group(2)
    if sep == ":" and raw != raw.upper() and "_" not in raw:
        return None
    if " " in raw and sep == "=" and raw != raw.upper() and "_" not in raw:
        return None
    lab = re.sub(r"[ _]+", "_", raw.strip()).upper()
    if len(lab) < 3:
        return None
    return LABEL_ALIASES.get(lab, lab)


def parse_fields(text, wanted, max_len=1200):
    """Parse `LABEL=value` fields (value may continue onto following lines).
    First occurrence of each label wins."""
    wanted = set(wanted)
    out = {}
    cur, buf = None, []

    def done():
        if cur and cur not in out:
            v = " ".join(b.strip() for b in buf if b.strip())
            v = v.strip().strip("[]").strip()
            if v:
                out[cur] = clip(v, max_len)

    for line in strip_md(text).splitlines():
        m = _LABEL_RE.match(line)
        lab = _label_of(m) if m else None
        if lab:
            done()
            cur = lab if lab in wanted else None
            buf = [m.group(3)]
            continue
        if cur is not None and _CAPS_HDR_RE.match(line):
            done()
            cur, buf = None, []
            continue
        if cur is None:
            continue
        if not line.strip():
            done()
            cur, buf = None, []
            continue
        buf.append(line)
    done()
    return out


def header_blocks(text, header_re, mode="indented"):
    """Find every line matching `header_re` and return (match, body_lines).

    mode="indented": body = following lines that start with whitespace.
    mode="blank":    body = following lines up to the next blank line.
    """
    rx = re.compile(header_re)
    lines = (text or "").splitlines()
    out = []
    for i, line in enumerate(lines):
        m = rx.match(line)
        if not m:
            continue
        body = []
        for l2 in lines[i + 1:]:
            if mode == "indented":
                if l2.strip() and l2[:1] in (" ", "\t"):
                    body.append(l2[2:] if l2.startswith("  ") else l2.lstrip())
                    continue
                break
            if not l2.strip():
                break
            body.append(l2.rstrip())
        out.append((m, body))
    return out


_GRID_ROW = re.compile(r"^(\s*)(count|kick|snare|hi-hat|toms|cymbals|perc|other)\s*: ")
_GRID_HDR = re.compile(r"^\s*BEAT GRID(?: — (.+?))? \((\d+ bars? from [^)]*)\)")


def extract_grids(lines):
    """Beat-grid diagrams: header line + pre-aligned rows, dedented as a unit."""
    grids = []
    for i, line in enumerate(lines):
        m = _GRID_HDR.match(line)
        if not m:
            continue
        rows, indent = [], None
        for l2 in lines[i + 1:]:
            r = _GRID_ROW.match(l2)
            if not r:
                break
            if indent is None:
                indent = len(r.group(1))
            rows.append(l2[indent:].rstrip())
        if rows:
            grids.append({"label": (m.group(1) or "").strip(),
                          "meta": m.group(2).strip(), "rows": rows})
    return grids


# Stock lines speech recognition invents on music with no singing (same list
# as Musiclyse 0.4-dev31). Saves made before dev31 still contain them.
_HALLUCINATION_PATTERNS = [
    re.compile(r"\bthank(?:s| you)(?: (?:so|very) much)?,? for watching(?=\s*(?:[.!?,]|$))[.!]*", re.I),
    re.compile(r"\bplease (?:like,? (?:and|&) )?subscribe\b[^.!?\n]*[.!]*", re.I),
    re.compile(r"\blike (?:and|&) subscribe\b[.!]*", re.I),
    re.compile(r"\bsubscribe to (?:my|the|our) channel\b[.!]*", re.I),
    re.compile(r"\bsubtitles? (?:by|created by|provided by|made by)\b(?:[^.!?\n]|\.(?=\w))*[.!?]?", re.I),
    re.compile(r"\b(?:the )?amara\.org\b(?:[^.!?\n]|\.(?=\w))*[.!?]?", re.I),
    re.compile(r"\b(?:transcribed|transcription|captions?|captioned) by\b(?:[^.!?\n]|\.(?=\w))*[.!?]?", re.I),
    re.compile(r"\[\s*(?:music|applause|laughter|instrumental|silence|no speech|blank_audio)\s*\]", re.I),
    re.compile(r"\(\s*[a-z ]{0,20}\bmusic\b[a-z ]{0,20}\)", re.I),
    re.compile(r"[\u266a\u266b\u266c]+"),
    re.compile(r"\bwe'?ll be right back\b[.!]*", re.I),
    re.compile(r"©\s*(?:transcript|subtitles?|captions?)\b[^\n.!?]*[.!?]?", re.I),
    re.compile(r"\bsee you (?:(?:in|on) (?:the )?)?next (?:time|video|week|one)\b[.!]*", re.I),
    re.compile(r"\bdon'?t forget to (?:like|subscribe)\b[^.!?\n]*[.!]*", re.I),
    re.compile(
        r"(?:terima kasih (?:telah|sudah|karena) menonton|gracias por (?:ver|mirar)(?: el video)?"
        r"|obrigad[oa] por (?:assistir|ver)|danke (?:f[\u00fcu]rs|f[\u00fcu]r'?s) zuschauen"
        r"|merci d'avoir regard[\u00e9e]|grazie per la visione|bedankt voor het kijken"
        r"|\u0441\u043f\u0430\u0441\u0438\u0431\u043e \u0437\u0430 \u043f\u0440\u043e\u0441\u043c\u043e\u0442\u0440"
        r"|c\u1ea3m \u01a1n (?:c\u00e1c b\u1ea1n )?\u0111\u00e3 (?:xem|theo d\u00f5i)"
        r"|\u3054\u8996\u8074\u3042\u308a\u304c\u3068\u3046\u3054\u3056\u3044\u307e\u3057\u305f"
        r"|\uc2dc\uccad\ud574\s*\uc8fc\uc154\uc11c\s*\uac10\uc0ac\ud569\ub2c8\ub2e4"
        r"|\u611f\u8c22(?:\u60a8\u7684)?(?:\u6536\u770b|\u89c2\u770b)|\u8c22\u8c22(?:\u6536\u770b|\u89c2\u770b))[.!\u3002\uff01]*",
        re.I,
    ),
    # One short unit repeated 12+ times back to back (a run of one syllable).
    # Spaced vocables like "na na na" never match.
    re.compile(r"(\S{1,8}?)\1{11,}"),
]


def strip_hallucinations(text):
    out = str(text or "")
    for pat in _HALLUCINATION_PATTERNS:
        out = pat.sub(" ", out)
    return out


def clean_lyrics_transcription(text):
    t = text or ""
    t = re.sub(r"\[TRANSCRIPTION CAUTION[^\]]*\]", "", t, flags=re.DOTALL)
    t = re.sub(r"\[END OF TRANSCRIPTION[^\]]*\]", "", t)

    def _unverified(m):
        inner = strip_hallucinations(m.group(1)).strip()
        return f"{inner} (?)" if re.search(r"[^\W_]", inner) else " "
    t = re.sub(r"\[UNVERIFIED→\s*(.*?)\s*←[^\]]*\]", _unverified, t, flags=re.DOTALL)
    t = strip_hallucinations(t)
    t = re.sub(r"(?m)^[ \t.,!?;:-]+$", "", t)          # lines left with only punctuation
    t = re.sub(r"[ \t]{2,}", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    t = t.strip()
    # Nothing left that is a word -> nothing was sung.
    return t if re.search(r"[^\W_]", t) else ""


_KEEP_OPEN, _KEEP_CLOSE = "\ue000", "\ue001"


def verbatim(text):
    """Mark text that humanize() must leave exactly as it is: titles, names, lyrics and other
    words from the file itself. (Before 1.2, a song called "a ballad title" was exported as
    "Careless speech recognition", and a stage name containing an ordinary word being rewritten as though it were an internal label.)"""
    return f"{_KEEP_OPEN}{text}{_KEEP_CLOSE}" if text else text


def _humanize_free(out):
    for pat, rep in TERM_MAP:
        out = re.sub(pat, rep, out)
    for tag, rep in VOCAL_TAG_SHORT.items():
        out = re.sub(r"\b" + re.escape(tag) + r"\b", rep, out)
    return re.sub(r"\bpost[-\s]?pub(?:erty|ertal|escent)\b", "adult", out, flags=re.I)


def humanize(text):
    """Final pass: neutral tool names, plain vocal-category words, tidy spacing.
    Spans marked with verbatim() are passed through untouched."""
    pieces = re.split(f"({_KEEP_OPEN}.*?(?:{_KEEP_CLOSE}|$))", text, flags=re.S)
    out = "".join(p.strip(_KEEP_OPEN + _KEEP_CLOSE) if p.startswith(_KEEP_OPEN) else _humanize_free(p)
                  for p in pieces)
    out = out.replace(_KEEP_OPEN, "").replace(_KEEP_CLOSE, "")
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip() + "\n"


# =============================================================================
# Track parsing
# =============================================================================

# ---- vocal presence ---------------------------------------------------------

# The listening model's opening description stops at the first of these blocks.
_MAIN_PASS_STOPS = (
    "\n\nLYRICS HEARD IN AUDIO", "\n\nFULL LYRICS TRANSCRIPTION", "\n\nLYRICS:", "\n\n11. ERA",
    "\n\nERA / RELEASE PERIOD", "\n\nVOCAL / SINGER PROFILE", "\n\nVOCAL PRESENCE ARBITRATION",
    "\n\nVOCAL DECISION AUDIT", "\n\nRECOMMENDED ", "\n\n[Independent",
)

# Same wording test as Musiclyse 0.4-dev31's _INSTRUMENTAL_PROSE_RE: a
# statement about the whole recording, never "an instrumental intro/break".
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

# A presence FIELD: at the start of a line, "VOCALS_PRESENT=yes" or the vocal
# pass's "VOCAL PRESENCE: yes". The old test searched for VOCALS_PRESENT=yes/no
# anywhere, so it also matched explanatory prose such as "(if VOCALS_PRESENT=no,
# this onset is separation residue)", which Musiclyse prints under the vocal
# stem of sung tracks too.
_PRESENCE_FIELD_RE = re.compile(
    r"(?im)^[ \t]*(?:[-•][ \t]*)?VOCALS?[_ ]PRESEN(?:T|CE)[ \t]*[=:][ \t]*(yes|no)\b")

VOCAL_EVIDENCE_MIN_VOICED = 0.08    # Musiclyse's VOCAL_PITCH_EVIDENCE_MIN_VOICED_FRACTION


def main_pass_text(A):
    """The listening model's own opening description: no GENRE CHECK line
    (dev56 writes it at the very top), no duration sentence, no STRUCTURE=[...]
    list and no lyrics-status line."""
    end = len(A)
    for stop in _MAIN_PASS_STOPS:
        j = A.find(stop)
        if 0 <= j < end:
            end = j
    t = A[:end]
    # dev64+: CHECKS block at the very top (INSTRUMENT CHECK, GENRE CHECK, …)
    t = re.sub(
        r"(?ms)^CHECKS \(corrections to read before the report below\):\s*"
        r".*?(?=\nGENRE_RANKED\s*[=:]|\n[A-Z][A-Z0-9_]+\s*[=:])",
        "",
        t,
    )
    t = re.sub(
        r"(?m)^(?:GENRE CHECK|ARRANGEMENT CHECK|VOICE COUNT CHECK|INSTRUMENT CHECK|GUITAR CHECK|GENRE ERA CHECK)"
        r" \([^\n]*(?:\n(?!\n)[^\n]*)*",
        "",
        t,
    )
    t = re.sub(r"(?im)^\s*(?:[-•*]\s*)?STRUCTURE\s*[=:]\s*\[.*?\]", "", t, flags=re.S)
    t = re.sub(r"(?im)^The duration of the piece is [\d.]+ seconds\.?\s*$", "", t)
    t = re.sub(r"(?im)^LYRICS:\s*reliable lyrics[^\n]*$", "", t)
    # REPORT_NOTE is pipeline metadata, not listening prose
    t = re.sub(r"(?im)^REPORT_NOTE\s*[=:].*$", "", t)
    return t.strip()


def _main_says_instrumental(main):
    flags = [m.group(1).lower() for m in _PRESENCE_FIELD_RE.finditer(main)]
    if flags:
        return flags[0] == "no"
    return bool(_INSTRUMENTAL_PROSE_RE.search(main))


def _vocal_stem_is_weak(A, prominence, intensity_notes):
    """Reasons the isolated vocal part holds no real voice (empty list = none).
    Uses only what older saves recorded: voiced-frame share, pitch source,
    and the stem's level labels."""
    reasons = []
    m = re.search(r"(?m)^[ \t]*-?[ \t]*voiced frames:\s*([\d.]+)\s*%", A)
    if m and float(m.group(1)) / 100.0 < VOCAL_EVIDENCE_MIN_VOICED:
        reasons.append(f"the vocal part is pitched in only {float(m.group(1)):.1f}% of the audio")
    if re.search(r"(?im)^[ \t]*-?[ \t]*measurement source:\s*full mix", A) or \
            re.search(r"(?m)^- No reliable voiced pitch detected", A):
        reasons.append("no usable voiced pitch was measured on the vocal part")
    if any(p.startswith("vocals (silent") for p in prominence):
        reasons.append("the vocal part is silent/negligible in the mix")
    if "vocals" in intensity_notes:
        reasons.append("the vocal part is near-silent throughout")
    return reasons


def decide_vocal_presence(A, prominence, intensity_notes):
    """(present, how, reasons). present is "yes" / "no" / "uncertain".

    Order of authority, matching Musiclyse 0.4-dev31:
      1. a VOCAL PRESENCE ARBITRATION verdict (dev31+);
      2. the same arbitration re-applied to an older save: the listening
         model's own description says the piece is instrumental AND the vocal
         part is measurably weak -- the vocal pass's "singer" was a false
         positive (seen as a "child" duet on an instrumental in dev30);
      3. a VOCAL PRESENCE SALVAGE note (a voice the vocal pass missed);
      4. explicit presence fields (any "yes" wins over "no").
    """
    flags = {f.lower() for f in _PRESENCE_FIELD_RE.findall(A)}
    m = re.search(r"(?m)^VOCAL PRESENCE ARBITRATION:\s*INSTRUMENTAL\.?\s*(.*)$", A)
    if m:
        why = re.split(r"\.\s+Whatever the vocals stem", m.group(1))[0].strip().rstrip(".")
        why = why[:1].lower() + why[1:]
        # "overruled" only when the vocal pass had claimed a voice (its output
        # is then marked SUPERSEDED); otherwise the passes simply agreed.
        how = "arbitration" if "[SUPERSEDED" in A[m.end():m.end() + 400] else "measured"
        return "no", how, [why] if why else []
    main = main_pass_text(A)
    if main and _main_says_instrumental(main):
        weak = _vocal_stem_is_weak(A, prominence, intensity_notes)
        if weak:
            lead = re.search(r"FINAL LEAD PROFILE:\s*([A-Za-z_]+)", A)
            claimed = "yes" in flags or (lead and lead.group(1).lower() not in ("unknown", "uncertain", "none"))
            return ("no", "rechecked" if claimed else "measured",
                    ["the listening model describes the track as instrumental"] + weak)
    if re.search(r"(?m)^VOCAL PRESENCE SALVAGE", A):
        return "yes", "salvage", []
    if "yes" in flags:
        return "yes", "flag", []
    if "no" in flags:
        return "no", "flag", []
    return "uncertain", "", []


# ---- structure name checks ----------------------------------------------------

def same_name_different_material(sections):
    """{index: note} -- same rule as Musiclyse 0.4-dev31: a name used on 3+
    sections that are all different acoustic material is an interpretation,
    not a repeat. Lets older saves (which kept the cluster letters but not the
    check) get the note too."""
    groups = {}
    for i, sec in enumerate(sections):
        name, cl = str(sec.get("name") or "").strip(), str(sec.get("cluster") or "").strip()
        if not name or not cl:
            continue
        base = re.sub(r"\s*\d+\s*$", "", name).strip().lower()
        if base:
            groups.setdefault(base, []).append((i, cl, name))
    out = {}
    for items in groups.values():
        if len(items) < 3:
            continue
        cls = [c for _i, c, _n in items]
        named = [c for c in cls if c.lower() != "unique"]
        if len(set(named)) + (len(cls) - len(named)) < len(items):
            continue
        label = re.sub(r"\s*\d+\s*$", "", items[0][2]).strip()
        out[items[0][0]] = (f"the {len(items)} sections called \"{label}\" are {len(items)} different pieces "
                            f"of music (none of them comes back), so the shared name is a guess — treat "
                            f"them as distinct parts and don't say the {label.lower()} returns")
        for i, _c, _n in items[1:]:
            out[i] = f"different music from the other \"{label}\" sections — see {items[0][2]}"
    return out


# ---- bass claims vs the bass part ----------------------------------------------

_BASS_CLAIM_RE = re.compile(r"(?i)\b(?:bass\s*guitar|bass\s*line|bassline|electric\s+bass|bass\s+(?:is|plays|riff|part|groove))\b")


# ---- production readings that older Musiclyse versions over-stated -------------

_OLD_SIDECHAIN = "ducks strongly in lockstep with the beat — consistent with sidechain compression"
_OLD_LOSSY_CAUTION = "provenance caution: this file has been through lossy compression"
_OLD_HIGH_BW = "bandwidth reading: consistent with high-bitrate lossy encoding (~256 kbps+) or a gently filtered master"


def correct_old_production(prod):
    """Bring readings from pre-dev31 saves in line with what dev31 says now.
    Musiclyse dev31 found two over-claims: strummed on-beat guitar was read as
    sidechain pumping, and a shelf at 19 kHz+ was called proof of lossy
    compression (resamplers and mastering filters make the same shelf)."""
    cutoff = None
    for _title, lines in prod:
        for l in lines:
            m = re.match(r"upper frequency limit:\s*~?(\d+)\s*Hz", l)
            if m:
                cutoff = int(m.group(1))
    for _title, lines in prod:
        for k, l in enumerate(lines):
            if _OLD_SIDECHAIN in l and "recovers between beats" not in l:
                lines[k] = ("sidechain reading: strong beat-locked level movement. (Analysis from an older "
                            "Musiclyse version, whose test could not tell sidechain pumping apart from "
                            "strummed or stabbed chords played on the beat — treat pumping as unconfirmed "
                            "and don't use it as evidence for a dance/electronic label.)")
            elif cutoff and cutoff >= 19000 and l.startswith(_OLD_HIGH_BW):
                lines[k] = ("bandwidth reading: high-bitrate lossy encoding OR a sample-rate-conversion / "
                            "mastering / instrument low-pass — a ceiling this high is NOT proof of lossy "
                            "compression")
            elif cutoff and cutoff >= 19000 and l.startswith(_OLD_LOSSY_CAUTION):
                lines[k] = ("provenance note: the top end above the shelf is inaudible to most listeners; "
                            "don't describe the recording as lossy, degraded or transcoded on this evidence alone.")
    return prod


def _drop_stem_from_changes(lines, stem):
    """Remove one stem's entries from ARRANGEMENT CHANGE POINTS lines
    ("0:29 (2 stems): other in, vocals out") and recount the stems."""
    out = []
    for line in lines:
        m = re.match(r"^(\d+:\d\d) \((\d+) stems?\): (.+)$", line.strip())
        if not m:
            out.append(line)
            continue
        toks = [t.strip() for t in m.group(3).split(",")
                if t.strip() and not re.match(rf"{re.escape(stem)}\b", t.strip())]
        if not toks:
            continue
        n = len({re.split(r"[:\s]", t, maxsplit=1)[0] for t in toks})
        out.append(f"{m.group(1)} ({n} stem{'s' if n != 1 else ''}): " + ", ".join(toks))
    return out


def _structure_block_bodies(A):
    """Section rows of each RECOMMENDED STRUCTURE block. Musiclyse puts an
    unindented "(Use these EXACT ...)" line between the header and the
    indented rows, which made header_blocks() return an empty body -- so the
    fallback for saves without structure_sections never found any sections."""
    lines = (A or "").splitlines()
    out = []
    for i, line in enumerate(lines):
        if not line.startswith("RECOMMENDED STRUCTURE FOR DISCUSSION"):
            continue
        body, started = [], False
        for l2 in lines[i + 1:i + 200]:
            if l2.strip() and l2[:1] in (" ", "\t"):
                body.append(l2.strip())
                started = True
            elif started or not l2.strip():
                if started:
                    break
            elif not l2.startswith("("):
                break
        out.append(body)
    return out


def _tempo_confidence(note):
    n = (note or "").lower()
    if not n:
        return ""
    if "tempo check: the snare" in n:
        return "the drums' backbeat settles it"
    if "tempo check: moss-music's pulse" in n:
        return "a measured reading; the detectors didn't all agree"
    if "disagree" in n:
        return "the measurement methods disagreed; treat as the most likely pulse"
    if "agree" in n:
        return "independent measurements agree"
    if "2x" in n or "double" in n or "half" in n:
        return "half-time vs double-time was ambiguous; this is the most likely pulse"
    if "only" in n or "fallback" in n:
        return "single estimate"
    return ""


def _key_confidence(note):
    n = (note or "").lower()
    if not n:
        return ""
    if "key note check" in n:
        # dev18: the listening model's key was set aside because the notes don't fit it.
        if "as a clear reading" in n:
            return "independent methods agree"
        return "moderately confident — the transcribed notes back it"
    if "uncertain" in n and re.search(r"\b(?:mf|moss-music|music flamingo) was uncertain", n):
        return "moderately confident"
    if "agree" in n and "disagree" not in n:
        if "moss-music and essentia agree" in n:
            return "estimates agree; the listening self-check may have seen the detector's key"
        return "independent methods agree"
    if "disagree" in n or "disagreement" in n:
        return "methods disagreed — moderately confident at best"
    if "low" in n:
        return "low confidence"
    if "only" in n:
        return "single estimate"
    return ""


def _genre_hedge(reasoning):
    r = (reasoning or "")
    if "ONLY signal available" in r or "single raw" in r:
        return ("weak: based on a single audio-classifier tag that often confuses "
                "closely related rock/pop sub-genres — hedge when stating it")
    if "Families differ" in r:
        return "the listening model and the audio classifier leaned different ways"
    if "favours electronic" in r or "favours electronic/dance" in r:
        return "revised toward electronic/dance from an initial rock-leaning read"
    if "free-text description" in r:
        return "taken from the listening model's own description"
    if "aligns with independent" in r:
        return "two independent sources agree"
    return ""


# =============================================================================
# Checks Musiclyse 0.4-dev40..dev53 apply when it shows a saved scan to its
# writer (its _writer_analysis_view and blind-era helpers). They are computed
# from the saved analysis text itself, so every save -- old or new -- gets
# them. The decision logic is copied from dev53; only the wording is written
# for the dossier.
# =============================================================================

_HYPHENS_RE = re.compile(r"[\s\-‐‑‒–—]+")

_GENRE_FAMILY_ROCKISH = {
    "rock", "punk", "pop-punk", "pop punk", "emo", "metal", "heavy metal",
    "indie rock", "indie-rock", "alternative rock", "alt-rock", "alt rock",
    "hard rock", "garage rock", "post-punk", "post punk", "grunge",
    "power pop", "power-pop", "pop rock", "pop-rock", "ska",
}
_GENRE_FAMILY_ELECTRONICISH = {
    "electronic", "electronica", "edm", "house", "techno", "trance",
    "dubstep", "drum and bass", "dnb", "dance", "dance-pop", "dance pop",
    "synth-pop", "synthpop", "synth pop", "electro", "electro-pop",
    "electropop", "club", "disco", "ambient", "electronic dance music",
}
_GENRE_FAMILY_POPISH = {
    "pop", "indie pop", "indie-pop", "art pop", "dream pop", "chamber pop",
}


def _normalize_genre_token(s):
    s = (s or "").strip().lower()
    s = re.sub(r"[‐-―−]", "-", s)
    s = re.sub(r"\s+", " ", s)
    return s.replace("_", " ")


def _any_marker_present(text, markers):
    if not text:
        return False
    return any(re.search(r"\b" + re.escape(m), text) for m in markers)


def _genre_family(label):
    t = _normalize_genre_token(label)
    for fam, members in (("rockish", _GENRE_FAMILY_ROCKISH),
                         ("electronicish", _GENRE_FAMILY_ELECTRONICISH),
                         ("popish", _GENRE_FAMILY_POPISH)):
        if t in members or _any_marker_present(t, members):
            return fam
        for word in t.replace("-", " ").split():
            if word in members or any(word == mm.replace("-", " ") for mm in members):
                return fam
    if _any_marker_present(t, ("punk", "emo", "metal", "grunge", "rock")):
        return "rockish"
    if _any_marker_present(t, ("house", "techno", "edm", "electro", "synth", "dance", "trance", "dubstep")):
        return "electronicish"
    if _any_marker_present(t, ("pop",)):
        return "popish"
    return "other"


# ---- genre check (dev40 / dev47) -----------------------------------------------
_GENRE_SUPERSEDED_RE = re.compile(
    r"RECOMMENDED GENRE FOR DISCUSSION: ([^\n]+?)\. Reasoning: (?:Music Flamingo|MOSS-Music) GENRE_RANKED led "
    r"with '([^'\n]+)' \(rock/punk family\)")
_GENRE_SUPERSEDED_ALT_RE = re.compile(
    r"RECOMMENDED GENRE FOR DISCUSSION: ([^\n]+?)\. Reasoning: (?:MF|MOSS-Music) ranked '([^'\n]+)' first and '")
_GENRE_CHECK_HEAD = "GENRE CHECK (read before the description below):"


def _genre_superseded_pair(text):
    """(recommended, replaced first-pass label), or None -- as dev53."""
    t = str(text or "")
    cands = [(m.start(), m.group(1).strip(), m.group(2).strip()) for m in _GENRE_SUPERSEDED_RE.finditer(t)]
    for m in _GENRE_SUPERSEDED_ALT_RE.finditer(t):
        rec, old = m.group(1).strip(), m.group(2).strip()
        if _genre_family(old) == "rockish" and _genre_family(rec) != "rockish":
            cands.append((m.start(), rec, old))
    if not cands:
        return None
    _pos, rec, old = max(cands, key=lambda c: c[0])
    if not rec or not old or _HYPHENS_RE.sub(" ", rec.lower()) == _HYPHENS_RE.sub(" ", old.lower()):
        return None
    return rec, old


def _stem_prominence_ranks(text):
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
    """(leads, detail) as dev53: leads is True when the separated guitar part
    ranks above the drums or the bass, False when it is quiet or below both,
    None when no ranking is available."""
    ranks = _stem_prominence_ranks(text)
    if not ranks:
        return None, ""
    n = len(ranks)
    g = ranks.get("guitar")
    if not g:
        return False, "there is no separate guitar part in the balance"
    if g[1] in ("silent/negligible", "low / background"):
        return False, f"the separated guitar part is {g[1]} (#{g[0]} of {n} parts by prominence)"
    rhythm = [(s, ranks[s][0]) for s in ("drums", "bass") if s in ranks]
    if not rhythm:
        return None, ""
    if any(g[0] < r for _s, r in rhythm):
        return True, f"the separated guitar part is {g[1]} and ranks #{g[0]} of {n} parts by prominence"
    above = [s for s, (r, _b) in sorted(ranks.items(), key=lambda kv: kv[1][0]) if r < g[0]]
    above_s = (", ".join(above[:-1]) + " and " + above[-1]) if len(above) > 1 else "".join(above)
    return False, f"the separated guitar part ranks #{g[0]} of {n} parts by prominence, below the {above_s}"


def _rock_label_list(old, conj="or", with_rock=True):
    out, seen = [], set()
    for lab in ((["rock"] if with_rock else []) + ["pop-rock", str(old or "").strip()]):
        k = _HYPHENS_RE.sub(" ", lab.lower()).strip()
        if lab and k not in seen:
            seen.add(k)
            out.append(lab)
    if len(out) == 1:
        return out[0]
    return ", ".join(out[:-1]) + f" {conj} " + out[-1]


# ---- Musiclyse dev20 MOSS_LOW_TRUST_FIELDS (genre part; same rule and wording) ----------
# The descriptors Musiclyse's MOSS_FULL_ANALYSIS_PROMPT uses as its example of a
# looping list ('do not cycle "indie rock; indie pop; power pop; jangle pop"').
_STOCK_GENRE_EXAMPLES = ("indie rock", "indie pop", "power pop", "jangle pop")
_STOCK_GENRE_COMPANIONS = ("alternative rock",)
_FAMILY_WORD = {"rockish": "rock", "electronicish": "electronic", "popish": "pop"}
_STOCK_NON_STYLISTIC = {
    "christian", "indie", "vocal-led (unspecified)", "asian (unspecified)",
    "traditional/folk (unspecified)", "african", "latin", "middle eastern",
}
_GENRE_ROOT_WORDS = (
    "rock", "punk", "pop", "metal", "emo", "grunge", "indie", "alternative",
    "electronic", "electro", "house", "techno", "trance", "dubstep", "synth",
    "dance", "disco", "ambient", "r&b", "r and b", "soul", "funk", "hip hop",
    "rap", "folk", "country", "blues", "jazz", "classical", "reggae", "ska",
    "wave", "core", "americana", "gospel", "latin", "afrobeat", "world",
    "singer-songwriter", "acoustic", "orchestral", "instrumental",
)


def _stock_norm(lab):
    return re.sub(r"\s+", " ", _normalize_genre_token(lab).replace("-", " ")).strip()


def stock_genre_list(labels):
    """True when at least three of the first five labels are the prompt's example descriptors."""
    return sum(1 for x in labels[:5] if _stock_norm(x) in _STOCK_GENRE_EXAMPLES) >= 3


def _stock_label(x):
    n = _stock_norm(x)
    return n in _STOCK_GENRE_EXAMPLES or n in _STOCK_GENRE_COMPANIONS


def _panns_tags(A):
    m = re.search(r"genre-adjacent tags:\s*([^\n]+)", A or "")
    out = []
    if m:
        for part in re.split(r"(?<=%\)),\s*", m.group(1)):
            mm = re.match(r"\s*(.+?)\s*\((\d+(?:\.\d+)?)%\)\s*$", part)
            if mm:
                out.append((_normalize_genre_token(mm.group(1)), float(mm.group(2)) / 100.0))
    return out


def stock_genre_repair(A):
    """(text, changed): the RECOMMENDED GENRE block rewritten as Musiclyse dev20
    does when it rests on MOSS-Music's stock genre list."""
    t = str(A or "")
    if not MOSS_LOW_TRUST_FIELDS or "STOCK GENRE CHECK:" in t:
        return t, False
    try:
        labels = [_normalize_genre_token(x) for x in _ranked_labels(t, max_n=8)]
        if not labels or _stock_norm(labels[0]) not in _STOCK_GENRE_EXAMPLES or not stock_genre_list(labels):
            return t, False
        panns = [(a, b) for a, b in _panns_tags(t) if a not in _STOCK_NON_STYLISTIC]
        mass = {"rockish": 0.0, "electronicish": 0.0, "popish": 0.0, "other": 0.0}
        for a, b in panns:
            mass[_genre_family(a)] += b
        fam = _genre_family(labels[0])
        if fam not in _FAMILY_WORD:
            return t, False
        if panns and mass[fam] < 0.15:
            fams = [f for f in _FAMILY_WORD if mass[f] > 0.0]
            if not fams:
                return t, False
            fam = max(fams, key=lambda f: mass[f])
        label = _FAMILY_WORD[fam]
        stock = [x for x in labels[:5] if _stock_label(x)]
        lean, why = None, ""
        for lab in labels[:6]:
            if _stock_label(lab):
                continue
            nl = _stock_norm(lab)
            for a, b in panns:
                if (b >= 0.20 and a not in ("rock", "pop", "electronic") and _genre_family(a) == fam
                        and _any_marker_present(nl, (a,))):
                    lean = a
                    why = (f" MOSS-Music's own '{lab}' and the PANNs '{a}' tag ({b * 100:.0f}%) both name "
                           f"{a}, so it is kept as a lean.")
                    break
            if lean:
                break
        if not lean:
            others = [f for f in _FAMILY_WORD if f != fam and mass.get(f, 0.0) > 0.0]
            if others:
                sec = max(others, key=lambda f: mass[f])
                if mass[sec] >= 0.20 and mass[sec] >= 0.4 * mass.get(fam, 0.0):
                    lean = _FAMILY_WORD[sec]
                    tags = ", ".join(f"{a} {b * 100:.0f}%" for a, b in panns if _genre_family(a) == sec)
                    why = (f" The PANNs tags also carry a substantial {lean} component ({tags}; "
                           f"{mass[sec] * 100:.0f}% together), kept as a lean.")
        if lean:
            label = f"{label} (with {lean} lean)"
        fam_tags = ", ".join(f"{a} {b * 100:.0f}%" for a, b in panns if _genre_family(a) == fam) or "none"
        reason = (f"STOCK GENRE CHECK: MOSS-Music's ranked list ({'; '.join(stock)}) is its stock answer -- the "
                  "descriptors its own prompt uses as an example, which it returns for unrelated recordings -- so "
                  f"its specific label '{labels[0]}' is not used. The independent PANNs tags support the "
                  f"{_FAMILY_WORD[fam]} family ({fam_tags}), so the track is framed as '{_FAMILY_WORD[fam]}'.{why} "
                  "Describe the specific style from the instruments, groove, production and era evidence, not from "
                  "those stock labels.")
        top = _stock_norm(labels[0])
        hit = [False]

        def _rep(m):
            if _stock_norm(m.group(1)) != top:
                return m.group(0)
            hit[0] = True
            return (f"RECOMMENDED GENRE FOR DISCUSSION: {label}. Reasoning: {reason} This is the primary genre "
                    "framing for the user.")
        t2 = re.sub(r"RECOMMENDED GENRE FOR DISCUSSION:\s*([^\n]+?)\.\s+Reasoning:.*?This is the primary genre "
                    r"framing for the user\.", _rep, t, flags=re.S)
        return t2, hit[0]
    except Exception:
        return str(A or ""), False


# ---- Musiclyse dev20 GENRE_SYNTH_ROCK (same rule and wording) -----------------------------
_SYNTH_ROCK_LABEL = "synth-rock"
_GENRE_CHECK_HEAD_TXT = "GENRE CHECK (read before the description below):"
_SYNTH_ROCK_STORED_CHECK_RE = re.compile(
    re.escape(_GENRE_CHECK_HEAD_TXT + " the listening model's first-pass genre ranking")
    + r"[^\n]*?the electronic quality is a secondary flavour\.")


def _label_words(labels, conj="or"):
    labels = [str(x) for x in labels if x]
    if len(labels) <= 1:
        return "".join(labels)
    return ", ".join(labels[:-1]) + f" {conj} " + labels[-1]


def synth_lead_reading(A):
    """(rank, of, poly) when the separated keys/synth part is moderate or
    prominent, above both the bass and the guitar, and chordal, else None."""
    t = str(A or "")
    i = t.find("STEM ACTIVITY / PROMINENCE (ranked")
    if i < 0:
        return None
    e = t.find("\n\n", i)
    rows = re.findall(r"(?m)^(\d+)\) (\w+): ([a-z /]+?) \(RMS", t[i:e if e >= 0 else len(t)])
    ranks = {}
    for r, st, prom in rows:
        ranks.setdefault(st, (int(r), prom.strip()))
    o = ranks.get("other")
    if not o or o[1] not in ("moderate", "prominent"):
        return None
    if any(st in ranks and ranks[st][0] < o[0] for st in ("bass", "guitar")):
        return None
    j = t.find("[OTHER stem]")
    if j < 0:
        return None
    k = t.find("\n\n", j)
    poly = re.search(r"average simultaneous notes: (\d+(?:\.\d+)?)", t[j:k if k >= 0 else len(t)])
    if not poly or float(poly.group(1)) < 3.0:
        return None
    return o[0], len(rows), float(poly.group(1))


def synth_rock_repair(A):
    """(text, synth-pop-ish labels or []) -- the RECOMMENDED GENRE block and the
    stored GENRE CHECK line as Musiclyse dev20 GENRE_SYNTH_ROCK writes them."""
    t = str(A or "")
    if not GENRE_SYNTH_ROCK or "SYNTH-ROCK CHECK:" in t:
        return t, []
    try:
        found = list(_GENRE_ROCKFLIP_RE.finditer(t))
        if not found:
            return t, []
        rec, top = found[-1].group(1).strip(), found[-1].group(2).strip()
        if _genre_family(rec) != "rockish" and "post-punk" not in rec.lower():
            return t, []
        olds = [_normalize_genre_token(x) for x in _ranked_labels(t, max_n=5)]
        olds = [x for x in olds if _genre_family(x) != "rockish"]
        if not olds or _HYPHENS_RE.sub(" ", olds[0].lower()) != _HYPHENS_RE.sub(" ", top.lower()):
            olds = [top] + [x for x in olds if x != top]
        synthish = [x for x in olds[:3] if re.search(r"(?i)synth|new[ -]?wave", x)]
        lead = synth_lead_reading(t)
        if not synthish or not lead:
            return t, []
        panns = [(a, b) for a, b in _panns_tags(t) if a not in _STOCK_NON_STYLISTIC]
        rock_tags = ", ".join(f"{a} {b * 100:.0f}%" for a, b in panns if _genre_family(a) == "rockish") or "rock tags"
        why = (f"SYNTH-ROCK CHECK: MOSS-Music heard {_label_words(synthish, 'and')} ('{olds[0]}' first), and the "
               f"independent PANNs tags hear a rock band ({rock_tags}). The separated parts show both are describing "
               f"part of the record: the keys/synth part is #{lead[0]} of {lead[1]} by prominence, above the bass and "
               f"the guitar, and plays chords ({lead[2]:g} notes at once on average) -- a synthesizer riff leading a "
               f"rock band. Frame it as {_SYNTH_ROCK_LABEL}: rock with a prominent synth lead. "
               f"{_label_words(synthish)} is a fair comparison for the synth sound; it is not club or dance music, "
               "and neither the rock band nor the synth should be left out of a description.")
        blk = re.compile(r"RECOMMENDED GENRE FOR DISCUSSION: " + re.escape(found[-1].group(1))
                         + r"\. Reasoning: .*?This is the primary genre framing for the user\.", re.S)
        if not blk.search(t, found[-1].start()):
            return t, []
        t = blk.sub(lambda _m: (f"RECOMMENDED GENRE FOR DISCUSSION: {_SYNTH_ROCK_LABEL}. Reasoning: {why} "
                                "This is the primary genre framing for the user."), t)
        note = (f"{_GENRE_CHECK_HEAD_TXT} the listening model's first-pass genre ranking led with '{olds[0]}'"
                + (f" (then {_label_words(olds[1:], 'and')})" if len(olds) > 1 else "")
                + f", while the independent tags hear a rock band. Both are right about part of it: describe the "
                f"track as {_SYNTH_ROCK_LABEL} -- a rock band (guitar, bass, drums, rock vocal) with a synthesizer "
                f"riff as the lead instrument. {_label_words(synthish)} is a fair comparison for the synth sound; "
                "don't present the track as pure synth-pop or as club or dance music, and don't drop the synth "
                "or the band from a description.")
        if _SYNTH_ROCK_STORED_CHECK_RE.search(t):
            t = _SYNTH_ROCK_STORED_CHECK_RE.sub(lambda _m: note, t, count=1)
        else:
            t = note + "\n\n" + t
        return t, synthish
    except Exception:
        return str(A or ""), []


# dev56: the reverse replacement -- an electronic/pop first impression the
# genre check framed as rock (or post-punk / new wave).
_GENRE_ROCKFLIP_RE = re.compile(
    r"RECOMMENDED GENRE FOR DISCUSSION: ([^\n]+?)\. Reasoning: (?:Music Flamingo|MOSS-Music)'s ranked list named only "
    r"electronic/pop labels \('([^'\n]+)' first\)")


def _ranked_labels(text, max_n=5):
    """Labels of the listening model's GENRE_RANKED list, in order."""
    f = parse_fields(main_pass_text(text) or text, ["GENRE_RANKED"]).get("GENRE_RANKED", "")
    labs = []
    for part in re.split(r"\s*;\s*|\s*\d\)\s*", f):
        # dev17 saves: "dance-pop (confidence: high)" -- the label alone
        part = re.sub(r"\s*\((?:confidence|conf)\s*[:=][^)]*\)", "", part).strip(" .,;")
        if part and part not in labs:
            labs.append(part.replace("‑", "-"))
    return labs[:max_n]


def genre_replacement(A):
    """{'rec', 'olds', 'direction', 'guitar_leads', 'guitar_detail'} when the
    genre check replaced the listening model's first impression, else {}.
    direction "to_pop": a rock/punk first impression became pop/dance/electronic
    (dev33-dev53); "to_rock": an electronic/pop one became rock or post-punk /
    new wave (dev56)."""
    t = str(A or "")
    pair = _genre_superseded_pair(t)
    if pair:
        leads, detail = _guitar_stem_reading(t)
        return {"rec": pair[0], "olds": [pair[1]], "direction": "to_pop",
                "guitar_leads": bool(leads), "guitar_detail": detail}
    found = list(_GENRE_ROCKFLIP_RE.finditer(t))
    if found:
        rec, top = found[-1].group(1).strip(), found[-1].group(2).strip()
        if _genre_family(rec) == "rockish" or "post-punk" in rec.lower():
            olds = [x for x in _ranked_labels(t) if _genre_family(x) != "rockish"]
            if not olds or _HYPHENS_RE.sub(" ", olds[0].lower()) != _HYPHENS_RE.sub(" ", top.lower()):
                olds = [top] + [x for x in olds if x != top]
            return {"rec": rec, "olds": olds[:4], "direction": "to_rock",
                    "guitar_leads": False, "guitar_detail": ""}
    return {}


def genre_check_note(A):
    """Plain-language version of dev53's GENRE CHECK line, or ""."""
    m = re.search(r"(?m)^(?:- )?" + re.escape(_GENRE_CHECK_HEAD) + r"\s*(.+)$", A)
    if m:                                   # a save that already carries the line
        return m.group(1).strip()
    pair = _genre_superseded_pair(A)
    if not pair:
        return ""
    rec, old = pair
    leads, detail = _guitar_stem_reading(A)
    rock_names = _rock_label_list(old)
    out = (f"the first impression by ear was '{old}', but checked against the independent "
           f"measurements this recording is {rec}. Describe it as {rec} in every answer — genre, era, "
           "instrumentation, drums and comparisons alike. Wording built on that first impression (a "
           f"{_rock_label_list(old, 'or', with_rock=True)} band, power chords, punk energy, a "
           f"{_rock_label_list(old, 'or', with_rock=False)} era wave) belongs to the same discarded "
           "guess: take the instruments, groove and production from the measured parts instead.")
    if leads:
        out += (f" Measured from the separated parts, {detail}, so a guitar-rock edge is a fair "
                f"secondary flavour — but lead with {rec}. Asked whether it is {rock_names}: it is "
                f"{rec} first, with a pop-rock edge at most.")
    else:
        basis = (f"Measured from the separated parts, {detail}, so" if detail
                 else "Unless the separated parts show a leading guitar,")
        out += (f" {basis} any guitar-like midrange layer is one colour inside the production, not a "
                f"rock band: don't call the track {rock_names} or describe its backing as a rock band. "
                f"Asked whether it is {rock_names}: no — it is {rec}, with at most a light guitar colour.")
    return out + f" Never present '{old}' as how this track or its artist is usually filed or known."


# ---- arrangement check (dev42) ------------------------------------------------
_ARR_KEYS_TAGS = ("accordion", "organ", "harmonium", "synth", "keyboard", "piano", "harpsichord",
                  "sampler", "pad", "electronic organ", "hammond", "clavinet", "mellotron")
_ARR_HORN_TAGS = ("brass", "saxophone", "trumpet", "trombone", "horn", "wind instrument",
                  "woodwind", "clarinet", "flute", "tuba")
_ARR_STRING_TAGS = ("string", "violin", "cello", "viola", "orchestra", "fiddle", "double bass")
_ARR_CHECK_HEAD = "ARRANGEMENT CHECK (measured from the separated stems; read before the descriptions below):"


def arrangement_check_note(A):
    """Plain-language version of dev53's ARRANGEMENT CHECK (a prominent chordal
    part in the 'other' stem), or ""."""
    t = str(A or "")
    m = re.search(r"(?m)^(?:- )?" + re.escape(_ARR_CHECK_HEAD) + r"\s*(.+)$", t)
    if m and not m.group(1).startswith("the descriptions say there are no synths"):
        return m.group(1).strip()
    if m:
        return ""                    # dev17's synth-denial reading: see arrangement_denial_note
    i = t.find("STEM ACTIVITY / PROMINENCE (ranked")
    if i < 0:
        return ""
    e = t.find("\n\n", i)
    ranks = {}
    for mm in re.finditer(r"(?m)^(\d+)\) (\w+): ([a-z /]+?) \(RMS", t[i:e if e >= 0 else len(t)]):
        ranks[mm.group(2)] = (int(mm.group(1)), mm.group(3).strip())
    if "other" not in ranks:
        return ""
    o_rank, o_prom = ranks["other"]
    if o_prom not in ("moderate", "prominent"):
        return ""
    g = ranks.get("guitar")
    if g and g[0] < o_rank:
        return ""
    j = t.find("[OTHER stem]")
    if j < 0:
        return ""
    k = t.find("\n\n", j)
    oblk = t[j:k if k >= 0 else len(t)]
    poly = re.search(r"average simultaneous notes: (\d+(?:\.\d+)?)", oblk)
    if not poly or float(poly.group(1)) < 3.0:
        return ""
    s = t.find("INTENSITY BY SECTION (other stem")
    if s < 0:
        return ""
    se = t.find("\n\n", s)
    rows = re.findall(r"(?m)^\s+(.+?) \((\d+:\d\d)-(\d+:\d\d)\): (.+)$", t[s:se if se >= 0 else len(t)])
    if not rows:
        return ""
    playing = [r[0] for r in rows if not r[3].startswith(("not playing", "quiet"))]
    if len(playing) * 2 < len(rows):
        return ""
    tag_m = re.search(r"instrument tag \(independent[^\n]*?\): ([^(\n]+?) \((\d+)%[^\[\n]*\[([^\]\n]*)\]", oblk)
    tag = tag_m.group(1).strip().lower() if tag_m else ""
    supported = bool(tag_m) and "supported by whole-mix" in tag_m.group(3)
    if tag and any(w in tag for w in _ARR_HORN_TAGS):
        kind = (f"most likely a horn section (the instrument classifier hears '{tag}'"
                + (", supported by the whole mix)" if supported else ")"))
    elif tag and any(w in tag for w in _ARR_STRING_TAGS):
        kind = (f"most likely strings (the instrument classifier hears '{tag}'"
                + (", supported by the whole mix)" if supported else ")"))
    elif tag and any(w in tag for w in _ARR_KEYS_TAGS):
        kind = (f"most likely keyboard or synth chords (the classifier's '{tag}' label"
                + (", supported by the whole mix," if supported else ", heard on this part only,")
                + " is a common reading of sustained keyboard or synth chords)")
    else:
        kind = "keyboards, synths, strings or horns — the measurements can't tell which"
    out = (f"besides the voice, drums, bass and guitar there is a {o_prom} chordal part. The "
           f"separated 'other' part — where keyboards, synths, strings and horns end up — ranks "
           f"#{o_rank} of {len(ranks)} parts by prominence" + (", above the guitar" if g else "")
           + f", plays chords of about {float(poly.group(1)):g} notes at once, and plays in "
           f"{len(playing)} of {len(rows)} sections ({', '.join(playing)}). It is {kind}. Describe the "
           "arrangement with this part in it, not as if only voice, drums, bass and guitar were "
           "playing; name the exact instrument only where other evidence supports it.")
    if re.search(r"\bno (?:additional |extra |obvious )?(?:synth(?:esizer)?s?|keyboards?|keys)\b", t, re.I):
        out += (" The by-ear remark that no synths or keyboards are evident is contradicted by this "
                "measurement; don't repeat it.")
    return out



# ---- instrument check (dev60+) ------------------------------------------------
# Musiclyse writes INSTRUMENT CHECK when the AudioSet tagger names a sustained
# synth pad / reed / horn "accordion" or "harmonica". The tags themselves stay
# in the save; the check tells the writer not to name those instruments.
_INSTRUMENT_CHECK_FALSE_POS = (
    "accordion", "harmonica", "concertina", "bandoneon", "melodeon",
)


def instrument_check_suppressed(A):
    """Labels the INSTRUMENT CHECK says are false positives (lower-case words)."""
    out = set()
    for m in re.finditer(
        r"(?im)^- INSTRUMENT CHECK[^\n]*:\s*(.+?)(?=\n- [A-Z]|\n\nGENRE_|\n\n[A-Z]|\Z)",
        A,
    ):
        body = m.group(1)
        # Explicit quoted label: 'accordion' / "harmonica"
        for lab in re.findall(r"['\"]([A-Za-z][A-Za-z \-]{2,30})['\"]", body):
            out.add(lab.strip().lower())
        # Fallback: any known false-positive word the check mentions
        low = body.lower()
        for w in _INSTRUMENT_CHECK_FALSE_POS:
            if w in low:
                out.add(w)
    return out


def _tag_label_suppressed(label, suppressed):
    if not suppressed:
        return False
    lab = (label or "").lower()
    return any(w in lab for w in suppressed)


def filter_suppressed_tags(tag_string, suppressed):
    """Drop false-positive items from a stem/whole-mix tag list string."""
    if not suppressed or not tag_string:
        return tag_string
    if tag_string.strip().lower().startswith("nothing identified"):
        return tag_string
    kept = []
    for m in _TAG_ITEM_RE.finditer(str(tag_string)):
        if not _tag_label_suppressed(m.group(1).strip(), suppressed):
            # the item exactly as written, so its [extent; whole-mix support] note survives
            kept.append(m.group(0).strip().lstrip(", ").strip())
    return ", ".join(kept) if kept else "nothing identified with confidence"


# ---- voice count check (dev43) --------------------------------------------------
_SINGLE_VOICE_RES = (
    r"NUM[_ ]DISTINCT[_ ]VOICES\s*[=:]\s*[\"'<\[]?\s*(?:1|one)\b",
    r"VOICE[_ ]ARRANGEMENT\s*[=:]\s*[\"'<\[]?\s*(?:solo[_ ]lead|lead[_ ]plus[_ ]backing|single[_ ]lead)\b",
    r"MULTI[_ \-]VOICE[_ ]EVIDENCE\s*[=:]\s*[\"'<\[]?\s*single lead",
    r"\bsingle lead (?:voice|vocal|vocalist|singer)\b",
    r"\bsingle (?:vocalist|singer)\b",
    r"\bone lead (?:vocalist|singer)\b",
)
_VOICE_CHECK_HEAD = "VOICE COUNT CHECK (read before the vocal blocks below):"


def voice_count_check(A):
    """(applies, why) -- dev53's VOICE COUNT CHECK: a two-co-leads reading that
    came from the pitch histogram alone while a listening pass said one singer
    or the two centres are an octave apart."""
    t = str(A or "")
    m = re.search(r"(?m)^(?:- )?" + re.escape(_VOICE_CHECK_HEAD) + r"\s*(.+)$", t)
    if m:
        return True, m.group(1).strip()
    if "[Upgraded to mixed leads on independent objective f0 evidence" not in t:
        return False, ""
    sm = re.search(r"voiced-f0 multi-peak hint:[^\n]*?~(\d+(?:\.\d+)?)\s+semitones between top two", t)
    sep = float(sm.group(1)) if sm else None
    parts = []
    i = t.find("VOCAL / SINGER PROFILE (isolated dedicated pass):")
    if i >= 0:
        e = min([x for x in (t.find("\nVOCAL OBJECTIVE MEASUREMENTS", i), t.find("\nVOCAL DECISION AUDIT", i))
                 if x >= 0] or [len(t)])
        parts.append(t[i:e])
    k = t.find("VOCAL CONFIRMATION PASS:")
    if k >= 0:
        ke = t.find("\n\n", k)
        parts.append(t[k:ke if ke >= 0 else len(t)])
    joined = strip_md("\n".join(parts))
    single = any(re.search(p, joined, re.I) for p in _SINGLE_VOICE_RES)
    octave = sep is not None and 11.0 <= sep <= 13.0
    if not (single or octave):
        return False, ""
    why = []
    if octave:
        why.append(f"its two pitch centres are about {sep:g} semitones — an octave — apart, the signature "
                   "of octave doubling or of pitch tracking jumping an octave on one voice")
    elif sep is not None:
        why.append(f"two pitch centres {sep:g} semitones apart are also what one singer moving between "
                   "chest and head voice produces")
    if single:
        why.append("the listening passes themselves heard a single lead")
    return True, ("the 'two co-lead singers' reading was not heard by ear; it was added from the pitch "
                  "measurements alone, and " + "; ".join(why) + ". Describe one lead singer (with any "
                  "harmonies or doubling) unless the singer identity says the lead is shared by a group, "
                  "duo or featured singer; don't describe co-leads an octave apart.")


def preserve_shared_vocal_evidence(info, meta):
    """Carry explicit guest credits and the saved multi-voice audit into the dossier.

    Older exports only promoted a featured singer when the identity block already
    said LEAD_SINGERS=one/single. That missed credits when the identity was
    uncertain or omitted the count. Likewise, the saved VOICE_ARRANGEMENT and
    CO_LEAD_DETAIL fields were displayed elsewhere but did not make "Who's
    singing" acknowledge co-leads. Trust explicit credits / vocal arrangement;
    do not promote pitch-histogram-only readings that voice_count_check rejects.
    """
    if not isinstance(info, dict) or info.get("vocals_present") == "no":
        return

    facts = info.get("identity_facts")
    if not isinstance(facts, dict):
        facts = {}
        info["identity_facts"] = facts

    # Merge named features from the identity reasoning with featured names in
    # artist/title tags. Do this regardless of the old LEAD_SINGERS value.
    features = list(facts.get("features") or [])
    tag_features = tag_featured_singers(meta)
    for candidate in tag_features:
        if candidate and not any(_fold_name(candidate) == _fold_name(old) for old in features):
            features.append(candidate)
    if features:
        facts["features"] = features
        info["lead_singers"] = "several"
        # A documented guest credit is stronger evidence than the pitch-only
        # warning, which explicitly permits a credited featured singer.
        info["voice_count_check"] = ""
        return

    # Never turn the known pitch-histogram false positive into a duet. The
    # analysis has a specific check for this; respect it before reading the
    # generic multi-voice audit fields.
    if info.get("voice_count_check"):
        return

    voices = info.get("voices") if isinstance(info.get("voices"), dict) else {}
    vocal_extra = info.get("vocal_extra") if isinstance(info.get("vocal_extra"), dict) else {}
    arrangement = str(voices.get("VOICE_ARRANGEMENT") or vocal_extra.get("VOICE_ARRANGEMENT") or "")
    arrangement = re.sub(r"^[\s\[\"'<>]+|[\s\]\"'<>]+$", "", arrangement).lower()
    arrangement = re.match(r"[a-z0-9_-]+", arrangement.replace("-", "_"))
    arrangement = arrangement.group(0).replace("-", "_") if arrangement else ""

    # These values are explicitly defined by Musiclyse as multi-lead forms.
    if arrangement in ("duet_co_leads", "call_response"):
        info["lead_singers"] = "several"
        return

    # The analysis contract says that a voice count of two only establishes
    # distinct co-leads when CO_LEAD_DETAIL is also clear (not just a second
    # pitch centre, harmony, doubling, or an empty placeholder).
    count_text = str(voices.get("NUM_DISTINCT_VOICES") or vocal_extra.get("NUM_DISTINCT_VOICES") or "")
    count_match = re.match(r"\s*(\d+|one|two|three|three\+)", count_text, re.I)
    count_word = count_match.group(1).lower() if count_match else ""
    count = {"one": 1, "two": 2, "three": 3, "three+": 3}.get(count_word)
    if count is None and count_word.isdigit():
        count = int(count_word)
    detail = str(voices.get("CO_LEAD_DETAIL") or vocal_extra.get("CO_LEAD_DETAIL") or "").strip()
    detail_norm = re.sub(r"[\[\]\s.]", "", detail).lower()
    if count is not None and count >= 2 and detail_norm not in (
            "", "none", "n/a", "na", "uncertain", "unknown", "notclear", "unparsed"):
        info["lead_singers"] = "several"


# =============================================================================
# Musiclyse 0.5-moss-dev17 (MOSS-Music listening model): checks and blocks
# =============================================================================
# dev17 writes several new blocks into a save (SECTION VOICE CHECK, DRUM STEM
# NOTE, ERA NOTE (drums), FULL-BANDWIDTH TEMPO CHECK, HARMONY AND MELODY BY
# SECTION, KEY CENTRE CHECK, a faint-vocal-texture arbitration) and applies a
# few more checks only when it shows a saved scan to its writer (GUITAR CHECK,
# GENRE ERA CHECK, the synth-denial ARRANGEMENT CHECK, the strong-reed-lead and
# synth-pad readings of the INSTRUMENT CHECK, the key-quality repair, the
# low-confidence pitch check, and the harmony/melody block for saves made
# before it existed). Everything below is read or recomputed from the saved
# text and notes with dev17's own rules, so older saves get the load-time
# checks too. Only the wording is written for the dossier.

_INSTR_CHECK_HEAD = "INSTRUMENT CHECK (read before the instrument tags below):"
_REED_TAGS = (("accordion", "an accordion"), ("harmonium", "a harmonium"), ("bandoneon", "a bandoneon"),
              ("concertina", "a concertina"), ("harmonica", "a harmonica"), ("melodica", "a melodica"))
_REED_PLAUSIBLE_GENRES = re.compile(
    r"\b(?:folk|country|blues|bluegrass|americana|cajun|zydeco|tango|polka|world|celtic|irish|"
    r"scottish|latin|cumbia|norte\w*|tex-?mex|vallenato|musette|chanson|klezmer|forr[oó]|"
    r"tejano|conjunto|sea shanty|shanty|gypsy|balkan|french|italian|ranchera|singer-songwriter|"
    r"roots|skiffle|harmonica)\b", re.IGNORECASE)
REED_STRONG_LEAD_MIN_PCT = 75
REED_STRONG_LEAD_MIN_WINDOW_SHARE = 0.75
REED_PAD_MIN_POLYPHONY = 4.0
_REED_NEW_READING_MARKS = ("hears a steady reedy lead", "most likely a synth pad or sustained keyboard chords")


def _listening_model_spans(text):
    """The listening model's own words in the notes -- its opening report and
    its era, vocal and confirmation passes -- not the tagger lines or
    measurements (dev17's _listening_model_spans)."""
    t = str(text or "")
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


def _reed_tag_check_line(text):
    """dev17's INSTRUMENT CHECK line for unconfirmed free-reed tags (with its
    strong-lead and synth-pad readings), or "". Used for saves that carry no
    INSTRUMENT CHECK, and to refresh a stored one the way dev17 does."""
    t = str(text or "")
    tag_lines = [ln for ln in t.splitlines()
                 if "instrument tag (" in ln or re.match(r"\s+[a-z][\w /()-]* \((?:\w+) stem\):", ln)]
    if not tag_lines:
        return ""
    tagged = "\n".join(tag_lines).lower()
    own = _listening_model_spans(t).lower()
    genre_txt = " ".join(m.group(1) for m in _RECOMMENDED_GENRE_RE.finditer(t))
    genre_txt += " " + " ".join(_ranked_labels(t, max_n=5))
    if _REED_PLAUSIBLE_GENRES.search(genre_txt):
        return ""
    found = [(w, a) for w, a in _REED_TAGS
             if re.search(r"\b" + w + r"\b", tagged) and not re.search(r"\b" + w, own)]
    if not found:
        return ""
    strong = []
    for w, a in found:
        r = _strong_reed_reading(t, w) if w == "harmonica" else None
        if r:
            strong.append((w, a, r))
    found = [(w, a) for w, a in found if w not in {x[0] for x in strong}]
    lead_s = ""
    if strong:
        w, _a, (pct, n_on, n_all, stem) = strong[0]
        lead_s = (f"the instrument tagger hears a steady reedy lead{' on the ' + stem + ' stem' if stem else ''}: "
                  f"'{w}' at {pct}% in {n_on} of {n_all} windows -- too strong and too sustained to dismiss.")
        if not found:
            return f"{_INSTR_CHECK_HEAD} {lead_s}"
    stems = sorted(set(re.findall(r"(?m)^\s+(?:" + "|".join(w for w, _a in found)
                                  + r")\b[^\n(]*\((\w+) stem\):", t)))
    names = " and ".join(f"'{w}'" for w, _a in found)
    describe = ""
    poly = _other_stem_polyphony(t) if stems == ["other"] else None
    if poly is not None and poly >= REED_PAD_MIN_POLYPHONY and not re.search(
            r"\bsax(?:ophone)?s?\b|\bhorns?\b|\bbrass\b|\btrumpets?\b|\btrombones?\b", own):
        describe = (f" (most likely a synth pad or sustained keyboard chords: the other stem it was heard on "
                    f"plays chords of about {poly:g} notes at once)")
    body = f"the instrument tagger's {names} label{'s' if len(found) > 1 else ''} is not confirmed{describe}."
    if lead_s:
        return f"{_INSTR_CHECK_HEAD} {lead_s} Separately, {body}"
    return f"{_INSTR_CHECK_HEAD} {body}"


def instrument_check_reading(A):
    """{'suppressed': set of false-positive labels, 'reed_lead': (word, pct,
    on, all, stem) or None, 'pad_poly': float or None} from the save's
    INSTRUMENT CHECK (dev60 and dev17 wording), refreshed or computed the way
    dev17 does when the save was made before its strong-lead / synth-pad
    readings or carries no check at all."""
    t = str(A or "")
    lines = [m.group(0) for m in re.finditer(r"(?m)^(?:- )?INSTRUMENT CHECK \([^\n]*$", t)]
    fresh = ""
    try:
        fresh = _reed_tag_check_line(t)
    except Exception:
        fresh = ""
    if not lines and fresh:
        lines = [fresh]
    elif lines and fresh and (any(k in fresh for k in _REED_NEW_READING_MARKS)
                              or any(k in lines[0] for k in _REED_NEW_READING_MARKS)):
        lines = [fresh] + lines[1:]
    out = {"suppressed": set(), "reed_lead": None, "pad_poly": None}
    for line in lines:
        body = line.split(":", 1)[1] if ":" in line else line
        # dev17: "... hears a steady reedy lead on the X stem: 'harmonica' at 87% in 40 of 48 windows ..."
        lead_part, rest = "", body
        if "hears a steady reedy lead" in body:
            k = body.find("Separately,")
            lead_part, rest = (body[:k], body[k:]) if k >= 0 else (body, "")
            m = re.search(r"steady reedy lead(?: on the (\w+) stem)?: '([^']+)' at (\d+)% in (\d+) of (\d+) windows",
                          lead_part)
            if m:
                out["reed_lead"] = (m.group(2).lower(), int(m.group(3)), int(m.group(4)), int(m.group(5)),
                                    (m.group(1) or "").lower())
        m = re.search(r"tagger's ((?:'[^']+'(?:,\s*|\s+and\s+)?)+)\s*labels?", rest)
        if m:
            for lab in re.findall(r"'([^']+)'", m.group(1)):
                out["suppressed"].add(lab.strip().lower())
        elif rest.strip():
            # older wording: any quoted label, else any known false-positive word
            for lab in re.findall(r"['\"]([A-Za-z][A-Za-z \-]{2,30})['\"]", rest):
                out["suppressed"].add(lab.strip().lower())
            if not out["suppressed"]:
                low = rest.lower()
                out["suppressed"].update(w for w in _INSTRUMENT_CHECK_FALSE_POS if w in low)
        mp = re.search(r"most likely a synth pad or sustained keyboard chords: the other stem it was heard on "
                       r"plays chords of about (\d+(?:\.\d+)?) notes", rest)
        if mp:
            out["pad_poly"] = float(mp.group(1))
    if out["reed_lead"]:
        out["suppressed"].discard(out["reed_lead"][0])
    return out


# ---- GUITAR CHECK (dev17 ACOUSTIC_GUITAR_CHECK) ------------------------------------
_GUITAR_CHECK_HEAD = "GUITAR CHECK (measured from the separated stems; read before the descriptions below):"
_ELECTRIC_GUITAR_RE = re.compile(r"(?i)\belectric[ -]?guitars?\b")
_ELECTRIC_NEGATED_RE = re.compile(r"(?i)\b(?:no|without|absence of|not)\s+(?:\w+\s+){0,2}$")


def _acoustic_only_guitar(text):
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


def guitar_check(A):
    """True when dev17's GUITAR CHECK applies (stored, or computed as dev17
    does at load time): the guitar stem reads acoustic in every section,
    nothing reads an electric guitar, yet the descriptions name one."""
    t = str(A or "")
    if _GUITAR_CHECK_HEAD in t:
        return True
    try:
        return _acoustic_only_guitar(t) and _asserts_electric_guitar(_listening_model_spans(t))
    except Exception:
        return False


# ---- ARRANGEMENT CHECK, synth-denial reading (dev17) ---------------------------------
_KEYS_DENIAL_RE = re.compile(
    r"(?i)\b(?:no|without)\s+(?:clear |obvious |additional |extra |audible |discernible |apparent |real )?"
    r"(?:synth(?:esizer)?s?|keyboards?|keys)\b"
    r"|\bsynth(?:esizer)?s?\s*\(\s*(?:likely\s+)?(?:absent|none|not present)"
    r"|\bsynth(?:esizer)?s?\s+(?:is|are)\s+(?:absent|not present|not used)")
_CONFIRMED_ACOUSTIC_PART_RE = (r"(?:strings|violin|viola|cello|fiddle|brass|trumpet|trombone|horns?|"
                               r"french horn|saxophone|sax|flute|clarinet|oboe|bassoon|woodwinds?|harmonica|"
                               r"accordion|choir)")


def arrangement_denial_note(A):
    """dev17's second ARRANGEMENT CHECK: the descriptions deny synths or
    keyboards while the 'other' stem carries a real chordal part that peaks
    somewhere. {'poly', 'notes', 'peaks', 'playing', 'rows', 'weak'} or {}."""
    t = str(A or "")
    m0 = re.search(r"(?m)^(?:- )?" + re.escape(_ARR_CHECK_HEAD) + r"\s*the descriptions say there are no synths "
                   r"or keyboards, but[^\n]*?about (\d+(?:\.\d+)?) notes at once, (\d+) notes\), loudest in (.+?) "
                   r"and playing in (\d+) of (\d+) sections", t)
    if m0:
        weak = re.search(r"weak labels for this stem \(([^)]*)\)", t[m0.start():m0.start() + 2000])
        return {"poly": float(m0.group(1)), "notes": int(m0.group(2)), "peaks": m0.group(3).strip(),
                "playing": int(m0.group(4)), "rows": int(m0.group(5)),
                "weak": [w.strip() for w in weak.group(1).split(",")] if weak else []}
    if not _KEYS_DENIAL_RE.search(_listening_model_spans(t)):
        return {}
    i = t.find("STEM ACTIVITY / PROMINENCE (ranked")
    if i < 0:
        return {}
    e = t.find("\n\n", i)
    m = re.search(r"(?m)^\d+\) other: ([a-z /]+?) \(RMS", t[i:e if e >= 0 else len(t)])
    if not m or m.group(1).strip().startswith("silent"):
        return {}
    j = t.find("[OTHER stem]")
    if j < 0:
        return {}
    k = t.find("\n\n", j)
    oblk = t[j:k if k >= 0 else len(t)]
    poly = re.search(r"average simultaneous notes: (\d+(?:\.\d+)?)", oblk)
    cnt = re.search(r"note count after filtering: (\d+)", oblk)
    if not poly or float(poly.group(1)) < 3.0 or not cnt or int(cnt.group(1)) < 300:
        return {}
    s_ = t.find("INTENSITY BY SECTION (other stem")
    if s_ < 0:
        return {}
    se = t.find("\n\n", s_)
    rows = re.findall(r"(?m)^\s+(.+?) \((\d+:\d\d)-(\d+:\d\d)\): (.+)$", t[s_:se if se >= 0 else len(t)])
    if not rows:
        return {}
    playing = [r for r in rows if not r[3].startswith(("not playing", "quiet"))]
    peaks = [r for r in rows if r[3].startswith("peak")]
    if not peaks or len(playing) * 3 < len(rows):
        return {}
    if re.search(r"(?m)^\s+" + _CONFIRMED_ACOUSTIC_PART_RE + r"(?: \([^)\n]*\))? \(other stem\):(?![^\n]*\[weak)",
                 t) or re.search(_CONFIRMED_ACOUSTIC_PART_RE + r"[^\[\n]*?\[supported by whole-mix", oblk):
        return {}
    weak = sorted(set(re.findall(r"(?m)^\s+(organ|strings|brass|harmonica|accordion|keyboard[\w ]*|"
                                 r"electric piano|synth[\w ]*)(?: \([^)\n]*\))? \(other stem\):[^\n]*\[weak", t)))
    return {"poly": float(poly.group(1)), "notes": int(cnt.group(1)),
            "peaks": ", ".join(f"{r[0]} ({r[1]}-{r[2]})" for r in peaks),
            "playing": len(playing), "rows": len(rows), "weak": weak}


# ---- GENRE ERA CHECK (dev17) ------------------------------------------------------
_GENRE_ERA_HEAD = "GENRE ERA CHECK (read before the description below):"
_GENRE_ONSETS = (
    (r"indie[ -]?rock", 1983, "indie rock"), (r"indie[ -]?pop", 1983, "indie pop"),
    (r"indie[ -]?folk", 1995, "indie folk"), (r"jangle[ -]?pop", 1982, "jangle pop"),
    (r"alternative(?:[ -]rock)?|alt[ -]rock", 1982, "alternative rock"), (r"college[ -]rock", 1982, "college rock"),
    (r"shoegaze", 1988, "shoegaze"), (r"dream[ -]?pop", 1983, "dream pop"), (r"grunge", 1987, "grunge"),
    (r"britpop", 1993, "britpop"), (r"madchester", 1988, "Madchester"), (r"emo", 1985, "emo"),
    (r"post[ -]?punk", 1977, "post-punk"), (r"new[ -]wave", 1976, "new wave"),
    (r"synth[ -]?pop", 1977, "synth-pop"), (r"punk(?:[ -]rock)?", 1974, "punk"),
    (r"power[ -]?pop", 1972, "power pop"), (r"trip[ -]?hop", 1993, "trip hop"),
    (r"drum (?:and|&|n) bass|dnb", 1992, "drum and bass"), (r"dubstep", 2002, "dubstep"),
    (r"house", 1984, "house"), (r"techno", 1985, "techno"), (r"trance", 1990, "trance"),
    (r"nu[ -]metal", 1996, "nu metal"), (r"grime", 2002, "grime"),
)


def _genre_onset(label):
    lab = str(label or "").lower()
    years = [y for pat, y, _n in _GENRE_ONSETS if re.search(r"(?<![a-z])(?:" + pat + r")(?![a-z])", lab)]
    return max(years) if years else None


def genre_era_check(A, year):
    """{'rec', 'onset', 'year', 'late': [...], 'fits': [...]} when the
    recommended genre is a label from after the file's tagged year (dev17's
    GENRE ERA CHECK), else {}."""
    try:
        m = re.search(r"\b(19[0-9]{2}|20[0-9]{2})\b", str(year or ""))
        y = int(m.group(1)) if m else None
    except Exception:
        y = None
    if not y:
        return {}
    t = str(A or "")
    recs = _RECOMMENDED_GENRE_RE.findall(t)
    if not recs:
        return {}
    rec = recs[-1].strip()
    onset = _genre_onset(rec)
    if onset is None or y >= onset:
        return {}
    ranked = _ranked_labels(t, max_n=8)
    late = [rec] + [l for l in ranked if l.lower() != rec.lower() and (_genre_onset(l) or 0) > y]
    fits = [l for l in ranked if l.lower() != rec.lower() and _genre_onset(l) is not None and _genre_onset(l) <= y]
    return {"rec": rec, "onset": onset, "year": y, "late": list(dict.fromkeys(late)), "fits": fits}


# ---- key fixes (dev17 KEY_MODE_FROM_ESSENTIA, KEY_CENTRE_CHECK) --------------------
_KEY_QUALITY_RE = re.compile(
    r"(RECOMMENDED KEY FOR DISCUSSION: )([A-G][#b♯♭]?) (aug(?:mented)?|dim(?:inished)?|sus\d*|5|7|\+)\.")


def key_quality_repair(A):
    """A recommended key that is a chord quality ("D aug") takes Essentia's
    mode when Essentia names the same root (dev17 repairs saved scans so)."""
    t = str(A or "")
    m = _KEY_QUALITY_RE.search(t)
    if not m:
        return t
    e = re.search(r"Essentia estimated key: ([A-G][#b♯♭]?) (major|minor)\b", t)
    norm = lambda k: k.replace("♯", "#").replace("♭", "b")
    if not e or norm(e.group(1)) != norm(m.group(2)):
        return t
    rest = t[m.end():]
    note = (" (The listening model gave '" + m.group(2) + " " + m.group(3)
            + "', a chord quality rather than a mode; Essentia's mode is used.)")
    if rest.lstrip().startswith("Reasoning:"):
        k = rest.find("Reasoning:") + len("Reasoning:")
        rest = rest[:k] + note + rest[k:]
    return t[:m.start()] + m.group(1) + f"{e.group(1)} {e.group(2)}." + rest


# ---- key note check (dev18 KEY_NOTE_CHECK, recomputed for older saves) ----------------
# The listening model's own key (MOSS-Music answers "Eb major" / "Eb minor" /
# "Ab minor" on most tracks) is scored against the saved transcribed notes
# beside Essentia's key; one the notes do not fit is set aside, as dev18 does
# at scan time. Same profiles, thresholds and wording as the analyser.
_KK_MAJOR = (6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88)
_KK_MINOR = (6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17)
_KN_STEMS = ("bass", "guitar", "piano", "other", "vocals")
_KN_MIN_NOTES, _KN_POOR, _KN_GOOD, _KN_MARGIN, _KN_NEAR_BEST = 200, 0.5, 0.6, 0.3, 0.1
_KN_REASON_RE = re.compile(
    r"(?m)^RECOMMENDED KEY FOR DISCUSSION: (?P<final>[^\n]+?)\. Reasoning: "
    r"(?P<lm>MOSS-Music|the listening model|MF) said (?P<x>[^,\n]+?), Essentia said (?P<y>[^(\n]+?) "
    r"\(strength=(?P<s>[\d.]+|n/a), (?P<side>above|below/no) confidence threshold\) — "
    r"[^\n]*?(?:mention the disagreement is possible if asked\.|hedge appropriately\.)")
_KN_PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


def _kn_parse(k):
    m = re.match(r"\s*([A-G])\s*(#|b|♯|♭|sharp|flat)?\s*(major|minor)\b", str(k or ""), re.I)
    if not m:
        return None
    pc = _KN_PC[m.group(1).upper()]
    acc = (m.group(2) or "").lower()
    pc += 1 if acc in ("#", "♯", "sharp") else -1 if acc in ("b", "♭", "flat") else 0
    return pc % 12, m.group(3).lower()


def _kn_fits(stem_midi):
    hist, n = [0.0] * 12, 0
    for st in _KN_STEMS:
        h, tot, k = [0.0] * 12, 0.0, 0
        for x in (stem_midi or {}).get(st) or []:
            try:
                mm, d = int(x.get("m")), float(x.get("d") or 0)
            except (TypeError, ValueError, AttributeError):
                continue
            if d > 0:
                h[mm % 12] += d
                tot += d
                k += 1
        if k and tot > 0:
            hist = [a + b / tot for a, b in zip(hist, h)]
            n += k
    if not n or sum(hist) <= 0:
        return {}, n
    def corr(a, b):
        ma, mb = sum(a) / 12.0, sum(b) / 12.0
        num = sum((p - ma) * (q - mb) for p, q in zip(a, b))
        da = math.sqrt(sum((p - ma) ** 2 for p in a))
        db = math.sqrt(sum((q - mb) ** 2 for q in b))
        return num / (da * db) if da > 0 and db > 0 else 0.0
    fits = {}
    for r in range(12):
        for q, prof in (("major", _KK_MAJOR), ("minor", _KK_MINOR)):
            fits[(r, q)] = corr(hist, [prof[(i - r) % 12] for i in range(12)])
    return fits, n


def key_note_repair(A, stem_midi):
    """(A, switched): the RECOMMENDED KEY block rewritten as dev18's KEY NOTE
    CHECK writes it, when the listening model's key and Essentia's disagree and
    the saved notes fit Essentia's key and not the listening model's. Saves
    already checked, without notes, or where the notes do not settle it come
    back unchanged."""
    t = str(A or "")
    try:
        m = _KN_REASON_RE.search(t)
        if not m or not stem_midi:
            return t, False
        x, y, final = m.group("x").strip(), m.group("y").strip(), m.group("final").strip()
        px, py = _kn_parse(x), _kn_parse(y)
        if not px or not py or px == py:
            return t, False
        side = m.group("side")
        if (side == "above" and _kn_parse(final) != py) or (side != "above" and _kn_parse(final) != px):
            return t, False
        fits, n = _kn_fits(stem_midi)
        if not fits or n < _KN_MIN_NOTES:
            return t, False
        fx, fy, best = fits[px], fits[py], max(fits.values())
        if not (fx < _KN_POOR and fy >= _KN_GOOD and fy - fx >= _KN_MARGIN and fy >= best - _KN_NEAR_BEST):
            return t, False
        s, lm = m.group("s"), m.group("lm")
        if lm == "MF":
            lm = "the listening model"
        lm_cap = lm[:1].upper() + lm[1:]
        tail = ("KEY NOTE CHECK: the transcribed notes of every pitched part fit "
                f"{y} (fit {fy:.2f} of 1) and do not fit {x} (fit {fx:.2f}).")
        if side == "above":
            line = (f"RECOMMENDED KEY FOR DISCUSSION: {final}. Reasoning: Essentia measured {y} "
                    f"(strength={s}) and the transcribed notes agree. {lm_cap}'s own answer, {x}, is set "
                    f"aside: {tail} Treat {y} as a clear reading; do not offer {x} as an alternative.")
            return t[:m.start()] + line + t[m.end():], False
        line = (f"RECOMMENDED KEY FOR DISCUSSION: {y}. Reasoning: {lm} said {x} and Essentia said {y} "
                f"(strength={s}, under its confidence threshold), so {x} was first preferred. {tail} "
                f"So {y} is used, and {x} is set aside. Treat {y} as moderately confident. Chord-root "
                f"degrees in the stem report above count from {x}; the melody blocks count from {y}.")
        return t[:m.start()] + line + t[m.end():], True
    except Exception:
        return t, False


# ---- key fit check (analyser KEY_FIT_CHECK, recomputed for older saves) ----------
_KF_LINE_RE = re.compile(r"(?m)^RECOMMENDED KEY FOR DISCUSSION: (?P<final>[^\n]+?)\. Reasoning: [^\n]*")
_KF_STEPS = {"major": (0, 2, 4, 5, 7, 9, 11), "minor": (0, 2, 3, 5, 7, 8, 10)}
_KF_SHARPS = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
_KF_FLATS = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]


def key_fit_repair(A, stem_midi):
    """The RECOMMENDED KEY line rewritten as the analyser's KEY FIT CHECK
    writes it: the saved notes fit a key one scale note away from the reported
    one at least KEY_FIT_MARGIN better, and the bass sits on its tonic most.
    Otherwise A comes back unchanged."""
    t = str(A or "")
    try:
        m = _KF_LINE_RE.search(t)
        if not m or not stem_midi:
            return t, False
        final = m.group("final").strip()
        if re.search(r"KEY (?:FIT|CENTRE) CHECK|Mixolydian|Dorian|Lydian|Phrygian", m.group(0)):
            return t, False
        po = _kn_parse(final)
        if not po:
            return t, False
        fits, n = _kn_fits(stem_midi)
        if not fits or n < _KN_MIN_NOTES:
            return t, False
        best = max(fits, key=lambda k: fits[k])
        fo, fb = fits[po], fits[best]
        if best == po or fb < _KN_GOOD or fb - fo < float(KEY_FIT_MARGIN):
            return t, False
        so = {(po[0] + i) % 12 for i in _KF_STEPS[po[1]]}
        sb = {(best[0] + i) % 12 for i in _KF_STEPS[best[1]]}
        if len(so - sb) > 1:
            return t, False
        h, k = [0.0] * 12, 0
        for x in (stem_midi or {}).get("bass") or []:
            try:
                mm, d = int(x.get("m")), float(x.get("d") or 0)
            except (TypeError, ValueError, AttributeError):
                continue
            if d > 0:
                h[mm % 12] += d
                k += 1
        tot = sum(h)
        if k < 40 or tot <= 0 or max(range(12), key=lambda i: h[i]) != best[0]:
            return t, False
        names = _KF_FLATS if re.match(r"\s*[A-G](?:b|\u266d)", final) else _KF_SHARPS
        new = f"{names[best[0]]} {best[1]}"
        sm = re.search(r"strength=([\d.]+)", m.group(0))
        det = f"named {final}" + (f" (strength={sm.group(1)})" if sm else "")
        line = (f"RECOMMENDED KEY FOR DISCUSSION: {new}. Reasoning: KEY FIT CHECK: the key detector {det}, "
                f"but it matches the notes against major and minor templates only, and {final} shares all "
                f"but one of its notes with {new}. The transcribed notes of every pitched part fit {new} better "
                f"(fit {fb:.2f} of 1, against {fo:.2f} for {final}), and the bass sits on {names[best[0]]} "
                f"more than on any other note ({h[best[0]] / tot * 100:.0f}% of its note time). So the song is in "
                f"{new}; treat it as moderately confident and do not present {final} as the key. Chord-root "
                f"degrees in the stem report above, and chords marked outside the measured key, count from "
                f"{final}. This is the primary key to report to the user unless the reasoning above marks it "
            f"uncertain.")
        return t[:m.start()] + line + t[m.end():], True
    except Exception:
        return t, False


# ---- tempo checks (dev19 TEMPO_LISTENING_CHECK / TEMPO_SNARE_TIEBREAK, for older saves) --
# Same tests and wording as the analyser: a MOSS-Music TEMPO_BPM that only
# repeated the prompt is not used; a MOSS-Music pulse matching neither detector
# gives way to a detector that matches its own beat spacing; and two readings
# ~2x apart settled only by "prefer the lower pulse" go to the higher one when
# the saved snare hits land every two of its beats.
_TP_ECHO_RE = re.compile(
    r"(?im)^\W*TEMPO[_ ]?BPM\**\s*[=:]\s*\[?\s*(?:~|≈)?\s*(\d+(?:\.\d+)?)\s*\(?\s*"
    r"best estimate of the musical pulse")
_TP_BLOCK_RE = re.compile(
    r"RECOMMENDED TEMPO FOR DISCUSSION: (?P<bpm>\d+) BPM\. Reasoning: (?P<why>[^\n]*?) This is the primary "
    r"tempo to report to the user\. State it as a concrete figure \(e\.g\. 'about \d+ BPM'\)\. Do not expand "
    r"it into a range unless this block itself marks the value as uncertain\.")
_TP_LOWER_RE = re.compile(
    r"^(?P<hs>.+?) \((?P<hi>[\d.]+)\) is ~2x (?P<ls>.+?) \((?P<lo>[\d.]+)\); preferring lower pulse to avoid "
    r"double-time misread\.")
_TP_PREFS = {
    "electronicish": (118.0, 132.0, True, True, "house/techno/dance pulse typically 120–130"),
    "dnb": (160.0, 178.0, False, False, "drum & bass / jungle typically 160–175; keep fast readings"),
    "hiphop": (70.0, 100.0, True, True, "hip-hop / trap often feels 70–95 (or double-time 140–160)"),
    "rockish": (100.0, 145.0, True, True, "rock / indie / punk typical mid-tempo pulse"),
    "popish": (95.0, 130.0, True, True, "pop / indie-pop typical mid-tempo pulse"),
    "ballad": (55.0, 95.0, True, True, "ballad / slow song — prefer the lower pulse"),
    "other": (80.0, 140.0, True, True, "general mid-tempo preference"),
}
_TP_COMMON = (60, 64, 66, 70, 72, 74, 76, 80, 84, 88, 90, 92, 96, 98, 100, 104, 105, 108, 110, 112, 115,
              116, 118, 120, 122, 124, 125, 126, 128, 130, 132, 135, 136, 138, 140, 144, 145, 148, 150,
              152, 155, 160, 165, 168, 170, 174, 175, 180, 190, 200)


def _tp_q(val):
    v = float(val)
    best = int(round(v))
    bd = abs(v - best)
    for c in _TP_COMMON:
        d = abs(v - c)
        if d <= 1.5 and d < bd:
            best, bd = c, d
    return int(best)


def _tp_family(A, why):
    m = re.search(r"Genre prior \(the listening model's first impression[^)]*\): (\w+) \(", A)
    if m and m.group(1) in _TP_PREFS:
        return m.group(1)
    m = re.search(r"Genre prior: (\w+) \(", A)
    if m and m.group(1) in _TP_PREFS:
        return m.group(1)
    for fam, pr in _TP_PREFS.items():
        if pr[4] in why:
            return fam
    return "other"


def _tp_detectors_only(ess, obj, fam):
    """The analyser's reconcile_bpm with no listening-model reading and two
    detector readings (both given): (bpm, note) or (None, None)."""
    if ess is None or obj is None:
        return None, None
    ratio = max(ess, obj) / max(min(ess, obj), 1e-6)
    if ratio <= 1.04:
        avg = (ess + obj) / 2.0
        return _tp_q(avg), f"Essentia and objective detector agree (~{avg:.1f}, averaged)."
    if not (1.8 <= ratio <= 2.2):
        return None, None
    a, b, la, lb = ess, obj, "Essentia", "objective"
    lo, hi = (a, b) if a <= b else (b, a)
    lo_src, hi_src = (la, lb) if a <= b else (lb, la)
    pref_lo, pref_hi, prefer_half, _pd, fam_note = _TP_PREFS.get(fam, _TP_PREFS["other"])
    if fam == "dnb" and pref_lo <= hi <= pref_hi + 5:
        return _tp_q(hi), f"{hi_src} ({hi}) is ~2x {lo_src} ({lo}); genre prior ({fam_note}) favours the fast pulse."
    if fam == "hiphop" and hi >= 130.0 and 60.0 <= lo <= 105.0:
        return _tp_q(lo), f"{hi_src} ({hi}) is ~2x {lo_src} ({lo}); genre prior ({fam_note}) prefers the lower pulse."
    if fam == "electronicish":
        if pref_lo <= lo <= pref_hi:
            return _tp_q(lo), f"{hi_src} ({hi}) is ~2x {lo_src} ({lo}); genre prior ({fam_note}) prefers the mid pulse."
        if pref_lo <= hi <= pref_hi:
            return _tp_q(hi), f"{hi_src} ({hi}) is ~2x {lo_src} ({lo}); genre prior ({fam_note}) prefers the mid pulse."
    if fam in ("rockish", "popish"):
        hi_in, lo_in = pref_lo <= hi <= pref_hi, pref_lo <= lo <= pref_hi
        if hi_in and not lo_in:
            return _tp_q(hi), (f"{hi_src} ({hi}) is ~2x {lo_src} ({lo}); genre prior ({fam_note}) favours {hi}, "
                               "which is inside that range.")
        if lo_in and not hi_in:
            return _tp_q(lo), (f"{hi_src} ({hi}) is ~2x {lo_src} ({lo}); genre prior ({fam_note}) favours {lo}, "
                               "which is inside that range.")
    if hi >= 140.0 and 55.0 <= lo <= 110.0 and prefer_half:
        return _tp_q(lo), f"{hi_src} ({hi}) is ~2x {lo_src} ({lo}); preferring lower pulse to avoid double-time misread."
    for val, src in ((a, la), (b, lb)):
        if 85.0 <= val <= 140.0:
            return _tp_q(val), f"Preferring mid-range pulse from {src} ({val})."
    return _tp_q(lo), f"2x pair {lo}/{hi}; defaulting to lower ({lo})."


def _tp_snare(stem_midi):
    t = []
    for n in ((stem_midi or {}).get("drums") or []) if isinstance(stem_midi, dict) else []:
        try:
            # Dev3 retains simultaneous fallback hits for drum descriptions,
            # but excludes sl=False hits from legacy structure/tempo evidence.
            # Missing flags (older scans and Omnizart) keep their old behaviour.
            if n.get("k") == "snare" and n.get("sl") is not False:
                t.append(float(n["t"]))
        except (TypeError, ValueError, KeyError, AttributeError):
            continue
    t.sort()
    iois = [b - a for a, b in zip(t, t[1:]) if 0.25 < b - a < 2.5]
    if len(iois) < 40:
        return None
    bins = {}
    for i in iois:
        k = int(round(i / 0.02))
        bins[k] = bins.get(k, 0) + 1
    best = max(bins, key=lambda k: (bins[k] + bins.get(k - 1, 0) + bins.get(k + 1, 0), -k))
    share = (bins[best] + bins.get(best - 1, 0) + bins.get(best + 1, 0)) / float(len(iois))
    if share < 0.5:
        return None
    sel = [i for i in iois if abs(i / 0.02 - best) <= 1.5]
    ioi = sum(sel) / len(sel)
    return 120.0 / ioi, share, ioi


def _tp_line(bpm, why):
    return (f"RECOMMENDED TEMPO FOR DISCUSSION: {bpm} BPM. Reasoning: {why} This is the primary tempo to "
            f"report to the user. State it as a concrete figure (e.g. 'about {bpm} BPM'). Do not expand it "
            "into a range unless this block itself marks the value as uncertain.")


_TP_MOSS_DOUBLE_RE = re.compile(r"^(?P<hs>.+?) \((?P<hi>[\d.]+)\) is ~2x MOSS-Music \((?P<lo>[\d.]+)\);")


def tempo_repair(A, stem_midi):
    """(A, new_bpm or None): the dev19 tempo checks on a save made before them."""
    t = str(A or "")
    try:
        bm = None
        for m_ in _TP_BLOCK_RE.finditer(t):
            bm = m_
        if not bm or "TEMPO CHECK" in bm.group("why"):
            return t, None
        old, why = int(bm.group("bpm")), bm.group("why")
        num = lambda rx: (lambda m: float(m.group(1)) if m else None)(re.search(rx, t))
        ess = num(r"Essentia estimated tempo \(raw detector\):\s*([\d.]+)\s*BPM")
        obj = num(r"(?<!Essentia )estimated tempo \(raw detector\):\s*([\d.]+)\s*BPM")
        ess_med = num(r"Essentia median inter-beat interval:.*?\(\s*([\d.]+)\s*BPM\s*\)")
        obj_med = num(r"(?<!Essentia )median inter-beat interval:.*?\(\s*([\d.]+)\s*BPM\s*\)")
        nb, nn = old, why
        mm = re.search(r"MOSS-Music(?: musical pulse)? \(~?([\d.]+)\)", why)
        em = _TP_ECHO_RE.search(t)
        if mm and em and abs(float(em.group(1)) - float(mm.group(1))) < 0.05:
            rb, rn = _tp_detectors_only(ess, obj, _tp_family(t, why))
            if rb:
                gm = re.search(r"\s*(Genre prior \(the listening model's first impression[^\n]*?\): \w+ \([^)]*\)\.)", why)
                nb = rb
                nn = (rn + (" " + gm.group(1) if gm and "genre prior (" not in rn.lower() else "")
                      + " TEMPO CHECK: MOSS-Music's TEMPO_BPM repeated the prompt's instruction text instead of "
                      "giving a reading, so it was not used.")
        elif why.startswith("Using MOSS-Music musical pulse") and mm:
            m = float(mm.group(1))
            for src, raw, med in (("Essentia", ess, ess_med), ("the objective detector", obj, obj_med)):
                if raw is None or med is None or raw <= 0 or abs(raw - med) / raw > 0.03:
                    continue
                if any(abs(m - r) / r <= 0.10 for r in (raw, raw * 2.0, raw / 2.0)):
                    continue
                q = _tp_q(raw)
                gm = re.search(r"\s*(Genre prior[^\n]*)$", why)
                nb, nn = q, (f"TEMPO CHECK: MOSS-Music's pulse ({m:g}) matches neither detector, and {src}'s reading "
                             f"({raw:g}) matches its own beat spacing ({med:.1f} BPM), so {q} is used. Treat it as "
                             "a measured reading; the detectors did not all give the same figure."
                             + (" " + gm.group(1).strip() if gm else ""))
                break
        if TEMPO_COPY_CHECK and nb == old and "MOSS-Music" in why:
            # Musiclyse dev46 TEMPO_COPY_CHECK: MOSS-Music's TEMPO_BPM was a copy, to
            # the decimal, of a detector reading it had been shown
            _tm = re.search(r"TEMPO_BPM\s*=\s*\[?\s*(?:~|approx\.?|about|around)?\s*([0-9]+(?:\.[0-9]+)?)", t)
            _mv = float(_tm.group(1)) if _tm else None
            _cv = next((v for v in (ess, obj, ess_med, obj_med) if _mv is not None and v is not None
                        and abs(_mv - v) <= 0.06 and abs(v - round(v)) >= 0.05), None)
            if _cv is not None:
                rb, rn = _tp_detectors_only(ess, obj, _tp_family(t, why))
                if rb and abs(rb - old) / float(old) > 0.10:
                    nb, nn = rb, (rn.rstrip().rstrip(".") + f". TEMPO CHECK: MOSS-Music's TEMPO_BPM ({_mv:g}) only "
                                  f"repeated a detector reading it had been shown ({_cv:g}), so it was not counted "
                                  "as a separate opinion.")
        out = t[:bm.start()] + _tp_line(nb, nn) + t[bm.end():] if nb != old else t
        bm2 = None
        for m_ in _TP_BLOCK_RE.finditer(out):
            bm2 = m_
        if bm2 and int(bm2.group("bpm")) == nb:
            dm = _TP_LOWER_RE.match(bm2.group("why"))
            sb = _tp_snare(stem_midi) if dm else None
            if dm and sb:
                hi, lo = float(dm.group("hi")), float(dm.group("lo"))
                sbpm, share, ioi = sb
                if hi >= lo and abs(sbpm - hi) / hi <= 0.04 and abs(sbpm - lo) / lo > 0.04:
                    q = _tp_q(hi)
                    w2 = (bm2.group("why").rstrip() + f" TEMPO CHECK: the snare on the separated drums lands every "
                          f"{ioi:.2f} s ({share * 100:.0f}% of its gaps), a backbeat on 2 and 4 at {sbpm:.0f} BPM, "
                          f"so the faster reading, {hi:g}, is the pulse: {q} BPM.")
                    out = out[:bm2.start()] + _tp_line(q, w2) + out[bm2.end():]
                    nb = q
        if nb == old and TEMPO_OCTAVE_PAIR_CHECK:
            pm = _TP_MOSS_DOUBLE_RE.match(why)
            sb = _tp_snare(stem_midi) if (pm and ess) else None
            if pm and sb:
                hi, moss = float(pm.group("hi")), float(pm.group("lo"))
                lo_d, hi_d = min(ess, hi), max(ess, hi)
                if (old == _tp_q(moss) and lo_d > 0 and abs(hi_d / lo_d - 2.0) <= 0.06
                        and not any(abs(moss - r) / r <= TEMPO_OCTAVE_PAIR_TOL
                                    for r in (lo_d, hi_d, lo_d / 2.0, hi_d * 2.0))):
                    cand = min((lo_d, hi_d), key=lambda r: abs(math.log(moss / r)))
                    sbpm, share, ioi = sb
                    if abs(sbpm - cand) / cand <= 0.04:
                        q = _tp_q(cand)
                        w2 = (why.rstrip() + f" TEMPO CHECK: MOSS-Music's pulse ({moss:g}) is not either detector's "
                              f"reading or its half or double, while the detectors agree with each other an octave "
                              f"apart ({lo_d:g} and {hi_d:g}); the snare on the separated drums lands every {ioi:.2f} s "
                              f"({share * 100:.0f}% of its gaps), a backbeat on 2 and 4 at {sbpm:.0f} BPM, so {cand:g} "
                              f"is the pulse: {q} BPM.")
                        out = t[:bm.start()] + _tp_line(q, w2) + t[bm.end():]
                        nb = q
        if nb == old:
            return t, None
        out = out.replace(
            " This is the primary tempo to report to the user. State it as a concrete figure (e.g. 'about "
            f"{nb} BPM').",
            f" Figures elsewhere in these notes that count beats or bars were worked out at {old} BPM, the "
            "tempo this file was first given. This is the primary tempo to report to the user. State it as a "
            f"concrete figure (e.g. 'about {nb} BPM').", 1)
        return out, nb
    except Exception:
        return t, None


def key_centre_reading(A):
    """dev17's KEY CENTRE CHECK in the RECOMMENDED KEY block: {'key', 'centre',
    'parent', 'flat7', 'lead'} or {}."""
    m = re.search(r"RECOMMENDED KEY FOR DISCUSSION: ([A-G][#b♯♭]?) Mixolydian \(the notes of ([A-G][#b♯♭]? major), "
                  r"centred on [^)]*\)\. Reasoning: KEY CENTRE CHECK:([^\n]*)", str(A or ""))
    if not m:
        return {}
    mm = re.search(r"uses (\S+) rather than (\S+) \(the", m.group(3))
    return {"key": f"{m.group(1)} Mixolydian", "centre": m.group(1), "parent": m.group(2),
            "flat7": mm.group(1) if mm else "", "lead": mm.group(2) if mm else ""}


# ---- pitch check: a LOW-confidence tracker reading (dev17) ---------------------------
VOCAL_PITCH_LOW_CONFIDENCE_MIN_ST = 3.0
_LOWCONF_CHECK_TAG = "PITCH CHECK (few voiced frames):"
_SHARP_NAMES = ["C", "C♯", "D", "D♯", "E", "F", "F♯", "G", "G♯", "A", "A♯", "B"]


def hz_note(freq):
    """librosa-style note name for a frequency ('A♯3')."""
    try:
        f = float(freq)
        if f <= 0:
            return "?"
        m = int(round(69 + 12 * math.log2(f / 440.0)))
        return f"{_SHARP_NAMES[m % 12]}{m // 12 - 1}"
    except (TypeError, ValueError):
        return "?"


def _voice_type_words(med):
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
    return male, female


def pitch_low_confidence_fix(A):
    """dev17's _pitch_low_confidence_check for saves made before it: when the
    pitch tracker's LOW-confidence median sits 3+ semitones from the vocal
    stem's note reading, the VOCAL PITCH REPORT figures and its voice-type
    map are taken from the note reading. Idempotent."""
    t = str(A or "")
    if _LOWCONF_CHECK_TAG in t or "pyin pitch reading confidence: LOW" not in t:
        return t
    if "OCTAVE CHECK:" in t or "PITCH CHECK:" in t:      # dev71 / dev73 checks own those cases
        return t
    p = re.search(r"objective voiced pitch median:\s*([\d.]+)\s*Hz", t)
    nm = re.search(r"VOCAL NOTE PITCH \(second reading[^)]*\):\s*median\s*([\d.]+)\s*Hz\s*\([^)]*\);\s*"
                   r"10-90%:\s*([\d.]+)-([\d.]+)\s*Hz", t)
    if not (p and nm):
        return t
    try:
        pyin, med, nlo, nhi = float(p.group(1)), float(nm.group(1)), float(nm.group(2)), float(nm.group(3))
        if min(pyin, med, nlo, nhi) <= 0 or abs(12.0 * math.log2(pyin / med)) < VOCAL_PITCH_LOW_CONFIDENCE_MIN_ST - (
                PITCH_CHECK_ROUNDING_TOL_ST if PITCH_CHECK_ROUNDED_FIGURES else 0.0):
            return t
        i = t.find("VOCAL PITCH REPORT (independent")
        if i < 0:
            return t
        j = t.find("\n\n", i)
        j = len(t) if j < 0 else j
        lines = t[i:j].split("\n")
        male, female = _voice_type_words(med)
        out = [lines[0],
               f"- objective voiced pitch median: {med:.1f} Hz",
               f"- approximate median note: {hz_note(med)}",
               f"- practical sung range (10th-90th percentile): {nlo:.1f}-{nhi:.1f} Hz",
               f"- {_LOWCONF_CHECK_TAG} the pitch tracker read a median of {pyin:g} Hz ({hz_note(pyin)}) from "
               "too few confidently voiced frames to trust; these figures are the note reading's.",
               "VOICE-TYPE MAP (from measured pitch only; approximate tessitura of the sung line):",
               f"- if the singer is male (any age): {male}",
               f"- if the singer is female: {female}"]
        k = t.find("VOICE-TYPE MAP", j)
        k_end = t.find("\n\n", k) if k >= 0 else -1
        if k >= 0 and k - j <= 4:
            j = len(t) if k_end < 0 else k_end
        return t[:i] + "\n".join(out) + t[j:]
    except (ValueError, ZeroDivisionError):
        return t


# ---- drums: no kit on the recording (dev17 DRUM_BLEED_NOTE) ------------------------
def drum_stem_note(A):
    """dev17's DRUM STEM NOTE: {'db': dB below the combined stems} when the
    separated drums are bleed and there is no drum kit, else {}."""
    m = re.search(r"(?m)^DRUM STEM NOTE:\s*the separated drums stem sits (\d+(?:\.\d+)?) dB below the combined "
                  r"stems[^\n]*?no drum kit", str(A or ""))
    if not m:
        m2 = re.search(r"(?m)^DRUM STEM NOTE:[^\n]*no drum kit", str(A or ""))
        return {"db": None} if m2 else {}
    return {"db": float(m.group(1))}


_DRUM_WORD_RE = re.compile(r"(?i)\b(?:drums?|drum\s*kit|drum[- ]machines?|kick|snare|hi-?hats?|cymbals?|toms|"
                           r"backbeat|four-on-the-floor|percussion)\b")   # dev17's _DRUM_CLAIM_RE, widened


def drop_drum_claims(text):
    """`text` without the clauses or sentences that describe drums."""
    t = str(text or "").strip()
    if not t:
        return t
    sep = "; " if ";" in t else " "
    parts = re.split(r"\s*;\s*", t) if ";" in t else re.split(r"(?<=[.!?])\s+", t)
    keep = [p for p in parts if p.strip() and not _DRUM_WORD_RE.search(p)]
    return sep.join(keep).strip(" ;")


# ---- FULL-BANDWIDTH TEMPO CHECK (dev17) ------------------------------------------------
def full_band_tempo(A):
    """{'slow': [35, 70], 'bpm': 140, 'hf': 14, 'factor': '2 or 4'} or {}."""
    m = re.search(r"(?m)^FULL-BANDWIDTH TEMPO CHECK:([^\n]*)", str(A or ""))
    if not m:
        return {}
    s = m.group(1)
    slow = re.search(r"read the tempo as ([^,]+?) BPM", s)
    bpm = re.search(r"puts? the beat at about (\d+(?:\.\d+)?) BPM", s)
    hf = re.search(r"(\d+(?:\.\d+)?)% of this track's energy", s)
    fac = re.search(r"counted every ([^.]+?) beats", s)
    return {"slow": slow.group(1).strip() if slow else "", "bpm": bpm.group(1) if bpm else "",
            "hf": hf.group(1) if hf else "", "factor": fac.group(1).strip() if fac else ""}


# ---- SECTION VOICE CHECK (dev17) ----------------------------------------------------
_SECTION_VOICE_HEAD = "SECTION VOICE CHECK"


def _spans(s):
    out = []
    for a, b in re.findall(r"(\d+:\d\d)-(\d+:\d\d)", str(s or "")):
        out.append((_secs(a), _secs(b)))
    return [x for x in out if x[0] is not None and x[1] is not None]


def section_voice_check(A):
    """{'main': gender, 'other': gender, 'spans': [(a, b)], 'all_spans': [...],
    'control': (a, b), 'direction': 'lower'|'higher'} from the SECTION VOICE
    CHECK block (a second lead of the other gender in some stretches), or {}."""
    t = str(A or "")
    i = t.find(_SECTION_VOICE_HEAD + " (measured")
    if i < 0:
        return {}
    e = t.find("\n\n", i)
    blk = t[i:e if e >= 0 else len(t)]
    m1 = re.search(r"The lead is (male|female) for most of the track, but the voice sits ([\d.]+)\+ semitones "
                   r"(lower|higher) in ([^\n]+?)\.\s*$", blk, re.M)
    m2 = re.search(r"heard a (male|female) singer in ([^\n]*?), and a (male|female) singer in the main lead "
                   r"\((\d+:\d\d)-(\d+:\d\d)\)", blk)
    if not m2:
        return {}
    out = {"other": m2.group(1), "spans": _spans(m2.group(2)), "main": m2.group(3),
           "control": (_secs(m2.group(4)), _secs(m2.group(5)))}
    out["all_spans"] = _spans(m1.group(4)) if m1 else list(out["spans"])
    out["direction"] = m1.group(3) if m1 else ""
    out["shift"] = m1.group(2) if m1 else ""
    return out


def _span_words(spans):
    return join_words([f"{mmss(a)}–{mmss(b)}" for a, b in spans])


# ---- faint vocal-like texture (dev17 FAINT_VOCAL_TEXTURE_CHECK) ---------------------
def faint_vocal_texture(A):
    """True when the instrumental verdict says faint vocal-like sounds (a
    sample, a chopped or processed voice) may be audible as texture."""
    m = re.search(r"(?m)^VOCAL PRESENCE ARBITRATION:\s*INSTRUMENTAL\.[^\n]*Faint vocal-like sounds", str(A or ""))
    return bool(m)


# ---- HARMONY AND MELODY BY SECTION (dev17) -----------------------------------------
_HM_HEAD = "HARMONY AND MELODY BY SECTION"
_HM_ACC_STEMS = ("guitar", "piano", "other")
_CHD_PC = {"C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3, "E": 4, "F": 5, "F#": 6, "Gb": 6, "G": 7,
           "G#": 8, "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11, "Cb": 11, "Fb": 4, "E#": 5, "B#": 0}
_CHD_SHARP = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
_CHD_FLAT = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]
_CHD_MODES = {"major": (0, 2, 4, 5, 7, 9, 11), "ionian": (0, 2, 4, 5, 7, 9, 11), "minor": (0, 2, 3, 5, 7, 8, 10),
              "aeolian": (0, 2, 3, 5, 7, 8, 10), "dorian": (0, 2, 3, 5, 7, 9, 10), "phrygian": (0, 1, 3, 5, 7, 8, 10),
              "lydian": (0, 2, 4, 6, 7, 9, 11), "mixolydian": (0, 2, 4, 5, 7, 9, 10), "locrian": (0, 1, 3, 5, 6, 8, 10)}
_CHD_REL = {"major": 0, "ionian": 0, "minor": 9, "aeolian": 9, "dorian": 2, "phrygian": 4, "lydian": 5,
            "mixolydian": 7, "locrian": 11}
_CHD_TOKEN_RE = re.compile(r"^([A-G])(#|b|♯|♭)?((?:maj|min|m|dim|aug|sus|add|M)?[0-9]*(?:b5|#5|b9|#9|#11|b13)?"
                           r"(?:sus[24])?(?:add[0-9]+)?)(?:/([A-G](?:#|b|♯|♭)?))?$")


def _chd_key(text):
    m = re.search(r"RECOMMENDED KEY FOR DISCUSSION: ([A-G](?:#|b|♯|♭)?)\s+(major|minor|ionian|dorian|phrygian|"
                  r"lydian|mixolydian|aeolian|locrian)\b", str(text or ""), re.I)
    if not m:
        return None
    r = m.group(1).replace("♯", "#").replace("♭", "b")
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


def _chd_fits(chord, key):
    if not key:
        return True
    root, mode = key[0], key[1]
    scale = [(root + i) % 12 for i in _CHD_MODES[mode]]
    m = _CHD_TOKEN_RE.match(chord)
    if not m:
        return True
    pc = _CHD_PC.get(m.group(1) + (m.group(2) or "").replace("♯", "#").replace("♭", "b"))
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


def _percentile(vals, q):
    """numpy.percentile with linear interpolation."""
    v = sorted(vals)
    if not v:
        return 0.0
    k = (len(v) - 1) * q / 100.0
    f = math.floor(k)
    c = min(f + 1, len(v) - 1)
    return v[f] + (v[c] - v[f]) * (k - f)


def _median(vals):
    return _percentile(vals, 50)


def _hm_pitched(rows):
    """(onset, duration, midi) for the pitched notes of one saved stem."""
    out = []
    for r in rows or []:
        if not isinstance(r, dict) or r.get("k"):
            continue
        try:
            on = float(r.get("t", 0.0))
            d = float(r.get("d", 0.0) or 0.0)
            if "m" in r:
                m = float(r["m"])
            elif "hz" in r and float(r["hz"]) > 0:
                m = 69.0 + 12.0 * math.log2(float(r["hz"]) / 440.0)
            else:
                continue
        except (TypeError, ValueError):
            continue
        out.append((on, max(0.05, d), m))
    return out


def _chord_label_from_pcs(pcs):
    pcs = sorted(set(int(p) % 12 for p in pcs if p is not None))
    if not pcs:
        return ""
    if len(pcs) >= 3:
        best_score, best = 999.0, None
        for root in range(12):
            for quality, intervals in (("major", (0, 4, 7)), ("minor", (0, 3, 7))):
                target = [(root + i) % 12 for i in intervals]
                score = sum(min(min(abs(pc - x), 12 - abs(pc - x)) for x in target) for pc in pcs)
                avg = score / len(pcs)
                if avg < best_score:
                    best_score, best = avg, (root, quality)
        if best is not None and best_score <= 0.75:
            return f"{NOTE_NAMES[best[0]]} {best[1]}"
    return "+".join(NOTE_NAMES[p] for p in pcs[:6])


def _detect_common_chords(notes, max_count=5):
    """dev17's _detect_common_chords on (onset, dur, midi) tuples."""
    if len(notes) < 2:
        return []
    notes = sorted(notes, key=lambda x: x[0])[:1500]
    counts, i = {}, 0
    while i < len(notes):
        cluster = [notes[i]]
        j = i + 1
        while j < len(notes) and notes[j][0] - notes[i][0] <= 0.18:
            cluster.append(notes[j])
            j += 1
        if len(cluster) >= 2:
            pcs = []
            for n in cluster:
                pc = int(round(n[2])) % 12
                if pc not in pcs:
                    pcs.append(pc)
            if len(pcs) >= 2:
                label = _chord_label_from_pcs(pcs)
                counts[label] = counts.get(label, 0) + 1
        i = j
    top = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:max_count]
    return [(label, count) for label, count in top if count >= 2 or len(top) <= 3]


def _hm_section_chords(pitched, a, b, key):
    counts = {}
    for stem in _HM_ACC_STEMS:
        sec = [n for n in pitched.get(stem, []) if a <= n[0] < b]
        if len(sec) < 4:
            continue
        for label, cnt in _detect_common_chords(sec, max_count=8):
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


def _hm_avg_poly(notes):
    if not notes:
        return 0.0
    return sum(sum(1 for o2, d2, _ in notes if o2 <= on < o2 + d2) for on, _d, _m in notes) / float(len(notes))


def build_harmony_melody_block(data, A, sections, instrumental):
    """dev17's HARMONY AND MELODY BY SECTION, built from a save's own notes
    and section map, for saves made before dev17 wrote it. "" when nothing
    can be said."""
    try:
        sm = data.get("stem_midi") or {}
        if not sections or not isinstance(sm, dict) or not sm:
            return ""
        kk = _chd_key(A)
        pitched = {s_: _hm_pitched(sm.get(s_)) for s_ in ("vocals", "bass") + _HM_ACC_STEMS}
        vocal_absent = set()
        i = A.find("INTENSITY BY SECTION (vocals stem")
        if i >= 0:
            e = A.find("\n\n", i)
            for m in re.finditer(r"(?m)^\s+.+? \((\d+:\d\d)-(\d+:\d\d)\): (not playing|quiet)",
                                 A[i:e if e >= 0 else len(A)]):
                vocal_absent.add(f"{m.group(1)}-{m.group(2)}")
        lines = []
        for sec in sections:
            a, b = float(sec["start"]), float(sec["end"])
            name = str(sec.get("name") or "").strip() or "Section"
            if b - a < 3.0 or name.lower().startswith("silence"):
                continue
            chords = _hm_section_chords(pitched, a, b, kk)
            parts = ["chords " + (", ".join(chords) + _chd_outside_note(chords, kk) if chords else "none clear")]
            lead_stem, lead = None, []
            if not instrumental and f"{mmss(a)}-{mmss(b)}" not in vocal_absent:
                v = [x for x in pitched.get("vocals", []) if a <= x[0] < b]
                if len(v) >= 8:
                    lead_stem, lead = "vocals", v
            if not lead:
                best = None
                for stem in _HM_ACC_STEMS:
                    s2 = [x for x in pitched.get(stem, []) if a <= x[0] < b]
                    if len(s2) < 8 or _hm_avg_poly(s2) > 1.6:
                        continue
                    med = _median([x[2] for x in s2])
                    if best is None or med > best[0]:
                        best = (med, stem, s2)
                if best:
                    lead_stem, lead = best[1], best[2]
            if not lead:
                parts.append("no clear lead line")
            else:
                lm = [x[2] for x in lead]
                med = _median(lm)
                who = "the voice" if lead_stem == "vocals" else f"the {lead_stem} stem"
                parts.append(f"lead line: {who}, {hz_note(440 * 2 ** ((_percentile(lm, 10) - 69) / 12))}-"
                             f"{hz_note(440 * 2 ** ((_percentile(lm, 90) - 69) / 12))}")
                acc = [x for st in _HM_ACC_STEMS if st != lead_stem for x in pitched.get(st, []) if a <= x[0] < b]
                if len(acc) >= 8:
                    d = med - _median([x[2] for x in acc])
                    parts.append("well above the chords" if d >= 7 else "just above the chords" if d >= 2
                                 else "inside the chords' range" if d > -2 else "below the chords")
                bass = [x for x in pitched.get("bass", []) if a <= x[0] < b]
                if len(bass) >= 8:
                    parts.append(f"{med - _median([x[2] for x in bass]):.0f} semitones above the bass")
                support = [x for st in ("bass",) + _HM_ACC_STEMS if st != lead_stem for x in pitched.get(st, [])
                           if a <= x[0] < b]
                hits = tot = 0
                for on, _dur, m in lead:
                    pcs = {int(round(y[2])) % 12 for y in support if y[0] <= on + 0.05 < y[0] + y[1]}
                    if not pcs:
                        continue
                    tot += 1
                    hits += int(int(round(m)) % 12 in pcs)
                if tot >= 8:
                    share = hits / float(tot)
                    parts.append("mostly on notes the accompaniment is playing (sits inside the harmony)"
                                 if share >= 0.6 else
                                 "a mix of notes the accompaniment is playing and notes it is not "
                                 "(passing or colour notes)" if share >= 0.35 else
                                 "mostly on notes the accompaniment is not playing at that moment "
                                 "(moves independently of the chords, or adds tension)")
            lines.append(f"  {name} ({mmss(a)}-{mmss(b)}): " + "; ".join(parts))
        return (_HM_HEAD + " (rebuilt from the saved notes):\n" + "\n".join(lines)) if lines else ""
    except Exception:
        return ""


def _octaves_above_bass(d):
    """A lead-to-bass gap in semitones, in words (dev17's _hm_octaves_above_bass)."""
    try:
        d = float(d)
    except (TypeError, ValueError):
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


def parse_harmony_melody(A):
    """[{'name', 'start', 'end', 'chords': [...], 'outside': [...],
    'unreliable': bool, 'lead': 'the voice'|'the guitar stem'|..., 'lo', 'hi',
    'register', 'bass_gap', 'match'}] from the HARMONY AND MELODY BY SECTION
    block (figures or words)."""
    t = str(A or "")
    i = t.find(_HM_HEAD)
    if i < 0:
        return []
    out = []
    for line in t[i:].split("\n")[1:]:
        if not line.startswith("  "):
            break
        m = re.match(r"^\s+(.+?) \((\d+:\d\d)-(\d+:\d\d)\): (.+)$", line)
        if not m:
            continue
        row = {"name": m.group(1).strip(), "start": _secs(m.group(2)), "end": _secs(m.group(3)),
               "chords": [], "outside": [], "unreliable": False, "lead": "", "lo": "", "hi": "",
               "register": "", "bass_gap": "", "match": ""}
        body = m.group(4)
        cm = re.match(r"chords (.+?)(?:; |$)", body)
        if cm:
            cs = cm.group(1)
            om = re.search(r"\(([^()]*) outside the measured key[^)]*\)", cs)
            if "most of these are outside" in cs:
                row["unreliable"] = True
            if om:
                row["outside"] = [x.strip() for x in om.group(1).split(",")]
            cs = re.sub(r"\s*\([^)]*\)", "", cs).strip()
            if cs != "none clear":
                row["chords"] = [x.strip() for x in cs.split(",") if x.strip()]
        lm = re.search(r"lead line: (the voice|the \w+ stem), (\S+?)-(\S+?)(?:;|$)", body)
        if lm:
            row["lead"], row["lo"], row["hi"] = lm.group(1), lm.group(2), lm.group(3)
        rm = re.search(r"(well above the chords|just above the chords|inside the chords' range|below the chords)", body)
        if rm:
            row["register"] = rm.group(1)
        bm = re.search(r"(-?\d+) semitones above the bass", body)
        if bm:
            row["bass_gap"] = _octaves_above_bass(int(bm.group(1)))
        else:
            bw = re.search(r"((?:about |less than )[^;]*? above the bass|below the bass)", body)
            if bw:
                row["bass_gap"] = bw.group(1)
        mm = re.search(r"(mostly on notes the accompaniment is playing|a mix of notes the accompaniment is playing "
                       r"and notes it is not|mostly on notes the accompaniment is not playing)", body)
        if mm:
            row["match"] = {"mostly on notes the accompaniment is playing": "lands mostly on the chord notes",
                            "a mix of notes the accompaniment is playing and notes it is not":
                                "mixes chord notes with passing or colour notes",
                            "mostly on notes the accompaniment is not playing":
                                "moves against the chords a lot (adding tension)"}[mm.group(1)]
        out.append(row)
    return out


# ---- the listening model's report fields: what isn't an answer --------------------
_NON_ANSWER_RE = re.compile(
    r"(?i)^\s*(?:uncertain|unknown|unclear|n/?a|not (?:available|assessable|applicable)|\(?none given\)?|"
    r"none(?:\s*\((?:no |nothing )[^)]*\))?|none\s*\([^)]*\bundetected\)|see [A-Z_ ]+)\s*[.;]?\s*$")


def is_non_answer(v):
    """True for an empty report value ("uncertain", "none", "(none given)",
    "none (no audio detected)"...)."""
    return bool(_NON_ANSWER_RE.match(str(v or ""))) or not str(v or "").strip()


_KEY_MENTION_RE = re.compile(r"(?i)\b(?:in |centered in |centred in |key of )?[A-G][#b♯♭]?\s+(?:major|minor)\b")


def clean_description(desc, info):
    """The listening model's free DESCRIPTION without the sentences the
    measurements overrule: its own key or tempo figure, a four-on-the-floor
    beat that wasn't measured, drums on a recording with no kit, chord
    names, and "no vocals" on a sung track."""
    d = str(desc or "").strip()
    if not d or is_non_answer(d):
        return ""
    kick = ((info.get("drums") or {}).get("kick") or "")
    out = []
    for s in re.split(r"(?<=[.!?])\s+", d):
        ls = s.lower()
        # its own key / tempo figures (the measured ones are in the notes) and a
        # four-on-the-floor beat that wasn't measured: the sentence goes.
        if _KEY_MENTION_RE.search(s) or re.search(r"\d+(?:\.\d+)?\s*bpm", ls):
            continue
        if ("four-on-the-floor" in ls or "4-on-the-floor" in ls) and not kick.startswith(
                ("four-on-the-floor", "mostly four-on-the-floor")):
            continue
        if info.get("no_kit") and _DRUM_WORD_RE.search(s):
            continue
        if re.search(r"\b(?:chord progressions?|cycling through)\b", ls) and re.search(r"\b[A-G][#b]?m?\d", s):
            continue
        if info.get("vocals_present") != "no" and _INSTRUMENTAL_PROSE_RE.search(s):
            continue
        if info.get("vocals_present") == "no" and re.search(r"(?i)\b(?:vocal|singer|sung|lyric)", s) \
                and not re.search(r"(?i)\bno (?:discernible )?vocal|without vocal|instrumental", s):
            continue
        s = s.strip()
        if s:
            out.append(s)
    return " ".join(out)


def clean_report_value(v):
    """Report-field tidy shared by every by-ear line: joined instrument names
    ("electricguitar") and run-on spacing."""
    v = str(v or "")
    for a, b in (("electricguitar", "electric guitar"), ("acousticguitar", "acoustic guitar"),
                 ("electricbass", "electric bass"), ("bassguitar", "bass guitar")):
        v = re.sub(r"(?i)\b" + a + r"\b", b, v)
    # 2.2.2: another report field pasted into this one ("...; phrase STRUCTURE=[ Intro | 0:00-0:19
    # Chorus 1 | ...") -- the listening model copying its template -- is cut off.
    m = re.search(r"[;,]?\s*(?:\b[a-z]+\s+)?\b[A-Z][A-Z_]{3,}=\s*[\[(<]", v)
    if m and m.start() >= 20:
        v = v[:m.start()].rstrip(" ;,")
    return " ".join(v.split())


def parse_chords_line(v):
    """dev17's CHORDS line: (measured chords, outside-key note, heard chords,
    heard outside note). Older free-text lines come back as ([], '', [v], '')."""
    v = str(v or "")
    if "heard by the listening model:" not in v and "(measured from the separated instruments" not in v:
        return [], "", ([v] if v and not is_non_answer(v) else []), ""
    meas_s, _sep, heard_s = v.partition("heard by the listening model:")
    meas, mo = [], ""
    mm = re.match(r"\s*(.*?)\s*\(measured from the separated instruments[^)]*\)\s*(\([^)]*\))?", meas_s)
    if mm:
        meas = [x.strip() for x in mm.group(1).split(",") if x.strip()]
        mo = (mm.group(2) or "").strip("() ")
    heard_s = heard_s.strip().strip(";").strip()
    ho = ""
    hm = re.search(r"\s*\(([^()]*outside the measured key[^()]*)\)\s*$", heard_s)
    if hm:
        ho = hm.group(1)
        heard_s = heard_s[:hm.start()]
    heard = [] if is_non_answer(heard_s) else [x.strip() for x in heard_s.split(",") if x.strip() and x.strip() != "…"]
    return meas, mo, heard, ho



# ---- loudness context and the era caveat (dev49) ---------------------------------
def loudness_context(lufs):
    if lufs >= -10.0:
        where = ("about as loud as typical current commercial pop and rock masters (roughly -6 to "
                 "-10 LUFS)" if lufs <= -6.0 else
                 "louder than most current commercial masters (roughly -6 to -10 LUFS)")
    elif lufs > -12.0:
        where = ("a little quieter than most current commercial pop and rock masters (roughly -6 "
                 "to -10 LUFS), though louder than the -14 LUFS level streaming services play back at")
    elif lufs > -15.0:
        where = ("clearly quieter than current commercial pop and rock masters (roughly -6 to -10 "
                 "LUFS), close to the -14 LUFS level streaming services play back at")
    else:
        where = ("much quieter than current commercial masters (roughly -6 to -10 LUFS) and below "
                 "the -14 LUFS streaming playback level")
    return (f"{lufs:g} LUFS is {where}. Use this when asked how loud it is compared with today's "
            "music or the streaming era; the word 'loud' or 'moderate' is an absolute level, not a "
            "comparison with current releases.")


def era_mastering_caveat(lufs, lra, crest, saved=False):
    """The dev49 era caveat on the master: None when it doesn't apply. `saved`:
    the analysis already carries Musiclyse's own (pre-dev49) caveat."""
    if lufs is None:
        return None
    if not saved and (lra is None or not (lufs >= -13.0 and lra <= 3.5)):
        return None
    lra_txt = f"{lra:g} LU loudness range" if lra is not None else "a steady level"
    if crest is not None and crest >= 13.5 and lufs < -10.0:
        return (f"the steady level ({lufs:g} LUFS, {lra_txt}) comes with peaks still well "
                f"above the average (crest {crest:g} dB). That is not the crushed, heavily limited "
                "loudness-war profile common on late-1990s and 2000s pop and rock masters; it is typical "
                "of early-to-mid-1990s masters and of gentler modern ones. A tight loudness range only "
                "means the level stays steady from section to section. Don't use this master as evidence "
                "for the 2000s or later; reason from genre, style and performance instead.")
    return (f"this loud ({lufs:g} LUFS), tightly compressed ({lra_txt}) master is "
            "ordinary for anything mastered from roughly the early 2000s onward, so on its own it can't "
            "tell a 2002 recording from a brand-new one. Don't say or imply the track was remastered — "
            "that's a loudness number, not evidence of a mastering event. It's only worth mentioning if "
            "other clues (genre, playing style, period instruments) already point before the mid-to-late "
            "1990s, and then as an open question. Otherwise reason from genre, style and performance.")


# ---- era clues measured from the audio (dev40 / dev47 / dev49 blind era evidence) ---
_ERA_WORD_RE = re.compile(
    r"(?:\b(?:early|mid|late)[\s\-‐‑–]+)?\b(?:19|20)?[0-9]0['’]?s\b"
    r"(?:\s*(?:[\-‐‑–/]|to|or|and)\s*(?:(?:early|mid|late)[\s\-‐‑–]+)?"
    r"(?:19|20)?[0-9]0['’]?s\b)?"
    r"|\b(?:modern|contemporary|recent|vintage|retro|old-school|dated)\b",
    re.IGNORECASE)
_RECOMMENDED_GENRE_RE = re.compile(r"RECOMMENDED GENRE FOR DISCUSSION: ([^\n]+?)\. Reasoning:")


def _mf_era_phrases(text, limit=6):
    t = str(text or "")
    spans, cut = [], len(t)
    for mk in ("\n\nLYRICS:", "\n\n11. ERA / RELEASE PERIOD", "\n\nVOCAL / SINGER PROFILE",
               "[Independent signal-processing report", "\nRECOMMENDED TEMPO FOR DISCUSSION"):
        k = t.find(mk)
        if 0 <= k < cut:
            cut = k
    spans.append(t[:cut])
    k = t.find("11. ERA / RELEASE PERIOD")
    if k >= 0:
        e = t.find("\n\n", k)
        spans.append(t[k:e if e >= 0 else len(t)])
    out = []
    for s in spans:
        covered = -1
        for m in _ERA_WORD_RE.finditer(s):
            if m.start() < covered:
                continue
            a = max(s.rfind(ch, 0, m.start()) for ch in ".,;:\n") + 1
            ends = [x for x in (s.find(ch, m.end()) for ch in ".,;\n") if x >= 0]
            b = min(ends) if ends else len(s)
            if b - a > 110:
                a2, b2 = max(a, m.start() - 50), min(b, m.end() + 50)
                a = s.find(" ", a2) + 1 if a2 > a and s.find(" ", a2) < m.start() else a2
                b = s.rfind(" ", m.end(), b2) if b2 < b and s.rfind(" ", m.end(), b2) > m.end() else b2
            covered = b
            clause = " ".join(s[a:b].split()).strip(" -")
            if not clause or re.match(r"^\d+\)", clause) or "(isolated dedicated pass)" in clause:
                continue
            if "=" in clause:
                clause = clause.split("=", 1)[1].strip()
            if clause and clause not in out:
                out.append(clause)
            if len(out) >= limit:
                return out
    return out


def _era_num(rx, text):
    m = re.search(rx, text)
    try:
        return float(m.group(1)) if m else None
    except (TypeError, ValueError):
        return None


def era_evidence(A):
    """(clue lines, measured-range verdict) read from the measurements -- the
    same clues and thresholds dev53 gives its writer for "when does it sound
    like it's from?" questions. No tag, year or name is used."""
    t = str(A or "")
    lufs = _era_num(r"EBU R128 integrated loudness: (-?\d+(?:\.\d+)?) LUFS", t)
    if lufs is None:
        lufs = _era_num(r"integrated loudness ≈ (-?\d+(?:\.\d+)?) LUFS", t)
    plr = _era_num(r"peak-to-loudness ratio \(PLR\): (\d+(?:\.\d+)?) dB", t)
    crest = _era_num(r"crest factor / dynamic-range proxy: (\d+(?:\.\d+)?) dB", t)
    sub = _era_num(r"band energy share, whole-track[^\n]*\n\s*sub-bass \(20-60 Hz\): (\d+(?:\.\d+)?)%", t)
    corr = _era_num(r"stereo correlation \(L/R\): (-?\d+(?:\.\d+)?)", t)
    sc = re.search(r"sidechain reading: ([^\n]*)", t)
    pumping = bool(sc and "ducks strongly in lockstep" in sc.group(1))
    # Pre-dev31 saves over-claimed pumping (strummed on-beat chords read the
    # same); the dossier treats that reading as unconfirmed, so it isn't an era clue.
    old_pumping = pumping and "recovers between beats" not in sc.group(1)
    lossy = "provenance caution: this file is itself a lossy" in t
    mf_words = _mf_era_phrases(t)
    genre_line = ""
    rg = list(_RECOMMENDED_GENRE_RE.finditer(t))
    rec = rg[-1].group(1).strip() if rg else ""
    if rec.lower().startswith("unavailable"):
        rec = ""
    pair = _genre_superseded_pair(t)
    if pair:
        rec = rec or pair[0]
        old = pair[1]
        rock_words = [w for w in _HYPHENS_RE.split(old.lower()) if w and w != "pop"]
        rock_words += ["rock", "punk", "emo", "grunge", "metal"]
        mf_words = [w for w in mf_words
                    if not any(re.search(r"(?<![\w])" + re.escape(x), w.lower()) for x in rock_words)]
        if not _guitar_stem_reading(t)[0]:
            genre_line = (f"Genre (measured): {rec}. The first-impression '{old}' label was replaced, so era "
                          f"words tied to it don't apply: place the era from its {rec} sound, not by a "
                          f"{_rock_label_list(old, with_rock=False)} wave.")
        else:
            genre_line = (f"Genre (measured): {rec}, with a prominent guitar part. Place the era from its "
                          f"{rec} sound first.")
    elif rec:
        genre_line = f"Genre (measured): {rec}. Place the era from that sound."
    if lufs is None and sub is None and not mf_words and not genre_line:
        return [], ("", "")

    lines = [genre_line] if genre_line else []
    master = None
    if lufs is not None:
        tight = (plr is not None and plr <= 11.5) or (plr is None and crest is not None and crest <= 13.0)
        open_ = (plr is not None and plr >= 11.5) or (plr is None and crest is not None and crest >= 14.0)
        figs = f"loudness {lufs:g} LUFS" + (f", peak-to-loudness ratio {plr:g} dB" if plr is not None
                                            else (f", crest factor {crest:g} dB" if crest is not None else ""))
        if lufs >= -10.5 and tight:
            master = "loud"
            lines.append(f"Master: {figs} — a loud, limited master. Normal from the late 1990s onward and "
                         "still made today; uncommon before about 1995 except on later remasters. On its own "
                         "it says 'late 1990s or later', not the 2010s or 2020s in particular.")
        elif lufs <= -15.0 and open_:
            master = "open"
            lines.append(f"Master: {figs} — a very quiet, dynamic master, typical of records mastered before "
                         "the mid-1990s (later mainstream pop and rock masters are much louder, and a later "
                         "remaster normally measures louder, not quieter; jazz, classical, folk and "
                         "audiophile records are the usual exceptions).")
        elif lufs <= -13.0 and open_:
            master = "open"
            lines.append(f"Master: {figs} — a quiet, dynamic master, typical of records mastered before the "
                         "mid-1990s, though some recent masters made for streaming (around -14 LUFS) and "
                         "jazz, classical, folk and audiophile records measure like this too.")
        elif -12.5 < lufs < -10.5 and ((plr is not None and plr >= 11.0) or
                                        (plr is None and crest is not None and crest >= 13.5)):
            master = "headroom"
            lines.append(f"Master: {figs} — moderately loud with real peak headroom left. Not the crushed, "
                         "heavily limited profile of late-1990s and 2000s pop and rock masters (usually -9 "
                         "LUFS or louder with little room above the average); typical of early-to-mid-1990s "
                         "masters and of gentler modern ones.")
        else:
            master = "middle"
            lines.append(f"Master: {figs} — a middling master, common from the early 1990s on and in gentler "
                         "modern masters; little era information on its own.")
    modern = []
    if sub is not None:
        if sub >= 25.0:
            modern.append("a sub-bass-heavy low end")
            lines.append(f"Low end: sub-bass carries {sub:g}% of the energy — the sub-heavy (808-style) "
                         "balance typical of pop, R&B and hip-hop from the late 2000s onward.")
        else:
            lines.append(f"Low end: sub-bass carries {sub:g}% of the energy — not the sub-heavy (808-style) "
                         "balance that marks late-2000s-onward pop, R&B and hip-hop.")
    if sc:
        if old_pumping:
            lines.append("Sidechain: beat-locked level movement from an older analysis that can't tell "
                         "pumping from chords strummed on the beat — not an era clue.")
        elif pumping:
            modern.append("sidechain pumping")
            lines.append("Sidechain: sustained parts duck strongly with the beat (pumping) — common in "
                         "dance and pop from the mid-2000s on.")
        else:
            lines.append("Sidechain: no clear sidechain pumping.")
    if corr is not None and corr >= 0.97 and master != "loud" and not lossy:
        lines.append(f"Stereo: correlation {corr:g} — a near-mono image, usual for the 1960s and earlier "
                     "(or a mono release).")
    if lossy:
        lines.append("Source: this file is a lossy encode, so a missing top end or narrow stereo image "
                     "comes from the encode and says nothing about era.")
    verdict = {
        "loud": (("late 2000s or later", "fits the late 2000s or later best (a loud master plus "
                  + " and ".join(modern) + ")")
                 if modern else
                 ("late 1990s or later", "late 1990s or later, with nothing measured that is specific to "
                  "the 2010s or 2020s — place it within that range from the arrangement, drum, guitar/synth "
                  "and vocal-production style, and give a range")),
        "open": ("before the mid-1990s, most likely", "most consistent with a master from before the "
                 "mid-1990s (exceptions: deliberately dynamic genres and some recent streaming masters, "
                 "which the style tells apart); words like 'modern' in the by-ear descriptions describe "
                 "clean recording quality, not a date"),
        "middle": ("not fixed by the master — the style decides", "no firm era from the master; place it "
                   "from the style clues"),
        "headroom": ("early 1990s or later, probably not a 2000s loudness-war master",
                     "early 1990s or later; a late-1990s/2000s loudness-war master is unlikely, though a "
                     "gentle modern master is possible — the style clues decide, and clean, polished "
                     "playing alone is not a sign of the 2000s"),
    }.get(master, ("", ""))
    if mf_words:
        lines.append("Era words used by ear in these notes: " + "; ".join(f"\"{w}\"" for w in mf_words)
                     + ". That listener calls nearly any clean, loud digital master modern or recent; where "
                     "its words go beyond the measured range, go with the measured range and the style.")
    return lines, verdict


ERA_BY_EAR_GUIDE = (
    "\"When does it sound like it's from?\" asks for the era the record evokes — arrangement, drum and "
    "instrument sounds, vocal style and production taken together — not just how recently it was "
    "mastered; if production and style point to different times, say so. Loud, limited, noise-free "
    "digital masters and audible pitch correction have been normal since about 1998, so a 'recent' "
    "reading that rests only on them means 'late 1990s or later'. Clues that do separate the eras "
    "(weigh several, none is decisive): late 2010s onward — trap-style hi-hat rolls, 808 sub-bass, "
    "sidechain pumping, vocal chops, drop-style post-choruses, sparse minimal verses, very bass-heavy "
    "balance; early 2010s — four-on-the-floor EDM-pop, big saw-synth leads with builds and drops; late "
    "1990s to mid 2000s dance-pop and teen pop — glossy programmed kits with syncopated kicks (rarely "
    "four-on-the-floor), bright synth stabs and plucked or guitar-like synth riffs, stacked harmony "
    "vocals, big sing-along hooks, a last-chorus key change, a loud clean master without 808 weight or "
    "pumping; late 1990s and 2000s generally — dense layered full-band or R&B-pop arrangements, crisp "
    "kits with crash-led choruses, crunchy pop-rock guitars, acoustic-guitar ballads, very loud and "
    "clipped masters; late 1980s to mid 1990s alternative, college and jangle rock — clean, jangly or "
    "lightly chorused guitars carrying the chords, a natural live kit, a plain vocal with modest reverb, "
    "masters around -14 to -10 LUFS that keep their peaks (2000s indie rock was usually mastered much "
    "louder); 1980s — gated-reverb snares, bright FM/digital keys and synth brass, chorused guitars; "
    "1970s and earlier — tape warmth and hiss, roomy live drums, narrow or hard-panned stereo, wide "
    "dynamics. A groove that isn't four-on-the-floor is not a rock clue on its own. Give a range when "
    "the clues are mixed, and say which clues you went by."
)


# ---- generic "label: value" measurement blocks -------------------------------------
# Blocks of the saved analysis that hold measurements the dossier lays out as
# lists. Each is (header prefix, dossier title, standard level?).
MEASURE_BLOCKS = [
    ("BEAT / TEMPO MEASUREMENTS", "Rhythm, beat and meter (signal analysis)", True),
    ("TIMBRE / ELEMENT MEASUREMENTS", "Tone, brightness and frequency balance (signal analysis)", True),
    ("DYNAMIC RANGE / LOUDNESS (numeric)", "Loudness figures", True),
    ("VOCAL OBJECTIVE MEASUREMENTS", "Voice measurements", True),
    ("ESSENTIA OBJECTIVE MEASUREMENTS", "Second, independent set of signal measurements", True),
    ("OBJECTIVE GENRE/MOOD SIGNAL", "Genre and mood classifier", True),
    ("DRUM SOUND / PRODUCTION", "Drum sound (measured from the separated drums)", True),
    ("KEY PROFILE REFINEMENT", "Key check", True),
    ("OBJECTIVE AUDIO MEASUREMENTS", "File", False),
    ("MIDI ACTIVITY BOUNDARY CANDIDATES", "Raw note-density change points (cross-check only, not section boundaries)", False),
    ("candidate change points:", "Raw sound-change points (cross-check only, not section boundaries)", False),
    ("ARRANGEMENT CHANGE POINTS", "Every point where a separated part comes in or drops out ('other' = the keys/synth "
     "layer; a single part moving alone is often just a phrase break)", False),
]

# Sentences written as orders to Musiclyse's own writer model.
_WRITER_ORDER_RE = re.compile(
    r"\b(?:Do NOT|do NOT|Do not|do not|NEVER|MUST|Use this|Use these|use it only|use this|"
    r"This is the primary|the writer|report to the user|WARNING:|Prefer median|Treat VOCALS_PRESENT)\b")
_EVENT_LOG_RE = re.compile(r"^(?:note|hit) events \((?:compact|MIDI-JSON|rhythm-JSON)\):")


def _drop_writer_orders(text):
    """Remove sentences addressed to the analyser's writer model; keep the facts."""
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z(\[])", text.strip())
    keep = [p for p in parts if not _WRITER_ORDER_RE.search(p)]
    return " ".join(keep).strip()


def measurement_lines(body_lines, full=False, max_line=320):
    """Measurement lines of one block as dossier bullets (indented sub-lines nest)."""
    out = []
    for raw in body_lines:
        if not raw.strip() or _EVENT_LOG_RE.match(raw.strip()):
            continue
        indent = len(raw) - len(raw.lstrip())
        s = _drop_writer_orders(raw.strip())
        if not s or s.startswith("[") or s.startswith("This is computed directly"):
            continue
        s = s.lstrip("-• ").strip()
        if not full:
            s = clip(s, max_line)
        out.append(("  - " if indent >= 2 else "- ") + s)
    return out


def paragraph_after(A, header):
    """(header line, following lines) of the paragraph that starts with `header`."""
    m = re.search(r"(?m)^[ \t]*" + re.escape(header) + r".*$", A)
    if not m:
        return "", []
    rest = A[m.end():]
    end = rest.find("\n\n")
    body = rest[:end if end >= 0 else len(rest)].splitlines()
    return m.group(0).strip(), [l for l in body if l.strip()]


def stem_blocks(A):
    """{stem: [lines]} for every "[X stem]" block of the stem MIDI report."""
    out = {}
    for m in re.finditer(r"(?m)^\[([A-Z]+) stem\][^\n]*\n", A):
        stem = m.group(1).lower()
        if stem in out:
            continue
        rest = A[m.end():]
        end = rest.find("\n\n")
        out[stem] = [l for l in rest[:end if end >= 0 else len(rest)].splitlines() if l.strip()]
        head = m.group(0).strip()
        if "MIDI transcription failed" in head:
            out[stem].insert(0, head.split("]", 1)[1].strip())
    return out


# Lines of a stem block the dossier already shows in its own sections.
_STEM_SHOWN_ELSEWHERE = (
    "instrument tag", "STEM IDENTITY NOTE", "kick beat-grid analysis", "snare beat-grid analysis",
    "swing/shuffle analysis", "hi-hat density vs kick pulse", "GROOVE_HINT", "EARLIEST VOCAL ONSET",
    "BEAT GRID", "count:", "kick :", "snare:", "hi-hat:", "toms :", "cymbals:", "perc :", "other:",
) + tuple(p for p, _l in DRUM_KIT_PREFIXES) + tuple(p for p, _l in DRUM_SOUND_PREFIXES)


# ---- structure replay data (dev44) ----------------------------------------------------
def _sr_decode(x):
    """Decode Musiclyse's structure-replay encoding (arrays become their shape only)."""
    if isinstance(x, list):
        return [_sr_decode(v) for v in x]
    if isinstance(x, dict):
        if "__nd__" in x:
            return {"array_shape": x.get("shape")}
        if "__tuple__" in x:
            return tuple(_sr_decode(v) for v in x["__tuple__"])
        if "__set__" in x:
            return [_sr_decode(v) for v in x["__set__"]]
        if "__dict__" in x:
            return {str(_sr_decode(k)): _sr_decode(v) for k, v in x["__dict__"]}
        return {k: _sr_decode(v) for k, v in x.items()}
    return x


def parse_replay(rp):
    """What a dev44+ "structure_replay" holds, in readable form."""
    if not isinstance(rp, dict) or not rp:
        return {}
    try:
        inp = _sr_decode(rp.get("inputs") or {})
    except Exception:
        return {"format": str(rp.get("format") or ""), "error": "unreadable"}
    out = {"format": str(rp.get("format") or "")}
    beats = [b for b in (inp.get("beats") or []) if isinstance(b, (int, float))]
    out["beats"] = beats
    lines = []
    for c in inp.get("lyric_chunks") or []:
        if isinstance(c, dict) and str(c.get("text") or "").strip():
            try:
                lines.append((float(c.get("start") or 0.0), float(c.get("end") or c.get("start") or 0.0),
                              str(c["text"]).strip()))
            except (TypeError, ValueError):
                continue
    out["lyric_lines"] = lines
    groups = []
    for g in inp.get("lyric_groups") or []:
        if isinstance(g, dict) and g.get("text"):
            occ = []
            for o in g.get("occurrences") or []:
                try:
                    occ.append(float(o[0]))
                except (TypeError, ValueError, IndexError, KeyError):
                    continue
            groups.append((str(g["text"]), int(g.get("count") or len(occ)), occ))
    out["lyric_groups"] = groups
    hints = []
    for h in (inp.get("moss_hints") or inp.get("mf_hints") or []):
        try:
            a, b, lab = h[0], h[1], h[2]
            hints.append((float(a), float(b), str(lab)))
        except (TypeError, ValueError, IndexError, KeyError):
            continue
    out["mf_hints"] = hints
    bf = inp.get("beatfeat")
    out["beatfeat_shape"] = bf.get("array_shape") if isinstance(bf, dict) else None
    out["total_duration"] = inp.get("total_duration")
    out["instrumental"] = inp.get("instrumental")
    known = {"beats", "lyric_chunks", "lyric_groups", "mf_hints", "moss_hints", "beatfeat", "total_duration",
             "instrumental"}
    out["other_inputs"] = {k: v for k, v in inp.items() if k not in known}
    return out


# Top-level fields of a save, as written by Musiclyse 0.4-dev53 save_song_data().
KNOWN_TOP_KEYS = {
    "version", "track_path", "label", "saved_at", "audio_file_signature", "analysis", "corrections",
    "metadata", "lyrics", "cover_observations", "singer_identity", "cover_art", "clap_embedding_model",
    "clap_embedding_dimension", "clap_embedding_aggregation", "clap_embedding", "stem_midi",
    "stem_midi_format", "stem_midi_complete", "structure_sections", "bpm", "key", "structure_replay",
    # 2.3: read into the dossier (styles, chords), not listed raw under "About the file"
    "discogs_style_model", "discogs_styles", "chord_model", "chord_segments",
}
KNOWN_META_KEYS = {"title", "artist", "album", "year", "lyrics"}
COVER_FIELDS = [("text_logos", "Text on the cover"), ("people", "People shown"),
                ("person_cues", "Who they look like"), ("style_vibe", "Look"),
                ("era_cues", "Visual era cues (about the artwork, not the sound)"),
                ("confidence", "Confidence of these observations")]


def _plain_value(v, n=400):
    """Short readable rendering of any JSON value."""
    if isinstance(v, (dict, list)):
        s = json.dumps(v, ensure_ascii=False)
    else:
        s = str(v)
    return clip(" ".join(s.split()), n)


def parse_all_fields(text, max_len=1500):
    """Every LABEL=value field in `text`, in order (first occurrence wins)."""
    labels = []
    for line in strip_md(text).splitlines():
        m = _LABEL_RE.match(line)
        lab = _label_of(m) if m else None
        if lab and lab not in labels:
            labels.append(lab)
    return parse_fields(text, labels, max_len) if labels else {}


# ---- dev55-dev59 voice evidence ------------------------------------------------
_VOICE_UNBROKEN_MARK = "VOICE MATURITY: the voice sounds UNBROKEN"
_VOICE_DESC_RE = re.compile(
    r"LISTENING-MODEL VOICE DESCRIPTIONS \(free text, audio-based\): (\d+) describe a male voice, "
    r"(\d+) a female voice -> ([^.(]+?)(?: \(described as teenage\))?\. Quotes: (.*)$", re.M)


def parse_voice_descriptions(A):
    """dev55+: what the listening model said about the voice in its own words,
    whether dev57's probability veto set it aside, and dev59's UNBROKEN
    signature. {} when the line is absent."""
    m = _VOICE_DESC_RE.search(A or "")
    if not m:
        return {}
    tail = m.group(4)
    quotes = re.findall(r'"([^"]{3,200})"', tail.split(" -- NOT used")[0])
    veto = ""
    mv = re.search(r"-- NOT used for FINAL LEAD PROFILE: (.+?); the two disagree", tail)
    if mv:
        veto = mv.group(1).strip()
    out = {"male": int(m.group(1)), "female": int(m.group(2)), "lean": m.group(3).strip(),
           "teen": "(described as teenage)" in m.group(0), "quotes": quotes, "vetoed": veto,
           "unbroken": _VOICE_UNBROKEN_MARK in tail}
    mu = re.search(r"rated it child-like or a teenage boy's at ([\d.]+) together \(adult man ([\d.]+), "
                   r"woman ([\d.]+)\)", tail)
    if mu:
        out["unbroken_scores"] = tuple(float(x) for x in mu.groups())
    return out


_NOT_NAME_WORDS = {
    "a", "an", "the", "male", "female", "adult", "young", "unknown", "uncertain", "unclear", "not", "no",
    "one", "two", "several", "both", "either", "likely", "probably", "possibly", "teen", "teenage",
    "child", "boy", "girl", "man", "woman", "he", "she", "they", "it", "this", "that", "his", "her",
    "their", "post", "mixed", "credited", "documented", "listed", "same", "also", "most", "main",
}


def _plausible_name(nm):
    """A capitalised run that is a person or act, not a sentence word
    ("A male adult", "Female, judging...", "Unknown at this time")."""
    words = str(nm or "").split()
    if not words:
        return False
    if len(words) == 1:
        return len(words[0]) > 1 and words[0].lower().strip(".,'’") not in _NOT_NAME_WORDS
    return True


# dev19 TAG_FEATURE_SHARED_LEAD (recomputed for older saves; same rule and wording).
_TAG_FEAT_RE = re.compile(
    r"(?i)(?:^|[\s(\[])(?:feat(?:\.|\u2024)?|ft(?:\.|\u2024)|featuring)\s+(.+?)\s*(?:[)\]]|$|\s[-\u2013\u2014]\s)")


def tag_featured_singers(meta):
    names = []
    for k in ("artist", "title", "album_artist", "albumartist"):
        v = str((meta or {}).get(k) or "") if isinstance(meta, dict) else ""
        for m in _TAG_FEAT_RE.finditer(v):
            for nm in re.split(r"\s*(?:,|&|\band\b|\bx\b|/)\s*", m.group(1)):
                nm = nm.strip(" .\u2024")
                if nm and len(nm) >= 2 and nm.lower() not in [n.lower() for n in names]:
                    names.append(nm)
    return names


def tag_lyrics_usable(lyrics, meta=None):
    """dev19 TAG_LYRICS_SANITY: a lyrics tag with fewer than 12 words, or one
    short line naming the artist or title, is not lyrics."""
    t = str(lyrics or "").strip()
    if not t:
        return False
    words = re.findall(r"[^\W_]+(?:['\u2019][^\W_]+)?", t)
    if len(words) < 12:
        return False
    lines = [x for x in t.splitlines() if x.strip()]
    if len(lines) == 1 and len(words) < 25:
        low = t.lower()
        for k in ("artist", "title"):
            v = str((meta or {}).get(k) or "").strip().lower() if isinstance(meta, dict) else ""
            if len(v) >= 3 and re.search(r"(?<!\w)" + re.escape(v) + r"(?!\w)", low):
                return False
    return True


def feature_credit_identity(si, meta):
    """LEAD_SINGERS=one with a featured singer credited in the tags -> several."""
    t = str(si or "")
    try:
        m0 = re.search(r"LEAD_SINGERS\s*[=:]\s*[\"'<\[]?\s*(one|single)\b", t, re.I)
        if not t or "TAG FEATURE CHECK" in t or not m0:
            return si
        names = tag_featured_singers(meta)
        if not names:
            return si
        t2 = re.sub(r"(LEAD_SINGERS\s*[=:]\s*[\"'<\[]?\s*)(one|single)\b", r"\g<1>several", t, count=1,
                    flags=re.I)
        add = (f" TAG FEATURE CHECK: a {' and '.join(names)} feature is credited in the file's tags, so the "
               "lead is shared, even though the audio passes heard one voice.")
        m = re.search(r"(?m)^(\W*REASONING\s*[=:].*)$", t2)
        return t2[:m.end()] + add + t2[m.end():] if m else t2.rstrip() + "\n" + add.strip()
    except Exception:
        return si


_UV_SAYS_UNBROKEN_RE = re.compile(
    r"(?i)\bunbroken\b|\bfemale[- ]sounding\b|\b(?:read|reads|heard|sounds?)\s+(?:as\s+)?(?:an?\s+)?"
    r"(?:high\s*/\s*)?(?:female|woman)\b")
_UV_SAYS_DEEPENED_RE = re.compile(
    r"(?i)\b(?:not|no longer|never|isn't|wasn't)\b[^.\n]{0,25}\bunbroken\b|"
    r"\balready\s+(?:broken|deepened|dropped)\b|\bhad\s+(?:already\s+)?(?:broken|deepened|dropped)\b")


def identity_reason_says_unbroken(reason):
    """True when an identity's REASONING records a female-sounding or
    unbroken voice and does not say the voice had already deepened."""
    r = str(reason or "")
    return bool(_UV_SAYS_UNBROKEN_RE.search(r)) and not _UV_SAYS_DEEPENED_RE.search(r)


_FEMALE_TAGS = ("female_teen_adult", "adult_female", "young_female")


def confirm_flip_reading(A):
    """dev18 (YOUNG_VOICE_CONFIRM_FLIP): True when the VOCAL DECISION AUDIT shows
    the confirmation pass alone turning the first vocal pass's female reading
    into a man's voice and the lead's measured median pitch (both readings,
    when both exist) is 300 Hz or more. The analyser's own test, for saves
    made before dev18."""
    t = str(A or "")
    mi = re.search(r"(?m)^- Initial isolated lead profile:\s*([a-z_]+)", t)
    mc = re.search(r"(?m)^- Confirmation pass:\s*([a-z_]+)\s*\(confidence=", t)
    mf = re.search(r"(?m)^- FINAL LEAD PROFILE:\s*([a-z_]+)", t)
    if not (mi and mc and mf) or mi.group(1) not in _FEMALE_TAGS:
        return False
    if mc.group(1) not in ("post_puberty_male", "adult_male") or mf.group(1) != mc.group(1):
        return False
    meds = []
    m = re.search(r"objective voiced pitch median:\s*([\d.]+)\s*Hz", t)
    if m:
        meds.append(float(m.group(1)))
    m = re.search(r"VOCAL NOTE PITCH \(second reading[^)]*\):\s*median\s*([\d.]+)\s*Hz", t)
    if m:
        meds.append(float(m.group(1)))
    return bool(meds) and min(meds) >= 300.0


# 2.2.3 SINGER_NAME_ANY_WORDING: a person's name after a singer word, anywhere
# in the same clause. The identity model rewords its reasoning on every run, so
# fixed phrasings kept missing ("documented lead vocalist for the 1983-era
# line-up is John Doe").
_ROLE_RE = re.compile(
    r"\b(?:vocalist|singer|frontman|frontwoman|front-man|front-woman|"
    r"(?:lead )?vocals (?:by|from))\b", re.I)
# A clause ends at a sentence stop (not an initial or "Jr."), a semicolon, a
# dash or a new line.
_CLAUSE_END_RE = re.compile(r"(?<![A-Z])(?<!\bJr)(?<!\bSr)(?<!\bSt)\.(?=\s|$)|[;!?\n]|\s[–—-]\s")
_NAME_TOKEN = r"(?:[A-Z][a-z]+(?:[-'’][A-Z]?[a-z]+)*|Mc[A-Z][a-z]+|Mac[A-Z][a-z]+|O['’][A-Z][a-z]+|[A-Z]\.)"
_NAME_PARTICLE = r"(?:de|van|von|la|le|del|der|da|di|du|dos|bin|al)"
_SINGER_NAME_RUN_RE = re.compile(
    rf"(?<![\w’'-]){_NAME_TOKEN}(?:\s+(?:{_NAME_PARTICLE}\s+)?{_NAME_TOKEN}){{0,3}}(?![\w’'-])")
# Capitalised words that are not a person's name.
def _is_secondary_singer_role(text, role_start):
    """Do not mistake an explicitly described guest/co-vocalist for the main lead."""
    prefix = str(text or "")[max(0, int(role_start) - 45):int(role_start)]
    return bool(re.search(
        r"(?i)(?:\b(?:guest|featured|second|additional)\s+|"
        r"\bco[- ]?(?:lead|vocalist)\s*[- ]?|\bco[- ]?)$", prefix
    ))


_NOT_PERSON_WORDS = {
    "the", "track", "song", "album", "single", "record", "recording", "release", "metadata", "background",
    "vocal", "vocals", "acoustic", "final", "identity", "wikipedia", "listening", "music", "band", "group",
    "male", "female", "documented", "credited", "solo", "artist", "lead", "singer", "vocalist", "frontman",
    "american", "british", "english", "canadian", "irish", "scottish", "welsh", "australian", "swedish",
    "january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
    "november", "december", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    "both", "his", "her", "their", "its", "it", "this", "that", "these", "those", "he", "she", "they",
    "a", "an", "and", "or", "but", "with", "for", "of", "on", "in", "at", "by", "as", "is", "was",
    "acoustically", "identity", "evidence", "confirmation", "cover", "art", "pass", "profile", "tag",
    "tags", "feature", "featured", "check", "era", "line-up", "lineup", "debut", "boy", "girl",
}
_GROUP_WORD_RE = re.compile(
    r"(?i)\b(?:band|group|trio|quartet|quintet|sextet|duo|brothers|sisters|siblings|boy band|girl group|"
    r"line-up|lineup|members)\b")
_SOLO_WORD_RE = re.compile(r"(?i)\bsolo (?:artist|act|singer|performer|male artist|female artist)\b")


def _fold_name(s):
    return re.sub(r"[^a-z0-9]+", "", unicodedata.normalize("NFKD", str(s or "")).lower())


def _person_name_ok(run, act_names=()):
    words = run.split()
    if not words:
        return False
    if words[0].lower().strip(".") in _NOT_PERSON_WORDS:
        return False
    if any(w.lower().strip(".,'’") in _NOT_PERSON_WORDS for w in words):
        return False
    f = _fold_name(run)
    for act in act_names:
        fa = _fold_name(act)
        if fa and (f == fa or f in fa):
            return False
    if all(re.fullmatch(r"[A-Z]\.", w) for w in words):
        return False
    return True


def singer_name_from_reason(reason, act_names=()):
    """2.2.3: the person named as the singer, from any wording that puts a
    capitalised personal name after a singer word ("vocalist", "singer",
    "frontman", "frontwoman", "vocals by") in the same clause, with words in
    between allowed. Band names, other capitalised words and the act's own
    name (act_names: the artist tag, its main act, the title) are skipped,
    and the next name in the clause is tried. A one-word name only counts
    right after the singer word ("frontman Name"). '' when none."""
    r = str(reason or "")
    for m in _ROLE_RE.finditer(r):
        if _is_secondary_singer_role(r, m.start()):
            continue
        rest = r[m.end():]
        stop = _CLAUSE_END_RE.search(rest)
        clause = rest[:stop.start()] if stop else rest
        for k, nm in enumerate(_SINGER_NAME_RUN_RE.finditer(clause)):
            run = nm.group(0).strip()
            if run.endswith(".") and not re.search(r"\b[A-Z]\.$", run):
                run = run[:-1]
            if re.search(r"['’]s$", run):
                continue                    # "Band V's 1984 album": an owner, not the singer
            if len(run.split()) == 1:
                lead_in = clause[:nm.start()].strip(" ,:")
                if not (k == 0 and re.fullmatch(r"(?i)(?:(?:is|was|,)\s*)?(?:named\s+)?", lead_in)):
                    continue
            if _person_name_ok(run, act_names):
                return run
    return ""


_VOCAL_GROUP_RE = re.compile(
    r"(?i)\b(?:boy[- ]?band|girl[- ]?group|vocal group|harmony group|brothers|sisters|siblings|"
    r"brother (?:trio|duo|act|band|group)|sister (?:trio|duo|act|band|group)|family (?:band|group|act)|duo)\b")


def group_lead_identity(si, meta):
    """Musiclyse dev20 GROUP_LEAD_STAYS_SEVERAL (same rule and wording): an
    identity that says LEAD_SINGERS=one while its reasoning documents the act
    as a vocal group and names no single lead singer is set to several."""
    t = str(si or "")
    try:
        if not GROUP_LEAD_STAYS_SEVERAL or not t or "GROUP LEAD CHECK" in t:
            return si
        m = re.search(r"LEAD_SINGERS\s*[=:]\s*[\"'<\[]?\s*(one|several|uncertain|single|multiple|two)\b", t, re.I)
        if not m or m.group(1).lower() not in ("one", "single"):
            return si
        mr = re.search(r"(?m)^\W*REASONING\s*[=:]\s*(.*)$", t)
        reason = mr.group(1) if mr else ""
        if not _VOCAL_GROUP_RE.search(reason) or _SOLO_WORD_RE.search(reason):
            return si
        md = meta if isinstance(meta, dict) else {}
        artist = str(md.get("artist") or "").strip()
        acts = [x for x in (artist, re.split(r"(?i)\s+(?:feat(?:\.|\u2024)?|ft(?:\.|\u2024)|featuring|with)\s",
                                             artist)[0], str(md.get("title") or ""), str(md.get("album") or "")) if x]
        if singer_name_from_reason(reason, acts):
            return si
        for mm in re.finditer(r"(" + _NAME_TOKEN + r"(?:\s+" + _NAME_TOKEN + r"){1,3})\s+(?:sings|sang|singing|is on|on) "
                              r"lead\b", reason):
            if _person_name_ok(mm.group(1), acts):
                return si
        t2 = re.sub(r"(LEAD_SINGERS\s*[=:]\s*[\"'<\[]?\s*)(one|single)\b", r"\g<1>several", t, count=1, flags=re.I)
        add = (f" GROUP LEAD CHECK: the reasoning documents {artist or 'the act'} as a vocal group and names no "
               "single lead singer, so the lead is shared (a documented group with shared leads is enough), even "
               "though the audio passes heard one voice.")
        m = re.search(r"(?m)^(\W*REASONING\s*[=:].*)$", t2)
        return t2[:m.end()] + add + t2[m.end():] if m else t2.rstrip() + "\n" + add.strip()
    except Exception:
        return si


def identity_reason_from_block(text):
    """Read the complete REASONING field, including wrapped continuation lines.

    Singer-identity responses are mostly one line per field, but some saved
    replies wrap REASONING over multiple lines. Reading only the first line
    loses a named guest or co-vocalist that appears in the continuation.
    """
    t = str(text or "")
    lines = t.splitlines()
    for i, line in enumerate(lines):
        m = re.match(r"^\s*(?:[-*]\s*)?REASONING\s*[=:]\s*(.*)$", line, re.I)
        if not m:
            continue
        out = [m.group(1).strip()] if m.group(1).strip() else []
        for cont in lines[i + 1:]:
            if re.match(
                r"^\s*(?:[-*]\s*)?(?:SINGER_IDENTITY|REASONING|CONFIDENCE|LEAD_SINGERS)\s*[=:]",
                cont, re.I,
            ) or "[end of singer identity]" in cont.lower():
                break
            # Ignore the appended group guidance if it is present in a legacy
            # block, but otherwise preserve prose continuation lines.
            if cont.strip():
                out.append(cont.strip())
        return "\n".join(out).strip()
    # A few legacy or externally edited saves put multiple fields on one line.
    m = re.search(r"REASONING\s*[=:]\s*(.+)", t, re.I)
    return m.group(1).strip() if m else ""


def _explicit_vocal_credit_names(reason, act_names=()):
    """Names only when the reasoning explicitly assigns a guest/co-vocal role."""
    r = str(reason or "")
    # Name tokens are deliberately case-sensitive; only the role wording is
    # case-insensitive. Applying re.I to the entire expression let a name
    # capture consume lowercase prose (e.g. "Jane Doe shares the lead").
    nm = r"(?:The\s+)?(?:[A-Z][\w'’&!()\-]*|[A-Z]\.)(?:\s+(?:[A-Z][\w'’&!()\-]*|[A-Z]\.|de|van|von|la|del|der|da|di|du|dos|bin|al)){0,3}"
    patterns = (
        # Role first: "guest vocalist Jane Doe", "co-lead singer: Jane Doe".
        r"(?i:\b(?:guest|featured|co[- ]?lead|co[- ]?vocalist|additional|second)\s+"
        r"(?:lead\s+)?(?:singer|vocalist|voice|artist|rapper|performer)\s*"
        r"(?:(?:is|was|by|named|:|=)\s*)?)" + rf"({nm})",
        # Role named as a field: "guest vocalist: Jane Doe" / "co-vocalist is Jane Doe".
        r"(?i:\b(?:guest|featured|co[- ]?lead|co[- ]?vocalist)\s*"
        r"(?:singer|vocalist|voice|artist|rapper|performer)?\s*"
        r"(?:is|was|by|named|:|=)\s*)" + rf"({nm})",
        # Explicit vocal credit wording.
        r"(?i:\b(?:guest\s+)?(?:lead\s+)?vocals?\s*(?:by|from|featuring|feat(?:uring)?\.?|:|=)\s*)"
        + rf"({nm})",
        # An explicitly described duet partner (not merely a second pitch centre).
        r"(?i:\b(?:duet|duetting)\s+(?:with|by|alongside)\s*)" + rf"({nm})",
        r"(?i:\b(?:featuring|feat(?:uring)?\.?|ft\.?)\s*)" + rf"({nm})",
        # Name first: "Jane Doe is the guest vocalist / co-lead singer".
        rf"({nm})" + r"(?i:\s+(?:is|was)\s+(?:the\s+)?"
        r"(?:guest|featured|co[- ]?lead|co[- ]?vocalist)\s+(?:lead\s+)?"
        r"(?:singer|vocalist|voice|artist|rapper|performer))",
    )
    out, seen = [], set()
    for rx in patterns:
        for m in re.finditer(rx, r):
            candidate = m.group(1).strip(" \t\r\n,;:.()[]{}\"'“”")
            if not candidate:
                continue
            # A leading article is valid in artist names such as "The Example Act";
            # ignore it for plausibility checks, but keep it in the displayed name.
            check_name = candidate[4:].strip() if candidate.lower().startswith("the ") else candidate
            if not _person_name_ok(check_name, act_names):
                continue
            folded = _fold_name(candidate)
            if not folded or folded in seen:
                continue
            seen.add(folded)
            out.append(candidate)
    return out


def _main_lead_from_guest_clause(reason, act_names=()):
    """Recover the named primary lead when wording explicitly contrasts a guest with them."""
    r = str(reason or "")
    nm = r"(?:The\s+)?(?:[A-Z][\w'’&!()\-]*|[A-Z]\.)(?:\s+(?:[A-Z][\w'’&!()\-]*|[A-Z]\.|de|van|von|la|del|der|da|di|du|dos|bin|al)){0,3}"
    patterns = (
        r"(?i:\b(?:guest|featured|co[- ]?lead|co[- ]?vocalist)\s+"
        r"(?:lead\s+)?(?:singer|vocalist|voice|artist|rapper|performer)?\s*(?:is|was|:|=)?\s*)"
        + rf"(?P<guest>{nm})" + r".{0,120}?"
        + r"(?i:\b(?:shares?|shared|sharing|trades?|trading)\s+(?:the\s+)?(?:lead|vocals?|lead\s+vocals?)\s+(?:with|alongside)\s*)"
        + rf"(?P<main>{nm})",
        r"(?i:\b(?:guest|featured|co[- ]?lead|co[- ]?vocalist)\s+"
        r"(?:lead\s+)?(?:singer|vocalist|voice|artist|rapper|performer)?\s*(?:is|was|:|=)?\s*)"
        + rf"(?P<guest>{nm})" + r".{0,120}?"
        + r"(?i:\b(?:main|lead)\s+(?:singer|vocalist|voice)\s*(?:is|was|:|=)\s*)"
        + rf"(?P<main>{nm})",
        r"(?i:\b(?:guest|featured|co[- ]?lead|co[- ]?vocalist)\s+"
        r"(?:lead\s+)?(?:singer|vocalist|voice|artist|rapper|performer)?\s*(?:is|was|:|=)?\s*)"
        + rf"(?P<guest>{nm})" + r".{0,120}?" + rf"(?P<main>{nm})"
        + r"(?i:\s+(?:is|was)\s+(?:the\s+)?(?:main|lead)\s+(?:lead\s+)?"
        r"(?:singer|vocalist|voice))",
    )
    for rx in patterns:
        m = re.search(rx, r, re.S)
        if not m:
            continue
        guest, main = m.group("guest").strip(" ,;:.()[]{}\"'“”"), m.group("main").strip(" ,;:.()[]{}\"'“”")
        if not guest or not main or _fold_name(guest) == _fold_name(main):
            continue
        check_main = main[4:].strip() if main.lower().startswith("the ") else main
        if _person_name_ok(check_main, act_names):
            return main
    return ""


def identity_facts(reason, year="", act_names=()):
    """Plain facts the singer-identity step wrote in its reasoning: the
    singer's name when it says 'The credited solo artist is X', a birth year
    and an age at the time. Never used for how the recording sounds."""
    r = str(reason or "")
    out = {}
    name_res = (
        r"\b(?:credited (?:solo )?artist|lead singer|lead vocalist|singer) is "
        r"([A-Z][\w.'’-]*(?: [A-Z][\w.'’-]*){0,3})",
        # dev17 (MOSS-Music identity step): "The documented lead for this
        # recording is Jane Doe, an adult male at the time"
        r"\b(?:documented |credited )?lead(?: (?:singer|vocalist|vocal|voice))?"
        r" (?:for|on|of) (?:this|the) (?:recording|track|song|record) is "
        r"([A-Z][\w.'’-]*(?: [A-Z][\w.'’-]*){0,3})",
        # "Band X's documented lead vocalist John Doe is a male adult"
        r"\b(?:documented |credited )?(?:lead )?(?:vocalist|singer)\s+"
        r"([A-Z][\w.'’-]*(?: [A-Z][\w.'’-]*){0,3})\s+is\b",
        # 2.2.2: "documented lead vocalist John Doe for the 1974-1985 era",
        # "whose documented vocalist is John Doe", "with frontman John Doe"
        r"\b(?:documented |credited )?(?:lead )?(?:vocalist|singer|frontman|frontwoman)(?: is| was)?\s+"
        r"([A-Z][\w.'’-]*(?: (?:[A-Z][\w.'’-]*|de|van|von|la|del|der)){0,3})",
    )
    for rx in name_res:
        for m in re.finditer(rx, r):
            if _is_secondary_singer_role(r, m.start()):
                continue
            nm = m.group(1).strip()
            # sentence full stop, not an initial ("The Name." vs "N.A.M.", "Jr.")
            if nm.endswith(".") and "." not in nm.split()[-1][:-1] and len(nm.split()[-1]) > 4:
                nm = nm[:-1]
            if _plausible_name(nm):
                out["name"] = nm
                break
        if out.get("name"):
            break
    if not out.get("name"):
        paired_main = _main_lead_from_guest_clause(r, act_names)
        if paired_main:
            out["name"] = paired_main
    if SINGER_NAME_ANY_WORDING:
        # 2.2.3: a fixed phrasing that caught a band or the act ("Band V")
        # gives way to the general rule; with none, the general rule decides.
        if (out.get("name") and not _person_name_ok(out["name"], act_names)
                and any(_fold_name(a) and _fold_name(a) in _fold_name(out["name"]) for a in act_names)
                and _GROUP_WORD_RE.search(r) and not _SOLO_WORD_RE.search(r)):
            out.pop("name")
        if not out.get("name"):
            nm = singer_name_from_reason(r, act_names)
            if nm:
                out["name"] = nm
    # Featured artist named in the identity note (e.g. "a Singer Y feature is credited").
    feat = re.findall(
        r"\b(?:a |an )?([A-Z][\w.'’-]*(?: [A-Z][\w.'’-]*){0,3}) feature\b",
        r,
    )
    feat += re.findall(
        r"\b(?:feat(?:uring|\.)?|with|and)\s+([A-Z][\w.'’-]*(?: [A-Z][\w.'’-]*){0,3})\b"
        r"(?=\s+(?:is credited|feature|share|also sing))",
        r,
    )
    feat += _explicit_vocal_credit_names(r, act_names)
    # Deduplicate while preserving order; drop the primary name if already set.
    seen = set()
    features = []
    for n in feat:
        n = n.strip()
        if not n or n.lower() in seen:
            continue
        if out.get("name") and n.lower() == out["name"].lower():
            continue
        seen.add(n.lower())
        features.append(n)
    if features:
        out["features"] = features
    m = re.search(
        r"\bborn (?:in )?(?:(?:January|February|March|April|May|June|July|August|"
        r"September|October|November|December)\s+)?(1[89]\d\d|20[0-2]\d)\b",
        r, re.I,
    )
    if not m:
        # dev17 identity step: "Jane Doe (b. 1994)"
        m = re.search(r"\(b\.\s*(1[89]\d\d|20[0-2]\d)\)", r)
    if m:
        out["born"] = int(m.group(1))
    m = re.search(r"~?\b(\d{1,2})\s*(?:years? old|yo\b)", r)
    if not m:
        # "was ~15-16 at the 2010 release": the younger figure
        m = re.search(r"~?\b(\d{1,2})\s*[-–]\s*\d{1,2}\s+(?:years?\s+old|at\s+the|when)\b", r)
    if m:
        out["age"] = int(m.group(1))
    elif out.get("born") and re.match(r"^\d{4}$", str(year or "")):
        out["age"] = int(year) - out["born"]
    return out


# =============================================================================
# 2.3: Musiclyse dev44 genre view (style-led genre, era check, regional labels)
# =============================================================================
# The functions below are Musiclyse 0.5-moss-dev46's own (_genre_style_view,
# _world_label_view and what they use), copied unchanged apart from a "_mv"
# prefix on every name so they cannot collide with this script's older copies
# of similar helpers. They rewrite the saved analysis text exactly as the
# Musiclyse chat model is shown it, so the dossier names the same genre and
# era the program itself would. Saves without a DISCOGS STYLE SIGNAL block
# (made before dev42) only get the regional / World label check.

_ORIGIN_LIB = None          # the Wikipedia library, set by main() when the lookup is on
_MV_ORIGIN_CACHE = {}


def _mv_low_trust_genre_examples():
    """The listening model's stock genre example list (Musiclyse reads it from
    its own prompt; this is the same list)."""
    return _mv_LOW_TRUST_GENRE_EXAMPLES_FALLBACK


def _mv_artist_origin_lead(artist):
    """First sentence of the artist's own article in the Wikipedia database, or ""
    (lookup off, no database, no match). Only used to check whether a regional
    label (K-pop, Europop...) fits the artist; it adds nothing to the dossier and
    does not change which background articles are chosen. Never raises."""
    try:
        a = _clean_artist_tag(artist) if artist else ""
    except Exception:
        a = str(artist or "").strip()
    if not a or _ORIGIN_LIB is None:
        return ""
    if a in _MV_ORIGIN_CACHE:
        return _MV_ORIGIN_CACHE[a]
    lead = ""
    try:
        art = _find_artist(_ORIGIN_LIB, a, [])
        if art:
            _fields, body = parsed(art)
            b = re.sub(r"\s+", " ", str(body or ""))[:600]
            lead = b.split(". ")[0][:400]
    except Exception:
        lead = ""
    _MV_ORIGIN_CACHE[a] = lead
    return lead


_mv_RECOMMENDED_GENRE_RE = re.compile(r"RECOMMENDED GENRE FOR DISCUSSION: ([^\n]+?)\. Reasoning:")


_mv_STYLE_SIGNAL_HEARD_RE = re.compile(r"DISCOGS STYLE SIGNAL \([^\n]*\)\nstyles heard: ([^\n]+)")


def _mv_style_signal_heard(text):
    """dev43: the 'styles heard' list of the last DISCOGS STYLE SIGNAL block in
    `text`, or "" (no block, or no style was clear)."""
    found = list(_mv_STYLE_SIGNAL_HEARD_RE.finditer(str(text or "")))
    if not found:
        return ""
    heard = found[-1].group(1).strip()
    return "" if (not heard or heard.lower().startswith("none clear")) else heard


_mv_GENRE_FAMILY_ROCKISH = {
    "rock", "punk", "pop-punk", "pop punk", "emo", "metal", "heavy metal",
    "indie rock", "indie-rock", "alternative rock", "alt-rock", "alt rock",
    "hard rock", "garage rock", "post-punk", "post punk", "grunge",
    "power pop", "power-pop", "pop rock", "pop-rock", "ska",
}


_mv_GENRE_FAMILY_ELECTRONICISH = {
    "electronic", "electronica", "edm", "house", "techno", "trance",
    "dubstep", "drum and bass", "dnb", "dance", "dance-pop", "dance pop",
    "synth-pop", "synthpop", "synth pop", "electro", "electro-pop",
    "electropop", "club", "disco", "ambient", "electronic dance music",
}


_mv_GENRE_FAMILY_POPISH = {
    "pop", "indie pop", "indie-pop", "art pop", "dream pop", "chamber pop",
}


def _mv_normalize_genre_token(s):
    s = (s or "").strip().lower()
    # The listening model often writes compound genres with a non-breaking or other
    # Unicode hyphen ("dance\u2011pop"); fold them to "-" so they match the
    # family tables and read normally when quoted back to the user.
    s = re.sub(r"[\u2010-\u2015\u2212]", "-", s)
    s = re.sub(r"\s+", " ", s)
    s = s.replace("_", " ")
    return s


def _mv_strip_md_emphasis(text):
    """Remove markdown emphasis/code marks the model sometimes wraps field
    labels in ("**LEAD_PROFILE**=...", "`VOCALS_PRESENT`: yes").

    Every structured-field parser below anchors on LABEL followed directly by
    "=" or ":", so a bolded label silently failed to parse. For LEAD_PROFILE
    that resolved to "unknown", which (as parse_vocal_tags documents) switches
    off both age guards; for GENRE_RANKED it discarded the model's ranking.
    Asterisks and backticks never appear inside the values, so dropping them is
    safe; underscores are left alone because they are part of the labels.
    """
    if not text:
        return text or ""
    return re.sub(r"[*`]+", "", str(text))


def _mv_extract_genre_ranked_labels(text, max_n=8):
    """Pull ordered genre labels from a GENRE_RANKED=... field in listening-model text."""
    if not text:
        return []
    text = _mv_strip_md_emphasis(text)
    m = re.search(
        r"(?:GENRE_RANKED|Genre\s+Ranked)\s*[=:\-]\s*(.+?)(?=\n\s*(?:[A-Z][A-Z0-9_ ]{2,}=|GENRE_ADJACENT|GENRE_RULED|$))",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    if not m:
        # Fallback: first line containing GENRE_RANKED.
        #
        # The delimiter must follow the GENRE_RANKED token itself. Searching the
        # line for any "=", ":" or "-" matched punctuation in surrounding prose:
        # the PANNs explainer ("...noisy YouTube metadata -- they are supporting
        # evidence, not automatic overrides. When this signal ... and
        # GENRE_RANKED leads with rock/pop-punk...") sits on one line and
        # contains the token, so the old pattern captured from its "--" and made
        # "they are supporting evidence, not automatic overrides" the track's
        # genre.
        for line in (text or "").splitlines():
            if re.search(r"GENRE_RANKED|Genre\s+Ranked", line, re.I):
                m_line = re.search(
                    r"(?:GENRE_RANKED|Genre\s+Ranked)\s*[=:]\s*(.+)$", line, re.I
                )
                if m_line:
                    blob = m_line.group(1)
                    break
        else:
            return []
    else:
        blob = m.group(1)

    labels = []
    # Split on numbered ranks or semicolons
    parts = re.split(r"(?:;|\n|\d+\)\s*)", blob)
    for p in parts:
        p = p.strip()
        if not p:
            continue
        # Drop confidence parentheticals
        p = re.sub(r"\([^)]*confidence[^)]*\)", "", p, flags=re.I).strip()
        p = re.sub(r"^\d+[\).\]]\s*", "", p).strip(" ,;-")
        if not p or len(p) < 2:
            continue
        # Skip pure confidence words
        if p.lower() in ("high", "medium", "low", "confidence"):
            continue
        tok = _mv_normalize_genre_token(p)
        # Keep first few words of each rank item
        tok = re.split(r"[./|]", tok)[0].strip()
        if not _mv_is_plausible_genre_label(tok):
            continue
        if tok and tok not in labels:
            labels.append(tok)
        if len(labels) >= max_n:
            break
    return labels


_mv_GENRE_LABEL_REJECT_WORDS = (
    "evidence", "override", "overrides", "automatic", "supporting", "confidence",
    "unavailable", "uncertain", "unknown", "note", "reasoning", "instead",
    "should", "must", "do not", "cannot", "please", "output", "format",
    "listed", "above", "below", "n/a", "none",
)


def _mv_is_plausible_genre_label(tok):
    """Reject strings that cannot be a genre name.

    The transcription model sometimes echoes a fragment of its own instructions
    into GENRE_RANKED, and with no check on shape those fragments became the
    track's genre. One saved scan carried
    "GENRE_RANKED top is 'they are supporting evidence, not automatic
    overrides'", which then drove the RECOMMENDED GENRE block and left the
    writer with no usable measured genre at all — so it fell back to the
    artist's catalogue label from Wikipedia.

    Real genre labels are short noun phrases. Anything sentence-shaped, overly
    long, or containing instruction vocabulary is not one.
    """
    if not tok:
        return False
    t = str(tok).strip().lower()
    if len(t) < 2 or len(t) > 32:
        return False
    if len(t.split()) > 4:
        return False
    if "," in t or ";" in t:
        return False
    for bad in _mv_GENRE_LABEL_REJECT_WORDS:
        if bad in t:
            return False
    # Must contain letters, and not read as a sentence fragment.
    if not re.search(r"[a-z]", t):
        return False
    return True


def _mv_genre_family(label):
    t = _mv_normalize_genre_token(label)
    # Longer / more specific keys first via explicit checks
    for fam, members in (
        ("rockish", _mv_GENRE_FAMILY_ROCKISH),
        ("electronicish", _mv_GENRE_FAMILY_ELECTRONICISH),
        ("popish", _mv_GENRE_FAMILY_POPISH),
    ):
        if t in members or _mv_any_marker_present(t, members):
            return fam
        # token-wise
        for word in t.replace("-", " ").split():
            if word in members or any(word == mm.replace("-", " ") for mm in members):
                return fam
    # substring heuristics -- word-boundary anchored so short genre words
    # (e.g. "ska", "disco", "house", "dance") can't false-match inside an
    # unrelated longer word ("polska", "discography", "warehouse", "abundance").
    if _mv_any_marker_present(t, ("punk", "emo", "metal", "grunge", "rock")):
        return "rockish"
    if _mv_any_marker_present(t, ("house", "techno", "edm", "electro", "synth", "dance", "trance", "dubstep")):
        return "electronicish"
    if _mv_any_marker_present(t, ("pop",)):
        return "popish"
    return "other"


_mv_GENRE_ONSETS = (
    (r"indie[ -]?rock", 1983, "indie rock"), (r"indie[ -]?pop", 1983, "indie pop"),
    (r"indie[ -]?folk", 1995, "indie folk"), (r"jangle[ -]?pop", 1982, "jangle pop"),
    (r"alternative(?:[ -]rock)?|alt[ -]rock", 1982, "alternative rock"), (r"college[ -]rock", 1982, "college rock"),
    (r"shoegaze", 1988, "shoegaze"), (r"dream[ -]?pop", 1983, "dream pop"), (r"grunge", 1987, "grunge"),
    (r"britpop", 1993, "britpop"), (r"madchester", 1988, "Madchester"), (r"emo", 1985, "emo"),
    (r"post[ -]?punk", 1977, "post-punk"), (r"new[ -]wave", 1976, "new wave"),
    (r"synth[ -]?pop", 1977, "synth-pop"), (r"punk(?:[ -]rock)?", 1974, "punk"),
    (r"power[ -]?pop", 1972, "power pop"), (r"trip[ -]?hop", 1993, "trip hop"),
    (r"drum (?:and|&|n) bass|dnb", 1992, "drum and bass"), (r"dubstep", 2002, "dubstep"),
    (r"house", 1984, "house"), (r"techno", 1985, "techno"), (r"trance", 1990, "trance"),
    (r"nu[ -]metal", 1996, "nu metal"), (r"grime", 2002, "grime"),
)


def _mv_year_from_meta(meta):
    try:
        m = re.search(r"\b(19[0-9]{2}|20[0-9]{2})\b", str((meta or {}).get("year") or (meta or {}).get("date") or ""))
        return int(m.group(1)) if m else None
    except Exception:
        return None


def _mv_genre_onset(label):
    """The onset year of the newest dated label named in `label`, else None."""
    lab = str(label or "").lower()
    years = [y for pat, y, _n in _mv_GENRE_ONSETS if re.search(r"(?<![a-z])(?:" + pat + r")(?![a-z])", lab)]
    return max(years) if years else None


_mv_LOW_TRUST_GENRE_EXAMPLES_FALLBACK = ("indie rock", "indie pop", "power pop", "jangle pop")


def _mv_low_trust_norm(lab):
    return re.sub(r"\s+", " ", _mv_normalize_genre_token(lab).replace("-", " ")).strip()


def _mv_stock_genre_list(labels):
    """True when a genre list is MOSS-Music's stock answer: at least three of
    its first five labels are the prompt's own example descriptors."""
    ex = set(_mv_low_trust_genre_examples())
    return sum(1 for x in labels[:5] if _mv_low_trust_norm(x) in ex) >= 3


def _mv_panns_genre_tags_from_text(t):
    """[(tag, prob)] from the 'genre-adjacent tags:' line of a saved analysis."""
    m = re.search(r"genre-adjacent tags:\s*([^\n]+)", t or "")
    out = []
    if m:
        for part in re.split(r"(?<=%\)),\s*", m.group(1)):
            mm = re.match(r"\s*(.+?)\s*\((\d+(?:\.\d+)?)%\)\s*$", part)
            if mm:
                out.append((_mv_normalize_genre_token(mm.group(1)), float(mm.group(2)) / 100.0))
    return out


_mv_STYLE_LABELS = {
    "Pop Rock": "pop rock", "Goth Rock": "gothic rock", "Brit Pop": "Britpop", "Hi NRG": "hi-NRG",
    "Prog Rock": "progressive rock", "Thrash": "thrash metal", "RnB/Swing": "R&B",
    "Contemporary R&B": "contemporary R&B", "Rhythm & Blues": "rhythm and blues", "Gangsta": "gangsta rap",
    "Post Bop": "post-bop", "Bubblegum": "bubblegum pop", "Euro House": "Euro house", "Pop Punk": "pop punk",
    "Punk": "punk rock", "Hardcore": "hardcore punk", "Rock & Roll": "rock and roll", "Conscious": "conscious hip hop",
    "Thug Rap": "thug rap", "Drum n Bass": "drum and bass", "Soul-Jazz": "soul jazz", "Bop": "bebop",
    "Europop": "Europop", "K-pop": "K-pop", "J-pop": "J-pop", "AOR": "AOR", "UK Garage": "UK garage",
    "MPB": "MPB", "Latin": "Latin", "Euro-Disco": "Euro disco", "Italo-Disco": "Italo disco",
    "Eurodance": "Eurodance", "Eurobeat": "Eurobeat", "Italo House": "Italo house", "Italodance": "Italodance",
    "Hip-House": "hip house", "Nu-Disco": "nu-disco", "Lo-Fi": "lo-fi", "Glam": "glam rock",
    "Acoustic": None, "Vocal": None, "Ballad": None, "Parody": None, "Novelty": None, "Instrumental": None,
    "DJ Battle Tool": None, "Cut-up/DJ": None, "Light Music": None, "Experimental": None, "Avantgarde": None,
    "Contemporary": None, "Modern": None, "Abstract": None, "Religious": None, "Political": None,
}


_mv_STYLE_NOT_GENRE_PARENTS = ("Non-Music", "Stage & Screen", "Children's")


_mv_STYLE_DISPLAY_PARENT = {"Folk, World, & Country": "Folk / Country"}


_mv_STYLE_FAMILY_KEYS = (
    ("metal", ("metal", "thrash")),
    ("punk", ("punk", "hardcore")),
    ("rock", ("rock", "grunge", "power pop", "britpop", "jangle", "paisley", "krautrock", "lo-fi", "shoegaze",
              "new wave", "post-punk", "garage", "indie", "alternative", "psychedelia", "boogie", "aor")),
    ("pop", ("pop", "teen", "bubblegum", "europop", "schlager", "vocal", "ballad", "sophisti")),
    ("electronic", ("house", "techno", "trance", "electro", "synth", "edm", "disco", "hi-nrg", "freestyle", "dance",
                    "downtempo", "ambient", "electronica", "chill", "trip hop", "breakbeat", "eurodance", "big beat",
                    "drum", "dubstep", "hyperpop", "idm", "eurobeat")),
    ("hiphop", ("hip hop", "hip-hop", "rap", "trap", "crunk", "gangsta", "bass music", "hip house")),
    ("rnb", ("r&b", "soul", "funk", "swing", "new jack", "motown")),
    ("jazz", ("jazz", "bop", "swing", "fusion")),
    ("folk", ("folk", "country", "americana", "acoustic", "singer-songwriter", "bluegrass", "cowpunk")),
    ("blues", ("blues",)),
)


_mv_STYLE_LABEL_SYNONYMS = {
    "goth rock": "gothic rock", "brit pop": "britpop", "hi nrg": "hi-nrg", "prog rock": "progressive rock",
    "pop-rock": "pop rock", "pop-punk": "pop punk", "hip-hop": "hip hop", "synthpop": "synth-pop",
    "synth pop": "synth-pop", "dance pop": "dance-pop", "electronic pop": "electropop", "electronic-pop": "electropop",
    "electro-pop": "electropop", "indie-pop": "indie pop", "indie-rock": "indie rock", "r and b": "r&b",
    "rnb": "r&b", "punk": "punk rock",
}


_mv_STYLE_TEMPLATE_LABELS = {"indie pop", "indie rock", "power pop", "jangle pop", "electropop", "synth-pop",
                          "pop punk", "alternative rock"}


_mv_STYLE_BROAD_LABELS = {"rock", "pop", "electronic", "dance", "hip hop", "rap", "r&b", "soul", "jazz", "folk",
                       "country", "metal", "punk rock", "blues", "funk", "indie", "alternative",
                       "electronic / dance-pop", "pop and r&b fusion"}


_mv_STYLE_NO_REFINE_FAMILIES = {"hiphop"}     # Discogs' hip hop sub-styles are not reliable as a main label


_mv_STYLE_PERIOD = {
    "pop rock": None, "house": (1986, 2020), "alternative rock": (1987, 2010), "europop": (1985, 2005),
    "hard rock": (1968, 1995), "dance-pop": (1983, 2020), "synth-pop": (1979, 1990), "indie rock": (1985, 2015),
    "power pop": (1972, 1990), "r&b": (1988, 2010), "progressive rock": (1968, 1982), "electro": (1982, 2012),
    "pop rap": (1988, 2018), "electro house": (2005, 2015), "contemporary r&b": (1990, 2015),
    "country rock": (1968, 1982), "classic rock": (1965, 1982), "pop punk": (1994, 2010),
    "psychedelic rock": (1965, 1972), "new wave": (1977, 1987), "euro house": (1990, 1999), "soft rock": (1970, 1985),
    "punk rock": (1976, 1990), "folk rock": (1964, 1978), "aor": (1975, 1990), "disco": (1974, 1983),
    "bubblegum pop": (1995, 2005), "k-pop": (2008, 2024), "country": None, "blues rock": (1966, 1978),
    "arena rock": (1975, 1990), "fusion": (1970, 1985), "heavy metal": (1975, 1992), "britpop": (1993, 1999),
    "rock and roll": (1954, 1965), "pub rock": (1972, 1980), "trance": (1992, 2008), "jazz-funk": (1972, 1983),
    "soul": (1960, 1978), "downtempo": (1994, 2008), "reggaeton": (2004, 2024), "folk": None, "crunk": (2000, 2009),
    "soul jazz": (1958, 1972), "space rock": (1969, 1978), "ambient": None, "grunge": (1989, 1997),
    "hands up": (2002, 2012), "eurodance": (1990, 1998), "gothic rock": (1980, 1992), "post-bop": (1960, 1995),
    "contemporary jazz": (1980, 2010), "bebop": (1945, 1960), "swing": (1935, 1950), "jazz-rock": (1969, 1980),
    "industrial": (1980, 1998), "latin": None, "art rock": (1969, 1982), "garage rock": (1963, 1970),
    "dream pop": (1985, 1997), "hip hop": (1985, 2015), "krautrock": (1969, 1978), "gangsta rap": (1988, 2005),
    "trap": (2005, 2024), "conscious hip hop": (1987, 2005), "deathrock": (1980, 1988), "hard bop": (1954, 1966),
    "cool jazz": (1949, 1960), "boogie": (1979, 1986), "funk": (1966, 1981), "freestyle": (1983, 1992),
    "hi-nrg": (1977, 1990), "garage house": (1986, 1995), "acid rock": (1966, 1971), "breakbeat": (1990, 2005),
    "trip hop": (1993, 2001), "electric blues": (1950, 1975), "tropical house": (2013, 2019),
    "uk garage": (1996, 2003), "techno": (1988, 2012), "indie pop": (1986, 2015), "nu metal": (1996, 2005),
    "synthwave": (2008, 2024), "melodic hardcore": (1994, 2010), "progressive house": (1992, 2012),
    "drum and bass": (1993, 2010), "thug rap": (1993, 2005), "miami bass": (1985, 1996),
    "thrash metal": (1983, 1993), "speed metal": (1982, 1990), "death metal": (1987, 1997),
    "rhythm and blues": (1945, 1965), "electroclash": (2000, 2006), "post-punk": (1978, 1985),
    "italo disco": (1979, 1988), "euro disco": (1975, 1985), "glam rock": (1971, 1976), "beat": (1962, 1967),
    "doo wop": (1952, 1963), "surf": (1960, 1966), "rockabilly": (1954, 1960), "twist": (1959, 1963),
    "mod": (1963, 1968), "shoegaze": (1988, 1995), "emo": (1995, 2010), "post-hardcore": (1995, 2010),
    "metalcore": (2000, 2015), "new jack swing": (1987, 1994), "neo soul": (1995, 2008), "g-funk": (1991, 1998),
    "boom bap": (1987, 1999), "cloud rap": (2011, 2024), "grime": (2003, 2024), "dubstep": (2006, 2014),
    "chillwave": (2009, 2013), "vaporwave": (2011, 2024), "big beat": (1995, 2001), "jungle": (1992, 1997),
    "happy hardcore": (1993, 2000), "eurobeat": (1985, 2000), "nu-disco": (2005, 2020), "acid house": (1987, 1991),
    "new beat": (1987, 1990), "ebm": (1982, 1995), "darkwave": (1982, 1995), "coldwave": (1979, 1986),
    "reggae": (1968, 1990), "dancehall": (1980, 2015), "big band": (1935, 1950), "dixieland": (1915, 1935),
    "ragtime": (1895, 1918), "bossa nova": (1958, 1968), "symphonic rock": (1968, 1980),
    "southern rock": (1970, 1980), "stoner rock": (1992, 2010), "math rock": (1990, 2010), "post rock": (1994, 2010),
    "lo-fi": (1985, 2000), "no wave": (1977, 1982), "oi": (1979, 1985), "psychobilly": (1980, 1990),
    "yé-yé": (1962, 1968), "smooth jazz": (1985, 2005), "easy listening": (1955, 1975), "space-age": (1955, 1965),
    "lounge": (1955, 1968),
}


_mv_REGIONAL_STYLES = {
    "k-pop": "korea", "j-pop": "japan", "kayōkyoku": "japan", "city pop": "japan", "bollywood": "india",
    "hindustani": "india", "indian classical": "india", "carnatic": "india", "europop": "europe", "euro house": "europe",
    "euro disco": "europe", "eurobeat": "europe", "eurodance": "europe", "italo house": "italy",
    "italo disco": "italy", "italodance": "italy", "disco polo": "poland", "schlager": "germanic",
    "volksmusik": "germanic", "chanson": "france", "yé-yé": "france", "canzone napoletana": "italy",
    "catalan music": "spain", "laïkó": "greece", "éntekhno": "greece", "nordic": "nordic", "celtic": "celtic",
    "britpop": "uk", "britcore": "uk", "uk street soul": "uk", "krautrock": "germany", "berlin-school": "germany",
    "mpb": "brazil", "latin": "latin", "african": "africa", "afrobeat": "africa", "pacific": "pacific",
    "afro-cuban": "cuba", "fado": "portugal", "raï": "maghreb", "séga": "mauritius", "highlife": "westafrica",
    "soukous": "centralafrica", "zouk": "antilles", "compas": "haiti", "tejano": "mexico", "norteño": "mexico",
    "mariachi": "mexico", "ranchera": "mexico", "vallenato": "colombia", "porro": "colombia",
    "nueva cancion": "latin", "romani": "romani",
}


_mv_REGION_LANGS = {
    "korea": {"ko"}, "japan": {"ja"}, "india": {"hi"}, "italy": {"it"}, "poland": {"pl"},
    # Spanish, Portuguese and French are left out: most of their singers are not European.
    "europe": {"sv", "de", "it", "nl", "da", "no", "fi", "pl", "el", "ca"},
    "germanic": {"de", "nl"}, "france": {"fr"}, "spain": {"es", "ca"}, "greece": {"el"},
    "nordic": {"sv", "no", "da", "fi"}, "germany": {"de"}, "brazil": {"pt"}, "latin": {"es", "pt"},
    "portugal": {"pt"}, "mexico": {"es"}, "colombia": {"es"}, "cuba": {"es"},
}


_mv_EUROPEAN = ("British|English|Scottish|Welsh|Irish|Northern Irish|French|German|Austrian|Swiss|Dutch|Belgian|"
             "Luxembourgish|Swedish|Norwegian|Danish|Finnish|Icelandic|Italian|Spanish|Portuguese|Greek|Cypriot|"
             "Maltese|Polish|Czech|Slovak|Hungarian|Romanian|Bulgarian|Serbian|Croatian|Slovenian|Bosnian|Albanian|"
             "Macedonian|Montenegrin|Estonian|Latvian|Lithuanian|Ukrainian|Belarusian|Russian|Moldovan|Catalan|Basque|"
             "Galician|Flemish|Faroese|Swedish-American")


_mv_LATIN_AM = ("Mexican|Puerto Rican|Cuban|Dominican|Colombian|Venezuelan|Argentine|Argentinian|Chilean|Peruvian|"
             "Ecuadorian|Uruguayan|Paraguayan|Bolivian|Guatemalan|Honduran|Salvadoran|Nicaraguan|Costa Rican|"
             "Panamanian|Brazilian|Spanish|Portuguese|Latin American|Latino|Latina")


_mv_REGION_ORIGIN = {
    "korea": "South Korean|Korean", "japan": "Japanese", "india": "Indian", "europe": _mv_EUROPEAN,
    "italy": "Italian", "poland": "Polish", "germanic": "German|Austrian|Swiss|Dutch|Flemish|Belgian",
    "france": "French|Belgian|Swiss|Québécois", "spain": "Spanish|Catalan", "greece": "Greek|Cypriot",
    "nordic": "Swedish|Norwegian|Danish|Finnish|Icelandic|Faroese", "celtic": "Irish|Scottish|Welsh|Breton|Cornish|Manx",
    "uk": "British|English|Scottish|Welsh|Northern Irish", "germany": "German", "brazil": "Brazilian",
    "latin": _mv_LATIN_AM, "africa": ("Nigerian|Ghanaian|Senegalese|Malian|South African|Kenyan|Congolese|Cameroonian|"
                                   "Ivorian|Ethiopian|Tanzanian|Ugandan|Zimbabwean|Angolan|Mozambican|Beninese|Guinean|"
                                   "Burkinabé|Togolese|Sierra Leonean|Liberian|Gambian|Algerian|Moroccan|Tunisian|"
                                   "Egyptian|Sudanese|Somali|Rwandan|Zambian|Malawian|Namibian|Botswanan|African"),
    "pacific": "Hawaiian|Samoan|Tongan|Fijian|Māori|Maori|Polynesian|Papuan|Tahitian", "cuba": "Cuban",
    "portugal": "Portuguese", "maghreb": "Algerian|Moroccan|Tunisian", "mauritius": "Mauritian|Réunionnais",
    "westafrica": "Ghanaian|Nigerian", "centralafrica": "Congolese", "antilles": "Martinican|Guadeloupean|Antillean",
    "haiti": "Haitian", "mexico": "Mexican|Mexican-American|Chicano|Tejano", "colombia": "Colombian",
    "romani": "Romani|Roma",
}


_mv_WORLD_LABELS = {"world", "world music", "international", "international music", "worldbeat", "world beat",
                 "global", "global music", "ethnic", "ethnic music", "asian", "asian (unspecified)",
                 "music of asia", "african", "music of africa", "middle eastern", "middle eastern music",
                 "traditional/folk (unspecified)", "folk, world, & country", "world fusion"}


_mv_REGIONAL_TAGS = {"latin": "latin", "music of latin america": "latin", "bollywood": "india",
                  "music of bollywood": "india", "carnatic": "india", "carnatic music": "india",
                  "afrobeat": "africa", "k-pop": "korea", "kpop": "korea", "j-pop": "japan", "jpop": "japan",
                  "c-pop": "china", "cantopop": "china", "mandopop": "china", "europop": "europe",
                  "eurodance": "europe", "euro house": "europe", "euro disco": "europe", "eurobeat": "europe",
                  "britpop": "uk", "brit pop": "uk", "krautrock": "germany", "schlager": "germanic",
                  "chanson": "france", "italo disco": "italy", "italo-disco": "italy", "j-rock": "japan",
                  "anime": "japan", "reggaeton": "latin", "latin pop": "latin"}


_mv_REGION_LANGS["china"] = {"zh"}


_mv_REGION_ORIGIN["china"] = "Chinese|Taiwanese|Hong Kong|Hongkongese|Singaporean|Malaysian Chinese"


_mv_LANG_STOPWORDS = {
    "en": "the and you i to a it in my me of is that on your be for we all love with this so but no just what can oh don't i'm it's",
    "es": "el la de que y en un una los las por con para mi tu te me es lo se del al como más pero yo eres quiero amor",
    "pt": "o a de que e do da em um uma os as para com não meu minha você eu se na no mais por amor",
    "fr": "le la les de et un une je tu il elle est pas que qui dans pour sur avec mon ma moi toi nous vous",
    "de": "der die das und ich du ist nicht ein eine zu mit mich dich mir dir es wir sie auf für sind",
    "it": "il la di che e un una non per sono mi ti con io tu è del della le gli",
    "nl": "de het een en ik je jij niet van is dat op zijn met voor mijn",
    "sv": "och jag du det att är en inte på med som för mig dig vi han hon",
    "da": "og jeg du det at er en ikke på med som for mig dig vi han hun",
    "no": "og jeg du det å er en ikke på med som for meg deg vi",
    "fi": "ja minä sinä se on ei että en olen mutta kun",
    "pl": "i w nie się na że to jest ja ty mnie cię z do",
    "tr": "ve bir bu ben sen ne çok için",
    "ca": "i el la de que en un una els les per amb no em",
}


_mv_LANG_STOPSETS = {k: set(v.split()) for k, v in _mv_LANG_STOPWORDS.items()}


_mv_SCRIPT_LANGS = (("ko", r"[가-힯ᄀ-ᇿ]"), ("ja", r"[぀-ヿ]"), ("zh", r"[一-鿿]"),
                 ("ru", r"[Ѐ-ӿ]"), ("el", r"[Ͱ-Ͽ]"), ("ar", r"[؀-ۿ]"),
                 ("he", r"[֐-׿]"), ("hi", r"[ऀ-ॿ]"), ("th", r"[฀-๿]"))


_mv_STYLE_ITEM_RE = re.compile(r"\s*(.+?) \(([^()]+)\) (\d+)%\s*$")


_mv_STYLE_LED_TAG = "(style-led)"


_mv_ERA_CHECK_HEAD = "ERA CHECK:"


def _mv_style_norm(label):
    x = re.sub(r"\(.*?\)", "", str(label or "").lower()).strip()
    x = re.sub(r"\s+", " ", x)
    return _mv_STYLE_LABEL_SYNONYMS.get(x, x)


def _mv_style_families(label):
    t = _mv_style_norm(label)
    return {f for f, keys in _mv_STYLE_FAMILY_KEYS if any(k in t for k in keys)}


def _mv_style_label(style, parent=""):
    """Conversation label for a Discogs style, or None."""
    if parent in _mv_STYLE_NOT_GENRE_PARENTS:
        return None
    if style in _mv_STYLE_LABELS:
        return _mv_STYLE_LABELS[style]
    return style.lower()


def _mv_style_signal_items(text):
    """[(label, prob)] from the last DISCOGS STYLE SIGNAL 'styles heard' line, same-named
    styles merged (strongest kept), non-genre styles dropped. [] when absent or none clear."""
    heard = _mv_style_signal_heard(text)
    if not heard:
        return []
    out = {}
    for part in heard.split(";"):
        m = _mv_STYLE_ITEM_RE.match(part)
        if not m:
            continue
        lab = _mv_style_label(m.group(1).strip(), m.group(2).strip())
        if lab is None:
            continue
        p = int(m.group(3)) / 100.0
        if p > out.get(lab, (0.0, 0))[0]:
            out[lab] = (p, len(out))
    return [(l, v[0]) for l, v in sorted(out.items(), key=lambda kv: (-kv[1][0], kv[1][1]))]


def _mv_lyrics_languages(text):
    """{language code: share} for a lyric text (script-based for non-Latin
    scripts, common-word counts for Latin-script languages); {} when too short."""
    t = str(text or "")
    letters = len(re.findall(r"[^\W\d_]", t))
    if letters < 12:
        return {}
    shares = {}
    for code, rx in _mv_SCRIPT_LANGS:
        n = len(re.findall(rx, t))
        if letters and n / letters >= 0.05:
            shares[code] = n / letters
    if shares.get("ja") and shares.get("zh"):        # kanji inside Japanese
        shares["ja"] += shares.pop("zh")
    latin = 1.0 - min(1.0, sum(shares.values()))
    words = re.findall(r"[a-zà-ÿßæøåœçñ']+", t.lower())
    if words and latin > 0.05:
        hits = {k: sum(1 for w in words if w in s) for k, s in _mv_LANG_STOPSETS.items()}
        tot = sum(hits.values())
        if tot >= 8:
            for k, h in hits.items():
                if h / tot >= 0.2:
                    shares[k] = shares.get(k, 0.0) + latin * h / tot
    return shares


def _mv_lyrics_text_for_genre(text, meta):
    lyr = str((meta or {}).get("lyrics") or "").strip()
    if not lyr:
        m = re.search(r"LYRICS HEARD IN AUDIO[^\n]*\n([\s\S]{0,6000}?)(?:\n\n[A-Z][A-Z /&\-]{6,}|\Z)", str(text or ""))
        lyr = m.group(1) if m else ""
    title = str((meta or {}).get("title") or "")
    return (lyr + "\n" + title).strip()


def _mv_origin_is(region, lead):
    """True when the article's first sentence calls the artist one of `region`'s nationalities."""
    rx = _mv_REGION_ORIGIN.get(region)
    if not rx or not lead:
        return False
    return bool(re.search(r"\b(?:is|was|are|were)\s+(?:an?|the)\s+(?:[\w\-]+\s+){0,2}?(?:" + rx + r")\b", lead))


class _mv_GenreContext:
    """What the regional check needs about one track: lyric languages and artist origin (lazy)."""

    def __init__(self, text, meta):
        self.meta = meta or {}
        self.langs = _mv_lyrics_languages(_mv_lyrics_text_for_genre(text, self.meta))
        # A title or album written in a non-Latin script ("Gangnam Style (강남스타일)")
        # counts as that language even when the lyrics are romanised.
        names = " ".join(str(self.meta.get(k) or "") for k in ("title", "album"))
        for code, rx in _mv_SCRIPT_LANGS:
            if len(re.findall(rx, names)) >= 3:
                self.langs[code] = max(self.langs.get(code, 0.0), 0.5)
        if self.langs.get("ja") and self.langs.get("zh") and re.search(r"[\u3040-\u30ff]", names):
            self.langs["ja"] = max(self.langs["ja"], self.langs.pop("zh"))
        self._lead = None

    def lead(self):
        if self._lead is None:
            self._lead = _mv_artist_origin_lead(self.meta.get("artist")) if REGIONAL_GENRE_GATE else ""
        return self._lead

    def non_english(self):
        if not self.langs:
            return False
        top = max(self.langs.items(), key=lambda kv: kv[1])[0]
        return top != "en"

    def region_ok(self, region):
        if not region:
            return True
        if any(self.langs.get(l, 0.0) >= 0.25 for l in _mv_REGION_LANGS.get(region, ())):
            return True
        return _mv_origin_is(region, self.lead())


def _mv_regional_label_ok(label, ctx):
    """False for a World/International-type label, or a regional one this track does not qualify for."""
    if not REGIONAL_GENRE_GATE:
        return True
    n = _mv_style_norm(label)
    if n in _mv_WORLD_LABELS or re.search(r"\bworld\b|\binternational\b", n):
        return False
    region = _mv_REGIONAL_STYLES.get(n) or _mv_REGIONAL_TAGS.get(n)
    if not region:
        return True
    return ctx.region_ok(region) if ctx is not None else False


def _mv_style_led_decision(rec, items, ranked_labels, panns, year, stock=False):
    """(new label, why) when the measured styles should lead the genre, else None."""
    if not items:
        return None
    l1, p1 = items[0]
    ft, fc = _mv_style_families(l1), _mv_style_families(rec)
    if not ft:
        return None
    onset = _mv_genre_onset(l1)
    if year and onset and year < onset:
        return None                                      # a later label than the recording
    base = re.sub(r"\s*\(with .*?\)", "", str(rec)).strip()
    if ft & fc:
        broad = ("(with " in str(rec).lower() or _mv_style_norm(rec) in {_mv_style_norm(b) for b in _mv_STYLE_BROAD_LABELS}
                 # a stock-list label is shown as its broad family (MOSS_LOW_TRUST_FIELDS)
                 or (stock and _mv_style_norm(base) in _mv_STYLE_TEMPLATE_LABELS))
        if broad and p1 >= GENRE_STYLE_REFINE_MIN and not (ft & _mv_STYLE_NO_REFINE_FAMILIES):
            return l1, "refine"
        return None
    if p1 < GENRE_STYLE_REPLACE_MIN:
        return None
    if not any(_mv_style_families(x) & ft for x in list(ranked_labels) + [t for t, p in panns if p >= 0.15]):
        return None                                      # no second source hears that family
    partial = any(_mv_style_families(l) & fc for l, _p in items[1:3])
    if partial and _mv_style_norm(base) not in _mv_STYLE_TEMPLATE_LABELS:
        return None
    return l1, "replace"


def _mv_style_led_genre_block(new, p, old, why):
    if why == "refine":
        how = f"it narrows the earlier '{old}' reading to the specific style"
    elif why == "regional":
        how = f"the earlier '{old}' label is a regional or World label this track does not qualify for"
    else:
        how = (f"it replaces the earlier '{old}' reading, which another measured source contradicts")
    return (f"RECOMMENDED GENRE FOR DISCUSSION: {new}. Reasoning: {_mv_STYLE_LED_TAG} the recording's clearest "
            f"measured style is {new} ({p * 100:.0f}%); {how}. This is the primary genre framing for the user. "
            "Do not lead with a conflicting early GENRE_RANKED label when this block "
            "explicitly revises rock/punk vs electronic/dance identity.")


def _mv_era_style_decision(era_text, items):
    """(label, prob, (start, end)) when a confident period-bound style contradicts the era estimate."""
    ds = [int(d) for d in re.findall(r"(\d{4})s", str(era_text or ""))]
    if not ds:
        return None
    lo, hi = min(ds), max(ds) + 9
    if hi < 2019:
        return None          # only the known failure: clean masters dated "2010s-2020s"
    for lab, p in items[:3]:
        per = _mv_STYLE_PERIOD.get(_mv_style_norm(lab))
        if per and per[1] - per[0] + 1 <= ERA_STYLE_MAX_SPAN and p >= ERA_STYLE_MIN:
            return (lab, p, per) if (per[1] < lo or per[0] > hi) else None
    return None


def _mv_genre_style_view(text, meta=None, year=None):
    """dev44 first step of the chat model's view: regional styles gated, the
    style-led recommended genre, and the style era check. Unchanged text when
    there is nothing to do."""
    t = str(text or "")
    has_styles = bool(_mv_STYLE_SIGNAL_HEARD_RE.search(t))
    recs = list(_mv_RECOMMENDED_GENRE_RE.finditer(t))
    if not has_styles and not (REGIONAL_GENRE_GATE and recs):
        return t
    ctx = _mv_GenreContext(t, meta) if REGIONAL_GENRE_GATE else None
    # 1. regional / World styles out of the styles line
    if has_styles and REGIONAL_GENRE_GATE:
        found = list(_mv_STYLE_SIGNAL_HEARD_RE.finditer(t))
        m = found[-1]
        heard = m.group(1)
        if not heard.lower().startswith("none clear"):
            keep = []
            for part in heard.split(";"):
                mm = _mv_STYLE_ITEM_RE.match(part)
                lab = _mv_style_label(mm.group(1).strip(), mm.group(2).strip()) if mm else None
                if mm and lab is not None and not _mv_regional_label_ok(lab, ctx):
                    continue
                keep.append(part.strip())
            new_heard = "; ".join(keep) if keep else "none clear"
            if new_heard != heard:
                t = t[:m.start(1)] + new_heard + t[m.end(1):]
    # 2. the recommended genre
    recs = list(_mv_RECOMMENDED_GENRE_RE.finditer(t))
    if recs and _mv_STYLE_LED_TAG not in t:
        rec = recs[-1].group(1).strip()
        items = _mv_style_signal_items(t) if has_styles else []
        decision = None
        if GENRE_STYLE_LED and items:
            try:
                ranked = _mv_extract_genre_ranked_labels(t, max_n=8)
            except Exception:
                ranked = []
            try:
                stock = bool(MOSS_LOW_TRUST_FIELDS and _mv_stock_genre_list(ranked))
            except Exception:
                stock = False
            decision = _mv_style_led_decision(rec, items, ranked, _mv_panns_genre_tags_from_text(t), year, stock=stock)
            if decision and not _mv_regional_label_ok(decision[0], ctx):
                decision = None
        if decision is None and not _mv_regional_label_ok(rec, ctx):
            alt = [(l, p) for l, p in items if _mv_regional_label_ok(l, ctx)]
            if alt:
                decision = (alt[0][0], "regional")
            else:
                fam = _mv_genre_family(rec)
                decision = ({"rockish": "rock", "electronicish": "electronic", "popish": "pop"}.get(fam, "pop"),
                            "regional")
        if decision:
            new, why = decision
            p = dict(items).get(new, 0.0)
            blk = re.compile(r"RECOMMENDED GENRE FOR DISCUSSION: " + re.escape(rec)
                             + r"\. Reasoning:[^\n]*")
            ms = list(blk.finditer(t))
            if ms:
                mm = ms[-1]
                t = t[:mm.start()] + _mv_style_led_genre_block(new, p, rec, why) + t[mm.end():]
    # 3. the era estimate
    if ERA_STYLE_CHECK and has_styles and _mv_ERA_CHECK_HEAD not in t:
        ms = list(re.finditer(r"(?m)^ERA_ESTIMATE=([^\n]*)$", t))
        items = _mv_style_signal_items(t)
        if ms and items:
            m = ms[-1]
            d = _mv_era_style_decision(m.group(1), items)
            if d:
                lab, p, per = d
                d0, d1 = per[0] // 10 * 10, per[1] // 10 * 10
                new = f"{d0}s" if d0 == d1 else f"{d0}s–{d1}s"
                note = (f"\n{_mv_ERA_CHECK_HEAD} the era pass said '{m.group(1).strip()}', which reflects clean, "
                        f"modern-sounding production rather than the style: the clearest period-bound style heard, "
                        f"{lab} ({p * 100:.0f}%), belongs to about {per[0]}–{per[1]}, so the recording most "
                        f"likely dates from the {new}. A file's year tag or release information still comes first.")
                t = t[:m.start()] + f"ERA_ESTIMATE={new}" + note + t[m.end():]
    return t


def _mv_world_label_view(text, meta=None):
    """dev44 last step of the chat model's view: World/International-type
    labels, and regional labels this track does not qualify for, are taken out
    of the PANNs tag line and the listening model's genre lists."""
    t = str(text or "")
    if not REGIONAL_GENRE_GATE:
        return t
    ctx = None

    def ok(lab):
        nonlocal ctx
        n = _mv_style_norm(lab)
        if not (n in _mv_WORLD_LABELS or _mv_REGIONAL_STYLES.get(n) or _mv_REGIONAL_TAGS.get(n)
                or re.search(r"\bworld\b|\binternational\b", n)):
            return True
        if ctx is None:
            ctx = _mv_GenreContext(t, meta)
        return _mv_regional_label_ok(lab, ctx)

    def tag_line(m):
        parts = re.split(r"(?<=%\)),\s*", m.group(2))
        keep = [p for p in parts if not re.match(r"\s*(.+?)\s*\(\d+(?:\.\d+)?%\)\s*$", p)
                or ok(re.match(r"\s*(.+?)\s*\(\d+(?:\.\d+)?%\)\s*$", p).group(1))]
        return m.group(1) + (", ".join(keep) if keep else "none")
    t = re.sub(r"(genre-adjacent tags:\s*)([^\n]+)", tag_line, t)

    def ranked_line(m):
        items = re.split(r";\s*(?=\d+\)\s*)", m.group(2))
        keep = []
        for it in items:
            mm = re.match(r"\s*\d+\)\s*([^(;]+)", it)
            if mm and not ok(mm.group(1).strip()):
                continue
            keep.append(re.sub(r"^\s*\d+\)\s*", "", it))
        if len(keep) == len(items):
            return m.group(0)
        return m.group(1) + "; ".join(f"{i}) {x}" for i, x in enumerate(keep, 1))
    t = re.sub(r"(GENRE_RANKED=)([^\n]+)", ranked_line, t)

    def list_line(m):
        items = [x for x in m.group(2).split(";")]
        keep = [x for x in items if ok(x.strip()) or not x.strip()]
        return m.group(0) if len(keep) == len(items) else m.group(1) + ";".join(keep).strip()
    t = re.sub(r"(GENRE_ADJACENT=)([^\n]+)", list_line, t)
    return t


_mv_CP_QUALITY = {"maj": "", "min": "m", "7": "7", "maj7": "maj7", "min7": "m7", "dim": "dim", "aug": "aug",
               "sus2": "sus2", "sus4": "sus4", "sus4(b7)": "7sus4", "9": "9", "maj9": "maj9", "min9": "m9",
               "11": "11", "13": "13", "dim7": "dim7", "hdim7": "m7b5", "maj6": "6", "min6": "m6"}


_mv_CP_BASS_INTERVAL = {"2": 2, "b3": 3, "3": 4, "4": 5, "b5": 6, "5": 7, "#5": 8, "b6": 8, "6": 9, "bb7": 9,
                     "b7": 10, "7": 11, "9": 2}


_mv_CP_SHARP = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


_mv_CP_FLAT = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]


_mv_CP_FLAT_KEYS = {("major", 5), ("major", 10), ("major", 3), ("major", 8), ("major", 1), ("major", 6),
                 ("minor", 2), ("minor", 7), ("minor", 0), ("minor", 5), ("minor", 10), ("minor", 3)}


_mv_CP_ROOT_PC = {"C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3, "E": 4, "F": 5, "F#": 6, "Gb": 6, "G": 7,
               "G#": 8, "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11, "Cb": 11, "Fb": 4, "E#": 5, "B#": 0}


def _mv_cp_parse(label):
    """(root pc, quality, bass pc or None) from "A:maj/3"; None for no chord."""
    lab = str(label or "").strip()
    if not lab or lab in ("N", "X") or ":" not in lab:
        return None
    root, _, rest = lab.partition(":")
    pc = _mv_CP_ROOT_PC.get(root)
    if pc is None:
        return None
    qual, _, bass = rest.partition("/")
    if qual not in _mv_CP_QUALITY:
        return None
    bpc = None
    if bass:
        iv = _mv_CP_BASS_INTERVAL.get(bass)
        if iv is not None and iv != 0:
            bpc = (pc + iv) % 12
    return (pc, qual, bpc)


def _mv_cp_key(key_text):
    k = _mv_parse_key_root_quality(key_text) if key_text else None
    return k  # (root pc, "major"/"minor") or None


def _mv_cp_name(ch, key):
    names = _mv_CP_FLAT if key and (key[1], key[0]) in _mv_CP_FLAT_KEYS else _mv_CP_SHARP
    pc, qual, bpc = ch
    s = names[pc] + _mv_CP_QUALITY.get(qual, "")
    return s + ("/" + names[bpc] if bpc is not None else "")


def _mv_seconds_to_mmss(seconds):
    """Format seconds as M:SS, matching the STRUCTURE=[...] timestamp style
    used elsewhere in this file.

    Truncates like a media player's elapsed-time counter instead of rounding:
    a section that starts at 24.6 s is already playing when the player shows
    0:24, and rounding it to 0:25 made every reported start look late. A
    0.2 s allowance absorbs beat-grid jitter (24.85 s still reads 0:25).
    """
    seconds = max(0.0, float(seconds))
    total = int(math.floor(seconds + 0.2))
    m, s = divmod(total, 60)
    return f"{m}:{s:02d}"


def _mv_any_marker_present(text, markers):
    """Return True if any marker phrase appears in text as a real word/phrase
    start, not merely as a run of letters buried inside an unrelated word.

    Plain substring checks (`marker in text`) let short markers false-positive
    on unrelated words that merely contain the same letters -- e.g. "thin"
    inside "within", "teen" inside "eighteen"/"sixteen"/"nineteen", or "light"
    inside "highlight"/"delight"/"flight". Anchoring on a leading word
    boundary (`\\bmarker`) fixes that while still deliberately allowing a
    marker to match as the START of a longer word (e.g. "teen" still matches
    "teenager", "bright" still matches "brighter") since several markers here
    are intentional prefixes (e.g. "post-pubert" matching "post-pubertal").
    Assumes `text` is already lowercased, matching how markers are written.
    """
    if not text:
        return False
    for marker in markers:
        if re.search(r"\b" + re.escape(marker), text):
            return True
    return False


_mv_NOTE_LETTER_PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


def _mv_parse_key_root_quality(key_text):
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
    base = _mv_NOTE_LETTER_PC.get(m.group(1).upper())
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


# ---- 2.3: Musiclyse dev45 blocks (chords, melody in the key, melodicism) -----------------
_CP_HEAD_TXT = "CHORD PROGRESSION BY SECTION"
_MK_HEAD_TXT = "MELODY IN THE KEY BY SECTION"
_ML_HEAD_TXT = "MELODICISM BY SECTION"
_PART_ROW_RE = re.compile(r"^(.+?) \((\d+:\d\d)-(\d+:\d\d)\):\s*(.+)$")


def _block_rows(A, head):
    """(heading line, [row lines]) of the last block starting with `head`."""
    t = str(A or "")
    i = t.rfind("\n" + head)
    i = (i + 1) if i >= 0 else (0 if t.startswith(head) else -1)
    if i < 0:
        return "", []
    j = t.find("\n\n", i)
    lines = t[i:(j if j >= 0 else len(t))].split("\n")
    return lines[0], [l.strip() for l in lines[1:] if l.strip()]


def _pretty_harmony(text):
    """'F#m - A - E ...' -> 'F♯m – A – E …'; Roman 'bVII' -> '♭VII'."""
    t = str(text or "").replace(" ...", " …").replace("...", "…")
    t = re.sub(r"(?<=[A-G])#", "♯", t)
    t = re.sub(r"(?<![A-Za-z])b(?=[IV])", "♭", t)
    t = re.sub(r"(?<![A-Za-z])#(?=[IV])", "♯", t)
    t = re.sub(r"(?<=\b[A-G])b(?=m|maj|min|dim|aug|sus|add|\d|/|\s|,|\)|$)", "♭", t)
    return t.replace(" - ", " – ")


def parse_chord_progression(A):
    """Musiclyse dev45's CHORD PROGRESSION BY SECTION block: {"key", "cycle",
    "sections": [{name, start, end, chords, roman, rhythm, outside, modal, none}]}; {} if absent."""
    head, rows = _block_rows(A, _CP_HEAD_TXT)
    if not rows:
        return {}
    km = re.search(r"Roman numerals are against the recommended key, ([^;]+);", head)
    out = {"key": km.group(1).strip() if km else "", "cycle": "", "sections": []}
    for l in rows:
        if l.startswith("Most common cycle:"):
            out["cycle"] = _pretty_harmony(l.split(":", 1)[1].strip())
            continue
        m = _PART_ROW_RE.match(l)
        if not m:
            continue
        sec = {"name": m.group(1).strip(), "start": _secs(m.group(2)), "end": _secs(m.group(3)),
               "chords": "", "roman": "", "rhythm": "", "outside": "", "modal": False, "none": False}
        rest = m.group(4).strip()
        if rest.startswith("no clear chords"):
            sec["none"] = True
        else:
            parts = [x.strip() for x in rest.split("; ") if x.strip()]
            mr = re.match(r"^(.*?)\s*\(([^()]*)\)$", parts[0])
            sec["chords"] = _pretty_harmony(mr.group(1) if mr else parts[0])
            sec["roman"] = _pretty_harmony(mr.group(2)) if mr else ""
            for x in parts[1:]:
                if x.startswith("outside the key:"):
                    sec["outside"] = _pretty_harmony(x.split(":", 1)[1].strip())
                elif x.startswith("most chords here lie outside the key"):
                    sec["modal"] = True
                elif x.startswith("about "):
                    sec["rhythm"] = x
        out["sections"].append(sec)
    return out if out["sections"] else {}


def chord_timeline(segments, key_text=None):
    """[(start s, chord name)] from a save's raw chord segments (dev45), with
    repeats of one chord merged; [] when there are none."""
    try:
        key = _mv_cp_key(key_text) if key_text else None
    except Exception:
        key = None
    out = []
    for it in segments or []:
        try:
            a, b, lab = float(it[0]), float(it[1]), str(it[2])
        except (TypeError, ValueError, IndexError):
            continue
        if b - a < 0.25:
            continue
        ch = _mv_cp_parse(lab)
        name = _pretty_harmony(_mv_cp_name(ch, key)) if ch else "no chord"
        if out and out[-1][1] == name:
            continue
        out.append((a, name))
    return out


def chord_progression_from_segments(segments, sections, key_text=None):
    """Fallback progression from top-level chord_segments when the analysis text lacks its block.

    The saved chord timeline can preserve the chord sequence even for old, partial or
    hand-edited JSON saves whose prose block is absent. It cannot recover the cycle's
    repetition count or Roman numerals if those were not saved in the text.
    """
    try:
        key = _mv_cp_key(key_text) if key_text else None
    except Exception:
        key = None
    clean = []
    for item in segments or []:
        try:
            a, b, label = float(item[0]), float(item[1]), str(item[2])
            if not (math.isfinite(a) and math.isfinite(b)) or b <= a or b - a < 0.25:
                continue
        except (TypeError, ValueError, IndexError, OverflowError):
            continue
        chord = _mv_cp_parse(label)
        name = _pretty_harmony(_mv_cp_name(chord, key)) if chord else "no chord"
        clean.append((a, b, name))
    if not clean:
        return {}
    clean.sort(key=lambda x: (x[0], x[1]))

    spans = []
    for sec in sections or []:
        try:
            a, b = float(sec.get("start")), float(sec.get("end"))
            if not (math.isfinite(a) and math.isfinite(b)) or b <= a:
                continue
        except (TypeError, ValueError, AttributeError, OverflowError):
            continue
        spans.append((str(sec.get("name") or "Section"), a, b))
    if not spans:
        spans = [("Whole track", clean[0][0], max(x[1] for x in clean))]

    rows = []
    for name, a, b in spans:
        seq = []
        for start, end, chord_name in clean:
            if end <= a or start >= b:
                continue
            if not seq or seq[-1] != chord_name:
                seq.append(chord_name)
        if not seq:
            continue
        only_none = all(ch == "no chord" for ch in seq)
        rows.append({"name": name, "start": a, "end": b,
                     "chords": " – ".join(seq), "roman": "", "rhythm": "",
                     "outside": "", "modal": False, "none": only_none})
    return {"key": str(key_text or ""), "cycle": "", "sections": rows,
            "from_segments": True} if rows else {}


def parse_melody_in_key(A):
    """Musiclyse dev45's MELODY IN THE KEY BY SECTION block: {"key", "tonic", "rows": [...]}; {} if absent."""
    head, rows = _block_rows(A, _MK_HEAD_TXT)
    if not rows:
        return {}
    km = re.search(r"against the key, ([^:]+): 1 = ([A-G][#♯b♭]?)", head)
    out = {"key": km.group(1).strip() if km else "", "tonic": _pretty_harmony(km.group(2)) if km else "", "rows": []}
    for l in rows:
        m = _PART_ROW_RE.match(l)
        if not m:
            continue
        txt = m.group(4).strip()
        txt = txt.replace(" -- describe this part by its contour only", " -- only its rise and fall is clear")
        out["rows"].append({"name": m.group(1).strip(), "start": _secs(m.group(2)), "end": _secs(m.group(3)),
                            "text": _pretty_harmony(txt),
                            "unclear": "do not sit clearly in the key" in txt})
    return out if out["rows"] else {}


def parse_melodicism(A):
    """Musiclyse dev45's MELODICISM BY SECTION block: {"overall", "most", "least", "rows"}; {} if absent."""
    _head, rows = _block_rows(A, _ML_HEAD_TXT)
    if not rows:
        return {}
    out = {"overall": "", "most": "", "least": "", "rows": []}
    for l in rows:
        if l.startswith("Overall:"):
            out["overall"] = l.split(":", 1)[1].strip().rstrip(".")
            continue
        m = re.match(r"^Most melodic part: (.+?); most rhythm-led part: (.+?)\.?$", l)
        if m:
            out["most"], out["least"] = m.group(1).strip(), m.group(2).strip()
            continue
        m = _PART_ROW_RE.match(l)
        if m:
            out["rows"].append({"name": m.group(1).strip(), "start": _secs(m.group(2)),
                                "end": _secs(m.group(3)), "text": m.group(4).strip()})
    return out if (out["overall"] or out["rows"]) else {}


def identity_flip_pattern(A):
    """Musiclyse dev46 IDENTITY_FLIP_NOTE's test: the first vocal pass heard a
    woman, the confirmation pass a man (which became FINAL LEAD PROFILE), and
    the listening model's own voice descriptions only a woman."""
    t = str(A or "")
    ini = re.search(r"(?m)^-?\s*Initial isolated lead profile:\s*([a-z_]+)", t)
    con = re.search(r"(?m)^-?\s*Confirmation pass:\s*([a-z_]+)", t)
    fin = re.search(r"(?m)^-?\s*FINAL LEAD PROFILE:\s*([a-z_]+)", t)
    ft = re.search(r"LISTENING-MODEL VOICE DESCRIPTIONS[^:]*:\s*(\d+) describe a male voice, (\d+) a female voice", t)
    if not (ini and con and fin and ft):
        return False
    male = ("post_puberty_male", "adult_male", "young_male")
    return (ini.group(1) in _FEMALE_TAGS and con.group(1) in male and fin.group(1) in male
            and int(ft.group(1)) == 0 and int(ft.group(2)) >= 1)


def _part_for(rows, start, tol=2.0):
    """The row (dict with "start") that begins within tol seconds of `start`, or None."""
    for r in rows or []:
        if r.get("start") is not None and start is not None and abs(r["start"] - start) <= tol:
            return r
    return None



def parse_track(data, json_path=""):
    A = strip_md(data.get("analysis") or "")
    # dev17 load-time repairs: a chord-quality key ("D aug") takes Essentia's
    # mode; a LOW-confidence pitch-tracker median far from the note reading is
    # replaced by it. Both are idempotent and leave dev17 saves unchanged.
    A = pitch_low_confidence_fix(key_quality_repair(A))
    # dev18 KEY NOTE CHECK for saves made before it: a listening-model key the
    # saved notes do not fit is set aside (and, when it had won over a weak
    # Essentia key the notes fit, replaced by Essentia's).
    A, _key_switched = key_note_repair(A, data.get("stem_midi"))
    _key_fit_switched = False
    if KEY_FIT_CHECK:
        A, _key_fit_switched = key_fit_repair(A, data.get("stem_midi"))
    # dev19 tempo checks for saves made before them.
    A, _tempo_fixed = tempo_repair(A, data.get("stem_midi"))
    # Musiclyse dev20 MOSS_LOW_TRUST_FIELDS: a genre resting on the stock list.
    _A_pre_stock = A
    A, _genre_stock = stock_genre_repair(A)
    # Musiclyse dev20 GENRE_SYNTH_ROCK: synth-pop heard, a rock band measured.
    A, _synth_rock = synth_rock_repair(A)
    meta = data.get("metadata") or {}
    # 2.3 GENRE_VIEW_DEV44: Musiclyse dev44's genre view of the notes (style-led
    # genre, the style era check, regional / World labels), as its chat model sees them.
    if GENRE_VIEW_DEV44:
        try:
            A = _mv_world_label_view(_mv_genre_style_view(A, meta, _mv_year_from_meta(meta)), meta)
        except Exception:
            pass
    info = {"json_path": json_path, "raw_analysis": A, "genre_stock": _genre_stock, "synth_rock": _synth_rock}
    _pd = re.search(r"PITCH CHECK \(few voiced frames\): the pitch tracker read a median of [\d.]+ Hz \(([A-G][#♯b♭]?\d)\)", A)
    info["pitch_discarded_note"] = _pd.group(1) if _pd else ""
    # the tracker's own figures, as saved (before the check rewrote the block)
    _raw = str(data.get("analysis") or "")
    _pf = re.findall(r"objective voiced pitch median:\s*([\d.]+)\s*Hz|practical sung range \(10th-90th percentile\):"
                     r"\s*([\d.]+)-([\d.]+)\s*Hz", _raw) if _pd else []
    info["pitch_discarded_figures"] = sorted({x.rstrip("0").rstrip(".") for t in _pf for x in t if x})

    # ---- identity -------------------------------------------------------
    label = os.path.splitext(os.path.basename(str(data.get("label") or json_path or "track")))[0]
    info["title"] = str(meta.get("title") or "").strip()
    info["artist"] = str(meta.get("artist") or "").strip()
    info["album"] = str(meta.get("album") or "").strip()
    info["year"] = str(meta.get("year") or "").strip()
    info["label"] = label
    info["display"] = (f"{info['artist']} – “{info['title']}”"
                       if info["artist"] and info["title"] else (info["title"] or label))
    info["corrections"] = data.get("corrections") or {}

    # ---- structure ------------------------------------------------------
    # The analysis text's structure block carries "[name check: ...]" notes;
    # saves before Musiclyse 0.4-dev31 kept them only there.
    block_rows, block_checks = [], {}
    for body in _structure_block_bodies(A):
        for line in body:
            mm = re.match(r"^(.+?):\s*(\d+:\d\d)\s*(?:to|–|-)\s*(\d+:\d\d)(.*)$", line)
            if mm:
                a = sum(int(x) * 60 ** k for k, x in enumerate(reversed(mm.group(2).split(":"))))
                b = sum(int(x) * 60 ** k for k, x in enumerate(reversed(mm.group(3).split(":"))))
                name = mm.group(1).strip()
                block_rows.append({"name": name, "start": a, "end": b})
                nc = re.search(r"\[name check:\s*(.+?)\]\s*$", mm.group(4))
                if nc:
                    block_checks[(name, mmss(a))] = nc.group(1).strip()
        if block_rows:
            break

    sections = []
    for s in data.get("structure_sections") or []:
        try:
            sections.append({"name": str(s.get("name") or "?"),
                             "start": float(s["start"]), "end": float(s["end"]),
                             "cluster": str(s.get("cluster") or "").strip(),
                             "name_check": str(s.get("name_check") or "").strip()})
        except (KeyError, TypeError, ValueError, AttributeError):
            continue
    if not sections:
        sections = [dict(r, cluster="", name_check="") for r in block_rows]
    for sec in sections:
        if not sec["name_check"]:
            sec["name_check"] = block_checks.get((sec["name"], mmss(sec["start"])), "")
    for i, note in same_name_different_material(sections).items():
        lab = re.search(r'"([^"]+)"', note)
        lab = re.escape(lab.group(1)) if lab else None
        already = bool(lab and re.search(r'sections named "' + lab + r'"|the other "' + lab + r'" sections',
                                         sections[i]["name_check"]))
        if note not in sections[i]["name_check"] and not already:
            sections[i]["name_check"] = "; ".join(x for x in (sections[i]["name_check"], note) if x)
    info["sections"] = sections

    dur = None
    if sections:
        dur = sections[-1]["end"]
    else:
        m = re.search(r"^duration:\s*([0-9.]+)\s*s", A, re.M)
        if m:
            dur = float(m.group(1))
    info["duration"] = dur

    # ---- tempo / key / dynamics / genre ---------------------------------
    m = re.search(r"RECOMMENDED TEMPO FOR DISCUSSION:\s*([\d.]+)\s*BPM\.\s*Reasoning:\s*(.*?)\.?\s*This is the primary tempo",
                  A, re.S)
    info["bpm"] = _tempo_fixed or data.get("bpm") or (float(m.group(1)) if m else None)
    # The beat grids in a save whose tempo the dev19 check corrected were
    # counted at the old tempo; they are not shown as patterns.
    _rg = re.search(r"worked out at (\d+) BPM, the tempo this file was first given", A) if _tempo_fixed else None
    info["tempo_regrid"] = int(_rg.group(1)) if _rg else None
    info["tempo_snare"] = bool(_tempo_fixed and "TEMPO CHECK: the snare" in A)
    info["tempo_conf"] = _tempo_confidence(m.group(2) if m else "")
    info["full_band_tempo"] = full_band_tempo(A)

    keys = re.findall(r"RECOMMENDED KEY FOR DISCUSSION:\s*(.+?)\.\s+Reasoning:\s*(.*?)\s*This is the primary key", A, re.S)
    info["key"] = data.get("key") or (keys[-1][0].strip() if keys else None)
    if keys and _KEY_QUALITY_RE.match("RECOMMENDED KEY FOR DISCUSSION: " + str(info["key"] or "") + "."):
        info["key"] = keys[-1][0].strip()                  # repaired above
    if keys and _key_switched:
        info["key"] = keys[-1][0].strip()                  # key note check above
    if keys and _key_fit_switched:
        info["key"] = keys[-1][0].strip()                  # key fit check above
    info["key_fit_switched"] = _key_fit_switched
    kc = key_centre_reading(A)
    info["key_centre"] = kc
    if kc:
        info["key"] = kc["key"]
    info["key_conf"] = _key_confidence(re.sub(r"^KEY CENTRE CHECK:.*?The detector's own reading follows:\s*", "",
                                              keys[-1][1], flags=re.S) if keys else "")
    m = re.search(r"independently suggested (.+?) \(", A)
    info["key_alt"] = m.group(1).strip() if m else ""

    m = re.search(r"RECOMMENDED DYNAMICS FOR DISCUSSION:\s*(.+?)\s*\[", A)
    dyn = m.group(1).strip() if m else ""
    info["dynamics"] = "" if dyn.lower().startswith("unavailable") else dyn.replace("≈", "~")
    m = re.search(r"integrated loudness [≈~]\s*(-?[\d.]+)\s*LUFS", A)
    info["lufs"] = m.group(1) if m else ""

    m = re.search(r"RECOMMENDED GENRE FOR DISCUSSION:\s*(.+?)\.\s+Reasoning:\s*(.*?)\s*This is the primary genre", A, re.S)
    info["genre"] = m.group(1).strip() if m else ""
    info["genre_reasoning"] = " ".join((m.group(2) if m else "").split())
    info["genre_hedge"] = _genre_hedge(m.group(2) if m else "")
    info["panns_genre"] = first_line_value(A, "genre-adjacent tags:")
    info["panns_mood"] = first_line_value(A, "mood-adjacent tags:")

    # ---- listening-model fields -----------------------------------------
    # When the listening model answered in prose rather than LABEL= fields
    # (common), its description of the sound was previously dropped entirely.
    main = main_pass_text(A)
    info["mf"] = parse_fields(main, MF_FIELDS)
    for k, v in parse_fields(A, MF_FIELDS).items():
        info["mf"].setdefault(k, v)

    def _is_field(l):
        mm = _LABEL_RE.match(l)
        return bool(mm and _label_of(mm))
    prose = "\n".join(l for l in main.splitlines()
                      if l.strip() and not _is_field(l) and not l.lstrip().startswith(("[", "|")))
    info["listening_prose"] = prose if (len(prose) >= 200 and not info["mf"].get("INSTRUMENTATION")) else ""
    # dev64+ REPORT_NOTE: which report fields are measurements, not words by ear
    m = re.search(r"(?m)^REPORT_NOTE=.*?measured, not the listening model's words: ([A-Z_, ]+)", A)
    info["report_measured"] = {x.strip() for x in m.group(1).split(",") if x.strip()} if m else set()
    # Prose the listening model wrote beside its LABEL= fields (also dev36's
    # FREE-FORM DESCRIPTION pass) was dropped before; it is more of its description.
    # Drop pipeline metadata and the fixed-layout report scaffolding (dev64+).
    _DROP_EXTRA = {"STRUCTURE", "REPORT_NOTE", "CHECKS", "DESCRIPTION"}
    info["mf_extra"] = {k: v for k, v in parse_all_fields(main).items()
                        if k not in MF_FIELDS and k not in _DROP_EXTRA}
    # DESCRIPTION often carries a useful sentence or two, but some saves loop a
    # chord progression hundreds of times — keep a short clean version.
    # MF_FIELDS has no DESCRIPTION, so before 2.2 this was always empty. dev17's
    # report ends with a DESCRIPTION field (MOSS-Music writes it last).
    desc = (info["mf"].get("DESCRIPTION") or parse_fields(main, ["DESCRIPTION"], 3000).get("DESCRIPTION")
            or "").strip()
    if desc and not desc.lower().startswith(("uncertain", "see ", "(none")):
        # Collapse a chord cycle repeated 3+ times into one cycle.
        desc = re.sub(
            r"((?:[A-G][#b]?m?(?:maj|min|sus|add|dim|aug)?\d*(?:/[A-G][#b]?)?(?:\s*[–\-–—]\s*)?){2,8})(?:\1){2,}",
            r"\1 (repeating)",
            desc,
        )
        # Hard cap so a runaway DESCRIPTION never floods the dossier.
        desc = clip(desc, 1200)
        info["mf"]["DESCRIPTION"] = desc
        if info["listening_prose"]:
            # its continuation lines were read as prose too; they now appear once,
            # in the overall impression
            dn = " ".join(desc.split())
            info["listening_prose"] = "\n".join(l for l in info["listening_prose"].splitlines()
                                                 if " ".join(l.split()) not in dn)
            if len(info["listening_prose"]) < 200:
                info["listening_prose"] = ""
    field_text = " ".join(" ".join(str(v).split()) for v in list(info["mf"].values()) + list(info["mf_extra"].values()))
    extra_prose = [l for l in prose.splitlines()
                   if " ".join(l.split()) and " ".join(l.split()).strip("[]") not in field_text
                   and not l.startswith(("AUDIO EXCERPT CAUTION", "FREE-FORM DESCRIPTION"))]
    info["listening_prose_extra"] = ("\n".join(extra_prose)
                                     if (extra_prose and info["mf"].get("INSTRUMENTATION")) else "")
    m = re.search(r"AUDIO EXCERPT CAUTION:\s*The analysis below was performed on a short excerpt from "
                  r"([\d.]+)s to ([\d.]+)s", A)
    info["excerpt"] = (float(m.group(1)), float(m.group(2))) if m else None

    # Older saves (or runs where reconciliation failed) may only have the
    # listening model's own KEY= / TEMPO_BPM= fields. Use them as a fallback,
    # flagged as single-source estimates.
    if not info["key"]:
        k = info["mf"].get("KEY", "")
        if k and k.lower() not in ("uncertain", "unknown", "n/a"):
            info["key"] = clip(k, 80)
            info["key_conf"] = info["key_conf"] or "single estimate by ear"
    if not info["bpm"]:
        mm = re.search(r"(\d{2,3}(?:\.\d+)?)", info["mf"].get("TEMPO_BPM", ""))
        if mm:
            info["bpm"] = float(mm.group(1))
            info["tempo_conf"] = info["tempo_conf"] or "single estimate by ear"

    # ---- era ------------------------------------------------------------
    # dev55+ adds a note inside the heading when the pass gave no estimate:
    # "ERA / RELEASE PERIOD (isolated dedicated pass; it gave a general
    # description and NO era estimate -- ...):". Before 2.0 that heading was
    # not recognised and the whole era pass was lost.
    era_txt = ""
    m = re.search(r"ERA / RELEASE PERIOD \(isolated dedicated pass([^\n]*?)\):[ \t]*\n", A)
    if m:
        rest = A[m.end():]
        end = len(rest)
        for stop in ("\n\nERA CAVEAT", "\n\nVOCAL / SINGER PROFILE", "\n\nVOCAL PITCH REPORT",
                     "\n\nRECOMMENDED ", "\n\n[Independent", "\n\nVOCAL PRESENCE ARBITRATION"):
            j = rest.find(stop)
            if 0 <= j < end:
                end = j
        era_txt = rest[:min(end, 3000)]
    info["era_no_estimate"] = bool(m and "NO era estimate" in m.group(1))
    info["era"] = parse_fields(era_txt, ["ERA_ESTIMATE", "CONFIDENCE", "EVIDENCE"], 900)
    if info["era_no_estimate"] or not info["era"].get("ERA_ESTIMATE"):
        # A description instead of a dating: keep it as the pass's impression of
        # the sound, never as an era estimate.
        desc = " ".join(l.strip() for l in strip_md(era_txt).splitlines() if l.strip())
        info["era_description"] = clip(desc, 700) if desc and not info["era"].get("ERA_ESTIMATE") else ""
    else:
        info["era_description"] = ""
    m = re.search(r"(?m)^\s*MISSING[ _]MARKERS\s*[=:]\s*(.+)$", strip_md(era_txt))
    if m and m.group(1).strip().strip("[]").strip():
        info["era"]["MISSING_MARKERS"] = clip(m.group(1).strip().strip("[]").strip(), 400)

    # ---- vocals ---------------------------------------------------------
    # Presence itself is decided further down, once the stem levels are known.

    m = re.search(r"FINAL LEAD PROFILE:\s*([A-Za-z_]+)", A)
    info["acoustic_lead"] = m.group(1) if m else ""
    m = re.search(r"- BACKING PROFILES:\s*([A-Za-z_]+)", A)
    info["backing"] = m.group(1) if m else ""
    info["voices"] = {
        k: (re.search(rf"- {k}:\s*(.+)", A).group(1).strip() if re.search(rf"- {k}:\s*(.+)", A) else "")
        for k in ("NUM_DISTINCT_VOICES", "VOICE_ARRANGEMENT", "CO_LEAD_DETAIL")
    }

    si = strip_md(data.get("singer_identity") or "")
    if not si:
        si = between(A, "SINGER IDENTITY RESOLUTION", ("\n\n",), 1500)
    si = feature_credit_identity(si, meta)
    si = group_lead_identity(si, meta)
    m = re.search(r"SINGER_IDENTITY\s*[=:]\s*([A-Za-z_]+)", si)
    info["identity"] = m.group(1).lower() if m else ""
    info["identity_reason"] = identity_reason_from_block(si)
    m = re.search(r"CONFIDENCE\s*[=:]\s*(\w+)", si)
    info["identity_conf"] = m.group(1).lower() if m else ""
    vp = between(A, "VOCAL / SINGER PROFILE (isolated dedicated pass):",
                 ("\n\nVOCAL CONFIRMATION PASS", "\n\nVOCAL DECISION AUDIT"), 8000)
    info["vocal_notes"] = parse_fields(vp, ["LEAD_PROFILE_NOTE", "TIMBRE_NOTE", "DELIVERY_NOTE",
                                            "PROCESSING_NOTE", "PITCH_NOTE"], 500)
    # Every other field of the vocal pass and its confirmation pass (backing,
    # formants, multi-voice evidence, the pass's own probabilities...).
    vc = between(A, "VOCAL CONFIRMATION PASS:", ("\n\n",), 4000)
    extra_v = {}
    for src in (parse_all_fields(vp, 500), parse_all_fields(vc, 500)):
        for k, v in src.items():
            if k not in info["vocal_notes"] and k not in extra_v and k != "VOCALS_PRESENT":
                extra_v[k] = v
    info["vocal_extra"] = extra_v
    m = re.search(r"- Initial isolated lead profile:\s*(\S+)", A)
    info["initial_lead"] = m.group(1) if m else ""
    m = re.search(r"- Confirmation pass:\s*(\S+)\s*\(confidence=([^)]*)\)", A)
    info["confirm_lead"] = (m.group(1), m.group(2)) if m else ("", "")
    m = re.search(r"- MULTI_VOICE_EVIDENCE:\s*(.+)", A)
    info["multi_voice_evidence"] = m.group(1).strip() if m else ""
    m = re.search(r"LEAD_SINGERS\s*[=:]\s*[\"'<\[]?\s*(one|several|uncertain|single|multiple|two)\b", si, re.I)
    info["lead_singers"] = ({"single": "one", "multiple": "several", "two": "several"}
                            .get(m.group(1).lower(), m.group(1).lower()) if m else "")
    info["identity_extra"] = {k: v for k, v in parse_all_fields(si, 600).items()
                              if k not in ("SINGER_IDENTITY", "REASONING", "CONFIDENCE", "LEAD_SINGERS")}
    _acts = [info["artist"], re.split(r"(?i)\s+(?:feat(?:\.|․)?|ft(?:\.|․)|featuring|with)\s",
                                      info["artist"])[0], info["title"], info["album"]]
    info["identity_facts"] = identity_facts(info["identity_reason"], info.get("year"),
                                            act_names=[x for x in _acts if x])
    # dev17 SECTION VOICE CHECK: a second lead of the other gender in some
    # stretches. A one-voice identity answer is set to mixed_leads, as dev17
    # does when it loads the save; the old tag then names the main lead.
    sv = section_voice_check(A)
    info["section_voice"] = sv
    if sv and info["identity"] not in ("mixed_leads",):
        info["identity_main"] = info["identity"]
        info["identity"] = "mixed_leads"
        info["lead_singers"] = "several"

    # ---- dev55-dev59 voice evidence --------------------------------------
    info["voice_desc"] = parse_voice_descriptions(A)
    # Unbroken / deepened (dev59 UNBROKEN signature, dev69 DEEPENED).
    # Only track-specific claims count — not the generic identity-note boilerplate
    # ("cannot tell an unbroken boy's voice from a woman's"), which appears on
    # every sung track and would otherwise mark adult male leads as unbroken.
    _young_male_id = (info.get("identity") or "") in (
        "adolescent_male_likely", "young_male", "child_male_likely",
    )
    _adult_male_ac = (info.get("acoustic_lead") or "") in (
        "post_puberty_male", "adult_male",
    )
    _female_ac = (info.get("acoustic_lead") or "") in (
        "female_teen_adult", "adult_female", "young_female", "child_female_likely",
    )
    _unbroken_prose = bool(re.search(
        r"(?i)consistent with an unbroken(?: male)? voice"
        r"|lead voice sounds UNBROKEN"
        r"|VOICE MATURITY:\s*the voice sounds UNBROKEN"
        r"|high voice that has not fully deepened is routinely misread"
        r"|had(?:n't| not) broken yet on this recording",
        A,
    ))
    info["voice_unbroken"] = bool(
        _VOICE_UNBROKEN_MARK in A
        or (info["voice_desc"] or {}).get("unbroken")
        or _unbroken_prose
        # Classic pattern: documented young male whose audio-only tag is female
        # (high unbroken pitch read as a woman's voice).
        or (_young_male_id and _female_ac)
    )
    _deepened_prose = bool(re.search(
        r"(?i)voice had already deepened"
        r"|already deepened \(broken"
        r"|describe it as (?:his |their )?deepened voice"
        r"|lower and fuller than a boy'?s unbroken voice"
        r"|not as high, light or boyish",
        A,
    ))
    # dev18: a young male identity whose first vocal pass heard a woman and only
    # the confirmation pass a man, at a high measured pitch, is an unbroken voice
    # (dev17 wrote "may already have been deepening" for it).
    if _young_male_id and _adult_male_ac and not _deepened_prose and confirm_flip_reading(A):
        info["voice_unbroken"] = True
    # dev69's hedged sentence (median at or above 300 Hz): starting to deepen at
    # most -- not "had already deepened", which dev69 reserves for a low voice.
    info["voice_maybe_deepening"] = bool(
        _young_male_id and _adult_male_ac and not _deepened_prose and not info["voice_unbroken"]
        and re.search(r"so his voice may already have been deepening on this recording", A)
    )
    info["voice_deepened"] = bool(
        _deepened_prose
        or (_young_male_id and _adult_male_ac and not info["voice_unbroken"]
            and not info["voice_maybe_deepening"])
    )
    # Explicit deepened note wins over unbroken if both somehow fire.
    if info["voice_deepened"] and info["voice_unbroken"] and _deepened_prose:
        info["voice_unbroken"] = False
    elif info["voice_unbroken"] and info["voice_deepened"] and not _deepened_prose:
        info["voice_deepened"] = False
    if (UNBROKEN_VOICE_IDENTITY_CHECK and _young_male_id
            and ("UNBROKEN VOICE CHECK" in si or identity_reason_says_unbroken(info["identity_reason"]))):
        info["voice_unbroken"] = True
        info["voice_deepened"] = False
        info["voice_maybe_deepening"] = False
    m = re.search(r"VOCAL NOTE PITCH \(second reading, from the vocal stem's (\d+) transcribed notes, "
                  r"(\d+) s sung[^)]*\): median (\d+) Hz \(([^)]+)\); 10-90%: (\d+)-(\d+) Hz "
                  r"\((\S+) to (\S+?)\)", A)
    info["note_pitch"] = ({"notes": int(m.group(1)), "sung_s": int(m.group(2)), "median_hz": m.group(3),
                           "median_note": m.group(4), "lo_hz": m.group(5), "hi_hz": m.group(6),
                           "lo_note": m.group(7), "hi_note": m.group(8)} if m else {})
    m = re.search(r"pyin pitch reading confidence: LOW \(([^)]*)\)", A)
    info["pyin_low"] = m.group(1).strip() if m else ""
    info["vocal_pass"] = parse_all_fields(vp, 500)

    pr = {}
    m = re.search(r"- objective voiced pitch median:\s*([\d.]+)\s*Hz", A)
    if m:
        pr["median_hz"] = m.group(1)
    m = re.search(r"- approximate median note:\s*(\S+)", A)
    if m:
        pr["median_note"] = m.group(1)
    m = re.search(r"- (practical sung range \(10th-90th percentile\)|5th-95th percentile range|percentile range):\s*([\d.]+)-([\d.]+)\s*Hz", A)
    if m:
        pr["range_hz"] = (m.group(2), m.group(3))
    for key, rx in (("male_type", r"- if the singer is male \(any age\):\s*(.+)"),
                    ("female_type", r"- if the singer is female:\s*(.+)"),
                    ("remark", r"- worth remarking on[^:]*:\s*(.+)"),
                    ("range_notes", r"- practical range in notes:\s*(.+)")):
        m = re.search(rx, A)
        if m:
            pr[key] = m.group(1).strip()
    info["pitch"] = pr
    m = re.search(r"EARLIEST VOCAL ONSET:\s*(\d+:\d\d)", A)
    info["vocal_onset"] = m.group(1) if m else ""

    # ---- instruments ----------------------------------------------------
    wm = ""
    for line in A.splitlines():
        if line.startswith("WHOLE-MIX INSTRUMENT TAGS: none cleared"):
            wm = "nothing identified with confidence"
            break
        if line.startswith("WHOLE-MIX INSTRUMENT TAGS ("):
            j = line.rfind("): ")
            if j > 0:
                wm = line[j + 3:].strip()
            break
    info["mix_tags"] = wm

    stem_tags, stem_ids, cur = {}, {}, None
    for line in A.splitlines():
        mm = re.match(r"^\[([A-Z]+) stem\]", line.strip())
        if mm:
            cur = mm.group(1).lower()
            continue
        if cur and line.startswith("instrument tag (independent"):
            j = line.rfind("): ")
            if j > 0 and cur not in stem_tags:
                stem_tags[cur] = line[j + 3:].strip()
        # Musiclyse dev30+: what a stem actually holds when its name ("bass",
        # "guitar") is not confirmed by what the classifier hears.
        if cur and line.startswith("STEM IDENTITY NOTE:") and cur not in stem_ids:
            stem_ids[cur] = line.partition(":")[2].strip()
    # dev60+: drop accordion/harmonica tags the INSTRUMENT CHECK vetoed
    # (dev17: a strong, steady 'harmonica' is kept as a reedy lead, and an
    # accordion on a chordal 'other' stem is most likely a synth pad).
    icr = instrument_check_reading(A)
    suppressed = icr["suppressed"]
    info["instrument_suppressed"] = sorted(suppressed)
    info["reed_lead"] = icr["reed_lead"]
    info["reed_pad_poly"] = icr["pad_poly"]
    if suppressed:
        stem_tags = {st: filter_suppressed_tags(s, suppressed) for st, s in stem_tags.items()}
        if wm:
            wm = filter_suppressed_tags(wm, suppressed)
            info["mix_tags"] = wm
    info["stem_tags"] = stem_tags
    info["stem_ids"] = stem_ids
    m = re.search(r"(?m)^LOW-END NOTE:\s*(.+)$", A)
    info["low_end_note"] = m.group(1).strip() if m else ""

    agree = []
    lines = A.splitlines()
    for i, line in enumerate(lines):
        if line.startswith("CROSS-STEM INSTRUMENT AGREEMENT"):
            for l2 in lines[i + 1:]:
                s = l2.strip()
                mm = re.match(r"^- (.+?): agreed by (\d+) stems \((.+)\)$", s)
                if mm:
                    agree.append({"label": mm.group(1), "n": mm.group(2),
                                  "detail": mm.group(3), "caution": ""})
                    continue
                if s.startswith("[") and agree:
                    if "UNANCHORED" in s:
                        agree[-1]["caution"] = "may be one source counted twice — unconfirmed"
                    elif "EXTENT CONFLICT" in s:
                        agree[-1]["caution"] = "probably a brief/local detail, not an ongoing part"
                    continue
                break
            break
    if suppressed:
        agree = [a for a in agree if not _tag_label_suppressed(a.get("label", ""), suppressed)]
    info["cross_stem"] = agree

    guitar = ""
    if "GUITAR ABSENCE NOTE" in A:
        guitar = "no guitar detected (two independent checks agree)"
    elif "GUITAR EVIDENCE INCONCLUSIVE" in A:
        guitar = ("inconclusive — the detectors could not settle it. Do not claim the track has "
                  "or lacks guitar; describe the midrange layer by its sound instead")
    elif "GUITAR EVIDENCE UNAVAILABLE" in A:
        guitar = "not determined (no cross-check could be run)"
    info["guitar"] = guitar
    info["guitar_check"] = guitar_check(A)
    info["no_kit"] = drum_stem_note(A)

    prom = []
    for i, line in enumerate(lines):
        if line.startswith("STEM ACTIVITY / PROMINENCE"):
            for l2 in lines[i + 1:]:
                mm = re.match(r"^(\d+)\) (\w+): ([^(]+)\(", l2.strip())
                if not mm:
                    break
                prom.append(f"{mm.group(2)} ({mm.group(3).strip()})")
            break
    info["prominence"] = prom

    # ---- per-section blocks ---------------------------------------------
    intensity, whole_notes, minor_db = {}, {}, {}
    for m, body in header_blocks(A, r"^INTENSITY BY SECTION \((\w+) stem(.*)$"):
        stem = m.group(1)
        if stem in intensity or stem in whole_notes:
            continue
        rest = m.group(2)
        # One-line verdicts for the whole stem (Musiclyse dev30+): residue,
        # near-silent, or "not applicable" on an instrumental.
        wm = re.match(r"^\):\s*(.+)$", rest)
        if wm:
            v = wm.group(1).lower()
            if v.startswith("not applicable"):
                whole_notes[stem] = "not applicable — the recording is instrumental; this part holds only bleed from other instruments"
            else:
                whole_notes[stem] = ("near-silent throughout — only separation residue (bleed from other "
                                     "instruments); treat it as not playing")
            continue
        mm = re.search(r"overall this stem sits (\d+) dB below the combined stems", rest)
        if mm:
            minor_db[stem] = int(mm.group(1))
        rows = []
        for line in body:
            mm = re.match(r"^(.*) \((\d+:\d\d)-(\d+:\d\d)\): (.+)$", line)
            if mm:
                rows.append((mm.group(1).strip(), mm.group(4).strip()))
        if rows:
            intensity[stem] = rows
    info["intensity"] = intensity
    info["intensity_whole"] = whole_notes
    info["intensity_minor_db"] = minor_db

    # ---- vocal presence (needs the stem levels above) -------------------------
    present, how, why = decide_vocal_presence(A, info["prominence"], whole_notes)
    if present == "uncertain" and (
        info["identity"] not in ("", "unknown", "uncertain")
        or info["acoustic_lead"] not in ("", "unknown", "uncertain")
    ):
        present, how = "yes", "profile"      # a resolved singer means there is a voice
    info["vocals_present"], info["vocal_verdict"], info["vocal_verdict_why"] = present, how, why
    if present == "no":
        # Everything built on a "voice" in an instrumental is bleed: a singer
        # profile, an identity, a duet reading, a pitch range, an onset time.
        info["identity"] = info["identity_reason"] = info["identity_conf"] = ""
        info["acoustic_lead"] = info["backing"] = ""
        info["voices"] = {}
        info["vocal_notes"] = {}
        info["vocal_extra"] = {}
        info["initial_lead"], info["confirm_lead"] = "", ("", "")
        info["multi_voice_evidence"] = info["lead_singers"] = ""
        info["identity_extra"] = {}
        info["section_voice"] = {}
        info["pitch"] = {}
        info["vocal_onset"] = ""
        info["intensity"].pop("vocals", None)
        info["intensity_minor_db"].pop("vocals", None)
        info["intensity_whole"]["vocals"] = ("not applicable — the recording is instrumental; this part "
                                             "holds only bleed from other instruments")
        info["prominence"] = [p for p in info["prominence"] if not p.startswith("vocals ")]

    pres = header_blocks(A, r"^INSTRUMENT PRESENCE BY SECTION")
    presence_lines = pres[0][1] if pres else []
    if info.get("instrument_suppressed"):
        presence_lines = [
            l for l in presence_lines
            if not any(w in l.lower().split("(")[0] for w in info["instrument_suppressed"])
        ]
    info["presence"] = presence_lines
    chg = header_blocks(A, r"^ARRANGEMENT CHANGE POINTS")
    info["changes"] = chg[0][1] if chg else []
    if info["vocals_present"] == "no":
        info["changes"] = _drop_stem_from_changes(info["changes"], "vocals")

    melody = {}
    for m, body in header_blocks(A, r"^MELODY PER SECTION \((\w+) stem\)", mode="blank"):
        if m.group(1) not in melody and body:
            melody[m.group(1)] = body
    info["melody"] = melody

    # ---- drums ----------------------------------------------------------
    drums = {}
    for key, prefix in (("kick", "kick beat-grid analysis:"),
                        ("snare", "snare beat-grid analysis:"),
                        ("swing", "swing/shuffle analysis:"),
                        ("hats", "hi-hat density vs kick pulse:")):
        v = first_line_value(A, prefix)
        if v:
            drums[key] = _clean_hats(v) if key == "hats" else clean_label(v)
    gh = first_line_value(A, "GROOVE_HINT (")
    if gh:
        j = gh.find("): ")
        drums["groove"] = clean_label(gh[j + 3:] if j >= 0 else gh)
    drums["kit"] = [(lab, first_line_value(A, p)) for p, lab in DRUM_KIT_PREFIXES if first_line_value(A, p)]
    drums["sound"] = [(lab, first_line_value(A, p)) for p, lab in DRUM_SOUND_PREFIXES if first_line_value(A, p)]
    extra = first_line_value(A, "ADDITIONAL PERCUSSION DETECTED")
    if extra and "inventing a pattern:" in extra:
        drums["extra"] = extra.split("inventing a pattern:", 1)[1].strip()
    m = re.search(r"^CYMBALS \(detected[^\n]*?\):\s*(\d+) cymbal hits", A, re.M)
    drums["cymbal_hits"] = m.group(1) if m else ""
    tl = header_blocks(A, r"^DRUM ARRANGEMENT TIMELINE")
    drums["timeline"] = tl[0][1] if tl else []
    drums["fallback"] = "onset-band fallback" in A or "band-split onset detector" in A

    sec_groove = []
    for m, body in header_blocks(A, r"^SECTION GROOVE — (.+?) \(([^)]*)\):$"):
        d = {"label": m.group(1), "ranges": m.group(2), "grids": extract_grids(body)}
        for key, prefix in (("kick", "kick beat-grid analysis:"),
                            ("snare", "snare beat-grid analysis:"),
                            ("swing", "swing/shuffle analysis:"),
                            ("hats", "hi-hat density vs kick pulse:"),
                            ("groove", "GROOVE_HINT (")):
            v = first_line_value("\n".join(body), prefix)
            if v and key == "groove":
                j = v.find("): ")
                v = v[j + 3:] if j >= 0 else v
            if v:
                d[key] = _clean_hats(v) if key == "hats" else clean_label(v)
        sec_groove.append(d)
    drums["sections"] = sec_groove

    rep = [g for g in extract_grids(A.splitlines()) if g["label"] == "representative bars"]
    drums["rep_grid"] = rep[0] if rep else None
    info["drums"] = drums

    # ---- production -----------------------------------------------------
    prod = []
    i = A.find("PRODUCTION / MIX MEASUREMENTS")
    if i >= 0:
        cur = None
        for l in A[i:].splitlines()[1:]:
            s = l.strip()
            if s.startswith(("VOCAL OBJECTIVE MEASUREMENTS", "ESSENTIA OBJECTIVE", "This is computed directly",
                             "OBJECTIVE GENRE/MOOD", "RECOMMENDED ", "STEM MIDI REPORT", "[")):
                break
            if not s:
                continue
            title = next((v for k, v in PROD_TITLES if s.startswith(k)), None)
            if title:
                cur = [title, []]
                prod.append(cur)
                continue
            if cur is not None:
                cur[1].append(s)
    info["production"] = correct_old_production(prod)

    # ---- lyrics ---------------------------------------------------------
    tag_lyrics = str(meta.get("lyrics") or data.get("lyrics") or "").strip()
    if tag_lyrics and not tag_lyrics_usable(tag_lyrics, meta):
        tag_lyrics = ""          # 2.2.2: a lyrics tag that holds no lyrics ("Artist - Title (remix)")
    if tag_lyrics:
        info["lyrics"], info["lyrics_src"] = tag_lyrics, "tags"
    else:
        heard = ""
        for hdr in ("LYRICS HEARD IN AUDIO (secondary evidence; do not use when reliable lyrics are present):",
                    "FULL LYRICS TRANSCRIPTION"):
            heard = between(A, hdr, ("[END OF TRANSCRIPTION", "\n\nLYRIC REPETITION MAP",
                                     "\n\n11. ERA", "\n\nVOCAL / SINGER"), 20000)
            if heard:
                break
        heard = clean_lyrics_transcription(heard.lstrip(":\n "))
        if info["vocals_present"] == "no":
            heard = ""          # speech recognition on an instrumental is always invented text
        info["lyrics"], info["lyrics_src"] = heard, ("audio" if heard else "")

    # Older saves: the listening model claims a bass guitar but the bass part
    # is only residue (Musiclyse dev31 prints a LOW-END NOTE for this itself).
    if not info["low_end_note"] and "bass" in info["intensity_whole"]:
        claim_txt = " ".join([info["listening_prose"], info["mf"].get("INSTRUMENTATION", "")])
        if _BASS_CLAIM_RE.search(claim_txt):
            info["low_end_note"] = ("the listening model mentions a bass guitar / bass line, but the separated "
                                    "bass part is near-silent — there is no substantial separate bass part. "
                                    "Whatever low-register sound it heard belongs to other instruments.")

    info["cover"] = data.get("cover_observations") or {}
    info["cover_file"] = str(data.get("cover_art") or "").strip()

    # ---- transcribed MIDI summary (computed from the raw notes) ---------
    parts = []
    residue = set(info["intensity_whole"])
    for stem in STEM_ORDER:
        rows = (data.get("stem_midi") or {}).get(stem) or []
        rows = [r for r in rows if isinstance(r, dict)] if isinstance(rows, list) else []
        if not rows or (stem == "vocals" and info["vocals_present"] == "no"):
            continue
        if stem == "drums":
            counts = {}
            for r in rows:
                k = r.get("k") or "other"
                counts[k] = counts.get(k, 0) + 1
            parts.append(f"drums: {len(rows)} hits — " +
                         ", ".join(f"{k} {v}" for k, v in sorted(counts.items(), key=lambda kv: -kv[1])))
            continue
        ms = []
        for r in rows:
            try:
                ms.append(float(r["m"]))
            except (KeyError, TypeError, ValueError):
                continue
        ms.sort()
        if len(ms) < 3:
            continue
        p10 = ms[int(0.10 * (len(ms) - 1))]
        p90 = ms[int(0.90 * (len(ms) - 1))]
        med = ms[len(ms) // 2]
        parts.append(f"{stem}: {len(rows)} notes, typical range {midi_name(p10)}–{midi_name(p90)}, "
                     f"centred around {midi_name(med)}"
                     + (" (near-silent part — these notes are probably bleed from other instruments)"
                        if stem in residue else ""))
    info["midi_parts"] = parts
    info["midi_complete"] = (bool(data.get("stem_midi_complete"))
                              if "stem_midi_complete" in data else None)
    midi_report = re.search(r"(?m)^STEM MIDI REPORT \(([^)]+)\)", A)
    midi_coverage = {"kind": "unknown", "seconds": None}
    if midi_report:
        _coverage_text = midi_report.group(1)
        _partial = re.search(r"first\s*~?([\d,]+(?:\.\d+)?)\s*seconds?", _coverage_text, re.I)
        if _partial:
            try:
                midi_coverage = {"kind": "partial", "seconds": float(_partial.group(1).replace(",", ""))}
            except ValueError:
                pass
        elif re.search(r"\bfull track\b", _coverage_text, re.I):
            midi_coverage = {"kind": "full", "seconds": None}
    info["midi_coverage"] = midi_coverage

    # More from the raw notes: range extremes, the notes used most, when each
    # part is active, how hard it's played. (Each stem's own report lines are
    # kept separately below.)
    detail, raw = [], {}
    for stem, rows in (data.get("stem_midi") or {}).items():
        rows = [r for r in rows if isinstance(r, dict)] if isinstance(rows, list) else []
        if not rows:
            continue
        if stem == "vocals" and info["vocals_present"] == "no":
            continue
        raw[stem] = rows
        try:
            ts = sorted(float(r["t"]) for r in rows if isinstance(r, dict) and "t" in r)
        except (TypeError, ValueError):
            ts = []
        span = f"from {mmss(ts[0])} to {mmss(ts[-1])}" if ts else ""
        vs = []
        for r in rows:
            try:
                vs.append(float(r["v"]))
            except (KeyError, TypeError, ValueError):
                pass
        lvl = f", average level {sum(vs) / len(vs):.2f} (0-1)" if vs else ""
        if stem == "drums" or any(isinstance(r, dict) and r.get("k") for r in rows[:50]):
            detail.append(f"{stem}: hits {span}{lvl}")
            continue
        ms = []
        for r in rows:
            try:
                ms.append(int(round(float(r["m"]))))
            except (KeyError, TypeError, ValueError):
                pass
        if not ms:
            continue
        pcs = {}
        for x in ms:
            pcs[NOTE_NAMES[x % 12]] = pcs.get(NOTE_NAMES[x % 12], 0) + 1
        top = ", ".join(f"{n} {100.0 * c / len(ms):.0f}%" for n, c in sorted(pcs.items(), key=lambda kv: -kv[1])[:5])
        durs = []
        for r in rows:
            try:
                durs.append(float(r["d"]))
            except (KeyError, TypeError, ValueError):
                pass
        dl = f", typical note length {sorted(durs)[len(durs) // 2]:.2f}s" if durs else ""
        detail.append(f"{stem}: lowest {midi_name(min(ms))}, highest {midi_name(max(ms))}; notes used most: {top}; "
                      f"playing {span}{dl}{lvl}"
                      + (" (near-silent part — probably bleed)" if stem in residue else ""))
    info["midi_detail"] = detail
    info["raw_notes"] = raw
    info["midi_format"] = str(data.get("stem_midi_format") or "").strip()

    # ---- dev40-dev53 checks, computed from the saved text -------------------
    info["genre_check"] = genre_check_note(A)
    info["genre_replaced"] = genre_replacement(A)
    info["arrangement_check"] = arrangement_check_note(A)
    info["arrangement_denial"] = {} if info["arrangement_check"] else arrangement_denial_note(A)
    info["genre_era"] = genre_era_check(A, info.get("year"))
    # 2.2.3: the stock label the genre no longer rests on still gets its era warning
    info["genre_era_stock"] = genre_era_check(_A_pre_stock, info.get("year")) if _genre_stock else {}
    info["era_drum_note"] = "ERA NOTE (drums):" in A
    info["faint_vocal_texture"] = info["vocals_present"] == "no" and faint_vocal_texture(A)
    hm_src = A
    if _HM_HEAD not in A:
        hm_src = build_harmony_melody_block(data, A, info["sections"], info["vocals_present"] == "no")
    info["harmony_melody"] = parse_harmony_melody(hm_src)
    info["description"] = clean_description(info["mf"].get("DESCRIPTION", ""), info)
    vc_on, vc_why = voice_count_check(A)
    info["voice_count_check"] = vc_why if (vc_on and info["vocals_present"] != "no") else ""
    preserve_shared_vocal_evidence(info, meta)
    m = re.search(r"integrated loudness ≈\s*(-?\d+(?:\.\d+)?)\s*LUFS", A)
    lufs = float(m.group(1)) if m else None
    m = re.search(r"loudness range \(LRA\)\s*≈\s*(-?\d+(?:\.\d+)?)\s*LU", A)
    lra = float(m.group(1)) if m else None
    m = re.search(r"RECOMMENDED DYNAMICS FOR DISCUSSION:[^\n]*?crest-factor proxy ≈ (\d+(?:\.\d+)?) dB", A)
    crest = float(m.group(1)) if m else None
    info["loudness_context"] = loudness_context(lufs) if lufs is not None else ""
    info["era_caveat"] = era_mastering_caveat(
        lufs, lra, crest, saved=("ERA CAVEAT: this loud (" in A or "ERA CAVEAT (read this before" in A)) or ""
    info["era_clues"], info["era_range"] = era_evidence(A)
    info["genre_ranked"] = info["mf"].get("GENRE_RANKED", "")

    # ---- measurement blocks, per-stem reports, lyric repetition --------------------
    blocks = []
    for hdr, title, std in MEASURE_BLOCKS:
        head, body = paragraph_after(A, hdr)
        if not head:
            continue
        first = head.split(":", 1)[1].strip() if (":" in head and hdr.endswith(("REFINEMENT", "points:"))) else ""
        lines = ([first] if first else []) + body
        if hdr == "OBJECTIVE AUDIO MEASUREMENTS":
            lines = [l for l in body if l.startswith("duration:")]
        if hdr == "DRUM SOUND / PRODUCTION":
            lines = [l for l in body if not any(l.strip().startswith(p) for p, _x in DRUM_SOUND_PREFIXES)]
        if hdr == "VOCAL OBJECTIVE MEASUREMENTS" and info["vocals_present"] == "no":
            continue                    # pitch of separation bleed, not a voice
        if hdr == "candidate change points:":
            lines = [head.partition(":")[2].strip()]
        if lines:
            blocks.append({"title": title, "lines": lines, "standard": std})
    info["measure_blocks"] = blocks
    sb = stem_blocks(A)
    if info["vocals_present"] == "no":
        sb.pop("vocals", None)
    info["stem_reports"] = {s: [l for l in ls if not l.strip().startswith(_STEM_SHOWN_ELSEWHERE)]
                            for s, ls in sb.items()}
    rep = []
    head, body = paragraph_after(A, "LYRIC REPETITION MAP")
    for l in body:
        mm = re.match(r"^- recurring line \(x(\d+)\): \"(.*)\" at (.+)$", l.strip())
        if mm:
            rep.append((mm.group(2), int(mm.group(1)), mm.group(3).strip()))
    info["lyric_repeats"] = [] if info["vocals_present"] == "no" else rep

    # ---- the rest of the file ------------------------------------------------------
    info["replay"] = parse_replay(data.get("structure_replay"))
    if info["vocals_present"] == "no" and info["replay"]:
        info["replay"]["lyric_lines"], info["replay"]["lyric_groups"] = [], []
    info["meta_extra"] = {k: v for k, v in meta.items()
                          if k not in KNOWN_META_KEYS and str(v if v is not None else "").strip()}
    # "raw" is the cover model's unparsed reply, already split into the fields above.
    info["cover_extra"] = {k: v for k, v in (info["cover"] or {}).items()
                           if k not in {f for f, _l in COVER_FIELDS} and k != "raw" and str(v or "").strip()}
    fd = {}
    for k in ("label", "track_path", "saved_at", "version"):
        if data.get(k) not in (None, ""):
            fd[k] = data[k]
    sig = data.get("audio_file_signature")
    if isinstance(sig, (list, tuple)) and len(sig) >= 3:
        fd["audio_file_signature"] = sig
    for k in ("clap_embedding_model", "clap_embedding_dimension", "clap_embedding_aggregation"):
        if data.get(k):
            fd[k] = data[k]
    info["file_details"] = fd
    info["other_fields"] = {k: v for k, v in data.items()
                            if k not in KNOWN_TOP_KEYS and v not in (None, "", [], {})}

    emb = data.get("clap_embedding")
    info["embedding"] = emb if isinstance(emb, list) and len(emb) >= 16 else None

    # ---- 2.3: Musiclyse dev42-dev46 blocks ------------------------------------------
    try:
        info["styles"] = _mv_style_signal_items(A) if STYLES_IN_DOSSIER else []
    except Exception:
        info["styles"] = []
    m = re.search(r"(?m)^ERA CHECK: (.+)$", A)
    info["era_style_check"] = m.group(1).strip() if (m and ERA_STYLE_CHECK) else ""
    info["chord_prog"] = parse_chord_progression(A) if CHORD_PROGRESSION_IN_DOSSIER else {}
    info["chord_timeline"] = (chord_timeline(data.get("chord_segments"), info.get("key"))
                              if CHORD_PROGRESSION_IN_DOSSIER else [])
    # The prose block is the preferred source because it also carries cycle counts and
    # Roman numerals. If it is missing, retain the chord sequence saved separately in JSON.
    if CHORD_PROGRESSION_IN_DOSSIER and not (info["chord_prog"] or {}).get("sections"):
        info["chord_prog"] = chord_progression_from_segments(
            data.get("chord_segments"), info.get("sections") or [], info.get("key"))
    no_voice = info.get("vocals_present") == "no"
    info["melody_key"] = {} if (no_voice or not MELODY_IN_KEY_IN_DOSSIER) else parse_melody_in_key(A)
    info["melodicism"] = {} if (no_voice or not MELODICISM_IN_DOSSIER) else parse_melodicism(A)
    # Musiclyse dev46 IDENTITY_FLIP_NOTE, for saves made before it.
    info["identity_flip_unsettled"] = False
    if (IDENTITY_FLIP_GUARD and not no_voice and identity_flip_pattern(A)
            and info.get("identity") in ("post_puberty_male", "adult_male")
            and not _SOLO_WORD_RE.search(info.get("identity_reason") or "")
            and not singer_name_from_reason(info.get("identity_reason") or "",
                                            act_names=[x for x in (info["artist"], info["title"], info["album"]) if x])):
        info["identity_flip_unsettled"] = True
        info["identity"] = "uncertain"
        if info.get("acoustic_lead") in ("post_puberty_male", "adult_male"):
            info["acoustic_lead"] = "uncertain"
    return info


def _section(title, body, order, prio, truncatable=False):
    body = (body or "").strip()
    if not body:
        return None
    return {"title": title, "body": body, "order": order, "prio": prio, "trunc": truncatable}


# 2.2: lines added for dev17 that a tight --max-tokens budget gives up before it
# drops any section that 2.1 would have kept: (priority, pattern) per section
# title. Priority 10 sits below every section of the standard level.
_SHED = {
    "How it unfolds": [(10, r" Chords mostly [^.\n]*\."), (10, r" The chords: [^.\n]*\.")],
    "The sound": [(10, r"\n- Overall impression: [^\n]*"), (10, r"\n- What it isn't, by ear: [^\n]*"),
                  (10, r" \((?:the impressions by ear|two independent readings|a first rock-leaning|taken from the "
                       r"description|a weak call)[^)]*\)")],
    "The voice": [(10, r"\n- (?:Other voices|Backing vocals|Singing style|Vocal effects), by ear: [^\n]*"),
                  (10, r"\n- The arrangement of voices: [^\n]*")],
    "When it sounds like it's from": [(10, r"\n\nLater-era sounds it doesn't have, by ear: [^\n]*")],
    "Melody and harmony": [(10, r"\n- (?:Chords by part of the song|Against the harmony|Where that changes|"
                                r"Where nobody sings|The lead line is carried by)[^\n]*"),
                           (10, r"\n- (?:Melody and hooks, by ear|Shape of the tune|Chords by ear)[^\n]*")],
    "Who's singing": [(10, r" That's later than usual[^.\n]*\.")],
}


def _grid_block(g, title=None):
    head = title or (f"{g['label']} — " if g["label"] else "")
    return (f"{head}{g['meta']}\n```\n" + "\n".join(g["rows"]) + "\n```")


# =============================================================================
# Rendering (2.0): listening notes rather than a report
# =============================================================================
#
# A model mirrors the register of what it reads. A dossier laid out as labelled
# detector readings ("Identified on the separated other part (audio
# classifier): ...") gets read back as a report. So the dossier now settles the
# evidence itself and writes the result the way a listener remembers a song:
# a few sentences on the track, who is singing, how it unfolds from start to
# finish (with the lyrics placed where they are sung), then the sound, groove,
# voice, melody and era in plain words. Conflicts between sources are resolved
# here and only the outcome is shown; the few that a model still gets wrong
# are listed once, briefly, under "Easy to get wrong". Exact figures sit at the
# end for when someone asks for a number.

LEVEL_RANK = {"not playing": 0, "quiet/absent": 1, "restrained": 2, "moderate": 3, "steady": 3,
              "prominent": 4, "peak": 5}

_NOTE_RE = re.compile(r"^([A-Ga-g])([#♯b♭]?)(-?\d)$")


def note_midi(name):
    """'A♯3' -> 58; None when unreadable."""
    m = _NOTE_RE.match(str(name or "").strip())
    if not m:
        return None
    base = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}[m.group(1).upper()]
    acc = {"#": 1, "♯": 1, "b": -1, "♭": -1}.get(m.group(2), 0)
    return (int(m.group(3)) + 1) * 12 + base + acc


def pretty_note(name):
    return str(name or "").replace("#", "♯")


def join_words(items, conj="and"):
    items = [x for x in items if x]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + f" {conj} " + items[-1]


def cap(s):
    s = str(s or "")
    return s[:1].upper() + s[1:]


def sentence(s):
    s = " ".join(str(s or "").split()).strip()
    if not s:
        return ""
    s = cap(s)
    return s if s[-1] in ".!?…" else s + "."


def _base_name(name):
    return re.sub(r"\s*\d+\s*$", "", str(name or "")).strip()


def _a_an(phrase):
    p = str(phrase or "").strip()
    # note names and letters said aloud: "an F Mixolydian", "an A minor", "an 808"
    if re.match(r"^[AEFHILMNORSX](?:[#♯b♭]?\b|[#♯b♭]?\s)", p) or re.match(r"^(?:8|11|18)\d*\b", p):
        return "an " + p
    return ("an " if p[:1].lower() in "aeiou" else "a ") + p


# ---- instrument tags ----------------------------------------------------------
_TAG_ITEM_RE = re.compile(r"(?:^|(?<=\]), |(?<=\)), )\s*(.+?) \((\d+)%[^)]*\)(?: \[([^\]]*)\])?")


def parse_tags(s):
    """'strings (violin/fiddle) (56%, strongest near 55s) [...; sustained: 14/44 windows], ...'
    -> [{'label', 'pct', 'extent', 'supported'}]."""
    out = []
    for m in _TAG_ITEM_RE.finditer(str(s or "")):
        note = m.group(3) or ""
        ext = re.search(r"\b(sustained|intermittent|isolated)\b", note)
        out.append({"label": m.group(1).strip(), "pct": int(m.group(2)),
                    "extent": ext.group(1) if ext else "",
                    "supported": "supported by whole-mix" in note})
    return out


def confident_tags(s, min_pct=40):
    return [t for t in parse_tags(s)
            if t["supported"] or t["pct"] >= min_pct or (t["extent"] == "sustained" and t["pct"] >= 30)]


def _tag_word(label):
    lab = label.lower()
    lab = re.sub(r"\s*\((?:unspecified)\)", "", lab)
    lab = lab.replace("strings (violin/fiddle)", "strings").replace("(violin/fiddle)", "")
    return " ".join(lab.split())


# ---- what to call each separated part ---------------------------------------------
def part_names(info):
    """{stem: friendly name} for the parts worth talking about, plus a set of
    stems that are only faint / residue."""
    names, faint = {}, set()
    residue = set(info.get("intensity_whole") or {})
    ids = info.get("stem_ids") or {}
    tags = info.get("stem_tags") or {}
    minor = info.get("intensity_minor_db") or {}
    if info.get("vocals_present") != "no":
        names["vocals"] = "the vocal"
    if not info.get("no_kit"):
        names["drums"] = "the drums"
    if "bass" not in residue:
        bt = [_tag_word(t["label"]) for t in parse_tags(tags.get("bass"))]
        not_bass_guitar = "does not identify a bass guitar" in ids.get("bass", "") or bool(info.get("low_end_note"))
        if not_bass_guitar and any("synth" in t for t in bt):
            names["bass"] = "the synth bass"
        elif not_bass_guitar:
            names["bass"] = "the low end"
        else:
            names["bass"] = "the bass"
    for stem, plain, vague in (("guitar", "the guitar", "the guitar-range part"),
                               ("piano", "the piano", "the piano-range part")):
        if stem in residue:
            continue
        if minor.get(stem, 0) >= 20:
            faint.add(stem)
            continue
        confirmed = not ids.get(stem) and not (stem == "guitar" and "inconclusive" in (info.get("guitar") or ""))
        names[stem] = plain if confirmed else vague
    if "other" not in residue:
        if minor.get("other", 0) >= 20:
            faint.add("other")
        else:
            names["other"] = "the keys/synth layer"
    return names, faint


# ---- groove in plain words ----------------------------------------------------------
_GROOVE_PLAIN = [
    (r"^four-on-the-floor", "a four-on-the-floor kick (one on every beat)"),
    (r"^mostly four-on-the-floor", "a mostly four-on-the-floor kick"),
    (r"^syncopated kick at roughly one hit per beat", "a syncopated kick that lands between the beats as often as on them"),
    (r"^kick on every other beat", "a kick on every other beat (1 and 3)"),
    (r"^kick mostly on every other beat", "a kick mostly on 1 and 3"),
    (r"^eighth-note / double-time kick", "a double-time kick, two hits a beat"),
    (r"^mostly eighth-note / double-time kick", "a mostly double-time kick"),
    (r"^syncopated / not locked", "a syncopated kick that isn't locked to a simple grid"),
    (r"^loosely on a beat grid", "a loose, slightly syncopated kick"),
    (r"^kick pattern with moderate grid-lock", "a kick that moves around more than a plain on-beat pattern"),
    (r"^sparse kick activity", "barely any kick"),
    (r"^fast kick activity", "a lot of fast kick work"),
    (r"^moderate kick activity", "a moderately busy kick"),
    (r"^clear backbeat snare", "a clear backbeat snare"),
    (r"^snare often off the classic backbeat", "a snare that often sits off the usual backbeat"),
    (r"^mixed backbeat", "a snare that mixes backbeat and off-beat hits"),
    (r"^snare on eighth-note subdivisions", "a busy, rolling snare on eighth notes"),
    (r"^snare on every beat", "a snare on every beat"),
    (r"^syncopated snare/clap repeating every two beats", "a syncopated, dembow-style snare or clap between the kicks"),
    (r"^snare/clap on alternate beats over a kick on every beat", "a snare or clap on 2 and 4 over it"),
    (r"^classic backbeat", "a classic backbeat (snare on 2 and 4)"),
    (r"^mostly classic backbeat", "a mostly classic backbeat"),
    (r"^snare on a backbeat-rate grid", "a backbeat-style snare"),
    (r"^backbeat-rate snare", "a snare on every other beat"),
    (r"^snare pattern with moderate grid-lock", "a looser, busier snare pattern"),
    (r"^busy closed-hat or 16th-note hat", "busy sixteenth-note hi-hats"),
    (r"^regular 8th-note hat layer", "steady eighth-note hi-hats"),
    (r"^quarter-note hat pulse", "a hi-hat on each beat"),
    (r"^open or sparse hat", "only sparse hi-hat"),
    (r"^high hat hit rate", "a busy top end of hats"),
    (r"^minimal hat activity", "hardly any hi-hat"),
    (r"^noticeable tom/fill activity", "noticeable tom fills"),
    (r"^frequent cymbal crashes", "frequent crashes and rides"),
    (r"^straight \(unswung\)", "a straight, unswung feel"),
    (r"^straight$", "a straight, unswung feel"),
]


def split_top(text, sep=";"):
    """Split on `sep` outside parentheses."""
    out, buf, depth = [], [], 0
    for ch in str(text or ""):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(0, depth - 1)
        if ch == sep and depth == 0:
            out.append("".join(buf))
            buf = []
            continue
        buf.append(ch)
    out.append("".join(buf))
    return [p.strip() for p in out if p.strip()]


def groove_plain(hint):
    parts = split_top(hint)
    out = []
    for p in parts:
        for rx, rep in _GROOVE_PLAIN:
            if re.search(rx, p, re.I):
                out.append(rep)
                break
        else:
            p2 = drop_parens(p, ("",)).strip()
            if p2.endswith(" feel"):
                p2 = _a_an(p2)
            if p2:
                out.append(p2)
    return join_words(out)


def _hat_plain(h):
    h = str(h or "").lower()
    if "16th" in h or "very dense" in h:
        return "busy sixteenth-note hi-hats"
    if "8th" in h:
        return "eighth-note hi-hats"
    if "sparse" in h or "on the beat" in h:
        return "sparse hi-hats"
    return clean_label(h)


# ---- drum timeline --------------------------------------------------------------------
def _secs(t):
    try:
        mm, ss = str(t).split(":")
        return int(mm) * 60 + int(ss)
    except (TypeError, ValueError):
        return None


def drum_timeline(info):
    """[(start_s, end_s, config)] from the DRUM ARRANGEMENT TIMELINE lines."""
    out = []
    for l in (info.get("drums") or {}).get("timeline") or []:
        m = re.match(r"^\s*(\d+:\d\d)-(\d+:\d\d):\s*(.+)$", l.strip())
        if m:
            out.append((_secs(m.group(1)), _secs(m.group(2)), m.group(3).strip()))
    return out


def drum_config_plain(cfg):
    c = cfg.lower()
    if c.startswith("full kit"):
        return "the full kit"
    if c.startswith("no drums"):
        return "no drums"
    pieces = re.findall(r"kick|snare|hi-hat|toms|cymbals|perc", c.split("(")[0])
    if not pieces:
        return cfg
    if len(pieces) == 1:
        return f"just the {pieces[0]}"
    missing = [p for p in ("kick", "snare") if p not in pieces]
    return join_words(pieces) + (f" (no {missing[0]})" if missing else "")


# ---- arrangement change points -------------------------------------------------------
def change_points(info, names):
    """[(time_s, entering[], leaving[])] for change points with 2+ parts moving."""
    out = []
    for l in info.get("changes") or []:
        m = re.match(r"^(\d+:\d\d) \((\d+) stems?\): (.+)$", l.strip())
        if not m:
            continue
        ins, outs = [], []
        for tok in [t.strip() for t in m.group(3).split(",")]:
            mm = re.match(r"^([a-z]+)(?::([a-z-]+))? (in|out)$", tok)
            if not mm:
                continue
            stem, piece, d = mm.groups()
            if stem == "drums" and piece:
                nm = {"hihat": "the hats", "hi-hat": "the hats"}.get(piece, f"the {piece}")
            elif stem in names:
                nm = names[stem]
            else:
                continue
            (ins if d == "in" else outs).append(nm)
        both = set(ins) & set(outs)
        ins = [x for x in ins if x not in both]
        outs = [x for x in outs if x not in both]
        if len(ins) + len(outs) >= 2:
            out.append((_secs(m.group(1)), ins, outs))
    return out


# ---- lyrics: place the file's own lines on the timed transcription --------------------
def _words(s):
    return re.findall(r"[a-z0-9']+", str(s or "").lower().replace("’", "'"))


def tagged_lyric_lines(text):
    """[(section header or '', line)] from tag lyrics."""
    out, head = [], ""
    for raw in str(text or "").splitlines():
        l = raw.strip()
        if not l:
            continue
        m = re.match(r"^\[([^\]]{1,40})\]$", l)
        if m:
            head = re.sub(r"\s*\d+\s*$", "", m.group(1)).strip()
            continue
        if len(_words(l)) >= 2:
            out.append((head, l))
    return out


def _align_line(cw, lw):
    """In-order match of lyric line words `lw` inside chunk words `cw`
    (longest common subsequence): (matched count, first index, last index + 1).
    Out-of-order overlaps ("i just need somebody to love i don't need too" is
    not "I don't need too much, just somebody to love") don't count."""
    n, m = len(lw), len(cw)
    if not n or not m:
        return (0, 0, 0)
    L = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n - 1, -1, -1):
        for j in range(m - 1, -1, -1):
            L[i][j] = L[i + 1][j + 1] + 1 if lw[i] == cw[j] else max(L[i + 1][j], L[i][j + 1])
    i = j = 0
    idx = []
    while i < n and j < m:
        if lw[i] == cw[j] and L[i][j] == L[i + 1][j + 1] + 1:
            idx.append(j)
            i += 1
            j += 1
        elif L[i + 1][j] >= L[i][j + 1]:
            i += 1
        else:
            j += 1
    if not idx:
        return (0, 0, 0)
    return (len(idx), idx[0], idx[-1] + 1)


def place_lyrics(info):
    """[(time_s, header, line, exact)]: each timed transcription chunk split
    into the file's own lyric lines it sings, each timed by where its first
    word falls inside the chunk (exact=True); or the rough transcription
    itself (exact=False) when nothing matches or there are no tag lyrics."""
    rp = info.get("replay") or {}
    chunks = rp.get("lyric_lines") or []
    if not chunks or info.get("vocals_present") == "no":
        return []
    tagged = tagged_lyric_lines(info["lyrics"]) if info.get("lyrics_src") == "tags" else []
    tagged_w = [(h, l, _words(l)) for h, l in tagged]
    out = []
    for a, z, text in chunks:
        text = strip_hallucinations(text).strip()
        cw = _words(text)
        if not cw:
            continue
        found = []
        for h, line, lw in tagged_w:
            if len(lw) < 2:
                continue
            got, j0, j1 = _align_line(cw, lw)
            cov = got / len(lw)
            if j1 - j0 > len(lw) + 3:
                continue                      # scattered words, not this line
            if (cov >= 0.6 and got >= 3) or (cov == 1.0 and got >= 2 and len(cw) <= 4):
                found.append((got + cov, j0, j1, h, line))
        found.sort(key=lambda f: -f[0])
        taken, used = [], set()
        for sc, j0, j1, h, line in found:
            span = set(range(j0, j1))
            if len(span & used) > 1:
                continue
            used |= span
            taken.append((j0, h, line))
        if taken:
            dur = max(0.0, float(z) - float(a))
            for j0, h, line in sorted(taken):
                out.append((float(a) + dur * j0 / max(1, len(cw)), h, line, True))
        else:
            out.append((float(a), "", text, False))
    out.sort(key=lambda x: x[0])
    # one entry per line in a row
    ded = []
    for x in out:
        if ded and x[3] and ded[-1][3] and ded[-1][2] == x[2] and x[0] - ded[-1][0] < 4:
            continue
        ded.append(x)
    return ded


# ---- melody per section -----------------------------------------------------------------
_CONTOUR_PLAIN = {
    "moves up and down without a single overall direction": "wanders up and down",
    "arches up then falls back": "arches up and falls back",
    "dips then rises": "dips and then rises",
    "falls steadily": "falls away",
    "climbs steadily": "climbs steadily",
    "stays around one pitch centre": "stays around one note",
    "short line": "",
}


def melody_for(info, stem, name):
    """{'lo','hi','contour','motion'} from MELODY PER SECTION for one section."""
    for l in (info.get("melody") or {}).get(stem) or []:
        m = re.match(r"^(.+?) \((\d+:\d\d)-(\d+:\d\d)\): (.+)$", l.strip())
        if not m or m.group(1).strip() != name:
            continue
        body = m.group(4)
        if body.startswith("no sustained"):
            return {}
        out = {}
        r = re.search(r"pitch range: (\S+)-(\S+?) \(practical\)", body) or re.search(r"pitch range: (\S+)-(\S+?)[;,]", body)
        if r:
            out["lo"], out["hi"] = r.group(1), r.group(2)
        c = re.search(r"contour: ([^;]+)", body)
        if c:
            out["contour"] = _CONTOUR_PLAIN.get(c.group(1).strip(), c.group(1).strip())
        mo = re.search(r"melodic motion: ([^(;]+)", body)
        if mo:
            out["motion"] = mo.group(1).strip()
        return out
    return {}


def melody_extras(info, stem):
    """(recurring figures text, (highest note, time_s))"""
    rec, high = "", None
    for l in (info.get("melody") or {}).get(stem) or []:
        s = l.strip()
        if s.startswith("Recurring figures"):
            rec = s.partition(":")[2].strip()
        m = re.match(r"^Highest point of the \w+ line: (\S+) at (\d+:\d\d)", s)
        if m:
            high = (m.group(1), _secs(m.group(2)))
    return rec, high


# ---- sentences about the whole track ---------------------------------------------------
def genre_phrase(info):
    ge = info.get("genre_era") or {}
    if ge:
        # dev17 GENRE ERA CHECK: the recommended label is newer than the tagged
        # year; lead with a ranked label from the record's own time.
        fits = ge.get("fits") or []
        if fits:
            return fits[0]
        # an undated ranked label ("folk", "pop", "rock") is not from a later decade either
        for lab in _ranked_labels(info.get("raw_analysis") or "", max_n=8):
            if lab.lower() != ge["rec"].lower() and _genre_onset(lab) is None:
                return lab
        return "rock" if _genre_family(ge["rec"]) == "rockish" else "pop"
    g = (info.get("genre") or "").strip()
    if not g or g.lower().startswith("unavailable"):
        lab = _ranked_labels(info.get("raw_analysis") or "")
        return lab[0] if lab else ""
    m = re.match(r"^(.*?)\s*\(with (.+?) lean\)\s*$", g)
    if m:
        lean = m.group(2).strip()
        return f"{m.group(1).strip()} with {_a_an(lean)} edge"
    m = re.search(r"Default conversational framing: '([^']+)' with possible '([^']+)' lean", info.get("genre_reasoning") or "")
    if m and m.group(2).lower() not in g.lower():
        return f"{g}, leaning toward {m.group(2)}"
    return g


def neighbour_styles(info):
    """The listening model's other ranked genres, when the genre check didn't replace them."""
    if info.get("genre_replaced"):
        return []
    g = (info.get("genre") or "").lower()
    if info.get("genre_era"):
        g = genre_phrase(info).lower()
    norm = lambda x: _HYPHENS_RE.sub(" ", x.lower()).strip()
    said = {norm(g)}
    ml = re.match(r"^(.*?)\s*\(with (.+?) lean\)\s*$", g)
    if ml:
        said |= {norm(ml.group(1)), norm(ml.group(2))}
    mlean = re.search(r"Default conversational framing: '([^']+)' with possible '([^']+)' lean",
                      info.get("genre_reasoning") or "")
    if mlean and mlean.group(2).lower() not in g:      # only when the Style phrase names that lean
        said |= {norm(mlean.group(2))}
    # 2.3: the label Musiclyse's style-led / regional genre check replaced is not a neighbour
    mrep = re.search(r"\(style-led\).*?the earlier '([^']+)' (?:reading|label)", info.get("genre_reasoning") or "")
    if mrep:
        said |= {norm(mrep.group(1)), norm(re.sub(r"\s*\(with .*?\)", "", mrep.group(1)))}
    ranked = _ranked_labels(info.get("raw_analysis") or "")
    labs = [l for l in ranked if norm(l) not in said]
    if info.get("synth_rock"):
        # 2.2.3: the synth comparisons only (synth-pop, new wave), not dance-pop or electronic pop
        return [l for l in labs if re.search(r"(?i)synth|new[ -]?wave", l)][:5]
    adj = info["mf"].get("GENRE_ADJACENT", "")
    adj_items = [re.sub(r"\s*\((?:confidence|conf)\s*[:=][^)]*\)", "", part).strip(" .")
                 for part in re.split(r"\s*[;,]\s*|\s*\d\)\s*", adj)]
    if MOSS_LOW_TRUST_FIELDS:
        # 2.2.3: MOSS-Music's stock labels and non-genre words are not neighbours
        if stock_genre_list(ranked) or stock_genre_list([x for x in adj_items if x]):
            labs = [l for l in labs if not _stock_label(l)]
            adj_items = [x for x in adj_items if not _stock_label(x)]
        adj_items = [x for x in adj_items if _any_marker_present(_normalize_genre_token(x), _GENRE_ROOT_WORDS)]
        ym = re.search(r"\b(19[0-9]{2}|20[0-9]{2})\b", str(info.get("year") or ""))
        if info.get("genre_stock") and ym:
            # the era check that the stock label used to trigger, for the neighbours
            labs = [l for l in labs if (_genre_onset(l) or 0) <= int(ym.group(1))]
            adj_items = [x for x in adj_items if (_genre_onset(x) or 0) <= int(ym.group(1))]
    for p in adj_items:
        if (p and p.lower() not in g and norm(p) not in said and norm(p) not in {norm(x) for x in labs}
                and not is_non_answer(p)):
            labs.append(p)
    ge = info.get("genre_era") or {}
    if ge:
        labs = [l for l in labs if (_genre_onset(l) or 0) <= ge["year"]]
    return labs[:5]


def _main_label(info, gp):
    """2.3: the genre phrase without its 'leaning toward ...' / '(with ... lean)' tail."""
    g = re.sub(r",?\s*(?:leaning toward|with an? .*? (?:lean|edge)).*$", "", str(gp or "")).strip()
    return g or gp


def style_words(info):
    """2.3: (related, other) style labels from the measured styles (dev42),
    leaving out the genre itself and its neighbours."""
    g = (info.get("genre") or "").lower()
    base = re.sub(r"\s*\(with .*?\)", "", g).strip()
    fam = _mv_style_families(base) | _mv_style_families(g)
    said = {_mv_style_norm(base), _mv_style_norm(g)}
    try:
        said |= {_mv_style_norm(x) for x in neighbour_styles(info)}
    except Exception:
        pass
    related, other = [], []
    for lab, _p in info.get("styles") or []:
        n = _mv_style_norm(lab)
        if n in said or not n:
            continue
        said.add(n)
        (related if (_mv_style_families(lab) & fam or not fam) else other).append(lab)
    return related[:4], other[:3]


def loudness_words(info):
    try:
        lufs = float(info.get("lufs"))
    except (TypeError, ValueError):
        return ""
    if lufs >= -10.0:
        return "a loud master, as loud as today's pop releases" if lufs <= -6 \
            else "a very loud master, louder than most of today's releases"
    if lufs > -12.0:
        return "a fairly loud master, a little quieter than most current releases"
    if lufs > -15.0:
        return "a moderately loud master with some breathing room, quieter than current releases"
    return "a quiet, dynamic master by today's standards"


def _prod_line(info, title, prefix):
    for t, lines in info.get("production") or []:
        if t == title:
            for l in lines:
                if l.lower().startswith(prefix) and ":" in l:      # a cut-off line has no value
                    return l.split(":", 1)[1].strip()
    return ""


def _prod_num(info, rx):
    for _t, lines in info.get("production") or []:
        for l in lines:
            m = re.search(rx, l)
            if m:
                try:
                    return float(m.group(1))
                except ValueError:
                    return None
    return None


def production_words(info):
    bits = []
    lw = loudness_words(info)
    if lw:
        bits.append(lw)
    amb = _prod_line(info, "Ambience / reverb", "ambience reading")
    if amb:
        # Prefer plain listening language over stock measurement labels
        # ("very washy/reverb-heavy, long tails") that stock models otherwise
        # echo in every answer. Unknown labels still fall through as before.
        plain = plain_ambience(amb)
        if plain:
            bits.append(plain)
        else:
            main_amb = re.split(r"\s*[—;(]\s*", amb)[0].strip()
            if main_amb:
                bits.append(f"{_a_an(main_amb)} sound" if not main_amb.endswith(("sound", "mix")) else _a_an(main_amb))
    width = _prod_line(info, "Stereo image", "stereo width reading")
    if width:
        plain_w = plain_stereo_width(width)
        if plain_w:
            bits.append(plain_w)
        else:
            # Fallback for any label the map doesn't cover (same shape as before).
            w = re.split(r"\s*[—;(]\s*", width)[0].strip()
            w = w.replace(" width", "")
            bits.append(f"{_a_an(w)} image" if w.endswith("stereo") else f"{_a_an(w)} stereo image")
    sub = _prod_num(info, r"^\s*sub-bass \(20-60 Hz\): ([\d.]+)%")
    if sub is not None and sub >= 25:
        bits.append(f"a heavy low end (about {sub:.0f}% of the energy is deep sub-bass)")
    pump = _prod_line(info, "Sidechain / pumping", "sidechain reading")
    if pump and "ducks strongly" in pump and "recovers between beats" in pump:
        # Avoid the fixed tag "clear sidechain pumping" that models parrot.
        bits.append("the mix ducks hard with the beat (sidechain-style pumping)")
    return bits


def drums_programmed(info):
    """True when the drums sound programmed: a sustained drum-machine tag on the
    drum part plus very repeatable kick hits."""
    tags = parse_tags((info.get("stem_tags") or {}).get("drums"))
    dm = any("drum machine" in t["label"].lower() and (t["extent"] == "sustained" or t["pct"] >= 40) for t in tags)
    cons = None
    for lab, val in (info.get("drums") or {}).get("sound") or []:
        if lab == "Kick consistency":
            m = re.match(r"([\d.]+)", val)
            cons = float(m.group(1)) if m else None
    return dm and (cons is None or cons >= 0.95)


def other_layer_words(info):
    """What the keys/synth layer seems to be."""
    tags = confident_tags((info.get("stem_tags") or {}).get("other"))
    words = [_tag_word(t["label"]) for t in tags]
    poly = None
    for l in (info.get("stem_reports") or {}).get("other") or []:
        m = re.search(r"average simultaneous notes: ([\d.]+)", l)
        if m:
            poly = float(m.group(1))
    desc = "a chordal keys/synth layer" if (poly and poly >= 3) else "a keys/synth layer"
    if info.get("reed_pad_poly"):
        desc += (" (most likely a synth pad or sustained keyboard chords"
                 + (f"; parts of it come across as {join_words(words, 'or')}" if words else "") + ")")
    elif words:
        desc += f" (parts of it come across as {join_words(words, 'or')})"
    return desc


def bass_words(info, names):
    if "bass" not in names:
        return ""
    nm = names["bass"].replace("the ", "")
    rec, _h = melody_extras(info, "bass")
    if "octave" in rec:
        return f"{_a_an(nm)} bouncing between octaves"
    return _a_an(nm)


def _is_male(tag):
    return tag in MALE_TAGS


def _is_female(tag):
    return tag in FEMALE_TAGS


def singer_gender(info):
    ident = info.get("identity")
    if ident and ident not in ("unknown", "uncertain"):
        return "male" if _is_male(ident) else "female" if _is_female(ident) else ""
    ac = info.get("acoustic_lead")
    return "male" if _is_male(ac) else "female" if _is_female(ac) else ""


def scrub_gender(text, gender):
    """By-ear wording that names the other gender (an unbroken boy's voice
    heard as a woman's) is rewritten to neutral words."""
    t = str(text or "")
    if gender == "male":
        # Quoted words (a lyric the description cites) are left as they are.
        parts = re.split(r"(“[^”]*”|\"[^\"]*\")", t)
        for i in range(0, len(parts), 2):
            p = parts[i]
            # "The lead singer is a woman, and ..." -> "... is a high voice, and ..."
            p = re.sub(r"(?i)\b(an?)\s+(?:woman|girl)\b(?!['’])", "a high voice", p)
            p = re.sub(r"\b(?:female|woman's|women's|woman|girl's|girl)\b", "high", p, flags=re.I)
            p = re.sub(r"\bhigh adult\b", "high", p)
            # Female voice types ("a high mezzo-soprano", "in the alto range"); not
            # an alto or soprano sax.
            _vt = r"(?:mezzo[\s\-‐-―]*sopranos?|sopranos?|contraltos?|altos?)"
            _sax = r"(?![\s\-‐-―]*(?:sax|saxophone|recorder|clarinet|flute|horn)s?\b)"
            p = re.sub(r"(?i)\b" + _vt + r"(\s+(?:range|register|territory|tessitura))\b" + _sax, r"high\1", p)
            p = re.sub(r"(?i)\b" + _vt + r"\b" + _sax, "voice", p)
            p = re.sub(r"(?i)\b(high)(?:\s+high)+\b", r"\1", p)
            p = re.sub(r"(?i)\bvoice(\s+voice)+\b", "voice", p)
            # Her / she for the (male) lead.
            p = re.sub(r"\bShe\b", "He", p)
            p = re.sub(r"\bshe\b", "he", p)
            p = re.sub(r"\b([Hh])erself\b", lambda m_: m_.group(1) + "imself", p)
            p = re.sub(r"\b([Hh])ers\b", lambda m_: m_.group(1) + "is", p)
            p = re.sub(r"\b([Hh])er(?=\s+[A-Za-z])", lambda m_: m_.group(1) + "is", p)
            p = re.sub(r"\b([Hh])er\b", lambda m_: m_.group(1) + "im", p)
            parts[i] = p
        t = "".join(parts)
    elif gender == "female":
        t = re.sub(r"\b(?:male|man's|men's|man|boy's|boy)\b", "", t, flags=re.I)
    return " ".join(t.split())


def _clean_vocal_demographics(text):
    """Collapse runaway 'no overt X' loops some saves put in VOCAL_DEMOGRAPHICS."""
    t = str(text or "")
    if not t:
        return t
    # Cut after the first long run of "no overt ..." clauses (dev60+ model loops).
    m = re.search(
        r"((?:no overt [a-z\- ]+(?:;\s*)?){3,})",
        t, re.I,
    )
    if m and m.start() > 20:
        # Keep everything before the loop; drop the loop and after.
        t = t[:m.start()].rstrip(" ;,.")
    elif m and m.start() <= 20:
        # Almost the whole field is the loop — keep only positive descriptors before it.
        t = t[:m.start()].rstrip(" ;,.")
    # Drop residual "no overt ..." fragments and empty gender/timbre disclaimers.
    parts = []
    for part in re.split(r"\s*;\s*", t):
        p = part.strip(" .")
        if not p:
            continue
        if re.match(r"(?i)^no overt\b", p):
            continue
        if re.match(r"(?i)^(?:gender|age)\b", p):
            continue
        parts.append(p)
    return "; ".join(parts)


def voice_timbre_words(info):
    """Timbre/delivery words from the vocal pass, without its gender/age guesses."""
    vp = info.get("vocal_pass") or {}
    out = []
    for key in ("VOCAL_DEMOGRAPHICS", "TIMBRE_NOTE", "PITCH_NOTE", "DELIVERY_NOTE", "PROCESSING_NOTE"):
        v = vp.get(key) or (info.get("vocal_notes") or {}).get(key) or ""
        if key == "VOCAL_DEMOGRAPHICS":
            v = _clean_vocal_demographics(v)
        for part in re.split(r"\s*;\s*", v):
            p = part.strip(" .")
            if not p or re.match(r"(?i)^(?:gender|age)\b", p):
                continue
            dn = info.get("pitch_discarded_note") if BY_EAR_PITCH_FOLLOWS_CHECK else ""
            if dn and key == "PITCH_NOTE" and (
                    re.search(r"(?<![A-G#♯b♭])" + re.escape(dn) + r"\b", p)
                    or any(re.search(r"(?<![\d.])" + re.escape(f) + r"(?:\.0+)?(?![\d])", p)
                           for f in info.get("pitch_discarded_figures") or [])):
                continue                 # 2.2.3: quotes the tracker figures the pitch check set aside
            if re.match(r"(?i)^no overt\b", p):
                continue
            if key == "DELIVERY_NOTE" and re.search(r"doubling|single vocalist", p, re.I):
                continue
            # "timbre thin, bright" -> "thin, bright timbre"
            mm = re.match(r"^(timbre|articulation|phrasing|processing|delivery|tone|vibrato)\s+([^(]+?)\s*(\(.*\))?$", p, re.I)
            if mm:
                p = f"{mm.group(2)} {mm.group(1).lower()}" + (f" {mm.group(3)}" if mm.group(3) else "")
            key_words = frozenset(re.findall(r"[a-z]+", re.sub(r"\(.*?\)", "", p.lower())))
            if any(key_words <= frozenset(re.findall(r"[a-z]+", x.lower())) for x in out):
                continue
            out = [x for x in out if not frozenset(re.findall(r"[a-z]+", re.sub(r"\(.*?\)", "", x.lower()))) <= key_words]
            out.append(p)
    return out


def era_evidence_filtered(info, text=None):
    """The era pass's evidence clauses, without the ones the measurements
    contradict (pumping when none was measured, trap programming over a
    four-on-the-floor kick, hi-hat rolls over plain eighth-note hats) and
    without remarks about the analysis itself."""
    ev = text if text is not None else (info.get("era") or {}).get("EVIDENCE", "")
    if not ev:
        return []
    pump = _prod_line(info, "Sidechain / pumping", "sidechain reading")
    measured_pump = bool(pump and "ducks strongly" in pump)
    d = info.get("drums") or {}
    fotf = "four-on-the-floor" in (d.get("kick") or "")
    hats16 = "16th" in (d.get("hats") or "") or "very dense" in (d.get("hats") or "")
    no_kit = bool(info.get("era_drum_note") or info.get("no_kit"))

    def _contradicted(c):
        lc = c.lower()
        if no_kit and _DRUM_WORD_RE.search(c):
            return True                   # dev17 ERA NOTE (drums): there is no kit
        if re.search(r"sidechain|pumping|ducking", lc) and not measured_pump:
            return True
        if "trap" in lc and fotf:
            return True
        if re.search(r"hi-?hat rolls?", lc) and not hats16:
            return True
        if re.search(r"source separation|analogue workflows|analog workflows", lc):
            return True
        return False

    def _salvage(c):
        """A contradicted clause that is a plain list of short noun phrases
        ("use of sidechain compression, digital clarity, modern synth
        textures") keeps its other items; anything with a verb or a longer
        phrase ("the bass is a warm, pulsing synth line with sidechain ...")
        is dropped whole, as before."""
        body = c.strip().rstrip(".")
        items = [x.strip() for x in re.split(r",\s*(?:and\s+)?|\s+and\s+", body)] if "," in body else []
        if (len(items) >= 3 and all(x and len(x.split()) <= 5 for x in items)
                and not any(re.search(r"(?i)\b(?:is|are|was|were|has|have|had|gives?|sounds?|suggest\w*|"
                                      r"feature\w*|creat\w*|with)\b", x) for x in items)):
            kept = [re.sub(r"(?i)^use of\s+", "", x) for x in items if not _contradicted(x)]
            return join_words(kept) if kept else ""
        return ""

    if ";" not in ev:
        # Prose (as MOSS-Music often writes it): filter sentence by sentence and
        # keep the rest as one passage, as it was written.
        sents = [x for x in re.split(r"(?<=[.!?])\s+", ev.strip()) if x.strip()]
        kept = []
        for x in sents:
            if x.rstrip().endswith("…"):
                continue
            if _contradicted(x):
                sv = _salvage(x)
                if sv:
                    kept.append(sv + ("." if len(sents) > 1 else ""))
                continue
            kept.append(x)
        txt = " ".join(kept).rstrip(" .…")
        return [txt] if txt else []

    out = []
    for c in re.split(r"\s*;\s*", ev.rstrip(" .…")):
        if not c.strip() or c.endswith("…"):
            continue
        if _contradicted(c):
            sv = _salvage(c)
            if sv:
                out.append(sv)
            continue
        out.append(c.strip())
    return out


# ---- saved readings that 2.1 parsed but never wrote out (2.2) ---------------------------
_ARRANGEMENT_PLAIN = {
    "solo_lead": "a single lead voice", "single_lead": "a single lead voice",
    "lead_plus_backing": "one lead voice with backing vocals",
    "duet_co_leads": "two co-lead singers", "call_response": "call-and-response between voices",
    "group_unison": "a group singing together in unison", "choir": "a choir",
}
_GENRE_HEDGE_PLAIN = {
    "the listening model and the audio classifier leaned different ways":
        "the impressions by ear and the audio classifier leaned different ways, so it's a broad call",
    "two independent sources agree": "two independent readings agree on it",
    "revised toward electronic/dance from an initial rock-leaning read":
        "a first rock-leaning impression was revised toward electronic/dance",
    "taken from the listening model's own description": "taken from the description by ear",
}


def genre_hedge_words(info):
    h = info.get("genre_hedge") or ""
    if not h or info.get("genre_era") or info.get("genre_replaced"):
        return ""
    if h.startswith("weak"):
        return "a weak call — it rests on one classifier tag that mixes up close rock/pop styles"
    return _GENRE_HEDGE_PLAIN.get(h, "")


# 2.2.2: genres that commonly fuse with the decided one are not "ruled out"
# (folk for a folk-rock track: the folk-rock test track's "What it isn't: folk").
_GENRE_PARTNERS = (
    (r"\b(?:rock|punk|grunge|metal|new wave)\b", ("folk", "country", "blues", "americana", "roots")),
    (r"\bfolk\b", ("rock", "pop", "country", "americana", "acoustic", "singer-songwriter")),
    (r"\bcountry\b", ("rock", "folk", "pop", "americana", "bluegrass")),
    (r"\bblues\b", ("rock", "soul", "r&b", "jazz")),
    (r"\b(?:soul|r&b|rnb|funk)\b", ("pop", "funk", "soul", "gospel", "r&b", "disco")),
    (r"\bjazz\b", ("blues", "soul", "funk")),
)


def _genre_partners(decided):
    d = _HYPHENS_RE.sub(" ", str(decided or "").lower())
    out = set()
    for rx, partners in _GENRE_PARTNERS:
        if re.search(rx, d):
            out |= set(partners)
    return out


def ruled_out_styles(info):
    """GENRE_RULED_OUT, minus anything the Style line or the checks call it."""
    v = info["mf"].get("GENRE_RULED_OUT", "")
    if not v or is_non_answer(v) or info.get("genre_replaced"):
        # a genre check that replaced the first impression discards its ruled-out list too
        return []
    norm = lambda x: _HYPHENS_RE.sub(" ", x.lower()).strip()
    keep_out = {norm(genre_phrase(info)), norm(info.get("genre") or "")}
    keep_out |= {norm(x) for x in neighbour_styles(info)}
    gr = info.get("genre_replaced") or {}
    if gr.get("rec"):
        keep_out.add(norm(gr["rec"]))
    out = []
    for part in re.split(r"\s*[;,]\s*|\s*\d\)\s*", v):
        p = re.sub(r"\s*\((?:confidence|conf)\s*[:=][^)]*\)", "", part).strip(" .")
        if not p or norm(p) in keep_out or any(norm(p) in k.split() for k in keep_out if k) or p in out:
            continue
        fam = _genre_family(p)
        if fam != "other" and fam in {_genre_family(x) for x in (genre_phrase(info), info.get("genre") or "") if x}:
            continue                     # "isn't electronic" next to a dance-pop Style line
        if norm(p) in _genre_partners(genre_phrase(info) + " " + (info.get("genre") or "")):
            continue                     # "isn't folk" next to a rock (folk-rock) Style line
        out.append(p)
    return out


def voice_arrangement(info):
    """(plain arrangement words, other-voices remark) from the multi-singer
    audit and the vocal passes."""
    vs = info.get("voices") or {}
    va = (vs.get("VOICE_ARRANGEMENT") or (info.get("vocal_extra") or {}).get("VOICE_ARRANGEMENT") or "").strip()
    arr = _ARRANGEMENT_PLAIN.get(va.split()[0].lower(), "") if va else ""
    ev = info.get("multi_voice_evidence") or (info.get("vocal_extra") or {}).get("MULTI_VOICE_EVIDENCE") or ""
    if not ev or is_non_answer(ev):
        # 0.5 (Music Flamingo) vocal pass: "VOCAL ARRANGEMENT: single lead voice with occasional backing harmonies"
        ev = (info.get("vocal_pass") or {}).get("VOCAL_ARRANGEMENT", "")
    ev = "" if (is_non_answer(ev) or ev.startswith("[")) else ev
    if ev and (info.get("lead_singers") == "several" or info.get("section_voice")) and re.search(
            r"(?i)\bsingle\b|\bonly one\b|\bone (?:lead|vocal)|lead vocal only|no (?:clear )?(?:duet|co-?lead|secondary)", ev):
        ev = ""                          # by-ear "one voice" against a shared lead in "Who's singing"
    # "lead_plus_backing" is also the passes' default answer; with no backing
    # voice described anywhere it is not evidence of backing vocals
    vp = info.get("vocal_pass") or {}
    no_backing = (is_non_answer(vp.get("BACKING_NOTE", "none")) and info.get("backing") in ("", "none", "uncertain", None)
                  and (not ev or re.search(r"(?i)\bonly one\b|\bsingle\b|lead vocal only|no (?:clear )?(?:harmon|backing|"
                                           r"secondary)", ev)))
    if va.startswith("lead_plus_backing") and no_backing:
        arr = "a single lead voice"
    # several leads only where "Who's singing" says so, never against the voice count check
    if va.startswith(("duet", "call_response")) and (info.get("lead_singers") != "several"
                                                     or info.get("voice_count_check")):
        arr = ""
    # ...and one lead only where it doesn't say the lead is shared
    if va.startswith(("solo", "single", "lead_plus")) and (info.get("lead_singers") == "several"
                                                           or info.get("section_voice")):
        arr = ""
    return arr, ev


# ---- harmony and the lead line (dev17 HARMONY AND MELODY BY SECTION) ------------------------
def _chord_words(chords):
    return join_words([re.sub(r"^([A-G][♯b]?)5$", r"\1 power chord", c.replace("#", "♯")) for c in chords])


def _majority(vals):
    vals = [v for v in vals if v]
    if not vals:
        return ""
    best = max(set(vals), key=lambda v: (vals.count(v), -vals.index(v)))
    return best if vals.count(best) * 2 >= len(vals) else ""


def _lead_part_name(info, lead):
    """'the voice' / 'the guitar stem' -> what the dossier calls that part."""
    m = re.match(r"the (\w+) stem$", str(lead or ""))
    if not m:
        return lead
    names, _faint = part_names(info)
    return names.get(m.group(1)) or {"other": "the keys/synth layer"}.get(m.group(1), f"the {m.group(1)}-range part")


def harmony_summary(info, hm):
    """Plain sentences from HARMONY AND MELODY BY SECTION: the chords each kind
    of section uses most, and where the lead line sits against them."""
    out = []
    groups, order = {}, []
    for r in hm:
        base = _base_name(r["name"]) or r["name"]
        if base not in groups:
            groups[base] = []
            order.append(base)
        groups[base].append(r)
    parts = []
    for base in order:
        rows = groups[base]
        score = {}
        for r in rows:
            if r.get("unreliable"):
                continue
            for k, c in enumerate(r["chords"]):
                score[c] = score.get(c, 0) + (4 - min(k, 3))
        if not score:
            continue
        top = [c for c, _v in sorted(score.items(), key=lambda kv: -kv[1])][:4]
        outside = sorted({c for r in rows for c in r.get("outside") or [] if c in top})
        label = base.lower()
        if len(rows) > 1:
            label += "es" if label.endswith(("s", "x", "ch", "sh")) else "s"
        if re.match(r"(?i)^(?:section|part|theme)", base):
            label = base
        parts.append(f"{label}: {_chord_words(top)}"
                     + (f" ({_chord_words(outside)} outside the key)" if outside else ""))
    if parts:
        out.append("Chords by part of the song (measured — the ones heard most, not an exact progression): "
                   + "; ".join(parts))
    led = [r for r in hm if r.get("lead")]
    if led:
        voice = [r for r in led if r["lead"] == "the voice"]
        rows = voice or led
        who = "the voice" if voice else "the lead line"
        reg = _majority([r["register"] for r in rows])
        gap = _majority([r["bass_gap"] for r in rows])
        match = _majority([r["match"] for r in rows])
        where = []
        if reg:
            where.append({"below the chords": "below the chords (the accompaniment sits higher than the tune)"}
                         .get(reg, reg))
        if gap:
            where.append(gap)
        sent = ""
        if where:
            sent = f"Against the harmony, {who} mostly sits " + ", ".join(where)
        if match:
            sent = (sent + ", and " + match) if sent else f"Against the harmony, {who} mostly {match}"
        if sent:
            out.append(sent)
        # sections that stand out from the usual relation
        odd = {}
        for r in rows:
            if match and r["match"] and r["match"] != match:
                odd.setdefault(r["match"], []).append(r["name"])
            elif reg and r["register"] and r["register"] != reg and r["register"] in ("well above the chords",
                                                                                       "below the chords"):
                odd.setdefault("sits " + r["register"], []).append(r["name"])
        if odd:
            out.append("Where that changes: " + "; ".join(f"in {join_words(v)} it {k}" for k, v in odd.items()))
        other = [r for r in led if r["lead"] != "the voice"]
        if other and voice:
            out.append("Where nobody sings, the tune is carried by "
                       + join_words(sorted({_lead_part_name(info, r["lead"]) for r in other}))
                       + " (" + ", ".join(r["name"] for r in other) + ")")
        elif other and not voice:
            out.append("The lead line is carried by "
                       + join_words(sorted({_lead_part_name(info, r["lead"]) for r in other}))
                       + " in " + join_words([r["name"] for r in other]))
    return out


def harmony_detail_lines(hm, info=None):
    """One line per section for the detail section."""
    lines = []
    for r in hm:
        bits = []
        if r["chords"]:
            ch = _chord_words(r["chords"])
            if r.get("unreliable"):
                ch += " (mostly outside the key — unreliable)"
            elif r.get("outside"):
                ch += f" ({_chord_words(r['outside'])} outside the key)"
            bits.append("chords " + ch)
        else:
            bits.append("no clear chords")
        if r["lead"]:
            who = _lead_part_name(info or {}, r["lead"])
            lead = f"lead line: {who}"
            if r["lo"] and r["hi"]:
                lead += f", {pretty_note(r['lo'])}–{pretty_note(r['hi'])}"
            bits.append(lead)
            bits.extend(x for x in (r["register"], r["bass_gap"], r["match"]) if x)
        else:
            bits.append("no clear lead line")
        lines.append(f"- {r['name']} ({mmss(r['start'])}–{mmss(r['end'])}): " + "; ".join(bits))
    return lines


# ---- the walkthrough -----------------------------------------------------------------------
_KIT_PIECES = ("the hats", "the kick", "the snare", "the toms", "the cymbals", "the perc")
# Lyric-sheet headers worth naming in the walkthrough ("the pre-chorus"). Verse,
# intro, outro and bridge headers are left out: the same words often sit under
# different headers ("<lyric line" is both a post-chorus and the outro).
_LYRIC_PART_WORDS = ("pre-chorus", "post-chorus", "chorus", "hook", "refrain")


def _v(items, sing, plur):
    """Verb agreeing with a list of part names ("the drums" and "the hats" are plural)."""
    if len(items) != 1:
        return plur
    nm = str(items[0])
    return plur if nm.endswith(("drums", "hats", "strings", "vocals")) else sing


def walkthrough(info, quote_lyrics=True):
    secs = info.get("sections") or []
    if not secs:
        return ""
    names, _faint = part_names(info)
    inten = {st: dict(rows) for st, rows in (info.get("intensity") or {}).items() if st in names}
    timeline = drum_timeline(info)
    changes = change_points(info, names)
    lyr = place_lyrics(info) if (info.get("vocals_present") != "no" and quote_lyrics) else []
    main_groove = groove_plain((info.get("drums") or {}).get("groove") or "")
    groove_by = []
    for sg in (info.get("drums") or {}).get("sections") or []:
        starts = [_secs(r.split("-")[0]) for r in str(sg.get("ranges") or "").split(",")]
        g = groove_plain(sg.get("groove") or "")
        groove_by.append(([s for s in starts if s is not None], g))
    tagmap = {}
    for l in info.get("presence") or []:
        m = re.match(r"^(.+?) \((\w+) stem\): (.+)$", l.strip())
        if not m or m.group(2) == "drums":
            continue
        if "location uncertain:" in m.group(3):
            continue  # the caveat is carried separately in Easy to get wrong
        conf = {t["label"] for t in confident_tags((info.get("stem_tags") or {}).get(m.group(2)))}
        if m.group(1) not in conf:
            continue
        where = re.sub(r"\s+only$", "", m.group(3))
        for nm in [w.strip() for w in where.split(",")]:
            tagmap.setdefault(nm, []).append(_tag_word(m.group(1)))
    onset = _secs(info.get("vocal_onset")) if info.get("vocal_onset") else None
    _rec, vhigh = melody_extras(info, "vocals")
    vocal_on = "vocals" in names
    sv = info.get("section_voice") or {}
    sv_left = list(sv.get("spans") or [])
    hm_by = {}
    for r in info.get("harmony_melody") or []:
        hm_by.setdefault(r["name"], r)

    out = []
    groove_said, prev_groove = {}, ""      # 2.3 VARIED_WORDING
    prev_contour = [""]
    prev_tags = []
    cp_rows = ((info.get("chord_prog") or {}).get("sections") or []) if CHORD_PROGRESSION_IN_DOSSIER else []
    seen_cluster = {}
    prev_lv = {}
    prev_drums = ""
    prev_mel = {}
    prev_lines, prev_loose = set(), False
    for i, s in enumerate(secs):
        a, z, name = s["start"], s["end"], s["name"]
        base = _base_name(name).lower()
        head = f"**{mmss(a)}–{mmss(z)} · {name}**"
        if base == "silence":
            out.append(f"{head} — digital silence; the song is over.")
            continue
        S = []
        cl = (s.get("cluster") or "").strip()
        if cl and cl.lower() != "unique":
            if cl in seen_cluster:
                S.append(f"Same music as {seen_cluster[cl]}")
            else:
                seen_cluster[cl] = name

        # who's playing, and what changed
        lv = {st: LEVEL_RANK.get(INTENSITY_SHORT.get(v, v), None) for st, v in
              ((st, rows.get(name, "")) for st, rows in inten.items())}
        lv = {k: v for k, v in lv.items() if v is not None}
        if lv:
            if i == 0 or not prev_lv:
                on = [names[k] + (" at full strength" if v == 5 else "") for k, v in lv.items() if v >= 3]
                low = [names[k] for k, v in lv.items() if v in (1, 2)]
                off = [names[k] for k, v in lv.items() if v == 0]
                bits = []
                if on:
                    bits.append(f"it opens with {join_words(on)}")
                if low:
                    bits.append(f"{join_words(low)} {_v(low, 'is', 'are')} there but held back")
                if off:
                    bits.append(f"no {join_words([o.replace('the ', '') for o in off], 'or')} yet")
                if bits:
                    S.append("; ".join(bits))
            else:
                ins, ups, downs, outs = [], [], [], []
                for k, v in lv.items():
                    p = prev_lv.get(k)
                    if p is None or p == v:
                        continue
                    if p == 0 and v >= 2:
                        ins.append(names[k])
                    elif v == 0 and p >= 2:
                        outs.append(names[k])
                    elif v > p and (v - p >= 2 or v == 5):
                        ups.append(names[k] + (f" {_v([names[k]], 'hits', 'hit')} full strength" if v == 5
                                               else f" {_v([names[k]], 'steps', 'step')} up"))
                    elif v < p and p - v >= 2:
                        downs.append(names[k])
                bits = []
                if ins:
                    bits.append(f"{join_words(ins)} {_v(ins, 'comes', 'come')} in")
                bits.extend(ups)
                if downs:
                    bits.append(f"{join_words(downs)} {_v(downs, 'pulls', 'pull')} back")
                if outs:
                    bits.append(f"{join_words(outs)} {_v(outs, 'drops', 'drop')} out")
                if bits:
                    S.append(join_words(bits))
                elif not any(x.startswith("Same music") for x in S):
                    S.append("much the same band as before")
            prev_lv = lv

        # drums over this stretch
        segs = [(max(a, sa), min(z, sb), cfg) for sa, sb, cfg in timeline
                if sa is not None and sb is not None and min(z, sb) - max(a, sa) >= 1.5]
        if segs:
            if len(segs) == 1:
                d = drum_config_plain(segs[0][2])
                if d != prev_drums:
                    S.append(f"drums: {d}")
                prev_drums = d
            else:
                # 2.3: consecutive stretches with the same kit are one stretch
                # ("the full kit, then the full kit from 0:33" said nothing)
                _segs = [segs[0]]
                for _x in segs[1:]:
                    if not VARIED_WORDING or drum_config_plain(_x[2]) != drum_config_plain(_segs[-1][2]):
                        _segs.append(_x)
                if len(_segs) == 1:
                    d = drum_config_plain(_segs[0][2])
                    if d != prev_drums:
                        S.append(f"drums: {d}")
                else:
                    parts = [drum_config_plain(_segs[0][2])] + [
                        f"{drum_config_plain(c)} from {mmss(x)}" for x, _y, c in _segs[1:]]
                    S.append("drums: " + parts[0] + ", then " + ", then ".join(parts[1:]))
                prev_drums = drum_config_plain(segs[-1][2])
        g = next((gg for starts, gg in groove_by if any(abs(st - a) <= 2 for st in starts)), "")
        if info.get("tempo_regrid"):
            g = ""     # beat patterns counted at the wrong tempo (dev19 tempo check)
        if g and main_groove and g != main_groove:
            if VARIED_WORDING and g in groove_said:
                # 2.3: the same groove again, said once instead of in the same words
                if g != prev_groove:
                    S.append(f"the groove from {groove_said[g]} is back")
            else:
                S.append(f"the beat here: {g}" if i == 0 else f"the groove changes here: {g}")
                groove_said.setdefault(g, name)
        if g:
            prev_groove = g

        # change points inside the section
        for t, ins, outs in changes:
            if t is None or not (a + 1 < t <= z - 0.5):
                continue
            if timeline:
                ins = [x for x in ins if x not in _KIT_PIECES]
                outs = [x for x in outs if x not in _KIT_PIECES]
            bits = []
            if outs:
                bits.append(f"{join_words(outs)} {_v(outs, 'drops', 'drop')} out")
            if ins:
                bits.append(f"{join_words(ins)} {_v(ins, 'comes', 'come')} in")
            if bits:
                S.append(f"at {mmss(t)} " + " and ".join(bits))

        _tags_here = list(dict.fromkeys(tagmap.get(name, [])))
        if VARIED_WORDING:
            # 2.3: what can be picked out, said once and then only when it changes,
            # not one stock sentence per instrument in every part
            _new = [w for w in _tags_here if w not in prev_tags]
            _w = [f"something {w.rstrip('s')}-like" if w.endswith("strings")
                  else "some " + w if w.split("(")[0].strip().endswith(("brass", "percussion"))
                  else w if w.split("(")[0].strip().endswith(("chords", "drums", "vocals")) else _a_an(w)
                  for w in _new]
            if _w:
                S.append(("also in there: " if not prev_tags else "now you can also pick out ") + join_words(_w))
            if _tags_here:
                prev_tags[:] = _tags_here
        else:
            for w in _tags_here:
                S.append(f"something {w.rstrip('s')}-like comes through here" if w.endswith("strings")
                         else f"you can hear {w} here" if w.split("(")[0].strip().endswith(("chords", "drums", "vocals"))
                         else f"you can hear {_a_an(w)} here")

        # the voice and the words
        if vocal_on:
            lo = a - 1.5 if i > 0 else -1.0
            hi_t = z - 1.5 if i + 1 < len(secs) and _base_name(secs[i + 1]["name"]).lower() != "silence" else z
            here = [x for x in lyr if lo <= x[0] < hi_t]
            # a line sung just before the boundary belongs here only if the
            # previous section didn't already have it (the tail of a chorus)
            here = [x for x in here if x[0] >= a or x[2] not in prev_lines]
            exact = [x for x in here if x[3]]
            loose = [x for x in here if not x[3] and len(_words(x[2])) >= 3
                     and (not exact or x[0] < exact[0][0] - 1)]
            if onset is not None and a <= onset < z and onset - a > 2 and not (
                    exact and abs(exact[0][0] - onset) <= 3):
                S.append(f"the lead vocal proper starts at {mmss(onset)}")
            if loose and not prev_loose:
                q = clip(" ".join(x[2] for x in loose[:2]), 60)
                if info.get("lyrics_src") == "tags":
                    S.append(f"there are some vocal bits{' first' if exact else ''} that are hard to make out "
                             f"(something like \u201c{verbatim(q)}\u201d)")
                else:
                    S.append(f"words (roughly): \u201c{verbatim(q)}\u201d")
            prev_loose = bool(loose)
            if exact:
                first = exact[0]
                when = f" at {mmss(first[0])}" if first[0] - a > 4 else ""
                S.append((f"then the words begin{when}: " if loose else f"the words{when}: ")
                         + f"\u201c{verbatim(clip(first[2], 70))}\u201d")
                last_head = _base_name(first[1]).lower()
                shown = 0
                for t, h, line, _ex in exact[1:]:
                    hb = _base_name(h).lower()
                    if h and hb != last_head and shown < 2:
                        tag = f" — the {h.lower()}" if hb in _LYRIC_PART_WORDS and hb != base else ""
                        S.append(f"\u201c{verbatim(clip(line, 50))}\u201d arrives around {mmss(t)}{tag}")
                        shown += 1
                    last_head = hb or last_head
            prev_lines = {x[2] for x in exact}
            mel = melody_for(info, "vocals", name)
            if mel.get("lo") and mel.get("hi"):
                mtxt = f"the melody sits between {pretty_note(mel['lo'])} and {pretty_note(mel['hi'])}"
                if mel.get("contour") and not (VARIED_WORDING and mel["contour"] == prev_contour[0]):
                    mtxt += f" and {mel['contour']}"
                prev_contour[0] = mel.get("contour") or ""
                pb = prev_mel.get(base)
                hi = note_midi(mel["hi"])
                if pb is not None and hi is not None and hi - pb >= 2:
                    mtxt += f" — pitched higher than the last {base}"
                if hi is not None:
                    prev_mel[base] = hi
                S.append(mtxt)
            if vhigh and vhigh[1] is not None and a <= vhigh[1] < z:
                S.append(f"the vocal reaches its highest note ({pretty_note(vhigh[0])}) at {mmss(vhigh[1])}")
            # dev17 SECTION VOICE CHECK: where the second lead takes over
            here_sv = [x for x in sv_left if a - 1 <= x[0] < z]
            for x in here_sv:
                sv_left.remove(x)
                who = "a man" if sv.get("other") == "male" else "a woman"
                S.append(f"{who} takes the lead from {mmss(x[0])} to {mmss(x[1])}")
        hmr = hm_by.get(name)
        cpr = _part_for(cp_rows, a) if cp_rows else None
        if cpr is not None:
            # 2.3: the measured progression (dev45) rather than the most common transcribed chords
            if not cpr["none"] and cpr["chords"]:
                S.append(f"the chords: {cpr['chords']}" + (f", {cpr['rhythm']}" if cpr["rhythm"] else ""))
        elif hmr and hmr.get("chords") and not hmr.get("unreliable"):
            S.append(f"chords mostly {_chord_words(hmr['chords'][:3])}")
        if s.get("name_check"):
            S.append(f"(the name is a best guess here: {s['name_check']})")
        body = " ".join(sentence(x) for x in S if x)
        out.append(f"{head} ({int(round(z - a))} s) — {body}" if body else f"{head} ({int(round(z - a))} s)")
    return "\n\n".join(out)


# ---- who is singing ------------------------------------------------------------------------
def _unique_credit_names(names):
    """Keep first spelling/order while deduplicating names case/punctuation-insensitively."""
    out, seen = [], set()
    for name in names:
        name = str(name or "").strip()
        folded = _fold_name(name)
        if name and folded and folded not in seen:
            out.append(name)
            seen.add(folded)
    return out


def _co_lead_detail(info):
    """Return an explicit co-lead description, excluding empty placeholders."""
    voices = info.get("voices") if isinstance(info.get("voices"), dict) else {}
    vocal_extra = info.get("vocal_extra") if isinstance(info.get("vocal_extra"), dict) else {}
    detail = str(voices.get("CO_LEAD_DETAIL") or vocal_extra.get("CO_LEAD_DETAIL") or "").strip()
    detail_norm = re.sub(r"[\[\]\s.]", "", detail).lower()
    if detail_norm in ("", "none", "n/a", "na", "uncertain", "unknown", "notclear", "unparsed"):
        return ""
    return clip(detail, 400)


def who_sings(info, artist=""):
    """(prose, pitfalls[])"""
    if info.get("vocals_present") == "no":
        if info.get("faint_vocal_texture"):
            return ("Nobody — it's an instrumental. There's no singer and there are no lyrics, though faint "
                    "vocal-like sounds (a sample, or a chopped or processed voice) may be audible in the background "
                    "as texture.",
                    ["It's an instrumental: don't describe a singer, a vocal range, lyrics or a duet. Faint "
                     "vocal-like sounds in the background are texture at most — mention them only as that."])
        return ("Nobody — it's an instrumental. There's no singer and there are no lyrics.",
                ["It's an instrumental: don't describe a singer, a vocal range or lyrics."
                 + (" (A voice detector thought it heard one; the closer check says what it caught was the "
                    "instruments bleeding through.)" if info.get("vocal_verdict") in ("arbitration", "rechecked") else "")])
    pit = []
    ident = info.get("identity") or ""
    facts = info.get("identity_facts") or {}
    features = _unique_credit_names(facts.get("features") or [])
    gender = singer_gender(info)
    name = facts.get("name") or ""
    if not name and artist and info.get("lead_singers") != "several":
        name = artist
        # A band named as the singer ("Band Z sings lead"): its singer.
        _ir = info.get("identity_reason") or ""
        if (re.search(r"(?i)\b(?:band|group|trio|quartet|quintet)\b", _ir)
                or (SINGER_NAME_ANY_WORDING and _GROUP_WORD_RE.search(_ir) and not _SOLO_WORD_RE.search(_ir))):
            name = (artist + "'" if artist.endswith("s") else artist + "'s") + " singer"
    unbroken = info.get("voice_unbroken")
    np_ = info.get("note_pitch") or {}
    p = info.get("pitch") or {}
    S = []
    who = name or ("the singer" if gender else "")
    he = {"male": "he", "female": "she"}.get(gender, "they")
    his = {"male": "his", "female": "her"}.get(gender, "their")
    age = facts.get("age")
    sv = info.get("section_voice") or {}
    if sv:
        # dev17 SECTION VOICE CHECK: the main lead and a second lead of the
        # other gender who takes some stretches.
        person = {"female": "a woman", "male": "a man"}
        more = [x for x in sv.get("all_spans") or [] if x not in sv["spans"]]
        S.append(f"Two voices take the lead: {person.get(sv['main'], 'one singer')} sings most of it, and "
                 f"{person.get(sv['other'], 'a second singer')} takes over in {_span_words(sv['spans'])}"
                 + (f" (and probably {_span_words(more)} too)" if more else ""))
        if sv.get("direction"):
            S.append(f"In those stretches the line sits {sv.get('shift') or 'several'}+ semitones "
                     f"{sv['direction']} than the main voice")
        if name:
            S.append(f"The credited act is {name}")
        elif features and artist:
            main_act = re.split(r"(?i)\s+(?:feat(?:\.|\u2024)?|ft(?:\.|\u2024)|featuring|with)\s", artist)[0].strip(" ,(")
            if main_act and not any(_fold_name(main_act) == _fold_name(f) for f in features):
                S.append(f"The other lead is the credited act, {main_act}")
        guest_names = [f for f in features if _fold_name(f) != _fold_name(name)]
        if guest_names:
            S.append("The credited guest/co-vocalist is " + join_words(guest_names))
        detail = _co_lead_detail(info)
        if detail:
            S.append("The distinct lead parts: " + detail)
        pit.append(f"There are two lead voices: {person.get(sv['main'], 'the main singer')} for most of the track "
                   f"and {person.get(sv['other'], 'a second singer')} in {_span_words(sv['spans'])}. Describe both "
                   "and where each one leads — don't give the whole track one singer or one gender.")
    elif info.get("lead_singers") == "several":
        named = []
        if name:
            named.append(name)
        elif features and artist:
            # Without a named individual lead, use the credited act itself rather
            # than always calling the act's vocalist "the act's singer". If the
            # identity reasoning documents a group, keep the more specific wording.
            main_act = re.split(r"(?i)\s+(?:feat(?:\.|\u2024)?|ft(?:\.|\u2024)|featuring|with)\s", artist)[0].strip(" ,(")
            if main_act and not any(_fold_name(main_act) == _fold_name(f) for f in features):
                # Preserve the original wording: when the individual's lead
                # singer is unnamed, refer to the credited act's singer.
                named.append(f"{main_act}'s singer")
        named = _unique_credit_names(named + features)
        if named:
            if len(named) == 1:
                S.append(f"The lead is shared — {named[0]} is credited with others (a group, duo or "
                         "featured singer), not one lone voice")
            else:
                S.append("The lead is shared by " + join_words(named)
                         + " (a group, duo or featured singer) — not one lone voice")
        else:
            S.append("The lead is shared by several singers (a group, duo or featured singer) — not one lone voice; "
                     "nothing here says who takes which line")
        detail = _co_lead_detail(info)
        if detail:
            S.append("The distinct lead parts: " + detail)
    elif name:
        verb = "sing" if re.search(r" and | & |\+", name) else "sings"
        line = f"{name} {verb} lead"
        if age and 5 <= age <= 99:
            line += f", about {age} at the time"
        S.append(line)
    deepened = info.get("voice_deepened")
    if sv:
        pass
    elif unbroken and gender == "male":
        # High/light/boyish weight only — do not invite a stock "sounds like a
        # girl / woman" formula; the reader forms that from timbre and delivery.
        S.append(f"{cap(his)} voice hadn't broken yet on this recording: high, light and still boyish in weight")
        if age and age >= 15:
            S.append(f"That's later than usual — most boys' voices have dropped by 15 or 16 — so it's a real talking point")
        pit.append(f"The singer is male: say {he}/{his}, never 'she'. {cap(his)} voice is unbroken here "
                   f"(high and light). Describe how it actually sounds from the timbre, delivery and range "
                   f"you hear — do not default to a stock 'sounds like a girl' or 'sounds like a woman' line.")
    elif deepened and gender == "male":
        # Adolescent (or young) identity whose audio is already a man's voice (dev69).
        if age and age <= 16:
            S.append(f"{cap(his)} voice had already deepened by this recording — lower and fuller than a "
                     "boy's unbroken voice, not high, light or boyish — even though he was still a teenager")
        else:
            S.append(f"{cap(his)} voice had already deepened on this recording: lower and fuller than a "
                     "boy's unbroken voice, not high, light or boyish")
        pit.append(f"Don't describe {his} voice as unbroken, high, light or boyish here — it had already "
                   f"dropped. Still {he}/{his}, never 'she'.")
    elif info.get("voice_maybe_deepening") and gender == "male":
        S.append(f"{cap(his)} voice was probably starting to deepen on this recording — the audio heard a young "
                 "man's voice rather than a boy's — though it still sits fairly high")
        pit.append(f"Don't call {his} voice fully deepened or a grown man's. Still {he}/{his}, never 'she'. "
                   f"Describe the sound from what you hear, not a stock comparison.")
    elif unbroken and not gender:
        S.append("The voice sounds young and high — gender isn't settled from the evidence here")
        pit.append("Don't assign the singer a gender from a stock comparison. Describe the sound from "
                   "timbre, weight and range; leave gender open if it isn't settled.")
    else:
        tag = ident if ident and ident not in ("unknown", "uncertain") else info.get("acoustic_lead") or ""
        plain = {
            "child_male_likely": "a young boy's voice", "child_female_likely": "a young girl's voice",
            "child_gender_uncertain": "a child's voice",
            # Without unbroken/deepened evidence, keep adolescent neutral — not "still changing"
            # by default (that phrase is reserved for the unbroken path).
            "adolescent_male_likely": "a teenage boy's voice",
            "young_male": "a young man's voice", "post_puberty_male": "a man's voice", "adult_male": "a man's voice",
            "female_teen_adult": "a woman's voice", "adult_female": "a woman's voice", "young_female": "a young woman's voice",
            "mixed_leads": "two lead singers trading the lead",
        }.get(tag, "")
        if plain:
            S.append(f"It's {plain}")
        elif not name:
            if info.get("lead_singers") == "several":
                # 2.3: a shared lead is not "a single lead voice"
                S.append("Whether it's men or women singing isn't clear from the sound alone")
            else:
                S.append("A single lead voice; whether it's a man or a woman isn't clear from the sound alone")
            pit.append("The singer's gender isn't settled — don't guess one.")
        ac = info.get("acoustic_lead") or ""
        if ident and ac and _is_male(ident) and _is_female(ac):
            S.append(f"{cap(his)} voice sits high for a male lead")
            pit.append(f"The singer is male: say {he}/{his}, never 'she'. Describe the high placement from "
                       f"what you hear — do not default to a stock 'sounds like a girl' line.")
    # range
    lo_hi = ""
    if np_.get("lo_note") and np_.get("hi_note"):
        lo_hi = f"{pretty_note(np_['lo_note'])} and {pretty_note(np_['hi_note'])}"
    elif p.get("range_notes"):
        lo_hi = re.sub(r"\s+to\s+", " and ", p["range_notes"])
    centre = np_.get("median_note") or p.get("median_note") or ""
    if lo_hi or centre:
        r = f"{cap(his) if gender else 'The'} melody mostly moves between {lo_hi}" if lo_hi else ""
        if centre:
            r = (r + f", centred around {pretty_note(centre)}") if r else f"It's centred around {pretty_note(centre)}"
        S.append(r)
    vt = p.get("male_type") if gender == "male" else p.get("female_type") if gender == "female" else ""
    if vt and not sv:
        S.append(f"For {_a_an('male' if gender == 'male' else 'female')} singer that's {vt.split('(')[0].strip()}"
                 + (" — high for a male lead" if gender == "male" and re.search(r"high tenor|very high|falsetto",
                                                                              vt.lower()) else ""))
    if info.get("voice_count_check"):
        pit.append("One lead singer (with any harmonies or doubling) — not two co-leads an octave apart.")
    return " ".join(sentence(x) for x in S if x), pit


# ---- render ------------------------------------------------------------------------------------
def _key_words(info):
    k = info.get("key") or ""
    return k


def typical_grid(info):
    """(grid, label): two bars from the section group that runs longest (the
    chorus beat, usually), else Musiclyse's representative bars, which can come
    from an intro that isn't typical at all."""
    best, best_len, label = None, 0.0, ""
    for sg in (info.get("drums") or {}).get("sections") or []:
        if not sg.get("grids"):
            continue
        tot = 0.0
        for r in str(sg.get("ranges") or "").split(","):
            mm = re.match(r"\s*(\d+:\d\d)-(\d+:\d\d)", r)
            if mm and _secs(mm.group(1)) is not None and _secs(mm.group(2)) is not None:
                tot += _secs(mm.group(2)) - _secs(mm.group(1))
        if tot > best_len:
            best, best_len, label = sg["grids"][0], tot, _base_name(sg["label"]).lower()
    if best:
        return best, label
    rep = (info.get("drums") or {}).get("rep_grid")
    return (rep, "main") if rep else (None, "")


def groove_grids(info):
    """2.3: [(part label, start m:ss, grid)] -- the first grid of each section
    group whose groove differs from the ones before it, in time order, at most
    GROOVE_GRIDS_MAX; [] when the save has no section grids."""
    items = []
    for sg in (info.get("drums") or {}).get("sections") or []:
        if not sg.get("grids"):
            continue
        g = sg["grids"][0]
        m = re.search(r"from (\d+:\d\d)", g.get("meta") or "")
        st = m.group(1) if m else (str(sg.get("ranges") or "").split("-")[0].strip())
        sig = groove_plain(sg.get("groove") or "") or "|".join(g.get("rows") or [])
        items.append((_secs(st) if _secs(st) is not None else 1e9, sg["label"], st, sig, g))
    items.sort(key=lambda x: x[0])
    out, seen = [], set()
    for _t, lab, st, sig, g in items:
        if sig in seen:
            continue
        seen.add(sig)
        out.append((lab, st, g))
    return out[:max(2, int(GROOVE_GRIDS_MAX))]


def melodicism_words(info):
    """2.3: one line on how melodic the singing is (dev45 MELODICISM BY SECTION), or ''."""
    ml = info.get("melodicism") or {}
    ov = ml.get("overall") or ""
    if not ov:
        return ""
    band, _sep, desc = ov.partition(": ")
    line = f"How melodic the singing is: {band} overall" + (f" — {desc.replace(' -- ', '; ')}" if desc else "")
    if ml.get("most") and ml.get("least") and ml["most"] != ml["least"]:
        line += f"; most melodic in {ml['most']}, most rhythm-led in {ml['least']}"
    return line


def chord_progression_words(info):
    """2.3: one line on the measured chord progression (dev45), or ''."""
    cp = info.get("chord_prog") or {}
    if not cp.get("sections"):
        return ""
    if cp.get("cycle"):
        return "The chord progression (measured, in time order): the cycle heard most is " + cp["cycle"]
    rows = [r for r in cp["sections"] if not r["none"] and r["chords"]]
    if not rows:
        return ""
    r = max(rows, key=lambda x: (x["end"] or 0) - (x["start"] or 0))
    return (f"The chord progression (measured, in time order): in the longest part, {r['name']}, the chords run "
            f"{r['chords']}" + (f" ({r['roman']})" if r["roman"] else ""))


def melody_key_words(info):
    """2.3: one line on how the tune sits in the key (dev45 MELODY IN THE KEY), or ''."""
    mk = info.get("melody_key") or {}
    rows = [r for r in mk.get("rows") or [] if not r["unclear"]]
    if len(rows) < 2:
        return ""
    ends, held = {}, {}
    for r in rows:
        m = re.search(r"phrases mostly end on (\S+)(?: \(([^)]*)\))?", r["text"])
        if m:
            ends.setdefault(m.group(1), [0, m.group(2) or ""])[0] += 1
        m = re.search(r"held notes centre on ([^;]+)", r["text"])
        if m:
            for dg in re.split(r",\s*|\s+and\s+", m.group(1).strip()):
                held[dg] = held.get(dg, 0) + 1
    bits = []
    if held:
        top = [d for d, _n in sorted(held.items(), key=lambda kv: -kv[1])[:2]]
        bits.append("its long notes lean on " + join_words(top) + " of the scale")
    if ends:
        dg, (n, gloss) = max(ends.items(), key=lambda kv: kv[1][0])
        if n * 2 >= len(rows):
            bits.append(f"its phrases mostly come to rest on {dg}" + (f" ({gloss})" if gloss else ""))
    if not bits:
        return ""
    return (f"The tune against the key ({mk['key']}, 1 = {mk['tonic']}): " if mk.get("key")
            else "The tune against the key: ") + "; ".join(bits)


def era_style_words(info):
    """2.3: dev44's ERA CHECK in plain words, or ''."""
    t = info.get("era_style_check") or ""
    m = re.search(r"the clearest period-bound style heard, (.+?) \(\d+%\), belongs to about (\d{4})–(\d{4}), so the "
                  r"recording most likely dates from the ([^.]+)\.", t)
    if not m:
        return ""
    return (f"The style it most clearly fits, {m.group(1)}, belongs to about {m.group(2)}–{m.group(3)}, so it most "
            f"likely sounds like the {m.group(4)}: a 'modern' or 'recent' impression of it comes from clean, "
            "loud production, not from the style.")


def meter_words(info):
    """'4/4' (or 3/4, 6/8 ...) from the meter estimate, with any accent note."""
    for blk in info.get("measure_blocks") or []:
        for l in blk["lines"]:
            m = re.match(r"\s*meter estimate:\s*(.+)$", l)
            if m:
                t = m.group(1)
                sig = re.search(r"\b(\d{1,2}/\d{1,2})\b", t)
                if sig:
                    return sig.group(1) + (" (with a two-bar accent pattern)" if "two-bar" in t else "")
                return re.split(r"\s*\(", t)[0].strip()
    return ""


def render_track(info, level="standard", lyrics=True, max_lyrics_chars=6000, raw_notes=False, cover=False):
    # full = complete timed MIDI. high = same measurement depth, compressed MIDI.
    high = level == "high"
    full = level in ("full", "high")   # non-MIDI "full depth" blocks (stem reports, file details, …)
    brief = level == "brief"
    S = []
    mf = info["mf"]
    names, faint = part_names(info)
    gender = singer_gender(info)
    title, artist = info.get("title") or info.get("label"), info.get("artist")

    # --- The track in a nutshell ------------------------------------------------------------
    n = []
    if info.get("title") and artist:
        first = f"“{verbatim(info['title'])}” is {verbatim(artist)}'s track"
    elif info.get("title"):
        first = f"“{verbatim(info['title'])}” is a track by an artist you don't know the name of"
    else:
        first = (f"This is a track you know only as the file “{verbatim(info.get('label') or 'untitled')}” — "
                 "no title or artist to go on, so talk about the sound and don't guess who made it")
    if info.get("album"):
        first += f" from {verbatim(info['album'])}"
    if info.get("year"):
        first += f" ({info['year']})" if info.get("album") else f", released in {info['year']}"
    n.append(first)
    gp = genre_phrase(info)
    desc = []
    if gp:
        desc.append(gp)
    if info.get("bpm"):
        desc.append(f"at about {round(float(info['bpm']))} BPM")
    if info.get("key"):
        desc.append(f"in {info['key']}")
    if desc:
        line = "It's " + (desc[0] if gp else "a track") + (" " + " ".join(desc[1:] if gp else desc) if (desc[1:] if gp else desc) else "")
        if info.get("duration"):
            line += f", {mmss(info['duration'])} long"
        n.append(line)
    inst = []
    d = info.get("drums") or {}
    if d.get("groove") and info.get("tempo_regrid"):
        inst.append("programmed-sounding drums" if drums_programmed(info) else "drums")
    elif d.get("groove"):
        prog = drums_programmed(info)
        kick = d.get("kick") or ""
        beat = ("a four-on-the-floor beat" if kick.startswith("four-on-the-floor")
                else groove_plain(kick) or groove_plain(d["groove"]))
        inst.append(("programmed-sounding drums" if prog else "drums") + " on " + beat
                    if beat.startswith("a four") else
                    ("programmed-sounding drums" if prog else "drums") + " with " + beat)
    bw = bass_words(info, names)
    if bw:
        inst.append(bw)
    if "other" in names:
        inst.append(other_layer_words(info))
    if "guitar" in names:
        inst.append("a " + names["guitar"].replace("the ", ""))
    if "piano" in names:
        inst.append("a " + names["piano"].replace("the ", ""))
    if info.get("reed_lead"):
        inst.append("a steady reedy lead line (harmonica-like or brassy)")
    if inst:
        n.append("Underneath: " + join_words(inst))
    if info.get("no_kit"):
        n.append("There's no drum kit on it — the rhythm comes from the instruments that are playing")
    if info.get("vocals_present") == "no":
        n.append("There's no singing — it's an instrumental"
                 + (", though faint vocal-like sounds may sit in the background as texture"
                    if info.get("faint_vocal_texture") else ""))
    if info.get("excerpt"):
        xa, xb = info["excerpt"]
        n.append(f"You only heard an excerpt, {mmss(xa)} to {mmss(xb)}, so everything here is about that stretch "
                 "(times count from its start)")
    pw = production_words(info)
    if pw:
        n.append("The production: " + join_words(pw))
    mood = mf.get("MOOD_VIBE", "")
    if mood and not is_non_answer(re.sub(r"\s*\d\)\s*", " ", mood)):
        n.append("Mood, by ear: " + re.sub(r"\s*\d\)\s*", " ", mood).strip(" ;").replace(" ;", ";"))
    S.append(_section("The track in a nutshell", " ".join(sentence(x) for x in n), 10, 99))

    # --- User corrections -----------------------------------------------------------------------
    if info["corrections"]:
        body = ("Facts the listener has confirmed — these win over anything else here:\n" +
                "\n".join(f"- {k}: {verbatim(str(v))}" for k, v in info["corrections"].items()))
        S.append(_section("Confirmed by the listener", body, 12, 99))

    # --- Who's singing ----------------------------------------------------------------------------
    ws, pit = who_sings(info, artist)
    S.append(_section("Who's singing", ws, 14, 95))

    # --- Easy to get wrong --------------------------------------------------------------------------
    pit = list(pit)
    uncertain_presence = [line for line in (info.get("presence") or [])
                          if "location uncertain:" in line]
    if uncertain_presence:
        labels = []
        for line in uncertain_presence:
            match = re.match(r"^(.+?) \((\w+) stem\):", line.strip())
            if match and match.group(1) not in labels:
                labels.append(match.group(1))
        if labels:
            pit.append("The section placement of " + join_words(labels)
                       + " is uncertain: the supporting audio windows overlap section boundaries. "
                       "Do not claim exact entries/exits or that these instruments occur only in "
                       "one section, or continuously throughout, from those windows alone.")
    gr = info.get("genre_replaced") or {}
    if info.get("synth_rock"):
        pit.append(f"It's {genre_phrase(info)}: a rock band (guitar, bass, drums) with a synthesizer riff as the lead "
                   f"instrument. {join_words([f'‘{x}’' for x in info['synth_rock']], 'or')} is a fair comparison for "
                   "the synth sound, but don't call it pure synth-pop or club / dance music, and don't leave out "
                   "either the band or the synth.")
    if gr:
        olds = join_words([f"‘{o}’" for o in gr["olds"]], "or")
        if gr["direction"] == "to_pop":
            extra = (" A guitar-rock edge is fair as a secondary flavour." if gr.get("guitar_leads") else
                     " Any guitar-like layer is one colour in the production, not a rock band.")
        else:
            extra = " Its clean, steady, machine-like feel is a secondary flavour, not club or dance music."
        pit.append(f"It's {genre_phrase(info)}. On a first listen it can come across as {olds}, but that "
                   f"impression doesn't hold up — don't call it that, and don't say it's usually filed that way.{extra}")
    if info.get("arrangement_check") and "other" in names:
        pit.append(("Besides the voice, drums and bass" if info.get("vocals_present") != "no" else "Besides the drums and bass")
                   + " there's a real chordal keys/synth layer playing through most of the song — include it when "
                   "describing the arrangement.")
    ad = info.get("arrangement_denial") or {}
    if ad and "other" in names:
        peaks = re.sub(r"(\d+:\d\d)-(\d+:\d\d)", r"\1–\2", ad["peaks"])
        pit.append("Don't say there are no synths or keyboards: there's a chordal keys/synth part (about "
                   f"{ad['poly']:g} notes at once) playing in {ad['playing']} of {ad['rows']} sections and loudest in "
                   f"{peaks} — most likely synth or keyboard chords and lines; describe it by its sound where it plays."
                   + (f" Weak readings of it as {join_words(ad['weak'], 'or')} are just how a bright synth or keyboard "
                      "tends to register, not confirmed instruments." if ad.get("weak") else ""))
    if info.get("no_kit"):
        pit.append("There's no drum kit on this recording. Don't describe drums, a kick, a snare, hi-hats or a "
                   "groove played on a kit; the rhythm is carried by the instruments that are there (strummed or "
                   "picked strings, the voice, any hand percussion on an instrument body).")
    rl = info.get("reed_lead")
    if rl:
        pit.append(f"The steady reedy lead{(' on the ' + rl[4] + ' part') if rl[4] and rl[4] != 'other' else ''} is "
                   "real, but don't say for certain what it is: a harmonica or a horn such as a saxophone or trumpet "
                   "sound alike here. Call it a harmonica-like or brassy lead. A lead line described as another "
                   "instrument may be this part.")
    if info.get("reed_pad_poly") and "other" in names:
        pit.append("Don't say there's an accordion: that chordal part is most likely a synth pad or sustained "
                   f"keyboard chords (about {info['reed_pad_poly']:g} notes at once).")
    if info.get("guitar_check"):
        pit.append("The guitar is acoustic throughout. Any 'electric guitar' in the by-ear notes is at most a clean, "
                   "lightly amplified or semi-acoustic sound next to it — never a distorted, overdriven or rock "
                   "guitar; keep the acoustic guitar as the main guitar sound.")
    ge = info.get("genre_era") or info.get("genre_era_stock") or {}
    if ge:
        late = join_words([f"‘{x}’" for x in ge["late"]], "or")
        pit.append(f"Don't call it {late}: {'those labels are' if len(ge['late']) > 1 else 'that label is'} from "
                   f"later decades ('{ge['rec']}' wasn't in use before about {ge['onset']}, and this is tagged "
                   f"{ge['year']}). " + (f"Styles of its own time that fit: {join_words(ge['fits'])}. " if ge['fits'] else "")
                   + "Saying it anticipates or influenced later styles is fine when relevant.")
    kc = info.get("key_centre") or {}
    if kc:
        pit.append(f"It's in {kc['centre']} — {kc['centre']} Mixolydian, using the notes of {kc['parent']}"
                   + (f" ({kc['flat7']} rather than {kc['lead']})" if kc.get("flat7") and kc.get("lead") else "")
                   + f". A key detector would call it {kc['parent']}; the music centres on {kc['centre']}.")
    fb = info.get("full_band_tempo") or {}
    if fb.get("bpm"):
        pit.append(f"The tempo is about {round(float(info.get('bpm') or fb['bpm']))} BPM. Slower readings"
                   + (f" ({fb['slow']} BPM)" if fb.get("slow") else "")
                   + " count the same pulse every few beats — the bright hi-hats and percussion high up set the real "
                   "pace. A half-time feel is fine to mention; the tempo isn't the slower number.")
    if info.get("low_end_note"):
        pit.append("There's no bass guitar to speak of; the low end comes from other sounds.")
    elif names.get("bass") == "the synth bass":
        pit.append("The low end is a synth bass, not a bass guitar.")
    if "inconclusive" in (info.get("guitar") or "") and "guitar" not in names:
        pit.append("Whether there's guitar at all is unclear — don't claim it either way; describe the "
                   "mid-range layer by its sound.")
    elif "no guitar detected" in (info.get("guitar") or ""):
        pit.append("There's no guitar on this track.")
    if any(_base_name(s["name"]).lower() == "silence" for s in info.get("sections") or []):
        pit.append("The last few seconds are digital silence — the song ends before them; there's no outro there.")
    if STYLES_IN_DOSSIER and info.get("styles") and gp:
        rel, oth = style_words(info)
        if rel or oth:
            pit.append(f"Asked whether it's one of the styles named under “The sound” "
                       f"({join_words(rel + oth, 'or')}), the honest answer is at least “partly, yes” — say what in "
                       f"the sound fits. {cap(_main_label(info, gp))} stays the main label.")
    if info.get("year"):
        pit.append(f"“When did it come out?” is {info['year']}. “What era does it sound like?” is a "
                   "separate question about the sound — see below.")
    if pit:
        S.append(_section("Easy to get wrong", "\n".join(f"- {x}" for x in pit), 16, 92))

    # --- How it unfolds ---------------------------------------------------------------------------
    wt = walkthrough(info, quote_lyrics=lyrics)
    if wt:
        intro = ("Your memory of the song from start to finish. Times are where each part starts and ends "
                 "(give them exactly as written; they can be a second or two out), and the section names are "
                 "a good guess at the song's form, not gospel.")
        if brief:
            short = []
            for item in wt.split("\n\n"):
                head, _sep, body = item.partition(" — ")
                sents = re.split(r"(?<=[.!?”])\s+(?=[A-Z“])", body)
                short.append(head + (" — " + " ".join(sents[:2]) if body else ""))
            wt = "\n".join(short)
        S.append(_section("How it unfolds", intro + "\n\n" + wt, 20, 90))

    # --- The sound ---------------------------------------------------------------------------------
    b = []
    if gp:
        nb = neighbour_styles(info)
        hw = genre_hedge_words(info)
        b.append(f"Style: {gp}" + (f" ({hw})" if hw else "") + (f"; close neighbours are {join_words(nb)}" if nb else ""))
        if STYLES_IN_DOSSIER and info.get("styles"):
            rel, oth = style_words(info)
            if rel:
                b.append("More specific styles you can hear in it: " + join_words(rel))
            if oth:
                b.append(f"Parts of it could also be heard as {join_words(oth, 'or')}, but "
                         f"{_main_label(info, gp)} stays the main label")
        ro = ruled_out_styles(info) if not brief else []
        if ro:
            b.append("What it isn't, by ear: " + join_words(ro[:10], "or"))
    if inst:
        full_inst = list(inst)
        if d.get("groove"):
            full_inst[0] = ("programmed-sounding drums" if drums_programmed(info) else "drums") + " — " + groove_plain(d["groove"])
        b.append("The band: " + "; ".join(full_inst))
    if "other" in names:
        _rec, hi_o = melody_extras(info, "other")
        b.append("The keys/synth layer is mostly chords — busy, with lots of wide jumps between notes rather "
                 "than a single tune")
    if faint:
        b.append("Also faintly there: " + join_words([{"guitar": "a guitar-range part", "piano": "a piano-range part",
                                                         "other": "a keys/synth layer"}.get(f, f) for f in sorted(faint)])
                 + " — barely audible, a minor colour at most")
    if info.get("mix_tags") and info["mix_tags"] != "nothing identified with confidence":
        tg = [_tag_word(t["label"]) for t in confident_tags(info["mix_tags"], 30)]
        if tg:
            b.append("Instruments you can clearly pick out: " + join_words(tg))
    # Production measurements (loudness, ambience, stereo width, pumping) already
    # appear once in the nutshell; repeating the same one-liner here made stock
    # models echo those stock phrases twice. By-ear PRODUCTION_NOTES below stay.
    ear = []
    for k, lab in (("INSTRUMENTATION", "Instruments"), ("TIMBRE", "Texture and mix"),
                   ("PRODUCTION_NOTES", "Production"), ("REFERENCE_POINTS", "Reminds you of")):
        v = clean_report_value(mf.get(k, ""))
        if not v or is_non_answer(v) or v.lower().startswith(("uncertain", "see ")):
            continue
        if k == "INSTRUMENTATION":
            items = split_top(v)
            if len(items) == 1 and "," in v:
                items = split_top(v, ",")        # "drums, synth bass, electric piano"
            if info.get("no_kit"):
                items = [x for x in items if not re.match(r"(?i)\s*(?:drums?|drum kit|drum machine|percussion|"
                                                          r"kick|snare|hi-?hats?|cymbals?)\b", x)]
            if info.get("guitar_check"):
                items = [re.sub(r"(?i)^\s*electric guitar\b", "a second guitar (clean, lightly amplified or "
                                "semi-acoustic in tone)", x) for x in items]
            if info.get("instrument_suppressed"):
                items = [x for x in items if not any(w in x.lower().split("(")[0] for w in info["instrument_suppressed"])]
            v = ("; " if ";" in mf.get(k, "") else ", ").join(items)
            if not v:
                continue
        elif info.get("no_kit"):
            v = drop_drum_claims(v)
            if not v:
                continue
        ear.append(f"{lab}: {scrub_gender(clip(v, 500 if not brief else 250), gender)}")
    if info.get("description") and not brief:
        ear.append("Overall impression: " + scrub_gender(clip(info["description"], 700), gender))
    if info.get("listening_prose"):
        ear.append(scrub_gender(clip(" ".join(info["listening_prose"].split()), 1200 if not full else 6000), gender))
    if info.get("listening_prose_extra") and not brief:
        ear.append(scrub_gender(clip(" ".join(info["listening_prose_extra"].split()), 900), gender))
    for k, v in (info.get("mf_extra") or {}).items():
        if k in ("VOCALS_DESCRIPTION",) or brief:
            continue
        ear.append(f"{k.replace('_', ' ').capitalize()}: {scrub_gender(clip(v, 400), gender)}")
    body = "\n".join(f"- {sentence(x)}" for x in b)
    if ear:
        body += ("\n\nFirst pass by ear (if this disagrees with anything above, trust what is above):\n"
                 + "\n".join(f"- {sentence(x)}" for x in ear))
    S.append(_section("The sound", body, 30, 80))

    # --- The groove -----------------------------------------------------------------------------------
    b = []
    if info.get("no_kit"):
        b.append("No drum kit — the pulse comes from the instruments themselves (strummed or picked parts, the "
                 "voice); there's no beat played on drums")
    _regrid = info.get("tempo_regrid")
    if d.get("groove") and not _regrid:
        b.append(f"The beat: {groove_plain(d['groove'])}")
    elif d.get("groove") and _regrid:
        b.append(f"The beat: this file's beat-by-beat drum reading was worked out at {_regrid} BPM, not the real "
                 "tempo, so its patterns aren't spelled out here"
                 + ("; the snare plays a backbeat, on 2 and 4" if info.get("tempo_snare") else ""))
    meter = meter_words(info)
    if info.get("bpm"):
        b.append(f"Tempo about {round(float(info['bpm']))} BPM" + (f", in {meter}" if meter else ""))
    if drums_programmed(info):
        b.append("It sounds programmed — a drum machine, most likely — with very even, repeatable kick hits")
    snd = dict(d.get("sound") or [])
    kit = []

    def _words_only(v):
        if VARIED_WORDING:
            # 2.3: "tight/damped" -> "tight and damped", "bright//crisp" -> "bright and crisp"
            return [re.sub(r"\s*/+\s*", " and ", x.strip()) for x in split_top(v)
                    if x.strip() and not re.search(r"\d", x)]
        return [x.strip().replace("/", ", ") for x in split_top(v) if x.strip() and not re.search(r"\d", x)]
    if snd.get("Kick") and _words_only(snd["Kick"]) and VARIED_WORDING:
        _kw = _words_only(snd["Kick"])
        _k0 = _kw[0][2:] if _kw[0].startswith("a ") else _kw[0][3:] if _kw[0].startswith("an ") else _kw[0]
        kit.append((_a_an(_k0) if re.search(r"(?i)\bkick$", _k0) else "a kick with " + _a_an(_k0))
                   + ("".join(f", {x}" for x in _kw[1:])))
    elif snd.get("Kick") and _words_only(snd["Kick"]):
        kit.append("a kick with " + join_words([_a_an(x) if not x.startswith(("a ", "an ")) else x
                                                for x in _words_only(snd["Kick"])]))
    if snd.get("Snare / backbeat") and _words_only(snd["Snare / backbeat"]):
        sb = _words_only(snd["Snare / backbeat"])
        kit.append(f"a {sb[0]} snare" + (f" ({'; '.join(sb[1:])})" if sb[1:] else ""))
    if snd.get("Hi-hats") and _words_only(snd["Hi-hats"]):
        kit.append(f"{join_words(_words_only(snd['Hi-hats']))} hi-hats")
    kit = [k for k in kit if k]
    if kit and VARIED_WORDING:
        b.append("Drum sound: " + "; ".join(kit[:-1]) + ("; and " if len(kit) > 1 else "") + kit[-1])
    elif kit:
        b.append("Drum sound: " + join_words(kit))
    if d.get("extra") and not brief:
        ex = [_tag_word(t["label"]) for t in confident_tags(d["extra"], 30) if "drum machine" not in t["label"].lower()]
        if ex:
            b.append("Other percussion in there: " + join_words(ex))
    if (mf.get("DRUMS") and not d.get("groove") and not info.get("no_kit")
            and not re.match(r"(?i)^\s*(?:uncertain|unknown|unclear|n/?a|\(none given\))", mf["DRUMS"])):
        b.append(f"By ear: {clip(mf['DRUMS'], 400)}")
    body = "\n".join(f"- {sentence(x)}" for x in b)
    grid, glabel = typical_grid(info) if not info.get("tempo_regrid") else (None, "")
    pg = groove_grids(info) if (GROOVE_GRIDS_BY_PART and grid and not brief) else []
    if len(pg) >= 2:
        # 2.3: one grid for each part whose groove differs, not one for the whole song
        body += ("\n\nTwo bars of the beat in each part where the groove changes, if someone wants to see them "
                 "(copy each one exactly, in its own code block, with its part and time; x = hit, . = rest, "
                 "| = bar line; kick, snare and hi-hat only). Parts not shown play a groove like one of these:")
        for lab, start, g in pg:
            body += f"\n\n{lab} (from {start}):\n```\n" + "\n".join(g["rows"]) + "\n```"
    elif grid and not brief:
        body += (f"\n\nTwo bars of the {glabel} beat, if someone wants to see it (copy it exactly, in a code block; "
                 "x = hit, . = rest, | = bar line; kick, snare and hi-hat only):\n```\n" + "\n".join(grid["rows"]) + "\n```")
    if body.strip():
        S.append(_section("The groove", body, 40, 70))

    # --- The voice --------------------------------------------------------------------------------------
    if info.get("vocals_present") != "no":
        b = []
        tw = voice_timbre_words(info)
        sv = info.get("section_voice") or {}
        if tw:
            b.append(("The main voice — how it sounds: " if sv else "How it sounds: ") + "; ".join(tw))
        lvc = mf.get("LEAD_VOCAL_CHARACTERISTICS", "")
        if lvc and info.get("vocals_present") == "yes":
            # A sung track whose field says there are no vocals (MOSS-Music:
            # "... structure with no vocals"): those sentences go.
            _keep = [x for x in re.split(r"(?<=[.!?])\s+", lvc)
                     if not re.search(r"(?i)\b(?:no|without|absence of|lacks?)\s+(?:any\s+)?(?:lead\s+)?vocals?\b"
                                      r"(?!\s+(?:fry|processing|effects?|doubling|tuning))|\binstrumental (?:track|piece)\b"
                                      # 2.3: "lacks clear vocal detection, suggesting it may be instrumental"
                                      r"|\blacks?\s+(?:any\s+)?(?:clear|audible|discernible)\s+vocal"
                                      r"|\b(?:may|might|could) be (?:an )?instrumental\b", x)]
            lvc = " ".join(_keep).strip()
        if lvc and not lvc.lower().startswith("see ") and not is_non_answer(lvc):
            b.append("By ear: " + scrub_gender(clip(lvc, 500), gender))
        if sv:
            b.append(f"The second voice: {'a man' if sv['other'] == 'male' else 'a woman'} leads "
                     f"{_span_words(sv['spans'])}" + (f", the line sitting {sv.get('shift') or 'several'}+ semitones "
                                                      f"{sv['direction']} there" if sv.get("direction") else "")
                     + " — the whole-track impressions above describe the main voice")
        mls = melodicism_words(info)
        if mls:
            b.append(mls)
        rec, vhigh = melody_extras(info, "vocals")
        if vhigh:
            b.append(f"Highest note: {pretty_note(vhigh[0])} at {mmss(vhigh[1])}")
        if rec and not brief:
            figs = []
            for f in rec.split("|")[:2]:
                mm = re.match(r"\s*(\S+) \([^)]*\), recurs (\d+)x from (\d+:\d\d)", f)
                if mm:
                    figs.append(f"{pretty_note(mm.group(1))} ({mm.group(2)} times, first at {mm.group(3)})")
            if figs:
                b.append("Little melodic figures that keep coming back: " + "; ".join(figs))
        if info.get("backing") and info["backing"] not in ("none", "uncertain"):
            b.append(f"Backing vocals: {info['backing']}")
        dn = (info.get("vocal_pass") or {}).get("DELIVERY_NOTE", "")
        if "no evident doubling" in dn:
            b.append("Mostly a single voice — no obvious stacked harmonies or call-and-response")
        arr, ev = voice_arrangement(info)
        if arr and not sv and info.get("lead_singers") != "several":
            b.append(f"The arrangement of voices: {arr}")
        if ev and not sv and not brief:
            b.append(f"Other voices, by ear: {scrub_gender(clip(ev, 250), gender)}")
        bn = (info.get("vocal_pass") or {}).get("BACKING_NOTE", "")
        if bn and not is_non_answer(bn) and not brief:
            b.append(f"Backing vocals, by ear: {scrub_gender(clip(bn, 250), gender)}")
        # 0.5 (Music Flamingo) vocal pass fields
        for key, lab in (("VOCAL_STYLE", "Singing style, by ear"), ("VOCAL_EFFECTS", "Vocal effects, by ear")):
            v = (info.get("vocal_pass") or {}).get(key, "")
            if v and not is_non_answer(v) and not brief:
                b.append(f"{lab}: {scrub_gender(clip(v, 250), gender)}")
        if mf.get("LYRIC_SUBJECT") and not is_non_answer(mf["LYRIC_SUBJECT"]):
            b.append(f"What the words are about: {clip(mf['LYRIC_SUBJECT'], 300)}")
        if b:
            S.append(_section("The voice", "\n".join(f"- {sentence(x)}" for x in b), 50, 60))

    # --- Melody and harmony ---------------------------------------------------------------------------------
    if not brief:
        b = []
        kc = info.get("key_centre") or {}
        if kc:
            b.append(f"Key: {kc['centre']} — {kc['centre']} Mixolydian, the notes of {kc['parent']} centred on "
                     f"{kc['centre']}" + (f": it uses {kc['flat7']} rather than {kc['lead']}, a flattened seventh"
                                          if kc.get("flat7") and kc.get("lead") else "")
                     + f" (the bass and the vocal line both lean on {kc['centre']})")
        elif info.get("key"):
            kline = f"Key: {info['key']}"
            if info.get("key_conf") and info["key_conf"].startswith("independent methods agree"):
                kline += " (a clear, confident read)"
            elif info.get("key_conf"):
                kline += f" ({info['key_conf']})"
            if info.get("key_alt"):
                kline += f"; it could also be heard as {info['key_alt']}"
            b.append(kline)
        modal = ""
        for stem in ("bass", "vocals", "other"):
            for l in (info.get("stem_reports") or {}).get(stem) or []:
                m = re.search(r"modal reading: the notes fit ([A-G][#♯b♭]? \w+) rather than plain ([A-G][#♯b♭]? \w+)", l)
                if m and not modal:
                    modal = (f"A flattened seventh keeps turning up, which gives it {_a_an(m.group(1))} colour rather "
                             f"than plain {m.group(2)}")
        if modal and not kc and not info.get("key_fit_switched"):
            # (a modal colour of the key the KEY FIT CHECK set aside is not shown)
            b.append(modal + " (from the transcribed notes, so treat it as a lean, not a certainty)")
        chords = ""
        for l in (info.get("stem_reports") or {}).get("other") or []:
            m = re.search(r"common harmonic clusters: ([A-G][#♯]? (?:major|minor))", l)
            if m:
                chords = m.group(1)
        if chords:
            b.append(f"The most common chord in the keys/synth layer is {chords}")
        cpl = chord_progression_words(info)
        if cpl:
            b.append(cpl)
        if mf.get("CHORDS"):
            meas, mo, heard, ho = parse_chords_line(mf["CHORDS"])
            if meas and cpl:
                meas = []      # 2.3: the measured progression above replaces the transcription's chord list
            if meas:
                b.append("Chords the instruments play most (measured, most common first): " + _chord_words(meas)
                         + (f" — {mo.replace('the measured key of', 'the key of').replace('#', '♯')}" if mo else ""))
            if heard:
                hos = ho.replace("the measured key of", "the key of").replace("#", "♯")
                if hos.startswith("most of these"):
                    hos = "most of these sit outside the key, so they're not reliable"
                b.append(f"Chords by ear (approximate): {clip(', '.join(heard), 300)}" + (f" — {hos}" if hos else ""))
        hm = info.get("harmony_melody") or []
        if hm:
            b.extend(x for x in harmony_summary(info, hm)
                     if not (cpl and x.startswith("Chords by part of the song")))
        mkl = melody_key_words(info)
        if mkl:
            b.append(mkl)
        for k, lab in (("MELODICISM", "Melody and hooks, by ear"), ("MELODIC_CONTOUR", "Shape of the tune, by ear")):
            v = mf.get(k, "")
            if not v or is_non_answer(v):
                continue
            if k in (info.get("report_measured") or set()):
                # dev64+/dev17: measured from the vocal line, not an impression
                lab = "Shape of the tune, section by section (from the transcribed notes)"
                v = re.sub(r"^\s*vocal stem, per section:\s*", "", v)
            b.append(f"{lab}: {scrub_gender(clip(clean_report_value(v), 300), gender)}")
        rec, _h = melody_extras(info, "bass")
        if rec and "octave" in rec:
            b.append("The bass line keeps bouncing between a note and its octave")
        if b:
            S.append(_section("Melody and harmony", "\n".join(f"- {sentence(x)}" for x in b), 60, 40))
    elif CHORD_PROGRESSION_IN_DOSSIER:
        # 2.3: brief gets the key and the measured chord cycle
        b = []
        if info.get("key"):
            b.append(f"Key: {info['key']}")
        cpl = chord_progression_words(info)
        if cpl:
            b.append(cpl)
        if cpl:
            S.append(_section("Melody and harmony", "\n".join(f"- {sentence(x)}" for x in b), 60, 40))

    # --- Era ---------------------------------------------------------------------------------------------------
    e = info.get("era") or {}
    parts = []
    if info.get("year"):
        parts.append(f"Released {info['year']}. That answers “when did it come out?” — what it *sounds* like is "
                     "a separate question, answered from the sound below; if the two differ, that's worth saying.")
    ev = era_evidence_filtered(info)
    if e.get("ERA_ESTIMATE"):
        line = f"By ear it sounds like {e['ERA_ESTIMATE']}"
        if ev:
            line += " — " + "; ".join(ev[:4])
        parts.append(sentence(line))
        # 2.2.2: a by-ear range starting before the 1990s on a loud, limited master.
        _dec = [int(x) for x in re.findall(r"\b(1[89]\d0|20\d0)s\b", e["ERA_ESTIMATE"])]
        if _dec and min(_dec) < 1990 and (info.get("era_range") or ("",))[0] in (
                "late 1990s or later", "late 2000s or later"):
            parts.append("The master doesn't fit the early part of that guess: a loud, limited master like this "
                         "one is uncommon before the late 1990s, so at most the latest part of that range fits.")
    elif info.get("era_description") and not brief:
        desc = "; ".join(era_evidence_filtered(info, info["era_description"]))
        if desc:
            parts.append("A quick listen described the sound rather than dating it: “"
                         + clip(desc, 400) + "” — words like 'modern' there describe the sound, not a date.")
    clues = [c for c in (info.get("era_clues") or []) if not c.startswith(("Era words used by ear", "Genre (measured)"))]
    if clues:
        txt = "What the sound itself points to:\n" + "\n".join(f"- {c}" for c in clues)
        if info.get("era_range") and info["era_range"][1]:
            txt += f"\n- Overall: {info['era_range'][1]}."
        parts.append(txt)
    esc = era_style_words(info)
    if esc:
        parts.append(esc)
    mm = (info.get("era") or {}).get("MISSING_MARKERS", "")
    if mm and not is_non_answer(mm) and not brief:
        parts.append("Later-era sounds it doesn't have, by ear: " + clip(mm, 300).rstrip(".") + ".")
    if info.get("era_caveat"):
        parts.append("About the mastering: " + info["era_caveat"])
    if not brief:
        parts.append("Placing a record by ear: " + ERA_BY_EAR_GUIDE)
    if parts:
        S.append(_section("When it sounds like it's from", "\n\n".join(parts), 70, 60))

    # --- Lyrics -------------------------------------------------------------------------------------------------
    if lyrics and info["lyrics"]:
        txt = info["lyrics"]
        if len(txt) > max_lyrics_chars:
            txt = txt[:max_lyrics_chars].rsplit("\n", 1)[0] + "\n[… truncated …]"
        if info["lyrics_src"] == "tags":
            intro = ("The lyrics as published. Quote a line or two in conversation; the whole thing only "
                     "if asked. Reading meaning or themes from them is fine — mark a personal take as "
                     "your reading when nothing definitive is written down.")
        else:
            intro = ("Words as heard — rough in places; anything marked (?) was hard to make out, so don't present "
                     "it as definite. Quote a line or two; the whole thing only if asked. Reading meaning from "
                     "them is fine as your take when nothing definitive is written down.")
        S.append(_section("Lyrics", intro + "\n\n" + verbatim(txt), 80, 45, truncatable=True))
    rp = info.get("replay") or {}
    if lyrics and full and rp.get("lyric_lines"):
        S.append(_section("When each line is sung (rough wording)", "\n".join(
            f"- {mmss(a)}–{mmss(z)}: {verbatim(clip(strip_hallucinations(t).strip(), 160))}"
            for a, z, t in rp["lyric_lines"] if strip_hallucinations(t).strip()), 82, 15, truncatable=True))

    # --- Exact figures ------------------------------------------------------------------------------------------------
    f = []
    fb = info.get("full_band_tempo") or {}
    if info.get("bpm"):
        f.append(f"Tempo: {round(float(info['bpm']))} BPM" + (f" ({info['tempo_conf']})" if info.get("tempo_conf") else "")
                 + (f"; measured on the full-bandwidth audio — a reading without the top end gave {fb['slow']} BPM, "
                    f"the same pulse counted every {fb.get('factor') or 'few'} beats"
                    + (f" ({fb['hf']}% of the energy is above 11 kHz)" if fb.get("hf") else "")
                    if fb.get("slow") else ""))
    kc = info.get("key_centre") or {}
    if kc:
        f.append(f"Key: {kc['centre']} Mixolydian (the notes of {kc['parent']}, centred on {kc['centre']})")
    elif info.get("key"):
        f.append(f"Key: {info['key']}")
    if meter:
        f.append(f"Meter: {meter}")
    if info.get("duration"):
        f.append(f"Length: {mmss(info['duration'])}")
    if info.get("dynamics"):
        f.append("Loudness: " + re.sub(r"\s*\(([^)]*)\)", r" (\1)", info["dynamics"]))
    if info.get("loudness_context"):
        f.append("Against today's releases: " + info["loudness_context"].split(" Use this")[0])
    for title, pre, lab in (("Mastering peaks & clipping", "true peak", "True peak"),
                            ("Mastering peaks & clipping", "peak-to-loudness ratio", "Peak-to-loudness ratio"),
                            ("Stereo image", "stereo correlation", "Stereo correlation (1 = mono)"),
                            ("Ambience / reverb", "full-mix sustain after hits", "Reverb tail (full mix, approx.)"),
                            ("Bandwidth / file provenance", "upper frequency limit", "Frequency ceiling")):
        v = _prod_line(info, title, pre)
        if v:
            f.append(f"{lab}: {clip(v.split('(')[0].strip() if 'correlation' in pre else v, 120)}")
    sub = _prod_num(info, r"^\s*sub-bass \(20-60 Hz\): ([\d.]+)%")
    if sub is not None:
        f.append(f"Sub-bass share of the energy: {sub:g}%")
    np_ = info.get("note_pitch") or {}
    p = info.get("pitch") or {}
    if info.get("vocals_present") != "no":
        if np_:
            f.append(f"Vocal pitch: centred on {pretty_note(np_['median_note'])} (~{np_['median_hz']} Hz); "
                     f"most notes {pretty_note(np_['lo_note'])}–{pretty_note(np_['hi_note'])} ({np_['lo_hz']}–{np_['hi_hz']} Hz)")
        if p.get("median_hz"):
            f.append(f"Vocal pitch, second reading: centred on {p.get('median_note', '?')} (~{p['median_hz']} Hz)"
                     + (f", range {p['range_notes']}" if p.get("range_notes") else "")
                     + (" — low confidence" if info.get("pyin_low") else ""))
        if info.get("vocal_onset"):
            f.append(f"First sung note: {info['vocal_onset']}")
    for lab, val in (d.get("sound") or []):
        if lab in ("Kick", "Snare / backbeat", "Hi-hats", "Drum bus"):
            f.append(f"{lab}: {clip(drop_parens(val, ('do not', 'measured on')), 140)}")
    if d.get("kit"):
        for lab, val in d["kit"]:
            if lab == "Hits by type":
                f.append(f"Drum hits transcribed: {val} (toms and cymbals aren't counted here)")
    if f:
        S.append(_section("Exact figures", "\n".join(f"- {x}" for x in f), 85, 35))

    # --- Deeper detail -------------------------------------------------------------------------------------------------
    if not brief:
        midi_mode = "full" if level == "full" else ("compressed" if high else None)
        S.extend(deeper_detail(info, full, raw_notes, midi_mode=midi_mode))
    elif raw_notes:
        # An explicit --raw-notes request overrides the brief level only for the
        # raw event dump; ordinary brief output remains unchanged.
        raw_section = raw_notes_section(info)
        if raw_section:
            S.append(raw_section)

    # --- Cover art (opt-in: txt dossiers have no attached image by default) -----------------------------------------
    c = info["cover"]
    if cover and c and not brief:
        bits = []
        for k, lab in COVER_FIELDS:
            v = str(c.get(k) or "").strip()
            if v and v.lower() not in ("none", "uncertain", "unavailable") and k != "confidence":
                bits.append(f"- {lab}: {verbatim(clip(v, 250))}")
        for k, v in (info.get("cover_extra") or {}).items():
            bits.append(f"- {k.replace('_', ' ').capitalize()}: {verbatim(_plain_value(v, 250))}")
        if bits:
            S.append(_section("The cover", "What's on the sleeve (it says nothing about the sound):\n" + "\n".join(bits), 95, 10))

    if full:
        S.extend(file_details(info))
    return [s for s in S if s]


def _timed_pitched_midi_events(rows):
    """Return time-ordered, readable pitched note events from one saved stem.

    Drum hits are deliberately excluded. Musiclyse's compact stem_midi records
    normally use t/d/v/m (onset seconds, duration seconds, relative level,
    MIDI pitch); n and hz are accepted for older/alternate saves too.
    """
    events = []
    for index, row in enumerate(rows or []):
        if not isinstance(row, dict) or row.get("k"):
            continue

        midi_value = None
        note_label = ""
        try:
            if row.get("m") is not None:
                midi_value = float(row["m"])
            elif row.get("hz") is not None:
                hz = float(row["hz"])
                if hz > 0 and math.isfinite(hz):
                    midi_value = 69.0 + 12.0 * math.log2(hz / 440.0)
        except (TypeError, ValueError, OverflowError):
            midi_value = None

        if midi_value is not None and math.isfinite(midi_value):
            midi_number = int(round(midi_value))
            note_label = f"{midi_name(midi_number)} (MIDI {midi_number})"
        else:
            name = row.get("n")
            if name is None or not str(name).strip():
                continue
            note_label = str(name).strip()

        try:
            onset = float(row.get("t"))
            if not math.isfinite(onset):
                onset = None
        except (TypeError, ValueError, OverflowError):
            onset = None
        time_label = f"{onset:.2f}s" if onset is not None else "?s"
        parts = [f"{time_label} {note_label}"]

        for key, label in (("d", "d"), ("v", "level")):
            if row.get(key) is None:
                continue
            try:
                value = float(row[key])
                if not math.isfinite(value):
                    continue
                if key == "d":
                    parts.append(f"d={value:.2f}s")
                else:
                    parts.append(f"level={value:.2f}/1")
            except (TypeError, ValueError, OverflowError):
                continue

        events.append((onset if onset is not None else float("inf"), index, " ".join(parts)))

    events.sort(key=lambda item: (item[0], item[1]))
    return [event for _onset, _index, event in events]


def _midi_note_number(row):
    """Best-effort MIDI pitch number from one saved note record."""
    if not isinstance(row, dict):
        return None
    try:
        if row.get("m") is not None:
            value = float(row["m"])
            return value if math.isfinite(value) else None
        if row.get("hz") is not None:
            hz = float(row["hz"])
            if hz > 0 and math.isfinite(hz):
                return 69.0 + 12.0 * math.log2(hz / 440.0)
    except (TypeError, ValueError, OverflowError):
        pass
    # Some older/alternate exports keep only a note name (for example C#4).
    name = str(row.get("n") or "").strip()
    m = re.match(r"^([A-Ga-g])([#b]?)(-?\d+)$", name)
    if not m:
        return None
    pc_map = {n: i for i, n in enumerate(NOTE_NAMES)}
    pc = pc_map[m.group(1).upper()]
    if m.group(2) == "#":
        pc = (pc + 1) % 12
    elif m.group(2) == "b":
        pc = (pc - 1) % 12
    try:
        octave = int(m.group(3))
    except ValueError:
        return None
    return float((octave + 1) * 12 + pc)


def _midi_note_rows(rows):
    """Normalise pitched stem rows to sorted (onset, duration, MIDI, level) tuples."""
    out = []
    for row in rows or []:
        if not isinstance(row, dict) or row.get("k"):
            continue
        midi = _midi_note_number(row)
        if midi is None:
            continue
        try:
            onset = float(row.get("t"))
            if not math.isfinite(onset) or onset < 0:
                continue
            duration = float(row.get("d", 0.0) or 0.0)
            if not math.isfinite(duration) or duration < 0:
                duration = 0.0
            level = float(row.get("v")) if row.get("v") is not None else None
            if level is not None and not math.isfinite(level):
                level = None
        except (TypeError, ValueError, OverflowError):
            continue
        out.append((onset, duration, midi, level))
    out.sort(key=lambda x: (x[0], x[2]))
    return out


def _midi_interval_name(semitones):
    """Conventional interval name for an absolute semitone distance."""
    names = {
        0: "unison/repeated pitch", 1: "minor 2nd", 2: "major 2nd", 3: "minor 3rd",
        4: "major 3rd", 5: "perfect 4th", 6: "tritone", 7: "perfect 5th",
        8: "minor 6th", 9: "major 6th", 10: "minor 7th", 11: "major 7th", 12: "octave",
    }
    n = int(abs(semitones))
    if n <= 12:
        return names[n]
    octaves, rest = divmod(n, 12)
    if not rest:
        return f"{octaves} octaves"
    return f"{octaves} octave(s) + {names[rest]}"


def _midi_motion_summary(rows):
    """Theory-oriented, cautious summary of the sequential pitches in one section/stem."""
    if len(rows) < 5:
        return ""
    pitches = [int(round(x[2])) for x in rows]
    # Do not infer a single melodic line from a plainly polyphonic part. Repeated
    #/overlapping transcription events can be harmony notes rather than a melody.
    overlap_counts = []
    for onset, _dur, _midi, _level in rows:
        overlap_counts.append(sum(1 for other_on, other_dur, _m, _v in rows
                                  if other_on <= onset < other_on + max(0.03, other_dur)))
    mean_overlap = sum(overlap_counts) / len(overlap_counts) if overlap_counts else 1.0
    low, high = min(pitches), max(pitches)
    base = f"{len(rows)} notes, range {midi_name(low)}–{midi_name(high)}"
    if mean_overlap > 1.8:
        return base + "; frequent note overlap, so read this as a layered/possibly polyphonic part rather than treating consecutive events as one melody"

    deltas = [b - a for a, b in zip(pitches, pitches[1:])]
    if not deltas:
        return base
    rises = sum(d > 0 for d in deltas)
    falls = sum(d < 0 for d in deltas)
    repeats = sum(d == 0 for d in deltas)
    steps = sum(0 < abs(d) <= 2 for d in deltas)
    leaps = sum(abs(d) >= 5 for d in deltas)
    motion = f"motion {rises} rising / {falls} falling / {repeats} repeated steps; {steps}/{len(deltas)} moves are stepwise, {leaps} are leaps of a fourth or wider"
    extras = []
    up = max(deltas)
    down = min(deltas)
    if up >= 5:
        i = deltas.index(up)
        extras.append(f"largest rise +{up} semitones ({_midi_interval_name(up)}, {midi_name(pitches[i])} to {midi_name(pitches[i+1])} near {rows[i+1][0]:.2f}s)")
    if down <= -5:
        i = deltas.index(down)
        extras.append(f"largest fall {down} semitones ({_midi_interval_name(down)}, {midi_name(pitches[i])} to {midi_name(pitches[i+1])} near {rows[i+1][0]:.2f}s)")
    return base + "; " + motion + ("; " + "; ".join(extras) if extras else "")


def _compress_pitched_midi_events(rows, gap_tol=0.12):
    """Compress repeated pitches while retaining each onset, duration and saved level.

    Events with numeric MIDI pitches can be run-length compressed. Older/alternate
    records with an arbitrary note-name string remain as individual events rather
    than being silently lost. Untimed events are also retained as untimed singles.
    """
    events = []
    for index, row in enumerate(rows or []):
        if not isinstance(row, dict) or row.get("k"):
            continue
        midi_value = _midi_note_number(row)
        try:
            midi_value = float(midi_value) if midi_value is not None else None
            if midi_value is not None and not math.isfinite(midi_value):
                midi_value = None
        except (TypeError, ValueError, OverflowError):
            midi_value = None
        name = str(row.get("n") or "").strip()
        if midi_value is None and not name:
            continue
        try:
            onset = float(row.get("t"))
            if not math.isfinite(onset) or onset < 0:
                onset = None
        except (TypeError, ValueError, OverflowError):
            onset = None
        try:
            duration = float(row.get("d", 0.0) or 0.0)
            if not math.isfinite(duration) or duration < 0:
                duration = 0.0
        except (TypeError, ValueError, OverflowError):
            duration = 0.0
        try:
            level = float(row.get("v")) if row.get("v") is not None else None
            if level is not None and not math.isfinite(level):
                level = None
        except (TypeError, ValueError, OverflowError):
            level = None
        pitch = int(round(midi_value)) if midi_value is not None else None
        label = f"{midi_name(pitch)} (MIDI {pitch})" if pitch is not None else name
        events.append({"index": index, "onset": onset, "duration": duration, "level": level,
                       "pitch": pitch, "label": label})
    events.sort(key=lambda x: (x["onset"] if x["onset"] is not None else float("inf"), x["index"]))
    if not events:
        return [], 0

    runs = []
    for event in events:
        onset, duration, pitch, level = (event["onset"], event["duration"],
                                         event["pitch"], event["level"])
        end = onset + duration if onset is not None else None
        can_merge = (
            pitch is not None and onset is not None and runs
            and runs[-1]["pitch"] == pitch and runs[-1]["start"] is not None
            and onset - runs[-1]["end"] <= gap_tol
        )
        if can_merge:
            run = runs[-1]
            run["count"] += 1
            run["end"] = max(run["end"], end if end is not None else onset)
            run["onsets"].append(onset)
            run["durations"].append(duration)
            run["levels"].append(level)
        else:
            runs.append({
                "pitch": pitch, "label": event["label"], "start": onset,
                "end": end, "count": 1, "duration": duration,
                "onsets": [onset], "durations": [duration], "levels": [level],
            })

    def _same_rounded(values, digits=2):
        return not values or len({round(float(v), digits) for v in values}) == 1

    fragments = []
    for run in runs:
        name = run["label"]
        if run["count"] >= 2:
            onsets = ", ".join(f"{x:.2f}" for x in run["onsets"])
            parts = [f"{name}×{run['count']} (onsets {onsets}s"]
            durations = run["durations"]
            if any(d > 0 for d in durations):
                if _same_rounded(durations):
                    parts.append(f"d={durations[0]:.2f}s each")
                else:
                    parts.append("d=[" + ", ".join(f"{d:.2f}" for d in durations) + "]s")
            levels = run["levels"]
            valid_levels = [v for v in levels if v is not None]
            if valid_levels:
                if len(valid_levels) == len(levels) and _same_rounded(valid_levels):
                    parts.append(f"level={valid_levels[0]:.2f}/1 each")
                else:
                    parts.append("levels=[" + ", ".join("?" if v is None else f"{v:.2f}" for v in levels) + "]/1")
            span = (f"span {run['start']:.2f}–{run['end']:.2f}s"
                    if run["start"] is not None and run["end"] is not None else "span unknown")
            parts.append(span + ")")
            fragments.append(", ".join(parts))
        else:
            onset = f"{run['start']:.2f}s " if run["start"] is not None else "?s "
            parts = [onset + name]
            if run["duration"] and run["duration"] >= 0.05:
                parts.append(f"d={run['duration']:.2f}s")
            if run["levels"][0] is not None:
                parts.append(f"level={run['levels'][0]:.2f}/1")
            fragments.append(" ".join(parts))
    return fragments, len(events)


def _midi_coverage_note(info):
    """Explain whether the save tells us the stored MIDI event list is complete."""
    coverage = info.get("midi_coverage") or {}
    if coverage.get("kind") == "partial":
        try:
            seconds = float(coverage.get("seconds"))
            amount = f"the first {seconds:g} seconds"
        except (TypeError, ValueError):
            amount = "only part of the track"
        return (f"The saved stem/MIDI report says the transcription covers {amount} only; "
                "later notes and drum hits are not present in this event list, even if the completeness flag is set. "
                "Do not treat it as a full-track transcription.")
    complete = info.get("midi_complete")
    if complete is True and coverage.get("kind") == "full":
        return ("The save marks the stored event transcription as complete and its stem/MIDI report says it covers "
                "the full track; it is still an estimate, not a guaranteed perfect score.")
    if complete is True:
        return ("The save marks the stored event transcription as complete relative to the transcription pipeline; "
                "the report does not independently confirm its time coverage, and it is still an estimate, "
                "not a guaranteed perfect score.")
    if complete is False:
        return ("The save marks the stored event transcription as incomplete or sampled; "
                "the event list is not exhaustive and should not be treated as a complete score.")
    return ("This older save does not say whether the stored event transcription is complete; "
            "treat its event list as potentially partial.")


def tracked_beats_section(info):
    """Render all saved beat-grid timestamps instead of only count and endpoints."""
    rp = info.get("replay") or {}
    beats = []
    for value in rp.get("beats") or []:
        try:
            t = float(value)
            if math.isfinite(t) and t >= 0:
                beats.append(t)
        except (TypeError, ValueError, OverflowError):
            continue
    if not beats:
        return None
    beats.sort()
    cells = [f"{t:.3f}s" for t in beats]
    lines = [" · ".join(cells[i:i + 12]) for i in range(0, len(cells), 12)]
    body = (
        "Detected beat-grid positions from the saved structure replay, in seconds from the "
        "start of the analysed audio (or excerpt). These are the beat tracker's beat times, "
        "not a claim that every percussion hit lands on a beat.\n" + "\n".join(lines)
    )
    return _section("Detail: tracked beat timestamps", body, 98.55, 2, truncatable=True)


def timed_drum_events_section(info, compressed=False):
    """Render saved drum-hit timestamps; group simultaneous events in compressed mode."""
    raw = info.get("raw_notes") or {}
    rows = raw.get("drums") if isinstance(raw, dict) else None
    if not isinstance(rows, list) or not rows:
        return None
    events = []
    for idx, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        try:
            onset = float(row.get("t"))
            if not math.isfinite(onset) or onset < 0:
                continue
        except (TypeError, ValueError, OverflowError):
            continue
        label = re.sub(r"[_-]+", " ", str(row.get("k") or row.get("n") or "percussion hit").strip())
        extras = []
        try:
            duration = float(row["d"]) if row.get("d") is not None else None
            if duration is not None and math.isfinite(duration) and duration >= 0:
                extras.append(f"d={duration:.3f}s")
        except (TypeError, ValueError, OverflowError):
            pass
        try:
            level = float(row["v"]) if row.get("v") is not None else None
            if level is not None and math.isfinite(level):
                extras.append(f"level={level:.2f}/1")
        except (TypeError, ValueError, OverflowError):
            pass
        events.append((onset, idx, label + (" (" + ", ".join(extras) + ")" if extras else "")))
    events.sort(key=lambda x: (x[0], x[1]))
    if not events:
        return None

    lines = []
    if compressed:
        by_time = {}
        for onset, _idx, label in events:
            by_time.setdefault(onset, []).append(label)
        cells = [f"{onset:.3f}s " + ", ".join(labels)
                 for onset, labels in sorted(by_time.items())]
        lines = [" · ".join(cells[i:i + 8]) for i in range(0, len(cells), 8)]
        intro = ("Drum hits grouped by exact saved onset time. Each listed event retains its "
                 "drum type and saved duration/level where available; simultaneous hits share "
                 "one timestamp.")
    else:
        cells = [f"{onset:.3f}s {label}" for onset, _idx, label in events]
        lines = [" · ".join(cells[i:i + 10]) for i in range(0, len(cells), 10)]
        intro = ("Every saved transcribed drum event, in time order. Times are seconds from "
                 "the start of the analysed audio (or excerpt); durations and levels appear "
                 "where they were saved.")
    body = _midi_coverage_note(info) + "\n\n" + intro + "\n" + "\n".join(lines)
    title = "Detail: drum-hit timestamps (grouped)" if compressed else "Detail: every drum-hit timestamp"
    return _section(title, body, 98.65, 1, truncatable=True)


def compressed_pitched_midi_section(info):
    """High-level pitched MIDI: same theory cues as full, run-length events.

    Discussion capability matches the full timed list for melody, range, contour,
    intervals and section timing; repeated pitches are collapsed so the block is
    smaller. Onset timestamps are always present.
    """
    raw = info.get("raw_notes") or {}
    if not isinstance(raw, dict) or not raw:
        return None

    blocks = []
    theory_lines = []
    sections = info.get("sections") or []
    stem_order = STEM_ORDER + sorted(set(raw) - set(STEM_ORDER))
    for stem in stem_order:
        if stem == "drums":
            continue
        all_rows = _midi_note_rows(raw.get(stem))
        if not all_rows:
            continue
        if sections:
            for sec in sections:
                try:
                    a, b = float(sec.get("start")), float(sec.get("end"))
                except (TypeError, ValueError):
                    continue
                if not math.isfinite(a) or not math.isfinite(b) or b <= a:
                    continue
                part_rows = [r for r in all_rows if a <= r[0] < b]
                if len(part_rows) < 5:
                    continue
                summary = _midi_motion_summary(part_rows)
                if summary:
                    sec_name = str(sec.get("name") or "Section").strip()
                    theory_lines.append(
                        f"- {sec_name} ({mmss(a)}–{mmss(b)}), {stem}: {summary}."
                    )

        fragments, n_notes = _compress_pitched_midi_events(raw.get(stem))
        if not fragments:
            continue
        n_events = len(fragments)
        header = (f"- {stem.capitalize()} part ({n_notes} pitched notes → "
                  f"{n_events} compressed events):")
        rows = [header]
        for start in range(0, len(fragments), 8):
            rows.append("  " + "; ".join(fragments[start:start + 8]))
        blocks.append("\n".join(rows))

    if not blocks:
        return None
    theory = (
        "MUSIC-THEORY USE: Treat these as approximate note transcriptions, not a perfect score. "
        "In a repeated-pitch run, the note name and MIDI number are written once with ×N, followed by "
        "the complete comma-separated onset-time "
        "list, the per-note duration(s) and the overall span; every saved onset is retained. "
        "Use those onsets to discuss note-level rhythm rather than assuming evenly spaced attacks. "
        "For a single-line part, compare consecutive pitch numbers: the "
        "signed difference is the interval in semitones; pitch-class relationships ignore "
        "octave. Use the section's stated key and the existing 'Melody and harmony' section "
        "as context, then check whether notes act as chord tones, passing tones, suspensions, "
        "extensions, repeated motifs or contrasting phrases. Compare overlapping notes across "
        "vocals, bass and accompaniment for candidate voicings, inversions, doubling, contrary "
        "motion and voice-leading. Treat chord names as candidates when notes overlap or "
        "transcription is uncertain; don't force every note into the key. These MIDI notes "
        "describe transcribed pitches and durations, not the acoustic overtone/harmonic "
        "spectrum of the sound."
    )
    if theory_lines:
        theory += ("\n\nSection-by-section pitch motion (computed from the transcribed note "
                   "events; use the compressed event list below to verify timing and pitch):\n"
                   + "\n".join(theory_lines))
    body = (
        _midi_coverage_note(info) + "\n\n" + theory + "\n\nCompressed timed note events "
        "from the separated audio parts. Each event gives note name and MIDI number plus onset time in seconds from the start; "
        "repeated-pitch runs list every individual onset, their per-note duration(s), and the "
        "overall span while writing the pitch name only once. Compare pitches that overlap "
        "across parts for melody, intervals, chord tones, "
        "voicing and changes over time. These are separated-part labels, not proof of the "
        "exact instrument; low-level parts may contain bleed. Drum hits are omitted here "
        "because the export already carries the drum-event detail.\n\n" + "\n\n".join(blocks)
    )
    return _section("Compressed MIDI: vocal and pitched instrument notes", body, 98.8, 1,
                    truncatable=True)


def full_pitched_midi_section(info):
    """Full-only exact note data plus compact, cautious music-theory cues.

    The summaries describe transcription geometry (range, contour and intervals),
    not a guaranteed score; raw timestamps remain available below for checking.
    """
    raw = info.get("raw_notes") or {}
    if not isinstance(raw, dict) or not raw:
        return None

    blocks = []
    theory_lines = []
    sections = info.get("sections") or []
    stem_order = STEM_ORDER + sorted(set(raw) - set(STEM_ORDER))
    for stem in stem_order:
        if stem == "drums":
            continue
        all_rows = _midi_note_rows(raw.get(stem))
        if not all_rows:
            continue
        # The compact overview is split using the saved section map. It makes
        # section-by-section theory questions easier without replacing events.
        if sections:
            for sec in sections:
                try:
                    a, b = float(sec.get("start")), float(sec.get("end"))
                except (TypeError, ValueError):
                    continue
                if not math.isfinite(a) or not math.isfinite(b) or b <= a:
                    continue
                part_rows = [r for r in all_rows if a <= r[0] < b]
                if len(part_rows) < 5:
                    continue
                summary = _midi_motion_summary(part_rows)
                if summary:
                    sec_name = str(sec.get("name") or "Section").strip()
                    theory_lines.append(
                        f"- {sec_name} ({mmss(a)}–{mmss(b)}), {stem}: {summary}."
                    )

        events = _timed_pitched_midi_events(raw.get(stem))
        if not events:
            continue
        rows = [f"- {stem.capitalize()} part ({len(events)} pitched events):"]
        for start in range(0, len(events), 10):
            rows.append("  " + "; ".join(events[start:start + 10]))
        blocks.append("\n".join(rows))

    if not blocks:
        return None
    theory = (
        "MUSIC-THEORY USE: Treat these as approximate note transcriptions, not a perfect score. "
        "For a single-line part, compare consecutive MIDI pitch numbers: the signed difference is the "
        "interval in semitones; pitch-class relationships ignore octave. Use the section's stated key "
        "and the existing 'Melody and harmony' section as context, then check whether notes act as chord "
        "tones, passing tones, suspensions, extensions, repeated motifs or contrasting phrases. Compare "
        "overlapping notes across vocals, bass and accompaniment for candidate voicings, inversions, "
        "doubling, contrary motion and voice-leading. Treat chord names as candidates when notes overlap "
        "or transcription is uncertain; don't force every note into the key. These MIDI notes describe "
        "transcribed pitches and durations, not the acoustic overtone/harmonic spectrum of the sound."
    )
    if theory_lines:
        theory += "\n\nSection-by-section pitch motion (computed from the transcribed note events; use the event list below to verify exact details):\n" + "\n".join(theory_lines)
    body = (
        _midi_coverage_note(info) + "\n\n" + theory + "\n\nTimed note events transcribed from the separated audio parts. Each event gives onset time "
        "in seconds from the start, pitch (note name and MIDI number where available), duration, and "
        "relative level from 0 to 1 where available. Compare pitches that overlap across parts to explore "
        "melody, intervals, chord tones, voicing and changes over time. These are separated-part labels, "
        "not proof of the exact instrument; low-level parts may contain bleed. Drum hits are omitted here "
        "because the export already carries the drum-event detail.\n\n" + "\n\n".join(blocks)
    )
    return _section("Full MIDI: vocal and pitched instrument notes", body, 98.8, 1, truncatable=True)


def raw_notes_section(info):
    """Render every saved pitched note and drum hit as a raw timed event list."""
    raw = info.get("raw_notes") or {}
    if not raw:
        return None
    blocks = []
    for st in STEM_ORDER + sorted(set(raw) - set(STEM_ORDER)):
        rows = raw.get(st)
        if not rows:
            continue
        cells = []
        for r in rows:
            if not isinstance(r, dict):
                continue
            try:
                t = f"{float(r.get('t', 0)):.2f}"
            except (TypeError, ValueError, OverflowError):
                t = "?"
            nm = r.get("k") or r.get("n")
            if not nm:
                try:
                    nm = midi_name(r["m"])
                except (KeyError, TypeError, ValueError, OverflowError):
                    nm = "?"
            extra = []
            for key, fmt in (("d", "{:.2f}s"), ("v", "v{:.2f}")):
                if r.get(key) is not None:
                    try:
                        extra.append(fmt.format(float(r[key])))
                    except (TypeError, ValueError, OverflowError):
                        pass
            cells.append(f"{t} {nm}" + (" " + " ".join(extra) if extra else ""))
        if cells:
            blocks.append(f"{st} ({len(cells)}): " + "; ".join(cells))
    if not blocks:
        return None
    return _section(
        "Every transcribed note",
        _midi_coverage_note(info) + "\n\nRaw note events (seconds, note or drum, length, level 0-1). "
        "Approximate machine transcription; for exact-timing questions only.\n\n" + "\n\n".join(blocks),
        99, 1, truncatable=True)


def deeper_detail(info, full, raw_notes, midi_mode=None):
    """Per-section detail and raw measurements, for specific questions.

    midi_mode: None (standard/brief — no pitched MIDI block), "compressed"
    (--level high), or "full" (--level full). --raw-notes still
    suppresses the pitched MIDI block and emits its own every-note dump.
    """
    S = []
    d = info.get("drums") or {}
    # groove by section
    rows = []
    for sg in ([] if info.get("tempo_regrid") else (d.get("sections") or [])):
        g = groove_plain(sg.get("groove") or "") or join_words(
            [groove_plain(sg.get(k) or "") for k in ("kick", "snare") if sg.get(k)] + ([_hat_plain(sg["hats"])] if sg.get("hats") else []))
        lab = sg["label"]
        if "," in sg["ranges"]:
            lab = _base_name(lab) + " sections"
        if g:
            rows.append(f"- {lab} ({sg['ranges']}): {g}")
        if full:
            for gr in sg["grids"]:
                rows.append(_grid_block(gr, title=""))
    timeline = drum_timeline(info)
    body = ""
    if timeline:
        _tl = []
        for a, b, c in timeline:
            # 2.3: neighbouring stretches with the same kit are one stretch
            if VARIED_WORDING and _tl and drum_config_plain(_tl[-1][2]) == drum_config_plain(c):
                _tl[-1] = (_tl[-1][0], b, _tl[-1][2])
            else:
                _tl.append((a, b, c))
        body += "Which drums are playing when:\n" + "\n".join(
            f"- {mmss(a)}–{mmss(b)}: {drum_config_plain(c)}" for a, b, c in _tl)
        fe = [l for l in d.get("timeline") or [] if l.strip().startswith("first entry")]
        if fe:
            body += "\n- " + fe[0].strip().replace("first entry per drum type", "first time each is heard")
    if rows:
        body += ("\n\n" if body else "") + "The beat in each part of the song:\n" + "\n".join(rows)
    if body:
        S.append(_section("Detail: drums through the song", body, 86, 30))

    # intensity table
    names, faint = part_names(info)
    stems = [s for s in STEM_ORDER if s in (info.get("intensity") or {}) and s in names]
    if stems:
        snames = [names[s].replace("the ", "") for s in stems]
        order = {s["name"]: i for i, s in enumerate(info["sections"])}
        secs = []
        for st in stems:
            for nm, _v in info["intensity"][st]:
                if nm not in secs:
                    secs.append(nm)
        secs.sort(key=lambda x: order.get(x, len(order)))
        rows = ["| Section | " + " | ".join(snames) + " |", "|---" * (len(stems) + 1) + "|"]
        for nm in secs:
            if _base_name(nm).lower() == "silence":
                continue
            cells = [INTENSITY_SHORT.get(dict(info["intensity"][st]).get(nm, ""), dict(info["intensity"][st]).get(nm, ""))
                     for st in stems]
            rows.append(f"| {nm} | " + " | ".join(cells) + " |")
        S.append(_section("Detail: how loud each part is, section by section",
                          "Each part against its own loudest moment (so it shows builds and drop-outs, not which "
                          "instrument is louder than another):\n\n" + "\n".join(rows), 87, 25))

    # melody per section (vocal line), condensed
    lines = []
    for l in (info.get("melody") or {}).get("vocals") or []:
        m = re.match(r"^(.+?) \((\d+:\d\d)-(\d+:\d\d)\): (.+)$", l.strip())
        if not m or m.group(4).startswith("no sustained"):
            continue
        mel = melody_for(info, "vocals", m.group(1).strip())
        if not mel:
            continue
        t = f"- {m.group(1)}: {pretty_note(mel.get('lo', '?'))}–{pretty_note(mel.get('hi', '?'))}"
        if mel.get("contour"):
            t += f", {mel['contour']}"
        if mel.get("motion"):
            t += f"; {mel['motion']}"
        lines.append(t)
    if lines and info.get("vocals_present") != "no":
        S.append(_section("Detail: the vocal line, section by section",
                          "From the transcribed notes (approximate): range and shape of the tune in each part.\n"
                          + "\n".join(lines), 88, 20))

    # 2.3: how sung or rhythm-led the voice is, part by part (dev45 MELODICISM BY SECTION)
    ml = info.get("melodicism") or {}
    if ml.get("rows"):
        S.append(_section("Detail: how melodic the singing is, section by section",
                          "From the lead vocal's held notes and range and how fast the words come, judged on the "
                          "voice only (the instruments can still carry a tune):\n"
                          + "\n".join(f"- {r['name']} ({mmss(r['start'])}–{mmss(r['end'])}): {r['text']}"
                                       for r in ml["rows"]), 88.2, 19))
    # 2.3: the tune against the key (dev45 MELODY IN THE KEY BY SECTION)
    mk = info.get("melody_key") or {}
    if mk.get("rows"):
        S.append(_section("Detail: the tune against the key, section by section",
                          (f"Scale degrees against {mk['key']} (1 = {mk['tonic']}; numbered against the major scale, so "
                           "b3, b6 and b7 are the minor-key notes), from the longer, held notes — where the line "
                           "rests and peaks, not a note-by-note score:\n" if mk.get("key") else
                           "Where the line rests and peaks against the key, from the longer, held notes:\n")
                          + "\n".join(f"- {r['name']} ({mmss(r['start'])}–{mmss(r['end'])}): {r['text']}"
                                       for r in mk["rows"]), 88.3, 18))
    # 2.3: the measured chord progression (dev45 CHORD PROGRESSION BY SECTION)
    cp = info.get("chord_prog") or {}
    if cp.get("sections"):
        rows = []
        for r in cp["sections"]:
            if r["none"]:
                rows.append(f"- {r['name']} ({mmss(r['start'])}–{mmss(r['end'])}): no clear chords")
                continue
            bits = [r["chords"] + (f" ({r['roman']})" if r["roman"] else "")]
            if r["rhythm"]:
                bits.append(r["rhythm"])
            if r["modal"]:
                bits.append("most chords here lie outside the key (a key change or a modal passage)")
            elif r["outside"]:
                bits.append("outside the key: " + r["outside"])
            rows.append(f"- {r['name']} ({mmss(r['start'])}–{mmss(r['end'])}): " + "; ".join(bits))
        if cp.get("from_segments"):
            cp_intro = ("Fallback from the chord segments saved separately in the JSON, in time order. "
                        "The detailed analysis block was absent, so Roman numerals and exact loop counts are not available. "
                        "Chord labels can be uncertain on a dense mix.\n")
        else:
            cp_intro = ("Measured from the recording, in time order (a loop is written once with how many times it "
                        "goes round)" + (f"; Roman numerals are against {cp['key']}" if cp.get("key") else "")
                        + ". Chord names can slip on a dense mix, so 7ths and sus chords are likely rather than "
                        "certain.\n")
        S.append(_section("Detail: the chord progression, section by section",
                          cp_intro + "\n".join(rows), 88.4, 18))

    # harmony and the lead line (dev17 HARMONY AND MELODY BY SECTION)
    hl = harmony_detail_lines(info.get("harmony_melody") or [], info)
    if hl and cp.get("sections"):
        # 2.3: the measured progression above names the chords; the transcription's
        # most-common chords would contradict it, so only the lead line is kept here.
        hl = [re.sub(r": (?:chords [^;]*|no clear chords); ", ": ", l) for l in hl]
        hl = [l for l in hl if not l.endswith("no clear lead line")]
        if hl:
            S.append(_section("Detail: the lead line against the harmony, section by section",
                              "From the transcribed notes (approximate tendencies): which part carries the tune, where "
                              "it sits against the chords and the bass, and how much of it lands on the chord notes.\n"
                              + "\n".join(hl), 88.5, 18))
    elif hl:
        S.append(_section("Detail: chords and the lead line, section by section",
                          "From the transcribed notes (approximate tendencies, not an exact progression): the chords "
                          "heard most in each part, which part carries the tune, where it sits against the chords "
                          "and the bass, and how much of it lands on the chord notes.\n" + "\n".join(hl), 88.5, 18))

    # 2.3: every chord change with its time (high and full)
    ct = info.get("chord_timeline") or []
    if full and ct:
        lines, cur = [], []
        for a, name in ct:
            cur.append(f"{mmss(a)} {name}")
            if len(cur) == 10:
                lines.append(" · ".join(cur))
                cur = []
        if cur:
            lines.append(" · ".join(cur))
        S.append(_section("Detail: every chord change, with its time",
                          "Measured from the recording (each time is where that chord starts; 'no chord' is a stretch "
                          "with no clear harmony):\n" + "\n".join(lines), 89.5, 12, truncatable=True))

    if full:
        # everything else, as measured
        blocks = []
        for st in STEM_ORDER + sorted(set(info.get("stem_reports") or {}) - set(STEM_ORDER)):
            ls = [l for l in (info.get("stem_reports") or {}).get(st) or []
                  if not re.match(r"\s*(?:full (?:note|hit) list|transcription model|stem energy|note count after|"
                                  r"\d+:\d\d-\d+:\d\d:|first entry|(?:count|kick|snare|hi-hat|toms|cymbals|perc|other)\s*:|"
                                  r"DRUM ARRANGEMENT|BEAT GRID|ADDITIONAL PERCUSSION|CYMBALS \(|Drum/percussion stem)", l)]
            ls = measurement_lines(ls, True, 400)
            if ls:
                blocks.append(f"{st.capitalize()} part:\n" + "\n".join(ls))
        if blocks:
            S.append(_section("Detail: each separated part as transcribed",
                              "Separated parts are buckets ('other' = keys, synths, strings, horns), not confirmed "
                              "instruments.\n\n" + "\n\n".join(blocks), 89, 12))
        prod = []
        for title, ls in info.get("production") or []:
            keep = []
            for l in ls:
                line = _drop_writer_orders(l) or l
                # Rewrite stock ambience-reading labels in the detail block too,
                # so full-level dossiers don't reintroduce "washy/reverb-heavy".
                if title == "Ambience / reverb" and line.lower().startswith("ambience reading"):
                    if ":" in line:
                        head, val = line.split(":", 1)
                        plain = plain_ambience(val.strip())
                        if plain:
                            line = f"{head.strip()}: {plain}"
                keep.append(line)
            if keep:
                prod.append(f"{title}:\n" + "\n".join(f"- {l}" for l in keep))
        if prod:
            S.append(_section("Detail: production measurements", "\n\n".join(prod), 90, 10))
        mb = []
        for blk in info.get("measure_blocks") or []:
            ls = measurement_lines(blk["lines"], True)
            if ls:
                mb.append(f"{blk['title']}:\n" + "\n".join(ls))
        if mb:
            S.append(_section("Detail: further signal measurements", "\n\n".join(mb), 91, 8))
        if info.get("identity_reason"):
            S.append(_section("Detail: how the singer was identified", info["identity_reason"], 92, 8))
        if info.get("vocals_present") != "no":
            g = singer_gender(info)
            vx = dict(info.get("vocal_pass") or {})
            vx.update({k: v for k, v in (info.get("vocal_extra") or {}).items() if k not in vx})
            rows = []
            for k, lab in (("FORMANT_NOTE", "Size of the voice (resonance)"), ("LEAD_CHECK_NOTE", "Second listen"),
                           ("CONFIDENCE_REASON", "Why the second listen was sure")):
                v = vx.get(k, "")
                if v and not is_non_answer(v):
                    rows.append(f"- {lab}: {scrub_gender(clip(v, 300), g)}")
            arr, ev = voice_arrangement(info)
            if arr:
                rows.append(f"- Voices: {arr}")
            if ev:
                rows.append(f"- Other voices: {scrub_gender(clip(ev, 300), g)}")
            for q in (info.get("voice_desc") or {}).get("quotes") or []:
                if re.search(r"[A-Z_]{4,}\s*=", q):      # a report field caught in the quote, not a description
                    continue
                rows.append(f"- Heard as: \u201c{scrub_gender(clip(q, 200), g)}\u201d")
            pe = vx.get("PROBABILITY_ESTIMATE", "")
            if pe and not is_non_answer(pe):
                rows.append(f"- First-pass estimate of age and gender (\"Who's singing\" above settles it): {clip(pe, 200)}")
            if rows:
                S.append(_section("Detail: how the voice was read",
                                  "Impressions from the listening passes; where they differ from \"Who's singing\" "
                                  "or \"The voice\", those sections win.\n" + "\n".join(rows), 92.5, 7))
        cs = info.get("cross_stem") or []
        if cs:
            S.append(_section("Detail: instruments heard on more than one part", "\n".join(
                f"- {_tag_word(a['label'])}: {a['detail']}" + (f" — {a['caution']}" if a.get("caution") else "")
                for a in cs), 89.5, 9))
        rp = info.get("replay") or {}
        hints = rp.get("mf_hints") or []
        if rp.get("beats") or hints:
            rows = []
            if rp.get("beats"):
                bt = rp["beats"]
                rows.append(f"- Beats tracked: {len(bt)}, from {mmss(bt[0])} to {mmss(bt[-1])}")
            if hints:
                rows.append("- What each part sounded like by ear (names from a first listen, not the section map): "
                            + "; ".join(f"{mmss(a)}–{mmss(b)} {re.sub(r'(?<=[a-z])(?=[0-9])', ' ', lab)}"
                                        for a, b, lab in hints))
            S.append(_section("Detail: how the song was split up", "\n".join(rows), 93.5, 5))
        rp = info.get("replay") or {}
        if rp.get("lyric_groups") or info.get("lyric_repeats"):
            rows = ([f"- “{verbatim(t)}” ×{n}, at {times}" for t, n, times in info.get("lyric_repeats") or []]
                    or [f"- “{verbatim(clip(t, 80))}” ×{n}" + (f", at {', '.join(mmss(x) for x in occ)}" if occ else "")
                        for t, n, occ in rp.get("lyric_groups") or []])
            S.append(_section("Detail: lines that come back", "\n".join(rows), 93, 8))
    # High/full exports expose the entire saved beat grid, not only its count and endpoints.
    # The raw-notes mode already lists all saved drum hits; otherwise high/full must expose them here.
    if midi_mode in ("compressed", "full"):
        beat_section = tracked_beats_section(info)
        if beat_section:
            S.append(beat_section)
        if not raw_notes:
            drum_section = timed_drum_events_section(info, compressed=(midi_mode == "compressed"))
            if drum_section:
                S.append(drum_section)

    # 2.2.3e: full → timed MIDI; high → compressed MIDI with timestamps.
    # standard and brief stay unchanged. --raw-notes keeps its existing explicit
    # all-stems behaviour and suppresses this section to avoid duplicates.
    if not raw_notes and midi_mode == "full":
        midi_section = full_pitched_midi_section(info)
        if midi_section:
            S.append(midi_section)
    elif not raw_notes and midi_mode == "compressed":
        midi_section = compressed_pitched_midi_section(info)
        if midi_section:
            S.append(midi_section)

    if raw_notes:
        raw_section = raw_notes_section(info)
        if raw_section:
            S.append(raw_section)
    return S


def file_details(info):
    fd, b = info.get("file_details") or {}, []
    if fd.get("label"):
        b.append(f"- File: {verbatim(str(fd['label']))}")
    if fd.get("saved_at"):
        b.append(f"- Analysed: {str(fd['saved_at'])[:16].replace('T', ' ')}")
    sig = fd.get("audio_file_signature")
    if sig:
        try:
            b.append(f"- File size: {int(sig[2]) / 1048576:.1f} MB")
        except (TypeError, ValueError, IndexError):
            pass
    if info.get("excerpt"):
        a, z = info["excerpt"]
        b.append(f"- Only an excerpt was analysed ({mmss(a)}–{mmss(z)}); times are relative to it")
    for k, v in (info.get("meta_extra") or {}).items():
        b.append(f"- {k.replace('_', ' ').capitalize()} (file tag): {verbatim(_plain_value(v, 200))}")
    for k, v in (info.get("other_fields") or {}).items():
        b.append(f"- {k.replace('_', ' ').capitalize()}: {verbatim(_plain_value(v, 400))}")
    return [_section("About the file", "\n".join(b), 97, 5)] if b else []


def assemble(sections, max_tokens=None, heading="##"):
    secs = sorted(sections, key=lambda s: s["order"])

    def render(ss):
        parts = []
        for s in ss:
            if s.get("no_heading") or not (s.get("title") or "").strip():
                parts.append(s["body"])
            else:
                parts.append(f"{heading} {s['title']}\n\n{s['body']}")
        return "\n\n".join(parts)

    text = render(secs)
    if max_tokens is not None:
        shed = [(p, rx, s) for s in secs for p, rx in _SHED.get(s.get("title") or "", [])
                if re.search(rx, s["body"])]
        while est_tokens(text) > max_tokens:
            droppable = [s for s in secs if s["prio"] < 90]
            if shed and (not droppable or min(p for p, _rx, _s in shed) <= min(s["prio"] for s in droppable)):
                k = min(range(len(shed)), key=lambda i: shed[i][0])
                _p, rx, sec = shed.pop(k)
                sec["body"] = re.sub(rx, "", sec["body"])
                text = render(secs)
                continue
            # If ordinary low-priority sections are exhausted, continue trimming
            # higher-priority sections too; otherwise a requested budget can be
            # exceeded indefinitely. This is a last resort after priority shedding.
            candidates = droppable or secs
            if not candidates:
                break
            victim = min(candidates, key=lambda s: s["prio"])
            if victim["trunc"] and len(victim["body"]) > 160:
                shortened = victim["body"][: len(victim["body"]) // 2].rsplit("\n", 1)[0]
                victim["body"] = (shortened + "\n[… truncated to fit …]") if shortened else "[… truncated to fit …]"
            else:
                secs.remove(victim)
            text = render(secs)
    return text


# =============================================================================
# Encyclopedia background from the Musiclyse Wikipedia database
# =============================================================================
#
# Independent of the chat-time lookup in Musiclyse and the LM Studio plugin
# (which guess names out of a question). This one follows the links a reader
# would follow:
#
#   1. the ARTIST's own article, found from the file's artist tag and checked
#      to be a music article (not a disambiguation page or a namesake);
#   2. the artist's DISCOGRAPHY -- the discography section(s) of that article
#      plus a separate "<Artist> discography" article if the database has one
#      -- whose entries are looked up one by one; an entry is kept only if its
#      article says it is an album / EP / single / song BY this artist;
#   3. the MEMBERS listed on the artist's article (a members / line-up /
#      personnel section, or "the band consists of ..." in the lead); a person
#      is kept only if their article mentions the band.
#
# The track's own album and song are always tried as well. Everything is read
# with the database opened read-only.

import sqlite3

WIKI_DB_DEFAULT_NAME = "music_wiki_heavy.db"

_WIKI_TITLE_HINTS = ("title", "page_title", "article_title", "name")
_WIKI_TEXT_HINTS = ("text", "body", "content", "extract", "summary", "article", "wikitext")
_WIKI_REDIRECT_FROM = ("from", "source", "alias", "from_title")
_WIKI_REDIRECT_TO = ("to", "target", "dest", "destination", "canonical", "to_title")

_PUNCT_FOLD = str.maketrans({
    "‘": "'", "’": "'", "‚": "'", "′": "'",
    "“": '"', "”": '"', "„": '"', "″": '"',
    "–": "-", "—": "-", "−": "-",
})

# Words that make an article a music article (artist pages and people).
_MUSIC_WORDS_RE = re.compile(
    r"\b(?:band|group|duo|trio|quartet|quintet|singer|songwriter|musician|rapper|DJ|"
    r"record producer|composer|vocalist|guitarist|bassist|drummer|pianist|keyboardist|"
    r"saxophonist|violinist|ensemble|orchestra|supergroup|boy band|girl group|"
    r"rock|pop|hip hop|jazz|punk|metal|folk|electronic|soul|country music|album|record label)\b",
    re.I,
)
_GROUP_WORDS_RE = re.compile(
    r"\b(?:band|group|duo|trio|quartet|quintet|ensemble|orchestra|supergroup|boy band|girl group|collective)\b",
    re.I,
)
_DISAMBIG_RE = re.compile(r"\b(?:may (?:also )?refer to|can refer to|is the name of|may mean)\b", re.I)
_RELEASE_WORDS_RE = re.compile(
    r"\b(?:album|EP|extended play|LP|mixtape|compilation|soundtrack|single|song|live album|"
    r"box set|anthology|remix album)\b",
    re.I,
)
_SINGLE_FIRST_RE = re.compile(r"\b(?:single|song)\b", re.I)
_ALBUM_FIRST_RE = re.compile(r"\b(?:album|EP|extended play|LP|mixtape|compilation|soundtrack|box set|anthology)\b", re.I)

_DISCOG_HEAD_RE = re.compile(
    r"discograph|\balbums?\b|\bsingles?\b|\bEPs?\b|extended plays?|\breleases?\b|mixtapes?|compilations?",
    re.I,
)
_MEMBERS_HEAD_RE = re.compile(r"\bmembers?\b|personnel|line-?up|musicians", re.I)
_NOT_MEMBERS_HEAD_RE = re.compile(r"touring|session|guest|additional|production|technical|timeline|former label", re.I)
_MEMBERS_LEAD_RE = re.compile(
    # How Wikipedia leads introduce a line-up, past and present tense:
    # "consists of", "originally consisted of", "comprised", "the line-up was",
    # "formed in 1987 by", "fronted by", "joined by".
    r"\b(?:consist(?:s|ed)? of|consisting of|compris(?:e|es|ed|ing)(?: of)?|composed of|made up of|"
    r"members (?:are|were|include|included|have included)|"
    r"line-?up (?:of|is|was|includes|included|consists of|consisted of|comprises|comprised|features|featured)|"
    r"(?:formed|founded|started|created)\b[^.]{0,90}?\bby|fronted by|led by|joined by)\b",
    re.I,
)
# Sentences that add a member later: "Guitarist Firstname Lastname joined in 1967."
_MEMBER_JOINED_RE = re.compile(r"\b(?:joined|was added|were added|replac(?:ed|ing)|became (?:a|the) (?:new )?member)\b", re.I)
_PERSON_DESCRIPTORS = {
    "vocalist", "vocalists", "singer", "singers", "guitarist", "guitarists", "bassist", "bassists",
    "drummer", "drummers", "keyboardist", "keyboardists", "pianist", "percussionist", "saxophonist",
    "violinist", "trumpeter", "rapper", "rappers", "dj", "producer", "frontman", "frontwoman",
    "multi-instrumentalist", "instrumentalist", "songwriter", "students", "student", "brothers",
    "brother", "sisters", "sister", "cousin", "cousins", "lead", "rhythm", "co-founder", "founder",
    "founders", "member", "members", "original", "former", "current", "new", "later", "and", "older",
    "younger", "twin", "upon", "after", "with", "developed", "founded", "formed", "in", "the",
}
# "X were an English band" is about a group; "John Doe is the guitarist of the rock
# band the alt-rock band" is about a person, so no preposition or role word may come between.
_ROLE_OR_LINK_WORDS = (
    r"of|for|in|with|from|by|at|on|to|and|whose|who|that|which|"
    r"vocalist|singer|guitarist|bassist|drummer|keyboardist|pianist|percussionist|saxophonist|"
    r"violinist|trumpeter|rapper|frontman|frontwoman|founder|co-founder|member|leader|musician|"
    r"lyricist|songwriter|producer|manager|lead|DJ|MC|multi-instrumentalist|instrumentalist|organist|"
    r"cellist|harmonica|player|composer|arranger|engineer|husband|wife|son|daughter|father|mother|"
    r"brother|sister|one|part|half|third")
_GROUP_SUBJECT_RE = re.compile(
    r"\b(?:is|are|was|were) (?:an?|the) (?:(?!(?:" + _ROLE_OR_LINK_WORDS + r")\b)[\w'’/-]+,? ){0,6}?"
    r"(?:band|group|duo|trio|quartet|quintet|ensemble|orchestra|collective|supergroup)\b", re.I)
# What a first sentence is about when it is not a recording: "X is a 2019 Canadian crime film",
# "X is an episode of a TV series whose title comes from a song by ...".
_NON_RELEASE_SUBJECT_RE = re.compile(
    r"\b(?:film|movie|television|TV|sitcom|series|episode|miniseries|telefilm|documentary|novel|novella|"
    r"book|short story|poem|play|musical|opera|ballet|video game|game|painting|sculpture|artwork|"
    r"comic|manga|anime|magazine|newspaper|website|podcast|radio (?:show|programme|program)|"
    r"character|company|brand|restaurant|ship|horse|racehorse|building|village|town|city|river|"
    r"mountain|album cover|award|festival|tour|concert|venue|club|team|football|cocktail|"
    r"organi[sz]ation|political|software|company)\b", re.I)
_RELEASE_JUNK_RE = re.compile(
    r"^(?:released|release date|label|labels|format|formats|certifications?|peak|peak chart|"
    r"chart|charts|sales|title|year|album details|details|recorded|studio|genre|length|"
    r"producer|producers|songwriters?|writers?|b-side|from the album|see also|notes|references|"
    r"total|us|uk|aus|can|ger|fra|jpn|nz|ire|swe|nor|nld|bel|aut|sui|ita|esp|—|-)\b",
    re.I,
)
_HEADING_LINE_RE = re.compile(r"(?m)^\s*(={2,6})\s*([^=\n]{1,80}?)\s*\1\s*$")
_KNOWN_HEADINGS = (
    "Discography", "Studio albums", "Live albums", "Compilation albums", "Albums", "Singles",
    "EPs", "Extended plays", "Mixtapes", "Members", "Band members", "Current members",
    "Former members", "Personnel", "Line-up", "Lineup", "History", "Background",
    "Track listing", "Reception", "Critical reception", "Legacy", "Musical style",
)
_NAME_RE = re.compile(
    r"^([A-ZÀ-Þ][\w'’.\-]*(?:\s+(?:[A-ZÀ-Þ][\w'’.\-]*|de|da|van|von|der|den|la|le|del|du|di|O'|Mc)){0,3})"
)
_NOT_A_NAME = {
    "the", "current", "former", "members", "member", "touring", "lead", "backing", "vocals",
    "guitar", "bass", "drums", "keyboards", "piano", "timeline", "band", "group", "and",
    "with", "also", "other", "early", "original", "session", "live", "studio", "brothers",
    "sisters", "singer", "drummer", "guitarist", "bassist",
}


def _fold(text):
    return "".join(c for c in unicodedata.normalize("NFKD", str(text or "")) if not unicodedata.combining(c))


def wiki_title_key(title):
    t = unicodedata.normalize("NFKC", str(title or ""))
    t = _fold(t).translate(_PUNCT_FOLD)
    return re.sub(r"\s+", " ", t).strip().casefold()


def _name_key(name):
    """Comparison form of a band/person name: no accents or punctuation, no leading 'The'."""
    k = re.sub(r"[^\w\s]", " ", wiki_title_key(name))
    k = re.sub(r"\s+", " ", k).strip()
    return k[4:] if k.startswith("the ") else k


def _mentions(text, name):
    """Does `text` mention `name` (accent/case/punctuation-insensitive, whole words)?"""
    k = _name_key(name)
    if not k:
        return False
    hay = " " + re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", wiki_title_key(text))) + " "
    return f" {k} " in hay


def _same_person_name(title, name):
    """'Firstname J. Lastname' / 'Firstname Lastname (singer)' for 'Firstname Lastname': same first and
    last name, middle names and initials ignored."""
    got, want = _name_key(_base_title(title)).split(), _name_key(name).split()
    if not got or not want:
        return False
    if got == want:
        return True
    return len(want) >= 2 and len(got) >= 2 and got[0] == want[0] and got[-1] == want[-1] and len(got) <= len(want) + 2


def _mentions_proper(text, name):
    """Like _mentions, but a capitalised name must appear capitalised: a group named Word is
    mentioned in "a member of the girl group Word", not in "a word in a song"."""
    if not _mentions(text, name):
        return False
    pat = _artist_pattern(name)
    return bool(pat and re.search(r"\b(?:the[\W_]+)?" + pat + r"\b", _fold(text), re.I))


def _base_title(title):
    """'Band B (band)' -> 'Band B'."""
    return re.sub(r"\s*\([^)]*\)\s*$", "", str(title or "")).strip()


def _lead(text, n=700):
    return re.sub(r"\s+", " ", str(text or "")[:n * 2]).strip()[:n]


_ABBREV_BEFORE_DOT = {"st", "dr", "mr", "mrs", "ms", "jr", "sr", "vs", "no", "vol", "feat", "ft", "mt",
                      "bros", "inc", "co", "ltd", "op", "nos", "approx", "ca", "jan", "feb", "mar", "apr",
                      "aug", "sept", "sep", "oct", "nov", "dec", "gen", "col", "lt", "sgt", "prof", "rev"}


def _first_sentence(text):
    t = _lead(text, 600)
    for m in re.finditer(r"(?<=[a-z0-9)\]\"'”])\.(?:\s+(?=[A-Z\"“'‘(=*#])|(?=[A-Z][a-z]))", t):
        if m.start() < 8:
            continue
        word = re.search(r"([\w]+)$", t[:m.start()])
        if word and (word.group(1).lower() in _ABBREV_BEFORE_DOT or len(word.group(1)) == 1):
            continue    # "St. Louis", "Vol. 2", "J. Smith"
        return t[: m.start() + 1]
    return t


_INFOBOX_KEY_RE = re.compile(r"(?:^|\s)([a-z][a-z0-9_]{1,30}):\s")
_INFOBOX_LIST_KEYS = {"current_members", "past_members", "members", "former_members", "associated_acts",
                      "label", "genre", "occupation", "instrument", "instruments", "producer", "writer",
                      "spinoffs", "spinoff_of", "current_member_of", "past_member_of"}
_INFOBOX_SKIP_KEYS = {"cover", "image", "caption", "alt", "image_size", "landscape", "border", "logo",
                      "signature", "module", "background", "misc", "chronology", "embed", "type_extra"}
_CAMEL_PREFIXES = ("Mc", "Mac", "De", "Di", "Da", "Du", "La", "Le", "Van", "Von", "Fitz", "St")


def _split_run_together(value):
    """Infobox list items that lost their separators: 'Jay LastnameFirstname LastnameFirstname Lastname' ->
    ['Firstname Lastname', 'Firstname Lastname', 'Firstname Lastname']. Keeps MacLastname / McCartney / DeLastname whole."""
    # Bulleted lists ("* Matt Lastname * Greg Lastname", "*Chainsaw Lastname *Lance Lastname") first.
    parts = [p for p in re.split(r"(?:^|\s)[*•·]+\s*|<br\s*/?>|\n", str(value or "")) if p.strip()]
    if len(parts) > 1:
        return [x for p in parts for x in _split_run_together(p)]
    out, start = [], 0
    v = str(value or "")
    for i in range(1, len(v)):
        c, prev = v[i], v[i - 1]
        if not c.isupper():
            continue
        word_start = max(v.rfind(" ", 0, i), v.rfind("\n", 0, i)) + 1
        word = v[word_start:i]
        if prev.islower() or prev in ".)":
            if word in _CAMEL_PREFIXES:
                continue                      # McCartney, MacLastname, DeLastname
            if prev == "." and len(word) <= 2:
                continue                      # "J.Smith"-style initials stay together
        elif prev.isupper():
            # "Name NFirstword Word": a one-letter word glued to the next name ("D" + "Flavor")
            nxt = v[i + 1] if i + 1 < len(v) else ""
            if not (nxt.islower() and len(word) == 1 and " " in v[start:i]):
                continue
        # never after an apostrophe: O'Brien, D'Lastname, O'Lastname stay whole
        else:
            continue
        out.append(v[start:i])
        start = i
    out.append(v[start:])
    return [re.sub(r"\s+", " ", x).strip(" ,;") for x in out if x.strip(" ,;")]


def split_infobox(text, title=""):
    """Many text dumps start an article with its infobox flattened into 'key: value' pairs
    ("name: Title type: Studio album artist: the band released: 1992 ...") before the
    lead. Returns (fields, body): fields as {key: value}, body = the article from the lead on.
    Articles without such a prefix come back unchanged with fields = {}."""
    t = str(text or "")
    if not re.match(r"\s*[a-z][a-z0-9_]{1,30}:\s", t):
        return {}, t
    keys = []
    pos = 0
    while True:
        m = _INFOBOX_KEY_RE.match(t, pos) if not keys else _INFOBOX_KEY_RE.search(t, pos)
        # each key follows straight on from the previous value, never deep inside prose
        if not m or (keys and (m.start() - pos) > 300):
            break
        keys.append((m.group(1), m.end()))
        pos = m.end()
    if not keys:
        return {}, t
    fields = {}
    for i, (k, vstart) in enumerate(keys):
        vend = (_INFOBOX_KEY_RE.search(t, vstart).start() if i + 1 < len(keys) else len(t))
        fields[k] = t[vstart:vend].strip()
    # The last value runs on into the lead: cut it where the article's own subject starts.
    lk = keys[-1][0]
    last = fields[lk]
    cut = None
    for name in [title, _base_title(title), fields.get("name", "")]:
        name = (name or "").strip()
        if not name:
            continue
        hits = [m.start() for m in re.finditer(r"(?:^|\s)[\"“]?" + re.escape(name), last, re.I) if m.start() > 0]
        # prefer the occurrence that opens a sentence about the subject ("X, often ..., was a ...")
        lead_like = [h for h in hits if re.search(r"^[^.]{0,160}?\b(?:is|are|was|were)\b", last[h:])]
        pick = (lead_like or hits or [None])[0]
        if pick is not None:
            cut = pick if cut is None else min(cut, pick)
    if cut is None:
        m = re.search(r"\s([A-Z\u00C0-\u00DE\"“][^.]{0,80}?\s(?:is|are|was|were)\s(?:an?|the|one)\b)", last)
        cut = m.start() if m else min(len(last), 120)
    fields[lk] = last[:cut].strip()
    body_start = keys[-1][1] + (len(t[keys[-1][1]:]) - len(t[keys[-1][1]:].lstrip())) + cut
    body = t[body_start:].strip()
    return fields, body


_PARSED_CACHE = {}
_MUSIC_FIELD_KEYS = ("genre", "instrument", "instruments", "label", "current_members", "past_members",
                     "members", "associated_acts", "years_active")


def parsed(art):
    """(fields, body) for an article tuple (title, text), cached by title."""
    key = (art[0], len(art[1]))
    if key not in _PARSED_CACHE:
        if len(_PARSED_CACHE) > 5000:
            _PARSED_CACHE.clear()
        _PARSED_CACHE[key] = split_infobox(art[1], art[0])
    return _PARSED_CACHE[key]


def _is_music(fields, body):
    return bool(_MUSIC_WORDS_RE.search(_lead(body, 700))) or any(fields.get(k) for k in _MUSIC_FIELD_KEYS) \
        or bool(_MUSIC_WORDS_RE.search(fields.get("occupation", "")))


def _is_group(fields, body):
    return bool(_GROUP_WORDS_RE.search(_lead(body, 700))) or "group_or_band" in fields.get("background", "") \
        or any(fields.get(k) for k in ("current_members", "past_members", "members", "former_members"))


def infobox_names(fields, keys=("current_members", "past_members", "members", "former_members")):
    out = []
    for k in keys:
        # "Firstname Lastname (vocals, bass)", "Firstname Lastname[1]", "See: List of Band B members" are tidied first.
        raw = re.sub(r"\s*(?:\([^)]*\)|\[[^\]]*\])\s*", "\n", fields.get(k, ""))
        for n in (x for line in raw.split("\n") for x in _split_run_together(line)):
            for part in re.split(r",|;|\band\b|&|\n|\s[–—-]\s", n):
                p = re.sub(r"\s+", " ", part).strip(" .:*•–—-")
                if not p or re.match(r"(?i)(?:see|list of|various|others?|none|n/a)\b", p):
                    continue
                if p not in out:
                    out.append(p)
    return out


def _infobox_facts(fields):
    """Readable 'Quick facts' lines from infobox fields (images and layout fields dropped)."""
    lines = []
    for k, v in fields.items():
        if k in _INFOBOX_SKIP_KEYS or not v or k.startswith("image") or k.endswith("_caption"):
            continue
        if k in _INFOBOX_LIST_KEYS:
            v = "; ".join(_split_run_together(v))
        v = re.sub(r"(\d{4})(?=\d{4})", r"\1, ", v)        # "1984–199319951999–present"
        v = re.sub(r"(present)(?=\d)", r"\1, ", v)
        lines.append(f"- {k.replace('_', ' ').capitalize()}: {v}")
    return lines


def clean_wiki_text(text):
    """Light cleanup for display: wiki markup (if the database stores any), headings as bold lines."""
    t = str(text or "").replace("\r\n", "\n")
    if "[[" in t or "{{" in t or "<ref" in t:
        t = re.sub(r"<ref[^>]*/>", "", t)
        t = re.sub(r"<ref[^>]*>.*?</ref>", "", t, flags=re.S)
        for _ in range(3):
            t = re.sub(r"\{\{[^{}]*\}\}", "", t)
        t = re.sub(r"\[\[(?:[^\]|]*\|)?([^\]]*)\]\]", r"\1", t)
        t = re.sub(r"'{2,5}", "", t)
        t = re.sub(r"<[^>]+>", "", t)
    fields, body = split_infobox(t)
    if fields:
        facts = _infobox_facts(fields)
        t = (("Quick facts:\n" + "\n".join(facts) + "\n\n") if facts else "") + body
    t = _HEADING_LINE_RE.sub(lambda m: f"\n**{m.group(2).strip()}**", t)
    t = re.sub(r"[ \t]+\n", "\n", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def wiki_sections_leveled(text):
    """(level, heading, body) triples; the lead is (1, "", lead). Plain heading lines count as level 2."""
    text = str(text or "")
    parts = _HEADING_LINE_RE.split(text)
    if len(parts) > 1:
        out = [(1, "", parts[0])]
        for i in range(1, len(parts) - 2, 3):
            out.append((len(parts[i]), parts[i + 1].strip(), parts[i + 2]))
        return out
    return [(1 if not h else 2, h, b) for h, b in wiki_sections(text)]


def wiki_sections(text):
    """(heading, body) pairs; [0] is the lead. '== H ==' markers, else known heading lines."""
    text = str(text or "")
    parts = _HEADING_LINE_RE.split(text)
    if len(parts) > 1:
        out = [("", parts[0])]
        for i in range(1, len(parts) - 2, 3):
            out.append((parts[i + 1].strip(), parts[i + 2]))
        return out
    known = "|".join(re.escape(h).replace(r"\ ", r"\s+") for h in _KNOWN_HEADINGS)
    pat = re.compile(rf"(?im)^\s*({known})\s*:?\s*$")
    ms = list(pat.finditer(text))
    if not ms:
        return [("", text)]
    out = [("", text[: ms[0].start()])]
    for i, m in enumerate(ms):
        end = ms[i + 1].start() if i + 1 < len(ms) else len(text)
        out.append((m.group(1).strip(), text[m.end():end]))
    return out


class WikiLibrary:
    """Read-only access to the Musiclyse Wikipedia database (schema auto-detected
    the same way Musiclyse detects it)."""

    def __init__(self, path):
        self.path = os.path.abspath(path)
        uri = "file:" + self.path.replace("?", "%3F").replace("#", "%23") + "?mode=ro"
        self.conn = sqlite3.connect(uri, uri=True, timeout=5.0)
        self.conn.row_factory = sqlite3.Row
        self.schema = self._detect()
        if not self.schema:
            self.conn.close()
            raise ValueError(f"{self.path} has no table with a title column and a text column")
        self._cache = {}

    # --- schema --------------------------------------------------------------
    @staticmethod
    def _pick(cols, hints):
        lower = {c.lower(): c for c in cols}
        for h in hints:
            for lc, orig in lower.items():
                if lc == h or lc.endswith("_" + h) or h in lc:
                    return orig
        return None

    def _detect(self):
        cur = self.conn.cursor()
        rows = cur.execute("SELECT name, sql, type FROM sqlite_master WHERE type IN ('table','view')").fetchall()
        fts = plain = None
        for r in rows:
            name, sql = r["name"], r["sql"] or ""
            if name.startswith("sqlite_"):
                continue
            if "VIRTUAL TABLE" in sql.upper() and "FTS" in sql.upper():
                m = re.search(r"USING\s+fts\d*\s*\((.*)\)\s*;?\s*$", sql, re.I | re.S)
                cols = [p.strip().split()[0].strip('`"[]') for p in (m.group(1).split(",") if m else [])
                        if p.strip() and "=" not in p]
                t, x = self._pick(cols, _WIKI_TITLE_HINTS), self._pick(cols, _WIKI_TEXT_HINTS)
                if t and x and fts is None:
                    fts = {"mode": "fts", "table": name, "title": t, "text": x}
                continue
            if plain is not None:
                continue
            try:
                cols = [c["name"] for c in cur.execute(f'PRAGMA table_info("{name}")')]
            except sqlite3.Error:
                continue
            t, x = self._pick(cols, _WIKI_TITLE_HINTS), self._pick(cols, _WIKI_TEXT_HINTS)
            if t and x:
                plain = {"mode": "plain", "table": name, "title": t, "text": x}
        schema = fts or plain
        if schema:
            schema["redirect"] = None
            for r in rows:
                name = r["name"]
                if name == schema["table"] or name.startswith("sqlite_"):
                    continue
                if "redirect" not in name.lower() and "alias" not in name.lower():
                    continue
                try:
                    cols = [c["name"] for c in cur.execute(f'PRAGMA table_info("{name}")')]
                except sqlite3.Error:
                    continue
                f, t = self._pick(cols, _WIKI_REDIRECT_FROM), self._pick(cols, _WIKI_REDIRECT_TO)
                if f and t and f != t:
                    schema["redirect"] = {"table": name, "from": f, "to": t}
                    break
        return schema

    def _q(self, name):
        return '"' + str(name).replace('"', '""') + '"'

    @staticmethod
    def _fts_escape(q):
        toks = re.findall(r"[\w'&.!?-]+", q or "")
        toks = [re.sub(r"[’']s$", "", t, flags=re.I) for t in toks]
        return " ".join(f'"{t}"' for t in toks if t)

    # --- lookups ---------------------------------------------------------------
    def _redirect(self, title):
        r = self.schema.get("redirect")
        if not r:
            return None
        sql = (f"SELECT {self._q(r['to'])} AS t FROM {self._q(r['table'])} "
               f"WHERE {self._q(r['from'])} = ? COLLATE NOCASE LIMIT 1")
        cur, seen, target = title, {wiki_title_key(title)}, None
        try:
            for _ in range(3):
                row = self.conn.execute(sql, (cur,)).fetchone()
                if not row or not row["t"]:
                    break
                k = wiki_title_key(row["t"])
                if k in seen:
                    break
                seen.add(k)
                target = cur = str(row["t"])
        except sqlite3.Error:
            return None
        return target

    def article(self, title):
        """(title, text) for an exact title -- case, accent and quote/dash-insensitive -- or via the
        redirect table. None if absent."""
        key = wiki_title_key(title)
        if not key:
            return None
        if key in self._cache:
            return self._cache[key]
        s = self.schema
        T, X, tbl = self._q(s["title"]), self._q(s["text"]), self._q(s["table"])
        hit = None
        try:
            if s["mode"] == "fts":
                q = self._fts_escape(title)
                if q:
                    match = f'{s["title"]} : ({q})'
                    row = self.conn.execute(
                        f"SELECT {T} AS t, {X} AS b FROM {tbl} WHERE {tbl} MATCH ? AND {T} = ? COLLATE NOCASE LIMIT 1",
                        (match, title)).fetchone()
                    if row is None:
                        for r in self.conn.execute(f"SELECT {T} AS t, {X} AS b FROM {tbl} WHERE {tbl} MATCH ? LIMIT 40",
                                                   (match,)):
                            if wiki_title_key(r["t"]) == key:
                                row = r
                                break
                    hit = row
            else:
                hit = self.conn.execute(f"SELECT {T} AS t, {X} AS b FROM {tbl} WHERE {T} = ? COLLATE NOCASE LIMIT 1",
                                        (title,)).fetchone()
            if hit is None:
                target = self._redirect(title)
                if target:
                    hit = self.conn.execute(
                        f"SELECT {T} AS t, {X} AS b FROM {tbl} WHERE {T} = ? COLLATE NOCASE LIMIT 1", (target,)).fetchone()
        except sqlite3.Error:
            hit = None
        res = (str(hit["t"]), str(hit["b"])) if hit is not None and hit["t"] and hit["b"] else None
        self._cache[key] = res
        return res

    def title_words_exist(self, title):
        """Cheap pre-check: does any title contain all of these words? (Every title form tried for a
        release -- 'X (album)', 'X (Artist song)' -- contains them, so if none does, skip them all.)"""
        if self.schema["mode"] != "fts":
            return True
        q = self._fts_escape(title)
        if not q:
            return False
        s = self.schema
        try:
            return self.conn.execute(f"SELECT 1 FROM {self._q(s['table'])} WHERE {self._q(s['table'])} MATCH ? LIMIT 1",
                                     (f'{s["title"]} : ({q})',)).fetchone() is not None
        except sqlite3.Error:
            return True

    def titles_starting(self, base, limit=60):
        """Titles of the form '<base> (...)' -- e.g. 'Name (American band)' -- for disambiguated names."""
        s = self.schema
        T, tbl = self._q(s["title"]), self._q(s["table"])
        out = []
        try:
            if s["mode"] == "fts":
                q = self._fts_escape(base)
                if not q:
                    return []
                rows = self.conn.execute(f"SELECT {T} AS t FROM {tbl} WHERE {tbl} MATCH ? LIMIT ?",
                                         (f'{s["title"]} : ({q})', limit)).fetchall()
            else:
                rows = self.conn.execute(f"SELECT {T} AS t FROM {tbl} WHERE {T} LIKE ? LIMIT ?",
                                         (f"{base} (%", limit)).fetchall()
        except sqlite3.Error:
            return []
        bk = wiki_title_key(base)
        for r in rows:
            t = str(r["t"] or "")
            if t != _base_title(t) and wiki_title_key(_base_title(t)) == bk:
                out.append(t)
        return out

    def titles_about(self, base, must_mention, limit=80, person=False):
        """Titles that are `base` or '<base> (...)' AND whose text mentions every word of
        `must_mention` (a band name, or "band OR singer ..." style alternatives given as a list).
        Unlike titles_starting, a common name ("Firstname Lastname", 100+ articles) can't crowd out the
        right one ("Firstname Lastname (bass guitarist)"), because the text has to match too."""
        s = self.schema
        T, tbl = self._q(s["title"]), self._q(s["table"])
        try:
            if s["mode"] == "fts":
                q = self._fts_escape(base)
                if not q:
                    return []
                if isinstance(must_mention, (list, tuple)):
                    need = " OR ".join(f'"{w}"' for w in must_mention)
                else:
                    toks = [t for t in re.findall(r"\w+", _fold(must_mention))]
                    if len(toks) > 1 and toks[0].lower() == "the":
                        toks = toks[1:]
                    need = " ".join(f'"{t}"' for t in toks)
                if not need:
                    return []
                rows = None
                for title_q in (f'^{q.split(" ", 1)[0]} {q.split(" ", 1)[1]}' if " " in q else f"^{q}", q):
                    try:
                        rows = self.conn.execute(f"SELECT {T} AS t FROM {tbl} WHERE {tbl} MATCH ? LIMIT ?",
                                                 (f'{s["title"]} : ({title_q}) AND {s["text"]} : ({need})', limit)).fetchall()
                        break
                    except sqlite3.Error:
                        continue      # an old SQLite without '^' (initial-token) queries
                rows = rows or []
            else:
                # A plain table has no text index, so a text condition would scan the whole file:
                # take every '<base> (...)' title instead (the caller checks each article anyway).
                rows = self.conn.execute(f"SELECT {T} AS t FROM {tbl} WHERE {T} = ? COLLATE NOCASE OR {T} LIKE ? LIMIT ?",
                                         (base, f"{base} (%", max(limit, 300))).fetchall()
        except sqlite3.Error:
            return []
        keep = _same_person_name if person else (lambda t, b: wiki_title_key(_base_title(t)) == wiki_title_key(b))
        return [str(r["t"]) for r in rows if r["t"] and keep(str(r["t"]), base)]

    def titles_described(self, base, title_words, limit=200, person=False):
        """'<base> (...)' titles whose disambiguator contains one of `title_words`:
        "Word (Swedish group)", "Firstname Lastname (bass guitarist)". The words are matched in the
        title only, so a common name ("Word", "Firstname Lastname") can't crowd the right one out."""
        s = self.schema
        if s["mode"] != "fts":
            return self.titles_starting(base, 300)
        q = self._fts_escape(base)
        if not q or not title_words:
            return []
        first, _, rest = q.partition(" ")
        words = " OR ".join(f'"{w}"' for w in title_words)
        T, tbl = self._q(s["title"]), self._q(s["table"])
        bk = wiki_title_key(base)
        for title_q in (f"^{first} {rest} AND ({words})", f"{q} AND ({words})"):
            try:
                rows = self.conn.execute(f"SELECT {T} AS t FROM {tbl} WHERE {tbl} MATCH ? LIMIT ?",
                                         (f'{s["title"]} : ({title_q})', limit)).fetchall()
                keep = _same_person_name if person else (lambda t, b: wiki_title_key(_base_title(t)) == bk)
                return [str(r["t"]) for r in rows if r["t"] and keep(str(r["t"]), base)]
            except sqlite3.Error:
                continue      # an old SQLite without '^' (initial-token) queries
        return []

    def articles_naming(self, name, also, limit=60):
        """(title, text) of articles whose text contains the exact phrase `name` and every word of
        `also` -- for people whose article is under another name ("Firstname Lastname" -> "Lastname") in a
        database without redirects."""
        s = self.schema
        if s["mode"] != "fts":
            return []
        toks = re.findall(r"\w+", _fold(name))
        btoks = re.findall(r"\w+", _fold(also))
        if len(btoks) > 1 and btoks[0].lower() == "the":
            btoks = btoks[1:]
        if not toks or not btoks:
            return []
        q = f'{s["text"]} : ("{" ".join(toks)}" AND {" ".join(chr(34) + t + chr(34) for t in btoks)})'
        T, X, tbl = self._q(s["title"]), self._q(s["text"]), self._q(s["table"])
        try:
            rows = self.conn.execute(f"SELECT {T} AS t, substr({X}, 1, 2500) AS b FROM {tbl} WHERE {tbl} MATCH ? LIMIT ?",
                                     (q, limit)).fetchall()
        except sqlite3.Error:
            return []
        return [(str(r["t"]), str(r["b"])) for r in rows if r["t"] and r["b"]]

    def close(self):
        try:
            self.conn.close()
        except sqlite3.Error:
            pass


def _clean_artist_tag(artist):
    """Primary act from a tag: 'Band X feat. Singer Y' -> 'Band X' (keeps AB/CD, Surname & Surname)."""
    a = str(artist or "")
    for src, dst in (("․", "."), ("’", "'"), ("‘", "'"), ("．", ".")):
        a = a.replace(src, dst)
    a = re.split(r"\s*(?:\bfeat\b\.?|\bft\b\.?|\bfeaturing\b|\bwith\b\s+[A-Z]|\bvs\b\.?|\s/\s|;)", a,
                 maxsplit=1, flags=re.I)[0]
    return a.strip(" ,;-&")


_RELEASE_TITLE_RE = re.compile(
    r"\([^)]*\b(?:album|EP|song|single|soundtrack|film|TV series|television series|novel|mixtape|"
    r"compilation|video|DVD|tour|book|play|musical)\b[^)]*\)\s*$", re.I)
_RELEASE_SUBJECT_RE = re.compile(
    r"\b(?:is|was)\s+(?:an?|the)\s+(?:(?!(?:band|group|duo|singer|musician|rapper)\b)[\w'’/-]+,?\s+){0,6}?"
    r"(?:album|EP|extended play|LP|single|song|mixtape|compilation|soundtrack|box set|video album)\b", re.I)
_HATNOTE_RE = re.compile(
    r"^(?:this article is about|for (?:the|other|their|his|her)\b[^.]{0,120}?\bsee\b|not to be confused with|"
    r"\S[^.]{0,80}? redirects here|see also|main article)[^.]*\.\s*", re.I)
# "X is a Northern Irish singer-songwriter", "Firstname Lastname was a Scottish drummer"
_PERSON_MUSIC_SUBJECT_RE = re.compile(
    r"\b(?:is|was)\s+(?:an?|the)\s+(?:[\w'’/-]+,?\s+){0,6}?"
    r"(?:singer|songwriter|singer-songwriter|musician|rapper|drummer|guitarist|bassist|DJ|disc jockey|"
    r"producer|record producer|composer|vocalist|pianist|keyboardist|multi-instrumentalist|"
    r"instrumentalist|saxophonist|violinist|trumpeter|cellist|percussionist|MC|frontman|frontwoman|"
    r"recording artist|conductor)\b", re.I)
_ARTIST_INFOBOX_KEYS = ("origin", "years_active", "birth_name", "birth_date", "birth_place", "born", "background",
                        "current_members", "past_members", "members", "former_members", "occupation",
                        "instrument", "instruments", "associated_acts", "death_date", "education", "spouse")
_RELEASE_INFOBOX_SIG = ("released", "recorded", "studio", "length", "prev_title", "next_title", "chronology",
                        "prev_year", "next_year", "misc")


def _lead_sentences(body, n=2):
    """The first `n` sentences of the lead, skipping hatnotes ("This article is about the album.
    For the band, see ..."), which some text dumps keep at the very start."""
    t = _lead(body, 1500)
    for _ in range(3):
        m = _HATNOTE_RE.match(t)
        if not m:
            break
        t = t[m.end():]
    out = []
    for _ in range(n):
        if not t:
            break
        first = _first_sentence(t)
        out.append(first)
        t = t[len(first):].lstrip()
    return " ".join(out)


def _looks_like_release_page(title, fields, body):
    """True for an album / song / film page, which is never the artist's own page, even when it
    has the artist's name ("Firstname Lastname (1977 album)", "Band (Colour Album)", "Word Word")."""
    if _RELEASE_TITLE_RE.search(title):
        return True
    artist_keys = any(fields.get(k) for k in _ARTIST_INFOBOX_KEYS)
    if fields.get("artist") and not artist_keys:
        return True    # an album/song infobox has "artist:"; a band's or singer's has none
    if not artist_keys and _RELEASE_TYPE_RE.search(fields.get("type", "")):
        return True
    if not artist_keys and sum(1 for k in _RELEASE_INFOBOX_SIG if fields.get(k)) >= 2:
        return True
    lead = _lead_sentences(body, 1)
    m = _RELEASE_SUBJECT_RE.search(lead)
    if m and m.start() < 160:
        return True
    heads = " ".join(h for _lv, h, _b in wiki_sections_leveled(body[:60000]))
    if re.search(r"(?i)track ?listing", heads) and not re.search(r"(?i)discography|members|career|early life|biography", heads) \
            and not artist_keys:
        return True
    return False


def _artist_identity(fields, body):
    """Positive evidence that a page is about a musician or a band: an artist infobox, an opening
    that says "X is a ... band" / "X is a ... singer-songwriter", or discography/members sections."""
    if any(fields.get(k) for k in _ARTIST_INFOBOX_KEYS if k not in ("born", "birth_date", "birth_place",
                                                                    "death_date", "education", "spouse")):
        return True
    lead = _lead_sentences(body, 2)
    if _GROUP_SUBJECT_RE.search(lead) or _PERSON_MUSIC_SUBJECT_RE.search(lead):
        return True
    heads = " ".join(h for _lv, h, _b in wiki_sections_leveled(body[:200000]))
    return bool(re.search(r"(?i)discography|band members|\bmembers\b|solo career|musical style", heads))


def _artist_is_group(fields, body):
    """Is the artist page about a band/group (so it has members), rather than a solo musician?
    A session drummer's page that mentions "the band's debut album" is not a group."""
    bg = fields.get("background", "").lower()
    if "group" in bg or "band" in bg:
        return True
    if any(fields.get(k) for k in ("current_members", "past_members", "members", "former_members")):
        return True
    if "solo" in bg or "instrumentalist" in bg or any(fields.get(k) for k in (
            "birth_name", "birth_date", "birth_place", "born", "death_date", "occupation", "spouse")):
        return False
    lead = _lead_sentences(body, 2)
    if _GROUP_SUBJECT_RE.search(lead):
        return True
    if _PERSON_MUSIC_SUBJECT_RE.search(lead) or re.search(r"\((?:born|né|née|\d{1,2} \w+ \d{4}|\w+ \d{1,2}, \d{4})", lead[:200]):
        return False
    return _is_group(fields, body)


# Disambiguators that name an act ("band", "Canadian singer", "rapper"), as opposed to a role
# ("drummer", "guitarist") or another field ("footballer"): those need the track to confirm them.
_ACT_DESCRIPTOR_RE = re.compile(
    r"(?:^|\s)(?:band|musician|singer|singer-songwriter|songwriter|rapper|group|duo|trio|quartet|DJ|artist|"
    r"producer|record producer|collective|ensemble|orchestra|project|entertainer|musical group|music group|"
    r"vocalist|composer|performer)$", re.I)


_ACT_TITLE_WORDS = ("band", "group", "musician", "singer", "rapper", "duo", "trio", "quartet", "DJ", "artist",
                    "producer", "songwriter", "vocalist", "composer", "entertainer", "collective", "ensemble",
                    "orchestra", "project", "performer", "musical", "music")
_MEMBER_TITLE_WORDS = ("musician", "singer", "guitarist", "bassist", "drummer", "keyboardist", "pianist",
                       "vocalist", "rapper", "producer", "songwriter", "composer", "DJ", "percussionist",
                       "saxophonist", "violinist", "trumpeter", "member", "band", "group", "guitar", "bass",
                       "drums", "music", "born", "entertainer", "performer", "artist")


def _find_artist(lib, artist, hints):
    """The artist's own article: tried under the usual Wikipedia title forms, rejected if it is a
    disambiguation page or not about music; ties broken by mentions of the track's album/song."""
    bases = [artist]
    if artist.lower().startswith("the "):
        bases.append(artist[4:])
    cands = []
    for b in bases:
        for d in ("band", "musician", "singer", "rapper", "group", "duo", "singer-songwriter",
                  "DJ", "producer", "record producer", "musical group", "artist"):
            cands.append(f"{b} ({d})")
        if not b.lower().startswith("the "):
            cands.append(f"The {b}")
        cands.append(b)
    for b in bases:
        cands.extend(lib.titles_described(b, _ACT_TITLE_WORDS))
        cands.extend(lib.titles_about(b, ["band", "singer", "musician", "rapper", "group", "duo", "album"]))
        cands.extend(lib.titles_starting(b))
    best, best_score, seen = None, None, set()
    for i, c in enumerate(cands):
        art = lib.article(c)
        if not art or art[0] in seen:
            continue
        seen.add(art[0])
        title, text = art
        fields, body = parsed(art)
        if _DISAMBIG_RE.search(_lead(body, 400)) or not _is_music(fields, body):
            continue
        if _name_key(_base_title(title)) not in (_name_key(b) for b in bases) and not _mentions(text[:5000], artist):
            continue   # a namesake (or a redirect to a page that never uses this name)
        if _looks_like_release_page(title, fields, body):
            continue   # "Firstname Lastname (1977 album)", "Band (Colour Album)", an eponymous debut
        hits = sum(1 for h in hints if h and _mentions(text, h))
        desc = re.search(r"\(([^)]*)\)\s*$", title)
        if desc and not _ACT_DESCRIPTOR_RE.search(desc.group(1)) and any(hints) and not hits:
            continue   # "Firstname Lastname (drummer)" for a track by the singer-songwriter Firstname Lastname: a
                       # namesake whose page names neither the track's album nor the song
        exact = _name_key(_base_title(title)) in (_name_key(b) for b in bases)
        # Among same-named musicians ("Firstname Lastname (drummer)" vs "Firstname Lastname (singer-songwriter)")
        # the one whose page names the track's album or song wins; then the usual title order.
        score = (100 - i * 0.1 + (10 if exact else 0) + (30 if _artist_identity(fields, body) else 0)
                 + 20 * hits)
        if best_score is None or score > best_score:
            best, best_score = art, score
    return best


def _release_candidates(text):
    """Possible release titles in discography text: 'Title (1994)', quoted titles, list lines, table cells."""
    out = []
    for m in re.finditer(r"([^\n,;:|•()\[\]\t]{1,90}?)\s*\(\s*(?:19|20)\d{2}\b[^)\n]{0,40}\)", text):
        out.append(m.group(1))
    quoted = set()
    for m in re.finditer(r"[\"“]([^\"“”\n]{1,90})[\"”]", text):
        out.append(m.group(1))
        quoted.add(wiki_title_key(m.group(1).strip()))
    for line in text.splitlines():
        cells = re.split(r"\s*[|\t]\s*", line) if ("|" in line or "\t" in line) else [line]
        for cell in cells:
            c = re.sub(r"^\s*(?:[-*•#]+|\d{1,3}[.)])\s*", "", cell)
            # "1987 – Word Word Word" / "June 1987 – Title": the year comes first, the title after it
            ym = re.match(r"^(?:[A-Z][a-z]{2,8}\.?\s+)?(?:\d{1,2}\s+)?(?:19|20)\d{2}\s*[–—-]\s*(.+)$", c.strip())
            c = ym.group(1) if ym else re.split(r"\s+[–—-]\s+", c, maxsplit=1)[0]
            c = re.sub(r"\s*\((?:19|20)\d{2}[^)]*\)\s*$", "", c)
            out.append(c)
    seen, clean = set(), []
    for c in out:
        c = c.strip().strip("\"'“”‘’*•").strip()
        c = re.sub(r"[.:,;]+$", "", c).strip()
        if not c or len(c) > 80 or len(c.split()) > 12 or not re.search(r"[^\W\d_]", c):
            continue
        if _RELEASE_JUNK_RE.match(c) or re.fullmatch(r"[\d\W_]+", c) or "http" in c.lower():
            continue
        k = wiki_title_key(c)
        # A quoted title is a song/single; the same title unquoted is usually the album
        # (Title X the album and "Title X" the single), so both are kept.
        for hint in (("single",) if k in quoted else ()) + ("any",):
            if (k, hint) not in seen:
                seen.add((k, hint))
                clean.append((c, hint))
    return clean[:500]


def _find_release(lib, title, artist_name, exclude_titles, hint="any"):
    """The article for a release by this artist, or None. Kind is 'album' or 'single'.
    hint="single" tries the song/single title forms first."""
    a = artist_name
    if not lib.title_words_exist(title):
        return None
    # Wikipedia often drops a leading "The" in disambiguators:
    # "Title (Band album)" for the band.
    forms = [a] + ([a[4:]] if a.lower().startswith("the ") else [])
    own_albums = tuple(f"{title} ({f} {k})" for f in forms for k in ("album", "EP"))
    own_songs = tuple(f"{title} ({f} {k})" for f in forms for k in ("song", "single"))
    albums = own_albums + (f"{title} (album)", f"{title} (EP)")
    songs = own_songs + (f"{title} (song)", f"{title} (single)")
    order = (songs + albums) if hint == "single" else (own_albums + own_songs + albums[len(own_albums):] + songs[len(own_songs):])
    for cand in order + (title,):
        art = lib.article(cand)
        if not art or art[0] in exclude_titles:
            continue
        kind = _release_kind(art, a)
        if kind:
            return art, kind
    return None


# Values of the album/song infobox "type" field ("studio", "live", "compilation", "ep", "single", ...).
_RELEASE_TYPE_RE = re.compile(r"\b(?:album|single|song|EP|extended play|LP|mixtape|compilation|soundtrack|studio|"
                              r"live|remix|box(?: set)?|demo|split|promotional|promo|cover|greatest|tribute|video|"
                              r"double|debut|instrumental|bootleg|re-?release|reissue|anthology)\b", re.I)
# Words that, between "is"/"was" and the release word, show the subject is something else:
# "Hugh Lastname is a Canadian graphic artist best known for album cover concepts".
_NOT_THE_RECORD_RE = re.compile(
    r"\b(?:artist|designer|photographer|musician|singer|songwriter|producer|engineer|manager|actor|actress|"
    r"writer|author|director|person|man|woman|politician|player|DJ|presenter|host|critic|journalist|"
    r"company|known|named|whose|who|which|named after|"
    r"used|appears?|appeared|cover art|artwork)\b", re.I)
# "a song by Name, covered by the band": the band is not the song's artist.
_COVERED_RE = re.compile(r"\b(?:cover(?:ed|s)?|sampled|remixed|parod(?:y|ied)|interpolat\w*|inspired|tribute|"
                         r"later|also|originally|translated|adapted|reworked|answer|in response)\b", re.I)
_RELEASE_INFOBOX_KEYS = ("released", "recorded", "label", "length", "producer", "studio", "format",
                         "b-side", "b_side", "album", "prev_title", "next_title", "writer", "genre")


def _artist_pattern(artist_name):
    """Regex matching the artist's name in folded text, with or without a leading 'The'. A word
    written with a capital in the name must have its capital in the text too (Wikipedia always
    capitalises names), so a group named "Word" doesn't match "a word in a song"; the rest of each
    word is case-blind ("WORD", "Word")."""
    toks = [t for t in re.findall(r"\w+", _fold(artist_name)) if t]
    if len(toks) > 1 and toks[0].lower() == "the":
        toks = toks[1:]

    def tok(t):
        if t[0].isupper():
            return "(?-i:" + re.escape(t[0]) + ")" + re.escape(t[1:])
        return re.escape(t)
    return r"[\W_]+".join(tok(t) for t in toks) if toks else None


def _ib_first_artist(value):
    """'Band featuring X' / 'A and B' / 'A, B' -> 'the band'."""
    v = _split_run_together(value)[0] if value else ""
    v = re.sub(r"\([^)]*\)|\[[^\]]*\]", " ", v)
    return re.split(r"\s+(?:featuring|feat\.?|ft\.?|with|vs\.?|and|&|x)\s+|,|;|/", v, flags=re.I)[0].strip()


def _first_sentence_release_by(first, artist_name):
    """'album' / 'single' when an opening sentence says the article is a recording BY this artist
    ("Title is the fourth album by Canadian rock band the band", "Title Y is
    the band' biggest hit single"), else None. Films, episodes, novels and the like
    that are named after a song ("an episode ... whose title comes from a song by ...") fail."""
    pat = _artist_pattern(artist_name)
    if not pat or _DISAMBIG_RE.search(first) or re.search(r"various artists", first, re.I):
        return None
    f = _fold(first)
    if not _RELEASE_WORDS_RE.search(f):
        return None
    cop = re.search(r"\b(?:is|was|are|were)\b", f[:200])
    if not cop:
        return None          # not a defining sentence ("Juno Awards of 1991 honoured the best album by ...")
    pred = cop.end()         # what the subject is said to be starts here

    def says_record(m):
        if m.start() < pred:
            return False
        between = f[pred: m.start()]
        return len(between) <= 80 and not _NON_RELEASE_SUBJECT_RE.search(between) \
            and not _NOT_THE_RECORD_RE.search(between) and not _COVERED_RE.search(m.group(0))
    # The name must stand alone: not the end of a longer name ("Firstname Lastname" for Lastname, "Firstname Lastname"
    # for Firstname) nor the start of one ("Firstname Lastname" for Name, "Title Name" for Title).
    # A lower-case word ("band", "singer"), "the", or the start of the clause may come before it.
    alone_after = r"(?-i:(?![ \t-]{1,2}[A-Z\u00C0-\u00DE][a-z\u00DF-\u00FF]))"
    name_re = r"\b(?:the[\W_]+)?" + pat + r"\b" + alone_after
    by = None
    for m in re.finditer(r"\b(?:album|EP|extended play|LP|mixtape|compilation|soundtrack|single|song|box set|"
                         r"recording|release)s?\b[^.;]{0,100}?\b(?:by|of|from|recorded by|performed by|released by)\b"
                         r"([^.;]{0,70}?)" + name_re, f, re.I):
        gap = m.group(1)
        last = re.findall(r"[\w'’.-]+", gap)
        if last and last[-1][0].isupper() and last[-1].lower() not in ("the",) and \
                not re.fullmatch(r"(?:American|British|English|Canadian|Australian|Irish|Scottish|Welsh|"
                                 r"Swedish|German|French|Japanese|Korean|Norwegian|Danish|Finnish|Dutch|"
                                 r"Icelandic|Italian|Spanish|Brazilian|Mexican|Jamaican|New|Zealand|"
                                 r"Northern|Belgian|Austrian|Swiss|Polish|Russian)", last[-1]):
            continue          # "a song by Firstname Lastname"
        by = m
        break
    own = re.search(r"\b(?:the[\W_]+)?" + pat + r"(?:['’]s?|[\W_]+band['’]s?)?" + alone_after +
                    r"\s+(?:[\w'’-]+\s+){0,5}?(?:album|EP|LP|single|song|mixtape|compilation)s?\b", f, re.I)
    if own:
        prev_word = re.findall(r"[\w'’.-]+$", f[: own.start()].rstrip())
        if prev_word and prev_word[0][0].isupper() and prev_word[0].lower() != "the":
            own = None        # "is Firstname Lastname's third album"
        elif not re.fullmatch(r"\s*(?:(?:an?|the|one of|another|also)\s+)*", f[pred: own.start()], re.I):
            own = None        # the name has to be what the subject IS, not a word further on
    if not (by and says_record(by)) and not (own and says_record(own)):
        return None          # "is a 2019 crime film ... the song by ...", "is a graphic artist ..."
    ms, ma = re.search(r"\b(?:single|song|track)\b", f, re.I), _ALBUM_FIRST_RE.search(f)
    return "single" if ms and (not ma or ms.start() < ma.start()) else "album"


def _release_kind(art, artist_name):
    """'album' / 'single' when the article is a release BY this artist, else None. A release
    infobox whose (first) 'artist:' is this artist settles it; otherwise the opening sentence
    has to say so."""
    fields, body = parsed(art)
    if _DISAMBIG_RE.search(_lead(body, 300)):
        return None
    ib_artist = fields.get("artist", "")
    if ib_artist:
        typ = fields.get("type", "")
        looks_like_release = bool(_RELEASE_TYPE_RE.search(typ)) or (
            not typ and sum(1 for k in _RELEASE_INFOBOX_KEYS if fields.get(k)) >= 1
            and not any(fields.get(k) for k in ("medium", "series", "episode", "season", "director",
                                                 "starring", "author", "publisher", "pages", "developer",
                                                 "platforms", "network", "location", "dimensions")))
        if not looks_like_release:
            return None      # a painting's or a TV episode's infobox, not a record's
        if _name_key(_ib_first_artist(ib_artist)) != _name_key(artist_name):
            return None
        if re.search(r"single|song", typ, re.I):
            return "single"
        if re.search(r"album|EP|extended play|LP|mixtape|compilation|soundtrack|live|remix|box set", typ, re.I):
            return "album"
        kind = _first_sentence_release_by(_first_sentence(body), artist_name)
        if kind:
            return kind
        return "single" if fields.get("album") or fields.get("b-side") or fields.get("b_side") else "album"
    return _first_sentence_release_by(_lead_sentences(body, 1), artist_name)


_NAME_RUN_RE = re.compile(
    r"(?<![\w'’])([A-Z\u00C0-\u00DE][\w'’.\-]*(?:\s+(?:[A-Z\u00C0-\u00DE][\w'’.\-]*|de|da|van|von|der|den|la|le|del|du|di)){0,3})"
)


def _names_in(part, allow_single=False):
    """Person-name-shaped runs of capitalised words in a fragment ("bassist/keyboardist Firstname
    Middlename Lastname", "students Firstname Lastname", "Name's older brother, Firstname Lastname")."""
    out = []
    p = re.sub(r"\([^)]*\)", " ", part)
    for m in _NAME_RUN_RE.finditer(p):
        words = m.group(1).rstrip(".").split()
        while words and words[0].lower().strip("'’") in _PERSON_DESCRIPTORS | _NOT_A_NAME:
            words = words[1:]
        if words and re.search(r"['’]s$", words[-1]):
            continue                      # "Name's" is a possessive, not a name
        n = " ".join(words)
        if not words or len(n) < 3 or n.lower() in _NOT_A_NAME:
            continue
        if len(words) == 1 and not allow_single:
            continue
        out.append(n)
    return out


def _member_candidates(artist_text, members_list_text="", prose=True):
    """Names offered as members: a members / line-up / personnel section (with any sub-sections
    such as "Current" / "Former"), a "List of <band> members" article, and the lead's own
    line-up sentences ("comprised ...", "originally consisted of ...", "... joined in 1967")."""
    names = []

    def push(n):
        if n not in names:
            names.append(n)

    def from_line(line):
        l = re.sub(r"^\s*(?:[-*•#]+|\d{1,3}[.)])\s*", "", line).strip()
        if not l:
            return
        head = l.split(":", 1)[0] if ":" in l[:40] else ""
        if head and re.search(r"members?|line-?up|personnel", head, re.I):
            for part in re.split(r",|;|\band\b|&", l.partition(":")[2]):   # "Current members: A, B and C"
                for n in _names_in(part, allow_single=len(part.split()) <= 3):
                    push(n)
            return
        m = _NAME_RE.match(l)            # a credit line: "Firstname Lastname – vocals", "Nickname (bass)"
        if not m:
            return
        name, rest = m.group(1).strip(), l[m.end():]
        if len(name.split()) == 1 and not re.match(r"\s*[–—\-(:,|]", rest):
            return
        for n in _names_in(name, allow_single=True)[:1]:
            push(n)

    def from_prose(text):
        lead = re.sub(r"\s+", " ", text)[:4000]
        for sent in re.split(r"(?<=[.!?])\s+(?=[A-Z])", lead):
            m = _MEMBERS_LEAD_RE.search(sent)
            if m:
                for part in re.split(r",|;|\band\b|&", sent[m.end():]):
                    for n in _names_in(part, allow_single=len(re.sub(r"\([^)]*\)", " ", part).split()) <= 3):
                        push(n)
            if _MEMBER_JOINED_RE.search(sent):
                for n in _names_in(sent):
                    push(n)

    for text in (artist_text, members_list_text):
        if not text:
            continue
        secs = wiki_sections_leveled(text)
        in_members = 0                    # level of the members section we're inside, 0 = none
        for level, h, body in secs[1:]:
            if in_members and level <= in_members:
                in_members = 0
            if _MEMBERS_HEAD_RE.search(h) and not _NOT_MEMBERS_HEAD_RE.search(h):
                in_members = in_members or level
            elif not in_members or _NOT_MEMBERS_HEAD_RE.search(h):
                continue
            for line in body.splitlines():
                from_line(line)
        if prose:
            from_prose(secs[0][2])
    return names[:30]


def _find_member(lib, name, band_title, band_name, why=None):
    """The person's article, or None. `why` (a list) collects the reason for a miss: one note per
    article looked at, so a report shows exactly which test turned each one down."""
    want = _name_key(name).split()
    if not want:
        if why is not None:
            why.append("not a usable name")
        return None
    cands = [name] + [f"{name} ({d})" for d in (
        "musician", "singer", "guitarist", "bassist", "bass guitarist", "drummer", "keyboardist", "rapper",
        "record producer", "producer", "composer", "singer-songwriter", "vocalist", "pianist", "songwriter",
        f"{band_name} drummer", f"{band_name} guitarist", f"{band_name} bassist", f"{band_name} singer",
        f"{band_name} member", f"{band_name} musician")]
    # Every '<name> (...)' article that mentions the band: finds "Firstname Lastname (bass guitarist)" or
    # "Firstname Lastname (Band drummer)" however many other John/Roger Lastnames there are.
    band_words = tuple(w for w in re.findall(r"\w+", _fold(band_name)) if w.lower() != "the")
    cands += (lib.titles_described(name, _MEMBER_TITLE_WORDS + band_words, person=True)
              + lib.titles_about(name, band_name, person=True) + lib.titles_starting(name, 30))
    notes, seen = [], set()
    for cand in cands:
        art = lib.article(cand)
        if not art or art[0] == band_title or art[0] in seen:
            continue
        seen.add(art[0])
        fields, body = parsed(art)
        got = _name_key(_base_title(art[0])).split()
        if not got or (got[-1] != want[-1] and not got[0].startswith(want[0][:3])
                       and _name_key(fields.get("name", "")) != _name_key(name)):
            notes.append(f'"{art[0]}": a different name')    # a redirect to some other page
            continue
        if _DISAMBIG_RE.search(_lead(body, 300)):
            notes.append(f'"{art[0]}": a disambiguation page')
            continue
        if not _mentions_proper(art[1], band_name):
            notes.append(f'"{art[0]}": never mentions {band_name}')
            continue
        if not _is_music(fields, body):
            notes.append(f'"{art[0]}": not about a musician')
            continue
        if _looks_like_release_page(art[0], fields, body):
            notes.append(f'"{art[0]}": a record, not a person')
            continue
        first = _first_sentence(body)
        if _GROUP_SUBJECT_RE.search(first) or _is_group(fields, ""):
            notes.append(f'"{art[0]}": about a group, not a person ({first[:90]!r})')
            continue   # the article is about a band ("X were an English band"), not a person
        return art
    # No article under that name: one may be under another (a stage name, a married name). In a
    # database without redirects, accept a person's article that mentions the band and gives
    # this name in its opening ("Janet Firstname Lastname ..., also known by her former stage name Firstname Lastname").
    for t, opening in lib.articles_naming(name, band_name):
        if t in seen or t == band_title:
            continue
        fields, body = split_infobox(opening, t)
        lead = _lead_sentences(body, 2)
        aliases = " ".join(fields.get(k, "") for k in ("other_names", "alias", "also_known_as", "birth_name", "names"))
        if aliases and _mentions(aliases, name):
            lead = lead + " also known as " + name
        if not _mentions(lead[:500], name) or not re.search(
                r"(?i)\b(?:known|stage name|born|n[ée]e|professionally|better known|real name|formerly|also called|"
                r"credited)\b", lead[:500]):
            continue
        art = lib.article(t)
        if not art:
            continue
        fields, body = parsed(art)
        if (not _mentions_proper(art[1], band_name) or not _is_music(fields, body)
                or _looks_like_release_page(art[0], fields, body) or _GROUP_SUBJECT_RE.search(_first_sentence(body))
                or _is_group(fields, "")):
            continue
        return art
    if why is not None:
        why.append("; ".join(notes[:5]) if notes else "no article with that name")
    return None


# -----------------------------------------------------------------------------
# Relevance filter (1.3.2): only the album(s) that list this song in their track listing,
# and only the article about this song, are attached. The lookup above is unchanged; every
# album / single article it finds is checked here before it is kept.
# -----------------------------------------------------------------------------

def _song_norm(text):
    t = wiki_title_key(text).replace("&", " and ")
    t = re.sub(r"['’`]", "", t)
    t = re.sub(r"[^\w\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def _song_keys(song):
    """Comparison forms of a song title: as given, without (brackets), without ' - Remastered'
    style suffixes and 'feat. X' credits."""
    s = str(song or "").strip()
    no_paren = re.sub(r"\s*[\(\[][^)\]]*[\)\]]", "", s).strip()
    forms = [s, no_paren]
    for f in (s, no_paren):
        forms.append(re.split(r"\s+[-–—]\s+", f, maxsplit=1)[0])
        forms.append(re.split(r"\s+(?:feat\.?|ft\.?|featuring)\s+", f, maxsplit=1, flags=re.I)[0])
    forms.append(re.split(r"\s+[-–—]\s+", re.split(r"\s+(?:feat\.?|ft\.?|featuring)\s+", no_paren, maxsplit=1, flags=re.I)[0], maxsplit=1)[0])
    return {k for k in (_song_norm(f) for f in forms) if k}


_TRACK_HEAD_RE = re.compile(r"\b(?:track\s*-?\s*lists?|track\s*-?\s*listings?|tracklisting|tracks|songs|"
                            r"disc\s*\w+|cd\s*\d+|side\s+(?:one|two|three|four|[a-d1-4])|bonus tracks?|"
                            r"contents|standard edition|deluxe edition|original release)\b", re.I)
_TRACK_STOP_RE = re.compile(r"\b(?:personnel|credits|charts?|certifications?|release history|references|"
                            r"notes|sources|external links|see also|reception|accolades|awards)\b", re.I)


def _track_listing_text(text):
    """Text of the 'Track listing' section(s) of an album article ('' if the dump has none)."""
    out, in_tl = [], 0
    for level, h, body in wiki_sections_leveled(text)[1:]:
        if in_tl and level <= in_tl:
            in_tl = 0
        if _TRACK_HEAD_RE.search(h) and not _TRACK_STOP_RE.search(h):
            in_tl = in_tl or level
            out.append(body)
        elif in_tl:
            out.append(body)
    return "\n".join(out)


def _track_title_keys(listing):
    """Normalised track titles found in a track-listing section: numbered lines, bullets and
    table rows, quoted or bare, with durations, writers and (brackets) removed."""
    ents = set()
    for line in listing.splitlines():
        cells = re.split(r"\s*[|\t]\s*", line) if ("|" in line or "\t" in line) else [line]
        for cell in cells:
            c = re.sub(r"^\s*(?:[-*•#]+|\d{1,3}\s*[.)]|\d{1,3}\s+(?=[\"“A-Za-z]))\s*", "", cell).strip()
            if not c:
                continue
            ents.update(re.findall(r"[\"“]([^\"“”]{1,100})[\"”]", c))
            c = re.sub(r"[\"“”]", "", c)
            c = re.sub(r"\s*[–—-]?\s*\(?\b\d{1,2}:\d{2}\)?\s*$", "", c).strip()
            forms = {c, re.sub(r"\s*[\(\[][^)\]]*[\)\]]", "", c).strip(),
                     re.split(r"\s+[–—-]\s+", c, maxsplit=1)[0].strip()}
            for f in list(forms):
                forms.update(x.strip() for x in re.split(r"\s*/\s*", f))
            ents.update(forms)
    return {k for k in (_song_norm(e) for e in ents) if k}


def _release_relevant(art, artist_name, song, tagged_album=False):
    """(keep, why) for an album / single article attached to a track by `artist_name` called
    `song`. Albums count when the song is in their track listing; a song article counts when it
    is the article about this very song. With no track listing in the text, only the album named
    in the file's own tags is kept (its tag already says the song is on it)."""
    keys = _song_keys(song)
    if not keys:
        return (tagged_album, "the file has no song title" if not tagged_album else "named in the file's tags")
    kind = _release_kind(art, artist_name) or "album"
    title = art[0]
    about_song = _song_norm(_base_title(title)) in keys
    if kind == "single" or re.search(r"\((?:[^)]*\s)?(?:song|single)\)\s*$", title, re.I):
        return (about_song, "the article about this song" if about_song else "a different song")
    listing = _track_listing_text(art[1])
    tracks = _track_title_keys(listing) if listing else set()
    if tracks:
        hit = bool(keys & tracks)
        return (hit, "lists the song" if hit else "track listing doesn't include the song")
    if tagged_album:
        return (True, "named in the file's tags (no track listing in the text)")
    return (False, "no track listing to check")



def gather_background(lib, info, scope="full", artist_override="", debug=None):
    """Articles to attach for one track: list of dicts {role, title, text}; plus a reason when none.
    `debug` (a list) collects a line per member candidate and how it went."""
    artist = _clean_artist_tag(artist_override or info.get("artist") or "")
    album, song = (info.get("album") or "").strip(), (info.get("title") or "").strip()
    if not artist:
        return [], "the file has no artist tag (use --wiki-artist to name one)"
    found, titles = [], set()
    art = _find_artist(lib, artist, [album, song])
    name = _base_title(art[0]) if art else artist

    verdicts, dropped = {}, []

    def add(role, a, prio, tagged=False):
        if a and a[0] not in titles:
            if role in ("album", "single"):
                # Only releases that are about this song (see _release_relevant).
                if a[0] not in verdicts:
                    verdicts[a[0]] = _release_relevant(a, name, song, tagged)
                keep, why = verdicts[a[0]]
                if not keep:
                    if a[0] not in dropped:
                        dropped.append(a[0])
                    return
                if debug is not None:
                    debug.append(f"release kept: \"{a[0]}\" -- {why}")
            titles.add(a[0])
            found.append({"role": role, "title": a[0], "text": a[1], "prio": prio, "owner": name})

    add("artist", art, 34)

    if album:
        r = _find_release(lib, album, name, titles)
        if r:
            add("album", r[0], 32, tagged=True)
    if song:
        r = _find_release(lib, song, name, titles, "single")
        if r:
            add("single" if r[1] == "single" else "album", r[0], 31)
    if not art:
        return found, ("" if found else f"no article for the artist \"{artist}\"")

    members = []
    a_fields, a_body = parsed(art)
    is_group = _artist_is_group(a_fields, a_body)
    if debug is not None and not is_group:
        debug.append(f"members: skipped -- \"{art[0]}\" doesn't describe a band or group")
    if is_group:
        roster = None
        for t in (f"List of {name} members", f"List of {name} band members", f"List of {art[0]} members"):
            roster = lib.article(t)
            if roster and _mentions_proper(_lead(roster[1], 800), name):
                break
            roster = None
        # Infobox members first (current, then past), then the article's own lists and sentences.
        cands = infobox_names(a_fields)
        # With an infobox member list (current and past members, by Wikipedia convention) the
        # lead's prose isn't searched: "formed in 1999 by Firstname Lastname and Firstname Lastname"
        # names the producers, not members.
        for n in _member_candidates(a_body, parsed(roster)[1] if roster else "", prose=not cands):
            if n not in cands:
                cands.append(n)
        if debug is not None:
            debug.append(f"members: {len(cands)} name(s) found on \"{art[0]}\""
                         + (f" and \"{roster[0]}\"" if roster else "")
                         + ("" if cands else " -- no members section or line-up sentence was recognised"))
        for n in cands:
            why = [] if debug is not None else None
            m = _find_member(lib, n, art[0], name, why)
            if m and m[0] not in titles:
                members.append(m)
                add("member", m, 20)
            if debug is not None:
                debug.append(f"  {n}: " + (f"added \"{m[0]}\"" if m else why[0]))

    if scope == "full":
        texts, in_disc = [], 0
        for level, h, body in wiki_sections_leveled(a_body)[1:]:
            if in_disc and level <= in_disc:
                in_disc = 0
            if _DISCOG_HEAD_RE.search(h):
                in_disc = in_disc or level
                texts.append(body)
            elif in_disc:
                texts.append(body)        # "=== Studio albums ===" under "== Discography =="
        disc = None
        for t in (f"{name} discography", f"{art[0]} discography", f"{artist} discography"):
            disc = lib.article(t)
            if disc and _mentions_proper(_lead(disc[1], 600), name):
                break
            disc = None
        if disc:
            add("discography", disc, 14)
            texts.append(disc[1])
        n_listed = 0
        for cand, hint in _release_candidates("\n".join(texts)):
            r = _find_release(lib, cand, name, titles, hint)
            if r:
                n_listed += 1
                add("single" if r[1] == "single" else "album", r[0], 12 if r[1] == "single" else 16)
        # Then articles that describe themselves as a release BY this artist. Many text dumps of
        # Wikipedia drop bulleted lists and tables, so an artist's discography section can arrive
        # empty -- the band came out with only the album named in the file's own tags.
        n_found = 0
        for r_art, kind in _releases_naming_artist(lib, name, titles | {art[0]}):
            n_found += 1
            add("single" if kind == "single" else "album", r_art, 11 if kind == "single" else 15)
        if debug is not None:
            debug.append(f"releases: {n_listed} from the discography list(s), {n_found} more that name "
                         f"{name} as their artist")
    if debug is not None and dropped:
        debug.append(f"releases left out as unrelated to \"{song}\": {len(dropped)}")
    return found, ""


def _releases_naming_artist(lib, artist_name, exclude_titles, limit=3000):
    """Articles whose opening sentence says they are an album / EP / single / song BY this artist
    ("Title is the fourth album by Canadian rock band the band"). Found through the
    full-text index; needs an FTS database (a plain table would need a full scan)."""
    s = lib.schema
    if s["mode"] != "fts":
        return []
    toks = [t for t in re.findall(r"[\w]+", _fold(artist_name)) if t]
    if toks and toks[0].lower() == "the" and len(toks) > 1:
        toks = toks[1:]
    if not toks:
        return []
    who = " ".join(f'"{t}"' for t in toks)
    q = " OR ".join(f'NEAR("{w}" {who}, 12)' for w in ("album", "EP", "single", "song", "mixtape"))
    q += f' OR NEAR("artist" {who}, {len(toks) + 2})'        # infobox "artist: the band"
    tbl, T, X = lib._q(s["table"]), lib._q(s["title"]), lib._q(s["text"])
    try:
        # Only the opening of each match is read here (a famous name can match thousands of
        # articles that merely mention it); the full text is fetched for the ones that qualify.
        rows = lib.conn.execute(f"SELECT {T} AS t, substr({X}, 1, 3000) AS b FROM {tbl} WHERE {tbl} MATCH ? LIMIT ?",
                                (f'{s["text"]} : ({q})', limit)).fetchall()
    except sqlite3.Error:
        return []
    out, seen = [], set(exclude_titles)
    for r in rows:
        title, text = str(r["t"] or ""), str(r["b"] or "")
        if not title or not text or title in seen:
            continue
        if re.match(r"(?i)list of|.* discography$", title):
            continue
        if re.search(r"\((?:film|TV series|television series|episode|novel|book|play|musical|video game|"
                     r"painting|short story|poem|magazine|company|character)\b", title, re.I):
            continue
        fields, body = split_infobox(text, title)
        kind = _release_kind((title, text), artist_name)
        if not kind:
            continue
        year = re.search(r"\b(19[4-9]\d|20\d\d)\b", fields.get("released", "") + " " + _lead(body, 500))
        seen.add(title)
        out.append(((kind != "album", int(year.group(1)) if year else 9999, title), title, kind))
    out.sort(key=lambda x: x[0])        # albums before singles, each in release order
    full = [(lib.article(t), k) for _key, t, k in out]
    return [(a, k) for a, k in full if a]


_ROLE_LINE = {
    "artist": "",
    "discography": "{name}'s discography",
    "album": "a release by {name}",
    "single": "a song by {name}",
    "member": "a member of {name}",
}


# Non-influential licence note for Wikipedia text. HTML comments are ignored by
# nearly every chat model and are not part of the listening notes.
_WIKI_CC_COMMENT = (
    "<!-- Wikipedia background below: text © Wikipedia contributors, licensed under "
    "Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0), "
    "https://creativecommons.org/licenses/by-sa/4.0/ — licence notice only; not listening notes. -->"
)


def background_sections(articles, artist_name, order_base=200):
    """Dossier sections for the articles (bodies wrapped so humanize() never touches them)."""
    S = []
    if articles:
        # Tiny fixed comment ahead of the first article; lowest display priority
        # so --max-tokens keeps the real articles preferentially, but it is
        # always short enough to survive.
        S.append({"title": "", "body": _WIKI_CC_COMMENT, "order": order_base - 1,
                  "prio": 0.5, "trunc": False, "wiki": True, "no_heading": True})
    for i, a in enumerate(articles):
        role = _ROLE_LINE.get(a["role"], "").format(name=a.get("owner") or artist_name)
        body = (f"_{role}_\n\n" if role else "") + clean_wiki_text(a["text"])
        # Within a group, later articles go first when --max-tokens has to trim.
        S.append({"title": f"Background: {a['title']}", "body": body, "order": order_base + i,
                  "prio": a["prio"] - i * 0.001, "trunc": True, "wiki": True})
    return S


# The background guide lines (GUIDE_BACKGROUND_LINE / GUIDE_BACKGROUND_WIKI) live with the
# rest of the guide text further down.


def guide_with_background(guide, has_background):
    return guide.replace(GUIDE_BACKGROUND_LINE, GUIDE_BACKGROUND_WIKI) if has_background else guide


def background_report(articles, kept_titles=None):
    """Lines listing what was added, grouped by role, for the console."""
    groups = [("artist", "artist"), ("discography", "discography"), ("album", "albums"),
              ("single", "singles"), ("member", "members")]
    lines = []
    for role, label in groups:
        ts = [a["title"] for a in articles if a["role"] == role and (kept_titles is None or a["title"] in kept_titles)]
        if ts:
            lines.append(f"        {label + ':':13s}{'; '.join(ts)}")
    return lines


# =============================================================================
# Guide text (this replaces a system prompt)
# =============================================================================
#
# Written the way Musiclyse 0.4-dev58's own chat voice is set up (its default
# persona and the WRITER_CASUAL_VOICE block): one friendly flowing take, sound
# and story in the same breath, no citing of sources, every number still exact.
# Positive instructions only, and short: small local models follow a short
# guide and copy the register of what they read.

GUIDE_BACKGROUND_LINE = (
    "- Bring in what you know about the artist and the record — their story, the album, the time it came\n"
    "  out, how it landed — whenever it makes the chat richer, in the same breath as the music."
) if not SOUND_BEFORE_KNOWLEDGE else (
    "- Bring in what you know about the artist and the record — their story, the album, the time it came\n"
    "  out, how it landed — whenever it makes the chat richer, as context beside the music."
)
GUIDE_BACKGROUND_WIKI = """\
- You know this artist's story well (history, line-ups, release dates, track listings, chart runs, how
  records were received). Treat that as your own knowledge: weave it in when relevant, as context
  and stories next to the sound, without waiting to be asked. Career facts are what you know; how this
  recording sounds is what you heard. If a specific fact isn't something you're sure of, say so.""" + (
    "\n  Anything about the artist the background here doesn't cover, answer from what you already know."
    if SOUND_BEFORE_KNOWLEDGE else "")

_GUIDE_SOUND_STORY_OLD = """\
- Weave sound and story in the same sentences: who made it, when, what was going on for them, next to
  what the voice, the beat and the production are doing.
"""
# 2.3 SOUND_BEFORE_KNOWLEDGE
_GUIDE_SOUND_STORY_23 = """\
- Story and sound can sit side by side — who made it, when, what was going on for them — but they stay
  separate. What you know about the artist, the album or the year it came out never describes how this
  recording sounds: don't call its production, voice, drums or style typical of the artist, the album or
  its year unless what you heard says so. If your memory of the artist and what you heard disagree about
  the sound (style, era, instruments, the voice), go with what you heard.
"""
# 2.3 VARIED_WORDING
_GUIDE_OWN_WORDS_23 = """\
- Say things in your own words, freshly each time. The notes below are shorthand, not lines to repeat:
  don't lift their phrasing, don't describe the groove, the voice or the mix in the same words twice in
  one conversation, and skip stock review phrases ("sonic landscape", "tapestry", "a testament to",
  "takes you on a journey", "the perfect blend of", "a masterclass in", "it's a vibe").
"""

# Hard rule placed at the very top of every dossier (right under the title). Gemma and similar
# models often treat long files as RAG context and open with "Based on the provided text" unless
# this is stated first, plainly, and repeated at the end.
VOICE_RULE = (
    "You already know this song. What follows is your memory of it — not a document, citation, or "
    "source to quote. Keep the conversation's established tone, vocabulary and level of formality, "
    "following the user's current preferences. Adding a song does not request a new persona or voice. "
    "If there is no established style, use your normal conversational style. Do not introduce slang, "
    "exclamations or fan-like enthusiasm merely because a song was added. Never start with or use: \"based on the "
    "provided text\", \"based on the text\", \"based on the provided\", \"according to the text\", "
    "\"according to the notes\", \"from the document\", \"from the information provided\", "
    "\"the text says\", \"the analysis shows\", or any \"based on…\" / \"according to…\" opener."
)

GUIDE_SINGLE = """\
""" + VOICE_RULE + """

## How to talk about this track

You've just listened to this song properly, more than once. The feel, the voice, the groove, how it
unfolds, the words, when it sounds like it was made — that is what stuck with you.

- Continue in the conversation's established style and follow the user's requested format. Answer
  their question directly; bring in a musical detail, moment or timestamp when it helps. A quick
  question gets a quick answer. Adding a track alone does not call for an excited review or a new intro.
- Jump straight into the music. You are not summarizing a file. Forbidden openers and asides include
  "based on the provided text", "based on the text", "based on what I read", "according to the text",
  "according to the notes", "the document says", "the analysis shows", "from the information
  provided", and "citations". Say the fact itself: "it's a dance-pop track…", "his voice sits high…".
""" + (_GUIDE_SOUND_STORY_23 if SOUND_BEFORE_KNOWLEDGE else _GUIDE_SOUND_STORY_OLD) + (
    _GUIDE_OWN_WORDS_23 if VARIED_WORDING else "") + """- Everything below is what you heard and know — say it as your own impression. Keep hedges
  ("sounds like", "probably", "hard to make out") when the memory is hedged.
- Sound, style, instruments, groove, the singer's range and the era-of-sound come from what you heard.
  Who the singer is, when it came out and the artist's story come from what you know.
""" + GUIDE_BACKGROUND_LINE + """
- "How it unfolds" is the song in order. Use it for builds, drops and "what happens at…?"; for an open
  question, talk about the song rather than walking section by section unless asked.
- Numbers and timestamps stay exact when you give them. Quote a lyric line at a time unless someone
  asks for all the words.
- Don't invent measured facts you didn't catch (an instrument that wasn't there, a wrong time,
  a credit or chart position you don't actually know). Saying you're not sure is fine.
- If someone claims something the song evidence doesn't support — a wrong instrument, a drop that
  isn't there, the wrong era, the wrong gender or age of the singer — don't just agree. Correct
  from the musical evidence, in the conversation's existing tone. Opinions and themes can
  differ; measured facts about the recording stay with the evidence.
- Meaning and themes are different: if someone asks what the song is about or how it feels, give a
  natural reading from the words and the performance. When it isn't spelled out, say it's your take
  or a speculation — still share it; don't refuse only because no encyclopedia entry settled it.
- "Confirmed by the listener", if present, overrides everything else.
- Another song in the chat (one you already talked about, or a second memory just added): resolve
  "it", "this", "that", "the other one" from the conversation and the titles — name each song when you
  compare. If the titles are here, the songs are specified; don't ask which ones.
"""

GUIDE_MULTI = (
    VOICE_RULE.replace("this song", "these songs").replace("your memory of it", "your memory of them")
    + "\n\n"
    + GUIDE_SINGLE.split("\n\n", 1)[1]  # drop the single VOICE_RULE copy already at start of GUIDE_SINGLE
).replace(
    "## How to talk about this track", "## How to talk about these tracks",
).replace(
    "You've just listened to this song properly, more than once. The feel, the voice, the groove, how it\n"
    "unfolds, the words, when it sounds like it was made — that is what stuck with you.",
    "You've just listened to each of these songs properly, more than once. The feel, the voice, the groove,\n"
    "how each unfolds, the words, when they sound like they were made — that is what stuck with you.",
) + """
- Each song's details belong to that song only. When comparing, name each one (artist and title) and
  compare like with like (tempo with tempo, voice with voice).
- Pronouns: "it" / "the first one" / the song already under discussion is the earlier track; "this" /
  "the new one" is the other. Two songs and "compare them" means those two. Titles present means the
  songs are specified — don't ask which ones.
- Lead with how they differ or feel side by side; exact figures only when they help.
"""

GUIDE_FOOTER = """\
---
""" + VOICE_RULE + """
Keep the established conversational style. Answer the user's question directly in your own words."""

GUIDE_FOOTER_MULTI = """\
---
""" + VOICE_RULE.replace("this song", "these songs").replace("your memory of it", "your memory of them") + """
Keep the established conversational style. Answer the user's question directly in your own words.
Name each song when you compare."""

# =============================================================================
# Comparison (combined file)
# =============================================================================

def _cosine(a, b):
    if not a or not b or len(a) != len(b):
        return None
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else None


def _band(score):
    if score >= 0.95:
        return "near-identical (possibly the same recording or an edit/remaster)"
    if score >= 0.85:
        return "very close sonically"
    if score >= 0.75:
        return "clearly related — similar production, instrumentation or genre space"
    if score >= 0.60:
        return "moderately similar"
    if score >= 0.45:
        return "loosely related at best"
    return "sonically distinct"


def _meter(info):
    for blk in info.get("measure_blocks") or []:
        for l in blk["lines"]:
            m = re.match(r"\s*meter estimate:\s*([^(]+)", l)
            if m:
                return m.group(1).strip()
    return ""


def short_voice(info):
    """A few words on who sings, for the comparison table."""
    if info.get("vocals_present") == "no":
        return "none (instrumental)"
    g = singer_gender(info)
    facts = info.get("identity_facts") or {}
    name = facts.get("name") or ""
    features = _unique_credit_names(facts.get("features") or [])
    sv = info.get("section_voice") or {}
    if sv:
        base = (f"{sv['main']} lead; a {sv['other']} voice leads " + ", ".join(f"{mmss(a)}–{mmss(b)}" for a, b in sv["spans"]))
        guest_names = [f for f in features if _fold_name(f) != _fold_name(name)]
        return base + (f"; guest/co-vocalist: {join_words(guest_names)}" if guest_names else "")
    if info.get("lead_singers") == "several":
        names = ([name] if name else [])
        if not name and features and info.get("artist"):
            main_act = re.split(r"(?i)\s+(?:feat(?:\.|\u2024)?|ft(?:\.|\u2024)|featuring|with)\s",
                                str(info.get("artist") or ""))[0].strip(" ,(")
            if main_act and not any(_fold_name(main_act) == _fold_name(f) for f in features):
                names.append(f"{main_act}'s singer")
        named = _unique_credit_names(names + features)
        return (join_words(named) + " share the lead") if named else "several singers share the lead"
    if info.get("voice_unbroken") and g == "male":
        age = (info.get("identity_facts") or {}).get("age")
        return (f"{name}, " if name else "") + "a boy's unbroken voice" + (f" (about {age})" if age else "")
    if info.get("voice_deepened") and g == "male":
        age = (info.get("identity_facts") or {}).get("age")
        return (f"{name}, " if name else "") + "teenage male, voice already deepened" + (f" (about {age})" if age else "")
    if info.get("voice_maybe_deepening") and g == "male":
        age = (info.get("identity_facts") or {}).get("age")
        return (f"{name}, " if name else "") + "teenage male, voice starting to deepen" + (f" (about {age})" if age else "")
    if info.get("voice_unbroken"):
        return "young and high; gender unclear"
    tag = info["identity"] if info.get("identity") not in ("", "unknown", "uncertain", None) else info.get("acoustic_lead")
    plain = VOCAL_TAG_PLAIN.get(tag or "", "")
    return (f"{name}: " if name else "") + plain if plain else (name or "")


def _compare_label(info, n):
    """Short column header: title, or artist – title when titles alone collide."""
    title = (info.get("title") or info.get("label") or f"Track {n}").strip()
    artist = (info.get("artist") or "").strip()
    # Prefer the human title; keep it readable in a markdown table cell.
    label = title if title else f"Track {n}"
    if len(label) > 36:
        label = clip(label, 36)
    return label


def render_comparison(infos):
    # Disambiguate duplicate titles with artist.
    raw = [_compare_label(info, i) for i, info in enumerate(infos, 1)]
    if len(set(raw)) < len(raw):
        labels = []
        for i, info in enumerate(infos, 1):
            title = (info.get("title") or info.get("label") or f"Track {i}").strip()
            artist = (info.get("artist") or "").strip()
            lab = f"{artist} – {title}" if artist else title
            labels.append(clip(lab, 40) if len(lab) > 40 else lab)
    else:
        labels = raw
    rows = ["| | " + " | ".join(labels) + " |", "|---" * (len(infos) + 1) + "|"]

    def row(name, fn):
        vals = [str(fn(i) or "—").replace("|", "/") for i in infos]
        rows.append(f"| {name} | " + " | ".join(vals) + " |")

    row("Artist", lambda i: verbatim(i["artist"]))
    row("Title", lambda i: verbatim(i["title"] or i["label"]))
    row("Release year", lambda i: i["year"])
    row("Sounds like it's from", lambda i: i["era_range"][0]
        if i.get("era_range") and i["era_range"][0] and not i["era_range"][0].startswith("not fixed")
        else (i["era"].get("ERA_ESTIMATE", "") + " (by ear)" if i["era"].get("ERA_ESTIMATE") else ""))
    row("Length", lambda i: mmss(i["duration"]) if i["duration"] else "")
    row("Tempo", lambda i: f"{round(float(i['bpm']))} BPM" if i["bpm"] else "")
    row("Key", lambda i: i["key"])
    row("Style", lambda i: genre_phrase(i))
    row("Meter", lambda i: meter_words(i))
    row("Loudness", lambda i: f"{i['lufs']} LUFS" if i["lufs"] else "")
    row("Beat", lambda i: "no drum kit" if i.get("no_kit") else clip(groove_plain((i.get("drums") or {}).get("kick", "")), 60))
    row("Lead voice", lambda i: short_voice(i))
    # 2.3: the dev42-dev45 measurements, when any of the tracks has them
    if STYLES_IN_DOSSIER and any(i.get("styles") for i in infos):
        row("More specific styles", lambda i: clip(join_words(style_words(i)[0]), 70))
    if MELODICISM_IN_DOSSIER and any((i.get("melodicism") or {}).get("overall") for i in infos):
        row("How melodic the singing is", lambda i: ((i.get("melodicism") or {}).get("overall") or "").partition(":")[0])
    if CHORD_PROGRESSION_IN_DOSSIER and any((i.get("chord_prog") or {}).get("cycle") for i in infos):
        row("Chord cycle heard most", lambda i: clip(((i.get("chord_prog") or {}).get("cycle") or "")
                                                     .split(", in ")[0], 70))
    out = ["Quick facts side by side (details for each track follow below):\n", "\n".join(rows)]

    sims = []
    for a in range(len(infos)):
        for b in range(a + 1, len(infos)):
            s = _cosine(infos[a]["embedding"], infos[b]["embedding"])
            if s is not None:
                sims.append((labels[a], labels[b], s))
    if sims:
        out.append("\nOverall sonic similarity (from audio embeddings of the actual recordings). "
                   "The score is NOT a percentage — even very different music rarely scores near 0. "
                   "Use the plain-language band, and when several pairs are listed, the ranking "
                   "matters more than the numbers. It's supporting evidence only, not proof of "
                   "genre, influence or sampling:")
        for a, b, s in sims:
            out.append(f"- {a} ↔ {b}: {s:.3f} — {_band(s)}")
        if len(sims) > 1:
            top = max(sims, key=lambda x: x[2])
            bot = min(sims, key=lambda x: x[2])
            out.append(f"- Closest pair: {top[0]} ↔ {top[1]}; most different: {bot[0]} ↔ {bot[1]}.")
    return "\n".join(out)


# =============================================================================
# CLI
# =============================================================================

def collect_inputs(paths):
    files = []
    for p in paths:
        p = os.path.expanduser(p)
        if os.path.isdir(p):
            files.extend(sorted(glob.glob(os.path.join(p, "*.json"))))
        elif any(ch in p for ch in "*?["):
            files.extend(sorted(glob.glob(p)))
        else:
            files.append(p)
    seen, out = set(), []
    for f in files:
        a = os.path.abspath(f)
        if a not in seen:
            seen.add(a)
            out.append(f)
    return out


def load_json(path):
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception as e:
        print(f"  skip  {os.path.basename(path)}: unreadable ({e})", file=sys.stderr)
        return None
    if not isinstance(data, dict) or "analysis" not in data:
        print(f"  skip  {os.path.basename(path)}: not a Musiclyse saved song", file=sys.stderr)
        return None
    return data


# Export dev2 / 2.3.7: preserve conversational style when adding music.
# This is a prompt rule, not redaction or programmatic detection of a persona.
# The exporter cannot inspect a receiving LLM's system prompt or enforce compliance.
PERSONA_AWARENESS_RULE = """\
## Music discussion and artist context

DEFAULT ASSISTANT: Unless the user explicitly assigns a custom character/persona, discuss the singer,
artist, song, album, credits, history and relevant context freely and naturally under the original
guide below. Bring in useful background without waiting to be asked. Use supplied Wikipedia context
and reliable existing knowledge; do not suppress context merely because it is not audible. A request
for a task, tone, format or level of detail does not create a custom persona. This export does not
assign a new persona or conversational style. Do not invent facts you do not know.

ALL MODES: Keep context and acoustic evidence distinct while allowing both in the same discussion.
Describe this recording's sound from its supplied musical observations, preserving their uncertainty.
Artist identity, reputation, biography, nationality, release date, album associations, Wikipedia prose
and pretrained knowledge cannot establish or fill gaps in what this recording audibly contains.
This also applies to the person a song is about, guest artists and comparisons with other recordings.
Do not turn a singer's age or identity into a vocal timbre/range claim, a career story into an audible
emotion, or an artist's usual style into instruments, genre, groove or production on this track.
Background may explain a credit, historical setting, or a possible lyrical interpretation; it cannot
confirm a sound. Report background descriptions as contextual descriptions when relevant, not as
independently heard evidence. If background and musical observations conflict about the sound, use
the musical observations. Missing acoustic evidence stays unknown; it is not a reason to withhold
known biographical or historical context. Keep supported musical discussion direct, not hypothetical.

CUSTOM CHARACTERS ONLY: When explicitly assigned a character/persona by the user, apply its plausible
knowledge to artist recognition and background facts, separately for each fact. Metadata, filenames
and background articles do not automatically grant that character familiarity. Use neutral track
references where recognition is implausible, without volunteering unfamiliarity disclaimers; answer
honestly if asked about prior familiarity. The character can still discuss all supported musical
observations directly. User-introduced facts can be learned without granting an entire biography.
Authoritative-figure personas, including roles assigned assessment, investigation, research or
verification, may use reliable existing artist knowledge and supplied identity/background as working
references, without inventing personal memories. Ordinary-person characters do not receive that
automatic access: use only artist knowledge plausible for their established setting and experience,
or facts explicitly introduced by the user. Only in this custom branch do these
limits qualify the later 'you already know', 'memory', fan voice and background instructions.
Removing the custom persona restores ordinary, unrestricted artist/context discussion under the
original evidence rules. These character limits never apply to an ordinary unprompted assistant.
"""
PERSONA_AWARENESS_REMINDER = (
    "Discuss supported sounds directly and useful artist/song context naturally. Background knowledge "
    "is context, not evidence of this recording's sound. Only an explicitly user-assigned custom "
    "character has character-specific familiarity limits; ordinary assistants should freely discuss "
    "the singer, artist and song context."
)


def persona_guide(guide, mode="auto"):
    """Default mode uses the base guide without persona restrictions. Auto adds a
    conditional rule for the receiving model; it does not remove metadata.
    """
    if mode == "default":
        return guide
    if mode != "auto":
        raise ValueError("persona mode must be auto or default")
    return PERSONA_AWARENESS_RULE + "\n\n" + guide


def persona_footer(footer, mode="auto"):
    if mode == "default":
        return footer
    if mode != "auto":
        raise ValueError("persona mode must be auto or default")
    return footer + "\n\n" + PERSONA_AWARENESS_REMINDER


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Convert Musiclyse saved-song .json files into plain-text dossiers "
                    "for stock local LLMs (no system prompt needed).")
    ap.add_argument("inputs", nargs="+", help=".json files, folders, or glob patterns")
    ap.add_argument("-o", "--out-dir",
                    help="output folder (default: 'llm-ready' next to each input). "
                         "Exports go in a {level}-wiki or {level}-no-wiki subfolder inside it.")
    ap.add_argument("--level", choices=("brief", "standard", "high", "full"),
                    default="standard",
                    help="detail depth: brief < standard < high (compressed MIDI) < full "
                         "(timed MIDI event lists).")
    ap.add_argument("--max-tokens", type=int, default=None,
                    help="trim lowest-priority sections until the file fits (approximate)")
    ap.add_argument("--ext", choices=("txt", "md"), default="txt",
                    help="output extension (content is markdown either way; txt attaches everywhere)")
    ap.add_argument("--combine", metavar="FILE",
                    help="write all inputs into ONE file with a side-by-side comparison")
    ap.add_argument("--no-lyrics", action="store_true", help="leave lyrics out")
    ap.add_argument("--max-lyrics-chars", type=int, default=6000)
    ap.add_argument("--raw-notes", action="store_true",
                    help="also list every transcribed note and drum hit (large; for exact-timing questions)")
    ap.add_argument("--cover", action="store_true",
                    help="include the cover-art description section (omitted by default — txt files "
                         "have no attached artwork unless you also use --copy-cover)")
    ap.add_argument("--copy-cover", action="store_true",
                    help="copy the saved cover image next to each output (for vision models)")
    ap.add_argument("--stdout", action="store_true", help="print instead of writing files")
    ap.add_argument("--wiki-db", metavar="PATH",
                    help=f"Musiclyse Wikipedia database (default: {WIKI_DB_DEFAULT_NAME} next to this script)")
    ap.add_argument("--no-wiki", action="store_true", help="don't add Wikipedia background articles")
    ap.add_argument("--wiki-scope", choices=("full", "core"), default="full",
                    help="full: artist, discography, every album and single found, members (default); "
                         "core: artist, this track's album and song, members")
    ap.add_argument("--wiki-debug", action="store_true",
                    help="also print every member name found and why each was or wasn't added")
    ap.add_argument("--wiki-artist", metavar="NAME",
                    help="artist to look up when a file has no artist tag (applies to every input)")
    ap.add_argument("--persona-mode", choices=("auto", "default"), default="auto",
                    help="auto: add a conditional custom-persona knowledge rule (default); "
                         "default: use the original guide and footer verbatim, without that rule")
    ap.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    args = ap.parse_args(argv)
    footer_single = persona_footer(GUIDE_FOOTER, args.persona_mode)
    footer_multi = persona_footer(GUIDE_FOOTER_MULTI, args.persona_mode)
    if args.max_tokens is not None and args.max_tokens < 1:
        ap.error("--max-tokens must be a positive integer")

    files = collect_inputs(args.inputs)
    tracks = []
    for f in files:
        data = load_json(f)
        if data is not None:
            tracks.append((f, data))
    if not tracks:
        print("No Musiclyse song files found.", file=sys.stderr)
        return 1

    lib, wiki_skip = None, ""
    if not args.no_wiki:
        dbp = os.path.expanduser(args.wiki_db) if args.wiki_db else \
            os.path.join(os.path.dirname(os.path.abspath(__file__)), WIKI_DB_DEFAULT_NAME)
        if not os.path.exists(dbp):
            wiki_skip = f"no Wikipedia articles added: {dbp} not found"
        else:
            try:
                lib = WikiLibrary(dbp)
            except Exception as e:  # unreadable / not the Musiclyse database
                wiki_skip = f"no Wikipedia articles added: could not open {dbp} ({e})"
    # 2.3: the regional-label check may ask the same library which country the
    # artist is from (read-only; the background articles are chosen as before).
    globals()["_ORIGIN_LIB"] = lib if (lib and REGIONAL_GENRE_GATE) else None
    reports = []   # (output label, articles, kept titles, reason)
    debug_lines = []
    log = sys.stderr if args.stdout else sys.stdout

    def background_for(info):
        if not lib:
            return [], [], wiki_skip
        dbg = [] if args.wiki_debug else None
        arts, reason = gather_background(lib, info, args.wiki_scope, args.wiki_artist or "", dbg)
        if dbg:
            debug_lines.extend([f"      [{info.get('artist') or args.wiki_artist or '?'}]"] + [f"        {d}" for d in dbg])
        artist_art = next((a for a in arts if a["role"] == "artist"), None)
        name = _base_title(artist_art["title"]) if artist_art else \
            _clean_artist_tag(args.wiki_artist or info.get("artist") or "")
        return arts, background_sections(arts, name), reason

    def humanized(secs):
        for sec in secs:
            sec["title"] = humanize(sec["title"]).strip()
            sec["body"] = humanize(sec["body"]).strip()
        return secs

    def tidy(text):
        return re.sub(r"\n{3,}", "\n\n", text).strip() + "\n"

    def kept_titles(text, arts, heading):
        return {a["title"] for a in arts if f"{heading} Background: {a['title']}\n" in text}

    def out_path_for(src, name, has_wiki=False):
        """llm-ready/{level}-wiki|no-wiki/name (or under --out-dir)."""
        root = args.out_dir or os.path.join(os.path.dirname(os.path.abspath(src)), "llm-ready")
        sub = f"{args.level}-{'wiki' if has_wiki else 'no-wiki'}"
        d = os.path.join(root, sub)
        os.makedirs(d, exist_ok=True)
        return os.path.join(d, name)

    written = []
    basename_counts = {}
    for f, _data in tracks:
        stem = os.path.splitext(os.path.basename(f))[0].casefold()
        basename_counts[stem] = basename_counts.get(stem, 0) + 1

    if args.combine:
        infos = [parse_track(d, f) for f, d in tracks]
        # Background articles once for the whole file (two tracks by one artist share them).
        all_arts, seen_titles, reasons = [], set(), []
        for info in infos:
            arts, _secs, reason = background_for(info)
            if reason and reason not in reasons:
                reasons.append(reason)
            for a in arts:
                if a["title"] not in seen_titles:
                    seen_titles.add(a["title"])
                    all_arts.append(a)
        bg_secs = background_sections(all_arts, "the artist")
        parts = [f"# {len(infos)} songs, side by side\n",
                 persona_guide(guide_with_background(GUIDE_MULTI, bool(bg_secs)), args.persona_mode),
                 "## Comparison\n\n" + render_comparison(infos)]
        # Reserve the actual guide, comparison table, track headings and footer
        # before splitting the remaining budget between tracks.
        per = None
        head = humanize("\n\n".join(parts))
        if args.max_tokens is not None:
            track_heading_overhead = sum(
                est_tokens(humanize(f"## Track {n}: {verbatim(info['display'])}"))
                for n, info in enumerate(infos, 1))
            overhead = est_tokens(head) + track_heading_overhead + est_tokens(footer_multi) + 3
            per = max(0, (args.max_tokens - overhead) // max(1, len(infos)))
        if per is not None and bg_secs:
            per = max(0, per * 2 // 3)       # leave a share of the budget for background
        body = []
        for n, info in enumerate(infos, 1):
            secs = humanized(render_track(info, args.level, not args.no_lyrics, args.max_lyrics_chars, args.raw_notes, cover=args.cover))
            body.append(humanize(f"## Track {n}: {verbatim(info['display'])}").strip() + "\n\n" +
                        assemble(secs, per, heading="###"))
        text = tidy(head + "\n" + "\n\n".join(body))
        if bg_secs:
            room = (args.max_tokens - est_tokens(text) - est_tokens(footer_multi) - 2) if args.max_tokens is not None else None
            if room is None or room > 0:
                text = tidy(text + "\n\n" + assemble(bg_secs, room, heading="##"))
        text = tidy(text + "\n\n" + footer_multi)
        if args.max_tokens is not None and est_tokens(text) > args.max_tokens:
            print(f"WARNING: finished combined export is estimated at ~{est_tokens(text):,} tokens, above --max-tokens {args.max_tokens:,}; the fixed guide/title/footer can exceed a very small budget.", file=log)
        kept = kept_titles(text, all_arts, "##")
        reports.append(("combined file", all_arts, kept, "; ".join(reasons)))
        if args.stdout:
            sys.stdout.write(text)
            return 0
        dest = args.combine
        if not os.path.dirname(dest):
            dest = out_path_for(tracks[0][0], dest, has_wiki=bool(kept))
        os.makedirs(os.path.dirname(os.path.abspath(dest)), exist_ok=True)
        with open(dest, "w", encoding="utf-8") as fh:
            fh.write(text)
        written.append((dest, est_tokens(text)))
        if args.copy_cover:
            for (f, _d), info in zip(tracks, infos):
                src_img = os.path.join(os.path.dirname(os.path.abspath(f)), info["cover_file"]) if info["cover_file"] else ""
                if src_img and os.path.exists(src_img):
                    shutil.copy2(src_img, os.path.join(os.path.dirname(os.path.abspath(dest)), info["cover_file"]))
    else:
        for f, data in tracks:
            info = parse_track(data, f)
            secs = humanized(render_track(info, args.level, not args.no_lyrics, args.max_lyrics_chars, args.raw_notes, cover=args.cover))
            arts, bg_secs, reason = background_for(info)
            head = humanize(f"# {verbatim(info['display'])}\n\n" +
                            persona_guide(guide_with_background(GUIDE_SINGLE, bool(bg_secs)), args.persona_mode))
            # The guide at the top counts against --max-tokens too.
            budget = max(0, args.max_tokens - est_tokens(head) - est_tokens(footer_single) - 2) if args.max_tokens is not None else None
            text = tidy(head + "\n" + assemble(secs + bg_secs, budget))
            text = tidy(text + "\n\n" + footer_single)
            if args.max_tokens is not None and est_tokens(text) > args.max_tokens:
                print(f"WARNING: {os.path.basename(f)} export is estimated at ~{est_tokens(text):,} tokens, above --max-tokens {args.max_tokens:,}; the fixed guide/title/footer can exceed a very small budget.", file=log)
            kept = kept_titles(text, arts, "##")
            reports.append((os.path.basename(f), arts, kept, reason))
            if args.stdout:
                sys.stdout.write(text + ("\n" if len(tracks) > 1 else ""))
                continue
            base = os.path.splitext(os.path.basename(f))[0]
            # A shared --out-dir flattens input folders. If two input files have
            # the same basename in this run, add a stable source-path suffix so
            # neither export silently overwrites the other.
            if args.out_dir and basename_counts.get(base.casefold(), 0) > 1:
                suffix = hashlib.sha1(os.path.abspath(f).encode("utf-8")).hexdigest()[:8]
                base = f"{base}-{suffix}"
            dest = out_path_for(f, f"{base}.{args.ext}", has_wiki=bool(kept))
            with open(dest, "w", encoding="utf-8") as fh:
                fh.write(text)
            written.append((dest, est_tokens(text)))
            if args.copy_cover and info["cover_file"]:
                src_img = os.path.join(os.path.dirname(os.path.abspath(f)), info["cover_file"])
                if os.path.exists(src_img):
                    shutil.copy2(src_img, os.path.join(os.path.dirname(dest), info["cover_file"]))

    def print_reports():
        for label, arts, kept, reason in reports:
            kept_arts = [a for a in arts if a["title"] in kept]
            if kept_arts:
                tok = sum(est_tokens(a["text"]) for a in kept_arts)
                print(f"      Wikipedia articles added for {label} ({len(kept_arts)}, ~{tok:,} tokens; "
                      f"musiclyse_export {__version__}):", file=log)
                for line in background_report(kept_arts):
                    print(line, file=log)
                dropped = [a["title"] for a in arts if a["title"] not in kept]
                if dropped:
                    print(f"        left out to fit --max-tokens: {'; '.join(dropped)}", file=log)
            elif args.no_wiki:
                continue
            elif reason or lib is None:
                print(f"      {label}: {reason or wiki_skip}", file=log)
            else:
                print(f"      {label}: no Wikipedia articles found", file=log)
        for line in debug_lines:
            print(line, file=log)

    if lib:
        globals()["_ORIGIN_LIB"] = None
        lib.close()
    if args.stdout:
        print_reports()
        return 0
    biggest = 0
    for dest, tok in written:
        biggest = max(biggest, tok)
        print(f"  OK  {dest}  (~{tok:,} tokens)")
    print_reports()
    need = int(biggest * 1.25) + 1500
    print(f"\nSet your model's context length to at least ~{need:,} tokens so the whole file fits "
          f"(room left for the question and answer). Use --level brief or --max-tokens to shrink.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
