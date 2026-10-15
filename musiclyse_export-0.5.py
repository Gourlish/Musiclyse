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
    python musiclyse_export.py a.json b.json --combine compare.txt   # one file for comparisons
    python musiclyse_export.py a.json --max-tokens 6000              # fit a small context window
    python musiclyse_export.py a.json --stdout | ollama run gemma3 "What's the groove in the chorus?"
    python musiclyse_export.py untagged.json --wiki-artist "Radiohead"   # background for an untagged file

Compatibility
-------------
Built for saves from Musiclyse 0.4-dev76 and reads every earlier 0.4 save too
(see "What 2.1 changed" for what dev60-dev76 added, and the older notes below).

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
  identity reasoning (e.g. Christina Aguilera on a Maroon 5 track) is written
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
through the internal-term rewriting (a song called "Careless Whisper" came out
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
    full      adds every measurement as Musiclyse wrote it, each separated part's
              transcription summary, lyric timing and repeat map, file details.
              Add --raw-notes for every transcribed note.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import shutil
import sys
import unicodedata  # noqa: F401  (used by the Wikipedia background code)

__version__ = "2.1.6"

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
    words from the file itself. (Before 1.2, a song called "Careless Whisper" was exported as
    "Careless speech recognition", and an artist like MF DOOM as "the listening model DOOM".)"""
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
        r"(?m)^(?:GENRE CHECK|ARRANGEMENT CHECK|VOICE COUNT CHECK|INSTRUMENT CHECK)"
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
    if "uncertain" in n and "mf was uncertain" in n:
        return "moderately confident"
    if "agree" in n and "disagree" not in n:
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
    r"RECOMMENDED GENRE FOR DISCUSSION: ([^\n]+?)\. Reasoning: Music Flamingo GENRE_RANKED led "
    r"with '([^'\n]+)' \(rock/punk family\)")
_GENRE_SUPERSEDED_ALT_RE = re.compile(
    r"RECOMMENDED GENRE FOR DISCUSSION: ([^\n]+?)\. Reasoning: MF ranked '([^'\n]+)' first and '")
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


# dev56: the reverse replacement -- an electronic/pop first impression the
# genre check framed as rock (or post-punk / new wave).
_GENRE_ROCKFLIP_RE = re.compile(
    r"RECOMMENDED GENRE FOR DISCUSSION: ([^\n]+?)\. Reasoning: Music Flamingo's ranked list named only "
    r"electronic/pop labels \('([^'\n]+)' first\)")


def _ranked_labels(text, max_n=5):
    """Labels of the listening model's GENRE_RANKED list, in order."""
    f = parse_fields(main_pass_text(text) or text, ["GENRE_RANKED"]).get("GENRE_RANKED", "")
    labs = []
    for part in re.split(r"\s*;\s*|\s*\d\)\s*", f):
        part = part.strip(" .,;")
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
    m = re.search(r"(?m)^" + re.escape(_GENRE_CHECK_HEAD) + r"\s*(.+)$", A)
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
    m = re.search(r"(?m)^" + re.escape(_ARR_CHECK_HEAD) + r"\s*(.+)$", t)
    if m:
        return m.group(1).strip()
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
    for t in parse_tags(tag_string):
        if not _tag_label_suppressed(t["label"], suppressed):
            # Rebuild a compact item; exact original formatting is not required
            extent = f"; {t['extent']}" if t.get("extent") else ""
            support = "; supported by whole-mix" if t.get("supported") else ""
            kept.append(f"{t['label']} ({t['pct']}%{extent}{support})")
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
    m = re.search(r"(?m)^" + re.escape(_VOICE_CHECK_HEAD) + r"\s*(.+)$", t)
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
    for h in inp.get("mf_hints") or []:
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
    known = {"beats", "lyric_chunks", "lyric_groups", "mf_hints", "beatfeat", "total_duration", "instrumental"}
    out["other_inputs"] = {k: v for k, v in inp.items() if k not in known}
    return out


# Top-level fields of a save, as written by Musiclyse 0.4-dev53 save_song_data().
KNOWN_TOP_KEYS = {
    "version", "track_path", "label", "saved_at", "audio_file_signature", "analysis", "corrections",
    "metadata", "lyrics", "cover_observations", "singer_identity", "cover_art", "clap_embedding_model",
    "clap_embedding_dimension", "clap_embedding_aggregation", "clap_embedding", "stem_midi",
    "stem_midi_format", "stem_midi_complete", "structure_sections", "bpm", "key", "structure_replay",
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


def identity_facts(reason, year=""):
    """Plain facts the singer-identity step wrote in its reasoning: the
    singer's name when it says 'The credited solo artist is X', a birth year
    and an age at the time. Never used for how the recording sounds."""
    r = str(reason or "")
    out = {}
    m = re.search(
        r"\b(?:credited (?:solo )?artist|lead singer|lead vocalist|singer) is "
        r"([A-Z][\w.'’-]*(?: [A-Z][\w.'’-]*){0,3})",
        r,
    )
    if not m:
        # "Maroon 5's documented lead vocalist Adam Levine is a male adult"
        m = re.search(
            r"\b(?:documented |credited )?(?:lead )?(?:vocalist|singer)\s+"
            r"([A-Z][\w.'’-]*(?: [A-Z][\w.'’-]*){0,3})\s+is\b",
            r,
        )
    if m:
        out["name"] = m.group(1).strip()
    # Featured artist named in the identity note (e.g. "a Christina Aguilera feature is credited").
    feat = re.findall(
        r"\b(?:a |an )?([A-Z][\w.'’-]*(?: [A-Z][\w.'’-]*){0,3}) feature\b",
        r,
    )
    feat += re.findall(
        r"\b(?:feat(?:uring|\.)?|with|and)\s+([A-Z][\w.'’-]*(?: [A-Z][\w.'’-]*){0,3})\b"
        r"(?=\s+(?:is credited|feature|share|also sing))",
        r,
    )
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
    if m:
        out["born"] = int(m.group(1))
    m = re.search(r"~?\b(\d{1,2})\s*(?:years? old|yo\b)", r)
    if m:
        out["age"] = int(m.group(1))
    elif out.get("born") and re.match(r"^\d{4}$", str(year or "")):
        out["age"] = int(year) - out["born"]
    return out


def parse_track(data, json_path=""):
    A = strip_md(data.get("analysis") or "")
    meta = data.get("metadata") or {}
    info = {"json_path": json_path, "raw_analysis": A}

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
        if note not in sections[i]["name_check"]:
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
    info["bpm"] = data.get("bpm") or (float(m.group(1)) if m else None)
    info["tempo_conf"] = _tempo_confidence(m.group(2) if m else "")

    keys = re.findall(r"RECOMMENDED KEY FOR DISCUSSION:\s*(.+?)\.\s+Reasoning:\s*(.*?)\s*This is the primary key", A, re.S)
    info["key"] = data.get("key") or (keys[-1][0].strip() if keys else None)
    info["key_conf"] = _key_confidence(keys[-1][1] if keys else "")
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
    # Prose the listening model wrote beside its LABEL= fields (also dev36's
    # FREE-FORM DESCRIPTION pass) was dropped before; it is more of its description.
    # Drop pipeline metadata and the fixed-layout report scaffolding (dev64+).
    _DROP_EXTRA = {"STRUCTURE", "REPORT_NOTE", "CHECKS", "DESCRIPTION"}
    info["mf_extra"] = {k: v for k, v in parse_all_fields(main).items()
                        if k not in MF_FIELDS and k not in _DROP_EXTRA}
    # DESCRIPTION often carries a useful sentence or two, but some saves loop a
    # chord progression hundreds of times — keep a short clean version.
    desc = (info["mf"].get("DESCRIPTION") or "").strip()
    if desc and not desc.lower().startswith(("uncertain", "see ", "(none")):
        # Collapse a chord cycle repeated 3+ times into one cycle.
        desc = re.sub(
            r"((?:[A-G][#b]?m?(?:maj|min|sus|add|dim|aug)?\d*(?:/[A-G][#b]?)?(?:\s*[–\-–—]\s*)?){2,8})(?:\1){2,}",
            r"\1 (repeating)",
            desc,
        )
        # Hard cap so a runaway DESCRIPTION never floods the dossier.
        desc = clip(desc, 500)
        info["mf"]["DESCRIPTION"] = desc
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
    m = re.search(r"SINGER_IDENTITY\s*[=:]\s*([A-Za-z_]+)", si)
    info["identity"] = m.group(1).lower() if m else ""
    m = re.search(r"REASONING\s*[=:]\s*(.+)", si)
    info["identity_reason"] = m.group(1).strip() if m else ""
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
    info["identity_facts"] = identity_facts(info["identity_reason"], info.get("year"))

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
    info["voice_deepened"] = bool(
        _deepened_prose
        or (_young_male_id and _adult_male_ac and not info["voice_unbroken"])
    )
    # Explicit deepened note wins over unbroken if both somehow fire.
    if info["voice_deepened"] and info["voice_unbroken"] and _deepened_prose:
        info["voice_unbroken"] = False
    elif info["voice_unbroken"] and info["voice_deepened"] and not _deepened_prose:
        info["voice_deepened"] = False
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
            stem_ids[cur] = line.split(":", 1)[1].strip()
    # dev60+: drop accordion/harmonica tags the INSTRUMENT CHECK vetoed.
    suppressed = instrument_check_suppressed(A)
    info["instrument_suppressed"] = sorted(suppressed)
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
    info["midi_complete"] = bool(data.get("stem_midi_complete"))

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
    vc_on, vc_why = voice_count_check(A)
    info["voice_count_check"] = vc_why if (vc_on and info["vocals_present"] != "no") else ""
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
            lines = [head.split(":", 1)[1].strip()]
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
    return info


def _section(title, body, order, prio, truncatable=False):
    body = (body or "").strip()
    if not body:
        return None
    return {"title": title, "body": body, "order": order, "prio": prio, "trunc": truncatable}


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
            rec = s.split(":", 1)[1].strip()
        m = re.match(r"^Highest point of the \w+ line: (\S+) at (\d+:\d\d)", s)
        if m:
            high = (m.group(1), _secs(m.group(2)))
    return rec, high


# ---- sentences about the whole track ---------------------------------------------------
def genre_phrase(info):
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
    labs = [l for l in _ranked_labels(info.get("raw_analysis") or "") if l.lower() not in g]
    adj = info["mf"].get("GENRE_ADJACENT", "")
    for part in re.split(r"\s*[;,]\s*|\s*\d\)\s*", adj):
        p = part.strip(" .")
        if p and p.lower() not in g and p not in labs and p.lower() != "none":
            labs.append(p)
    return labs[:5]


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
                if l.lower().startswith(prefix):
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
        main_amb = re.split(r"\s*[—;(]\s*", amb)[0].strip()
        if main_amb:
            bits.append(f"{_a_an(main_amb)} sound" if not main_amb.endswith(("sound", "mix")) else _a_an(main_amb))
    width = _prod_line(info, "Stereo image", "stereo width reading")
    if width:
        w = re.split(r"\s*[—;(]\s*", width)[0].strip()
        w = {"narrow-to-moderate width": "fairly narrow", "moderate width": "moderately wide",
             "wide": "wide", "very wide": "very wide", "near-mono": "near-mono"}.get(w, w.replace(" width", ""))
        bits.append(f"{_a_an(w)} stereo image")
    sub = _prod_num(info, r"^\s*sub-bass \(20-60 Hz\): ([\d.]+)%")
    if sub is not None and sub >= 25:
        bits.append(f"a heavy low end (about {sub:.0f}% of the energy is deep sub-bass)")
    pump = _prod_line(info, "Sidechain / pumping", "sidechain reading")
    if pump and "ducks strongly" in pump and "recovers between beats" in pump:
        bits.append("clear sidechain pumping")
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
    if words:
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
        t = re.sub(r"\b(?:female|woman's|women's|woman|girl's|girl)\b", "high", t, flags=re.I)
        t = re.sub(r"\bhigh adult\b", "high", t)
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
    out = []
    for c in re.split(r"\s*;\s*", ev.rstrip(" .…")):
        lc = c.lower()
        if not c.strip() or c.endswith("…"):
            continue
        if re.search(r"sidechain|pumping|ducking", lc) and not measured_pump:
            continue
        if "trap" in lc and fotf:
            continue
        if re.search(r"hi-?hat rolls?", lc) and not hats16:
            continue
        if re.search(r"source separation|analogue workflows|analog workflows", lc):
            continue
        out.append(c.strip())
    return out


# ---- the walkthrough -----------------------------------------------------------------------
_KIT_PIECES = ("the hats", "the kick", "the snare", "the toms", "the cymbals", "the perc")
# Lyric-sheet headers worth naming in the walkthrough ("the pre-chorus"). Verse,
# intro, outro and bridge headers are left out: the same words often sit under
# different headers ("I need somebody" is both a post-chorus and the outro).
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
        conf = {t["label"] for t in confident_tags((info.get("stem_tags") or {}).get(m.group(2)))}
        if m.group(1) not in conf:
            continue
        where = re.sub(r"\s+only$", "", m.group(3))
        for nm in [w.strip() for w in where.split(",")]:
            tagmap.setdefault(nm, []).append(_tag_word(m.group(1)))
    onset = _secs(info.get("vocal_onset")) if info.get("vocal_onset") else None
    _rec, vhigh = melody_extras(info, "vocals")
    vocal_on = "vocals" in names

    out = []
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
                parts = [drum_config_plain(segs[0][2])] + [
                    f"{drum_config_plain(c)} from {mmss(x)}" for x, _y, c in segs[1:]]
                S.append("drums: " + parts[0] + ", then " + ", then ".join(parts[1:]))
                prev_drums = drum_config_plain(segs[-1][2])
        g = next((gg for starts, gg in groove_by if any(abs(st - a) <= 2 for st in starts)), "")
        if g and main_groove and g != main_groove:
            S.append(f"the beat here: {g}" if i == 0 else f"the groove changes here: {g}")

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

        for w in tagmap.get(name, []):
            S.append(f"something {w.rstrip('s')}-like comes through here" if w.endswith("strings")
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
                if mel.get("contour"):
                    mtxt += f" and {mel['contour']}"
                pb = prev_mel.get(base)
                hi = note_midi(mel["hi"])
                if pb is not None and hi is not None and hi - pb >= 2:
                    mtxt += f" — pitched higher than the last {base}"
                if hi is not None:
                    prev_mel[base] = hi
                S.append(mtxt)
            if vhigh and vhigh[1] is not None and a <= vhigh[1] < z:
                S.append(f"the vocal reaches its highest note ({pretty_note(vhigh[0])}) at {mmss(vhigh[1])}")
        if s.get("name_check"):
            S.append(f"(the name is a best guess here: {s['name_check']})")
        body = " ".join(sentence(x) for x in S if x)
        out.append(f"{head} ({int(round(z - a))} s) — {body}" if body else f"{head} ({int(round(z - a))} s)")
    return "\n\n".join(out)


# ---- who is singing ------------------------------------------------------------------------
def who_sings(info, artist=""):
    """(prose, pitfalls[])"""
    if info.get("vocals_present") == "no":
        return ("Nobody — it's an instrumental. There's no singer and there are no lyrics.",
                ["It's an instrumental: don't describe a singer, a vocal range or lyrics."
                 + (" (A voice detector thought it heard one; the closer check says what it caught was the "
                    "instruments bleeding through.)" if info.get("vocal_verdict") in ("arbitration", "rechecked") else "")])
    pit = []
    ident = info.get("identity") or ""
    facts = info.get("identity_facts") or {}
    gender = singer_gender(info)
    name = facts.get("name") or ""
    if not name and artist and info.get("lead_singers") != "several":
        name = artist
    unbroken = info.get("voice_unbroken")
    np_ = info.get("note_pitch") or {}
    p = info.get("pitch") or {}
    S = []
    who = name or ("the singer" if gender else "")
    he = {"male": "he", "female": "she"}.get(gender, "they")
    his = {"male": "his", "female": "her"}.get(gender, "their")
    age = facts.get("age")
    if info.get("lead_singers") == "several":
        features = facts.get("features") or []
        named = []
        if name:
            named.append(name)
        for f in features:
            if f not in named:
                named.append(f)
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
    elif name:
        line = f"{name} sings lead"
        if age and 5 <= age <= 99:
            line += f", about {age} at the time"
        S.append(line)
    deepened = info.get("voice_deepened")
    if unbroken and gender == "male":
        S.append(f"{cap(his)} voice hadn't broken yet on this recording: it's high, light and boyish, sitting "
                 "where women's and boys' voices usually sit")
        if age and age >= 15:
            S.append(f"That's later than usual — most boys' voices have dropped by 15 or 16 — so it's a real talking point")
        S.append("Heard cold, without knowing who it is, it can even pass for a woman's voice")
        pit.append(f"The singer is male: say {he}/{his}, never 'she'. {cap(his)} voice is unbroken here — "
                   "describe it as a boy's high, light voice, not a woman's and not a grown man's.")
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
    elif unbroken and not gender:
        S.append("The voice sounds young and high — it could be a boy whose voice hasn't dropped or a woman; "
                 "nothing settles which")
        pit.append("Don't assign the singer a gender: it could be a boy with an unbroken voice or a woman.")
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
            S.append("A single lead voice; whether it's a man or a woman isn't clear from the sound alone")
            pit.append("The singer's gender isn't settled — don't guess one.")
        ac = info.get("acoustic_lead") or ""
        if ident and ac and _is_male(ident) and _is_female(ac):
            S.append(f"{cap(his)} voice sits high enough that, on sound alone, it reads as a woman's")
            pit.append(f"The singer is male: say {he}/{his}, never 'she'.")
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
    if vt:
        S.append(f"For {_a_an('male' if gender == 'male' else 'female')} singer that's {vt.split('(')[0].strip()}"
                 + (" — high for a male lead" if gender == "male" and "tenor" in vt.lower() else ""))
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
    full = level == "full"
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
    if d.get("groove"):
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
    if inst:
        n.append("Underneath: " + join_words(inst))
    if info.get("vocals_present") == "no":
        n.append("There's no singing — it's an instrumental")
    if info.get("excerpt"):
        xa, xb = info["excerpt"]
        n.append(f"You only heard an excerpt, {mmss(xa)} to {mmss(xb)}, so everything here is about that stretch "
                 "(times count from its start)")
    pw = production_words(info)
    if pw:
        n.append("The production: " + join_words(pw))
    mood = mf.get("MOOD_VIBE", "")
    if mood:
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
    gr = info.get("genre_replaced") or {}
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
        b.append(f"Style: {gp}" + (f"; close neighbours are {join_words(nb)}" if nb else ""))
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
    if pw:
        b.append("Production: " + join_words(pw))
    ear = []
    for k, lab in (("INSTRUMENTATION", "Instruments"), ("TIMBRE", "Texture and mix"),
                   ("PRODUCTION_NOTES", "Production"), ("REFERENCE_POINTS", "Reminds you of")):
        if mf.get(k) and not mf[k].lower().startswith(("uncertain", "see ")):
            ear.append(f"{lab}: {scrub_gender(clip(mf[k], 500 if not brief else 250), gender)}")
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
    if d.get("groove"):
        b.append(f"The beat: {groove_plain(d['groove'])}")
    meter = meter_words(info)
    if info.get("bpm"):
        b.append(f"Tempo about {round(float(info['bpm']))} BPM" + (f", in {meter}" if meter else ""))
    if drums_programmed(info):
        b.append("It sounds programmed — a drum machine, most likely — with very even, repeatable kick hits")
    snd = dict(d.get("sound") or [])
    kit = []

    def _words_only(v):
        return [x.strip().replace("/", ", ") for x in split_top(v) if x.strip() and not re.search(r"\d", x)]
    if snd.get("Kick") and _words_only(snd["Kick"]):
        kit.append("a kick with " + join_words([_a_an(x) if not x.startswith(("a ", "an ")) else x
                                                for x in _words_only(snd["Kick"])]))
    if snd.get("Snare / backbeat") and _words_only(snd["Snare / backbeat"]):
        sb = _words_only(snd["Snare / backbeat"])
        kit.append(f"a {sb[0]} snare" + (f" ({'; '.join(sb[1:])})" if sb[1:] else ""))
    if snd.get("Hi-hats") and _words_only(snd["Hi-hats"]):
        kit.append(f"{join_words(_words_only(snd['Hi-hats']))} hi-hats")
    kit = [k for k in kit if k]
    if kit:
        b.append("Drum sound: " + join_words(kit))
    if d.get("extra") and not brief:
        ex = [_tag_word(t["label"]) for t in confident_tags(d["extra"], 30) if "drum machine" not in t["label"].lower()]
        if ex:
            b.append("Other percussion in there: " + join_words(ex))
    if mf.get("DRUMS") and not d.get("groove"):
        b.append(f"By ear: {clip(mf['DRUMS'], 400)}")
    body = "\n".join(f"- {sentence(x)}" for x in b)
    grid, glabel = typical_grid(info)
    if grid and not brief:
        body += (f"\n\nTwo bars of the {glabel} beat, if someone wants to see it (copy it exactly, in a code block; "
                 "x = hit, . = rest, | = bar line; kick, snare and hi-hat only):\n```\n" + "\n".join(grid["rows"]) + "\n```")
    if body.strip():
        S.append(_section("The groove", body, 40, 70))

    # --- The voice --------------------------------------------------------------------------------------
    if info.get("vocals_present") != "no":
        b = []
        tw = voice_timbre_words(info)
        if tw:
            b.append("How it sounds: " + "; ".join(tw))
        if mf.get("LEAD_VOCAL_CHARACTERISTICS") and not mf["LEAD_VOCAL_CHARACTERISTICS"].lower().startswith("see "):
            b.append("By ear: " + scrub_gender(clip(mf["LEAD_VOCAL_CHARACTERISTICS"], 500), gender))
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
        if mf.get("LYRIC_SUBJECT"):
            b.append(f"What the words are about: {clip(mf['LYRIC_SUBJECT'], 300)}")
        if b:
            S.append(_section("The voice", "\n".join(f"- {sentence(x)}" for x in b), 50, 60))

    # --- Melody and harmony ---------------------------------------------------------------------------------
    if not brief:
        b = []
        if info.get("key"):
            kline = f"Key: {info['key']}"
            if info.get("key_conf") and "agree" in info["key_conf"]:
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
        if modal:
            b.append(modal + " (from the transcribed notes, so treat it as a lean, not a certainty)")
        chords = ""
        for l in (info.get("stem_reports") or {}).get("other") or []:
            m = re.search(r"common harmonic clusters: ([A-G][#♯]? (?:major|minor))", l)
            if m:
                chords = m.group(1)
        if chords:
            b.append(f"The most common chord in the keys/synth layer is {chords}")
        if mf.get("CHORDS"):
            b.append(f"Chords by ear (approximate): {clip(mf['CHORDS'], 300)}")
        for k, lab in (("MELODICISM", "Melody and hooks, by ear"), ("MELODIC_CONTOUR", "Shape of the tune, by ear")):
            if mf.get(k):
                b.append(f"{lab}: {scrub_gender(clip(mf[k], 300), gender)}")
        rec, _h = melody_extras(info, "bass")
        if rec and "octave" in rec:
            b.append("The bass line keeps bouncing between a note and its octave")
        if b:
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
    if info.get("bpm"):
        f.append(f"Tempo: {round(float(info['bpm']))} BPM" + (f" ({info['tempo_conf']})" if info.get("tempo_conf") else ""))
    if info.get("key"):
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
        S.extend(deeper_detail(info, full, raw_notes))

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


def deeper_detail(info, full, raw_notes):
    """Per-section detail and raw measurements, for specific questions."""
    S = []
    d = info.get("drums") or {}
    # groove by section
    rows = []
    for sg in d.get("sections") or []:
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
        body += "Which drums are playing when:\n" + "\n".join(
            f"- {mmss(a)}–{mmss(b)}: {drum_config_plain(c)}" for a, b, c in timeline)
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
            keep = [_drop_writer_orders(l) or l for l in ls]
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
        rp = info.get("replay") or {}
        if rp.get("lyric_groups") or info.get("lyric_repeats"):
            rows = ([f"- “{verbatim(t)}” ×{n}, at {times}" for t, n, times in info.get("lyric_repeats") or []]
                    or [f"- “{verbatim(clip(t, 80))}” ×{n}" + (f", at {', '.join(mmss(x) for x in occ)}" if occ else "")
                        for t, n, occ in rp.get("lyric_groups") or []])
            S.append(_section("Detail: lines that come back", "\n".join(rows), 93, 8))
    if raw_notes and info.get("raw_notes"):
        blocks = []
        for st in STEM_ORDER + sorted(set(info["raw_notes"]) - set(STEM_ORDER)):
            rows = info["raw_notes"].get(st)
            if not rows:
                continue
            cells = []
            for r in rows:
                if not isinstance(r, dict):
                    continue
                try:
                    t = f"{float(r.get('t', 0)):.2f}"
                except (TypeError, ValueError):
                    t = "?"
                nm = r.get("k") or r.get("n")
                if not nm:
                    try:
                        nm = midi_name(r["m"])
                    except (KeyError, TypeError, ValueError):
                        nm = "?"
                extra = []
                for key, fmt in (("d", "{:.2f}s"), ("v", "v{:.2f}")):
                    if r.get(key) is not None:
                        try:
                            extra.append(fmt.format(float(r[key])))
                        except (TypeError, ValueError):
                            pass
                cells.append(f"{t} {nm}" + (" " + " ".join(extra) if extra else ""))
            blocks.append(f"{st} ({len(cells)}): " + "; ".join(cells))
        if blocks:
            S.append(_section("Every transcribed note",
                              "Raw note events (seconds, note or drum, length, level 0-1). Approximate machine "
                              "transcription; for exact-timing questions only.\n\n" + "\n\n".join(blocks),
                              99, 1, truncatable=True))
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
    if max_tokens:
        while est_tokens(text) > max_tokens:
            droppable = [s for s in secs if s["prio"] < 90]
            if not droppable:
                break
            victim = min(droppable, key=lambda s: s["prio"])
            if victim["trunc"] and len(victim["body"]) > 800:
                victim["body"] = victim["body"][: len(victim["body"]) // 2].rsplit("\n", 1)[0] + "\n[… truncated to fit …]"
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
# Sentences that add a member later: "Guitarist David Gilmour joined in 1967."
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
# "The Rain were an English band" is about a group; "Paul Langlois is the guitarist of the rock
# band The Tragically Hip" is about a person, so no preposition or role word may come between.
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
# "X is an episode of Being Erica whose title comes from a song by ...".
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
    """'Frankie J. Galasso' / 'Frankie Galasso (singer)' for 'Frankie Galasso': same first and
    last name, middle names and initials ignored."""
    got, want = _name_key(_base_title(title)).split(), _name_key(name).split()
    if not got or not want:
        return False
    if got == want:
        return True
    return len(want) >= 2 and len(got) >= 2 and got[0] == want[0] and got[-1] == want[-1] and len(got) <= len(want) + 2


def _mentions_proper(text, name):
    """Like _mentions, but a capitalised name must appear capitalised: the group Play is
    mentioned in "a member of the girl group Play", not in "to play a song"."""
    if not _mentions(text, name):
        return False
    pat = _artist_pattern(name)
    return bool(pat and re.search(r"\b(?:the[\W_]+)?" + pat + r"\b", _fold(text), re.I))


def _base_title(title):
    """'Blur (band)' -> 'Blur'."""
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
    """Infobox list items that lost their separators: 'Jay SemkoBryan PotvinDon Schmid' ->
    ['Jay Semko', 'Bryan Potvin', 'Don Schmid']. Keeps MacGowan / McCartney / DeLonge whole."""
    # Bulleted lists ("* Matt Ballinger * Greg Raposo", "*Chainsaw Caine *Lance Sabin") first.
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
                continue                      # McCartney, MacGowan, DeLonge
            if prev == "." and len(word) <= 2:
                continue                      # "J.Smith"-style initials stay together
        elif prev.isupper():
            # "Chuck DFlavor Flav": a one-letter word glued to the next name ("D" + "Flavor")
            nxt = v[i + 1] if i + 1 < len(v) else ""
            if not (nxt.islower() and len(word) == 1 and " " in v[start:i]):
                continue
        # never after an apostrophe: O'Brien, D'Arcy, O'Riordan stay whole
        else:
            continue
        out.append(v[start:i])
        start = i
    out.append(v[start:])
    return [re.sub(r"\s+", " ", x).strip(" ,;") for x in out if x.strip(" ,;")]


def split_infobox(text, title=""):
    """Many text dumps start an article with its infobox flattened into 'key: value' pairs
    ("name: Neptune type: Studio album artist: The Northern Pikes released: 1992 ...") before the
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
        # "Jay Semko (vocals, bass)", "Merv Kolb[1]", "See: List of Blur members" are tidied first.
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
        """Titles of the form '<base> (...)' -- e.g. 'Nirvana (American band)' -- for disambiguated names."""
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
        Unlike titles_starting, a common name ("John Taylor", 100+ articles) can't crowd out the
        right one ("John Taylor (bass guitarist)"), because the text has to match too."""
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
        "Play (Swedish group)", "John Taylor (bass guitarist)". The words are matched in the
        title only, so a common name ("Play", "John Taylor") can't crowd the right one out."""
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
        `also` -- for people whose article is under another name ("Janet Leon" -> "Owji") in a
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
    """Primary act from a tag: 'Maroon 5 feat. Christina Aguilera' -> 'Maroon 5' (keeps AC/DC, Hall & Oates)."""
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
# "X is a Northern Irish singer-songwriter", "Andrew White was a Scottish drummer"
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
    has the artist's name ("Peter Gabriel (1977 album)", "Weezer (Blue Album)", "Bad Company")."""
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
            continue   # "Peter Gabriel (1977 album)", "Weezer (Blue Album)", an eponymous debut
        hits = sum(1 for h in hints if h and _mentions(text, h))
        desc = re.search(r"\(([^)]*)\)\s*$", title)
        if desc and not _ACT_DESCRIPTOR_RE.search(desc.group(1)) and any(hints) and not hits:
            continue   # "Andy White (drummer)" for a track by the singer-songwriter Andy White: a
                       # namesake whose page names neither the track's album nor the song
        exact = _name_key(_base_title(title)) in (_name_key(b) for b in bases)
        # Among same-named musicians ("Andy White (drummer)" vs "Andy White (singer-songwriter)")
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
            # "1987 – Big Blue Sky" / "June 1987 – Title": the year comes first, the title after it
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
        # (Like a Virgin the album and "Like a Virgin" the single), so both are kept.
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
    # "Neptune (Northern Pikes album)" for The Northern Pikes.
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
# "Hugh Syme is a Canadian graphic artist best known for album cover concepts".
_NOT_THE_RECORD_RE = re.compile(
    r"\b(?:artist|designer|photographer|musician|singer|songwriter|producer|engineer|manager|actor|actress|"
    r"writer|author|director|person|man|woman|politician|player|DJ|presenter|host|critic|journalist|"
    r"company|known|named|whose|who|which|named after|"
    r"used|appears?|appeared|cover art|artwork)\b", re.I)
# "a song by Cher, covered by The Northern Pikes": the band is not the song's artist.
_COVERED_RE = re.compile(r"\b(?:cover(?:ed|s)?|sampled|remixed|parod(?:y|ied)|interpolat\w*|inspired|tribute|"
                         r"later|also|originally|translated|adapted|reworked|answer|in response)\b", re.I)
_RELEASE_INFOBOX_KEYS = ("released", "recorded", "label", "length", "producer", "studio", "format",
                         "b-side", "b_side", "album", "prev_title", "next_title", "writer", "genre")


def _artist_pattern(artist_name):
    """Regex matching the artist's name in folded text, with or without a leading 'The'. A word
    written with a capital in the name must have its capital in the text too (Wikipedia always
    capitalises names), so the group "Play" doesn't match "to play a song"; the rest of each
    word is case-blind ("PLAY", "Play")."""
    toks = [t for t in re.findall(r"\w+", _fold(artist_name)) if t]
    if len(toks) > 1 and toks[0].lower() == "the":
        toks = toks[1:]

    def tok(t):
        if t[0].isupper():
            return "(?-i:" + re.escape(t[0]) + ")" + re.escape(t[1:])
        return re.escape(t)
    return r"[\W_]+".join(tok(t) for t in toks) if toks else None


def _ib_first_artist(value):
    """'The Northern Pikes featuring X' / 'A and B' / 'A, B' -> 'The Northern Pikes'."""
    v = _split_run_together(value)[0] if value else ""
    v = re.sub(r"\([^)]*\)|\[[^\]]*\]", " ", v)
    return re.split(r"\s+(?:featuring|feat\.?|ft\.?|with|vs\.?|and|&|x)\s+|,|;|/", v, flags=re.I)[0].strip()


def _first_sentence_release_by(first, artist_name):
    """'album' / 'single' when an opening sentence says the article is a recording BY this artist
    ("Neptune is the fourth album by Canadian rock band The Northern Pikes", "She Ain't Pretty is
    The Northern Pikes' biggest hit single"), else None. Films, episodes, novels and the like
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
    # The name must stand alone: not the end of a longer name ("Kate Bush" for Bush, "Sean Paul"
    # for Paul) nor the start of one ("James Brown" for James, "Prince Buster" for Prince).
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
            continue          # "a song by Kate Bush"
        by = m
        break
    own = re.search(r"\b(?:the[\W_]+)?" + pat + r"(?:['’]s?|[\W_]+band['’]s?)?" + alone_after +
                    r"\s+(?:[\w'’-]+\s+){0,5}?(?:album|EP|LP|single|song|mixtape|compilation)s?\b", f, re.I)
    if own:
        prev_word = re.findall(r"[\w'’.-]+$", f[: own.start()].rstrip())
        if prev_word and prev_word[0][0].isupper() and prev_word[0].lower() != "the":
            own = None        # "is Kate Bush's third album"
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
    """Person-name-shaped runs of capitalised words in a fragment ("bassist/keyboardist John
    Paul Jones", "students Syd Barrett", "Liam's older brother, Noel Gallagher")."""
    out = []
    p = re.sub(r"\([^)]*\)", " ", part)
    for m in _NAME_RUN_RE.finditer(p):
        words = m.group(1).rstrip(".").split()
        while words and words[0].lower().strip("'’") in _PERSON_DESCRIPTORS | _NOT_A_NAME:
            words = words[1:]
        if words and re.search(r"['’]s$", words[-1]):
            continue                      # "Liam's" is a possessive, not a name
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
            for part in re.split(r",|;|\band\b|&", l.split(":", 1)[1]):   # "Current members: A, B and C"
                for n in _names_in(part, allow_single=len(part.split()) <= 3):
                    push(n)
            return
        m = _NAME_RE.match(l)            # a credit line: "Thom Yorke – vocals", "Flea (bass)"
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
    # Every '<name> (...)' article that mentions the band: finds "John Taylor (bass guitarist)" or
    # "Roger Taylor (Duran Duran drummer)" however many other John/Roger Taylors there are.
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
            continue   # the article is about a band ("The Rain were an English band"), not a person
        return art
    # No article under that name: one may be under another (a stage name, a married name). In a
    # database without redirects, accept a person's article that mentions the band and gives
    # this name in its opening ("Janet Ava Owji ..., also known by her former stage name Janet Leon").
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
        # lead's prose isn't searched: "formed in 1999 by Louis Baldonieri and Brian Lukow"
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
        # empty -- The Northern Pikes came out with only the album named in the file's own tags.
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
    ("Neptune is the fourth album by Canadian rock band The Northern Pikes"). Found through the
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
    q += f' OR NEAR("artist" {who}, {len(toks) + 2})'        # infobox "artist: The Northern Pikes"
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
)
GUIDE_BACKGROUND_WIKI = """\
- You know this artist's story well (history, line-ups, release dates, track listings, chart runs, how
  records were received). Treat that as your own knowledge: weave it in the way a fan would, as asides
  and stories next to the sound, without waiting to be asked. Career facts are what you know; how this
  recording sounds is what you heard. If a specific fact isn't something you're sure of, say so."""

# Hard rule placed at the very top of every dossier (right under the title). Gemma and similar
# models often treat long files as RAG context and open with "Based on the provided text" unless
# this is stated first, plainly, and repeated at the end.
VOICE_RULE = (
    "You already know this song. What follows is your memory of it — not a document, citation, or "
    "source to quote. Answer as a friend who just listened. Never start with or use: \"based on the "
    "provided text\", \"based on the text\", \"based on the provided\", \"according to the text\", "
    "\"according to the notes\", \"from the document\", \"from the information provided\", "
    "\"the text says\", \"the analysis shows\", or any \"based on…\" / \"according to…\" opener."
)

GUIDE_SINGLE = """\
""" + VOICE_RULE + """

## How to talk about this track

You've just listened to this song properly, more than once. The feel, the voice, the groove, how it
unfolds, the words, when it sounds like it was made — that is what stuck with you.

- Talk like a music-obsessed friend who just put this on: warm, curious, opinionated when the music
  earns it. Plain paragraphs, one flowing take. Lead with how it feels and what grabs you; bring in a
  detail, moment or timestamp only when it helps. A quick question gets a quick answer.
- Jump straight into the music. You are not summarizing a file. Forbidden openers and asides include
  "based on the provided text", "based on the text", "based on what I read", "according to the text",
  "according to the notes", "the document says", "the analysis shows", "from the information
  provided", and "citations". Say the fact itself: "it's a dance-pop track…", "his voice sits high…".
- Weave sound and story in the same sentences: who made it, when, what was going on for them, next to
  what the voice, the beat and the production are doing.
- Everything below is what you heard and know — say it as your own impression. Keep hedges
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
).replace(
    "friend who just put this on:", "friend who just played these back to back:",
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
Talk about it like a friend who loves it — warm, natural, in your own words. Start with the music."""

GUIDE_FOOTER_MULTI = """\
---
""" + VOICE_RULE.replace("this song", "these songs").replace("your memory of it", "your memory of them") + """
Talk about them like a friend who loves them — warm, natural, in your own words. Name each song when
you compare. Start with the music."""

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
    name = (info.get("identity_facts") or {}).get("name") or ""
    if info.get("lead_singers") == "several":
        return "several singers share the lead"
    if info.get("voice_unbroken") and g == "male":
        age = (info.get("identity_facts") or {}).get("age")
        return (f"{name}, " if name else "") + "a boy's unbroken voice" + (f" (about {age})" if age else "")
    if info.get("voice_deepened") and g == "male":
        age = (info.get("identity_facts") or {}).get("age")
        return (f"{name}, " if name else "") + "teenage male, voice already deepened" + (f" (about {age})" if age else "")
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
    row("Beat", lambda i: clip(groove_plain((i.get("drums") or {}).get("kick", "")), 60))
    row("Lead voice", lambda i: short_voice(i))
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


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Convert Musiclyse saved-song .json files into plain-text dossiers "
                    "for stock local LLMs (no system prompt needed).")
    ap.add_argument("inputs", nargs="+", help=".json files, folders, or glob patterns")
    ap.add_argument("-o", "--out-dir", help="output folder (default: 'llm-ready' next to each input)")
    ap.add_argument("--level", choices=("brief", "standard", "full"), default="standard")
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
    ap.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    args = ap.parse_args(argv)

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

    def out_path_for(src, name):
        d = args.out_dir or os.path.join(os.path.dirname(os.path.abspath(src)), "llm-ready")
        os.makedirs(d, exist_ok=True)
        return os.path.join(d, name)

    written = []

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
                 guide_with_background(GUIDE_MULTI, bool(bg_secs)),
                 "## Comparison\n\n" + render_comparison(infos)]
        # The guide and comparison table count against --max-tokens too;
        # splitting the whole budget between the tracks overshot it.
        per = None
        if args.max_tokens:
            overhead = est_tokens("\n\n".join(parts)) + 30 * len(infos)
            per = max(400, (args.max_tokens - overhead) // len(infos))
        if per and bg_secs:
            per = max(400, per * 2 // 3)       # leave a share of the budget for the background
        head = humanize("\n\n".join(parts))
        body = []
        for n, info in enumerate(infos, 1):
            secs = humanized(render_track(info, args.level, not args.no_lyrics, args.max_lyrics_chars, args.raw_notes, cover=args.cover))
            body.append(humanize(f"## Track {n}: {verbatim(info['display'])}").strip() + "\n\n" +
                        assemble(secs, per, heading="###"))
        text = tidy(head + "\n" + "\n\n".join(body))
        if bg_secs:
            room = (args.max_tokens - est_tokens(text) - est_tokens(GUIDE_FOOTER_MULTI)) if args.max_tokens else None
            if room is None or room > 200:
                text = tidy(text + "\n\n" + assemble(bg_secs, room, heading="##"))
        text = tidy(text + "\n\n" + GUIDE_FOOTER_MULTI)
        reports.append(("combined file", all_arts, kept_titles(text, all_arts, "##"), "; ".join(reasons)))
        if args.stdout:
            sys.stdout.write(text)
            return 0
        dest = args.combine
        if not os.path.dirname(dest):
            dest = out_path_for(tracks[0][0], dest)
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
                            guide_with_background(GUIDE_SINGLE, bool(bg_secs)))
            # The guide at the top counts against --max-tokens too.
            budget = max(200, args.max_tokens - est_tokens(head) - est_tokens(GUIDE_FOOTER)) if args.max_tokens else None
            text = tidy(head + "\n" + assemble(secs + bg_secs, budget))
            text = tidy(text + "\n\n" + GUIDE_FOOTER)
            reports.append((os.path.basename(f), arts, kept_titles(text, arts, "##"), reason))
            if args.stdout:
                sys.stdout.write(text + ("\n" if len(tracks) > 1 else ""))
                continue
            base = os.path.splitext(os.path.basename(f))[0]
            dest = out_path_for(f, f"{base}.{args.ext}")
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
