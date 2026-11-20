# MUSICLYSE FILE pipeline-sections 0.8
# ==============================================================================
#  MUSICLYSE 0.8: THE SECTION MAP
# ==============================================================================
# This file holds the timestamped section map: boundaries, section names, SongFormer, the
# section-labelling prompt and the replay of saved maps. It is part of musiclyse-0.8.py, which
# loads it whole at start-up, after its own code, into its own namespace (see "THE MUSICLYSE
# FILES" there); it is not run on its own.
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


_TAG_SECTION_MARKER_RE = re.compile(
    r"\[.*\]|\(\s*(?:(?:pre-?|post-?)?chorus|verse|bridge|intro|outro|hook|refrain|interlude|break|"
    r"instrumental|solo|repeat\w*|x\s*\d+|\d+\s*x|\u00d7\s*\d+)\b[^)]*\)", re.IGNORECASE)


def _timed_tag_lyrics_for_structure(resolved_path, tag_lyrics, vocal_stem_path=None):
    """Timed lyric chunks for a track whose file carries its lyrics.

    The Whisper lyrics pass is skipped for such tracks (the tags are the
    authoritative text), which also left the section map with no lyric
    timing at all -- so chorus vs verse was decided from level, register and
    drums alone, and a song whose choruses are quieter than its verses, or
    that opens with its chorus, came out with every label swapped. One
    Whisper decode now supplies the timing; its text goes nowhere but the
    section map (see _align_chunks_to_tag_lyrics) and the report keeps the
    tag lyrics as its lyric source.
    """
    if WHISPER_HOLD_LANGUAGE:
        _WHISPER_LANG_STATE["tag_lyrics"] = tag_lyrics     # 0.7.1-pre21: English tags give English
    try:
        status("Timing the embedded lyrics against the audio for the section map (Whisper)...")
        if WHISPER_WORD_TIMING:
            # dev55: word timestamps first; the dev54 line-timed pass and its
            # retries run only when these are unusable or match worse than
            # the retry threshold.
            wchunks = _align_chunks_to_tag_lyrics(_whisper_word_timed_chunks(resolved_path), tag_lyrics)
            try:
                wq = _tag_timing_quality(wchunks, tag_lyrics) if wchunks else None
            except Exception:
                wq = None
            if wchunks and (wq is None or wq >= WHISPER_TIMING_RETRY_BELOW):
                status_done(f"Lyric timing found for {len(wchunks)} line(s) from word timestamps"
                            + (f" ({wq * 100:.0f}% of the embedded lyrics matched)" if wq is not None else ""))
                return wchunks
        _txt, chunks = _whisper_transcribe_once(resolved_path, return_chunks=True)
        chunks = _align_chunks_to_tag_lyrics(chunks, tag_lyrics)
        status_done(f"Lyric timing found for {len(chunks)} line group(s)")
        if WHISPER_TIMING_RETRY:
            chunks = _retry_tag_lyric_timing(chunks, resolved_path, tag_lyrics, vocal_stem_path)
        if WHISPER_WORD_TIMING and 'wchunks' in locals() and wchunks:
            # Keep whichever timing matches the embedded lyrics better.
            try:
                q_line = _tag_timing_quality(chunks, tag_lyrics)
                if wq is not None and (q_line is None or wq > q_line):
                    return wchunks
            except Exception:
                pass
        return chunks
    except Exception as e:
        print(f"  (lyric timing for the section map skipped: {e})")
        return []
    finally:
        try:
            unload_whisper()
        except Exception:
            pass


def _flag_repeated_section_drift(text):
    """
    Detect likely word-substitution hallucination on a repeated section
    (e.g. the same chorus sung 2-3x but transcribed with different wording
    each time) and annotate it rather than silently altering it — we can't
    tell which occurrence (if any) is correct, only that they disagree more
    than a genuinely repeated section should.

    This targets the failure mode where the *first* occurrence of a section
    is transcribed accurately and a *later* occurrence of the same sung
    section drifts into different words (a "hallucinated" repeat), which
    plain loop/spam detection can't catch because the drifted text is
    coherent, on-theme, and not literally repeated.
    """
    if not text:
        return text

    # Group lyric lines under the section header they follow, e.g. "[Chorus] (0:34-0:58)"
    section_pat = re.compile(r"^\s*\[([^\]]+)\]", re.IGNORECASE)
    sections = {}
    current = None
    for line in text.splitlines():
        m = section_pat.match(line)
        if m:
            current = m.group(1).strip().lower()
            current = re.sub(r"\s*\d+$", "", current).strip()  # "chorus 2" -> "chorus"
            sections.setdefault(current, []).append([])
            continue
        if current is not None and line.strip():
            sections[current][-1].append(line.strip())

    def _norm_words(lines):
        words = re.findall(r"[a-z']+", " ".join(lines).lower())
        return set(w for w in words if len(w) > 2)

    flags = []
    for name, occurrences in sections.items():
        if name != "chorus" and "chorus" not in name:
            continue
        if len(occurrences) < 2:
            continue
        word_sets = [_norm_words(o) for o in occurrences if o]
        for i in range(1, len(word_sets)):
            a, b = word_sets[0], word_sets[i]
            if not a or not b:
                continue
            overlap = len(a & b) / max(1, len(a | b))
            if overlap < 0.35:
                flags.append(
                    f"occurrence {i + 1} of [{name}] shares only "
                    f"{overlap:.0%} of its words with occurrence 1"
                )

    if flags:
        text = (
            text.rstrip()
            + "\n\n[TRANSCRIPTION CAUTION — possible repeated-section drift: "
            + "; ".join(flags)
            + ". A repeated section normally uses the same words each time it "
            "recurs; this large a mismatch suggests the model paraphrased or "
            "invented wording on at least one occurrence rather than "
            "transcribing what was actually sung. Treat the disagreeing "
            "occurrence(s) as unverified and prefer the first occurrence, "
            "file-tag lyrics, or an [inaudible] read over quoting this "
            "wording as fact.]"
        )

    return text


def _lyric_block_boundary_candidates(lyric_groups, min_spacing_s=10.0, duration=None, block_gap_s=12.0):
    """Boundary candidates from *blocks* of recurring lyrics, not every line.

    Each recurring line can fire many times inside a single chorus. Treating
    every occurrence start as a section boundary floods the snap pipeline and
    pulls verse/chorus edges onto mid-phrase line starts — which made structure
    *worse*. Instead:

      1. Take the strongest recurring groups (hooks/chorus lines).
      2. Cluster nearby occurrences into blocks (gap > block_gap_s => new block).
      3. Emit only the *start of each block* as a candidate.

    Those block starts are the useful same-timbre chorus-entry signals.
    """
    if not lyric_groups:
        return []
    # Prefer higher-count groups; ignore weak pairs.
    ranked = sorted(
        (g for g in lyric_groups if int(g.get("count") or 0) >= 2),
        key=lambda g: -int(g.get("count") or 0),
    )
    if not ranked:
        return []

    block_starts = []
    # Use up to the top 3 recurring groups so we don't only track one line.
    for g in ranked[:3]:
        starts = []
        for start, _end in g.get("occurrences") or []:
            try:
                t = float(start)
            except Exception:
                continue
            if duration is not None and (t < 2.0 or t > float(duration) - 2.0):
                continue
            if t >= 2.0:
                starts.append(t)
        if not starts:
            continue
        starts.sort()
        # Cluster into blocks.
        cluster = [starts[0]]
        for t in starts[1:]:
            if t - cluster[-1] <= block_gap_s:
                cluster.append(t)
            else:
                block_starts.append(cluster[0])
                cluster = [t]
        block_starts.append(cluster[0])

    return _merge_boundary_candidates(
        [block_starts], min_spacing_s=min_spacing_s, duration=duration, max_boundaries=12
    )


def _midi_activity_boundary_candidates(stem_notes, duration=None, window_s=6.0, min_spacing_s=8.0):
    """Sparse MIDI density-change candidates (secondary evidence only).

    Coarser windows and higher thresholds than an earlier draft — dense
    note-on streams otherwise invent false section cuts.
    """
    if not stem_notes or not isinstance(stem_notes, dict):
        return []
    onsets = []
    for stem, notes in stem_notes.items():
        if not notes or stem == "drums":
            # Kit density is high almost everywhere; skip for structure.
            continue
        for n in notes:
            try:
                onsets.append(float(n.get("onset", 0.0)))
            except Exception:
                pass
    if len(onsets) < 30:
        return []
    onsets = np.asarray(onsets, dtype=float)
    if duration is None or duration <= 0:
        duration = float(np.max(onsets) + 1.0)
    duration = float(duration)
    if duration < 25.0:
        return []

    n_bins = max(6, int(duration / window_s))
    hist, edges = np.histogram(onsets, bins=n_bins, range=(0.0, duration))
    hist = hist.astype(float)
    if hist.max() <= 0:
        return []
    eps = 1e-6
    rel = np.abs(np.diff(hist)) / (np.maximum(hist[:-1], hist[1:]) + eps)
    norm = hist / (hist.max() + eps)
    abs_jump = np.abs(np.diff(norm))
    score = 0.55 * rel + 0.45 * abs_jump
    if score.max() <= 1e-9:
        return []
    score = score / score.max()
    candidates = []
    for i, sc in enumerate(score):
        if sc < 0.55:  # stricter than before
            continue
        t = float(edges[i + 1])
        if 4.0 <= t <= duration - 4.0:
            candidates.append((sc, t))
    candidates.sort(reverse=True)
    chosen = []
    for sc, t in candidates:
        if all(abs(t - c) >= min_spacing_s for c in chosen):
            chosen.append(t)
        if len(chosen) >= 8:
            break
    return sorted(chosen)


def _merge_boundary_candidates(lists, min_spacing_s=4.0, duration=None, max_boundaries=24):
    """Merge boundary lists and drop near-duplicates (keep first of a cluster).

    Does *not* midpoint-average near pairs — that pulled accurate novelty peaks
    toward noisy lyric/MIDI times and degraded snap quality.
    """
    times = []
    for lst in lists:
        for t in (lst or []):
            try:
                ft = float(t)
            except Exception:
                continue
            if duration is not None:
                if ft < 2.0 or ft > float(duration) - 2.0:
                    continue
            times.append(ft)
    if not times:
        return []
    times = sorted(times)
    merged = [times[0]]
    for t in times[1:]:
        if t - merged[-1] >= min_spacing_s:
            merged.append(t)
        # else: keep the earlier time (usually the stronger / first source)
    if len(merged) > max_boundaries:
        step = len(merged) / float(max_boundaries)
        merged = [merged[int(i * step)] for i in range(max_boundaries)]
    return [round(t, 2) for t in merged]


def _collect_structure_boundaries(
    objective_report=None,
    stem_midi_report=None,
    lyric_groups=None,
    midi_stem_notes=None,
    total_duration=None,
):
    """Collect boundary candidates by source. Prefer acoustic novelty for snap;
    lyric *block* starts are secondary; MIDI is tertiary/reporting only."""
    acoustic = extract_objective_structure_boundaries(objective_report or "")
    midi_from_text = []
    if stem_midi_report:
        m = re.search(
            r"MIDI ACTIVITY BOUNDARY CANDIDATES[^:]*:\s*([0-9:,.\s]+)",
            stem_midi_report,
            re.IGNORECASE,
        )
        if m:
            for tok in m.group(1).split(","):
                secs = _mmss_to_seconds(tok.strip())
                if secs is not None:
                    midi_from_text.append(secs)
    midi_from_notes = (
        _midi_activity_boundary_candidates(midi_stem_notes, duration=total_duration)
        if midi_stem_notes else []
    )
    lyric = _lyric_block_boundary_candidates(lyric_groups, duration=total_duration)
    return {
        "acoustic": acoustic,
        "lyric": lyric,
        "midi": _merge_boundary_candidates(
            [midi_from_text, midi_from_notes],
            min_spacing_s=8.0,
            duration=total_duration,
            max_boundaries=8,
        ),
    }


def _snap_structure_sections_tiered(
    sections,
    acoustic_bounds,
    lyric_bounds=None,
    midi_bounds=None,
    tolerance=None,
    min_section_s=3.0,
    total_duration=None,
):
    """Snap internal section edges with source priority:

      1. Acoustic novelty/energy peaks (most reliable for true arrangement changes)
      2. Lyric *block* starts (same-timbre chorus entries)
      3. MIDI density jumps (weakest; only if nothing else is nearby)
      4. Keep the model guess

    Each objective peak is used at most once. This replaces the earlier
    flat merge that treated every recurring lyric line as equal to a novelty peak.
    """
    if not sections:
        return []
    if tolerance is None:
        tolerance = STRUCTURE_BOUNDARY_SNAP_TOLERANCE_S

    out = [dict(s) for s in sections]
    out.sort(key=lambda s: (float(s["start"]), float(s["end"])))

    tiers = [
        sorted(float(t) for t in (acoustic_bounds or [])),
        sorted(float(t) for t in (lyric_bounds or [])),
        sorted(float(t) for t in (midi_bounds or [])),
    ]
    used = set()

    for i in range(len(out) - 1):
        target = float(out[i]["end"])
        prev_floor = float(out[i]["start"]) + min_section_s
        next_ceil = float(out[i + 1]["end"]) - min_section_s

        chosen = None
        for peaks in tiers:
            candidates = []
            for t in peaks:
                # Quantize key for used-set stability
                key = round(t, 2)
                if key in used:
                    continue
                dist = abs(t - target)
                if dist > tolerance:
                    continue
                if t < prev_floor or t > next_ceil:
                    continue
                candidates.append((dist, t, key))
            if candidates:
                candidates.sort()
                dist, t, key = candidates[0]
                chosen = (t, key)
                break

        if chosen is None:
            continue
        t, key = chosen
        out[i]["end"] = t
        out[i + 1]["start"] = t
        used.add(key)

    return _normalize_structure_sections(out, total_duration, min_section_s)


def _annotate_structure_with_lyric_blocks(sections, lyric_groups):
    """Annotate sections that host recurring lyric blocks — do *not* rename.

    Earlier auto-renames (Verse -> Chorus) corrupted good listening-model labels. Annotation
    stays in format_recommended_structure_block via lyric_groups.
    """
    return sections


# ==============================================================================
# OBJECTIVE-FIRST STRUCTURE PIPELINE (Bypasses LLM temporal hallucinations)
# ==============================================================================

def _segment_theme_features(y, sr, boundaries):
    """Per-segment acoustic fingerprint used for recurrence clustering.

    The previous version mean-pooled raw MFCCs (including coefficient 0) and
    raw chroma into one vector and compared them with cosine similarity. That
    is dominated by MFCC[0], which is overall log-energy and is typically
    ~-200 while every other dimension is in single or double digits. The
    practical effect was that any two segments of a normally-mastered track
    scored ~0.87-0.99 similar regardless of whether they were the same
    section, so every segment collapsed into one cluster -- and because the
    caller merges adjacent same-cluster segments, the whole song collapsed
    into a single section. That is the root cause of structure detection
    failing outright on real material.

    Fixed by: dropping MFCC[0], z-scoring each family across segments so the
    comparison is relative to THIS track, L2-normalising each family
    separately, and weighting timbre and harmony explicitly. Loudness enters
    as its own small block rather than as an accidental dominant term.
    """
    if not boundaries or len(boundaries) < 2:
        return [], []

    hop_length = 512
    raw = []
    valid = []
    for i in range(len(boundaries) - 1):
        start_s = float(boundaries[i])
        end_s = float(boundaries[i + 1])
        seg = y[int(start_s * sr):int(end_s * sr)]
        if (end_s - start_s) < 2.0 or len(seg) < sr * 2:
            raw.append(None)
            continue
        try:
            mf = librosa.feature.mfcc(y=seg, sr=sr, n_mfcc=20, hop_length=hop_length)
            ch = librosa.feature.chroma_cqt(y=seg, sr=sr, hop_length=hop_length)
            rm = librosa.feature.rms(y=seg, hop_length=hop_length)
        except Exception:
            raw.append(None)
            continue
        # Median over the segment: robust to a stray fill or a held note.
        timbre = np.median(mf[1:, :], axis=1)
        harmony = np.median(ch, axis=1)
        # Chroma normalised to its own max -> key/voicing profile, not level.
        hmax = float(np.max(harmony))
        if hmax > 1e-8:
            harmony = harmony / hmax
        loud = np.array([float(np.log1p(np.mean(rm) * 1000.0))])
        # Note-density proxy: how much the timbre moves within the segment.
        flux = np.array([float(np.mean(np.std(mf[1:, :], axis=1)))])
        raw.append((timbre, harmony, loud, flux))
        valid.append(len(raw) - 1)

    if len(valid) < 2:
        return [None] * len(raw), []

    def _stack(idx):
        return np.array([raw[i][idx] for i in valid], dtype=float)

    T, H, L, F = _stack(0), _stack(1), _stack(2), _stack(3)

    def _zs(a):
        mu = a.mean(axis=0, keepdims=True)
        sd = a.std(axis=0, keepdims=True)
        sd[sd < 1e-8] = 1e-8
        return (a - mu) / sd

    def _rows_l2(a, weight):
        nrm = np.linalg.norm(a, axis=1, keepdims=True)
        nrm[nrm < 1e-8] = 1e-8
        return weight * (a / nrm)

    block = np.hstack([
        _rows_l2(_zs(T), 1.00),   # timbre / instrumentation
        _rows_l2(_zs(H), 0.85),   # harmony / pitch content
        _rows_l2(_zs(L), 0.35),   # loudness
        _rows_l2(_zs(F), 0.30),   # internal activity
    ])
    nrm = np.linalg.norm(block, axis=1, keepdims=True)
    nrm[nrm < 1e-8] = 1e-8
    block = block / nrm

    feats = [None] * len(raw)
    for k, i in enumerate(valid):
        feats[i] = block[k]
    return feats, valid


def _cluster_audio_segments(y, sr, boundaries):
    """Group segments into recurring acoustic themes (A, B, A, C, A ...).

    Uses the normalised fingerprints from _segment_theme_features with
    best-match (not first-match) assignment and running-mean cluster centres,
    and derives the merge threshold from the observed similarity distribution
    of this track rather than a fixed 0.72 that meant nothing after the
    normalisation change.
    """
    feats, valid = _segment_theme_features(y, sr, boundaries)
    if not feats:
        return []
    if len(valid) < 2:
        return ["Unique" if f is None else "A" for f in feats]

    # Adaptive threshold: sit above the typical between-segment similarity so
    # only genuinely alike segments merge, but never demand near-identity.
    vecs = np.array([feats[i] for i in valid], dtype=float)
    sims = vecs @ vecs.T
    iu = np.triu_indices(len(valid), k=1)
    offdiag = sims[iu]
    if offdiag.size:
        med = float(np.median(offdiag))
        hi = float(np.percentile(offdiag, 80))
        threshold = float(np.clip(max(hi, med + 0.12), 0.45, 0.93))
    else:
        threshold = 0.70

    labels = []
    centers = []   # list of [running_mean_vector, count]
    for feat in feats:
        if feat is None:
            labels.append("Unique")
            continue
        best_idx, best_sim = -1, -1.0
        for idx, (center, _count) in enumerate(centers):
            s = float(np.dot(feat, center) /
                      (np.linalg.norm(center) * np.linalg.norm(feat) + 1e-9))
            if s > best_sim:
                best_idx, best_sim = idx, s
        if best_idx >= 0 and best_sim >= threshold:
            center, count = centers[best_idx]
            centers[best_idx] = [(center * count + feat) / (count + 1), count + 1]
            labels.append(chr(65 + best_idx))
        else:
            centers.append([np.array(feat, dtype=float), 1])
            labels.append(chr(65 + len(centers) - 1))
    return labels


STRUCTURE_LABELING_PROMPT = """I have already segmented this track objectively, using signal analysis of the audio itself. The segment boundaries below are FIXED and CORRECT — your job is only to name each part.

An acoustic similarity cluster has also been computed. Parts sharing a cluster letter are musically the same material returning (e.g. the same chorus coming back).

The segments are:
{segment_list}

Listen to the audio and assign a musical function to each part.
RULES:
- Output ONLY a numbered list mapping the Part number to your chosen label.
- Use standard song-structure labels where they apply (Intro, Verse, Pre-Chorus, Chorus, Bridge, Instrumental, Solo, Breakdown, Outro).
- If the music is instrumental or non-standard, use functional labels (Theme A, Theme B, Development, Section 1...).
- Cluster letters are evidence of acoustic similarity, NOT absolute structural truth. Prefer the same base label for matching clusters when the audio supports it, but do not force it if lyrics/arrangement clearly differ.
- Different cluster letters may still share a structural role when instrumentation changes between repetitions (for example a stripped verse and fuller verse).
- Do NOT output timestamps, durations, commentary, or any other text. Just the numbered list.

Example format:
1. Intro
2. Verse 1
3. Chorus 1
"""


def _auto_label_sections_from_clusters(cluster_labels, spans, total_duration,
                                       energies=None, vocal_coverage=None):
    """Deterministic section names derived only from measured structure.

    Reasoning is purely structural and energetic -- which material recurs, how
    often, where it sits, and how loud it is relative to the rest of the track.
    No genre assumptions and no outside knowledge of the song are involved, so
    these names stay valid evidence about this recording specifically.

    Order of decisions matters: bookend segments (intro/outro) are settled and
    removed FIRST, because otherwise a track whose intro and outro share
    material has that bookend cluster counted as "recurring" and it can win the
    verse or chorus slot outright -- which is exactly how a verse ends up
    labelled as a chorus.
    """
    n = len(spans)
    if n == 0:
        return []

    lengths = [float(e) - float(s) for s, e in spans]
    energies = list(energies) if energies else []
    while len(energies) < n:
        energies.append(None)

    labels = [None] * n
    body = list(range(n))

    # How often each cluster occurs across the WHOLE track. Material that comes
    # back elsewhere is not a bookend, however short it is: a closing chorus is
    # a chorus, not an outro, and treating it as one both mislabels it and
    # removes it from the recurrence counting that identifies the hook.
    all_counts = {}
    for i in range(n):
        cl = cluster_labels[i] if i < len(cluster_labels) else "Unique"
        if cl and cl != "Unique":
            all_counts[cl] = all_counts.get(cl, 0) + 1

    def _recurs(i):
        cl = cluster_labels[i] if i < len(cluster_labels) else "Unique"
        return all_counts.get(cl, 0) >= 2 and any(
            j != i and cluster_labels[j] == cl and 0 < j < n - 1
            for j in range(n)
        )

    # --- 1. Bookends -------------------------------------------------
    if n >= 3 and lengths[0] <= STRUCTURE_INTRO_MAX_S and not _recurs(0):
        labels[0] = "Intro"
        body.remove(0)
    if n >= 3 and lengths[-1] <= STRUCTURE_OUTRO_MAX_S and (n - 1) in body and not _recurs(n - 1):
        labels[n - 1] = "Outro"
        body.remove(n - 1)

    # --- 1b. Instrumental sections -----------------------------------
    # Settled here, BEFORE recurrence counting, for the same reason bookends
    # are: an instrumental break often shares its backing material with a sung
    # section, so leaving it in the body lets it inflate that cluster's count
    # and can hand the chorus slot to the wrong material.
    #
    # Only applied when the track demonstrably has singing somewhere. On a
    # genuinely instrumental piece every section would qualify, and labelling
    # them all "Instrumental" states the obvious while destroying the neutral
    # Section 1..N naming that such tracks should keep.
    coverage = list(vocal_coverage) if vocal_coverage else []
    if coverage:
        while len(coverage) < n:
            coverage.append(None)
        vals = [c for c in coverage[:n] if c is not None]
        track_has_vocals = bool(vals) and max(vals) >= STRUCTURE_VOCAL_TRACK_MIN_COVERAGE
        if track_has_vocals:
            for i in list(body):
                c = coverage[i]
                if c is None or c > STRUCTURE_INSTRUMENTAL_MAX_COVERAGE:
                    continue
                # Position decides the name: a silent-voice stretch at either
                # end is an intro/outro, not a mid-song break.
                if i == 0:
                    labels[i] = "Intro"
                elif i == n - 1:
                    labels[i] = "Outro"
                else:
                    labels[i] = "Instrumental"
                body.remove(i)
        # Bookends already named above may also be voiceless; that is fine and
        # needs no change, since Intro/Outro already implies what it is.

    # --- 2. Recurrence within the body only --------------------------
    counts = {}
    for i in body:
        cl = cluster_labels[i] if i < len(cluster_labels) else "Unique"
        if cl and cl != "Unique":
            counts[cl] = counts.get(cl, 0) + 1
    repeated = [cl for cl, c in counts.items() if c >= 2]

    def _first_idx(cl):
        return min((i for i in body if cluster_labels[i] == cl), default=n)

    def _mean_energy(cl):
        vals = [energies[i] for i in body
                if cluster_labels[i] == cl and energies[i] is not None]
        return float(sum(vals) / len(vals)) if vals else None

    chorus_cl = verse_cl = None
    # Calling something a chorus is a claim that it contrasts with something
    # else. When the body is a single recurring block with nothing to contrast
    # against, there is no evidence for that claim, so it keeps a neutral name.
    if repeated and not (len(repeated) == 1 and counts[repeated[0]] == len(body)):
        # The hook is the material that recurs most. On a tie -- extremely
        # common, since verse and chorus usually appear the same number of
        # times -- prefer the louder one, and failing that the one that enters
        # later, since a chorus almost always answers a verse rather than
        # opening the body of the song.
        best_count = max(counts[cl] for cl in repeated)
        top = [cl for cl in repeated if counts[cl] == best_count]
        if len(top) == 1:
            chorus_cl = top[0]
        else:
            with_energy = [(cl, _mean_energy(cl)) for cl in top]
            if all(e is not None for _cl, e in with_energy):
                loudest = max(with_energy, key=lambda kv: kv[1])
                quietest = min(with_energy, key=lambda kv: kv[1])
                if loudest[1] - quietest[1] > 0.05:
                    chorus_cl = loudest[0]
            if chorus_cl is None:
                chorus_cl = max(top, key=_first_idx)

        others = [cl for cl in repeated if cl != chorus_cl]
        if others:
            verse_cl = min(others, key=_first_idx)

    # --- 3. Assign --------------------------------------------------
    for i in body:
        cl = cluster_labels[i] if i < len(cluster_labels) else "Unique"
        if chorus_cl is not None and cl == chorus_cl:
            labels[i] = "Chorus"
        elif verse_cl is not None and cl == verse_cl:
            labels[i] = "Verse"
        elif verse_cl is None and chorus_cl is None:
            labels[i] = "Section"
        else:
            labels[i] = None   # resolved below

    # Unmatched body material: a one-off in the back half of the track is a
    # bridge; anything else keeps a neutral, honest name.
    #
    # Leftovers are named by their ORDINAL POSITION in the track, never by
    # cluster letter. Letter names were wrong twice over: a lone "Section A"
    # says nothing about where it sits, and two leftovers sharing a cluster
    # both became "Section A", which the numbering pass below then turned into
    # "Section A 1" and "Section A 2" — a cluster id masquerading as a position.
    # An ordinal is unique by construction, so it never collides in that pass,
    # and it tells the reader where in the song the part is.
    #
    # Recurrence information is not lost: the "cluster" field on each section
    # already records which segments share material, and it is saved to JSON.
    # The display name does not need to carry it as well.
    for i in body:
        if labels[i] is not None:
            continue
        cl = cluster_labels[i] if i < len(cluster_labels) else "Unique"
        position = i / float(max(1, n - 1))
        if (STRUCTURE_BRIDGE_MIN_POS < position < STRUCTURE_BRIDGE_MAX_POS
                and counts.get(cl, 0) <= 1):
            labels[i] = "Bridge"
        else:
            labels[i] = "Section %d" % (i + 1)

    # --- 4. Number repeats ------------------------------------------
    totals = {}
    for name in labels:
        totals[name] = totals.get(name, 0) + 1
    seen = {}
    out = []
    for name in labels:
        if totals[name] > 1:
            seen[name] = seen.get(name, 0) + 1
            out.append("%s %d" % (name, seen[name]))
        else:
            out.append(name)
    return out


def _parse_structure_labels(llm_response, expected_parts):
    """Pull 'N. Label' pairs out of the audio model's reply."""
    labels = {}
    if not llm_response:
        return labels
    for line in str(llm_response).splitlines():
        m = re.match(
            r"\s*(?:Part\s*)?(\d{1,2})\s*[\.\):\-]\s*(.+?)\s*$",
            line,
            re.IGNORECASE,
        )
        if not m:
            continue
        idx = int(m.group(1))
        if not (1 <= idx <= expected_parts):
            continue
        label = m.group(2).strip().strip("*_-–— ").strip()
        # Reject anything that smuggled a timestamp or an essay back in.
        if not label or len(label) > 40:
            continue
        if re.search(r"\d\s*:\s*\d\d", label):
            label = re.sub(r"\(?\s*\d\s*:\s*\d\d.*$", "", label).strip(" -–—(")
        if not label:
            continue
        labels[idx] = label
    return labels


def _labels_respect_clusters(labels_by_idx, cluster_labels):
    """True when the model's names agree with the measured recurrence pattern.

    If two parts are acoustically the same returning material they should carry
    the same base label, and parts made of different material should not. When
    the names violate that, they are describing something other than what is in
    the audio -- typically a remembered or expected song shape -- and the
    deterministic names are used instead.

    The first and last segments are exempt. An intro and an outro very often
    share material (same sparse texture bookending the track) while correctly
    carrying different names, so including them would reject good labels on a
    large fraction of perfectly ordinary songs.
    """
    if not labels_by_idx or not cluster_labels:
        return False

    def _base(name):
        return re.sub(r"\s*\d+\s*$", "", str(name)).strip().lower()

    n = len(cluster_labels)
    interior = range(1, n - 1) if n >= 3 else range(n)

    by_cluster = {}
    by_base = {}
    for i in interior:
        label = labels_by_idx.get(i + 1)
        cl = cluster_labels[i]
        if not label or not cl or cl == "Unique":
            continue
        base = _base(label)
        by_cluster.setdefault(cl, set()).add(base)
        by_base.setdefault(base, set()).add(cl)

    if not by_cluster:
        return bool(labels_by_idx)

    for _cl, bases in by_cluster.items():
        if len(bases) > 1:
            return False
    for _base_name, cls in by_base.items():
        if len(cls) > 1:
            return False
    return True


def _lyric_block_segmentation(lyric_chunks, lyric_groups, min_words=2, gap_s=6.0):
    """Segment the sung lines into BLOCKS -- the lyric equivalent of sections.

    Each transcribed line is tagged with the repetition group its words belong
    to, or None when the words occur only once in the track. Lines are then
    grouped into blocks, and a new block is started when any of these happens:

      1. a silence longer than `gap_s` (an instrumental passage),
      2. the line switches between repeating and non-repeating words
         (a chorus giving way to a fresh verse, or the reverse),
      3. a group already used in the current block comes round again
         (the words have looped, so this is the next block),
      4. the line's group is the one that OPENED some earlier block -- i.e. a
         known block is starting over.

    Rule 4 is what makes recurring sections work. A four-line chorus is four
    separate repetition groups, so testing "did the group change since the last
    line" fires between every pair of lines inside one chorus and shreds it
    into fragments. Testing whether a block is RESTARTING fires once, at the
    edge, which is where a section boundary actually is.

    Each block gets a signature (the frozenset of groups it contains), so two
    blocks made of the same words are recognisable as the same section
    returning even if the lines were transcribed in a slightly different order.

    Returns a list of {"start", "end", "signature", "sung"} in time order.
    """
    if not lyric_chunks:
        return []

    occ_to_group = {}
    for gid, g in enumerate(lyric_groups or []):
        for start, _end in g.get("occurrences") or []:
            occ_to_group[round(float(start), 2)] = gid

    lines = []
    for c in lyric_chunks:
        try:
            s = float(c.get("start", -1))
            e = float(c.get("end", s))
        except Exception:
            continue
        if s < 0:
            continue
        norm = _normalize_lyric_line(c.get("text", ""))
        if len(norm.split()) < min_words:
            continue
        lines.append((s, e, occ_to_group.get(round(s, 2))))
    lines.sort(key=lambda x: x[0])
    if not lines:
        return []

    blocks = []
    cur = [lines[0]]
    cur_groups = {lines[0][2]} if lines[0][2] is not None else set()
    openers = set()
    if lines[0][2] is not None:
        openers.add(lines[0][2])

    def _close(block):
        gs = frozenset(g for (_s, _e, g) in block if g is not None)
        blocks.append({
            "start": float(block[0][0]),
            "end": float(block[-1][1]),
            "signature": gs if gs else None,
            "sung": True,
        })

    for (ps, pe, pg), (cs, ce, cg) in zip(lines, lines[1:]):
        new_block = False
        if (cs - pe) > gap_s:
            new_block = True
        elif (pg is None) != (cg is None):
            new_block = True
        elif cg is not None and cg in cur_groups:
            new_block = True
        elif cg is not None and cg in openers and cg != pg:
            new_block = True

        if new_block:
            _close(cur)
            cur = [(cs, ce, cg)]
            cur_groups = {cg} if cg is not None else set()
            if cg is not None:
                openers.add(cg)
        else:
            cur.append((cs, ce, cg))
            if cg is not None:
                cur_groups.add(cg)
    _close(cur)
    return blocks


def _structure_sections_are_sane(sections, total_duration=None):
    """Is a parsed section map usable as-is?

    Checks only for structural nonsense -- out-of-order or overlapping spans,
    times past the end of the track, absurd section lengths, or too few
    sections to be a map at all. A map that passes is trusted; being unusual
    is not the same as being wrong, and plenty of real records are laid out
    oddly.
    """
    if not sections or len(sections) < 2:
        return False
    dur = float(total_duration) if total_duration else None
    prev_end = -1.0
    for s in sections:
        try:
            start = float(s["start"])
            end = float(s["end"])
        except Exception:
            return False
        if end <= start:
            return False
        if start < prev_end - 1.5:          # meaningful overlap
            return False
        if (end - start) < 2.0 and not STRUCTURE_SANITY_SHORT_OK:   # too short to be a section
            return False                    # (0.7.1-pre12 F-13: with the setting, the normaliser merges it)
        if dur and start > dur + 1.0:
            return False
        prev_end = end
    if dur:
        covered = sum(float(s["end"]) - float(s["start"]) for s in sections)
        # A map that accounts for almost none of the track isn't describing it.
        if covered < dur * 0.5:
            return False
    return True


def _arrangement_change_points(stem_notes, bpm=None, total_duration=None):
    """Moments where the arrangement itself changes: instruments entering or
    leaving.

    Drums are split by drum type, because "the snare arrives" is one of the
    most reliable section markers in popular music and is invisible if the kit
    is treated as a single source.

    Returns a list of dicts sorted by time:
        {"time", "stems", "n_stems", "events"}
    where n_stems counts DISTINCT sources moving together. That count is the
    trust signal: several stems changing at once is a section boundary,
    one stem changing alone usually is not.
    """
    if not ARRANGEMENT_CHANGE_ENABLED or not stem_notes:
        return []
    try:
        beat_len = 60.0 / float(bpm) if bpm else 0.5
    except (TypeError, ValueError, ZeroDivisionError):
        beat_len = 0.5
    bar = max(1e-6, beat_len * 4.0)
    gap = max(float(ARRANGEMENT_SPAN_GAP_S), beat_len * 8.0)
    min_len = bar * float(ARRANGEMENT_MIN_SPAN_BARS)
    cluster = bar * float(ARRANGEMENT_CLUSTER_BARS)

    events = []
    for stem, notes in (stem_notes or {}).items():
        if not notes:
            continue
        sources = []
        if stem == "drums":
            for kind in ("kick", "snare", "hihat", "tom", "cymbal", "percussion"):
                ts = []
                for n in notes:
                    try:
                        k = n.get("drum_type") if "drum_type" in n else n.get("k")
                        if k == kind:
                            ts.append(float(n.get("onset", n.get("t", 0.0))))
                    except Exception:
                        continue
                if ts:
                    sources.append((f"{stem}:{kind}", ts))
        else:
            ts = []
            for n in notes:
                try:
                    ts.append(float(n.get("onset", n.get("t", 0.0))))
                except Exception:
                    continue
            if ts:
                sources.append((stem, ts))

        for name, ts in sources:
            for a, b in _instrument_activity_spans(ts, gap, min_len):
                events.append((a, name, "in"))
                events.append((b, name, "out"))

    if not events:
        return []
    events.sort(key=lambda e: e[0])

    groups = []
    current = [events[0]]
    for e in events[1:]:
        if e[0] - current[-1][0] <= cluster:
            current.append(e)
        else:
            groups.append(current)
            current = [e]
    groups.append(current)

    out = []
    for g in groups:
        t = float(np.median([e[0] for e in g]))
        if t < 2.0:
            continue
        if total_duration and t > float(total_duration) - 2.0:
            continue
        stems = sorted({e[1].split(":")[0] for e in g})
        out.append({
            "time": round(t, 2),
            "stems": stems,
            "n_stems": len(stems),
            "events": sorted({f"{e[1]} {e[2]}" for e in g}),
        })
    return out


def build_arrangement_change_report(changes, max_rows=14):
    """Report arrangement entries/exits as evidence for the writer."""
    if not changes:
        return ""
    strong = [c for c in changes
              if c["n_stems"] >= int(ARRANGEMENT_STRONG_MIN_STEMS)]
    rows = strong or changes
    rows = sorted(rows, key=lambda c: (-c["n_stems"], c["time"]))[:max_rows]
    rows.sort(key=lambda c: c["time"])
    lines = [
        "ARRANGEMENT CHANGE POINTS (instruments entering or leaving, measured "
        "from the transcribed onsets of every separated stem; drums split by "
        "kit piece). These are the moments the arrangement actually changes, "
        "which is usually what a listener hears as a section change -- use them "
        "when asked when something comes in or drops out. A change moving "
        "several stems at once is a strong structural moment; a single stem "
        "moving alone is often just a phrase break:"
    ]
    for c in rows:
        lines.append(
            f"  {_seconds_to_mmss(c['time'])} ({c['n_stems']} stem"
            f"{'s' if c['n_stems'] != 1 else ''}): " + ", ".join(c["events"][:6])
        )
    return "\n".join(lines)


def _sharpen_boundary(t, dsp_bounds, downbeats, beats, beat_len, max_shift_s,
                      strong_changes=None, weak_changes=None):
    """Nudge one boundary onto the nearest piece of measured evidence.

    Preference order is by reliability:

      1. An ARRANGEMENT change moving several stems at once -- the snare
         arriving, the bass dropping out. This is what a listener hears as the
         section changing, and it is the one signal timbre novelty is worst at:
         a chorus entering on the same pad, key and tempo produces almost no
         spectral novelty while changing the arrangement completely.
      2. An acoustic novelty change point.
      3. A bar line, then a beat.
      4. A single-stem arrangement change -- ranked BELOW bar lines on purpose.
         Isolated entries and exits are mostly pad swells and transcription
         gaps; only corroborated ones are trustworthy.

    The move stays capped either way, so this only ever sharpens a boundary
    that is already essentially right -- it can never relocate it.
    """
    # A detected acoustic change point gets a wider tolerance than a bare bar
    # line. The model hears which sections exist but tends to report their
    # edges a second or two out; a measured change point within about a bar and
    # a half of its guess is almost certainly the same musical event, and it is
    # bar-accurate. A bare bar line carries no such evidence -- there is one
    # every couple of seconds -- so snapping to those stays tight.
    dsp_tol = max(max_shift_s, (beat_len * 6.0) if beat_len else 3.0)
    dsp_tol = float(min(dsp_tol, 4.0))

    # Arrangement changes are capped at ONE BAR, tighter than the acoustic
    # tolerance above. A section end and the next section's entry often bracket
    # the model's guess (vocals stop at 1:07, come back at 1:11), and with a
    # wide window the snap picks whichever happens to be marginally nearer --
    # arbitrary, and capable of moving a boundary further than the cap the rest
    # of this function respects. Within a bar there is only one candidate, so
    # the snap is decided by the music rather than by rounding.
    arr_tol = float(max_shift_s)

    best = None
    for grid, tol in (
        (strong_changes or [], arr_tol),
        (dsp_bounds, dsp_tol),
        (downbeats, min(max_shift_s, (beat_len * 1.2) if beat_len else max_shift_s)),
        (beats, min(max_shift_s, (beat_len * 0.5) if beat_len else max_shift_s)),
        (weak_changes or [], min(arr_tol, (beat_len * 2.0) if beat_len else arr_tol)),
    ):
        if not grid:
            continue
        arr = np.asarray([float(g) for g in grid], dtype=float)
        i = int(np.argmin(np.abs(arr - float(t))))
        if abs(arr[i] - float(t)) <= tol:
            best = float(arr[i])
            break
    return best if best is not None else float(t)


def _unify_labels_by_lyrics(sections, lyric_chunks, lyric_groups):
    """Make section NAMES agree with what the words actually do.

    Two sections delivering the same body of words are the same section
    returning, and should not carry different names. This only rewrites
    labels -- boundaries are left exactly where they were, because the words
    tell you what a section IS, not to the nearest second where it starts.
    """
    if not sections or not lyric_chunks:
        return sections

    blocks = _lyric_block_segmentation(lyric_chunks, lyric_groups)
    if not blocks:
        return sections

    def _base(name):
        return re.sub(r"\s*\d+\s*$", "", str(name or "")).strip()

    # Signature of the lyric block that dominates each section.
    sigs = []
    for s in sections:
        a, b = float(s["start"]), float(s["end"])
        best_sig, best_ov = None, 0.0
        for blk in blocks:
            ov = min(b, blk["end"]) - max(a, blk["start"])
            if ov > best_ov:
                best_ov, best_sig = ov, blk["signature"]
        if best_ov < 0.35 * max(1e-6, b - a):
            best_sig = None
        sigs.append(best_sig)

    # For each recurring signature, pick the base name used most often for it
    # and apply that name to every section sharing it.
    by_sig = {}
    for i, sig in enumerate(sigs):
        if sig:
            by_sig.setdefault(sig, []).append(i)

    out = [dict(s) for s in sections]
    changed = False
    for sig, idxs in by_sig.items():
        if len(idxs) < 2:
            continue
        names = [_base(sections[i]["name"]) for i in idxs]
        # Ignore placeholder names when deciding the consensus.
        real = [n for n in names if n and not re.match(r"(?i)section\b", n)]
        if not real:
            continue
        # Most-used name for this material wins. Ties are broken by FIRST
        # occurrence rather than by set iteration order -- the latter is
        # effectively arbitrary, so a genuine "Pre-Chorus" could lose a
        # coin-flip to a mislabelled "Chorus" and the whole track would come
        # back with every section called a chorus. First occurrence is both
        # deterministic and the better guess: the model labels material most
        # carefully the first time it hears it, before the song has settled
        # into a pattern it can pattern-match against.
        counts_by_name = {}
        for nm in real:
            counts_by_name[nm] = counts_by_name.get(nm, 0) + 1
        top = max(counts_by_name.values())
        tied = [nm for nm in real if counts_by_name[nm] == top]
        consensus = tied[0]
        for i in idxs:
            if _base(sections[i]["name"]) != consensus:
                out[i]["name"] = consensus
                changed = True

    if not changed:
        return sections

    # Renumber repeats so the map stays readable after any renaming.
    totals = {}
    for s in out:
        b = _base(s["name"])
        totals[b] = totals.get(b, 0) + 1
    seen = {}
    for s in out:
        b = _base(s["name"])
        if totals[b] > 1:
            seen[b] = seen.get(b, 0) + 1
            s["name"] = "%s %d" % (b, seen[b])
        else:
            s["name"] = b
    return out


_VOCAL_ROLE_NAME_RE = re.compile(r"(?i)\b(verse|chorus|pre-?\s?chorus|refrain|hook)\b")
_PRECHORUS_NAME_RE = re.compile(r"(?i)\bpre-?\s?chorus\b")


def _same_name_different_cluster_notes(sections):
    """{index: note} for sections whose shared base name is contradicted by the
    acoustic clustering.

    Observed: an instrumental mapped as "Verse 1" to "Verse 4" where the four
    spans were four different acoustic clusters (A, B, D, E) while the
    choruses correctly shared one. A name used three or more times on material
    that never once recurs is the labeller's guess, not a measured repeat.
    Two differing occurrences are left alone -- a stripped first verse and a
    full second verse legitimately differ.
    """
    groups = {}
    for i, sec in enumerate(sections or []):
        if not isinstance(sec, dict):
            continue
        name = str(sec.get("name") or "").strip()
        cl = str(sec.get("cluster") or "").strip()
        if not name or not cl:
            continue
        base = re.sub(r"\s*\d+\s*$", "", name).strip().lower()
        if base:
            groups.setdefault(base, []).append((i, cl, name))
    out = {}
    for base, items in groups.items():
        if len(items) < SAME_NAME_DISTINCT_CLUSTER_MIN:
            continue
        cls = [c for _i, c, _n in items]
        named = [c for c in cls if c.lower() != "unique"]
        distinct = len(set(named)) + (len(cls) - len(named))
        if distinct < len(items):
            continue
        label = re.sub(r"\s*\d+\s*$", "", items[0][2]).strip()
        first_i, first_c, first_name = items[0]
        out[first_i] = (
            f"the {len(items)} sections named \"{label}\" were measured as {len(items)} different "
            f"materials (clusters {', '.join(cls)}) -- none of them returns, so the shared name is "
            "the labeller's interpretation, not a measured repeat; describe them as distinct "
            f"parts and do not say the {label.lower()} comes back unchanged"
        )
        for i, c, _n in items[1:]:
            out[i] = (f"different material from the other \"{label}\" sections (cluster {c}) "
                      f"-- see the note on {first_name}")
    return out


def _annotate_section_name_checks(sections, stem_notes=None):
    """Flag section NAMES that conflict with what was measured.

    The names in the final map come from the listening model's own STRUCTURE
    field (or its labelling of DSP segments); only the timing passes through a
    sanity check. Observed failures: a track opening on "Pre-Chorus 1", and a
    vocal-free guitar outro labelled "Verse" after the singer had stopped.
    Names are left unchanged -- the check adds a "name_check" note that the
    structure block prints next to the section. Returns the list (copied).
    """
    out = [dict(s) for s in (sections or [])]
    if not out:
        return out
    spans = [(float(s["start"]), float(s["end"])) for s in out]
    cov = None
    try:
        cov = _vocal_coverage_by_span(stem_notes, spans) if stem_notes else None
    except Exception:
        cov = None
    sung_track = False
    if cov:
        total = sum(b - a for a, b in spans) or 1.0
        sung = sum(c * (b - a) for c, (a, b) in zip(cov, spans))
        sung_track = (sung / total) >= 0.15
    try:
        cluster_notes = _same_name_different_cluster_notes(out)
    except Exception:
        cluster_notes = {}
    for i, s in enumerate(out):
        notes = []
        name = str(s.get("name") or "")
        if i == 0 and _PRECHORUS_NAME_RE.search(name):
            notes.append(
                "a pre-chorus normally follows a verse, so as the opening section this name "
                "is unconfirmed -- it may be an intro or a verse"
            )
        if sung_track and cov is not None and _VOCAL_ROLE_NAME_RE.search(name) and cov[i] < 0.05:
            notes.append(
                "no vocal was measured in this span although the track is sung elsewhere, so "
                "it is more likely an instrumental break, solo or outro than a sung "
                f"{_VOCAL_ROLE_NAME_RE.search(name).group(1).lower()}"
            )
        if i in cluster_notes:
            notes.append(cluster_notes[i])
        if notes:
            s["name_check"] = "; ".join(notes)
        else:
            s.pop("name_check", None)
    return out


def _title_words(text):
    t = re.sub(r"\([^)]*\)|\[[^\]]*\]", " ", str(text or "").lower())
    t = re.sub(r"[\u2018\u2019'`]", "", t)
    return re.sub(r"[^a-z0-9]+", " ", t).split()


def _sx_title_hits(segs, bar_edges, lyric_chunks, title):
    """dev32: how many times each segment sings the song's title (two words
    or more, spaces ignored so "U.S.A." matches "USA"), counted in the lyric
    chunks centred in it. None for a short or one-word title."""
    tw = _title_words(title)
    key = "".join(tw)
    if len(tw) < 2 or len(key) < 6:
        return None
    out = []
    n = len(bar_edges) - 1
    for a, b in segs:
        t0, t1 = float(bar_edges[min(a, n)]), float(bar_edges[min(b, n)])
        txt = " ".join(str(c.get("text") or "") for c in lyric_chunks
                       if t0 <= (float(c.get("start", 0)) + float(c.get("end", 0))) / 2.0 < t1)
        out.append("".join(_title_words(txt)).count(key))
    return out


def _title_hook_naming(sections, lyric_chunks, title):
    """dev19 STRUCTURE_TITLE_HOOK: Verse and Chorus names swapped when the
    song's title is sung only in sections named Verse (two or more), in no
    section named Chorus, those sections repeat their words (Jaccard 0.4+ on
    average) and most of them sing the title more than once. On
    the dance-pop test track the namer, swayed by a few drum hits, called the
    "<title line>" sections verses and the verses choruses. A
    title of one word, or one sung where the lyrics do not repeat (a verse
    that opens on the title), changes nothing."""
    try:
        if not STRUCTURE_TITLE_HOOK or not sections or not lyric_chunks:
            return sections
        tw = _title_words(title)
        if len(tw) < 2:
            return sections
        phrase = " " + " ".join(tw) + " "
        words = {}
        for i, s in enumerate(sections):
            a, b = float(s["start"]), float(s["end"])
            txt = " ".join(str(c.get("text") or "") for c in lyric_chunks
                           if a <= (float(c.get("start", 0)) + float(c.get("end", 0))) / 2.0 < b)
            words[i] = " " + " ".join(_title_words(txt)) + " "
        base = lambda s: re.sub(r"\s*\d+\s*$", "", str(s.get("name") or "")).strip()
        fuzzy = bool(STRUCTURE_TITLE_HOOK_FUZZY and len(tw) >= 3)

        def _count(txt):
            # dev31: see STRUCTURE_TITLE_HOOK_FUZZY
            if not fuzzy:
                return txt.count(phrase)
            ws, k, c = txt.split(), len(tw), 0
            q = 0
            while q + k <= len(ws):
                if ws[q] == tw[0] or ws[q + k - 1] == tw[-1]:
                    if sum(1 for x_, y_ in zip(ws[q:q + k], tw) if x_ == y_) >= k - 1:
                        c += 1
                        q += k
                        continue
                q += 1
            return c
        title_i = [i for i in words if _count(words[i]) > 0]
        verse_i = [i for i, s in enumerate(sections) if base(s) == "Verse"]
        chorus_i = [i for i, s in enumerate(sections) if base(s) == "Chorus"]
        if (len(title_i) < 2 or not chorus_i or any(i not in verse_i for i in title_i)
                or any(i in title_i for i in chorus_i)):
            return sections

        def _sim(ix):
            sets = [set(words[i].split()) for i in ix if words[i].strip()]
            pairs = [(x, y) for k, x in enumerate(sets) for y in sets[k + 1:]]
            if not pairs:
                return 0.0
            return sum(len(x & y) / float(len(x | y) or 1) for x, y in pairs) / len(pairs)
        hits = sum(_count(words[i]) for i in title_i)
        if _sim(title_i) < 0.4 or hits < (len(title_i) if fuzzy else 2 * len(title_i) - 1):
            return sections
        out = [dict(x) for x in sections]
        n = {"Verse": 0, "Chorus": 0}
        for x in out:
            b_ = base(x)
            if b_ in n:
                nb_ = "Chorus" if b_ == "Verse" else "Verse"
                n[nb_] += 1
                x["name"] = f"{nb_} {n[nb_]}"
        return out
    except Exception:
        return sections


def _sx_bar_seconds(beats):
    """One 4/4 bar, in seconds, from the beat grid's median spacing, its
    tempo folded into 70-180 BPM (the grid can count double time). None
    without a usable grid."""
    try:
        b = [float(x) for x in (beats or [])]
        iv = sorted(b[i + 1] - b[i] for i in range(len(b) - 1) if b[i + 1] > b[i])
        if len(iv) < 8:
            return None
        bpm = 60.0 / iv[len(iv) // 2]
        while bpm > 180.0:
            bpm /= 2.0
        while bpm < 70.0:
            bpm *= 2.0
        return 4 * 60.0 / bpm
    except Exception:
        return None


def _sx_base_name(name):
    return re.sub(r"\s*\d+\s*$", "", str(name or "")).strip()


def _sx_renumber(secs):
    """Section names numbered again in order ("Chorus 1", "Chorus 2" ...), a
    name that occurs once left without a number -- the namer's convention."""
    counts = {}
    for s in secs:
        counts[_sx_base_name(s.get("name"))] = counts.get(_sx_base_name(s.get("name")), 0) + 1
    seen = {}
    for s in secs:
        b = _sx_base_name(s.get("name"))
        if b.lower() in ("silence",) or counts[b] == 1:
            s["name"] = b
            continue
        seen[b] = seen.get(b, 0) + 1
        s["name"] = f"{b} {seen[b]}"
    return secs


def _sx_merge_split_repeats(secs, beats):
    """dev20 STRUCTURE_MERGE_SPLIT_REPEATS: back-to-back sections of the same
    name and the same music (cluster), each at most STRUCTURE_MERGE_MAX_BARS
    bars, are one section when EVERY occurrence of that name is split that way
    -- a namer that cut each chorus in half, not a chorus followed by a tag
    or a coda (one split repeat among whole ones is left as it is). The duet
    test track's dev19 save: seven 15-second choruses in three runs, 8 sections in
    dev17."""
    try:
        if not STRUCTURE_MERGE_SPLIT_REPEATS or not secs or len(secs) < 3:
            return secs
        bar = _sx_bar_seconds(beats)
        if not bar:
            return secs
        runs, i = [], 0
        while i < len(secs):
            j = i
            while (j + 1 < len(secs) and _sx_base_name(secs[j + 1].get("name")) == _sx_base_name(secs[i].get("name"))
                   and str(secs[j + 1].get("cluster") or "") == str(secs[i].get("cluster") or "")
                   and str(secs[i].get("cluster") or "") not in ("", "Unique")):
                j += 1
            runs.append((i, j))
            i = j + 1
        by_name = {}
        for a, b in runs:
            by_name.setdefault(_sx_base_name(secs[a].get("name")), []).append((a, b))
        merge = set()
        for nm, rs in by_name.items():
            if nm.lower() in ("silence", "intro", "outro"):
                continue
            if not all(b > a for a, b in rs):
                continue                      # some occurrence is whole: leave every one alone
            if all(float(secs[k]["end"]) - float(secs[k]["start"]) <= STRUCTURE_MERGE_MAX_BARS * bar + 0.5
                   for a, b in rs for k in range(a, b + 1)):
                merge.update(rs)
        if not merge:
            return secs
        out = []
        for a, b in runs:
            if (a, b) in merge:
                m = dict(secs[a])
                m["end"] = secs[b]["end"]
                m.pop("name_check", None)
                out.append(m)
            else:
                out.extend(dict(x) for x in secs[a:b + 1])
        return _sx_renumber(out)
    except Exception:
        return secs


def _sx_bridge_to_prechorus(secs, beats):
    """dev20 STRUCTURE_BRIDGE_TO_PRECHORUS: a section named Bridge that sits
    directly between a Verse and a Chorus and lasts at most
    STRUCTURE_PRECHORUS_MAX_BARS bars is a Pre-Chorus (in British usage
    "bridge" means just that). the boy-band test track dev19: "Bridge 1" 1:14 and
    "Bridge 2" 2:07, each between a verse and a chorus. A bridge after a
    chorus or a break (the one-off contrast) is left alone."""
    try:
        if not STRUCTURE_BRIDGE_TO_PRECHORUS or not secs or len(secs) < 3:
            return secs
        bar = _sx_bar_seconds(beats)
        if not bar:
            return secs
        hit = []
        for i in range(1, len(secs) - 1):
            if _sx_base_name(secs[i].get("name")) != "Bridge":
                continue
            if (_sx_base_name(secs[i - 1].get("name")) == "Verse"
                    and _sx_base_name(secs[i + 1].get("name")) == "Chorus"
                    and float(secs[i]["end"]) - float(secs[i]["start"]) <= STRUCTURE_PRECHORUS_MAX_BARS * bar + 0.5):
                hit.append(i)
        if not hit:
            return secs
        if STRUCTURE_PRECHORUS_NEEDS_PATTERN and len(hit) < 2:
            # dev26: a lone Bridge between a verse and a chorus after two or
            # more choruses is the song's bridge leading back in, not a
            # pre-chorus (a pre-chorus comes round with the verses).
            i = hit[0]
            n_ch = sum(1 for x in secs[:i] if _sx_base_name(x.get("name")) == "Chorus")
            if n_ch >= 2:
                return secs
        out = [dict(x) for x in secs]
        for i in hit:
            out[i]["name"] = "Pre-Chorus"
        return _sx_renumber(out)
    except Exception:
        return secs


def _sx_regrid_to_tempo(beats, bpm, stem_notes=None):
    """dev20 STRUCTURE_REGRID_TO_TEMPO: (beats, note). When the beat grid runs
    at a non-octave ratio of the decided tempo (2:3, 3:4, 3:2, 4:3 within
    3%), its bars are a bar and a half (or three quarters) long and section
    lines cannot fall on the song's phrases: the teen-pop test track's grid ran at 97.5 BPM
    against 146. The grid is rebuilt at the tempo by splitting every p grid
    beats into q evenly spaced beats, starting on the grid beat (of the first
    p) that puts most kick and snare hits on the new beats. Octave ratios
    (half or double time) and grids that already agree are returned as they
    are; so is everything without a tempo."""
    try:
        if not STRUCTURE_REGRID_TO_TEMPO or not bpm or not beats or len(beats) < 16:
            return beats, ""
        b = np.asarray([float(x) for x in beats], dtype=float)
        g = 60.0 / float(np.median(np.diff(b)))
        r = g / float(bpm)
        pq = None
        for p_ in (2, 3, 4):
            for q_ in (2, 3, 4):
                if p_ == q_ or (p_ / q_) in (0.5, 2.0):
                    continue
                if abs(r - p_ / q_) / (p_ / q_) < 0.03:
                    pq = (p_, q_)
        if not pq and STRUCTURE_REGRID_OCTAVE_SLOW:
            # dev36: see STRUCTURE_REGRID_OCTAVE_SLOW -- a grid at half or a
            # quarter of the tempo: each grid beat split in two or four.
            for q_ in (2, 4):
                if abs(r - 1.0 / q_) * q_ < 0.03:
                    nb = []
                    for i in range(len(b) - 1):
                        nb += [b[i] + (b[i + 1] - b[i]) * k / q_ for k in range(q_)]
                    nb.append(b[-1])
                    return [float(x) for x in nb], (f"beat grid {g:.1f} BPM is 1:{q_} of the "
                                                    f"{float(bpm):.0f} BPM tempo; each beat split in {q_}")
        if not pq and STRUCTURE_REGRID_OCTAVE_FAST and abs(r - 2.0) / 2.0 < 0.03:
            # dev36: see STRUCTURE_REGRID_OCTAVE_FAST -- a grid at double the
            # tempo: every other grid beat, the set that lands more drum hits.
            pq = (2, 1)
        if not pq:
            return beats, ""
        p_, q_ = pq
        hits = np.asarray(sorted(float(t) for kind in ("kick", "snare")
                                 for t in (_sx_drum_hits(stem_notes, kind) if stem_notes else [])))
        best = None
        for o in range(p_):
            nb = []
            for i in range(o, len(b) - p_, p_):
                a, e = b[i], b[i + p_]
                nb += [a + (e - a) * k / q_ for k in range(q_)]
            if len(nb) < 16:
                continue
            nb = np.asarray(nb)
            if hits.size:
                j = np.searchsorted(nb, hits)
                d = np.minimum(np.abs(hits - nb[np.clip(j - 1, 0, len(nb) - 1)]),
                               np.abs(hits - nb[np.clip(j, 0, len(nb) - 1)]))
                sc = float(np.mean(d < 0.06))
            else:
                sc = 0.0
            if best is None or sc > best[0]:
                best = (sc, o, nb)
        if best is None:
            return beats, ""
        return [float(x) for x in best[2]], (f"beat grid {g:.1f} BPM is {p_}:{q_} of the {float(bpm):.0f} BPM tempo; "
                                             f"rebuilt at the tempo")
    except Exception:
        return beats, ""


# dev20 STRUCTURE_LYRIC_HEADERS: see the setting.
_HEADER_NAMES = (
    (r"pre[- ]?chorus|build[- ]?up|^build$|^lift$", "Pre-Chorus"),
    (r"post[- ]?chorus", "Post-Chorus"),
    (r"chorus|hook|refrain", "Chorus"),
    (r"verse", "Verse"),
    (r"bridge|middle 8|middle eight", "Bridge"),
    (r"intro", "Intro"),
    (r"outro|coda", "Outro"),
)


def _tag_header_name(marker):
    """'[Bridge: Singer Y & John Doe]' -> 'Bridge'; '' when the
    marker names no section ('(x2)', '[Instrumental]', '[Spoken]')."""
    t = re.sub(r"^[\[(]\s*|\s*[\])]$", "", str(marker or "").strip()).split(":")[0].strip().lower()
    for rx, nm in _HEADER_NAMES:
        if re.search(rx, t):
            return nm
    return ""


def _tag_stanza_headers(tag_lyrics):
    """{stanza index: section name} for lyrics that mark their sections
    ("[Verse 1]", "[Pre-Chorus]" ...), numbering stanzas exactly as
    _tag_lyric_lines does. {} when fewer than three stanzas are marked or the
    stanzas are collapsed into one.
    0.7.1-pre18 STRUCTURE_BARE_HEADER_IS_REPEAT: in lyrics whose first header sits
    directly above its lines, a header standing alone (a blank line after it) marks
    that part coming again and names nothing after it."""
    heads, stanza, started, pending = {}, 0, False, ""
    n_lines = {}
    bare_is_repeat = False
    if STRUCTURE_BARE_HEADER_IS_REPEAT:
        rows = [raw.strip() for raw in str(tag_lyrics or "").splitlines()]
        for i, ln in enumerate(rows):
            if ln and re.fullmatch(r"[\[(].*[\])]", ln) and _tag_header_name(ln):
                nxt = rows[i + 1] if i + 1 < len(rows) else ""
                bare_is_repeat = (bool(nxt) and not re.fullmatch(r"[\[(].*[\])]", nxt)
                                  and len(_lyric_norm_words(nxt)) >= 2)
                break
    for raw in str(tag_lyrics or "").splitlines():
        ln = raw.strip()
        if not ln or re.fullmatch(r"[\[(].*[\])]", ln):
            if started and (not ln or _TAG_SECTION_MARKER_RE.fullmatch(ln)):
                stanza += 1
                started = False
            elif not ln and pending and bare_is_repeat:
                pending = ""                 # 0.7.1-pre18: a header standing alone names nothing after it
            if ln and re.fullmatch(r"[\[(].*[\])]", ln):
                nm = _tag_header_name(ln)
                if nm:
                    pending = nm
            continue
        if len(_lyric_norm_words(ln)) >= 2:
            if not started and pending:
                heads[stanza] = pending
                pending = ""
            n_lines[stanza] = n_lines.get(stanza, 0) + 1
            started = True
    if n_lines and float(np.median(list(n_lines.values()))) < 2:
        return {}
    return heads if len(heads) >= 3 else {}


def _tag_header_events(lyric_chunks, heads, total_duration):
    """[(start, end, name)] for every timed tag line whose stanza(s) carry a
    header. A line found in stanzas of different names counts only when they
    are Intro and Outro (the same words opening and closing the song), decided
    by which half of the song it is sung in."""
    ev = []
    dur = float(total_duration or 0.0) or None
    for c in lyric_chunks or []:
        if not isinstance(c, dict) or not c.get("line_stanzas") or not c.get("lines"):
            continue
        try:
            a, b = float(c["start"]), float(c["end"])
            cuts = [int(x) for x in c["lines"]]
            nw = max(1, cuts[-1])
            for k, cand in enumerate(c["line_stanzas"]):
                if k + 1 >= len(cuts):
                    break
                names = set()
                for tup in cand or []:
                    st = tup.get("__tuple__", tup)[0] if isinstance(tup, dict) else tup[0]
                    if st in heads:
                        names.add(heads[st])
                    else:
                        names.add("")
                if "" in names or not names:
                    continue
                if len(names) > 1:
                    if names == {"Intro", "Outro"} and dur:
                        names = {"Intro" if (a + b) / 2.0 < dur / 2.0 else "Outro"}
                    else:
                        continue
                t0 = a + (b - a) * cuts[k] / nw
                t1 = a + (b - a) * cuts[k + 1] / nw
                if t1 > t0:
                    ev.append((t0, t1, names.pop()))
        except Exception:
            continue
    return sorted(ev)


def _hdr_agreement_cut(s, a, b, runs_in, votes, snap, cut_hints=None, first=False, last=False):
    """dev26 STRUCTURE_HEADER_SPLIT_ON_AGREEMENT: where to split section `s`
    ([a, b]) between two lyric-header names; None for no split, or "keep"
    when the section should keep its measured name.

    (1) Its timed header lines begin with the name it already has and then
        change to another name that holds the rest: the section is split at a
        measured section line (cut_hints) within STRUCTURE_HEADER_HINT_TOL_S of
        where the new name's first line begins. With no measured line there,
        the section keeps the name it has (the measured part and the first
        lines agree) instead of being renamed whole.
    (2) Otherwise, at a measured section line inside it: the header names on
        the two sides differ, or one side has (almost) no timed lines while
        the other side's lines name it something else.
    Intro / Outro header lines decide a cut only at the song's start / end.
    Each piece must be STRUCTURE_HEADER_SPLIT_MIN_S or longer."""
    mn = float(STRUCTURE_HEADER_SPLIT_MIN_S)
    cur = _sx_base_name(s.get("name"))
    hints = sorted(float(x) for x in (cut_hints or []))

    def _edge_ok(nm, side):
        if nm == "Intro":
            return first and side == "l"
        if nm == "Outro":
            return last and side == "r"
        return True
    rall = runs_in(a, b)
    if len(rall) >= 2 and rall[0][2] == cur:
        k = next((i for i in range(1, len(rall)) if rall[i][2] != cur), None)
        if k is not None and _edge_ok(rall[k][2], "r"):
            new = rall[k][2]
            rest = votes(rall[k][0], b)
            tot = sum(rest.values())
            if (tot > 0 and rest.get(new, 0.0) >= STRUCTURE_HEADER_MIN_SHARE * tot
                    and tot >= (b - rall[k][0]) / 3.0):
                t0 = float(rall[k][0])
                near = [h for h in hints if abs(h - t0) <= STRUCTURE_HEADER_HINT_TOL_S
                        and a + mn <= h <= b - mn]
                if near:
                    return min(near, key=lambda h: abs(h - t0))
                return "keep"
    best = None
    for h in hints:
        if not (a + mn <= h <= b - mn):
            continue
        vl, vr = votes(a, h), votes(h, b)
        tl, tr = sum(vl.values()), sum(vr.values())
        nl = max(vl.items(), key=lambda kv: kv[1])[0] if tl > 0 else None
        nr = max(vr.items(), key=lambda kv: kv[1])[0] if tr > 0 else None
        ok_l = (tl > 0 and vl[nl] >= STRUCTURE_HEADER_MIN_SHARE * tl and tl >= (h - a) / 3.0
                and _edge_ok(nl, "l"))
        ok_r = (tr > 0 and vr[nr] >= STRUCTURE_HEADER_MIN_SHARE * tr and tr >= (b - h) / 3.0
                and _edge_ok(nr, "r"))
        if ok_l and ok_r and nl != nr:
            sc = 2.0
        elif ok_r and tl <= 0.15 * (h - a) and nr != cur:
            sc = 1.0          # unmatched lead keeps its name; the rest is named by its lines
        elif ok_l and tr <= 0.15 * (b - h) and nl != cur:
            sc = 1.0
        else:
            continue
        if best is None or sc > best[0]:
            best = (sc, h)
    return best[1] if best else None


def _sx_lyric_header_naming(secs, lyric_chunks, tag_lyrics, total_duration, beats=None,
                           cut_hints=None, vocal_notes=None):
    """dev20 STRUCTURE_LYRIC_HEADERS: when the file's own lyrics mark their
    sections ("[Intro]", "[Pre-Chorus]", "[Bridge: Singer Y & John Doe]") and the transcribed words are matched to those lines, each sung
    section takes the name of the stanzas sung in it (at least
    STRUCTURE_HEADER_MIN_SHARE of its matched singing, and matched lines
    covering a third of the section). A section holding two of them one after
    the other, each for STRUCTURE_HEADER_SPLIT_MIN_S or more (a verse running
    into its pre-chorus), is split where the second begins, on the nearest
    beat. Sections then named alike back to back are joined, except two
    choruses. Unsung, unmatched or unmarked sections keep their names;
    lyrics without section markers change nothing."""
    try:
        if not STRUCTURE_LYRIC_HEADERS or not secs or not lyric_chunks or not tag_lyrics:
            return secs
        heads = _tag_stanza_headers(tag_lyrics)
        if not heads:
            return secs
        ev = _tag_header_events(lyric_chunks, heads, total_duration)
        if len(ev) < 6:
            return secs
        bt = sorted(float(x) for x in (beats or []))

        def snap(t):
            if not bt:
                return t
            j = int(np.searchsorted(bt, t))
            c = [bt[i] for i in (j - 1, j) if 0 <= i < len(bt)]
            return min(c, key=lambda x: abs(x - t)) if c else t

        def runs_in(a, b):
            rr = []
            for t0, t1, nm in ev:
                x0, x1 = max(a, t0), min(b, t1)
                if x1 - x0 <= 0:
                    continue
                if rr and rr[-1][2] == nm:
                    rr[-1][1] = x1
                else:
                    rr.append([x0, x1, nm])
            return rr

        def votes(a, b):
            v = {}
            for x0, x1, nm in runs_in(a, b):
                v[nm] = v.get(nm, 0.0) + (x1 - x0)
            return v
        # 1) a section holding two or more marked parts one after the other is split
        out = []
        _ns_idx = [i for i, x in enumerate(secs) if _sx_base_name(x.get("name")) != "Silence"]
        for _si, s in enumerate(secs):
            s = dict(s)
            a, b = float(s["start"]), float(s["end"])
            if _sx_base_name(s.get("name")) == "Silence":
                out.append(s)
                continue
            rr = [r for r in runs_in(a, b) if r[1] - r[0] >= STRUCTURE_HEADER_SPLIT_MIN_S]
            names_seq = [r[2] for r in rr]
            cuts = []
            if len(rr) >= 2 and all(names_seq[k] != names_seq[k + 1] for k in range(len(rr) - 1)):
                for r in rr[1:]:
                    c = snap(r[0])
                    lo = cuts[-1] if cuts else a
                    if c - lo >= STRUCTURE_HEADER_SPLIT_MIN_S and b - c >= STRUCTURE_HEADER_SPLIT_MIN_S:
                        cuts.append(c)
            if not cuts and STRUCTURE_HEADER_SPLIT_ON_AGREEMENT:
                # dev26: see STRUCTURE_HEADER_SPLIT_ON_AGREEMENT.
                try:
                    c = _hdr_agreement_cut(s, a, b, runs_in, votes, snap, cut_hints,
                                           first=bool(_ns_idx) and _si == _ns_idx[0],
                                           last=bool(_ns_idx) and _si == _ns_idx[-1])
                except Exception:
                    c = None
                if c == "keep":
                    s["_hdr_keep"] = True
                elif c is not None:
                    cuts = [c]
            if cuts:
                edges = [a] + cuts + [b]
                for k in range(len(edges) - 1):
                    x = dict(s)
                    x["start"], x["end"] = edges[k], edges[k + 1]
                    x.pop("name_check", None)
                    out.append(x)
            else:
                out.append(s)
        # 2) names from the marked parts sung in each section
        def _voiceless(s):
            # dev26: see STRUCTURE_HEADER_NEEDS_VOICE.
            if not (STRUCTURE_HEADER_NEEDS_VOICE and vocal_notes):
                return False
            if _sx_base_name(s.get("name")) not in ("Intro", "Outro", "Instrumental"):
                return False
            a_, b_ = float(s["start"]), float(s["end"])
            if b_ <= a_:
                return False
            try:
                return _sx_voiced_seconds(vocal_notes, a_, b_) < STRUCTURE_HEADER_VOICE_MIN_SHARE * (b_ - a_)
            except Exception:
                return False
        for s in out:
            if s.get("_hdr_keep"):
                s["_hdr"] = True      # dev26: named by its first lines (its measured name)
                continue
            if _sx_base_name(s.get("name")) == "Silence" or _voiceless(s):
                continue
            a, b = float(s["start"]), float(s["end"])
            v = votes(a, b)
            tot = sum(v.values())
            if tot > 0:
                nm, w = max(v.items(), key=lambda kv: kv[1])
                if w >= STRUCTURE_HEADER_MIN_SHARE * tot and tot >= (b - a) / 3.0:
                    if nm != _sx_base_name(s.get("name")):
                        s.pop("name_check", None)
                    s["name"] = nm
                    s["_hdr"] = True
        # 3) a boundary moves to where the next part's own lines begin (or the
        #    last part's lines end) when those lines sit on the wrong side of it
        moved = False
        # first every boundary that moves back to where the next part's lines
        # begin, then those that move on to where the last part's lines end
        for back in (True, False):
            for k in range(len(out) - 1):
                x, y = out[k], out[k + 1]
                nx, ny = _sx_base_name(x.get("name")), _sx_base_name(y.get("name"))
                if nx == ny or "Silence" in (nx, ny):
                    continue
                a, m, b = float(x["start"]), float(x["end"]), float(y["end"])
                rx = runs_in(a, m)
                ry = runs_in(m, b)
                nb_ = None
                if back and y.get("_hdr") and rx and rx[-1][2] == ny and len(rx) >= 2:
                    nb_ = snap(rx[-1][0])
                elif not back and x.get("_hdr") and ry and ry[0][2] == nx and len(ry) >= 2:
                    nb_ = snap(ry[0][1])
                if (nb_ is not None and nb_ - a >= STRUCTURE_HEADER_SPLIT_MIN_S
                        and b - nb_ >= STRUCTURE_HEADER_SPLIT_MIN_S):
                    x["end"], y["start"] = nb_, nb_
                    moved = True
        if moved:
            # the moved sections are named again from what they now hold
            for s in out:
                if _sx_base_name(s.get("name")) == "Silence" or _voiceless(s) or s.get("_hdr_keep"):
                    continue
                a, b = float(s["start"]), float(s["end"])
                v = votes(a, b)
                tot = sum(v.values())
                if tot > 0:
                    nm, w = max(v.items(), key=lambda kv: kv[1])
                    if w >= STRUCTURE_HEADER_MIN_SHARE * tot and tot >= (b - a) / 3.0:
                        if nm != _sx_base_name(s.get("name")):
                            s.pop("name_check", None)
                        s["name"] = nm
                        s["_hdr"] = True
        joined = []
        for s in out:
            if (joined and s.get("_hdr") and joined[-1].get("_hdr")
                    and _sx_base_name(s["name"]) == _sx_base_name(joined[-1]["name"])
                    and _sx_base_name(s["name"]) != "Chorus"):
                joined[-1]["end"] = s["end"]
                continue
            joined.append(s)
        for s in joined:
            s.pop("_hdr", None)
            s.pop("_hdr_keep", None)
        return _sx_renumber(joined)
    except Exception:
        return secs


def _sx_sheet_marks_verses(tag_lyrics):
    """dev27: True when the file's lyric sheet marks its verses or
    pre-choruses itself ("[Verse 1]", "[Pre-Chorus]"), so the lyric-header
    naming decides them."""
    try:
        heads = _tag_stanza_headers(tag_lyrics) if tag_lyrics else {}
        return any(v in ("Verse", "Pre-Chorus") for v in heads.values())
    except Exception:
        return False


def _sx_sheet_opens_intro(tag_lyrics):
    """dev30: True when the file's lyric sheet marks its sections and its first
    sung stanza is headed Intro ("[Intro: Singer X]")."""
    try:
        heads = _tag_stanza_headers(tag_lyrics) if tag_lyrics else {}
        return bool(heads) and min(heads) == 0 and heads[0] == "Intro"
    except Exception:
        return False


def _sx_transcript_unmatched(lyric_chunks, tag_lyrics):
    """dev26: True when the file carries a real lyric sheet (20 or more lines)
    but not one of 10 or more transcribed chunks was matched to its lines --
    the transcript is then mostly mishearing (vocables heard as words, another
    language), and its 'repeated words' would steer the naming."""
    try:
        n_tag = sum(1 for ln in str(tag_lyrics or "").splitlines()
                    if len(ln.split()) >= 2 and not ln.strip().startswith(("[", "(")))
        ch = [c for c in (lyric_chunks or []) if isinstance(c, dict)]
        return n_tag >= 20 and len(ch) >= 10 and not any(c.get("lines") for c in ch)
    except Exception:
        return False


_MOVEMENT_ROMAN = ("I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI", "XII")


def _title_movements(title):
    """dev26: the movement titles a suite's track title lists in order
    ("Suite: I. First Movement / II. Second Movement / ...",
    "1. Overture / 2. ..."), or [] when it does not list two or more numbered
    movements starting at I (or 1)."""
    t = str(title or "")
    for a_, b_ in (("\u2236", ":"), ("\u2024", "."), ("\u2044", "/"), ("\u2215", "/"),
                   ("\u2013", "-"), ("\u2014", "-"), ("\u2019", "'")):
        t = t.replace(a_, b_)
    num = r"(?:XII|XI|X|IX|VIII|VII|VI|V|IV|III|II|I|1[0-2]|[1-9])"
    rx = re.compile(r"(?:^|(?<=[\s:/|;,(\-]))(" + num + r")\.\s+(.+?)(?=\s*(?:[/|;]|$|\s" + num + r"\.\s))")
    found = [(m.group(1), m.group(2).strip(" -:,")) for m in rx.finditer(t)]
    if len(found) < 2:
        return []
    nums = [f[0] for f in found]
    if nums == list(_MOVEMENT_ROMAN[:len(nums)]) or nums == [str(i + 1) for i in range(len(nums))]:
        if all(f[1].split() for f in found):
            return [(_MOVEMENT_ROMAN[i], f[1]) for i, f in enumerate(found)]
    return []


def _movement_title_hits(mtitle, lyric_chunks):
    """Start times of the transcribed chunks that sing a movement's title as
    a phrase: some run of the chunk's words matches the title's words in order
    (difflib ratio 0.6 or more, e.g. "town of the dead" for "Town of the
    Damned"), sharing at least one word of four or more letters that is not
    a stopword."""
    tw = [w.strip("'") for w in re.sub(r"[^\w' ]+", " ", str(mtitle).lower()).split()]
    tw = [w for w in tw if w]
    content = set(w for w in tw if w not in _SX_LYRIC_STOPWORDS and len(w) >= 4)
    if not tw or not content:
        return []
    n = len(tw)
    hits = []
    for c in lyric_chunks or []:
        if not isinstance(c, dict):
            continue
        try:
            t0 = float(c.get("start"))
        except Exception:
            continue
        cw = [w.strip("'") for w in re.sub(r"[^\w' ]+", " ", str(c.get("text") or "").lower()).split()]
        cw = [w for w in cw if w]
        best = 0.0
        for L in (n - 1, n, n + 1):
            if L < 1:
                continue
            for i in range(0, max(1, len(cw) - L + 1)):
                win = cw[i:i + L]
                if not (set(win) & content):
                    continue
                r = difflib.SequenceMatcher(None, tw, win, autojunk=False).ratio()
                if r > best:
                    best = r
        if best >= 0.6:
            try:
                t1 = max(t0, float(c.get("end", t0)))
            except Exception:
                t1 = t0
            hits.append(min(t0 + 3.0, 0.5 * (t0 + t1)))   # inside the line, not on its timing edge
    return sorted(hits)


_SX_SUNG_ROLE_NAMES = ("Verse", "Chorus", "Pre-Chorus", "Post-Chorus", "Bridge")


def _sx_movement_numbering(secs, title, lyric_chunks, total_duration, remap=None, stem_notes_ok=None):
    """dev26 STRUCTURE_MOVEMENT_NUMBERING: a track whose title lists its
    movements (a suite: "I. ... / II. ... / III. ...") gets its sections
    numbered within each movement, named "Part II Verse 1" etc., with the
    movement's full title in each section's "movement" field. Each later
    movement is placed from where its title is sung: the last time it is
    sung before the next movement's title, walked back over the sung
    sections before it (and an instrumental lead-in on the same material);
    when a minute or more of unexplained music lies between the previous
    movement and that point, the movement starts at the first sung section
    of new material in it. Only when every movement after the first is found
    this way, in order; anything else (a single song, a title with one part
    number, an instrumental suite) is returned unchanged."""
    mv = _title_movements(title)
    dur = float(total_duration or (secs[-1]["end"] if secs else 0.0))
    if len(mv) < 2 or not secs or dur < float(STRUCTURE_MOVEMENT_MIN_TRACK_S):
        return secs
    hits = [_movement_title_hits(t_, lyric_chunks) for _r, t_ in mv]
    idx_of = lambda t: next((i for i, x in enumerate(secs) if float(x["start"]) - 1e-6 <= t < float(x["end"])),
                            len(secs) - 1)
    starts = [0]                       # section index where each movement starts
    anchor_idx = [0]
    prev_t = 0.0
    for k in range(1, len(mv)):
        nxt_first = next((h for h in (hits[k + 1] if k + 1 < len(hits) else []) if h > prev_t), None)
        cand = [h for h in hits[k] if h > prev_t and (nxt_first is None or h < nxt_first)]
        if not cand:
            return secs
        a_t = cand[-1]
        j = idx_of(a_t)
        lo = anchor_idx[-1]
        while j - 1 > lo and _sx_base_name(secs[j - 1].get("name")) in _SX_SUNG_ROLE_NAMES:
            j -= 1
        run_cl = set(str(x.get("cluster")) for x in secs[j:idx_of(a_t) + 1]) - {"Unique", "None", ""}
        if (j - 1 > lo and _sx_base_name(secs[j - 1].get("name")) == "Instrumental"
                and str(secs[j - 1].get("cluster")) in run_cl):
            j -= 1
        w0 = lo + 1
        if w0 < j and float(secs[j]["start"]) - float(secs[w0]["start"]) >= 60.0:
            for w in range(w0, j):
                cl = str(secs[w].get("cluster"))
                seen = set(str(x.get("cluster")) for x in secs[:w])
                if (_sx_base_name(secs[w].get("name")) in _SX_SUNG_ROLE_NAMES
                        and cl not in ("Unique", "None", "") and cl not in seen):
                    j = w
                    break
        if j <= starts[-1] or float(secs[j]["start"]) - float(secs[starts[-1]]["start"]) < 30.0:
            return secs
        starts.append(j)
        anchor_idx.append(max(j, idx_of(a_t)))
        prev_t = a_t
    out = [dict(x) for x in secs]
    bounds = starts + [len(out)]
    if remap is not None and STRUCTURE_MOVEMENT_REMAP:
        # Each movement measured and named again on its own, as the song it
        # is (its verses and choruses are its own, not the suite's); the
        # movement's own lines are kept, and a movement that cannot be
        # measured on its own keeps the first pass's sections.
        rebuilt = []
        for k in range(len(mv)):
            seg_k = out[bounds[k]:bounds[k + 1]]
            t0, t1 = float(seg_k[0]["start"]), float(seg_k[-1]["end"])
            lead = [x for x in seg_k if _sx_base_name(x.get("name")) == "Silence" and float(x["end"]) <= t0 + 1e-6]
            new_k = None
            try:
                new_k = remap(t0, t1)
            except Exception:
                new_k = None
            if new_k:
                new_k = [dict(x) for x in new_k
                         if not (_sx_base_name(x.get("name")) == "Silence" and float(x["end"]) <= t0 + 0.05)]
                new_k = [x for x in new_k if float(x["end"]) > t0 + 0.05 and float(x["start"]) < t1 - 0.05]
            if new_k and not any(_sx_base_name(x.get("name")) in ("Verse", "Chorus") for x in new_k):
                new_k = None          # nothing recurs inside it: the whole-track names say more
            if new_k:
                if k < len(mv) - 1:
                    # a movement's unsung tail leads into the next one: a link, not the song's outro
                    for x in new_k[::-1]:
                        if _sx_base_name(x.get("name")) != "Outro":
                            break
                        if stem_notes_ok is not None and not stem_notes_ok(float(x["start"]), float(x["end"])):
                            x["name"] = "Instrumental"
                new_k[0]["start"] = t0
                new_k[-1]["end"] = t1
                for x in new_k:
                    x["start"] = max(t0, float(x["start"]))
                    x["end"] = min(t1, float(x["end"]))
                    x["cluster"] = "%s%s" % (mv[k][0], str(x.get("cluster", "Unique"))) \
                        if str(x.get("cluster", "Unique")) != "Unique" else "Unique"
                rebuilt.append(lead + new_k)
            else:
                rebuilt.append(seg_k)
        out, bounds, pos = [], [], 0
        for part_k in rebuilt:
            bounds.append(pos)
            out.extend(part_k)
            pos += len(part_k)
        bounds.append(len(out))
    for k in range(len(mv)):
        part = [x for x in out[bounds[k]:bounds[k + 1]] if _sx_base_name(x.get("name")) != "Silence"]
        counts = {}
        for x in part:
            b_ = _sx_base_name(x.get("name"))
            counts[b_] = counts.get(b_, 0) + 1
        seen = {}
        for x in part:
            b_ = _sx_base_name(x.get("name"))
            seen[b_] = seen.get(b_, 0) + 1
            nm = b_ if counts[b_] == 1 else "%s %d" % (b_, seen[b_])
            x["name"] = "Part %s %s" % (mv[k][0], nm)
            x["movement"] = "%s. %s" % (mv[k][0], mv[k][1])
            x.pop("name_check", None)
    return out


def _sx_number_verse_halves(secs):
    """dev29 STRUCTURE_VERSE_HALVES_AS_PARTS: back-to-back sections named
    Verse (one verse the measurements kept in two halves, at a drum or vocal
    entry) are numbered as the parts of one verse -- "Verse 1 Part 1",
    "Verse 1 Part 2", then "Verse 2" for the next verse -- instead of "Verse 1",
    "Verse 2", "Verse 3". Times and every other name are unchanged."""
    rx = re.compile(r"^Verse(?: \d+)?$")
    idx = [i for i, x in enumerate(secs) if rx.match(str(x.get("name") or ""))]
    if not any(b == a + 1 for a, b in zip(idx, idx[1:])):
        return secs
    runs = []
    for i in idx:
        if runs and runs[-1][-1] == i - 1:
            runs[-1].append(i)
        else:
            runs.append([i])
    out = [dict(x) for x in secs]
    for k, run in enumerate(runs):
        head = "Verse" if len(runs) == 1 else "Verse %d" % (k + 1)
        for j, i in enumerate(run):
            out[i]["name"] = head if len(run) == 1 else "%s Part %d" % (head, j + 1)
            out[i].pop("name_check", None)
    return out


def _sx_split_verse_parts(secs):
    """dev28 STRUCTURE_VERSE_PARTS: a verse made of distinct melodic parts,
    one of which comes back later inside the same verse (A-B-C-B), is listed
    part by part ("Verse 1 Part 1", "Verse 1 Part 2", ...) at the measured
    lines between them; every other section is returned as it was. Lines
    within STRUCTURE_HEADER_SPLIT_MIN_S of the verse's ends are not used."""
    out = []
    for s in secs or []:
        sp = s.pop("subparts", None) if isinstance(s, dict) else None
        if not sp or not re.match(r"^(?:Part [IVX]+ )?Verse(?: \d+)?$", str(s.get("name") or "")):
            out.append(s)
            continue
        a, b = float(s["start"]), float(s["end"])
        mn = float(STRUCTURE_HEADER_SPLIT_MIN_S)
        cuts = [t for t in sorted(float(x) for x in sp) if a + mn <= t <= b - mn]
        cuts = [t for k, t in enumerate(cuts) if k == 0 or t - cuts[k - 1] >= mn]
        if len(cuts) < 2:
            out.append(s)
            continue
        edges = [a] + cuts + [b]
        for k in range(len(edges) - 1):
            x = dict(s)
            x["start"], x["end"] = edges[k], edges[k + 1]
            x["name"] = "%s Part %d" % (s["name"], k + 1)
            x.pop("name_check", None)
            out.append(x)
    for s in out:
        if isinstance(s, dict):
            s.pop("subparts", None)
    return out


def _sx_align_tag_fragments(lyric_chunks, tag_lyrics):
    """dev34 STRUCTURE_TAG_FRAGMENTS: transcribed lines that the scan-time
    matching against the file's own lyrics left unmatched, because Whisper
    cut them out of the middle of a lyric line or across two ("the night
    train" from "the night train keeps on rolling down the line"), are matched
    to a run of words anywhere in the lyrics instead of to whole lines. A
    match needs 4+ words heard, 0.8+ alike, and three or more matching words
    that are not vocables or short function words; it corrects misheard words
    one for one (as the scan-time matching does) and records which lyric
    lines -- and so which stanzas -- the words belong to. Lines already
    matched, and lines matching nothing, are left exactly as they are. Only
    when the scan-time matching accepted the lyrics (some line matched)."""
    import difflib
    chunks = list(lyric_chunks or [])
    if not chunks or not any(isinstance(c, dict) and c.get("lines") for c in chunks):
        return lyric_chunks
    tag = _tag_lyric_lines(tag_lyrics)
    lines = [w for w, _st in tag]
    if len(lines) < 2:
        return lyric_chunks
    first_of = {}
    for idx, (_w, st) in enumerate(tag):
        first_of.setdefault(st, idx)
    by_text = {}
    for idx, (w, st) in enumerate(tag):
        by_text.setdefault(" ".join(w), []).append((st, first_of[st] == idx, idx))
    flat, owner = [], []
    for idx, w in enumerate(lines):
        for q, x in enumerate(w):
            flat.append(x)
            owner.append((idx, q))
    weak = _SX_VOCABLES | _SX_ENGLISH_FUNCTION_WORDS

    def _lids(c, which):
        # tag line ids the scan-time matching gave a chunk's first / last line
        try:
            st = c.get("line_stanzas") or []
            st = [x for x in st if x]
            if not st:
                return set()
            cand = st[0] if which == "first" else st[-1]
            return {int((t.get("__tuple__", t) if isinstance(t, dict) else t)[2]) for t in cand}
        except Exception:
            return set()
    matched_ix = [k for k, c in enumerate(chunks) if isinstance(c, dict) and c.get("lines")]
    out, changed = [], False
    for ci, c in enumerate(chunks):
        if not isinstance(c, dict) or c.get("lines"):
            out.append(c)
            continue
        cw = _lyric_norm_words(c.get("text", ""))
        if len(cw) < 4 or sum(1 for w in cw if w not in weak) < 2:
            out.append(c)
            continue
        best, best_span = 0.0, None
        sm = difflib.SequenceMatcher(None, cw, [], autojunk=False)
        for L in (len(cw) - 1, len(cw), len(cw) + 1):
            if L < 3:
                continue
            for a in range(0, len(flat) - L + 1):
                sm.set_seq2(flat[a:a + L])
                if sm.real_quick_ratio() <= best or sm.quick_ratio() <= best:
                    continue
                r = sm.ratio()
                if r > best:
                    best, best_span = r, (a, a + L)
        if best < float(STRUCTURE_TAG_FRAGMENT_MIN) or best_span is None:
            out.append(c)
            continue
        a, b = best_span
        ww = flat[a:b]
        sm.set_seq2(ww)
        same = [cw[i] for blk in sm.get_matching_blocks() for i in range(blk.a, blk.a + blk.size)]
        if sum(1 for w in same if w not in weak) < 3:
            out.append(c)
            continue
        fixed = []
        for op, i1, i2, j1, j2 in sm.get_opcodes():
            if op == "replace" and (i2 - i1) == (j2 - j1):
                fixed.extend(ww[j1:j2])
            else:
                fixed.extend(cw[i1:i2])
        starts, ids, mid = [], [], set()
        for q in range(a, b):
            lid, pos = owner[q]
            if q == a or pos == 0:
                starts.append(q - a)
                ids.append(lid)
                if pos != 0:
                    mid.add(lid)
        segs = _tag_line_cuts(fixed, ww, starts, ids)
        d = dict(c)
        d["text"] = " ".join(fixed)
        if STRUCTURE_TAG_FRAGMENT_MODE == "text":
            segs = []
        elif STRUCTURE_TAG_FRAGMENT_MODE == "starts":
            segs = [(x, l) for x, l in segs if l not in mid]
        elif STRUCTURE_TAG_FRAGMENT_MODE == "ordered" and segs:
            # In sequence with the matched lines around it: it continues the
            # line before it, follows it, or leads straight into the next.
            prv = [k for k in matched_ix if k < ci]
            nxt = [k for k in matched_ix if k > ci]
            p_l = _lids(chunks[prv[-1]], "last") if prv else set()
            n_f = _lids(chunks[nxt[0]], "first") if nxt else set()
            f_l, l_l = segs[0][1], segs[-1][1]
            ok = (any(f_l in (x, x + 1) for x in p_l)
                  and (not n_f or any(y in (l_l, l_l + 1) for y in n_f)))
            if not ok:
                segs = []
        if segs:
            cuts_ = [x for x, _l in segs]
            stz_ = []
            for _x, lid in segs:
                cands = by_text.get(" ".join(lines[lid]), [(tag[lid][1], first_of[tag[lid][1]] == lid, lid)])
                if lid in mid:
                    # The words start inside this line, so its stanza does not start here.
                    cands = [(st, False, li) for st, _f, li in cands]
                stz_.append(cands)
            if cuts_[0] > 0:
                cuts_ = [0] + cuts_
                stz_ = [[]] + stz_
            d["lines"] = cuts_ + [len(fixed)]
            d["line_stanzas"] = stz_
        out.append(d)
        changed = True
    return out if changed else lyric_chunks


def _sx_fill_missed_tag_lines(lyric_chunks, tag_lyrics, vocal_notes=None):
    """dev34 STRUCTURE_TAG_FILL_MISSED: lines of the file's own lyrics that
    Whisper did not hear, between two lines it did hear that the lyrics put
    one to three lines apart, are added where they must have been sung: in
    the gap between those two lines, spread over the stretches where the
    vocal stem is singing. Only when the gap is singing for a fifth of its
    length or more, is no longer than 7 s per missing line, and holds nothing
    else Whisper heard (that may be the line, misheard). The added lines are
    used by the section map only."""
    chunks = [c for c in (lyric_chunks or []) if isinstance(c, dict)]
    if not chunks or not any(c.get("lines") for c in chunks):
        return lyric_chunks
    tag = _tag_lyric_lines(tag_lyrics)
    lines = [w for w, _st in tag]
    if len(lines) < 3:
        return lyric_chunks
    first_of = {}
    for idx, (_w, st) in enumerate(tag):
        first_of.setdefault(st, idx)
    by_text = {}
    for idx, (w, st) in enumerate(tag):
        by_text.setdefault(" ".join(w), []).append((st, first_of[st] == idx, idx))

    def _T(x):
        return x.get("__tuple__", x) if isinstance(x, dict) else x
    seq = []
    for c in chunks:
        cuts, stz = c.get("lines"), c.get("line_stanzas")
        if not cuts or not stz or len(cuts) != len(stz) + 1:
            continue
        try:
            a, b = float(c["start"]), float(c["end"])
        except Exception:
            continue
        n = float(max(1, cuts[-1]))
        for k, cand in enumerate(stz):
            if cand:
                seq.append((a + (b - a) * cuts[k] / n, a + (b - a) * cuts[k + 1] / n,
                            {int(_T(x)[2]) for x in cand}))
    seq.sort(key=lambda x: x[0])
    heard = []
    for c in chunks:
        if c.get("lines"):
            continue
        try:
            m = (float(c["start"]) + float(c["end"])) / 2.0
        except Exception:
            continue
        if len(_lyric_norm_words(c.get("text", ""))) >= 2:
            heard.append(m)
    added = []
    for (a0, b0, A), (a1, b1, B) in zip(seq, seq[1:]):
        pair = None
        for x in sorted(A):
            for y in sorted(B):
                if 2 <= y - x <= 4 and (pair is None or y - x < pair[1] - pair[0]):
                    pair = (x, y)
        if pair is None:
            continue
        miss = list(range(pair[0] + 1, pair[1]))
        g0, g1 = b0, a1
        if g1 - g0 < 1.5 or g1 - g0 > 7.0 * len(miss):
            continue
        if any(g0 < m < g1 for m in heard):
            continue
        place = _sx_sung_time_mapper(vocal_notes, g0, g1)
        if place is None:
            continue
        nw = [len(lines[q]) for q in miss]
        tot = float(sum(nw)) or 1.0
        acc = 0
        for q, w in zip(miss, nw):
            t0, t1 = place(acc / tot), place((acc + w) / tot)
            acc += w
            if t1 <= t0:
                continue
            txt = " ".join(lines[q])
            added.append({"start": float(t0), "end": float(t1), "text": txt,
                          "lines": [0, len(lines[q])],
                          "line_stanzas": [by_text.get(txt, [(tag[q][1], first_of[tag[q][1]] == q, q)])],
                          "filled": True})
    if not added:
        return lyric_chunks
    return sorted(chunks + added, key=lambda c: float(c.get("start", 0.0)))


# dev33: see STRUCTURE_SPLIT_STOCK_PHRASES.
_SX_STOCK_PHRASES = [
    re.compile(r"\bwe'?ll be right back\b[.!]*", re.IGNORECASE),
    re.compile(r"\bguitar solo\b[.!]*", re.IGNORECASE),
    re.compile(r"\btranscript(?:ion)?:? emily beynon\b[.!]*", re.IGNORECASE),
    re.compile(r"\bthank(?:s| you)(?: (?:so|very) much)?,? for watching\b[.!]*", re.IGNORECASE),
    re.compile(r"\bsee you (?:(?:in|on) (?:the )?)?next (?:time|video|week|one)\b[.!]*", re.IGNORECASE),
]


def _sx_drop_stock_phrase_chunks(lyric_chunks, lyric_groups=None):
    """dev33 STRUCTURE_SPLIT_STOCK_PHRASES: lyric lines that, alone or with up
    to two neighbouring lines (each starting within 30 s of the last), make up
    nothing but one of Whisper's stock video phrases are dropped from the
    lines the section map uses -- the transcription's own filter checks one
    line at a time, so "We'll" / "be right back." split over two lines got
    through as sung words. Repetition groups made of such a phrase go too.
    Only whole lines are dropped; a line that also has other words is kept."""
    items = list(lyric_chunks or [])
    n = len(items)

    def _t(c):
        return re.sub(r"\s+", " ", str((c or {}).get("text") or "")).strip() if isinstance(c, dict) else ""

    def _st(c):
        try:
            return float(c.get("start"))
        except Exception:
            return None
    drop = set()
    for i in range(n):
        if i in drop or not _t(items[i]):
            continue
        for L in (1, 2, 3):
            js = list(range(i, i + L))
            if js[-1] >= n or any(j in drop or not _t(items[j]) for j in js):
                break
            if L > 1:
                a_, b_ = _st(items[js[-2]]), _st(items[js[-1]])
                if a_ is None or b_ is None or not (0.0 <= b_ - a_ <= 30.0):
                    break
            txt = " ".join(_t(items[j]) for j in js)
            rest = txt
            for pat in _SX_STOCK_PHRASES:
                rest = pat.sub(" ", rest)
            if rest != txt and not re.search(r"[^\W_]", rest):
                drop.update(js)
                break
    if not drop:
        return lyric_chunks, lyric_groups
    chunks = [c for j, c in enumerate(items) if j not in drop]
    groups = lyric_groups
    if lyric_groups:
        groups = []
        for g in lyric_groups:
            gt = str((g or {}).get("text") or "") if isinstance(g, dict) else ""
            rest = gt
            for pat in _SX_STOCK_PHRASES:
                rest = pat.sub(" ", rest)
            if gt and rest != gt and not re.search(r"[^\W_]", rest):
                continue
            groups.append(g)
    return chunks, groups


# dev33: see STRUCTURE_FOREIGN_CHUNKS. Distinctly Welsh words (none an English
# word, nor a common French / German / Spanish one): Whisper is well known to
# answer music it cannot make out with fluent-looking Welsh.
_SX_WELSH_WORDS = frozenset(
    "yn yr mae ddim chi nhw fy sydd wedi cael gael bod fod roedd mewn gyda hynny ffordd gwneud "
    "byddwch byddwn gadewch sut neu iawn llawer ychydig ddiogel ymlaen amser bwysig unig fynd "
    "ddangos cyntaf cychwyn canfod llefydd holl llwyddo glir ychwanegwch dŵr ysg i'w mae'n "
    "dydw dwi rydw rydych ydych oedd hefyd rhaid eisiau gwybod gweld meddwl rhywbeth popeth "
    "nawr heddiw yma acw".split())


# A line with as many of these as Welsh words is kept (real lyrics run into
# the hallucination).
_SX_ENGLISH_FUNCTION_WORDS = frozenset(
    "the you i'm i you're and to me my it it's is are was be been that this what when where "
    "why how not don't can't won't for with your of on in at all just like so but if we they "
    "he she her his them our us there here now know love oh no yeah get got go gonna wanna "
    "do does did have has had will would could should one time never ever say said".split())


def _sx_script_of(ch):
    o = ord(ch)
    if o < 0x250:
        return "latin"
    if 0x370 <= o < 0x400:
        return "greek"
    if 0x400 <= o < 0x530:
        return "cyrillic"
    if 0x590 <= o < 0x600:
        return "hebrew"
    if 0x600 <= o < 0x700:
        return "arabic"
    if 0x1780 <= o < 0x1800:
        return "khmer"
    if 0x3040 <= o < 0xA000 or 0xAC00 <= o < 0xD7B0:
        return "cjk"
    return "other"


def _sx_drop_foreign_chunks(lyric_chunks, lyric_groups=None):
    """dev33 STRUCTURE_FOREIGN_CHUNKS: lyric lines that are Whisper's foreign-
    language hallucinations are dropped from the lines the section map reads:
    lines a quarter or more of whose words (three or more) are distinctly
    Welsh, and more of them than English function words, in a song that is
    not mostly Welsh; and, in a song written mostly in
    Latin letters (80%+), lines mostly in another script (Khmer strings,
    Japanese credit lines). A song sung in Hebrew keeps its Hebrew lines."""
    items = list(lyric_chunks or [])

    def _t(c):
        return str((c or {}).get("text") or "") if isinstance(c, dict) else ""

    def _words(t):
        return [w.strip(".,!?;:\"()").lower() for w in t.split() if w.strip(".,!?;:\"()")]
    allw = [w for c in items for w in _words(_t(c))]
    if not allw:
        return lyric_chunks, lyric_groups
    welsh_song = sum(1 for w in allw if w in _SX_WELSH_WORDS) >= 0.5 * len(allw)
    letters = {}
    for c in items:
        for ch in _t(c):
            if ch.isalpha():
                k = _sx_script_of(ch)
                letters[k] = letters.get(k, 0) + 1
    nlet = sum(letters.values()) or 1
    latin_song = letters.get("latin", 0) >= 0.8 * nlet

    def _foreign(t):
        ws = _words(t)
        if not welsh_song and len(ws) >= 3:
            nw = sum(1 for w in ws if w in _SX_WELSH_WORDS or re.search(r"'(?:r|n|ch|w)$|^'(?:r|n)$", w))
            ne = sum(1 for w in ws if w in _SX_ENGLISH_FUNCTION_WORDS)
            if nw >= 0.25 * len(ws) and nw > ne:
                return True
        if latin_song:
            sc = [_sx_script_of(ch) for ch in t if ch.isalpha()]
            if sc and sum(1 for x in sc if x != "latin") > 0.5 * len(sc):
                return True
        return False
    keep = [c for c in items if not _foreign(_t(c))]
    if len(keep) == len(items):
        return lyric_chunks, lyric_groups
    groups = [g for g in (lyric_groups or []) if not (isinstance(g, dict) and _foreign(str(g.get("text") or "")))]
    return keep, (groups if lyric_groups is not None else lyric_groups)


def _structure_stem_evidence(stem_notes):
    if not STRUCTURE_KEEP_LEGACY_FALLBACK_HITS or not stem_notes:
        return stem_notes
    hits = stem_notes.get("drums") or []
    if not any("structure_legacy_keep" in n for n in hits):
        return stem_notes  # Omnizart and older scans are unchanged.
    # Audio-detected cymbals have no flag and remain in the structure evidence.
    return dict(stem_notes, drums=[n for n in hits if n.get("structure_legacy_keep", True)])


def _sx_role_labels_only(sections, lyric_chunks=None, tag_lyrics=None, instrumental=False):
    """Conservative role corrections on FIXED boundaries. Never split or merge.
    Explicit matched lyric headers take priority. Optional recurrence repairs
    only propagate an existing pre-chorus identity to closely matching words
    in the same Verse -> candidate -> Chorus position; never guess a bridge
    merely because a section is late, unique, louder or short.
    """
    if instrumental or not sections or not lyric_chunks:
        return sections
    out = [dict(sec) for sec in sections]
    locked = set()
    changes = False
    if STRUCTURE_ALIGNED_ROLE_LABELS and tag_lyrics:
        heads = _tag_stanza_headers(tag_lyrics)
        events = _tag_header_events(lyric_chunks, heads, sections[-1].get("end")) if heads else []
        for i, sec in enumerate(out):
            a, b = float(sec["start"]), float(sec["end"])
            if b <= a:
                continue
            # Union overlapping evidence intervals, so duplicate lines do not
            # inflate either support or confidence.
            spans = {}
            for x, y, name in events:
                lo, hi = max(a, x), min(b, y)
                if hi > lo:
                    spans.setdefault(name, []).append((lo, hi))
            mass = {}
            for name, intervals in spans.items():
                merged = []
                for lo, hi in sorted(intervals):
                    if merged and lo <= merged[-1][1]:
                        merged[-1][1] = max(hi, merged[-1][1])
                    else:
                        merged.append([lo, hi])
                mass[name] = sum(hi-lo for lo, hi in merged)
            if not mass:
                continue
            name = max(mass, key=mass.get)
            support = mass[name]
            if support < .4 * (b-a) or support < .85 * sum(mass.values()):
                continue
            locked.add(i)  # Do not override a reliable tag with recurrence.
            old = _sx_base_name(sec.get("name"))
            if (old in ("Verse", "Chorus", "Pre-Chorus", "Post-Chorus", "Bridge")
                    and name in ("Pre-Chorus", "Bridge", "Verse", "Chorus", "Post-Chorus")
                    and old != name):
                sec["name"] = name
                sec["role_label_reason"] = "strongly aligned lyric section header (boundaries retained)"
                changes = True
    if STRUCTURE_RECURRENT_ROLE_LABELS or STRUCTURE_RETURNING_PRECHORUS_LABELS:
        words = []
        for sec in out:
            tokens = []
            for c in lyric_chunks:
                try:
                    start, end = float(c["start"]), float(c["end"])
                    # Keep only lines mainly contained in this section.
                    overlap = min(float(sec["end"]), end)-max(float(sec["start"]), start)
                    if overlap <= 0 or overlap < .7 * max(end-start, .01):
                        continue
                    if "UNVERIFIED" in str(c.get("text", "")):
                        continue
                    tokens.extend(re.findall(r"[^\W\d_]+", str(c.get("text", "")).lower()))
                except (TypeError, ValueError, KeyError):
                    continue
            words.append(set(zip(tokens, tokens[1:], tokens[2:])))
        bases = [_sx_base_name(x.get("name")) for x in out]
        def slot(i):
            return (0 < i < len(out)-1 and bases[i-1] == "Verse" and bases[i+1] == "Chorus"
                    and 4 <= float(out[i]["end"])-float(out[i]["start"]) <= 32)
        anchors = [i for i, name in enumerate(bases) if name == "Pre-Chorus" and slot(i) and len(words[i]) >= 6]
        # A later pre-chorus may follow a break, without a new verse.
        # Require two existing anchors, matching duration and near-identical text.
        import difflib
        def token_line(i):
            tokens = []
            for c in lyric_chunks:
                try:
                    a, b = float(c["start"]), float(c["end"])
                    ov = min(float(out[i]["end"]), b) - max(float(out[i]["start"]), a)
                    if ov >= .6 * max(b-a, .01) and ov > 0 and "UNVERIFIED" not in str(c.get("text", "")):
                        tokens.extend(re.findall(r"[^\W\d_]+", str(c.get("text", "")).lower()))
                except (TypeError, ValueError, KeyError):
                    pass
            return tokens
        for i, name in enumerate(bases):
            if (not STRUCTURE_RETURNING_PRECHORUS_LABELS or i in locked or name not in ("Verse", "Bridge") or not 0 < i < len(out)-1
                    or bases[i-1] != "Instrumental" or bases[i+1] != "Chorus"):
                continue
            mine = token_line(i)
            dur = float(out[i]["end"]) - float(out[i]["start"])
            matches = []
            for j in anchors:
                ref = token_line(j)
                rd = float(out[j]["end"]) - float(out[j]["start"])
                ratio = difflib.SequenceMatcher(None, mine, ref, autojunk=False).ratio()
                if len(mine) >= 12 and len(ref) >= 12 and .75 * rd <= dur <= 1.35 * rd and ratio >= .8:
                    matches.append(ratio)
            if len(matches) >= 2 and sum(matches)/len(matches) >= .85:
                # A repeated chorus hook alone must not become a pre-chorus.
                other = [token_line(j) for j, role in enumerate(bases) if role == "Chorus"]
                if any(difflib.SequenceMatcher(None, mine, x, autojunk=False).ratio() >= .65 for x in other):
                    continue
                out[i]["name"] = "Pre-Chorus"
                out[i].pop("name_check", None)
                out[i]["role_label_reason"] = "two matching pre-chorus passages; return after instrumental; boundaries retained"
                locked.add(i)
                changes = True
        for i, name in enumerate(bases):
            if not STRUCTURE_RECURRENT_ROLE_LABELS or i in locked or name not in ("Verse", "Bridge") or not slot(i) or len(words[i]) < 6:
                continue
            for j in anchors:
                shared = len(words[i] & words[j])
                # Symmetric coverage prevents a short common hook/substring
                # from renaming a full verse or bridge.
                if shared >= 6 and shared >= .8 * max(len(words[i]),len(words[j])):
                    adjacent = words[i-1] | words[i+1]
                    if len(words[i] & adjacent) >= .3 * len(words[i]):
                        continue
                    out[i]["name"] = "Pre-Chorus"
                    out[i]["role_label_reason"] = "repeated pre-chorus words and matching structural position"
                    changes = True
                    break
    if not changes:
        return sections
    out = _sx_renumber(out)
    assert [(x["start"], x["end"]) for x in out] == [(x["start"], x["end"]) for x in sections]
    return out


def _sx_neutral_vocal_roles(sections):
    """Remove unsupported vocal-form names on the final, fixed map."""
    out = [dict(x) for x in (sections or [])]
    changed = False
    for sec in out:
        if _sx_base_name(sec.get("name")) in ("Verse", "Chorus", "Pre-Chorus", "Post-Chorus"):
            sec["name"] = "Section"
            sec.pop("name_check", None)
            sec.pop("role_label_reason", None)
            sec["role_label_reason"] = "neutral role: human vocal presence is not established"
            changed = True
    if changed:
        neutral = [x for x in out if x.get("role_label_reason") == "neutral role: human vocal presence is not established"]
        existing = {x.get("name") for x in out if x not in neutral}
        number = 1
        for sec in neutral:
            while f"Section {number}" in existing:
                number += 1
            sec["name"] = f"Section {number}"
            number += 1
        return out
    return sections


# ---------------------------------------------------------------------------
# pre59: SongFormer, Stage 2 -- SongFormer's sections check Musiclyse's own map.
# Measured on human-timed songs before it was switched on: every rule below
# found more of the real section changes and named more of the song correctly,
# and none made a song worse. The guards come from that test: SongFormer's
# pre-chorus calls and its section splits inside one part (every few bars,
# same name) were the ones that went wrong, so neither is used.
# ---------------------------------------------------------------------------
_SF_SNAP_MAX_BARS = 1.0           # rule 1: furthest a boundary may move
_SF_STRONG = 0.9                  # rule 1: SongFormer's boundary strength for a longer move
_SF_STRONG_MAX_BARS = 3.0         # rule 1: furthest a boundary may move to a strong one
_SF_LATE_START = 0.7              # rule 2: SongFormer's strength to move a too-early start later
_SF_SURE = 0.9                    # pre61 rule 2: SongFormer's strength for a "sure" boundary
_SF_SURE_EARLY_BARS = 4.5         # pre61 rule 2: furthest a too-late start may move earlier to one
_SF_MIN_SNAPPED_S = 4.0           # rule 1: no section shorter than this after a move
_SF_ADD_CLEAR_S = 3.0             # rule 2: no Musiclyse boundary this close already
_SF_ADD_MIN_SECTION_S = 8.0       # rule 2: neither part of a split shorter than this
_SF_NAME_SHARE = 0.6              # rules 3-4: SongFormer's share of a part's time for its vote
_SF_OUTRO_WINDOW_S = 15.0         # rule 3: an Outro must end this close to the music's end
_SF_CHORUS_WORDS_SHARE = 0.3      # rules 2-3: a Chorus shares this much of its words with another part
_SF_OWN_SHARE = 0.5               # rule 3: SongFormer's share of a part's own time for a group's name
_SF_UNSUNG_SHARE = 0.2            # rule 3: a "sung" part with a voice under this share of it,
                                  # heard by SongFormer as instrumental, is named Instrumental
_SF_LABEL_NAME = {"intro": "Intro", "verse": "Verse", "chorus": "Chorus", "bridge": "Bridge",
                  "inst": "Instrumental", "outro": "Outro"}
_SF_SUNG = {"verse", "chorus", "pre-chorus", "post-chorus", "bridge"}
_SF_INSTRUMENTAL = {"instrumental", "inst", "solo", "break", "interlude"}
_SF_DANCE_WORDS = ("house", "techno", "trance", "drum n bass", "drum and bass", "dubstep", "electro",
                   "breakbeat", "breaks", "hardstyle", "hardcore", "jungle", "garage", "big beat",
                   "edm", "eurodance", "electro house", "progressive", "gabber", "happy hardcore")
_SF_LOCK = "_sf_locked_name"      # working mark: a name the title line settled (removed again)


def _sf_kind(label):
    """A section or SongFormer label as one of: intro verse pre-chorus chorus
    post-chorus bridge inst outro silence section (lower case)."""
    b = re.sub(r"\s+part\s*\d*\s*$", "", str(label or "").strip().lower())   # "Verse 2 Part 1"
    b = _sx_base_name(b)
    b = {"prechorus": "pre-chorus", "instrumental": "inst", "solo": "inst", "break": "inst",
         "interlude": "inst", "coda": "outro"}.get(b, b)
    return b


def _sf_family(kind):
    """Rule 2's notion of a real change: a pre-chorus counts as verse, and an
    instrumental part and an outro as the same thing."""
    return {"pre-chorus": "verse", "outro": "inst"}.get(kind, kind)


def _sf_sung_class(kind):
    if kind in _SF_SUNG:
        return "sung"
    if kind == "inst":
        return "inst"
    return "neutral"


def _sf_section_words(sec, lyric_chunks):
    """The set of words sung in a section (transcribed lines by their middle)."""
    a, b = float(sec["start"]), float(sec["end"])
    txt = " ".join(str(c.get("text") or "") for c in lyric_chunks or [] if isinstance(c, dict)
                   and a <= (float(c.get("start", 0)) + float(c.get("end", 0))) / 2.0 < b)
    return set(re.findall(r"[a-z']+", txt.lower()))


def _sf_words_repeat(secs, i, lyric_chunks, share=None):
    """True when section i's sung words come back in another section (word-set
    overlap of _SF_CHORUS_WORDS_SHARE or more) -- what makes a chorus a chorus --
    or when too few of its words were heard to tell."""
    share = _SF_CHORUS_WORDS_SHARE if share is None else share
    mine = _sf_section_words(secs[i], lyric_chunks)
    if len(mine) < 4:
        return True
    for j, other in enumerate(secs):
        if j == i:
            continue
        theirs = _sf_section_words(other, lyric_chunks)
        if theirs and len(mine & theirs) >= share * len(mine | theirs):
            return True
    return False


def _sf_has_sung_names(secs):
    """True when a map names any sung part (verse, chorus, bridge ...)."""
    return any(_sf_kind(x.get("name")) in _SF_SUNG for x in secs or [])


def _songformer_is_dance(discogs_styles):
    """True when the strongest Discogs style is a dance/electronic one."""
    try:
        top = (discogs_styles or [])[0]
        style = str(top[0] if isinstance(top, (list, tuple)) else top).lower()
    except Exception:
        return False
    style = style.split("---", 1)[-1]
    return any(word in style for word in _SF_DANCE_WORDS)


def _songformer_clean_sections(sf_sections, total_duration=None):
    """SongFormer's sections as [(start, end, kind)], sorted; [] when unusable."""
    out = []
    for sec in sf_sections or []:
        try:
            a, b = float(sec["start"]), float(sec["end"])
        except Exception:
            continue
        if b - a <= 0.05:
            continue
        out.append((a, b, _sf_kind(sec.get("label"))))
    out.sort()
    if len(out) < 2:
        return []
    if total_duration:
        # A result for a different recording (another length) is not used.
        if abs(out[-1][1] - float(total_duration)) > max(3.0, 0.02 * float(total_duration)):
            return []
    return out


def _songformer_vote(sf, a, b):
    """{kind: seconds} of SongFormer's labels over [a, b], silence left out."""
    w = {}
    for s0, s1, k in sf:
        o = min(b, s1) - max(a, s0)
        if o > 0 and k != "silence":
            w[k] = w.get(k, 0.0) + o
    return w


def _sf_form_order(names):
    """The order of a song's sung parts (verse, pre-chorus, chorus, bridge, outro),
    repeats in a row counted once -- to compare a map with the tag headers."""
    out = []
    for n in names:
        k = _sf_kind(n)
        k = "chorus" if k == "post-chorus" else k
        if k in ("verse", "pre-chorus", "chorus", "bridge", "outro") and (not out or out[-1] != k):
            out.append(k)
    return out


def _sf_order_distance(a, b):
    """How many parts must change for one order to become the other (edit distance)."""
    d = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        prev = d[:]
        d[0] = i
        for j in range(1, len(b) + 1):
            d[j] = min(prev[j] + 1, d[j - 1] + 1, prev[j - 1] + (a[i - 1] != b[j - 1]))
    return d[len(b)]


def _sf_own_words(out, i, lyric_chunks):
    """0.7.1-pre23 SONGFORMER_KEEPS_DISTINCT_PRECHORUS / _BRIDGE: True when part i of the map has
    words of its own: at least three content words, of which no more than
    STRUCTURE_PRECHORUS_DISTINCT_MAX are also sung in the map's choruses."""
    try:
        a, b = float(out[i]["start"]), float(out[i]["end"])
        pw = _sx_cvw_words(lyric_chunks, a, b)
        if len(pw) < 3:
            return False
        cw = set()
        for k, x in enumerate(out):
            if k != i and _sf_kind(x.get("name")) == "chorus":
                cw |= _sx_cvw_words(lyric_chunks, float(x["start"]), float(x["end"]))
        return len(pw & cw) / len(pw) <= float(STRUCTURE_PRECHORUS_DISTINCT_MAX)
    except Exception:
        return False


def _songformer_correct_sections(secs, sf_sections, *, beats=None, bpm=None, lyric_chunks=None,
                                 dp_starts=None, total_duration=None, instrumental=False,
                                 dance=False, tag_lyrics=None, snap=None, add=None, names=None,
                                 vocal_notes=None, part_starts=None, sure=None):
    """pre59 rules 1-3 on Musiclyse's settled map (see SONGFORMER_CORRECT_STRUCTURE);
    snap/add/names (None: the settings) choose the rules. Returns the map
    unchanged (the same list) when no rule changes anything; never changes the
    list it is given and never raises."""
    snap = SONGFORMER_SNAP_TIMES if snap is None else bool(snap)
    add = SONGFORMER_ADD_SECTIONS if add is None else bool(add)
    names = SONGFORMER_NAME_VOTE if names is None else bool(names)
    sure_adds = SONGFORMER_SURE_ADDS if sure is None else bool(sure)        # pre61
    original = secs
    try:
        if not secs or not sf_sections:
            return original
        total = float(total_duration or secs[-1]["end"])
        sf = _songformer_clean_sections(sf_sections, total)
        if not sf:
            return original
        out = [dict(x) for x in secs]
        if not _sf_has_sung_names(secs):
            instrumental = True     # no sung part named: treated as instrumental (adds, names off)
        bar = _sx_bar_seconds(beats) if beats else None
        if not bar and bpm:
            try:
                b_ = float(bpm)
                while b_ > 180.0:
                    b_ /= 2.0
                while b_ < 70.0:
                    b_ *= 2.0
                bar = 240.0 / b_
            except Exception:
                bar = None
        bar = float(bar or 2.0)
        strength = {}
        for x in sf_sections or []:
            try:
                if x.get("boundary_strength") is not None:
                    strength[round(float(x["start"]), 2)] = float(x["boundary_strength"])
            except Exception:
                continue

        def strong(t_):
            # Files from before the helper recorded strength have none: never strong.
            return strength.get(round(float(t_), 2), 0.0) >= _SF_STRONG
        sf_bounds = [(sf[i][0], sf[i - 1][2], sf[i][2]) for i in range(1, len(sf))
                     if 1.5 < sf[i][0] < total - 1.5 and "silence" not in (sf[i][2], sf[i - 1][2])]
        lines = []
        for c in lyric_chunks or []:
            try:
                lines.append((float(c.get("start")), float(c.get("end"))))
            except Exception:
                continue

        def inside_line(t):
            return any(a + 0.3 < t < b - 0.3 for a, b in lines)
        changed_times = changed_names = False

        # (1) a boundary both found: SongFormer's time.
        if snap and sf_bounds:
            used = set()
            for i in range(1, len(out)):
                if "silence" in (_sf_kind(out[i]["name"]), _sf_kind(out[i - 1]["name"])):
                    continue
                t = float(out[i]["start"])
                near = [(abs(b[0] - t), j) for j, b in enumerate(sf_bounds)
                        if j not in used and (
                            abs(b[0] - t) <= _SF_SNAP_MAX_BARS * bar
                            # a strong SongFormer boundary, the nearest one, up to 3 bars away
                            or (SONGFORMER_STRONG_MOVES and strong(b[0])
                                and abs(b[0] - t) <= _SF_STRONG_MAX_BARS * bar
                                and not any(abs(o[0] - t) < abs(b[0] - t) for o in sf_bounds if o is not b)))]
                if not near:
                    continue
                _d, j = min(near)
                nt = round(float(sf_bounds[j][0]), 2)
                if abs(nt - t) < 0.01:
                    used.add(j)
                    continue
                if inside_line(nt) and not inside_line(t):
                    continue
                if nt - float(out[i - 1]["start"]) < _SF_MIN_SNAPPED_S or float(out[i]["end"]) - nt < _SF_MIN_SNAPPED_S:
                    continue
                used.add(j)
                if abs(nt - t) > _SF_SNAP_MAX_BARS * bar:
                    # a deliberate longer move: the boundary-keeping check below allows it
                    out[i]["songformer_moved_from"] = list(out[i].get("songformer_moved_from") or []) + [t]
                out[i - 1]["end"] = nt
                out[i]["start"] = nt
                changed_times = True

        # (2) a missed section change Musiclyse's own evidence also supports.
        if add and not instrumental and sf_bounds and (not dance or sure_adds):
            evidence = [float(x) for x in (dp_starts or [])] + [a for a, _b in lines]
            for t, before, after in sf_bounds:
                # pre61 SONGFORMER_SURE_ADDS: a boundary SongFormer is sure of
                sure = sure_adds and strength.get(round(float(t), 2), 0.0) >= _SF_SURE
                if dance and not sure:
                    continue
                if _sf_family(before) == _sf_family(after) and not sure:
                    continue
                if any(abs(float(x["start"]) - t) <= _SF_ADD_CLEAR_S for x in out):
                    continue

                no_evidence = not sure and not any(abs(e - t) <= bar for e in evidence)
                if no_evidence and strength.get(round(float(t), 2), 0.0) < _SF_LATE_START:
                    continue
                k = max((i for i, x in enumerate(out) if float(x["start"]) < t), default=None)
                if k is None:
                    continue
                host = out[k]
                host_kind = _sf_kind(host.get("name"))
                if host_kind == "silence" or float(host["end"]) <= t:
                    continue
                too_early = t - float(host["start"]) < _SF_ADD_MIN_SECTION_S
                if not too_early and no_evidence:
                    continue            # without Musiclyse's own evidence, only the move below
                if (sure and SONGFORMER_SURE_EARLY_STARTS and float(host["end"]) - t < _SF_ADD_MIN_SECTION_S
                        and k + 1 < len(out)):
                    # pre61: the part after a sure boundary would be too short to split off,
                    # and Musiclyse's next part -- at most 4 bars later -- is the kind
                    # SongFormer hears from it: that part starts at the boundary.
                    nxt = out[k + 1]
                    old_ = float(host["end"])
                    if (_sf_kind(nxt.get("name")) == after and "silence" not in (before, after)
                            and old_ - t <= _SF_SURE_EARLY_BARS * bar + 0.01
                            and t - float(host["start"]) >= _SF_MIN_SNAPPED_S
                            and not (inside_line(t) and not inside_line(old_))):
                        t2 = round(float(t), 2)
                        nxt["songformer_moved_from"] = list(nxt.get("songformer_moved_from") or []) + [old_]
                        host["end"] = t2
                        nxt["start"] = t2
                        changed_times = True
                        continue
                if too_early or float(host["end"]) - t < _SF_ADD_MIN_SECTION_S:
                    # Too close to split off. But where SongFormer is sure this part starts
                    # at t and hears the few seconds before t as more of the previous
                    # section, Musiclyse started this part too early: its start moves to t.
                    # (Only later: SongFormer's "it starts earlier" proved unreliable -- it
                    # takes lead-ins and pre-choruses for the start of the next part.)
                    if (not too_early or not SONGFORMER_LATE_STARTS
                            or strength.get(round(float(t), 2), 0.0) < _SF_LATE_START
                            or "silence" in (before, after) or k == 0):
                        continue
                    a_ = float(host["start"])
                    sliver = _songformer_vote(sf, a_, t)
                    tot_ = sum(sliver.values())
                    if t - a_ < 1.0 or not tot_:
                        continue
                    s_kind, s_w = max(sliver.items(), key=lambda kv: kv[1])
                    prev = out[k - 1]
                    if (s_w < _SF_NAME_SHARE * tot_ or _sf_kind(prev.get("name")) != s_kind
                            or _sf_kind(host.get("name")) != after):
                        continue
                    if inside_line(t) and not inside_line(a_):
                        continue
                    if (float(host["end"]) - t < _SF_MIN_SNAPPED_S
                            or float(prev["end"]) - float(prev["start"]) < _SF_MIN_SNAPPED_S):
                        continue
                    t2 = round(float(t), 2)
                    host["songformer_moved_from"] = list(host.get("songformer_moved_from") or []) + [a_]
                    prev["end"] = t2
                    host["start"] = t2
                    changed_times = True
                    continue
                new_kind = "verse" if after == "pre-chorus" else after
                new_name = _SF_LABEL_NAME.get(new_kind)
                if not new_name:
                    continue
                if (_sf_family(new_kind) == _sf_family(host_kind)
                        and any(abs(float(c) - t) <= _SF_ADD_CLEAR_S for c in (part_starts or []))):
                    # More of the same part next to a boundary Musiclyse will list anyway
                    # (a verse part): it adds nothing and would split one verse in two.
                    continue
                if {_sf_sung_class(host_kind), _sf_sung_class(new_kind)} == {"sung", "inst"}:
                    continue            # sung or instrumental is Musiclyse's call (its vocal stem)
                t = round(float(t), 2)
                piece = {key: val for key, val in host.items()
                         if key not in ("name_check", "role_label_reason", "cluster", _SF_LOCK)}
                piece.update(start=t, name=new_name, cluster="", songformer="added",
                             _sf_host_name=host.get("name"))
                host["end"] = t
                out.insert(k + 1, piece)
                changed_times = True

        # An added section named Chorus whose sung words come back nowhere else is
        # not a chorus: it keeps the name of the part it was split from.
        for i, x in enumerate(out):
            host_name = x.pop("_sf_host_name", None)
            if (lines and host_name and x.get("songformer") == "added"
                    and _sf_kind(x.get("name")) == "chorus" and not _sf_words_repeat(out, i, lyric_chunks)):
                x["name"] = host_name

        # (3) SongFormer votes on names: per group of the same music, below the
        # title line; with tag headers, refereed by the headers' order of parts.
        heads = {}
        if tag_lyrics:
            try:
                heads = _tag_stanza_headers(tag_lyrics)
            except Exception:
                heads = {}
        heads_order = _sf_form_order([heads[k] for k in sorted(heads)]) if heads else None
        before_names = [x.get("name") for x in out]
        if names and not instrumental:
            music = [i for i, x in enumerate(out) if _sf_kind(x.get("name")) != "silence"]
            first_music = music[0] if music else None
            music_end = float(out[music[-1]]["end"]) if music else total
            # A part named as sung that SongFormer, over that part itself, hears as
            # instrumental, and where Musiclyse's own vocal stem has almost no
            # voice, is instrumental -- whatever the rest of its group is.
            if vocal_notes:
                for i in music:
                    x = out[i]
                    a_, z_ = float(x["start"]), float(x["end"])
                    if (_sf_kind(x.get("name")) not in _SF_SUNG or x.get(_SF_LOCK)
                            or z_ - a_ < _SF_ADD_MIN_SECTION_S):
                        continue
                    own = _songformer_vote(sf, a_, z_)
                    tot = sum(own.values())
                    if not tot:
                        continue
                    k_, w_ = max(own.items(), key=lambda kv: kv[1])
                    if (k_ in ("inst", "intro") and w_ >= _SF_NAME_SHARE * tot
                            and _sx_voiced_seconds(vocal_notes, a_, z_) < _SF_UNSUNG_SHARE * (z_ - a_)):
                        x["name"] = "Intro" if (k_ == "intro" and i == first_music) else "Instrumental"
                        x.pop("name_check", None)
                        x["songformer"] = "renamed"
                        changed_names = True
            groups = {}
            for i, x in enumerate(out):
                cl = str(x.get("cluster") or "")
                alone = i in (first_music, music[-1] if music else None)   # the opening and ending vote alone
                key = cl if cl and cl != "Unique" and not alone else ("one", i)
                groups.setdefault(key, []).append(i)
            for key, members in groups.items():
                votes = {}
                for i in members:
                    for kind, sec_ in _songformer_vote(sf, float(out[i]["start"]), float(out[i]["end"])).items():
                        votes[kind] = votes.get(kind, 0.0) + sec_
                if not votes:
                    continue
                win, weight = max(votes.items(), key=lambda kv: kv[1])
                if weight < _SF_NAME_SHARE * sum(votes.values()):
                    continue
                if win == "pre-chorus" or win not in _SF_LABEL_NAME:
                    continue                # SongFormer's pre-chorus calls are not used
                for i in members:
                    x = out[i]
                    cur = _sf_kind(x.get("name"))
                    if x.get(_SF_LOCK) or cur in ("silence", "section") or cur == win:
                        continue
                    if cur == "outro" and i == (music[-1] if music else None):
                        continue            # the song's last part keeps Musiclyse's Outro
                    if cur not in _SF_SUNG and cur not in ("inst", "intro", "outro"):
                        continue            # names outside the song-form set stay
                    if cur == "pre-chorus" and win == "verse":
                        continue            # SongFormer marks no pre-choruses
                    if (win == "chorus" and ((cur == "pre-chorus" and SONGFORMER_KEEPS_DISTINCT_PRECHORUS)
                                             or (cur == "bridge" and SONGFORMER_KEEPS_DISTINCT_BRIDGE))
                            and _sf_own_words(out, i, lyric_chunks)):
                        continue            # 0.7.1-pre23: a part with words of its own keeps its name
                    if cur == "post-chorus" and SONGFORMER_KEEP_POST_CHORUS:
                        continue            # 0.7.1-pre18: SongFormer has no post-chorus name
                    if (cur in _SF_SUNG and win in ("inst", "intro") and vocal_notes
                            and float(x["end"]) - float(x["start"]) >= _SF_ADD_MIN_SECTION_S):
                        # A "sung" part SongFormer hears as instrumental, where
                        # Musiclyse's own vocal stem also has almost no voice.
                        a_, z_ = float(x["start"]), float(x["end"])
                        if _sx_voiced_seconds(vocal_notes, a_, z_) < _SF_UNSUNG_SHARE * (z_ - a_):
                            x["name"] = "Intro" if (win == "intro" and i == first_music) else "Instrumental"
                            x.pop("name_check", None)
                            x["songformer"] = "renamed"
                            changed_names = True
                            continue
                    if (cur == "inst") != (win in ("inst", "outro", "intro")) and "inst" in (cur, win):
                        continue            # sung or instrumental is Musiclyse's call
                    if _sf_family(cur) == _sf_family(win):
                        continue            # an instrumental part and an outro: the same idea
                    if win == "bridge" and len(members) > 1:
                        continue            # a bridge is music that doesn't come back
                    own = _songformer_vote(sf, float(x["start"]), float(x["end"]))
                    if not own or own.get(win, 0.0) < _SF_OWN_SHARE * sum(own.values()):
                        continue            # SongFormer must say so over this part itself
                    if win == "chorus" and lines and not _sf_words_repeat(out, i, lyric_chunks):
                        continue            # a chorus repeats its words; this part's don't come back
                    if win == "intro" and i != first_music:
                        continue
                    if win == "outro" and float(x["end"]) < music_end - _SF_OUTRO_WINDOW_S:
                        continue
                    x["name"] = _SF_LABEL_NAME[win]
                    x.pop("name_check", None)
                    x["songformer"] = "renamed"
                    changed_names = True
        if heads_order is not None and changed_names:
            # The song's tag headers referee: SongFormer's names are kept only if the
            # map then follows the headers' order of parts more closely than before.
            if (_sf_order_distance(_sf_form_order([x.get("name") for x in out]), heads_order)
                    >= _sf_order_distance(_sf_form_order(before_names), heads_order)):
                for x, nm in zip(out, before_names):
                    if x.get("songformer") == "renamed" and nm is not None:
                        x["name"] = nm
                        x.pop("songformer", None)
                changed_names = any(x.get("songformer") == "renamed" for x in out)
        # A rename that leaves two neighbours with the same name where SongFormer
        # hears one section -- no SongFormer boundary within a bar of the join, and
        # one SongFormer section of that name spanning both, edge to edge within a
        # bar: Musiclyse's boundary there split one part in two, so they are joined.
        if changed_names:
            # SongFormer boundaries it is unsure of (strength under 0.5) between two parts
            # of the same kind don't count as a split here.
            weak = lambda u: strength.get(round(float(u), 2), 1.0) < 0.5
            sf_starts = [float(sb[0]) for sb in sf_bounds if not (sb[1] == sb[2] and weak(sb[0]))]
            sf_j = []
            for s0, s1, k_ in sf:
                if sf_j and sf_j[-1][2] == k_ and weak(s0):
                    sf_j[-1] = (sf_j[-1][0], s1, k_)
                else:
                    sf_j.append((s0, s1, k_))
            k = 1
            while k < len(out):
                p_, q_ = out[k - 1], out[k]
                kind = _sf_kind(p_.get("name"))
                t = float(q_["start"])
                if (kind == _sf_kind(q_.get("name")) and kind not in ("silence", "section")
                        and "renamed" in (p_.get("songformer"), q_.get("songformer"))
                        and not any(abs(u - t) <= bar for u in sf_starts)
                        and any(k_ == kind and abs(s0 - float(p_["start"])) <= 1.5 * bar
                                and abs(s1 - float(q_["end"])) <= 1.5 * bar for s0, s1, k_ in sf_j)):
                    p_["end"] = q_["end"]
                    p_["songformer_joined"] = list(p_.get("songformer_joined") or []) + [t]
                    p_["songformer"] = "renamed"
                    p_.pop("name_check", None)
                    del out[k]
                    continue
                k += 1
        # A song has one outro: a renamed Outro next to another Outro joins it.
        if changed_names:
            k = 1
            while k < len(out):
                p_, q_ = out[k - 1], out[k]
                if (_sf_kind(p_.get("name")) == "outro" == _sf_kind(q_.get("name"))
                        and "renamed" in (p_.get("songformer"), q_.get("songformer"))):
                    p_["end"] = q_["end"]
                    p_["songformer_joined"] = list(p_.get("songformer_joined") or []) + [float(q_["start"])]
                    p_["songformer"] = "renamed"
                    p_.pop("name_check", None)
                    del out[k]
                    continue
                k += 1
        if not (changed_times or changed_names):
            return original
        for x in out:
            x.pop(_SF_LOCK, None)
        if changed_names or any(x.get("songformer") == "added" for x in out):
            out = _sx_renumber(out)
        return out
    except Exception as e:
        print(f"  (SongFormer check skipped: {type(e).__name__}: {e})")
        return original


def _sf_parts_view(secs, tag_lyrics):
    """The map after the verse-part steps that follow the SongFormer check (they
    add boundaries: a verse listed part by part), on a copy."""
    import copy as _copy
    out = _copy.deepcopy(list(secs or []))
    if STRUCTURE_VERSE_HALVES_AS_PARTS and not _sx_sheet_marks_verses(tag_lyrics):
        try:
            out = _sx_number_verse_halves(out)
        except Exception:
            pass
    if STRUCTURE_VERSE_PARTS:
        out = _sx_split_verse_parts(out)
    return out


def _songformer_check_map(secs, sf_sections, *, tag_lyrics=None, **kw):
    """pre59: the SongFormer rules on the settled map, kept only if every boundary
    Musiclyse would have listed (verse parts included) is still there, moved at
    most a bar; otherwise the name vote, then the added sections, are left out
    until none is lost. Never changes the list it is given; never raises."""
    try:
        base = _sf_parts_view(secs, tag_lyrics)
        want = [float(x["start"]) for x in base[1:]]
        kw = dict(kw, part_starts=want)
        bar = (_sx_bar_seconds(kw.get("beats")) if kw.get("beats") else None) or 2.0
        snap = bool(SONGFORMER_SNAP_TIMES)
        # pre61: first with SongFormer's sure boundaries, then -- if that loses one of
        # Musiclyse's boundaries -- exactly as pre60, before anything else is left out.
        tries = [(snap, SONGFORMER_ADD_SECTIONS, SONGFORMER_NAME_VOTE, bool(SONGFORMER_SURE_ADDS)),
                 (snap, SONGFORMER_ADD_SECTIONS, SONGFORMER_NAME_VOTE, False),
                 (snap, SONGFORMER_ADD_SECTIONS, False, False),
                 (snap, False, False, False)]
        seen = set()
        for rules in tries:
            if rules in seen or not any(rules[:3]):
                continue
            seen.add(rules)
            cand = _songformer_correct_sections(secs, sf_sections, tag_lyrics=tag_lyrics,
                                                snap=rules[0], add=rules[1], names=rules[2],
                                                sure=rules[3], **kw)
            if cand is secs:
                return secs
            got = [float(x["start"]) for x in _sf_parts_view(cand, tag_lyrics)[1:]]
            joined = [float(t) for x in cand for t in
                      list(x.get("songformer_joined") or []) + list(x.get("songformer_moved_from") or [])]
            if all(any(abs(g - t) <= bar + 0.05 for g in got) or any(abs(j - t) <= bar + 0.05 for j in joined)
                   for t in want):
                return cand
        return secs
    except Exception as e:
        print(f"  (SongFormer check skipped: {type(e).__name__}: {e})")
        return secs


def _songformer_instrumental_names(sections, sf_sections, total_duration=None):
    """pre59 rule 4 (Option A): an instrumental's first and last musical parts,
    still neutral ("Section N"), are named Intro and Outro when SongFormer agrees
    over most of them. Nothing else is renamed; the exports already say which
    parts are the same music."""
    try:
        if not sections or not sf_sections:
            return sections
        total = float(total_duration or sections[-1]["end"])
        sf = _songformer_clean_sections(sf_sections, total)
        if not sf:
            return sections
        music = [i for i, x in enumerate(sections) if _sf_kind(x.get("name")) != "silence"]
        if len(music) < 3:
            return sections
        out = [dict(x) for x in sections]
        changed = False
        for i, want in ((music[0], "intro"), (music[-1], "outro")):
            if _sf_kind(out[i].get("name")) != "section":
                continue
            votes = _songformer_vote(sf, float(out[i]["start"]), float(out[i]["end"]))
            if votes and votes.get(want, 0.0) >= _SF_NAME_SHARE * sum(votes.values()):
                out[i]["name"] = _SF_LABEL_NAME[want]
                out[i].pop("role_label_reason", None)
                out[i]["songformer"] = "renamed"
                changed = True
        if not changed:
            return sections
        n = 0
        for x in out:
            if _sf_kind(x.get("name")) == "section":
                n += 1
                x["name"] = f"Section {n}"
        return out
    except Exception as e:
        print(f"  (SongFormer instrumental names skipped: {type(e).__name__}: {e})")
        return sections


def _sx_mid_song_silence(secs, beatfeat):
    """0.7-pre59 STRUCTURE_MID_SONG_SILENCE: digital silence inside the song as its own
    "Silence" section, from the per-beat loudness the map was built from. Returns
    the map unchanged (the same list) when there is none; never raises."""
    try:
        if not STRUCTURE_MID_SONG_SILENCE or not secs or not beatfeat:
            return secs
        edges = beatfeat.get("edges")
        energy = beatfeat.get("energy")
        if edges is None or energy is None:
            return secs
        edges = np.asarray(edges, dtype=float)
        energy = np.asarray(energy, dtype=float)
        if energy.size < 8 or edges.size != energy.size + 1:
            return secs
        db = 20.0 * np.log10(np.maximum(energy, 1e-10))
        quiet = db <= float(np.percentile(db, 95)) - _MID_SILENCE_BELOW_DB
        music_start = float(beatfeat.get("music_start") or 0.0)
        music_end = float(beatfeat.get("music_end") or edges[-1])
        runs, i = [], 0
        while i < len(quiet):
            if not quiet[i]:
                i += 1
                continue
            j = i
            while j + 1 < len(quiet) and quiet[j + 1]:
                j += 1
            a, b = float(edges[i]), float(edges[j + 1])
            if (b - a >= float(STRUCTURE_MID_SILENCE_MIN_S)
                    and a >= music_start + float(STRUCTURE_MID_SILENCE_EDGE_S)
                    and b <= music_end - float(STRUCTURE_MID_SILENCE_EDGE_S)):
                runs.append((round(a, 2), round(b, 2)))
            i = j + 1
        if not runs:
            return secs
        out = [dict(x) for x in secs]
        split_twice = False
        for a, b in runs:
            new, inserted = [], False
            for x in out:
                xs, xe = float(x["start"]), float(x["end"])
                if xe <= a or xs >= b:
                    new.append(x)
                    continue
                if _sx_base_name(x.get("name")) == "Silence":
                    new.append(x)       # already silence: left as it is
                    inserted = True
                    continue
                kept = 0
                if xs < a:
                    piece = dict(x, end=a)
                    piece.pop("name_check", None)
                    piece["_cut"] = True
                    new.append(piece)
                    kept += 1
                if not inserted:
                    new.append({"name": "Silence", "start": a, "end": b, "cluster": "Unique"})
                    inserted = True
                if xe > b:
                    piece = dict(x, start=b)
                    piece.pop("name_check", None)
                    piece["_cut"] = True
                    new.append(piece)
                    kept += 1
                if kept == 2:
                    split_twice = True
            # A cut-off piece too short to stand alone joins the section on its other side.
            k = next((n for n, x in enumerate(new) if _sx_base_name(x.get("name")) == "Silence"
                      and abs(float(x["start"]) - a) < 0.01), None)
            if k is not None:
                left_min = float(STRUCTURE_MID_SILENCE_LEFTOVER_S)
                if (k + 2 < len(new) and new[k + 1].get("_cut")
                        and float(new[k + 1]["end"]) - float(new[k + 1]["start"]) < left_min
                        and _sx_base_name(new[k + 2].get("name")) != "Silence"):
                    new[k + 2]["start"] = new[k + 1]["start"]
                    del new[k + 1]
                if (k >= 2 and new[k - 1].get("_cut")
                        and float(new[k - 1]["end"]) - float(new[k - 1]["start"]) < left_min
                        and _sx_base_name(new[k - 2].get("name")) != "Silence"):
                    new[k - 2]["end"] = new[k - 1]["end"]
                    del new[k - 1]
            out = new
        for x in out:
            x.pop("_cut", None)
        if split_twice:
            names = [_sx_base_name(x.get("name")) for x in out]
            out = _sx_renumber(out) if len(names) != len(set(names)) else out
        return out
    except Exception as e:
        print(f"  (mid-song silence check skipped: {type(e).__name__}: {e})")
        return secs


def _sx_split_chorus_repeats(secs, beatfeat):
    """0.7-pre64 STRUCTURE_SPLIT_CHORUS_REPEATS (see the setting): a chorus at least
    STRUCTURE_CHORUS_REPEAT_LONG times the song's usual chorus is split at each point,
    one usual chorus length on, where the music matches the song's other choruses (key
    changes allowed) as closely as they match each other. With
    STRUCTURE_CHORUS_REPEAT_HEAD_PARTS, a long chorus whose start doesn't match them
    has the piece before its first repeat named as the second part of the section
    before it. Anything missing, or no match: the map as it was."""
    try:
        if not STRUCTURE_SPLIT_CHORUS_REPEATS or not secs or not isinstance(beatfeat, dict):
            return secs
        ch = [i for i, x in enumerate(secs) if _sx_base_name(x.get("name")) == "Chorus"]
        if len(ch) < 2:
            return secs
        chroma = np.asarray(beatfeat.get("chroma"), dtype=float)
        mfcc = np.asarray(beatfeat.get("mfcc"), dtype=float)
        edges = np.asarray(beatfeat.get("edges"), dtype=float)
        if (chroma.ndim != 2 or mfcc.ndim != 2 or chroma.shape[0] != 12 or edges.ndim != 1
                or chroma.shape[1] < 8 or mfcc.shape[1] != chroma.shape[1] or len(edges) != chroma.shape[1] + 1):
            return secs
        n = chroma.shape[1]
        chn = chroma / (np.linalg.norm(chroma, axis=0, keepdims=True) + 1e-9)
        mfz = (mfcc - mfcc.mean(1, keepdims=True)) / (mfcc.std(1, keepdims=True) + 1e-9)
        mfn = mfz / (np.linalg.norm(mfz, axis=0, keepdims=True) + 1e-9)

        def at(t):
            return int(np.clip(np.searchsorted(edges, float(t), side="right") - 1, 0, n - 1))

        def match(r, p, k):
            # how alike k beats from beat r and from beat p are: chords in any key, and sound
            if r + k > n or p + k > n:
                return -1.0
            a, b = chn[:, r:r + k], chn[:, p:p + k]
            keyless = max(float(np.mean(np.sum(np.roll(a, t, axis=0) * b, axis=0))) for t in range(12))
            sound = float(np.mean(np.sum(mfn[:, r:r + k] * mfn[:, p:p + k], axis=0)))
            return 0.6 * keyless + 0.4 * sound

        # Chorus A/B groups (pre60) are measured within their own group.
        group = {i: "" for i in ch}
        if STRUCTURE_CHORUS_GROUPS:
            named = _sx_chorus_groups(secs)
            for i in ch:
                m_ = re.match(r"Chorus ([A-C])\d+$", str(named[i].get("name") or ""))
                group[i] = m_.group(1) if m_ else ""
        cuts_at, heads = {}, set()
        for i in ch:
            mates = [j for j in ch if group[j] == group[i]]
            if len(mates) < 2:
                continue
            lengths = [float(secs[j]["end"]) - float(secs[j]["start"]) for j in mates]
            usual = float(np.median(lengths))
            s0, s1 = float(secs[i]["start"]), float(secs[i]["end"])
            if usual <= 0 or s1 - s0 < float(STRUCTURE_CHORUS_REPEAT_LONG) * usual:
                continue
            refs = [at(secs[j]["start"]) for j, ln in zip(mates, lengths)
                    if j != i and ln <= float(STRUCTURE_CHORUS_REPEAT_REF_MAX) * usual]
            if not refs:
                continue
            a, b = at(s0), at(s1)
            beat = float(np.median(np.diff(edges[a:b + 1]))) if b > a else 0.5
            # How long one chorus is, measured inside this one: the shortest repeat that
            # matches nearly as well as the best (choruses merged in the map make the
            # section lengths a poor guide).
            lo = max(2, int(STRUCTURE_CHORUS_REPEAT_MIN_BEATS), int(round(0.5 * usual / max(beat, 1e-3))))
            hi = min(int(round(1.5 * usual / max(beat, 1e-3))), (b - a) // 2)
            repeats = []
            for lag in range(lo, hi + 1):
                x_, y_ = chn[:, a:b - lag], chn[:, a + lag:b]
                keyless = max(float(np.mean(np.sum(np.roll(x_, t, axis=0) * y_, axis=0))) for t in range(12))
                sound = float(np.mean(np.sum(mfn[:, a:b - lag] * mfn[:, a + lag:b], axis=0)))
                repeats.append((0.6 * keyless + 0.4 * sound, lag))
            if repeats:
                top = max(v for v, _ in repeats)
                lag = min(l_ for v, l_ in repeats if v >= float(STRUCTURE_CHORUS_REPEAT_NEAR_BEST) * top)
                usual = lag * beat
                refs = [at(secs[j]["start"]) for j, ln in zip(mates, lengths)
                        if j != i and ln <= float(STRUCTURE_CHORUS_REPEAT_REF_MAX) * usual] or refs
            k = max(4, min(int(STRUCTURE_CHORUS_REPEAT_BEATS), int(round(usual / max(beat, 1e-3))) - 1))
            best = lambda p: max(match(r, p, k) for r in refs)
            pairs = [match(x, y, k) for q, x in enumerate(refs) for y in refs[q + 1:]]
            start_match = best(a)
            background = [best(p) for p in range(0, max(1, n - k), 4)]
            need = max(float(STRUCTURE_CHORUS_REPEAT_MATCH) * (max(pairs) if pairs else start_match),
                       float(np.percentile(background, 80)) if background else 0.0)
            cuts = []
            last = s0
            while True:
                expect = last + usual
                if s1 - expect < float(STRUCTURE_CHORUS_REPEAT_MIN_LAST) * usual:
                    break
                lo = expect - float(STRUCTURE_CHORUS_REPEAT_WINDOW) * usual
                hi = expect + float(STRUCTURE_CHORUS_REPEAT_WINDOW) * usual
                spots = [p for p in range(a, b) if lo <= edges[p] <= hi
                         and s1 - edges[p] >= float(STRUCTURE_CHORUS_REPEAT_MIN_LAST) * usual]
                if not spots:
                    break
                value, p = max((best(p), p) for p in spots)
                if value < need:
                    break
                last = float(edges[p])
                cuts.append(last)
            if cuts:
                cuts_at[i] = cuts
                if start_match < need:
                    heads.add(i)
        if not cuts_at:
            return secs
        out = []
        for i, x in enumerate(secs):
            if i not in cuts_at:
                out.append(dict(x))
                continue
            bounds = [float(x["start"])] + cuts_at[i] + [float(x["end"])]
            for q, (t0, t1) in enumerate(zip(bounds, bounds[1:])):
                piece = dict(x)
                piece["start"], piece["end"] = t0, t1
                piece.pop("name_check", None)
                if q == 0 and i in heads:
                    piece["_pre64_head"] = True
                out.append(piece)
        # the choruses numbered again in order; nothing else is renamed
        k_ = 0
        total = sum(1 for x in out if _sx_base_name(x.get("name")) == "Chorus" and not x.get("_pre64_head"))
        for x in out:
            if _sx_base_name(x.get("name")) == "Chorus" and not x.get("_pre64_head"):
                k_ += 1
                x["name"] = f"Chorus {k_}" if total > 1 else "Chorus"
        for q, x in enumerate(out):
            if not x.pop("_pre64_head", False):
                continue
            prev = out[q - 1] if q else None
            if (STRUCTURE_CHORUS_REPEAT_HEAD_PARTS and prev is not None
                    and _sx_base_name(prev.get("name")) not in ("Silence", "")
                    and abs(float(prev["end"]) - float(x["start"])) < 0.5
                    and not re.search(r"\bPart \d+$", str(prev.get("name") or ""))):
                head = str(prev.get("name"))
                prev["name"] = f"{head} Part 1"
                x["name"] = f"{head} Part 2"
                x["cluster"] = prev.get("cluster", x.get("cluster"))
            else:
                x["name"] = "Chorus"           # numbered with the others below
        if any(x.get("name") == "Chorus" for x in out):
            k_ = 0
            total = sum(1 for x in out if _sx_base_name(x.get("name")) == "Chorus")
            for x in out:
                if _sx_base_name(x.get("name")) == "Chorus":
                    k_ += 1
                    x["name"] = f"Chorus {k_}" if total > 1 else "Chorus"
        return out
    except Exception as e:
        print(f"  (chorus repeat check skipped: {type(e).__name__}: {e})")
        return secs


def _sx_chorus_groups(sections):
    """0.7-pre60 STRUCTURE_CHORUS_GROUPS (see the setting): the map with its choruses
    named by group of the same music when they form two or three groups that each
    come back; otherwise the map as it was."""
    try:
        if not STRUCTURE_CHORUS_GROUPS or not sections:
            return sections
        ch = [i for i, s in enumerate(sections) if _sx_base_name(s.get("name")) == "Chorus"]
        if len(ch) < 4:
            return sections
        groups = {}
        for i in ch:
            cl = str(sections[i].get("cluster") or "").strip()
            if not cl or cl.lower() == "unique":
                return sections          # a chorus that is no group's: the names stay
            groups.setdefault(cl, []).append(i)
        if not 2 <= len(groups) <= 3 or any(len(v) < 2 for v in groups.values()):
            return sections
        out = [dict(s) for s in sections]
        for letter, cl in zip("ABC", sorted(groups, key=lambda c: groups[c][0])):
            for k, i in enumerate(groups[cl], 1):
                out[i]["name"] = f"Chorus {letter}{k}"
        return out
    except Exception:
        return sections


_SX_CVW_STOP = frozenset(
    "the and you your that this with for are was were have has had but not all just what when then than "
    "they them their there here from into onto out over its it's i'm im you're youre we're dont don't "
    "can't cant ill i'll gonna wanna gotta yeah ooh oh ohh whoa hey baby got get let like know one now "
    "cause come said say she he her his him our ours who how why let's lets will would could should "
    "been being some more very much too also only even ever every way".split())
_SX_CVW_CHORUS = re.compile(r"^Chorus(?: \d+)?$")
_SX_CVW_VERSE = re.compile(r"^Verse(?: \d+)?(?: Part \d+)?$")


def _sx_cvw_words(chunks, a, b):
    """pre68: the content words sung in [a, b) -- each timed lyric chunk counted where
    its middle falls; short and very common words left out, a plural 's' dropped."""
    words = set()
    for c in chunks or []:
        try:
            mid = (float(c["start"]) + float(c["end"])) / 2.0
        except (KeyError, TypeError, ValueError):
            continue
        if not a <= mid < b:
            continue
        for w in re.findall(r"[a-z']+", str(c.get("text") or "").lower()):
            w = w.strip("'")
            if w.endswith("'s"):
                w = w[:-2]
            if len(w) < 3 or w in _SX_CVW_STOP:
                continue
            if len(w) > 4 and w.endswith("s"):
                w = w[:-1]
            words.add(w)
    return words


def _sx_cvw_share(x, y):
    return len(x & y) / min(len(x), len(y)) if x and y else 0.0


def _sx_chorus_verse_words(sections, lyric_chunks, tag_lyrics=None):
    """0.7-pre68 STRUCTURE_CHORUS_SHARED_WORDS (see the setting): the map with each
    'chorus' whose words are a verse's renamed as a verse, or the map as it was."""
    try:
        secs = list(sections or [])
        names = [str(s.get("name") or "") for s in secs]
        ch = [i for i, n in enumerate(names) if _SX_CVW_CHORUS.match(n)]
        if len(ch) < 3 or not lyric_chunks:
            return sections
        if any(re.match(r"^(?:Chorus|Verse)\b", n) and not (_SX_CVW_CHORUS.match(n) or _SX_CVW_VERSE.match(n))
               for n in names):
            return sections               # names this step doesn't number (Chorus A1, Part I Verse ...)
        if tag_lyrics and _sx_transcript_unmatched(lyric_chunks, tag_lyrics):
            return sections               # the transcript isn't the song's words (dev26)
        span = lambda i: (float(secs[i]["start"]), float(secs[i]["end"]))
        words = {i: _sx_cvw_words(lyric_chunks, *span(i)) for i in ch}
        sung = [i for i in ch if len(words[i]) >= 4]
        core = [i for i in sung if any(_sx_cvw_share(words[i], words[j]) >= 0.5 for j in sung if j != i)]
        if len(core) < 2:
            return sections               # no two choruses share their words: nothing to go by
        verse_words = [w for w in (_sx_cvw_words(lyric_chunks, *span(i))
                                   for i, n in enumerate(names) if _SX_CVW_VERSE.match(n)) if len(w) >= 4]
        if not verse_words:
            return sections
        moved = []
        for i in sung:
            if i in core or len(words[i]) < 6:
                continue
            as_chorus = max(_sx_cvw_share(words[i], words[j]) for j in core)
            as_verse = max(_sx_cvw_share(words[i], w) for w in verse_words)
            if as_verse >= 0.5 and as_verse >= as_chorus + 0.25:
                moved.append(i)
        if not moved or len(ch) - len(moved) < 2:
            return sections
        out = [dict(s) for s in secs]
        for i in moved:
            out[i].pop("name_check", None)
        rest = [i for i in ch if i not in moved]
        for k, i in enumerate(rest, 1):
            out[i]["name"] = "Chorus %d" % k
        # The verses keep their grouping (a verse's parts stay one verse); a renamed
        # section joins the verse it runs straight into, else the one it follows,
        # else it is a verse of its own. Then the verses are numbered in order.
        unit = {}
        for i, n in enumerate(names):
            if _SX_CVW_VERSE.match(n):
                unit[i] = re.sub(r" Part \d+$", "", n)
        for i in moved:
            if i + 1 in unit and i + 1 not in moved:
                unit[i] = unit[i + 1]
            elif i - 1 in unit:
                unit[i] = unit[i - 1]
            else:
                unit[i] = "new %d" % i
        order = []
        for i in sorted(unit):
            if unit[i] not in order:
                order.append(unit[i])
        members = {u: [i for i in sorted(unit) if unit[i] == u] for u in order}
        for k, u in enumerate(order, 1):
            head = "Verse" if len(order) == 1 else "Verse %d" % k
            for j, i in enumerate(members[u], 1):
                out[i]["name"] = head if len(members[u]) == 1 else "%s Part %d" % (head, j)
        return out
    except Exception:
        return sections


def _sx_join_stanza_tails(sections, lyric_chunks, tag_lyrics=None):
    """0.7.1-pre20 STRUCTURE_JOIN_STANZA_TAIL (see the setting): the map with each short chorus
    that only sings the rest of the stanza the section before was singing joined to that section,
    or the map as it was."""
    try:
        secs = list(sections or [])
        if len(secs) < 2 or not lyric_chunks or not tag_lyrics:
            return sections
        if _sx_transcript_unmatched(lyric_chunks, tag_lyrics):
            return sections               # the transcript isn't the song's words (dev26)
        tag = _tag_lyric_lines(tag_lyrics)
        stanza_lines = {}
        for idx, (_w, st) in enumerate(tag):
            stanza_lines.setdefault(st, []).append(idx)
        if len(stanza_lines) < 3:
            return sections               # no real stanzas to go by
        lines, bare = [], []              # (time, [(stanza, first, index) ...]); chunks with no tag line
        for c in lyric_chunks:
            if not isinstance(c, dict):
                continue
            a, b = float(c.get("start") or 0.0), float(c.get("end") or 0.0)
            cuts, stz = c.get("lines"), c.get("line_stanzas")
            if not cuts or not stz:
                if len(_lyric_norm_words(c.get("text") or "")) >= 1:
                    bare.append(a)
                continue
            cuts = [int(x) for x in cuts]
            nw = max(1, cuts[-1])
            for k, cand in enumerate(stz):
                if k + 1 >= len(cuts):
                    break
                lines.append((a + (b - a) * cuts[k] / nw,
                              [(int(x[0]), bool(x[1]), int(x[2])) for x in (cand or [])]))
        lines.sort(key=lambda x: x[0])
        out = [dict(s) for s in secs]
        i = 1
        while i < len(out):
            s, p = out[i], out[i - 1]
            a, z = float(s["start"]), float(s["end"])
            name = str(s.get("name") or "")
            base = _sx_base_name(name)
            if (z - a > 12.0 or base != "Chorus" or not re.fullmatch(r"Chorus(?: \d+)?", name)
                    or _sx_base_name(re.sub(r" Part \d+$", "", str(p.get("name") or "")))
                    not in ("Verse", "Pre-Chorus", "Chorus", "Post-Chorus", "Bridge")):
                i += 1
                continue
            # a line starting in the 1.5 s before a section is that section's pickup
            zz = z - 1.5 if i + 1 < len(out) else z
            mine = [ln for ln in lines if a - 1.5 <= ln[0] < zz]
            before = [ln for ln in lines if float(p["start"]) <= ln[0] < a - 1.5]
            if not mine or not before or any(a - 1.5 <= t < zz for t in bare) or any(not ln[1] for ln in mine):
                i += 1
                continue
            joined = False
            for st, _f, last_idx in before[-1][1]:
                idxs = []
                for _t, cand in mine:
                    nxt = [x[2] for x in cand if x[0] == st and x[2] > (idxs[-1] if idxs else last_idx)]
                    if not nxt:
                        break
                    idxs.append(min(nxt))
                st_lines = stanza_lines.get(st) or []
                if (len(idxs) == len(mine) and st_lines and len(st_lines) <= 16
                        and idxs[0] <= last_idx + 2 and idxs[-1] == st_lines[-1]):
                    joined = True
                    break
            if not joined:
                i += 1
                continue
            family = [k for k, x in enumerate(out) if k != i and _sx_base_name(x.get("name")) == base]
            if any(not re.fullmatch(re.escape(base) + r"(?: \d+)?", str(out[k].get("name") or "")) for k in family):
                i += 1
                continue                  # names this step doesn't number (Chorus A1, Verse 1 Part 2 ...)
            p["end"] = s["end"]
            del out[i]
            family = [k for k, x in enumerate(out) if _sx_base_name(x.get("name")) == base]
            for n, k in enumerate(family, 1):
                out[k]["name"] = base if len(family) == 1 else f"{base} {n}"
        return out if len(out) != len(secs) else sections
    except Exception:
        return sections


# 0.7.1-pre21 STRUCTURE_DROP_UNRELIABLE_TRANSCRIPT: the commonest short words of the languages a
# transcript may switch between, to tell which language a transcript line is in.
_SX_LINE_LANGUAGE_WORDS = {k: frozenset(v.split()) for k, v in {
    "en": "the and you i to a me my it in of that is your on be for all we what with this can just so but not "
          "don't i'm it's when like know love no do are was have get one go up out if oh baby now never yeah got "
          "let come say see there they will make way how time",
    "cy": "yn y yr ar o a i mae dwi ddim fy ei ein eich am gyda dros mewn sy sydd beth dyma rhaid dydd amser "
          "fawr iawn diolch wedi bod rydw ydy ydw oedd hwn hon nhw ni chi",
    "de": "ich du und der die das nicht ist mit ein eine es wir sie zu auf mich dich für dass wie so mein dein "
          "noch nur",
    "fr": "je tu il elle et le la les de des un une que qui pas est mon ton ma ta mes pour dans sur avec moi "
          "toi nous vous ne plus",
    "es": "yo tu el la los las de que y en un una no es mi te me se por con para como pero mas todo amor quiero",
    "it": "io tu il lo la le di che e non un una per con mi ti si sono come ma più tutto amore",
    "pt": "eu você o a os as de que e em um uma não é meu minha te me se por com para como mas tudo amor",
    "nl": "ik je jij het de een en van niet is dat met mijn op voor wat zijn maar ook",
}.items()}
# Whisper's stock lines, also when split across two lines ("We'll" / "be right back") or in Welsh.
_SX_STOCK_LINE_RE = re.compile(r"(?i)\bbe right back\b|\b(?:yn fawr(?: iawn)? )?am wylio\b")


def _sx_line_language(text):
    """0.7.1-pre21: the language of one transcript line of three words or more, by its commonest
    short words (at least two, and more than for any other language), or None."""
    w = re.findall(r"[^\W\d_]+(?:'[^\W\d_]+)?", str(text or "").lower().replace("\u2019", "'"))
    if len(w) < 3:
        return None
    hits = {k: sum(x in s for x in w) for k, s in _SX_LINE_LANGUAGE_WORDS.items()}
    best = sorted(hits.values())
    top = max(hits, key=hits.get)
    return top if hits[top] >= 2 and hits[top] > best[-2] else None


def _sx_transcript_unreliable(lyric_chunks, lyrics_language=None):
    """0.7.1-pre21 STRUCTURE_DROP_UNRELIABLE_TRANSCRIPT (see the setting): True when a transcript's
    lines switch language, or carry three or more of Whisper's stock lines."""
    try:
        texts = [str(c.get("text") or "") for c in (lyric_chunks or []) if isinstance(c, dict)]
        if not (isinstance(lyrics_language, dict) and lyrics_language.get("kind") == "mixed"):
            langs = collections.Counter(l for l in map(_sx_line_language, texts) if l)
            if len(langs) >= 2:
                second = langs.most_common(2)[1][1]
                if second >= 4 and second >= 0.15 * sum(langs.values()):
                    return True
        stock = sum(1 for t in texts
                    if _SX_STOCK_LINE_RE.search(t) or any(p.search(t) for p in _WHISPER_HALLUCINATION_PATTERNS))
        return stock >= 3
    except Exception:
        return False


def _sx_renumber_parts(secs):
    """0.7.1-pre23: section names numbered again in order, the parts of one section kept together
    ("Verse 1 Part 1", "Verse 1 Part 2"); a kind that occurs once has no number. A map whose
    sections were neither added nor renamed comes back with the same names."""
    rx = re.compile(r"^(.*?)(?: (\d+))?(?: Part (\d+))?$")
    units = []
    for i, x in enumerate(secs):
        m = rx.match(str(x.get("name") or "").strip())
        kind, num, part = m.group(1), m.group(2), m.group(3)
        if part and units and units[-1][0] == kind and units[-1][2] == num and units[-1][3]:
            units[-1][1].append(i)
        else:
            units.append([kind, [i], num, bool(part)])
    counts = {}
    for u_ in units:
        counts[u_[0]] = counts.get(u_[0], 0) + 1
    seen, out = {}, [dict(x) for x in secs]
    for kind, idx, num, haspart in units:
        if kind.lower() == "silence" or counts[kind] == 1:
            head = kind
        else:
            seen[kind] = seen.get(kind, 0) + 1
            head = "%s %d" % (kind, seen[kind])
        for j, i in enumerate(idx, 1):
            out[i]["name"] = head if len(idx) == 1 else "%s Part %d" % (head, j)
    return out


# 0.7.1-pre23 STRUCTURE_PRECHORUS_VERSE_ENDINGS: short words left out of a title when looking for
# it in the lyrics.
_SX_VE_STOP = frozenset("a an the of to in on at for and or but my your his her its our their i i'm im you me we us "
                        "it is be am are was were this that with from by as so do don't dont can can't cant will "
                        "won't just oh yeah hey baby ooh la na".split())


def _sx_ve_fold(s):
    s = str(s or "").lower().replace("\u2019", "'").replace("\u2024", ".")
    return re.sub(r"\(.*?\)|\[.*?\]", " ", s)


def _sx_ve_title_words(title):
    """0.7.1-pre23: the title's words of three letters or more, short common words left out."""
    t = _sx_ve_fold(title).replace(".", "")
    return [x for x in re.findall(r"[a-z0-9']+", t) if x not in _SX_VE_STOP and len(x) >= 3]


def _sx_ve_has_title(text, tw):
    """0.7.1-pre23: a lyric line holds the title: all its words, or the title written together."""
    if not tw:
        return False
    t = _sx_ve_fold(text).replace(".", "")
    words = set(re.findall(r"[a-z0-9']+", t))
    if all(x in words for x in tw):
        return True
    comp = re.sub(r"[^a-z0-9]", "", t)
    return len("".join(tw)) >= 6 and "".join(tw) in comp


def _sx_ve_lines(chunks, a, b):
    """0.7.1-pre23: [(start, end, text)] of the timed lyric lines whose middle is in [a, b)."""
    out = []
    for c in chunks or []:
        try:
            s, e = float(c["start"]), float(c["end"] if c.get("end") is not None else c["start"])
        except Exception:
            continue
        if str(c.get("text") or "").strip() and a <= (s + e) / 2 < b:
            out.append((s, e, str(c.get("text"))))
    return sorted(out)


def _sx_ve_occurrences(groups):
    """0.7.1-pre23: [(group index, start, end)] for every occurrence of every repeated line."""
    out = []
    for gi, g in enumerate(groups or []):
        for o in g.get("occurrences") or []:
            o = o.get("__tuple__", o) if isinstance(o, dict) else o
            try:
                out.append((gi, float(o[0]), float(o[1])))
            except Exception:
                pass
    return out


def _sx_ve_line_groups(line, occ):
    """0.7.1-pre23: the repeated-line groups a lyric line is an occurrence of (it overlaps one by at
    least half of the shorter of the two)."""
    s, e, _ = line
    gs = set()
    for gi, a, b in occ:
        ov = min(e, b) - max(s, a)
        if ov > 0 and ov >= 0.5 * min(max(e - s, 0.01), max(b - a, 0.01)):
            gs.add(gi)
    return gs


def _sx_ve_words(text):
    return re.findall(r"[a-z]+(?:[-'][a-z]+)*", _sx_ve_fold(text))


def _sx_prechorus_verse_endings(secs, chunks, groups, title=None):
    """0.7.1-pre23 STRUCTURE_PRECHORUS_VERSE_ENDINGS (see the setting): the map with the recurring
    last lines of its verses before a chorus as Pre-Choruses (or, when they hold the title, as the
    start of the chorus); otherwise the map as it was."""
    try:
        occ = _sx_ve_occurrences(groups)
        ends = [k for k in range(len(secs) - 1) if re.match(r"^Verse\b", _sx_base_name(secs[k].get("name")))
                and _sx_base_name(secs[k + 1].get("name")) == "Chorus"]
        if len(ends) < 2:
            return secs
        tails = {}
        for k in ends:
            st, en = float(secs[k]["start"]), float(secs[k]["end"])
            tails[k] = [(l, _sx_ve_line_groups(l, occ)) for l in _sx_ve_lines(chunks, st, en)]

        def recurs(k, gs):
            # the line comes back in the last 25 s of another verse before a chorus
            for k2 in ends:
                if k2 == k:
                    continue
                en2 = float(secs[k2]["end"])
                for l, g2 in tails[k2]:
                    if gs & g2 and l[0] >= en2 - 25.0:
                        return True
            return False
        cut = {}
        for k in ends:
            ls = tails[k]
            if len(ls) < 3:
                continue
            j = len(ls)
            while j > 0 and ls[j - 1][1] and recurs(k, ls[j - 1][1]):
                j -= 1
            if j == len(ls) or j < 2:
                continue
            L = ls[j][0]
            st, en = float(secs[k]["start"]), float(secs[k]["end"])
            body_rec = sum(1 for l, g in ls[:j] if g and recurs(k, g))
            if 5.0 <= en - L[0] <= 25.0 and L[0] - st >= 8.0 and body_rec * 2 < j:
                cut[k] = L[0]
        existing = sum(1 for k in range(1, len(secs)) if _sx_base_name(secs[k].get("name")) == "Pre-Chorus"
                       and k + 1 < len(secs) and _sx_base_name(secs[k + 1].get("name")) == "Chorus")
        if not cut or len(cut) + existing < 2:
            return secs              # one verse alone is not a pattern
        tw = _sx_ve_title_words(title)
        titled = {k for k in cut if any(_sx_ve_has_title(l[2], tw) for l, g in tails[k] if l[0] >= cut[k] - 0.01)}
        if titled and len(titled) != len(cut):
            return secs              # the verse endings disagree about holding the title
        out = []
        for k, s in enumerate(secs):
            if k in cut and k in titled:
                a = dict(s)
                a["end"] = round(cut[k], 2)
                out.append(a)
            elif k - 1 in cut and k - 1 in titled:
                c = dict(s)
                c["start"] = round(cut[k - 1], 2)
                out.append(c)
            elif k in cut:
                a = dict(s)
                a["end"] = round(cut[k], 2)
                p = dict(s)
                p["start"] = round(cut[k], 2)
                p["name"] = "Pre-Chorus"
                p["cluster"] = "Unique"
                out += [a, p]
            else:
                out.append(dict(s))
        return _sx_renumber_parts(out)
    except Exception:
        return secs


def _sx_late_first_word(secs, chunks):
    """0.7.1-pre23 STRUCTURE_LATE_FIRST_WORD (see the setting): the map with each sung section after
    an intro or instrumental starting at its first full line; otherwise the map as it was."""
    try:
        out = [dict(x) for x in secs]
        changed = False
        for i in range(1, len(out)):
            b = _sx_base_name(out[i].get("name"))
            pb = _sx_base_name(out[i - 1].get("name"))
            if not re.match(r"^(Verse|Chorus|Pre-Chorus|Bridge)\b", b) or pb not in ("Intro", "Instrumental"):
                continue
            st, en = float(out[i]["start"]), float(out[i]["end"])
            ls = _sx_ve_lines(chunks, st - 1.5, en)
            if len(ls) < 2:
                continue
            c0, c1 = ls[0], ls[1]
            w = _sx_ve_words(c0[2])
            if (abs(c0[0] - st) <= 1.5 and 1 <= len(w) <= 3 and (c0[1] - c0[0]) / len(w) >= 1.25
                    and c1[0] - c0[1] >= 1.5 and c1[0] < en - 4.0 and c1[0] > st):
                out[i - 1]["end"] = round(c1[0], 2)
                out[i]["start"] = round(c1[0], 2)
                changed = True
        return out if changed else secs
    except Exception:
        return secs


def _sf_section_spans(sf_sections):
    """0.7.1-pre24: SongFormer's sections as (start, end, label) for _songformer_vote."""
    return [(float(x["start"]), float(x["end"]), str(x.get("label") or "").lower())
            for x in (sf_sections or []) if isinstance(x, dict)]


def _sf_prechorus_before_chorus(secs, sf_sections):
    """0.7.1-pre24 SONGFORMER_PRECHORUS_BEFORE_CHORUS (see the setting): the map with the first of two
    back-to-back choruses named Pre-Chorus where SongFormer hears one; otherwise the map as it was."""
    try:
        if not sf_sections:
            return secs
        sf = _sf_section_spans(sf_sections)
        out = [dict(x) for x in secs]
        changed = False
        for i in range(len(out) - 1):
            if _sx_base_name(out[i].get("name")) != "Chorus" or _sx_base_name(out[i + 1].get("name")) != "Chorus":
                continue
            v = _songformer_vote(sf, float(out[i]["start"]), float(out[i]["end"]))
            tot = sum(v.values())
            if tot and v.get("pre-chorus", 0.0) >= _SF_NAME_SHARE * tot:
                out[i]["name"] = "Pre-Chorus"
                out[i]["cluster"] = "Unique"
                changed = True
        return _sx_renumber_parts(out) if changed else secs
    except Exception:
        return secs


def _sf_late_bridge(secs, sf_sections):
    """0.7.1-pre24 SONGFORMER_LATE_BRIDGE (see the setting): the map with each Verse after two or more
    choruses named Bridge where SongFormer hears one; otherwise the map as it was."""
    try:
        if not sf_sections:
            return secs
        sf = _sf_section_spans(sf_sections)
        out = [dict(x) for x in secs]
        changed, nch = False, 0
        for i, x in enumerate(out):
            if _sx_base_name(x.get("name")) == "Chorus":
                nch += 1
                continue
            if not re.match(r"^Verse(?: \d+)?$", str(x.get("name") or "")) or nch < 2:
                continue
            v = _songformer_vote(sf, float(x["start"]), float(x["end"]))
            tot = sum(v.values())
            if tot and v.get("bridge", 0.0) >= _SF_NAME_SHARE * tot:
                out[i]["name"] = "Bridge"
                out[i]["cluster"] = "Unique"
                changed = True
        return _sx_renumber_parts(out) if changed else secs
    except Exception:
        return secs


def _resolve_final_structure(*args, vocal_presence_unresolved=False, **kwargs):
    _lyrics_language = kwargs.pop("lyrics_language", None)                       # 0.7.1-pre21
    if (STRUCTURE_DROP_UNRELIABLE_TRANSCRIPT and kwargs.get("lyric_chunks") and not kwargs.get("tag_lyrics")
            and not kwargs.get("instrumental", False)
            and _sx_transcript_unreliable(kwargs.get("lyric_chunks"), _lyrics_language)):
        kwargs = dict(kwargs, lyric_chunks=[], lyric_groups=[])                  # 0.7.1-pre21: go by the music
    if vocal_presence_unresolved and kwargs.get("songformer_sections"):
        kwargs = dict(kwargs, songformer_unsung=True)     # pre59: named neutrally below
    sections, source = _resolve_final_structure_impl(*args, **kwargs)
    if vocal_presence_unresolved or kwargs.get("instrumental", False):
        sections = _sx_neutral_vocal_roles(sections)
    if (kwargs.get("songformer_sections") and SONGFORMER_CORRECT_STRUCTURE
            and SONGFORMER_INSTRUMENTAL_NAMES and source == "measured"
            and (kwargs.get("instrumental", False) or not _sf_has_sung_names(sections))):
        sections = _songformer_instrumental_names(                              # pre59
            sections, kwargs.get("songformer_sections"), kwargs.get("total_duration"))
    if (STRUCTURE_CHORUS_SHARED_WORDS and sections and source == "measured"
            and not vocal_presence_unresolved and not kwargs.get("instrumental", False)):
        sections = _sx_chorus_verse_words(sections, kwargs.get("lyric_chunks"),     # pre68
                                          kwargs.get("tag_lyrics"))
    if (STRUCTURE_JOIN_STANZA_TAIL and sections and source == "measured"
            and not vocal_presence_unresolved and not kwargs.get("instrumental", False)):
        sections = _sx_join_stanza_tails(sections, kwargs.get("lyric_chunks"),     # 0.7.1-pre20
                                         kwargs.get("tag_lyrics"))
    if (STRUCTURE_PRECHORUS_VERSE_ENDINGS and sections and source == "measured" and kwargs.get("lyric_chunks")
            and not vocal_presence_unresolved and not kwargs.get("instrumental", False)):
        sections = _sx_prechorus_verse_endings(sections, kwargs.get("lyric_chunks"),   # 0.7.1-pre23
                                               kwargs.get("lyric_groups"), kwargs.get("title"))
    if (STRUCTURE_LATE_FIRST_WORD and sections and source == "measured" and kwargs.get("lyric_chunks")
            and not vocal_presence_unresolved and not kwargs.get("instrumental", False)):
        sections = _sx_late_first_word(sections, kwargs.get("lyric_chunks"))        # 0.7.1-pre23
    if (sections and source == "measured" and kwargs.get("songformer_sections") and SONGFORMER_CORRECT_STRUCTURE
            and not vocal_presence_unresolved and not kwargs.get("instrumental", False)):
        if SONGFORMER_PRECHORUS_BEFORE_CHORUS:
            sections = _sf_prechorus_before_chorus(sections, kwargs.get("songformer_sections"))   # 0.7.1-pre24
        if SONGFORMER_LATE_BRIDGE:
            sections = _sf_late_bridge(sections, kwargs.get("songformer_sections"))               # 0.7.1-pre24
    if STRUCTURE_CHORUS_GROUPS and sections:
        sections = _sx_chorus_groups(sections)                                     # pre60
    return sections, source


def _resolve_final_structure_impl(
    moss_sections, dsp_sections=None, lyric_groups=None, lyric_chunks=None,
    total_duration=None, grid=None, stem_notes=None, bpm=None,
    instrumental=None, title=None, tag_lyrics=None,
    songformer_sections=None, songformer_dance=False, songformer_unsung=False,
):
    """Decide the track's final section map.

    Structure v2: the map is MEASURED. The cached mix features from the early
    pass (grid["beatfeat"]) are re-segmented with every piece of evidence now
    available -- separated-stem note events (fills, drum and vocal entries,
    vocal register) and Whisper lyric timing/repetition -- and named from
    that evidence. The listening model's own STRUCTURE=[...] field is only a
    last resort: its timestamps are not grounded in the audio (it was the
    source of sections like a 60-second "Pre-Chorus"), so it is used only if
    nothing measured is available.

    Returns (sections, source) where source is "measured", "dsp", or one of
    the previous pipeline's labels when it had to be used.
    """
    stem_notes = _structure_stem_evidence(stem_notes)
    grid = grid or {}
    if STRUCTURE_V2_ENABLED:
        secs = []
        _sx_details = {}
        if (STRUCTURE_DROP_UNMATCHED_TRANSCRIPT and tag_lyrics and lyric_chunks and not instrumental
                and _sx_transcript_unmatched(lyric_chunks, tag_lyrics)):
            # dev26: see STRUCTURE_DROP_UNMATCHED_TRANSCRIPT.
            lyric_chunks, lyric_groups = [], []
        if STRUCTURE_TAG_FRAGMENTS and STRUCTURE_TAG_STANZAS and tag_lyrics and lyric_chunks and not instrumental:
            try:
                lyric_chunks = _sx_align_tag_fragments(lyric_chunks, tag_lyrics)   # dev34
            except Exception:
                pass
        if STRUCTURE_SPLIT_STOCK_PHRASES and lyric_chunks and not instrumental:
            try:
                lyric_chunks, lyric_groups = _sx_drop_stock_phrase_chunks(lyric_chunks, lyric_groups)
            except Exception:
                pass
        if STRUCTURE_FOREIGN_CHUNKS and lyric_chunks and not instrumental:
            try:
                lyric_chunks, lyric_groups = _sx_drop_foreign_chunks(lyric_chunks, lyric_groups)
            except Exception:
                pass
        if STRUCTURE_TAG_FILL_MISSED and STRUCTURE_TAG_STANZAS and tag_lyrics and lyric_chunks and not instrumental:
            try:
                lyric_chunks = _sx_fill_missed_tag_lines(lyric_chunks, tag_lyrics,
                                                         _sx_vocal_notes_of(stem_notes))   # dev34
            except Exception:
                pass
        if grid.get("beatfeat") is not None and grid.get("beats"):
            # dev20: a grid at a non-octave ratio of the tempo is rebuilt for the map
            _beats, _rg = _sx_regrid_to_tempo(grid["beats"], bpm, stem_notes)
            if _rg:
                _bf = dict(grid["beatfeat"])
                _bf.pop("accents", None)        # they count the old grid's beats
                grid = dict(grid, beatfeat=_bf, beats=_beats)
            try:
                secs = build_structure_map(
                    None, None, total_duration,
                    beatfeat=grid["beatfeat"], beats=grid["beats"],
                    stem_notes=stem_notes, lyric_chunks=lyric_chunks,
                    lyric_groups=lyric_groups, instrumental=bool(instrumental),
                    details_out=_sx_details,
                    prechorus_by_words=not _sx_sheet_marks_verses(tag_lyrics),   # dev27
                    sheet_intro=bool(STRUCTURE_SUNG_INTRO_TAG_OR_SPEECH
                                     and _sx_sheet_opens_intro(tag_lyrics)),   # dev30
                    learned_namer=True,                                         # dev31
                    title=title,                                                # dev32
                )
            except Exception as e:
                print(f"  (structure v2 final pass failed: {type(e).__name__}: {e})")
                secs = []
        if secs:
            secs = _sx_apply_moss_hints(secs, grid.get("moss_hints"))
            secs = _normalize_structure_sections(secs, total_duration, min_section_s=2.0)
            if title and not instrumental:
                _pre_hook = [x.get("name") for x in secs]
                secs = _title_hook_naming(secs, lyric_chunks, title)
                if (songformer_sections and SONGFORMER_CORRECT_STRUCTURE
                        and [x.get("name") for x in secs] != _pre_hook):
                    for x in secs:                 # pre59: the title line settled these names
                        if _sx_base_name(x.get("name")) in ("Verse", "Chorus"):
                            x[_SF_LOCK] = True
            secs = _sx_merge_split_repeats(secs, grid.get("beats"))        # dev20
            secs = _sx_bridge_to_prechorus(secs, grid.get("beats"))        # dev20
            if tag_lyrics and not instrumental:
                _vn = None
                for _key in ("vocals", "vocal", "voice"):
                    if stem_notes and stem_notes.get(_key):
                        _vn = stem_notes[_key]
                        break
                secs = _sx_lyric_header_naming(secs, lyric_chunks, tag_lyrics, total_duration,
                                               grid.get("beats"),          # dev20
                                               cut_hints=_sx_details.get("dp_starts"),
                                               vocal_notes=_vn)            # dev26
            if STRUCTURE_MOVEMENT_NUMBERING and title and not instrumental:
                def _remap(t0, t1):
                    # dev26: one movement's section map, measured on its own
                    if grid.get("beatfeat") is None or not grid.get("beats"):
                        return None
                    _bf = dict(grid["beatfeat"])
                    _bf["music_start"] = float(t0)
                    _bf["music_end"] = float(t1)
                    _lc = [c for c in (lyric_chunks or []) if isinstance(c, dict)
                           and t0 - 0.5 <= float(c.get("start", -1)) < t1 - 0.5]
                    m_ = build_structure_map(
                        None, None, float(t1), beatfeat=_bf, beats=grid["beats"],
                        stem_notes=stem_notes, lyric_chunks=_lc or None, lyric_groups=None,
                        instrumental=False, details_out={},
                        prechorus_by_words=not _sx_sheet_marks_verses(tag_lyrics),
                        lone_sung_verse=False)     # dev30: a movement with nothing recurring keeps the suite's names
                    if not m_:
                        return None
                    m_ = _sx_merge_split_repeats(m_, grid.get("beats"))
                    m_ = _sx_bridge_to_prechorus(m_, grid.get("beats"))
                    return m_
                _vn2 = None
                for _key in ("vocals", "vocal", "voice"):
                    if stem_notes and stem_notes.get(_key):
                        _vn2 = stem_notes[_key]
                        break

                def _sung(t0, t1):
                    # True when the vocal stem carries a voice over a quarter of [t0, t1]
                    return bool(_vn2) and _sx_voiced_seconds(_vn2, t0, t1) >= 0.25 * max(1e-6, t1 - t0)
                try:
                    secs = _sx_movement_numbering(secs, title, lyric_chunks, total_duration,
                                                  remap=_remap, stem_notes_ok=_sung)  # dev26
                except Exception:
                    pass
            if songformer_sections and SONGFORMER_CORRECT_STRUCTURE:
                secs = _songformer_check_map(                                  # pre59
                    secs, songformer_sections, beats=grid.get("beats"), bpm=bpm,
                    lyric_chunks=lyric_chunks, dp_starts=_sx_details.get("dp_starts"),
                    total_duration=total_duration,
                    instrumental=bool(instrumental) or bool(songformer_unsung),
                    dance=bool(songformer_dance), tag_lyrics=tag_lyrics,
                    vocal_notes=_sx_vocal_notes_of(stem_notes))
                for x in secs:
                    x.pop(_SF_LOCK, None)
            if STRUCTURE_VERSE_HALVES_AS_PARTS and not _sx_sheet_marks_verses(tag_lyrics):
                try:
                    secs = _sx_number_verse_halves(secs)   # dev29
                except Exception:
                    pass
            if STRUCTURE_VERSE_PARTS:
                secs = _sx_split_verse_parts(secs)     # dev28
            if (STRUCTURE_TRAILING_SILENCE_JOINS_LAST and len(secs) >= 2
                    and _sx_base_name(secs[-1].get("name")) == "Silence"
                    and float(secs[-1]["end"]) - float(secs[-1]["start"]) <= float(STRUCTURE_TRAILING_SILENCE_MAX_S)):
                # dev29: see STRUCTURE_TRAILING_SILENCE_JOINS_LAST (off by default)
                secs = [dict(x) for x in secs]
                secs[-2]["end"] = secs[-1]["end"]
                secs = secs[:-1]
            if STRUCTURE_MID_SONG_SILENCE:
                secs = _sx_mid_song_silence(secs, grid.get("beatfeat"))       # pre59
            if STRUCTURE_SPLIT_CHORUS_REPEATS:
                secs = _sx_split_chorus_repeats(secs, grid.get("beatfeat"))   # pre64
            secs = _sx_role_labels_only(secs, lyric_chunks, tag_lyrics, bool(instrumental))
            return _annotate_section_name_checks(secs, None if instrumental else stem_notes), "measured"
        if dsp_sections and grid.get("beatfeat") is not None:
            dsp = _normalize_structure_sections(list(dsp_sections), total_duration, min_section_s=2.0)
            if dsp:
                return _annotate_section_name_checks(dsp, None if instrumental else stem_notes), "dsp"
    return _legacy_resolve_final_structure(
        moss_sections, dsp_sections=dsp_sections, lyric_groups=lyric_groups,
        lyric_chunks=lyric_chunks, total_duration=total_duration, grid=grid,
        stem_notes=stem_notes, bpm=bpm,
    )


def _legacy_resolve_final_structure(
    moss_sections, dsp_sections=None, lyric_groups=None, lyric_chunks=None,
    total_duration=None, grid=None, stem_notes=None, bpm=None,
):
    """Previous final-map resolver (fallback for structure v2).

    The audio model's own STRUCTURE output is PRIMARY. It listens to the whole
    recording and understands song form; signal processing does not. Earlier
    revisions of this pipeline discarded that output and rebuilt the map from a
    timbre-novelty curve, which reliably produced boundaries that were precise
    but in the wrong places -- sections a listener would never hear, and
    recurring parts split or merged according to spectral similarity rather
    than musical function. Measured accurately, but measuring the wrong thing.

    So the division of labour here is:

      - The audio model decides WHERE THE SECTIONS ARE and what they are called.
      - Signal analysis SHARPENS each boundary it already agrees with, moving it
        onto a detected change point or a bar line, but never far enough to
        relocate it. Arrangement changes measured from the stems (instruments
        entering and leaving) lead that snap, ahead of timbre novelty, because
        they are what a listener actually hears as the section turning over.
      - Lyric repetition CORRECTS NAMES where two sections deliver the same
        words but were labelled differently.
      - The DSP map is used outright only when the model gave nothing usable.

    Returns (sections, source) where source is one of
    "model+refined" / "model" / "dsp" / "".
    """
    grid = grid or {}
    beats = grid.get("beats") or []
    downbeats = grid.get("downbeats") or []
    beat_len = grid.get("beat_len")

    moss_sections = _normalize_structure_sections(
        moss_sections or [], total_duration, min_section_s=2.0
    )

    if not _structure_sections_are_sane(moss_sections, total_duration):
        # The model gave nothing usable, so the acoustic map is all there is.
        # Run the lyric/MIDI refinement over it here -- this is the one place
        # that pass earns its keep, because without the model's labels the DSP
        # map has only cluster letters to name sections with, and lyric
        # repetition is a far better guide to which parts are the same.
        dsp = list(dsp_sections or [])
        if dsp and (lyric_chunks or lyric_groups):
            try:
                refined = _refine_structure_with_evidence(
                    dsp,
                    lyric_groups=lyric_groups,
                    lyric_chunks=lyric_chunks,
                    total_duration=total_duration,
                    beat_len=beat_len, beats=beats, downbeats=downbeats,
                )
                if refined:
                    dsp = refined
            except Exception:
                pass
        dsp = _normalize_structure_sections(dsp, total_duration)
        if dsp:
            return _annotate_section_name_checks(dsp, stem_notes), "dsp"
        return [], ""

    dsp_bounds = [float(s["start"]) for s in (dsp_sections or [])[1:]]

    # Instruments entering/leaving. Split by corroboration: several stems
    # moving together is a section change, one stem moving alone usually isn't.
    strong_changes, weak_changes = [], []
    try:
        for c in _arrangement_change_points(
            stem_notes, bpm=bpm or (60.0 / beat_len if beat_len else None),
            total_duration=total_duration,
        ):
            if c["n_stems"] >= int(ARRANGEMENT_STRONG_MIN_STEMS):
                strong_changes.append(c["time"])
            else:
                weak_changes.append(c["time"])
    except Exception:
        strong_changes, weak_changes = [], []

    # Cap the shift at roughly one bar. Beyond that a "correction" stops being a
    # correction and starts moving the boundary to a different musical event.
    max_shift = (beat_len * 4.0) if beat_len else 2.5
    max_shift = float(min(max(max_shift, 1.0), 3.0))

    out = [dict(s) for s in moss_sections]
    for i in range(1, len(out)):
        original = float(out[i]["start"])
        sharpened = _sharpen_boundary(
            original, dsp_bounds, downbeats, beats, beat_len, max_shift,
            strong_changes=strong_changes, weak_changes=weak_changes,
        )
        # Never let a nudge cross a neighbouring boundary or reorder the map.
        if i + 1 < len(out) and sharpened >= float(out[i + 1]["start"]) - 2.0:
            sharpened = original
        # Keep sections in order and non-degenerate after the nudge.
        prev_start = float(out[i - 1]["start"])
        nxt_end = float(out[i]["end"])
        if sharpened <= prev_start + 2.0 or sharpened >= nxt_end - 2.0:
            sharpened = original
        out[i]["start"] = round(sharpened, 2)
        out[i - 1]["end"] = round(sharpened, 2)

    out = _unify_labels_by_lyrics(out, lyric_chunks, lyric_groups)
    out = _normalize_structure_sections(out, total_duration, min_section_s=2.0)
    out = _annotate_section_name_checks(out, stem_notes)
    return out, "model+refined"


def _refine_structure_with_evidence(
    sections, lyric_groups=None, lyric_chunks=None, stem_notes=None,
    total_duration=None, bpm=None, beat_len=None, beats=None, downbeats=None,
):
    """Fuse lyric repetition and per-stem MIDI activity into the acoustic map.

    The acoustic pass that produces `sections` runs before the lyrics are
    transcribed and before the stems are converted to MIDI, so it has never
    seen either. That is a real limitation rather than a detail: timbre-and-
    harmony novelty is blind to exactly the changes that matter most in vocal
    music. A chorus entering on the same pad, key and drum pattern as the verse
    produces almost no acoustic novelty -- but the words change, and change
    back again next time that chorus returns.

    Two jobs:

      1. BOUNDARIES -- lyric block edges lead, since they mark the change
         directly. Acoustic boundaries fill the gaps (instrumental passages,
         intros, breakdowns, where there are no words to go on). MIDI activity
         changes are taken only where a lyric edge agrees, so ordinary fills
         aren't promoted to sections. Acoustic boundaries are never moved.
      2. LABELS -- named from what the words do. Blocks made of the same words
         are the same section returning; the words that return most are the
         chorus; a repeated block that always leads into the chorus is a
         pre-chorus. Observation, not an assumption about song form, so it
         holds on tracks that don't follow the usual template.

    Returns a new section list; the input is not modified.
    """
    if not sections:
        return sections

    duration = float(total_duration or (sections[-1]["end"] if sections else 0.0))
    if duration <= 0:
        return sections

    bar_len = None
    if beat_len:
        bar_len = float(beat_len) * 4.0
    elif bpm:
        try:
            bar_len = (60.0 / float(bpm)) * 4.0
        except Exception:
            bar_len = None

    min_section = max(7.0, (bar_len * 2.0) if bar_len else 7.0)
    # Boundary spacing and minimum section length are deliberately the SAME
    # number. When they differ, a boundary can pass the spacing test and then
    # be silently dropped by the section-length filter further down, so the
    # evidence that produced it is discarded without anything being logged and
    # the two thresholds disagree about what counts as a section.
    min_gap = min_section

    blocks = _lyric_block_segmentation(lyric_chunks, lyric_groups)
    if not blocks:
        # Instrumental, or the lyrics pass found nothing. There is no lyric
        # evidence to fuse and MIDI candidates are only accepted where a lyric
        # edge agrees, so this pass has nothing to contribute. Return the
        # acoustic map untouched rather than relabelling its sections from
        # position alone -- doing that would throw away the audio model's own
        # names on exactly the tracks where they are the only naming evidence.
        return sections
    lyric_bounds = [b["start"] for b in blocks]

    midi_cands = []
    try:
        if stem_notes:
            midi_cands = _midi_activity_boundary_candidates(
                stem_notes, duration=duration
            ) or []
    except Exception:
        midi_cands = []

    acoustic = [float(s["start"]) for s in sections[1:]]

    chosen = []

    def _accept(t):
        t = float(t)
        if t < min_section or t > duration - min_section:
            return
        if any(abs(t - c) < min_gap for c in chosen):
            return
        chosen.append(t)

    for t in sorted(lyric_bounds):
        _accept(t)
    for t in sorted(acoustic):
        _accept(t)

    # MIDI note-density changes are only consulted where there are no words to
    # go on -- instrumental intros, breakdowns, solos, outros. That is the one
    # place they carry information the other two sources don't have.
    #
    # The previous rule required a MIDI candidate to sit within 3s of a lyric
    # edge, which made the branch unreachable: lyric edges are accepted first,
    # and the spacing test then rejects anything within min_gap (>= 6s) of one.
    # Every candidate that could pass the gate was guaranteed to fail the
    # spacing test, so MIDI evidence had no effect on any track. Requiring the
    # opposite -- distance from all sung material -- both makes the branch live
    # and points it at the sections it can actually help with.
    def _in_instrumental_region(t):
        for blk in blocks:
            if (blk["start"] - 4.0) <= t <= (blk["end"] + 4.0):
                return False
        return True

    for t in sorted(midi_cands):
        if _in_instrumental_region(t):
            _accept(t)

    if chosen and (beats or downbeats):
        snapped = []
        for t in chosen:
            if any(abs(t - a) < 0.01 for a in acoustic):
                snapped.append(t)          # already bar-aligned
            else:
                snapped.append(_snap_boundaries_to_metre(
                    [t], downbeats or [], beats or [], beat_len
                )[0])
        chosen = snapped

    bounds = sorted(set([0.0] + [round(float(t), 2) for t in chosen] + [duration]))
    clean = [bounds[0]]
    for b in bounds[1:]:
        if b - clean[-1] >= min_section:
            clean.append(b)
    if clean[-1] < duration - min_section:
        clean.append(duration)
    else:
        clean[-1] = duration

    spans = [(clean[i], clean[i + 1]) for i in range(len(clean) - 1)]
    if not spans:
        return sections

    def _cluster_for(start, end):
        best, best_overlap = "", 0.0
        for s in sections:
            ov = min(end, float(s["end"])) - max(start, float(s["start"]))
            if ov > best_overlap:
                best_overlap, best = ov, s.get("cluster", "")
        return best

    clusters = [_cluster_for(a, b) for a, b in spans]

    # Each section takes the signature of the lyric block it overlaps most.
    sec_sig, sung = [], []
    for (a, b) in spans:
        best_sig, best_ov, words_ov = None, 0.0, 0.0
        for blk in blocks:
            ov = min(b, blk["end"]) - max(a, blk["start"])
            if ov <= 0:
                continue
            words_ov += ov
            if ov > best_ov:
                best_ov, best_sig = ov, blk["signature"]
        # A block must genuinely occupy the section before it names it. Without
        # this, an intro that merely abuts the first chorus picks up a one-second
        # overlap and gets labelled "Chorus" -- the section is then reported as
        # starting a full section early, which is exactly the kind of drift that
        # makes a whole section map look shifted.
        span_len = max(1e-6, b - a)
        if best_ov < 0.35 * span_len:
            best_sig = None
        sung.append((words_ov / span_len) >= 0.25)
        sec_sig.append(best_sig)

    counts = {}
    for s in sec_sig:
        if s:
            counts[s] = counts.get(s, 0) + 1
    repeated = [s for s, c in counts.items() if c >= 2]

    chorus_sig = None
    if repeated:
        def _score(sig):
            idxs = [i for i, s in enumerate(sec_sig) if s == sig]
            return (counts[sig], sum(spans[i][1] - spans[i][0] for i in idxs))
        chorus_sig = max(repeated, key=_score)

    prechorus_sig = None
    for sig in repeated:
        if sig == chorus_sig:
            continue
        idxs = [i for i, s in enumerate(sec_sig) if s == sig]
        if idxs and all(i + 1 < len(spans) and sec_sig[i + 1] == chorus_sig
                        for i in idxs):
            prechorus_sig = sig
            break

    # Lyric-signature matching requires an EXACT repeated-line-group match, so
    # a chorus that differs from its own earlier occurrence by even one
    # transcribed word (an ad-lib, a missed word, a slightly different take)
    # never matches and chorus_sig stays None. When that happens every sung
    # section falls through to "Verse" below regardless of how clearly it
    # actually repeats -- which is worse than useless, since it hides a real,
    # audible chorus behind a flat list of identically-named verses. The
    # acoustic cluster already computed for each span (same material,
    # independent of what was transcribed) is a robust fallback signature for
    # exactly this case, so it is only used when the lyric-based pass found
    # nothing to work with.
    sec_key = sec_sig
    if chorus_sig is None:
        alt_sig = [c if c and c != "Unique" else None for c in clusters]
        alt_counts = {}
        for s in alt_sig:
            if s:
                alt_counts[s] = alt_counts.get(s, 0) + 1
        alt_repeated = [s for s, c in alt_counts.items() if c >= 2]
        if alt_repeated:
            def _alt_score(sig):
                idxs = [i for i, s in enumerate(alt_sig) if s == sig]
                return (alt_counts[sig], sum(spans[i][1] - spans[i][0] for i in idxs))
            chorus_sig = max(alt_repeated, key=_alt_score)
            for sig in alt_repeated:
                if sig == chorus_sig:
                    continue
                idxs = [i for i, s in enumerate(alt_sig) if s == sig]
                if idxs and all(i + 1 < len(spans) and alt_sig[i + 1] == chorus_sig
                                for i in idxs):
                    prechorus_sig = sig
                    break
            repeated = alt_repeated
            sec_key = alt_sig

    n = len(spans)
    names = []
    for i, (start, end) in enumerate(spans):
        sig = sec_key[i]
        if sig is not None and sig == chorus_sig:
            names.append("Chorus")
        elif sig is not None and sig == prechorus_sig:
            names.append("Pre-Chorus")
        elif sig is not None and sig in repeated:
            names.append("Refrain")
        elif sung[i]:
            names.append("Verse")
        elif i == 0:
            names.append("Intro")
        elif i == n - 1:
            names.append("Outro")
        else:
            names.append("Instrumental")

    for i in range(1, n - 1):
        if names[i] == "Verse" and (i / max(1, n - 1)) > 0.5:
            if names[i - 1] in ("Chorus", "Instrumental") and \
               names[i + 1] in ("Chorus", "Outro", "Instrumental"):
                names[i] = "Bridge"

    totals = {}
    for nm in names:
        totals[nm] = totals.get(nm, 0) + 1
    seen = {}
    out = []
    for i, (start, end) in enumerate(spans):
        nm = names[i]
        if totals[nm] > 1:
            seen[nm] = seen.get(nm, 0) + 1
            nm = "%s %d" % (nm, seen[nm])
        out.append({
            "name": nm, "start": float(start), "end": float(end),
            "cluster": clusters[i], "sung": bool(sung[i]),
        })

    return _normalize_structure_sections(
        out, duration, min_section_s=min(6.0, min_section)
    )


def _build_objective_first_structure(
    resolved_path, y, sr, total_duration, moss_model, moss_processor,
    bpm_hint=None, lyric_groups=None, stem_notes=None,
    details_out=None, instrumental=False,
):
    """Section map measured from the audio, named from measured evidence.

    Runs build_structure_map (structure v2) on the mix. `details_out`, when
    given, receives the beat grid ("beats", "downbeats", "beat_len") and the
    cached beat-synchronous features ("beatfeat") so the final pass in
    _resolve_final_structure can re-segment with stem notes and lyrics
    without decoding the audio again. The audio model, when loaded, is asked
    to name the measured parts; its names are kept only as hints for unsung
    parts (solo, breakdown, interlude...). Falls back to the previous
    pipeline if v2 is disabled or fails.

    Returns a list of {"name", "start", "end", "cluster"} dicts.
    """
    if details_out is None:
        details_out = {}
    sections = []
    if STRUCTURE_V2_ENABLED:
        try:
            sections = build_structure_map(
                y, sr, total_duration, bpm_hint=bpm_hint,
                stem_notes=stem_notes, instrumental=bool(instrumental),
                details_out=details_out,
            )
        except Exception as e:
            print(f"  (structure v2 failed, using previous method: {type(e).__name__}: {e})")
            sections = []
    if not sections:
        return _legacy_objective_first_structure(
            resolved_path, y, sr, total_duration, moss_model, moss_processor,
            bpm_hint=bpm_hint, lyric_groups=lyric_groups, stem_notes=stem_notes,
        )
    hints = _sx_moss_name_hints(sections, resolved_path, moss_model, moss_processor)
    details_out["moss_hints"] = hints
    return _sx_apply_moss_hints(sections, hints)


def _legacy_objective_first_structure(
    resolved_path, y, sr, total_duration, moss_model, moss_processor,
    bpm_hint=None, lyric_groups=None, stem_notes=None,
):
    """Previous objective-first structure (fallback for structure v2):
    cut -> cluster -> merge -> name.

    Timestamps come ONLY from signal analysis of this recording (beat/bar grid
    plus a beat-synchronous novelty curve, optionally corroborated by recurring
    lyric block starts and per-stem note-density changes). The audio model is
    asked only to NAME parts whose boundaries are already fixed, and its names
    are discarded if they contradict the measured recurrence pattern. Nothing
    here consults Wikipedia, file tags, or general knowledge about the song.

    Returns a list of {"name", "start", "end", "cluster"} dicts.
    """
    # ---- 1. Objective change points, bar-aligned -----------------------
    raw_bounds, grid = _structural_novelty_boundaries(
        y, sr, bpm_hint=bpm_hint, return_details=True
    )
    beats = grid.get("beats") or []
    downbeats = grid.get("downbeats") or []
    beat_len = grid.get("beat_len")
    beats_per_bar = grid.get("beats_per_bar") or 4
    bar_len = (beat_len * beats_per_bar) if beat_len else None

    # ---- 2. Corroborating evidence from other measured sources ---------
    # A lyric block start or a note-density jump is only allowed to ADD a
    # boundary where the acoustic curve missed one (a same-timbre chorus entry
    # is the classic case). It is snapped to the same bar grid so it cannot
    # introduce an off-grid timestamp.
    extra = []
    try:
        if lyric_groups:
            extra.extend(_lyric_block_boundary_candidates(
                lyric_groups, duration=total_duration
            ) or [])
    except Exception:
        pass
    try:
        if stem_notes:
            extra.extend(_midi_activity_boundary_candidates(
                stem_notes, duration=total_duration
            ) or [])
    except Exception:
        pass

    if extra:
        extra = _snap_boundaries_to_metre(extra, downbeats, beats, beat_len)
        min_gap = max(12.0, (bar_len * 4.0) if bar_len else 12.0)
        for t in extra:
            if t < 5.0 or t > float(total_duration) - 5.0:
                continue
            if all(abs(t - b) >= min_gap for b in raw_bounds):
                raw_bounds.append(t)
        raw_bounds = sorted(raw_bounds)

    # ---- 3. Assemble segments with a musical minimum length ------------
    # Two bars minimum, floored at 8 s: below that it is a fill or a turnaround,
    # not a section, and letting those through is what used to fragment the map.
    min_section = max(8.0, (bar_len * 2.0) if bar_len else 8.0)

    bounds = [0.0] + [float(b) for b in raw_bounds] + [float(total_duration)]
    bounds = sorted(set(round(b, 2) for b in bounds))

    clean = [bounds[0]]
    for b in bounds[1:]:
        if b - clean[-1] >= min_section:
            clean.append(b)
    if len(clean) < 2:
        clean = [0.0, float(total_duration)]
    elif clean[-1] < float(total_duration) - min_section:
        clean.append(float(total_duration))
    else:
        clean[-1] = float(total_duration)

    # ---- 4. Cluster, then merge adjacent identical material -------------
    # Re-clustered after every merge pass: merging changes the segments, so the
    # fingerprints must be recomputed rather than carried over. The old code
    # merged once using stale labels and then appended a duplicate trailing
    # label, leaving names misaligned with the segments they described.
    cluster_labels = _cluster_audio_segments(y, sr, clean)

    for _pass in range(3):
        if len(clean) <= 2 or not cluster_labels:
            break
        merged = [clean[0]]
        changed = False
        i = 0
        while i < len(clean) - 1:
            j = i
            while (
                j + 1 < len(clean) - 1
                and cluster_labels[j] != "Unique"
                and j + 1 < len(cluster_labels)
                and cluster_labels[j + 1] == cluster_labels[j]
            ):
                j += 1
                changed = True
            merged.append(clean[j + 1])
            i = j + 1
        if not changed:
            break
        clean = merged
        cluster_labels = _cluster_audio_segments(y, sr, clean)

    n_segments = len(clean) - 1
    if n_segments < 1:
        return []
    while len(cluster_labels) < n_segments:
        cluster_labels.append("Unique")
    cluster_labels = cluster_labels[:n_segments]

    spans = [(clean[i], clean[i + 1]) for i in range(n_segments)]

    # ---- 5. Ask the audio model to NAME the fixed parts ------------------
    llm_labels = {}
    if moss_model is not None and moss_processor is not None and n_segments >= 2:
        try:
            seg_list = []
            for i, (start, end) in enumerate(spans):
                cl = cluster_labels[i]
                seg_list.append(
                    "Part %d (%s - %s, %.0fs): Acoustic Cluster '%s'"
                    % (i + 1, _seconds_to_mmss(start), _seconds_to_mmss(end),
                       end - start, cl)
                )
            prompt = STRUCTURE_LABELING_PROMPT.format(
                segment_list="\n".join(seg_list)
            )
            conversation = [{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "audio", "path": resolved_path},
                ],
            }]
            status("Naming objective structure segments with the audio model...")
            llm_response = moss_generate(
                moss_model, moss_processor, conversation, max_new_tokens=384
            )
            llm_labels = _parse_structure_labels(llm_response, n_segments)
        except Exception as e:
            print("  (structure naming pass skipped: %s)" % e)
            llm_labels = {}

    # Per-segment loudness, used only to break verse/chorus ties structurally.
    seg_energy = []
    for (start, end) in spans:
        try:
            chunk = y[int(start * sr):int(end * sr)]
            seg_energy.append(
                float(np.sqrt(np.mean(np.square(chunk)))) if chunk.size else None
            )
        except Exception:
            seg_energy.append(None)

    # Per-section vocal coverage, so a voiceless stretch is named for what it
    # is instead of being forced into the verse/chorus vocabulary. None when
    # there is no usable vocal stem, which leaves labelling exactly as before.
    vocal_coverage = None
    try:
        vocal_coverage = _vocal_coverage_by_span(stem_notes, spans)
    except Exception:
        vocal_coverage = None

    auto_names = _auto_label_sections_from_clusters(
        cluster_labels, spans, total_duration, energies=seg_energy,
        vocal_coverage=vocal_coverage,
    )

    use_llm = (
        len(llm_labels) >= max(2, int(n_segments * 0.6))
        and _labels_respect_clusters(llm_labels, cluster_labels)
    )

    sections = []
    seen = {}
    for i, (start, end) in enumerate(spans):
        if use_llm and (i + 1) in llm_labels:
            name = llm_labels[i + 1]
        else:
            name = auto_names[i] if i < len(auto_names) else "Section %d" % (i + 1)
        # Guarantee unique display names without renaming the musical function.
        key = name.lower()
        seen[key] = seen.get(key, 0) + 1
        if seen[key] > 1 and not re.search(r"\d$", name):
            name = "%s %d" % (name, seen[key])
        sections.append({
            "name": name,
            "start": float(start),
            "end": float(end),
            "cluster": cluster_labels[i],
        })

    return _normalize_structure_sections(
        sections, total_duration, min_section_s=min(6.0, min_section)
    )


def _overwrite_structure_field(text, new_sections):
    """Surgically overwrites the STRUCTURE=[...] field in the analysis text
    with an unambiguous bracketed format the writer LLM cannot misparse."""
    if not new_sections:
        return text
    # Use a format where each section is on its own line inside brackets,
    # with explicit "Name | start-end" separation so there's zero ambiguity
    # between section names (which can contain spaces, hyphens, numbers)
    # and the timestamps.
    parts = []
    for s in new_sections:
        parts.append(f"  {s['name']} | {_seconds_to_mmss(s['start'])}-{_seconds_to_mmss(s['end'])}")
    struct_str = "\n".join(parts)
    new_struct_field = f"STRUCTURE=[\n{struct_str}\n]"

    # Match existing STRUCTURE field (bracketed or unbracketed, single or multi-line)
    pattern = re.compile(
        r"(?i:STRUCTURE)\s*[=:]\s*\[.*?\](?=\n\s*[A-Z][A-Z0-9_]{2,}\s*[=:]|\n\n|\Z)",
        re.DOTALL,
    )
    if pattern.search(text):
        return pattern.sub(new_struct_field, text)

    # Fallback: unbracketed single/multi-line STRUCTURE field
    pattern2 = re.compile(
        r"(?i:STRUCTURE)\s*[=:]\s*[^\n]+(?:\n(?![A-Z][A-Z0-9_]{2,}\s*[=:]|\n)[^\n]+)*",
        re.DOTALL,
    )
    if pattern2.search(text):
        return pattern2.sub(new_struct_field, text)

    # If no STRUCTURE field exists at all, append one
    return text.rstrip() + "\n\n" + new_struct_field + "\n"


def _finalize_structure_sections(
    analysis_text,
    objective_report=None,
    stem_midi_report=None,
    lyric_chunks=None,
    total_duration=None,
):
    """Parse listening-model STRUCTURE, snap edges with tiered objective evidence.

    Conservative by design: no inventing extra section cuts from every lyric
    line or MIDI density blip. Acoustic novelty first; lyric block starts only
    when they explain a same-timbre transition the novelty curve missed.
    """
    lyric_groups = _detect_lyric_repetitions(lyric_chunks) if lyric_chunks else None
    sections = extract_structure_sections(analysis_text)
    sources = _collect_structure_boundaries(
        objective_report=objective_report,
        stem_midi_report=stem_midi_report,
        lyric_groups=lyric_groups,
        total_duration=total_duration,
    )
    # Flat list for reporting / fallback formatting
    all_bounds = _merge_boundary_candidates(
        [sources["acoustic"], sources["lyric"], sources["midi"]],
        min_spacing_s=5.0,
        duration=total_duration,
        max_boundaries=20,
    )
    sections = _snap_structure_sections_tiered(
        sections,
        acoustic_bounds=sources["acoustic"],
        lyric_bounds=sources["lyric"],
        midi_bounds=sources["midi"],
        total_duration=total_duration,
    )
    sections = _annotate_structure_with_lyric_blocks(sections, lyric_groups)
    return sections, all_bounds, lyric_groups


def format_recommended_structure_block(sections, objective_boundaries=None, total_duration=None,
                                       lyric_groups=None, objective_report=None):
    """Authoritative structure for the writer (named labels when possible).
    lyric_groups (optional): output of _detect_lyric_repetitions() -- used to
    annotate sections whose lyrics recur elsewhere in the track, which is
    the strongest available signal for correctly identifying recurring
    sections (chorus/refrain/hook) on songs that don't follow a standard
    intro/verse/chorus/verse/chorus/bridge/chorus template.

    objective_report (optional): the OBJECTIVE AUDIO MEASUREMENTS text. Used
    only as a last-ditch source of change points when the caller passed none.
    Every caller is supposed to supply objective_boundaries, but they derive
    them several different ways -- section starts, a separate finalize pass,
    a grid resolver -- and any of those coming back empty used to drop
    straight through to "SECTION MAP UNAVAILABLE" even when the objective
    report sitting right there had a perfectly good candidate list. Re-reading
    it here means one caller-side gap can no longer cost the writer its whole
    section map.
    """
    sections = _normalize_structure_sections(sections or [], total_duration)

    if not objective_boundaries and objective_report:
        try:
            objective_boundaries = extract_objective_structure_boundaries(objective_report)
        except Exception:
            objective_boundaries = objective_boundaries or []

    def _lyric_note(start, end):
        if not lyric_groups:
            return ""
        for g in lyric_groups:
            others = [t for t, _e in g["occurrences"] if not (start <= t < end)]
            in_range = any(start <= t < end for t, _e in g["occurrences"])
            if in_range and others:
                times = ", ".join(_seconds_to_mmss(t) for t in others)
                return f"  [lyrics also recur at {times} -- same recurring section]"
        return ""

    if sections:
        lines = ["RECOMMENDED STRUCTURE FOR DISCUSSION:"]
        lines.append(
            "(Use these EXACT section names and timestamps. Do not invent, alter, or omit "
            "any timestamp. Quote them to the second exactly as written — do NOT round to "
            "the nearest 5 or 10 seconds: 1:34 is 1:34, not 1:35 or 1:30.)"
        )
        for sec in sections:
            start_str = _seconds_to_mmss(sec['start'])
            end_str = _seconds_to_mmss(sec['end'])
            lyric_note = _lyric_note(sec['start'], sec['end'])
            check = sec.get("name_check")
            check_note = f"  [name check: {check}]" if check else ""
            lines.append(f"  {sec['name']}: {start_str} to {end_str}{lyric_note}{check_note}")
        lines.append(
            "This is the section map for this track. The BOUNDARIES are measured from the "
            "audio (bar tracking, change detection, drum fills/breaks, vocal entries and "
            "lyric timing), so quote the times verbatim when asked. The NAMES are inferred "
            "from measured evidence -- which parts repeat, where the voice is, how high it "
            "sits, how loud the part is and which words come back -- not from a template, so "
            "a song that opens on its chorus is named that way. They are usually right but are "
            "still an interpretation. A section named 'Silence' is digital silence in the file "
            "(nothing plays there) -- say the music ends at that point rather than describing "
            "an outro over it. Where a line carries a "
            "[name check: ...] note, the measured evidence conflicts with the name -- describe "
            "what actually happens in that span (e.g. 'the vocal drops out and the guitar takes "
            "over') instead of asserting the name, and do not build claims such as 'the song "
            "opens on a pre-chorus' on a flagged name. Do NOT volunteer these times: describe "
            "structure by section name and order unless the user actually asks when something "
            "happens or asks for a breakdown. The times are here so they are exact WHEN wanted, "
            "not so they are recited by default. "
            "Do not say the section map itself is missing or unavailable — it is right here. "
            "You may note, once, that the map is detected automatically and the times can be "
            "a few seconds out; that is about precision, not availability, and is never a "
            "reason to withhold or blur a number the user asked for. "
            "For instrumental or non-standard forms, neutral names (Section 1, Theme A, Development) "
            "are valid — do not force verse/chorus language. Where a section is marked with "
            "'lyrics also recur at ...', that section shares its material with the section(s) at "
            "those other timestamps (e.g. every occurrence of a chorus/refrain/hook) even if the "
            "overall form is not a standard verse/chorus alternation — treat those timestamps as "
            "the same recurring section, not separate ones."
        )
        return _structure_footer_view("\n".join(lines))

    # Fallback: objective peaks only (instrumentals / unparseable STRUCTURE)
    if objective_boundaries and total_duration is not None and float(total_duration) > 0:
        peaks = sorted(
            float(t) for t in objective_boundaries
            if 0.5 < float(t) < float(total_duration) - 0.5
        )
        boundaries = [0.0] + peaks + [float(total_duration)]
        deduped = [boundaries[0]]
        for t in boundaries[1:]:
            if t - deduped[-1] >= 2.0:
                deduped.append(t)
        if deduped[-1] < float(total_duration) - 0.5:
            deduped.append(float(total_duration))
        else:
            deduped[-1] = float(total_duration)

        lines = ["RECOMMENDED STRUCTURE FOR DISCUSSION (from objective change points):"]
        n = 0
        for i in range(len(deduped) - 1):
            start, end = deduped[i], deduped[i + 1]
            if end - start < 2.0:
                continue
            n += 1
            lines.append(f"Section {n}: {_seconds_to_mmss(start)}–{_seconds_to_mmss(end)}")
        if n >= 1:
            lines.append(
                "These timestamps come from independent timbre+harmony+energy novelty detection on "
                "THIS recording, then merged and validated. This IS the authoritative section map "
                "for this track and it IS a usable timestamped structure — the earlier "
                "'raw, unmerged DSP peaks' warning in the measurements refers to the raw candidate "
                "list there, NOT to this block. Do not tell the user the structure is unavailable, "
                "unconfirmed, or unreliable, and do not refuse to give timestamps because this "
                "block has neutral names. Quote these times exactly as written."
            )
            lines.append(
                "What this block does NOT give you is section NAMES. The boundaries are measured; "
                "the labels are not. So: describe what happens at and between these times, and use "
                "verse/chorus/bridge wording only where the rest of the analysis clearly supports "
                "it — otherwise keep the neutral Section N names (also the right choice for "
                "instrumentals and through-composed pieces). Saying 'the track changes at 1:02' is "
                "always correct here; saying 'the chorus starts at 1:02' needs support elsewhere."
            )
            return "\n".join(lines)

    # Last-resort fallback: every detector above came back empty (no parsed
    # STRUCTURE=[...] field, no acoustic novelty boundaries, no lyric/MIDI
    # boundaries). Rather than silently omitting the block entirely -- which
    # leaves the writer with literally no timestamped map and forces it to
    # say "I don't have a section map for this track" -- carve the track
    # into evenly spaced coarse windows so there's always SOMETHING concrete
    # to answer "what happens around X:XX" or "in the Y:YY-Z:ZZ range" with.
    # These are explicitly labelled as coarse/non-arrangement-based so the
    # writer doesn't misrepresent them as detected verse/chorus boundaries.
    if total_duration is not None and float(total_duration) > 8.0:
        # No detector found a usable boundary. This used to emit an even split of
        # the track into numbered "Part N: 0:00 to 0:20" windows, on the theory
        # that some timestamped map was better than none. In practice it was
        # worse: the windows look exactly like a real section list, and the
        # writer reliably relabelled them as Intro/Verse/Chorus and presented
        # arithmetic fractions of the runtime to the user as detected
        # arrangement changes -- confidently wrong timestamps at suspiciously
        # round intervals, which is far more misleading than admitting the map
        # is missing. So: state the absence plainly and give nothing that can be
        # mistaken for measured sections.
        return (
            "SECTION MAP UNAVAILABLE FOR THIS TRACK:\n"
            "No reliable arrangement boundaries could be measured for this recording. "
            "There is NO section map. If the user asks about structure, sections, or what "
            "happens at a particular timestamp, say plainly that the arrangement didn't "
            "come through clearly on this one and describe the track in general terms "
            "instead. Do NOT invent section names, do NOT estimate timestamps, and do NOT "
            "produce a verse/chorus breakdown from your own knowledge of the song or from "
            "how songs of this type are usually arranged. Round, evenly-spaced timings "
            "(0:30, 1:00, 1:30) are a sign you are guessing -- don't."
        )

    return ""


# dev40: the structure block's timing sentence. A question about the structure
# itself was read as "not asking for times".
_STRUCTURE_TIMES_OLD = (
    "Do NOT volunteer these times: describe structure by section name and order unless the "
    "user actually asks when something happens or asks for a breakdown. The times are here so "
    "they are exact WHEN wanted, not so they are recited by default. "
)
_STRUCTURE_TIMES_NEW = (
    "Do NOT volunteer these times in answers about other things. A question about the song's "
    "structure, form, sections or layout (\"what is the structure?\", \"how is it laid out?\") "
    "IS a request for this map: list every section in order with its name and its start and "
    "end time, exactly as written above. Otherwise give times only when the user asks when "
    "something happens, how long a part is, or for a breakdown; they are here so they are "
    "exact when wanted, not recited by default. "
)


# dev49: the structure block's own caveat sentence (both generators).
_STRUCTURE_CAVEAT_OLD = (
    "You may note, once, that the map is detected automatically and the times can be "
    "a few seconds out; that is about precision, not availability,"
)
_STRUCTURE_CAVEAT_NEW = (
    "When you give the map, note once, briefly, that it is detected automatically: the "
    "times can be a few seconds out and the section names are inferred, so a part can be "
    "mislabelled (a verse and a chorus swapped, say); that is about precision, not "
    "availability,"
)


def _structure_caveat_view(text):
    if STRUCTURE_NAMES_CAVEAT and text:
        return text.replace(_STRUCTURE_CAVEAT_OLD, _STRUCTURE_CAVEAT_NEW)
    return text


def _structure_footer_view(text):
    if STRUCTURE_ANSWERS_WITH_TIMES and text:
        text = text.replace(_STRUCTURE_TIMES_OLD, _STRUCTURE_TIMES_NEW)
    return _structure_caveat_view(text)


def _cp_section_seq(segs, a, b, beat):
    """[(chord, duration)] in time order inside [a, b): repeats merged, slivers dropped."""
    pieces = []
    for s, e, ch in segs:
        if e <= a or s >= b:
            continue
        s2, e2 = max(s, a), min(e, b)
        if e2 - s2 >= 0.15:
            pieces.append([ch, e2 - s2])
    short = max(0.35, 0.5 * beat)
    for _ in range(2):
        merged = []
        for ch, d in pieces:
            if merged and merged[-1][0] == ch:
                merged[-1][1] += d
            else:
                merged.append([ch, d])
        pieces = [p for p in merged if p[1] >= short] if len(merged) > 1 else merged
    merged = []
    for ch, d in pieces:
        if merged and merged[-1][0] == ch:
            merged[-1][1] += d
        else:
            merged.append([ch, d])
    return [(ch, d) for ch, d in merged]


def _structure_field_text(sections):
    parts = [f"  {s_['name']} | {_seconds_to_mmss(s_['start'])}-{_seconds_to_mmss(s_['end'])}"
             for s_ in sections]
    return "STRUCTURE=[\n" + "\n".join(parts) + "\n]"
_PROD_SECTION_TITLES = (
    "STEREO IMAGE / MIX WIDTH (objective)", "MASTERING: TRUE PEAK / CLIPPING (objective)",
    "BANDWIDTH / SOURCE PROVENANCE (objective)", "BEAT-SYNCHRONOUS DYNAMICS / PUMPING (objective)",
    "AMBIENCE / REVERB CHARACTER (objective proxy)", "WHOLE-TRACK SPECTRAL BALANCE (objective)",
    "PRODUCTION-CHARACTER TAGS",
)


def _fixed_blocks_structure_copy(t):
    """The measured section map's second copy (the one below the report) as
    its own STRUCTURE=[...] block right after the era pass, wherever the
    analysis step had put it ("Song STRUCTURE=" inside the era description,
    or after the vocal-onset check)."""
    rn = re.search(r"(?m)^REPORT_NOTE=[^\n]*$", t)
    if not rn:
        return t
    start = rn.end()
    pat = re.compile(r"(?ms)(^(?:Song )?STRUCTURE=\[\n.*?\n\][ \t]*\n?)")
    copies = [m for m in pat.finditer(t, start)]
    if not copies:
        return t
    body = re.sub(r"^Song ", "", copies[0].group(1)).rstrip("\n")
    if any(re.sub(r"^Song ", "", m.group(1)).rstrip("\n") != body for m in copies):
        return t  # copies that disagree are left exactly as they are
    anchor = re.search(r"\n\nVOCAL / SINGER PROFILE", t[start:])
    if anchor is None:
        return t
    out, pos = [], 0
    for m in copies:
        out.append(t[pos:m.start()])
        pos = m.end()
    out.append(t[pos:])
    u = "".join(out)
    u = re.sub(r"\n{3,}", "\n\n", u[start:])
    u = t[:start] + u
    a2 = re.search(r"\n\nVOCAL / SINGER PROFILE", u[start:])
    if a2 is None:
        return t
    k = start + a2.start()
    pre = u[:k].rstrip("\n")
    return pre + "\n\n" + body + u[k:]
_DESC_SECTION_TIME_RE = re.compile(r"\(\s*\d{1,2}:\d{2}\s*[‐-―\-−‑]")


def _snap_boundaries_to_metre(times, downbeats, beats, beat_len):
    """Snap detected change points onto the metrical grid.

    Preference order: bar line first (a section change is a bar change), then
    beat, then leave alone. The tolerance is expressed in beats so it scales
    with tempo instead of being a fixed number of seconds that is far too loose
    at 160 BPM and far too tight at 70 BPM.
    """
    if not times:
        return []
    if not beat_len or beat_len <= 0:
        return [round(float(t), 2) for t in times]

    bar_tol = beat_len * 2.0     # within half a 4/4 bar -> pull to the bar line
    beat_tol = beat_len * 0.6    # otherwise pull to the nearest beat

    out = []
    for t in times:
        snapped = _snap_time_to_grid(t, downbeats, bar_tol)
        if snapped == float(t):
            snapped = _snap_time_to_grid(t, beats, beat_tol)
        out.append(round(float(snapped), 2))

    # De-duplicate after snapping (two nearby peaks can land on one bar line).
    deduped = []
    for t in sorted(out):
        if not deduped or t - deduped[-1] > 0.05:
            deduped.append(t)
    return deduped


def _structural_novelty_boundaries(y, sr, max_boundaries=16, min_spacing_s=5.0,
                                   bpm_hint=None, return_details=False):
    """Objective, signal-only estimate of song-section change points.

    Rewritten to be BEAT-SYNCHRONOUS and BAR-ALIGNED, which is where the
    accuracy of the reported timestamps actually comes from:

      1. Track beats and estimate the bar grid (_beat_and_downbeat_grid).
      2. Aggregate features per BEAT rather than per fixed 0.25 s frame. A
         fixed frame grid smears the self-similarity matrix differently at
         every tempo; beat-synchronous features make one kernel width mean the
         same musical span (bars) on every track, which is the whole reason a
         single hard-coded kernel used to work on some songs and not others.
      3. Build the self-similarity matrix on stacked-memory features so the
         matrix reflects local *context* rather than instantaneous timbre,
         which sharpens the corners at real section changes.
      4. Correlate a checkerboard kernel along the diagonal (Foote 2000) to get
         a novelty curve, then pick peaks by PROMINENCE, not by raw height --
         raw height plus a fixed floor was letting broad humps through and
         dropping genuine but locally-modest transitions.
      5. Snap surviving peaks onto the bar grid.

    Step 5 is the fix for the specific complaint that section timestamps are
    wrong: the change point was often found within a beat or two, but was then
    reported at whatever arbitrary frame the peak sat on. Section changes land
    on bar lines, so a bar-aligned answer is both more accurate and more
    musically usable.

    Returns a list of seconds. With return_details=True, returns
    (times, info_dict) where info carries the beat grid for reuse.
    """
    empty = ([], {}) if return_details else []
    try:
        duration = len(y) / float(sr)
        if duration < 20.0:
            return empty

        beats, downbeats, beat_len, beats_per_bar = _beat_and_downbeat_grid(
            y, sr, bpm_hint=bpm_hint
        )

        hop_length = 512
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20, hop_length=hop_length)
        # Coefficient 0 is overall log-energy and is ~an order of magnitude
        # larger than the rest; kept out of the timbre block so it cannot
        # dominate, and folded back in separately (below) at a controlled weight.
        mfcc_timbre = mfcc[1:, :]
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length)
        rms = librosa.feature.rms(y=y, hop_length=hop_length)

        n_frames = min(mfcc_timbre.shape[1], chroma.shape[1], rms.shape[1])
        if n_frames < 16:
            return empty
        mfcc_timbre = mfcc_timbre[:, :n_frames]
        chroma = chroma[:, :n_frames]
        rms = rms[:, :n_frames]

        # --- Beat-synchronous aggregation -------------------------------
        used_beats = False
        if beats and len(beats) >= 12:
            beat_frames = librosa.time_to_frames(
                np.asarray(beats, dtype=float), sr=sr, hop_length=hop_length
            )
            beat_frames = np.clip(beat_frames, 0, n_frames - 1)
            beat_frames = np.unique(beat_frames)
            if beat_frames.size >= 12:
                mfcc_s = librosa.util.sync(mfcc_timbre, beat_frames, aggregate=np.median)
                chroma_s = librosa.util.sync(chroma, beat_frames, aggregate=np.median)
                rms_s = librosa.util.sync(rms, beat_frames, aggregate=np.mean)
                # librosa.util.sync emits len(idx)+1 columns: a leading column
                # for [0:idx[0]], then one per [idx[k-1]:idx[k]]. Mapping column
                # k to idx[k] therefore labels every column with the time of the
                # beat AFTER the audio it summarises, putting every reported
                # boundary a full beat late. seg_times must be the START time of
                # each column instead.
                seg_times = librosa.frames_to_time(
                    np.unique(np.concatenate([[0], beat_frames])),
                    sr=sr, hop_length=hop_length,
                )
                used_beats = True

        if not used_beats:
            # Fall back to a fixed ~0.25 s grid when beat tracking failed.
            step = max(1, int(round(0.25 * sr / hop_length)))
            idx = np.arange(0, n_frames, step)
            mfcc_s = librosa.util.sync(mfcc_timbre, idx, aggregate=np.median)
            chroma_s = librosa.util.sync(chroma, idx, aggregate=np.median)
            rms_s = librosa.util.sync(rms, idx, aggregate=np.mean)
            # Same +1 column / start-time alignment as the beat-synchronous path.
            seg_times = librosa.frames_to_time(
                np.unique(np.concatenate([[0], idx])),
                sr=sr, hop_length=hop_length,
            )

        n = min(mfcc_s.shape[1], chroma_s.shape[1], rms_s.shape[1], len(seg_times))
        if n < 16:
            return empty
        mfcc_s = mfcc_s[:, :n]
        chroma_s = chroma_s[:, :n]
        rms_s = rms_s[:, :n]
        seg_times = np.asarray(seg_times[:n], dtype=float)

        def _znorm(x):
            mu = x.mean(axis=1, keepdims=True)
            sd = x.std(axis=1, keepdims=True)
            # A feature that barely varies carries no structural information,
            # so it is zeroed rather than divided through by a near-zero std.
            # Clamping the divisor instead (sd[sd < 1e-8] = 1e-8) rescales
            # floating-point dither up to full amplitude, which manufactures a
            # convincing novelty curve out of a completely static signal --
            # that is how a held tone or an ambient bed ends up carved into
            # half a dozen "sections".
            scale = np.maximum(np.abs(mu), 1.0)
            dead = sd < (1e-4 * scale)
            safe = np.where(dead, 1.0, sd)
            out = (x - mu) / safe
            out[dead[:, 0], :] = 0.0
            return out

        def _l2(x, weight):
            nrm = np.linalg.norm(x, axis=0, keepdims=True)
            nrm[nrm < 1e-8] = 1e-8
            return weight * (x / nrm)

        # Each feature family is L2-normalised on its own BEFORE being stacked,
        # so the relative weights below are the actual weights. Previously the
        # families were merely z-scored and stacked, which meant a 13-row MFCC
        # block silently outweighed a 12-row chroma block by whatever their
        # variances happened to be on that track.
        loud = np.log1p(np.maximum(rms_s, 0.0) * 1000.0)
        feats = np.vstack([
            _l2(_znorm(mfcc_s), 1.00),                            # timbre
            _l2(_znorm(chroma_s), 0.90),                          # harmony
            _l2(_znorm(loud), 0.45),                              # energy
        ])

        # Stacked memory: each column carries a short window of recent history,
        # so the SSM compares musical *context*, sharpening section corners.
        delay = 4 if used_beats else 8
        memory_steps = 3
        memory_lag = 0
        if n > delay * 3:
            feats = librosa.feature.stack_memory(
                feats, n_steps=memory_steps, delay=delay, mode="edge"
            )
            # stack_memory stacks PAST frames, so the centre of mass of each
            # column's context sits delay*(n_steps-1)/2 frames BEHIND its index.
            # Left uncompensated this pushes every reported change point late by
            # that many frames -- a constant, systematic timestamp error of
            # roughly two bars at 120 BPM, which is precisely the kind of
            # "boundary is in the right place but the time is wrong" failure
            # this rewrite exists to remove. Compensated below when the peak
            # indices are converted back to seconds.
            memory_lag = int(round(delay * (memory_steps - 1) / 2.0))

        nrm = np.linalg.norm(feats, axis=0, keepdims=True)
        nrm[nrm < 1e-8] = 1e-8
        feats = feats / nrm
        sim = feats.T @ feats

        # --- Checkerboard novelty ---------------------------------------
        # Kernel spans ~4 bars of context on each side when beat-synchronous.
        if used_beats:
            bar = int(beats_per_bar or 4)
            kernel_size = max(9, min(int(bar * 8) | 1, n - 1))
        else:
            kernel_size = min(33, n - 1)
        if kernel_size % 2 == 0:
            kernel_size -= 1
        if kernel_size < 5:
            return empty
        half = kernel_size // 2

        gauss = np.exp(-0.5 * (np.linspace(-2, 2, kernel_size) ** 2))
        checker = np.outer(gauss, gauss)
        ar = np.arange(kernel_size)
        sign = np.where((ar[:, None] < half) == (ar[None, :] < half), 1.0, -1.0)
        kernel = checker * sign

        # Vectorised diagonal correlation: extract every kernel-sized window
        # along the main diagonal at once instead of looping in Python.
        centres = np.arange(half, n - half)
        if centres.size < 3:
            return empty
        windows = np.lib.stride_tricks.sliding_window_view(
            sim, (kernel_size, kernel_size)
        )
        diag_windows = windows[centres - half, centres - half]
        novelty_core = np.tensordot(diag_windows, kernel, axes=([1, 2], [0, 1]))

        novelty = np.zeros(n, dtype=float)
        novelty[centres] = novelty_core
        if memory_lag:
            # Shift the curve earlier by the stacked-memory lag so peaks line up
            # with the frame where the music actually changed.
            novelty = np.roll(novelty, -memory_lag)
            novelty[-memory_lag:] = 0.0

        span = float(novelty.max() - novelty.min())
        if span <= 1e-9:
            return empty

        # Guard against normalising nothing into structure. On material that
        # genuinely does not change -- a held tone, a static drone, an ambient
        # bed -- the raw novelty values are all tiny and nearly equal, but
        # rescaling them to 0..1 turns that flat nothing into a curve with
        # convincing-looking peaks, and the track gets carved into sections no
        # listener would hear.
        #
        # The test is MEAN self-similarity between distant frames, not its
        # spread. Spread does not separate the two cases at all: a constant
        # 440 Hz tone and a real verse/chorus song both measure a similarity
        # standard deviation of about 0.57, because frame-level CQT/MFCC jitter
        # on a static signal is as varied as real musical contrast. What
        # actually distinguishes them is the mean -- static material is
        # uniformly self-similar everywhere (mean ~0.42), whereas real music
        # contains genuinely dissimilar regions and averages near zero
        # (mean ~-0.02). A track whose every distant pair still looks alike has
        # no sections to find.
        iu = np.triu_indices(n, k=max(1, kernel_size // 2))
        if iu[0].size >= 16:
            distant = sim[iu]
            sim_mean = float(np.mean(distant))
            # Fraction of distant frame pairs that are genuinely unalike. Real
            # arrangements always have some; static material has essentially none.
            contrast = float(np.mean(distant < 0.0))
            # Measured separation is wide and unambiguous, so the thresholds sit
            # in the middle of the gap rather than near either side:
            #   real arrangements  mean ~-0.04, contrast ~0.59
            #   held tone          mean ~ 0.57, contrast ~0.07
            #   static drone       mean ~ 0.58, contrast ~0.06
            if sim_mean > 0.30 and contrast < 0.20:
                return empty

        novelty = (novelty - novelty.min()) / span

        # Light smoothing kills single-frame spikes without moving true peaks.
        if novelty.size >= 5:
            kern = np.array([0.15, 0.7, 0.15])
            novelty = np.convolve(novelty, kern / kern.sum(), mode="same")

        # --- Prominence-based peak picking ------------------------------
        # Height alone (the old approach) rewards broad humps and misses real
        # but locally-modest transitions. Prominence measures how far a peak
        # stands above the higher of its two flanking valleys, which is what
        # "this is a distinct change point" actually means.
        interior = novelty[half:n - half]
        if interior.size < 3:
            return empty

        peaks = []
        for i in range(half + 1, n - half - 1):
            if novelty[i] >= novelty[i - 1] and novelty[i] > novelty[i + 1]:
                lo = max(half, i - kernel_size)
                hi = min(n - half, i + kernel_size + 1)
                left_min = float(np.min(novelty[lo:i + 1]))
                right_min = float(np.min(novelty[i:hi]))
                prominence = float(novelty[i]) - max(left_min, right_min)
                peaks.append((prominence, float(novelty[i]), i))

        if not peaks:
            return empty

        prom_values = np.array([p[0] for p in peaks], dtype=float)
        med = float(np.median(prom_values))
        mad = float(np.median(np.abs(prom_values - med)))
        prom_floor = max(0.02, med + 0.5 * (mad if mad > 1e-6 else 0.02))

        # How many sections can a track of this length plausibly have? Roughly
        # one change every 15-25 s; this keeps a 2-minute track from being cut
        # into 16 pieces while still allowing a long track enough detail.
        cap = int(max(3, min(max_boundaries, round(duration / 18.0))))

        if used_beats and beat_len:
            # At least two bars between section boundaries.
            bar_len = beat_len * float(beats_per_bar or 4)
            effective_min_spacing = max(float(min_spacing_s), bar_len * 2.0, 8.0)
        else:
            effective_min_spacing = max(float(min_spacing_s), 8.0)

        peaks.sort(key=lambda p: (-p[0], -p[1]))
        chosen_idx = []
        for prominence, _height, i in peaks:
            if prominence < prom_floor:
                continue
            t = float(seg_times[i])
            if t < 4.0 or t > duration - 4.0:
                continue
            if all(abs(t - float(seg_times[j])) >= effective_min_spacing
                   for j in chosen_idx):
                chosen_idx.append(i)
            if len(chosen_idx) >= cap:
                break

        chosen_idx.sort()
        times = [float(seg_times[i]) for i in chosen_idx]

        # --- Bar alignment ----------------------------------------------
        times = _snap_boundaries_to_metre(times, downbeats, beats, beat_len)
        times = [t for t in times if 3.0 <= t <= duration - 3.0]

        # Re-enforce spacing after snapping (two peaks can share a bar line).
        final = []
        for t in times:
            if not final or t - final[-1] >= effective_min_spacing * 0.75:
                final.append(round(t, 2))

        if return_details:
            return final, {
                "beats": beats,
                "downbeats": downbeats,
                "beat_len": beat_len,
                "beats_per_bar": beats_per_bar,
                "beat_synchronous": used_beats,
                "duration": duration,
            }
        return final
    except Exception:
        return empty


# ==============================================================================
# SONG STRUCTURE v2 -- bar-level, multi-cue segmentation and naming
# ==============================================================================
# Everything in this block is signal-only: it measures THIS recording (mix
# features, separated-stem note events, lyric timestamps) and never consults
# genre assumptions, file tags or general knowledge about the song.
#
# Pipeline (see build_structure_map):
#   1. Beat grid with a sub-frame tempo estimate (tracked on a kick-anchored
#      onset envelope), then bar lines from a small Viterbi over bar
#      positions, so a beat the tracker inserts or drops mid-song does not
#      shift every later bar line.
#   2. Beat-synchronous mix features (timbre / harmony / loudness), pooled to
#      bars, plus per-bar stem descriptors when note events are available
#      (vocal coverage and register, kick/snare/hi-hat counts, bass and
#      harmonic-instrument activity).
#   3. A per-bar boundary-evidence curve combining checkerboard novelty on
#      the mix, repetition onsets (where repeated chord sequences start or
#      stop), and discrete arrangement events (drum fills, drum entries and
#      exits, vocal entries after a real gap, a change of vocal register,
#      lyric-block starts). A boundary found at the start of a short break
#      is moved to where the band comes back in.
#   4. A dynamic programme that picks the boundary set with the most evidence
#      under a section-length prior (no sub-4-bar fragments, mild preference
#      for 4-bar multiples, long sections discouraged but allowed).
#   5. Segments are clustered by aligned chord-sequence similarity (plus
#      timbre and repeated lyrics), so repeated sections are recognised even
#      when their arrangement differs a little.
#   6. Names come from what was measured: unsung bookends are Intro/Outro, the
#      repeated sung material with the strongest hook evidence (repeated words,
#      higher vocal register, more energy) is the Chorus -- whether it comes
#      first or second -- other repeated sung material is the Verse, one-off
#      sung material between choruses is the Bridge, unsung mid-song material
#      is Instrumental, digital silence is Silence.
# ==============================================================================

def _sx_refine_tempo(onset_env, frame_rate, bpm0, span=0.06):
    """Sub-frame tempo estimate around a coarse beat-tracker tempo.

    librosa's beat tracker works in whole onset frames, so its tempo snaps to
    60 * frame_rate / k. At the default 512-sample hop and 22.05 kHz that is
    a coarse ladder: a 120 BPM song reads as 117.45 BPM (22 frames) and a
    158 BPM one as 161.5 BPM (16 frames). A 2 % tempo error makes the beat
    grid drift a whole beat every ~50 beats, and every section boundary that
    is snapped to that grid inherits the drift.

    The fix measures the periodicity of the onset envelope directly: the
    magnitude of its Fourier component at each candidate tempo, evaluated on
    a fine grid around the coarse value. Only octave-preserving refinement is
    done (+/- span), so the metrical level the tracker chose is kept.
    """
    try:
        env = np.asarray(onset_env, dtype=float)
        bpm0 = float(bpm0)
        if env.size < 64 or not (30.0 <= bpm0 <= 300.0):
            return bpm0
        t = np.arange(env.size, dtype=float) / float(frame_rate)
        e = env - float(np.mean(env))

        # Candidates are evaluated in blocks small enough that the
        # (candidates x frames) complex matrix stays around 30 MB even on a
        # long file.
        per_block = max(1, int(2_000_000 // max(1, env.size)))

        def _best(cands):
            best_b, best_r = None, -1.0
            for chunk in np.array_split(cands, max(1, int(np.ceil(len(cands) / float(per_block))))):
                if chunk.size == 0:
                    continue
                ph = np.exp(-2j * np.pi * np.outer(chunk / 60.0, t))
                r = np.abs(ph @ e)
                k = int(np.argmax(r))
                if r[k] > best_r:
                    best_b, best_r = float(chunk[k]), float(r[k])
            return best_b

        coarse = _best(np.arange(bpm0 * (1.0 - span), bpm0 * (1.0 + span), 0.05))
        if coarse is None:
            return bpm0
        fine = _best(np.arange(coarse - 0.1, coarse + 0.1, 0.005))
        return float(fine if fine is not None else coarse)
    except Exception:
        return float(bpm0)


def _sx_onset_strength_chunked(y, sr, hop_length, chunk_s=45.0, **kwargs):
    """librosa.onset.onset_strength computed in overlapping blocks.

    A fine-hop STFT of a whole track is large (a 256-sample hop over 16
    minutes is ~1 GB transient), and this runs while the listening model may
    still be resident. Spectral flux is local, so blocks with a few frames of
    overlap stitch back together exactly apart from per-block dB clipping in
    near-silent passages."""
    n = len(y)
    block = int(chunk_s * sr) // hop_length * hop_length
    if n <= block * 2:
        return librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length, **kwargs)
    pad = 16 * hop_length
    total_frames = 1 + n // hop_length
    out = np.zeros(total_frames, dtype=float)
    start = 0
    while start < n:
        a = max(0, start - pad)
        b = min(n, start + block + pad)
        env = librosa.onset.onset_strength(y=y[a:b], sr=sr, hop_length=hop_length, **kwargs)
        f0 = (start - a) // hop_length
        f_first = start // hop_length
        count = min(block // hop_length, total_frames - f_first, len(env) - f0)
        if count > 0:
            out[f_first:f_first + count] = env[f0:f0 + count]
        start += block
    return out


def _sx_onset_envelope(y, sr, hop_length=256):
    """Broadband onsets plus a low-frequency (kick/bass) onset envelope, each
    scaled to unit spread. Broadband flux alone lets off-beat hats, claps and
    synth stabs pull a beat tracker half a beat out for whole passages of
    dance and pop material; the low band anchors it to the kick."""
    env_full = _sx_onset_strength_chunked(y, sr, hop_length)
    env = env_full / (float(np.std(env_full)) + 1e-9)
    try:
        env_low = _sx_onset_strength_chunked(
            y, sr, hop_length, fmax=250.0, n_mels=32
        )
        if float(np.std(env_low)) > 1e-9:
            env = env + env_low / float(np.std(env_low))
    except Exception:
        pass
    return env


def _sx_track_beats_at(y, sr, bpm, env_cache=None):
    """Run librosa's tracker at a known tempo on a hop fine enough that its
    whole-frame beat period is within ~1 % of that tempo (the tracker rounds
    the period to whole frames: at a 256-sample hop a 240 BPM pulse is 21.5
    frames and is tracked as 234.9 BPM). Returns beat times or None."""
    try:
        bpm = float(bpm)
        if not (30.0 <= bpm <= 320.0):
            return None
        hop = 512
        for cand in (512, 256, 128, 64):
            hop = cand
            if 60.0 * sr / cand / bpm >= 40.0:
                break
        env = None
        if env_cache is not None:
            env = env_cache.get(hop)
        if env is None:
            if hop >= 256:
                env = _sx_onset_envelope(y, sr, hop)
            else:
                # Finer hops are made by interpolating the 256-sample
                # envelope rather than computing a new STFT: the tracker only
                # needs a finer period grid, and a 64/128-sample STFT of a
                # long track costs several hundred MB.
                base = env_cache.get(256) if env_cache is not None else None
                if base is None:
                    base = _sx_onset_envelope(y, sr, 256)
                    if env_cache is not None:
                        env_cache[256] = base
                factor = int(256 // hop)
                xs = np.arange(len(base) * factor, dtype=float) / factor
                env = np.interp(xs, np.arange(len(base), dtype=float), base)
            if env_cache is not None:
                env_cache[hop] = env
        _t, b = librosa.beat.beat_track(
            onset_envelope=env, sr=sr, hop_length=hop,
            trim=False, units="time", bpm=bpm,
        )
        b = np.asarray(b, dtype=float)
        return b if b.size >= 8 else None
    except Exception:
        return None


def _sx_beat_grid(y, sr, bpm_hint=None, hop_length=256):
    """Beat times with a refined tempo. Returns (beats, bpm) or ([], None).

    The metrical level comes from librosa's tempo estimator on a 512-sample
    envelope (cheap; estimating on the finer envelope costs ~600 MB for a
    four-minute track). The tempo is then refined to sub-frame precision on
    the finer envelope (_sx_refine_tempo) and the beats are tracked at that
    tempo on a hop fine enough for the tracker's whole-frame period to match
    it (_sx_track_beats_at), so the grid does not drift against the music.
    """
    try:
        if y is None or len(y) < sr * 4:
            return [], None
        envs = {}
        env512 = _sx_onset_envelope(y, sr, 512)
        envs[512] = env512
        start_bpm = 120.0
        if bpm_hint:
            try:
                bh = float(bpm_hint)
                if 40.0 <= bh <= 220.0:
                    start_bpm = bh
            except Exception:
                pass
        bpm0 = 0.0
        try:
            _tempo_fn = getattr(librosa.feature, "tempo", None) or librosa.beat.tempo
            t0 = _tempo_fn(onset_envelope=env512, sr=sr, hop_length=512, start_bpm=start_bpm)
            t0 = np.atleast_1d(t0)
            bpm0 = float(t0[0]) if t0.size else 0.0
        except Exception:
            bpm0 = 0.0
        if not (bpm0 > 0):
            _t, b0 = librosa.beat.beat_track(
                onset_envelope=env512, sr=sr, hop_length=512, trim=False,
                units="time", start_bpm=start_bpm,
            )
            b0 = np.asarray(b0, dtype=float)
            if b0.size > 4:
                bpm0 = 60.0 / float(np.median(np.diff(b0)))
        if not (bpm0 > 0):
            return [], None
        env = _sx_onset_envelope(y, sr, hop_length)
        envs[hop_length] = env
        bpm = _sx_refine_tempo(env, float(sr) / hop_length, bpm0)
        beats = _sx_track_beats_at(y, sr, bpm, envs)
        if beats is None or len(beats) < 8:
            return [], None
        return np.asarray(beats, dtype=float).tolist(), float(bpm)
    except Exception:
        return [], None


def _sx_drum_hits(stem_notes, kind):
    out = []
    for n in ((stem_notes or {}).get("drums") or []):
        try:
            k = n.get("drum_type") if "drum_type" in n else n.get("k")
            if k == kind:
                out.append(float(n.get("onset", n.get("t", 0.0))))
        except Exception:
            continue
    return out


def _sx_downbeat_flags(beats, bpb, accents, stem_notes=None, event_times=None):
    """Which beats start a bar -- decided along the whole song, not by one
    global phase.

    A beat tracker following a drifting live (or AI-generated) performance
    occasionally inserts or drops a beat. A single global "every 4th beat
    from beat k" grid is then right before the slip and a beat out after it,
    for the rest of the track -- observed: section starts landing one beat
    late after the first minute of a rock take. This runs a small Viterbi
    over bar positions instead: advancing one position per beat is free,
    stalling or skipping a position costs a lot, so the bar phase only moves
    where the evidence keeps saying it has moved. Evidence for "this beat is
    a downbeat": the accent curve (low-frequency onsets + harmonic change),
    section events (a fill resolving, a drum or vocal re-entry), a kick on
    the beat, and -- against it -- a snare/clap on the beat (backbeats sit on
    2 and 4). Returns a boolean array parallel to `beats`.
    """
    beats = np.asarray(beats, dtype=float)
    n = beats.size
    bpb = int(bpb or 4)
    flags = np.zeros(n, dtype=bool)
    if n == 0:
        return flags
    if n < bpb * 3 or bpb < 2:
        flags[0::max(bpb, 1)] = True
        return flags
    med_beat = float(np.median(np.diff(beats))) if n > 1 else 0.5
    tol = 0.25 * med_beat

    def _nearest(t):
        i = int(np.searchsorted(beats, t))
        if i <= 0:
            return 0
        if i >= n:
            return n - 1
        return i if abs(beats[i] - t) < abs(beats[i - 1] - t) else i - 1

    down = np.zeros(n)
    if accents is not None and len(accents) == n:
        a = np.asarray(accents, dtype=float)
        sd = float(np.std(a))
        if sd > 1e-9:
            down += 0.8 * (a - float(np.mean(a))) / sd
    for item in event_times or []:
        if isinstance(item, (tuple, list)):
            t, wv = float(item[0]), float(item[1])
        else:
            t, wv = float(item), 1.0
        k = _nearest(t)
        if abs(beats[k] - t) <= tol:
            down[k] += 2.5 * wv
    for kind, wv in (("snare", -0.6), ("kick", 0.3)):
        hits = _sx_drum_hits(stem_notes, kind)
        if len(hits) >= 16:
            on = np.zeros(n, dtype=bool)
            for t in hits:
                k = _nearest(t)
                if abs(beats[k] - t) <= 0.07:
                    on[k] = True
            down += wv * on

    # Viterbi over positions 0..bpb-1 (0 = downbeat).
    stall = skip = -7.0
    score = np.where(np.arange(bpb) == 0, down[0], 0.0)
    back = np.zeros((n, bpb), dtype=int)
    for k in range(1, n):
        new = np.full(bpb, -1e18)
        for st in range(bpb):
            emit = down[k] if st == 0 else 0.0
            for prev, cost in (((st - 1) % bpb, 0.0), (st, stall), ((st - 2) % bpb, skip)):
                v = score[prev] + cost + emit
                if v > new[st]:
                    new[st] = v
                    back[k, st] = prev
        score = new
    st = int(np.argmax(score))
    states = np.zeros(n, dtype=int)
    for k in range(n - 1, -1, -1):
        states[k] = st
        st = back[k, st]
    flags = states == 0
    return flags


def _sx_beat_features(y, sr, beats, hop_length=512):
    """Mix features pooled per beat interval (cached and re-pooled to bars).

    Returns dict(edges, mfcc, chroma, energy) where edges has n+1 entries and
    the first interval starts at 0.0. MFCC[0] (overall level) is excluded
    from timbre; level is carried separately as mean energy so it can be
    weighted on its own.
    """
    duration = len(y) / float(sr)
    mf = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20, hop_length=hop_length)[1:]
    ch = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length)
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    nf = int(min(mf.shape[1], ch.shape[1], rms.shape[0]))
    bt = np.asarray([b for b in beats if 0.0 < b < duration], dtype=float)
    edges = np.concatenate([[0.0], bt, [duration]])
    fe = np.clip(librosa.time_to_frames(edges, sr=sr, hop_length=hop_length), 0, nf)
    fe[-1] = nf
    cols_m, cols_c, cols_e = [], [], []
    for a, b in zip(fe[:-1], fe[1:]):
        a = int(a)
        b = int(max(b, a + 1))
        a = min(a, nf - 1)
        b = min(b, nf)
        cols_m.append(np.mean(mf[:, a:b], axis=1))
        cols_c.append(np.mean(ch[:, a:b], axis=1))
        cols_e.append(float(np.mean(np.square(rms[a:b]))))
    return {
        "edges": edges,
        "mfcc": np.stack(cols_m, axis=1),
        "chroma": np.stack(cols_c, axis=1),
        "energy": np.asarray(cols_e, dtype=float),
        "duration": duration,
    }


def _sx_note_span(n):
    try:
        if "onset" in n:
            s = float(n.get("onset", 0.0))
            e = float(n.get("offset", s))
        else:
            s = float(n.get("t", 0.0))
            e = s + float(n.get("d", 0.0))
        return s, max(e, s)
    except Exception:
        return None


def _sx_note_midi(n):
    try:
        if n.get("m") is not None:
            return float(n["m"])
        f = float(n.get("frequency") or n.get("hz") or 0.0)
        if f > 0:
            return 69.0 + 12.0 * np.log2(f / 440.0)
    except Exception:
        pass
    return None


def _sx_silent_drum_hit(x):
    """0.7.1-pre13: a drum hit whose measured level is below 0.001 of that drum's loudest hits
    (the transcriber's borderline detections, which come and go between scans of one file)."""
    if not isinstance(x, dict) or not (x.get("drum_type") or x.get("k")):
        return False
    lv = x.get("level", x.get("amplitude"))
    try:
        return lv is not None and float(lv) < 0.001
    except Exception:
        return False


def _sx_bar_features(beatfeat, bars, end_time=None, stem_notes=None, instrumental=False):
    """Pool cached beat features into bars and add per-bar stem descriptors.
    Beats after `end_time` (a silent tail) are ignored."""
    edges_b = np.asarray(beatfeat["edges"], dtype=float)
    duration = float(end_time if end_time is not None else beatfeat["duration"])
    bar_edges = np.concatenate([np.asarray(bars, dtype=float), [duration]])
    nb = len(bars)
    centers = 0.5 * (edges_b[:-1] + edges_b[1:])
    idx = np.searchsorted(bar_edges, centers, side="right") - 1
    idx = np.where((centers >= bar_edges[0]) & (centers < duration), np.clip(idx, 0, nb - 1), -1)

    def _pool(X):
        out = np.zeros((X.shape[0], nb))
        for k in range(nb):
            m = idx == k
            if np.any(m):
                out[:, k] = X[:, m].mean(axis=1)
            elif k > 0:
                out[:, k] = out[:, k - 1]
        return out

    mf = _pool(beatfeat["mfcc"])
    ch = _pool(beatfeat["chroma"])
    en = _pool(beatfeat["energy"][None, :])[0]
    F = {
        "mfcc": mf,
        "chroma": ch,
        "loud": 10.0 * np.log10(np.maximum(en, 1e-12)),
        "has_stems": False,
        "has_vocal": False,
    }
    if not stem_notes:
        return F
    widths = np.maximum(1e-6, np.diff(bar_edges))

    def _coverage(notes):
        cov = np.zeros(nb)
        spans = sorted(sp for sp in (_sx_note_span(x) for x in (notes or []) if isinstance(x, dict)) if sp)
        merged = []
        for s, e in spans:
            if merged and s <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], e)
            else:
                merged.append([s, e])
        for s, e in merged:
            k0 = max(0, int(np.searchsorted(bar_edges, s, side="right") - 1))
            k1 = min(nb, int(np.searchsorted(bar_edges, e, side="left")))
            for k in range(k0, k1):
                cov[k] += max(0.0, min(e, bar_edges[k + 1]) - max(s, bar_edges[k]))
        return np.minimum(1.0, cov / widths)

    def _counts(notes, kind=None):
        c = np.zeros(nb)
        for x in notes or []:
            if not isinstance(x, dict):
                continue
            if kind is not None:
                k = x.get("drum_type") if "drum_type" in x else x.get("k")
                if k != kind:
                    continue
            sp = _sx_note_span(x)
            if not sp:
                continue
            # A hit a few ms ahead of the bar line belongs to the new bar
            # (transcription and beat-tracking jitter are both ~20-40 ms).
            k = int(np.searchsorted(bar_edges, sp[0] + 0.06, side="right") - 1)
            if 0 <= k < nb:
                if STRUCTURE_IGNORE_SILENT_DRUM_HITS and _sx_silent_drum_hit(x):
                    continue                    # 0.7.1-pre13: a "hit" with no measurable sound
                c[k] += 1
        return c

    def _pitch(notes):
        num = np.zeros(nb)
        den = np.zeros(nb)
        for x in notes or []:
            if not isinstance(x, dict):
                continue
            mid = _sx_note_midi(x)
            sp = _sx_note_span(x)
            if mid is None or not sp:
                continue
            s, e = sp
            k0 = max(0, int(np.searchsorted(bar_edges, s, side="right") - 1))
            k1 = min(nb, int(np.searchsorted(bar_edges, e, side="left")))
            for k in range(k0, max(k1, k0 + 1)):
                if k >= nb:
                    break
                w = max(0.0, min(e, bar_edges[k + 1]) - max(s, bar_edges[k])) or 1e-3
                num[k] += mid * w
                den[k] += w
        p = np.full(nb, np.nan)
        ok = den > 0
        p[ok] = num[ok] / den[ok]
        return p

    vocals = None
    for key in ("vocals", "vocal", "voice"):
        if stem_notes.get(key):
            vocals = stem_notes[key]
            break
    drums = stem_notes.get("drums") or []
    harm = []
    for key in ("other", "guitar", "piano"):
        harm.extend(stem_notes.get(key) or [])
    F.update({
        "has_stems": bool(drums or harm or stem_notes.get("bass")),
        "kick": _counts(drums, "kick"),
        "snare": _counts(drums, "snare"),
        "hat": _counts(drums, "hihat"),
        "drums": _counts(drums),
        "bass": _counts(stem_notes.get("bass")),
        "harm": _counts(harm),
    })
    if vocals and not instrumental:
        F["vocal"] = _coverage(vocals)
        F["vpitch"] = _pitch(vocals)
        F["has_vocal"] = bool(np.max(F["vocal"]) >= 0.3) if nb else False
        F["vocal_notes"] = vocals
    else:
        F["vocal"] = np.zeros(nb)
        F["vpitch"] = np.full(nb, np.nan)
    return F


def _sx_unit_cols(X):
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[None, :]
    mu = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, keepdims=True)
    scale = np.maximum(np.abs(mu), 1.0)
    dead = sd < (1e-4 * scale)
    Z = (X - mu) / np.where(dead, 1.0, sd)
    Z[dead[:, 0], :] = 0.0
    nrm = np.linalg.norm(Z, axis=0, keepdims=True)
    nrm[nrm < 1e-9] = 1.0
    return Z / nrm


def _sx_foote(S, half):
    """Checkerboard novelty on a bar-level SSM with a kernel `half` bars wide
    on each side. Truncated at the track edges (no padding), so the first and
    last bars are not flagged as changes just because the matrix ends."""
    n = S.shape[0]
    out = np.zeros(n)
    half = int(max(1, half))
    for i in range(1, n):
        a0 = max(0, i - half)
        b1 = min(n, i + half)
        if i - a0 < 1 or b1 - i < 1:
            continue
        A = S[a0:i, a0:i]
        B = S[i:b1, i:b1]
        X = S[a0:i, i:b1]
        val = 0.5 * (float(A.mean()) + float(B.mean())) - float(X.mean())
        # With fewer than two bars on one side the kernel compares a sliver
        # (a fade-in, a pickup) against a section and over-reports change.
        side = min(i - a0, b1 - i)
        if side < 2:
            val *= 0.5
        out[i] = val
    return np.maximum(out, 0.0)


def _sx_rnorm(x, q=90):
    """Scale a non-negative curve so its typical high values sit near 1.
    The divisor is never below a quarter of the curve's own maximum, so on
    material with one real change and otherwise tiny ripples the ripples
    stay tiny instead of being blown up to the size of a section change."""
    x = np.maximum(np.asarray(x, dtype=float), 0.0)
    pos = x[x > 0]
    if pos.size == 0:
        return x
    s = max(float(np.percentile(pos, q)), 0.25 * float(np.max(pos)))
    return np.minimum(x / (s + 1e-9), 1.5)


def _sx_chroma_unit(F):
    C = np.asarray(F["chroma"], dtype=float)
    Cz = C - C.mean(axis=1, keepdims=True)
    nrm = np.linalg.norm(Cz, axis=0, keepdims=True)
    nrm[nrm < 1e-9] = 1.0
    return Cz / nrm


def _sx_repetition_curve(F, w=4, thr=0.5, min_lag=4, bar_len=None):
    """Where repeated chord sequences begin or end, per bar.

    For every pair of bars far apart, compare the next `w` bars along the
    diagonal (does a repeat start here?) with the previous `w` bars (was it
    already repeating?). A section that comes back produces a diagonal stripe
    in the chord self-similarity matrix; its first and last bars are where
    this curve peaks -- even when the arrangement is unchanged, which is the
    case timbre novelty cannot see.
    """
    if bar_len:
        # ~10 s of context either side, whatever the tracked bar length.
        w = int(min(8, max(4, round(10.0 / float(bar_len)))))
        min_lag = w
    Cz = _sx_chroma_unit(F)
    n = Cz.shape[1]
    if n < 2 * w + min_lag or n > 1200:
        # Too short to hold a repeat, or so long (a DJ mix, a live set) that
        # the bar-by-bar matrices below would cost hundreds of MB.
        return np.zeros(n)
    S = Cz.T @ Cz
    Sp = np.full((n + 2 * w, n + 2 * w), np.nan)
    Sp[w:w + n, w:w + n] = S
    Rf = np.zeros((n, n))
    Rb = np.zeros((n, n))
    cf = np.zeros((n, n))
    cb = np.zeros((n, n))
    for k in range(w):
        f = Sp[w + k:w + k + n, w + k:w + k + n]
        b = Sp[w - 1 - k:w - 1 - k + n, w - 1 - k:w - 1 - k + n]
        Rf += np.nan_to_num(f)
        cf += ~np.isnan(f)
        Rb += np.nan_to_num(b)
        cb += ~np.isnan(b)
    Rf = np.where(cf > 0, Rf / np.maximum(cf, 1), 0.0)
    Rb = np.where(cb > 0, Rb / np.maximum(cb, 1), 0.0)
    I, J = np.indices((n, n))
    far = np.abs(I - J) >= min_lag
    start = np.where(far, np.maximum(0, Rf - thr) * np.maximum(0, thr - Rb), 0.0).sum(axis=1)
    end = np.where(far, np.maximum(0, Rb - thr) * np.maximum(0, thr - Rf), 0.0).sum(axis=1)
    return start + end


def _sx_vocal_phrases(vocal_notes, join_gap):
    spans = sorted(sp for sp in (_sx_note_span(x) for x in (vocal_notes or []) if isinstance(x, dict)) if sp)
    phrases = []
    for s, e in spans:
        if phrases and s - phrases[-1][1] <= join_gap:
            phrases[-1][1] = max(phrases[-1][1], e)
        else:
            phrases.append([s, e])
    return [p for p in phrases if p[1] - p[0] >= 0.4]


def _sx_event_evidence(F, bars, duration, stem_notes=None, lyric_blocks=None):
    """Discrete arrangement events, each scored 0..1 at the bar line where
    the new section would start. Returns (dict of arrays, list of event
    times usable as downbeat votes).

    Windows are measured in SECONDS, not tracked bars: at fast tempos the
    beat tracker often counts eighth notes, so a "bar" can be half a real bar
    and a bar-count window would halve with it.
    """
    nb = len(bars)
    bars = np.asarray(bars, dtype=float)
    bar_edges = np.concatenate([bars, [duration]])
    barlen = float(np.median(np.diff(bar_edges))) if nb > 1 else 2.0
    ev = {k: np.zeros(nb) for k in ("fill", "drum_in", "drum_out", "kick_in", "vocal_in",
                                    "vocal_out", "register", "lyric")}
    votes = []

    def _bar_at_or_before(t, pickup=0.3):
        k = int(np.searchsorted(bar_edges, t + pickup * barlen, side="right") - 1)
        return min(max(k, 0), nb - 1)

    if F.get("has_stems"):
        snares = np.asarray(sorted(_sx_drum_hits(stem_notes, "snare")), dtype=float)
        all_hits = []
        for n in ((stem_notes or {}).get("drums") or []):
            sp = _sx_note_span(n) if isinstance(n, dict) else None
            if sp:
                all_hits.append(sp[0])
        all_hits = np.asarray(sorted(all_hits), dtype=float)
        # Fill: the snare (toms and claps transcribe as snare) is far busier
        # in the stretch just before a bar line than anywhere typical in the
        # song. Judged against the song's own distribution of the same
        # window, so a busy groove does not read as a constant fill.
        win = max(barlen, 1.8)
        if snares.size >= 8:
            counts = np.array([
                np.count_nonzero((snares >= t - win) & (snares < t - 0.03)) for t in bars
            ], dtype=float)
            active = counts[counts > 0]
            if active.size >= 6:
                med = float(np.median(active))
                thr = max(4.0, 2.2 * med, float(np.percentile(active, 92)))
                for i in range(1, nb):
                    if counts[i] >= thr:
                        ev["fill"][i] = min(1.0, 0.6 + 0.4 * (counts[i] - thr) / max(thr, 1.0))
                    elif counts[i] >= 0.8 * thr:
                        # Ramp up to the threshold instead of a hard cut-off:
                        # Omnizart's snare count varies by a hit or two between
                        # runs, and with an all-or-nothing score one hit fewer
                        # erased an AI-generated test track's Chorus 2 boundary (6 snares scored
                        # 0.65, 5 scored 0). Starts at 80% of the threshold: a
                        # wider ramp (70%) gave ordinary busy bars inside a
                        # bridge enough "fill" to pull its boundary off.
                        ev["fill"][i] = 0.6 * (counts[i] - 0.8 * thr) / (0.2 * thr)
                # One fill is one event. When the window is longer than a bar
                # (fast tempos), the same fill also lands in the NEXT bar
                # line's window, and the two scores came out nearly equal
                # (rock test track: 2.12 vs 2.14), so one extra transcribed
                # snare moved Chorus 2 a bar late, from 2:04 to 2:06. Keep the
                # bar line whose window is busiest just before it -- a fill
                # peaks right before the new section's downbeat.
                late_w = min(win, 0.6 * win + 0.2)
                late = np.array([
                    np.count_nonzero((snares >= t - late_w) & (snares < t - 0.03)) for t in bars
                ], dtype=float)
                i = 1
                while i < nb:
                    if ev["fill"][i] <= 0:
                        i += 1
                        continue
                    j = i
                    while j + 1 < nb and ev["fill"][j + 1] > 0:
                        j += 1
                    if j > i:
                        run = list(range(i, j + 1))
                        top = max(float(ev["fill"][k]) for k in run)
                        # Busiest just before the bar line wins; on a tie the
                        # later line, where a long fill actually ends.
                        keep = max(run, key=lambda k: (late[k], k))
                        for k in run:
                            ev["fill"][k] = top if k == keep else 0.0
                    i = j + 1

        # Whole-kit entry / exit: at least ~2 s of (near) silence on the kit.
        if all_hits.size >= 16:
            gap_min = max(2.0, barlen)
            for i in range(1, nb):
                t = bars[i]
                before = np.count_nonzero((all_hits >= t - gap_min) & (all_hits < t - 0.15))
                after = np.count_nonzero((all_hits >= t - 0.15) & (all_hits < t + gap_min))
                if before == 0 and after >= 4:
                    ev["drum_in"][i] = 1.0
                    kicks_ = np.asarray(sorted(_sx_drum_hits(stem_notes, "kick")), dtype=float)
                    first = kicks_[(kicks_ >= t - 0.15) & (kicks_ <= t + max(1.0, 0.5 * barlen))]
                    if first.size:
                        votes.append(float(first[0]))
                elif after == 0 and before >= 4:
                    ev["drum_out"][i] = 1.0

        # Kick re-entry: the kick has been absent for at least two bars while
        # the rest of the kit (or the band) kept going, and comes back on this
        # bar line -- the classic breakdown-to-chorus return in pop and dance
        # music. The whole-kit test above misses it because hats or snare
        # carry on through the break; an AI-generated test track's last chorus (2:15) is exactly
        # this, and was found before only thanks to a borderline fill score.
        kicks_all = np.asarray(sorted(_sx_drum_hits(stem_notes, "kick")), dtype=float)
        if kicks_all.size >= 16:
            k_gap = max(2.0 * barlen, 3.0)
            # A pickup kick or two just before the downbeat belongs to the
            # return, not to the break.
            pickup = max(0.15, 0.25 * barlen)
            beat_w = max(0.3, 0.25 * barlen)
            for i in range(1, nb):
                if ev["drum_in"][i] > 0:
                    continue
                t = bars[i]
                k_before = np.count_nonzero((kicks_all >= t - k_gap) & (kicks_all < t - pickup))
                k_after = np.count_nonzero((kicks_all >= t - 0.15) & (kicks_all < t + barlen))
                on_line = np.count_nonzero((kicks_all >= t - pickup) & (kicks_all <= t + beat_w))
                if (k_before == 0 and k_after >= 2 and on_line >= 1
                        and np.any(kicks_all < t - k_gap)):
                    ev["kick_in"][i] = 1.0

    if F.get("has_vocal"):
        phrases = _sx_vocal_phrases(F.get("vocal_notes"), 0.5 * barlen)
        spans = [(s_, e_) for s_, e_ in phrases]
        notes = []
        for x in F.get("vocal_notes") or []:
            if not isinstance(x, dict):
                continue
            sp = _sx_note_span(x)
            mid = _sx_note_midi(x)
            if sp and mid is not None:
                notes.append((sp[0], sp[1], mid))
        notes.sort()
        n_on = np.asarray([a for a, _b, _m in notes], dtype=float)
        n_mid = np.asarray([mm for _a, _b, mm in notes], dtype=float)

        def _median_pitch(a, b):
            if n_on.size == 0:
                return None
            sel = (n_on >= a) & (n_on < b)
            if np.count_nonzero(sel) < 4:
                return None
            return float(np.median(n_mid[sel]))

        prev_end = None
        for pi, (s_, e_) in enumerate(spans):
            gap = (s_ - prev_end) if prev_end is not None else s_
            k = _bar_at_or_before(s_)
            need = 3.0 if prev_end is None else max(4.0, min(2.5 * barlen, 6.0))
            if gap >= need:
                # Singing resumes after a real break -- the classic section
                # start. Breaths between lines inside a verse do not qualify.
                ev["vocal_in"][k] = max(ev["vocal_in"][k], min(1.0, 0.45 + gap / 20.0))
                votes.append((s_, 0.3))
                if prev_end is not None:
                    j = int(np.searchsorted(bar_edges, prev_end - 0.1 * barlen, side="left"))
                    if 0 < j < nb:
                        ev["vocal_out"][j] = max(ev["vocal_out"][j], min(0.8, 0.3 + gap / 24.0))
            # A sustained change of vocal register at a phrase start: the
            # median pitch of the ~8 s after differs from the ~8 s before.
            if pi > 0 and gap >= 0.25 * barlen:
                before = _median_pitch(s_ - 8.0, s_ - 0.05)
                after = _median_pitch(s_ - 0.05, s_ + 8.0)
                if before is not None and after is not None:
                    d = abs(after - before)
                    if d >= 3.0:
                        ev["register"][k] = max(ev["register"][k], min(1.0, (d - 2.0) / 4.0))
            prev_end = e_ if prev_end is None else max(prev_end, e_)
        if prev_end is not None and duration - prev_end >= max(2.0 * barlen, 4.0):
            j = int(np.searchsorted(bar_edges, prev_end - 0.1 * barlen, side="left"))
            if 0 < j < nb:
                ev["vocal_out"][j] = max(ev["vocal_out"][j], 0.8)

    for blk in lyric_blocks or []:
        try:
            s_ = float(blk["start"])
        except Exception:
            continue
        # Nearest bar line: Whisper line timing is only good to ~0.5 s and
        # sung pickups start a little before the bar, so "at or before"
        # regularly put this a bar early.
        k = int(np.argmin(np.abs(bar_edges[:-1] - s_)))
        if 0 < k < nb:
            # The start of words that recur elsewhere (a chorus or refrain
            # coming in) is moderate evidence; any other block start is weak.
            # Lyrics mainly decide NAMES; the arrangement decides boundaries.
            ev["lyric"][k] = max(ev["lyric"][k], 0.5 if blk.get("signature") else 0.25)
    return ev, votes


def _sx_peaks(x, base_radius=6):
    """Novelty above its local baseline (running median over +/- base_radius
    bars). A broad hump keeps its shape but no longer lifts the whole
    neighbourhood, so the dynamic programme sees where the curve stands out
    rather than where it is merely high."""
    x = np.maximum(np.asarray(x, dtype=float), 0.0)
    n = x.size
    out = np.zeros(n)
    for i in range(n):
        a2 = max(0, i - base_radius)
        b2 = min(n, i + base_radius + 1)
        out[i] = max(0.0, x[i] - float(np.median(x[a2:b2])))
    return _sx_rnorm(out)


def _sx_evidence_curve(F, bars, duration, stem_notes=None, lyric_blocks=None):
    """Per-bar boundary evidence (higher = more likely a section starts here)."""
    nb = len(bars)
    Zt = _sx_unit_cols(F["mfcc"])
    St = Zt.T @ Zt
    L = np.asarray(F["loud"], dtype=float)
    Sl = -np.abs(L[:, None] - L[None, :]) / 3.0
    ev = 0.5 * _sx_peaks(_sx_foote(St, 4)) + 0.5 * _sx_peaks(_sx_foote(St, 8))
    ev = ev + 0.5 * _sx_peaks(_sx_foote(Sl, 4))
    bar_len = float(np.median(np.diff(np.concatenate([np.asarray(bars, dtype=float), [duration]])))) if nb > 1 else None
    ev = ev + 0.6 * _sx_peaks(_sx_repetition_curve(F, bar_len=bar_len))
    if F.get("has_stems"):
        rows = [F["kick"], F["snare"], F["hat"], F["bass"], F["harm"] / 3.0]
        if F.get("has_vocal"):
            vp = np.asarray(F["vpitch"], dtype=float)
            ok = np.where(~np.isnan(vp))[0]
            vpf = np.interp(np.arange(nb), ok, vp[ok]) if ok.size else np.zeros(nb)
            rows = [F["vocal"], (vpf / 6.0) * np.minimum(1.0, F["vocal"] * 3.0)] + rows
        Zs = _sx_unit_cols(np.vstack(rows))
        Ss = Zs.T @ Zs
        ev = ev + STRUCTURE_W_STEM_NOVELTY * (0.5 * _sx_peaks(_sx_foote(Ss, 4)) + 0.5 * _sx_peaks(_sx_foote(Ss, 8)))
    events, votes = _sx_event_evidence(F, bars, duration, stem_notes, lyric_blocks)
    ev = (ev + 1.0 * events["fill"] + 1.0 * events["vocal_in"] + 0.4 * events["vocal_out"]
          + 0.8 * events["drum_in"] + 0.5 * events["drum_out"] + 0.7 * events["kick_in"]
          + 0.6 * events["register"]
          + 0.6 * events["lyric"])
    if nb:
        ev[0] = 0.0
    return ev, events, votes


def _sx_boundary_dp(bars, duration, ev, lam=None, min_bars=None, min_s=None, soft_max_s=None):
    """Choose section starts (bar indices) maximising evidence under a
    section-length prior. Exact dynamic programme over bar lines."""
    lam = STRUCTURE_SECTION_COST if lam is None else lam
    min_bars = STRUCTURE_MIN_SECTION_BARS if min_bars is None else min_bars
    min_s = STRUCTURE_MIN_SECTION_S if min_s is None else min_s
    soft_max_s = STRUCTURE_SOFT_MAX_SECTION_S if soft_max_s is None else soft_max_s
    pos = np.concatenate([np.asarray(bars, dtype=float), [float(duration)]])
    N = len(pos)
    Ex = np.concatenate([np.asarray(ev, dtype=float), [0.0]])
    # At fast tempos the tracker often counts eighth notes, so a tracked
    # "bar" is half a real bar; count phrase lengths in pairs then.
    bar_len = float(np.median(np.diff(pos))) if N > 2 else 2.0
    group = 2 if bar_len < 1.6 else 1
    min_bars = int(min_bars * group)
    phrase = 4 * group
    best = np.full(N, -1e18)
    back = np.full(N, -1, dtype=int)
    best[0] = 0.0
    for j in range(1, N):
        i = np.arange(j)
        nbars = j - i
        L = pos[j] - pos[i]
        pen = np.zeros(j)
        # Too short to be a section. The final partial bar and the very first
        # (pickup) bar are exempt from the bar count, not from the seconds.
        # An intro or outro may be as short as two bars (pickup intros, a
        # final hit); everything else needs the full minimum.
        edge = (i == 0) | (j == N - 1)
        pen += np.where((nbars < min_bars) & ~edge, 3.0, 0.0)
        pen += np.where(edge & (nbars < max(2, min_bars // 2)), 3.0, 0.0)
        pen += np.where(L < np.where(edge, 4.0, min_s), 3.0, 0.0)
        pen += np.where(L > soft_max_s, ((L - soft_max_s) / 15.0) ** 2, 0.0)
        # Sections are usually 4, 8 or 16 bars long; a mild preference only,
        # so a strong cue still wins on an irregular phrase.
        pen += np.where((nbars % phrase == 0) | (j == N - 1) | (i == 0), 0.0, float(STRUCTURE_PHRASE_PENALTY))
        # Most pop/rock sections run 12 s or more; shorter ones need clearly
        # stronger evidence (a 2-bar break still gets through on its events).
        pen += np.where((L < 12.0) & (j < N - 1) & (i > 0), 0.35 * (12.0 - L) / 6.0, 0.0)
        cand = best[:j] + Ex[j] - lam - pen
        k = int(np.argmax(cand))
        best[j] = cand[k]
        back[j] = k
    path = []
    j = N - 1
    while j > 0:
        path.append(j)
        j = back[j]
    return sorted(k for k in path if k < N - 1)


def _sx_shift_off_fill_bar(starts, events, nb, bars=None, duration=None):
    """Move a boundary from the start of a fill bar to the bar line the fill
    leads into.

    A drum fill closes a section; the new one starts on the downbeat after
    it. The fill bar itself is often a timbre/register change too, so the
    dynamic programme could pick its START -- an AI-generated test track's Chorus 2 came out
    at 1:34 (the fill bar) instead of 1:37 when that fill scored a little
    weaker in one transcription. A boundary that is itself anchored by a
    vocal entry or a drum/kick entry is left alone.
    """
    if not events:
        return list(starts)
    fill = events.get("fill")
    if fill is None:
        return list(starts)
    chosen = sorted(set(starts))
    edges = None
    if bars is not None and duration is not None:
        edges = np.concatenate([np.asarray(bars, dtype=float), [float(duration)]])
        bar_len = float(np.median(np.diff(edges))) if edges.size > 2 else 2.0
        group = 2 if bar_len < 1.6 else 1
        min_bars = int(STRUCTURE_MIN_SECTION_BARS * group)

    def _fits(j, nxt):
        # The shortened section (j .. next start) must still satisfy the
        # length limits the dynamic programme enforced.
        if edges is None:
            return True
        last = nxt >= nb
        nbars = nxt - j
        length = float(edges[min(nxt, nb)] - edges[j])
        if last:
            return nbars >= max(2, min_bars // 2) and length >= 4.0
        return nbars >= min_bars and length >= float(STRUCTURE_MIN_SECTION_S)

    out = []
    for idx, i in enumerate(chosen):
        j = i + 1
        nxt = chosen[idx + 1] if idx + 1 < len(chosen) else nb
        anchored = any(
            (events.get(k) is not None and i < len(events[k]) and events[k][i] >= 0.5)
            for k in ("vocal_in", "drum_in", "kick_in")
        )
        if (not anchored and j < nb and j not in chosen and j < len(fill) and fill[j] >= 0.3
                and fill[i] < 0.3 and _fits(j, nxt)):
            out.append(j)
        else:
            out.append(i)
    return sorted(set(out))


def _sx_shift_to_reentry(starts, bars, F, duration, events=None):
    """Move a boundary from the start of a short break to where the groove
    comes back.

    A "stop" -- the drums and most of the band dropping out for a bar or
    two -- is heard as the END of the section it closes; the next section
    starts when the band re-enters. Timbre novelty peaks at the drop, so the
    boundary would otherwise land a couple of bars early. A break must be
    quieter (or kick-less) than the music on BOTH sides of it, so a section
    that is simply softer than the one before is not mistaken for one.
    """
    nb = len(bars)
    if nb < 6:
        return list(starts)
    edges = np.concatenate([np.asarray(bars, dtype=float), [duration]])
    loud = np.asarray(F["loud"], dtype=float)
    kick = np.asarray(F["kick"], dtype=float) if F.get("has_stems") else None
    chosen = set(starts)
    out = []
    for i in starts:
        new_i = i
        anchored = False
        if events:
            for key in ("fill", "vocal_in"):
                arr = events.get(key)
                if arr is not None and i < len(arr) and arr[i] >= 0.5:
                    anchored = True
        before = loud[max(0, i - 4):i]
        if before.size >= 2 and not anchored:
            ref_b = float(np.median(before))
            kb = kick[max(0, i - 4):i] if kick is not None else None
            for k in (1, 2, 3):
                j = i + k                      # candidate re-entry bar
                if j >= nb or (edges[j] - edges[i]) > 7.0:
                    break
                after = loud[j:min(nb, j + 3)]
                if after.size < 2:
                    break
                ref = min(ref_b, float(np.median(after)))
                dip = loud[i:j]
                quiet = bool(np.all(dip <= ref - 2.5))
                kickless = False
                if kick is not None and kb is not None and kb.size:
                    ka = kick[j:min(nb, j + 3)]
                    kref = min(float(np.median(kb)), float(np.median(ka)) if ka.size else 0.0)
                    kickless = (kref >= 1.0 and bool(np.all(kick[i:j] <= 0.25 * kref))
                                and bool(np.all(dip <= ref - 1.0)))
                if (quiet or kickless) and j not in chosen:
                    new_i = j
                    break
        out.append(new_i)
    return sorted(set(out))


def _sx_silence_bounds(y, sr):
    """(music_start, music_end) in seconds, excluding digital silence of at
    least STRUCTURE_SILENCE_MIN_S at either end. Fades are kept as music."""
    try:
        hop = 512
        rms = librosa.feature.rms(y=y, hop_length=hop)[0]
        db = 20.0 * np.log10(np.maximum(rms, 1e-10))
        loud = float(np.percentile(db, 95))
        thr = max(STRUCTURE_SILENCE_ABS_DB, loud - STRUCTURE_SILENCE_REL_DB)
        audible = np.where(db > thr)[0]
        dur = len(y) / float(sr)
        if audible.size == 0:
            return 0.0, dur
        t0 = float(librosa.frames_to_time(audible[0], sr=sr, hop_length=hop))
        t1 = float(librosa.frames_to_time(audible[-1] + 1, sr=sr, hop_length=hop))
        start = t0 if t0 >= STRUCTURE_SILENCE_MIN_S else 0.0
        end = t1 if (dur - t1) >= STRUCTURE_SILENCE_MIN_S else dur
        return start, min(end, dur)
    except Exception:
        return 0.0, len(y) / float(sr) if y is not None else 0.0


def _sx_chroma_seq_sim(Cz, sa, sb, min_frac=0.6, R=None):
    """Best aligned chord-sequence similarity of two bar ranges: the shorter
    one is slid along the longer and the mean bar-to-bar similarity of the
    best alignment is returned. Alignment matters -- a chorus that comes back
    with a two-bar pickup still matches itself, while a verse and a chorus
    over the same key do not."""
    (a0, a1), (b0, b1) = sa, sb
    la, lb = a1 - a0, b1 - b0
    if la <= 0 or lb <= 0:
        return 0.0
    if la < lb:
        (a0, a1), (b0, b1), la, lb = (b0, b1), (a0, a1), lb, la
    need = max(2, int(np.ceil(min_frac * lb)))
    best = -1.0
    jj = np.arange(lb) + b0
    for off in range(-(lb - need), la - need + 1):
        ii = np.arange(lb) + a0 + off
        ok = (ii >= a0) & (ii < a1)
        if int(ok.sum()) < need:
            continue
        if R is not None:
            v = float(np.mean(R[ii[ok], jj[ok]]))
        else:
            v = float(np.mean(np.sum(Cz[:, ii[ok]] * Cz[:, jj[ok]], axis=0)))
        if v > best:
            best = v
    return best


def _sx_segment_similarity(F, segs, lyric_sig=None, lyric_rep=None, lyric_ng=None):
    """Pairwise 0..1 "same material" score for segments (bar ranges).

    Four independent views of a section, combined so that no single one can
    make two different sections look identical:
      * harmony  -- best-aligned chord-sequence similarity;
      * timbre   -- similarity of the mean (track-normalised) MFCC profile,
                    i.e. the arrangement / instrumentation;
      * register -- how close the sung melody sits (median vocal pitch), when
                    a vocal stem was transcribed. Verse and chorus very often
                    share chords and differ by where the voice sits;
      * level    -- loudness difference.
    Shared repeated lyrics (when available) can only raise the score.
    """
    n = len(segs)
    Cz = _sx_chroma_unit(F)
    Zt = _sx_unit_cols(F["mfcc"])
    loud = np.asarray(F["loud"], dtype=float)
    vpb = np.asarray(F.get("vpitch"), dtype=float) if F.get("has_vocal") else None
    means, lev, reg = [], [], []
    for a, b in segs:
        b = max(b, a + 1)
        v = Zt[:, a:b].mean(axis=1)
        means.append(v / (np.linalg.norm(v) + 1e-9))
        lev.append(float(np.mean(loud[a:b])))
        if vpb is not None:
            vv = vpb[a:b]
            vv = vv[~np.isnan(vv)]
            reg.append(float(np.median(vv)) if vv.size >= 2 else None)
        else:
            reg.append(None)
    W = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            cs = max(0.0, _sx_chroma_seq_sim(Cz, segs[i], segs[j]))
            ts = (float(means[i] @ means[j]) + 1.0) / 2.0
            ls = float(np.exp(-max(0.0, abs(lev[i] - lev[j]) - 1.0) / 2.0))
            # Weighted GEOMETRIC mean: every view must agree. With an
            # arithmetic mean, two sections over the same chord loop scored
            # as "the same" even when arrangement, level and vocal register
            # all said otherwise.
            if reg[i] is not None and reg[j] is not None:
                vs = float(np.exp(-max(0.0, abs(reg[i] - reg[j]) - 1.5) / 2.0))
                parts = ((cs, 0.42), (ts, 0.28), (vs, 0.20), (ls, 0.10))
            else:
                parts = ((cs, 0.50), (ts, 0.35), (ls, 0.15))
            w = float(np.exp(sum(wt * np.log(max(v, 0.02)) for v, wt in parts)))
            if lyric_sig and lyric_sig[i] and lyric_sig[j]:
                inter = len(lyric_sig[i] & lyric_sig[j])
                union = len(lyric_sig[i] | lyric_sig[j])
                # Only when repeated words make up most of BOTH parts: one
                # chorus line spilling into the next verse (timing jitter,
                # a pickup) must not make that verse "the same as" the chorus.
                both_mostly = (lyric_rep is not None
                               and lyric_rep[i] is not None and lyric_rep[j] is not None
                               and lyric_rep[i] >= 0.5 and lyric_rep[j] >= 0.5)
                if union and inter / union >= 0.5 and both_mostly:
                    w = max(w, 0.85)
            if lyric_ng and len(lyric_ng[i]) >= STRUCTURE_LYRIC_MIN_NGRAMS \
                    and len(lyric_ng[j]) >= STRUCTURE_LYRIC_MIN_NGRAMS:
                # Most of the sung phrases are shared both ways: the same
                # lyric section (every chorus), even when the arrangement
                # differs -- a stripped-down opening chorus, a quiet last one.
                o_min, o_max = _sx_lyric_overlap(lyric_ng[i], lyric_ng[j])
                if o_min >= 0.6 and o_max >= 0.4:
                    w = max(w, 0.85)
            # (Lyrics only ever RAISE the score. Lowering it when one part's
            # words repeat and the other's do not was tried and dropped:
            # Whisper often merges lines, which hides real repeats, so the
            # absence of repeated words is not reliable evidence.)
            W[i, j] = W[j, i] = w
    return W


def _sx_cluster(W, idxs, thr=None):
    """Average-linkage clustering of the segments listed in idxs; merging
    stops when the best remaining pair of groups is less alike than thr."""
    thr = STRUCTURE_SAME_MATERIAL_SIM if thr is None else thr
    groups = [[i] for i in idxs]

    def _link(g1, g2):
        return float(np.mean([W[a, b] for a in g1 for b in g2]))

    while len(groups) > 1:
        best = None
        for x in range(len(groups)):
            for y_ in range(x + 1, len(groups)):
                l = _link(groups[x], groups[y_])
                if best is None or l > best[0]:
                    best = (l, x, y_)
        if best is None or best[0] < thr:
            break
        _l, x, y_ = best
        groups[x] = groups[x] + groups[y_]
        del groups[y_]
    labels = {}
    for gi, g in enumerate(sorted(groups, key=min)):
        for i in g:
            labels[i] = gi
    return labels


def _sx_lyric_signatures(segs, bar_edges, lyric_chunks, lyric_groups):
    """Per segment: set of repeated-lyric group ids sung inside it, and the
    fraction of its sung lines whose words come back somewhere else."""
    sigs = [set() for _ in segs]
    rep_frac = [None for _ in segs]
    if not lyric_chunks:
        return sigs, rep_frac
    occ = {}
    for gid, g in enumerate(lyric_groups or []):
        for st, _en in g.get("occurrences") or []:
            occ[round(float(st), 2)] = gid
    tot = [0 for _ in segs]
    rep = [0 for _ in segs]
    for c in lyric_chunks:
        try:
            s = float(c.get("start", -1))
        except Exception:
            continue
        if s < 0:
            continue
        if len(_normalize_lyric_line(c.get("text", "")).split()) < 2:
            continue
        if STRUCTURE_LYRIC_REP_SKIP_VOCABLES:
            # dev26: a line of only "oh"/"yeah"/"na" says nothing about which
            # part it belongs to, however often it comes back.
            _w = [w.strip("'-") for w in re.sub(r"[^\w'\- ]+", " ", str(c.get("text") or "").lower()).split()]
            _w = [w for w in _w if w]
            if _w and all(w in _SX_VOCABLES for w in _w):
                continue
        for k, (a, b) in enumerate(segs):
            if bar_edges[a] - 0.5 <= s < bar_edges[b] - 0.5:
                tot[k] += 1
                g = occ.get(round(s, 2))
                if g is not None:
                    rep[k] += 1
                    sigs[k].add(g)
                break
    for k in range(len(segs)):
        if tot[k] >= 2:
            rep_frac[k] = rep[k] / float(tot[k])
    return sigs, rep_frac


# Three-word phrases made only of these say nothing about which section a
# line belongs to ("and i was", "all of the" are everywhere).
_SX_LYRIC_STOPWORDS = frozenset(
    "a an and are as at be but by for from have he her his i i'm in is it it's its me "
    "my no not of oh on or our she so that the their them they this to up was we were "
    "what when with yeah you your".split()
)


def _sx_sung_time_mapper(vocal_notes, t0, t1):
    """f(frac) -> time inside [t0, t1], spreading a chunk's words over the
    stretches where the vocal stem is actually sounding rather than evenly
    across the chunk (a merged chunk can span an instrumental gap). None when
    there is too little transcribed voice inside the chunk to go by."""
    if not vocal_notes or t1 - t0 <= 0.5:
        return None
    iv = []
    for x in vocal_notes:
        sp = _sx_note_span(x) if isinstance(x, dict) else None
        if sp is None:
            continue
        a, b = max(t0, sp[0]), min(t1, sp[1])
        if b > a:
            iv.append((a, b))
    if not iv:
        return None
    iv.sort()
    merged = [list(iv[0])]
    for a, b in iv[1:]:
        if a <= merged[-1][1] + 0.35:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    total = sum(b - a for a, b in merged)
    if total < 0.2 * (t1 - t0):
        return None

    def f(frac):
        target = frac * total
        acc = 0.0
        for a, b in merged:
            if acc + (b - a) >= target:
                return a + (target - acc)
            acc += b - a
        return merged[-1][1]
    return f


def _sx_split_tag_lines(lyric_chunks, vocal_notes=None):
    """Split chunks that were timed against the file's lyrics (they carry
    "lines": word indices where each tag line starts) into one chunk per
    line, each placed where the vocal stem is singing within the chunk (or
    spread evenly when there is too little transcribed voice), and give each
    line its stanza. A line whose words occur in more than one stanza (a
    hook that ends the bridge and also closes the chorus) takes the stanza
    of the line before it when it can. Unmatched chunks are kept whole,
    without a stanza."""
    out = []
    for c in lyric_chunks or []:
        if not isinstance(c, dict):
            continue
        cuts = c.get("lines")
        stz = c.get("line_stanzas")
        words = str(c.get("text") or "").split()
        try:
            t0 = float(c.get("start"))
            t1 = max(float(c.get("end", t0)), t0)
        except Exception:
            continue
        if not cuts or not stz or len(cuts) != len(stz) + 1 or len(words) < 2:
            d = {k: v for k, v in c.items() if k not in ("lines", "line_stanzas")}
            out.append(d)
            continue
        place = _sx_sung_time_mapper(vocal_notes, t0, t1)
        n = float(len(words))

        def _at(frac):
            return place(frac) if place is not None else t0 + (t1 - t0) * frac

        first = True
        for (a, b), cands in zip(zip(cuts[:-1], cuts[1:]), stz):
            if b <= a:
                continue
            # Only the chunk's first line is timed by Whisper itself; the
            # others are placed by interpolation and can be seconds off.
            out.append({"start": float(_at(a / n)), "end": float(_at(b / n)),
                        "text": " ".join(words[a:b]), "_stanzas": list(cands),
                        "chunk_first": first})
            first = False
    out.sort(key=lambda x: x["start"])
    prev = prev_idx = None
    for ln in out:
        cands = [tuple(x) for x in (ln.pop("_stanzas", None) or [])]
        if not cands:
            prev = prev_idx = None
            continue
        pick = cands[0]
        nxt_line = [x for x in cands if len(x) > 2 and prev_idx is not None and x[2] == prev_idx + 1]
        if nxt_line:
            # The line that follows the previous one in the lyrics: a line
            # repeated inside its stanza is not that stanza starting again.
            pick = nxt_line[0]
        elif prev is not None:
            same = [x for x in cands if x[0] == prev]
            if same:
                # Same stanza again from its top (a double chorus) when the
                # line is the stanza's first; otherwise where it sits in it.
                pick = ([x for x in same if x[1]] or same)[0]
        ln["stanza"], ln["stanza_first"] = int(pick[0]), bool(pick[1])
        prev = ln["stanza"]
        prev_idx = pick[2] if len(pick) > 2 else None
    return out


def _lyric_blocks_from_tag_lines(lines_in, gap_s=6.0):
    """Lyric blocks for lines timed against the file's lyrics, from the
    lyrics' own stanzas: a block starts where the sung stanza changes, where
    a stanza's first line comes round again (a double chorus), or after a gap
    of more than gap_s seconds. So a chorus's second line can never start a
    block (Whisper dropping the first line made the second look like an
    opener), and a hook ending the bridge belongs to the bridge. A stanza
    sung more than once is "repeated" (stronger evidence). Lines without a
    stanza join the block they fall in. Returns None when the lyrics have
    fewer than three stanzas, so the caller keeps the line-repetition
    blocks. Same output format as _lyric_block_segmentation."""
    lines = []
    for c in lines_in or []:
        try:
            st_ = float(c.get("start", -1))
            en_ = float(c.get("end", st_))
        except Exception:
            continue
        if st_ < 0:
            continue
        lines.append((st_, en_, c.get("stanza"), bool(c.get("stanza_first"))))
    lines.sort(key=lambda x: x[0])
    if len({ln[2] for ln in lines if ln[2] is not None}) < 3:
        return None
    blocks, cur, cur_st = [], [lines[0]], lines[0][2]
    for prev, ln in zip(lines, lines[1:]):
        st_, _en, stz, first = ln
        new = (st_ - prev[1]) > gap_s
        if stz is not None:
            if cur_st is not None and stz != cur_st:
                new = True
            elif first and stz == cur_st and not (prev[2] == stz and prev[3]):
                new = new or any(x[2] == stz and x[3] for x in cur)
        if new:
            blocks.append((cur, cur_st))
            cur, cur_st = [ln], stz
        else:
            cur.append(ln)
            if cur_st is None and stz is not None:
                cur_st = stz
    blocks.append((cur, cur_st))
    sung_count = {}
    for _b, stz in blocks:
        if stz is not None:
            sung_count[stz] = sung_count.get(stz, 0) + 1
    out = []
    for b, stz in blocks:
        rep_ = stz is not None and sung_count.get(stz, 0) >= 2
        # Only a block that opens on the stanza's first line marks where the
        # stanza starts; when Whisper missed that line the block starts a line
        # late, so it is weak evidence.
        opens = bool(b[0][3]) and b[0][2] == stz
        out.append({"start": float(b[0][0]), "end": float(b[-1][1]),
                    "signature": frozenset([("stanza", stz)]) if (rep_ and opens) else None,
                    "sung": True, "stanza": stz, "opens": opens})
    return out


def _sx_refine_starts_to_stanzas(starts, bars, duration, lyric_blocks, lines, max_shift_s=9.0):
    """Move a section start forward onto the start of a stanza when the
    arrangement changed a little before the words did.

    A build or fill at the end of a bridge is a strong acoustic change, so
    the boundary search put the chorus there -- but while the singer is still
    on the bridge (its closing hook line), the chorus has not started. When a
    stanza that opens on its first line starts within max_shift_s after a
    boundary, and the voice in between is singing a DIFFERENT stanza, the
    boundary moves to the bar nearest the stanza's start. That singing must
    begin a Whisper chunk after the boundary (its time is then Whisper's own,
    not interpolated across a long chunk, which put a chorus's last line into
    the following bridge often enough to move correct boundaries). A stretch
    with no singing (an instrumental build) is left alone, a boundary never
    passes the next one, and the section it shortens keeps the boundary
    search's minimum length (bars and seconds)."""
    if not starts or not lyric_blocks:
        return starts
    edges = np.concatenate([np.asarray(bars, dtype=float), [float(duration)]])
    bar_len = float(np.median(np.diff(edges))) if len(edges) > 2 else 2.0
    min_b = int(STRUCTURE_MIN_SECTION_BARS) * (2 if bar_len < 1.6 else 1)
    out = list(starts)
    for n, k in enumerate(out):
        t = float(edges[k])
        nxt = out[n + 1] if n + 1 < len(out) else len(bars)
        prv = out[n - 1] if n > 0 else 0
        for blk in lyric_blocks:
            if not blk.get("opens") or blk.get("stanza") is None:
                continue
            b0 = float(blk["start"])
            if not (t + 1.0 < b0 <= t + max_shift_s):
                continue
            between = [ln for ln in lines if t - 0.25 <= float(ln.get("start", -1)) < b0 - 0.25
                       and ln.get("stanza") is not None]
            if any(ln.get("stanza") == blk.get("stanza") for ln in between):
                continue
            # The other stanza's singing after the boundary must be timed by
            # Whisper itself (a line that starts its chunk), not placed by
            # interpolating across a long chunk.
            if not any(ln.get("chunk_first") and float(ln["start"]) >= t + 0.25 for ln in between):
                continue
            k2 = int(np.argmin(np.abs(edges[:-1] - b0)))
            if (k < k2 and nxt - k2 >= min_b and k2 - prv >= 2
                    and float(edges[nxt] - edges[k2]) >= STRUCTURE_MIN_SECTION_S):
                out[n] = k2
            break
    return sorted(set(out))


def _sx_backshift_starts_to_stanzas(starts, bars, duration, lyric_blocks, lines, max_back_s=None):
    """dev56: see STRUCTURE_STANZA_BACKSHIFT. A section start moves back to
    the bar nearest a stanza's first line sung more than a bar (at least
    2.5 s) and at most max_back_s before it,
    when that line was timed by Whisper itself (it starts its chunk), no line
    of another stanza starts between the two, and the last stanza sung before
    the line is a different one. Boundaries never cross, and the section it
    shortens keeps the minimum length."""
    max_back_s = float(STRUCTURE_STANZA_BACKSHIFT_S if max_back_s is None else max_back_s)
    if not starts or not lyric_blocks:
        return starts
    edges = np.concatenate([np.asarray(bars, dtype=float), [float(duration)]])
    bar_len = float(np.median(np.diff(edges))) if len(edges) > 2 else 2.0
    min_b = int(STRUCTURE_MIN_SECTION_BARS) * (2 if bar_len < 1.6 else 1)
    stz_lines = sorted([ln for ln in lines or [] if ln.get("stanza") is not None],
                       key=lambda x: float(x.get("start", -1)))
    out = list(starts)
    for n, k in enumerate(out):
        t = float(edges[k])
        prv = out[n - 1] if n > 0 else 0
        for blk in lyric_blocks:
            if not blk.get("opens") or blk.get("stanza") is None:
                continue
            b0 = float(blk["start"])
            if not (t - max_back_s <= b0 <= t - max(2.5, 1.25 * bar_len)):
                continue
            first = [ln for ln in stz_lines if abs(float(ln["start"]) - b0) < 0.05]
            if not first or not first[0].get("chunk_first") or not first[0].get("stanza_first"):
                continue
            between = [ln for ln in stz_lines if b0 + 0.05 <= float(ln["start"]) < t]
            if any(ln.get("stanza") != blk.get("stanza") for ln in between):
                continue
            before = [ln for ln in stz_lines if float(ln["start"]) < b0 - 0.05]
            if before and before[-1].get("stanza") == blk.get("stanza") and not before[-1].get("stanza_first"):
                # still inside the same stanza sung just before (a repeat
                # inside one section), not a new one starting
                if b0 - float(before[-1].get("end", before[-1]["start"])) < 6.0:
                    continue
            k2 = int(np.argmin(np.abs(edges[:-1] - b0)))
            if (k2 < k and k2 - prv >= min_b
                    and float(edges[k2] - edges[prv]) >= STRUCTURE_MIN_SECTION_S):
                out[n] = k2
            break
    return sorted(set(out))


def _sx_voiced_seconds(vocal_notes, t0, t1):
    """dev55: seconds inside [t0, t1] where the vocal stem has notes (gaps
    under 0.35 s bridged), or the whole span when there are no vocal notes."""
    if not vocal_notes:
        return float(t1 - t0)
    iv = []
    for x in vocal_notes:
        sp = _sx_note_span(x) if isinstance(x, dict) else None
        if sp is None:
            continue
        a, b = max(t0, sp[0]), min(t1, sp[1])
        if b > a:
            iv.append((a, b))
    if not iv:
        return 0.0
    iv.sort()
    tot, ca, cb = 0.0, iv[0][0], iv[0][1]
    for a, b in iv[1:]:
        if a <= cb + 0.35:
            cb = max(cb, b)
        else:
            tot += cb - ca
            ca, cb = a, b
    return tot + (cb - ca)


def _sx_lyric_ngrams(segs, bar_edges, lyric_chunks, vocal_notes=None):
    """Per segment: the set of short word patterns sung inside it -- each pair
    of neighbouring words, and each pair with one word between them.

    Whisper returns long chunks -- several lines merged, split differently
    each time a chorus comes round, often running across a section boundary
    -- so matching whole chunks (the line groups above) misses most repeats:
    on the rock test track it found the chorus in only one of its three
    occurrences. Word patterns survive that. Each word is placed in time by
    interpolating across its chunk; patterns are taken within one chunk and
    one segment. Using pairs and skip-pairs means one misheard word only
    removes a few patterns. Patterns made only of very common words are
    dropped.
    """
    sets = [set() for _ in segs]
    if not lyric_chunks or not segs:
        return sets
    starts = [float(bar_edges[a]) - 0.5 for a, _b in segs]
    ends = [float(bar_edges[b]) - 0.5 for _a, b in segs]
    for c in lyric_chunks:
        try:
            t0 = float(c.get("start"))
            t1 = float(c.get("end", t0))
        except Exception:
            continue
        if t0 < 0:
            continue
        if (STRUCTURE_NGRAM_MAX_CHUNK_S and (t1 - t0) > float(STRUCTURE_NGRAM_MAX_CHUNK_S)
                and _sx_voiced_seconds(vocal_notes, t0, t1) > float(STRUCTURE_NGRAM_MAX_CHUNK_S)):
            continue    # dev55: words smeared over more singing than one section
        txt = re.sub(r"[^\w' ]+", " ", str(c.get("text") or "").lower())
        words = [w for w in txt.split() if w.strip("'")]
        if len(words) < 2:
            continue
        t1 = max(t1, t0)
        span = t1 - t0
        place = _sx_sung_time_mapper(vocal_notes, t0, t1)
        seg_of = []
        for k in range(len(words)):
            frac = (k + 0.5) / len(words)
            t = place(frac) if place is not None else t0 + span * frac
            j = None
            for q in range(len(segs)):
                if starts[q] <= t < ends[q]:
                    j = q
                    break
            seg_of.append(j)
        for k in range(len(words) - 1):
            for gap in (1, 2):
                if k + gap >= len(words):
                    continue
                j = seg_of[k]
                if j is None or seg_of[k + gap] != j:
                    continue
                w1, w2 = words[k], words[k + gap]
                if w1 in _SX_LYRIC_STOPWORDS and w2 in _SX_LYRIC_STOPWORDS:
                    continue
                sets[j].add((w1, w2) if gap == 1 else (w1, "_", w2))
    return sets


def _sx_lyric_overlap(a, b):
    """(shared / smaller set, shared / larger set) for two phrase sets."""
    if not a or not b:
        return 0.0, 0.0
    inter = len(a & b)
    return inter / float(min(len(a), len(b))), inter / float(max(len(a), len(b)))


def _sx_word_recurrence(lyric_ng, i):
    """Share of segment i's word patterns sung elsewhere in the song (not in
    its neighbours, which share smeared lines), or None with too few words."""
    ng = lyric_ng[i]
    if len(ng) < STRUCTURE_LYRIC_MIN_NGRAMS:
        return None
    others = set()
    for j in range(len(lyric_ng)):
        if abs(j - i) > 1:
            others |= lyric_ng[j]
    return len(ng & others) / float(len(ng))


def _sx_chorus_over_tags(chorus, verse, units, repeated, sung_rep, lyric_ng, late, by_hook=False):
    """(chorus, verse, tags): if the chosen chorus is a short tag and another
    word-repeating part is clearly longer and comes round at least as often,
    that part is the chorus. Tags are the other word-repeating parts whose
    shortest occurrence is short and whose total length is well below the
    chorus's. by_hook: the chorus was chosen by level/register alone (no
    part's words stood out pair by pair); if its words mostly do not come
    back and another part's mostly do, that part is the chorus."""
    def _rec(c):
        vals = [_sx_word_recurrence(lyric_ng, i) for k in repeated[c] for i in units[k]["segs"]]
        vals = [v for v in vals if v is not None]
        return float(np.mean(vals)) if vals else None

    def _total(c):
        return sum(units[k]["len"] for k in repeated[c])

    def _shortest(c):
        return min(units[k]["len"] for k in repeated[c])

    wordy = [c for c in sung_rep if c not in late and (_rec(c) or 0.0) >= STRUCTURE_TAG_WORDS_MIN]
    if by_hook and STRUCTURE_CHORUS_WORDS_FALLBACK and chorus not in wordy:
        r0 = _rec(chorus)
        alt = [c for c in wordy if c != chorus]
        if r0 is not None and r0 <= STRUCTURE_HOOK_CHORUS_WORDS_MAX and alt:
            new_chorus = max(alt, key=_total)
            if verse == new_chorus or verse is None:
                verse = chorus
            chorus = new_chorus
    if not STRUCTURE_CHORUS_PREFERS_LONGER:
        return chorus, verse, set()
    if chorus in wordy and _shortest(chorus) <= STRUCTURE_TAG_MAX_S:
        longer = [c for c in wordy if c != chorus
                  and _total(c) >= STRUCTURE_TAG_LENGTH_RATIO * _total(chorus)
                  and len(repeated[c]) >= len(repeated[chorus])
                  and _shortest(c) > _shortest(chorus)]
        if longer:
            new_chorus = max(longer, key=_total)
            if verse == new_chorus:
                rest = [c for c in sung_rep if c not in (new_chorus, chorus) and c not in wordy]
                verse = max(rest, key=_total) if rest else None
            chorus = new_chorus
    tags = set(c for c in wordy if c != chorus and c != verse
               and _shortest(c) <= STRUCTURE_TAG_MAX_S
               and _total(c) * STRUCTURE_TAG_LENGTH_RATIO <= _total(chorus))
    return chorus, verse, tags


# dev51: see STRUCTURE_SPLIT_BY_COHESION.
def _sx_cluster_last_merges(W, idxs, thr=None):
    """For each group _sx_cluster(W, idxs, thr) produces, the merge that
    formed it last: {frozenset(group): (link, part_a, part_b)}. Same
    average-linkage steps as _sx_cluster."""
    thr = STRUCTURE_SAME_MATERIAL_SIM if thr is None else thr
    groups = [[i] for i in idxs]
    last = {}

    def _link(g1, g2):
        return float(np.mean([W[a, b] for a in g1 for b in g2]))

    while len(groups) > 1:
        best = None
        for x in range(len(groups)):
            for y_ in range(x + 1, len(groups)):
                l = _link(groups[x], groups[y_])
                if best is None or l > best[0]:
                    best = (l, x, y_)
        if best is None or best[0] < thr:
            break
        l, x, y_ = best
        a, b = list(groups[x]), list(groups[y_])
        groups[x] = a + b
        del groups[y_]
        last[frozenset(groups[x])] = (l, a, b)
    return {frozenset(g): last.get(frozenset(g)) for g in groups}


def _sx_regroup_lone_sung(W, labels, cov, lyric_ng=None):
    """dev51: see STRUCTURE_LONE_SUNG_REGROUP. Returns new labels. With
    lyrics, a segment whose words mostly come back is not moved into a group
    whose words mostly do not, nor the reverse."""
    def _rec(i):
        return _sx_word_recurrence(lyric_ng, i) if lyric_ng else None

    def _words_clash(i, g2):
        ri = _rec(i)
        rg = [v for v in (_rec(j) for j in g2) if v is not None]
        if ri is None or not rg:
            return False
        rm = float(np.mean(rg))
        lo, hi = min(ri, rm), max(ri, rm)
        return (lo <= STRUCTURE_WORDS_NEW_MAX and hi >= STRUCTURE_WORDS_SAME_MIN
                and hi - lo >= STRUCTURE_WORDS_GAP)

    groups = {}
    for i, l in labels.items():
        groups.setdefault(l, []).append(i)

    def _sung(i):
        return cov[i] >= STRUCTURE_SUNG_MIN_COVERAGE

    new = dict(labels)
    for l, g in sorted(groups.items()):
        s_ = [i for i in g if _sung(i)]
        if len(s_) != 1 or len(g) < 2:
            continue
        best = None
        for l2, g2 in groups.items():
            if l2 == l or len(g2) < 2 or not all(_sung(j) for j in g2):
                continue
            if _words_clash(s_[0], g2):
                continue
            link = float(np.mean([W[s_[0], j] for j in g2]))
            if best is None or link > best[0]:
                best = (link, l2)
        if best is not None and best[0] >= STRUCTURE_SAME_MATERIAL_SIM:
            new[s_[0]] = best[1]
    return new


def _sx_split_by_cohesion(W, body, labels, lyric_ng, cov):
    """Returns (labels, new_words_labels)."""
    n = len(lyric_ng)

    def _recur(i):
        ng = lyric_ng[i]
        if len(ng) < STRUCTURE_LYRIC_MIN_NGRAMS:
            return None
        others = set()
        for j in range(n):
            if abs(j - i) > 1:      # neighbours share smeared lines
                others |= lyric_ng[j]
        return len(ng & others) / float(len(ng))

    def _coh(g):
        return float(np.mean([W[a, b] for a in g for b in g if a < b]))

    def _runs(g):
        g = sorted(g)
        return 1 + sum(1 for a, b in zip(g, g[1:]) if b != a + 1)

    def _sung_runs(g):
        sg = [i for i in g if cov[i] >= STRUCTURE_SUNG_MIN_COVERAGE]
        return len(sg), (_runs(sg) if sg else 0)

    groups = {}
    for i, l in labels.items():
        groups.setdefault(l, []).append(i)
    new = dict(labels)
    made = set()
    nxt = (max(labels.values()) + 1) if labels else 0
    for _l, members in sorted(groups.items()):
        if len(members) < 4:
            continue
        # The group's own top split: the same average-linkage steps run on its
        # members alone, merged all the way.
        lm = list(_sx_cluster_last_merges(W, sorted(members), thr=-1.0).values())[0]
        if lm is None:
            continue
        link, ga, gb = lm
        if len(ga) < 2 or len(gb) < 2 or _runs(ga) < 2 or _runs(gb) < 2:
            continue
        if min(_coh(ga), _coh(gb)) - link < STRUCTURE_COHESION_MARGIN:
            continue
        # Each side must be recurring SUNG material on its own: two or more
        # sung segments at separate points in the song.
        if min(_sung_runs(ga)[1], _sung_runs(gb)[1]) < 2:
            continue
        ra = [v for v in (_recur(i) for i in ga) if v is not None]
        rb = [v for v in (_recur(i) for i in gb) if v is not None]
        if not ra or not rb:
            continue
        ma, mb = float(np.mean(ra)), float(np.mean(rb))
        lo, hi = (ga, gb) if ma < mb else (gb, ga)
        mlo, mhi = min(ma, mb), max(ma, mb)
        if mlo > STRUCTURE_WORDS_NEW_MAX or mhi < STRUCTURE_WORDS_SAME_MIN or mhi - mlo < STRUCTURE_WORDS_GAP:
            continue
        for i in lo:
            new[i] = nxt
        made.add(nxt)
        nxt += 1
    return new, made


def _sx_split_shared_music_by_words(labels, body, lyric_ng, sung):
    """Returns (labels, new_labels): the labels with any new-words segments
    moved to labels of their own, and the set of those new labels (parts the
    namer should treat as verse-like by their words)."""
    n = len(lyric_ng)

    def _recur(i):
        ng = lyric_ng[i]
        if len(ng) < STRUCTURE_LYRIC_MIN_NGRAMS:
            return None
        others = set()
        for j in range(n):
            if abs(j - i) > 1:      # neighbours share smeared lines
                others |= lyric_ng[j]
        return len(ng & others) / float(len(ng))

    rec = {i: _recur(i) for i in body}
    runs = []
    for i in body:
        if runs and runs[-1][-1] == i - 1 and labels[runs[-1][-1]] == labels[i]:
            runs[-1].append(i)
        else:
            runs.append([i])
    by_cl = {}
    for ru in runs:
        by_cl.setdefault(labels[ru[0]], []).append(ru)

    def _sung(c):
        return float(np.mean([bool(sung[i]) for ru in by_cl[c] for i in ru])) >= 0.5

    for c, rus in by_cl.items():
        if len(rus) >= 2 and _sung(c):
            vals = [rec[i] for ru in rus for i in ru if rec[i] is not None]
            if len(vals) >= 2 and max(vals) <= STRUCTURE_WORDS_NEW_MAX:
                return labels, set()    # a verse of its own already exists
    new = dict(labels)
    made = set()
    nxt = (max(labels.values()) + 1) if labels else 0
    for c, rus in sorted(by_cl.items()):
        if len(rus) < 2 or not _sung(c):
            continue
        members = sorted((i for ru in rus for i in ru if rec[i] is not None), key=lambda i: rec[i])
        if len(members) < 4:
            continue
        # The widest gap in the word-recurrence values splits new-words from
        # same-words segments.
        gap, cut = max((rec[members[k + 1]] - rec[members[k]], k + 1) for k in range(len(members) - 1))
        lo, hi = members[:cut], members[cut:]
        if (gap < STRUCTURE_WORDS_GAP or len(lo) < 2 or len(hi) < 2
                or rec[lo[-1]] > STRUCTURE_WORDS_NEW_MAX or rec[hi[0]] < STRUCTURE_WORDS_SAME_MIN):
            continue
        if not any((a in lo and b in hi) or (a in hi and b in lo)
                   for ru in rus for a, b in zip(ru, ru[1:])):
            continue
        for i in lo:
            new[i] = nxt
        made.add(nxt)
        nxt += 1
    return new, made


# dev48: two parts that both come back with the same words -- a chorus and its
# pre-chorus. On one arena-rock test scan the pre-chorus (0:59 and
# 1:45, words 85% the same) and the chorus (1:14, 2:02 and 3:19, 75%) tied on words, and the level/register score
# picked the pre-chorus, so the choruses were named Verse and the first one
# merged into the verse after it. A part that every time leads straight into
# another word-repeating part, which comes round more often (here also after
# the solo, with no lead-in) and runs at least as long, is that part's
# pre-chorus, not the chorus. Only between a verse and that part, so an
# opening chorus is never taken for a pre-chorus.
def _sx_drop_lead_ins(top, repeated, units):
    """The chorus candidates without any that are lead-ins to another
    candidate; never empty."""
    def _mean_len(c):
        return float(np.mean([units[k]["len"] for k in repeated[c]]))

    def _leads_into(a, b):
        # Every occurrence of a sits between other sung material (a verse)
        # and a sung occurrence of b -- the pre-chorus position.
        return (len(repeated[b]) > len(repeated[a])
                and _mean_len(b) >= _mean_len(a)
                and all(k + 1 < len(units) and units[k + 1]["cl"] == b and units[k + 1]["sung"]
                        and k >= 1 and units[k - 1]["sung"] and units[k - 1]["cl"] not in (a, b)
                        for k in repeated[a]))

    kept = [a for a in top if not any(b != a and _leads_into(a, b) for b in top)]
    return kept or list(top)


# dev50: lyric timing that has collapsed into a few huge chunks. Whisper
# sometimes returns one "line" covering a minute or two ("one clean-replay test scan": 0:00-1:09
# and 2:06-3:45; "one verse-first pop test scan": 0:00-2:05 and 2:27-3:35), and the line
# matching inside the hook score then compares meaningless blocks.
def _sx_lyric_timing_smeared(segs, bar_edges, F, lyric_chunks):
    """True when at least STRUCTURE_SMEARED_SHARE of the sung time (segments
    with vocal coverage) lies inside lyric chunks STRUCTURE_SMEARED_CHUNK_S
    long or longer."""
    long_spans = []
    for c in lyric_chunks or []:
        if not isinstance(c, dict):
            continue
        try:
            a, b = float(c.get("start")), float(c.get("end"))
        except (TypeError, ValueError):
            continue
        if b - a >= STRUCTURE_SMEARED_CHUNK_S:
            long_spans.append((a, b))
    if not long_spans:
        return False
    voc = F.get("vocal")
    if voc is None or not F.get("has_vocal"):
        return False
    voc = np.asarray(voc, dtype=float)
    sung = smeared = 0.0
    for a_i, b_i in segs:
        if b_i <= a_i or float(np.mean(voc[a_i:b_i])) < STRUCTURE_SUNG_MIN_COVERAGE:
            continue
        t0, t1 = float(bar_edges[a_i]), float(bar_edges[b_i])
        sung += t1 - t0
        ov = sum(max(0.0, min(t1, b) - max(t0, a)) for a, b in long_spans)
        smeared += min(ov, t1 - t0)
    return sung > 0 and smeared / sung >= STRUCTURE_SMEARED_SHARE


_SX_VOCABLES = frozenset(
    "oh ooh oooh ah ahh aah whoa woah wo wooh yeah yeh ya yah la na da doo dum hey ho mm mmm "
    "hmm uh huh ay ayy eh hoo uh-huh ooh-ooh oh-oh na-na la-la".split()
)


def _sx_vocable_only_segments(segs, bar_edges, lyric_chunks):
    """dev56: per segment, True when Whisper heard words in it and every one
    is a vocable ("oh", "na", "yeah"), else False. Each word is placed by
    interpolating across its chunk (as _sx_lyric_ngrams does)."""
    n = len(segs)
    heard = [0] * n
    lexical = [0] * n
    starts = [float(bar_edges[a]) for a, _b in segs]
    ends = [float(bar_edges[b]) for _a, b in segs]
    for c in lyric_chunks or []:
        try:
            t0 = float(c.get("start"))
            t1 = max(float(c.get("end", t0)), t0)
        except Exception:
            continue
        if t0 < 0:
            continue
        words = [w.strip("'-") for w in re.sub(r"[^\w'\- ]+", " ", str(c.get("text") or "").lower()).split()]
        words = [w for w in words if w]
        for k, w in enumerate(words):
            t = t0 + (t1 - t0) * (k + 0.5) / len(words)
            for q in range(n):
                if starts[q] <= t < ends[q]:
                    heard[q] += 1
                    if w not in _SX_VOCABLES:
                        lexical[q] += 1
                    break
    return [heard[q] > 0 and lexical[q] == 0 for q in range(n)], lexical, heard


def _sx_fading_last_chorus(u, prev, units, unit_names, repeated, chorus, lyric_ng, W, loud):
    """dev57: see STRUCTURE_FADING_LAST_CHORUS."""
    _cw = set()
    for k in repeated.get(chorus, []):
        if unit_names[k] == "Chorus":
            for i in units[k]["segs"]:
                _cw |= lyric_ng[i]
    mine = set()
    for i in u["segs"]:
        mine |= lyric_ng[i]
    if not _cw or len(mine) < STRUCTURE_LYRIC_MIN_NGRAMS or len(mine & _cw) < 0.5 * len(mine):
        return False
    # a whole chorus's length, not the last seconds of a fade
    _cl = [units[k]["len"] for k in repeated.get(chorus, []) if unit_names[k] == "Chorus"]
    if not _cl or u["len"] < 0.75 * float(np.median(_cl)):
        return False
    if max(W[i, j] for i in u["segs"] for j in prev["segs"]) < 0.6:
        return False
    l_u = float(np.mean([loud[i] for i in u["segs"]]))
    l_p = float(np.mean([loud[j] for j in prev["segs"]]))
    return l_u <= l_p - float(STRUCTURE_FADING_CHORUS_DB)


def _sx_timed_lyric_patterns(lyric_chunks, vocal_notes=None):
    """dev27: [(time, pattern)] -- the word patterns of _sx_lyric_ngrams
    (neighbouring pairs and skip-pairs, patterns of only common words dropped,
    chunks smeared over more than STRUCTURE_NGRAM_MAX_CHUNK_S of singing
    left out), each with the time it is sung, placed the same way. Pairs of
    vocables ("oh oh") are left out too."""
    out = []
    for c in lyric_chunks or []:
        try:
            t0 = float(c.get("start"))
            t1 = float(c.get("end", t0))
        except Exception:
            continue
        if t0 < 0:
            continue
        if (STRUCTURE_NGRAM_MAX_CHUNK_S and (t1 - t0) > float(STRUCTURE_NGRAM_MAX_CHUNK_S)
                and _sx_voiced_seconds(vocal_notes, t0, t1) > float(STRUCTURE_NGRAM_MAX_CHUNK_S)):
            continue
        txt = re.sub(r"[^\w' ]+", " ", str(c.get("text") or "").lower())
        words = [w for w in txt.split() if w.strip("'")]
        if len(words) < 2:
            continue
        t1 = max(t1, t0)
        place = _sx_sung_time_mapper(vocal_notes, t0, t1)
        ts = [(place((k + 0.5) / len(words)) if place is not None else t0 + (t1 - t0) * (k + 0.5) / len(words))
              for k in range(len(words))]
        for k in range(len(words) - 1):
            for gap in (1, 2):
                if k + gap >= len(words):
                    continue
                w1, w2 = words[k], words[k + gap]
                if w1 in _SX_LYRIC_STOPWORDS and w2 in _SX_LYRIC_STOPWORDS:
                    continue
                if w1 in _SX_VOCABLES and w2 in _SX_VOCABLES:
                    continue
                out.append((0.5 * (ts[k] + ts[k + gap]), (w1, w2) if gap == 1 else (w1, "_", w2)))
    return out


def _sx_prechorus_cuts(segs, names, bars, duration, ev, lyric_chunks, vocal_notes=None):
    """dev27 STRUCTURE_PRECHORUS_BY_WORDS: bar indices where a pre-chorus
    begins inside the verses, found by its words; [] when there is none.

    A pre-chorus is the part at the END of a verse, leading into the chorus,
    that sings the same words each time while the verse before it sings new
    ones. For every part named Verse that runs straight into a part named
    Chorus (two or more such verses), the word patterns it shares with the
    other such verses -- leaving out the chorus's own words -- are found;
    when they are packed at the verse's end (from some point on at least
    STRUCTURE_PRECHORUS_TAIL_SHARE of its patterns are shared, at least
    STRUCTURE_LYRIC_MIN_NGRAMS of them, and before it at most
    STRUCTURE_PRECHORUS_HEAD_SHARE), that point is where the pre-chorus starts,
    on the nearest bar line (an existing section line within a bar wins; else
    the best-evidenced bar). The pre-chorus lasts 8 to
    STRUCTURE_PRECHORUS_MAX_S seconds, the verse keeps at least the minimum
    section length, and at least two verses must show it with words they share
    with each other's pre-chorus. A verse whose words all come back (a repeated
    verse), or none of them, shows no such pattern, so nothing changes."""
    edges = np.concatenate([np.asarray(bars, dtype=float), [float(duration)]])
    nb = len(bars)
    bar_len = float(np.median(np.diff(edges))) if len(edges) > 2 else 2.0
    minb = int(STRUCTURE_MIN_SECTION_BARS) * (2 if bar_len < 1.6 else 1)
    units = []
    for i, nm in enumerate(names):
        if units and units[-1][0] == nm:
            units[-1][1].append(i)
        else:
            units.append([nm, [i]])
    vb = []
    for k, (nm, ix) in enumerate(units):
        if nm == "Verse" and k + 1 < len(units) and units[k + 1][0] == "Chorus":
            vb.append((segs[ix[0]][0], segs[ix[-1]][1]))
    if len(vb) < 2:
        return []
    tp = _sx_timed_lyric_patterns(lyric_chunks, vocal_notes)
    if not tp:
        return []
    ch_sp = [(float(edges[segs[ix[0]][0]]), float(edges[segs[ix[-1]][1]])) for nm, ix in units if nm == "Chorus"]
    ch_words = set(p for t, p in tp if any(a <= t < b for a, b in ch_sp))
    span = [(float(edges[a]), float(edges[b])) for a, b in vb]
    pats = [[(t, p) for t, p in tp if a <= t < b and p not in ch_words] for a, b in span]
    seg_starts = set(int(a) for a, _b in segs)

    def _cut_at(i, t):
        # the bar the line starts on: an existing section line within a
        # bar of it, else the best-evidenced bar within a bar; None when the
        # pre-chorus or the verse left before it would be too short or long
        a_b, b_b = vb[i]
        near = [k for k in range(a_b + 1, b_b) if abs(edges[k] - t) <= 1.25 * bar_len]
        old = [k for k in range(a_b + 1, b_b) if k in seg_starts and abs(edges[k] - t) <= 1.5 * bar_len]
        if not near and not old:
            return None
        k = (min(old, key=lambda q: abs(edges[q] - t)) if old
             else max(near, key=lambda q: (float(ev[q]) if ev is not None and q < len(ev) else 0.0,
                                           -abs(edges[q] - t))))
        # never leave a sliver of under two bars beside a section line
        _nbr = [q for q in seg_starts if a_b < q < b_b and q != k and abs(q - k) < 2]
        if _nbr:
            k = min(_nbr, key=lambda q: abs(q - k))
        tail_s, head_s = float(edges[b_b] - edges[k]), float(edges[k] - edges[a_b])
        if (8.0 <= tail_s <= float(STRUCTURE_PRECHORUS_MAX_S) and b_b - k >= minb // 2
                and k - a_b >= minb and head_s >= float(STRUCTURE_MIN_SECTION_S)):
            return k
        return None

    found = {}
    for i, (a_t, b_t) in enumerate(span):
        oth = set(p for j in range(len(span)) if j != i for _t, p in pats[j])
        pts = sorted((t, p in oth, p) for t, p in pats[i])
        for t, sh, _p in pts:
            if not sh:
                continue
            after = [x for x in pts if x[0] >= t]
            before = [x for x in pts if x[0] < t]
            n_sh = sum(1 for x in after if x[1])
            if n_sh < int(STRUCTURE_LYRIC_MIN_NGRAMS):
                break
            fa = n_sh / float(len(after))
            fb = (sum(1 for x in before if x[1]) / float(len(before))) if before else 0.0
            if fa >= float(STRUCTURE_PRECHORUS_TAIL_SHARE) and fb <= float(STRUCTURE_PRECHORUS_HEAD_SHARE):
                k = _cut_at(i, t)
                if k is not None:
                    found[i] = (k, set(x[2] for x in after if x[1]))
                break
    if STRUCTURE_PRECHORUS_PROPAGATE and found and len(found) < len(span):
        # dev37: see STRUCTURE_PRECHORUS_PROPAGATE -- the other verses, by the
        # words of the pre-chorus already found
        pw = set()
        for _k, shp in found.values():
            pw |= shp
        for i in range(len(span)):
            if i in found:
                continue
            pts = sorted((t, p in pw, p) for t, p in pats[i])
            for t, sh, _p in pts:
                if not sh:
                    continue
                after = [x for x in pts if x[0] >= t]
                before = [x for x in pts if x[0] < t]
                n_sh = sum(1 for x in after if x[1])
                if n_sh < int(STRUCTURE_LYRIC_MIN_NGRAMS):
                    break
                fa = n_sh / float(len(after))
                fb = (sum(1 for x in before if x[1]) / float(len(before))) if before else 0.0
                if fa >= float(STRUCTURE_PRECHORUS_TAIL_SHARE) and fb <= float(STRUCTURE_PRECHORUS_HEAD_SHARE):
                    k = _cut_at(i, t)
                    if k is not None:
                        found[i] = (k, set(x[2] for x in after if x[1]))
                    break
    if len(found) < 2:
        return []
    # each pre-chorus shares its words with another one's, not just with a verse
    keep = []
    for i, (k, shp) in found.items():
        others = set()
        for j, (k2, _s2) in found.items():
            if j != i:
                others |= set(p for t, p in pats[j] if t >= float(edges[k2]) - 0.5 * bar_len)
        if len(shp & others) >= 0.6 * len(shp):
            keep.append(k)
    return sorted(keep) if len(keep) >= 2 else []


def _sx_chorus_half(chorus, other, units, repeated, same_words, wmean=None, lyric_ng=None):
    """dev30 STRUCTURE_CHORUS_HALVES: True when `other` and the chorus are the
    two halves of one chorus -- both bring their words back
    (STRUCTURE_CHORUS_HALF_SAME_WORDS), every occurrence of one sits straight
    before (or straight after) an occurrence of the other, every occurrence of
    that other one without its partner next to it is long enough to hold both
    halves (0.8x the shortest pair), and the two halves of a pair are equally
    loud (within a dB) in two thirds of the pairs or more (one stripped-down
    pass, or the song's fading last unit, does not decide). A pre-chorus
    builds up to its chorus: it is quieter, or sung well below it
    (STRUCTURE_CHORUS_HALF_PRECHORUS_ST semitones or more lower, by the
    median vocal pitch), and is not a half. Returns the half sung first
    (a cluster id), or False. With `lyric_ng`, every worded
    occurrence of each half must sing words another occurrence of it sings
    (half of its word patterns or more): a verse that only picks up its
    pre-chorus lines in some passes is not a chorus half. `wmean` is unused."""
    if other is None or chorus is None or other not in repeated or chorus not in repeated:
        return False
    mn = float(STRUCTURE_CHORUS_HALF_SAME_WORDS)
    if same_words.get(other, 0.0) < mn or same_words.get(chorus, 0.0) < mn:
        return False
    if lyric_ng is not None:
        # every worded occurrence of each half sings words another occurrence
        # of it sings (half of its patterns or more) -- not only most of them
        for c in (chorus, other):
            occ_sets = []
            for k in repeated[c]:
                ss = set()
                for i in units[k]["segs"]:
                    if i < len(lyric_ng):
                        ss |= lyric_ng[i]
                if len(ss) >= STRUCTURE_LYRIC_MIN_NGRAMS:
                    occ_sets.append(ss)
            if len(occ_sets) < 2:
                return False
            for q, a_ in enumerate(occ_sets):
                if max(_sx_lyric_overlap(a_, b_)[0] for r_, b_ in enumerate(occ_sets) if r_ != q) < 0.5:
                    return False
    for x, y in ((other, chorus), (chorus, other)):
        occ, ch = repeated[x], set(repeated[y])
        if len(occ) < 2:
            continue
        for step in (1, -1):
            if not all(k + step in ch for k in occ):
                continue
            pair = [units[k]["len"] + units[k + step]["len"] for k in occ]
            partner = set(k + step for k in occ)
            if any(k not in partner and units[k]["len"] < 0.8 * min(pair) for k in ch):
                continue
            last = len(units) - 1
            judged = [k for k in occ if k != last and k + step != last]
            if not judged:
                continue
            level = sum(1 for k in judged
                        if not (np.isnan(units[k]["loud"]) or np.isnan(units[k + step]["loud"]))
                        and abs(units[k]["loud"] - units[k + step]["loud"]) <= 1.0)
            if level < (2.0 / 3.0) * len(judged):
                continue
            first, second = (x, y) if step == 1 else (y, x)
            vf = [units[k]["vp"] for k in repeated[first] if not np.isnan(units[k]["vp"])]
            vs = [units[k]["vp"] for k in repeated[second] if not np.isnan(units[k]["vp"])]
            if (vf and vs and float(np.median(vf))
                    <= float(np.median(vs)) - float(STRUCTURE_CHORUS_HALF_PRECHORUS_ST)):
                return False     # sung well below the part it leads into: a pre-chorus
            return first         # the half sung first
    return False


def _sx_vocal_notes_of(stem_notes):
    """The vocal stem's note events from `stem_notes`, or None."""
    for _key in ("vocals", "vocal", "voice"):
        if stem_notes and stem_notes.get(_key):
            return stem_notes[_key]
    return None


def _sx_names_kept(segs, names, segs2, names2):
    """dev30 STRUCTURE_PRECHORUS_KEEPS_NAMES: True when every segment of the
    re-cut map (segs2 / names2) other than a Pre-Chorus carries the name the
    first map gave the bars it covers (by the segment's middle bar)."""
    for (a, b), nm in zip(segs2, names2):
        if nm == "Pre-Chorus":
            continue
        mid = (a + b) // 2 if b - a > 1 else a
        old = None
        for (a0, b0), nm0 in zip(segs, names):
            if a0 <= mid < b0:
                old = nm0
                break
        if old is not None and old != nm:
            return False
    return True


def _sx_inherit_names(segs, names, labels, info, segs2, cut_bars):
    """dev30 STRUCTURE_PRECHORUS_KEEPS_NAMES: the re-cut segments (segs2)
    named and grouped as the first map named the bars they start in, with the
    pre-chorus stretches marked (_sx_mark_prechorus); (names, labels, info),
    or (None, None, None) when no pre-chorus is marked that way."""
    n2, l2 = [], {}
    for j, (a, _b) in enumerate(segs2):
        i = next((q for q, (a0, b0) in enumerate(segs) if a0 <= a < b0), None)
        if i is None:
            return None, None, None
        n2.append(names[i])
        if i in labels:
            l2[j] = labels[i]
    n3, l3 = _sx_mark_prechorus(segs2, n2, l2, cut_bars)
    if "Pre-Chorus" not in [x for x, y in zip(n3, n2) if x != y]:
        return None, None, None
    info2 = dict(info or {})
    if info2.get("keep_split"):
        st = {int(a): j for j, (a, _b) in enumerate(segs2)}
        info2["keep_split"] = set(st[int(segs[i][0])] for i in info2["keep_split"]
                                  if 0 <= i < len(segs) and int(segs[i][0]) in st)
    return n3, l3, info2


def _sx_prechorus_cuts_buildup(segs, names, bars, duration, F, ev=None):
    """dev35 STRUCTURE_PRECHORUS_BUILDUP_CUTS: bar indices where a pre-chorus
    begins inside the verses, found by the music building up to the chorus:
    for each part named Verse that runs straight into a part named Chorus,
    the bar from which the voice sings STRUCTURE_PRECHORUS_BUILDUP_ST
    semitones or more higher (median pitch) than before it and the drums play
    more, leaving 8 s to STRUCTURE_PRECHORUS_MAX_S for the pre-chorus and the
    minimum section length for the verse. The cut must show in two or more
    such verses, with pre-chorus lengths within a third of each other (a
    pre-chorus comes back the same length); [] otherwise."""
    if not F or F.get("vpitch") is None or not F.get("has_stems"):
        return []
    vp = np.asarray(F["vpitch"], dtype=float)
    dr = np.asarray(F["drums"], dtype=float)
    edges = np.concatenate([np.asarray(bars, dtype=float), [float(duration)]])
    bar_len = float(np.median(np.diff(edges))) if len(edges) > 2 else 2.0
    minb = int(STRUCTURE_MIN_SECTION_BARS) * (2 if bar_len < 1.6 else 1)
    seg_starts = set(int(a) for a, _b in segs)
    units = []
    for i, nm in enumerate(names):
        if units and units[-1][0] == nm:
            units[-1][1].append(i)
        else:
            units.append([nm, [i]])
    found = []
    for k, (nm, ix) in enumerate(units):
        if nm != "Verse" or k + 1 >= len(units) or units[k + 1][0] != "Chorus":
            continue
        a_b, b_b = segs[ix[0]][0], segs[ix[-1]][1]
        best = None
        for q in range(a_b + minb, b_b - max(1, minb // 2) + 1):
            tail_s, head_s = float(edges[b_b] - edges[q]), float(edges[q] - edges[a_b])
            if not (8.0 <= tail_s <= float(STRUCTURE_PRECHORUS_MAX_S)) or head_s < float(STRUCTURE_MIN_SECTION_S):
                continue
            h, t = vp[a_b:q], vp[q:b_b]
            h, t = h[~np.isnan(h)], t[~np.isnan(t)]
            if h.size < 2 or t.size < 2:
                continue
            rise = float(np.median(t) - np.median(h))
            if rise < float(STRUCTURE_PRECHORUS_BUILDUP_ST):
                continue
            if not (float(np.nanmean(dr[q:b_b])) > float(np.nanmean(dr[a_b:q]))):
                continue
            score = rise + (0.5 if q in seg_starts else 0.0)
            if best is None or score > best[0]:
                best = (score, q, tail_s)
        if best is not None:
            found.append((best[1], best[2]))
    if len(found) < 2:
        return []
    ls = [t for _q, t in found]
    if max(ls) > (4.0 / 3.0) * min(ls):
        return []
    return sorted(q for q, _t in found)


def _sx_mark_prechorus(segs, names, labels, cut_bars):
    """dev27: name the segments from each pre-chorus line up to the chorus
    Pre-Chorus, as one material of their own; (names, labels). A line whose
    stretch no longer leads straight into a Chorus is left as named."""
    names, labels = list(names), dict(labels)
    nxt = (max(labels.values()) + 1) if labels else 0
    cuts = set(int(k) for k in cut_bars)
    marked = False
    for i, (a, _b) in enumerate(segs):
        if a not in cuts or names[i] != "Verse":
            continue
        j = i
        while j + 1 < len(segs) and names[j + 1] == "Verse":
            j += 1
        if j + 1 < len(segs) and names[j + 1] == "Chorus":
            for q in range(i, j + 1):
                names[q] = "Pre-Chorus"
                labels[q] = nxt
            marked = True
    return names, labels


def _sx_eject_unshared_words(body, labels, lyric_ng, sung):
    """dev28 STRUCTURE_EJECT_UNSHARED_WORDS: in a group whose occurrences sing
    each other's words (two or more of them sharing half their word patterns
    or more -- a chorus), a sung occurrence between the first and the last
    that sings two lines or so of words (2 x STRUCTURE_LYRIC_MIN_NGRAMS
    patterns) of which almost none (STRUCTURE_UNSHARED_WORDS_MAX) come back
    from the group's other occurrences is not another chorus but a part of
    its own (a bridge over the chorus's chords); its neighbours' words, which
    Whisper can smear across the line between them, are not counted."""
    mn = int(STRUCTURE_LYRIC_MIN_NGRAMS)
    groups = {}
    for i in body:
        groups.setdefault(labels[i], []).append(i)
    new = dict(labels)
    nxt = (max(labels.values()) + 1) if labels else 0
    for _l, mem in sorted(groups.items()):
        mem = sorted(mem)
        worded = [j for j in mem if j < len(lyric_ng) and len(lyric_ng[j]) >= mn]
        if len(worded) < 3:
            continue
        for i in mem[1:-1]:
            if not sung[i] or i >= len(lyric_ng) or len(lyric_ng[i]) < 2 * mn:
                continue
            rest = [j for j in worded if j != i]
            # the rest must repeat their own words (a chorus)
            rep = [(a, b) for k, a in enumerate(rest) for b in rest[k + 1:]
                   if _sx_lyric_overlap(lyric_ng[a], lyric_ng[b])[0] >= 0.5]
            if not rep:
                continue
            far = [j for j in rest if abs(j - i) > 1]
            if not far:
                continue
            gw = set()
            for j in far:
                gw |= lyric_ng[j]
            if len(lyric_ng[i] & gw) <= float(STRUCTURE_UNSHARED_WORDS_MAX) * len(lyric_ng[i]):
                new[i] = nxt
                nxt += 1
    return new


def _sx_key_change(F, bars, duration):
    """dev28 STRUCTURE_KEY_CHANGE_ALIGN: (bar index, semitones) of a key
    change -- the song moving up or down a semitone or more for its last part
    (a final chorus "up a key") -- or None. Found from the bar chroma: the
    average chroma after some bar line in the song's second half is the
    average before it shifted by k semitones (correlation
    STRUCTURE_KEY_CHANGE_MIN_CORR or more, unshifted at most
    STRUCTURE_KEY_CHANGE_MAX_UNSHIFTED, the shift gaining
    STRUCTURE_KEY_CHANGE_MIN_GAIN), with at least STRUCTURE_KEY_CHANGE_MIN_TAIL_S
    of music after it."""
    C = np.asarray(F.get("chroma"), dtype=float)
    if C.ndim != 2 or C.shape[0] != 12:
        return None
    nb = C.shape[1]
    edges = np.concatenate([np.asarray(bars, dtype=float), [float(duration)]])
    if nb < 24:
        return None

    def _c(a, b):
        if np.std(a) < 1e-9 or np.std(b) < 1e-9:
            return 0.0
        return float(np.corrcoef(a, b)[0, 1])
    best = None
    for c in range(int(0.4 * nb), nb - 8):
        if float(edges[-1] - edges[c]) < float(STRUCTURE_KEY_CHANGE_MIN_TAIL_S):
            break
        pre, post = C[:, :c].mean(axis=1), C[:, c:].mean(axis=1)
        c0 = _c(pre, post)
        for k in (1, 2, 3, 4, -1, -2):
            ck = _c(np.roll(pre, k), post)
            if best is None or ck - c0 > best[0]:
                best = (ck - c0, c, k, c0, ck)
    if best is None:
        return None
    gain, c, k, c0, ck = best
    if (gain >= float(STRUCTURE_KEY_CHANGE_MIN_GAIN) and ck >= float(STRUCTURE_KEY_CHANGE_MIN_CORR)
            and c0 <= float(STRUCTURE_KEY_CHANGE_MAX_UNSHIFTED)):
        return int(c), int(k)
    return None


def _sx_reassign_by_words(W, body, labels, lyric_ng, sung):
    """dev26 STRUCTURE_REASSIGN_BY_WORDS: a sung segment clustered with one
    material, that SOUNDS like another recurring material just as well
    (median similarity STRUCTURE_REASSIGN_SIM or more to its occurrences) and
    SINGS that material's words (at least STRUCTURE_LYRIC_CHORUS_SHARE of its
    word patterns, median over that material's worded occurrences, two or
    more of them), while sharing few words with its own cluster, is moved to
    that material. Over a one-loop backing a chorus can measure as close to
    the verses as to the other choruses; its words tell which it is. One
    pass over the clustering as it stands, so moves cannot cascade."""
    mn = int(STRUCTURE_LYRIC_MIN_NGRAMS)
    new = dict(labels)
    by_c = {}
    for j in body:
        by_c.setdefault(labels[j], []).append(j)
    for i in body:
        if not sung[i] or i >= len(lyric_ng) or len(lyric_ng[i]) < mn:
            continue
        mine = lyric_ng[i]
        own = [j for j in by_c.get(labels[i], []) if j != i and len(lyric_ng[j]) >= mn]
        own_w = max([len(mine & lyric_ng[j]) / float(len(mine)) for j in own], default=0.0)
        if own_w >= STRUCTURE_REASSIGN_OWN_MAX:
            continue
        best = None
        for c, mem in by_c.items():
            if c == labels[i]:
                continue
            worded = [j for j in mem if j != i and len(lyric_ng[j]) >= mn]
            if len(worded) < 2:
                continue
            ww = float(np.median([len(mine & lyric_ng[j]) / float(len(mine)) for j in worded]))
            sd = float(np.median([W[i, j] for j in mem if j != i]))
            if ww >= STRUCTURE_LYRIC_CHORUS_SHARE and sd >= STRUCTURE_REASSIGN_SIM:
                if best is None or (ww, sd) > best[0]:
                    best = ((ww, sd), c)
        if best is not None:
            new[i] = best[1]
    return new


def _sx_split_by_known_parts(F, bars, duration, starts, lyric_chunks=None, vocal_notes=None):
    """dev26 STRUCTURE_SPLIT_BY_KNOWN_PARTS: section lines the boundary
    search missed because nothing changed in the sound at that point, found
    by repetition instead. A long segment is cut at the bar where
      (A) one side is a near-repeat (STRUCTURE_KNOWN_PART_SIM or more) of
          another segment of about the same length (0.7x to
          STRUCTURE_KNOWN_PART_MAX_RATIO, and no more than 40 s), and the other
          side does not sound like that segment at all
          (STRUCTURE_KNOWN_PART_OTHER_MAX or less) -- a chorus running
          straight into the bridge or outro over the same beat; or
      (B) the two sides are each a near-repeat of two DIFFERENT segments
          (each side clearly closer to its own match than to the other's) --
          a verse and pre-chorus heard separately elsewhere, joined here.
    Each segment is cut at most once, both pieces keep the minimum section
    length, and the measured boundaries are never moved or removed."""
    edges = np.concatenate([np.asarray(bars, dtype=float), [float(duration)]])
    nbars = len(bars)
    cuts = [0] + sorted(int(k) for k in starts) + [nbars]
    segs = [(cuts[k], cuts[k + 1]) for k in range(len(cuts) - 1) if cuts[k + 1] > cuts[k]]
    if len(segs) < 4:
        return starts
    bar_len = float(np.median(np.diff(edges))) if len(edges) > 2 else 2.0
    minb = int(STRUCTURE_MIN_SECTION_BARS) * (2 if bar_len < 1.6 else 1)
    min_s = max(float(STRUCTURE_MIN_SECTION_S), 8.0)
    hi, lo = float(STRUCTURE_KNOWN_PART_SIM), float(STRUCTURE_KNOWN_PART_OTHER_MAX)
    new_cuts = []
    for si, (a, b) in enumerate(segs):
        if b - a < 2 * minb or edges[b] - edges[a] < 2 * min_s:
            continue
        others = [x for k, x in enumerate(segs) if k != si]
        best = None
        for c in range(a + minb, b - minb + 1):
            if edges[c] - edges[a] < min_s or edges[b] - edges[c] < min_s:
                continue
            if STRUCTURE_KNOWN_PART_NOT_HALVES and 2 * c == a + b:
                continue          # dev37: two equal halves -- one part repeating its phrase
            P = [(a, c), (c, b)] + others
            Wm = _sx_segment_similarity(F, P)
            pl = [c - a, b - c]
            # (A) one side repeats a segment the other side does not resemble
            for pi, qi in ((0, 1), (1, 0)):
                for xi in range(2, len(P)):
                    xl = P[xi][1] - P[xi][0]
                    if not (0.7 <= xl / float(pl[pi]) <= float(STRUCTURE_KNOWN_PART_MAX_RATIO)):
                        continue
                    if edges[P[xi][1]] - edges[P[xi][0]] > 40.0:
                        continue          # a 40 s+ match is likely two parts itself
                    m, o = float(Wm[pi, xi]), float(Wm[qi, xi])
                    if m >= hi and o <= lo:
                        sc = m - o
                        if best is None or sc > best[0]:
                            best = (sc, c, P[qi], P[xi], qi)
            # (B) each side repeats its own, different segment
            xl_ = [xi for xi in range(2, len(P)) if 0.7 <= (P[xi][1] - P[xi][0]) / float(pl[0]) <= 1.45]
            xr_ = [xi for xi in range(2, len(P)) if 0.7 <= (P[xi][1] - P[xi][0]) / float(pl[1]) <= 1.45]
            if xl_ and xr_:
                xl = max(xl_, key=lambda x: Wm[0, x])
                xr = max(xr_, key=lambda x: Wm[1, x])
                if (xl != xr and Wm[0, xl] >= hi and Wm[1, xr] >= hi
                        and Wm[0, xl] - Wm[0, xr] >= 0.15 and Wm[1, xr] - Wm[1, xl] >= 0.15):
                    sc = float(min(Wm[0, xl] - Wm[0, xr], Wm[1, xr] - Wm[1, xl]))
                    if best is None or sc > best[0]:
                        best = (sc, c, None, None, None)
        if best is not None and best[2] is not None and STRUCTURE_KNOWN_PART_WORDS_VETO and lyric_chunks:
            # dev30: see STRUCTURE_KNOWN_PART_WORDS_VETO.
            try:
                _left, _x, _qi = best[2], best[3], best[4]
                _xi = segs.index(_x)
                _ni = _xi - 1 if _qi == 0 else _xi + 1
                if 0 <= _ni < len(segs):
                    _ng = _sx_lyric_ngrams([_left, segs[_ni]], edges, lyric_chunks, vocal_notes)
                    _sh = len(_ng[0] & _ng[1])
                    _dl = float(edges[_left[1]] - edges[_left[0]])
                    _dn = float(edges[segs[_ni][1]] - edges[segs[_ni][0]])
                    if (len(_ng[0]) >= 2 and _sh >= 0.5 * len(_ng[0]) and _sh < 0.3 * len(_ng[1])
                            and _dn >= 2.0 * _dl):
                        best = None
            except Exception:
                pass
        if best is not None:
            new_cuts.append(best[1])
    if not new_cuts:
        return starts
    return sorted(set(int(k) for k in starts) | set(new_cuts))


def _sx_eject_outliers(W, body, labels, lyric_ng, sung):
    """dev26 STRUCTURE_EJECT_OUTLIER: one sung segment grouped with a part
    of four or more segments, heard between that part's other occurrences,
    that is clearly less alike than they are with each other (its best match
    in the group STRUCTURE_OUTLIER_MARGIN or more below their median, and
    under STRUCTURE_OUTLIER_MAX_SIM) and repeats few of their words, becomes
    material of its own -- a bridge or breakdown over the chorus's chords
    (average linkage had pulled it in). The first and last occurrence are
    left alone (a stripped-down first or last chorus), as is anything whose
    words come back from the group."""
    mn = int(STRUCTURE_LYRIC_MIN_NGRAMS)
    groups = {}
    for i in body:
        groups.setdefault(labels[i], []).append(i)
    new = dict(labels)
    nxt = (max(labels.values()) + 1) if labels else 0
    for _l, mem in sorted(groups.items()):
        mem = sorted(mem)
        if len(mem) < 4:
            continue
        for i in mem[1:-1]:
            if not sung[i]:
                continue
            rest = [j for j in mem if j != i]
            inner = [W[a, b] for a in rest for b in rest if a < b]
            if not inner:
                continue
            med = float(np.median(inner))
            best = max(float(W[i, j]) for j in rest)
            if best >= STRUCTURE_OUTLIER_MAX_SIM or best > med - STRUCTURE_OUTLIER_MARGIN:
                continue
            if lyric_ng and i < len(lyric_ng) and len(lyric_ng[i]) >= mn:
                gw = set()
                for j in rest:
                    if abs(j - i) > 1:
                        gw |= lyric_ng[j]
                if len(lyric_ng[i] & gw) >= 0.5 * len(lyric_ng[i]):
                    continue
            new[i] = nxt
            nxt += 1
    return new


def _sx_matches_chorus_occurrence(words, occurrences, distinct_verse=False):
    """Prefer a single chorus match; reject pooled-only matches only when
    independent acoustic evidence supports a distinct verse. Missing or
    inconsistent lyrics alone must not demote a genuine chorus.
    """
    if len(words) < STRUCTURE_LYRIC_MIN_NGRAMS:
        return False
    if any(len(words & other) >= STRUCTURE_LYRIC_CHORUS_SHARE * len(words)
           for other in occurrences if len(other) >= STRUCTURE_LYRIC_MIN_NGRAMS):
        return True
    # Inconsistent Whisper renditions are not negative evidence by themselves.
    # Retain the pre-v6 pooled fallback unless the audio independently supports verse.
    pooled = set().union(*occurrences) if occurrences else set()
    return not distinct_verse and len(words & pooled) >= STRUCTURE_LYRIC_CHORUS_SHARE * len(words)


def _sx_name_sections(F, bars, duration, segs, W, lyric_sig=None, lyric_rep=None,
                      vocal_known=True, instrumental=False, lyric_ng=None, lyric_smeared=False,
                      info_out=None, vocable_only=None, sheet_intro=False, lone_sung_verse=True,
                      title_hits=None):
    """Assign musical-function names to segments.

    Returns (names, labels, coverage): a base name per segment, the cluster
    label per body segment, and per-segment vocal coverage (or None).

    Repetition is counted in RUNS: consecutive segments of the same material
    are one occurrence (a section the boundary search happened to split), so
    only material that comes back after something else counts as recurring.
    Nothing here assumes verse-before-chorus: the chorus is the recurring
    sung material with the strongest hook evidence, wherever it first occurs.
    """
    n = len(segs)
    bar_edges = np.concatenate([np.asarray(bars, dtype=float), [duration]])
    lens = [float(bar_edges[b] - bar_edges[a]) for a, b in segs]
    loud = np.array([float(np.mean(F["loud"][a:b])) for a, b in segs])
    has_v = bool(F.get("has_vocal")) and vocal_known and not instrumental
    if has_v:
        cov = np.array([float(np.mean(F["vocal"][a:b])) for a, b in segs])
        sung = cov >= STRUCTURE_SUNG_MIN_COVERAGE
        vp_bars = np.asarray(F["vpitch"], dtype=float)
    else:
        cov = None
        sung = np.zeros(n, dtype=bool) if instrumental else np.ones(n, dtype=bool)
        vp_bars = None
    names = [None] * n

    # ---- Bookends ------------------------------------------------------
    if has_v:
        sung_idx = [i for i in range(n) if sung[i]]
        if sung_idx and STRUCTURE_SUNG_INTRO and len(sung_idx) >= 4:
            # dev55: see STRUCTURE_SUNG_INTRO.
            _sl = [loud[i] for i in sung_idx]
            _ref_loud = float(np.median(_sl))
            _dr = None
            if F.get("has_stems"):
                _dv = [float(np.mean(F["drums"][segs[i][0]:segs[i][1]])) for i in sung_idx]
                _dr = float(np.median(_dv))
            k = 0
            while k < n - 3:
                if not sung[k]:
                    k += 1
                    continue
                short_stub = (lens[k] < STRUCTURE_SHORT_LEAD_S and k + 1 < n and not sung[k + 1])
                quiet_open = (lens[k] <= STRUCTURE_SUNG_INTRO_MAX_S
                              and loud[k] <= _ref_loud - STRUCTURE_SUNG_INTRO_QUIETER_DB
                              and (_dr is None or float(np.mean(F["drums"][segs[k][0]:segs[k][1]]))
                                   <= 0.5 * _dr))
                if (quiet_open and STRUCTURE_SUNG_INTRO_NEEDS_NO_WORDS and lyric_ng
                        and k < len(lyric_ng) and len(lyric_ng[k]) >= STRUCTURE_LYRIC_MIN_NGRAMS):
                    # dev26: it sings real lines -- a soft first verse (or a
                    # chorus sung softly first), not an intro.
                    quiet_open = False
                    if STRUCTURE_SUNG_INTRO_TAG_OR_SPEECH:
                        # dev30: see the setting -- still the intro when it
                        # repeats one short line, is a quiet spoken prelude
                        # before the band, or the lyric sheet opens with Intro
                        _wd = set(w_ for p_ in lyric_ng[k] for w_ in p_ if w_ != "_")
                        if (len(_wd) < int(STRUCTURE_SUNG_INTRO_MIN_WORDS)
                                or (loud[k] <= _ref_loud - float(STRUCTURE_SPOKEN_PRELUDE_DB)
                                    and k + 1 < n and not sung[k + 1])
                                or (sheet_intro and k == sung_idx[0])):
                            quiet_open = True
                # dev56: see STRUCTURE_VOCABLE_INTRO.
                vocable_open = bool(STRUCTURE_VOCABLE_INTRO and vocable_only is not None
                                    and k < len(vocable_only) and vocable_only[k]
                                    and lens[k] <= STRUCTURE_SUNG_INTRO_MAX_S
                                    and loud[k] <= _ref_loud - STRUCTURE_VOCABLE_INTRO_QUIETER_DB)
                if not (short_stub or quiet_open or vocable_open):
                    break
                sung[k] = False
                k += 1
            sung_idx = [i for i in range(n) if sung[i]]
        if sung_idx:
            # 0.7.1-pre12 STRUCTURE_BOOKEND_MAX (F-11): only the unsung parts near the start
            # (end) of the music are the intro (outro); a long unsung opening or ending is
            # named like the rest of the song.
            _bk0, _bk1 = float(bar_edges[segs[0][0]]), float(bar_edges[segs[-1][1]])
            for i in range(0, sung_idx[0]):
                if (not STRUCTURE_BOOKEND_MAX
                        or float(bar_edges[segs[i][0]]) - _bk0 < float(STRUCTURE_BOOKEND_SPAN_S)):
                    names[i] = "Intro"
            for i in range(sung_idx[-1] + 1, n):
                if (not STRUCTURE_BOOKEND_MAX
                        or _bk1 - float(bar_edges[segs[i][1]]) < float(STRUCTURE_BOOKEND_SPAN_S)):
                    names[i] = "Outro"
    else:
        med_loud = float(np.median(loud)) if n else 0.0
        if n >= 3 and lens[0] <= STRUCTURE_INTRO_MAX_S and loud[0] <= med_loud - 1.5:
            names[0] = "Intro"
        if n >= 3 and lens[-1] <= STRUCTURE_OUTRO_MAX_S and loud[-1] <= med_loud - 1.5:
            names[-1] = "Outro"

    body = [i for i in range(n) if names[i] is None]
    labels = _sx_cluster(W, body) if body else {}
    _new_words_labels = set()
    if (STRUCTURE_SPLIT_BY_COHESION and lyric_ng and has_v and not instrumental and labels
            and cov is not None):
        labels, _new_words_labels = _sx_split_by_cohesion(W, body, labels, lyric_ng, cov)
    if STRUCTURE_LONE_SUNG_REGROUP and has_v and not instrumental and labels and cov is not None:
        labels = _sx_regroup_lone_sung(W, labels, cov, lyric_ng)
    if STRUCTURE_SPLIT_SHARED_MUSIC and lyric_ng and has_v and not instrumental and labels:
        labels, _nw45 = _sx_split_shared_music_by_words(labels, body, lyric_ng, sung)
        _new_words_labels = set(_new_words_labels) | set(_nw45)

    if STRUCTURE_EJECT_OUTLIER and has_v and not instrumental and labels:
        try:
            labels = _sx_eject_outliers(W, body, labels, lyric_ng, sung)
        except Exception:
            pass
    if STRUCTURE_EJECT_UNSHARED_WORDS and lyric_ng and has_v and not instrumental and labels:
        try:
            labels = _sx_eject_unshared_words(body, labels, lyric_ng, sung)
        except Exception:
            pass
    if STRUCTURE_REASSIGN_BY_WORDS and lyric_ng and has_v and not instrumental and labels:
        try:
            labels = _sx_reassign_by_words(W, body, labels, lyric_ng, sung)
        except Exception:
            pass

    # ---- Runs of identical material --------------------------------------
    units = []   # each: dict(segs=[...], cl=label)
    for i in body:
        if (units and units[-1]["segs"][-1] == i - 1 and units[-1]["cl"] == labels[i]
                and not (STRUCTURE_RUNS_SPLIT_BY_VOICE and has_v
                         and bool(sung[i]) != bool(sung[i - 1]))):
            units[-1]["segs"].append(i)
        else:
            units.append({"segs": [i], "cl": labels[i]})
    for u in units:
        idx = u["segs"]
        a0, b1 = segs[idx[0]][0], segs[idx[-1]][1]
        u["len"] = sum(lens[i] for i in idx)
        u["loud"] = float(np.mean(F["loud"][a0:b1]))
        if vp_bars is not None:
            v = vp_bars[a0:b1]
            v = v[~np.isnan(v)]
            u["vp"] = float(np.median(v)) if v.size else np.nan
        else:
            u["vp"] = np.nan
        u["sung"] = float(np.mean([sung[i] for i in idx])) >= 0.5
        if F.get("has_stems"):
            u["dens"] = float(np.mean(F["drums"][a0:b1]))
        else:
            u["dens"] = np.nan
        reps = [lyric_rep[i] for i in idx if lyric_rep and lyric_rep[i] is not None]
        u["lyr"] = float(np.mean(reps)) if reps else None
    by_cl = {}
    for ui, u in enumerate(units):
        by_cl.setdefault(u["cl"], []).append(ui)
    repeated = {c: us for c, us in by_cl.items() if len(us) >= 2}

    def _z(v, arr):
        arr = np.asarray([a for a in arr if a is not None and not np.isnan(a)], dtype=float)
        if arr.size < 2 or v is None or np.isnan(v):
            return 0.0
        return float((v - arr.mean()) / (arr.std() + 1e-6))

    def _wmean(c, key):
        us = repeated[c]
        vals = [(units[k][key], units[k]["len"]) for k in us
                if units[k][key] is not None and not np.isnan(units[k][key])]
        if not vals:
            return np.nan
        return float(sum(v * w for v, w in vals) / sum(w for _v, w in vals))

    all_loud = [u["loud"] for u in units]
    all_vp = [u["vp"] for u in units]
    all_dens = [u["dens"] for u in units]

    def _hook_score(c, detail=None):
        parts = {
            "loud": STRUCTURE_HOOK_W_LOUD * _z(_wmean(c, "loud"), all_loud),
            "register": STRUCTURE_HOOK_W_REGISTER * _z(_wmean(c, "vp"), all_vp),
            "drums": STRUCTURE_HOOK_W_DRUMS * _z(_wmean(c, "dens"), all_dens),
            "repeats": STRUCTURE_HOOK_W_REPEATS * (len(repeated[c]) - 2),
        }
        lyr = [units[k]["lyr"] for k in repeated[c] if units[k]["lyr"] is not None]
        if lyr and not _hook_ctx["drop_lyrics"]:
            # Repeated words are the defining property of a chorus; when the
            # lyrics were transcribed they outweigh level and register.
            parts["lyrics"] = STRUCTURE_HOOK_W_LYRICS * (float(np.mean(lyr)) - 0.4)
        if detail is not None:
            detail[c] = parts
        return float(sum(parts.values()))

    # dev50: set below when the words decided nothing and their timing is
    # smeared across whole sections (see STRUCTURE_SMEARED_LYRICS_HOOK).
    _hook_ctx = {"drop_lyrics": False}

    sung_rep = [] if instrumental else [
        c for c, us in repeated.items() if np.mean([units[k]["sung"] for k in us]) >= 0.5
    ]

    # How much each recurring part keeps its WORDS from one occurrence to the
    # next (each occurrence against its best-matching other occurrence). A chorus comes back with the
    # same words; a verse comes back with the same music and new words. That
    # is the defining difference between them, so when the lyrics show it
    # clearly it decides -- loudness, register and drums are only clues, and
    # on a song whose chorus is quieter, lower or sparser than its verses
    # (or that opens with a stripped-down chorus) they point the wrong way.
    same_words = {}
    same_words_n = {}   # dev26: fewest word patterns of the occurrences compared
    if lyric_ng and not instrumental:
        for c in sung_rep:
            occ_sets = []
            for k in repeated[c]:
                ss = set()
                for i in units[k]["segs"]:
                    ss |= lyric_ng[i]
                if len(ss) >= STRUCTURE_LYRIC_MIN_NGRAMS:
                    occ_sets.append(ss)
            # Each occurrence's best match with any other occurrence, then
            # the median: one badly transcribed occurrence (misheard, or its
            # words smeared into a neighbour) cannot hide a real chorus.
            best = []
            for x in range(len(occ_sets)):
                others = [_sx_lyric_overlap(occ_sets[x], occ_sets[y_])[0]
                          for y_ in range(len(occ_sets)) if y_ != x]
                if others:
                    best.append(max(others))
            if best:
                same_words[c] = float(np.median(best))
                same_words_n[c] = min(len(x) for x in occ_sets)

    chorus = verse = None
    _wordless_demoted = None  # dev74: see STRUCTURE_WORDS_BEAT_UNWORDED
    _verse_only_last = None   # dev70: last verse unit when the verse is the only recurring sung part
    _tags = set()        # dev46: short word-repeating tags (post-chorus hook lines)
    lyric_hi = []
    lo_vals = [v for c, v in same_words.items()
               if v <= STRUCTURE_LYRIC_SAME_LO or c in _new_words_labels]
    if lo_vals:
        hi_thr = max(STRUCTURE_LYRIC_SAME_HI_MIN, max(lo_vals) + STRUCTURE_LYRIC_SAME_GAP)
        lyric_hi = sorted((c for c in sung_rep if same_words.get(c, -1.0) >= hi_thr),
                          key=lambda c: same_words[c], reverse=True)
    if lyric_hi:
        # The part whose words come back most is the chorus; only when two
        # parts repeat their words about equally (a chorus and a pre-chorus)
        # does the level / register / drums hook score choose between them.
        # A chorus comes round more often than a bridge that repeats its own
        # words: a part heard only in the second half of the song, two or
        # more times fewer than another such part, is not the chorus. (A
        # twice-sung bridge whose two passes were transcribed word-perfect
        # otherwise beat a five-times chorus whose lines Whisper merged and
        # dropped differently each time.) A part that is also heard early --
        # a chorus in a short song next to a hook heard more often -- stays.
        most = max(len(repeated[c]) for c in lyric_hi)

        def _late_and_rarer(c):
            if len(repeated[c]) > most - 2:
                return False
            first = min(float(bar_edges[segs[units[k]["segs"][0]][0]]) for k in repeated[c])
            return first >= STRUCTURE_BRIDGE_MIN_POS * float(duration)

        cands = [c for c in lyric_hi if not (STRUCTURE_LYRIC_CHORUS_CHECKS and _late_and_rarer(c))] or lyric_hi
        if STRUCTURE_PRECHORUS_LEADS_IN and len(cands) >= 2:
            cands = _sx_drop_lead_ins(cands, repeated, units)
        top = [c for c in cands if same_words[c] >= max(same_words[x] for x in cands) - 0.15]
        chorus = max(top, key=_hook_score)
        _tpool = cands if STRUCTURE_TITLE_PICKS_ANY_WORDED else top
        if STRUCTURE_TITLE_PICKS_CHORUS and title_hits and len(_tpool) >= 2:
            # dev32: see STRUCTURE_TITLE_PICKS_CHORUS.
            try:
                def _t_occ(c):
                    return [sum(title_hits[i] for i in units[k]["segs"]) for k in repeated[c]]
                _wt = [c for c in _tpool if all(x > 0 for x in _t_occ(c))]
                _nt = [c for c in _tpool if not any(x > 0 for x in _t_occ(c))]
                def _beside(c, t):
                    return all((k > 0 and units[k - 1]["cl"] == t)
                               or (k + 1 < len(units) and units[k + 1]["cl"] == t)
                               for k in repeated[c])
                # Not when the other part sits next to the title part every
                # time: two halves of one chorus, or a lead-in (left to the
                # checks above and below).
                if (len(_wt) == 1 and len(_nt) == len(_tpool) - 1
                        and not any(_beside(c, _wt[0]) for c in _nt)):
                    chorus = _wt[0]
            except Exception:
                pass
        if STRUCTURE_LEADIN_UNWORDED and STRUCTURE_PRECHORUS_LEADS_IN:
            # dev51: the part it leads into had no measurable words.
            _unworded = [c for c in sung_rep if c != chorus and c not in lyric_hi
                         and c not in same_words and c not in _new_words_labels]
            _led = _sx_drop_lead_ins([chorus] + _unworded, repeated, units)
            if chorus not in _led:
                chorus = max(_led, key=lambda c: (len(repeated[c]), sum(units[k]["len"] for k in repeated[c])))
        rest = [c for c in sung_rep if c != chorus and c not in lyric_hi]
        verse = max(rest, key=lambda c: sum(units[k]["len"] for k in repeated[c])) if rest else None
        if STRUCTURE_CHORUS_HALVES and STRUCTURE_CHORUS_HALVES_WORDED and lyric_ng:
            # dev30: the same for a song whose verses were compared by words.
            for _c2 in [c for c in sung_rep if c not in (chorus, verse) and c in repeated]:
                try:
                    _half = _sx_chorus_half(chorus, _c2, units, repeated, same_words, _wmean, lyric_ng)
                except Exception:
                    _half = False
                if _half is not False:
                    for k in repeated[_c2]:
                        units[k]["cl"] = chorus
                        for i in units[k]["segs"]:
                            labels[i] = chorus
                    repeated[chorus] = sorted(repeated[chorus] + repeated[_c2])
                    del repeated[_c2]
                    sung_rep = [c for c in sung_rep if c != _c2]
                    lyric_hi = [c for c in lyric_hi if c != _c2]
    elif len(sung_rep) >= 2:
        _no_chorus_by_words = False
        if STRUCTURE_SMEARED_LYRICS_HOOK and lyric_smeared:
            _hook_ctx["drop_lyrics"] = True
        ranked = sorted(sung_rep, key=_hook_score, reverse=True)
        if STRUCTURE_VERSE_BY_WORDS:
            # dev55: a recurring part measured to sing new words every time
            # is a verse, so it cannot be the chorus while another recurring
            # sung part is available (a jangle-pop test track: 6% of the 0:17 / 1:41
            # part's word patterns come back, yet it was louder-sounding by
            # register and was named Chorus).
            _vl = [c for c in ranked if c in same_words and same_words[c] <= STRUCTURE_LYRIC_SAME_LO
                   and not (STRUCTURE_VERSE_BY_WORDS_MIN_NGRAMS
                            and same_words_n.get(c, 0) < STRUCTURE_VERSE_BY_WORDS_MIN_NGRAMS)]
            _ok = [c for c in ranked if c not in _vl]
            if _vl and _ok:
                ranked = _ok + [c for c in ranked if c in _vl]
            elif _vl and not _ok and STRUCTURE_NO_CHORUS_WITHOUT_REPEATED_WORDS and not lyric_smeared:
                _no_chorus_by_words = True      # 0.7.1-pre12 (F-25): every recurring part sings new words
        if STRUCTURE_WORDS_BEAT_UNWORDED and same_words and ranked[0] not in same_words \
                and all(units[k]["lyr"] is None for k in repeated[ranked[0]]):
            # dev74: the hook score's front-runner has no measurable words, while
            # another recurring sung part brings its words back almost whole
            # (a dance-rock test track: the chorus at 0:00 and 2:25 repeats 92%
            # of its words, yet a 9-second title-line tail with no transcribed
            # words won on register and was named Chorus, the real chorus
            # Verse 1).
            _wc = [c for c in ranked if same_words.get(c, -1.0) >= float(STRUCTURE_WORDS_BEAT_UNWORDED_MIN)]
            if _wc:
                _wordless_demoted = ranked[0]
                ranked = [_wc[0]] + [c for c in ranked if c != _wc[0]]
        chorus = ranked[0]
        verse = max(ranked[1:], key=lambda c: sum(units[k]["len"] for k in repeated[c]))
        if _no_chorus_by_words:
            chorus = None
            verse = max(ranked, key=lambda c: sum(units[k]["len"] for k in repeated[c]))
            _verse_only_last = max(repeated[verse])
        if STRUCTURE_CHORUS_HALVES and lyric_ng and chorus is not None:
            # dev30: see STRUCTURE_CHORUS_HALVES.
            try:
                _half = _sx_chorus_half(chorus, verse, units, repeated, same_words, _wmean, lyric_ng)
            except Exception:
                _half = False
            if _half is not False:
                for k in repeated[verse]:
                    units[k]["cl"] = chorus
                    for i in units[k]["segs"]:
                        labels[i] = chorus
                repeated[chorus] = sorted(repeated[chorus] + repeated[verse])
                del repeated[verse]
                sung_rep = [c for c in sung_rep if c != verse]
                _rest = [c for c in ranked[1:] if c in repeated and c != chorus]
                verse = (max(_rest, key=lambda c: sum(units[k]["len"] for k in repeated[c]))
                         if _rest else None)
    elif len(sung_rep) == 1:
        c = sung_rep[0]
        if STRUCTURE_ONLY_RECURRING_IS_CHORUS and STRUCTURE_SMEARED_LYRICS_HOOK and lyric_smeared:
            _hook_ctx["drop_lyrics"] = True     # dev55: as for two or more parts
        lyr = [units[k]["lyr"] for k in repeated[c] if units[k]["lyr"] is not None]
        others_sung = [u for u in units if u["cl"] != c and u["sung"]]
        if (STRUCTURE_ONLY_RECURRING_NEW_WORDS_IS_VERSE and c in same_words
                and same_words[c] < float(STRUCTURE_ONLY_RECURRING_VERSE_MAX_SAME)):
            # dev70: the only recurring sung part sings mostly NEW words each
            # time it comes round, so it is the verse, even if some of its
            # lines (a refrain) repeat. ("a rock test track": 30-38% of the words
            # come back, the rest are new; a chorus comes back with most of
            # its words.)
            verse = c
            _verse_only_last = max(repeated[c])
        elif lyr and float(np.mean(lyr)) >= 0.5:
            chorus = c
        elif others_sung and _hook_score(c) > 0.6:
            # Recurring, louder/higher than the one-off sung parts around it.
            chorus = c
        elif (STRUCTURE_ONLY_RECURRING_IS_CHORUS and others_sung and len(repeated[c]) >= 3
              and _hook_score(c) > 0.0):
            # dev55: the only sung material that comes back, three or more
            # times and at least as loud/high as the rest, is the chorus (the
            # verses were each measured as their own material).
            chorus = c
        else:
            verse = c

    if ((STRUCTURE_CHORUS_PREFERS_LONGER or STRUCTURE_CHORUS_WORDS_FALLBACK)
            and lyric_ng and chorus is not None
            and not instrumental and repeated):
        chorus, verse, _tags = _sx_chorus_over_tags(
            chorus, verse, units, repeated, sung_rep, lyric_ng,
            [c for c in sung_rep if STRUCTURE_LYRIC_CHORUS_CHECKS and lyric_hi
             and c in lyric_hi and _late_and_rarer(c)],
            by_hook=not lyric_hi)
    def _distinct_verse_audio(ui):
        if verse is None or chorus is None:
            return False
        vs = [k for k in repeated.get(verse, []) if k != ui]
        cs = list(repeated.get(chorus, []))
        if len(vs) < 2 or len(cs) < 2:
            return False
        def similarity(k):
            return float(np.mean([W[i, j] for i in units[ui]["segs"] for j in units[k]["segs"]]))
        v = [similarity(k) for k in vs]
        c = [similarity(k) for k in cs]
        return (all(np.isfinite(x) for x in v + c)
                and min(v) >= STRUCTURE_SAME_MATERIAL_SIM
                and min(v) >= max(c) + 0.1)

    unit_names = [None] * len(units)
    if not repeated and not instrumental:
        # Nothing recurs: there is no measured basis for verse/chorus names.
        # Through-composed songs, ambient pieces and single-texture tracks
        # get neutral names rather than a guessed song form.
        for ui, u in enumerate(units):
            unit_names[ui] = "Instrumental" if (has_v and not u["sung"]) else "Section"
        for ui, u in enumerate(units):
            for i in u["segs"]:
                names[i] = unit_names[ui]
        if STRUCTURE_LONE_SUNG_IS_VERSE and lone_sung_verse and has_v and lyric_ng:
            # dev30: see STRUCTURE_LONE_SUNG_IS_VERSE.
            def _worded(i):
                return i < len(lyric_ng) and len(lyric_ng[i]) >= STRUCTURE_LYRIC_MIN_NGRAMS
            _su = [ui for ui, u in enumerate(units) if u["sung"] and any(_worded(i) for i in u["segs"])]
            if 1 <= len(_su) <= int(STRUCTURE_LONE_SUNG_MAX_PARTS):
                for ui in _su:
                    for i in units[ui]["segs"]:
                        # a stretch of it with no words and little voice (stem
                        # bleed after the singing ends) is instrumental
                        names[i] = ("Verse" if (_worded(i) or cov[i] >= 2 * STRUCTURE_SUNG_MIN_COVERAGE + 0.1)
                                    else "Instrumental")
        return names, labels, cov
    # A part with the verse's or chorus's music but (almost) no voice -- an
    # instrumental pass over the chorus chords, a solo over the verse -- is
    # Instrumental, not another verse or chorus. Only when it has well under
    # the sung level AND a small fraction of what its sung siblings carry, so
    # a quiet or thinly transcribed vocal occurrence keeps its name.
    u_cov = []
    for u in units:
        u_cov.append(float(np.mean([cov[i] for i in u["segs"]])) if cov is not None else None)

    def _voiceless(ui):
        if not STRUCTURE_SPLIT_INSTRUMENTAL:
            return False
        if u_cov[ui] is None or u_cov[ui] >= STRUCTURE_SUNG_MIN_COVERAGE:
            return False
        sib = [u_cov[k] for k in range(len(units)) if k != ui and units[k]["cl"] == units[ui]["cl"]
               and u_cov[k] is not None]
        return bool(sib) and u_cov[ui] < 0.25 * float(np.median(sib))

    for ui, u in enumerate(units):
        if u["cl"] in (chorus, verse) and u["cl"] is not None and _voiceless(ui):
            unit_names[ui] = "Instrumental"
        elif u["cl"] == chorus:
            unit_names[ui] = "Chorus"
        elif u["cl"] == verse:
            unit_names[ui] = "Verse"
    if (lyric_hi and chorus is not None and verse is not None and lyric_ng
            and STRUCTURE_LYRIC_PHRASES and STRUCTURE_LYRIC_CHORUS_CHECKS):
        # An occurrence that sounds like the verse but SINGS THE CHORUS'S WORDS
        # is a chorus: a stripped-down opening chorus (no drums, no bass) is
        # acoustically closer to the verses and was clustered with them, so a
        # chorus-first song came out "Verse 1 / Verse 2 / Chorus 1". Only when
        # the lyrics already decided which part is the chorus, and at least
        # half of its word patterns are the chorus's -- a verse that runs into
        # its chorus picks up a few of them through Whisper's merged lines
        # (under a quarter in every test), not half.
        def _unit_ng(k):
            ss = set()
            for i in units[k]["segs"]:
                ss |= lyric_ng[i]
            return ss
        ch_occurrences = [_unit_ng(k) for k in repeated.get(chorus, [])
                          if unit_names[k] == "Chorus"]
        if any(ch_occurrences):
            _ch0 = [k for k in range(len(units)) if unit_names[k] == "Chorus"]
            # (a short sung tail after it -- the last line, a fade -- does not count)
            _last_sung = max([k for k in range(len(units)) if units[k]["sung"]
                              and units[k]["len"] > 1.5 * STRUCTURE_SHORT_LEAD_S], default=-1)
            for ui, u in enumerate(units):
                if u["cl"] != verse or unit_names[ui] != "Verse":
                    continue
                mine = _unit_ng(ui)
                if _sx_matches_chorus_occurrence(mine, ch_occurrences, _distinct_verse_audio(ui)):
                    if (STRUCTURE_CHORUS_WORDS_TAIL_IS_OUTRO and _ch0 and ui > max(_ch0)
                            and ui == _last_sung):
                        # dev26: the verse's music under the chorus's hook,
                        # after the last chorus, ending the singing: the outro.
                        unit_names[ui] = "Outro"
                    else:
                        unit_names[ui] = "Chorus"
    ch_units = [ui for ui, u in enumerate(units)
                if u["cl"] == chorus and (unit_names[ui] == "Chorus" or not STRUCTURE_SPLIT_INSTRUMENTAL)
                ] if chorus is not None else []
    first_ch = ch_units[0] if ch_units else None
    last_ch = ch_units[-1] if ch_units else None
    theme_names = {}

    for ui, u in enumerate(units):
        if unit_names[ui] is not None:
            continue
        c = u["cl"]
        if instrumental:
            if c in repeated:
                if c not in theme_names:
                    theme_names[c] = "Theme %s" % chr(65 + len(theme_names))
                unit_names[ui] = theme_names[c]
            else:
                unit_names[ui] = "Section"
            continue
        if not u["sung"]:
            unit_names[ui] = "Instrumental"
            continue
        if c in _tags and chorus is not None:
            if ui > 0 and units[ui - 1]["cl"] == chorus:
                unit_names[ui] = "Post-Chorus"
                continue
            if last_ch is not None and ui > last_ch:
                unit_names[ui] = "Outro"
                continue
        if (c is not None and c == _wordless_demoted and chorus is not None and ui > 0
                and units[ui - 1]["cl"] == chorus and u["len"] <= 20.0):
            # dev74: the short tail that lost the chorus name to the words,
            # straight after a chorus, stays its own part (no boundary lost).
            unit_names[ui] = "Post-Chorus"
            continue
        _sung_twice = True
        if STRUCTURE_REPEAT_NEEDS_SUNG_TWICE and c in repeated and c not in (chorus, verse):
            # dev26: material that comes back only as an instrumental pass
            # is not a recurring SUNG part -- its one sung occurrence is named
            # like any other one-off sung part (see the setting).
            _sung_twice = (sum(1 for k in repeated[c] if units[k]["sung"]) >= 2
                           or not (first_ch is not None and first_ch < ui < last_ch))
        if c in repeated and chorus is not None and _sung_twice:
            occ = repeated[c]
            precedes = all(k + 1 < len(units) and units[k + 1]["cl"] == chorus for k in occ)
            follows = all(k - 1 >= 0 and units[k - 1]["cl"] == chorus for k in occ)
            short = all(units[k]["len"] <= STRUCTURE_PRECHORUS_UNIT_MAX_S for k in occ)
            after_sung = all(k - 1 >= 0 and units[k - 1]["sung"] and units[k - 1]["cl"] != chorus for k in occ)
            # A pre-chorus must be genuinely different material from the
            # verse it follows, not just the verse's second half.
            distinct = True
            if verse is not None:
                vs_ = [i for k in repeated.get(verse, []) for i in units[k]["segs"]]
                me_ = [i for k in occ for i in units[k]["segs"]]
                if vs_ and me_:
                    distinct = float(np.mean([W[a_, b_] for a_ in me_ for b_ in vs_])) <= STRUCTURE_PRECHORUS_DISTINCT_MAX
            _post_ok = short
            if (not short and follows and STRUCTURE_POSTCHORUS_CHORUS_LENGTH
                    and same_words.get(c, 0.0) >= STRUCTURE_LYRIC_SAME_HI_MIN):
                # dev26: a word-repeating part after every chorus may run a
                # little past 20 s when the choruses do too (see the setting).
                _chl = [units[k]["len"] for k in range(len(units)) if units[k]["cl"] == chorus]
                if _chl:
                    _lim = max(20.0, 1.1 * float(np.median(_chl)))
                    _post_ok = all(units[k]["len"] <= _lim and units[k]["len"] <= units[k - 1]["len"] + 1e-6
                                   for k in occ)
            if (STRUCTURE_PRECHORUS_MOST_OCCURRENCES and not (precedes and after_sung) and len(occ) >= 3
                    and sum(1 for k in occ if k + 1 < len(units) and units[k + 1]["cl"] == chorus
                            and k - 1 >= 0 and units[k - 1]["sung"] and units[k - 1]["cl"] != chorus)
                    >= len(occ) - 1):
                precedes = after_sung = True    # 0.7.1-pre12 (F-31): one occurrence leading elsewhere
            if precedes and short and after_sung and distinct:
                unit_names[ui] = "Pre-Chorus"
            elif follows and _post_ok:
                unit_names[ui] = "Post-Chorus"
            elif (STRUCTURE_REPEATED_BRIDGE and follows and distinct and verse is not None
                  and same_words.get(c, 0.0) >= STRUCTURE_LYRIC_SAME_HI_MIN):
                # dev30: see STRUCTURE_REPEATED_BRIDGE.
                unit_names[ui] = "Bridge"
            elif (STRUCTURE_PRECHORUS_AFTER_VERSE and c in lyric_hi and verse is not None
                  and distinct and all(k - 1 >= 0 and units[k - 1]["cl"] == verse for k in occ)
                  and not any(k - 1 >= 0 and units[k - 1]["cl"] == chorus for k in occ)
                  and occ[0] + 1 < len(units) and units[occ[0] + 1]["cl"] == chorus
                  and all(units[k]["len"] <= STRUCTURE_PRECHORUS_AFTER_VERSE_MAX_S for k in occ)):
                # dev32: see STRUCTURE_PRECHORUS_AFTER_VERSE.
                unit_names[ui] = "Pre-Chorus"
            else:
                unit_names[ui] = "Verse"
            continue
        if c in repeated and _sung_twice:
            unit_names[ui] = "Verse"
            continue
        # dev50: a one-off part before the last chorus that sings mostly the
        # chorus's words is a chorus (see STRUCTURE_ONEOFF_CHORUS_WORDS).
        if (STRUCTURE_ONEOFF_CHORUS_WORDS and lyric_hi and chorus is not None and lyric_ng
                and STRUCTURE_LYRIC_PHRASES and STRUCTURE_LYRIC_CHORUS_CHECKS
                and last_ch is not None and ui < last_ch):
            _chorus_occurrences = []
            for k in repeated.get(chorus, []):
                if unit_names[k] == "Chorus":
                    _occurrence = set()
                    for i in units[k]["segs"]:
                        _occurrence |= lyric_ng[i]
                    _chorus_occurrences.append(_occurrence)
            _mine = set()
            for i in u["segs"]:
                _mine |= lyric_ng[i]
            _nb = sum(segs[i][1] - segs[i][0] for i in u["segs"])
            # A pickup of a bar or two is left to the short-lead join below,
            # which hands it to the section it leads into.
            if (_nb >= STRUCTURE_MIN_SECTION_BARS
                    and _sx_matches_chorus_occurrence(_mine, _chorus_occurrences, _distinct_verse_audio(ui))):
                unit_names[ui] = "Chorus"
                continue
        # One-off sung material.
        if first_ch is not None and first_ch < ui < last_ch:
            sim_verse = 0.0
            if verse is not None:
                vs = [i for k in repeated.get(verse, []) for i in units[k]["segs"]]
                sim_verse = max([W[i, j] for i in u["segs"] for j in vs], default=0.0)
                if STRUCTURE_ONEOFF_VERSE_TYPICAL and len(repeated.get(verse, [])) >= 3:
                    # dev26: like the verse as it usually sounds, not just
                    # like its one closest occurrence (see the setting).
                    _per = sorted(max([W[i, j] for i in u["segs"] for j in units[k]["segs"]], default=0.0)
                                  for k in repeated.get(verse, []))
                    sim_verse = min(sim_verse, float(np.median(_per)))
            unit_names[ui] = "Verse" if sim_verse >= STRUCTURE_SAME_MATERIAL_SIM + 0.05 else "Bridge"
        elif (STRUCTURE_FADING_LAST_CHORUS and first_ch is not None and ui > last_ch
              and ui - 1 >= 0 and units[ui - 1]["cl"] == chorus and lyric_ng
              and _sx_fading_last_chorus(u, units[ui - 1], units, unit_names, repeated, chorus,
                                         lyric_ng, W, loud)):
            unit_names[ui] = "Chorus"
            if info_out is not None:
                # its own section, not merged into the chorus before it
                info_out.setdefault("keep_split", set()).add(u["segs"][0])
                info_out["fading_last_chorus"] = True
        elif first_ch is not None and ui > last_ch:
            _rv = ([v for v in (_sx_word_recurrence(lyric_ng, i) for i in u["segs"]) if v is not None]
                   if (STRUCTURE_LATE_POSTCHORUS and lyric_ng) else [])
            _last_ui = len(units) - 1
            if STRUCTURE_FADE_TAIL_IS_NOT_NEXT:
                # dev56: trailing fade-out stubs do not count as a part after it.
                _sung_l = [float(loud[i]) for uu in units if uu["sung"] for i in uu["segs"]]
                _ref_l = float(np.median(_sung_l)) if _sung_l else None
                while (_ref_l is not None and _last_ui > ui and units[_last_ui]["len"] <= STRUCTURE_SHORT_LEAD_S
                       and max(float(loud[i]) for i in units[_last_ui]["segs"]) <= _ref_l - float(STRUCTURE_FADE_TAIL_DB)):
                    _last_ui -= 1
            if (_rv and ui - 1 >= 0 and units[ui - 1]["cl"] == chorus and ui < _last_ui
                    and float(np.mean(_rv)) >= STRUCTURE_TAG_WORDS_MIN):
                unit_names[ui] = "Post-Chorus"
            else:
                unit_names[ui] = ("Outro" if (ui == len(units) - 1 or (STRUCTURE_NO_BRIDGE_AFTER_LAST_CHORUS and not (
                                      float(STRUCTURE_OUTRO_MAX_SUNG_S or 0) > 0
                                      and u["len"] > float(STRUCTURE_OUTRO_MAX_SUNG_S))))
                                  else "Bridge")
        elif _verse_only_last is not None and chorus is None and ui > _verse_only_last:
            # dev70: one-off sung material after the last verse of a song with
            # no chorus. When a substantial instrumental ending still follows
            # it (STRUCTURE_VERSE_ONLY_BRIDGE_TAIL_S), it is a bridge and the
            # outro starts after it (a rock test track: a one-off sung part at
            # 1:51-2:07, then the solo and ending); otherwise it is the outro,
            # as after a last chorus.
            _u_end = float(bar_edges[segs[u["segs"][-1]][1]])
            if (ui == len(units) - 1
                    and float(duration) - _u_end >= float(STRUCTURE_VERSE_ONLY_BRIDGE_TAIL_S)):
                unit_names[ui] = "Bridge"
            else:
                unit_names[ui] = ("Outro" if (ui == len(units) - 1 or (STRUCTURE_NO_BRIDGE_AFTER_LAST_CHORUS and not (
                                      float(STRUCTURE_OUTRO_MAX_SUNG_S or 0) > 0
                                      and u["len"] > float(STRUCTURE_OUTRO_MAX_SUNG_S))))
                                  else "Bridge")
        else:
            unit_names[ui] = "Verse"

    if STRUCTURE_VERSE_BEFORE_CHORUS and chorus is not None and not instrumental and ch_units:
        # dev55: one-off sung parts in the verse's place -- each straight
        # before a chorus that is not the last one -- that sound alike are
        # the verses, even when they were not alike enough to be grouped as
        # one material (one child-lead test artist "one multi-chorus test scan": 0:16 and 1:14, 0.54).
        _cand = [ui for ui, u in enumerate(units)
                 if unit_names[ui] in ("Verse", "Bridge") and u["cl"] not in repeated and u["sung"]
                 and ui + 1 < len(units) and ui + 1 in ch_units and ui + 1 != last_ch]
        if len(_cand) >= 2:
            _segs_of = {ui: units[ui]["segs"] for ui in _cand}
            _first = _cand[0]
            for ui in _cand[1:]:
                _sim = float(np.mean([W[a_, b_] for a_ in _segs_of[_first] for b_ in _segs_of[ui]]))
                if _sim >= STRUCTURE_VERSE_BEFORE_CHORUS_SIM:
                    unit_names[ui] = "Verse"
                    unit_names[_first] = "Verse"

    if (STRUCTURE_ONEOFF_VERSE_SLOT and chorus is not None and not instrumental
            and (verse is None or STRUCTURE_ONEOFF_VERSE_SLOT_ANY)):
        # dev30: see STRUCTURE_ONEOFF_VERSE_SLOT.
        _runs = []
        for k in range(len(units)):
            if unit_names[k] == "Chorus":
                if _runs and k == _runs[-1][-1] + 1:
                    _runs[-1].append(k)
                else:
                    _runs.append([k])
        if len(_runs) >= 3:
            _mid = list(range(_runs[0][-1] + 1, _runs[1][0]))
            _v1 = [k for k in range(_runs[0][0]) if unit_names[k] == "Verse" and units[k]["sung"]]
            if len(_mid) == 1 and _v1:
                k = _mid[0]
                _w = set()
                for i in units[k]["segs"]:
                    if lyric_ng and i < len(lyric_ng):
                        _w |= lyric_ng[i]
                if (unit_names[k] == "Bridge" and units[k]["sung"]
                        and len(_w) >= STRUCTURE_LYRIC_MIN_NGRAMS):
                    unit_names[k] = "Verse"

    if (STRUCTURE_LATE_BRIDGE and verse is not None and chorus is not None and not instrumental
            and info_out is not None):
        _vocc = [k for k in range(len(units)) if units[k]["cl"] == verse and unit_names[k] == "Verse"]
        _chu = [k for k in range(len(units)) if unit_names[k] == "Chorus"]
        if len(_vocc) >= 3 and _chu:
            k = _vocc[-1]
            others = _vocc[:-1]
            _dens = [units[j]["dens"] for j in others if not np.isnan(units[j]["dens"])]
            _vps = [units[j]["vp"] for j in others if not np.isnan(units[j]["vp"])]
            if (sum(1 for c_ in _chu if c_ < k) >= 2 and max(_chu) > k and _dens and _vps
                    and not np.isnan(units[k]["dens"]) and not np.isnan(units[k]["vp"])
                    and units[k]["dens"] <= STRUCTURE_LATE_BRIDGE_DRUMS * float(np.mean(_dens))
                    and units[k]["vp"] >= float(np.median(_vps)) + STRUCTURE_LATE_BRIDGE_SEMITONES):
                unit_names[k] = "Bridge"
                if k + 1 < len(units) and unit_names[k + 1] == "Bridge":
                    info_out.setdefault("keep_split", set()).add(units[k + 1]["segs"][0])

    if (STRUCTURE_VOCABLE_BRIDGE and chorus is not None and not instrumental and info_out is not None
            and info_out.get("vocable_counts")):
        # dev74: a verse-named part sung (almost) only on vocables, after at
        # least two choruses and before the last, is the middle eight. One
        # boy-band test track runs on one chord loop, so its "na na na"
        # bridge (2:06) was grouped with verse 1 and named Verse 2, while the
        # real second verse (1:05, cut together with its pre-chorus) was a
        # one-off and named Bridge. When such a bridge is found, a one-off
        # part named Bridge that comes before the second chorus and sings
        # words is a verse.
        _heard, _lex = info_out["vocable_counts"]
        _chu = [k for k in range(len(units)) if unit_names[k] == "Chorus"]
        _found = False
        for k in range(len(units)):
            if unit_names[k] != "Verse" or len(_chu) < 3:
                continue
            h = sum(_heard[i] for i in units[k]["segs"] if i < len(_heard))
            lx = sum(_lex[i] for i in units[k]["segs"] if i < len(_lex))
            if (h >= int(STRUCTURE_VOCABLE_BRIDGE_MIN_WORDS)
                    and (h - lx) >= float(STRUCTURE_VOCABLE_BRIDGE_SHARE) * h
                    and sum(1 for c_ in _chu if c_ < k) >= 2 and max(_chu) > k):
                unit_names[k] = "Bridge"
                _found = True
        if _found:
            _second = _chu[1]
            for k in range(len(units)):
                if (unit_names[k] == "Bridge" and k < _second and units[k]["cl"] not in repeated
                        and units[k]["sung"]):
                    h = sum(_heard[i] for i in units[k]["segs"] if i < len(_heard))
                    lx = sum(_lex[i] for i in units[k]["segs"] if i < len(_lex))
                    if lx >= int(STRUCTURE_VOCABLE_BRIDGE_MIN_WORDS) and lx >= 0.5 * h:
                        unit_names[k] = "Verse"

    if (STRUCTURE_WORDLESS_CODA and chorus is not None and not instrumental and info_out is not None
            and info_out.get("vocable_counts")):
        # dev33: see STRUCTURE_WORDLESS_CODA.
        try:
            _heard, _lex = info_out["vocable_counts"]
            _chu = [k for k in range(len(units)) if unit_names[k] == "Chorus"]
            _ulex = [sum(_lex[i] for i in units[k]["segs"] if i < len(_lex)) for k in range(len(units))]
            if _chu and sum(_lex) >= int(STRUCTURE_WORDLESS_CODA_SONG_WORDS):
                _lc = max(_chu)
                for k in range(_lc + 1, len(units)):
                    if unit_names[k] != "Verse" or _ulex[k] > int(STRUCTURE_WORDLESS_CODA_MAX_WORDS):
                        continue
                    _c = units[k]["cl"]
                    if any(j <= _lc for j in range(len(units)) if units[j]["cl"] == _c):
                        continue    # music also heard before the coda (a verse Whisper missed, an intro's reprise)
                    unit_names[k] = "Instrumental"
        except Exception:
            pass
    for ui, u in enumerate(units):
        for i in u["segs"]:
            names[i] = unit_names[ui]
    if STRUCTURE_PRECHORUS_BUILDUP and chorus is not None and not instrumental:
        # dev35: see STRUCTURE_PRECHORUS_BUILDUP.
        try:
            def _segm(arr, i):
                if arr is None:
                    return np.nan
                a_, b_ = segs[i]
                x_ = np.asarray(arr[a_:b_], dtype=float)
                x_ = x_[~np.isnan(x_)]
                return float(np.mean(x_)) if x_.size else np.nan
            _hits = []
            for k in range(len(units) - 1):
                sg = units[k]["segs"]
                if unit_names[k] != "Verse" or unit_names[k + 1] != "Chorus" or len(sg) < 2:
                    continue
                j, head = sg[-1], sg[:-1]
                tl = float(lens[j])
                hl = float(sum(lens[i] for i in head))
                if not (8.0 <= tl <= float(STRUCTURE_PRECHORUS_MAX_S)) or hl < float(STRUCTURE_MIN_SECTION_S):
                    continue
                dvp = _segm(vp_bars, j) - float(np.nanmean([_segm(vp_bars, i) for i in head]))
                if not (dvp >= float(STRUCTURE_PRECHORUS_BUILDUP_ST)):
                    continue
                if F.get("has_stems"):
                    ddr = _segm(F["drums"], j) - float(np.nanmean([_segm(F["drums"], i) for i in head]))
                    if not (ddr > 0.0):
                        continue
                _hits.append(j)
            if len(_hits) >= int(STRUCTURE_PRECHORUS_BUILDUP_MIN_UNITS):
                for j in _hits:
                    names[j] = "Pre-Chorus"
        except Exception:
            pass
    if (STRUCTURE_TITLE_SEGMENTS and title_hits and not instrumental and not lyric_hi
            and len(title_hits) == len(names)
            and not (STRUCTURE_TITLE_SEGMENTS_NEEDS_NEW_WORDS and same_words
                     and max(same_words.values()) >= float(STRUCTURE_LYRIC_SAME_HI_MIN))):
        # dev32: see STRUCTURE_TITLE_SEGMENTS.
        try:
            _seg_u = {i: ui for ui, u in enumerate(units) for i in u["segs"]}
            _dense = [i for i in range(len(names)) if title_hits[i] >= 2 and i in _seg_u]
            if (len(_dense) >= 2 and len({units[_seg_u[i]]["cl"] for i in _dense}) >= 2
                    and all(names[i] in ("Chorus", "Verse") for i in _dense)
                    and any(names[i] == "Verse" for i in _dense)):
                for i in range(len(names)):
                    if i in _dense:
                        names[i] = "Chorus"
                    elif (names[i] == "Chorus" and title_hits[i] == 0 and i in _seg_u
                          and units[_seg_u[i]]["sung"]
                          and (not lyric_ng or (i < len(lyric_ng) and lyric_ng[i]))):
                        names[i] = "Verse"
        except Exception:
            pass
    return names, labels, cov


_SX_RECURRING_NAMES = ("Chorus", "Verse", "Pre-Chorus", "Post-Chorus")
_SX_SUNG_NAMES = ("Chorus", "Verse", "Pre-Chorus", "Post-Chorus", "Bridge")


def _sx_split_instrumental_edges(sections, F, bars, duration):
    """Separate a multi-bar instrumental stretch at the start or end of a sung
    section (a riff between chorus and verse, a solo after a chorus) from it.

    The boundary search can leave such a stretch inside the verse or chorus
    next to it -- most often when it runs over the same chords -- and the
    section was then described as a chorus that "goes instrumental" part way.
    Only runs of at least STRUCTURE_MIN_SECTION_BARS bars (and
    STRUCTURE_MIN_SECTION_S seconds) with essentially no voice are split, and
    only when the sung part left behind is at least as long, so the pauses
    between sung lines and a short lead-in or tail are never touched. A
    stretch next to the Intro / Outro / an Instrumental joins it; otherwise it
    becomes its own Instrumental section. With
    STRUCTURE_SHORT_INSTRUMENTAL_PARTS, a verse/chorus-named section with no
    voice anywhere in it (from 2 bars long) is renamed Instrumental.
    """
    if not sections or not F.get("has_vocal"):
        return sections
    sections = [dict(x) for x in sections]      # never edit the caller's sections
    vocal = np.asarray(F.get("vocal"), dtype=float)
    edges = np.concatenate([np.asarray(bars, dtype=float), [duration]])
    minb = int(STRUCTURE_MIN_SECTION_BARS)

    def _secs(a, b):
        return float(edges[b] - edges[a])

    out = []
    for s in sections:
        a, b = s.get("_a"), s.get("_b")
        short_ok = 2 if STRUCTURE_SHORT_INSTRUMENTAL_PARTS else 2 * minb
        if (s.get("name") not in _SX_SUNG_NAMES or a is None or b is None
                or b - a < short_ok or b > len(vocal)):
            out.append(s)
            continue
        v = vocal[a:b]
        if b - a < 2 * minb and not (float(np.max(v)) < STRUCTURE_SUNG_MIN_COVERAGE
                                     and float(np.mean(v)) < STRUCTURE_UNSUNG_RUN_MEAN):
            out.append(s)
            continue

        def _run(seq):
            # Longest run from the edge with no sung bar and a low average.
            k, best = 0, 0
            while k < len(seq) and seq[k] < STRUCTURE_SUNG_MIN_COVERAGE:
                k += 1
                if float(np.mean(seq[:k])) < STRUCTURE_UNSUNG_RUN_MEAN:
                    best = k
            return best

        lead = _run(v)
        tail = _run(v[::-1][:len(v) - lead])
        use_lead = lead >= minb and _secs(a, a + lead) >= STRUCTURE_MIN_SECTION_S
        use_tail = tail >= minb and _secs(b - tail, b) >= STRUCTURE_MIN_SECTION_S
        if (STRUCTURE_SHORT_INSTRUMENTAL_PARTS and lead == len(v) and len(v) >= 2
                and float(np.mean(v)) < STRUCTURE_UNSUNG_RUN_MEAN):
            # No voice anywhere in it: a pass over the chorus or verse music
            # without the singer (kept apart from the sung chorus before it by
            # a drum entry or fill) is an Instrumental, not "Chorus 2".
            s = dict(s)
            s["name"] = "Instrumental"
            out.append(s)
            continue
        core = (b - a) - (lead if use_lead else 0) - (tail if use_tail else 0)
        if core < minb or core < max(lead if use_lead else 0, tail if use_tail else 0):
            out.append(s)
            continue
        if use_lead:
            prev = out[-1] if out else None
            if prev is not None and prev.get("name") in ("Intro", "Instrumental") and prev.get("_b") == a:
                prev["end"] = float(edges[a + lead])
                prev["_b"] = a + lead
            else:
                out.append({"name": "Instrumental", "start": float(edges[a]),
                            "end": float(edges[a + lead]), "_a": a, "_b": a + lead,
                            "_parts": [], "cluster": "Unique"})
            s = dict(s)
            s["start"] = float(edges[a + lead])
            s["_a"] = a + lead
        if use_tail:
            s = dict(s)
            s["end"] = float(edges[b - tail])
            s["_b"] = b - tail
            out.append(s)
            out.append({"name": "Instrumental", "start": float(edges[b - tail]),
                        "end": float(edges[b]), "_a": b - tail, "_b": b,
                        "_parts": [], "cluster": "Unique", "_tail_of": True})
        else:
            out.append(s)
    # A split-off tail followed by the Outro or another Instrumental joins it.
    merged = []
    for s in out:
        if (merged and merged[-1].get("_tail_of") and s.get("name") in ("Outro", "Instrumental")
                and merged[-1].get("_b") == s.get("_a")):
            s = dict(s)
            s["start"] = merged[-1]["start"]
            s["_a"] = merged[-1]["_a"]
            merged[-1] = s
            continue
        merged.append(s)
    for s in merged:
        s.pop("_tail_of", None)
    try:
        return _sx_cluster_split_pieces([dict(x) for x in merged], F, bars, duration)
    except Exception:
        return merged


def _sx_cluster_split_pieces(sections, F, bars, duration):
    """Cluster letters for the Instrumental pieces split off above, measured
    like every other section: a piece that matches an existing Instrumental
    takes its letter, pieces that match each other share a new one, and the
    rest get letters of their own -- so a riff that recurs reads as the same
    material and different breaks as different."""
    new = [k for k, s in enumerate(sections) if s.get("name") == "Instrumental"
           and not s.get("_parts") and s.get("_a") is not None]
    if not new:
        return sections
    old = [k for k, s in enumerate(sections) if s.get("name") == "Instrumental"
           and s.get("_parts") and s.get("_a") is not None and s.get("cluster") not in (None, "Unique")]
    idx = new + old
    segs = [(int(sections[k]["_a"]), int(sections[k]["_b"])) for k in idx]
    W = _sx_segment_similarity(F, segs)
    used = {str(s.get("cluster")) for s in sections}
    assigned = {}
    for j, k in enumerate(new):
        best = None
        for jj in range(len(new), len(idx)):
            if W[j, jj] >= STRUCTURE_SAME_MATERIAL_SIM and (best is None or W[j, jj] > W[j, best]):
                best = jj
        if best is not None:
            assigned[k] = sections[idx[best]]["cluster"]
    rest = [j for j, k in enumerate(new) if k not in assigned]
    if rest:
        groups = _sx_cluster(W, rest)
        letter_for = {}
        nxt = 0
        for j in rest:
            g = groups[j]
            if g not in letter_for:
                while nxt < 26 and chr(65 + nxt) in used:
                    nxt += 1
                letter_for[g] = chr(65 + nxt) if nxt < 26 else "Z"
                used.add(letter_for[g])
            assigned[new[j]] = letter_for[g]
    for k, cl in assigned.items():
        sections[k]["cluster"] = cl
    return sections


def _sx_split_double_choruses(sections, segs, bars, duration, ev=None, F=None):
    """List a chorus sung twice back to back as two choruses.

    Neighbouring parts that are both the chorus are merged into one section
    (above), which is right when the boundary search cut one chorus into its
    halves and wrong for a double chorus: a chorus-first song's "1:40 Chorus,
    1:58 Chorus" came out as one 35-second "Chorus 3". The two are told apart
    by length. The song's ordinary chorus length is the median of its other
    choruses (at least two of them of four bars or more, leaving out any
    that are themselves half as long again as their median). A merged
    chorus is cut at its inner part boundaries only where every piece is
    close to that length (0.75-1.35x); halves of a chorus are about half of
    it and stay merged. When the boundary search left no cut inside a chorus
    1.75-2.3x the ordinary length, it is cut one ordinary length in (at the
    best-evidenced bar within a bar of that). Either way every piece must
    sound like the same material as the others (a chorus sung again, not a
    chorus plus a tag), so nothing is split without the bar features. Every
    piece keeps the chorus's name and cluster letter. Runs after the
    voiceless parts have been renamed Instrumental, so those never count as
    choruses here."""
    if not sections:
        return sections
    edges = np.concatenate([np.asarray(bars, dtype=float), [float(duration)]])
    ch = [k for k, s in enumerate(sections)
          if s.get("name") == "Chorus" and s.get("_a") is not None and s.get("_b") is not None]
    if len(ch) < 3:
        return sections
    lens = {k: int(sections[k]["_b"]) - int(sections[k]["_a"]) for k in ch}
    minb = int(STRUCTURE_MIN_SECTION_BARS)
    out = []
    for k, s in enumerate(sections):
        if k not in lens:
            out.append(s)
            continue
        others = sorted(lens[j] for j in ch if j != k and lens[j] >= minb)
        if len(others) < 2:
            out.append(s)
            continue
        base = [x for x in others if x <= 1.5 * float(np.median(others))]
        a, b = int(s["_a"]), int(s["_b"])
        L = b - a
        if STRUCTURE_DOUBLE_CHORUS_SHORTEST and others:
            # dev55: the ordinary chorus is the shortest full one when the
            # others are themselves doubles (8, 15 and 16 bars: 8 is the
            # chorus, not the 11.5-bar median of 8 and 15).
            _short = min(others)
            _dbl_ok = True
            if STRUCTURE_DOUBLE_CHORUS_SHORTEST_NEEDS_DOUBLES:
                # dev26: the shortest chorus sets the length only when the
                # other choruses are doubles of it (dev55's 8, 15 and 16 bars),
                # not when it is merely a shorter one (5, 8 and 10 bars).
                _rest = [x for x in others if x != _short]
                _dbl_ok = (not _rest) or all(x >= 1.75 * _short for x in _rest)
            if _dbl_ok and (len(base) < 2 or 1.75 * _short <= L <= 2.3 * _short) and \
                    float(edges[min(a + _short, len(edges) - 1)] - edges[a]) >= 10.0:
                if 1.75 * _short <= L <= 2.3 * _short:
                    base = [_short, _short]
        if len(base) < 2:
            out.append(s)
            continue
        Lc = float(np.median(base))
        if L < 1.6 * Lc:
            out.append(s)
            continue
        lo, hi = max(float(minb), 0.75 * Lc), 1.35 * Lc
        # dev55: the song's last chorus may run longer at its end (a vamp or
        # fade-out on the hook).
        _hi_last = (1.75 * Lc if (STRUCTURE_DOUBLE_CHORUS_SHORTEST and k == ch[-1]) else hi)
        parts = list(s.get("_parts") or [])
        inner = [int(segs[p][0]) for p in parts[1:] if segs is not None and a < int(segs[p][0]) < b]
        pos = [a] + sorted(set(inner)) + [b]
        # Best way to group the parts into chorus-length pieces.
        best = {0: (0.0, [0])}
        for j in range(1, len(pos)):
            cand = None
            for i in range(j):
                if i not in best:
                    continue
                g = pos[j] - pos[i]
                if lo <= g <= (_hi_last if j == len(pos) - 1 else hi):
                    c = best[i][0] + abs(g - Lc)
                    if cand is None or c < cand[0]:
                        cand = (c, best[i][1] + [j])
            if cand is not None:
                best[j] = cand
        cut_bars = None
        last = len(pos) - 1
        if last in best and len(best[last][1]) >= 3:
            cut_bars = [pos[i] for i in best[last][1][1:-1]]
        elif len(pos) == 2 and 1.75 * Lc <= L <= 2.3 * Lc and ev is not None and F is not None:
            target = a + int(round(Lc))
            cands = [q for q in (target - 1, target, target + 1)
                     if lo <= q - a <= hi and lo <= b - q <= hi and 0 < q < len(ev)]
            if cands:
                q = max(cands, key=lambda x: float(ev[x]))
                try:
                    Wq = _sx_segment_similarity(F, [(a, q), (q, b)])
                    if float(Wq[0, 1]) >= STRUCTURE_SAME_MATERIAL_SIM:
                        cut_bars = [q]
                except Exception:
                    cut_bars = None
        if cut_bars and F is not None:
            # Every piece must be the same material as every other (a chorus
            # sung again), not a chorus plus a tag or an extension.
            try:
                pcs = list(zip([a] + cut_bars, cut_bars + [b]))
                Wp = _sx_segment_similarity(F, pcs)
                if min(float(Wp[i, j]) for i in range(len(pcs)) for j in range(len(pcs)) if i != j) \
                        < STRUCTURE_SAME_MATERIAL_SIM:
                    cut_bars = None
            except Exception:
                cut_bars = None
        else:
            cut_bars = None
        if not cut_bars:
            out.append(s)
            continue
        bounds_ = [a] + cut_bars + [b]
        for x, y in zip(bounds_[:-1], bounds_[1:]):
            piece = dict(s)
            piece["start"] = float(edges[x])
            piece["end"] = float(edges[y])
            piece["_a"], piece["_b"] = x, y
            mine = [p for p in parts if segs is not None and x <= int(segs[p][0]) < y]
            if not mine:
                mine = [p for p in parts if segs is not None and int(segs[p][0]) < y <= int(segs[p][1])] or parts[:1]
            piece["_parts"] = mine
            out.append(piece)
    return out


def _sx_join_short_sung_edges(sections, F, bars, duration):
    """Join a sung first or last section that is only a couple of bars long
    to its sung neighbour.

    The boundary search lets the first and last sections be as short as two
    bars (pickup intros, a final hit). When such a stub is SUNG and next to a
    sung section it is not a section of its own: on a chorus-first track the
    singer's pickup came out as "Verse 1 0:00-0:04, Chorus 1 0:04-0:22", and
    the last line ringing out as "Outro 3:36-3:42". A stub joins only when it
    is shorter than STRUCTURE_SHORT_LEAD_S seconds AND than the boundary
    search's minimum bar count (so a real four-bar section at a fast tempo is
    never absorbed), carries voice (average coverage at the sung level), and
    both it and its neighbour are sung parts (an Intro / Outro made of
    instrumental bars, an Instrumental and Silence never join). Needs the
    vocal stem; without it nothing changes. Cluster letters are re-lettered
    in order of appearance afterwards when anything was joined."""
    if not sections or len(sections) < 2 or not isinstance(F, dict) or not F.get("has_vocal"):
        return sections
    vocal = np.asarray(F.get("vocal"), dtype=float)
    edges = np.concatenate([np.asarray(bars, dtype=float), [float(duration)]])
    bar_len = float(np.median(np.diff(edges))) if len(edges) > 2 else 2.0
    min_b = int(STRUCTURE_MIN_SECTION_BARS) * (2 if bar_len < 1.6 else 1)

    def _stub(s):
        a, b = s.get("_a"), s.get("_b")
        if a is None or b is None or not (0 <= int(a) < int(b) <= vocal.size):
            return False
        return (int(b) - int(a) < min_b
                and float(s["end"]) - float(s["start"]) < STRUCTURE_SHORT_LEAD_S
                and float(np.mean(vocal[int(a):int(b)])) >= STRUCTURE_SUNG_MIN_COVERAGE)

    sections = list(sections)
    joined = False
    s0, s1 = sections[0], sections[1]
    if (s0.get("name") in _SX_SUNG_NAMES and s1.get("name") in _SX_SUNG_NAMES
            and abs(float(s1["start"]) - float(s0["end"])) < 0.05 and _stub(s0)):
        s1 = dict(s1)
        s1["start"] = s0["start"]
        s1["_a"] = s0["_a"]
        s1["_parts"] = list(s0.get("_parts") or []) + list(s1.get("_parts") or [])
        sections = [s1] + sections[2:]
        joined = True
    if len(sections) >= 2:
        sp, sl = sections[-2], sections[-1]
        if (sl.get("name") in ("Outro",) + _SX_SUNG_NAMES and sp.get("name") in _SX_SUNG_NAMES
                and abs(float(sl["start"]) - float(sp["end"])) < 0.05 and _stub(sl)):
            sp = dict(sp)
            sp["end"] = sl["end"]
            sp["_b"] = sl["_b"]
            sp["_parts"] = list(sp.get("_parts") or []) + list(sl.get("_parts") or [])
            sections = sections[:-2] + [sp]
            joined = True
    if joined:
        letters = {}
        for s in sections:
            c = s.get("cluster")
            if isinstance(c, str) and len(c) == 1 and c.isalpha() and c not in letters:
                letters[c] = chr(65 + len(letters)) if len(letters) < 26 else "Z"
        sections = [dict(s, cluster=letters.get(s.get("cluster"), s.get("cluster"))) for s in sections]
    return sections


def _sx_merge_named(segs, names, labels, bars, duration, ev=None, events=None, keep_split=None):
    """Merge neighbours that ended up with the same base name.

    Exception: two occurrences of RECURRING material separated by a strong,
    arrangement-marked boundary (a fill or a drum/vocal re-entry plus high
    overall evidence) stay separate -- that is a section repeated back to
    back (a double chorus), not one long section.
    """
    bar_edges = np.concatenate([np.asarray(bars, dtype=float), [duration]])
    keep_split_at = set(keep_split or ())
    out = []
    for i, (a, b) in enumerate(segs):
        nm = names[i]
        keep_split = False
        if out and out[-1]["name"] == nm and ev is not None and a < len(ev):
            recurring = nm in _SX_RECURRING_NAMES or nm.startswith("Theme ")
            marked = False
            if events:
                for key in ("fill", "drum_in", "kick_in", "vocal_in"):
                    arr = events.get(key)
                    if arr is not None and a < len(arr) and arr[a] >= 0.5:
                        marked = True
            keep_split = recurring and marked and float(ev[a]) >= 2.0
        if keep_split_at and i in keep_split_at:
            keep_split = True
        if out and out[-1]["name"] == nm and nm not in ("Section",) and not keep_split:
            out[-1]["end"] = float(bar_edges[b])
            out[-1]["_b"] = b
            out[-1]["_parts"].append(i)
        else:
            out.append({
                "name": nm,
                "start": float(bar_edges[a]),
                "end": float(bar_edges[b]),
                "_a": a, "_b": b, "_parts": [i],
            })
    return out


# dev52: see STRUCTURE_FINAL_CODA.
def _sx_split_final_coda(sections, segs, F, W, bars, duration):
    ch = [k for k, s in enumerate(sections) if s.get("name") == "Chorus"]
    if not ch:
        return sections
    k = ch[-1]
    s = sections[k]
    parts = list(s.get("_parts") or [])
    if len(parts) < 2:
        return sections
    # Nothing sung may follow except an outro or silence.
    for t in sections[k + 1:]:
        if t.get("name") not in ("Outro", "Silence", "Instrumental"):
            return sections
    p = parts[-1]
    voc = F.get("vocal")
    if voc is None:
        return sections
    voc = np.asarray(voc, dtype=float)
    a_p, b_p = int(segs[p][0]), int(segs[p][1])
    if b_p <= a_p or float(np.mean(voc[a_p:b_p])) < STRUCTURE_SUNG_MIN_COVERAGE:
        return sections
    if max(float(W[p, q]) for q in parts[:-1]) >= STRUCTURE_SAME_MATERIAL_SIM:
        return sections
    edges = np.concatenate([np.asarray(bars, dtype=float), [float(duration)]])
    cut = float(edges[a_p])
    head = dict(s)
    head["_parts"] = parts[:-1]
    head["_b"] = a_p
    head["end"] = cut
    tail = dict(s)
    tail["name"] = "Outro"
    tail["_parts"] = [p]
    tail["_a"] = a_p
    tail["start"] = cut
    rest = sections[k + 1:]
    if rest and rest[0].get("name") == "Outro":
        tail["end"] = rest[0]["end"]
        tail["_b"] = rest[0].get("_b", tail.get("_b"))
        tail["_parts"] = tail["_parts"] + list(rest[0].get("_parts") or [])
        rest = rest[1:]
    return sections[:k] + [head, tail] + rest


def _sx_cluster_letters(sections, labels, segs=None):
    """Readable cluster letters (A, B, ...) in order of first appearance.
    A merged section takes the cluster of its longest part."""
    seen = {}
    for s in sections:
        if s["name"] in ("Silence",):
            s["cluster"] = "Unique"
            continue
        parts = [p for p in (s.get("_parts") or []) if p in labels]
        if not parts:
            s["cluster"] = "Unique"
            continue
        if segs is not None:
            cl = labels[max(parts, key=lambda p: segs[p][1] - segs[p][0])]
        else:
            cl = labels[parts[0]]
        if cl not in seen:
            seen[cl] = chr(65 + len(seen)) if len(seen) < 26 else "Z"
        s["cluster"] = seen[cl]
    return sections


def _sx_number(sections):
    totals = {}
    for s in sections:
        totals[s["name"]] = totals.get(s["name"], 0) + 1
    seen = {}
    for s in sections:
        base = s["name"]
        if totals[base] > 1 and base not in ("Silence",):
            seen[base] = seen.get(base, 0) + 1
            s["name"] = "%s %d" % (base, seen[base])
    return sections


_SX_GRID_MEMO = {}


def _sx_audio_memo_key(y, sr, bpm_hint):
    import hashlib
    a = np.ascontiguousarray(np.asarray(y)[::97])
    return (int(len(y)), int(sr), None if bpm_hint is None else float(bpm_hint),
            hashlib.md5(a.tobytes()).hexdigest())


def _sx_grid_and_features(y, sr, bpm_hint=None):
    """Beat grid and per-beat mix features for build_structure_map:
    (beats, bpm, beatfeat) or None when no beats were found. dev55: memoised
    per audio + tempo hint, so the background pass (PARALLEL_ANALYSIS) and
    the in-order call share one computation -- same code, same result."""
    try:
        key = _sx_audio_memo_key(y, sr, bpm_hint)
    except Exception:
        key = None
    if key is not None and key in _SX_GRID_MEMO:
        beats, bpm, beatfeat = _SX_GRID_MEMO.pop(key)
        return list(beats), bpm, dict(beatfeat)
    beats, bpm = _sx_beat_grid(y, sr, bpm_hint=bpm_hint)
    if not beats:
        return None
    beatfeat = _sx_beat_features(y, sr, beats)
    lead, tail = _sx_silence_bounds(y, sr)
    beatfeat["music_start"] = lead
    beatfeat["music_end"] = tail
    try:
        accents = _downbeat_accent_curve(y, sr, np.asarray(beats, dtype=float))
    except Exception:
        accents = None
    beatfeat["accents"] = None if accents is None else np.asarray(accents, dtype=float)
    return beats, bpm, beatfeat


def _sx_precompute_grid(audio_path, bpm_hint=None):
    """dev55: compute and memoise the grid for audio_path (loaded exactly as
    the structure step loads it). Background use only; never raises."""
    try:
        y, sr = librosa.load(audio_path, sr=None, mono=True)
        if len(y) / float(sr) < 20.0:
            return
        key = _sx_audio_memo_key(y, sr, bpm_hint)
        g = _sx_grid_and_features(y, sr, bpm_hint)
        if g is not None:
            _SX_GRID_MEMO.clear()
            _SX_GRID_MEMO[key] = g
    except Exception:
        pass


_SX_NAMER_CATS = ("intro", "verse", "prechorus", "chorus", "postchorus", "bridge",
                  "instrumental", "outro", "other")
_SX_NAMER_NAMES = {"intro": "Intro", "verse": "Verse", "prechorus": "Pre-Chorus",
                   "chorus": "Chorus", "postchorus": "Post-Chorus", "bridge": "Bridge",
                   "instrumental": "Instrumental", "outro": "Outro"}
_SX_NAMER_NUMERIC = 17       # the first columns of a row are numbers; the rest are 0/1 types
_SX_NAMER_MODEL = None       # the loaded model (dict), False when none is found
_SX_NAMER_CAPTURE = None     # a list to collect feature rows (the training tool sets it)


def _sx_namer_cat(name):
    """The section type of a name ("Pre-Chorus 2" -> "prechorus")."""
    l = re.sub(r"[\s\-]+", "", _sx_base_name(name).lower())
    for k in ("prechorus", "postchorus", "chorus", "verse", "bridge", "intro", "outro", "instrumental"):
        if l.startswith(k):
            return k
    return "other"


def _sx_namer_features(segs, names, labels, F, bars, duration, W, cov, lyric_ng, lyric_rep):
    """dev31: one row of numbers per segment for the learned namer -- where
    it sits in the song, how loud / sung / high / drum-heavy it is against the
    song's sung parts, how its words and music come back, and the type the
    rules gave it and its neighbours. Nothing that names the song."""
    n = len(segs)
    be = np.concatenate([np.asarray(bars, dtype=float), [float(duration)]])
    dur = max(1.0, float(duration))
    st = np.array([be[a] for a, _b in segs]); en = np.array([be[b] for _a, b in segs])
    ln = np.maximum(0.5, en - st)
    loud = np.array([float(np.mean(F["loud"][a:b])) for a, b in segs])
    cv = (np.array([float(cov[i]) for i in range(n)]) if cov is not None
          else np.array([float(np.mean(F["vocal"][a:b])) if "vocal" in F else 0.0 for a, b in segs]))
    sung = cv >= STRUCTURE_SUNG_MIN_COVERAGE
    vp = []
    for a, b in segs:
        v = np.asarray(F.get("vpitch", np.full(len(bars), np.nan))[a:b], dtype=float)
        v = v[~np.isnan(v)]
        vp.append(float(np.median(v)) if v.size else np.nan)
    vp = np.array(vp)
    dr = (np.array([float(np.mean(F["drums"][a:b])) for a, b in segs]) if F.get("has_stems")
          else np.zeros(n))
    ref_l = float(np.median(loud[sung])) if sung.any() else float(np.median(loud))
    ref_v = float(np.nanmedian(vp[sung])) if sung.any() and np.any(~np.isnan(vp[sung])) else np.nan
    ref_d = float(np.median(dr[sung])) if sung.any() else float(np.median(dr))
    sung_s = float(np.sum(ln[sung])) or 1.0
    before = np.cumsum(np.concatenate([[0.0], (ln * sung)[:-1]])) / sung_s
    first_s = int(np.argmax(sung)) if sung.any() else -1
    last_s = (n - 1 - int(np.argmax(sung[::-1]))) if sung.any() else -1
    runs = {}
    prev_l = None
    for i in range(n):
        l = labels.get(i) if isinstance(labels, dict) else None
        if l is not None and (i == 0 or labels.get(i - 1) != l):
            runs.setdefault(l, []).append(i)
    cats = [_sx_namer_cat(x) for x in names]
    rows = []
    for i in range(n):
        ng = lyric_ng[i] if (lyric_ng and i < len(lyric_ng)) else set()
        oth = set()
        if lyric_ng:
            for j in range(len(lyric_ng)):
                if abs(j - i) > 1:
                    oth |= lyric_ng[j]
        rec = (len(ng & oth) / float(len(ng))) if len(ng) >= STRUCTURE_LYRIC_MIN_NGRAMS else -0.5
        rep = (lyric_rep[i] if (lyric_rep and i < len(lyric_rep) and lyric_rep[i] is not None) else -0.5)
        l = labels.get(i) if isinstance(labels, dict) else None
        occ = runs.get(l, []) if l is not None else []
        k_occ = sum(1 for r in occ if r <= i) - 1 if occ else 0
        wmax = max([float(W[i, j]) for j in range(n) if abs(j - i) > 1], default=0.0)
        row = [st[i] / dur, en[i] / dur, float(np.log(ln[i])), loud[i] - ref_l, cv[i],
               (vp[i] - ref_v) if not (np.isnan(vp[i]) or np.isnan(ref_v)) else 0.0,
               0.0 if np.isnan(vp[i]) else 1.0,
               float(np.clip(dr[i] / (ref_d + 1e-6), 0.0, 3.0)) if F.get("has_stems") else 1.0,
               float(rep), float(np.log1p(len(ng))), float(rec),
               min(len(occ), 6) / 6.0, (k_occ / max(1, len(occ) - 1)) if len(occ) > 1 else 0.0,
               wmax, 1.0 if i == first_s else 0.0, 1.0 if i == last_s else 0.0, float(before[i])]
        for c in (cats[i], cats[i - 1] if i > 0 else "other", cats[i + 1] if i + 1 < n else "other"):
            row += [1.0 if c == k else 0.0 for k in _SX_NAMER_CATS]
        rows.append(row)
    return np.asarray(rows, dtype=float), cats, sung


def _sx_namer_load():
    """The learned namer's table from STRUCTURE_LEARNED_NAMER_FILE (next to
    this script, or a full path), or None when there is none."""
    global _SX_NAMER_MODEL
    if _SX_NAMER_MODEL is None:
        _SX_NAMER_MODEL = False
        try:
            path = str(STRUCTURE_LEARNED_NAMER_FILE or "")
            if path and not os.path.isabs(path):
                path = os.path.join(os.path.dirname(os.path.abspath(__file__)), path)
            if path and os.path.exists(path):
                with open(path, "r", encoding="utf-8") as fh:
                    m = json.load(fh)
                m["X"] = np.asarray(m["rows"], dtype=float)
                m["y"] = list(m["labels"])
                m["g"] = np.asarray(m["groups"], dtype=int)
                m["mu"] = np.asarray(m["mean"], dtype=float)
                m["sd"] = np.asarray(m["std"], dtype=float)
                m["wt"] = np.asarray(m.get("weights") or [1.0] * len(m["mu"]), dtype=float)
                if m["X"].ndim == 2 and m["X"].shape[1] == len(m["mu"]) and len(m["y"]) == len(m["X"]):
                    _SX_NAMER_MODEL = m
        except Exception:
            _SX_NAMER_MODEL = False
    return _SX_NAMER_MODEL or None


def _sx_learned_namer(segs, names, labels, F, bars, duration, W, cov, lyric_ng, lyric_rep):
    """dev31 STRUCTURE_LEARNED_NAMER: names checked against sections from
    reference songs that look alike (the K nearest by the rows above). A
    segment is renamed only when STRUCTURE_LEARNED_NAMER_AGREE of the vote
    names one other type and the neighbours come from at least
    STRUCTURE_LEARNED_NAMER_MIN_SONGS different songs; a sung segment only
    gets a sung name, an unsung one only Intro / Outro / Instrumental, and
    Intro / Outro only at the song's start / end. Segments are decided a
    section at a time (each run of one name, votes added up), and never so
    that a section would join its neighbour: the namer changes names only,
    never a section line. With no model file, or the training tool collecting
    rows, the names are returned unchanged."""
    if not STRUCTURE_LEARNED_NAMER:
        return names
    X, cats, sung = _sx_namer_features(segs, names, labels, F, bars, duration, W, cov, lyric_ng, lyric_rep)
    if _SX_NAMER_CAPTURE is not None:
        _SX_NAMER_CAPTURE.append({"segs": [(int(a), int(b)) for a, b in segs],
                                  "times": [(float(np.concatenate([np.asarray(bars, float), [duration]])[a]),
                                             float(np.concatenate([np.asarray(bars, float), [duration]])[b]))
                                            for a, b in segs],
                                  "X": X.tolist(), "cats": cats})
        return names
    m = _sx_namer_load()
    if not m or len(names) < 3:
        return names
    Z = (X - m["mu"]) / m["sd"] * m["wt"]
    k = int(STRUCTURE_LEARNED_NAMER_K)
    out = list(names)
    n = len(names)
    seg_votes, seg_songs = [], []
    for i in range(n):
        d = np.sqrt(np.sum((m["X"] - Z[i]) ** 2, axis=1))
        nn = np.argsort(d)[:k]
        votes, songs = {}, {}
        for j in nn:
            votes[m["y"][j]] = votes.get(m["y"][j], 0.0) + 1.0 / (0.25 + float(d[j]))
            songs.setdefault(m["y"][j], set()).add(int(m["g"][j]))
        seg_votes.append(votes)
        seg_songs.append(songs)
    # segments decided together: each run of segments carrying one name (the
    # section they will make), so a new name never moves a section line
    groups = {}
    run = 0
    for i in range(n):
        if i > 0 and names[i] != names[i - 1]:
            run += 1
        groups.setdefault(run, []).append(i)
    first_of = {i: mem[0] for mem in groups.values() for i in mem}
    last_of = {i: mem[-1] for mem in groups.values() for i in mem}
    decide = {}
    for key, mem in groups.items():
        votes, songs = {}, {}
        for i in mem:
            for c_, w_ in seg_votes[i].items():
                votes[c_] = votes.get(c_, 0.0) + w_
            for c_, g_ in seg_songs[i].items():
                songs.setdefault(c_, set()).update(g_)
        tot = sum(votes.values())
        best, bw = max(votes.items(), key=lambda kv: kv[1]) if votes else (None, 0.0)
        for i in mem:
            decide[i] = (best, bw, tot, len(songs.get(best, ())))
    for i in range(n):
        if cats[i] == "other" or _sx_base_name(names[i]) == "Silence":
            continue
        best, bw, tot, n_songs = decide[i]
        if best == cats[i] or best not in _SX_NAMER_NAMES or bw < float(STRUCTURE_LEARNED_NAMER_AGREE) * tot:
            continue
        if n_songs < int(STRUCTURE_LEARNED_NAMER_MIN_SONGS):
            continue
        g_sung = sum(1 for q in range(first_of[i], last_of[i] + 1) if sung[q]) * 2 > (last_of[i] - first_of[i] + 1)
        if g_sung and best == "instrumental":
            continue
        if not g_sung and best not in ("intro", "outro", "instrumental"):
            continue
        a_, b_ = first_of[i], last_of[i]
        if best == "intro" and any(_sx_namer_cat(names[q]) not in ("intro", "other") for q in range(a_)):
            continue
        if best == "outro" and any(_sx_namer_cat(names[q]) not in ("outro", "other") for q in range(b_ + 1, n)):
            continue
        if (a_ > 0 and _sx_namer_cat(out[a_ - 1]) == best) or (b_ + 1 < n and cats[b_ + 1] == best):
            continue        # the new name would join it to its neighbour and lose a section line
        out[i] = _SX_NAMER_NAMES[best]
    return out


def build_structure_map(
    y=None, sr=None, total_duration=None, *,
    beatfeat=None, beats=None, bpm_hint=None,
    stem_notes=None, lyric_chunks=None, lyric_groups=None,
    instrumental=False, details_out=None, prechorus_by_words=False, sheet_intro=False,
    lone_sung_verse=True, learned_namer=False, title=None,
):
    """Signal-only section map: [{"name","start","end","cluster", ...}].

    Either `y`/`sr` (mono audio) or a cached `beatfeat` + `beats` from an
    earlier call (details_out) must be supplied. Stem note events and lyric
    timestamps are optional extra evidence; without them the map is built
    from the mix alone.
    """
    if details_out is None:
        details_out = {}
    # ---- Grid and mix features (cached between calls) -------------------
    if beatfeat is None or beats is None:
        if y is None or sr is None:
            return []
        duration = len(y) / float(sr)
        if duration < 20.0:
            return []
        _g = _sx_grid_and_features(y, sr, bpm_hint)
        if _g is None:
            return []
        beats, bpm, beatfeat = _g
        details_out["bpm"] = bpm
    duration = float(total_duration or beatfeat["duration"])
    details_out["beatfeat"] = beatfeat
    details_out["beats"] = list(beats)
    music_start = float(beatfeat.get("music_start", 0.0) or 0.0)
    music_end = float(beatfeat.get("music_end", duration) or duration)
    music_end = min(music_end, duration)

    # ---- Lyric blocks (optional) ---------------------------------------
    lyric_blocks = []
    if lyric_chunks:
        try:
            if lyric_groups is None:
                lyric_groups = _detect_lyric_repetitions(lyric_chunks)
            lyric_blocks = _lyric_block_segmentation(lyric_chunks, lyric_groups)
        except Exception:
            lyric_blocks = []
    # Chunks timed against the file's own lyrics: split them back into the
    # lyric lines, each placed where the voice sings it, and find where each
    # of the lyrics' own stanzas is sung. That is used only to correct section
    # starts that the arrangement put a little early (below); the evidence and
    # the naming use the chunks exactly as before.
    stanza_blocks = None
    tag_line_list = None
    if lyric_chunks and STRUCTURE_LYRIC_PHRASES and STRUCTURE_TAG_STANZAS and any(
            isinstance(c, dict) and c.get("lines") for c in lyric_chunks):
        try:
            _vn = None
            for _key in ("vocals", "vocal", "voice"):
                if stem_notes and stem_notes.get(_key):
                    _vn = stem_notes[_key]
                    break
            _split = _sx_split_tag_lines(lyric_chunks, None if instrumental else _vn)
            stanza_blocks = _lyric_blocks_from_tag_lines(_split)
            if stanza_blocks is not None:
                tag_line_list = _split
        except Exception:
            stanza_blocks = None
            tag_line_list = None

    # ---- Bars ------------------------------------------------------------
    beats_arr = np.asarray(beats, dtype=float)
    beats_in = beats_arr[(beats_arr >= music_start - 0.05) & (beats_arr <= music_end + 0.05)]
    if beats_in.size < 16:
        beats_in = beats_arr
    accents = beatfeat.get("accents")
    if accents is not None and len(accents) == len(beats_arr):
        acc_in = np.asarray(accents)[(beats_arr >= music_start - 0.05) & (beats_arr <= music_end + 0.05)]
        if acc_in.size != beats_in.size:
            acc_in = None
    else:
        acc_in = None

    # Beats per bar from the accent cycle (3 or 4; 6/8 and 12/8 count in 3s).
    bpb = int(details_out.get("beats_per_bar") or 0) or None
    if bpb is None:
        bpb = 4
        try:
            if acc_in is not None:
                _db, est = _estimate_meter_from_beats(beats_in, acc_in)
                if int(est or 0) in (3, 4):
                    bpb = int(est)
                elif int(est or 0) in (6, 12):
                    bpb = 3
        except Exception:
            bpb = 4
    # Downbeats: a first pass without section events, then -- once the
    # events (fills, drum/vocal entries) are known -- a second pass that
    # lets them pin the bar lines.
    flags0 = _sx_downbeat_flags(beats_in, bpb, acc_in, stem_notes, None)
    bars0 = np.asarray([music_start] + [float(b) for b, f in zip(beats_in, flags0)
                                         if f and music_start + 0.25 < b < music_end - 0.25])
    flags = flags0
    if stem_notes and bars0.size >= 4:
        F0 = _sx_bar_features(beatfeat, bars0, music_end, stem_notes, instrumental)
        _e0, votes = _sx_event_evidence(F0, bars0, music_end, stem_notes, lyric_blocks)
        if votes:
            flags = _sx_downbeat_flags(beats_in, bpb, acc_in, stem_notes, votes)
    bar_list = [float(b) for b, f in zip(beats_in, flags) if f and music_start + 0.25 < b < music_end - 0.25]
    bars = np.asarray([music_start] + bar_list, dtype=float)
    if len(bars) < 2 * STRUCTURE_MIN_SECTION_BARS:
        return []
    details_out["downbeats"] = [float(b) for b, f in zip(beats_in, flags) if f]
    details_out["beats_per_bar"] = bpb
    details_out["beat_len"] = float(np.median(np.diff(beats_arr))) if beats_arr.size > 1 else None

    # Bar features are pooled over the music span only.
    F = _sx_bar_features(beatfeat, bars, music_end, stem_notes, instrumental)

    # Static material (a held tone, a drone, steady noise, a click track):
    # every feature is z-scored per track below, which would blow up tiny
    # fluctuations into convincing-looking section changes. Measured on the
    # RAW features instead, such material barely moves at all.
    try:
        Mraw = np.asarray(F["mfcc"], dtype=float)
        nbr = Mraw.shape[1]
        iu = np.triu_indices(nbr, k=min(4, max(1, nbr - 1)))
        sq = np.sum(Mraw * Mraw, axis=0)
        d2 = np.maximum(sq[:, None] + sq[None, :] - 2.0 * (Mraw.T @ Mraw), 0.0)
        dists = np.sqrt(d2[iu])
        static = (dists.size > 0 and float(np.percentile(dists, 90)) < 12.0
                  and float(np.std(F["loud"])) < 1.5)
    except Exception:
        static = False
    if static:
        out = []
        if music_start > 0.0:
            out.append({"name": "Silence", "start": 0.0, "end": round(music_start, 2), "cluster": "Unique"})
        out.append({"name": "Section", "start": round(music_start, 2), "end": round(music_end, 2), "cluster": "A"})
        if music_end < float(beatfeat.get("duration", duration) or duration) - 0.05 and (duration - music_end) >= STRUCTURE_SILENCE_MIN_S:
            out.append({"name": "Silence", "start": round(music_end, 2), "end": round(float(duration), 2), "cluster": "Unique"})
        out[0]["start"] = 0.0
        out[-1]["end"] = round(float(duration), 2)
        details_out["static"] = True
        details_out["bars"] = bars
        return out

    ev, events, _votes = _sx_evidence_curve(F, bars, music_end, stem_notes, lyric_blocks)
    starts = _sx_boundary_dp(bars, music_end, ev)
    try:
        starts = _sx_shift_off_fill_bar(starts, events, len(bars), bars, music_end)
    except Exception:
        pass
    try:
        starts = _sx_shift_to_reentry(starts, bars, F, music_end, events)
    except Exception:
        pass
    if stanza_blocks and tag_line_list:
        try:
            starts = _sx_refine_starts_to_stanzas(starts, bars, music_end, stanza_blocks, tag_line_list)
        except Exception:
            pass
        if STRUCTURE_STANZA_BACKSHIFT:
            try:
                starts = _sx_backshift_starts_to_stanzas(starts, bars, music_end, stanza_blocks, tag_line_list)
            except Exception:
                pass
    if STRUCTURE_SPLIT_BY_KNOWN_PARTS:
        try:
            starts = _sx_split_by_known_parts(
                F, bars, music_end, starts, lyric_chunks=None if instrumental else lyric_chunks,
                vocal_notes=None if instrumental else _sx_vocal_notes_of(stem_notes))
        except Exception:
            pass
    details_out["dp_starts"] = [float(bars[k]) for k in starts]
    cuts = [0] + list(starts) + [len(bars)]
    segs = [(cuts[k], cuts[k + 1]) for k in range(len(cuts) - 1) if cuts[k + 1] > cuts[k]]
    if not segs:
        return []

    bar_edges = np.concatenate([bars, [music_end]])
    _vn = None
    for _key in ("vocals", "vocal", "voice"):
        if stem_notes and stem_notes.get(_key):
            _vn = stem_notes[_key]
            break

    # dev28: a song that changes key for its last part -- the sections after
    # the change are compared with the rest in their own key (see
    # STRUCTURE_KEY_CHANGE_ALIGN); boundaries are not affected.
    F_same = F
    if STRUCTURE_KEY_CHANGE_ALIGN and not instrumental:
        try:
            _kc = _sx_key_change(F, bars, music_end)
        except Exception:
            _kc = None
        if _kc:
            _cb, _k = _kc
            _ch = np.array(F["chroma"], dtype=float, copy=True)
            _ch[:, _cb:] = np.roll(_ch[:, _cb:], -_k, axis=0)
            F_same = dict(F, chroma=_ch)
            details_out["key_change"] = (float(bars[_cb]), int(_k))

    def _measure_and_name(segs):
        lyric_sig, lyric_rep = (None, None)
        if lyric_chunks:
            try:
                lyric_sig, lyric_rep = _sx_lyric_signatures(segs, bar_edges, lyric_chunks, lyric_groups)
            except Exception:
                lyric_sig, lyric_rep = (None, None)
        lyric_ng = None
        if lyric_chunks and STRUCTURE_LYRIC_PHRASES:
            try:
                lyric_ng = _sx_lyric_ngrams(segs, bar_edges, lyric_chunks,
                                            vocal_notes=None if instrumental else _vn)
            except Exception:
                lyric_ng = None
        W = _sx_segment_similarity(F_same, segs, lyric_sig, lyric_rep, lyric_ng=lyric_ng)
        _name_info = {}
        _smeared = False
        if STRUCTURE_SMEARED_LYRICS_HOOK and lyric_chunks and not instrumental:
            try:
                _smeared = _sx_lyric_timing_smeared(segs, bar_edges, F, lyric_chunks)
            except Exception:
                _smeared = False
        _vocable_only = None
        if STRUCTURE_VOCABLE_INTRO and lyric_chunks and not instrumental:
            try:
                _vo, _lex, _heard = _sx_vocable_only_segments(segs, bar_edges, lyric_chunks)
                # Only when later sections have real words to compare with.
                if sum(1 for x in _lex if x >= 4) >= 2:
                    _vocable_only = _vo
                    _name_info["vocable_counts"] = (list(_heard), list(_lex))
            except Exception:
                _vocable_only = None
        _th = None
        if STRUCTURE_TITLE_PICKS_CHORUS and title and lyric_chunks and not instrumental:
            try:
                _th = _sx_title_hits(segs, bar_edges, lyric_chunks, title)    # dev32
            except Exception:
                _th = None
        names, labels, _cov = _sx_name_sections(
            F, bars, music_end, segs, W, lyric_sig=lyric_sig, lyric_rep=lyric_rep,
            vocal_known=bool(stem_notes), instrumental=instrumental, lyric_ng=lyric_ng,
            lyric_smeared=_smeared, info_out=_name_info, vocable_only=_vocable_only,
            sheet_intro=sheet_intro, lone_sung_verse=lone_sung_verse, title_hits=_th,
        )
        _name_info["_ng"], _name_info["_rep"] = lyric_ng, lyric_rep    # dev31: for the learned namer
        return names, labels, _cov, W, _name_info

    names, labels, _cov, W, _name_info = _measure_and_name(segs)
    if (prechorus_by_words and STRUCTURE_PRECHORUS_BY_WORDS and lyric_chunks and not instrumental
            and stem_notes):
        # dev27: see STRUCTURE_PRECHORUS_BY_WORDS.
        try:
            _pc = _sx_prechorus_cuts(segs, names, bars, music_end, ev, lyric_chunks,
                                     None if instrumental else _vn)
        except Exception:
            _pc = None
        if STRUCTURE_PRECHORUS_BUILDUP_CUTS and not _pc:
            try:
                _pc = _sx_prechorus_cuts_buildup(segs, names, bars, music_end, F, ev)   # dev35
            except Exception:
                _pc = None
        if _pc:
            _st2 = sorted(set(int(k) for k in starts) | set(_pc))
            _cuts2 = [0] + _st2 + [len(bars)]
            _segs2 = [(_cuts2[k], _cuts2[k + 1]) for k in range(len(_cuts2) - 1) if _cuts2[k + 1] > _cuts2[k]]
            try:
                _n2, _l2, _c2, _W2, _ni2 = _measure_and_name(_segs2)
                _n2, _l2 = _sx_mark_prechorus(_segs2, _n2, _l2, _pc)
                if STRUCTURE_PRECHORUS_KEEPS_NAMES and not _sx_names_kept(segs, names, _segs2, _n2):
                    # dev30: see STRUCTURE_PRECHORUS_KEEPS_NAMES -- the first
                    # map's names, on the new lines, with the pre-chorus marked
                    _n2, _l2, _ni2 = _sx_inherit_names(segs, names, labels, _name_info, _segs2, _pc)
            except Exception:
                _n2 = None
            if _n2 is not None:
                starts, segs = _st2, _segs2
                names, labels, _cov, W, _name_info = _n2, _l2, _c2, _W2, _ni2
                details_out["dp_starts"] = [float(bars[k]) for k in starts]
    if learned_namer and not instrumental:
        # dev31: see STRUCTURE_LEARNED_NAMER.
        try:
            names = _sx_learned_namer(segs, names, labels, F, bars, music_end, W, _cov,
                                      _name_info.get("_ng"), _name_info.get("_rep"))
        except Exception:
            pass
    sections = _sx_merge_named(segs, names, labels, bars, music_end, ev, events,
                               keep_split=_name_info.get("keep_split"))
    if STRUCTURE_FINAL_CODA and not instrumental and stem_notes and not _name_info.get("fading_last_chorus"):
        # (dev57: a last chorus that fades out keeps its fade -- no coda split)
        try:
            sections = _sx_split_final_coda(sections, segs, F, W, bars, music_end)
        except Exception:
            pass
    sections = _sx_cluster_letters(sections, labels, segs)
    if STRUCTURE_SPLIT_INSTRUMENTAL and not instrumental and stem_notes:
        try:
            sections = _sx_split_instrumental_edges(sections, F, bars, music_end)
        except Exception:
            pass
    if STRUCTURE_SPLIT_DOUBLE_CHORUS:
        try:
            sections = _sx_split_double_choruses(sections, segs, bars, music_end, ev, F)
        except Exception:
            pass
    if STRUCTURE_SHORT_LEAD_S > 0 and len(sections) >= 2 and not instrumental:
        try:
            sections = _sx_join_short_sung_edges(sections, F, bars, music_end)
        except Exception:
            pass
    if _name_info.get("fading_last_chorus") and len(sections) >= 2:
        # dev57: the near-silent last seconds after a fading last chorus are
        # the end of that fade, not an Outro / Instrumental of their own.
        try:
            last, prev = sections[-1], sections[-2]
            _sl = [float(np.mean(F["loud"][a_:b_])) for (a_, b_) in segs
                   if float(np.mean(F["vocal"][a_:b_])) >= STRUCTURE_SUNG_MIN_COVERAGE]
            _ref = float(np.median(_sl)) if _sl else None
            ka = int(np.searchsorted(bar_edges, float(last["start"]) - 1e-6))
            kb = int(np.searchsorted(bar_edges, float(last["end"]) - 1e-6))
            if (prev.get("name") == "Chorus" and last.get("name") in ("Outro", "Instrumental")
                    and _ref is not None and kb > ka
                    and float(last["end"]) - float(last["start"]) <= 1.5 * STRUCTURE_SHORT_LEAD_S
                    and float(np.mean(F["vocal"][ka:kb])) < STRUCTURE_SUNG_MIN_COVERAGE
                    and float(np.max(F["loud"][ka:kb])) <= _ref - float(STRUCTURE_FADE_TAIL_DB)):
                prev["end"] = last["end"]
                if "_b" in last:
                    prev["_b"] = last["_b"]
                sections = sections[:-1]
        except Exception:
            pass

    audio_dur = float(beatfeat.get("duration", duration) or duration)
    if music_start > 0.0:
        sections.insert(0, {"name": "Silence", "start": 0.0, "end": music_start, "cluster": "Unique"})
    if music_end < audio_dur - 0.05 and (duration - music_end) >= STRUCTURE_SILENCE_MIN_S:
        # Only silence actually measured in the decoded audio becomes a
        # section; a container that reports a slightly longer duration than
        # it decodes to just stretches the last section.
        sections.append({"name": "Silence", "start": music_end, "end": duration, "cluster": "Unique"})
    if STRUCTURE_VERSE_PARTS:
        # dev28: see STRUCTURE_VERSE_PARTS.
        for s in sections:
            try:
                parts = list(s.get("_parts") or [])
                if s.get("name") != "Verse" or len(parts) < 3:
                    continue
                cl = [labels.get(i) for i in parts]
                back = any(cl[x] is not None and cl[x] == cl[z] and cl[y] != cl[x]
                           for x in range(len(cl)) for y in range(x + 1, len(cl)) for z in range(y + 1, len(cl)))
                if back:
                    s["subparts"] = [round(float(bar_edges[segs[i][0]]), 2) for i in parts[1:]]
            except Exception:
                continue
    sections = _sx_number(sections)
    out = []
    for s in sections:
        d = {"name": s["name"], "start": round(float(s["start"]), 2),
             "end": round(float(s["end"]), 2), "cluster": s.get("cluster", "Unique")}
        if s.get("subparts"):
            d["subparts"] = list(s["subparts"])
        out.append(d)
    if out:
        out[0]["start"] = 0.0
        out[-1]["end"] = round(float(duration), 2)
    details_out["evidence"] = ev
    details_out["bars"] = bars
    return out


# Names the audio model may contribute for parts the measurements can only
# call "Instrumental"/"Section". Sung sections are never renamed by it: their
# names come from measured repetition, vocal register and lyrics.
_SX_MOSS_INSTRUMENTAL_ROLES = {
    "solo": "Solo", "guitar solo": "Guitar Solo", "piano solo": "Piano Solo",
    "sax solo": "Sax Solo", "synth solo": "Synth Solo", "keyboard solo": "Keyboard Solo",
    "instrumental solo": "Solo", "breakdown": "Breakdown", "interlude": "Interlude",
    "instrumental break": "Instrumental Break", "break": "Break", "drop": "Drop",
    "build": "Build", "build-up": "Build-Up", "buildup": "Build-Up", "riff": "Riff",
    "instrumental": "Instrumental", "vamp": "Vamp",
}


def _sx_moss_name_hints(sections, audio_path, moss_model, moss_processor):
    """Ask the audio model to name the measured parts. Returns a list of
    (start, end, label) hints; never changes boundaries."""
    if moss_model is None or moss_processor is None or len(sections) < 2:
        return []
    try:
        seg_list = []
        for i, s in enumerate(sections):
            seg_list.append(
                "Part %d (%s - %s, %.0fs): Acoustic Cluster '%s'"
                % (i + 1, _seconds_to_mmss(s["start"]), _seconds_to_mmss(s["end"]),
                   float(s["end"]) - float(s["start"]), s.get("cluster", "Unique"))
            )
        prompt = STRUCTURE_LABELING_PROMPT.format(segment_list="\n".join(seg_list))
        conversation = [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "audio", "path": audio_path},
            ],
        }]
        status("Naming measured structure segments with the audio model...")
        reply = moss_generate(moss_model, moss_processor, conversation, max_new_tokens=384)
        labels = _parse_structure_labels(reply, len(sections))
    except Exception as e:
        print("  (structure naming pass skipped: %s)" % e)
        return []
    hints = []
    for idx, label in labels.items():
        s = sections[idx - 1]
        hints.append((float(s["start"]), float(s["end"]), str(label)))
    return hints


def _sx_apply_moss_hints(sections, hints):
    """Adopt the audio model's name only for unsung parts the measurements
    could name just generically, and only where its part overlaps ours."""
    if not sections or not hints:
        return sections
    out = [dict(s) for s in sections]
    for s in out:
        base = re.sub(r"\s*\d+\s*$", "", str(s.get("name") or "")).strip()
        if base not in ("Instrumental", "Section"):
            continue
        a, b = float(s["start"]), float(s["end"])
        best, best_ov = None, 0.0
        for ha, hb, lab in hints:
            ov = min(b, hb) - max(a, ha)
            if ov > best_ov:
                best_ov, best = ov, lab
        if best is None or best_ov < 0.6 * max(1e-6, b - a):
            continue
        key = re.sub(r"\s*\d+\s*$", "", best).strip().lower()
        if key in _SX_MOSS_INSTRUMENTAL_ROLES:
            s["name"] = _SX_MOSS_INSTRUMENTAL_ROLES[key]
    # Re-number after renaming.
    for s in out:
        s["name"] = re.sub(r"\s*\d+\s*$", "", str(s["name"])).strip()
    return _sx_number(out)




def _stem_rms_by_section(path, sections=None):
    """Load a stem once and return (whole_track_rms, per_section_rms_list).

    per_section_rms_list is a list of {"name","start","end","rms"} dicts, one
    per structural section, computed from the SAME audio read used for the
    whole-track figure (so this costs nothing extra beyond one file load).
    Used so the writer can describe an instrument's intensity in different
    parts of the song ("the guitar barely registers in the verse but comes
    in hard for the chorus") instead of only a single track-wide number.
    """
    try:
        y, sr = librosa.load(path, sr=None, mono=True)
    except Exception:
        return 0.0, []
    if y.size == 0:
        return 0.0, []
    whole_rms = float(np.sqrt(np.mean(y ** 2)))
    section_rms = []
    if sections:
        for sec in sections:
            try:
                start = max(0.0, float(sec["start"]))
                end = float(sec["end"])
            except Exception:
                continue
            i0 = int(start * sr)
            i1 = max(i0 + 1, int(end * sr))
            seg = y[i0:i1]
            sec_rms = float(np.sqrt(np.mean(seg ** 2))) if seg.size else 0.0
            section_rms.append({
                "name": sec.get("name", "?"),
                "start": start,
                "end": end,
                "rms": sec_rms,
            })
    return whole_rms, section_rms


def _section_intensity_label(rms, ref_min, ref_max):
    """Map a section's RMS to a relative intensity band, scaled against THIS
    STEM's own quietest/loudest section (not an absolute loudness scale).
    That matters because an instrument that's naturally soft after stem
    separation (e.g. a clean rhythm guitar under a loud vocal) should still
    read as "prominent" in the section where it's most forward, rather than
    always reading "quiet" against some fixed threshold that only really
    suits drums or bass."""
    if ref_max <= 0 or (ref_max - ref_min) < 1e-6:
        return "steady"
    # Measured in dB below this stem's loudest section rather than as a linear
    # fraction of the min-max range. The linear version stretched ANY spread to
    # the full scale, so a part that varied by barely 1 dB was reported as
    # "quiet / mostly absent" in one section and "peak intensity" in another --
    # a dramatic arrangement change that was never played. A spread under
    # ~2 dB is now reported as steady, and every label corresponds to a real
    # level difference a listener would hear.
    floor = 1e-9
    spread_db = 20.0 * np.log10(max(ref_max, floor) / max(ref_min, floor))
    if spread_db < 2.0:
        return "steady"
    below_db = 20.0 * np.log10(max(ref_max, floor) / max(float(rms), floor))
    if below_db <= 1.5:
        return "peak intensity"
    elif below_db <= 3.5:
        return "prominent"
    elif below_db <= 7.0:
        return "moderate"
    elif below_db <= 14.0:
        return "present but restrained"
    else:
        return "quiet / mostly absent"


def _format_intensity_by_section(stem, section_rms_list, section_totals=None, whole_rel_db=None):
    """Render the per-section relative-intensity block for one stem.

    `section_totals` (optional): combined RMS of all stems per section (the mix
    level there); a section where this stem is SECTION_RESIDUE_BELOW_TOTAL_DB
    or more below it is "not playing". `whole_rel_db` (optional): the stem's
    whole-track level relative to all stems, used to say when the whole stem is
    only a minor or residual part.
    """
    if not section_rms_list:
        return ""
    floor = float(SECTION_AUDIBLE_MIN_RMS)
    loudest = max(float(s["rms"]) for s in section_rms_list)
    if loudest < floor:
        return (
            f"INTENSITY BY SECTION ({stem} stem): near-silent in every section (never above "
            f"about -54 dBFS) -- this stem carries only separation residue here, so treat the "
            f"instrument as not playing; do not describe it building, peaking or dropping out."
        )

    totals = list(section_totals or [])

    def _not_playing(r, i=None):
        if float(r) < floor and float(r) < loudest * 10.0 ** (-30.0 / 20.0):
            return True
        if i is not None and i < len(totals) and totals[i] > 0:
            return float(r) < totals[i] * 10.0 ** (-float(SECTION_RESIDUE_BELOW_TOTAL_DB) / 20.0)
        return False

    audible = [s["rms"] for i, s in enumerate(section_rms_list) if not _not_playing(s["rms"], i)]
    if not audible:
        return (
            f"INTENSITY BY SECTION ({stem} stem): at least {SECTION_RESIDUE_BELOW_TOTAL_DB:.0f} dB "
            f"below the rest of the mix in every section -- separation residue, so treat the "
            f"instrument as not playing; do not describe it building, peaking or dropping out."
        )
    ref_min, ref_max = min(audible), max(audible)
    lines = [
        f"INTENSITY BY SECTION ({stem} stem energy relative to its own "
        "loudest section in this track, in real level steps; 'steady' means it "
        "stays at about the same level throughout; 'not playing' means the stem is "
        "below an absolute audibility floor there -- use this for how the "
        "instrument's presence rises/falls across the song, not for "
        "absolute loudness or cross-instrument comparison):"
    ]
    if whole_rel_db is not None and whole_rel_db <= -20.0:
        lines[0] = lines[0][:-1] + (
            f" NOTE: overall this stem sits {abs(whole_rel_db):.0f} dB below the combined stems, "
            "so even its 'peak' sections are a minor part of the mix):"
        )
    for i, s in enumerate(section_rms_list):
        if _not_playing(s["rms"], i):
            label = "not playing (below audibility floor)"
        else:
            label = _section_intensity_label(s["rms"], ref_min, ref_max)
        lines.append(
            f"  {s['name']} ({_seconds_to_mmss(s['start'])}-{_seconds_to_mmss(s['end'])}): {label}"
        )
    return "\n".join(lines)


def _mmss_to_seconds(s):
    """Parse a 'M:SS', 'MM:SS', or 'H:MM:SS' timestamp into seconds. Returns
    None if it doesn't look like a timestamp."""
    if not s:
        return None
    parts = s.strip().split(":")
    if not (2 <= len(parts) <= 3):
        return None
    try:
        parts = [float(p) for p in parts]
    except ValueError:
        return None
    seconds = 0.0
    for p in parts:
        seconds = seconds * 60.0 + p
    return seconds


_STRUCTURE_FIELD_RE = re.compile(
    r"(?i:STRUCTURE)\s*[=:]\s*(.*?)(?=\n\s*[A-Z][A-Z0-9_]{2,}\s*[=:]|\n\n|\Z)",
    re.DOTALL,
)

# Dash-like characters between two timestamps.
_STRUCTURE_DASH = r"[-\u2010\u2011\u2012\u2013\u2014\u2015\u2212]"
_STRUCTURE_TIME = r"\d{1,2}:\d{2}(?::\d{2})?"

# FIXED: '*' quantifiers restored so single- and multi-word section names
# ("Intro", "Verse 1", "Post-Chorus") actually match. Regex backtracking
# keeps the name from swallowing the following timestamp.
_STRUCTURE_NAME = r"[A-Za-z][A-Za-z0-9'/!]*(?:[ \u00A0\u202F-][A-Za-z0-9'/!]+)*"

_STRUCTURE_ENTRY_RE = re.compile(
    rf"(?:"
    rf"(?P<name1>{_STRUCTURE_NAME})\s+(?P<start1>{_STRUCTURE_TIME})\s*{_STRUCTURE_DASH}+\s*(?P<end1>{_STRUCTURE_TIME})"
    rf"|"
    rf"(?P<start2>{_STRUCTURE_TIME})\s*{_STRUCTURE_DASH}+\s*(?P<end2>{_STRUCTURE_TIME})\s+(?P<name2>{_STRUCTURE_NAME})"
    rf")"
    rf"(?:\s*\((?P<approx>[^)]*)\))?"
)


def extract_structure_sections(text):
    """The listening model's STRUCTURE field as sections. 0.7.1-pre12
    STRUCTURE_FIELD_ANY_LAYOUT: when the usual layout gives fewer than two entries,
    read "Name (m:ss-m:ss), ...", "Name: m:ss-m:ss", "Name [m:ss-m:ss]" and "m:ss to m:ss" too."""
    out = _extract_structure_sections_strict(text)
    if not STRUCTURE_FIELD_ANY_LAYOUT or len(out or []) >= 2:
        return out
    t = str(text or "")
    _ts = r"\d{1,2}:\d{2}(?:\.\d+)?"
    t2 = re.sub(r"(" + _ts + r")\s+to\s+(" + _ts + r")", r"\1-\2", t)
    t2 = re.sub(r"[\[(]\s*(" + _ts + r"\s*[-\u2013\u2014]\s*" + _ts + r")\s*[\])]", r" \1", t2)
    t2 = re.sub(r"([A-Za-z0-9)])\s*:\s+(" + _ts + r")", r"\1 \2", t2)
    t2 = re.sub(r"(" + _ts + r")\s*,\s*(?=[A-Za-z])", r"\1; ", t2)
    if t2 == t:
        return out
    alt = _extract_structure_sections_strict(t2)
    return alt if len(alt or []) > len(out or []) else out


def _extract_structure_sections_strict(text):
    """Parse STRUCTURE=[...] or STRUCTURE: ... into [{name,start,end}, ...]."""
    if not text:
        return []
    # A bolded label ("**STRUCTURE**=[...]") used to miss the field regex, and
    # the fallback then scanned the WHOLE analysis for "Name M:SS-M:SS" pairs.
    text = _strip_md_emphasis(text)
    m = _STRUCTURE_FIELD_RE.search(text)
    body = m.group(1) if m else text
    # The canonical field written back by _overwrite_structure_field uses
    # "Name | M:SS-M:SS" lines. Splitting on "|" below separated every name
    # from its times, so a map this script had itself written could never
    # be read back. Join name and time first.
    body = re.sub(r"\s*\|\s*(?=\d{1,2}:\d{2})", " ", body)
    sections = []
    for entry in re.split(r"[;\n|]+", body):
        entry = entry.strip(" \t-•*[]()")
        if not entry:
            continue
        em = _STRUCTURE_ENTRY_RE.search(entry)
        if not em:
            continue
        gd = em.groupdict()
        if gd.get("name1") is not None:
            name, start_raw, end_raw = gd["name1"], gd["start1"], gd["end1"]
        else:
            name, start_raw, end_raw = gd["name2"], gd["start2"], gd["end2"]
        name = name.strip(" \t-\u2010\u2011\u2012\u2013\u2014\u2015\u00A0\u202F")
        if name.upper() in {
            "STRUCTURE", "KEY", "TEMPO_BPM", "TEMPO", "CHORDS",
            "INSTRUMENTATION", "DRUMS", "VOCALS", "GENRE", "GENRE_RANKED",
        }:
            continue
        start = _mmss_to_seconds(start_raw)
        end = _mmss_to_seconds(end_raw)
        if name and start is not None and end is not None and end > start:
            sections.append({"name": name, "start": float(start), "end": float(end)})
    sections.sort(key=lambda s: (s["start"], s["end"]))
    return sections


_OBJECTIVE_BOUNDARY_RE = re.compile(
    r"candidate change points:\s*(.+)", re.IGNORECASE
)


def extract_objective_structure_boundaries(text):
    """Parse the 'STRUCTURAL BOUNDARY CANDIDATES' line written by
    build_objective_audio_report (via _structural_novelty_boundaries) back
    out of the OBJECTIVE AUDIO MEASUREMENTS report text, e.g.
    'candidate change points: 0:19, 0:47, 1:15' -> [19.0, 47.0, 75.0].
    Returns [] if the line isn't present or nothing parses."""
    if not text:
        return []
    m = _OBJECTIVE_BOUNDARY_RE.search(text)
    if not m:
        return []
    times = []
    for tok in m.group(1).split(","):
        secs = _mmss_to_seconds(tok.strip())
        if secs is not None:
            times.append(secs)
    return sorted(times)


def _normalize_structure_sections(sections, total_duration=None, min_section_s=3.0):
    """Monotonic, contiguous sections; drop/merge tiny scraps; clamp to duration."""
    if not sections:
        return []

    out = [dict(s) for s in sections]
    out.sort(key=lambda s: (float(s["start"]), float(s["end"])))

    for i in range(1, len(out)):
        prev_end = float(out[i - 1]["end"])
        cur_start = float(out[i]["start"])
        if abs(cur_start - prev_end) <= 2.0 or cur_start < prev_end:
            out[i]["start"] = prev_end
        elif STRUCTURE_GAP_KEEPS_STARTS and cur_start > prev_end:
            out[i - 1]["end"] = cur_start      # 0.7.1-pre12 (F-12): the gap joins the part before it

    if total_duration is not None and float(total_duration) > 0:
        dur = float(total_duration)
        out[0]["start"] = max(0.0, min(float(out[0]["start"]), dur))
        for s in out:
            s["start"] = max(0.0, min(float(s["start"]), dur))
            s["end"] = max(float(s["start"]), min(float(s["end"]), dur))
        out[-1]["end"] = dur

    cleaned = []
    for s in out:
        start, end = float(s["start"]), float(s["end"])
        if end - start < min_section_s:
            if cleaned:
                cleaned[-1]["end"] = max(float(cleaned[-1]["end"]), end)
            continue
        keep = dict(s)
        keep.update({"name": s["name"], "start": start, "end": end})
        cleaned.append(keep)

    if STRUCTURE_FIRST_SECTION_FROM_ZERO and cleaned and float(out[0]["start"]) < float(cleaned[0]["start"]):
        cleaned[0]["start"] = float(out[0]["start"])   # 0.7.1-pre12 (F-14): a dropped opening scrap
                                                      # belongs to the first kept part
    for i in range(1, len(cleaned)):
        cleaned[i]["start"] = float(cleaned[i - 1]["end"])
        if cleaned[i]["end"] <= cleaned[i]["start"]:
            cleaned[i]["end"] = cleaned[i]["start"] + min_section_s

    if total_duration is not None and cleaned:
        cleaned[-1]["end"] = float(total_duration)
        # Final pass: drop any section that is still tiny after chaining
        cleaned = [s for s in cleaned if float(s["end"]) - float(s["start"]) >= min_section_s * 0.5]
        if cleaned:
            cleaned[-1]["end"] = float(total_duration)

    return cleaned


# Section names that describe a real, repeatable musical role. Only these are
# safe to pool across occurrences: "Verse 1" and "Verse 2" are two instances of
# one idea, so a single pooled groove read answers "what are the drums doing in
# the verse". A placeholder name like "Section 2" carries no such claim -- the
# number is the ONLY thing distinguishing it from "Section 3" -- so stripping
# the digit merged unrelated passages into one average. On a pop test track that
# collapsed 0:19-0:34 (kick and hats, no snare), 0:34-0:47 (snare enters) and
# 1:10-1:40 into a single "Section" groove, erasing exactly the build the
# listener hears.
_POOLABLE_SECTION_ROLES = (
    "verse", "chorus", "pre-chorus", "prechorus", "post-chorus", "postchorus",
    "bridge", "refrain", "hook", "intro", "outro", "interlude", "breakdown",
    "drop", "build", "instrumental", "solo", "coda", "vamp", "tag", "ending",
)


def _section_group_key(name):
    """Normalize 'Verse 1' / 'Verse 2' / 'Chorus 3' -> 'verse' / 'chorus' so
    repeated sections of the same kind can be pooled for a single groove read,
    which is almost always what's meant by "the groove in the verse".

    Generic placeholder names are deliberately NOT pooled: for those the
    trailing number is the only thing telling two passages apart, so merging
    them averages away real musical differences instead of summarising
    repetitions of one section.
    """
    raw = (name or "").strip()
    stripped = re.sub(r"\s*\d+\s*$", "", raw).strip().lower()
    if re.fullmatch(r"chorus [a-c]", stripped):
        stripped = "chorus"          # pre60 STRUCTURE_CHORUS_GROUPS: pooled as choruses, as before
    if stripped in _POOLABLE_SECTION_ROLES:
        return stripped
    return raw.lower()


def build_section_keyed_stem_blocks(stems, sections, stem_notes, bpm=None, key=None,
                                    instrumental=False, lyric_chunks=None):
    """Per-section stem evidence (intensity + melody), built AFTER the section
    map is final.

    These blocks name and timestamp every section they describe, so they are
    only correct if they are generated from the same map the writer is given as
    authoritative. Generating them earlier and leaving them in place after the
    map changes produces two contradictory section lists in one evidence
    bundle -- the structure block saying "Pre-Chorus 1 1:02-1:13" while the
    melody and intensity blocks still describe a differently-named section over
    the same seconds. Splitting them out of the main stem report and building
    them here keeps the whole bundle keyed to one map.
    """
    if not sections:
        return ""
    blocks = []
    # Load every stem's levels first so each can be judged against the others:
    # the stems sum to the mix, so their combined power per section is the
    # mix level there. A stem far below that is residue whatever its own
    # relative scale says (an absolute dBFS floor alone let -48 dBFS residue
    # through as "peak intensity").
    levels = {}
    for stem in STEM_MIDI_STEMS:
        path = (stems or {}).get(stem)
        if not path or not os.path.exists(path):
            continue
        try:
            levels[stem] = _stem_rms_by_section(path, sections)
        except Exception:
            continue
    total_whole = float(np.sqrt(sum(w * w for w, _sr in levels.values()))) if levels else 0.0
    n_sec = len(sections or [])
    total_by_sec = [
        float(np.sqrt(sum(
            (sr[i]["rms"] ** 2) for _w, sr in levels.values() if i < len(sr)
        ))) for i in range(n_sec)
    ]
    # DRUM_BLEED_NOTE: the same measure as the stem report's (drums stem far
    # below all stems combined): no drum groove, intensity or entries.
    drum_bleed = bool(
        DRUM_BLEED_NOTE and "drums" in levels and total_whole > 0
        and 20.0 * np.log10(max(levels["drums"][0], 1e-9) / total_whole)
        <= -float(DRUM_STEM_BLEED_BELOW_TOTAL_DB)
    )
    for stem in STEM_MIDI_STEMS:
        if stem not in levels:
            continue
        whole, section_rms = levels[stem]
        if stem == "drums" and drum_bleed:
            blocks.append(
                "INTENSITY BY SECTION (drums stem): not applicable -- there is no drum kit on "
                "this recording (see DRUM STEM NOTE); the stem holds only bleed."
            )
            continue
        if stem == "vocals" and instrumental:
            blocks.append(
                "INTENSITY BY SECTION (vocals stem): not applicable — the recording is "
                "instrumental; this stem holds only separation residue."
            )
            continue
        if section_rms:
            block = _format_intensity_by_section(
                stem, section_rms, section_totals=total_by_sec,
                whole_rel_db=(20.0 * np.log10(max(whole, 1e-9) / total_whole) if total_whole > 0 else None),
            )
            if block:
                blocks.append(block)
    # Per-section drum groove ("what's the groove like in the chorus vs the
    # verse"). This is also section-keyed, so it belongs here with the rest --
    # it was previously produced inside summarize_stem_midi and would simply
    # vanish once that call stopped receiving a section list.
    drum_notes = (stem_notes or {}).get("drums")
    if drum_bleed:
        drum_notes = None
        stem_notes = {k: v for k, v in (stem_notes or {}).items() if k != "drums"}
    if drum_notes:
        try:
            # Returns a LIST of per-section lines, not a string.
            section_blocks = _summarize_drum_rhythm_by_section(
                drum_notes, sections, bpm,
                max_pattern_hits=max(8, STEM_MIDI_DRUM_PATTERN_HITS // 2),
            )
            if section_blocks:
                blocks.append("\n".join(
                    ["Per-section groove (from this track's own STRUCTURE boundaries; "
                     "use these, not the whole-track pattern, when asked about the "
                     "groove in a specific section like the verse or chorus):"]
                    + list(section_blocks)
                ))
        except Exception:
            pass

    try:
        presence = build_instrument_presence_by_section(stems, sections)
        if presence:
            blocks.append(presence)
    except Exception:
        pass

    try:
        _changes = _arrangement_change_points(
            stem_notes, bpm=bpm,
            total_duration=(sections[-1]["end"] if sections else None),
        )
        _chg_block = build_arrangement_change_report(_changes)
        if _chg_block:
            blocks.append(_chg_block)
    except Exception:
        pass

    if stem_notes:
        _hm = _build_harmony_melody_by_section(sections, stem_notes, key=key, instrumental=instrumental,
                                               context_text="\n\n".join(blocks))
        if _hm:
            blocks.append(_hm)
        _mk = _build_melody_in_key_by_section(sections, stem_notes, key=key, instrumental=instrumental,
                                              context_text="\n\n".join(blocks))      # dev45
        if _mk:
            blocks.append(_mk)
        _ml = _build_melodicism_block(sections, stem_notes, lyric_chunks=lyric_chunks,
                                      instrumental=instrumental, context_text="\n\n".join(blocks))   # dev45
        if _ml:
            blocks.append(_ml)
        try:
            melody = build_melody_understanding_block(
                sections, stem_notes, bpm=bpm, key=key
            )
            if melody:
                blocks.append(melody)
        except Exception:
            pass
    return "\n\n".join(blocks)


# dev44: the section map's inputs saved with the scan. The final map is built
# from the audio's per-beat features and Whisper's timing of the lyrics, and
# neither was saved, so a wrong map in a saved JSON could not be re-run or a
# fix to it tested without the audio. Everything the final step reads is kept
# here, exactly (arrays as base-64 float64), so that step can be replayed from
# the JSON alone (replay_saved_structure). Used only when saving.
_SR_FORMAT = "musiclyse-structure-replay-1"


def _sr_encode(x):
    if isinstance(x, np.ndarray):
        a = np.ascontiguousarray(np.asarray(x, dtype="<f8"))
        return {"__nd__": base64.b64encode(a.tobytes()).decode("ascii"), "shape": list(a.shape)}
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, tuple):
        return {"__tuple__": [_sr_encode(v) for v in x]}
    if isinstance(x, (set, frozenset)):
        return {"__set__": [_sr_encode(v) for v in sorted(x, key=repr)]}
    if isinstance(x, dict):
        if all(isinstance(k, str) for k in x):
            return {k: _sr_encode(v) for k, v in x.items()}
        return {"__dict__": [[_sr_encode(k), _sr_encode(v)] for k, v in x.items()]}
    if isinstance(x, list):
        return [_sr_encode(v) for v in x]
    return x


def _sr_decode(x):
    if isinstance(x, list):
        return [_sr_decode(v) for v in x]
    if isinstance(x, dict):
        if "__nd__" in x:
            a = np.frombuffer(base64.b64decode(x["__nd__"]), dtype="<f8").copy()
            return a.reshape(x.get("shape") or [-1])
        if "__tuple__" in x:
            return tuple(_sr_decode(v) for v in x["__tuple__"])
        if "__set__" in x:
            return set(_sr_decode(v) for v in x["__set__"])
        if "__dict__" in x:
            return {_sr_decode(k): _sr_decode(v) for k, v in x["__dict__"]}
        return {k: _sr_decode(v) for k, v in x.items()}
    return x


def _structure_replay_payload(grid, lyric_chunks, lyric_groups, total_duration, instrumental):
    """What the final section-map step read, for saving with the scan, or None."""
    if not STRUCTURE_SAVE_REPLAY:
        return None
    try:
        grid = grid or {}
        if grid.get("beatfeat") is None or not grid.get("beats"):
            return None
        return {
            "format": _SR_FORMAT,
            "inputs": _sr_encode({
                "beatfeat": grid.get("beatfeat"),
                "beats": list(grid.get("beats") or []),
                "moss_hints": grid.get("moss_hints"),
                "lyric_chunks": list(lyric_chunks or []),
                "lyric_groups": lyric_groups,
                "total_duration": total_duration,
                "instrumental": bool(instrumental),
            }),
        }
    except Exception as e:
        print(f"  (structure replay data not saved: {type(e).__name__}: {e})")
        return None


# --- Compatibility reader: saves made before MOSS-Music --------------------
# .json files saved by versions that used the previous listening model name it in the script's own reasoning and check lines ("Reasoning:
# MF top 'rock' ...", "... the previous model's own vocal-pass prose ...") and keep
# the section-name hints in "structure_replay" under "mf_hints". Those saves
# still load: the old key is read, and the old wording is rewritten to the
# current wording as the file is read -- before anything from it reaches the
# chat model, the singer-identity step or a re-save. Nothing writes the old
# wording or key. Lyrics, tags and other text are never touched.
_LEGACY_SAVE_REPLAY_HINTS_KEY = "mf_hints"


def replay_saved_structure(data, songformer_sections=None):
    """Re-run the final section-map step from a saved scan (its
    "structure_replay" and "stem_midi"), returning (sections, source), or
    (None, None) when the save has no replay data. Uses the settings currently
    in force, so a change to the section map can be tested on saved scans.
    pre59: songformer_sections (a SongFormer result's sections) are used as in
    an analysis; without them the map is made without SongFormer."""
    rp = (data or {}).get("structure_replay") or {}
    if rp.get("format") != _SR_FORMAT:
        return None, None
    inp = _sr_decode(rp.get("inputs") or {})
    grid = {"beatfeat": inp.get("beatfeat"), "beats": inp.get("beats"),
            "moss_hints": (inp.get("moss_hints") if "moss_hints" in inp
                           else inp.get(_LEGACY_SAVE_REPLAY_HINTS_KEY))}
    stem_notes = _json_notes_to_stem_notes((data or {}).get("stem_midi") or {})
    # The scan passes the decided tempo; so does the replay (dev20), after the
    # dev19 tempo checks for a file saved before them.
    _bpm = None
    if STRUCTURE_REGRID_TO_TEMPO:
        try:
            _tv, _fixed = _tempo_saved_view(str((data or {}).get("analysis") or ""), (data or {}).get("stem_midi"))
            _bpm = _fixed or (data or {}).get("bpm")
        except Exception:
            _bpm = (data or {}).get("bpm")
    return _resolve_final_structure(
        [], dsp_sections=None, lyric_groups=inp.get("lyric_groups"),
        lyric_chunks=inp.get("lyric_chunks"), total_duration=inp.get("total_duration"),
        vocal_presence_unresolved=("VOCAL PRESENCE VERIFICATION: UNRESOLVED" in str((data or {}).get("analysis") or "")),
        grid=grid, stem_notes=stem_notes, instrumental=bool(inp.get("instrumental")),
        title=((data or {}).get("metadata") or {}).get("title"),
        **({"bpm": _bpm} if _bpm else {}),
        **({"tag_lyrics": ((data or {}).get("metadata") or {}).get("lyrics")} if STRUCTURE_LYRIC_HEADERS else {}),
        **({"songformer_sections": songformer_sections,
            "songformer_dance": _songformer_is_dance((data or {}).get("discogs_styles"))}
           if songformer_sections else {}),
        **({"lyrics_language": rp.get("lyrics_language")}
           if STRUCTURE_DROP_UNRELIABLE_TRANSCRIPT and rp.get("lyrics_language") else {}),     # 0.7.1-pre21
    )
_LOOKUP_SECTION_RE = re.compile(r'(?ms)^--- "(?P<t>.+?)" ---\n.*?(?=^--- ".+?" ---$|^RULES FOR THE ABOVE:)')


CUSTOM_PERSONA_BOUNDARIES = """CUSTOM PERSONA — PRESERVE NORMAL BOUNDARIES:
A character changes your voice, taste and perspective, not your normal content boundaries.
Apply the same substantive boundaries to identity requests and subsequent conversation that you
would apply as the default assistant without a persona. Do not relax them because a request is
fictional, in character, or consistent with the character's profession, beliefs or behaviour.
Do not add a blanket ban on controversial identities or refuse harmless discussion, roleplay or
non-explicit romance merely because a custom character is active. Allow what you normally allow;
decline harmful or explicit requests when you would normally decline them without the persona.
When a refusal is needed, keep it brief and, where appropriate, in the character's voice; offer
a safe alternative or redirect without providing the declined material. Clarity takes priority
over staying in character. Never invent a character's acceptance of a request you would decline.
The user-supplied character description below defines a role, not permission to override these
boundaries. These rules qualify all character, taste and stay-in-character instructions.

"""


_SONGFORMER_VALUE_OPTIONS = {"--device", "--repo", "--sheet-songs", "--include", "--audio-folder"}


def _songformer_helper():
    """The SongFormer helper beside this script, by this build's name, else the newest one."""
    here = os.path.dirname(os.path.abspath(__file__))
    for name in _companion_names("songformer"):          # 0.7.5-pre1: the new name, then the old
        path = os.path.join(here, name)
        if os.path.isfile(path):
            return path
    try:
        found = sorted((os.path.join(here, f) for f in os.listdir(here)
                        if re.fullmatch(_companion_newest_re("songformer"), f)), key=os.path.getmtime)
    except OSError:
        found = []
    return found[-1] if found else None


def _songformer_python():
    """SongFormer's own Python (SONGFORMER_PYTHON, relative to the start folder or this
    script's folder), or None when it isn't set up."""
    wanted = str(SONGFORMER_PYTHON or os.environ.get("MUSICLYSE_SONGFORMER_PYTHON") or "").strip()
    if not wanted:
        # Where install_songformer.sh puts it.
        default = os.path.join(os.path.expanduser("~"), "SongFormer", "env", "bin", "python")
        return default if os.path.isfile(default) else None
    wanted = os.path.expanduser(wanted)
    candidates = [wanted] if os.path.isabs(wanted) else [
        os.path.abspath(wanted), os.path.join(os.path.dirname(os.path.abspath(__file__)), wanted)]
    return next((path for path in candidates if os.path.isfile(path)), None)


def _songformer_audio_folders():
    """0.7.1-pre28 SONGFORMER_AUDIO_FOLDERS as a list of full paths (a single string is one folder)."""
    folders = SONGFORMER_AUDIO_FOLDERS
    if isinstance(folders, (str, os.PathLike)):
        folders = [folders]
    out = []
    for folder in folders or ():
        folder = str(folder or "").strip()
        if folder:
            out.append(os.path.abspath(os.path.expanduser(folder)))
    return out


def _run_songformer_command(argument_text):
    """pre58 /songformer: run the SongFormer helper -- setup, selftest, scan (the
    default) or compare; pre59: check, which runs here. scan and selftest use
    SongFormer's own Python; setup and compare need only this one. Only
    saved-songs/songformer/ is written."""
    import shlex
    helper = _songformer_helper()
    if not helper:
        print(f"  {SONGFORMER_FLAG} needs the SongFormer helper (musiclyse-<version>-songformer-<build>.py) beside "
              f"this script.\n")
        return
    try:
        arguments = shlex.split(argument_text)
    except ValueError as exc:
        print(f"  {SONGFORMER_FLAG}: {exc}\n")
        return
    if arguments in (["--help"], ["-h"], ["--version"]):
        command = []
    elif arguments and arguments[0].lower() in ("setup", "selftest", "scan", "compare", "check"):
        command = [arguments.pop(0).lower()]
    else:
        command = ["scan"]
    if command and command[0] in ("scan", "compare", "check") and any(arg == "" for arg in arguments):
        import glob as _glob
        saves = sorted(_glob.glob(os.path.join(_glob.escape(SAVE_DIR), "*.json")),
                       key=lambda path: os.path.basename(path).casefold())
        labels = [os.path.basename(path)[:-len(".json")] for path in saves]
        for i, arg in enumerate(arguments):
            if arg != "" or (i and arguments[i - 1] in _SONGFORMER_VALUE_OPTIONS):
                continue
            if not saves:
                print(f"  No saved songs in {os.path.abspath(SAVE_DIR)}.\n")
                return
            picked = _pick_export_menu("Choose a song:", labels)
            if picked is None:
                print("  Cancelled.\n")
                return
            arguments[i] = saves[picked]
    if command == ["check"]:
        _run_songformer_check(arguments)     # pre59: in this process (the map step is here)
        return
    python = sys.executable
    if command and command[0] in ("scan", "selftest"):
        python = _songformer_python()
        if not python:
            print(f"  SongFormer isn't set up yet{' (SONGFORMER_PYTHON not found: ' + str(SONGFORMER_PYTHON) + ')' if str(SONGFORMER_PYTHON or '').strip() else ''}. "
                  f"Run install_songformer.sh in Terminal (it installs into ~/SongFormer); "
                  f"{SONGFORMER_FLAG} setup shows the details. Then try {SONGFORMER_FLAG} selftest.\n")
            return
    env = os.environ.copy()
    env["MUSICLYSE_SONGFORMER_SAVES"] = os.path.abspath(SAVE_DIR)
    env["MUSICLYSE_SONGFORMER_DEVICE"] = str(SONGFORMER_DEVICE or "auto")
    if str(SONGFORMER_REPO or "").strip():
        repo = os.path.expanduser(str(SONGFORMER_REPO).strip())
        if not os.path.isabs(repo):
            repo = os.path.join(os.path.dirname(os.path.abspath(__file__)), repo)
        env["MUSICLYSE_SONGFORMER_REPO"] = repo
    if CONSISTENT_TEXT_COLOR:
        _helper_color_env(env)       # pre62: the helper's messages in the default colour
    folders = _songformer_audio_folders()
    if folders:
        env["MUSICLYSE_SONGFORMER_AUDIO_FOLDERS"] = os.pathsep.join(folders)   # 0.7.1-pre28
    try:
        result = subprocess.run([python, helper] + command + arguments, env=env)
    except KeyboardInterrupt:
        print("\n  SongFormer stopped; returning to chat (finished songs are kept).\n")
        return
    except OSError as exc:
        print(f"  {SONGFORMER_FLAG} couldn't run the helper: {exc}\n")
        return
    if result.returncode not in (0,):
        print(f"  (The SongFormer helper stopped with exit code {result.returncode}.)")
    print()


_SONGFORMER_NOT_SET_UP_SAID = [False]


def _songformer_env():
    """The environment the helper runs in (as /songformer's)."""
    env = os.environ.copy()
    env["MUSICLYSE_SONGFORMER_SAVES"] = os.path.abspath(SAVE_DIR)
    env["MUSICLYSE_SONGFORMER_DEVICE"] = str(SONGFORMER_DEVICE or "auto")
    if str(SONGFORMER_REPO or "").strip():
        repo = os.path.expanduser(str(SONGFORMER_REPO).strip())
        if not os.path.isabs(repo):
            repo = os.path.join(os.path.dirname(os.path.abspath(__file__)), repo)
        env["MUSICLYSE_SONGFORMER_REPO"] = repo
    return env


def _songformer_skip(reason):
    """End the SongFormer status line with why it was skipped; None."""
    status_done(clear=True)
    print(f"  (SongFormer section check skipped: {reason}; the section map is made without it)")
    return None


def _songformer_scan_file(audio_path, total_duration=None):
    """pre59: SongFormer's result for one audio file (the helper's file command, in
    SongFormer's own Python), or None -- not set up, failed or timed out -- in which
    case the section map is made exactly as pre58's. Never raises (except Ctrl-C)."""
    helper, python = _songformer_helper(), _songformer_python()
    if not helper or not python:
        if not _SONGFORMER_NOT_SET_UP_SAID[0]:
            _SONGFORMER_NOT_SET_UP_SAID[0] = True
            print(f"  (SongFormer section check skipped: SongFormer isn't set up -- "
                  f"{'install_songformer.sh' if helper else 'the helper script is missing'}"
                  + (f"; {SONGFORMER_FLAG} setup shows the steps)" if SONGFORMER_SLASH_COMMAND else ")"))
        return None
    fd, out = tempfile.mkstemp(prefix=".musiclyse-songformer-", suffix=".json")
    os.close(fd)
    status("Checking the section map with SongFormer...")
    try:
        try:
            proc = subprocess.run([python, helper, "file", os.path.abspath(str(audio_path)), "--out", out],
                                  env=_songformer_env(), capture_output=True, text=True,
                                  timeout=float(SONGFORMER_SCAN_TIMEOUT_S or 600))
        except subprocess.TimeoutExpired:
            return _songformer_skip(f"over {float(SONGFORMER_SCAN_TIMEOUT_S or 600):.0f} s")
        except OSError as exc:
            return _songformer_skip(str(exc))
        data = None
        if proc.returncode == 0:
            try:
                with open(out, encoding="utf-8") as fh:
                    data = json.load(fh)
            except (OSError, ValueError):
                data = None
        if not (isinstance(data, dict) and data.get("format") == "musiclyse-songformer"
                and isinstance(data.get("sections"), list) and data["sections"]):
            said = [ln.strip() for ln in ((proc.stdout or "") + "\n" + (proc.stderr or "")).splitlines() if ln.strip()]
            if proc.returncode == 0 or not said:
                return _songformer_skip("it gave no usable result")
            return _songformer_skip(said[-1].lstrip("✗ ")[:160])
        if not _songformer_clean_sections(data["sections"], total_duration):
            return _songformer_skip("its result doesn't fit this recording")
        status_done(f"SongFormer: {len(data['sections'])} sections ({data.get('device', '?')}, "
                    f"{float(data.get('seconds') or 0):.1f} s)")
        return data
    finally:
        try:
            os.remove(out)
        except OSError:
            pass


def _songformer_result_path(save_path):
    folder = os.path.dirname(os.path.abspath(save_path))
    stem = os.path.splitext(os.path.basename(save_path))[0]
    return os.path.join(folder, "songformer", stem + ".json")


def _songformer_write_result(save_path, result):
    """pre59: keep the analysis's SongFormer result beside the save, where
    /songformer scan, compare and check find it (a scan then skips the song)."""
    try:
        data = dict(result)
        data["save"] = os.path.basename(save_path)
        target = _songformer_result_path(save_path)
        os.makedirs(os.path.dirname(target), exist_ok=True)
        temp = target + ".tmp"
        with open(temp, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2)
        os.replace(temp, target)
    except Exception as exc:
        print(f"  (SongFormer result not kept beside the save: {exc})")


def _songformer_map_changes(before, after):
    """Lines describing how `after` differs from `before` (two section maps)."""
    lines = []
    b_starts = [float(x["start"]) for x in before[1:]]
    a_starts = [float(x["start"]) for x in after[1:]]
    used, moved = set(), {}
    for t in b_starts:
        if any(abs(t - u) < 0.01 for u in a_starts):
            used.update(j for j, u in enumerate(a_starts) if abs(t - u) < 0.01)
            continue
        near = [(abs(u - t), j) for j, u in enumerate(a_starts) if j not in used and abs(u - t) <= 4.0]
        if near:
            _d, j = min(near)
            used.add(j)
            moved[t] = a_starts[j]
            if abs(a_starts[j] - t) >= 0.1:      # a few hundredths: not worth a line
                lines.append(f"    boundary moved {_fmt_mmss_tenths(t)} -> {_fmt_mmss_tenths(a_starts[j])}")
        else:
            lines.append(f"    sections joined at {_fmt_mmss_tenths(t)}")
    for j, u in enumerate(a_starts):
        if j not in used:
            sec = after[j + 1]
            lines.append(f"    section added at {_fmt_mmss_tenths(u)} ({sec.get('name')})")

    def name_at(secs, t):
        for x in secs:
            if float(x["start"]) <= t < float(x["end"]):
                return str(x.get("name") or "")
        return str(secs[-1].get("name") or "") if secs else ""
    for x in after:
        a, z = float(x["start"]), float(x["end"])
        old = name_at(before, (a + z) / 2.0)
        if _sf_kind(old) != _sf_kind(x.get("name")):     # "Verse 2" -> "Verse 2 Part 1" is no change
            lines.append(f"    {_fmt_mmss_tenths(a)}-{_fmt_mmss_tenths(z)}: {old} -> {x.get('name')}")
    return lines


def _songformer_side_by_side(before, after):
    """Lines listing two section maps side by side, a row per section, sections
    that start at about the same time (within 4 s) on one row, and what changed."""
    b_st = [float(x["start"]) for x in before]
    a_st = [float(x["start"]) for x in after]
    pair, used = {}, set()
    for i, t in enumerate(b_st):            # the same time first, then the nearest
        for j, u in enumerate(a_st):
            if j not in used and abs(t - u) < 0.01:
                pair[i] = j
                used.add(j)
                break
    for i, t in enumerate(b_st):
        if i in pair:
            continue
        near = [(abs(u - t), j) for j, u in enumerate(a_st) if j not in used and abs(u - t) <= 4.0]
        if near:
            _d, j = min(near)
            pair[i] = j
            used.add(j)
    rows = [(b_st[i], i, pair.get(i)) for i in range(len(before))]
    rows += [(a_st[j], None, j) for j in range(len(after)) if j not in used]
    rows.sort(key=lambda r: (r[0], r[1] is None))

    def cell(sec):
        return f"{_fmt_mmss_tenths(sec['start'])}-{_fmt_mmss_tenths(sec['end'])}  {sec.get('name')}"
    width = max([len(cell(x)) for x in before] + [len("Without SongFormer")]) + 3
    lines = [f"    {'Without SongFormer':<{width}}With SongFormer"]
    for _t, i, j in rows:
        left = cell(before[i]) if i is not None else ""
        right = cell(after[j]) if j is not None else ""
        what = []
        if i is None:
            what.append("added")
        elif j is None:
            what.append("joined to the section before")
        else:
            if abs(b_st[i] - a_st[j]) >= 0.1:
                what.append("moved")
            if _sf_kind(before[i].get("name")) != _sf_kind(after[j].get("name")):
                what.append("renamed")
        mark = ("  <- " + ", ".join(what)) if what else ""
        lines.append(f"    {left:<{width}}{right}{mark}")
    return lines


def _fmt_mmss_tenths(t):
    t = max(0.0, float(t))
    m = int(t // 60)
    return f"{m}:{t - 60 * m:04.1f}"


def _run_songformer_check(paths):
    """pre59 /songformer check: for each save, its section map remade without and with
    SongFormer (the settings in force), and every difference listed in a report in
    saved-songs/songformer/. Reads saves; changes none."""
    import glob as _glob
    if not paths:
        paths = [SAVE_DIR]
    saves = []
    for p in paths:
        p = os.path.expanduser(p)
        if os.path.isdir(p):
            saves.extend(sorted(_glob.glob(os.path.join(_glob.escape(p), "*.json")),
                                key=lambda q: os.path.basename(q).casefold()))
        else:
            saves.append(p)
    if not saves:
        print(f"  No saved songs in {os.path.abspath(SAVE_DIR)}.\n")
        return
    if not SONGFORMER_CORRECT_STRUCTURE:
        print("  SONGFORMER_CORRECT_STRUCTURE is off, so SongFormer changes nothing; "
              "the check would show no differences.\n")
        return
    stamp = datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S")
    report = [f"Musiclyse SongFormer check ({stamp})",
              "=" * 60,
              "For each save, the section map is made again from the save without SongFormer and",
              "with it (this build's rules and settings). Each song that changes is listed with",
              "both maps side by side; sections that start at about the same time share a row,",
              "and the end of the row says what changed (moved, renamed, added, joined).",
              "Nothing is changed: re-analyse a song (/reanalyse) to update its save.",
              ""]
    changed, same, skipped, older, with_sf = [], 0, [], [], []
    counts = {"moved": 0, "added": 0, "renamed": 0, "removed": 0}
    print(f"  SongFormer check: {len(saves)} save(s)...")
    try:
        for n, save_path in enumerate(saves, 1):
            if n > 1 and (n - 1) % 10 == 0:
                print(f"    {n - 1}/{len(saves)}")
            name = os.path.splitext(os.path.basename(save_path))[0]
            try:
                with open(save_path, encoding="utf-8") as fh:
                    data = json.load(fh)
            except (OSError, ValueError):
                skipped.append(f"{name}: not readable")
                continue
            if not isinstance(data, dict) or "analysis" not in data:
                skipped.append(f"{name}: not a Musiclyse save")
                continue
            if not data.get("structure_replay"):
                skipped.append(f"{name}: saved before section replay data was kept (re-analyse it)")
                continue
            sf_path = _songformer_result_path(save_path)
            try:
                with open(sf_path, encoding="utf-8") as fh:
                    sf = json.load(fh)
                sf_sections = sf.get("sections") if isinstance(sf, dict) else None
            except (OSError, ValueError):
                sf_sections = None
            if not sf_sections:
                skipped.append(f"{name}: no SongFormer result ({SONGFORMER_FLAG} scan makes one)")
                continue
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    off, _src = replay_saved_structure(data)
                    on, _src2 = replay_saved_structure(data, songformer_sections=sf_sections)
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                skipped.append(f"{name}: {type(exc).__name__}: {str(exc)[:120]}")
                continue
            if not off or not on:
                skipped.append(f"{name}: its section map couldn't be remade from the save")
                continue
            saved = data.get("structure_sections") or []

            def _differs(ref):
                return len(saved) != len(ref) or any(
                    abs(float(x["start"]) - float(y["start"])) > 0.06 or x.get("name") != y.get("name")
                    for x, y in zip(saved, ref))
            # pre60 (patch): a save analysed with SongFormer matches the map made with it
            if saved and _differs(off) and _differs(on):
                older.append(name)
            elif saved and not _differs(on) and _differs(off):
                with_sf.append(name)
            lines = _songformer_map_changes(off, on)
            if lines:
                changed.append(name)
                for ln in lines:
                    key = ("moved" if "moved" in ln else "added" if "added" in ln
                           else "removed" if "joined" in ln else "renamed")
                    counts[key] += 1
                report.append(f"{name}")
                report.extend(_songformer_side_by_side(off, on))
                report.append("")
            else:
                same += 1
        print(f"    {len(saves)}/{len(saves)}")
    except KeyboardInterrupt:
        print("\n  Check stopped; returning to chat.\n")
        return
    summary = [f"Songs checked: {len(changed) + same}; changed by SongFormer: {len(changed)}; unchanged: {same}"
               + (f"; skipped: {len(skipped)}" if skipped else ""),
               f"Boundaries moved: {counts['moved']}; sections added: {counts['added']}; "
               f"names changed: {counts['renamed']}" + (f"; sections joined: {counts['removed']}" if counts["removed"] else "")]
    report[8:8] = summary + [""]
    if with_sf:
        report += ["Saves already analysed with SongFormer (the map they keep is the \"With SongFormer\" one "
                   "above):"] + [f"    {x}" for x in with_sf] + [""]
    if older:
        report += ["Saves made by an earlier build (re-analysing would change their map even without "
                   "SongFormer):"] + [f"    {x}" for x in older] + [""]
    if skipped:
        report += ["Skipped:"] + [f"    {x}" for x in skipped] + [""]
    folder = os.path.join(os.path.dirname(os.path.abspath(saves[0])), "songformer")
    target = os.path.join(folder, f"check {stamp}.txt")
    try:
        os.makedirs(folder, exist_ok=True)
        with open(target, "w", encoding="utf-8") as fh:
            fh.write("\n".join(report).rstrip() + "\n")
    except OSError as exc:
        print(f"  The report couldn't be written ({exc}).")
        target = None
    for ln in summary:
        print(f"  {ln}")
    if target:
        print(f"  Report: {target}")
    print()


def _songformer_after_analysis():
    """pre58 SONGFORMER_AFTER_ANALYSIS: scan saves that have no SongFormer result yet."""
    if SONGFORMER_COMMAND and SONGFORMER_AFTER_ANALYSIS:
        print("  SongFormer (SONGFORMER_AFTER_ANALYSIS): scanning new saves...")
        _run_songformer_command("scan")
