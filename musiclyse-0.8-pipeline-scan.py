# MUSICLYSE FILE pipeline-scan 0.8
# ==============================================================================
#  MUSICLYSE 0.8: THE ANALYSIS RUN
# ==============================================================================
# This file holds the analysis run: /analyse, /reanalyse and /batch, the order of the stages of
# a scan, music videos, and the song's saved .json. It is part of musiclyse-0.8.py, which loads
# it whole at start-up, after its own code, into its own namespace (see "THE MUSICLYSE FILES"
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


@contextlib.contextmanager
def _analysis_audio_source(path):
    """Yield an audio-only source while retaining the original path for saves."""
    if not _is_video_input(path):
        yield path
        return
    with tempfile.TemporaryDirectory(prefix="musiclyse-video-audio-") as folder:
        wav = os.path.join(folder, "audio.wav")
        result = subprocess.run(
            ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
             "-i", path, "-map", "0:a:0", "-vn", "-c:a", "pcm_f32le", "-rf64", "auto", wav],
            capture_output=True, text=True, stdin=subprocess.DEVNULL,
        )
        if result.returncode:
            message = result.stderr.strip()
            if "matches no streams" in message:
                raise RuntimeError(f"No audio stream found in video: {path}")
            raise RuntimeError(f"Could not extract audio from video {path}: {message}")
        if not os.path.isfile(wav):
            raise RuntimeError(f"No audio extracted from video: {path}")
        yield wav


def _audio_file_signature(path):
    """Stable identity for a local recording used by all interactive caches.

    A path is not a recording identity: files can be replaced in-place while
    retaining the same pathname. Size + nanosecond mtime cheaply detects the
    common replacement/edit case without hashing every audio file on every turn.
    URLs are treated as their URL identity because they have no local stat.
    """
    if path is None:
        return None
    path = str(path)
    if path.startswith(("http://", "https://")):
        return ("url", path)
    try:
        real = os.path.abspath(os.path.expanduser(path))
        st = os.stat(real)
        return ("file", real, int(st.st_size), int(st.st_mtime_ns))
    except OSError:
        return None


def _process_rss_mb():
    """Best-effort current process RSS in MiB (for batch logging)."""
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    except Exception:
        pass
    try:
        import resource
        # ru_maxrss is bytes on macOS, KiB on Linux
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform == "darwin":
            return rss / (1024 * 1024)
        return rss / 1024.0
    except Exception:
        return None


def _log_batch_rss(prefix="RSS"):
    if not BATCH_LOG_RSS:
        return
    mb = _process_rss_mb()
    if mb is not None:
        print(f"  ({prefix}: ~{mb:.0f} MiB)")


def _cleanup_track_temp_files(track_path, audio_temp_files, dsp_temp_files, stem_temp_files, demucs_out_dirs):
    """Delete per-track temp WAVs and Demucs output dirs so batch runs do not
    keep every intermediate file (and any memory-mapped data) until the end."""
    for store in (audio_temp_files, dsp_temp_files, stem_temp_files):
        temp_path = store.pop(track_path, None)
        if not temp_path:
            continue
        try:
            if os.path.isfile(temp_path) and os.path.abspath(temp_path) != os.path.abspath(track_path):
                os.remove(temp_path)
        except Exception:
            pass

    # demucs_out_dirs is a shared list; remove and delete everything currently
    # recorded (each track appends its own dir during analysis).
    while demucs_out_dirs:
        d = demucs_out_dirs.pop()
        try:
            shutil.rmtree(d, ignore_errors=True)
        except Exception:
            pass

    # Normalised-audio retry copies from the listening-model main pass.
    while _MOSS_RETRY_TEMP_FILES:
        try:
            os.remove(_MOSS_RETRY_TEMP_FILES.pop())
        except Exception:
            pass

    # Omnizart intermediate dirs, if any were tracked globally
    try:
        while _OMNIZART_OUTPUT_DIRS:
            d = _OMNIZART_OUTPUT_DIRS.pop()
            try:
                shutil.rmtree(d, ignore_errors=True)
            except Exception:
                pass
    except Exception:
        pass


_CHECKS_BLOCK_HEAD = "CHECKS (corrections to read before the report below):"
_CHECK_PARA_RE = re.compile(r"[A-Z][A-Z /-]+ CHECK \(")


def _saved_analysis_fixed_blocks(analysis, has_cover=False):
    """dev67: see REPORT_FIXED_BLOCKS."""
    if not REPORT_FIXED_BLOCKS:
        return analysis
    try:
        t = str(analysis or "")
        if not t:
            return analysis
        t = _fixed_blocks_production_tags(t)
        t = _fixed_blocks_cover(t, has_cover)
        t = _fixed_blocks_structure_copy(t)
        return t
    except Exception:
        return analysis


def _saved_analysis_fixed_layout(analysis, track_key=None, structure_sections=None):
    """dev64: see REPORT_FIXED_LAYOUT. The listening model's report at the top
    of the analysis in the fixed layout, with the measured section map in
    STRUCTURE=. Correction notes already on top are kept on top; an analysis
    already in the layout (it has REPORT_NOTE=) is left as it is."""
    if not REPORT_FIXED_LAYOUT:
        return analysis
    try:
        t = str(analysis or "")
        if not t:
            return analysis
        lead = ""
        if t.startswith(_CHECKS_BLOCK_HEAD):
            k = t.find("\n\n")
            lead, t = t[:k + 2], t[k + 2:]
        while _CHECK_PARA_RE.match(t):
            k = t.find("\n\n")
            if k < 0:
                break
            lead, t = lead + t[:k + 2], t[k + 2:]
        if re.search(r"(?m)^REPORT_NOTE=", t[:_moss_report_head_end(t)]):
            if not REPORT_FILL_FROM_EVIDENCE:
                return analysis
            t2 = _era_drum_evidence_check(_saved_report_tidy_then_fill(
                _report_fill_from_evidence(t, structure_given=bool(structure_sections)), structure_sections))
            return analysis if t2 == t else lead + t2
        first = _MOSS_FIRST_PASS_BY_TRACK.get(str(track_key), "") if track_key is not None else ""
        out = _moss_canonical_report(t, first)
        if structure_sections:
            cut = _moss_report_head_end(out)
            head, tail = out[:cut], out[cut:]
            head = re.sub(r"(?ms)^STRUCTURE=.*?(?=^[A-Z][A-Z0-9_]{2,}=)",
                          lambda m_: _structure_field_text(structure_sections) + "\n", head, count=1)
            out = head + tail
        out = _report_fill_from_evidence(out, structure_given=bool(structure_sections))
        out = _saved_report_tidy_then_fill(out, structure_sections)    # dev73
        out = _era_drum_evidence_check(out)
        return lead + out
    except Exception:
        return analysis


def _saved_analysis_with_checks_block(analysis):
    return _normalize_unresolved_vocal_report(_saved_analysis_with_checks_block_impl(analysis))


def _saved_analysis_with_checks_block_impl(analysis):
    """dev63: with MOSS_CONSISTENT_REPORT, every saved analysis opens the same
    way: a CHECKS block holding the GENRE / INSTRUMENT CHECK lines written
    above the report (or "none"), then the report. Unchanged otherwise."""
    if not (MOSS_CONSISTENT_REPORT or REPORT_FIXED_LAYOUT):
        return analysis
    t = str(analysis or "")
    if not t or t.startswith(_CHECKS_BLOCK_HEAD):
        return analysis
    paras = []
    while _CHECK_PARA_RE.match(t):
        k = t.find("\n\n")
        if k < 0:
            break
        paras.append(t[:k].strip())
        t = t[k + 2:]
    body = "\n".join("- " + p for p in paras) if paras else "none"
    return _CHECKS_BLOCK_HEAD + "\n" + body + "\n\n" + t


# --- Save/load helpers ------------------------------------------------------
# Characters that must never appear in a saved basename (path separators,
# Windows-reserved, control chars). Spaces, apostrophes, parentheses, dashes,
# unicode letters, etc. are intentionally kept so "/save" and "/load" work with
# natural names like "Artist Name - Song's Title (Live).json".
_SAVED_NAME_FORBIDDEN = re.compile(r'[/\\:\*\?"<>\|\x00-\x1f]')


def _sanitize_saved_name(name):
    """Keep spaces and most printable characters; only strip path-hostile chars.

    Previously this collapsed everything outside [A-Za-z0-9._-] to underscores,
    which made it impossible to save or load names containing spaces or common
    punctuation. We now preserve the user's intended filename as closely as
    the filesystem allows.
    """
    name = (name or "").strip().strip('"').strip("'")
    if not name:
        return ""

    base = os.path.basename(name)
    # Normalise runs of whitespace to a single space; drop path-hostile chars.
    safe = _SAVED_NAME_FORBIDDEN.sub("_", base)
    safe = re.sub(r"\s+", " ", safe).strip()
    if not safe:
        return ""
    # Keep a clean ".json" suffix when present (avoid "Name .json").
    if safe.lower().endswith(".json"):
        stem = safe[:-5].rstrip(" .")
        safe = (stem if stem else "song") + ".json"
    else:
        safe = safe.rstrip(" .")
    if not safe:
        return ""
    if len(safe) > 200:
        # Prefer keeping the extension when truncating.
        if safe.lower().endswith(".json"):
            stem = safe[:-5][:195].rstrip(" .")
            safe = (stem if stem else "song") + ".json"
        else:
            safe = safe[:200].rstrip(" .")
    return safe


def _metadata_artist_title(metadata, audio_path=None):
    """Return (artist, title) for batch/save naming from tags or filename."""
    artist = ""
    title = ""
    if metadata:
        artist = str(metadata.get("artist") or "").strip()
        title = str(metadata.get("title") or "").strip()

    if not title and audio_path:
        base = os.path.basename(audio_path or "")
        stem, _ext = os.path.splitext(base)
        title = stem.strip() if stem.strip() else ""

    if not artist:
        artist = "Unknown"
    if not title:
        title = f"song_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    return artist, title


def _batch_save_basename(audio_path, metadata=None):
    """Derive a .json basename as 'Artist Name - Song Name.json'.

    Uses file-tag artist/title when available. Missing artist → 'Unknown';
    missing title → original audio filename stem. Spaces and normal
    punctuation are kept (only path-hostile characters are replaced).

    e.g. tags Artist='Artist', Title='Title' → 'Artist - Title.json'
         no tags, file '01 Song Title.m4a' → 'Unknown - 01 Song Title.json'
    """
    artist, title = _metadata_artist_title(metadata, audio_path)
    # Keep video inputs distinct in save names while metadata uses the song title.
    qualifier = _video_filename_qualifier(audio_path)
    if qualifier and not re.search(r"\((?:music\s+)?video\)\s*$", title, re.I):
        title += " " + qualifier

    def _part(s):
        s = _SAVED_NAME_FORBIDDEN.sub("_", str(s or ""))
        s = re.sub(r"\s+", " ", s).strip(" .")
        return s or "Unknown"

    stem = f"{_part(artist)} - {_part(title)}"
    if not stem.strip(" -"):
        stem = f"song_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if len(stem) > 180:
        stem = stem[:180].rstrip(" .")
    return stem + ".json"


def _saved_json_source_fields__pre6(json_path):
    """Read a complete, usable save before considering it safe to skip.

    A readable JSON header does not prove that the rest of a save survived a
    crash. Never treat truncated JSON or an empty analysis as a finished scan.
    """
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict) or not isinstance(data.get("analysis"), str) or not data["analysis"].strip():
            return None
        return {k: data.get(k) for k in ("track_path", "audio_file_signature")}
    except (OSError, ValueError, UnicodeError):
        return None


def _saved_json_source_fields__07(json_path):
    """{"track_path": ..., "audio_file_signature": ...} as recorded in a saved
    JSON (either may be missing / None), or None when the file cannot be read.
    Reads only the head when it can: both fields sit at the top and saved
    analyses run to several MB."""
    try:
        with open(json_path, "r", encoding="utf-8", errors="replace") as f:
            head = f.read(65536)
        dec = json.JSONDecoder()
        out = {}
        found = False
        for key in ("track_path", "audio_file_signature"):
            m = re.search(r'"' + key + r'"\s*:\s*', head)
            if not m:
                continue
            try:
                out[key], _ = dec.raw_decode(head, m.end())
                found = found or key == "track_path"
            except ValueError:
                pass
        if not found:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f) or {}
            out = {k: data.get(k) for k in ("track_path", "audio_file_signature")}
        return out
    except Exception:
        return None


def _saved_json_source_fields(*args, **kwargs):
    """BATCH_SAVE_CHECKS picks the 0.7.1-pre6 version; False gives 0.7's."""
    return (_saved_json_source_fields__pre6 if BATCH_SAVE_CHECKS else _saved_json_source_fields__07)(*args, **kwargs)


def _nfc_fold(name):
    return unicodedata.normalize("NFC", str(name or "")).casefold()


def _batch_resolve_save_name__pre6(audio_path, metadata=None):
    """Pick the saved-songs/ name for a batch track: (json_name, already_saved).

    "Artist - Title.json" alone is not unique -- two files with the same tags
    (a FLAC and a WAV of one song, or a different master) used to share it, so
    /batch skipped the second as "already saved" (or, with skipping off,
    overwrote the first). Each existing "Artist - Title.json",
    "Artist - Title (2).json", ... is checked against the source file it
    records, and a save counts as ANOTHER file's only when that source still
    exists and is a different recording (a different file that is not an
    untouched copy of it). Otherwise the most specific match wins:
      1. the recorded source is this very file;
      2. same file name, size and modification time (library moved or copied);
      3. the source has gone and had this file's name (folder moved);
      4. the source has gone and had this file's size + mtime (renamed);
    Unknown, unreadable and unrelated saves are preserved under their own names.
    With no source match the next free " (N)" name is used. A matched name is
    reused for reanalysis when its recorded audio signature has changed. Valid
    legacy saves without a signature retain the source-path/name fallback.
    """
    base = _batch_save_basename(audio_path, metadata=metadata)
    stem = base[:-5]
    this = os.path.abspath(os.path.expanduser(str(audio_path or "")))
    this_name = _nfc_fold(os.path.basename(this))
    try:
        st = os.stat(this)
        this_sig = (int(st.st_size), int(st.st_mtime_ns))
    except OSError:
        this_sig = None

    names = [base]
    try:
        stem_n = unicodedata.normalize("NFC", stem)
        pat = re.compile(r"^" + re.escape(stem_n) + r" \((\d+)\)\.json$", re.IGNORECASE)
        numbered = []
        for fn in os.listdir(SAVE_DIR):
            mm = pat.match(unicodedata.normalize("NFC", fn))
            if mm:
                numbered.append((int(mm.group(1)), fn))
        names += [fn for _n, fn in sorted(numbered)]
    except OSError:
        pass

    best = None   # (tier, order, name, unchanged)
    for order, name in enumerate(names):
        path = os.path.join(SAVE_DIR, name)
        if not os.path.exists(path):
            continue
        fields = _saved_json_source_fields(path)
        src = (fields or {}).get("track_path")
        tier = None
        stored_sig = None
        if not src:
            continue                           # preserve unknown or damaged saves
        elif str(src).startswith(("http://", "https://")):
            tier = None                        # made from a URL, not this file
        else:
            stored = os.path.abspath(os.path.expanduser(str(src)))
            sig = (fields or {}).get("audio_file_signature")
            try:
                stored_sig = (int(sig[2]), int(sig[3])) if sig else None
            except Exception:
                stored_sig = None
            same_name = _nfc_fold(os.path.basename(stored)) == this_name
            same_sig = this_sig is not None and stored_sig == this_sig
            if os.path.exists(stored):
                try:
                    same = os.path.samefile(stored, this)
                except OSError:
                    same = stored == this
                if same:
                    tier = 1
                elif same_name and same_sig:
                    tier = 2                   # a copy of the library (original kept)
                else:
                    tier = None                # another file's save
            elif same_name and same_sig:
                tier = 2                       # moved, untouched
            elif same_name:
                tier = 3                       # moved
            elif same_sig:
                tier = 4                       # renamed
        if tier is not None and (best is None or (tier, order) < best[:2]):
            unchanged = this_sig is not None and (sig is None or stored_sig == this_sig)
            best = (tier, order, name, unchanged)
    if best is not None:
        return best[2], best[3]

    if not os.path.exists(os.path.join(SAVE_DIR, base)):
        return base, False
    n = 2
    while os.path.exists(os.path.join(SAVE_DIR, f"{stem} ({n}).json")):
        n += 1
    return f"{stem} ({n}).json", False


def _batch_resolve_save_name__07(audio_path, metadata=None):
    """Pick the saved-songs/ name for a batch track: (json_name, already_saved).

    "Artist - Title.json" alone is not unique -- two files with the same tags
    (a FLAC and a WAV of one song, or a different master) used to share it, so
    /batch skipped the second as "already saved" (or, with skipping off,
    overwrote the first). Each existing "Artist - Title.json",
    "Artist - Title (2).json", ... is checked against the source file it
    records, and a save counts as ANOTHER file's only when that source still
    exists and is a different recording (a different file that is not an
    untouched copy of it). Otherwise the most specific match wins:
      1. the recorded source is this very file;
      2. same file name, size and modification time (library moved or copied);
      3. the source has gone and had this file's name (folder moved);
      4. the source has gone and had this file's size + mtime (renamed);
      5. the plain name records no source or cannot be read (older versions);
      6. the source has gone, whatever it was -- exactly the old behaviour.
    With no match the next free " (N)" name is used.
    """
    base = _batch_save_basename(audio_path, metadata=metadata)
    stem = base[:-5]
    this = os.path.abspath(os.path.expanduser(str(audio_path or "")))
    this_name = _nfc_fold(os.path.basename(this))
    try:
        st = os.stat(this)
        this_sig = (int(st.st_size), int(st.st_mtime_ns))
    except OSError:
        this_sig = None

    names = [base]
    try:
        stem_n = unicodedata.normalize("NFC", stem)
        pat = re.compile(r"^" + re.escape(stem_n) + r" \((\d+)\)\.json$", re.IGNORECASE)
        numbered = []
        for fn in os.listdir(SAVE_DIR):
            mm = pat.match(unicodedata.normalize("NFC", fn))
            if mm:
                numbered.append((int(mm.group(1)), fn))
        names += [fn for _n, fn in sorted(numbered)]
    except OSError:
        pass

    best = None   # (tier, order, name)
    for order, name in enumerate(names):
        path = os.path.join(SAVE_DIR, name)
        if not os.path.exists(path):
            continue
        fields = _saved_json_source_fields(path)
        src = (fields or {}).get("track_path")
        tier = None
        if not src:
            if order == 0:
                tier = 5                       # older save (no source) / unreadable
        elif str(src).startswith(("http://", "https://")):
            tier = None                        # made from a URL, not this file
        else:
            stored = os.path.abspath(os.path.expanduser(str(src)))
            sig = (fields or {}).get("audio_file_signature")
            try:
                stored_sig = (int(sig[2]), int(sig[3])) if sig else None
            except Exception:
                stored_sig = None
            same_name = _nfc_fold(os.path.basename(stored)) == this_name
            same_sig = this_sig is not None and stored_sig == this_sig
            if os.path.exists(stored):
                try:
                    same = os.path.samefile(stored, this)
                except OSError:
                    same = stored == this
                if same:
                    tier = 1
                elif same_name and same_sig:
                    tier = 2                   # a copy of the library (original kept)
                else:
                    tier = None                # another file's save
            elif same_name and same_sig:
                tier = 2                       # moved, untouched
            elif same_name:
                tier = 3                       # moved
            elif same_sig:
                tier = 4                       # renamed
            else:
                tier = 6
        if tier is not None and (best is None or (tier, order) < best[:2]):
            best = (tier, order, name)
    if best is not None:
        return best[2], True

    if not os.path.exists(os.path.join(SAVE_DIR, base)):
        return base, False
    n = 2
    while os.path.exists(os.path.join(SAVE_DIR, f"{stem} ({n}).json")):
        n += 1
    return f"{stem} ({n}).json", False


def _batch_resolve_save_name(*args, **kwargs):
    """BATCH_SAVE_CHECKS picks the 0.7.1-pre6 version; False gives 0.7's."""
    return (_batch_resolve_save_name__pre6 if BATCH_SAVE_CHECKS else _batch_resolve_save_name__07)(*args, **kwargs)


# ---- 0.8-pre31 VIDEO_PERFORMERS: who a music video shows singing ----------------------------------
# For a video file with no cover art, before the singer identity's rule steps: frames taken at
# moments the lead vocal is sounding (black, blank and near-identical frames skipped) go to the chat
# model in one question about the people seen singing. The answer is read the way the cover is,
# by the rule steps only (_video_rule_cover); the identity question never sees it.
_VIDEO_PERFORMERS_LAST = {}      # the last music video's answer, until it is saved ("video_performers")
_VIDEO_PERFORMER_PROMPT = """Inspect only the attached frames from a music video. They were taken at moments when the song's lead vocal is sounding. The frames are numbered in the order attached, with their times listed below.

Do not infer anything about the audio, and do not identify anyone from their face.

Return exactly these lines:

PEOPLE=none|one|two|three_or_more   (how many different people are seen singing or miming the vocal across all the frames)
PERSON_CUES=<for each of them: cautious apparent presentation and age range (child, teenager, adult), hair and clothing; if none say none>
SINGING_FRAMES=<the numbers of the frames where someone is seen singing or miming the vocal; if none say none>
OTHER_PEOPLE=<anyone else seen (band members playing, dancers, actors), briefly; if none say none>
CONFIDENCE=low|medium|high

RULES:

- Count someone as singing only when the frame shows it: mouth open at a microphone, or clearly miming the words to camera.
- The same person can look different from frame to frame (lighting, outfits, angles): count people, not outfits.
- Apparent age is visual presentation, not verified age.
- Do not infer vocal characteristics.
"""


def _video_performer_times(heavy, duration, count=24):
    """Candidate times (seconds): the middle of sung notes in the vocal stem's transcription, spread
    across the song; evenly spaced through the song when there are none."""
    notes = ((heavy or {}).get("stem_notes") or {}).get("vocals") or []
    times = []
    for n in notes:
        if not isinstance(n, dict):
            continue
        try:
            t = float(n.get("onset", n.get("t")))
            d = n.get("duration", n.get("d"))
            d = float(d) if d is not None else float(n.get("offset", t)) - t
        except (TypeError, ValueError):
            continue
        if d >= 0.25 and 0.0 <= t < duration:
            times.append(t + min(d / 2.0, 0.5))
    times.sort()
    if len(times) > count:
        times = [times[round(i * (len(times) - 1) / (count - 1))] for i in range(count)]
    if times:
        return times
    return [duration * (0.05 + 0.9 * i / (count - 1)) for i in range(count)]


def _video_grab_frame(path, at, small=False):
    """One frame of the video at `at` seconds: a JPEG at most 768 pixels on its longer side, or
    (small) 16x16 grey pixels for telling blank and repeated frames apart. None on failure."""
    vf = "scale=16:16,format=gray" if small else _ffmpeg_scale_filter_variants(768)[0]
    cmd = (["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-ss", f"{max(0.0, at):.3f}", "-i", path,
            "-frames:v", "1", "-vf", vf]
           + (["-f", "rawvideo", "-"] if small else ["-c:v", "mjpeg", "-q:v", "4", "-f", "image2", "-"]))
    try:
        done = subprocess.run(cmd, capture_output=True, timeout=60, stdin=subprocess.DEVNULL)
    except Exception:
        return None
    return done.stdout if done.returncode == 0 and done.stdout else None


def _parse_video_performers(text):
    t = re.sub(r"(?m)^[ \t]*(?:[-*•]|\d+[.)])[ \t]+", "", _strip_md_emphasis(text or ""))
    obs = {"raw": (text or "").strip()[:2000]}
    for field in ("PEOPLE", "PERSON_CUES", "SINGING_FRAMES", "OTHER_PEOPLE", "CONFIDENCE"):
        m = re.search(rf"^{field}\s*[:=]\s*(.+)$", t, re.MULTILINE | re.IGNORECASE)
        obs[field.lower()] = m.group(1).strip() if m else ""
    obs["people"] = _cover_people_count(obs["people"])
    return obs


def _video_performers(heavy):
    """The chat model's answer about the people a music video shows singing ({} when the track isn't
    a video file, can't be read, or the model can't look at pictures)."""
    path = str((heavy or {}).get("track_path") or "")
    if not (path and _is_video_input(path) and os.path.isfile(path) and OLLAMA_SUPPORTS_IMAGES):
        return {}
    try:
        duration = float(heavy.get("_video_audio_duration") or 0.0)
    except (TypeError, ValueError):
        duration = 0.0
    if duration <= 0:
        return {}
    limit = max(2, min(8, int(MAX_WRITER_IMAGES_PER_TURN or 8), int(IMAGE_RETAIN_LAST_N_MESSAGES or 8)))
    chosen, seen = [], []
    for at in _video_performer_times(heavy, duration):
        small = _video_grab_frame(path, at, small=True)
        if not small or len(small) < 256:
            continue
        px = list(small[:256])
        if sum(px) / 256.0 < 20 or max(px) - min(px) < 12:
            continue                                  # black or blank
        if any(sum(abs(a - b) for a, b in zip(px, s)) / 256.0 < 8 for s in seen):
            continue                                  # much the same picture as one already taken
        image = _video_grab_frame(path, at)
        if not image:
            continue
        chosen.append((at, image))
        seen.append(px)
        if len(chosen) >= limit:
            break
    if len(chosen) < 2:
        return {}
    status("Looking at who the music video shows singing...")
    labels = "; ".join(f"Frame {i} (video {int(at) // 60}:{int(at) % 60:02d})" for i, (at, image) in enumerate(chosen, 1))
    messages = [{"role": "user",          # one message, as an image-required request needs (none can be dropped)
                 "content": (_VIDEO_PERFORMER_PROMPT + f"\nThe {len(chosen)} attached frames, in order: {labels}."
                             "\n\nReturn the lines now."),
                 "images": [base64.b64encode(image).decode("ascii") for at, image in chosen]}]
    _ka = None
    if globals().get("BATCH_UNLOAD_OLLAMA", True) and _is_batch_context():
        _ka = globals().get("BATCH_OLLAMA_KEEP_ALIVE", 0)
    try:
        text, _usage = ollama_chat(messages, num_ctx=16384 if _is_batch_context() else None, keep_alive=_ka,
                                   require_images=True, max_output_tokens=4096, options=_scan_llm_options())
    except Exception as e:
        print(f"  (music video performers skipped: {e})")
        return {}
    finally:
        status_done(clear=True)
        if _ka is not None:
            try:
                ollama_unload_model()
            except Exception:
                pass
    obs = _parse_video_performers(text)
    obs["frame_times"] = [round(at, 2) for at, image in chosen]
    return obs


def save_song_data(
    filename,
    track_key,
    analysis,
    corrections,
    metadata=None,
    cover_observations=None,
    singer_identity=None,
    cover_bytes=None,
    cover_mime="image/jpeg",
    preserve_audio_basename=False,
    stem_midi=None,
    structure_sections=None,
    bpm=None,
    key=None,
    clap_embedding=None,
    structure_replay=None,
    discogs_styles=None,
    chord_segments=None,
    jamendo_instruments=None,
    listening_details=None,
    tuned_percussion=None,
    songformer_result=None,
):
    os.makedirs(SAVE_DIR, exist_ok=True)

    if preserve_audio_basename and track_key and not str(track_key).startswith(("http://", "https://")):
        # Same rule as the /batch skip check: never overwrite a save made
        # from a different file that happens to share Artist - Title.
        safe, _ = _batch_resolve_save_name(track_key, metadata=metadata)
    else:
        safe = _sanitize_saved_name(filename) or f"song_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        if not safe.lower().endswith(".json"):
            safe += ".json"

    path = os.path.join(SAVE_DIR, safe)

    lyrics_text = ""
    if metadata:
        lyrics_text = str(metadata.get("lyrics") or "").strip()

    # Prefer the embedding already computed at the end of the analysis
    # (resolve_identity_and_finalize). Fall back to computing it here so any
    # caller that doesn't pass one — an older/other call path, or a re-save of
    # a track whose analysis predates this — still produces a complete JSON,
    # exactly as before. compute_clap_embedding() is cached, so even the
    # fallback is free when the vector was already made for this file.
    clap_vector = _extract_clap_embedding(clap_embedding)
    if clap_vector is not None:
        # _extract_clap_embedding returns full float32 precision; round to the
        # same CLAP_EMBEDDING_PRECISION the compute path uses so a JSON written
        # from a passed-in vector is byte-identical to one written from a
        # freshly computed vector.
        clap_vector = [round(float(x), CLAP_EMBEDDING_PRECISION) for x in clap_vector]
    if clap_vector is None and ENABLE_CLAP_EMBEDDING and track_key and not str(track_key).startswith(("http://", "https://")):
        clap_vector = compute_clap_embedding(track_key)

    # Write the prepared cover art next to the JSON so saves are self-contained.
    # Cover art uses the same stem as the JSON (including spaces / special chars).
    cover_path = None
    if cover_bytes:
        stem_name = safe[:-5] if safe.lower().endswith(".json") else safe
        img_suffix = _cover_temp_suffix(cover_mime or _guess_image_mime(cover_bytes))
        cover_path = os.path.join(SAVE_DIR, stem_name + img_suffix)
        try:
            with open(cover_path, "wb") as f:
                f.write(cover_bytes)
        except Exception as e:
            print(f"  (could not write cover art to save: {e})")
            cover_path = None

    data = {
        "version": 2,
        "track_path": track_key,
        "label": track_label(track_key),
        "saved_at": datetime.datetime.now().isoformat(),
        "audio_file_signature": _audio_file_signature(track_key),
        "analysis": _saved_analysis_with_checks_block(
            _saved_analysis_with_instrument_check(_saved_analysis_with_genre_check(
                _saved_analysis_fixed_blocks(
                    _saved_analysis_fixed_layout(analysis, track_key, structure_sections),
                    has_cover=bool(cover_bytes) or bool(cover_observations))))),
        "corrections": corrections or {},
        "metadata": metadata or {},
        "lyrics": lyrics_text,
        "cover_observations": cover_observations or {},
        "singer_identity": singer_identity or "",
        "cover_art": os.path.basename(cover_path) if cover_path else "",
        "clap_embedding_model": CLAP_MODEL_ID if clap_vector is not None else "",
        "clap_embedding_dimension": CLAP_EMBEDDING_DIMENSION if clap_vector is not None else 0,
        "clap_embedding_aggregation": (
            "mean_of_l2_normalized_10s_windows_then_l2_normalized"
            if clap_vector is not None else ""
        ),
        "clap_embedding": clap_vector or [],
        # Raw per-stem transcribed MIDI for this recording. Stored so melody,
        # contour, motif and range questions asked after a /load are answered
        # from the actual notes rather than from whatever prose happened to be
        # written during the original scan.
        "stem_midi": _stem_notes_to_json(stem_midi) if stem_midi else {},
        "stem_midi_format": (
            "compact_v1: t=onset_s, d=duration_s, m=midi_note, n=note_name, "
            "v=amplitude, k=drum_type"
        ) if stem_midi else "",
        "stem_midi_complete": bool(stem_midi) and bool(FULL_MIDI_TRANSCRIPTION),
        "structure_sections": [
            {
                "name": s.get("name"),
                "start": round(float(s.get("start", 0.0)), 2),
                "end": round(float(s.get("end", 0.0)), 2),
                "cluster": s.get("cluster", ""),
                **({"name_check": s["name_check"]} if s.get("name_check") else {}),
            }
            for s in (structure_sections or [])
        ],
        "bpm": bpm,
        "key": key,
    }
    if STRUCTURE_SAVE_REPLAY and structure_replay:
        data["structure_replay"] = structure_replay
    # dev42: the Discogs style tags (ENABLE_DISCOGS_STYLES); key absent when none.
    try:
        _ds = _discogs_styles_for_save_list(discogs_styles) if discogs_styles else _discogs_styles_for_save(track_key)
    except Exception:
        _ds = []
    if _ds:
        data["discogs_style_model"] = DISCOGS_STYLE_MODEL_ID
        data["discogs_styles"] = _ds
    # 0.6.2: the MTG-Jamendo instrument tags (ENABLE_JAMENDO_INSTRUMENTS); key absent
    # when none. 0.6.2-dev3: a batch save runs in a fresh process, so the tags
    # come in with the analysis (jamendo_instruments=); otherwise this
    # process's own result for the track is used, as before.
    if ENABLE_JAMENDO_INSTRUMENTS:
        try:
            _ji = (jamendo_instruments if isinstance(jamendo_instruments, dict) and jamendo_instruments
                   else _jamendo_for_save(track_key, structure_sections))
        except Exception:
            _ji = None
        if _ji:
            data["jamendo_instruments"] = _ji
    # 0.7-pre3: the listening details (ENABLE_LISTENING_DETAILS); key absent when none.
    if listening_details and isinstance(listening_details, dict):
        data["listening_details"] = listening_details
    # 0.7-pre5: tuned-percussion tagger readings (TUNED_PERCUSSION_PROBE); key absent when none.
    if tuned_percussion and isinstance(tuned_percussion, dict):
        data["tuned_percussion"] = tuned_percussion
    # dev45: the chord segments (ENABLE_CHORD_PROGRESSION); key absent when none.
    try:
        _cs = _chord_segments_for_save(chord_segments, track_key)
    except Exception:
        _cs = []
    if _cs:
        data["chord_model"] = "lv-chordia/" + CHORD_PROGRESSION_DICTIONARY
        data["chord_segments"] = _cs

    # 0.7.1-pre19 SINGER_DETAIL_DIAGNOSTICS: the singer lookup's record for this track, in its own
    # field (the exporter leaves it out of exports); key absent when off or when there is none.
    if SINGER_DETAIL_DIAGNOSTICS and _SINGER_DETAIL_DIAG:
        _sd = dict(_SINGER_DETAIL_DIAG)
        _SINGER_DETAIL_DIAG.clear()
        _md = metadata if isinstance(metadata, dict) else {}
        if (_sd.get("artist") == str(_md.get("artist") or "").strip()
                and _sd.get("title") == str(_md.get("title") or "").strip()
                and time.time() - float(_sd.pop("_at", 0) or 0) < 3600):
            data["singer_detail_lookup"] = _sd

    # 0.8-pre31 VIDEO_PERFORMERS: who the music video shows singing (the frames' times and the
    # chat model's answer), in its own field (the exporter leaves it out); key absent when none.
    if VIDEO_PERFORMERS and _VIDEO_PERFORMERS_LAST:
        _vp = dict(_VIDEO_PERFORMERS_LAST)
        _VIDEO_PERFORMERS_LAST.clear()
        if (_vp.pop("_source", None) == str(track_key or "")
                and time.time() - float(_vp.pop("_at", 0) or 0) < 3600):
            data["video_performers"] = _vp

    # dev2: write to a temporary file in the same folder, then swap it in, so
    # a save that fails part-way (or a crash) never truncates an earlier save
    # of the same track. The bytes written are exactly as before. The
    # temporary name does not end in .json, so /load, /batch and the export
    # script never pick it up.
    # A short fixed-form name, so a long song name cannot exceed the
    # filesystem's name limit for the temporary file when the save would fit.
    tmp_path = os.path.join(SAVE_DIR, f".musiclyse-save-{os.getpid()}-{threading.get_ident()}.tmp")
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        os.replace(tmp_path, path)
    except BaseException:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise
    if songformer_result:
        _songformer_write_result(path, songformer_result)      # pre59

    return path, cover_path


def list_audio_files_in_folder(folder):
    """Non-recursive listing of audio files in folder (stable sorted order)."""
    folder = os.path.abspath(os.path.expanduser(folder))
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Not a directory: {folder}")
    out = []
    for name in sorted(os.listdir(folder)):
        path = os.path.join(folder, name)
        if not os.path.isfile(path):
            continue
        ext = os.path.splitext(name)[1].lower()
        if ext in ANALYSIS_EXTENSIONS:
            out.append(path)
    return out


def _dsp_stage(track_path, resolved_path, dsp_temp_files, stem_temp_files, demucs_out_dirs, *,
               stem_source=None, objective_original=None, snippet_dsp_path=None,
               register=True, parallel=False, struct_grid=False):
    """dev55: the signal-processing block that ran after the listening
    model's main pass (analysis WAV, Demucs, objective report, vocal pitch,
    Essentia, PANNs genre/mood), as one function so it can run in the
    background (parallel=True: Demucs alongside the objective measurements,
    then the beat grid for the section map and the Omnizart prefetch). Same
    calls, same inputs, same results as the in-order code it replaces."""
    out = {
        "dsp_path": None, "precomputed_stems": {}, "precomputed_demucs_out_dir": None,
        "vocal_pitch_source": "full mix (no isolated vocal stem available)",
        "objective_report": "", "vocal_objective_report": "", "essentia_report": "",
        "genre_mood_report": "", "panns_genre_tags": [],
        "discogs_style_report": "", "discogs_styles": [], "chord_segments": [],
    }
    if snippet_dsp_path is not None:
        dsp_path = snippet_dsp_path
    elif track_path in dsp_temp_files:
        dsp_path = dsp_temp_files[track_path]
    else:
        status("Preparing higher-rate WAV for signal processing...")
        try:
            dsp_path = convert_to_wav(objective_original or track_path, sample_rate=22050)
            dsp_temp_files[track_path] = dsp_path
        except Exception:
            _ext = os.path.splitext(str(track_path))[1].lower()
            dsp_path = track_path if _ext in (".wav", ".flac") else resolved_path
    if register and dsp_path and dsp_path != track_path:
        register_panns_source(dsp_path, objective_original or track_path)
    out["dsp_path"] = dsp_path

    want_stems = ENABLE_VOCAL_OBJECTIVE_REPORT and (ENABLE_STEM_MIDI or ENABLE_VOCAL_STEM_ISOLATION)

    def _demucs():
        return _prepare_demucs_stems_for_track(
            stem_source or track_path, stem_temp_files, demucs_out_dirs,
            status_fn=status, deep_mode=DEEP_MODE,
        )

    dem_task = None
    if want_stems:
        if parallel:
            dem_task = _BackgroundTask("Demucs stem separation", _demucs)
        else:
            out["precomputed_stems"], out["precomputed_demucs_out_dir"] = _demucs()

    def _join_demucs():
        if dem_task is None:
            return
        ok, val = dem_task.result()
        if not ok or val is None:
            val = _demucs()
        out["precomputed_stems"], out["precomputed_demucs_out_dir"] = val
        out["precomputed_stems"] = out["precomputed_stems"] or {}

    try:
        _dsp_stage_measure(out, dsp_path, track_path, resolved_path, objective_original,
                           parallel, struct_grid)
    finally:
        _join_demucs()
    if register and out.get("discogs_styles"):
        _discogs_style_remember(track_path, out["discogs_styles"])     # dev42: for the save
    if register and ENABLE_JAMENDO_INSTRUMENTS:
        jamendo_tag_track(track_path, out.get("dsp_path"), out.get("precomputed_stems") or None)  # dev3
    if register and out.get("chord_segments"):
        _chord_remember(track_path, out["chord_segments"])             # dev45: for the save
    if parallel and ENABLE_STEM_MIDI and out["precomputed_stems"]:
        _start_omnizart_prefetch(out["precomputed_stems"])
    if parallel and dsp_path is not None and ENABLE_VOCAL_OBJECTIVE_REPORT:
        _vocal_pitch_into(out, dsp_path)
    if out["precomputed_stems"].get("vocals"):
        out["vocal_pitch_source"] = "isolated vocal stem (Demucs)"
    return out


def _dsp_stage_measure(out, dsp_path, track_path, resolved_path, objective_original,
                       parallel, struct_grid):
    if dsp_path is not None:
        if ENABLE_OBJECTIVE_AUDIO_REPORT:
            status("Measuring beat/timbre with signal processing...")
            out["objective_report"] = build_objective_audio_report(
                dsp_path, original_path=objective_original or track_path
            )
        if not parallel and ENABLE_VOCAL_OBJECTIVE_REPORT:
            _vocal_pitch_into(out, dsp_path)
        if ENABLE_ESSENTIA_REPORT and ESSENTIA_AVAILABLE:
            status("Measuring tempo/key/spectral features with Essentia...")
            out["essentia_report"] = build_essentia_report(dsp_path)
        if FULL_BAND_TEMPO_CHECK and out.get("objective_report"):
            try:
                _full_band_tempo_check(out, objective_original or track_path, dsp_path)
            except Exception as e:
                print(f"  (full-bandwidth tempo check skipped: {e})")
        if ENABLE_GENRE_MOOD_TAGGING:
            status("Cross-checking genre/mood with an independent classifier...")
            try:
                out["genre_mood_report"], out["panns_genre_tags"] = build_genre_mood_signal_report(dsp_path)
            except Exception as e:
                print(f"  (genre/mood signal skipped: {e})")
                out["genre_mood_report"], out["panns_genre_tags"] = "", []
        if ENABLE_DISCOGS_STYLES:
            out["discogs_style_report"], out["discogs_styles"] = build_discogs_style_report(dsp_path)
        if ENABLE_CHORD_PROGRESSION:
            out["chord_segments"] = chord_segments_for(dsp_path)       # dev45
        if parallel and struct_grid and STRUCTURE_V2_ENABLED:
            try:
                _hint = (extract_objective_bpm(out["objective_report"])
                         or extract_essentia_bpm(out["essentia_report"]))
            except Exception:
                _hint = None
            _sx_precompute_grid(dsp_path or resolved_path or track_path, _hint)


def _prefetch_tags_and_clap(stems, mix_path, clap_source, audio_source=None):
    """dev55 background work after Demucs: the instrument tagger's window
    probabilities for the mix and each stem (filled into the same cache the
    stem report reads, with the same settings), and the CLAP embedding (into
    its cache). Returns the CLAP vector or None."""
    paths = []
    if ENABLE_WHOLE_MIX_INSTRUMENT_TAGGING and mix_path:
        paths.append(mix_path)
    if ENABLE_INSTRUMENT_TAGGING:
        for st in INSTRUMENT_TAG_STEMS:
            p = (stems or {}).get(st)
            if p and os.path.exists(p):
                paths.append(p)
    for p in paths[: max(1, int(PANNS_WINDOW_CACHE_SIZE)) - 1]:
        try:
            _panns_window_probs(p, INSTRUMENT_TAG_WINDOW_SECONDS, INSTRUMENT_TAG_HOP_SECONDS,
                                INSTRUMENT_TAG_MAX_WINDOWS, INSTRUMENT_TAG_NORMALIZE)
        except Exception:
            pass
    vec = None
    if ENABLE_CLAP_EMBEDDING and clap_source and not str(clap_source).startswith(("http://", "https://")):
        try:
            vec = compute_clap_embedding(clap_source, audio_source=audio_source)
        except Exception:
            vec = None
    return vec


def run_fresh_track_analysis_heavy(
    track_path, *, audio_temp_files, dsp_temp_files, stem_temp_files, demucs_out_dirs
):
    track_path = os.path.abspath(os.path.expanduser(track_path))
    if not os.path.exists(track_path):
        raise FileNotFoundError(f"File not found: {track_path}")
    with _analysis_audio_source(track_path) as audio_source:
        try:
            return _run_fresh_track_analysis_heavy(
                track_path, audio_source=audio_source,
                audio_temp_files=audio_temp_files, dsp_temp_files=dsp_temp_files,
                stem_temp_files=stem_temp_files, demucs_out_dirs=demucs_out_dirs,
            )
        except BaseException:
            if audio_source != track_path:
                _settle_background_tasks()
            raise


def _run_fresh_track_analysis_heavy(
    track_path,
    *,
    audio_source,
    audio_temp_files,
    dsp_temp_files,
    stem_temp_files,
    demucs_out_dirs,
):
    """
    Full /listen-quality analysis for one local file, MINUS singer identity
    resolution. Does not touch writer_history or session token counters.

    Singer identity resolution is deliberately NOT done here — see
    resolve_identity_and_finalize(). It's a plain-text Ollama call with no
    dependency on MOSS-Music/Demucs/Omnizart/Essentia, so splitting it
    out lets it run in a separate, model-free process for isolated batch
    tracks (see run_batch_one_finish() / BATCH_SPLIT_IDENTITY_PROCESS): the
    torch/TF/MPS allocations from this heavy stage are not always reliably
    reclaimed within the SAME process even after unload + gc + cache-empty
    (a known MPS/TF limitation), so previously Ollama could try to load the
    writer model for identity resolution on top of that stranded memory and
    get OOM-killed (SIGKILL / "child exited -9") right at that stage.

    Returns a dict with everything resolve_identity_and_finalize() and
    save_song_data() need.
    """
    if WHISPER_HOLD_LANGUAGE:          # 0.7.1-pre21: this track's lyrics language is decided afresh
        _WHISPER_LANG_STATE.update(decided={}, record=None, tag_lyrics=None)
    track_path = os.path.abspath(os.path.expanduser(track_path))
    if not os.path.exists(track_path):
        raise FileNotFoundError(f"File not found: {track_path}")

    # Free Ollama writer weights before the heavy analysis peak so MOSS-Music/Demucs/
    # Omnizart are not competing with a resident 30B model. Chat history in the
    # parent Musiclyse process is unaffected (separate address space / stays
    # in Python RAM). Results of this analysis stay in local variables below.
    if globals().get("BATCH_UNLOAD_OLLAMA", True) and _is_batch_context():
        try:
            status("Unloading Ollama writer before analysis (batch)...")
            ollama_unload_model()
            status_done("Ollama writer unloaded for analysis phase")
        except Exception:
            pass

    metadata = {}
    cover_b64 = None
    cover_bytes_for_save = None
    cover_mime = "image/jpeg"
    cover_observations = {}
    singer_identity = ""

    if ENABLE_FILE_METADATA:
        status("Reading file metadata/tags...")
        try:
            meta, cover_bytes, cover_mime = extract_audio_metadata(track_path)
            metadata = meta or {}
            if cover_bytes:
                prepared = prepare_cover_image_for_ollama(cover_bytes, cover_mime)
                if prepared:
                    cover_bytes_for_save = prepared
                    cover_b64 = base64.b64encode(prepared).decode("utf-8")
                    cover_mime = _guess_image_mime(prepared)
        except Exception as e:
            print(f"  (metadata/cover extraction skipped: {e})")

    _user_cover = os.environ.get("MUSICLYSE_USER_COVER", "") if SCAN_IMAGE_OPTIONS else ""
    if _user_cover:
        # 0.8-pre34 SCAN_IMAGE_OPTIONS: --cover -- this image is the track's cover for this scan
        try:
            with open(_user_cover, "rb") as f:
                _user_cover_raw = f.read()
            prepared = prepare_cover_image_for_ollama(_user_cover_raw, _guess_image_mime(_user_cover_raw))
            if prepared:
                cover_bytes_for_save = prepared
                cover_b64 = base64.b64encode(prepared).decode("utf-8")
                cover_mime = _guess_image_mime(prepared)
        except Exception as e:
            print(f"  (--cover image skipped: {e})")

    if (
        ENABLE_COVER_ART_DESCRIPTION
        and cover_b64
        and cover_b64 != NO_COVER_SENTINEL
    ):
        status("Describing embedded cover art...")
        try:
            obs = describe_cover_art(cover_b64)
            if obs:
                cover_observations = obs
        except Exception as e:
            print(f"  (cover description skipped: {e})")

    ext = os.path.splitext(track_path)[1].lower()
    # WAV/FLAC used to be passed straight through. The listening model's audio
    # front end expects 16 kHz mono; 44.1/48/96 kHz, 24/32-bit float or
    # multichannel masters could reach it as unusable features, and it then
    # answered "no clear genre / instrumentation not discernible". Always
    # convert (cached per track). DSP and stems still read the original.
    if track_path in audio_temp_files:
        resolved_path = audio_temp_files[track_path]
    else:
        status(f"Converting {ext} to 16 kHz mono WAV for the listening model...")
        try:
            resolved_path = convert_to_wav(audio_source)
            audio_temp_files[track_path] = resolved_path
        except Exception as e:
            if ext not in (".wav", ".flac"):
                raise
            print(f"  (conversion failed, using original file: {e})")
            resolved_path = track_path

    # dev55: signal processing, Demucs and the vocal pitch report run in the
    # background while the listening model loads and does its first passes
    # (see PARALLEL_ANALYSIS); joined after the main pass, exactly where the
    # in-order run did this work.
    video_sources = ({"stem_source": audio_source, "objective_original": audio_source}
                     if audio_source != track_path else {})
    _dsp_task = None
    if PARALLEL_ANALYSIS:
        _clear_omnizart_prefetch()
        try:
            _dsp_task = _BackgroundTask(
                "signal processing + Demucs + vocal pitch", _dsp_stage,
                track_path, resolved_path, dsp_temp_files, stem_temp_files, demucs_out_dirs,
                parallel=True, struct_grid=True, **video_sources,
            )
        except Exception:
            _dsp_task = None

    ollama_unload_model()
    moss_model, moss_processor, _moss_device = get_moss_music()

    status("Listening — estimating era...")
    # Only gathered when the file has no usable YEAR tag -- that absence is
    # the whole reason era detection has to guess. Tracks with a real tag are
    # completely unaffected (see _build_era_supplementary_evidence). Wrapped
    # here too so any failure just falls back to the original audio-only
    # prompt, unchanged.
    _era_supplement = ""
    if not (metadata.get("year") if isinstance(metadata, dict) else None):
        try:
            _era_supplement = _build_era_supplementary_evidence(track_path, metadata, audio_source=audio_source)
        except Exception:
            _era_supplement = ""
    era_conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": ERA_ANALYSIS_PROMPT + _era_supplement},
                {"type": "audio", "path": resolved_path},
            ],
        }
    ]
    era_result = moss_generate(moss_model, moss_processor, era_conversation, max_new_tokens=1024)

    status("Listening — running full analysis (this can take a while)...")
    first_pass, moss_conversation, main_max_tokens = _moss_run_main_analysis(
        moss_model, moss_processor, resolved_path, deep=DEEP_MODE
    )

    _dsp = None
    if _dsp_task is not None:
        status("Finishing background signal processing...")
        _ok, _dsp = _dsp_task.result()
        if not _ok:
            _dsp = None
    if _dsp is None:
        _dsp = _dsp_stage(track_path, resolved_path, dsp_temp_files, stem_temp_files, demucs_out_dirs,
                          **video_sources)
    dsp_path = _dsp["dsp_path"]
    precomputed_stems = _dsp["precomputed_stems"] or {}
    precomputed_demucs_out_dir = _dsp["precomputed_demucs_out_dir"]
    vocal_pitch_source = _dsp["vocal_pitch_source"]
    objective_report = _dsp["objective_report"]
    vocal_objective_report = _dsp["vocal_objective_report"]
    essentia_report = _dsp["essentia_report"]
    genre_mood_report = _dsp["genre_mood_report"]
    panns_genre_tags = _dsp["panns_genre_tags"]
    discogs_style_report = _dsp.get("discogs_style_report") or ""      # dev42
    discogs_styles = _dsp.get("discogs_styles") or []
    chord_segments = _dsp.get("chord_segments") or []                  # dev45
    # dev55: instrument-tagger windows and the CLAP embedding in the
    # background while the listening model, Whisper and Omnizart work.
    _tag_task = None
    if PARALLEL_ANALYSIS:
        try:
            _tag_task = _BackgroundTask("instrument-tagger windows + CLAP embedding",
                                        _prefetch_tags_and_clap, precomputed_stems, dsp_path, track_path,
                                        audio_source if video_sources else None)
        except Exception:
            _tag_task = None

    vocal_result = ""
    confirmation_result = ""
    initial_lead = ""
    initial_backing = ""
    confirm_lead = ""
    confirm_confidence = ""
    final_lead = ""
    vocal_pitch = {"median": None, "low": None, "high": None, "note": None, "band": None}
    median_f0 = None
    no_clear_vocals = False
    multi_voice_fields = {}

    if ENABLE_VOCAL_PASS:
        status("Listening — estimating singer profile...")
        vocal_prompt_text = VOCAL_ANALYSIS_PROMPT
        if vocal_objective_report:
            vocal_prompt_text += (
                "\n\nObjective vocal measurements (evidence only, not proof):\n"
                f"{vocal_objective_report}"
            )
        vocal_conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": vocal_prompt_text},
                    {"type": "audio", "path": resolved_path},
                ],
            }
        ]
        vocal_result = moss_generate(moss_model, moss_processor, vocal_conversation, max_new_tokens=768)
        vocal_result = _collapse_repeated_line_blocks(vocal_result or "")
        if vocal_result:
            initial_lead, initial_backing = parse_vocal_tags(vocal_result)
        vocal_pitch = extract_vocal_pitch_summary(vocal_objective_report) if vocal_objective_report else vocal_pitch
        median_f0 = vocal_pitch.get("median")
        _vp_flag = _parse_vocals_present_flag(vocal_result or "")
        no_clear_vocals = _vp_flag in ("no",)
        no_clear_vocals, _early_vocal_salvage, _ = _salvage_false_no_vocals(
            vocal_result,
            no_clear_vocals,
            median_f0=median_f0,
            vocal_objective_report=vocal_objective_report or "",
            stem_midi_report="",
            initial_lead=initial_lead,
        )
        if ENABLE_VOCAL_CONFIRMATION_PASS and vocal_result and not no_clear_vocals:
            should_confirm = False
            if initial_lead in UNCERTAIN_YOUNG_CATEGORIES:
                should_confirm = True
            elif initial_lead in FEMALE_LEAD_CATEGORIES and (
                (median_f0 is not None and median_f0 >= VOCAL_CONFIRMATION_F0_THRESHOLD)
                or (median_f0 is None and VOCAL_CONFIRMATION_WITHOUT_F0)
            ):
                should_confirm = True
            elif initial_lead in (MALE_LEAD_CATEGORIES | ADOLESCENT_MALE_CATEGORIES) and (
                median_f0 is not None
                and F0_MALE_HIGH_CONFIRM_HZ is not None
                and median_f0 >= float(F0_MALE_HIGH_CONFIRM_HZ)
            ):
                # High track-wide median on a male-tagged lead is a common
                # female→male mislabel path; run confirmation before locking in.
                should_confirm = True
            elif initial_lead == "mixed_leads":
                should_confirm = True
            else:
                # Confirm when the vocal pass itself flags multiple distinct voices.
                _mv_early = parse_multi_voice_fields(vocal_result or "")
                if _mv_early.get("voice_arrangement") in ("duet_co_leads", "call_response"):
                    should_confirm = True
                elif _mv_early.get("num_distinct_voices") in ("2", "3", "3+", "two", "three"):
                    should_confirm = True
            if should_confirm:
                status("Listening — confirming lead voice category...")
                confirmation_conversation = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": VOCAL_CONFIRMATION_PROMPT},
                            {"type": "audio", "path": resolved_path},
                        ],
                    }
                ]
                confirmation_result = moss_generate(
                    moss_model, moss_processor, confirmation_conversation, max_new_tokens=256
                )
                if confirmation_result:
                    confirm_lead, confirm_confidence = parse_vocal_confirmation(confirmation_result)
        # Populate multi_voice_fields BEFORE the final-lead decision so duets are honoured.
        multi_voice_fields = parse_multi_voice_fields(vocal_result or "")
        if confirmation_result:
            for _k, _v in parse_multi_voice_fields(confirmation_result).items():
                if _v and not multi_voice_fields.get(_k):
                    multi_voice_fields[_k] = _v

        _decision_f0 = _identity_pitch_median(median_f0, vocal_objective_report,
            initial_lead, confirm_lead, confirm_confidence)
        final_lead = choose_final_vocal_lead(
            initial_lead, confirm_lead, confirm_confidence, confirmation_result,
            median_f0=_decision_f0,
            multi_voice_fields=multi_voice_fields,
        )
        final_lead = _apply_vocal_age_guard(
            final_lead,
            (vocal_result or "") + "\n" + (confirmation_result or ""),
        )
        final_lead = _apply_f0_pitch_guard(
            final_lead,
            _decision_f0,
            low_f0=vocal_pitch.get("low") if isinstance(vocal_pitch, dict) else None,
            high_f0=vocal_pitch.get("high") if isinstance(vocal_pitch, dict) else None,
            analysis_text=(vocal_result or "") + "\n" + (confirmation_result or ""),
            confirmed_adult_male=(
                confirm_lead in ("post_puberty_male", "adult_male")
                and confirm_confidence in ("medium", "high")
            ),
        )
        # dev55: the listening model's own free descriptions of the voice
        # (see VOCAL_FREE_TEXT_GENDER), used only when nothing above settled.
        final_lead, _vft_info = _free_text_vocal_step(
            moss_model, moss_processor, resolved_path, first_pass, era_result,
            vocal_result, final_lead, _decision_f0, no_clear_vocals)


    # Is there a voice at all? Decided once, from the main pass + the measured
    # vocal stem, before any singer profile, duet reading, lyric pass or
    # identity resolution is built on the vocal pass's answer.
    _vocal_arb = {"instrumental": False}
    try:
        _vocal_arb = _arbitrate_vocal_presence(
            first_pass, vocal_result, vocal_objective_report,
            vocal_stem_path=precomputed_stems.get("vocals"),
            mix_path=(audio_source if os.path.exists(audio_source) else dsp_path),
            other_text=(locals().get("era_result") or ""),
        )
    except Exception as _arb_err:
        print(f"  (vocal-presence arbitration skipped: {_arb_err})")
    if _vocal_arb.get("instrumental"):
        status_done("Vocal check: instrumental (vocal-pass claim not supported by the vocal stem)")
        no_clear_vocals = True
        initial_lead = initial_backing = confirm_lead = final_lead = ""
        multi_voice_fields = {}
        median_f0 = None

    # If the main listening-model pass still claims "no music" but DSP heard a track, replace
    # the failure text with a conservative scaffold before self-check / save.
    first_pass = _moss_salvage_empty_analysis(
        first_pass, objective_report, essentia_report, vocal_result
    )
    _remember_moss_first_pass(track_path, first_pass)

    if FAST_MODE:
        revised = first_pass
    else:
        status("Double-checking its own analysis for overconfident claims...")
        self_check_text = SELF_CHECK_PROMPT + "\n\n" + STYLE_EVIDENCE_FIREWALL
        if vocal_result or confirmation_result:
            self_check_text += "\n\nVocal profile evidence:\n"
            if vocal_result:
                self_check_text += f"{vocal_result}\n"
            if confirmation_result:
                self_check_text += f"{confirmation_result}\n"
        objective_crosscheck_parts = []
        if essentia_report:
            objective_crosscheck_parts.append(essentia_report)
        if objective_report:
            objective_crosscheck_parts.append(objective_report)
        if objective_crosscheck_parts:
            self_check_text += (
                "\n\nIndependent signal-processing cross-checks "
                "(use only for tempo/beat, key/key strength when explicitly reported, timbre, "
                "element activity, dynamic range, and -- when a STRUCTURAL BOUNDARY CANDIDATES "
                "line is present -- section-timing in STRUCTURE; do NOT use librosa/Essentia "
                "spectral stats to invent GENRE or change vocal identity):\n"
                + "\n\n".join(objective_crosscheck_parts)
            )
            if genre_mood_report:
                self_check_text += (
                    "\n\nIndependent genre/mood classifier (PANNs/AudioSet) — this MAY be used "
                    "to revise GENRE_RANKED when it clearly conflicts with a rock/pop-punk top rank "
                    "driven mainly by guitar texture while this signal is electronic/dance/synth-pop. "
                    "See GENRE self-check rules.\n"
                    + genre_mood_report
                )
        moss_conversation.append(
            {"role": "assistant", "content": [{"type": "text", "text": first_pass}]}
        )
        moss_conversation.append(
            {"role": "user", "content": [{"type": "text", "text": self_check_text}]}
        )
        revised = _moss_self_check_generate(moss_model, moss_processor, moss_conversation, main_max_tokens)

    # ------------------------------------------------------------------
    # SONG STRUCTURE (objective-first) — resolved here, once, for the run.
    # ------------------------------------------------------------------
    # Placed immediately before the unload because this is the last moment the
    # audio model is resident. It is only asked to NAME parts whose boundaries
    # have already been fixed by signal analysis, so the timestamps never
    # depend on it. Running it here (rather than in the finalize stage, as it
    # used to) avoids reloading the model after it has been freed, and makes
    # the resulting section map available to the stem MIDI pass below — so the
    # melody and intensity blocks are keyed to the SAME sections the writer is
    # later told are authoritative, instead of an earlier, different estimate.
    _structure_sections = []
    _bnd_pre = []
    _struct_grid = {}
    try:
        _total_duration_pre = _resolve_total_duration(
            track_path=audio_source, dsp_path=dsp_path, objective_report=objective_report
        )
        _struct_audio_path = dsp_path or resolved_path or track_path
        if (
            _struct_audio_path
            and _total_duration_pre
            and not str(_struct_audio_path).startswith(("http://", "https://"))
            and os.path.exists(_struct_audio_path)
        ):
            # Tempo reconciliation (final_bpm) does not happen until later in
            # this function, so it cannot be referenced here -- doing so raised
            # NameError, which the except below swallowed, silently degrading
            # every scan to the evenly-spaced filler windows. The beat tracker
            # works without a hint; pass a measured tempo only when one already
            # exists at this point, purely to help it lock on faster.
            _struct_bpm_hint = None
            try:
                _struct_bpm_hint = (
                    extract_objective_bpm(objective_report)
                    or extract_essentia_bpm(essentia_report)
                )
            except Exception:
                _struct_bpm_hint = None

            status("Resolving song structure from the audio (bars -> evidence -> sections)...")
            _y_struct, _sr_struct = librosa.load(_struct_audio_path, sr=None, mono=True)
            # The beat grid and the cached beat features land in _struct_grid,
            # so the final pass (after stems and lyrics exist) re-segments on
            # the same beats without decoding the audio again.
            _struct_grid = {}
            _structure_sections = _build_objective_first_structure(
                _struct_audio_path, _y_struct, _sr_struct, _total_duration_pre,
                moss_model, moss_processor, bpm_hint=_struct_bpm_hint,
                details_out=_struct_grid,
                instrumental=bool(_vocal_arb.get("instrumental")),
            )
            if not _struct_grid.get("beats"):
                try:
                    _sb, _sd, _sbl, _ = _beat_and_downbeat_grid(
                        _y_struct, _sr_struct, bpm_hint=_struct_bpm_hint
                    )
                    _struct_grid = {"beats": _sb, "downbeats": _sd, "beat_len": _sbl}
                except Exception:
                    _struct_grid = {}
            del _y_struct
            if _structure_sections:
                _bnd_pre = [float(s["start"]) for s in _structure_sections[1:]]
                status_done(
                    f"Structure resolved from audio ({len(_structure_sections)} sections)"
                )
    except Exception as e:
        # Loud, with a traceback. This used to print a single easy-to-miss line,
        # so a hard failure looked exactly like a successful scan right up until
        # the section map turned out to be evenly-spaced filler.
        import traceback
        print(f"  !! STRUCTURE DETECTION FAILED: {type(e).__name__}: {e}")
        traceback.print_exc()
        _structure_sections = []

    # The listening model's job for this track is done as of the self-check
    # revision above — free it now, before possibly loading Whisper for the
    # lyrics pass, so the two models are never resident in RAM/VRAM at once.
    # dev68: drop this function's own references first. unload_moss_music()
    # only cleared its cache, so these two locals kept the whole model alive
    # (and its GPU memory allocated) through Whisper and the rest of the
    # analysis, and its clean-up ran while the model was still referenced.
    _svc = None
    if not _vocal_arb.get("instrumental"):
        _svc = _section_voice_check(
            moss_model, moss_processor, precomputed_stems.get("vocals"), locals().get("final_lead"),
            locals().get("initial_lead"), locals().get("confirm_lead"), locals().get("vocal_objective_report"),
            mix_path=locals().get("resolved_path"),
            section_hints=locals().get("_structure_sections"),
            vocals_present=_parse_vocals_present_flag(locals().get("vocal_result") or ""))
    if ((_svc or {}).get("presence_unresolved")
            or _vocal_presence_still_unresolved(vocal_result, _svc, final_lead)):
        revised += VOCAL_PRESENCE_UNRESOLVED_NOTE
        final_lead = "uncertain"
    if (_svc or {}).get("presence_arbitration"):
        _vocal_arb = dict(_vocal_arb, **_svc["presence_arbitration"])
        no_clear_vocals = True
        initial_lead = initial_backing = confirm_lead = final_lead = ""
        multi_voice_fields = {}
        median_f0 = None
        _svc = None
    # 0.7-pre3 Stage 1: the listening details, after every existing listening
    # pass (each pass is seeded on its own, so none of the above changes).
    # Saved only; nothing below reads it.
    _listening_details = None
    if ENABLE_LISTENING_DETAILS:
        try:
            _ld_unresolved = bool(LISTENING_DETAILS_NARROW and VOCAL_PRESENCE_UNRESOLVED_MARKER in str(revised or ""))
            _listening_details = _listening_details_pass(
                moss_model, moss_processor, locals().get("resolved_path"),
                list(locals().get("_structure_sections") or []),
                bool(_vocal_arb.get("instrumental")) or bool(locals().get("no_clear_vocals")) or _ld_unresolved)
            if _listening_details and _ld_unresolved:
                _listening_details["vocal_presence"] = "unresolved (treated as no voice)"
        except Exception as e:
            print(f"  (listening details skipped: {type(e).__name__}: {e})")
            _listening_details = None
    _genre_clarifications = {}
    try:
        _genre_clarifications = _moss_genre_clarifications_for_track(
            revised, moss_model, moss_processor, resolved_path)
    except Exception as e:
        print(f"  (genre clarification skipped: {type(e).__name__}: {e})")
    if RELEASE_MOSS_LOCALS_BEFORE_UNLOAD:
        moss_model = moss_processor = None
        if MOSS_RELEASE_REFRESH_LOCALS:
            locals()    # dev41: refresh the variable snapshot (see the setting)
    unload_moss_music()

    tag_lyrics = str((metadata or {}).get("lyrics") or "").strip()
    _lyric_chunks = []  # default when the Whisper lyrics pass below is skipped
    # Any non-empty file-tag lyrics are authoritative. Do not let a short tag
    # block fall through to snippet-level Whisper transcription.
    skip_lyrics_transcription = (
        SKIP_LYRICS_TRANSCRIPTION_WHEN_TAGS_PRESENT
        and bool(tag_lyrics)
    )
    if _vocal_arb.get("instrumental"):
        # Whisper on an instrumental only ever produces hallucinations
        # ("we'll be right back", "thanks for watching" in other languages,
        # runs of repeated characters).
        status("Skipping lyrics transcription (measured instrumental)...")
        revised += "\n\nLYRICS: none — the recording is instrumental (see VOCAL PRESENCE ARBITRATION)."
    elif VOCAL_PRESENCE_UNRESOLVED_MARKER in revised:
        status("Skipping lyrics transcription (human vocal presence unresolved)...")
        full_lyrics, _lyric_chunks, _whisper_ok = "", [], False
        revised += "\n\nLYRICS: audio transcription withheld because human vocal presence is unresolved."
    elif skip_lyrics_transcription:
        status("Skipping lyrics transcription pass (file-tag lyrics already present)...")
        revised += (
            "\n\nLYRICS: reliable lyrics are present and are the preferred source for lyric questions."
        )
        if STRUCTURE_TIME_TAG_LYRICS:
            _lyric_chunks = _timed_tag_lyrics_for_structure(
                resolved_path, tag_lyrics,
                vocal_stem_path=(locals().get("precomputed_stems") or {}).get("vocals"))
    else:
        status("Transcribing full lyrics with Whisper (dedicated pass)...")
        # Whisper (WHISPER_MODEL_ID) does this pass, not the listening model —
        # audio-language models hallucinate readily on song lyrics (inventing
        # lines, drifting into unrelated prose, "correcting" words toward
        # remembered/expected lyrics), whereas Whisper is a dedicated,
        # comparatively literal ASR model. Whisper is lazy-loaded here (only
        # for the duration of this pass) and released again immediately
        # after, same RAM-optimization principle MOSS-Music uses above.
        # As before, two independent decodes (deterministic + sampled) are
        # cross-checked against each other to catch ungrounded/hallucinated
        # spans that a single pass wouldn't reveal.
        try:
            full_lyrics, _lyric_chunks = whisper_transcribe_lyrics_full(resolved_path, cross_check=True)
            _whisper_ok = True
        except Exception as e:
            print(f"  (lyrics transcription skipped: {e})")
            full_lyrics, _lyric_chunks = "", []
            _whisper_ok = False
        unload_whisper()

        if full_lyrics and full_lyrics.strip():
            revised += f"\n\nLYRICS HEARD IN AUDIO (secondary evidence; do not use when reliable lyrics are present):\n{full_lyrics}"

        lyric_repetition_report = build_lyric_repetition_report(_lyric_chunks)
        if lyric_repetition_report:
            revised += "\n\n" + lyric_repetition_report

    revised += f"\n\n11. ERA / RELEASE PERIOD (isolated dedicated pass{_era_pass_label(era_result)}):\n{era_result}"
    try:
        revised += _build_era_remaster_caveat(objective_report)
    except Exception:
        pass
    if (not _vocal_arb.get("instrumental") and locals().get("_whisper_ok")
            and not str(locals().get("tag_lyrics") or "").strip()):
        _tx = _faint_vocal_texture_reasons(_vocal_arb, locals().get("full_lyrics") or "")
        if _tx:
            _vocal_arb = dict(_vocal_arb, instrumental=True, texture=True, reasons=_tx)
            no_clear_vocals = True
            initial_lead = initial_backing = confirm_lead = final_lead = ""
            multi_voice_fields = {}
            median_f0 = None
            status_done("Vocal check: only a faint vocal-like texture, no sung lead -> instrumental")
    if _vocal_arb.get("instrumental"):
        revised += _vocal_arbitration_block(_vocal_arb, vocal_result)
    elif vocal_result:
        revised += (
            "\n\nVOCAL / SINGER PROFILE (isolated dedicated pass):\n"
            + _mark_superseded_vocal_tags(vocal_result, locals().get("final_lead"))
        )
        if confirmation_result:
            revised += (
                "\n\nVOCAL CONFIRMATION PASS:\n"
                + _mark_superseded_vocal_tags(confirmation_result, locals().get("final_lead"))
            )
        f0_text = f"{median_f0} Hz" if median_f0 is not None else "unavailable"
        # Merge any additional detail not already captured above.
        if confirmation_result:
            conf_mv = parse_multi_voice_fields(confirmation_result)
            for k, v in conf_mv.items():
                if v and (not multi_voice_fields.get(k) or multi_voice_fields.get(k) in ("", "unparsed", "uncertain")):
                    multi_voice_fields[k] = v
        final_lead = _apply_independent_f0_multi_singer(
            final_lead, initial_lead, vocal_objective_report, multi_voice_fields,
            vocal_text=(vocal_result or "") + "\n" + (confirmation_result or ""),
        )
        final_lead = _apply_section_voice_check(final_lead, multi_voice_fields, _svc)

        if (final_lead or initial_lead) == "mixed_leads" and not multi_voice_fields.get("voice_arrangement"):
            multi_voice_fields["voice_arrangement"] = "duet_co_leads"
            if not multi_voice_fields.get("num_distinct_voices"):
                multi_voice_fields["num_distinct_voices"] = "2"
            # Refine multi-singer detection using objective f0 multi-peak hint
            if vocal_objective_report:
                f0_suggest = _refine_multi_singer_from_f0(vocal_objective_report)
                if f0_suggest:
                    for k, v in f0_suggest.items():
                        if v and (not multi_voice_fields.get(k) or multi_voice_fields.get(k) in ("", "unparsed", "uncertain")):
                            multi_voice_fields[k] = v
                    # If f0 suggests two distinct centres, strongly consider mixed_leads
                    if f0_suggest.get("num_distinct_voices") in ("2", "3+"):
                        if final_lead not in ("mixed_leads",):
                            final_lead = "mixed_leads"
                            multi_voice_fields["voice_arrangement"] = f0_suggest.get("voice_arrangement", "duet_co_leads")
                            if not multi_voice_fields.get("co_lead_detail"):
                                multi_voice_fields["co_lead_detail"] = f0_suggest.get("co_lead_detail", "two distinct pitch centres from f0 analysis")
        revised += (
            "\n\nVOCAL DECISION AUDIT (audio-only evidence for vocal age/gender):\n"
            f"- Initial isolated lead profile: {initial_lead or 'unparsed'}\n"
            f"- Confirmation pass: {confirm_lead or 'not run/unparsed'} (confidence={confirm_confidence or 'n/a'})\n"
            f"- Objective median f0: {f0_text}\n"
            f"- FINAL LEAD PROFILE: {final_lead or 'unknown'}\n"
            f"- BACKING PROFILES: {initial_backing or 'uncertain'}\n"
            + _section_voice_audit_line(_svc, final_lead, initial_lead, confirm_lead, vocal_objective_report)
            + ((f"- {_free_form_vocal_lean_line(locals().get('_vft_info'))}"
                + (" (FINAL LEAD PROFILE was set from these descriptions; the structured passes were unparsed/uncertain.)"
                   if (locals().get('_vft_info') or {}).get('applied') else "") + "\n")
               if _free_form_vocal_lean_line(locals().get('_vft_info')) else "")
            + "This is audio-only evidence. If a SINGER IDENTITY RESOLUTION block appears later in this analysis, use that for user-facing singer-identity claims; otherwise use FINAL LEAD PROFILE. Do not override a well-supported combined judgment with pitch impressions alone. If FINAL LEAD PROFILE is unknown/unparsed, do not treat free-text LEAD_PROFILE / LEAD_CATEGORY lines earlier in the vocal pass as a settled gender/age claim — those lines were not accepted as structured final tags. A very high objective median f0 can demote male/adolescent tags to uncertain; when FINAL is uncertain, prefer SINGER IDENTITY RESOLUTION (metadata + cover + pitch constraint) over any earlier free-text male claim."
        )
        mv_block = format_multi_voice_audit(
            multi_voice_fields, final_lead or initial_lead, initial_backing or "uncertain"
        )
        if mv_block:
            revised += "\n\n" + mv_block
        if _svc and _svc.get("mixed"):
            revised += _svc["block"]
        # Presence salvage: failed lyric pass / false listening-model 'no vocals' must not
        # poison the saved analysis or the writer reply.
        _salv_note = locals().get('_early_vocal_salvage') or ''
        if not _salv_note:
            _, _salv_note, _ = _salvage_false_no_vocals(
                vocal_result,
                no_clear_vocals,
                median_f0=median_f0,
                vocal_objective_report=(vocal_objective_report or ''),
                stem_midi_report='',
                initial_lead=(initial_lead or final_lead or ''),
            )
        if _salv_note and 'VOCAL PRESENCE SALVAGE' not in (revised or ''):
            revised += '\n\n' + _salv_note
            if 'VOCALS_PRESENT=yes' not in revised:
                revised += '\nVOCALS_PRESENT=yes  # from objective/structured salvage'
    elif median_f0 is not None and _objective_evidence_of_vocals(
        median_f0=median_f0, vocal_objective_report=(locals().get('vocal_objective_report') or '')):
        # The listening model returned empty/unusable vocal pass but pyin heard a voice
        _ps = (
            "VOCAL PRESENCE SALVAGE: no usable MOSS-Music vocal pass, but objective "
            f"voiced pitch median ~{median_f0} Hz indicates a human voice is present. "
            "Treat VOCALS_PRESENT=yes; age/gender may be uncertain. Do not claim the track "
            "is instrumental solely because lyrics failed or the clip is short."
        )
        if 'VOCAL PRESENCE SALVAGE' not in (revised or ''):
            revised += '\n\n' + _ps + '\nVOCALS_PRESENT=yes  # from objective pitch salvage'
            no_clear_vocals = False

    if ENABLE_VOCAL_OBJECTIVE_REPORT and _vocal_arb.get("instrumental"):
        revised += (
            "\n\nVOCAL PITCH REPORT: not applicable — the recording is instrumental; pitch readings "
            "from the vocals stem are separation residue, not a singer's range."
        )
    elif ENABLE_VOCAL_OBJECTIVE_REPORT:
        pitch_lines = ["VOCAL PITCH REPORT (independent of lead age/gender category):"]
        if median_f0 is None:
            pitch_lines.append("- No reliable voiced pitch detected in the scanned portion.")
        else:
            pitch_lines.append(f"- objective voiced pitch median: {median_f0} Hz")
            if vocal_pitch.get("note"):
                pitch_lines.append(f"- approximate median note: {vocal_pitch['note']}")
            if vocal_pitch.get("low") is not None and vocal_pitch.get("high") is not None:
                _band_label = {
                    "10-90": "practical sung range (10th-90th percentile)",
                    "5-95": "5th-95th percentile range",
                }.get(vocal_pitch.get("band"), "percentile range")
                pitch_lines.append(
                    f"- {_band_label}: {vocal_pitch['low']}-{vocal_pitch['high']} Hz"
                )
            pitch_lines.extend(_voice_type_map_lines(
                median_f0, vocal_pitch.get("low"), vocal_pitch.get("high")))
            pitch_lines.append(
                "- Prefer median + the percentile range above (not absolute min/max) for the main vocal range. Median f0 is also used as a soft constraint on lead age/gender tags (high median can demote an overconfident post-puberty-male claim; it does not invent gender alone). "
                "Do not treat absolute extremes from stem MIDI as the sung range."
            )
        revised += "\n\n" + "\n".join(pitch_lines)

    moss_bpm_val = extract_bpm_from_text(first_pass)
    revised_bpm_val = extract_bpm_from_text(revised)
    if revised_bpm_val is not None:
        moss_bpm_val = revised_bpm_val
    moss_bpm_val, _tempo_echo = _moss_tempo_reading_check(
        moss_bpm_val, revised if revised_bpm_val is not None else first_pass)
    essentia_bpm_val = extract_essentia_bpm(essentia_report) if essentia_report else None
    objective_bpm_val = extract_objective_bpm(objective_report) if objective_report else None
    essentia_median_bpm_val = extract_essentia_median_bpm(essentia_report) if essentia_report else None
    objective_median_bpm_val = extract_objective_median_bpm(objective_report) if objective_report else None
    _tempo_copy = _moss_tempo_copies_detector(moss_bpm_val, essentia_bpm_val, objective_bpm_val,
                                              essentia_median_bpm_val, objective_median_bpm_val)   # dev46
    _bpm_genre_hint = _genre_hint_from_analysis_text(
        revised or first_pass,
        panns_genre_tags,
    )
    final_bpm, bpm_note = reconcile_bpm(
        moss_bpm_val, essentia_bpm_val, objective_bpm_val,
        # Median IBI is an internal consistency check on each beat tracker, not an
        # independent detector. Do not let it outweigh cross-source disagreement.
        essentia_median_bpm=None,
        objective_median_bpm=None,
        genre_hint=_bpm_genre_hint,
    )
    final_bpm, bpm_note = _tempo_listening_check(
        final_bpm, bpm_note, moss_bpm_val, essentia_bpm_val, objective_bpm_val,
        essentia_median_bpm_val, objective_median_bpm_val, _tempo_echo)
    if final_bpm and _tempo_copy is not None:
        final_bpm, bpm_note = _tempo_without_copy(
            final_bpm, bpm_note, moss_bpm_val, _tempo_copy, essentia_bpm_val, objective_bpm_val,
            essentia_median_bpm_val, objective_median_bpm_val, _bpm_genre_hint, _tempo_echo)
    if final_bpm:
        revised += (
            f"\n\nRECOMMENDED TEMPO FOR DISCUSSION: {final_bpm} BPM. "
            f"Reasoning: {_sentence(bpm_note)} "
            "This is the primary tempo to report to the user. "
            f"State it as a concrete figure (e.g. 'about {final_bpm} BPM'). "
            "Do not expand it into a range unless this block itself marks the value as uncertain."
        )

    moss_key_val = extract_key_from_text(first_pass)
    revised_key_val = extract_key_from_text(revised)
    if revised_key_val is not None:
        moss_key_val = revised_key_val
    essentia_key_val, essentia_key_strength = (None, None)
    if essentia_report:
        parsed = extract_essentia_key_from_text(essentia_report)
        if parsed:
            essentia_key_val, essentia_key_strength = parsed
    final_key, key_note = reconcile_key(moss_key_val, essentia_key_val, essentia_key_strength)
    if final_key:
        revised += (
            f"\n\nRECOMMENDED KEY FOR DISCUSSION: {final_key}. "
            f"Reasoning: {key_note} "
            "This is the primary key to report to the user unless the reasoning above marks it uncertain."
        )

    # Numeric dynamics / loudness — prefer the ORIGINAL local file (full duration,
    # native rate/channels) so LUFS is not biased by the mono 22.05 kHz DSP WAV.
    # ReplayGain/RGAD tags are intentionally ignored (raw PCM decode only).
    crest_db_val = None
    crest_src = "objective (librosa) crest-factor proxy"
    lufs_val = lra_val = loudness_src = None
    try:
        _dyn_path = None
        if (
            track_path
            and not str(track_path).startswith(("http://", "https://"))
            and os.path.exists(track_path)
        ):
            _dyn_path = audio_source
        elif track_path in dsp_temp_files:
            _dyn_path = dsp_temp_files[track_path]
        elif dsp_path is not None:
            _dyn_path = dsp_path
        if _dyn_path:
            crest_db_val = measure_crest_factor_db(_dyn_path)
            if crest_db_val is not None and _dyn_path == audio_source:
                # Measured on the original file, so it can read a little higher
                # than the "crest factor" line of the signal-processing report,
                # which comes from the mono, downsampled analysis copy.
                crest_src = (
                    "crest factor measured on the original file at full resolution "
                    "(supersedes the figure from the mono, downsampled analysis copy)"
                )
            lufs_val, lra_val, loudness_src = measure_ebur128_loudness(_dyn_path)
    except Exception:
        crest_db_val = None
    if crest_db_val is None:
        crest_src = "objective (librosa) crest-factor proxy"
        crest_db_val = extract_crest_db_from_text(objective_report) if objective_report else None
    if crest_db_val is None and essentia_report:
        crest_db_val = extract_crest_db_from_text(essentia_report)
        if crest_db_val is not None:
            crest_src = "Essentia crest-factor proxy"
    dyn_block = format_recommended_dynamics_block(
        crest_db=crest_db_val,
        source_note=crest_src,
        lufs=lufs_val,
        lra=lra_val,
        loudness_source=loudness_src,
    )
    if dyn_block:
        revised += dyn_block
    else:
        revised += (
            "\n\nRECOMMENDED DYNAMICS FOR DISCUSSION: unavailable "
            "(could not compute EBU R128 loudness or crest-factor proxy from this file). "
            "Do not invent a dB/LUFS figure; fall back only to qualitative production notes."
        )

    measurement_parts = []
    if objective_report:
        measurement_parts.append(objective_report)
    if vocal_objective_report:
        measurement_parts.append(vocal_objective_report)
    if essentia_report:
        measurement_parts.append(essentia_report)
    if measurement_parts:
        revised += (
            "\n\n[Independent signal-processing report:\n"
            + "\n\n".join(measurement_parts)
            + "\nThis is computed directly from the audio (librosa/Essentia), not from the model above — "
            "use it only for tempo/beat, key/key strength when explicitly reported, timbre, "
            "low/mid/high element activity, dynamic range / loudness (crest-factor proxy), "
            "and vocal pitch/formant proxies. "
            "Do NOT use it to infer or revise GENRE."
        )

    if genre_mood_report:
        revised += "\n\n" + genre_mood_report
    try:
        _rec_g, _rec_g_note = reconcile_genre(revised, panns_genre_tags, discogs_styles)
        revised += format_recommended_genre_block(_rec_g, _rec_g_note)
        # Genre-conditioned Essentia key refinement (edma vs temperley/krumhansl).
        if _rec_g and dsp_path is not None:
            _prev_k = extract_essentia_key_from_text(essentia_report) if essentia_report else None
            _pk, _ps = (None, None)
            if _prev_k:
                _pk, _ps = _prev_k if isinstance(_prev_k, tuple) else (_prev_k, None)
            _nk, _ns, _nnote = refine_essentia_key_with_genre(
                dsp_path, _rec_g, previous_key=_pk, previous_strength=_ps
            )
            if _nnote:
                revised += f"\n\nKEY PROFILE REFINEMENT: {_nnote}"
    except Exception:
        pass
    if discogs_style_report:
        revised += "\n\n" + discogs_style_report      # dev42: after the recommended genre

    stem_midi_report = ""
    _clap_prefetched = None
    if _tag_task is not None:
        _ok, _clap_prefetched = _tag_task.result()
        _tag_task = None
    if ENABLE_STEM_MIDI:
        try:
            if not _OMNIZART_PREFETCH.get("paths"):
                _get_omnizart()
            else:
                # Fail fast exactly as before when the background process
                # could not transcribe anything.
                if not any(v[0] for v in _omnizart_prefetch_results().values()):
                    _get_omnizart()
            if precomputed_stems:
                # Reuse the separation already run above for vocal pitch
                # isolation -- avoids a second, redundant Demucs pass.
                stems = precomputed_stems
            else:
                if track_path in stem_temp_files:
                    stem_wav = stem_temp_files[track_path]
                else:
                    status("Preparing stereo WAV for Demucs/MIDI...")
                    stem_wav = convert_to_wav_for_stems(
                        audio_source,
                        sample_rate=44100,
                        channels=2,
                        max_seconds=STEM_MIDI_MAX_SECONDS,
                    )
                    stem_temp_files[track_path] = stem_wav
                status("Running Demucs 6s stem separation (this can be slow)...")
                out_dir = tempfile.mkdtemp(prefix="demucs_")
                demucs_out_dirs.append(out_dir)
                stems = run_demucs_stems(
                    stem_wav, out_dir, shifts=DEMUCS_SHIFTS_DEEP if DEEP_MODE else DEMUCS_SHIFTS_FAST,
                )
                if ENABLE_JAMENDO_INSTRUMENTS and stems:
                    jamendo_add_stems(track_path, stems)                         # dev3
            if not stems:
                stem_midi_report = "STEM MIDI REPORT unavailable: Demucs did not produce expected stems."
            else:
                whole_mix_tags = None
                if ENABLE_WHOLE_MIX_INSTRUMENT_TAGGING and dsp_path is not None:
                    try:
                        status("Tagging instruments on the full (un-separated) mix...")
                        whole_mix_tags = tag_full_mix_instruments(dsp_path)
                    except Exception as e:
                        print(f"  (whole-mix instrument tagging skipped: {e})")
                        whole_mix_tags = None
                status("Running Omnizart on each separated stem...")
                _total_duration = _resolve_total_duration(
                    track_path=audio_source, dsp_path=dsp_path, objective_report=objective_report
                )
                # Reuse the section map resolved before the model unload. Only
                # fall back to parsing the model's own STRUCTURE= guess if that
                # pass produced nothing at all.
                if not _structure_sections:
                    _structure_sections, _bnd_pre, _ = _finalize_structure_sections(
                        revised or first_pass,
                        objective_report=objective_report,
                        lyric_chunks=locals().get("_lyric_chunks"),
                        total_duration=_total_duration,
                    )

                # Raw per-stem note events, kept so the melodic analysis and the
                # saved .json can work from actual notes rather than prose.
                _stem_notes_raw = {}
                # Per-stem energy/density, needed below to cross-check a
                # missing guitar tag against the actual Demucs guitar stem.
                _stem_activity_raw = []
                # Cross-stem instrument agreement (label -> [stem, ...]).
                _stem_convergence = {}
                # sections=None on purpose: the section-keyed blocks
                # (INTENSITY BY SECTION, MELODY PER SECTION) are built after the
                # refinement pass below, from the FINAL map, by
                # build_section_keyed_stem_blocks(). Generating them here would
                # bake in the pre-refinement section names and timings, leaving
                # two contradictory section lists in the evidence.
                stem_midi_report = build_omnizart_summaries(
                    stems, whole_mix_tags=whole_mix_tags,
                    bpm=final_bpm, sections=None, key=final_key,
                    notes_out=_stem_notes_raw,
                    mix_audio_path=dsp_path,
                    activity_out=_stem_activity_raw,
                    convergence_out=_stem_convergence,
                    instrumental=bool(_vocal_arb.get("instrumental")),
                    listening_text=first_pass,
                )
                # dev19 TEMPO_SNARE_TIEBREAK: the tempo is settled from the
                # drums stem's snare, then the stem report is rebuilt at it
                # (the transcriptions are cached, so only the summaries rerun).
                _tb = _tempo_snare_tiebreak(revised, final_bpm, (_stem_notes_raw or {}).get("drums"))
                if _tb and int(_tb[1]) == int(final_bpm):
                    revised = _tb[0]       # pre60: a note only (TEMPO_SNARE_HALFTIME_GUARD); the tempo stands
                elif _tb:
                    try:
                        _sn2, _sa2, _sc2 = {}, [], {}
                        _rep2 = build_omnizart_summaries(
                            stems, whole_mix_tags=whole_mix_tags,
                            bpm=_tb[1], sections=None, key=final_key,
                            notes_out=_sn2,
                            mix_audio_path=dsp_path,
                            activity_out=_sa2,
                            convergence_out=_sc2,
                            instrumental=bool(_vocal_arb.get("instrumental")),
                            listening_text=first_pass,
                        )
                        stem_midi_report, _stem_notes_raw = _rep2, _sn2
                        _stem_activity_raw, _stem_convergence = _sa2, _sc2
                        revised, final_bpm = _tb
                    except Exception as _tb_err:
                        print(f"  (tempo snare check skipped: {_tb_err})")
                # Melody/intensity are appended after refinement (see below).
                if whole_mix_tags is not None:
                    whole_mix_report = build_whole_mix_instrument_report(
                        dsp_path, tags=whole_mix_tags
                    )
                    if whole_mix_report:
                        stem_midi_report = stem_midi_report + "\n\n" + whole_mix_report
                        try:
                            _gan = guitar_absence_note_from_tags(
                                locals().get("whole_mix_tags"), whole_mix_report,
                                mix_audio_path=dsp_path,
                                stem_notes=locals().get("_stem_notes_raw"),
                                activity_metas=locals().get("_stem_activity_raw"),
                                cross_stem_agreement=locals().get("_stem_convergence"),
                            )
                            if _gan:
                                stem_midi_report = stem_midi_report + "\n\n" + _gan
                        except Exception:
                            pass
                _release_omnizart_memory()
        except Exception as e:
            print(f"  (stem MIDI skipped/unavailable: {e})")
            stem_midi_report = f"STEM MIDI REPORT unavailable: {e}"
    stem_midi_report = stem_midi_report or "STEM MIDI REPORT not run (disabled)."
    revised += "\n\n" + stem_midi_report
    revised, final_key = _apply_key_note_check(revised, final_key, locals().get("_stem_notes_raw") or {})
    revised, final_key = _apply_key_centre_check(revised, final_key, locals().get("_stem_notes_raw") or {})
    revised, final_key = _apply_key_fit_check(revised, final_key, locals().get("_stem_notes_raw") or {})
    revised, final_key = _apply_key_scale_check(revised, final_key, locals().get("_stem_notes_raw") or {})  # dev26
    revised, final_key = _apply_key_change_check(revised, final_key, locals().get("_stem_notes_raw") or {})  # dev28
    # dev55: second pitch reading from the vocal notes (see VOCAL_NOTE_PITCH_LINE).
    _vnp_line = "" if _vocal_arb.get("instrumental") else _vocal_note_pitch_line(locals().get("_stem_notes_raw"))
    _pyin_conf = "" if _vocal_arb.get("instrumental") else _pyin_reading_confidence_note(vocal_objective_report)
    if _vnp_line or _pyin_conf:
        revised += "\n\n" + "\n".join(x for x in (_pyin_conf, _vnp_line) if x)
    _oct_err = _pitch_octave_error(revised)    # dev71
    revised = _pitch_octave_check(revised)
    revised = _pitch_stuck_check(revised)    # dev73
    revised = _pitch_low_confidence_check(revised)

    # Presence salvage, second pass. The two _salvage_false_no_vocals() calls
    # above ran with stem_midi_report hardcoded to "" -- Demucs/Omnizart had
    # not run yet at that point in the pipeline, so they could only ever see
    # median_f0 / vocal_objective_report. If the listening model's vocal pass never wrote an
    # explicit VOCALS_PRESENT field at all (not "no", just absent -- a real,
    # separate listening-model failure mode from the explicit-"no" case those calls
    # already handle) and neither earlier check caught it, the track can
    # reach the writer with NO explicit vocals flag whatsoever, even when the
    # vocal stem's own transcription just measured a note a fraction of a
    # second into the track. This is the strongest evidence this pipeline
    # produces (see EARLIEST VOCAL ONSET above), and now that it exists,
    # this is the first point it CAN be checked.
    revised += _stem_onset_vocal_salvage(vocal_result, revised, stem_midi_report)

    # Add authoritative structure block
    total_duration = _resolve_total_duration(
        track_path=audio_source, dsp_path=dsp_path, objective_report=objective_report
    )

    _lyric_groups_for_structure = None
    try:
        _lc = locals().get("_lyric_chunks")
        _lyric_groups_for_structure = _detect_lyric_repetitions(_lc) if _lc else None
    except Exception:
        _lyric_groups_for_structure = None

    # ------------------------------------------------------------------
    # FINAL SECTION MAP — the audio model leads, measurement sharpens.
    # ------------------------------------------------------------------
    # The model's own STRUCTURE=[...] is the primary source: it listened to the
    # whole recording and understands song form. The acoustic pass is used to
    # pull each of its boundaries onto a detected change point or bar line
    # (precision), and lyric repetition to correct names where the same words
    # were labelled two different ways. The DSP map replaces it only when the
    # model produced nothing usable.
    _moss_sections = []
    try:
        _moss_sections = extract_structure_sections(revised or "")
    except Exception:
        _moss_sections = []

    # pre59: SongFormer's sections, to check the map (SONGFORMER_CORRECT_STRUCTURE).
    _songformer_result = None
    if SONGFORMER_COMMAND and SONGFORMER_CORRECT_STRUCTURE:
        _songformer_result = _songformer_scan_file(
            track_path if os.path.isfile(str(track_path)) else audio_source, total_duration)
    structure_sections, _structure_source = _resolve_final_structure(
        _moss_sections,
        **({"songformer_sections": _songformer_result["sections"],
            "songformer_dance": _songformer_is_dance(locals().get("discogs_styles"))}
           if _songformer_result else {}),
        dsp_sections=_structure_sections,
        lyric_groups=_lyric_groups_for_structure,
        lyric_chunks=locals().get("_lyric_chunks"),
        total_duration=total_duration,
        grid=_struct_grid,
        stem_notes=locals().get("_stem_notes_raw"),
        bpm=locals().get("final_bpm"),
        vocal_presence_unresolved=("VOCAL PRESENCE VERIFICATION: UNRESOLVED" in revised),
        instrumental=bool(_vocal_arb.get("instrumental")),
        title=(metadata or {}).get("title") if isinstance(metadata, dict) else None,
        tag_lyrics=(metadata or {}).get("lyrics") if isinstance(metadata, dict) else None,
        **({"lyrics_language": _WHISPER_LANG_STATE.get("record")}
           if STRUCTURE_DROP_UNRELIABLE_TRANSCRIPT and WHISPER_HOLD_LANGUAGE and _WHISPER_LANG_STATE.get("record")
           else {}),                                                                # 0.7.1-pre21
    )
    _structure_replay = _structure_replay_payload(
        _struct_grid, locals().get("_lyric_chunks"), _lyric_groups_for_structure,
        total_duration, bool(_vocal_arb.get("instrumental")),
    ) if STRUCTURE_SAVE_REPLAY else None
    if WHISPER_HOLD_LANGUAGE and isinstance(_structure_replay, dict) and _WHISPER_LANG_STATE.get("record"):
        _structure_replay["lyrics_language"] = dict(_WHISPER_LANG_STATE["record"])     # 0.7.1-pre21
    if structure_sections:
        boundary_seconds = [float(s["start"]) for s in structure_sections[1:]]
        revised = _overwrite_structure_field(revised, structure_sections)
        revised = _correct_stated_duration(revised, structure_sections[-1].get("end"))
        status_done(
            f"Section map resolved ({_structure_source}, "
            f"{len(structure_sections)} sections)"
        )
    else:
        structure_sections, boundary_seconds, _lg = _finalize_structure_sections(
            revised,
            objective_report=objective_report,
            stem_midi_report=locals().get("stem_midi_report") or "",
            lyric_chunks=locals().get("_lyric_chunks"),
            total_duration=total_duration,
        )
        _lyric_groups_for_structure = _lg or _lyric_groups_for_structure

    # Section-keyed stem evidence is generated here, once, from the FINAL map,
    # so the structure, melody, intensity and groove blocks can never disagree
    # about section names or timings.
    if structure_sections:
        try:
            _sk = build_section_keyed_stem_blocks(
                locals().get("stems") or {}, structure_sections,
                locals().get("_stem_notes_raw") or {},
                bpm=final_bpm, key=final_key,
                instrumental=bool(_vocal_arb.get("instrumental")),
                lyric_chunks=locals().get("_lyric_chunks"),
            )
            if _sk:
                stem_midi_report = (
                    (locals().get("stem_midi_report") or "").rstrip() + "\n\n" + _sk
                )
                # `revised` was given a COPY of stem_midi_report above, and
                # Python strings are immutable, so the rebinding on the line
                # above never reached the analysis the writer actually sees.
                # Every section-keyed block -- per-stem INTENSITY BY SECTION,
                # the per-section drum groove, and MELODY PER SECTION -- was
                # computed correctly and then silently discarded, which is why
                # questions about a specific verse or chorus could only ever be
                # answered from the whole-track averages.
                revised += "\n\n" + _sk
        except Exception as e:
            print(f"  (section-keyed stem blocks skipped: {e})")
        _cpb = build_chord_progression_block(chord_segments, structure_sections, final_key, final_bpm)   # dev45
        if _cpb:
            revised += "\n\n" + _cpb

    structure_block = format_recommended_structure_block(
        structure_sections,
        objective_boundaries=boundary_seconds,
        total_duration=total_duration,
        lyric_groups=_lyric_groups_for_structure,
        objective_report=objective_report,
    )
    if structure_block:
        revised += "\n\n" + structure_block

    # Whole-mix tags are folded into the stem report above when both paths run.
    # If stem MIDI was disabled but whole-mix tagging is still on, run it alone.
    if (
        ENABLE_WHOLE_MIX_INSTRUMENT_TAGGING
        and dsp_path is not None
        and "WHOLE-MIX INSTRUMENT TAGS" not in (stem_midi_report or "")
    ):
        status("Tagging instruments on the full (un-separated) mix...")
        try:
            whole_mix_report = build_whole_mix_instrument_report(dsp_path)
            if whole_mix_report:
                revised += "\n\n" + whole_mix_report
                _gan = ""
                try:
                    _gan = guitar_absence_note_from_tags(
                        locals().get("whole_mix_tags"), whole_mix_report,
                        mix_audio_path=dsp_path,
                        stem_notes=locals().get("_stem_notes_raw"),
                        activity_metas=locals().get("_stem_activity_raw"),
                        cross_stem_agreement=locals().get("_stem_convergence"),
                    )
                except Exception:
                    _gan = ""
                if _gan:
                    revised += "\n\n" + _gan
        except Exception as e:
            print(f"  (whole-mix instrument tagging skipped: {e})")

    # Singer identity resolution is intentionally NOT done here — see
    # resolve_identity_and_finalize(). Just precompute the plain-text
    # evidence summary it needs, then release the heavy models.
    vocal_audit_for_resolution = None
    if (ENABLE_SINGER_IDENTITY_RESOLUTION and VOCAL_PRESENCE_UNRESOLVED_MARKER not in revised and not _vocal_arb.get("instrumental")
            and (_has_real_metadata(metadata) or cover_observations or _scan_user_photo_given()
                 or _audio_only_singer_identity("FINAL LEAD PROFILE: " + str(final_lead or "")))):
        f0_text_res = f"{median_f0} Hz" if median_f0 is not None else "unavailable"
        vocal_audit_for_resolution = (
            f"Initial isolated lead profile: {initial_lead or 'unparsed'}\n"
            f"Confirmation pass: {confirm_lead or 'not run/unparsed'} (confidence={confirm_confidence or 'n/a'})\n"
            f"FINAL LEAD PROFILE: {final_lead or 'unknown'}\n"
            f"BACKING PROFILES: {initial_backing or 'uncertain'}\n"
            f"objective median f0: {f0_text_res}"
        )
        _vft_line = _free_form_vocal_lean_line(locals().get("_vft_info"))
        if _vft_line:
            vocal_audit_for_resolution += "\n" + _vft_line
        for _extra in (_pyin_reading_confidence_note(vocal_objective_report),
                       _vocal_note_pitch_line(locals().get("_stem_notes_raw")),
                       _pitch_octave_audit_line(locals().get("_oct_err"))):
            if _extra:
                vocal_audit_for_resolution += "\n" + _extra
        if no_clear_vocals:
            vocal_audit_for_resolution = "No clear vocals detected.\n" + vocal_audit_for_resolution
        _svl = _section_voice_identity_line(locals().get("_svc"))
        if _svl:
            vocal_audit_for_resolution += "\n" + _svl

    # Always release MOSS-Music/Omnizart/Ollama here, not just when identity
    # resolution is enabled — this is the point where this process's peak
    # (MOSS-Music + Demucs + Omnizart all having been resident) has passed, so
    # freeing what we can now minimises what the OS has to reclaim on exit.
    _release_heavy_analysis_memory_before_identity()

    # dev54: observational per-stage timing, to stderr so stdout is untouched.
    if PRINT_STAGE_TIMING:
        try:
            import io as _io, contextlib as _cl
            _buf = _io.StringIO()
            with _cl.redirect_stdout(_buf):
                _print_stage_log()
            sys.stderr.write(_default_color_lines(_buf.getvalue(), sys.stderr))   # pre62: yellow like the rest
            sys.stderr.flush()
        except Exception:
            pass

    if _tag_task is not None:
        _ok, _clap_prefetched = _tag_task.result()
    _clear_omnizart_prefetch()
    return {
        "clap_prefetched": locals().get("_clap_prefetched"),
        "track_path": track_path,
        **({"_video_audio_duration": total_duration} if video_sources else {}),
        "analysis": revised,
        "corrections": {},
        "metadata": metadata,
        "cover_observations": cover_observations,
        "cover_bytes": cover_bytes_for_save,
        "cover_mime": cover_mime,
        "stem_midi_report": stem_midi_report,
        # Carried forward so the saved .json can hold the real note events and
        # the resolved section map, rather than only the prose written above.
        "stem_notes": locals().get("_stem_notes_raw") or {},
        "structure_sections": structure_sections,
        **({"structure_replay": _structure_replay} if STRUCTURE_SAVE_REPLAY else {}),
        **({"songformer_result": _songformer_result} if _songformer_result else {}),   # pre59
        "bpm": final_bpm,
        "key": final_key,
        "discogs_styles": _discogs_styles_for_save_list(discogs_styles),     # dev42
        "chord_segments": _chord_segments_for_save(chord_segments),          # dev45
        # 0.6.2-dev3: the MTG-Jamendo tags, ready to save, so a batch save made
        # in a fresh process (--batch-one-finish) still has them.
        **({"jamendo_instruments": _jamendo_for_save_safe(track_path, structure_sections)}
           if ENABLE_JAMENDO_INSTRUMENTS else {}),
        # 0.7-pre3: the listening details, saved only (ENABLE_LISTENING_DETAILS).
        **({"listening_details": locals().get("_listening_details")}
           if locals().get("_listening_details") else {}),
        # 0.7-pre5: bell/chime/mallet tagger readings, saved only (TUNED_PERCUSSION_PROBE).
        **({"tuned_percussion": _tuned_percussion_for_save(locals().get("dsp_path"),
                                                               locals().get("precomputed_stems"))}
           if TUNED_PERCUSSION_PROBE else {}),
        "vocal_audit_for_resolution": vocal_audit_for_resolution,
        # dev64: for the save-time layout (REPORT_FIXED_LAYOUT); carried in the
        # batch staging file because stage 2 runs in a fresh process.
        "moss_first_pass": first_pass if REPORT_FIXED_LAYOUT else None,
        **({"_genre_clarifications": _genre_clarifications,
            "_genre_clarification_panns": [[str(tag[0]), float(tag[1])]
                                           for tag in (panns_genre_tags or [])]}
           if _genre_clarifications else {}),
        "final_lead": final_lead,
        "initial_backing": initial_backing,
        "vocal_result_present": bool(vocal_result) and not _vocal_arb.get("instrumental"),
        "instrumental": bool(_vocal_arb.get("instrumental")),
        "_objective_report": objective_report,
    }


def run_fresh_track_analysis(
    track_path,
    *,
    audio_temp_files,
    dsp_temp_files,
    stem_temp_files,
    demucs_out_dirs,
):
    """Back-compat wrapper: heavy analysis + identity resolution in one call,
    both stages in THIS process. Used by the non-isolated in-process batch
    fallback (BATCH_ISOLATE_PER_TRACK = False) and anywhere splitting into
    two OS processes isn't applicable. The (default) isolated batch path
    instead runs run_fresh_track_analysis_heavy() and
    resolve_identity_and_finalize() as two separate subprocess stages — see
    run_batch_one_analyze() / run_batch_one_finish() — so identity
    resolution's Ollama model load never has to coexist with this stage's
    stranded MOSS-Music/Demucs/Omnizart memory.
    """
    heavy = run_fresh_track_analysis_heavy(
        track_path,
        audio_temp_files=audio_temp_files,
        dsp_temp_files=dsp_temp_files,
        stem_temp_files=stem_temp_files,
        demucs_out_dirs=demucs_out_dirs,
    )
    return resolve_identity_and_finalize(heavy)


def run_batch_one_track(track_path):
    """Analyse a single local audio file (heavy + identity, ONE process) and
    write JSON+cover into SAVE_DIR.

    Used when BATCH_SPLIT_IDENTITY_PROCESS is False. When it's True
    (default), the isolated batch path instead uses run_batch_one_analyze()
    + run_batch_one_finish() as two separate subprocess stages — see
    _batch_run_isolated().
    """
    track_path = os.path.abspath(os.path.expanduser(track_path))
    if not os.path.exists(track_path):
        raise FileNotFoundError(f"File not found: {track_path}")

    audio_temp_files = {}
    dsp_temp_files = {}
    stem_temp_files = {}
    demucs_out_dirs = []

    try:
        result = run_fresh_track_analysis(
            track_path,
            audio_temp_files=audio_temp_files,
            dsp_temp_files=dsp_temp_files,
            stem_temp_files=stem_temp_files,
            demucs_out_dirs=demucs_out_dirs,
        )
        save_name = _batch_save_basename(track_path, metadata=result.get("metadata"))
        out_path, cover_path = save_song_data(
            save_name,
            result["track_path"],
            result["analysis"],
            result.get("corrections") or {},
            metadata=result.get("metadata"),
            cover_observations=result.get("cover_observations"),
            singer_identity=result.get("singer_identity"),
            cover_bytes=result.get("cover_bytes"),
            cover_mime=result.get("cover_mime") or "image/jpeg",
            preserve_audio_basename=True,
            stem_midi=result.get("stem_notes"),
            structure_sections=result.get("structure_sections"),
            structure_replay=result.get("structure_replay"),
            bpm=result.get("bpm"),
            key=result.get("key"),
            clap_embedding=result.get("clap_embedding"),
            discogs_styles=result.get("discogs_styles"),
            chord_segments=result.get("chord_segments"),
            jamendo_instruments=result.get("jamendo_instruments"),
            listening_details=result.get("listening_details"),
            tuned_percussion=result.get("tuned_percussion"),
            songformer_result=result.get("songformer_result"),
        )
        print(
            f"  ✓ saved {os.path.basename(out_path)}"
            + (f" + {os.path.basename(cover_path)}" if cover_path else "")
        )
        return out_path
    finally:
        try:
            _settle_background_tasks()
        except Exception:
            pass
        try:
            _cleanup_track_temp_files(
                track_path, audio_temp_files, dsp_temp_files, stem_temp_files, demucs_out_dirs
            )
        except Exception:
            pass
        try:
            _aggressive_memory_cleanup()
        except Exception:
            pass


def run_batch_one_analyze(track_path, staging_path):
    """Stage 1 of the split-process batch pipeline (BATCH_SPLIT_IDENTITY_PROCESS):
    run the heavy audio analysis (MOSS-Music/Demucs/Omnizart/Essentia/PANNs) and dump
    the result to staging_path as JSON, then this process exits. Deliberately
    does NOT resolve singer identity here — see run_batch_one_finish(), which
    runs as a fresh, model-free process so it can't inherit any of this
    stage's stranded torch/TF/MPS memory.
    """
    track_path = os.path.abspath(os.path.expanduser(track_path))
    if not os.path.exists(track_path):
        raise FileNotFoundError(f"File not found: {track_path}")

    audio_temp_files = {}
    dsp_temp_files = {}
    stem_temp_files = {}
    demucs_out_dirs = []

    try:
        heavy = run_fresh_track_analysis_heavy(
            track_path,
            audio_temp_files=audio_temp_files,
            dsp_temp_files=dsp_temp_files,
            stem_temp_files=stem_temp_files,
            demucs_out_dirs=demucs_out_dirs,
        )
        payload = dict(heavy)
        cover_bytes = payload.pop("cover_bytes", None)
        payload["cover_bytes_b64"] = (
            base64.b64encode(cover_bytes).decode("ascii") if cover_bytes else None
        )
        with open(staging_path, "w", encoding="utf-8") as f:
            json.dump(payload, f)
    finally:
        try:
            _settle_background_tasks()
        except Exception:
            pass
        try:
            _cleanup_track_temp_files(
                track_path, audio_temp_files, dsp_temp_files, stem_temp_files, demucs_out_dirs
            )
        except Exception:
            pass
        try:
            _aggressive_memory_cleanup()
        except Exception:
            pass


def run_batch_one_finish(staging_path):
    """Stage 2 of the split-process batch pipeline: load stage 1's JSON,
    resolve singer identity (a plain-text Ollama call — no MOSS-Music/
    Demucs/Omnizart/Essentia involved), and save. Runs in a fresh process
    that starts with a clean memory baseline, regardless of how much of
    stage 1's memory the OS was slow to reclaim.
    """
    with open(staging_path, "r", encoding="utf-8") as f:
        heavy = json.load(f)

    cover_b64 = heavy.pop("cover_bytes_b64", None)
    heavy["cover_bytes"] = base64.b64decode(cover_b64) if cover_b64 else None
    _remember_moss_first_pass(heavy.get("track_path"), heavy.pop("moss_first_pass", None))

    result = resolve_identity_and_finalize(heavy)

    save_name = _batch_save_basename(result["track_path"], metadata=result.get("metadata"))
    out_path, cover_path = save_song_data(
        save_name,
        result["track_path"],
        result["analysis"],
        result.get("corrections") or {},
        metadata=result.get("metadata"),
        cover_observations=result.get("cover_observations"),
        singer_identity=result.get("singer_identity"),
        cover_bytes=result.get("cover_bytes"),
        cover_mime=result.get("cover_mime") or "image/jpeg",
        preserve_audio_basename=True,
        stem_midi=result.get("stem_notes"),
        structure_sections=result.get("structure_sections"),
        structure_replay=result.get("structure_replay"),
        bpm=result.get("bpm"),
        key=result.get("key"),
        clap_embedding=result.get("clap_embedding"),
        discogs_styles=result.get("discogs_styles"),
            chord_segments=result.get("chord_segments"),
        jamendo_instruments=result.get("jamendo_instruments"),
        listening_details=result.get("listening_details"),
        tuned_percussion=result.get("tuned_percussion"),
        songformer_result=result.get("songformer_result"),
    )
    print(
        f"  ✓ saved {os.path.basename(out_path)}"
        + (f" + {os.path.basename(cover_path)}" if cover_path else "")
    )
    return out_path


def _batch_stage_timeout():
    """dev2: BATCH_TRACK_TIMEOUT_S as a positive number of seconds, or None."""
    try:
        v = globals().get("BATCH_TRACK_TIMEOUT_S")
        return float(v) if v is not None and float(v) > 0 else None
    except (TypeError, ValueError):
        return None


def _batch_stage_run(cmd, env):
    """Run one batch stage like subprocess.run(cmd, env=env, timeout=BATCH_TRACK_TIMEOUT_S).
    pre64 BATCH_STALL_RETRY: also stopped when the listening model has produced nothing for
    BATCH_STALL_MINUTES; the result then has .stalled = True. Off: subprocess.run as before."""
    if not BATCH_STALL_RETRY:
        return subprocess.run(cmd, env=env, timeout=_batch_stage_timeout())
    fd, stall_path = tempfile.mkstemp(prefix="musiclyse_stall_", suffix=".txt")
    os.close(fd)
    env = dict(env, MUSICLYSE_STALL_FILE=stall_path)
    try:
        limit = max(60.0, float(BATCH_STALL_MINUTES) * 60.0)
    except (TypeError, ValueError):
        limit = 600.0
    timeout = _batch_stage_timeout()
    started = time.time()
    proc = subprocess.Popen(cmd, env=env)
    try:
        while True:
            try:
                return subprocess.CompletedProcess(cmd, proc.wait(timeout=5))
            except subprocess.TimeoutExpired:
                pass
            if timeout is not None and time.time() - started > timeout:
                proc.kill()
                proc.wait()
                raise subprocess.TimeoutExpired(cmd, timeout)
            try:
                with open(stall_path, encoding="utf-8") as fh:
                    state, stamp = fh.read().split()
                stamp = float(stamp)
            except (OSError, ValueError):
                continue
            if state == "gen" and time.time() - stamp > limit:
                proc.terminate()
                try:
                    proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
                result = subprocess.CompletedProcess(cmd, proc.returncode if proc.returncode is not None else -9)
                result.stalled = True
                return result
    except BaseException:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
        raise
    finally:
        for path in (stall_path, stall_path + ".tmp"):
            try:
                os.remove(path)
            except OSError:
                pass


def _batch_stall_note():
    print(f"  \u21bb the listening model produced nothing for {float(BATCH_STALL_MINUTES):g} minutes "
          f"(it had hung); restarting this track once in a fresh process...")
    try:
        time.sleep(max(0.0, float(BATCH_PAUSE_BETWEEN_TRACKS_S)))
    except Exception:
        pass


def _batch_run_isolated(track_path):
    """Run one track through the isolated batch pipeline; returns (ok: bool, detail).

    When BATCH_SPLIT_IDENTITY_PROCESS is True (default), this spawns TWO
    fresh Python processes in sequence — one for the heavy MOSS-Music/Demucs/
    Omnizart/Essentia analysis, one for singer-identity resolution + save —
    instead of one process doing both. The heavy stage's process exits
    completely between the two, which is the only fully reliable way to
    reset MPS/TF residual memory (in-process unload/gc/cache-clear is
    best-effort and can leave allocations stranded); the identity/save
    stage then starts from a clean baseline instead of asking Ollama to
    load the writer model on top of whatever the heavy stage couldn't
    fully release, which is what was causing "child exited -9" specifically
    at the "Resolving singer identity" step. Set BATCH_SPLIT_IDENTITY_PROCESS
    = False to fall back to the single-process-per-track behaviour.
    """
    script = os.path.abspath(__file__)
    env = os.environ.copy()
    # Child uses ephemeral Ollama loads (keep_alive=0) and unloads the writer
    # around heavy analysis. Parent chat history stays in this process's RAM.
    env["MUSICLYSE_BATCH_ONE"] = "1"
    env["MUSICLYSE_IN_BATCH"] = "1"
    try:
        # Free parent-side Ollama weights so the child can claim RAM for the listening model.
        if globals().get("BATCH_UNLOAD_OLLAMA", True):
            try:
                ollama_unload_model()
            except Exception:
                pass

        if globals().get("BATCH_SPLIT_IDENTITY_PROCESS", True):
            staging_fd, staging_path = tempfile.mkstemp(
                prefix="musiclyse_batch_", suffix=".json"
            )
            os.close(staging_fd)
            try:
                analyze_cmd = [sys.executable, script, "--batch-one-analyze", track_path, staging_path]
                analyze_result = _batch_stage_run(analyze_cmd, env)
                if getattr(analyze_result, "stalled", False):          # pre64 BATCH_STALL_RETRY
                    _batch_stall_note()
                    analyze_result = _batch_stage_run(analyze_cmd, env)
                    if getattr(analyze_result, "stalled", False):
                        return False, "the listening model hung twice"
                # dev65: a crash (killed by a signal: negative exit code, e.g.
                # -6 abort, -11 segfault) is retried once in a fresh process.
                # Decoding is greedy, so a retry gives the same result.
                if (analyze_result.returncode < 0
                        and globals().get("BATCH_RETRY_CRASHED_ANALYSIS", True)):
                    print(f"  ↻ analysis stage crashed (exit {analyze_result.returncode}); "
                          f"retrying this track once in a fresh process...")
                    try:
                        time.sleep(max(0.0, float(BATCH_PAUSE_BETWEEN_TRACKS_S)))
                    except Exception:
                        pass
                    analyze_result = _batch_stage_run(analyze_cmd, env)
                    if getattr(analyze_result, "stalled", False):
                        return False, "the listening model hung"
                if analyze_result.returncode != 0:
                    return False, f"analysis stage exited {analyze_result.returncode}"

                # The analysis process has now fully exited, so the OS has
                # reclaimed 100% of its memory (unlike in-process cleanup,
                # which is best-effort). This finish stage re-imports the
                # module (cheap — no big model weights loaded) but starts
                # from that clean baseline before Ollama loads the writer.
                finish_result = subprocess.run(
                    [sys.executable, script, "--batch-one-finish", staging_path],
                    env=env,
                    timeout=_batch_stage_timeout(),
                )
                if finish_result.returncode != 0:
                    return False, f"identity/save stage exited {finish_result.returncode}"
                return True, "ok"
            finally:
                try:
                    os.remove(staging_path)
                except Exception:
                    pass

        # Single-process fallback (BATCH_SPLIT_IDENTITY_PROCESS = False).
        one_cmd = [sys.executable, script, "--batch-one", track_path]
        result = _batch_stage_run(one_cmd, env)
        if getattr(result, "stalled", False):                      # pre64 BATCH_STALL_RETRY
            _batch_stall_note()
            result = _batch_stage_run(one_cmd, env)
            if getattr(result, "stalled", False):
                return False, "the listening model hung twice"
        if result.returncode == 0:
            return True, "ok"
        return False, f"child exited {result.returncode}"
    except subprocess.TimeoutExpired:
        return False, f"stopped after BATCH_TRACK_TIMEOUT_S ({_batch_stage_timeout():g} s)"
    except Exception as e:
        return False, str(e)


def batch_scan_folder_to_saved_songs(
    folder,
    *,
    audio_temp_files,
    dsp_temp_files,
    stem_temp_files,
    demucs_out_dirs,
    skip_existing=True,
):
    """
    Scan every audio file in folder and write JSON (+ cover art) into SAVE_DIR.
    Does not import anything into the chat. (0.6.2-dev4: the per-file work is
    scan_files_to_saved_songs, which /analyse and /reanalyse use for one file.)
    """
    files = list_audio_files_in_folder(folder)
    if not files:
        print(f"  No audio files found in {folder}\n")
        return
    scan_files_to_saved_songs(
        files,
        folder=folder,
        audio_temp_files=audio_temp_files,
        dsp_temp_files=dsp_temp_files,
        stem_temp_files=stem_temp_files,
        demucs_out_dirs=demucs_out_dirs,
        skip_existing=skip_existing,
    )


def scan_files_to_saved_songs(
    files,
    *,
    folder=None,
    audio_temp_files,
    dsp_temp_files,
    stem_temp_files,
    demucs_out_dirs,
    skip_existing=True,
):
    """
    Analyse each audio file in `files` and write JSON (+ cover art) into
    SAVE_DIR -- the /batch steps, unchanged, for a folder's files (/batch) or
    for one file (/analyse; /reanalyse passes skip_existing=False, so the
    file's existing save is replaced). Does not import anything into the chat.

    When BATCH_ISOLATE_PER_TRACK is True (default), each track is analysed in a
    fresh subprocess so MPS/TensorFlow residual memory cannot accumulate. That
    is the recommended path for overnight runs on macOS.
    """
    single = folder is None
    os.makedirs(SAVE_DIR, exist_ok=True)
    isolate = bool(BATCH_ISOLATE_PER_TRACK)
    os.environ["MUSICLYSE_IN_BATCH"] = "1"
    # Drop any resident writer model before the first track so analysis can use
    # the RAM. Session chat messages stay in the parent Python process.
    if globals().get("BATCH_UNLOAD_OLLAMA", True):
        try:
            status("Unloading Ollama writer for batch scan (session history kept)...")
            ollama_unload_model()
            status_done("Ollama writer unloaded — chat history still in memory")
        except Exception:
            pass
    split_identity = isolate and bool(globals().get("BATCH_SPLIT_IDENTITY_PROCESS", True))
    print(
        (f"  Analysing {os.path.basename(files[0])}\n" if single else
         f"  Batch scan: {len(files)} track(s) in {folder}\n")
        + f"  Saving to {os.path.abspath(SAVE_DIR)} (not imported into chat)\n"
        f"  Isolation: {'subprocess per track (recommended on macOS)' if isolate else 'in-process'}\n"
        f"  Identity stage: "
        f"{'separate process (BATCH_SPLIT_IDENTITY_PROCESS)' if split_identity else 'same process as analysis'}\n"
        f"  Ollama during batch: "
        f"{'unload between stages (BATCH_UNLOAD_OLLAMA)' if BATCH_UNLOAD_OLLAMA else 'left loaded'}\n"
    )

    ok, skipped, failed = 0, 0, 0
    for i, path in enumerate(files, 1):
        # Peek tags so the skip-check uses the same Artist - Title naming
        # that the actual save will use after analysis.
        peek_meta = {}
        if ENABLE_FILE_METADATA:
            try:
                peek_meta, _, _ = extract_audio_metadata(path)
            except Exception:
                peek_meta = {}
        save_name, already_saved = _batch_resolve_save_name(path, metadata=peek_meta)
        label = os.path.basename(path)
        print(f"\n  [{i}/{len(files)}] {label}")

        if skip_existing and already_saved:
            print(f"  (skip — already saved as {save_name})"
                  + (f" -- {REANALYSE_FLAG} analyses it again" if single else ""))
            skipped += 1
            continue
        if not skip_existing and already_saved:
            print(f"  (analysing again -- replaces {save_name})")
        if save_name != _batch_save_basename(path, metadata=peek_meta):
            print(
                f"  (another file is already saved under the same artist and title — "
                f"this one will be saved as {save_name})"
            )

        if isolate:
            success, detail = _batch_run_isolated(path)
            if success:
                ok += 1
            else:
                failed += 1
                print(f"  ✗ failed: {detail}")
            # Short pause even with isolation — lets the OS reclaim the dead
            # child's pages and reduces pressure spikes on the next launch.
            pause = float(BATCH_PAUSE_BETWEEN_TRACKS_S or 0)
            if pause > 0 and i < len(files):
                status(f"Batch pause {pause:.0f}s (letting RAM settle)...")
                try:
                    import time as _time
                    _time.sleep(pause)
                except Exception:
                    pass
                status_done()
            continue

        # In-process fallback (BATCH_ISOLATE_PER_TRACK = False)
        try:
            result = run_fresh_track_analysis(
                path,
                audio_temp_files=audio_temp_files,
                dsp_temp_files=dsp_temp_files,
                stem_temp_files=stem_temp_files,
                demucs_out_dirs=demucs_out_dirs,
            )
            out_path, cover_path = save_song_data(
                save_name,
                result["track_path"],
                result["analysis"],
                result.get("corrections") or {},
                metadata=result.get("metadata"),
                cover_observations=result.get("cover_observations"),
                singer_identity=result.get("singer_identity"),
                cover_bytes=result.get("cover_bytes"),
                cover_mime=result.get("cover_mime") or "image/jpeg",
                preserve_audio_basename=True,
                stem_midi=result.get("stem_notes"),
                structure_sections=result.get("structure_sections"),
                structure_replay=result.get("structure_replay"),
                bpm=result.get("bpm"),
                key=result.get("key"),
                clap_embedding=result.get("clap_embedding"),
                discogs_styles=result.get("discogs_styles"),
            chord_segments=result.get("chord_segments"),
                jamendo_instruments=result.get("jamendo_instruments"),
                listening_details=result.get("listening_details"),
                tuned_percussion=result.get("tuned_percussion"),
                songformer_result=result.get("songformer_result"),
            )
            print(f"  ✓ saved {os.path.basename(out_path)}"
                  + (f" + {os.path.basename(cover_path)}" if cover_path else ""))
            ok += 1
        except Exception as e:
            failed += 1
            print(f"  ✗ failed: {e}")
        finally:
            try:
                _cleanup_track_temp_files(
                    path, audio_temp_files, dsp_temp_files, stem_temp_files, demucs_out_dirs
                )
            except Exception:
                pass
            try:
                _aggressive_memory_cleanup()
            except Exception:
                pass
            pause = float(BATCH_PAUSE_BETWEEN_TRACKS_S or 0)
            if pause > 0 and i < len(files):
                status(f"Batch pause {pause:.0f}s (letting RAM settle)...")
                try:
                    import time as _time
                    _time.sleep(pause)
                except Exception:
                    pass
                status_done()

    print(
        f"\n  {'Done' if single else 'Batch done'} — saved: {ok}, skipped: {skipped}, failed: {failed}\n"
        f"  Chat context was not modified.\n"
    )
    _print_export_reminder()


def _media_has_audio(path):
    """False only when ffprobe reports no audio stream (if it can't tell, the audio step decides)."""
    try:
        result = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries",
                                 "stream=index", "-of", "json", str(path)],
                                capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            return True
        return bool((json.loads(result.stdout or "{}") or {}).get("streams"))
    except Exception:
        return True


def _musicvideo_save_index():
    """{folded source file name: [save paths]} for every song save (reads only each save's head)."""
    import glob as _glob
    index = {}
    for path in sorted(_glob.glob(os.path.join(_glob.escape(SAVE_DIR), "*.json"))):
        fields = _saved_json_source_fields(path) or {}
        source = str(fields.get("track_path") or "")
        if source and not source.startswith(("http://", "https://")):
            index.setdefault(_nfc_fold(os.path.basename(source)), []).append(path)
    return index


def _musicvideo_saved_song_for(source):
    """The newest song save that records this very file as its source (its name without
    .json), or None -- a fallback for when the save's name isn't the one predicted."""
    import glob as _glob
    best = None
    for path in _glob.glob(os.path.join(_glob.escape(SAVE_DIR), "*.json")):
        recorded = str((_saved_json_source_fields(path) or {}).get("track_path") or "")
        if not recorded or recorded.startswith(("http://", "https://")):
            continue
        stored = os.path.abspath(os.path.expanduser(recorded))
        try:
            same = os.path.samefile(stored, source)
        except OSError:
            same = stored == source
        if same:
            try:
                stamp = os.path.getmtime(path)
            except OSError:
                continue
            if best is None or stamp > best[0]:
                best = (stamp, path)
    return os.path.basename(best[1])[:-len(".json")] if best else None


def _video_parse_analyse(video, arguments, flag, options_word):
    """pre65 VIDEO_FRIENDLY_USAGE: the video companion's analyse options, or None after a
    usage line when there is no video file or an option is mistyped (argparse's own
    usage text and error are not shown). --help still prints the full option list."""
    caught = io.StringIO()
    try:
        with contextlib.redirect_stderr(caught):
            return vars(video.make_parser().parse_args(["analyse"] + list(arguments)))
    except SystemExit as exc:
        if exc.code in (0, None):
            return None                      # --help: the option list was printed
        reason = ""
        for line in caught.getvalue().splitlines():
            if ": error: " in line:
                reason = line.split(": error: ", 1)[1].strip()
        missing = "required" in reason and "source" in reason
        print(f'  Usage: {flag} "<video file>" {options_word}'
              + ("" if missing or not reason else f"  ({reason})")
              + f"\n  ({flag} --help lists the options.)\n")
        return None


def _musicvideo_analyse(argument_text, *, audio_temp_files, dsp_temp_files, stem_temp_files, demucs_out_dirs):
    """/musicvideoanalyse: the song (audio analysis, as /analyse), then the pictures (as /videoanalyse)."""
    import shlex
    video = _load_video_extension()
    argument_text, _images, _image_error = _scan_image_options(argument_text)   # 0.8-pre34
    if _image_error:
        raise ValueError(_image_error)
    arguments = shlex.split(argument_text)
    if VIDEO_FRIENDLY_USAGE:
        options = _video_parse_analyse(video, arguments, MUSICVIDEO_ANALYSE_FLAG, "[video options]")
        if options is None:
            return
    else:
        options = vars(video.make_parser().parse_args(["analyse"] + arguments))
    options.pop("command")
    if not str(options.get("source") or "").strip():
        print(f'  Usage: {MUSICVIDEO_ANALYSE_FLAG} "<video file>" [video options]\n')
        return
    if options.get("song") or options.get("offset") is not None:
        raise ValueError("--song and --offset don't apply here: the song is analysed from this same video "
                         "file, so their timings already match.")
    source = os.path.abspath(os.path.expanduser(options["source"]))
    try:
        video.probe_video(source)
    except ValueError as exc:
        if str(exc) == "The input has no video stream.":
            raise ValueError("This file has no pictures -- use /analyse for an audio file.")
        raise
    if not _media_has_audio(source):
        raise ValueError("This video has no soundtrack -- use /videoanalyse for a silent video.")
    custom = options.get("name") or colored_input("  Name (Enter keeps the original name): ", Ansi.LIGHT_GREEN,
                                                  prompt_color=Ansi.YELLOW).strip() or None
    if custom:
        custom = video.folder_name(custom)   # refuse path characters before the long work starts

    print("\n  Music video -- step 1 of 2: the song (audio analysis)")
    audio_error = None
    if _images:
        _scan_images_use(_images)     # 0.8-pre34 SCAN_IMAGE_OPTIONS: an image given scans the song again
    try:
        scan_files_to_saved_songs([source], audio_temp_files=audio_temp_files, dsp_temp_files=dsp_temp_files,
                                  stem_temp_files=stem_temp_files, demucs_out_dirs=demucs_out_dirs,
                                  skip_existing=not _images)
    except Exception as exc:
        audio_error = str(exc) or exc.__class__.__name__
    finally:
        if _images:
            _scan_images_use({})
    meta = {}
    if ENABLE_FILE_METADATA:
        try:
            meta = extract_audio_metadata(source)[0] or {}
        except Exception:
            meta = {}
    save_name, saved = _batch_resolve_save_name(source, metadata=meta)
    song = save_name[:-len(".json")] if saved else _musicvideo_saved_song_for(source)
    if song is None:
        print("  The song wasn't saved" + (f" ({audio_error})" if audio_error else "")
              + ". The video step runs anyway; afterwards, run the same command again to retry the song "
                "(finished parts are skipped).\n")

    print("  Music video -- step 2 of 2: the pictures (video analysis)")
    options["source"] = source
    options["name"] = custom or song or _batch_save_basename(source, metadata=meta)[:-len(".json")]
    if "--model" not in arguments and not any(a.startswith("--model=") for a in arguments):
        options["model"] = OLLAMA_MODEL
    if "--endpoint" not in arguments and not any(a.startswith("--endpoint=") for a in arguments):
        options["endpoint"] = OLLAMA_URL
    try:
        report_path = video.analyse_video(**options)
    except (ValueError, OSError, KeyError, TypeError, RuntimeError) as exc:
        raise RuntimeError(f"{exc}" + (" -- the song is saved; run the same command again to continue the "
                                       "video analysis (finished parts are skipped)." if song else ""))
    folder = os.path.dirname(str(report_path))
    name = os.path.basename(folder)
    if not song:
        print(f"  The video analysis is saved as {name!r}, but without its song it can't be loaded "
              f"with {MUSICVIDEO_LOAD_FLAG} yet.\n")
        return
    _musicvideo_write_pair(folder, song, source)
    export = _musicvideo_export_path(song)
    print(f"\n  ✓ Music video ready: {name}\n"
          f"    Song:  {os.path.normpath(os.path.join(SAVE_DIR, song + '.json'))}\n"
          f"    Video: {os.path.relpath(folder) if not os.path.relpath(folder).startswith('..') else folder}")
    if os.path.isfile(export) and os.path.getmtime(export) >= os.path.getmtime(os.path.join(SAVE_DIR, song + ".json")):
        print(f"  The song's export is already there. Load both together with: "
              f"{MUSICVIDEO_LOAD_FLAG} {json.dumps(name, ensure_ascii=False)}\n")
    else:
        print(f"  Next, export the song for chat:\n    {_musicvideo_export_command(song)}\n"
              f"  then load both together with: {MUSICVIDEO_LOAD_FLAG} {json.dumps(name, ensure_ascii=False)}\n")
