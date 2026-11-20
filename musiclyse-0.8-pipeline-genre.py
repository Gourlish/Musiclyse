# MUSICLYSE FILE pipeline-genre 0.8
# ==============================================================================
#  MUSICLYSE 0.8: GENRE, STYLE, MOOD AND ERA
# ==============================================================================
# This file holds genre, style, mood and era: CLAP, the Discogs styles, the genre checks and the
# era evidence. It is part of musiclyse-0.8.py, which loads it whole at start-up, after its own
# code, into its own namespace (see "THE MUSICLYSE FILES" there); it is not run on its own.
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


_RECOMMENDED_GENRE_RE = re.compile(r"RECOMMENDED GENRE FOR DISCUSSION: ([^\n]+?)\. Reasoning:")


# --- Lazy loading / swapping for the CLAP sonic-similarity model -------------
_clap_state = {"model": None, "processor": None, "device": None}


def _clap_planned_device():
    if CLAP_USE_MPS and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _load_clap_impl():
    if not CLAP_TRANSFORMERS_AVAILABLE or ClapModel is None or ClapProcessor is None:
        raise RuntimeError(
            "This transformers installation does not provide ClapModel/ClapProcessor; "
            "upgrade transformers to enable CLAP JSON embeddings."
        )
    if DEBUG_MODEL_LOADING:
        print(f"Loading {CLAP_MODEL_ID} ...")
    else:
        status(f"Loading CLAP ({CLAP_MODEL_ID})...")

    # The main application deliberately runs Hugging Face in offline mode so
    # previously-installed analysis models never unexpectedly hit the network.
    # CLAP is explicitly opt-in and may need its first download, so temporarily
    # relax that flag for this one model load and restore it immediately.
    old_offline = os.environ.pop("HF_HUB_OFFLINE", None)
    try:
        device = _clap_planned_device()
        model = ClapModel.from_pretrained(
            CLAP_MODEL_ID,
            low_cpu_mem_usage=True,
            use_safetensors=False,
        )
        processor = ClapProcessor.from_pretrained(CLAP_MODEL_ID)
        model.to(device)
        model.eval()
    finally:
        if old_offline is not None:
            os.environ["HF_HUB_OFFLINE"] = old_offline

    if DEBUG_MODEL_LOADING:
        print(f"CLAP loaded on device: {device}")
    else:
        status_done(f"CLAP loaded on device: {device}")
    return model, processor, device


def get_clap():
    if _clap_state["model"] is None:
        model, processor, device = _load_clap_impl()
        _clap_state["model"] = model
        _clap_state["processor"] = processor
        _clap_state["device"] = device
    return _clap_state["model"], _clap_state["processor"], _clap_state["device"]


def unload_clap():
    if _clap_state["model"] is None:
        return
    status("Freeing CLAP from memory...")
    model = _clap_state["model"]
    processor = _clap_state["processor"]
    _clap_state["model"] = None
    _clap_state["processor"] = None
    _clap_state["device"] = None
    try:
        del model
    except Exception:
        pass
    try:
        del processor
    except Exception:
        pass
    for _ in range(2):
        if _gpu_cache_clear_ok():
            gc.collect()
        try:
            if torch.backends.mps.is_available() and _gpu_cache_clear_ok():
                torch.mps.empty_cache()
            elif torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
    status_done("CLAP unloaded")


def _normalize_clap_vector(vector):
    try:
        arr = np.asarray(vector, dtype=np.float32).reshape(-1)
    except Exception:
        return None
    if arr.size != CLAP_EMBEDDING_DIMENSION or not np.all(np.isfinite(arr)):
        return None
    norm = float(np.linalg.norm(arr))
    if norm <= 0.0:
        return None
    return arr / norm


def _clap_window_starts(duration_s):
    duration_s = float(max(0.0, duration_s))
    win = float(CLAP_WINDOW_SECONDS)
    hop = float(CLAP_WINDOW_HOP_SECONDS)
    if duration_s <= win:
        return [0.0]

    max_start = max(0.0, duration_s - win)
    starts = []
    pos = 0.0
    while pos <= max_start + 1e-6 and len(starts) < int(CLAP_MAX_WINDOWS):
        starts.append(float(pos))
        pos += hop

    if starts and starts[-1] < max_start - 1e-6 and len(starts) < int(CLAP_MAX_WINDOWS):
        starts.append(float(max_start))

    # For very long tracks, replace the uniform early-only coverage with
    # evenly-spaced windows spanning the entire recording. This keeps the
    # representation global without processing an unbounded number of windows.
    if len(starts) >= int(CLAP_MAX_WINDOWS) and max_start > 0:
        count = int(CLAP_MAX_WINDOWS)
        starts = np.linspace(0.0, max_start, count).tolist()

    return starts


_CLAP_EMBEDDING_CACHE = {}


def _clap_cache_key__pre6(path):
    """Identity for a computed embedding: path + size + mtime, so a replaced
    or re-encoded file at the same path never returns a stale vector."""
    try:
        st = os.stat(path)
        return (os.path.abspath(str(path)), int(st.st_size), int(st.st_mtime_ns))
    except Exception:
        return None


def _clap_cache_key__07(path):
    """Identity for a computed embedding: path + size + mtime, so a replaced
    or re-encoded file at the same path never returns a stale vector."""
    try:
        st = os.stat(path)
        return (os.path.abspath(str(path)), int(st.st_size), int(st.st_mtime))
    except Exception:
        return None


def _clap_cache_key(*args, **kwargs):
    """AUDIO_CACHE_FIXES picks the 0.7.1-pre6 version; False gives 0.7's."""
    return (_clap_cache_key__pre6 if AUDIO_CACHE_FIXES else _clap_cache_key__07)(*args, **kwargs)


def compute_clap_embedding(path, keep_loaded=False, *, audio_source=None):
    """Return a normalized 512-d CLAP audio embedding aggregated across
    deterministic 10-second windows, or None when the feature cannot be made.

    The LAION checkpoint's audio preprocessor is configured around 48 kHz / 10 s
    chunks, so using explicit windows avoids relying on its random truncation for
    long tracks. Each window is normalized before averaging, then the aggregate
    is normalized again for cosine similarity.

    Results are cached per (path, size, mtime) for the life of the process.
    The embedding is now computed once at the end of analysis and then reused
    by the save; without the cache that hand-off would silently re-run the
    whole CLAP pass a second time.

    `keep_loaded=True` leaves the model resident instead of unloading in the
    finally block. The unload-every-call default is correct for the one-shot
    save path, but it makes a loop over several tracks pay a full model load
    per track, so batch backfills pass True for all but the final call.
    """
    if not ENABLE_CLAP_EMBEDDING:
        return None
    if not path or str(path).startswith(("http://", "https://")) or not os.path.exists(path):
        return None

    cache_key = _clap_cache_key(path)
    if cache_key is not None and cache_key in _CLAP_EMBEDDING_CACHE:
        cached = _CLAP_EMBEDDING_CACHE[cache_key]
        return list(cached) if cached is not None else None

    model = processor = None
    try:
        model, processor, device = get_clap()
        source = contextlib.nullcontext(audio_source) if audio_source else _analysis_audio_source(path)
        with source as decoded_path:
            audio, sr = librosa.load(decoded_path, sr=CLAP_SAMPLE_RATE, mono=True)
        if audio.size == 0:
            return None
        duration_s = float(audio.size) / float(sr)
        starts = _clap_window_starts(duration_s)
        embeddings = []
        with torch.inference_mode():
            for start_s in starts:
                start = int(round(start_s * sr))
                end = start + int(round(CLAP_WINDOW_SECONDS * sr))
                chunk = audio[start:end]
                if chunk.size == 0:
                    continue
                # Explicitly pad short final chunks. ClapProcessor's standard
                # repeat-pad behaviour is equivalent in spirit, but making the
                # length explicit keeps window construction deterministic.
                target = int(round(CLAP_WINDOW_SECONDS * sr))
                if chunk.size < target:
                    if chunk.size == 1:
                        chunk = np.repeat(chunk, target)
                    else:
                        reps = int(np.ceil(target / chunk.size))
                        chunk = np.tile(chunk, reps)[:target]
                # Prefer the official ClapProcessor keyword; fall back to audio=
                # for older/newer argument name differences.
                inputs = processor(
                    audio=chunk,
                    sampling_rate=sr,
                    return_tensors="pt",
                )
                # Only forward keys ClapModel.get_audio_features accepts.
                _clap_input_keys = ("input_features", "is_longer", "attention_mask")
                inputs = {
                    k: (v.to(device) if hasattr(v, "to") else v)
                    for k, v in inputs.items()
                    if k in _clap_input_keys
                }
                emb = model.get_audio_features(**inputs)
                # Newer transformers return BaseModelOutputWithPooling from
                # get_audio_features; older versions return a plain tensor.
                if torch.is_tensor(emb):
                    pass
                elif hasattr(emb, "pooler_output") and emb.pooler_output is not None:
                    emb = emb.pooler_output
                elif hasattr(emb, "last_hidden_state") and emb.last_hidden_state is not None:
                    emb = emb.last_hidden_state
                elif isinstance(emb, (tuple, list)) and len(emb) > 0:
                    emb = emb[0]
                else:
                    raise TypeError(
                        f"Unexpected get_audio_features return type: {type(emb)!r}"
                    )
                emb = emb.detach().float().cpu().numpy()
                if getattr(emb, "ndim", 1) > 1:
                    emb = emb[0]
                emb = _normalize_clap_vector(emb)
                if emb is not None:
                    embeddings.append(emb)

        if not embeddings:
            return None
        aggregate = _normalize_clap_vector(np.mean(np.stack(embeddings, axis=0), axis=0))
        if aggregate is None:
            return None
        result = [round(float(x), CLAP_EMBEDDING_PRECISION) for x in aggregate.tolist()]
        if cache_key is not None:
            try:
                if len(_CLAP_EMBEDDING_CACHE) > 32:
                    _CLAP_EMBEDDING_CACHE.clear()
                _CLAP_EMBEDDING_CACHE[cache_key] = list(result)
            except Exception:
                pass
        return result
    except Exception as e:
        print(f"  (CLAP embedding skipped: {e})")
        return None
    finally:
        # Save-time use is intentionally one-shot; do not retain another large
        # model in the interactive process between songs. Callers looping over
        # several tracks pass keep_loaded=True so they don't reload per track.
        if not keep_loaded:
            try:
                unload_clap()
            except Exception:
                pass
        try:
            del audio
        except Exception:
            pass
        if _gpu_cache_clear_ok():
            gc.collect()


def _extract_clap_embedding(value):
    if isinstance(value, dict):
        value = value.get("vector") or value.get("embedding")
    vec = _normalize_clap_vector(value)
    return vec.tolist() if vec is not None else None


def _genre_hint_from_analysis_text(text, panns_genre_tags=None):
    """Best-effort genre string for BPM bias: listening-model GENRE_RANKED top, else PANNs."""
    labels = extract_genre_ranked_labels(text or "", max_n=1)
    if labels:
        return labels[0]
    if panns_genre_tags:
        for item in panns_genre_tags:
            if item and len(item) >= 1 and item[0]:
                return str(item[0])
    return None


# Genre families for conflict detection between listening-model GENRE_RANKED and PANNs tags.
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
    # The listening model often writes compound genres with a non-breaking or other
    # Unicode hyphen ("dance\u2011pop"); fold them to "-" so they match the
    # family tables and read normally when quoted back to the user.
    s = re.sub(r"[\u2010-\u2015\u2212]", "-", s)
    s = re.sub(r"\s+", " ", s)
    s = s.replace("_", " ")
    return s


def extract_genre_ranked_labels(text, max_n=8):
    """Pull ordered genre labels from a GENRE_RANKED=... field in listening-model text."""
    if not text:
        return []
    text = _strip_md_emphasis(text)
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
        tok = _normalize_genre_token(p)
        # Keep first few words of each rank item
        tok = re.split(r"[./|]", tok)[0].strip()
        if not _is_plausible_genre_label(tok):
            continue
        if tok and tok not in labels:
            labels.append(tok)
        if len(labels) >= max_n:
            break
    return labels


_GENRE_LABEL_REJECT_WORDS = (
    "evidence", "override", "overrides", "automatic", "supporting", "confidence",
    "unavailable", "uncertain", "unknown", "note", "reasoning", "instead",
    "should", "must", "do not", "cannot", "please", "output", "format",
    "listed", "above", "below", "n/a", "none",
)


def _is_plausible_genre_label(tok):
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
    for bad in _GENRE_LABEL_REJECT_WORDS:
        if bad in t:
            return False
    # Must contain letters, and not read as a sentence fragment.
    if not re.search(r"[a-z]", t):
        return False
    return True


def _genre_family(label):
    t = _normalize_genre_token(label)
    # Longer / more specific keys first via explicit checks
    for fam, members in (
        ("rockish", _GENRE_FAMILY_ROCKISH),
        ("electronicish", _GENRE_FAMILY_ELECTRONICISH),
        ("popish", _GENRE_FAMILY_POPISH),
    ):
        if t in members or _any_marker_present(t, members):
            return fam
        # token-wise
        for word in t.replace("-", " ").split():
            if word in members or any(word == mm.replace("-", " ") for mm in members):
                return fam
    # substring heuristics -- word-boundary anchored so short genre words
    # (e.g. "ska", "disco", "house", "dance") can't false-match inside an
    # unrelated longer word ("polska", "discography", "warehouse", "abundance").
    if _any_marker_present(t, ("punk", "emo", "metal", "grunge", "rock")):
        return "rockish"
    if _any_marker_present(t, ("house", "techno", "edm", "electro", "synth", "dance", "trance", "dubstep")):
        return "electronicish"
    if _any_marker_present(t, ("pop",)):
        return "popish"
    return "other"


_GENRE_PROSE_LEADING_ADJ = re.compile(
    r"^(?:high[\u2010-\u2015-]?energy|fast[\u2010-\u2015-]?paced|slow|mid[\u2010-\u2015-]?tempo|"
    r"up[\u2010-\u2015-]?beat|laid[\u2010-\u2015-]?back|driving|mellow|gentle|heavy|light|"
    r"aggressive|energetic|dreamy|moody|dark|bright|really|very|quite|fairly|somewhat|"
    r"well[\u2010-\u2015-]?produced|well[\u2010-\u2015-]?mixed|well[\u2010-\u2015-]?recorded|"
    r"nicely[\u2010-\u2015-]?produced|beautifully[\u2010-\u2015-]?recorded|polished|great|good|"
    r"solid|decent|excellent|impressive|professional)\s+",
    re.IGNORECASE,
)

# A candidate phrase must contain at least one recognisable genre-family root
# to be trusted as a genre at all. Without this gate, an innocuous sentence
# like "This is a really well-produced track" matches the same "is a ___
# song/track" pattern and would be accepted as a "genre" -- production-quality
# praise, containing no genre information whatsoever, stated as the track's
# genre. Reuses _any_marker_present's word-boundary matching (not a naive
# substring check) so short roots like "pop" or "wave" cannot false-positive
# inside an unrelated word.
_GENRE_ROOT_WORDS = (
    "rock", "punk", "pop", "metal", "emo", "grunge", "indie", "alternative",
    "electronic", "electro", "house", "techno", "trance", "dubstep", "synth",
    "dance", "disco", "ambient", "r&b", "r and b", "soul", "funk", "hip hop",
    "rap", "folk", "country", "blues", "jazz", "classical", "reggae", "ska",
    "wave", "core", "americana", "gospel", "latin", "afrobeat", "world",
    "singer-songwriter", "acoustic", "orchestral", "instrumental",
)


_GENRE_PROSE_EXTRA_ADJ = re.compile(
    r"^(?:uplifting|catchy|infectious|anthemic|melancholic|melancholy|atmospheric|lush|"
    r"introspective|emotional|romantic|nostalgic|vibrant|punchy|sleek|glossy|slick|gritty|"
    r"sunny|cheerful|joyful|playful|bittersweet|brooding|haunting|ethereal|soaring|pulsing|"
    r"pulsating|propulsive|sentimental|feel[\u2010-\u2015-]?good|radio[\u2010-\u2015-]?friendly|"
    r"modern|contemporary|retro|vintage[\u2010-\u2015-]?style|stripped[\u2010-\u2015-]?down|"
    r"mid[\u2010-\u2015-]?paced|up[\u2010-\u2015-]?tempo|down[\u2010-\u2015-]?tempo|"
    r"guitar[\u2010-\u2015-]?driven|synth[\u2010-\u2015-]?driven|piano[\u2010-\u2015-]?driven|"
    r"bass[\u2010-\u2015-]?driven|dance[\u2010-\u2015-]?floor[\u2010-\u2015-]?ready)\s+",
    re.IGNORECASE,
)


# 0.8-pre31 GENRE_PROSE_TRAILING_CLAUSE: the part of a sentence after the genre's own noun ("...
# track that never lets up", "... song with no shortage of hooks") is left out of the check for
# negative and hedging words; the sentence itself is kept whole.
_GENRE_TRAILING_CLAUSE_RE = re.compile(
    r"(?i)\b(songs?|tracks?|piece|tune|number|cut|recording|single|record|ballad|anthem|jam|instrumental|"
    r"composition|production|arrangement|groove|banger|stomper)\b\s*,?\s*"
    r"(?:that|which|who|with|whose|where|from|full\s+of|built\s+on|driven\s+by)\b.*$")


def _genre_asserted_prose(text):
    """Only positive, current-track prose may supply fallback genre labels.

    Structured alternatives/exclusions are not descriptions of the track.
    Conservative clause filtering also avoids quoting another recording's genre.
    """
    t = _strip_md_emphasis(str(text or ""))
    t = re.sub(r"(?im)^\s*GENRE_[A-Z_]+\s*[:=][^\n]*", "", t)
    # Split an explicitly rejected alternative away from a positive genre.
    t = re.sub(r",\s*(?=not\b)", "; ", t, flags=re.I)
    t = re.sub(r"\bunlike any other\b", "", t, flags=re.I)
    kept = []
    for sentence in re.split(r"[.!?;\n]+", t):
        # Keep an explicitly asserted current-track clause after a comparison.
        if re.search(r"\b(?:unlike|previous|another|other)\b", sentence, re.I):
            parts = re.split(r",\s*(?=this (?:track |song |recording )?is\b)", sentence, flags=re.I)
            if len(parts) < 2:
                continue
            sentence = parts[-1]
        check = re.sub(r"(?i)\b(?:with\s+)?no\s+(?:vocals?|lyrics|singing|drums|guitars?|synths?|percussion)\b", "",
                       _GENRE_TRAILING_CLAUSE_RE.sub(r"\1", sentence) if GENRE_PROSE_TRAILING_CLAUSE else sentence)
        if re.search(r"\b(?:not|never|no|wrong|incorrect\w*|mislabel\w*|mistaken\w*|"
                     r"ruled out|reject\w*|unlikely|possibly|perhaps|maybe|might|could|"
                     r"would|whether|rather than)\b|\b\w+n['’]t\b", check, re.I):
            continue
        kept.append(sentence)
    return ". ".join(kept)


def _genre_prose_candidates__pre6(text, window=None):
    """dev55: genre phrases in the listening model's prose, in order. Accepts
    "is a/an X song|track|piece|..." and also "X / Y" compounds with spaced
    slashes and adjective lists; each candidate is cleaned of descriptors and
    must contain a genre root. Returns normalised labels (compounds as
    "a / b", at most two parts)."""
    if not text:
        return []
    t = _genre_asserted_prose(str(text) if window is None else str(text)[:window])
    _HY = r"\u2010-\u2015\-"
    out = []
    for m in re.finditer(
            r"\b(?:is|sounds like|reads as|as)\s+(?:an?|the)\s+"
            rf"([A-Za-z][A-Za-z0-9{_HY}'.&]*(?:(?:[\s{_HY},]+|\s*/\s*)[A-Za-z][A-Za-z0-9{_HY}'.&]*){{0,7}})"
            r"\s+(?:song|track|record|tune|number|cut|piece|jam|ballad|instrumental|groove|anthem|"
            r"banger|mix|composition|production|cut)\b", t, re.IGNORECASE):
        phrase = m.group(1)
        for ch in "\u2010\u2011\u2012\u2013\u2014\u2015":
            phrase = phrase.replace(ch, "-")
        parts = [p.strip() for p in re.split(r"\s*/\s*", phrase) if p.strip()]
        labs = []
        for p in parts[:2]:
            p = p.replace(",", " ")
            for _ in range(4):
                q = _GENRE_PROSE_EXTRA_ADJ.sub("", _GENRE_PROSE_LEADING_ADJ.sub("", p)).strip(" -")
                if q == p:
                    break
                p = q
            tok = _normalize_genre_token(p)
            words = tok.split()
            # drop leading non-genre words until a plausible genre phrase
            k = 0
            while k < len(words) and not (_is_plausible_genre_label(" ".join(words[k:]))
                                          and _any_marker_present(" ".join(words[k:]), _GENRE_ROOT_WORDS)):
                k += 1
            if k < len(words):
                labs.append(" ".join(words[k:]))
        if labs:
            lab = " / ".join(dict.fromkeys(labs))
            if lab not in out:
                out.append(lab)
    return out


def _genre_prose_candidates__07(text, window=None):
    """dev55: genre phrases in the listening model's prose, in order. Accepts
    "is a/an X song|track|piece|..." and also "X / Y" compounds with spaced
    slashes and adjective lists; each candidate is cleaned of descriptors and
    must contain a genre root. Returns normalised labels (compounds as
    "a / b", at most two parts)."""
    if not text:
        return []
    t = str(text) if window is None else str(text)[:window]
    _HY = r"\u2010-\u2015\-"
    out = []
    for m in re.finditer(
            r"\b(?:is|sounds like|reads as|as)\s+(?:an?|the)\s+"
            rf"([A-Za-z][A-Za-z0-9{_HY}'.&]*(?:(?:[\s{_HY},]+|\s*/\s*)[A-Za-z][A-Za-z0-9{_HY}'.&]*){{0,7}})"
            r"\s+(?:song|track|record|tune|number|cut|piece|jam|ballad|instrumental|groove|anthem|"
            r"banger|mix|composition|production|cut)\b", t, re.IGNORECASE):
        phrase = m.group(1)
        for ch in "\u2010\u2011\u2012\u2013\u2014\u2015":
            phrase = phrase.replace(ch, "-")
        parts = [p.strip() for p in re.split(r"\s*/\s*", phrase) if p.strip()]
        labs = []
        for p in parts[:2]:
            p = p.replace(",", " ")
            for _ in range(4):
                q = _GENRE_PROSE_EXTRA_ADJ.sub("", _GENRE_PROSE_LEADING_ADJ.sub("", p)).strip(" -")
                if q == p:
                    break
                p = q
            tok = _normalize_genre_token(p)
            words = tok.split()
            # drop leading non-genre words until a plausible genre phrase
            k = 0
            while k < len(words) and not (_is_plausible_genre_label(" ".join(words[k:]))
                                          and _any_marker_present(" ".join(words[k:]), _GENRE_ROOT_WORDS)):
                k += 1
            if k < len(words):
                labs.append(" ".join(words[k:]))
        if labs:
            lab = " / ".join(dict.fromkeys(labs))
            if lab not in out:
                out.append(lab)
    return out


def _genre_prose_candidates(*args, **kwargs):
    """GENRE_ASSERTED_PROSE picks the 0.7.1-pre6 version; False gives 0.7's."""
    return (_genre_prose_candidates__pre6 if GENRE_ASSERTED_PROSE else _genre_prose_candidates__07)(*args, **kwargs)


def _extract_genre_hint_from_moss_prose__pre6(text):
    """Pull a genre-ish phrase out of the listening model's FREE-FORM prose when it
    ignores the requested structured GENRE_RANKED=... format entirely.

    Observed on a real scan: the listening model's main pass produced a full, accurate
    description -- "a high-energy Pop-Punk song that leans heavily into
    Emo-Pop sensibilities" -- with none of the structured labels the prompt
    asked for, so extract_genre_ranked_labels() correctly found nothing
    parseable and the entire genre framing silently fell back to a single raw
    PANNs AudioSet tag ('grunge'), discarding the listening model's considerably more accurate
    read that was sitting right there in the text.

    This is deliberately a LAST-RESORT signal, used only when the structured
    field genuinely is not present. It is intentionally simple (one regex over
    the listening model's common "is a [descriptor] song/track" opening, which is how it starts
    the overwhelming majority of its main-pass responses) rather than a
    general prose-genre classifier, and the result is still filtered through
    _is_plausible_genre_label() before use, and always feeds into the SAME
    family-vs-PANNs reconciliation as a real GENRE_RANKED top would -- so a
    noisy extraction gets the same cross-checking a structured one does,
    rather than being trusted blindly.
    """
    if not text:
        return None
    # Word-internal chars include the plain ASCII hyphen (U+002D, what "Pop-Rock"
    # or "Synth-Pop" is actually typed with) as well as the Unicode dash variants
    # a model sometimes uses instead, plus '/' and '&' for compound genres
    # ("Hard Rock/Heavy Metal", "R&B/Soul"). The separator between words also
    # accepts a comma, since a leading adjective list ("raw, lo-fi Garage Rock")
    # is common before the genre noun. Missing any one of these gives the
    # pattern no valid path from the article ("a") to the trailing noun
    # ("piece"/"song"/...) -- it back-tracks through every rep count and still
    # fails, so the whole match silently returns nothing even though a good
    # genre phrase is sitting right there. A 5-sentence battery of realistic
    # phrasings caught FOUR separate misses from this before the widening:
    # a plain hyphen in "Synth-Pop", a plain hyphen in "Pop-Rock", a comma
    # before "Garage Rock", and the trailing noun list itself being too short.
    text = _genre_asserted_prose(text)
    _HY = r"\u2010-\u2015\-"
    m = re.search(
        r"\b(?:is|sounds like|reads as)\s+(?:an?|the)\s+"
        rf"([A-Za-z][A-Za-z0-9{_HY}'./&]*(?:[\s{_HY},]+[A-Za-z][A-Za-z0-9{_HY}'./&]*){{0,5}})"
        r"\s+(?:song|track|record|tune|number|cut|piece|jam|ballad|number|"
        r"instrumental|groove|anthem|banger|tune|mix|cut)\b",
        text[:600],
        re.IGNORECASE,
    )
    if not m:
        return None
    phrase = m.group(1).replace("\u2010", "-").replace("\u2011", "-").replace("\u2012", "-").replace("\u2013", "-").replace("\u2014", "-").replace("\u2015", "-")
    # Commas can appear WITHIN the captured span (a leading adjective list --
    # "raw, lo-fi Garage Rock" -- needs a comma in the separator to be reached
    # at all), but a comma inside the FINAL phrase reads as a list, not a
    # single label, and _is_plausible_genre_label rejects it outright. Strip
    # commas out here rather than in the separator class, so the comma can
    # still do its job of letting the match continue through an adjective
    # list without the final label carrying it.
    phrase = phrase.replace(",", " ")
    phrase = _GENRE_PROSE_LEADING_ADJ.sub("", phrase).strip(" -")
    # A second pass: the FIRST leading-adjective strip only removes ONE
    # matched adjective from the front, but "raw, lo-fi Garage Rock" has two
    # in sequence ("raw" then "lo-fi") once the comma above becomes a space --
    # applying the same strip again catches a second stacked adjective without
    # having to make the regex itself repeat-aware.
    phrase = _GENRE_PROSE_LEADING_ADJ.sub("", phrase).strip(" -")
    if not phrase:
        return None
    tok = _normalize_genre_token(phrase)
    if GENRE_PROSE_HINT_TRIM and not (_is_plausible_genre_label(tok)
                                      and _any_marker_present(tok, _GENRE_ROOT_WORDS)):
        # Drop leading descriptors ("confidence-driven rock") until a
        # genre phrase is left.
        _words = phrase.split()
        for _k in range(1, len(_words)):
            _cand = _normalize_genre_token(" ".join(_words[_k:]))
            if _is_plausible_genre_label(_cand) and _any_marker_present(_cand, _GENRE_ROOT_WORDS):
                return _cand
        return None
    if not _is_plausible_genre_label(tok):
        return None
    if not _any_marker_present(tok, _GENRE_ROOT_WORDS):
        return None
    return tok


def _extract_genre_hint_from_moss_prose__07(text):
    """Pull a genre-ish phrase out of the listening model's FREE-FORM prose when it
    ignores the requested structured GENRE_RANKED=... format entirely.

    Observed on a real scan: the listening model's main pass produced a full, accurate
    description -- "a high-energy Pop-Punk song that leans heavily into
    Emo-Pop sensibilities" -- with none of the structured labels the prompt
    asked for, so extract_genre_ranked_labels() correctly found nothing
    parseable and the entire genre framing silently fell back to a single raw
    PANNs AudioSet tag ('grunge'), discarding the listening model's considerably more accurate
    read that was sitting right there in the text.

    This is deliberately a LAST-RESORT signal, used only when the structured
    field genuinely is not present. It is intentionally simple (one regex over
    the listening model's common "is a [descriptor] song/track" opening, which is how it starts
    the overwhelming majority of its main-pass responses) rather than a
    general prose-genre classifier, and the result is still filtered through
    _is_plausible_genre_label() before use, and always feeds into the SAME
    family-vs-PANNs reconciliation as a real GENRE_RANKED top would -- so a
    noisy extraction gets the same cross-checking a structured one does,
    rather than being trusted blindly.
    """
    if not text:
        return None
    # Word-internal chars include the plain ASCII hyphen (U+002D, what "Pop-Rock"
    # or "Synth-Pop" is actually typed with) as well as the Unicode dash variants
    # a model sometimes uses instead, plus '/' and '&' for compound genres
    # ("Hard Rock/Heavy Metal", "R&B/Soul"). The separator between words also
    # accepts a comma, since a leading adjective list ("raw, lo-fi Garage Rock")
    # is common before the genre noun. Missing any one of these gives the
    # pattern no valid path from the article ("a") to the trailing noun
    # ("piece"/"song"/...) -- it back-tracks through every rep count and still
    # fails, so the whole match silently returns nothing even though a good
    # genre phrase is sitting right there. A 5-sentence battery of realistic
    # phrasings caught FOUR separate misses from this before the widening:
    # a plain hyphen in "Synth-Pop", a plain hyphen in "Pop-Rock", a comma
    # before "Garage Rock", and the trailing noun list itself being too short.
    _HY = r"\u2010-\u2015\-"
    m = re.search(
        r"\b(?:is|sounds like|reads as)\s+(?:an?|the)\s+"
        rf"([A-Za-z][A-Za-z0-9{_HY}'./&]*(?:[\s{_HY},]+[A-Za-z][A-Za-z0-9{_HY}'./&]*){{0,5}})"
        r"\s+(?:song|track|record|tune|number|cut|piece|jam|ballad|number|"
        r"instrumental|groove|anthem|banger|tune|mix|cut)\b",
        text[:600],
        re.IGNORECASE,
    )
    if not m:
        return None
    phrase = m.group(1).replace("\u2010", "-").replace("\u2011", "-").replace("\u2012", "-").replace("\u2013", "-").replace("\u2014", "-").replace("\u2015", "-")
    # Commas can appear WITHIN the captured span (a leading adjective list --
    # "raw, lo-fi Garage Rock" -- needs a comma in the separator to be reached
    # at all), but a comma inside the FINAL phrase reads as a list, not a
    # single label, and _is_plausible_genre_label rejects it outright. Strip
    # commas out here rather than in the separator class, so the comma can
    # still do its job of letting the match continue through an adjective
    # list without the final label carrying it.
    phrase = phrase.replace(",", " ")
    phrase = _GENRE_PROSE_LEADING_ADJ.sub("", phrase).strip(" -")
    # A second pass: the FIRST leading-adjective strip only removes ONE
    # matched adjective from the front, but "raw, lo-fi Garage Rock" has two
    # in sequence ("raw" then "lo-fi") once the comma above becomes a space --
    # applying the same strip again catches a second stacked adjective without
    # having to make the regex itself repeat-aware.
    phrase = _GENRE_PROSE_LEADING_ADJ.sub("", phrase).strip(" -")
    if not phrase:
        return None
    tok = _normalize_genre_token(phrase)
    if GENRE_PROSE_HINT_TRIM and not (_is_plausible_genre_label(tok)
                                      and _any_marker_present(tok, _GENRE_ROOT_WORDS)):
        # Drop leading descriptors ("confidence-driven rock") until a
        # genre phrase is left.
        _words = phrase.split()
        for _k in range(1, len(_words)):
            _cand = _normalize_genre_token(" ".join(_words[_k:]))
            if _is_plausible_genre_label(_cand) and _any_marker_present(_cand, _GENRE_ROOT_WORDS):
                return _cand
        return None
    if not _is_plausible_genre_label(tok):
        return None
    if not _any_marker_present(tok, _GENRE_ROOT_WORDS):
        return None
    return tok


def _extract_genre_hint_from_moss_prose(*args, **kwargs):
    """GENRE_ASSERTED_PROSE picks the 0.7.1-pre6 version; False gives 0.7's."""
    return (_extract_genre_hint_from_moss_prose__pre6 if GENRE_ASSERTED_PROSE else _extract_genre_hint_from_moss_prose__07)(*args, **kwargs)


# dev40: last-resort genre from the PANNs tag list alone. Its single top tag
# is a fine-grained guess the classifier is known to get wrong ("ska" 35% on a
# synth-led rock track whose other tags were psychedelic rock, progressive
# rock, dubstep and electronic). The tags are grouped by family instead.
_GENRE_FAMILY_WORDS = {"rockish": "rock", "electronicish": "electronic", "popish": "pop"}


def _genre_family_framing(panns, fam_mass):
    """(label, note) framing a PANNs-only genre at the family level, or None
    to keep the single-tag framing (a lone tag outside the three families,
    such as jazz or reggae, that outweighs every family)."""
    if not panns:
        return None
    ranked = sorted((f for f in _GENRE_FAMILY_WORDS if fam_mass.get(f, 0.0) > 0.0),
                    key=lambda f: -fam_mass[f])
    if not ranked:
        return None
    top_tag, top_p = panns[0]
    dom = ranked[0]
    if _genre_family(top_tag) == "other" and top_p >= fam_mass[dom]:
        return None
    label = _GENRE_FAMILY_WORDS[dom]
    if len(ranked) > 1:
        sec = ranked[1]
        if fam_mass[sec] >= 0.20 and fam_mass[sec] >= 0.4 * fam_mass[dom]:
            label = f"{label} (with {_GENRE_FAMILY_WORDS[sec]} lean)"

    def _tags(f):
        return ", ".join(f"{a} {b * 100:.0f}%" for a, b in panns if _genre_family(a) == f)

    fam_desc = "; ".join(
        f"{_GENRE_FAMILY_WORDS[f]} family: {_tags(f)} ({fam_mass[f] * 100:.0f}% together)" for f in ranked)
    note = (
        "No parseable GENRE_RANKED and no usable genre description in MOSS-Music's own text; the only "
        "genre signal is the PANNs/AudioSet tag list, which is unreliable on fine-grained "
        "subgenres. Its tags are grouped by family here instead of taking the single top tag "
        f"('{top_tag}', {top_p * 100:.0f}%): {fam_desc}. Frame the track at that level "
        f"('{label}') and describe the actual style from the instrumentation evidence "
        "(instrument tags, stems and the other listening passes). Do not name "
        f"'{top_tag}' or any other single PANNs subgenre as the track's genre, and hedge: this "
        "is a weak, single-source hint, not a confident measurement."
    )
    return label, note


# dev42: a second genre family strong enough to mention. Shared by the
# family-level fallback's rule and the prose-genre path below.
def _genre_second_family(fam_mass, fam, panns):
    """(word, tags text, mass) of the strongest family other than `fam` when
    it carries at least 20% and 40% of `fam`'s mass, else None."""
    others = [f for f in _GENRE_FAMILY_WORDS if f != fam and fam_mass.get(f, 0.0) > 0.0]
    if not others:
        return None
    sec = max(others, key=lambda f: fam_mass[f])
    if fam_mass[sec] >= 0.20 and fam_mass[sec] >= 0.4 * fam_mass.get(fam, 0.0):
        tags = ", ".join(f"{a} {b * 100:.0f}%" for a, b in panns if _genre_family(a) == sec)
        return _GENRE_FAMILY_WORDS[sec], tags, fam_mass[sec]
    return None


def _reconcile_genre_core__pre6(moss_text, panns_genre_tags):


    """Compare listening-model GENRE_RANKED with PANNs genre tags. Returns (recommended_label, note)
    or (None, reason) when nothing useful can be said."""
    moss_labels = extract_genre_ranked_labels(moss_text or "")
    moss_from_prose = False
    if not moss_labels:
        hint = _extract_genre_hint_from_moss_prose(moss_text or "")
        if GENRE_DEV55:
            # dev55: also catches "Indie Pop / Synth-Pop piece" and adjective
            # lists; the dev54 extractor's answer is kept when it found one
            # and this finds nothing more specific.
            _c = _genre_prose_candidates(moss_text or "", window=600)
            if _c and (not hint or (hint in _c[0] and hint != _c[0]) or hint not in " / ".join(_c)):
                hint = _c[0]
        if hint:
            moss_labels = [hint]
            moss_from_prose = True
    panns = []
    for item in (panns_genre_tags or []):
        if len(item) >= 2:
            panns.append((_normalize_genre_token(item[0]), float(item[1])))
    if not moss_labels and not panns:
        return None, "unavailable"
    # AudioSet "genre" classes that say nothing about how a record SOUNDS:
    # they come from YouTube metadata about lyrics, scene or origin
    # ("Christian music", "Independent music", "Vocal music"...). Observed:
    # an indie-rock track framed as "indie rock (with christian lean)"
    # because "Christian" topped the list at 30% while grunge + indie, both
    # rock, together outweighed it. They stay visible in the PANNs report,
    # but never steer the recommended genre.
    _non_stylistic = {
        "christian", "indie", "vocal-led (unspecified)", "asian (unspecified)",
        "traditional/folk (unspecified)", "african", "latin", "middle eastern",
    }
    panns = [(lab, p) for lab, p in panns if lab not in _non_stylistic]
    if not moss_labels and not panns:
        return None, "unavailable"

    moss_top = moss_labels[0] if moss_labels else ""
    moss_fam = _genre_family(moss_top) if moss_top else "other"
    panns_top = panns[0][0] if panns else ""
    panns_top_p = panns[0][1] if panns else 0.0
    panns_fam = _genre_family(panns_top) if panns_top else "other"

    # Aggregate PANNs mass per family
    fam_mass = {"rockish": 0.0, "electronicish": 0.0, "popish": 0.0, "other": 0.0}
    for lab, prob in panns:
        fam_mass[_genre_family(lab)] += prob

    # Strong electronic/dance PANNs vs rock/punk listening-model top → prefer electronic/pop framing
    _elec_lead = (fam_mass["electronicish"] + 0.5 * fam_mass["popish"]) > 1.2 * fam_mass["rockish"]
    rock_vs_elec = (
        moss_fam == "rockish"
        and _elec_lead
        and (
            fam_mass["electronicish"] >= 0.12
            or (panns_fam == "electronicish" and panns_top_p >= 0.10)
            or (fam_mass["electronicish"] + fam_mass["popish"] >= 0.18 and fam_mass["rockish"] < 0.08)
        )
    )
    if rock_vs_elec:
        # Build a human label from top PANNs electronic/pop tags
        prefer = [lab for lab, p in panns if _genre_family(lab) in ("electronicish", "popish")]
        if not prefer:
            prefer = [panns_top] if panns_top else ["electronic pop"]
        # Prefer compound if both pop and electronic mass present
        if fam_mass["electronicish"] >= 0.10 and fam_mass["popish"] >= 0.08:
            rec = "electronic / dance-pop"
        elif prefer:
            rec = prefer[0]
            if "pop" not in rec and fam_mass["popish"] >= 0.08:
                rec = f"{rec} pop" if "dance" in rec or "electronic" in rec else f"pop / {rec}"
        else:
            rec = "electronic pop"
        note = (
            f"MOSS-Music GENRE_RANKED led with '{moss_top}' (rock/punk family), but the "
            f"independent PANNs genre signal favours electronic/dance/pop "
            f"(tags: {', '.join(f'{a} {b*100:.0f}%' for a,b in panns[:4]) or 'n/a'}). "
            "A secondary guitar texture alone should not keep a dance/electronic track "
            "ranked as pop-punk/rock. Use this recommended label as the primary genre "
            "framing; mention rock/punk only as a minor secondary flavour if still audible."
        )
        return rec, note

    # Agreeing families or weak PANNs → keep listening-model top when present
    moss_src = (
        "extracted from MOSS-Music's free-text description (it did not "
        "produce the structured GENRE_RANKED field this pass, so treat this as "
        "a looser signal than a real ranked list, though still MOSS-Music's own "
        "assessment of the recording rather than a generic AudioSet tag)"
        if moss_from_prose else "GENRE_RANKED"
    )
    if moss_top:
        if panns and (panns_top == moss_top or (moss_fam != "other" and panns_fam == moss_fam)):
            note = (
                f"MOSS-Music top '{moss_top}' ({moss_src}) aligns with independent PANNs signal "
                f"('{panns_top}' {panns_top_p*100:.0f}%). Use it as primary."
            )
            if GENRE_SECOND_FAMILY_LEAN and moss_from_prose and moss_fam in _GENRE_FAMILY_WORDS:
                _lean = _genre_second_family(fam_mass, moss_fam, panns)
                if _lean:
                    _fam_tags = ", ".join(
                        f"{a} {b * 100:.0f}%" for a, b in panns if _genre_family(a) == moss_fam)
                    return f"{moss_top} (with {_lean[0]} lean)", (
                        f"MOSS-Music top '{moss_top}' ({moss_src}) is in the same family as the independent "
                        f"PANNs tags ({_fam_tags}; {fam_mass[moss_fam] * 100:.0f}% together); those "
                        "single subgenre tags are too unreliable to name. The PANNs tags also carry "
                        f"a substantial {_lean[0]} component ({_lean[1]}; {_lean[2] * 100:.0f}% "
                        f"together), so use '{moss_top} (with {_lean[0]} lean)' as primary and "
                        f"mention the {_lean[0]} element as a secondary flavour."
                    )
            return moss_top, note
        # Families are compared by total PANNs mass, not by the single top
        # tag: two rock tags at 21% + 15% outweigh one gospel tag at 19%.
        if panns and panns_fam != moss_fam and moss_fam != "other" and fam_mass.get(moss_fam, 0.0) >= panns_top_p:
            note = (
                f"MOSS-Music top '{moss_top}' ({moss_src}) is supported by the independent PANNs "
                f"signal as a family ({moss_fam.replace('ish', '')} tags total "
                f"{fam_mass.get(moss_fam, 0.0)*100:.0f}%, vs top single tag '{panns_top}' "
                f"{panns_top_p*100:.0f}%). Use it as primary."
            )
            return moss_top, note
        # The independent signal points at a different family, but the listening model's OWN
        # ranked list names a genre in that family a place or two down: use
        # the listening model's label. an AI-generated test track (2010s dance-pop) ranked "pop, dance-pop,
        # synth-pop, electronic" while PANNs gave disco + electronic + house
        # (53% together); it came out as "pop (with disco lean)" when the listening model's
        # own second choice already reconciled the two.
        # Only between the three families this module can tell apart: a listening-model
        # top outside them (hip hop, country...) is not "the umbrella" of an
        # electronic alternate, and swapping it out would lose the listening model's call.
        if panns and len(moss_labels) > 1 and moss_fam != "other":
            _fams = [f for f in fam_mass if f != "other"]
            dom_fam = max(_fams, key=lambda f: fam_mass[f])
            dom_mass = fam_mass[dom_fam]
            if dom_fam != moss_fam and dom_mass >= max(0.20, fam_mass.get(moss_fam, 0.0) + 0.10):
                # dev47: a rock/punk top against pop/dance-leaning tags may
                # give way to a pop OR dance label from the listening model's own list ("pop
                # rock; power pop; dance-pop" with tags led by "pop music").
                # The strongest family's label is still preferred.
                _alt_fams = (dom_fam,)
                if (GENRE_ROCK_TOP_POP_SIDE_ALT and moss_fam == "rockish"
                        and dom_fam in ("popish", "electronicish")):
                    _alt_fams = (dom_fam, "electronicish" if dom_fam == "popish" else "popish")
                _alt_order = [a for f in _alt_fams for a in moss_labels[1:4] if _genre_family(a) == f
                              # a club label ("house") from the other family
                              # needs tags of its own; a pop compound
                              # ("dance-pop", "synth-pop") does not.
                              and (f == dom_fam or f == "popish"
                                   or "pop" in _normalize_genre_token(a)
                                   or fam_mass.get(f, 0.0) >= 0.10)]
                for alt in _alt_order:
                    if _genre_family(alt) in _alt_fams:
                        _tags = ", ".join(
                            f"{a} {b*100:.0f}%" for a, b in panns[:4] if _genre_family(a) == dom_fam
                        )
                        _tags = (_tags + "; ") if _tags else ""
                        _moss_words = [w for w in re.split(r"[\s/-]+", moss_top) if w]
                        _umbrella = bool(_moss_words) and all(
                            re.search(rf"\b{re.escape(w)}\b", alt) for w in _moss_words
                        )
                        note = (
                            f"MOSS-Music ranked '{moss_top}' first and '{alt}' among its next choices "
                            f"({moss_src}); the independent PANNs signal sits in the "
                            f"{dom_fam.replace('ish', '')} family ({_tags}"
                            f"{dom_mass*100:.0f}% together), which MOSS-Music's '{alt}' already names. "
                            if _genre_family(alt) == dom_fam else
                            f"MOSS-Music ranked '{moss_top}' first and '{alt}' among its next choices "
                            f"({moss_src}); the independent PANNs signal sits on the pop/dance "
                            f"side, led by the {dom_fam.replace('ish', '')} family ({_tags}"
                            f"{dom_mass*100:.0f}% together, against "
                            f"{fam_mass.get(moss_fam, 0.0)*100:.0f}% for rock/punk), which MOSS-Music's "
                            f"'{alt}' already names. "
                        ) + (
                            f"Use '{alt}' as primary"
                            + (f"; '{moss_top}' is the broader umbrella." if _umbrella
                               else f", with '{moss_top}' as MOSS-Music's broader first impression.")
                        )
                        return alt, note
        if panns and panns_top != moss_top and (panns_fam != moss_fam or moss_fam == "other") and panns_top_p >= 0.15:
            note = (
                f"MOSS-Music top is '{moss_top}' ({moss_src}); independent PANNs top is '{panns_top}' "
                f"({panns_top_p*100:.0f}%). Families differ — prefer a broader description "
                f"that acknowledges both rather than forcing one scene label. "
                f"Default conversational framing: '{moss_top}' with possible '{panns_top}' lean."
            )
            return f"{moss_top} (with {panns_top} lean)", note
        return moss_top, (
            f"Primary framing from MOSS-Music ({moss_src}): '{moss_top}'. "
            + (
                f"Independent PANNs tags: {', '.join(f'{a} {b*100:.0f}%' for a,b in panns[:4])}."
                if panns else "No strong independent PANNs genre tags."
            )
        )

    if panns_top and GENRE_LAST_RESORT_BY_FAMILY:
        _fam_framing = _genre_family_framing(panns, fam_mass)
        if _fam_framing:
            return _fam_framing
    if panns_top:
        # This is the true last resort: no structured GENRE_RANKED field AND
        # no usable genre-ish phrase anywhere in the listening model's own prose either, so the
        # ONLY signal left is a single raw AudioSet tag. PANNs was trained on
        # broad, often noisy web audio labels and is known to conflate closely
        # related guitar-rock subgenres (grunge / pop-punk / punk / alt-rock /
        # indie rock routinely score similarly on genuinely ambiguous guitar
        # tracks) -- treating its bare top-1 as a confident, singular genre
        # claim is exactly what produced a real "grunge" call on a pop-punk
        # song and a "punk" call on an indie-rock one. Flag it as weak,
        # single-signal evidence rather than a measured, confident framing.
        return panns_top, (
            f"No parseable GENRE_RANKED and no usable genre description in MOSS-Music's own text; "
            f"the ONLY signal available is a single raw PANNs/AudioSet tag ('{panns_top}', "
            f"{panns[0][1]*100:.0f}%). This classifier is trained on broad web-audio labels "
            "and routinely confuses closely-related guitar-rock subgenres (grunge, pop-punk, "
            "punk, alt-rock, indie rock often score similarly on the same track), so treat "
            "this as a weak, single-source hint, not a confident measurement. Hedge "
            "accordingly ('the audio tags lean toward X, though that classifier isn't "
            "reliable for this kind of fine-grained rock/pop distinction') rather than "
            "stating it as the track's genre."
        )
    return None, "unavailable"


def _reconcile_genre_core__07(moss_text, panns_genre_tags):


    """Compare listening-model GENRE_RANKED with PANNs genre tags. Returns (recommended_label, note)
    or (None, reason) when nothing useful can be said."""
    moss_labels = extract_genre_ranked_labels(moss_text or "")
    moss_from_prose = False
    if not moss_labels:
        hint = _extract_genre_hint_from_moss_prose(moss_text or "")
        if GENRE_DEV55:
            # dev55: also catches "Indie Pop / Synth-Pop piece" and adjective
            # lists; the dev54 extractor's answer is kept when it found one
            # and this finds nothing more specific.
            _c = _genre_prose_candidates(moss_text or "", window=600)
            if _c and (not hint or (hint in _c[0] and hint != _c[0]) or hint not in " / ".join(_c)):
                hint = _c[0]
        if hint:
            moss_labels = [hint]
            moss_from_prose = True
    panns = []
    for item in (panns_genre_tags or []):
        if len(item) >= 2:
            panns.append((_normalize_genre_token(item[0]), float(item[1])))
    if not moss_labels and not panns:
        return None, "unavailable"
    # AudioSet "genre" classes that say nothing about how a record SOUNDS:
    # they come from YouTube metadata about lyrics, scene or origin
    # ("Christian music", "Independent music", "Vocal music"...). Observed:
    # an indie-rock track framed as "indie rock (with christian lean)"
    # because "Christian" topped the list at 30% while grunge + indie, both
    # rock, together outweighed it. They stay visible in the PANNs report,
    # but never steer the recommended genre.
    _non_stylistic = {
        "christian", "indie", "vocal-led (unspecified)", "asian (unspecified)",
        "traditional/folk (unspecified)", "african", "latin", "middle eastern",
    }
    panns = [(lab, p) for lab, p in panns if lab not in _non_stylistic]
    if not moss_labels and not panns:
        return None, "unavailable"

    moss_top = moss_labels[0] if moss_labels else ""
    moss_fam = _genre_family(moss_top) if moss_top else "other"
    panns_top = panns[0][0] if panns else ""
    panns_top_p = panns[0][1] if panns else 0.0
    panns_fam = _genre_family(panns_top) if panns_top else "other"

    # Aggregate PANNs mass per family
    fam_mass = {"rockish": 0.0, "electronicish": 0.0, "popish": 0.0, "other": 0.0}
    for lab, prob in panns:
        fam_mass[_genre_family(lab)] += prob

    # Strong electronic/dance PANNs vs rock/punk listening-model top → prefer electronic/pop framing
    _elec_lead = (fam_mass["electronicish"] + 0.5 * fam_mass["popish"]) > 1.2 * fam_mass["rockish"]
    rock_vs_elec = (
        moss_fam == "rockish"
        and _elec_lead
        and (
            fam_mass["electronicish"] >= 0.12
            or (panns_fam == "electronicish" and panns_top_p >= 0.10)
            or (fam_mass["electronicish"] + fam_mass["popish"] >= 0.18 and fam_mass["rockish"] < 0.08)
        )
    )
    if rock_vs_elec:
        # Build a human label from top PANNs electronic/pop tags
        prefer = [lab for lab, p in panns if _genre_family(lab) in ("electronicish", "popish")]
        if not prefer:
            prefer = [panns_top] if panns_top else ["electronic pop"]
        # Prefer compound if both pop and electronic mass present
        if fam_mass["electronicish"] >= 0.10 and fam_mass["popish"] >= 0.08:
            rec = "electronic / dance-pop"
        elif prefer:
            rec = prefer[0]
            if "pop" not in rec and fam_mass["popish"] >= 0.08:
                rec = f"{rec} pop" if "dance" in rec or "electronic" in rec else f"pop / {rec}"
        else:
            rec = "electronic pop"
        note = (
            f"MOSS-Music GENRE_RANKED led with '{moss_top}' (rock/punk family), but the "
            f"independent PANNs genre signal favours electronic/dance/pop "
            f"(tags: {', '.join(f'{a} {b*100:.0f}%' for a,b in panns[:4]) or 'n/a'}). "
            "A secondary guitar texture alone should not keep a dance/electronic track "
            "ranked as pop-punk/rock. Use this recommended label as the primary genre "
            "framing; mention rock/punk only as a minor secondary flavour if still audible."
        )
        return rec, note

    # Agreeing families or weak PANNs → keep listening-model top when present
    moss_src = (
        "extracted from MOSS-Music's free-text description (it did not "
        "produce the structured GENRE_RANKED field this pass, so treat this as "
        "a looser signal than a real ranked list, though still MOSS-Music's own "
        "assessment of the recording rather than a generic AudioSet tag)"
        if moss_from_prose else "GENRE_RANKED"
    )
    if moss_top:
        if panns and panns_fam == moss_fam:
            note = (
                f"MOSS-Music top '{moss_top}' ({moss_src}) aligns with independent PANNs signal "
                f"('{panns_top}' {panns_top_p*100:.0f}%). Use it as primary."
            )
            if GENRE_SECOND_FAMILY_LEAN and moss_from_prose and moss_fam in _GENRE_FAMILY_WORDS:
                _lean = _genre_second_family(fam_mass, moss_fam, panns)
                if _lean:
                    _fam_tags = ", ".join(
                        f"{a} {b * 100:.0f}%" for a, b in panns if _genre_family(a) == moss_fam)
                    return f"{moss_top} (with {_lean[0]} lean)", (
                        f"MOSS-Music top '{moss_top}' ({moss_src}) is in the same family as the independent "
                        f"PANNs tags ({_fam_tags}; {fam_mass[moss_fam] * 100:.0f}% together); those "
                        "single subgenre tags are too unreliable to name. The PANNs tags also carry "
                        f"a substantial {_lean[0]} component ({_lean[1]}; {_lean[2] * 100:.0f}% "
                        f"together), so use '{moss_top} (with {_lean[0]} lean)' as primary and "
                        f"mention the {_lean[0]} element as a secondary flavour."
                    )
            return moss_top, note
        # Families are compared by total PANNs mass, not by the single top
        # tag: two rock tags at 21% + 15% outweigh one gospel tag at 19%.
        if panns and panns_fam != moss_fam and moss_fam != "other" and fam_mass.get(moss_fam, 0.0) >= panns_top_p:
            note = (
                f"MOSS-Music top '{moss_top}' ({moss_src}) is supported by the independent PANNs "
                f"signal as a family ({moss_fam.replace('ish', '')} tags total "
                f"{fam_mass.get(moss_fam, 0.0)*100:.0f}%, vs top single tag '{panns_top}' "
                f"{panns_top_p*100:.0f}%). Use it as primary."
            )
            return moss_top, note
        # The independent signal points at a different family, but the listening model's OWN
        # ranked list names a genre in that family a place or two down: use
        # the listening model's label. an AI-generated test track (2010s dance-pop) ranked "pop, dance-pop,
        # synth-pop, electronic" while PANNs gave disco + electronic + house
        # (53% together); it came out as "pop (with disco lean)" when the listening model's
        # own second choice already reconciled the two.
        # Only between the three families this module can tell apart: a listening-model
        # top outside them (hip hop, country...) is not "the umbrella" of an
        # electronic alternate, and swapping it out would lose the listening model's call.
        if panns and len(moss_labels) > 1 and moss_fam != "other":
            _fams = [f for f in fam_mass if f != "other"]
            dom_fam = max(_fams, key=lambda f: fam_mass[f])
            dom_mass = fam_mass[dom_fam]
            if dom_fam != moss_fam and dom_mass >= max(0.20, fam_mass.get(moss_fam, 0.0) + 0.10):
                # dev47: a rock/punk top against pop/dance-leaning tags may
                # give way to a pop OR dance label from the listening model's own list ("pop
                # rock; power pop; dance-pop" with tags led by "pop music").
                # The strongest family's label is still preferred.
                _alt_fams = (dom_fam,)
                if (GENRE_ROCK_TOP_POP_SIDE_ALT and moss_fam == "rockish"
                        and dom_fam in ("popish", "electronicish")):
                    _alt_fams = (dom_fam, "electronicish" if dom_fam == "popish" else "popish")
                _alt_order = [a for f in _alt_fams for a in moss_labels[1:4] if _genre_family(a) == f
                              # a club label ("house") from the other family
                              # needs tags of its own; a pop compound
                              # ("dance-pop", "synth-pop") does not.
                              and (f == dom_fam or f == "popish"
                                   or "pop" in _normalize_genre_token(a)
                                   or fam_mass.get(f, 0.0) >= 0.10)]
                for alt in _alt_order:
                    if _genre_family(alt) in _alt_fams:
                        _tags = ", ".join(
                            f"{a} {b*100:.0f}%" for a, b in panns[:4] if _genre_family(a) == dom_fam
                        )
                        _tags = (_tags + "; ") if _tags else ""
                        _moss_words = [w for w in re.split(r"[\s/-]+", moss_top) if w]
                        _umbrella = bool(_moss_words) and all(
                            re.search(rf"\b{re.escape(w)}\b", alt) for w in _moss_words
                        )
                        note = (
                            f"MOSS-Music ranked '{moss_top}' first and '{alt}' among its next choices "
                            f"({moss_src}); the independent PANNs signal sits in the "
                            f"{dom_fam.replace('ish', '')} family ({_tags}"
                            f"{dom_mass*100:.0f}% together), which MOSS-Music's '{alt}' already names. "
                            if _genre_family(alt) == dom_fam else
                            f"MOSS-Music ranked '{moss_top}' first and '{alt}' among its next choices "
                            f"({moss_src}); the independent PANNs signal sits on the pop/dance "
                            f"side, led by the {dom_fam.replace('ish', '')} family ({_tags}"
                            f"{dom_mass*100:.0f}% together, against "
                            f"{fam_mass.get(moss_fam, 0.0)*100:.0f}% for rock/punk), which MOSS-Music's "
                            f"'{alt}' already names. "
                        ) + (
                            f"Use '{alt}' as primary"
                            + (f"; '{moss_top}' is the broader umbrella." if _umbrella
                               else f", with '{moss_top}' as MOSS-Music's broader first impression.")
                        )
                        return alt, note
        if panns and panns_fam != moss_fam and panns_top_p >= 0.15:
            note = (
                f"MOSS-Music top is '{moss_top}' ({moss_src}); independent PANNs top is '{panns_top}' "
                f"({panns_top_p*100:.0f}%). Families differ — prefer a broader description "
                f"that acknowledges both rather than forcing one scene label. "
                f"Default conversational framing: '{moss_top}' with possible '{panns_top}' lean."
            )
            return f"{moss_top} (with {panns_top} lean)", note
        return moss_top, (
            f"Primary framing from MOSS-Music ({moss_src}): '{moss_top}'. "
            + (
                f"Independent PANNs tags: {', '.join(f'{a} {b*100:.0f}%' for a,b in panns[:4])}."
                if panns else "No strong independent PANNs genre tags."
            )
        )

    if panns_top and GENRE_LAST_RESORT_BY_FAMILY:
        _fam_framing = _genre_family_framing(panns, fam_mass)
        if _fam_framing:
            return _fam_framing
    if panns_top:
        # This is the true last resort: no structured GENRE_RANKED field AND
        # no usable genre-ish phrase anywhere in the listening model's own prose either, so the
        # ONLY signal left is a single raw AudioSet tag. PANNs was trained on
        # broad, often noisy web audio labels and is known to conflate closely
        # related guitar-rock subgenres (grunge / pop-punk / punk / alt-rock /
        # indie rock routinely score similarly on genuinely ambiguous guitar
        # tracks) -- treating its bare top-1 as a confident, singular genre
        # claim is exactly what produced a real "grunge" call on a pop-punk
        # song and a "punk" call on an indie-rock one. Flag it as weak,
        # single-signal evidence rather than a measured, confident framing.
        return panns_top, (
            f"No parseable GENRE_RANKED and no usable genre description in MOSS-Music's own text; "
            f"the ONLY signal available is a single raw PANNs/AudioSet tag ('{panns_top}', "
            f"{panns[0][1]*100:.0f}%). This classifier is trained on broad web-audio labels "
            "and routinely confuses closely-related guitar-rock subgenres (grunge, pop-punk, "
            "punk, alt-rock, indie rock often score similarly on the same track), so treat "
            "this as a weak, single-source hint, not a confident measurement. Hedge "
            "accordingly ('the audio tags lean toward X, though that classifier isn't "
            "reliable for this kind of fine-grained rock/pop distinction') rather than "
            "stating it as the track's genre."
        )
    return None, "unavailable"


def _reconcile_genre_core(*args, **kwargs):
    """GENRE_OTHER_FAMILY_STRICT picks the 0.7.1-pre6 version; False gives 0.7's."""
    return (_reconcile_genre_core__pre6 if GENRE_OTHER_FAMILY_STRICT else _reconcile_genre_core__07)(*args, **kwargs)


_GENRE_PLAIN_LINE_RE = re.compile(r"(?im)^[ \t*_]*genre[ \t*_]*[:=][ \t]*(.+)$")


def _genre_plain_list_line(moss_text):
    """dev56: "GENRE_RANKED=1) a; 2) b; 3) c" built from a plain
    "Genre: a / b / c (...)" line in the listening model's opening, or ""
    when the text already has a ranked list or no such line."""
    t = str(moss_text or "")
    if extract_genre_ranked_labels(t):
        return ""
    cut = len(t)
    for mk in ("\n\nLYRICS", "\n\n11. ERA / RELEASE PERIOD", "\n\nVOCAL / SINGER PROFILE",
               "[Independent signal-processing report", "\nRECOMMENDED "):
        k = t.find(mk)
        if 0 <= k < cut:
            cut = k
    m = _GENRE_PLAIN_LINE_RE.search(t[:cut])
    if not m:
        return ""
    blob = re.sub(r"\([^)]*\)", "", _strip_md_emphasis(m.group(1))).strip()
    parts = [p.strip(" .,;-*_") for p in re.split(r"\s*[/;,|]\s*", blob)]
    labels = []
    for p in parts:
        tok = _normalize_genre_token(p)
        if tok and _is_plausible_genre_label(tok) and _genre_family(tok) != "other" and tok not in labels:
            labels.append(tok)
    if not labels:
        return ""
    return "GENRE_RANKED=" + "; ".join(f"{i + 1}) {lab}" for i, lab in enumerate(labels[:5]))


def _genre_postpunk_master(moss_text):
    """dev56: "-15.7 LUFS integrated, peak-to-loudness ratio 15.6 dB" when
    the notes measure a dynamic, lightly limited master (-12.5 LUFS or
    quieter, PLR 12 dB or more), else None."""
    t = str(moss_text or "")
    m1 = re.search(r"EBU R128 integrated loudness: (-?\d+(?:\.\d+)?) LUFS", t)
    m2 = re.search(r"peak-to-loudness ratio \(PLR\): (-?\d+(?:\.\d+)?) dB", t)
    if not m1 or not m2:
        return None
    lufs, plr = float(m1.group(1)), float(m2.group(1))
    if lufs <= -12.5 and plr >= 12.0:
        return f"{lufs:.1f} LUFS integrated, peak-to-loudness ratio {plr:.1f} dB"
    return None


def _reconcile_genre_legacy(moss_text, panns_genre_tags):
    """Compare listening-model GENRE_RANKED with PANNs genre tags. Returns (recommended_label, note)
    or (None, reason) when nothing useful can be said. dev55 adds the checks
    described at GENRE_DEV55 around the dev54 decision."""
    if GENRE_DEV55 and GENRE_DEV56:
        try:
            _pl = _genre_plain_list_line(moss_text)
            if _pl:
                moss_text = _pl + "\n" + str(moss_text or "")
        except Exception:
            pass
    rec, note = _reconcile_genre_core(moss_text, panns_genre_tags)
    if not GENRE_DEV55 or not rec:
        return rec, note
    try:
        moss_labels = extract_genre_ranked_labels(moss_text or "")
        panns = [(_normalize_genre_token(it[0]), float(it[1])) for it in (panns_genre_tags or []) if len(it) >= 2]
        _non_sty = {"christian", "indie", "vocal-led (unspecified)", "asian (unspecified)",
                    "traditional/folk (unspecified)", "african", "latin", "middle eastern"}
        panns = [(a, b) for a, b in panns if a not in _non_sty]
        fam_mass = {"rockish": 0.0, "electronicish": 0.0, "popish": 0.0, "other": 0.0}
        for a, b in panns:
            fam_mass[_genre_family(a)] += b
        base = re.sub(r"\s*\(with [^)]*lean\)\s*$", "", rec).strip()
        # (a) electronic/pop ranked list with no rock label vs rock-dominant tags
        if moss_labels and _genre_family(base) in ("electronicish", "popish") and \
                not any(_genre_family(x) == "rockish" for x in moss_labels[:5]):
            other = fam_mass["electronicish"] + 0.5 * fam_mass["popish"]
            if fam_mass["rockish"] >= 0.40 and fam_mass["rockish"] >= 1.25 * other:
                _rt = ", ".join(f"{a} {b * 100:.0f}%" for a, b in panns if _genre_family(a) == "rockish")
                lean = _genre_family(moss_labels[0]).replace("ish", "") if _genre_family(moss_labels[0]) != "other" else moss_labels[0]
                _pp = _genre_postpunk_master(moss_text) if GENRE_DEV56 else None
                if _pp:
                    return f"post-punk / new wave (with {lean} lean)", (
                        f"MOSS-Music's ranked list named only electronic/pop labels ('{moss_labels[0]}' first), but "
                        f"the independent PANNs signal is led by the rock family ({_rt}; {fam_mass['rockish'] * 100:.0f}% "
                        f"together, against {fam_mass['electronicish'] * 100:.0f}% electronic and "
                        f"{fam_mass['popish'] * 100:.0f}% pop). A clean, steady, drum-machine-like production can read as "
                        f"electronic to the listening model. With a dynamic, lightly limited master ({_pp}) -- not "
                        "the loud, compressed master of modern electronic or dance pop -- this is the post-punk / "
                        "new wave sound: rock-band music (bass, guitar, drums) with a cool, mechanical, "
                        f"electronic-leaning feel. Frame the track as post-punk / new wave first, with the {lean} "
                        "quality as a secondary flavour, and describe the actual instruments from the stem evidence.")
                return f"rock (with {lean} lean)", (
                    f"MOSS-Music's ranked list named only electronic/pop labels ('{moss_labels[0]}' first), but "
                    f"the independent PANNs signal is led by the rock family ({_rt}; {fam_mass['rockish'] * 100:.0f}% "
                    f"together, against {fam_mass['electronicish'] * 100:.0f}% electronic and "
                    f"{fam_mass['popish'] * 100:.0f}% pop). A clean, steady, drum-machine-like production can read as "
                    "electronic to the listening model; frame the track as rock first, with the "
                    f"{lean} quality as a secondary flavour, and describe the actual instruments from the stem evidence.")
        # (b) bare family word refined by a specific label from the prose
        if base in ("rock", "pop", "electronic", "dance"):
            fam = _genre_family(base)
            for cand in _genre_prose_candidates(moss_text or ""):
                parts = [p.strip() for p in cand.split("/")]
                if parts and all(_genre_family(p) == fam for p in parts) and cand != base and \
                        any(len(p.split()) >= 2 or "-" in p for p in parts):
                    rest = rec[len(base):]
                    return cand + rest, (note + f" The listening model's own description names it more "
                                         f"specifically as '{cand}' (same family), so that is the label to use.")
        # (d) a "lean" taken from one lone tag of another family is dropped
        # (PANNs' single subgenre tags are the unreliable part: "grunge" 33%
        # on a synth-pop track).
        m = re.match(r"^(.*?)\s*\(with (.+?) lean\)\s*$", rec)
        if m and "Families differ" in (note or ""):
            lean_tag = _normalize_genre_token(m.group(2))
            lf = _genre_family(lean_tag)
            same = [(a, b) for a, b in panns if _genre_family(a) == lf] if lf != "other" else \
                [(a, b) for a, b in panns if a == lean_tag]
            if len(same) < 2 and sum(b for _a, b in same) < 0.40:
                return m.group(1).strip(), (note + f" The '{lean_tag}' lean rests on a single independent tag "
                                            "and is not used.")
        # (c) no redundant pop lean
        m = re.match(r"^(.*?)\s*\(with pop lean\)\s*$", rec)
        if m and re.search(r"\bpop\b", m.group(1)):
            return m.group(1).strip(), note
    except Exception:
        pass
    return rec, note


def _genre_guard_style_rows(text):
    """Read the displayed independent style evidence, not generated recommendations."""
    match = re.search(r"(?m)^styles heard: (.+)$", str(text or ""))
    if not match:
        return []
    return [(label.strip(), float(score) / 100.0)
            for label, score in re.findall(r"(?:^|;)\s*(.*?)\s+(\d+(?:\.\d+)?)%", match.group(1))]


def _genre_guard_labels(text):
    """Keep explicit ranked labels and discard their confidence annotations."""
    match = re.search(r"(?m)^GENRE_RANKED\s*=\s*(.+)$", str(text or ""))
    if not match:
        return []
    return [re.sub(r"\s*\([^)]*\)\s*$", "", label).strip().lower()
            for label in re.split(r"(?:^|;)\s*\d+[).]\s*", match.group(1)) if label.strip()]


def _genre_guard_style_reading(styles, recommended, labels):
    """GENRE_EVIDENCE_STYLE_READING: when the mastering-based post-punk / new wave promotion fired,
    decide from what the independent style model hears instead of falling back to the listening
    model's first label. Returns None to keep the promotion (the styles name post-punk or new wave),
    a (label, reason) pair for a rock reading, or False to leave the fallback to the caller."""
    rows = []
    for row in styles or []:
        if not isinstance(row, (list, tuple)) or len(row) < 2 or isinstance(row[1], bool):
            continue
        try:
            label, score = str(row[0]), float(row[1])
        except (TypeError, ValueError, OverflowError):
            continue
        if not 0 <= score <= 1 or score < GENRE_EVIDENCE_STYLE_MIN:
            continue
        parent, _, style = label.partition("---")
        if not style:                       # the displayed form: "Indie Rock (Rock)"
            m = re.match(r"^(.*?)\s*\(([^()]*)\)\s*$", label)
            style, parent = (m.group(1), m.group(2)) if m else (label, "")
        rows.append((style.strip(), parent.strip(), score))
    rows = rows[:GENRE_EVIDENCE_STYLE_TOP_K]
    if not rows:
        return False
    names = [re.sub(r"[\u2010-\u2015_-]+", " ", style.lower()).strip() for style, _p, _s in rows]
    if any(name in ("post punk", "new wave") for name in names):
        return None
    rock = [row for row in rows if row[1].lower() == "rock"]
    if len(rows) < 3 or len(rock) * 2 <= len(rows):
        return False
    m = re.search(r"\(with (.+?) lean\)", str(recommended))
    lean = m.group(1).strip() if m else _genre_family(labels[0]).replace("ish", "")
    indie = any(name in ("indie rock", "alternative rock") for name in names[:2])
    base = "alternative / indie rock" if indie else "rock"
    heard = "; ".join(f"{style} {score * 100:.0f}%" for style, _p, score in rows)
    # The reasoning opens like the earlier rock framing, so the scan's GENRE CHECK line is written
    # for it in the same way.
    return f"{base} (with {lean} lean)", (
        "MOSS-Music's ranked list named only electronic/pop labels ('" + labels[0] + "' first), but "
        "most of the independent style model's leading styles are rock (" + heard + "), agreeing "
        "with the PANNs rock-family reading. Frame the track as " + base + " first, with the " + lean
        + " quality as a secondary flavour, and describe the actual instruments from the stem "
        "evidence. GENRE EVIDENCE GUARD: mastering loudness and peak-to-loudness ratio cannot "
        "establish post-punk / new wave, and the style model does not name it here.")


def _genre_evidence_revision(text, styles, recommended, reason):
    """Revise only the two aggressive legacy overrides; return None otherwise.
    The migration tool loads these pure helpers directly, so its decisions match scans.
    """
    if not GENRE_EVIDENCE_GUARD:
        return None
    master_flip = (str(recommended).startswith("post-punk / new wave")
                   and "MOSS-Music's ranked list named only electronic/pop labels" in str(reason))
    dance_flip = ("MOSS-Music GENRE_RANKED led with" in str(reason)
                  and "independent PANNs genre signal favours electronic/dance/pop" in str(reason))
    if not (master_flip or dance_flip):
        return None
    labels = _genre_guard_labels(text)
    if not labels:
        return None
    def family_label(value):
        value = re.sub(r"[\u2010-\u2015_-]+", " ", str(value).lower()).strip()
        if re.fullmatch(r"post punk(?: revival)?", value):
            return "post-punk revival" if "revival" in value else "post-punk"
        return "new wave" if value == "new wave" else ""
    explicit = list(dict.fromkeys(family_label(x) for x in labels[:5] if family_label(x)))
    shown = []
    for row in styles or []:
        if not isinstance(row, (list, tuple)) or len(row) < 2 or isinstance(row[1], bool):
            continue
        try:
            label, score = row[0], float(row[1])
            if not 0 <= score <= 1 or score < GENRE_EVIDENCE_STYLE_MIN:
                continue
            label = str(label).split("---")[-1]
            label = re.sub(r"\s*\([^)]*\)\s*$", "", label).strip()
            shown.append(label)
        except (IndexError, TypeError, ValueError, OverflowError):
            continue
    shown = shown[:GENRE_EVIDENCE_STYLE_TOP_K]
    style_pp = {family_label(x) for x in shown if family_label(x)}
    supported = [x for x in explicit if x in style_pp]
    if supported:
        # Closely related explicit labels may coexist, but no absent scene label is invented.
        kept = [x for x in explicit if x in supported or (x in ("post-punk", "new wave")
                and any(y in ("post-punk", "new wave") for y in supported))]
        base = " / ".join(kept)
        influences = list(dict.fromkeys(x.lower() for x in shown if x.lower() in ("disco", "funk")))
        label = base + (" (with " + " / ".join(influences) + " influences)" if influences else "")
        return label, ("GENRE EVIDENCE GUARD: the listening ranking explicitly includes "
            + " / ".join(kept) + ", and independent style evidence also identifies "
            + " / ".join(supported) + ". Retain that supported identity alongside any disco/funk "
            "influences; dance-related tags do not exclude a rock/new-wave identity. "
            "These are stylistic interpretations, not mutually exclusive measured facts. "
            "Mastering loudness and peak-to-loudness ratio do not establish a genre.")
    if master_flip and globals().get("GENRE_EVIDENCE_STYLE_READING", True):
        try:
            reading = _genre_guard_style_reading(styles, recommended, labels)
        except NameError:
            reading = False                 # a tool that loads only some of these helpers keeps the earlier decision
        if reading is None:
            return None                     # the styles name post-punk / new wave: keep the promotion
        if reading:
            return reading
    if master_flip:
        return labels[0], ("GENRE EVIDENCE GUARD: retain the listening model's first choice, '"
            + labels[0] + "', provisionally. The broad PANNs rock-family reading disagrees, "
            "but mastering loudness and peak-to-loudness ratio cannot establish post-punk/new wave. "
            "There is no corroborated explicit listening/style identification of that scene here. "
            "Preserve the disagreement rather than asserting an exclusive genre or banning alternatives.")
    return None


def reconcile_genre(moss_text, panns_genre_tags, discogs_styles=None):
    """Keep legacy decisions except the setting-gated evidence corrections."""
    label, reason = _reconcile_genre_legacy(moss_text, panns_genre_tags)
    revision = _genre_evidence_revision(moss_text,
        discogs_styles if discogs_styles is not None else _genre_guard_style_rows(moss_text), label, reason)
    return revision if revision is not None else (label, reason)


_MOSS_GENRE_FIELDS = ('GENRE_RANKED', 'GENRE_ADJACENT', 'GENRE_RULED_OUT')
_MOSS_GENRE_LINE = re.compile(r'(?m)^(?P<prefix>[ \t]*(?P<name>GENRE_RANKED|GENRE_ADJACENT|GENRE_RULED_OUT)[ \t]*=[ \t]*)(?P<value>[^\r\n]*)')
_MOSS_GENRE_AMBIGUOUS = {'indie', 'alternative', 'garage', 'dance'}
# Match whole list entries, retaining rank numbers, confidence and separators.
_MOSS_GENRE_AMBIGUOUS_ENTRY = re.compile(
    r'(?P<prefix>^|[;,/]\s*)(?P<rank>\s*(?:\d+[).]\s*)?)'
    r'(?P<label>indie|alternative|garage|dance)'
    r'(?=\s*(?:\((?:confidence:\s*)?(?:high|medium|low)\))?\s*(?:[;,/]|$))', re.I)


def _moss_genre_ambiguous_entries(fields):
    entries = []
    for field in _MOSS_GENRE_FIELDS:
        if GENRE_CLARIFY_SKIP_RULED_OUT and field == "GENRE_RULED_OUT":
            continue    # pre53: narrowing an exclusion ("dance" -> "dance-pop") weakens it
        for match in _MOSS_GENRE_AMBIGUOUS_ENTRY.finditer(fields[field]):
            entries.append({'id': str(len(entries) + 1), 'field': field,
                            'label': match['label'], 'start': match.start('label'),
                            'end': match.end('label')})
    return entries


def _moss_genre_clarification_name(response, original):
    """Validate one answer; reject copied question/role text before accepting a name."""
    if not isinstance(response, str):
        raise ValueError('Wrong response type: expected one genre name')
    answer = response.strip()
    # Accept a single conventional answer prefix, but never interpret JSON or lists.
    answer = re.sub(r'^(?:ANSWER|GENRE|LABEL_\d+)\s*[=:]\s*', '', answer, flags=re.I)
    answer = answer.strip('"\'').strip()
    if GENRE_CLARIFY_SKIP_RULED_OUT:
        answer = answer.rstrip('.').strip()   # pre53: "indie folk." is "indie folk"
    if re.search(r'\b(?:excluded genre|ranked candidate|adjacent candidate|GENRE_RULED_OUT|GENRE_RANKED|GENRE_ADJACENT)\b', answer, re.I):
        raise ValueError('Model copied a role annotation instead of clarifying the genre')
    if answer.casefold().rstrip('.') in {'uncertain', 'unknown', 'unclear', 'null'}:
        return None
    if answer.casefold().rstrip('.') == original.casefold():
        raise ValueError('Model repeated the original ambiguous label')
    if not (len(answer) <= 80
            and re.fullmatch(r'[A-Za-z][A-Za-z -]*', answer)
            and len(answer.split()) <= 6
            and re.search(r'\b' + re.escape(original) + r'\b', answer, re.I)
            and answer.casefold() not in _MOSS_GENRE_AMBIGUOUS
            and not re.search(r'\b(?:uncertain|unknown|unclear|none|null|or|maybe|possibly|not)\b', answer, re.I)):
        raise ValueError('Rejected answer: expected one qualified genre name preserving the original label word')
    return answer


def _moss_clarify_genres(model, processor, audio, fields, item):
    """One question per ambiguous entry, at most one content/format retry each."""
    entries = _moss_genre_ambiguous_entries(fields)
    if not entries:
        return dict(fields)
    roles = {
        'GENRE_RANKED': 'The earlier report ranked this term as a possible match to the recording.',
        'GENRE_ADJACENT': 'The earlier report listed this term as an adjacent style.',
        'GENRE_RULED_OUT': 'The earlier report excluded this term. Do not turn it into a detected genre. '
                          'Audio alone may not reveal which subtype was meant by an exclusion; '
                          'in that case answer uncertain.',
    }
    item['clarification_attempts'] = []
    item['clarifications'] = []
    accepted, reasons = {}, {}
    for entry in entries:
        original = entry['label']
        context = (f'The ambiguous musical term is {json.dumps(original)}. '
                   + roles[entry['field']] + ' ')
        instruction = ('Listen to the recording. What specific musical style, if any, can this term '
                       'be clarified to without changing its intended meaning? Do not assume a rock '
                       'interpretation or invent specificity. Reply with only a single qualified '
                       'genre name containing the original term, or the single word uncertain. '
                       'No role annotations, parentheses, explanation, JSON, numbered list or field labels.')
        for attempt in range(2):
            if attempt == 0:
                request = context + '\n\n' + instruction
            else:
                # Do not include the rejected answer: that would invite another echo.
                request = ('Name the specific style intended by the term ' + json.dumps(original)
                           + ', only if the audio supports that interpretation. ' + roles[entry['field']]
                           + '\n\nYour previous answer repeated the input or was not a usable genre name. '
                           'Give only the qualified musical name, retaining the original term. '
                           'Do not repeat the bare term or describe its role in the report. '
                           'If you cannot resolve it, reply uncertain. No other words or formatting.')
            record = {'id': entry['id'], 'field': entry['field'], 'original': original,
                      'attempt': attempt + 1, 'prompt': request}
            item['clarification_attempts'].append(record)
            try:
                response = moss_generate(model, processor,
                    [{'role': 'user', 'content': [{'type': 'text', 'text': request},
                     {'type': 'audio', 'path': str(audio)}]}],
                    max_new_tokens=96, do_sample=False, loop_guard=True, fields=False)
                record['response'] = response
            except Exception as exc:
                reasons[entry['id']] = f'Clarification inference failed: {type(exc).__name__}: {exc}'
                record['error'] = reasons[entry['id']]
                break
            try:
                candidate = _moss_genre_clarification_name(response, original)
            except ValueError as exc:
                reasons[entry['id']] = str(exc)
                record['error'] = str(exc)
                continue
            if candidate is None:
                reasons[entry['id']] = 'Model explicitly returned uncertainty'
                record['status'] = 'explicit uncertainty'
            else:
                accepted[entry['id']] = candidate
                record['status'] = 'clarified'
            break
    result = dict(fields)
    for entry in reversed(entries):
        candidate = accepted.get(entry['id'])
        if candidate:
            text = result[entry['field']]
            result[entry['field']] = text[:entry['start']] + candidate + text[entry['end']:]
    for entry in entries:
        candidate = accepted.get(entry['id'])
        detail = {'field': entry['field'], 'original': entry['label'], 'replacement': candidate,
                  'status': 'clarified' if candidate else 'retained ambiguous'}
        if not candidate:
            detail['reason'] = reasons[entry['id']]
            item.setdefault('warnings', []).append(
                f"Retained {entry['label']!r} in {entry['field']}: {detail['reason']}; no candidate removed")
        item['clarifications'].append(detail)
    return result


def _moss_genre_clarifications_for_track(text, model, processor, audio):
    """Collect label changes before unloading MOSS; apply them after analysis."""
    if model is None or not audio:
        return {}
    matches = list(_MOSS_GENRE_LINE.finditer(text or ""))
    fields = {m["name"]: m["value"] for m in matches}
    if len(matches) != len(_MOSS_GENRE_FIELDS) or set(fields) != set(_MOSS_GENRE_FIELDS):
        return {}
    audit = {}
    clarified = _moss_clarify_genres(model, processor, audio, fields, audit)
    return {key: [value, clarified[key]] for key, value in fields.items()
            if value != clarified[key]}


def _deduplicate_final_genre_ranking(text):
    """Remove duplicate ranked labels only; preserve first spelling/confidence."""
    def clean(match):
        body = match.group(2)
        entries = re.split(r";[ \t]*", body)
        parsed = [re.fullmatch(r"[ \t]*\d+\)[ \t]*(.+?)[ \t]*", entry) for entry in entries]
        if not parsed or not all(parsed):
            return match.group(0)
        seen, kept = set(), []
        for entry in parsed:
            value = entry.group(1)
            label = re.sub(r"\s*\((?:confidence\s*:\s*)?(?:high|medium|low)\)\s*$", "", value, flags=re.I)
            key = re.sub(r"[\s_\-\u2010-\u2015]+", " ", label).strip().casefold()
            if key not in seen:
                seen.add(key)
                kept.append(value)
        if len(kept) == len(entries):
            return match.group(0)
        return match.group(1) + "; ".join(f"{i}) {value}" for i, value in enumerate(kept, 1))
    return re.sub(r"(?m)^([ \t]*GENRE_RANKED[ \t]*=[ \t]*)([^\r\n]*)", clean, text)


def _apply_moss_genre_clarifications(text, changes, panns_tags):
    """Update only the genre fields and recommendation after musical analysis."""
    if not changes:
        return text
    matches = list(_MOSS_GENRE_LINE.finditer(text))
    applicable = {key: pair for key, pair in changes.items()
                  if [m["value"] for m in matches if m["name"] == key] == [pair[0]]}
    if not applicable:
        return text
    recommendation = re.compile(r"(?m)^RECOMMENDED GENRE FOR DISCUSSION:[^\r\n]*")
    blocks = list(recommendation.finditer(text))
    if len(blocks) != 1:
        return text

    updated = _MOSS_GENRE_LINE.sub(
        lambda m: m["prefix"] + applicable[m["name"]][1] if m["name"] in applicable else m[0],
        text)
    # The original recommendation precedes Discogs-400 and later analysis blocks.
    # Use the same evidence boundary and normalize short confidence syntax only
    # in the temporary input to reconciliation.
    block = recommendation.search(updated)
    basis = _MOSS_GENRE_LINE.sub(
        lambda m: m["prefix"] + re.sub(r"\((high|medium|low)\)", r"(confidence: \1)",
                                      m["value"], flags=re.I)
        if m["name"] == "GENRE_RANKED" else m[0],
        updated[:block.start()])
    label, reason = reconcile_genre(basis, panns_tags, _genre_guard_style_rows(updated))
    replacement = format_recommended_genre_block(label, reason).strip()
    return updated[:block.start()] + replacement + updated[block.end():]


def format_recommended_genre_block(recommended, note):
    if not recommended:
        return (
            "\n\nRECOMMENDED GENRE FOR DISCUSSION: unavailable. "
            "Fall back to GENRE_RANKED with caution; do not over-commit to scene labels "
            "from a single instrument texture."
        )
    return (
        f"\n\nRECOMMENDED GENRE FOR DISCUSSION: {recommended}. "
        f"Reasoning: {note} "
        "This is the primary genre framing for the user. "
        "Do not lead with a conflicting early GENRE_RANKED label when this block "
        "explicitly revises rock/punk vs electronic/dance identity."
    )


def _extract_era_soft_hints(path):
    """Pull encoder/ISRC tag values for era estimation ONLY.

    Neither value is a confirmed release year (an encoder string dates when
    THIS FILE was ripped/encoded, and an ISRC's embedded year is a
    registration year that can predate or postdate the actual release), so
    both are kept out of meta['year'] / the "confirmed metadata" pipeline
    entirely and are only ever surfaced to the isolated ERA pass as hedged,
    explicitly-labelled soft evidence.

    Read-only, self-contained (opens its own MutagenFile handle rather than
    threading through the main extract_audio_metadata() call), and never
    raises -- any failure just yields an empty dict, which callers already
    treat as "nothing to add".
    """
    out = {"encoder": "", "isrc": "", "isrc_year_hint": None}
    if not path or str(path).startswith(("http://", "https://")):
        return out
    if not (METADATA_AVAILABLE and MutagenFile is not None):
        return out
    try:
        f = MutagenFile(path)
        tags = getattr(f, "tags", None)
        if tags is None:
            return out
        out["encoder"] = _first_tag(tags, ENCODER_KEYS)
        isrc = _first_tag(tags, ISRC_KEYS)
        out["isrc"] = isrc
        if isrc:
            compact = re.sub(r"[^A-Za-z0-9]", "", isrc)
            if len(compact) == 12 and compact[5:7].isdigit():
                yy = int(compact[5:7])
                # Standard rolling two-digit-year guess: nothing in the ISRC
                # itself states the century, so treat "not yet reached this
                # year" as 2000s and everything else as 1900s. This is a
                # rough bound, not a precise reading -- phrased as such below.
                current_yy = int(str(datetime.datetime.now().year)[-2:])
                century = 2000 if yy <= current_yy else 1900
                out["isrc_year_hint"] = str(century + yy)
    except Exception:
        pass
    return out


def _era_acoustic_signal_snapshot(original_path):
    """Cheap, read-only measurements used ONLY to hand the isolated ERA pass
    real numbers instead of asking it to judge mastering loudness and
    bandwidth purely by ear in the same breath as guessing a decade.

    Deliberately independent of the main objective-report pipeline: reuses
    the same existing, already-cached, side-effect-free measurement
    functions (measure_ebur128_loudness, measure_crest_factor_db,
    _prod_hf_cutoff via _prod_load_excerpts) on the ORIGINAL, untouched local
    file path only -- never resolved_path (the low-rate MOSS-Music input WAV),
    dsp_path, or any shared temp-file dict -- so this cannot perturb any
    other measurement, cache, or code path used elsewhere in the pipeline.
    Every step is independently wrapped, so a failure here just means fewer
    numbers get returned; it never raises and never blocks the (unchanged)
    audio-only era pass from running.

    Returns a dict with keys lufs, lra, crest_db, bandwidth_lines
    (list[str]); any of which may be None / empty.
    """
    out = {"lufs": None, "lra": None, "crest_db": None, "bandwidth_lines": []}
    if not original_path or str(original_path).startswith(("http://", "https://")):
        return out
    if not os.path.exists(original_path):
        return out
    try:
        lufs, lra, _src = measure_ebur128_loudness(original_path)
        out["lufs"], out["lra"] = lufs, lra
    except Exception:
        pass
    try:
        out["crest_db"] = measure_crest_factor_db(original_path)
    except Exception:
        pass
    try:
        dur = _resolve_total_duration(track_path=original_path)
        if dur and dur >= 3.0:
            chunks, sr = _prod_load_excerpts(original_path, dur, mono=False)
            if chunks and sr:
                out["bandwidth_lines"] = _prod_hf_cutoff(chunks, sr, source_path=original_path) or []
    except Exception:
        pass
    return out


def _era_pass_label(era_result):
    """dev55: heading note for an era pass that gave no ERA_ESTIMATE."""
    if not ERA_DEV55 or re.search(r"ERA[_ ]ESTIMATE\s*[=:]", str(era_result or ""), re.I):
        return ""
    return ("; it gave a general description and NO era estimate -- its words such as 'modern' or "
            "a decade mentioned in passing describe sound, not a dating; date the track from the "
            "release information and the measured era evidence")


def _era_loudness_prior(lufs, lra, crest_db):
    """Hedged decade-band prior from mastering-loudness trends alone.

    Based on the well-documented "loudness war" progression in mastering
    practice: wide/quiet masters predate roughly 1990, loudness climbed
    through the 1990s-2000s, peaked in tightly-limited masters roughly
    2000-2013, and has eased only slightly (not reverted) since streaming
    loudness normalization became common after roughly 2015. This is a soft,
    overlapping prior -- genre, mastering-house choice, and remasters all
    break it -- so every band is phrased as a range that INCLUDES the
    present, never as a decade claim by itself.
    """
    wide = (crest_db is not None and crest_db >= 14.0) or (lra is not None and lra >= 9.0)
    loud_tight = (
        lufs is not None and lufs >= -9.0
        and ((lra is not None and lra <= 4.0) or (crest_db is not None and crest_db <= 7.0))
    )
    if wide:
        return (
            "Loudness prior: wide dynamic range / high crest factor is most consistent with "
            "lightly-mastered material, more common before roughly 1990 -- but audiophile, "
            "classical, jazz, and plenty of intentionally dynamic modern masters also measure "
            "this way at any point since. Weak evidence by itself, not a decade."
        )
    if loud_tight:
        return (
            "Loudness prior: loud, tightly limited mastering. This has been completely "
            "ordinary for anything mastered from roughly the early 2000s through today -- "
            "loudness-war-style limiting did not meaningfully ease until streaming loudness "
            "normalization started pulling it back only slightly after roughly 2015, and "
            "current commercial masters are still frequently this loud. This reading alone "
            "cannot separate a 2004 release from a brand-new one."
        )
    if lufs is not None or lra is not None or crest_db is not None:
        return (
            "Loudness prior: moderate loudness/dynamics. This band overlaps 1990s mainstream "
            "mastering through today's streaming-era masters and is the least discriminating "
            "reading -- treat it as uninformative for era unless other evidence corroborates "
            "a specific period."
        )
    return ""


def _build_era_supplementary_evidence(original_path, meta=None, *, audio_source=None):
    """Assemble a short, hedged evidence block for the ISOLATED era pass, but
    ONLY when the file has no usable YEAR tag -- that absence is the entire
    reason era detection has to guess in the first place, so tracks that
    already resolve a confirmed year via the normal metadata path are left
    completely untouched by this function and by its caller.

    Strictly additive: every measurement here is independent/read-only (see
    _era_acoustic_signal_snapshot and _extract_era_soft_hints docstrings),
    and this function itself is wrapped by its caller so any failure just
    means the era pass falls back to its original audio-only prompt.
    Everything produced here is explicitly framed as soft/uncertain evidence,
    matching ERA_ANALYSIS_PROMPT's own caution -- never as a confirmed year,
    which remains reserved for the separate YEAR-tag directive.
    """
    if not original_path or str(original_path).startswith(("http://", "https://")):
        return ""
    if not os.path.exists(original_path):
        return ""

    lines = []

    try:
        hints = _extract_era_soft_hints(original_path)
        if hints.get("isrc_year_hint"):
            lines.append(
                f"- ISRC embedded in file tags: {hints['isrc']} -> registration year suggests "
                f"~{hints['isrc_year_hint']}. ISRC registration year is NOT the release year "
                "(reissues and compilations can carry an older or newer ISRC than the "
                "recording itself) -- treat this only as a loose bound, not a date."
            )
        if hints.get("encoder"):
            lines.append(
                f"- Encoder tag embedded in file: \"{hints['encoder']}\". This dates when THIS "
                "FILE was encoded/ripped, not necessarily when the music was recorded or "
                "released -- it can rule out a later date for this file's own digitization, "
                "but says nothing definitive about the recording's era."
            )
    except Exception:
        pass

    try:
        snap = _era_acoustic_signal_snapshot(audio_source or original_path)
        if snap.get("lufs") is not None:
            lines.append(f"- Measured integrated loudness: {snap['lufs']} LUFS.")
        if snap.get("lra") is not None:
            lines.append(f"- Measured loudness range (LRA): {snap['lra']} LU.")
        if snap.get("crest_db") is not None:
            lines.append(f"- Measured crest factor: {snap['crest_db']} dB.")
        prior = _era_loudness_prior(snap.get("lufs"), snap.get("lra"), snap.get("crest_db"))
        if prior:
            lines.append(f"- {prior}")
        for bw_line in (snap.get("bandwidth_lines") or []):
            lines.append(f"- {bw_line}")
    except Exception:
        pass

    if not lines:
        return ""

    return (
        "\n\nSUPPLEMENTARY MEASURED EVIDENCE (this file has no usable release-year tag, so "
        "these numbers are provided to ground your estimate in something more concrete than "
        "impression alone):\n" + "\n".join(lines) +
        "\n\nTreat every line above as one additional data point, not a verdict. None of them "
        "individually fixes a decade -- weigh them alongside what you actually hear, and say "
        "uncertain or give a broad range if they don't converge with the audible evidence."
    )


def _build_era_remaster_caveat(objective_report):
    """Attach the loud/tight-mastering era caveat DIRECTLY to the ERA /
    RELEASE PERIOD block, not only to the RECOMMENDED DYNAMICS block.

    Observed regression: the exact same caveat, sitting only in the dynamics
    block, correctly steered one run to a genre/style-based 1990s call for a
    remastered track, then a separately-run conversation on the SAME saved
    data landed on "modern... sonically it feels current" -- reading the
    caveat's own "modern master or remaster" phrase and leaning on "modern"
    as the conclusion, the opposite of the hedge it was written to produce.
    The likely mechanism: an era question naturally draws the model to the
    ERA / RELEASE PERIOD block, which is exactly where the listening model's own text still
    says "contemporary" (the pass this whole caveat exists to correct for) --
    the dynamics block sits elsewhere and can be consulted or not, so whether
    the caveat gets read at all before answering an era question was
    inconsistent. Duplicating a short, blunt version of the SAME caveat right
    next to the era pass's own text removes that dependency: whichever block
    the model reaches for, the caveat is right there.

    Reads the LUFS/LRA figures back out of the already-built objective report
    text (rather than depending on numeric variables staying in scope across
    a long function) so this is a small, self-contained, read-only addition.
    Returns "" when the profile doesn't meet the loud/tight threshold or the
    figures aren't present in the text.
    """
    if not objective_report:
        return ""
    m_lufs = re.search(r"integrated loudness\s*\u2248\s*(-?\d+(?:\.\d+)?)\s*LUFS", objective_report)
    m_lra = re.search(r"loudness range \(LRA\)\s*\u2248\s*(-?\d+(?:\.\d+)?)\s*LU", objective_report)
    if not m_lufs or not m_lra:
        return ""
    try:
        lufs = float(m_lufs.group(1))
        lra = float(m_lra.group(1))
    except ValueError:
        return ""
    if not (lufs >= -13.0 and lra <= 3.5):
        return ""
    return (
        "\n\nERA CAVEAT (read this before answering any era/release-period question "
        f"about THIS track): the measured mastering is loud ({lufs:.1f} LUFS) and "
        f"tightly compressed ({lra:.1f} LU LRA). This is completely ORDINARY for "
        "anything mastered from roughly the early 2000s onward -- loud, tightly "
        "compressed mastering has been the norm across that entire span (loudness has "
        "not meaningfully changed since then), so this reading by itself does NOT "
        "distinguish a 2002 recording from a 2012 or a brand-new one, and is not "
        "unusual or noteworthy on its own. Do NOT say or imply the track has been "
        "remastered -- you have no evidence that any remastering happened, only a "
        "loudness number, and stating a specific mastering EVENT as fact from that "
        "number alone is not something you're able to support. If asked when this "
        "track is from without using the year tag: this reading only becomes worth "
        "mentioning at all when OTHER evidence -- genre, playing style, harmonic/"
        "melodic vocabulary, period-specific instrumentation or effects -- already "
        "points toward an era BEFORE roughly the mid-to-late 1990s, since a genuinely "
        "untouched master from that far back would rarely measure this loud. Even then, "
        "phrase it as an open question ('the mastering here is louder than you'd expect "
        "for that era, which could mean the piece is a bit more recent than the style "
        "suggests, or that an older recording has had its levels brought up for a later "
        "release') -- never as a settled claim that remastering took place. If nothing "
        "else points to an era that old, say nothing about mastering age at all and lean "
        "entirely on genre/style/performance cues instead."
    )


# dev40: a first-pass rock/punk genre that the genre check replaced. The listening model's
# opening description is the first thing the writer reads; on a pop track it
# called the song a "Pop-Punk anthem" four times, and the writer then told the
# user the act is "often filed as pop-punk".
_GENRE_SUPERSEDED_RE = re.compile(
    r"RECOMMENDED GENRE FOR DISCUSSION: ([^\n]+?)\. Reasoning: MOSS-Music GENRE_RANKED led "
    r"with '([^'\n]+)' \(rock/punk family\)"
)
_GENRE_CHECK_HEAD = "GENRE CHECK (read before the description below):"
_HYPHENS_RE = re.compile(r"[\s\-\u2010\u2011\u2012\u2013\u2014]+")


# dev47: the genre check can also replace a rock/punk top with a pop/dance
# label from the listening model's own ranked list ("MOSS-Music ranked 'pop rock' first and 'dance-pop'
# among its next choices"). That is the same kind of replacement.
_GENRE_SUPERSEDED_ALT_RE = re.compile(
    r"RECOMMENDED GENRE FOR DISCUSSION: ([^\n]+?)\. Reasoning: MOSS-Music ranked '([^'\n]+)' first and '"
)


def _genre_superseded_pair(text):
    """(recommended, replaced first-pass label) for the last genre-check
    replacement of a rock/punk label in the notes, or None."""
    t = str(text or "")
    cands = [(m.start(), m.group(1).strip(), m.group(2).strip())
             for m in _GENRE_SUPERSEDED_RE.finditer(t)]
    if GENRE_CHECK_FIRM:
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


# dev56: the reverse replacement -- an electronic/pop first impression framed
# as rock (GENRE_DEV55 (a), GENRE_DEV56).
_GENRE_ROCKFLIP_RE = re.compile(
    r"RECOMMENDED GENRE FOR DISCUSSION: ([^\n]+?)\. Reasoning: MOSS-Music's ranked list named only "
    r"electronic/pop labels \('([^'\n]+)' first\)"
)


def _genre_rockflip_pair(text):
    """dev56: (recommended, [replaced first-pass labels]) when the genre
    check framed an electronic/pop first impression as rock, else None."""
    if not (GENRE_DEV55 and GENRE_DEV56):
        return None
    t = str(text or "")
    found = list(_GENRE_ROCKFLIP_RE.finditer(t))
    if not found:
        return None
    rec, top = found[-1].group(1).strip(), found[-1].group(2).strip()
    if not rec or _genre_family(rec) != "rockish":
        return None
    olds = [lab for lab in extract_genre_ranked_labels(t, max_n=5) if _genre_family(lab) != "rockish"]
    if not olds or _HYPHENS_RE.sub(" ", olds[0].lower()) != _HYPHENS_RE.sub(" ", top.lower()):
        olds = [top] + [x for x in olds if x != top]
    return rec, olds[:4]


def _label_list(labels, conj="or"):
    labels = [str(x) for x in labels if x]
    if len(labels) <= 1:
        return "".join(labels)
    return ", ".join(labels[:-1]) + f" {conj} " + labels[-1]


def _genre_rockflip_note(text):
    """dev56: GENRE CHECK line for an electronic/pop first impression the
    genre check replaced with a rock label, or ""."""
    pair = _genre_rockflip_pair(text)
    if not pair:
        return ""
    rec, olds = pair
    return (
        f"{_GENRE_CHECK_HEAD} the listening model's first-pass genre ranking at the top of these "
        f"notes led with '{olds[0]}'"
        + (f" (then {_label_list(olds[1:], 'and')})" if len(olds) > 1 else "")
        + ". That was its first impression of a clean, steady, drum-machine-like production; the "
        f"genre check against independent measurements replaced it with '{rec}' (see RECOMMENDED "
        f"GENRE FOR DISCUSSION). Describe the track as {rec} in every answer -- genre, era (including "
        "blind 'when does it sound like it's from?' answers), instrumentation, drums and comparisons "
        "alike -- and take the instruments from the measured stems, not from that first impression. "
        f"Do not call the track {_label_list(olds)}, do not describe it as club or dance-floor music, "
        "and never present those labels as how this track or its artist is usually filed, classed or "
        "known; the electronic quality is a secondary flavour."
    )


# Sentence of the dev33+ genre-check reasoning that the writer turned into
# "pop with a secondary pop-punk/rock flavour" and then "comfortable to call it
# a pop-rock song".
_GENRE_MINOR_FLAVOUR_OLD = "mention rock/punk only as a minor secondary flavour if still audible."
_GENRE_MINOR_FLAVOUR_NEW = (
    "do not call it rock, pop-rock or pop-punk (see the GENRE CHECK line at the top of these "
    "notes); a guitar-like layer, where audible, is one colour inside this production."
)


def _rock_label_list(old, conj="or", with_rock=True):
    """'rock, pop-rock or pop-punk' without repeats ("pop rock" and "pop-rock"
    are one label)."""
    out, seen = [], set()
    for lab in ((["rock"] if with_rock else []) + ["pop-rock", str(old or "").strip()]):
        k = _HYPHENS_RE.sub(" ", lab.lower()).strip()
        if lab and k not in seen:
            seen.add(k)
            out.append(lab)
    if len(out) == 1:
        return out[0]
    return ", ".join(out[:-1]) + f" {conj} " + out[-1]


def _genre_supersession_note(text):
    """A GENRE CHECK line naming the first-pass label the genre check replaced,
    or "" when nothing was replaced."""
    t = str(text or "")
    pair = _genre_superseded_pair(t)
    if not pair:
        return ""
    rec, old = pair
    # The opening description is the text before the first report heading.
    cut = len(t)
    for mk in ("\n\nLYRICS:", "\n\n11. ERA / RELEASE PERIOD", "\n\nVOCAL / SINGER PROFILE",
               "[Independent signal-processing report"):
        k = t.find(mk)
        if 0 <= k < cut:
            cut = k
    toks = [re.escape(w) for w in _HYPHENS_RE.split(old) if w]
    m = None
    if toks:
        pat = re.compile(r"(?<![\w])" + _HYPHENS_RE.pattern.join(toks) + r"(?![\w])", re.IGNORECASE)
        m = pat.search(t[:cut])
    where = (f"the opening description calls it '{m.group(0)}'" if m
             else f"the listening model's first-pass genre ranking led with '{old}'")
    if GENRE_CHECK_FIRM:
        leads, detail = _guitar_stem_reading(t)
        rock_names = _rock_label_list(old)
        if leads:
            guitar = (
                f"Measured from the stems, {detail}, so a guitar-rock edge may be named as a "
                f"secondary flavour -- but lead with {rec}. Asked whether it is {rock_names}: it is "
                f"{rec} first; with the guitar this prominent a pop-rock edge is a fair secondary "
                "description, not the main label."
            )
        else:
            basis = (f"Measured from the stems, {detail}, so" if detail
                     else "Unless the measured stems show a leading guitar part,")
            guitar = (
                f"{basis} any guitar-like midrange layer (guitar or a guitar-like synth) is one "
                "colour inside the production, not a rock band: do not call the track "
                f"{rock_names}, and do not describe its backing as a rock or pop-rock band. "
                f"Asked whether it is {rock_names}: no -- it is {rec}, with at most a light guitar "
                "colour."
            )
        return (
            f"{_GENRE_CHECK_HEAD} {where}. That was the listening model's first impression; the "
            f"genre check against independent measurements replaced it with '{rec}' (see "
            f"RECOMMENDED GENRE FOR DISCUSSION). Describe the track as {rec} in every answer -- "
            "genre, era (including blind 'when does it sound like it's from?' answers), "
            "instrumentation, drums and comparisons alike. The descriptions' other wording built "
            f"on that first impression (a {_rock_label_list(old, 'or', with_rock=True)} band, power "
            f"chords, punk energy, a {_rock_label_list(old, 'or', with_rock=False)} era wave) "
            "belongs to the same discarded guess: take the "
            "instruments, groove and production from the measured blocks instead (stems, "
            f"instrument tags, beat grid, loudness). {guitar} Never present '{old}' as how this "
            "track or its artist is usually filed, classed or known."
        )
    return (
        f"{_GENRE_CHECK_HEAD} {where}. That was the listening model's first impression; the "
        f"genre check against independent measurements replaced it with '{rec}' (see "
        f"RECOMMENDED GENRE FOR DISCUSSION). Describe the track as {rec}. Mention {old} at most "
        f"as a minor flavour where the notes support it, and never present '{old}' as how this "
        f"track or its artist is usually filed, classed or known."
    )


# --- dev44: style-led genre, style era check, regional and World labels -------
# See GENRE_STYLE_LED, ERA_STYLE_CHECK and REGIONAL_GENRE_GATE. All three work on
# the chat model's view of the notes (_writer_analysis_view and the GENRE
# CONSISTENCY note) only: the analysis, the saved .json, the key/tempo decisions
# and every other result are computed exactly as before.


_STYLE_DISPLAY_PARENT = {"Folk, World, & Country": "Folk / Country"}


_REGION_LANGS = {
    "korea": {"ko"}, "japan": {"ja"}, "india": {"hi"}, "italy": {"it"}, "poland": {"pl"},
    # Spanish, Portuguese and French are left out: most of their singers are not European.
    "europe": {"sv", "de", "it", "nl", "da", "no", "fi", "pl", "el", "ca"},
    "germanic": {"de", "nl"}, "france": {"fr"}, "spain": {"es", "ca"}, "greece": {"el"},
    "nordic": {"sv", "no", "da", "fi"}, "germany": {"de"}, "brazil": {"pt"}, "latin": {"es", "pt"},
    "portugal": {"pt"}, "mexico": {"es"}, "colombia": {"es"}, "cuba": {"es"},
}
_EUROPEAN = ("British|English|Scottish|Welsh|Irish|Northern Irish|French|German|Austrian|Swiss|Dutch|Belgian|"
             "Luxembourgish|Swedish|Norwegian|Danish|Finnish|Icelandic|Italian|Spanish|Portuguese|Greek|Cypriot|"
             "Maltese|Polish|Czech|Slovak|Hungarian|Romanian|Bulgarian|Serbian|Croatian|Slovenian|Bosnian|Albanian|"
             "Macedonian|Montenegrin|Estonian|Latvian|Lithuanian|Ukrainian|Belarusian|Russian|Moldovan|Catalan|Basque|"
             "Galician|Flemish|Faroese|Swedish-American")
_LATIN_AM = ("Mexican|Puerto Rican|Cuban|Dominican|Colombian|Venezuelan|Argentine|Argentinian|Chilean|Peruvian|"
             "Ecuadorian|Uruguayan|Paraguayan|Bolivian|Guatemalan|Honduran|Salvadoran|Nicaraguan|Costa Rican|"
             "Panamanian|Brazilian|Spanish|Portuguese|Latin American|Latino|Latina")
_REGION_ORIGIN = {
    "korea": "South Korean|Korean", "japan": "Japanese", "india": "Indian", "europe": _EUROPEAN,
    "italy": "Italian", "poland": "Polish", "germanic": "German|Austrian|Swiss|Dutch|Flemish|Belgian",
    "france": "French|Belgian|Swiss|Québécois", "spain": "Spanish|Catalan", "greece": "Greek|Cypriot",
    "nordic": "Swedish|Norwegian|Danish|Finnish|Icelandic|Faroese", "celtic": "Irish|Scottish|Welsh|Breton|Cornish|Manx",
    "uk": "British|English|Scottish|Welsh|Northern Irish", "germany": "German", "brazil": "Brazilian",
    "latin": _LATIN_AM, "africa": ("Nigerian|Ghanaian|Senegalese|Malian|South African|Kenyan|Congolese|Cameroonian|"
                                   "Ivorian|Ethiopian|Tanzanian|Ugandan|Zimbabwean|Angolan|Mozambican|Beninese|Guinean|"
                                   "Burkinabé|Togolese|Sierra Leonean|Liberian|Gambian|Algerian|Moroccan|Tunisian|"
                                   "Egyptian|Sudanese|Somali|Rwandan|Zambian|Malawian|Namibian|Botswanan|African"),
    "pacific": "Hawaiian|Samoan|Tongan|Fijian|Māori|Maori|Polynesian|Papuan|Tahitian", "cuba": "Cuban",
    "portugal": "Portuguese", "maghreb": "Algerian|Moroccan|Tunisian", "mauritius": "Mauritian|Réunionnais",
    "westafrica": "Ghanaian|Nigerian", "centralafrica": "Congolese", "antilles": "Martinican|Guadeloupean|Antillean",
    "haiti": "Haitian", "mexico": "Mexican|Mexican-American|Chicano|Tejano", "colombia": "Colombian",
    "romani": "Romani|Roma",
}


_REGION_LANGS["china"] = {"zh"}
_REGION_ORIGIN["china"] = "Chinese|Taiwanese|Hong Kong|Hongkongese|Singaporean|Malaysian Chinese"

_LANG_STOPWORDS = {
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
_LANG_STOPSETS = {k: set(v.split()) for k, v in _LANG_STOPWORDS.items()}


def _saved_analysis_with_genre_check(analysis):
    """dev56: see GENRE_CHECK_IN_SAVED_ANALYSIS. The analysis with its GENRE
    CHECK line (and the matching reasoning wording) at the top, exactly as the
    writer is shown it, or unchanged when there is none."""
    t = str(analysis or "")
    if not (GENRE_CHECK_IN_SAVED_ANALYSIS and GENRE_SUPERSEDED_NOTE) or not t or _GENRE_CHECK_HEAD in t:
        return analysis
    try:
        note = _genre_supersession_note(t)
        if note:
            if GENRE_CHECK_FIRM and _GENRE_MINOR_FLAVOUR_OLD in t and not _guitar_stem_reading(t)[0]:
                t = t.replace(_GENRE_MINOR_FLAVOUR_OLD, _GENRE_MINOR_FLAVOUR_NEW)
            return note + "\n\n" + t
        note = _genre_rockflip_note(t)
        if note:
            return note + "\n\n" + t
    except Exception:
        pass
    return analysis


def _any_marker_present(text, markers):
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

# Genre-adjacent AudioSet classes -- a broad, YouTube-metadata-derived taxonomy,
# not a music-critic's genre ontology. Kept intentionally coarse; multiple
# entries can and do co-fire on the same track (e.g. "Electronic music" and
# "House music"), which is expected and left visible rather than collapsed.
_GENRE_LABEL_MAP = {
    "pop music": "pop", "hip hop music": "hip hop", "rock music": "rock",
    "heavy metal": "heavy metal", "punk rock": "punk", "grunge": "grunge",
    "progressive rock": "progressive rock", "rock and roll": "rock and roll",
    "psychedelic rock": "psychedelic rock", "rhythm and blues": "R&B",
    "soul music": "soul", "reggae": "reggae", "country": "country",
    "swing music": "swing", "bluegrass": "bluegrass", "funk": "funk",
    "folk music": "folk", "middle eastern music": "Middle Eastern",
    "jazz": "jazz", "disco": "disco", "classical music": "classical",
    "opera": "opera", "electronic music": "electronic",
    "house music": "house", "techno": "techno", "dubstep": "dubstep",
    "drum and bass": "drum and bass", "electronica": "electronica",
    "electronic dance music": "EDM", "ambient music": "ambient",
    "trance music": "trance", "music of latin america": "Latin",
    "salsa music": "salsa", "flamenco": "flamenco", "blues": "blues",
    "new-age music": "new-age", "vocal music": "vocal-led (unspecified)",
    "a capella": "a cappella", "music of africa": "African",
    "afrobeat": "afrobeat", "christian music": "Christian",
    "gospel music": "gospel", "music of asia": "Asian (unspecified)",
    "carnatic music": "Carnatic", "music of bollywood": "Bollywood",
    "ska": "ska", "traditional music": "traditional/folk (unspecified)",
    "independent music": "indie",
}

# Mood-adjacent AudioSet classes -- for cross-checking MOOD_VIBE the same way
# the genre map cross-checks GENRE_RANKED.
_MOOD_LABEL_MAP = {
    "happy music": "happy/upbeat", "funny music": "playful/quirky",
    "sad music": "sad/melancholic", "tender music": "tender/gentle",
    "exciting music": "exciting/energetic", "angry music": "angry/aggressive",
    "scary music": "tense/scary",
}


def tag_track_genre(mix_audio_path, top_k=None, min_prob=None):
    """Broad AudioSet genre tags for the full mix -- see _GENRE_LABEL_MAP
    and ENABLE_GENRE_MOOD_TAGGING docstring above. Supporting evidence only;
    never treated as a replacement for GENRE_RANKED."""
    if top_k is None:
        top_k = GENRE_TAG_TOP_K
    if min_prob is None:
        min_prob = GENRE_TAG_MIN_PROB
    return _panns_tag_windowed(mix_audio_path, _GENRE_LABEL_MAP, top_k, min_prob)


def tag_track_mood(mix_audio_path, top_k=None, min_prob=None):
    """Broad AudioSet mood tags for the full mix -- see _MOOD_LABEL_MAP.
    Supporting evidence only; never treated as a replacement for MOOD_VIBE."""
    if top_k is None:
        top_k = MOOD_TAG_TOP_K
    if min_prob is None:
        min_prob = MOOD_TAG_MIN_PROB
    return _panns_tag_windowed(mix_audio_path, _MOOD_LABEL_MAP, top_k, min_prob)


def build_genre_mood_signal_report(mix_audio_path):
    """Independent, low-cost genre/mood cross-check built from the same
    already-loaded PANNs tagger used for instrument tagging (see
    ENABLE_GENRE_MOOD_TAGGING docstring).

    Returns (report_text, genre_tags) where genre_tags is the list from
    tag_track_genre() (possibly empty). report_text is "" when nothing useful
    was produced. genre_tags are still returned when present so reconcile_genre()
    can build RECOMMENDED GENRE FOR DISCUSSION even if the prose block is used
    separately.
    """
    if not ENABLE_GENRE_MOOD_TAGGING:
        return "", []
    if mix_audio_path is None or mix_audio_path.startswith(("http://", "https://")):
        return "", []

    genre_tags = tag_track_genre(mix_audio_path)
    mood_tags = tag_track_mood(mix_audio_path)
    if not genre_tags and not mood_tags:
        return "", genre_tags or []

    lines = ["OBJECTIVE GENRE/MOOD SIGNAL (independent AudioSet classifier, PANNs)"]
    if genre_tags:
        lines.append(
            "genre-adjacent tags: "
            + ", ".join(f"{label} ({prob * 100:.0f}%)" for label, prob, _t in genre_tags)
        )
    if mood_tags:
        lines.append(
            "mood-adjacent tags: "
            + ", ".join(f"{label} ({prob * 100:.0f}%)" for label, prob, _t in mood_tags)
        )
    lines.append(
        "These categories are broad, overlapping, and derived from noisy YouTube "
        "metadata -- they are supporting evidence, not automatic overrides. "
        "When this signal clearly favours electronic/dance/synth-pop and GENRE_RANKED "
        "leads with rock/pop-punk mainly from guitar texture, prefer revising "
        "GENRE_RANKED toward the electronic/dance identity (see self-check genre rules). "
        "When both agree, keep GENRE_RANKED. When mixed, put the production-led label "
        "first and the secondary flavour second with lower confidence."
    )
    return "\n".join(lines), genre_tags


# --- dev42: Discogs style tags (MAEST) ----------------------------------------
# See ENABLE_DISCOGS_STYLES. MAEST (Music Audio Efficient Spectrogram
# Transformer, Music Technology Group, Universitat Pompeu Fabra) names the
# Discogs styles a recording sounds like, from 400 (Disco, Europop, Beat,
# Psychedelic Rock, Britpop, ...). It is an ordinary transformers
# audio-spectrogram-transformer model, so it runs on the PyTorch/transformers
# already installed: no TensorFlow and no change to Essentia. Its mel
# spectrogram is computed here exactly as its published feature extractor does
# (transformers.audio_utils, same settings), so no downloaded code is run.
# CPU only, one 30-second window at a time; loaded for each track and freed
# straight after. Never raises out of build_discogs_style_report().
_DISCOGS_STYLE_LOCK = threading.Lock()
_DISCOGS_STYLE_STATE = {"unavailable": None}
_DISCOGS_STYLE_RESULTS = collections.OrderedDict()   # track path -> full [(label, prob)], for the save
_DISCOGS_STYLE_SR = 16000
_DISCOGS_STYLE_N_FFT = 512
_DISCOGS_STYLE_HOP = 256
_DISCOGS_STYLE_MEAN = 2.06755686098554     # MAEST's published normalisation (Discogs20)
_DISCOGS_STYLE_STD = 1.268292820667291
_DISCOGS_STYLE_HEAD = "DISCOGS STYLE SIGNAL"


class _DiscogsStyleModelMissing(RuntimeError):
    """The MAEST model could not be loaded (not downloaded, damaged, or this
    transformers lacks the model class): skipped for the rest of the session."""


def _discogs_style_load():
    """(model, labels, n_mels, max_frames) from the local Hugging Face cache
    only (the app runs offline; musiclyse_style_setup.py downloads it once)."""
    from transformers import ASTForAudioClassification
    model = ASTForAudioClassification.from_pretrained(DISCOGS_STYLE_MODEL_ID, local_files_only=True)
    model = model.float().to("cpu")
    model.eval()
    cfg = model.config
    n = int(getattr(cfg, "num_labels", 0) or 0)
    labels = [str((cfg.id2label or {}).get(i, (cfg.id2label or {}).get(str(i), ""))) for i in range(n)]
    n_mels = int(getattr(cfg, "num_mel_bins", 0) or 0)
    max_frames = int(getattr(cfg, "max_length", 0) or 0)
    if n < 2 or n_mels <= 0 or max_frames <= 0 or sum(1 for l in labels if "---" in l) < n // 2:
        raise RuntimeError("the downloaded model is not a MAEST Discogs style classifier")
    return model, labels, n_mels, max_frames


def _discogs_style_features(y, n_mels, max_frames):
    """MAEST's log-mel input for one window (as MAESTFeatureExtractor)."""
    from transformers.audio_utils import mel_filter_bank, spectrogram, window_function
    window = window_function(window_length=_DISCOGS_STYLE_N_FFT, name="hann")
    mel_fb = mel_filter_bank(
        num_frequency_bins=_DISCOGS_STYLE_N_FFT // 2 + 1, num_mel_filters=n_mels,
        min_frequency=0, max_frequency=_DISCOGS_STYLE_SR / 2, sampling_rate=_DISCOGS_STYLE_SR,
        norm="slaney", mel_scale="slaney")
    S = spectrogram(np.asarray(y, dtype=np.float32), window=window, frame_length=_DISCOGS_STYLE_N_FFT,
                    hop_length=_DISCOGS_STYLE_HOP, power=2, mel_filters=mel_fb, min_value=1e-30,
                    mel_floor=1e-30, pad_mode="constant").T
    S = np.log10(1 + S * 10000)
    if S.shape[0] < max_frames:
        S = np.concatenate([S, np.zeros((max_frames - S.shape[0], S.shape[1]), dtype=S.dtype)], axis=0)
    else:
        S = S[:max_frames]
    return ((S - _DISCOGS_STYLE_MEAN) / (_DISCOGS_STYLE_STD * 2)).astype(np.float32)


def _discogs_style_windows(y, max_frames, n_windows):
    """Evenly spaced windows of the model's length over the whole recording."""
    win = (max_frames - 1) * _DISCOGS_STYLE_HOP          # samples that give max_frames frames
    if len(y) <= win:
        return [y]
    n = max(1, min(int(n_windows), int(math.ceil(len(y) / float(win)))))
    if n == 1:
        s = (len(y) - win) // 2
        return [y[s:s + win]]
    starts = np.linspace(0, len(y) - win, n).astype(int)
    return [y[s:s + win] for s in starts]


def discogs_style_tags(audio_path):
    """[(label, probability)] for every style, highest first. Raises on failure."""
    with _DISCOGS_STYLE_LOCK:
        try:
            model, labels, n_mels, max_frames = _discogs_style_load()
        except Exception as e:
            raise _DiscogsStyleModelMissing(str(e).strip() or type(e).__name__) from e
        try:
            y, _sr = librosa.load(str(audio_path), sr=_DISCOGS_STYLE_SR, mono=True)
            if y is None or len(y) < _DISCOGS_STYLE_SR:          # under a second: nothing to classify
                return []
            acc = np.zeros(len(labels), dtype=np.float64)
            nwin = 0
            with torch.inference_mode():
                for seg in _discogs_style_windows(y, max_frames, DISCOGS_STYLE_WINDOWS):
                    x = torch.from_numpy(_discogs_style_features(seg, n_mels, max_frames))[None]
                    acc += torch.sigmoid(model(input_values=x).logits[0].float()).cpu().numpy().astype(np.float64)
                    nwin += 1
            probs = acc / float(max(1, nwin))
        finally:
            del model
            gc.collect()
    order = np.argsort(-probs)
    return [(labels[i], float(probs[i])) for i in order]


def _discogs_style_display(label):
    """'Funk / Soul---Disco' -> 'Disco (Funk / Soul)'."""
    parent, _, style = str(label).partition("---")
    parent = {"Folk, World, & Country": "Folk / Country"}.get(parent, parent)     # dev44: never "World"
    return f"{style} ({parent})" if style else str(label)


def build_discogs_style_report(audio_path):
    """(report_text, [(label, prob)] all styles) -- ("", []) when disabled,
    unavailable or failed. Never raises."""
    if not ENABLE_DISCOGS_STYLES or not audio_path or str(audio_path).startswith(("http://", "https://")):
        return "", []
    if _DISCOGS_STYLE_STATE["unavailable"]:
        return "", []
    try:
        status("Naming Discogs styles with MAEST...")
        tags = discogs_style_tags(audio_path)
    except Exception as e:
        msg = str(e).strip().splitlines()[0][:200] if str(e).strip() else type(e).__name__
        missing = isinstance(e, _DiscogsStyleModelMissing)
        if missing:
            _DISCOGS_STYLE_STATE["unavailable"] = msg      # stop trying for this session
        hint = (" for this session -- run musiclyse_style_setup.py once to download (or repair) "
                "the MAEST model" if missing else "")
        print(f"  (Discogs style tags skipped{hint}: {msg})")
        return "", []
    if not tags:
        status_done("Discogs styles: recording too short to classify")
        return "", []
    shown = [(l, p) for l, p in tags if p >= float(DISCOGS_STYLE_MIN_PROB)][: max(1, int(DISCOGS_STYLE_TOP_K))]
    if shown:
        heard = "; ".join(f"{_discogs_style_display(l)} {p * 100:.0f}%" for l, p in shown)
    else:
        heard = ("none clear (strongest: "
                 + "; ".join(f"{_discogs_style_display(l)} {p * 100:.0f}%" for l, p in tags[:3]) + ")")
    text = (
        f"{_DISCOGS_STYLE_HEAD} (independent style classifier, MAEST: 400 Discogs record-shop styles, "
        "from the audio alone)\n"
        f"styles heard: {heard}\n"
        "These name the finer styles this recording most resembles, judged from the sound only, with no "
        "artist, title or date information. Use them to describe the style more specifically when one fits "
        "the recommended genre, and as one clue to the era (many styles belong to a period). They do not "
        "replace RECOMMENDED GENRE FOR DISCUSSION; when a style here points to a different family, mention "
        "it only as an alternative reading. Each percentage is the classifier's confidence in that style on "
        "its own, so several can be high at once. Talk about them as styles you hear; never name the "
        "classifier or Discogs."
    )
    status_done("Discogs styles: " + ", ".join(_discogs_style_display(l) for l, _p in (shown or tags[:1])[:3]))
    return text, tags


def _discogs_styles_for_save(track_key):
    """The saved form of the last style tags for this track: [[label, prob]] for
    the strongest DISCOGS_STYLE_SAVE_TOP styles, or [] (none / disabled)."""
    try:
        tags = _DISCOGS_STYLE_RESULTS.get(str(track_key)) if track_key else None
    except Exception:
        tags = None
    if not tags:
        return []
    return [[l, round(float(p), 4)] for l, p in tags[: max(0, int(DISCOGS_STYLE_SAVE_TOP))]]


def _discogs_styles_for_save_list(tags):
    """[[label, prob]] for the strongest DISCOGS_STYLE_SAVE_TOP of `tags`
    ([(label, prob)] or the saved [[label, prob]] form); [] when none."""
    out = []
    for it in (tags or [])[: max(0, int(DISCOGS_STYLE_SAVE_TOP))]:
        try:
            out.append([str(it[0]), round(float(it[1]), 4)])
        except Exception:
            continue
    return out


def _discogs_style_remember(track_path, tags):
    if not tags or not track_path:
        return
    _DISCOGS_STYLE_RESULTS[str(track_path)] = tags
    _DISCOGS_STYLE_RESULTS.move_to_end(str(track_path))
    while len(_DISCOGS_STYLE_RESULTS) > 16:
        _DISCOGS_STYLE_RESULTS.popitem(last=False)
