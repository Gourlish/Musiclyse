# MUSICLYSE FILE pipeline-lyrics 0.8
# ==============================================================================
#  MUSICLYSE 0.8: LYRICS
# ==============================================================================
# This file holds lyrics: Whisper and its transcription passes, the lyrics language, tag lyrics
# and lyric timing, lyric repeats. It is part of musiclyse-0.8.py, which loads it whole at
# start-up, after its own code, into its own namespace (see "THE MUSICLYSE FILES" there); it is
# not run on its own.
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


# --- Lazy loading / swapping for the Whisper lyrics-transcription model ----
# Same rationale as the listening model above: Whisper's weights are large and are
# only actually needed for the one dedicated lyrics pass, so they're loaded
# on first use and explicitly released again right after that pass finishes,
# rather than held resident for the rest of the turn. The listening model and
# Whisper are never intentionally kept loaded at the same time — callers
# unload the listening model before starting the lyrics pass.
_whisper_state = {"pipe": None, "device": None}


def _whisper_planned_device():
    """Which device Whisper will load onto, without loading it.
    Safe to call at startup / for display purposes."""
    return "mps" if torch.backends.mps.is_available() else "cpu"


def _load_whisper_impl():
    """Do the actual (slow, RAM/VRAM-heavy) work of loading the Whisper
    lyrics-transcription pipeline. Callers should go through get_whisper()
    instead, which caches the result so this only runs when the model isn't
    already resident."""
    if DEBUG_MODEL_LOADING:
        print(f"Loading {WHISPER_MODEL_ID} ...")
    else:
        status(f"Loading {WHISPER_MODEL_ID}...")
    device = _whisper_planned_device()
    if WHISPER_RELEASE_BEFORE_LOAD and device == "mps":
        _whisper_full_release()     # dev41: see the setting
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        WHISPER_MODEL_ID,
        torch_dtype=WHISPER_TORCH_DTYPE,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    )
    model.to(device)
    processor = AutoProcessor.from_pretrained(WHISPER_MODEL_ID)
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=WHISPER_TORCH_DTYPE,
        device=device,
        chunk_length_s=30,
        stride_length_s=5,
    )
    _WHISPER_PASS["n"] = 0
    if DEBUG_MODEL_LOADING:
        print(f"Whisper loaded on device: {device}")
    else:
        status_done(f"Whisper loaded on device: {device}" + _whisper_memory_suffix())
    return pipe, device


def get_whisper():
    """Return (pipe, device), loading Whisper into RAM/VRAM on first call
    and reusing the cached instance thereafter. Call unload_whisper() when
    done with the lyrics pass to free that memory again."""
    if _whisper_state["pipe"] is None:
        pipe, device = _load_whisper_impl()
        _whisper_state["pipe"] = pipe
        _whisper_state["device"] = device
    return _whisper_state["pipe"], _whisper_state["device"]


def unload_whisper():
    """Drop Whisper's weights and free the RAM/VRAM they occupied. Safe to
    call even if it was never loaded (no-op). The next call to get_whisper()
    will transparently reload it."""
    if _whisper_state["pipe"] is None:
        return

    status("Freeing Whisper from memory...")
    pipe = _whisper_state["pipe"]
    _whisper_state["pipe"] = None
    _whisper_state["device"] = None

    try:
        del pipe
    except Exception:
        pass

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

    status_done("Whisper unloaded" + _whisper_memory_suffix())


# --- dev40: Whisper pass progress, memory readout and GPU cache release -----
_WHISPER_PASS = {"n": 0, "t0": None, "label": ""}


def _whisper_memory_suffix():
    """' (memory: ...)' for the Whisper status lines, or '' (WHISPER_PROGRESS off,
    or nothing measurable)."""
    if not WHISPER_PROGRESS:
        return ""
    try:
        m = _memory_readout()
    except Exception:
        m = ""
    return f" (memory: {m})" if m else ""


def _whisper_full_release():
    """dev41 WHISPER_RELEASE_BEFORE_LOAD: garbage collection, wait for the
    Apple GPU, release its cache -- as unload_whisper() does. Main thread only;
    never raises."""
    try:
        if not torch.backends.mps.is_available() or not _gpu_cache_clear_ok():
            return
        for _ in range(2):
            gc.collect()
            try:
                torch.mps.synchronize()
            except Exception:
                pass
            torch.mps.empty_cache()
    except Exception:
        pass


def _whisper_mps_trim():
    """Release the GPU cache after a Whisper pass (WHISPER_MPS_CACHE_TRIM_EVERY);
    dev41: the full clean-up (WHISPER_RELEASE_BEFORE_LOAD). Main thread only;
    never raises."""
    if int(WHISPER_MPS_CACHE_TRIM_EVERY or 0) > 0 and _whisper_state.get("device") == "mps":
        if WHISPER_RELEASE_BEFORE_LOAD:
            _whisper_full_release()
        else:
            _moss_mps_trim()


def _whisper_trim_kwargs(generate_kwargs):
    """generate_kwargs plus the GPU cache release step when Whisper runs on an
    Apple GPU and WHISPER_MPS_CACHE_TRIM_EVERY is on; otherwise unchanged."""
    try:
        every = int(WHISPER_MPS_CACHE_TRIM_EVERY or 0)
        if every <= 0 or _whisper_state.get("device") != "mps" or _HFLogitsProcessor is object:
            return generate_kwargs
        from transformers import LogitsProcessorList
        out = dict(generate_kwargs)
        out["logits_processor"] = LogitsProcessorList([_WhisperMpsTrim(every)])
        return out
    except Exception:
        return generate_kwargs


def _whisper_pipe(pipe, path, return_timestamps, generate_kwargs):
    """pipe(path, ...) with the GPU cache release step added
    (WHISPER_MPS_CACHE_TRIM_EVERY). Should the call fail with the step added
    and succeed without it, the step is turned off for the rest of the session
    and the result without it is used -- so it can never cost a pass. Errors
    that are not the step's raise exactly as dev39's call would.
    0.7.1-pre21 WHISPER_HOLD_LANGUAGE: the track's language is passed when one is held; should this
    transformers version refuse it, the call runs without it, as before, for the rest of the session."""
    if WHISPER_HOLD_LANGUAGE and "language" not in (generate_kwargs or {}) and not _WHISPER_LANG_STATE.get("off"):
        _lang = _whisper_track_language(pipe, path)
        if _lang:
            try:
                return _whisper_pipe(pipe, path, return_timestamps, dict(generate_kwargs, language=_lang))
            except (TypeError, ValueError) as _e:
                if "language" not in str(_e).lower():
                    raise
                _WHISPER_LANG_STATE["off"] = True
                print(f"  (Whisper refused the lyrics language on this transformers version; each stretch is "
                      f"guessed as before: {type(_e).__name__}: {_e})")
    gk = generate_kwargs
    if not _WHISPER_PASS.get("trim_off"):
        gk2 = _whisper_trim_kwargs(generate_kwargs)
        if gk2 is not generate_kwargs:
            try:
                return pipe(path, return_timestamps=return_timestamps, generate_kwargs=gk2)
            except Exception:
                out = pipe(path, return_timestamps=return_timestamps, generate_kwargs=gk)
                _WHISPER_PASS["trim_off"] = True
                return out
    return pipe(path, return_timestamps=return_timestamps, generate_kwargs=gk)


# 0.7.1-pre21 WHISPER_HOLD_LANGUAGE: this track's lyrics language, decided once per audio file
# ("decided": file -> code or None), the record of the decision for the save, the tag lyrics when
# Whisper only times them, and "off" once a transformers version has refused a language.
_WHISPER_LANG_STATE = {"decided": {}, "record": None, "tag_lyrics": None, "off": False}
_WHISPER_ENGLISH_WORDS = frozenset("""the and you i to a me my it in of that is your on be for all we what with this
can just so but not don't i'm it's when like know love no do are was have get one go up out if oh baby now never
yeah got let come say see there they will make way how time from at want need ever take give feel heart by he she
her his our us them were been had has would could""".split())


def _text_looks_english(text):
    """0.7.1-pre21: True for text of 30 words or more of which at least a quarter are the most common
    English words (English lyrics run at about a third to two thirds; French, German, Spanish,
    Italian or Welsh at an eighth or less)."""
    w = re.findall(r"[^\W\d_]+(?:'[^\W\d_]+)?", str(text or "").lower().replace("\u2019", "'"))
    return len(w) >= 30 and sum(x in _WHISPER_ENGLISH_WORDS for x in w) >= 0.25 * len(w)


def _whisper_language_name(code):
    try:
        from transformers.models.whisper.tokenization_whisper import LANGUAGES as _L
        return str(_L.get(code) or code).title()
    except Exception:
        return str(code)


def _whisper_stretch_language_scores(pipe, path):
    """0.7.1-pre21: [(language probabilities {code: p}, no-speech probability)] for each 30-second
    stretch the transcription pipeline decodes (a stretch every 20 s), from the same first step
    Whisper uses to guess a stretch's language. [] when the model has no language list."""
    from transformers.pipelines.audio_utils import ffmpeg_read
    model, fe = pipe.model, pipe.feature_extractor
    gc = model.generation_config
    lang_to_id = dict(getattr(gc, "lang_to_id", None) or {})
    if not lang_to_id:
        return []
    sr = int(fe.sampling_rate)
    with open(path, "rb") as fh:
        audio = ffmpeg_read(fh.read(), sr)
    chunk, stride = 30 * sr, 5 * sr
    wins = []
    for start in range(0, len(audio), chunk - 2 * stride):
        piece = audio[start:start + chunk]
        if len(piece) > (0 if start == 0 else stride):
            wins.append(piece)
        if start + chunk >= len(audio):
            break
    codes = [str(k)[2:-2] for k in lang_to_id]
    ids = [int(v) for v in lang_to_id.values()]
    ns_id = getattr(gc, "no_timestamps_token_id", None)
    ns_id = int(ns_id) - 1 if ns_id is not None else None
    out = []
    for i in range(0, len(wins), 4):
        feats = fe(wins[i:i + 4], sampling_rate=sr, return_tensors="pt").input_features
        feats = feats.to(model.device, dtype=model.dtype)
        dec = torch.full((feats.shape[0], 1), int(gc.decoder_start_token_id), dtype=torch.long, device=model.device)
        with torch.no_grad():
            logits = model(input_features=feats, decoder_input_ids=dec).logits[:, -1].float()
        lang = torch.softmax(logits[:, ids], dim=-1).cpu()
        full = torch.softmax(logits, dim=-1)[:, ns_id].cpu() if ns_id is not None else None
        for r in range(lang.shape[0]):
            out.append(({codes[j]: float(lang[r, j]) for j in range(len(codes))},
                        float(full[r]) if full is not None else 0.0))
        del feats, logits, lang, full, dec
    try:
        _whisper_mps_trim()                       # release the GPU cache, as after each pass
    except Exception:
        pass
    return out


def _whisper_language_decision(stretches):
    """0.7.1-pre21: (language code, or None to guess each stretch as before; record) from the
    stretch scores. Stretches Whisper itself would treat as silence (no-speech 0.6 or more) don't
    count. English unless another language has at least twice English's added-up score and is the
    confident pick (0.8 or more) in at least three of every four stretches with singing; None when
    English and another language are each the confident pick in a quarter of them (at least two),
    or when no language is clearly ahead."""
    guesses = collections.Counter(max(p, key=p.get) for p, _ns in stretches if p)
    sung = [(p, 1.0 - ns) for p, ns in stretches if p and ns < 0.6]
    rec = {"stretches": len(stretches), "sung": len(sung),
           "guesses": {k: v for k, v in guesses.most_common(4)}}
    if not sung:
        return None, dict(rec, kind="no singing", language="")
    tot = collections.Counter()
    for p, w in sung:
        for k, v in p.items():
            tot[k] += w * v
    s = sum(tot.values()) or 1.0
    rec["scores"] = {k: round(v / s, 3) for k, v in tot.most_common(3)}
    conf = collections.Counter(max(p, key=p.get) for p, _w in sung if max(p.values()) >= 0.8)
    n = len(sung)
    need = max(2, math.ceil(0.25 * n))
    top = max(tot, key=tot.get)
    if conf["en"] >= need and any(k != "en" and c >= need for k, c in conf.items()):
        return None, dict(rec, kind="mixed", language="")
    if top == "en" or tot[top] < 2 * tot["en"]:
        return "en", dict(rec, kind="held", language="en")
    if conf[top] >= 0.75 * n:
        return top, dict(rec, kind="held", language=top)
    return None, dict(rec, kind="unclear", language="")


def _whisper_track_language(pipe, path):
    """0.7.1-pre21 WHISPER_HOLD_LANGUAGE: the language code Whisper is given for this audio file, or
    None (each 30-second stretch guessed, as before). Decided once per file and shown; anything that
    goes wrong gives None."""
    if not WHISPER_HOLD_LANGUAGE or _WHISPER_LANG_STATE.get("off"):
        return None
    try:
        st = os.stat(path)
        key = (os.path.abspath(str(path)), st.st_size, st.st_mtime_ns)
    except Exception:
        key = (str(path), None, None)
    decided = _WHISPER_LANG_STATE["decided"]
    if key in decided:
        return decided[key]
    code = None
    try:
        if str(WHISPER_LANGUAGE or "").strip():
            code = str(WHISPER_LANGUAGE).strip().lower()
            rec = {"kind": "setting", "language": code}
        elif _text_looks_english(_WHISPER_LANG_STATE.get("tag_lyrics")):
            code, rec = "en", {"kind": "tag lyrics", "language": "en"}
        else:
            status("Checking the lyrics language (Whisper)...")
            code, rec = _whisper_language_decision(_whisper_stretch_language_scores(pipe, path))
    except Exception as e:
        code, rec = None, {"kind": "not checked", "language": "", "error": f"{type(e).__name__}: {e}"[:300]}
    rec = dict(rec, file=os.path.basename(str(path)))
    decided[key] = code
    if _WHISPER_LANG_STATE.get("record") is None:
        _WHISPER_LANG_STATE["record"] = rec
    g = rec.get("guesses") or {}
    seen = (" (each stretch on its own would have been guessed as "
            + ", ".join(f"{_whisper_language_name(k)} {v}" for k, v in g.items()) + ")") if g else ""
    if code:
        why = {"setting": "set in the settings", "tag lyrics": "the tag lyrics are English"}.get(rec.get("kind"), "")
        msg = f"Lyrics language: {_whisper_language_name(code)}, kept for the whole track" + (f" ({why})" if why else seen)
    else:
        why = {"mixed": "two languages are each clearly sung", "unclear": "no language is clearly ahead",
               "no singing": "no stretch clearly has singing"}.get(rec.get("kind"), "not checked: " + str(rec.get("error") or ""))
        msg = f"Lyrics language: guessed for each stretch as before ({why})" + seen
    try:
        status_done(msg)
    except Exception:
        pass
    return code


def _whisper_pass_begin(label):
    """Start of one Whisper decoding pass (WHISPER_PROGRESS)."""
    try:
        _WHISPER_PASS["n"] += 1
        _WHISPER_PASS["t0"] = time.perf_counter()
        _WHISPER_PASS["label"] = str(label)
        if WHISPER_PROGRESS:
            status(f"Whisper pass {_WHISPER_PASS['n']}: {label}...")
    except Exception:
        pass


def _whisper_pass_end():
    """End of one Whisper decoding pass: release the GPU cache
    (WHISPER_MPS_CACHE_TRIM_EVERY) and report it (WHISPER_PROGRESS)."""
    _whisper_mps_trim()
    try:
        if WHISPER_PROGRESS and _WHISPER_PASS.get("t0") is not None:
            dt = time.perf_counter() - _WHISPER_PASS["t0"]
            status_done(f"Whisper pass {_WHISPER_PASS['n']} ({_WHISPER_PASS['label']}) done in {dt:.0f} s"
                        + _whisper_memory_suffix())
        _WHISPER_PASS["t0"] = None
    except Exception:
        pass


# Stock phrases Whisper is well known to emit over music and silence, learned
# from subtitled video rather than heard in the audio ("Thanks for watching!",
# "Subtitles by the Amara.org community", "[Music]", "♪"). Left in, they were
# presented as sung lyrics, and because they repeat across instrumental gaps
# they also showed up in the LYRIC REPETITION MAP as a "recurring line",
# which then fed section naming. Deliberately narrow: only phrases that are
# near-certain artefacts are removed, so genuine lyrics are never touched.
_WHISPER_HALLUCINATION_PATTERNS = [
    re.compile(r"\bthank(?:s| you)(?: (?:so|very) much)?,? for watching(?=\s*(?:[.!?,]|$))[.!]*", re.IGNORECASE),
    re.compile(r"\bplease (?:like,? (?:and|&) )?subscribe\b[^.!?\n]*[.!]*", re.IGNORECASE),
    re.compile(r"\blike (?:and|&) subscribe\b[.!]*", re.IGNORECASE),
    re.compile(r"\bsubscribe to (?:my|the|our) channel\b[.!]*", re.IGNORECASE),
    re.compile(r"\bsubtitles? (?:by|created by|provided by|made by)\b(?:[^.!?\n]|\.(?=\w))*[.!?]?", re.IGNORECASE),
    re.compile(r"\b(?:the )?amara\.org\b(?:[^.!?\n]|\.(?=\w))*[.!?]?", re.IGNORECASE),
    re.compile(r"\b(?:transcribed|transcription|captions?|captioned) by\b(?:[^.!?\n]|\.(?=\w))*[.!?]?", re.IGNORECASE),
    re.compile(r"\[\s*(?:music|applause|laughter|instrumental|silence|no speech|blank_audio)\s*\]", re.IGNORECASE),
    re.compile(r"\(\s*[a-z ]{0,20}\bmusic\b[a-z ]{0,20}\)", re.IGNORECASE),
    re.compile(r"[\u266a\u266b\u266c]+"),
    # Other stock video-outro lines Whisper produces on music without singing.
    re.compile(r"\bwe'?ll be right back\b[.!]*", re.IGNORECASE),
    re.compile(r"\bsee you (?:(?:in|on) (?:the )?)?next (?:time|video|week|one)\b[.!]*", re.IGNORECASE),
    re.compile(r"\bdon'?t forget to (?:like|subscribe)\b[^.!?\n]*[.!]*", re.IGNORECASE),
    # "Thanks for watching" in the languages it most often hallucinates in.
    re.compile(
        r"(?:terima kasih (?:telah|sudah|karena) menonton|gracias por (?:ver|mirar)(?: el video)?"
        r"|obrigad[oa] por (?:assistir|ver)|danke (?:f[\u00fcu]rs|f[\u00fcu]r'?s) zuschauen"
        r"|merci d'avoir regard[\u00e9e]|grazie per la visione|bedankt voor het kijken"
        r"|\u0441\u043f\u0430\u0441\u0438\u0431\u043e \u0437\u0430 \u043f\u0440\u043e\u0441\u043c\u043e\u0442\u0440|c\u1ea3m \u01a1n (?:c\u00e1c b\u1ea1n )?\u0111\u00e3 (?:xem|theo d\u00f5i)"
        r"|\u3054\u8996\u8074\u3042\u308a\u304c\u3068\u3046\u3054\u3056\u3044\u307e\u3057\u305f|\uc2dc\uccad\ud574\s*\uc8fc\uc154\uc11c\s*\uac10\uc0ac\ud569\ub2c8\ub2e4|\u611f\u8c22(?:\u60a8\u7684)?(?:\u6536\u770b|\u89c2\u770b)|\u8c22\u8c22(?:\u6536\u770b|\u89c2\u770b))[.!\u3002\uff01]*",
        re.IGNORECASE,
    ),
    # Degenerate repetition: the same short unit (1-8 characters, no spaces)
    # repeated back-to-back 12+ times (observed: a long run of one Khmer
    # syllable on an instrumental). Spaced vocables ("na na na") never match.
    re.compile(r"(\S{1,8}?)\1{11,}"),
]


def _strip_whisper_hallucinations(text):
    if not text:
        return text or ""
    out = str(text)
    for pat in _WHISPER_HALLUCINATION_PATTERNS:
        out = pat.sub(" ", out)
    out = re.sub(r"[ \t]{2,}", " ", out)
    # Nothing left but punctuation -> nothing was sung.
    if not re.search(r"[^\W_]", out):
        return ""
    return out.strip()


def _whisper_transcribe_once(path, do_sample=False, temperature=None, top_p=None, return_chunks=False,
                             no_repeat_ngram_size=4):
    """Run a single Whisper decode over the whole track and return plain
    transcribed text (chunked internally for tracks longer than 30s).
    If return_chunks=True, also return the per-segment timestamped chunks
    Whisper already computes (result['chunks']) -- previously discarded
    entirely, which meant no lyric line ever had a timestamp anywhere in
    the pipeline, so nothing could detect where lyrics repeat (chorus/
    refrain) independent of a rigid verse-chorus template."""
    pipe, _device = get_whisper()
    if WHISPER_HOLD_LANGUAGE:
        _whisper_track_language(pipe, path)       # 0.7.1-pre21: decided (and shown) before the pass starts

    if do_sample and _TEST_SEED is not None:
        try:
            torch.manual_seed(_TEST_SEED)   # dev54 test-only; off by default
        except Exception:
            pass

    generate_kwargs = {"task": "transcribe"}
    if do_sample:
        generate_kwargs["do_sample"] = True
        if temperature is not None:
            generate_kwargs["temperature"] = float(temperature)
        if top_p is not None:
            generate_kwargs["top_p"] = float(top_p)
    else:
        generate_kwargs["do_sample"] = False

    # Anti-hallucination long-form settings: condition_on_prev_tokens=False
    # stops one bad/garbled chunk from poisoning every chunk that follows it
    # (Whisper's single biggest hallucination-cascade failure mode), and the
    # threshold trio makes it fall back to silence rather than invent text
    # over quiet/ambiguous passages. Older transformers builds may not accept
    # these kwargs, so fall back to a minimal call if they're rejected.
    rich_kwargs = dict(
        generate_kwargs,
        condition_on_prev_tokens=False,
        compression_ratio_threshold=1.35,
        logprob_threshold=-1.0,
        no_speech_threshold=0.6,
    )
    # dev53: the lyric-timing retry passes 0 (no rule), every other caller
    # keeps the original 4.
    if no_repeat_ngram_size:
        rich_kwargs["no_repeat_ngram_size"] = int(no_repeat_ngram_size)
    if WHISPER_EXPLICIT_TEMPERATURE and "temperature" not in rich_kwargs:
        # dev60: see WHISPER_EXPLICIT_TEMPERATURE.
        rich_kwargs["temperature"] = 0.0
    # dev40: see WHISPER_PROGRESS / WHISPER_MPS_CACHE_TRIM_EVERY.
    _whisper_pass_begin("cross-check (sampled)" if do_sample else
                        ("re-timing" if not no_repeat_ngram_size else "transcription"))
    _pass_ok = False
    try:
        _no_temp = "temperature" not in rich_kwargs
        if _no_temp and _WORD_TIMING_NOTE.get("rich_rejected_no_temp"):
            # dev60: this transformers version already rejected these settings
            # without a temperature; go straight to the plain settings it ran
            # with (identical result, without decoding a first stretch to throw
            # away on every pass).
            result = _whisper_pipe(pipe, path, True, generate_kwargs)
        else:
            try:
                result = _whisper_pipe(pipe, path, True, rich_kwargs)
            except (TypeError, ValueError) as _rich_err:
                # dev55: say so once -- on some transformers releases the
                # threshold settings are rejected (e.g. "'>' not supported between
                # 'NoneType' and 'float'" without a temperature), and every pass then
                # silently ran with plain settings. Behaviour is unchanged.
                if _no_temp:
                    _WORD_TIMING_NOTE["rich_rejected_no_temp"] = True
                if not _WORD_TIMING_NOTE.get("rich_warned"):
                    _WORD_TIMING_NOTE["rich_warned"] = True
                    if _no_temp and "NoneType" in str(_rich_err):
                        print("  (Whisper: this transformers version only accepts the anti-hallucination "
                              "settings together with an explicit temperature, so lyrics are transcribed "
                              "with the standard settings -- as in all earlier scans. Set "
                              "WHISPER_EXPLICIT_TEMPERATURE = True to use them.)")
                    else:
                        print(f"  (Whisper rejected the anti-hallucination settings on this transformers "
                              f"version and ran with plain settings: {type(_rich_err).__name__}: {_rich_err})")
                result = _whisper_pipe(pipe, path, True, generate_kwargs)
        _pass_ok = True
    finally:
        if _pass_ok:
            _whisper_pass_end()
        else:
            _whisper_mps_trim()
            _WHISPER_PASS["t0"] = None

    text = _strip_whisper_hallucinations((result or {}).get("text", "") or "")
    if return_chunks:
        raw_chunks = (result or {}).get("chunks") or []
        chunks = []
        for c in raw_chunks:
            try:
                ts = c.get("timestamp") or (None, None)
                start, end = ts[0], ts[1]
                ctext = _strip_whisper_hallucinations(c.get("text") or "").strip()
                if not ctext or start is None:
                    continue
                chunks.append({
                    "start": float(start),
                    "end": float(end) if end is not None else float(start),
                    "text": ctext,
                })
            except Exception:
                continue
        return text.strip(), chunks
    return text.strip()


def whisper_transcribe_lyrics_full(path, cross_check=True):
    """Transcribe the lyrics (one Whisper decode, or two if cross_check) and
    also return the per-line timestamped chunks from the primary
    (deterministic) decode pass, so lyric repetition can be measured against
    real timing rather than discarded. Returns
    (full_lyrics_text, chunks) where chunks is a list of {"start", "end",
    "text"} dicts in track time (seconds)."""
    full_lyrics, chunks = _whisper_transcribe_once(path, return_chunks=True)
    full_lyrics = _sanitize_lyrics_transcription(full_lyrics)

    if not cross_check or not full_lyrics or not full_lyrics.strip():
        return full_lyrics, chunks

    full_lyrics_alt = _whisper_transcribe_once(
        path, do_sample=True, temperature=0.7, top_p=0.9
    )
    full_lyrics_alt = _sanitize_lyrics_transcription(full_lyrics_alt)

    final_text = _cross_check_lyrics_transcriptions(full_lyrics, full_lyrics_alt)
    # An unverified span left empty by the hallucination filter, or a result
    # with no words at all, is not a lyric.
    final_text = re.sub(r"\[UNVERIFIED→\s*\]", "", final_text or "").strip()
    if not re.search(r"[^\W_]", final_text or ""):
        final_text = ""
    if final_text and WHISPER_WORD_TIMING and _chunks_look_smeared(chunks):
        # dev55: collapsed line timing -> time the words (timing only; the
        # lyric text above is unchanged).
        status("Line timing of the lyrics is collapsed; timing the words (Whisper)...")
        wchunks = _whisper_word_timed_chunks(path)
        if wchunks and not _chunks_look_smeared(wchunks):
            status_done(f"Lyric timing from word timestamps ({len(wchunks)} lines)")
            chunks = wchunks
        else:
            status_done("Word timing did not improve the line timing; keeping it")
    return final_text, chunks


_WORD_TIMING_NOTE = {"warned": False}


def _lexical_share(text):
    """dev36: share of a line's words that are not vocables ("oh", "yeah")."""
    ws = [w.strip("'-") for w in re.sub(r"[^\w'\- ]+", " ", str(text or "").lower()).split()]
    ws = [w for w in ws if w]
    if not ws:
        return 0.0
    return sum(1 for w in ws if w not in _SX_VOCABLES) / float(len(ws))


def _chunks_look_smeared(chunks, smear_s=None):
    """dev55: True when a quarter or more of the transcribed words sit in
    chunks longer than smear_s seconds (Whisper's segment timing collapsed),
    or -- dev36 -- in chunks timed faster than WHISPER_WORD_TIMING_DENSE_WPS,
    or there are words but no usable chunk timing at all."""
    smear_s = float(WHISPER_WORD_TIMING_SMEAR_S if smear_s is None else smear_s)
    tot = smeared = 0
    for c in chunks or []:
        try:
            dur = float(c.get("end")) - float(c.get("start"))
        except Exception:
            continue
        n = len(str(c.get("text") or "").split())
        tot += n
        if dur > smear_s:
            smeared += n
        elif (WHISPER_WORD_TIMING_DENSE_WPS and n >= 40
              and n / max(dur, 0.1) > float(WHISPER_WORD_TIMING_DENSE_WPS)
              and _lexical_share(c.get("text")) >= 0.5):
            # dev36: several lines' worth of real words crammed into an instant
            # (a run of "oh, oh, oh" stamped with no length is a different slip)
            smeared += n
    if tot == 0:
        return True
    return smeared >= 0.25 * tot


def _lines_from_timed_words(words, gap_s=None, max_line_s=None, max_words=14):
    """dev55: [(start, end, word)] -> line chunks [{"start","end","text"}].
    A new line starts at a pause of gap_s or more, when the line would run
    past max_line_s or max_words, after sentence punctuation (with any pause),
    or at a capitalised word after a short pause (Whisper capitalises line
    starts). Held notes keep their length but never more than 4 s per word."""
    gap_s = float(WHISPER_WORD_LINE_GAP_S if gap_s is None else gap_s)
    max_line_s = float(WHISPER_WORD_LINE_MAX_S if max_line_s is None else max_line_s)
    ws = []
    for s_, e_, w in sorted(words or [], key=lambda x: x[0]):
        w = str(w or "").strip()
        if not w:
            continue
        s_ = float(s_)
        e_ = float(e_) if e_ is not None else s_
        e_ = min(max(e_, s_), s_ + 4.0)
        ws.append((s_, e_, w))
    lines, cur = [], []
    for s_, e_, w in ws:
        if cur:
            gap = s_ - cur[-1][1]
            brk = (gap >= gap_s or (e_ - cur[0][0]) > max_line_s or len(cur) >= max_words)
            if not brk and len(cur) >= 3 and gap >= 0.15 and re.search(r"[.!?]$", cur[-1][2]):
                brk = True
            if (not brk and len(cur) >= 3 and gap >= 0.25 and w[:1].isupper()
                    and re.sub(r"[^\w']", "", w) not in ("I", "I'm", "I'll", "I've", "I'd")):
                brk = True
            if brk:
                lines.append(cur)
                cur = []
        cur.append((s_, e_, w))
    if cur:
        lines.append(cur)
    out = []
    for ln in lines:
        text = _strip_whisper_hallucinations(" ".join(w for _s, _e, w in ln)).strip()
        if not text or not re.search(r"[^\W_]", text):
            continue
        out.append({"start": float(ln[0][0]), "end": float(ln[-1][1]), "text": text})
    return out


def _whisper_word_timed_chunks(path):
    """dev55: one Whisper pass with word timestamps, regrouped into timed
    lines. Plain greedy decoding (the settings every transformers version
    accepts). Returns [] on any failure, so callers keep their dev54 timing."""
    if not WHISPER_WORD_TIMING:
        return []
    try:
        pipe, _device = get_whisper()
        if WHISPER_HOLD_LANGUAGE:
            _whisper_track_language(pipe, path)   # 0.7.1-pre21
        _whisper_pass_begin("word timing")     # dev40
        try:
            result = _whisper_pipe(pipe, path, "word", {"task": "transcribe", "do_sample": False})
        except Exception:
            _whisper_mps_trim()
            _WHISPER_PASS["t0"] = None
            raise
        _whisper_pass_end()
    except Exception as e:
        if not _WORD_TIMING_NOTE["warned"]:
            _WORD_TIMING_NOTE["warned"] = True
            print(f"  (word-level lyric timing unavailable, using line timing: {type(e).__name__}: {e})")
        return []
    words = []
    for c in (result or {}).get("chunks") or []:
        try:
            ts = c.get("timestamp") or (None, None)
            if ts[0] is None:
                continue
            words.append((float(ts[0]), None if ts[1] is None else float(ts[1]), c.get("text") or ""))
        except Exception:
            continue
    if WHISPER_WORD_STRIDE_DEDUPE:
        words = _dedupe_window_overlap_words(words)
    return _lines_from_timed_words(words)


def _dedupe_window_overlap_words(words, back_s=0.05):
    """dev56: see WHISPER_WORD_STRIDE_DEDUPE. `words` are (start, end, text)
    in the order the pipeline returned them. Within one decoding window word
    starts never go backwards, so a start earlier than the previous word's
    marks the next window re-reading the overlap: the earlier window keeps
    its words before the middle of the overlap, the later window the rest.
    Words in order (no overlap re-read) are returned unchanged."""
    def _end(x):
        s0 = float(x[0])
        e0 = float(x[1]) if x[1] is not None else s0
        return min(max(e0, s0), s0 + 2.0)

    out = []
    reach = None     # furthest time the words so far cover
    cut = None       # words of the new window before this are dropped
    for w in words or []:
        try:
            s_ = float(w[0])
            _e = _end(w)
        except Exception:
            continue
        if cut is not None:
            if s_ < cut:
                continue
            cut = None
        if out and s_ < float(out[-1][0]) - back_s:
            mid = 0.5 * (s_ + max(reach if reach is not None else s_, s_))
            while out and float(out[-1][0]) >= mid:
                out.pop()
            # The new window resumes after the last word the earlier one keeps.
            cut = max(mid, (_end(out[-1]) - 0.1) if out else mid)
            reach = max([_end(x) for x in out], default=None)
            if s_ < cut:
                continue
            cut = None
        out.append(w)
        reach = _e if reach is None else max(reach, _e)
    return out


def _normalize_lyric_line(text):
    text = (text or "").lower().strip()
    text = re.sub(r"[^a-z0-9' ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _lyric_norm_words(text):
    """Lower-case words of a lyric line, keeping accented letters."""
    return re.sub(r"[^\w' ]+", " ", str(text or "").lower()).split()


def _tag_lyric_lines(tag_lyrics):
    """The file's lyrics as [(words, stanza_index)], one entry per line of
    two words or more. Stanzas are separated by blank lines or by section
    markers such as "[Chorus]" / "(Chorus)" / "(x2)". A line that is all in
    brackets is not lyric text (as before), but only a section marker starts
    a new stanza -- a backing-vocal line such as "(along the line)" sits
    inside its stanza. Lyrics with a blank line after nearly every line have
    no real stanzas, so they are returned as one stanza."""
    out, stanza, started = [], 0, False
    for raw in str(tag_lyrics or "").splitlines():
        ln = raw.strip()
        if not ln or re.fullmatch(r"[\[(].*[\])]", ln):
            if started and (not ln or _TAG_SECTION_MARKER_RE.fullmatch(ln)):
                stanza += 1
                started = False
            continue
        words = _lyric_norm_words(ln)
        if len(words) >= 2:
            out.append((words, stanza))
            started = True
    if out:
        sizes = {}
        for _w, st in out:
            sizes[st] = sizes.get(st, 0) + 1
        if float(np.median(list(sizes.values()))) < 2:
            out = [(w, 0) for w, _st in out]
    return out


def _tag_line_cuts(cw, ww, starts, ids):
    """Where each tag line of window ww (line starts `starts`, line ids `ids`)
    begins inside the heard words cw: [(word index, line id)]. A line begins
    where one of its first three words lines up exactly with a heard word,
    counted back to the line's first word (a misheard first word does not
    lose the line) but never before the previous line's last matched word.
    A later line with none of its first three words heard is left inside the
    line before it; the first line begins at the chunk's first word unless
    a match places it later."""
    import difflib
    sm = difflib.SequenceMatcher(None, cw, ww, autojunk=False)
    eq = {}
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "equal":
            for q in range(j2 - j1):
                eq[j1 + q] = i1 + q
    ends = list(starts[1:]) + [len(ww)]
    segs = []
    for n, (js, je, lid) in enumerate(zip(starts, ends, ids)):
        i0 = None
        for q in range(js, min(js + 3, je)):
            if q in eq:
                i0 = eq[q] - (q - js)
                prior = [eq[q2] for q2 in eq if q2 < js]
                if prior:
                    i0 = max(i0, max(prior) + 1)
                break
        if i0 is None:
            if n > 0:
                continue
            i0 = 0
        i0 = max(int(i0), 0)
        if (segs and i0 <= segs[-1][0]) or i0 >= len(cw):
            continue
        segs.append((i0, lid))
    return segs


def _align_chunks_to_tag_lyrics(chunks, tag_lyrics):
    """Correct Whisper's timed chunks toward the file's own lyrics.

    Whisper supplies WHEN words are sung; the tag lyrics say WHAT they are.
    Each chunk is compared word by word with the run of 1-4 consecutive tag
    lines it most resembles, and where the two line up one word for one word
    the tag's word replaces the misheard one -- so every chorus reads the
    same and its repeats are found. Nothing is added, moved or dropped
    (that would shift where words fall in time), and a chunk that resembles
    no tag line keeps its own text. A matched chunk also records where each
    tag line starts inside it ("lines") and which stanza(s) that line belongs
    to ("line_stanzas": (stanza, first-line-of-stanza, tag line index)
    candidates -- the same words can sit in two stanzas, a hook ending the
    bridge and the chorus, or twice in one),
    so the section map can time the lines and see where stanzas begin. If the
    tags hardly match the audio at all (another version, the wrong song) the
    chunks are returned untouched, exactly as for a track without tags.
    """
    import difflib
    chunks = list(chunks or [])
    tag = _tag_lyric_lines(tag_lyrics)
    lines = [w for w, _st in tag]
    if not chunks or len(lines) < 2:
        return chunks
    first_of = {}
    for idx, (_w, st) in enumerate(tag):
        first_of.setdefault(st, idx)
    by_text = {}
    for idx, (w, st) in enumerate(tag):
        by_text.setdefault(" ".join(w), []).append((st, first_of[st] == idx, idx))
    # Line timing and stanzas are only for STRUCTURE_TAG_STANZAS; with it off
    # this is exactly the dev38 word correction.
    want_lines = bool(STRUCTURE_TAG_STANZAS and STRUCTURE_LYRIC_PHRASES)
    # Runs that join the end of one stanza to the start of another (or of the
    # same one): a hook closing the bridge sung straight into the chorus, or
    # a chorus going round again when the tags write it out once. Whisper
    # often merges such lines into one chunk, and without these runs the
    # chunk's first words would be taken for the chorus's first line. Used
    # only to find where lines begin -- never to correct words.
    joined = []
    if want_lines and len(first_of) <= 30:
        last_of = {}
        for idx, (_w, st) in enumerate(tag):
            last_of[st] = idx
        seen_j = set()
        opens_idx = sorted(set(first_of.values()))
        for st_e, e in last_of.items():
            for pre in (1, 2):
                pre_ids = list(range(e - pre + 1, e + 1))
                if pre_ids[0] < 0 or any(tag[q][1] != st_e for q in pre_ids):
                    continue
                for f in opens_idx:
                    if f == e + 1:
                        continue
                    for post in range(1, 5 - pre):
                        post_ids = list(range(f, f + post))
                        if post_ids[-1] >= len(lines) or tag[post_ids[-1]][1] != tag[f][1]:
                            break
                        ids = pre_ids + post_ids
                        acc, starts = [], []
                        for q in ids:
                            starts.append(len(acc))
                            acc = acc + lines[q]
                        key = tuple(acc)
                        if key in seen_j:
                            continue
                        seen_j.add(key)
                        joined.append((acc, starts, ids))
    windows, window_starts, window_ids = [], [], []
    for i in range(len(lines)):
        acc, starts, ids = [], [], []
        for w in range(4):
            if i + w >= len(lines):
                break
            starts = starts + [len(acc)]
            ids = ids + [i + w]
            acc = acc + lines[i + w]
            windows.append(list(acc))
            window_starts.append(list(starts))
            window_ids.append(list(ids))
    out, matched = [], 0
    for c in chunks:
        cw = _lyric_norm_words(c.get("text", ""))
        d = dict(c)
        if not cw:
            out.append(d)
            continue
        best, best_k = 0.0, None
        sm = difflib.SequenceMatcher(None, cw, [], autojunk=False)
        for k, ww in enumerate(windows):
            sm.set_seq2(ww)
            if sm.real_quick_ratio() <= best or sm.quick_ratio() <= best:
                continue
            r = sm.ratio()
            if r > best:
                best, best_k = r, k
        if best >= 0.4:
            matched += 1
        if best >= 0.5 and best_k is not None:
            best_w = windows[best_k]
            sm.set_seq2(best_w)
            fixed = []
            for op, i1, i2, j1, j2 in sm.get_opcodes():
                if op == "replace" and (i2 - i1) == (j2 - j1):
                    fixed.extend(best_w[j1:j2])
                else:
                    fixed.extend(cw[i1:i2])
            d["text"] = " ".join(fixed)
            # Where each tag line begins inside this chunk (word indices), so
            # the section map can time the lines one by one: Whisper merges
            # them into chunks whose start can be several seconds before the
            # line that matters.
            # Only a close match is split into lines (and given stanzas): a
            # chunk that runs across lines that are not consecutive in the
            # tags (a chorus written out once) matches some window only
            # loosely, and its cut points would be guesses.
            # A joined run is used only when it fits clearly better.
            if not want_lines:
                out.append(d)
                continue
            cut = (windows[best_k], window_starts[best_k], window_ids[best_k], best)
            if joined and best < 0.999:
                jbest = None
                for jw, js_, ji in joined:
                    sm.set_seq2(jw)
                    floor = best + 0.05 if jbest is None else jbest[3]
                    if sm.real_quick_ratio() <= floor or sm.quick_ratio() <= floor:
                        continue
                    r = sm.ratio()
                    if r > floor and (jbest is None or r > jbest[3]):
                        jbest = (jw, js_, ji, r)
                if jbest is not None and jbest[3] >= best + 0.05:
                    cut = jbest
            segs = []
            if cut[3] >= STRUCTURE_TAG_LINE_MIN_MATCH:
                segs = _tag_line_cuts(cw, cut[0], cut[1], cut[2])
            if segs:
                cuts_ = [a for a, _l in segs]
                stz_ = [by_text.get(" ".join(lines[lid]), [(tag[lid][1], first_of[tag[lid][1]] == lid, lid)])
                        for _a, lid in segs]
                if cuts_[0] > 0:
                    # Words before the first line found (sung but not in that
                    # line): kept as a part of their own, without a stanza.
                    cuts_ = [0] + cuts_
                    stz_ = [[]] + stz_
                d["lines"] = cuts_ + [len(fixed)]
                d["line_stanzas"] = stz_
        out.append(d)
    if matched < 0.3 * len(chunks):
        return chunks
    return out


def _tag_timing_quality(chunks, tag_lyrics):
    """dev53: share of the embedded lyrics' words that a timed pass matched
    with usable line timing -- words in chunks no longer than
    WHISPER_TIMING_MAX_CHUNK_S that match a run of one to four consecutive tag
    lines at 0.5 or better -- or None when the tags have too few lines to
    judge (fewer than four)."""
    import difflib
    tag = _tag_lyric_lines(tag_lyrics)
    lines = [w for w, _st in tag]
    total = sum(len(w) for w in lines)
    if len(lines) < 4 or total == 0:
        return None
    windows = []
    for i in range(len(lines)):
        acc = []
        for w in range(4):
            if i + w >= len(lines):
                break
            acc = acc + lines[i + w]
            windows.append(list(acc))
    matched = 0
    for c in chunks or []:
        try:
            dur = float(c["end"]) - float(c["start"])
        except Exception:
            continue
        cw = _lyric_norm_words(c.get("text", ""))
        if not cw or dur > WHISPER_TIMING_MAX_CHUNK_S:
            continue
        sm = difflib.SequenceMatcher(None, cw, [], autojunk=False)
        best = 0.0
        for ww in windows:
            sm.set_seq2(ww)
            if sm.real_quick_ratio() <= best or sm.quick_ratio() <= best:
                continue
            best = max(best, sm.ratio())
        if best >= 0.5:
            matched += len(cw)
    return min(1.0, matched / float(total))


def _retry_tag_lyric_timing(chunks, resolved_path, tag_lyrics, vocal_stem_path=None):
    """dev53: see WHISPER_TIMING_RETRY. Returns the chunks to use."""
    try:
        q0 = _tag_timing_quality(chunks, tag_lyrics)
    except Exception:
        return chunks
    if q0 is None or q0 >= WHISPER_TIMING_RETRY_BELOW:
        return chunks
    attempts = []
    if vocal_stem_path and os.path.exists(str(vocal_stem_path)):
        attempts.append(("the isolated vocal", vocal_stem_path))
    attempts.append(("the mix", resolved_path))
    best_q, best_chunks, best_src = q0, chunks, None
    for label, path in attempts:
        try:
            status(f"Lyric timing matched {q0 * 100:.0f}% of the embedded lyrics; "
                   f"re-timing from {label}...")
            _t, ch = _whisper_transcribe_once(path, return_chunks=True, no_repeat_ngram_size=0)
            ch = _align_chunks_to_tag_lyrics(ch, tag_lyrics)
            q = _tag_timing_quality(ch, tag_lyrics)
        except Exception as e:
            print(f"  (lyric re-timing from {label} skipped: {e})")
            continue
        if q is not None and q > best_q:
            best_q, best_chunks, best_src = q, ch, label
        if best_q >= WHISPER_TIMING_RETRY_BELOW:
            break
    if best_src is not None and best_q >= q0 + WHISPER_TIMING_RETRY_GAIN:
        status_done(f"Lyric timing from {best_src}: {best_q * 100:.0f}% of the embedded lyrics matched "
                    f"(first pass {q0 * 100:.0f}%)")
        return best_chunks
    status_done(f"Lyric re-timing did not improve on the first pass ({q0 * 100:.0f}% matched); keeping it")
    return chunks


def _detect_lyric_repetitions(chunks, similarity_threshold=0.82, min_words=3):
    """Cluster timestamped lyric lines by text similarity so recurring
    lines (chorus/refrain/hook) can be identified by where the words
    actually repeat, independent of any assumed intro/verse/chorus
    template. Returns a list of {"text", "count", "occurrences": [(start,
    end), ...]} for groups that recur 2+ times, ordered by first
    occurrence."""
    import difflib

    candidates = []
    for c in (chunks or []):
        norm = _normalize_lyric_line(c.get("text", ""))
        if len(norm.split()) < min_words:
            continue
        candidates.append({"norm": norm, "raw": c["text"].strip(),
                            "start": float(c["start"]), "end": float(c["end"])})

    used = [False] * len(candidates)
    groups = []
    for i in range(len(candidates)):
        if used[i]:
            continue
        group = [candidates[i]]
        used[i] = True
        for j in range(i + 1, len(candidates)):
            if used[j]:
                continue
            ratio = difflib.SequenceMatcher(
                None, candidates[i]["norm"], candidates[j]["norm"]
            ).ratio()
            ai = set(candidates[i]["norm"].split())
            aj = set(candidates[j]["norm"].split())
            jaccard = (len(ai & aj) / float(len(ai | aj))) if (ai | aj) else 0.0
            # Similarity must be both lexical and structural; this avoids using a
            # short generic phrase as a false chorus anchor.
            if ratio >= similarity_threshold and (jaccard >= 0.55 or min(len(ai), len(aj)) >= 6):
                group.append(candidates[j])
                used[j] = True
        if len(group) >= 2:
            groups.append({
                "text": group[0]["raw"],
                "count": len(group),
                "occurrences": [(g["start"], g["end"]) for g in group],
            })

    groups.sort(key=lambda g: g["occurrences"][0][0])
    return groups


def build_lyric_repetition_report(chunks, max_groups=8):
    """Format _detect_lyric_repetitions() output into an evidence block for
    the model, alongside the objective audio measurements."""
    groups = _detect_lyric_repetitions(chunks)
    if not groups:
        return ""

    lines = [
        "LYRIC REPETITION MAP (measured directly from timestamped lyric "
        "transcription -- independent of any assumed verse/chorus template)",
    ]
    for idx, g in enumerate(groups[:max_groups], start=1):
        times = ", ".join(_seconds_to_mmss(t[0]) for t in g["occurrences"])
        snippet = g["text"] if len(g["text"]) <= 60 else g["text"][:57] + "..."
        lines.append(f'- recurring line (x{g["count"]}): "{snippet}" at {times}')
    lines.append(
        "Sections whose lyrics recur here are the SAME recurring section (e.g. every "
        "occurrence of a chorus/refrain/hook) even if the song does not follow a "
        "standard intro/verse/chorus/verse/chorus/bridge/chorus template -- use this "
        "to assign matching STRUCTURE labels to matching timestamps, and distinct "
        "labels (or neutral Section 1/2/3 names) where nothing here repeats, rather "
        "than forcing every song into a fixed verse/chorus alternation. "
        "Starts of *blocks* of the same recurring material (not every single line "
        "hit) are used as secondary structure-boundary candidates when a section "
        "change has little/no timbral shift — acoustic novelty peaks still take priority."
    )
    return "\n".join(lines)


class _WhisperMpsTrim(_HFLogitsProcessor):
    """Calls _moss_mps_trim() every `every` generation steps (Whisper,
    WHISPER_MPS_CACHE_TRIM_EVERY). Returns the scores untouched."""

    def __init__(self, every):
        self.every = max(1, int(every))
        self.steps = 0

    def __call__(self, input_ids, scores):
        try:
            self.steps += 1
            if self.steps % self.every == 0:
                if WHISPER_RELEASE_BEFORE_LOAD and _gpu_cache_clear_ok():
                    try:
                        torch.mps.synchronize()     # dev41: let finished work let go of its buffers
                    except Exception:
                        pass
                _moss_mps_trim()
        except Exception:
            pass
        return scores


def _sanitize_lyrics_transcription(text):
    """
    Cut token-loop / runaway repetition and spam/hallucination tails in
    lyrics transcriptions (first written for the previous listening model's
    lyrics pass; the pass now uses Whisper).

    Handles:
      - formal [END OF TRANSCRIPTION] and broken variants ([endtranscription})
      - mid-line space-collapse (Thisisalinewithitsspacesgone…)
      - fake section tags (**instrumentalsolo**, JSON blobs)
      - chatty outros, bios, URLs, copyright dumps, letter-salad
    """
    if not text:
        return text

    # Accept several end-marker spellings; truncate to the first one found.
    end_m = re.search(
        r"\[?\s*END[\s_]*OF[\s_]*TRANSCRIPTION\s*\]?"
        r"|\[?\s*endtranscription\s*[\]\}]?",
        text,
        re.IGNORECASE,
    )
    if end_m:
        text = text[: end_m.start()].rstrip() + "\n[END OF TRANSCRIPTION]"

    # --- Spam / hallucination onset markers ---------------------------------
    # These are split into two tiers, because this function was originally
    # tuned for the previous listening model (Qwen2.5-7B), which liked to finish a
    # transcription with a chatty sign-off, a bio, or a copyright dump. The
    # pipeline now transcribes with Whisper, which does not editorialise --
    # but it DOES faithfully transcribe a singer saying "see ya later",
    # "bye bye" or "peace out". So the same patterns that used to catch model
    # spam now mostly catch real lyrics, and because a match truncates
    # everything after it, one such phrase mid-song silently deleted the rest
    # of the words. Measured on ordinary lyrics, that was throwing away
    # roughly two thirds of the song.
    #
    # STRUCTURAL: things a singer essentially never sings -- URLs, emails,
    # phone numbers, JSON, licensing metadata, explicit statements about the
    # transcription itself. Still truncate on sight.
    _SPAM_STRUCTURAL_PATTERNS = [
        r"(?i)\bI[\s'\u2019\u00b4´`]*[Mm]\s+sorry\s+but\s+the\s+rest\s+is\s+(?:just\s+)?gibberish",
        r"(?i)\bthe\s+rest\s+is\s+(?:just\s+)?gibberish\s+and\s+stupid",
        r"(?i)\bplease\s+ignore\s+my\s+(?:explanately\s+)?long\s+bio",
        r"(?i)\bfor\s+more\s+information\s+about\s+me\b",
        r"(?i)\bI\s+am\s+also\s+available\b",
        r"(?i)\bmy\s+email\s+address\b",
        r"(?i)\bUSPS\s+Address\s*:",
        r"(?i)\bPO\s*Box\s*#?\s*\d+",
        r"(?i)\bMy\s+Phone\s+Number\b",
        r"(?i)\+1\s*\(\d{3}\)\s*\d{3}",
        r"(?i)https?://[^\s]+",
        r"(?i)www\.[a-z0-9\-]+\.[a-z]{2,}",
        r"(?i)en\.wiki(?:pedia)?\.org",
        r"(?i)\bCopyright\s+Owner\s*\(?s?\)?\s*:",
        r"(?i)\bLicensees?\s*:",
        r"(?i)\bSong\s+Title\s*:\s*[\"\u201c]",
        r"(?i)\bComposer\s+ID\s*#",
        r"(?i)@\w+\s*[®©℗™]",
        r"(?i)\bASCAP\s+LLC\b",
        r"(?i)\bUniversal\s+Music\s+Publishing\b",
        r"(?i)\bWarner\s+Chappell\b",
        r"(?i)\.{2,}\s*INAUDIBLE\s*\.{2,}\s*I[\s'\u2019\u00b4´`]*[Mm]\s+SORRY",
        r"(?i)\*{0,3}\s*TRANSCRIBING\s+INCOMPLETE\s*!?\s*\*{0,3}",
        r"(?i)\*{0,3}\s*transcribing\s+completed\s*\*{0,3}",
        r"(?i)\btranscription\s+(?:incomplete|failed|aborted)\b",
        r"(?i)```\s*json\b",
        r"(?i)\{\s*[\"']genre[\"']\s*:",
        r"(?i)\{\s*[\"']artist[\"']\s*:",
        r"(?i)\{\s*[\"']duration[\"']\s*:",
        r"(?i)\(Note\)\s*:\s*These\s+annotations",
        r"(?i)\bCorrect\s+edits\s+welcome\b",
        r"(?i)\bWELCOME\s+HOME\s+MR\.?\s+WHITE\b",
        r"(?i)\bWHAT\s+DID\s+NIGEL\s+SAINT\b",
        r"(?i)\bHORRIFICANT\b",
        r"[~°•⁣⁡˚˖✧⋆♾️🍁💔❤😭☹👽❄🔥]{3,}",
        # Space-collapsed word salad: 28+ letters with no whitespace.
        r"[A-Za-z]{28,}",
    ]

    # AMBIGUOUS: ordinary things people sing. These are only treated as spam
    # when a STRUCTURAL marker also fired somewhere -- i.e. the decode has
    # demonstrably gone off the rails and produced a URL, a bio or a JSON
    # blob. Position alone is not enough: plenty of songs genuinely end on
    # "see ya later" or "bye bye", and cutting the closing line of a song is
    # exactly the failure this tier exists to avoid.
    _SPAM_AMBIGUOUS_PATTERNS = [
        r"(?i)\bAll\s+rights\s+reserved\b",
        r"(?i)\bThank\s*you\s+kindly\b",
        r"(?i)\bHorrorcore\s+Dark\s+Ambient\b",
        r"(?i)\*{0,3}\s*instrumental\s*solo\s*\*{0,3}",
        r"(?i)\*{0,3}\s*guitar\s*solo\s*\*{0,3}",
        r"(?i)\bsee\s+ya\s+later\b",
        r"(?i)\bbyebye\b",
        r"(?i)\bstay\s+positive\s+all\s+right\b",
        # Sustained sung vowels are normal ("<sung vowel>"); only an extreme run is
        # a token loop, so this needs far more repetition than it used to.
        r"(?:[a-z]{1,4}){0,2}(?:o{6,}|a{6,}|e{6,}|i{6,}|u{6,}|y{6,}){6,}",
    ]

    # SIGN-OFF: emphatic, punctuation-heavy farewells that read as the model
    # closing out rather than as sung words. Rare enough in lyrics to act on
    # position, so these fire either with structural corroboration OR when
    # they sit in the closing stretch of the transcription.
    _SPAM_SIGNOFF_PATTERNS = [
        r"(?i)\bPeace\s+Out\s*!{2,}",
        r"(?i)\bLove\s+everybody\s*!!",
        r"(?i)\bHope\s+yall\s+have\s+fun\b",
        r"(?i)\bImma\s+go\s+(?:now|home)\b",
        r"(?i)\bTHANKYOU\s+VERY\s+much\b",
        r"(?i)\bgood\s+luck\s+have\s+fun\b",
        r"(?i)\balright\s+alright\s+good\s+luck\b",
    ]

    earliest = None
    structural_hit = False
    for pat in _SPAM_STRUCTURAL_PATTERNS:
        m = re.search(pat, text)
        if m:
            pos = m.start()
            # For pure letter-runs, only treat as spam if there is already
            # some coherent lyric text before it (avoid cutting short words).
            if pat == r"[A-Za-z]{28,}" and pos < 40:
                continue
            structural_hit = True
            if earliest is None or pos < earliest:
                earliest = pos

    # Ambiguous phrases need the decode to have demonstrably failed elsewhere.
    if structural_hit:
        for pat in _SPAM_AMBIGUOUS_PATTERNS:
            m = re.search(pat, text)
            if m and (earliest is None or m.start() < earliest):
                earliest = m.start()

    # Emphatic sign-offs also fire on position, in the closing stretch.
    tail_zone = len(text) * 0.85
    for pat in _SPAM_SIGNOFF_PATTERNS:
        m = re.search(pat, text)
        if m:
            pos = m.start()
            if not (structural_hit or pos >= tail_zone):
                continue
            if earliest is None or pos < earliest:
                earliest = pos

    if earliest is not None and earliest > 20:
        cut = text[:earliest].rstrip()
        # Only fall back to the previous line break when the cut actually
        # lands mid-line. Doing it unconditionally threw away the last
        # complete lyric line before every junk block, which on a short
        # transcription is a large fraction of the song.
        mid_line = earliest > 0 and text[earliest - 1] not in "\n\r"
        last_nl = cut.rfind("\n")
        if mid_line and last_nl > 20:
            cut = cut[:last_nl].rstrip()
        elif mid_line and len(cut) > 80:
            # Prefer last ordinary space before the collapse so we don't leave
            # a half-glued word (And nostranger / Living In the‎…).
            soft = cut
            for ch in ("\u2009", "\u202f", "\u2005", "\u2006", "\u200a", "\u00a0", "\u200b"):
                soft = soft.replace(ch, " ")
            last_sp = max(soft.rfind(" "), soft.rfind("\t"))
            if last_sp > 40:
                cut = cut[:last_sp].rstrip()
            else:
                for sep in (". ", "! ", "? ", "... ", ", "):
                    pos = cut.rfind(sep)
                    if pos > 40:
                        cut = cut[: pos + len(sep)].rstrip()
                        break
        # Drop a trailing fragment that looks like a glued partial word
        tail = cut[-24:] if len(cut) > 24 else cut
        if " " not in tail and re.search(r"[A-Za-z]{8,}$", cut):
            sp = cut.rfind(" ")
            if sp > 40:
                cut = cut[:sp].rstrip()
        text = cut + "\n[END OF TRANSCRIPTION]"

    def _is_letter_salad(s):
        if not s or len(s) < 24:
            return False
        letters = re.sub(r"[^a-zA-Z]", "", s)
        if len(letters) < 20:
            return False
        repeats = sum(1 for a, b in zip(letters, letters[1:]) if a.lower() == b.lower())
        if repeats / max(1, len(letters) - 1) > 0.45:
            return True
        unique = len(set(letters.lower()))
        if unique <= 8 and len(letters) > 40:
            return True
        if re.search(r"[a-zA-Z]{28,}", s):
            return True
        return False

    def _is_chatty_meta_line(s):
        """Drop a whole line only for markers a singer never sings.

        This previously also matched "peace out", "see ya later", "byebye",
        "love everybody", "stay positive", "hope yall", "imma go",
        "take care everyone" and "instrumental solo" -- all ordinary lyric
        content -- and deleted the entire line wherever they appeared, mid-song
        included. Genuine model sign-offs at the end of a transcription are
        handled by the ambiguous-marker tier above, which requires
        corroboration; they don't need a second, indiscriminate pass here.
        """
        if not s:
            return False
        low = s.lower()
        markers = (
            "transcribing", "transcription incomplete",
            "transcription completed", "welcome home mr", "horrificant",
            "correct edits welcome", "annotations were generated",
            "```json", '"genre"', '"artist"', '"duration"',
        )
        return any(m in low for m in markers)

    lines = text.splitlines()
    out = []
    prev = None
    streak = 0
    for line in lines:
        stripped = line.strip()
        if stripped and re.fullmatch(r"(.)\1{7,}", stripped.replace(" ", "")):
            continue
        if _is_letter_salad(stripped) or _is_chatty_meta_line(stripped):
            continue
        if stripped and prev is not None and stripped == prev:
            streak += 1
            if streak >= 3:
                continue
        else:
            streak = 1 if stripped else 0
            prev = stripped if stripped else prev
        out.append(line)

    cleaned = "\n".join(out).strip()

    def _collapse_phrase_loop(s):
        """Tame a runaway phrase loop without flattening a real hook.

        Sung repetition is normal -- "na na na na na na na na" is the lyric,
        not a decoding fault -- so this now needs a longer run before it acts
        (8+ repeats rather than 5) and leaves more of it intact (4 rather
        than 2), which still defuses a token loop while keeping hooks.
        """
        m = re.search(r"(.{3,40}?)\1{7,}", s)
        if not m:
            return s
        phrase = m.group(1)
        return re.sub(re.escape(phrase) + r"{5,}", phrase * 4, s)

    cleaned = _collapse_phrase_loop(cleaned)
    # Collapse only genuine character loops. At the old threshold (10+) an
    # ordinary sustained sung vowel -- "<sung vowel>", which Whisper writes out
    # faithfully -- was being cut back to three letters.
    cleaned = re.sub(r"(.)\1{15,}", r"\1" * 6, cleaned)
    cleaned = re.sub(r"[~°•⁣⁡˚˖✧⋆♾️🍁💔❤😭☹👽❄🔥]{2,}", " ", cleaned)
    # Normalize exotic Unicode spaces that often appear at the onset of collapse
    for ch in ("\u2009", "\u202f", "\u2005", "\u2006", "\u200a", "\u00a0", "\u200b", "\u200e", "\u200f"):
        cleaned = cleaned.replace(ch, " ")
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    # Drop a trailing glued fragment like "And nostranger" when the last token
    # looks like two words jammed together.
    #
    # The length half of this test used to fire on any final token of 10+
    # lowercase letters, which silently deleted the last word of any song
    # ending on "everything", "understand", "remembering" and so on. A plain
    # long word is not evidence of collapse, so length alone now only counts
    # when the decode was ALREADY truncated for spam -- i.e. there is
    # independent evidence this transcription went wrong. An internal capital
    # (camelCase glue) remains evidence on its own.
    parts = cleaned.rsplit(" ", 1)
    if len(parts) == 2:
        _glued = re.search(r"[a-z]{3,}[A-Z]", parts[1]) or (
            earliest is not None and re.fullmatch(r"[a-z]{14,}", parts[1] or "")
        )
        if _glued:
            cleaned = parts[0].rstrip()

    # Strip any trailing junk that looks like JSON / markdown fences
    cleaned = re.split(r"\n\s*```|\n\s*\{\s*[\"'](?:genre|artist|duration)", cleaned, maxsplit=1)[0].rstrip()

    if not cleaned.endswith("[END OF TRANSCRIPTION]"):
        # Ensure a clean terminator when we truncated mid-stream
        if len(cleaned) > 0:
            cleaned = cleaned.rstrip() + "\n[END OF TRANSCRIPTION]"

    # Length ceiling. This used to cut anything over 3500 characters down to
    # 2500 and label it "spam/loop detected" -- but a rap track or any song
    # much over four minutes routinely transcribes longer than that, so real
    # lyrics were being discarded and then reported to the writer as model
    # spam. Runaway loops are already collapsed above; this is now only a
    # backstop against a pathological decode flooding the context window, set
    # well clear of normal song lengths, and it says honestly why it trimmed.
    if len(cleaned) > LYRICS_MAX_CHARS:
        head = cleaned[:LYRICS_MAX_CHARS]
        last_nl = head.rfind("\n")
        if last_nl > 800:
            head = head[:last_nl]
        cleaned = head.rstrip() + "\n[END OF TRANSCRIPTION — truncated for length]"

    cleaned = _flag_repeated_section_drift(cleaned)

    return cleaned


def _cross_check_lyrics_transcriptions(primary, secondary, min_unverified_run=5):
    """
    Compare two INDEPENDENT decodes of the same lyrics-transcription prompt
    against the same audio (one greedy, one sampled) and flag stretches of
    `primary` where they diverge.

    Why this catches what n-gram/repetition tuning cannot: greedy decoding
    is deterministic, so a single decode can be fluently, confidently wrong
    with no repetition anywhere to detect. When the listening model is actually
    grounded in the audio for a passage, independent decodes — even with
    different sampling — tend to converge on the same or near-same words,
    because the audio is anchoring the output. When grounding is weak and
    the model is filling in from its language-model prior instead of the
    audio (a generic "song about games" continuation, say), independent
    decodes tend to diverge, because that prior isn't anchored to anything
    and can resolve differently each time. Divergence is treated purely as
    a confidence signal — NOT as evidence that either reading is correct.
    """
    if not primary or not secondary:
        return primary

    import difflib

    def _tokenize(s):
        return re.findall(r"\n|\S+", s)

    def _key(tok):
        return tok.lower().strip(".,!?\"'()[]")

    a_tokens = _tokenize(primary)
    b_tokens = _tokenize(secondary)
    a_keys = [_key(t) for t in a_tokens]
    b_keys = [_key(t) for t in b_tokens]

    sm = difflib.SequenceMatcher(None, a_keys, b_keys, autojunk=False)

    out_pieces = []

    def _emit(tok):
        if tok == "\n":
            out_pieces.append("\n")
        else:
            if out_pieces and out_pieces[-1] not in ("\n", ""):
                out_pieces.append(" ")
            out_pieces.append(tok)

    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        run_len = i2 - i1
        if tag == "equal" or run_len < min_unverified_run:
            for tok in a_tokens[i1:i2]:
                _emit(tok)
            continue

        # A disagreement long enough to be a plausible fabricated line/phrase
        # rather than incidental word-choice noise ("played" vs "plays").
        alt_words = [t for t in b_tokens[j1:j2] if t != "\n"]
        alt = " ".join(alt_words).strip()

        _emit("[UNVERIFIED\u2192")
        for tok in a_tokens[i1:i2]:
            _emit(tok)
        n_primary = len([t for t in a_tokens[i1:i2] if t != "\n"])
        if n_primary >= 6 and len(alt_words) <= max(2, int(0.25 * n_primary)):
            # The second decode has (almost) nothing here -- it skipped the
            # passage. Saying the two "disagree" (rock test track: a whole
            # chorus against the other reading "Bye.") overstates it: the
            # passage is unconfirmed, not contradicted.
            note = (
                " \u2190the second, independent transcription has almost nothing here"
                + (f' (only "{alt}")' if alt else "")
                + ", so this passage is unconfirmed rather than contradicted; treat it as "
                "unverified, not as wrong]"
            )
        else:
            note = (
                f" \u2190two independent transcriptions disagree here"
                + (f'; other reading: "{alt}"' if alt else "; no comparable alt reading")
                + "; do not treat either as confirmed]"
            )
        out_pieces.append(note)

    return "".join(out_pieces)
