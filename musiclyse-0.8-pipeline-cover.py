# MUSICLYSE FILE pipeline-cover 0.8
# ==============================================================================
#  MUSICLYSE 0.8: COVER ART
# ==============================================================================
# This file holds cover art: the cover-art prompt and reading its answer. It is part of
# musiclyse-0.8.py, which loads it whole at start-up, after its own code, into its own namespace
# (see "THE MUSICLYSE FILES" there); it is not run on its own.
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
# (COVER_IMAGE_SENDABLE_MIMES moved to the settings file beside this script.)

NO_COVER_SENTINEL = "__no_cover__"


def prepare_cover_image_for_ollama(image_bytes, mime="image/jpeg"):
    if not image_bytes:
        return None

    guessed_mime = _guess_image_mime(image_bytes)
    sendable = (mime in COVER_IMAGE_SENDABLE_MIMES) or (guessed_mime in COVER_IMAGE_SENDABLE_MIMES)

    # If the embedded art is already small enough, avoid an ffmpeg round-trip.
    if sendable and len(image_bytes) <= 350_000:
        return image_bytes

    attempts = [
        (COVER_IMAGE_MAX_DIMENSION, "2"),
        (384, "3"),
        (256, "4"),
        (192, "5"),
    ]

    last = None
    for dim, q in attempts:
        data = _ffmpeg_scale_to_jpeg(image_bytes, dim, q)
        if data:
            last = data
            # Keep the base64 payload reasonably small.
            if len(data) <= 350_000:
                return data

    # Last resort: send the original embedded art only if Ollama can accept it directly.
    if sendable and image_bytes and len(image_bytes) <= MAX_COVER_BYTES_TO_SEND_RAW:
        return image_bytes

    if last and len(last) <= MAX_COVER_BYTES_TO_SEND_RAW:
        return last

    return None


COVER_ART_OBSERVATION_PROMPT = """Inspect only the attached album/cover artwork.

Do not infer anything about the audio.

Return exactly these lines:

PEOPLE=none|one|two|three_or_more
PERSON_CUES=<visible people and cautious apparent presentation/age range; if none say none>
TEXT_LOGOS=<visible title/artist/label/logos/text; if none say none>
ERA_CUES=<visual era clues from typography/fashion/photography/design/colour; if uncertain say uncertain>
STYLE_VIBE=<brief visual description>
CONFIDENCE=low|medium|high

RULES:

- Describe what is visibly present.
- Do not identify a person unless the image/text makes that identity reasonably clear.
- Apparent age is visual presentation, not verified age.
- Do not infer anything about vocal characteristics.
- Do not infer musical genre solely from artwork.
- Do not infer exact release year solely from artwork.
"""


# dev67: the same blocks in the same places in every saved file (see REPORT_FIXED_BLOCKS).
_COVER_NONE_TEXT = "none -- this file has no embedded cover art, so there is no visual evidence for this track."
_COVER_NONE_RECORDED = "none recorded for this file's artwork."


def _has_cover_observations(text):
    """True when the analysis carries real cover-art observations (the dev67
    "none" placeholder does not count)."""
    t = str(text or "")
    if "COVER ART OBSERVATIONS" not in t:
        return False
    return not re.search(r"COVER ART OBSERVATIONS:\n(?:" + re.escape(_COVER_NONE_TEXT) + "|"
                         + re.escape(_COVER_NONE_RECORDED) + r")", t) or t.count("COVER ART OBSERVATIONS") > 1


def _fixed_blocks_cover(t, has_cover):
    if "COVER ART OBSERVATIONS" in t:
        return t
    block = "COVER ART OBSERVATIONS:\n" + (_COVER_NONE_RECORDED if has_cover else _COVER_NONE_TEXT)
    m = re.search(r"\n\nSINGER IDENTITY RESOLUTION \(", t)
    if m is None:
        m = re.search(r"\n\nACOUSTIC VOCAL READING \(", t)
    if m is None:
        m = re.search(r"\n\nRECOMMENDED STRUCTURE FOR DISCUSSION", t)
    if m is None:
        return t.rstrip("\n") + "\n\n" + block
    return t[:m.start()] + "\n\n" + block + t[m.start():]


def _cover_reply_missing_image(text):
    """Reject model attachment refusals, not legitimate covers with no people."""
    t = str(text or "")
    return bool(re.search(
        r"(?i)(?:no|not? an?y?)\s+(?:album[/ -]*)?(?:cover\s+)?(?:artwork|image|picture|photo)"
        r"[^.\n]{0,90}\b(?:attached|provided|uploaded|received)|"
        r"(?:artwork|image|picture|photo)[^.\n]{0,60}\b(?:not|wasn't|isn't)\s+(?:attached|provided|uploaded)|"
        r"(?:please|need (?:you )?to)\s+upload\s+(?:the|an?|your)\s+(?:cover\s+)?(?:image|artwork|picture)", t))


def _cover_observation_is_useful(obs):
    if not obs or _cover_reply_missing_image(str(obs.get("raw") or "") + " " + str(obs.get("style_vibe") or "")):
        return False
    keys = ("people", "person_cues", "text_logos", "era_cues", "style_vibe")
    return any(str(obs.get(k) or "").strip() for k in keys)


def _parse_cover_observation(text):
    obs = {"raw": (text or "").strip()}
    if COVER_FIELDS_TOLERANT:                     # 0.7.1-pre12: bold labels and bullets
        text = re.sub(r"(?m)^[ \t]*(?:[-*\u2022]|\d+[.)])[ \t]+", "", _strip_md_emphasis(text or ""))
    for field in ("PEOPLE", "PERSON_CUES", "TEXT_LOGOS", "ERA_CUES", "STYLE_VIBE", "CONFIDENCE"):
        m = re.search(rf"^{field}\s*[:=]\s*(.+)$", text or "", re.MULTILINE | re.IGNORECASE)
        obs[field.lower()] = m.group(1).strip() if m else ""
    if COVER_FIELDS_TOLERANT:
        obs["people"] = _cover_people_count(obs["people"])
    return obs


# 0.7.1-pre12 COVER_FIELDS_TOLERANT: PEOPLE reduced to none/one/two/three_or_more.
_COVER_PEOPLE_FORMS = (
    ("three_or_more", r"three_or_more|three|four|five|six|seven|eight|nine|ten|several|many|multiple|"
                      r"(?:a\s+)?(?:group|band)\b|[3-9]\b|\d\d"),
    ("none", r"none|zero|0\b|no\s+(?:people|person|one|humans?)\b|nobody"),
    ("one", r"one\b|1\b|a\s+single\b|single\b"),
    ("two", r"two\b|2\b|a\s+(?:pair|couple|duo)\b|pair\b|couple\b|duo\b"),
)


def _cover_people_count(value):
    v = str(value or "").strip().strip("*_`\"'<>[] ").lower()
    for canon, pat in _COVER_PEOPLE_FORMS:
        if re.match(r"(?:" + pat + r")(?![a-z_])", v):
            return canon
    return str(value or "").strip()


def _format_cover_observation_block(obs):
    if not obs:
        return ""

    lines = [
        "COVER ART OBSERVATIONS (structured visual evidence from the attached artwork; not direct audio proof):"
    ]

    labels = [
        ("people", "People"),
        ("person_cues", "Person cues"),
        ("text_logos", "Text/logos"),
        ("era_cues", "Era cues"),
        ("style_vibe", "Style/vibe"),
        ("confidence", "Confidence"),
    ]

    for key, label in labels:
        val = str(obs.get(key) or "").strip()
        lines.append(f"{label}: {val if val else 'unavailable'}")

    return "\n".join(lines)


def describe_cover_art(cover_b64):
    """
    Return a structured dict of visual observations from an attached image.
    Returns {} on failure or if nothing useful was extracted.
    """
    if not ENABLE_COVER_ART_DESCRIPTION or not OLLAMA_SUPPORTS_IMAGES:
        return {}

    if not cover_b64 or cover_b64 == NO_COVER_SENTINEL:
        return {}

    # In batch, unload the writer immediately after this short helper so it
    # does not sit in RAM while MOSS-Music / Demucs / Omnizart run.
    _ka = None
    if globals().get("BATCH_UNLOAD_OLLAMA", True) and _is_batch_context():
        _ka = globals().get("BATCH_OLLAMA_KEEP_ALIVE", 0)

    # Ollama allocates its context buffer at model-load time, so a request
    # with a different num_ctx than the currently-resident instance forces a
    # full reload -- discarding any cached conversation state. In batch,
    # there's no ongoing conversation to protect (the model is unloaded right
    # after this call anyway via keep_alive above), so the small dedicated
    # context is a genuine, harmless memory/time saving. In an interactive
    # session, though, this call runs mid-conversation whenever a track is
    # loaded (see resolve_singer_identity's call site too) -- forcing the
    # resident large-context writer instance to reload down to 8192 for this
    # one message, then reload again back up for the very next reply, losing
    # everything cached from the conversation so far. Passing None here lets
    # ollama_chat() fall back to its normal default (OLLAMA_NUM_CTX), which
    # is what the surrounding interactive conversation already requests as
    # its first, preferred attempt -- so this call reuses the resident
    # instance instead of evicting it.
    _cover_art_ctx = COVER_ART_DESCRIPTION_NUM_CTX if _is_batch_context() else None

    try:
        text, _usage = ollama_chat([
                {
                    "role": "user",
                    "content": COVER_ART_OBSERVATION_PROMPT,
                    "images": [cover_b64],
                }
            ],
            num_ctx=_cover_art_ctx,
            keep_alive=_ka,
            require_images=True,
            max_output_tokens=4096,   # dev2: was 1024; reasoning could use it all, leaving no description
            options=_scan_llm_options(),
        )
    except Exception as e:
        print(f"  (cover art description failed: {e})")
        return {}
    finally:
        if _ka is not None:
            try:
                ollama_unload_model()
            except Exception:
                pass

    if _cover_reply_missing_image(text):
        print("  (artwork is present, but the vision model did not inspect it; no description recorded)")
        return {}
    obs = _parse_cover_observation(text)

    if not _cover_observation_is_useful(obs):
        stripped = (text or "").strip()
        if len(stripped) > 20:
            # Fallback: keep the prose as a style/vibe note rather than discarding it entirely.
            obs["style_vibe"] = stripped[:500]
            return obs
        return {}

    return obs
