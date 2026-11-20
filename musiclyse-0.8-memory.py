# MUSICLYSE FILE memory 0.8
# ==============================================================================
#  MUSICLYSE 0.8: MEMORIES
# ==============================================================================
# This file holds memories: /memory, the memory file, the screens for saving and listing
# memories, and the memory note the default chat gets. It is part of musiclyse-0.8.py, which
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


# ---- 0.7.1-pre30 CHAT_MEMORY: /memory -------------------------------------------------------------
# A memory is a short note the chat model writes about a conversation when asked (/memory, or
# /memory <topic> for one thing), kept only after the user has seen it on a screen (Enter saves it,
# e edits it, Esc cancels). Memories live in one file (CHAT_MEMORY_FILE). In the default chat, a new
# conversation gets them as one note right after the system prompt -- memories about the loaded
# tracks first, then the newest, up to CHAT_MEMORY_MAX_TOKENS -- and a track loaded later brings
# any of its memories that weren't in that note. /memory list shows them (d deletes one);
# /memory off and /memory on turn bringing them into conversations off and on.
_MEMORY_TAG = "[Musiclyse memories]"
_MEMORY_EXTRA_TAG = "[Musiclyse memories about a loaded track]"
_MEMORY_NOTE_INTRO = (
    "Short notes you saved from earlier conversations with this user, each with its date and the songs "
    "loaded then. They are your own recollections of those chats: bring one up when it's naturally "
    "relevant -- for example when a song you talked about comes up again -- but don't recite them, and "
    "don't mention them in every reply. They aren't measurements: for anything about a song, the loaded "
    "exports come first. This note is application bookkeeping: don't mention the note itself.")
_MEMORY_EXTRA_INTRO = ("More notes you saved from earlier conversations, about a track that has just been "
                       "loaded. Use them as the earlier memory note says.")
_MEMORY_REQUEST = (
    "[Memory request -- application bookkeeping] The user has asked you to remember this conversation{about}. "
    "Write a short memory of it for you to look back on in future chats with this user: two or three "
    "plain sentences, written as your own note (for example \"We talked about ...\"). Name the songs and "
    "artists, and say what the user liked, thought or asked for, and anything you worked out together. "
    "Leave out measurements unless they mattered to the conversation. Reply with the memory only -- no "
    "heading, no list and no quotation marks.")
# 0.8-pre31 CHAT_MEMORY_NOT_SOUND: the note's wording with the sound kept to the exports.
_MEMORY_NOTE_INTRO_SOUND = (
    "Short notes you saved from earlier conversations with this user, each with its date and the songs "
    "loaded then. They are your own recollections of those chats: bring one up when it's naturally "
    "relevant -- for example when a song you talked about comes up again -- but don't recite them, and "
    "don't mention them in every reply. They remind you what was said, not how the recordings sound: for "
    "anything about a song, the loaded exports come first, and with no export loaded, say what the two of "
    "you talked about rather than describing the song's sound as if you'd just heard it. What the user "
    "thought or corrected is their view: bring it up as theirs. This note is application bookkeeping: "
    "don't mention the note itself.")
# 0.8-pre31 CHAT_MEMORY_SAYS_WHAT_WAS_SAID: added to the request, before "Leave out measurements".
# 0.8-pre33 CHAT_MEMORY_REQUEST_KEEPS_NOTES: added to the request, before "Reply with the memory only",
# when the conversation sent with it holds a memory note.
_MEMORY_REQUEST_NOT_OLD = (
    "The notes marked as memories from earlier conversations are only there for reference: leave out "
    "anything from them that didn't come up again in this conversation. ")
_MEMORY_REQUEST_SAID = (
    "Write what was said about a song as what was said (\"you felt the chorus lifted\", \"we talked about "
    "the bass coming in late\"), not as facts about the recording, and note the user's corrections as "
    "theirs. ")


def _memory_note_intro():
    """0.8-pre31 CHAT_MEMORY_NOT_SOUND: the memory note's opening; False gives pre30's."""
    return _MEMORY_NOTE_INTRO_SOUND if CHAT_MEMORY_NOT_SOUND else _MEMORY_NOTE_INTRO


def _memory_request_text():
    """0.8-pre31 CHAT_MEMORY_SAYS_WHAT_WAS_SAID: the memory request; False gives pre30's."""
    if CHAT_MEMORY_SAYS_WHAT_WAS_SAID:
        return _MEMORY_REQUEST.replace("Leave out measurements", _MEMORY_REQUEST_SAID + "Leave out measurements", 1)
    return _MEMORY_REQUEST


_MEMORY_STATE = {"system": None, "status": None, "since": ""}   # the conversation the note was decided for
_MEMORY_NOTE_IDS = {}            # id(note message) -> (the message, ids of the memories it holds)
_MEMORY_BAD_FILE = []            # a memory file that couldn't be read: kept aside before the next save


def _memory_path():
    """The memory file: CHAT_MEMORY_FILE, or memories.json beside the main script."""
    if CHAT_MEMORY_FILE:
        return os.path.abspath(os.path.expanduser(CHAT_MEMORY_FILE))
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "memories.json")


def _memory_data():
    """{"enabled": bool, "memories": [...]} from the memory file; a missing file is no memories.
    A file that can't be read counts as no memories, and is set aside (not overwritten) at the next
    save."""
    path = _memory_path()
    empty = {"version": 1, "enabled": True, "memories": []}
    if not os.path.isfile(path):
        return empty
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, dict) or not isinstance(data.get("memories"), list):
            raise ValueError("not a Musiclyse memory file")
        memories = [m for m in data["memories"]
                    if isinstance(m, dict) and isinstance(m.get("text"), str) and m.get("text").strip()
                    and isinstance(m.get("id"), str)]
        return {"version": 1, "enabled": data.get("enabled", True) is not False, "memories": memories}
    except (OSError, ValueError, TypeError) as exc:
        if path not in _MEMORY_BAD_FILE:
            _MEMORY_BAD_FILE.append(path)
            print(f"  (The memory file {path} couldn't be read ({exc}); it is left as it is, and set "
                  "aside as a .unreadable copy if a memory is saved.)")
        return empty


def _memory_write(data):
    """Write the memory file in one step (a temporary file, then renamed over it)."""
    import tempfile
    path = _memory_path()
    folder = os.path.dirname(path) or "."
    os.makedirs(folder, exist_ok=True)
    if path in _MEMORY_BAD_FILE and os.path.isfile(path):
        aside = path + datetime.datetime.now().strftime(".%Y-%m-%d-%H-%M-%S.unreadable")
        os.replace(path, aside)
        _MEMORY_BAD_FILE.remove(path)
        print(f"  (The unreadable memory file was kept as {aside}.)")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=folder, prefix=".memories-",
                                         suffix=".tmp", delete=False) as fh:
            temporary = fh.name
            json.dump({"version": 1, "enabled": bool(data.get("enabled", True)),
                       "memories": list(data.get("memories") or [])}, fh, ensure_ascii=False, indent=1)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            try:
                os.remove(temporary)
            except OSError:
                pass


def _memory_tracks(loaded_exports):
    """[{"artist", "title"}] of the loaded exports' tracks, each once."""
    out, seen = [], set()
    for entry in (loaded_exports or {}).values():
        for track in entry.get("tracks") or []:
            artist = str(track.get("artist") or "").strip()
            title = str(track.get("title") or "").strip()
            key = (_txt_name_fold(artist), _txt_name_fold(title))
            if title and key not in seen:
                seen.add(key)
                out.append({"artist": artist, "title": title})
    return out


def _memory_about(memory, tracks):
    """True when `memory` was saved with one of `tracks` loaded."""
    want = {(_txt_name_fold(t["artist"]), _txt_name_fold(t["title"])) for t in tracks}
    return any((_txt_name_fold(t.get("artist") or ""), _txt_name_fold(t.get("title") or "")) in want
               for t in memory.get("tracks") or [] if isinstance(t, dict))


def _memory_day(memory):
    try:
        d = datetime.date.fromisoformat(str(memory.get("date") or "")[:10])
        return f"{d.day} {d.strftime('%b %Y')}"
    except ValueError:
        return "an earlier day"


def _memory_track_words(memory):
    names = [(f"{t.get('artist')} – {t.get('title')}" if t.get("artist") else str(t.get("title") or ""))
             for t in memory.get("tracks") or [] if isinstance(t, dict) and t.get("title")]
    return "; ".join(names)


def _memory_line(memory):
    songs = _memory_track_words(memory)
    return f"- {_memory_day(memory)}" + (f" ({songs})" if songs else "") + ": " + " ".join(memory["text"].split())


def _memory_note(memories, tracks, tag, intro):
    """(note text, ids) for the memories that fit in CHAT_MEMORY_MAX_TOKENS: those about `tracks`
    first, then the rest, newest first in each group; ("", set()) when none fits."""
    newest = sorted(memories, key=lambda m: str(m.get("created") or m.get("date") or ""), reverse=True)
    order = [m for m in newest if _memory_about(m, tracks)] + [m for m in newest if not _memory_about(m, tracks)]
    budget = int(CHAT_MEMORY_MAX_TOKENS or 0) - _chat_text_tokens(
        tag + "\n" + (_MEMORY_NOTE_INTRO if intro == _MEMORY_NOTE_INTRO_SOUND else intro))   # 0.8-pre31: as many memories as pre30 fitted
    lines, ids = [], set()
    for m in order:
        line = _memory_line(m)
        cost = _chat_text_tokens(line + "\n")
        if cost > budget:
            continue
        budget -= cost
        lines.append(line)
        ids.add(m["id"])
    if not lines:
        return "", set()
    return tag + "\n" + intro + "\n" + "\n".join(lines), ids


def _memory_notes_in(history):
    return [m for m in history if m.get("role") == "user"
            and str(m.get("content") or "").startswith((_MEMORY_TAG + "\n", _MEMORY_EXTRA_TAG + "\n"))]


def _memory_remove_notes(history):
    """Take every memory note out of `history`; True when there was one."""
    notes = {id(m) for m in _memory_notes_in(history)}
    if notes:
        history[:] = [m for m in history if id(m) not in notes]
        for i in notes:
            _PROTECTED_EVIDENCE_MESSAGE_IDS.discard(i)
    return bool(notes)


def _memory_new_note(content, ids):
    msg = {"role": "user", "content": content}
    _PROTECTED_EVIDENCE_MESSAGE_IDS.add(id(msg))       # kept like a loaded export
    _MEMORY_NOTE_IDS[id(msg)] = (msg, set(ids))
    return msg


def _memory_sync(history, default_chat, loaded_exports, question=None):
    """0.7.1-pre30 CHAT_MEMORY: before each chat request, the memory note as it should be. In the
    default chat with memories on, a new conversation (start-up, /clear, /clearall, a change of
    character, a resumed save) gets one note right after the system prompt, decided once (a memory
    saved later in the same conversation waits for the next one); a track loaded later adds a short
    note, just before the question, with its memories that weren't in it. Otherwise no note."""
    if not CHAT_MEMORY or not history:
        return
    data = _memory_data()
    if not default_chat or not data["enabled"]:
        _memory_remove_notes(history)
        return
    system = history[0]
    if _MEMORY_STATE["system"] is not system:
        _MEMORY_STATE.update(system=system, status=None, since=datetime.datetime.now().isoformat())
        _memory_remove_notes(history)          # a resumed save's note is replaced by today's
    memories = [m for m in data["memories"] if str(m.get("created") or "") <= _MEMORY_STATE["since"]]
    tracks = _memory_tracks(loaded_exports)
    notes = _memory_notes_in(history)
    if _MEMORY_STATE["status"] is None or (_MEMORY_STATE["status"] == "added"
                                           and not any(str(m["content"]).startswith(_MEMORY_TAG + "\n")
                                                       for m in notes)):
        _memory_remove_notes(history)
        content, ids = _memory_note(memories, tracks, _MEMORY_TAG, _memory_note_intro())
        if content:
            history.insert(1, _memory_new_note(content, ids))
        _MEMORY_STATE["status"] = "added" if content else "none"
        return
    if _MEMORY_STATE["status"] != "added" or not tracks:
        return
    have = set()
    for m in notes:
        have |= _MEMORY_NOTE_IDS.get(id(m), (None, set()))[1]
    extra = [m for m in memories if m["id"] not in have and _memory_about(m, tracks)]
    content, ids = _memory_note(extra, tracks, _MEMORY_EXTRA_TAG, _MEMORY_EXTRA_INTRO) if extra else ("", set())
    if content:
        position = len(history)
        if (question is not None and history and history[-1] is question and history[-1].get("role") == "user"
                and not str(history[-1].get("content") or "").startswith(
                    ("[Musiclyse export loaded:", "[Background looked up for this conversation]"))):
            position -= 1
        history.insert(position, _memory_new_note(content, ids))


def _memory_reset_note(history):
    """Take the memory notes out of the conversation; the next request decides the note afresh."""
    _memory_remove_notes(history)
    _MEMORY_STATE["status"] = None


def _memory_request(messages):
    """One request for the memory text; not part of the conversation."""
    payload = {"model": OLLAMA_MODEL, "messages": messages, "stream": False, "think": False,
               "options": {"num_ctx": _chat_num_ctx(), "num_predict": 1024, "temperature": 0.3}}
    if OLLAMA_INTERACTIVE_KEEP_ALIVE is not None:
        payload["keep_alive"] = OLLAMA_INTERACTIVE_KEEP_ALIVE
    response = requests.post(OLLAMA_URL, json=payload, timeout=900)
    if response.status_code == 400 and "think" in response.text.lower():
        payload.pop("think")
        response = requests.post(OLLAMA_URL, json=payload, timeout=900)
    response.raise_for_status()
    data = response.json()
    if not isinstance(data, dict) or data.get("error"):
        raise RuntimeError("Ollama reported an error")
    content = (data.get("message") or {}).get("content") if isinstance(data.get("message"), dict) else None
    if not isinstance(content, str) or not content.strip():
        raise ValueError("the chat model returned nothing")
    return content


def _memory_write_summary(history, topic):
    """The chat model's memory of the conversation in `history` (or of `topic` in it)."""
    work = list(history)                       # the same messages, so loaded exports stay protected
    _chat_trim_history(work, _chat_num_ctx())
    tags = (_MEMORY_TAG + "\n", _MEMORY_EXTRA_TAG + "\n")
    request = _memory_request_text()
    if CHAT_MEMORY_REQUEST_KEEPS_NOTES:
        # 0.8-pre33: the messages as the chat sends them, notes and pictures included, so the
        # conversation Ollama has already read is reused rather than read again
        messages = []
        for m in work:
            msg = {"role": m.get("role"), "content": str(m.get("content") or "")}
            if m.get("images"):
                msg["images"] = list(m["images"])
            messages.append(msg)
        if any(msg["content"].startswith(tags) for msg in messages):
            request = request.replace("Reply with the memory only", _MEMORY_REQUEST_NOT_OLD + "Reply with the memory only", 1)
    else:
        messages = [{"role": m.get("role"), "content": str(m.get("content") or "")} for m in work
                    if not str(m.get("content") or "").startswith(tags)]
    about = f", in particular this: {topic}" if topic else ""
    messages.append({"role": "user", "content": request.replace("{about}", about)})
    text = " ".join(_memory_request(messages).split()).strip().strip('"“”').strip()
    if CHAT_SPACED_DASHES:
        text = _chat_space_dashes(text)
    return text


def _memory_review_screen(text):
    """The memory on a screen in the style of the menus: Enter saves it, e edits it, Esc cancels.
    Returns "save", "edit" or "cancel". Without a terminal, asks on the line instead."""
    def ask():
        print("  Memory: " + text)
        answer = input("  Save this memory? (y = save, e = edit, anything else = cancel) ").strip().lower()
        return {"y": "save", "yes": "save", "e": "edit"}.get(answer, "cancel")
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        return ask()
    try:
        import curses
    except ImportError:
        return ask()

    def show(screen):
        colour = 0                          # MEMORY_PREVIEW_STEADY_COLOUR: the text's own colour
        if USE_COLOR:
            try:
                curses.start_color()
                curses.use_default_colors()
                curses.init_pair(1, curses.COLOR_YELLOW, -1)
                screen.bkgd(" ", curses.color_pair(1))
                if MEMORY_PREVIEW_STEADY_COLOUR:
                    colour = curses.color_pair(1)
            except curses.error:
                pass
        try:
            curses.curs_set(0)
        except curses.error:
            pass
        first = MEMORY_PREVIEW_STEADY_COLOUR
        while True:
            height, width = screen.getmaxyx()
            if first:
                screen.clear()              # the whole screen is drawn again when the preview opens
                first = False
            else:
                screen.erase()

            def line(row, words, attr=0):
                if row < height and width > 1:
                    try:
                        screen.addnstr(row, 0, words, width - 1, attr | colour)
                    except curses.error:
                        pass
            line(0, "Save this memory?", curses.A_BOLD)
            rows = textwrap.wrap(text, max(10, width - 2)) or [""]
            for row, words in enumerate(rows[:max(1, height - 4)]):
                line(row + 2, words)
            line(height - 1, "Enter: save   e: edit   Esc: cancel", curses.A_BOLD)
            screen.refresh()
            key = screen.getch()
            if key in (10, 13, curses.KEY_ENTER):
                return "save"
            if key in (ord("e"), ord("E")):
                return "edit"
            if key in (27, ord("q"), ord("Q"), 3, 4):
                return "cancel"
    try:
        return curses.wrapper(show)
    except KeyboardInterrupt:
        return "cancel"
    except curses.error:
        return ask()


def _memory_edit(text):
    """The memory text edited on the input line (Enter keeps it); None when cancelled."""
    try:
        if sys.stdin.isatty() and sys.stdout.isatty():
            return _edit_load_draft("  Memory: ", Ansi.LIGHT_GREEN, text).strip()
        return input("  Memory: ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return None


def _memory_list_screen():
    """/memory list: the memories in a menu like /lookup-search's (Enter reads one in full, d deletes
    one after a y/n check). Returns the ids deleted."""
    data = _memory_data()
    memories = sorted(data["memories"], key=lambda m: str(m.get("created") or m.get("date") or ""), reverse=True)
    state = "on" if data["enabled"] else f"off ({MEMORY_FLAG} on brings them into conversations)"
    if not memories:
        print(f"  No memories saved yet -- {MEMORY_FLAG} saves one from this conversation. "
              f"Memories are {state}.\n")
        return set()
    labels = [_memory_line(m)[2:] for m in memories]
    deleted = set()

    def read(index):
        if memories[index]["id"] in deleted:
            return ("message", "(already deleted)")
        m = memories[index]
        songs = _memory_track_words(m)
        return ("show", f"{_memory_day(m)}" + (f" -- {songs}" if songs else "")
                + (f" -- about: {m['topic']}" if m.get("topic") else "") + "\n\n" + m["text"])

    def delete(index):
        if memories[index]["id"] in deleted:
            return ("message", "(already deleted)")

        def then():
            fresh = _memory_data()
            fresh["memories"] = [m for m in fresh["memories"] if m.get("id") != memories[index]["id"]]
            _memory_write(fresh)
            deleted.add(memories[index]["id"])
            labels[index] = "(deleted)"
            return "Deleted."
        return ("confirm", "Delete this memory? (y/n)", then)
    pick = _pick_export_menu(f"Your memories ({len(memories)}) -- memories are {state}", labels,
                             actions={"enter": read, "d": delete}, help_extra="Enter: read   d: delete")
    if pick is not None and not (sys.stdin.isatty() and sys.stdout.isatty()):
        # without a terminal the menu is a numbered list: the memory chosen is printed
        print("  " + read(pick)[1].replace("\n", "\n  ") + "\n")
    return deleted


def _memory_command(rem, history, default_chat, loaded_exports):
    """0.7.1-pre30 CHAT_MEMORY: /memory [topic], /memory list, /memory on, /memory off."""
    word = (rem or "").strip()
    low = word.lower()
    if low in ("on", "off"):
        data = _memory_data()
        data["enabled"] = low == "on"
        try:
            _memory_write(data)
        except OSError as exc:
            print(f"  Couldn't change the memory setting: {exc}\n")
            return
        _memory_reset_note(history)
        if low == "on":
            print("  Memories are on: they're brought into the default chat from your next message.\n")
        else:
            print(f"  Memories are off: they aren't brought into conversations (they're kept; "
                  f"{MEMORY_FLAG} on brings them back). You can still save and list them.\n")
        return
    if low == "list":
        deleted = _memory_list_screen()
        if deleted:
            if any(ids & deleted for _m, ids in (_MEMORY_NOTE_IDS.get(id(n), (None, set()))
                                                 for n in _memory_notes_in(history))):
                _memory_reset_note(history)
            print(f"  Deleted {len(deleted)} memor{'y' if len(deleted) == 1 else 'ies'}.\n")
        return
    if not default_chat:
        print(f"  Memories are for the default chat only: {PERSONA_FLAG} reset (or {GAME_EXIT_FLAG}) "
              "first.\n")
        return
    said = [m for m in history[1:] if m.get("role") == "assistant" and str(m.get("content") or "").strip()]
    if not said:
        print(f"  Nothing to remember yet: chat about something first, then use {MEMORY_FLAG}.\n")
        return
    status("Writing a memory of this conversation...")
    try:
        text = _memory_write_summary(history, word)
    except Exception as exc:
        status_done(clear=True)
        print(f"  Couldn't write a memory: {exc}\n")
        return
    status_done(clear=True)
    while True:
        choice = _memory_review_screen(text)
        if choice == "edit":
            edited = _memory_edit(text)
            if edited:
                text = edited
            continue
        break
    if choice != "save":
        print("  (Memory not saved.)\n")
        return
    data = _memory_data()
    now = datetime.datetime.now()
    data["memories"].append({"id": now.strftime("%Y%m%d%H%M%S") + "-" + os.urandom(3).hex(),
                             "date": now.date().isoformat(), "created": now.isoformat(),
                             "text": text, "tracks": _memory_tracks(loaded_exports), "topic": word})
    try:
        _memory_write(data)
    except OSError as exc:
        print(f"  Couldn't save the memory: {exc}\n")
        return
    print(f"  Memory saved ({len(data['memories'])} in all). New conversations in the default chat will "
          "have it" + ("." if data["enabled"] else f" once memories are on ({MEMORY_FLAG} on).") + "\n")
