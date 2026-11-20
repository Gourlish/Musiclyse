# MUSICLYSE FILE wiki 0.8
# ==============================================================================
#  MUSICLYSE 0.8: THE LOCAL WIKIPEDIA DATABASE
# ==============================================================================
# This file holds the local Wikipedia database: /lookup and /lookup-search, the background
# blocks for the chat and the singer lookups, article search and ranking, and the wording of the
# background blocks. It is part of musiclyse-0.8.py, which loads it whole at start-up, after its
# own code, into its own namespace (see "THE MUSICLYSE FILES" there); it is not run on its own.
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


# --- Wikipedia background-context (local RAG) -------------------------------
# Read-only lookups against WIKI_DB_PATH, scoped strictly to non-technical
# background context for the writer prompt (see build_wiki_context_block()).
# The schema is auto-detected the first time it's needed so this works with
# whatever table/column names the database happens to use, rather than
# assuming one specific import-tool's layout.

_WIKI_DB_CONN = None
_WIKI_DB_SCHEMA = None            # dict: {"mode": "fts"|"plain", "table", "title_col", "text_col"}
_WIKI_DB_UNAVAILABLE_REASON = None

_WIKI_TITLE_COL_HINTS = ("title", "page_title", "article_title", "name")
_WIKI_TEXT_COL_HINTS = ("text", "body", "content", "extract", "summary", "article", "wikitext")


def _wiki_quote_ident(name):
    return '"' + str(name).replace('"', '""') + '"'


def _wiki_pick_col(cols, hints):
    lower = {c.lower(): c for c in cols}
    for hint in hints:
        for lc, orig in lower.items():
            if lc == hint or lc.endswith("_" + hint) or hint in lc:
                return orig
    return None


def _wiki_parse_fts_columns(create_sql):
    """Crude parse of 'CREATE VIRTUAL TABLE x USING fts5(title, body, ...)' to
    pull out the declared column names."""
    m = re.search(r"USING\s+fts\d*\s*\((.*)\)\s*;?\s*$", create_sql or "", re.IGNORECASE | re.DOTALL)
    if not m:
        return []
    cols = []
    for part in m.group(1).split(","):
        part = part.strip()
        if not part or "=" in part:  # skip options like content='other_table'
            continue
        cols.append(part.split()[0].strip('`"[]'))
    return cols


def _wiki_detect_schema(conn):
    """Inspect sqlite_master once to find an article table with a title-like
    and a body-like text column, preferring an existing FTS5 table (fast
    full-text search) over a plain table (exact/prefix match only, since a
    LIKE %term% scan over a multi-GB table with no index is too slow to run
    per track switch)."""
    cur = conn.cursor()
    cur.execute("SELECT name, sql, type FROM sqlite_master WHERE type IN ('table','view')")
    rows = cur.fetchall()

    fts_schema = None
    plain_schema = None

    for row in rows:
        name, sql, kind = row["name"], row["sql"] or "", row["type"]
        if name.startswith("sqlite_"):
            continue

        if "VIRTUAL TABLE" in sql.upper() and "FTS" in sql.upper():
            cols = _wiki_parse_fts_columns(sql)
            title_col = _wiki_pick_col(cols, _WIKI_TITLE_COL_HINTS)
            text_col = _wiki_pick_col(cols, _WIKI_TEXT_COL_HINTS)
            if title_col and text_col and fts_schema is None:
                fts_schema = {
                    "mode": "fts", "table": name, "title_col": title_col,
                    "text_col": text_col, "columns": cols,
                }
            continue

        if plain_schema is not None:
            continue
        try:
            cur.execute(f"PRAGMA table_info({_wiki_quote_ident(name)})")
            cols = [c["name"] for c in cur.fetchall()]
        except Exception:
            continue
        title_col = _wiki_pick_col(cols, _WIKI_TITLE_COL_HINTS)
        text_col = _wiki_pick_col(cols, _WIKI_TEXT_COL_HINTS)
        if title_col and text_col:
            plain_schema = {"mode": "plain", "table": name, "title_col": title_col, "text_col": text_col}

    schema = fts_schema or plain_schema
    if schema:
        # Optional: a redirect/alias mapping table (e.g. a dump of Wikipedia's
        # own redirects -- "Nickname" -> "The Group Name", a former band name, a
        # common misspelling). schema["redirect"] is None whenever no such
        # table exists, which is true for every database that predates this
        # feature, so this is a pure addition: nothing that already reads
        # this schema dict looks at a "redirect" key, and every code path
        # that DOES use it treats None exactly like "no redirect table".
        schema["redirect"] = _wiki_detect_redirect_schema(conn, schema["table"])
    return schema


_WIKI_REDIRECT_FROM_COL_HINTS = ("from", "source", "alias", "from_title")
_WIKI_REDIRECT_TO_COL_HINTS = ("to", "target", "dest", "destination", "canonical", "to_title")


def _wiki_detect_redirect_schema(conn, article_table):
    """Optionally detect a redirect/alias mapping table alongside the main
    article table.

    Deliberately conservative to keep false positives at zero: only tables
    whose NAME contains "redirect" or "alias" are even considered, so an
    unrelated two-text-column table already used for the article store can
    never be mistaken for one. Returns None when no such table is found --
    which every database used before this feature existed -- and callers
    must (and do) treat that exactly like the feature not existing at all.
    """
    try:
        cur = conn.cursor()
        cur.execute("SELECT name, type FROM sqlite_master WHERE type IN ('table','view')")
        rows = cur.fetchall()
    except Exception:
        return None

    for row in rows:
        name = row["name"]
        if not name or name == article_table or name.startswith("sqlite_"):
            continue
        name_l = name.lower()
        if "redirect" not in name_l and "alias" not in name_l:
            continue
        try:
            cur.execute(f"PRAGMA table_info({_wiki_quote_ident(name)})")
            cols = [c["name"] for c in cur.fetchall()]
        except Exception:
            continue
        from_col = _wiki_pick_col(cols, _WIKI_REDIRECT_FROM_COL_HINTS)
        to_col = _wiki_pick_col(cols, _WIKI_REDIRECT_TO_COL_HINTS)
        if from_col and to_col and from_col != to_col:
            return {"table": name, "from_col": from_col, "to_col": to_col}
    return None


_WIKI_REDIRECT_MAX_HOPS = 3


def _wiki_resolve_redirect(conn, schema, title_text):
    """Follow a redirect/alias table (if the DB has one) from title_text to
    its canonical target title, chasing a short bounded chain of redirects
    (an alias that itself points to another alias -- real Wikipedia dumps
    have this on occasion, e.g. after a band renamed twice).

    Returns None immediately -- with no query executed at all -- whenever
    the DB has no such table, which makes this a pure no-op for every
    database that predates this feature."""
    redirect = schema.get("redirect") if schema else None
    if not redirect or not title_text:
        return None
    try:
        cur = conn.cursor()
        table = _wiki_quote_ident(redirect["table"])
        from_col = _wiki_quote_ident(redirect["from_col"])
        to_col = _wiki_quote_ident(redirect["to_col"])

        current = title_text
        seen = {_wiki_title_compare_key(current)}
        target = None
        for _hop in range(_WIKI_REDIRECT_MAX_HOPS):
            cur.execute(
                f"SELECT {to_col} AS target FROM {table} WHERE {from_col} = ? COLLATE NOCASE LIMIT 1",
                (current,),
            )
            row = cur.fetchone()
            if not row or not row["target"]:
                break
            next_target = str(row["target"])
            key = _wiki_title_compare_key(next_target)
            if key in seen:
                # A cycle (A -> B -> A) -- stop and use whatever we already
                # resolved rather than looping forever.
                break
            seen.add(key)
            target = next_target
            current = next_target
        if target:
            return target
    except Exception:
        pass
    return None


def _get_wiki_db():
    """Lazily open a read-only connection to WIKI_DB_PATH and auto-detect its
    schema. Returns None (permanently, for this process) if the feature is
    disabled, the file is missing, or no usable article table is found."""
    global _WIKI_DB_CONN, _WIKI_DB_SCHEMA, _WIKI_DB_UNAVAILABLE_REASON

    if _WIKI_DB_CONN is not None:
        return _WIKI_DB_CONN
    if _WIKI_DB_UNAVAILABLE_REASON is not None:
        return None
    if not ENABLE_WIKI_CONTEXT:
        _WIKI_DB_UNAVAILABLE_REASON = "disabled"
        return None
    if not os.path.exists(WIKI_DB_PATH):
        _WIKI_DB_UNAVAILABLE_REASON = f"not found at {WIKI_DB_PATH}"
        return None

    try:
        import sqlite3
        uri = f"file:{os.path.abspath(WIKI_DB_PATH)}?mode=ro"
        conn = sqlite3.connect(uri, uri=True, timeout=WIKI_DB_TIMEOUT_S, check_same_thread=False)
        conn.row_factory = sqlite3.Row

        schema = _wiki_detect_schema(conn)
        if schema is None:
            conn.close()
            _WIKI_DB_UNAVAILABLE_REASON = "no title/text article table detected"
            print(
                "  (Wikipedia context DB found but its schema wasn't recognized "
                "— set ENABLE_WIKI_CONTEXT = False or check WIKI_DB_PATH's tables)"
            )
            return None

        _WIKI_DB_CONN = conn
        _WIKI_DB_SCHEMA = schema
        return conn
    except Exception as e:
        _WIKI_DB_UNAVAILABLE_REASON = str(e)
        return None


def _wiki_strip_possessive(word):
    """Strip a trailing possessive so search tokens actually match indexed
    text: "Band R's" -> "Band R", "Band'" -> "Band". Without
    this, our own tokenizer keeps the apostrophe-s attached to the token
    (see the regex below), so the FTS query ends up searching for the
    literal string "exampleband's" — which essentially never appears in
    indexed article text, since FTS5's own tokenizer splits that into
    "exampleband" + "s" — and the lookup silently finds nothing."""
    w = re.sub(r"[\u2019']s$", "", word, flags=re.IGNORECASE)
    w = re.sub(r"[\u2019']$", "", w)
    return w


def _wiki_fold_diacritics(text):
    """ASCII-fold diacritics for query expansion: 'Name' -> 'Name'.

    Used only to *add* extra search strings alongside the original form.
    Does not replace or mutate anything stored in the wiki DB, and does not
    change FTS/plain lookup logic — callers simply submit both variants."""
    if not text:
        return ""
    # NFKD splits ö → o + combining diaeresis; drop combining marks.
    decomposed = unicodedata.normalize("NFKD", str(text))
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def _wiki_fts_query_escape(q):
    """Quote each token individually so punctuation/apostrophes in artist
    names (e.g. Words N' Words) can't break FTS5 query syntax, after first
    stripping any trailing possessive so "Band R's" searches for
    "Band R" rather than the literal (unmatchable) token "Band R's".

    Also keeps common title punctuation (! ? . &) and Unicode letters so
    pages like "Name" or "Title!" are not token-stripped away."""
    # \w already matches Unicode letters (ö etc.) under Python 3 default flags.
    # Include a small set of title punctuation so "Title!" / "X.Y.Z." stay intact.
    tokens = re.findall(r"[\w'&.!?-]+", q or "")
    tokens = [_wiki_strip_possessive(t) for t in tokens]
    return " ".join(f'"{t}"' for t in tokens if t)


_WIKI_QUESTION_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "do", "does", "did", "this", "that",
    "these", "those", "what", "when", "where", "who", "whom", "which", "why", "how",
    "in", "on", "of", "for", "to", "and", "or", "it", "its", "song", "track", "album",
    "albums", "band", "artist", "music", "about", "tell", "me", "us", "can", "you",
    "please", "i", "we", "they", "he", "she", "be", "been", "with", "from", "by", "like",
    "good", "bad", "any", "some", "all", "just", "really", "very", "also", "too",
    "than", "then", "there", "here", "into", "over", "under", "after", "before",
    "debut", "first", "second", "third", "self", "titled", "eponymous", "full",
    "list", "listing", "tracklist", "tracklisting", "songs", "tracks", "reception",
    "reviews", "review", "regarded", "critical", "commercially", "commercially",
    "best", "greatest", "top", "favorite", "favourite", "favorites", "favourites",
    "worst", "essential", "underrated", "overrated",
    # Pronouns / follow-up filler that must never become wiki search tokens
    "them", "him", "her", "their", "theirs", "more", "else", "something", "anything",
    "someone", "anyone", "stuff", "things", "thing", "one", "ones",
}

# Sentence-initial words to strip off a capitalized-phrase match so e.g.
# "What Band R album..." yields the entity "Band R album" rather than
# "What Band R" — these are common question-starters, not band/album names.
_WIKI_LEADING_STOP_CAPS = {
    "the", "a", "an", "what", "who", "when", "where", "why", "how", "does", "did",
    "is", "are", "can", "could", "would", "will", "should", "tell", "please", "i",
    "was", "were", "do", "about",
}

# Imperative / interrogative words that open a question ("Compare Band G
# and Band F", "Describe Band R's sound", "Which Band album...").
# Being capitalised at the start of the sentence, they were glued onto the
# first proper noun ("Compare Band G", "Has Name"), and that phrase then
# matched no article, so the named act was never looked up at all. Stripped
# ONLY when they are the very first word of the question, so a band whose name
# begins with one of them is untouched when mentioned mid-sentence.
_WIKI_QUESTION_START_WORDS = {
    "compare", "comparing", "contrast", "describe", "explain", "rank", "rate",
    "recommend", "suggest", "which", "whose", "has", "have", "had", "list",
    "discuss", "summarise", "summarize", "review", "analyse", "analyze",
}

# Words that commonly appear inside album/artist names and should not break a
# title-case run (e.g. "Words N' Words", "Of Place", "The Group Name").
_WIKI_TITLE_INFIX = {
    "a", "an", "the", "of", "and", "or", "n", "n'", "for", "to", "in", "on", "at",
    "vs", "vs.", "with", "&",
}


def _wiki_question_keywords(question, max_terms=6):
    """Pull a handful of content words out of a user question for use as
    extra FTS search terms, dropping common stopwords/question words so the
    search isn't over-constrained by filler like 'what' or 'the'."""
    if not question:
        return []
    words = re.findall(r"[A-Za-z0-9']+", question.lower())
    seen = set()
    out = []
    for w in words:
        if len(w) > 2 and w not in _WIKI_QUESTION_STOPWORDS and w not in seen:
            seen.add(w)
            out.append(w)
    return out[:max_terms]


def _wiki_normalize_entity(phrase):
    phrase = _wiki_strip_possessive((phrase or "").strip())
    # Keep ! (real titles e.g. "Title!") but drop a trailing sentence "?"
    # that often sticks to the last token when the user asked a question.
    phrase = re.sub(r"\s+", " ", phrase).strip(" .,;:\"'")
    if phrase.endswith("?") and not phrase.endswith("!?"):
        phrase = phrase[:-1].rstrip()
    return phrase


def _wiki_extract_by_patterns(question):
    """Pull album/artist pairs from common English patterns that the pure
    title-case heuristic often misses, especially when the user doesn't
    capitalize every word:

      - "Title by Band"
      - "Band's Title"
      - "Band album Title"
      - "the album Title!"
    """
    if not question:
        return []

    entities = []
    q = question

    # Unicode-aware "starts with a capital letter" (covers Name, Ö, etc.).
    # [A-Z] alone is ASCII-only and silently dropped accented capitals.
    _cap = r"[A-Z\u00C0-\u00D6\u00D8-\u00DE]"
    # Continuation chars: word chars (incl. Unicode letters) + common title punct.
    _cont = r"[\w'&.!?-]*"

    # "TITLE by ARTIST" — locate "by ARTIST" first (artist is title-case),
    # then take the 1–4 tokens immediately before "by" as the title. This
    # avoids the left-greedy span "Tell me about Title by …".
    for m in re.finditer(
        rf"\sby\s+(?P<artist>{_cap}{_cont}(?:\s+(?:{_cap}{_cont}|N['\u2019]?)){{0,5}})",
        q,
    ):
        artist = _wiki_normalize_entity(m.group("artist"))
        # Tokens immediately before " by "
        before = q[:m.start()]
        before_tokens = re.findall(rf"[A-Za-z0-9\u00C0-\u00FF]{_cont}", before)
        # Drop trailing stopwords from the left context, then keep up to 4
        while before_tokens and before_tokens[-1].lower() in _WIKI_QUESTION_STOPWORDS | _WIKI_LEADING_STOP_CAPS:
            before_tokens.pop()
        title_tokens = before_tokens[-4:]
        while title_tokens and title_tokens[0].lower() in _WIKI_LEADING_STOP_CAPS | _WIKI_QUESTION_STOPWORDS:
            title_tokens = title_tokens[1:]
        # A real album/song title in a user's question is almost always
        # capitalized ("Title by Band"). Generic descriptive
        # phrasing that slips past the stopword strip above ("the best
        # albums by X", "any good songs by X") is not — so require at least
        # one capitalized token before trusting this as a title entity,
        # rather than searching the DB for whatever words happened to sit
        # before "by". Without this, a phrase like "best albums" gets
        # treated as a literal release title and matches generic pages
        # (award lists, "best of" compilations) that merely contain those
        # words, instead of the artist's own page.
        if title_tokens and not any(re.match(rf"{_cap}", t) for t in title_tokens):
            title_tokens = []
        title = _wiki_normalize_entity(" ".join(title_tokens))
        if artist and len(artist) > 1:
            entities.append(artist)
            if title and len(title) > 1:
                entities.append(title)
                entities.append(f"{artist} {title}")

    # "ARTIST's ALBUM" / "ARTIST' ALBUM" — artist is title-case; rest may be
    # the album title or words like "debut album".
    for m in re.finditer(
        rf"\b(?P<artist>{_cap}{_cont}(?:\s+(?:{_cap}{_cont}|N['\u2019]?)){{0,5}})"
        r"[\u2019']s?\s+"
        rf"(?P<rest>[A-Za-z0-9\u00C0-\u00FF]{_cont}(?:\s+[A-Za-z0-9\u00C0-\u00FF]{_cont}){{0,5}})",
        q,
    ):
        artist = _wiki_normalize_entity(m.group("artist"))
        # Strip leading stop-caps from artist ("What was Band R" -> "Band R")
        awords = artist.split()
        if m.start() == 0 and awords and awords[0].lower() in _WIKI_QUESTION_START_WORDS:
            awords = awords[1:]
        while awords and awords[0].lower() in _WIKI_LEADING_STOP_CAPS:
            awords = awords[1:]
        artist = " ".join(awords)
        rest = _wiki_normalize_entity(m.group("rest"))
        rwords = rest.split()
        while rwords and rwords[-1].lower() in (
            _WIKI_ALBUM_SIGNAL_WORDS
            | {"debut", "first", "self-titled", "eponymous", "self", "titled", "like", "about"}
        ):
            rwords = rwords[:-1]
        rest = " ".join(rwords)
        if artist and len(artist) > 1:
            entities.append(artist)
            if rest and len(rest) > 1 and rest.lower() not in _WIKI_QUESTION_STOPWORDS:
                entities.append(f"{artist} {rest}")
                entities.append(rest)

    # "album TITLE" / "record TITLE"
    for m in re.finditer(
        r"\b(?:album|record|lp|ep)\s+[\"'\u2018\u2019\u201c\u201d]?"
        rf"([A-Za-z0-9\u00C0-\u00FF]{_cont}(?:\s+[A-Za-z0-9\u00C0-\u00FF]{_cont}){{0,5}})"
        r"[\"'\u2018\u2019\u201c\u201d]?",
        q,
        re.IGNORECASE,
    ):
        title = _wiki_normalize_entity(m.group(1))
        twords = title.split()
        while twords and twords[-1].lower() in _WIKI_QUESTION_STOPWORDS | {"like", "about"}:
            twords = twords[:-1]
        title = " ".join(twords)
        if title and len(title) > 1 and title.lower() not in _WIKI_ALBUM_SIGNAL_WORDS:
            entities.append(title)

    return entities


_WIKI_DEMONYM_STOPWORDS = {
    "british", "english", "scottish", "irish", "welsh", "american", "canadian",
    "australian", "german", "french", "italian", "spanish", "swedish",
    "norwegian", "danish", "dutch", "belgian", "swiss", "austrian", "greek",
    "portuguese", "polish", "russian", "icelandic", "finnish", "japanese",
    "korean", "chinese", "indian", "brazilian", "mexican", "argentine",
    "chilean", "jamaican", "nigerian", "european", "african", "asian",
    "scandinavian", "nordic", "mancunian", "londoner",
}

_WIKI_ADVERB_STOPWORDS = {
    "however", "overall", "moreover", "furthermore", "nevertheless",
    "nonetheless", "meanwhile", "regardless", "conversely", "otherwise",
    "instead", "besides", "anyway", "though", "although",
}


def _wiki_is_adverbial_token(word):
    """Whether a single capitalised word should never be treated as an entity.

    Two families, both of which the title-case scanner cannot tell from a
    proper noun:

    Sentence-opening adverbs. "Stylistically, how different were..." made
    "Stylistically" an entity, which matched "The Words" and "An Example
    Stylist" through FTS stemming. The -ally/-ically test is length-gated at 8
    characters so real one-word names survive: Sally, Wally and Ally are all
    shorter, while Stylistically, Musically, Lyrically and Technically are
    caught.

    Nationality adjectives. "the British band Name" yielded ['British',
    'Name'], so "British Black music" and "British West Africa" filled slots;
    worse, "the English band Band B" yielded only ['English'] and lost the band
    altogether.

    Only applied to SINGLE words, so multi-word entities that legitimately
    begin with one of these — American Word, Japanese Word, British
    Word Word — are never affected.
    """
    if not word:
        return False
    w = str(word).strip().lower()
    if " " in w:
        return False
    if w in _WIKI_ADVERB_STOPWORDS or w in _WIKI_DEMONYM_STOPWORDS:
        return True
    return len(w) >= 8 and (w.endswith("ically") or w.endswith("ally"))


def _wiki_title_case_runs(question):
    """Find runs of title-case / proper-noun-ish words, allowing small
    lowercase infix words (of, the, and, N') so 'Words N' Words' and
    'The Word Words' still match as single entities.

    Token classes are Unicode-aware so names like Name / Name Ñame and
    titles with trailing ! (Title!) are kept intact."""
    if not question:
        return []

    _cap = r"[A-Z\u00C0-\u00D6\u00D8-\u00DE]"
    # Tokenize preserving apostrophes, diacritics, and common title punctuation
    tokens = re.findall(
        rf"[A-Za-z0-9\u00C0-\u00FF][\w'&.!?-]*|[\"'\u2018\u2019\u201c\u201d]",
        question,
    )
    entities = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if not re.match(rf"{_cap}", tok):
            i += 1
            continue
        run = [tok]
        j = i + 1
        while j < len(tokens):
            nxt = tokens[j]
            if re.match(rf"{_cap}", nxt):
                run.append(nxt)
                j += 1
            elif nxt.lower().strip(".'!?") in _WIKI_TITLE_INFIX and j + 1 < len(tokens) and re.match(rf"{_cap}", tokens[j + 1]):
                run.append(nxt)
                j += 1
            else:
                break
        # Strip leading stop-caps
        if i == 0 and run and run[0].lower() in _WIKI_QUESTION_START_WORDS:
            run = run[1:]
        while run and run[0].lower() in _WIKI_LEADING_STOP_CAPS:
            run = run[1:]
        phrase = _wiki_normalize_entity(" ".join(run))
        if phrase and len(phrase) > 2:
            # Prefer multi-word; single capitalized words only if reasonably long
            # (avoids "Tell", "Is", but keeps "Band R", "Band B", "Name")
            if " " in phrase or len(phrase) >= 5:
                entities.append(phrase)
        i = j if j > i else i + 1

    # Rescue short single-word names the length gate above discards. "Band B",
    # "Band C", "Band D", "BAND" and similar 3-4 letter names fail the `len >= 5`
    # test (which exists to keep "Tell", "Is", "Was" out), so "Tell me about
    # the English band Band B" lost the band entirely — the demonym filter
    # removed "English" and nothing else was left standing.
    #
    # Only rescued when DIRECTLY preceded by a word that names the kind of act
    # being discussed: band, artist, group, singer, duo, rapper, DJ, act. That
    # is a strong, deliberate signal — "the band Band B" is naming a subject, not
    # using an ordinary word incidentally. Without this anchor a rescue would
    # readmit exactly the noise the length gate exists to block.
    role_rescue = re.findall(
        rf"\b(?:band|artist|group|singer|duo|rapper|dj|act)\s+"
        rf"({_cap}[\w'&.!?-]{{1,4}})\b",
        question,
    )
    for tok in role_rescue:
        phrase = _wiki_normalize_entity(tok)
        if phrase and phrase not in entities:
            entities.append(phrase)

    # Rescue short SUBJECTS of a comparison question. "How did Band B compare to
    # Name?" has no role word before "Band B" for the rescue above to anchor on
    # — it's the grammatical subject of "did ... compare", not "the band X".
    # Name (5 letters) survived the normal length gate; Band B (4) did not, so
    # the question searched only for Name.
    #
    # Anchored on an auxiliary that directly precedes it, which is what marks
    # it as a sentence subject rather than an incidental word: "did/does/do/is/
    # was/were/would/has/have" + short capitalised word. A pronoun stoplist
    # keeps "Did She really..." or "Was It any good?" from being rescued as if
    # they were band names.
    _WIKI_RESCUE_PRONOUNS = {
        "he", "she", "it", "they", "we", "you", "i", "this", "that", "there",
        "here", "now", "then", "who", "what", "when", "where", "why", "how",
    }
    subject_rescue = re.findall(
        rf"\b(?i:did|does|do|is|was|were|would|has|have)\s+"
        rf"({_cap}[\w'&.!?-]{{1,4}})\b",
        question,
    )
    for tok in subject_rescue:
        if tok.lower() in _WIKI_RESCUE_PRONOUNS:
            continue
        phrase = _wiki_normalize_entity(tok)
        if phrase and phrase not in entities:
            entities.append(phrase)

    # Rescue short names that are the object of a comparison: "something like
    # Band B", "better than Band C", "Band A versus Band B". These are named subjects,
    # but a 3-4 letter name fails the length gate above and has no role word or
    # auxiliary in front of it, so "Recommend something like Band B" searched for
    # nothing at all.
    compare_rescue = re.findall(
        rf"\b(?i:like|than|versus|vs\.?)\s+({_cap}[\w'&.!?-]{{1,4}})\b",
        question,
    )
    compare_rescue += re.findall(
        rf"\b({_cap}[\w'&.!?-]{{1,4}})\s+(?i:versus|vs\.?)\s",
        question,
    )
    for tok in compare_rescue:
        if tok.lower() in _WIKI_RESCUE_PRONOUNS:
            continue
        phrase = _wiki_normalize_entity(tok)
        if phrase and phrase not in entities:
            entities.append(phrase)

    return entities


_WIKI_CONJOINED_SPLIT_RE = re.compile(r"\s+(?:and|or|&|vs\.?|versus|to)\s+", re.IGNORECASE)


def _wiki_split_conjoined_entity(entity, _depth=0):
    """Split "Word Words and Name" into its two acts.

    Title-case runs treat and/or as infix connectors, which is right for real
    names (Name and Name, First Last and the Group) but wrong when a
    question names two separate acts — the commonest shape in this app, since
    comparisons are most of what gets asked.

    Guards, tuned against real band names rather than in the abstract:
      * both halves must start with a capital;
      * neither half may begin "the " — that is what makes "Florence and the
        Machine", "First Last and the Group", "Name and the Group" and
        "Name and the Waves" survive intact;
      * at least one half must be two or more words, so one-word-plus-one-word
        names ("Name and Name", "Al and Bo", "Surname and Sons",
        "Surname & Surname") are never split.

    "Word Words and Name" passes (2 words + 1), which the earlier both-halves
    rule rejected — leaving the whole phrase searched as one string, matching
    "The Word Garden" and "Words of Words and Words" instead of either band.

    Returns [] when the entity should not be split. Callers ADD these to the
    existing entity rather than replacing it, so a genuine band name whose own
    article exists is still searched for first.
    """
    if not entity or _depth > 2:
        return []
    # maxsplit=1 so a chain is handled by the recursion below rather than
    # collapsing to "not exactly two parts" and being rejected outright.
    parts = _WIKI_CONJOINED_SPLIT_RE.split(entity, maxsplit=1)
    if len(parts) != 2:
        return []
    out = []
    for p in parts:
        p = p.strip(" ,;-")
        if not p or not p[:1].isupper():
            return []
        if p.lower().startswith("the "):
            return []
        # A comma INSIDE a half means the name is itself a serial list, not two
        # acts joined by a conjunction: "Word, Word & Word", "Name, Name and
        # Name", "Surname, Surname, Surname & Surname".
        if "," in p:
            return []
        out.append(p)
    # At least one half must be multi-word. This is what keeps one-plus-one
    # names ("Name and Name", "Al and Bo") and short titles ("Word
    # to Word", "Title A", "Title B") from ever being split.
    if max(len(p.split()) for p in out) < 2:
        return []

    # Recurse so a chain splits fully: "Band P or Band Q to
    # Band G" must yield all three acts, not stop at the first separator.
    leaves = []
    for p in out:
        for s_ in (_wiki_split_conjoined_entity(p, _depth + 1) or [p]):
            if s_ not in leaves:
                leaves.append(s_)
    return leaves


# Question verbs that are never the subject of a lookup on their own. Used only
# by the single-word keyword fallback, so "who produced this?" does not send the
# bare word "produced" to the article search as if it were a name.
_WIKI_BACKGROUND_VERB_STOPWORDS = {
    "produced", "producer", "producers", "released", "release", "written",
    "wrote", "writer", "recorded", "recording", "charted", "sampled", "samples",
    "played", "playing", "listen", "listening", "compare", "compared",
    "similar", "sounds", "genre", "lyrics", "meaning", "originally", "covered",
    "remixed", "inspired", "influenced", "influences",
}


def _wiki_question_entity_queries(question, max_entities=5):
    """Heuristically pull candidate entity phrases (artist/album/song names)
    out of a free-form question. Combines:
      - quoted phrases
      - "TITLE by ARTIST" / possessive patterns
      - title-case runs (with small-word infix tolerance)
      - keyword fallback
    """
    if not question:
        return []

    entities = []

    for m in re.finditer(
        r'["\u2018\u2019\u201c\u201d]([^"\u2018\u2019\u201c\u201d]{2,60})["\u2018\u2019\u201c\u201d]',
        question,
    ):
        phrase = _wiki_normalize_entity(m.group(1))
        if phrase:
            entities.append(phrase)

    entities.extend(_wiki_extract_by_patterns(question))
    title_runs = [r for r in _wiki_title_case_runs(question)
                  if not _wiki_is_adverbial_token(r)]
    entities.extend(title_runs)
    # A run like "Band G and Band F" names two acts, not one. Keep the
    # combined form first (real band names must still match their own article)
    # and append the halves so each act is actually searched for.
    for run in title_runs:
        entities.extend(_wiki_split_conjoined_entity(run))

    # Drop pure stopword / filler entities and lone words that are too generic
    junk = _WIKI_QUESTION_STOPWORDS | {
        "like", "about", "badly", "well", "good", "bad", "debut", "first",
        "second", "third", "self", "titled", "eponymous", "full", "list",
        "songs", "tracks", "reception", "review", "reviews", "regarded",
        "them", "him", "her", "more", "else", "something", "anything",
    }
    seen = set()
    out = []
    for e in entities:
        k = e.lower().strip()
        if not k or k in seen:
            continue
        if k in junk:
            continue
        # Lone short words are usually noise unless they look like a band name
        # (Band B, Band A, BAND) — keep length >= 4 for singles, or any multi-word.
        if " " not in e and len(e) < 4:
            continue
        seen.add(k)
        out.append(e)

    if out:
        return out[:max_entities]

    kw = _wiki_question_keywords(question, max_terms=6)
    # Only use keyword fallback when we have at least one solid content token
    # (avoids feeding "badly regarded" style questions as pseudo-entities).
    solid = [w for w in kw if w not in junk and len(w) > 3]
    if len(solid) >= 2:
        return [" ".join(solid[:4])]
    # A single, sufficiently long content word -- e.g. an unquoted,
    # uncapitalized "tell me about yesterday" -- is still worth trying, but
    # ONLY at the lowest possible priority: this returns a single bare word,
    # which the caller (build_wiki_context_block_multi) routes into
    # weak_q_entities exactly like any other single-word entity -- tried
    # dead last, after every stronger/continuity signal. It can therefore
    # only ever fill a slot the question's real subject left empty, never
    # displace it. The length-6 floor keeps this to reasonably distinctive
    # words rather than short common nouns that would just add search noise
    # (the same "one-word entities are risky" concern _WIKI_QUESTION_STOPWORDS
    # and the length>=4 rule above already apply to every other single word).
    #
    # Strip a trailing possessive/contraction FIRST: _wiki_question_keywords
    # tokenizes on [A-Za-z0-9'] and does not strip "'s", so a filler word
    # like "what's" (6 chars) would otherwise pass the length floor as if it
    # were a real content word, since it isn't in the stopword set under
    # that exact spelling. Re-checking the stripped form against `junk`
    # catches this the same way the rest of this function already handles
    # possessives elsewhere.
    if len(solid) == 1:
        candidate = _wiki_strip_possessive(solid[0])
        if (candidate and candidate.lower() not in junk and len(candidate) >= 6
                and candidate.lower() not in _WIKI_BACKGROUND_VERB_STOPWORDS):
            return [candidate]
    return []


_WIKI_ALBUM_SIGNAL_WORDS = {
    "album", "albums", "record", "records", "lp", "ep", "release", "released",
    "compilation", "compilations", "anthology", "anthologies",
}

_WIKI_TRACKLIST_SIGNAL = {
    "track", "tracks", "tracklist", "tracklisting", "track-list", "songs", "song",
    "listing", "side", "sides", "disc", "cds", "discography", "discog",
    "albums", "releases", "catalogue", "catalog", "singles",
}

_WIKI_RECEPTION_SIGNAL = {
    "reception", "review", "reviews", "reviewed", "regarded", "critical",
    "critics", "acclaim", "criticised", "criticized", "response", "chart",
    "charts", "sales", "sold", "peaked", "rating", "ratings", "good", "bad",
    "poorly", "well", "received", "commercial", "commercially",
}

_WIKI_DEBUT_SIGNAL = {
    "debut", "first album", "first record", "eponymous", "self-titled",
    "self titled", "same name",
}

# Multi-word phrases that signal a compilation/best-of release, checked as
# substrings (like _WIKI_DEBUT_SIGNAL) since "greatest hits" and "best of"
# don't survive single-word tokenization the way "compilation" does.
_WIKI_COMPILATION_PHRASES = (
    "greatest hits", "best of", "hits collection", "singles collection",
)


def _wiki_mentions_album(question):
    """Whether the question seems to be asking about an album at all (vs. a
    song, the artist generally, etc.) — used to trigger the self-titled-
    album search tier below, not as a literal search token."""
    if not question:
        return False
    q = question.lower()
    words = set(re.findall(r"[a-z0-9']+", q))
    if words & _WIKI_ALBUM_SIGNAL_WORDS:
        return True
    if words & _WIKI_TRACKLIST_SIGNAL or words & _WIKI_RECEPTION_SIGNAL:
        return True
    if any(s in q for s in _WIKI_DEBUT_SIGNAL):
        return True
    if any(s in q for s in _WIKI_COMPILATION_PHRASES):
        return True
    return False


# Anaphora that can refer to a RECORDING or an act — "it", "they", "this
# album". Deliberately excludes the singular personal pronouns below: those
# refer to a person, and in a conversation that is not always the artist. "Do
# you think he'll like any of these?" is about the listener, not the loaded
# track, so it must not pull the track's identity into the search.
_WIKI_TRACK_ANAPHORA_RE = re.compile(
    r"\b(it|its|it's|they|their|theirs|them|"
    # Bare "this" used as a pronoun for the loaded recording: "who produced
    # this?", "when was this released?", "who wrote this". These are among
    # the commonest background questions, and without this the loaded track
    # was dropped from the search and only the word "produced" was looked up.
    r"this(?=\s*[?.!,;]|\s*$|\s+(?:is|was|were|come|came|get|got|released|recorded|written|"
    r"made|produced|sound|sounds|feel|feels|chart|charted|about|from|by|originally)\b)|"
    r"this (?:song|track|album|record|single|lp|ep|one|artist|band|group|act)|"
    r"that (?:song|track|album|record|single|lp|ep|one|artist|band|group|act)|"
    r"the (?:song|track|album|record|artist|band|group|act|singer|vocalist)\b(?!\s+[A-Z])|"
    r"(?:more|tell me more|what else|anything else) about (?:them|it|the (?:album|song|track|band|artist|group)))",
    re.IGNORECASE,
)


_WIKI_ANAPHORA_RE = re.compile(
    r"\b(it|its|it's|they|their|theirs|them|"
    # Singular personal pronouns. Without these, "When did she first start out
    # as a musician?" after a Name question read as a self-contained
    # question: continuity never fired, the subject was lost, and the weak
    # keyword fallback ("start musician") searched alone — returning unrelated
    # musicians. "them"/"their" covered bands; a solo artist is "she"/"he".
    r"she|he|her|hers|him|his|"
    # Bare "this" used as a pronoun for the loaded recording: "who produced
    # this?", "when was this released?", "who wrote this". These are among
    # the commonest background questions, and without this the loaded track
    # was dropped from the search and only the word "produced" was looked up.
    r"this(?=\s*[?.!,;]|\s*$|\s+(?:is|was|were|come|came|get|got|released|recorded|written|"
    r"made|produced|sound|sounds|feel|feels|chart|charted|about|from|by|originally)\b)|"
    r"this (?:song|track|album|record|single|lp|ep|one|artist|band|group|act)|"
    r"that (?:song|track|album|record|single|lp|ep|one|artist|band|group|act)|"
    r"the (?:song|track|album|record|artist|band|group|act|singer|vocalist)\b(?!\s+[A-Z])|"
    r"(?:more|tell me more|what else|anything else) about (?:them|it|him|her|the (?:album|song|track|band|artist|group)))",
    re.IGNORECASE,
)


def _wiki_has_anaphora(question):
    """Whether the question refers back to something ('it', 'their',
    'this album') rather than naming its own subject — the strongest signal
    that this is a follow-up needing continuity from recent history rather
    than a question that stands on its own."""
    return bool(question and _WIKI_ANAPHORA_RE.search(question))


def _wiki_is_track_referential(question):
    """True when the question is about the currently discussed track / artist /
    album without introducing a new named subject of its own.

    Used to keep track-identity wiki queries (artist, album, title from file
    tags) at the front of the search list instead of letting noisy entities
    scraped from analysis dumps or incidental capitalized words dominate.
    """
    if not question:
        return False
        
    # [PATCH] Image-referential override:
    # If the user is explicitly referring to a newly uploaded image/photo, 
    # or asking a generic identity question (common with image drops), 
    # break the history inheritance so we don't inject the previous 
    # track's Wikipedia page and cause an observation hallucination.
    q_lower = question.lower()
    image_signals = {"image", "photo", "picture", "pic", "cover"}
    words = set(re.findall(r"[a-z0-9']+", q_lower))
    
    if words & image_signals or re.search(r"\bwho\s+(is|are|was|'s)\s+this\b", q_lower):
        return False

    entities = _wiki_question_entity_queries(question, max_entities=5)
    # Any solid named entity in the question (multi-word phrase OR a reasonably
    # long single proper noun like "Band R" / "Band B") means the user is
    # naming a subject themselves — do not force loaded-track identity first.
    # Anaphora still wins only when those entities are absent.
    solid = [
        e for e in entities
        if ((" " in e and len(e) >= 5) or (len(e) >= 4 and e[:1].isupper()))
    ]
    if solid:
        return False
    if _wiki_has_anaphora(question):
        return True
    focus = _wiki_question_focus(question)
    # Album/artist-focused questions with no external named subject are about
    # the loaded material ("tell me about the album", "who are they").
    if focus in ("song", "tracklist", "reception", "album", "debut"):
        return True
    # Short bare follow-ups with no entities at all.
    if not entities and len(question.split()) <= 12:
        return True
    return False


# A single, specific song/track/single, referred to with a determiner
# ("this song", "that track", "the single") -- as opposed to a PLURAL or
# listing reference ("these songs", "what songs are on...", "tracklist",
# "discography"), which must keep routing to "tracklist" exactly as before.
# Word-boundary matching already excludes "songs"/"tracks"/"singles" (the
# boundary after "song" fails to match inside "songs"), so no plural can
# match this pattern.
_WIKI_SONG_SINGULAR_RE = re.compile(r"\b(?:this|that|the)\s+(?:song|track|single)\b", re.IGNORECASE)


def _wiki_is_singular_song_question(q, words):
    """True only for the CLEANEST single-song questions: a singular
    song/track/single reference with no other tracklist/reception/album
    signal word also present. The moment any other signal is present, this
    deliberately returns False so the existing (already-established)
    tracklist/reception/album classification for that compound/ambiguous
    question is completely unaffected -- this only ever pulls NEW territory
    away from the catch-all that "song" alone used to fall into, never
    territory another category was already claiming."""
    if not _WIKI_SONG_SINGULAR_RE.search(q):
        return False
    other_tracklist_words = words & (_WIKI_TRACKLIST_SIGNAL - {"song", "songs", "track", "tracks"})
    if other_tracklist_words or (words & _WIKI_RECEPTION_SIGNAL) or (words & _WIKI_ALBUM_SIGNAL_WORDS):
        return False
    return True


def _wiki_question_focus(question):
    """Coarse focus tag for the question: song | tracklist | reception | debut | album | general.
    Used to prefer the right Wikipedia page and the right sections inside it."""
    if not question:
        return "general"
    q = question.lower()
    words = set(re.findall(r"[a-z0-9']+", q))
    if any(s in q for s in _WIKI_DEBUT_SIGNAL) or ("debut" in words):
        return "debut"
    if _wiki_is_singular_song_question(q, words):
        return "song"
    if words & _WIKI_TRACKLIST_SIGNAL:
        return "tracklist"
    if words & _WIKI_RECEPTION_SIGNAL:
        return "reception"
    if words & _WIKI_ALBUM_SIGNAL_WORDS or _wiki_mentions_album(question):
        return "album"
    return "general"


def _wiki_split_sections(text):
    """Split a Wikipedia plain-text dump into (heading, body) pairs.

    Handles common export styles: '== Heading ==', 'Heading\\n----', or a
    line that looks like a section title (short, title-ish) followed by body.
    Always yields a leading ('', lead_text) chunk for the article intro.
    """
    if not text:
        return [("", "")]

    # Prefer explicit wiki heading markers if present
    parts = re.split(r"(?m)^(?:={2,}\s*([^=\n]{2,80}?)\s*={2,})\s*$", text)
    if len(parts) > 1:
        sections = [("", parts[0])]
        i = 1
        while i + 1 < len(parts):
            sections.append((parts[i].strip(), parts[i + 1]))
            i += 2
        return sections

    # Fallback: scan for well-known section title lines
    known = (
        r"Track\s*listing|Track\s*list|Tracklist|Personnel|Credits|"
        r"Critical\s+reception|Reception|Critical\s+response|Reviews?|"
        r"Commercial\s+performance|Charts?|Singles?|Release|"
        r"Background|Recording|Production|Composition|Legacy|"
        r"Discography|Studio\s+albums|Album\s+list|Selected\s+discography|"
        r"Compilation\s+albums|Live\s+albums|EPs?"
    )
    pattern = re.compile(
        rf"(?im)(?:^|\n)(?:={0,6}\s*)({known})(?:\s*={0,6})?\s*(?:\n|$)"
    )
    matches = list(pattern.finditer(text))
    if not matches:
        return [("", text)]

    sections = []
    if matches[0].start() > 0:
        sections.append(("", text[: matches[0].start()]))
    for i, m in enumerate(matches):
        heading = m.group(1).strip()
        body_start = m.end()
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sections.append((heading, text[body_start:body_end]))
    return sections


def _wiki_section_relevance(heading, body, focus):
    """Score how useful a section is for the current question focus."""
    h = (heading or "").lower()
    b = (body or "")[:500].lower()
    score = 0

    tracklist_h = ("track", "listing", "tracklist", "side a", "side b", "personnel", "credits")
    reception_h = ("reception", "review", "critical", "commercial", "chart", "legacy", "acclaim")
    disco_h = ("discography", "studio album", "album list", "selected discography", "compilation", "live album", "eps")
    background_h = ("background", "recording", "production", "composition", "release", "single")
    # Headings typical of a single SONG's own article (as opposed to an
    # album/artist page) that none of the buckets above already cover.
    # Used both by the dedicated "song" focus branch below and, at a lower
    # weight, by the "general" catch-all -- additive only, so it can add
    # points to a section that previously scored 0 but never changes the
    # score of anything the existing buckets above already recognize.
    song_h = ("music video", "chart performance", "certification",
              "cover versions", "cover version", "live performances", "b-side")

    def hit(words):
        return any(w in h for w in words) or any(w in b[:120] for w in words)

    if focus == "tracklist":
        if hit(tracklist_h):
            score += 10
        if hit(disco_h):
            score += 8
        if hit(reception_h):
            score += 3
        if hit(background_h):
            score += 2
    elif focus == "reception":
        if hit(reception_h):
            score += 10
        if hit(tracklist_h):
            score += 3
        if hit(background_h):
            score += 2
    elif focus == "debut":
        if hit(background_h) or hit(tracklist_h) or hit(reception_h):
            score += 6
        if hit(disco_h):
            score += 4
    elif focus == "album":
        if hit(tracklist_h) or hit(reception_h) or hit(background_h):
            score += 6
    elif focus == "song":
        # A single song's own article: composition/background and reception
        # are the two things most worth surfacing, with song-specific
        # headings (chart performance, music video, certifications) as a
        # strong signal in their own right. Tracklist/discography headings
        # are deliberately NOT boosted here -- those belong to the album's
        # article, not this song's.
        if hit(song_h):
            score += 8
        if hit(background_h):
            score += 6
        if hit(reception_h):
            score += 5
    else:
        if hit(disco_h):
            score += 5
        if hit(background_h) or hit(reception_h):
            score += 3
        if hit(song_h):
            score += 3

    # Numbered track-like content even without a clean heading
    if focus in ("tracklist", "album", "debut") and re.search(
        r"(?m)^\s*(?:\d{1,2}[\).\:]|Track\s*\d)", body[:1500]
    ):
        score += 7

    # Year + album title patterns typical of discography tables
    if focus in ("tracklist", "general", "debut") and re.search(
        r"\b(?:19|20)\d{2}\b.{0,40}\b(?:album|ep|lp|studio)\b", body[:1500], re.I
    ):
        if hit(disco_h) or "discog" in h:
            score += 5

    return score


def _wiki_pick_relevant_snippet(raw_text, question, max_chars):
    """Prefer deep article sections (Track listing, Discography, Reception)
    over only the lead. Wikipedia often puts these further down the page;
    without them the writer model tends to invent track lists from memory.
    """
    if not raw_text:
        return ""

    text = str(raw_text)
    focus = _wiki_question_focus(question)
    sections = _wiki_split_sections(text)

    # Always keep a short lead for identity / release context
    lead_budget = min(700, max_chars // 4) if focus in ("tracklist", "reception", "debut", "album", "song") else min(900, max_chars // 2)
    lead_text = " ".join((sections[0][1] if sections else text).split())
    if len(lead_text) > lead_budget:
        lead_text = lead_text[:lead_budget].rsplit(" ", 1)[0] + "…"

    scored = []
    for heading, body in sections[1:] if len(sections) > 1 else []:
        body_clean = body.strip()
        if len(body_clean) < 40:
            continue
        sc = _wiki_section_relevance(heading, body_clean, focus)
        if sc <= 0 and focus not in ("general",):
            continue
        scored.append((sc, heading, body_clean))

    # If section split failed, fall back to regex windows over the full text
    if not scored:
        fallback_pats = []
        if focus in ("tracklist", "album", "debut", "general"):
            fallback_pats += [
                r"(?i)\b(?:track\s*listing|track\s*list|tracklist|discography|studio\s+albums?)\b",
                r"(?i)(?:^|\n)\s*1[\).\:]\s+[\"'A-Za-z]",
            ]
        if focus in ("reception", "album", "debut", "general", "song"):
            fallback_pats += [
                r"(?i)\b(?:critical\s+reception|critical\s+response|commercial\s+performance)\b",
            ]
        for pat in fallback_pats:
            for m in re.finditer(pat, text):
                start = max(0, m.start() - 20)
                end = min(len(text), m.start() + max(1200, max_chars // 2))
                chunk = text[start:end]
                scored.append((6, "match", chunk))
                if len(scored) >= 4:
                    break
            if len(scored) >= 4:
                break

    scored.sort(key=lambda x: -x[0])

    pieces = []
    total = 0
    if lead_text:
        pieces.append(lead_text)
        total += len(lead_text)

    seen_words = set(re.findall(r"[a-z0-9]{4,}", lead_text.lower()))
    for sc, heading, body in scored:
        body_one = " ".join(body.split())
        # Per-section budget: give tracklist/discography more room
        sec_cap = max_chars - total
        if sec_cap < 150:
            break
        if focus == "tracklist" and sc >= 7:
            take = min(len(body_one), max(sec_cap, min(1800, max_chars // 2)))
        elif focus == "reception" and sc >= 7:
            take = min(len(body_one), max(sec_cap, min(1400, max_chars // 2)))
        else:
            take = min(len(body_one), sec_cap)

        # Avoid near-duplicate sections
        words = set(re.findall(r"[a-z0-9]{4,}", body_one[:800].lower()))
        if words and len(words & seen_words) >= max(4, len(words) // 2):
            continue
        seen_words |= words

        chunk = body_one[:take]
        if len(body_one) > take:
            chunk = chunk.rsplit(" ", 1)[0] + "…"
        label = f"[{heading}] " if heading and heading != "match" else ""
        pieces.append(label + chunk)
        total += len(chunk)
        if total >= max_chars:
            break

    if len(pieces) == 1 and not scored:
        # Pure lead fallback
        full = " ".join(text.split())
        if len(full) > max_chars:
            return full[:max_chars].rsplit(" ", 1)[0] + "…"
        return full

    return "\n\n".join(pieces)


# Punctuation variants that are visually/semantically the same character but
# encoded differently -- curly vs straight quotes, en/em dash vs a plain
# hyphen, a prime mark used in place of an apostrophe. Wikipedia titles are
# not consistent about which form they use, so without folding these two
# otherwise-identical titles can fail to compare equal here, the same gap
# that diacritic-folding already closes for accents. Deliberately narrow:
# this only merges unambiguous character-encoding variants of the same
# punctuation mark. It never substitutes semantically different words (e.g.
# "&" is NOT folded to "and" here) -- that would risk two genuinely
# different titles being treated as the same one.
_WIKI_TITLE_PUNCT_FOLD_MAP = str.maketrans({
    "\u2018": "'", "\u2019": "'", "\u201a": "'", "\u2032": "'",
    "\u201c": '"', "\u201d": '"', "\u201e": '"', "\u2033": '"',
    "\u2013": "-", "\u2014": "-", "\u2212": "-",
})


def _wiki_title_compare_key(title):
    """Canonical comparison key for Wikipedia titles during *exact-title* lookup.

    This is intentionally only used as a fallback after the FTS index has already
    narrowed candidates. It does not alter stored titles or ordinary ranking.
    NFKC handles compatibility punctuation/width, casefold handles Unicode case,
    diacritic-folding handles accents, punctuation-folding handles quote/dash
    variants, and whitespace collapsing avoids harmless spacing differences.
    """
    if not title:
        return ""
    text = unicodedata.normalize("NFKC", str(title))
    text = _wiki_fold_diacritics(text)
    text = text.translate(_WIKI_TITLE_PUNCT_FOLD_MAP)
    text = re.sub(r"\s+", " ", text).strip()
    return text.casefold()


def _wiki_rank_prefer_album_title(rows, query_hint=""):
    """Among FTS hits, prefer titles that actually look like the thing being
    asked about — an exact/near match on the query, or an '(album)'/'(EP)'
    disambiguator — over pages that just happen to mention the artist
    somewhere in the body (a member's solo project, a video, a related
    label). Most real album titles on Wikipedia are NOT disambiguated
    ('Word Title', not 'Word Title (album)') — only ambiguous ones are — so
    this can't rely on the '(album)' suffix alone; it also has to reward a
    close title match on its own.

    Diacritics are folded for comparison so a query of "Name" ranks the
    title "Name" as an exact match above "List of songs recorded by Name".
    """
    if not rows:
        return rows

    def score(title):
        t = (title or "").strip()
        tl = t.lower()
        tl_fold = _wiki_fold_diacritics(tl).lower()
        s = 0
        if "(album)" in tl or "(ep)" in tl or "(lp)" in tl:
            s += 5
        # Same treatment for song disambiguators -- "(song)"/"(single)" pages
        # were previously left out of this bonus even though they're exactly
        # the same kind of legitimate disambiguated title as "(album)"/"(ep)".
        if "(song)" in tl or "(single)" in tl:
            s += 5
        if query_hint:
            qh = query_hint.lower()
            qh_fold = _wiki_fold_diacritics(qh).lower()
            # Exact title match (diacritic-insensitive) is the strongest signal —
            # this is what makes "Name" prefer the page "Name" over list pages.
            if tl == qh or tl_fold == qh_fold:
                s += 20
            elif tl_fold.startswith(qh_fold + " (") or tl_fold.startswith(qh_fold + ","):
                # "Band R (band)", "Name (singer)" style disambiguators
                s += 14
            elif qh_fold == tl_fold.split("(")[0].strip():
                s += 12
            elif qh_fold in tl_fold or tl_fold in qh_fold:
                # Substring match only — much weaker than exact title
                s += 3
            # Prefer titles that contain most/all of the query's tokens —
            # this is what actually catches unadorned album titles like
            # "Word Title" for a query built from that same phrase.
            tokens = [w for w in re.findall(r"[a-z0-9]+", qh_fold) if len(w) > 2]
            if tokens:
                hits = sum(1 for tok in tokens if tok in tl_fold)
                s += hits
                if tokens and hits == len(tokens):
                    s += 3
        # Pages that are ABOUT something else — a member's side project, a
        # video, a specific song/single, a label — but merely mention the
        # queried artist/album in passing are exactly the false positives
        # this function exists to suppress.
        if re.search(r"\([^)]*\b(?:video|film|song|single|record label)\b[^)]*\)", tl):
            s -= 4
        if tl.endswith(" (band)") or tl.endswith(" (musician)") or tl.endswith(" (singer)"):
            # Mild penalty only when we already have a stronger exact undambiguated
            # title candidate; the disambiguated form is still a real artist page.
            s -= 1
        # "List of songs recorded by X" / "List of awards..." crowd out the
        # actual artist page for short name queries — demote hard.
        if re.match(r"^list of\b", tl) or tl.startswith("lists of "):
            s -= 15
        if "discography" in tl and "(" not in tl:
            # Bare "X discography" is useful; still below the main bio for
            # "tell me about X" but above random list pages.
            s -= 2
        return s

    return sorted(rows, key=lambda r: score(r["t"] if isinstance(r, dict) else r[0]), reverse=True)


def _wiki_multi_search(queries, max_articles=None, prefer_album=False):
    """Run several independent searches (each a string of search terms) and
    collect up to max_articles DISTINCT articles, deduped by title, in the
    order the queries are given. This is what lets one request pull in the
    song article AND the album article (with its track listing) AND the
    artist article together, instead of stopping at a single best guess."""
    conn = _get_wiki_db()
    if conn is None:
        return []

    max_articles = WIKI_MAX_ARTICLES if max_articles is None else max_articles
    schema = _WIKI_DB_SCHEMA
    table = _wiki_quote_ident(schema["table"])
    title_col = _wiki_quote_ident(schema["title_col"])
    text_col = _wiki_quote_ident(schema["text_col"])
    cur = conn.cursor()

    # Weight the title column much more heavily than body text in FTS
    # ranking. Without this, FTS5's default `rank` (bm25 across all indexed
    # columns equally) treats a body mention the same as a title match — so
    # a short, dense page that happens to name-check the artist a few times
    # (a member's solo project, a video, a related label) can out-rank the
    # actual artist/album page, where the same terms appear less densely
    # relative to a much longer article. Boosting the title column fixes
    # that without needing to know anything about the DB's specific schema.
    _bm25_order = "rank"
    if schema.get("columns"):
        weights = [
            "10.0" if c == schema["title_col"] else "1.0"
            for c in schema["columns"]
        ]
        _bm25_order = f"bm25({table}, {', '.join(weights)})"

    seen_titles = set()
    results = []

    # Do not let the first successful query monopolise the whole result set.
    # Previously an artist/album query could return three highly-ranked pages
    # and prevent later queries (song, album, artist, question entity) from
    # ever contributing. After a few turns this looked like a stuck cache
    # because the same three articles kept winning every request.
    # Reserve room for multiple query families, then use the remaining slots
    # for the strongest matches.
    per_query_budget = max(1, min(2, max_articles // 2))

    # TWO PASSES. Pass 1 gives every query at most ONE article; pass 2 tops the
    # strongest up to per_query_budget with whatever slots remain.
    #
    # A single pass let one subject take several slots before a later, equally
    # important query ran at all. Asking to compare two people put four of six
    # articles on the first name -- its page, its discography, and two weak
    # name-fragment matches -- so the second person's page never made it in and
    # the answer was written without it. Round-robin guarantees each distinct
    # query is represented before any query gets a second article, which is the
    # same fair-share principle already applied across tracks in /loadcompare.
    # Only MULTI-WORD queries take part in the fairness pass. A bare single
    # word ("Name", "Lastname") is a name fragment that matches disambiguation
    # and surname pages, and guaranteeing those a slot each would crowd out the
    # real subjects -- trading one failure for another. Weak queries keep their
    # existing last-priority role: they fill leftover slots only.
    # Budget is counted per DISTINCT query. The same query is revisited by up
    # to three passes below; counting every visit exhausted the budget on
    # repeats, so later entities never got an exact-title lookup and fell back
    # to fuzzy ranking. Repeat visits are served from _WIKI_EXACT_TITLE_CACHE.
    exact_lookup_queries = set()
    query_list = [q for q in queries if q and str(q).strip()]
    strong_list = [q for q in query_list if " " in str(q).strip()]
    added_by_query = {}

    for _pass_queries, pass_budget in (
        (strong_list, 1),                 # every strong entity gets one slot
        (strong_list, per_query_budget),  # then top them up
        (query_list, per_query_budget),   # then weak queries fill what is left
    ):
      for q in _pass_queries:
        if len(results) >= max_articles:
            break
        already = added_by_query.get(q, 0)
        if already >= pass_budget:
            continue

        rows = []
        exact_rows = []
        # EXACT-TITLE PASS. Settles the canonical article deterministically
        # instead of hoping it survives bm25 ranking (see WIKI_EXACT_TITLE_LOOKUP
        # for the measured ranks that make ranking unreliable here). Anything
        # found is put at the FRONT of rows; the re-ranker then scores it +20 for
        # an exact title match, so it leads the results for this query.
        #
        # Capped per request and fully isolated: any failure leaves rows empty
        # and the FTS path below runs exactly as it did before.
        if (WIKI_EXACT_TITLE_LOOKUP
                and (q in exact_lookup_queries
                     or len(exact_lookup_queries) < WIKI_EXACT_TITLE_MAX_PER_REQUEST)):
            exact_lookup_queries.add(q)
            try:
                q_exact = " ".join(_wiki_strip_possessive(w) for w in str(q).split())
                folded = _wiki_fold_diacritics(q_exact)
                bases = [q_exact]
                if folded and folded != q_exact:
                    bases.append(folded)

                cands = []
                if prefer_album:
                    # Album disambiguators go FIRST when an album is wanted.
                    # Eponymous releases make the order decisive: for "What was
                    # Name's debut album like?" the query is "Name", and
                    # the bare title matches the ARTIST article. Since the loop
                    # below stops at the first hit, trying the bare title first
                    # meant "Name (album)" was never reached and an album
                    # question silently got the biography instead. Same for
                    # Band W, Band, Band and every other self-titled
                    # record.
                    #
                    # Costs nothing when the album is not eponymous: "Title
                    # (album)" simply misses and the bare title is tried next.
                    for b in bases:
                        cands.extend([f"{b} (album)", f"{b} (EP)"])
                # Band-name title forms, tried BEFORE the bare name.
                #
                # Entity extraction drops a leading "The" ("What about The
                # Band Z?" -> "Band Z") while the article is titled "Band Z",
                # so without these the lookup missed and FTS stemming matched
                # "Name (surname)" instead.
                #
                # "(band)" comes first because in a music conversation it is
                # almost always the wanted article, whereas the bare title is
                # frequently a disambiguation or given-name page. "Name" is the
                # clear case: the band is at "Name (band)", but a plain "Name"
                # article exists and was matching first, so the band page was
                # never reached. Costs nothing when no such page exists —
                # "Name (band)" simply misses and the bare title is next.
                # Preserve the original exact-title candidate order first.
                # Role-disambiguated music pages are an additive fallback only,
                # so an existing successful lookup cannot be displaced.
                for b in list(bases):
                    cands.append(f"{b} (band)")
                    if not b.lower().startswith("the "):
                        cands.append(f"The {b}")
                cands.extend(bases)
                for b in list(bases):
                    cands.extend([
                        f"{b} (musician)",
                        f"{b} (singer)",
                        f"{b} (rapper)",
                        f"{b} (group)",
                    ])
                # Song/single disambiguator forms. Individual song/track
                # Wikipedia articles are frequently titled "Title (song)" or
                # "Title (single)" when the bare title collides with
                # something unrelated (a film, another act's song, an
                # ordinary phrase). Tried LAST, after every artist/band-
                # oriented form above, since those are more likely to be
                # what a bare capitalized query actually refers to in a
                # music conversation -- this is strictly an additional
                # fallback and, like the role forms above it, cannot
                # displace an earlier hit.
                for b in list(bases):
                    cands.extend([
                        f"{b} (song)",
                        f"{b} (single)",
                    ])

                # Collect up to WIKI_EXACT_TITLE_MAX_HITS distinct titles rather
                # than stopping at the first. "What are the best Band Z songs?"
                # sets prefer_album, so "Band Z (album)" was tried first, matched
                # a compilation, and the loop stopped -- the band article "The
                # Band Z" was never reached. Both readings of a query are
                # legitimate and there is room for both in the article budget,
                # so take the album form AND the artist/band form.
                exact_titles_seen = set()
                # Two hits only when an ALBUM is wanted, where both readings are
                # useful ("best Band Z songs" -> the compilation AND the band).
                # For a plain artist question one article is right, and taking a
                # second actively hurt: for "Name" the lookup found "Name
                # (band)" first, but then also took the bare "Name" page, which
                # the re-ranker scores higher (+20 exact vs +14 prefix). The
                # band page was pushed out by a given-name/surname page.
                max_exact_hits = WIKI_EXACT_TITLE_MAX_HITS if prefer_album else 1

                for cand in cands:
                    if len(exact_titles_seen) >= max_exact_hits:
                        break
                    key = _wiki_title_compare_key(cand)
                    if key in _WIKI_EXACT_TITLE_CACHE:
                        hit = _WIKI_EXACT_TITLE_CACHE[key]
                        if hit:
                            ht = str(hit[0][0] or "").casefold()
                            if ht not in exact_titles_seen:
                                exact_titles_seen.add(ht)
                                # Must match the uncached branch below: exact
                                # hits go to exact_rows, never rows, or a
                                # SECOND question in the same session (once the
                                # cache is warm) puts the hit back in the pool
                                # that gets re-ranked and demoted.
                                exact_rows.extend(hit)
                        continue

                    # Find the row via the FTS INDEX, restricting the match to
                    # the title column, then confirm exactness in Python.
                    #
                    # "WHERE title = ?" on an FTS5 virtual table is a full scan
                    # of the whole corpus -- FTS5 keeps no ordinary index on its
                    # columns, so an index elsewhere in the database does not
                    # help it. On a full music wiki that scan takes seconds, and
                    # doing it several times per message is the delay this
                    # replaces. A title-column MATCH uses the same inverted
                    # index the body search already uses, so it is fast.
                    hit = []
                    try:
                        cand_fts = _wiki_fts_query_escape(cand)
                        if cand_fts:
                            # MATCH narrows to candidate rows through the FTS
                            # index; the equality test then runs only over those
                            # few rows. Ordering is deliberately NOT used to pick
                            # the winner -- a title-restricted MATCH still ranks
                            # by bm25 over the whole row, so the long canonical
                            # article loses to short list pages exactly as it
                            # does in the body search. Asking SQL for the exact
                            # title sidesteps ranking entirely.
                            cur.execute(
                                f"SELECT {title_col} AS t, {text_col} AS b FROM {table} "
                                f"WHERE {table} MATCH ? AND {title_col} = ? COLLATE NOCASE "
                                f"LIMIT 1",
                                (f'{schema["title_col"]} : ({cand_fts})', cand),
                            )
                            hit = cur.fetchall()

                            # SQLite's built-in NOCASE collation is ASCII-centric.
                            # A title such as "Name" therefore still misses an
                            # input such as "CAFÉ" even though the FTS index can
                            # narrow to the right title. As a fallback ONLY, compare
                            # the already-indexed title candidates in Python using
                            # Unicode-aware casefolding + diacritic folding.
                            if not hit:
                                cur.execute(
                                    f"SELECT {title_col} AS t, {text_col} AS b FROM {table} "
                                    f"WHERE {table} MATCH ? LIMIT 32",
                                    (f'{schema["title_col"]} : ({cand_fts})',),
                                )
                                target_key = _wiki_title_compare_key(cand)
                                for candidate_row in cur.fetchall():
                                    candidate_title = candidate_row["t"]
                                    if _wiki_title_compare_key(candidate_title) == target_key:
                                        hit = [candidate_row]
                                        break

                            # Optional redirect/alias table (e.g. "Nickname"
                            # -> "The Group Name", a former band name, a common
                            # misspelling). _wiki_resolve_redirect returns
                            # None immediately -- no query executed at all --
                            # whenever the DB has no such table, so this is a
                            # genuine no-op for every database that predates
                            # this feature, exactly as if this block were
                            # not here.
                            if not hit:
                                redirect_target = _wiki_resolve_redirect(conn, schema, cand)
                                if redirect_target:
                                    cur.execute(
                                        f"SELECT {title_col} AS t, {text_col} AS b FROM {table} "
                                        f"WHERE {title_col} = ? COLLATE NOCASE LIMIT 1",
                                        (redirect_target,),
                                    )
                                    hit = cur.fetchall()
                    except Exception:
                        hit = []

                    _WIKI_EXACT_TITLE_CACHE[key] = hit
                    if hit:
                        ht = str(hit[0][0] or "").casefold()
                        if ht not in exact_titles_seen:
                            exact_titles_seen.add(ht)
                            # Kept SEPARATE from `rows`, not appended to it.
                            # The FTS body search below always runs afterward
                            # and its results get re-ranked with
                            # _wiki_rank_prefer_album_title -- which scores an
                            # EXACT string match to the query at +20, higher
                            # than a "Name (...)" PREFIX match at +14. If the
                            # exact-title hit were mixed into `rows` before that
                            # re-rank, the bare "Name" page (an exact match to
                            # the query "Name") would jump back ahead of
                            # "Name (band)" (the exact-title hit we just found
                            # on purpose) and undo the whole point of this pass.
                            # exact_rows is merged back in AFTER re-ranking,
                            # always in front, so it cannot be demoted.
                            exact_rows.extend(hit)
            except Exception:
                rows = []
                exact_rows = []

        try:
            if schema["mode"] == "fts":
                fts_q = _wiki_fts_query_escape(q)
                if not fts_q:
                    # No usable FTS query — but an exact-title hit may already
                    # be in `exact_rows`, so only abandon this query if both
                    # are empty (exact_rows moved out of `rows` above).
                    if not rows and not exact_rows:
                        continue
                    fts_variants = []
                else:
                    # When we want an album page, also try an explicit "(album)" disambiguator
                    fts_variants = [fts_q]
                    if prefer_album and "(album)" not in q.lower():
                        fts_variants.append(_wiki_fts_query_escape(f"{q} album"))
                for fv in fts_variants:
                    try:
                        cur.execute(
                            f"SELECT {title_col} AS t, {text_col} AS b FROM {table} "
                            f"WHERE {table} MATCH ? ORDER BY {_bm25_order} LIMIT ?",
                            (fv, WIKI_SEARCH_ROW_LIMIT),
                        )
                        batch = cur.fetchall()
                    except Exception:
                        # Title-weighted bm25() can fail on some FTS5 configs
                        # (e.g. contentless tables where column weighting
                        # isn't supported the same way) — fall back to the
                        # plain default rank rather than losing the query.
                        cur.execute(
                            f"SELECT {title_col} AS t, {text_col} AS b FROM {table} "
                            f"WHERE {table} MATCH ? ORDER BY rank LIMIT ?",
                            (fv, WIKI_SEARCH_ROW_LIMIT),
                        )
                        batch = cur.fetchall()
                    if batch:
                        rows.extend(batch)
                # Always re-rank so exact title matches (e.g. "Name" for query
                # "Name") beat list/discography pages that merely contain the name.
                if rows:
                    rows = _wiki_rank_prefer_album_title(rows, query_hint=q)
                # Exact-title hits go BACK in front here, after re-ranking, so
                # nothing in the body-search results can outrank them. Deduped
                # against `rows` by title so the same article is not listed
                # twice for this one query.
                if exact_rows:
                    exact_keys = set()
                    for er in exact_rows:
                        et = er["t"] if hasattr(er, "keys") else er[0]
                        exact_keys.add(str(et or "").strip().casefold())
                    rows = exact_rows + [
                        r for r in rows
                        if str((r["t"] if hasattr(r, "keys") else r[0]) or "")
                            .strip().casefold() not in exact_keys
                    ]
            else:
                # No FTS index: only cheap, targeted lookups — a full LIKE
                # scan over a multi-GB table with no index is too slow to
                # run repeatedly per request. Strip possessives here too,
                # word by word, so "Band R's" still exact/prefix-matches
                # a title of "Band R".
                q_clean = " ".join(_wiki_strip_possessive(w) for w in str(q).split())
                q_folded = _wiki_fold_diacritics(q_clean)
                candidates = [q_clean]
                if q_folded and q_folded != q_clean:
                    candidates.append(q_folded)
                if prefer_album:
                    for base in list(candidates):
                        candidates.extend([
                            f"{base} (album)",
                            f"{base} (EP)",
                        ])
                for cand in candidates:
                    cur.execute(
                        f"SELECT {title_col} AS t, {text_col} AS b FROM {table} "
                        f"WHERE {title_col} = ? COLLATE NOCASE LIMIT 1",
                        (cand,),
                    )
                    rows = cur.fetchall()
                    if rows:
                        break
                # Same optional redirect/alias table used by the FTS branch
                # above. Tried before the LIKE prefix scan below since an
                # exact redirect target is a far higher-precision answer than
                # a guessed prefix match. A genuine no-op when the DB has no
                # such table (_wiki_resolve_redirect returns None immediately).
                if not rows:
                    for cand in candidates:
                        redirect_target = _wiki_resolve_redirect(conn, schema, cand)
                        if redirect_target:
                            cur.execute(
                                f"SELECT {title_col} AS t, {text_col} AS b FROM {table} "
                                f"WHERE {title_col} = ? COLLATE NOCASE LIMIT 1",
                                (redirect_target,),
                            )
                            rows = cur.fetchall()
                            if rows:
                                break
                if not rows:
                    # Prefix search on both original and folded forms
                    like_rows = []
                    for base in (q_clean, q_folded):
                        if not base:
                            continue
                        cur.execute(
                            f"SELECT {title_col} AS t, {text_col} AS b FROM {table} "
                            f"WHERE {title_col} LIKE ? COLLATE NOCASE LIMIT 5",
                            (f"{base}%",),
                        )
                        like_rows.extend(cur.fetchall())
                    rows = like_rows
                    if rows:
                        rows = _wiki_rank_prefer_album_title(rows, query_hint=q_clean)
        except Exception:
            continue

        added_this_query = added_by_query.get(q, 0)
        for row in rows:
            t, b = row["t"], row["b"]
            if not t or not b:
                continue
            dedup_key = str(t).strip().lower()
            if dedup_key in seen_titles:
                continue
            seen_titles.add(dedup_key)
            results.append((t, b))
            added_this_query += 1
            added_by_query[q] = added_this_query
            if len(results) >= max_articles or added_this_query >= pass_budget:
                break

    return results


def _wiki_clean_artist_for_search(artist):
    """Reduce a file-tag artist string to the primary act, for search queries.

    Tag artists carry credits that are poison as search terms. "Band X feat. Singer Y" (note the U+2024 one-dot-leader some taggers write
    instead of a full stop) produced the eight-word query "Band X feat. Singer Y Title", which matched no song article at all
    — instead it pulled pages that merely MENTION both, like the songwriter and
    the video director, and those consumed the whole article budget.

    Only featured-artist credits are stripped. "&" and "and" are left alone
    because they are part of many real band names.
    """
    if not artist:
        return ""
    a = str(artist)
    # Normalise unicode punctuation that taggers substitute for ASCII.
    for src, dst in (("\u2024", "."), ("\u2019", "'"), ("\u2018", "'"),
                     ("\u201c", '"'), ("\u201d", '"'), ("\uff0e", ".")):
        a = a.replace(src, dst)
    # "/" is only a credit separator when surrounded by spaces ("Artist A /
    # Artist B"). Bare, it is part of names like AB/CD, so splitting on it
    # unconditionally turned that into "AB".
    a = re.split(
        r"\s*(?:\bfeat\b\.?|\bft\b\.?|\bfeaturing\b|\bwith\b\s+[A-Z]|\bvs\b\.?|\s/\s)",
        a,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    return a.strip(" ,;-&")


def build_wiki_context_block_multi(artist=None, title=None, album=None, question=None, use_question_entities=True,
                                   extra_entities=None, priority_entities=None, persona_gate=False, _lookup_articles=None):
    """Return a WIKIPEDIA BACKGROUND CONTEXT block that can include MULTIPLE
    distinct articles — e.g. the song's own article, its album's article
    (with track listing / reception), and the artist's article — rather than
    a single best-guess match. Returns "" if unavailable/no confident match.

    Search queries are built from whichever of (artist, title, album) are
    given, plus — when use_question_entities is True — named-entity-like
    phrases pulled from `question` itself, so this also works for general
    questions ("what albums did Band R release in the 90s?") that aren't
    anchored to a specific loaded track.

    SCOPE: this is background/biographical + encyclopedic context only —
    formation, members, discography, TRACK LISTINGS, chart/critical
    reception, cultural history. It must never be used as a source for
    audio-technical facts (tempo, key, instrumentation, structure,
    production/mix qualities, vocal analysis); those come exclusively from
    the PRIVATE TRACK NOTES (MOSS-Music + Essentia + stem/MIDI), which
    are grounded in the actual audio. The rules text below enforces that
    separation for the writer model.

    persona_gate (optional): pass True whenever a custom roleplay persona is
    active. CUSTOM_PERSONA_RULES already tells the writer this block is
    production reference, not the character's memory, and to only recite it
    when the persona would plausibly know the subject — but that instruction
    alone has proven unreliable once the actual facts are sitting right there
    in context on a smaller local model. persona_gate adds a note directly
    beside the content itself, the same technique already used for the
    audio-evidence firewall, rather than relying on a rule stated once,
    earlier, and further away. It is independent of has_track_context below:
    a persona needs this reminder on a general "have you heard of X" question
    with nothing loaded just as much as it does with a track playing.
    """
    if not ENABLE_WIKI_CONTEXT:
        return ""
    if _get_wiki_db() is None:
        return ""

    focus = _wiki_question_focus(question)
    prefer_album = focus in ("tracklist", "reception", "album", "debut")

    # Per-article budget: give album-detail questions more room so track
    # listings and reception paragraphs survive truncation.
    per_article_cap = WIKI_CONTEXT_MAX_CHARS_PER_ARTICLE
    total_cap = WIKI_CONTEXT_TOTAL_MAX_CHARS
    if focus == "tracklist":
        # Track lists and discographies sit deep in articles and need room
        per_article_cap = max(per_article_cap, 3200)
        total_cap = max(total_cap, 7000)
    elif focus == "song":
        # A single song's article usually has a short lead but useful named
        # sections further down (Composition, Release, Chart performance) --
        # more room than the plain default, but less than a full tracklist
        # or album-detail question needs.
        per_article_cap = max(per_article_cap, 2000)
        total_cap = max(total_cap, 5500)
    elif prefer_album:
        per_article_cap = max(per_article_cap, 2400)
        total_cap = max(total_cap, 6000)

    # total_cap must be able to hold WIKI_MAX_ARTICLES articles at
    # per_article_cap each, or the block-building loop below will silently
    # drop articles _wiki_multi_search legitimately found (WIKI_CONTEXT_TOTAL_MAX_CHARS
    # was never updated when WIKI_MAX_ARTICLES was raised, so e.g. 5000 / 1800
    # only ever fit ~2-3 of the up-to-6 articles the search actually returns).
    total_cap = max(total_cap, per_article_cap * WIKI_MAX_ARTICLES)

    queries = []

    # Priority anchors run BEFORE anything else. _wiki_multi_search stops once
    # max_articles is reached, and later queries then never run at all — so on
    # /loadcompare the first track's album/title anchors filled all
    # WIKI_MAX_ARTICLES slots and the second track was never searched for. The
    # caller interleaves one anchor per track here so each loaded track gets a
    # turn before any track gets a second. Empty for single-track and
    # standalone questions, which keep the original ordering exactly.
    if priority_entities:
        for e in priority_entities:
            if e and e not in queries:
                queries.append(e)

    # Explicit album/title anchors first — highest precision
    if album:
        queries.append(f"{artist} {album}" if artist else album)
        queries.append(f"{album} (album)")
        if artist:
            queries.append(f"{artist} {album} (album)")
    if title:
        queries.append(f"{artist} {title}" if artist else title)
        # Song disambiguator forms, mirroring the album treatment above.
        # Individual song articles on Wikipedia are frequently titled
        # "Title (Artist song)" -- the actual disambiguation convention
        # Wikipedia itself uses -- when the bare title collides with
        # something unrelated (a film, another act's song, an ordinary
        # phrase). Without these, the exact-title pass in _wiki_multi_search
        # only ever saw the combined "artist title" string above and never
        # tried the real disambiguator, so a colliding title either missed
        # entirely or exact-matched the WRONG page and stopped there.
        # Purely additive: appended after the anchor query above, so it can
        # only add another chance to find/rank the right article, never
        # displace one that already worked.
        if artist:
            queries.append(f"{title} ({artist} song)")
        queries.append(f"{title} (song)")
        queries.append(f"{title} (single)")

    # Question entities. Split into "strong" (multi-word phrases — e.g. "Example
    # Song Title", much less likely to collide with an unrelated page) and
    # "weak" (bare single capitalized words — e.g. "Warner" from "the Warner
    # era" — which are common enough as label/company/place names to swamp
    # the result budget with irrelevant matches). Continuity entities carried
    # over from earlier turns are the best available anchor for what's
    # actually being discussed, so they're tried before either entity tier —
    # otherwise a single incidental capitalized word in a brand-new follow-up
    # question (e.g. "the Warner era") can fill every article slot before the
    # real subject of the conversation is ever searched for. See
    # _wiki_multi_search: each query can claim up to WIKI_MAX_ARTICLES slots,
    # and once max_articles is hit, later queries in the list never run.
    strong_q_entities, weak_q_entities = [], []
    if use_question_entities and question:
        q_entities = _wiki_question_entity_queries(question)
        for e in q_entities:
            (strong_q_entities if " " in e else weak_q_entities).append(e)
    else:
        q_entities = []

    if extra_entities:
        for e in extra_entities:
            if e:
                queries.append(e)
                if prefer_album:
                    queries.append(f"{e} (album)")

    queries.extend(strong_q_entities)
    if prefer_album:
        for e in strong_q_entities[:4]:
            queries.append(f"{e} (album)")
            queries.append(f"{e} album")

    if artist:
        if focus == "debut":
            # Target eponymous / debut album pages before the plain artist bio
            queries.insert(0, f"{artist} (album)")
            queries.insert(1, f"{artist} debut album")
            queries.insert(2, f"{artist} album")
            queries.append(artist)
        elif prefer_album:
            queries.append(f"{artist} album")
            queries.append(f"{artist} (album)")
            queries.append(artist)
        else:
            queries.append(artist)
            if album or (use_question_entities and _wiki_mentions_album(question)):
                queries.append(f"{artist} album")

    # Weak (single-word) question entities go last — lowest priority, only
    # reached if the stronger/continuity queries above didn't already fill
    # the article budget. Still worth trying since sometimes a single word
    # really is the whole subject (e.g. "Band R").
    queries.extend(weak_q_entities)
    if prefer_album:
        for e in weak_q_entities[:4]:
            queries.append(f"{e} (album)")
            queries.append(f"{e} album")

    if use_question_entities and question:
        kw = _wiki_question_keywords(question)
        if kw and artist:
            queries.append(f"{artist} " + " ".join(kw))
        # Debut without a resolved artist: still try keyword + album
        if focus == "debut" and kw:
            queries.append(" ".join(kw) + " album")
            queries.append(" ".join(kw) + " (album)")

    # Dedup queries while preserving order. Also add an ASCII-folded twin
    # (Name → Name) so a user who types without diacritics still hits
    # pages whose titles (or FTS index) use the accented form — or vice
    # versa when the local FTS tokenizer strips diacritics. Original form
    # is always kept first; folding never replaces it.
    seen_q = set()
    deduped_queries = []
    for q in queries:
        if not q:
            continue
        raw = str(q).strip()
        if not raw:
            continue
        candidates = [raw]
        folded = _wiki_fold_diacritics(raw)
        if folded and folded != raw:
            candidates.append(folded)
        for cand in candidates:
            k = cand.lower()
            if k and k not in seen_q:
                seen_q.add(k)
                deduped_queries.append(cand)
    queries = deduped_queries

    if not queries and _lookup_articles is None:
        return ""

    # IMPORTANT: query ordering depends on whether the user is still talking
    # about the loaded track vs. naming a new subject.
    #
    # - Track-referential / anaphoric questions ("tell me about the album",
    #   "tell me more about them", "what were its tracks?") MUST keep the
    #   permanent identity anchors (artist / album / title from file tags) at
    #   the front. Promoting noisy entities scraped from analysis dumps or
    #   incidental capitalized words was causing three unrelated FTS hits to
    #   fill the entire article budget.
    # - Questions that introduce their own strong named entities (e.g. "what
    #   albums did Band R release in the 90s?") promote those entities so
    #   track identity does not monopolise retrieval for an unrelated topic.
    if use_question_entities and question:
        track_ref = _wiki_is_track_referential(question)
        if track_ref:
            # Identity first; continuity extras next; question entities last
            # (and only the strong multi-word ones — weak single tokens from
            # anaphoric questions are almost always noise).
            ordered = []
            for q in queries:
                if q and q not in ordered:
                    ordered.append(q)
            # Continuity extras that aren't already identity anchors go right
            # after the existing identity block (they were appended earlier).
            for q in (extra_entities or []):
                if q and q not in ordered:
                    ordered.append(q)
            for q in strong_q_entities:
                if q and q not in ordered:
                    ordered.append(q)
            # Deliberately omit weak_q_entities for track-referential turns —
            # single capitalized words in "the Warner era"-style asides must
            # not displace the loaded artist/album.
            queries = ordered
        else:
            # Question-named subjects lead, then continuity extras, then the
            # rest. priority_entities must come first here: this branch rebuilt
            # the list as extras-then-question-entities, so with a track loaded
            # the track's identity anchors led every unrelated question and the
            # real subject ("that male-group test scan", "Firstname Lastname") sat behind them,
            # outside the article budget.
            intent_queries = []
            for q in (list(priority_entities or [])
                      + (extra_entities or []) + strong_q_entities + weak_q_entities):
                if q and q not in intent_queries:
                    intent_queries.append(q)
            if intent_queries:
                queries = intent_queries + [q for q in queries if q not in intent_queries]

    articles = (_lookup_articles if _lookup_articles is not None
                else _wiki_multi_search(queries, prefer_album=prefer_album))
    if not articles:
        return ""

    # Prefer album-titled articles first when the question is about an album
    if prefer_album:
        def _album_sort_key(item):
            t = (item[0] or "").lower()
            score = 0
            if "(album)" in t or "(ep)" in t:
                score -= 10
            # Boost titles that match any album/entity query token densely
            for q in queries[:6]:
                qt = q.lower().replace("(album)", "").strip()
                if qt and qt in t:
                    score -= 5
            return score
        articles = sorted(articles, key=_album_sort_key)

    sections = []
    total_len = 0
    for art_title, raw_text in articles:
        snippet = (raw_text if _lookup_articles is not None
                   else _wiki_pick_relevant_snippet(raw_text, question, per_article_cap))
        if not snippet:
            continue
        if total_len + len(snippet) > total_cap:
            remaining = total_cap - total_len
            if remaining < 200:
                break
            snippet = snippet[:remaining].rsplit(" ", 1)[0] + "…"
        sections.append(f'--- "{art_title}" ---\n{snippet}')
        total_len += len(snippet)
        if total_len >= total_cap:
            break

    if not sections:
        return ""

    body = "\n\n".join(sections)

    # The sonic firewall below exists to stop background text overriding
    # MEASURED audio. That premise only holds when a recording is actually in
    # play. On a general artist question with nothing loaded ("Who is X?", "How
    # did their sound evolve?") there are no measurements, so emitting it told
    # the model that instrumentation, arrangement, groove and production "are
    # MEASURED elsewhere in this conversation" when they were not — leaving it
    # forbidden to discuss sound from the only material it had, and visibly
    # degrading those answers.
    #
    # A track anchor (artist/title/album) is passed by every caller that has a
    # loaded recording, so it is a reliable signal for which note to use.
    has_track_context = bool(artist or title or album)
    if has_track_context:
        scope_note = (
            "SCOPE OF THIS BLOCK: WHO / WHEN / WHY ONLY.\n"
            "This block may be used for: who made the record, when it came out, what album "
            "it belongs to, the artist's history and line-up, chart performance, reception, "
            "awards, and cultural context.\n"
            "This block may NOT be used for: what the recording SOUNDS like. Tempo, key, "
            "chords, harmony, melody, song structure and section timings, instrumentation "
            "(including whether an instrument is present at all), arrangement, groove, "
            "vocal timbre/range/age/gender, mix, production, loudness or dynamics are "
            "MEASURED elsewhere in this conversation and are not open to revision from "
            "here. If this block describes the music and the measurements disagree, this "
            "block is simply wrong about this recording and must be ignored on that point. "
            "A genre or era named here does not license a sonic claim the measurements do "
            "not support.\n\n"
        )
    else:
        # NO scope note for general questions. This is what 0.3 did, and it is
        # what makes the assistant able to talk about anything the model knows.
        #
        # The firewall above exists solely to stop background text overriding
        # MEASURED audio; with no track loaded there is nothing to protect, and
        # every version of a "scope" preamble tried here made the assistant
        # rigid instead — it began reporting which articles it had been given
        # and refusing questions it could plainly answer, because a block
        # telling it what it may and may not use reads as a boundary however it
        # is phrased. The RULES text below is permissive and sufficient.
        scope_note = ""

    # Persona knowledge gate: independent of has_track_context above. Without
    # a track loaded there is no SOUND to protect, but there is still a
    # CHARACTER to protect from reciting production notes as their own memory
    # — exactly the "have you heard of this obscure band" case where no track
    # is playing at all, so has_track_context is False and scope_note above
    # is empty. This must stack with the audio scope note, not replace it,
    # since a persona can also be active while a track is loaded.
    if persona_gate:
        scope_note += (
            "PERSONA REMINDER: a custom roleplay persona is active this turn. The material "
            "below is production reference, not this character's memory. Before using ANYTHING "
            "in it as spoken fact, decide whether THIS persona would plausibly know it — do not "
            "recite chart positions, release dates, discography detail, or band history for an "
            "act they would not realistically have encountered, however accurate the fact is. "
            "See CUSTOM PERSONA — AWARENESS AND DEPTH LIMITS for the full rule.\n\n"
        )

    # Footer matches the scope note: the "nothing above describes how the
    # recording sounds" caveat only makes sense when a recording exists.
    end_note = (
        "=== END WIKIPEDIA BACKGROUND CONTEXT (nothing above this line describes how the "
        "recording sounds) ===\n"
        if has_track_context
        else "=== END WIKIPEDIA BACKGROUND CONTEXT ===\n"
    )

    return (
        f"\n\n=== WIKIPEDIA BACKGROUND CONTEXT (local reference DB — {len(sections)} article(s)) ===\n"
        f"{scope_note}"
        f"{body}\n"
        "RULES FOR THE ABOVE:\n"
        "- This is OPTIONAL background material from a local reference DB. Prefer it when it "
        "clearly covers what the user asked (artist/band history, formation, members, "
        "album/song release info, TRACK LISTINGS, chart performance, critical reception, "
        "awards, cultural context) — go into real depth from it when it has the details.\n"
        "- When the user asks for a track list, singles, or critical/commercial reception, "
        "prefer the details in these excerpts (including any Track listing / Reception / "
        "Critical response sections). Quote or paraphrase concrete track titles and review "
        "points when they are present — do not claim you lack them if they appear above.\n"
        "- NEVER use it as a source for tempo, BPM, key, chords, instrumentation, song "
        "structure, mix/production qualities, vocal analysis, or vocal age classification based on performer age or biography — those come exclusively "
        "from the PRIVATE TRACK NOTES (when present), which are grounded in the actual audio. "
        "Biographical information may still be used for factual context, but age information must not influence acoustic judgement. Vocalist age can be discussed in context. "
        "If this context and the track notes ever disagree on anything technical, the track "
        "notes are always right and this is ignored.\n"
        "- These may be excerpts from different articles (song / album / artist) — each is "
        "about the subject named in its heading, not about the others.\n"
        "- Put it in your own words rather than reciting long passages verbatim, and don't "
        "cite 'Wikipedia' or a database as your source — just answer naturally, as background "
        "you already knew.\n"
        "- If none of it actually matches what the user is asking about, ignore it entirely "
        "and answer from your normal general knowledge instead — but see the SPECIFIC "
        "FACTS rule below first.\n"
        "- QUALITATIVE FALLBACK ONLY: This block is a supplement, not a limit — but the "
        "fallback to your own knowledge is for QUALITATIVE discussion only: what an album "
        "or artist is like, influences, vibe, comparisons, general career arc/reputation. "
        "For that kind of discussion, where the excerpts are thin or off-topic, answer from "
        "your general musical knowledge as you would without this block.\n"
        "- SPECIFIC FACTS — DO NOT INVENT: Anything that is a specific, checkable claim — "
        "who/what a song samples or interpolates, songwriting/production/performer credits, "
        "chart positions, release dates, award wins, personnel changes, why a track was or "
        "wasn't included on a given release — must come from the excerpts above (or from "
        "knowledge you are genuinely highly confident in, the way you'd state a very famous, "
        "undisputed fact). If you are not sure, say so plainly or leave the detail out "
        "entirely. Do not fabricate a specific, factual-sounding detail just because it fits "
        "the vibe of the answer — a plausible-sounding fake fact (e.g. a made-up sample or "
        "co-writer) is worse than admitting you don't know.\n"
        "- STRUCTURED LISTS — BE CAREFUL: Full track listings, complete discographies, and "
        "exact chart positions are easy to get wrong from memory. If those details appear "
        "above, use them. If an album/artist article is present above but the track list or "
        "discography is clearly missing or truncated, say you don't have the full list "
        "rather than inventing song titles or album sequences. Partial lists from above are "
        "fine to share as partial. Never invent a confident full tracklist or discography "
        "that is not supported here or by very high-confidence knowledge of a famous release.\n"
        "- Never refer to 'the excerpt', 'the context/background I was given', 'the database', "
        "or similar meta-phrasing — the user should never see the plumbing.\n"
        f"{end_note}"
    )
# --- Wikipedia background-context (local RAG) ------------------------------
# A local SQLite database of Wikipedia articles used ONLY to give the writer
# model general background knowledge (artist history, formation, discography,
# reception, cultural context) so it hallucinates less about things audio
# analysis can't tell it. It is explicitly walled off from anything audio-
# technical — tempo, key, instrumentation, structure, and production
# characteristics always come from the PRIVATE TRACK NOTES (MOSS-Music +
# Essentia + stem/MIDI), never from this database. See build_wiki_context_block().
# (ENABLE_WIKI_CONTEXT moved to the settings file beside this script.)
# (WIKI_DB_PATH moved to the settings file beside this script.)
# (WIKI_DB_TIMEOUT_S moved to the settings file beside this script.)
# (WIKI_CONTEXT_MAX_CHARS_PER_ARTICLE moved to the settings file beside this script.)
# (WIKI_CONTEXT_TOTAL_MAX_CHARS moved to the settings file beside this script.)
# (WIKI_MAX_ARTICLES moved to the settings file beside this script.)
# (WIKI_SEARCH_ROW_LIMIT moved to the settings file beside this script.)

# Exact-title lookup, run before the FTS body search.
#
# Measured against a real music wiki DB, the canonical article ranks far
# outside any sane FTS window because bm25 rewards short, dense pages:
#     query           canonical article's body-search rank
#     "Band G"      39      (top hits: List of songs recorded by..., a festival article)
#     "Band F"   35      (top hits: List of awards..., UK Band F)
#     "Name"        74      (top hits: List of songs..., Bibliography of works on...)
#     "Band R"      11
# _wiki_rank_prefer_album_title scores an exact title match +20 and demotes
# "List of ..." by -15, but it can only reorder rows that were fetched — at a
# window of 8 or 24 the real article was never in the set. Raising the window
# far enough (100+) to catch rank 74 would mean pulling full article bodies for
# every query, which is too slow.
#
# An exact title lookup settles it in one step and cannot be out-ranked. It is
# issued as an FTS MATCH on the title column ANDed with a title equality test:
# the MATCH narrows through the inverted index, and the equality test then runs
# over only those few rows. A bare "WHERE title = ?" would instead scan the
# whole corpus, because FTS5 keeps no ordinary index on its columns -- that
# costs ~95 ms per lookup on a full music wiki versus ~0.1 ms this way.
# (WIKI_EXACT_TITLE_LOOKUP moved to the settings file beside this script.)
# Raised from 4. With round-robin allocation every query now gets a first-pass
# slot, so a later strong entity ("Firstname Lastname" in a two-person comparison) must
# still be able to reach its exact-title lookup -- otherwise it falls back to
# fuzzy ranking and loses to a page that merely shares a surname. Repeats are
# cheap: _WIKI_EXACT_TITLE_CACHE memoises hits AND misses for the session.
# (WIKI_EXACT_TITLE_MAX_PER_REQUEST moved to the settings file beside this script.)
# (WIKI_EXACT_TITLE_MAX_HITS moved to the settings file beside this script.)
                                           # band/artist form when both exist.

# Session cache for exact-title lookups, keyed on a normalized candidate.
# Entity names repeat constantly across a conversation ("Band G" recurs in
# every follow-up), and a miss is worth caching as much as a hit — repeatedly
# re-searching for a title that does not exist was a large part of the delay.
_WIKI_EXACT_TITLE_CACHE = {}
# When True, re-run the wiki lookup on EVERY question (folding the question's
# own keywords/entities into the search) instead of caching one lookup per
# track. Off by default, since a single per-track lookup already covers the
# common "artist/album/song background" case cheaply. Turn this on if you
# want the background context to react to what's actually being asked (e.g.
# "when did they form" vs. "what other albums did they release"). Works best
# with an FTS5 index in music_wiki_heavy.db to stay fast; without one, per-question
# re-querying falls back to the same cheap exact/prefix title match every
# time (see _wiki_multi_search) rather than a slow full-table scan.
# (WIKI_CONTEXT_REFRESH_EVERY_QUESTION moved to the settings file beside this script.)
# When True, ALSO search the wiki DB on general chat messages (the ones that
# aren't /listen or /load — see the 'else' branch at the bottom of main()),
# not just track-anchored ones. Lets things like "what other albums did
# Band R release in the 90s?" pull in relevant background even when no
# track is loaded, or when the question is about something other than the
# currently-loaded track. Off by default since general chat has no
# guaranteed track/artist anchor, so this runs a broader, question-only
# search on every such message — turn on if you want that coverage and are
# fine with the extra DB query per general message (cheap with an FTS5 index).
# (WIKI_SEARCH_EVERY_MESSAGE moved to the settings file beside this script.)
# When True, prints a one-line status after every general-chat question showing
# whether the local Wikipedia DB actually returned a match (and which article(s)),
# or whether it came up empty and the writer model is answering from its own
# knowledge with no grounding at all. Silent misses here are indistinguishable
# from a grounded answer in the transcript, which makes hallucinations hard to
# spot — this makes retrieval failures visible in real time instead of only via
# /debug after the fact.
# (WIKI_DEBUG_LOG moved to the settings file beside this script.)

# (DEBUG_FLAG moved to the settings file beside this script.)

# ---- /lookup: background from the local Wikipedia database -----------------------

# Chat-only Wikipedia retrieval. Scan-stage callers retain the legacy algorithm.


def _chat_wiki_normalize(text):
    text = unicodedata.normalize("NFKC", str(text or ""))
    text = text.translate(_WIKI_TITLE_PUNCT_FOLD_MAP)
    return re.sub(r"\s+", " ", text).strip()


def _chat_wiki_key(text):
    key = _wiki_title_compare_key(_chat_wiki_normalize(text))
    if re.fullmatch(r"(?:[^\W\d_]\.)+[^\W\d_]\.?", key):
        key = key.rstrip(".")
    return key


# 0.7.1-pre25 LOOKUP_BETTER_SUBJECTS: band-member words before "of"/"in": "the drummer in X" is about X.
_CHAT_WIKI_MEMBER_ROLE = (
    r"(?:(?:lead|main|original|current|former|founding|first|new|bass|rhythm|backing|other)\s+)*"
    r"(?:singers?|vocalists?|guitarists?|bassists?|drummers?|keyboardists?|pianists?|frontm[ae]n|"
    r"frontwom[ae]n|members?|founders?|leaders?|players?|line-?up)")

# 0.7.1-pre25 LOOKUP_BETTER_SUBJECTS: pre24's openings of a question about one thing, and more
# ("can you tell me about", "I'd like to know about", "who are" ...). What pre24's matched, this matches
# the same way.
_CHAT_WIKI_DIRECT_MORE_RE = re.compile(
    r"(?i)^(?:(?:please|so|ok(?:ay)?|hey|and)\s*,?\s+)?"
    r"(?:what (?:is|was|are|were)|what about|who (?:is|was|are|were)|tell me (?:more |something )?about|"
    r"do you know(?: about)?|have you heard of|explain|how does|"
    r"(?:can|could|would|will) you (?:please )?(?:tell me|explain)(?: (?:more|something|a bit|a little))?(?: about)?|"
    r"(?:i'?d|i would) (?:like|love) to (?:know|learn|hear)(?: (?:more|something|a bit|a little))? about|"
    r"i want to (?:know|learn|hear)(?: (?:more|something|a bit|a little))? about|"
    r"what do you know about|what can you tell me about|(?:i'?m|i am) (?:curious|interested) (?:about|in)|"
    r"give me (?:some )?(?:background|info(?:rmation)?) (?:on|about))\s+(.+?)[?.!]*$")

_CHAT_WIKI_CAP = "[A-Z\u00C0-\u00D6\u00D8-\u00DE]"
_CHAT_WIKI_CAP_RUN = _CHAT_WIKI_CAP + r"[\w'&.!?-]*(?:\s+" + _CHAT_WIKI_CAP + r"[\w'&.!?-]*)*"


def _chat_wiki_shorter_forms(phrase):
    """0.7.1-pre25 LOOKUP_BETTER_SUBJECTS: shorter readings of a question's whole topic phrase,
    tried only when the whole phrase finds no article: [(name, kind, owner)]."""
    s = phrase.strip()
    forms = []
    m = re.match(r"(?i)^" + _CHAT_WIKI_MEMBER_ROLE + r"\s+(?:of|in|for|with|from)\s+(?:the\s+)?(.+)$", s)
    if m:
        forms.append((m.group(1), "artist", ""))
    m = re.match(r"(?i)^(?:producers?|songwriters?|writers?|composers?|authors?)\s+(?:of|for|on)\s+(.+)$", s)
    if m:
        forms.append((m.group(1), "", ""))
    m = re.match(r"(?i)^(?:reception|meaning|lyrics|story|background|origins?|production|recording|sound|"
                 r"style|music|genre|influences?|success|legacy|line-?up|career|discography)\s+of\s+(.+)$", s)
    if m:
        forms.append((m.group(1), "", ""))
    m = re.match(r"(?i)^(.+?)['\u2019]s\s+(.+?)\s+(album|record|lp|ep|song|single|track)$", s)
    if m:
        forms.append((m.group(2), "album" if m.group(3).lower() in ("album", "record", "lp", "ep") else "song",
                      m.group(1)))
    t = re.sub(r"(?i)\s+(?:about|mean|means|like|made (?:from|of)|used (?:for|in)|known for|famous for|called)$",
               "", s)
    if t != s:
        forms.append((t, "", ""))
    u = re.sub(r"\s+in\s+" + _CHAT_WIKI_CAP + r"[\w'-]*(?:\s+" + _CHAT_WIKI_CAP + r"[\w'-]*)*$", "", t)
    if u != t and u:
        forms.append((u, "", ""))
    for base in [f[0] for f in forms] + [s]:
        m = re.match(r"^(.+?)\s+by\s+(?:the\s+)?(" + _CHAT_WIKI_CAP_RUN + r")$", base)
        if m:
            forms.append((m.group(1), "", m.group(2)))
    return forms


def _chat_wiki_joined_titles(clause):
    """0.7.1-pre25 LOOKUP_BETTER_SUBJECTS: capitalised names joined by a small word and "the"/"a"
    ("Name in the Sky with Words", "Title in the U.S.A.", "Word on the Water"), which the
    title-case scan reads as separate names."""
    word = _CHAT_WIKI_CAP + r"[\w'&.!?-]*"
    run = word + r"(?:\s+(?:" + word + r"|(?:of|and|or|for|to|in|on|at|with|&|n')\s+" + word + r"))*"
    stop = _WIKI_LEADING_STOP_CAPS | _WIKI_QUESTION_START_WORDS
    out = []
    pattern = (r"(?<![\w'])" + run + r"\s+(?:of|in|on|for|to|with|at|and|from|into|over|under|by|out of)"
               r"\s+(?:the|a|an)\s+" + run)
    for m in re.finditer(pattern, clause):
        words = m.group(0).split()
        while words and words[0].lower().strip(".'!?") in stop:
            words = words[1:]
        phrase = _wiki_normalize_entity(" ".join(words))
        if phrase and len(phrase.split()) >= 3 and phrase not in out:
            out.append(phrase)
    return out


def _chat_wiki_subjects(question, tracks=(), history=()):
    """Bounded, deterministic subject extraction; descriptors remain topic terms.

    Known track identities are matched before heuristics. Clause boundaries are
    preserved, and digit-led names/acronyms are not dependent on capitalisation.
    No artist-only whitelist: explicit general-topic requests are first-class.
    """
    q = _chat_wiki_normalize(question)
    q = re.sub(r"^(?:comrade|musiclyse|sir|madam)\s*,\s*", "", q, flags=re.I)
    result = []
    seen = set()
    better = LOOKUP_BETTER_SUBJECTS
    shorter = []                 # 0.7.1-pre25: readings tried only when their whole phrase finds nothing
    def dotted(name):
        # 0.7.1-pre25: a name ending in "U.S.A." keeps its last dot.
        name = _wiki_normalize_entity(name)
        if better and re.search(r"(?:\b[A-Za-z]\.)+[A-Za-z]$", name) and (name + ".") in q:
            name += "."
        return name
    def add(name, kind="", owner="", source="question", parent=None):
        name = dotted(name)
        key = _chat_wiki_key(name)
        if not key or key in seen or key in _WIKI_QUESTION_STOPWORDS:
            return
        seen.add(key)
        if parent:
            shorter.append(dict(name=name, kind=kind, owner=owner, source=source, parent=dotted(parent)))
        else:
            result.append(dict(name=name, kind=kind, owner=owner, source=source))
    # Match entire metadata names, not accidental name fragments.
    folded_q = _chat_wiki_key(q)
    for tr in tracks:
        for field, kind in (("artist", "artist"), ("album", "album"), ("title", "song")):
            name = tr.get(field) or ""
            if name and re.search(r"(?<!\w)" + re.escape(_chat_wiki_key(name)) + r"(?!\w)", folded_q):
                if (better and field == "album" and tr.get("title")
                        and _chat_wiki_key(tr["title"]) == _chat_wiki_key(name)):
                    # 0.7.1-pre25: the loaded song and its album share the name: the song, unless
                    # the question says album.
                    kind = "album" if re.search(r"(?i)\b(?:album|record|LP|EP)\b", q) else "song"
                add(name, kind, tr.get("artist", "") if field != "artist" else "", "metadata mention")
    # Quoted titles retain their punctuation, including explicit disambiguators.
    for m in re.finditer(r'"([^"\n]{2,100})"', q):
        prefix = q[max(0, m.start()-20):m.start()].lower()
        kind = "album" if re.search(r"\b(?:album|record|ep)\s*$", prefix) else ""
        add(m.group(1), kind, source="quoted")
    # Direct noun-phrase questions also handle lowercase non-music subjects.
    if better:
        direct = _CHAT_WIKI_DIRECT_MORE_RE.match(q)
    else:
        direct = re.match(r"(?i)^(?:what (?:is|was|are)|what about|who (?:is|was)|tell me (?:more )?about|do you know(?: about)?|have you heard of|explain|how does)\s+(.+?)[?.!]*$", q)
    direct_subject = ""
    if direct:
        direct_subject = re.sub(r"(?i)\s+(?:work|works)$", "", direct.group(1)).strip(" ?!.")
        direct_subject = re.sub(r"(?i)^(?:the|a|an)\s+", "", direct_subject)
        direct_subject = re.sub(r"(?i)^(?:history|origins|meaning|purpose) of\s+", "", direct_subject)
        # Complex questions need the names below, not the whole sentence.
        if (len(direct_subject.split()) <= 7 and not re.search(
                r"(?i)\b(?:their|they|this|that|its|his|her|were|besides|versus|vs|compare)\b", direct_subject)
                and direct_subject.lower() not in {"it", "them", "him", "her", "the album", "album", "song", "band"}):
            kind = ""
            role = re.match(r"(?i)^(band|group|artist|singer|album|song|instrument|genre)\s+(.+)$", direct_subject)
            if role:
                kind = {"band":"artist", "group":"artist", "singer":"artist"}.get(role.group(1).lower(), role.group(1).lower())
                direct_subject = role.group(2)
            add(direct_subject, kind, source="explicit topic")
            if better:
                for name, form_kind, owner in _chat_wiki_shorter_forms(direct_subject):
                    add(name, form_kind or ("" if owner else kind), owner, "shorter form", parent=direct_subject)
        else:
            direct_subject = ""
    # Strip contractions before scanning so Here's is never an artist possessive.
    scan = re.sub(r"(?i)\b(?:here|there|what|who|it|that|let)'s\b", "", q)
    scan = re.sub(r'"[^"\n]*"', " ", scan)
    if direct_subject and not better:
        scan = re.sub(re.escape(direct_subject), " ", scan, flags=re.I)
    for subject in list(result):
        if better and direct_subject and subject["source"] == "explicit topic":
            continue             # 0.7.1-pre25: names inside the phrase are its shorter readings
        scan = re.sub(r"(?<!\w)" + re.escape(subject["name"]) + r"(?!\w)", " ", scan, flags=re.I)
    for clause in re.split(r"[,;:!?\n]", scan):
        runs = _wiki_title_case_runs(clause)
        # Names beginning with digits and dotted acronyms survive intact.
        runs += re.findall(r"\b(?:\d+[A-Za-z][\w.-]*|\d+(?:-\d+)+|(?:[A-Z]\.){2,}[A-Z]?\.?)", clause)
        joined = _chat_wiki_joined_titles(clause) if better else []
        if joined:
            runs = joined + [r for r in runs if r not in joined]
        for phrase in runs:
            phrase = _wiki_normalize_entity(phrase)
            if _wiki_is_adverbial_token(phrase) or phrase.lower() in {"here", "comrade", "unfortunately", "okay", "thanks"}:
                continue
            # Adjectives preceding a class noun describe a subject; they don't
            # introduce a person named Christian, a given name, etc.
            if re.search(re.escape(phrase) + r"\s+(?:group|band|artist|music|era)\b", clause, re.I) and len(phrase.split()) == 1:
                continue
            prefix = clause[:max(0, clause.find(phrase))].lower()
            kind = "album" if re.search(r"\b(?:album|record|on)\s*$", prefix) else ""
            if re.search(r"\b(?:band|group|singer|artist)\s*$", prefix):
                kind = "artist"
            owner, parent = "", None
            if better:
                # 0.7.1-pre25: inside a longer name (joined title, or the question's whole topic
                # phrase), a name is a shorter reading; "X by Y": X made by Y.
                plain = _wiki_normalize_entity(phrase)
                parent = next((j for j in joined if plain != j and plain in j), None)
                if parent is None and direct_subject and plain != direct_subject and plain in direct_subject:
                    parent = direct_subject
                end = clause.find(phrase)
                by = (re.match(r"\s+by\s+(?:the\s+)?(" + _CHAT_WIKI_CAP_RUN + r")", clause[end + len(phrase):])
                      if end >= 0 else None)
                if by:
                    owner = _wiki_normalize_entity(by.group(1))
                    if kind == "album" and not re.search(r"\b(?:album|record)\s*$", prefix):
                        kind = ""        # "on X by Y": a song or an album
            add(phrase, kind, owner, parent=parent)
            # Keep the full name as well: a real 'X and Y' band can win exactly.
            for part in _wiki_split_conjoined_entity(phrase):
                add(part, source="comparison", parent=parent)
            if re.search(r"(?i)\b(?:compare|comparison|versus|vs)\b", q):
                parts = re.split(r"\s+(?:and|versus|vs\.?|with|to)\s+", phrase, flags=re.I)
                if len(parts) > 1 and all(p[:1].isupper() for p in parts):
                    for part in parts:
                        add(part, source="comparison")
    if not result and len(q.split()) <= 5:
        words = set(re.findall(r"\w+", q.lower()))
        if words and not words & (_WIKI_QUESTION_STOPWORDS | {"hello", "thanks", "okay", "yes", "no"}):
            add(q.rstrip("?"), source="bare subject")
    if result:
        return result[:8] + shorter[:6]
    # Explicit recording references bind to metadata, not a historical aside.
    track_ref = bool(re.search(r"(?i)\b(?:this|the|that) (?:song|track|album|band|artist)\b", q))
    if history and not track_ref and _wiki_has_anaphora(q):
        for msg in reversed(list(history)[-8:]):
            if msg.get("role") != "user" or not isinstance(msg.get("content"), str):
                continue
            content = msg["content"]
            if content.startswith("["):
                continue
            previous = _chat_wiki_subjects(content[:600], tracks, ())
            if previous:
                return [dict(s, source="conversation") for s in previous[:4]]
    if tracks and (track_ref or _wiki_has_anaphora(q) or not q.strip() or _wiki_question_focus(q) != "general"):
        focus = _wiki_question_focus(q)
        fields = (("album", "album"), ("artist", "artist")) if focus in {"tracklist", "album"} else (("artist", "artist"), ("title", "song"))
        if focus == "song":
            fields = (("title", "song"), ("artist", "artist"))
        for field, kind in fields:
            for tr in tracks:
                add(tr.get(field, ""), kind, tr.get("artist", "") if field != "artist" else "", "track reference")
    return result[:8]


def _chat_wiki_rows(conn, schema, subject):
    """Bounded indexed retrieval; no changes to legacy caches or scan search."""
    table = _wiki_quote_ident(schema["table"])
    title = _wiki_quote_ident(schema["title_col"])
    body = _wiki_quote_ident(schema["text_col"])
    name = subject["name"]
    variants = [name]
    if not name.lower().startswith("the "):
        variants.append("The " + name)
    if subject["kind"] == "artist":
        variants += [name + " (band)", name + " (musician)", "The " + name]
    elif subject["kind"] in {"song", "album"}:
        kind = subject["kind"]
        variants += [name + " (" + kind + ")"]
        if subject["owner"]:
            variants += [name + " (" + subject["owner"] + " " + kind + ")"]
    found, aliases = {}, set()
    def collect(rows):
        for row in rows:
            if row["t"] and row["b"]:
                found[_chat_wiki_key(row["t"])] = (str(row["t"]), str(row["b"]))
    def exact(candidate):
        if schema["mode"] == "fts":
            query = _wiki_fts_query_escape(_chat_wiki_normalize(candidate))
            if not query:
                return
            collect(conn.execute(
                f'SELECT {title} AS t, {body} AS b FROM {table} WHERE {table} MATCH ? AND {title} = ? COLLATE NOCASE LIMIT 1',
                (f'{schema["title_col"]} : ({query})', candidate)))
        else:
            collect(conn.execute(f'SELECT {title} AS t, {body} AS b FROM {table} WHERE {title} = ? COLLATE NOCASE LIMIT 1', (candidate,)))
    for variant in variants:
        exact(variant)
        target = _wiki_resolve_redirect(conn, schema, variant)
        if target:
            exact(target)
            aliases.add(_chat_wiki_key(target))
    query = _wiki_fts_query_escape(_chat_wiki_normalize(name))
    if schema["mode"] == "fts" and query:
        # Title-only search retains disambiguated and Unicode-normalised forms.
        collect(conn.execute(f'SELECT {title} AS t, {body} AS b FROM {table} WHERE {table} MATCH ? LIMIT 64',
                             (f'{schema["title_col"]} : ({query})',)))
        weights = ["10.0" if c == schema["title_col"] else "1.0" for c in schema.get("columns", [])]
        order = f"bm25({table}, {', '.join(weights)})" if weights else "rank"
        try:
            rows = conn.execute(f'SELECT {title} AS t, {body} AS b FROM {table} WHERE {table} MATCH ? ORDER BY {order} LIMIT ?', (query, WIKI_SEARCH_ROW_LIMIT)).fetchall()
        except Exception:
            rows = conn.execute(f'SELECT {title} AS t, {body} AS b FROM {table} WHERE {table} MATCH ? ORDER BY rank LIMIT ?', (query, WIKI_SEARCH_ROW_LIMIT)).fetchall()
        collect(rows)
    elif schema["mode"] != "fts":
        escaped = name.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        collect(conn.execute(f"SELECT {title} AS t, {body} AS b FROM {table} WHERE {title} LIKE ? ESCAPE '\\' LIMIT 24", (escaped + "%",)))
    return list(found.values()), aliases


def _chat_wiki_score(subject, title, body, aliases):
    name, key = subject["name"], _chat_wiki_key(title)
    wanted = _chat_wiki_key(name)
    base = re.sub(r"\s*\([^)]*\)\s*$", "", key)
    lead = _chat_wiki_key(body[:1800])
    if "(disambiguation)" in key or re.search(r"\bmay (?:also )?refer to\b", lead[:200]):
        return 0, "disambiguation page"
    exact = key == wanted or key in aliases
    base_match = base == wanted or base == "the " + wanted or "the " + base == wanted
    score = 100 if exact else 88 if base_match else 0
    # Acronym expansion is allowed only in the article's introductory identity,
    # never just because an acronym appears somewhere in a long article.
    if not score and re.fullmatch(r"[A-Z][A-Z0-9.]{1,9}", name):
        if re.search(r"\(" + re.escape(wanted) + r"\)", lead[:500]):
            score = 85
    section_match = False
    if not score:
        # A topic can be a named section of a broader reference article, e.g.
        # plate reverb within Reverberation. That is stronger than a body mention.
        section_match = any(_chat_wiki_key(h) == wanted for h, _ in _wiki_split_sections(body)[1:])
        if section_match:
            score = 84
    if not score:
        return 0, "body mention or unrelated title"
    kind = subject["kind"]
    clues = {
        "artist": r"\b(?:band|musician|singer|rapper|musical group|duo|orchestra)\b",
        "album": r"\b(?:album|extended play|ep)\b",
        "song": r"\b(?:song|single)\b",
    }
    if kind in clues:
        if re.search(clues[kind], key + " " + lead[:500]):
            score += 12
        elif re.search(r"\b(?:given name|surname|film|village|city)\b", key + " " + lead[:250]):
            return 0, "wrong subject type"
    owner = _chat_wiki_key(subject["owner"])
    if owner and kind in {"song", "album"}:
        if owner in lead or owner in key:
            score += 20
        else:
            return 0, "artist relationship not established"
    return score, ("exact/alias identity" if exact else
                   "matching named section" if section_match else "matching subject identity")


def _chat_wiki_snippet(body, question, cap):
    """Question-specific paragraphs with a short lead; retain list formatting."""
    sections = _wiki_split_sections(body)
    focus = _wiki_question_focus(question)
    terms = set(_wiki_question_keywords(question, max_terms=20)) - {"comrade", "group", "tell", "know", "think"}
    q = question.lower()
    if re.search(r"christian|religio|faith", q):
        terms.update({"christian", "religion", "religious", "faith"})
    if re.search(r"produc|credit|wrote|writer", q):
        terms.update({"producer", "produced", "personnel", "credits", "songwriter"})
    lead = sections[0][1].strip() if sections else body.strip()
    lead_cap = min(650, cap//3)
    lead_piece = lead if len(lead) <= lead_cap else lead[:lead_cap].rsplit(" ", 1)[0] + "…"
    pieces = [lead_piece] if lead_piece else []
    ranked = []
    for index, (heading, content) in enumerate(sections):
        paragraphs = re.split(r"\n\s*\n", content.strip())
        for pi, paragraph in enumerate(paragraphs):
            if not paragraph.strip():
                continue
            words = set(re.findall(r"\w+", (heading + " " + paragraph).lower()))
            topical = len(terms & words)
            score = _wiki_section_relevance(heading, paragraph, focus) + 4 * topical
            ranked.append((score, -index, -pi, heading, paragraph.strip()))
    used = sum(map(len, pieces))
    for score, _i, _p, heading, paragraph in sorted(ranked, key=lambda v:v[:3], reverse=True):
        if used >= cap - 100:
            break
        if any(_chat_wiki_key(paragraph) in _chat_wiki_key(p) for p in pieces):
            continue
        # Avoid spending the remainder on the same lead if deep topic text exists.
        if _i == 0 and pieces and _chat_wiki_key(pieces[0]) in _chat_wiki_key(paragraph):
            if len(ranked) > 1:
                continue
        label = f"[{heading}] " if heading else ""
        available = cap - used - len(label) - 2
        if available < 80:
            break
        chunk = paragraph if len(paragraph) <= available else paragraph[:available-1].rsplit(" ", 1)[0] + "…"
        pieces.append(label + chunk)
        used += len(label + chunk) + 2
    return "\n\n".join(pieces)[:cap]


def _chat_wiki_novel(snippet, existing):
    """Deduplicate excerpt content, not the article's title alone."""
    pieces = []
    for piece in snippet.split("\n\n"):
        content = re.sub(r"^\[[^\]]+\]\s*", "", piece)
        key = _chat_wiki_key(content).rstrip("…")
        if key and not any(key in text for text in existing):
            pieces.append(piece)
    return "\n\n".join(pieces)


# 0.7.1-pre25 LOOKUP_PREFER_MUSIC_ARTICLES: the nouns that say what an article is about.
_CHAT_WIKI_IDENTITY_NOUNS = {
    "artist": (r"band|group|duo|trio|quartet|supergroup|singer|rapper|musician|singer-songwriter|songwriter|"
               r"dj|disc jockey|composer|guitarist|drummer|bassist|vocalist|record producer|orchestra|choir|"
               r"boy band|girl group|ensemble|musical act|musical project"),
    "album": r"album|studio album|live album|compilation|extended play|ep|mixtape|soundtrack",
    "song": r"song|single|ballad|anthem|instrumental",
}


def _chat_wiki_identity(title, body):
    """0.7.1-pre25 LOOKUP_PREFER_MUSIC_ARTICLES: what an article is about -- "artist", "album",
    "song", "other" (something else) or None (can't tell) -- from its title's bracketed part, else
    from the noun its first sentence names it with ("... is the fifth studio album by ...")."""
    nouns = _CHAT_WIKI_IDENTITY_NOUNS
    bracket = re.search(r"\(([^)]*)\)\s*$", str(title or ""))
    if bracket:
        text = bracket.group(1).lower()
        for kind in ("album", "song", "artist"):
            if re.search(r"\b(?:" + nouns[kind] + r")\b", text):
                return kind
        return "other"
    first = _chat_wiki_normalize(str(body or "")[:400])
    m = re.search(r"\b(?:is|was|are|were)\s+(?:a|an|the)\s+(.{0,160})", first, re.I)
    if not m:
        return None
    head = re.split(r"\s(?:by|from|formed|released|founded|that|which|who|whose|known|of|in|for|with|about|starring|"
                    r"developed|written|recorded|based|consisting|featuring|created|made|produced|"
                    r"originally|first)\b", m.group(1).lower(), 1)[0]
    found = None
    for kind in ("artist", "album", "song"):
        n = re.search(r"\b(?:" + nouns[kind] + r")\b", head)
        if n and (found is None or n.start() < found[0]):
            found = (n.start(), kind)
    return found[1] if found else "other"


def _chat_wiki_music_cue(question, name):
    """0.7.1-pre25 LOOKUP_PREFER_MUSIC_ARTICLES: True when the question treats `name`, typed with a
    capital, as a musical act: a word such as genre, band, albums, formed or drummer within two
    words of it ("What genre is Band?", "When did Band form?", "the drummer in Band").
    After "in", "at", "from" ... only the band-member words count ("the band played in City")."""
    if not name[:1].isupper():
        return False
    q = _chat_wiki_normalize(question)
    member = re.compile(r"(?:singers?|vocalists?|guitarists?|bassists?|bass|drummers?|drums|keyboardists?|"
                        r"members?|frontm[ae]n|frontwom[ae]n|line-?up)")
    other = re.compile(r"(?:genres?|bands?|group|formed|form|discography|albums|singles|hits|tours?|toured|"
                       r"touring|debut(?:ed)?|reunited?|rappers?|musicians?)")
    for m in re.finditer(r"(?<!\w)" + re.escape(name) + r"(?!\w)", q):
        before = re.findall(r"[\w'-]+", q[:m.start()].lower())[-3:]
        after = re.findall(r"[\w'-]+", q[m.end():].lower())[:3]
        near = before + after
        if any(member.fullmatch(w) for w in near):
            return True
        if before and before[-1] in {"in", "at", "from", "near", "to", "into", "around", "outside", "inside"}:
            continue
        if any(other.fullmatch(w) for w in near) or re.match(r"(?:\w+ )?(?:broke|split) up\b", " ".join(after)):
            return True
    return False


def _chat_wiki_rescore(subject, question, rows, diagnostics):
    """0.7.1-pre25 LOOKUP_PREFER_MUSIC_ARTICLES: rows [(score, title, body)] rescored when the subject
    is an artist, album or song (its kind, or a capitalised name the question treats as an act):
    when some article is of that kind, one about something else is left out; with the kind only
    inferred, those of a musical kind gain 12, as a stated kind's clue word does."""
    kind = subject["kind"]
    if kind == "album" and subject.get("source") == "metadata mention" and not LOOKUP_BETTER_SUBJECTS:
        return rows              # pre24 reads a loaded song with a same-name album as the album: not enforced
    if kind in ("artist", "album", "song"):
        want = {kind}
    elif not kind and _chat_wiki_music_cue(question, subject["name"]):
        want = {"artist", "album", "song"}
    else:
        return rows
    kinds = [_chat_wiki_identity(title, body) for score, title, body in rows]
    if not any(k in want for (score, _t, _b), k in zip(rows, kinds) if score > 0):
        return rows
    out = []
    for (score, title, body), k in zip(rows, kinds):
        if score > 0 and k is not None and k not in want:
            diagnostics.append(f'{subject["name"]} -> {title}: 0 (an article about something else: {k})')
            score = 0
        elif score > 0 and not kind and k in want:
            score += 12
        out.append((score, title, body))
    return out


def _chat_wiki_tie_break(subject, tracks, tied):
    """0.7.1-pre25 LOOKUP_PREFER_MUSIC_ARTICLES: of articles scored too close to choose, the only one
    naming the subject's artist ("X by Y") or a loaded song's artist; None when not exactly one does."""
    names = [subject.get("owner") or ""] + [str(t.get("artist") or "") for t in tracks or ()]
    keys = [k for k in (_chat_wiki_key(n) for n in names) if k]
    if not keys:
        return None
    hits = [row for row in tied if any(k in _chat_wiki_key(row[1]) or k in _chat_wiki_key(row[2][:1800])
                                       for k in keys)]
    return hits[0] if len(hits) == 1 else None


def _chat_wiki_lookup(question, tracks, history, existing_texts, diagnostics, hidden=None, choose=None):
    """Chat-only entry point: select first, deduplicate excerpts, reuse firewall.
    hidden (pre63 WIKI_HIDE_MARKED): article titles never chosen.
    choose (0.7.1-pre25 LOOKUP_CHOOSE_WHEN_UNSURE): choose(name, titles) -> index or None, asked when
    a name's articles are too close to call."""
    conn = _get_wiki_db()
    if conn is None:
        diagnostics.append("Database unavailable: " + str(_WIKI_DB_UNAVAILABLE_REASON or "disabled"))
        return ""
    subjects = _chat_wiki_subjects(question, tracks, history)
    diagnostics.append("Subjects: " + "; ".join(f'{s["name"]} [{s["kind"] or "any type"}; {s["source"]}]' for s in subjects))
    existing = [_chat_wiki_key(t) for t in existing_texts if t]
    per_subject = []
    resolved_names = []
    matched_names = set()        # 0.7.1-pre25: names that found articles (chosen or too close to call)
    for subject in subjects:
        if subject["source"] == "comparison" and any(
                _chat_wiki_key(subject["name"]) in name for name in resolved_names):
            # If the combined name really is an article, don't split the band.
            continue
        if subject.get("parent") and _chat_wiki_key(subject["parent"]) in matched_names:
            continue             # 0.7.1-pre25: the whole phrase found its article
        candidates, aliases = _chat_wiki_rows(conn, _WIKI_DB_SCHEMA, subject)
        scored = []
        rows = []
        for title, body in candidates:
            if hidden and title in hidden:
                diagnostics.append(f'{subject["name"]} -> {title}: marked for removal ({WIKI_DELETE_FLAG})')
                continue
            score, reason = _chat_wiki_score(subject, title, body, aliases)
            diagnostics.append(f'{subject["name"]} -> {title}: {score} ({reason})')
            if LOOKUP_PREFER_MUSIC_ARTICLES:
                rows.append((score, title, body))
            elif score >= 84:
                scored.append((score, title, body))
        if LOOKUP_PREFER_MUSIC_ARTICLES:
            scored = [row for row in _chat_wiki_rescore(subject, question, rows, diagnostics) if row[0] >= 84]
        scored.sort(key=lambda row:(-row[0], _chat_wiki_key(row[1])))
        if scored:
            matched_names.add(_chat_wiki_key(subject["name"]))
        if len(scored) > 1 and scored[0][0] - scored[1][0] < 8:
            tied = [row for row in scored if scored[0][0] - row[0] < 8]
            pick = _chat_wiki_tie_break(subject, tracks, tied) if LOOKUP_PREFER_MUSIC_ARTICLES else None
            if pick is None and choose is not None:
                index = choose(subject["name"], [row[1] for row in tied[:10]])
                pick = tied[index] if isinstance(index, int) and 0 <= index < min(len(tied), 10) else None
                if pick is None:
                    diagnostics.append(f'Ambiguous subject {subject["name"]}: none chosen')
                    continue
            if pick is None:
                choices = ", ".join(r[1] for r in scored[:3])
                diagnostics.append(f'Ambiguous subject {subject["name"]}: {choices}. Specify the full article title.')
                continue
            diagnostics.append(f'{subject["name"]}: {pick[1]} chosen of {len(tied)} close articles')
            scored = [pick]
        if scored:
            # A subject gets its best article, not a quota of loosely related hits.
            per_subject.append(scored[0])
            resolved_names.append(_chat_wiki_key(subject["name"]))
    selected, seen = [], set()
    for score, title, body in per_subject:
        key = _chat_wiki_key(title)
        if key in seen:
            continue
        seen.add(key)
        cap = 3200 if _wiki_question_focus(question) in {"tracklist", "debut"} else 2400
        snippet = _chat_wiki_snippet(body, question, cap)
        novel = _chat_wiki_novel(snippet, existing)
        if not novel:
            diagnostics.append(f"Already supplied: {title}")
            continue
        selected.append((title, novel))
        if len(selected) >= WIKI_MAX_ARTICLES:
            break
    if not selected:
        return ""
    # Keep track acoustic protections whenever any recording is in focus.
    anchor = tracks[0] if tracks else {}
    return build_wiki_context_block_multi(
        artist=anchor.get("artist"), title=anchor.get("title"), album=anchor.get("album"),
        question=question, use_question_entities=False, _lookup_articles=selected)


# ---- 0.7-pre62 /lookup-search: Wikipedia articles chosen by title ------------------
# The titles typed in quotes choose the articles; the rest of the prompt plays no part
# in which ones (only in which paragraphs of them are sent, as with /lookup).

_LOOKUP_SEARCH_ROW_LIMIT = 5000        # titles read for one phrase (the rest are counted as "more")

# One quoted title after /lookup-search: "..." or “...” or ‘...’ or '...' (a closing
# single quote only counts before a space, a comma, another quote or the end, so an
# apostrophe inside a title doesn't end it). A comma between titles is allowed.
_LOOKUP_SEARCH_NAME_RE = re.compile(
    "\\s*(?:\"([^\"\\n]*)\"|\u201c([^\u201d\\n]*)\u201d|\u2018([^\u2019\\n]*)\u2019|'([^\\n]*?)')"
    "(?=[\\s,\"'\u201c\u2018]|$)\\s*,?")


def _lookup_search_fold(text):
    """A title or a typed phrase folded for "contains" matching: case, accents, quote
    and dash styles and spacing don't count."""
    text = _wiki_fold_diacritics(_chat_wiki_normalize(_fold_quotes_for_match(str(text or ""))))
    return re.sub(r"\s+", " ", text).strip().casefold()


def _lookup_search_extract(text):
    """Take every `/lookup-search "Title" ["Title" ...]` out of `text`.
    Returns (the text without them, the titles in order without repeats, a usage
    message when a /lookup-search has no title in quotes, else ""). Text without
    /lookup-search comes back unchanged."""
    flag = re.compile(r"(?i)(?<!\S)" + re.escape(LOOKUP_SEARCH_FLAG) + r"(?=\s|$)")
    if not flag.search(text or ""):
        return text, [], ""
    names, seen, pieces, problem, i = [], set(), [], "", 0
    while True:
        m = flag.search(text, i)
        if not m:
            pieces.append(text[i:])
            break
        pieces.append(text[i:m.start()])
        j, got = m.end(), False
        while True:
            q = _LOOKUP_SEARCH_NAME_RE.match(text, j)
            if not q:
                break
            j = q.end()
            name = re.sub(r"\s+", " ", next(g for g in q.groups() if g is not None)).strip()
            if name:
                got = True
                key = _lookup_search_fold(name)
                if key and key not in seen:
                    seen.add(key)
                    names.append(name)
        if not got:
            problem = (f'  {LOOKUP_SEARCH_FLAG} needs an article title in quotes: {LOOKUP_SEARCH_FLAG} '
                       f'"Article title" ["Another title" ...] [question]\n')
        pieces.append(" ")
        i = j
    return re.sub(r"[ \t]{2,}", " ", "".join(pieces)).strip(), names, problem


def _lookup_search_menu(conn, schema, name, titles, rowids, more, starts_only, marked, highlight=None):
    """The menu for one /lookup-search phrase found in several titles (pre62's, shared
    by pre66's menu on the input line). `marked` holds the titles marked for removal.
    Returns (the chosen index in `titles`, or None when cancelled; whether marks changed)."""
    menu_max = int(LOOKUP_SEARCH_MENU_MAX or 0)       # pre63: 0 = every matching title
    shown = titles[:menu_max] if menu_max > 0 else list(titles)
    count = f"More than {len(titles):,}" if more else f"{len(titles):,}"
    prompt = f'{count} article titles {"start with" if starts_only else "contain"} "{name}"'
    if len(titles) == 1 and not more:      # 0.7.1-pre25 (LOOKUP_SEARCH_ALWAYS_MENU): one title
        prompt = f'1 article title {"starts with" if starts_only else "contains"} "{name}"'
    if menu_max > 0 and len(shown) < len(titles):
        prompt += f" (the first {len(shown)} are listed; a longer phrase narrows it down)"
    elif more:
        prompt += " (a longer phrase narrows it down)"
    labels = [t + (_WIKI_MARK_SUFFIX if t in marked else "") for t in shown]
    actions, extra, changes = None, "", []
    if WIKI_DELETE_COMMAND:      # pre63: d marks for removal, p shows the opening
        actions = _wiki_mark_actions(conn, schema, shown, rowids, labels, changes)
        extra = "d: mark/unmark for removal   p: preview"
    pick = _pick_export_menu(prompt + " -- which one?", labels, highlight=highlight,
                             actions=actions, help_extra=extra)
    for title, now in changes:
        print(f'  ({WIKI_DELETE_FLAG}: "{title}" ' + ("marked for removal)" if now else "unmarked)"))
    return pick, bool(changes)


# pre66 LOOKUP_SEARCH_DRAFT_MENU: what the input line found for its /lookup-search names --
# {name: (titles, rowids, more, starts_only)} -- for the lookup to use as it is: a phrase
# with one title (or none), and each title chosen in a menu, as its only title (so no
# second menu). Emptied for every new line.
_LOOKUP_SEARCH_FOUND = {}


def _expand_lookup_search_draft(text, highlight=None):
    """pre66 LOOKUP_SEARCH_DRAFT_MENU: open the menu of each /lookup-search phrase that is
    in several titles while the line is still a draft, and put the chosen title in place
    of the phrase. Returns (text, review, cancelled): review when a menu opened (the line
    goes back for editing), cancelled when one was left with Esc (the line is then
    returned as it was before that menu, and no later menu opens). Phrases with one
    title or none, and anything that can't be looked up now, are left for the lookup."""
    flag = re.compile(r"(?i)(?<!\S)" + re.escape(LOOKUP_SEARCH_FLAG) + r"(?=\s|$)")
    if not (LOOKUP_SEARCH_COMMAND and ENABLE_WIKI_CONTEXT and flag.search(text or "")):
        return text, False, False
    conn = _get_wiki_db()
    if conn is None:
        return text, False, False
    schema = _WIKI_DB_SCHEMA
    marked = set(_wiki_marked_titles()) if WIKI_DELETE_COMMAND else set()
    replacements, review, cancelled, i = [], False, False, 0
    chosen = {}                    # folded phrase -> the title chosen for it on this line
    while not cancelled:
        m = flag.search(text, i)
        if not m:
            break
        j = m.end()
        while True:
            q = _LOOKUP_SEARCH_NAME_RE.match(text, j)
            if not q:
                break
            j = q.end()
            group = next(g for g in range(1, 5) if q.group(g) is not None)
            name = re.sub(r"\s+", " ", q.group(group)).strip()
            key = _lookup_search_fold(name)
            if not name or not key or name in _LOOKUP_SEARCH_FOUND:
                continue
            if key in chosen:      # the same phrase again: the same title
                replacements.append((q.start(group) - 1, q.end(group) + 1, chosen[key]))
                continue
            watch = sys.stdout if isinstance(sys.stdout, _OutputWatch) else None
            quiet = watch is not None and not watch.wrote
            status(f'Finding "{name}" in the local Wikipedia database...')
            try:
                titles, rowids, more, starts_only = _lookup_search_titles(conn, schema, name)
            except Exception:
                titles = None      # the lookup tries again and reports the problem
            status_done(clear=True)
            if quiet and FRIENDLY_PROCESSING_DISPLAY and sys.stdout.isatty():
                watch.wrote = False    # the status line was erased: the typed line can still be replaced
            if titles is None:
                continue
            if len(titles) < 2 and not (LOOKUP_SEARCH_ALWAYS_MENU and sys.stdin.isatty() and sys.stdout.isatty()):
                _LOOKUP_SEARCH_FOUND[name] = (titles, rowids, more, starts_only)
                continue
            if not titles:
                # 0.7.1-pre25 LOOKUP_SEARCH_ALWAYS_MENU: a screen says no title has the phrase; then
                # back to the line as typed. Kept as found, so Enter on the unchanged line sends it.
                _LOOKUP_SEARCH_FOUND[name] = (titles, rowids, more, starts_only)
                _menu_message_screen(f'{LOOKUP_SEARCH_FLAG} "{name}"', [
                    f'No matching articles: no article title in the local Wikipedia database '
                    f'{"starts with" if starts_only else "contains"} "{name}".',
                    "",
                    "Change the phrase on your line, or press Enter there to send the line as it is."])
                review = cancelled = True
                break
            review = True
            pick, changed = _lookup_search_menu(conn, schema, name, titles, rowids, more, starts_only,
                                                marked, highlight)
            if changed:
                marked = set(_wiki_marked_titles()) if WIKI_DELETE_COMMAND else set()
            if pick is None:
                cancelled = True
                break
            title = titles[pick]
            # Quotes the title doesn't contain, so the line still reads as one title.
            for opening, closing in (('"', '"'), ("\u201c", "\u201d"), ("\u2018", "\u2019")):
                if closing not in title:
                    chosen[key] = opening + title + closing
                    replacements.append((q.start(group) - 1, q.end(group) + 1, chosen[key]))
                    _LOOKUP_SEARCH_FOUND[title] = ([title], {title: rowids.get(title)}, False, starts_only)
                    break
            else:
                # Every kind of quote is in the title: the phrase stays on the line and
                # stands for the chosen title (so Enter doesn't open the menu again).
                _LOOKUP_SEARCH_FOUND[name] = ([title], {title: rowids.get(title)}, False, starts_only)
        i = max(j, m.end())
    for start, end, value in reversed(replacements):
        text = text[:start] + value + text[end:]
    return text, review, cancelled


def _lookup_search_titles(conn, schema, phrase):
    """The article titles that contain `phrase` (anything before or after it; case,
    accents and quote styles ignored), best first: the exact title -- or the article a
    redirect of that name leads to -- then its bracketed versions ("... (song)",
    "... (album)"), then the rest alphabetically.
    Returns (titles, {title: rowid or None}, more, starts_only): `more` when there were
    more than _LOOKUP_SEARCH_ROW_LIMIT; `starts_only` when the database has no
    full-text index, so only titles that start with the phrase can be found quickly."""
    table = _wiki_quote_ident(schema["table"])
    tcol = _wiki_quote_ident(schema["title_col"])
    want = _lookup_search_fold(phrase)
    found, more = {}, False
    if not want:
        return [], found, False, False

    def run(sql, params):
        """(title, rowid) rows; rowid None where the table has none (a view)."""
        for select, has_rowid in ((f"rowid AS r, {tcol} AS t", True), (f"{tcol} AS t", False)):
            try:
                rows = conn.execute(sql.replace("__SEL__", select), params).fetchall()
            except Exception:
                continue
            return [(str(row["t"] or ""), row["r"] if has_rowid else None) for row in rows]
        return []

    # The phrase as whole words, anything before or after it ("Band Name" is in "Band Name
    # discography" but not in "Band Namesake"), as the full-text index matches words.
    contains = re.compile(r"(?<!\w)" + re.escape(want) + r"(?!\w)")

    def collect(rows, always=False):
        for t, r in rows:
            if t and t not in found and (always or contains.search(_lookup_search_fold(t))):
                found[t] = r

    normal = _chat_wiki_normalize(phrase)
    starts_only = schema["mode"] != "fts"
    if not starts_only:
        col = schema["title_col"]
        words = _wiki_fts_query_escape(normal)
        quoted = '"' + normal.replace('"', '""') + '"'
        if words:
            collect(run(f"SELECT __SEL__ FROM {table} WHERE {table} MATCH ? AND {tcol} = ? COLLATE NOCASE LIMIT 5",
                        (f"{col} : ({words})", phrase)))
        rows = []
        for match in (f"{col} : {quoted}", f"{col}:{quoted}", f"{col} : ({words})" if words else None):
            if not match:
                continue
            rows = run(f"SELECT __SEL__ FROM {table} WHERE {table} MATCH ? LIMIT ?",
                       (match, _LOOKUP_SEARCH_ROW_LIMIT + 1))
            if rows:
                break
        more = len(rows) > _LOOKUP_SEARCH_ROW_LIMIT
        collect(rows[:_LOOKUP_SEARCH_ROW_LIMIT])
    else:
        for typed in dict.fromkeys((phrase, normal)):      # as typed, and with plain quotes and dashes
            escaped = typed.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            collect(run(f"SELECT __SEL__ FROM {table} WHERE {tcol} = ? COLLATE NOCASE LIMIT 5", (typed,)))
            rows = run(f"SELECT __SEL__ FROM {table} WHERE {tcol} LIKE ? ESCAPE '\\' LIMIT ?",
                       (escaped + "%", _LOOKUP_SEARCH_ROW_LIMIT + 1))
            more = more or len(rows) > _LOOKUP_SEARCH_ROW_LIMIT
            collect(rows[:_LOOKUP_SEARCH_ROW_LIMIT])
    # A redirect with exactly the typed name (a former name, a nickname) leads to its
    # article, which goes first unless an article has exactly that title.
    target = None
    if not any(_lookup_search_fold(t) == want for t in found):
        target = _wiki_resolve_redirect(conn, schema, phrase) or (
            _wiki_resolve_redirect(conn, schema, normal) if normal != phrase else None)
        if target and target not in found:
            if schema["mode"] == "fts":
                words = _wiki_fts_query_escape(_chat_wiki_normalize(target))
                rows = run(f"SELECT __SEL__ FROM {table} WHERE {table} MATCH ? AND {tcol} = ? COLLATE NOCASE LIMIT 1",
                           (f"{schema['title_col']} : ({words})", target)) if words else []
            else:
                rows = run(f"SELECT __SEL__ FROM {table} WHERE {tcol} = ? COLLATE NOCASE LIMIT 1", (target,))
            collect(rows, always=True)
            target = rows[0][0] if rows else None

    def rank(t):
        f = _lookup_search_fold(t)
        group = 0 if (f == want or t == target) else 1 if f.startswith(want + " (") else 2
        natural = [(0, int(p), "") if p.isdigit() else (1, 0, p) for p in re.split(r"(\d+)", f) if p]
        return (group, natural, t)      # "number 2" before "number 10"

    titles = sorted(found, key=rank)
    return titles, found, more, starts_only


def _lookup_search_article(conn, schema, title, rowid=None):
    """The text of the article called exactly `title` ("" when it can't be read)."""
    table = _wiki_quote_ident(schema["table"])
    tcol = _wiki_quote_ident(schema["title_col"])
    bcol = _wiki_quote_ident(schema["text_col"])
    attempts = []
    if rowid is not None:
        attempts.append((f"SELECT {tcol} AS t, {bcol} AS b FROM {table} WHERE rowid = ?", (rowid,)))
    if schema["mode"] == "fts":
        words = _wiki_fts_query_escape(_chat_wiki_normalize(title))
        if words:
            attempts.append((f"SELECT {tcol} AS t, {bcol} AS b FROM {table} WHERE {table} MATCH ? AND {tcol} = ? LIMIT 1",
                             (f"{schema['title_col']} : ({words})", title)))
    else:
        attempts.append((f"SELECT {tcol} AS t, {bcol} AS b FROM {table} WHERE {tcol} = ? LIMIT 1", (title,)))
    for sql, params in attempts:
        try:
            row = conn.execute(sql, params).fetchone()
        except Exception:
            continue
        if row is not None and str(row["t"] or "") == title and row["b"]:
            return str(row["b"])
    return ""


# ---- 0.7-pre63 /wikidelete: a clean copy of the Wikipedia database --------------------
# The database is only ever read. Marked titles go on a plain-text list; apply copies the
# database, takes the marked articles out of the copy, compacts it, checks it and compares
# every remaining article with the original before the copy gets its final name.

_WIKI_MARKED_CACHE = {"key": None, "titles": []}
_WIKI_DELETE_LIST_HEAD = (
    "# Musiclyse: Wikipedia articles to leave out of a clean copy of {db}.\n"
    "# One exact article title per line; lines starting with # are ignored.\n"
    "# /wikidelete apply writes the clean copy; the database itself is never changed.\n"
)


def _wiki_delete_list_path():
    """Where the list of marked titles is kept: WIKI_DELETE_LIST, else
    "<database>-deletions.txt" beside the database."""
    wanted = str(WIKI_DELETE_LIST or "").strip()
    if wanted:
        return os.path.abspath(os.path.expanduser(wanted))
    return os.path.splitext(os.path.abspath(os.path.expanduser(str(WIKI_DB_PATH))))[0] + "-deletions.txt"


def _wiki_marked_titles():
    """The titles on the list, in order, exactly as written ([] when there is no list)."""
    path = _wiki_delete_list_path()
    try:
        st = os.stat(path)
    except OSError:
        return []
    key = (path, st.st_mtime_ns, st.st_size)
    if _WIKI_MARKED_CACHE["key"] != key:
        titles, seen = [], set()
        try:
            with open(path, encoding="utf-8") as fh:
                for raw in fh:
                    title = raw.rstrip("\r\n")
                    if not title.strip() or title.lstrip().startswith("#") or title in seen:
                        continue
                    seen.add(title)
                    titles.append(title)
        except (OSError, UnicodeDecodeError):
            return []
        _WIKI_MARKED_CACHE.update(key=key, titles=titles)
    return list(_WIKI_MARKED_CACHE["titles"])


def _wiki_hidden_titles():
    """The marked titles that /lookup and /lookup-search leave out (WIKI_HIDE_MARKED)."""
    if not (WIKI_DELETE_COMMAND and WIKI_HIDE_MARKED):
        return set()
    return set(_wiki_marked_titles())


def _wiki_write_marked(titles):
    """Write the list (atomically: a new file, then renamed over the old one)."""
    path = _wiki_delete_list_path()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    temp = path + ".tmp"
    with open(temp, "w", encoding="utf-8") as fh:
        fh.write(_WIKI_DELETE_LIST_HEAD.format(db=os.path.basename(str(WIKI_DB_PATH))))
        for title in titles:
            fh.write(title + "\n")
    os.replace(temp, path)
    _WIKI_MARKED_CACHE["key"] = None


def _wiki_set_marked(title, marked):
    """Put a title on the list (marked=True) or take it off; True when the list changed."""
    titles = _wiki_marked_titles()
    if marked and title not in titles:
        _wiki_write_marked(titles + [title])
        return True
    if not marked and title in titles:
        _wiki_write_marked([t for t in titles if t != title])
        return True
    return False


def _wiki_preview_text(conn, schema, title, rowid=None, limit=1500):
    """The opening of an article, for the p key in the menus."""
    body = _lookup_search_article(conn, schema, title, rowid)
    if not body:
        return f"{title}\n\n(The article's text could not be read.)"
    sections = _wiki_split_sections(body)
    lead = (sections[0][1] if sections else body).strip() or body.strip()
    if len(lead) > limit:
        lead = lead[:limit].rsplit(" ", 1)[0] + " ..."
    heads = [h for h, _ in sections[1:] if h][:12]
    return (f"{title}\n\n{lead}" + ("\n\nSections: " + "; ".join(heads) if heads else "")
            + f"\n\n({len(body):,} characters in all)")


_WIKI_MARK_SUFFIX = "   [marked for removal]"


def _wiki_mark_actions(conn, schema, titles, rowids, labels, changes, enter_marks=False):
    """The extra keys of a menu of article titles: d (and Enter, for /wikidelete) marks or
    unmarks the highlighted one, after a y; p shows its opening. `labels` follow the
    marks; `changes` collects (title, marked)."""

    def toggle(index):
        title = titles[index]
        marked = title in set(_wiki_marked_titles())
        question = (f'Take "{title}" off the removal list?' if marked else
                    f'Mark "{title}" for removal from the clean copy?') + "  y = yes, any other key = no"

        def then():
            _wiki_set_marked(title, not marked)
            labels[index] = title + ("" if marked else _WIKI_MARK_SUFFIX)
            changes.append((title, not marked))
            return (f'"{title}" is no longer marked' if marked else
                    f'"{title}" marked -- {WIKI_DELETE_FLAG} apply makes the clean copy')
        return ("confirm", question, then)

    def preview(index):
        return ("show", _wiki_preview_text(conn, schema, titles[index], rowids.get(titles[index])))

    actions = {"d": toggle, "p": preview}
    if enter_marks:
        actions["enter"] = toggle
    return actions


def _wiki_sqlite_uri(path, mode="ro"):
    import urllib.parse
    return f"file:{urllib.parse.quote(os.path.abspath(path))}?mode={mode}"


def _wiki_delete_plan(conn):
    """How articles are taken out of this database: the table that holds them, its key
    and title columns, and the search index to keep in step. None (with a reason) when
    the layout can't be handled safely."""
    schema = _wiki_detect_schema(conn)
    if schema is None:
        return None, "no article table was recognised"
    plan = {"schema": schema, "fts": None, "trigger": False, "fts_columns": []}
    title = schema["title_col"]
    if schema["mode"] == "fts":
        fts = schema["table"]
        row = conn.execute("SELECT sql FROM sqlite_master WHERE name = ?", (fts,)).fetchone()
        sql = str(row["sql"] if row else "")
        content = re.search(r"""(?i)\bcontent\s*=\s*(['"]?)([^'",)\s]*)\1""", sql)
        if content is not None and not content.group(2):
            return None, "its search index stores no text of its own (contentless), so it can't be cleaned safely"
        plan["fts"] = fts
        plan["fts_columns"] = list(schema.get("columns") or [])
        if content is not None:
            key = re.search(r"""(?i)\bcontent_rowid\s*=\s*(['"]?)([^'",)\s]+)\1""", sql)
            plan.update(table=content.group(2), key=key.group(2) if key else "rowid", title=title)
            triggers = conn.execute("SELECT sql FROM sqlite_master WHERE type = 'trigger' AND tbl_name = ?",
                                    (plan["table"],)).fetchall()
            plan["trigger"] = any(re.search(r"(?i)\bAFTER\s+DELETE\b", str(t["sql"] or "")) and
                                  fts.lower() in str(t["sql"] or "").lower() and
                                  "'delete'" in str(t["sql"] or "").lower() for t in triggers)
        else:
            plan.update(table=fts, key="rowid", title=title)
    else:
        info = conn.execute(f"PRAGMA table_info({_wiki_quote_ident(schema['table'])})").fetchall()
        ipk = [c["name"] for c in info if c["pk"] == 1 and str(c["type"]).upper() == "INTEGER"
               and sum(1 for x in info if x["pk"]) == 1]
        plan.update(table=schema["table"], key=ipk[0] if ipk else "rowid", title=title)
    plan["stable_keys"] = plan["key"] != "rowid" or plan["fts"] is not None
    return plan, ""


def _wiki_find_ids(conn, plan, titles):
    """{title: [keys]} for the titles found in the article table, and the titles not found."""
    table, key, tcol = (_wiki_quote_ident(plan["table"]), plan["key"], _wiki_quote_ident(plan["title"]))
    keysql = "rowid" if key == "rowid" else _wiki_quote_ident(key)
    found, missing = {}, []
    for title in titles:
        rows = []
        for wanted in dict.fromkeys((title, title.strip())):
            if plan["table"] == plan["fts"]:
                words = _wiki_fts_query_escape(_chat_wiki_normalize(wanted))
                if not words:
                    continue
                rows = conn.execute(f"SELECT rowid AS k FROM {table} WHERE {table} MATCH ? AND {tcol} = ?",
                                    (f"{plan['title']} : ({words})", wanted)).fetchall()
            else:
                rows = conn.execute(f"SELECT {keysql} AS k FROM {table} WHERE {tcol} = ?", (wanted,)).fetchall()
            if rows:
                break
        if rows:
            found[title] = [r["k"] for r in rows]
        else:
            missing.append(title)
    return found, missing


def _wiki_remove_files(*paths):
    for path in paths:
        for extra in ("", "-journal", "-wal", "-shm"):
            try:
                os.remove(path + extra)
            except OSError:
                pass


def _wiki_make_clean_copy():
    """/wikidelete apply: write "<database>-clean.db" without the marked articles. The
    database is opened read-only throughout; the new file only gets its name once the
    integrity checks pass and every remaining article matches the original."""
    import sqlite3
    db = os.path.abspath(os.path.expanduser(str(WIKI_DB_PATH)))
    titles = _wiki_marked_titles()
    if not os.path.isfile(db):
        print(f"  The Wikipedia database isn't at {db} (WIKI_DB_PATH).\n")
        return
    if not titles:
        print(f"  Nothing is marked yet: {WIKI_DELETE_FLAG} \"Title\" (or d in a {LOOKUP_SEARCH_FLAG} menu) "
              f"marks articles.\n")
        return
    base, ext = os.path.splitext(db)
    out = base + "-clean" + (ext or ".db")
    work, part = out + ".work", out + ".part"
    size = os.path.getsize(db) + (os.path.getsize(db + "-wal") if os.path.exists(db + "-wal") else 0)
    need = int(size * 2.5) + (1 << 30)
    free = shutil.disk_usage(os.path.dirname(out)).free
    if free < need:
        print(f"  Not enough free space beside the database: making the clean copy needs about "
              f"{need / 1e9:.1f} GB while it runs (the copy, its search index being tidied, and the "
              f"finished file); {free / 1e9:.1f} GB is free. Nothing was changed.\n")
        return
    started = time.time()
    src = dst = new = None
    _wiki_remove_files(work, part)
    try:
        src = sqlite3.connect(_wiki_sqlite_uri(db), uri=True, timeout=WIKI_DB_TIMEOUT_S)
        src.row_factory = sqlite3.Row
        plan, why = _wiki_delete_plan(src)
        if plan is None:
            print(f"  {WIKI_DELETE_FLAG} apply can't clean this database: {why}. Nothing was changed.\n")
            return
        found, missing = _wiki_find_ids(src, plan, titles)
        if missing:
            print(f"  Not in the database (left on the list, nothing to remove): "
                  + "; ".join(f'"{t}"' for t in missing))
        if not found:
            print("  None of the marked titles is in the database. Nothing was changed.\n")
            return
        keys = sorted({k for ks in found.values() for k in ks})
        table = _wiki_quote_ident(plan["table"])
        keysql = "rowid" if plan["key"] == "rowid" else _wiki_quote_ident(plan["key"])
        total = src.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]

        # 1. A copy, made through SQLite (consistent even while the file is in use).
        dst = sqlite3.connect(work, isolation_level=None)

        def progress(status_code, remaining, page_total):
            status(f"Copying the database ({100 * (page_total - remaining) // max(1, page_total)}%)...")
        status("Copying the database...")
        src.backup(dst, pages=4096, progress=progress)
        status_done(clear=True)

        # 2. The marked articles out of the copy; the search index follows.
        status(f"Removing {len(keys)} article(s) from the copy...")
        dst.row_factory = sqlite3.Row
        dst.execute("BEGIN")
        try:
            if plan["fts"] and plan["table"] != plan["fts"] and not plan["trigger"]:
                fts, cols = _wiki_quote_ident(plan["fts"]), plan["fts_columns"]
                colsql = ", ".join(_wiki_quote_ident(c) for c in cols)
                for k in keys:
                    row = dst.execute(f"SELECT {colsql} FROM {table} WHERE {keysql} = ?", (k,)).fetchone()
                    if row is not None:
                        dst.execute(f"INSERT INTO {fts}({fts}, rowid, {colsql}) VALUES ('delete', ?, "
                                    + ", ".join("?" for _ in cols) + ")", (k, *tuple(row)))
            removed = 0
            for k in keys:
                removed += dst.execute(f"DELETE FROM {table} WHERE {keysql} = ?", (k,)).rowcount
            if removed != len(keys):
                raise RuntimeError(f"{removed} rows removed where {len(keys)} were expected")
            dst.execute("COMMIT")
        except BaseException:
            dst.execute("ROLLBACK")
            raise
        status_done(clear=True)
        if plan["fts"]:
            status("Tidying the search index (several minutes on a large database)...")
            fts = _wiki_quote_ident(plan["fts"])
            dst.execute(f"INSERT INTO {fts}({fts}) VALUES ('optimize')")
            status_done(clear=True)

        # 3. Compacted into the new file, which keeps a temporary name until checked.
        status("Compacting the copy into the new file...")
        dst.execute("VACUUM INTO ?", (part,))
        dst.close()
        dst = None
        _wiki_remove_files(work)
        status_done(clear=True)

        # 4. Checks: SQLite's own and the search index's.
        status("Checking the new file (SQLite's integrity check)...")
        new = sqlite3.connect(_wiki_sqlite_uri(part, "rw"), uri=True, isolation_level=None)
        new.row_factory = sqlite3.Row
        problems = [r[0] for r in new.execute("PRAGMA integrity_check").fetchall() if r[0] != "ok"]
        status_done(clear=True)
        if not problems and plan["fts"]:
            status("Checking the new file's search index...")
            fts = _wiki_quote_ident(plan["fts"])
            try:
                if plan["table"] != plan["fts"]:
                    new.execute(f"INSERT INTO {fts}({fts}, rank) VALUES ('integrity-check', 1)")
                else:
                    new.execute(f"INSERT INTO {fts}({fts}) VALUES ('integrity-check')")
            except sqlite3.DatabaseError as exc:
                problems.append(f"search index: {exc}")
            status_done(clear=True)

        # 5. Every remaining article, and every other table, compared with the original.
        if not problems:
            status("Comparing every remaining article with the original...")
            new.execute("ATTACH DATABASE ? AS original", (_wiki_sqlite_uri(db),))
            new.execute("CREATE TEMP TABLE gone (k PRIMARY KEY)")
            new.executemany("INSERT INTO temp.gone VALUES (?)", [(k,) for k in keys])
            cols = [c["name"] for c in new.execute(f"PRAGMA main.table_info({table})").fetchall()]
            if plan["fts"] == plan["table"]:
                cols = list(plan["fts_columns"])
            left = total - new.execute(f"SELECT COUNT(*) FROM main.{table}").fetchone()[0]
            if left != len(keys):
                problems.append(f"{left} articles fewer than the original, where {len(keys)} "
                                f"{'was' if len(keys) == 1 else 'were'} removed")
            if new.execute(f"SELECT COUNT(*) FROM main.{table} WHERE {keysql} IN (SELECT k FROM temp.gone)").fetchone()[0]:
                problems.append("a removed article is still in the new file")
            if plan["stable_keys"]:
                differ = " OR ".join(f"a.{_wiki_quote_ident(c)} IS NOT b.{_wiki_quote_ident(c)}" for c in cols)
                bad = new.execute(
                    f"SELECT COUNT(*) FROM original.{table} a LEFT JOIN main.{table} b ON a.{keysql} = b.{keysql} "
                    f"WHERE a.{keysql} NOT IN (SELECT k FROM temp.gone) AND (b.{keysql} IS NULL"
                    + (f" OR {differ}" if differ else "") + ")").fetchone()[0]
            else:
                colsql = ", ".join(_wiki_quote_ident(c) for c in cols)
                bad = new.execute(f"SELECT COUNT(*) FROM (SELECT {colsql} FROM main.{table} EXCEPT "
                                  f"SELECT {colsql} FROM original.{table})").fetchone()[0]
            if bad:
                problems.append(f"{bad} remaining article(s) differ from the original")
            shadow = ({f"{plan['fts']}_{x}".lower() for x in ("data", "idx", "docsize", "config", "content")}
                      if plan["fts"] else set())
            others = [r["name"] for r in new.execute(
                "SELECT name FROM main.sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'").fetchall()
                if r["name"] not in (plan["table"], plan["fts"]) and r["name"].lower() not in shadow]
            for name in others:
                q = _wiki_quote_ident(name)
                a = new.execute(f"SELECT COUNT(*) FROM original.{q}").fetchone()[0]
                b = new.execute(f"SELECT COUNT(*) FROM main.{q}").fetchone()[0]
                d = new.execute(f"SELECT COUNT(*) FROM (SELECT * FROM original.{q} EXCEPT SELECT * FROM main.{q})").fetchone()[0]
                if a != b or d:
                    problems.append(f"table {name} differs from the original")
            if plan["fts"] and not problems:
                fts = _wiki_quote_ident(plan["fts"])
                tq = _wiki_quote_ident(plan["title"])
                for title in found:
                    words = _wiki_fts_query_escape(_chat_wiki_normalize(title))
                    if words and new.execute(f"SELECT COUNT(*) FROM main.{fts} WHERE {fts} MATCH ? AND {tq} = ?",
                                             (f"{plan['title']} : ({words})", title)).fetchone()[0]:
                        problems.append(f'the search index still finds "{title}"')
                sample = new.execute(f"SELECT {keysql} AS k, {tq} AS t FROM main.{table} "
                                     f"ORDER BY random() LIMIT 200").fetchall()
                for row in sample:
                    words = _wiki_fts_query_escape(_chat_wiki_normalize(str(row["t"] or "")))
                    if words and not new.execute(f"SELECT COUNT(*) FROM main.{fts} WHERE {fts} MATCH ? AND rowid = ?",
                                                 (f"{plan['title']} : ({words})", row["k"])).fetchone()[0]:
                        problems.append(f'the search index can\'t find "{row["t"]}"')
                        break
            new.execute("DETACH DATABASE original")
            status_done(clear=True)
        new.close()
        new = None
        if problems:
            print("  The clean copy did NOT pass its checks, so it was deleted: " + "; ".join(problems[:5])
                  + ". The database itself is unchanged.\n")
            return
        os.replace(part, out)
        minutes = (time.time() - started) / 60
        checks = "SQLite's and the search index's checks" if plan["fts"] else "SQLite's check"
        print(f"  Clean copy written: {out}\n"
              f"  {len(keys)} article(s) removed; every one of the other {total - len(keys):,} is identical "
              f"to the original, and {checks} passed.\n"
              f"  Size: {os.path.getsize(db) / 1e9:.2f} GB -> {os.path.getsize(out) / 1e9:.2f} GB "
              f"({minutes:.1f} minutes). The original database is unchanged.\n"
              f"  To use the clean copy, set WIKI_DB_PATH to it (the list of removed titles stays in "
              f"{os.path.basename(_wiki_delete_list_path())}).\n")
    except KeyboardInterrupt:
        status_done(clear=True)
        print(f"\n  {WIKI_DELETE_FLAG} apply stopped; the unfinished copy was deleted. The database is unchanged.\n")
    except (sqlite3.Error, OSError, RuntimeError) as exc:
        status_done(clear=True)
        print(f"  {WIKI_DELETE_FLAG} apply failed ({exc}); the unfinished copy was deleted. "
              f"The database is unchanged.\n")
    finally:
        for connection in (src, dst, new):
            try:
                if connection is not None:
                    connection.close()
            except Exception:
                pass
        _wiki_remove_files(work, part)


def _quoted_titles(text):
    """Titles in quotes at the start of text (as /lookup-search reads them): (titles, the rest)."""
    titles, j = [], 0
    while True:
        q = _LOOKUP_SEARCH_NAME_RE.match(text, j)
        if not q:
            break
        j = q.end()
        name = re.sub(r"\s+", " ", next(g for g in q.groups() if g is not None)).strip()
        if name:
            titles.append(name)
    return titles, text[j:].strip()


def _run_wikidelete_command(argument_text, highlight=None):
    """pre63 /wikidelete: mark articles to leave out of a clean copy of the Wikipedia
    database, list or unmark them, and make the copy (apply). Only the list file and the
    new copy are ever written."""
    text = (argument_text or "").strip()
    word = text.split(None, 1)[0].lower() if text else ""
    usage = (f'  {WIKI_DELETE_FLAG} "Title" ["Title 2" ...]   a menu of the matching article titles: '
             f'Enter or d marks or unmarks one, p shows its opening\n'
             f'  {WIKI_DELETE_FLAG} list                     the marked titles\n'
             f'  {WIKI_DELETE_FLAG} undo "Title" ...         takes titles off the list\n'
             f'  {WIKI_DELETE_FLAG} apply                    writes a clean copy of the database without them '
             f'(the database itself is never changed)\n'
             f'  (In {LOOKUP_SEARCH_FLAG} menus, d marks or unmarks an article and p shows its opening.)\n')
    if not text:
        print(usage)
        return
    if text.lower() == "apply":
        _wiki_make_clean_copy()
        return
    titles_now = _wiki_marked_titles()
    if text.lower() == "list":
        path = _wiki_delete_list_path()
        if not titles_now:
            print(f"  Nothing is marked yet (the list would be {path}).\n")
            return
        print(f"  {len(titles_now)} article(s) marked for removal (list: {path}):")
        for n, title in enumerate(titles_now, 1):
            print(f"    {n}. {title}")
        print(f"  {WIKI_DELETE_FLAG} apply writes the clean copy; {WIKI_DELETE_FLAG} undo \"Title\" unmarks one.\n")
        return
    if word == "undo":
        names, rest = _quoted_titles(text[4:].lstrip())
        if not names or rest:
            print(f'  Use {WIKI_DELETE_FLAG} undo "Title" ["Title 2" ...] (titles in quotes).\n')
            return
        for name in names:
            titles_now = _wiki_marked_titles()
            want = _lookup_search_fold(name)
            hits = ([t for t in titles_now if _lookup_search_fold(t) == want]
                    or [t for t in titles_now if want in _lookup_search_fold(t)])
            if not hits:
                print(f'  "{name}" isn\'t on the list.')
                continue
            if len(hits) > 1:
                pick = _pick_export_menu(f'{len(hits)} marked titles contain "{name}" -- which one to unmark?',
                                         hits, highlight=highlight)
                if pick is None:
                    print(f'  (Nothing unmarked for "{name}".)')
                    continue
                hits = [hits[pick]]
            _wiki_set_marked(hits[0], False)
            print(f'  "{hits[0]}" is no longer marked ({len(_wiki_marked_titles())} left on the list).')
        print()
        return
    names, rest = _quoted_titles(text)
    if not names or rest:
        print(usage)
        return
    conn = _get_wiki_db()
    if conn is None:
        print(f"  The local Wikipedia database isn't available ({_WIKI_DB_UNAVAILABLE_REASON or 'disabled'}).\n")
        return
    schema = _WIKI_DB_SCHEMA
    for name in names:
        status(f'Finding "{name}" in the local Wikipedia database...')
        try:
            titles, rowids, more, starts_only = _lookup_search_titles(conn, schema, name)
        except Exception as e:
            status_done(clear=True)
            print(f'  "{name}" could not be looked up: {e}')
            continue
        status_done(clear=True)
        if not titles:
            print(f'  No article title {"starts with" if starts_only else "contains"} "{name}".')
            continue
        marked = set(_wiki_marked_titles())
        labels = [t + (_WIKI_MARK_SUFFIX if t in marked else "") for t in titles]
        changes = []
        actions = _wiki_mark_actions(conn, schema, titles, rowids, labels, changes, enter_marks=True)
        count = f"More than {len(titles):,}" if more else f"{len(titles):,}"
        prompt = (f'{count} article title{"s" if len(titles) != 1 else ""} '
                  f'{"start" if starts_only else "contain"}{"s" if len(titles) == 1 else ""} "{name}" -- '
                  f'Enter or d marks/unmarks one for removal, p shows its opening')
        pick = _pick_export_menu(prompt, labels, highlight=highlight, actions=actions, help_extra="p: preview")
        if pick is not None:
            # Only without the full-screen menu (a numbered list): confirm here instead.
            title = titles[pick]
            now = title not in set(_wiki_marked_titles())
            answer = colored_input(f'  {"Mark" if now else "Unmark"} "{title}"? (y/N) ', Ansi.LIGHT_GREEN,
                                   prompt_color=Ansi.YELLOW).strip().lower()
            if answer in ("y", "yes"):
                _wiki_set_marked(title, now)
                changes.append((title, now))
        for title, now in changes:
            print(f'  "{title}" ' + ("marked for removal" if now else "unmarked"))
        if not changes:
            print(f'  (Nothing changed for "{name}".)')
    total = len(_wiki_marked_titles())
    print(f"  {total} article(s) on the list ({_wiki_delete_list_path()}); "
          f"{WIKI_DELETE_FLAG} apply writes the clean copy.\n")


_LOOKUP_HEAD_RE = re.compile(r"=== WIKIPEDIA BACKGROUND CONTEXT \(local reference DB — (\d+) article\(s\)\) ===")
_LOOKUP_PERSONA_NOTE = (
    "CUSTOM CHARACTER: a custom character is active. This background is for the default assistant and "
    "for expert or authority characters; use only as much of it as this character would plausibly know "
    "-- none of it for an artist they wouldn't know.\n\n"
)


def _lookup_titles(block):
    return [m.group("t") for m in _LOOKUP_SECTION_RE.finditer(block or "")]


def _lookup_filter(block, skip_folded, persona):
    """The /lookup block without the articles already in this conversation,
    worded for loaded exports; "" when nothing new is left."""
    if not block:
        return ""
    keep, drop = [], []
    for m in _LOOKUP_SECTION_RE.finditer(block):
        (drop if _txt_name_fold(m.group("t")) in skip_folded else keep).append(m)
    if not keep:
        return ""
    out = block
    for m in reversed(drop):
        out = out[:m.start()] + out[m.end():]
    out = _LOOKUP_HEAD_RE.sub(
        f"=== WIKIPEDIA BACKGROUND CONTEXT (local reference DB — {len(keep)} article(s)) ===", out, count=1)
    out = (out.replace("from the PRIVATE TRACK NOTES (when present)", "from the loaded tracks (when present)")
              .replace("the track notes are always right", "the loaded tracks are always right")
              .replace("If this context and the track notes ever disagree",
                       "If this context and the loaded tracks ever disagree"))
    if persona:
        out = re.sub(r"(=== WIKIPEDIA BACKGROUND CONTEXT \(local reference DB [^\n]*===\n)",
                     lambda mm: mm.group(1) + _LOOKUP_PERSONA_NOTE, out, count=1)
    return out
