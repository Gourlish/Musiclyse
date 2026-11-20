# MUSICLYSE SETTINGS 0.8
# ==============================================================================
#  MUSICLYSE 0.8 SETTINGS
# ==============================================================================
# Every setting you can change, in one place: the main script's settings first, then the
# exporter's. Each script reads its own part when it starts, so after a change, start
# Musiclyse again (or run the exporter again).
#
#  * Values are written as in Python: True / False, numbers, "text" in quotes, lists in [ ],
#    and None.
#  * Many settings switch on a rule added in a particular build. Its note says which build
#    ("added in ...") and what False gives instead.
#    Builds 0.7.1-pre1 to 0.7.1-pre30 and 0.8-pre31 to 0.8-pre35 were the test builds of 0.8;
#    the notes name the build that added each change.
#  * To keep your changes when you move to a new build, put them in your own settings file
#    instead (PERSONAL_SETTINGS_FILE, musiclyse-my-settings.py beside the scripts): it carries
#    over to every build, this file doesn't.
#  * Internal detail rather than something you'd want to tune (prompt texts, lookup tables,
#    regular expressions) stays next to the code that uses it.
#  * This file belongs to version 0.8 and must stay in the same folder as
#    musiclyse-0.8.py and musiclyse-0.8-export.py. Keep its first line and the
#    two "SETTINGS" heading lines below as they are: the scripts use them to find their part.
#
# CONTENTS
#   Main script (musiclyse-0.8.py)
#     1. Display and console
#     2. Ollama and the chat model
#     3. Chat, commands and menus
#     4. Wikipedia lookup
#     5. File tags, saving, batch and export
#     6. Speed, memory, model loading and debugging
#     7. Listening model (MOSS-Music)
#     8. Lyrics transcription (Whisper) and tag lyrics
#     9. Vocals
#     10. Singer identity
#     11. Cover art
#     12. Section maps
#     13. SongFormer
#     14. Stems, notes and MIDI
#     15. Drums and rhythm
#     16. Tempo, key, chords and melody
#     17. Genre, style and sound similarity
#     18. Instruments and arrangement
#     19. Production, mix and loudness
#     20. The analysis report
#     21. Video
#     22. Other settings
#   Exporter (musiclyse-0.8-export.py)
#     E1. Who sings: singer identity wording
#     E2. The voice: range and vocal wording
#     E3. Melody, key and notes
#     E4. Tempo, drums and groove
#     E5. Genre, style, era and instruments
#     E6. Sections and song structure
#     E7. The chat guide and its wording
#     E8. Reading saves and writing exports
#     E9. Other settings
# ==============================================================================


# ##############################################################################
# MAIN SCRIPT SETTINGS
# ##############################################################################


# ==============================================================================
#  1. DISPLAY AND CONSOLE
# ==============================================================================

FRIENDLY_PROCESSING_DISPLAY = True

SHOW_OMNIZART_LOGS = False

# The stage timing table, and the exporter's and SongFormer helper's own messages when
# /export and /songformer run them, in the default yellow (colour terminals only).
# (added in 0.7-pre62; False gives pre61's behaviour; the song and video analysis are unchanged)
CONSISTENT_TEXT_COLOR = True

PRINT_STAGE_TIMING = True   # print a per-stage timing table after each fresh /listen analysis

# (added in 0.7.1-pre12; False gives pre11's map)
CLIP_TIMES_LIKE_SECTIONS = True     # clip times printed exactly as the section map prints them

# 0.8-pre31: a file of your own changed settings, beside the scripts, read on top of this file
# every time a script starts -- name it without a version, and the settings in it carry over to
# every new build. Write each one as here (NAME = value), one per line. Each script takes the
# ones in its own part of this file, in place, so settings worked out from them follow; a name
# that isn't a setting here (misspelt, or removed in a later build) gets a warning and is left
# out. "" (or no such file): only this file is read, as before.
PERSONAL_SETTINGS_FILE = "musiclyse-my-settings.py"


# ==============================================================================
#  2. OLLAMA AND THE CHAT MODEL
# ==============================================================================

OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_NUM_CTX = 131072
OLLAMA_SUPPORTS_IMAGES = True
OLLAMA_MODEL = "gemma4:31b"   # or "gemma4:26b" as an alternative
OLLAMA_INTERACTIVE_KEEP_ALIVE = "30m"

# 0.6.2-dev7: leaving Musiclyse (exit, quit, Ctrl-D or Ctrl-C) asks Ollama to
# unload the chat model at once, instead of keeping it in memory for
# OLLAMA_INTERACTIVE_KEEP_ALIVE after the last reply. False gives dev6.
UNLOAD_CHAT_MODEL_ON_EXIT = True

# 0.6.2-dev15: how Musiclyse starts Ollama when none is running at startup.
#   "off"             you start Ollama yourself (as before)
#   "background"      hidden; nothing written to disk; stopped when Musiclyse ends
#   "background_log"  hidden, with its output (what an Ollama window shows) in
#                     OLLAMA_LAUNCH_LOG (logs/ollama.log next to this script),
#                     overwritten each launch; stopped on exit
#   "log_window"      hidden, with that log shown live in its own Terminal window;
#                     Ollama stops and the window closes when Musiclyse ends
#   "window"          Ollama runs in its own Terminal window, like starting it by
#                     hand; left running after Musiclyse ends (close it to stop it)
# An Ollama that is already running is used as it is. The two window modes are
# for macOS; elsewhere they act as "background_log". An override file next to
# this script, named after it with "-ollama-override.py" in place of ".py"
# (e.g. Musiclyse-0.6.2-dev16-ollama-override.py), sets the mode instead.
# "window" for now (development builds); "background_log" is planned for 0.7.
OLLAMA_LAUNCH_MODE = "window"

# The settings Ollama is started with (the same as starting it by hand with
# these variables). Prompt cache off and 4 checkpoints only change what the
# engine keeps in memory for reuse, not what the model writes.
OLLAMA_LAUNCH_ENV = {
    "OLLAMA_FLASH_ATTENTION": "1",
    "LLAMA_ARG_CACHE_RAM": "0",
    "LLAMA_ARG_CTX_CHECKPOINTS": "4",
}

# The log for "background_log" and "log_window". A relative path is inside the
# folder this script is in (next to saved-songs), so the default is logs/ollama.log there.
OLLAMA_LAUNCH_LOG = "logs/ollama.log"

# 0.6.2-dev16: where the "log_window" window goes when it opens: "behind"
# Musiclyse's window, or "minimized" into the Dock (click it there to watch).
OLLAMA_LOG_WINDOW = "behind"

# 0.7-pre5: the cover-art description and singer-identity decision use a fixed
# seed and temperature 0, so the same evidence always gives the same identity.
SCAN_LLM_REPEATABLE = True
SCAN_LLM_SEED = 1234

# 0.8: when Ollama can't be reached, its network error is shown once rather than again for every
# retry, and the chat says plainly that Ollama isn't answering rather than suggesting the model isn't
# installed. False gives pre35's messages.
OLLAMA_UNREACHABLE_PLAIN = True


# ==============================================================================
#  3. CHAT, COMMANDS AND MENUS
# ==============================================================================

MAX_WRITER_IMAGES_PER_TURN = 8          # images actually sent to Ollama in one request
IMAGE_RETAIN_LAST_N_MESSAGES = 8
MAX_WRITER_HISTORY_MESSAGES = 80       # non-chat helper requests only; chat uses its context budget
MAX_MESSAGE_CHARS_FOR_OLLAMA = 120_000
HISTORY_CHAR_BUDGET_FACTOR = 3.2        # conservative chars-per-token estimate for trimming
MAX_EXPLICIT_IMAGE_BYTES = 2_000_000
EXPLICIT_IMAGE_MAX_DIMENSION = 1568   # larger/unsupported images are converted, not rejected

# 0.6.2-dev4: what one image in the chat is counted as when fitting a request
# into the context (vision models turn an image into a fixed number of tokens).
# The general estimate counts a third of the image's base64 length, which made
# any photo over about 800 KB look bigger than the whole context, so it was
# silently left out. Used only while the chat sends a request with images.
CHAT_IMAGE_TOKENS = 1500

# Saved chats retain the full archive; this caps the window used when resuming.
# None uses the model window. Counts are estimated, with reply space reserved.
CHAT_RESUME_MAX_TOKENS = None

# 0.6.2-dev13: whether the chat keeps images in the conversation instead of
# dropping each one IMAGE_RETAIN_LAST_N_MESSAGES messages later. Dropping one
# changes the start of the conversation, so the model has to re-read
# everything after it; Ollama's MLX engine needed that (it can't reuse its
# cache past an image), its llama.cpp engine (GGUF models) doesn't.
# "auto": keep them on GGUF models, drop them as before on others.
# True: always keep. False: always drop (dev12).
CHAT_KEEP_IMAGES = "auto"

# 0.6.2-dev14: when the conversation no longer fits the chat model's context
# remove its oldest messages in
# one go instead of a few more every turn (which made the model re-read the
# whole conversation on every reply). 0.7-pre2: the conversation keeps
# CHAT_TRIM_TARGET of its own share of the room -- what loaded exports and
# looked-up background leave free -- so each trim takes its oldest quarter.
# Loaded exports and looked-up background are never removed. False: dev13.
CHAT_TRIM_HISTORY = True
CHAT_TRIM_TARGET = 0.75

# 0.6.2-dev7: the chat's context window for particular chat models. Ollama
# sets aside memory for the whole window when it loads a model, so a big dense
# model at OLLAMA_NUM_CTX can fill the Mac's memory and swap. Each entry is
# part of a model name (case and punctuation are ignored, so "gemma4" matches
# "gemma4:31b" and "hf.co/unsloth/gemma-4-31B-it-GGUF:Q8_K_XL") and the
# window to use; the first match wins, and a model with no match uses
# OLLAMA_NUM_CTX. At 65536 the chat holds about 157k characters of
# conversation, about three plus-level exports. Only the chat uses this;
# scans keep their own windows. Empty (the default) gives dev6 exactly.
# Example: CHAT_NUM_CTX_BY_MODEL = {"gemma4": 65536}
CHAT_NUM_CTX_BY_MODEL = {"gemma4:31b": 180000, "gemma-4-31b:latest": 180000}

# 0.6.2-dev9: a /load with no question (and an export named when Musiclyse
# starts) gets a reply straight away; add /quiet to a /load to skip it.
# False: loading never replies on its own (dev8).
LOAD_REPLIES_BY_DEFAULT = True

# 0.6.2-dev12: whether chat models that can think before answering do so.
# False: they answer straight away, and /think in a prompt lets that one reply
# think first. True: they always think. None: Musiclyse doesn't say, and the
# model or Ollama decides (dev11). Models that can't think are not affected.
# Scans are not affected either way.
CHAT_THINKING = False

# dev58: the chat voice. The genre rule's example sentences ("the measurements
# put this squarely in dance-pop ...; the article files the band under pop
# rock") taught the writer exactly the source-citing tone the rest of the
# prompt bans, so they are rewritten as plain talk, and one short block says
# how sound, artist and singer fit together in a reply. As a backstop, a reply
# sentence that opens by citing its source ("According to the background, ...",
# "The notes say that ...", "Based on the analysis, ...") loses that opener;
# nothing else in the reply is touched (numbers and timestamps never are), and
# it is skipped when the user asks where something came from. False restores
# dev57.
WRITER_CASUAL_VOICE = True

ANALYSE_FLAG = "/analyse"     # analyse one audio file into saved-songs/ (skipped if already saved)
REANALYSE_FLAG = "/reanalyse" # analyse one audio file again, replacing its save
BATCH_FLAG = "/batch"         # analyse every audio file in a folder into saved-songs/
LOAD_FLAG = "/load"           # load one or more export .txt files (single or combined) into the chat
LOOKUP_FLAG = "/lookup"       # in any prompt: look up background in the local Wikipedia database
LOOKUP_SEARCH_FLAG = "/lookup-search"   # 0.7-pre62: Wikipedia articles chosen by title (LOOKUP_SEARCH_COMMAND)
WIKI_DELETE_FLAG = "/wikidelete"        # 0.7-pre63: articles to leave out of a clean copy (WIKI_DELETE_COMMAND)
CLEAR_FLAG = "/clear"         # wipe the conversation; loaded exports and looked-up background are resent
CLEAR_ALL_FLAG = "/clearall"  # wipe everything, including loaded exports and looked-up background
PERSONA_FLAG = "/persona"     # set / show / reset the chat's custom character

# Commands added after 0.7-pre55 (moved here from further down in 0.7-pre69; unchanged):
MUSICVIDEO_ANALYSE_FLAG = "/musicvideoanalyse"
MUSICVIDEO_LOAD_FLAG = "/musicvideoload"

EXPORT_FLAG = "/export"
SONGFORMER_FLAG = "/songformer"
HELP_FLAG = "/help"               # 0.7-pre65 (HELP_COMMAND)

# 0.7-pre10: the "Persona set to:" confirmation shows the whole description you typed
# (False: only its first 160 characters / first line, as before).
PERSONA_CONFIRM_FULL = True

# /persona waits for a reference image only when the description points to a
# picture ("the person in this photo"), not for roles like "a portrait painter".
# (added in 0.7-pre53; False gives pre52's behaviour)
PERSONA_IMAGE_REFERENCE_ONLY = True

# After a "" menu, the filled-in line replaces the line you entered instead of
# appearing on a new line below it (skipped if anything was printed in between).
# (added in 0.7-pre55; False gives pre54's behaviour)
DRAFT_REPLACES_SUBMITTED_LINE = True

# /load "" after /videoload opens the song menu too; menus open in the order written.
# (added in 0.7-pre55; False gives pre54's behaviour)
LOAD_PICKER_AFTER_VIDEOLOAD = True

# (added in 0.7-pre65; no change to the analysis or the exports)
HELP_COMMAND = True                # /help, typed on its own, lists every command (the start-up list)
CHAT_USAGE_HELP_PRE65 = True       # the default chat's how-to notes know /export, video, /wikidelete, /help

# (added in 0.7-pre65; no change to the analysis or the exports)
CHATLOAD_NAME_MATCH = True         # /chatload name finds "<date-time>-name.json" (a menu when several)

# Left/Right page and g/G jump to the first/last entry in the full-screen menus.
# (added in 0.7-pre63; False gives pre62's behaviour; the song and video analysis are unchanged)
MENU_ARROW_PAGING = True

# The default chat and the games decline anything they won't do briefly, in their own
# voice, offering something close instead. Only how they say no, not what they'll discuss.
# (added in 0.7-pre62; False gives pre61's behaviour; the song and video analysis are unchanged)
CHAT_SOFT_REFUSALS = True

GAME_FLAG = "/game"           # 0.6.2-dev8: choose a game (a built-in character) from a menu
GAME_EXIT_FLAG = "/game-exit" # 0.6.2-dev8: leave the game (clears everything) for the default chat
QUIET_FLAG = "/quiet"         # 0.6.2-dev9: add to a /load to load without a reply
THINK_FLAG = "/think"         # 0.6.2-dev12: add to a prompt to let the chat model think before this reply
DEBUG_FLAG = "/debug"

CHAT_WIKI_LOOKUP_DEBUG = False

# 0.7.1-pre25: /undo removes the last thing you sent and everything it added to the conversation:
# your message and its images, the reply, and songs, Wikipedia articles and videos loaded with it.
# Repeat it to go further back, up to the last /clear, /clearall, /chatload, persona or game
# change. False: there is no /undo command.
UNDO_COMMAND = True
UNDO_FLAG = "/undo"

# 0.7.1-pre25: after /undo the window is cleared and the conversation is shown again without the
# removed part, start-up text included (not during games). False: the window is left as it is.
UNDO_REDRAWS_SCREEN = True

# 0.7.1-pre25: after /undo the removed message is put back on the input line to edit or send
# again. False: the input line starts empty.
UNDO_RESTORES_MESSAGE = True

# 0.7.1-pre25: the default chat assistant is told today's date (from this computer's clock when
# the conversation starts, with a one-line update if it carries on past midnight) and roughly when
# its own knowledge stops (CHAT_MODEL_KNOWLEDGE_CUTOFF), and not to bring them up unless they
# matter. Custom characters and games are unchanged. False gives pre24's prompt.
CHAT_KNOWS_DATE = True

# When each chat model's own knowledge stops, by part of the model's name (matched as in
# CHAT_NUM_CTX_BY_MODEL). A model not listed is told only that it stops some time before today.
CHAT_MODEL_KNOWLEDGE_CUTOFF = {"gemma4": "January 2025"}

# 0.7.1-pre25: the default chat assistant is told it is running on this computer through Ollama
# (with the model's name and, on a Mac, the chip) -- only when OLLAMA_URL is on this computer and
# the model isn't one of Ollama's cloud models. False gives pre24's prompt.
CHAT_KNOWS_IT_IS_LOCAL = True

# 0.7.1-pre26: the chat sends every image in the conversation as long as the conversation fits
# the model's context window (each counted as CHAT_IMAGE_TOKENS), instead of only the newest
# MAX_WRITER_IMAGES_PER_TURN. When it no longer fits, its oldest quarter goes in one go
# (CHAT_TRIM_HISTORY), images included. Scans keep MAX_WRITER_IMAGES_PER_TURN. False gives pre25's
# (past 8 images the oldest is left out of each request, so the model re-reads the conversation
# from there with every new image).
CHAT_IMAGES_FILL_CONTEXT = True

# 0.7.1-pre26: the most images one chat message can attach (pictures and video frames together)
# with CHAT_IMAGES_FILL_CONTEXT; None: as many as fit. With CHAT_IMAGES_FILL_CONTEXT False,
# MAX_WRITER_IMAGES_PER_TURN applies as before.
CHAT_MAX_IMAGES_PER_MESSAGE = None

# 0.7.1-pre26: the default chat assistant is asked not to guess facts aloud (trying one name after
# another) but to work out what it knows first, and say briefly when it isn't sure. Custom
# characters and games are unchanged. False gives pre25's prompt.
CHAT_NO_GUESSING_ALOUD = True

# 0.7.1-pre26: the default chat assistant is told it has no web links of its own and never to
# write a web address that isn't in the conversation. False gives pre25's prompt.
CHAT_NO_MADE_UP_LINKS = True

# 0.7.1-pre26: after each chat reply (any character or game), a web address in it that appears
# nowhere in the conversation gets a short note saying it's probably made up. The reply itself is
# kept as it is. False: no note.
CHAT_FLAG_MADE_UP_LINKS = True

# 0.7.1-pre26: after /undo, Ollama unloads the chat model, so the next reply loads it again and
# reads the whole conversation from scratch instead of going back to a saved point. Slower; for
# when replies straight after /undo seem confused. False: Ollama carries on from its saved point.
UNDO_FRESH_READ = False

# 0.7.1-pre27: the chat counts a message's size the way the chat model's tokenizer splits text:
# every digit is a token of its own, so timestamps, chord times and note lists take far more
# tokens than ordinary words. A message never counts for less than before (one token per
# HISTORY_CHAR_BUDGET_FACTOR characters), so standard and plus exports and ordinary chat count as
# before, while --level high and full exports count at about their real size and requests stay
# inside the context window. Scans keep the old count. False gives pre26's count.
CHAT_REALISTIC_TOKEN_COUNT = True

# 0.7.1-pre27: how much of the chat's prompt space (the context window less the reply's share) the
# loaded exports may fill together, leaving the rest for the conversation; less 2,000 tokens for
# the labels and notes that go with them. An export that doesn't fit whole keeps everything a
# --level plus export has, plus as many of the extra detail sections of --level high and full as
# fit, shared fairly between the loaded exports; the chat model is told which sections were left
# out. Loading another export can change what the others keep. None gives pre26's (exports always
# sent whole, and the oldest left out of each request when they don't fit).
CHAT_EXPORTS_MAX_SHARE = 0.75

# 0.7.1-pre27: /load says whether each export fits whole in the chat's context window and how much
# of the room for exports the loaded ones use; and after a reply, a note when part of the loaded
# exports had to be left out of that request. False: no such lines.
CHAT_LOAD_SHOWS_FIT = True

# 0.7.1-pre27: chat requests ask Ollama not to drop the conversation's oldest messages by itself
# when a request is longer than the context window (it otherwise does that without saying). If
# Ollama then turns the request down as too long, it is sent again the old way and a note after
# the reply says so. False gives pre26's requests.
CHAT_NO_SILENT_TRUNCATION = True

# 0.7.1-pre28: the default chat assistant's "when you're not sure" rule (CHAT_NO_GUESSING_ALOUD) is
# limited to specific facts it can't recall (a name, a date, a chart position) and says plainly
# that opinions, comparisons, connections and enthusiasm are as free as ever. False gives pre27's
# wording of the rule.
CHAT_NOT_SURE_NARROW = True

# 0.7.1-pre28: a last line in the default chat assistant's prompt reminding it to keep the voice
# described at the start, so the prompt doesn't end on rules. Off for now: switch it on to try it.
CHAT_VOICE_REMINDER = False

# 0.7.1-pre29: a long dash the chat model writes straight between two words ("word—word") gets a
# space on each side ("word — word"), as Musiclyse's exports write it -- on screen as the reply
# comes in, and in the copy kept in the conversation and in chat saves, so the model sees its own
# replies written that way. Hyphens ("well-known"), short dashes in ranges and pairs ("0:45–1:12",
# "1970–1975"), a dash between two digits, a dash ending a cut-off sentence and anything inside
# ``` code blocks are left as they are. Every chat reply (any character or game); analyses and
# exports are not touched. False gives pre28's replies.
CHAT_SPACED_DASHES = True

# 0.7.1-pre30: memories. MEMORY_FLAG on its own asks the chat model for a short memory of the
# conversation (with a topic after it, of that one thing) and shows it on a screen: Enter saves it,
# e edits it, Esc cancels. "MEMORY_FLAG list" shows them (Enter reads one, d deletes one);
# "MEMORY_FLAG off" / "on" stop or restart bringing them into conversations (remembered between
# sessions). In the default chat only (not characters or games), each new conversation gets the
# saved memories as one note after the system prompt -- memories saved with a loaded track first,
# then the newest -- up to CHAT_MEMORY_MAX_TOKENS; a track loaded later adds its own memories that
# weren't in that note. The chat's how-to notes mention the command. With no memories saved,
# requests are as before apart from that how-to line. False gives pre29's chat.
CHAT_MEMORY = True
CHAT_MEMORY_FILE = ""              # "" = memories.json beside this script
CHAT_MEMORY_MAX_TOKENS = 2000
MEMORY_FLAG = "/memory"

# 0.8-pre31: the name in front of each chat reply while a custom persona is on (one set with
# PERSONA_FLAG, one made from a picture, or one brought back with a saved chat), so its replies
# aren't taken for the default assistant's. The default chat and games keep their own names. Only
# the screen changes: nothing is sent to the chat model differently. False or "" gives pre30's
# ("Musiclyse").
CHAT_PERSONA_SPEAKER = "Musiclyse (custom persona)"

# 0.8-pre31: the default chat's how-to notes keep the sentence asking for a plain answer when the
# user only asks whether something can be done (commands only when they ask how). False gives 0.7's
# how-to notes, without that sentence.
CHAT_HOWTO_PLAIN_ANSWER = True

# 0.8-pre31: the how-to notes (and the reminder shown when EXPORT_COMMAND is off) name this build's
# own exporter. False gives pre30's (the newest exporter in the folder, which can be another
# build's).
CHAT_HOWTO_THIS_EXPORTER = True

# 0.8-pre31: the memory note says memories remind the chat model what was said, not how the
# recordings sound: a loaded song's export comes first, with no export loaded it says what was
# talked about rather than describing the sound, and the user's views and corrections come back
# as theirs. False gives pre30's note.
CHAT_MEMORY_NOT_SOUND = True

# 0.8-pre31: when writing a memory, the chat model is asked to put what was said about a song as
# what was said ("you felt the chorus lifted"), not as facts about the recording, and to note the
# user's corrections as theirs. False gives pre30's request.
CHAT_MEMORY_SAYS_WHAT_WAS_SAID = True

# 0.8-pre33: /memory sends the conversation exactly as the chat does, memory notes and pictures
# included, so Ollama reuses what it has already read instead of reading every loaded export again
# (for the memory and once more for the next message). The request asks for nothing from the
# earlier memories that didn't come up again. False gives pre32's request, without the notes.
CHAT_MEMORY_REQUEST_KEEPS_NOTES = True

# 0.8: the memory preview gives its text its colour directly and redraws the whole screen each time
# it opens. Before, text changed with e (edit) could show in the terminal's default colour (white)
# when the preview reopened, because only the changed part was redrawn and it lost the colour.
# False gives the previous drawing.
MEMORY_PREVIEW_STEADY_COLOUR = True

# 0.8-pre31: when part of a loaded export is left out to fit the chat's context window, its note
# also asks the chat model not to fill those parts in from what it knows of the song. False gives
# pre30's note.
CHAT_TRIMMED_EXPORT_NO_FILL = True

# 0.8-pre31 (patch): the default chat is told to reply in the language the user writes in (English
# unless they use another); commands, file names and quoted lyrics stay as they are. A note beside
# the date note, not part of the persona. Games and custom characters keep their own language rules.
# False gives pre31's prompt (no language note).
CHAT_REPLY_IN_USER_LANGUAGE = True

# 0.8-pre32: when the user offers or shares a song that has been loaded ("wanna hear this?",
# "check this out"), the default chat is told it has already heard it, so it gives a brief first
# reaction to the song instead of asking them to play it. A note beside the language note, not part
# of the persona. Games and custom characters are unchanged. False gives pre31's prompt.
CHAT_SHARED_TRACK_ALREADY_HEARD = True


# ==============================================================================
#  4. WIKIPEDIA LOOKUP
# ==============================================================================

ENABLE_WIKI_CONTEXT = True
WIKI_DB_PATH = "music_wiki_heavy.db"
WIKI_DB_TIMEOUT_S = 5.0
WIKI_CONTEXT_MAX_CHARS_PER_ARTICLE = 1800   # per-article cap — enough for lead + a track listing or reception section
WIKI_CONTEXT_TOTAL_MAX_CHARS = 5000         # hard ceiling on the WHOLE block regardless of article count
WIKI_MAX_ARTICLES = 6                       # e.g. song article + album article + artist article, combined
WIKI_SEARCH_ROW_LIMIT = 24                 # FTS window for the body search.
WIKI_EXACT_TITLE_LOOKUP = True
WIKI_EXACT_TITLE_MAX_PER_REQUEST = 8       # cap the lookups; strongest queries come first
WIKI_EXACT_TITLE_MAX_HITS = 2              # per query: keep both the album form and the

# /lookup-search: when a phrase is in several titles, the menu opens while the line is still
# being typed and the chosen title goes back on the line (as /load "" does), to add more
# before Enter; the title chosen there is then looked up without a second menu.
# (added in 0.7-pre66; False gives pre65's behaviour; no change to the analysis or the exports)
LOOKUP_SEARCH_DRAFT_MENU = True

# /wikidelete: mark Wikipedia articles (also with d in /lookup-search menus) and make a
# clean copy of the database without them (see CHANGES IN 0.7-pre63).
# (added in 0.7-pre63; False gives pre62's behaviour; the song and video analysis are unchanged)
WIKI_DELETE_COMMAND = False
WIKI_DELETE_LIST = ""              # the list of marked titles; "" = "<database>-deletions.txt" beside it
WIKI_HIDE_MARKED = True            # marked articles are left out of /lookup and /lookup-search at once

# /lookup-search "Title" ["Title 2" ...] [question]: Wikipedia articles chosen by title;
# a phrase in several titles opens a menu (see CHANGES IN 0.7-pre62).
# (added in 0.7-pre62; False gives pre61's behaviour; the song and video analysis are unchanged)
LOOKUP_SEARCH_COMMAND = True
LOOKUP_SEARCH_MENU_MAX = 0         # titles listed in one /lookup-search menu; 0 = all (pre63; pre62 listed 30)

# /lookup and /lookup-search in any order with /load, /videoload and /musicvideoload.
# (added in 0.7-pre62; False gives pre61's behaviour; the song and video analysis are unchanged)
LOOKUP_WITH_LOAD_COMMANDS = True

# 0.7.1-pre25: /lookup-search on the input line opens its screen for every phrase: a phrase in
# only one article title shows that title on its own (Enter puts it on the line, Esc goes back),
# and a phrase in none shows a screen saying so (any key goes back to the line as typed; Enter on
# the unchanged line then sends it as before). Needs LOOKUP_SEARCH_DRAFT_MENU and a terminal.
# False gives pre24's (one title is added straight away; none is reported when the line is sent).
LOOKUP_SEARCH_ALWAYS_MENU = True

# 0.7.1-pre25: /lookup finds what a question is about in more wordings: "the drummer in X" and
# "the producer of X" (X), "I'd like to know about X", "can you tell me about X" and similar,
# titles with small words in them ("Name in the Sky with Words", "Title in the U.S.A."), "X by Y"
# (X, made by Y), and a loaded song whose album has the same name (the song, unless the question
# says album). The whole phrase is always tried first. False gives pre24's.
LOOKUP_BETTER_SUBJECTS = True

# 0.7.1-pre25: when /lookup's question is about an artist, album or song, an article about
# something else with the same name is left out (an ordinary word's article for "What genre is
# Word?" when a band called Word has one; a song's article when the album is asked about), and of
# two equally good articles the one that names the artist the question or the loaded song is
# about is taken. False gives pre24's.
LOOKUP_PREFER_MUSIC_ARTICLES = True

# 0.7.1-pre25: when /lookup still can't choose between articles, a menu lists them (Esc: none of
# them, as before); with no confident match it mentions /lookup-search. Terminal only. False gives
# pre24's (nothing is added for that name).
LOOKUP_CHOOSE_WHEN_UNSURE = True


# ==============================================================================
#  5. FILE TAGS, SAVING, BATCH AND EXPORT
# ==============================================================================

ENABLE_FILE_METADATA = True
METADATA_MAX_LYRICS_CHARS = 10000
LYRICS_MAX_CHARS = 12000

BATCH_PAUSE_BETWEEN_TRACKS_S = 20.0

# dev65: retry a track once when its analysis process crashes (see
# _batch_run_isolated). False restores dev64.
BATCH_RETRY_CRASHED_ANALYSIS = True

# dev2: optional time limit, in seconds, for each stage of one /batch track
# (each runs in its own process). None (default) = no limit, as before. When
# set, a stage still running at the limit is stopped and the track is
# reported as failed, so one stuck track cannot stall an overnight batch.
# Leave generous room for long tracks and for the first run, which loads
# the listening model from disk.
BATCH_TRACK_TIMEOUT_S = None
BATCH_ISOLATE_PER_TRACK = True
BATCH_SPLIT_IDENTITY_PROCESS = True
BATCH_LOG_RSS = True
BATCH_UNLOAD_OLLAMA = True
BATCH_OLLAMA_KEEP_ALIVE = 0
BATCH_PRE_IDENTITY_SETTLE_S = 5.0

# Untagged files: "Song - Live" isn't read as artist "Song", title "Live".
# (added in 0.7-pre53; False gives pre52's behaviour)
FILENAME_VERSION_GUARD = True

# 0.8: a bare /batch (no folder given) analyses the folder with this name beside the script, when there
# is one; an absolute path works too. "" gives pre35's /batch, which then only shows how to use it.
BATCH_DEFAULT_FOLDER = "batch"

# 0.8: /batch with --cover or --photo says these work on one track at a time (use /analyse), instead
# of reading them as part of the folder name, and the command list says so. False gives pre35's.
BATCH_IMAGE_OPTIONS_NOTE = True

# /export runs the exporter script beside this one (kept as its own file) with the
# same arguments it takes in Terminal; no file or folder means every save in
# saved-songs/, and empty quotes choose songs from a menu.
# (added in 0.7-pre57; False gives pre56's behaviour)
EXPORT_COMMAND = True

# /batch: a track whose listening model produces nothing for BATCH_STALL_MINUTES is
# stopped and analysed again once (it had hung; results are not affected).
# (added in 0.7-pre64; False gives pre63's behaviour)
BATCH_STALL_RETRY = True
BATCH_STALL_MINUTES = 10.0

SAVE_DIR = os.path.join(".", "saved-songs")

# 0.6.2-dev4: /load looks for export .txt files here and in every
# folder inside it (the export script writes to saved-songs/llm-ready/<level>-...
# by default). A full path to a .txt anywhere also works.
EXPORT_SEARCH_DIR = os.path.join(SAVE_DIR, "llm-ready")

# Analysis caches use the file's exact modification time, and a failed loudness measurement is
# not cached or reported as a whole-file reading.
AUDIO_CACHE_FIXES = True

# /batch: a save is "already done" only when it reads back complete; a changed audio file is
# re-analysed; saves with no recorded source are never overwritten. Off by default: older saves
# with no recorded source, and files whose tags were edited since, would be analysed again.
BATCH_SAVE_CHECKS = False


# ==============================================================================
#  6. SPEED, MEMORY, MODEL LOADING AND DEBUGGING
# ==============================================================================

UNLOAD_OMNIZART_AFTER_STEM_MIDI = True

FAST_MODE = False    # safe speedup: skip self-check pass (one fewer listening-model generation)
DEEP_MODE = False     # if True: larger token budgets and two extra analysis categories — more
DEBUG_MODEL_LOADING = False  # True to show raw model-loading log lines instead of a status line
DEBUG_STEM_MIDI = False  # True = verbose per-stem note-count debugging

# Run independent work at the same time instead of one step after another:
#  * signal processing (analysis WAV, objective/beat report, Essentia, PANNs
#    genre/mood, the section map's beat grid and features) and Demucs + the
#    vocal pitch report run in the background while the listening model does
#    its era and main passes on the GPU;
#  * Omnizart transcribes the separated stems in a separate process as soon as
#    Demucs finishes, while the listening model and Whisper keep working (a
#    process rather than a thread: TensorFlow, spleeter and fork-based
#    multiprocessing are not safe to mix with a GPU job in the same process);
#  * the per-stem instrument-tagger windows and the CLAP embedding are
#    computed in the background as well.
# Every piece runs the same code on the same input as before and is used
# exactly as the in-order run would have used it; anything that fails in the
# background is simply redone in order. Excerpt (/listen 0:30-1:00) and URL
# analyses always run in order. False restores dev54's strictly sequential run.
# (added in dev55; speed; results unchanged)
PARALLEL_ANALYSIS = True

# dev68: free the listening model for real before Whisper loads. The batch/analysis
# path kept its own references to the model after unloading it, so the whole
# listening model (~16 GB in bfloat16) stayed in memory while Whisper loaded
# and ran, and until the analysis finished; /listen dropped them only after
# the clean-up had run, leaving the freed memory cached on the GPU. Memory
# only; no result changes. False restores dev67.
RELEASE_MOSS_LOCALS_BEFORE_UNLOAD = True

# dev65: Apple-GPU (MPS) memory clean-up (torch.mps.synchronize/empty_cache and
# the gc sweeps around them) only runs on the main thread. With
# PARALLEL_ANALYSIS the CLAP embedding is computed in a background thread and,
# when it finished, unload_clap() cleared the MPS cache from that thread while
# the listening model was generating on the GPU in the main thread. That race
# can abort the whole process on macOS ("Cannot form weak reference to
# instance ... of class MPSGraph ... over-released", batch "analysis stage
# exited -6"). It is timing-dependent, so the same song can pass one run and
# crash the next. The cache is still cleared by the next main-thread unload;
# no result changes. False restores dev64.
GPU_CLEANUP_MAIN_THREAD_ONLY = True


# ==============================================================================
#  7. LISTENING MODEL (MOSS-MUSIC)
# ==============================================================================

# The listening model: MOSS-Music-8B-Instruct (OpenMOSS, Apache-2.0; Qwen3-8B
# language model + MOSS audio encoder, ~9.1B parameters, ~18 GB in bf16).
# Weights and tokenizer come from Hugging Face; the model code comes from a
# clone of OpenMOSS's GitHub repo (see MOSS_MUSIC_CODE_DIR).
MOSS_MODEL_ID = "OpenMOSS-Team/MOSS-Music-8B-Instruct"

# Optional: a commit hash from the model's Hugging Face page, to pin the exact
# weights you have tested (None = latest).
MOSS_MODEL_REVISION = None

# Folder of your clone of https://github.com/OpenMOSS/MOSS-Music (the folder
# that contains src/modeling_moss_music.py). None = look in the
# MOSS_MUSIC_CODE_DIR environment variable, then a "MOSS-Music" folder next to
# this script, then ~/MOSS-Music. Tested at commit ad107c7d (May 2026).
MOSS_MUSIC_CODE_DIR = None
MOSS_TORCH_DTYPE = torch.bfloat16

# MOSS-Music's processor can put an elapsed-seconds marker into the audio
# token stream every 2 seconds, which is what lets the model give timestamps
# (STRUCTURE, section names). Its from_pretrained() turns them OFF by
# default, so they are switched on explicitly here.
MOSS_ENABLE_TIME_MARKERS = True

# Audio fed to MOSS-Music is cut at this many seconds -- the same 10-minute
# window the previous listening model had (its processor stopped there). It
# also bounds memory: MOSS-Music's audio encoder attends over the whole clip
# at once, which for a 10-minute clip briefly needs a few GB on top of the
# weights (lower this if long tracks run out of memory). Signal processing,
# Demucs/Omnizart and Whisper still use the whole track.
MOSS_MAX_AUDIO_SECONDS = 600

# When MOSS-Music's main pass comes back empty ("does not contain any
# audible ... information"), first retry the same full instructions as a
# system message with a short request beside the audio, before falling back
# to the short recovery prompt (which tends to return a free-form caption without the structured
# GENRE_RANKED / KEY / TEMPO_BPM / CHORDS / STRUCTURE fields). Costs one
# extra MOSS-Music generation, and only on tracks whose main pass failed.
MOSS_EMPTY_RETRY_SYSTEM_PROMPT = True

# If that retry's report stops early (two or fewer of its descriptive fields:
# instrumentation, timbre, mood, vocals...), add one free-form description
# pass so those details are not lost. One more MOSS-Music generation, failed tracks only.
MOSS_EMPTY_RETRY_ADD_DESCRIPTION = True

# End a runaway repeating line in MOSS-Music's main analysis (a chord list
# looping until the token budget runs out) so the report reaches its other
# fields. Leaves every report that does not loop exactly as before.
MOSS_LOOP_GUARD = True

# Show live progress on the status line while MOSS-Music is generating: how
# long it took to read the audio and prompt, then tokens written and speed,
# e.g. "Listening — estimating era... (read in 41s · 96/1024 tokens · 6.2/s)".
# Display only; replies are unaffected.
MOSS_SHOW_PROGRESS = True

# How MOSS-Music picks each word. "sampled" (default) follows Qwen3's advice
# for this kind of model (its text side is Qwen3-8B): always taking the single
# likeliest word ("greedy") makes it fall into endless repetition and drop
# report fields. Sampling is done on the CPU (PyTorch's sampling on Apple
# GPUs returned stray tokens) from a fixed seed, so the same track gives the
# same report again on the same machine. "greedy" restores the old behaviour.
MOSS_DECODING = "sampled"
MOSS_TEMPERATURE = 0.7      # Qwen3's recommended settings (non-thinking mode)
MOSS_TOP_P = 0.8
MOSS_TOP_K = 20
MOSS_SEED = 1234            # change for a different (but again repeatable) draw

# Templated passes (main report, self-check, recovery, era, vocal,
# confirmation): the script writes each field name itself, in the template's
# order, and MOSS-Music writes only the values. Without this, MOSS-Music
# mangles or skips field names (e.g. "NUM_DETECTED_DETECTED..." for
# NUM_DISTINCT_VOICES) and the parsers lose those fields. An early end of the
# reply becomes a line break while fields remain, and the reply ends after
# the last field.
MOSS_GUIDED_FIELDS = True

# Main report and self-check: stop MOSS-Music copying one field's answer
# into the fields after it. With the script writing the field names, a reply
# could fall into pasting the same value into every remaining field
# ("none (0.00s-13.92s)" twelve times on the post-punk test track, "uncertain (based on
# metadata alone, without actual listening)" ten times on a funk test track). While a value's first words repeat the start of an earlier
# field's value, the word that would continue the copy is ruled out, so the
# model writes its own answer. Values of a few words ("yes", "uncertain")
# are never affected; the vocal and confirmation passes, whose tags
# legitimately repeat (LEAD_CATEGORY / LEAD_PROFILE), are not touched.
MOSS_NO_FIELD_COPYING = True

# Apple GPUs: every this many generated tokens (and at the end of each pass)
# release PyTorch's cache of freed GPU memory blocks -- a backstop to the
# allocator's own clean-up (PYTORCH_MPS_LOW_WATERMARK_RATIO, set near the
# imports). Main thread only (see GPU_CLEANUP_MAIN_THREAD_ONLY); live data
# is untouched, so replies are unchanged. 0 = off.
MOSS_MPS_CACHE_TRIM_EVERY = 256

# Longest value one field may run to (tokens) before the script moves on to
# the next field -- a ceiling for runaway lists, far above a normal field. A
# field is also never allowed more than its fair share of the reply's
# remaining token budget, so every field of the template gets written.
MOSS_FIELD_MAX_TOKENS = 300

# Era, vocal and vocal-confirmation passes: True (default) sends MOSS-Music
# the short versions of those three prompts (same questions and the same
# output fields, without the long rule lists, which pushed it towards
# "child"/"adolescent" for high adult male voices and a 2010s era for a 1978
# record). False sends the long versions used before. Every rule applied to
# the replies afterwards is the same either way.
MOSS_SHORT_PROMPTS = True

# dev41 MOSS_RELEASE_REFRESH_LOCALS: with RELEASE_MOSS_LOCALS_BEFORE_UNLOAD,
# also refresh the analysis function's own variable snapshot after dropping
# the listening model. On Python 3.11/3.12, an earlier locals() call in that
# function keeps a snapshot of its variables, so the model was still alive
# when its unload clean-up ran and its memory was only freed later, into the
# GPU cache (Python 3.13 is not affected). Memory only. False restores dev40.
MOSS_RELEASE_REFRESH_LOCALS = True

# A main pass that answers as if it had been given only text ("no genre is
# discernible from the provided data ... details are not available") now
# triggers the existing retries, including the loudness-normalised one for
# quiet masters. Only affects new scans. False restores the dev39 behaviour.
# (added in dev41)
MOSS_RETRY_ON_NON_ANSWER = True

# dev60: one shape for the listening model's report. Its self-check reply (the
# text the saved analysis opens with) comes back in one of three shapes:
#  * the compact fields it was asked for ("The duration of the piece is ... /
#    Genre_Ranked=1) ... / KEY= / TEMPO_BPM= / STRUCTURE=[...]");
#  * the same, cut short: it got stuck repeating its genre list (a funk test track
#    257 times, another self-check-loop test scan 166) until it ran out of room, so
#    KEY, TEMPO and STRUCTURE never came;
#  * a prose description with no fields at all, when it slips into its
#    captioning habit (that stuck-pitch scan, the arena-rock scan, one multi-chorus test scan's dev54 scan).
# Only the first gives the genre, key and tempo checks the model's own values.
# Set True to (1) start its reply with "GENRE_RANKED=1)" so it continues in the
# field format; (2) cut a runaway line in it and in the main pass -- including,
# since dev61, a numbered list repeating with changing numbers ("6) rock;
# 7) pop; ...", which the guard missed, so dev60 fell back on a funk test track
# a funk test track and another self-check-loop test scan); (3) add any of GENRE_RANKED, KEY, TEMPO_BPM,
# STRUCTURE... that the self-check reply still lacks from the first pass
# (marked as such); (4) spell the field names one way and drop the
# caption-style "The duration of the piece is ..." opener; and (5, dev63) save
# every analysis in one fixed layout -- a CHECKS block, then the same 18 fields
# in the same order ("uncertain" where none was given; STRUCTURE = the measured
# map), then DESCRIPTION= and REPORT_NOTE= (see CHANGES IN 0.4-dev63). It changes
# the listening model's text (and so possibly the key, tempo or genre evidence)
# on tracks that came back cut short or as prose, and this build could not be
# run against the real model, so it is off until you have compared rescans.
# False keeps dev59's behaviour.
MOSS_CONSISTENT_REPORT = False

# Era (dev55):
#  * era questions that are NOT blind ("what era is this from?") now also get
#    the ERA EVIDENCE measured from the audio (it was only built for blind
#    questions), plus a note that a year tag can be a compilation's or
#    reissue's year (one post-punk structure test scan is tagged 1988 from a compilation; the single
#    is 1979) -- the background article's original date wins.
#  * a sub-bass-heavy low end counts as a late-2000s marker only on a very
#    loud master (-9 LUFS or louder): 1990s kick-led dance music (a Eurodance act,
#    -10.2 LUFS, 30% sub-bass) measured "late 2000s or later".
#  * an era pass that gave no ERA_ESTIMATE (6 of 8 test scans: a genre/
#    instrument description instead) is labelled as such in the notes, so its
#    "modern"/"1970s arena rock" wording is not read as a dating.
# False restores dev54.
ERA_DEV55 = True

# 0.7-pre3/pre4 (Stage 1) LISTENING DETAILS: extra MOSS-Music passes run after
# all the existing ones -- the whole track (how it's played and produced) and
# even 30-second windows across the song (what plays there, energy, what
# stands out). Words about the singer's sex or age are taken out of what is
# saved. Saved in the .json as "listening_details" only; nothing in the
# analysis reads it, so every existing result is unchanged. False: no extra
# passes, no new field.
ENABLE_LISTENING_DETAILS = True
LISTENING_DETAILS_SECTIONS = True      # False: whole-track pass only (no windows)
LISTENING_DETAILS_WINDOW_S = 30.0      # length of each window
LISTENING_DETAILS_MAX_WINDOWS = 16     # 16 x 30 s covers 8 minutes; longer tracks: an even spread
LISTENING_DETAILS_CLIP_MIN_S = 4.0     # a shorter last window joins the one before it

# 0.7-pre5: only voice delivery/production, effects and playing style (no
# instrument lists or energy); unresolved vocal presence is treated as no voice.
LISTENING_DETAILS_NARROW = True
LISTENING_DETAILS_DROP_BLANK = True    # a reply blank in every field is not saved

# Listening details: a second LABEL=value on the same line is its own field.
LISTENING_FIELDS_INLINE_SPLIT = True

# 0.7.1-pre28: when a listening-model reply is judged blank and dropped, the save keeps its first
# 240 characters (sex and age words taken out as usual) beside the "blank" mark, so a run of blank
# replies can be looked into. Never shown in exports or to the chat. False: not kept (pre27).
LISTENING_KEEP_BLANK_EXCERPT = True


# ==============================================================================
#  8. LYRICS TRANSCRIPTION (WHISPER) AND TAG LYRICS
# ==============================================================================

# dev40 WHISPER_PROGRESS: the Whisper stage shows each decoding pass as it
# runs ("Whisper pass 2: cross-check (sampled)...") and, after loading, after
# each pass and after unloading, a memory readout: this process, the Apple
# GPU memory PyTorch has in use and holds cached, the Omnizart helper process
# when it is running, and the Mac's free memory. Display only -- nothing it
# prints feeds the analysis. False restores dev39's messages.
WHISPER_PROGRESS = True

# dev40 WHISPER_MPS_CACHE_TRIM_EVERY: on Apple GPUs, release PyTorch's cache
# of freed GPU blocks every this many tokens Whisper generates, and after each
# pass -- what MOSS_MPS_CACHE_TRIM_EVERY already does for the listening model.
# Without it the cache could grow towards the allocator's own reclaim level
# (PYTORCH_MPS_LOW_WATERMARK_RATIO, ~29 GB on a 64 GB Mac) while Whisper
# decoded. Main thread only; only freed memory is released, so transcriptions
# are unchanged. 0 = off (dev39 behaviour).
WHISPER_MPS_CACHE_TRIM_EVERY = 256

# dev41 WHISPER_RELEASE_BEFORE_LOAD: just before Whisper loads, and after
# each Whisper pass, the full clean-up the unload routines use (Python garbage
# collection, wait for the Apple GPU to finish, then release its cache) instead
# of releasing the cache alone. dev40's readouts showed 18-27 GB of the
# listening model's freed memory still cached when Whisper loaded; releasing
# the cache alone did not free it, the full clean-up did. Main thread only;
# memory only. False restores dev40.
WHISPER_RELEASE_BEFORE_LOAD = True

WHISPER_MODEL_ID = "openai/whisper-large-v3"
WHISPER_TORCH_DTYPE = torch.float16

SKIP_LYRICS_TRANSCRIPTION_WHEN_TAGS_PRESENT = True

# Lyric timing for the section map: the single Whisper pass over the mix
# matched only 6-29% of the embedded lyrics with usable line timing on every
# saved test scan (one verse-first pop test scan: four "lines", one of them 0:00-2:05 of
# words that are not in the song), and the same file gives the same result on
# every scan. When the pass matches less than WHISPER_TIMING_RETRY_BELOW of
# the embedded lyrics, Whisper is run again on the isolated vocal stem and, if
# that still falls short, on the mix -- both times without the
# no-repeated-4-word-phrase rule, which makes repeated lyric lines ("I need
# somebody, I need somebody") impossible to transcribe within a 30-second
# window. The best-matching result is kept only when it beats the first pass
# by WHISPER_TIMING_RETRY_GAIN; otherwise the first pass is used as before.
# Costs one or two extra Whisper passes, only on such tracks. False restores
# dev52.
# (added in dev53)
WHISPER_TIMING_RETRY = True
WHISPER_TIMING_RETRY_BELOW = 0.35
WHISPER_TIMING_RETRY_GAIN = 0.10
WHISPER_TIMING_MAX_CHUNK_S = 20.0

# Lyric timing from word timestamps. Whisper's line ("segment") timestamps
# collapse on sung audio: every saved test scan has "lines" 30-125 s long
# (one verse-first pop test scan 0:00-2:05, one self-check-loop test scan 0:01-1:46), which leaves
# the section map almost no lyric timing to name verse and chorus by. Word
# timestamps come from the decoder's attention and do not collapse; the words
# are regrouped into lines at pauses. Used for the section map / lyric
# repetition map only -- the lyric TEXT in the report is unchanged.
#  * tracks with embedded lyrics: the timing pass asks for word timestamps
#    first; if that fails or matches the lyrics worse, the dev54 passes run.
#  * other tracks: when the transcription pass's lines are collapsed (a
#    quarter or more of its words in lines over WHISPER_WORD_TIMING_SMEAR_S),
#    one extra Whisper pass times the words.
# Any failure falls back to the dev54 timing. False restores dev54.
WHISPER_WORD_TIMING = True
WHISPER_WORD_TIMING_SMEAR_S = 20.0

# dev36: timing also counts as collapsed when Whisper crams 40 real words or
# more (several lines' worth, not a run of "oh"s) into a line faster than this
# many words a second (one soft-rock scan: 266 words in a line lasting no
# time, 76% of the song), which the 20-second test above could not see.
# 0 restores dev35.
WHISPER_WORD_TIMING_DENSE_WPS = 10.0
WHISPER_WORD_LINE_GAP_S = 0.6
WHISPER_WORD_LINE_MAX_S = 8.0

# dev56: Whisper decodes long audio in overlapping 30 s windows, and with word
# timestamps the overlap is not always merged: the words of the 5-10 s each
# window shares with the next came back twice (one multi-chorus test scan, 0:41
# and 3:27: every word of a line doubled, "a a b b c c"), which made the repeated chorus lines look like new words. When
# the word times run backwards (the next window starting again inside the
# overlap), the earlier window keeps the words before the middle of the
# overlap and the later one the words after it. False restores dev55.
WHISPER_WORD_STRIDE_DEDUPE = True

# dev60: Whisper's anti-hallucination settings on newer transformers. Its
# low-confidence check compares the decoding temperature with 0, and the
# greedy passes never set one, so transformers 5.x stops with "'>' not
# supported between 'NoneType' and 'float'" and every such pass has been
# re-run with plain settings -- in every scan so far. The failed first attempt
# is no longer repeated on every pass (same result, less wasted decoding).
# Setting this True passes temperature 0 so the settings really apply (no
# carrying one bad passage into the next, a repeat blocker, and quiet
# passages skipped as silence instead of filled with invented words). That
# changes the lyric text and timing the section map was tuned on, so it is
# off until tested on your tracks. False keeps dev59's results.
WHISPER_EXPLICIT_TEMPERATURE = False

# dev19: a lyrics tag with fewer than 12 words, or one short line naming the
# artist or title, is not lyrics (_tag_lyrics_usable): the lyrics are
# transcribed instead, and files saved with one are read without it. False
# restores dev18.
TAG_LYRICS_SANITY = True

# 0.7.1-pre21: Whisper is told the song's language once for the whole track, instead of guessing it
# afresh for each 30-second stretch (a stretch where the voice is hard to hear, or has no singing,
# could come out as Welsh or another language). The language is English unless Whisper's own
# language scores, added up over the stretches with singing, are well ahead for another language
# and it is Whisper's confident pick in at least three of every four of them. A song with several
# stretches confidently in English and in another language, or an unclear one, is transcribed as
# before (each stretch guessed). English tag lyrics give English without the check. The decision
# is shown during the scan and kept in the save ("lyrics_language" in "structure_replay").
# False gives pre20's transcription.
WHISPER_HOLD_LANGUAGE = True

# 0.7.1-pre21: with WHISPER_HOLD_LANGUAGE, a language code ("en", "fr", "de" ...) to use for every
# track instead of deciding per track; "" decides per track.
WHISPER_LANGUAGE = ""


# ==============================================================================
#  9. VOCALS
# ==============================================================================

ENABLE_VOCAL_PASS = True
ENABLE_VOCAL_OBJECTIVE_REPORT = True
ENABLE_VOCAL_CONFIRMATION_PASS = True

# Run Demucs vocal-stem separation for pitch tracking even when ENABLE_STEM_MIDI
# is off. Vocal range/pitch accuracy depends heavily on isolating the singer
# before pyin runs (otherwise it can lock onto a bass/pad/guitar drone in the
# full mix) -- that isolation shouldn't be tied to the unrelated instrument
# MIDI-transcription toggle. If ENABLE_STEM_MIDI is already True the stems are
# reused there too, so this adds no extra Demucs runs in the default config.
ENABLE_VOCAL_STEM_ISOLATION = True
ENABLE_INDEPENDENT_F0_MULTI_SINGER_TRIGGER = True

# 7.5, was 6.0. A single singer's verse-to-chorus lift is routinely 3-7
# semitones (the rock test track: verse around B3, chorus around F#4 -- two
# pitch centres exactly 6.0 semitones apart), and at 6.0 that one voice was
# upgraded to "mixed leads / duet" with "do not flatten this to a single
# singer". A male/female split, which this trigger exists to catch when the
# listening pass misses it, is typically 7-12+ semitones.
INDEPENDENT_MULTI_SINGER_MIN_SEP_SEMITONES = 7.5
INDEPENDENT_MULTI_SINGER_MIN_RATIO = 0.25

VOCAL_CONFIRMATION_F0_THRESHOLD = 210.0
VOCAL_CONFIRMATION_WITHOUT_F0 = True
F0_MALE_HIGH_CONFIRM_HZ = 260.0   # male/adolescent lead + median ≥ this → run confirmation

# The two caps below were 280/320 Hz. A sung median of C#4-E4 (280-330 Hz) is
# ordinary for an adult male pop/rock lead in chest or mixed voice -- the rock
# test track's singer sits at 331 Hz -- so those values would have re-labelled
# a typical adult rock tenor "adolescent" (from pitch alone, which cannot show
# age) or "uncertain". The soft cap now equals the hard cap, so pitch alone no
# longer produces an "adolescent" tag; a male tag on a median of F4 or above
# is still sent back to "uncertain".
F0_POST_PUBERTY_SOFT_CAP_HZ = 350.0  # post_puberty_male + median ≥ this → demote to adolescent_male_likely
F0_POST_PUBERTY_HARD_CAP_HZ = 350.0  # post_puberty / adolescent male + median ≥ this → uncertain
F0_MALE_ANY_HARD_CAP_HZ = 350.0   # any male-tagged lead (incl. adolescent) + median ≥ this → uncertain
F0_FEMALE_RECOVERY_MIN_HZ = 320.0  # male-tagged lead + median ≥ this + strong female confirmation → female
F0_CHILD_SOFT_FLOOR_HZ = 180.0    # child_* + median below this → do not keep child on pitch grounds alone

# The pitch-histogram upgrade to "mixed leads" no longer runs when a listening
# pass reported one singer in any spelling ("NUM_DISTINCT VOICES: 1", "single
# lead voice") or when the two pitch centres are an octave apart (octave
# doubling / octave errors). False restores the dev42 behaviour.
# (added in dev43)
VOCAL_F0_UPGRADE_GUARDS = True

# Vocal gender from what the listening model says freely. Its structured vocal
# pass came back "LEAD_CATEGORY: uncertain / gender uncertain, likely male;
# age uncertain, likely adult" -- the prompt's own wording -- on all eight test
# scans, female singers included, so no scan had an acoustic lead profile. Its
# free descriptions in the same scans were right every time they named one
# ("led by a male tenor", "a female lead in a mid-to-high soprano range":
# one self-check-loop test scan, one multi-chorus test scan, another self-check-loop test scan, that stuck-pitch scan, the arena-rock scan; none
# wrong). These sentences are collected from the main and era passes plus one
# short extra question (VOCAL_QUICK_PASS, ~1-2 s of listening-model time);
# when they agree and the structured passes settled nothing, they set the lead
# profile: female -> female_teen_adult; male -> post_puberty_male (only below
# the male pitch cap, and adolescent_male_likely when called teenage). They
# never override a parsed profile, and always go to the singer-identity
# evidence. False restores dev54.
VOCAL_FREE_TEXT_GENDER = True

# dev57: the free descriptions do not set the lead's gender when the dedicated
# vocal pass's own PROBABILITY ESTIMATE line clearly favours the other gender
# (0.4+ against 0.15 or less, and at least 3x): one verse-first pop test scan's main and
# quick passes said "female adult" while that line read "adolescent male 0.3;
# post-puberty male 0.2; female teen/adult 0.1", and FINAL LEAD PROFILE came
# out female_teen_adult. The profile then stays unknown (the singer identity
# step decides, as it did). False restores dev56.
VOCAL_PROBABILITY_VETO = True

# dev59: an unbroken voice. When the listening model's free descriptions call
# the lead a woman but its dedicated vocal pass rates the voice mostly as a
# child's or a teenage boy's (child-like + adolescent male 0.5 or more), the
# gender stays open (VOCAL_PROBABILITY_VETO) but the voice is recorded as
# sounding UNBROKEN -- the usual signature of a boy whose voice has not dropped.
# The singer identity step and the chat both get that: with a documented male
# artist it is his voice before it broke, and when he was already in his mid
# teens (a known teen male artist at 16 on "one verse-first pop test scan") that is worth saying, since
# most boys' voices have dropped by then. dev57 had removed this evidence along
# with the wrong "female" label. False restores dev58.
VOCAL_UNBROKEN_SIGNATURE = True

# dev69: the mirror of VOCAL_UNBROKEN_SIGNATURE. When the singer identity puts a
# boy or teenager on the recording (child/adolescent male, from his age) but
# the audio heard a man's voice (post_puberty_male) and the measured median
# pitch is below VOCAL_DEEPENED_MAX_MEDIAN_HZ, the chat is told his voice had
# already deepened -- instead of the general example sentence, which said
# "the voice still sits high and light" (one teen-male test case, "the later teen-male scan",
# 2001: man's voice, median D#4 / C#4, against F4 / E4 on 2000's "(Have Some)
# the earlier teen-male scan"). Above that pitch it only says the voice may have been
# deepening. Saved files get the same sentence when loaded. Only that one
# combination of identity and audio reading changes; False restores dev68.
VOCAL_DEEPENED_SIGNATURE = True
VOCAL_DEEPENED_MAX_MEDIAN_HZ = 300.0

# dev71: when the pitch tracker's median sits an octave (12 +/- this many
# semitones) above the vocal stem's note-by-note median and the tracker also
# found that lower octave itself, its figure is an overtone: the VOCAL PITCH
# REPORT gives the sung line's median (an octave down), range and voice-type
# map instead, and says so. One test scan (male group, high lead): the
# tracker read E5 (659 Hz), the notes E4 (330 Hz), so the voice was described
# as "boy-soprano"/"falsetto-led". Saved files get it when loaded. False
# restores dev70.
VOCAL_PITCH_OCTAVE_CHECK = True
VOCAL_PITCH_OCTAVE_TOL_ST = 1.5

# SECTION VOICE CHECK: mixed-gender lead vocals. Both vocal passes judge the
# whole track at once, so a male singer who takes only the verses of a song
# led by a woman is missed (a vocal-group test track: male verses at 0:17-0:35
# and 1:10-1:26, read as one female lead). When the lead is a settled adult
# (female_teen_adult or post_puberty_male, with both vocal passes agreeing on
# the gender) and the pitch histogram shows two centres 4+ semitones apart
# (not an octave), the separated vocal track is pitch-tracked in 8-second
# windows; stretches of 12+ seconds sitting 4+ semitones below a female lead
# (at or below E4) or above a male lead (at or above C4) are each played to
# MOSS-Music on their own -- a clip of the full mix -- with the scan's own
# singer-confirmation question, as is a stretch of the main lead. (A first
# version played isolated vocal-stem clips back to back with a new question;
# on a dance-pop test track: MOSS-Music answered "unclear" to all of it.) The lead
# becomes mixed_leads only when the main stretch is heard as the lead's gender
# and 2+ stretches (or one of 24+ seconds, with high confidence) as the other
# gender, at medium or high confidence (high only, for a female stretch in a
# male-led track). Young, child, uncertain and mixed leads are never checked, and a
# second singer of the same gender (harmonies, a duo) changes nothing. Costs
# one pitch-tracking pass (~10-20 s) on tracks passing the first checks, and
# one short MOSS-Music answer per stretch. False turns it off.
SECTION_VOICE_CHECK = True

# Additional coverage checks use full-mix clips even when whole-track passes agree.
# Pitch reliability is not a prerequisite. Section hints select clips, not identities.
SECTION_VOICE_CONFLICT_RECOVERY = True
SECTION_VOICE_COVERAGE_CHECK = True
SECTION_VOICE_RECOVERY_MAX_CLIPS = 12

# Extra disjoint six-second samples only after conflicting clips lack repetition.
# Set to 0 to restore the v3 sampling behaviour.
SECTION_VOICE_RECOVERY_GAP_CLIPS = 8

SECTION_VOICE_RECOVERY_CLIP_S = 12.0

SECTION_VOICE_MIN_SHIFT_ST = 4.0
SECTION_VOICE_MIN_REGION_S = 12.0
SECTION_VOICE_SINGLE_REGION_MIN_S = 24.0
SECTION_VOICE_CLIP_S = 15.0
SECTION_VOICE_MAX_REGIONS = 3

VOCAL_QUICK_PASS = True

# A second, independent pitch reading from the vocal stem's transcribed notes
# (Omnizart), added as VOCAL NOTE PITCH next to the pyin reading and given to
# singer identity. The pyin reading is labelled low-confidence when it rests
# on under 5% voiced frames or spans under a semitone (one post-punk structure test scan: 0.2%;
# that stuck-pitch scan: 195-197 Hz). Neither reading is changed. False restores dev54.
VOCAL_NOTE_PITCH_LINE = True

# 0.7-pre5: a section voice check whose other-sex clips all had backing
# harmonies, and whose direct comparison heard the same singer, is one lead.
SECTION_VOICE_HARMONY_CLIPS = True

SANITIZE_VOCAL_CATEGORY_LANGUAGE = True

# dev20: the low-confidence pitch check (VOCAL_PITCH_LOW_CONFIDENCE_CHECK)
# compares figures the notes write rounded to whole hertz, so E4 (329.6 Hz,
# written "330") and G4 (392 Hz) came out 2.98 semitones apart and the synth-rock test track's
# tracker median G4 -- from 4% voiced frames, marked LOW -- was kept against
# the note reading's E4 ("very high for a man"). A distance within
# PITCH_CHECK_ROUNDING_TOL_ST of the threshold now counts. Load time and new
# scans; of the 21 saved scans tested only the synth-rock test track changes. False restores dev19.
PITCH_CHECK_ROUNDED_FIGURES = True
PITCH_CHECK_ROUNDING_TOL_ST = 0.1

# Minimum share of scanned frames that must be confidently voiced before a
# pitch reading counts as evidence of a SINGER. Separation residue (a lead
# guitar or synth leaking into the vocal stem of an instrumental) can produce a
# handful of voiced frames; a real vocal part produces far more.
VOCAL_PITCH_EVIDENCE_MIN_VOICED_FRACTION = 0.08

# Vocal stem this far below the mix (RMS) is separation residue, not a part.
VOCAL_ARBITRATION_STEM_BELOW_MIX_DB = 20.0

# ...and this far below with almost no voiced frames is decisive on its own.
VOCAL_ARBITRATION_STRONG_BELOW_MIX_DB = 30.0
VOCAL_ARBITRATION_STRONG_MAX_VOICED = 0.04

# Faint vocal stem (14 dB: halfway). Real voices in 52 saved scans sit 1-11 dB below the mix
# (a test track with one sung verse, at 10.4; a classic-rock test track at 11.2);
# the glitch-electronic test track's vocal stem -- a sampled vocal texture, no sung lead -- sat 16.8 dB
# below, between those and the 20 dB "weak" mark, and the old build called it
# a male lead. (1) With the main pass saying instrumental, a stem this far
# below the mix now counts as weak. (2) Whatever the listening passes said, a
# stem this far below the mix from which Whisper could transcribe almost no
# words (and with no tagged lyrics) is ruled instrumental, noting that faint
# vocal-like textures may be present. False turns both off.
FAINT_VOCAL_TEXTURE_CHECK = True
VOCAL_ARBITRATION_FAINT_BELOW_MIX_DB = 14.0
FAINT_VOCAL_MAX_WHISPER_WORDS = 40

# The vocal stem within this many dB of the mix, with the dedicated vocal pass
# hearing a voice, keeps the main pass's "instrumental" from deciding on its
# own (see _arbitrate_vocal_presence). False restores the previous rule.
VOCAL_ARBITRATION_LOUD_STEM_GUARD = True
VOCAL_ARBITRATION_LOUD_STEM_DB = 12.0

# dev73: the main pass's "instrumental" is not taken as settled when another
# listening pass heard singing (see _other_pass_heard_singing). False
# restores dev72.
VOCAL_ARBITRATION_HEARD_LYRICS_GUARD = True

# dev73: a pitch tracker reading that collapsed to one pitch. False restores dev72.
VOCAL_PITCH_STUCK_CHECK = True

# A pitch-tracker reading marked LOW confidence (see _pyin_reading_confidence_note)
# that disagrees with the vocal stem's note reading by this many semitones or
# more is replaced by the note reading in the VOCAL PITCH REPORT. False
# leaves the report as the tracker read it.
VOCAL_PITCH_LOW_CONFIDENCE_CHECK = True
VOCAL_PITCH_LOW_CONFIDENCE_MIN_ST = 3.0

# (added in 0.7.1-pre12; audit fixes; False gives pre11's behaviour)
CLIP_ANSWER_TOLERANT = True         # clip answers read past formatting ("different lead", "Yes.", a bullet, a note in brackets)
VOCALS_FLAG_FALLBACK_STRICT = True  # "no vocal harmonies" / "no vocals in the intro" is not read as an instrumental


# ==============================================================================
#  10. SINGER IDENTITY
# ==============================================================================

# Singer identity prompt sized to fit its context (it was cut in /batch), track
# evidence and answer format last, background looked up by artist and title
# (not album). False restores the dev39 behaviour.
# (added in dev40)
SINGER_IDENTITY_FIT_PROMPT = True

# Show the parsed SINGER IDENTITY RESOLUTION block to the writer (an "already
# present" check had hidden it), and re-resolve on /load a saved identity that
# never parsed. False restores the dev39 behaviour.
# (added in dev40)
SINGER_IDENTITY_TO_WRITER = True

# Identity also says whether the lead is shared by several singers (a vocal
# group, duo or featured artist), and the writer then says so even when the
# audio passes counted one voice. False restores the dev39 behaviour.
# (added in dev40)
SINGER_IDENTITY_GROUP_LEADS = True

# Singer identity says several lead singers only for a documented group, duo
# or featured singer -- never from the audio's mixed-leads or pitch-centre
# reading alone. False restores the dev42 identity prompt.
# (added in dev43)
IDENTITY_SOLO_STAYS_ONE = True

# dev58: the singer-identity step sets the lead's gender from a documented solo
# artist even when the audio-only reading is unknown or was left open. Its
# instructions said both "a documented solo artist SHOULD set the category" and
# "uncertain -> do not invent a gender from metadata", and the second could win
# -- more often since dev57 leaves a contradicted reading open (a multi-chorus test track
# Love). Bands with no documented lead singer still go by the listening
# model's voice descriptions. Gender and age only: never style or genre.
# False restores dev57.
IDENTITY_FROM_ARTIST_WHEN_OPEN = True

# dev18: a boy or teenager (child/adolescent male identity) whose voice the
# first vocal pass heard as a woman's, where only the confirmation pass heard
# a man's voice and the measured median pitch is at or above
# VOCAL_DEEPENED_MAX_MEDIAN_HZ, is described as his unbroken voice -- the
# same reading a female first pass gets on its own -- instead of "may already
# have been deepening". MOSS-Music's confirmation pass called 15-year-old
# the teen-pop test track (median E4, F#4 note by note) "post_puberty_male
# ... a mature voice" against its own first pass's "female_teen_adult" (95%),
# and chat replies called him a woman or a man with a deepened voice. Saved
# files get the same sentence when loaded. Nothing changes without a young
# male identity, or below that pitch (the dev69 deepened voice). False
# restores dev17.
YOUNG_VOICE_CONFIRM_FLIP = True

# 0.7-pre5: a credited solo male heard as female-sounding, with a young person
# on the cover and nothing documenting an adult, is adolescent_male_likely.
YOUNG_MALE_VOICE_CHECK = True

# 0.7-pre9: clips that heard both sexes on a one-act credit whose cover shows one
# teenage boy (identity answer male or uncertain, never female) give one young
# male lead instead of "uncertain".
YOUNG_SOLO_MALE_CLIPS = True

# dev46 IDENTITY_FLIP_NOTE: when the first vocal pass and the listening
# model's own voice descriptions heard a female (or unbroken) voice and only
# the confirmation pass heard an adult man, the singer-identity step is told
# not to defer to that male reading and to decide from the tags, cover art and
# knowledge of who sings. Every other track's identity prompt is unchanged.
# Takes effect on a new scan. False restores dev45.
IDENTITY_FLIP_NOTE = True

ENABLE_SINGER_IDENTITY_RESOLUTION = True

# Keep the text model resident across checks within one batch identity stage.
# The stage still unloads it in finally before subsequent heavy audio work.
# False restores v10 request-by-request unloading.
SINGER_IDENTITY_REUSE_BATCH_MODEL = False
SINGER_IDENTITY_NUM_CTX = 8192

# dev19: a featured singer credited in the artist or title tag ("feat.",
# "ft.", "featuring") makes LEAD_SINGERS=several when the identity step said
# one (_feature_credit_identity). False restores dev18. With IDENTITY_SHARED_LEAD_EVIDENCE on
# (0.7.1-pre6), only when the reasoning gives explicit shared-lead evidence.
TAG_FEATURE_SHARED_LEAD = True

# dev20: a singer identity that says LEAD_SINGERS=one while its own reasoning
# documents the act as a vocal group (boy band, girl group, brothers, sisters,
# a duo ...) and names no single lead singer is set to several, the identity
# prompt's own rule (_group_lead_identity). the sibling-band test track came out as several in dev17
# and one in dev19 from the same facts, worded differently. A band with a named
# frontman, or a solo artist, is left alone. New scans and /load. False
# restores dev19. With IDENTITY_SHARED_LEAD_EVIDENCE on (0.7.1-pre6), only with explicit
# shared-lead evidence.
GROUP_LEAD_STAYS_SEVERAL = True

# dev20 (patch): a credited male whose voice had not broken. The identity
# model sometimes keeps post_puberty_male for a solo male lead the listening
# model heard as female, letting the structured audio profile outrank the
# release-era evidence its own prompt asks for. When the identity says
# post_puberty_male / adult_male, its reasoning records a female-sounding read,
# there is one lead and no featured credit, and the reasoning shows he was 16 or
# younger at release (a birth year against the tagged release year, a stated
# age, or wording such as "teen-pop debut" / "voice had not broken"), the
# category becomes adolescent_male_likely (child_male_likely at 11 or younger),
# with the reason. For a solo act with no background on the artist, a cover
# showing one male-presenting person under the credited artist's name gives
# the age (child or teenage presentation). With no age evidence, or an adult
# age, nothing changes: a
# high adult male voice also reads as female. The same setting makes a young
# male identity whose reasoning records a female-sounding or unbroken voice
# keep that in the SINGER IDENTITY note (and in the chat view of older
# saves), instead of the "had already deepened" / "may have been deepening"
# sentence taken from the audio-only male profile. New scans and /load. False
# restores the earlier behaviour.
UNBROKEN_VOICE_IDENTITY_CHECK = True
PRESERVE_UNBROKEN_COLEAD_IDENTITY = True

# The singer-identity step is told when artist/title were guessed from the filename.
# (added in 0.7-pre53; False gives pre52's behaviour)
IDENTITY_FILENAME_NOTE = True

# 0.8-pre35: when an identity rule's evidence came from the --photo image or from the people a music
# video shows singing, its sentence in the saved identity says so ("the photo of the artist shows one
# young boy") instead of "the cover". False gives pre34's wording.
IDENTITY_RULE_SOURCE_WORDING = True

# Who's singing: ages worked out from a birth year bound to the singer's name, the voice of a
# single young male lead described as unbroken / not fully deepened / deepened, a former or
# guest singer never named as the lead. A young male the 0.7 rules found (a boy heard as
# female-sounding) is never turned into an adult male unless documented as an adult.
# (added in 0.7.1-pre6; False gives 0.7's behaviour)
SINGER_IDENTITY_PRE6 = True

# 0.7.1-pre7: the young-male voice check also accepts a whole-track pass that heard a
# female-sounding voice at 75-89% (it otherwise needs 90%) when at least 4 confidently sampled
# clips were heard and at least 80% of them were female-sounding. All its other conditions still
# apply (one credited solo male, one teenager on the cover, no falsetto, no documented adult).
# Needs SINGER_IDENTITY_PRE6. False gives pre6's behaviour.
YOUNG_MALE_CLIP_CORROBORATION = True

# 0.7.1-pre9: the young-male voice checks read the whole-track pass's female share however the
# pass words it ("adult female 95%" as well as "female teen/adult 95%"); before, any other wording
# was missed and the checks never ran. Needs SINGER_IDENTITY_PRE6. False gives pre8's behaviour.
FEMALE_READ_ANY_WORDING = True

# 0.7.1-pre9: the young-male voice check also runs when there is no cover art, or none showing
# a person (common for lesser-known artists), on stronger evidence: a whole-track pass heard as
# female at 90%+, at least 6 confidently sampled clips, 90%+ of them female-sounding, and a median
# pitch of 250 Hz or more. It also accepts an adult_male identity as well as post_puberty_male.
# A cover showing an adult or several people never qualifies. False gives pre8's behaviour.
YOUNG_MALE_NO_COVER = True

# (added in 0.7.1-pre12; audit fixes; False gives pre11's behaviour)
YOUNG_MALE_NEEDS_MALE_EVIDENCE = True   # no young-male result over a female-presenting cover or a name-only gender with no male read
YOUNG_MALE_FALSETTO_HEARD_ONLY = True   # only "falsetto"/"head voice" the listening passes heard (not Musiclyse's own wording, not "no falsetto") stops the check
YOUNG_MALE_MULTI_CREDIT_STOP = True     # "A & B", "A, B", "A + B", "A / B" credits stop the young-male check
STATED_AGE_GUARDED = True               # "was 95% female-sounding" is not a stated age of 95
DOCUMENTED_ADULT_NAME_FORMS = True      # birth years given with a middle name, a birthplace, or the surname alone
YOUNG_MALE_CLIPS_LEAD_ONLY = True       # clips heard as a duet or group don't count as one young voice
IDENTITY_CANONICAL_REPLY = True         # the identity reply's fields are tidied to exact KEY=value form before the rules read them
IDENTITY_KEEP_LAST_GOOD = True          # if a rule step fails, keep the identity found so far instead of blanking it
ADOLESCENT_TAG_NEEDS_NOT_FEMALE = True  # an "adolescent female" reading in the vocal passes stays female, never a young male

# (added in 0.7.1-pre12; audit fixes; False gives pre11's behaviour)
FEATURE_CREDIT_KEEP_ACT_NAMES = True  # "feat. X and the Y" is one act; a title's "(with X)" credits a guest singer
REASONING_STOPS_AT_KNOWN_FIELDS = True  # a wrapped reasoning line starting "Note:" stays part of the reasoning
CHILD_FLIP_EVIDENCE_PROSE_ONLY = True  # a male voice becomes a child only on the reply's own words, not its template labels or "no prepubertal traits"
AGE_GUARD_LEAD_FIELDS_ONLY = True   # "teen" in a backing-vocal note or the "female teen/adult" label never makes an adult man adolescent
PROBABILITY_LINE_ANY_FORMAT = True  # the vocal pass's probability line is read in percentages or decimals, in any common layout

# (added in 0.7.1-pre13)
FEATURED_SINGER_NOT_LEAD = True     # a female identity crediting a documented featured singer, on a solo credit whose
                                    # cover shows one young male, becomes the credited boy's unbroken voice (guest kept)
SINGER_DETAIL_LOOKUP = True         # after the identity is settled, one short extra request for documented facts only:
                                    # who sings lead on this recording, their birth years, and whether a group has men
                                    # and women. It never changes the identity; it adds SINGER_DETAIL_* lines.

# (added in 0.7.1-pre15)
FEATURED_SINGER_ONE_VOICE = True    # FEATURED_SINGER_NOT_LEAD also covers a shared-lead answer when the listening passes
                                    # heard one female-sounding voice and the clips confirmed no second lead
SINGER_DETAIL_STATUS_LINE = True    # the singer lookup adds SINGER_DETAIL_STATUS=<why nothing, or only part, was added>

# 0.7.1-pre17: when the singer lookup's reply comes back empty (a model that thinks before answering
# can use up its whole reply allowance thinking), it is asked once more with thinking turned off.
# A reply that arrives the first time is never asked again. False gives pre16's behaviour.
SINGER_DETAIL_RETRY_WITHOUT_THINKING = True

# 0.7.1-pre19: keeps a record of the singer lookup in the save, for checking why it added
# nothing (or only part): the chat model's answer to each request, the tokens each answer used
# against its allowance, how long each took, and the background text it was given. It goes in
# a separate "singer_detail_lookup" field of the .json, which the exporter leaves out of exports
# (EXPORT_SKIP_LOOKUP_RECORD), and an empty first answer's SINGER_DETAIL_STATUS line says how
# many tokens it used.
# False: nothing is recorded, as in pre18.
SINGER_DETAIL_DIAGNOSTICS = False

# 0.8-pre31: FEATURED_SINGER_NOT_LEAD is held back when the reasoning gives the featured guest a part
# other than singing the lead (on guitar, producer, backing vocals, a rap verse), or when anything
# says the credited artist is a woman (the singer lookup's lines, a plain statement in the
# reasoning, or the artist's own Wikipedia article). False gives pre30's rule.
FEATURED_SINGER_SAFEGUARDS = True

# 0.8-pre31: identity rule steps run one by one: a step that fails is skipped with a note, and
# the steps after it (the age and young-male checks among them) still run. False gives pre30's
# (a failing step ends the rule steps; IDENTITY_KEEP_LAST_GOOD then decides what is kept).
IDENTITY_STEPS_SEPARATE = True

# 0.7.1-pre20: the singer lookup is given the background as plain articles: the song's own
# article first (when its opening names the credited act), then the act's own article, then the
# rest, each once, without the chat's notes on how to use the background. A long act article no
# longer pushes the song's article (which usually says who sings lead) out of the lookup's
# 6000 characters. The names in the answer are still checked against the whole background, the
# identity step's background is unchanged, and with no background the request is as before.
# False gives pre19's lookup request.
SINGER_DETAIL_BACKGROUND_SONG_FIRST = True

# 0.7.1-pre11: a young-male single lead becomes "uncertain" (as 0.7 already does for an adult or
# female lead) when confident clips heard both a male and a female lead and the cover shows a
# group; a group's mixed clips are also never read as one voice that hadn't fully deepened.
# Solo singers are unaffected. False gives pre10's behaviour.
YOUNG_LEAD_GROUP_CHECK = True

# 0.7.1-pre10: with no cover picture, the young-male check also needs a whole-track pass that heard
# a male voice, since "male" may otherwise rest on the artist's first name alone (a woman with a
# typically male name must not become a boy). False gives pre9's behaviour.
YOUNG_MALE_NO_COVER_MALE_READ = True

# 0.7.1-pre10: the singer-identity step is told a clearly gendered first name is only weak evidence
# (never a unisex, country-dependent or stage name) and to mark it with GENDER_BASIS=name_only.
NAME_GENDER_NOTE = True

# 0.7.1-pre8: the singer-identity step is told about bands with more than one lead singer who
# take different songs (it names this song's singer only when that is documented, otherwise it
# lists them as LEAD_SINGER_CANDIDATES rather than guessing or calling it a shared lead), and may
# name a group's lead singers (LEAD_SINGER_NAMES) when they are documented. False gives pre7's.
MULTI_LEAD_BAND_RULE = True

# A feature credit or group membership counts as shared lead vocals only when the identity
# reasoning says the leads are shared. Off by default: with it on, a featured rapper or guest
# singer (for example on a young singer's track) can be missed as a second lead.
IDENTITY_SHARED_LEAD_EVIDENCE = False


# ==============================================================================
#  11. COVER ART
# ==============================================================================

COVER_IMAGE_MAX_DIMENSION = 512
COVER_IMAGE_QUALITY = "2"
MAX_COVER_BYTES_TO_SEND_RAW = 1_000_000
COVER_IMAGE_SENDABLE_MIMES = {
    "image/jpeg",
    "image/png",
    "image/gif",
    "image/webp",
}

ENABLE_COVER_ART_DESCRIPTION = True

COVER_ART_DESCRIPTION_NUM_CTX = 8192

# (added in 0.7.1-pre12; audit fixes; False gives pre11's behaviour)
COVER_FIELDS_TOLERANT = True        # cover-art answers with bold labels or bullets are read; PEOPLE "a group of five" counts as a group


# ==============================================================================
#  12. SECTION MAPS
# ==============================================================================

STRUCTURE_BOUNDARY_SNAP_TOLERANCE_S = 6.0
STRUCTURE_INTRO_MAX_S = 25.0
STRUCTURE_OUTRO_MAX_S = 30.0
STRUCTURE_BRIDGE_MIN_POS = 0.5
STRUCTURE_BRIDGE_MAX_POS = 0.92
STRUCTURE_INSTRUMENTAL_MAX_COVERAGE = 0.15
STRUCTURE_VOCAL_TRACK_MIN_COVERAGE = 0.35

# Structure v2 (bar-level multi-cue segmentation; see "SONG STRUCTURE v2").
# Set STRUCTURE_V2_ENABLED = False to fall back to the previous pipeline.
STRUCTURE_V2_ENABLED = True

# Minimum section length in bars (fragments shorter than this are fills or
# turnarounds, not sections). The absolute floor in seconds also applies.
STRUCTURE_MIN_SECTION_BARS = 4
STRUCTURE_MIN_SECTION_S = 6.0

# Sections longer than this are allowed but discouraged by the length prior.
STRUCTURE_SOFT_MAX_SECTION_S = 50.0

# Cost of opening a section. Higher = fewer, longer sections.
STRUCTURE_SECTION_COST = 0.6

# Mild preference for sections that are a multiple of 4 bars long. Lower it
# (e.g. 0.08) for songs with irregular phrase lengths, where it can pull a
# boundary a bar onto the "tidier" count; 0 loses real boundaries that only
# the phrase length supports (an AI-generated test track's Chorus 1 at 0:37 and, with lyrics,
# its Bridge at 1:53). 0.15 tested best overall.
STRUCTURE_PHRASE_PENALTY = 0.15

# Similarity (0..1) above which two segments are treated as the same material.
STRUCTURE_SAME_MATERIAL_SIM = 0.60

# A segment counts as sung when this fraction of it carries vocal notes.
STRUCTURE_SUNG_MIN_COVERAGE = 0.15

# Match lyrics between sections by shared three-word phrases (robust to the
# way Whisper merges and splits lines) and let clearly repeated words decide
# chorus vs verse. False restores the dev37 behaviour.
STRUCTURE_LYRIC_PHRASES = True

# Tracks whose file tags carry the lyrics skip the Whisper lyrics pass, which
# left their section map with no lyric timing at all. One Whisper decode now
# times the tag lyrics for the section map only (the report's lyric text is
# still the tags). Costs one Whisper pass on those tracks; False skips it.
STRUCTURE_TIME_TAG_LYRICS = True

# Separate a multi-bar instrumental stretch (riff, solo, break) at the start or
# end of a verse/chorus from it, and call a part with the verse's or chorus's
# music but no voice "Instrumental". False restores the dev37 behaviour.
STRUCTURE_SPLIT_INSTRUMENTAL = True

# A part named after the verse or chorus that has no voice in it at all is
# also Instrumental, however short (from 2 bars; a riff over the chorus chords
# between an opening chorus and the first verse). False restores the dev38
# behaviour.
STRUCTURE_SHORT_INSTRUMENTAL_PARTS = True

# A sung first section shorter than this (seconds) joins the sung section after
# it: the boundary search allows a two-bar first section for pickups, and on a
# chorus-first song that came out as "Verse 1 0:00-0:04, Chorus 1 0:04-0:22".
# Likewise a sung last section this short (a final line ringing out) joins the
# section before it instead of being an "Outro". 0 restores the dev38 behaviour.
STRUCTURE_SHORT_LEAD_S = 8.0

# Two extra lyric checks when choosing the chorus: a part heard two or more
# times fewer than another part that also repeats its words is not the chorus
# (a twice-sung bridge), and an occurrence that sounds like the verse but sings
# the chorus's words is a chorus (a stripped-down opening chorus). False
# restores the dev38 behaviour.
STRUCTURE_LYRIC_CHORUS_CHECKS = True

# A chorus sung twice back to back (a double chorus) is listed as two
# choruses when each half is as long as the song's other choruses, instead
# of one chorus twice their length. False restores the dev38 behaviour.
STRUCTURE_SPLIT_DOUBLE_CHORUS = True

# Embedded lyrics written in stanzas (verse / chorus / bridge blocks): a section
# start that the arrangement put a few seconds before the words of a new
# stanza (a build or fill under the bridge's last line) moves to where that
# stanza is sung. False restores the dev38 behaviour.
STRUCTURE_TAG_STANZAS = True

# dev56: the other direction. A section start that the boundary search put
# more than a bar (2.5 s or more) AFTER the first line of a stanza (the words of the file's own
# lyrics, timed by Whisper itself at the start of a line) moves back to the bar
# where that line starts, when nothing of another stanza is sung in between
# (one multi-chorus test scan's second chorus, sung from 1:38,
# started at 1:41). At most STRUCTURE_STANZA_BACKSHIFT_S seconds. False
# restores dev55.
STRUCTURE_STANZA_BACKSHIFT = True
STRUCTURE_STANZA_BACKSHIFT_S = 7.5   # dev26: was 4.5

# A question about the song's structure, form, sections or layout is answered
# with every section's name and its start and end time; other questions still
# get no section list or times. False restores the dev39 behaviour.
# (added in dev40)
STRUCTURE_ANSWERS_WITH_TIMES = True

# Save the section map's inputs (beat grid, per-beat mix features, timed lyric
# lines, the listening model's name hints) in the JSON as "structure_replay",
# so a wrong map can be re-run and a fix tested from the JSON alone
# (replay_saved_structure). Adds roughly 150-250 KB per JSON; changes nothing
# else. False restores the dev43 behaviour.
# (added in dev44)
STRUCTURE_SAVE_REPLAY = True

# Verse and chorus over the same music: when the section map put them in one
# cluster (so "verse + chorus" was named "Chorus"), and the words clearly
# separate them -- new words each time vs. words that come back -- the
# new-words parts are treated as their own material. Only when no other
# recurring part already reads as a verse. False restores the dev44 behaviour.
# (added in dev45)
STRUCTURE_SPLIT_SHARED_MUSIC = True

# A short hook line sung after each chorus that repeats its words more
# exactly than the chorus no longer becomes the "Chorus": a word-repeating
# part at least 1.5x as long in total (and coming round as often) is the
# chorus, and such a tag is named Post-Chorus where it follows a chorus and
# Outro after the last one. False restores the dev45 behaviour.
# (added in dev46)
STRUCTURE_CHORUS_PREFERS_LONGER = True

# When no part's words stand out pair by pair and the chorus is picked by
# level/register, a part whose words mostly come back is preferred over one
# whose words mostly don't. False restores the dev45 choice.
# (added in dev46)
STRUCTURE_CHORUS_WORDS_FALLBACK = True

# Digital silence: frames this far below the track's loud level, for at least
# STRUCTURE_SILENCE_MIN_S, become their own "Silence" section at either end.
# (added in dev46)
STRUCTURE_SILENCE_REL_DB = 50.0
STRUCTURE_SILENCE_ABS_DB = -60.0
STRUCTURE_SILENCE_MIN_S = 3.0

# 0.7-pre59: digital silence in the middle of a song -- the sound exactly zero (not
# a fade or a quiet passage) for at least STRUCTURE_MID_SILENCE_MIN_S, at least
# STRUCTURE_MID_SILENCE_EDGE_S from the music's start and end -- becomes its own
# "Silence" section; a piece of a section it cuts that would be shorter than
# STRUCTURE_MID_SILENCE_LEFTOVER_S joins the section on its other side. False gives
# pre58's maps (a silence inside a section is part of it).
STRUCTURE_MID_SONG_SILENCE = True
STRUCTURE_MID_SILENCE_MIN_S = 1.5
STRUCTURE_MID_SILENCE_EDGE_S = 5.0
STRUCTURE_MID_SILENCE_LEFTOVER_S = 4.0
_MID_SILENCE_BELOW_DB = 140.0     # below the track's loud level: nothing at all is sounding

# Weight of whole-stem-profile novelty. Off by default: per-stem note
# density changes at every phrase, and the discrete stem events (fills,
# entries, vocal register) carry the useful part of that information.
STRUCTURE_W_STEM_NOVELTY = 0.0

# Chorus vs pre-chorus when both come back with the same words: a part that
# sits between a verse and another word-repeating part every time, where that
# other part comes round more often and runs at least as long, is the
# pre-chorus, and the other part is the chorus (one arena-rock test scan). Only
# section NAMES change -- every boundary is as before, except that two parts
# that were both wrongly named Verse, and so merged, are now listed
# separately. False restores dev47.
# (added in dev48)
STRUCTURE_PRECHORUS_LEADS_IN = True

# Whenever the section map is given, its one short caveat also says the
# section NAMES are inferred and can be wrong (a verse and chorus swapped),
# not only that the times can be a few seconds out. False restores dev48.
# (added in dev49)
STRUCTURE_NAMES_CAVEAT = True

# Smeared lyric timing: when the words do not decide which part is the chorus
# (no part's words stand out) and at least half of the sung time sits inside
# lyric chunks 30 s or longer, the level/register/drums score that decides
# instead leaves out its line-matching lyrics term, which such chunks make
# meaningless (a jangle-pop test track's "one clean-replay test scan": verse and chorus came out swapped).
# Only names change, never a boundary. False restores dev49.
# (added in dev50)
STRUCTURE_SMEARED_LYRICS_HOOK = True
STRUCTURE_SMEARED_CHUNK_S = 30.0
STRUCTURE_SMEARED_SHARE = 0.5

# A part heard only once (at least STRUCTURE_MIN_SECTION_BARS long), before
# the last chorus, that sings mostly the chorus's words is a chorus -- the dev39 rule for verse-like occurrences,
# extended to parts that were grouped on their own (a quieter opening chorus,
# a pop test track's "a pop-ballad test track"). Only when the words already decided the
# chorus. False restores dev49.
# (added in dev50)
STRUCTURE_ONEOFF_CHORUS_WORDS = True

# Verse and chorus grouped as one part although each is clearly tighter on its
# own: the grouping's last merge joined two groups whose own members match
# each other at least STRUCTURE_COHESION_MARGIN better than they match the
# other group, both recurring and sung, and one sings mostly new words each
# time while the other's words mostly come back. Those two groups are kept
# apart, and the new-words one is treated as verse-like ("one verse-first pop test scan":
# verse 0:08 / 1:07 and chorus 0:45 / 1:51 / 2:35 were one part; the synth-rock test track's
# dev49 scan put two choruses with the verses). Only names change, never a
# boundary. False restores dev50.
# (added in dev51)
STRUCTURE_SPLIT_BY_COHESION = True
STRUCTURE_COHESION_MARGIN = 0.12

# A sung segment grouped only with voiceless ones (that arena-rock scan's dev49 scan: the
# 1:14 chorus with a stretch of the guitar solo that Whisper had filled with
# hallucinated words) is moved to the group of sung parts it matches best,
# when that match is at least the usual same-material level. False restores
# dev50.
# (added in dev51)
STRUCTURE_LONE_SUNG_REGROUP = True

# A part heard once, after the last chorus and straight after a chorus, whose
# words mostly come back elsewhere is a Post-Chorus, not a "Bridge" (a bridge
# cannot follow the final chorus). False restores dev50.
# (added in dev51)
STRUCTURE_LATE_POSTCHORUS = True

# dev56: ...but not when the only thing after it is the last few seconds of a
# fade (8 s or less, 15 dB or more below the sung parts): that part is the
# Outro (one verse-first pop test scan 3:03-3:37 became "Post-Chorus" when a 4 s fade tail
# with a breath in it was split off after it). False restores dev55.
STRUCTURE_FADE_TAIL_IS_NOT_NEXT = True

# dev57: how far below the sung parts a short tail must be to count as the
# fade (dev56: 15 dB). one verse-first pop test scan's dev56 scan cut its fade 1.4 s
# earlier, leaving the tail 13.7 dB down -- and 3:04-3:37 "Post-Chorus" again.
# 15.0 restores dev56.
STRUCTURE_FADE_TAIL_DB = 12.0

# dev57: a part straight after the last chorus that sings mostly the chorus's
# words (half or more of its word patterns, at least STRUCTURE_LYRIC_MIN_NGRAMS),
# lasts at least three quarters of a usual chorus (not just the last seconds of
# the fade), sounds like it (similarity 0.6 or more) and is STRUCTURE_FADING_CHORUS_DB or
# more quieter than it is the last chorus fading out, not an Outro
# (one multi-chorus test scan 3:13-3:57, "Chorus (fade)" on the user's map). A sung outro at
# the chorus's level (one verse-first pop test scan) is still the Outro. Its fade stays part
# of it (no final-coda split). False restores dev56.
STRUCTURE_FADING_LAST_CHORUS = True
STRUCTURE_FADING_CHORUS_DB = 4.0

# The dev48 pre-chorus rule also applies when the part it leads into has too
# few transcribed words to measure (none of its occurrences reaches the
# minimum): the word-repeating part that always sits between a verse and that
# part, which comes round more often and runs at least as long, is its
# pre-chorus. False restores dev50.
STRUCTURE_LEADIN_UNWORDED = True

# A late part with the verse's backing that sheds its drums and sits clearly
# higher is a bridge, not "Verse 3": the last of three or more occurrences of
# the verse material, after two or more choruses and before the last one,
# with at most STRUCTURE_LATE_BRIDGE_DRUMS of the other occurrences' drum
# density and a vocal at least STRUCTURE_LATE_BRIDGE_SEMITONES above theirs
# ("one verse-first pop test scan" 2:05: drums 6.1 against 10-12, vocal +2.5 semitones).
# It keeps its own boundary against a following bridge. False restores dev51.
# (added in dev52)
STRUCTURE_LATE_BRIDGE = True

# dev74: when no recurring part was measured as low-repeat (so the words
# could not rank the parts), a part whose words come back almost whole is
# preferred as the chorus over one with no measurable words at all.
# False restores dev73.
STRUCTURE_WORDS_BEAT_UNWORDED = True
STRUCTURE_WORDS_BEAT_UNWORDED_MIN = 0.8

# dev74: a "na na na" middle eight grouped with the verse music (same
# chords) is the bridge, and a one-off sung part that was called Bridge
# before the second chorus is then a verse. False restores dev73.
STRUCTURE_VOCABLE_BRIDGE = True
STRUCTURE_VOCABLE_BRIDGE_SHARE = 0.7
STRUCTURE_VOCABLE_BRIDGE_MIN_WORDS = 6
STRUCTURE_LATE_BRIDGE_DRUMS = 0.6
STRUCTURE_LATE_BRIDGE_SEMITONES = 1.5

# The song's last chorus, when its final part is not the same material as the
# rest of it (below the same-material level against every other part) and
# nothing sung follows, ends where that part begins; the part is the Outro
# ("one verse-first pop test scan": 3:04-3:37 matched its final chorus at 0.24). Only
# detected boundaries are used. False restores dev51.
STRUCTURE_FINAL_CODA = False   # dev37: off -- see the dev37 settings

# Words from a lyric chunk longer than this (seconds) are not placed into
# sections when comparing section words: Whisper's "lines" of 30-100 s (every
# saved test scan has them) put the words of a verse inside the chorus next to
# it, and the word rules then split one material into two or name a chorus
# "Verse". Such sections count as having no measurable words, as a section
# with too few words always did. 0 restores dev54.
# (added in dev55)
STRUCTURE_NGRAM_MAX_CHUNK_S = 30.0

# Consecutive segments of the same material count as one occurrence -- but
# not when one is sung and the other has no voice: a verse, the riff over its
# chords and the next verse (one post-punk test scan) are three
# occurrences, so the verse is recognised as recurring and the riff named
# Instrumental, instead of one unnamed "Section". False restores dev54.
# (added in dev55)
STRUCTURE_RUNS_SPLIT_BY_VOICE = True

# The opening is the Intro even when a voice is heard in it, when that voice
# is only a short pickup before an instrumental stretch (a Eurodance test track's spoken
# "a soul test track'" at 0:00 before the 0:03-0:18 intro groove) or the opening part is
# much quieter and sparser than the sung parts (one child-lead test artist's "one multi-chorus test scan..."
# intro: 9 dB quieter, a third of the drums). False restores dev54.
# (added in dev55)
STRUCTURE_SUNG_INTRO = True
STRUCTURE_SUNG_INTRO_MAX_S = 20.0
STRUCTURE_SUNG_INTRO_QUIETER_DB = 5.0

# dev56: an opening sung only on vocables ("oh, oh, oh...") is the Intro when
# it is at least STRUCTURE_VOCABLE_INTRO_QUIETER_DB quieter than the sung parts
# and no longer than STRUCTURE_SUNG_INTRO_MAX_S (one multi-chorus test scan 0:07-0:16 was
# "Verse 1": 4 dB quieter, half the drums -- just short of the rule above).
# Only openings where Whisper heard words, all of them vocables, and later
# sections have real words. False restores dev55.
STRUCTURE_VOCABLE_INTRO = True
STRUCTURE_VOCABLE_INTRO_QUIETER_DB = 3.0

# Double choruses: when a chorus is 1.75-2.3x as long as the song's shortest
# full chorus (10 s or more), that shortest one sets the ordinary chorus length
# (the median used before mixed single and double choruses: a Eurodance test track's 8, 15
# and 16 bars gave 11.5). The pieces must still sound alike. False restores
# dev54.
STRUCTURE_DOUBLE_CHORUS_SHORTEST = True

# When no part's words stand out as the chorus's, a recurring part whose words
# are measured to be new each time (verse-like) is not chosen as the chorus by
# the level/register score while another recurring sung part is available.
# False restores dev54.
STRUCTURE_VERSE_BY_WORDS = True

# One-off sung material after the last chorus is the Outro (a bridge leads
# back into a chorus; "one verse-first pop test scan" 3:02-3:37 was listed as "Bridge 2").
# A word-repeating tag straight after the last chorus is still a Post-Chorus.
# False restores dev54.
STRUCTURE_NO_BRIDGE_AFTER_LAST_CHORUS = True

# dev70: when only one sung part recurs and each time it comes round it sings
# mostly new words (under STRUCTURE_ONLY_RECURRING_VERSE_MAX_SAME of its word
# patterns come back), it is the Verse, not the Chorus -- even when a refrain
# line inside it repeats. A rock test track (verses with a one-line
# refrain, no chorus) was "Chorus 1 / Chorus 2". False restores dev69.
STRUCTURE_ONLY_RECURRING_NEW_WORDS_IS_VERSE = True
STRUCTURE_ONLY_RECURRING_VERSE_MAX_SAME = 0.5

# In such a song, a one-off sung part after the last verse that is followed
# by at least this much instrumental ending is a Bridge (the Outro starts after
# it); with less after it, it is the Outro as before.
STRUCTURE_VERSE_ONLY_BRIDGE_TAIL_S = 16.0

# When the only sung material that recurs comes round three or more times and
# scores at least average on level/register/drums, it is the chorus (it used
# to need a clearly above-average score, and otherwise the whole song was
# named Verse). False restores dev54.
STRUCTURE_ONLY_RECURRING_IS_CHORUS = True

# One-off sung parts each sitting straight before a chorus that is not the
# last, and sounding alike (similarity at least STRUCTURE_VERSE_BEFORE_CHORUS_SIM
# with the first of them), are verses rather than bridges. False restores dev54.
STRUCTURE_VERSE_BEFORE_CHORUS = True
STRUCTURE_VERSE_BEFORE_CHORUS_SIM = 0.45

# dev19: Verse / Chorus names follow the song title when it is sung, with
# repeating words, only in sections named Verse (_title_hook_naming). False
# restores dev18.
STRUCTURE_TITLE_HOOK = True

# dev20 section map (new scans; replay_saved_structure shows the effect on
# saved ones):
# * STRUCTURE_MERGE_SPLIT_REPEATS: back-to-back sections of the same name and
#   music, each at most STRUCTURE_MERGE_MAX_BARS bars, become one section when
#   every occurrence of that name is split that way (the duet test track dev19:
#   seven 15-second "Chorus" blocks, each chorus cut in half). A lone split
#   (a chorus and a tag, the synth-rock test track's last chorus) is left alone.
# * STRUCTURE_BRIDGE_TO_PRECHORUS: a "Bridge" directly between a Verse and a
#   Chorus, at most STRUCTURE_PRECHORUS_MAX_BARS bars, is a Pre-Chorus (a boy-band
#   test track). A bridge after a chorus or a break is left alone.
# False restores dev19.
STRUCTURE_MERGE_SPLIT_REPEATS = True
STRUCTURE_MERGE_MAX_BARS = 12
STRUCTURE_BRIDGE_TO_PRECHORUS = True
STRUCTURE_PRECHORUS_MAX_BARS = 16

# dev20: a beat grid at a non-octave ratio of the decided tempo (2:3, 3:4 ...)
# is rebuilt at the tempo before the section map is measured
# (_sx_regrid_to_tempo). the teen-pop test track's grid ran at 97.5 BPM against 146, so every
# "bar" was a bar and a half and the sections could not land on its phrases.
# Half / double-time grids are left alone. New scans and replay. False
# restores dev19.
STRUCTURE_REGRID_TO_TEMPO = True

# dev20: when the file's lyrics mark their sections ("[Verse 1]",
# "[Pre-Chorus]", "[Bridge: ...]") and the transcribed words are matched to
# those lines, sung sections take the names the lyrics give them, and a
# section holding two marked parts back to back (verse, then pre-chorus) is
# split where the second begins (_sx_lyric_header_naming). Lyrics without
# markers change nothing. New scans and replay. False restores dev19.
STRUCTURE_LYRIC_HEADERS = True
STRUCTURE_HEADER_MIN_SHARE = 0.6
STRUCTURE_HEADER_SPLIT_MIN_S = 8.0

# Each can be set False on its own to restore dev25 behaviour for that rule.
# * STRUCTURE_HEADER_SPLIT_ON_AGREEMENT: a section whose lyric-header runs
#   begin with the name the measurements already gave it, then change (a verse
#   whose last lines are the pre-chorus), is split where the new name's lines
#   begin, instead of being renamed whole when the later name happens to have
#   more timed lines. The split is also allowed at a measured section line
#   inside it where the header names change. Each piece must be at least
#   STRUCTURE_HEADER_SPLIT_MIN_S long.
# * STRUCTURE_HEADER_NEEDS_VOICE: an Intro / Outro / Instrumental with
#   (almost) no voice on the vocal stem keeps its name -- Whisper's line
#   timing can spill a first or last line over the music around it.
# (added in dev26; section-naming; new scans and replay)
STRUCTURE_HEADER_SPLIT_ON_AGREEMENT = True
STRUCTURE_HEADER_NEEDS_VOICE = True
STRUCTURE_HEADER_VOICE_MIN_SHARE = 0.25
STRUCTURE_HEADER_HINT_TOL_S = 3.0

# * STRUCTURE_PRECHORUS_NEEDS_PATTERN: the dev20 Bridge -> Pre-Chorus rename
#   (STRUCTURE_BRIDGE_TO_PRECHORUS) needs two such places, or one before the
#   second chorus; a single "Bridge" between a verse and a chorus later in the
#   song is the song's bridge (it was renaming real bridges "Pre-Chorus").
# (added in dev26; section-naming; new scans and replay)
STRUCTURE_PRECHORUS_NEEDS_PATTERN = True

# * STRUCTURE_ONEOFF_VERSE_TYPICAL: a one-off sung part between the first and
#   last chorus is a Verse only when it sounds like the verse's typical
#   occurrence (median over its three or more occurrences), not merely like
#   the one closest occurrence -- a bridge next to the last verse can share
#   its sound with that verse alone.
# (added in dev26; section-naming; new scans and replay)
STRUCTURE_ONEOFF_VERSE_TYPICAL = False   # dev38: off -- see the dev38 settings

# * STRUCTURE_REPEAT_NEEDS_SUNG_TWICE: a sung part whose music comes back only
#   in instrumental passes (a bridge over the solo's chords) is named as a
#   one-off sung part (Bridge / Verse by position), not as a recurring Verse.
#   Only between the first and last chorus.
# dev36: off. Tested on the reference songs timed by ear, a recurring part
# heard sung only once now counts as recurring again: label match up on both
# the working and the held-out half (+0.2 / +0.3 points; a dance-pop and a
# Latin-pop test song +9 each, one 80s pop song -4).
# (added in dev26; section-naming; new scans and replay)
STRUCTURE_REPEAT_NEEDS_SUNG_TWICE = False

# * STRUCTURE_SUNG_INTRO_NEEDS_NO_WORDS: a quiet, sparse sung opening is the
#   Intro (STRUCTURE_SUNG_INTRO) only when it carries fewer than a line and a
#   half of real words (STRUCTURE_LYRIC_MIN_NGRAMS): vocables, a pickup, a
#   spoken tag. One singing real lines is the first verse sung softly (or the
#   chorus sung first), and is named with the sung parts.
# (added in dev26; section-naming; new scans and replay)
STRUCTURE_SUNG_INTRO_NEEDS_NO_WORDS = True

# * STRUCTURE_DOUBLE_CHORUS_SHORTEST_NEEDS_DOUBLES: dev55's "the shortest full
#   chorus is the ordinary one" (STRUCTURE_DOUBLE_CHORUS_SHORTEST) applies only
#   when the other choruses are about twice as long as it (1.75x or more).
# (added in dev26; section-naming; new scans and replay)
STRUCTURE_DOUBLE_CHORUS_SHORTEST_NEEDS_DOUBLES = True

# * STRUCTURE_POSTCHORUS_CHORUS_LENGTH: a recurring part heard straight after
#   every chorus, that repeats its words (not a verse), is a Post-Chorus up to
#   1.1x the chorus's median length (and no longer than the chorus before
#   it), not only up to 20 s.
# (added in dev26; section-naming; new scans and replay)
STRUCTURE_POSTCHORUS_CHORUS_LENGTH = True

# * STRUCTURE_MOVEMENT_NUMBERING: a track whose title lists numbered movements
#   ("I. ... / II. ... / III. ..."), at least STRUCTURE_MOVEMENT_MIN_TRACK_S
#   long, gets its sections numbered within each movement ("Part II Verse 1"),
#   each movement placed from where its title is sung
#   (_sx_movement_numbering). Single songs are never changed: without a title
#   listing two or more numbered movements, and every movement found in the
#   sung words in order, the map is returned as it was.
# (added in dev26; section-naming; new scans and replay)
STRUCTURE_MOVEMENT_NUMBERING = True
STRUCTURE_MOVEMENT_MIN_TRACK_S = 300.0
STRUCTURE_MOVEMENT_REMAP = True    # each movement measured and named on its own

# * STRUCTURE_LYRIC_REP_SKIP_VOCABLES: Whisper lines made only of vocables
#   ("oh, oh, oh", "yeah yeah") do not count as repeated lyrics when a part's
#   repeated-words share is measured (an "oh-oh" part outscored the real
#   chorus as the most word-repeating part).
# (added in dev26; section-naming; new scans and replay)
STRUCTURE_LYRIC_REP_SKIP_VOCABLES = True

# * STRUCTURE_VERSE_BY_WORDS_MIN_NGRAMS: the dev55 rule "a recurring part that
#   sings new words every time cannot be the chorus" (STRUCTURE_VERSE_BY_WORDS)
#   needs every compared occurrence to hold at least this many word patterns
#   (two lines or so); a hook Whisper caught only a few words of each time
#   (the title word twice) is not evidence of new words. 0 = dev25.
# (added in dev26; section-naming; new scans and replay)
STRUCTURE_VERSE_BY_WORDS_MIN_NGRAMS = 16

# * STRUCTURE_REASSIGN_BY_WORDS: a sung segment that sounds as much like
#   another recurring part (STRUCTURE_REASSIGN_SIM) as like the material it
#   was grouped with, sings mostly that part's words and few of its own
#   group's (under STRUCTURE_REASSIGN_OWN_MAX), joins that part (a chorus
#   over the verse's loop was merged into one long "Verse").
# (added in dev26; section-naming; new scans and replay)
STRUCTURE_REASSIGN_BY_WORDS = True
STRUCTURE_REASSIGN_SIM = 0.7
STRUCTURE_REASSIGN_OWN_MAX = 0.3

# * STRUCTURE_EJECT_OUTLIER: a single sung segment that average linkage pulled
#   into a recurring part (four or more segments) although it matches that part
#   clearly worse than its other occurrences match each other, and repeats few
#   of their words, is material of its own (a bridge over the chorus's chords)
#   -- only between the part's first and last occurrence (_sx_eject_outliers).
# (added in dev26; section-naming; new scans and replay)
STRUCTURE_EJECT_OUTLIER = True
STRUCTURE_OUTLIER_MARGIN = 0.15
STRUCTURE_OUTLIER_MAX_SIM = 0.75

# * STRUCTURE_CHORUS_WORDS_TAIL_IS_OUTRO: a part on the verse's music that
#   sings the chorus's words is a chorus (dev39) -- except when it comes after
#   every chorus and is the song's last sung part: that is the outro (a vamp
#   on the hook), not one more chorus merged into the last.
# (added in dev26; section-naming; new scans and replay)
STRUCTURE_CHORUS_WORDS_TAIL_IS_OUTRO = True

# * STRUCTURE_DROP_UNMATCHED_TRANSCRIPT: when the file has a lyric sheet of 20+
#   lines but none of the transcribed chunks matched it, the transcript is not
#   used for the section map (its words are mishearings); the map is named
#   from the audio alone, as for a file without lyrics.
# (added in dev26; section-naming; new scans and replay)
STRUCTURE_DROP_UNMATCHED_TRANSCRIPT = True

# * STRUCTURE_SPLIT_BY_KNOWN_PARTS: after the boundary search, a long segment
#   is cut where one side repeats another segment that the other side does not
#   resemble, or where its two sides repeat two different segments
#   (_sx_split_by_known_parts). Finds section lines with no change in the
#   sound itself (a chorus running into the bridge over the same beat); the
#   boundary search's own lines are kept as they are.
# * STRUCTURE_PRECHORUS_BY_WORDS (dev27): a pre-chorus inside the verses --
#   the part at the end of every verse that sings the SAME words each time,
#   leading into the chorus, while the verse before it sings new ones -- is cut
#   out and named Pre-Chorus (_sx_prechorus_cuts). Only for songs whose lyric
#   sheet does not mark its verses itself (lyric headers decide those, as in
#   dev20). Never from the sound alone: a verse whose second half merely
#   changes its backing stays one verse.
# (added in dev26; section-naming; new scans and replay)
STRUCTURE_PRECHORUS_BY_WORDS = True

# * STRUCTURE_KEY_CHANGE_ALIGN (dev28): a song that moves up (or down) a key
#   for its last part has the sections after the change compared with the rest
#   in their own key, so a final chorus "up a key" is still recognised as the
#   chorus (its chords no longer matched the earlier choruses, and everything
#   after the first chorus was named Outro / Verse). Only for a clear, lasting
#   change (_sx_key_change); boundaries are untouched.
# (added in dev26; section-naming; new scans and replay)
STRUCTURE_KEY_CHANGE_ALIGN = True
STRUCTURE_KEY_CHANGE_MIN_GAIN = 0.3
STRUCTURE_KEY_CHANGE_MIN_CORR = 0.85
STRUCTURE_KEY_CHANGE_MAX_UNSHIFTED = 0.6
STRUCTURE_KEY_CHANGE_MIN_TAIL_S = 30.0

# * STRUCTURE_EJECT_UNSHARED_WORDS (dev28): a middle occurrence of a
#   word-repeating part (a chorus) that sings two lines or more of which
#   almost none come back from the part's other occurrences is a part of its
#   own (a bridge over the chorus's chords that measured close enough to join
#   it) -- _sx_eject_unshared_words.
# (added in dev26; section-naming; new scans and replay)
STRUCTURE_EJECT_UNSHARED_WORDS = True

# * STRUCTURE_VERSE_PARTS (dev28): a verse made of distinct melodic parts,
#   one of which returns later inside the same verse (A-B-C-B), is listed part
#   by part ("Verse 1 Part 1" ... "Verse 1 Part 4") at the measured lines
#   between them -- they were merged into one long verse. A verse of two
#   halves, or of parts that never return inside it, stays one section.
# (added in dev26; section-naming; new scans and replay)
STRUCTURE_VERSE_PARTS = True

# * STRUCTURE_VERSE_HALVES_AS_PARTS (dev29): two or more back-to-back sections
#   named Verse are numbered as parts of one verse ("Verse 1 Part 1",
#   "Verse 1 Part 2") so the next verse is "Verse 2", not "Verse 3". Songs whose
#   lyric sheet numbers its verses keep the sheet's numbering.
# (added in dev26; section-naming; new scans and replay)
STRUCTURE_VERSE_HALVES_AS_PARTS = True

# * STRUCTURE_TRAILING_SILENCE_JOINS_LAST (dev29, OFF by default): the silent
#   last few seconds of a file (at most STRUCTURE_TRAILING_SILENCE_MAX_S; a
#   fade's end, the file's padding) are counted in the last section instead of
#   being listed as "Silence" -- the way section times are usually written, to
#   the end of the file. Longer silences (a hidden track's gap) stay listed.
# (added in dev26; section-naming; new scans and replay)
STRUCTURE_TRAILING_SILENCE_JOINS_LAST = False
STRUCTURE_TRAILING_SILENCE_MAX_S = 10.0

# * STRUCTURE_PRECHORUS_KEEPS_NAMES (dev30): a pre-chorus cut out by its words
#   (STRUCTURE_PRECHORUS_BY_WORDS) is kept only when the rest of the song keeps
#   the names it had without the cut. The cut re-groups the song, and the new
#   short part, which sings the same words every time, could be taken for the
#   chorus -- the real chorus then came out as "Verse" (a pop test song: Verse
#   0:14, "Chorus" 0:41 = the pre-chorus, "Verse 2" 0:57 = the chorus).
# (added in dev30; section-naming; False restores dev29)
STRUCTURE_PRECHORUS_KEEPS_NAMES = True

# * STRUCTURE_CHORUS_HALVES (dev30): a chorus of two halves that sound unlike
#   each other (a first half and a second half of different lines) is one
#   chorus when both halves bring their words back every time, always come
#   one straight after the other, and are sung at the same level. With no
#   verse group to compare (the verses were each their own material), the
#   first half had been named Verse and the real verses Bridge.
# (added in dev30; section-naming; False restores dev29)
STRUCTURE_CHORUS_HALVES = True
STRUCTURE_CHORUS_HALF_SAME_WORDS = 0.6
STRUCTURE_CHORUS_HALVES_WORDED = True
STRUCTURE_CHORUS_HALF_PRECHORUS_ST = 4.0

# * STRUCTURE_ONEOFF_VERSE_SLOT (dev30): in a song whose verses were each
#   measured as their own material (no recurring verse), the one sung part
#   that sits straight between the first and second chorus, singing words,
#   after a verse before the first chorus, is the second verse -- it had been
#   named Bridge because it matched no recurring verse. Needs three choruses
#   or more; a part followed by a solo or anything else before the second
#   chorus keeps its name.
# (added in dev30; section-naming; False restores dev29)
STRUCTURE_ONEOFF_VERSE_SLOT = True

# * STRUCTURE_SUNG_INTRO_TAG_OR_SPEECH (dev30): a quiet opening with words is
#   still the Intro (not "Verse 1", as dev26 made it) when it only repeats one
#   short line (fewer than STRUCTURE_SUNG_INTRO_MIN_WORDS different words:
#   one four-word line sung four times), when it is a spoken prelude -- a speech sample at
#   least STRUCTURE_SPOKEN_PRELUDE_DB quieter than the singing, followed by
#   the band before any singing -- or when the file's lyric sheet itself opens
#   with an [Intro] stanza. A soft first verse or an opening chorus is named
#   as dev26 does.
# (added in dev30; section-naming; False restores dev29)
STRUCTURE_SUNG_INTRO_TAG_OR_SPEECH = True
STRUCTURE_SUNG_INTRO_MIN_WORDS = 8
STRUCTURE_SPOKEN_PRELUDE_DB = 15.0

# * STRUCTURE_REPEATED_BRIDGE (dev30): a part heard straight after a chorus
#   every time it comes, that sings the same words each time, sounds unlike
#   the verse and is too long for a post-chorus, is a Bridge (a middle eight
#   heard twice), not another Verse -- named Verse, it was merged with the
#   verse after it into one long "verse".
# (added in dev30; section-naming; False restores dev29)
STRUCTURE_REPEATED_BRIDGE = True

# * STRUCTURE_LONE_SUNG_IS_VERSE (dev30): in a track with singing where no
#   part comes round again, a sung part is named Verse when there are at most
#   STRUCTURE_LONE_SUNG_MAX_PARTS of them that sing words (a mostly
#   instrumental track with one sung passage) instead of the neutral
#   "Section"; a wordless, barely voiced stretch of such a part (stem bleed
#   after the singing) is Instrumental. Songs with more sung parts, none
#   repeating (through-composed), keep "Section".
# (added in dev30; section-naming; False restores dev29)
STRUCTURE_LONE_SUNG_IS_VERSE = True
STRUCTURE_LONE_SUNG_MAX_PARTS = 2

# * STRUCTURE_KNOWN_PART_WORDS_VETO (dev30): the dev26 known-part split
#   (STRUCTURE_SPLIT_BY_KNOWN_PARTS) is not made when the piece it would cut
#   off sings words that, next to the matched part elsewhere, sit inside a
#   segment twice as long or more (they are under a third of its words): there the
#   section line was placed wrong, and copying it here moved a correct one
#   (on a folk-rock test song the chorus's opening line was cut off the
#   first chorus because in the second it had been left at the end of the
#   verse).
# (added in dev30; section-naming; False restores dev29)
STRUCTURE_KNOWN_PART_WORDS_VETO = False   # dev38: off -- see the dev38 settings

# * STRUCTURE_LEARNED_NAMER (dev31): section names checked against reference
#   songs (sections someone timed and named). Each section is described by
#   numbers only -- where it sits, how loud, sung, high and drum-heavy it is
#   against the song's other sung parts, how its words and music come back,
#   the type the rules gave it and its neighbours -- and the
#   STRUCTURE_LEARNED_NAMER_K most alike reference sections vote. A name
#   changes only when STRUCTURE_LEARNED_NAMER_AGREE of the vote agrees on one
#   other type, from at least STRUCTURE_LEARNED_NAMER_MIN_SONGS songs. The
#   table is read from STRUCTURE_LEARNED_NAMER_FILE (next to this script);
#   without it nothing changes. musiclyse_check.py --train-namer makes it from
#   your reference timings, so every reference you add helps any song with a
#   section like its sections. The table holds no titles, names or words.
# (added in dev31)
STRUCTURE_LEARNED_NAMER = False   # dev38: off -- see the dev38 settings
STRUCTURE_LEARNED_NAMER_FILE = "musiclyse_namer.json"
STRUCTURE_LEARNED_NAMER_K = 15
STRUCTURE_LEARNED_NAMER_AGREE = 0.85
STRUCTURE_LEARNED_NAMER_MIN_SONGS = 3

# * STRUCTURE_TITLE_HOOK_FUZZY (dev31): for titles of three words or more, the
#   title-hook check (STRUCTURE_TITLE_HOOK) also counts the title sung with one
#   word misheard (Whisper's usual slip), and needs it once in each of the
#   verse-named sections that sing it, not twice: a dance-pop test song whose
#   verses repeat their words too had verse and chorus swapped, its title line
#   heard right only once of three times.
# (added in dev31)
STRUCTURE_TITLE_HOOK_FUZZY = True

# * STRUCTURE_TITLE_PICKS_CHORUS (dev32): when two recurring parts bring their
#   words back about equally (a chorus and a pre-chorus) and the loudness /
#   register hook score would choose between them, the one that sings the
#   song's title (two words or more) in every occurrence is the chorus, if the
#   other never sings it -- unless the other sits next to it every time (two
#   halves of one chorus, or a lead-in). A power-pop test song: its pre-chorus
#   scored higher and was named Chorus, the title-line chorus merged into the
#   verses.
# (added in dev32)
STRUCTURE_TITLE_PICKS_CHORUS = True

# * STRUCTURE_TITLE_PICKS_ANY_WORDED (dev33): the title check above weighs
#   every part that clearly repeats its words, not only those within 0.15 of
#   the most repetitive one: a rescan of the power-pop test song measured its
#   pre-chorus at 1.00 and its title chorus at 0.83, just outside that margin,
#   and the pre-chorus was named Chorus again.
# (added in dev32)
STRUCTURE_TITLE_PICKS_ANY_WORDED = True

# * STRUCTURE_WORDLESS_CODA (dev33): after the last chorus, a part named Verse
#   whose passages carry at most two real words (the rest vocables -- "ooh",
#   "yeah" -- or nothing Whisper could make out) is Instrumental: a long
#   jangle-pop coda of wordless "ooh"s over a guitar solo was named Verse 3
#   and Verse 4. Only
#   music heard nowhere before the last chorus: a part that is also heard
#   earlier is left alone (a last verse Whisper failed to transcribe, or the
#   intro's music coming back as the outro). Only in songs with at least 30
#   transcribed words.
# (added in dev32)
STRUCTURE_WORDLESS_CODA = True
STRUCTURE_WORDLESS_CODA_MAX_WORDS = 2
STRUCTURE_WORDLESS_CODA_SONG_WORDS = 30

# * STRUCTURE_SPLIT_STOCK_PHRASES (dev33): Whisper's stock video phrases ("We'll
#   be right back", "guitar solo", "Thanks for watching") split over two or
#   three lyric lines are dropped from the lines the section map reads; the
#   transcription's own filter checks one line at a time, so they counted as
#   sung words in instrumental passages. Whole lines only; the lyrics text in
#   the report is not changed.
# (added in dev32)
STRUCTURE_SPLIT_STOCK_PHRASES = True

# * STRUCTURE_ONEOFF_VERSE_SLOT_ANY (dev33): STRUCTURE_ONEOFF_VERSE_SLOT also in
#   songs that have a recurring verse: the one sung part, with words, between
#   the first and second choruses of a song with three or more choruses and a
#   verse before the first is a verse, not a bridge. A jazz-pop test song's
#   second verse (its words new, so its music did not match the first and
#   last verses that share their words) was named Bridge.
# (added in dev32)
STRUCTURE_ONEOFF_VERSE_SLOT_ANY = True

# * STRUCTURE_FOREIGN_CHUNKS (dev33): lyric lines that are Whisper's foreign-
#   language hallucinations -- fluent-looking Welsh over music it cannot make
#   out, Khmer strings, Japanese credit lines in an English song -- are dropped
#   from the lines the section map reads (see _sx_drop_foreign_chunks). A
#   jangle-rock test song's verses and choruses were "transcribed" as one
#   Welsh sentence repeated twenty times, which looked like a chorus. The
#   lyrics text in the report is not changed.
# (added in dev32)
STRUCTURE_FOREIGN_CHUNKS = True

# Both OFF: tested on the 148 saved scans, each makes the lyric input more
# complete but moves the section map both ways (two songs closer to their
# reference timings, two further), so neither is used yet. Kept so they can
# be re-tested as more hand-timed references come in.
# * STRUCTURE_TAG_FRAGMENTS (dev34): for a file with its own lyrics, transcribed
#   lines the scan-time matching left unmatched because Whisper cut them out
#   of the middle of a lyric line are matched to the run of lyric words they
#   resemble (see _sx_align_tag_fragments), so their words are corrected and
#   their stanza is known to the section map. Lines that match nothing are
#   kept as they are. With MODE "ordered" a fragment's stanza is used only
#   when it fits in sequence with the matched lines around it (on the saved
#   scans: one song closer, one a second further); "text" only corrects its
#   words; "all" uses every match (seven songs further).
# (added in dev34)
STRUCTURE_TAG_FRAGMENTS = False
STRUCTURE_TAG_FRAGMENT_MIN = 0.8
STRUCTURE_TAG_FRAGMENT_MODE = "ordered"

# * STRUCTURE_TAG_FILL_MISSED (dev34): lines of the file's own lyrics Whisper
#   missed between two it heard are added for the section map where they
#   must have been sung (see _sx_fill_missed_tag_lines). On the saved scans:
#   562 such lines in 92 songs; two maps closer, two further.
# (added in dev34)
STRUCTURE_TAG_FILL_MISSED = False

# Weights of the evidence that picks the chorus among recurring sung parts
# when the words do not decide it (the "hook score"), and two limits of the
# pre-chorus test; dev34's values were fixed by hand.
# (added in dev35; tuned on all reference songs together)
STRUCTURE_HOOK_W_LOUD = 0.8
STRUCTURE_HOOK_W_REGISTER = 1.0
STRUCTURE_HOOK_W_DRUMS = 0.4
STRUCTURE_HOOK_W_REPEATS = 0.25
STRUCTURE_HOOK_W_LYRICS = 3.0
STRUCTURE_PRECHORUS_UNIT_MAX_S = 20.0
STRUCTURE_PRECHORUS_DISTINCT_MAX = 0.45

# * STRUCTURE_PRECHORUS_BUILDUP (dev35): the last segment of a verse that leads
#   straight into the chorus is a Pre-Chorus when it builds up to the chorus
#   -- the voice sings higher (STRUCTURE_PRECHORUS_BUILDUP_ST semitones or more
#   above the rest of the verse, by median pitch) and the drums play more --
#   and it lasts 8 s to STRUCTURE_PRECHORUS_MAX_S. Such a pre-chorus often
#   sounds enough like its verse to be grouped with it, so it had been named
#   Verse.
# (added in dev35; tuned on all reference songs together)
STRUCTURE_PRECHORUS_BUILDUP = False
STRUCTURE_PRECHORUS_BUILDUP_ST = 0.6
STRUCTURE_PRECHORUS_BUILDUP_MIN_UNITS = 1

# * STRUCTURE_PRECHORUS_BUILDUP_CUTS (dev35): where the words show no
#   pre-chorus inside the verses (STRUCTURE_PRECHORUS_BY_WORDS), the build-up
#   above is looked for inside them too, and a section line added where it
#   starts (see _sx_prechorus_cuts_buildup).
# (added in dev35; tuned on all reference songs together)
STRUCTURE_PRECHORUS_BUILDUP_CUTS = False

# * STRUCTURE_REGRID_OCTAVE_SLOW (dev36): STRUCTURE_REGRID_TO_TEMPO also when
#   the beat grid runs at half or a quarter of the decided tempo (within 3%):
#   each grid beat is split in two (or four) before the map is measured. With
#   a grid that slow every "bar" is two or four bars long, so the map could
#   not draw a line closer than every eight or sixteen bars: on the reference
#   songs with such a grid, sections came out double length and misnamed.
# (added in dev36)
STRUCTURE_REGRID_OCTAVE_SLOW = True

# * STRUCTURE_REGRID_OCTAVE_FAST (dev36): and when it runs at double the
#   tempo, every other grid beat is kept (the set that lands more kick and
#   snare hits).
# (added in dev36)
STRUCTURE_REGRID_OCTAVE_FAST = False

# * STRUCTURE_TITLE_SEGMENTS_NEEDS_NEW_WORDS (dev36): STRUCTURE_TITLE_SEGMENTS
#   (dev32) is meant for songs whose parts the words cannot tell apart
#   because none repeats them (a heartland-rock test song: 6% and 3%); when
#   every part repeats its words equally, nothing was undecided and it is not
#   used. A post-grunge test song's chorus, repeating 94% of its words, had
#   been renamed Verse and its title-line pre-chorus Chorus.
# (added in dev36)
STRUCTURE_TITLE_SEGMENTS_NEEDS_NEW_WORDS = True

# * STRUCTURE_PRECHORUS_AFTER_VERSE (dev32): a recurring part that repeats its
#   words, comes straight after the verse's music every time (never after a
#   chorus), is different music from the verse, runs at most 24 s and leads
#   into the chorus the first time it is heard is a Pre-Chorus even when later
#   passes lead into a bridge or a solo instead (the pre-chorus rule otherwise
#   needs every pass to lead into the chorus); it was named Verse and merged
#   into the verse before it.
# (added in dev36)
STRUCTURE_PRECHORUS_AFTER_VERSE = True
STRUCTURE_PRECHORUS_AFTER_VERSE_MAX_S = 24.0

# * STRUCTURE_TITLE_SEGMENTS (dev32): in a song whose recurring parts cannot be
#   told apart by their words (none repeats them clearly more than another),
#   when the title is sung two or more times inside single passages and those
#   passages fall in different parts of the music, one of them named Verse,
#   the music is not separating verse from chorus (one riff throughout): the
#   title passages are named Chorus, and Chorus-named passages that sing words
#   but never the title are named Verse. A heartland-rock test song had its
#   title-line choruses named Verse and its verses Chorus.
# (added in dev36)
STRUCTURE_TITLE_SEGMENTS = True
STRUCTURE_UNSHARED_WORDS_MAX = 0.1
STRUCTURE_PRECHORUS_TAIL_SHARE = 0.7
STRUCTURE_PRECHORUS_HEAD_SHARE = 0.2
STRUCTURE_PRECHORUS_MAX_S = 26.0

# * STRUCTURE_PRECHORUS_PROPAGATE: the pre-chorus found by its words
#   (STRUCTURE_PRECHORUS_BY_WORDS) needed two verses that each sing only new
#   words before it. A verse whose opening lines come back elsewhere (a
#   post-chorus or later verse that repeats them) failed that test even when
#   its end sings the same pre-chorus words. Once one verse shows the
#   pre-chorus, the other verses are searched for those same words, packed at
#   their end, with the same shares and lengths. False restores dev36.
# (added in dev37)
STRUCTURE_PRECHORUS_PROPAGATE = True

# * STRUCTURE_KNOWN_PART_NOT_HALVES: the cut by repetition
#   (STRUCTURE_SPLIT_BY_KNOWN_PARTS) is never made in the exact middle of a
#   segment. A part made of two equal halves that each sound like other
#   material is one part singing its phrase twice (a chorus whose two halves
#   come back separately), not two parts: on the reference songs such cuts
#   fell on a real section line 1 time in 13, other cuts 19 times in 31.
#   False restores dev36.
# (added in dev37)
STRUCTURE_KNOWN_PART_NOT_HALVES = True

# * STRUCTURE_FINAL_CODA is now off. It cut the last chorus where its final
#   part sounded different and called that part the Outro; on the 91 reference
#   songs it fired three times and each time the chorus carried on past the
#   cut (three rock and pop test tracks). True restores dev36.
# Every section rule added since dev25 was switched in turn and checked on all
# reference songs (91 timed by ear or by Harmonix, 70 with lyric-derived
# references) and on the 39 saved scans with no reference. One rule for all:
# a rule added since dev25 stays on only when it helps the reference songs or
# at least never makes one worse. Three did not qualify and are off:
# * STRUCTURE_ONEOFF_VERSE_TYPICAL: touched one reference song and made it
#   worse. True restores dev37.
# * STRUCTURE_LEARNED_NAMER: needs a table file that was never shipped, so it
#   never ran. True restores dev37 (no change without the file).
# * STRUCTURE_KNOWN_PART_WORDS_VETO: changed nothing on any of the 216 saved
#   scans. True restores dev37.
# (added in dev38)
STRUCTURE_SPLIT_BY_KNOWN_PARTS = True
STRUCTURE_KNOWN_PART_SIM = 0.85
STRUCTURE_KNOWN_PART_OTHER_MAX = 0.45
STRUCTURE_KNOWN_PART_MAX_RATIO = 1.6

# A section named "Chorus" whose sung words are a verse's words, and not the words the
# other choruses share, is a verse that reuses the chorus's music: it is named as a
# verse (part of the verse next to it). Needs at least two choruses that share their
# words and the timed lyrics; section times never change.
# (added in 0.7-pre68; False gives pre67's section names)
STRUCTURE_CHORUS_SHARED_WORDS = True

# A chorus at least STRUCTURE_CHORUS_REPEAT_LONG times as long as the song's usual chorus
# is split where the music turns into the song's chorus again (key changes allowed):
# one chorus length on (measured inside it), give or take STRUCTURE_CHORUS_REPEAT_WINDOW, and
# only where it matches the song's other choruses as closely as they match each other
# (times STRUCTURE_CHORUS_REPEAT_MATCH). Choruses in Chorus A/B groups are measured
# within their own group.
# (added in 0.7-pre64; False gives pre63's behaviour)
STRUCTURE_SPLIT_CHORUS_REPEATS = True
STRUCTURE_CHORUS_REPEAT_LONG = 1.8
STRUCTURE_CHORUS_REPEAT_REF_MAX = 1.35     # choruses up to this times the usual length are the yardstick
STRUCTURE_CHORUS_REPEAT_MATCH = 0.85
STRUCTURE_CHORUS_REPEAT_WINDOW = 0.15
STRUCTURE_CHORUS_REPEAT_MIN_LAST = 0.45    # the last piece is at least this much of a usual chorus
STRUCTURE_CHORUS_REPEAT_BEATS = 64         # most beats of a chorus compared

# How long one chorus lasts is measured inside the long one: the shortest repeat, from
# half to one and a half times the song's usual chorus and at least this many beats (six
# bars), whose music matches itself within STRUCTURE_CHORUS_REPEAT_NEAR_BEST of the best.
# (added in 0.7-pre64; False gives pre63's behaviour)
STRUCTURE_CHORUS_REPEAT_MIN_BEATS = 24
STRUCTURE_CHORUS_REPEAT_NEAR_BEST = 0.97

# ...and when that long "chorus" doesn't match the song's choruses where it starts, the
# piece before its first real repeat carries on the section before it ("Bridge Part 1",
# "Bridge Part 2") instead of being called a chorus.
# (added in 0.7-pre64; False gives pre63's behaviour)
STRUCTURE_CHORUS_REPEAT_HEAD_PARTS = True

# * STRUCTURE_CHORUS_GROUPS: a song whose choruses are two (or three) different
#   tunes, each of them coming back -- every chorus in a group of the same music,
#   every group heard at least twice -- names them by group: Chorus A1, Chorus B1,
#   Chorus A2 ... (A for the tune heard first). The times and every other name stay.
#   A chorus outside any group, or heard only once, leaves the names as they were.
#   False gives pre60's names.
# (added in the 0.7-pre60 patch)
STRUCTURE_CHORUS_GROUPS = True

SAME_NAME_DISTINCT_CLUSTER_MIN = 3

# dev3: fixes in the drum report must not silently retune a trusted section
# map. Keep the old fallback hit selection ONLY for structure/tempo evidence.
STRUCTURE_KEEP_LEGACY_FALLBACK_HITS = True
STRUCTURE_ALIGNED_ROLE_LABELS = True

# Requires validation on the user's reference tracks before enabling broadly.
STRUCTURE_RECURRENT_ROLE_LABELS = False

# Fixed-boundary repair after an instrumental, requiring two lyric-matched anchors.
STRUCTURE_RETURNING_PRECHORUS_LABELS = True
STRUCTURE_LYRIC_MIN_NGRAMS = 8        # a segment needs about a line and a half of words
STRUCTURE_LYRIC_SAME_LO = 0.3         # verse: same music, mostly new words each time (dev38: 0.2)
STRUCTURE_TAG_LINE_MIN_MATCH = 0.75   # chunk-to-tag-lines match needed to time lines / stanzas
STRUCTURE_LYRIC_SAME_HI_MIN = 0.3     # chorus: its words come back each time...
STRUCTURE_LYRIC_CHORUS_SHARE = 0.5    # a verse-like part singing this much of the chorus's words is a chorus
STRUCTURE_LYRIC_SAME_GAP = 0.25       # ...clearly more than the verse's do

# dev45: verse and chorus over the same music. one verse-first pop test scan's verse,
# pre-chorus and chorus share their backing, so all of them landed in one
# cluster; consecutive segments of one cluster count as a single occurrence,
# and each verse + chorus pair was named "Chorus 1 0:08-1:00", "Chorus 2" --
# a verse-first song read as chorus-first. Their WORDS still differ: the verse
# brings new words each time and the chorus repeats its own. When no other
# recurring part already reads as a verse by its words, and a cluster's
# measurable segments fall into new-words and same-words groups (two or more
# of each, split by a clear gap in how much of their words recur) with a new-words segment running
# straight into a same-words one, the new-words segments become their own
# material. Tracks with a verse cluster of their own are left alone.
STRUCTURE_WORDS_NEW_MAX = 0.55    # share of a segment's word patterns heard elsewhere: verse-like
STRUCTURE_WORDS_SAME_MIN = 0.55   # ... chorus-like (above the widest gap)
STRUCTURE_WORDS_GAP = 0.2         # the two groups must be at least this far apart

# dev46: a short tag chosen over the chorus. "The part whose words come back
# most" is the chorus -- but a short hook line sung after each chorus ("I need
# somebody") repeats its words more exactly than the chorus does, so it was
# named the Chorus and the real chorus became a "Verse" merged into the
# verses. Word recurrence here is measured per segment against the rest of
# the song (not pair by pair inside a part, which an occurrence with no
# transcribed words can drag down).
STRUCTURE_TAG_WORDS_MIN = 0.55   # a part whose word patterns mostly come back elsewhere
STRUCTURE_TAG_MAX_S = 16.0       # a tag's shortest occurrence is at most this long
STRUCTURE_TAG_LENGTH_RATIO = 1.5 # the chorus is at least this much longer in total
STRUCTURE_HOOK_CHORUS_WORDS_MAX = 0.45  # a level/register "chorus" whose words mostly don't come back

# An edge stretch counts as instrumental when no bar in it reaches the "sung"
# level (STRUCTURE_SUNG_MIN_COVERAGE) and its average voice coverage is below
# this -- a little above zero, because separated vocal stems pick up bleed.
STRUCTURE_UNSUNG_RUN_MEAN = 0.08

# (added in 0.7.1-pre12; False gives pre11's map)
STRUCTURE_BOOKEND_MAX = True        # only unsung parts within STRUCTURE_BOOKEND_SPAN_S of the start/end are Intro/Outro
STRUCTURE_BOOKEND_SPAN_S = 120.0     # (60 s split the long intros of several rock songs on a library check)
STRUCTURE_FIRST_SECTION_FROM_ZERO = True  # a dropped opening scrap (a second of silence) belongs to the first section
STRUCTURE_GAP_KEEPS_STARTS = True   # listening-model map: a gap joins the part before it, so the next start isn't moved earlier
STRUCTURE_SANITY_SHORT_OK = True    # listening-model map: one very short section no longer discards the whole map
STRUCTURE_NO_CHORUS_WITHOUT_REPEATED_WORDS = False  # when every recurring sung part sings new words each time, none is "Chorus"
                                    # (off: Whisper can hear a repeated chorus line differently each time, and then
                                    # a real chorus loses its label)
STRUCTURE_PRECHORUS_MOST_OCCURRENCES = False  # a pre-chorus keeps its name when one of 3+ occurrences leads elsewhere
                                    # (off: on a library check it named an opening verse line "Pre-Chorus")
STRUCTURE_OUTRO_MAX_SUNG_S = 0      # a one-off sung part after the last chorus longer than this is a Bridge (0: always Outro)

# (added in 0.7.1-pre12; False gives pre11's map)
STRUCTURE_FIELD_ANY_LAYOUT = True   # listening-model map: "Name (0:00-0:10), ...", "Name: 0:00-0:10" and "0:00 to 0:10" are read too

# (added in 0.7.1-pre13)
STRUCTURE_IGNORE_SILENT_DRUM_HITS = True  # drum "hits" measured at almost no level (the borderline detections that come
                                    # and go between scans of one file) don't count as drums in the section map

# 0.7.1-pre18: in a song's tag lyrics whose section headers sit directly above their lines, a
# header standing alone ("[Chorus]" with a blank line after it, meaning that part comes again
# here) marks a repeat and doesn't name the lyrics after it. False gives pre17's section maps.
STRUCTURE_BARE_HEADER_IS_REPEAT = True

# 0.7.1-pre20: a chorus of 12 seconds or less whose sung lines are only the last lines of the
# tag-lyric stanza the section before (a verse, pre-chorus, chorus, post-chorus or bridge) was
# singing, carrying straight on from it up to the stanza's end, joins that section, and the
# choruses are numbered again (a bridge whose last two lines were split off as a short
# "Chorus 3" is one bridge again). A line starting in the 1.5 s before a section counts as
# that section's pickup. Needs the song's tag lyrics. False gives pre19's section maps.
STRUCTURE_JOIN_STANZA_TAIL = True

# 0.7.1-pre21: for a song without tag lyrics, the section map doesn't use the Whisper transcript
# (its repeated lines, which name choruses) when the transcript is unreliable: at least four of its
# lines, and at least 15% of the lines whose language can be told, are in a second language (unless
# the lyrics-language check found the song really is sung in two), or at least three of its lines
# are Whisper's stock phrases ("be right back", "thanks for watching" in any language). The map
# then goes by the music. False gives pre20's section maps.
STRUCTURE_DROP_UNRELIABLE_TRANSCRIPT = True

# 0.7.1-pre23: when the last lines of at least two verses come back right before a chorus (lines
# the rest of those verses doesn't repeat), those lines are a Pre-Chorus of their own (5 to 25
# seconds, after at least 8 seconds of verse). When those lines hold the song's title, they are
# the start of the chorus instead, and the chorus starts there. Needs the song's lyrics. False
# gives pre22's section maps.
STRUCTURE_PRECHORUS_VERSE_ENDINGS = True

# 0.7.1-pre23: a sung section after the intro or an instrumental doesn't start on one to three
# words sung at least 1.25 s per word with a gap of 1.5 s or more after them (a pickup, or words
# heard in the playing): it starts at the next line, and the part before it is longer. Needs the
# song's lyrics. False gives pre22's section maps.
STRUCTURE_LATE_FIRST_WORD = True


# ==============================================================================
#  13. SONGFORMER
# ==============================================================================

# SongFormer, Stage 1: /songformer runs the SongFormer helper beside this script
# (musiclyse-songformer-<build>.py) in SongFormer's own Python. It only writes
# saved-songs/songformer/ and comparison reports. (pre59: analyses also run it --
# see SONGFORMER_CORRECT_STRUCTURE below.) /songformer setup shows the one-time
# setup steps.
# (added in 0.7-pre58; False gives pre57's behaviour)
SONGFORMER_COMMAND = True
SONGFORMER_PYTHON = ""             # SongFormer's own Python; "" = ~/SongFormer/env/bin/python (install_songformer.sh)
SONGFORMER_REPO = ""               # SongFormer's code folder; "" = SongFormer beside this script, else ~/SongFormer/code
SONGFORMER_DEVICE = "auto"         # auto (the Mac GPU when it works, else the CPU), mps or cpu
SONGFORMER_AFTER_ANALYSIS = False  # True: /analyse, /reanalyse, /batch and /musicvideoanalyse then scan new saves

# /songformer itself (True is pre64's behaviour); False hides the command and its line in the
# command list, while analyses still use SongFormer as SONGFORMER_COMMAND and
# SONGFORMER_CORRECT_STRUCTURE say.
# (added in 0.7-pre65; no change to the analysis or the exports)
SONGFORMER_SLASH_COMMAND = False

# * SONGFORMER_SURE_ADDS: a section change SongFormer is sure of (boundary strength
#   0.9 or more) that Musiclyse's map lacks (none within 3 s) is added like any other
#   (rule 2), also on dance tracks, between two parts of the same kind, and without
#   Musiclyse's own evidence -- the other safeguards stay (no part under 8 s, sung or
#   instrumental is Musiclyse's call, no boundary lost; if one would be, the map is
#   made exactly as pre60 makes it). In 16 songs with listened timestamps, every such
#   boundary missing from the map was one the listener marked. False gives pre60.
#   SONGFORMER_SURE_EARLY_STARTS (off): where the part after such a boundary would be
#   too short and Musiclyse's next part, at most 4.5 bars later, is the kind SongFormer
#   hears from it, that part starts there instead. Right on two timed songs, wrong on
#   one -- SongFormer took a pre-chorus for the chorus -- so it stays off.
# (added in 0.7-pre61)
SONGFORMER_SURE_ADDS = True
SONGFORMER_SURE_EARLY_STARTS = False   # tested: right on 2 timed songs, wrong on 1 (a pre-chorus)

# SongFormer, Stage 2: while a song is analysed, SongFormer's sections check and
# correct Musiclyse's own section map. Musiclyse's map is always built first, as
# before; SongFormer may then only (1) move a boundary both of them found onto
# SongFormer's more precise time, (2) add a missed section change that Musiclyse's
# own evidence also supports, (3) vote on section names below song tag headers
# and the title line, and (4) name an instrumental's opening and closing parts
# when both agree. No SongFormer set up, or a failed scan: the map is as pre58's.
# Lyrics, lyric timing, vocalist identity, key and tempo are never changed.
# (added in 0.7-pre59; False gives pre58's section maps)
SONGFORMER_CORRECT_STRUCTURE = True   # master switch (False: every rule below is off)
SONGFORMER_SNAP_TIMES = True          # (1) a boundary both found moves to SongFormer's time (at most a bar)
SONGFORMER_ADD_SECTIONS = True        # (2) a missed section change is added (not on instrumentals or dance)
SONGFORMER_NAME_VOTE = True           # (3) SongFormer votes on section names
SONGFORMER_INSTRUMENTAL_NAMES = True  # (4) an instrumental's first/last part: Intro/Outro when both agree
SONGFORMER_STRONG_MOVES = True        # (1) where SongFormer is very sure of a boundary (strength 0.9+), a
                                      #     boundary of Musiclyse's may move up to 3 bars to it (else 1)
SONGFORMER_LATE_STARTS = True         # (2) a part Musiclyse starts a few seconds early, where SongFormer
                                      #     (strength 0.7+) hears the gap as the previous part, starts later
SONGFORMER_SCAN_TIMEOUT_S = 600       # a scan taking longer is abandoned (the map stays as pre58's)

# 0.7.1-pre18: SongFormer has no post-chorus name, so its name vote never renames a Post-Chorus
# (as it already never turns a Pre-Chorus into a verse). False gives pre17's section maps.
SONGFORMER_KEEP_POST_CHORUS = True

# 0.7.1-pre23: SongFormer's name vote doesn't turn a Pre-Chorus into a Chorus when most of the
# Pre-Chorus's words are its own: no more of them than STRUCTURE_PRECHORUS_DISTINCT_MAX (the share
# the pre-chorus test already uses) are also sung in the choruses. A Pre-Chorus that sings the
# chorus's words is still renamed. Only used when the song has a SongFormer result. False gives
# pre22's section maps.
SONGFORMER_KEEPS_DISTINCT_PRECHORUS = True

# 0.7.1-pre23: the same for a Bridge: SongFormer's name vote doesn't turn a Bridge with words of its
# own into a Chorus. False gives pre22's section maps.
SONGFORMER_KEEPS_DISTINCT_BRIDGE = True

# 0.7.1-pre24: of two choruses back to back, the first is a Pre-Chorus when SongFormer, over that
# part itself, hears a pre-chorus (at least the share of the part its name vote needs). Only used
# when the song has a SongFormer result. False gives pre23's section maps.
SONGFORMER_PRECHORUS_BEFORE_CHORUS = True

# 0.7.1-pre24: a Verse that comes after two or more choruses is a Bridge when SongFormer, over that
# part itself, hears a bridge (at least the share of the part its name vote needs) -- such as a
# featured singer's section late in the song. Only used when the song has a SongFormer result.
# False gives pre23's section maps.
SONGFORMER_LATE_BRIDGE = True

# 0.7.1-pre28: folders (and their subfolders) where /songformer scan looks for a song's audio when
# it is no longer where the song was analysed -- moved or renamed, recognised by its file name and
# by the size and modification time recorded in the save, as /analyse and /batch recognise a moved
# file. /songformer scan --audio-folder "<folder>" adds one for that run. Saves are never changed.
# Example: SONGFORMER_AUDIO_FOLDERS = ["~/Music/Musiclyse"]. Empty: only where it was analysed.
SONGFORMER_AUDIO_FOLDERS = []


# ==============================================================================
#  14. STEMS, NOTES AND MIDI
# ==============================================================================

DEMUCS_MODEL = "htdemucs_6s"
DEMUCS_SHIFTS_FAST = 0
DEMUCS_SHIFTS_DEEP = 2

# Omnizart's "music" app transcribes with its PIANO checkpoint unless told
# otherwise, so bass/guitar/other stems are transcribed by a piano-trained
# model. Name a different Omnizart mode per stem here (e.g. "Pop", "Stream",
# "PianoV2" -- whatever `omnizart download-checkpoints` installed) to change
# that. None keeps the previous behaviour. A mode that fails to load falls back
# to the default model automatically, and the report states which model ran.
OMNIZART_MUSIC_MODEL_BY_STEM = {"bass": None, "guitar": None, "piano": None, "other": None}

ENABLE_STEM_MIDI = True
STEM_MIDI_MAX_SECONDS = None   # cap stem/MIDI work for speed; set None for full-track MIDI.
STEM_MIDI_EVENT_LOG_MAX_NOTES = 24
STEM_MIDI_INCLUDE_EVENT_LOGS = False
FULL_MIDI_TRANSCRIPTION = True
STEM_MIDI_MELODY_LINE_NOTES = 24
STEM_MIDI_DRUM_PATTERN_HITS = 24
STEM_MIDI_COMPACT_EVENT_FORMAT = True
STEM_NOTE_FILTER_PRESET_WEIGHT = 0.6

# 0.7.1-pre6: retuned note filters for every stem (STEM_MIDI_PRESETS and the three floors
# below) -- more of the real notes kept, fewer artefacts. False gives 0.7's filters.
STEM_MIDI_PRE6_TUNING = True

# Steadies the vocal MIDI: a note a semitone off the sung pitch is corrected where the vocal's
# pitch contour settles, and broken-up fragments of one note are joined. Takes extra time on
# the vocal stem. False gives 0.7's vocal MIDI.
STEM_MIDI_STABILIZE_VOCAL_PITCH = True
STEM_NOTE_MIN_DURATION_SEC_FLOOR = 0.02 if STEM_MIDI_PRE6_TUNING else 0.03
STEM_NOTE_MIN_AMPLITUDE_FLOOR = 0.03 if STEM_MIDI_PRE6_TUNING else 0.05
STEM_NOTE_MIN_ABS_RMS = 0.0003 if STEM_MIDI_PRE6_TUNING else 0.001   # ~ -70.5 dBFS (0.7: -60 dBFS)

OMNIZART_PREFETCH_PROCESS = True
OMNIZART_PREFETCH_TIMEOUT_S = 1800


# ==============================================================================
#  15. DRUMS AND RHYTHM
# ==============================================================================

DRUM_EXTRAS_PROBE_ENABLED = True
DRUM_EXTRAS_FIRE_PROB = 0.15
DRUM_EXTRAS_REPORT_PEAK = 0.30
DRUM_EXTRAS_REPORT_COVERAGE = 0.20
BEAT_GRID_ENABLED = True
BEAT_GRID_BARS = 2
BEAT_GRID_SUBDIV = 4          # cells per beat (4 = sixteenth notes)
BEAT_GRID_MIN_HITS = 6
CYMBAL_ONSET_DETECT_ENABLED = True
CYMBAL_BAND_HZ = (6000.0, 16000.0)
CYMBAL_FLOOR_RATIO = 3.0
CYMBAL_MIN_SUSTAIN_RATIO = 0.10
CYMBAL_MIN_ATTACK_RATIO = 1.8
CYMBAL_MERGE_GAP_S = 0.30
CYMBAL_MIN_DETECTIONS = 3

# 0.7-pre5: bell/chime/glockenspiel/mallet tagger readings saved as
# "tuned_percussion" (saved only).
TUNED_PERCUSSION_PROBE = True

BEAT_FOLD_MIN_CONSISTENT_DB = 3.0

# DRUM_BLEED_NOTE / DRUM_STEM_BLEED_BELOW_TOTAL_DB: see _drum_stem_bleed_db.
DRUM_BLEED_NOTE = True
DRUM_STEM_BLEED_BELOW_TOTAL_DB = 18.0


# ==============================================================================
#  16. TEMPO, KEY, CHORDS AND MELODY
# ==============================================================================

# Key: when MOSS-Music and Essentia share the root but MOSS-Music gives no
# mode -- a bare root or a chord quality ("D aug" on an indie test track
# Garden, reported as the song's key) -- Essentia's mode is used. A mode name
# from MOSS-Music (major, minor, Dorian, Mixolydian, ...) is kept as before.
# Saved scans are repaired when loaded. False restores the earlier rule.
KEY_MODE_FROM_ESSENTIA = True

# FULL-BANDWIDTH TEMPO CHECK: the tempo detectors run on a 22.05 kHz mono copy,
# which drops everything above 11 kHz. On a track whose hi-hats and glitchy
# percussion live up there (the glitch-electronic test track: 15% of its energy above 11 kHz) the
# 140 BPM pulse disappears and the detectors lock onto a slow loop instead
# (35 and 70 BPM, reported as 35). When BOTH detectors read below
# FULL_BAND_TEMPO_SLOW_BPM, the original is re-measured at 44.1 kHz; if it
# really has content above 11 kHz and both detectors then agree on a tempo
# that is the same pulse at 2x or 4x the slow readings, the beat/tempo reports
# are rebuilt from the full-bandwidth copy. No other track is touched: of 53
# saved scans, only the glitch-electronic test track had both detectors below 80 BPM. False turns it off.
FULL_BAND_TEMPO_CHECK = True

# CHORDS line, one format on every track: the chords measured from the
# separated guitar / piano / other stems (major and minor triads and
# root-fifth "power chords", most common first, spelled for the key), then
# the listening model's own chords, each listed once with loops collapsed.
# Either part is marked where its chords fall outside the measured key.
# MOSS-Music's chord lists were wrong or missing on most scans (A1's
# "C - G - Ebmaj7" in C major, a dance-pop test track's "G - C6" in every section,
# a classic-rock test track's "Ab-C#" fifteen times over, "uncertain" on folk test track
# Fair), while the measured chords fit the key (C, G, F; C, F, G; Em, D, G).
# Applied when a scan is saved and when a saved scan is loaded. False keeps
# the listening model's line as it was.
CHORDS_MEASURED_LINE = True

# HARMONY AND MELODY BY SECTION: one line per section, from the transcribed
# notes of the pitched stems (drums never count): the chords heard most in
# that section (guitar / piano / other; ranked, not an exact progression),
# which part carries the lead line (the voice when it sings there, otherwise
# the most melody-like single-note part: guitar, piano or other -- never the
# bass), where that line sits against the chords and the bass, and how much of
# it lands on notes the accompaniment is playing at that moment. Lets the chat
# talk about the melody against everything else, section by section. Added to
# every track with transcribed notes and a section map, at scan time and when
# a saved scan is loaded. False turns it off.
HARMONY_MELODY_BY_SECTION = True
FULL_BAND_TEMPO_SLOW_BPM = 80.0
FULL_BAND_TEMPO_MIN_HF_SHARE = 0.02

# 0.7-pre8: ...but only when those other-sex clips are scattered among the
# majority's; two back to back are a second singer's passage.
HARMONY_CLIPS_SCATTERED_ONLY = True

# dev45 ENABLE_CHORD_PROGRESSION: a chord-recognition model (lv-chordia, MIT
# licence, PyTorch, weights included: pip install lv-chordia) reads the chords
# of the full mix in time order during the background signal processing (CPU,
# about 1 GB while it runs). A CHORD PROGRESSION BY SECTION block then gives each
# section's progression -- its repeating cycle folded ("A - D/F# - E - F#m, four
# times"), Roman numerals against the recommended key, how often the chords
# change and any chord outside the key -- and the chord segments are saved in
# the .json ("chord_segments"). Not installed: skipped with a one-line note.
# The existing chord lists, key and everything else are unchanged. False
# restores dev44.
ENABLE_CHORD_PROGRESSION = True
CHORD_PROGRESSION_DICTIONARY = "submission"   # lv-chordia's vocabulary: "submission", "ismir2017" or "full"

# dev45 MELODY_IN_KEY_BY_SECTION: the lead line of each section as scale degrees
# of the recommended key, from the transcribed notes already in every scan:
# where its phrases come to rest, its opening phrase, the song's recurring
# four-note figure and its highest note. Saved scans get it on /load from their
# stored notes, so no rescan is needed. False restores dev44.
MELODY_IN_KEY_BY_SECTION = True
MELODY_IN_KEY_MIN_FIT = 0.75   # a part is described only when this share of its held notes fits the key

# dev45 MELODICISM_MEASURED: a grounded reading of how melodic the song is,
# overall and part by part, from the lead vocal's range and share of long held
# notes and the word rate of the timed lyric lines. On the scanned library it
# put 9 of 11 rap tracks in its "rhythm-led" band and none of 9 ballads. Saved
# scans get it on /load. False restores dev44.
MELODICISM_MEASURED = True

# dev46 TEMPO_COPY_CHECK: the listening model's TEMPO_BPM is not counted as a
# separate opinion when it is a copy, to the decimal, of a detector reading it
# was shown (a disco test track: its 198.8 was the beat tracker's own beat spacing,
# so "two sources agree" picked 200 BPM over Essentia's 100). On 213 saved
# scans this changed six tempos, all to the musically natural count (a disco test track 100, a pop test track 105, a rock test track 115, another rock test track 86, a gothic-rock test track
# 120) and nothing else. Saved scans get it on load. False restores dev45.
TEMPO_COPY_CHECK = True

MODAL_DETECTION_MIN_NOTES = 24     # too few notes can't settle a mode
MODAL_DETECTION_MIN_SHARE = 0.04   # deciding pair must carry >=4% of note weight
MODAL_DETECTION_MIN_RATIO = 4.0    # characteristic degree must beat its rival 4:1

# Key centre check (see _key_centre_reading): a major key whose music
# centres on its fifth, with the flattened seventh of that fifth and no
# leading tone, is reported as Mixolydian on the fifth ("D Mixolydian (the
# notes of G major, centred on D)"). Every test must pass; otherwise the key
# is left exactly as decided. False switches the check off.
KEY_CENTRE_CHECK = True

# dev20 (patch) KEY FIT CHECK: the key detector only knows major and minor
# templates, so it can name a neighbouring key that shares all but one note
# (a song in D# minor -- the notes of F# major -- named C# major). When the
# transcribed notes of every pitched part fit another key at least
# KEY_FIT_MARGIN better than the reported one, that key differs from it by at
# most one scale note, and the bass sits on that key's tonic more than on any
# other note, the RECOMMENDED KEY becomes that key, with the reason. Scans
# and /load. False restores the earlier behaviour.
KEY_FIT_CHECK = True
KEY_FIT_MARGIN = 0.05
KEY_CENTRE_MIN_NOTES = 40          # notes in the low part and in the vocal line
KEY_CENTRE_MIN_RATIO = 1.5         # the fifth must outweigh the old root this much
KEY_CENTRE_FLAT7_MIN_SHARE = 0.03  # the new centre's flat 7th in the chord parts
KEY_CENTRE_FLAT7_RATIO = 3.0       # ... against its leading tone

# dev18 key note check (see _key_note_reading): when the listening model and
# Essentia name different keys, both are scored against the transcribed notes
# of every pitched stem (duration-weighted, Krumhansl-Kessler key profiles).
# A listening-model key the notes do not fit is set aside: next to a strong
# Essentia key it no longer counts as a disagreement, and it no longer wins
# over a weak Essentia key the notes fit. MOSS-Music answers "Eb major",
# "Eb minor" or "Ab minor" on most tracks whatever they are in; on fourteen
# saved scans those answers fit the notes at -0.65 to 0.07 while Essentia's
# keys fit at 0.73 to 0.96. A listening-model key the notes fit (a relative
# major/minor, as on the child-pop test track) is left alone.
KEY_NOTE_CHECK = True
KEY_NOTE_MIN_NOTES = 200           # pitched notes across the stems
KEY_NOTE_POOR_FIT = 0.5            # the listening model's key fits below this
KEY_NOTE_GOOD_FIT = 0.6            # Essentia's key fits at least this
KEY_NOTE_MARGIN = 0.3              # ... and this much better
KEY_NOTE_NEAR_BEST = 0.1           # ... and within this of the best-fitting key

# dev26 KEY_SCALE_CHECK: the key detector (Essentia, and the listening
# model) can settle on a key a fifth away from the song's own -- the two share
# six of seven notes, and the dominant or subdominant chord's notes can
# outweigh the tonic's (a ballad's I-V-vi-IV on C, F and G read as C major
# when the song is in F). The one note that tells the two keys apart decides:
# when the reported key's own note (C major's B) is all but absent from the
# transcribed notes of the pitched parts -- at most KEY_SCALE_ABSENT_RATIO of
# the neighbouring key's note (B flat), which carries KEY_SCALE_PRESENT_MIN or
# more of their note time -- the key is the best-fitting of that neighbour's
# major and relative minor (or of the next one along, if it fits
# KEY_SCALE_NEXT_MARGIN better: a song a whole step away after one
# modulation). Not when the old tonic still rules both the bass and the
# chord parts (KEY_SCALE_GUARD_RATIO times the new tonic): that is a song
# keeping its centre and borrowing one note (Mixolydian, Dorian), and its key
# stays. New scans and saved files when loaded. False restores dev25.
KEY_SCALE_CHECK = True
KEY_SCALE_ABSENT_RATIO = 0.2
KEY_SCALE_PRESENT_MIN = 0.05
KEY_SCALE_NEXT_MARGIN = 0.15
KEY_SCALE_GUARD_RATIO = 1.5

# dev28 KEY_CHANGE_CHECK: a song that moves to a new key for its last part
# (a final chorus up a semitone) is reported in the key it starts in, with
# the change named -- the key detector, which measures the whole song at once,
# averaged the two keys into a third ("Bb minor" for a song in G major that
# moves to Ab). Only when the notes before and after the change each fit
# their key well (KEY_CHANGE_MIN_FIT) and better than any one key fits the
# whole song (KEY_CHANGE_MIN_GAIN). New scans and saved files when loaded.
KEY_CHANGE_CHECK = True
KEY_CHANGE_MIN_FIT = 0.75
KEY_CHANGE_MIN_GAIN = 0.1
KEY_CHANGE_MIN_TAIL_S = 30.0

# dev19 tempo checks (see _tempo_listening_check / _tempo_snare_tiebreak).
# MOSS-Music's TEMPO_BPM decided three of ten tempos wrongly where 0.5's
# listening model had them right (the teen-pop test track 127 for 146, the boy-band test track 120 for
# 108, the alt-rock test track 78 for 156). A TEMPO_BPM that only repeats the prompt
# is not used; when the listening model alone would decide against two
# disagreeing detectors, a detector that matches its own beat spacing decides
# instead; and when two readings ~2x apart are settled only by the "prefer the
# lower pulse" default, a snare that lands every two beats of the higher one
# breaks the tie. Every other tempo on the 28 saved scans tested is unchanged.
TEMPO_LISTENING_CHECK = True
TEMPO_SELF_CONSISTENT_TOL = 0.03   # raw estimate vs its own median beat spacing
TEMPO_SNARE_TIEBREAK = True
TEMPO_SNARE_TOL = 0.04             # snare backbeat vs the higher reading
TEMPO_SNARE_MIN_INTERVALS = 40     # snare-to-snare gaps needed
TEMPO_SNARE_MIN_SHARE = 0.5        # share of gaps in the dominant spacing

# dev20 (patch) TEMPO OCTAVE PAIR CHECK: the two detectors agree with each
# other an octave apart (105.1 and 210.2) and the listening model's pulse is
# neither of them nor their half or double (114), yet counted as "~2x" the
# faster detector and kept. When the snare on the separated drums lands every
# two beats of the detector reading nearest the listening model's pulse, that
# reading is used instead. False restores the earlier behaviour.
TEMPO_OCTAVE_PAIR_CHECK = True
TEMPO_OCTAVE_PAIR_TOL = 0.05       # the listening model's pulse vs each detector octave

# 0.7-pre60 TEMPO_HALFTIME_CHECK: both detectors agree on a fast tempo -- at least
# TEMPO_HALFTIME_MIN_BPM and above the usual range of the genre family the tempo
# note names (never drum & bass) -- and the separated drums play one main snare
# per bar at it (a half-time backbeat) in at least TEMPO_HALFTIME_SHARE of the
# song's TEMPO_HALFTIME_WINDOW_BEATS-beat stretches: the detectors counted an
# ordinary 2-and-4 backbeat at double speed, and half their reading is used. The
# detectors share one blind spot, so their agreement had settled the tempo with no
# other check. New scans (the drum report and its beat grids are then made at the
# new tempo, as for the dev19 snare check) and saved files in the section-map
# replay. False restores pre59.
TEMPO_HALFTIME_CHECK = True
TEMPO_HALFTIME_MIN_BPM = 145.0
TEMPO_HALFTIME_SHARE = 0.4
TEMPO_HALFTIME_WINDOW_BEATS = 64
TEMPO_HALFTIME_MIN_WINDOWS = 4

# 0.7-pre60 TEMPO_HALFTIME_LISTENING_PAIR: TEMPO_HALFTIME_CHECK also when the fast
# reading was settled by the listening model agreeing with one detector, provided
# the other detector measured half of it (an exact octave, within 3%). False: only
# when the two detectors agree, as above.
TEMPO_HALFTIME_LISTENING_PAIR = True

# 0.7-pre60 TEMPO_HALFTIME_BOTH_READINGS: both detectors agree on a tempo from
# TEMPO_HALFTIME_BOTH_MIN_BPM up to TEMPO_HALFTIME_MIN_BPM, above the genre family's
# usual range (never drum & bass), with the half-time drums TEMPO_HALFTIME_CHECK
# looks for: the tempo stays, and its note says the song can also be counted at
# half speed (the exporter then states both). Too close to ordinary fast tempos to
# halve. False restores pre59.
TEMPO_HALFTIME_BOTH_READINGS = True
TEMPO_HALFTIME_BOTH_MIN_BPM = 140.0

# 0.7-pre60 TEMPO_SNARE_HALFTIME_GUARD: the dev19 snare tie-break and the dev20
# octave pair check move the tempo to the faster of two readings only when the
# drums do not play a half-time backbeat at that faster tempo (the measure of
# TEMPO_HALFTIME_CHECK). The tie-break reads the snare's most common spacing,
# which extra snare hits can make look like a backbeat at double speed while the
# main snare falls once per bar. False restores pre59.
TEMPO_SNARE_HALFTIME_GUARD = True

# 0.7-pre60 (patch): the guard's own share of half-time stretches. Lower than
# TEMPO_HALFTIME_SHARE because the guard only keeps the slower of two readings the
# detectors already gave: the drum transcription varies a little between scans, and
# one song measured 4 of 8 stretches on one scan and 3 of 8 on the next; songs the
# tie-break rightly moves to the faster reading measure none. 0.4 gives pre60.
TEMPO_SNARE_HALFTIME_GUARD_SHARE = 0.3

# dev39 TEMPO_MEASURED_OCTAVE_PAIR: when Essentia and the beat tracker are an
# exact octave apart (ratio 2 within TEMPO_MEASURED_OCTAVE_PAIR_TOL) and the
# listening model's own figure matches neither of them, the two measurements
# decide between themselves (genre range, then mid-range pulse, as for any
# pair ~2x apart). The model's figure had been paired with one measurement
# first, and any ratio from 1.8 to 2.2 counted as "double": one pop test track
# measured 130.5 and 262 (131) but was reported at the model's 142. On 106
# songs with a published or Harmonix tempo: exact 83% -> 86%, wrong 8% -> 5%;
# 9 of 189 saved songs change, all to the published tempo or an octave of it.
# False restores dev38.
TEMPO_MEASURED_OCTAVE_PAIR = True
TEMPO_MEASURED_OCTAVE_PAIR_TOL = 0.04

# The block's semitone gaps and match percentages were read out to the
# person as they stood ("the line about 15 semitones below the chords, ~67%
# of the vocal notes matching the accompaniment"), section after section.
# With this on the block says the same things in words -- above / inside /
# below the chords, roughly how many octaves above the bass, and whether the
# line mostly sits on the accompaniment's notes -- with the same thresholds.
# Saved scans get the reworded block when loaded. False restores the figures.
HM_PLAIN_FIGURES = True


# ==============================================================================
#  17. GENRE, STYLE AND SOUND SIMILARITY
# ==============================================================================

ENABLE_CLAP_EMBEDDING = True
CLAP_MODEL_ID = "laion/larger_clap_music_and_speech"
CLAP_EMBEDDING_DIMENSION = 512
CLAP_SAMPLE_RATE = 48000
CLAP_WINDOW_SECONDS = 10.0
CLAP_WINDOW_HOP_SECONDS = 5.0
CLAP_MAX_WINDOWS = 24
CLAP_USE_MPS = False  # save-time feature; avoid competing with Ollama on unified memory
CLAP_EMBEDDING_PRECISION = 8  # decimal places stored in JSON; vectors remain normalized

# When the genre check replaced the listening model's first-pass rock/punk
# label, a GENRE CHECK line at the top of the notes says so. False restores
# the dev39 behaviour.
# (added in dev40)
GENRE_SUPERSEDED_NOTE = True

# With no genre from the listening model, the PANNs tags are grouped by
# family ("rock, with electronic lean") instead of crowning the single top
# tag ("ska" on a synth-led rock track). False restores the dev39 behaviour.
# (added in dev41)
GENRE_LAST_RESORT_BY_FAMILY = True

# The genre phrase in the listening model's own prose is still found when a
# leading descriptor trips the junk-label filter ("confidence-driven rock"
# was thrown out because of "confidence"). False restores dev41.
# (added in dev41)
GENRE_PROSE_HINT_TRIM = True

# A genre taken from that prose keeps a strong second family from the
# independent tags as a lean ("rock (with electronic lean)"). False restores
# the dev41 behaviour.
# (added in dev41)
GENRE_SECOND_FAMILY_LEAN = True

# The GENRE CHECK line (a rock/punk first impression replaced by the genre
# check) now also covers the listening model's guitar-band wording built on
# that first impression, says how to answer "is it rock / pop-rock?", and
# holds for era and instrumentation answers. Unless the separated guitar stem
# is measured as a leading part, the track is not to be called rock, pop-rock
# or the replaced label. Also fires when the genre check took the listening model's own
# lower-ranked pop/dance label over a rock/punk top. False restores dev46.
# (added in dev47)
GENRE_CHECK_FIRM = True

# Genre check: when the listening model's top label is rock/punk and the independent tags lean
# pop/dance, a pop OR dance label from the listening model's own ranked list can be used (dev46
# accepted only one in the single strongest family). New scans only. False
# restores dev46.
# (added in dev47)
GENRE_ROCK_TOP_POP_SIDE_ALT = True

# Genre (dev55):
#  * the genre phrase in the listening model's prose is found when written
#    "Indie Pop / Synth-Pop piece" (a spaced slash) or after an adjective list
#    ("an uplifting, high-energy electronic pop song"); compound labels keep
#    both parts ("indie pop / synth-pop"). that stuck-pitch scan came out "rock" from a single
#    AudioSet tag because the prose phrase was missed.
#  * a bare family word from the ranked list ("rock") is refined by a more
#    specific label of the same family the listening model wrote in its own
#    prose ("Classic Rock / Hard Rock"; one self-check-loop test scan).
#  * the reverse of the existing rock->electronic check: an electronic/pop
#    ranked list with no rock label at all, against independent tags whose
#    rock family clearly outweighs electronic+pop (rock 40%+ and 1.25x), is
#    framed "rock (with <ranked top> lean)" (one post-punk structure test scan: "electronic" while
#    grunge + psychedelic rock + rock made 54% against 41%).
#  * no "(with pop lean)" on a label that already says pop.
# False restores dev54.
GENRE_DEV55 = True

# dev56 genre checks:
#  * the rock-over-electronic reading above (one post-punk structure test scan) on a dynamic,
#    lightly limited master (integrated loudness -12.5 LUFS or quieter and a
#    peak-to-loudness ratio of 12 dB or more -- before the loudness race) is
#    framed "post-punk / new wave (with electronic lean)": rock heard by the
#    listening model as electronic, with an undriven, uncompressed master, is
#    that sound; the plain "rock (with electronic lean)" did not say so.
#  * a "Genre: electronic pop / synth-pop / dance-pop" line (the listening
#    model's own list, written without GENRE_RANKED) is read as its ranked
#    list (one multi-chorus test scan fell back to the AudioSet tags: "electronic (with pop
#    lean)").
#  * the tempo block's "Genre prior" says it is the first impression used for
#    half/double-time choices only, not the track's genre.
#  * when the genre check turned an electronic/pop first impression into
#    rock, a GENRE CHECK line says so, as it already did the other way round
#    (one post-punk structure test scan's notes still opened "Genre_Ranked=1) electronic;
#    2) synth-pop; 3) dance-pop; 4) house-influenced pop"), and genre
#    questions and blind era answers get the same warning.
# False restores dev55.
GENRE_DEV56 = True

# Correct only the legacy mastering-based scene inference and corroborated dance/rock conflict.
# False restores the previous decisions. No artist, title or release-year conditions.
GENRE_EVIDENCE_GUARD = True
GENRE_EVIDENCE_STYLE_MIN = 0.10
GENRE_EVIDENCE_STYLE_TOP_K = 5

# 0.8: with GENRE_EVIDENCE_GUARD, when the mastering-based post-punk / new wave promotion fired, the
# independent style model decides what happens instead of the listening model's first label:
# post-punk or new wave among its leading styles keeps the promotion; mostly rock styles give
# "alternative / indie rock (with ... lean)" when indie or alternative rock leads, otherwise
# "rock (with ... lean)"; anything else keeps the listening model's first label as before.
# Uses GENRE_EVIDENCE_STYLE_MIN and GENRE_EVIDENCE_STYLE_TOP_K. False gives the guard's previous
# decisions.
GENRE_EVIDENCE_STYLE_READING = True

# dev56: the GENRE CHECK line (either direction) is also written at the top of
# the saved scan's "analysis", so the JSON itself says that the opening genre
# ranking was replaced. The writer already saw this line; it is not added
# twice. False restores dev55.
GENRE_CHECK_IN_SAVED_ANALYSIS = True

ENABLE_GENRE_MOOD_TAGGING = True
GENRE_TAG_TOP_K = 5
GENRE_TAG_MIN_PROB = 0.05
MOOD_TAG_TOP_K = 3
MOOD_TAG_MIN_PROB = 0.08

# dev42 ENABLE_DISCOGS_STYLES: a third, finer genre signal. MAEST, a music
# classifier trained on Discogs, names the record-shop styles the recording
# sounds like from 400 (Disco, Europop, Beat, Psychedelic Rock, Britpop, Glam,
# New Wave, ...). Shown as a DISCOGS STYLE SIGNAL block after the recommended
# genre and saved in the song's .json; it adds detail and does not change the
# recommended genre, the key/tempo decisions or any other result. Runs on the
# CPU during the background signal processing (a few seconds per song; about
# 0.5 GB while it runs, freed afterwards). Needs a one-time download (~350 MB):
#     python musiclyse_style_setup.py
# Until then it is skipped with a one-line note. Model licence: CC BY-NC-SA 4.0
# (non-commercial), like Essentia's models. False restores dev41.
ENABLE_DISCOGS_STYLES = True
DISCOGS_STYLE_MODEL_ID = "mtg-upf/discogs-maest-30s-pw-73e-ts"
DISCOGS_STYLE_WINDOWS = 5          # 30-second windows spread over the song, averaged
DISCOGS_STYLE_TOP_K = 5            # styles listed in the block
DISCOGS_STYLE_MIN_PROB = 0.10      # ... at or above this confidence
DISCOGS_STYLE_SAVE_TOP = 40        # strongest styles kept in the saved .json

# Genre clarification leaves ruled-out labels as they are and accepts "indie folk."
GENRE_CLARIFY_SKIP_RULED_OUT = True

# 0.8-pre31: Genre: a negative or hedging word after the genre's own noun ("a synth-pop track
# that never lets up", "an indie rock song with no shortage of hooks") no longer throws out the
# sentence a fallback genre is read from. Needs GENRE_ASSERTED_PROSE. False gives pre30's.
GENRE_PROSE_TRAILING_CLAUSE = True

# Genre: a fallback genre is never taken from a negated or hedged sentence ("not jazz",
# "could be read as house").
GENRE_ASSERTED_PROSE = True

# Genre: two different labels that are both outside the rock/electronic/pop families (jazz and
# swing, folk and country) are not treated as agreeing.
GENRE_OTHER_FAMILY_STRICT = True


# ==============================================================================
#  18. INSTRUMENTS AND ARRANGEMENT
# ==============================================================================

# dev60: the instrument tagger (AudioSet) names sustained, reedy synth pads,
# organ-like synths and sax lines "accordion", "harmonica" or "harmonium":
# one verse-first pop test scan's and one multi-chorus test scan's synth pads were tagged "accordion",
# one self-check-loop test scan' saxophone "harmonica" (81%), and the chat repeated
# them. When such a tag is the only evidence -- the listening model never
# names that instrument and the genre is not one where it is common (folk,
# country, blues, cajun/zydeco, tango, polka, world...) -- an INSTRUMENT CHECK
# line says it is unconfirmed and to describe the part by its sound. Tags and
# every measurement are unchanged. False restores dev59.
INSTRUMENT_REED_TAG_CHECK = True

# A 'harmonica' tag that is strong (75%+), steady (75%+ of windows) and not a
# weak stem-only reading is kept as a real reedy lead -- a harmonica or a horn
# such as a saxophone -- instead of being dismissed: on a pop-rock test
# track (83-85%, 43 of 48 windows) the INSTRUMENT CHECK told the writer
# to drop the track's main lead instrument, which the listening model had
# missed. Weaker readings (a folk test track's 61% stem-only) are still
# dismissed. And when a dismissed reed tag sits on an 'other' stem playing
# chords of 4+ notes at once, the check says it is most likely a synth pad
# (the accordion readings on a pop test track, the teen-pop test track and two rock-ballad
# test tracks). False restores the earlier wording.
INSTRUMENT_REED_STRONG_LEAD = True
REED_STRONG_LEAD_MIN_PCT = 75
REED_STRONG_LEAD_MIN_WINDOW_SHARE = 0.75
REED_PAD_MIN_POLYPHONY = 4.0

# GUITAR CHECK: when the instrument tagger reads the guitar stem as acoustic
# guitar in every section and nothing (no section, stem or whole-mix tag)
# reads an electric guitar, a listening-model mention of electric guitar is
# recast as at most a clean, lightly amplified or semi-acoustic guitar (never
# distorted), in the writer's notes and in the report's INSTRUMENTATION.
# MOSS-Music added "electric guitar (subtle textures)" to a folk-rock test track
# (acoustic, at most semi-acoustic, throughout) and "electric guitar
# (likely clean or lightly overdriven)" to the amplified-folk test track (an amplified
# semi-acoustic-like part, not distorted). False turns it off.
ACOUSTIC_GUITAR_CHECK = True

# Verify ANY human voice over the entire mix only after all lead samples fail.
INSTRUMENTAL_FULL_COVERAGE_CHECK = True

# False restores v9 retry depth: faster on ambiguous windows, but may leave
# vocal presence unresolved. This does not change Wikipedia identity lookup.
INSTRUMENTAL_EXTRA_PRESENCE_RETRY = True
INSTRUMENTAL_PRESENCE_MAX_CLIPS = 90

# A time-stamped instrument list ("electricguitar (1.00s-11.99s);
# electricguitar (11.99s-22.98s); bass (22.98s-23.98s); ...") is shown with
# each instrument once, in order of first appearance, any description kept.
INSTRUMENTATION_COLLAPSE_SPANS = True

# Tag up to this many 10-second windows per instrument-tagger call instead of
# one at a time (same windows, same model; the network is run batched).
PANNS_BATCH_WINDOWS = 8

ENABLE_INSTRUMENT_TAGGING = True
INSTRUMENT_TAG_MIN_PROB = 0.18
INSTRUMENT_TAG_TOP_K = 4
INSTRUMENT_TAG_GUITAR_MIN_PROB = 0.35
INSTRUMENT_TAG_STEMS = ("other", "guitar", "piano", "bass", "drums")
INSTRUMENT_TAG_REQUIRE_STEM_ACTIVITY = True
INSTRUMENT_TAG_MIN_NOTES_FOR_WEAK = 8
INSTRUMENT_TAG_STRONG_PROB = 0.45
INSTRUMENT_TAG_WINDOW_SECONDS = 10.0
INSTRUMENT_TAG_HOP_SECONDS = 5.0
INSTRUMENT_TAG_MAX_WINDOWS = 56
INSTRUMENT_TAG_NORMALIZE = True
INSTRUMENT_TAG_NORMALIZE_PEAK = 0.95
INSTRUMENT_TAG_EVEN_COVERAGE = True
INSTRUMENT_TAG_MIN_WINDOW_HITS = 2
INSTRUMENT_TAG_SINGLE_WINDOW_STRONG_PROB = 0.50
INSTRUMENT_TAG_SUSTAINED_COVERAGE = 0.25
INSTRUMENT_TAG_SUSTAINED_PROB_RELIEF = 0.85

# dev54: 3 -> 12. Output-identical (pure cache of per-window probabilities);
# stops mix/stem matrices evicting each other and being recomputed (~50x527 float32 each).
PANNS_WINDOW_CACHE_SIZE = 12
PANNS_TAG_STATS_CACHE_SIZE = 32

# Windows this far below the file's loudest window, or below this absolute RMS,
# are not tagged (they are peak-normalised residue, not an instrument).
PANNS_QUIET_WINDOW_REL_DB = 35.0
PANNS_QUIET_WINDOW_ABS_RMS = 1e-4
ENABLE_WHOLE_MIX_INSTRUMENT_TAGGING = True
WHOLE_MIX_INSTRUMENT_TAG_TOP_K = 6
WHOLE_MIX_INSTRUMENT_TAG_MIN_PROB = 0.15
INSTRUMENT_TAG_MIX_AGREE_MIN_PROB = 0.18

GUITAR_PROBE_ENABLED = True
GUITAR_PROBE_FIRE_PROB = 0.12
GUITAR_PROBE_PRESENT_PEAK = 0.22
GUITAR_PROBE_PRESENT_COVERAGE = 0.15
GUITAR_PROBE_PRESENT_COVERAGE_PEAK = 0.15
GUITAR_PROBE_ABSENT_PEAK = 0.10
GUITAR_PROBE_ABSENT_COVERAGE = 0.04
GUITAR_STEM_ACTIVE_MIN_RMS = 0.02
GUITAR_STEM_ACTIVE_MIN_NOTES = 60

INSTRUMENT_SECTION_PRESENCE_ENABLED = True
INSTRUMENT_SECTION_MIN_PROB = 0.30
INSTRUMENT_SECTION_MAX_LABELS = 6
ARRANGEMENT_CHANGE_ENABLED = True
ARRANGEMENT_SPAN_GAP_S = 3.0
ARRANGEMENT_MIN_SPAN_BARS = 2.0
ARRANGEMENT_CLUSTER_BARS = 1.0
ARRANGEMENT_STRONG_MIN_STEMS = 2

CROSS_STEM_CONVERGENCE_MIN_STEMS = 2
CROSS_STEM_CONVERGENCE_MIN_PROB = 0.45
CROSS_STEM_CONVERGENCE_MIN_SUSTAINED_STEMS = 2
CONVERGENCE_EXTENT_CONFLICT_ENABLED = True

# 0.6.2 (Stage 1) MTG-JAMENDO INSTRUMENT TAGS: a second, independent instrument
# classifier (MTG's Discogs-EffNet + MTG-Jamendo instrument head, 40 classes,
# finer on electronic sounds), run on the full mix and the separated stems with
# onnxruntime on the CPU. Saved in the .json as "jamendo_instruments" only; it is
# NOT added to the analysis text, so the writer and every existing decision are
# unchanged. One-time download: python musiclyse_jamendo_setup.py (until then
# skipped with a one-line note). Licence CC BY-NC-SA 4.0 (non-commercial).
# False: no tagging and no new field -- the scan and save are exactly as before.
ENABLE_JAMENDO_INSTRUMENTS = True

JAMENDO_MODEL_DIR = ""             # "" = ~/mtg_jamendo_data (shared by every install)
JAMENDO_EMBED_FILE = "discogs-effnet-bsdynamic-1.onnx"
JAMENDO_HEAD_FILE = "mtg_jamendo_instrument-discogs-effnet-1.onnx"
JAMENDO_LABELS_FILE = "mtg_jamendo_instrument-discogs-effnet-1.json"
JAMENDO_STEMS = True               # also tag each separated stem
JAMENDO_COVERAGE_PROB = 0.3        # a patch counts towards "coverage" at or above this
JAMENDO_SECTION_MIN_PROB = 0.25    # labels kept per section when they reach this
JAMENDO_QUIET_REL_DB = 35.0        # patches this far below the source's loudest are skipped
JAMENDO_THREADS = 4                # onnxruntime CPU threads

# Absolute floor for INTENSITY BY SECTION: a stem whose RMS stays below this
# (~ -54 dBFS) in every section is reported as not playing, and a single
# section below it AND at least 30 dB under the stem's own loudest section is
# reported as not playing there, whatever the stem's relative scale says.
# Without it a stem that is separation residue throughout (e.g. a "piano" stem
# on a track with no piano) was labelled "peak intensity" in whichever section
# its residue happened to be loudest. The second condition keeps genuinely
# quiet passages of very quiet recordings on the relative scale.
SECTION_AUDIBLE_MIN_RMS = 0.002

# A stem section this far below the combined stems in that section is residue.
SECTION_RESIDUE_BELOW_TOTAL_DB = 30.0

BASS_STEM_ABSENT_BELOW_TOTAL_DB = 20.0


# ==============================================================================
#  19. PRODUCTION, MIX AND LOUDNESS
# ==============================================================================

ENABLE_ESSENTIA_REPORT = True
ESSENTIA_MAX_SECONDS = 900.0
ESSENTIA_LOWLEVEL_MAX_SECONDS = 60.0
ESSENTIA_FRAME_SIZE = 4096
ESSENTIA_HOP_SIZE = 2048

MIX_PASS_LOW_POWER_MAX_TAGS = 2
MIX_PASS_LOW_POWER_MAX_TOP_PROB = 0.35

PRODUCTION_ANALYSIS_ENABLED = True
PRODUCTION_EXCERPT_COUNT = 8
PRODUCTION_EXCERPT_SECONDS = 6.0
PRODUCTION_ENVELOPE_SR = 22050


# ==============================================================================
#  20. THE ANALYSIS REPORT
# ==============================================================================

COMPACT_OBJECTIVE_REPORT = True

ENABLE_OBJECTIVE_AUDIO_REPORT = True

# dev64: every saved analysis gets the fixed layout of CHANGES IN 0.4-dev63 --
# the CHECKS block, then the same 18 report fields in the same order
# ("uncertain" where none was given; STRUCTURE = the measured section map),
# DESCRIPTION= and REPORT_NOTE= -- whether or not MOSS_CONSISTENT_REPORT is on.
# This is done when the JSON is written, after every analysis decision, from
# the listening model's own report and its first pass, so section maps,
# genre, key, tempo and everything else measured are exactly as without it.
# (MOSS_CONSISTENT_REPORT additionally changes how the model is asked, so its
# report is more complete before the checks use it.) False restores dev63.
REPORT_FIXED_LAYOUT = True

# dev66: with REPORT_FIXED_LAYOUT, report fields the listening model left out
# are filled instead of saying "uncertain" (most scans: its report is prose, or
# a ranked genre list that looped, so 12-18 of the 18 fields were empty):
#  * descriptive fields (INSTRUMENTATION, DRUMS, TIMBRE, MOOD_VIBE, MELODICISM,
#    MELODIC_CONTOUR, CHORDS, LEAD_VOCAL_CHARACTERISTICS, LYRIC_SUBJECT) with
#    the listening model's own sentences from its report prose or its era-pass
#    description (sentences that date the track, name a genre or give a key or
#    tempo are left out -- those have their own checked blocks);
#  * KEY, TEMPO_BPM, VOCALS_PRESENT, LYRICS_PRESENT and MELODIC_RANGE from the
#    measured results already in the file (RECOMMENDED KEY / TEMPO, the vocal
#    decision, the lyrics, the VOCAL PITCH REPORT);
#  * DESCRIPTION, when none was given, from the era-pass description.
# REPORT_NOTE says which fields came from where. Genre fields are never filled.
# Done when the JSON is written; no measurement or decision changes.
# False restores dev65.
REPORT_FILL_FROM_EVIDENCE = True

# dev67: the blocks below the report in the same places in every saved file:
#  * PRODUCTION-CHARACTER TAGS is always the last part of the production
#    report ("none above the detection threshold" when the classifier found
#    nothing -- it used to be left out);
#  * COVER ART OBSERVATIONS is always there, before the singer-identity block
#    ("none -- this file has no embedded cover art" when there is none; the
#    chat side does not treat that as cover art);
#  * the measured section map's second copy is always its own STRUCTURE=[...]
#    block right after the era pass (it was inside the era description on
#    some songs and after the vocal-onset check on others).
# Done when the JSON is written; the blocks' contents are unchanged.
# False restores dev66.
REPORT_FIXED_BLOCKS = True

# dev73: tidy two report fields in saved files (see _saved_report_tidy).
# False restores dev72.
REPORT_TIDY_SAVED = True


# ==============================================================================
#  21. VIDEO
# ==============================================================================

# Experimental visual-report input budget (estimated text tokens, before song context).
VIDEO_CHAT_MAX_TOKENS = 32000
VIDEO_CHAT_AUTO_FRAMES = 0  # 0–8 evenly spaced reference images per loaded video; zero keeps text-only loading.

MUSICVIDEO_PAIR_FILE = "musicvideo-pair.json"   # in the video's folder: which song save it pairs with

# Video files: embedded artist/title tags are ignored; both come from the filename.
# (added in 0.7-pre53; False gives pre52's behaviour)
VIDEO_IGNORE_TAG_ARTIST_TITLE = True

# /videoanalyse with no name uses the audio analysis save name ("Artist - Song (music video)").
VIDEO_DEFAULT_SAVE_NAME = True

# When a loaded song's analysis was made from the same video file (same file name,
# durations within a second), chat is told song time = video time.
# (added in 0.7-pre54; False gives pre53's behaviour)
VIDEO_SAME_SOURCE_SYNC = True

# /musicvideoanalyse <video>: analyse the song (audio), then the pictures (video), from one
# music video file. /musicvideoload <name>: load the song's text export and the video
# together, already paired, with matching timings.
# (added in 0.7-pre56; False gives pre55's behaviour)
MUSICVIDEO_COMMANDS = True

# The song export /musicvideoload loads, and the one its export command makes:
# a level (brief, standard, plus, high, full) and whether it has Wikipedia background.
# (added in 0.7-pre56; False gives pre55's behaviour)
MUSICVIDEO_EXPORT_LEVEL = "plus"
MUSICVIDEO_EXPORT_WIKI = False

# (added in 0.7-pre65; no change to the analysis or the exports)
VIDEO_FRIENDLY_USAGE = True        # /videoanalyse or /musicvideoanalyse without a video: a usage line, not argparse's

# 0.8-pre31: a video file's cover art is only a picture actually embedded as cover art; one
# without no longer gets the video's first frame (usually black) as its "cover". False gives
# pre30's.
VIDEO_NO_FRAME_COVER = True

# 0.8-pre31: music videos (/musicvideoanalyse, or /analyse of a video file) with no cover art:
# before the singer identity, up to 8 frames taken at moments the lead vocal is sounding go to the
# chat model in one question -- how many different people are seen singing, their apparent sex
# and age range, and what they look like. The identity question itself is unchanged; the answer
# only feeds the rule steps that already read the cover, and only when someone is seen singing
# (never for an artist documented or described as a woman). It is kept in the save
# ("video_performers"). False gives pre30's scans.
VIDEO_PERFORMERS = True

# 0.8-pre31: when a loaded video's timing matches its song (both from the same file, or an offset
# given when it was analysed), the note pairing them also lists what the video shows in each of
# the song's sections. False gives pre30's note.
VIDEO_SECTIONS_NOTE = True

# 0.8-pre31: a loaded video also comes with up to this many of its frames for the chat model to see
# (0-8), picked from different shots, frames where someone is seen singing first, never black or
# blank ones. VIDEO_CHAT_AUTO_FRAMES above 0 takes its place (evenly spaced frames). 0 gives pre30's
# text-only loading.
VIDEO_CHAT_KEY_FRAMES = 6


# ==============================================================================
#  22. OTHER SETTINGS
# ==============================================================================

# 0.8-pre34: /analyse, /reanalyse and /musicvideoanalyse take two image options.
#   --cover "<image>"  the track's cover for this scan, in place of any artwork in the file: it is
#                      described, used to work out who sings, saved and exported like embedded artwork.
#   --photo "<image>"  a picture of the artist, used only to work out who sings (shown to that step as
#                      a photo of the artist, not as the cover). It isn't saved; the cover stays the cover.
# With either one, /musicvideoanalyse scans the song again even when it is already saved.
# False: the options aren't offered.
SCAN_IMAGE_OPTIONS = True

# 0.8-pre35: with --photo, the photo identifies the lead singer in every identity rule, while a group or
# duo the cover shows (or the documentation names) stays one -- a photo of one singer never makes it a
# solo act. False gives pre34's handling (the photo only when the cover shows nobody).
SCAN_PHOTO_IS_LEAD = True


# ##############################################################################
# EXPORTER SETTINGS
# ##############################################################################


# ==============================================================================
#  E1. WHO SINGS: SINGER IDENTITY WORDING
# ==============================================================================

# The singer named from any wording that puts a person's name after
# "vocalist" / "singer" / "frontman" / "frontwoman" in the same clause, words
# in between allowed ("lead vocalist for the 1983-era line-up is Firstname
# Lastname"); a band with no singer named is "Band H's singer", not "Band H sings".
# (added in 2.2.3; False gives 2.2.2's output)
SINGER_NAME_ANY_WORDING = True

# Musiclyse dev20 GROUP_LEAD_STAYS_SEVERAL, recomputed for older saves: an
# identity that says one lead while its reasoning documents a vocal group (boy
# band, girl group, brothers, sisters, a duo ...) and names no single lead
# singer means a shared lead (the sibling-band test track: several in dev17, one in dev19).
# (added in 2.2.3; False gives 2.2.2's output)
GROUP_LEAD_STAYS_SEVERAL = True

# Analyser UNBROKEN VOICE CHECK: when the saved identity carries that note (a
# credited boy or teenager the listening model heard as female-sounding, kept
# as a man by the identity model), or a young male identity's reasoning itself
# records a female-sounding or unbroken voice, the voice is described as unbroken, not as
# already deepened or deepening from the audio-only male profile. False
# restores the earlier behaviour.
# (added in 2.2.3; False gives 2.2.2's output)
UNBROKEN_VOICE_IDENTITY_CHECK = True

# 2.4.4: a shared lead says "two singers" when the save shows two (a voice count
# of two, two co-leads, two credited names, a lead and a named co-lead, or two
# leads trading), and "more than one singer" when it doesn't say how many --
# "several" only for three or more. False: 2.4.3's "several singers" wording.
LEAD_COUNT_WORDING = True

# 2.4.10.1: a solo artist whose young or unbroken voice made the listening passes
# disagree on gender -- while every sampled clip heard the same one voice -- keeps
# its settled identity in "Who's singing", with a note on why the passes split,
# instead of the "lead coverage is unresolved" caution. Anything that hints at a
# second lead (credits, features, co-lead or duet readings, a voice count of two
# or more, sampled clips that disagree) keeps the caution exactly as before.
# False: 2.4.10.
SOLO_YOUNG_LEAD = True

# A second-listen voice reason that argues against the settled singer's gender
# is left out (it came out garbled: "consistent with a adult high"); "a adult"
# reads "an adult".
# (added in 2.4.10.2; consistency fixes; False gives 2.4.10.1's output)
VOICE_REASON_CHECK = True

# The 2.4.10.1 solo-voice note says "every clip sampled", not "throughout".
# (added in 2.4.10.2; consistency fixes; False gives 2.4.10.1's output)
SOLO_NOTE_WORDING = True

# An act named "The ...s" given as the singer reads "The ...s' singer".
# (added in 2.4.10.3; False gives 2.4.10.2's output)
BAND_NAME_SINGER = True

# A shared lead whose main singer's voice hadn't broken: "Who's singing" says
# plainly who sings and how (not a paragraph of cautions repeated three
# times); the cautions stay once, in "Easy to get wrong".
# (added in 2.4.10.11; False gives 2.4.10.10's output)
UNBROKEN_COLEAD_PLAIN = True

# (added in 2.4.10.12; False gives 2.4.10.11's output)
ORIGINAL_YEAR_FROM_WIKIPEDIA = True   # reissue year -> the album's original year (Wikipedia lookup on)
AGE_FROM_WIKIPEDIA = True             # the named singer's age at the time (Wikipedia lookup on)

# (added in 2.4.10.12; False gives 2.4.10.11's output)
LIGHTER_COVERAGE_NOTE = True          # one agreed lead, unconfirmed coverage: describe it, one short caution

# (added in 2.4.10.12; False gives 2.4.10.11's output)
YOUNG_VOICE_WORDING = True            # young-male voice check (0.7-pre5): "hadn't fully deepened yet"
SINGER_OVER_ACT_NAME = True           # "the credited artist is Band X, whose lead vocalist is Y": Y sings lead
SINGLE_SINGER_EVIDENCE = True         # a multi-voice answer saying "a single singer" is not a second-lead hint

# (added in 2.4.10.14; False gives 2.4.10.13's output)
ACT_NAME_NEEDS_SOLO_EVIDENCE = True   # "Band sings lead" only with solo evidence; a group: "Band's singer"
SINGER_NOT_FORMER = True              # never name a former/original/founding singer from the older patterns

# (added in 2.4.10.15; False gives 2.4.10.14's output)
SOLO_EVIDENCE_STRICT = True           # only "solo artist/singer..." wording counts; a band-like name ignores the cover

# (added in 2.4.10.16; False gives 2.4.10.15's output)
ADOLESCENT_CHANGING_WORDING = True    # adolescent male: "hadn't fully deepened" unless the save marks it unbroken

# (added in 2.4.10.17; False gives 2.4.10.16's output)
SOLO_RESCUE_SINGLE_CREDIT = True      # young solo singer rescue: a one-person credit counts without the word "solo"
DOCUMENTED_SINGER_COVERAGE = True     # a named, documented adult singer: describe them with the short caution
GROUP_SINGER_NAMES = True             # a shared lead names the singers the identity reasoning lists
AGE_FROM_REASONING_WORDING = True     # the age at the time in "when he was 13" / "was 16 at the time" wording

# Musiclyse dev46 IDENTITY_FLIP_NOTE, for saves made before it: when the first
# vocal pass and the voice descriptions heard a woman, only the confirmation
# pass a man, and the saved identity simply followed that male reading (no
# singer named), the dossier does not state the singer's gender.
# (added in 2.4.10.17; False gives 2.4.10.16's output)
IDENTITY_FLIP_GUARD = True

SOLO_SEX_EVIDENCE = True       # 2.4.10.18: "a documented solo male/female" counts as solo evidence
COVER_TWO_NOT_GROUP = True     # 2.4.10.18: two people on the cover alone don't make the act a group
UNRESOLVED_LEAD_DETAIL = True  # 2.4.10.18: say what the clips heard when lead coverage is unresolved

# (added in 2.4.10.19; False gives 2.4.10.18's output)
UNRESOLVED_MIXED_VOICE = True  # unresolved mixed/uncertain leads: say so, and keep by-ear voice lines from naming one sex

YOUNG_SOLO_MALE_CLIPS_LEAD = True  # 0.7-pre9 saves: a young solo male settled by the scan isn't 'unresolved'

# (added in 2.4.10.21; False gives 2.4.10.20's output)
CLIPS_AGREE_LEAD = True  # whole-track passes conflicted but every confident clip heard the identity's sex: not 'unresolved'

# (added in 2.4.10.22; False gives 2.4.10.21's behaviour; export text is the same either way)
SAFE_COVER_NAME = True     # --copy-cover only copies a cover named as a plain file name next to the save

# (added in 2.4.10.23; False gives 2.4.10.22's output)
ACT_NAME_NOT_SINGER = True  # a band's own name isn't its singer; "X, documented as the lead vocalist" names X

# (added in 0.7.1-pre6; False gives 0.7's output)
EXPORT_IDENTITY_PRE6 = True       # who's singing: ages from birth year, young-voice maturity, guest voices kept apart, low-confidence label

# (added in 0.7.1-pre6; False gives 0.7's output)
IDENTITY_SHARED_LEAD_EVIDENCE = False  # a feature credit or group counts as shared lead vocals only when the identity notes say so
                                  # (off: as in 0.7, so a featured guest or a group's shared leads are not missed)

# (added in 0.7.1-pre6; False gives 0.7's output)
SHARED_LEAD_PLAIN_NAMES = True    # a shared lead names a duo's singers, and a solo artist with a guest as "X and Y", not "X's singer"
CHILD_AGE_WORDING = True          # a deepened voice under 13: "still a boy", not "still a teenager"
DENIED_SINGER_CHECK = True        # when the notes say the credited artist isn't the singer, name the singer they give (or no one)

# (added in 0.7.1-pre7; False gives pre6's output)
SINGLE_SINGER_LEAD_ONLY = True    # "lead only; no backing vocals" counts as one singer, so a young solo lead isn't left "unresolved"

# (added in 0.7.1-pre8; False gives pre7's output)
SHARED_LEAD_GROUP_VOICE = True    # a lead shared by several singers: "the lead singers are men", "the lead line moves...", not "his melody"
MIXED_GROUP_CLIPS = True          # a mixed-gender group whose clips heard both: the lead is shared by its male and female singers
MULTI_LEAD_BAND_CANDIDATES = True # a band with several lead singers, this song's not documented: "one of A or B", not a guess or "shared"
GROUP_LEAD_NAMES = True           # a shared lead names the group's lead singers when the singer identity lists them

# (added in 0.7.1-pre10; exporter update; False gives the earlier pre10 output)
SOLO_YOUNG_MATURITY_WORDING = True  # the "passes split" note says "hasn't broken yet" for an unbroken voice, matching "Who's singing"

# (added in 0.7.1-pre11; False gives pre10's output)
SINGLE_SINGER_NO_SECOND_VOICE = True  # "no distinct second voice is audible" counts as one singer
LIGHT_COVERAGE_CLIP_GUARD = True      # never "found no second lead singer" when 2+ confident clips heard the other sex
GROUP_NOT_ONE_YOUNG_VOICE = True      # a group's mixed clips are not read as one voice that hadn't fully deepened
UNRESOLVED_BOTH_SEXES_DETAIL = True   # a group (cover) with an unresolved lead whose confident clips heard both sexes: say where each was heard

# (added in 0.7.1-pre12; audit fixes; False gives pre11's output)
DOCUMENTED_ADULT_NAME_FORMS = True    # birth years given with a middle name, a birthplace, or the surname alone
SHORT_VOICE_UNBROKEN_NEEDS_OPEN_GENDER = True  # the comparison summary never says "gender unclear" for a settled female (or male) singer
COLEAD_AGE_BOUND = True               # a shared lead's age is taken only from a clause about the main singer, never a guest's
SINGLE_SINGER_STRICT = True           # a voice note mentioning a guest, a rapper, someone joining or alternating isn't "one singer"
IDENTITY_TAG_ALIASES = True           # an older save's "male", "female", "adult_female_likely"... read as the main program reads it
COVER_FIELDS_TOLERANT = True          # a cover PEOPLE value "group", "band" or "a group of five" counts as a group
FEATURE_CREDIT_KEEP_ACT_NAMES = True  # "feat. X and the Y" is one act; a title's "(with X)" credits a guest singer
REASONING_STOPS_AT_KNOWN_FIELDS = True  # a wrapped reasoning line starting "Note:" stays part of the reasoning
SAMPLED_WINDOWS_BOTH_SEXES = False    # the walkthrough also places the main singer's sampled clip windows
                                      # (off: on a library check it repeated the main singer in nearly every section, and placed a young male lead's misheard "female" clips)

FEATURED_SINGER_NOT_LEAD = True       # 0.7.1-pre13: a saved female identity crediting a documented featured singer, on a
                                      # solo credit whose cover shows one young male, is read as the boy's unbroken voice
SINGER_DETAIL_WORDING = True          # 0.7.1-pre13: a scan's SINGER_DETAIL_* lines name a group's lead singer, give ages at
                                      # the time, and stop a mixed group being described as all women or all men
FEATURED_SINGER_ONE_VOICE = True      # 0.7.1-pre15: the featured-singer check also covers a saved shared-lead answer when the
                                      # listening passes heard one female-sounding voice and the clips confirmed no second lead

# 0.8-pre31: FEATURED_SINGER_NOT_LEAD is held back when the identity reasoning gives the featured
# guest a part other than singing the lead, or when the singer lookup's lines or the reasoning say
# the credited artist is a woman. False gives pre30's rule.
FEATURED_SINGER_SAFEGUARDS = True

# 0.8-pre35: a teenage-male identity whose reasoning plainly records an unbroken or female-sounding
# voice is described as unbroken ("hadn't broken yet"), and wording such as "not yet fully deepened
# (unbroken voices)" is no longer read as "not unbroken". False gives pre34's wording.
UNBROKEN_FROM_IDENTITY_REASON = True

# 0.8-pre35: a lead shared by a group's own singers gets plural voice wording ("Their voices hadn't
# broken yet on this recording"), not "His voice". False gives pre34's wording.
SHARED_LEAD_PLURAL_VOICE = True

# Keep accepted medium/high-confidence identity separate from unresolved passage checks.
# Retains coverage cautions and sampled evidence; False restores original exporter behavior.
SAVED_IDENTITY_COVERAGE = True

# Accepted multiple-performer details fill otherwise missing shared-lead names.
DOCUMENTED_SHARED_NAMES = True
# Positive reference corrections supersede old inconclusive coverage flags.
REFERENCE_COVERAGE_PRECEDENCE = True
# Prefer measured track duration over the end of a possibly partial section map.
EXPORT_MEASURED_DURATION = True
# Discard invalid section/chord intervals before calculations and time formatting.
EXPORT_VALID_TIMESTAMPS = True
# Set each new option False to restore its previous exporter behavior.

VOCAL_COVERAGE_CAUTION = True  # qualify unresolved conflicting vocal passes


# ==============================================================================
#  E2. THE VOICE: RANGE AND VOCAL WORDING
# ==============================================================================

# Musiclyse dev20 PITCH_CHECK_ROUNDED_FIGURES: the low-confidence pitch check
# allows for the saved figures being rounded to whole hertz (the synth-rock test track: tracker G4
# from 4% voiced frames against the notes' E4, 2.98 semitones as written).
# (added in 2.2.3; False gives 2.2.2's output)
PITCH_CHECK_ROUNDED_FIGURES = True
PITCH_CHECK_ROUNDING_TOL_ST = 0.1

# 2.2.3: when that check sets the pitch tracker's median aside, a by-ear pitch
# clause quoting it ("range spans from B3 to A4, centered around G4" -- the
# vocal pass was shown the tracker's figures) is left out of "How it sounds",
# so it no longer contradicts "Who's singing".
BY_EAR_PITCH_FOLLOWS_CHECK = True

# 2.4.4: evidence wording -- the voice description is labelled "by ear", the
# guide ranks measured figures above by-ear impressions (studio processing such
# as pitch correction, doubling and delay is never measured here), and the era
# clues say they are general clues, not a description of the track.
# False: 2.4.3's wording.
EVIDENCE_RULES = True

# 2.4.5: when the scan could not confirm a voice ("vocal presence unresolved"),
# the separated vocal stem is called "the vocal-range part" (it may hold a voice
# or an instrument in that range); singing-only claims are left out. False: 2.4.4.
VOCAL_RANGE_WORDING = True

# 2.4.7: with vocal presence unresolved, the model is told to describe the
# vocal-range part in plain words, not by that label. False: 2.4.6.
VOCAL_RANGE_TALK = True

# A one-note range reads "on F♯4", not "between F♯4 and F♯4" / "F♯4–F♯4"; the
# lead-line ranges (10th-90th percentile) read "most notes between ...".
# (added in 2.4.10.2; consistency fixes; False gives 2.4.10.1's output)
RANGE_WORDING = True

# (added in 0.7-pre55; False gives the pre54 exporter's output)
RANGE_LABEL_BOTH_READINGS = True  # the range label agrees with the centre shown; two readings that differ give a span
RANGE_NOT_TIMBRE_NOTE = True      # say range is how high the notes sit, not how deep or light the voice sounds

# 0.7.1-pre20: the "How it sounds, by ear" voice line gets the same gender wording check as the
# other by-ear voice lines ("typical for a female pop vocalist" is not said of a male lead).
# False gives pre19's line.
VOCAL_TIMBRE_GENDER_SCRUB = True

# 0.7.1-pre20: a by-ear voice line that runs away into a list of near-identical items ("vocal
# clarity, vocal presence, vocal emotion, ...": five or more in a row starting with the same word)
# is cut after the first of them. False gives pre19's lines.
VOCAL_RUNAWAY_LIST_CUT = True


# ==============================================================================
#  E3. MELODY, KEY AND NOTES
# ==============================================================================

# Analyser KEY FIT CHECK, for saves made before it: a reported key one scale
# note away from the key the saved notes fit clearly better (by KEY_FIT_MARGIN),
# whose tonic the bass sits on most, is replaced by that key. False restores
# the earlier behaviour.
# (added in 2.2.3; False gives 2.2.2's output)
KEY_FIT_CHECK = True
KEY_FIT_MARGIN = 0.05

# dev45 blocks: the measured chord progression, the melody against the key and
# how melodic the voice is, section by section.
CHORD_PROGRESSION_IN_DOSSIER = True
MELODY_IN_KEY_IN_DOSSIER = True
MELODICISM_IN_DOSSIER = True

# Plain melody and delivery wording at every level: no catch-all "moves up and
# down" contour line, steps/leaps labelled from their own percentages, and no
# "rap" wording from the word-rate measure. False: 2.3.17's wording.
# (added in export dev2 (2.4))
MELODY_WORDING_DEV2 = True

# The phrase-by-phrase reading of the lead line (--level plus, high and full;
# never brief or standard). False: left out at every level.
# (added in export dev2 (2.4))
PHRASE_READING = True

# 0.7-pre67 HIDDEN_BASS_NOTE: heavy whole-track sub-bass but a separated bass part that
# reads as silent for much of the time the drums play -> a caution in "Easy to get
# wrong" that the bass is probably there (separated into another part). False gives
# pre66's export.
HIDDEN_BASS_NOTE = True
HIDDEN_BASS_MIN_SUB = 25.0       # % of the whole-track energy in the 20-60 Hz band
HIDDEN_BASS_MIN_SILENT = 0.40    # share of the drum-playing time with the bass part silent

# 0.7-pre61 PHRASE_WORDS: in the phrase-by-phrase melody notes, the words each listed
# phrase is roughly sung over (from the timed lyric lines), and the "end higher /
# lower" count says it counts a rise or fall of a minor third or more -- before, a
# phrase rising a whole step was listed as rising but not counted. False: pre60's text.
PHRASE_WORDS = True

# --level high writes its note and drum-hit lists compactly (2.4.1): the same
# notes, times, lengths and levels as --level full, about a third smaller,
# grouped by section. False: 2.4.0's run-length layout (no smaller than full).
COMPACT_HIGH_MIDI = True

# 2.4.7: when the bass stem is too quiet to count but a bass guitar was still
# picked out somewhere, the warning says so instead of "no bass guitar".
# False: 2.4.6.
BASS_FAINT_NOTE = True

# 2.4.7: "something string-like (cello)" instead of "a strings (cello)" for a
# bracketed strings label. False: 2.4.6.
STRINGS_LABEL_FIX = True

# 2.4.10: the "where the tune sits in each section" line gives each section's
# range ("Intro between A4 and B4; Section 1 and Section 3 on F♯4"), so one note isn't
# read across every section in the group. False: 2.4.9.
CONTOUR_RANGES = True

# In a settled flat key, note names are spelled with flats (B♭4 in E♭ major, not
# A♯4); raw MIDI lists, lyrics and titles are untouched; sharp keys, unclear keys
# and the leading note of D and G minor keep their sharps.
# (added in 2.4.10.2; consistency fixes; False gives 2.4.10.1's output)
FLAT_KEY_SPELLING = True

# "STEM-ONLY" in by-ear lines reads "heard only in the separated part".
# (added in 2.4.10.2; consistency fixes; False gives 2.4.10.1's output)
STEM_ONLY_WORDING = True

# Flat-key spelling by each note's relation to the key: flats for the key's own
# flats and for ♭2, ♭3, ♭6, ♭7; in minor keys the major 3rd, major 6th, leading
# note and sharp 4th keep their sharps (F♯ and C♯ in D minor, not G♭ and D♭).
# (added in 2.4.10.3; False gives 2.4.10.2's output)
SPELLING_BY_INTERVAL = True

# The "modal colour" line names the note that makes each mode (a flattened
# second for Phrygian, a raised sixth for Dorian, a raised fourth for Lydian,
# a flattened seventh for Mixolydian) instead of "a flattened seventh" for
# every mode, keeps the full scale name ("natural minor"), and uses only
# readings centred on the track's key -- the most common one when there are
# several -- leaving the line out when none are.
# (added in 2.4.10.8; False gives 2.4.10.7's output)
MODE_WORDING = True

# (added in 2.4.10.12; False gives 2.4.10.11's output)
ONEOFF_PLUCKED_CHECK = True           # brief, unconfirmed zither/ukulele/mandolin... tags left out
A_AN_YOU_WORDS = True                 # "a ukulele", "a unison line"
STRINGS_HORNS_PART_NAME = True        # strings/brass through most of the track: "the strings"/"the horns"

# (added in 0.7.1-pre6; False gives 0.7's output)
MIDRANGE_PART_WORDING = True      # "the midrange part" for an unconfirmed guitar-range part; "the piano part" when that part is piano

# 0.7.1-pre28: when HIDDEN_BASS_NOTE's caution applies (the separated bass part reads as silent
# for much of a sub-heavy track, so the bass was probably separated into another part), the
# walkthrough no longer says "no bass yet", "the bass comes in" or "the bass drops out": the
# caution already says those can't be read from the bass part. False gives pre27's walkthrough.
HIDDEN_BASS_WALKTHROUGH = True

# 0.7.1-pre30: when HIDDEN_BASS_NOTE's caution applies, the rest of the export agrees with it: the
# "how loud each part is" table leaves out the bass column (with a line saying why), bass moves are
# left out of the list of parts coming in and dropping out, and the bass part's notes start with a
# line saying they hold only some of the low end. False gives pre29's export.
HIDDEN_BASS_DETAIL = True

# 0.7.1-pre30: the key is marked doubtful (on its Key line, and with a note under "Easy to get
# wrong") when the measured chords plainly don't support it: its own home chord takes less than
# KEY_CHORD_CHECK_HOME_MAX of the chord time and chords outside the key at least
# KEY_CHORD_CHECK_OUTSIDE_MIN. The key itself is not changed. Needs the save's chord segments.
# False gives pre29's key wording.
KEY_CHORD_CHECK = True
KEY_CHORD_CHECK_HOME_MAX = 0.03
KEY_CHORD_CHECK_OUTSIDE_MIN = 0.45


# ==============================================================================
#  E4. TEMPO, DRUMS AND GROOVE
# ==============================================================================

# Analyser TEMPO OCTAVE PAIR CHECK, for saves made before it: detectors an
# octave apart, a listening-model pulse that is neither of them (nor half or
# double) kept as "~2x", and a snare backbeat at the detector reading nearest
# that pulse: the detector reading is used. False restores the earlier behaviour.
# (added in 2.2.3; False gives 2.2.2's output)
TEMPO_OCTAVE_PAIR_CHECK = True
TEMPO_OCTAVE_PAIR_TOL = 0.05

# 0.7-pre61 TEMPO_CHANGE_NOTE: from the save's beat times, whether the tempo moves.
# A steady rise or fall of 5% or more from the first quarter of the song to the last
# is stated with start and end figures, the BPM called an average; stretches of 20 s
# or more running clearly (5%+) off the song's usual beat -- not just counted double
# or half -- are named by time only (the beat tracker may count them at another
# octave, so a figure there could be wrong). Steady songs are unchanged. False: pre60.
TEMPO_CHANGE_NOTE = True
_TC_WIN, _TC_STEP, _TC_CHANGE, _TC_RUN_S, _TC_TREND = 16, 8, 0.05, 20.0, 0.6

# High and full already list every drum hit in their own section; the scan's
# second copy of the same hits ("full hit list (N hits): hihat@0.22, ...") in
# "Detail: further signal measurements" is left out when that section carries
# the same number of hits (2.4.2). False: kept, as in 2.4.1.
DROP_DUPLICATE_HIT_LIST = True

# 2.4.5: the era clue "gated-reverb snares" says what it means, so a snare
# measured as gated or heavily damped is not taken for it. False: 2.4.4.
GATED_REVERB_CLARITY = True

# 2.4.5: "programmed-sounding drums" needs firmer evidence -- the scan's own
# reading of the kick (0.97+ "very repeatable", 0.995+ "a retriggered sample"),
# machine-even hit levels, or an electronic style -- and is never used when the
# hit levels read as human-feeling. A drum-machine tag alone is not enough, and
# when it's overruled it isn't listed as an instrument. False: 2.4.4.
DRUM_MACHINE_EVIDENCE = True

# "Machine-even" hit levels count only when every measured drum is even; one
# drum with wide accenting (hi-hats at 0.88) means the levels aren't machine-even.
# (added in 2.4.10.3; False gives 2.4.10.2's output)
DRUMS_ALL_EVEN = True

# (added in 2.4.10.12; False gives 2.4.10.11's output)
TEMPO_OCTAVE_WORDING = True           # a possibly double-time tempo is worded with both counts

# One drum grid for each part whose groove differs (up to GROOVE_GRIDS_MAX),
# instead of a single grid, in "The groove" (standard and above).
# (added in 2.4.10.17; False gives 2.4.10.16's output)
GROOVE_GRIDS_BY_PART = True
GROOVE_GRIDS_MAX = 4

# Musiclyse dev46 TEMPO_COPY_CHECK, for saves made before it: a listening-model
# tempo that only copied a detector reading is not a second opinion, so a
# half/double-time choice it decided is made again from the detectors
# (a disco test track: 100 BPM, not 200).
# (added in 2.4.10.17; False gives 2.4.10.16's output)
TEMPO_COPY_CHECK = True


# ==============================================================================
#  E5. GENRE, STYLE, ERA AND INSTRUMENTS
# ==============================================================================

# Musiclyse dev20 MOSS_LOW_TRUST_FIELDS, recomputed for older saves: a genre
# decision resting on MOSS-Music's stock genre list ("indie rock; power pop;
# jangle pop", the example its own prompt gives) falls back to the family the
# PANNs tags support ("rock", "pop", with a lean only when other evidence names
# one), and those stock labels and non-genre words ("layered production") are
# not listed as close neighbours.
# (added in 2.2.3; False gives 2.2.2's output)
MOSS_LOW_TRUST_FIELDS = True

# Musiclyse dev20 GENRE_SYNTH_ROCK, recomputed for older saves: the dev56
# "post-punk / new wave" reading of a synth-pop / new wave first impression is
# "synth-rock" (rock with a prominent synth lead) when the separated keys/synth
# part leads the band, and synth-pop / new wave stay fair comparisons instead of
# labels to avoid (the synth-rock test track; the post-punk test track keeps post-punk / new wave).
# (added in 2.2.3; False gives 2.2.2's output)
GENRE_SYNTH_ROCK = True

# 2.3 settings (Musiclyse 0.5-moss-dev42 to dev46). False gives 2.2.3e's output
# for that part.
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

# "Detail: what the tune suggests about mood" (2.4.2), in --level plus, high and
# full only: cues measured from the saved vocal notes and chords, each with how
# it tends to be heard, plus a guide rule on using them alongside the by-ear
# mood and the words. False: left out, and the guide is as in 2.4.1.
MOOD_CUES = True

# 2.4.7: the quiet-master lines say 1980s releases were mastered that way too.
# False: 2.4.6.
ERA_80S_MASTERS = True

# 2.4.9: a by-ear "wide stereo" era clue next to a measured normal or narrow
# image gets the measured reading beside it. False: 2.4.8.
STEREO_BY_EAR_CHECK = True

# A by-ear clue of a gated-reverb snare or sidechain pumping that the
# measurements don't bear out gets the measured reading right beside it.
# (added in 2.4.10.3; False gives 2.4.10.2's output)
BY_EAR_MEASURED_CHECK = True

# Musiclyse 0.6.2's second instrument tags back up or question the rest of the
# save in three cases only: no keyboard at all in a guitar-led track, synths
# through practically the whole track, a strong piano through most of it.
# (added in 2.4.10.5; False gives 2.4.10.4's output)
JAMENDO_GUIDE = True

# "Synths run through practically the whole track" only when the separated
# keys/synth part carries synth for at least half the track as well.
# (added in 2.4.10.6; False gives 2.4.10.5's output)
JAMENDO_SYNTH_STEM = True

# (added in 0.7.1-pre6; False gives 0.7's output)
STYLE_LEAN_KEPT = True            # a measured style that refines the genre keeps the earlier "(with X lean)"
ERA_GENERIC_CLUE_CAVEAT = True    # a 2010s/2020s by-ear era from general clean-production clues, on an older release, says they don't pin the decade


# ==============================================================================
#  E6. SECTIONS AND SONG STRUCTURE
# ==============================================================================

# Leave out a by-ear lyric topic of "none (instrumental)" when the track has a
# voice or lyrics.
# (added in export dev2 (2.4))
LYRIC_SUBJECT_CHECK = True

# 2.4.5: digital silence is described by where it is -- before the music starts,
# a gap, or after the song ends (it was always "the song is over"). False: 2.4.4.
SILENCE_POSITION = True

# "It opens with ..." leaves out parts whose own entry comes later in that same
# section (drums that start at 0:21, a part that "comes in" at 0:20).
# (added in 2.4.10.2; consistency fixes; False gives 2.4.10.1's output)
OPENING_ENTRIES = True

# "The beat in each part" uses the song map's section times (it copied the
# scan's own, rounded differently: 0:06 next to 0:07).
# (added in 2.4.10.2; consistency fixes; False gives 2.4.10.1's output)
SECTION_TIMES_MATCH = True

# A lyric-sheet heading that lands inside another section reads "where the lyric
# sheet's chorus starts", not "— the chorus" inside Verse 1.
# (added in 2.4.10.2; consistency fixes; False gives 2.4.10.1's output)
LYRIC_HEAD_WORDING = True

# Every section, not just the first: a part listed as coming in, whose own
# timed entry comes later in that section, is left to that timed entry.
# (added in 2.4.10.3; False gives 2.4.10.2's output)
SECTION_ENTRIES = True

# Laying out the sections or listing several timestamps, the chat says once,
# briefly and in its own voice, that section names and timings can be a
# little off.
# (added in 2.4.10.9; False gives 2.4.10.8's output)
SECTION_TIMING_ASIDE = True

# 0.7.1-pre30: in "When each line is sung", a line that starts and ends within the same second
# shows one time ("1:19: together") instead of "1:19–1:19". False gives pre29's times.
LYRIC_TIME_SINGLE = True

# 0.7.1-pre20: in tag lyrics whose section headers sit directly above their lines, a header names
# only its own stanza (up to the next blank line), and a header standing alone ("[Chorus]" with a
# blank line after it, meaning that part comes again) names nothing after it. Lyrics with a blank
# line between each header and its lines are read as before. False gives pre19's reading, where a
# header named every line after it up to the next header.
LYRIC_HEADER_OWN_BLOCK = True

# 0.7.1-pre20: a section's "The words" no longer starts with the last line of the stanza the
# section before was singing, when that line is sung right at the section's start straight after
# the rest of that stanza (the tail of the previous part carried over the boundary); it starts
# with the next line. Only with tag lyrics.
# False gives pre19's walkthrough.
SECTION_WORDS_SKIP_CARRYOVER = True

# 0.7.1-pre20: a lyric line from a sparse transcription chunk (1.25 s or more per word) that runs
# on past the section's end, into a next section of the part the line's header names, is not
# reported as that part arriving early ("... arrives around 0:13 -- where the lyric sheet's chorus
# starts" when the chorus section starts at 0:20): the line was timed from where its chunk began,
# which can be well before it is sung.
# False gives pre19's walkthrough.
LYRIC_HEAD_NEXT_SECTION_CHECK = True


# ==============================================================================
#  E7. THE CHAT GUIDE AND ITS WORDING
# ==============================================================================

# Guide rules: descriptions anchored to this track instead of stock phrases,
# the melody and fast-delivery rules, and plausible user corrections accepted.
# False: 2.3.17's guide text.
GUIDE_DEV2 = True

# 2.4.5: a listening-model answer that only repeats the field's own instructions
# ("brief subject/theme/tone based only on ...") or is empty ("none") is left
# out, like other non-answers. False: 2.4.4.
TEMPLATE_ECHO_CHECK = True

# 2.4.5: a correction is judged by how noticeable the part would be: a new name
# for a part the notes have, or something a scan can miss, is taken on board; a
# prominent part (a solo, a lead instrument, a drop...) that nothing in the notes
# could be gets one polite push-back, and each correction is weighed on its own.
# False: 2.4.4's correction bullets.
CORRECTION_PROMINENCE = True

# 0.7.5-pre1: "listener-confirmed" covers who sings and background facts the notes
# don't settle; a correction about what the recording contains is weighed like any
# other (taken on board when plausible, one polite push-back when it can't be what
# was heard or clashes with a firm measurement). False: the earlier wording, under
# which any explicit factual correction counted as confirmed.
CORRECTION_SCOPE_NARROW = True

# 2.4.6: a looser guide: description tied to this song where it comes naturally
# (not in every sentence), the melody rule for when the melody comes up, one hedge
# per by-ear point, and overlapping bullets merged. Every guardrail is kept.
# False: 2.4.5's guide wording.
FLUENT_GUIDE = True

# 2.4.6: custom characters react to a song with the taste the character would
# plausibly have (dislike and indifference included); the ordinary assistant is
# unaffected. False: 2.4.5's persona rule.
PERSONA_TASTE = True

# 2.4.7: a prominent part the person says is there (a solo, a lead instrument...)
# is answered as something you'd have heard -- with no time given, across the
# whole song -- never "plausible"; "didn't catch it" is kept for smaller details.
# False: 2.4.6.
PROMINENT_CLAIMS = True

# 2.4.7: talk as someone who has heard the song -- part names and headings are
# shorthand, never repeated -- and an unprompted assistant asked how it listened
# says plainly that an analysis tool did the listening. False: 2.4.6.
LISTENER_VOICE = True

# 2.4.8: the listener rule says technical detail is welcome in a listener's voice
# (tempo, key, chords, note names, loudness, mix) and bans only "reading about
# it" framing -- the 2.4.7 wording "never mention notes" could be read as
# musical notes. False: 2.4.7's wording.
TECHNICAL_TALK = True

# 2.4.9: the listener rule spells out plain wording for each part name
# ("piano-range part" -> a piano or another keyboard...). False: 2.4.8.
PART_NAME_PLAIN = True

# 2.4.9: the closing reminder adds one line -- plain words, no stock phrases, the
# sound not described as typical of the artist. False: 2.4.8.
FOOTER_REMINDERS = True

# 2.4.9: an ordinary custom character who wouldn't know the artist doesn't know
# the title, artist, album, year or the band's nationality either -- only what
# it could hear (a title guessed from a lyric, the language of words it caught).
# Experts and authority figures, and the default assistant, are unaffected.
# False: 2.4.8.
PERSONA_TITLE_KNOWLEDGE = True

# 2.4.10: generic praise words ("driving", "lush", "soaring") are left out of the
# by-ear lines, and the mood cue for busy notes no longer says "driving"; the
# .json keeps the original wording. False: 2.4.9.
STOCK_WORDS_BY_EAR = True

# Sentences in the listening model's by-ear descriptions start with a capital.
# (added in 2.4.10.2; consistency fixes; False gives 2.4.10.1's output)
BY_EAR_CAPITALS = True

# An ordinary custom character that recognises an artist still knows only as
# much about them as someone like it plausibly would -- not the whole
# encyclopedia entry. Experts, authority figures and the default assistant
# are unaffected.
# (added in 2.4.10.7; False gives 2.4.10.6's output)
PERSONA_KNOWLEDGE_DEPTH = True

# The same reminder in one line just above the first background article.
# (added in 2.4.10.7; False gives 2.4.10.6's output)
PERSONA_WIKI_REMINDER = True

# Wording: "How it unfolds" says a repeated groove, tune shape or instrument
# list once instead of in the same words for every part, and the guide asks for
# fresh wording rather than the notes' phrases and stock music-review lines.
# (added in 2.4.10.17; False gives 2.4.10.16's output)
VARIED_WORDING = True

# Guide: the sound is described from what was heard only; what is known about
# the artist, album or year is casual context beside it, never a description of
# the sound.
# (added in 2.4.10.17; False gives 2.4.10.16's output)
SOUND_BEFORE_KNOWLEDGE = True

# (added in 0.7-pre53; False gives the pre52 exporter's output)
LISTENING_TIDY = True             # split "VOICE=...; STANDS_OUT=..." leaks; drop "none"/"nothing in particular" cues
LISTENING_UNCHECKED_LABEL = True  # say the listening details are unchecked impressions that rank below measurements

# (added in 0.7-pre57; False gives the pre56 exporter's behaviour)
WIKI_OFF_BY_DEFAULT = True        # Wikipedia background only with --wiki (--no-wiki still accepted)

# 0.7.1-pre28: the chat model is asked to check a correction against the notes before taking it on
# board, and to say so once, politely, when a measurement plainly says otherwise -- a part shown
# playing or silent right there, or a section start well away from the one given -- as it already
# does for a tempo or key. The user's ear still has the last word. False gives pre27's guide.
CORRECTION_CHECK_MEASURED = True

# 0.8-pre31: the guide's note about laying out sections asks for one mention, in the chat model's
# own words and where it fits, that the times and the section names can both be off (a part a
# second or two from its time, a name only a guess at the form), not a stock opener such as
# "just a heads-up". Needs SECTION_TIMING_ASIDE. False gives pre30's note.
GUIDE_SECTION_ASIDE_FRESH = True

# 0.8-pre31: corrections from the user: a correction that plainly contradicts something these notes
# measure directly (where a melody line starts, ends or peaks among them) gets one friendly mention of
# what the notes show and why the two might differ, without refusing; plausible corrections are
# taken on board in the chat model's own words, not a set phrase. False gives pre30's guide.
GUIDE_CORRECTIONS_BALANCED = True


# ==============================================================================
#  E8. READING SAVES AND WRITING EXPORTS
# ==============================================================================

# A save field of the wrong type (hand-edited or damaged) is ignored instead of
# stopping the export; real Musiclyse saves are never touched.
# (added in 2.4.10.2; consistency fixes; False gives 2.4.10.1's output)
ROBUST_FIELDS = True

# (added in 2.4.10.22; False gives 2.4.10.21's behaviour; export text is the same either way)
UNREADABLE_INPUT_FAILS = True  # an unreadable save (or a named non-save) makes the run exit with code 1

# (added in 0.7-pre57; False gives the pre56 exporter's behaviour)
DEFAULT_INPUT_SAVED_SONGS = True  # no input named: export every save in saved-songs (beside this script)

# (added in 0.7.1-pre6; False gives 0.7's output)
EXPORT_INPUT_PATH_FIXES = True    # folders with [ ] in their names, and plain file paths, are read correctly

# 0.7.1-pre19: a save's record of the singer lookup (the main script's SINGER_DETAIL_DIAGNOSTICS)
# is left out of exports, so the chat never sees it. False lists it under "About the file" like
# any other field this exporter doesn't recognise.
EXPORT_SKIP_LOOKUP_RECORD = True

# 0.7.1-pre27: the size printed for each finished export ("~N tokens") and the context length
# advised after them count tokens the way the chat model's tokenizer splits text, where every
# digit is a token of its own: --level high and full exports, full of timestamps and notes, come
# out at two to three times the old figure. Never less than the old figure. --max-tokens keeps
# the old count. False gives pre26's printout.
EXPORT_REALISTIC_TOKEN_COUNT = True

# 0.8-pre31: a music video's record of who it shows singing (the main script's VIDEO_PERFORMERS) is
# left out of exports. False lists it under "About the file" like any other field this exporter
# doesn't recognise.
EXPORT_SKIP_VIDEO_PERFORMERS = True

# 0.8-pre31: a file of your own changed settings, beside the scripts, read on top of this file (see
# PERSONAL_SETTINGS_FILE in the main script's part). "" (or no such file): only this file is read.
PERSONAL_SETTINGS_FILE = "musiclyse-my-settings.py"


# ==============================================================================
#  E9. OTHER SETTINGS
# ==============================================================================

# 0.7.1-pre28: --combine with only one song to write (the same save named twice, or a single
# input) writes an ordinary single-song export under the --combine name, instead of "# 1 songs,
# side by side" with a one-column comparison table. False gives pre27's file.
COMBINE_ONE_SONG_AS_SINGLE = True

# 0.7.1-pre28: a name ending in "s" takes just an apostrophe ("Band Names' track", not "Band
# Names's track"). False gives pre27's wording.
POSSESSIVE_S_APOSTROPHE = True

# 0.7.1-pre30: the listening model's instrument list ("First pass by ear") when the model stopped
# mid-item or repeated itself: an unfinished last item keeps only what was finished ("voice (3.9"
# becomes "voice"; an instrument already listed is not repeated), an item repeated word for word is
# listed once, and a run of one instrument numbered in brackets ("synthesizer (source 12)") is
# written once without the numbers. A complete list is unchanged. False gives pre29's list.
BY_EAR_CUT_ITEMS = True
