# Musiclyse
A highly configurable piece of local AI chatbot software that can "listen" to music via a multi-step analysis engine, compare tracks and more.

![Musiclyse - Local Music Discussion](https://gourlish.altervista.org/musiclyse/musiclyselogo.svg)

![The Free Music Analyser](https://gourlish.altervista.org/musiclyse/slogan.svg)

![Chat About Your Songs](https://gourlish.altervista.org/musiclyse/chat.svg)

![100 Percent Local](https://gourlish.altervista.org/musiclyse/local.svg)

![Musiclyse - Screenshot](https://gourlish.altervista.org/musiclyse/Musiclyse04.png) 

# About:

Musiclyse is a Python script designed for high-spec Macs (it is easily portable to good quality Linux PCs) that enables local conversations about music. It utilises a multi-step song processing pipeline linked to an Ollama hosted local LLM of the user's choice. This consists of a pipeline process of Gemma 4 (in 0.7 and later), MOSS-Music, Essentia/PANNs, Discogs-400 genre detection to aid PANNs, Demucs separation fed to Omnizart, Wikipedia database RAG retrieval for artist context, metadata analysis (which will improve recognition for lyrics), several other smaller models, and a whole load of algorithmic code to glue these together so they work optimally. You're welcome to help improve the project to make its song recognition better!

# Key features:

The following features are all included in Musiclyse 0.7:

* Beat, melody, genre, vocal, song key, song structure, scale and era analysis. Accuracy for many things has been significantly improved in 0.7.
* Talk about the music theory behind a song.
* Can get accurate facts about music adjacent topics via an optional Wikipedia-based database add-on file.
* Will talk about the concept of the songs or compare and contrast two or more different songs. You can ask it anything you can ask a normal chatbot, and it will be able to bring songs into context with the chat.
* Supports image analysis (if used with an image-supporting model).
* LLMs can be swapped out easily in the script. Currently it is optimised for use with Gemma 4 31B; for less powerful systems Gemma 4 26B is recommended.
* Save your analysed song data to JSON files, then load them quickly in a new session.
* Run an overnight batch of songs to JSON files that can be fast loaded at a later time.
* Create your own personas and hear their opinions about songs, as well as testing their knowledge.
* Import two tracks together to compare them easily via both musical and sonic signature analysis.
* Specify a range (x seconds to y seconds) in a track to analyse for faster processing and more precise analysis.
* Export your .json files to specially formatted .txt files, allowing any general purpose LLM to "listen" to music and have highly contextual discussions about one or more songs. You can do this via musiclyse_export.py.
* Seamlessly analyse music videos

## Version history:

* 2026/11/09 - Musiclyse 0.7. Completely overhauled the built in chatbot functionality. It no longer accepts raw .JSON files and now instead relies on .txt export files, which contain the same data in LLM-prompted form, for much better efficiency and fluidity. It should feel a lot less stiff or "drunk" than it did before, and these files contain all of the same descriptive data about tracks. The new default chatbot is really fun and witty; you'll love it. You can now save and load conversations. Unsaved conversation text remains in RAM only, as was the case previously. Now uses Gemma 4 31B as the output LLM. Conversations should feel much more fluid than they did with Muse Glimmer, which sounded sort of drunk. You can now enter /load "" or '' and it will give you a dropdown of tracks you've analysed and exported to select from. Commands have been updated to reflect the overhauled chatbot. Further refined vocalist identity detection. Added a game mode - right now there are 4 games (they're custom LLM prompts that serve specific functions). 3 are different censo from historic contexts (the Brezhnev-era USSR, the US during McCarthyism and the explicit lyric panic of the 1980s) that you can test, while one is a tool designed to tell you if a song would likely be suitable for a pop hit today. Vastly strengthened the singer identity algorithm (it still relies heavily for ambiguous cases on metadata and the wiki database, though). Added MTG-Jamendo model for additional instrument detection details. MOSS-Music now gives additional passes to allow for additional instrument playing style or production details on some tracks. Fixed a number of small bugs. Personas now take a few seconds to be accepted (or declined). I introduced this safety check due to stumbling upon an unintended problem where /persona mode in the rewritten chat was bypassing Gemma's usual safety guardrails on who it would roleplay as. Typing /load ""/'' will now bring down a dropdown menu to make it easier for you to select your song(s). Fixed a problem where certain genres listings were defaulting to scripted examples; specific genre detection should now be better. Exporter scripts now contain more information about the production details of tracks via the listening_details section of .json analysis files. Audio from video files can now be analysed without the need to convert first. Musiclyse now supports video analysis alongside talking about the music of a video! "Thinking..." progress notices in the chat now disappear when the output begins generating. Unneccessary LLM-written changelogs have been removed from the public version, making the file sizes smaller. If you would like a copy with LLM notes for development purposes (under the terms of the MIT license), please contact me at rjdgourley@outlook.com and I'll send them over to you. Integrated the functionality of the export script inside the main script. Added SongFormer for improved section labelling. Timestamps and section labels should now be noticeably more accurate on many tracks. Made BPM fixes that were causing double speed detection on some tracks. Added a license disclaimer on the startup screen. Added a /help menu. You no longer need to start Ollama manually before launching the script - it will open, if a local Ollama server is not already running, when you launch Musiclyse.
* 2026/10/30 - Musiclyse 0.6.2. Brings major improvements to the export script, allowing for more in-depth and accurate conversations about tracks with general purpose LLMs. Fixes a number of minor bugs in the main script, but changes little in the song analysis pipeline.
* 2026/10/28 - Musiclyse 0.6.1. A patch that makes improvements to vocal identity recognition. If you already have 0.6 installed, you can simply copy the 0.6.1 main script and export files to the same folder and run as before without any reinstall needed. Some tracks where there is conflicting evidence may take a bit longer to resolve the vocalist identity.
* 2026/10/27 - Musiclyse 0.6. Now uses Python 3.11 (due to the end of Python 3.10 support). Replaced Music Flamingo with MOSS-Music as the listening model, which scores higher in benchmarks and is less prone to erratic hallucinations (being built on Qwen 3 8B instead of Qwen 2.5 7B). Mood and style detection should now be noticeably more detailed. This does require a different installation process to before. Noticeable improvements to timestamp and song section detection accuracy, as well as song key and BPM detection accuracy. Refined the .json-to-txt export script to take into account the updates and improvements made. Much improved genre classification. Fixed a number of memory caching and disk swap bugs that were causing high memory pressure on Macs with 64GB RAM. You should notice that songs process noticeably faster than in 0.5 (which brought down times noticeably from 0.4). Refinements to vocal identity detection. Implemented a system attempting to allow output LLMs to discuss music theory in a song based on the extracted MIDI and time signature information, notably things like chord progressions. Added a new install script that should make the software easier to setup.
* 2026/10/15 - Musiclyse 0.5. Makes various improvements from 0.4.
* 2026/09/03 - Musiclyse 0.4. Implemented Whisper transcription for lyrics, replacing Music Flamingo for this task (you will need to download a new model). Added CLAP embeddings to further help determine song similarity in comparisons (this also requires a new model). Further refined instrument and melody recognition capabilities. Improved song structure recognition with more accurate timestamps. Further refined vocalist recognition capabilities. Musiclyse can now talk about where instruments are in a track and output drum grooves. Implemented a flag in the script for full MIDI analysis (at the expense of using more tokens). Fixed an issue that was truncating the middle of the system prompts, reducing output accuracy (if your token context window is set too low, you will now see a warning about prompt truncation on startup). Fixed an issue that was causing all tracks to be treated as mono. Implemented mode detection for analysed songs. Improved custom persona functionality so that custom personas are much less likely to know about things they would not be aware of, while still being able to talk about topics they are aware of. Improved era detection for songs without metadata. Reduced the likelyhood of a couple of other Music Flamingo/Qwen 2.5 7B hallucination bugs. Wikipedia database lookup logic has been noticeably improved as a result of debugging (it uses the same database as 0.3), and now produces relevant results for both songs when comparing tracks. Fixed problems with guitar being wrongly identified as not present, and organ incorrectly identified. Fixed filename import and .json extension requirement issues with /loadcompare. Fixed Ollama caching problem to make long conversations faster (this may cause old messages to disappear from the context suddenly rather than "fade" away. Fixed a bug that caused the script to crash when loading a model without vision support, or when loading an invalid model. Fixed a problem whereby the script would crash on song or image import where running with a non-vision capable output LLM. Fixed a problem whereby the script would crash if a non-valid output model was defined.
* 2026/08/28 - Musiclyse 0.3. Larger Wikipedia database (optional). Proper database search implementation, meaning much less lookup latency. Vastly strengthened lyric transcription hallucinations in Music Flamingo's Qwen 2.5 7B (though may transcribe fewer lyrics). Improved song import context reasoning. Improved track instrument recognition. Improved multi-voice recognition. Improved genre recognition. Improved BPM recognition. Fixed issue with phantom instruments being detected on subsequent tracks without them where they were present in the first track of a conversation. Musiclyse can now talk about the loudness of music in LUFS as well as dynamics crest factor. Improved vocal age range classification logic. Fixed problem that was only analysing vocal range for first 30 seconds. Prevented the Wikipedia and native LLM logic bases from using their knowledge of an artist to influence their description of the style of an analysed song, while still being able to talk about the artist. Files can now be loaded and saved with spaces and non-standard characters. Batch saves now use an "Artist Name - Song Name" naming structure. Wikipedia context is now cleared on /clear command. Added /clearall command to also purge cached songs. You can now /listen and analyse a specified second range of a track for increased speed and accuracy. The batch processing crash issues on Macs with 64GB RAM have now been properly fixed. Fixed a problem where the LLM did not recognise other musicians that the model was capable of recognising from images mid-conversation. IMPORTANT NOTE - Due to how prone Music Flamingo was to hallucinating on lyric transcriptions, I have implemented strict transcription guardrails that now commonly result in it not outputting lyrics at all, so make sure your files have lyric metadata to allow the LLM to talk about lyrics on a track! I intend to implement an alternative transcription model on a future release.
* 2026/08/24 - Musiclyse 0.2. Adds Wikipedia database integration for much more reliable chats about artists, albums and songs. Omnizart processing logs are now hidden by default. Improved batch processing optimisations to make crashes on Macs with 64GB RAM less likely. Improved BPM and instrument recognition (though there may still be some minor accuracy issues). Overall a much better chat experience from the added bank of data. Is now set to use muse-glimmer:30b-mlx by default, which significantly improves LLM processing times on MacOS. If using Linux (or Windows if you can get it working), change to the non-MLX version or a different version.
* 2026/08/22 - Musiclyse 0.1. Beat, melody, genre, vocal, song key, scale and era analysis. There are currently some minor accuracy issues which I am hoping to improve. Will talk about the concept of the songs or compare and contrast two or more different songs. You can ask it anything you can ask a normal chatbot, and it will be able to bring songs into context with the chat. Supports image analysis (if used with an image-supporting model). LLMs can be swapped out easily in the script. Currently it is optimised for use with Muse Glimmer 30B; for less powerful systems Gemma 4 26B is recommended. Save your analysed song data to JSON files, then load them quickly in a new session. Run an overnight batch of songs to JSON files that can be fast loaded at a later time. Create your own personas and hear their opinions about songs.





## System requirements

Due to the nature of what it's doing, it's intended to be run on powerful machines with a good quality GPU and 64GB RAM or higher. Currently it's only been tested on MacOS, but it should be pretty easy to get running on Linux, and is potentially runnable on Windows. For users with less powerful machines, I'd advise swapping Gemma 4 31B for Gemma 4 26B A4B, which has reasonable knowledge but is not as resource heavy. Users with lower spec machines might benefit from installing a 4-8B model via Ollama. You can swap out the Ollama model by altering the "OLLAMA_MODEL" value with the exact name of your model.

# Manual Installation for 0.7 (for MacOS):

1: Ensure Python 3.11 is installed on your system

2: Clone the provided files onto your machine for the latest version, or alternatively download a version from the Releases section.

3: [Download the heavy version of the Musiclyse Wikipedia Database via this link](https://github.com/Gourlish/Musiclyse-Wikipedia-Database/releases/tag/music_wiki_heavy-august2026). Alternatively you can [download a lite version](https://github.com/Gourlish/Musiclyse-Wikipedia-Database/releases/tag/music_wiki_lite-august2026) that covers fewer topics but consumes less storage (it will run without either, but it's likely to give far better context about artists if you install it). Once extracted (the heavy database should be listed as 8.33GB on MacOS, or a bit less on Windows and many distributions of Linux) place the music_wiki_heavy.db file in the same directory as the Musiclyse Python script. If you are using the Lite version of the database, you will need to search the script for mentions of "music_wiki_heavy.db" and replace this with "music_wiki_lite.db".

3: Install required system level dependencies (if you don't already have Homebrew installed you'll need it too)

```
brew install ollama ffmpeg portaudio
```

4: In a new terminal window, start Ollama and install Gemma 4 31B (if you want a different model for output, change the specification in musiclyse-0.7.py via the "OLLAMA_MODEL =" flag to the exact Ollama name of the model of your choice, and pull that different model with Ollama). By default the model is configured to use Ollama's default version of Gemma 4 31B.

```
ollama serve
ollama pull gemma4:31b
```

5: Return to your PREVIOUS prompt, leaving the Ollama one open (once the model has installed and the server is running) since the script will not run without an Ollama server running, and create a virtual Python 3.11 environment in the folder with the git clone you made:
```
python3.11 -m venv ./venv
source venv/bin/activate
```

6: Run the main installer, which installs requirements.txt and some of the models used:

```
bash install_musiclyse.sh
```

7: Once the main installer has finished, download the Omnizart models:

```
omnizart download-checkpoints
```

9: Run the software:

```
python musiclyse-0.7 .py
```

# Configurable Variables & Flags in Musiclyse 0.7

## Command-line

| Usage | Description |
|---|---|
| `python musiclyse-0.7.py` | Start the chat. Analyse tracks with `/analyse` or `/batch`, export them with `/export` and load the exports with `/load`. |
| `python musiclyse-0.7.py "saved-songs/llm-ready/plus-wiki/Artist - Title.txt"` | Start with an export already loaded. An audio file given here is not analysed: a note says to use `/analyse`. |
| `python musiclyse-0.7.py --batch-one /path/to/song.mp3` | Internal: isolated single-track batch worker used by `/batch`. |
| `python musiclyse-0.7.py --batch-one-analyze /path/to/song.mp3 /tmp/staging.json` | Internal: heavy-analysis stage for split-process batch mode. |
| `python musiclyse-0.7.py --batch-one-finish /tmp/staging.json` | Internal: identity/save stage for split-process batch mode. |
| `python musiclyse-0.7.py --omnizart-prefetch /tmp/input.json /tmp/output.json` | Internal: separate-process Omnizart transcription worker; not a normal startup option. |
| `python musiclyse-0.7-export.py [files or folder] [options]` | The exporter: turns saved `.json` analyses into `.txt` exports for chatting. With no files or folder it exports every save in `saved-songs/`. `/export` runs it from inside the chat with the same options (see **Export options**). |

## In-chat slash commands

| Command | Description |
|---|---|
| `/analyse <audio file>` | Analyse one track (audio or video file) and save it to `saved-songs/`; skipped if it is already saved. |
| `/reanalyse <audio file>` | Analyse a track again, replacing its save. |
| `/batch <folder>` | Analyse every audio file in a folder into `saved-songs/`; skips files already saved. |
| `/export [files or folder] [options]` | Export saves for chatting, exactly as the exporter script does. No path exports every save in `saved-songs/`; empty quotes (`/export ""`) choose songs from a menu; `/export --help` lists the options. |
| `/load "Artist - Title" [question]` | Load an export (or a combined export) to chat about; the question is optional. `/load ""` opens a menu of your exports. Exports are found under `saved-songs/llm-ready/` (any folder inside it) or by path. |
| `/load "Artist 1 - Title 1" "Artist 2 - Title 2" … [question]` | Load several exports at once, e.g. to compare them. |
| `/quiet` | Add to a `/load` to load without a reply. |
| `/lookup` | Add to any prompt (also with `/load`) to look up background in the local Wikipedia database. |
| `/lookup-search "Title" ["Title 2" …] [question]` | Add Wikipedia articles by title. When a phrase is in several titles, a menu opens on the input line and the chosen title goes back on the line. Also works with `/load`, `/videoload` and `/musicvideoload`. |
| `/chatsave [name]` | Save the conversation to `chats/`. Without a name it updates the conversation's current save (or makes a timestamped one); a name makes a separate copy. Saves include songs, images and the full transcript. |
| `/chatload [name]` | Resume a saved conversation. Alone (or with empty quotes) it opens a menu; a name finds the save made with that name (a menu when several match); `.json` is optional. |
| `/videoanalyse "<video>" [options]` | Visual analysis of a video's pictures; asks for an optional name and saves resumable reports in `video-analysis/`. `/videoanalyse --help` lists the options. |
| `/videoload <name or path> [question]` | Load a video report to chat about; empty quotes open a menu. Can be combined with `/load` and a question. |
| `/frame <loaded video name> <index>` | Attach a saved frame (indices start at 0) to a follow-up question. |
| `/videoframe <report.json> <time> [question]` | Show the retained frame nearest a video time. |
| `/videoembed <report.json>` | Embed an older report's saved JPEGs without rescanning. |
| `/videostrip <report>` | Write a text-only copy of a report (no frame images) for sharing. |
| `/musicvideoanalyse "<video>" [options]` | Analyse a music video's song (audio), then its pictures; asks for an optional name. |
| `/musicvideoload <name> [question]` | Load a music video's song export and its video report together; empty quotes open a menu. |
| `/persona <description>` | Chat as a custom character. The description is checked by the default assistant first (it takes a few seconds), and a description that points to a picture waits for a reference image. |
| `/persona reset` | Return to the default assistant. |
| `/think` | Add to any prompt to let the chat model think before answering (slower). |
| `/game` | Play a game: chat with a built-in character chosen from a menu. |
| `/game-exit` | Leave the game and return to the default chat. |
| `/clear` | Clear the conversation and reset token counters; loaded exports and looked-up background are kept and sent again. |
| `/clearall` | Clear everything, including loaded exports and looked-up background. Files on disk are untouched. |
| `/debug` | Show the last message sent to the chat model. |
| `/help` | On its own: show the list of commands again. |
| Images | Drag an image file into the window (or type its path or web address) in any prompt to show it to the chat. |
| `quit` | Exit the application (`exit` also works). |

Video commands are unavailable in games. Names with spaces can be quoted.

## Export options (exporter script and `/export`)

| Option | Description |
|---|---|
| `[files, folders or patterns]` | `.json` saves, folders or glob patterns; default: every save in the `saved-songs` folder beside the script. |
| `--level {brief,standard,plus,high,full}` | Detail depth: brief < standard < plus (standard + the tune phrase by phrase) < high (adds every note, written compactly) < full (the same notes as full timed event lists). Default `standard`. |
| `-o DIR`, `--out-dir DIR` | Output folder (default: `llm-ready` next to each input). Exports go in a `{level}-wiki` or `{level}-no-wiki` subfolder inside it. |
| `--max-tokens N` | Trim the lowest-priority sections until the file fits (approximate). |
| `--ext {txt,md}` | Output extension; the content is markdown either way. |
| `--combine FILE` | Write all inputs into one file with a side-by-side comparison. |
| `--no-lyrics` | Leave lyrics out. |
| `--max-lyrics-chars N` | Limit the length of the lyrics included. |
| `--raw-notes` | Also list every transcribed note and drum hit (large; for exact-timing questions). |
| `--cover` | Include the cover-art description section (omitted by default). |
| `--copy-cover` | Copy the saved cover image next to each output (for vision models). |
| `--stdout` | Print instead of writing files. |
| `--wiki` / `--no-wiki` | Add Wikipedia background articles, or don't (the default is `--no-wiki`). |
| `--wiki-db PATH` | The Wikipedia database (default: `music_wiki_heavy.db` next to the script). |
| `--wiki-scope {full,core}` | `full`: artist, discography, every album and single found, members (default); `core`: artist, this track's album and song, members. |
| `--wiki-debug` | Also print every member name found and why each was or wasn't added. |
| `--wiki-artist NAME` | Artist to look up when a file has no artist tag (applies to every input). |
| `--persona-mode {auto,default}` | `auto` adds a conditional custom-persona knowledge rule (default); `default` uses the original guide and footer without that rule. |
| `--version` | Show the exporter's version. |

## Video options (`/videoanalyse` and `/musicvideoanalyse`)

| Option | Description |
|---|---|
| `--name NAME` | Display and analysis folder name (otherwise asked for). |
| `--song "Artist - Title"` | The song save to pair the video with. |
| `--offset SECONDS` | Video time minus song time, to align the two. |
| `--fps N` | Frames sampled per second (e.g. `--fps 1`). |
| `--frames-per-request N` | Frames sent to the model per request (e.g. 8). |

Further options (`--model`, `--endpoint`, `--max-side`, `--scene-threshold`, `--max-frames`, `--num-ctx`, `--reply-tokens`, `--keep-alive`, `--output-root`) are listed by `/videoanalyse --help`. Similar song and video names are associated automatically; timing alignment still needs an explicit offset.

## Console / display

| Variable | Default | Description |
|---|---:|---|
| `FRIENDLY_PROCESSING_DISPLAY` | `True` | Use a compact, progress-bar-style display instead of a scrolling log. |
| `USE_COLOR` | Automatic | Enables terminal colour when `sys.stdout.isatty()` is true. This is detected by the script, not a normal user-config constant. |
| `DEBUG_MODEL_LOADING` | `False` | Show raw model-loading logs instead of a status line. |
| `SHOW_OMNIZART_LOGS` | `False` | Allow Omnizart stdout/stderr through instead of suppressing it. |
| `DEBUG_STEM_MIDI` | `False` | Print verbose per-stem note-count debugging. |

## Ollama / model settings

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_URL` | `"http://localhost:11434/api/chat"` | Ollama chat endpoint. |
| `OLLAMA_MODEL` | `"gemma4:31b"` | Chat model (Ollama). Pull this model in Ollama, or change the name to another installed model (for example `"gemma4:26b"`). |
| `OLLAMA_NUM_CTX` | `131072` | Context-window size requested from Ollama. |
| `OLLAMA_SUPPORTS_IMAGES` | `True` | Whether the selected writer model is treated as accepting image input. |
| `OLLAMA_INTERACTIVE_KEEP_ALIVE` | `"30m"` | How long Ollama keeps the interactive writer model loaded after a request. |
| `MOSS_MODEL_ID` | `"OpenMOSS-Team/MOSS-Music-8B-Instruct"` | Hugging Face model ID for MOSS-Music, the audio-listening model. |
| `MOSS_TORCH_DTYPE` | `torch.bfloat16` | PyTorch data type for MOSS-Music. |
| `WHISPER_MODEL_ID` | `"openai/whisper-large-v3"` | Whisper model used for lyric transcription/timing. |
| `WHISPER_TORCH_DTYPE` | `torch.float16` | PyTorch data type for Whisper. |
| `DEMUCS_MODEL` | `"htdemucs_6s"` | Demucs source-separation model. |
| `DEMUCS_SHIFTS_FAST` | `0` | Demucs test-time shifts in fast mode (`0` disables shift ensembling). |
| `DEMUCS_SHIFTS_DEEP` | `2` | Demucs test-time shifts in deep mode. More shifts can improve separation at additional cost. |
| `MOSS_EMPTY_RETRY_SYSTEM_PROMPT` | `True` | Retry MOSS-Music's main pass with the full instructions as a system prompt if the first answer is empty. Adds a generation only for affected tracks. |
| `MOSS_EMPTY_RETRY_ADD_DESCRIPTION` | `True` | Add a description pass if that recovery answer lacks enough descriptive fields. Adds a generation only for affected tracks. |
| `MOSS_LOOP_GUARD` | `True` | Stop runaway repetitive lines in MOSS-Music's main report so other fields can be produced. |
| `UNLOAD_OMNIZART_AFTER_STEM_MIDI` | `True` | Unload Omnizart after stem/MIDI transcription to reduce memory use. |
| `OMNIZART_MUSIC_MODEL_BY_STEM` | `{"bass": None, "guitar": None, "piano": None, "other": None}` | Optional Omnizart mode/checkpoint override per stem. `None` keeps the default model; only use modes/checkpoints installed in your Omnizart setup. |

## Analysis pipeline toggles (speed, depth and report content)

| Variable | Default | Description |
|---|---:|---|
| `FAST_MODE` | `False` | Skip the MOSS-Music self-check pass to save time and one generation. |
| `DEEP_MODE` | `False` | Use larger token budgets and two extra analysis categories; more thorough but slower. |
| `PARALLEL_ANALYSIS` | `True` | Run independent analysis tasks concurrently where supported. Background failures are retried in order. |
| `ENABLE_OBJECTIVE_AUDIO_REPORT` | `True` | Include signal-processing-based objective audio measurements. |
| `COMPACT_OBJECTIVE_REPORT` | `True` | Use a compact objective-report format. |
| `SKIP_LYRICS_TRANSCRIPTION_WHEN_TAGS_PRESENT` | `True` | Skip lyric transcription when usable embedded lyrics are available; lyric timing can still require Whisper. |
| `ENABLE_VOCAL_PASS` | `True` | Run the dedicated vocal-analysis pass. |
| `ENABLE_VOCAL_OBJECTIVE_REPORT` | `True` | Include objective vocal measurements. |
| `ENABLE_VOCAL_CONFIRMATION_PASS` | `True` | Run the additional vocal classification/confirmation pass when needed. |
| `ENABLE_VOCAL_STEM_ISOLATION` | `True` | Separate the vocal stem for pitch tracking even when stem/MIDI transcription is disabled. |
| `ENABLE_INDEPENDENT_F0_MULTI_SINGER_TRIGGER` | `True` | Use independent pitch evidence to help trigger multi-singer/duet checks. |
| `INDEPENDENT_MULTI_SINGER_MIN_SEP_SEMITONES` | `7.5` | Minimum pitch-centre separation used by the independent multi-singer trigger. |
| `INDEPENDENT_MULTI_SINGER_MIN_RATIO` | `0.25` | Minimum share/ratio of usable pitch evidence required by that trigger. |
| `ENABLE_ESSENTIA_REPORT` | `True` | Include Essentia-based audio features when Essentia is installed. |
| `ENABLE_STEM_MIDI` | `True` | Run stem separation and per-stem MIDI/note analysis. |
| `FULL_MIDI_TRANSCRIPTION` | `True` | Use the full MIDI-transcription path rather than the lighter summary-only path. |
| `STEM_MIDI_INCLUDE_EVENT_LOGS` | `False` | Include detailed stem/MIDI event logs. |
| `STEM_MIDI_COMPACT_EVENT_FORMAT` | `True` | Use a compact format for stem/MIDI event output. |
| `STRUCTURE_V2_ENABLED` | `True` | Use the newer bar-level, multi-cue song-structure segmentation. Set `False` to fall back to the previous structure pipeline. |
| `STRUCTURE_ANSWERS_WITH_TIMES` | `True` | Include measured times in structure answers. |
| `WHISPER_WORD_TIMING` | `True` | Use Whisper word timing to help align lyrics with the section map. |
| `WHISPER_EXPLICIT_TEMPERATURE` | `False` | Explicitly set Whisper's anti-hallucination temperature settings. Enabling it can change lyric text and timing; the default leaves it off for compatibility. |
| `WHISPER_TIMING_RETRY` | `True` | Retry word timing when the first timing result is too weak. |
| `MOSS_CONSISTENT_REPORT` | `False` | Ask the listening model for a consistent field-based report. This can change the evidence supplied by the model; normally leave off unless testing. |
| `REPORT_FIXED_LAYOUT` | `True` | Standardise the saved analysis fields and their order when writing JSON files. |
| `REPORT_FILL_FROM_EVIDENCE` | `True` | Fill otherwise-empty report fields from suitable evidence already present in the analysis. |
| `REPORT_FIXED_BLOCKS` | `True` | Keep key report blocks present and in a consistent location in saved JSON. |
| `VOCAL_QUICK_PASS` | `True` | Run the short vocal-description pass. |
| `VOCAL_NOTE_PITCH_LINE` | `True` | Include the vocal stem's note-by-note pitch line as a cross-check. |
| `VOCAL_PITCH_OCTAVE_CHECK` | `True` | Check for a pitch tracker reading that is an octave above the note-by-note reading. |
| `VOCAL_UNBROKEN_SIGNATURE` | `True` | Preserve evidence that an adolescent male voice may be unbroken when supported by the vocal analysis. |
| `VOCAL_DEEPENED_SIGNATURE` | `True` | Distinguish a young male artist whose voice has already deepened. |
| `INSTRUMENT_REED_TAG_CHECK` | `True` | Flag unsupported reed-instrument tags when other evidence suggests a synth or sax-like sound. |
| `ENABLE_CLAP_EMBEDDING` | `True` | Compute and save CLAP audio embeddings (sonic-similarity data) with each analysis. Requires compatible Transformers CLAP classes. |
| `PRODUCTION_ANALYSIS_ENABLED` | `True` | Run the dedicated production-era analysis. |
| `ENABLE_COVER_ART_DESCRIPTION` | `True` | Describe embedded cover art when available. |
| `ENABLE_SINGER_IDENTITY_RESOLUTION` | `True` | Resolve the singer's likely identity from audio and available context. |
| `ENABLE_WIKI_CONTEXT` | `True` | Allow `/lookup` and `/lookup-search` to add background from the local Wikipedia database when it is available. |

## CLAP sonic-similarity embeddings

| Variable | Default | Description |
|---|---:|---|
| `CLAP_MODEL_ID` | `"laion/larger_clap_music_and_speech"` | CLAP model used to produce audio embeddings. |
| `CLAP_EMBEDDING_DIMENSION` | `512` | Expected embedding size. Normally leave unchanged for the configured model. |
| `CLAP_SAMPLE_RATE` | `48000` | Audio sample rate used for embedding extraction. |
| `CLAP_WINDOW_SECONDS` | `10.0` | Length of each audio window. |
| `CLAP_WINDOW_HOP_SECONDS` | `5.0` | Hop between embedding windows. |
| `CLAP_MAX_WINDOWS` | `24` | Maximum windows embedded per track. |
| `CLAP_USE_MPS` | `False` | Use Apple's MPS device for save-time embedding work. Off by default to avoid competing with Ollama for unified memory. |
| `CLAP_EMBEDDING_PRECISION` | `8` | Decimal places stored for embedding values in JSON. The vectors remain normalised. |

## Instrument tagging thresholds (PANNs / AudioSet)

| Variable | Default | Description |
|---|---:|---|
| `ENABLE_INSTRUMENT_TAGGING` | `True` | Enable per-stem instrument tagging; requires `panns-inference`. |
| `INSTRUMENT_TAG_MIN_PROB` | `0.18` | Minimum probability for most instrument labels. |
| `INSTRUMENT_TAG_TOP_K` | `4` | Maximum number of instrument tags kept per stem. |
| `INSTRUMENT_TAG_GUITAR_MIN_PROB` | `0.35` | Higher probability threshold for guitar-family labels, which are noisier. |
| `INSTRUMENT_TAG_STEMS` | `("other", "guitar", "piano", "bass", "drums")` | Stems that receive instrument tags. |
| `INSTRUMENT_TAG_REQUIRE_STEM_ACTIVITY` | `True` | Suppress tags on near-empty stems unless the evidence is strong. |
| `INSTRUMENT_TAG_MIN_NOTES_FOR_WEAK` | `8` | Minimum note count before weaker tags can be accepted. |
| `INSTRUMENT_TAG_STRONG_PROB` | `0.45` | Probability above which stem-activity checks are relaxed. |
| `INSTRUMENT_TAG_WINDOW_SECONDS` | `10.0` | Tagging window length; `None` means the whole clip. |
| `INSTRUMENT_TAG_HOP_SECONDS` | `5.0` | Hop between tagging windows. |
| `INSTRUMENT_TAG_MAX_WINDOWS` | `56` | Maximum windows sampled per clip. |
| `INSTRUMENT_TAG_NORMALIZE` | `True` | Peak-normalise each window before tagging. |
| `INSTRUMENT_TAG_NORMALIZE_PEAK` | `0.95` | Target peak after normalisation. |
| `INSTRUMENT_TAG_EVEN_COVERAGE` | `True` | Encourage coverage across a track rather than concentrating samples in one region. |
| `INSTRUMENT_TAG_MIN_WINDOW_HITS` | `2` | Minimum windows supporting a tag under the normal coverage rule. |
| `INSTRUMENT_TAG_SINGLE_WINDOW_STRONG_PROB` | `0.50` | Strong probability needed to accept a tag supported by only one window. |
| `INSTRUMENT_TAG_SUSTAINED_COVERAGE` | `0.25` | Coverage level used by the sustained-instrument rule. |
| `INSTRUMENT_TAG_SUSTAINED_PROB_RELIEF` | `0.85` | Probability relief factor for sustained tags. |
| `ENABLE_WHOLE_MIX_INSTRUMENT_TAGGING` | `True` | Tag the original full mix independently of Demucs. |
| `WHOLE_MIX_INSTRUMENT_TAG_TOP_K` | `6` | Maximum whole-mix instrument tags to retain. |
| `WHOLE_MIX_INSTRUMENT_TAG_MIN_PROB` | `0.15` | Minimum probability for whole-mix tags. |
| `INSTRUMENT_TAG_MIX_AGREE_MIN_PROB` | `0.18` | Minimum whole-mix probability used to support a stem tag. |
| `INSTRUMENT_SECTION_PRESENCE_ENABLED` | `True` | Detect instrument presence by song section. |
| `INSTRUMENT_SECTION_MIN_PROB` | `0.30` | Minimum probability for section-level instrument presence. |
| `INSTRUMENT_SECTION_MAX_LABELS` | `6` | Maximum labels reported for a section. |
| `ARRANGEMENT_CHANGE_ENABLED` | `True` | Detect arrangement changes over time. |
| `BEAT_GRID_ENABLED` | `True` | Include beat-grid evidence used by rhythm and arrangement analysis. |
| `CYMBAL_ONSET_DETECT_ENABLED` | `True` | Enable the dedicated cymbal-onset detector. |
| `DRUM_EXTRAS_PROBE_ENABLED` | `True` | Enable additional checks for drum details beyond the main instrument tags. |

## Genre / mood tagging (PANNs)

| Variable | Default | Description |
|---|---:|---|
| `ENABLE_GENRE_MOOD_TAGGING` | `True` | Use PANNs as an independent genre/mood cross-check. Requires `panns-inference`. |
| `GENRE_TAG_TOP_K` | `5` | Number of genre tags considered. |
| `GENRE_TAG_MIN_PROB` | `0.05` | Minimum probability for genre tags. |
| `MOOD_TAG_TOP_K` | `3` | Number of mood tags considered. |
| `MOOD_TAG_MIN_PROB` | `0.08` | Minimum probability for mood tags. |

## Essentia / audio-analysis limits

| Variable | Default | Description |
|---|---:|---|
| `ESSENTIA_MAX_SECONDS` | `900.0` | Maximum track duration used for the main Essentia report. |
| `ESSENTIA_LOWLEVEL_MAX_SECONDS` | `60.0` | Maximum duration for low-level frame-based features. |
| `ESSENTIA_FRAME_SIZE` | `4096` | Analysis frame size. |
| `ESSENTIA_HOP_SIZE` | `2048` | Analysis hop size. |

## Stem / MIDI limits

| Variable | Default | Description |
|---|---:|---|
| `STEM_MIDI_MAX_SECONDS` | `None` | Maximum duration for stem/MIDI work. `None` means analyse the full track; set a number to cap work for speed. |
| `STEM_MIDI_EVENT_LOG_MAX_NOTES` | `24` | Maximum notes included in an event-log summary. |
| `STEM_MIDI_MELODY_LINE_NOTES` | `24` | Maximum notes in a melody-line summary. |
| `STEM_MIDI_DRUM_PATTERN_HITS` | `24` | Maximum hits in a drum-pattern summary. |

## Vocal-classification thresholds

| Variable | Default | Description |
|---|---:|---|
| `VOCAL_CONFIRMATION_F0_THRESHOLD` | `210.0` | Fundamental-frequency threshold (Hz) used by vocal confirmation. |
| `VOCAL_CONFIRMATION_WITHOUT_F0` | `True` | Allow confirmation without a reliable F0 estimate. |
| `F0_MALE_HIGH_CONFIRM_HZ` | `260.0` | Male/adolescent lead median at or above this frequency can trigger confirmation. |
| `F0_POST_PUBERTY_SOFT_CAP_HZ` | `350.0` | Median at or above this value weakens a post-puberty-male assignment based on pitch alone. |
| `F0_POST_PUBERTY_HARD_CAP_HZ` | `350.0` | High-median cap for post-puberty/adolescent male labels. |
| `F0_MALE_ANY_HARD_CAP_HZ` | `350.0` | High-median cap for any male-tagged lead, including adolescent labels. |
| `F0_FEMALE_RECOVERY_MIN_HZ` | `320.0` | Minimum median pitch for the high-pitch male-tag recovery check to consider strong female confirmation. |
| `F0_CHILD_SOFT_FLOOR_HZ` | `180.0` | A child-category assignment below this median should not be retained on pitch evidence alone. |

## Metadata / cover art / images

| Variable | Default | Description |
|---|---:|---|
| `ENABLE_FILE_METADATA` | `True` | Read supported file metadata/tags. |
| `METADATA_MAX_LYRICS_CHARS` | `10000` | Maximum lyric characters accepted from file metadata. |
| `LYRICS_MAX_CHARS` | `12000` | General maximum lyric text length. |
| `COVER_IMAGE_MAX_DIMENSION` | `512` | Maximum cover-art dimension in the prepared image. |
| `COVER_IMAGE_QUALITY` | `"2"` | MJPEG qscale for cover-image conversion; lower values mean better quality. |
| `MAX_COVER_BYTES_TO_SEND_RAW` | `1_000_000` | Maximum cover-image byte size sent without further conversion. |
| `MAX_EXPLICIT_IMAGE_BYTES` | `2_000_000` | Byte limit used when preparing explicit images supplied in chat. |
| `EXPLICIT_IMAGE_MAX_DIMENSION` | `1568` | Maximum dimension for explicit images; larger/unsupported images are converted rather than rejected. |
| `COVER_ART_DESCRIPTION_NUM_CTX` | `8192` | Context size for cover-art description requests. |
| `SINGER_IDENTITY_NUM_CTX` | `8192` | Context size for singer-identity requests. |

## Chat history / context budget

| Variable | Default | Description |
|---|---:|---|
| `MAX_WRITER_IMAGES_PER_TURN` | `8` | Maximum images actually sent to Ollama in one request. |
| `IMAGE_RETAIN_LAST_N_MESSAGES` | `8` | Retain image data for this many recent history messages. |
| `MAX_WRITER_HISTORY_MESSAGES` | `80` | Maximum writer-history messages retained. |
| `MAX_MESSAGE_CHARS_FOR_OLLAMA` | `120_000` | Maximum message length sent to Ollama. |
| `HISTORY_CHAR_BUDGET_FACTOR` | `3.2` | Conservative characters-per-token estimate for trimming history. |

## Wikipedia context

| Variable | Default | Description |
|---|---:|---|
| `WIKI_DB_PATH` | `"music_wiki_heavy.db"` | Path to the local Wikipedia database. |
| `WIKI_DB_TIMEOUT_S` | `5.0` | Database timeout. |
| `WIKI_CONTEXT_MAX_CHARS_PER_ARTICLE` | `1800` | Maximum characters taken from each article. |
| `WIKI_CONTEXT_TOTAL_MAX_CHARS` | `5000` | Maximum total Wikipedia context block size. |
| `WIKI_MAX_ARTICLES` | `6` | Maximum articles included in a context block. |
| `WIKI_SEARCH_ROW_LIMIT` | `24` | Full-text-search result window. |
| `WIKI_EXACT_TITLE_LOOKUP` | `True` | Try exact-title lookup in addition to body search. |
| `WIKI_EXACT_TITLE_MAX_PER_REQUEST` | `8` | Maximum exact-title lookups per request. |
| `WIKI_EXACT_TITLE_MAX_HITS` | `2` | Maximum hits retained per exact-title query. |

## Production-era analysis

| Variable | Default | Description |
|---|---:|---|
| `PRODUCTION_EXCERPT_COUNT` | `8` | Number of audio excerpts used for this analysis. |
| `PRODUCTION_EXCERPT_SECONDS` | `6.0` | Duration of each excerpt in seconds. |
| `PRODUCTION_ENVELOPE_SR` | `22050` | Sample rate used by the production-envelope analysis. |

## Modal (key / scale) detection

| Variable | Default | Description |
|---|---:|---|
| `MODAL_DETECTION_MIN_NOTES` | `24` | Minimum note count before mode detection is trusted. |
| `MODAL_DETECTION_MIN_SHARE` | `0.04` | Minimum share of note weight for the deciding pitch-class pair. |
| `MODAL_DETECTION_MIN_RATIO` | `4.0` | Required ratio between a characteristic scale degree and its rival. |

## Batch mode / memory management

| Variable | Default | Description |
|---|---:|---|
| `BATCH_PAUSE_BETWEEN_TRACKS_S` | `20.0` | Pause between tracks during `/batch` to let memory settle. |
| `BATCH_RETRY_CRASHED_ANALYSIS` | `True` | Retry a track once in a fresh process if its analysis process crashes. |
| `BATCH_TRACK_TIMEOUT_S` | `None` | Optional time limit in seconds for each subprocess stage of a `/batch` track. `None` means no limit. With split-process batch mode, the limit applies separately to analysis and identity/save. A timed-out stage is stopped and the track is reported as failed. Applies when `BATCH_ISOLATE_PER_TRACK` is enabled; allow extra time for long tracks and first-run model downloads. |
| `BATCH_ISOLATE_PER_TRACK` | `True` | Run each track in a fresh Python subprocess; recommended on macOS. |
| `BATCH_SPLIT_IDENTITY_PROCESS` | `True` | Separate heavy analysis from identity/save work to reduce out-of-memory risk. |
| `BATCH_LOG_RSS` | `True` | Log approximate process resident memory after cleanup. |
| `BATCH_UNLOAD_OLLAMA` | `True` | Unload the Ollama writer model during heavy batch-analysis stages. |
| `BATCH_OLLAMA_KEEP_ALIVE` | `0` | `keep_alive` for short batch helper requests; `0` unloads immediately. |
| `BATCH_PRE_IDENTITY_SETTLE_S` | `5.0` | Wait before the identity/save stage to let memory settle. |
| `SINGER_IDENTITY_REUSE_BATCH_MODEL` | `False` | Experimental batch option to retain the writer model briefly between singer-identity requests instead of unloading after each request. Disabled by default; the default retains the established request-by-request unloading behaviour. |

## Saved files

| Variable | Default | Description |
|---|---|---|
| `SAVE_DIR` | `./saved-songs` | Directory for saved track JSON files and associated assets. |


## MOSS-Music loading and generation

| Variable | Default | Description |
|---|---|---|
| `MOSS_MODEL_REVISION` | `None` | Optional Hugging Face model revision or commit hash; `None` uses the latest available revision. |
| `MOSS_MUSIC_CODE_DIR` | `None` | Local MOSS-Music repository folder containing `src/modeling_moss_music.py`. `None` checks the same-named environment variable, then `MOSS-Music` beside the script, then `~/MOSS-Music`. |
| `MOSS_ENABLE_TIME_MARKERS` | `True` | Enable elapsed-time markers in the listening model’s audio input for timestamped output. |
| `MOSS_MAX_AUDIO_SECONDS` | `600` | Maximum seconds sent to MOSS-Music. This does not impose the same limit on Whisper, Demucs or signal processing. |
| `MOSS_SHOW_PROGRESS` | `True` | Show listening-model generation progress and token speed. |
| `MOSS_DECODING` | `"sampled"` | Decoding mode: `"sampled"` or `"greedy"`. Sampling with a fixed seed is the default to reduce repetition. |
| `MOSS_TEMPERATURE` | `0.7` | Sampling temperature. |
| `MOSS_TOP_P` | `0.8` | Nucleus-sampling probability cutoff. |
| `MOSS_TOP_K` | `20` | Top-k sampling limit. |
| `MOSS_SEED` | `1234` | Sampling seed; change it for a different repeatable draw on the same machine. |
| `MOSS_GUIDED_FIELDS` | `True` | Have the script supply field names while MOSS-Music generates their values. |
| `MOSS_NO_FIELD_COPYING` | `True` | Prevent long answers being copied between fields in the main report and self-check. |
| `MOSS_FIELD_MAX_TOKENS` | `300` | Maximum generated tokens per field, also bounded by the remaining reply budget. |
| `MOSS_SHORT_PROMPTS` | `True` | Use shorter prompts for era, vocal and vocal-confirmation passes. |
| `MOSS_RETRY_ON_NON_ANSWER` | `True` | Retry a listening-model response that does not answer the requested analysis. |


## Whisper timing and lyric handling

| Variable | Default | Description |
|---|---|---|
| `WHISPER_PROGRESS` | `True` | Show decoding-pass progress and memory readouts. |
| `WHISPER_WORD_TIMING_SMEAR_S` | `20.0` | Line duration in seconds used to detect collapsed timing. |
| `WHISPER_WORD_TIMING_DENSE_WPS` | `10.0` | Words-per-second threshold for implausibly dense timing; `0` disables this check. |
| `WHISPER_WORD_LINE_GAP_S` | `0.6` | Pause length in seconds used to regroup timed words into lines. |
| `WHISPER_WORD_LINE_MAX_S` | `8.0` | Maximum duration of a regrouped word-timed line. |
| `WHISPER_WORD_STRIDE_DEDUPE` | `True` | Remove duplicated words from overlapping Whisper windows using their timestamps. |
| `WHISPER_TIMING_RETRY_BELOW` | `0.35` | Lyric-match score below which timing recovery can be attempted. |
| `WHISPER_TIMING_RETRY_GAIN` | `0.10` | Minimum score improvement required to accept recovered timing. |
| `WHISPER_TIMING_MAX_CHUNK_S` | `20.0` | Maximum chunk duration used by the timing-quality check. |
| `TAG_LYRICS_SANITY` | `True` | Reject very short or title/artist-only lyric tags so transcription can run instead. |


## Memory, parallel processing and inference caches

| Variable | Default | Description |
|---|---|---|
| `MOSS_MPS_CACHE_TRIM_EVERY` | `256` | Release unused MPS cache every this many generated MOSS tokens; `0` disables periodic trimming. |
| `WHISPER_MPS_CACHE_TRIM_EVERY` | `256` | Release unused MPS cache every this many generated Whisper tokens; `0` disables periodic trimming. |
| `WHISPER_RELEASE_BEFORE_LOAD` | `True` | Run full cleanup before loading Whisper and after each pass. |
| `RELEASE_MOSS_LOCALS_BEFORE_UNLOAD` | `True` | Drop local references to the listening model before its unload cleanup. |
| `MOSS_RELEASE_REFRESH_LOCALS` | `True` | Refresh the analysis function’s local-variable snapshot so it does not retain the unloaded model on affected Python versions. |
| `GPU_CLEANUP_MAIN_THREAD_ONLY` | `True` | Restrict MPS cleanup to the main thread to avoid concurrent GPU cleanup races. |
| `OMNIZART_PREFETCH_PROCESS` | `True` | Run background Omnizart transcription in a separate process. |
| `OMNIZART_PREFETCH_TIMEOUT_S` | `1800` | Timeout in seconds for the Omnizart helper process. |
| `PANNS_BATCH_WINDOWS` | `8` | Maximum instrument-tagging windows processed together in one inference batch. |
| `PANNS_WINDOW_CACHE_SIZE` | `12` | Number of window-probability results retained in the PANNs cache. |
| `PANNS_TAG_STATS_CACHE_SIZE` | `32` | Number of PANNs tag-statistics results cached. |
| `PANNS_QUIET_WINDOW_REL_DB` | `35.0` | Exclude tagging windows this far below the loudest window, in dB. |
| `PANNS_QUIET_WINDOW_ABS_RMS` | `1e-4` | Absolute RMS floor for tagging windows. |


## Discogs styles and genre / era checks

| Variable | Default | Description |
|---|---|---|
| `ENABLE_DISCOGS_STYLES` | `True` | Add MAEST Discogs style classification. The script points to `python musiclyse_style_setup.py` for model setup; missing model assets cause this stage to be skipped. |
| `DISCOGS_STYLE_MODEL_ID` | `"mtg-upf/discogs-maest-30s-pw-73e-ts"` | MAEST model used for Discogs style classification. |
| `DISCOGS_STYLE_WINDOWS` | `5` | Number of 30-second windows sampled across the track and averaged. |
| `DISCOGS_STYLE_TOP_K` | `5` | Maximum styles displayed in the style block. |
| `DISCOGS_STYLE_MIN_PROB` | `0.10` | Minimum probability for displayed styles. |
| `DISCOGS_STYLE_SAVE_TOP` | `40` | Number of strongest style scores retained in saved JSON. |
| `GENRE_CHECK_IN_SAVED_ANALYSIS` | `True` | Include the genre-check warning at the top of saved analysis text. |


## Chords, melody, key and tempo checks

| Variable | Default | Description |
|---|---|---|
| `ENABLE_CHORD_PROGRESSION` | `True` | Run optional `lv-chordia` full-mix chord recognition and build per-section progressions; skipped if unavailable. |
| `CHORD_PROGRESSION_DICTIONARY` | `"submission"` | Chord vocabulary: `"submission"`, `"ismir2017"` or `"full"`. |
| `CHORDS_MEASURED_LINE` | `True` | Combine measured stem chord lists with the listening model’s chords in a consistent report line. |
| `HARMONY_MELODY_BY_SECTION` | `True` | Describe measured chords, lead melody and their relationship for each section. |
| `MELODY_IN_KEY_BY_SECTION` | `True` | Describe each section’s lead line using scale degrees, phrase endings and recurring figures. |
| `MELODY_IN_KEY_MIN_FIT` | `0.75` | Minimum share of held-note time fitting the key before a part is described in scale degrees. |
| `MELODICISM_MEASURED` | `True` | Derive melodicism descriptions from measured vocal range, held notes and timed word rate. |
| `KEY_MODE_FROM_ESSENTIA` | `True` | Use Essentia’s mode when the listening model gives the same root without a valid mode. |
| `KEY_CENTRE_CHECK` | `True` | Check whether note evidence supports a Mixolydian centre on the reported major key’s fifth. |
| `KEY_FIT_CHECK` | `True` | Check a nearby key against transcribed note fit and bass tonic evidence. |
| `KEY_NOTE_CHECK` | `True` | Check conflicting listening-model and Essentia keys against transcribed pitched notes. |
| `KEY_SCALE_CHECK` | `True` | Use distinguishing scale notes to check neighbouring-key assignments. |
| `KEY_CHANGE_CHECK` | `True` | Detect supported late key changes and report the starting key plus modulation. |
| `FULL_BAND_TEMPO_CHECK` | `True` | Re-measure suitably slow, high-frequency-rich tracks at full bandwidth to recover a missed faster pulse. |
| `FULL_BAND_TEMPO_SLOW_BPM` | `80.0` | Both detectors must read below this BPM for the full-bandwidth check. |
| `FULL_BAND_TEMPO_MIN_HF_SHARE` | `0.02` | Minimum high-frequency energy share for the full-bandwidth tempo correction. |
| `TEMPO_LISTENING_CHECK` | `True` | Reject unhelpful model tempo evidence and use measured beat-spacing consistency to resolve conflicts. |
| `TEMPO_SNARE_TIEBREAK` | `True` | Use a measured snare backbeat to help settle half/double-time ambiguity. |
| `TEMPO_OCTAVE_PAIR_CHECK` | `True` | Check detector octave pairs against the listening-model estimate and snare evidence. |
| `TEMPO_MEASURED_OCTAVE_PAIR` | `True` | Let two octave-related measured tempos decide when the listening-model estimate matches neither. |
| `TEMPO_COPY_CHECK` | `True` | Do not count the listening model’s tempo as independent corroboration when it exactly copies a detector reading shown in its prompt. |


## Additional vocal and instrument evidence controls

| Variable | Default | Description |
|---|---|---|
| `VOCAL_FREE_TEXT_GENDER` | `True` | Use agreeing free-form vocal descriptions when structured vocal passes do not settle the profile. |
| `VOCAL_PROBABILITY_VETO` | `True` | Keep the profile open when structured vocal probabilities strongly contradict free-form descriptions. |
| `IDENTITY_FROM_ARTIST_WHEN_OPEN` | `True` | Allow documented solo-artist identity to resolve an otherwise open age/gender category. |
| `VOCAL_DEEPENED_MAX_MEDIAN_HZ` | `300.0` | Median-pitch threshold used in the young-male deepened-voice check. |
| `YOUNG_VOICE_CONFIRM_FLIP` | `True` | Preserve an unbroken-voice reading in the specified young-male, high-pitch confirmation-conflict case. |
| `VOCAL_PITCH_OCTAVE_TOL_ST` | `1.5` | Tolerance in semitones around a one-octave disagreement between pitch readings. |
| `SECTION_VOICE_CHECK` | `True` | Check candidate stretches for a second lead gender using pitch evidence and short listening-model passes. |
| `SECTION_VOICE_MIN_SHIFT_ST` | `4.0` | Minimum pitch shift in semitones for candidate regions. |
| `SECTION_VOICE_MIN_REGION_S` | `12.0` | Minimum duration in seconds of a candidate region. |
| `SECTION_VOICE_SINGLE_REGION_MIN_S` | `24.0` | Minimum duration for a single region to support the stronger one-region decision. |
| `SECTION_VOICE_CLIP_S` | `15.0` | Length in seconds of listening-model clips for region checks. |
| `SECTION_VOICE_MAX_REGIONS` | `3` | Maximum candidate regions checked. |
| `TAG_FEATURE_SHARED_LEAD` | `True` | Use featured-singer credits to correct an identity result that says there is only one lead. |
| `GROUP_LEAD_STAYS_SEVERAL` | `True` | Keep several leads when identity reasoning describes a vocal group without identifying one lead. |
| `UNBROKEN_VOICE_IDENTITY_CHECK` | `True` | Check young-male identity against release-age evidence and an unbroken/female-sounding audio reading. |
| `SANITIZE_VOCAL_CATEGORY_LANGUAGE` | `True` | Clean up internal vocal-category wording in user-facing language. |
| `GUITAR_PROBE_ENABLED` | `True` | Enable the dedicated guitar-presence probe. |
| `ACOUSTIC_GUITAR_CHECK` | `True` | Check unsupported electric-guitar claims against consistently acoustic tagging evidence. |
| `INSTRUMENT_REED_STRONG_LEAD` | `True` | Preserve strongly supported reed-like leads and distinguish likely chordal synth-pad tags. |
| `REED_STRONG_LEAD_MIN_PCT` | `75` | Minimum tag percentage for the strong reed-lead check. |
| `REED_STRONG_LEAD_MIN_WINDOW_SHARE` | `0.75` | Minimum share of windows supporting a strong reed lead. |
| `REED_PAD_MIN_POLYPHONY` | `4.0` | Minimum polyphony used to identify a likely chordal pad behind a rejected reed tag. |
| `INSTRUMENTATION_COLLAPSE_SPANS` | `True` | Collapse repeated timestamped instrument names into one entry per instrument. |
| `STEM_NOTE_MIN_ABS_RMS` | `0.001` | Reject notes below this RMS in their stem as likely separation residue. |
| `BEAT_GRID_BARS` | `2` | Number of bars shown in a beat grid. |
| `BEAT_GRID_SUBDIV` | `4` | Cells per beat; `4` gives sixteenth-note resolution. |
| `BEAT_GRID_MIN_HITS` | `6` | Minimum hits for beat-grid reporting. |


## Chat evidence and saved-track behaviour

| Variable | Default | Description |
|---|---|---|
| `WRITER_CASUAL_VOICE` | `True` | Encourage plain conversational wording and remove unnecessary source-citing sentence openers. |
| `STRUCTURE_SAVE_REPLAY` | `True` | Save inputs used to replay the structure map’s final stage. |

## Additional vocal-presence and co-lead controls

| Variable | Default | Description |
|---|---:|---|
| `SECTION_VOICE_CONFLICT_RECOVERY` | `True` | Enable recovery checks when the whole-track vocal passes disagree on lead gender. |
| `SECTION_VOICE_COVERAGE_CHECK` | `True` | Sample full-mix clips across the track to check for additional leads, including when the whole-track passes agree. Reliable pitch tracking is not required for this coverage path. |
| `SECTION_VOICE_RECOVERY_MAX_CLIPS` | `12` | Maximum initial clips selected by the coverage/recovery sampler. Gap checks, corroboration and direct comparisons can add further requests. |
| `SECTION_VOICE_RECOVERY_GAP_CLIPS` | `8` | Maximum additional six-second gap clips used when conflicting lead samples lack repeated support. `0` disables this extra gap sampling. |
| `SECTION_VOICE_RECOVERY_CLIP_S` | `12.0` | Target duration in seconds of the initial coverage/recovery clips. Additional checks can use different durations. |
| `INSTRUMENTAL_FULL_COVERAGE_CHECK` | `True` | Run a dedicated human-voice presence check across the full mix when all sampled lead checks fail. Speech, chanting, rap and wordless singing count as human vocals. An absence verdict requires high-confidence negative coverage; a failed lead check alone does not establish an instrumental. |
| `INSTRUMENTAL_EXTRA_PRESENCE_RETRY` | `True` | Allow an extra subdivision of ambiguous presence-check windows. Setting this to `False` can improve vocal identity detection speeds at the expense of accuracy on certain tracks by reducing presence verification; more results may remain unresolved. It does not change the Wikipedia lookup or directly disable co-lead checks. |
| `INSTRUMENTAL_PRESENCE_MAX_CLIPS` | `90` | Maximum listening-model requests for a full-mix presence-verification run, including retries and its positive-vote corroboration. If full coverage cannot be completed confidently, the check cannot declare the track instrumental. |
| `PRESERVE_UNBROKEN_COLEAD_IDENTITY` | `True` | Preserve a supported unbroken male identity when resolving confirmed mixed leads, including a separately documented female co-lead. Female-sounding samples alone do not establish a female singer's identity. |

## Additional structure-label control

| Variable | Default | Description |
|---|---:|---|
| `STRUCTURE_RETURNING_PRECHORUS_LABELS` | `True` | Allow a returning section to receive a pre-chorus label when it closely matches repeated earlier pre-choruses and passes the structural and lyric checks. This rule changes labels and numbering, not section boundaries. |

## Vocal-presence behaviour notes

- Failed vocal-presence corroboration can leave human vocal presence unresolved. In that case the pipeline skips Whisper transcription and singer-identity resolution, and the report/export withholds unsupported generated lyric and identity blocks. This is not automatically an instrumental verdict.
- Missed co-lead recovery can use additional mix/stem checks and two disjoint longer-context clips. Confirming mixed leads still requires corroboration and direct different-person comparisons. If sampled male and female leads disagree without sufficient confirmation, the identity/count is qualified as uncertain rather than confidently reported as one singer.
- Untagged tracks can record an accepted audio-only vocalist category without inventing a named performer. An unreliable pitch reading cannot overturn two agreeing, highly confirmed adult-male vocal passes under the supported recovery conditions.
- Cover-art description requests preserve the supplied image through retries and do not fall back to a text-only description. A model reply claiming no image was attached is not saved as a visual observation. The existing artwork file can still be saved if a usable description is unavailable.
- Skipping transcription removes lyric timing from the structure detector's inputs. This can change section counts and boundary times even though the vocal-presence changes do not edit the section algorithm itself.
- These behaviours do not introduce additional command-line options. The variables in this reference are Python configuration constants unless explicitly listed under **Environment overrides**.

## Ollama startup and chat

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_LAUNCH_MODE` | `"window"` | How Musiclyse starts Ollama when none is running: `"off"` (start it yourself), `"background"` (hidden; stopped on exit), `"background_log"` (hidden, with its output in `OLLAMA_LAUNCH_LOG`; stopped on exit), `"log_window"` (hidden, with the log shown live in its own Terminal window) or `"window"` (its own Terminal window, left running after Musiclyse ends). An Ollama that is already running is used as it is. The window modes are for macOS; elsewhere they act as `"background_log"`. |
| `OLLAMA_LAUNCH_LOG` | `"logs/ollama.log"` | The log file for `"background_log"` and `"log_window"`; a relative path is inside the script's folder. |
| `OLLAMA_LOG_WINDOW` | `"behind"` | Where the `"log_window"` window opens: `"behind"` Musiclyse's window, or `"minimized"` into the Dock. |
| `UNLOAD_CHAT_MODEL_ON_EXIT` | `True` | Free the chat model's memory in Ollama when Musiclyse closes. |
| `CHAT_NUM_CTX_BY_MODEL` | *(see script)* | Context window per chat model name; only the chat uses it, scans keep their own windows. |
| `CHAT_THINKING` | `False` | Whether the chat model thinks before replying: `False` never (`/think` asks for one reply), `True` always, `None` leaves it to the model or Ollama. |
| `CHAT_KEEP_IMAGES` | `"auto"` | Keep earlier images in the conversation: `"auto"` keeps them on GGUF models and drops them on others; `True` always keeps; `False` always drops. |
| `CHAT_TRIM_HISTORY` | `True` | When the conversation fills its share of the context window, drop its oldest quarter. Loaded exports and looked-up background are never removed. |
| `CHAT_TRIM_TARGET` | `0.75` | Share of the conversation's room kept after a trim. |
| `CHAT_RESUME_MAX_TOKENS` | `None` | Cap on the window used when resuming a saved chat (`None` uses the model's window). Saved chats keep the full archive. |
| `CHAT_IMAGE_TOKENS` | `1500` | Token estimate per image used while the chat sends images. |
| `CHAT_SOFT_REFUSALS` | `True` | The default chat and the games decline anything they won't do briefly, in their own voice, offering something close instead. Only changes how they say no, not what they'll discuss. |
| `PERSONA_CONFIRM_FULL` | `True` | The "Persona set to:" confirmation shows the whole description (`False`: only its first 160 characters / first line). |
| `PERSONA_IMAGE_REFERENCE_ONLY` | `True` | `/persona` waits for a reference image only when the description points to a picture ("the person in this photo"), not for roles like "a portrait painter". |
| `CONSISTENT_TEXT_COLOR` | `True` | The stage timing table and the exporter's and SongFormer helper's messages (when run by `/export`) use the same colours as Musiclyse's own output. |
| `PRINT_STAGE_TIMING` | `True` | Print a per-stage timing table after each fresh analysis. |

## Commands on and off

| Variable | Default | Description |
|---|---|---|
| `EXPORT_COMMAND` | `True` | Enable `/export`, which runs the exporter script beside this one with the same arguments it takes in Terminal. |
| `EXPORT_SEARCH_DIR` | `saved-songs/llm-ready` | Where `/load` looks for exports (and in every folder inside it); a full path to a `.txt` anywhere also works. |
| `MUSICVIDEO_COMMANDS` | `True` | Enable `/musicvideoanalyse` and `/musicvideoload`. |
| `MUSICVIDEO_EXPORT_LEVEL` | `"plus"` | The export level `/musicvideoload` loads, and the one its export command makes. |
| `MUSICVIDEO_EXPORT_WIKI` | `False` | Whether that export includes Wikipedia background. |
| `MUSICVIDEO_PAIR_FILE` | `"musicvideo-pair.json"` | File in the video's folder recording which song save it pairs with. |
| `LOOKUP_SEARCH_COMMAND` | `True` | Enable `/lookup-search`. |
| `LOOKUP_SEARCH_MENU_MAX` | `0` | Titles listed in one `/lookup-search` menu; `0` lists all. |
| `LOOKUP_SEARCH_DRAFT_MENU` | `True` | The `/lookup-search` menu opens on the input line and the chosen title goes back on the line (as `/load ""` does). |
| `LOOKUP_WITH_LOAD_COMMANDS` | `True` | `/lookup` and `/lookup-search` work in any order with `/load`, `/videoload` and `/musicvideoload`. |
| `CHAT_WIKI_LOOKUP_DEBUG` | `False` | Print details of each `/lookup` search. |
| `HELP_COMMAND` | `True` | Enable `/help`. |
| `CHATLOAD_NAME_MATCH` | `True` | `/chatload name` finds saves named `<date-time>-name.json` (a menu when several match). |
| `VIDEO_FRIENDLY_USAGE` | `True` | `/videoanalyse` or `/musicvideoanalyse` without a video shows a usage line instead of an error dump. |
| `VIDEO_CHAT_MAX_TOKENS` | `32000` | Token budget for a loaded video report (before song context). |
| `VIDEO_CHAT_AUTO_FRAMES` | `0` | Evenly spaced reference images sent with each loaded video (`0`–`8`; `0` loads text only). |
| `VIDEO_IGNORE_TAG_ARTIST_TITLE` | `True` | For video files, embedded artist/title tags are ignored; both come from the filename. |
| `VIDEO_DEFAULT_SAVE_NAME` | `True` | `/videoanalyse` with no name uses the audio analysis's save name ("Artist - Song (music video)"). |
| `VIDEO_SAME_SOURCE_SYNC` | `True` | When a loaded song was analysed from the same video file, the chat is told song time = video time. |
| `SONGFORMER_SLASH_COMMAND` | `False` | Enable the `/songformer` command (setup, selftest, scan, compare, check). Analyses still use SongFormer when it is installed. |
| `WIKI_DELETE_COMMAND` | `False` | Enable `/wikidelete` (mark Wikipedia articles and make a clean copy of the database without them) and the mark/preview keys in `/lookup-search` menus. |
| `WIKI_DELETE_LIST` | `""` | The list of marked titles; `""` uses `<database>-deletions.txt` beside the database. |
| `WIKI_HIDE_MARKED` | `True` | With `WIKI_DELETE_COMMAND` on, marked articles are left out of `/lookup` and `/lookup-search`. |

## SongFormer (optional section-map check)

| Variable | Default | Description |
|---|---|---|
| `SONGFORMER_COMMAND` | `True` | Use SongFormer (installed with `install_songformer.sh`) during analyses. |
| `SONGFORMER_CORRECT_STRUCTURE` | `True` | Let SongFormer correct section maps; `False` turns every rule below off. |
| `SONGFORMER_SNAP_TIMES` | `True` | A boundary both find moves to SongFormer's time (at most a bar). |
| `SONGFORMER_ADD_SECTIONS` | `True` | A missed section change is added (not on instrumentals or dance tracks). |
| `SONGFORMER_NAME_VOTE` | `True` | SongFormer votes on section names. |
| `SONGFORMER_INSTRUMENTAL_NAMES` | `True` | An instrumental's first/last part is named Intro/Outro when both agree. |
| `SONGFORMER_STRONG_MOVES` | `True` | Where SongFormer is very sure of a boundary, it can move a nearby one. |
| `SONGFORMER_LATE_STARTS` | `True` | A part Musiclyse starts a few seconds early can start where SongFormer hears it. |
| `SONGFORMER_SURE_ADDS` | `True` | Adds a part SongFormer is sure of when Musiclyse's next part is the kind SongFormer hears. |
| `SONGFORMER_SURE_EARLY_STARTS` | `False` | Experimental: earlier starts where SongFormer is sure. |
| `SONGFORMER_AFTER_ANALYSIS` | `False` | Also scan each new save with the SongFormer helper after `/analyse`, `/reanalyse`, `/batch` and `/musicvideoanalyse`. |
| `SONGFORMER_PYTHON` | `""` | SongFormer's own Python; `""` uses `~/SongFormer/env/bin/python`. |
| `SONGFORMER_REPO` | `""` | SongFormer's code folder; `""` uses `SongFormer` beside the script, else `~/SongFormer/code`. |
| `SONGFORMER_DEVICE` | `"auto"` | `auto` (the Mac GPU when it works, else the CPU), `mps` or `cpu`. |
| `SONGFORMER_SCAN_TIMEOUT_S` | `600` | A scan taking longer is abandoned and the map stays as it was. |

## Further analysis controls

| Variable | Default | Description |
|---|---|---|
| `ENABLE_JAMENDO_INSTRUMENTS` | `True` | MTG-Jamendo instrument tagging (models downloaded by the installer to `~/mtg_jamendo_data`; licence CC BY-NC-SA 4.0, non-commercial). |
| `JAMENDO_MODEL_DIR` | `""` | Model folder; `""` uses `~/mtg_jamendo_data`. |
| `JAMENDO_STEMS` | `True` | Also tag each separated stem. |
| `JAMENDO_THREADS` | `4` | onnxruntime CPU threads. |
| `ENABLE_LISTENING_DETAILS` | `True` | Extra listening-model passes describing voice delivery and production, effects and playing style, saved as `listening_details` and used by plus-level and higher exports. |
| `LISTENING_DETAILS_SECTIONS` | `True` | Also describe each window across the song (`False`: whole-track pass only). |
| `LISTENING_DETAILS_WINDOW_S` | `30.0` | Length of each window in seconds. |
| `LISTENING_DETAILS_MAX_WINDOWS` | `16` | Maximum windows; longer tracks get an even spread. |
| `TEMPO_HALFTIME_CHECK` | `True` | Detect songs whose drums play a half-time backbeat at a fast measured tempo, and report the slower tempo. |
| `TEMPO_HALFTIME_BOTH_READINGS` | `True` | Where halving is too close to call, keep the tempo and note that the song can also be counted at half speed. |
| `STRUCTURE_SPLIT_CHORUS_REPEATS` | `True` | Split a long chorus that is really the chorus sung twice. |
| `STRUCTURE_CHORUS_SHARED_WORDS` | `True` | A section named Chorus whose sung words are a verse's (not the words the other choruses share) is named as a verse. |
| `BATCH_STALL_RETRY` | `True` | `/batch`: a track whose listening model produces nothing for `BATCH_STALL_MINUTES` is stopped and analysed again once. |
| `BATCH_STALL_MINUTES` | `10.0` | Minutes without output before a stalled track is retried. |

## Exporter settings (at the top of `musiclyse-0.7-export.py`)

| Variable | Default | Description |
|---|---|---|
| `MOSS_LOW_TRUST_FIELDS` | `True` | Treat prompt echoes and stock example answers as unreliable evidence in the chat view. |
| `GENRE_STYLE_LED` | `True` | Refine or replace the chat-view recommended genre with sufficiently supported style evidence. |
| `GENRE_STYLE_REFINE_MIN` | `0.35` | Minimum style confidence for refining a broad genre within its family. |
| `GENRE_STYLE_REPLACE_MIN` | `0.30` | Minimum style confidence for replacing a genre family, with additional corroboration checks. |
| `ERA_STYLE_CHECK` | `True` | Use a sufficiently confident period-specific style to check overly recent era estimates in the chat view. |
| `ERA_STYLE_MIN` | `0.25` | Minimum style confidence for the era check. |
| `ERA_STYLE_MAX_SPAN` | `25` | Maximum style heyday span in years for era checking. |
| `REGIONAL_GENRE_GATE` | `True` | Require language or artist-background support for regional genre labels; suppress generic world/international labels. |
| `GENRE_SYNTH_ROCK` | `True` | Combine supported rock-band and prominent synth evidence into a synth-rock reading. |
| `HIDDEN_BASS_NOTE` | `True` | When deep sub-bass is a large share of the track's energy but the separated bass part reads as silent for much of the time the drums play, the export warns that the bass is probably there (a synth bass can end up in another part). |
| `HIDDEN_BASS_MIN_SUB` | `25.0` | Minimum whole-track sub-bass share (%) for that warning. |
| `HIDDEN_BASS_MIN_SILENT` | `0.40` | Minimum share of the drum-playing time with the bass part silent. |
| `CORRECTION_SCOPE_NARROW` | `True` | In the export's instructions, only who sings and background facts the notes don't settle count as confirmed when the user states them; corrections about what the recording contains are weighed instead (`False`: any explicit factual correction counts as confirmed). |


## Configurable command names

The slash-command spellings above can be changed in `USER CONFIGURATION`:

| Variable | Default |
|---|---|
| `ANALYSE_FLAG` | `"/analyse"` |
| `REANALYSE_FLAG` | `"/reanalyse"` |
| `BATCH_FLAG` | `"/batch"` |
| `LOAD_FLAG` | `"/load"` |
| `EXPORT_FLAG` | `"/export"` |
| `LOOKUP_FLAG` | `"/lookup"` |
| `LOOKUP_SEARCH_FLAG` | `"/lookup-search"` |
| `MUSICVIDEO_ANALYSE_FLAG` | `"/musicvideoanalyse"` |
| `MUSICVIDEO_LOAD_FLAG` | `"/musicvideoload"` |
| `QUIET_FLAG` | `"/quiet"` |
| `THINK_FLAG` | `"/think"` |
| `GAME_FLAG` | `"/game"` |
| `GAME_EXIT_FLAG` | `"/game-exit"` |
| `CLEAR_FLAG` | `"/clear"` |
| `CLEAR_ALL_FLAG` | `"/clearall"` |
| `PERSONA_FLAG` | `"/persona"` |
| `HELP_FLAG` | `"/help"` |
| `DEBUG_FLAG` | `"/debug"` |

## Environment overrides

These are actual environment variables, unlike the Python constants above. Set them before starting the script.

| Variable | Script fallback | Description |
|---|---|---|
| `MOSS_MUSIC_CODE_DIR` | Search beside the script, then `~/MOSS-Music` | Repository location when the Python constant of the same name is `None`. A configured Python path takes priority. |
| `PYTORCH_MPS_HIGH_WATERMARK_RATIO` | `"0.0"` | PyTorch MPS allocator upper watermark; this fallback removes its upper allocation limit. An existing environment value wins. |
| `PYTORCH_MPS_LOW_WATERMARK_RATIO` | `"0.6"` | MPS allocator cleanup watermark. An existing environment value wins. |

---

**Scope:** This is a practical configuration reference, not an exhaustive dump of every tuning constant. Additional `STRUCTURE_*` rules, instrument-probe thresholds, vocal/genre safeguards and algorithm thresholds are documented beside their assignments in `USER CONFIGURATION`. Runtime state, lookup tables and full prompt bodies are omitted. The default chat's prompt (`DEFAULT_CHAT_PROMPT`) is defined elsewhere in the script; use `/persona` for routine persona changes.


# Possible future features:

These are a few things that I want to implement in the future but so far haven't done

* Separate the configuration flags from the main script in a different file
* Most of the project was vibecoded with the help of several LLMs, and while the current build appears to be fully usable, those with better coding knowledge could probably help patch any issues.
* Further improved timestamp labelling
* Further improved estimated era accuracy for songs without a year tag



![Musiclyse - A Gourlish Vibe Project](https://pbs.twimg.com/media/HQH-IMYWoAAX-p6.jpg)
