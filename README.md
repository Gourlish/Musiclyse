# Musiclyse
A highly configurable piece of local AI chatbot software that can "listen" to music via a multi-step analysis engine, compare tracks and more.

![Musiclyse - Local Music Discussion](https://gourlish.altervista.org/musiclyse/musiclyselogo.svg)

![The Free Music Analyser](https://gourlish.altervista.org/musiclyse/slogan.svg)

![Chat About Your Songs](https://gourlish.altervista.org/musiclyse/chat.svg)

![100 Percent Local](https://gourlish.altervista.org/musiclyse/local.svg)

![Musiclyse - Screenshot](https://gourlish.altervista.org/musiclyse/Musiclyse04.png) 

# About:

Musiclyse is a Python script designed for high-spec Macs (it is easily portable to good quality Linux PCs) that enables local conversations about music. It utilises a multi-step song processing pipeline linked to an Ollama hosted local LLM of the user's choice. This consists of a pipeline process of MOSS-Music, Essentia/PANNs, Discogs-400 genre detection, Demucs separation fed to Omnizart, Wikipedia database RAG retrieval for artist context, and metadata analysis (which will improve recognition for lyrics). You're welcome to help improve the project to make its song recognition better!

# Key features:

The following features are all included in Musiclyse 0.6:

* Beat, melody, genre, vocal, song key, song structure, scale and era analysis. Accuracy for many things has been significantly improved since 0.3.
* Can get accurate facts about music adjacent topics via an optional Wikipedia-based database add-on file.
* Will talk about the concept of the songs or compare and contrast two or more different songs. You can ask it anything you can ask a normal chatbot, and it will be able to bring songs into context with the chat.
* Supports image analysis (if used with an image-supporting model).
* LLMs can be swapped out easily in the script. Currently it is optimised for use with Muse Glimmer 30B; for less powerful systems Gemma 4 26B is recommended.
* Save your analysed song data to JSON files, then load them quickly in a new session.
* Run an overnight batch of songs to JSON files that can be fast loaded at a later time.
* Create your own personas and hear their opinions about songs, as well as testing their knowledge.
* Import two tracks together to compare them easily via both musical and sonic signature analysis.
* Specify a range (x seconds to y seconds) in a track to analyse for faster processing and more precise analysis.
* Export your .json files to specially formatted .txt files, allowing any general purpose LLM to "listen" to music. You can do this via musiclyse_export.py.

## Version history:

* 2026/10/27 - Musiclyse 0.6. Now uses Python 3.11 (due to the end of Python 3.10 support). Replaced Music Flamingo with MOSS-Music as the listening model, which scores higher in benchmarks and is less prone to erratic hallucinations (being built on Qwen 3 8B instead of Qwen 2.5 7B). Mood and style detection should now be noticeably more detailed. This does require a different installation process to before. Noticeable improvements to timestamp and song section detection accuracy, as well as song key and BPM detection accuracy. Refined the .json-to-txt export script to take into account the updates and improvements made. Much improved genre classification. Fixed a number of memory caching and disk swap bugs that were causing high memory pressure on Macs with 64GB RAM. You should notice that songs process noticeably faster than in 0.5 (which brought down times noticeably from 0.4). Refinements to vocal identity detection. Implemented a system attempting to allow output LLMs to discuss music theory in a song based on the extracted MIDI and time signature information, notably things like chord progressions. Added a new install script that should make the software easier to setup.
* 2026/10/15 - Musiclyse 0.5. Makes various improvements from 0.4.
* 2026/09/03 - Musiclyse 0.4. Implemented Whisper transcription for lyrics, replacing Music Flamingo for this task (you will need to download a new model). Added CLAP embeddings to further help determine song similarity in comparisons (this also requires a new model). Further refined instrument and melody recognition capabilities. Improved song structure recognition with more accurate timestamps. Further refined vocalist recognition capabilities. Musiclyse can now talk about where instruments are in a track and output drum grooves. Implemented a flag in the script for full MIDI analysis (at the expense of using more tokens). Fixed an issue that was truncating the middle of the system prompts, reducing output accuracy (if your token context window is set too low, you will now see a warning about prompt truncation on startup). Fixed an issue that was causing all tracks to be treated as mono. Implemented mode detection for analysed songs. Improved custom persona functionality so that custom personas are much less likely to know about things they would not be aware of, while still being able to talk about topics they are aware of. Improved era detection for songs without metadata. Reduced the likelyhood of a couple of other Music Flamingo/Qwen 2.5 7B hallucination bugs. Wikipedia database lookup logic has been noticeably improved as a result of debugging (it uses the same database as 0.3), and now produces relevant results for both songs when comparing tracks. Fixed problems with guitar being wrongly identified as not present, and organ incorrectly identified. Fixed filename import and .json extension requirement issues with /loadcompare. Fixed Ollama caching problem to make long conversations faster (this may cause old messages to disappear from the context suddenly rather than "fade" away. Fixed a bug that caused the script to crash when loading a model without vision support, or when loading an invalid model. Fixed a problem whereby the script would crash on song or image import where running with a non-vision capable output LLM. Fixed a problem whereby the script would crash if a non-valid output model was defined.
* 2026/08/28 - Musiclyse 0.3. Larger Wikipedia database (optional). Proper database search implementation, meaning much less lookup latency. Vastly strengthened lyric transcription hallucinations in Music Flamingo's Qwen 2.5 7B (though may transcribe fewer lyrics). Improved song import context reasoning. Improved track instrument recognition. Improved multi-voice recognition. Improved genre recognition. Improved BPM recognition. Fixed issue with phantom instruments being detected on subsequent tracks without them where they were present in the first track of a conversation. Musiclyse can now talk about the loudness of music in LUFS as well as dynamics crest factor. Improved vocal age range classification logic. Fixed problem that was only analysing vocal range for first 30 seconds. Prevented the Wikipedia and native LLM logic bases from using their knowledge of an artist to influence their description of the style of an analysed song, while still being able to talk about the artist. Files can now be loaded and saved with spaces and non-standard characters. Batch saves now use an "Artist Name - Song Name" naming structure. Wikipedia context is now cleared on /clear command. Added /clearall command to also purge cached songs. You can now /listen and analyse a specified second range of a track for increased speed and accuracy. The batch processing crash issues on Macs with 64GB RAM have now been properly fixed. Fixed a problem where the LLM did not recognise other musicians that the model was capable of recognising from images mid-conversation. IMPORTANT NOTE - Due to how prone Music Flamingo was to hallucinating on lyric transcriptions, I have implemented strict transcription guardrails that now commonly result in it not outputting lyrics at all, so make sure your files have lyric metadata to allow the LLM to talk about lyrics on a track! I intend to implement an alternative transcription model on a future release.
* 2026/08/24 - Musiclyse 0.2. Adds Wikipedia database integration for much more reliable chats about artists, albums and songs. Omnizart processing logs are now hidden by default. Improved batch processing optimisations to make crashes on Macs with 64GB RAM less likely. Improved BPM and instrument recognition (though there may still be some minor accuracy issues). Overall a much better chat experience from the added bank of data. Is now set to use muse-glimmer:30b-mlx by default, which significantly improves LLM processing times on MacOS. If using Linux (or Windows if you can get it working), change to the non-MLX version or a different version.
* 2026/08/22 - Musiclyse 0.1. Beat, melody, genre, vocal, song key, scale and era analysis. There are currently some minor accuracy issues which I am hoping to improve. Will talk about the concept of the songs or compare and contrast two or more different songs. You can ask it anything you can ask a normal chatbot, and it will be able to bring songs into context with the chat. Supports image analysis (if used with an image-supporting model). LLMs can be swapped out easily in the script. Currently it is optimised for use with Muse Glimmer 30B; for less powerful systems Gemma 4 26B is recommended. Save your analysed song data to JSON files, then load them quickly in a new session. Run an overnight batch of songs to JSON files that can be fast loaded at a later time. Create your own personas and hear their opinions about songs.


Below is an example of a chat in Musiclyse 0.2:

```
You: Yo Musiclyse! I got two tracks for you to ask questions about and compare. I'll send one to you first, then you'll wait for the second before answering my questions.
  ⏳ Thinking...

Musiclyse: Nice, I'm in. Send the first track over and I'll hold off until you've dropped the second before we compare.

  (tokens: this turn ~2579 [prompt 2355 + reply 224] · session ~2579 · context window 131072)

You: /load cutemloosebruce
  (loaded saved song '06 Cut em Loose Bruce.m4a' into this session)

You: /load marysgarden I wondered which of these two songs is more melodic?
  (loaded saved song '09 - Marys Garden.m4a' into this session)

  ⏳ Writing...

Musiclyse: Mary’s Garden is the more openly melodic of the two, in my ears.

Cut ’em Loose Bruce from Happy Mondays is a 1992 Yes, Please! groove — it’s a Shaun Ryder-ish story song with that late-70s/early-90s Madchester bounce, a post-puberty male lead that’s bright and slightly nasal up front, and a sunshine refrain that lands more like a spoken chant than a sung hook. The track is built around the rhythm section and that kind of half-talk, half-melody delivery, with the guitar sitting back in the mix and the whole thing feeling more narrative and dance-floor narrative than sing-along.

Mary’s Garden from The Mock Turtles’ 1990 Turtle Soup sits a little higher in the pocket — around 129 bpm vs the Mondays’ ~99 bpm — and it’s much more chorus-forward. You get that clean, slightly bright male lead asking “Oh Mary, how does your garden grow?” over and over, and the guitars are really out front with a choppy, jangly hook that you can hum straight away. The verses paint the little pictures — ballet horses, crystal bells, pretty maids in china blue — and then the chorus snaps back in a very singable, almost music-hall pop way. The Mock Turtles were part of that early-90s Manchester indie wave that loved a big, memorable refrain, and this is a textbook example.

So if you’re after the one you’ll find yourself humming on the way out the door, it’s the Mock Turtles track. The Mondays track is great for vibe and storytelling, but Mary’s Garden is the more melodic earworm of the pair.

  (tokens: this turn ~15155 [prompt 13741 + reply 1414] · session ~17734 · context window 131072)
You: How do the two compare in instrumentation and production?
  ⏳ Thinking...

Musiclyse: They’re built from very different palettes, even though both are that late-80s/early-90s Manchester-adjacent polish.

Cut ’em Loose Bruce is a rhythm-and-brass number. The drums sit right up front — punchy kit, pretty laid-back at ~99 BPM, with a tight, rhythmic electric bass that’s driving almost everything you feel in the low end. The guitar is there but it’s more of a texture, clean with a little overdrive tucked behind the groove. What really makes it pop is the brass — trumpets, trombones, saxes with melodic lines and little stabs that answer the vocal. The mix is wide and well-separated, almost hi-fi for the time, with the lead vocal dry and front-and-center, just a touch of natural room reverb. It feels loose, dance-floor storytelling, with a lot of sub-bass and low-mid weight and a pretty open dynamic range.

Mary’s Garden is much more guitar-first indie pop. It’s quicker, about 129 BPM, and the drums are busy in a different way — crisp hi-hat chatter with a tight kick/snare backbeat that gives it that bouncy, radio-ready pulse. The guitar is the star: jangly, layered, constantly moving, with a bright mid-range presence that the mix leans into. Bass is solid and supportive but stays in a more traditional pop pocket, and there’s no brass at all — just drums, bass, guitars and the lead vocal. Production-wise it’s clean and bright too, but more mid-forward and compressed in the feel, the kind of forward, radio-friendly sheen you get on early-90s indie singles. The vocal is clean and slightly bright, also front-and-center but with a more traditional pop delivery, and the whole thing is tighter and more compact than the Mondays track, which sprawls a bit more around the groove.

So one is funk/dance with horns and a deep groove, the other is jangly guitar-pop with a busy, bouncy drum pulse.

  (tokens: this turn ~16516 [prompt 14974 + reply 1542] · session ~34250 · context window 131072)
You: What about in terms of mastering?
  ⏳ Thinking...

Musiclyse: Good question — both sit in that early-90s analog-to-digital transition window, so they’re both pretty open and dynamic, but they read differently on the meters.

Cut ’em Loose Bruce from Happy Mondays’ 1992 Yes, Please! feels a touch more open. The file I’m hearing shows a higher crest factor, about 16.3 dB, which gives you that more dynamic, less squashed master — peaks breathe a bit, the drums and brass have room to hit and then fall back. The overall tonal balance is darker down low, with a big sub-bass and bass footprint and a lot of low-mid weight. The top end is polite, a little airy but not aggressive, which matches the wide stereo imaging and that classic late-80s/early-90s hi-fi feel.

Mary’s Garden from The Mock Turtles’ 1990 Turtle Soup is a little more radio-forward. Crest factor sits around 14.1 dB, so it’s still dynamic for the era, but a touch more contained — you get that slightly tighter, more compressed-for-single feel. It’s brighter overall, with more midrange presence and a little less sub content; the mix leans into the guitars and vocal, and the drums sit a bit hotter in the mix. The stereo image is still wide, but the mastering is a bit more mid-forward and a bit more polished in the top, which helps it cut on radio and smaller speakers.

So in practice: the Mondays track feels a little looser and more spacious, with more low-end weight and dynamic headroom; the Mock Turtles track feels a bit tighter, brighter, and more immediate for single play.

  (tokens: this turn ~16630 [prompt 16277 + reply 353] · session ~50880 · context window 131072)
You: I hear a somewhat Bowieish sound in Mary's Garden
  ⏳ Thinking...

Musiclyse: That’s a fun catch — I hear it too.

Mary’s Garden has that early-90s Manchester indie-pop sheen that was chewing on a lot of late-70s pop glam influences. The Mock Turtles were coming out of that 1985 Middleton scene with Martin Coogan at the helm, and by the time Turtle Soup landed in 1990 they were leaning into big, singable choruses and jangly guitars with a slightly theatrical vocal delivery. The track is built around a bright, bouncy guitar hook, a clean slightly bright male lead asking “Oh Mary, how does your garden grow?” and those little music-hall images — crystal bells, pretty maids in china blue, walking under the gazing moon — that give it a storybook, almost Ziggy-era playfulness.

It’s not a direct Bowie quote, but the combination of that chiming guitar, the bouncy 129 bpm pulse, the wry, slightly cheeky lyricism and the big, open chorus is exactly the kind of thing you’d get when a bunch of Manchester kids were digesting Bowie, Roxy Music, and the whole glam-pop lineage via the Madchester filter. The Mock Turtles were famously part of that early-90s indie pop wave that loved a memorable hook and a bit of drama, so the Bowieish sparkle you’re picking up is really that glam-pop sensibility filtered through a 1990 indie rock lens.

Nice ear — it’s that little theatrical, slightly theatrical pop lift that makes the song stick.

  (tokens: this turn ~18623 [prompt 18303 + reply 320] · session ~69503 · context window 131072)
You: In the case of that Happy Mondays album, it was widely panned by critics
  ⏳ Thinking...

Musiclyse: Yeah, Yes Please! was a rough landing.

Happy Mondays went out to Barbados in 1991 to record with Chris Frantz and Tina Weymouth at Eddy Grant’s studio. When Yes Please! came out in 1992 it was a commercial failure — it basically helped bankrupt Factory Records — and the critical reception was pretty savage. The press were tired of the excess and the whole “Madchester” hangover, and the album landed with a thud compared to the platinum peak of Pills ’n’ Thrills and Bellyaches.

Cut ’em Loose Bruce is actually one of the brighter moments on that record; the album’s press was mainly negative overall, with reviewers calling the whole thing unfocused and overblown. So you’re right — the album was widely panned, even if that one track still has its fans.

  (tokens: this turn ~20612 [prompt 20435 + reply 177] · session ~90115 · context window 131072)
You: Even Cut Em Loose Bruce isn't great in my view, and it was probably one of the better songs on the album, which speaks for just how awful Yes Please was
  ⏳ Thinking...

Musiclyse: Yeah, I hear you. Yes Please! is the Happy Mondays record most fans try to make peace with and then quietly shelve.

By ’92 the whole Madchester moment was already fizzling out — Nevermind had landed, the scene was tired, and the band was in a bad way on and off the road. They went down to Barbados to record with Chris Frantz and Tina Weymouth at Eddy Grant’s studio, and what came back felt loose in all the wrong places. It flopped commercially, it was a big part of what finally bankrupted Factory Records, and the press at the time were pretty brutal about it.

Cut ’em Loose Bruce is the kind of track that ends up being “one of the better ones” by default — it’s got a hook and a bit of that old Mondays swagger, but even that feels a little thin and distracted compared to Pills ’n’ Thrills. When the best song on an album is a shrug, that says it all about where the record is.

It’s wild how much the Mock Turtles’ Turtle Soup from the same era feels more alive in the moment — same Manchester scene, same year-ish window, but Mary’s Garden still wants to be sung along to.

  (tokens: this turn ~23070 [prompt 22177 + reply 893] · session ~113185 · context window 131072)
You: Anyway, before we go, which Madchester bands sounded closer to the Mock Turtles and which sounded closer to Happy Mondays?
  ⏳ Thinking...

Musiclyse: That’s a nice way to split the scene up.

The Mock Turtles were really the pop-jangle side of the Manchester bubble. Turtle Soup in 1990 is all clean guitars, big choruses and a music-hall wink — that’s the indie-pop lineage that sits much closer to the more song-focused end of Madchester. Think The Stone Roses when they’re at their melodic, The Charlatans UK with their organ-driven pop, The La’s, The Chameleons in their more romantic moments, and bands on the Imaginary/Rough Trade orbit that wanted a hook first and a groove second. It’s the “sing-along on a sunlit afternoon” end of the scene.

Happy Mondays are the other pole — the baggy/dance-funk side. Shaun Ryder and the crew fused indie guitars with house, funk and northern soul, and their records were constantly being remixed by DJs. That puts them closer to the rave-leaning Manchester acts: 808 State, A Guy Called Gerald, The Shamen, Inspiral Carpets, and that whole Haçienda-driven set where the rhythm section and the dancefloor were the point. Even The Stone Roses sit a bit closer to the Mondays in vibe and attitude than the Mock Turtles do, with their psychedelic, groove-heavy sprawl.

So basically: jangly pop and big choruses — Mock Turtles territory. Loose funk, house-tinged baggy and dance-floor storytelling — Happy Mondays territory.

  (tokens: this turn ~25026 [prompt 24049 + reply 977] · session ~138211 · context window 131072)
You: Nice talking to you - you got all of that brilliant! It's proof you're working as I intended.
  ⏳ Thinking...

Musiclyse: That made my day — glad it landed how you wanted. Always happy to nerd out about Madchester with you.

  (tokens: this turn ~25353 [prompt 25254 + reply 99] · session ~163564 · context window 131072)
```


## System requirements

Due to the nature of what it's doing, it's intended to be run on powerful machines with a good quality GPU and 64GB RAM or higher. Currently it's only been tested on MacOS, but it should be pretty easy to get running on Linux, and is potentially runnable on Windows. For users with less powerful machines, I'd advise swapping Muse Glimmer for Google's Gemma 4 26B A4B, which has reasonable knowledge but is not as resource heavy. Users with lower spec machines might benefit from installing a 4-8B model via Ollama. You can swap out the Ollama model by altering the "OLLAMA_MODEL" value with the exact name of your model.

# Manual Installation for 0.6 (for MacOS):

1: Ensure Python 3.11 is installed on your system

2: Clone the provided files onto your machine for the latest version, or alternatively download a version from the Releases section.

3: [Download the heavy version of the Musiclyse Wikipedia Database via this link](https://github.com/Gourlish/Musiclyse-Wikipedia-Database/releases/tag/music_wiki_heavy-august2026). Alternatively you can [download a lite version](https://github.com/Gourlish/Musiclyse-Wikipedia-Database/releases/tag/music_wiki_lite-august2026) that covers fewer topics but consumes less storage (it will run without either, but it's likely to give far better context about artists if you install it). Once extracted (the heavy database should be listed as 8.33GB on MacOS, or a bit less on Windows and many distributions of Linux) place the music_wiki_heavy.db file in the same directory as the Musiclyse Python script. If you are using the Lite version of the database, you will need to search the script for mentions of "music_wiki_heavy.db" and replace this with "music_wiki_lite.db".

3: Install required system level dependencies (if you don't already have Homebrew installed you'll need it too)

```
brew install ollama ffmpeg portaudio
```

4: In a new terminal window, start Ollama and install Muse Glimmer (if you want a different model for output, change the specification in musiclyse-0.6.py via the "OLLAMA_MODEL =" flag to the exact Ollama name of the model of your choice, and pull that different model with Ollama). By default the model is configured to use the MLX version of Muse Glimmer, though this will only work on a Mac.

```
ollama serve
ollama pull muse-glimmer:30b-mlx
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
python musiclyse-0.6.py
```

# Configurable Variables & Flags in Musiclyse 0.6

## Command-line

| Usage | Description |
|---|---|
| `python musiclyse.py` | Start without a track; choose one with `/listen`. |
| `python musiclyse.py /path/to/song.m4a` | Start with a local audio file. A URL can also be supplied where supported. Use the actual script filename if it has not been renamed to `musiclyse.py`. |
| `python musiclyse.py --batch-one /path/to/song.mp3` | Internal: isolated single-track batch worker used by `/batch`. |
| `python musiclyse.py --batch-one-analyze /path/to/song.mp3 /tmp/staging.json` | Internal: heavy-analysis stage for split-process batch mode. |
| `python musiclyse.py --batch-one-finish /tmp/staging.json` | Internal: identity/save stage for split-process batch mode. |
| `python musiclyse.py --omnizart-prefetch /tmp/input.json /tmp/output.json` | Internal: separate-process Omnizart transcription worker; not a normal startup option. |

## In-chat slash commands

| Command | Description |
|---|---|
| `/listen <question>` | Analyse the current track; the first full analysis is cached for follow-up questions. |
| `/listen <path or URL> <question>` | Switch to another track and analyse it. |
| `/listen <path> [start-end] <question>` | Analyse an audio excerpt, with the window in seconds; e.g. `/listen song.mp3 [30-60] analyse the solo`. An excerpt cannot be saved as a full-track scan. |
| `/relisten [<path or URL>] <question>` | Force a fresh full analysis instead of reusing the cache; a path/URL can be supplied to switch tracks. |
| `/correct field=value[, field=value…]` | Record a user-confirmed fact that overrides the analysis for the current track (e.g. `/correct year=1966`). |
| `/save=filename.json` | Save technical details for the most recently scanned full track in `saved-songs/` (or the configured `SAVE_DIR`). |
| `/load filename.json [question]` | Load a saved track; the question is optional. Names with spaces and punctuation are supported. |
| `/loadcompare "track1.json" "track2.json" question` | Load two or more saved tracks together and ask a comparison question. Saved CLAP embeddings can add sonic-similarity evidence when available. |
| `/batch /path/to/folder` | Scan audio files in a folder to JSON files in `saved-songs/`; does not import them into chat and skips files already saved. |
| `/clear` | Wipe chat context and reset session token counters; keep in-memory track analyses. |
| `/clearall` | Wipe chat context and in-memory track analyses. JSON files on disk are kept and can still be loaded. |
| `/persona <description>` | Set the writer's chat voice/taste; music-evidence rules stay in force. |
| `/persona` | Show the current persona. |
| `/persona reset` | Restore the default music-obsessed-friend persona (`default` also works). |
| `/debug` | Show debug information, including the most recent writer message. |
| `quit` | Exit the application. |

## Console / display

| Variable | Default | Description |
|---|---:|---|
| `FRIENDLY_PROCESSING_DISPLAY` | `True` | Use a compact, progress-bar-style display instead of a scrolling log. |
| `USE_COLOR` | Automatic | Enables terminal colour when `sys.stdout.isatty()` is true. This is detected by the script, not a normal user-config constant. |
| `SHOW_RAW_ANALYSIS` | `False` | Print MOSS-Music's full raw analysis for debugging. |
| `DEBUG_MODEL_LOADING` | `False` | Show raw model-loading logs instead of a status line. |
| `SHOW_OMNIZART_LOGS` | `False` | Allow Omnizart stdout/stderr through instead of suppressing it. |
| `DEBUG_STEM_MIDI` | `False` | Print verbose per-stem note-count debugging. |

## Ollama / model settings

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_URL` | `"http://localhost:11434/api/chat"` | Ollama chat endpoint. |
| `OLLAMA_MODEL` | `"muse-glimmer:30b-mlx"` | Writer model used for the final response and general chat. Pull this model in Ollama, or change the name to another installed model. |
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
| `PARALLEL_ANALYSIS` | `True` | Run independent analysis tasks concurrently where supported. Background failures are retried in order. URL and excerpt analyses run sequentially. |
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
| `CHAT_HIDE_SECTION_NAMES` | `True` | In chat evidence, refer to measured song sections by part number and time rather than names such as verse/chorus. |
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
| `COMPARE_VOICE_PITCH` | `True` | Include measured lead-voice pitch comparisons in `/loadcompare` where available. |
| `INSTRUMENT_REED_TAG_CHECK` | `True` | Flag unsupported reed-instrument tags when other evidence suggests a synth or sax-like sound. |
| `ENABLE_CLAP_EMBEDDING` | `True` | Compute and save CLAP audio embeddings for sonic-similarity evidence in multi-track comparisons. Requires compatible Transformers CLAP classes. |
| `PRODUCTION_ANALYSIS_ENABLED` | `True` | Run the dedicated production-era analysis. |
| `ENABLE_COVER_ART_DESCRIPTION` | `True` | Describe embedded cover art when available. |
| `ENABLE_SINGER_IDENTITY_RESOLUTION` | `True` | Resolve the singer's likely identity from audio and available context. |
| `ENABLE_IMAGE_OBSERVATIONS_FOR_GENERAL` | `False` | Allow image observations in ordinary general-chat turns. This is off by default; track cover-art description is controlled separately. |
| `ENABLE_WIKI_CONTEXT` | `True` | Add locally indexed Wikipedia background to relevant questions when the database is available. |

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
| `SEND_COVER_ART_TO_OLLAMA` | `True` | Send prepared embedded cover art to the writer model when image input is supported. |
| `METADATA_MAX_LYRICS_CHARS` | `10000` | Maximum lyric characters accepted from file metadata. |
| `LYRICS_MAX_CHARS` | `12000` | General maximum lyric text length. |
| `COVER_IMAGE_MAX_DIMENSION` | `512` | Maximum cover-art dimension in the prepared image. |
| `COVER_IMAGE_QUALITY` | `"2"` | MJPEG qscale for cover-image conversion; lower values mean better quality. |
| `MAX_COVER_IMAGES_PER_REQUEST` | `1` | Maximum cover images sent per request. |
| `MAX_COVER_BYTES_TO_SEND_RAW` | `1_000_000` | Maximum cover-image byte size sent without further conversion. |
| `MAX_EXPLICIT_IMAGE_BYTES` | `2_000_000` | Byte limit used when preparing explicit images supplied in chat. |
| `EXPLICIT_IMAGE_MAX_DIMENSION` | `1568` | Maximum dimension for explicit images; larger/unsupported images are converted rather than rejected. |
| `MAX_IMAGES_PER_REQUEST` | `8` | Maximum images considered for a request. |
| `MAX_IMAGES_TO_DESCRIBE` | `2` | Maximum images to describe in a turn. |
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
| `WIKI_CONTEXT_REFRESH_EVERY_QUESTION` | `False` | Refresh cached Wikipedia context for every question. |
| `WIKI_SEARCH_EVERY_MESSAGE` | `True` | Search for relevant context on every message. |
| `WIKI_DEBUG_LOG` | `True` | Print a one-line status after general-chat questions showing whether the database returned a match. |

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
| `BATCH_ISOLATE_PER_TRACK` | `True` | Run each track in a fresh Python subprocess; recommended on macOS. |
| `BATCH_SPLIT_IDENTITY_PROCESS` | `True` | Separate heavy analysis from identity/save work to reduce out-of-memory risk. |
| `BATCH_LOG_RSS` | `True` | Log approximate process resident memory after cleanup. |
| `BATCH_UNLOAD_OLLAMA` | `True` | Unload the Ollama writer model during heavy batch-analysis stages. |
| `BATCH_OLLAMA_KEEP_ALIVE` | `0` | `keep_alive` for short batch helper requests; `0` unloads immediately. |
| `BATCH_PRE_IDENTITY_SETTLE_S` | `5.0` | Wait before the identity/save stage to let memory settle. |

## Persona / prompts / saved files

| Variable | Default | Description |
|---|---|---|
| `DEFAULT_PERSONA_PROMPT` | *(see script)* | Default writer persona; custom personas can be set at runtime with `/persona`. |
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
| `MOSS_LOW_TRUST_FIELDS` | `True` | Treat prompt echoes and stock example answers as unreliable evidence in the chat view. |


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
| `FRESH_STEM_STATE_PER_TRACK` | `True` | Reset per-track stem state before `/listen` analysis to prevent reuse of another track’s evidence. |
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
| `GENRE_STYLE_NOTE` | `True` | Allow supported style labels to refine genre answers. |
| `GENRE_STYLE_LED` | `True` | Refine or replace the chat-view recommended genre with sufficiently supported style evidence. |
| `GENRE_STYLE_REFINE_MIN` | `0.35` | Minimum style confidence for refining a broad genre within its family. |
| `GENRE_STYLE_REPLACE_MIN` | `0.30` | Minimum style confidence for replacing a genre family, with additional corroboration checks. |
| `ERA_STYLE_CHECK` | `True` | Use a sufficiently confident period-specific style to check overly recent era estimates in the chat view. |
| `ERA_STYLE_MIN` | `0.25` | Minimum style confidence for the era check. |
| `ERA_STYLE_MAX_SPAN` | `25` | Maximum style heyday span in years for era checking. |
| `REGIONAL_GENRE_GATE` | `True` | Require language or artist-background support for regional genre labels; suppress generic world/international labels. |
| `GENRE_ERA_CHECK` | `True` | Check genre labels against the file’s year tag to avoid anachronistic labels. |
| `GENRE_SYNTH_ROCK` | `True` | Combine supported rock-band and prominent synth evidence into a synth-rock reading. |
| `GENRE_CHECK_IN_SAVED_ANALYSIS` | `True` | Include the genre-check warning at the top of saved analysis text. |


## Chords, melody, key and tempo checks

| Variable | Default | Description |
|---|---|---|
| `ENABLE_CHORD_PROGRESSION` | `True` | Run optional `lv-chordia` full-mix chord recognition and build per-section progressions; skipped if unavailable. |
| `CHORD_PROGRESSION_DICTIONARY` | `"submission"` | Chord vocabulary: `"submission"`, `"ismir2017"` or `"full"`. |
| `CHORD_PROGRESSION_IN_SAVED` | `True` | Rebuild a missing chord-progression block on load from saved chord segments. |
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
| `WRITER_RULES_DEV45` | `True` | Enable writer rules for style, regional genre, era, chord-progression and melody evidence blocks. |
| `FOLLOWUP_QUESTION_NOTES` | `True` | Apply relevant question-specific evidence notes to single-track follow-up questions. |
| `SOUND_SOURCE_NOTE` | `True` | Require sound descriptions to follow measured track evidence while allowing artist and release background. |
| `BEAT_DRAWING_NOTE` | `True` | For an unspecified drum-drawing request, show grids for parts with different grooves. |
| `BEAT_DRAWING_MAX` | `4` | Maximum differing part grids requested by the beat-drawing note. |
| `LOAD_CHECKS_TO_CHAT` | `True` | Send load-time corrected analysis to the chat model. |
| `STRUCTURE_SAVE_REPLAY` | `True` | Save inputs used to replay the structure map’s final stage. |

## Configurable command names

The slash-command spellings above can be changed in `USER CONFIGURATION`:

| Variable | Default |
|---|---|
| `LISTEN_FLAG` | `"/listen"` |
| `RELISTEN_FLAG` | `"/relisten"` |
| `CORRECT_FLAG` | `"/correct"` |
| `SAVE_FLAG` | `"/save"` |
| `LOAD_FLAG` | `"/load"` |
| `LOADCOMPARE_FLAG` | `"/loadcompare"` |
| `CLEAR_FLAG` | `"/clear"` |
| `CLEAR_ALL_FLAG` | `"/clearall"` |
| `BATCH_FLAG` | `"/batch"` |
| `PERSONA_FLAG` | `"/persona"` |
| `DEBUG_FLAG` | `"/debug"` |

## Environment overrides

These are actual environment variables, unlike the Python constants above. Set them before starting the script.

| Variable | Script fallback | Description |
|---|---|---|
| `MOSS_MUSIC_CODE_DIR` | Search beside the script, then `~/MOSS-Music` | Repository location when the Python constant of the same name is `None`. A configured Python path takes priority. |
| `PYTORCH_MPS_HIGH_WATERMARK_RATIO` | `"0.0"` | PyTorch MPS allocator upper watermark; this fallback removes its upper allocation limit. An existing environment value wins. |
| `PYTORCH_MPS_LOW_WATERMARK_RATIO` | `"0.6"` | MPS allocator cleanup watermark. An existing environment value wins. |

## Changed since 0.5

| Previous name | 0.6 name |
|---|---|
| `MF_MODEL_ID` | `MOSS_MODEL_ID` — now `OpenMOSS-Team/MOSS-Music-8B-Instruct` |
| `MF_TORCH_DTYPE` | `MOSS_TORCH_DTYPE` |
| `MF_EMPTY_RETRY_SYSTEM_PROMPT` | `MOSS_EMPTY_RETRY_SYSTEM_PROMPT` |
| `MF_EMPTY_RETRY_ADD_DESCRIPTION` | `MOSS_EMPTY_RETRY_ADD_DESCRIPTION` |
| `MF_LOOP_GUARD` | `MOSS_LOOP_GUARD` |
| `MF_CONSISTENT_REPORT` | `MOSS_CONSISTENT_REPORT` |
| `SKIP_MF_LYRICS_WHEN_TAGS_PRESENT` | `SKIP_LYRICS_TRANSCRIPTION_WHEN_TAGS_PRESENT` |

The listening-model change also requires the MOSS-Music model code; changing the model ID alone is not sufficient. Earlier saved JSONs remain readable through the script’s compatibility handling.

---

**Scope:** This remains a practical configuration reference, not an exhaustive dump of every tuning constant. Additional `STRUCTURE_*` rules, instrument-probe thresholds, vocal/genre safeguards and algorithm thresholds are documented beside their assignments in `USER CONFIGURATION`. Runtime state, lookup tables and full prompt bodies are omitted. `DEFAULT_PERSONA_PROMPT` is defined elsewhere in the script; use `/persona` for routine persona changes.


# Possible future features:

These are a few things that I want to implement in the future but so far haven't done

* Separate the configuration flags from the main script in a different file
* Most of the project was vibecoded with the help of several LLMs, and while the current build appears to be fully usable, those with better coding knowledge could probably help patch any issues.
* Further improved timestamp labelling
* Further improved estimated era accuracy for songs without a year tag




![Musiclyse - A Gourlish Vibe Project](https://pbs.twimg.com/media/HQH-IMYWoAAX-p6.jpg)
