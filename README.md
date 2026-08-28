# Musiclyse
A highly configurable piece of local Ollama-based chatbot software that can "listen" to music via a multi-step analysis engine, compare tracks and more.

![Musiclyse - Local Music Discussion](https://gourlish.altervista.org/musiclyse/musiclyselogo.png)

# About:

Musiclyse is a Python script designed for high-spec Macs (it is easily portable to good quality Linux PCs) that enables local conversations about music. It utilises a multi-step song processing pipeline linked to an Ollama hosted local LLM of the user's choice. This consists of a pipeline process of Music Flamingo, Essentia, Demucs separation fed to Omnizart and metadata analysis (which will improve recognition for lyrics). You're welcome to help improve the project to make its song recognition better!

# Key features:

The following features are all included in Musiclyse 0.3:

* Beat, melody, genre, vocal, song key, scale and era analysis. There are still some minor accuracy issues which I am hoping to improve.
* Can get accurate facts about music adjacent topics via an optional Wikipedia-based database add-on file.
* Will talk about the concept of the songs or compare and contrast two or more different songs. You can ask it anything you can ask a normal chatbot, and it will be able to bring songs into context with the chat.
* Supports image analysis (if used with an image-supporting model).
* LLMs can be swapped out easily in the script. Currently it is optimised for use with Muse Glimmer 30B; for less powerful systems Gemma 4 26B is recommended.
* Save your analysed song data to JSON files, then load them quickly in a new session.
* Run an overnight batch of songs to JSON files that can be fast loaded at a later time.
* Create your own personas and hear their opinions about songs.
* Import two tracks together to compare them easily.
* Specify a range (x seconds to y seconds) in a track to analyse for faster processing and more precise analysis.

## Version history:

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

# Installation for 0.3 (for MacOS):

1: Ensure Python 3.10 is installed on your system

2: Clone the provided files onto your machine for the latest version, or alternatively download a version from the Releases section.

3: [Download the heavy version of the Musiclyse Wikipedia Database via this link](https://github.com/Gourlish/Musiclyse-Wikipedia-Database/releases/tag/music_wiki_heavy-august2026). Alternatively you can [download a lite version](https://github.com/Gourlish/Musiclyse-Wikipedia-Database/releases/tag/music_wiki_lite-august2026) that covers fewer topics but consumes less storage (it will run without either, but it's likely to give far better context about artists if you install it). Once extracted (the heavy database should be listed as 8.33GB on MacOS, or a bit less on Windows and many distributions of Linux) place the music_wiki_heavy.db file in the same directory as the Musiclyse Python script. If you are using the Lite version of the database, you will need to search the script for mentions of "music_wiki_heavy.db" and replace this with "music_wiki_lite.db".

3: Install required system level dependencies (if you don't already have Homebrew installed you'll need it too)

```
brew install ollama ffmpeg portaudio
```

4: In a new terminal window, start Ollama and install Muse Glimmer (if you want a different model for output, change the specification in musiclyse-0.1.py via the "OLLAMA_MODEL =" flag to the exact Ollama name of the model of your choice, and pull that different model with Ollama). By default the model is configured to use the MLX version of Muse Glimmer, though this will only work on a Mac.

```
ollama serve
ollama pull muse-glimmer:30b-mlx
```

5: Return to your PREVIOUS prompt, leaving the Ollama one open (once the model has installed and the server is running) since the script will not run without an Ollama server running, and create a virtual Python 3.10 folder in the folder with the git clone you made:
```
python3.10 -m venv ./venv
source venv/bin/activate
```

6: Install the requirements:

```
pip install -r requirements.txt
```

7: Install Music Flamingo:

```
pip install huggingface_hub
hf download nvidia/music-flamingo-hf
```

8: Download Omnizart models:

```
omnizart download-checkpoints
```

9: Run the software:

```
python musiclyse-0.3.py
```

# Configurable Variables & Flags in 0.3

## Command-line

| Usage | Description |
|---|---|
| `python musiclyse.py` | No starting track — pick one with `/listen` |
| `python musiclyse.py /path/to/song.m4a` | Optional starting track |
| `python musiclyse.py --batch-one /path/to/song.mp3` | Internal: isolated single-track batch worker (used by `/batch`) |
| `python musiclyse.py --batch-one-analyze /path/to/song.mp3 /tmp/staging.json` | Internal: heavy analysis stage (split-process batch) |
| `python musiclyse.py --batch-one-finish /tmp/staging.json` | Internal: identity/save stage (split-process batch) |

## In-chat slash commands

| Command | Description |
|---|---|
| `/listen <question>` | Analyse the current track |
| `/listen <path or URL> <question>` | Switch current track and analyse it |
| `/relisten` | Force a fresh full analysis even if one is cached |
| `/correct field=value[, field=value…]` | Record a user-confirmed fact that overrides the model's perception (e.g. `year=1966`, `singer=post_puberty_male`) |
| `/save=filename.json` | Save technical details for the most recently scanned track |
| `/load filename.json [question]` | Load a previously saved track; optional question after the filename |
| `/loadcompare "track1.json" "track2.json" question` | Load two or more saved tracks and ask a comparison question about all of them |
| `/clear` | Wipe chat context + token counters (analysis cache kept) |
| `/clearall` | Wipe chat + in‑memory track analysis caches (saved‑song files on disk are kept) |
| `/batch /path/to/folder` | Overnight scan every audio file in a folder into `saved-songs/` (does not import into chat) |
| `/persona <description>` | Switch chat voice/taste (music evidence rules stay) |
| `/persona reset` | Restore the default music‑obsessed friend persona |
| `/debug` | Show the last writer message (debug) |

## Console / display

| Variable | Default | Description |
|---|---|---|
| `FRIENDLY_PROCESSING_DISPLAY` | `True` | Progress‑bar‑style overwrite vs. old scrolling log |
| `USE_COLOR` | *(auto)* | Set from `sys.stdout.isatty()` |
| `SHOW_RAW_ANALYSIS` | `False` | Print Music Flamingo's full raw analysis (debugging) |
| `DEBUG_MODEL_LOADING` | `False` | Show raw model‑loading logs instead of a status line |
| `SHOW_OMNIZART_LOGS` | `False` | Let Omnizart's stdout/stderr through instead of suppressing it |
| `DEBUG_STEM_MIDI` | `False` | Verbose per‑stem note‑count debugging |
| `SHOW_LAST_WRITER_MESSAGE_ON_DEBUG` | `False` | Show the last writer message when `/debug` is used |

## Ollama / model

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_URL` | `http://localhost:11434/api/chat` | Ollama chat endpoint |
| `OLLAMA_MODEL` | `muse-glimmer:30b-mlx` | Allows you to set your model |
| `OLLAMA_NUM_CTX` | `131072` | Context window size |
| `OLLAMA_SUPPORTS_IMAGES` | `True` | Whether the chosen model accepts image input |
| `MF_MODEL_ID` | `nvidia/music-flamingo-hf` | Music Flamingo model ID |
| `MF_TORCH_DTYPE` | `torch.bfloat16` | Torch dtype for Music Flamingo |
| `DEMUCS_MODEL` | `htdemucs_6s` | Demucs stem‑separation model |
| `DEMUCS_SHIFTS_FAST` | `0` | Test‑time shift ensembling in fast mode (0 = off) |
| `DEMUCS_SHIFTS_DEEP` | `2` | Test‑time shift ensembling in deep mode (2 = extra shifts) |

## Analysis pipeline toggles (speed vs. depth)

| Variable | Default | Description |
|---|---|---|
| `FAST_MODE` | `False` | Safe speedup: skip self‑check pass (one fewer MF generation) |
| `DEEP_MODE` | `False` | Larger token budgets + two extra analysis categories |
| `ENABLE_OBJECTIVE_AUDIO_REPORT` | `True` | |
| `ENABLE_VOCAL_PASS` | `True` | |
| `ENABLE_VOCAL_OBJECTIVE_REPORT` | `True` | |
| `ENABLE_VOCAL_CONFIRMATION_PASS` | `True` | |
| `ENABLE_ESSENTIA_REPORT` | `True` | |
| `ENABLE_STEM_MIDI` | `True` | |
| `ENABLE_COVER_ART_DESCRIPTION` | `True` | |
| `ENABLE_SINGER_IDENTITY_RESOLUTION` | `True` | |
| `ENABLE_IMAGE_OBSERVATIONS_FOR_GENERAL` | `True` | |
| `ENABLE_WIKI_CONTEXT` | `True` | |
| `SKIP_MF_LYRICS_WHEN_TAGS_PRESENT` | `True` | |
| `COMPACT_OBJECTIVE_REPORT` | `True` | |
| `STEM_MIDI_INCLUDE_EVENT_LOGS` | `False` | |
| `STEM_MIDI_COMPACT_EVENT_FORMAT` | `True` | |
| `UNLOAD_OMNIZART_AFTER_STEM_MIDI` | `True` | |
| `ENABLE_INSTRUMENT_TAGGING` | `True` | Per‑stem instrument tagging (requires `panns-inference`) |
| `ENABLE_WHOLE_MIX_INSTRUMENT_TAGGING` | `True` | Instrument tagging on the original mix (independent of Demucs) |
| `ENABLE_GENRE_MOOD_TAGGING` | `True` | Genre/mood cross‑check using the same PANNs classifier |

## Instrument‑tagging thresholds (PANNs / AudioSet)

| Variable | Default | Description |
|---|---|---|
| `INSTRUMENT_TAG_MIN_PROB` | `0.18` | Minimum probability for most instrument labels |
| `INSTRUMENT_TAG_TOP_K` | `4` | Maximum number of instrument tags to keep per stem |
| `INSTRUMENT_TAG_GUITAR_MIN_PROB` | `0.35` | Higher threshold for guitar‑family labels (noisy) |
| `INSTRUMENT_TAG_GUITAR_LABELS` | `("electric guitar", …)` | Labels considered "guitar" for thresholding |
| `INSTRUMENT_TAG_STEMS` | `("other", "guitar", "piano", "bass")` | Which stems get instrument tags |
| `INSTRUMENT_TAG_REQUIRE_STEM_ACTIVITY` | `True` | Drop tags on near‑empty stems unless very strong |
| `INSTRUMENT_TAG_MIN_NOTES_FOR_WEAK` | `8` | Minimum note count before accepting weak tags |
| `INSTRUMENT_TAG_STRONG_PROB` | `0.45` | Probability above which stem‑activity checks are relaxed |
| `INSTRUMENT_TAG_WINDOW_SECONDS` | `10.0` | Window length for windowed tagging (None = whole clip) |
| `INSTRUMENT_TAG_HOP_SECONDS` | `5.0` | Hop between windows |
| `INSTRUMENT_TAG_MAX_WINDOWS` | `40` | Maximum number of windows per clip |
| `INSTRUMENT_TAG_NORMALIZE` | `True` | Peak‑normalize each window before tagging |
| `INSTRUMENT_TAG_NORMALIZE_PEAK` | `0.95` | Target peak after normalisation |
| `WHOLE_MIX_INSTRUMENT_TAG_TOP_K` | `6` | |
| `WHOLE_MIX_INSTRUMENT_TAG_MIN_PROB` | `0.15` | |
| `INSTRUMENT_TAG_MIX_AGREE_MIN_PROB` | `0.18` | Minimum whole‑mix probability to consider a stem tag supported |

## Genre/mood tagging (PANNs)

| Variable | Default | Description |
|---|---|---|
| `GENRE_TAG_TOP_K` | `5` | |
| `GENRE_TAG_MIN_PROB` | `0.05` | |
| `MOOD_TAG_TOP_K` | `3` | |
| `MOOD_TAG_MIN_PROB` | `0.08` | |

## Essentia / audio‑analysis limits

| Variable | Default | Description |
|---|---|---|
| `ESSENTIA_MAX_SECONDS` | `300.0` | |
| `ESSENTIA_LOWLEVEL_MAX_SECONDS` | `60.0` | |
| `ESSENTIA_FRAME_SIZE` | `4096` | |
| `ESSENTIA_HOP_SIZE` | `2048` | |

## Stem/MIDI limits

| Variable | Default | Description |
|---|---|---|
| `STEM_MIDI_MAX_SECONDS` | `None` | `None` = full‑track MIDI; set a number to cap for speed |
| `STEM_MIDI_EVENT_LOG_MAX_NOTES` | `24` | |
| `STEM_MIDI_MELODY_LINE_NOTES` | `24` | |
| `STEM_MIDI_DRUM_PATTERN_HITS` | `24` | |

## Vocal‑classification thresholds

| Variable | Default | Description |
|---|---|---|
| `VOCAL_CONFIRMATION_F0_THRESHOLD` | `210.0` | |
| `VOCAL_CONFIRMATION_WITHOUT_F0` | `True` | |
| `F0_MALE_HIGH_CONFIRM_HZ` | `260.0` | |
| `F0_POST_PUBERTY_SOFT_CAP_HZ` | `280.0` | |
| `F0_POST_PUBERTY_HARD_CAP_HZ` | `320.0` | |
| `F0_MALE_ANY_HARD_CAP_HZ` | `320.0` | |
| `F0_CHILD_SOFT_FLOOR_HZ` | `180.0` | |
| `METADATA_LYRICS_MIN_CHARS_TO_SKIP_MF` | `80` | |

## Metadata / cover art

| Variable | Default | Description |
|---|---|---|
| `ENABLE_FILE_METADATA` | `True` | |
| `SEND_COVER_ART_TO_OLLAMA` | `True` | |
| `METADATA_MAX_LYRICS_CHARS` | `10000` | |
| `COVER_IMAGE_MAX_DIMENSION` | `512` | |
| `COVER_IMAGE_QUALITY` | `"2"` | (valid mjpeg qscale, lower = better quality) |
| `MAX_COVER_IMAGES_PER_REQUEST` | `1` | |
| `MAX_COVER_BYTES_TO_SEND_RAW` | `1_000_000` | |
| `MAX_EXPLICIT_IMAGE_BYTES` | `2_000_000` | |
| `MAX_IMAGES_PER_REQUEST` | `8` | |
| `MAX_IMAGES_TO_DESCRIBE` | `2` | |
| `COVER_ART_DESCRIPTION_NUM_CTX` | `8192` | |
| `SINGER_IDENTITY_NUM_CTX` | `8192` | |

## Chat history / context budget

| Variable | Default | Description |
|---|---|---|
| `MAX_WRITER_IMAGES_PER_TURN` | `8` | Images actually sent to Ollama in one request |
| `MAX_STORED_IMAGES_IN_HISTORY` | `8` | Base64 images kept in `writer_history` |
| `MAX_WRITER_HISTORY_MESSAGES` | `80` | |
| `MAX_MESSAGE_CHARS_FOR_OLLAMA` | `60_000` | |
| `HISTORY_CHAR_BUDGET_FACTOR` | `3.2` | Conservative chars‑per‑token estimate for trimming |

## Wikipedia context

| Variable | Default | Description |
|---|---|---|
| `ENABLE_WIKI_CONTEXT` | `True` | |
| `WIKI_DB_PATH` | `music_wiki_heavy.db` | |
| `WIKI_DB_TIMEOUT_S` | `5.0` | |
| `WIKI_CONTEXT_MAX_CHARS_PER_ARTICLE` | `1800` | Per‑article cap — enough for lead + a track listing or reception section |
| `WIKI_CONTEXT_TOTAL_MAX_CHARS` | `5000` | Hard ceiling on the whole block regardless of article count |
| `WIKI_MAX_ARTICLES` | `6` | e.g. song + album + artist article, combined |
| `WIKI_SEARCH_ROW_LIMIT` | `8` | Slightly wider FTS window so (album) disambiguation can rank |
| `WIKI_CONTEXT_REFRESH_EVERY_QUESTION` | `False` | |
| `WIKI_SEARCH_EVERY_MESSAGE` | `True` | |
| `WIKI_DEBUG_LOG` | `True` | Print a one‑line status after every general‑chat question showing whether the wiki DB returned a match |

## Batch mode

| Variable | Default | Description |
|---|---|---|
| `BATCH_PAUSE_BETWEEN_TRACKS_S` | `20.0` | Pause between tracks during `/batch` to let memory settle |
| `BATCH_ISOLATE_PER_TRACK` | `True` | Run each track in a fresh Python subprocess (recommended on macOS) |
| `BATCH_SPLIT_IDENTITY_PROCESS` | `True` | Split heavy analysis and identity/save into separate processes (avoids OOM) |
| `BATCH_LOG_RSS` | `True` | Log approximate process RSS after cleanup |
| `BATCH_UNLOAD_OLLAMA` | `True` | Unload the Ollama writer model during heavy analysis stages |
| `BATCH_OLLAMA_KEEP_ALIVE` | `0` | `keep_alive` for short batch helper requests (0 = unload immediately) |
| `BATCH_PRE_IDENTITY_SETTLE_S` | `5.0` | Settle time before identity stage (reclaims RAM) |

## Persona / prompts

| Variable | Default | Description |
|---|---|---|
| `DEFAULT_PERSONA_PROMPT` | *(see script)* | Swappable at runtime via `/persona` |
| `SAVE_DIR` | `./saved-songs` | Directory where saved tracks are stored |

---

**Not included:** internal data structures such as `NOTE_NAMES`, `GM_DRUM_PITCH_TYPE`, `STEM_MIDI_STEMS`, `OMNIZART_STEM_APP`, the vocal‑category tag sets, and the prompt template bodies themselves (e.g. `MF_FULL_ANALYSIS_PROMPT`) — these are technically editable but function as implementation detail rather than user‑facing settings.

When the context gets over the user-specified token context window, it will forget context from outside of it.

# Possible future features:

These are a few things that I want to implement in the future but so far haven't done

* Separate the configuration flags from the main script in a different file
* Improved melody detection. I've had to limit the amount of MIDI data the JSON file stores to avoid excessive token bloat (it's still relatively high as is), and might look at better algorithms for which it can capture the melody of songs to "hear" the music.
* Most of the project was vibecoded with the help of several LLMs, and while the current build appears to be fully usable, those with better coding knowledge could probably help patch any issues.
* Implement improved AI lyric transcription, possibly via a Whisper model
* Improved estimated era accuracy for songs without a year tag




![Musiclyse - A Gourlish Vibe Project](https://pbs.twimg.com/media/HQH-IMYWoAAX-p6.jpg)
