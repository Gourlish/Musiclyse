# Musiclyse
An local LLM music-listening client that communicates between Ollama and a multi-step music analysis engine to "hear" music and discuss songs.

![Musiclyse - Local Music Discussion](https://gourlish.altervista.org/musiclyse/musiclyselogo.png)

## About:

Musiclyse is a Python script that enables local conversations about music. It utilises a multi-step song processing engine linked to an Ollama hosted local LLM of the user's choice. This consists of a pipeline process of Music Flamingo, Essentia, Demucs separation fed to Omnizart and metadata analysis (which will improve recognition for lyrics). You're welcome to help improve the project to make its song recognition better!

## Version history:

* 2026/08/22 - Musiclyse 0.1 released. Its core features are listed in the "key features" section.

## System requirements

Due to the nature of what it's doing, it's intended to be run on powerful machines with a good quality GPU and 64GB RAM or higher. Currently it's only been tested on MacOS, but it should be pretty easy to get running on Linux, and is potentially runnable on Windows. For users with less powerful machines, I'd advise swapping Muse Glimmer for Google's Gemma 4 26B A4B, which has reasonable knowledge but is not as resource heavy. Users with lower spec machines might benefit from installing a 4-8B model via Ollama. You can swap out the Ollama model by altering the "

## Key features:

The following features are all included in Musiclyse 0.1:

* Beat, melody, genre, vocal, song key, scale and era analysis. There are currently some minor accuracy issues which I am hoping to improve.
* Will talk about the concept of the songs or compare and contrast two or more different songs. You can ask it anything you can ask a normal chatbot, and it will be able to bring songs into context with the chat.
* Supports image analysis (if used with an image-supporting model)
* LLMs can be swapped out easily in the script. Currently it is optimised for use with Muse Glimmer 30B; for less powerful systems Gemma 4 26B is recommended
* Save your analysed song data to JSON files, then load them quickly in a new session
* Run an overnight batch of songs to JSON files that can be fast loaded at a later time
* Create your own personas and hear their opinions about songs

## Installation (for MacOS):

1: Ensure Python 3.10 is installed on your system

2: Clone the provided files onto your machine

3: Install required system level dependencies (if you don't already have Homebrew installed you'll need it too)

```
brew install ollama ffmpeg
```

4: In a new terminal window, start Ollama and install Muse Glimmer (if you want a different model for output, change the specification in musiclyse-0.1.py via the "OLLAMA_MODEL =" flag to the exact Ollama name of the model of your choice, and pull that different model with Ollama)

```
ollama serve
ollama pull muse-glimmer:30b
```

5: Return to your PREVIOUS prompt, leaving the Ollama one open (once the model has installed and the server is running) and create a virtual Python 3.10 folder in the folder with the git clone you made:
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
huggingface-cli download nvidia/music-flamingo-hf
```

8: Run the software:

```
python musiclyse-0.1.py
```

## Usage:

* ```/listen <question>``` -	Analyzes the current track and answers your question
* ```/listen <path or URL> <question>``` -		Switches to a new track, then analyses it
* ```/save=filename.json``` -		Saves the most recently analysed track's data
* ```/load filename.json [question]``` -		Reloads a previously saved track
* ```/batch /path/to/folder``` -		Scans every audio file in a folder into saved-songs/
* ```/clear``` -		Wipes the chat history and resets token counters
* ```/persona <description>``` -		Changes the chat voice/tone (analysis rules stay the same)
* ```/persona reset``` -		Restores the default persona

When the context gets over the user-specified token context window, it will forget context from outside of it.

## Possible future features:

These are a few things that I want to implement in the future but so far haven't done

* Separate the configuration flags from the main script in a different file
* A Wikipedia-based database about music adjacent topics to expand the local LLM's knowledge; this will likely add an approximately 1GB overhead, but will be an optional feature
* Improved BPM recognition. The BPM recognition in 0.1 isn't bad but often off by 1-2 values
* Improved melody detection. I've had to limit the amount of MIDI data the JSON file stores to avoid excessive token bloat (it's still relatively high as is), and might look at better algorithms for which it can capture the melody of songs to "hear" the music.
* *Most of the project was vibecoded with the help of several LLMs, and while the current build appears to be fully usable, those with better coding knowledge could probably help patch any issues.




![Musiclyse - A Gourlish Vibe Project](https://pbs.twimg.com/media/HQH-IMYWoAAX-p6.jpg)
*A Gourlish Vibe Project!*
