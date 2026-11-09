# Musiclyse-0.6.2-dev16-ollama-override.py -- optional; keep it in the same
# folder as Musiclyse-0.6.2-dev16.py. It belongs to that build only: each build
# reads the file named after it ("-ollama-override.py" in place of ".py"), so
# for another build, copy this file and rename it to match. Delete or rename it
# to use the main script's own setting.
#
# REMINDER -- to start Ollama yourself with the same optimisations, in Terminal:
#   alias ollama-serve='OLLAMA_FLASH_ATTENTION=1 LLAMA_ARG_CACHE_RAM=0 LLAMA_ARG_CTX_CHECKPOINTS=4 ollama serve'
#   ollama-serve
# (Type the alias line once per Terminal window, or add it to ~/.zshrc to keep it.)
#
# What Musiclyse does when no Ollama is running at startup. This file's value
# replaces OLLAMA_LAUNCH_MODE in the main script. Musiclyse only reads the
# value below; it never runs this file.
#   "off"             you start Ollama yourself (for example with ollama-serve above)
#   "background"      hidden; nothing written to disk; stopped when Musiclyse ends
#   "background_log"  hidden, with a log in logs/ollama.log next to the script
#   "log_window"      hidden, with that log shown live in its own Terminal window;
#                     Ollama stops and the window closes when Musiclyse ends
#   "window"          Ollama in its own Terminal window, like ollama-serve above;
#                     left running after Musiclyse ends (close it to stop it)
# An Ollama that is already running is always used as it is.
#
# Where the "log_window" window opens: "behind" Musiclyse's window, or
# "minimized" straight into the Dock (click it there to watch the log).

OLLAMA_LAUNCH_MODE = "background"
OLLAMA_LOG_WINDOW = "behind"
