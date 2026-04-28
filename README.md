# Vocaloid

Voice cloning using [Qwen3-TTS-1.7B-Base](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base).

This is just messing around for fun, the whole thing was written by Claude 
over like 90 minutes of me giggling while bullying any friends who send voice 
notes.

## Setup

It's set up to use [Nix](https://nixos.org) with flakes enabled. See the flake 
for a list of dependencies if you're installing them some other way. Otherwise 
use `nix develop` to get everything set up. Works on macOS and Linux.

## Voice samples

Place `.wav` files (and optionally `.txt` transcripts) anywhere under `voices/<name>/`:

```
voices/
  victim/
    samples/
      sample1.wav
      sample1.txt
```

Multiple samples are supported and will all be used for cloning. Transcripts are auto-generated on first run and saved alongside the audio — using mlx-whisper on Apple Silicon or faster-whisper elsewhere.

If your sample is in another format (m4a, opus, etc.), convert it first:

```bash
ffmpeg -i sample.m4a -ar 16000 -ac 1 sample.wav
```

## Clone a voice

```bash
uv run clone.py --voice victim --text "Text to synthesise in the cloned voice."
uv run clone.py --voice isaac --text "..." --language English
```

Outputs are saved to `voices/<name>/out/output-1.wav`, incrementing on each run.
