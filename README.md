# vocaloid

Voice cloning using [Qwen3-TTS-1.7B-Base](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base).

## Setup

Requires [Nix](https://nixos.org) with flakes enabled.

```bash
git add .
nix develop
```

## Voice samples

Place `.wav` files (and optionally `.txt` transcripts) anywhere under `voices/<name>/`:

```
voices/
  isaac/
    sample.wav
    sample.txt
  shayden/
    samples/
      sample.wav
      sample2.wav
```

Multiple samples are supported and will all be used for cloning. Transcripts are auto-generated on first run (Apple Silicon only) and saved alongside the audio.

If your sample is in another format (m4a, opus, etc.), convert it first:

```bash
ffmpeg -i sample.m4a -ar 16000 -ac 1 sample.wav
```

## Clone a voice

```bash
uv run clone.py --voice isaac --text "Text to synthesise in the cloned voice."
uv run clone.py --voice isaac --text "..." --language English
```

Outputs are saved to `voices/<name>/out/output-1.wav`, incrementing on each run.

Runs on CUDA, MPS (Apple Silicon), or CPU automatically.
