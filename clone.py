import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import soundfile as sf
from rich.console import Console

# qwen_tts unconditionally prints a flash-attn warning on every import if
# flash-attn isn't installed (whisper_encoder.py, line 35 — a bare print(),
# not a logging call, so logging filters won't catch it). flash-attn requires
# CUDA to build and isn't available on MPS, so we can't just install it.
#
# We suppress it by briefly redirecting stdout to /dev/null for the duration
# of the import. This is the least invasive option short of patching the venv,
# which would be wiped on the next `uv sync`. Nothing else prints to stdout
# during this import, so nothing legitimate is lost.
with open(os.devnull, "w") as _devnull:
    sys.stdout = _devnull
    from qwen_tts import Qwen3TTSModel
    sys.stdout = sys.__stdout__

# transformers warns about pad_token_id defaulting to eos_token_id during
# open-ended generation. This is expected behaviour for this model and not
# actionable, so we silence that specific logger.
import logging
logging.getLogger("transformers.generation.utils").setLevel(logging.ERROR)

MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
VOICES_DIR = Path(__file__).parent / "voices"
console = Console()


def get_device():
    if torch.cuda.is_available():
        return "cuda:0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model():
    device = get_device()
    console.print(f"Device: [bold]{device}[/bold]")
    kwargs = dict(
        device_map=device,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2" if device.startswith("cuda") else "eager",
    )
    try:
        with console.status("Loading model..."):
            return Qwen3TTSModel.from_pretrained(MODEL, local_files_only=True, **kwargs)
    except Exception:
        with console.status("Downloading model..."):
            return Qwen3TTSModel.from_pretrained(MODEL, **kwargs)


def transcribe(wav_path: Path) -> str:
    console.print(f"  Transcribing [dim]{wav_path.name}[/dim]...")
    try:
        import mlx_whisper
        result = mlx_whisper.transcribe(
            str(wav_path),
            path_or_hf_repo="mlx-community/whisper-large-v3-turbo",
        )
        return result["text"].strip()
    except ImportError:
        console.print("[red]mlx-whisper not available.[/red] Only supported on Apple Silicon.")
        raise SystemExit(1)


def load_samples(voice_dir: Path) -> tuple[tuple[np.ndarray, int], str]:
    wavs = sorted(p for p in voice_dir.rglob("*.wav") if "out" not in p.parts)
    if not wavs:
        console.print(f"[red]No .wav files found in {voice_dir}[/red]")
        raise SystemExit(1)

    # The model's voice_clone_prompt API accepts either a single sample or a
    # batch (one prompt per output). It does not support combining multiple
    # samples into a single voice identity via the prompt list.
    #
    # Instead, we concatenate all reference clips into one longer audio array.
    # More context generally helps the model capture the speaker's character,
    # and a single concatenated clip stays within the single-prompt path.
    clips, texts, sr = [], [], None
    for wav in wavs:
        txt = wav.with_suffix(".txt")
        if not txt.exists():
            transcript = transcribe(wav)
            txt.write_text(transcript)
        else:
            transcript = txt.read_text().strip()
        data, file_sr = sf.read(str(wav))
        if sr is None:
            sr = file_sr
        elif file_sr != sr:
            raise ValueError(f"Sample rate mismatch: {wav} is {file_sr}Hz, expected {sr}Hz")
        clips.append(data)
        texts.append(transcript)
        console.print(f"  [green]✓[/green] {wav.relative_to(voice_dir)}")

    combined_audio = np.concatenate(clips)

    # Real recordings often have peak values slightly above 1.0 due to
    # floating-point conversion or minor clipping at capture time. The model
    # warns (via a bare print) when it sees out-of-range values. Peak-normalise
    # here so the signal is clean before it reaches the model.
    peak = np.max(np.abs(combined_audio))
    if peak > 1.0:
        combined_audio = combined_audio / peak

    combined_text = " ".join(texts)
    return (combined_audio, sr), combined_text


def next_output_path(out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(out_dir.glob("output-*.wav"))
    n = len(existing) + 1
    return out_dir / f"output-{n}.wav"


def main():
    parser = argparse.ArgumentParser(description="Clone a voice using Qwen3-TTS Base")
    parser.add_argument("--voice", required=True, help="Voice name (folder under voices/)")
    parser.add_argument("--text", required=True, help="Text to synthesise in the cloned voice")
    parser.add_argument("--language", default="English", help="Output language (default: English)")
    args = parser.parse_args()

    voice_dir = VOICES_DIR / args.voice
    if not voice_dir.exists():
        console.print(f"[red]Voice folder not found: {voice_dir}[/red]")
        raise SystemExit(1)

    console.print(f"Loading samples for [bold]{args.voice}[/bold]...")
    ref_audio, ref_text = load_samples(voice_dir)

    model = load_model()

    with console.status("Processing reference audio..."):
        voice_prompt = model.create_voice_clone_prompt(
            ref_audio=ref_audio,
            ref_text=ref_text,
        )

    with console.status("Synthesising..."):
        wavs, sr = model.generate_voice_clone(
            text=args.text,
            language=args.language,
            voice_clone_prompt=voice_prompt,
        )

    out_path = next_output_path(voice_dir / "out")
    sf.write(str(out_path), wavs[0], sr)
    out_path.with_suffix(".txt").write_text(args.text)
    console.print(f"Saved to [bold]{out_path}[/bold]")


if __name__ == "__main__":
    main()
