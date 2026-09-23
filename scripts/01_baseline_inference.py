"""
01_baseline_inference.py: Level 1 Baseline ASR Inference
Loads bodhan-ai/indic-transcribe-core from the verified local snapshot,
runs zero-shot transcription on a gold Marathi audio sample from ai4bharat/Kathbath,
and exports outputs/baseline_sample.json.
"""

import os
import sys
import glob
import json
import torch
import soundfile as sf

# Force UTF-8 stdout on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def find_local_model_dir():
    """Locates the cached model snapshot directory without network calls."""
    pattern = os.path.expanduser("~/.cache/huggingface/hub/models--bodhan-ai--indic-transcribe-core/snapshots/*")
    matches = glob.glob(pattern)
    for m in matches:
        if os.path.exists(os.path.join(m, "model.safetensors")):
            return m
    raise FileNotFoundError("Local snapshot with model.safetensors not found in Hugging Face cache.")


def run_baseline():
    print("=" * 65)
    print("LEVEL 1: BODHAN INDIC-TRANSCRIBE-CORE BASELINE INFERENCE")
    print("=" * 65)

    audio_path = "data/sample_marathi.wav"
    ground_truth = "अबोटाबाद येथे ओसामा बिन लादेनच्या घरात शिरून त्यांनी ओसामाचा खात्मा केला"

    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file {audio_path} not found.")

    info = sf.info(audio_path)
    print(f"Sample Audio    : {audio_path}")
    print(f"Duration        : {info.duration:.2f} s")
    print(f"Sample Rate     : {info.samplerate} Hz (Channels: {info.channels})")
    print(f"Ground Truth    : {ground_truth}")

    # 1. Locate verified local model directory
    model_dir = find_local_model_dir()
    print(f"\nUsing local model directory: {model_dir}")

    # 2. Add model dir to sys.path
    if model_dir not in sys.path:
        sys.path.insert(0, model_dir)

    # 3. Import official Bodhan interface
    from indic_transcribe import IndicTranscribe

    # 4. Check hardware capabilities
    has_cuda = torch.cuda.is_available()
    device = "cuda" if has_cuda else "cpu"
    print(f"\nExecution device: {device.upper()}")
    if has_cuda:
        vram_mb = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
        print(f"GPU: {torch.cuda.get_device_name(0)} ({vram_mb:.0f} MiB VRAM)")

    # 5. Load model with fallback to CPU if GPU memory allocation fails
    try:
        print("\nLoading model via IndicTranscribe.from_pretrained()...")
        asr = IndicTranscribe.from_pretrained(model_dir, device=device)
        print("[SUCCESS] Model loaded into memory successfully!")
    except (torch.cuda.OutOfMemoryError, RuntimeError) as err:
        print(f"[WARNING] Loading on GPU hit an issue: {err}")
        print("Falling back to CPU for baseline inference...")
        torch.cuda.empty_cache()
        asr = IndicTranscribe.from_pretrained(model_dir, device="cpu")
        device = "cpu"

    # 6. Execute transcription on Marathi audio
    print(f"\nTranscribing Marathi audio '{audio_path}' (lang='mr')...")
    transcription = asr(audio_path, lang="mr")
    print("\n" + "=" * 65)
    print(f"Reference (Ground Truth) : {ground_truth}")
    print(f"Bodhan Zero-Shot Output  : {transcription}")
    print("=" * 65)

    # 7. Export artifact
    os.makedirs("outputs", exist_ok=True)
    out_file = "outputs/baseline_sample.json"
    result = {
        "model": "bodhan-ai/indic-transcribe-core",
        "task": "automatic-speech-recognition",
        "language": "mr",
        "device_used": device,
        "audio_file": audio_path,
        "audio_duration_seconds": round(info.duration, 3),
        "ground_truth": ground_truth,
        "baseline_transcription": transcription,
        "status": "success"
    }

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\n[ARTIFACT GENERATED] Baseline results saved to: {out_file}")
    return result


if __name__ == "__main__":
    run_baseline()
