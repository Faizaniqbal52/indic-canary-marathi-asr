"""
prepare_smoke_data.py: Level 3 Data Preparation
Extracts exactly 10 training examples (from exclusive training speakers)
and 5 validation examples (from official valid split).
Saves 16kHz mono WAV files and writes JSONL manifests with full Canary v2 schema.
"""

import os
import sys
import json
import fsspec
import pyarrow.parquet as pq
import subprocess

# Force UTF-8 stdout on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Known validation speaker IDs to strictly exclude from our derived training slice
VALID_SPEAKER_IDS = {40, 102, 113, 161, 291, 293, 365, 421, 427, 511, 527, 614, 615, 861, 867, 948, 978, 990, 1106, 1129}

def extract_samples():
    print("=" * 65)
    print("LEVEL 3: PREPARING 10 TRAIN + 5 VALID SMOKE DATASET")
    print("=" * 65)

    fs = fsspec.filesystem("hf")
    wav_dir = "data/smoke_wavs"
    manifest_dir = "data/manifests"
    os.makedirs(wav_dir, exist_ok=True)
    os.makedirs(manifest_dir, exist_ok=True)

    # 1. Extract 10 Train Samples from exclusive speakers
    train_parquet = "datasets/ai4bharat/Kathbath/marathi/train-00000-of-00038.parquet"
    train_manifest_path = os.path.join(manifest_dir, "smoke_train_manifest.jsonl")
    train_records = []

    print(f"\n[1/2] Extracting 10 train samples from exclusive speakers...")
    with fs.open(train_parquet) as f:
        pf = pq.ParquetFile(f)
        for batch in pf.iter_batches(batch_size=50):
            d = batch.to_pydict()
            for i in range(len(d["fname"])):
                spk = d["speaker_id"][i]
                dur = d["duration"][i]
                txt = d["text"][i]
                fname = d["fname"][i]
                raw_bytes = d["audio_filepath"][i]["bytes"]

                # Enforce exclusive speaker constraint and clean duration
                if spk not in VALID_SPEAKER_IDS and 2.0 <= dur <= 12.0 and txt.strip():
                    wav_name = f"train_{len(train_records):02d}_{fname.replace('.m4a', '.wav')}"
                    out_wav = os.path.join(wav_dir, wav_name)
                    raw_temp = out_wav + ".raw"

                    with open(raw_temp, "wb") as rf:
                        rf.write(raw_bytes)

                    # Convert to 16kHz mono PCM WAV via ffmpeg
                    subprocess.run(
                        ["ffmpeg", "-y", "-i", raw_temp, "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", out_wav],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True
                    )
                    if os.path.exists(raw_temp):
                        os.remove(raw_temp)

                    train_records.append({
                        "audio_filepath": os.path.abspath(out_wav),
                        "duration": round(float(dur), 3),
                        "text": txt.strip(),
                        "source_lang": "mr",
                        "target_lang": "mr",
                        "pnc": "yes",
                        "task": "asr",
                        "speaker_id": int(spk)
                    })
                    print(f"  Train [{len(train_records):02d}/10] Speaker {spk:04d} ({dur:.2f}s)")
                    if len(train_records) >= 10:
                        break
            if len(train_records) >= 10:
                break

    with open(train_manifest_path, "w", encoding="utf-8") as f:
        for r in train_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Saved: {train_manifest_path} (10 entries)")

    # 2. Extract 5 Valid Samples from official valid
    valid_parquet = "datasets/ai4bharat/Kathbath/marathi/valid-00000-of-00001.parquet"
    valid_manifest_path = os.path.join(manifest_dir, "smoke_val_manifest.jsonl")
    valid_records = []

    print(f"\n[2/2] Extracting 5 validation samples...")
    with fs.open(valid_parquet) as f:
        pf = pq.ParquetFile(f)
        for batch in pf.iter_batches(batch_size=50):
            d = batch.to_pydict()
            for i in range(len(d["fname"])):
                spk = d["speaker_id"][i]
                dur = d["duration"][i]
                txt = d["text"][i]
                fname = d["fname"][i]
                raw_bytes = d["audio_filepath"][i]["bytes"]

                if 2.0 <= dur <= 12.0 and txt.strip():
                    wav_name = f"val_{len(valid_records):02d}_{fname.replace('.m4a', '.wav')}"
                    out_wav = os.path.join(wav_dir, wav_name)
                    raw_temp = out_wav + ".raw"

                    with open(raw_temp, "wb") as rf:
                        rf.write(raw_bytes)

                    subprocess.run(
                        ["ffmpeg", "-y", "-i", raw_temp, "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", out_wav],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True
                    )
                    if os.path.exists(raw_temp):
                        os.remove(raw_temp)

                    valid_records.append({
                        "audio_filepath": os.path.abspath(out_wav),
                        "duration": round(float(dur), 3),
                        "text": txt.strip(),
                        "source_lang": "mr",
                        "target_lang": "mr",
                        "pnc": "yes",
                        "task": "asr",
                        "speaker_id": int(spk)
                    })
                    print(f"  Valid [{len(valid_records):02d}/05] Speaker {spk:04d} ({dur:.2f}s)")
                    if len(valid_records) >= 5:
                        break
            if len(valid_records) >= 5:
                break

    with open(valid_manifest_path, "w", encoding="utf-8") as f:
        for r in valid_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Saved: {valid_manifest_path} (5 entries)")

    print("\n" + "=" * 65)
    print("DATA PREPARATION FOR LEVEL 3 SMOKE TEST COMPLETE!")
    print("=" * 65)


if __name__ == "__main__":
    extract_samples()
