"""
05_prepare_level4_data.py: Level 4 Dataset Preparation & Profiling
Extracts a controlled, speaker-disjoint dataset from ai4bharat/Kathbath (Marathi):
  - Training Partition: ~400 utterances across 35+ exclusive training speakers (zero validation leakage)
  - Validation Partition: Exactly 100 utterances (5 per speaker across all 20 official validation speakers)
Uses local cached Parquet shards directly for maximum I/O performance.
Saves 16kHz mono PCM WAVs and writes Canary JSONL manifests.
Generates comprehensive demographic & acoustic telemetry (speaker distributions, duration statistics).
Prepares directory structure for Kaggle dataset upload.
"""

import os
import sys
import glob
import json
import pyarrow.parquet as pq
import subprocess
from collections import defaultdict

# Force UTF-8 on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# The 20 official Kathbath Marathi validation speakers (MUST NOT appear in training)
OFFICIAL_VALID_SPEAKERS = {
    40, 102, 113, 161, 291, 293, 365, 421, 427, 511,
    527, 614, 615, 861, 867, 948, 978, 990, 1106, 1129
}


def find_cached_parquet(filename_pattern):
    cache_pattern = os.path.expanduser(
        f"~/.cache/huggingface/hub/datasets--ai4bharat--Kathbath/snapshots/*/marathi/{filename_pattern}"
    )
    matches = glob.glob(cache_pattern)
    for m in matches:
        if os.path.exists(m):
            return m
    raise FileNotFoundError(f"Cached parquet {filename_pattern} not found in local Hugging Face cache.")


def convert_bytes_to_wav(raw_bytes, out_wav_path):
    if os.path.exists(out_wav_path) and os.path.getsize(out_wav_path) > 1000:
        return  # already converted

    raw_temp = out_wav_path + ".raw"
    with open(raw_temp, "wb") as f:
        f.write(raw_bytes)
    subprocess.run(
        ["ffmpeg", "-y", "-i", raw_temp, "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", out_wav_path],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True
    )
    if os.path.exists(raw_temp):
        os.remove(raw_temp)


def extract_validation_benchmark(base_out_dir, per_speaker_target=5):
    print("=" * 65)
    print(f"EXTRACTING VALIDATION BENCHMARK (Target: {per_speaker_target} samples x 20 speakers = 100)")
    print("=" * 65)
    sys.stdout.flush()

    val_wav_dir = os.path.join(base_out_dir, "wavs", "val")
    os.makedirs(val_wav_dir, exist_ok=True)

    val_parquet = find_cached_parquet("valid*.parquet")
    print(f"Using local cached validation parquet: {val_parquet}")
    sys.stdout.flush()

    speaker_counts = defaultdict(int)
    val_records = []

    pf = pq.ParquetFile(val_parquet)
    # Read entire validation table from local SSD into memory (only ~180MB)
    table = pf.read()
    d = table.to_pydict()
    total_rows = len(d["fname"])

    for i in range(total_rows):
        spk = int(d["speaker_id"][i])
        dur = float(d["duration"][i])
        txt = str(d["text"][i]).strip()
        gender = str(d["gender"][i]).strip()
        fname = str(d["fname"][i])

        if spk in OFFICIAL_VALID_SPEAKERS and speaker_counts[spk] < per_speaker_target:
            if 2.0 <= dur <= 12.0 and txt:
                wav_filename = f"val_spk{spk:04d}_{speaker_counts[spk]:02d}_{fname.replace('.m4a', '.wav')}"
                out_wav = os.path.join(val_wav_dir, wav_filename)
                convert_bytes_to_wav(d["audio_filepath"][i]["bytes"], out_wav)

                rec = {
                    "audio_filepath": f"wavs/val/{wav_filename}",
                    "duration": round(dur, 3),
                    "text": txt,
                    "speaker_id": spk,
                    "gender": gender,
                    "source_lang": "mr",
                    "target_lang": "mr",
                    "pnc": "yes",
                    "task": "asr"
                }
                val_records.append(rec)
                speaker_counts[spk] += 1

        if len(val_records) >= len(OFFICIAL_VALID_SPEAKERS) * per_speaker_target:
            break

    print(f"Validation extraction complete! Total records: {len(val_records)} across {len(speaker_counts)} speakers.")
    sys.stdout.flush()
    return val_records


def extract_training_partition(base_out_dir, target_speakers=38, max_per_speaker=11, total_target=400):
    print("\n" + "=" * 65)
    print(f"EXTRACTING TRAINING PARTITION (Target: ~{total_target} samples across {target_speakers} exclusive speakers)")
    print("=" * 65)
    sys.stdout.flush()

    train_wav_dir = os.path.join(base_out_dir, "wavs", "train")
    os.makedirs(train_wav_dir, exist_ok=True)

    train_parquet = find_cached_parquet("train-00000*.parquet")
    print(f"Using local cached training parquet shard: {train_parquet}")
    sys.stdout.flush()

    speaker_counts = defaultdict(int)
    train_records = []

    pf = pq.ParquetFile(train_parquet)
    # Read entire shard 0 into memory (~170MB)
    table = pf.read()
    d = table.to_pydict()
    total_rows = len(d["fname"])

    for i in range(total_rows):
        spk = int(d["speaker_id"][i])
        dur = float(d["duration"][i])
        txt = str(d["text"][i]).strip()
        gender = str(d["gender"][i]).strip()
        fname = str(d["fname"][i])

        # Enforce strict non-overlap with validation speakers
        if spk not in OFFICIAL_VALID_SPEAKERS:
            if speaker_counts[spk] < max_per_speaker:
                if 2.5 <= dur <= 10.0 and txt:
                    wav_filename = f"train_spk{spk:04d}_{speaker_counts[spk]:02d}_{fname.replace('.m4a', '.wav')}"
                    out_wav = os.path.join(train_wav_dir, wav_filename)
                    convert_bytes_to_wav(d["audio_filepath"][i]["bytes"], out_wav)

                    rec = {
                        "audio_filepath": f"wavs/train/{wav_filename}",
                        "duration": round(dur, 3),
                        "text": txt,
                        "speaker_id": spk,
                        "gender": gender,
                        "source_lang": "mr",
                        "target_lang": "mr",
                        "pnc": "yes",
                        "task": "asr"
                    }
                    train_records.append(rec)
                    speaker_counts[spk] += 1

                    if len(train_records) % 50 == 0:
                        print(f"  Extracted {len(train_records)} / {total_target} samples across {len(speaker_counts)} speakers...")
                        sys.stdout.flush()

                    if len(train_records) >= total_target and len(speaker_counts) >= target_speakers:
                        break

    print(f"Training extraction complete! Total records: {len(train_records)} across {len(speaker_counts)} exclusive speakers.")
    sys.stdout.flush()
    return train_records


def compute_telemetry(records, name):
    durations = [r["duration"] for r in records]
    speakers = set(r["speaker_id"] for r in records)
    genders = defaultdict(int)
    for r in records:
        genders[r["gender"]] += 1

    durations.sort()
    n = len(durations)
    total_dur_s = sum(durations)
    mean_dur = total_dur_s / n if n > 0 else 0
    median_dur = durations[n // 2] if n > 0 else 0
    p10 = durations[int(n * 0.1)] if n > 0 else 0
    p90 = durations[int(n * 0.9)] if n > 0 else 0

    return {
        "split_name": name,
        "sample_count": n,
        "unique_speakers": len(speakers),
        "speaker_list": sorted(list(speakers)),
        "gender_breakdown": dict(genders),
        "total_duration_s": round(total_dur_s, 2),
        "total_duration_minutes": round(total_dur_s / 60, 2),
        "total_duration_hours": round(total_dur_s / 3600, 3),
        "duration_min_s": round(durations[0], 2) if n > 0 else 0,
        "duration_max_s": round(durations[-1], 2) if n > 0 else 0,
        "duration_mean_s": round(mean_dur, 2),
        "duration_median_s": round(median_dur, 2),
        "duration_p10_s": round(p10, 2),
        "duration_p90_s": round(p90, 2)
    }


def main():
    print("=" * 65)
    print("LEVEL 4 DATASET PREPARATION & PROFILING PIPELINE")
    print("=" * 65)
    sys.stdout.flush()

    base_out_dir = "data/kaggle_dataset"
    os.makedirs(base_out_dir, exist_ok=True)

    # 1. Extract Validation Benchmark (100 samples)
    val_records = extract_validation_benchmark(base_out_dir, per_speaker_target=5)
    val_manifest = os.path.join(base_out_dir, "val_manifest.jsonl")
    with open(val_manifest, "w", encoding="utf-8") as f:
        for r in val_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # 2. Extract Training Partition (~400 samples, 35+ exclusive speakers)
    train_records = extract_training_partition(base_out_dir, target_speakers=35, max_per_speaker=12, total_target=400)
    train_manifest = os.path.join(base_out_dir, "train_manifest.jsonl")
    with open(train_manifest, "w", encoding="utf-8") as f:
        for r in train_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # 3. Compute Demographic & Acoustic Telemetry
    val_telemetry = compute_telemetry(val_records, "validation_benchmark")
    train_telemetry = compute_telemetry(train_records, "training_partition")

    # 4. Strict Overlap Assertion
    train_spk_set = set(train_telemetry["speaker_list"])
    val_spk_set = set(val_telemetry["speaker_list"])
    overlap = train_spk_set & val_spk_set

    print("\n" + "=" * 65)
    print("DATASET PROFILE & LEAKAGE VERIFICATION SUMMARY")
    print("=" * 65)
    print(f"Training Samples    : {train_telemetry['sample_count']} across {train_telemetry['unique_speakers']} speakers")
    print(f"Training Audio      : {train_telemetry['total_duration_minutes']:.1f} minutes ({train_telemetry['total_duration_hours']:.2f} hours)")
    print(f"Training Genders    : {train_telemetry['gender_breakdown']}")
    print(f"Validation Samples  : {val_telemetry['sample_count']} across {val_telemetry['unique_speakers']} speakers (5 per speaker)")
    print(f"Validation Audio    : {val_telemetry['total_duration_minutes']:.1f} minutes")
    print(f"Validation Genders  : {val_telemetry['gender_breakdown']}")
    print(f"Speaker Overlap     : {len(overlap)} speakers ({overlap})")
    assert len(overlap) == 0, f"FATAL: Speaker leakage detected! Overlapping speakers: {overlap}"
    print("Verification Gate   : [PASS] ZERO SPEAKER OVERLAP CONFIRMED!")
    sys.stdout.flush()

    # 5. Export Profile Report
    os.makedirs("artifacts/06_dataset_manifests", exist_ok=True)
    profile_path = "artifacts/06_dataset_manifests/dataset_profile.json"
    full_profile = {
        "experiment_id": "EXP-007-PREP",
        "dataset_name": "ai4bharat/Kathbath_marathi_level4_partition",
        "leakage_audit": {
            "validation_speakers_count": len(val_spk_set),
            "training_speakers_count": len(train_spk_set),
            "overlap_count": len(overlap),
            "leakage_status": "ZERO_LEAKAGE_VERIFIED"
        },
        "training_partition": train_telemetry,
        "validation_benchmark": val_telemetry
    }

    with open(profile_path, "w", encoding="utf-8") as f:
        json.dump(full_profile, f, ensure_ascii=False, indent=2)
    print(f"\nProfile report exported to: {profile_path}")

    # Copy manifests to data/manifests
    os.makedirs("data/manifests", exist_ok=True)
    with open("data/manifests/train_manifest_400.jsonl", "w", encoding="utf-8") as f:
        for r in train_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open("data/manifests/val_manifest_100.jsonl", "w", encoding="utf-8") as f:
        for r in val_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Create dataset-metadata.json for Kaggle
    dataset_metadata = {
        "title": "Kathbath Marathi Level4 Fine-Tuning Partition",
        "id": "ac2sny/kathbath-marathi-level4",
        "licenses": [{"name": "CC0-1.0"}]
    }
    with open(os.path.join(base_out_dir, "dataset-metadata.json"), "w", encoding="utf-8") as f:
        json.dump(dataset_metadata, f, indent=2)
    print(f"Kaggle dataset metadata created in: {base_out_dir}/dataset-metadata.json")
    print("=" * 65)
    print("LEVEL 4 DATASET PREPARATION COMPLETED SUCCESSFULLY!")
    print("=" * 65)
    sys.stdout.flush()


if __name__ == "__main__":
    main()
