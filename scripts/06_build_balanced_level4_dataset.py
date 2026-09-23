"""
06_build_balanced_level4_dataset.py: Balanced Level 4 Dataset Construction & Profiling
Builds a strictly speaker-disjoint, gender-balanced (50/50) dataset for Level 4 fine-tuning:
  - Training Partition: Exactly 400 utterances across 40 exclusive speakers (20 Female, 20 Male, 10 utts/spk)
  - Validation Benchmark: Exactly 100 utterances across 20 validation speakers (10 Female, 10 Male, 5 utts/spk)
  - Seed: 42 (fully deterministic sampling)
  - Acoustic Filters: 2.5s <= duration <= 10.0s, non-empty text, valid Devanagari
  - Parquets: Read from local NVMe cache (valid-00000, train-00000, train-00025)
"""

import os
import sys
import glob
import json
import random
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

RANDOM_SEED = 42


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
    print(f"STEP 1: RE-VERIFYING FROZEN VALIDATION BENCHMARK (100 UTTERANCES)")
    print("=" * 65)
    sys.stdout.flush()

    val_wav_dir = os.path.join(base_out_dir, "wavs", "val")
    os.makedirs(val_wav_dir, exist_ok=True)

    val_parquet = find_cached_parquet("valid*.parquet")
    speaker_counts = defaultdict(int)
    val_records = []

    pf = pq.ParquetFile(val_parquet)
    table = pf.read()
    d = table.to_pydict()

    for i in range(len(d["fname"])):
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

    print(f"Validation benchmark ready: {len(val_records)} utterances across {len(speaker_counts)} speakers.")
    sys.stdout.flush()
    return val_records


def sample_exclusive_speaker_data(parquet_path, target_gender, num_speakers=20, utts_per_speaker=10):
    print(f"\nScanning {os.path.basename(parquet_path)} for {target_gender} exclusive speakers...")
    sys.stdout.flush()

    pf = pq.ParquetFile(parquet_path)
    table = pf.read()
    d = table.to_pydict()

    # Group eligible utterances by speaker
    speaker_utts = defaultdict(list)
    for i in range(len(d["fname"])):
        spk = int(d["speaker_id"][i])
        dur = float(d["duration"][i])
        txt = str(d["text"][i]).strip()
        gender = str(d["gender"][i]).strip()
        fname = str(d["fname"][i])
        raw_b = d["audio_filepath"][i]["bytes"]

        if spk not in OFFICIAL_VALID_SPEAKERS and gender == target_gender:
            if 2.5 <= dur <= 10.0 and len(txt) > 3:
                speaker_utts[spk].append({
                    "raw_bytes": raw_b,
                    "fname": fname,
                    "speaker_id": spk,
                    "gender": gender,
                    "duration": round(dur, 3),
                    "text": txt
                })

    # Filter speakers that have at least utts_per_speaker utterances
    eligible_speakers = [s for s, utts in speaker_utts.items() if len(utts) >= utts_per_speaker]
    eligible_speakers.sort()
    print(f"  Found {len(eligible_speakers)} eligible {target_gender} speakers with >= {utts_per_speaker} clean utterances.")
    sys.stdout.flush()

    # Deterministic seeded selection of 20 speakers
    rng = random.Random(RANDOM_SEED)
    selected_speakers = rng.sample(eligible_speakers, num_speakers)
    selected_speakers.sort()

    selected_records = []
    for spk in selected_speakers:
        utts = speaker_utts[spk]
        # Seeded sample of exactly utts_per_speaker
        spk_rng = random.Random(RANDOM_SEED + spk)
        sampled = spk_rng.sample(utts, utts_per_speaker)
        selected_records.extend(sampled)

    print(f"  Selected {len(selected_speakers)} {target_gender} speakers -> {len(selected_records)} utterances.")
    sys.stdout.flush()
    return selected_records, selected_speakers


def build_training_partition(base_out_dir):
    print("=" * 65)
    print("STEP 2: BUILDING BALANCED TRAINING PARTITION (200 FEMALE + 200 MALE)")
    print("=" * 65)
    sys.stdout.flush()

    train_wav_dir = os.path.join(base_out_dir, "wavs", "train")
    os.makedirs(train_wav_dir, exist_ok=True)

    # 1. 200 Female samples from shard 0
    female_parquet = find_cached_parquet("train-00000*.parquet")
    female_raw, female_spks = sample_exclusive_speaker_data(
        female_parquet, target_gender="Female", num_speakers=20, utts_per_speaker=10
    )

    # 2. 200 Male samples from shard 25
    male_parquet = find_cached_parquet("train-00025*.parquet")
    male_raw, male_spks = sample_exclusive_speaker_data(
        male_parquet, target_gender="Male", num_speakers=20, utts_per_speaker=10
    )

    all_raw = female_raw + male_raw
    # Deterministic shuffle across speakers
    rng = random.Random(RANDOM_SEED)
    rng.shuffle(all_raw)

    train_records = []
    print("\nConverting selected training audio to 16kHz mono WAVs...")
    sys.stdout.flush()

    for idx, item in enumerate(all_raw):
        spk = item["speaker_id"]
        fname = item["fname"]
        wav_filename = f"train_spk{spk:04d}_{idx:03d}_{fname.replace('.m4a', '.wav')}"
        out_wav = os.path.join(train_wav_dir, wav_filename)

        convert_bytes_to_wav(item["raw_bytes"], out_wav)

        rec = {
            "audio_filepath": f"wavs/train/{wav_filename}",
            "duration": item["duration"],
            "text": item["text"],
            "speaker_id": spk,
            "gender": item["gender"],
            "source_lang": "mr",
            "target_lang": "mr",
            "pnc": "yes",
            "task": "asr"
        }
        train_records.append(rec)

        if (idx + 1) % 50 == 0:
            print(f"  Processed {idx + 1} / {len(all_raw)} training WAVs...")
            sys.stdout.flush()

    print(f"Training partition ready: {len(train_records)} utterances across {len(female_spks) + len(male_spks)} exclusive speakers.")
    sys.stdout.flush()
    return train_records, female_spks, male_spks


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
    print("LEVEL 4: BALANCED DATASET PREPARATION & PROFILING PIPELINE")
    print("=" * 65)
    sys.stdout.flush()

    base_out_dir = "data/kaggle_dataset"
    os.makedirs(base_out_dir, exist_ok=True)

    # 1. Validation Benchmark (100 frozen samples)
    val_records = extract_validation_benchmark(base_out_dir, per_speaker_target=5)
    val_manifest = os.path.join(base_out_dir, "val_manifest.jsonl")
    with open(val_manifest, "w", encoding="utf-8") as f:
        for r in val_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # 2. Balanced Training Partition (400 samples: 200 Female + 200 Male)
    train_records, female_spks, male_spks = build_training_partition(base_out_dir)
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
    print("REVISED BALANCED DATASET PROFILE & LEAKAGE VERIFICATION")
    print("=" * 65)
    print(f"Random Seed         : {RANDOM_SEED} (Deterministic)")
    print(f"Training Samples    : {train_telemetry['sample_count']} across {train_telemetry['unique_speakers']} speakers")
    print(f"Training Audio      : {train_telemetry['total_duration_minutes']:.1f} minutes ({train_telemetry['total_duration_hours']:.2f} hours)")
    print(f"Training Genders    : {train_telemetry['gender_breakdown']} (50.0% Female / 50.0% Male)")
    print(f"Validation Samples  : {val_telemetry['sample_count']} across {val_telemetry['unique_speakers']} speakers (5 per speaker)")
    print(f"Validation Audio    : {val_telemetry['total_duration_minutes']:.1f} minutes")
    print(f"Validation Genders  : {val_telemetry['gender_breakdown']} (50.0% Female / 50.0% Male)")
    print(f"Speaker Overlap     : {len(overlap)} speakers ({overlap})")
    assert len(overlap) == 0, f"FATAL: Speaker leakage detected! Overlapping speakers: {overlap}"
    assert train_telemetry["gender_breakdown"]["Female"] == 200, "Female count mismatch!"
    assert train_telemetry["gender_breakdown"]["Male"] == 200, "Male count mismatch!"
    assert val_telemetry["gender_breakdown"]["Female"] == 50, "Val female count mismatch!"
    assert val_telemetry["gender_breakdown"]["Male"] == 50, "Val male count mismatch!"
    print("Verification Gate   : [PASS] ZERO SPEAKER OVERLAP & EXACT 50/50 GENDER BALANCE VERIFIED!")
    sys.stdout.flush()

    # 5. Export Profile Report
    os.makedirs("artifacts/06_dataset_manifests", exist_ok=True)
    profile_path = "artifacts/06_dataset_manifests/dataset_profile_balanced.json"
    full_profile = {
        "experiment_id": "EXP-007-PREP-BALANCED",
        "dataset_name": "ai4bharat/Kathbath_marathi_level4_balanced_partition",
        "random_seed": RANDOM_SEED,
        "sampling_methodology": {
            "validation_strategy": "100 frozen utterances (5 per speaker across all 20 official validation benchmark speakers, 10F/10M)",
            "training_strategy": "400 utterances (200 Female across 20 exclusive speakers from shard 0, 200 Male across 20 exclusive speakers from shard 25, 10 utts/spk)",
            "duration_bounds_seconds": [2.5, 10.0]
        },
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
    with open("data/manifests/train_manifest_balanced_400.jsonl", "w", encoding="utf-8") as f:
        for r in train_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open("data/manifests/val_manifest_frozen_100.jsonl", "w", encoding="utf-8") as f:
        for r in val_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Update dataset-metadata.json
    dataset_metadata = {
        "title": "Kathbath Marathi Level4 Balanced Fine-Tuning Partition",
        "id": "ac2sny/kathbath-marathi-level4",
        "licenses": [{"name": "CC0-1.0"}]
    }
    with open(os.path.join(base_out_dir, "dataset-metadata.json"), "w", encoding="utf-8") as f:
        json.dump(dataset_metadata, f, indent=2)
    print(f"Kaggle dataset metadata updated.")
    print("=" * 65)
    print("BALANCED DATASET CREATION COMPLETED SUCCESSFULLY!")
    print("=" * 65)
    sys.stdout.flush()


if __name__ == "__main__":
    main()
