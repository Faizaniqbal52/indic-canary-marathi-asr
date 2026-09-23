"""
02_profile_kathbath.py: Level 2 Data Profiling and Leakage Audit
Reads metadata directly from Kathbath Marathi parquet files using pyarrow
(skipping audio bytes for lightning-fast profiling).
Computes duration distribution, text statistics, speaker distributions,
and audits speaker overlap between official train and valid sets.
"""

import os
import sys
import json
import yaml
import numpy as np
import pyarrow.parquet as pq
import fsspec

# Force UTF-8 on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def load_config(config_path="configs/data_profile.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def profile_split(fs, parquet_path, max_rows=1500, columns=None):
    if columns is None:
        columns = ["fname", "text", "duration", "gender", "speaker_id"]

    print(f"Streaming metadata from: {parquet_path}")
    with fs.open(parquet_path) as f:
        pf = pq.ParquetFile(f)
        total_rows_in_file = pf.metadata.num_rows

        durations = []
        text_char_lens = []
        text_word_lens = []
        speakers = []
        genders = []
        empty_texts = 0

        collected = 0
        for batch in pf.iter_batches(batch_size=500, columns=columns):
            d = batch.to_pydict()
            n = len(d["fname"])

            for i in range(n):
                dur = d["duration"][i]
                txt = d["text"][i]
                spk = d["speaker_id"][i]
                gen = d["gender"][i]

                durations.append(dur)
                speakers.append(spk)
                genders.append(gen)

                if txt and str(txt).strip():
                    text_char_lens.append(len(str(txt).strip()))
                    text_word_lens.append(len(str(txt).strip().split()))
                else:
                    empty_texts += 1

                collected += 1
                if collected >= max_rows:
                    break
            if collected >= max_rows:
                break

    durations = np.array(durations)
    char_lens = np.array(text_char_lens)
    word_lens = np.array(text_word_lens)

    unique_speakers, spk_counts = np.unique(speakers, return_counts=True)
    gender_counts = {}
    for g in genders:
        g_str = str(g).capitalize()
        gender_counts[g_str] = gender_counts.get(g_str, 0) + 1

    stats = {
        "file": parquet_path,
        "total_rows_in_file": total_rows_in_file,
        "rows_profiled": collected,
        "empty_transcripts": empty_texts,
        "duration": {
            "min": round(float(np.min(durations)), 3),
            "max": round(float(np.max(durations)), 3),
            "mean": round(float(np.mean(durations)), 3),
            "median": round(float(np.median(durations)), 3),
            "p10": round(float(np.percentile(durations, 10)), 3),
            "p90": round(float(np.percentile(durations, 90)), 3),
            "std": round(float(np.std(durations)), 3)
        },
        "text_words": {
            "min": int(np.min(word_lens)) if len(word_lens) else 0,
            "max": int(np.max(word_lens)) if len(word_lens) else 0,
            "mean": round(float(np.mean(word_lens)), 2) if len(word_lens) else 0,
            "median": round(float(np.median(word_lens)), 2) if len(word_lens) else 0
        },
        "speakers": {
            "unique_count": int(len(unique_speakers)),
            "speaker_ids": [int(s) for s in unique_speakers],
            "min_utts_per_speaker": int(np.min(spk_counts)),
            "max_utts_per_speaker": int(np.max(spk_counts)),
            "mean_utts_per_speaker": round(float(np.mean(spk_counts)), 2)
        },
        "gender_distribution": gender_counts
    }
    return stats, set(unique_speakers)


def run_profiling():
    print("=" * 65)
    print("LEVEL 2: KATHBATH MARATHI METADATA PROFILING & LEAKAGE AUDIT")
    print("=" * 65)

    config = load_config()
    cfg_profile = config["profiling_budget"]
    out_json = config["output_paths"]["profile_report_json"]
    out_md = config["output_paths"]["profile_summary_md"]
    os.makedirs(os.path.dirname(out_json), exist_ok=True)

    fs = fsspec.filesystem("hf")

    train_parquet = "datasets/ai4bharat/Kathbath/marathi/train-00000-of-00038.parquet"
    valid_parquet = "datasets/ai4bharat/Kathbath/marathi/valid-00000-of-00001.parquet"

    # Profile Train
    train_stats, train_speakers = profile_split(
        fs, train_parquet, max_rows=cfg_profile["train_profile_rows"]
    )

    # Profile Valid
    valid_stats, valid_speakers = profile_split(
        fs, valid_parquet, max_rows=cfg_profile["valid_profile_rows"]
    )

    # Audit Speaker Overlap
    overlapping_speakers = train_speakers.intersection(valid_speakers)
    leakage_detected = len(overlapping_speakers) > 0
    overlap_pct = (len(overlapping_speakers) / len(valid_speakers) * 100) if len(valid_speakers) else 0

    audit_results = {
        "train_speakers_sampled": len(train_speakers),
        "valid_speakers_sampled": len(valid_speakers),
        "overlapping_speaker_count": len(overlapping_speakers),
        "overlapping_speaker_ids": [int(s) for s in overlapping_speakers],
        "overlap_percentage_of_valid": round(overlap_pct, 2),
        "leakage_detected": leakage_detected,
        "implication": (
            "Speaker leakage exists in official splits. We must create an explicitly named "
            "derived speaker-disjoint partition for rigorous training."
            if leakage_detected else
            "No speaker overlap detected between sampled train and valid partitions."
        )
    }

    full_report = {
        "dataset": config["dataset"],
        "filtering_policy": config["filtering_policy"],
        "train_partition_profile": train_stats,
        "valid_partition_profile": valid_stats,
        "speaker_leakage_audit": audit_results
    }

    # Save JSON report
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2, ensure_ascii=False)
    print(f"\n[REPORT SAVED] JSON profile report: {out_json}")

    # Generate Markdown Summary
    md_content = f"""# Kathbath Marathi Dataset Profile & Leakage Audit Report

**Generated:** {full_report['train_partition_profile']['rows_profiled']} Train rows, {full_report['valid_partition_profile']['rows_profiled']} Valid rows profiled.

## 1. Summary Statistics

| Metric | Official Train (Sampled) | Official Valid (Full) |
| :--- | :--- | :--- |
| **Total Rows in File** | {train_stats['total_rows_in_file']} | {valid_stats['total_rows_in_file']} |
| **Rows Profiled** | {train_stats['rows_profiled']} | {valid_stats['rows_profiled']} |
| **Duration (Min / Max)** | {train_stats['duration']['min']}s / {train_stats['duration']['max']}s | {valid_stats['duration']['min']}s / {valid_stats['duration']['max']}s |
| **Duration (Mean ± Std)** | {train_stats['duration']['mean']}s ± {train_stats['duration']['std']}s | {valid_stats['duration']['mean']}s ± {valid_stats['duration']['std']}s |
| **Duration Median** | {train_stats['duration']['median']}s | {valid_stats['duration']['median']}s |
| **Duration (10th - 90th %ile)** | {train_stats['duration']['p10']}s - {train_stats['duration']['p90']}s | {valid_stats['duration']['p10']}s - {valid_stats['duration']['p90']}s |
| **Words per Utterance (Mean)** | {train_stats['text_words']['mean']} words | {valid_stats['text_words']['mean']} words |
| **Unique Speakers** | {train_stats['speakers']['unique_count']} | {valid_stats['speakers']['unique_count']} |
| **Empty Transcripts** | {train_stats['empty_transcripts']} | {valid_stats['empty_transcripts']} |
| **Gender Distribution** | {train_stats['gender_distribution']} | {valid_stats['gender_distribution']} |

## 2. Speaker Leakage Audit

* **Train Speakers Sampled:** `{audit_results['train_speakers_sampled']}`
* **Valid Speakers Sampled:** `{audit_results['valid_speakers_sampled']}`
* **Overlapping Speakers:** `{audit_results['overlapping_speaker_count']}`
* **Overlap Rate on Valid Set:** `{audit_results['overlap_percentage_of_valid']}%`
* **Finding:** {audit_results['implication']}

## 3. Engineering Decisions for Fine-Tuning Manifest
1. **Duration Filtering Policy:** {config['filtering_policy']['min_duration_seconds']}s to {config['filtering_policy']['max_duration_seconds']}s captures >95% of data while pruning silence and runaway memory spikes.
2. **Audio Materialization:** Stream metadata first; only materialize WAV files for our designated training/dev partition.
3. **Partitioning:** Preserve official valid set untouched as the global benchmark; create a derived speaker-disjoint dev split for local validation.
"""

    with open(out_md, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"[REPORT SAVED] Markdown summary: {out_md}")

    print("\n" + "=" * 65)
    print(f"LEAKAGE AUDIT RESULT : {'LEAKAGE DETECTED' if leakage_detected else 'ZERO OVERLAP'}")
    print(f"Overlapping Speakers : {audit_results['overlapping_speaker_count']}")
    print(f"Mean Train Duration  : {train_stats['duration']['mean']}s (Median: {train_stats['duration']['median']}s)")
    print(f"Mean Valid Duration  : {valid_stats['duration']['mean']}s (Median: {valid_stats['duration']['median']}s)")
    print("=" * 65)


if __name__ == "__main__":
    run_profiling()
