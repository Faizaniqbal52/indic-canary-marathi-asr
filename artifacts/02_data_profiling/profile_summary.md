# Kathbath Marathi Dataset Profile & Split Audit Report

**Status:** Level 2 Complete (Full 100% Dataset Census)  
**Dataset:** `ai4bharat/Kathbath` (Language: Marathi / `mr`)  
**Official Split Dimensions:**
* Official Train Split: **84,070** utterances across 38 Parquet archives (~185.2 hours)
* Official Valid Split: **2,378** utterances in 1 Parquet archive (`valid-00000-of-00001.parquet`, ~4.27 hours)

---

## 1. 100% Complete Census of Official Validation Split (N = 2,378)

Every single record of the official validation split ($N = 2,378$) was audited.

| Metric | Official Valid Set (100% Full Audit, N=2,378) |
| :--- | :--- |
| **Total Utterances** | 2,378 |
| **Total Audio Duration** | ~4.27 hours |
| **Duration Minimum / Maximum** | 1.872s / 14.892s |
| **Duration Median** | 6.246s |
| **Duration 10th - 90th Percentile** | 4.481s - 8.826s |
| **Duration 99th Percentile** | 12.310s |
| **Corrupted / Empty Transcripts** | **0** (Zero corruption detected) |
| **Exact Unique Speaker Count** | **20 speakers** |
| **Complete Validation Speaker IDs** | `[40, 102, 113, 161, 291, 293, 365, 421, 427, 511, 527, 614, 615, 861, 867, 948, 978, 990, 1106, 1129]` |

---

## 2. 100% Complete Census of Official Training Split (N = 84,070)

All 38 Parquet shards of the training split ($N = 84,070$) were scanned across the `speaker_id` column in parallel.

* **Total Training Shards:** 38 Parquet files
* **Total Training Utterances:** 84,070
* **Total Unique Training Speakers:** **123 speakers**

### Groundbreaking Leakage Finding: 100% Speaker Overlap
The set intersection between the complete training speaker population and the complete validation speaker population reveals:
$$\text{Train Speakers (123)} \cap \text{Valid Speakers (20)} = \mathbf{20\text{ Overlapping Speakers (100.0\%)}}$$

Every single speaker present in the official Kathbath validation set ($20/20$) also has recordings in the official training set.

### Scientific & Evaluator Implication:
1. **The Nature of Official Valid:** The official validation set in Kathbath is strictly a **known-speaker evaluation benchmark**. Scoring on it measures in-distribution acoustic memorization and general ASR decoding, not out-of-distribution speaker generalization.
2. **Our Rigorous Split Architecture:**
   * **Benchmark Set:** Official Kathbath Valid ($N=2,378$, 20 speakers) — kept 100% untouched for global comparability.
   * **Derived Training Partition:** Sampled from the **103 exclusive training speakers** (123 total - 20 validation = 103 purely unseen speakers).
   * **Derived Development Partition:** Carved from a subset of those exclusive speakers, guaranteeing a **true speaker-disjoint validation split** with zero acoustic tract leakage!

---

## 3. Verified Canary v2 Manifest Contract

Audited directly against `bodhan-ai/indic-transcribe-core/nemo/canary_multilingual_tokenizer.py` and `load_nemo.py`:

```json
{
  "audio_filepath": "data/wavs/sample_001.wav",
  "duration": 4.899,
  "text": "अबोटाबाद येथे ओसामा बिन लादेनच्या घरात शिरून त्यांनी ओसामाचा खात्मा केला",
  "source_lang": "mr",
  "target_lang": "mr",
  "pnc": "yes",
  "task": "asr",
  "speaker_id": 1106
}
```

---

## 4. Engineering Decisions Locked for Training

1. **Filtering Policy:** Duration bounds $1.0\text{s} \le \text{duration} \le 15.0\text{s}$ are a **training engineering choice** for batch memory efficiency, not a hard Bodhan model limitation.
2. **Derived Split Guarantee:** Local training/validation sets will be partitioned by `speaker_id` from the 103 exclusive training speakers.
