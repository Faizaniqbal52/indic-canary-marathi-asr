"""
audit_all_train_speakers.py: Complete 100% Training Speaker Census
Reads the 'speaker_id' column across all 38 train parquet files in parallel
and computes the exact mathematical intersection with the 20 official validation speakers.
"""

import time
import fsspec
import pyarrow.parquet as pq
from concurrent.futures import ThreadPoolExecutor

VALID_SPEAKER_IDS = {40, 102, 113, 161, 291, 293, 365, 421, 427, 511, 527, 614, 615, 861, 867, 948, 978, 990, 1106, 1129}

def read_file_speakers(path_and_idx):
    idx, path, fs = path_and_idx
    try:
        with fs.open(path) as f:
            pf = pq.ParquetFile(f)
            tbl = pf.read(columns=['speaker_id'])
            spks = set(tbl['speaker_id'].to_pylist())
            print(f"[{idx+1:02d}/38] Shard read ({len(tbl)} rows): {len(spks)} unique speakers.")
            return spks
    except Exception as e:
        print(f"[{idx+1:02d}/38] Error reading shard: {e}")
        return set()

def main():
    print("=" * 65)
    print("FULL 100% CENSUS: AUDITING ALL 38 KATHBATH MARATHI TRAIN SHARDS")
    print("=" * 65)
    t0 = time.time()
    fs = fsspec.filesystem('hf')
    files = [(i, f"datasets/ai4bharat/Kathbath/marathi/train-{i:05d}-of-00038.parquet", fs) for i in range(38)]

    # Use 8 parallel workers
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(read_file_speakers, files))

    all_train_speakers = set().union(*results)
    elapsed = time.time() - t0

    overlap = all_train_speakers.intersection(VALID_SPEAKER_IDS)
    overlap_pct = (len(overlap) / len(VALID_SPEAKER_IDS)) * 100

    print("\n" + "=" * 65)
    print(f"AUDIT COMPLETE in {elapsed:.2f} seconds!")
    print(f"Total Unique Train Speakers Identified : {len(all_train_speakers)}")
    print(f"Total Official Valid Speakers           : {len(VALID_SPEAKER_IDS)}")
    print(f"Exact Overlapping Speaker Count        : {len(overlap)} / {len(VALID_SPEAKER_IDS)} ({overlap_pct:.1f}%)")
    print(f"Overlapping Speaker IDs                : {sorted(list(overlap))}")
    print("=" * 65)

    return len(all_train_speakers), len(overlap)

if __name__ == "__main__":
    main()
