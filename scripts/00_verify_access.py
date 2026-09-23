"""
00_verify_access.py: Level 0 Preflight Verification
Verifies Hugging Face credentials, gated repo access for bodhan-ai/indic-transcribe-core,
and dataset access for ai4bharat/Kathbath.
"""

import sys
import os
from huggingface_hub import HfApi, whoami, hf_hub_download
from datasets import load_dataset


def check_auth():
    print("=" * 60)
    print("STEP 1: Checking Hugging Face Local Authentication")
    print("=" * 60)
    try:
        user_info = whoami()
        username = user_info.get("name", user_info.get("fullname", "Unknown"))
        email = user_info.get("email", "N/A")
        print(f"[SUCCESS] Authenticated as user: {username} ({email})")
    except Exception as e:
        print(f"[FAILED] Local token not found or invalid: {e}")
        print("\nPlease run in terminal: py -m huggingface_hub.cli login")
        print("or set the HF_TOKEN environment variable.")
        return False
    return True


def check_model_access():
    print("\n" + "=" * 60)
    print("STEP 2: Checking Access to Gated Model: bodhan-ai/indic-transcribe-core")
    print("=" * 60)
    repo_id = "bodhan-ai/indic-transcribe-core"
    try:
        config_path = hf_hub_download(repo_id=repo_id, filename="config.json")
        print(f"[SUCCESS] Successfully accessed and downloaded config.json from {repo_id}")
        print(f"          Cached at: {config_path}")
        return True
    except Exception as e:
        print(f"[FAILED] Cannot access model {repo_id}: {e}")
        print("\nMake sure your access request has been submitted and approved at:")
        print(f"https://huggingface.co/{repo_id}")
        return False


def check_dataset_access():
    print("\n" + "=" * 60)
    print("STEP 3: Checking Access to Gated Dataset: ai4bharat/Kathbath (Marathi)")
    print("=" * 60)
    dataset_id = "ai4bharat/Kathbath"
    try:
        print(f"Connecting to {dataset_id} via streaming...")
        ds = load_dataset(dataset_id, "marathi", split="train", streaming=True)
        sample = next(iter(ds))
        keys = list(sample.keys())
        print(f"[SUCCESS] Successfully streamed 1 record from {dataset_id} (marathi split)!")
        print(f"          Available keys: {keys}")
        return True
    except Exception as e:
        print(f"[FAILED] Cannot access dataset {dataset_id}: {e}")
        print("\nMake sure your access request has been submitted and approved at:")
        print(f"https://huggingface.co/datasets/{dataset_id}")
        return False


if __name__ == "__main__":
    print("RUNNING LEVEL 0 PREFLIGHT ACCESS CHECK...\n")
    auth_ok = check_auth()
    if not auth_ok:
        sys.exit(1)

    model_ok = check_model_access()
    dataset_ok = check_dataset_access()

    print("\n" + "=" * 60)
    print("PREFLIGHT SUMMARY:")
    print(f"  Hugging Face Authentication : {'PASS' if auth_ok else 'FAIL'}")
    print(f"  Bodhan Model Access        : {'PASS' if model_ok else 'FAIL'}")
    print(f"  Kathbath Dataset Access    : {'PASS' if dataset_ok else 'FAIL'}")
    print("=" * 60)

    if auth_ok and model_ok and dataset_ok:
        print("\n>>> ALL ACCESS CHECKS PASSED. Ready for Level 1 Baseline Inference!")
        sys.exit(0)
    else:
        print("\n>>> ONE OR MORE CHECKS FAILED. Resolve before continuing.")
        sys.exit(1)
