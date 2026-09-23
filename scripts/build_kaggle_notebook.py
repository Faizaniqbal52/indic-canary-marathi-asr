"""
build_kaggle_notebook.py
Generates the self-contained indic_canary_kaggle_smoke.ipynb notebook for Kaggle.
"""

import json
import os

def build_notebook():
    os.makedirs("kaggle", exist_ok=True)

    metadata = {
        "id": "ac2sny/indic-canary-kaggle-smoke-test",
        "title": "Indic Canary Kaggle Smoke Test",
        "code_file": "indic_canary_kaggle_smoke.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": "true",
        "enable_gpu": "true",
        "enable_tpu": "false",
        "enable_internet": "true",
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": []
    }

    with open("kaggle/kernel-metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    cells = []

    def add_cell(cell_type, source):
        cells.append({
            "cell_type": cell_type,
            "metadata": {},
            "source": [line + "\n" for line in source.split("\n")]
        })

    # Header
    add_cell("markdown", """# Indic-Canary Marathi ASR: Kaggle Hardware & Training Verification
**Project:** AI4Bharat / Bodhan AI Take-Home Coding Assignment (IIT Madras)  
**Experiment:** Kaggle Cloud GPU Replication & Smoke Verification  
**Objective:** Verify that `bodhan-ai/indic-transcribe-core` with decoder LoRA ($r=16, \alpha=32$) installs, loads, forward-passes, backpropagates, updates weights, and saves checkpoints cleanly on Kaggle's 16GB GPU hardware before launching multi-step fine-tuning.""")

    # Cell 1: Environment & GPU Check
    add_cell("code", """# 1. Inspect Hardware & Environment
import os
import sys
import torch

print("=" * 60)
print("KAGGLE GPU & ENVIRONMENT INSPECTION")
print("=" * 60)
print("Python Version :", sys.version)
print("PyTorch Version:", torch.__version__)
print("CUDA Available :", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device Name    :", torch.cuda.get_device_name(0))
    print("VRAM (Total)   :", round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2), "GB")
else:
    print("WARNING: CUDA is NOT available! Ensure GPU accelerator is enabled in Kaggle Settings.")

!pip uninstall -y torchao
!pip install -q peft==0.19.1 soundfile torchaudio fsspec pyarrow jiwer""")

    # Cell 2: Hugging Face Authentication
    add_cell("code", """# 2. Hugging Face Authentication for Gated Access
import os
from huggingface_hub import login, whoami

try:
    from kaggle_secrets import UserSecretsClient
    user_secrets = UserSecretsClient()
    hf_token = user_secrets.get_secret("HF_TOKEN")
except Exception:
    hf_token = os.environ.get("HF_TOKEN", "")

if not hf_token:
    hf_token = input("Enter your Hugging Face Access Token: ").strip()

login(token=hf_token)
user_info = whoami()
print("Successfully authenticated as:", user_info["name"])""")

    # Cell 3: Download & Load Bodhan Foundation Model
    add_cell("code", """# 3. Download & Import bodhan-ai/indic-transcribe-core
import glob
import torch
from huggingface_hub import snapshot_download

print("Downloading model snapshot (model.safetensors + configs + tokenizer)...")
model_dir = snapshot_download(
    repo_id="bodhan-ai/indic-transcribe-core",
    token=hf_token,
    allow_patterns=["*.safetensors", "*.py", "*.json", "*.model"]
)
print("Snapshot cached at:", model_dir)

if model_dir not in sys.path:
    sys.path.insert(0, model_dir)

from modeling_indic_canary import IndicCanaryForConditionalGeneration
from feature_extraction_indic_canary import IndicCanaryFeatureExtractor
from tokenization_indic_canary import IndicCanaryTokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading IndicCanary in float32 onto {device}...")
model = IndicCanaryForConditionalGeneration.from_pretrained(model_dir, dtype=torch.float32)
model = model.to(device).train()
print("Base model loaded successfully!")""")

    # Cell 4: Apply LoRA Configuration
    add_cell("code", """# 4. Apply LoRA on Decoder Cross & Self-Attention (r=16, alpha=32)
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=r".*decoder.*(query_net|value_net)",
    lora_dropout=0.05,
    bias="none"
)

peft_model = get_peft_model(model, lora_config)
print("Trainable parameter summary:")
peft_model.print_trainable_parameters()

total_p = sum(p.numel() for p in peft_model.parameters())
train_p = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
assert train_p == 3145728, f"Unexpected trainable params: {train_p} (expected 3,145,728)"
print(f"Trainable parameters verified: {train_p:,} / {total_p:,} ({train_p/total_p*100:.2f}%)")""")

    # Cell 5: Stream Verified Marathi Audio Sample
    add_cell("code", """# 5. Load Verified Marathi Sample & Construct Canary Tensors
import fsspec
import pyarrow.parquet as pq
import subprocess
import soundfile as sf
import torchaudio

os.makedirs("/kaggle/working/data", exist_ok=True)
sample_wav = "/kaggle/working/data/smoke_sample.wav"

# Stream sample directly from Kathbath Parquet on HF
fs = fsspec.filesystem("hf", token=hf_token)
train_parquet = "datasets/ai4bharat/Kathbath/marathi/train-00000-of-00038.parquet"

print("Fetching verified Marathi sample from Kathbath...")
with fs.open(train_parquet) as f:
    pf = pq.ParquetFile(f)
    first_batch = next(pf.iter_batches(batch_size=10))
    d = first_batch.to_pydict()

raw_bytes = d["audio_filepath"][0]["bytes"]
reference_text = d["text"][0]
speaker_id = d["speaker_id"][0]
duration = d["duration"][0]

raw_temp = "/kaggle/working/data/temp.raw"
with open(raw_temp, "wb") as f:
    f.write(raw_bytes)

# Convert to 16kHz mono WAV via ffmpeg
subprocess.run(
    ["ffmpeg", "-y", "-i", raw_temp, "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", sample_wav],
    check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
)
if os.path.exists(raw_temp):
    os.remove(raw_temp)

print(f"Sample loaded: Duration={duration:.2f}s | Speaker={speaker_id}")
print(f"Reference Text: {reference_text}")

# Extract features & tokens
fe = IndicCanaryFeatureExtractor.from_pretrained(model_dir, device=device)
tokenizer = IndicCanaryTokenizer.from_pretrained(model_dir)

wav, sr = sf.read(sample_wav, dtype="float32", always_2d=True)
wav = torch.from_numpy(wav.mean(axis=1))
batch = wav.unsqueeze(0).to(device)
lens = torch.tensor([wav.shape[0]], device=device)
features, feature_lens = fe(batch, lens)

prompt_ids = tokenizer.encode_prompt(lang="mr", itn=False, romanized=False)
raw_pieces = tokenizer.multi.encode(reference_text)
text_token_ids = [p + tokenizer.spl_size for p in raw_pieces]
full_seq = prompt_ids + text_token_ids + [tokenizer.eos_id]

decoder_input_ids = torch.tensor([full_seq[:-1]], dtype=torch.long, device=device)
target_seq = full_seq[1:]
labels_list = [-100 if idx < len(prompt_ids) - 1 else tok for idx, tok in enumerate(target_seq)]
labels = torch.tensor([labels_list], dtype=torch.long, device=device)

print(f"Input features shape: {features.shape}")
print(f"Decoder input shape: {decoder_input_ids.shape}")
print(f"Labels shape       : {labels.shape}")""")

    # Cell 6: Forward, Backward & Optimizer Step
    add_cell("code", """# 6. Execute Forward, Backward & AdamW Optimizer Step
import time
import torch.nn.functional as F

optimizer = torch.optim.AdamW(
    [p for p in peft_model.parameters() if p.requires_grad],
    lr=1e-4, betas=(0.9, 0.98), eps=1e-8
)
optimizer.zero_grad()

print("Executing forward pass...")
torch.cuda.reset_peak_memory_stats()
t0 = time.time()

enc_outputs = peft_model.model.model.encoder(features, attention_mask=None)
cross_mask = peft_model.model._cross_mask_from_lengths(enc_outputs[1], enc_outputs[0].size(1))
hidden = peft_model.model.model.decoder(decoder_input_ids, enc_outputs[0], cross_mask)
logits = peft_model.model.lm_head(hidden)

loss = F.cross_entropy(logits.view(-1, tokenizer.vocab_size), labels.view(-1), ignore_index=-100, label_smoothing=0.1)
t_fwd = time.time() - t0

print(f"Forward Pass Completed! Loss: {loss.item():.4f} (Time: {t_fwd*1000:.1f}ms)")
assert not torch.isnan(loss) and not torch.isinf(loss), "Loss is NaN/Inf!"
assert loss.item() > 0, "Loss is non-positive!"

print("Executing backward pass...")
t1 = time.time()
loss.backward()
t_bwd = time.time() - t1

grad_norm = sum(p.grad.norm(2).item()**2 for p in peft_model.parameters() if p.grad is not None)**0.5
print(f"Backward Pass Completed! Gradient Norm: {grad_norm:.4f} (Time: {t_bwd*1000:.1f}ms)")
assert grad_norm > 0, "Zero gradients detected!"

print("Executing optimizer step...")
t2 = time.time()
optimizer.step()
t_opt = time.time() - t2
print(f"Optimizer Step Completed! (Time: {t_opt*1000:.1f}ms)")

peak_vram_mb = torch.cuda.max_memory_allocated() / (1024**2)
peak_res_mb = torch.cuda.max_memory_reserved() / (1024**2)
print(f"Peak VRAM Allocated: {peak_vram_mb:.1f} MB / Reserved: {peak_res_mb:.1f} MB")""")

    # Cell 7: Checkpoint Serialization & Reload Verification
    add_cell("code", """# 7. Checkpoint Serialization & Reload Verification
ckpt_dir = "/kaggle/working/checkpoints"
os.makedirs(ckpt_dir, exist_ok=True)
ckpt_path = os.path.join(ckpt_dir, "kaggle_smoke_lora.pt")

print(f"Saving LoRA checkpoint to {ckpt_path}...")
lora_state = {k: v.cpu() for k, v in peft_model.state_dict().items() if "lora" in k}
torch.save({
    "step": 1,
    "loss": loss.item(),
    "grad_norm": grad_norm,
    "state_dict": lora_state
}, ckpt_path)

ckpt_size_mb = os.path.getsize(ckpt_path) / (1024**2)
print(f"Checkpoint saved successfully! File size: {ckpt_size_mb:.2f} MB")

# Verify reload
print("Verifying checkpoint reload...")
loaded = torch.load(ckpt_path, map_location="cpu", weights_only=False)
assert loaded["step"] == 1
assert len(loaded["state_dict"]) == len(lora_state)
print(f"[PASS] Checkpoint reloaded and verified! ({len(loaded['state_dict'])} tensors)")""")

    # Cell 8: Telemetry Export
    add_cell("code", """# 8. Export Kaggle Telemetry Artifact
import json

telemetry = {
    "experiment_id": "EXP-005-KAGGLE",
    "experiment_name": "kaggle_gpu_replication_smoke_test",
    "status": "PASS",
    "hardware": {
        "gpu_name": torch.cuda.get_device_name(0),
        "total_vram_gb": round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2),
        "peak_vram_allocated_mb": round(peak_vram_mb, 1),
        "peak_vram_reserved_mb": round(peak_res_mb, 1)
    },
    "model": {
        "base_model": "bodhan-ai/indic-transcribe-core",
        "adaptation": "LoRA (r=16, alpha=32, target=decoder attention)",
        "trainable_parameters": train_p,
        "total_parameters": total_p
    },
    "metrics": {
        "loss": round(loss.item(), 4),
        "gradient_norm": round(grad_norm, 4),
        "forward_latency_ms": round(t_fwd * 1000, 1),
        "backward_latency_ms": round(t_bwd * 1000, 1),
        "optimizer_latency_ms": round(t_opt * 1000, 1)
    },
    "checkpoint": {
        "path": ckpt_path,
        "file_size_mb": round(ckpt_size_mb, 2),
        "reload_verified": True
    }
}

out_path = "/kaggle/working/kaggle_smoke_result.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(telemetry, f, indent=2, ensure_ascii=False)

print("=" * 60)
print("KAGGLE REPLICATION SMOKE TEST COMPLETED SUCCESSFULLY!")
print("Artifact saved to:", out_path)
print("=" * 60)""")

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.10.12"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

    nb_path = "kaggle/indic_canary_kaggle_smoke.ipynb"
    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=2)

    print(f"Generated: {nb_path} ({len(cells)} cells)")
    print(f"Generated: kaggle/kernel-metadata.json")

if __name__ == "__main__":
    build_notebook()
