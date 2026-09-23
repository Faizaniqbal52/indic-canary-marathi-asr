"""
03_smoke_test.py: Level 3 1-Batch Training Smoke Test
Audits and executes one complete end-to-end training step on a single Marathi sample:
  1. Data Loading: Loads 1 verified Marathi sample from smoke manifest.
  2. Preprocessing: Computes 80-channel log-mel filterbanks and Canary token IDs with prompt masking.
  3. Forward Pass: Passes acoustic features through FastConformer encoder and Transformer decoder.
  4. Canary Seq2Seq Loss: Computes cross-entropy over target tokens with prompt masked as -100.
  5. Backward Pass: Computes gradients via backpropagation.
  6. Optimizer Step: Updates weights via AdamW.
  7. Checkpoint Verification: Serializes state_dict to disk and reloads to confirm recovery.
  8. Artifact Export: Saves outputs/smoke_test_result.json.
"""

import os
import sys
import glob
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import soundfile as sf

# Force UTF-8 on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def find_local_model_dir():
    pattern = os.path.expanduser("~/.cache/huggingface/hub/models--bodhan-ai--indic-transcribe-core/snapshots/*")
    matches = glob.glob(pattern)
    for m in matches:
        if os.path.exists(os.path.join(m, "model.safetensors")):
            return m
    raise FileNotFoundError("Cached model directory not found.")


def build_canary_training_tensors(audio_path, text, fe, tokenizer, device="cuda"):
    """
    Constructs the exact input and target tensors matching Canary Seq2Seq training:
      - input_features: (1, T_mel, 80)
      - attention_mask: (1,) lengths tensor
      - decoder_input_ids: (1, L) prompt + text_ids (excluding last token)
      - labels: (1, L) target tokens shifted, with prompt positions masked as -100
    """
    # 1. Load and resample audio
    wav, sr = sf.read(audio_path, dtype="float32", always_2d=True)
    wav = torch.from_numpy(wav.mean(axis=1))  # mono
    if sr != fe.sample_rate:
        import torchaudio
        wav = torchaudio.functional.resample(wav, sr, fe.sample_rate)

    # 2. Extract 80-channel log-mel filterbanks
    min_len = fe.sample_rate
    n = wav.shape[0]
    if n < min_len:
        batch = torch.zeros(1, min_len)
        off = round((min_len - n) / 2)
        batch[0, off : off + n] = wav
        lens = torch.tensor([min_len])
    else:
        batch, lens = wav.unsqueeze(0), torch.tensor([n])

    batch = batch.to(device)
    lens = lens.to(device)
    features, feature_lens = fe(batch, lens)

    # 3. Construct Canary prompt and target tokens
    # Prompt is 10 tokens: <|startofcontext|><|startoftranscript|><|emo:undefined|><|mr|><|mr|><|pnc|><|noitn|><|noromanized|><|notimestamp|><|nodiarize|>
    prompt_ids = tokenizer.encode_prompt(lang="mr", itn=False, romanized=False)

    # Marathi text tokens are encoded via multilingual sub-tokenizer (offset by spl_size = 1152)
    raw_pieces = tokenizer.multi.encode(text)
    text_token_ids = [p + tokenizer.spl_size for p in raw_pieces]
    eos_id = tokenizer.eos_id  # 3

    # Full sequence: [PROMPT_TOKENS] + [TEXT_TOKENS] + [EOS]
    full_seq = prompt_ids + text_token_ids + [eos_id]

    # Teacher-forcing:
    # decoder_input_ids = full_seq[:-1]
    # labels = full_seq[1:] with prompt positions set to -100 (ignored in loss)
    decoder_input_ids = torch.tensor([full_seq[:-1]], dtype=torch.long, device=device)

    target_seq = full_seq[1:]
    num_prompt = len(prompt_ids)
    labels_list = []
    for idx, tok in enumerate(target_seq):
        if idx < num_prompt - 1:
            # Mask out prompt tokens so loss is NOT computed on prompt
            labels_list.append(-100)
        else:
            labels_list.append(tok)

    labels = torch.tensor([labels_list], dtype=torch.long, device=device)

    return features, feature_lens, decoder_input_ids, labels, len(prompt_ids), len(text_token_ids)


def run_smoke_test():
    print("=" * 65)
    print("LEVEL 3: 1-BATCH TRAINING SMOKE TEST (CANARY SEQ2SEQ)")
    print("=" * 65)

    model_dir = find_local_model_dir()
    if model_dir not in sys.path:
        sys.path.insert(0, model_dir)

    from feature_extraction_indic_canary import IndicCanaryFeatureExtractor
    from modeling_indic_canary import IndicCanaryForConditionalGeneration
    from tokenization_indic_canary import IndicCanaryTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Target Device: {device.upper()}")

    # 1. Load tokenizer & feature extractor
    tokenizer = IndicCanaryTokenizer.from_pretrained(model_dir)
    fe = IndicCanaryFeatureExtractor.from_pretrained(model_dir, device=device)

    # 2. Load model in float32 for training stability
    print("Loading model weights for training...")
    model = IndicCanaryForConditionalGeneration.from_pretrained(model_dir, dtype=torch.float32)
    model = model.to(device).train()

    # 3. Read 1 sample from smoke manifest
    manifest_path = "data/manifests/smoke_train_manifest.jsonl"
    with open(manifest_path, "r", encoding="utf-8") as f:
        sample_entry = json.loads(f.readline().strip())

    audio_path = sample_entry["audio_filepath"]
    text = sample_entry["text"]
    speaker_id = sample_entry["speaker_id"]
    duration = sample_entry["duration"]

    print(f"\nTraining Sample:")
    print(f"  Audio File : {os.path.basename(audio_path)}")
    print(f"  Duration   : {duration:.2f}s")
    print(f"  Speaker ID : {speaker_id}")
    print(f"  Reference  : {text}")

    # 4. Build training tensors
    features, feature_lens, decoder_input_ids, labels, prompt_len, text_len = build_canary_training_tensors(
        audio_path, text, fe, tokenizer, device=device
    )

    print(f"\nTensor Shape Telemetry:")
    print(f"  input_features     : {features.shape} (B, T_mel, n_mels)")
    print(f"  feature_lens       : {feature_lens.shape}")
    print(f"  decoder_input_ids  : {decoder_input_ids.shape} (Prompt: {prompt_len}, Text: {text_len})")
    print(f"  labels             : {labels.shape} (Target tokens to compute loss)")

    # 5. Parameter Optimization Strategy
    # To fit within the local 4GB VRAM budget without OOM on AdamW momentum buffers:
    # Freeze FastConformer encoder (811M) + lower 22 decoder layers (384M)
    # Train top 2 Transformer decoder layers + LM head (~35M params)
    print("\nApplying Parameter Optimization Strategy (Parameter-Efficient Top-Layer Tuning):")
    frozen_params = 0
    trainable_params = 0

    # Freeze entire encoder
    for p in model.model.encoder.parameters():
        p.requires_grad = False
        frozen_params += p.numel()

    # Freeze decoder embeddings & lower 22 layers
    for p in model.model.decoder.embedding.parameters():
        p.requires_grad = False
        frozen_params += p.numel()
    for layer in model.model.decoder.layers[:22]:
        for p in layer.parameters():
            p.requires_grad = False
            frozen_params += p.numel()

    # Train top 2 decoder layers
    for layer in model.model.decoder.layers[22:]:
        for p in layer.parameters():
            p.requires_grad = True
            trainable_params += p.numel()

    # Train final LayerNorm & LM head
    for p in model.model.decoder.final_layer_norm.parameters():
        p.requires_grad = True
        trainable_params += p.numel()
    for p in model.lm_head.parameters():
        p.requires_grad = True
        trainable_params += p.numel()

    print(f"  Frozen Parameters    : {frozen_params:,} (~1.19B)")
    print(f"  Trainable Parameters : {trainable_params:,} (~35M)")

    # 6. Initialize Optimizer
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-4, betas=(0.9, 0.98), eps=1e-8
    )
    optimizer.zero_grad()

    # 7. Forward Pass
    print("\nExecuting Forward Pass...")
    # Encoder forward with auto lengths (or attention_mask=None)
    encoder_outputs = model.model.encoder(features, attention_mask=None)
    enc_states = encoder_outputs[0]
    enc_lens = encoder_outputs[1]

    # Cross mask
    cross_mask = model._cross_mask_from_lengths(enc_lens, enc_states.size(1))

    # Decoder forward
    hidden = model.model.decoder(decoder_input_ids, enc_states, cross_mask)
    logits = model.lm_head(hidden)
    print(f"  Logits Output Shape: {logits.shape} (B, Seq_len, Vocab_size)")

    # 8. Compute Canary Cross-Entropy Loss with label smoothing
    vocab_size = tokenizer.vocab_size  # 7152
    loss = F.cross_entropy(
        logits.view(-1, vocab_size),
        labels.view(-1),
        ignore_index=-100,
        label_smoothing=0.1
    )
    print(f"  Computed Loss: {loss.item():.4f}")

    assert not torch.isnan(loss) and not torch.isinf(loss), "Loss computed as NaN or Inf!"
    assert loss.item() > 0, "Loss is non-positive!"

    # 9. Backward Pass
    print("\nExecuting Backward Pass...")
    loss.backward()

    # Check gradient norms
    total_grad_norm = 0.0
    grad_count = 0
    for p in model.parameters():
        if p.requires_grad and p.grad is not None:
            total_grad_norm += p.grad.norm(2).item() ** 2
            grad_count += 1
    total_grad_norm = total_grad_norm ** 0.5
    print(f"  Backprop Successful! Gradient Norm: {total_grad_norm:.4f} across {grad_count} tensors.")
    assert total_grad_norm > 0, "Zero gradients detected!"

    # 10. Optimizer Step
    print("\nExecuting Optimizer Step...")
    optimizer.step()
    print("  Optimizer step completed successfully!")

    # 11. Checkpoint Save & Reload Verification
    os.makedirs("outputs/checkpoints", exist_ok=True)
    checkpoint_path = "outputs/checkpoints/smoke_checkpoint.pt"
    print(f"\nSaving Test Checkpoint to: {checkpoint_path}...")

    trainable_state = {
        name: p.data.cpu() for name, p in model.named_parameters()
        if p.requires_grad
    }
    torch.save({
        "step": 1,
        "loss": loss.item(),
        "grad_norm": total_grad_norm,
        "model_state": trainable_state
    }, checkpoint_path)
    file_size_mb = os.path.getsize(checkpoint_path) / (1024 * 1024)
    print(f"  Checkpoint Saved! File size: {file_size_mb:.2f} MB")

    # Reload test
    print("Testing Checkpoint Reload...")
    loaded_ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert loaded_ckpt["step"] == 1, "Reloaded step mismatch!"
    assert len(loaded_ckpt["model_state"]) == len(trainable_state), "Reloaded state key count mismatch!"
    print("  [PASS] Checkpoint reloaded and verified!")

    # 12. Export Telemetry Report
    out_file = "outputs/smoke_test_result.json"
    smoke_result = {
        "experiment_id": "EXP-004",
        "experiment_name": "one_batch_training_smoke_test",
        "status": "PASS",
        "hardware": {
            "device": device,
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        },
        "training_sample": {
            "audio": os.path.basename(audio_path),
            "duration_s": duration,
            "speaker_id": speaker_id,
            "text": text
        },
        "tensors": {
            "input_features": list(features.shape),
            "decoder_input_ids": list(decoder_input_ids.shape),
            "labels": list(labels.shape),
            "logits": list(logits.shape)
        },
        "metrics": {
            "computed_loss": round(loss.item(), 4),
            "loss_function": "CrossEntropyLoss(ignore_index=-100, label_smoothing=0.1)",
            "gradient_norm": round(total_grad_norm, 4),
            "trainable_parameters": trainable_params,
            "frozen_parameters": frozen_params
        },
        "checkpoint": {
            "path": checkpoint_path,
            "file_size_mb": round(file_size_mb, 2),
            "reload_verified": True
        }
    }

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(smoke_result, f, ensure_ascii=False, indent=2)

    print(f"\n[ARTIFACT EXPORTED] Smoke test telemetry saved to: {out_file}")
    print("=" * 65)
    print("LEVEL 3 1-BATCH SMOKE TEST: ALL GATES PASSED (100% SUCCESS)!")
    print("=" * 65)
    return smoke_result


if __name__ == "__main__":
    run_smoke_test()
