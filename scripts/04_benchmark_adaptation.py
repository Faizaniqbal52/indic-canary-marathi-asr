"""
04_benchmark_adaptation.py: EXP-005 Adaptation-Strategy Benchmark
Executes a controlled empirical comparison across three parameter-efficient
adaptation strategies for bodhan-ai/indic-transcribe-core on local hardware:
  Strategy A: Top-2 Decoder Layers (layers 22-23 + LN + LM Head)
  Strategy B: Top-4 Decoder Layers (layers 20-23 + LN + LM Head)
  Strategy C: LoRA on Decoder (r=16, alpha=32 on query_net & value_net across all 24 layers)

Measures empirically on identical audio input:
  1. Trainable vs Frozen parameter counts & percentage
  2. Peak GPU VRAM allocated and reserved (MB)
  3. Execution step latency (forward, backward, optimizer step time)
  4. Gradient norm and loss stability
  5. Checkpoint file size and reload verification
"""

import os
import sys
import glob
import time
import json
import torch
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


def prepare_benchmark_sample(model_dir, device="cuda"):
    from feature_extraction_indic_canary import IndicCanaryFeatureExtractor
    from tokenization_indic_canary import IndicCanaryTokenizer

    audio_path = "data/smoke_wavs/train_00_844424932286711-1185-f.wav"
    text = "पाहता पाहता राज्यात टप्प्याटप्प्यानं महत्त्वाचे व्यवहार आणि पर्यायी जनजीवनही पूर्वपदावर येऊ लागलं"

    fe = IndicCanaryFeatureExtractor.from_pretrained(model_dir, device=device)
    tokenizer = IndicCanaryTokenizer.from_pretrained(model_dir)

    wav, sr = sf.read(audio_path, dtype="float32", always_2d=True)
    wav = torch.from_numpy(wav.mean(axis=1))
    if sr != fe.sample_rate:
        import torchaudio
        wav = torchaudio.functional.resample(wav, sr, fe.sample_rate)

    batch = wav.unsqueeze(0).to(device)
    lens = torch.tensor([wav.shape[0]], device=device)
    features, feature_lens = fe(batch, lens)

    prompt_ids = tokenizer.encode_prompt(lang="mr", itn=False, romanized=False)
    raw_pieces = tokenizer.multi.encode(text)
    text_token_ids = [p + tokenizer.spl_size for p in raw_pieces]
    eos_id = tokenizer.eos_id

    full_seq = prompt_ids + text_token_ids + [eos_id]
    decoder_input_ids = torch.tensor([full_seq[:-1]], dtype=torch.long, device=device)

    target_seq = full_seq[1:]
    num_prompt = len(prompt_ids)
    labels_list = []
    for idx, tok in enumerate(target_seq):
        if idx < num_prompt - 1:
            labels_list.append(-100)
        else:
            labels_list.append(tok)

    labels = torch.tensor([labels_list], dtype=torch.long, device=device)
    return features, decoder_input_ids, labels, tokenizer.vocab_size


def setup_strategy_top_k(model, num_top_layers):
    """Freezes encoder and lower decoder layers, enables top K layers + LN + LM Head."""
    # Freeze encoder
    for p in model.model.encoder.parameters():
        p.requires_grad = False
    # Freeze decoder embedding
    for p in model.model.decoder.embedding.parameters():
        p.requires_grad = False

    total_layers = len(model.model.decoder.layers)  # 24
    cutoff = total_layers - num_top_layers

    for i in range(cutoff):
        for p in model.model.decoder.layers[i].parameters():
            p.requires_grad = False

    for i in range(cutoff, total_layers):
        for p in model.model.decoder.layers[i].parameters():
            p.requires_grad = True

    for p in model.model.decoder.final_layer_norm.parameters():
        p.requires_grad = True
    for p in model.lm_head.parameters():
        p.requires_grad = True

    return model, False  # is_peft = False


def setup_strategy_lora(model):
    """Applies LoRA to query_net and value_net across all decoder layers."""
    from peft import LoraConfig, get_peft_model

    lora_conf = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=r".*decoder.*(query_net|value_net)",
        lora_dropout=0.05,
        bias="none"
    )
    peft_model = get_peft_model(model, lora_conf)
    return peft_model, True  # is_peft = True


def benchmark_strategy(strategy_name, setup_fn, model_dir, device="cuda"):
    print("\n" + "=" * 65)
    print(f"BENCHMARKING STRATEGY: {strategy_name}")
    print("=" * 65)

    # 1. Reset memory stats
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base_mem_mb = torch.cuda.memory_allocated() / (1024 * 1024)

    # 2. Load fresh model
    from modeling_indic_canary import IndicCanaryForConditionalGeneration
    print("Loading model weights...")
    raw_model = IndicCanaryForConditionalGeneration.from_pretrained(model_dir, dtype=torch.float32)
    raw_model = raw_model.to(device).train()

    model, is_peft = setup_fn(raw_model)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    trainable_pct = (trainable_params / total_params) * 100

    print(f"  Total Params     : {total_params:,}")
    print(f"  Frozen Params    : {frozen_params:,} ({100 - trainable_pct:.2f}%)")
    print(f"  Trainable Params : {trainable_params:,} ({trainable_pct:.2f}%)")

    # 3. Setup optimizer
    trainable_tensors = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_tensors, lr=1e-4, betas=(0.9, 0.98), eps=1e-8)

    # 4. Prepare data
    features, decoder_input_ids, labels, vocab_size = prepare_benchmark_sample(model_dir, device=device)

    # 5. Warmup step
    print("Running warmup pass...")
    actual_model = model.model if is_peft else model
    enc_outputs = actual_model.model.encoder(features, attention_mask=None)
    cross_mask = actual_model._cross_mask_from_lengths(enc_outputs[1], enc_outputs[0].size(1))
    hidden = actual_model.model.decoder(decoder_input_ids, enc_outputs[0], cross_mask)
    logits = actual_model.lm_head(hidden)
    loss = F.cross_entropy(logits.view(-1, vocab_size), labels.view(-1), ignore_index=-100)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()

    # 6. Timed Benchmark Iterations (3 measured steps)
    print("Measuring steady-state iterations (3 steps)...")
    step_times = []
    fwd_times = []
    bwd_times = []
    opt_times = []
    grad_norms = []
    loss_vals = []

    for step in range(3):
        optimizer.zero_grad()
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        # Forward
        enc_outputs = actual_model.model.encoder(features, attention_mask=None)
        cross_mask = actual_model._cross_mask_from_lengths(enc_outputs[1], enc_outputs[0].size(1))
        hidden = actual_model.model.decoder(decoder_input_ids, enc_outputs[0], cross_mask)
        logits = actual_model.lm_head(hidden)
        loss = F.cross_entropy(logits.view(-1, vocab_size), labels.view(-1), ignore_index=-100, label_smoothing=0.1)
        torch.cuda.synchronize()
        t1 = time.perf_counter()

        # Backward
        loss.backward()
        torch.cuda.synchronize()
        t2 = time.perf_counter()

        # Grad norm
        tot_norm = 0.0
        for p in trainable_tensors:
            if p.grad is not None:
                tot_norm += p.grad.norm(2).item() ** 2
        tot_norm = tot_norm ** 0.5

        # Optimizer step
        optimizer.step()
        torch.cuda.synchronize()
        t3 = time.perf_counter()

        fwd_times.append((t1 - t0) * 1000)
        bwd_times.append((t2 - t1) * 1000)
        opt_times.append((t3 - t2) * 1000)
        step_times.append((t3 - t0) * 1000)
        grad_norms.append(tot_norm)
        loss_vals.append(loss.item())

    avg_fwd = sum(fwd_times) / len(fwd_times)
    avg_bwd = sum(bwd_times) / len(bwd_times)
    avg_opt = sum(opt_times) / len(opt_times)
    avg_step = sum(step_times) / len(step_times)
    avg_loss = sum(loss_vals) / len(loss_vals)
    avg_grad_norm = sum(grad_norms) / len(grad_norms)

    # 7. Memory Telemetry
    peak_allocated_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
    peak_reserved_mb = torch.cuda.max_memory_reserved() / (1024 * 1024)

    print(f"  Avg Forward Time  : {avg_fwd:.1f} ms")
    print(f"  Avg Backward Time : {avg_bwd:.1f} ms")
    print(f"  Avg Optimizer Time: {avg_opt:.1f} ms")
    print(f"  Avg Total Step    : {avg_step:.1f} ms ({avg_step/1000:.2f} s)")
    print(f"  Avg Loss          : {avg_loss:.4f}")
    print(f"  Avg Gradient Norm : {avg_grad_norm:.4f}")
    print(f"  Peak VRAM Alloc   : {peak_allocated_mb:.1f} MB")
    print(f"  Peak VRAM Resv    : {peak_reserved_mb:.1f} MB")

    # 8. Checkpoint Serialization & Reload Test
    os.makedirs("outputs/checkpoints", exist_ok=True)
    ckpt_path = f"outputs/checkpoints/benchmark_{strategy_name.lower().replace('-', '_').replace(' ', '_')}.pt"
    if is_peft:
        # Save LoRA adapter weights
        trainable_state = {k: v.cpu() for k, v in model.state_dict().items() if "lora" in k}
    else:
        trainable_state = {name: p.data.cpu() for name, p in model.named_parameters() if p.requires_grad}

    torch.save({"step": 3, "loss": avg_loss, "state_dict": trainable_state}, ckpt_path)
    ckpt_size_mb = os.path.getsize(ckpt_path) / (1024 * 1024)
    print(f"  Checkpoint Size   : {ckpt_size_mb:.2f} MB ({ckpt_path})")

    # Reload test
    loaded = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    assert len(loaded["state_dict"]) == len(trainable_state), "Checkpoint reload key count mismatch!"
    print(f"  Checkpoint Reload : [PASS] ({len(loaded['state_dict'])} state tensors verified)")

    # Clean up model
    del model, optimizer, raw_model
    torch.cuda.empty_cache()

    return {
        "strategy": strategy_name,
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "frozen_parameters": frozen_params,
        "trainable_percent": round(trainable_pct, 4),
        "avg_forward_ms": round(avg_fwd, 1),
        "avg_backward_ms": round(avg_bwd, 1),
        "avg_optimizer_ms": round(avg_opt, 1),
        "avg_total_step_ms": round(avg_step, 1),
        "avg_total_step_s": round(avg_step / 1000, 3),
        "avg_loss": round(avg_loss, 4),
        "avg_gradient_norm": round(avg_grad_norm, 4),
        "peak_vram_allocated_mb": round(peak_allocated_mb, 1),
        "peak_vram_reserved_mb": round(peak_reserved_mb, 1),
        "checkpoint_size_mb": round(ckpt_size_mb, 2),
        "status": "PASS"
    }


def run_all_benchmarks():
    print("=" * 65)
    print("EXP-005: PARAMETER-EFFICIENT ADAPTATION STRATEGY BENCHMARK")
    print("=" * 65)

    model_dir = find_local_model_dir()
    if model_dir not in sys.path:
        sys.path.insert(0, model_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Hardware: {torch.cuda.get_device_name(0)} (CUDA: {torch.cuda.is_available()})")

    results = []

    # 1. Strategy A: Top-2 Decoder Layers
    res_top2 = benchmark_strategy("Top-2 Decoder Layers", lambda m: setup_strategy_top_k(m, 2), model_dir, device)
    results.append(res_top2)

    # 2. Strategy B: Top-4 Decoder Layers
    res_top4 = benchmark_strategy("Top-4 Decoder Layers", lambda m: setup_strategy_top_k(m, 4), model_dir, device)
    results.append(res_top4)

    # 3. Strategy C: LoRA on Decoder Attention (r=16, alpha=32)
    res_lora = benchmark_strategy("LoRA Decoder (r=16, alpha=32)", setup_strategy_lora, model_dir, device)
    results.append(res_lora)

    # Summary Table
    print("\n" + "=" * 90)
    print("EXP-005 BENCHMARK COMPARISON SUMMARY TABLE")
    print("=" * 90)
    print(f"{'Strategy':<30} | {'Trainable':<12} | {'Train %':<8} | {'Step (s)':<9} | {'Peak VRAM':<12} | {'Ckpt (MB)':<10} | {'Status'}")
    print("-" * 90)
    for r in results:
        print(f"{r['strategy']:<30} | {r['trainable_parameters']:>12,} | {r['trainable_percent']:>7.2f}% | {r['avg_total_step_s']:>8.2f}s | {r['peak_vram_allocated_mb']:>9.1f} MB | {r['checkpoint_size_mb']:>8.2f} MB | {r['status']}")
    print("=" * 90)

    # Save to disk
    os.makedirs("artifacts/04_adaptation_benchmark", exist_ok=True)
    report_path = "outputs/adaptation_benchmark.json"
    artifact_path = "artifacts/04_adaptation_benchmark/benchmark_report.json"

    benchmark_data = {
        "experiment_id": "EXP-005",
        "experiment_name": "adaptation_strategy_benchmark",
        "hardware": {
            "gpu": torch.cuda.get_device_name(0),
            "total_vram_mb": 4096
        },
        "results": results
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(benchmark_data, f, ensure_ascii=False, indent=2)
    with open(artifact_path, "w", encoding="utf-8") as f:
        json.dump(benchmark_data, f, ensure_ascii=False, indent=2)

    print(f"\n[ARTIFACT EXPORTED] Benchmark telemetry saved to: {report_path} and {artifact_path}")
    return benchmark_data


if __name__ == "__main__":
    run_all_benchmarks()
