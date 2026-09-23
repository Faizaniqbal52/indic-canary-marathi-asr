"""
07_analyze_finetune_results.py
Analyzes training curves and held-out evaluation results from the Kaggle fine-tuning run.
Generates:
1. Training convergence telemetry (initial vs final loss, grad norm stability, epoch milestones)
2. Aggregate WER/CER comparative metrics (Base vs Fine-Tuned, Raw vs Normalized)
3. Demographic stratification analysis (Female vs Male WER deltas)
4. Speaker-level consistency metrics across all 20 held-out speakers
5. Qualitative transcription diffs (improvements, regressions, phonetic subtleties)
6. Markdown summary tables ready for technical report and experiment log
"""

import json
import os
import sys

def analyze(results_dir="kaggle/output_finetune", output_dir="artifacts/07_finetune_results"):
    os.makedirs(output_dir, exist_ok=True)
    
    curves_file = os.path.join(results_dir, "training_curves.json")
    eval_file = os.path.join(results_dir, "final_evaluation_results.json")
    
    if not os.path.exists(curves_file) or not os.path.exists(eval_file):
        # Check subdirectories
        for root, dirs, files in os.walk(results_dir):
            if "training_curves.json" in files:
                curves_file = os.path.join(root, "training_curves.json")
            if "final_evaluation_results.json" in files:
                eval_file = os.path.join(root, "final_evaluation_results.json")

    print(f"Loading training curves from: {curves_file}")
    print(f"Loading evaluation results from: {eval_file}")

    if not os.path.exists(curves_file) or not os.path.exists(eval_file):
        print(f"ERROR: Files not found in {results_dir}")
        return

    with open(curves_file, "r", encoding="utf-8") as f:
        curves = json.load(f)
    with open(eval_file, "r", encoding="utf-8") as f:
        eval_data = json.load(f)

    # 1. Training Convergence Analysis
    initial_loss = curves[0]["loss"]
    final_loss = curves[-1]["loss"]
    loss_reduction_pct = ((initial_loss - final_loss) / initial_loss) * 100
    mean_grad_norm = sum(c["grad_norm"] for c in curves) / len(curves)
    peak_vram = max(c["vram_mb"] for c in curves)
    total_time_s = curves[-1]["elapsed_s"]

    print("=" * 75)
    print("TRAINING CONVERGENCE SUMMARY (EXP-007)")
    print("=" * 75)
    print(f"Total Steps            : {curves[-1]['step']} (Effective Batch Size = 4, Total Utterances = 800)")
    print(f"Initial Step Loss      : {initial_loss:.4f}")
    print(f"Final Step Loss        : {final_loss:.4f} (-{loss_reduction_pct:.2f}% relative drop)")
    print(f"Mean Gradient Norm     : {mean_grad_norm:.4f}")
    print(f"Peak VRAM Allocated    : {peak_vram:.1f} MB (Tesla T4)")
    print(f"Total Training Runtime : {total_time_s:.1f} s ({total_time_s/60:.2f} min)")

    # 2. Benchmark Evaluation Analysis
    samples = eval_data["detailed_results"]
    n_samples = len(samples)
    
    mean_wer_base = eval_data["mean_wer_base"]
    mean_wer_ft = eval_data["mean_wer_finetuned"]
    mean_wer_base_norm = eval_data["mean_wer_base_norm"]
    mean_wer_ft_norm = eval_data["mean_wer_finetuned_norm"]
    mean_cer_base = eval_data["mean_cer_base"]
    mean_cer_ft = eval_data["mean_cer_finetuned"]
    
    delta_wer = (mean_wer_ft - mean_wer_base) * 100
    delta_wer_norm = (mean_wer_ft_norm - mean_wer_base_norm) * 100
    delta_cer = (mean_cer_ft - mean_cer_base) * 100

    print("\n" + "=" * 75)
    print("HELD-OUT BENCHMARK EVALUATION (N = 100 Unseen Utterances, 20 Speakers)")
    print("=" * 75)
    print(f"Metric                 | Base Model | Fine-Tuned (LoRA) | Delta")
    print(f"-----------------------|------------|-------------------|-------")
    print(f"Raw WER                | {mean_wer_base*100:9.2f}% | {mean_wer_ft*100:16.2f}% | {delta_wer:+5.2f}%")
    print(f"Normalized WER         | {mean_wer_base_norm*100:9.2f}% | {mean_wer_ft_norm*100:16.2f}% | {delta_wer_norm:+5.2f}%")
    print(f"Raw CER                | {mean_cer_base*100:9.2f}% | {mean_cer_ft*100:16.2f}% | {delta_cer:+5.2f}%")

    # 3. Demographic Breakdown
    female_base = eval_data["gender_breakdown"]["female_wer_base"]
    female_ft = eval_data["gender_breakdown"]["female_wer_finetuned"]
    male_base = eval_data["gender_breakdown"]["male_wer_base"]
    male_ft = eval_data["gender_breakdown"]["male_wer_finetuned"]

    print("\n" + "=" * 75)
    print("GENDER-STRATIFIED BREAKDOWN (50 Female / 50 Male Utterances)")
    print("=" * 75)
    print(f"Sub-group              | Base WER   | Fine-Tuned WER    | Delta")
    print(f"-----------------------|------------|-------------------|-------")
    print(f"Female Speakers (N=50) | {female_base*100:9.2f}% | {female_ft*100:16.2f}% | {(female_ft - female_base)*100:+5.2f}%")
    print(f"Male Speakers (N=50)   | {male_base*100:9.2f}% | {male_ft*100:16.2f}% | {(male_ft - male_base)*100:+5.2f}%")

    # 4. Speaker-Level Breakdown
    speaker_stats = {}
    for s in samples:
        spk = s["speaker_id"]
        if spk not in speaker_stats:
            speaker_stats[spk] = {"gender": s["gender"], "base_wers": [], "ft_wers": []}
        speaker_stats[spk]["base_wers"].append(s["wer_base_norm"])
        speaker_stats[spk]["ft_wers"].append(s["wer_finetuned_norm"])

    print("\n" + "=" * 75)
    print(f"SPEAKER-LEVEL ROBUSTNESS (All 20 Held-Out Benchmark Speakers)")
    print("=" * 75)
    print(f"Speaker ID | Gender | Utterances | Base Norm WER | FT Norm WER | Delta")
    print(f"-----------|--------|------------|---------------|-------------|-------")
    for spk, data in sorted(speaker_stats.items()):
        b_wer = sum(data["base_wers"]) / len(data["base_wers"])
        f_wer = sum(data["ft_wers"]) / len(data["ft_wers"])
        d = (f_wer - b_wer) * 100
        print(f"Spk {str(spk):7s} | {data['gender']:6s} | {len(data['base_wers']):10d} | {b_wer*100:12.2f}% | {f_wer*100:10.2f}% | {d:+5.2f}%")

    # 5. Categorize Sample Outcomes: Improved, Unchanged, Degraded
    improved = [s for s in samples if s["wer_finetuned_norm"] < s["wer_base_norm"]]
    unchanged = [s for s in samples if s["wer_finetuned_norm"] == s["wer_base_norm"]]
    degraded = [s for s in samples if s["wer_finetuned_norm"] > s["wer_base_norm"]]

    print("\n" + "=" * 75)
    print(f"SAMPLE BEHAVIOR DISTRIBUTION (N = {n_samples})")
    print(f"  - Improved Transcript Fidelity : {len(improved)} / {n_samples} ({len(improved)/n_samples*100:.1f}%)")
    print(f"  - Unchanged (Identical Metric) : {len(unchanged)} / {n_samples} ({len(unchanged)/n_samples*100:.1f}%)")
    print(f"  - Degraded / Shifted          : {len(degraded)} / {n_samples} ({len(degraded)/n_samples*100:.1f}%)")
    print("=" * 75)

    # Save comprehensive analysis report
    analysis_report = {
        "training_convergence": {
            "total_steps": curves[-1]["step"],
            "initial_loss": initial_loss,
            "final_loss": final_loss,
            "loss_reduction_pct": round(loss_reduction_pct, 2),
            "mean_grad_norm": round(mean_grad_norm, 4),
            "peak_vram_mb": peak_vram,
            "total_time_s": total_time_s
        },
        "benchmark_metrics": {
            "mean_wer_base": mean_wer_base,
            "mean_wer_finetuned": mean_wer_ft,
            "mean_wer_base_norm": mean_wer_base_norm,
            "mean_wer_finetuned_norm": mean_wer_ft_norm,
            "mean_cer_base": mean_cer_base,
            "mean_cer_finetuned": mean_cer_ft
        },
        "gender_stratification": {
            "female_wer_base": female_base,
            "female_wer_finetuned": female_ft,
            "male_wer_base": male_base,
            "male_wer_finetuned": male_ft
        },
        "outcome_distribution": {
            "improved": len(improved),
            "unchanged": len(unchanged),
            "degraded": len(degraded)
        },
        "sample_improvements": improved[:5],
        "sample_degradations": degraded[:5]
    }

    report_path = os.path.join(output_dir, "finetune_analysis_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(analysis_report, f, indent=2, ensure_ascii=False)
    print(f"\nAnalysis report saved to: {report_path}")

if __name__ == "__main__":
    analyze()
