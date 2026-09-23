import json

with open('artifacts/07_finetune_results/finetune_analysis_report.json', encoding='utf-8') as f:
    d = json.load(f)

lines = []
lines.append("# Qualitative Transcription Diff Analysis (Held-Out Marathi Benchmark)")
lines.append("")
lines.append("## 1. Improved Utterances (Fine-Tuning Outperformed Zero-Shot Base)")
lines.append("")

for s in d['sample_improvements'][:5]:
    bw = s['wer_base_norm'] * 100
    fw = s['wer_finetuned_norm'] * 100
    lines.append(f"### Sample {s['sample_id']} (Speaker: {s['speaker_id']}, Gender: {s['gender']})")
    lines.append(f"* **Reference:** `{s['reference']}`")
    lines.append(f"* **Base Model:** `{s['pred_base']}` (WER: **{bw:.1f}%**)")
    lines.append(f"* **Fine-Tuned:** `{s['pred_finetuned']}` (WER: **{fw:.1f}%**, $\Delta = {fw-bw:+.1f}\\%$)")
    lines.append("")

lines.append("## 2. Shifted / Degraded Utterances (Distribution Drift Examples)")
lines.append("")

for s in d['sample_degradations'][:5]:
    bw = s['wer_base_norm'] * 100
    fw = s['wer_finetuned_norm'] * 100
    lines.append(f"### Sample {s['sample_id']} (Speaker: {s['speaker_id']}, Gender: {s['gender']})")
    lines.append(f"* **Reference:** `{s['reference']}`")
    lines.append(f"* **Base Model:** `{s['pred_base']}` (WER: **{bw:.1f}%**)")
    lines.append(f"* **Fine-Tuned:** `{s['pred_finetuned']}` (WER: **{fw:.1f}%**, $\Delta = {fw-bw:+.1f}\\%$)")
    lines.append("")

with open('artifacts/07_finetune_results/qualitative_analysis.md', 'w', encoding='utf-8') as out:
    out.write('\n'.join(lines))

print("Saved qualitative analysis to artifacts/07_finetune_results/qualitative_analysis.md")
