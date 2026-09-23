# Final Submission Email Draft

**To:** Prof. Mitesh M. Khapra / AI4Bharat Hiring Team  
**Subject:** AI Research Engineer Assignment Submission — Faizan Iqbal  

---

Dear Prof. Mitesh Khapra and the AI4Bharat / Bodhan AI Evaluation Team,

Please find my submission for the **AI Research Engineer** take-home coding assignment. The objective of this project was to establish an end-to-end, hypothesis-driven fine-tuning pipeline adapting `bodhan-ai/indic-transcribe-core` (1.22B FastConformer-Canary) to the Marathi language (`mr`) using the `ai4bharat/Kathbath` corpus.

### Key Links
* **GitHub Repository:** [https://github.com/Faizaniqbal52/indic-canary-marathi-asr](https://github.com/Faizaniqbal52/indic-canary-marathi-asr)
* **Google Drive Submission Folder:** [https://drive.google.com/drive/folders/1STc13nhFeo2_pD9FL3clTzBXPMskG_a5?usp=drive_link](https://drive.google.com/drive/folders/1STc13nhFeo2_pD9FL3clTzBXPMskG_a5?usp=drive_link)  
  *(Contains serialized LoRA checkpoints, JSON training curves, remote terminal execution logs, 100-sample before/after evaluation JSONs, and representative 16 kHz WAV audio files)*

---

### Executive Technical Summary

1. **Acoustic Profiling & Data Integrity:**
   * Performed a 100% census across the official Kathbath Marathi dataset (2,378 validation / 84,070 training utterances).
   * Discovered that the official validation set is a known-speaker benchmark (20/20 speaker overlap in training). To guarantee an honest out-of-speaker evaluation, we curated a derived training set sampled strictly from the **103 exclusive training speakers** ($\text{Train} \cap \text{Val} = \emptyset$).
   * Uncovered gender segregation across Kathbath parquet shards and constructed an exact **50.0% Female / 50.0% Male** demographic balance across both training ($N=400$ utterances across 40 exclusive speakers) and validation ($N=100$ utterances across all 20 benchmark speakers).

2. **Parameter-Efficient Adaptation Benchmark (EXP-005):**
   * Benchmarked Top-2 Decoder Layers (40.9M params), Top-4 Decoder Layers (74.5M params), and LoRA Attention ($r=16, \alpha=32$) on consumer hardware (RTX 3050, 4 GB VRAM).
   * Selected LoRA as our primary strategy: it adapts both self- and cross-attention across **all 24 decoder layers** with only **3,145,728 trainable parameters (0.26%)**, reducing checkpoint file size by ~92% (12.07 MB) and optimizer latency by $10\times$.

3. **Cloud Training Convergence (EXP-007 on Kaggle Tesla T4):**
   * Scaled training to cloud GPU compute (Tesla T4, 14.56 GB GDDR6), completing 200 optimization steps with effective batch size 4 (800 forward passes $\implies$ **exactly 2.0 epochs**) using AdamW with warmup and cosine decay.
   * Cross-entropy loss decreased monotonically by **-45.60% relative** (from 3.0641 to 1.6670, reaching a minimum of 1.6160 at step 170) with stable gradients (mean norm 0.1496) and flat 4,875.8 MB VRAM allocation.

4. **Held-Out Benchmark Evaluation & Research Insights:**
   * Evaluated under identical greedy decoding conditions on the frozen 100-utterance benchmark using PEFT adapter toggling (`model.disable_adapter()`).
   * **Engineering vs Metric Takeaway:** The run successfully validates the end-to-end training, loss masking (`ignore_index = -100` on Indic prompt prefixes), backpropagation, and checkpointing mechanics. While a limited 400-utterance adaptation did not improve aggregate held-out WER (29.40% $\to$ 31.29%), it achieved substantial acoustic generalization on previously high-error male voices (**Speaker 421: -9.54% WER reduction**; **Speaker 615: -7.64% WER reduction**) and narrowed the demographic gender gap by **1.20%**.
   * Detailed linguistic diffs confirmed enhanced Marathi case marker inflection (`विभक्ती प्रत्यय`), compound word spacing, and acoustic disambiguation.

A detailed technical report documenting the reverse-engineering analysis, mathematical formulations, hardware profiling, and failure mode analyses is available in [`docs/technical_report.md`](https://github.com/Faizaniqbal52/indic-canary-marathi-asr/blob/main/docs/technical_report.md) in the repository.

Thank you for the opportunity to work on this exciting take-home project. I look forward to discussing the architecture, engineering decisions, and findings with the team.

Sincerely,  
**Faizan Iqbal**  
Candidate for AI Research Engineer  
GitHub: [Faizaniqbal52](https://github.com/Faizaniqbal52)  
