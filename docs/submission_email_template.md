# Submission Email Draft (Concise Academic Format)

**To:** Prof. Mitesh M. Khapra / AI4Bharat Hiring Team  
**Subject:** AI Research Engineer Assignment Submission - Faizan Iqbal  

---

Dear Prof. Mitesh Khapra and the AI4Bharat / Bodhan AI Evaluation Team,

Please find my submission for the AI Research Engineer take-home coding assignment. The objective of this project was to establish an end-to-end, hypothesis-driven fine-tuning pipeline adapting `bodhan-ai/indic-transcribe-core` (1.22B FastConformer-Canary) to the Marathi language (`mr`) using the `ai4bharat/Kathbath` corpus.

### Key Submission Links
* **GitHub Repository:** [https://github.com/Faizaniqbal52/indic-canary-marathi-asr](https://github.com/Faizaniqbal52/indic-canary-marathi-asr)  
  *(Modular Python package in `src/`, reproducible configs in `configs/`, execution scripts, and complete documentation)*
* **Formal Technical Report (PDF Attached):** [Faizan_Iqbal_Technical_Report.pdf](https://github.com/Faizaniqbal52/indic-canary-marathi-asr/raw/main/docs/Faizan_Iqbal_Technical_Report.pdf)  
  *(16-page publication-grade report with 5 empirical figures, mathematical formulations, hardware profiling, and failure analyses)*
* **Google Drive Submission Package:** [https://drive.google.com/drive/folders/1STc13nhFeo2_pD9FL3clTzBXPMskG_a5?usp=drive_link](https://drive.google.com/drive/folders/1STc13nhFeo2_pD9FL3clTzBXPMskG_a5?usp=drive_link)  
  *(Contains serialized LoRA checkpoints at steps 100 and 200, training loss curves JSON, remote Kaggle execution log, 100-sample before/after evaluation JSONs, and 16 kHz audio WAV files)*

---

### Executive Engineering Summary

1. **Working End-to-End Pipeline (EXP-001 to EXP-007):**
   Reverse-engineered the Hugging Face model wrapper to implement custom sequence-to-sequence loss masking over Indic prompt prefixes (`ignore_index = -100`). Scaled from local RTX 3050 (4 GB) to cloud Tesla T4 (15 GB), completing 200 optimization steps (2.0 epochs) with a **-45.60% relative reduction in training loss** (3.0641 to 1.6670, reaching a minimum of 1.6160 at step 170). Checkpoints serialized and reloaded successfully.

2. **Data Leakage & Demographic Integrity:**
   Discovered that the official Kathbath validation set shares all 20 speakers with the training partition. Curated a **strictly speaker-disjoint partition** sampled exclusively from 103 exclusive training speakers (`0 speaker overlap`). Enforced an exact **50.0% Female / 50.0% Male** demographic balance across both training ($N=400$ utterances) and validation ($N=100$ utterances).

3. **Parameter-Efficient Adaptation Benchmark (EXP-005):**
   Benchmarked Top-2 (40.9M params), Top-4 (74.5M params), and Decoder LoRA ($r=16, \alpha=32$). Selected LoRA for adapting all 24 decoder layers with only **3,145,728 trainable parameters (95.8% reduction)** and a compact **12.07 MB checkpoint footprint**, avoiding memory paging and accelerating optimizer latency by $10\times$.

4. **Rigorous Benchmark Evaluation & Acoustic Insights:**
   Evaluated under identical greedy decoding on the frozen 100-sample benchmark. While limited 400-utterance adaptation did not reduce aggregate held-out WER (29.40% to 31.29%), substantial speaker-level WER reductions were observed on previously high-error male voices (**Speaker 421: -9.54% WER reduction**; **Speaker 615: -7.64% WER reduction**), and the demographic gender gap narrowed by **1.20%**. Output invariance was preserved across 62% of evaluation samples.

The attached 16-page technical report details the complete methodology, empirical loss curves, error breakdowns, and ablation analyses. All code, configuration files, and reproduction commands are available in the public repository.

Thank you for your time and consideration. I look forward to discussing the architecture, engineering decisions, and research findings with the committee.

Sincerely,  
**Faizan Iqbal**  
Candidate for AI Research Engineer  
Email: ifaizan041@gmail.com  
GitHub: [https://github.com/Faizaniqbal52](https://github.com/Faizaniqbal52)  
