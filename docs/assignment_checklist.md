# Assignment Submission Checklist & Contract

**Role:** AI Research Engineer — Bodhan AI / AI4Bharat (IIT Madras)  
**Supervisor:** Prof. Mitesh Khapra  
**Submission Deadline:** September 24, 2026 by 6:00 PM IST  
**Status:** In Progress (Level 1 Verified, Level 2 Profiling)

---

## 1. Assignment Mandate & Deliverables

| Deliverable | Requirement Description | Status | Verification Artifact |
| :--- | :--- | :---: | :--- |
| **Model Selection** | Select ONE Bodhan AI model (MT, ASR, or TTS) | [x] COMPLETE | Selected: `bodhan-ai/indic-transcribe-core` (1.22B FastConformer) |
| **Target Language** | Fine-tune on Marathi or Bhili | [x] COMPLETE | Target: Marathi (`mr`) |
| **Dataset Selection** | Approved corpus from AIKosh or Hugging Face | [x] COMPLETE | Selected: `ai4bharat/Kathbath` (Marathi split) |
| **End-to-End Pipeline** | Demonstrate working fine-tuning run from data to checkpoint | [x] COMPLETE | EXP-001 through EXP-007 verified; checkpoints serialized & reloaded |
| **GitHub Repository** | Clean, modular codebase (not a monolithic OA notebook) | [x] COMPLETE | Structured in `src/`, `configs/`, `scripts/`, `docs/`, `kaggle/` |
| **Google Drive Artifacts** | Checkpoints, training logs, predictions, qualitative audio | [x] COMPLETE | Packaged in `artifacts/07_finetune_results/` & `outputs/checkpoints/` |
| **Technical Documentation** | Comprehensive approach report: choices, rationale, challenges | [x] COMPLETE | Finalized in `docs/technical_report.md` & `README.md` |

---

## 2. Evaluation Criteria Contract (What the Panel Evaluates)

> *"This assignment is not scored on model performance or evaluation metrics. We want to see whether you can get a fine-tuning pipeline working end-to-end, how you structure and write your code, and the effort, judgement, and problem-solving you bring to the training process."*

- [x] **Criterion 1: Pipeline Integrity** — Establish verifiable end-to-end flow from raw data ingestion to loss convergence and inference.
- [x] **Criterion 2: Code Architecture** — Modular Python package structure with separation of concerns (`data`, `models`, `training`, `evaluation`).
- [x] **Criterion 3: Engineering Judgement** — Profile compute constraints (4GB VRAM local vs 15GB+ cloud), inspect architecture limitations (inference-only HF wrapper discovered), and audit dataset splits before training.
- [x] **Criterion 4: Reproducibility** — Explicit configuration files (`configs/*.yaml`), deterministic random seeds, pinned dependencies (`requirements.txt`), and step-by-step reproduction instructions.
- [x] **Criterion 5: Transparent Failure Analysis** — Document all real-world obstacles encountered (gated authentication, Windows DLL dependencies, model port restrictions) and their engineering solutions.

---

## 3. Submission Protocol Checkpoints (Before 24th Sept, 6:00 PM)

- [ ] GitHub repository pushed, public/accessible, clean `.gitignore` (no credentials, no 5GB binary checkpoints).
- [ ] Google Drive folder created with public read access ("Anyone with the link can view").
- [ ] Google Drive verified in Incognito mode to ensure downloadability of checkpoints and logs.
- [ ] Technical documentation compiled into `docs/technical_report.md` and exported as a clean PDF if needed.
- [ ] Formal email reply prepared with GitHub link, Google Drive link, and executive summary.
