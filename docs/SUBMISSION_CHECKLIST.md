# SiteSafe — Kaggle Submission Checklist

Final pre-flight before clicking **Submit** on the Gemma 4 Good Hackathon
Kaggle page. Track items here as they're completed.

---

## A. Required Deliverables

- [ ] **Kaggle account** registered and identity verified.
- [ ] **Public code repository** (GitHub) hosting this repo with Apache 2.0 license.
- [ ] **README** at the repo root with quick-start instructions and screenshots.
- [ ] **Working demo** — `python app/sitesafe_app.py` runs, loads in a browser, and analyses a sample image end-to-end.
- [ ] **Technical write-up** — `docs/technical_writeup.md`.
- [ ] **Demo video** — 3–5 minutes, uploaded to YouTube (unlisted or public).
- [ ] **Cover image / media-gallery asset** — banner JPG + 2–3 screenshots.
- [ ] **Kaggle Notebook** publicly visible (`training/finetune_kaggle_notebook.ipynb` re-uploaded as a Kaggle Notebook).

---

## B. Quality Checks

### Repo hygiene
- [ ] `requirements.txt` installs cleanly on Python 3.10 + 3.11.
- [ ] `bash setup.sh` succeeds end-to-end on a fresh clone (Linux + macOS + WSL).
- [ ] `python data/build_osha_db.py` writes `data/osha_regulations.db` and the smoke test passes.
- [ ] `data/download_datasets.sh` either downloads sample images successfully or prints clear instructions when offline.
- [ ] `python app/sitesafe_app.py` launches Gradio on `http://localhost:7860`.

### Application
- [ ] App handles "no Ollama running" with a clear, actionable error.
- [ ] App handles "Ollama running but model not registered" with a clear remediation step.
- [ ] App handles oversized / unreadable / 0-byte images gracefully.
- [ ] Loading spinner appears during inference; no UI freeze.
- [ ] PDF export works on at least one report and produces a 1–2-page document with the embedded photo.
- [ ] Webcam capture works in Chrome, Firefox, Safari.

### Knowledge base
- [ ] `regulations` table has **≥ 50 rows** spanning Subparts C, E, F, G, H, K, L, M, N, O, P, X.
- [ ] All four Fatal Four categories have at least 3 mapped regulations.
- [ ] Penalty figures match the OSHA 2024 inflation-adjusted schedule.
- [ ] Function-calling tools (`lookup_regulation`, `get_fatal_four_info`, `get_penalty_info`) all return non-empty results for at least one query each.

### Training pipeline
- [ ] `training/finetune_gemma4_e4b.py --help` lists all hyperparameters.
- [ ] `finetune_kaggle_notebook.ipynb` runs end-to-end on free Kaggle T4 within 30 minutes.
- [ ] OOM fallback path (text-only LoRA) is documented and verified.
- [ ] GGUF export succeeds (q4_k_m).

### Evaluation
- [ ] `python evaluation/evaluate.py --test-jsonl ...` runs over a labelled set and writes `metrics.json` + `results.md` + `confusion_matrix.json` + `calibration_pairs.csv`.
- [ ] `python evaluation/plot_calibration.py --input ...` writes `reliability_diagram.png` showing raw vs. calibrated curves.
- [ ] Reported metrics include macro recall, FPR, citation accuracy, ECE, and mean latency.

---

## C. Narrative — what the judges will read

- [ ] Problem statement leads with **the Fatal Four statistic** ("one death every eight hours").
- [ ] "Why Gemma 4" ties to all five differentiators: multimodal, function calling, edge deployment, fine-tuning, Apache 2.0 license.
- [ ] Impact section quantifies expected lives saved with conservative assumptions.
- [ ] Technical depth is visible: Unsloth + LoRA + Platt scaling + reliability diagram.
- [ ] Future work is realistic and exciting (Spanish, mobile app, OSHA general industry).
- [ ] Disclaimers are explicit — SiteSafe assists, does not replace inspections.

---

## D. Demo Video Outline (target: 3–5 minutes)

1. **Hook (0:00–0:20)** — "Every eight hours, a U.S. construction worker dies. Most of those deaths are preventable. The problem isn't enforcement — OSHA inspects 0.004% of sites a year. The problem is **getting compliance feedback to the foreman, on the site, today.**"
2. **Demo (0:20–1:50)** — Live walkthrough: photo upload → analyze → structured report → PDF export. Show one Fatal Four violation per category if possible.
3. **Architecture (1:50–3:00)** — Diagram. "Gemma 4 E4B, fine-tuned with Unsloth, function-calling against a local OSHA SQLite DB, calibrated confidences. **All offline.**"
4. **Reproducibility (3:00–4:00)** — Show the Kaggle notebook training and the GGUF export. Show `docker-compose up`.
5. **Impact + close (4:00–5:00)** — Lives saved estimate. Spanish + mobile roadmap. "If you build, inspect, or supervise — fork the repo. It's Apache 2.0."

---

## E. Cover Image / Media Gallery

- [ ] **Banner image** (1500×500 or similar) — SiteSafe logo + tagline + a clean job-site photo.
- [ ] **Screenshot 1** — Gradio UI with a violation report rendered.
- [ ] **Screenshot 2** — PDF export thumbnail.
- [ ] **Screenshot 3** — Reliability diagram from the technical write-up.

---

## F. Final 24-Hour Pre-Flight

- [ ] All `[BLOCKS]` items in this checklist are checked.
- [ ] All `[TUNE]` items have at least been considered.
- [ ] PR title and submission abstract are < 280 characters and front-load the impact.
- [ ] Submission references the **Global Resilience track** explicitly.
- [ ] Repository is set to **public** (with the Apache 2.0 LICENSE file present).
- [ ] Demo video is **viewable** (not private; not still processing).
- [ ] Kaggle Notebook is **published** (not draft).
- [ ] Repository tag/release `v1.0` cut so the submission is reproducible at a fixed commit.

🦺 Good luck.
