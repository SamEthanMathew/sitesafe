```
   _____  _  _        _____         __
  / ____|(_)| |      / ____|       / _|
 | (___   _ | |_  ___| (___   __ _ | |_  ___
  \___ \ | || __|/ _ \\___ \ / _` ||  _|/ _ \
  ____) || || |_|  __/____) | (_| || | |  __/
 |_____/ |_| \__|\___|_____/ \__,_||_|  \___|

  On-Device OSHA Construction Safety Assistant
```

# SiteSafe — On-Device OSHA Construction Safety Violation Detector

> **One construction worker dies every eight hours in the United States.** SiteSafe puts an OSHA inspector in every site supervisor's pocket — fully offline, on-device, fine-tuned for the Fatal Four.

**Gemma 4 Good Hackathon Submission — Global Resilience Track**

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Powered by Gemma 4](https://img.shields.io/badge/Powered%20by-Gemma%204%20E4B-orange.svg)](https://ai.google.dev/gemma)
[![Fine-tuned with Unsloth](https://img.shields.io/badge/Fine--tuned%20with-Unsloth-green.svg)](https://github.com/unslothai/unsloth)
[![Deployed via Ollama](https://img.shields.io/badge/Deployed%20via-Ollama-black.svg)](https://ollama.com)

---

## The Problem

OSHA's "Fatal Four" hazards cause **58.6%** of all construction worker deaths annually — approximately **1,069 deaths in 2024** alone:

| Fatal Four Hazard       | % of Deaths | Annual Deaths | Primary OSHA Standard         |
|-------------------------|-------------|---------------|-------------------------------|
| Falls                   | 33.5%       | ~358          | 29 CFR 1926.501               |
| Struck-By               | 11.4%       | ~122          | 29 CFR 1926.602               |
| Electrocution           | 8.4%        | ~90           | 29 CFR 1926.405               |
| Caught-In/Between       | 5.4%        | ~58           | 29 CFR 1926.652               |

OSHA conducted only **34,625 inspections in FY2024** across all industries, against **~900,000 active U.S. construction sites**. Small contractors (<20 employees) account for the majority of fatalities but rarely employ a dedicated safety officer. Remote and rural sites often lack reliable cellular service.

**SiteSafe fills the gap** — photo-based OSHA compliance checking that runs entirely on-device, with calibrated confidence scores and actionable corrective guidance.

---

## How It Works

```
┌──────────────────────────────────────────────────────────────────────┐
│                                                                      │
│  📸 Photo               🧠 Gemma 4 E4B                  📄 Report   │
│  ────────  ─────────►   ───────────────────  ─────────►  ────────   │
│  Job site               Vision + reasoning              CFR-cited    │
│  image                  Fine-tuned via Unsloth          violations   │
│                              │                          + corrective │
│                              ▼                            actions    │
│                         📚 OSHA tools                                 │
│                         (function calling                             │
│                          → SQLite KB)                                 │
│                                                                      │
│              All inference local. No data leaves device.             │
└──────────────────────────────────────────────────────────────────────┘
```

1. **User snaps a photo** of a job site (web app, webcam, or upload).
2. **Gemma 4 E4B** — fine-tuned on OSHA violation reports — analyzes the image.
3. **Function calling** lets the model query a local SQLite knowledge base of 50+ OSHA standards for the precise CFR citation, penalty range, and corrective action language.
4. **Calibrated confidence scores** (Platt scaling, ECE < 0.10) accompany each detected violation.
5. **A structured report** is generated in the app, exportable to PDF.

Everything runs on a laptop, edge box, or workstation. No cloud calls. No data exfiltration.

---

## Quick Start

```bash
# 1. Clone and set up
git clone https://github.com/<you>/sitesafe
cd sitesafe
bash setup.sh                         # installs Python deps + builds the OSHA DB

# 2. Pull base model with Ollama (or use the fine-tuned GGUF — see training/)
ollama pull gemma3:4b                 # placeholder until your fine-tuned GGUF is built
ollama create sitesafe -f training/Modelfile

# 3. Run the app
python app/sitesafe_app.py            # opens Gradio UI on http://localhost:7860
```

Or with Docker:

```bash
docker-compose up
# UI on http://localhost:7860
```

---

## Repository Layout

```
sitesafe/
├── data/                    # OSHA knowledge base + training data construction
├── training/                # Unsloth fine-tuning + GGUF export + Ollama Modelfile
├── app/                     # Gradio UI, inference, function-calling tools, calibration, PDF reports
├── evaluation/              # Detection / calibration / latency metrics + reliability diagram
└── docs/                    # Technical write-up + submission checklist
```

See [`docs/technical_writeup.md`](docs/technical_writeup.md) for the full architecture, fine-tuning details, evaluation, and impact analysis.

---

## Why Gemma 4

SiteSafe is a portrait-fit for what makes Gemma 4 distinctive:

| Gemma 4 Capability                           | How SiteSafe Uses It                                                                |
|----------------------------------------------|-------------------------------------------------------------------------------------|
| **Native multimodality (image + text)**       | Job-site photos in, structured violation reports out                                 |
| **Function calling / tool use**               | Model autonomously queries the local OSHA SQLite DB for CFR citation + penalty data |
| **Edge-friendly E4B variant**                 | Runs on consumer hardware (8 GB RAM minimum) and Ollama-compatible mobile devices    |
| **Apache 2.0 license**                        | No restrictions for safety-of-life commercial deployment                            |
| **Unsloth fine-tuning on free Kaggle T4 GPU** | Reproducible by any judge or contractor with a Kaggle account                       |

---

## Evaluation Snapshot

See [`docs/technical_writeup.md`](docs/technical_writeup.md) §Results for the full table, but in summary:

| Metric                                | Target     | Achieved   |
|---------------------------------------|------------|------------|
| Violation Detection Rate (recall)     | > 0.70     | reported in writeup |
| False Positive Rate                   | < 0.20     | reported in writeup |
| Correct CFR Citation Rate             | > 0.85     | reported in writeup |
| Expected Calibration Error (ECE)      | < 0.10     | reported in writeup |
| Mean Inference Latency (E4B q4_k_m)   | < 8 sec    | reported in writeup |

Numbers are produced by `python evaluation/evaluate.py`, which writes JSON + a markdown table + a reliability diagram into `evaluation/results/`.

---

## Data Sources & Attribution

Training data is constructed from publicly licensed image datasets:

- **Roboflow Construction Site Safety Image Dataset** — Roboflow Universe (CC BY 4.0). 2,801 images, 10 PPE-related classes.
- **SHEL5K** — Otgonbold et al. — 5,000 images, helmet/vest annotations.
- **CHVG Dataset** — Wang et al., "PPE detector: a YOLO-based architecture", PeerJ Computer Science 2022.
- **SHWD (Safety Helmet Wearing Dataset)** — Public benchmark dataset.

OSHA regulation text is a public-domain U.S. federal record (29 CFR Part 1926). See `data/build_osha_db.py` for the full citation list.

---

## License

Apache License 2.0 — see [`LICENSE`](LICENSE).

This permits commercial and non-commercial use without restriction, including bundling SiteSafe into proprietary safety-management platforms.

---

## Author

**Sam Mathew** — Carnegie Mellon University — `semathew@andrew.cmu.edu`

Built for the [Gemma 4 Good Hackathon on Kaggle](https://www.kaggle.com) (Global Resilience track).

---

## Disclaimer

SiteSafe is an assistive tool. It does **not** replace certified OSHA inspections, qualified competent persons, or formal safety audits. Photo-based assessment cannot evaluate every hazard (e.g., electrical grounding, chemical exposure, structural shoring). Always pair SiteSafe with on-the-ground inspection and your jurisdiction's safety program.
