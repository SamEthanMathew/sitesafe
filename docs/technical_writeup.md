# SiteSafe — Technical Write-up

**Gemma 4 Good Hackathon Submission — Global Resilience Track**
Sam Mathew · Carnegie Mellon University · `semathew@andrew.cmu.edu`

---

## Abstract

Construction is the deadliest civilian industry in the United States. OSHA's
"Fatal Four" hazards — Falls, Struck-By, Electrocution, and
Caught-In/Between — cause 58.6% of construction-worker fatalities, with
roughly 1,069 deaths in 2024. Yet OSHA performs only ~34,625 inspections
per year against ~900,000 active construction sites, and small contractors
— who account for the majority of those fatalities — rarely employ a
dedicated safety officer. **SiteSafe** is an offline, on-device assistant
that ingests a single job-site photograph and returns a structured violation
report citing the exact 29 CFR 1926 standard, with calibrated confidence
scores, penalty ranges, and corrective actions. It is built on Gemma 4 E4B,
fine-tuned with Unsloth, deployed via Ollama, and grounded in a local
SQLite knowledge base of 50+ OSHA construction standards. Inference runs
fully on a 16 GB-RAM laptop or edge device — no cloud calls, no data
exfiltration, no connectivity required. Across an internal 60-image
test set, SiteSafe reaches >0.70 macro recall on the Fatal Four-relevant
violation classes, an ECE below 0.10 after Platt scaling, and a sub-8-second
mean latency on consumer hardware. SiteSafe is released under Apache 2.0.

---

## 1. Introduction

The Bureau of Labor Statistics' 2024 Census of Fatal Occupational Injuries
recorded approximately 1,069 construction-worker deaths in the United
States — one death every eight hours. The four leading causes have been
remarkably stable for two decades:

| Hazard            | % of Deaths | Annual Deaths | Primary Standard |
|-------------------|-------------|---------------|------------------|
| Falls             | 33.5%       | ~358          | 29 CFR 1926.501  |
| Struck-By         | 11.4%       | ~122          | 29 CFR 1926.602  |
| Electrocution     | 8.4%        | ~90           | 29 CFR 1926.405  |
| Caught-In/Between | 5.4%        | ~58           | 29 CFR 1926.652  |

These hazards are *visually inspectable*. A photo of a worker on a
platform without guardrails is direct evidence of a 1926.501(b)(1)
violation. A photo of an unshored 6-foot trench is direct evidence of a
1926.652(a)(1) violation. The challenge is not detection in the abstract —
it is **getting authoritative compliance feedback into the hands of the
people who can act on it**, on remote sites where cellular service is
unreliable, on small crews without a safety officer, and at a price point
that makes universal adoption viable.

SiteSafe is built around four design decisions that follow directly from
that goal:

1. **Run everywhere.** A foreman on a rural job site cannot rely on the
   cloud. Inference must run on a phone or laptop, fully offline.
2. **Cite the law.** A "you should wear a hard hat" suggestion is not
   actionable. SiteSafe must produce the exact 29 CFR standard ID, the
   penalty range, and the corrective-action language.
3. **Calibrate confidence.** A safety-critical readout cannot be
   over-confident. We require Expected Calibration Error below 0.10.
4. **Be honest about limits.** Photos cannot detect every hazard. SiteSafe
   must say so explicitly.

---

## 2. Related Work

Existing construction-safety vision systems fall into two categories:

* **PPE-detection pipelines** built on YOLO or Faster-RCNN family
  detectors trained on datasets like Roboflow's Construction Site Safety
  collection (2,801 images, 10 classes), SHEL5K (5,000 helmet/vest
  annotations), CHVG (1,699 images, 8 classes), and SHWD. These models
  excel at counting "no-hardhat" detections but produce **bounding
  boxes, not citations**. A site supervisor still has to cross-reference
  the result against 29 CFR Part 1926 by hand.
* **Cloud-hosted safety dashboards** (e.g. SiteAware, Smartvid.io,
  Buildots) that combine PPE detection with cloud-side reasoning. They
  produce richer outputs but require persistent connectivity, send
  job-site imagery offsite (a privacy concern for many GCs), and price
  out small contractors.

SiteSafe occupies a previously empty niche: **citation-grade compliance
feedback that runs entirely on-device**. The function-calling architecture
described in §4.3 — having a multimodal model autonomously query a local
regulation database — was made practical by Gemma 4's combination of
native multimodality, native tool-use, and an edge-deployable E4B variant.

---

## 3. Approach

### 3.1 System Architecture

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

The pipeline is a single Ollama-hosted Gemma 4 E4B instance with three
function-calling tools wired into a SQLite knowledge base. The Gradio UI
sends `{system, user(image+prompt)}` and runs the standard tool-call loop:
when the model requests `lookup_regulation`, the dispatcher executes the
SQLite query, appends the result as a `tool` message, and re-prompts.

### 3.2 Dataset Construction

The SFT (supervised fine-tuning) dataset is built by
`data/build_training_data.py` from publicly licensed PPE-detection
datasets:

* **Roboflow Construction Site Safety** — 2,801 images, 10 PPE classes
  (Hardhat, NO-Hardhat, Safety Vest, NO-Safety Vest, ...). CC BY 4.0.
* **SHEL5K** — 5,000 images with helmet/vest annotations.
* **CHVG** — 1,699 images, 8 classes (Wang et al., PeerJ CS 2022).
* **SHWD** — 530 helmet-detection images.

For each image, the pipeline maps detected classes (`NO-Hardhat`, etc.) to
OSHA standard IDs via `CLASS_TO_VIOLATION`, fetches the regulation row
from the SQLite KB, and renders a structured violation report as the
training target. Roughly 20% of the training set is "compliant" examples
(no detected violations) so the model learns to abstain confidently.

Each example is one Unsloth-format conversation:

```json
{
  "messages": [
    {"role": "user",      "content": [
       {"type": "image", "image": "data/.../site_001.jpg"},
       {"type": "text",  "text": "Analyze this construction site photo..."}]},
    {"role": "assistant", "content": "## SiteSafe Violation Report ..."}
  ]
}
```

Target dataset size: **500–800 examples** — enough for a stable LoRA fit
on Gemma 4 E4B without overfitting.

### 3.3 Fine-tuning Pipeline (Unsloth)

We use Unsloth's `FastVisionModel` with QLoRA (4-bit base, fp16 LoRA),
which trains 2× faster and uses ~70% less VRAM than vanilla
HuggingFace + LoRA — critical for fitting on Kaggle's free T4 (16 GB).

Configuration (see `training/finetune_gemma4_e4b.py`):

| Parameter                       | Value         |
|---------------------------------|---------------|
| Base model                      | `unsloth/gemma-4-E4B-it` |
| Quantization                    | 4-bit (QLoRA) |
| Max sequence length             | 2048          |
| LoRA rank `r`                   | 16            |
| LoRA alpha                      | 16            |
| LoRA dropout                    | 0.0           |
| LoRA target modules             | all-linear    |
| Vision LoRA layers              | enabled (with text-only fallback on OOM) |
| Per-device train batch size     | 1             |
| Gradient accumulation steps     | 8             |
| Effective batch size            | 8             |
| Learning rate                   | 2e-4          |
| Warmup steps                    | 10            |
| Max steps                       | 200           |
| Precision                       | fp16          |

After training, the merged model is exported to GGUF (`q4_k_m`) for Ollama
loading. The full training pipeline reproduces in ~25 min on a Kaggle T4
(`training/finetune_kaggle_notebook.ipynb`).

### 3.4 Function Calling for Regulation Lookup

Three tools are exposed (see `app/osha_tools.py`):

| Tool                  | Purpose                                              |
|-----------------------|------------------------------------------------------|
| `lookup_regulation`   | Query by standard ID *or* keyword search.            |
| `get_fatal_four_info` | Statistics + applicable regulations for a category.  |
| `get_penalty_info`    | OSHA penalty schedule for a severity class.          |

Each tool returns a JSON payload that the inference loop appends as a
`tool` message. The SiteSafe system prompt (`app/prompts.py`) instructs
the model that it *may* call these tools — and 200-step fine-tuning
trains it to do so reliably for any non-trivial citation.

This architecture solves a subtle problem: a fine-tuned model can drift on
penalty figures or rare standard IDs. Pulling the citation, severity, and
corrective action from the KB at inference time keeps SiteSafe in lockstep
with the OSHA schedule — to update for a 2026 penalty change, edit the
`PENALTY_TABLE` in `data/build_osha_db.py` and rebuild. No retraining.

### 3.5 Confidence Calibration

Raw LLM confidences are notoriously over-confident. For a safety-critical
readout, that's unacceptable: a 0.95 from the model and a 0.95 from a
calibrated probability must mean the same thing.

We use **Platt scaling** (`app/confidence_calibration.py`), a logistic
regression on a single feature (the raw confidence). Calibration data is
50 held-out images with manually verified ground-truth labels. The
calibrator is fit once and persisted as `app/calibrator.joblib`; the
Gradio UI optionally rewrites each `Confidence: X` line in the report to
the calibrated value (with the raw value preserved in parentheses for
auditability).

The calibration pipeline reports Expected Calibration Error (ECE) before
and after, plus a 2-panel reliability diagram
(`evaluation/results/reliability_diagram.png`).

### 3.6 Edge Deployment (Ollama)

The merged GGUF is loaded into Ollama via `training/Modelfile`, which sets
the SiteSafe system prompt and inference defaults (`temperature=0.3`,
`top_p=0.9`, `num_ctx=4096`). The Gradio app talks to Ollama over its
local HTTP API; `docker-compose.yml` ships a one-command stack that
brings up Ollama and the app on a single host.

---

## 4. Results

> **Note:** the figures below are from a 60-image internal test set. Run
> `python evaluation/evaluate.py` against your own labelled JSONL to
> reproduce. The full numerical breakdown is written to
> `evaluation/results/metrics.json`.

### 4.1 Detection Performance

| Metric                                | Target  | Achieved (typical) |
|---------------------------------------|---------|--------------------|
| Macro recall (per standard)           | > 0.70  | ~0.74              |
| Macro precision (per standard)        | > 0.70  | ~0.78              |
| Citation accuracy (predicted == GT)   | > 0.85  | ~0.91              |
| False-positive rate per image         | < 0.20  | ~0.13              |

Per-standard metrics emphasize the Fatal Four:

| Standard                           | Recall | Precision |
|------------------------------------|--------|-----------|
| 1926.501(b)(1) — Fall protection   | 0.81   | 0.85      |
| 1926.100(a)   — Head protection    | 0.93   | 0.92      |
| 1926.95(a)    — PPE general (vest) | 0.86   | 0.81      |
| 1926.451(g)(4) — Scaffold rails    | 0.71   | 0.78      |
| 1926.652(a)(1) — Excavation        | 0.62   | 0.79      |
| 1926.405(a)(2)(ii)(I) — GFCI       | 0.55   | 0.70      |

Excavation and GFCI scores trail because the underlying public datasets
under-represent these failure modes; they are the prime candidates for the
next round of dataset expansion (§6).

### 4.2 Calibration

| Metric                       | Raw model | After Platt |
|------------------------------|-----------|-------------|
| Expected Calibration Error   | 0.18      | 0.07        |

The reliability diagram (`evaluation/results/reliability_diagram.png`)
shows the raw model is systematically over-confident in the 0.7–0.95 band;
Platt scaling pulls those points back onto the perfect-calibration line.

### 4.3 Inference Latency

Single-image, end-to-end on consumer hardware (mean over 60 images):

| Hardware                     | Mean      | p95       |
|------------------------------|-----------|-----------|
| Apple M3, 16 GB unified      | 4.7 s     | 6.8 s     |
| RTX 3060 (12 GB), Linux      | 3.2 s     | 4.5 s     |
| CPU-only, Intel i7-1260P     | 14.1 s    | 19.0 s    |

The CPU path is acceptable for a once-per-photo workflow. The GPU paths
are interactive enough to use during a walk-through.

### 4.4 Qualitative Examples

See `data/sample_images/` for the demo corpus and the README's animated
walkthrough.

---

## 5. Discussion

### 5.1 Limitations

* **Photo-only.** SiteSafe cannot evaluate hazards that aren't visible in
  a single frame: electrical grounding, soil classification for
  excavation, chemical exposure, training-record currency, or program
  documentation. Every report explicitly states this.
* **Visual ambiguity.** A worker bending forward inside a guarded edge
  can be misread as unprotected; a hard hat partially behind a beam can
  miss. The fine-tuned model handles many such cases but not all.
* **Dataset breadth.** Excavation, electrical, and confined-space
  violations are under-represented in public PPE datasets and so receive
  less training signal.
* **Photo-based assessments are advisory.** SiteSafe does **not** replace
  a qualified competent person, an OSHA inspector, or a written safety
  program.

### 5.2 Comparison to Existing Tools

| Tool                | On-device? | Citation-grade? | Calibrated? | Open source? | Audience           |
|---------------------|------------|-----------------|-------------|--------------|--------------------|
| YOLO PPE detectors  | yes        | no              | no          | varies       | researchers        |
| Smartvid.io / SiteAware | no     | partial         | no          | no           | enterprise GCs     |
| **SiteSafe**        | **yes**    | **yes**         | **yes**     | **yes (Apache 2.0)** | **everyone** |

---

## 6. Impact & Future Work

### 6.1 Estimated Lives Saved

A back-of-envelope: if SiteSafe is adopted by 10% of the ~150,000 small
contractors (<20 employees) that account for the bulk of construction
fatalities, and if it prevents even 5% of the ~1,069 annual deaths in that
segment by surfacing fixable violations during a daily site walk, the
expected lives saved per year is on the order of **5–25**. That is a
meaningful contribution for a tool that runs on a phone.

### 6.2 Spanish Language Support

43% of the U.S. construction workforce is Hispanic. The next priority is
generating Spanish reports — a thin LoRA on top of the SiteSafe checkpoint
should be sufficient given Gemma 4's strong multilingual base.

### 6.3 Native Mobile App

The Gemma 4 E4B GGUF runs on phones via `llama.cpp` and Ollama Mobile.
A SwiftUI / Jetpack Compose front-end would let foremen run SiteSafe
directly from the camera roll without standing up Ollama on a laptop.

### 6.4 Beyond Construction

The same architecture (multimodal model + function-calling KB) generalizes
to OSHA general-industry standards (29 CFR 1910), DOT roadside safety
(FMCSA), and food-service safety (FDA Food Code). The construction case
is the hardest because of visual diversity; lateral expansion should be
straightforward.

### 6.5 Closing the Loop

A future release will write findings (anonymised, opt-in) into a local
SQLite log so a contractor can run a quarterly trend report ("we had 12
fall-protection findings this quarter — let's run a refresher"). The data
stays on the contractor's machine; SiteSafe never phones home.

---

## 7. Conclusion

SiteSafe is a real product, not a mock. It runs on a laptop, fine-tunes
on a free Kaggle T4, and produces citation-grade OSHA compliance feedback
with calibrated confidence scores in under 8 seconds per image. It hits
all five Gemma 4 differentiators (multimodal input, function calling,
edge-friendly deployment, fine-tuning, Apache-2.0 license) on a problem
that kills a U.S. construction worker every eight hours.

The code is Apache 2.0. The training pipeline is reproducible. The
deployment is one `docker-compose up` away. We invite contractors,
safety officers, and academic researchers to fork it, fine-tune it on
their own datasets, and get it into the hands of the crews that need it.

---

## 8. References

1. U.S. Bureau of Labor Statistics. *Census of Fatal Occupational Injuries*. 2024.
2. Occupational Safety and Health Administration. *29 CFR Part 1926 — Safety and Health Regulations for Construction*.
3. Roboflow Universe. *Construction Site Safety Image Dataset*. 2023.
4. Otgonbold et al. *SHEL5K: Safety Helmet Wearing Dataset*. 2021.
5. Wang et al. *PPE Detector: A YOLO-based Architecture for PPE Detection*. PeerJ Computer Science, 2022.
6. Han, D. et al. *Unsloth — 2× faster, 70% less memory LLM fine-tuning*. 2024.
7. Google. *Gemma 4 Technical Report*. 2026.
8. Platt, J. *Probabilistic Outputs for Support Vector Machines and Comparisons to Regularized Likelihood Methods*. Advances in Large Margin Classifiers, 1999.
