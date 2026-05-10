#!/usr/bin/env bash
# SiteSafe — Dataset Acquisition Script
#
# Downloads the publicly available image datasets used for training and
# evaluation, and a handful of royalty-free demo images from Wikimedia
# Commons. Some datasets require manual download (academic licenses) — for
# those we just print instructions.
#
# Usage:
#     bash data/download_datasets.sh
#
# The script is idempotent: rerunning skips files that already exist.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${SCRIPT_DIR}"
SAMPLE_DIR="${DATA_DIR}/sample_images"
DATASETS_DIR="${DATA_DIR}/datasets"

mkdir -p "${SAMPLE_DIR}" "${DATASETS_DIR}"

green()  { printf '\033[0;32m%s\033[0m\n' "$1"; }
yellow() { printf '\033[0;33m%s\033[0m\n' "$1"; }

# ---------------------------------------------------------------------------
# 1. Demo images — Wikimedia Commons (CC-licensed)
# ---------------------------------------------------------------------------
# We resolve files through `Special:FilePath/<name>?width=1024`, which is the
# only Commons URL guaranteed not to bit-rot — the upload.wikimedia.org thumb
# paths shift whenever a file is renamed or rehashed.
#
# Usage of these images is strictly demonstrative. SiteSafe does NOT assert
# that any worker visible in the source photos committed the cited
# violation.

UA="Mozilla/5.0 (compatible; SiteSafe/1.0; +https://github.com/sitesafe)"
COMMONS_FP_BASE="https://commons.wikimedia.org/wiki/Special:FilePath"

green "==> Fetching demo images from Wikimedia Commons"

# local => Commons filename (do NOT URL-encode here; we encode below)
declare -A DEMO_IMAGES=(
  ["construction_workers.jpg"]="A construction worker plastering a house 01.jpg"
  ["scaffolding_workers.jpg"]="A worker wears a helmet and visor at a Hong Kong construction site during a heatwave.jpg"
  ["construction_excavation.jpg"]="Excavation.jpg"
  ["construction_site_with_ppe.jpg"]="Construction Worker On Footpath.jpg"
  ["construction_concrete_pour.jpg"]="Rebar worker.jpg"
)

for name in "${!DEMO_IMAGES[@]}"; do
  out="${SAMPLE_DIR}/${name}"
  if [[ -s "${out}" ]]; then
    yellow "    [skip] ${name} already present"
    continue
  fi
  src="${DEMO_IMAGES[${name}]}"
  # Use Python for reliable URL-encoding (handles spaces, parens, accents).
  encoded=$(python -c "import sys, urllib.parse; print(urllib.parse.quote(sys.argv[1]))" "${src}")
  url="${COMMONS_FP_BASE}/${encoded}?width=1024"
  if curl --fail --silent --show-error --location \
        -A "${UA}" --max-time 30 \
        -o "${out}" "${url}"; then
    green "    [ok]   ${name}"
  else
    yellow "    [warn] failed to fetch ${name} (offline or rate-limited?). Continuing."
    rm -f "${out}"
  fi
  # Be polite to upstream — Commons doesn't love bursty downloads.
  sleep 1
done

# ---------------------------------------------------------------------------
# 2. Roboflow Construction Site Safety Dataset (Kaggle / Roboflow Universe)
# ---------------------------------------------------------------------------
green "==> Roboflow Construction Site Safety dataset"

ROBOFLOW_DIR="${DATASETS_DIR}/roboflow"
if [[ -d "${ROBOFLOW_DIR}/train/images" ]]; then
  yellow "    [skip] Roboflow dataset already present at ${ROBOFLOW_DIR}"
else
  cat <<EOF
    To download via Kaggle CLI (recommended):
        pip install kaggle
        kaggle datasets download -d snehilsanyal/construction-site-safety-image-dataset-roboflow
        unzip construction-site-safety-image-dataset-roboflow.zip -d "${ROBOFLOW_DIR}"

    Or via Roboflow Universe (browser):
        https://universe.roboflow.com/roboflow-universe-projects/construction-site-safety
        Export as YOLOv8 → ZIP → unzip into ${ROBOFLOW_DIR}

    Expected layout after extraction:
        ${ROBOFLOW_DIR}/train/images/*.jpg
        ${ROBOFLOW_DIR}/train/labels/*.txt
        ${ROBOFLOW_DIR}/data.yaml
EOF
fi

# ---------------------------------------------------------------------------
# 3. Other useful datasets (manual download)
# ---------------------------------------------------------------------------
green "==> Other recommended datasets (manual download)"
cat <<'EOF'
    - SHEL5K (5,000 images, helmet/vest annotations)
        https://github.com/shubham991/SHEL5K_Dataset

    - CHVG Dataset (1,699 images, 8 classes)
        Wang et al., PeerJ Computer Science 2022
        https://github.com/ZijianWang1995/PPE_detection

    - Safety Helmet Wearing Dataset (SHWD)
        https://github.com/njvisionpower/Safety-Helmet-Wearing-Dataset

    Place each under data/datasets/ in its own subdirectory.
EOF

green "==> Dataset acquisition step complete"
