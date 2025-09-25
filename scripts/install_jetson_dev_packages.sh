#!/usr/bin/env bash
# install_jetson_dev_packages.sh (interactive)
# JP 5.x (Xavier/Orin). Accepts selections at runtime or as 1,2,3,4,5 list arg.
# Env: OPENCV_VER=4.10.0 WORKDIR=$HOME/opencv-workspace JOBS=4 DS_VER=6.3

set -euo pipefail

SELECTIONS="${1:-}"
OPENCV_VER="${OPENCV_VER:-4.10.0}"
WORKDIR="${WORKDIR:-$HOME/opencv-workspace}"
JOBS_OVERRIDE="${JOBS:-}"
DS_VER="${DS_VER:-6.3}"

log()  { printf "\n\033[1;36m[*]\033[0m %s\n" "$*"; }
warn() { printf "\n\033[1;33m[!]\033[0m %s\n" "$*"; }
ok()   { printf "\033[1;32m[✓]\033[0m %s\n" "$*\n"; }
require_cmd(){ command -v "$1" >/dev/null; }
detect_model(){ tr -d '\0' < /proc/device-tree/model 2>/dev/null || true; }

######## [1] Max performance mode ########
do_max_perf() {
  set -euo pipefail
  echo "[*] Setting max performance (nvpmodel + jetson_clocks)…"

  MODE_ID=""

  # --- 1) Prefer parsing /etc/nvpmodel.conf (most reliable) ---
  if [ -r /etc/nvpmodel.conf ]; then
    MODE_ID="$(awk '
      BEGIN{bestW=-1; bestC=-1; bestID=""}
      $1=="<" && $2=="POWER_MODEL" {
        id=""; name=""
        for (i=3;i<=NF;i++){
          if ($i ~ /^ID=/)   { split($i,a,"="); id=a[2] }
          if ($i ~ /^NAME=/) { split($i,a,"="); name=a[2] }
        }
        low=tolower(name)
        # Score: prefer MAXN; else highest W, then highest CORE count.
        w=0; c=0
        if (index(low,"maxn")) { w=9999; c=999 }
        if (match(low, /[0-9]+[[:space:]]*w/)) { s=substr(low,RSTART,RLENGTH); gsub(/[^0-9]/,"",s); w=s+0 }
        if (match(low, /[0-9]+[[:space:]]*core/)) { s=substr(low,RSTART,RLENGTH); gsub(/[^0-9]/,"",s); c=s+0 }
        if (w>bestW || (w==bestW && c>bestC)) { bestW=w; bestC=c; bestID=id }
      }
      END{ if (bestID!="") print bestID }
    ' /etc/nvpmodel.conf)"
  fi

  # --- 2) Fallback: parse a mode list if the BSP provides it ---
  if [ -z "$MODE_ID" ]; then
    LIST_OUT="$(LC_ALL=C sudo nvpmodel -q --list 2>/dev/null || true)"
    if [ -n "$LIST_OUT" ]; then
      MODE_ID="$(printf '%s\n' "$LIST_OUT" | awk '
        BEGIN{best=-1; bestid=""; bestC=-1}
        /^[[:space:]]*[0-9]+[[:space:]]/{
          id=$1; line=$0; low=line
          # normalize for case-insensitive checks
          gsub(/W/,"w",low); gsub(/CORE/,"core",low)
          if (index(tolower(low),"maxn")) { print id; exit }   # prefer MAXN if present
          w=0; c=0
          if (match(low, /[0-9]+[[:space:]]*w/)) { s=substr(low,RSTART,RLENGTH); gsub(/[^0-9]/,"",s); w=s+0 }
          if (match(low, /[0-9]+[[:space:]]*core/)) { s=substr(low,RSTART,RLENGTH); gsub(/[^0-9]/,"",s); c=s+0 }
          score=w*1000 + c
          if (score>best) {best=score; bestid=id}
        }
        END{ if (bestid!="") print bestid }
      ')"
    fi
  fi

  # --- 3) Last resort: probe a small ID range and score the names ---
  if [ -z "$MODE_ID" ]; then
    echo "[i] No list available; probing IDs 0–12 to find the strongest profile…"
    bestScore=-1; bestID=""
    for id in 0 1 2 3 4 5 6 7 8 9 10 11 12; do
      if sudo nvpmodel -m "$id" >/dev/null 2>&1; then
        name="$(LC_ALL=C sudo nvpmodel -q 2>/dev/null | awk -F': ' '/NV Power Mode/{print $2; exit}')"
        low="$(printf '%s' "$name" | tr 'A-Z' 'a-z')"
        if echo "$low" | grep -q "maxn"; then bestID="$id"; bestScore=999999; break; fi
        w=$(printf '%s' "$low" | sed -n 's/.*\b\([0-9][0-9]*\)[[:space:]]*w\b.*/\1/p'); w=${w:-0}
        c=$(printf '%s' "$low" | sed -n 's/.*\b\([0-9][0-9]*\)[[:space:]]*core\b.*/\1/p'); c=${c:-0}
        score=$(( w*1000 + c ))
        if [ "$score" -gt "$bestScore" ]; then bestScore="$score"; bestID="$id"; fi
      fi
    done
    MODE_ID="${bestID:-0}"
  fi

  echo "[*] Using nvpmodel mode id: ${MODE_ID}"
  sudo nvpmodel -m "${MODE_ID}" || sudo nvpmodel -m 0

  if command -v jetson_clocks >/dev/null 2>&1; then
    sudo jetson_clocks
  else
    echo "[!] jetson_clocks not found; skipping."
  fi

  echo "[✓] Now:"
  LC_ALL=C sudo nvpmodel -q | sed -n '1,2p'
}

######## [2] gh CLI ########
do_gh_cli() {
  log "Installing GitHub CLI (gh)…"
  (type -p wget >/dev/null || (sudo apt update && sudo apt install -y wget)) \
  && sudo mkdir -p -m 755 /etc/apt/keyrings \
  && out=$(mktemp) && wget -nv -O "$out" https://cli.github.com/packages/githubcli-archive-keyring.gpg \
  && sudo tee /etc/apt/keyrings/githubcli-archive-keyring.gpg < "$out" >/dev/null \
  && sudo chmod go+r /etc/apt/keyrings/githubcli-archive-keyring.gpg \
  && sudo mkdir -p -m 755 /etc/apt/sources.list.d \
  && echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" \
     | sudo tee /etc/apt/sources.list.d/github-cli.list >/dev/null \
  && sudo apt update && sudo apt install -y gh
  ok "gh installed."
}

######## [3] jtop ########
do_jtop() {
  log "Installing jtop (jetson-stats)…"
  sudo apt-get update
  sudo apt-get install -y python3-pip
  sudo -H pip3 install -U jetson-stats
  sudo systemctl restart jetson_stats.service || true
  ok "jtop installed. Run: jtop"
}

######## [4] Jetson runtime + container runtime ########
do_runtime() {
  log "Installing Jetson stack via meta-packages…"
  sudo apt-get update

  # Choose flavor with env var: runtime|dev (default: dev)
  JETPACK_FLAVOR="${JETPACK_FLAVOR:-dev}"

  if [[ "$JETPACK_FLAVOR" == "runtime" ]]; then
    sudo apt-get install -y nvidia-jetpack-runtime
    # Optional: add nvcc for builds without pulling all Dev extras
    sudo apt-get install -y cuda-nvcc-11-4 || sudo apt-get install -y cuda-toolkit-11-4 || sudo apt-get install -y cuda || true
  else
    # 'dev' gives you toolkit + samples/docs
    sudo apt-get install -y nvidia-jetpack
  fi

  # Container runtime (handy even if you don’t use it right away)
  sudo apt-get install -y nvidia-container-toolkit
  sudo nvidia-ctk runtime configure --runtime=docker || true
  sudo systemctl restart docker || true

  ok "JetPack (${JETPACK_FLAVOR}) installed."
}
######## [5] Build OpenCV (CUDA/FFmpeg/GStreamer) ########
do_opencv() {
  local MODEL ARCH_BIN ARCH_PTX JOBS PY3_EXEC PY3_INC PY3_SITE
  MODEL="$(detect_model)"
  log "Building OpenCV ${OPENCV_VER} (CUDA/FFmpeg/GStreamer)…  Model: ${MODEL}"

  if [[ "$MODEL" == *"Xavier"* ]]; then
    ARCH_BIN="7.2"; ARCH_PTX="sm_72"; JOBS="${JOBS_OVERRIDE:-4}"
  elif [[ "$MODEL" == *"Orin"* ]]; then
    ARCH_BIN="8.7"; ARCH_PTX="sm_87"; JOBS="${JOBS_OVERRIDE:-$(nproc)}"
  else
    warn "Unknown Jetson model. Expected Xavier/Orin. Skipping OpenCV build."; return 0
  fi

  sudo apt-get update
  sudo apt-get install -y \
    build-essential cmake git unzip curl pkg-config \
    libgtk-3-dev libavcodec-dev libavformat-dev libswscale-dev \
    libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
    libtbb-dev libjpeg-dev libpng-dev libtiff-dev libdc1394-22-dev \
    libv4l-dev v4l-utils ffmpeg \
    python3-dev python3-numpy

  mkdir -p "${WORKDIR}" && cd "${WORKDIR}"
  [[ -d "opencv-${OPENCV_VER}" ]] || curl -L -o "opencv-${OPENCV_VER}.zip" "https://github.com/opencv/opencv/archive/${OPENCV_VER}.zip"
  [[ -d "opencv_contrib-${OPENCV_VER}" ]] || curl -L -o "opencv_contrib-${OPENCV_VER}.zip" "https://github.com/opencv/opencv_contrib/archive/${OPENCV_VER}.zip"
  [[ -d "opencv-${OPENCV_VER}" ]] || unzip -q "opencv-${OPENCV_VER}.zip"
  [[ -d "opencv_contrib-${OPENCV_VER}" ]] || unzip -q "opencv_contrib-${OPENCV_VER}.zip"

  cd "opencv-${OPENCV_VER}"
  mkdir -p build && cd build

  PY3_EXEC=$(command -v python3)
  PY3_INC=$($PY3_EXEC - <<'PY'
import sysconfig; print(sysconfig.get_paths()["include"])
PY
)
  PY3_SITE=$($PY3_EXEC - <<'PY'
import site; print(site.getsitepackages()[0])
PY
)

  log "Configuring CMake (ARCH_BIN=${ARCH_BIN}, jobs=${JOBS})…"
  cmake -D CMAKE_BUILD_TYPE=Release \
        -D CMAKE_INSTALL_PREFIX=/usr/local \
        -D OPENCV_GENERATE_PKGCONFIG=ON \
        -D OPENCV_EXTRA_MODULES_PATH="${WORKDIR}/opencv_contrib-${OPENCV_VER}/modules" \
        -D BUILD_opencv_python3=ON \
        -D BUILD_TESTS=OFF -D BUILD_PERF_TESTS=OFF -D BUILD_EXAMPLES=OFF \
        -D WITH_CUDA=ON -D CUDA_FAST_MATH=ON -D ENABLE_FAST_MATH=ON \
        -D WITH_CUDNN=ON -D WITH_CUFFT=ON -D OPENCV_DNN_CUDA=ON \
        -D CUDA_ARCH_BIN="${ARCH_BIN}" -D CUDA_ARCH_PTX="${ARCH_PTX}" \
        -D WITH_GSTREAMER=ON -D WITH_FFMPEG=ON -D WITH_V4L=ON \
        -D WITH_TBB=ON -D WITH_OPENMP=ON -D WITH_OPENGL=ON \
        -D PYTHON3_EXECUTABLE="${PY3_EXEC}" \
        -D PYTHON3_INCLUDE_DIR="${PY3_INC}" \
        -D PYTHON3_PACKAGES_PATH="${PY3_SITE}" \
        ..
  make -j"${JOBS}"
  sudo make install
  sudo ldconfig
  ok "OpenCV ${OPENCV_VER} installed to /usr/local"
}

######## [6] DeepStream ########
do_deepstream() {
  log "Installing DeepStream (trying deepstream-${DS_VER}, then 'deepstream')…"
  sudo apt-get update
  if apt-cache policy "deepstream-${DS_VER}" | grep -q Candidate; then
    sudo apt-get install -y "deepstream-${DS_VER}"
  elif apt-cache policy deepstream | grep -q Candidate; then
    sudo apt-get install -y deepstream
  else
    warn "DeepStream package not found. Enable the NVIDIA DeepStream repo for your JP version if needed."
    return 0
  fi
  ok "DeepStream installed."
}

######## Interactive menu (if no arg) ########
if [[ -z "$SELECTIONS" ]]; then
  cat <<'MENU'

Select items to install (comma-separated). Press ENTER for default [1,2,3,4,5].

  1) Max performance mode (nvpmodel + jetson_clocks)   [recommended]
  2) GitHub CLI (gh)
  3) jtop (jetson-stats)
  4) Jetson runtime (CUDA/cuDNN/TensorRT/Multimedia + container runtime)
  5) Build OpenCV (CUDA + FFmpeg + GStreamer)
  6) DeepStream

MENU
  read -rp "Your selection [default 1,2,3,4,5]: " SELECTIONS
  SELECTIONS="${SELECTIONS//[[:space:]]/}"
  [[ -z "$SELECTIONS" ]] && SELECTIONS="1,2,3,4,5"
fi

# Validate input
if ! [[ "$SELECTIONS" =~ ^[1-6](,[1-6])*$ ]]; then
  warn "Invalid selection format. Use numbers 1-6 separated by commas (e.g., 1,3,5)."
  exit 1
fi

# De-duplicate while preserving order
IFS=',' read -ra _CHOICES <<< "$SELECTIONS"
declare -A seen=()
TO_RUN=()
for x in "${_CHOICES[@]}"; do
  if [[ -z "${seen[$x]+x}" ]]; then TO_RUN+=("$x"); seen[$x]=1; fi
done

echo -e "\nYou selected: ${TO_RUN[*]}"
read -rp "Proceed? [Y/n]: " yn
yn=${yn:-Y}
if [[ ! "$yn" =~ ^[Yy]$ ]]; then echo "Aborted."; exit 1; fi

######## Execute ########
for item in "${TO_RUN[@]}"; do
  case "$item" in
    1) do_max_perf ;;
    2) do_gh_cli ;;
    3) do_jtop ;;
    4) do_runtime ;;
    5) do_opencv ;;
    6) do_deepstream ;;
    *) warn "Unknown selection: $item (skipping)";;
  esac
done

######## Post checks ########
log "Quick post-install checks:"
if command -v nvpmodel >/dev/null; then sudo nvpmodel -q || true; fi
if command -v gh       >/dev/null; then gh --version | head -n1 || true; fi
if command -v jtop     >/dev/null; then jtop -v || true; fi
if command -v nvcc     >/dev/null; then nvcc --version | tail -n1 || true; fi
ldconfig -p | grep -E 'nvinfer|cudnn' || true
python3 - <<'PY' || true
import cv2
try:
    print("OpenCV:", cv2.__version__, "CUDA devices:", cv2.cuda.getCudaEnabledDeviceCount())
except Exception as e:
    print("cv2 import failed:", e)
PY
ok "All requested steps completed."
