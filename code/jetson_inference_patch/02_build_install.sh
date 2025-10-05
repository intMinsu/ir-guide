#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 [/abs/path/to/jetson-inference]"
  echo "If omitted, you'll be prompted interactively."
}

# ----- 0) Optional arg or interactive prompt for repo path -----
resolve_path() {
  if command -v realpath >/dev/null 2>&1; then
    realpath "$1"
  else
    readlink -f "$1"
  fi
}

REPO="${1:-}"

if [[ -z "${REPO}" ]]; then
  GUESS1="$(pwd)/jetson-inference"
  GUESS2="$HOME/jetson-inference"

  echo "Enter ABSOLUTE path to your jetson-inference repo."
  if [[ -d "${GUESS1}" ]]; then
    read -r -p "Path [${GUESS1}]: " INP
    REPO="${INP:-$GUESS1}"
  elif [[ -d "${GUESS2}" ]]; then
    read -r -p "Path [${GUESS2}]: " INP
    REPO="${INP:-$GUESS2}"
  else
    read -rp "Path: " REPO
  fi
fi

REPO="$(resolve_path "${REPO}")" || { echo "[err] cannot resolve path"; exit 1; }
[[ -d "${REPO}" ]] || { echo "[err] repo path not found: ${REPO}"; exit 1; }
[[ -f "${REPO}/CMakeLists.txt" ]] || { echo "[err] not a jetson-inference root (missing CMakeLists.txt): ${REPO}"; exit 1; }
[[ -d "${REPO}/utils/python/bindings" ]] || { echo "[err] missing utils/python/bindings in repo: ${REPO}"; exit 1; }
[[ -d "${REPO}/python/bindings" ]] || { echo "[err] missing python/bindings in repo: ${REPO}"; exit 1; }

echo "[cfg] repo = ${REPO}"

# ----- 1) Must be in an activated conda env -----
: "${CONDA_PREFIX:?Activate your conda env first (conda activate <env>)}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATCH_DIR="${SCRIPT_DIR}/patches"

need_files=(
  "${PATCH_DIR}/utils/python/bindings/CMakeLists.txt"
  "${PATCH_DIR}/python/bindings/CMakeLists.txt"
  "${PATCH_DIR}/utils/python/bindings/PyNumpy.cpp"
)
for f in "${need_files[@]}"; do
  [[ -f "$f" ]] || { echo "[err] missing patch file: $f"; exit 1; }
done

# ----- 2) Python/NumPy paths from THIS env -----
PY="${CONDA_PREFIX}/bin/python3"
PY_ABI="$("${PY}" -c 'import sys;print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
SITE="$("${PY}" -c 'import sysconfig;print(sysconfig.get_paths()["platlib"])')"
NP_INC="$("${PY}" -c 'import numpy;print(numpy.get_include())')"
PY_INC="${CONDA_PREFIX}/include/python${PY_ABI}"
PY_LIB="${CONDA_PREFIX}/lib/libpython${PY_ABI}.so"

echo "[cfg] PY     = ${PY}"
echo "[cfg] PY_ABI = ${PY_ABI}"
echo "[cfg] SITE   = ${SITE}"
echo "[cfg] NP_INC = ${NP_INC}"
echo "[cfg] PY_INC = ${PY_INC}"
echo "[cfg] PY_LIB = ${PY_LIB}"

# ----- 3) Backup + copy helper -----
backup_and_copy() {
  local src="$1" dst="$2"
  if [[ ! -f "${dst}" ]]; then
    echo "[warn] destination missing; creating: ${dst}"
    install -d "$(dirname "${dst}")"
    install -m 0644 "${src}" "${dst}"
    return 0
  fi
  local bak="${dst}~.bak"
  if [[ ! -f "${bak}" ]]; then
    echo "[bak] ${dst} -> ${bak}"
    cp -a "${dst}" "${bak}"
  else
    echo "[bak] exists (leaving): ${bak}"
  fi
  echo "[cp ] ${src} -> ${dst}"
  install -m 0644 "${src}" "${dst}"
}

echo
echo "==> Patching CMake & PyNumpy…"
backup_and_copy "${PATCH_DIR}/utils/python/bindings/CMakeLists.txt" \
                "${REPO}/utils/python/bindings/CMakeLists.txt"
backup_and_copy "${PATCH_DIR}/python/bindings/CMakeLists.txt" \
                "${REPO}/python/bindings/CMakeLists.txt"
backup_and_copy "${PATCH_DIR}/utils/python/bindings/PyNumpy.cpp" \
                "${REPO}/utils/python/bindings/PyNumpy.cpp"

# ----- 4) Clean build dir, then configure, build, install -----
BUILD_DIR="${REPO}/build"
echo
echo "==> Resetting build dir…"
if [[ -d "${BUILD_DIR}" ]]; then
  echo "[clean] removing existing ${BUILD_DIR}"
  rm -rf "${BUILD_DIR}"
fi
mkdir -p "${BUILD_DIR}"

# rely on RPATH; set LD_LIBRARY_PATH here only for this shell's checks
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH-}"

echo
echo "==> Configuring CMake…"
cmake -S "${REPO}" -B "${BUILD_DIR}" \
  -G "Unix Makefiles" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="${CONDA_PREFIX}" \
  -DCMAKE_PREFIX_PATH="${CONDA_PREFIX}" \
  -DPython3_EXECUTABLE="${PY}" \
  -DPYTHON_EXECUTABLE="${PY}" \
  -DPYTHON_INSTALL_DIR="${SITE}" \
  -DPYTHON_INCLUDE_DIR="${PY_INC}" \
  -DPYTHON_LIBRARY="${PY_LIB}" \
  -DNUMPY_INCLUDE_DIR="${NP_INC}" \
  -DCMAKE_INSTALL_RPATH="${CONDA_PREFIX}/lib" \
  -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON \
  -DCMAKE_CXX_FLAGS="-DHAS_NUMPY -I${NP_INC}" \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5

echo
echo "==> Building…"
cmake --build "${BUILD_DIR}" -j"$(nproc)"

echo
echo "==> Installing into conda env…"
cmake --install "${BUILD_DIR}"

# ----- 5) Quick checks -----
echo
echo "[test] ldd checks"
ldd "${SITE}/jetson_utils_python.so" | egrep 'libpython|libjetson-(utils|inference)\.so' || true
ldd "${CONDA_PREFIX}/lib/libjetson-inference.so" | grep libjetson-utils || true

echo
echo "[test] python imports"
"${PY}" - <<'PY'
import jetson_utils
import jetson_inference
print("OK: imports succeeded")
PY
