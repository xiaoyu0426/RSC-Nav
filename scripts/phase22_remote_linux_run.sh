#!/usr/bin/env bash
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAMP="$(date +%Y%m%d-%H%M%S)"
HOST="$(hostname | tr -c 'A-Za-z0-9_.-' '_')"
OUT_DIR="${ROOT}/outputs/phase22_remote/${HOST}_${STAMP}"
LOG_DIR="${OUT_DIR}/logs"
ENV_NAME="${RSCNAV_ENV_NAME:-rscnav-habitat22}"
CONDA_PREFIX_DIR="${RSCNAV_CONDA_PREFIX:-${HOME}/.rscnav/miniforge3}"
HABITAT_LAB_DIR="${RSCNAV_HABITAT_LAB_DIR:-${HOME}/.rscnav/habitat-lab}"
SCENE_PATH="${RSCNAV_HABITAT_SCENE:-}"
DATA_PATH="${RSCNAV_HABITAT_DATA:-${HOME}/.rscnav/habitat_data}"
DOWNLOAD_TEST_SCENES="${RSCNAV_DOWNLOAD_TEST_SCENES:-1}"

mkdir -p "${LOG_DIR}"

log() {
  printf '[%s] %s\n' "$(date -Is)" "$*" | tee -a "${LOG_DIR}/runner.log"
}

run_capture() {
  local name="$1"
  shift
  log "RUN ${name}: $*"
  set +e
  "$@" >"${LOG_DIR}/${name}.stdout.log" 2>"${LOG_DIR}/${name}.stderr.log"
  local code=$?
  set -e
  printf '%s\n' "${code}" >"${LOG_DIR}/${name}.exit"
  if [ "${code}" -eq 0 ]; then
    log "OK ${name}"
  else
    log "FAIL ${name} exit=${code}"
  fi
  return 0
}

write_env_report() {
  {
    echo "timestamp=${STAMP}"
    echo "host=${HOST}"
    echo "root=${ROOT}"
    echo "user=$(id)"
    echo
    echo "## uname"
    uname -a || true
    echo
    echo "## os-release"
    cat /etc/os-release || true
    echo
    echo "## git"
    git --version || true
    git -C "${ROOT}" rev-parse HEAD || true
    git -C "${ROOT}" status --short || true
    echo
    echo "## nvidia-smi"
    nvidia-smi || true
    echo
    echo "## glvnd vendors"
    find /usr/share/glvnd /etc/glvnd -maxdepth 3 -type f 2>/dev/null -print -exec cat {} \; || true
    echo
    echo "## ldconfig gpu libs"
    ldconfig -p 2>/dev/null | grep -Ei 'libEGL|libGLX|libcuda|libnvidia' | sort || true
    echo
    echo "## env knobs"
    env | grep -E '^RSCNAV_|^CUDA_|^NVIDIA_|^DISPLAY=|^WAYLAND_DISPLAY=|^LD_LIBRARY_PATH=' | sort || true
  } >"${OUT_DIR}/environment_report.txt"
}

find_or_install_conda() {
  if [ -n "${RSCNAV_CONDA_EXE:-}" ] && [ -x "${RSCNAV_CONDA_EXE}" ]; then
    CONDA_EXE="${RSCNAV_CONDA_EXE}"
    return 0
  fi
  if command -v mamba >/dev/null 2>&1; then
    CONDA_EXE="$(command -v mamba)"
    return 0
  fi
  if command -v conda >/dev/null 2>&1; then
    CONDA_EXE="$(command -v conda)"
    return 0
  fi
  if command -v micromamba >/dev/null 2>&1; then
    CONDA_EXE="$(command -v micromamba)"
    return 0
  fi
  if [ -x "/opt/conda/bin/mamba" ]; then
    CONDA_EXE="/opt/conda/bin/mamba"
    return 0
  fi
  if [ -x "/opt/conda/bin/conda" ]; then
    CONDA_EXE="/opt/conda/bin/conda"
    return 0
  fi
  if [ -x "${CONDA_PREFIX_DIR}/bin/mamba" ]; then
    CONDA_EXE="${CONDA_PREFIX_DIR}/bin/mamba"
    return 0
  fi
  if [ -x "${CONDA_PREFIX_DIR}/bin/conda" ]; then
    CONDA_EXE="${CONDA_PREFIX_DIR}/bin/conda"
    return 0
  fi

  if [ "${RSCNAV_SKIP_SETUP:-0}" = "1" ]; then
    log "No conda/mamba found and RSCNAV_SKIP_SETUP=1; cannot continue"
    return 1
  fi

  log "No conda/mamba found; installing Miniforge under ${CONDA_PREFIX_DIR}"
  mkdir -p "$(dirname "${CONDA_PREFIX_DIR}")"
  local installer="${OUT_DIR}/Miniforge3-Linux-x86_64.sh"
  curl -L -o "${installer}" "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"
  bash "${installer}" -b -p "${CONDA_PREFIX_DIR}"
  CONDA_EXE="${CONDA_PREFIX_DIR}/bin/mamba"
}

conda_run() {
  "${CONDA_EXE}" run -n "${ENV_NAME}" "$@"
}

ensure_env() {
  if "${CONDA_EXE}" env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    log "Conda env ${ENV_NAME} already exists"
  else
    log "Creating conda env ${ENV_NAME}"
    "${CONDA_EXE}" env create -f "${ROOT}/envs/rscnav-habitat22.yml"
  fi

  log "Installing/confirming habitat-sim headless"
  "${CONDA_EXE}" install -n "${ENV_NAME}" habitat-sim headless -c conda-forge -c aihabitat -y
  "${CONDA_EXE}" install -n "${ENV_NAME}" pillow=10.4.0 numpy=1.26.4 -c conda-forge -y

  if [ ! -d "${HABITAT_LAB_DIR}/.git" ]; then
    log "Cloning Habitat-Lab stable into ${HABITAT_LAB_DIR}"
    mkdir -p "$(dirname "${HABITAT_LAB_DIR}")"
    git clone --branch stable --depth 1 https://github.com/facebookresearch/habitat-lab.git "${HABITAT_LAB_DIR}"
  fi
  log "Installing/confirming habitat-lab editable"
  conda_run python -m pip install -e "${HABITAT_LAB_DIR}/habitat-lab"
}

download_test_scene_if_needed() {
  if [ -n "${SCENE_PATH}" ]; then
    return 0
  fi
  if [ "${DOWNLOAD_TEST_SCENES}" != "1" ]; then
    return 0
  fi
  log "No RSCNAV_HABITAT_SCENE set; downloading Habitat test scenes to ${DATA_PATH}"
  conda_run python -m habitat_sim.utils.datasets_download --uids habitat_test_scenes --data-path "${DATA_PATH}"
  local found
  found="$(find "${DATA_PATH}" -type f -name '*.glb' | head -n 1 || true)"
  if [ -n "${found}" ]; then
    SCENE_PATH="${found}"
    log "Using downloaded test scene: ${SCENE_PATH}"
  fi
}

run_smokes() {
  run_capture import_check conda_run python - <<'PY'
import json
import platform
import sys
import cv2
import habitat
import habitat_sim
import numpy
import PIL
print(json.dumps({
    "python": sys.version,
    "implementation": platform.python_implementation(),
    "habitat_sim": habitat_sim.__version__,
    "habitat": habitat.__version__,
    "numpy": numpy.__version__,
    "pillow": PIL.__version__,
    "cv2": cv2.__version__,
}, indent=2))
PY

  run_capture pip_check conda_run python -m pip check
  run_capture contract_smoke conda_run python "${ROOT}/scripts/phase22_habitat_adapter_contract_test.py"
  run_capture none_live_smoke conda_run python "${ROOT}/scripts/phase22_habitat_sim_none_smoke.py"

  download_test_scene_if_needed
  if [ -n "${SCENE_PATH}" ] && [ -f "${SCENE_PATH}" ]; then
    run_capture scene_live_smoke conda_run python "${ROOT}/scripts/phase22_habitat_live_scene_smoke.py" \
      --scene "${SCENE_PATH}" \
      --out-dir "${ROOT}/outputs/phase22_live"
  else
    log "SKIP scene_live_smoke: no scene path available"
    printf '%s\n' "skipped: no scene path" >"${LOG_DIR}/scene_live_smoke.exit"
  fi
}

make_bundle() {
  local bundle="${ROOT}/outputs/phase22_remote/${HOST}_${STAMP}.tar.gz"
  {
    echo "out_dir=${OUT_DIR}"
    echo "bundle=${bundle}"
    echo "scene_path=${SCENE_PATH}"
    echo "env_name=${ENV_NAME}"
  } >"${OUT_DIR}/run_summary.txt"

  for dir in "${ROOT}/outputs/phase22" "${ROOT}/outputs/phase22_sim" "${ROOT}/outputs/phase22_live"; do
    if [ -d "${dir}" ]; then
      mkdir -p "${OUT_DIR}/repo_outputs"
      cp -a "${dir}" "${OUT_DIR}/repo_outputs/"
    fi
  done

  tar -czf "${bundle}" -C "$(dirname "${OUT_DIR}")" "$(basename "${OUT_DIR}")"
  log "Result bundle: ${bundle}"
}

on_exit() {
  local code=$?
  printf '%s\n' "${code}" >"${OUT_DIR}/runner_exit_code.txt"
  if [ -d "${OUT_DIR}" ] && [ ! -f "${ROOT}/outputs/phase22_remote/${HOST}_${STAMP}.tar.gz" ]; then
    set +e
    make_bundle >/dev/null 2>&1
  fi
  exit "${code}"
}

main() {
  set -e
  trap on_exit EXIT
  log "Phase 2.2 remote Linux runner started"
  write_env_report
  find_or_install_conda
  if [ "${RSCNAV_SKIP_SETUP:-0}" != "1" ]; then
    ensure_env
  else
    log "RSCNAV_SKIP_SETUP=1; skipping conda/Habitat setup"
  fi
  run_smokes
  make_bundle
  log "Done"
}

main "$@"
