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
TEST_SCENE_MIRROR_URL="${RSCNAV_TEST_SCENE_URL:-https://hf-mirror.com/datasets/ai-habitat/habitat_test_scenes/resolve/main/apartment_1.glb}"

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

cuda_device_candidates() {
  if [ -n "${RSCNAV_CUDA_DEVICE_TRIES:-}" ]; then
    printf '%s\n' "${RSCNAV_CUDA_DEVICE_TRIES}" | tr ',' ' '
    return 0
  fi
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | tr '\n' ' '
    return 0
  fi
  printf '%s\n' "default"
}

habitat_gpu_device_candidates() {
  if [ -n "${RSCNAV_HABITAT_GPU_DEVICE_TRIES:-}" ]; then
    printf '%s\n' "${RSCNAV_HABITAT_GPU_DEVICE_TRIES}" | tr ',' ' '
    return 0
  fi
  cuda_device_candidates
}

run_capture_habitat_gpu_retry() {
  local name="$1"
  shift
  local cuda_candidates
  local habitat_candidates
  cuda_candidates="$(cuda_device_candidates)"
  habitat_candidates="$(habitat_gpu_device_candidates)"
  if [ -z "${cuda_candidates// }" ]; then
    cuda_candidates="default"
  fi
  if [ -z "${habitat_candidates// }" ]; then
    habitat_candidates="default"
  fi

  local last_attempt=""
  local selected_cuda=""
  local selected_habitat_gpu=""

  try_habitat_gpu_attempt() {
    local attempt_name="$1"
    local cuda_visible="$2"
    local habitat_gpu="$3"
    shift 3

    local args=("$@")
    if [ "${habitat_gpu}" != "default" ]; then
      args+=("--gpu-device-id" "${habitat_gpu}")
    fi

    last_attempt="${attempt_name}"
    if [ "${cuda_visible}" = "default" ]; then
      run_capture "${attempt_name}" "${args[@]}"
    else
      CUDA_VISIBLE_DEVICES="${cuda_visible}" run_capture "${attempt_name}" "${args[@]}"
    fi

    local code
    code="$(cat "${LOG_DIR}/${attempt_name}.exit" 2>/dev/null || printf '1')"
    if [ "${code}" = "0" ]; then
      selected_cuda="${cuda_visible}"
      selected_habitat_gpu="${habitat_gpu}"
      return 0
    fi
    return 1
  }

  if try_habitat_gpu_attempt "${name}.default" "default" "default" "$@"; then
    :
  else
    local habitat_gpu
    local cuda_device
    for habitat_gpu in ${habitat_candidates}; do
      if [ "${habitat_gpu}" = "default" ]; then
        continue
      fi
      if try_habitat_gpu_attempt "${name}.gpu${habitat_gpu}" "default" "${habitat_gpu}" "$@"; then
        break
      fi
    done

    if [ -z "${selected_habitat_gpu}" ]; then
      for cuda_device in ${cuda_candidates}; do
        if [ "${cuda_device}" = "default" ]; then
          continue
        fi
        if try_habitat_gpu_attempt "${name}.cuda${cuda_device}.gpu0" "${cuda_device}" "0" "$@"; then
          break
        fi
        if [ "${cuda_device}" != "0" ]; then
          if try_habitat_gpu_attempt "${name}.cuda${cuda_device}.gpu${cuda_device}" "${cuda_device}" "${cuda_device}" "$@"; then
            break
          fi
        fi
      done
    fi
  fi

  if [ -n "${selected_habitat_gpu}" ]; then
    cp "${LOG_DIR}/${last_attempt}.stdout.log" "${LOG_DIR}/${name}.stdout.log"
    cp "${LOG_DIR}/${last_attempt}.stderr.log" "${LOG_DIR}/${name}.stderr.log"
    printf '0\n' >"${LOG_DIR}/${name}.exit"
    printf '%s\n' "${selected_cuda}" >"${LOG_DIR}/${name}.selected_cuda_visible_devices"
    printf '%s\n' "${selected_habitat_gpu}" >"${LOG_DIR}/${name}.selected_habitat_gpu_device_id"
    log "OK ${name} selected_cuda_visible_devices=${selected_cuda} selected_habitat_gpu_device_id=${selected_habitat_gpu}"
    return 0
  fi

  if [ -n "${last_attempt}" ]; then
    cp "${LOG_DIR}/${last_attempt}.stdout.log" "${LOG_DIR}/${name}.stdout.log"
    cp "${LOG_DIR}/${last_attempt}.stderr.log" "${LOG_DIR}/${name}.stderr.log"
    cp "${LOG_DIR}/${last_attempt}.exit" "${LOG_DIR}/${name}.exit"
  else
    printf '1\n' >"${LOG_DIR}/${name}.exit"
  fi
  log "FAIL ${name}; tried CUDA candidates: ${cuda_candidates}; Habitat gpu candidates: ${habitat_candidates}"
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
    echo "## nvidia-smi gpu indices"
    nvidia-smi --query-gpu=index,name,uuid --format=csv,noheader 2>/dev/null || true
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

conda_env_prefix() {
  conda_run python -c 'import sys; print(sys.prefix)'
}

ensure_conda_nvidia_egl_vendor() {
  if ! ldconfig -p 2>/dev/null | grep -q 'libEGL_nvidia.so.0'; then
    log "No libEGL_nvidia.so.0 in ldconfig; skipping conda GLVND NVIDIA vendor setup"
    return 0
  fi

  local prefix
  prefix="$(conda_env_prefix)"
  local vendor_dir="${prefix}/etc/glvnd/egl_vendor.d"
  local vendor_file="${vendor_dir}/10_nvidia.json"
  mkdir -p "${vendor_dir}"
  cat >"${vendor_file}" <<'JSON'
{
    "file_format_version" : "1.0.0",
    "ICD" : {
        "library_path" : "libEGL_nvidia.so.0"
    }
}
JSON
  log "Wrote conda NVIDIA EGL vendor JSON: ${vendor_file}"
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
  if ! conda_run python -m habitat_sim.utils.datasets_download --uids habitat_test_scenes --data-path "${DATA_PATH}"; then
    log "Habitat dataset downloader failed; trying direct mirror scene download"
    local mirror_dir="${DATA_PATH}/versioned_data/habitat_test_scenes"
    mkdir -p "${mirror_dir}"
    if command -v curl >/dev/null 2>&1; then
      curl -L --retry 3 --retry-delay 2 -o "${mirror_dir}/apartment_1.glb" "${TEST_SCENE_MIRROR_URL}"
    else
      log "curl unavailable; cannot download mirror scene"
    fi
  fi
  local found
  found="$(find "${DATA_PATH}" -type f -name '*.glb' -size +1M | head -n 1 || true)"
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
  run_capture_habitat_gpu_retry none_live_smoke conda_run python "${ROOT}/scripts/phase22_habitat_sim_none_smoke.py"

  download_test_scene_if_needed
  if [ -n "${SCENE_PATH}" ] && [ -f "${SCENE_PATH}" ]; then
    run_capture_habitat_gpu_retry scene_live_smoke conda_run python "${ROOT}/scripts/phase22_habitat_live_scene_smoke.py" \
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
  ensure_conda_nvidia_egl_vendor
  run_smokes
  make_bundle
  log "Done"
}

main "$@"
