#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  launch_azureml_ray.sh --dataset-root <mounted_dataset_dir> [--output-root <output_dir>]

Options:
  --dataset-root, -d  Mounted dataset root containing train/val/test directories (required)
  --output-root, -o   Output root directory for Ray artifacts (default: ./ray_results)
  --help, -h          Show this help message
EOF
}

DATASET_ROOT=""
OUTPUT_ROOT="./ray_results"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset-root|-d)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for $1" >&2
        usage
        exit 1
      fi
      DATASET_ROOT="$2"
      shift 2
      ;;
    --output-root|-o)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for $1" >&2
        usage
        exit 1
      fi
      OUTPUT_ROOT="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

RANK=${OMPI_COMM_WORLD_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-${AZ_BATCH_MASTER_NODE%%:*}}
HEAD_PORT=${RAY_HEAD_PORT:-6379}
DASHBOARD_PORT=${RAY_DASHBOARD_PORT:-8265}
NODE_CPUS=${RAY_NODE_CPUS:-4}
NODE_GPUS=${RAY_NODE_GPUS:-1}
BOHB_OPTIMIZATION_PROFILE=${BOHB_OPTIMIZATION_PROFILE:-bayesian_linearinterp_smoke}

cleanup() {
  ray stop >/dev/null 2>&1 || true
}
trap cleanup EXIT

if [[ -z "${MASTER_ADDR}" ]]; then
  echo "MASTER_ADDR could not be resolved" >&2
  exit 1
fi

if [[ -z "${DATASET_ROOT}" ]]; then
  echo "--dataset-root is required. Provide the mounted dataset path from the AML job command." >&2
  usage
  exit 1
fi

if [[ -z "${OUTPUT_ROOT}" ]]; then
  echo "--output-root cannot be empty." >&2
  usage
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SRC_DIR="${REPO_ROOT}/src"

if [[ ! -d "${SRC_DIR}/cfw" ]]; then
  echo "CFW source package not found at expected path: ${SRC_DIR}/cfw" >&2
  exit 1
fi

# Ensure Ray workers on all nodes can import `cfw`.
export PYTHONPATH="${SRC_DIR}:${REPO_ROOT}:${PYTHONPATH:-}"
echo "[rank${RANK}] PYTHONPATH=${PYTHONPATH}"

if [[ "${RANK}" == "0" ]]; then
  echo "[rank0] Starting Ray head on ${MASTER_ADDR}:${HEAD_PORT}"
  ray start --head \
    --node-ip-address="${MASTER_ADDR}" \
    --port="${HEAD_PORT}" \
    --dashboard-host=0.0.0.0 \
    --dashboard-port="${DASHBOARD_PORT}" \
    --num-cpus="${NODE_CPUS}" \
    --num-gpus="${NODE_GPUS}"

  sleep 20
  export RAY_ADDRESS="${MASTER_ADDR}:${HEAD_PORT}"
  export CFW_STATEFARM_MULTICLASS_DIR="${DATASET_ROOT}"

  for split in train val test; do
    if [[ ! -d "${DATASET_ROOT}/${split}" ]]; then
      echo "Expected split directory not found: ${DATASET_ROOT}/${split}" >&2
      exit 1
    fi
  done

  mkdir -p "${OUTPUT_ROOT}/ray_results"

  python scripts/bayesian_optimize.py \
    dataset=statefarm_multiclass \
    model=dinov2_vitb14 \
    dataloader=baseline \
    trainer=single_gpu \
    runtime=azureml \
    augmentation=no_aug \
    dataset.train_dir="${DATASET_ROOT}/train" \
    dataset.val_dir="${DATASET_ROOT}/val" \
    dataset.test_dir="${DATASET_ROOT}/test" \
    trainer.num_epochs=3 \
    +optimization="${BOHB_OPTIMIZATION_PROFILE}" \
    optimization.ray.mode=cluster \
    optimization.ray.address=auto \
    optimization.ray.namespace=cfw-ray-smoke \
    optimization.storage.path="${OUTPUT_ROOT}/ray_results" \
    optimization.storage.name=statefarm_smoke_baseline_bohb \
    optimization.evaluate_test_on_best=true \
    optimization.non_interactive=true
else
  echo "[rank${RANK}] Joining Ray head at ${MASTER_ADDR}:${HEAD_PORT}"
  ray start \
    --address="${MASTER_ADDR}:${HEAD_PORT}" \
    --num-cpus="${NODE_CPUS}" \
    --num-gpus="${NODE_GPUS}"

  # Keep worker alive while head is reachable, then exit 0 so MPI can finish.
  HEAD_POLL_INTERVAL_SEC=${RAY_HEAD_POLL_INTERVAL_SEC:-15}
  HEAD_MISS_LIMIT=${RAY_HEAD_MISS_LIMIT:-4}

  head_alive() {
    python - "$MASTER_ADDR" "$HEAD_PORT" <<'PY'
import socket
import sys

host = sys.argv[1]
port = int(sys.argv[2])

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.settimeout(2.0)
try:
    sock.connect((host, port))
except OSError:
    sys.exit(1)
finally:
    sock.close()

sys.exit(0)
PY
  }

  worker_shutdown() {
    echo "[rank${RANK}] Received shutdown signal, exiting worker cleanly."
    exit 0
  }
  trap worker_shutdown TERM INT HUP

  head_misses=0
  while true; do
    if head_alive; then
      head_misses=0
    else
      head_misses=$((head_misses + 1))
      echo "[rank${RANK}] Ray head ${MASTER_ADDR}:${HEAD_PORT} unreachable (${head_misses}/${HEAD_MISS_LIMIT})"
      if (( head_misses >= HEAD_MISS_LIMIT )); then
        echo "[rank${RANK}] Head appears down. Exiting worker cleanly."
        exit 0
      fi
    fi
    sleep "${HEAD_POLL_INTERVAL_SEC}"
  done
fi
