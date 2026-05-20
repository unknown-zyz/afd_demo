#!/usr/bin/env bash
# Generate gRPC Python stubs from coordinator.proto.
# Run inside the NPU container (has grpcio-tools) or any env with grpcio-tools.
#
# Usage: bash scripts/gen_proto.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PROTO_DIR="${REPO_ROOT}/src/coordinator_arch/proto"
OUT_DIR="${PROTO_DIR}"

if ! python3 -c "import grpc_tools" 2>/dev/null; then
  echo "[gen_proto] grpcio-tools not found, installing..."
  pip install grpcio-tools >/dev/null
fi

python3 -m grpc_tools.protoc \
  -I"${PROTO_DIR}" \
  --python_out="${OUT_DIR}" \
  --grpc_python_out="${OUT_DIR}" \
  "${PROTO_DIR}/coordinator.proto"

# Fix relative import in generated _grpc.py (protoc emits `import coordinator_pb2`
# which doesn't work as a package import). Patch to `from . import coordinator_pb2`.
GRPC_FILE="${OUT_DIR}/coordinator_pb2_grpc.py"
if [[ -f "${GRPC_FILE}" ]]; then
  sed -i.bak 's/^import coordinator_pb2/from . import coordinator_pb2/' "${GRPC_FILE}"
  rm -f "${GRPC_FILE}.bak"
fi

echo "[gen_proto] OK. Generated:"
ls -la "${OUT_DIR}"/coordinator_pb2*.py
