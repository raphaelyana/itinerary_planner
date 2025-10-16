#!/usr/bin/env bash
set -euo pipefail

# Resolve repository root so the script works regardless of call location.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Resolve environment files (supports .env + aura.env overrides).
declare -a _env_files=()
if [[ -n "${AURA_ENV_FILE:-}" ]]; then
  _env_files+=("${AURA_ENV_FILE}")
else
  for candidate in "${ROOT_DIR}/.env" "${ROOT_DIR}/scripts/aura.env"; do
    [[ -f "${candidate}" ]] && _env_files+=("${candidate}")
  done
fi

if [[ ${#_env_files[@]} -eq 0 ]]; then
  cat <<EOF >&2
[run_with_aura] No environment file found.
Create .env at the repo root or scripts/aura.env with entries like:

NEO4J_URI=neo4j+s://<your-aura-host>
NEO4J_USERNAME=<your-username>
NEO4J_PASSWORD=<your-password>

You can point to a custom file by exporting AURA_ENV_FILE before running this script.
EOF
  exit 1
fi

# shellcheck source=/dev/null
set -a
for env_file in "${_env_files[@]}"; do
  source "${env_file}"
done
set +a

if [[ -z "${NEO4J_URI:-}" || -z "${NEO4J_USERNAME:-}" || -z "${NEO4J_PASSWORD:-}" ]]; then
  cat <<EOF >&2
[run_with_aura] Environment variables NEO4J_URI, NEO4J_USERNAME, and NEO4J_PASSWORD must be set.
Files sourced: ${_env_files[*]}
EOF
  exit 1
fi

# Accept a custom command; default to launching the API locally.
if [[ $# -eq 0 ]]; then
  set -- uvicorn scripts.api:app --host 127.0.0.1 --port 8000
fi

exec "$@"
