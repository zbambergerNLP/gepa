#!/bin/bash
# Open, check, or close the persistent SSH master connections to Della.
#
# Della authenticates with an SSH key AND a keyboard-interactive step (password,
# plus Duo when off-campus). Gilad's launcher scripts run ssh/rsync with
# BatchMode=yes, which cannot answer prompts. This helper opens one interactive
# master connection per host; every later BatchMode ssh/rsync reuses it through
# ControlMaster multiplexing, so the launcher never sees a prompt.
#
# Requires this block in ~/.ssh/config (see examples/hotpotqa/DELLA_CAMPAIGN.md):
#   Host della.princeton.edu della-vis1.princeton.edu
#       User <netid>
#       IdentityFile ~/.ssh/id_ed25519
#       IdentitiesOnly yes
#       ControlMaster auto
#       ControlPath ~/.ssh/cm/%r@%h:%p
#       ControlPersist yes
#
# Usage:
#   scripts/della/della_session.sh open    # prompts for password (+ Duo) once per host
#   scripts/della/della_session.sh status
#   scripts/della/della_session.sh close
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ENV_FILE="${SCRIPT_DIR}/.env"
if [[ ! -f "${ENV_FILE}" ]]; then
    echo "ERROR: ${ENV_FILE} not found." >&2
    exit 1
fi
source "${ENV_FILE}"

HOSTS=("${REMOTE_HOST}" "${REMOTE_VIS_HOST}")
ACTION="${1:-status}"

master_alive() {
    ssh -o BatchMode=yes -O check "${REMOTE_USER}@$1" >/dev/null 2>&1
}

case "${ACTION}" in
    open)
        mkdir -p ~/.ssh/cm && chmod 700 ~/.ssh/cm
        for host in "${HOSTS[@]}"; do
            if master_alive "${host}"; then
                echo "==> ${host}: master already running"
                continue
            fi
            echo "==> ${host}: opening master connection (enter password / approve Duo if asked)"
            ssh -fN -o ControlMaster=yes -o StrictHostKeyChecking=yes "${REMOTE_USER}@${host}"
            ssh -o BatchMode=yes -o StrictHostKeyChecking=yes "${REMOTE_USER}@${host}" \
                'echo "    BatchMode ok on $(hostname)"'
        done
        ;;
    status)
        rc=0
        for host in "${HOSTS[@]}"; do
            if master_alive "${host}"; then
                echo "${host}: master running"
            else
                echo "${host}: NO master (run: $0 open)"
                rc=1
            fi
        done
        exit "${rc}"
        ;;
    close)
        for host in "${HOSTS[@]}"; do
            ssh -O exit "${REMOTE_USER}@${host}" 2>/dev/null && echo "${host}: closed" || echo "${host}: no master"
        done
        ;;
    *)
        echo "usage: $0 {open|status|close}" >&2
        exit 2
        ;;
esac
