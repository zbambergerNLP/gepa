#!/bin/bash
# Keep Harbor, task containers, and local model servers on the same private loopback.
set -euo pipefail

if [[ $# -eq 0 ]]; then
    printf 'Usage: %s COMMAND [ARG ...]\n' "${0##*/}" >&2
    exit 2
fi
if [[ "$(uname -s)" != Linux ]]; then
    echo "Private network execution requires Linux" >&2
    exit 1
fi
for required_command in unshare ip; do
    if ! command -v "${required_command}" >/dev/null; then
        printf 'Missing required command: %s\n' "${required_command}" >&2
        exit 1
    fi
done

exec unshare --user --map-root-user --net -- bash -c '
    set -euo pipefail
    ip link set lo up
    export VLLM_HOST_IP=127.0.0.1
    export NCCL_SOCKET_IFNAME="=lo"
    export GLOO_SOCKET_IFNAME=lo
    exec "$@"
' with-private-network "$@"
