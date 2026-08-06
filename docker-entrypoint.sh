#!/bin/sh
set -e

# Run the server as the non-root "lightrag" user while staying compatible with
# deployments whose bind-mounted data dirs are root-owned.
#
# Image-internal chown cannot fix runtime mounts, so when the container starts
# as root we create + chown the writable data dirs (covering bind mounts whose
# leaf does not exist yet, e.g. only the parent is mounted) and then drop
# privileges via gosu. When the orchestrator already starts us as non-root
# (compose `user:` / k8s `runAsUser`), we skip the chown and exec directly.
#
# Preserve the pre-split behavior where `docker run <image> --port 9622`
# appended flags to the server. Now that ENTRYPOINT is this script, a first arg
# starting with "-" means the user only passed flags, so prepend the default
# command.
# Keep host-side HOME/cache variables from leaking through --env-file into the
# fixed non-root container user. Libraries may otherwise try to create caches
# under paths such as /home/ubuntu, which are not writable inside the image.
HOME=/home/lightrag
XDG_CACHE_HOME=/home/lightrag/.cache
PIP_CACHE_DIR=/home/lightrag/.cache/pip
TMPDIR=/tmp
export HOME XDG_CACHE_HOME PIP_CACHE_DIR TMPDIR

if [ "${1#-}" != "$1" ]; then
    set -- python -m lightrag.api.lightrag_server "$@"
fi

if [ "$(id -u)" = "0" ]; then
    # ERROR_LOG/ACCESS_LOG (gunicorn) are *file* paths, so we chown their parent
    # directory rather than the file itself. Unset values and system roots are
    # skipped; read-only mounts (PROMPT_DIR) fail the mkdir/chown harmlessly.
    _error_log_dir=""
    _access_log_dir=""
    [ -n "$ERROR_LOG" ] && _error_log_dir=$(dirname "$ERROR_LOG")
    [ -n "$ACCESS_LOG" ] && _access_log_dir=$(dirname "$ACCESS_LOG")
    for _d in "$WORKING_DIR" "$INPUT_DIR" "$LOG_DIR" "$PROMPT_DIR" \
        "$TIKTOKEN_CACHE_DIR" "$_error_log_dir" "$_access_log_dir"; do
        case "$_d" in
            ""|.|/|/bin|/boot|/dev|/etc|/home|/lib|/lib64|/proc|/root|/run|/sbin|/sys|/usr|/var) continue ;;
        esac
        mkdir -p "$_d" 2>/dev/null || true
        chown -R lightrag:lightrag "$_d" 2>/dev/null || true
    done
    # NOTE: we deliberately do NOT touch /app/.env. It is baked into the image
    # (LIGHTRAG_RUNTIME_TARGET=compose, no secrets) and the real config arrives
    # via env_file/environment, so there is no host file to corrupt.
    exec gosu lightrag "$@"
fi

exec "$@"
