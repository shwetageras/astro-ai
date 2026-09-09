#!/usr/bin/env bash
set -euo pipefail

SERVICE="${1:-}"

REPO_DIR="${REPO_DIR:-/home/ubuntu/astro-ai-split}"
APP_DIR="${APP_DIR:-/home/ubuntu/astro-ai}"
STATE_DIR="${STATE_DIR:-/var/lib/astro-ai-deploy}"

if [[ -z "$SERVICE" ]]; then
    echo "Usage: $0 <service>"
    exit 1
fi

MANIFEST="$REPO_DIR/deployment/manifests/${SERVICE}.txt"
STATE_FILE="$STATE_DIR/${SERVICE}.files"

if [[ ! -d "$REPO_DIR/.git" ]]; then
    echo "ERROR: Git repository not found: $REPO_DIR"
    exit 1
fi

if [[ ! -f "$MANIFEST" ]]; then
    echo "ERROR: Manifest not found: $MANIFEST"
    exit 1
fi

case "$SERVICE" in
    chat)       SYSTEMD_UNIT="api-chat.service" ;;
    retrieval)  SYSTEMD_UNIT="retrieval-api.service" ;;
    llm)        SYSTEMD_UNIT="llm-api.service" ;;
    embedding)  SYSTEMD_UNIT="embedding-api.service" ;;
    chart)      SYSTEMD_UNIT="api-chart.service" ;;
    admin)      SYSTEMD_UNIT="api-admin.service" ;;
    *)
        echo "ERROR: Unsupported service: $SERVICE"
        exit 1
        ;;
esac

STAGE="$(mktemp -d)"
BACKUP="$(mktemp -d)"

cleanup() {
    rm -rf "$STAGE" "$BACKUP"
}
trap cleanup EXIT

echo "=== Deploying $SERVICE ==="
echo "Repository: $REPO_DIR"
echo "Runtime:    $APP_DIR"
echo "Manifest:   $MANIFEST"
echo "Systemd:    $SYSTEMD_UNIT"
echo

declare -a TARGETS=()

while IFS= read -r line; do
    [[ -z "$line" ]] && continue

    SOURCE="${line%% -> *}"
    TARGET="${line#* -> }"

    if [[ "$SOURCE" == "$TARGET" ]]; then
        echo "ERROR: Invalid manifest line: $line"
        exit 1
    fi

    if [[ ! -f "$REPO_DIR/$SOURCE" ]]; then
        echo "ERROR: Source file missing: $REPO_DIR/$SOURCE"
        exit 1
    fi

    cp "$REPO_DIR/$SOURCE" "$STAGE/$TARGET"
    TARGETS+=("$TARGET")
done < "$MANIFEST"

echo "--- Staged files ---"
printf '%s\n' "${TARGETS[@]}" | sort
echo

echo "--- Python validation ---"

PY_FILES=()

for target in "${TARGETS[@]}"; do
    if [[ "$target" == *.py ]]; then
        PY_FILES+=("$STAGE/$target")
    fi
done

if (( ${#PY_FILES[@]} > 0 )); then
    "$APP_DIR/venv/bin/python" -m py_compile "${PY_FILES[@]}"
fi

echo "Python compilation: OK"
echo

echo "--- Backing up current runtime files ---"

sudo mkdir -p "$STATE_DIR"

declare -a PREVIOUS_TARGETS=()

if [[ -f "$STATE_FILE" ]]; then
    while IFS= read -r old_target; do
        [[ -z "$old_target" ]] && continue
        PREVIOUS_TARGETS+=("$old_target")
    done < "$STATE_FILE"
fi

for target in "${TARGETS[@]}"; do
    if [[ -f "$APP_DIR/$target" ]]; then
        cp "$APP_DIR/$target" "$BACKUP/$target"
    fi
done

for old_target in "${PREVIOUS_TARGETS[@]}"; do
    if [[ -f "$APP_DIR/$old_target" && ! -f "$BACKUP/$old_target" ]]; then
        cp "$APP_DIR/$old_target" "$BACKUP/$old_target"
    fi
done

echo "Backup complete."
echo

echo "--- Installing runtime files ---"

for old_target in "${PREVIOUS_TARGETS[@]}"; do
    found=false

    for target in "${TARGETS[@]}"; do
        if [[ "$target" == "$old_target" ]]; then
            found=true
            break
        fi
    done

    if [[ "$found" == false ]]; then
        echo "Removing previously managed file: $APP_DIR/$old_target"
        sudo rm -f "$APP_DIR/$old_target"
    fi
done

for target in "${TARGETS[@]}"; do
    sudo install -m 0644 "$STAGE/$target" "$APP_DIR/$target"
done

printf '%s\n' "${TARGETS[@]}" | sort | sudo tee "$STATE_FILE" >/dev/null

echo "Runtime installation: OK"
echo

echo "--- Restarting service ---"
sudo systemctl restart "$SYSTEMD_UNIT"

echo "--- Verifying service ---"

if sudo systemctl is-active --quiet "$SYSTEMD_UNIT"; then
    echo "$SYSTEMD_UNIT: ACTIVE"
    echo "=== Deployment successful: $SERVICE ==="
    exit 0
fi

echo "ERROR: $SYSTEMD_UNIT failed after deployment."
echo "Starting rollback..."

for target in "${TARGETS[@]}"; do
    sudo rm -f "$APP_DIR/$target"
done

for old_target in "${PREVIOUS_TARGETS[@]}"; do
    sudo rm -f "$APP_DIR/$old_target"
done

for backup_file in "$BACKUP"/*; do
    [[ -e "$backup_file" ]] || continue
    sudo install -m 0644 "$backup_file" "$APP_DIR/$(basename "$backup_file")"
done

if (( ${#PREVIOUS_TARGETS[@]} > 0 )); then
    printf '%s\n' "${PREVIOUS_TARGETS[@]}" | sort | sudo tee "$STATE_FILE" >/dev/null
else
    sudo rm -f "$STATE_FILE"
fi

sudo systemctl restart "$SYSTEMD_UNIT"

if sudo systemctl is-active --quiet "$SYSTEMD_UNIT"; then
    echo "Rollback successful. Previous runtime restored."
else
    echo "CRITICAL: rollback completed, but $SYSTEMD_UNIT is still not active."
    sudo systemctl status "$SYSTEMD_UNIT" --no-pager -l
fi

exit 1
