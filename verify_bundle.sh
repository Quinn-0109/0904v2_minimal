#!/usr/bin/env bash
set -Eeuo pipefail

BUNDLE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$BUNDLE_ROOT"
sha256sum --check --quiet --strict MANIFEST.sha256
for entry in run.sh preflight.sh verify_bundle.sh mission/run_native.sh; do
  [ -x "$entry" ] || { echo "ERROR: not executable: $entry" >&2; exit 3; }
done
while IFS= read -r link; do
  resolved="$(readlink -f "$link" || true)"
  case "$resolved" in
    "$BUNDLE_ROOT"/*|/opt/ros/noetic/*|/opt/libtorch/*) ;;
    *) echo "ERROR: broken or external symlink: $link -> $resolved" >&2; exit 4 ;;
  esac
done < <(find "$BUNDLE_ROOT" -type l -print)
echo BUNDLE_VERIFY_OK
