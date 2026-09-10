#!/usr/bin/env bash
# Compatibility entrypoint for run_multiseed.sh.
# Keep the minimal bundle's mission framework as the implementation.
set -Eeuo pipefail

BUNDLE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$BUNDLE_ROOT/mission/run_native.sh" "$@"
