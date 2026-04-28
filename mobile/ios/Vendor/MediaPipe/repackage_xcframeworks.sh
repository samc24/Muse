#!/usr/bin/env bash
# Repackage MediaPipe xcframeworks to fix Info.plist metadata.
#
# Google's MediaPipe iOS xcframeworks ship with Info.plist metadata that
# describes the wrong layout: it claims `LibraryPath = MediaPipeTasksVision.a`
# (static library at slice root) but actually packages a static framework
# (`MediaPipeTasksVision.framework/MediaPipeTasksVision`, where the binary
# inside the .framework is itself a static lib). This works under CocoaPods
# because CocoaPods rewrites linker flags, but breaks under direct
# xcframework consumption (Swift Package Manager binaryTarget, manual
# Xcode UI add).
#
# This script fixes both Info.plist files in place by:
#   1. Setting LibraryPath to <Name>.framework
#   2. Removing HeadersPath (the .framework's own Headers/ is used instead)
#
# Run ONCE after dropping new MediaPipe xcframeworks into Vendor/MediaPipe/.
# Idempotent -- safe to re-run.

set -euo pipefail

VENDOR_DIR="$(cd "$(dirname "$0")" && pwd)"

repackage() {
    local name="$1"
    local plist="${VENDOR_DIR}/${name}.xcframework/Info.plist"

    if [[ ! -f "${plist}" ]]; then
        echo "  [skip] ${plist} not found"
        return
    fi

    # Iterate the entries in AvailableLibraries; for each, set LibraryPath
    # and delete HeadersPath.
    local count
    count=$(/usr/libexec/PlistBuddy -c "Print :AvailableLibraries" "${plist}" \
        | grep -c "^    Dict {" || true)

    for ((i = 0; i < count; i++)); do
        # Set LibraryPath to <Name>.framework (idempotent).
        /usr/libexec/PlistBuddy \
            -c "Set :AvailableLibraries:${i}:LibraryPath ${name}.framework" \
            "${plist}"

        # Remove HeadersPath if present (idempotent).
        /usr/libexec/PlistBuddy \
            -c "Delete :AvailableLibraries:${i}:HeadersPath" \
            "${plist}" 2>/dev/null || true
    done

    echo "  [ok]   ${name}: LibraryPath -> ${name}.framework, HeadersPath removed"
}

echo "Repackaging xcframeworks in ${VENDOR_DIR}..."
repackage "MediaPipeTasksVision"
repackage "MediaPipeTasksCommon"
echo "Done."
