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
# It also generates stub Info.plist files INSIDE each .framework slice.
# Static frameworks don't strictly need an Info.plist, but Xcode's
# binaryTarget auto-embed step copies the .framework into <App>.app/
# Frameworks/ at build time and refuses to copy a framework without an
# Info.plist. The stub satisfies the requirement without changing semantics
# (the inner binary is still a static lib that's force-loaded into the
# main app binary).
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

    # Stub Info.plists inside each slice's .framework so Xcode's auto-embed
    # build step can copy them into <App>.app/Frameworks/.
    for slice_dir in "${VENDOR_DIR}/${name}.xcframework"/*/; do
        local fw_plist="${slice_dir}${name}.framework/Info.plist"
        # If the framework dir doesn't exist (skipped slice), continue.
        [[ -d "${slice_dir}${name}.framework" ]] || continue
        /usr/libexec/PlistBuddy \
            -c "Clear dict" \
            -c "Add :CFBundleDevelopmentRegion string en" \
            -c "Add :CFBundleExecutable string ${name}" \
            -c "Add :CFBundleIdentifier string com.google.mediapipe.${name}" \
            -c "Add :CFBundleInfoDictionaryVersion string 6.0" \
            -c "Add :CFBundleName string ${name}" \
            -c "Add :CFBundlePackageType string FMWK" \
            -c "Add :CFBundleShortVersionString string 0.10.33" \
            -c "Add :CFBundleSupportedPlatforms array" \
            -c "Add :CFBundleVersion string 0.10.33" \
            -c "Add :MinimumOSVersion string 17.4" \
            "${fw_plist}" >/dev/null
    done
    echo "  [ok]   ${name}: stub Info.plist written inside each .framework slice"
}

echo "Repackaging xcframeworks in ${VENDOR_DIR}..."
repackage "MediaPipeTasksVision"
repackage "MediaPipeTasksCommon"
echo "Done."
