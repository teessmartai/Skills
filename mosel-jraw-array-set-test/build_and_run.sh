#!/bin/bash
# ==============================================================
# Build and run script for testing Mosel jraw array-of-set binding
#
# Prerequisites:
#   - FICO Xpress installed (community edition is fine)
#   - XPRESSDIR environment variable set (e.g. /opt/xpressmp)
#   - Java JDK 11+
#
# Usage:
#   chmod +x build_and_run.sh
#   ./build_and_run.sh
# ==============================================================

set -e

# -- Verify environment --
if [ -z "$XPRESSDIR" ]; then
    echo "ERROR: XPRESSDIR environment variable is not set."
    echo "Set it to your Xpress installation directory, e.g.:"
    echo "  export XPRESSDIR=/opt/xpressmp"
    exit 1
fi

XPRM_JAR="$XPRESSDIR/lib/xprm.jar"
if [ ! -f "$XPRM_JAR" ]; then
    echo "ERROR: xprm.jar not found at $XPRM_JAR"
    echo "Ensure FICO Xpress is properly installed with Mosel/Java support."
    exit 1
fi

echo "Using XPRESSDIR=$XPRESSDIR"
echo "Using xprm.jar=$XPRM_JAR"
echo ""

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SRC_DIR="$SCRIPT_DIR/src"
BUILD_DIR="$SCRIPT_DIR/build"

# -- Clean & create build dir --
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"

# -- Copy .mos files to build dir (mosel needs them at runtime) --
cp "$SRC_DIR"/*.mos "$BUILD_DIR/"

# -- Compile Java --
echo "Compiling Java files..."
javac -cp "$XPRM_JAR" -d "$BUILD_DIR" "$SRC_DIR"/*.java
echo "Java compilation successful."
echo ""

# -- Run tests --
echo "Running tests..."
echo "=============================================="
cd "$BUILD_DIR"
java \
    -cp ".:$XPRM_JAR" \
    -Djava.library.path="$XPRESSDIR/lib" \
    -DXPRESSDIR="$XPRESSDIR" \
    TestArrayOfSet 2>&1
echo "=============================================="
echo ""
echo "Done. Check output above for results."
