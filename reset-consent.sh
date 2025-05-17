#\!/bin/sh
# Reset consent script for macOS

# Get script directory
SCRIPT_DIR=$(dirname "$0")

# Run the JS script
node "$SCRIPT_DIR/reset-consent.js"

echo "Script execution complete. Restart the VG Control app now."
