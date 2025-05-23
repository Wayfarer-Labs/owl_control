#!/bin/bash

echo "🧪 Quick Packaging Test for OWL Control"
echo "======================================="

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to check command exists
check_command() {
    if ! command -v $1 &> /dev/null; then
        echo -e "${RED}❌ $1 is not installed${NC}"
        return 1
    else
        echo -e "${GREEN}✅ $1 is available${NC}"
        return 0
    fi
}

# Check prerequisites
echo -e "\n${YELLOW}Checking prerequisites...${NC}"
check_command node || exit 1
check_command npm || exit 1
check_command python || check_command python3 || exit 1
check_command pip || check_command pip3 || exit 1

# Quick test - just bundle Python without full packaging
echo -e "\n${YELLOW}Running quick Python bundle test...${NC}"

# Clean previous Python dist
rm -rf python-dist/
rm -f vg_control.spec

# Install Python deps
echo "Installing Python dependencies..."
pip install -r requirements.txt || pip3 install -r requirements.txt

# Run Python bundler
echo "Bundling Python code..."
node scripts/bundle-python.js

# Check results
if [ -d "python-dist" ]; then
    echo -e "\n${GREEN}✅ Python bundling successful!${NC}"
    echo "Contents of python-dist:"
    ls -la python-dist/
    
    # Test if the bundled executable works
    if [ -f "python-dist/vg_control_backend/vg_control_backend" ] || [ -f "python-dist/vg_control_backend/vg_control_backend.exe" ]; then
        echo -e "\n${GREEN}✅ Python executable found!${NC}"
    else
        echo -e "\n${RED}❌ Python executable not found${NC}"
    fi
else
    echo -e "\n${RED}❌ Python bundling failed!${NC}"
    exit 1
fi

echo -e "\n${YELLOW}Quick test completed!${NC}"
echo "To run full packaging test: node scripts/test-packaging.js"