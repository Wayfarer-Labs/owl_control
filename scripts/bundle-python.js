const { execSync } = require('child_process');
const fs = require('fs');
const path = require('path');

const pythonDistDir = path.join(__dirname, '..', 'python-dist');

// Create python-dist directory
if (!fs.existsSync(pythonDistDir)) {
  fs.mkdirSync(pythonDistDir, { recursive: true });
}

console.log('Installing Python dependencies...');
try {
  // Install dependencies using pip
  execSync('pip install -r requirements.txt', { stdio: 'inherit' });
} catch (error) {
  console.log('No requirements.txt found, trying poetry...');
  execSync('poetry install --no-dev', { stdio: 'inherit' });
}

console.log('Creating PyInstaller spec file...');
const specContent = `# -*- mode: python ; coding: utf-8 -*-

import sys
from pathlib import Path

block_cipher = None

# Get the path to vg_control module
vg_control_path = Path('${path.join(__dirname, '..', 'vg_control').replace(/\\/g, '/')}')

a = Analysis(
    ['${path.join(__dirname, '..', 'vg_control', '__init__.py').replace(/\\/g, '/')}'],
    pathex=[str(vg_control_path.parent)],
    binaries=[
        ('${path.join(__dirname, '..', 'rawinputlib.dll').replace(/\\/g, '/')}', '.')
    ] if sys.platform == 'win32' else [],
    datas=[
        (str(vg_control_path), 'vg_control')
    ],
    hiddenimports=[
        'pynput',
        'pynput.keyboard',
        'pynput.mouse',
        'numpy',
        'cv2',
        'dotenv',
        'vg_control.auth',
        'vg_control.data',
        'vg_control.input_tracking',
        'vg_control.video',
        'vg_control.recorder',
        'vg_control.recording_bridge',
        'vg_control.upload_bridge',
        'vg_control.upload_worker',
    ] + (['win32api', 'win32con', 'win32gui', 'pywin32'] if sys.platform == 'win32' else []),
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='vg_control_backend',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='vg_control_backend',
)
`;

fs.writeFileSync(path.join(__dirname, '..', 'vg_control.spec'), specContent);

console.log('Building Python executable with PyInstaller...');
execSync('pyinstaller vg_control.spec --distpath python-dist --workpath build/pyinstaller --clean -y', { 
  stdio: 'inherit',
  cwd: path.join(__dirname, '..')
});

console.log('Python bundling complete!');