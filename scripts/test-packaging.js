#!/usr/bin/env node

const { execSync } = require('child_process');
const fs = require('fs');
const path = require('path');

const colors = {
  reset: '\x1b[0m',
  bright: '\x1b[1m',
  green: '\x1b[32m',
  red: '\x1b[31m',
  yellow: '\x1b[33m',
  blue: '\x1b[34m'
};

function log(message, color = colors.reset) {
  console.log(`${color}${message}${colors.reset}`);
}

function runCommand(command, description) {
  log(`\n📦 ${description}...`, colors.blue);
  try {
    execSync(command, { stdio: 'inherit' });
    log(`✅ ${description} completed successfully!`, colors.green);
    return true;
  } catch (error) {
    log(`❌ ${description} failed!`, colors.red);
    console.error(error.message);
    return false;
  }
}

function checkPrerequisites() {
  log('\n🔍 Checking prerequisites...', colors.bright);
  
  const checks = [
    {
      name: 'Node.js',
      command: 'node --version',
      minVersion: '16.0.0'
    },
    {
      name: 'npm',
      command: 'npm --version',
      minVersion: '7.0.0'
    },
    {
      name: 'Python',
      command: 'python --version || python3 --version',
      minVersion: '3.8.0'
    }
  ];

  let allPassed = true;

  for (const check of checks) {
    try {
      const version = execSync(check.command, { encoding: 'utf8' }).trim();
      log(`✅ ${check.name}: ${version}`, colors.green);
    } catch (error) {
      log(`❌ ${check.name} not found! Please install ${check.name} ${check.minVersion} or higher.`, colors.red);
      allPassed = false;
    }
  }

  // Check for required files
  const requiredFiles = [
    'package.json',
    'electron-builder.yml',
    'webpack.config.js',
    'requirements.txt',
    'vg_control/__init__.py',
    'scripts/bundle-python.js'
  ];

  log('\n🔍 Checking required files...', colors.bright);
  for (const file of requiredFiles) {
    const filePath = path.join(__dirname, '..', file);
    if (fs.existsSync(filePath)) {
      log(`✅ ${file} found`, colors.green);
    } else {
      log(`❌ ${file} not found!`, colors.red);
      allPassed = false;
    }
  }

  return allPassed;
}

function cleanBuildArtifacts() {
  log('\n🧹 Cleaning build artifacts...', colors.yellow);
  
  const dirsToClean = ['dist', 'python-dist', 'build'];
  
  for (const dir of dirsToClean) {
    const dirPath = path.join(__dirname, '..', dir);
    if (fs.existsSync(dirPath)) {
      log(`Removing ${dir}/...`);
      fs.rmSync(dirPath, { recursive: true, force: true });
    }
  }
  
  // Clean PyInstaller spec file
  const specFile = path.join(__dirname, '..', 'vg_control.spec');
  if (fs.existsSync(specFile)) {
    fs.unlinkSync(specFile);
  }
}

async function runTests() {
  log('\n🚀 Starting packaging test...', colors.bright);
  
  // Check prerequisites
  if (!checkPrerequisites()) {
    log('\n❌ Prerequisites check failed. Please fix the issues above and try again.', colors.red);
    process.exit(1);
  }

  // Clean previous builds
  cleanBuildArtifacts();

  // Install dependencies
  if (!runCommand('npm ci', 'Installing Node.js dependencies')) {
    process.exit(1);
  }

  // Install Python dependencies
  if (!runCommand('pip install -r requirements.txt', 'Installing Python dependencies')) {
    log('Trying with pip3...', colors.yellow);
    if (!runCommand('pip3 install -r requirements.txt', 'Installing Python dependencies with pip3')) {
      process.exit(1);
    }
  }

  // Build webpack
  if (!runCommand('npm run build', 'Building webpack bundle')) {
    process.exit(1);
  }

  // Bundle Python
  if (!runCommand('npm run bundle-python', 'Bundling Python with PyInstaller')) {
    process.exit(1);
  }

  // Check if Python bundle was created
  const pythonDistPath = path.join(__dirname, '..', 'python-dist');
  if (!fs.existsSync(pythonDistPath)) {
    log('❌ Python distribution directory not created!', colors.red);
    process.exit(1);
  }

  log('\n✅ Python bundle created successfully!', colors.green);

  // Platform-specific packaging
  const platform = process.platform;
  let packageCommand;
  
  switch (platform) {
    case 'win32':
      packageCommand = 'npm run package:win';
      break;
    case 'darwin':
      packageCommand = 'npm run package:mac';
      break;
    case 'linux':
      packageCommand = 'npm run package:linux';
      break;
    default:
      log(`❌ Unsupported platform: ${platform}`, colors.red);
      process.exit(1);
  }

  log(`\n📦 Building for ${platform}...`, colors.blue);
  
  if (!runCommand(packageCommand, `Packaging Electron app for ${platform}`)) {
    process.exit(1);
  }

  // Check output
  log('\n📂 Checking build output...', colors.bright);
  const distPath = path.join(__dirname, '..', 'dist');
  
  if (fs.existsSync(distPath)) {
    const files = fs.readdirSync(distPath);
    log(`\nBuild artifacts in dist/:`, colors.green);
    files.forEach(file => {
      const stats = fs.statSync(path.join(distPath, file));
      const size = (stats.size / 1024 / 1024).toFixed(2);
      log(`  📦 ${file} (${size} MB)`);
    });

    // Check for expected output files
    let expectedFiles = [];
    switch (platform) {
      case 'win32':
        expectedFiles = ['.exe'];
        break;
      case 'darwin':
        expectedFiles = ['.dmg', '.zip'];
        break;
      case 'linux':
        expectedFiles = ['.AppImage', '.deb'];
        break;
    }

    const hasExpectedFiles = expectedFiles.some(ext => 
      files.some(file => file.endsWith(ext))
    );

    if (hasExpectedFiles) {
      log(`\n✅ Packaging test completed successfully!`, colors.green);
      log(`\n📌 Next steps:`, colors.bright);
      log(`1. Test the installer/package in dist/`);
      log(`2. Create a git tag: git tag v1.0.0`);
      log(`3. Push the tag: git push origin v1.0.0`);
      log(`4. The GitHub Action will build for all platforms automatically`);
    } else {
      log(`\n⚠️  Expected output files not found!`, colors.yellow);
      log(`Expected extensions: ${expectedFiles.join(', ')}`);
    }
  } else {
    log('❌ No build output found in dist/', colors.red);
    process.exit(1);
  }
}

// Run the tests
runTests().catch(error => {
  log(`\n❌ Unexpected error: ${error.message}`, colors.red);
  process.exit(1);
});