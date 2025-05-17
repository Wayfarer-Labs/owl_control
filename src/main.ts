import { app, BrowserWindow, ipcMain, dialog, Tray, Menu, nativeImage, shell } from 'electron';
import * as path from 'path';
import * as fs from 'fs';
import { spawn } from 'child_process';
import { join } from 'path';
import owlIcon from './assets/owl-emoji.svg';

// Keep references
let mainWindow: BrowserWindow | null = null;
let settingsWindow: BrowserWindow | null = null;
let tray: Tray | null = null;
let pythonProcess: any = null;
let isRecording = false;

// Secure store for credentials and preferences
const secureStore = {
  credentials: {} as Record<string, string>,
  preferences: {
    uploadFrequency: 'one',
    showRecordButton: true
  } as Record<string, any>
};

// Path to store config
const configPath = join(app.getPath('userData'), 'config.json');

// Load config if it exists
function loadConfig() {
  try {
    if (fs.existsSync(configPath)) {
      const config = JSON.parse(fs.readFileSync(configPath, 'utf8'));
      if (config.credentials) {
        secureStore.credentials = config.credentials;
      }
      if (config.preferences) {
        secureStore.preferences = config.preferences;
      }
    }
  } catch (error) {
    console.error('Error loading config:', error);
  }
}

// Save config
function saveConfig() {
  try {
    fs.writeFileSync(configPath, JSON.stringify({
      credentials: secureStore.credentials,
      preferences: secureStore.preferences
    }));
  } catch (error) {
    console.error('Error saving config:', error);
  }
}

// Check if authenticated
function isAuthenticated() {
  return (
    secureStore.credentials.apiKey && 
    secureStore.credentials.hasConsented === 'true'
  );
}

// Create the main window
function createMainWindow() {
  if (mainWindow) {
    mainWindow.close();
  }

  mainWindow = new BrowserWindow({
    width: 1200,
    height: 800,
    webPreferences: {
      nodeIntegration: true,
      contextIsolation: false,
      preload: path.join(__dirname, 'preload.js')
    },
    frame: true,
    transparent: false,
    resizable: true,
    fullscreenable: false
  });

  // Load index.html
  mainWindow.loadFile(path.join(__dirname, 'index.html'));

  // Open DevTools in development
  if (process.env.NODE_ENV === 'development') {
    mainWindow.webContents.openDevTools();
  }

  mainWindow.on('closed', () => {
    mainWindow = null;
  });
}

// Create settings window
function createSettingsWindow() {
  if (settingsWindow) {
    settingsWindow.focus();
    return;
  }

  // Create settings window with proper flags
  settingsWindow = new BrowserWindow({
    width: 800,
    height: 630,
    webPreferences: {
      nodeIntegration: true,
      contextIsolation: false,
      preload: path.join(__dirname, 'preload.js')
    },
    parent: mainWindow || undefined,
    modal: mainWindow ? true : false,
    show: false,  // Hide until ready to show
    backgroundColor: '#0c0c0f',  // Dark background color
    resizable: false,  // Prevent resizing
    fullscreenable: false,  // Prevent fullscreen
    minimizable: false,  // Prevent minimize
    maximizable: false,  // Prevent maximize
    frame: true,  // Keep the frame to avoid flickering
    titleBarStyle: 'hidden',  // Hide title bar on macOS but keep frame
    title: 'VG Control - Settings'  // Window title
  });
  
  // Directly load settings page with query parameters - single load
  settingsWindow.loadURL('file://' + path.join(__dirname, 'index.html?page=settings&direct=true#settings'));
  
  // Use the global switch-to-widget handler from setupIpcHandlers
  
  // Set up DOM ready handler to apply CSS immediately
  settingsWindow.webContents.on('dom-ready', () => {
    // Apply simple CSS to prevent white flash during load
    settingsWindow.webContents.insertCSS(`
      html, body { background-color: #0c0c0f !important; }
      #root { background-color: #0c0c0f !important; }
    `);
  });
  
  // Set credentials directly in localStorage after content is fully loaded
  settingsWindow.webContents.once('did-finish-load', () => {
    // Restore the original full CSS with detailed styling
    const css = `
      /* Force dark mode throughout the app */
      html, body, #root, [class*="bg-background"], .bg-background {
        background-color: #0c0c0f !important;
        color: #f8f9fa !important;
      }
      
      /* Fix black box issues - make text containers transparent */
      h1, h2, h3, h4, h5, h6, p, span, label, a, 
      div.text-sm, div.text-muted-foreground, div.flex.items-center {
        background-color: transparent !important;
      }

      /* Tailwind card styles */
      .bg-card, [class*="bg-card"], div[class*="card"] {
        background-color: #13151a !important;
        border-color: #2a2d35 !important;
        color: #f8f9fa !important;
      }

      /* Handle all card variations */
      [class*="rounded-lg"], [class*="border"], [class*="shadow"], [class*="p-"], [class*="bg-popover"] {
        background-color: #13151a !important;
        border-color: #2a2d35 !important;
      }

      /* Button styling - cyan accent color */
      button, [role="button"], [type="button"] {
        background-color: hsl(186, 90%, 61%) !important;
        color: #0c0c0f !important;
        border: none !important;
      }
      
      /* Primary button styling */
      button[class*="primary"], [role="button"][class*="primary"], .btn-primary, .button-primary {
        background-color: hsl(186, 90%, 61%) !important;
        color: #0c0c0f !important;
        border: none !important;
      }
      
      /* Button variants */
      [class*="btn-secondary"], [class*="btn-outline"], [class*="ghost"] {
        background-color: transparent !important;
        border-color: #2a2d35 !important;
        color: #f8f9fa !important;
      }
      
      /* Form inputs */
      input, select, textarea, [type="text"], [type="password"], [type="email"], [class*="input"] {
        background-color: #1f2028 !important;
        border-color: #2a2d35 !important;
        color: #f8f9fa !important;
      }
      
      /* Text colors */
      p, h1, h2, h3, h4, h5, h6, span, label {
        color: #f8f9fa !important;
      }
      
      /* Tailwind specific text classes */
      [class*="text-"], [class*="text-muted"] {
        color: #f8f9fa !important;
      }
      
      /* Secondary text */
      [class*="text-muted"], [class*="text-secondary"] {
        color: #a0aec0 !important;
      }
      
      /* Fix any potential black boxes around titles and text content */
      [class*="card-header"], [class*="card-title"], [class*="card-description"],
      .text-sm, .text-muted-foreground, .col-span-2, .flex.items-center,
      p.text-sm, div.text-sm, .space-y-2, .space-y-4 {
        background-color: transparent !important;
      }
      
      /* Fix black boxes around specific elements */
      .col-span-2 *, div.flex.items-center *, .rounded-lg.border.p-4 * {
        background-color: transparent !important;
      }
      
      /* Fix any specific components */
      .fixed.inset-0.bg-background\\/80.backdrop-blur-sm.z-50 {
        background-color: rgba(12, 12, 15, 0.8) !important;
      }
      
      /* Ensure all UI components are dark */
      .bg-background, .bg-card, [class*="muted"], [class*="popover"] {
        background-color: #13151a !important;
      }
    `;
    
    // Inject CSS first
    settingsWindow.webContents.insertCSS(css);
    
    // First set credentials to ensure auth works
    settingsWindow.webContents.executeJavaScript(`
      // Set credentials directly in localStorage
      localStorage.setItem('apiKey', '${secureStore.credentials.apiKey || ''}');
      localStorage.setItem('hasConsented', 'true');
      document.documentElement.classList.add('dark');
      
      // Force dark theme using body class as well
      document.body.classList.add('dark-theme');
      
      // Set a global variable to tell React to show settings
      window.DIRECT_SETTINGS = true;
      window.SKIP_AUTH = true;
      
      // We're ready to show the window now
      true; // Return value for promise
    `)
    .then(() => {
      // After the page has applied dark theme, we can safely show the window
      if (settingsWindow) {
        settingsWindow.show();
      }
    });
  });

  settingsWindow.on('closed', () => {
    settingsWindow = null;
  });
}

// Create tray icon
function createTray() {
  try {
    console.log('Creating tray icon from owl emoji');

    // "owlIcon" is the path to the packaged asset provided by Webpack
    const icon = nativeImage.createFromPath(owlIcon);

    // Create the tray using the loaded image
    tray = new Tray(icon);

    // On macOS the tray title helps with visibility; use the emoji directly
    if (process.platform === 'darwin') {
      tray.setTitle('🦉');
    }
    
    updateTrayMenu();
    
    // Double-click on tray icon opens settings
    tray.on('double-click', () => {
      if (isAuthenticated()) {
        createSettingsWindow();
      } else {
        createMainWindow();
      }
    });
  } catch (error) {
    console.error('Error creating tray:', error);
  }
}

// Update the tray menu
function updateTrayMenu() {
  if (!tray) return;
  
  const menuTemplate = [];
  
  // Add status item
  menuTemplate.push({
    label: isRecording ? 'Recording...' : 'Not Recording',
    enabled: false
  });
  
  menuTemplate.push({ type: 'separator' });
  
  // Add recording controls if authenticated
  if (isAuthenticated()) {
    if (isRecording) {
      menuTemplate.push({
        label: 'Stop Recording',
        click: stopRecording
      });
    } else {
      menuTemplate.push({
        label: 'Start Recording',
        click: startRecording
      });
    }
    
    menuTemplate.push({ type: 'separator' });
    
    menuTemplate.push({
      label: 'Settings',
      click: () => createSettingsWindow()
    });
  } else {
    menuTemplate.push({
      label: 'Setup',
      click: () => createMainWindow()
    });
  }
  
  menuTemplate.push({ type: 'separator' });
  
  menuTemplate.push({
    label: 'Help',
    click: () => {
      shell.openExternal('https://openworldlabs.ai/contribute');
    }
  });
  
  menuTemplate.push({
    label: 'Quit',
    click: () => {
      app.quit();
    }
  });
  
  // Update tray icon color/label based on recording state
  if (process.platform === 'darwin') {
    tray.setTitle(isRecording ? 'Recording' : '');
  }
  
  const contextMenu = Menu.buildFromTemplate(menuTemplate);
  tray.setContextMenu(contextMenu);
  tray.setToolTip(isRecording ? 'VG Control - Recording' : 'VG Control');
}

// Start recording
function startRecording() {
  if (!isAuthenticated()) return;

  const recordingPath = secureStore.preferences.recordingPath;
  const outputPath = secureStore.preferences.outputPath;

  if (!recordingPath || !outputPath) {
    dialog.showMessageBox({
      type: 'error',
      title: 'Configuration Missing',
      message: 'Please set recording and output paths in settings.'
    });
    return;
  }

  startPythonProcess(recordingPath, outputPath);
  updateTrayMenu();
}

// Stop recording
function stopRecording() {
  stopPythonProcess();
  updateTrayMenu();
}

// Start Python tracking process
function startPythonProcess(recordingPath: string, outputPath: string) {
  try {
    // Stop existing process if running
    stopPythonProcess();

    // Path to Python script
    const scriptPath = path.join(__dirname, '..', 'vg_control', 'main.py');
    
    // Check if the file exists
    if (!fs.existsSync(scriptPath)) {
      console.error(`Python script not found at: ${scriptPath}`);
      return false;
    }

    // Launch Python process
    pythonProcess = spawn('python', [
      scriptPath,
      '--recording-path', recordingPath,
      '--output-path', outputPath,
      '--api-key', secureStore.credentials.apiKey || ''
    ]);

    // Handle output
    pythonProcess.stdout.on('data', (data: Buffer) => {
      console.log(`Python stdout: ${data.toString()}`);
    });

    pythonProcess.stderr.on('data', (data: Buffer) => {
      console.error(`Python stderr: ${data.toString()}`);
    });

    pythonProcess.on('close', (code: number) => {
      console.log(`Python process exited with code ${code}`);
      pythonProcess = null;
      isRecording = false;
      updateTrayMenu();
    });

    isRecording = true;
    return true;
  } catch (error) {
    console.error('Error starting Python process:', error);
    return false;
  }
}

// Stop Python tracking process
function stopPythonProcess() {
  if (pythonProcess) {
    try {
      // Gracefully signal Python process to stop
      pythonProcess.stdin.write('STOP\n');
      
      // Give it a second to clean up
      setTimeout(() => {
        if (pythonProcess) {
          // Force kill if still running
          pythonProcess.kill();
          pythonProcess = null;
        }
      }, 1000);
      
      isRecording = false;
      return true;
    } catch (error) {
      console.error('Error stopping Python process:', error);
      return false;
    }
  }
  return true;
}

// App ready event
app.on('ready', () => {
  // Load config
  loadConfig();
  
  // Set up IPC handlers
  setupIpcHandlers();
  
  // Create the tray
  createTray();
  
  // If not authenticated, show main window for setup
  if (!isAuthenticated()) {
    createMainWindow();
  }
});

// Prevent app from closing when all windows are closed (keep tray icon)
app.on('window-all-closed', () => {
  // Do nothing to keep the app running in the background
});

app.on('activate', () => {
  if (mainWindow === null) {
    if (!isAuthenticated()) {
      createMainWindow();
    }
  }
});

// Quit app completely when exiting
app.on('before-quit', () => {
  stopPythonProcess();
});

// Set up IPC handlers
function setupIpcHandlers() {
  // Widget mode has been removed

  // Open directory dialog
  ipcMain.handle('open-directory-dialog', async () => {
    if (!mainWindow && !settingsWindow) return '';
    
    const parentWindow = settingsWindow || mainWindow;
    const result = await dialog.showOpenDialog(parentWindow!, {
      properties: ['openDirectory']
    });
    
    if (result.canceled || result.filePaths.length === 0) {
      return '';
    }
    
    return result.filePaths[0];
  });

  // Open save dialog
  ipcMain.handle('open-save-dialog', async () => {
    if (!mainWindow && !settingsWindow) return '';
    
    const parentWindow = settingsWindow || mainWindow;
    const result = await dialog.showSaveDialog(parentWindow!, {
      properties: ['createDirectory']
    });
    
    if (result.canceled || !result.filePath) {
      return '';
    }
    
    return result.filePath;
  });

  // Save credentials
  ipcMain.handle('save-credentials', async (_, key: string, value: string) => {
    try {
      secureStore.credentials[key] = value;
      saveConfig();
      
      // Update tray menu if authentication state changed
      if (key === 'apiKey' || key === 'hasConsented') {
        updateTrayMenu();
      }
      
      return { success: true };
    } catch (error) {
      console.error('Error saving credentials:', error);
      return { success: false, error: String(error) };
    }
  });

  // Load credentials
  ipcMain.handle('load-credentials', async () => {
    try {
      return { success: true, data: secureStore.credentials };
    } catch (error) {
      console.error('Error loading credentials:', error);
      return { success: false, data: {}, error: String(error) };
    }
  });

  // Save preferences
  ipcMain.handle('save-preferences', async (_, preferences: any) => {
    try {
      secureStore.preferences = { ...secureStore.preferences, ...preferences };
      saveConfig();
      return { success: true };
    } catch (error) {
      console.error('Error saving preferences:', error);
      return { success: false, error: String(error) };
    }
  });

  // Load preferences
  ipcMain.handle('load-preferences', async () => {
    try {
      return { success: true, data: secureStore.preferences };
    } catch (error) {
      console.error('Error loading preferences:', error);
      return { success: false, data: {}, error: String(error) };
    }
  });

  // Start recording
  ipcMain.handle('start-recording', async (_, recordingPath: string, outputPath: string) => {
    const success = startPythonProcess(recordingPath, outputPath);
    updateTrayMenu();
    return success;
  });

  // Stop recording
  ipcMain.handle('stop-recording', async () => {
    const success = stopPythonProcess();
    updateTrayMenu();
    return success;
  });

  // Close settings window
  ipcMain.handle('close-settings', async () => {
    if (settingsWindow) {
      settingsWindow.close();
    }
    return true;
  });
  
  // Authentication completed
  ipcMain.handle('authentication-completed', async () => {
    updateTrayMenu();
    
    // Close main window if it exists
    if (mainWindow) {
      mainWindow.close();
    }
    
    return true;
  });
}