import { app, BrowserWindow, ipcMain, dialog, Tray, Menu, nativeImage, shell } from 'electron';
import * as path from 'path';
import * as fs from 'fs';
import { spawn } from 'child_process';
import { join } from 'path';

// Keep references
let mainWindow: BrowserWindow | null = null;
let settingsWindow: BrowserWindow | null = null;
let tray: Tray | null = null;
let pythonProcess: any = null;
let isRecording = false;

const GAME_LIST = [
  'DOOMEternalx64vk',
  'DOOMx64',
  'FactoryGame',
  'Titanfall2',
  'SkyrimSE',
  'TESV',
  'Crysis3', 'Crysis3_x64', 'Crysis2', 'Crysis2_x64',
  'OblivionRemastered',
  'MCC-Win64-Shipping',
  'farcry3', 'fc3', 'farcry4', 'farcry5', 'fc3_blooddragon',
  'Cyberpunk2077',
  'Frog_Detective', 'Frog_Detective_2', 'Frog_Detective_3', 'The_Haunted_Island',
]

// Secure store for credentials and preferences
const secureStore = {
  credentials: {} as Record<string, string>,
  preferences: {
    startRecordingKey: 'f4',
    stopRecordingKey: 'f5',
    apiToken: '',
    deleteUploadedFiles: false
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
        secureStore.preferences = { ...config.preferences };
        // Ensure hotkeys have default values if not set
        if (!secureStore.preferences.startRecordingKey) {
          secureStore.preferences.startRecordingKey = 'f4';
        }
        if (!secureStore.preferences.stopRecordingKey) {
          secureStore.preferences.stopRecordingKey = 'f5';
        }
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

// Note: Hotkeys are handled by the Python recording bridge

// Create the main window
function createMainWindow() {
  if (mainWindow) {
    mainWindow.close();
  }

  mainWindow = new BrowserWindow({
    width: 440,
    height: 380,
    webPreferences: {
      nodeIntegration: true,
      contextIsolation: false,
      preload: path.join(__dirname, 'preload.js')
    },
    frame: true,
    transparent: false,
    resizable: false,
    fullscreenable: false,
    minimizable: true,
    maximizable: false,
    backgroundColor: '#0c0c0f',
    center: true,
    titleBarStyle: process.platform === 'darwin' ? 'hiddenInset' : 'default'
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
    modal: false,  // Allow independent movement
    show: false,  // Hide until ready to show
    backgroundColor: '#0c0c0f',  // Dark background color
    resizable: false,  // Prevent resizing
    fullscreenable: false,  // Prevent fullscreen
    minimizable: true,  // Allow minimize
    maximizable: false,  // Prevent maximize
    frame: true,  // Keep the frame for window controls
    titleBarStyle: process.platform === 'darwin' ? 'hiddenInset' : 'default',  // macOS style
    title: 'OWL Control Settings'  // Window title
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
    console.log('Creating tray icon - cyan with dark "OWL" text');

    // Use a simple 16x16 image created directly with Electron's nativeImage
    const size = { width: 16, height: 16 };
    const icon = nativeImage.createEmpty();

    // Define colors
    const cyan = { r: 66, g: 226, b: 245, a: 255 };    // #42E2F5 - Cyan background
    const dark = { r: 12, g: 12, b: 15, a: 255 };      // #0C0C0F - Dark text

    // Create buffer for the icon
    const trayIconBuffer = Buffer.alloc(size.width * size.height * 4);

    // First, fill the entire buffer with cyan (background)
    for (let i = 0; i < trayIconBuffer.length; i += 4) {
      trayIconBuffer[i] = cyan.r;     // R
      trayIconBuffer[i + 1] = cyan.g; // G
      trayIconBuffer[i + 2] = cyan.b; // B
      trayIconBuffer[i + 3] = cyan.a; // A
    }

    // Define letters with points (1 = dark pixel, 0 = background)
    const textMap = [
      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], // Row 0
      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], // Row 1
      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], // Row 2
      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], // Row 3
      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], // Row 4
      [0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0], // Row 5: O W L
      [0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0], // Row 6: O W L
      [0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0], // Row 7: O W L
      [0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0], // Row 8: O W L
      [0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0], // Row 9: O W L
      [0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0], // Row 10: O W L
      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], // Row 11
      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], // Row 12
      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], // Row 13
      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], // Row 14
      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], // Row 15
    ];

    // Override the cyan background with text pixels
    for (let y = 0; y < size.height; y++) {
      for (let x = 0; x < size.width; x++) {
        if (textMap[y][x] === 1) {
          // Calculate position in buffer (4 bytes per pixel)
          const pos = (y * size.width + x) * 4;
          // Set the dark pixel for text
          trayIconBuffer[pos] = dark.r;     // R
          trayIconBuffer[pos + 1] = dark.g; // G
          trayIconBuffer[pos + 2] = dark.b; // B
          trayIconBuffer[pos + 3] = dark.a; // A
        }
      }
    }

    // Create the icon from the buffer data
    icon.addRepresentation({
      width: size.width,
      height: size.height,
      buffer: trayIconBuffer,
      scaleFactor: 1.0
    });

    console.log('Created cyan tray icon with OWL text');

    // Create tray with our icon
    tray = new Tray(icon);

    // On macOS, we can use a title to ensure visibility
    if (process.platform === 'darwin') {
      tray.setTitle('VG');
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

  // Remove recording controls from menu as Python bridge handles this independently
  if (isAuthenticated()) {

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
  tray.setToolTip(isRecording ? 'OWL Control - Recording' : 'OWL Control');
}

// Start Python bridges after authentication

// Start Python recording bridge
function startRecordingBridge(startKey: string, stopKey: string, debugLevel?: string, saveDebugLog?: boolean, gstreamerLoggingEnabled?: boolean, gstreamerTracingEnabled?: boolean) {
  try {
    // Stop existing process if running
    if (pythonProcess) {
      pythonProcess.kill();
      pythonProcess = null;
    }

    console.log(`Starting recording bridge`);

    const gstreamer_bin = `${process.env["GSTREAMER_1_0_ROOT_MSVC_X86_64"]}\\bin`;

    pythonProcess = spawn(recorderCommand(), [
      '--recording-location', "./data_dump/games/",
      '--start-key', startKey,
      '--stop-key', stopKey,
      ...GAME_LIST.flatMap(game => ['--games', game]),
      ...(debugLevel ? ['--debug-level', debugLevel] : []),
      ...(saveDebugLog ? ['--save-debug-log'] : []),
      ...(gstreamerLoggingEnabled ? ['--gstreamer-logging-enabled'] : []),
      ...(gstreamerTracingEnabled ? ['--gstreamer-tracing-enabled'] : []),
    ], {
      cwd: rootDir(),
      env: {
        ...process.env,
        PATH: `${gstreamer_bin};${process.env["PATH"]}`,
      },
    });

    // Handle output
    pythonProcess.stdout.on('data', (data: Buffer) => {
      console.log(`Recording bridge stdout: ${data.toString()}`);
    });

    pythonProcess.stderr.on('data', (data: Buffer) => {
      console.error(`Recording bridge stderr: ${data.toString()}`);
    });

    pythonProcess.on('close', (code: number) => {
      console.log(`Recording bridge process exited with code ${code}`);
      pythonProcess = null;
    });

    return true;
  } catch (error) {
    console.error('Error starting recording bridge:', error);
    return false;
  }
}

function recorderCommand() {
  if (process.env.NODE_ENV === 'development') {
    return String.raw`target\x86_64-pc-windows-msvc\debug\owl-recorder`;
  } else {
    return "owl-recorder";
  }
}

// Start Python upload bridge
function startUploadBridge(apiToken: string, deleteUploadedFiles: boolean) {
  try {
    console.log(`Starting upload bridge module from vg_control package`);

    const uploadProcess = spawn('uv', [
      'run',
      '-m',
      'vg_control.upload_bridge',
      '--api-token', apiToken,
      ...(deleteUploadedFiles ? ['--delete-uploaded-files'] : [])
    ], {
      cwd: rootDir(),
    });

    // Handle output
    uploadProcess.stdout.on('data', (data: Buffer) => {
      console.log(`Upload bridge stdout: ${data.toString()}`);
    });

    uploadProcess.stderr.on('data', (data: Buffer) => {
      console.error(`Upload bridge stderr: ${data.toString()}`);
    });

    uploadProcess.on('close', (code: number) => {
      console.log(`Upload bridge process exited with code ${code}`);
    });

    return true;
  } catch (error) {
    console.error('Error starting upload bridge:', error);
    return false;
  }
}

function rootDir() {
  if (process.env.NODE_ENV === 'development') {
    return ".";
  } else {
    return path.join(__dirname, '..', '..');
  }
}

// App ready event
app.on('ready', () => {
  // Load config
  loadConfig();

  // Set up IPC handlers
  setupIpcHandlers();

  // Create the tray
  createTray();

  // Start the Python bridges if authenticated
  if (isAuthenticated()) {
    const startKey = secureStore.preferences.startRecordingKey || 'f4';
    const stopKey = secureStore.preferences.stopRecordingKey || 'f5';
    const debugLevel = secureStore.preferences.debugLevel || false;
    const saveDebugLog = secureStore.preferences.saveDebugLog || false;
    const gstreamerLoggingEnabled = secureStore.preferences.gstreamerLoggingEnabled || false;
    const gstreamerTracingEnabled = secureStore.preferences.gstreamerTracingEnabled || false;
    const apiToken = secureStore.credentials.apiKey || '';
    const deleteUploadedFiles = secureStore.preferences.deleteUploadedFiles || false;

    // Start the recording bridge
    startRecordingBridge(startKey, stopKey, debugLevel, saveDebugLog, gstreamerLoggingEnabled, gstreamerTracingEnabled);

    // Start the upload bridge
    startUploadBridge(apiToken, deleteUploadedFiles);
  }

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
  // Kill any running Python processes
  if (pythonProcess) {
    pythonProcess.kill();
    pythonProcess = null;
  }

  // Additional cleanup if needed
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

      // Restart the Python bridges with new preferences if authenticated
      if (isAuthenticated()) {
        const startKey = secureStore.preferences.startRecordingKey || 'f4';
        const stopKey = secureStore.preferences.stopRecordingKey || 'f5';
        const apiToken = secureStore.credentials.apiKey || '';
        const deleteUploadedFiles = secureStore.preferences.deleteUploadedFiles || false;
        const debugLevel = secureStore.preferences.debugLevel;
        const saveDebugLog = secureStore.preferences.saveDebugLog || false;
        const gstreamerLoggingEnabled = secureStore.preferences.gstreamerLoggingEnabled || false;
        const gstreamerTracingEnabled = secureStore.preferences.gstreamerTracingEnabled || false;

        // Restart the recording bridge
        startRecordingBridge(startKey, stopKey, debugLevel, saveDebugLog, gstreamerLoggingEnabled, gstreamerTracingEnabled);

        // Restart the upload bridge
        startUploadBridge(apiToken, deleteUploadedFiles);
      }

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

  // Start recording bridge
  ipcMain.handle('start-recording-bridge', async (_, startKey: string, stopKey: string, debugLevel?: string, saveDebugLog?: boolean, gstreamerLoggingEnabled?: boolean, gstreamerTracingEnabled?: boolean) => {
    return startRecordingBridge(startKey, stopKey, debugLevel, saveDebugLog, gstreamerLoggingEnabled, gstreamerTracingEnabled);
  });

  // Start upload bridge
  ipcMain.handle('start-upload-bridge', async (_, apiToken: string, deleteUploadedFiles: boolean) => {
    return startUploadBridge(apiToken, deleteUploadedFiles);
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

    // Start the Python bridges after authentication
    const startKey = secureStore.preferences.startRecordingKey || 'f4';
    const stopKey = secureStore.preferences.stopRecordingKey || 'f5';
    const apiToken = secureStore.credentials.apiKey || '';
    const deleteUploadedFiles = secureStore.preferences.deleteUploadedFiles || false;
    const debugLevel = secureStore.preferences.debugLevel;
    const saveDebugLog = secureStore.preferences.saveDebugLog || false;
    const gstreamerLoggingEnabled = secureStore.preferences.gstreamerLoggingEnabled || false;
    const gstreamerTracingEnabled = secureStore.preferences.gstreamerTracingEnabled || false;

    // Start the recording bridge
    startRecordingBridge(startKey, stopKey, debugLevel, saveDebugLog, gstreamerLoggingEnabled, gstreamerTracingEnabled);

    // Start the upload bridge
    startUploadBridge(apiToken, deleteUploadedFiles);

    // Close main window if it exists
    if (mainWindow) {
      mainWindow.close();
    }

    return true;
  });

  // Resize window for consent page
  ipcMain.handle('resize-for-consent', async () => {
    if (mainWindow) {
      mainWindow.setSize(760, 700);
      mainWindow.center();
    }
    return true;
  });

  // Resize window for API key page
  ipcMain.handle('resize-for-api-key', async () => {
    if (mainWindow) {
      mainWindow.setSize(440, 380);
      mainWindow.center();
    }
    return true;
  });
}
