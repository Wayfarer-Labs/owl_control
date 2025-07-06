import React, { useState, useEffect } from 'react';
// import { Logo } from '@/components/logo';
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Button } from '@/components/ui/button';
// Theme toggle removed - always dark theme
import { AuthService } from '@/services/auth-service';
import { PythonBridge, AppPreferences } from '@/services/python-bridge';
import { Check } from 'lucide-react';

interface SettingsPageProps {
  onClose: () => void;
}

export function SettingsPage({ onClose }: SettingsPageProps) {
  const [statusMessage, setStatusMessage] = useState('');
  const [userInfo, setUserInfo] = useState<any>(null);
  const [startRecordingKey, setStartRecordingKey] = useState('f4');
  const [stopRecordingKey, setStopRecordingKey] = useState('f5');
  const [apiToken, setApiToken] = useState('');
  const [deleteUploadedFiles, setDeleteUploadedFiles] = useState(false);
  const [debugEnabled, setDebugEnabled] = useState(false);
  const [saveDebugLog, setSaveDebugLog] = useState(true);
  const [gstreamerLoggingEnabled, setGstreamerLoggingEnabled] = useState(false);
  const [debugLevel, setDebugLevel] = useState('4,GST_*:2,task:2,base*:2,aggregator:2,structure:2,ringbuffer:2,structure:2,audio*:2');
  const [enableTraceDumps, setEnableTraceDumps] = useState(false);
  
  // Define the button styles directly in the component for reliability
  const buttonStyle = {
    backgroundColor: 'hsl(186, 90%, 61%)',
    color: '#0c0c0f',
    border: 'none',
    borderRadius: '0.5rem',
    padding: '0.5rem 1rem',
    cursor: 'pointer',
    fontWeight: 'medium',
    fontSize: '0.875rem',
    transition: 'all 0.3s ease'
  };
  
  const authService = AuthService.getInstance();
  const pythonBridge = new PythonBridge();
  
  // Load preferences on component mount
  useEffect(() => {
    // Direct loading of credentials from Electron
    const isSettingsDirectNavigation = window.location.search.includes('page=settings');
    
    if (isSettingsDirectNavigation) {
      // When opened directly from system tray, load credentials directly from Electron
      try {
        const { ipcRenderer } = window.require('electron');
        ipcRenderer.invoke('load-credentials').then((result) => {
          if (result.success && result.data) {
            // Update auth service with the credentials
            const authService = AuthService.getInstance();
            if (result.data.apiKey) {
              authService.validateApiKey(result.data.apiKey);
            }
            if (result.data.hasConsented === 'true') {
              authService.setConsent(true);
            }
            
            // Now load user info
            loadUserInfo();
          }
        });
      } catch (error) {
        console.error('Error loading credentials from Electron:', error);
      }
    }
    
    // Load preferences
    const prefs = pythonBridge.loadPreferences();
    if (prefs.startRecordingKey) setStartRecordingKey(prefs.startRecordingKey);
    if (prefs.stopRecordingKey) setStopRecordingKey(prefs.stopRecordingKey);
    if (prefs.apiToken) setApiToken(prefs.apiToken);
    if (prefs.deleteUploadedFiles !== undefined) setDeleteUploadedFiles(prefs.deleteUploadedFiles);
    
    // Load debug preferences
    if (prefs.debugEnabled !== undefined) setDebugEnabled(prefs.debugEnabled);
    if (prefs.saveDebugLog !== undefined) setSaveDebugLog(prefs.saveDebugLog);
    if (prefs.gstreamerLoggingEnabled !== undefined) setGstreamerLoggingEnabled(prefs.gstreamerLoggingEnabled);
    if (prefs.gstreamerTracingEnabled !== undefined) setEnableTraceDumps(prefs.gstreamerTracingEnabled);
    if (prefs.debugLevel) setDebugLevel(prefs.debugLevel);
    
    // Handle legacy debugLevel loading for backwards compatibility
    if (prefs.debugLevel && prefs.debugEnabled === undefined) {
      if (prefs.debugLevel && prefs.debugLevel !== 'none' && prefs.debugLevel !== '') {
        setDebugEnabled(true);
        setGstreamerLoggingEnabled(true);
      }
    }
    
    // Always load user info after preferences
    loadUserInfo();
  }, []);
  
  const loadUserInfo = async () => {
    try {
      // Check if we're in direct settings mode
      if ((window as any).SKIP_AUTH === true || (window as any).DIRECT_SETTINGS === true) {
        // Try to load credentials directly from Electron
        try {
          const { ipcRenderer } = window.require('electron');
          const result = await ipcRenderer.invoke('load-credentials');
          if (result.success && result.data) {
            // Set credentials in auth service
            if (result.data.apiKey) {
              await authService.validateApiKey(result.data.apiKey);
            }
            if (result.data.hasConsented === 'true') {
              await authService.setConsent(true);
            }
          }
        } catch (error) {
          console.error('Error loading credentials from Electron:', error);
        }
      }

      // Now get user info
      const info = await authService.getUserInfo();
      setUserInfo(info);
    } catch (error) {
      console.error('Error loading user info:', error);
    }
  };
  
  const savePreferences = () => {
    pythonBridge.savePreferences({
      startRecordingKey,
      stopRecordingKey,
      apiToken,
      deleteUploadedFiles,
      debugEnabled,
      saveDebugLog,
      gstreamerLoggingEnabled,
      gstreamerTracingEnabled: enableTraceDumps,
      debugLevel: (debugEnabled && gstreamerLoggingEnabled) ? debugLevel : undefined
    });
    
    // After saving preferences, automatically start the Python bridges
    pythonBridge.startRecordingBridge();
    pythonBridge.startUploadBridge();
  };
  
  
  const handleLogout = async () => {
    await authService.logout();
    
    // Check if this is a direct settings window
    const isSettingsDirectNavigation = window.location.search.includes('page=settings');
    
    if (isSettingsDirectNavigation) {
      // Close the window via IPC if it's a direct settings window
      try {
        const { ipcRenderer } = window.require('electron');
        await ipcRenderer.invoke('close-settings');
      } catch (error) {
        console.error('Error closing settings window:', error);
      }
    } else {
      // Otherwise reload the page to show login
      window.location.reload();
    }
  };

  const handleSaveAndExit = () => {
    // Save preferences
    savePreferences();
    
    // Close window right after saving
    const isSettingsDirectNavigation = window.location.search.includes('page=settings');
    if (isSettingsDirectNavigation) {
      try {
        const { ipcRenderer } = window.require('electron');
        ipcRenderer.invoke('close-settings');
      } catch (error) {
        console.error('Error closing settings window:', error);
      }
    } else {
      onClose();
    }
  };

  const toggleDeleteUploadedFiles = () => {
    const newValue = !deleteUploadedFiles;
    console.log("Setting deleteUploadedFiles to:", newValue);
    setDeleteUploadedFiles(newValue);
  };
  
  return (
    <div className="fixed inset-0 bg-[#0c0c0f] z-50 flex flex-col select-none overflow-hidden">
      {/* Draggable header area */}
      <div className="h-8" style={{ WebkitAppRegion: 'drag', '-webkit-app-region': 'drag' } as any}></div>
      
      <div className="flex flex-col p-6 h-full overflow-hidden">
        {/* Header */}
        <div className="mb-6">
        <h1 className="text-2xl font-bold text-white select-none">Settings</h1>
        <p className="text-gray-400 select-none">Configure your recording preferences</p>
      </div>
      
      {/* Main Content */}
      <div className="flex-1 space-y-6 overflow-y-auto overflow-x-hidden pr-2">
        {/* Account Section */}
        {userInfo && (
          <div className="bg-[#13151a] rounded-lg border border-[#2a2d35] p-4">
            <h3 className="mb-2 text-sm font-medium text-white select-none">Account</h3>
            <div className="flex items-center justify-between">
              <p className="text-[#42e2f5] select-none">
                {userInfo.email ? `${userInfo.email}` : 'API Key authenticated'}
              </p>
              <button 
                className="bg-black text-white px-4 py-2 rounded-md font-medium select-none"
                onClick={handleLogout}
              >
                Logout
              </button>
            </div>
          </div>
        )}
        
        {/* API Token Section */}
        <div className="bg-[#13151a] rounded-lg border border-[#2a2d35] p-4">
          <h3 className="mb-4 text-sm font-medium text-white select-none">OWL API Token</h3>
          <div className="space-y-4">
            <div className="grid grid-cols-1 gap-4 items-center">
              <div>
                <Input
                  id="apiToken"
                  value={apiToken}
                  onChange={(e) => setApiToken(e.target.value)}
                  className="bg-[#0c0c0f] border-[#2a2d35] text-white"
                  placeholder="Enter your OWL API token"
                />
              </div>
            </div>
          </div>
        </div>
        
        {/* Upload Settings */}
        <div className="bg-[#13151a] rounded-lg border border-[#2a2d35] p-4">
          <h3 className="mb-4 text-sm font-medium text-white select-none">Upload Settings</h3>
          <div className="space-y-4">
            <div className="grid grid-cols-[1fr,1fr] gap-4 items-center">
              <Label className="text-sm text-white select-none">Delete Files After Upload</Label>
              <div className="flex items-center">
                {/* Custom checkbox implementation */}
                <div 
                  className="relative flex items-center select-none cursor-pointer" 
                  onClick={toggleDeleteUploadedFiles}
                >
                  {/* Custom checkbox */}
                  <div 
                    className={`w-5 h-5 mr-2 border rounded flex items-center justify-center ${
                      deleteUploadedFiles 
                        ? 'bg-[#1a73e8] border-[#1a73e8]' 
                        : 'bg-[#0c0c0f] border-[#2a2d35]'
                    }`}
                  >
                    {deleteUploadedFiles && (
                      <Check className="h-3.5 w-3.5 text-white" />
                    )}
                  </div>
                  
                  {/* Checkbox label */}
                  <span className="text-white select-none">
                    Delete local files after successful upload
                  </span>
                </div>
              </div>
            </div>
          </div>
        </div>
        
        {/* Keyboard Shortcuts */}
        <div className="bg-[#13151a] rounded-lg border border-[#2a2d35] p-4">
          <h3 className="mb-4 text-sm font-medium text-white select-none">Keyboard Shortcuts</h3>
          <div className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="startRecordingKey" className="text-sm text-white select-none">Start Recording</Label>
              <Input
                id="startRecordingKey"
                value={startRecordingKey}
                onChange={(e) => setStartRecordingKey(e.target.value)}
                placeholder="e.g., f4"
                className="bg-[#0c0c0f] border-[#2a2d35] text-white"
                onKeyDown={(e) => {
                  e.preventDefault();
                  const keys = [];
                  if (e.metaKey || e.ctrlKey) keys.push('CommandOrControl');
                  if (e.shiftKey) keys.push('Shift');
                  if (e.altKey) keys.push('Alt');
                  if (e.key && e.key !== 'Control' && e.key !== 'Shift' && e.key !== 'Alt' && e.key !== 'Meta') {
                    keys.push(e.key.toUpperCase());
                  }
                  setStartRecordingKey(keys.join('+'));
                }}
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="stopRecordingKey" className="text-sm text-white select-none">Stop Recording</Label>
              <Input
                id="stopRecordingKey"
                value={stopRecordingKey}
                onChange={(e) => setStopRecordingKey(e.target.value)}
                placeholder="e.g., f5"
                className="bg-[#0c0c0f] border-[#2a2d35] text-white"
                onKeyDown={(e) => {
                  e.preventDefault();
                  const keys = [];
                  if (e.metaKey || e.ctrlKey) keys.push('CommandOrControl');
                  if (e.shiftKey) keys.push('Shift');
                  if (e.altKey) keys.push('Alt');
                  if (e.key && e.key !== 'Control' && e.key !== 'Shift' && e.key !== 'Alt' && e.key !== 'Meta') {
                    keys.push(e.key.toUpperCase());
                  }
                  setStopRecordingKey(keys.join('+'));
                }}
              />
              <p className="text-xs text-gray-400">Enter the key for starting/stopping recording (defaults: f4/f5). Note: Only simple keys like F1-F12, or letters are supported by the Python hotkey system.</p>
            </div>
          </div>
        </div>
        
        {/* Debug Settings */}
        <div className="bg-[#13151a] rounded-lg border border-[#2a2d35] p-4">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-sm font-medium text-white select-none">Debug Settings</h3>
            <div className="flex items-center">
              <div 
                className="relative flex items-center select-none cursor-pointer" 
                onClick={() => setDebugEnabled(!debugEnabled)}
              >
                <div 
                  className={`w-5 h-5 mr-2 border rounded flex items-center justify-center ${
                    debugEnabled 
                      ? 'bg-[#1a73e8] border-[#1a73e8]' 
                      : 'bg-[#0c0c0f] border-[#2a2d35]'
                  }`}
                >
                  {debugEnabled && (
                    <Check className="h-3.5 w-3.5 text-white" />
                  )}
                </div>
                <span className="text-sm text-white select-none">
                  Enable debug
                </span>
              </div>
            </div>
          </div>
          <div className="space-y-4">
            
            {/* Debug sub-options - only shown when debug is enabled */}
            {debugEnabled && (
              <div className="space-y-4 ml-4 pl-4 border-l border-[#2a2d35]">
                {/* Save debug log option */}
                <div className="flex items-center justify-between">
                  <Label className="text-sm text-white select-none">Save debug log with each recording</Label>
                  <div className="flex items-center">
                    <div 
                      className="relative flex items-center select-none cursor-pointer" 
                      onClick={() => setSaveDebugLog(!saveDebugLog)}
                    >
                      <div 
                        className={`w-5 h-5 mr-2 border rounded flex items-center justify-center ${
                          saveDebugLog 
                            ? 'bg-[#1a73e8] border-[#1a73e8]' 
                            : 'bg-[#0c0c0f] border-[#2a2d35]'
                        }`}
                      >
                        {saveDebugLog && (
                          <Check className="h-3.5 w-3.5 text-white" />
                        )}
                      </div>
                      <span className="text-sm text-white select-none">
                        Save debug log
                      </span>
                    </div>
                  </div>
                </div>
                
                {/* GStreamer debugging section */}
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <Label className="text-sm text-white select-none">GStreamer debugging</Label>
                    <div className="flex items-center">
                      <div 
                        className="relative flex items-center select-none cursor-pointer" 
                        onClick={() => setGstreamerLoggingEnabled(!gstreamerLoggingEnabled)}
                      >
                        <div 
                          className={`w-5 h-5 mr-2 border rounded flex items-center justify-center ${
                            gstreamerLoggingEnabled 
                              ? 'bg-[#1a73e8] border-[#1a73e8]' 
                              : 'bg-[#0c0c0f] border-[#2a2d35]'
                          }`}
                        >
                          {gstreamerLoggingEnabled && (
                            <Check className="h-3.5 w-3.5 text-white" />
                          )}
                        </div>
                        <span className="text-sm text-white select-none">
                          Enable GStreamer debugging
                        </span>
                      </div>
                    </div>
                  </div>
                  
                  {/* GStreamer sub-options - only shown when GStreamer debugging is enabled */}
                  {gstreamerLoggingEnabled && (
                    <div className="space-y-4 ml-4 pl-4 border-l border-[#2a2d35]">
                      {/* Debug level input */}
                      <div className="space-y-2">
                        <div className="flex items-center gap-2">
                          <Label htmlFor="debugLevel" className="text-sm text-white select-none">GST_DEBUG=</Label>
                          <Input
                            id="debugLevel"
                            value={debugLevel}
                            onChange={(e) => setDebugLevel(e.target.value)}
                            placeholder="4,GST_*:2,task:2,base*:2,aggregator:2,structure:2,ringbuffer:2,structure:2,audio*:2"
                            className="bg-[#0c0c0f] border-[#2a2d35] text-white flex-1"
                          />
                        </div>
                        <p className="text-xs text-gray-400">
                          GStreamer debug format string. Examples: "*:3" (all categories at level 3), 
                          "audiotestsrc:6,*:2" (audiotestsrc at level 6, others at level 2), 
                          "audio*:5" (all audio categories at level 5). 
                          Levels: 0=None, 1=Error, 2=Warning, 3=Fixme, 4=Info, 5=Debug, 6=Log, 7=Trace, 9=Memdump.
                        </p>
                      </div>
                      
                      {/* GStreamer trace dumps (disabled) */}
                      <div className="flex items-center justify-between">
                        <Label className="text-sm text-white select-none">GStreamer trace dumps</Label>
                        <div className="flex items-center">
                          <div className="relative flex items-center select-none opacity-50 cursor-not-allowed">
                            <div className="w-5 h-5 mr-2 border rounded flex items-center justify-center bg-[#0c0c0f] border-[#2a2d35]">
                              {/* Always unchecked and disabled - no check mark */}
                            </div>
                            <span className="text-sm text-white select-none">
                              Enable trace dumps
                            </span>
                          </div>
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
      
      {/* Footer */}
      <div className="flex items-center justify-between mt-6 pt-4 border-t border-[#2a2d35]">
        <div className="text-gray-500 text-sm select-none">
          Open World Labs © {new Date().getFullYear()}
        </div>
        
        <button
          className="bg-[#42e2f5] text-black px-6 py-2 rounded-md font-medium select-none"
          onClick={handleSaveAndExit}
        >
          Save Settings
        </button>
      </div>
      </div>
    </div>
  );
}
