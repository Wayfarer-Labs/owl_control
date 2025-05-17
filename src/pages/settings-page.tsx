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
  const [recordingPath, setRecordingPath] = useState('');
  const [outputPath, setOutputPath] = useState('');
  const [uploadFrequency, setUploadFrequency] = useState<AppPreferences['uploadFrequency']>('one');
  const [showRecordButton, setShowRecordButton] = useState(true);
  const [statusMessage, setStatusMessage] = useState('');
  const [userInfo, setUserInfo] = useState<any>(null);
  
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
    if (prefs.recordingPath) setRecordingPath(prefs.recordingPath);
    if (prefs.outputPath) setOutputPath(prefs.outputPath);
    if (prefs.uploadFrequency) setUploadFrequency(prefs.uploadFrequency);
    if (prefs.showRecordButton !== undefined) setShowRecordButton(prefs.showRecordButton);
    
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
      recordingPath, 
      outputPath,
      uploadFrequency,
      showRecordButton
    });
  };
  
  const handleBrowseOutputPath = async () => {
    try {
      const { ElectronService } = require('../services/electron-service');
      const path = await ElectronService.openSaveDialog();
      if (path) setOutputPath(path);
    } catch (error) {
      console.error('Error browsing for output path:', error);
    }
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

  const handleUploadFrequencyChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    setUploadFrequency(e.target.value as AppPreferences['uploadFrequency']);
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

  const toggleRecordButton = () => {
    const newValue = !showRecordButton;
    console.log("Setting showRecordButton to:", newValue);
    setShowRecordButton(newValue);
  };
  
  return (
    <div className="fixed inset-0 bg-[#0c0c0f] z-50 flex flex-col select-none">
      {/* Draggable header area */}
      <div className="h-8" style={{ WebkitAppRegion: 'drag', '-webkit-app-region': 'drag' } as any}></div>
      
      <div className="flex flex-col p-6">
        {/* Header */}
        <div className="mb-6">
        <h1 className="text-2xl font-bold text-white select-none">Settings</h1>
        <p className="text-gray-400 select-none">Configure your recording preferences</p>
      </div>
      
      {/* Main Content */}
      <div className="flex-1 space-y-6 overflow-auto">
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
        
        {/* Output Path Section */}
        <div className="bg-[#13151a] rounded-lg border border-[#2a2d35] p-4">
          <h3 className="mb-4 text-sm font-medium text-white select-none">Output Path</h3>
          <div className="space-y-4">
            <div className="grid grid-cols-[1fr,auto] gap-4 items-center">
              <div>
                <Input
                  id="outputPath"
                  value={outputPath}
                  onChange={(e) => setOutputPath(e.target.value)}
                  className="bg-[#0c0c0f] border-[#2a2d35] text-white"
                />
              </div>
              <button
                className="bg-black text-white px-4 py-2 rounded-md font-medium h-10 self-end select-none"
                onClick={handleBrowseOutputPath}
              >
                Browse
              </button>
            </div>
          </div>
        </div>
        
        {/* Upload & Recording Settings */}
        <div className="bg-[#13151a] rounded-lg border border-[#2a2d35] p-4">
          <h3 className="mb-4 text-sm font-medium text-white select-none">Upload & Recording Settings</h3>
          <div className="space-y-4">
            <div className="grid grid-cols-[1fr,1fr] gap-4 items-center">
              <Label htmlFor="uploadFrequency" className="text-sm text-white select-none">Upload Frequency</Label>
              <select
                id="uploadFrequency"
                value={uploadFrequency}
                onChange={handleUploadFrequencyChange}
                className="bg-[#0c0c0f] border border-[#2a2d35] rounded-md px-3 py-2 text-white"
              >
                <option value="one">One-time</option>
                <option value="daily">Daily</option>
                <option value="weekly">Weekly</option>
                <option value="monthly">Monthly</option>
              </select>
            </div>
            <div className="grid grid-cols-[1fr,1fr] gap-4 items-center">
              <Label className="text-sm text-white select-none">Show Record Button</Label>
              <div className="flex items-center">
                {/* Custom checkbox implementation */}
                <div 
                  className="relative flex items-center select-none cursor-pointer" 
                  onClick={toggleRecordButton}
                >
                  {/* Custom checkbox */}
                  <div 
                    className={`w-5 h-5 mr-2 border rounded flex items-center justify-center ${
                      showRecordButton 
                        ? 'bg-[#1a73e8] border-[#1a73e8]' 
                        : 'bg-[#0c0c0f] border-[#2a2d35]'
                    }`}
                  >
                    {showRecordButton && (
                      <Check className="h-3.5 w-3.5 text-white" />
                    )}
                  </div>
                  
                  {/* Checkbox label */}
                  <span className="text-white select-none">
                    Display record button on main screen
                  </span>
                </div>
              </div>
            </div>
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