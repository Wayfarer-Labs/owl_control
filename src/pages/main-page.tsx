import React, { useState, useEffect } from 'react';
import { Logo } from '@/components/logo';
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Button } from '@/components/ui/button';
import { ThemeToggle } from '@/components/theme-toggle';
import { AuthService } from '@/services/auth-service';
import { PythonBridge } from '@/services/python-bridge';
import { Settings, Video, Play, Square, Home, LogOut } from 'lucide-react';

export function MainPage() {
  const [activeTab, setActiveTab] = useState<'home' | 'settings'>('home');
  const [recordingPath, setRecordingPath] = useState('');
  const [outputPath, setOutputPath] = useState('');
  const [isRecording, setIsRecording] = useState(false);
  const [statusMessage, setStatusMessage] = useState('Ready');
  const [userInfo, setUserInfo] = useState<any>(null);
  
  const authService = AuthService.getInstance();
  const pythonBridge = new PythonBridge();
  
  // Load preferences on component mount
  useEffect(() => {
    const prefs = pythonBridge.loadPreferences();
    if (prefs.recordingPath) setRecordingPath(prefs.recordingPath);
    if (prefs.outputPath) setOutputPath(prefs.outputPath);
    
    loadUserInfo();
  }, []);
  
  const loadUserInfo = async () => {
    const info = await authService.getUserInfo();
    setUserInfo(info);
  };
  
  const savePreferences = () => {
    pythonBridge.savePreferences({
      recordingPath,
      outputPath
    });
    setStatusMessage('Preferences saved');
    setTimeout(() => setStatusMessage('Ready'), 2000);
  };
  
  const handleStartRecording = async () => {
    if (!recordingPath || !outputPath) {
      setStatusMessage('Please set recording and output paths first');
      return;
    }
    
    setStatusMessage('Starting recording...');
    const success = await pythonBridge.startRecording(recordingPath, outputPath);
    
    if (success) {
      setIsRecording(true);
      setStatusMessage('Recording in progress');
    } else {
      setStatusMessage('Failed to start recording');
    }
  };
  
  const handleStopRecording = async () => {
    setStatusMessage('Stopping recording...');
    const success = await pythonBridge.stopRecording();
    
    if (success) {
      setIsRecording(false);
      setStatusMessage('Recording saved');
    } else {
      setStatusMessage('Failed to stop recording');
    }
  };
  
  const handleBrowseRecordingPath = () => {
    // In Electron, we would open a dialog here
    // For now, we'll just simulate it
    const path = prompt('Enter recording path:');
    if (path) setRecordingPath(path);
  };
  
  const handleBrowseOutputPath = () => {
    // In Electron, we would open a dialog here
    // For now, we'll just simulate it
    const path = prompt('Enter output path:');
    if (path) setOutputPath(path);
  };
  
  const handleLogout = async () => {
    await authService.logout();
    window.location.reload();
  };
  
  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="border-b border-border">
        <div className="container flex h-16 items-center justify-between px-4">
          <div className="flex items-center space-x-2">
            <Logo width={32} height={32} />
            <span className="text-lg font-bold">VG Control</span>
          </div>
          
          <div className="flex items-center space-x-4">
            {userInfo && (
              <div className="text-sm text-muted-foreground">
                {userInfo.email 
                  ? `${userInfo.email}` 
                  : 'API Key Access'}
              </div>
            )}
            <ThemeToggle />
          </div>
        </div>
      </header>
      
      {/* Main content */}
      <main className="container py-6 px-4">
        <div className="flex gap-6">
          {/* Sidebar */}
          <aside className="hidden w-[200px] flex-col md:flex">
            <nav className="grid gap-2 px-2">
              <Button
                variant={activeTab === 'home' ? 'default' : 'ghost'}
                className="justify-start"
                onClick={() => setActiveTab('home')}
              >
                <Home className="mr-2 h-4 w-4" />
                Home
              </Button>
              <Button
                variant={activeTab === 'settings' ? 'default' : 'ghost'}
                className="justify-start"
                onClick={() => setActiveTab('settings')}
              >
                <Settings className="mr-2 h-4 w-4" />
                Settings
              </Button>
              <Button
                variant="ghost"
                className="justify-start"
                onClick={handleLogout}
              >
                <LogOut className="mr-2 h-4 w-4" />
                Logout
              </Button>
            </nav>
          </aside>
          
          {/* Content area */}
          <div className="flex-1">
            {activeTab === 'home' ? (
              <div className="grid gap-6">
                <Card className="col-span-2">
                  <CardHeader>
                    <CardTitle>Control Panel</CardTitle>
                    <CardDescription>Manage your recording session</CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-6">
                    <div className="rounded-lg border p-4">
                      <div className="flex items-center justify-between">
                        <div className="flex items-center space-x-4">
                          <div className="rounded-full bg-primary/10 p-2">
                            <Video className="h-5 w-5 text-primary" />
                          </div>
                          <div>
                            <p className="text-sm font-medium">Status</p>
                            <p className="text-sm text-muted-foreground">{statusMessage}</p>
                          </div>
                        </div>
                        <div>
                          {isRecording ? (
                            <Button 
                              variant="destructive" 
                              size="sm"
                              onClick={handleStopRecording}
                              className="flex items-center"
                            >
                              <Square className="mr-2 h-4 w-4" />
                              Stop Recording
                            </Button>
                          ) : (
                            <Button 
                              variant="default" 
                              size="sm"
                              onClick={handleStartRecording}
                              className="flex items-center"
                            >
                              <Play className="mr-2 h-4 w-4" />
                              Start Recording
                            </Button>
                          )}
                        </div>
                      </div>
                    </div>
                    
                    <div className="rounded-lg border p-4">
                      <div className="mb-4">
                        <h3 className="text-sm font-medium">Recording Paths</h3>
                      </div>
                      <div className="grid gap-4">
                        <div className="grid grid-cols-3 items-center gap-4">
                          <Label htmlFor="recordingPath" className="text-sm">OBS Recording Path</Label>
                          <Input
                            id="recordingPath"
                            value={recordingPath}
                            onChange={(e) => setRecordingPath(e.target.value)}
                            className="col-span-2 h-8"
                          />
                        </div>
                        <div className="grid grid-cols-3 items-center gap-4">
                          <Label htmlFor="outputPath" className="text-sm">Output Data Path</Label>
                          <Input
                            id="outputPath"
                            value={outputPath}
                            onChange={(e) => setOutputPath(e.target.value)}
                            className="col-span-2 h-8"
                          />
                        </div>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </div>
            ) : (
              <div className="grid gap-6">
                <Card className="col-span-2">
                  <CardHeader>
                    <CardTitle>Settings</CardTitle>
                    <CardDescription>Configure your recording preferences</CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-6">
                    <div className="rounded-lg border p-4">
                      <h3 className="mb-4 text-sm font-medium">Recording Paths</h3>
                      <div className="grid gap-4">
                        <div className="grid grid-cols-4 items-center gap-4">
                          <Label htmlFor="recordingPathFull" className="text-sm">OBS Recording Path</Label>
                          <Input
                            id="recordingPathFull"
                            value={recordingPath}
                            onChange={(e) => setRecordingPath(e.target.value)}
                            className="col-span-2 h-8"
                          />
                          <Button 
                            variant="outline" 
                            size="sm"
                            onClick={handleBrowseRecordingPath}
                          >
                            Browse
                          </Button>
                        </div>
                        <div className="grid grid-cols-4 items-center gap-4">
                          <Label htmlFor="outputPathFull" className="text-sm">Output Data Path</Label>
                          <Input
                            id="outputPathFull"
                            value={outputPath}
                            onChange={(e) => setOutputPath(e.target.value)}
                            className="col-span-2 h-8"
                          />
                          <Button 
                            variant="outline" 
                            size="sm"
                            onClick={handleBrowseOutputPath}
                          >
                            Browse
                          </Button>
                        </div>
                      </div>
                    </div>
                    
                    <div className="flex justify-end">
                      <Button onClick={savePreferences}>
                        Save Settings
                      </Button>
                    </div>
                  </CardContent>
                </Card>
              </div>
            )}
          </div>
        </div>
      </main>
      
      {/* Footer */}
      <footer className="border-t border-border py-4">
        <div className="container flex items-center justify-between px-4">
          <p className="text-sm text-muted-foreground">
            Open World Labs &copy; {new Date().getFullYear()}
          </p>
          <p className="text-sm text-muted-foreground">
            Version 1.0.0
          </p>
        </div>
      </footer>
    </div>
  );
}