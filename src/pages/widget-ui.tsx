import React, { useState, useEffect } from 'react';
import { Settings, Play, Square } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { PythonBridge } from '@/services/python-bridge';

interface WidgetUIProps {
  onOpenSettings: () => void;
}

export function WidgetUI({ onOpenSettings }: WidgetUIProps) {
  const [isRecording, setIsRecording] = useState(false);
  const [statusMessage, setStatusMessage] = useState('Ready');
  const [showRecordButton, setShowRecordButton] = useState(true);
  
  const pythonBridge = new PythonBridge();
  
  // Load preferences on component mount
  useEffect(() => {
    const prefs = pythonBridge.loadPreferences();
    if (prefs.showRecordButton !== undefined) {
      setShowRecordButton(prefs.showRecordButton);
    }
  }, []);
  
  const handleStartRecording = async () => {
    const prefs = pythonBridge.loadPreferences();
    if (!prefs.recordingPath || !prefs.outputPath) {
      setStatusMessage('Please set paths in settings');
      return;
    }
    
    setStatusMessage('Starting...');
    const success = await pythonBridge.startRecording(prefs.recordingPath, prefs.outputPath);
    
    if (success) {
      setIsRecording(true);
      setStatusMessage('Recording');
    } else {
      setStatusMessage('Failed to start');
    }
  };
  
  const handleStopRecording = async () => {
    setStatusMessage('Stopping...');
    const success = await pythonBridge.stopRecording();
    
    if (success) {
      setIsRecording(false);
      setStatusMessage('Saved');
      setTimeout(() => setStatusMessage('Ready'), 3000);
    } else {
      setStatusMessage('Failed to stop');
    }
  };
  
  return (
    <div className="fixed top-4 right-4 p-3 bg-background/80 backdrop-blur-md border border-border rounded-lg shadow-lg transition-all duration-300 hover:bg-background">
      <div className="flex items-center space-x-3">
        <div className="flex flex-col">
          <div className="flex items-center space-x-2">
            <div className={`w-3 h-3 rounded-full ${isRecording ? 'bg-red-500 animate-pulse' : 'bg-green-500'}`}></div>
            <span className="text-sm font-medium">{statusMessage}</span>
          </div>
        </div>
        
        <div className="flex items-center space-x-2">
          {showRecordButton && (
            isRecording ? (
              <Button 
                variant="ghost" 
                size="icon"
                className="h-8 w-8 rounded-full bg-red-500/10 text-red-500 hover:bg-red-500/20 hover:text-red-600"
                onClick={handleStopRecording}
              >
                <Square className="h-4 w-4" />
              </Button>
            ) : (
              <Button 
                variant="ghost" 
                size="icon"
                className="h-8 w-8 rounded-full bg-green-500/10 text-green-500 hover:bg-green-500/20 hover:text-green-600"
                onClick={handleStartRecording}
              >
                <Play className="h-4 w-4" />
              </Button>
            )
          )}
          
          <Button 
            variant="ghost" 
            size="icon"
            className="h-8 w-8 rounded-full"
            onClick={onOpenSettings}
          >
            <Settings className="h-4 w-4" />
          </Button>
        </div>
      </div>
    </div>
  );
}