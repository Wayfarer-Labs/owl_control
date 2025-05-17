import { ElectronService } from './electron-service';

/**
 * Interface for app preferences
 */
export interface AppPreferences {
  recordingPath?: string;
  outputPath?: string;
  uploadFrequency?: 'one' | 'daily' | 'weekly' | 'monthly';
  showRecordButton?: boolean;
  startRecordingKey?: string;
  stopRecordingKey?: string;
}

/**
 * Bridge to Python backend
 */
export class PythonBridge {
  private preferences: AppPreferences = {
    uploadFrequency: 'one',
    showRecordButton: true,
    outputPath: this.getDefaultOutputPath()
  };

  constructor() {
    this.loadPreferences();
  }
  
  private getDefaultOutputPath(): string {
    // Use app's document directory and create a VGControl folder
    try {
      const { app } = require('electron').remote || require('electron');
      const path = require('path');
      const fs = require('fs');
      
      const documentsPath = app.getPath('documents');
      const vgControlPath = path.join(documentsPath, 'VGControl', 'Recordings');
      
      // Ensure the directory exists
      if (!fs.existsSync(vgControlPath)) {
        fs.mkdirSync(vgControlPath, { recursive: true });
      }
      
      return vgControlPath;
    } catch (error) {
      console.error('Error getting default output path:', error);
      // Fallback to documents folder
      if (process.platform === 'darwin') {
        return `${process.env.HOME}/Documents/VGControl/Recordings`;
      } else if (process.platform === 'win32') {
        return `${process.env.USERPROFILE}\\Documents\\VGControl\\Recordings`;
      } else {
        return `${process.env.HOME}/Documents/VGControl/Recordings`;
      }
    }
  }

  /**
   * Load preferences from storage
   */
  public loadPreferences(): AppPreferences {
    try {
      // Try to load from Electron's secure storage
      ElectronService.loadPreferences()
        .then(result => {
          if (result.success && result.data) {
            this.preferences = {
              ...this.preferences,
              ...result.data
            };
            // Ensure outputPath has a default value if not set
            if (!this.preferences.outputPath) {
              this.preferences.outputPath = this.getDefaultOutputPath();
            }
          }
        })
        .catch(error => {
          console.error('Error loading preferences from Electron:', error);
        });
      
      // For immediate return, check localStorage
      try {
        const storedPrefs = localStorage.getItem('appPreferences');
        if (storedPrefs) {
          const parsed = JSON.parse(storedPrefs);
          this.preferences = {
            ...this.preferences,
            ...parsed
          };
        }
      } catch (error) {
        console.error('Error loading preferences from localStorage:', error);
      }
    } catch (error) {
      console.error('Error loading preferences:', error);
    }
    
    // Ensure outputPath has a default value if not set
    if (!this.preferences.outputPath) {
      this.preferences.outputPath = this.getDefaultOutputPath();
    }
    
    return this.preferences;
  }

  /**
   * Save preferences to storage
   */
  public async savePreferences(preferences: AppPreferences): Promise<boolean> {
    try {
      this.preferences = {
        ...this.preferences,
        ...preferences
      };
      
      // Save to Electron's secure storage
      await ElectronService.savePreferences(this.preferences);
      
      // Also save to localStorage as fallback
      try {
        localStorage.setItem('appPreferences', JSON.stringify(this.preferences));
      } catch (error) {
        console.error('Error saving preferences to localStorage:', error);
      }
      
      return true;
    } catch (error) {
      console.error('Error saving preferences:', error);
      return false;
    }
  }

  /**
   * Start recording
   */
  public async startRecording(recordingPath: string, outputPath: string): Promise<boolean> {
    try {
      // Call Electron service to start Python recording process
      return await ElectronService.startRecording(recordingPath, outputPath);
    } catch (error) {
      console.error('Error starting recording:', error);
      return false;
    }
  }

  /**
   * Stop recording
   */
  public async stopRecording(): Promise<boolean> {
    try {
      // Call Electron service to stop Python recording process
      return await ElectronService.stopRecording();
    } catch (error) {
      console.error('Error stopping recording:', error);
      return false;
    }
  }

  /**
   * Upload recorded data
   */
  public async uploadRecordedData(options: {
    videoFilePath: string;
    controlFilePath: string;
    tags?: string[];
  }): Promise<boolean> {
    try {
      // In a real implementation, this would call into the Python script
      // or the Electron main process to handle the upload
      
      // For now, just return success
      return true;
    } catch (error) {
      console.error('Error uploading recorded data:', error);
      return false;
    }
  }
}