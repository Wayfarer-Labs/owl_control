import React, { useState, useRef, useEffect } from 'react';
// import { Logo } from '@/components/logo';
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
// Theme toggle removed - always dark theme
import { AuthService } from '@/services/auth-service';

interface ConsentPageProps {
  apiKey: string;
  onConsent: () => void;
  onCancel: () => void;
}

export function ConsentPage({ apiKey, onConsent, onCancel }: ConsentPageProps) {
  const [hasReadConsent, setHasReadConsent] = useState(false);
  const consentRef = useRef<HTMLDivElement>(null);
  
  // Function to check if user has scrolled to bottom of consent
  const checkScrollPosition = () => {
    if (consentRef.current) {
      const { scrollTop, scrollHeight, clientHeight } = consentRef.current;
      // Make the threshold larger to ensure it's easier to trigger
      if (scrollTop + clientHeight >= scrollHeight - 50) {
        setHasReadConsent(true);
      }
    }
  };
  
  // Also check scroll on mount and add click handler as fallback
  useEffect(() => {
    // Check initial scroll position
    checkScrollPosition();
    
    // Add click handler to content as fallback to enable button
    const handleContentClick = () => {
      setTimeout(() => {
        setHasReadConsent(true);
      }, 2000); // Enable after 2 seconds of user interaction
    };
    
    const currentRef = consentRef.current;
    if (currentRef) {
      currentRef.addEventListener('click', handleContentClick);
    }
    
    return () => {
      if (currentRef) {
        currentRef.removeEventListener('click', handleContentClick);
      }
    };
  }, []);
  
  const handleConsent = async () => {
    try {
      // Store API key and mark as consented
      const authService = AuthService.getInstance();
      await authService.validateApiKey(apiKey);
      await authService.setConsent(true);
      
      // Notify Electron to update the tray menu and close the window
      try {
        const { ipcRenderer } = window.require('electron');
        await ipcRenderer.invoke('authentication-completed');
      } catch (error) {
        console.error('Error notifying Electron about authentication completion:', error);
      }
      
      onConsent();
    } catch (error) {
      console.error("Error saving consent:", error);
    }
  };
  
  const handleCancel = async () => {
    try {
      // Ensure user is not marked as consented
      const authService = AuthService.getInstance();
      await authService.setConsent(false);
      onCancel();
    } catch (error) {
      console.error("Error canceling consent:", error);
      onCancel();
    }
  };
  
  return (
    <div className="flex min-h-screen flex-col items-center justify-center p-4">
      {/* Theme toggle removed */}
      
      <div className="flex items-center justify-center mb-8">
        {/* Logo disabled */}
        <h1 className="text-3xl font-bold">VG Control</h1>
      </div>
      
      <Card className="w-full max-w-3xl">
        <CardHeader>
          <CardTitle className="text-xl font-bold">Informed Consent</CardTitle>
          <CardDescription>
            Please read the following information carefully
          </CardDescription>
        </CardHeader>
        
        <CardContent>
          <div 
            ref={consentRef} 
            className="h-80 overflow-y-auto border p-4 rounded mb-6"
            onScroll={checkScrollPosition}
          >
            <h2 className="text-lg font-bold mb-4">Informed Consent</h2>
            <h3 className="text-md font-bold">Research Study: OpenWorld Labs Interactive Game Data Pilot</h3>
            <p className="mb-3">
              Principal Investigators: Louis Castricato, Shahbuland Matiana<br />
              Contact: louis@openworldlabs.ai, shab@openworldlabs.ai<br />
              Affiliated Institution: Brown University
            </p>
            
            <h3 className="text-md font-bold mt-4">Purpose of Study</h3>
            <p className="mb-3">
              You are invited to participate in a research study aiming to collect combined game and
              control data for the purposes of training world models and subsequently AI agents. The
              curated dataset will be open sourced and made publicly available for research purposes. This
              software will record your game and inputs to potentially contribute to this dataset. There is no
              minimum required
            </p>
            
            <h3 className="text-md font-bold mt-4">Procedures</h3>
            <p className="mb-3">
              If you agree to participate our software will:<br />
              - Record your screen during gameplay sessions<br />
              - Log keyboard and mouse inputs<br />
              - Store this data for research purposes
            </p>
            
            <h3 className="text-md font-bold mt-4">Data Collection and Privacy</h3>
            <p className="mb-3">
              - In order to prevent any identifying information such as gamertags or text chat from leaking
              into the dataset, we are limiting data collection to a fixed set (whitelist) of single player
              games.<br />
              - The recording software is limited to only capturing full screen applications.<br />
              - The recording software will stop recording if no activity is detected for a sufficient period of
              time.<br />
              - Further processing and cleaning will be done before any open source release. During this
              process, the data will be stored securely and anonymized.<br />
              - Upon full open source release, there will be no identifying information in the dataset.<br />
              - The software cannot record microphone audio<br />
              - The software records all desktop audio<br />
              - When idle, this software will update stored data and (optionally) delete any local videos to
              preserve storage space on your device.<br />
              - The software avoids recording overlays.
            </p>
            
            <h3 className="text-md font-bold mt-4">Potential Risks</h3>
            <p className="mb-3">
              - The software records all desktop audio while you are playing a game. This will also record
              any background audio sources that are active while you are playing.
            </p>
            
            <h3 className="text-md font-bold mt-4">Voluntary Participation</h3>
            <p className="mb-3">
              Your participation is entirely voluntary. You may:<br />
              - Choose not to participate<br />
              - Stop recording at any time<br />
              - Request deletion of your recorded data<br />
              - Withdraw from the study without penalty
            </p>
            
            <h3 className="text-md font-bold mt-4">Compensation</h3>
            <p className="mb-3">
              - There is no compensation for this study
            </p>
            
            <h3 className="text-md font-bold mt-4">Questions or Concerns</h3>
            <p className="mb-3">
              For questions about this research, contact shab@openworldlabs.ai
            </p>
            
            <h3 className="text-md font-bold mt-4">Consent</h3>
            <p className="mb-3">
              By clicking "Accept" below you confirm that:<br />
              - You have read and understood the above information<br />
              - You are 18 years or older<br />
              - You voluntarily agree to participate<br />
              - You understand you can withdraw at any time
            </p>
            
            <div className="h-40 mb-3">
              <h3 className="text-md font-bold mt-4" id="end-of-document">End of Document</h3>
              <p className="mb-3">
                Click "I Agree to the Informed Consent" below if you wish to participate in this research.
              </p>
            </div>
          </div>
          
          <div className="flex justify-between">
            <Button 
              variant="outline"
              onClick={handleCancel}
            >
              I Do Not Consent
            </Button>
            
            <Button 
              disabled={!hasReadConsent}
              onClick={handleConsent}
            >
              I Agree to the Informed Consent
            </Button>
          </div>
        </CardContent>
        
        {!hasReadConsent && (
          <CardFooter className="flex justify-center">
            <p className="text-sm text-amber-500">
              Please scroll to the bottom of the document to enable the consent button
            </p>
          </CardFooter>
        )}
      </Card>
    </div>
  );
}