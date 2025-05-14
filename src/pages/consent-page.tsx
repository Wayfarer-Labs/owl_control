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
            <h2 className="text-lg font-bold mb-4">Research Participant Information and Consent Form</h2>
            
            <h3 className="text-md font-bold mt-4">Purpose of Study</h3>
            <p className="mb-3">
              This research aims to understand and improve human-computer interaction in gaming environments. 
              The data collected will help us develop better systems for game control and accessibility.
            </p>
            
            <h3 className="text-md font-bold mt-4">Procedures</h3>
            <p className="mb-3">
              VG Control will collect your input data (keyboard, mouse, controller inputs) while you play games.
              This data will be securely uploaded to our research database. The software also captures video of your 
              gameplay for research purposes only.
            </p>
            
            <h3 className="text-md font-bold mt-4">Risks and Benefits</h3>
            <p className="mb-3">
              There are minimal risks associated with this study. Your personal information will be kept confidential.
              The benefits include contributing to research that will improve gaming experiences for all players.
            </p>
            
            <h3 className="text-md font-bold mt-4">Confidentiality</h3>
            <p className="mb-3">
              All data collected will be stored securely. Your identity will not be linked to the data in any 
              published materials. We follow strict data protection protocols.
            </p>
            
            <h3 className="text-md font-bold mt-4">Participation</h3>
            <p className="mb-3">
              Your participation is voluntary. You can withdraw at any time by uninstalling the software.
              If you choose to withdraw, you can request that your data be deleted by contacting us.
            </p>
            
            <h3 className="text-md font-bold mt-4">Contact Information</h3>
            <p className="mb-3">
              If you have questions or concerns about this research, please contact:
              research@openworldlabs.com
            </p>
            
            <h3 className="text-md font-bold mt-4">Consent</h3>
            <p className="mb-3">
              By agreeing below, you confirm that you have read and understood this information, 
              and that you consent to participate in this research study.
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