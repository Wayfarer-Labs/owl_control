import React, { useState } from 'react';
// import { Logo } from '@/components/logo';
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Button } from '@/components/ui/button';
// Theme toggle removed - always dark theme
import { AuthService } from '@/services/auth-service';

interface ApiKeyPageProps {
  onApiKeySuccess: () => void;
  onShowConsent: (apiKey: string) => void;
}

export function ApiKeyPage({ onApiKeySuccess, onShowConsent }: ApiKeyPageProps) {
  const [apiKey, setApiKey] = useState('');
  const [error, setError] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  
  const authService = AuthService.getInstance();
  
  const handleApiKeyValidation = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    setIsLoading(true);
    
    try {
      const result = await authService.validateApiKey(apiKey);
      
      if (result.success) {
        // Show consent instead of directly proceeding
        onShowConsent(apiKey);
      } else {
        setError(result.message || 'API key validation failed');
      }
    } catch (err) {
      setError('API key validation failed');
    } finally {
      setIsLoading(false);
    }
  };
  
  return (
    <div className="flex min-h-screen flex-col items-center justify-center p-4">
      {/* Theme toggle removed */}
      
      <div className="flex items-center justify-center mb-8">
        {/* Logo disabled */}
        <h1 className="ml-3 text-3xl font-bold">VG Control</h1>
      </div>
      
      <Card className="w-full max-w-md">
        <CardHeader>
          <CardTitle className="text-xl font-bold">Welcome</CardTitle>
          <CardDescription>
            Enter your API key to get started
          </CardDescription>
        </CardHeader>
        
        <CardContent>
          <form onSubmit={handleApiKeyValidation}>
            <div className="space-y-4">
              <div className="space-y-2">
                <Label htmlFor="apiKey">API Key</Label>
                <Input
                  id="apiKey"
                  placeholder="Enter your API key"
                  value={apiKey}
                  onChange={(e) => setApiKey(e.target.value)}
                  required
                />
              </div>
              
              {error && (
                <div className="text-red-500 text-sm">
                  {error}
                </div>
              )}
              
              <Button 
                className="w-full" 
                type="submit"
                disabled={isLoading}
              >
                {isLoading ? 'Validating...' : 'Continue'}
              </Button>
            </div>
          </form>
        </CardContent>
        
        <CardFooter className="flex justify-center border-t pt-6">
          <p className="text-sm text-muted-foreground">
            Open World Labs &copy; {new Date().getFullYear()}
          </p>
        </CardFooter>
      </Card>
    </div>
  );
}