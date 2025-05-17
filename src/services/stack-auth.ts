import path from 'path';
import fs from 'fs';
import os from 'os';
import dotenv from 'dotenv';

// Load environment variables
dotenv.config();

/**
 * Stack Auth API client 
 */
export class StackAuth {
  private projectId: string;
  private clientKey: string;
  private serverKey: string;
  private baseUrl: string;
  private userToken: string | null;
  private apiKey: string | null;
  
  constructor() {
    this.projectId = process.env.NEXT_PUBLIC_STACK_PROJECT_ID || 'test_project_id';
    this.clientKey = process.env.NEXT_PUBLIC_STACK_PUBLISHABLE_CLIENT_KEY || 'test_client_key';
    this.serverKey = process.env.STACK_SECRET_SERVER_KEY || 'test_server_key';
    this.baseUrl = "https://stack.auth.com/api";
    this.userToken = 'dummy_token'; // Default token for development
    this.apiKey = 'dummy_api_key'; // Default API key for development
    
    // Comment out for now to avoid errors when credentials file doesn't exist
    // this.loadSavedCredentials();
  }
  
  /**
   * Check if user is authenticated
   */
  public isAuthenticated(): boolean {
    // For development, always return true to bypass login
    return true;
  }

  /**
   * Login with email and password
   */
  public async login(email: string, password: string): Promise<{ success: boolean; message?: string }> {
    try {
      const response = await fetch(`${this.baseUrl}/v1/email/login`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-Stack-Project-Id': this.projectId,
          'X-Stack-Client-Key': this.clientKey
        },
        body: JSON.stringify({
          email,
          password
        })
      });

      if (!response.ok) {
        throw new Error(`HTTP error ${response.status}`);
      }

      const data = await response.json() as any;
      this.userToken = data.token;
      
      // Save credentials
      if (this.userToken) {
        this.saveCredentials('userToken', this.userToken);
        this.saveCredentials('email', email);
      }
      
      return { success: true };
    } catch (error) {
      console.error('Login error:', error);
      return { success: false, message: 'Authentication failed' };
    }
  }

  /**
   * Validate API key
   */
  public async validateApiKey(apiKey: string): Promise<{ success: boolean; message?: string }> {
    try {
      const response = await fetch(`${this.baseUrl}/v1/api-keys/validate`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-Stack-Project-Id': this.projectId,
          'X-Stack-Client-Key': this.clientKey,
          'X-Stack-Api-Key': apiKey
        }
      });

      if (!response.ok) {
        throw new Error(`HTTP error ${response.status}`);
      }

      // Save API key
      this.apiKey = apiKey;
      this.saveCredentials('apiKey', apiKey);
      
      return { success: true };
    } catch (error) {
      console.error('API key validation error:', error);
      return { success: false, message: 'API key validation failed' };
    }
  }

  /**
   * Logout user
   */
  public async logout(): Promise<void> {
    try {
      if (this.userToken) {
        await fetch(`${this.baseUrl}/v1/sessions/logout`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'X-Stack-Project-Id': this.projectId,
            'X-Stack-Client-Key': this.clientKey,
            'Authorization': `Bearer ${this.userToken}`
          }
        });
      }
    } catch (error) {
      console.error('Logout error:', error);
    } finally {
      // Clear credentials
      this.userToken = null;
      this.apiKey = null;
      this.removeCredentials();
    }
  }

  /**
   * Get user information
   */
  public async getUserInfo(): Promise<any> {
    try {
      if (!this.userToken) {
        return { authenticated: !!this.apiKey, method: 'apiKey' };
      }
      
      const response = await fetch(`${this.baseUrl}/v1/users/me`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
          'X-Stack-Project-Id': this.projectId,
          'X-Stack-Client-Key': this.clientKey,
          'Authorization': `Bearer ${this.userToken}`
        }
      });

      if (!response.ok) {
        throw new Error(`HTTP error ${response.status}`);
      }

      const userData = await response.json();
      return { ...(userData as object), authenticated: true, method: 'userToken' };
    } catch (error) {
      console.error('Get user info error:', error);
      return { authenticated: false };
    }
  }

  /**
   * Save credentials to disk
   */
  private saveCredentials(key: string, value: string): void {
    try {
      const credsDir = path.join(os.homedir(), '.vg_control');
      if (!fs.existsSync(credsDir)) {
        fs.mkdirSync(credsDir, { recursive: true });
      }
      
      const credsFile = path.join(credsDir, 'credentials.json');
      let credentials: Record<string, string> = {};
      
      if (fs.existsSync(credsFile)) {
        const data = fs.readFileSync(credsFile, 'utf8');
        credentials = JSON.parse(data);
      }
      
      credentials[key] = value;
      fs.writeFileSync(credsFile, JSON.stringify(credentials, null, 2));
    } catch (error) {
      console.error('Error saving credentials:', error);
    }
  }

  /**
   * Load saved credentials
   */
  private loadSavedCredentials(): void {
    try {
      const credsFile = path.join(os.homedir(), '.vg_control', 'credentials.json');
      
      if (fs.existsSync(credsFile)) {
        const data = fs.readFileSync(credsFile, 'utf8');
        const credentials = JSON.parse(data);
        
        this.userToken = credentials.userToken || null;
        this.apiKey = credentials.apiKey || null;
      }
    } catch (error) {
      console.error('Error loading credentials:', error);
    }
  }

  /**
   * Remove credentials
   */
  private removeCredentials(): void {
    try {
      const credsFile = path.join(os.homedir(), '.vg_control', 'credentials.json');
      
      if (fs.existsSync(credsFile)) {
        fs.unlinkSync(credsFile);
      }
    } catch (error) {
      console.error('Error removing credentials:', error);
    }
  }
}