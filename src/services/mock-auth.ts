/**
 * Mock authentication service for development
 */
export class MockAuth {
  private static instance: MockAuth;
  private apiKey: string | null = null;

  private constructor() {
    // Try to load stored API key from localStorage if available
    try {
      const storedKey = localStorage.getItem('apiKey');
      if (storedKey) {
        this.apiKey = storedKey;
      }
    } catch (error) {
      console.error('Error loading stored API key:', error);
    }
  }

  /**
   * Get singleton instance
   */
  public static getInstance(): MockAuth {
    if (!MockAuth.instance) {
      MockAuth.instance = new MockAuth();
    }
    return MockAuth.instance;
  }

  /**
   * Check if user is authenticated
   */
  public isAuthenticated(): boolean {
    return !!this.apiKey;
  }

  /**
   * Validate API key (mock implementation)
   */
  public async validateApiKey(apiKey: string): Promise<{ success: boolean; message?: string }> {
    try {
      // Mock validation - any non-empty key is valid for development
      if (!apiKey || apiKey.trim() === '') {
        return { success: false, message: 'API key cannot be empty' };
      }
      
      // Store the API key
      this.apiKey = apiKey;
      try {
        localStorage.setItem('apiKey', apiKey);
      } catch (error) {
        console.error('Error saving API key to localStorage:', error);
      }
      
      return { success: true };
    } catch (error) {
      console.error('API key validation error:', error);
      return { success: false, message: 'API key validation failed' };
    }
  }

  /**
   * Get user information (mock implementation)
   */
  public async getUserInfo(): Promise<any> {
    if (!this.apiKey) {
      return { authenticated: false };
    }
    
    // Return mock user info
    return { 
      authenticated: true, 
      method: 'apiKey',
      apiKey: this.apiKey
    };
  }

  /**
   * Logout user
   */
  public async logout(): Promise<void> {
    this.apiKey = null;
    try {
      localStorage.removeItem('apiKey');
    } catch (error) {
      console.error('Error removing API key from localStorage:', error);
    }
  }
}