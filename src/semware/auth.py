"""
Authentication module for SemWare API
"""

import os
from fastapi import HTTPException, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional


class APIKeyAuth:
    """API Key authentication handler"""
    
    def __init__(self):
        self.api_key = os.getenv("SEMWARE_API_KEY")
        if not self.api_key:
            raise ValueError("SEMWARE_API_KEY environment variable is required")
        self.security = HTTPBearer()
    
    async def __call__(self, request: Request) -> Optional[str]:
        """Validate API key from request"""
        try:
            # Get the Authorization header
            auth_header = request.headers.get("Authorization")
            if not auth_header:
                raise HTTPException(
                    status_code=401, 
                    detail="API key is required. Please provide Authorization header with Bearer token."
                )
            
            # Check if it's a Bearer token
            if not auth_header.startswith("Bearer "):
                raise HTTPException(
                    status_code=401, 
                    detail="Invalid authorization format. Use 'Bearer <api_key>'"
                )
            
            # Extract the API key
            api_key = auth_header.split(" ")[1]
            
            # Validate the API key
            if api_key != self.api_key:
                raise HTTPException(
                    status_code=401, 
                    detail="Invalid API key"
                )
            
            return api_key
            
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=401, 
                detail=f"Authentication failed: {str(e)}"
            )


# Global instance
api_key_auth = APIKeyAuth() 