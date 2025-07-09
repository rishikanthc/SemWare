#!/usr/bin/env python3
"""
Script to run the SemWare server
"""

import uvicorn
from semware.main import app

if __name__ == "__main__":
    print("🚀 Starting SemWare API Server...")
    print("📖 API Documentation will be available at: http://localhost:8000/docs")
    print("🔍 Health check available at: http://localhost:8000/health")
    print("=" * 50)
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    ) 