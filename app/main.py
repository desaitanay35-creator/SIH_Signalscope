"""
SignalScope Unified Application Launcher.
Responsible Team Member: Member 4 (Backend API & Service Layer)
"""

from api.main import app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
