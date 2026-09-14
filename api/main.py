"""
FastAPI Main Application Entrypoint.
Initializes web service, mounts routers, and manages application lifecycle.
Responsible Team Member: Member 4 (Backend API & Service Layer)
"""

from fastapi import FastAPI

app = FastAPI(
    title="SignalScope API",
    description="AI-Generated Image Detection and Explainability API",
    version="1.0.0"
)

@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "SignalScope API"}
