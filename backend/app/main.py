"""
Plant Guard AI - FastAPI Backend Entry Point
=============================================
Main application entry point. Initializes the FastAPI app,
registers routers, and defines core endpoints.
"""

from fastapi import FastAPI

# TODO: Import and include routers from app/api/
# TODO: Setup CORS middleware from app/middleware/
# TODO: Initialize database connection from app/database/

app = FastAPI(
    title="Plant Guard AI API",
    description="Backend API for the Plant Guard AI disease detection system.",
    version="0.1.0",
)


@app.get("/health", tags=["Health"])
async def health_check():
    """
    Health check endpoint to verify the API is running.

    Returns:
        dict: Status message.
    """
    return {"status": "ok", "service": "Plant Guard AI API", "version": "0.1.0"}


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint."""
    return {"message": "Welcome to Plant Guard AI. Visit /docs for API documentation."}


# TODO: Add startup and shutdown event handlers
# TODO: Register exception handlers
