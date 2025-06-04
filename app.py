#!/usr/bin/env python3
"""
Main application entry point for MedicalVoice - AI call agent using VoIPstudio and OpenAI
"""

import os
import uvicorn
from fastapi import FastAPI, WebSocket, Request
from fastapi.responses import HTMLResponse, Response
from fastapi.middleware.cors import CORSMiddleware
import logging
from dotenv import load_dotenv

from src.config import Settings
from src.call_handler import CallHandler
from src.voipstudio_client import VoIPstudioClient

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Custom XMLResponse class since FastAPI doesn't have one built-in
class XMLResponse(Response):
    media_type = "application/xml"

# Initialize settings
settings = Settings()

# Initialize FastAPI app
app = FastAPI(
    title="MedicalVoice",
    description="A custom AI call agent using VoIPstudio and OpenAI's Realtime API",
    version="0.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize clients
voipstudio_client = VoIPstudioClient(api_key=settings.voipstudio_api_key)

@app.get("/")
async def get_home():
    """Home endpoint to verify the service is running"""
    return {"status": "online", "service": "MedicalVoice"}

@app.post("/incoming-call")
async def incoming_call(request: Request):
    """
    Handle incoming calls from VoIPstudio
    Returns XML response to connect the call to the WebSocket endpoint
    """
    logger.info("Received incoming call webhook")
    
    # Get the host from the request
    host = request.headers.get("host", "localhost:5000")
    
    # Create XML response to establish WebSocket connection
    xml_response = f"""
    <?xml version="1.0" encoding="UTF-8"?>
    <Response>
        <Say>Welcome to Medical Voice Assistant. How can I help you today?</Say>
        <Connect>
            <Stream url="wss://{host}/media-stream"/>
        </Connect>
    </Response>
    """
    
    return XMLResponse(content=xml_response)

@app.websocket("/media-stream")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for streaming audio between VoIPstudio and OpenAI
    """
    await websocket.accept()
    logger.info("WebSocket connection established")
    
    # Initialize call handler
    call_handler = CallHandler(settings=settings)
    
    try:
        # Start handling the call
        await call_handler.handle_call(websocket)
    except Exception as e:
        logger.error(f"Error in websocket connection: {str(e)}")
    finally:
        logger.info("WebSocket connection closed")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    logger.info(f"Starting server on port {port}")
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True) 