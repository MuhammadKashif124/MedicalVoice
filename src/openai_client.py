"""
OpenAI client for handling Realtime API interactions
"""

import json
import logging
import websocket
import asyncio
from typing import Callable, Dict, Any, Optional, List
import base64

from .config import Settings

logger = logging.getLogger(__name__)

class OpenAIRealtimeClient:
    """Client for interacting with OpenAI's Realtime API via WebSockets"""
    
    def __init__(self, settings: Settings):
        """
        Initialize the OpenAI Realtime client
        
        Args:
            settings: Application settings
        """
        self.settings = settings
        self.ws_url = f"{settings.openai_websocket_url}?model={settings.openai_model}"
        self.ws = None
        self.connected = False
        self._on_message_callback = None
        self._on_close_callback = None
        self.active_function_calls = {}
        
    async def connect(self, 
                     on_message_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
                     on_close_callback: Optional[Callable[[], None]] = None,
                     tools: Optional[List[Dict[str, Any]]] = None):
        """
        Connect to the OpenAI Realtime API WebSocket
        
        Args:
            on_message_callback: Callback for handling incoming messages
            on_close_callback: Callback for handling WebSocket closure
            tools: Function definitions for OpenAI function calling
        """
        logger.info(f"Connecting to OpenAI Realtime API: {self.ws_url}")
        
        self._on_message_callback = on_message_callback
        self._on_close_callback = on_close_callback
        
        # Initialize WebSocket connection
        self.ws = await self._create_websocket_connection()
        
        # Send session configuration
        await self.update_session(tools=tools)
        
        self.connected = True
        logger.info("Connected to OpenAI Realtime API")
        
    async def _create_websocket_connection(self):
        """Create a WebSocket connection to the OpenAI Realtime API"""
        headers = {
            "Authorization": f"Bearer {self.settings.openai_api_key}",
            "OpenAI-Beta": "realtime=v1"
        }
        
        # Create connection
        ws = websocket.WebSocketApp(
            self.ws_url,
            header=headers,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close,
            on_open=self._on_open
        )
        
        # Start WebSocket connection in a separate thread
        websocket_thread = asyncio.to_thread(ws.run_forever)
        asyncio.create_task(websocket_thread)
        
        # Wait for connection to be established
        while not ws.sock or not ws.sock.connected:
            await asyncio.sleep(0.1)
            
        return ws
        
    def _on_message(self, ws, message):
        """Handle incoming WebSocket messages"""
        try:
            data = json.loads(message)
            logger.debug(f"Received message from OpenAI: {data.get('type')}")
            
            # Track function calls
            if data.get("type") == "response.function_call_arguments.done":
                function_call = data.get("function_call", {})
                call_id = function_call.get("id")
                if call_id:
                    self.active_function_calls[call_id] = function_call
            
            if self._on_message_callback:
                self._on_message_callback(data)
                
        except Exception as e:
            logger.error(f"Error processing OpenAI message: {str(e)}")
    
    def _on_error(self, ws, error):
        """Handle WebSocket errors"""
        logger.error(f"OpenAI WebSocket error: {str(error)}")
    
    def _on_close(self, ws, close_status_code, close_msg):
        """Handle WebSocket closure"""
        logger.info(f"OpenAI WebSocket closed: {close_status_code} - {close_msg}")
        self.connected = False
        
        if self._on_close_callback:
            self._on_close_callback()
    
    def _on_open(self, ws):
        """Handle WebSocket connection open"""
        logger.info("OpenAI WebSocket connection opened")
    
    async def update_session(self, tools: Optional[List[Dict[str, Any]]] = None):
        """
        Update the OpenAI session configuration
        
        Args:
            tools: Function definitions for OpenAI function calling
        """
        if not self.ws or not self.connected:
            logger.error("Cannot update session: WebSocket not connected")
            return
        
        session_config = {
            "type": "session.update",
            "session": {
                "voice": self.settings.openai_voice,
                "instructions": self.settings.system_instructions,
                "input_audio_format": self.settings.input_audio_format,
                "output_audio_format": self.settings.output_audio_format
            }
        }
        
        # Add tools if provided
        if tools:
            session_config["session"]["tools"] = tools
            
        self.ws.send(json.dumps(session_config))
        logger.info("Sent session configuration to OpenAI")
    
    async def send_audio(self, audio_data: bytes):
        """
        Send audio data to OpenAI
        
        Args:
            audio_data: Audio data in the format specified in settings
        """
        if not self.ws or not self.connected:
            logger.error("Cannot send audio: WebSocket not connected")
            return
        
        audio_message = {
            "type": "input_audio_buffer.append",
            "audio": base64.b64encode(audio_data).decode('utf-8')
        }
        
        self.ws.send(json.dumps(audio_message))
    
    async def send_function_result(self, function_name: str, result: str, call_id: Optional[str] = None):
        """
        Send a function call result back to OpenAI
        
        Args:
            function_name: The name of the function that was called
            result: The result of the function call
            call_id: The ID of the function call (if known)
        """
        if not self.ws or not self.connected:
            logger.error("Cannot send function result: WebSocket not connected")
            return
        
        # If call_id is not provided, try to find it from active function calls
        if not call_id:
            for active_call_id, call_info in self.active_function_calls.items():
                if call_info.get("name") == function_name:
                    call_id = active_call_id
                    break
        
        if not call_id:
            logger.error(f"Cannot send function result: No call_id found for function {function_name}")
            return
        
        function_result = {
            "type": "conversation.item.create",
            "item": {
                "type": "function_call_output",
                "call_id": call_id,
                "output": result
            }
        }
        
        self.ws.send(json.dumps(function_result))
        logger.info(f"Sent function result for {function_name} with call_id {call_id}")
        
        # Remove from active function calls
        if call_id in self.active_function_calls:
            del self.active_function_calls[call_id]
    
    async def close(self):
        """Close the WebSocket connection"""
        if self.ws and self.connected:
            self.ws.close()
            self.connected = False
            logger.info("Closed OpenAI WebSocket connection") 