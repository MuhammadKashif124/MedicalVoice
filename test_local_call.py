#!/usr/bin/env python3
"""
Local voice call simulation for MedicalVoice
This script allows you to test the voice interaction with the agent locally,
simulating a real call without needing to configure VoIPstudio.
Uses sounddevice instead of PyAudio for audio input/output.
"""

import asyncio
import json
import os
import logging
import sounddevice as sd
import numpy as np
import wave
import base64
import time
import threading
import queue
import websockets.client as websockets
import aiohttp
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Audio settings
SAMPLE_RATE = 24000  # Match the OpenAI Realtime API requirements
CHANNELS = 1
DTYPE = 'int16'
BLOCKSIZE = 1024
SILENCE_THRESHOLD = 0.005  # Adjust based on your microphone sensitivity (for sounddevice)
MAX_SILENCE_FRAMES = 30  # Number of silent frames before considering end of speech

class AudioProcessor:
    """Handles audio input and output via microphone and speakers using sounddevice"""
    
    def __init__(self):
        self.audio_queue = queue.Queue()
        self.is_recording = False
        self.recording_thread = None
        self.is_playing = False
        
    def start_recording(self):
        """Start recording audio from microphone"""
        self.is_recording = True
        self.recording_thread = threading.Thread(target=self._record_audio)
        self.recording_thread.daemon = True
        self.recording_thread.start()
        logger.info("Started recording")
        
    def stop_recording(self):
        """Stop recording audio"""
        self.is_recording = False
        if self.recording_thread:
            self.recording_thread.join(timeout=1)
        logger.info("Stopped recording")
        
    def _record_audio(self):
        """Record audio from microphone in a separate thread"""
        silent_frames = 0
        
        def audio_callback(indata, frames, time, status):
            """This is called for each audio block"""
            if status:
                logger.warning(f"Audio callback status: {status}")
            
            # Convert to bytes
            audio_data = indata.copy().tobytes()
            self.audio_queue.put(audio_data)
            
            # Check for silence to auto-detect end of speech
            if self._is_silent(indata):
                nonlocal silent_frames
                silent_frames += 1
                if silent_frames > MAX_SILENCE_FRAMES:
                    # End of speech detected
                    self.audio_queue.put(None)  # Signal end of utterance
                    silent_frames = 0
            else:
                silent_frames = 0
        
        try:
            with sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                dtype=DTYPE,
                blocksize=BLOCKSIZE,
                callback=audio_callback
            ):
                while self.is_recording:
                    time.sleep(0.1)  # Just to avoid busy waiting
                    
        except Exception as e:
            logger.error(f"Error recording audio: {str(e)}")
    
    def _is_silent(self, audio_data):
        """Check if the audio chunk is silent"""
        # Calculate RMS amplitude
        rms = np.sqrt(np.mean(audio_data**2))
        return rms < SILENCE_THRESHOLD
    
    def get_audio_chunk(self) -> Optional[bytes]:
        """Get a chunk of audio from the queue, or None if end of utterance"""
        try:
            return self.audio_queue.get_nowait()
        except queue.Empty:
            return b''
    
    def play_audio(self, audio_data: bytes):
        """Play audio through speakers"""
        if not audio_data:
            return
        
        # Avoid playing if already playing
        if self.is_playing:
            return
            
        self.is_playing = True
        threading.Thread(
            target=self._play_audio_thread, 
            args=(audio_data,),
            daemon=True
        ).start()
    
    def _play_audio_thread(self, audio_data: bytes):
        """Play audio in a separate thread"""
        try:
            # Convert bytes to numpy array
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            
            # Play the audio
            sd.play(audio_array, SAMPLE_RATE, blocking=True)
            sd.wait()
        except Exception as e:
            logger.error(f"Error playing audio: {str(e)}")
        finally:
            self.is_playing = False
    
    def close(self):
        """Close the audio processor"""
        self.stop_recording()

class LocalCallSimulator:
    """Simulates a local call with the MedicalVoice agent"""
    
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o-realtime-preview-2024-10-01")
        self.voice = os.getenv("OPENAI_VOICE", "alloy")
        self.websocket_url = os.getenv("OPENAI_WEBSOCKET_URL", "wss://api.openai.com/v1/realtime")
        self.url = f"{self.websocket_url}?model={self.model}"
        
        self.ws = None
        self.audio_processor = AudioProcessor()
        self.conversation_active = False
        self.listening_for_user = False
        self.agent_speaking = False
        
    async def connect(self):
        """Connect to the OpenAI Realtime API"""
        logger.info(f"Connecting to OpenAI Realtime API")
        try:
            # Create custom connection to OpenAI with headers
            import ssl
            
            # Create a custom SSL context that doesn't verify certificates
            # Note: In production, you should use proper certificate verification
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
            
            # This session will be used for websocket connection
            self.session = aiohttp.ClientSession()
            
            # Connect with proper headers
            self.ws = await self.session.ws_connect(
                self.url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "OpenAI-Beta": "realtime=v1"
                },
                ssl=ssl_context  # Use our custom SSL context
            )
            
            logger.info("Connected successfully")
            
            # Update session with tools and configuration
            await self.update_session()
        except Exception as e:
            logger.error(f"Failed to connect to OpenAI: {str(e)}")
            raise
        
    async def update_session(self):
        """Configure the session with medical billing tools and settings"""
        # Import the Settings to get system instructions
        from src.config import Settings
        settings = Settings()
        
        session_config = {
            "type": "session.update",
            "session": {
                "voice": self.voice,
                "instructions": settings.system_instructions,
                "input_audio_format": "pcm16",
                "output_audio_format": "pcm16",
                "turn_detection": {"type": "server_vad"}
                # Removed tools for now to get basic functionality working
            }
        }
        
        await self.ws.send_str(json.dumps(session_config))
        logger.info("Session configured")
    
    async def start_call(self):
        """Start a simulated call with the agent"""
        self.conversation_active = True
        self.audio_processor.start_recording()
        
        print("\n" + "-" * 50)
        print("🎙️ Medical Billing Voice Assistant - Local Call Simulation")
        print("-" * 50)
        print("🔈 Speak into your microphone to talk to the agent")
        print("🛑 Press Ctrl+C to end the call")
        print("-" * 50 + "\n")
        
        # Welcome message
        greeting_message = {
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "assistant",
                "content": [{
                    "type": "text",
                    "text": "Hello, I'm your Medical Billing Assistant. How can I help you today?"
                }]
            }
        }
        await self.ws.send_str(json.dumps(greeting_message))
        print("🤖 Assistant: Hello, I'm your Medical Billing Assistant. How can I help you today?")
        
        # Start listening for user input
        await self.listen_for_user_input()
        
    async def listen_for_user_input(self):
        """Listen for user input and send to OpenAI"""
        print("\n🎤 Listening... (speak now)")
        self.listening_for_user = True
        audio_buffer = bytearray()
        end_of_speech = False
        
        # Start a new buffer for this utterance
        while self.listening_for_user and self.conversation_active:
            chunk = self.audio_processor.get_audio_chunk()
            
            if chunk is None:
                # End of utterance marker
                end_of_speech = True
                break
                
            if chunk:
                audio_buffer.extend(chunk)
            
            # Small pause to prevent CPU spinning
            await asyncio.sleep(0.01)
            
            # If we've collected enough audio, send it
            if len(audio_buffer) > SAMPLE_RATE * 0.5:  # Send every 0.5 seconds
                if audio_buffer:
                    await self.send_audio_chunk(bytes(audio_buffer))
                    audio_buffer = bytearray()
        
        # Send any remaining audio
        if audio_buffer:
            await self.send_audio_chunk(bytes(audio_buffer))
        
        if end_of_speech:
            print("\n🎤 End of speech detected")
            await self.commit_audio_buffer()
        
    async def send_audio_chunk(self, audio_data: bytes):
        """Send an audio chunk to OpenAI"""
        if not audio_data or not self.ws:
            return
            
        audio_message = {
            "type": "input_audio_buffer.append",
            "audio": base64.b64encode(audio_data).decode('utf-8')
        }
        
        await self.ws.send_str(json.dumps(audio_message))
        
    async def commit_audio_buffer(self):
        """Commit the audio buffer and request a response"""
        # Commit the buffer
        await self.ws.send_str(json.dumps({"type": "input_audio_buffer.commit"}))
        
        # Request a response with both text and audio
        await self.ws.send_str(json.dumps({
            "type": "response.create",
            "response": {
                "modalities": ["text", "audio"]
            }
        }))
        
        self.listening_for_user = False
        print("⏳ Processing...")
        
    async def handle_messages(self):
        """Handle messages from OpenAI"""
        try:
            while self.conversation_active:
                msg = await self.ws.receive()
                
                if msg.type == aiohttp.WSMsgType.TEXT:
                    data = json.loads(msg.data)
                    message_type = data.get("type")
                    
                    # Handle different message types
                    if message_type == "response.audio.delta":
                        # Audio response from the assistant
                        audio_data = data.get("delta")
                        if audio_data:
                            # Decode base64 audio
                            decoded_audio = base64.b64decode(audio_data)
                            self.audio_processor.play_audio(decoded_audio)
                            self.agent_speaking = True
                            
                    elif message_type == "response.audio.done":
                        # Audio response complete
                        self.agent_speaking = False
                        # Start listening for the next user input
                        if self.conversation_active and not self.listening_for_user:
                            await asyncio.sleep(0.5)  # Small pause before listening again
                            await self.listen_for_user_input()
                            
                    elif message_type == "response.text.delta":
                        # Text delta from the assistant
                        text_delta = data.get("delta", "")
                        # Only print the first delta to avoid cluttering the console
                        if not self.agent_speaking:
                            print("\n🤖 Assistant: ", end="", flush=True)
                            self.agent_speaking = True
                        print(text_delta, end="", flush=True)
                        
                    elif message_type == "response.text.done":
                        # Text response complete
                        print()  # New line
                        
                    elif message_type == "response.function_call_arguments.delta":
                        # Function call
                        function_name = data.get("function_call", {}).get("name")
                        if function_name:
                            print(f"\n⚙️ Looking up information: {function_name}")
                        
                    elif message_type == "error":
                        # Error from the API
                        error = data.get("error", {})
                        logger.error(f"Error from OpenAI: {error.get('message')}")
                        print(f"\n❌ Error: {error.get('message')}")
                elif msg.type == aiohttp.WSMsgType.CLOSED:
                    logger.info("WebSocket connection closed")
                    break
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    logger.error(f"WebSocket error: {msg.data}")
                    break
                    
        except Exception as e:
            logger.error(f"Error handling messages: {str(e)}")
            if self.conversation_active:
                print(f"\n❌ Connection error: {str(e)}")
                self.conversation_active = False
    
    async def close(self):
        """Close the call simulation"""
        self.conversation_active = False
        self.listening_for_user = False
        
        if self.ws:
            await self.ws.close()
            
        if hasattr(self, 'session'):
            await self.session.close()
            
        self.audio_processor.close()
        print("\n👋 Call ended. Thank you for using the Medical Billing Voice Assistant.")

async def main():
    """Main function to run the call simulation"""
    simulator = LocalCallSimulator()
    
    try:
        await simulator.connect()
        
        # Start handling messages in the background
        message_handler = asyncio.create_task(simulator.handle_messages())
        
        # Start the call
        await simulator.start_call()
        
        # Keep the simulation running until user interrupts
        while simulator.conversation_active:
            await asyncio.sleep(1)
            
    except KeyboardInterrupt:
        print("\n\n🛑 Ending call...")
    except Exception as e:
        logger.error(f"Error in call simulation: {str(e)}")
        print(f"\n❌ Error: {str(e)}")
    finally:
        if 'simulator' in locals():
            await simulator.close()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n👋 Call ended by user.") 