#!/usr/bin/env python3
"""
Local voice call simulation for MedicalVoice
This script allows you to test the voice interaction with the agent locally,
simulating a real call without needing to configure VoIPstudio.
Uses sounddevice for audio input/output with minimal latency.

Note: While OpenAI recommends WebRTC for browser implementations,
      this server-side implementation uses WebSockets with sounddevice
      which is more appropriate for Python server applications.
"""

import asyncio
import json
import os
import logging
import sounddevice as sd
import numpy as np
import base64
import time
import queue
import aiohttp
import argparse
import threading
import traceback
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv

# Parse command line arguments
parser = argparse.ArgumentParser(description='Local voice call simulation for MedicalVoice')
parser.add_argument('--debug', action='store_true', help='Enable debug logging')
parser.add_argument('--latency', type=float, default=0.05, 
                   help='Target latency in seconds (default: 0.05)')
parser.add_argument('--blocksize', type=int, default=1024,
                   help='Audio block size (default: 1024)')
parser.add_argument('--model', type=str, default=None,
                   help='OpenAI model to use (default: from .env or gpt-4o-realtime-preview)')
parser.add_argument('--voice', type=str, default=None,
                   help='Voice to use (default: from .env or "alloy")')
args = parser.parse_args()

# Load environment variables
load_dotenv()

# Configure logging
log_level = logging.DEBUG if args.debug else logging.INFO
logging.basicConfig(
    level=log_level,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Audio settings
SAMPLE_RATE = 24000  # Match the OpenAI Realtime API requirements
CHANNELS = 1
DTYPE = np.int16
BLOCKSIZE = args.blocksize
SILENCE_THRESHOLD = 0.005
MAX_SILENCE_FRAMES = 30
LATENCY = args.latency

class AudioProcessor:
    """Handles audio input and output via microphone and speakers"""
    
    def __init__(self):
        self.input_queue = queue.Queue()
        self.output_queue = queue.Queue()
        self.is_active = False
        self.stream = None
        self.silent_frames = 0
        self.event = threading.Event()
        
        # Check audio devices
        try:
            devices = sd.query_devices()
            logger.info(f"Found {len(devices)} audio devices")
            if args.debug:
                for i, device in enumerate(devices):
                    logger.debug(f"Device {i}: {device['name']}")
        except Exception as e:
            logger.error(f"Error querying audio devices: {str(e)}")
        
    def audio_callback(self, indata, outdata, frames, time, status):
        """Main audio callback for both input and output"""
        if status:
            logger.warning(f"Audio callback status: {status}")
            
        # Process input: check for silence and queue the data
        try:
            is_silent = self._is_silent(indata)
            if is_silent:
                self.silent_frames += 1
                if self.silent_frames > MAX_SILENCE_FRAMES:
                    # End of speech detected, send special marker
                    self.input_queue.put(None)
                    self.silent_frames = 0
            else:
                self.silent_frames = 0
                
            # Always queue the input data for processing
            self.input_queue.put(indata.copy())
            
            # Process output: get data from the queue or generate silence
            try:
                output_data = self.output_queue.get_nowait()
                outdata[:] = output_data
            except queue.Empty:
                # No output data available, output silence
                outdata.fill(0)
        except Exception as e:
            # Make sure we don't crash the audio stream if there's an error
            logger.error(f"Error in audio callback: {str(e)}")
            outdata.fill(0)  # Always provide silence in case of error
    
    def _is_silent(self, audio_data):
        """Check if the audio chunk is silent"""
        # Check for invalid values to avoid sqrt warnings
        if np.any(np.isnan(audio_data)) or np.all(audio_data == 0):
            return True
            
        # Calculate RMS with try/except to handle any numerical issues
        try:
            # Convert to float64 to avoid numerical issues
            audio_float = audio_data.astype(np.float64)
            # Use safe operations to avoid warnings
            squared = np.square(audio_float)
            mean_squared = np.mean(squared)
            if mean_squared <= 0:
                return True
            rms = np.sqrt(mean_squared)
            return rms < SILENCE_THRESHOLD
        except Exception as e:
            logger.warning(f"Error calculating audio RMS: {str(e)}, treating as silence")
            return True
    
    def start(self):
        """Start the audio stream for simultaneous input/output"""
        if self.is_active:
            return
            
        self.is_active = True
        
        # Clear any existing queues
        while not self.input_queue.empty():
            self.input_queue.get()
        while not self.output_queue.empty():
            self.output_queue.get()
            
        # Start the stream with our callback
        try:
            self.stream = sd.Stream(
                samplerate=SAMPLE_RATE,
                blocksize=BLOCKSIZE,
                channels=CHANNELS,
                dtype=DTYPE,
                callback=self.audio_callback,
                latency=LATENCY
            )
            
            self.stream.start()
            logger.info(f"Started audio stream with latency={LATENCY}s, blocksize={BLOCKSIZE}")
        except Exception as e:
            logger.error(f"Failed to start audio stream: {str(e)}")
            if args.debug:
                logger.debug(traceback.format_exc())
            raise
    
    def stop(self):
        """Stop the audio stream"""
        if not self.is_active:
            return
            
        self.is_active = False
        if self.stream:
            try:
                # Add a small delay to prevent KeyboardInterrupt issues
                time.sleep(0.1)
                self.stream.stop()
                self.stream.close()
            except Exception as e:
                logger.error(f"Error stopping audio stream: {str(e)}")
            self.stream = None
        logger.info("Stopped audio stream")
    
    def get_audio_chunk(self) -> Optional[bytes]:
        """Get a chunk of audio from the input queue"""
        try:
            data = self.input_queue.get_nowait()
            # If None, it's an end-of-speech marker
            if data is None:
                return None
            # Convert numpy array to bytes
            return data.tobytes()
        except queue.Empty:
            return b''
    
    def play_audio(self, audio_data: bytes):
        """Queue audio data for playback"""
        if not audio_data or not self.is_active:
            return
            
        # Convert bytes to numpy array
        try:
            audio_array = np.frombuffer(audio_data, dtype=DTYPE)
            
            # Reshape to match the expected format
            audio_array = audio_array.reshape(-1, CHANNELS)
            
            # Queue chunks of blocksize for smooth playback
            for i in range(0, len(audio_array), BLOCKSIZE):
                chunk = audio_array[i:i+BLOCKSIZE]
                # Pad with zeros if needed
                if len(chunk) < BLOCKSIZE:
                    padded = np.zeros((BLOCKSIZE, CHANNELS), dtype=DTYPE)
                    padded[:len(chunk)] = chunk
                    chunk = padded
                
                self.output_queue.put(chunk)
                
        except Exception as e:
            logger.error(f"Error queuing audio for playback: {str(e)}")
            if args.debug:
                logger.debug(traceback.format_exc())
    
    def close(self):
        """Close the audio processor"""
        self.stop()


class LocalCallSimulator:
    """Simulates a local call with the MedicalVoice agent using OpenAI Realtime API"""
    
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.model = args.model or os.getenv("OPENAI_MODEL", "gpt-4o-realtime-preview-2024-10-01")
        self.voice = args.voice or os.getenv("OPENAI_VOICE", "alloy")
        self.websocket_url = os.getenv("OPENAI_WEBSOCKET_URL", "wss://api.openai.com/v1/realtime")
        self.url = f"{self.websocket_url}?model={self.model}"
        
        self.ws = None
        self.audio_processor = AudioProcessor()
        self.conversation_active = False
        self.listening_for_user = False
        self.agent_speaking = False
        self.current_response_id = None
        self.session_id = None
        self.conversation_items = []
        self.user_speech_started = False
        
        # WebSocket connection retry settings
        self.max_retries = 3
        self.retry_delay = 2  # seconds
        
        logger.info(f"Initialized with model={self.model}, voice={self.voice}")
        
    async def connect(self):
        """Connect to the OpenAI Realtime API with retry logic"""
        logger.info(f"Connecting to OpenAI Realtime API: {self.url}")
        
        retries = 0
        last_error = None
        
        while retries < self.max_retries:
            try:
                # Create a custom SSL context with proper verification
                import ssl
                import certifi
                ssl_context = ssl.create_default_context(cafile=certifi.where())
                
                # This session will be used for websocket connection
                if hasattr(self, 'session') and not self.session.closed:
                    await self.session.close()
                    
                self.session = aiohttp.ClientSession()
                
                # Connect with proper headers per the OpenAI Realtime API documentation
                self.ws = await self.session.ws_connect(
                    self.url,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "OpenAI-Beta": "realtime=v1",
                        "Content-Type": "application/json"
                    },
                    ssl=ssl_context,
                    heartbeat=30.0  # Send a ping every 30 seconds to keep connection alive
                )
                
                logger.info("Connected successfully to Realtime API")
                
                # Update session with required settings
                await self.update_session()
                return
                
            except aiohttp.ClientError as e:
                last_error = e
                logger.warning(f"Connection attempt {retries+1} failed: {str(e)}")
                retries += 1
                if retries < self.max_retries:
                    await asyncio.sleep(self.retry_delay)
            except Exception as e:
                logger.error(f"Failed to connect to OpenAI Realtime API: {str(e)}")
                if args.debug:
                    logger.debug(traceback.format_exc())
                raise
                
        if last_error:
            logger.error(f"Failed to connect after {self.max_retries} attempts")
            raise last_error
        
    async def update_session(self):
        """
        Configure the session with medical billing tools and settings
        Using the session.update event type as per OpenAI Realtime API docs
        """
        # Import the Settings to get system instructions
        from src.config import Settings
        settings = Settings()
        
        # Create a session config according to the API documentation
        session_config = {
            "type": "session.update",
            "session": {
                "voice": self.voice,
                "instructions": settings.system_instructions,
                "input_audio_format": "pcm16",
                "output_audio_format": "pcm16",
                "turn_detection": {"type": "server_vad"}
            }
        }
        
        try:
            await self.ws.send_str(json.dumps(session_config))
            logger.info("Session configured with server-side VAD and pcm16 audio format")
        except Exception as e:
            logger.error(f"Failed to update session configuration: {str(e)}")
            if args.debug:
                logger.debug(traceback.format_exc())
            raise
    
    async def start_call(self):
        """Start a simulated call with the agent"""
        self.conversation_active = True
        self.audio_processor.start()
        
        print("\n" + "-" * 50)
        print("🎙️ Medical Billing Voice Assistant - Local Call Simulation")
        print("-" * 50)
        print("🔈 Speak into your microphone to talk to the agent")
        print("🛑 Press Ctrl+C to end the call")
        print("-" * 50 + "\n")
        
        # Create a welcome message as the first conversation item
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
        
        try:
            await self.ws.send_str(json.dumps(greeting_message))
            print("🤖 Assistant: Hello, I'm your Medical Billing Assistant. How can I help you today?")
            
            # Start listening for user input
            await self.listen_for_user_input()
        except Exception as e:
            logger.error(f"Error starting call: {str(e)}")
            if args.debug:
                logger.debug(traceback.format_exc())
            self.conversation_active = False
        
    async def listen_for_user_input(self):
        """Listen for user input and send to OpenAI"""
        print("\n🎤 Listening... (speak now)")
        self.listening_for_user = True
        self.user_speech_started = False
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
                    await self.append_audio_chunk(bytes(audio_buffer))
                    audio_buffer = bytearray()
        
        # Send any remaining audio
        if audio_buffer:
            await self.append_audio_chunk(bytes(audio_buffer))
        
        if end_of_speech:
            print("\n🎤 End of speech detected")
            await self.commit_audio_buffer()
        
    async def append_audio_chunk(self, audio_data: bytes):
        """
        Append an audio chunk to the input buffer using input_audio_buffer.append event
        as per OpenAI Realtime API docs
        """
        if not audio_data or not self.ws:
            return
            
        audio_message = {
            "type": "input_audio_buffer.append",
            "audio": base64.b64encode(audio_data).decode('utf-8')
        }
        
        try:
            await self.ws.send_str(json.dumps(audio_message))
            if args.debug:
                logger.debug(f"Sent {len(audio_data)} bytes of audio to Realtime API")
                
            # If the agent is speaking and user starts speaking, we'll need to handle interruption
            # The server will send input_audio_buffer.speech_started event when detected
            if self.agent_speaking and not self.user_speech_started:
                self.user_speech_started = True
        except Exception as e:
            logger.error(f"Failed to append audio chunk: {str(e)}")
            if args.debug:
                logger.debug(traceback.format_exc())
            # If we encounter a connection issue, try to recover
            if isinstance(e, (aiohttp.ClientError, aiohttp.WSServerHandshakeError)):
                logger.info("Attempting to reconnect...")
                await self.reconnect()
        
    async def commit_audio_buffer(self):
        """
        Commit the audio buffer and request a response
        Following the Realtime API documentation flow
        """
        try:
            # Step 1: Commit the audio buffer
            await self.ws.send_str(json.dumps({
                "type": "input_audio_buffer.commit"
            }))
            
            # Step 2: Create a response with both text and audio modalities
            await self.ws.send_str(json.dumps({
                "type": "response.create",
                "response": {
                    "modalities": ["text", "audio"]
                }
            }))
            
            self.listening_for_user = False
            print("⏳ Processing...")
        except Exception as e:
            logger.error(f"Failed to commit audio buffer: {str(e)}")
            if args.debug:
                logger.debug(traceback.format_exc())
            self.listening_for_user = True  # Reset to allow retrying
            
    async def reconnect(self):
        """Attempt to reconnect to the Realtime API if connection is lost"""
        if self.ws:
            try:
                await self.ws.close()
            except:
                pass
                
        try:
            await self.connect()
            logger.info("Successfully reconnected to Realtime API")
            return True
        except Exception as e:
            logger.error(f"Failed to reconnect: {str(e)}")
            return False
        
    async def handle_messages(self):
        """Handle messages from OpenAI Realtime API"""
        try:
            received_audio_chunks = 0
            current_item_id = None
            content_index = 0
            
            while self.conversation_active:
                try:
                    msg = await self.ws.receive(timeout=60.0)  # Add timeout to detect stalled connections
                    
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        data = json.loads(msg.data)
                        message_type = data.get("type")
                        
                        if args.debug:
                            logger.debug(f"Received event: {message_type}")
                        
                        # Track session ID if provided
                        if data.get("session", {}).get("id") and not self.session_id:
                            self.session_id = data["session"]["id"]
                            logger.info(f"Session ID: {self.session_id}")
                        
                        # Handle different message types according to the Realtime API docs
                        if message_type == "input_audio_buffer.speech_started":
                            # User started speaking while assistant is speaking - implement barge-in
                            logger.info("Speech started while assistant is speaking - implementing barge-in")
                            if self.agent_speaking:
                                # Cancel the current response
                                await self.interrupt_response()
                                print("\n🎤 You: [interrupting...]")
                                
                                # Mark that user speech has started
                                self.user_speech_started = True
                                
                                # Make sure we're now listening for user input
                                if not self.listening_for_user:
                                    self.listening_for_user = True
                        
                        elif message_type == "response.audio.delta":
                            # Audio response from the assistant
                            audio_data = data.get("delta")
                            if audio_data:
                                # Decode base64 audio and play immediately
                                decoded_audio = base64.b64decode(audio_data)
                                self.audio_processor.play_audio(decoded_audio)
                                received_audio_chunks += 1
                                
                                # Set speaking flag on first chunk
                                if received_audio_chunks == 1:
                                    self.agent_speaking = True
                                    self.current_response_id = data.get("response_id")
                                
                        elif message_type == "response.audio.done":
                            # Audio response complete
                            logger.info(f"Audio response complete - received {received_audio_chunks} chunks")
                            received_audio_chunks = 0
                            self.current_response_id = None
                            
                            self.agent_speaking = False
                            # Start listening for the next user input after a short pause
                            if self.conversation_active and not self.listening_for_user:
                                await asyncio.sleep(0.2)  # Small pause before listening again
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
                            
                        elif message_type == "conversation.item.created":
                            # A new item was added to the conversation
                            item = data.get("item", {})
                            
                            # Track conversation items
                            if item:
                                self.conversation_items.append(item)
                                current_item_id = item.get("id")
                                # Reset content index for new items
                                content_index = 0
                                
                            if item.get("type") == "function_call":
                                function_name = item.get("function_call", {}).get("name")
                                if function_name:
                                    print(f"\n⚙️ Looking up information: {function_name}")
                        
                        elif message_type == "response.content_part.added":
                            # Track the content part index
                            content_index = data.get("index", 0)
                        
                        elif message_type == "conversation.item.truncated":
                            # Item was truncated (after an interruption)
                            logger.info(f"Conversation item truncated: {data.get('item_id')}")
                        
                        elif message_type == "session.created" or message_type == "session.updated":
                            # Session created or updated
                            logger.info(f"Session {message_type.split('.')[1]}")
                            
                        elif message_type == "error":
                            # Error from the API
                            error = data.get("error", {})
                            error_type = error.get("type", "unknown")
                            error_msg = error.get("message", "Unknown error")
                            
                            # If error relates to cancellation, just log it but don't show to user
                            # since it's likely because we tried to cancel an already completed response
                            if "Cancellation failed" in error_msg:
                                logger.warning(f"Cancellation error: {error_msg}")
                            else:
                                logger.error(f"Error from OpenAI Realtime API: {error_type} - {error_msg}")
                                print(f"\n❌ Error: {error_msg}")
                            
                    elif msg.type == aiohttp.WSMsgType.CLOSED:
                        logger.info("WebSocket connection closed")
                        if await self.reconnect():
                            continue
                        else:
                            break
                            
                    elif msg.type == aiohttp.WSMsgType.ERROR:
                        logger.error(f"WebSocket error: {msg.data}")
                        if await self.reconnect():
                            continue
                        else:
                            break
                            
                    elif msg.type == aiohttp.WSMsgType.PING:
                        # Respond to ping with pong
                        await self.ws.pong()
                        if args.debug:
                            logger.debug("Received ping, sent pong")
                            
                except asyncio.TimeoutError:
                    # Check if connection is still alive
                    logger.warning("Connection timeout detected, sending ping")
                    try:
                        pong_waiter = await self.ws.ping()
                        await asyncio.wait_for(pong_waiter, timeout=5)
                        logger.info("Ping successful, connection is alive")
                    except:
                        logger.error("Ping failed, reconnecting...")
                        if await self.reconnect():
                            continue
                        else:
                            break
                            
        except Exception as e:
            logger.error(f"Error handling messages: {str(e)}")
            if args.debug:
                logger.debug(traceback.format_exc())
            if self.conversation_active:
                print(f"\n❌ Connection error: {str(e)}")
                self.conversation_active = False
    
    async def interrupt_response(self):
        """Interrupt the current response if the user starts speaking"""
        if not self.agent_speaking:
            # Already interrupted or not speaking
            return
            
        if self.current_response_id:
            try:
                # Send response.cancel event to stop the assistant's response
                await self.ws.send_str(json.dumps({
                    "type": "response.cancel",
                    "response_id": self.current_response_id  # This is required
                }))
                logger.info(f"Cancelled response {self.current_response_id}")
                
                # Truncate the conversation item to synchronize the audio state
                # This ensures the model knows the user didn't hear the entire response
                if len(self.conversation_items) > 0:
                    last_item = self.conversation_items[-1]
                    if last_item.get("role") == "assistant" and last_item.get("id"):
                        try:
                            await self.ws.send_str(json.dumps({
                                "type": "conversation.item.truncate",
                                "item_id": last_item.get("id"),
                                "content_index": 0,  # First content part (audio)
                                "audio_end_ms": 0  # Truncate at current position
                            }))
                            logger.info(f"Truncated conversation item {last_item.get('id')}")
                        except Exception as e:
                            logger.error(f"Failed to truncate conversation item: {str(e)}")
                            if args.debug:
                                logger.debug(traceback.format_exc())
            except Exception as e:
                logger.error(f"Failed to cancel response: {str(e)}")
                if args.debug:
                    logger.debug(traceback.format_exc())
        
        # Mark that the agent is no longer speaking
        self.agent_speaking = False
        self.current_response_id = None
    
    async def close(self):
        """Close the call simulation"""
        self.conversation_active = False
        self.listening_for_user = False
        
        if self.ws:
            try:
                await self.ws.close()
            except:
                pass
            
        if hasattr(self, 'session') and not self.session.closed:
            try:
                await self.session.close()
            except:
                pass
            
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
        if args.debug:
            logger.debug(traceback.format_exc())
        print(f"\n❌ Error: {str(e)}")
    finally:
        if 'simulator' in locals():
            try:
                await simulator.close()
            except Exception as e:
                logger.error(f"Error during cleanup: {str(e)}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n👋 Call ended by user.")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}") 