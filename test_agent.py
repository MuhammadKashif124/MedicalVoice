#!/usr/bin/env python3
"""
Test script for the Medical Billing Voice Agent using text input/output
This allows testing the functionality without a VoIPstudio integration
"""

import asyncio
import json
import os
import logging
from dotenv import load_dotenv

from src.config import Settings
from src.openai_client import OpenAIRealtimeClient
from src.call_handler import CallHandler

# Load environment variables and configure logging
load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Sample prompts to test the agent
TEST_PROMPTS = [
    "Can you explain what CPT code 99213 is for?",
    "What's the status of my insurance claim CL12345?",
    "How much would an MRI cost with Blue Cross insurance?",
    "What is an EOB statement?",
    "Can you explain the difference between copay and coinsurance?"
]

class MockWebSocket:
    """A mock WebSocket for testing the voice agent with text input/output"""
    
    def __init__(self):
        self.received_messages = []
    
    async def send_text(self, message):
        """Mock method to receive text from the agent"""
        data = json.loads(message)
        if data.get("event") == "media":
            # In a real implementation, this would be audio data
            # For testing, we'll just log that audio was received
            logger.info("Received audio response (not playing in text mode)")
        else:
            logger.info(f"Received message: {message}")
        
        self.received_messages.append(message)
    
    async def receive_text(self):
        """Mock method to send text to the agent"""
        # This would normally get data from the WebSocket
        # For testing, we'll just return an empty event since we're using text input
        return json.dumps({"event": "no_audio"})
    
    async def accept(self):
        """Mock method to accept the WebSocket connection"""
        logger.info("WebSocket connection accepted")
    
    async def close(self):
        """Mock method to close the WebSocket connection"""
        logger.info("WebSocket connection closed")

async def send_text_to_agent(client, text):
    """Send text input to the agent instead of audio"""
    logger.info(f"Sending text: {text}")
    
    # Create a text message to send to OpenAI
    text_message = {
        "type": "conversation.item.create",
        "item": {
            "type": "message",
            "role": "user",
            "content": [{
                "type": "input_text",
                "text": text
            }]
        }
    }
    
    # Send the text message
    client.ws.send(json.dumps(text_message))
    
    # Tell the model to generate a response
    response_message = {
        "type": "response.create",
        "response": {
            "modalities": ["text"]  # We only want text back for testing
        }
    }
    
    client.ws.send(json.dumps(response_message))

async def process_message(message):
    """Process a message from OpenAI"""
    data = json.loads(message)
    message_type = data.get("type")
    
    if message_type == "response.text.delta":
        # Text response from the model
        text_delta = data.get("delta", "")
        print(text_delta, end="", flush=True)
    elif message_type == "response.text.done":
        print("\n")
    elif message_type == "response.function_call_arguments.delta":
        # Function call
        function_name = data.get("function_call", {}).get("name")
        logger.info(f"Function call: {function_name}")

async def main():
    """Main test function"""
    try:
        # Initialize settings
        settings = Settings()
        
        # Initialize OpenAI client directly (bypassing call handler for simpler testing)
        client = OpenAIRealtimeClient(settings)
        
        # Define message handler to print text responses
        def message_handler(message):
            asyncio.create_task(process_message(message))
        
        # Initialize the call handler with tools
        call_handler = CallHandler(settings)
        tools = call_handler._get_tool_definitions()
        
        # Connect to OpenAI
        await client.connect(
            on_message_callback=message_handler,
            tools=tools
        )
        
        # Wait for connection to be established
        await asyncio.sleep(1)
        
        # Process each test prompt
        for i, prompt in enumerate(TEST_PROMPTS):
            print(f"\n--- Test Prompt {i+1}: {prompt} ---")
            await send_text_to_agent(client, prompt)
            
            # Wait for response to complete
            await asyncio.sleep(5)
            
            # Ask if user wants to continue
            if i < len(TEST_PROMPTS) - 1:
                user_input = input("\nPress Enter for next prompt or 'q' to quit: ")
                if user_input.lower() == 'q':
                    break
        
        # Test interactive mode
        print("\n--- Interactive Mode: Type your questions (or 'quit' to exit) ---")
        while True:
            user_input = input("\nYour question: ")
            if user_input.lower() in ('quit', 'exit', 'q'):
                break
                
            await send_text_to_agent(client, user_input)
            await asyncio.sleep(5)  # Wait for response
            
    except Exception as e:
        logger.error(f"Error in test: {str(e)}")
    finally:
        # Close the connection
        if 'client' in locals() and client.connected:
            await client.close()

if __name__ == "__main__":
    asyncio.run(main()) 