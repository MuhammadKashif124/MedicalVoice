#!/usr/bin/env python3
"""
Test script specifically focused on testing function calling with the OpenAI Realtime API
"""

import asyncio
import json
import os
import logging
import base64
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

# Sample medical billing data
CPT_CODES = {
    "99213": "Office or other outpatient visit for the evaluation and management of an established patient (15 minutes)",
    "99214": "Office or other outpatient visit for the evaluation and management of an established patient (25 minutes)",
    "70450": "Computed tomography, head or brain; without contrast material",
}

# Tool definitions
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "lookup_cpt_code",
            "description": "Look up the description of a CPT (Current Procedural Terminology) code",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "The CPT code to look up (e.g., '99213')"
                    }
                },
                "required": ["code"]
            }
        }
    }
]

# Function implementations
async def lookup_cpt_code(code: str) -> str:
    """Look up a CPT code and return its description"""
    if code in CPT_CODES:
        return f"CPT Code {code}: {CPT_CODES[code]}"
    else:
        return f"CPT Code {code} not found in database."

# Map of function names to implementations
FUNCTION_MAP = {
    "lookup_cpt_code": lookup_cpt_code
}

class RealtimeTester:
    """Class for testing the OpenAI Realtime API with function calling"""
    
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o-realtime-preview-2024-10-01")
        self.websocket_url = os.getenv("OPENAI_WEBSOCKET_URL", "wss://api.openai.com/v1/realtime")
        self.url = f"{self.websocket_url}?model={self.model}"
        self.ws = None
        self.active_function_calls = {}
        
    async def connect(self):
        """Connect to the OpenAI Realtime API"""
        import websockets
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "OpenAI-Beta": "realtime=v1"
        }
        
        logger.info(f"Connecting to {self.url}")
        self.ws = await websockets.connect(self.url, extra_headers=headers)
        logger.info("Connected to OpenAI Realtime API")
        
        # Update session with system instructions and tools
        await self.update_session()
        
    async def update_session(self):
        """Update the session with system instructions and tools"""
        session_config = {
            "type": "session.update",
            "session": {
                "instructions": """
                You are a helpful medical billing assistant. Your job is to help patients understand
                medical billing codes, insurance claims, and costs. Use the tools available to you
                to look up information when needed.
                """,
                "tools": TOOLS
            }
        }
        
        await self.ws.send(json.dumps(session_config))
        logger.info("Sent session configuration")
        
    async def send_message(self, text: str):
        """Send a user message to the API"""
        message = {
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
        
        await self.ws.send(json.dumps(message))
        logger.info(f"Sent message: {text}")
        
        # Request a response
        await self.ws.send(json.dumps({
            "type": "response.create",
            "response": {
                "modalities": ["text"]
            }
        }))
        
    async def handle_function_call(self, function_name: str, arguments: dict, call_id: str):
        """Handle a function call from the API"""
        logger.info(f"Handling function call: {function_name} with arguments: {arguments}")
        
        # Execute the function
        function = FUNCTION_MAP.get(function_name)
        if function:
            result = await function(**arguments)
        else:
            result = f"Function {function_name} not implemented."
            
        logger.info(f"Function result: {result}")
        
        # Send the result back to the API
        function_result = {
            "type": "conversation.item.create",
            "item": {
                "type": "function_call_output",
                "call_id": call_id,
                "output": result
            }
        }
        
        await self.ws.send(json.dumps(function_result))
        logger.info(f"Sent function result for {function_name}")
        
    async def listen(self):
        """Listen for messages from the API"""
        while True:
            try:
                message = await self.ws.recv()
                data = json.loads(message)
                message_type = data.get("type")
                
                if message_type == "response.text.delta":
                    # Text response
                    text_delta = data.get("delta", "")
                    print(text_delta, end="", flush=True)
                    
                elif message_type == "response.text.done":
                    print("\n")
                    
                elif message_type == "response.function_call_arguments.delta":
                    # Function call argument delta
                    function_name = data.get("function_call", {}).get("name")
                    logger.info(f"Function call delta: {function_name}")
                    
                elif message_type == "response.function_call_arguments.done":
                    # Function call complete
                    function_call = data.get("function_call", {})
                    function_name = function_call.get("name")
                    call_id = function_call.get("id")
                    arguments_json = function_call.get("arguments", "{}")
                    
                    try:
                        arguments = json.loads(arguments_json)
                        await self.handle_function_call(function_name, arguments, call_id)
                    except json.JSONDecodeError:
                        logger.error(f"Error parsing function arguments: {arguments_json}")
                        
                elif message_type == "error":
                    # Error
                    error = data.get("error", {})
                    logger.error(f"Error: {error.get('message')} (Code: {error.get('code')})")
                    
            except Exception as e:
                logger.error(f"Error in message handler: {str(e)}")
                break
                
    async def close(self):
        """Close the connection"""
        if self.ws:
            await self.ws.close()
            logger.info("Closed connection")

async def run_test():
    """Run the test"""
    tester = RealtimeTester()
    
    try:
        await tester.connect()
        
        # Start listening for messages in the background
        listen_task = asyncio.create_task(tester.listen())
        
        # Send a few test messages
        test_messages = [
            "Can you tell me what CPT code 99213 is for?",
            "I have a CPT code on my bill: 70450. What is that?",
            "What about code 12345? Is that a valid code?",
            "Can you explain what an EOB is?",
        ]
        
        for message in test_messages:
            print(f"\n--- PROMPT: {message} ---\n")
            await tester.send_message(message)
            await asyncio.sleep(8)  # Wait for response
            
        # Interactive mode
        print("\n--- Interactive Mode: Type your questions (or 'quit' to exit) ---\n")
        while True:
            user_input = input("\nYour question: ")
            if user_input.lower() in ('quit', 'exit', 'q'):
                break
                
            await tester.send_message(user_input)
            await asyncio.sleep(5)  # Wait for response
        
        # Cancel the listening task
        listen_task.cancel()
        
    except Exception as e:
        logger.error(f"Error in test: {str(e)}")
    finally:
        await tester.close()

if __name__ == "__main__":
    asyncio.run(run_test()) 