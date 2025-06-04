"""
Call handler for managing call interactions between VoIPstudio and OpenAI
"""

import json
import logging
import asyncio
from typing import Dict, Any, Optional, List
import base64

from fastapi import WebSocket, WebSocketDisconnect

from .config import Settings
from .openai_client import OpenAIRealtimeClient

logger = logging.getLogger(__name__)

class CallHandler:
    """Handler for call interactions between VoIPstudio and OpenAI"""
    
    def __init__(self, settings: Settings):
        """
        Initialize the call handler
        
        Args:
            settings: Application settings
        """
        self.settings = settings
        self.openai_client = OpenAIRealtimeClient(settings)
        self.call_active = False
        self.voipstudio_ws = None
        
        # Sample medical billing data (in a real app, this would come from a database)
        self.cpt_codes = {
            "99213": "Office or other outpatient visit for the evaluation and management of an established patient (15 minutes)",
            "99214": "Office or other outpatient visit for the evaluation and management of an established patient (25 minutes)",
            "99215": "Office or other outpatient visit for the evaluation and management of an established patient (40 minutes)",
            "70450": "Computed tomography, head or brain; without contrast material",
            "73610": "Radiologic examination, ankle; complete, minimum of 3 views",
            "80053": "Comprehensive metabolic panel"
        }
        
        self.claim_statuses = {
            "CL12345": {"status": "Approved", "payment_date": "2024-10-15", "amount_paid": "$520.75"},
            "CL67890": {"status": "Pending", "additional_info": "Awaiting provider documentation"},
            "CL24680": {"status": "Denied", "reason": "Service not covered under current plan"}
        }
        
        self.procedure_costs = {
            "Annual physical": {
                "Blue Cross": "$0 (covered 100%)", 
                "Medicare": "$0 with Part B", 
                "Uninsured": "$150-300"
            },
            "MRI": {
                "Blue Cross": "$250 (after deductible)", 
                "Medicare": "$170 with Part B", 
                "Uninsured": "$1,500-3,000"
            },
            "X-ray": {
                "Blue Cross": "$50 (after deductible)", 
                "Medicare": "$40 with Part B", 
                "Uninsured": "$150-400"
            }
        }
    
    async def handle_call(self, websocket: WebSocket):
        """
        Handle a call WebSocket connection
        
        Args:
            websocket: FastAPI WebSocket connection from VoIPstudio
        """
        self.voipstudio_ws = websocket
        self.call_active = True
        
        # Set up OpenAI message handler
        async def handle_openai_message(message: Dict[str, Any]):
            if not self.call_active:
                return
                
            message_type = message.get("type")
            
            if message_type == "response.audio.delta" and message.get("delta"):
                # Forward audio from OpenAI to VoIPstudio
                try:
                    audio_data = message.get("delta")
                    # Note: The audio data is already in base64 format from OpenAI
                    await self.voipstudio_ws.send_text(json.dumps({
                        "event": "media",
                        "media": {
                            "payload": audio_data
                        }
                    }))
                except Exception as e:
                    logger.error(f"Error sending audio to VoIPstudio: {str(e)}")
            
            elif message_type == "response.function_call_arguments.delta":
                # Handle function call
                try:
                    function_name = message.get("function_call", {}).get("name")
                    arguments = json.loads(message.get("delta", "{}"))
                    
                    result = await self._handle_function_call(function_name, arguments)
                    
                    # Send function call result back to OpenAI
                    await self.openai_client.send_function_result(
                        function_name=function_name,
                        result=result
                    )
                except Exception as e:
                    logger.error(f"Error handling function call: {str(e)}")
        
        # Set up OpenAI connection close handler
        async def handle_openai_close():
            logger.info("OpenAI connection closed, ending call")
            self.call_active = False
            
            try:
                await self.voipstudio_ws.close()
            except Exception as e:
                logger.error(f"Error closing VoIPstudio WebSocket: {str(e)}")
        
        # Connect to OpenAI with function definitions
        tools = self._get_tool_definitions()
        await self.openai_client.connect(
            on_message_callback=handle_openai_message,
            on_close_callback=handle_openai_close,
            tools=tools
        )
        
        try:
            # Process messages from VoIPstudio
            while self.call_active:
                try:
                    # Receive message from VoIPstudio
                    message = await self.voipstudio_ws.receive_text()
                    data = json.loads(message)
                    
                    # Process the message based on event type
                    event_type = data.get("event")
                    
                    if event_type == "media" and data.get("media", {}).get("payload"):
                        # Forward audio from VoIPstudio to OpenAI
                        audio_payload = data["media"]["payload"]
                        
                        # Convert base64 string to bytes
                        audio_bytes = base64.b64decode(audio_payload)
                        
                        # Send to OpenAI
                        await self.openai_client.send_audio(audio_bytes)
                    
                    elif event_type == "hangup":
                        # Call ended
                        logger.info("Received hangup event, ending call")
                        self.call_active = False
                        break
                        
                except WebSocketDisconnect:
                    logger.info("VoIPstudio WebSocket disconnected")
                    self.call_active = False
                    break
                    
                except Exception as e:
                    logger.error(f"Error processing VoIPstudio message: {str(e)}")
                    continue
                    
        finally:
            # Clean up
            self.call_active = False
            await self.openai_client.close()
            
            if self.voipstudio_ws:
                try:
                    await self.voipstudio_ws.close()
                except:
                    pass
    
    def _get_tool_definitions(self) -> List[Dict[str, Any]]:
        """Get the tool definitions for OpenAI function calling"""
        return [
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
            },
            {
                "type": "function",
                "function": {
                    "name": "check_claim_status",
                    "description": "Check the status of an insurance claim by claim number",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "claim_number": {
                                "type": "string",
                                "description": "The claim number to check (e.g., 'CL12345')"
                            }
                        },
                        "required": ["claim_number"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_procedure_cost",
                    "description": "Get the estimated cost of a medical procedure based on insurance type",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "procedure_name": {
                                "type": "string",
                                "description": "The name of the procedure (e.g., 'Annual physical', 'MRI', 'X-ray')"
                            },
                            "insurance_type": {
                                "type": "string",
                                "description": "The type of insurance (e.g., 'Blue Cross', 'Medicare', 'Uninsured')"
                            }
                        },
                        "required": ["procedure_name", "insurance_type"]
                    }
                }
            }
        ]
    
    async def _handle_function_call(self, function_name: str, arguments: Dict[str, Any]) -> str:
        """
        Handle a function call from OpenAI
        
        Args:
            function_name: The name of the function to call
            arguments: The arguments to pass to the function
            
        Returns:
            The result of the function call as a string
        """
        logger.info(f"Handling function call: {function_name} with arguments: {arguments}")
        
        if function_name == "lookup_cpt_code":
            code = arguments.get("code")
            description = self.cpt_codes.get(code)
            if description:
                return f"CPT Code {code}: {description}"
            else:
                return f"CPT Code {code} not found in database."
                
        elif function_name == "check_claim_status":
            claim_number = arguments.get("claim_number")
            claim_info = self.claim_statuses.get(claim_number)
            if claim_info:
                return json.dumps(claim_info)
            else:
                return f"Claim {claim_number} not found in system."
                
        elif function_name == "get_procedure_cost":
            procedure_name = arguments.get("procedure_name")
            insurance_type = arguments.get("insurance_type")
            
            procedure_costs = self.procedure_costs.get(procedure_name, {})
            cost = procedure_costs.get(insurance_type)
            
            if cost:
                return f"The estimated cost for {procedure_name} with {insurance_type} is {cost}."
            elif procedure_name in self.procedure_costs:
                available_insurances = list(self.procedure_costs[procedure_name].keys())
                return f"Cost information for {insurance_type} not available. Available insurance types: {', '.join(available_insurances)}"
            else:
                available_procedures = list(self.procedure_costs.keys())
                return f"Procedure {procedure_name} not found. Available procedures: {', '.join(available_procedures)}"
        
        return f"Function {function_name} not implemented." 