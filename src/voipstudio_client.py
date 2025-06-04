"""
VoIPstudio client for handling API interactions
"""

import logging
import requests
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)

class VoIPstudioClient:
    """Client for interacting with VoIPstudio API"""
    
    def __init__(self, api_key: str, base_url: str = "https://api.voipstudio.com"):
        """
        Initialize the VoIPstudio client
        
        Args:
            api_key: VoIPstudio API key
            base_url: VoIPstudio API base URL
        """
        self.api_key = api_key
        self.base_url = base_url
        
    def _make_request(self, 
                     method: str, 
                     endpoint: str, 
                     data: Optional[Dict[str, Any]] = None, 
                     params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Make a request to the VoIPstudio API
        
        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint
            data: Request data
            params: Query parameters
            
        Returns:
            API response as a dictionary
        """
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        try:
            response = requests.request(
                method=method,
                url=url,
                headers=headers,
                json=data,
                params=params
            )
            
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error making request to VoIPstudio API: {str(e)}")
            raise
    
    def initiate_call(self, phone_number: str) -> Dict[str, Any]:
        """
        Initiate a call to a phone number
        
        Args:
            phone_number: Phone number to call
            
        Returns:
            Call details
        """
        logger.info(f"Initiating call to {phone_number}")
        
        data = {
            "phone_number": phone_number,
            "action": "originate"
        }
        
        return self._make_request("POST", "/v1/call", data=data)
    
    def get_call_details(self, call_id: str) -> Dict[str, Any]:
        """
        Get call details
        
        Args:
            call_id: Call ID
            
        Returns:
            Call details
        """
        logger.info(f"Getting details for call {call_id}")
        
        return self._make_request("GET", f"/v1/call/{call_id}")
    
    def list_calls(self, 
                  start_date: Optional[str] = None, 
                  end_date: Optional[str] = None, 
                  limit: int = 100) -> List[Dict[str, Any]]:
        """
        List calls
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            limit: Maximum number of calls to return
            
        Returns:
            List of calls
        """
        logger.info("Listing calls")
        
        params = {
            "limit": limit
        }
        
        if start_date:
            params["start_date"] = start_date
            
        if end_date:
            params["end_date"] = end_date
            
        return self._make_request("GET", "/v1/calls", params=params)
    
    def update_call(self, call_id: str, action: str) -> Dict[str, Any]:
        """
        Update a call
        
        Args:
            call_id: Call ID
            action: Action to perform (hangup, hold, etc.)
            
        Returns:
            Updated call details
        """
        logger.info(f"Updating call {call_id} with action {action}")
        
        data = {
            "action": action
        }
        
        return self._make_request("PUT", f"/v1/call/{call_id}", data=data) 