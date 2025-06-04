"""
Tests for the call handler module
"""

import pytest
import json
import base64
from unittest.mock import AsyncMock, MagicMock, patch

from src.config import Settings
from src.call_handler import CallHandler
from src.openai_client import OpenAIRealtimeClient

@pytest.fixture
def settings():
    """Test settings fixture"""
    return Settings(
        voipstudio_api_key="test_voipstudio_key",
        openai_api_key="test_openai_key",
        system_instructions="You are a test assistant"
    )

@pytest.fixture
def mock_websocket():
    """Mock WebSocket fixture"""
    ws = AsyncMock()
    ws.receive_text = AsyncMock()
    ws.send_text = AsyncMock()
    ws.close = AsyncMock()
    return ws

@pytest.mark.asyncio
async def test_call_handler_initialization(settings):
    """Test call handler initialization"""
    handler = CallHandler(settings)
    
    assert handler.settings == settings
    assert isinstance(handler.openai_client, OpenAIRealtimeClient)
    assert handler.call_active is False
    assert handler.voipstudio_ws is None

@pytest.mark.asyncio
@patch('src.openai_client.OpenAIRealtimeClient.connect')
@patch('src.openai_client.OpenAIRealtimeClient.send_audio')
@patch('src.openai_client.OpenAIRealtimeClient.close')
async def test_handle_call_audio_flow(mock_close, mock_send_audio, mock_connect, settings, mock_websocket):
    """Test the audio flow in the call handler"""
    # Arrange
    handler = CallHandler(settings)
    
    # Mock WebSocket to return a media message and then a hangup
    media_payload = base64.b64encode(b"test audio data").decode('utf-8')
    mock_websocket.receive_text.side_effect = [
        json.dumps({
            "event": "media",
            "media": {
                "payload": media_payload
            }
        }),
        json.dumps({
            "event": "hangup"
        })
    ]
    
    # Act
    await handler.handle_call(mock_websocket)
    
    # Assert
    mock_connect.assert_called_once()
    mock_send_audio.assert_called_once()
    mock_close.assert_called_once()
    assert mock_websocket.send_text.call_count == 0  # No response sent in this test
    
@pytest.mark.asyncio
@patch('src.openai_client.OpenAIRealtimeClient.connect')
@patch('src.openai_client.OpenAIRealtimeClient.close')
async def test_handle_openai_message(mock_close, mock_connect, settings, mock_websocket):
    """Test handling OpenAI messages"""
    # Arrange
    handler = CallHandler(settings)
    handler.voipstudio_ws = mock_websocket
    handler.call_active = True
    
    # Capture the on_message_callback
    def mock_connect_impl(on_message_callback=None, on_close_callback=None):
        handler._on_message_callback = on_message_callback
        handler._on_close_callback = on_close_callback
        
    mock_connect.side_effect = mock_connect_impl
    
    # Mock WebSocket to return a hangup after we process the OpenAI message
    mock_websocket.receive_text.return_value = json.dumps({
        "event": "hangup"
    })
    
    # Start the handler in a task
    import asyncio
    task = asyncio.create_task(handler.handle_call(mock_websocket))
    
    # Wait for connect to be called
    await asyncio.sleep(0.1)
    
    # Simulate an OpenAI audio response
    openai_message = {
        "type": "response.audio.delta",
        "delta": "base64_audio_data"
    }
    await handler._on_message_callback(openai_message)
    
    # Wait for the message to be processed
    await asyncio.sleep(0.1)
    
    # Complete the task
    handler.call_active = False
    await task
    
    # Assert
    mock_websocket.send_text.assert_called_once()
    sent_data = json.loads(mock_websocket.send_text.call_args[0][0])
    assert sent_data["event"] == "media"
    assert sent_data["media"]["payload"] == "base64_audio_data" 