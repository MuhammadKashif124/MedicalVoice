# Testing the Medical Billing Voice Agent

This document provides instructions on how to test the MedicalVoice agent locally before deploying it with VoIPstudio.

## Prerequisites

Before testing, make sure you have:

1. Created a `.env` file with your API keys (see below)
2. Installed all dependencies with `pip install -r requirements.txt`
3. Python 3.8+ installed

## Environment Setup

Create a `.env` file in the root directory with the following content (replace with your actual API keys):

```
# API Keys
VOIPSTUDIO_API_KEY=your_voipstudio_api_key
OPENAI_API_KEY=your_openai_api_key

# Server configuration
PORT=5000
HOST=0.0.0.0

# OpenAI Realtime API configuration
OPENAI_MODEL=gpt-4o-realtime-preview-2024-10-01
OPENAI_VOICE=alloy
OPENAI_WEBSOCKET_URL=wss://api.openai.com/v1/realtime

# Audio format settings
INPUT_AUDIO_FORMAT=g711_ulaw
OUTPUT_AUDIO_FORMAT=g711_ulaw
```

## Testing Options

### 1. Testing with Text Input/Output

The simplest way to test the agent is using text-based interactions. This allows you to verify the medical billing knowledge and function calling capabilities without requiring VoIPstudio integration:

```bash
python test_agent.py
```

This script will:
- Run through a series of predefined test prompts
- Allow you to enter your own prompts in interactive mode
- Display function calls and their results in the terminal

### 2. Testing Function Calling Specifically

To focus on testing the function calling capabilities of the agent:

```bash
python test_function_calling.py
```

This script:
- Uses a simplified implementation focused on CPT code lookups
- Shows the full Realtime API interaction
- Demonstrates how functions are called and responses handled

### 3. Local Voice Call Simulator

For a complete voice interaction experience without needing to configure VoIPstudio:

```bash
python test_local_call.py
```

This simulates a real phone call using:
- Your computer's microphone for input
- Your speakers for audio output
- The same medical billing tools and functions as the real system

Key features:
- Voice-based interaction with the agent
- Automatic speech detection with silence recognition
- Function calling triggered by voice commands
- Complete audio and text responses
- No need for phone number configuration

Requirements:
- Working microphone
- Speakers/headphones
- PyAudio library (included in requirements.txt)
- Quiet environment for best results

### 4. Running the Full Server

To test the full FastAPI server (requires ngrok for external access):

```bash
# Start the server
python app.py

# In another terminal, expose it publicly (if you have ngrok installed)
ngrok http 5000
```

You can then:
- Test the health endpoint: http://localhost:5000/health
- Use the ngrok URL to configure VoIPstudio for call testing

## Troubleshooting

### Common Issues

1. **"XMLResponse" import error**
   - This is fixed in the latest code - we've created a custom XMLResponse class

2. **Cannot connect to OpenAI API**
   - Check that your API key is valid
   - Verify you have access to the Realtime API (it's in limited preview)

3. **Function calling not working**
   - Check the tools definition format
   - Ensure the function response is properly formatted

4. **WebSocket errors**
   - Ensure your dependencies are up to date
   - Check network connectivity

5. **Microphone issues in local call simulator**
   - Make sure your microphone is properly connected and set as default
   - You may need to adjust the SILENCE_THRESHOLD in test_local_call.py for your specific microphone

6. **Audio playback issues**
   - Check your speaker/headphone connections
   - Make sure volume is turned up
   - Try adjusting audio format settings if audio sounds distorted

### Logs

Check the application logs for detailed information about each request and response:

```bash
# Add DEBUG logging for more detailed information
export LOG_LEVEL=DEBUG
python app.py
```

## Additional Testing Ideas

- Test with different medical billing scenarios and edge cases
- Test handling of different insurance types
- Test CPT code lookups for both valid and invalid codes
- Test explanation of EOB statements and billing terminology 