# MedicalVoice

A custom AI call agent using VoIPstudio and OpenAI's Realtime API for medical billing purposes.

## Features

- Voice interaction with AI assistant specialized in medical billing
- Integration with VoIPstudio for real telephone calls
- Function calling capabilities for:
  - CPT code lookups
  - Insurance claim status checks
  - Procedure cost estimation
- Professional, empathetic responses with clear medical terminology explanations

## Setup

1. Clone this repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Create a `.env` file with your API keys:
   ```
   VOIPSTUDIO_API_KEY=your_voipstudio_api_key
   OPENAI_API_KEY=your_openai_api_key
   ```

## Testing Options

### 1. Text-based Testing

Test the medical billing knowledge and function calling with text input/output:

```bash
python test_agent.py
```

### 2. Function Calling Test

Test specifically the function calling capabilities:

```bash
python test_function_calling.py
```

### 3. Local Voice Call Simulator

Test the full voice conversation experience using your microphone and speakers:

```bash
python test_local_call.py
```

This simulates a complete phone call without needing to configure VoIPstudio:
- Speak into your microphone to talk to the agent
- Hear the agent's responses through your speakers
- Experience the complete voice interaction flow
- Test function calling through voice commands
- Press Ctrl+C to end the call

### 4. Production Deployment

For full telephone integration:

1. Start the server:
   ```bash
   python app.py
   ```

2. Expose publicly with ngrok:
   ```bash
   ngrok http 5000
   ```

3. Configure VoIPstudio to use your ngrok URL

## Documentation

For more detailed testing instructions, see [TESTING.md](TESTING.md).

## Project Structure

```
MedicalVoice/
├── app.py                  # Main application entry point
├── requirements.txt        # Project dependencies
├── .env                    # Environment variables (create this file)
├── .gitignore              # Git ignore file
├── README.md               # Project documentation
├── src/
│   ├── __init__.py
│   ├── config.py           # Configuration settings
│   ├── openai_client.py    # OpenAI API integration
│   ├── voipstudio_client.py # VoIPstudio API integration
│   └── call_handler.py     # Call processing logic
└── tests/
    ├── __init__.py
    └── test_call_handler.py # Unit tests
```

## Configuration

Adjust the AI behavior by modifying the system prompts in `src/config.py`.

## License

MIT