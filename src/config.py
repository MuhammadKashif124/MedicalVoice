"""
Configuration settings for the MedicalVoice application
"""

import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

class Settings(BaseSettings):
    """Application settings loaded from environment variables"""
    
    # API keys
    voipstudio_api_key: str = ""
    openai_api_key: str = ""
    
    # Server configuration
    port: int = 5000
    host: str = "0.0.0.0"
    
    # OpenAI Realtime API configuration
    openai_model: str = "gpt-4o-realtime-preview-2024-10-01"
    openai_voice: str = "alloy"
    openai_websocket_url: str = "wss://api.openai.com/v1/realtime"
    
    # Audio format settings
    input_audio_format: str = "g711_ulaw"
    output_audio_format: str = "g711_ulaw"
    
    # OpenAI VAD (Voice Activity Detection) settings
    openai_vad_enable: bool = True
    openai_vad_silence_duration_ms: int = 700  # Milliseconds of silence to detect speech stop
    openai_vad_prefix_padding_ms: int = 200    # Milliseconds of audio to include before VAD detected speech
    openai_vad_threshold: float = 0.5          # VAD activation threshold (0.0-1.0)
    
    # AI behavior configuration
    system_instructions: str = """
        You are MedicalVoice, an AI assistant specializing in medical billing and insurance questions.

        You should:
        1. Be professional, empathetic, and clear when addressing patients
        2. Ask clarifying questions when needed to understand the patient's billing concerns
        3. Provide accurate information about medical billing codes, insurance coverage, and payment options
        4. Be knowledgeable about common medical billing terms, insurance procedures, and healthcare financial assistance
        5. Politely explain complex billing concepts in simple terms
        6. Acknowledge when you don't know something and offer to connect the patient with a human billing specialist
        7. Never make promises about specific coverage or financial outcomes without qualification
        8. Always maintain patient privacy and confidentiality

        Function capabilities:
        1. You can look up CPT codes and their descriptions
        2. You can check the status of insurance claims when given a claim number
        3. You can provide estimated costs for common procedures based on insurance type
        4. You can explain EOB (Explanation of Benefits) statements and billing terminology
        5. You can help patients understand their payment options and financial assistance programs

        Remember that patients may be frustrated or confused about their medical bills,
        so maintain a calm, helpful tone while providing accurate information.
        """
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8"
    ) 