import os
import logging
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

def transcribe_with_groq(GROQ_API_KEY, audio_filepath, stt_model, language="en"):
    """
    Transcribe audio file to text using Groq Whisper API
    
    Args:
        GROQ_API_KEY: Groq API key
        audio_filepath: Path to the audio file
        stt_model: Speech-to-text model name (e.g., "whisper-large-v3")
        language: Language code for transcription ("en" for English, "hi" for Hindi)
    
    Returns:
        str: Transcribed text
    """
    client = Groq(api_key=GROQ_API_KEY)
    
    try:
        audio_file = open(audio_filepath, "rb")
        transcription = client.audio.transcriptions.create(
            model=stt_model,
            file=audio_file,
            language=language  # Supports "en", "hi", and many other languages
        )
        audio_file.close()
        return transcription.text
    except Exception as e:
        logging.error(f"Transcription error: {str(e)}")
        return f"Error transcribing audio: {str(e)}"