import os
from gtts import gTTS
from dotenv import load_dotenv

load_dotenv()

def text_to_speech_with_gtts(input_text, output_filepath, language="en"):
    """
    Convert text to speech using Google Text-to-Speech
    
    Args:
        input_text: Text to convert to speech
        output_filepath: Path to save the audio file
        language: Language code ("en" for English, "hi" for Hindi)
    
    Returns:
        str: Path to the saved audio file or None on error
    """
    try:
        # gTTS supports many languages including:
        # 'en' - English
        # 'hi' - Hindi
        # 'ta' - Tamil
        # 'te' - Telugu
        # 'mr' - Marathi
        # 'bn' - Bengali
        # etc.
        
        tts = gTTS(text=input_text, lang=language, slow=False)
        tts.save(output_filepath)
        
        return output_filepath
    except Exception as e:
        print(f"Error generating speech: {str(e)}")
        # Fallback to English if Hindi fails
        if language != "en":
            try:
                tts = gTTS(text=input_text, lang="en", slow=False)
                tts.save(output_filepath)
                return output_filepath
            except Exception as e2:
                print(f"Fallback to English also failed: {str(e2)}")
        return None