#Step1: Setup Text to Speech–TTS–model with gTTS

import os
from gtts import gTTS

def text_to_speech_with_gtts_old(input_text, output_filepath):
    language="en"

    audioobj= gTTS(
        text=input_text,
        lang=language,
        slow=False
    )
    audioobj.save(output_filepath)

# import elevenlabs
# from elevenlabs.client import ElevenLabs

# ELEVENLABS_API_KEY=os.environ.get("ELEVEN_API_KEY")

# def text_to_speech_with_elevenlabs_old(input_text, output_filepath):
#     client=ElevenLabs(api_key=ELEVENLABS_API_KEY)
#     audio=client.generate(
#         text= input_text,
#         voice= "Aria",
#         output_format= "mp3_22050_32",
#         model= "eleven_turbo_v2"
#     )
#     elevenlabs.save(audio, output_filepath)

#text_to_speech_with_elevenlabs_old(input_text, output_filepath="elevenlabs_testing.mp3") 


# input_text="Hi this is MkM!"
# text_to_speech_with_gtts_old(input_text=input_text, output_filepath="temp_docs/gtts_testing.mp3")

#Step2: use models for text output to voice

import subprocess
import platform

def text_to_speech_with_gtts(input_text, output_filepath):
    language="en"

    audioobj= gTTS(
        text=input_text,
        lang=language,
        slow=False
    )
    audioobj.save(output_filepath)
    # os_name = platform.system()
    # try:
    #     if os_name == "Darwin":  # macOS
    #         subprocess.run(['afplay', output_filepath])
    #     elif os_name == "Windows":  # Windows
    #         subprocess.run(['powershell', '-c', f'Add-Type -AssemblyName presentationCore; $player = New-Object System.Windows.Media.MediaPlayer; $player.Open([System.Uri]::new((Resolve-Path "{output_filepath}"))); $player.Play(); Start-Sleep -Seconds 10; $player.Close()'])
    #     elif os_name == "Linux":  # Linux
    #         subprocess.run(['aplay', output_filepath])  # Alternative: use 'mpg123' or 'ffplay'
    #     else:
    #         raise OSError("Unsupported operating system")
    # except Exception as e:
    #     print(f"An error occurred while trying to play the audio: {e}")



# input_text="Hi this is Ai with MKM, autoplay testing!"
# text_to_speech_with_gtts(input_text=input_text, output_filepath="temp_docs/gtts_testing_autoplay.mp3")