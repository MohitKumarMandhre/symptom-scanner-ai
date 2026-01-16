# 🩺 AI Medical Assistant - Symptom Scanner

An AI-powered medical consultation system that provides preliminary health assessments using modern medicine (Allopathy), Homeopathy, and Ayurveda principles. The application accepts multimodal inputs including images, voice recordings, and text descriptions.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![License](https://img.shields.io/badge/License-Educational-green.svg)

---

## 📑 Table of Contents

- [Features](#-features)
- [Architecture](#-architecture)
- [High-Level Design (HLD)](#-high-level-design-hld)
- [Module Documentation](#-module-documentation)
- [Installation](#-installation)
- [Dependencies](#-dependencies)
- [Configuration](#-configuration)
- [Usage](#-usage)
- [API Reference](#-api-reference)
- [Limitations](#-limitations)
- [Future Enhancements](#-future-enhancements)
- [Contributing](#-contributing)
- [Disclaimer](#-disclaimer)

---

## ✨ Features

### Core Features
| Feature | Description |
|---------|-------------|
| 🏥 **Multi-Specialty Consultations** | Support for Allopathy, Homeopathy, and Ayurveda medical systems |
| 📷 **Image Analysis** | Upload medical images for AI-powered visual diagnosis |
| 🎤 **Voice Input** | Record symptoms verbally with speech-to-text transcription |
| ✍️ **Text Input** | Type symptoms directly for quick consultations |
| 🔊 **Voice Response** | AI doctor responses converted to speech for accessibility |
| 📥 **Report Generation** | Download consultation reports in text format |

### Technical Features
- **Multimodal Input Processing**: Combines image, voice, and text inputs
- **Real-time Audio Recording**: Browser-based audio capture
- **Asynchronous Processing**: Non-blocking API calls
- **Session State Management**: Persistent user session handling
- **Responsive UI**: Mobile-friendly Streamlit interface

---

## 🏗 Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        PRESENTATION LAYER                       │
│                     (streamlit_app.py)                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
│  │   Image     │  │   Voice     │  │    Text     │              │
│  │   Upload    │  │   Recorder  │  │   Input     │              │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘              │
└─────────┼────────────────┼────────────────┼─────────────────────┘
          │                │                │
          ▼                ▼                ▼
┌─────────────────────────────────────────────────────────────────┐
│                        PROCESSING LAYER                         │
│  ┌──────────────────┐  ┌──────────────────────────────────┐     │
│  │ voice_of_patient │  │      brain_of_the_doctor         │     │
│  │   (STT Module)   │  │    (AI Analysis Engine)          │     │
│  │                  │  │                                  │     │
│  │ • Audio Recording│  │ • Image Encoding                 │     │
│  │ • Transcription  │  │ • Multimodal Analysis            │     │
│  │ • Groq Whisper   │  │ • Groq LLaMA Integration         │     │
│  └────────┬─────────┘  └────────────┬─────────────────────┘     │
└───────────┼─────────────────────────┼───────────────────────────┘
            │                         │
            ▼                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                         OUTPUT LAYER                            │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              voice_of_the_doctor (TTS Module)            │   │
│  │                                                          │   │
│  │  • Text-to-Speech Conversion (gTTS)                      │   │
│  │  • Audio File Generation                                 │   │
│  │  • Response Delivery                                     │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
            │
            ▼
┌─────────────────────────────────────────────────────────────────┐
│                      EXTERNAL SERVICES                          │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐     │
│  │   Groq API     │  │   gTTS API     │  │  File System   │     │
│  │  (LLM + STT)   │  │    (TTS)       │  │  (temp_docs/)  │     │
│  └────────────────┘  └────────────────┘  └────────────────┘     │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📐 High-Level Design (HLD)

### System Flow Diagram

```
┌──────────────────────────────────────────────────────────────────────────┐
│                           USER INTERACTION FLOW                          │
└──────────────────────────────────────────────────────────────────────────┘

    ┌─────────┐
    │  User   │
    └────┬────┘
         │
         ▼
┌────────────────────┐
│ 1. Select Doctor   │ ──► Allopathy / Homeopathy / Ayurveda
│    Type            │
└────────┬───────────┘
         │
         ▼
┌────────────────────┐     ┌─────────────────────────────────────┐
│ 2. Provide Input   │ ──► │ At least ONE required:              │
│    (Multimodal)    │     │ • Image (PNG/JPG) - Optional        │
└────────┬───────────┘     │ • Voice Recording - Optional        │
         │                 │ • Text Description - Optional       │
         │                 └─────────────────────────────────────┘
         ▼
┌────────────────────┐
│ 3. Process Inputs  │
│                    │
│ ┌────────────────┐ │
│ │ Voice ──► STT  │ │ ──► Groq Whisper API
│ └────────────────┘ │
│ ┌────────────────┐ │
│ │ Image ──► B64  │ │ ──► Base64 Encoding
│ └────────────────┘ │
│ ┌────────────────┐ │
│ │ Text ──► Pass  │ │ ──► Direct Processing
│ └────────────────┘ │
└────────┬───────────┘
         │
         ▼
┌────────────────────┐
│ 4. Combine Inputs  │ ──► Structured Prompt Generation
│    + System Prompt │     (Doctor-specific prompts)
└────────┬───────────┘
         │
         ▼
┌────────────────────┐
│ 5. AI Analysis     │ ──► Groq LLaMA-4-Scout API
│    (LLM Call)      │     (Vision + Text Model)
└────────┬───────────┘
         │
         ▼
┌────────────────────┐
│ 6. Generate Voice  │ ──► Google Text-to-Speech (gTTS)
│    Response        │
└────────┬───────────┘
         │
         ▼
┌────────────────────┐
│ 7. Display Results │
│ • Text Response    │
│ • Audio Playback   │
│ • Download Report  │
└────────────────────┘
```

### Data Flow Specification

| Stage | Input | Process | Output | Module |
|-------|-------|---------|--------|--------|
| 1 | User Selection | Store doctor type | Session state update | `streamlit_app.py` |
| 2a | Image file | Save to disk | File path | `streamlit_app.py` |
| 2b | Audio stream | Record & save | WAV/MP3 file | `voice_of_the_patient.py` |
| 2c | Text string | Store in state | Session variable | `streamlit_app.py` |
| 3 | Audio file | Whisper STT | Transcription text | `voice_of_the_patient.py` |
| 4 | Image path | Base64 encode | Encoded string | `brain_of_the_doctor.py` |
| 5 | Combined prompt | LLM inference | Medical response | `brain_of_the_doctor.py` |
| 6 | Response text | TTS conversion | MP3 audio file | `voice_of_the_doctor.py` |
| 7 | All outputs | UI rendering | User display | `streamlit_app.py` |

---

## 📦 Module Documentation

### 1. `streamlit_app.py` - Main Application Controller

**Purpose**: Orchestrates the entire application flow, handles UI rendering, and manages user sessions.

```python
# Key Components

DOCTOR_PROMPTS = {
    "allopathy": {...},    # Modern medicine prompts
    "homeopathy": {...},   # Homeopathic prompts
    "ayurveda": {...}      # Ayurvedic prompts
}

# Session State Variables
- recorded_audio      # Stored audio bytes
- audio_saved         # Audio save status flag
- analysis_done       # Analysis completion flag
- results             # Consultation results dictionary
- uploaded_image_data # Image binary data
- image_saved         # Image save status flag
- selected_doctor     # Current doctor type
- text_symptoms       # User text input
- text_saved          # Text input status flag
```

**Function Scope**:

| Function/Section | Responsibility |
|-----------------|----------------|
| Doctor Selection UI | Render 3-column doctor type buttons |
| Image Upload Handler | Process and save uploaded images |
| Audio Recorder Integration | Capture and store voice recordings |
| Text Input Handler | Manage symptom text area |
| Analysis Pipeline | Coordinate STT → LLM → TTS flow |
| Results Display | Render consultation results |
| Report Generator | Create downloadable text reports |

---

### 2. `brain_of_the_doctor.py` - AI Analysis Engine

**Purpose**: Handles image encoding and multimodal AI analysis using Groq API.

```python
def encode_image(image_path: str) -> str | None:
    """
    Encode image file to base64 string for API transmission.
    
    Args:
        image_path: Filesystem path to image file
        
    Returns:
        Base64 encoded string or None if file doesn't exist
        
    Complexity: O(n) where n = file size
    """

def analyze_image_with_query(
    query: str, 
    encoded_image: str | None, 
    model: str
) -> str:
    """
    Perform multimodal analysis combining text query with optional image.
    
    Args:
        query: System prompt + user symptoms
        encoded_image: Base64 image or None for text-only
        model: Groq model identifier
        
    Returns:
        AI-generated medical assessment string
        
    API Calls:
        - Primary: meta-llama/llama-4-scout-17b-16e-instruct (vision)
        - Fallback: llama-3.3-70b-versatile (text-only)
        
    Error Handling:
        - Vision model failure → text model fallback
        - Complete failure → error message return
    """
```

**Processing Logic**:

```
Input Received
      │
      ▼
┌─────────────────┐
│ Check if image  │
│ is provided     │
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
    ▼         ▼
[Image+Text] [Text Only]
    │         │
    ▼         ▼
Build multi- Build simple
part message text message
    │         │
    └────┬────┘
         │
         ▼
┌─────────────────┐
│ Call Groq API   │
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
[Success]  [Error]
    │         │
    ▼         ▼
Return    Try fallback
response  text model
```

---

### 3. `voice_of_the_patient.py` - Speech-to-Text Module

**Purpose**: Records audio from microphone and transcribes speech using Groq Whisper.

```python
def record_audio(
    file_path: str, 
    timeout: int = 20, 
    phrase_time_limit: int | None = None
) -> None:
    """
    Record audio from system microphone and save as MP3.
    
    Args:
        file_path: Output file path for recording
        timeout: Max wait time for speech to start (seconds)
        phrase_time_limit: Max recording duration (seconds)
        
    Process:
        1. Initialize speech recognizer
        2. Calibrate for ambient noise (1 second)
        3. Listen for speech input
        4. Convert WAV → MP3 (128k bitrate)
        5. Save to specified path
        
    Dependencies:
        - speech_recognition (PyAudio backend)
        - pydub (FFmpeg for conversion)
        
    Exceptions:
        - Logs errors without raising (fault-tolerant)
    """

def transcribe_with_groq(
    stt_model: str, 
    audio_filepath: str, 
    GROQ_API_KEY: str
) -> str:
    """
    Transcribe audio file to text using Groq Whisper API.
    
    Args:
        stt_model: Model identifier (whisper-large-v3)
        audio_filepath: Path to audio file
        GROQ_API_KEY: API authentication key
        
    Returns:
        Transcribed text string
        
    API: Groq audio.transcriptions.create()
    Language: English (hardcoded)
    """
```

---

### 4. `voice_of_the_doctor.py` - Text-to-Speech Module

**Purpose**: Converts AI text responses to spoken audio using Google TTS.

```python
def text_to_speech_with_gtts(
    input_text: str, 
    output_filepath: str
) -> None:
    """
    Convert text to speech and save as audio file.
    
    Args:
        input_text: Text to convert to speech
        output_filepath: Output MP3 file path
        
    Configuration:
        - Language: English ('en')
        - Speed: Normal (slow=False)
        
    Output Format: MP3
    
    Note: Commented code includes cross-platform 
          audio playback (macOS/Windows/Linux)
    """
```

---

## 🚀 Installation

### Prerequisites

- Python 3.9 or higher
- FFmpeg installed and in system PATH
- PortAudio library (for microphone access)
- Groq API key

---

## 📚 Dependencies

### Python Packages

Refer `requirements.txt` file

### System Dependencies

| Dependency | Purpose | Installation |
|------------|---------|--------------|
| **FFmpeg** | Audio format conversion | `choco install ffmpeg` / `brew install ffmpeg` |
| **PortAudio** | Microphone access | `brew install portaudio` / `apt-get install portaudio19-dev` |
| **PyAudio** | Python audio interface | `pip install pyaudio` (may need wheel on Windows) |

### External APIs

| Service | Purpose | Model Used |
|---------|---------|------------|
| **Groq** | LLM & STT | `llama-4-scout-17b-16e-instruct`, `whisper-large-v3` |
| **Google TTS** | Text-to-Speech | gTTS default voice |

---

## ⚙️ Configuration

### Environment Variables (`.env`)

```env
# Required
GROQ_API_KEY=your_groq_api_key_here

# File Paths (configurable)
OUTPUT_AUDIO_PATH=temp_docs/doctor_response.mp3
PATIENT_AUDIO_PATH=temp_docs/patient_audio.wav
PATIENT_IMAGE_PATH=temp_docs/patient_image.png
```

### Directory Structure

```
symptom-scanner-ai/
├── .env                      # Environment configuration
├── streamlit_app.py          # Main application
├── brain_of_the_doctor.py    # AI analysis module
├── voice_of_the_patient.py   # STT module
├── voice_of_the_doctor.py    # TTS module
├── requirements.txt          # Python dependencies
├── README.md                 # Documentation
└── temp_docs/                # Temporary file storage
    ├── patient_image.png     # Uploaded images
    ├── patient_audio.wav     # Recorded audio
    └── doctor_response.mp3   # Generated speech
```

---

## 📖 Usage

### Basic Workflow

1. **Launch Application**
   ```bash
   streamlit run streamlit_app.py
   ```

2. **Select Consultation Type**
   - Click on Allopathic, Homeopathic, or Ayurvedic doctor

3. **Provide Symptoms** (at least one)
   - Upload an image of the affected area
   - Record voice describing symptoms
   - Type symptoms in text box

4. **Get Consultation**
   - Click "Get [Specialty] Consultation" button
   - Wait for AI processing

5. **Review Results**
   - Read the AI doctor's assessment
   - Listen to voice response
   - Download report if needed

---

## 🚧 Limitations

### Technical Limitations

| Limitation | Description | Impact |
|------------|-------------|--------|
| **Single Language** | Only English supported | Non-English users excluded |
| **No Real-time Streaming** | Batch processing only | Delayed responses |
| **File-based Audio** | No direct stream processing | Additional I/O overhead |
| **Session Volatility** | State lost on refresh | No persistence |
| **Single User Design** | No multi-tenancy | Not scalable as-is |

### Medical Limitations

| Limitation | Description |
|------------|-------------|
| **No Diagnostic Authority** | Cannot provide actual medical diagnoses |
| **Limited Visual Analysis** | AI may miss subtle image details |
| **No Medical History** | Cannot consider patient history |
| **No Drug Interactions** | Cannot check medication conflicts |
| **No Emergency Detection** | May not identify life-threatening conditions |

### API Limitations

- **Groq Rate Limits**: Subject to API quotas
- **gTTS Quality**: Basic voice synthesis quality
- **Model Hallucinations**: AI may generate incorrect information

---

## 🔮 Future Enhancements

### Short-term (v2.1)

- [ ] **Multi-language Support**: Add language selection for STT/TTS
- [ ] **Conversation History**: Implement chat-like follow-up questions
- [ ] **Image Annotation**: Highlight detected areas in images
- [ ] **Voice Selection**: Multiple TTS voice options

### Medium-term (v3.0)

- [ ] **User Authentication**: Login system with patient profiles
- [ ] **Medical History Integration**: Store and reference past consultations
- [ ] **Appointment Booking**: Connect with real healthcare providers
- [ ] **Symptom Checker Database**: Structured symptom input with autocomplete
- [ ] **Drug Database Integration**: Medication information and interactions

### Long-term (v4.0)

- [ ] **Real-time Video Consultation**: Live video with AI assistance
- [ ] **Wearable Integration**: Import data from health devices
- [ ] **Electronic Health Records (EHR)**: FHIR-compliant data exchange
- [ ] **Multi-specialist Consultation**: Combine opinions from multiple AI doctors
- [ ] **Federated Learning**: Privacy-preserving model improvements

### Infrastructure Improvements

- [ ] **Containerization**: Docker deployment
- [ ] **Cloud Deployment**: AWS/GCP/Azure hosting
- [ ] **API Gateway**: Rate limiting and authentication
- [ ] **Monitoring**: Prometheus/Grafana observability
- [ ] **CI/CD Pipeline**: Automated testing and deployment

---

## 🤝 Contributing

### Development Setup

```bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Code formatting
black .
isort .

# Linting
flake8 .
mypy .
```

### Contribution Guidelines

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

---

## ⚠️ Disclaimer

```
╔══════════════════════════════════════════════════════════════════╗
║                    IMPORTANT MEDICAL DISCLAIMER                   ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                    ║
║  This AI Medical Assistant is for EDUCATIONAL and INFORMATIONAL   ║
║  purposes ONLY. It is NOT a substitute for professional medical   ║
║  advice, diagnosis, or treatment.                                 ║
║                                                                    ║
║  • Do NOT use this application for emergency situations           ║
║  • Always consult qualified healthcare professionals              ║
║  • Do NOT make medical decisions based solely on AI output        ║
║  • The developers are NOT liable for any health outcomes          ║
║                                                                    ║
║  If you are experiencing a medical emergency, call your local     ║
║  emergency services immediately.                                  ║
║                                                                    ║
╚══════════════════════════════════════════════════════════════════╝
```

---

## 📄 License

This project is licensed for **Educational Purposes Only**.

---

## 📞 Contact

- **Project Maintainer**: MkM
- **Repository**: [GitHub Link]
- **Issues**: [GitHub Issues]

---

<div align="center">

**Built with ❤️ for Healthcare Education**

🏥 Allopathy • 🌿 Homeopathy • 🪷 Ayurveda

</div>