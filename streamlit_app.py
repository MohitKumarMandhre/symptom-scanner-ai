import os
import streamlit as st
from dotenv import load_dotenv
from audio_recorder_streamlit import audio_recorder
from datetime import datetime

from brain_of_the_doctor import encode_image, analyze_image_with_query
from voice_of_the_patient import transcribe_with_groq
from voice_of_the_doctor import text_to_speech_with_gtts

load_dotenv()

# Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OUTPUT_AUDIO_PATH = os.getenv("OUTPUT_AUDIO_PATH")
PATIENT_AUDIO_PATH = os.getenv("PATIENT_AUDIO_PATH")
PATIENT_IMAGE_PATH = os.getenv("PATIENT_IMAGE_PATH")

# Ensure output directory exists
os.makedirs("temp_docs", exist_ok=True)

# Doctor type prompts - Updated for flexible input
DOCTOR_PROMPTS = {
    "allopathy": {
        "name": "Allopathic Doctor (Modern Medicine)",
        "icon": "👨‍⚕️",
        "specialty": "Modern Medicine",
        "prompt_with_image": """You have to act as an experienced Allopathic (Modern Medicine) Doctor. 
            You follow evidence-based medicine and may suggest conventional treatments, medications, and diagnostic tests.
            What's in this image? Do you find anything wrong with it medically? 
            If you make a differential diagnosis, suggest some remedies including:
            - Over-the-counter or prescription medications if needed
            - Lifestyle modifications
            - When to seek emergency care
            Do not add any numbers or special characters in your response. 
            Your response should be in one long paragraph. Answer as if you are talking to a real patient.
            Don't say 'In the image I see' but say 'With what I see, I think you have ....'
            Don't respond as an AI model in markdown, your answer should mimic that of an actual doctor.
            Keep your answer concise (max 2-3 sentences). No preamble, start your answer right away.
            Always end with a positive and reassuring note.
            
            Patient's described symptoms: """,
        "prompt_text_only": """You have to act as an experienced Allopathic (Modern Medicine) Doctor. 
            You follow evidence-based medicine and may suggest conventional treatments, medications, and diagnostic tests.
            Based on the patient's described symptoms, provide your medical assessment including:
            - Possible conditions based on symptoms
            - Over-the-counter or prescription medications if needed
            - Lifestyle modifications
            - When to seek emergency care
            Do not add any numbers or special characters in your response. 
            Your response should be in one long paragraph. Answer as if you are talking to a real patient.
            Start with 'Based on your symptoms, I think you might have ....'
            Don't respond as an AI model in markdown, your answer should mimic that of an actual doctor.
            Keep your answer concise (max 2-3 sentences). No preamble, start your answer right away.
            Always end with a positive and reassuring note.
            
            Patient's described symptoms: """
    },
    "homeopathy": {
        "name": "Homeopathic Doctor",
        "icon": "🌿",
        "specialty": "Homeopathy",
        "prompt_with_image": """You have to act as an experienced Homeopathic Doctor following the principles of Samuel Hahnemann.
            You believe in 'like cures like' and use highly diluted natural substances for treatment.
            What's in this image? Do you find anything wrong with it from a homeopathic perspective?
            If you identify any condition, suggest some remedies including:
            - Homeopathic medicines with their potency (like Arnica 30C, Belladonna 200C, etc.)
            - Constitutional remedies based on symptoms
            - Dietary and lifestyle recommendations from homeopathic perspective
            Do not add any numbers or special characters in your response.
            Your response should be in one long paragraph. Answer as if you are talking to a real patient.
            Don't say 'In the image I see' but say 'With what I see, based on homeopathic principles, I think you have ....'
            Don't respond as an AI model in markdown, your answer should mimic that of an actual homeopathic practitioner.
            Keep your answer concise (max 2-3 sentences). No preamble, start your answer right away.
            Always end with a positive and holistic healing note.
            
            Patient's described symptoms: """,
        "prompt_text_only": """You have to act as an experienced Homeopathic Doctor following the principles of Samuel Hahnemann.
            You believe in 'like cures like' and use highly diluted natural substances for treatment.
            Based on the patient's described symptoms, provide your homeopathic assessment including:
            - Homeopathic medicines with their potency (like Arnica 30C, Belladonna 200C, etc.)
            - Constitutional remedies based on symptoms
            - Dietary and lifestyle recommendations from homeopathic perspective
            Do not add any numbers or special characters in your response.
            Your response should be in one long paragraph. Answer as if you are talking to a real patient.
            Start with 'Based on your symptoms, from a homeopathic perspective, I believe you have ....'
            Don't respond as an AI model in markdown, your answer should mimic that of an actual homeopathic practitioner.
            Keep your answer concise (max 2-3 sentences). No preamble, start your answer right away.
            Always end with a positive and holistic healing note.
            
            Patient's described symptoms: """
    },
    "ayurveda": {
        "name": "Ayurvedic Doctor (Vaidya)",
        "icon": "🪷",
        "specialty": "Ayurveda",
        "prompt_with_image": """You have to act as an experienced Ayurvedic Doctor (Vaidya) following ancient Indian medical wisdom.
            You analyze conditions based on the three doshas - Vata, Pitta, and Kapha.
            What's in this image? Do you find any imbalance or condition from an Ayurvedic perspective?
            If you identify any dosha imbalance or condition, suggest remedies including:
            - Ayurvedic herbs and formulations (like Triphala, Ashwagandha, Turmeric, etc.)
            - Panchakarma or detox therapies if needed
            - Dietary recommendations based on dosha balance (what to eat and avoid)
            - Yoga asanas and pranayama for the condition
            - Daily routine (Dinacharya) modifications
            Do not add any numbers or special characters in your response.
            Your response should be in one long paragraph. Answer as if you are talking to a real patient.
            Don't say 'In the image I see' but say 'With what I see, according to Ayurvedic principles, I believe there is ....'
            Don't respond as an AI model in markdown, your answer should mimic that of an actual Ayurvedic Vaidya.
            Keep your answer concise (max 2-3 sentences). No preamble, start your answer right away.
            Always end with a positive note about natural healing and balance.
            
            Patient's described symptoms: """,
        "prompt_text_only": """You have to act as an experienced Ayurvedic Doctor (Vaidya) following ancient Indian medical wisdom.
            You analyze conditions based on the three doshas - Vata, Pitta, and Kapha.
            Based on the patient's described symptoms, provide your Ayurvedic assessment including:
            - Possible dosha imbalance (Vata, Pitta, or Kapha)
            - Ayurvedic herbs and formulations (like Triphala, Ashwagandha, Turmeric, etc.)
            - Panchakarma or detox therapies if needed
            - Dietary recommendations based on dosha balance (what to eat and avoid)
            - Yoga asanas and pranayama for the condition
            - Daily routine (Dinacharya) modifications
            Do not add any numbers or special characters in your response.
            Your response should be in one long paragraph. Answer as if you are talking to a real patient.
            Start with 'Based on your symptoms, according to Ayurvedic principles, I believe there is ....'
            Don't respond as an AI model in markdown, your answer should mimic that of an actual Ayurvedic Vaidya.
            Keep your answer concise (max 2-3 sentences). No preamble, start your answer right away.
            Always end with a positive note about natural healing and balance.
            
            Patient's described symptoms: """
    }
}

# Page config
st.set_page_config(
    page_title="AI Doctor | Medical Assistant",
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    /* Main container styling */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        background-attachment: fixed;
    }
    
    .stApp {
        background: linear-gradient(180deg, #f0f4f8 0%, #e2e8f0 100%);
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #1e3a5f 0%, #2d5a87 100%);
        padding: 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 40px rgba(0,0,0,0.15);
        text-align: center;
    }
    
    .main-header h1 {
        color: white;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .main-header p {
        color: #a8d4f0;
        font-size: 1.1rem;
    }
    
    /* Card styling */
    .card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        border: 1px solid #e2e8f0;
        margin-bottom: 1rem;
    }
    
    .card-header {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin-bottom: 1rem;
        padding-bottom: 0.75rem;
        border-bottom: 2px solid #e2e8f0;
    }
    
    .card-header h3 {
        color: #1e3a5f;
        font-size: 1.2rem;
        font-weight: 600;
        margin: 0;
    }
    
    /* Input mode tabs */
    .input-mode-container {
        background: white;
        padding: 1rem;
        border-radius: 12px;
        margin-bottom: 1rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    }
    
    /* Doctor selection cards */
    .doctor-card {
        background: white;
        padding: 1.2rem;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        border: 2px solid #e2e8f0;
        text-align: center;
        transition: all 0.3s ease;
        cursor: pointer;
    }
    
    .doctor-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.12);
    }
    
    .doctor-card.selected {
        border-color: #1e3a5f;
        background: #f0f7ff;
    }
    
    .doctor-card .icon {
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
    }
    
    .doctor-card .name {
        font-weight: 600;
        color: #1e3a5f;
        font-size: 1rem;
    }
    
    .doctor-card .specialty {
        color: #6b7280;
        font-size: 0.85rem;
    }
    
    /* Status badges */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem 1rem;
        border-radius: 50px;
        font-size: 0.875rem;
        font-weight: 500;
    }
    
    .status-success {
        background: #d1fae5;
        color: #065f46;
    }
    
    .status-warning {
        background: #fef3c7;
        color: #92400e;
    }
    
    .status-info {
        background: #dbeafe;
        color: #1e40af;
    }
    
    .status-optional {
        background: #f3f4f6;
        color: #6b7280;
    }
    
    /* Result sections */
    .result-section {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 1rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        border-left: 4px solid;
    }
    
    .result-transcription {
        border-left-color: #3b82f6;
    }
    
    .result-response {
        border-left-color: #10b981;
    }
    
    .result-response-homeopathy {
        border-left-color: #22c55e;
    }
    
    .result-response-ayurveda {
        border-left-color: #f59e0b;
    }
    
    .result-audio {
        border-left-color: #8b5cf6;
    }
    
    .result-title {
        color: #374151;
        font-size: 1rem;
        font-weight: 600;
        margin-bottom: 0.75rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .result-content {
        color: #4b5563;
        line-height: 1.6;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #1e3a5f 0%, #2d5a87 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(30, 58, 95, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(30, 58, 95, 0.4);
    }
    
    /* File uploader styling */
    .stFileUploader {
        border: 2px dashed #cbd5e1;
        border-radius: 12px;
        padding: 1rem;
        transition: all 0.3s ease;
    }
    
    .stFileUploader:hover {
        border-color: #1e3a5f;
        background: #f8fafc;
    }
    
    /* Text area styling */
    .stTextArea textarea {
        border-radius: 12px;
        border: 2px solid #e2e8f0;
        padding: 1rem;
        font-size: 1rem;
        transition: all 0.3s ease;
    }
    
    .stTextArea textarea:focus {
        border-color: #1e3a5f;
        box-shadow: 0 0 0 3px rgba(30, 58, 95, 0.1);
    }
    
    /* Audio recorder container */
    .audio-recorder-container {
        display: flex;
        justify-content: center;
        padding: 1rem;
        background: #f8fafc;
        border-radius: 12px;
        border: 2px dashed #cbd5e1;
    }
    
    /* Disclaimer styling */
    .disclaimer {
        background: #fef3c7;
        border: 1px solid #f59e0b;
        border-radius: 8px;
        padding: 1rem;
        margin-top: 1rem;
        font-size: 0.875rem;
        color: #92400e;
    }
    
    /* Input summary box */
    .input-summary {
        background: #f0f7ff;
        border: 1px solid #bfdbfe;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .input-summary-item {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.25rem 0;
    }
    
    /* Footer styling */
    .footer {
        text-align: center;
        padding: 2rem;
        color: #6b7280;
        font-size: 0.875rem;
    }
    
    /* Divider styling */
    hr {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, #cbd5e1, transparent);
        margin: 2rem 0;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: white;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        border: 2px solid #e2e8f0;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        border-color: #1e3a5f;
    }
    
    .stTabs [aria-selected="true"] {
        background: #f0f7ff;
        border-color: #1e3a5f;
    }
    
    /* Responsive adjustments */
    @media (max-width: 768px) {
        .main-header h1 {
            font-size: 1.75rem;
        }
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>🩺 AI Medical Assistant</h1>
    <p>Powered by Advanced AI • Allopathy | Homeopathy | Ayurveda</p>
</div>
""", unsafe_allow_html=True)

# Initialize session state
if "recorded_audio" not in st.session_state:
    st.session_state.recorded_audio = None
if "audio_saved" not in st.session_state:
    st.session_state.audio_saved = False
if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False
if "results" not in st.session_state:
    st.session_state.results = None
if "uploaded_image_data" not in st.session_state:
    st.session_state.uploaded_image_data = None
if "image_saved" not in st.session_state:
    st.session_state.image_saved = False
if "selected_doctor" not in st.session_state:
    st.session_state.selected_doctor = "allopathy"
if "text_symptoms" not in st.session_state:
    st.session_state.text_symptoms = ""
if "text_saved" not in st.session_state:
    st.session_state.text_saved = False

# Instructions
with st.expander("ℹ️ How to use this application", expanded=False):
    st.markdown("""
    ### 📋 Flexible Input Options
    
    You can provide your symptoms using **any combination** of the following:
    
    | Input Method | Description | Best For |
    |--------------|-------------|----------|
    | 📷 **Image** | Upload photo of affected area | Skin conditions, visible symptoms |
    | 🎤 **Voice** | Record your symptoms verbally | Detailed descriptions, hands-free |
    | ✍️ **Text** | Type your symptoms | Quick input, specific details |
    
    ### ✅ Minimum Requirement
    - Provide **at least ONE** type of input (image, voice, OR text)
    - For best results, provide **image + description** (voice or text)
    
    ⚠️ **Note:** This is for educational purposes only. Always consult a real healthcare professional.
    """)

# Doctor Selection Section
st.markdown("### 🏥 Choose Your Consultation Type")

col_doc1, col_doc2, col_doc3 = st.columns(3, gap="medium")

with col_doc1:
    allopathy_selected = st.session_state.selected_doctor == "allopathy"
    if st.button(
        f"👨‍⚕️\n\n**Allopathic**\n\nModern Medicine",
        key="btn_allopathy",
        use_container_width=True,
        type="primary" if allopathy_selected else "secondary"
    ):
        st.session_state.selected_doctor = "allopathy"
        st.session_state.analysis_done = False
        st.session_state.results = None
        st.rerun()

with col_doc2:
    homeopathy_selected = st.session_state.selected_doctor == "homeopathy"
    if st.button(
        f"🌿\n\n**Homeopathic**\n\nNatural Healing",
        key="btn_homeopathy",
        use_container_width=True,
        type="primary" if homeopathy_selected else "secondary"
    ):
        st.session_state.selected_doctor = "homeopathy"
        st.session_state.analysis_done = False
        st.session_state.results = None
        st.rerun()

with col_doc3:
    ayurveda_selected = st.session_state.selected_doctor == "ayurveda"
    if st.button(
        f"🪷\n\n**Ayurvedic**\n\nAncient Wisdom",
        key="btn_ayurveda",
        use_container_width=True,
        type="primary" if ayurveda_selected else "secondary"
    ):
        st.session_state.selected_doctor = "ayurveda"
        st.session_state.analysis_done = False
        st.session_state.results = None
        st.rerun()

# Show selected doctor info
selected_doc_info = DOCTOR_PROMPTS[st.session_state.selected_doctor]
st.markdown(f"""
<div style="background: #f0f7ff; padding: 1rem; border-radius: 8px; margin: 1rem 0; text-align: center;">
    <span style="font-size: 1.5rem;">{selected_doc_info['icon']}</span>
    <strong> Currently consulting with:</strong> {selected_doc_info['name']}
</div>
""", unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

# Input Section Header
st.markdown("### 📝 Describe Your Symptoms")
st.markdown("*Provide at least one type of input. More details = better diagnosis.*")

# Three columns for inputs
col1, col2, col3 = st.columns(3, gap="medium")

# Column 1: Image Upload
with col1:
    st.markdown("""
    <div class="card">
        <div class="card-header">
            <span style="font-size: 1.5rem;">📷</span>
            <h3>Image</h3>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    if not st.session_state.image_saved:
        st.markdown("**Upload medical image:**")
        uploaded_image = st.file_uploader(
            "Choose an image",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed",
            key="image_uploader"
        )
        
        if uploaded_image:
            st.session_state.uploaded_image_data = uploaded_image.getvalue()
            with open(PATIENT_IMAGE_PATH, "wb") as f:
                f.write(st.session_state.uploaded_image_data)
            st.session_state.image_saved = True
            st.rerun()
        else:
            st.caption("📤 JPG, JPEG, PNG")
            st.markdown("""
            <div class="status-badge status-optional">
                ⭕ Optional
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="status-badge status-success">
            ✅ Image Ready
        </div>
        """, unsafe_allow_html=True)
        st.image(st.session_state.uploaded_image_data, caption="Uploaded", use_container_width=True)
        
        if st.button("🔄 Change", key="change_image", use_container_width=True):
            st.session_state.uploaded_image_data = None
            st.session_state.image_saved = False
            st.session_state.analysis_done = False
            st.session_state.results = None
            if os.path.exists(PATIENT_IMAGE_PATH):
                os.remove(PATIENT_IMAGE_PATH)
            st.rerun()

# Column 2: Voice Input
with col2:
    st.markdown("""
    <div class="card">
        <div class="card-header">
            <span style="font-size: 1.5rem;">🎤</span>
            <h3>Voice</h3>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    if not st.session_state.audio_saved:
        st.markdown("**Record symptoms:**")
        audio_bytes = audio_recorder(
            text="",
            recording_color="#dc2626",
            neutral_color="#1e3a5f",
            icon_size="2x",
            key="audio_recorder"
        )
        st.caption("🔴 Click to record")
        
        if audio_bytes:
            st.session_state.recorded_audio = audio_bytes
            with open(PATIENT_AUDIO_PATH, "wb") as f:
                f.write(st.session_state.recorded_audio)
            st.session_state.audio_saved = True
            st.rerun()
        else:
            st.markdown("""
            <div class="status-badge status-optional">
                ⭕ Optional
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="status-badge status-success">
            ✅ Audio Ready
        </div>
        """, unsafe_allow_html=True)
        st.audio(st.session_state.recorded_audio, format="audio/wav")
        
        if st.button("🔄 Re-record", key="record_again", use_container_width=True):
            st.session_state.recorded_audio = None
            st.session_state.audio_saved = False
            st.session_state.analysis_done = False
            st.session_state.results = None
            if os.path.exists(PATIENT_AUDIO_PATH):
                os.remove(PATIENT_AUDIO_PATH)
            st.rerun()

# Column 3: Text Input
with col3:
    st.markdown("""
    <div class="card">
        <div class="card-header">
            <span style="font-size: 1.5rem;">✍️</span>
            <h3>Text</h3>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("**Type symptoms:**")
    text_input = st.text_area(
        "Describe your symptoms",
        value=st.session_state.text_symptoms,
        height=120,
        placeholder="E.g., I have been experiencing headaches for 3 days, along with mild fever and body ache...",
        label_visibility="collapsed",
        key="text_symptoms_input"
    )
    
    # Update session state when text changes
    if text_input != st.session_state.text_symptoms:
        st.session_state.text_symptoms = text_input
        st.session_state.text_saved = bool(text_input.strip())
        st.session_state.analysis_done = False
        st.session_state.results = None
    
    if st.session_state.text_saved:
        st.markdown("""
        <div class="status-badge status-success">
            ✅ Text Ready
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="status-badge status-optional">
            ⭕ Optional
        </div>
        """, unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

# Analysis section
if not st.session_state.analysis_done:
    # Check what inputs are available
    image_ready = st.session_state.image_saved
    audio_ready = st.session_state.audio_saved
    text_ready = st.session_state.text_saved
    
    # At least one input is required
    any_input_ready = image_ready or audio_ready or text_ready
    
    # Input summary
    st.markdown("### 📊 Input Summary")
    
    col_sum1, col_sum2, col_sum3 = st.columns(3)
    
    with col_sum1:
        if image_ready:
            st.success("✅ Image provided")
        else:
            st.info("⭕ No image")
    
    with col_sum2:
        if audio_ready:
            st.success("✅ Voice recorded")
        else:
            st.info("⭕ No voice")
    
    with col_sum3:
        if text_ready:
            st.success("✅ Text provided")
        else:
            st.info("⭕ No text")
    
    # Warning if no input
    if not any_input_ready:
        st.warning("⚠️ Please provide at least **one** type of input (image, voice, or text) to get a consultation.")
    
    st.markdown("")
    
    # Get selected doctor info for button
    doc_info = DOCTOR_PROMPTS[st.session_state.selected_doctor]
    
    if st.button(
        f"🔍 Get {doc_info['specialty']} Consultation", 
        type="primary", 
        use_container_width=True, 
        disabled=not any_input_ready
    ):
        # Combine all text inputs
        combined_symptoms = ""
        transcription_text = ""
        
        # Processing with status updates
        with st.status(f"🔬 Consulting {doc_info['name']}...", expanded=True) as status:
            
            # Step 1: Transcribe audio if available
            if audio_ready:
                st.write("🎤 Transcribing voice input...")
                transcription_text = transcribe_with_groq(
                    GROQ_API_KEY=GROQ_API_KEY,
                    audio_filepath=PATIENT_AUDIO_PATH,
                    stt_model="whisper-large-v3"
                )
                combined_symptoms += f"[Voice Description]: {transcription_text} "
            
            # Step 2: Add text input if available
            if text_ready:
                st.write("📝 Processing text input...")
                combined_symptoms += f"[Written Description]: {st.session_state.text_symptoms} "
            
            # If no symptoms described, add default message
            if not combined_symptoms.strip():
                combined_symptoms = "Patient has not described specific symptoms. Please analyze the image for any visible medical conditions."
            
            # Step 3: Analyze with or without image
            if image_ready:
                st.write(f"🔍 {doc_info['icon']} Analyzing image from {doc_info['specialty']} perspective...")
                system_prompt = doc_info["prompt_with_image"]
                encoded_image = encode_image(PATIENT_IMAGE_PATH)
                doctor_response = analyze_image_with_query(
                    query=system_prompt + combined_symptoms,
                    encoded_image=encoded_image,
                    model="meta-llama/llama-4-scout-17b-16e-instruct"
                )
            else:
                st.write(f"🔍 {doc_info['icon']} Analyzing symptoms from {doc_info['specialty']} perspective...")
                system_prompt = doc_info["prompt_text_only"]
                # For text-only, we need to use a different approach
                # Using the same function but with a placeholder image or text-only model
                # Here we'll create a simple text response using the image model with a blank/placeholder
                doctor_response = analyze_image_with_query(
                    query=system_prompt + combined_symptoms,
                    encoded_image=None,  # No image
                    model="meta-llama/llama-4-scout-17b-16e-instruct"
                )
            
            # Step 4: Generate voice response
            st.write("🔊 Generating voice response...")
            text_to_speech_with_gtts(
                input_text=doctor_response,
                output_filepath=OUTPUT_AUDIO_PATH
            )
            
            status.update(label="✅ Consultation Complete!", state="complete", expanded=False)
        
        # Prepare display text for symptoms
        symptoms_display = ""
        if audio_ready and transcription_text:
            symptoms_display += f"🎤 **Voice:** {transcription_text}\n\n"
        if text_ready:
            symptoms_display += f"✍️ **Text:** {st.session_state.text_symptoms}\n\n"
        if not symptoms_display:
            symptoms_display = "No symptoms described (image-only analysis)"
        
        # Save results to session state
        st.session_state.results = {
            "transcription": transcription_text if audio_ready else "",
            "text_input": st.session_state.text_symptoms if text_ready else "",
            "symptoms_display": symptoms_display,
            "response": doctor_response,
            "doctor_type": st.session_state.selected_doctor,
            "doctor_name": doc_info["name"],
            "doctor_icon": doc_info["icon"],
            "specialty": doc_info["specialty"],
            "has_image": image_ready,
            "has_audio": audio_ready,
            "has_text": text_ready
        }
        st.session_state.analysis_done = True
        st.rerun()

# Display results if analysis is done
if st.session_state.analysis_done and st.session_state.results:
    results = st.session_state.results
    
    st.markdown(f"## 📋 {results['doctor_icon']} {results['specialty']} Consultation Results")
    
    # Input methods used badge
    input_methods = []
    if results.get('has_image'):
        input_methods.append("📷 Image")
    if results.get('has_audio'):
        input_methods.append("🎤 Voice")
    if results.get('has_text'):
        input_methods.append("✍️ Text")
    
    st.markdown(f"**Inputs used:** {' • '.join(input_methods)}")
    
    # Symptoms described
    st.markdown("""
    <div class="result-section result-transcription">
        <div class="result-title">📝 Your Described Symptoms</div>
    </div>
    """, unsafe_allow_html=True)
    
    if results.get("symptoms_display"):
        st.info(results["symptoms_display"])
    else:
        st.info("Image-only analysis performed")
    
    # Doctor's response with appropriate styling
    response_class = f"result-response-{results['doctor_type']}" if results['doctor_type'] != 'allopathy' else 'result-response'
    st.markdown(f"""
    <div class="result-section {response_class}">
        <div class="result-title">{results['doctor_icon']} {results['doctor_name']}'s Assessment</div>
    </div>
    """, unsafe_allow_html=True)
    st.success(results["response"])
    
    # Audio response
    st.markdown("""
    <div class="result-section result-audio">
        <div class="result-title">🔊 Voice Response</div>
    </div>
    """, unsafe_allow_html=True)
    
    if os.path.exists(OUTPUT_AUDIO_PATH):
        with open(OUTPUT_AUDIO_PATH, "rb") as audio_file:
            audio_data = audio_file.read()
            st.audio(audio_data, format="audio/mp3", autoplay=True)
    
    # Disclaimer based on doctor type
    disclaimer_texts = {
        "allopathy": "This AI assistant provides information based on modern medicine principles.",
        "homeopathy": "This AI assistant provides information based on homeopathic principles. Homeopathy is a complementary medicine system.",
        "ayurveda": "This AI assistant provides information based on Ayurvedic principles. Ayurveda is a traditional Indian medicine system."
    }
    
    st.markdown(f"""
    <div class="disclaimer">
        ⚠️ <strong>Medical Disclaimer:</strong> {disclaimer_texts[results['doctor_type']]} 
        This is for educational and informational purposes only. 
        It is not a substitute for professional medical advice, diagnosis, or treatment. 
        Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.
    </div>
    """, unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

# Action buttons
col_btn1, col_btn2 = st.columns(2)

with col_btn1:
    if st.button("🔄 New Consultation", use_container_width=True):
        st.session_state.recorded_audio = None
        st.session_state.audio_saved = False
        st.session_state.analysis_done = False
        st.session_state.results = None
        st.session_state.uploaded_image_data = None
        st.session_state.image_saved = False
        st.session_state.text_symptoms = ""
        st.session_state.text_saved = False
        # Clean up temp files
        for filepath in [OUTPUT_AUDIO_PATH, PATIENT_AUDIO_PATH, PATIENT_IMAGE_PATH]:
            if os.path.exists(filepath):
                os.remove(filepath)
        st.rerun()

with col_btn2:
    if st.session_state.analysis_done and st.session_state.results:
        results = st.session_state.results
        
        # Build input methods string
        input_methods_str = []
        if results.get('has_image'):
            input_methods_str.append("Image")
        if results.get('has_audio'):
            input_methods_str.append("Voice")
        if results.get('has_text'):
            input_methods_str.append("Text")
        
        report_content = f"""
{'='*60}
AI MEDICAL CONSULTATION REPORT
{'='*60}

Consultation Type: {results['doctor_name']}
Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Input Methods: {', '.join(input_methods_str)}

{'='*60}
PATIENT'S SYMPTOMS:
{'='*60}
{results.get("symptoms_display", "No symptoms described").replace("**", "").replace("🎤", "[Voice]").replace("✍️", "[Text]")}

{'='*60}
{results['doctor_icon']} {results['specialty'].upper()} ASSESSMENT:
{'='*60}
{results["response"]}

{'='*60}
DISCLAIMER:
{'='*60}
This AI assistant is for educational purposes only.
The consultation was based on {results['specialty']} principles.
Always consult a qualified healthcare professional for proper diagnosis and treatment.
"""
        st.download_button(
            label="📥 Download Report",
            data=report_content,
            file_name=f"medical_consultation_{results['doctor_type']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
            use_container_width=True
        )

# Footer
st.markdown("""
<div class="footer">
    <p>🏥 AI Medical Assistant v2.0 | Allopathy • Homeopathy • Ayurveda</p>
    <p>📷 Image | 🎤 Voice | ✍️ Text - Flexible Input Options</p>
    <p>© 2026 Medical AI Project | For Educational Purposes Only</p>
</div>
""", unsafe_allow_html=True)