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
OUTPUT_AUDIO_PATH = os.getenv("OUTPUT_AUDIO_PATH", "temp_docs/doctor_response.mp3")
PATIENT_AUDIO_PATH = os.getenv("PATIENT_AUDIO_PATH", "temp_docs/patient_audio.wav")
PATIENT_IMAGE_PATH = os.getenv("PATIENT_IMAGE_PATH", "temp_docs/patient_image.png")

# Ensure output directory exists
os.makedirs("temp_docs", exist_ok=True)

# Language configurations
LANGUAGE_CONFIG = {
    "english": {
        "name": "English",
        "flag": "🇬🇧",
        "code": "en",
        "whisper_lang": "en",
        "gtts_lang": "en",
        "ui": {
            "title": "🩺 AI Medical Assistant",
            "subtitle": "Powered by Advanced AI • Allopathy | Homeopathy | Ayurveda",
            "how_to_use": "ℹ️ How to use this application",
            "choose_consultation": "🏥 Choose Your Consultation Type",
            "describe_symptoms": "📝 Describe Your Symptoms",
            "input_hint": "*Provide at least one type of input. More details = better diagnosis.*",
            "image_label": "Image",
            "voice_label": "Voice",
            "text_label": "Text",
            "upload_image": "Upload medical image:",
            "record_symptoms": "Record symptoms:",
            "type_symptoms": "Type symptoms:",
            "text_placeholder": "E.g., I have been experiencing headaches for 3 days, along with mild fever and body ache...",
            "optional": "Optional",
            "ready": "Ready",
            "image_ready": "✅ Image Ready",
            "audio_ready": "✅ Audio Ready",
            "text_ready": "✅ Text Ready",
            "change": "🔄 Change",
            "rerecord": "🔄 Re-record",
            "input_summary": "📊 Input Summary",
            "image_provided": "✅ Image provided",
            "no_image": "⭕ No image",
            "voice_recorded": "✅ Voice recorded",
            "no_voice": "⭕ No voice",
            "text_provided": "✅ Text provided",
            "no_text": "⭕ No text",
            "warning_no_input": "⚠️ Please provide at least **one** type of input (image, voice, or text) to get a consultation.",
            "get_consultation": "🔍 Get {specialty} Consultation",
            "transcribing": "🎤 Transcribing voice input...",
            "processing_text": "📝 Processing text input...",
            "analyzing_image": "🔍 {icon} Analyzing image from {specialty} perspective...",
            "analyzing_symptoms": "🔍 {icon} Analyzing symptoms from {specialty} perspective...",
            "generating_voice": "🔊 Generating voice response...",
            "consultation_complete": "✅ Consultation Complete!",
            "consultation_results": "📋 {icon} {specialty} Consultation Results",
            "inputs_used": "Inputs used:",
            "your_symptoms": "📝 Your Described Symptoms",
            "assessment": "{icon} {doctor_name}'s Assessment",
            "voice_response": "🔊 Voice Response",
            "new_consultation": "🔄 New Consultation",
            "download_report": "📥 Download Report",
            "consulting": "🔬 Consulting {doctor_name}...",
            "currently_consulting": "Currently consulting with:",
            "select_language": "🌐 Select Language",
            "modern_medicine": "Modern Medicine",
            "natural_healing": "Natural Healing",
            "ancient_wisdom": "Ancient Wisdom"
        }
    },
    "hindi": {
        "name": "हिंदी",
        "flag": "🇮🇳",
        "code": "hi",
        "whisper_lang": "hi",
        "gtts_lang": "hi",
        "ui": {
            "title": "🩺 AI चिकित्सा सहायक",
            "subtitle": "उन्नत AI द्वारा संचालित • एलोपैथी | होम्योपैथी | आयुर्वेद",
            "how_to_use": "ℹ️ इस एप्लिकेशन का उपयोग कैसे करें",
            "choose_consultation": "🏥 अपना परामर्श प्रकार चुनें",
            "describe_symptoms": "📝 अपने लक्षण बताएं",
            "input_hint": "*कम से कम एक प्रकार का इनपुट प्रदान करें। अधिक विवरण = बेहतर निदान।*",
            "image_label": "छवि",
            "voice_label": "आवाज़",
            "text_label": "टेक्स्ट",
            "upload_image": "चिकित्सा छवि अपलोड करें:",
            "record_symptoms": "लक्षण रिकॉर्ड करें:",
            "type_symptoms": "लक्षण टाइप करें:",
            "text_placeholder": "उदाहरण: मुझे 3 दिनों से सिरदर्द हो रहा है, साथ में हल्का बुखार और बदन दर्द भी है...",
            "optional": "वैकल्पिक",
            "ready": "तैयार",
            "image_ready": "✅ छवि तैयार",
            "audio_ready": "✅ ऑडियो तैयार",
            "text_ready": "✅ टेक्स्ट तैयार",
            "change": "🔄 बदलें",
            "rerecord": "🔄 फिर से रिकॉर्ड करें",
            "input_summary": "📊 इनपुट सारांश",
            "image_provided": "✅ छवि प्रदान की गई",
            "no_image": "⭕ कोई छवि नहीं",
            "voice_recorded": "✅ आवाज़ रिकॉर्ड की गई",
            "no_voice": "⭕ कोई आवाज़ नहीं",
            "text_provided": "✅ टेक्स्ट प्रदान किया गया",
            "no_text": "⭕ कोई टेक्स्ट नहीं",
            "warning_no_input": "⚠️ कृपया परामर्श प्राप्त करने के लिए कम से कम **एक** प्रकार का इनपुट (छवि, आवाज़, या टेक्स्ट) प्रदान करें।",
            "get_consultation": "🔍 {specialty} परामर्श प्राप्त करें",
            "transcribing": "🎤 आवाज़ इनपुट को ट्रांसक्राइब कर रहे हैं...",
            "processing_text": "📝 टेक्स्ट इनपुट प्रोसेस कर रहे हैं...",
            "analyzing_image": "🔍 {icon} {specialty} दृष्टिकोण से छवि का विश्लेषण कर रहे हैं...",
            "analyzing_symptoms": "🔍 {icon} {specialty} दृष्टिकोण से लक्षणों का विश्लेषण कर रहे हैं...",
            "generating_voice": "🔊 आवाज़ प्रतिक्रिया उत्पन्न कर रहे हैं...",
            "consultation_complete": "✅ परामर्श पूर्ण!",
            "consultation_results": "📋 {icon} {specialty} परामर्श परिणाम",
            "inputs_used": "उपयोग किए गए इनपुट:",
            "your_symptoms": "📝 आपके बताए गए लक्षण",
            "assessment": "{icon} {doctor_name} का मूल्यांकन",
            "voice_response": "🔊 आवाज़ प्रतिक्रिया",
            "new_consultation": "🔄 नया परामर्श",
            "download_report": "📥 रिपोर्ट डाउनलोड करें",
            "consulting": "🔬 {doctor_name} से परामर्श कर रहे हैं...",
            "currently_consulting": "वर्तमान में परामर्श कर रहे हैं:",
            "select_language": "🌐 भाषा चुनें",
            "modern_medicine": "आधुनिक चिकित्सा",
            "natural_healing": "प्राकृतिक उपचार",
            "ancient_wisdom": "प्राचीन ज्ञान"
        }
    }
}

# Doctor type prompts - Updated for flexible input and multi-language
DOCTOR_PROMPTS = {
    "allopathy": {
        "name": {
            "english": "Allopathic Doctor (Modern Medicine)",
            "hindi": "एलोपैथिक डॉक्टर (आधुनिक चिकित्सा)"
        },
        "icon": "👨‍⚕️",
        "specialty": {
            "english": "Modern Medicine",
            "hindi": "आधुनिक चिकित्सा"
        },
        "prompt_with_image": {
            "english": """You have to act as an experienced Allopathic (Modern Medicine) Doctor. 
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
            "hindi": """आपको एक अनुभवी एलोपैथिक (आधुनिक चिकित्सा) डॉक्टर की तरह व्यवहार करना है।
                आप साक्ष्य-आधारित चिकित्सा का पालन करते हैं और पारंपरिक उपचार, दवाइयां और नैदानिक परीक्षणों का सुझाव दे सकते हैं।
                इस छवि में क्या है? क्या आपको इसमें चिकित्सकीय रूप से कुछ गलत लगता है?
                यदि आप विभेदक निदान करते हैं, तो कुछ उपचार सुझाएं जिनमें शामिल हैं:
                - यदि आवश्यक हो तो ओवर-द-काउंटर या प्रिस्क्रिप्शन दवाइयां
                - जीवनशैली में बदलाव
                - आपातकालीन देखभाल कब लेनी चाहिए
                अपनी प्रतिक्रिया में कोई नंबर या विशेष वर्ण न जोड़ें।
                आपकी प्रतिक्रिया हिंदी में एक लंबे पैराग्राफ में होनी चाहिए। ऐसे जवाब दें जैसे आप एक वास्तविक मरीज से बात कर रहे हों।
                'छवि में मुझे दिखता है' न कहें बल्कि कहें 'जो मुझे दिख रहा है, मुझे लगता है आपको....'
                AI मॉडल की तरह मार्कडाउन में जवाब न दें, आपका जवाब एक वास्तविक डॉक्टर जैसा होना चाहिए।
                अपना जवाब संक्षिप्त रखें (अधिकतम 2-3 वाक्य)। कोई प्रस्तावना नहीं, सीधे जवाब शुरू करें।
                हमेशा सकारात्मक और आश्वस्त करने वाले नोट के साथ समाप्त करें।
                
                मरीज के बताए गए लक्षण: """
        },
        "prompt_text_only": {
            "english": """You have to act as an experienced Allopathic (Modern Medicine) Doctor. 
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
                
                Patient's described symptoms: """,
            "hindi": """आपको एक अनुभवी एलोपैथिक (आधुनिक चिकित्सा) डॉक्टर की तरह व्यवहार करना है।
                आप साक्ष्य-आधारित चिकित्सा का पालन करते हैं और पारंपरिक उपचार, दवाइयां और नैदानिक परीक्षणों का सुझाव दे सकते हैं।
                मरीज के बताए गए लक्षणों के आधार पर, अपना चिकित्सा मूल्यांकन प्रदान करें जिसमें शामिल हैं:
                - लक्षणों के आधार पर संभावित स्थितियां
                - यदि आवश्यक हो तो ओवर-द-काउंटर या प्रिस्क्रिप्शन दवाइयां
                - जीवनशैली में बदलाव
                - आपातकालीन देखभाल कब लेनी चाहिए
                अपनी प्रतिक्रिया में कोई नंबर या विशेष वर्ण न जोड़ें।
                आपकी प्रतिक्रिया हिंदी में एक लंबे पैराग्राफ में होनी चाहिए। ऐसे जवाब दें जैसे आप एक वास्तविक मरीज से बात कर रहे हों।
                'आपके लक्षणों के आधार पर, मुझे लगता है आपको....' से शुरू करें
                AI मॉडल की तरह मार्कडाउन में जवाब न दें, आपका जवाब एक वास्तविक डॉक्टर जैसा होना चाहिए।
                अपना जवाब संक्षिप्त रखें (अधिकतम 2-3 वाक्य)। कोई प्रस्तावना नहीं, सीधे जवाब शुरू करें।
                हमेशा सकारात्मक और आश्वस्त करने वाले नोट के साथ समाप्त करें।
                
                मरीज के बताए गए लक्षण: """
        }
    },
    "homeopathy": {
        "name": {
            "english": "Homeopathic Doctor",
            "hindi": "होम्योपैथिक डॉक्टर"
        },
        "icon": "🌿",
        "specialty": {
            "english": "Homeopathy",
            "hindi": "होम्योपैथी"
        },
        "prompt_with_image": {
            "english": """You have to act as an experienced Homeopathic Doctor following the principles of Samuel Hahnemann.
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
            "hindi": """आपको सैमुअल हैनिमैन के सिद्धांतों का पालन करते हुए एक अनुभवी होम्योपैथिक डॉक्टर की तरह व्यवहार करना है।
                आप 'समान से समान का इलाज' में विश्वास करते हैं और उपचार के लिए अत्यधिक पतला प्राकृतिक पदार्थों का उपयोग करते हैं।
                इस छवि में क्या है? होम्योपैथिक दृष्टिकोण से क्या आपको इसमें कुछ गलत लगता है?
                यदि आप किसी स्थिति की पहचान करते हैं, तो कुछ उपचार सुझाएं जिनमें शामिल हैं:
                - उनकी शक्ति के साथ होम्योपैथिक दवाइयां (जैसे आर्निका 30C, बेलाडोना 200C, आदि)
                - लक्षणों के आधार पर संवैधानिक उपचार
                - होम्योपैथिक दृष्टिकोण से आहार और जीवनशैली की सिफारिशें
                अपनी प्रतिक्रिया में कोई नंबर या विशेष वर्ण न जोड़ें।
                आपकी प्रतिक्रिया हिंदी में एक लंबे पैराग्राफ में होनी चाहिए। ऐसे जवाब दें जैसे आप एक वास्तविक मरीज से बात कर रहे हों।
                'छवि में मुझे दिखता है' न कहें बल्कि कहें 'जो मुझे दिख रहा है, होम्योपैथिक सिद्धांतों के आधार पर, मुझे लगता है आपको....'
                AI मॉडल की तरह मार्कडाउन में जवाब न दें, आपका जवाब एक वास्तविक होम्योपैथिक चिकित्सक जैसा होना चाहिए।
                अपना जवाब संक्षिप्त रखें (अधिकतम 2-3 वाक्य)। कोई प्रस्तावना नहीं, सीधे जवाब शुरू करें।
                हमेशा सकारात्मक और समग्र उपचार नोट के साथ समाप्त करें।
                
                मरीज के बताए गए लक्षण: """
        },
        "prompt_text_only": {
            "english": """You have to act as an experienced Homeopathic Doctor following the principles of Samuel Hahnemann.
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
                
                Patient's described symptoms: """,
            "hindi": """आपको सैमुअल हैनिमैन के सिद्धांतों का पालन करते हुए एक अनुभवी होम्योपैथिक डॉक्टर की तरह व्यवहार करना है।
                आप 'समान से समान का इलाज' में विश्वास करते हैं और उपचार के लिए अत्यधिक पतला प्राकृतिक पदार्थों का उपयोग करते हैं।
                मरीज के बताए गए लक्षणों के आधार पर, अपना होम्योपैथिक मूल्यांकन प्रदान करें जिसमें शामिल हैं:
                - उनकी शक्ति के साथ होम्योपैथिक दवाइयां (जैसे आर्निका 30C, बेलाडोना 200C, आदि)
                - लक्षणों के आधार पर संवैधानिक उपचार
                - होम्योपैथिक दृष्टिकोण से आहार और जीवनशैली की सिफारिशें
                अपनी प्रतिक्रिया में कोई नंबर या विशेष वर्ण न जोड़ें।
                आपकी प्रतिक्रिया हिंदी में एक लंबे पैराग्राफ में होनी चाहिए। ऐसे जवाब दें जैसे आप एक वास्तविक मरीज से बात कर रहे हों।
                'आपके लक्षणों के आधार पर, होम्योपैथिक दृष्टिकोण से, मुझे लगता है आपको....' से शुरू करें
                AI मॉडल की तरह मार्कडाउन में जवाब न दें, आपका जवाब एक वास्तविक होम्योपैथिक चिकित्सक जैसा होना चाहिए।
                अपना जवाब संक्षिप्त रखें (अधिकतम 2-3 वाक्य)। कोई प्रस्तावना नहीं, सीधे जवाब शुरू करें।
                हमेशा सकारात्मक और समग्र उपचार नोट के साथ समाप्त करें।
                
                मरीज के बताए गए लक्षण: """
        }
    },
    "ayurveda": {
        "name": {
            "english": "Ayurvedic Doctor (Vaidya)",
            "hindi": "आयुर्वेदिक डॉक्टर (वैद्य)"
        },
        "icon": "🪷",
        "specialty": {
            "english": "Ayurveda",
            "hindi": "आयुर्वेद"
        },
        "prompt_with_image": {
            "english": """You have to act as an experienced Ayurvedic Doctor (Vaidya) following ancient Indian medical wisdom.
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
            "hindi": """आपको प्राचीन भारतीय चिकित्सा ज्ञान का पालन करते हुए एक अनुभवी आयुर्वेदिक डॉक्टर (वैद्य) की तरह व्यवहार करना है।
                आप तीन दोषों - वात, पित्त और कफ के आधार पर स्थितियों का विश्लेषण करते हैं।
                इस छवि में क्या है? आयुर्वेदिक दृष्टिकोण से क्या आपको कोई असंतुलन या स्थिति दिखती है?
                यदि आप किसी दोष असंतुलन या स्थिति की पहचान करते हैं, तो उपचार सुझाएं जिनमें शामिल हैं:
                - आयुर्वेदिक जड़ी-बूटियां और फॉर्मूलेशन (जैसे त्रिफला, अश्वगंधा, हल्दी, आदि)
                - यदि आवश्यक हो तो पंचकर्म या डिटॉक्स थेरेपी
                - दोष संतुलन के आधार पर आहार संबंधी सिफारिशें (क्या खाएं और क्या न खाएं)
                - स्थिति के लिए योग आसन और प्राणायाम
                - दैनिक दिनचर्या (दिनचर्या) में बदलाव
                अपनी प्रतिक्रिया में कोई नंबर या विशेष वर्ण न जोड़ें।
                आपकी प्रतिक्रिया हिंदी में एक लंबे पैराग्राफ में होनी चाहिए। ऐसे जवाब दें जैसे आप एक वास्तविक मरीज से बात कर रहे हों।
                'छवि में मुझे दिखता है' न कहें बल्कि कहें 'जो मुझे दिख रहा है, आयुर्वेदिक सिद्धांतों के अनुसार, मुझे लगता है....'
                AI मॉडल की तरह मार्कडाउन में जवाब न दें, आपका जवाब एक वास्तविक आयुर्वेदिक वैद्य जैसा होना चाहिए।
                अपना जवाब संक्षिप्त रखें (अधिकतम 2-3 वाक्य)। कोई प्रस्तावना नहीं, सीधे जवाब शुरू करें।
                हमेशा प्राकृतिक उपचार और संतुलन के बारे में सकारात्मक नोट के साथ समाप्त करें।
                
                मरीज के बताए गए लक्षण: """
        },
        "prompt_text_only": {
            "english": """You have to act as an experienced Ayurvedic Doctor (Vaidya) following ancient Indian medical wisdom.
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
                
                Patient's described symptoms: """,
            "hindi": """आपको प्राचीन भारतीय चिकित्सा ज्ञान का पालन करते हुए एक अनुभवी आयुर्वेदिक डॉक्टर (वैद्य) की तरह व्यवहार करना है।
                आप तीन दोषों - वात, पित्त और कफ के आधार पर स्थितियों का विश्लेषण करते हैं।
                मरीज के बताए गए लक्षणों के आधार पर, अपना आयुर्वेदिक मूल्यांकन प्रदान करें जिसमें शामिल हैं:
                - संभावित दोष असंतुलन (वात, पित्त, या कफ)
                - आयुर्वेदिक जड़ी-बूटियां और फॉर्मूलेशन (जैसे त्रिफला, अश्वगंधा, हल्दी, आदि)
                - यदि आवश्यक हो तो पंचकर्म या डिटॉक्स थेरेपी
                - दोष संतुलन के आधार पर आहार संबंधी सिफारिशें (क्या खाएं और क्या न खाएं)
                - स्थिति के लिए योग आसन और प्राणायाम
                - दैनिक दिनचर्या (दिनचर्या) में बदलाव
                अपनी प्रतिक्रिया में कोई नंबर या विशेष वर्ण न जोड़ें।
                आपकी प्रतिक्रिया हिंदी में एक लंबे पैराग्राफ में होनी चाहिए। ऐसे जवाब दें जैसे आप एक वास्तविक मरीज से बात कर रहे हों।
                'आपके लक्षणों के आधार पर, आयुर्वेदिक सिद्धांतों के अनुसार, मुझे लगता है....' से शुरू करें
                AI मॉडल की तरह मार्कडाउन में जवाब न दें, आपका जवाब एक वास्तविक आयुर्वेदिक वैद्य जैसा होना चाहिए।
                अपना जवाब संक्षिप्त रखें (अधिकतम 2-3 वाक्य)। कोई प्रस्तावना नहीं, सीधे जवाब शुरू करें।
                हमेशा प्राकृतिक उपचार और संतुलन के बारे में सकारात्मक नोट के साथ समाप्त करें।
                
                मरीज के बताए गए लक्षण: """
        }
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
    
    /* Language selector styling */
    .language-selector {
        background: white;
        padding: 0.75rem 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        display: flex;
        justify-content: center;
        gap: 1rem;
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
</style>
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
if "selected_language" not in st.session_state:
    st.session_state.selected_language = "english"

# Helper function to get UI text
def get_ui_text(key):
    return LANGUAGE_CONFIG[st.session_state.selected_language]["ui"].get(key, key)

# Helper function to get doctor info in current language
def get_doctor_info(doctor_type, field):
    info = DOCTOR_PROMPTS[doctor_type].get(field)
    if isinstance(info, dict):
        return info.get(st.session_state.selected_language, info.get("english", ""))
    return info

# Language Selection - At the top
st.markdown("---")
col_lang1, col_lang2, col_lang3 = st.columns([1, 2, 1])

with col_lang2:
    lang_col1, lang_col2 = st.columns(2)
    
    with lang_col1:
        if st.button(
            "🇬🇧 English",
            key="btn_english",
            use_container_width=True,
            type="primary" if st.session_state.selected_language == "english" else "secondary"
        ):
            st.session_state.selected_language = "english"
            st.session_state.analysis_done = False
            st.session_state.results = None
            st.rerun()
    
    with lang_col2:
        if st.button(
            "🇮🇳 हिंदी",
            key="btn_hindi",
            use_container_width=True,
            type="primary" if st.session_state.selected_language == "hindi" else "secondary"
        ):
            st.session_state.selected_language = "hindi"
            st.session_state.analysis_done = False
            st.session_state.results = None
            st.rerun()

# Get current language config
lang_config = LANGUAGE_CONFIG[st.session_state.selected_language]
ui = lang_config["ui"]

# Header
st.markdown(f"""
<div class="main-header">
    <h1>{ui['title']}</h1>
    <p>{ui['subtitle']}</p>
</div>
""", unsafe_allow_html=True)

# Instructions
with st.expander(ui["how_to_use"], expanded=False):
    if st.session_state.selected_language == "english":
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
    else:
        st.markdown("""
        ### 📋 लचीले इनपुट विकल्प
        
        आप निम्नलिखित में से **किसी भी संयोजन** का उपयोग करके अपने लक्षण प्रदान कर सकते हैं:
        
        | इनपुट विधि | विवरण | के लिए सर्वश्रेष्ठ |
        |------------|--------|------------------|
        | 📷 **छवि** | प्रभावित क्षेत्र की तस्वीर अपलोड करें | त्वचा की स्थिति, दिखाई देने वाले लक्षण |
        | 🎤 **आवाज़** | अपने लक्षण मौखिक रूप से रिकॉर्ड करें | विस्तृत विवरण, हैंड्स-फ्री |
        | ✍️ **टेक्स्ट** | अपने लक्षण टाइप करें | त्वरित इनपुट, विशिष्ट विवरण |
        
        ### ✅ न्यूनतम आवश्यकता
        - **कम से कम एक** प्रकार का इनपुट प्रदान करें (छवि, आवाज़, या टेक्स्ट)
        - सर्वोत्तम परिणामों के लिए, **छवि + विवरण** (आवाज़ या टेक्स्ट) प्रदान करें
        
        ⚠️ **नोट:** यह केवल शैक्षिक उद्देश्यों के लिए है। हमेशा एक वास्तविक स्वास्थ्य पेशेवर से परामर्श करें।
        """)

# Doctor Selection Section
st.markdown(f"### {ui['choose_consultation']}")

col_doc1, col_doc2, col_doc3 = st.columns(3, gap="medium")

with col_doc1:
    allopathy_selected = st.session_state.selected_doctor == "allopathy"
    btn_label = f"👨‍⚕️\n\n**{'एलोपैथिक' if st.session_state.selected_language == 'hindi' else 'Allopathic'}**\n\n{ui['modern_medicine']}"
    if st.button(
        btn_label,
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
    btn_label = f"🌿\n\n**{'होम्योपैथिक' if st.session_state.selected_language == 'hindi' else 'Homeopathic'}**\n\n{ui['natural_healing']}"
    if st.button(
        btn_label,
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
    btn_label = f"🪷\n\n**{'आयुर्वेदिक' if st.session_state.selected_language == 'hindi' else 'Ayurvedic'}**\n\n{ui['ancient_wisdom']}"
    if st.button(
        btn_label,
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
doctor_name = get_doctor_info(st.session_state.selected_doctor, "name")
st.markdown(f"""
<div style="background: #f0f7ff; padding: 1rem; border-radius: 8px; margin: 1rem 0; text-align: center;">
    <span style="font-size: 1.5rem;">{selected_doc_info['icon']}</span>
    <strong> {ui['currently_consulting']}</strong> {doctor_name}
</div>
""", unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

# Input Section Header
st.markdown(f"### {ui['describe_symptoms']}")
st.markdown(f"*{ui['input_hint']}*")

# Three columns for inputs
col1, col2, col3 = st.columns(3, gap="medium")

# Column 1: Image Upload
with col1:
    st.markdown(f"""
    <div class="card">
        <div class="card-header">
            <span style="font-size: 1.5rem;">📷</span>
            <h3>{ui['image_label']}</h3>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    if not st.session_state.image_saved:
        st.markdown(f"**{ui['upload_image']}**")
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
            st.markdown(f"""
            <div class="status-badge status-optional">
                ⭕ {ui['optional']}
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="status-badge status-success">
            {ui['image_ready']}
        </div>
        """, unsafe_allow_html=True)
        st.image(st.session_state.uploaded_image_data, caption="Uploaded", use_container_width=True)
        
        if st.button(ui['change'], key="change_image", use_container_width=True):
            st.session_state.uploaded_image_data = None
            st.session_state.image_saved = False
            st.session_state.analysis_done = False
            st.session_state.results = None
            if os.path.exists(PATIENT_IMAGE_PATH):
                os.remove(PATIENT_IMAGE_PATH)
            st.rerun()

# Column 2: Voice Input
with col2:
    st.markdown(f"""
    <div class="card">
        <div class="card-header">
            <span style="font-size: 1.5rem;">🎤</span>
            <h3>{ui['voice_label']}</h3>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    if not st.session_state.audio_saved:
        st.markdown(f"**{ui['record_symptoms']}**")
        audio_bytes = audio_recorder(
            text="",
            recording_color="#dc2626",
            neutral_color="#1e3a5f",
            icon_size="2x",
            key="audio_recorder"
        )
        st.caption("🔴 Click to record" if st.session_state.selected_language == "english" else "🔴 रिकॉर्ड करने के लिए क्लिक करें")
        
        if audio_bytes:
            st.session_state.recorded_audio = audio_bytes
            with open(PATIENT_AUDIO_PATH, "wb") as f:
                f.write(st.session_state.recorded_audio)
            st.session_state.audio_saved = True
            st.rerun()
        else:
            st.markdown(f"""
            <div class="status-badge status-optional">
                ⭕ {ui['optional']}
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="status-badge status-success">
            {ui['audio_ready']}
        </div>
        """, unsafe_allow_html=True)
        st.audio(st.session_state.recorded_audio, format="audio/wav")
        
        if st.button(ui['rerecord'], key="record_again", use_container_width=True):
            st.session_state.recorded_audio = None
            st.session_state.audio_saved = False
            st.session_state.analysis_done = False
            st.session_state.results = None
            if os.path.exists(PATIENT_AUDIO_PATH):
                os.remove(PATIENT_AUDIO_PATH)
            st.rerun()

# Column 3: Text Input
with col3:
    st.markdown(f"""
    <div class="card">
        <div class="card-header">
            <span style="font-size: 1.5rem;">✍️</span>
            <h3>{ui['text_label']}</h3>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"**{ui['type_symptoms']}**")
    text_input = st.text_area(
        "Describe your symptoms",
        value=st.session_state.text_symptoms,
        height=120,
        placeholder=ui['text_placeholder'],
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
        st.markdown(f"""
        <div class="status-badge status-success">
            {ui['text_ready']}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="status-badge status-optional">
            ⭕ {ui['optional']}
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
    st.markdown(f"### {ui['input_summary']}")
    
    col_sum1, col_sum2, col_sum3 = st.columns(3)
    
    with col_sum1:
        if image_ready:
            st.success(ui['image_provided'])
        else:
            st.info(ui['no_image'])
    
    with col_sum2:
        if audio_ready:
            st.success(ui['voice_recorded'])
        else:
            st.info(ui['no_voice'])
    
    with col_sum3:
        if text_ready:
            st.success(ui['text_provided'])
        else:
            st.info(ui['no_text'])
    
    # Warning if no input
    if not any_input_ready:
        st.warning(ui['warning_no_input'])
    
    st.markdown("")
    
    # Get selected doctor info for button
    doc_info = DOCTOR_PROMPTS[st.session_state.selected_doctor]
    specialty = get_doctor_info(st.session_state.selected_doctor, "specialty")
    doctor_name = get_doctor_info(st.session_state.selected_doctor, "name")
    
    button_label = ui['get_consultation'].format(specialty=specialty)
    
    if st.button(
        button_label, 
        type="primary", 
        use_container_width=True, 
        disabled=not any_input_ready
    ):
        # Combine all text inputs
        combined_symptoms = ""
        transcription_text = ""
        
        # Processing with status updates
        status_label = ui['consulting'].format(doctor_name=doctor_name)
        with st.status(status_label, expanded=True) as status:
            
            # Step 1: Transcribe audio if available
            if audio_ready:
                st.write(ui['transcribing'])
                transcription_text = transcribe_with_groq(
                    GROQ_API_KEY=GROQ_API_KEY,
                    audio_filepath=PATIENT_AUDIO_PATH,
                    stt_model="whisper-large-v3",
                    language=lang_config["whisper_lang"]
                )
                voice_label = "[Voice Description]" if st.session_state.selected_language == "english" else "[आवाज़ विवरण]"
                combined_symptoms += f"{voice_label}: {transcription_text} "
            
            # Step 2: Add text input if available
            if text_ready:
                st.write(ui['processing_text'])
                text_label = "[Written Description]" if st.session_state.selected_language == "english" else "[लिखित विवरण]"
                combined_symptoms += f"{text_label}: {st.session_state.text_symptoms} "
            
            # If no symptoms described, add default message
            if not combined_symptoms.strip():
                if st.session_state.selected_language == "english":
                    combined_symptoms = "Patient has not described specific symptoms. Please analyze the image for any visible medical conditions."
                else:
                    combined_symptoms = "मरीज ने विशिष्ट लक्षण नहीं बताए हैं। कृपया किसी भी दिखाई देने वाली चिकित्सा स्थिति के लिए छवि का विश्लेषण करें।"
            
            # Step 3: Analyze with or without image
            if image_ready:
                st.write(ui['analyzing_image'].format(icon=doc_info['icon'], specialty=specialty))
                system_prompt = doc_info["prompt_with_image"][st.session_state.selected_language]
                encoded_image = encode_image(PATIENT_IMAGE_PATH)
                doctor_response = analyze_image_with_query(
                    query=system_prompt + combined_symptoms,
                    encoded_image=encoded_image,
                    model="meta-llama/llama-4-scout-17b-16e-instruct"
                )
            else:
                st.write(ui['analyzing_symptoms'].format(icon=doc_info['icon'], specialty=specialty))
                system_prompt = doc_info["prompt_text_only"][st.session_state.selected_language]
                doctor_response = analyze_image_with_query(
                    query=system_prompt + combined_symptoms,
                    encoded_image=None,
                    model="meta-llama/llama-4-scout-17b-16e-instruct"
                )
            
            # Step 4: Generate voice response
            st.write(ui['generating_voice'])
            text_to_speech_with_gtts(
                input_text=doctor_response,
                output_filepath=OUTPUT_AUDIO_PATH,
                language=lang_config["gtts_lang"]
            )
            
            status.update(label=ui['consultation_complete'], state="complete", expanded=False)
        
        # Prepare display text for symptoms
        symptoms_display = ""
        if audio_ready and transcription_text:
            voice_emoji = "🎤"
            voice_text = "Voice" if st.session_state.selected_language == "english" else "आवाज़"
            symptoms_display += f"{voice_emoji} **{voice_text}:** {transcription_text}\n\n"
        if text_ready:
            text_emoji = "✍️"
            text_text = "Text" if st.session_state.selected_language == "english" else "टेक्स्ट"
            symptoms_display += f"{text_emoji} **{text_text}:** {st.session_state.text_symptoms}\n\n"
        if not symptoms_display:
            symptoms_display = "No symptoms described (image-only analysis)" if st.session_state.selected_language == "english" else "कोई लक्षण नहीं बताए गए (केवल छवि विश्लेषण)"
        
        # Save results to session state
        st.session_state.results = {
            "transcription": transcription_text if audio_ready else "",
            "text_input": st.session_state.text_symptoms if text_ready else "",
            "symptoms_display": symptoms_display,
            "response": doctor_response,
            "doctor_type": st.session_state.selected_doctor,
            "doctor_name": doctor_name,
            "doctor_icon": doc_info["icon"],
            "specialty": specialty,
            "has_image": image_ready,
            "has_audio": audio_ready,
            "has_text": text_ready,
            "language": st.session_state.selected_language
        }
        st.session_state.analysis_done = True
        st.rerun()

# Display results if analysis is done
if st.session_state.analysis_done and st.session_state.results:
    results = st.session_state.results
    
    results_title = ui['consultation_results'].format(icon=results['doctor_icon'], specialty=results['specialty'])
    st.markdown(f"## {results_title}")
    
    # Input methods used badge
    input_methods = []
    if results.get('has_image'):
        input_methods.append(f"📷 {ui['image_label']}")
    if results.get('has_audio'):
        input_methods.append(f"🎤 {ui['voice_label']}")
    if results.get('has_text'):
        input_methods.append(f"✍️ {ui['text_label']}")
    
    st.markdown(f"**{ui['inputs_used']}** {' • '.join(input_methods)}")
    
    # Symptoms described
    st.markdown(f"""
    <div class="result-section result-transcription">
        <div class="result-title">{ui['your_symptoms']}</div>
    </div>
    """, unsafe_allow_html=True)
    
    if results.get("symptoms_display"):
        st.info(results["symptoms_display"])
    else:
        no_symptoms_text = "Image-only analysis performed" if st.session_state.selected_language == "english" else "केवल छवि विश्लेषण किया गया"
        st.info(no_symptoms_text)
    
    # Doctor's response with appropriate styling
    response_class = f"result-response-{results['doctor_type']}" if results['doctor_type'] != 'allopathy' else 'result-response'
    assessment_title = ui['assessment'].format(icon=results['doctor_icon'], doctor_name=results['doctor_name'])
    st.markdown(f"""
    <div class="result-section {response_class}">
        <div class="result-title">{assessment_title}</div>
    </div>
    """, unsafe_allow_html=True)
    st.success(results["response"])
    
    # Audio response
    st.markdown(f"""
    <div class="result-section result-audio">
        <div class="result-title">{ui['voice_response']}</div>
    </div>
    """, unsafe_allow_html=True)
    
    if os.path.exists(OUTPUT_AUDIO_PATH):
        with open(OUTPUT_AUDIO_PATH, "rb") as audio_file:
            audio_data = audio_file.read()
            st.audio(audio_data, format="audio/mp3", autoplay=True)
    
    # Disclaimer based on doctor type and language
    if st.session_state.selected_language == "english":
        disclaimer_texts = {
            "allopathy": "This AI assistant provides information based on modern medicine principles.",
            "homeopathy": "This AI assistant provides information based on homeopathic principles. Homeopathy is a complementary medicine system.",
            "ayurveda": "This AI assistant provides information based on Ayurvedic principles. Ayurveda is a traditional Indian medicine system."
        }
        disclaimer_note = "This is for educational and informational purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition."
    else:
        disclaimer_texts = {
            "allopathy": "यह AI सहायक आधुनिक चिकित्सा सिद्धांतों के आधार पर जानकारी प्रदान करता है।",
            "homeopathy": "यह AI सहायक होम्योपैथिक सिद्धांतों के आधार पर जानकारी प्रदान करता है। होम्योपैथी एक पूरक चिकित्सा प्रणाली है।",
            "ayurveda": "यह AI सहायक आयुर्वेदिक सिद्धांतों के आधार पर जानकारी प्रदान करता है। आयुर्वेद एक पारंपरिक भारतीय चिकित्सा प्रणाली है।"
        }
        disclaimer_note = "यह केवल शैक्षिक और सूचनात्मक उद्देश्यों के लिए है। यह पेशेवर चिकित्सा सलाह, निदान या उपचार का विकल्प नहीं है। किसी भी चिकित्सा स्थिति के बारे में आपके किसी भी प्रश्न के लिए हमेशा अपने चिकित्सक या अन्य योग्य स्वास्थ्य प्रदाता की सलाह लें।"
    
    disclaimer_label = "Medical Disclaimer" if st.session_state.selected_language == "english" else "चिकित्सा अस्वीकरण"
    
    st.markdown(f"""
    <div class="disclaimer">
        ⚠️ <strong>{disclaimer_label}:</strong> {disclaimer_texts[results['doctor_type']]} 
        {disclaimer_note}
    </div>
    """, unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

# Action buttons
col_btn1, col_btn2 = st.columns(2)

with col_btn1:
    if st.button(ui['new_consultation'], use_container_width=True):
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
            if filepath and os.path.exists(filepath):
                os.remove(filepath)
        st.rerun()

with col_btn2:
    if st.session_state.analysis_done and st.session_state.results:
        results = st.session_state.results
        
        # Build input methods string
        input_methods_str = []
        if results.get('has_image'):
            input_methods_str.append("Image" if st.session_state.selected_language == "english" else "छवि")
        if results.get('has_audio'):
            input_methods_str.append("Voice" if st.session_state.selected_language == "english" else "आवाज़")
        if results.get('has_text'):
            input_methods_str.append("Text" if st.session_state.selected_language == "english" else "टेक्स्ट")
        
        if st.session_state.selected_language == "english":
            report_content = f"""
{'='*60}
AI MEDICAL CONSULTATION REPORT
{'='*60}

Consultation Type: {results['doctor_name']}
Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Input Methods: {', '.join(input_methods_str)}
Language: English

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
        else:
            report_content = f"""
{'='*60}
AI चिकित्सा परामर्श रिपोर्ट
{'='*60}

परामर्श प्रकार: {results['doctor_name']}
दिनांक: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
इनपुट विधियां: {', '.join(input_methods_str)}
भाषा: हिंदी

{'='*60}
मरीज के लक्षण:
{'='*60}
{results.get("symptoms_display", "कोई लक्षण नहीं बताए गए").replace("**", "").replace("🎤", "[आवाज़]").replace("✍️", "[टेक्स्ट]")}

{'='*60}
{results['doctor_icon']} {results['specialty'].upper()} मूल्यांकन:
{'='*60}
{results["response"]}

{'='*60}
अस्वीकरण:
{'='*60}
यह AI सहायक केवल शैक्षिक उद्देश्यों के लिए है।
परामर्श {results['specialty']} सिद्धांतों पर आधारित था।
उचित निदान और उपचार के लिए हमेशा योग्य स्वास्थ्य पेशेवर से परामर्श करें।
"""
        st.download_button(
            label=ui['download_report'],
            data=report_content,
            file_name=f"medical_consultation_{results['doctor_type']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
            use_container_width=True
        )

# Footer
if st.session_state.selected_language == "english":
    footer_text = """
    <div class="footer">
        <p>🏥 AI Medical Assistant v2.0 | Allopathy • Homeopathy • Ayurveda</p>
        <p>📷 Image | 🎤 Voice | ✍️ Text - Flexible Input Options</p>
        <p>🌐 English | हिंदी - Multilingual Support</p>
        <p>© 2026 Medical AI Project | For Educational Purposes Only</p>
    </div>
    """
else:
    footer_text = """
    <div class="footer">
        <p>🏥 AI चिकित्सा सहायक v2.0 | एलोपैथी • होम्योपैथी • आयुर्वेद</p>
        <p>📷 छवि | 🎤 आवाज़ | ✍️ टेक्स्ट - लचीले इनपुट विकल्प</p>
        <p>🌐 English | हिंदी - बहुभाषी समर्थन</p>
        <p>© 2026 मेडिकल AI प्रोजेक्ट | केवल शैक्षिक उद्देश्यों के लिए</p>
    </div>
    """
st.markdown(footer_text, unsafe_allow_html=True)