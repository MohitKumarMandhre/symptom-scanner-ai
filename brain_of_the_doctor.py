import os
import base64
from groq import Groq

def encode_image(image_path):
    """Encode image to base64 string"""
    if image_path is None or not os.path.exists(image_path):
        return None
    
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def analyze_image_with_query(query, encoded_image, model):
    """
    Analyze image with query or perform text-only analysis if no image
    
    Args:
        query: The prompt/query text
        encoded_image: Base64 encoded image or None for text-only
        model: The model to use
    
    Returns:
        str: The model's response
    """
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    
    # Build messages based on whether image is available
    if encoded_image:
        # Image + Text analysis
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": query
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{encoded_image}"
                        }
                    }
                ]
            }
        ]
    else:
        # Text-only analysis
        messages = [
            {
                "role": "user",
                "content": query
            }
        ]
    
    # Make API call
    try:
        chat_completion = client.chat.completions.create(
            messages=messages,
            model=model
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        # If vision model fails for text-only, try with a text model
        if not encoded_image:
            try:
                chat_completion = client.chat.completions.create(
                    messages=messages,
                    model="llama-3.3-70b-versatile"  # Fallback text model
                )
                return chat_completion.choices[0].message.content
            except Exception as e2:
                return f"Error processing your request: {str(e2)}"
        return f"Error analyzing image: {str(e)}"