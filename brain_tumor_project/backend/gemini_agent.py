import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

print("========== GEMINI CONFIG ==========")
print("KEY FOUND:", GEMINI_API_KEY is not None)
print("KEY PREFIX:", GEMINI_API_KEY[:10] if GEMINI_API_KEY else "NOT FOUND")
print("===================================")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

model = genai.GenerativeModel("gemini-1.5-flash")


def generate_medical_summary(tumor_type, confidence):
    try:
        prompt = f"""
You are an expert AI Radiology Assistant.

Brain MRI Classification Result:

Tumor Type: {tumor_type}
Confidence: {confidence}%

Generate:
1. Tumor explanation
2. Possible symptoms
3. Clinical recommendations
4. Patient-friendly summary

Keep response under 250 words.
Do not claim a final diagnosis.
"""

        response = model.generate_content(prompt)

        return response.text

    except Exception as e:
        print("GEMINI SUMMARY ERROR:", str(e))

        return (
            f"{tumor_type} detected with "
            f"{confidence}% confidence. "
            "Please consult a qualified radiologist "
            "for professional evaluation."
        )


def ask_radiology_agent(question):
    try:
        prompt = f"""
You are an AI Radiology Assistant.

Answer questions related to:
- Brain tumors
- MRI scans
- Grad-CAM
- Medical imaging
- Clinical interpretation

Question:
{question}

Provide a simple educational answer.
Do not provide medical diagnosis.
"""

        response = model.generate_content(prompt)

        return response.text

    except Exception as e:
        print("GEMINI CHAT ERROR:", str(e))

        return (
            "AI Assistant is temporarily unavailable. "
            "Please verify your Gemini API configuration."
        )