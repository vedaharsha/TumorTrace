import google.generativeai as genai
from dotenv import load_dotenv
import os

load_dotenv()

genai.configure(
    api_key=os.getenv("GEMINI_API_KEY")
)

model = genai.GenerativeModel("gemini-2.5-flash")


def generate_medical_summary(
    tumor_type,
    confidence
):
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


def ask_radiology_agent(question):

    prompt = f"""
You are an AI Radiology Assistant.

Answer the following question related to
brain tumors, MRI scans, Grad-CAM,
medical imaging and diagnosis support.

Question:
{question}

Keep answer simple and educational.
"""

    response = model.generate_content(prompt)

    return response.text