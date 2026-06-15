from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from pydantic import BaseModel

import numpy as np
import cv2
from PIL import Image
import io
import uuid
import os
import matplotlib.pyplot as plt

from model import predict_tumor
from gradcam import generate_gradcam
from report import create_pdf
from email_service import send_email_with_pdf

from gemini_agent import (
    generate_medical_summary,
    ask_radiology_agent
)

# ==================================================
# FASTAPI APP
# ==================================================

app = FastAPI(
    title="Brain Tumor Detection API"
)

# ==================================================
# CORS
# ==================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================================================
# FOLDERS
# ==================================================

os.makedirs("static", exist_ok=True)
os.makedirs("static/uploads", exist_ok=True)
os.makedirs("static/gradcam", exist_ok=True)
os.makedirs("static/reports", exist_ok=True)
os.makedirs("static/charts", exist_ok=True)
os.makedirs("static/qr", exist_ok=True)

# ==================================================
# STATIC FILES
# ==================================================

app.mount(
    "/static",
    StaticFiles(directory="static"),
    name="static"
)

# ==================================================
# FALLBACK AI SUMMARY
# ==================================================

AI_SUMMARY = {
    "glioma":
        "Glioma detected. Further clinical evaluation is recommended.",

    "meningioma":
        "Meningioma detected. Specialist consultation is advised.",

    "pituitary":
        "Pituitary tumor detected. Endocrinological evaluation is recommended.",

    "no_tumor":
        "No tumor detected. MRI appears normal according to the AI model."
}

# ==================================================
# HOME
# ==================================================

@app.get("/")
def home():

    return {
        "message":
        "Brain Tumor Detection API Running Successfully"
    }

# ==================================================
# PREDICT
# ==================================================

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    patient_name: str = Form(...),
    patient_age: str = Form(...),
    patient_email: str = Form(...)
):

    try:

        print("Prediction Request Received")

        # ==========================================
        # READ IMAGE
        # ==========================================

        image_bytes = await file.read()

        pil_image = Image.open(
            io.BytesIO(image_bytes)
        ).convert("RGB")

        original_image = np.array(pil_image)

        # ==========================================
        # SAVE ORIGINAL MRI
        # ==========================================

        image_filename = f"{uuid.uuid4()}.jpg"

        image_path = os.path.join(
            "static/uploads",
            image_filename
        )

        pil_image.save(image_path)

        # ==========================================
        # PREPROCESS
        # ==========================================

        resized = cv2.resize(
            original_image,
            (224, 224)
        )

        normalized = resized / 255.0

        model_input = np.expand_dims(
            normalized,
            axis=0
        )

        # ==========================================
        # MODEL PREDICTION
        # ==========================================

        prediction, confidence, probabilities, model = predict_tumor(
            model_input
        )

        print("Prediction:", prediction)

        # ==========================================
        # GRADCAM
        # ==========================================

        gradcam_path = generate_gradcam(
            model=model,
            img_array=model_input,
            original_image=original_image,
            output_dir="static/gradcam"
        )

        gradcam_url = (
            "/static/gradcam/"
            + os.path.basename(gradcam_path)
        )

        # ==========================================
        # GEMINI SUMMARY
        # ==========================================

        try:

            ai_summary = generate_medical_summary(
                prediction,
                confidence
            )

        except Exception as gemini_error:

            print(
                "Gemini Error:",
                str(gemini_error)
            )

            ai_summary = AI_SUMMARY.get(
                prediction,
                "Clinical correlation recommended."
            )

        # ==========================================
        # PROBABILITY CHART
        # ==========================================

        chart_path = (
            f"static/charts/{uuid.uuid4()}.png"
        )

        plt.figure(figsize=(6, 4))

        plt.bar(
            list(probabilities.keys()),
            list(probabilities.values())
        )

        plt.title(
            "Tumor Classification Probabilities"
        )

        plt.ylabel(
            "Probability (%)"
        )

        plt.xlabel(
            "Classes"
        )

        plt.tight_layout()

        plt.savefig(chart_path)

        plt.close()

        # ==========================================
        # PDF REPORT
        # ==========================================

        report_id = str(uuid.uuid4())

        pdf_path = create_pdf(
            report_id=report_id,
            patient_name=patient_name,
            patient_age=patient_age,
            patient_email=patient_email,
            tumor_type=prediction,
            confidence=float(confidence),
            probabilities=probabilities,
            original_mri_path=image_path,
            gradcam_path=gradcam_path,
            probability_chart_path=chart_path,
            ai_summary=ai_summary
        )

        pdf_url = (
            "/static/reports/"
            + os.path.basename(pdf_path)
        )

        # ==========================================
        # EMAIL
        # ==========================================

        email_sent = False

        try:

            email_sent = send_email_with_pdf(
                to_email=patient_email,
                patient_name=patient_name,
                pdf_path=pdf_path
            )

        except Exception as email_error:

            print(
                "Email Error:",
                str(email_error)
            )

            email_sent = False

        # ==========================================
        # RESPONSE
        # ==========================================

        return {

            "prediction":
                prediction,

            "confidence":
                float(confidence),

            "probabilities":
                probabilities,

            "ai_summary":
                ai_summary,
            # NEW FIELD
            "mri_url":
            "/static/uploads/" + image_filename,


            "gradcam_url":
                gradcam_url,

            "pdf_url":
                pdf_url,

            "email_sent":
                email_sent,

            "patient_name":
                patient_name,

            "patient_age":
                patient_age,

            "patient_email":
                patient_email
        }

    except Exception as e:

        print("ERROR:", str(e))

        return JSONResponse(
            status_code=500,
            content={
                "error": str(e)
            }
        )

# ==================================================
# GEMINI AI AGENT
# ==================================================

class ChatRequest(BaseModel):
    question: str


@app.post("/ask-agent")
async def ask_agent(
    request: ChatRequest
):

    try:

        answer = ask_radiology_agent(
            request.question
        )

        return {
            "answer": answer
        }

    except Exception as e:

        return {
            "error": str(e)
        }
