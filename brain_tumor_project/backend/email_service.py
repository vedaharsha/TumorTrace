import os
import smtplib
from email.message import EmailMessage
from dotenv import load_dotenv

load_dotenv()

MAIL_USERNAME = os.getenv("MAIL_USERNAME")
MAIL_PASSWORD = os.getenv("MAIL_PASSWORD")
MAIL_SERVER = os.getenv("MAIL_SERVER", "smtp.gmail.com")
MAIL_PORT = int(os.getenv("MAIL_PORT", 587))

print("========== EMAIL CONFIG ==========")
print("MAIL_USERNAME:", MAIL_USERNAME)
print("MAIL_SERVER:", MAIL_SERVER)
print("MAIL_PORT:", MAIL_PORT)
print("PASSWORD_LOADED:", MAIL_PASSWORD is not None)
print("==================================")


def send_email_with_pdf(
    to_email,
    patient_name,
    pdf_path
):
    try:

        msg = EmailMessage()

        msg["Subject"] = (
            "AI Brain Tumor Detection Report"
        )

        msg["From"] = MAIL_USERNAME
        msg["To"] = to_email

        msg.set_content(
            f"""
Hello {patient_name},

Your Brain Tumor Analysis Report
is attached.

This report contains:
• Tumor Prediction
• Confidence Score
• Grad-CAM Visualization
• AI Summary

Important:
This report is AI-generated and
must be reviewed by a qualified
radiologist before making any
clinical decisions.

Regards,
TumorTrace AI
"""
        )

        with open(pdf_path, "rb") as f:
            pdf_data = f.read()

        msg.add_attachment(
            pdf_data,
            maintype="application",
            subtype="pdf",
            filename="TumorTrace_Report.pdf"
        )

        print(f"Sending email to: {to_email}")

        with smtplib.SMTP(
            MAIL_SERVER,
            587
        ) as server:

            server.starttls()

            server.login(
                MAIL_USERNAME,
                MAIL_PASSWORD
            )

            server.send_message(msg)

        print("EMAIL SENT SUCCESSFULLY")

        return True

    except Exception as e:

        print(
            "EMAIL ERROR:",
            str(e)
        )

        return False