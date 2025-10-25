import os
import pdfplumber
import traceback
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import google.generativeai as genai

# -----------------------------
# Load environment variables
# -----------------------------
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    print("Warning: GEMINI_API_KEY is not set. /analyze will not work without it.")
else:
    genai.configure(api_key=GEMINI_API_KEY)

# -----------------------------
# Flask app + CORS
# -----------------------------
app = Flask(__name__)
# You can allow all origins or restrict to your frontend URL
CORS(app, supports_credentials=True)

# -----------------------------
# Utility: extract text from PDF
# -----------------------------
def extract_text_from_pdf(pdf_file):
    text = ""
    try:
        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        print(f"PDF extraction error: {e}")
    return text.strip()

# -----------------------------
# Health check
# -----------------------------
@app.route("/health", methods=["GET"])
def health():
    return {"status": "ok", "gemini_configured": bool(GEMINI_API_KEY)}

# -----------------------------
# Analyze resume
# -----------------------------
@app.route("/analyze", methods=["POST"])
def analyze_resume():
    try:
        if not GEMINI_API_KEY:
            return jsonify({"error": "GEMINI_API_KEY not set"}), 500

        # Check if required fields exist
        if "resume" not in request.files or "jobRole" not in request.form:
            return jsonify({"error": "Missing resume or jobRole"}), 400

        resume_file = request.files["resume"]
        job_role = request.form["jobRole"]

        # Extract text from PDF
        resume_text = extract_text_from_pdf(resume_file)
        if not resume_text:
            return jsonify({"error": "Could not extract text from resume"}), 400

        # Prepare prompt for AI
        valid_model_name = "models/gemini-2.5-flash"
        prompt = f"""
You are an AI interview coach. Analyze the following resume for the role: {job_role}.

Resume:
{resume_text}

Provide:
- Strengths
- Weaknesses
- Suggestions to improve resume for this job role.
"""

        # Generate AI response safely
        try:
            model = genai.GenerativeModel(valid_model_name)
            response = model.generate_content(prompt)
            analysis_text = response.text
        except Exception as ai_err:
            print("AI model call failed:", ai_err)
            return jsonify({"error": "AI model failed", "details": str(ai_err)}), 500

        return jsonify({"analysis": analysis_text, "model_used": valid_model_name})

    except Exception as e:
        print("Unexpected error:", traceback.format_exc())
        return jsonify({"error": "Unexpected server error", "details": str(e)}), 500

# -----------------------------
# Run Flask (Render uses gunicorn)
# -----------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Backend running at http://0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port, debug=True)
