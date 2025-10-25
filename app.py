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
CORS(app, origins=["*"], supports_credentials=True)  # Allow any origin for now

# -----------------------------
# Root route
# -----------------------------
@app.route("/", methods=["GET"])
def index():
    return """
    <h2>Resume Analyzer API</h2>
    <p>POST to <code>/analyze</code> with resume PDF and jobRole to analyze.</p>
    <p>Check health at <a href="/health">/health</a></p>
    """, 200

# -----------------------------
# Health check
# -----------------------------
@app.route("/health", methods=["GET"])
def health():
    return {"status": "ok", "gemini_configured": bool(GEMINI_API_KEY)}

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
# Analyze resume
# -----------------------------
@app.route("/analyze", methods=["POST", "OPTIONS"])
def analyze_resume():
    if request.method == "OPTIONS":
        response = jsonify({"status": "ok"})
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add("Access-Control-Allow-Headers", "Content-Type")
        response.headers.add("Access-Control-Allow-Methods", "POST")
        return response, 200

    try:
        if not GEMINI_API_KEY:
            return jsonify({"error": "GEMINI_API_KEY not set"}), 500

        if "resume" not in request.files or "jobRole" not in request.form:
            return jsonify({"error": "Missing resume or job role"}), 400

        resume_file = request.files["resume"]
        job_role = request.form["jobRole"]

        resume_text = extract_text_from_pdf(resume_file)
        if not resume_text:
            return jsonify({"error": "Could not extract text from resume"}), 400

        prompt = f"""
You are an AI interview coach. Analyze the following resume for the role: {job_role}.

Resume:
{resume_text}

Provide:
- Strengths
- Weaknesses
- Suggestions to improve resume for this job role.
"""

        model = genai.GenerativeModel("models/gemini-2.5-flash")
        response = model.generate_content(prompt)
        analysis_text = response.text

        return jsonify({"analysis": analysis_text, "model_used": "gemini-2.5-flash"})

    except Exception as e:
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

# -----------------------------
# Run Flask (Render sets $PORT)
# -----------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
