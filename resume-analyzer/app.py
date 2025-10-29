import os
import pdfplumber
import traceback
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import google.generativeai as genai
import time

# -----------------------------
# Load environment variables
# -----------------------------
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    print("⚠️ Warning: GEMINI_API_KEY is not set. /analyze will not work without it.")
else:
    genai.configure(api_key=GEMINI_API_KEY)

# -----------------------------
# Flask app + CORS
# -----------------------------
app = Flask(__name__)
CORS(app)

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
# Root endpoint
# -----------------------------
@app.route("/", methods=["GET"])
def index():
    return (
        "✅ Resume Analyzer API is running successfully! "
        "Use POST /analyze with a resume PDF and jobRole."
    )

# -----------------------------
# Health check
# -----------------------------
@app.route("/health", methods=["GET"])
def health():
    return {"status": "ok", "gemini_configured": bool(GEMINI_API_KEY)}

# -----------------------------
# Analyze resume (Optimized)
# -----------------------------
@app.route("/analyze", methods=["POST"])
def analyze_resume():
    start_time = time.time()
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

        # ✅ Trim to prevent memory overload
        MAX_CHARS = 4000
        truncated_resume = resume_text[:MAX_CHARS]
        if len(resume_text) > MAX_CHARS:
            print(f"Truncated resume text to {MAX_CHARS} characters to avoid memory issues.")

        valid_model_name = "models/gemini-2.5-flash"
        prompt = f"""
You are an AI interview coach. Analyze this resume for the role: {job_role}.

Resume (partial text if large):
{truncated_resume}

Provide:
1. Strengths
2. Weaknesses
3. Resume improvement suggestions
Make it concise and relevant to the {job_role} position.
"""

        model = genai.GenerativeModel(valid_model_name)

        # ✅ Safe timeout handling
        try:
            response = model.generate_content(prompt, request_options={"timeout": 60})
        except Exception as gemini_error:
            print("Gemini API Error:", gemini_error)
            return jsonify({"error": "Gemini API call failed or timed out"}), 500

        analysis_text = getattr(response, "text", None)
        if not analysis_text:
            return jsonify({"error": "No analysis returned from Gemini"}), 500

        duration = round(time.time() - start_time, 2)
        return jsonify({
            "analysis": analysis_text,
            "model_used": valid_model_name,
            "response_time_sec": duration
        })

    except MemoryError:
        print("💥 MemoryError: Resume too large for free Render instance.")
        return jsonify({"error": "Server out of memory. Try a smaller file or shorter resume."}), 500

    except Exception as e:
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

# -----------------------------
# Run Flask
# -----------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"🚀 Backend running at http://0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port)
