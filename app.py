import os
import pdfplumber
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Warn if key missing
if not GEMINI_API_KEY:
    print("⚠️ Warning: GEMINI_API_KEY not set in environment variables")

# Configure Gemini API
genai.configure(api_key=GEMINI_API_KEY)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# ✅ Root route for Render verification
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Resume Analyzer Backend is running successfully 🎉",
        "status": "live",
        "available_endpoints": ["/health", "/analyze"]
    }), 200


# ✅ Health check route
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy"}), 200


# ✅ Resume analysis route
@app.route("/analyze", methods=["POST"])
def analyze_resume():
    try:
        if "resume" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["resume"]

        # Extract text from PDF
        text = ""
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""

        if not text.strip():
            return jsonify({"error": "Could not extract text from resume"}), 400

        # Use Gemini model for analysis
        prompt = (
            "Analyze this resume and return JSON with strengths, weaknesses, and improvement suggestions:\n\n" + text
        )
        model = genai.GenerativeModel("gemini-pro")
        response = model.generate_content(prompt)

        # Clean and return response
        result = response.text
        return jsonify({"analysis": result})

    except Exception as e:
        print("❌ Error:", str(e))
        return jsonify({"error": str(e)}), 500


# ✅ Run Flask app locally (Render uses Gunicorn)
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
