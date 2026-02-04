import os
import time
import requests
import numpy as np
import cv2
import mss
import pytesseract
import speech_recognition as sr
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
from dotenv import load_dotenv

# -------------------- ENV --------------------
load_dotenv()

# -------------------- TESSERACT --------------------
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# -------------------- FLASK --------------------
app = Flask(__name__)
app.config["SECRET_KEY"] = os.getenv("SECRET_KEY")
socketio = SocketIO(app)

# -------------------- HUGGING FACE --------------------
HF_API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"
HF_HEADERS = {
    "Authorization": f"Bearer {os.getenv('HF_API_KEY')}",
    "Content-Type": "application/json"
}

def hf_generate(prompt, max_tokens=200, retries=3):
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": max_tokens,
            "temperature": 0.5,
            "return_full_text": False
        }
    }

    for attempt in range(retries):
        try:
            response = requests.post(
                HF_API_URL,
                headers=HF_HEADERS,
                json=payload,
                timeout=60
            )

            if response.status_code == 200:
                return response.json()[0]["generated_text"]

            # Model loading / cold start
            if response.status_code in (503, 429):
                time.sleep(5)
                continue

        except Exception:
            time.sleep(5)

    return "⚠️ AI is warming up. Please try again in a moment."


# -------------------- SPEECH --------------------
recognizer = sr.Recognizer()

# -------------------- INTERVIEW STATE --------------------
interview_sessions = {}

TECHNICAL_TOPICS = {
    "data structures": ["array", "linked list", "tree", "graph", "hash"],
    "algorithms": ["sort", "search", "dynamic programming", "recursion"],
    "os": ["process", "thread", "scheduling", "memory"],
    "dbms": ["sql", "transaction", "index", "normalization"],
    "system design": ["scalability", "latency", "throughput", "load balancing"]
}

# -------------------- OCR --------------------
def analyze_screen_content():
    with mss.mss() as sct:
        monitor = sct.monitors[1]
        img = np.array(sct.grab(monitor))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        text = pytesseract.image_to_string(gray)
        return detect_technical_content(text)

def detect_technical_content(text):
    detected = []
    for topic, keywords in TECHNICAL_TOPICS.items():
        if any(k in text.lower() for k in keywords):
            detected.append(topic)
    return detected

# -------------------- FEEDBACK --------------------
def generate_structured_feedback(answer, question):
    prompt = f"""
You are a technical interview coach.

Question:
{question}

Candidate Answer:
{answer}

Give structured feedback in this format:
- Approach:
- Concepts:
- Pitfalls:
- Improvements:
"""
    return hf_generate(prompt, max_tokens=300)

# -------------------- ROUTES --------------------
@app.route("/")
def index():
    return render_template("index.html")

# -------------------- SOCKET EVENTS --------------------
@socketio.on("start_interview")
def start_interview(data):
    user_id = data["user_id"]
    interview_sessions[user_id] = {
        "questions": [],
        "responses": [],
        "feedback": [],
        "current_question": None,
        "start_time": time.time()
    }

    emit("interview_status", {
        "status": "started",
        "message": "Interview started. Hugging Face AI active.",
    })

@socketio.on("process_question")
def process_question(data):
    user_id = data["user_id"]
    question_text = data.get("text", "").strip()

    if not question_text:
        topics = analyze_screen_content()
        question_text = " ".join(topics) if topics else "No question detected"

    interview_sessions[user_id]["current_question"] = question_text
    interview_sessions[user_id]["questions"].append(question_text)

    emit("question_processed", {
        "question": question_text,
        "topics": detect_technical_content(question_text)
    })

@socketio.on("process_response")
def process_response(data):
    user_id = data["user_id"]
    response_text = data.get("text", "")

    question = interview_sessions[user_id]["current_question"]
    feedback = generate_structured_feedback(response_text, question)

    interview_sessions[user_id]["responses"].append(response_text)
    interview_sessions[user_id]["feedback"].append(feedback)

    emit("feedback", {
        "response": response_text,
        "feedback": feedback
    })

@socketio.on("request_hint")
def request_hint(data):
    user_id = data["user_id"]
    level = data.get("level", 1)
    question = interview_sessions[user_id]["current_question"]

    prompt = f"""
Question:
{question}

Give a Level {level} interview hint.
Level 1: General approach
Level 2: Key concepts
Level 3: Partial structure
"""

    hint = hf_generate(prompt)

    emit("hint", {
        "hint": hint,
        "level": level,
        "notice": f"Hugging Face Hint (Level {level})"
    })

# -------------------- RUN --------------------
if __name__ == "__main__":
    socketio.run(app, debug=True)
