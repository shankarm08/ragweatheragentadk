# =========================================================
# MULTIMODAL OCR + VISION + VOICE RAG AGENT
# =========================================================

import os
import time
import smtplib
import numpy as np
from email.mime.text import MIMEText

import cv2
import chromadb
import vertexai
import pytesseract
import sounddevice as sd
import keyboard

from scipy.io.wavfile import write

from google.cloud import storage
from google.cloud import speech
from google.cloud import texttospeech

from vertexai.preview.language_models import (
    TextEmbeddingModel
)

from vertexai.generative_models import (
    GenerativeModel,
    Part
)

# =========================================================
# CONFIG
# =========================================================

# Set Environment Variables before running:
#
# Windows CMD:
# set PROJECT_ID=your_project_id
# set BUCKET_NAME=your_bucket_name
# set EMAIL=your_email@gmail.com
# set APP_PASSWORD=your_app_password
# set GOOGLE_APPLICATION_CREDENTIALS=credentials.json
#
# Linux/Mac:
# export PROJECT_ID=your_project_id
# export BUCKET_NAME=your_bucket_name
# export EMAIL=your_email@gmail.com
# export APP_PASSWORD=your_app_password
# export GOOGLE_APPLICATION_CREDENTIALS=credentials.json

PROJECT_ID = os.getenv("PROJECT_ID")
LOCATION = os.getenv("LOCATION", "us-central1")
BUCKET_NAME = os.getenv("BUCKET_NAME")

EMAIL = os.getenv("EMAIL")
APP_PASSWORD = os.getenv("APP_PASSWORD")

# =========================================================
# TESSERACT OCR
# =========================================================

# Change path if needed
pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

# =========================================================
# INITIALIZE VERTEX AI
# =========================================================

vertexai.init(
    project=PROJECT_ID,
    location=LOCATION
)

# =========================================================
# LOAD MODELS
# =========================================================

embedding_model = TextEmbeddingModel.from_pretrained(
    "text-embedding-004"
)

llm = GenerativeModel(
    "gemini-2.5-flash-lite"
)

vision_model = GenerativeModel(
    "gemini-2.5-flash"
)

# =========================================================
# CHROMADB
# =========================================================

chroma_client = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = chroma_client.get_or_create_collection(
    name="voice_rag_collection"
)

# =========================================================
# CHUNKING
# =========================================================

def chunk_text(text, size=150):

    words = text.split()

    return [
        " ".join(words[i:i + size])
        for i in range(0, len(words), size)
    ]

# =========================================================
# LOAD FILES FROM GOOGLE CLOUD STORAGE
# =========================================================

def load_all_files():

    try:

        client = storage.Client()

        bucket = client.bucket(BUCKET_NAME)

        docs = []

        for blob in bucket.list_blobs():

            if blob.name.endswith(".txt"):

                print(f"\n📄 Loading: {blob.name}")

                content = blob.download_as_text()

                chunks = chunk_text(content)

                docs.extend(chunks)

        return docs

    except Exception as e:

        print("❌ GCS Error:", e)

        return []

# =========================================================
# LOAD DOCUMENTS
# =========================================================

documents = load_all_files()

print(f"\n✅ Loaded {len(documents)} chunks")

# =========================================================
# STORE EMBEDDINGS
# =========================================================

print("\n📥 Storing embeddings in ChromaDB...")

existing_docs = collection.count()

if existing_docs == 0:

    for i, doc in enumerate(documents):

        embedding = embedding_model.get_embeddings(
            [doc]
        )[0].values

        collection.add(
            ids=[str(i)],
            documents=[doc],
            embeddings=[embedding]
        )

    print("✅ Documents stored successfully")

else:

    print("✅ ChromaDB already contains embeddings")

# =========================================================
# RETRIEVE DOCUMENTS
# =========================================================

def retrieve_docs(query, k=5):

    try:

        query_embedding = embedding_model.get_embeddings(
            [query]
        )[0].values

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=k
        )

        top_docs = results["documents"][0]

        context = "\n".join(top_docs)

        print("\n🔍 Retrieved Context:\n")
        print(context)

        return context

    except Exception as e:

        print("❌ Retrieval Error:", e)

        return ""

# =========================================================
# EMAIL FUNCTION
# =========================================================

def send_email(subject, body, to_email):

    try:

        msg = MIMEText(body)

        msg["Subject"] = subject
        msg["From"] = EMAIL
        msg["To"] = to_email

        server = smtplib.SMTP(
            "smtp.gmail.com",
            587
        )

        server.starttls()

        server.login(
            EMAIL,
            APP_PASSWORD
        )

        server.send_message(msg)

        server.quit()

        return "📧 Email sent successfully"

    except Exception as e:

        return f"❌ Email failed: {e}"

# =========================================================
# CAMERA FUNCTION
# =========================================================

def capture_image(output_file="captured_image.jpg"):

    try:

        sd.stop()

        time.sleep(1)

        cv2.destroyAllWindows()

        cap = cv2.VideoCapture(
            0,
            cv2.CAP_DSHOW
        )

        print("\n🎥 Opening Camera...")

        if not cap.isOpened():

            print("❌ Cannot access webcam")

            return None

        print("\n📸 Webcam Started")
        print("👉 Press SPACE to capture")
        print("👉 Press ESC to cancel")

        while True:

            ret, frame = cap.read()

            if not ret:

                print("❌ Failed to read webcam")

                break

            cv2.imshow(
                "AI Camera",
                frame
            )

            key = cv2.waitKey(1)

            # SPACE KEY
            if key == 32:

                cv2.imwrite(
                    output_file,
                    frame
                )

                print(f"\n✅ Image saved: {output_file}")

                break

            # ESC KEY
            elif key == 27:

                output_file = None

                print("❌ Capture cancelled")

                break

        cap.release()

        cv2.destroyAllWindows()

        return output_file

    except Exception as e:

        print("❌ Camera Error:", e)

        return None

# =========================================================
# OCR FUNCTION
# =========================================================

def extract_text_from_image(image_path):

    try:

        image = cv2.imread(image_path)

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

        text = pytesseract.image_to_string(gray)

        print("\n📄 OCR Extracted Text:\n")
        print(text)

        return text

    except Exception as e:

        return f"❌ OCR Error: {e}"

# =========================================================
# GEMINI VISION
# =========================================================

def analyze_image(image_path, query):

    try:

        with open(image_path, "rb") as f:

            image_bytes = f.read()

        image_part = Part.from_data(
            mime_type="image/jpeg",
            data=image_bytes
        )

        prompt = f"""
Analyze this image carefully.

User Question:
{query}
"""

        response = vision_model.generate_content(
            [
                prompt,
                image_part
            ]
        )

        return response.text

    except Exception as e:

        return f"❌ Vision Error: {e}"

# =========================================================
# AI AGENT
# =========================================================

def agent(query):

    print("\n🧠 Thinking...")

    q = query.lower()

    # EMAIL MODE
    is_email = any(
        word in q
        for word in [
            "email",
            "mail",
            "send"
        ]
    )

    # OCR MODE
    is_ocr = any(
        word in q
        for word in [
            "scan",
            "invoice",
            "receipt",
            "ocr",
            "extract text",
            "document"
        ]
    )

    # VISION MODE
    is_vision = any(
        word in q
        for word in [
            "image",
            "photo",
            "picture",
            "camera",
            "screenshot"
        ]
    )

    # =====================================================
    # OCR FLOW
    # =====================================================

    if is_ocr:

        image_path = capture_image()

        if image_path:

            extracted_text = extract_text_from_image(
                image_path
            )

            prompt = f"""
Analyze the following OCR text carefully.

OCR TEXT:
{extracted_text}

User Question:
{query}
"""

            response = llm.generate_content(prompt)

            answer = response.text

        else:

            answer = "❌ No image captured."

    # =====================================================
    # VISION FLOW
    # =====================================================

    elif is_vision:

        image_path = capture_image()

        if image_path:

            answer = analyze_image(
                image_path,
                query
            )

        else:

            answer = "❌ No image captured."

    # =====================================================
    # RAG FLOW
    # =====================================================

    else:

        context = retrieve_docs(query)

        if not context.strip():

            answer = "I don't know."

        else:

            prompt = f"""
Answer ONLY using the context below.

Context:
{context}

Question:
{query}
"""

            response = llm.generate_content(prompt)

            answer = response.text

    # =====================================================
    # EMAIL SUPPORT
    # =====================================================

    if is_email:

        result = send_email(
            subject="AI Assistant Response",
            body=answer,
            to_email=EMAIL
        )

        return f"{answer}\n\n{result}"

    return answer

# =========================================================
# RECORD AUDIO
# =========================================================

def record_audio_dynamic(
    filename="input.wav",
    fs=16000
):

    try:

        print("\n🎤 Press ENTER to start recording")
        keyboard.wait("enter")

        print("🔴 Recording... Press ENTER again to stop")

        recording = []

        def callback(indata, frames, time, status):

            recording.append(indata.copy())

        with sd.InputStream(
            samplerate=fs,
            channels=1,
            dtype="int16",
            callback=callback
        ):

            keyboard.wait("enter")

        audio = np.concatenate(
            recording,
            axis=0
        )

        write(
            filename,
            fs,
            audio
        )

        print(f"✅ Audio saved: {filename}")

        return filename

    except Exception as e:

        print("❌ Recording Error:", e)

        return None

# =========================================================
# SPEECH TO TEXT
# =========================================================

def speech_to_text(audio_file):

    try:

        client = speech.SpeechClient()

        with open(audio_file, "rb") as f:

            content = f.read()

        audio = speech.RecognitionAudio(
            content=content
        )

        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=16000,
            language_code="en-US"
        )

        response = client.recognize(
            config=config,
            audio=audio
        )

        if response.results:

            return response.results[0].alternatives[0].transcript

        return ""

    except Exception as e:

        print("❌ Speech Error:", e)

        return ""

# =========================================================
# TEXT TO SPEECH
# =========================================================

def text_to_speech(
    text,
    output_file="output.mp3"
):

    try:

        client = texttospeech.TextToSpeechClient()

        synthesis_input = texttospeech.SynthesisInput(
            text=text
        )

        voice = texttospeech.VoiceSelectionParams(
            language_code="en-US"
        )

        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3
        )

        response = client.synthesize_speech(
            input=synthesis_input,
            voice=voice,
            audio_config=audio_config
        )

        with open(output_file, "wb") as out:

            out.write(response.audio_content)

        print("\n🔊 Playing response...")

        os.system(
            f"start wmplayer {output_file}"
        )

    except Exception as e:

        print("❌ TTS Error:", e)

# =========================================================
# MAIN LOOP
# =========================================================

print("\n🎙️ Multimodal OCR Voice RAG Agent Ready\n")

while True:

    try:

        sd.stop()

        # =================================================
        # RECORD AUDIO
        # =================================================

        audio_file = record_audio_dynamic()

        if not audio_file:
            continue

        # =================================================
        # SPEECH TO TEXT
        # =================================================

        query = speech_to_text(audio_file)

        print("\n🗣 You said:")
        print(query)

        if not query:
            continue

        # =================================================
        # EXIT
        # =================================================

        if "exit" in query.lower():

            print("\n👋 Exiting...")

            break

        # =================================================
        # AI RESPONSE
        # =================================================

        answer = agent(query)

        print("\n🤖 Answer:\n")
        print(answer)

        # =================================================
        # TEXT TO SPEECH
        # =================================================

        text_to_speech(answer)

    except Exception as e:

        print("\n❌ Error:", e)

    print("\n" + "=" * 60 + "\n")
