# =========================================================
# MULTIMODAL OCR + VISION + VOICE RAG AGENT
# =========================================================

import os
import time
import smtplib
from email.mime.text import MIMEText

import cv2
import chromadb
import vertexai
import pytesseract
import sounddevice as sd

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

# Environment Variables
PROJECT_ID = os.getenv("PROJECT_ID")
LOCATION = "us-central1"
BUCKET_NAME = os.getenv("BUCKET_NAME")

EMAIL = os.getenv("EMAIL")
APP_PASSWORD = os.getenv("APP_PASSWORD")

# =========================================================
# TESSERACT OCR
# =========================================================

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
# MODELS
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

            if key == 32:

                cv2.imwrite(
                    output_file,
                    frame
                )

                print(f"\n✅ Image saved: {output_file}")

                break

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

    is_email = any(
        word in q
        for word in [
            "email",
            "mail",
            "send"
        ]
    )

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

    elif is_vision:

        image_path = capture_image()

        if image_path:

            answer = analyze_image(
                image_path,
                query
            )

        else:

            answer = "❌ No image captured."

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

    if is_email:

        result = send_email(
            subject="AI Assistant Response",
            body=answer,
            to_email=EMAIL
        )

        return f"{answer}\n\n{result}"

    return answer

print("\n🎙️ Secure Multimodal RAG Agent Ready")