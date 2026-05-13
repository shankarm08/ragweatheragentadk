import os
import smtplib
from email.mime.text import MIMEText

import chromadb
import vertexai
from vertexai.preview.language_models import TextEmbeddingModel
from vertexai.generative_models import GenerativeModel
from google.cloud import storage

import sounddevice as sd
from scipy.io.wavfile import write
from google.cloud import speech
from google.cloud import texttospeech

# =========================================================
# CONFIG
# =========================================================

# Use Environment Variables
PROJECT_ID = os.getenv("PROJECT_ID")
LOCATION = "us-central1"
BUCKET_NAME = os.getenv("BUCKET_NAME")

# Gmail Credentials
EMAIL = os.getenv("EMAIL")
APP_PASSWORD = os.getenv("APP_PASSWORD")

# =========================================================
# INITIALIZE VERTEX AI
# =========================================================

vertexai.init(
    project=PROJECT_ID,
    location=LOCATION
)

# Embedding Model
embedding_model = TextEmbeddingModel.from_pretrained(
    "text-embedding-004"
)

# Gemini Model
llm = GenerativeModel(
    "gemini-2.5-flash-lite"
)

# =========================================================
# CHROMADB SETUP
# =========================================================

# Create ChromaDB Client
chroma_client = chromadb.PersistentClient(
    path="./chroma_db"
)

# Create Collection
collection = chroma_client.get_or_create_collection(
    name="voice_rag_collection"
)

# =========================================================
# TEXT CHUNKING
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

        # TXT FILES
        if blob.name.endswith(".txt"):

            content = blob.download_as_text()

            chunks = chunk_text(content)

            docs.extend(chunks)

    return docs

# =========================================================
# LOAD DOCUMENTS
# =========================================================

documents = load_all_files()

print(f"\n✅ Loaded {len(documents)} document chunks")

# =========================================================
# STORE EMBEDDINGS IN CHROMADB
# =========================================================

print("\n📥 Creating embeddings and storing in ChromaDB...")

# Avoid duplicate insertion
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

    print("✅ Documents stored in ChromaDB")

else:

    print("✅ ChromaDB already contains embeddings")

# =========================================================
# RETRIEVAL
# =========================================================

def retrieve_docs(query, k=5):

    # Query Embedding
    query_embedding = embedding_model.get_embeddings(
        [query]
    )[0].values

    # Search ChromaDB
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
# EMAIL TOOL
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
# AI AGENT
# =========================================================

def agent(query):

    print("\n🧠 Thinking...")

    q = query.lower()

    is_email = any(
        word in q
        for word in ["email", "mail", "send"]
    )

    # Retrieve Context
    context = retrieve_docs(query)

    # Generate Answer
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

    # Send Email if requested
    if is_email:

        result = send_email(
            subject="AI Answer",
            body=answer,
            to_email=EMAIL
        )

        return f"{answer}\n\n{result}"

    return answer

# =========================================================
# RECORD AUDIO
# =========================================================

def record_audio(
    filename="input.wav",
    duration=5,
    fs=16000
):

    print("\n🎤 Speak now...")

    recording = sd.rec(
        int(duration * fs),
        samplerate=fs,
        channels=1,
        dtype="int16"
    )

    sd.wait()

    write(filename, fs, recording)

    return filename

# =========================================================
# SPEECH TO TEXT
# =========================================================

def speech_to_text(audio_file):

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

# =========================================================
# TEXT TO SPEECH
# =========================================================

def text_to_speech(
    text,
    output_file="output.mp3"
):

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

    os.system(f"start wmplayer {output_file}")

# =========================================================
# MAIN LOOP
# =========================================================

print("\n🎙️ Voice RAG Agent Ready\n")

while True:

    try:

        # Record Voice
        audio_file = record_audio()

        # Convert Speech → Text
        query = speech_to_text(audio_file)

        print("\n🗣 You said:")
        print(query)

        if not query:
            continue

        # Exit
        if "exit" in query.lower():

            print("\n👋 Exiting...")
            break

        # Get AI Answer
        answer = agent(query)

        print("\n🤖 Answer:\n")
        print(answer)

        # Speak Answer
        text_to_speech(answer)

    except Exception as e:

        print("\n❌ Error:", e)

    print("\n" + "=" * 60 + "\n")