import math
import os
import smtplib
from email.mime.text import MIMEText

import vertexai
from vertexai.preview.language_models import TextEmbeddingModel
from vertexai.generative_models import GenerativeModel
from google.cloud import storage

import sounddevice as sd
from scipy.io.wavfile import write
from google.cloud import speech
from google.cloud import texttospeech

# ---------------- CONFIG ----------------
PROJECT_ID = ""
LOCATION = " "
BUCKET_NAME = ""

vertexai.init(project=PROJECT_ID, location=LOCATION)

embedding_model = TextEmbeddingModel.from_pretrained("text-embedding-004")
llm = GenerativeModel("gemini-2.5-flash-lite")

# ---------------- CHUNKING ----------------
def chunk_text(text, size=150):
    words = text.split()
    return [" ".join(words[i:i+size]) for i in range(0, len(words), size)]

# ---------------- LOAD DATA ----------------
def load_all_files():
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)

    docs = []
    for blob in bucket.list_blobs():
        if blob.name.endswith(".txt"):
            content = blob.download_as_text()
            docs.extend(chunk_text(content))

    return docs

documents = load_all_files()
print(f"✅ Loaded {len(documents)} chunks")

# ---------------- EMBEDDINGS ----------------
doc_embeddings = [
    embedding_model.get_embeddings([doc])[0].values for doc in documents
]

# ---------------- SIMILARITY ----------------
def cosine_similarity(a, b):
    dot = sum(x*y for x,y in zip(a,b))
    norm_a = math.sqrt(sum(x*x for x in a))
    norm_b = math.sqrt(sum(x*x for x in b))
    return dot / (norm_a * norm_b + 1e-10)

# ---------------- RETRIEVAL ----------------
def retrieve_docs(query, k=5):
    q_emb = embedding_model.get_embeddings([query])[0].values

    scores = []
    for doc, emb in zip(documents, doc_embeddings):
        score = cosine_similarity(q_emb, emb)
        scores.append((score, doc))

    scores.sort(reverse=True)
    top_docs = [doc for _, doc in scores[:k]]

    context = "\n".join(top_docs)
    print("\n🔍 Retrieved Context:\n", context)

    return context

# ---------------- EMAIL TOOL ----------------
def send_email(subject, body, to_email):
    try:
        sender_email = ""
        app_password = ""  # remove spaces

        msg = MIMEText(body)
        msg["Subject"] = subject
        msg["From"] = sender_email
        msg["To"] = ""

        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(sender_email, app_password)
        server.send_message(msg)
        server.quit()

        return "📧 Email sent successfully"

    except Exception as e:
        return f"❌ Email failed: {e}"

# ---------------- AGENT ----------------
def agent(query):
    print("\n🧠 Thinking...")

    q = query.lower()
    is_email = any(word in q for word in ["email", "mail", "send"])

    # 🔥 Get RAG answer FIRST
    context = retrieve_docs(query)

    if not context.strip():
        answer = "I don't know"
    else:
        prompt = f"""
Answer ONLY using the context below.

Context:
{context}

Question:
{query}
"""
        answer = llm.generate_content(prompt).text

    # 🔥 Send answer via email if requested
    if is_email:
        result = send_email(
            subject="AI Answer",
            body=answer,
            to_email="your_email@gmail.com"
        )
        return f"{answer}\n\n{result}"

    return answer

# ---------------- VOICE ----------------
def record_audio(filename="input.wav", duration=5, fs=16000):
    print("🎤 Speak now...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()
    write(filename, fs, recording)
    return filename

def speech_to_text(audio_file):
    client = speech.SpeechClient()

    with open(audio_file, "rb") as f:
        content = f.read()

    audio = speech.RecognitionAudio(content=content)

    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code="en-US"
    )

    response = client.recognize(config=config, audio=audio)

    if response.results:
        return response.results[0].alternatives[0].transcript
    return ""

def text_to_speech(text, output_file="output.mp3"):
    client = texttospeech.TextToSpeechClient()

    synthesis_input = texttospeech.SynthesisInput(text=text)
    voice = texttospeech.VoiceSelectionParams(language_code="en-US")
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

    print("🔊 Playing response...")
    os.system(f"start wmplayer {output_file}")

# ---------------- MAIN ----------------
print("\n🎙️ Voice RAG Agent Ready\n")

while True:
    try:
        audio_file = record_audio()

        query = speech_to_text(audio_file)
        print("\n🗣 You said:", query)

        if not query:
            continue

        if "exit" in query.lower():
            break

        answer = agent(query)

        print("\n🤖 Answer:", answer)

        text_to_speech(answer)

    except Exception as e:
        print("Error:", e)

    print("\n" + "="*50 + "\n")