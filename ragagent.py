import math
import vertexai
from vertexai.preview.language_models import TextEmbeddingModel
from vertexai.generative_models import GenerativeModel
from google.cloud import storage

# ---------------- CONFIG ----------------
PROJECT_ID = ""
LOCATION = ""
BUCKET_NAME = ""
# ---------------------------------------

# Init Vertex AI
vertexai.init(project=PROJECT_ID, location=LOCATION)

# Models
embedding_model = TextEmbeddingModel.from_pretrained("text-embedding-004")

# ✅ UPDATED MODEL
llm = GenerativeModel("gemini-2.5-flash-lite")


# ---------------- LOAD FILES ----------------
def load_all_files():
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)

    docs = []
    print("📂 Loading files...\n")

    for blob in bucket.list_blobs():
        if blob.name.endswith(".txt"):
            print(f"Reading: {blob.name}")
            content = blob.download_as_text()

            chunks = [line.strip() for line in content.split("\n") if line.strip()]
            docs.extend(chunks)

    return docs


documents = load_all_files()

if not documents:
    raise Exception("❌ No documents found in bucket")

print(f"\n✅ Loaded {len(documents)} chunks\n")


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


# ---------------- RETRIEVE ----------------
def retrieve(query, k=3):
    q_emb = embedding_model.get_embeddings([query])[0].values

    scores = []
    for doc, emb in zip(documents, doc_embeddings):
        score = cosine_similarity(q_emb, emb)
        scores.append((score, doc))

    scores.sort(reverse=True)
    return [doc for _, doc in scores[:k]]


# ---------------- ANSWER ----------------
def answer(query):
    context = "\n".join(retrieve(query))

    prompt = f"""
Answer ONLY from the context below.
If not found, say "I don't know".

Context:
{context}

Question:
{query}
"""

    response = llm.generate_content(prompt)
    return response.text


# ---------------- CHAT ----------------
print("\n🤖 RAG Assistant Ready (Gemini 2.5 Flash Lite)\n")

while True:
    q = input(">> ")

    if q.lower() == "exit":
        break

    try:
        print("\nAnswer:\n", answer(q))
    except Exception as e:
        print("❌ Error:", e)

    print("\n" + "="*50 + "\n")