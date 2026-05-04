import math
import vertexai
from vertexai.preview.language_models import TextEmbeddingModel
from vertexai.generative_models import GenerativeModel
from google.cloud import storage

# ---------------- CONFIG ----------------
PROJECT_ID = ""
LOCATION = ""
BUCKET_NAME = ""

vertexai.init(project=PROJECT_ID, location=LOCATION)

embedding_model = TextEmbeddingModel.from_pretrained("text-embedding-004")
llm = GenerativeModel("gemini-2.5-flash-lite")

# ---------------- LOAD DATA ----------------
def load_all_files():
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)

    docs = []
    for blob in bucket.list_blobs():
        if blob.name.endswith(".txt"):
            docs.append(blob.download_as_text())

    return docs

documents = load_all_files()

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

# ---------------- TOOL ----------------
def retrieve_docs(query):
    q_emb = embedding_model.get_embeddings([query])[0].values

    scores = []
    for doc, emb in zip(documents, doc_embeddings):
        score = cosine_similarity(q_emb, emb)
        scores.append((score, doc))

    scores.sort(reverse=True)
    return "\n".join([doc for _, doc in scores[:3]])

# ---------------- AGENT LOOP ----------------
def agent(query):
    print("\n🧠 Thinking...")

    # Step 1: Decide if tool is needed
    decision_prompt = f"""
You are an AI agent.

User query: {query}

Decide:
- If the question needs document info → say: USE_TOOL
- Otherwise → say: DIRECT_ANSWER
"""

    decision = llm.generate_content(decision_prompt).text.strip()

    if "USE_TOOL" in decision:
        print("🔧 Using retrieval tool...")
        context = retrieve_docs(query)

        final_prompt = f"""
Answer based ONLY on this context:

{context}

Question: {query}
"""
        return llm.generate_content(final_prompt).text

    else:
        print("💬 Answering directly...")
        return llm.generate_content(query).text


# ---------------- CHAT ----------------
print("\n🤖 Custom Agent Ready\n")

while True:
    q = input(">> ")

    if q.lower() == "exit":
        break

    try:
        answer = agent(q)
        print("\nAnswer:\n", answer)

    except Exception as e:
        print("Error:", e)

    print("\n" + "="*50 + "\n")