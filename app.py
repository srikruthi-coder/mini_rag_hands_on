import os

import chromadb
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# -----------------------------
# Setup OpenAI
# -----------------------------
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# -----------------------------
# Setup ChromaDB
# -----------------------------
chroma_client = chromadb.PersistentClient(path="./chroma_db")

collection = chroma_client.get_or_create_collection(
    name="mini_rag"
)

# -----------------------------
# Load knowledge
# -----------------------------
with open("knowledge.txt", "r", encoding="utf-8") as file:
    text = file.read()

# -----------------------------
# Split text into chunks
# -----------------------------
chunk_size = 300

chunks = [
    text[i:i + chunk_size]
    for i in range(0, len(text), chunk_size)
]

# -----------------------------
# Create embeddings
# -----------------------------
for i, chunk in enumerate(chunks):

    embedding_response = client.embeddings.create(
        model="text-embedding-3-small",
        input=chunk
    )

    embedding = embedding_response.data[0].embedding

    collection.upsert(
        ids=[str(i)],
        documents=[chunk],
        embeddings=[embedding]
    )

print("Knowledge loaded into vector database!")


# -----------------------------
# Ask a question
# -----------------------------
def ask_rag(question):

    # Create embedding for question
    question_embedding = client.embeddings.create(
        model="text-embedding-3-small",
        input=question
    ).data[0].embedding

    # Search relevant chunks
    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=3
    )

    documents = results["documents"][0]

    context = "\n\n".join(documents)

    # -----------------------------
    # Generate answer
    # -----------------------------
    prompt = f"""
You are a helpful AI assistant.

Answer the question using ONLY the context provided below.

If the answer is not present in the context,
say "I don't know based on the provided knowledge."

Context:
{context}

Question:
{question}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response.choices[0].message.content


# -----------------------------
# Chat loop
# -----------------------------
print("\nMini RAG is ready!")
print("Type 'exit' to stop.\n")

while True:

    question = input("You: ")

    if question.lower() == "exit":
        break

    answer = ask_rag(question)

    print("\nAI:", answer)
    print()