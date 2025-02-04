import os
from dotenv import load_dotenv
from time import perf_counter
from langchain import hub
from langchain_groq import ChatGroq
from qdrant_cloud import get_qdrant_db

load_dotenv()

llm1 = ChatGroq(
    model="llama-3.3-70b-versatile",
    max_retries=2,
)

llm2 = ChatGroq(
    model="deepseek-r1-distill-llama-70b",
    max_retries=2,
)

retriever = get_qdrant_db()
prompt = hub.pull("rlm/rag-prompt")

start = perf_counter()

question = "How do I build a robust concurrent program in Go?"
retrieved = retriever.invoke(question)
docs = "\n\n".join(doc.page_content for doc in retrieved)
messages = prompt.invoke({"question": question, "context": docs})

n = 0
for chunk in llm1.stream(messages):
    if n == 0:
        end = perf_counter()
        print(f"time to first token for = {end - start}")
        n += 1
    print(chunk.content, end="", flush=True)

print(f"\n\nDOCS: {docs}")
