from dotenv import load_dotenv
import asyncio
from time import perf_counter
from langchain import hub
from langchain_groq import ChatGroq
from qdrant_cloud import get_qdrant_db
from fastapi import FastAPI

load_dotenv()


async def get_prompt():
    return hub.pull("rlm/rag-prompt")


async def get_llm():
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        max_retries=2,
    )


async def get_db():
    return get_qdrant_db()


app = FastAPI()


@app.post("/rag-query")
async def rag_query(query: str, index: int, prompt, llm):
    start = perf_counter()
    retriever = await get_db()
    retrieved = await retriever.ainvoke(query)
    end = perf_counter()
    print(f"time to retrieve for {index} = {end - start}")
    docs = "\n\n".join(doc.page_content for doc in retrieved)
    messages = await prompt.ainvoke({"question": query, "context": docs})
    print("started task")
    start = perf_counter()
    resp = await llm.ainvoke(messages)
    end = perf_counter()
    print(f"time to first token for {index} = {end - start}")
    return resp


@app.on_event("shutdown")
async def shutdown_event():
    db = await get_db()
    if hasattr(db, "aclose"):
        await db.aclose()

    llm = await get_llm()
    if hasattr(llm, "aclose"):
        await llm.aclose()


async def main():
    async for chunk in rag_query("what is concurrency?"):
        print(chunk.content, end="", flush=True)


if __name__ == "__main__":
    print(asyncio.run(rag_query("what is concurrency?")).content)
