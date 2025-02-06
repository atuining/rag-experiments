import os
from contextlib import asynccontextmanager
from dotenv import load_dotenv
import asyncio
from time import perf_counter
from langchain import hub
from langchain_groq import ChatGroq
from qdrant_cloud import get_qdrant_db
from fastapi import FastAPI, Depends, HTTPException, Security, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from prometheus_fastapi_instrumentator import Instrumentator
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()


def get_prompt():
    return hub.pull("rlm/rag-prompt")


def get_llm():
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        max_retries=2,
    )


retriever = get_qdrant_db()
prompt = get_prompt()
llm = get_llm()

security = HTTPBearer()

EXPECTED_BEARER_TOKEN = os.environ.get("EXPECTED_BEARER_TOKEN")


async def verify_bearer_token(
    credentials: HTTPAuthorizationCredentials = Security(security),
):
    if credentials.credentials != EXPECTED_BEARER_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="invalid bearer token"
        )


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

instrumentator = Instrumentator().instrument(app).expose(app)


@app.get("/test")
async def test():
    return {"hello": "world"}


@app.post("/rag-query")
async def rag_query(
    query: str,
    index: int,
):
    start = perf_counter()
    # retriever = await get_db()
    retrieval_task = asyncio.create_task(retriever.ainvoke(query))
    retrieved = await retrieval_task
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


@app.get("/metrics")
async def get_metrics(auth: str = Depends(verify_bearer_token)):
    pass


async def main():
    async for chunk in rag_query("what is concurrency?"):
        print(chunk.content, end="", flush=True)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app)
