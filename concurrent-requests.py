import asyncio
import time
import os
from api import rag_query, get_prompt, get_llm
from qdrant_cloud import get_qdrant_db, get_embedding_func

# Disable LangSmith tracking and progress bars
os.environ["LANGCHAIN_TRACING_V2"] = "false"


async def run_c_req():
    # Initialize resources
    prompt = await get_prompt()
    retriever = get_qdrant_db()
    llm = await get_llm()
    embedding_func = get_embedding_func()

    try:
        start_time = time.perf_counter()

        queries = [
            "what is concurrency?",
            "what is parallelism?",
            #     "what is async programming?",
            #     "what is multithreading?",
            #     "what is multiprocessing?",
        ]

        results = await asyncio.gather(
            *(rag_query(query, i, prompt, llm, retriever)
              for i, query in enumerate(queries)),
            return_exceptions=True,
        )
        end_time = time.perf_counter()

        # Print results and any errors
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"Request {i} failed: {result}")
            else:
                print(f"Request {i} succeeded")

        print(f"Total time: {end_time - start_time} seconds")
    finally:
        # Ensure resources are properly cleaned up
        if hasattr(llm, 'client'):
            await llm.client.close()
        if hasattr(llm, 'aclose'):
            await llm.aclose()
        if hasattr(embedding_func, 'client'):
            await embedding_func.client.close()
        if hasattr(retriever, 'vectorstore'):
            await retriever.vectorstore.aclose()
        del llm
        del retriever
        del prompt
        del embedding_func


if __name__ == "__main__":
    import multiprocessing.resource_tracker
    asyncio.run(run_c_req())
    # Clean up any remaining semaphores
    multiprocessing.resource_tracker._resource_tracker = None
