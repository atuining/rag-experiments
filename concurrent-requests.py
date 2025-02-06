import asyncio
import time
import os
from api import rag_query, get_prompt, get_llm
from qdrant_cloud import get_qdrant_db, get_embedding_func

# Disable LangSmith tracking and progress bars
os.environ["LANGCHAIN_TRACING_V2"] = "false"


async def run_c_req():
    # Initialize resources
    start_time = time.perf_counter()

    queries = [
        "what are graph based methods?",
        "what are density based methods?",
        #     "what is async programming?",
        #     "what is multithreading?",
        #     "what is multiprocessing?",
    ] * 5

    results = await asyncio.gather(
        *(rag_query(query, i) for i, query in enumerate(queries)),
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


if __name__ == "__main__":
    import multiprocessing.resource_tracker

    asyncio.run(run_c_req())
    # Clean up any remaining semaphores
    multiprocessing.resource_tracker._resource_tracker = None
