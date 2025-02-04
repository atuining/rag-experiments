import asyncio
import time
import os
from api import rag_query, get_prompt, get_llm

# Disable LangSmith tracking and progress bars
os.environ["LANGCHAIN_TRACING_V2"] = "false"


async def run_c_req():
    # Initialize resources
    prompt = await get_prompt()
    llm = await get_llm()

    queries = [
        "what is concurrency?",
        "what is parallelism?",
        "what is async programming?",
        "what is multithreading?",
        "what is multiprocessing?",
    ] * 2

    start_time = time.perf_counter()
    try:
        results = await asyncio.gather(
            *(rag_query(query, i, prompt, llm) for i, query in enumerate(queries)),
            return_exceptions=True,
        )
    finally:
        # Ensure resources are properly cleaned up
        del llm

    end_time = time.perf_counter()

    # Print results and any errors
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            print(f"Request {i} failed: {result}")
        else:
            print(f"Request {i} succeeded")

    print(f"Total time: {end_time - start_time} seconds")
    return results


if __name__ == "__main__":
    asyncio.run(run_c_req())
    time.sleep(0.1)
