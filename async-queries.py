import asyncio
from time import perf_counter
from qdrant_cloud import get_qdrant_db


async def many_requests(n: int = 200):
    qdrant = get_qdrant_db()
    start = perf_counter()
    for i in range(n):
        s = perf_counter()
        await qdrant.ainvoke("exercises for greedy algorithms")
        e = perf_counter()
        print(f"time for query {i + 1} = {e - s}")
    end = perf_counter()
    print(f"time for all queries = {end - start}")


async def main():
    await many_requests()


if __name__ == "__main__":
    asyncio.run(main())
