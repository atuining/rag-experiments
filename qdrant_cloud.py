import os
import asyncio
from dotenv import load_dotenv
from uuid import uuid4
from time import perf_counter
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader, PyPDFDirectoryLoader
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()
url = os.environ.get("QDRANT_URL")
api_key = os.environ.get("QDRANT_API_KEY")

embedding_instance = None


def get_embedding_func() -> OpenAIEmbeddings:
    global embedding_instance
    if embedding_instance is None:
        embedding_instance = OpenAIEmbeddings(model="text-embedding-3-small", )
    return embedding_instance


def get_qdrant_db():
    vectorstore = QdrantVectorStore.from_existing_collection(
        embedding=get_embedding_func(),
        collection_name="dsa_collection",
        url=url,
        api_key=api_key,
    )
    return vectorstore.as_retriever(k=3)


async def upload_dir(path):
    loader = PyPDFDirectoryLoader(path)
    pages = []
    start = perf_counter()
    async for page in loader.alazy_load():
        page.metadata["primary_key"] = uuid4()
        pages.append(page)
    end = perf_counter()
    print(f"time to load directory: {end - start}")

    start = perf_counter()
    qdrant = QdrantVectorStore.from_documents(
        documents=pages,
        embedding=get_embedding_func(),
        collection_name="dsa_collection",
        url=url,
        api_key=api_key,
    )
    # qdrant.add_documents(pages)
    end = perf_counter()
    print(f"time to upload dir: {end - start}")


async def upload_doc(path):
    loader = PyPDFLoader(path)
    pages = []
    start = perf_counter()
    async for page in loader.alazy_load():
        pages.append(page)
    end = perf_counter()
    print(f"time to load pdf = {end - start}")

    text_splitter = RecursiveCharacterTextSplitter()
    docs = text_splitter.split_documents(pages)

    for chunk in docs:
        chunk.metadata["primary_key"] = uuid4()

    start = perf_counter()
    # # add to new collection
    # QdrantVectorStore.from_documents(
    #     documents=docs,
    #     embedding=get_embedding_func(),
    #     collection_name="dsa_collection",
    #     url=url,
    #     api_key=api_key,
    # )
    qdrant = QdrantVectorStore.from_existing_collection(
        embedding=get_embedding_func(),
        collection_name="dsa_collection",
        url=url,
        api_key=api_key,
    )
    qdrant.add_documents(docs)
    end = perf_counter()
    print(f"time to upload= {end - start}")


async def main():
    await upload_doc("time series ad literature review.pdf")
    # await upload_dir("books")
    retriever = get_qdrant_db()
    start = perf_counter()
    docs = retriever.invoke("what are b+ trees?")
    end = perf_counter()
    print(f"time to answer query = {end - start}")
    print(docs)


if __name__ == "__main__":
    asyncio.run(main())
