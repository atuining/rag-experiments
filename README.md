Files:
- qdrant-cloud.py: set up qdrant cloud vector db connection and functions to add files or directories to db
- api.py: functions and api to benchmark
- concurrent-requests.py: main benchmark file
- aync-queries.py and agent-rag-benchmark: smaller tests on only rag or single rag+agent query
- llmsherpa.py: testing llmsherpa library with good pdf parsing

To run:
- clone repo
- add api keys and urls to .env file (use .env.example to check which api keys to add)
- run the benchmark file
- install uv here: https://docs.astral.sh/uv/getting-started/installation/
```bash
  uv run concurrent-requests.py
```


TODO:

- [ ] Test different query response
  - [ ] Add a document to different local vector DB's
  - [ ] Set up pipeline to qualitatively test different techniques
  - [ ] Set up automatic testing
- [ ] Test latency
  - [ ] Deploy vector db to gcp
  - [ ] Test single query latency
  - [ ] Test concurrent latency

Different strategies to test:

- Pinecone/Milvus hybrid search
- graphRAG
- LLM generated metadata
- prune results from RAG
- recursive retreival
- dynamic retrieval for summary/comparison type questions     
- maybe fine tuning embeddings

Try LLMSherpa selfhost: Might be good in agentic retrieval to get a particular section hi there not hi theere buddy
Try on CAT question answers
