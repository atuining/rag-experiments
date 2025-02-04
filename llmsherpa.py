import os
from time import perf_counter
from IPython.display import display, HTML
import openai
from llama_index.core import (
    Document,
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
)

openai.api_key = os.environ.get("OPENAI_API_KEY")

llmsherpa_api_url = (
    "http://localhost:5010/api/parseDocument?renderFormat=all&useNewIndentParser=yes"
)
pdf_url = "time series ad literature review.pdf"
pdf_reader = LayoutPDFReader(llmsherpa_api_url)
print("Reading with PDFReader")
doc = pdf_reader.read_pdf(pdf_url)

selected_section = None
for section in doc.sections():
    # print(section.title)
    if section.title == "6 Distance-based Methods":
        selected_section = section
        break
print(selected_section.to_text(include_children=True, recurse=True))
# HTML(selected_section.to_html(include_children=True, recurse=True))

if not os.path.exists("./index"):
    index = VectorStoreIndex([])
    n = 0
    print("Reading into vector DB now")
    for chunk in doc.chunks():
        n += 1
        print(f"reading chunk number: {n}")
        index.insert(Document(text=chunk.to_context_text(), extra_info={}))
    print("Done reading into vector DB")
    index.storage_context.persist("./index")
else:
    index = load_index_from_storage(
        StorageContext.from_defaults(persist_dir="./index"),
    )
query_engine = index.as_query_engine()

start = perf_counter()
response = query_engine.query("how was the study organized?")
end = perf_counter()
print(f"time to query: {end - start}")
print(response)
