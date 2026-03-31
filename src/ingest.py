from llama_index.core import VectorStoreIndex, StorageContext, Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
import pypdf
import os

def load_documents(data_dir="data"):
    docs = []
    for fname in os.listdir(data_dir):
        fpath = os.path.join(data_dir, fname)
        if fname.endswith(".pdf"):
            reader = pypdf.PdfReader(fpath)
            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                if text and text.strip():
                    docs.append(Document(
                        text=text,
                        metadata={"file_name": fname, "page_label": str(i+1)}
                    ))
        elif fname.endswith(".txt"):
            with open(fpath, "r") as f:
                docs.append(Document(
                    text=f.read(),
                    metadata={"file_name": fname, "page_label": "1"}
                ))
    return docs

def build_index(data_dir="data"):
    embed_model = OllamaEmbedding(model_name="nomic-embed-text")

    docs = load_documents(data_dir)

    splitter = SentenceSplitter(chunk_size=650, chunk_overlap=100)
    nodes = splitter.get_nodes_from_documents(docs)

    db = chromadb.PersistentClient(path="./chroma_db")
    collection = db.get_or_create_collection("rag_docs")
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex(
        nodes,
        embed_model=embed_model,
        storage_context=storage_context
    )

    print(f"Indexed {len(nodes)} chunks from {len(docs)} pages")
    return index, nodes

if __name__ == "__main__":
    build_index()