from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.chroma import ChromaVectorStore
from src.retrieve import hybrid_retrieve
import chromadb
import yaml

def load_prompt(path="prompts/qa_prompt.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def get_answer(query):
    # Load prompt config
    prompt_cfg = load_prompt()

    # Load LLM and embedding model
    llm = Ollama(model="llama3", request_timeout=120.0)
    embed_model = OllamaEmbedding(model_name="nomic-embed-text")

    # Load existing ChromaDB index
    db = chromadb.PersistentClient(path="./chroma_db")
    collection = db.get_or_create_collection("rag_docs")
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_vector_store(
        vector_store, embed_model=embed_model
    )

    # Get nodes for BM25
    from src.ingest import build_index
    _, nodes = build_index()

    # Hybrid retrieve + rerank
    retriever = index.as_retriever(similarity_top_k=10)
    top_nodes = hybrid_retrieve(query, retriever, nodes)

    # Citation enforcement
    if not top_nodes:
        return "I could not find relevant information to answer this question.", []

    # Build context with citations
    context = ""
    citations = []
    for i, node in enumerate(top_nodes):
        context += f"[{i+1}] {node.text}\n\n"
        citations.append({
            "id": i+1,
            "source": node.metadata.get("file_name", "unknown"),
            "page": node.metadata.get("page_label", "?")
        })

    # Build prompt from yaml config
    prompt = prompt_cfg["template"].format(context=context, query=query)

    # Generate answer
    response = llm.complete(prompt)
    answer = str(response)

    # Check if answer is grounded
    if "i don't know" in answer.lower() or "not mentioned" in answer.lower():
        return answer, []

    return answer, citations

if __name__ == "__main__":
    query = input("Ask a question: ")
    answer, citations = get_answer(query)
    print("\n--- Answer ---")
    print(answer)
    print("\n--- Sources ---")
    for c in citations:
        print(f"[{c['id']}] {c['source']} (page {c['page']})")