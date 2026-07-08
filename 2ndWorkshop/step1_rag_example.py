import os
from typing import List
import json
from pathlib import Path

import faiss
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "boilermaker_ta_data"


def _load_corpus():
    path = DATA_DIR / "knowledge_base.json"
    if not path.exists():
        return [
            {
                "title": "Fallback: RAG Intro",
                "text": "This is a fallback document. Add boilermaker_ta_data/knowledge_base.json for workshop data.",
            }
        ]
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


CORPUS = _load_corpus()


def build_retriever(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    embedder = SentenceTransformer(model_name)
    texts = [doc["text"] for doc in CORPUS]
    metadatas = [{"title": doc.get("title") or f"doc{i}"} for i, doc in enumerate(CORPUS)]
    embeddings = embedder.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    if embeddings.ndim == 1:
        embeddings = embeddings.reshape(1, -1)
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    return {
        "embedder": embedder,
        "index": index,
        "texts": texts,
        "metadatas": metadatas,
    }


def retrieve_docs(retriever, query: str, k: int = 3):
    q_emb = retriever["embedder"].encode(query, convert_to_numpy=True)
    if q_emb.ndim == 1:
        q_emb = q_emb.reshape(1, -1)
    faiss.normalize_L2(q_emb)
    scores, ids = retriever["index"].search(q_emb, min(k, len(retriever["texts"])))
    return [
        {
            "text": retriever["texts"][idx],
            "metadata": retriever["metadatas"][idx],
            "score": float(scores[0][i]),
        }
        for i, idx in enumerate(ids[0])
    ]


def make_prompt(query: str, context: str) -> str:
    return (
        "You are a RAG assistant. Use the retrieved context to answer the user query.\n\n"
        "Context:\n"
        + context
        + "\n\nQuestion:\n"
        + query
        + "\n\nAnswer in a concise, factual way."
    )


def get_llm():
    local_model_name = os.environ.get("LOCAL_MODEL", "tiiuae/falcon-7b-instruct")
    tokenizer = AutoTokenizer.from_pretrained(local_model_name)
    model_kwargs = {"device_map": "auto"}
    if torch.cuda.is_available():
        model_kwargs["torch_dtype"] = torch.bfloat16

    local_model = AutoModelForCausalLM.from_pretrained(local_model_name, **model_kwargs)
    return pipeline("text-generation", model=local_model, tokenizer=tokenizer, max_new_tokens=256)


def answer_queries(queries: List[str], llm, retriever):
    for query in queries:
        print(f"\n=== QUERY: {query}\n")
        docs = retrieve_docs(retriever, query, k=3)
        context = "\n\n".join([doc["text"] for doc in docs])
        prompt = make_prompt(query, context)
        output = llm(prompt, return_full_text=False)[0]["generated_text"]
        print("Answer:\n", output)
        print("Sources:\n", [doc["metadata"] for doc in docs])


def run_rag_example():
    retriever = build_retriever()
    llm = get_llm()

    queries = [
        "When are the office hours for this course?",
        "When are assignments due and what is the usual deadline?",
        "When are the exam weeks scheduled?",
        "What tools and resources does the course recommend for projects?",
        "When is the final project presentation scheduled?",
    ]

    answer_queries(queries, llm, retriever)


if __name__ == "__main__":
    run_rag_example()
