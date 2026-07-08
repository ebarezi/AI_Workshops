import json
import os
import re
from pathlib import Path

import faiss
from fastmcp import FastMCP
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "boilermaker_ta_data"
ANNOUNCEMENTS_FILE = BASE_DIR / "announcements.txt"


app = FastMCP(
    name="boilermaker-ta",
    instructions=(
        "Provides file-system tools for finding, reading, organizing, and "
        "summarizing academic PDF papers. Use find_documents to discover PDFs, "
        "read_pdf to extract their text, the file management tools to organize "
        "them into categorized directories, and send_email to notify a recipient "
        "with the summary and original PDF attached."
    ),
)


def _load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


class SimpleRetriever:
    def __init__(self, texts, metadatas, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.embedder = SentenceTransformer(model_name)
        self.texts = texts
        self.metadatas = metadatas
        self.index = self._build_index(texts)

    def _build_index(self, texts):
        embeddings = self.embedder.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        faiss.normalize_L2(embeddings)
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        return index

    def similarity_search(self, query: str, k: int = 3):
        q_emb = self.embedder.encode(query, convert_to_numpy=True)
        if q_emb.ndim == 1:
            q_emb = q_emb.reshape(1, -1)
        faiss.normalize_L2(q_emb)
        scores, ids = self.index.search(q_emb, min(k, len(self.texts)))
        return [
            type("Doc", (), {"page_content": self.texts[idx], "metadata": self.metadatas[idx]})
            for idx in ids[0]
        ]


def _build_retriever():
    """Build a small FAISS retriever from the local knowledge base.

    This is intentionally simple for workshop use; for production cache the
    vectorstore to avoid rebuilding on every call.
    """
    kb = _load_json(DATA_DIR / "knowledge_base.json")
    texts = [doc["text"] for doc in kb]
    metadatas = [{"title": doc.get("title")} for doc in kb]
    return SimpleRetriever(texts, metadatas)


def _build_calendar_retriever():
    """Build a FAISS retriever from the academic calendar events.

    Converts calendar events into text format for embedding-based retrieval.
    """
    calendar = _load_json(DATA_DIR / "purdue_calendar.json")
    events = calendar.get("events", [])
    if not events:
        return None

    texts = [f"{e['date']} {e['title']} {e['notes']}" for e in events]
    metadatas = [{"date": e.get("date"), "title": e.get("title")} for e in events]
    return SimpleRetriever(texts, metadatas)


@app.tool
def search_knowledge_base(query: str) -> str:
    """Search a small local knowledge base of course and schedule facts."""
    kb = _load_json(DATA_DIR / "knowledge_base.json")
    query_terms = re.findall(r"\w+", query.lower())
    scored = []
    for doc in kb:
        text = doc["text"].lower()
        score = sum(text.count(term) for term in query_terms)
        if score > 0:
            scored.append((score, doc))
    if not scored:
        return "No relevant knowledge found in the course knowledge base. Try a different question."

    scored.sort(key=lambda item: item[0], reverse=True)
    top_docs = scored[:2]
    output = []
    for score, doc in top_docs:
        output.append(f"{doc['title']}:\n{doc['text']}")
    return "\n\n".join(output)


@app.tool
def retrieve_docs(query: str, k: int = 3) -> str:
    """Retrieve top-k documents from the knowledge base using embeddings + FAISS.

    Returns a human-readable string with titles and excerpts for the top matches.
    """
    try:
        retriever = _build_retriever()
    except Exception as e:
        return f"Failed to build retriever: {e}"

    docs = retriever.similarity_search(query, k=k)
    if not docs:
        return "No documents returned by retriever."

    lines = []
    for d in docs:
        title = d.metadata.get("title") if d.metadata else "(no title)"
        excerpt = (d.page_content[:400] + "...") if len(d.page_content) > 400 else d.page_content
        lines.append(f"{title}: {excerpt}")
    return "\n\n".join(lines)


@app.tool
def get_academic_calendar(query: str = "next 30 days") -> str:
    """Retrieve calendar events using embeddings + FAISS.

    Matches user queries against calendar events using semantic similarity,
    demonstrating the same RAG approach used in retrieve_docs.
    """
    try:
        retriever = _build_calendar_retriever()
    except Exception as e:
        return f"Failed to build calendar retriever: {e}"
    
    if retriever is None:
        return "Academic calendar is empty."

    docs = retriever.similarity_search(query, k=5)
    if not docs:
        return "No calendar events matched your query."

    lines = []
    for d in docs:
        date = d.metadata.get("date") if d.metadata else "(no date)"
        title = d.metadata.get("title") if d.metadata else "(no title)"
        lines.append(f"{date} - {title}")
    
    return "Academic calendar events:\n" + "\n".join(lines)


@app.tool
def create_notification(subject: str, body: str) -> str:
    """Write a notification to announcements.txt and return its location."""
    ANNOUNCEMENTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    timestamp = Path(ANNOUNCEMENTS_FILE).stat().st_mtime if ANNOUNCEMENTS_FILE.exists() else 0
    entry = f"Subject: {subject}\n{body}\n---\n"
    with open(ANNOUNCEMENTS_FILE, "a", encoding="utf-8") as f:
        f.write(entry)

    return f"Notification written to {ANNOUNCEMENTS_FILE}."


if __name__ == "__main__":
    
    
    # Start the MCP server with HTTP transport. This makes it accessible
    # over the network at http://<host>:<port>/mcp — the agent connects
    # to this URL to discover and call tools.
    #
    # HTTP transport (aka "Streamable HTTP") is the recommended transport
    # for production MCP servers. The alternative is stdio (where the
    # client spawns the server as a subprocess), but HTTP is better for
    # our workshop because:
    #   1. You can see the server running in its own terminal
    #   2. Multiple clients could connect (useful for debugging)
    #   3. It mirrors real-world MCP deployments
    host = os.environ.get("MCP_HOST", "127.0.0.1")
    port = int(os.environ.get("MCP_PORT", "8001"))
    app.run(transport="http", host=host, port=port)
