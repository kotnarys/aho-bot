"""RAG service — loads corporate knowledge base into ChromaDB for retrieval."""

import os
import glob
import chromadb
from chromadb.config import Settings as ChromaSettings


KNOWLEDGE_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "knowledge")


def _chunk_markdown(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Split markdown by sections, then by chunk_size if section is too long."""
    sections = []
    current = []
    for line in text.split("\n"):
        if line.startswith("## ") and current:
            sections.append("\n".join(current))
            current = [line]
        else:
            current.append(line)
    if current:
        sections.append("\n".join(current))

    chunks = []
    for section in sections:
        if len(section) <= chunk_size:
            chunks.append(section.strip())
        else:
            words = section.split()
            buf = []
            buf_len = 0
            for w in words:
                buf.append(w)
                buf_len += len(w) + 1
                if buf_len >= chunk_size:
                    chunks.append(" ".join(buf).strip())
                    # overlap
                    keep = max(1, len(buf) * overlap // chunk_size)
                    buf = buf[-keep:]
                    buf_len = sum(len(x) + 1 for x in buf)
            if buf:
                chunks.append(" ".join(buf).strip())
    return [c for c in chunks if c]


class KnowledgeBase:
    def __init__(self):
        self._client = chromadb.Client(ChromaSettings(anonymized_telemetry=False))
        self._collection = self._client.get_or_create_collection(
            name="aho_knowledge",
            metadata={"hnsw:space": "cosine"},
        )
        self._loaded = False

    def load(self):
        if self._loaded:
            return
        knowledge_path = os.path.abspath(KNOWLEDGE_DIR)
        md_files = glob.glob(os.path.join(knowledge_path, "*.md"))
        all_chunks = []
        all_ids = []
        all_meta = []
        for fpath in md_files:
            fname = os.path.basename(fpath)
            with open(fpath, "r", encoding="utf-8") as f:
                text = f.read()
            chunks = _chunk_markdown(text)
            for i, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                all_ids.append(f"{fname}_{i}")
                all_meta.append({"source": fname})

        if all_chunks:
            self._collection.add(
                documents=all_chunks,
                ids=all_ids,
                metadatas=all_meta,
            )
        self._loaded = True

    def search(self, query: str, n_results: int = 3) -> list[str]:
        self.load()
        results = self._collection.query(query_texts=[query], n_results=n_results)
        docs = results.get("documents", [[]])[0]
        return docs


# Singleton
_kb: KnowledgeBase | None = None


def get_knowledge_base() -> KnowledgeBase:
    global _kb
    if _kb is None:
        _kb = KnowledgeBase()
        _kb.load()
    return _kb
