import os
import re
from datetime import datetime
from typing import List, Optional, Tuple
import chromadb
from sentence_transformers import SentenceTransformer
from config import (
    CHROMA_DB_PATH,
    COLLECTION_NAME,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    MERGE_THRESHOLD,
    EMBEDDING_MODEL,
    SUPPORTED_EXTENSIONS,
)

def parse_pdf(file_path: str) -> str:
    from pypdf import PdfReader
    reader = PdfReader(file_path)
    pages = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text)
    return "\n\n--- PAGE BREAK ---\n\n".join(pages)

def parse_docx(file_path: str) -> str:
    from docx import Document
    doc = Document(file_path)
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    return "\n".join(paragraphs)

def parse_txt(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()

PARSER_MAP = {
    ".pdf": parse_pdf,
    ".docx": parse_docx,
    ".txt": parse_txt,
    ".md": parse_txt,
}

def get_parser(file_path: str):
    ext = os.path.splitext(file_path)[1].lower()
    if ext not in PARSER_MAP:
        raise ValueError(
            f"Unsupported file type '{ext}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )
    return PARSER_MAP[ext]

def _word_count(text: str) -> int:
    return len(text.split())

def _greedy_merge(segments: List[str], chunk_size: int) -> List[str]:
    if not segments:
        return []

    chunks: List[str] = []
    buffer_words: List[str] = []

    for seg in segments:
        seg_words = seg.split()
        if buffer_words and len(buffer_words) + len(seg_words) > chunk_size:
            chunks.append(" ".join(buffer_words))
            buffer_words = seg_words
        else:
            buffer_words.extend(seg_words)

    if buffer_words:
        chunks.append(" ".join(buffer_words))

    return chunks

def _merge_small_chunks(chunks: List[str], chunk_size: int,
                        merge_threshold: float) -> List[str]:
    if not chunks:
        return []

    min_size = max(1, int(chunk_size * merge_threshold))
    result: List[str] = []
    buffer = ""

    for chunk in chunks:
        if buffer:
            combined = buffer + " " + chunk
            if _word_count(buffer) < min_size:
                buffer = combined
            else:
                result.append(buffer)
                buffer = chunk
        else:
            buffer = chunk

    if buffer:
        result.append(buffer)

    return result

def _add_overlap(chunks: List[str], overlap: int) -> List[str]:
    if len(chunks) <= 1 or overlap <= 0:
        return chunks

    result = [chunks[0]]
    for i in range(1, len(chunks)):
        prev_tail = " ".join(chunks[i - 1].split()[-overlap:])
        result.append(prev_tail + " " + chunks[i])

    return result

def _split_text_recursive(text: str, chunk_size: int,
                          depth: int = 0) -> Tuple[List[str], str]:
    LEVELS = [
        (r"\n\n+",                         "paragraph"),
        (r"\n",                             "line"),
        (r"(?<=[.!?])[ \t]+(?=[A-Z\'\"])",  "sentence"),
    ]

    text = text.strip()
    if not text:
        return [], "empty"

    if _word_count(text) <= chunk_size:
        return [text], LEVELS[min(depth, len(LEVELS) - 1)][1] if depth < len(LEVELS) else "word"

    if depth >= len(LEVELS):
        words = text.split()
        segments = []
        i = 0
        while i < len(words):
            segments.append(" ".join(words[i:i + chunk_size]))
            i += chunk_size
        return segments, "word"

    pattern, label = LEVELS[depth]
    parts = [p.strip() for p in re.split(pattern, text) if p.strip()]

    if len(parts) <= 1:
        return _split_text_recursive(text, chunk_size, depth + 1)

    final_segments: List[str] = []
    for part in parts:
        if _word_count(part) > chunk_size:
            sub_segs, _ = _split_text_recursive(part, chunk_size, depth + 1)
            final_segments.extend(sub_segs)
        else:
            final_segments.append(part)

    return final_segments, label

def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
    merge_threshold: float = MERGE_THRESHOLD,
) -> List[Tuple[str, str]]:
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()

    if not text:
        return []

    if _word_count(text) <= chunk_size:
        return [(text, "single")]

    raw_segments, chunk_type = _split_text_recursive(text, chunk_size)

    if not raw_segments:
        return []

    merged = _greedy_merge(raw_segments, chunk_size)
    merged = _merge_small_chunks(merged, chunk_size, merge_threshold)
    with_overlap = _add_overlap(merged, overlap)

    return [(c.strip(), chunk_type) for c in with_overlap if c.strip()]

_embedder: Optional[SentenceTransformer] = None

def get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBEDDING_MODEL)
    return _embedder

def _get_chroma_collection(replace: bool = False):
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    if replace:
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass
    return client.get_or_create_collection(COLLECTION_NAME)

def process_file(file_path: str, replace: bool = False) -> int:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    parser = get_parser(file_path)
    text = parser(file_path)
    filename = os.path.basename(file_path)
    print(f"[PARSE] Parsed '{filename}' -> {len(text)} characters")

    chunk_tuples = chunk_text(text)
    print(f"[CHUNK] Split into {len(chunk_tuples)} chunks")

    if not chunk_tuples:
        print(f"[WARN] No text content found in '{filename}'")
        return 0

    chunks = [c for c, _ in chunk_tuples]
    chunk_types = [t for _, t in chunk_tuples]

    embedder = get_embedder()
    embeddings = embedder.encode(chunks, show_progress_bar=True).tolist()
    print(f"[EMBED] Generated {len(embeddings)} embeddings")

    collection = _get_chroma_collection(replace=replace)
    ids = [f"{filename}_chunk_{i}" for i in range(len(chunks))]
    metadatas = [
        {
            "source": filename,
            "chunk_index": i,
            "total_chunks": len(chunks),
            "char_count": len(chunks[i]),
            "section_hint": chunks[i][:60].replace("\n", " ").strip(),
            "chunk_type": chunk_types[i],
        }
        for i in range(len(chunks))
    ]

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=chunks,
        metadatas=metadatas,
    )
    print(f"[STORE] Stored {len(chunks)} chunks in ChromaDB collection '{COLLECTION_NAME}'")

    return len(chunks)

def list_indexed_files() -> List[str]:
    try:
        client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        collection = client.get_or_create_collection(COLLECTION_NAME)
        results = collection.get(include=["metadatas"])
        sources = set()
        for meta in results["metadatas"]:
            if meta and "source" in meta:
                sources.add(meta["source"])
        return sorted(sources)
    except Exception:
        return []

def delete_collection():
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    try:
        client.delete_collection(COLLECTION_NAME)
        return True
    except (ValueError, Exception):
        return False

def get_chunk_count() -> int:
    try:
        client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        collection = client.get_or_create_collection(COLLECTION_NAME)
        return collection.count()
    except Exception:
        return 0

def get_file_chunks(filename: str) -> int:
    try:
        client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        collection = client.get_or_create_collection(COLLECTION_NAME)
        results = collection.get(where={"source": filename})
        return len(results["ids"]) if results and "ids" in results else 0
    except Exception:
        return 0

def delete_file(filename: str) -> int:
    try:
        client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        collection = client.get_or_create_collection(COLLECTION_NAME)
        results = collection.get(where={"source": filename})
        if results and "ids" in results and results["ids"]:
            collection.delete(ids=results["ids"])
            return len(results["ids"])
        return 0
    except Exception:
        return 0

def get_collection_stats() -> dict:
    try:
        client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        collection = client.get_or_create_collection(COLLECTION_NAME)
        all_data = collection.get(include=["metadatas"])
        
        sources = {}
        if all_data and "metadatas" in all_data and all_data["metadatas"]:
            for meta in all_data["metadatas"]:
                if meta and "source" in meta:
                    src = meta["source"]
                    sources[src] = sources.get(src, 0) + 1
                    
        return {
            "total_chunks": len(all_data["ids"]) if all_data and "ids" in all_data else 0,
            "total_documents": len(sources),
            "per_file": sources,
            "last_indexed": datetime.now().isoformat() if sources else None,
        }
    except Exception:
        return {
            "total_chunks": 0,
            "total_documents": 0,
            "per_file": {},
            "last_indexed": None,
        }

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python ingest.py <file_path> [--replace]")
        sys.exit(1)

    file_path = sys.argv[1]
    replace = "--replace" in sys.argv

    try:
        chunks = process_file(file_path, replace=replace)
        print(f"\nSuccess! {chunks} chunks indexed.")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
