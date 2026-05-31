from typing import List, Tuple, Optional, Dict
from collections import deque
import re
import chromadb
from groq import Groq
from sentence_transformers import SentenceTransformer
from retrieval import hybrid_search, rerank
from config import (
    CHROMA_DB_PATH,
    COLLECTION_NAME,
    GROQ_API_KEY,
    GROQ_MODEL,
    EMBEDDING_MODEL,
    TOP_K_RESULTS,
    ENABLE_MEMORY,
    MEMORY_MAX_TURNS,
    ENABLE_QUERY_TRANSFORMATION,
)

def llm_generate(prompt: str, temperature: float = 0.3, max_tokens: int = 2048) -> str:
    import config
    client = Groq(api_key=config.GROQ_API_KEY)
    response = client.chat.completions.create(
        model=config.GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content

def llm_generate_stream(prompt: str, temperature: float = 0.3, max_tokens: int = 2048):
    import config
    client = Groq(api_key=config.GROQ_API_KEY)
    stream = client.chat.completions.create(
        model=config.GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta
        if delta.content:
            yield delta.content

_embedder: Optional[SentenceTransformer] = None

def _get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBEDDING_MODEL)
    return _embedder

def retrieve_chunks(query: str, k: int = TOP_K_RESULTS) -> List[Tuple[str, str, float]]:
    embedder = _get_embedder()
    query_embedding = embedder.encode([query]).tolist()[0]

    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    collection = client.get_collection(COLLECTION_NAME)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )

    if not results["documents"]:
        return []

    docs = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    scored_chunks = []
    for doc, meta, dist in zip(docs, metadatas, distances):
        score = 1.0 / (1.0 + dist) if dist is not None else 0.0
        source = meta["source"] if meta and "source" in meta else "unknown"
        scored_chunks.append((doc, source, round(score, 4)))

    return scored_chunks

def _build_prompt(query: str, chunks: List[Tuple[str, str, float]]) -> str:
    context_parts = []
    for i, (chunk, source, score) in enumerate(chunks, 1):
        context_parts.append(
            f"[Document {i}: {source} | Relevance: {score:.2%}]\n{chunk}"
        )
    context_str = "\n\n---\n\n".join(context_parts)

    prompt = f"""You are a precise document Q&A assistant. Answer the user's question based ONLY on the provided context below.

## Rules:
1. Answer ONLY using information from the context. Do NOT use your general knowledge.
2. If the context doesn't contain enough information, say: "I cannot find enough information in the provided documents to answer this question."
3. Cite the source document name when referencing specific information.
4. If the question is ambiguous, ask for clarification.
5. Be concise but thorough. Use bullet points for lists.

## Context:
{context_str}

## Question:
{query}

## Answer:
"""

    return prompt

def generate_answer(query: str, chunks: List[Tuple[str, str, float]]) -> str:
    prompt = _build_prompt(query, chunks)
    return llm_generate(prompt)

class ConversationMemory:
    def __init__(self, max_turns: int = MEMORY_MAX_TURNS):
        self.max_turns = max_turns
        self.history: List[Dict[str, str]] = []
    
    def add_turn(self, user_query: str, assistant_response: str):
        self.history.append({"user": user_query, "assistant": assistant_response})
        if len(self.history) > self.max_turns:
            self.history.pop(0)
    
    def rewrite_query(self, user_query: str) -> str:
        if not self.history:
            return user_query
        
        if len(user_query.split()) >= 10:
            return user_query
        
        history_text = ""
        for turn in self.history[-3:]:
            history_text += f"User: {turn['user']}\nAssistant: {turn['assistant']}\n\n"
        
        rewrite_prompt = f"""Given the conversation history, rewrite the user's 
latest question to be a self-contained, standalone question that doesn't 
need context from previous messages.

Conversation History:
{history_text}

Latest Question: {user_query}

Rewritten Question (only output the rewritten question, nothing else):"""
        
        try:
            rewritten = llm_generate(rewrite_prompt, temperature=0.0, max_tokens=256).strip()
            if len(rewritten) > 5 * len(user_query):
                return user_query
            return rewritten
        except Exception:
            return user_query
    
    def clear(self):
        self.history.clear()

_conversation_memory = ConversationMemory(max_turns=MEMORY_MAX_TURNS)

def _is_ambiguous(query: str) -> bool:
    pronouns = {"it", "they", "this", "that", "these", "those", 
                "he", "she", "him", "her", "there"}
    words = set(query.lower().split())
    return len(query.split()) < 10 or bool(words & pronouns)

def decompose_query(query: str) -> List[str]:
    prompt = f"""Break this complex question into 2-3 simple, independent sub-questions 
that each cover a single aspect. Return ONLY the sub-questions as a numbered list.

Question: {query}

Sub-questions:"""
    
    try:
        response_text = llm_generate(prompt, temperature=0.0, max_tokens=512)
        lines = response_text.strip().split("\n")
        sub_queries = []
        for line in lines:
            cleaned = re.sub(r'^[\d\-\.\s]+', '', line).strip()
            if cleaned:
                sub_queries.append(cleaned)
        return sub_queries if sub_queries else [query]
    except Exception:
        return [query]

def hyde_query(query: str) -> str:
    prompt = f"""You are given a question. Write a hypothetical paragraph 
that would be the PERFECT passage to answer this question. 
Write it as if it's from a textbook or Wikipedia article. Be specific 
and factual in tone.

Question: {query}

Hypothetical passage:"""
    
    try:
        return llm_generate(prompt, temperature=0.7, max_tokens=512).strip()
    except Exception:
        return query

def transform_query(query: str, history: List[Dict] = None) -> Tuple[str, str, List[str]]:
    word_count = len(query.split())
    
    if word_count > 20 and any(w in query.lower() for w in ["compare", "contrast", "difference", "pros", "cons", "advantages", "disadvantages", "and", "or"]):
        return ("decomposition", query, decompose_query(query))
    
    elif word_count < 5:
        hyde_doc = hyde_query(query)
        return ("hyde", hyde_doc, [query])
    
    elif history and _is_ambiguous(query):
        rewritten = _conversation_memory.rewrite_query(query)
        return ("rewrite", rewritten, [query])
    
    else:
        return ("direct", query, [query])

def ask_document(query: str, k: int = TOP_K_RESULTS, use_hybrid: bool = False, use_reranker: bool = False) -> dict:
    if use_hybrid:
        chunks = hybrid_search(query, k=20)
    else:
        chunks = retrieve_chunks(query, k=20)

    if use_reranker and len(chunks) > 1:
        chunks = rerank(query, chunks, top_k=k)
    else:
        chunks = chunks[:k]

    if not chunks:
        return {
            "answer": "No documents have been indexed yet. Please upload and process documents first.",
            "sources": [],
        }

    answer = generate_answer(query, chunks)

    sources = [
        {
            "content": chunk[:300] + ("..." if len(chunk) > 300 else ""),
            "file": source,
            "score": score,
        }
        for chunk, source, score in chunks
    ]

    return {
        "answer": answer,
        "sources": sources,
    }

def ask_document_stream(query: str, k: int = TOP_K_RESULTS, use_memory: bool = ENABLE_MEMORY, use_hybrid: bool = False, use_reranker: bool = False, use_transform: bool = ENABLE_QUERY_TRANSFORMATION):
    strategy = "direct"
    effective_query = query
    sub_queries = [query]
    
    if use_transform:
        history = _conversation_memory.history if use_memory else None
        strategy, effective_query, sub_queries = transform_query(query, history)
        yield ("strategy", strategy)
    elif use_memory:
        effective_query = _conversation_memory.rewrite_query(query)
        strategy = "rewrite"
        yield ("strategy", strategy)
    
    all_chunks = []
    
    if strategy == "decomposition":
        for sq in sub_queries:
            if use_hybrid:
                sq_chunks = hybrid_search(sq, k=10)
            else:
                sq_chunks = retrieve_chunks(sq, k=10)
            all_chunks.extend(sq_chunks)
            
        seen = set()
        unique_chunks = []
        for chunk in all_chunks:
            if chunk[0] not in seen:
                seen.add(chunk[0])
                unique_chunks.append(chunk)
        all_chunks = unique_chunks
    else:
        if use_hybrid:
            all_chunks = hybrid_search(effective_query, k=20)
        else:
            all_chunks = retrieve_chunks(effective_query, k=20)
        
    if use_reranker and len(all_chunks) > 1:
        chunks = rerank(query, all_chunks, top_k=k)
    else:
        chunks = all_chunks[:k]
    
    if not chunks:
        yield ("done", "No documents have been indexed yet. Please upload and process documents first.")
        return
        
    sources = [
        {
            "content": chunk[:300] + ("..." if len(chunk) > 300 else ""),
            "file": source,
            "score": score,
        }
        for chunk, source, score in chunks
    ]
    yield ("sources", sources)
    
    prompt = _build_prompt(effective_query, chunks)
    
    full_text = ""
    for text_chunk in llm_generate_stream(prompt):
        full_text += text_chunk
        yield ("token", text_chunk)
            
    if use_memory:
        _conversation_memory.add_turn(query, full_text)
        
    yield ("done", full_text)

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python query.py \"<your question>\"")
        sys.exit(1)

    question = " ".join(sys.argv[1:])
    result = ask_document(question)

    print(f"\nQ: {question}")
    print(f"\nA: {result['answer']}\n")
    print("--- Sources ---")
    for s in result["sources"]:
        print(f"  [File] {s['file']} (score: {s['score']:.2%})")
        print(f"     {s['content'][:100]}...")