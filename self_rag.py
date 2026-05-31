import json
from typing import List
from query import llm_generate, _build_prompt
from retrieval import hybrid_search, rerank
from config import TOP_K_RESULTS, MAX_SELF_RAG_ITERATIONS

def check_answer_against_context(answer: str, context_chunks: List[str]) -> dict:
    context_text = "\n\n".join(context_chunks)
    
    prompt = f"""You are a fact-checker. Your job is to determine if the given 
ANSWER is FULLY supported by the PROVIDED CONTEXT.

## Instructions:
1. Break the answer into individual factual claims.
2. For each claim, check if it appears in or can be directly inferred from the context.
3. If ANY claim is NOT supported, list those claims.

## Context:
{context_text}

## Answer:
{answer}

## Output format (JSON only, no other text):
{{
    "is_supported": true/false,
    "unsupported_claims": ["claim 1", "claim 2"],
    "reasoning": "brief explanation"
}}"""
    
    response_text = llm_generate(prompt, temperature=0.0, max_tokens=1024)
    
    try:
        text = response_text.strip()
        start = text.find('{')
        end = text.rfind('}') + 1
        if start >= 0 and end > start:
            return json.loads(text[start:end])
    except Exception:
        pass
    
    return {"is_supported": True, "unsupported_claims": [], "reasoning": "parse failed"}

def self_rag_query(query: str, k: int = TOP_K_RESULTS) -> dict:
    chunks = hybrid_search(query, k=20)
    chunks = rerank(query, chunks, top_k=k)
    
    if not chunks:
        return {
            "answer": "No documents have been indexed yet.",
            "sources": [],
            "verified": False,
            "iterations": 0,
            "corrections_made": False,
        }
    
    context_texts = [c[0] for c in chunks]
    iteration = 0
    all_contexts = list(context_texts)
    
    while iteration < MAX_SELF_RAG_ITERATIONS:
        iteration += 1
        
        prompt = _build_prompt(query, chunks)
        answer = llm_generate(prompt)
        
        check_result = check_answer_against_context(answer, all_contexts)
        
        if check_result.get("is_supported", False):
            return {
                "answer": answer,
                "sources": chunks,
                "verified": True,
                "iterations": iteration,
                "corrections_made": iteration > 1,
            }
        
        missing_info = ". ".join(check_result.get("unsupported_claims", []))
        if not missing_info:
            break
            
        new_chunks = hybrid_search(missing_info, k=5)
        
        existing_texts = set(all_contexts)
        for chunk in new_chunks:
            if chunk[0] not in existing_texts:
                all_contexts.append(chunk[0])
                existing_texts.add(chunk[0])
                chunks.append(chunk)
        
        chunks = rerank(query, chunks, top_k=k + 3)
    
    prompt = _build_prompt(query, chunks)
    answer = llm_generate(prompt)
    
    return {
        "answer": answer,
        "sources": chunks,
        "verified": False,
        "iterations": iteration,
        "corrections_made": True,
        "warning": "Some claims may not be fully supported by available documents.",
    }
