import json
import time
from typing import List
import pandas as pd
from self_rag import self_rag_query
from query import llm_generate

def load_test_set(path: str = "test_set.json") -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def evaluate_system(test_set: List[dict]) -> pd.DataFrame:
    results = []
    
    for item in test_set:
        start_time = time.time()
        result = self_rag_query(item["question"])
        latency = time.time() - start_time
        
        answer = result.get("answer", "")
        sources_texts = [s[0] for s in result.get("sources", [])]
        sources_files = [s[1] for s in result.get("sources", [])]
        
        faithfulness = _compute_faithfulness(answer, sources_texts)
        relevancy = _compute_relevancy(answer, item["question"])
        context_recall = _compute_context_recall(item["ground_truth"], sources_texts)
        hit = _compute_hit_rate(item.get("expected_sources", []), sources_files)
        
        results.append({
            "id": item["id"],
            "question": item["question"][:50] + ("..." if len(item["question"]) > 50 else ""),
            "faithfulness": faithfulness,
            "relevancy": relevancy,
            "context_recall": context_recall,
            "hit": hit,
            "latency_seconds": round(latency, 2),
            "verified": result.get("verified", False),
        })
    
    return pd.DataFrame(results)

def _compute_faithfulness(answer: str, contexts: List[str]) -> float:
    if not contexts:
        return 0.0
    prompt = f"""On a scale of 0.0 to 1.0, how faithful is this answer to the 
provided context? 1.0 = every claim is supported, 0.0 = no claims supported.

Context:
{chr(10).join(contexts[:3])}

Answer: {answer}

Score (just a number 0.0-1.0):"""
    try:
        text = llm_generate(prompt, temperature=0.0, max_tokens=64).strip()
        import re
        match = re.search(r'0\.\d+|1\.0', text)
        if match:
            return float(match.group())
        return 0.5
    except Exception:
        return 0.5

def _compute_relevancy(answer: str, question: str) -> float:
    prompt = f"""On a scale of 0.0 to 1.0, how relevant is this answer 
to the question? 1.0 = directly answers, 0.0 = completely irrelevant.

Question: {question}
Answer: {answer}

Score (just a number 0.0-1.0):"""
    try:
        text = llm_generate(prompt, temperature=0.0, max_tokens=64).strip()
        import re
        match = re.search(r'0\.\d+|1\.0', text)
        if match:
            return float(match.group())
        return 0.5
    except Exception:
        return 0.5

def _compute_context_recall(ground_truth: str, contexts: List[str]) -> float:
    gt_lower = ground_truth.lower()
    for ctx in contexts:
        gt_words = set(gt_lower.split())
        ctx_words = set(ctx.lower().split())
        if gt_words and ctx_words:
            overlap = len(gt_words & ctx_words) / len(gt_words)
            if overlap > 0.3:
                return 1.0
    return 0.0

def _compute_hit_rate(expected_sources: List[str], actual_sources: List[str]) -> float:
    if not expected_sources:
        return 1.0
    for exp in expected_sources:
        if any(exp in actual for actual in actual_sources):
            return 1.0
    return 0.0
