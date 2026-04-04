# -*- coding: utf-8 -*-
"""
Top-K Select Node - 按分数排序并取 top-K（multi_doc 路径专用）

single_doc 路径经 Milvus RRF 融合后直接进入 generate，不经过此节点。
注意：此处是基于已有 RRF 分数的排序截断，不是 cross-encoder rerank。
"""

from typing import Dict, Any
from datetime import datetime

from ..state import KnowledgeAgentState


def _score(chunk) -> float:
    """兼容 RetrievedChunk dataclass 和 dict 两种格式取分数"""
    if isinstance(chunk, dict):
        return chunk.get("rerank_score") or chunk.get("score", 0.0) or 0.0
    return getattr(chunk, "rerank_score", None) or getattr(chunk, "score", 0.0) or 0.0


def select_top_k_chunks(state: KnowledgeAgentState) -> Dict[str, Any]:
    """
    对 filtered_chunks 按分数降序排列，取 top llm_context_top_k（multi_doc 路径）
    结果写入 reranked_chunks 和 merged_chunks（供 generate_answer 读取）
    """
    filtered_chunks = state["filtered_chunks"]
    config = state["config"]
    top_k = config.llm_context_top_k

    print(f"\n[TopKSelect] Sorting {len(filtered_chunks)} chunks, top_k={top_k}")

    try:
        sorted_chunks = sorted(filtered_chunks, key=_score, reverse=True)
        top_chunks = sorted_chunks[:top_k]

        print(f"[TopKSelect] Selected top {len(top_chunks)} chunks")

        metrics = state["metrics"]
        metrics.chunks_after_rerank = len(top_chunks)

        return {
            "reranked_chunks": top_chunks,
            "merged_chunks": top_chunks,
            "metrics": metrics,
            "processing_log": [{
                "stage": "select_top_k",
                "timestamp": datetime.now().isoformat(),
                "chunks_in": len(filtered_chunks),
                "chunks_out": len(top_chunks),
            }]
        }

    except Exception as e:
        print(f"[TopKSelect] Error: {e}")
        return {"all_errors": [f"Top-K selection failed: {e}"]}
