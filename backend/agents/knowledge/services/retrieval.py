# -*- coding: utf-8 -*-
"""
Retrieval Service - Milvus 混合检索
"""
from typing import List, Optional
import logging
from app.services.milvus_service import get_milvus_service

logger = logging.getLogger(__name__)


class RetrievalService:

    def vector_search(
        self,
        query: str,
        top_k: int = 10,
        filter_expr: Optional[str] = None,
        collection: Optional[str] = None,
    ) -> List[dict]:
        """Dense + BM25 混合检索（主要入口）"""
        if not collection:
            logger.warning("[Retrieval] collection 未指定，跳过检索")
            return []
        results = get_milvus_service().hybrid_search(
            collection_name=collection,
            query=query,
            top_k=top_k,
            filter_expr=filter_expr,
        )
        logger.info(f"[Retrieval] vector_search 返回 {len(results)} 个结果")
        return results

    def keyword_search(
        self,
        query: str,
        top_k: int = 10,
        filter_expr: Optional[str] = None,
        collection: Optional[str] = None,
    ) -> List[dict]:
        """纯 BM25 关键词检索（降级为 hybrid_search，BM25 权重更高）"""
        return self.vector_search(query=query, top_k=top_k, filter_expr=filter_expr, collection=collection)

    def hybrid_search(
        self,
        query: str,
        top_k: int = 10,
        filter_expr: Optional[str] = None,
        collection: Optional[str] = None,
    ) -> List[dict]:
        """混合检索（同 vector_search）"""
        return self.vector_search(query=query, top_k=top_k, filter_expr=filter_expr, collection=collection)


_retrieval_service = None


def get_retrieval_service() -> RetrievalService:
    global _retrieval_service
    if _retrieval_service is None:
        _retrieval_service = RetrievalService()
    return _retrieval_service
