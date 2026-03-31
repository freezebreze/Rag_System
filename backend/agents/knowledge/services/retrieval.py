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
        """Dense + BM25 双路混合检索（主要入口）"""
        if not collection:
            logger.warning("[Retrieval] collection 未指定，跳过检索")
            return []
        return get_milvus_service().hybrid_search(
            collection_name=collection,
            query=query,
            top_k=top_k,
            filter_expr=filter_expr,
        )

    def keyword_search(
        self,
        query: str,
        top_k: int = 10,
        filter_expr: Optional[str] = None,
        collection: Optional[str] = None,
        keyword_filter: Optional[str] = None,
    ) -> List[dict]:
        """TEXT_MATCH 倒排索引预过滤 + Dense + BM25 双路混合检索。
        keyword_filter 为空时退化为普通 hybrid_search。
        """
        if not collection:
            logger.warning("[Retrieval] collection 未指定，跳过检索")
            return []
        # 未指定关键词时用 query 本身作为关键词过滤
        kw = keyword_filter or query
        return get_milvus_service().hybrid_search(
            collection_name=collection,
            query=query,
            top_k=top_k,
            filter_expr=filter_expr,
            keyword_filter=kw,
        )

    def hybrid_search(
        self,
        query: str,
        top_k: int = 10,
        filter_expr: Optional[str] = None,
        collection: Optional[str] = None,
    ) -> List[dict]:
        """纯双路混合检索（同 vector_search）"""
        return self.vector_search(query=query, top_k=top_k, filter_expr=filter_expr, collection=collection)


_retrieval_service = None


def get_retrieval_service() -> RetrievalService:
    global _retrieval_service
    if _retrieval_service is None:
        _retrieval_service = RetrievalService()
    return _retrieval_service
