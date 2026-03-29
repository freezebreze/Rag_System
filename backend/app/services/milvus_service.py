# -*- coding: utf-8 -*-
"""
Milvus 向量数据库服务
负责 collection 管理、向量写入、混合检索、删除
"""
import logging
from typing import Any, Dict, List, Optional

from pymilvus import (
    MilvusClient,
    DataType,
    Function,
    FunctionType,
    AnnSearchRequest,
    RRFRanker,
)

from app.core.config import settings
from app.services.embedding_service import get_embedding_service

logger = logging.getLogger(__name__)

_ANALYZER_PARAMS = {"type": "chinese"}


class MilvusService:

    def __init__(self):
        uri = f"http://{settings.milvus_host}:{settings.milvus_port}"
        token = f"{settings.milvus_user}:{settings.milvus_password}"
        self.client = MilvusClient(uri=uri, token=token)
        logger.info(f"Milvus 连接成功: {uri}")

    # ── Collection 管理 ───────────────────────────────────────────────────────

    def get_or_create_collection(
        self,
        collection_name: str,
        dim: int = 1536,
        image_mode: bool = False,
    ) -> None:
        """幂等建 collection，schema 含 dense + BM25 + 业务字段"""
        if self.client.has_collection(collection_name):
            logger.info(f"Collection 已存在: {collection_name}")
            return

        schema = MilvusClient.create_schema(enable_dynamic_field=True)
        schema.add_field("chunk_id",    DataType.VARCHAR, max_length=256, is_primary=True)
        schema.add_field("job_id",      DataType.VARCHAR, max_length=128)
        schema.add_field("file_name",   DataType.VARCHAR, max_length=512)
        schema.add_field("chunk_index", DataType.INT64)
        schema.add_field(
            "content", DataType.VARCHAR, max_length=65535,
            enable_analyzer=True,
            analyzer_params=_ANALYZER_PARAMS,
            enable_match=True,
        )
        schema.add_field("sparse_bm25", DataType.SPARSE_FLOAT_VECTOR)
        schema.add_field("dense",       DataType.FLOAT_VECTOR, dim=dim)

        schema.add_function(Function(
            name="bm25",
            function_type=FunctionType.BM25,
            input_field_names=["content"],
            output_field_names="sparse_bm25",
        ))

        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="dense",
            index_name="dense_hnsw",
            index_type="HNSW",
            metric_type="IP",
            params={"M": 16, "efConstruction": 200},
        )
        index_params.add_index(
            field_name="sparse_bm25",
            index_name="sparse_bm25_idx",
            index_type="SPARSE_WAND",
            metric_type="BM25",
        )

        self.client.create_collection(
            collection_name=collection_name,
            schema=schema,
            index_params=index_params,
        )
        logger.info(f"Collection 创建成功: {collection_name}")

    def delete_collection(self, collection_name: str) -> None:
        if self.client.has_collection(collection_name):
            self.client.drop_collection(collection_name)
            logger.info(f"Collection 已删除: {collection_name}")

    def list_collections(self) -> List[str]:
        return self.client.list_collections()

    def has_collection(self, collection_name: str) -> bool:
        return self.client.has_collection(collection_name)

    # ── 数据写入 ──────────────────────────────────────────────────────────────

    def upsert_chunks(
        self,
        collection_name: str,
        chunks: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        批量 upsert 切片到 Milvus。
        chunks: [{"chunk_id", "job_id", "file_name", "chunk_index", "content", "metadata"}, ...]
        内部调用 EmbeddingService 生成 dense 向量。
        """
        if not chunks:
            return {"upsert_count": 0}

        embedding_svc = get_embedding_service()
        texts = [c["content"] for c in chunks]
        vectors = embedding_svc.embed_texts(texts)

        if len(vectors) != len(chunks):
            raise RuntimeError(f"Embedding 数量不匹配: {len(vectors)} vs {len(chunks)}")

        data = []
        for chunk, vec in zip(chunks, vectors):
            row: Dict[str, Any] = {
                "chunk_id":    chunk["chunk_id"],
                "job_id":      chunk.get("job_id", ""),
                "file_name":   chunk.get("file_name", ""),
                "chunk_index": int(chunk.get("chunk_index", 0)),
                "content":     chunk["content"],
                "dense":       vec,
            }
            for k, v in (chunk.get("metadata") or {}).items():
                if k not in row:
                    row[k] = v
            data.append(row)

        batch_size = 200
        total = 0
        for i in range(0, len(data), batch_size):
            batch = data[i: i + batch_size]
            res = self.client.upsert(collection_name=collection_name, data=batch)
            total += res.get("upsert_count", len(batch))

        logger.info(f"[Milvus] upsert {total} 条到 {collection_name}")
        return {"upsert_count": total}

    # ── 删除 ──────────────────────────────────────────────────────────────────

    def delete_by_job(self, collection_name: str, job_id: str) -> None:
        if not self.client.has_collection(collection_name):
            return
        self.client.delete(
            collection_name=collection_name,
            filter=f'job_id == "{job_id}"',
        )
        logger.info(f"[Milvus] 删除 job_id={job_id} from {collection_name}")

    def delete_by_file(self, collection_name: str, file_name: str) -> None:
        if not self.client.has_collection(collection_name):
            return
        escaped = file_name.replace('"', '\\"')
        self.client.delete(
            collection_name=collection_name,
            filter=f'file_name == "{escaped}"',
        )
        logger.info(f"[Milvus] 删除 file_name={file_name} from {collection_name}")

    def delete_by_chunk_ids(self, collection_name: str, chunk_ids: List[str]) -> None:
        if not chunk_ids or not self.client.has_collection(collection_name):
            return
        ids_str = ", ".join(f'"{cid}"' for cid in chunk_ids)
        self.client.delete(
            collection_name=collection_name,
            filter=f"chunk_id in [{ids_str}]",
        )

    # ── 检索 ──────────────────────────────────────────────────────────────────

    def hybrid_search(
        self,
        collection_name: str,
        query: str,
        top_k: int = 10,
        filter_expr: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Dense + BM25 混合检索，RRF 融合排序"""
        if not self.client.has_collection(collection_name):
            logger.warning(f"[Milvus] collection 不存在: {collection_name}")
            return []

        query_vector = get_embedding_service().embed_query(query)

        # AnnSearchRequest 用 expr 参数（不是 filter）
        dense_kwargs: Dict[str, Any] = {
            "data": [query_vector],
            "anns_field": "dense",
            "param": {"metric_type": "IP", "params": {"ef": 100}},
            "limit": top_k,
        }
        if filter_expr:
            dense_kwargs["expr"] = filter_expr

        bm25_kwargs: Dict[str, Any] = {
            "data": [query],
            "anns_field": "sparse_bm25",
            "param": {"metric_type": "BM25", "params": {"drop_ratio_search": 0.2}},
            "limit": top_k,
        }
        if filter_expr:
            bm25_kwargs["expr"] = filter_expr

        output_fields = ["chunk_id", "job_id", "file_name", "chunk_index", "content"]

        try:
            results = self.client.hybrid_search(
                collection_name=collection_name,
                reqs=[AnnSearchRequest(**dense_kwargs), AnnSearchRequest(**bm25_kwargs)],
                ranker=RRFRanker(k=60),
                limit=top_k,
                output_fields=output_fields,
            )
        except Exception as e:
            logger.error(f"[Milvus] hybrid_search 失败: {e}")
            raise

        hits = []
        for hit in (results[0] if results else []):
            entity = hit.get("entity", {})
            hits.append({
                "chunk_id":    entity.get("chunk_id") or hit.get("id"),
                "job_id":      entity.get("job_id", ""),
                "file_name":   entity.get("file_name", ""),
                "chunk_index": entity.get("chunk_index", 0),
                "content":     entity.get("content", ""),
                "score":       hit.get("distance", 0.0),
                "metadata":    {
                    k: v for k, v in entity.items()
                    if k not in ("chunk_id", "job_id", "file_name", "chunk_index", "content")
                },
            })

        logger.info(f"[Milvus] hybrid_search 返回 {len(hits)} 条")
        return hits

    def vector_search(
        self,
        collection_name: str,
        query: str,
        top_k: int = 10,
        filter_expr: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """纯 dense 向量检索（备用）"""
        if not self.client.has_collection(collection_name):
            return []

        query_vector = get_embedding_service().embed_query(query)
        search_kwargs: Dict[str, Any] = {
            "collection_name": collection_name,
            "data": [query_vector],
            "anns_field": "dense",
            "search_params": {"metric_type": "IP", "params": {"ef": 100}},
            "limit": top_k,
            "output_fields": ["chunk_id", "job_id", "file_name", "chunk_index", "content"],
        }
        if filter_expr:
            search_kwargs["filter"] = filter_expr

        results = self.client.search(**search_kwargs)
        hits = []
        for hit in (results[0] if results else []):
            entity = hit.get("entity", {})
            hits.append({
                "chunk_id":    entity.get("chunk_id") or hit.get("id"),
                "job_id":      entity.get("job_id", ""),
                "file_name":   entity.get("file_name", ""),
                "chunk_index": entity.get("chunk_index", 0),
                "content":     entity.get("content", ""),
                "score":       hit.get("distance", 0.0),
                "metadata":    {},
            })
        return hits


_instance: Optional[MilvusService] = None


def get_milvus_service() -> MilvusService:
    global _instance
    if _instance is None:
        _instance = MilvusService()
    return _instance
