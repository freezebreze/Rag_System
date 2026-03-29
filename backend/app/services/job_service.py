# -*- coding: utf-8 -*-
"""
任务业务逻辑
run_job_pipeline：切分 → 写 chunk → embedding → 写 Milvus → done
"""
import asyncio
import logging
from typing import Optional

from app.core.exceptions import NotFoundError
from app.db import get_job_repository, get_chunk_repository, get_file_repository, get_chunk_image_repository

logger = logging.getLogger(__name__)


# ── 查询 ──────────────────────────────────────────────────────────────────────

def list_jobs(kb_name: str, limit: int = 200) -> dict:
    from app.db import get_kb_repository
    kb = get_kb_repository().get_by_name(kb_name)
    if not kb:
        return {"jobs": [], "total": 0}
    jobs = get_job_repository().list_by_kb(kb["id"], limit=limit)
    return {"jobs": jobs, "total": len(jobs)}


def get_job_detail(job_id: str) -> dict:
    job = get_job_repository().get(job_id)
    if not job:
        raise NotFoundError("任务不存在")
    return {"job": job}


# ── 流水线 ────────────────────────────────────────────────────────────────────

async def run_job_pipeline(
    job_id: str,
    file_id: str,
    kb_id: str,
    kb_name: str,
    file_name: str,
    oss_key: str,
    image_mode: bool,
    chunk_size: int,
    chunk_overlap: int,
    image_dpi: int,
) -> None:
    """
    完整流水线：
    pending → chunking → chunked → embedding → done
    任何阶段失败 → error
    """
    job_repo = get_job_repository()
    file_repo = get_file_repository()

    try:
        # ── Step 1: 切分 ──────────────────────────────────────────────────────
        job_repo.update_status(job_id, "chunking", stage="正在切分文档")
        file_repo.update_status(file_id, "processing")

        file_content = await asyncio.to_thread(_download_file, oss_key)

        if image_mode:
            chunks, image_records = await asyncio.to_thread(
                _parse_image_mode,
                file_content=file_content,
                job_id=job_id,
                kb_name=kb_name,
                file_name=file_name,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                image_dpi=image_dpi,
            )
        else:
            chunks = await asyncio.to_thread(
                _parse_text_mode,
                file_content=file_content,
                file_name=file_name,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            image_records = []

        # ── Step 2: 写 chunk 到 PG ────────────────────────────────────────────
        chunk_repo = get_chunk_repository()
        if image_mode:
            await asyncio.to_thread(chunk_repo.bulk_insert_with_ids, job_id, file_name, chunks)
        else:
            await asyncio.to_thread(chunk_repo.bulk_insert, job_id, file_name, chunks)

        if image_records:
            await asyncio.to_thread(get_chunk_image_repository().bulk_insert, image_records)

        chunk_count = len(chunks)
        job_repo.update_status(job_id, "chunked", stage="切分完成，准备向量化", chunk_count=chunk_count, progress=50)

        # ── Step 3: Embedding + 写 Milvus ─────────────────────────────────────
        job_repo.update_status(job_id, "embedding", stage="正在生成向量并写入 Milvus", progress=60)

        # 从 PG 读取切片（含 chunk_id）
        pg_chunks = await asyncio.to_thread(chunk_repo.get_by_job, job_id)
        milvus_chunks = [
            {
                "chunk_id":    c["chunk_id"],
                "job_id":      job_id,
                "file_name":   file_name,
                "chunk_index": c["chunk_index"],
                "content":     c["current_content"],
                "metadata":    c.get("metadata") or {},
            }
            for c in pg_chunks if c.get("current_content")
        ]

        from app.services.milvus_service import get_milvus_service
        milvus_svc = get_milvus_service()
        # 确保 collection 存在
        from app.db import get_kb_repository
        kb = get_kb_repository().get_by_id(kb_id)
        milvus_svc.get_or_create_collection(
            kb_name,
            dim=kb["vector_dim"] if kb else 1536,
            image_mode=image_mode,
        )
        await asyncio.to_thread(milvus_svc.upsert_chunks, kb_name, milvus_chunks)

        # ── Step 4: 标记完成 ──────────────────────────────────────────────────
        job_repo.mark_vectorized(job_id)
        file_repo.update_status(file_id, "done")
        logger.info(f"[pipeline] job_id={job_id} 完成，共 {chunk_count} 个切片")

    except Exception as e:
        logger.error(f"[pipeline] job_id={job_id} 失败: {e}")
        job_repo.update_status(job_id, "error", stage="处理失败", error_msg=str(e))
        file_repo.update_status(file_id, "error", error_msg=str(e))


# ── 手动触发向量化（切片编辑后重新上传）─────────────────────────────────────

async def upsert_job_to_milvus(job_id: str) -> dict:
    """将已切分的 job 重新向量化写入 Milvus（用于切片编辑后手动触发）"""
    job_repo = get_job_repository()
    job = job_repo.get(job_id)
    if not job:
        raise NotFoundError("任务不存在")

    from app.db import get_kb_repository, get_file_repository
    file_record = get_file_repository().get_by_id(job["file_id"])
    kb = get_kb_repository().get_by_id(job["kb_id"])
    if not file_record or not kb:
        raise NotFoundError("文件或知识库不存在")

    chunk_repo = get_chunk_repository()
    pg_chunks = chunk_repo.get_by_job(job_id)
    if not pg_chunks:
        raise NotFoundError("该 job 暂无切片数据")

    milvus_chunks = [
        {
            "chunk_id":    c["chunk_id"],
            "job_id":      job_id,
            "file_name":   file_record["file_name"],
            "chunk_index": c["chunk_index"],
            "content":     c["current_content"],
            "metadata":    c.get("metadata") or {},
        }
        for c in pg_chunks if c.get("current_content")
    ]

    from app.services.milvus_service import get_milvus_service
    milvus_svc = get_milvus_service()
    milvus_svc.get_or_create_collection(kb["name"], dim=kb["vector_dim"])
    result = await asyncio.to_thread(milvus_svc.upsert_chunks, kb["name"], milvus_chunks)
    job_repo.mark_vectorized(job_id)
    return result


# ── 内部：文件下载 ────────────────────────────────────────────────────────────

def _download_file(oss_key: str) -> bytes:
    from app.services.oss_service import get_oss_service
    return get_oss_service().get_object_bytes(oss_key)


# ── 内部：图文模式切分 ────────────────────────────────────────────────────────

def _parse_image_mode(
    file_content: bytes,
    job_id: str,
    kb_name: str,
    file_name: str,
    chunk_size: int,
    chunk_overlap: int,
    image_dpi: int,
):
    from app.services.doc_image_parser import parse_pdf, parse_word
    ext = file_name.lower().rsplit(".", 1)[-1]
    if ext == "pdf":
        return parse_pdf(
            file_content=file_content,
            job_id=job_id,
            collection=kb_name,
            file_name=file_name,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            image_dpi=image_dpi,
        )
    elif ext in ("docx", "doc"):
        return parse_word(
            file_content=file_content,
            job_id=job_id,
            collection=kb_name,
            file_name=file_name,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    else:
        raise ValueError(f"图文模式不支持格式: {ext}")


# ── 内部：标准模式切分 ────────────────────────────────────────────────────────

def _parse_text_mode(
    file_content: bytes,
    file_name: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list:
    """
    标准模式：提取文本 → chunk_splitter 切分
    支持 PDF / DOCX / TXT / MD
    """
    from app.services.chunk_splitter import split_text_with_metadata

    ext = file_name.lower().rsplit(".", 1)[-1]
    text = _extract_text(file_content, ext, file_name)
    return split_text_with_metadata(
        text=text,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        base_metadata={"file_name": file_name, "source": ext},
    )


def _extract_text(file_content: bytes, ext: str, file_name: str) -> str:
    if ext == "pdf":
        import fitz  # PyMuPDF
        doc = fitz.open(stream=file_content, filetype="pdf")
        return "\n\n".join(page.get_text() for page in doc)
    elif ext in ("docx", "doc"):
        import io
        from docx import Document
        doc = Document(io.BytesIO(file_content))
        return "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())
    elif ext in ("txt", "md"):
        for enc in ("utf-8", "gbk", "utf-16"):
            try:
                return file_content.decode(enc)
            except UnicodeDecodeError:
                continue
        return file_content.decode("utf-8", errors="ignore")
    else:
        # 尝试 UTF-8 文本
        return file_content.decode("utf-8", errors="ignore")
