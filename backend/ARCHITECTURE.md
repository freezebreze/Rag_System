# 后端架构文档

## 技术栈

- **框架**：FastAPI + Uvicorn
- **Agent 编排**：LangGraph（StateGraph）
- **LLM**：阿里云 DashScope（Qwen 系列模型）
- **向量数据库**：Milvus Standalone（Docker）
- **业务数据库**：PostgreSQL（Docker）
- **对象存储**：阿里云 OSS
- **Embedding**：DashScope text-embedding-v3（自调用，批量）

---

## 目录结构

```
backend/
├── main.py                        # 启动入口
├── app/
│   ├── main.py                    # FastAPI app 工厂，注册路由、lifespan（启动时 init_db）
│   ├── core/
│   │   ├── config.py              # Settings：Milvus/PG/OSS/Embedding/LLM 配置
│   │   ├── exceptions.py          # 统一业务异常体系
│   │   ├── logging.py             # 结构化日志
│   │   └── prompts.py             # LLM prompt 模板
│   ├── models/
│   │   ├── requests.py            # Pydantic 请求模型
│   │   └── responses.py           # Pydantic 响应模型
│   ├── api/v1/
│   │   ├── chat.py                # POST /chat
│   │   ├── knowledge.py           # POST /knowledge（RAG 问答）
│   │   ├── documents.py           # 文档上传、类目文件上传、切分触发
│   │   ├── jobs.py                # 任务状态查询、手动向量化
│   │   ├── chunks.py              # 切片查看/编辑/清洗/撤回/向量化/图片管理
│   │   ├── categories.py          # 类目 CRUD
│   │   ├── files.py               # 知识库文件列表/删除
│   │   ├── system.py              # /health、/models
│   │   └── admin/
│   │       ├── collection.py      # 知识库 CRUD（Milvus + PG）
│   │       └── config.py          # 系统配置查询
│   ├── services/
│   │   ├── milvus_service.py      # Milvus：collection管理/upsert/hybrid_search/delete
│   │   ├── embedding_service.py   # DashScope embedding，批量调用，限流重试
│   │   ├── chunk_splitter.py      # 纯文本切分（标准模式），中文友好
│   │   ├── document_service.py    # 文档上传业务逻辑，触发 BackgroundTask
│   │   ├── job_service.py         # run_job_pipeline：chunking→embedding→done
│   │   ├── chunk_service.py       # 切片编辑/清洗/撤回/向量化/图片管理
│   │   ├── file_service.py        # 知识库文件列表/联动删除
│   │   ├── category_service.py    # 类目 CRUD + 文件管理
│   │   ├── knowledge_service.py   # Knowledge RAG 业务逻辑
│   │   ├── chat_service.py        # Chat 业务逻辑
│   │   ├── oss_service.py         # OSS 文件上传/下载/删除
│   │   ├── doc_image_parser.py    # PyMuPDF 解析 PDF，提取文字切片 + 图片
│   │   └── chunk_cleaner.py       # 切片清洗（正则 + LLM）
│   └── db/
│       ├── pg_client.py           # psycopg2 连接池，参数化查询
│       ├── init_db.py             # 建表脚本（幂等），启动时调用
│       ├── base_repository.py     # BaseRepository：封装 pg_client 调用
│       ├── kb_repository.py       # knowledge_base 表
│       ├── file_repository.py     # knowledge_file 表
│       ├── job_repository.py      # knowledge_job 表（自维护状态机）
│       ├── chunk_repository.py    # knowledge_chunk + knowledge_chunk_origin 表
│       ├── chunk_image_repository.py  # knowledge_chunk_image 表
│       ├── category_repository.py # knowledge_category 表
│       └── category_file_repository.py # knowledge_category_file 表
└── agents/
    ├── knowledge/                 # Knowledge Agent（RAG 流水线）
    │   ├── services/retrieval.py  # RetrievalService：调 milvus_service.hybrid_search
    │   └── nodes/                 # 各节点（query_rewrite/classify/retrieve/filter/generate...）
    ├── supervisor/                # Supervisor Agent（多 agent 协调）
    └── specialized/               # 专用 Agent（email/search，mock）
```

---

## 数据库表结构

### PostgreSQL（业务数据）

| 表名 | 用途 | 主键 |
|------|------|------|
| `knowledge_base` | 知识库配置（名称/image_mode/embedding_model） | UUID |
| `knowledge_category` | 类目（文件夹，独立体系） | UUID |
| `knowledge_category_file` | 类目文件（OSS 引用） | UUID |
| `knowledge_file` | 知识库文件（kb_id + oss_key，可追溯来源类目） | UUID |
| `knowledge_job` | 处理任务（状态机：pending→chunking→chunked→embedding→done/error） | UUID |
| `knowledge_chunk` | 切片当前内容（可编辑） | TEXT（`{job_id}_{chunk_index}`） |
| `knowledge_chunk_origin` | 切片原始内容（只写一次，用于撤回） | TEXT（FK→chunk） |
| `knowledge_chunk_image` | 切片图片（图文模式） | UUID |

### 实体关系

```
knowledge_base
    └── knowledge_file (kb_id, category_file_id?)
              └── knowledge_job (file_id, kb_id)
                        └── knowledge_chunk (job_id)
                                  ├── knowledge_chunk_origin (chunk_id)
                                  └── knowledge_chunk_image (chunk_id)

knowledge_category
    └── knowledge_category_file (category_id)
```

### Milvus（向量数据）

每个知识库对应一个 Milvus collection，schema：

| 字段 | 类型 | 说明 |
|------|------|------|
| `chunk_id` | VARCHAR PK | `{job_id}_{chunk_index}` |
| `job_id` | VARCHAR | 关联 PG knowledge_job.id |
| `file_name` | VARCHAR | 文件名 |
| `chunk_index` | INT64 | 切片序号 |
| `content` | VARCHAR | 文本内容（启用中文分析器 + BM25） |
| `sparse_bm25` | SPARSE_FLOAT_VECTOR | BM25 Function 自动生成 |
| `dense` | FLOAT_VECTOR(1536) | DashScope embedding |
| dynamic fields | - | metadata 字段 |

索引：HNSW（dense）+ SPARSE_WAND（sparse_bm25）

---

## 核心业务流程

### 1. 文档上传流程

```
POST /documents/upload (kb_name, file)
  → validate_file()
  → OSS 上传（kb/{kb_name}/{file_name}）
  → 写 knowledge_file(status=pending)
  → 写 knowledge_job(status=pending)
  → 立即返回 job_id
  → BackgroundTask: run_job_pipeline()
      ├── status=chunking → 下载 OSS → 切分（图文/标准）
      ├── status=chunked  → 写 knowledge_chunk + knowledge_chunk_origin
      ├── status=embedding → embed → upsert Milvus
      └── status=done / error
```

### 2. 类目批量切分流程

```
POST /documents/start-chunking/{category_id}?kb_name=xxx
  → 遍历 knowledge_category_file
  → 每个文件：写 knowledge_file + knowledge_job
  → 触发 BackgroundTask（同上流水线）
```

### 3. RAG 问答流程

```
POST /knowledge
  → knowledge_agent.invoke()
      query_rewrite → query_classify → determine_retrieval_strategy
        → [single_doc_retrieve 或 multi_doc_retrieve]
            调 milvus_service.hybrid_search()（dense + BM25 + RRF）
          → [filter_chunks → rerank_chunks]（multi_doc 路径）
            → relevance_filter（LLM 二次过滤）
              → generate_answer（图文模式时查 chunk_image）
                → check_quality → finalize_metrics
```

### 4. 切片编辑后重新向量化

```
POST /chunks/job/{job_id}/upsert
  → 读 knowledge_chunk（current content）
  → embed → upsert Milvus（覆盖旧向量）
  → mark_vectorized
```

---

## chunk_id 设计

- 格式：`{job_id}_{chunk_index}`（job_id 为 UUID）
- 同时作为 PG `knowledge_chunk.id` 主键和 Milvus `chunk_id` 主键
- 图文模式：parse_pdf/parse_word 阶段生成，should_merge 后序号可能有空洞
- 图片查询：通过 `knowledge_chunk.chunk_index` 查出真实 `chunk_id`，再查 `knowledge_chunk_image`

---

## 异常处理

```
业务异常（AppError 子类）→ FastAPI exception_handler → 对应 HTTP 状态码
未处理异常 → 500
```

---

## 配置管理

所有配置通过 `app/core/config.py` 的 `Settings` 类读取环境变量，启动时 `_validate_env()` 校验必填项。

必填项：`DASHSCOPE_API_KEY`, `OSS_BUCKET`, `OSS_ACCESS_KEY_*`, `PG_HOST`, `PG_USER`, `PG_PASSWORD`, `MILVUS_HOST`
