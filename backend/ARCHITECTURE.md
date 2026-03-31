# 后端架构文档

## 技术栈

- **框架**：FastAPI + Uvicorn
- **Agent 编排**：LangGraph（StateGraph）
- **LLM**：阿里云 DashScope（Qwen 系列模型）
- **向量数据库**：Milvus Standalone（Docker）
- **业务数据库**：PostgreSQL（Docker）
- **对象存储**：阿里云 OSS
- **Embedding**：DashScope text-embedding-v3 / text-embedding-v4（自调用，批量，支持自定义维度）

---

## 快速启动

```bash
# 1. 启动 Docker 服务（Milvus + PostgreSQL，首次约 30-60 秒）
docker-compose up -d

# 2. 配置环境变量
cd backend
cp .env.example .env   # 填写必填项见下方

# 3. 安装依赖
pip install -r requirements.txt

# 4. 启动后端（自动建表）
python main.py
```

### 必填环境变量

```env
DASHSCOPE_API_KEY=sk-xxxx
OSS_BUCKET=your-bucket
ALIBABA_CLOUD_ACCESS_KEY_ID=xxxx
ALIBABA_CLOUD_ACCESS_KEY_SECRET=xxxx
PG_HOST=localhost
PG_USER=kbuser
PG_PASSWORD=kbpass
MILVUS_HOST=localhost
```

---

## 目录结构

```
backend/
├── main.py
├── app/
│   ├── main.py                    # FastAPI app 工厂，注册路由、lifespan
│   ├── core/
│   │   ├── config.py              # Settings：所有配置，启动时校验必填项
│   │   ├── exceptions.py          # 统一业务异常
│   │   ├── logging.py             # 结构化日志
│   │   └── prompts.py             # LLM prompt 模板
│   ├── api/v1/
│   │   ├── chat.py                # POST /chat
│   │   ├── knowledge.py           # POST /knowledge（RAG 问答）
│   │   ├── documents.py           # 文档上传/切分/检索
│   │   ├── jobs.py                # 任务状态查询、手动向量化
│   │   ├── chunks.py              # 切片查看/编辑/清洗/撤回/向量化/图片管理
│   │   ├── categories.py          # 类目 CRUD
│   │   ├── files.py               # 知识库文件列表/删除
│   │   ├── system.py              # /health、/models
│   │   └── admin/
│   │       ├── collection.py      # 知识库 CRUD（Milvus + PG）
│   │       └── config.py          # 系统配置查询
│   ├── services/
│   │   ├── milvus_service.py      # Milvus：collection管理/upsert/hybrid_search
│   │   ├── embedding_service.py   # DashScope embedding，支持 dimension 参数
│   │   ├── chunk_splitter.py      # 纯文本切分（标准模式），中文友好
│   │   ├── document_service.py    # 文档上传业务逻辑
│   │   ├── job_service.py         # run_job_pipeline：chunking→chunked（手动触发向量化）
│   │   ├── chunk_service.py       # 切片编辑/清洗/撤回/向量化/图片管理
│   │   ├── file_service.py        # 知识库文件列表/联动删除（含 OSS 图片清理）
│   │   ├── category_service.py    # 类目 CRUD + 文件管理
│   │   ├── knowledge_service.py   # Knowledge RAG 业务逻辑
│   │   ├── chat_service.py        # Chat 业务逻辑
│   │   ├── oss_service.py         # OSS 文件上传/下载/删除
│   │   ├── doc_image_parser.py    # PyMuPDF 解析 PDF/DOCX，提取文字切片 + 图片
│   │   └── chunk_cleaner.py       # 切片清洗（正则 + LLM）
│   └── db/
│       ├── pg_client.py           # psycopg2 连接池
│       ├── init_db.py             # 建表脚本（幂等），启动时调用
│       ├── base_repository.py     # BaseRepository
│       ├── kb_repository.py       # knowledge_base 表
│       ├── file_repository.py     # knowledge_file 表
│       ├── job_repository.py      # knowledge_job 表
│       ├── chunk_repository.py    # knowledge_chunk + knowledge_chunk_origin 表
│       ├── chunk_image_repository.py  # knowledge_chunk_image 表
│       ├── category_repository.py
│       └── category_file_repository.py
└── agents/
    ├── knowledge/                 # Knowledge Agent（RAG 流水线）
    │   ├── services/retrieval.py  # RetrievalService：hybrid / keyword_filter+hybrid
    │   └── nodes/                 # query_rewrite/classify/retrieve/filter/generate...
    ├── supervisor/
    └── specialized/               # email/search（mock）
```

---

## 数据库表结构

### PostgreSQL

| 表名 | 用途 | 主键 |
|------|------|------|
| `knowledge_base` | 知识库配置（name/image_mode/embedding_model/vector_dim/metadata_fields） | UUID |
| `knowledge_category` | 类目（独立体系） | UUID |
| `knowledge_category_file` | 类目文件（OSS 引用） | UUID |
| `knowledge_file` | 知识库文件（kb_id + oss_key） | UUID |
| `knowledge_job` | 处理任务（状态机） | UUID |
| `knowledge_chunk` | 切片当前内容（可编辑） | UUID |
| `knowledge_chunk_origin` | 切片原始内容（只写一次，用于撤回） | UUID（FK→chunk） |
| `knowledge_chunk_image` | 切片图片（图文模式） | UUID |

`knowledge_base` 新增 `metadata_fields JSONB` 字段，存用户配置的元数据字段（key/type/fulltext/index/auto_inject）。

### 实体关系

```
knowledge_base
    └── knowledge_file (kb_id)
              └── knowledge_job (file_id, kb_id)
                        └── knowledge_chunk (job_id)  ← UUID 主键
                                  ├── knowledge_chunk_origin (chunk_id)
                                  └── knowledge_chunk_image (chunk_id)

knowledge_category
    └── knowledge_category_file (category_id)
```

### Milvus（向量数据）

每个知识库对应一个 Milvus collection，schema：

| 字段 | 类型 | 说明 |
|------|------|------|
| `chunk_id` | VARCHAR(36) PK | UUID，与 PG knowledge_chunk.id 对应 |
| `job_id` | VARCHAR | 关联 PG knowledge_job.id |
| `file_name` | VARCHAR | 文件名 |
| `chunk_index` | INT64 | 切片序号 |
| `content` | VARCHAR | 文本内容（enable_analyzer + enable_match，中文分析器） |
| `sparse_bm25` | SPARSE_FLOAT_VECTOR | BM25 Function 自动生成 |
| `dense` | FLOAT_VECTOR(dim) | DashScope embedding，dim 由知识库配置决定 |
| `[metadata_fields]` | VARCHAR | 用户配置的 fulltext 字段（如 title），显式声明 enable_match=True |
| dynamic fields | - | 其他 metadata |

索引：HNSW（dense）+ SPARSE_WAND（sparse_bm25）

---

## 核心业务流程

### 1. 文档切分流程（两阶段）

```
POST /documents/start-chunking/{category_id}?kb_name=xxx
  → 遍历 category_file → 写 knowledge_file + knowledge_job
  → BackgroundTask: run_job_pipeline()
      ├── chunking → 下载 OSS → 切分（图文/标准）
      │     └── 注入 auto_inject 元数据（如 title = 文件名前缀）
      └── chunked → 写 knowledge_chunk（UUID 主键）+ origin
          ← 停在此状态，等待人工审查后手动触发向量化

POST /chunks/job/{job_id}/upsert  （手动触发）
  → 读 PG chunks → embed（按 kb 的 embedding_model + vector_dim）
  → fulltext 字段拼接到 content 前（BM25 + dense 都感知）
  → upsert Milvus → mark_vectorized → done
```

### 2. RAG 问答检索策略

```
determine_retrieval_strategy:
  ├── KEYWORD_ONLY（含错误码/代码/精确匹配关键词）
  │     → TEXT_MATCH 预过滤 + dense + BM25 hybrid_search
  └── HYBRID（常规查询）
        → dense + BM25 hybrid_search（RRF 或 WeightedRanker）

hybrid_search 参数：
  - ranker: RRF（默认）| Weight（hybrid_alpha 控制 dense 权重）
  - keyword_filter: TEXT_MATCH(content, '关键词') 预过滤候选集
```

### 3. 完整 RAG 流水线

```
POST /knowledge
  → query_rewrite → query_classify → retrieval_strategy
    → [single_doc / multi_doc] retrieve
      → filter → rerank → relevance_filter（LLM）
        → generate_answer → quality_check → metrics
```

### 4. 删除联动

```
DELETE /files（file_id）
  → 查 OSS 图片 key → 删 OSS 图片
  → 删 Milvus 向量（by job_id）
  → 删 PG knowledge_file（CASCADE 删 job/chunk/origin/image）
```

---

## chunk_id 设计

- 格式：**纯 UUID**（`gen_random_uuid()`）
- PG `knowledge_chunk.id` 和 Milvus `chunk_id` 均为 UUID
- 定位切片用 `job_id + chunk_index` 组合查询，不依赖 chunk_id 编码位置信息
- 图文模式：`doc_image_parser` 在切分阶段为每个 chunk 生成 UUID，图片记录的 `chunk_id` 直接引用

---

## 元数据字段（metadata_fields）

创建知识库时可配置元数据字段，存储在 `knowledge_base.metadata_fields`：

```json
[{"key": "title", "type": "text", "fulltext": true, "index": false, "auto_inject": "filename_prefix"}]
```

- `fulltext: true`：在 Milvus schema 中显式声明该字段并开启 `enable_match=True`（倒排索引）
- `auto_inject: "filename_prefix"`：切分时自动将文件名前缀注入到每个 chunk 的 metadata
- upsert 时 fulltext 字段值拼接到 `content` 前面，BM25 和 dense embedding 均感知

---

## Embedding 配置

- 支持模型：`text-embedding-v3`（64/128/256/512/768/1024 维）、`text-embedding-v4`（1536/2048 维）
- 每个知识库独立配置 `embedding_model` + `vector_dim`，upsert 时按 kb 配置调用，并校验返回维度
- 全局默认值由 `.env` 的 `EMBEDDING_MODEL` / `EMBEDDING_DIMENSION` 控制

---

## API 路由

| 路径 | 说明 |
|------|------|
| `POST /api/v1/chat/` | 普通对话（Supervisor Agent） |
| `POST /api/v1/knowledge/` | RAG 问答 |
| `GET/POST /api/v1/admin/collections` | 知识库管理 |
| `POST /api/v1/documents/upload` | 单文件上传到知识库 |
| `POST /api/v1/documents/upload-to-category` | 上传文件到类目（OSS） |
| `POST /api/v1/documents/start-chunking/{category_id}` | 触发类目批量切分 |
| `POST /api/v1/documents/search` | 切片检索（支持 keyword_filter + hybrid） |
| `GET /api/v1/jobs` | 任务列表（需 kb_name） |
| `POST /api/v1/jobs/{job_id}/upsert` | 手动触发向量化 |
| `GET /api/v1/chunks/job/{job_id}` | 查看切片 |
| `POST /api/v1/chunks/job/{job_id}/upsert` | 切片向量化 |
| `GET /api/v1/files` | 文件列表（需 kb_name） |
| `DELETE /api/v1/files` | 删除文件（body: file_id） |
| `GET /api/v1/categories` | 类目列表 |
| `GET /api/v1/health` | 健康检查 |

---

## 异常处理

```
业务异常（AppError 子类）→ FastAPI exception_handler → 对应 HTTP 状态码
未处理异常 → 500
```

---

## 常见问题

**Q: Milvus 启动慢？**
首次启动需 30-60 秒初始化 etcd 和 MinIO，等 `docker-compose ps` 显示 healthy 再启后端。

**Q: embedding 报 Model access denied？**
检查 `DASHSCOPE_API_KEY` 是否开通了对应 embedding 模型的权限。

**Q: 向量维度不匹配？**
知识库创建时的 `vector_dim` 必须与 embedding 模型实际输出维度一致。已有 collection 需删除重建。

**Q: 图文模式支持哪些格式？**
仅 `.pdf` 和 `.docx`。标准模式支持 `.pdf`、`.doc`、`.docx`、`.txt`、`.md`、`.ppt`、`.pptx`。

**Q: 切分后 job 状态停在 chunked？**
正常，切分和向量化已解耦。在文件列表页选中文件点"上传向量库"手动触发。

**Q: Attu 端口冲突？**
`docker-compose.yml` 中 attu 映射的是宿主机 `8080:3000`，后端用 `8000`，不冲突。
