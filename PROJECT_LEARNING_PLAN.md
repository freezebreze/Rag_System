# Rag_System 项目学习规划与分析

## 1. 项目定位

这个仓库的主体是一个前后端分离的 RAG 系统：

- `backend/`：核心，负责 API、文档入库、切片管理、向量检索、LangGraph Agent、会话记忆
- `frontend/`：Vue 3 管理台和问答界面
- `knowledge-table/`：仓库里附带的另一个相对独立的子项目，短期内不用优先看
- `docs/`：补充文档

如果你现在是“开始学习这个项目”，建议主线只盯住 `backend/`，先把这 3 条链路看通：

1. 启动链路
2. 文档入库链路
3. RAG 问答链路

---

## 2. 我对这个项目的判断

这是一个“工程型 RAG 项目”，不是只做 demo 的最小原型。

你会同时接触到：

- FastAPI 路由层
- Service 业务层
- Repository 数据访问层
- PostgreSQL 业务数据
- Milvus 混合检索
- OSS 文件与图片存储
- LangGraph 多节点问答流程
- 会话持久化和多轮改写

它的难点不在单个文件，而在“链路长、模块多、状态多”。

所以学习方法不能按文件乱读，而要按调用链读。

---

## 3. 后端真实结构

### 3.1 目录职责

```text
backend/
├── app/
│   ├── main.py                  # FastAPI 真实入口
│   ├── api/v1/                  # 路由层
│   ├── core/                    # 配置、日志、异常、prompt、checkpointer
│   ├── db/                      # Repository + PG 连接池 + 建表
│   ├── models/                  # 请求/响应模型
│   └── services/                # 业务逻辑
├── agents/
│   ├── knowledge/               # RAG 主 Agent
│   ├── supervisor/              # 通用 supervisor
│   └── specialized/             # 专用 agent
└── requirements.txt
```

### 3.2 最重要的几个入口文件

- `backend/app/main.py`
- `backend/app/core/config.py`
- `backend/app/db/init_db.py`
- `backend/app/api/v1/knowledge.py`
- `backend/app/services/knowledge_service.py`
- `backend/agents/knowledge/graph.py`
- `backend/app/api/v1/documents.py`
- `backend/app/services/document_service.py`
- `backend/app/services/job_service.py`
- `backend/app/services/milvus_service.py`

---

## 4. 你应该先理解的三条主链路

### 4.1 启动链路

真实入口是：

`backend/app/main.py`

启动时会做几件关键事：

1. 加载 `Settings`
2. 初始化日志
3. 在 `lifespan` 里执行 `init_db()`
4. 初始化 LangGraph checkpointer
5. 注册 `/api/v1/*` 路由

你学这个项目，第一步不是去看 Agent，而是先搞清楚：

- 配置从哪里来
- 服务启动时自动初始化了什么
- FastAPI 路由是怎么挂进去的

---

### 4.2 文档入库链路

主入口：

`POST /api/v1/documents/upload`

真实调用链：

`documents.py`
-> `document_service.upload_document()`
-> 写 OSS / PG 文件记录 / job 记录
-> 后台任务 `_run_pipeline()`
-> `job_service.run_job_pipeline()`
-> 文本切分或图文解析
-> 写 `knowledge_chunk`
-> 状态停在 `chunked`
-> 手动调用 `/jobs/{job_id}/upsert` 或 `/chunks/job/{job_id}/upsert`
-> `job_service.upsert_job_to_milvus()`
-> `milvus_service.upsert_chunks()`

这条链路要重点理解两个设计：

- 切分和向量化是解耦的
- PG 存“源数据/可编辑切片”，Milvus 存“检索数据”

---

### 4.3 RAG 问答链路

主入口：

`POST /api/v1/knowledge/`

真实调用链：

`knowledge.py`
-> `knowledge_service.invoke_knowledge_qa()`
-> 读取知识库 `retrieval_config`
-> `create_initial_state()`
-> `agent.ainvoke(...)`
-> `agents/knowledge/graph.py`
-> `query_rewrite`
-> `query_classify`
-> `determine_retrieval_strategy`
-> `graph_retrieve`
-> `single_doc_retrieve` 或 `multi_doc_retrieve`
-> `filter_chunks`
-> `select_top_k_chunks`
-> `generate_answer`
-> `check_quality`
-> `finalize_metrics`

你学习这一段时，要重点盯 4 个问题：

1. 问题是怎么改写的
2. 何时走 single-doc，何时走 multi-doc
3. Milvus 检索结果怎么回填 PG 原始 chunk 内容
4. 对话历史为什么不用手动传，而是依赖 checkpointer

---

## 5. 当前项目的核心设计点

### 5.1 分层比较清晰

- API 层只负责收参与返回
- Service 层负责业务编排
- Repository 层负责 SQL
- Agent 层负责问答流程状态机

这是一个很适合练“后端分层思维”的项目。

### 5.2 数据是双存储

- PostgreSQL：业务真相、会话、chunk、图片占位符、配置
- Milvus：检索索引

学习时不要把 PG 和 Milvus 混在一起理解。

### 5.3 LangGraph 是问答主线

这个项目不是“API 里直接调用 LLM”，而是用 LangGraph 管理状态、节点和记忆。

所以你后面一定要理解：

- `KnowledgeAgentState`
- `RAGConfig`
- graph 编排
- checkpointer 的作用

### 5.4 图片模式是扩展复杂度来源

图文模式会引入：

- chunk 内图片占位符
- `knowledge_chunk_image`
- OSS 预签名 URL
- 前端渲染替换

第一轮学习建议先理解“纯文本链路”，再看图文模式。

---

## 6. 推荐学习顺序

### 第一阶段：先跑通系统，再读代码

目标：知道项目怎么启动、依赖什么。

建议顺序：

1. 读根目录 `README.md`
2. 读 `ARCHITECTURE.md`
3. 看 `docker-compose.yml`
4. 看 `backend/app/core/config.py`
5. 看 `backend/app/main.py`

学习产出：

- 画出系统依赖图：FastAPI / PG / Milvus / OSS / DashScope
- 写下必须的环境变量
- 确认真实启动命令

### 第二阶段：读数据层

目标：知道“数据存在哪里”。

建议顺序：

1. `backend/app/db/init_db.py`
2. `backend/app/db/pg_client.py`
3. `backend/app/db/base_repository.py`
4. `backend/app/db/kb_repository.py`
5. `backend/app/db/file_repository.py`
6. `backend/app/db/job_repository.py`
7. `backend/app/db/chunk_repository.py`
8. `backend/app/db/conversation_repository.py`

学习产出：

- 画出表关系图
- 搞清楚 `knowledge_base / file / job / chunk / conversation` 的关系

### 第三阶段：读文档入库链路

目标：知道文件怎样变成 chunk，再怎样进 Milvus。

建议顺序：

1. `backend/app/api/v1/documents.py`
2. `backend/app/services/document_service.py`
3. `backend/app/services/job_service.py`
4. `backend/app/services/chunk_splitter.py`
5. `backend/app/services/doc_image_parser.py`
6. `backend/app/services/chunk_service.py`
7. `backend/app/services/milvus_service.py`

学习产出：

- 画出上传后的状态流转图
- 搞清楚为什么要“先 chunk，再人工审查，再 upsert”

### 第四阶段：读问答主链路

目标：知道一个 query 如何变成答案。

建议顺序：

1. `backend/app/api/v1/knowledge.py`
2. `backend/app/services/knowledge_service.py`
3. `backend/agents/knowledge/state.py`
4. `backend/agents/knowledge/graph.py`
5. `backend/agents/knowledge/services/retrieval.py`
6. `backend/agents/knowledge/nodes/query_rewrite.py`
7. `backend/agents/knowledge/nodes/query_classify.py`
8. `backend/agents/knowledge/nodes/single_doc_retrieve.py`
9. `backend/agents/knowledge/nodes/multi_doc_retrieve.py`
10. `backend/agents/knowledge/nodes/generate.py`

学习产出：

- 画出 LangGraph 节点图
- 说明每个节点输入什么、输出什么

### 第五阶段：再看管理能力和前端联动

目标：知道整个系统如何被管理和使用。

建议顺序：

1. `backend/app/api/v1/admin/collection.py`
2. `backend/app/api/v1/chunks.py`
3. `backend/app/api/v1/jobs.py`
4. `backend/app/api/v1/conversations.py`
5. `frontend/src/views/DashboardView.vue`
6. `frontend/src/components/SimpleChat.vue`
7. `frontend/src/components/doc/*`

学习产出：

- 理解前端如何触发后端各条链路
- 知道哪些操作是后台异步任务

---

## 7. 7 天学习计划

### Day 1：整体认知

- 读 `README.md` 和 `ARCHITECTURE.md`
- 读 `docker-compose.yml`
- 读 `backend/app/main.py`
- 读 `backend/app/core/config.py`

目标：

- 说清项目用了哪些基础组件
- 说清服务启动时做了什么

### Day 2：数据库和数据模型

- 读 `backend/app/db/init_db.py`
- 读几个主要 repository
- 自己画 ER 图

目标：

- 说清每张核心表的职责

### Day 3：文档上传与切分

- 读 `documents.py`
- 读 `document_service.py`
- 读 `job_service.py`
- 读 `chunk_splitter.py`

目标：

- 说清上传后状态怎么变化

### Day 4：向量化和检索

- 读 `milvus_service.py`
- 读 `embedding_service.py`
- 读 `agents/knowledge/services/retrieval.py`

目标：

- 说清 hybrid search 怎么做
- 说清 PG 和 Milvus 数据为何要双写

### Day 5：LangGraph 主流程

- 读 `state.py`
- 读 `graph.py`
- 读 `query_rewrite.py`
- 读 `query_classify.py`

目标：

- 说清状态在节点间如何流动

### Day 6：答案生成和会话记忆

- 读 `generate.py`
- 读 `conversation_service.py`
- 读 `app/core/checkpointer.py`

目标：

- 说清历史消息存在哪
- 说清 checkpointer 和业务消息表的区别

### Day 7：自己跑一遍完整链路

建议你亲手做 1 次：

1. 创建知识库
2. 上传文档
3. 查看 job
4. 查看 chunk
5. 手动 upsert
6. 发起知识库问答
7. 查看会话消息

目标：

- 从“看懂代码”切到“能调试这个系统”

---

## 8. 实操建议

### 8.1 建议的学习方式

不要按目录从上往下硬读，建议每次只追一条链路：

- 一个 API 入口
- 对应一个 service
- 对应几个 repository
- 对应一个 agent 节点或 Milvus 调用

每读完一条链路，就自己回答 3 个问题：

1. 输入是什么
2. 中间状态写到哪里
3. 输出是什么

### 8.2 建议自己做的笔记

你最好自己维护 4 张图：

1. 系统组件图
2. 数据表关系图
3. 文档入库时序图
4. RAG 问答节点图

如果这 4 张图能讲清楚，这个项目你就已经入门了。

---

## 9. 目前值得你特别注意的几个“真实情况”

这些是我根据代码看到的、学习时要注意的点：

### 9.1 README 的启动说明和实际代码不完全一致

根目录 README 里写了：

`python main.py`

但 `backend/` 根目录下并没有 `main.py`。

真实入口是：

`backend/app/main.py`

更合理的启动方式应该是进入 `backend/` 后，用类似下面的命令启动：

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 9.2 上传文档后不会自动完成向量化

虽然一些文档文字描述像是“上传后后台处理”，但实际代码里：

- `run_job_pipeline()` 只做到切分写 PG
- 状态停在 `chunked`
- 后续需要手动调用 upsert 接口写入 Milvus

这点非常关键，学习和调试时不要误判。

### 9.3 `langgraph.json` 看起来是旧文件

`backend/langgraph.json` 指向的是：

`./my_agent/agent.py:agent`

但当前项目结构里没有这个路径，说明这份文件大概率不是现行主入口。

### 9.4 `knowledge-table/` 暂时可以不看

你当前目标是学这个项目主后端，优先级应该是：

`backend/ > frontend/ > knowledge-table/`

第一周不建议把 `knowledge-table/` 混进来，不然上下文会被打散。

### 9.5 有些说明文案比代码旧

例如系统根路由里还有 “mock retrieval” 之类的文案，但实际项目已经接入了真实的 Milvus 检索链路。

所以学习时以“代码实际行为”为准，不要只信文档描述。

---

## 10. 你现在最应该优先读的文件清单

如果只给你一份最小必读列表，我建议是：

1. `README.md`
2. `ARCHITECTURE.md`
3. `backend/app/main.py`
4. `backend/app/core/config.py`
5. `backend/app/db/init_db.py`
6. `backend/app/api/v1/documents.py`
7. `backend/app/services/document_service.py`
8. `backend/app/services/job_service.py`
9. `backend/app/services/milvus_service.py`
10. `backend/app/api/v1/knowledge.py`
11. `backend/app/services/knowledge_service.py`
12. `backend/agents/knowledge/state.py`
13. `backend/agents/knowledge/graph.py`
14. `backend/agents/knowledge/nodes/generate.py`

---

## 11. 我给你的最终建议

这个项目最适合你的学习目标不是“把所有代码看完”，而是分 3 次看通：

1. 先看“怎么启动、数据存哪”
2. 再看“文件怎么入库”
3. 最后看“问答怎么生成”

只要这三段打通，后面的 chunk 编辑、图片占位符、会话管理、知识图谱同步，都只是主链路上的扩展功能。

如果你愿意，下一步我可以继续帮你做两件事里的任意一个：

1. 给你画一版“后端学习脑图/模块关系图”
2. 按这个规划，继续带你逐文件讲解 `backend`，从入口开始讲第一轮
