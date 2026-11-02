<div align="center">
  <h1>CookWise AI</h1>
  <p><sub>知识图谱&nbsp;·&nbsp;混合检索&nbsp;·&nbsp;查询路由&nbsp;·&nbsp;多轮烹饪助手</sub></p>
  <p>
    <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&amp;logo=python&amp;logoColor=white" alt="Python 3.11"></a>
    <a href="https://fastapi.tiangolo.com/"><img src="https://img.shields.io/badge/FastAPI-0.115+-009688?style=flat-square&amp;logo=fastapi&amp;logoColor=white" alt="FastAPI"></a>
    <a href="https://neo4j.com/"><img src="https://img.shields.io/badge/Neo4j-5.11-4581C3?style=flat-square&amp;logo=neo4j&amp;logoColor=white" alt="Neo4j 5.11"></a>
    <a href="https://milvus.io/"><img src="https://img.shields.io/badge/Milvus-2.3.2-00A1EA?style=flat-square&amp;logo=milvus&amp;logoColor=white" alt="Milvus 2.3.2"></a>
    <a href="https://nextjs.org/"><img src="https://img.shields.io/badge/Next.js-14-000000?style=flat-square&amp;logo=nextdotjs&amp;logoColor=white" alt="Next.js 14"></a>
  </p>
  <p><a href="README.md">English</a>&nbsp;&nbsp;·&nbsp;&nbsp;<b>简体中文</b></p>
</div>

## 一、项目背景与目标

### 1.1 从传统RAG到图RAG的演进

传统RAG依靠向量检索和父子分块，能够回答简单的菜谱查询，但在强关系、多约束的烹饪领域仍有明显局限：

- **关系理解缺失**：无法显式建模菜谱、食材、步骤、烹饪方法之间的语义关系
- **跨文档关联困难**：难以发现菜谱之间的相似性、替代关系等隐含联系
- **推理能力有限**：缺乏基于知识图谱的多跳推理，难以回答需要组合条件的问题

### 1.2 本项目RAG系统的核心优势

本项目以 Neo4j 知识图谱为核心，叠加向量检索、BM25 与查询路由，构建可落地的智能烹饪助手系统：

- **结构化知识表达**：图结构显式编码实体与关系，同时保留菜谱原文
- **混合检索**：图检索、Milvus 向量检索、BM25 并行召回，RRF 融合
- **智能查询路由**：按意图选择证据引擎，支持多意图拆分与并行检索
- **可控回答**：置信度门控、证据校验与降级策略，证据不足时明确告知
- **工程化能力**：多轮会话、流式输出、语义缓存、Redis 限流与反馈日志

## 二、环境配置

> 以下命令均在项目根目录执行；远程部署时把 `localhost` 换成服务器 IP。

### 2.1 创建虚拟环境

```bash
conda create -n cookwise-ai python=3.11
conda activate cookwise-ai
```

### 2.2 安装核心依赖

```bash
pip install -r requirements.txt
```

> 只跑离线单测时安装 `requirements-test.txt`，不需要 torch、Neo4j 或 Milvus。

### 2.3 Neo4j数据库配置

Neo4j 由根目录的 [`docker-compose.yml`](docker-compose.yml) 管理，镜像为 `neo4j:5.11-community`，已启用 APOC 与 GDS。

#### 2.3.1 启动Neo4j服务

```bash
docker compose up -d neo4j
docker compose ps
```

#### 2.3.2 访问Neo4j Web界面

- **Web界面**：http://localhost:7474
- **用户名**：neo4j
- **密码**：all-in-rag

#### 2.3.3 数据导入

`data/cypher` 已挂载到容器 `/var/lib/neo4j/import`，`neo4j-init` 会在数据库为空时自动导入：

```bash
docker compose up neo4j-init
```

导入菜谱、食材、步骤节点及它们之间的关系，数据来自 `data/cypher/nodes.csv` 与 `relationships.csv`；数据库非空时自动跳过。

需要手动重新导入时（脚本按 `nodeId` MERGE，可重复执行）：

```bash
docker exec -it cookwise-ai-neo4j cypher-shell -u neo4j -p all-in-rag \
  -f /var/lib/neo4j/import/neo4j_import.cypher
```

### 2.4 Milvus向量数据库配置

Milvus Standalone 及依赖的 etcd、MinIO 同样在 `docker-compose.yml` 中定义。

#### 2.4.1 启动Milvus服务

```bash
docker compose up -d milvus-etcd milvus-minio milvus-standalone
docker compose ps
```

#### 2.4.2 构建向量索引

后端首次启动检测到集合 `cooking_knowledge` 不存在时，会自动用 BGE-small-zh-v1.5 构建索引。需要强制重建时：

```bash
python scripts/rebuild_index.py
```

#### 2.4.3 验证安装

```bash
docker compose ps                 # milvus-standalone 应为 healthy
curl -s localhost:9091/healthz    # 返回 OK
```

Milvus 端口 `19530`；MinIO 控制台 http://localhost:9001（`minioadmin / minioadmin`）。

### 2.5 配置连接参数

复制示例文件并按需填写：

```bash
cp .env.example .env
```  

### 2.6 启动服务

```bash
# 本机后端（需先启动 Neo4j / Milvus / Redis）
python main.py
```  

也可拉起完整容器栈（后端、前端、Nginx）：

```bash
docker compose up -d --build
```

### 🌐 访问应用

启动完成后，访问：

- **🏠 应用首页**: http://localhost
- **⚛️ 前端**: http://localhost:3000
- **🐍 后端API**: http://localhost:8000
- **📊 Neo4j**: http://localhost:7474 (neo4j/all-in-rag)
- **🗄️ MinIO（Milvus 存储）**: http://localhost:9001 (minioadmin/minioadmin)

## 三、系统架构设计

### 3.1 整体架构

本项目的RAG系统采用模块化设计，包含以下核心组件：  

```mermaid
flowchart TD

    %% ==================== 启动与知识库 ====================
    subgraph BOOT["启动与知识库准备"]
        START["main.py 启动"] --> CONFIG["加载 GraphRAGConfig<br/>Neo4j / Milvus / LLM / Redis"]
        CONFIG --> DEP_CHECK{"依赖检查"}
        DEP_CHECK -->|失败| BOOT_FAIL["启动失败<br/>输出具体依赖检查项"]
        DEP_CHECK -->|通过| INIT["按依赖顺序初始化 8 个步骤"]

        INIT --> DATA["GraphDataPreparation<br/>Neo4j 图数据 → 菜谱文档 → chunks"]
        INIT --> INDEX["MilvusIndexConstruction<br/>BGE-small-zh-v1.5 / 512 维"]
        INIT --> GEN_MOD["GenerationIntegration<br/>OpenAI 兼容 LLM 客户端"]
        INIT --> HYBRID["HybridRetrievalModule"]
        INIT --> GRAPH["GraphRAGRetrieval"]
        INIT --> ROUTER["IntelligentQueryRouter"]
        INIT --> CACHE["SessionCacheManager"]
        INIT --> LIMITER["RedisRateLimiter"]
        INIT --> RECIPE["RecipeRecommendationManager"]

        INDEX --> KB_CHECK{"Milvus 集合已存在？"}
        KB_CHECK -->|是| LOAD_KB["加载已有向量集合"]
        KB_CHECK -->|否| BUILD_DATA["Neo4j → 菜谱文档 / chunks"]
        LOAD_KB --> LOAD_DATA["加载图数据<br/>构建菜谱文档 / chunks"]
        LOAD_KB -.->|加载失败则重建| BUILD_DATA
        BUILD_DATA --> BUILD_VEC["新建集合并构建向量索引"]
        LOAD_DATA --> READY["初始化 BM25 / 图索引 / 检索引擎"]
        BUILD_VEC --> READY
        DATA -.->|提供图数据/文档能力| LOAD_DATA
        DATA -.->|提供图数据/文档能力| BUILD_DATA
        READY --> APP["uvicorn 启动 ASGI 服务"]

        CACHE --> REDIS_CONN["Redis 连接"]
        LIMITER --> REDIS_CONN
        REDIS_CONN -->|不可用且 required=false| MEM_FALLBACK["降级进程内会话/缓存<br/>限流放行"]
        REDIS_CONN -->|不可用且 required=true| BOOT_FAIL
    end

    %% ==================== Web / API ====================
    subgraph WEB["Web / API 层"]
        CLIENT["浏览器 / Next.js 前端"] --> NGINX["Nginx<br/>/ → 前端 · /api → 后端"]
        APP --> API["FastAPI 应用<br/>WebServiceHandler.create_app"]
        NGINX --> API
        API --> HEALTH["/health<br/>含 Redis ping"]
        API --> STATIC["静态文件 / SPA fallback"]
        API --> LIMIT{"Redis 固定窗口限流<br/>默认 60/min · chat 30/min"}
        LIMIT -->|超限| TOO_MANY["429 + Retry-After"]
        LIMIT --> CHAT_API["/api/chat"]
        LIMIT --> STREAM_API["/api/chat_stream<br/>SSE + [DONE]"]
        LIMIT --> RECIPE_API["/api/recipes/*"]
        LIMIT --> STATS_API["/api/stats"]
        LIMIT --> FEEDBACK_API["/api/feedback*"]
    end

    %% ==================== 聊天主流程 ====================
    subgraph CHATFLOW["聊天主流程"]
        CACHE_CHECK{"语义缓存命中？<br/>同 session · 同 topic<br/>相似度 ≥ 0.75"}
        CACHED["直接返回缓存答案"]
        STATE["读取会话状态<br/>active_recipe / active_topic<br/>constraints / recent_user_queries"]
        ROUTE["IntelligentQueryRouter.route_query"]
        RESP{"response_type<br/>+ needs_retrieval"}
        CLARIFY["clarify<br/>返回澄清问题"]
        DIRECT["chitchat / out_of_scope<br/>直接回答或固定拒识"]
        NO_DATA["no_data<br/>明确告知证据不足"]
        ANSWER["answer<br/>进入生成层"]
        UPDATE["update_state_after_turn<br/>更新 active_recipe / constraints"]
        WRITE_CACHE["写语义缓存<br/>no_data / clarify / refuse 不写"]
        WRITE_CTX["写会话上下文"]

        CHAT_API --> CACHE_CHECK
        STREAM_API --> CACHE_CHECK
        CACHE_CHECK -->|命中| CACHED
        CACHED --> WRITE_CTX
        CACHE_CHECK -->|未命中| STATE
        STATE --> ROUTE
        ROUTE --> RESP
        RESP -->|clarify| CLARIFY
        RESP -->|refuse 或不检索| DIRECT
        RESP -->|no_data| NO_DATA
        RESP -->|answer 且有证据| ANSWER
        CLARIFY --> UPDATE
        UPDATE --> WRITE_CACHE
        UPDATE --> WRITE_CTX
    end

    %% ==================== 查询路由包 ====================
    subgraph ROUTING["查询路由包 query_routing"]
        UNDERSTAND["understanding.py<br/>LLM 结构化输出 → 解析 → 一次修复"]
        RULES["rules.py + slots.py<br/>规则路由 / 追问 / 槽位归一化"]
        UNDER_OK["Understanding<br/>intents / entities / slots / context"]
        KEYS["RouteKeySpec × 12<br/>证据形态 / engine / validator / 阈值"]
        POLICY["policy.py<br/>确定性生成 RoutePlan"]
        THRESHOLD["置信度门控<br/>min_execute / retrieval_override / no_retrieval"]
        PLAN["RoutePlan<br/>单/多子计划 · top_k · depends_on"]
        EXEC["evidence.py<br/>按 engine 执行"]
        PARALLEL["单子计划直接执行<br/>多子计划 Semaphore(4) 并行<br/>gather 保持顺序"]
        VALIDATE["validators.py<br/>complete / partial / missing"]
        TEXT_RETRY["retry_hint=text<br/>补一次文本检索"]
        MERGE["合并去重<br/>保留 sub_query 元数据"]
        ANALYSIS["QueryAnalysis<br/>route_key / engine / response_type<br/>evidence_status / sub_evidence"]
        TELEMETRY["data/logs/route_decisions.jsonl"]

        ROUTE --> UNDERSTAND
        UNDERSTAND -->|成功| UNDER_OK
        UNDERSTAND -->|失败| RULES
        RULES --> UNDER_OK
        KEYS -.-> POLICY
        UNDER_OK --> POLICY
        POLICY --> THRESHOLD
        THRESHOLD --> PLAN
        PLAN -->|clarify| CLARIFY
        PLAN -->|chitchat / out_of_scope| DIRECT
        PLAN -->|需要检索| EXEC
        EXEC --> PARALLEL
        PARALLEL --> VALIDATE
        VALIDATE -->|retry_hint=text| TEXT_RETRY
        TEXT_RETRY --> VALIDATE
        VALIDATE --> MERGE
        MERGE -->|全部 missing| NO_DATA
        MERGE --> ANALYSIS
        ANALYSIS -.->|QueryAnalysis| RESP
        ROUTE -.-> TELEMETRY
    end

    %% ==================== 证据引擎 ====================
    subgraph ENGINES["证据引擎（按 SubPlan.engine 分发）"]
        GE["graph_entity<br/>食材 / 步骤时间 / 概览<br/>推荐候选 / 营养数值 / 功效"]
        GR["graph_relation<br/>graph_rag_search"]
        HY["hybrid<br/>三路召回 + RRF"]
        CO["combined<br/>文本 + 图并行"]
        TIPS["tips_*<br/>pairing / safety / technique / rule"]
        NONE["none<br/>不检索"]

        EXEC --> GE
        EXEC --> GR
        EXEC --> HY
        EXEC --> CO
        EXEC --> TIPS
        EXEC --> NONE
        GE -.->|图谱无结果，转文本兜底| HY
    end

    %% ==================== 检索内部实现 ====================
    subgraph RETRIEVAL["检索内部实现"]
        DUAL["双层检索<br/>实体级 + 主题级"]
        VECTOR["Milvus 向量检索<br/>一跳邻居增强"]
        BM25["BM25<br/>jieba 分词"]
        RRF["RRF 融合"]
        PARENT["父文档回填<br/>默认开启，可配置"]
        ROUND["Round-robin 合并<br/>图结果优先"]
        GQUERY["图查询理解<br/>entity_relation / multi_hop<br/>subgraph / path_finding"]
        MULTIHOP["多跳遍历<br/>最大 3 跳 + hub 剪枝"]
        SUBGRAPH["知识子图抽取<br/>最大 200 节点"]
        GREASON["图结构推理<br/>推理链校验"]
        RANK["图结构 + 词法 + 类型排序"]

        HY --> DUAL
        HY --> VECTOR
        HY --> BM25
        DUAL --> RRF
        VECTOR --> RRF
        BM25 --> RRF
        RRF --> PARENT
        GR --> GQUERY
        GQUERY --> MULTIHOP
        GQUERY --> SUBGRAPH
        MULTIHOP --> GREASON
        SUBGRAPH --> GREASON
        GREASON --> RANK
        CO --> HY
        CO --> GR
        CO --> ROUND
        HY --> ROUND
        GR --> ROUND
    end

    %% ==================== 生成层 ====================
    subgraph GENERATION["生成层"]
        GEN_DIRECT["generate_direct_answer<br/>闲聊 / 域外固定拒识"]
        NOEVID["generate_adaptive_answer<br/>无证据 Prompt"]
        SINGLE["单意图<br/>一次 LLM 调用<br/>howto ≥3 步修复分条"]
        SECTION["多意图<br/>按子问题分节"]
        ORDER["最多 4 路并行生成<br/>按原顺序输出<br/>流式 howto 首节先修复"]
        STREAM["流式输出<br/>SSE chunk"]
        NONSTREAM["非流式输出<br/>完整 JSON"]
        STREAM_FAIL["流式失败 → 非流式兜底"]

        DIRECT --> GEN_DIRECT
        ANSWER -->|单意图| SINGLE
        ANSWER -->|多意图| SECTION
        SECTION --> ORDER
        NO_DATA --> NOEVID
        SINGLE --> STREAM
        SINGLE --> NONSTREAM
        ORDER --> STREAM
        ORDER --> NONSTREAM
        NOEVID --> STREAM
        NOEVID --> NONSTREAM
        GEN_DIRECT --> STREAM
        GEN_DIRECT --> NONSTREAM
        STREAM -.->|失败| STREAM_FAIL
        STREAM_FAIL --> NONSTREAM
        STREAM --> UPDATE
        NONSTREAM --> UPDATE
    end

    %% ==================== 存储与可观测性 ====================
    subgraph STORAGE["存储与可观测性"]
        NEO4J_DB[("Neo4j<br/>Recipe / Ingredient / Step")]
        MILVUS_DB[("Milvus<br/>向量索引")]
        REDIS_DB[("Redis<br/>会话状态 / 语义缓存 / 限流")]
        TIPSDOC[("data/tips<br/>相克 / 安全 / 技法 / 选菜")]
        LOGS[("data/logs<br/>route_decisions / feedback")]
        DATA_FILES[("data/recipes_with_images.json<br/>data/dishes 菜谱数据 / 图片")]

        REDIS_CONN --> REDIS_DB
        GE --> NEO4J_DB
        GR --> NEO4J_DB
        HY --> NEO4J_DB
        HY --> MILVUS_DB
        VECTOR --> MILVUS_DB
        DUAL --> NEO4J_DB
        TIPSDOC -.->|解析| TIPS
        RECIPE_API --> RECIPE
        RECIPE --> DATA_FILES
        FEEDBACK_API --> LOGS
        TELEMETRY --> LOGS
        HEALTH -.-> REDIS_DB
        LIMIT --> REDIS_DB
        STATE -.-> REDIS_DB
        CACHE_CHECK -.-> REDIS_DB
        WRITE_CACHE --> REDIS_DB
        WRITE_CTX --> REDIS_DB
        STATS_API -.-> REDIS_DB
        STATS_API -.-> ROUTER
    end

    %% ==================== 降级与错误处理 ====================
    subgraph FAILOVER["降级与错误处理"]
        F_LLM["理解失败 → 规则路由"]
        F_EVID["证据不足 → 文本补检索 / no_data"]
        F_ROUTE["路由执行异常 → 传统 hybrid<br/>evidence_status=fallback"]
        F_STREAM["流式异常 → 非流式"]
        F_REDIS["Redis 不可用 → 进程内降级 / 启动失败"]
    end

    UNDERSTAND -.-> F_LLM
    VALIDATE -.-> F_EVID
    ROUTE -.-> F_ROUTE
    STREAM -.-> F_STREAM
    REDIS_CONN -.-> F_REDIS

    %% ==================== 样式 ====================
    %% 分组底板保持中性，节点用浅色填充区分阶段；连线退到背景层
    style BOOT fill:#fafaf7,stroke:#e1e0d9,stroke-width:1px,color:#52514e
    style WEB fill:#fafaf7,stroke:#e1e0d9,stroke-width:1px,color:#52514e
    style CHATFLOW fill:#fafaf7,stroke:#e1e0d9,stroke-width:1px,color:#52514e
    style ROUTING fill:#fafaf7,stroke:#e1e0d9,stroke-width:1px,color:#52514e
    style ENGINES fill:#fafaf7,stroke:#e1e0d9,stroke-width:1px,color:#52514e
    style RETRIEVAL fill:#fafaf7,stroke:#e1e0d9,stroke-width:1px,color:#52514e
    style GENERATION fill:#fafaf7,stroke:#e1e0d9,stroke-width:1px,color:#52514e
    style STORAGE fill:#fafaf7,stroke:#e1e0d9,stroke-width:1px,color:#52514e
    style FAILOVER fill:#fafaf7,stroke:#e1e0d9,stroke-width:1px,color:#52514e
    linkStyle default stroke:#9c9a92,stroke-width:1.1px

    %% 每个节点保留浅色填充；色相只表示阶段，统一低饱和，避免大色块抢文字
    classDef boot fill:#edf2f7,stroke:#64748b,stroke-width:1.4px,color:#0b0b0b
    classDef web fill:#eaf2fb,stroke:#2a78d6,stroke-width:1.4px,color:#0b0b0b
    classDef route fill:#d9e7f8,stroke:#2a78d6,stroke-width:2px,color:#0b0b0b
    classDef engine fill:#fceae3,stroke:#eb6834,stroke-width:1.4px,color:#0b0b0b
    classDef gen fill:#dff4ec,stroke:#1baf7a,stroke-width:1.4px,color:#0b0b0b
    classDef store fill:#f2f1ee,stroke:#898781,stroke-width:1.4px,color:#0b0b0b
    classDef fail fill:#f8e4e4,stroke:#d03b3b,stroke-width:1.6px,color:#0b0b0b
    classDef success fill:#ddf2dd,stroke:#0ca30c,stroke-width:1.4px,color:#0b0b0b

    class START,CONFIG,INIT,DATA,INDEX,GEN_MOD,HYBRID,GRAPH,ROUTER,CACHE,LIMITER,RECIPE,READY,APP,REDIS_CONN,LOAD_KB,LOAD_DATA,BUILD_DATA,BUILD_VEC boot
    class CLIENT,NGINX,API,HEALTH,CHAT_API,STREAM_API,RECIPE_API,STATS_API,FEEDBACK_API,STATIC web
    class LIMIT,TOO_MANY,BOOT_FAIL,MEM_FALLBACK,F_LLM,F_EVID,F_ROUTE,F_STREAM,F_REDIS fail
    class ROUTE,UNDERSTAND,RULES,UNDER_OK,KEYS,POLICY,THRESHOLD,PLAN,EXEC,PARALLEL,VALIDATE,TEXT_RETRY,MERGE,ANALYSIS,TELEMETRY route
    class GE,GR,HY,CO,TIPS,NONE,DUAL,VECTOR,BM25,RRF,PARENT,ROUND,GQUERY,MULTIHOP,SUBGRAPH,GREASON,RANK engine
    class CACHE_CHECK,CACHED,STATE,RESP,CLARIFY,DIRECT,NO_DATA,ANSWER,UPDATE,WRITE_CACHE,WRITE_CTX,GEN_DIRECT,NOEVID,SINGLE,SECTION,ORDER,STREAM,NONSTREAM,STREAM_FAIL gen
    class NEO4J_DB,MILVUS_DB,REDIS_DB,TIPSDOC,LOGS,DATA_FILES store
    class DEP_CHECK,KB_CHECK success
```  

### 3.2 核心模块说明

#### 图数据准备模块 (GraphDataPreparationModule)
- **功能**：从 Neo4j 加载菜谱、食材、步骤及关系，构建结构化菜谱文档并分块
- **特点**：保留图结构语义；按标题与长度双重分块；chunk 带父文档和章节元数据

#### 向量索引模块 (MilvusIndexConstructionModule)
- **功能**：构建和管理 Milvus 向量集合，提供语义相似度检索
- **特点**：BGE-small-zh-v1.5 / 512 维；HNSW + COSINE；重建时删除并重建集合

#### 混合检索模块 (HybridRetrievalModule)
- **功能**：三路召回（双层图检索、Milvus 向量检索、BM25 关键词检索）经 RRF 融合
- **特点**：实体级 + 主题级双层检索；jieba 分词；可选父文档回填

#### 图RAG检索模块 (GraphRAGRetrieval)
- **功能**：理解图查询，沿关系路径补充多跳推理和子图证据
- **特点**：最大 3 跳 + hub 剪枝；子图最多 200 节点；图结构推理与排序

#### 智能查询路由 (IntelligentQueryRouter)
- **功能**：把用户问题转成 `RoutePlan`，按引擎取证据，返回文档与 `QueryAnalysis`
- **特点**：LLM 结构化理解 + 规则兜底；12 个路由键；置信度门控；多意图并行；证据校验与文本补检

#### 生成集成模块 (GenerationIntegrationModule)
- **功能**：基于检索文档和路由分析生成最终答案，支持流式与非流式
- **特点**：单/多意图自适应；多意图最多 4 路并行、按序输出；流式失败转非流式；howto 步骤修复

#### 会话与限流模块 (SessionCacheManager + RedisRateLimiter)
- **功能**：维护多轮会话状态、语义缓存和 Redis 固定窗口限流
- **特点**：同 session / 同 topic 语义缓存；跨轮保留当前菜谱与约束；Redis 不可用时按配置降级或启动失败

#### 菜谱推荐模块 (RecipeRecommendationManager)
- **功能**：读取 `data/recipes_with_images.json` 与 `data/dishes`，提供随机推荐、菜谱详情和图片
- **特点**：不经过 RAG；推荐接口返回随机菜谱，偏好字段预留

### 3.3 数据流程

1. **启动与数据准备**
   - 按依赖顺序初始化数据、索引、生成、检索、路由、Redis、推荐模块
   - Milvus 集合不存在：Neo4j → 菜谱文档 → chunks → 构建向量索引
   - 集合已存在：直接加载，同时加载图数据并构建文档 / chunks
   - 初始化 BM25、图索引和检索引擎；Redis 不可用时按 `REDIS_REQUIRED` 降级或退出

2. **查询处理**
   - Nginx → FastAPI：限流、读取会话状态，命中语义缓存则直接返回
   - `route_query`：LLM 理解 + 一次修复，失败走规则；策略层按置信度生成 `RoutePlan`
   - 按 `engine` 并行取证据（图实体 / 图关系 / hybrid / combined / tips / none），校验后合并
   - 生成层按单意图或多意图分节输出，多意图最多 4 路并行、按原顺序返回
   - 更新会话状态，按条件写入语义缓存和上下文

3. **错误处理与降级**
   - 理解失败 → 规则路由；证据不足 → 文本补检索或 `no_data`
   - 路由执行异常 → 传统 hybrid，`evidence_status=fallback`
   - 流式异常 → 非流式；Redis 不可用 → 进程内降级或启动失败

## 🛠️ 技术栈

| 层级 | 技术栈 | 职责 |
| --- | --- | --- |
| 前端 | Next.js 14 · React · Tailwind CSS · Zustand · Framer Motion | 页面渲染、状态管理与动画 |
| 后端 | Python 3.11 · FastAPI · uvicorn | API 服务、SSE 流式输出 |
| 检索 | Neo4j 5.11 · Milvus 2.3.2 · BM25（jieba） | 图谱多跳、向量语义、关键词召回 |
| 模型 | BGE-small-zh-v1.5 · OpenAI 兼容 LLM | 512 维嵌入、查询理解与答案生成 |
| 状态 | Redis 7 | 多轮会话、语义缓存、固定窗口限流 |
| 部署 | Docker Compose · Nginx | 容器编排、统一入口与反向代理 |

**🍽️ 享受您的美食推荐之旅！** 如果这个项目对您有帮助，请给个⭐️支持一下！

## 📄 许可证

本项目采用 [MIT License](./LICENSE) 开源协议。

Copyright © 2026 cookwise-ai contributors

### 🔓 使用权限
- ✅ **商业使用** - 可以用于商业项目
- ✅ **修改** - 可以修改源代码
- ✅ **分发** - 可以分发原始或修改后的代码
- ✅ **私人使用** - 可以私人使用

### 📝 使用条件
- **保留版权声明** - 在所有副本中包含原始版权声明和许可证声明
- **保留许可证** - 在所有副本中包含 MIT 许可证

### ⚠️ 免责声明
- 软件按"原样"提供，不提供任何明示或暗示的保证
- 作者不承担任何责任或义务
