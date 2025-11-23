# RAG Agents Metadata Tables Usage Guide

**Document Version:** 1.0  
**Last Updated:** November 23, 2025  
**Scope:** Complete metadata table mapping for all RAG agents and their database interactions

---

## Executive Summary

The RAG system uses **8 core metadata tables** for:
- **Ingestion**: Document storage and embeddings
- **Retrieval**: Query tracking and results
- **Healing/Optimization**: Quality improvements and learning
- **Operations**: Audit trails and memory

This document maps **WHAT**, **WHEN**, **HOW**, **WHY** for each table and provides SQL details with ER relationships.

---

## Part 1: Core RAG Metadata Tables

### Table 1: `documents`

**Table Purpose:**
Master document storage for all ingested content. Single source of truth for document metadata and content.

**When Populated:**
- **On ingestion**: MasterOrchestrator.ingest_data() → IngestionAgent.ingest_document()
- **One row per document**: Created once, updated only if content changes

**How Populated:**
```python
# From: master_orchestrator.py, ingestion_agent.py
# Agent: IngestionAgent.ingest_document()

1. Read file content
2. Create doc_id from filename or auto-generate UUID
3. Extract title from metadata extraction
4. Store complete content in 'content' column
5. Record source (file path, URL, etc.)
6. Set doc_type from LLM classification
```

**Data Structure:**
| Column | Type | Source | Purpose |
|--------|------|--------|---------|
| `id` | TEXT (PK) | Auto-generated | Unique document identifier |
| `title` | TEXT | LLM extraction (extract_metadata_tool) | Document title for display |
| `content` | TEXT | File read/user input | Full document content |
| `source` | TEXT | File path or URL | Origin of document |
| `doc_type` | TEXT | LLM classification | Type: pdf, html, text, incident |
| `created_at` | TEXT (timestamp) | System datetime | Ingestion time |
| `updated_at` | TEXT (timestamp) | System datetime | Last modification time |

**Example:**
```sql
INSERT INTO documents 
(id, title, content, source, doc_type, created_at, updated_at)
VALUES 
('finance_policy_001', 
 'Finance Policy Guidelines',
 'Document content here...',
 'file://docs/finance_policy.txt',
 'policy',
 '2025-11-23T10:30:00',
 '2025-11-23T10:30:00');
```

**Relationships:**
- Referenced by: document_metadata (1:M via doc_id)
- Referenced by: embedding_metadata (1:M via document_id)
- Referenced by: synthetic_queries (1:M via doc_id)
- Referenced by: rl_episodes (1:M via document_id)

**Used By:**
- IngestionAgent: INSERT (during ingest_document)
- RetrievalAgent: SELECT (to fetch full content)
- MasterOrchestrator: SELECT (for query routing)

**Query Examples:**
```sql
-- Get document by ID
SELECT * FROM documents WHERE id = 'doc_123';

-- Get all documents of specific type
SELECT * FROM documents WHERE doc_type = 'policy';

-- Get recently ingested documents
SELECT * FROM documents 
ORDER BY created_at DESC LIMIT 10;
```

---

### Table 2: `document_metadata`

**Table Purpose:**
Flexible key-value metadata storage for domain-specific and RBAC attributes. Enables unlimited custom metadata without schema changes.

**When Populated:**
- **During ingestion**: MasterOrchestrator._extract_metadata()
- **On RBAC assignment**: During access control setup
- **Multiple rows per document**: Each key-value pair is one row

**How Populated:**
```python
# From: master_orchestrator.py -> MasterOrchestrator._extract_metadata()
# From: ingestion_agent.py -> IngestionAgent.ingest_document()

1. Extract text from document
2. Run LLM metadata extraction (extract_metadata_tool)
3. Get structured data: title, summary, keywords, topics, doc_type
4. Map doc_type to rbac_namespace:
   - 'technical_doc' → 'engineering'
   - 'policy' → 'security'
   - 'report' → 'finance'
   - 'incident' → 'security'
   - DEFAULT → 'general'
5. Insert each key-value pair as separate row
6. Additionally insert RBAC namespace for access control
```

**Data Structure:**
| Column | Type | Source | Purpose |
|--------|------|--------|---------|
| `metadata_id` | INTEGER (PK) | Auto-increment | Unique metadata row ID |
| `doc_id` | TEXT (FK) | documents.id | Reference to parent document |
| `key` | TEXT | LLM extraction or system | Metadata key |
| `value` | TEXT | LLM extraction or system | Metadata value |

**Standard Keys & Values:**
| Key | Example Values | Purpose |
|-----|-----------------|---------|
| `domain` | finance, travel, medical, technical, legal, hr | Domain classification |
| `rbac_namespace` | finance_dept, hr_team, engineering, security, general | Access control namespace |
| `summary` | "Policy about expense approval" | Document summary (1-3 sentences) |
| `keywords` | "expense, approval, budget, limits" | Comma-separated keywords |
| `topics` | "finance, compliance, policy" | Document topics |
| `source_system` | CRM, ERP, knowledge_base, email | System of origin |
| `region` | US, EU, APAC, Global | Geographic scope |
| `classification` | confidential, internal, public | Security level |
| `version` | 1.0, 2.1, etc. | Document version |
| `author` | "John Smith" | Document author |
| `department` | "Finance", "HR", "Engineering" | Owning department |

**Example:**
```sql
-- Extract metadata for document
INSERT INTO document_metadata (doc_id, key, value) VALUES
('finance_policy_001', 'domain', 'finance'),
('finance_policy_001', 'rbac_namespace', 'finance_dept'),
('finance_policy_001', 'summary', 'Policy covering expense approval procedures'),
('finance_policy_001', 'keywords', 'expense, approval, budget, limits'),
('finance_policy_001', 'topics', 'finance, compliance, policy'),
('finance_policy_001', 'classification', 'internal'),
('finance_policy_001', 'department', 'Finance');
```

**Relationships:**
- Foreign Key: doc_id → documents.id (ON DELETE CASCADE)
- No child tables

**Used By:**
- IngestionAgent: INSERT (during metadata extraction)
- RetrievalAgent.enforce_rbac(): SELECT (filter by rbac_namespace)
- MasterOrchestrator: SELECT (query metadata)

**RBAC Enforcement Example:**
```python
# From retrieval_agent.py -> RetrievalAgent.process_query()

1. Get user_id from request
2. Query: SELECT namespace FROM user_permissions WHERE user_id = ?
3. Query: SELECT doc_id FROM document_metadata 
          WHERE key='rbac_namespace' AND value=namespace
4. Only search in authorized doc_ids
```

**Query Examples:**
```sql
-- Get all metadata for a document
SELECT key, value FROM document_metadata 
WHERE doc_id = 'doc_123';

-- Get documents by domain
SELECT DISTINCT doc_id FROM document_metadata 
WHERE key = 'domain' AND value = 'finance';

-- RBAC: Get documents accessible by namespace
SELECT DISTINCT doc_id FROM document_metadata 
WHERE key = 'rbac_namespace' AND value = 'finance_dept';

-- Get documents with specific classification
SELECT DISTINCT doc_id FROM document_metadata 
WHERE key = 'classification' AND value = 'confidential';
```

---

### Table 3: `embedding_metadata`

**Table Purpose:**
Vector chunk metadata for semantic search. Stores embedding quality, chunking strategy, and version info for each document chunk.

**When Populated:**
- **During ingestion**: save_to_vectordb_tool() for each chunk
- **During healing**: healing_agent updates quality scores
- **Multiple rows per document**: One row per chunk

**How Populated:**
```python
# From: ingestion_tools.py -> save_to_vectordb_tool()
# From: master_orchestrator.py -> MasterOrchestrator.ingest_data()

For each chunk:
1. Generate chunk_id: f"{doc_id}_{chunk_index}"
2. Split document into chunks (RecursiveCharacterTextSplitter)
3. Generate embedding for chunk using LLM service
4. Create chunk_strategy record from splitting parameters
5. Calculate quality score (default 0.95 for new chunks)
6. Store embedding model version
7. Record reindex_count (0 for new)
8. Update metadata after healing
```

**Data Structure:**
| Column | Type | Source | Purpose |
|--------|------|--------|---------|
| `embedding_id` | INTEGER (PK) | Auto-increment | Unique embedding row ID |
| `document_id` | TEXT (FK) | documents.id | Reference to parent document |
| `chunk_id` | TEXT (UNIQUE) | Auto-generated | Unique chunk identifier |
| `chunk_strategy` | TEXT | Splitting algorithm | How chunk was created: recursive, semantic, fixed |
| `chunk_size` | INTEGER | Actual chunk length | Number of tokens/characters in chunk |
| `overlap` | INTEGER | Splitting parameter | Overlap with next chunk (usually 50) |
| `embedding_model` | TEXT | LLM config | Model used: text-embedding-3-small, ollama |
| `embedding_version` | TEXT | LLM version | Version of embedding model (e.g., "1.0") |
| `quality_score` | REAL (0-1) | Initial: 0.95, Healing: recalculated | Vector quality metric |
| `reindex_count` | INTEGER | Healing operations | Times chunk has been re-indexed |
| `last_modified` | TEXT (timestamp) | System datetime | Last update time |

**Example:**
```sql
INSERT INTO embedding_metadata 
(document_id, chunk_id, chunk_strategy, chunk_size, overlap, embedding_model, 
 embedding_version, quality_score, reindex_count, last_modified)
VALUES 
('finance_policy_001', 'finance_policy_001_chunk_0', 'recursive', 487, 50, 
 'ollama', '1.0', 0.95, 0, '2025-11-23T10:30:00');

-- After healing - quality score updated
UPDATE embedding_metadata 
SET quality_score = 0.98, reindex_count = 1, last_modified = NOW()
WHERE chunk_id = 'finance_policy_001_chunk_0';
```

**Relationships:**
- Foreign Key: document_id → documents.id (ON DELETE CASCADE)
- No child tables
- UNIQUE constraint on chunk_id

**Used By:**
- IngestionAgent: INSERT (during save_to_vectordb_tool)
- RetrievalAgent.search_semantic(): SELECT (retrieve chunks)
- HealingAgent: UPDATE (improve quality scores)
- Vector DB: Query by embedding vectors

**Quality Score Improvements (Healing):**
```python
# From: healing_agent.py -> HealingAgent.optimize_ingestion()

# Scenario 1: Re-embed with new model
UPDATE embedding_metadata 
SET embedding_version = 'enhanced', quality_score = 0.98
WHERE document_id = 'finance_policy_001';

# Scenario 2: Improve retrieval quality
UPDATE embedding_metadata 
SET quality_score = quality_score * 1.05  # Boost by 5%
WHERE quality_score < 0.75;

# Scenario 3: Re-index with better strategy
UPDATE embedding_metadata 
SET chunk_strategy = 'semantic', reindex_count = reindex_count + 1
WHERE document_id = 'finance_policy_001';
```

**Query Examples:**
```sql
-- Get all chunks for a document
SELECT chunk_id, chunk_size, quality_score 
FROM embedding_metadata 
WHERE document_id = 'finance_policy_001'
ORDER BY chunk_id;

-- Find low-quality chunks (quality < 0.75)
SELECT document_id, chunk_id, quality_score 
FROM embedding_metadata 
WHERE quality_score < 0.75
ORDER BY quality_score;

-- Get chunks that need re-indexing
SELECT chunk_id, reindex_count, last_modified 
FROM embedding_metadata 
WHERE reindex_count < 1
ORDER BY last_modified DESC LIMIT 100;

-- Average quality by document
SELECT document_id, AVG(quality_score) as avg_quality, COUNT(*) as chunk_count
FROM embedding_metadata 
GROUP BY document_id
ORDER BY avg_quality;
```

---

### Table 4: `synthetic_queries`

**Table Purpose:**
Auto-generated questions for document context understanding. Used for RAG quality testing and domain-specific QA training data.

**When Populated:**
- **After ingestion**: SyntheticQuestionsGenerator.generate_questions()
- **One or more rows per document**: Typically 3-10 questions per document

**How Populated:**
```python
# From: synthetic_questions_generator.py -> SyntheticQuestionsGenerator
# From: ingestion_agent.py -> IngestionAgent.ingest_document()

For each document:
1. Extract document summary and topics
2. For each topic, generate 2-5 questions using LLM
3. Questions should be answerable from document
4. Store in synthetic_queries table
5. Used for: retrieval quality test, QA training data, domain validation
```

**Data Structure:**
| Column | Type | Source | Purpose |
|--------|------|--------|---------|
| `query_id` | INTEGER (PK) | Auto-increment | Unique query ID |
| `doc_id` | TEXT (FK) | documents.id | Reference to parent document |
| `question` | TEXT | LLM generation | Generated question about document |
| `created_at` | TEXT (timestamp) | System datetime | Generation time |

**Example:**
```sql
-- Example questions generated for finance_policy_001
INSERT INTO synthetic_queries (doc_id, question, created_at) VALUES
('finance_policy_001', 'What is the expense approval process?', '2025-11-23T10:31:00'),
('finance_policy_001', 'What is the maximum expense limit per request?', '2025-11-23T10:31:00'),
('finance_policy_001', 'How long does expense approval take?', '2025-11-23T10:31:00'),
('finance_policy_001', 'Who approves expenses above $5000?', '2025-11-23T10:31:00'),
('finance_policy_001', 'What documentation is needed for expense claims?', '2025-11-23T10:31:00');
```

**Relationships:**
- Foreign Key: doc_id → documents.id (ON DELETE CASCADE)
- No child tables

**Used By:**
- IngestionAgent: INSERT (via SyntheticQuestionsGenerator)
- RAG testing: SELECT (to validate retrieval quality)
- Domain QA: SELECT (for training data)

**Query Examples:**
```sql
-- Get questions for a document (for quality testing)
SELECT question FROM synthetic_queries 
WHERE doc_id = 'finance_policy_001'
ORDER BY created_at;

-- Count questions per document
SELECT doc_id, COUNT(*) as question_count
FROM synthetic_queries
GROUP BY doc_id
ORDER BY question_count DESC;

-- Get sample questions across domain
SELECT DISTINCT sq.question 
FROM synthetic_queries sq
JOIN document_metadata dm ON sq.doc_id = dm.doc_id
WHERE dm.key = 'domain' AND dm.value = 'finance'
LIMIT 20;
```

---

## Part 2: Operations & Tracking Metadata Tables

### Table 5: `agent_operations`

**Table Purpose:**
Audit log for all RAG agent operations. Complete record of what agents did, when, status, tokens used, and execution time.

**When Populated:**
- **After each agent operation**: record_agent_operation_tool()
- **Ingestion**: When chunks are saved
- **Retrieval**: When queries are processed
- **Healing**: When optimizations are applied

**How Populated:**
```python
# From: ingestion_tools.py -> record_agent_operation_tool()
# Called by: All agents after major operations

1. Operation starts (record start time)
2. Operation completes or fails
3. Call record_agent_operation_tool with:
   - Agent name (IngestionAgent, RetrievalAgent, HealingAgent)
   - Operation type (process_query, ingest_document, optimize_document)
   - Status (success, failed, partial)
   - Input data (serialized as JSON)
   - Output data (serialized as JSON)
   - Error message (if failed)
   - Execution time in milliseconds
```

**Data Structure:**
| Column | Type | Source | Purpose |
|--------|------|--------|---------|
| `operation_id` | INTEGER (PK) | Auto-increment | Unique operation ID |
| `agent_id` | TEXT | Agent name | Which agent executed: ingestion_agent_01, retrieval_agent_01 |
| `operation_type` | TEXT | Agent method | Type: process_query, ingest_document, optimize_document |
| `status` | TEXT | Execution result | running, completed, failed, partial |
| `input_data` | TEXT (JSON) | Operation parameters | Serialized input (query, doc_id, etc.) |
| `output_data` | TEXT (JSON) | Operation result | Serialized result (chunks_saved, results_count, etc.) |
| `error_message` | TEXT | Exception if failed | Error details if status=failed |
| `timestamp` | TEXT (timestamp) | System datetime | When operation executed |
| `execution_time_ms` | INTEGER | Time measurement | Operation duration in milliseconds |

**Example:**
```sql
-- Ingestion operation
INSERT INTO agent_operations 
(agent_id, operation_type, status, input_data, output_data, timestamp, execution_time_ms)
VALUES 
('ingestion_agent_01', 'ingest_document', 'completed',
 '{"doc_id": "finance_policy_001", "file_path": "..."}',
 '{"success": true, "chunks_saved": 12, "vectors_saved": 12}',
 '2025-11-23T10:31:00', 1250);

-- Retrieval operation
INSERT INTO agent_operations 
(agent_id, operation_type, status, input_data, output_data, timestamp, execution_time_ms)
VALUES 
('retrieval_agent_01', 'process_query', 'completed',
 '{"query": "What is expense approval?", "user_id": "user123"}',
 '{"success": true, "results_count": 5, "query_tokens": 4, "response_tokens": 120}',
 '2025-11-23T10:35:00', 450);
```

**Relationships:**
- Referenced by: llm_token_usage (1:M via operation_id)
- No parent tables (independent audit log)

**Used By:**
- All agents: INSERT (after operations)
- Monitoring: SELECT (track execution times)
- Performance analysis: AGGREGATE (avg time by agent, error rates)

**Query Examples:**
```sql
-- Get all operations for an agent
SELECT * FROM agent_operations 
WHERE agent_id = 'retrieval_agent_01'
ORDER BY timestamp DESC LIMIT 50;

-- Get failed operations
SELECT * FROM agent_operations 
WHERE status = 'failed'
ORDER BY timestamp DESC;

-- Performance analysis: avg execution time by operation type
SELECT operation_type, AVG(execution_time_ms) as avg_time, 
       COUNT(*) as count, MAX(execution_time_ms) as max_time
FROM agent_operations
GROUP BY operation_type;

-- Get operations from last hour
SELECT * FROM agent_operations 
WHERE timestamp > datetime('now', '-1 hour')
ORDER BY timestamp DESC;
```

---

### Table 6: `llm_token_usage`

**Table Purpose:**
Track LLM API token consumption and costs. Used for cost accounting, budget monitoring, and performance optimization.

**When Populated:**
- **After LLM calls**: During ingestion, retrieval, and healing
- **One row per LLM API call**: Indirect recording via agent_operations

**How Populated:**
```python
# Estimated during agent operations:
# 1. Query tokens = len(query.split()) or len(context.split())
# 2. Response tokens = len(response.split())
# 3. Total tokens = query + response
# 4. Cost estimated based on model pricing
```

**Data Structure:**
| Column | Type | Source | Purpose |
|--------|------|--------|---------|
| `token_id` | INTEGER (PK) | Auto-increment | Unique token tracking ID |
| `operation_id` | INTEGER (FK) | agent_operations.operation_id | Link to operation |
| `model_name` | TEXT | LLM config | Model used: gpt-4, ollama, etc. |
| `input_tokens` | INTEGER | LLM API response | Tokens in prompt |
| `output_tokens` | INTEGER | LLM API response | Tokens generated |
| `total_tokens` | INTEGER | Calculated | input + output tokens |
| `cost_usd` | REAL | Model pricing | Estimated API cost |
| `timestamp` | TEXT (timestamp) | System datetime | Recording time |

**Example:**
```sql
INSERT INTO llm_token_usage 
(operation_id, model_name, input_tokens, output_tokens, total_tokens, cost_usd, timestamp)
VALUES 
(123, 'text-embedding-3-small', 250, 120, 370, 0.0037, '2025-11-23T10:35:00');
```

**Relationships:**
- Foreign Key: operation_id → agent_operations.operation_id (ON DELETE CASCADE)
- No child tables

**Used By:**
- LLM service: INSERT (after API calls)
- Cost tracking: SELECT (for billing)
- Optimization: SELECT (identify expensive operations)

**Query Examples:**
```sql
-- Total tokens by model
SELECT model_name, SUM(total_tokens) as total, SUM(cost_usd) as cost
FROM llm_token_usage
GROUP BY model_name;

-- Cost by agent operation
SELECT ao.operation_type, COUNT(*) as calls, 
       SUM(ltu.total_tokens) as tokens, SUM(ltu.cost_usd) as cost
FROM llm_token_usage ltu
JOIN agent_operations ao ON ltu.operation_id = ao.operation_id
GROUP BY ao.operation_type;
```

---

### Table 7: `agent_memory`

**Table Purpose:**
Persistent memory storage for agent context and learning. Agents store important facts, user preferences, domain insights.

**When Populated:**
- **During operations**: record_agent_memory_tool()
- **Query-specific memory**: RetrievalAgent stores query results
- **Learning memory**: Agents store insights for future use

**How Populated:**
```python
# From: ingestion_tools.py -> record_agent_memory_tool()
# Called by: Agents to store important context

1. After processing
2. Store key-value memory
3. Mark importance score (0-1)
4. Tag memory type (short_term, long_term, episodic)
5. Access timestamp for aging/eviction
```

**Data Structure:**
| Column | Type | Source | Purpose |
|--------|------|--------|---------|
| `memory_id` | INTEGER (PK) | Auto-increment | Unique memory ID |
| `agent_id` | TEXT | Agent name | Agent storing memory |
| `memory_type` | TEXT | Agent logic | Type: short_term, long_term, episodic |
| `content` | TEXT (JSON) | Agent-specific | Memory content (serialized) |
| `importance_score` | REAL (0-1) | Agent logic | Importance weight (0=forget, 1=critical) |
| `created_at` | TEXT (timestamp) | System datetime | When stored |
| `last_accessed` | TEXT (timestamp) | System datetime | Last access (for eviction) |

**Memory Types:**
- `short_term`: Recent operations (context window), evicted after 1 hour
- `long_term`: Learned patterns, kept indefinitely
- `episodic`: Specific interactions, kept for analysis

**Example:**
```sql
-- RetrievalAgent storing query result
INSERT INTO agent_memory 
(agent_id, memory_type, content, importance_score, created_at, last_accessed)
VALUES 
('retrieval_agent_01', 'short_term',
 '{"query": "expense approval", "results": 5, "tokens": 124, "avg_relevance": 0.89}',
 0.7, '2025-11-23T10:35:00', '2025-11-23T10:35:00');

-- Agent storing learned insight
INSERT INTO agent_memory 
(agent_id, memory_type, content, importance_score, created_at, last_accessed)
VALUES 
('healing_agent_01', 'long_term',
 '{"insight": "Recursive chunking works best for policy documents", "effectiveness": 0.92}',
 0.95, '2025-11-23T10:30:00', '2025-11-23T10:35:00');
```

**Relationships:**
- No foreign keys (independent memory)
- No child tables

**Used By:**
- All agents: INSERT (store context)
- Agents: SELECT (recall context)
- Analytics: SELECT (analyze patterns)

**Query Examples:**
```sql
-- Get agent's long-term learning
SELECT content FROM agent_memory 
WHERE agent_id = 'healing_agent_01' AND memory_type = 'long_term'
ORDER BY importance_score DESC;

-- Get recent context (short-term)
SELECT content FROM agent_memory 
WHERE agent_id = 'retrieval_agent_01' AND memory_type = 'short_term'
ORDER BY last_accessed DESC;
```

---

### Table 8: `agent_spawns`

**Table Purpose:**
Track agent hierarchy and spawning relationships. Records when one agent creates another (dynamic spawning).

**When Populated:**
- **During agent spawning**: MasterOrchestrator.spawn_agent()
- **When agents create sub-agents**: For specialized tasks or optimization

**How Populated:**
```python
# From: master_orchestrator.py -> MasterOrchestrator.spawn_agent()
# From: ingestion_tools.py -> record_agent_spawn_tool()

1. Parent agent calls spawn_agent()
2. MasterOrchestrator creates child agent instance
3. Record spawn relationship
4. Store reason for spawning
5. Record timestamp
```

**Data Structure:**
| Column | Type | Source | Purpose |
|--------|------|--------|---------|
| `spawn_id` | INTEGER (PK) | Auto-increment | Unique spawn record ID |
| `parent_agent_id` | TEXT | Calling agent | Parent agent (nullable for root agents) |
| `child_agent_id` | TEXT | Created agent | Child agent name |
| `spawn_reason` | TEXT | Agent logic | Why: domain_specific_retrieval, quality_improvement |
| `timestamp` | TEXT (timestamp) | System datetime | When spawned |

**Example:**
```sql
-- IngestionAgent spawning HealingAgent for quality improvement
INSERT INTO agent_spawns 
(parent_agent_id, child_agent_id, spawn_reason, timestamp)
VALUES 
('ingestion_agent_01', 'healing_agent_temp_001', 
 'suggest_optimization', '2025-11-23T10:31:30');

-- RetrievalAgent spawning HealingAgent for retrieval optimization
INSERT INTO agent_spawns 
(parent_agent_id, child_agent_id, spawn_reason, timestamp)
VALUES 
('retrieval_agent_01', 'healing_agent_temp_002', 
 'retrieval_quality_improvement', '2025-11-23T10:35:30');
```

**Relationships:**
- No foreign keys (agent_id values are names, not DB references)
- No child tables

**Used By:**
- MasterOrchestrator: INSERT (when spawning)
- Debugging: SELECT (trace agent family tree)
- Analytics: SELECT (understand spawning patterns)

**Query Examples:**
```sql
-- Get all agents spawned by a parent
SELECT child_agent_id, spawn_reason, timestamp 
FROM agent_spawns 
WHERE parent_agent_id = 'ingestion_agent_01'
ORDER BY timestamp DESC;

-- Trace complete agent family for document
SELECT * FROM agent_spawns 
ORDER BY timestamp DESC LIMIT 20;
```

---

## Part 3: Data Flow Diagrams

### Ingestion Data Flow

```
User Input (File/Text/URL)
    ↓
MasterOrchestrator.ingest_data(data)
    ↓
    ├─→ IngestionAgent.ingest_document()
    │    ├─→ chunk_document_tool()
    │    ├─→ extract_metadata_tool() [LLM]
    │    └─→ save_to_vectordb_tool()
    │         ├─→ INSERT INTO documents
    │         ├─→ INSERT INTO document_metadata (domain, rbac_namespace, etc.)
    │         ├─→ INSERT INTO embedding_metadata (for each chunk)
    │         ├─→ INSERT INTO agent_operations (log ingestion)
    │         └─→ Vector DB: Add chunks with embeddings
    │
    ├─→ SyntheticQuestionsGenerator
    │    └─→ INSERT INTO synthetic_queries
    │
    └─→ HealingAgent (optional spawned for optimization)
         └─→ UPDATE embedding_metadata (quality_score)

Result: Document fully indexed with metadata, vectors, and synthetic Q&A
```

### Retrieval (Query) Data Flow

```
User Query
    ↓
MasterOrchestrator.ask_question(query, user_id)
    ↓
    ├─→ RetrievalAgent.process_query()
    │    ├─→ SELECT FROM document_metadata (RBAC filter by user_id)
    │    ├─→ LLM embedding: generate_embedding(query)
    │    ├─→ Vector DB: collection.query(embedding, top_k=5)
    │    │    └─→ Returns documents matching embedding
    │    ├─→ SELECT FROM embedding_metadata (get chunk metadata)
    │    ├─→ INSERT INTO agent_operations (log query)
    │    ├─→ INSERT INTO agent_memory (store query context)
    │    └─→ INSERT INTO llm_token_usage (token tracking)
    │
    └─→ Return results to user

Result: Relevant documents retrieved with RBAC enforcement and token cost recorded
```

### Healing/Optimization Data Flow

```
HealingAgent triggered (from IngestionAgent or RetrievalAgent)
    ↓
    ├─→ HealingAgent.optimize_document(doc_id, strategy)
    │    ├─→ SELECT FROM embedding_metadata (get chunks)
    │    ├─→ If strategy='reindex':
    │    │    └─→ UPDATE embedding_metadata (quality_score, reindex_count)
    │    ├─→ If strategy='resample':
    │    │    └─→ DELETE low-quality chunks, re-embed new ones
    │    ├─→ If strategy='reembed':
    │    │    └─→ UPDATE embedding_metadata (embedding_version, quality_score)
    │    ├─→ INSERT INTO agent_operations (log optimization)
    │    ├─→ INSERT INTO agent_memory (store optimization insight)
    │    └─→ UPDATE Vector DB (refresh embeddings)
    │
    └─→ RL System: Update Q-table with rewards

Result: Improved document quality and learned optimization policies
```

---

## Part 4: Complete ER Diagram

```
┌──────────────────────────────────────────────────────────────────────┐
│                     CORE RAG TABLES                                  │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │            DOCUMENTS (Master)                               │   │
│  ├─────────────────────────────────────────────────────────────┤   │
│  │ id (PK) │ title │ content │ source │ doc_type │ timestamps  │   │
│  └─────────────────────────────────────────────────────────────┘   │
│    │ (1:M ref)              │ (1:M ref)         │ (1:M ref)       │
│    ↓                        ↓                   ↓                 │
│  ┌────────────────┐   ┌──────────────────┐  ┌─────────────────┐  │
│  │DOCUMENT_META   │   │EMBEDDING_META    │  │SYNTHETIC_       │  │
│  │                │   │                  │  │QUERIES          │  │
│  │doc_id (FK)     │   │document_id (FK)  │  │doc_id (FK)      │  │
│  │key, value      │   │chunk_id (UNIQUE) │  │question, date   │  │
│  │[rbac_ns, tags] │   │quality_score     │  │                 │  │
│  └────────────────┘   │reindex_count     │  └─────────────────┘  │
│                       │embedding_version │                        │
│                       └──────────────────┘                        │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│               OPERATIONS & TRACKING TABLES                           │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────────────────────────────────────────────────┐       │
│  │  AGENT_OPERATIONS (Central Hub)                          │       │
│  ├──────────────────────────────────────────────────────────┤       │
│  │operation_id (PK)                                         │       │
│  │agent_id, operation_type, status                          │       │
│  │input/output_data, error, execution_time_ms, timestamp    │       │
│  └──────────────────────────────────────────────────────────┘       │
│    │ (1:M ref)                                                       │
│    ↓                                                                 │
│  ┌──────────────────────────────────────────────────────────┐       │
│  │  LLM_TOKEN_USAGE                                         │       │
│  ├──────────────────────────────────────────────────────────┤       │
│  │token_id (PK)                                             │       │
│  │operation_id (FK) ← links to operation                    │       │
│  │model_name, input/output/total_tokens, cost_usd, timestamp│       │
│  └──────────────────────────────────────────────────────────┘       │
│                                                                      │
│  ┌────────────────────────────┐  ┌────────────────────────────┐    │
│  │ AGENT_MEMORY               │  │ AGENT_SPAWNS               │    │
│  ├────────────────────────────┤  ├────────────────────────────┤    │
│  │memory_id (PK)              │  │spawn_id (PK)               │    │
│  │agent_id, memory_type       │  │parent_agent_id             │    │
│  │content (JSON), importance  │  │child_agent_id, reason      │    │
│  │created/last_accessed       │  │timestamp                   │    │
│  └────────────────────────────┘  └────────────────────────────┘    │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘

RELATIONSHIPS SUMMARY:

1. documents (1:M)
   ├─→ document_metadata
   ├─→ embedding_metadata
   └─→ synthetic_queries

2. agent_operations (1:M)
   └─→ llm_token_usage

3. Independents (No FK):
   ├─ agent_memory
   └─ agent_spawns
```

---

## Part 5: Database Interaction Patterns

### Pattern 1: RBAC-Filtered Retrieval

**Objective:** Get documents accessible to specific user

**Tables Used:** documents, document_metadata, embedding_metadata

```sql
-- Step 1: Get user's namespace
SELECT allowed_namespace FROM user_permissions 
WHERE user_id = 'user123';
-- Returns: 'finance_dept'

-- Step 2: Get accessible documents
SELECT DISTINCT d.id, d.title 
FROM documents d
JOIN document_metadata dm ON d.id = dm.doc_id
WHERE dm.key = 'rbac_namespace' AND dm.value = 'finance_dept';

-- Step 3: Get embeddings from accessible documents
SELECT em.chunk_id, em.quality_score
FROM embedding_metadata em
WHERE em.document_id IN (list from step 2)
ORDER BY em.quality_score DESC;
```

### Pattern 2: Quality Audit

**Objective:** Find documents/chunks needing quality improvement

**Tables Used:** embedding_metadata, agent_operations

```sql
-- Find low-quality chunks
SELECT em.document_id, COUNT(*) as low_quality_count,
       AVG(em.quality_score) as avg_quality
FROM embedding_metadata em
WHERE em.quality_score < 0.75
GROUP BY em.document_id
ORDER BY avg_quality;

-- Check if healing was applied
SELECT ao.* FROM agent_operations ao
WHERE ao.operation_type = 'optimize_document'
AND ao.output_data LIKE '%finance_policy_001%'
ORDER BY ao.timestamp DESC;
```

### Pattern 3: Cost Analysis

**Objective:** Analyze LLM costs by agent and operation

**Tables Used:** llm_token_usage, agent_operations

```sql
-- Cost by operation type
SELECT ao.operation_type, 
       COUNT(*) as operations,
       SUM(ltu.total_tokens) as total_tokens,
       SUM(ltu.cost_usd) as total_cost,
       AVG(ltu.cost_usd) as avg_cost
FROM llm_token_usage ltu
JOIN agent_operations ao ON ltu.operation_id = ao.operation_id
GROUP BY ao.operation_type
ORDER BY total_cost DESC;

-- Most expensive operations
SELECT ltu.*, ao.operation_type
FROM llm_token_usage ltu
JOIN agent_operations ao ON ltu.operation_id = ao.operation_id
ORDER BY ltu.cost_usd DESC
LIMIT 20;
```

### Pattern 4: Agent Performance Monitoring

**Objective:** Track agent performance and efficiency

**Tables Used:** agent_operations, agent_spawns

```sql
-- Agent execution times
SELECT agent_id, COUNT(*) as operations,
       AVG(execution_time_ms) as avg_time,
       MAX(execution_time_ms) as max_time,
       MIN(execution_time_ms) as min_time
FROM agent_operations
WHERE status = 'completed'
GROUP BY agent_id;

-- Error rate by agent
SELECT agent_id, 
       COUNT(*) as total_operations,
       SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failed,
       ROUND(100.0 * SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) / COUNT(*), 2) as error_rate
FROM agent_operations
GROUP BY agent_id;
```

---

## Part 6: Implementation Checklist

### For Ingestion Agent

- [ ] Read input data
- [ ] Insert into `documents` table
- [ ] Extract metadata → Insert into `document_metadata` (with rbac_namespace)
- [ ] Chunk document
- [ ] For each chunk:
  - [ ] Generate embedding
  - [ ] Insert into `embedding_metadata`
  - [ ] Add to Vector DB
- [ ] Generate synthetic questions → Insert into `synthetic_queries`
- [ ] Record operation → Insert into `agent_operations`
- [ ] Call `record_agent_memory_tool()` with ingestion context

### For Retrieval Agent

- [ ] Get user_id and query
- [ ] Check RBAC namespace from `document_metadata`
- [ ] Generate query embedding
- [ ] Search Vector DB (filtered by authorized documents)
- [ ] SELECT from `embedding_metadata` for context
- [ ] SELECT from `documents` for full content
- [ ] Estimate tokens → INSERT into `llm_token_usage`
- [ ] Record operation → INSERT into `agent_operations`
- [ ] Store query context → INSERT into `agent_memory`

### For Healing Agent

- [ ] Analyze document quality
- [ ] SELECT low-quality chunks from `embedding_metadata`
- [ ] Apply optimization strategy
- [ ] UPDATE `embedding_metadata` with new scores
- [ ] Record optimization → INSERT into `agent_operations`
- [ ] Store learned insight → INSERT into `agent_memory`
- [ ] Optionally spawn child healing agent → INSERT into `agent_spawns`

---

## Summary Table: Quick Reference

| Table | Rows per Document | Populated By | Primary Use | Query Type |
|-------|-------------------|--------------|-------------|-----------|
| documents | 1 | IngestionAgent | Master reference | SELECT by id |
| document_metadata | 5-10 | IngestionAgent + system | RBAC, metadata | SELECT by key |
| embedding_metadata | 10-50 | IngestionAgent | Vector quality | SELECT by doc_id |
| synthetic_queries | 3-10 | SyntheticQuestionsGenerator | QA testing | SELECT by doc_id |
| agent_operations | 1+ | All agents | Audit trail | SELECT by agent_id |
| llm_token_usage | 0.5 per op | LLM service | Cost tracking | SELECT by model |
| agent_memory | 2-5 | All agents | Context storage | SELECT by agent_id |
| agent_spawns | 0.1 per doc | MasterOrchestrator | Debugging | SELECT hierarchy |

---

## Conclusion

The RAG system uses **8 core metadata tables** to:
1. **Store & index** documents with RBAC
2. **Track quality** through embeddings and healing
3. **Audit operations** for transparency
4. **Learn** from synthetic questions and agent memory
5. **Measure costs** for LLM usage
6. **Enable spawning** of specialized agents

All tables follow **cascade delete** patterns and maintain **data integrity** through foreign key constraints. The system is **domain-agnostic** using generic metadata rather than domain-specific columns.
