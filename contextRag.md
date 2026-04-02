# Task Definition: Context-Conditioned RAG with RAPTOR, COIL, and Simple Contextual Retrieval

## 1. Objective
Implement an accuracy-first retrieval pipeline for this repository that combines:
- User idea: context-conditioned chunk embeddings (progressive context influence)
- Simple Contextual Retrieval (Anthropic style)
- COIL-style contextual lexical retrieval
- RAPTOR hierarchical retrieval

Do not optimize for memory usage in this task. Optimize for retrieval relevance, precision, and evidence quality.

## 2. Core Requirement (User Idea)
Implement progressive context-conditioned embeddings so chunk vectors depend on prior document context.

Target behavior:
- Two documents containing an identical sentence should produce different chunk embeddings when preceding context differs.

Required mechanism:
- For each chunk `c_i`, build a contextual input that includes:
  - Document-level context (title/source/section/time)
  - Rolling context from previous chunks `c_1 ... c_(i-1)` (summarized, bounded)
  - Current chunk text `c_i`
- Embed the contextual input, not only the raw chunk text.

Recommended representation:
- Keep both vectors:
  - `local_embedding = E(c_i)`
  - `contextual_embedding = E(context_i + c_i)`
- Use `contextual_embedding` for primary retrieval and keep `local_embedding` for diagnostics/ablation.

## 3. Techniques to Implement

### 3.1 Simple Contextual Retrieval (Anthropic style)
Implement a preprocessing step that generates concise chunk-specific context (about 50 to 120 tokens) using whole-document information, then prepend it before embedding and lexical indexing.

Store both:
- `chunk_text_raw`
- `chunk_text_contextualized`

Context generation rules:
- Include only grounding facts that disambiguate the chunk (who, when, what section, what metric/table)
- Avoid verbose summaries
- Keep deterministic formatting for reproducibility

### 3.2 COIL-style contextual lexical retrieval
Implement a contextualized exact-match lexical retriever inspired by COIL:
- Build an inverted index over exact query/document token overlap
- Store contextualized token representations for postings
- At query time, score by overlapping token matches with contextual token similarity

Minimum viable COIL module:
- Tokenization + normalization
- Posting lists keyed by token
- Contextual token vectors per chunk (from a contextual encoder)
- Query-time scoring over overlapping tokens only

### 3.3 RAPTOR hierarchical retrieval
Implement RAPTOR-style tree-organized retrieval:
- Leaf level: base chunks
- Recursive steps: embed, cluster, summarize, create parent nodes
- Build multiple abstraction levels until stopping criteria

Query-time RAPTOR behavior:
- Retrieve relevant higher-level nodes first
- Drill down to child nodes and leaf chunks
- Return evidence chunks with parent-summary provenance

## 4. Unified Retrieval Design
Implement a hybrid retrieval orchestrator with the following stages:

1. Candidate generation:
- Dense contextual retrieval
- COIL retrieval
- RAPTOR retrieval (summary nodes and/or leaves)

2. Fusion:
- Weighted RRF over candidate lists
- Configurable weights per retriever

3. Reranking:
- Rerank fused top-N with a stronger reranker (cross-encoder or LLM reranker)
- Return top-K final chunks/documents

4. Evidence packaging:
- Return chunk text + contextual prefix + metadata + source path + confidence/rank components

## 5. Metadata to Store (Required)
At minimum, store these fields for every chunk/node:
- `doc_id`
- `source`
- `path`
- `chunk_id`
- `chunk_index`
- `chunk_method`
- `section_title` (if available)
- `time_period` (if detectable)
- `entity_candidates` (if detectable)
- `context_prefix`
- `parent_node_id` (for RAPTOR)
- `tree_level` (for RAPTOR)

Metadata must be used in retrieval filtering/boosting and in final result rendering.

## 6. Repository Integration Constraints
- Keep existing scripts working:
  - `retrieval/fixed/*.py`
  - `retrieval/semantic/*.py`
  - `retrieval/agentic/*.py`
- Add new implementation in a separate module path, recommended:
  - `retrieval/context_rag/`

Recommended new files:
- `retrieval/context_rag/config.py`
- `retrieval/context_rag/loaders.py`
- `retrieval/context_rag/contextualize.py`
- `retrieval/context_rag/embeddings.py`
- `retrieval/context_rag/coil_index.py`
- `retrieval/context_rag/raptor_index.py`
- `retrieval/context_rag/fusion.py`
- `retrieval/context_rag/rerank.py`
- `retrieval/context_rag/pipeline.py`
- `retrieval/context_rag/eval.py`

## 7. Implementation Phases

### Phase A: Baseline context-conditioned indexing
- Build chunker
- Build contextual prefix generator
- Store raw + contextualized chunk text
- Index contextualized chunks in vector store

### Phase B: COIL module
- Build contextualized lexical index and scorer
- Expose retrieval API returning ranked chunks

### Phase C: RAPTOR module
- Build recursive tree from chunks
- Index tree nodes and leaves
- Implement hierarchical query traversal

### Phase D: Hybrid fusion + reranking
- Add weighted RRF fusion across Dense/COIL/RAPTOR
- Add reranker for final top-N refinement

### Phase E: Evaluation and regression safety
- Add evaluation harness and report generation
- Validate no breakage of existing retrieval scripts

## 8. Acceptance Criteria
The task is complete only if all criteria below pass.

1. Functional criteria:
- Can build index and run retrieval for all four components:
  - context-conditioned dense retrieval
  - Simple Contextual Retrieval preprocessing
  - COIL
  - RAPTOR

2. Behavior criteria:
- Demonstrate that identical sentence in different prior contexts yields different contextual embeddings.
- Demonstrate improved retrieval relevance on an evaluation set compared to at least one existing baseline script.

3. Evaluation criteria:
- Report at least:
  - Recall@5
  - Recall@20
  - MRR@10
  - nDCG@10
- Provide per-query comparison table against baseline.

4. Quality criteria:
- Retrieval outputs include provenance metadata.
- Fusion weights and retrieval modes are configurable via CLI/env.

## 9. Required Tests
Implement tests for:
- Context conditioning correctness:
  - Same target sentence + different prior context -> different contextual embeddings
- COIL scoring behavior on exact token overlap
- RAPTOR tree construction and parent-child traversal
- End-to-end retrieval pipeline smoke test
- Fusion and reranker integration smoke test

## 10. CLI/Entrypoints (Required)
Provide runnable commands (or equivalent task scripts) for:
- Index build
- Query run
- Evaluation run

Example target interface:
- `python -m retrieval.context_rag.pipeline build-index --texts-dir texts`
- `python -m retrieval.context_rag.pipeline query --query "..." --top-k 10`
- `python -m retrieval.context_rag.eval --eval-file docs/eval/context_rag_eval.json`

## 11. Non-Goals
- Memory compression and storage footprint tuning
- Hardware-specific optimization
- UI/dashboard work

## 12. Final Deliverables
- Source code modules for context-conditioned retrieval, COIL, and RAPTOR
- Configurable hybrid pipeline with fusion and reranking
- Evaluation script and metric report
- Short implementation note in README or docs section describing:
  - architecture
  - how to run
  - observed metric deltas
