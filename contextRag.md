# Build a Context-Conditioned Retrieval POC

You are a coding agent working in this repository. Build an accuracy-first retrieval POC that implements and integrates:
- Context-conditioned chunk embeddings (core idea)
- Simple Contextual Retrieval (Anthropic style)
- COIL-style contextual lexical retrieval
- RAPTOR-style hierarchical retrieval

## Mission
Deliver a working end-to-end retrieval prototype that improves the chance of returning chunks/documents that are actually relevant and important to user intent and query, with strong provenance and explainability.

## Repository Context You Should Assume
- The repository already contains multiple retrieval variants under `retrieval/fixed`, `retrieval/semantic`, and `retrieval/agentic`.
- Source corpus consists of many markdown documents in `texts/`, including financial and filing-like content where context (entity, period, section) strongly affects meaning.
- Existing scripts should remain usable and not be broken by this work.

## Why This Work Exists (Problem Context)
This POC targets a real retrieval gap:
- Chunk-only embeddings often lose the broader reason a chunk exists in the document.
- Identical local sentences can mean different things under different preceding context.
- Dense retrieval alone can miss exact anchors (entity names, dates, line items, identifiers).
- Flat chunk retrieval can miss cross-section/global relevance in long documents.

The core inspiration comes from context-sensitive modeling ideas (Attention Is All You Need and follow-on work): meaning should depend on surrounding context, not only local chunk text.

## Practical Relevance Targets for This Repository
Assume the corpus contains many long, structured markdown documents (for example, filing-like reports) where context cues are often distributed.

Your retrieval should improve these concrete situations:
- Numeric/line-item questions where the right chunk depends on section context, not keyword overlap alone.
- Queries with exact anchors (codes, entity names, period labels) that should not be diluted by semantic-only matching.
- Cross-section questions where the best evidence is not in one isolated chunk.
- Ambiguous sentences whose interpretation depends on document-level context.

Treat these as target use cases when choosing tradeoffs.

## Hard Constraints
- Use UV as the Python project manager for dependency installation and command execution.
- Keep existing retrieval scripts functional.
- Prioritize correctness and retrieval quality over optimization.
- Memory/compression/storage optimization is explicitly out of scope for this stage.
- Full benchmark evaluation is not required at this stage.

## POC Success Goals (What Must Exist)
Your POC is successful when all goals below are met.

1. Context-conditioned dense retrieval exists and is queryable.
2. Simple Contextual Retrieval exists: chunk-specific context is generated/attached and influences retrieval.
3. COIL-style contextual lexical retrieval exists and contributes meaningful candidates.
4. RAPTOR-style hierarchical retrieval exists and contributes candidates from multiple abstraction levels.
5. A unified ranking path combines outputs from these components into one final result list.
6. Returned results include provenance metadata and enough scoring context to debug ranking behavior.
7. A demonstration exists showing context-conditioned behavior is observable (for example, same local sentence in different prior contexts does not behave as context-free retrieval).

## Expected User-Facing Behavior
When a query is executed, output should make it obvious why results were returned.

At minimum, each result should communicate:
- What evidence text was retrieved
- Where it came from (document/chunk provenance)
- Which retrieval signals contributed (dense, contextual lexical, hierarchical)
- Why it ranked where it did (even if approximate or summarized)

The goal is not only ranking quality, but also inspectability.

## Implementation Latitude (Important)
Do not follow rigid prescribed implementation steps.
- You decide architecture, data model, orchestration, and component boundaries.
- You decide command naming and module layout.
- You decide the depth of each component as long as the success goals are satisfied.
- You may seek approval and review for your plan.

## Quality Expectations for This POC
Focus on practical retrieval quality signals, not exhaustive evaluation.

At minimum, ensure:
- End-to-end indexing and querying works reliably.
- Query outputs are inspectable and grounded.
- Candidate fusion behavior is explainable enough for debugging.
- Component contributions can be inspected (dense vs lexical vs hierarchical).

## Suggested Retrieval Logic Principles (Non-Binding)
These are principles, not fixed steps:
- Prefer recall in early candidate generation, then precision in fusion/rerank.
- Preserve exact-match strength for anchor-heavy queries.
- Preserve context sensitivity for semantically similar but context-different chunks.
- Prefer provenance-rich outputs over opaque scores.

## Metadata Expectations
Store and return enough metadata to preserve context and support diagnostics. Minimum expected categories:
- Source identity (doc/source/path)
- Chunk identity (chunk id/index/order)
- Context identity (context prefix or equivalent contextual representation)
- Hierarchy identity (parent/level linkage for RAPTOR-like retrieval)
- Ranking identity (at least one per-component scoring signal or rank trace)

You may extend schema as needed.

## Minimal Validation Expectations (POC-Level)
Provide lightweight but meaningful evidence the system works:
- Smoke tests for indexing/query flow.
- Targeted sanity checks for at least these query types:
	- Exact-anchor query (identifier/name/date heavy)
	- Semantic paraphrase query
	- Context-disambiguation query (same sentence, different prior context)
	- Long-context query where hierarchical evidence helps
- Short write-up of observed behavior and known gaps.

Include at least one before/after comparison against a current baseline path in this repository to show practical improvement, even if informal.

## Output Contract
The final POC should expose:
- A reproducible index/build path under UV
- A reproducible query path under UV
- Human-readable retrieval output with provenance and rank context
- A concise implementation note that explains:
	- What is implemented
	- What is partial
	- What should be evaluated in the next phase

Equivalent command interfaces are acceptable; exact command names are your choice.

## Non-Goals for This Stage
- Memory footprint and low-level performance tuning
- Production hardening and deployment concerns
- Full-scale metrics benchmarking suite
- UI/dashboard work

## Deliverables
Provide all of the following:
- Runnable POC under UV
- Integrated retrieval path combining context-conditioned dense, contextual lexical, and hierarchical signals
- Lightweight validation artifacts and sample outputs
- Technical summary and next-phase evaluation plan

## Failure Handling and Fallback Expectations
If one advanced component is partially complete:
- Do not block the full POC.
- Expose degraded but functional behavior and clearly label what is active/inactive.
- Keep output contract stable so future upgrades can be plugged in without breaking usage.

Robust partial functionality is preferred over fragile all-or-nothing behavior.

## References (Use as Relevant)

### Foundation and Context Modeling
- Attention Is All You Need (Vaswani et al., 2017): https://arxiv.org/abs/1706.03762
- Transformer-XL (Dai et al., 2019): https://arxiv.org/abs/1901.02860

### Contextual and Late-Interaction Retrieval
- ColBERT (Khattab and Zaharia, 2020): https://arxiv.org/abs/2004.12832
- ColBERTv2 (Santhanam et al., 2022): https://arxiv.org/abs/2112.01488
- COIL (Gao et al., 2021): https://arxiv.org/abs/2104.07186

### Contextual Retrieval
- Anthropic, Introducing Contextual Retrieval (2024): https://www.anthropic.com/news/contextual-retrieval

### Hierarchical Retrieval
- RAPTOR (Sarthi et al., 2024): https://arxiv.org/abs/2401.18059

## Definition of Done (POC Stage)
Done means:
- The system runs end-to-end under UV.
- All four target concepts are present in integrated working form.
- Retrieval output shows practical contextual grounding improvements on representative sanity queries.
- A concise handoff note documents current status and next evaluation priorities.

## Next Phase (After This POC)
Do not implement this now unless it is low effort, but prepare for it:
- Structured offline evaluation (Recall@k, MRR, nDCG)
- Ablation studies per retrieval component
- Query-type-aware weighting/fusion analysis
- Error taxonomy for missed retrievals and false positives
