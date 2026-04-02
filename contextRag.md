# Build a Context-Conditioned Retrieval POC

You are a coding agent working in this repository. Build an accuracy-first retrieval POC that implements and integrates:
- Context-conditioned chunk embeddings (core idea)
- Simple Contextual Retrieval (Anthropic style)
- COIL-style contextual lexical retrieval
- RAPTOR-style hierarchical retrieval

## Mission
Deliver a working end-to-end proof of concept that improves retrieval relevance and evidence grounding for user queries.

## Why This Work Exists
This task exists to address a concrete retrieval failure mode:
- Chunk-only embeddings often lose why a chunk exists in the document narrative.
- Identical local sentences can have different meaning when prior context differs.
- Dense retrieval alone may underweight exact anchors (names, dates, codes, financial line items).

Your implementation must preserve context and improve the chance of retrieving chunks/documents that are actually important and relevant to the user intent. The Attention Is All You Need paper was an inspiration for this idea.

## Mandatory Constraints
- Use UV as the Python project manager for dependency installation and command execution.
- Keep existing retrieval scripts functional.
- Prioritize POC usability and correctness over system optimization.
- Memory/compression/storage optimization is out of scope for this stage.

## Target Outcomes
Your POC is successful when all outcomes below are true:

1. Context-conditioned dense retrieval is implemented and queryable.
2. Simple Contextual Retrieval is implemented so chunk-specific context influences retrieval.
3. COIL-style contextual lexical retrieval is implemented and contributes candidates.
4. RAPTOR-style hierarchical retrieval is implemented and contributes candidates from multiple abstraction levels.
5. A unified retrieval path combines these signals into one final ranking.
6. Returned results contain provenance metadata (at least source/path/chunk identity and rank context).
7. The system can demonstrate that context-conditioned behavior is observable (for example, same local sentence under different prior context does not collapse to identical retrieval behavior).

## Implementation Latitude
Do not follow rigid, predefined implementation steps.
- Choose architecture, data structures, and orchestration strategy yourself.
- Decide how to represent context and how to combine retriever outputs.
- Decide how much to build in this POC as long as all target outcomes are met.

## Evaluation Priority for This Stage
Full benchmark evaluation is not required now.

For this phase, provide:
- Working index/build path
- Working query path
- Lightweight validation (smoke tests and targeted sanity checks)
- Short handoff note: what works, what is partial, what should be evaluated next

Defer full retrieval benchmark suite (Recall@k, MRR, nDCG) to a later phase.

## Deliverables
Provide the following by the end of this stage:
- Runnable POC under UV
- Integrated retrieval output with provenance
- Minimal validation artifacts showing the POC works end-to-end
- Concise technical summary of implementation decisions and next steps

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

## Definition of Done for This Prompt
Done means a working POC exists under UV, all four retrieval concepts are present in integrated form, and the output demonstrates practically improved contextual grounding on representative queries.
