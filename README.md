# Advanced RAG: Chunking, Retrieval, Reranking & Evaluation

A month-long, hands-on build of a full Retrieval-Augmented Generation (RAG) pipeline 
from chunking strategy comparison through hybrid retrieval, reranking, and a
RAGAS evaluation harness. Built as the Month 1 capstone project of a
6-month AI learning roadmap.

## What this project does

Rather than picking one chunking method or retriever and hoping it works, this
project systematically compares options at every stage of the pipeline and
measures the impact with real evaluation metrics not just "it looks fine."

**Pipeline stages covered:**

1. **Chunking**: 7 strategies (fixed, recursive, sliding window, semantic,
   parent-child, markdown-aware, HTML-aware) benchmarked across 4 source
   documents at multiple target chunk sizes, scored on structural quality
   metrics (chunk count, length distribution, mid-sentence cut rate).
2. **Retrieval**: BM25 (sparse), dense (embedding-based via ChromaDB), and
   hybrid (Reciprocal Rank Fusion of both).
3. **Reranking**: BGE reranker (`bge-reranker-v2-m3`) as an optional
   post-retrieval step.
4. **Evaluation**: RAGAS (faithfulness, answer relevancy, context precision,
   context recall) with a DeepEval cross-check on a sample, run across all
   6 retrieval × reranking configurations.
5. **Synthetic test-set generation**: RAGAS `TestsetGenerator` with custom
   personas, generated incrementally per source document.


All code chunking harness, indexing, retrieval, reranking, synthetic
test-set generation, and evaluation lives in a single notebook,
`rag_pipeline.ipynb`, It was built incrementally over the course of the month, so
later cells depend on variables and functions defined earlier in the same
notebook (run top to bottom).

## How it works

**1. Chunk the source documents**
Run the chunking section of the notebook. It applies all 7 applicable
strategies to each source document, prints a structural comparison, and
saves every chunk (with source, page, and method metadata) to
`comparison_results_chunks.json`.

**2. Pick a winning chunking strategy**
Reviewed structural metrics (chunk size consistency, mid-sentence cut rate)
across all 7 methods. See Findings below for what won and why.

**3. Index for retrieval**
Chunks are embedded (`all-MiniLM-L6-v2`) and indexed in ChromaDB for dense
search, and separately indexed with BM25 for sparse search. Hybrid search
fuses both rankings via Reciprocal Rank Fusion.

**4. Generate a synthetic test set**
RAGAS `TestsetGenerator` with custom personas (ML researcher, security
analyst, software developer, curious reader) generates realistic questions
per source document, combined with 12 hand-written questions covering
exact-match, conceptual, multi-hop, and narrative query types.

**5. Evaluate every configuration**
All 6 combinations of {bm25, dense, hybrid} × {rerank on/off} are run through
the same query set and scored on faithfulness, answer relevancy, context
precision, and context recall.

## Key findings

- **Semantic chunking produced the cleanest sentence boundaries** (as low as
  0% mid-sentence cuts vs. 60–100% for other methods), but fragmented text
  into much smaller chunks than the 512-token target a real trade-off
  between clean boundaries and adequate context per chunk.
- **Reranking improved retrieval quality in every configuration tested** 
  context precision and recall both increased when reranking was applied,
  most notably for BM25 and hybrid retrieval.
- **BM25+rerank and hybrid+rerank were the strongest configs**, roughly tied,
  both reaching ~0.87 context precision and 1.0 context recall.
- **Dense-only retrieval was the weakest baseline** in this test lowest
  precision and recall of all six configurations.
- Some metric drops (e.g. `answer_relevancy` on short factual answers,
  `faithfulness` on "I don't know" refusals) were traced to known metric
  quirks rather than real generation failures documented in the eval
  notebook's case-by-case findings.

## Known limitations

- Full evaluation was run on a reduced set of ~10 queries due to Gemini
  free-tier API quota limits, rather than the originally planned 40–60
  query synthetic + hand-written set. Results are directionally consistent
  but not statistically validated at scale noted as future work.
- The evaluation judge model (`gemini-2.5-flash`) was also used as the
  generator for part of the run, which carries a known self-grading bias
  risk; DeepEval cross-checks were used to partially mitigate this.

## Stack

`langchain-text-splitters` · `pypdf` · `sentence-transformers` · `chromadb` ·
`bm25s` · `FlagEmbedding` (BGE reranker) · `ragas` · `deepeval` ·
`langchain-google-genai` (Gemini) · `pandas`

## Future work

- Re-run the full evaluation at the originally planned scale (40–60 queries)
  once quota allows.
- Formally benchmark semantic chunking against recursive chunking through
  the full retrieval pipeline (not just structural metrics) to resolve the
  chunk-size trade-off.
- Add a judge model independent of the generator to remove self-grading bias.
