# med-rag

A local, **evaluation-focused** Retrieval-Augmented Generation CLI over a Type 2
Diabetes corpus (ADA Standards of Care 2026, StatPearls, PMC open-access reviews).

Everything runs locally: Postgres + pgvector for storage and hybrid retrieval,
sentence-transformers for embeddings/reranking, and a quantized Qwen2.5 model via
llama.cpp for generation. The point of the project is the **eval harness** —
measuring retrieval quality, hallucination/faithfulness, and latency, and using
it to compare pipeline configurations (chunking, embeddings, dense vs hybrid,
reranking) as reproducible ablations.

## Stack

| Concern        | Choice                                                        |
| -------------- | ------------------------------------------------------------ |
| Vector store   | Postgres 18 + pgvector (HNSW dense + `tsvector` sparse)       |
| Embeddings     | `BAAI/bge-base-en-v1.5` (vs `NeuML/pubmedbert-base` in eval)  |
| Reranker       | `BAAI/bge-reranker-base` cross-encoder                        |
| LLM            | `Qwen2.5-7B-Instruct` Q4 GGUF served by Ollama (local, GPU)   |

## Setup

```bash
# Python 3.11 venv (already created as .venv)
source .venv/bin/activate
pip install -e .            # core CLI + DB layer
cp .env.example .env        # adjust DB URL if needed

rag init-db                 # apply migrations into the `medrag` database
rag info                    # show config + DB/pgvector health
```

Heavy ML deps install per build step:

```bash
pip install -e ".[embed]"   # Steps 2 & 5: embeddings + reranker
pip install -e ".[llm]"     # Step 3: Ollama python client
```

The LLM is served by [Ollama](https://ollama.com) (prebuilt GPU runtime). One-time setup:

```bash
# install + start Ollama (Arch: sudo pacman -S ollama; then `ollama serve` or the service)
# register the downloaded GGUF as the model named in configs/*.yaml:
ollama create medrag-qwen -f Modelfile
ollama list                 # should show medrag-qwen
```

## CLI

```bash
rag info                    # config + DB health
rag init-db                 # run migrations
rag config-show -c configs/default.yaml
rag ingest [PATH]           # Step 2
rag search "..."            # Step 2
rag query  "..."            # Step 3
rag eval-gen -n 50          # Step 4: draft a gold Q&A set for hand-verification
rag eval                    # Step 4: score retrieval + generation, write a report
rag ablate -c a.yaml -c b.yaml   # Step 6: ingest+eval several configs, compare
rag report-compare r1.json r2.json   # diff saved reports side by side
```

### Evaluation (Step 4)

The eval harness is the point of the project — it measures a pipeline config so
later steps can be justified with evidence instead of vibes.

```bash
rag eval-gen -n 50          # local model drafts Qs → data/gold/diabetes_qa.draft.jsonl
# hand-verify: fix answers/relevant spans, set "verified": true,
# save the kept lines as data/gold/diabetes_qa.jsonl
rag eval                    # full run: retrieval metrics + answers + LLM-as-judge
rag eval --no-generate      # retrieval metrics only (fast, no LLM)
rag eval --verified-only    # score only questions you've reviewed
```

What it reports, per config, into `reports/<config>_<timestamp>.{json,md}`:

- **Retrieval** — Recall@k, Precision@k, nDCG@k, MRR. Relevance is keyed to
  `(source file, page range)`, **not** chunk ids, so the same gold set scores
  every ablation fairly even after re-chunking/re-embedding.
- **Generation** — abstention rate, citations/answer, faithfulness & correctness
  (1-5, LLM-as-judge). The judge is the local generator model, so absolute
  judge scores carry a self-preference bias and are best read *relatively*
  across configs (disclosed in every report).
- **Latency** — retrieval & generation p50/p95.

## Layout

```
migrations/      SQL schema (documents, chunks)
configs/         experiment configs (the swappable pipeline)
data/raw/        corpus PDFs (gitignored): guidelines / statpearls / reviews
src/
  config.py      Settings (env) + ExperimentConfig (yaml)
  db.py          connection + migration runner
  cli.py         `rag` entrypoint
  ingest/ embed/ store/ retrieve/ generate/ eval/
```

## Build progress

- [x] **Step 1 — Foundation:** venv, deps, pgvector schema, config, CLI skeleton
- [x] **Step 2 — Ingest + dense search:** loaders, chunkers, embedder, store, `rag ingest`/`search`
- [x] **Step 3 — Local LLM + cited answers:** prompt, llama.cpp wrapper, RAG pipeline, `rag query`
- [x] **Step 4 — Eval harness + gold set:** metrics, gold drafting/loading, LLM-as-judge, runner, report, `rag eval`/`eval-gen` (gold set pending hand-verification)
- [x] **Step 5 — Hybrid + reranker + chunk cleanup:** sparse FTS, RRF hybrid, cross-encoder rerank, boilerplate stripping, chunk-size/section ablations (256-token chunks won: recall@20 0.882→0.971)
- [ ] Step 6 — Ablations + report
