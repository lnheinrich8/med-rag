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
| LLM            | `Qwen2.5-7B-Instruct` Q4 GGUF via `llama-cpp-python`          |

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
pip install -e ".[llm]"     # Step 3: local LLM
```

## CLI

```bash
rag info                    # config + DB health
rag init-db                 # run migrations
rag config-show -c configs/default.yaml
rag ingest [PATH]           # Step 2
rag search "..."            # Step 2
rag query  "..."            # Step 3
rag eval                    # Step 4
```

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
- [ ] Step 2 — Ingest + dense search
- [ ] Step 3 — Local LLM + cited answers
- [ ] Step 4 — Eval harness + gold set
- [ ] Step 5 — Hybrid + reranker
- [ ] Step 6 — Ablations + report
