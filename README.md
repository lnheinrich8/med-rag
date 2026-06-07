# med-rag

A local, **evaluation-focused** Retrieval-Augmented Generation CLI over a Type 2
Diabetes corpus (ADA Standards of Care 2026, StatPearls, PMC open-access reviews).

Everything runs locally: Postgres + pgvector for storage and hybrid retrieval,
sentence-transformers for embeddings/reranking, and a quantized Qwen2.5 model
served by Ollama for generation. The point of the project is the **eval harness** —
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
rag query  "..."            # Step 3: one-shot cited answer
rag chat                    # interactive REPL: ask question after question
rag eval-gen -n 50          # Step 4: draft a gold Q&A set for hand-verification
rag eval                    # Step 4: score retrieval + generation, write a report
rag ablate -c a.yaml -c b.yaml   # Step 6: ingest+eval several configs, compare
rag report-compare r1.json r2.json   # diff saved reports side by side
```

### Chat

`rag chat` opens an interactive session — type a question (no quotes, no command
prefix) and press Enter to get a cited answer, then ask the next one. The
embedding/rerank/LLM models stay warm across turns, so only the first answer pays
the load cost.

```bash
rag chat                              # uses configs/default.yaml
rag chat -c configs/experiments/tuned.yaml   # run under a specific pipeline
rag chat --show-context               # also print the chunks behind each answer
```

Ctrl+C clears the current line; pressing Ctrl+C again on an empty line (or Ctrl+D)
exits.

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
- [x] **Step 6 — Ablations + report:** one-command `rag ablate` (ingest+eval+compare), cross-config comparison reports. Headline **baseline → tuned** (naive 512-token dense vs 256-token hybrid+rerank): recall@5 0.79→0.97, MRR 0.62→0.76, abstentions 5.9%→2.9%, faithfulness 4.53→4.73, correctness 4.09→4.36 — and faster end-to-end (generation 2.6s→1.9s p50).
- [x] **Interactive chat:** `rag chat` REPL over the same RAG pipeline, with models kept warm across turns.

### Headline result

| metric | baseline (naive) | tuned | Δ |
| --- | ---: | ---: | ---: |
| recall@5 | 0.794 | 0.971 | +0.177 |
| MRR | 0.616 | 0.764 | +0.148 |
| abstention rate | 5.9% | 2.9% | −2.9pp |
| faithfulness (1-5) | 4.53 | 4.73 | +0.20 |
| correctness (1-5) | 4.09 | 4.36 | +0.27 |
| generation p50 | 2.6s | 1.9s | −0.7s |

Better retrieval (hybrid + reranker over 256-token cleaned chunks) put the answer
in context more often, which cut refusals and raised faithfulness/correctness —
while generating *faster*, because cleaner, on-topic context means less hedging.
Judge scores are self-judged (see the Evaluation note) and the gold set is 34
hand-verified questions, so the generation deltas are directional; the retrieval
deltas are exact.
