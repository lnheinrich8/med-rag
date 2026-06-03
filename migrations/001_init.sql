-- 001_init.sql — core schema for the med-rag corpus + chunk store.
--
-- Design notes:
--   * One row per source document, one row per chunk.
--   * `embedding` is fixed at 768 dims because both candidate embedding models
--     (BAAI/bge-base-en-v1.5 and NeuML/pubmedbert-base-embeddings) output 768.
--     Comparing embedding models in eval means re-embedding into this column.
--   * `tsv` is a generated full-text column => free sparse/BM25-style retrieval
--     in the same DB, which we fuse with dense vectors (RRF) for hybrid search.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    id             BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    source_path    TEXT NOT NULL UNIQUE,        -- absolute path under data/raw
    source_type    TEXT NOT NULL,               -- guideline | statpearl | review
    title          TEXT,                        -- from PDF metadata or first heading
    section_number TEXT,                        -- e.g. "9" for ADA SoC sections
    n_pages        INT,
    n_chunks       INT NOT NULL DEFAULT 0,
    content_hash   TEXT,                         -- for idempotent re-ingest
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS chunks (
    id           BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    document_id  BIGINT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index  INT NOT NULL,                  -- order within the document
    content      TEXT NOT NULL,
    token_count  INT,
    page_start   INT,
    page_end     INT,
    section      TEXT,                          -- heading the chunk falls under
    embedding    VECTOR(768),
    tsv          TSVECTOR GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (document_id, chunk_index)
);

-- Dense ANN index (cosine). Embeddings are L2-normalized at insert time.
CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw
    ON chunks USING hnsw (embedding vector_cosine_ops);

-- Sparse full-text index for the BM25-style leg of hybrid retrieval.
CREATE INDEX IF NOT EXISTS chunks_tsv_gin
    ON chunks USING gin (tsv);

CREATE INDEX IF NOT EXISTS chunks_document_id
    ON chunks (document_id);
