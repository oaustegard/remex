# Integration Guide: polar-embed with pgVector

This guide shows how to use polar-embed for embedding compression in production systems backed by PostgreSQL/pgVector. Three patterns are covered, from simplest to most sophisticated.

## Prerequisites

```bash
pip install polar-embed psycopg2-binary numpy
# For generating embeddings:
pip install sentence-transformers
```

```sql
-- PostgreSQL with pgVector extension
CREATE EXTENSION IF NOT EXISTS vector;
```

## Pattern 1: Compressed storage with Python-side search

Store compressed vectors in PostgreSQL as binary blobs. All search happens in Python using polar-embed. This gives maximum compression and full control over search strategy.

**When to use**: Small-to-medium corpora (<500k vectors), memory-constrained environments, or when you want training-free compression without reindexing.

### Schema

```sql
CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    content TEXT NOT NULL,
    -- Store compressed indices + norms as binary
    compressed_indices BYTEA NOT NULL,
    compressed_norms BYTEA NOT NULL,
    -- Metadata
    embedding_dim INTEGER NOT NULL,
    bits INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);
```

### Encode and store

```python
import numpy as np
import psycopg2
from sentence_transformers import SentenceTransformer
from polar_embed import PolarQuantizer

# Setup
model = SentenceTransformer("all-MiniLM-L6-v2")
d = model.get_sentence_embedding_dimension()  # 384
pq = PolarQuantizer(d=d, bits=4)

# Encode documents
documents = ["First document text...", "Second document text...", ...]
embeddings = model.encode(documents).astype(np.float32)
compressed = pq.encode(embeddings)

# Store to PostgreSQL
conn = psycopg2.connect("dbname=mydb")
cur = conn.cursor()

for i, doc in enumerate(documents):
    cur.execute(
        """INSERT INTO documents (content, compressed_indices, compressed_norms,
                                  embedding_dim, bits)
           VALUES (%s, %s, %s, %s, %s)""",
        (
            doc,
            compressed.indices[i].tobytes(),
            compressed.norms[i:i+1].tobytes(),
            d,
            4,
        ),
    )
conn.commit()
```

### Load and search

```python
from polar_embed import PolarQuantizer, CompressedVectors

pq = PolarQuantizer(d=384, bits=4)

# Load all compressed vectors from DB
cur.execute("SELECT id, compressed_indices, compressed_norms FROM documents ORDER BY id")
rows = cur.fetchall()

ids = [r[0] for r in rows]
indices = np.array([np.frombuffer(r[1], dtype=np.uint8).reshape(384) for r in rows])
norms = np.array([np.frombuffer(r[2], dtype=np.float32)[0] for r in rows])

compressed = CompressedVectors(indices, norms, d=384, bits=4)

# Search
query_embedding = model.encode("search query").astype(np.float32)
result_idx, scores = pq.search(compressed, query_embedding, k=10)
result_ids = [ids[i] for i in result_idx]
```

### Memory-efficient variant with ADC

For large corpora where you can't cache the dequantized matrix:

```python
# Uses lookup-table scoring — 5x less RAM, no persistent float32 cache
result_idx, scores = pq.search_adc(compressed, query_embedding, k=10)

# Or two-stage: fast coarse scan + precise rerank on candidates
result_idx, scores = pq.search_twostage(
    compressed, query_embedding, k=10, candidates=200, coarse_precision=2
)
```

---

## Pattern 2: pgVector for coarse retrieval, polar-embed for reranking

Store full-precision embeddings in pgVector for coarse kNN search, then use polar-embed's compressed representation for fast reranking or caching. This combines pgVector's indexing (IVFFlat/HNSW) with polar-embed's compression.

**When to use**: Large corpora (>500k vectors), when you need sublinear search, or when pgVector is already in your stack.

### Schema

```sql
CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    content TEXT NOT NULL,
    embedding vector(384) NOT NULL  -- pgVector type
);

-- HNSW index for fast approximate search
CREATE INDEX ON documents USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 200);
```

### Query flow

```python
import numpy as np
import psycopg2
from polar_embed import PolarQuantizer, CompressedVectors

# Step 1: pgVector coarse retrieval (sublinear via HNSW)
query_embedding = model.encode("search query").astype(np.float32)
query_list = query_embedding.tolist()

cur.execute(
    """SELECT id, content, embedding
       FROM documents
       ORDER BY embedding <=> %s::vector
       LIMIT 200""",  -- coarse candidates
    (query_list,),
)
candidates = cur.fetchall()

# Step 2: polar-embed compressed reranking
candidate_embeddings = np.array([row[2] for row in candidates], dtype=np.float32)
pq = PolarQuantizer(d=384, bits=8)  # 8-bit for high-quality rerank
compressed = pq.encode(candidate_embeddings)
rerank_idx, scores = pq.search(compressed, query_embedding, k=10)

results = [candidates[i] for i in rerank_idx]
```

This pattern is useful when you want to apply custom scoring logic, combine multiple signals, or rerank with a different metric than what pgVector indexes.

---

## Pattern 3: In-app embedding cache with polar-embed

Keep a compressed in-memory cache of your most-accessed embeddings, backed by PostgreSQL for persistence. Queries hit the cache first; cache misses fall through to the database.

**When to use**: Read-heavy workloads, frequently queried subsets, or when you want sub-millisecond search on a hot set.

### Architecture

```
Query → In-memory cache (polar-embed compressed)
          ↓ cache miss
        PostgreSQL (pgVector full embeddings)
          ↓ result
        Update cache with accessed vectors
```

### Implementation

```python
import numpy as np
import time
from polar_embed import PolarQuantizer, CompressedVectors


class EmbeddingCache:
    """In-memory compressed embedding cache backed by PostgreSQL."""

    def __init__(self, d: int, bits: int = 8, max_size: int = 100_000):
        self.pq = PolarQuantizer(d=d, bits=bits)
        self.d = d
        self.max_size = max_size

        # Cache state
        self._ids = []               # document IDs in cache order
        self._id_to_pos = {}         # id → position in cache
        self._indices = None         # (n, d) uint8
        self._norms = None           # (n,) float32
        self._compressed = None      # CompressedVectors (rebuilt on mutation)
        self._dirty = True

    def load_from_db(self, conn, query="SELECT id, embedding FROM documents LIMIT %s"):
        """Bulk-load embeddings from PostgreSQL into cache."""
        cur = conn.cursor()
        cur.execute(query, (self.max_size,))
        rows = cur.fetchall()

        if not rows:
            return

        ids = [r[0] for r in rows]
        embeddings = np.array([r[1] for r in rows], dtype=np.float32)

        compressed = self.pq.encode(embeddings)
        self._ids = ids
        self._id_to_pos = {id_: i for i, id_ in enumerate(ids)}
        self._indices = compressed.indices
        self._norms = compressed.norms
        self._dirty = True

        mb = compressed.resident_bytes / 1e6
        print(f"Cache loaded: {len(ids)} vectors, {mb:.1f} MB resident")

    def search(self, query: np.ndarray, k: int = 10):
        """Search the in-memory cache. Returns (doc_ids, scores)."""
        if self._indices is None or len(self._ids) == 0:
            return [], []

        if self._dirty:
            self._compressed = CompressedVectors(
                self._indices, self._norms, self.d, self.pq.bits
            )
            self._dirty = False

        idx, scores = self.pq.search(self._compressed, query, k=k)
        doc_ids = [self._ids[i] for i in idx]
        return doc_ids, scores

    def search_low_memory(self, query: np.ndarray, k: int = 10):
        """Search using ADC (no float32 cache). 5x less RAM."""
        if self._indices is None or len(self._ids) == 0:
            return [], []

        if self._dirty:
            self._compressed = CompressedVectors(
                self._indices, self._norms, self.d, self.pq.bits
            )
            self._dirty = False

        idx, scores = self.pq.search_adc(self._compressed, query, k=k)
        doc_ids = [self._ids[i] for i in idx]
        return doc_ids, scores

    def add(self, doc_id: int, embedding: np.ndarray):
        """Add a single vector to the cache."""
        compressed = self.pq.encode(embedding.reshape(1, -1))
        if self._indices is None:
            self._indices = compressed.indices
            self._norms = compressed.norms
        else:
            self._indices = np.vstack([self._indices, compressed.indices])
            self._norms = np.concatenate([self._norms, compressed.norms])
        self._ids.append(doc_id)
        self._id_to_pos[doc_id] = len(self._ids) - 1
        self._dirty = True

        # Evict oldest if over capacity
        if len(self._ids) > self.max_size:
            self._evict(len(self._ids) - self.max_size)

    def _evict(self, n: int):
        """Remove the n oldest entries."""
        for id_ in self._ids[:n]:
            del self._id_to_pos[id_]
        self._ids = self._ids[n:]
        self._indices = self._indices[n:]
        self._norms = self._norms[n:]
        self._id_to_pos = {id_: i for i, id_ in enumerate(self._ids)}
        self._dirty = True

    @property
    def size(self) -> int:
        return len(self._ids)

    @property
    def memory_mb(self) -> float:
        if self._compressed is None:
            return 0.0
        return self._compressed.resident_bytes / 1e6


# Usage
cache = EmbeddingCache(d=384, bits=8, max_size=100_000)
cache.load_from_db(conn)

# Fast search (sub-ms after first query warms the cache)
doc_ids, scores = cache.search(query_embedding, k=10)

# Or low-memory search (no float32 cache, ~5x less RAM)
doc_ids, scores = cache.search_low_memory(query_embedding, k=10)
```

### Memory budget

| Vectors | bits=8 | bits=4 | bits=8 (ADC) |
|---------|--------|--------|--------------|
| 10k | 19 MB | 11 MB | 4 MB |
| 100k | 192 MB | 108 MB | 39 MB |
| 500k | 960 MB | 540 MB | 193 MB |
| 1M | 1.9 GB | 1.1 GB | 385 MB |

"ADC" column shows `search_adc()` / `search_twostage()` which avoid the float32 dequantized cache. The trade-off is ~60x slower per query.

---

## Choosing a pattern

| Factor | Pattern 1 (Python search) | Pattern 2 (pgVector + rerank) | Pattern 3 (In-app cache) |
|--------|--------------------------|-------------------------------|--------------------------|
| Corpus size | <500k | Any | <1M hot set |
| Search latency | 1-3ms (cached) | 5-20ms (HNSW + rerank) | <1ms (cached) |
| Memory | Proportional to corpus | Minimal (DB handles it) | Proportional to cache |
| Requires pgVector | No (just PostgreSQL) | Yes | Optional |
| Training needed | None | None (HNSW is automatic) | None |
| Corpus updates | Re-encode changed vectors | INSERT + index rebuild | add() to cache |
| Best for | Self-contained services | Large production systems | Read-heavy hot paths |

## Tips

- **Start with 8-bit** (R@10=0.98, 4x compression). Drop to 4-bit only if you need more compression and can tolerate R@10~0.85.
- **Use `search_twostage()` for large corpora**: At 100k+ vectors, ADC two-stage gives the same recall as cached search at 5x less RAM.
- **Save/load for persistence**: `compressed.save("index.npz")` / `CompressedVectors.load("index.npz")` uses bit-packed format. Smaller than storing uint8 in the DB.
- **Deterministic quantizer**: `PolarQuantizer(d=384, bits=4, seed=42)` produces identical results everywhere. No need to ship a trained index — just agree on the parameters.
