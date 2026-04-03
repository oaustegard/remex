# Scale Architecture: Partitioned Two-Tier Retrieval

How polar-embed serves 100M+ vector corpora with sub-200ms query latency, no GPU, and no training.

## The problem

Large embedding corpora (Semantic Scholar's 200M abstracts, patent databases, legal archives) don't fit in RAM as float32 and can't be scanned linearly at interactive speeds. Traditional solutions (FAISS IVF, HNSW) require training on the data and maintaining specialized index structures.

polar-embed's value is **zero-training, deterministic, portable compression**. This document describes how to use that property at scale.

## Core insight: partition by metadata, not by geometry

Most large corpora come with categorical metadata — fields of study, document types, jurisdictions, date ranges. Instead of geometric partitioning (IVF centroids), partition by metadata that users already filter on.

This has three advantages over geometric partitions:
1. **No training.** Metadata partitions are defined by the domain, not learned from data.
2. **Natural query routing.** The user's field selection *is* the partition key.
3. **Interpretable sizing.** A "transformer architectures" partition has a predictable size based on the literature, not on centroid assignments.

## Memory budget

At d=768 (SPECTER2 embeddings), 2-bit packed storage:

| Partition size | Packed indices | Norms | Total | Scan time |
|---------------|---------------|-------|-------|-----------|
| 50K vectors | 9.6 MB | 0.2 MB | ~10 MB | 25 ms |
| 200K vectors | 38.4 MB | 0.8 MB | ~39 MB | 100 ms |
| 500K vectors | 96 MB | 2 MB | ~98 MB | 250 ms |
| 2M vectors | 384 MB | 8 MB | ~392 MB | 1,000 ms |

**Target: <200K vectors per partition for interactive queries (<100ms scan).**

At 200M total abstracts with ~200 fine-grained fields, average partition is ~1M vectors. Sub-fields or filtered views bring this to the 50K-200K interactive range.

## Architecture

```
                        Offline (batch)

  Corpus + metadata
       |
       v
  Partition by field-of-study
       |
       +---> Encode (PolarQuantizer, 8-bit)
       |       |
       |       +---> Store 8-bit in pgVector
       |       |     (fine rerank source)
       |       |
       |       +---> Derive 2-bit packed
       |             Save as .polar file
       |             (one per partition)
       |
       +---> Metadata index
             (partition -> file path mapping)
```

```
                   Runtime (per query)

  1. User selects domain / field filter
       |
       v
  2. mmap the partition's .polar file
     (instant if already in OS page cache)
       |
       v
  3. ADC coarse scan over 2-bit packed
     -> 200 candidate IDs              (~100 ms)
       |
       v
  4. Fetch 200 rows of 8-bit from pgVector
     (single round-trip)                (~15 ms)
       |
       v
  5. Dequantize 200 rows, rerank        (~0.1 ms)
       |
       v
  6. Return top-k results              (~120 ms)
```

## Why mmap instead of application-level caching

The partition files are flat binary arrays. Memory-mapping them gives:

- **Lazy loading.** Pages are read from disk on first access, not on file open. A 39 MB partition only occupies RAM for the pages the ADC scorer touches.
- **OS-managed eviction.** The kernel's page cache is an LRU with decades of optimization. It evicts cold partitions under memory pressure and keeps hot ones resident.
- **Zero-copy hot swap.** Switching from "NLP" to "protein folding" is opening a new mmap. If both fit in page cache, both stay warm. No explicit cache invalidation.
- **No serialization overhead.** The packed indices are used directly from the mmap'd buffer — no deserialization, no copy to a separate data structure.

An application-level cache would reimplement all of this, worse.

## File format: `.polar`

A flat, mmap-friendly binary format. No compression, no container overhead.

```
Offset  Size            Content
0       8               Magic: b"POLAR\x00\x01\x00"  (version 1)
8       8               n (uint64): number of vectors
16      4               d (uint32): dimension
20      1               bits (uint8): quantization bit-width
21      3               reserved (padding to 24-byte header)
24      n * 4           norms (float32 array)
24+n*4  ceil(n*d*bits/8)   packed indices (uint8 byte array)
```

Norms come first because they're accessed linearly during scoring (multiplied with chunk scores). Packed indices follow and are accessed via `unpack_rows(start, end)` in the ADC loop.

The entire file can be mmap'd. `unpack_rows` computes byte offsets into the packed region and unpacks a chunk to a temporary uint8 buffer for table lookup.

## The two-tier contract

| Tier | Storage | Bit-width | Purpose | Access pattern |
|------|---------|-----------|---------|----------------|
| Coarse | .polar file (mmap) | 2-bit packed | Full-partition scan | Sequential, chunked |
| Fine | pgVector (or any DB) | 8-bit | Candidate rerank | Random access, ~200 rows |

The quantizer is shared: same `PolarQuantizer(d=768, bits=8, seed=42)` produces the 8-bit encoding. The 2-bit coarse codes are derived by right-shifting (Matryoshka property). This means:

- **One encode pass** in the batch pipeline
- **Deterministic derivation** of coarse from fine (no separate codebook)
- **Seed is the only shared state** between the offline and runtime systems

## Capacity planning

A deployment serving all of Semantic Scholar (200M abstracts, ~200 partitions):

| Resource | Estimate |
|----------|----------|
| .polar files on disk | ~200 x 200 MB avg = **40 GB** |
| RAM for hot partitions (10 warm) | ~2 GB page cache |
| pgVector 8-bit storage | 200M x 768 bytes = **153 GB** |
| pgVector norms | 200M x 4 bytes = **800 MB** |
| Total DB storage | ~154 GB |

This fits on a single beefy VM (256 GB RAM, 500 GB SSD). No GPU. No distributed index. No training pipeline.

## What polar-embed provides vs. what the developer builds

**polar-embed provides:**
- `PolarQuantizer`: encode, derive coarse codes, score
- `PackedVectors`: packed-in-memory storage, `unpack_rows()`, mmap-backed load
- `.polar` file format: save/load with mmap support
- ADC scoring that works against packed/mmap'd data

**The developer builds:**
- Partition strategy (metadata -> partition mapping)
- Batch pipeline (encode corpus, generate .polar files, populate pgVector)
- Partition router (user query -> which .polar file to mmap)
- DB client (fetch 8-bit rows for fine rerank)
- Application server, API, etc.

polar-embed is a compression and scoring library, not a database or a search engine.

## Alternatives considered

**Why not FAISS IVF?** Requires training on data, geometric partitions don't align with user intent, and the index isn't portable (tied to the training run). Stronger recall at matched compression, but operationally heavier.

**Why not HNSW (pgVector native)?** Great for moderate corpora. At 200M vectors the index is enormous and the graph traversal has unpredictable latency. Also doesn't compress — you're storing full float32.

**Why not just pgVector with 8-bit stored as BYTEA?** You'd still need to scan the full partition for coarse retrieval. pgVector's sequential scan over BYTEA is slower than mmap'd ADC over packed 2-bit because it goes through the Postgres executor, row-by-row.

**Why not quantize norms too?** At 2-bit indices, norms are ~10% of partition file size (0.8 MB vs 38.4 MB at 200K vectors). The savings from quantizing norms are marginal, and any ranking error from norm quantization compounds with the already-lossy 2-bit indices. Keep norms at float32.
