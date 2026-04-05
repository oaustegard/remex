# Scale Architecture: Partitioned Two-Tier Retrieval

How remex serves 100M+ vector corpora with sub-200ms query latency, no GPU, and no training.

## The problem

Large embedding corpora (Semantic Scholar's 200M abstracts, patent databases, legal archives) don't fit in RAM as float32 and can't be scanned linearly at interactive speeds. Traditional solutions (FAISS IVF, HNSW) require training on the data and maintaining specialized index structures.

remex's value is **zero-training, deterministic, portable compression**. This document describes how to use that property at scale.

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
       +---> Encode (Quantizer, 8-bit)
       |       |
       |       +---> Store 8-bit in pgVector
       |       |     (fine rerank source)
       |       |
       |       +---> Derive 2-bit packed
       |             Save as .arrow file
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
  2. mmap the partition's .arrow file
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

## File format: `.arrow` (Arrow IPC)

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
| Coarse | .arrow file (mmap) | 2-bit packed | Full-partition scan | Sequential, chunked |
| Fine | pgVector (or any DB) | 8-bit | Candidate rerank | Random access, ~200 rows |

The quantizer is shared: same `Quantizer(d=768, bits=8, seed=42)` produces the 8-bit encoding. The 2-bit coarse codes are derived by right-shifting (Matryoshka property). This means:

- **One encode pass** in the batch pipeline
- **Deterministic derivation** of coarse from fine (no separate codebook)
- **Seed is the only shared state** between the offline and runtime systems

## Capacity planning

A deployment serving all of Semantic Scholar (200M abstracts, ~200 partitions):

| Resource | Estimate |
|----------|----------|
| .arrow files on disk | ~200 x 200 MB avg = **40 GB** |
| RAM for hot partitions (10 warm) | ~2 GB page cache |
| pgVector 8-bit storage | 200M x 768 bytes = **153 GB** |
| pgVector norms | 200M x 4 bytes = **800 MB** |
| Total DB storage | ~154 GB |

This fits on a single beefy VM (256 GB RAM, 500 GB SSD). No GPU. No distributed index. No training pipeline.

## What remex provides vs. what the developer builds

**remex provides:**
- `Quantizer`: encode, derive coarse codes, score
- `PackedVectors`: packed-in-memory storage, `unpack_rows()`, mmap-backed load
- `.arrow` (Arrow IPC) file format: save/load with mmap support
- ADC scoring that works against packed/mmap'd data

**The developer builds:**
- Partition strategy (metadata -> partition mapping)
- Batch pipeline (encode corpus, generate .arrow files, populate pgVector)
- Partition router (user query -> which .arrow file to mmap)
- DB client (fetch 8-bit rows for fine rerank)
- Application server, API, etc.

remex is a compression and scoring library, not a database or a search engine.

## Alternatives considered

**Why not FAISS IVF?** Requires training on data, geometric partitions don't align with user intent, and the index isn't portable (tied to the training run). Stronger recall at matched compression, but operationally heavier.

**Why not HNSW (pgVector native)?** Great for moderate corpora. At 200M vectors the index is enormous and the graph traversal has unpredictable latency. Also doesn't compress — you're storing full float32.

**Why not just pgVector with 8-bit stored as BYTEA?** You'd still need to scan the full partition for coarse retrieval. pgVector's sequential scan over BYTEA is slower than mmap'd ADC over packed 2-bit because it goes through the Postgres executor, row-by-row.

**Why not quantize norms too?** At 2-bit indices, norms are ~10% of partition file size (0.8 MB vs 38.4 MB at 200K vectors). The savings from quantizing norms are marginal, and any ranking error from norm quantization compounds with the already-lossy 2-bit indices. Keep norms at float32.


## S3 as the index store

The .arrow files are read-only after generation, flat binary, and partition-sized. This is exactly what object storage is built for. S3 (or R2, GCS, Azure Blob) becomes the canonical store; local SSD is a cache tier.

### Three-tier storage hierarchy

```
                S3 / R2 (canonical store)
                    ~40 GB for 200M vectors
                    ~$0.92/month (S3 Standard)
                         |
                         | fetch on demand
                         v
                Local SSD (cache tier)
                    ~4 GB for 20 hot partitions
                    LRU eviction by atime
                         |
                         | mmap
                         v
                OS page cache (hot tier)
                    ~2 GB for actively-queried partitions
                    Managed by kernel
```

### Economics

| Resource | Cost |
|----------|------|
| S3 Standard (40 GB) | $0.92/month |
| S3 Infrequent Access (40 GB) | $0.50/month |
| Cloudflare R2 (40 GB, no egress) | $0.60/month |
| S3 GET requests (1000/day) | $0.01/month |
| S3 transfer (same region) | $0.00 |

The entire Semantic Scholar coarse index costs less than a dollar a month to host.

### Cold query latency

When a user queries a partition that isn't on local SSD:

| Partition | S3 fetch (multipart 8x) | ADC scan | Total cold | Warm (mmap) |
|-----------|------------------------|----------|------------|-------------|
| 50K vectors (10 MB) | 20 ms | 25 ms | **45 ms** | 25 ms |
| 200K vectors (39 MB) | 78 ms | 100 ms | **178 ms** | 100 ms |
| 500K vectors (98 MB) | 196 ms | 250 ms | **446 ms** | 250 ms |

At the target partition size (200K), even a cold query is under 200ms. Multipart download (8 concurrent S3 range GETs) is key — a single GET would be 390ms.

### Local SSD cache management

The cache is just a directory of .arrow files with LRU eviction:

```python
# Pseudocode — not part of remex
class PartitionCache:
    def __init__(self, cache_dir: str, max_bytes: int, s3_bucket: str):
        self.cache_dir = cache_dir
        self.max_bytes = max_bytes
        self.s3 = boto3.client("s3")
        self.bucket = s3_bucket

    def get(self, partition_key: str) -> Path:
        local_path = self.cache_dir / f"{partition_key}.polar"
        if local_path.exists():
            local_path.touch()  # update atime for LRU
            return local_path
        self._evict_if_needed()
        self._download(partition_key, local_path)
        return local_path

    def _download(self, key: str, path: Path):
        # Multipart download for files > 8 MB
        self.s3.download_file(self.bucket, f"partitions/{key}.polar", str(path))

    def _evict_if_needed(self):
        # Remove oldest-accessed files until under budget
        files = sorted(self.cache_dir.glob("*.polar"), key=lambda p: p.stat().st_atime)
        total = sum(f.stat().st_size for f in files)
        while total > self.max_bytes and files:
            victim = files.pop(0)
            total -= victim.stat().st_size
            victim.unlink()
```

This is ~30 lines of application code. remex doesn't need to know about S3 — it just reads .arrow files from wherever they are.

### Prefetching via taxonomy

Semantic Scholar's field-of-study taxonomy is a tree. When a user selects a broad field, the sub-fields they'll likely drill into are predictable:

```
Computer Science                    ← user selects this
├── Artificial Intelligence         ← prefetch these
│   ├── Machine Learning           ← and these
│   ├── NLP
│   └── Computer Vision
├── Systems
├── Theory
└── ...
```

On the first query, kick off background downloads of sibling and child partitions. By the time the user narrows their search, the partition is already on local SSD. The .arrow file sizes (10-40 MB each) make this practical — prefetching 5 siblings is ~200 MB, downloadable in under a second.

### Deployment topology

```
┌──────────────────────────────────────────────────┐
│              Application Server                   │
│                                                   │
│  ┌─────────────┐  ┌──────────────────────┐       │
│  │ Partition    │  │ remex          │       │
│  │ Cache (SSD)  │──│ PackedVectors.mmap() │       │
│  │ 4-20 GB      │  │ ADC coarse scan      │       │
│  └──────┬───────┘  └──────────┬───────────┘       │
│         │                     │                    │
│         │ miss                │ 200 candidate IDs  │
│         v                     v                    │
│  ┌──────────────┐  ┌──────────────────────┐       │
│  │ S3 / R2      │  │ pgVector             │       │
│  │ (all .polar) │  │ (8-bit for rerank)   │       │
│  └──────────────┘  └──────────────────────┘       │
└──────────────────────────────────────────────────┘
```

The app server needs:
- **4-20 GB local SSD** for the partition cache (size based on working set)
- **Network access** to S3 (same region, ~100 MB/s+) and pgVector
- **No GPU.** The entire coarse scan is NumPy on CPU.

### Regeneration via content-hashed keys

When the corpus updates (Semantic Scholar releases weekly), regenerate affected partitions:

1. Encode new/updated documents with the same `Quantizer(d=768, bits=8, seed=42)`
2. Rebuild affected partition .arrow files (2-bit derived from 8-bit)
3. Upload to S3 with content-hashed key: `v1/{sha256[:16]}.arrow`
4. Update the manifest (`s3://bucket/manifest.json`) to point partition name → new key
5. App fetches updated manifest on next startup or periodic check
6. Next query for that partition downloads the new file; old file stays on SSD until LRU evicts

No explicit cache purge needed. Old and new versions coexist on S3 during rollover. The manifest is the single source of truth.

```json
{
  "format_version": 1,
  "quantizer": {"d": 768, "bits": 8, "seed": 42},
  "generated": "2026-04-07T00:00:00Z",
  "partitions": {
    "cs.AI": {"key": "v1/a3f8c1d2e9b74f01.arrow", "n": 185000, "bytes": 35520000},
    "cs.CL": {"key": "v1/7b2e4f91cc083a22.arrow", "n": 210000, "bytes": 40320000}
  }
}
```

The quantizer seed doesn't change, so existing 8-bit encodings in pgVector are still compatible. Only new documents need encoding.
