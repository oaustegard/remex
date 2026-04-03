"""Tests for polar-embed.

Covers: correctness, theoretical bounds, retrieval quality,
edge cases, and distribution robustness.
"""

import numpy as np
import pytest
from polar_embed import PolarQuantizer, CompressedVectors, lloyd_max_codebook
from polar_embed.packing import pack, unpack, packed_nbytes
from polar_embed.rotation import haar_rotation
from polar_embed.codebook import theoretical_mse, theoretical_lower_bound


# ── Fixtures ──

@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def unit_vectors(rng):
    """10k random unit vectors at d=256."""
    X = rng.standard_normal((10_000, 256)).astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X


@pytest.fixture
def queries(rng):
    """50 random unit query vectors at d=256."""
    Q = rng.standard_normal((50, 256)).astype(np.float32)
    Q /= np.linalg.norm(Q, axis=1, keepdims=True)
    return Q


# ── Rotation tests ──

class TestRotation:
    def test_orthogonality(self):
        R = haar_rotation(256)
        I_approx = R @ R.T
        np.testing.assert_allclose(I_approx, np.eye(256), atol=1e-5)

    def test_deterministic(self):
        R1 = haar_rotation(128, seed=99)
        R2 = haar_rotation(128, seed=99)
        np.testing.assert_array_equal(R1, R2)

    def test_different_seeds(self):
        R1 = haar_rotation(128, seed=1)
        R2 = haar_rotation(128, seed=2)
        assert not np.allclose(R1, R2)

    def test_coordinate_distribution(self, unit_vectors):
        """After rotation, coordinates should be ~N(0, 1/d)."""
        R = haar_rotation(256)
        X_rot = unit_vectors @ R.T
        expected_std = 1.0 / np.sqrt(256)
        actual_std = X_rot.std()
        assert abs(actual_std - expected_std) < 0.002, (
            f"Expected std ~{expected_std:.4f}, got {actual_std:.4f}"
        )
        assert abs(X_rot.mean()) < 0.002


# ── Codebook tests ──

class TestCodebook:
    def test_boundaries_sorted(self):
        boundaries, centroids = lloyd_max_codebook(256, 4)
        assert np.all(np.diff(boundaries) > 0), "Boundaries must be strictly increasing"
        assert np.all(np.diff(centroids) > 0), "Centroids must be strictly increasing"

    def test_centroids_count(self):
        for bits in [1, 2, 3, 4, 8]:
            _, centroids = lloyd_max_codebook(128, bits)
            assert len(centroids) == 2**bits

    def test_boundaries_between_centroids(self):
        boundaries, centroids = lloyd_max_codebook(256, 3)
        for i, b in enumerate(boundaries):
            assert centroids[i] < b < centroids[i + 1]

    def test_symmetric(self):
        """Codebook for symmetric distribution should be antisymmetric."""
        _, centroids = lloyd_max_codebook(256, 4)
        np.testing.assert_allclose(centroids, -centroids[::-1], atol=1e-6)


# ── Core quantizer tests ──

class TestPolarQuantizer:
    def test_encode_decode_shapes(self, unit_vectors):
        pq = PolarQuantizer(256, bits=4)
        comp = pq.encode(unit_vectors)
        assert comp.indices.shape == (10_000, 256)
        assert comp.norms.shape == (10_000,)
        assert comp.indices.dtype == np.uint8

        X_hat = pq.decode(comp)
        assert X_hat.shape == unit_vectors.shape
        assert X_hat.dtype == np.float32

    def test_single_vector(self):
        pq = PolarQuantizer(64, bits=4)
        x = np.random.randn(64).astype(np.float32)
        comp = pq.encode(x)
        assert comp.n == 1
        X_hat = pq.decode(comp)
        assert X_hat.shape == (1, 64)

    def test_norms_preserved(self, rng):
        """Original vector norms should be stored exactly."""
        X = rng.standard_normal((100, 128)).astype(np.float32) * 5.0
        pq = PolarQuantizer(128, bits=4)
        comp = pq.encode(X)
        true_norms = np.linalg.norm(X, axis=1)
        np.testing.assert_allclose(comp.norms, true_norms, rtol=1e-5)

    def test_dimension_mismatch_raises(self):
        pq = PolarQuantizer(128, bits=4)
        with pytest.raises(ValueError, match="Expected d=128"):
            pq.encode(np.zeros((10, 64)))

    def test_bits_validation(self):
        with pytest.raises(ValueError):
            PolarQuantizer(128, bits=0)
        with pytest.raises(ValueError):
            PolarQuantizer(128, bits=9)

    def test_deterministic(self, unit_vectors):
        pq1 = PolarQuantizer(256, bits=4, seed=42)
        pq2 = PolarQuantizer(256, bits=4, seed=42)
        c1 = pq1.encode(unit_vectors[:100])
        c2 = pq2.encode(unit_vectors[:100])
        np.testing.assert_array_equal(c1.indices, c2.indices)


# ── MSE and distortion tests ──

class TestDistortion:
    @pytest.mark.parametrize("bits", [2, 3, 4])
    def test_mse_decreases_with_bits(self, unit_vectors, bits):
        """More bits should always reduce MSE."""
        pq_lo = PolarQuantizer(256, bits=bits)
        pq_hi = PolarQuantizer(256, bits=bits + 1)
        assert pq_lo.mse(unit_vectors) > pq_hi.mse(unit_vectors)

    @pytest.mark.parametrize("bits", [2, 3, 4])
    def test_mse_below_upper_bound(self, unit_vectors, bits):
        """Empirical MSE should be within 2x of theoretical upper bound."""
        pq = PolarQuantizer(256, bits=bits)
        empirical = pq.mse(unit_vectors)
        upper = theoretical_mse(256, bits)
        assert empirical < upper * 2.0, (
            f"{bits}-bit: empirical MSE {empirical:.6f} > 2x upper bound {upper:.6f}"
        )

    @pytest.mark.parametrize("bits", [2, 3, 4])
    def test_mse_above_lower_bound(self, unit_vectors, bits):
        """Empirical MSE should be above information-theoretic lower bound."""
        pq = PolarQuantizer(256, bits=bits)
        empirical = pq.mse(unit_vectors)
        lower = theoretical_lower_bound(bits)
        assert empirical > lower * 0.5, (
            f"{bits}-bit: empirical MSE {empirical:.6f} < lower bound {lower:.6f}"
        )


# ── Retrieval quality tests ──

def recall_at_k(true_ips, est_ips, k=10):
    true_topk = set(np.argsort(-true_ips)[:k])
    est_topk = set(np.argsort(-est_ips)[:k])
    return len(true_topk & est_topk) / k


class TestRetrieval:
    def test_search_returns_correct_shapes(self, unit_vectors, queries):
        pq = PolarQuantizer(256, bits=4)
        comp = pq.encode(unit_vectors)
        idx, scores = pq.search(comp, queries[0], k=10)
        assert idx.shape == (10,)
        assert scores.shape == (10,)
        assert np.all(np.diff(scores) <= 0), "Scores should be descending"

    def test_search_k_larger_than_n(self, rng):
        X = rng.standard_normal((5, 64)).astype(np.float32)
        pq = PolarQuantizer(64, bits=4)
        comp = pq.encode(X)
        idx, scores = pq.search(comp, rng.standard_normal(64).astype(np.float32), k=100)
        assert idx.shape == (5,)

    @pytest.mark.parametrize("bits,min_recall", [(2, 0.3), (3, 0.5), (4, 0.7), (8, 0.95)])
    def test_recall_at_10(self, unit_vectors, queries, bits, min_recall):
        """Recall@10 should meet minimum thresholds for random unit vectors."""
        pq = PolarQuantizer(256, bits=bits)
        comp = pq.encode(unit_vectors)
        true_ips = unit_vectors @ queries.T

        recalls = []
        for qi in range(queries.shape[0]):
            X_hat = pq.decode(comp)
            est_ips = X_hat @ queries[qi]
            recalls.append(recall_at_k(true_ips[:, qi], est_ips))

        avg_recall = np.mean(recalls)
        assert avg_recall >= min_recall, (
            f"{bits}-bit recall@10 = {avg_recall:.3f} < {min_recall}"
        )

    def test_beats_naive_at_3bit(self, unit_vectors, queries):
        """polar-embed should beat naive minmax quantization at 3 bits."""
        d = 256
        bits = 3

        # polar-embed
        pq = PolarQuantizer(d, bits=bits)
        comp = pq.encode(unit_vectors)
        X_hat_pq = pq.decode(comp)

        # Naive per-coordinate minmax
        nl = 2**bits
        vmin, vmax = unit_vectors.min(0), unit_vectors.max(0)
        sc = np.maximum((vmax - vmin) / (nl - 1), 1e-12)
        idx = np.clip(np.round((unit_vectors - vmin) / sc), 0, nl - 1)
        X_hat_naive = vmin + idx * sc

        true_ips = unit_vectors @ queries.T

        recall_pq, recall_naive = [], []
        for qi in range(queries.shape[0]):
            recall_pq.append(recall_at_k(true_ips[:, qi], X_hat_pq @ queries[qi]))
            recall_naive.append(recall_at_k(true_ips[:, qi], X_hat_naive @ queries[qi]))

        assert np.mean(recall_pq) > np.mean(recall_naive), (
            f"polar-embed recall {np.mean(recall_pq):.3f} <= "
            f"naive recall {np.mean(recall_naive):.3f}"
        )


# ── Serialization tests ──

class TestSerialization:
    def test_save_load_roundtrip(self, unit_vectors, tmp_path):
        pq = PolarQuantizer(256, bits=4)
        comp = pq.encode(unit_vectors[:100])
        path = str(tmp_path / "test.npz")
        comp.save(path)

        loaded = CompressedVectors.load(path)
        np.testing.assert_array_equal(comp.indices, loaded.indices)
        np.testing.assert_array_equal(comp.norms, loaded.norms)
        assert comp.d == loaded.d
        assert comp.bits == loaded.bits

    def test_compression_ratio(self, unit_vectors):
        pq = PolarQuantizer(256, bits=4)
        comp = pq.encode(unit_vectors)
        assert comp.compression_ratio > 7.0
        assert comp.compression_ratio < 8.5


# ── Distribution robustness tests ──

class TestDistributions:
    """Test with non-uniform embedding distributions."""

    def test_anisotropic_embeddings(self, rng):
        """Embeddings where some dimensions have much higher variance."""
        d = 256
        scales = np.ones(d, dtype=np.float32)
        scales[:10] = 10.0
        X = rng.standard_normal((5000, d)).astype(np.float32) * scales
        X /= np.linalg.norm(X, axis=1, keepdims=True)

        pq = PolarQuantizer(d, bits=4)
        mse_aniso = pq.mse(X)

        X_iso = rng.standard_normal((5000, d)).astype(np.float32)
        X_iso /= np.linalg.norm(X_iso, axis=1, keepdims=True)
        mse_iso = pq.mse(X_iso)

        assert mse_aniso < mse_iso * 2.0, (
            f"Anisotropic MSE {mse_aniso:.6f} >> isotropic {mse_iso:.6f}"
        )

    def test_clustered_embeddings(self, rng):
        """Embeddings with distinct clusters (like real sentence embeddings)."""
        d = 256
        n_clusters = 20
        centers = rng.standard_normal((n_clusters, d)).astype(np.float32)
        centers /= np.linalg.norm(centers, axis=1, keepdims=True)

        labels = rng.integers(0, n_clusters, size=5000)
        X = centers[labels] + rng.standard_normal((5000, d)).astype(np.float32) * 0.1
        X /= np.linalg.norm(X, axis=1, keepdims=True)

        pq = PolarQuantizer(d, bits=4)
        comp = pq.encode(X)
        X_hat = pq.decode(comp)

        Q = centers[:5]
        true_ips = X @ Q.T
        est_ips = X_hat @ Q.T

        recalls = [recall_at_k(true_ips[:, qi], est_ips[:, qi], k=50) for qi in range(5)]
        assert np.mean(recalls) > 0.6, (
            f"Clustered recall@50 = {np.mean(recalls):.3f}, expected > 0.6"
        )
class TestCalibrate:
    """Tests for PolarQuantizer.calibrate() per-dimension codebooks."""

    def test_calibrate_sets_flag(self):
        pq = PolarQuantizer(d=32, bits=3)
        assert not pq.calibrated
        X = np.random.default_rng(0).standard_normal((100, 32)).astype(np.float32)
        pq.calibrate(X)
        assert pq.calibrated

    def test_calibrate_changes_codebook_shape(self):
        pq = PolarQuantizer(d=32, bits=3)
        assert pq.centroids.ndim == 1  # uniform: (8,)
        X = np.random.default_rng(0).standard_normal((100, 32)).astype(np.float32)
        pq.calibrate(X)
        assert pq.centroids.shape == (32, 8)  # per-dim: (d, 2^bits)
        assert pq.boundaries.shape == (32, 7)

    def test_calibrate_chaining(self):
        X = np.random.default_rng(0).standard_normal((100, 32)).astype(np.float32)
        pq = PolarQuantizer(d=32, bits=3).calibrate(X)
        assert pq.calibrated

    def test_calibrated_encode_decode_roundtrip(self):
        d, n = 64, 200
        rng = np.random.default_rng(42)
        X = rng.standard_normal((n, d)).astype(np.float32)

        pq = PolarQuantizer(d=d, bits=4).calibrate(X)
        compressed = pq.encode(X)
        X_hat = pq.decode(compressed)

        assert X_hat.shape == X.shape
        mse = np.mean(np.sum((X - X_hat) ** 2, axis=1))
        # Calibrated MSE should be reasonable
        assert mse < 1.0, f"Calibrated MSE too high: {mse}"

    def test_calibrated_search(self):
        d, n = 64, 500
        rng = np.random.default_rng(42)
        X = rng.standard_normal((n, d)).astype(np.float32)
        query = rng.standard_normal(d).astype(np.float32)

        pq = PolarQuantizer(d=d, bits=4).calibrate(X)
        compressed = pq.encode(X)
        indices, scores = pq.search(compressed, query, k=10)

        assert indices.shape == (10,)
        assert scores.shape == (10,)
        # Scores should be monotonically non-increasing
        assert np.all(np.diff(scores) <= 1e-6)

    def test_calibrated_mse_reasonable(self):
        """Calibrated MSE should be finite and in a reasonable range."""
        d = 64
        rng = np.random.default_rng(42)
        X = rng.standard_normal((500, d)).astype(np.float32)

        pq_calibrated = PolarQuantizer(d=d, bits=4).calibrate(X)
        mse = pq_calibrated.mse(X)

        assert np.isfinite(mse)
        assert mse < 5.0, f"Calibrated MSE unexpectedly high: {mse}"

    def test_calibrated_vs_oblivious_on_random(self):
        """On random data, calibrated and oblivious should perform similarly."""
        d = 64
        rng = np.random.default_rng(42)
        X = rng.standard_normal((1000, d)).astype(np.float32)
        # Normalize to unit sphere (typical embedding input)
        X = X / np.linalg.norm(X, axis=1, keepdims=True)

        pq_oblivious = PolarQuantizer(d=d, bits=4)
        pq_calibrated = PolarQuantizer(d=d, bits=4).calibrate(X)

        mse_oblivious = pq_oblivious.mse(X)
        mse_calibrated = pq_calibrated.mse(X)

        # On random unit vectors (ideal case for oblivious), calibrated should
        # be close but not necessarily better — within 50% is fine
        ratio = mse_calibrated / max(mse_oblivious, 1e-10)
        assert ratio < 1.5, (
            f"Calibrated MSE ({mse_calibrated:.4f}) much worse than "
            f"oblivious ({mse_oblivious:.4f}) on random data"
        )

    def test_calibrate_dimension_mismatch(self):
        pq = PolarQuantizer(d=32, bits=3)
        X_wrong = np.random.default_rng(0).standard_normal((100, 64)).astype(np.float32)
        try:
            pq.calibrate(X_wrong)
            assert False, "Should have raised ValueError"
        except ValueError:
            pass

    def test_uncalibrated_still_works(self):
        """Verify the default path is unaffected."""
        d, n = 64, 200
        rng = np.random.default_rng(42)
        X = rng.standard_normal((n, d)).astype(np.float32)

        pq = PolarQuantizer(d=d, bits=4)
        assert not pq.calibrated
        compressed = pq.encode(X)
        X_hat = pq.decode(compressed)
        indices, scores = pq.search(compressed, X[0], k=5)

        assert X_hat.shape == X.shape
        assert indices.shape == (5,)


# ---- Bit-packing tests ----

class TestPacking:
    """Tests for bit-packed index storage."""

    def test_pack_unpack_roundtrip_all_bits(self):
        from polar_embed.packing import pack, unpack
        rng = np.random.default_rng(99)
        for bits in range(1, 9):
            indices = rng.integers(0, 2**bits, size=(50, 384), dtype=np.uint8)
            packed = pack(indices, bits)
            unpacked = unpack(packed, bits, indices.size).reshape(indices.shape)
            assert np.array_equal(indices, unpacked), f"Roundtrip failed at {bits}-bit"

    def test_packed_size_correct(self):
        from polar_embed.packing import pack, packed_nbytes
        rng = np.random.default_rng(99)
        for bits in [2, 3, 4]:
            indices = rng.integers(0, 2**bits, size=(100, 384), dtype=np.uint8)
            packed = pack(indices, bits)
            expected = packed_nbytes(100, 384, bits)
            assert packed.nbytes == expected

    def test_compressed_vectors_packing(self):
        """CompressedVectors should pack transparently."""
        pq = PolarQuantizer(d=64, bits=4)
        X = np.random.default_rng(0).standard_normal((100, 64)).astype(np.float32)
        comp = pq.encode(X)

        # Packed storage should be ~half of uint8
        assert comp.nbytes < comp.nbytes_unpacked

        # But indices property still returns full uint8
        assert comp.indices.shape == (100, 64)
        assert comp.indices.dtype == np.uint8

    def test_save_load_packed_roundtrip(self):
        import tempfile, os
        pq = PolarQuantizer(d=64, bits=3)
        X = np.random.default_rng(0).standard_normal((50, 64)).astype(np.float32)
        comp = pq.encode(X)

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            path = f.name
        try:
            comp.save(path)
            comp2 = CompressedVectors.load(path)
            assert np.array_equal(comp.indices, comp2.indices)
            assert np.allclose(comp.norms, comp2.norms)
            assert comp2.bits == 3
            assert comp2.d == 64
        finally:
            os.unlink(path)

    def test_compression_ratios_honest(self):
        """Verify compression ratios reflect actual packed storage."""
        pq = PolarQuantizer(d=384, bits=4)
        X = np.random.default_rng(0).standard_normal((1000, 384)).astype(np.float32)
        comp = pq.encode(X)

        # 4-bit: 384 dims × 4 bits / 8 = 192 bytes indices + 4 bytes norm = 196 per vec
        # float32: 384 × 4 = 1536 per vec
        # ratio ≈ 1536/196 ≈ 7.8×
        assert 7.0 < comp.compression_ratio < 8.5

    def test_2bit_compression(self):
        pq = PolarQuantizer(d=384, bits=2)
        X = np.random.default_rng(0).standard_normal((100, 384)).astype(np.float32)
        comp = pq.encode(X)
        # 2-bit: 96 bytes indices + 4 bytes norm = 100 per vec
        # ratio ≈ 1536/100 ≈ 15.4×
        assert comp.compression_ratio > 14.0
