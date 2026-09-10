"""
Model Benchmark for LEDGER Retrieval Service

Benchmarks:
1. Embedding models: Small / Base / Large
2. CPU vs GPU inference
3. Embedding latency
4. Embedding dimension
5. GPU VRAM usage
6. Reranker models: Small / Base / Large (when available)
7. CPU vs GPU reranker latency
8. GPU VRAM usage

Important:
- This script does NOT modify production Embedder/Reranker classes.
- Models are benchmarked independently.
- Models are warmed up before timing.
- CUDA synchronization is u
sed for accurate GPU timings.
"""

import gc
import os
import time
from statistics import mean, median, stdev

import torch
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_huggingface import HuggingFaceEmbeddings

# ============================================================
# Configuration
# ============================================================

WARMUP_RUNS = 2
BENCHMARK_RUNS = 10

TEST_TEXT = (
    "The company reported revenue growth during the fiscal year. "
    "Operating income increased compared with the previous year, "
    "while finished goods inventory remained stable."
)

TEST_QUERY = "What was the company's revenue and operating income?"

TEST_DOCUMENTS = [
    (
        "The company reported revenue growth during the fiscal year "
        "and operating income increased compared with the previous year."
    ),
    ("Finished goods inventory decreased slightly during the year because of improved inventory management."),
    ("The company generated strong cash flow from operating activities despite changes in working capital."),
    ("Total assets increased as the company invested in property and equipment during the reporting period."),
    ("The annual report discusses revenue, operating expenses, net income, and cash flow for the fiscal year."),
]


# ============================================================
# Models
# ============================================================

EMBEDDING_MODELS = {
    "BGE-Small": "BAAI/bge-small-en-v1.5",
    "BGE-Base": "BAAI/bge-base-en-v1.5",
    "BGE-Large": "BAAI/bge-large-en-v1.5",
}


# NOTE:
# BGE reranker-large is your current production reranker.
#
# The smaller reranker models are intentionally kept separate
# because model availability/naming can change.
#
# Start with the production Large model first.
RERANKER_MODELS = {
    "BGE-Reranker-Large": "BAAI/bge-reranker-large",
}


# ============================================================
# Utility functions
# ============================================================


def cleanup():
    """
    Release Python and GPU memory between model tests.
    """
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def get_gpu_memory_mb():
    """
    Return currently allocated and reserved CUDA memory in MB.
    """
    if not torch.cuda.is_available():
        return 0.0, 0.0

    allocated = torch.cuda.memory_allocated() / (1024**2)
    reserved = torch.cuda.memory_reserved() / (1024**2)

    return allocated, reserved


def synchronize_if_cuda(device):
    """
    Synchronize GPU before/after measurements.
    """
    if device == "cuda":
        torch.cuda.synchronize()


def calculate_stats(times):
    """
    Calculate latency statistics.
    """
    result = {
        "mean_ms": mean(times),
        "median_ms": median(times),
        "min_ms": min(times),
        "max_ms": max(times),
    }

    if len(times) > 1:
        result["std_ms"] = stdev(times)
    else:
        result["std_ms"] = 0.0

    return result


# ============================================================
# Embedding Benchmark
# ============================================================


def benchmark_embedding_model(model_name, model_id, device):
    """
    Benchmark one embedding model on one device.
    """

    print("\n" + "=" * 80)
    print(f"Embedding Model : {model_name}")
    print(f"Model ID        : {model_id}")
    print(f"Device          : {device}")
    print("=" * 80)

    cleanup()

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    print("\nLoading model...")

    load_start = time.perf_counter()

    embeddings = HuggingFaceEmbeddings(
        model_name=model_id,
        model_kwargs={
            "device": device,
        },
        encode_kwargs={
            "normalize_embeddings": True,
        },
    )

    synchronize_if_cuda(device)

    load_time = (time.perf_counter() - load_start) * 1000

    print(f"Model load time : {load_time:.2f} ms")

    # --------------------------------------------------------
    # Memory after loading
    # --------------------------------------------------------

    allocated_after_load, reserved_after_load = get_gpu_memory_mb()

    if device == "cuda":
        print(f"VRAM allocated after load : {allocated_after_load:.2f} MB")
        print(f"VRAM reserved after load  : {reserved_after_load:.2f} MB")

    # --------------------------------------------------------
    # Warm-up
    # --------------------------------------------------------

    print(f"\nWarm-up runs: {WARMUP_RUNS}")

    for _ in range(WARMUP_RUNS):
        _ = embeddings.embed_query(TEST_TEXT)

    synchronize_if_cuda(device)

    # --------------------------------------------------------
    # Get embedding dimension
    # --------------------------------------------------------

    embedding = embeddings.embed_query(TEST_TEXT)

    synchronize_if_cuda(device)

    dimension = len(embedding)

    print(f"Embedding dimension : {dimension}")

    # --------------------------------------------------------
    # Benchmark
    # --------------------------------------------------------

    print(f"\nBenchmark runs: {BENCHMARK_RUNS}")

    times = []

    for i in range(BENCHMARK_RUNS):
        synchronize_if_cuda(device)

        start = time.perf_counter()

        _ = embeddings.embed_query(TEST_TEXT)

        synchronize_if_cuda(device)

        elapsed = (time.perf_counter() - start) * 1000

        times.append(elapsed)

        print(f"Run {i + 1:02d}: {elapsed:.2f} ms")

    stats = calculate_stats(times)

    # --------------------------------------------------------
    # Memory after inference
    # --------------------------------------------------------

    allocated_after_inference, reserved_after_inference = get_gpu_memory_mb()

    if device == "cuda":
        print(f"\nVRAM allocated after inference : {allocated_after_inference:.2f} MB")
        print(f"VRAM reserved after inference  : {reserved_after_inference:.2f} MB")

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    print("\n--- Results ---")
    print(f"Mean latency   : {stats['mean_ms']:.2f} ms")
    print(f"Median latency : {stats['median_ms']:.2f} ms")
    print(f"Min latency    : {stats['min_ms']:.2f} ms")
    print(f"Max latency    : {stats['max_ms']:.2f} ms")
    print(f"Std latency    : {stats['std_ms']:.2f} ms")

    result = {
        "type": "embedding",
        "model": model_name,
        "model_id": model_id,
        "device": device,
        "dimension": dimension,
        "load_ms": load_time,
        "mean_ms": stats["mean_ms"],
        "median_ms": stats["median_ms"],
        "min_ms": stats["min_ms"],
        "max_ms": stats["max_ms"],
        "std_ms": stats["std_ms"],
        "vram_allocated_mb": allocated_after_inference,
        "vram_reserved_mb": reserved_after_inference,
    }

    # Release model
    del embeddings
    cleanup()

    return result


# ============================================================
# Reranker Benchmark
# ============================================================


def benchmark_reranker_model(model_name, model_id, device):
    """
    Benchmark one cross-encoder reranker on one device.
    """

    print("\n" + "=" * 80)
    print(f"Reranker Model : {model_name}")
    print(f"Model ID       : {model_id}")
    print(f"Device         : {device}")
    print("=" * 80)

    cleanup()

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    print("\nLoading reranker...")

    load_start = time.perf_counter()

    reranker = HuggingFaceCrossEncoder(
        model_name=model_id,
        model_kwargs={
            "device": device,
        },
    )

    synchronize_if_cuda(device)

    load_time = (time.perf_counter() - load_start) * 1000

    print(f"Model load time : {load_time:.2f} ms")

    # --------------------------------------------------------
    # Prepare query-document pairs
    # --------------------------------------------------------

    pairs = [(TEST_QUERY, document) for document in TEST_DOCUMENTS]

    print(f"Documents per request: {len(pairs)}")

    # --------------------------------------------------------
    # Memory after loading
    # --------------------------------------------------------

    allocated_after_load, reserved_after_load = get_gpu_memory_mb()

    if device == "cuda":
        print(f"VRAM allocated after load : {allocated_after_load:.2f} MB")
        print(f"VRAM reserved after load  : {reserved_after_load:.2f} MB")

    # --------------------------------------------------------
    # Warm-up
    # --------------------------------------------------------

    print(f"\nWarm-up runs: {WARMUP_RUNS}")

    for _ in range(WARMUP_RUNS):
        _ = reranker.score(pairs)

    synchronize_if_cuda(device)

    # --------------------------------------------------------
    # Verify scores
    # --------------------------------------------------------

    scores = reranker.score(pairs)

    synchronize_if_cuda(device)

    print("\nSample scores:")
    print([round(float(score), 6) for score in scores])

    # --------------------------------------------------------
    # Benchmark
    # --------------------------------------------------------

    print(f"\nBenchmark runs: {BENCHMARK_RUNS}")

    times = []

    for i in range(BENCHMARK_RUNS):
        synchronize_if_cuda(device)

        start = time.perf_counter()

        _ = reranker.score(pairs)

        synchronize_if_cuda(device)

        elapsed = (time.perf_counter() - start) * 1000

        times.append(elapsed)

        print(f"Run {i + 1:02d}: {elapsed:.2f} ms")

    stats = calculate_stats(times)

    # --------------------------------------------------------
    # Memory
    # --------------------------------------------------------

    allocated_after_inference, reserved_after_inference = get_gpu_memory_mb()

    if device == "cuda":
        print(f"\nVRAM allocated after inference : {allocated_after_inference:.2f} MB")
        print(f"VRAM reserved after inference  : {reserved_after_inference:.2f} MB")

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    print("\n--- Results ---")
    print(f"Mean latency   : {stats['mean_ms']:.2f} ms")
    print(f"Median latency : {stats['median_ms']:.2f} ms")
    print(f"Min latency    : {stats['min_ms']:.2f} ms")
    print(f"Max latency    : {stats['max_ms']:.2f} ms")
    print(f"Std latency    : {stats['std_ms']:.2f} ms")

    result = {
        "type": "reranker",
        "model": model_name,
        "model_id": model_id,
        "device": device,
        "documents_per_request": len(pairs),
        "load_ms": load_time,
        "mean_ms": stats["mean_ms"],
        "median_ms": stats["median_ms"],
        "min_ms": stats["min_ms"],
        "max_ms": stats["max_ms"],
        "std_ms": stats["std_ms"],
        "vram_allocated_mb": allocated_after_inference,
        "vram_reserved_mb": reserved_after_inference,
    }

    del reranker
    cleanup()

    return result


# ============================================================
# Main
# ============================================================


def main():

    print("\n")
    print("=" * 80)
    print("LEDGER RETRIEVAL MODEL BENCHMARK")
    print("=" * 80)

    print("\nEnvironment:")
    print(f"PyTorch       : {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"CUDA version  : {torch.version.cuda}")
        print(f"GPU           : {torch.cuda.get_device_name(0)}")

    print("=" * 80)

    results = []

    # ========================================================
    # Embedding Benchmark
    # ========================================================

    print("\n\n")
    print("#" * 80)
    print("# EMBEDDING MODEL BENCHMARK")
    print("#" * 80)

    # --------------------------------------------------------
    # GPU embedding benchmark
    # --------------------------------------------------------

    if torch.cuda.is_available():
        for model_name, model_id in EMBEDDING_MODELS.items():
            result = benchmark_embedding_model(
                model_name=model_name,
                model_id=model_id,
                device="cuda",
            )

            results.append(result)

    # --------------------------------------------------------
    # CPU embedding benchmark
    # --------------------------------------------------------

    print("\n\n")
    print("#" * 80)
    print("# CPU EMBEDDING BENCHMARK")
    print("#" * 80)

    for model_name, model_id in EMBEDDING_MODELS.items():
        result = benchmark_embedding_model(
            model_name=model_name,
            model_id=model_id,
            device="cpu",
        )

        results.append(result)

    # ========================================================
    # Reranker Benchmark
    # ========================================================

    print("\n\n")
    print("#" * 80)
    print("# RERANKER MODEL BENCHMARK")
    print("#" * 80)

    # --------------------------------------------------------
    # GPU reranker
    # --------------------------------------------------------

    if torch.cuda.is_available():
        for model_name, model_id in RERANKER_MODELS.items():
            result = benchmark_reranker_model(
                model_name=model_name,
                model_id=model_id,
                device="cuda",
            )

            results.append(result)

    # --------------------------------------------------------
    # CPU reranker
    # --------------------------------------------------------

    print("\n\n")
    print("#" * 80)
    print("# CPU RERANKER BENCHMARK")
    print("#" * 80)

    for model_name, model_id in RERANKER_MODELS.items():
        result = benchmark_reranker_model(
            model_name=model_name,
            model_id=model_id,
            device="cpu",
        )

        results.append(result)

    # ========================================================
    # Final Summary
    # ========================================================

    print("\n\n")
    print("=" * 100)
    print("FINAL BENCHMARK SUMMARY")
    print("=" * 100)

    print(
        f"{'Type':<12}"
        f"{'Model':<24}"
        f"{'Device':<10}"
        f"{'Mean(ms)':>12}"
        f"{'Median(ms)':>14}"
        f"{'Min(ms)':>12}"
        f"{'VRAM(MB)':>14}"
        f"{'Dim':>8}"
    )

    print("-" * 100)

    for result in results:
        dimension = result.get("dimension", "-")

        print(
            f"{result['type']:<12}"
            f"{result['model']:<24}"
            f"{result['device']:<10}"
            f"{result['mean_ms']:>12.2f}"
            f"{result['median_ms']:>14.2f}"
            f"{result['min_ms']:>12.2f}"
            f"{result['vram_allocated_mb']:>14.2f}"
            f"{str(dimension):>8}"
        )

    print("=" * 100)

    # ========================================================
    # Save results
    # ========================================================

    try:
        import json

        output_path = os.path.join(
            os.path.dirname(__file__),
            "benchmark_results.json",
        )

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        print("\nResults saved to:")
        print(output_path)

    except Exception as exc:
        print(f"\nCould not save JSON results: {exc}")

    print("\nBenchmark completed successfully.")


if __name__ == "__main__":
    main()
