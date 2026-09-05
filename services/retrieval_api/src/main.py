"""
retrieval_api: Hybrid Dense + Sparse Search and Cross-Encoder Reranking microservice.
Owned by Salma (Member 2) - Initial scaffold by Khaled (Repo Lead).
"""

import logging
import time

from fastapi import FastAPI

from shared.config import ServiceName
from shared.models import (
    RetrievedChunk,
    SearchQueryRequest,
    SearchQueryResponse,
)

from .bm25_engine import BM25Engine
from .qdrant_store import QdrantStore
from .rrf_fusion import RRFFusion
from .reranker import Reranker


# ---------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
)

logger = logging.getLogger("RetrievalAPI")


# ---------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------

app = FastAPI(
    title="Project LEDGER - Hybrid Retrieval API",
    description=(
        "Dense (Qdrant) + Sparse (BM25) search "
        "fused via RRF and reranked via Cross-Encoder."
    ),
    version="0.1.0",
)


# ---------------------------------------------------------
# Initialize retrieval components
# ---------------------------------------------------------

# Initialize the Qdrant vector store.
# This also loads the BGE embedding model.
qdrant_store = QdrantStore()


# Initialize the BM25 sparse retrieval engine.
bm25_engine = BM25Engine()


# Initialize Reciprocal Rank Fusion.
rrf_fusion = RRFFusion()


# Initialize the Cross-Encoder reranker.
# This loads BAAI/bge-reranker-large.
reranker = Reranker()


# ---------------------------------------------------------
# Initialize BM25 index
# ---------------------------------------------------------

# Load all documents currently stored in Qdrant.
# These documents are used to build the BM25 index.
documents = qdrant_store.get_all_documents()


# Build the BM25 index from the retrieved documents.
bm25_engine.index_documents(documents)


logger.info(
    "BM25 index initialized with %d documents",
    len(documents),
)


# ---------------------------------------------------------
# Health check
# ---------------------------------------------------------

@app.get("/health")
def health_check():
    """
    Health check endpoint used by Docker and other services.
    """

    return {
        "status": "healthy",
        "service": ServiceName.RETRIEVAL.value,
        "port": 8003,
    }


# ---------------------------------------------------------
# Hybrid search endpoint
# ---------------------------------------------------------

@app.post(
    "/search",
    response_model=SearchQueryResponse,
)
def search_documents(request: SearchQueryRequest):
    """
    Execute hybrid retrieval:

        Query
          ↓
        Dense Search (Qdrant)
          +
        Sparse Search (BM25)
          ↓
        RRF Fusion
          ↓
        Top-K Candidates
          ↓
        Cross-Encoder Reranking
          ↓
        Top-N Results
    """

    start_time = time.time()

    logger.info(
        "Executing search query: '%s' "
        "(top_k=%d, top_n=%d, filters=%s, reranking=%s)",
        request.query,
        request.top_k,
        request.top_n,
        request.filters,
        request.use_reranking,
    )

    # -----------------------------------------------------
    # 1. Dense retrieval using Qdrant
    # -----------------------------------------------------

    qdrant_results = qdrant_store.search(
        query=request.query,
        top_k=request.top_k,
        filters=request.filters,
    )

    logger.info(
        "Dense search returned %d results",
        len(qdrant_results),
    )


    # -----------------------------------------------------
    # 2. Sparse retrieval using BM25
    # -----------------------------------------------------

    # Pass the same filters used by Qdrant to BM25.
    # This prevents sparse retrieval from returning
    # documents outside the requested filter scope.
    bm25_results = bm25_engine.search(
        query=request.query,
        top_k=request.top_k,
        filters=request.filters,
    )

    logger.info(
        "Sparse search returned %d results",
        len(bm25_results),
    )


    # -----------------------------------------------------
    # 3. Reciprocal Rank Fusion
    # -----------------------------------------------------

    fused_results = rrf_fusion.fuse(
        [
            qdrant_results,
            bm25_results,
        ]
    )

    logger.info(
        "Dense results: %d | Sparse results: %d | "
        "Fused results: %d",
        len(qdrant_results),
        len(bm25_results),
        len(fused_results),
    )


    # -----------------------------------------------------
    # 4. Select Top-K candidates for reranking
    # -----------------------------------------------------

    candidates = fused_results[: request.top_k]

    logger.info(
        "Selected %d candidates for reranking",
        len(candidates),
    )


    # -----------------------------------------------------
    # 5. Cross-Encoder reranking
    # -----------------------------------------------------

    if request.use_reranking and candidates:

        logger.info(
            "Running Cross-Encoder reranking on %d candidates",
            len(candidates),
        )

        final_results = reranker.rerank(
            query=request.query,
            results=candidates,
            top_n=request.top_n,
        )

    else:

        logger.info(
            "Cross-Encoder reranking disabled",
        )

        final_results = candidates[: request.top_n]


    # -----------------------------------------------------
    # 6. Convert results to RetrievedChunk
    # -----------------------------------------------------

    results = []

    for result in final_results:

        metadata = result.get("metadata", {})

        # If reranking was used successfully,
        # use the Cross-Encoder score as the final score.
        #
        # Otherwise, use the RRF score.
        if (
            request.use_reranking
            and result.get("rerank_score") is not None
        ):
            final_score = result["rerank_score"]

        else:
            final_score = result["rrf_score"]


        results.append(
            RetrievedChunk(
                chunk_id=result["chunk_id"],
                document_id=result["document_id"],
                page=result["page"],
                section=metadata.get("section", ""),
                content_type=result["content_type"],
                content=result["content"],
                score=final_score,
                dense_score=result.get("dense_score"),
                sparse_score=result.get("sparse_score"),
                rerank_score=result.get("rerank_score"),
                bbox=result.get("bbox"),
                metadata=metadata,
            )
        )


    # -----------------------------------------------------
    # 7. Calculate execution time
    # -----------------------------------------------------

    elapsed_ms = round(
        (time.time() - start_time) * 1000,
        2,
    )


    logger.info(
        "Search completed: %d results in %s ms",
        len(results),
        elapsed_ms,
    )


    # -----------------------------------------------------
    # 8. Return API response
    # -----------------------------------------------------

    return SearchQueryResponse(
        query=request.query,
        results=results,
        total_found=len(fused_results),
        execution_time_ms=elapsed_ms,
    )