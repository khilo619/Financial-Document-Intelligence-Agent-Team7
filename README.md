# 📊 Project LEDGER: Financial Document Intelligence Agent
### *Multi-Service Evidence Grounding, Discrete Reasoning & Evaluation Pipeline*
**MIA Robotics AI Team Training '27 — Team 7**

---

## 🌟 Executive Summary
**Project LEDGER** is an enterprise-grade financial intelligence system capable of ingesting corporate financial reports (SEC Form 10-K filings, annual statements, balance sheets) and answering complex analytical questions across the entire document collection without hallucination.

The system addresses four primary hurdles in automated financial document understanding:
1. **Multimodal Table & Layout Understanding**: Parses complex financial tables, merged headers, and parenthetical footnotes into layout-aware Markdown representations.
2. **Hybrid Semantic & Lexical Retrieval**: Combines dense vector search (`BAAI/bge-large-en-v1.5`) in Qdrant with sparse BM25 inverted indexing, fused via Reciprocal Rank Fusion (RRF) and reranked using a Cross-Encoder (`BAAI/bge-reranker-large`).
3. **Discrete Agentic Reasoning (LangGraph)**: Routes queries through cyclic multi-hop reasoning graphs and delegates all arithmetic to a sandboxed deterministic calculator tool.
4. **Strict Evidence Gatekeeping**: Every generated answer must pass an independent validation gatekeeper (`answer-validator-api`) that enforces strict schema constraints and verifies source PDF citations before reaching the user.

---

## 🏛️ System Architecture & Microservice Registry

LEDGER is built as a distributed microservice cluster of 7 independent services communicating over HTTP:

```
                                  Host Machine (localhost)
                                              │
           ┌───────────────────┬──────────────┼─────────────────┬──────────────────┐
           │                   │              │                 │                  │
        Port 8000           Port 8001      Port 8002         Port 8003          Port 6333
           ▼                   ▼              ▼                 ▼                  ▼
      [ui-service]       [orchestrator]  [doc-processor]   [retrieval-api]     [qdrant]
           │                   │                                │                  ▲
           │                   │                                └──────────────────┘
           │                   │ (Internal Docker Bridge: 'ledger-network')
           │                   ▼
           │             [agent-service] (Port 8004)
           │                   │
           │                   ▼
           └────────────►[answer-validator-api] (Port 8005)
```

| Port | Service Name | Lead / Owner | Core Functionality |
| :---: | :--- | :--- | :--- |
| **`8000`** | `ui-service` | **Ahmed** (Member 6) | Gradio Web App with Corpus Chat, Citation View & System Dashboard. |
| **`8001`** | `orchestrator-api` | **Ahmed** (Member 6) | Central API Gateway routing queries and enforcing gatekeeper validation. |
| **`8002`** | `doc-processor-api` | **Zeina** (Member 1) | Deep-learning Layout OCR, Markdown table parsing, and bounding boxes. |
| **`8003`** | `retrieval-api` | **Salma** (Member 2) | Hybrid Qdrant Vector Search + BM25, RRF fusion & Cross-Encoder reranker. |
| **`8004`** | `agent-service` | **Youssef** (Member 3) | LangGraph cyclic reasoning brain & sub-query decomposition. |
| **`8005`** | `answer-validator-api` | **Omar** (Member 4) | Strict Answer Schema gatekeeper & safe deterministic Python calculator. |
| **`8006`** | `eval-service` | **Khaled** (Member 5) | Automated benchmarking (`questions_setA_practice.json`) & Langfuse tracing. |
| *(6333)* | *qdrant* | *Infrastructure* | Standalone Qdrant vector database container. |
| *(3000)* | *langfuse* | *Infrastructure* | Observability dashboard and tracing engine. |

---

## 🔒 Contract-First Schema: The 4 Answer Types

All answers returned by LEDGER conform strictly to `shared/models.py`:

```json
{
  "answer_type": "<type_name>",
  "evidence": [
    {
      "document_id": "cts-corporation_2019.pdf",
      "page": 1,
      "section": "Note 4: Inventories",
      "bbox": [50.0, 90.0, 500.0, 200.0]
    }
  ],
  "params": { ... }
}
```

1. **`direct`**: Direct factual lookup from document evidence (`params: {"value": "$142.5M"}`). Requires $\ge 1$ evidence citation.
2. **`calculated`**: Result of deterministic arithmetic evaluation (`params: {"value": 304811.0, "formula": "abs(9447 - 314258)"}`). Requires $\ge 1$ citation for the operands.
3. **`multi_span`**: Two or more distinct items (`params: {"values": ["Marketing", "R&D", "Logistics"]}`). Requires $\ge 1$ citation.
4. **`insufficient_evidence`**: Abstention reason when grounding is absent (`params: {"reason": "No document reports restructuring charges for 2019."}`).

---

## 🚀 Quickstart Guide

### 1. Prerequisites
- Python 3.11+
- Git
- Docker & Docker Compose (optional for local standalone development)

### 2. Local Setup (Without Docker)
```bash
# Clone the repository
git clone https://github.com/khilo619/Financial-Document-Intelligence-Agent-Team7.git
cd Financial-Document-Intelligence-Agent-Team7

# Create virtual environment and initialize local .env
make venv
source .venv/bin/activate
make env

# Install dev dependencies
make install-dev

# Run automated tests and linter
make test
make lint
```

### 3. Multi-Service Cluster (With Docker Compose)
```bash
# Launch all 7 microservices + Qdrant
make up

# Inspect running cluster status
make status

# Stream container logs
make logs

# Tear down cluster
make down
```

---

## 📚 In-Depth Technical Documentation

For complete mathematical formulations, dataset breakdown, and architecture designs, refer to our dedicated guides in [`docs/`](file:///home/khaled/MIA/Project/docs/):

* 📘 **[LEDGER Engineering Handbook](file:///home/khaled/MIA/Project/docs/LEDGER_HANDBOOK.md)**: 50+ page handbook covering RAG mathematics (Cosine, BM25, RRF, Cross-Encoders), financial scale alignment, and Langfuse observability.
* 📋 **[Final Project Specification](file:///home/khaled/MIA/Project/docs/Final_Project.md)**: The core training specification, evaluation rules, and definition of "Done".
* 📖 **[Benchmark Record Schema Guide](file:///home/khaled/MIA/Project/docs/record_schema_guide_ar-1.md)**: Arabic guide explaining the benchmark schema layers in `questions_setA_practice.json`.

---

## 👥 Git Flow & PR Governance

1. **`main` is protected**: Direct pushes to `main` are restricted.
2. **Branching convention**:
   - `feature/doc-processor-ocr` (Zeina)
   - `feature/retrieval-service` (Salma)
   - `feature/agent-brain` (Youssef)
   - `feature/validator-service` (Omar)
   - `feature/eval-service` (Khaled)
   - `feature/ui-dashboard-compose` (Ahmed)
3. **CI/CD Quality Gate**: Every Pull Request automatically runs Ruff linter and the Pytest contract suite via GitHub Actions (`.github/workflows/ci.yml`). PRs cannot be merged if any check fails.
