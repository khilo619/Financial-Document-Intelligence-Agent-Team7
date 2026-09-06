# 📢 Team Update: Day 2 & Day 3 Milestone – Evaluation & MLOps Suite Complete! 🚀

---

### 🎯 Overview: `feature/eval-service` is LIVE on Remote! 🎉

Hey team! **Khaled** (Member 5 – Evaluation & MLOps Lead) here.

I have completed and pushed the full **Evaluation Engine & Langfuse MLOps Suite** on my feature branch:
🔗 **Branch**: [`feature/eval-service`](https://github.com/khilo619/Financial-Document-Intelligence-Agent-Team7/tree/feature/eval-service)

Everything is tested, verified, and passing:
* **32 Automated Unit Tests Passing** (`pytest tests/ -v`) in < 2 seconds!
* **Zero Linting / Formatting Errors** (`ruff check .` & `ruff format --check .`).
* Ready for automated benchmarking across the gold practice set (`questions_setA_practice.json`).

---

### 🌟 What Was Built & Established (Days 2 & 3 Deliverables)

#### 1. 🧮 Pure Metric Computation Engine (`services/eval_service/src/metrics.py`)
Implemented the official Project LEDGER financial benchmark formulas:
* **Exact Match (EM)**: Normalized text comparison (lowercasing, punctuation stripping, article removal, and whitespace standardization).
* **Token-Level F1**: Precision, Recall, and F1 harmonic mean for text and multi-span answers.
* **Numerical Accuracy with Relative Tolerance ($\epsilon = 0.01$, i.e., 1%)**:
  $$\text{Acc}_{\text{num}} = \mathbb{I}\left(\frac{|\hat{v} - v^*|}{\max(|v^*|, 10^{-5})} \le 0.01\right)$$
  - **Financial Text Parser**: Cleans currency symbols (`$`, `€`, `£`), commas in numbers (`"304,811"` $\rightarrow 304811.0$), accounting negative parentheses (`"(124.50)"` $\rightarrow -124.50$), and multipliers (`14.3M` $\rightarrow 14,300,000$).
  - **Scale Harmonization**: Automatically converts `"scale": "thousand"`, `"million"`, and `"billion"` to base units so that models preserving table units and models outputting raw dollars are both scored fairly!
* **Retrieval Recall@K**: Verifies whether gold source documents and page numbers were successfully fetched in the top retrieved chunks.

#### 2. 🧪 Decoupled Standalone Pipeline Emulator (`services/eval_service/src/mock_pipeline.py`)
* **`MockPipelineClient`**: Emulates realistic, schema-compliant `StrictAnswer` responses directly from `questions_setA_practice.json`.
* **Controlled Error Injection**: Simulates retrieval misses, formula arithmetic errors, and model abstentions with seeded randomness. This allowed us to build and thoroughly test the evaluation engine **today** without waiting for upstream services!
* **`HttpPipelineClient`**: The production client that will query the live system (`http://localhost:8001/ask`) with zero code changes needed when the agent is ready.

#### 3. 📊 Automated Benchmark Harness (`services/eval_service/src/benchmark.py`)
* Reads `questions_setA_practice.json`, filters by sample size or `task_family`, queries the pipeline, computes all metrics, and generates a formatted summary report.
* **Automatic Failure Analysis Stage Tagging**: Categorizes failed questions into root-cause stages:
  - `Retrieval`
  - `Reasoning & Calculation`
  - `Agent Abstention`
  - `Extraction / Schema`

#### 4. 🔭 Langfuse Telemetry & Observability (`services/eval_service/src/langfuse_reporter.py`)
* Connects to Langfuse to version datasets, track experiment runs, and log quantitative scores (`exact_match`, `f1_score`, `numerical_accuracy`, `retrieval_recall`).
* **Graceful Offline Fallback**: If Langfuse is offline or keys are absent in CI, the system logs cleanly to the console without throwing errors or crashing builds.

#### 5. 🌐 FastAPI Service Endpoints (`services/eval_service/src/main.py`)
* `GET /health`: Service health check (Port 8006).
* `POST /run_benchmark`: Triggers automated evaluation sweeps (supports sample size, mock vs. live, task family filters).
* `GET /metrics`: Returns baseline targets vs. actuals.
* `GET /failure_analysis`: Pre-formats the **5-Example Failure Analysis** report required by the project handbook.

---

### 🔍 Real Terminal Benchmark Output (Sample Run on 10 Questions)

```text
=======================================================
           PROJECT LEDGER BENCHMARK REPORT             
=======================================================
Total Evaluated:        10
Overall Pass Rate:      80.0%
Mean Exact Match:       0.8000
Mean Token F1:          0.8000
Mean Numerical Acc:     0.8000 (epsilon=0.01)
Mean Retrieval Recall:  0.9000
Average Latency:        112.0 ms
=======================================================

Sample Failure Analysis (2 failures recorded):
  - [A001] Stage: Retrieval | Relevant gold evidence documents ['cts-corporation_2019.pdf', 'jabil-circuit-inc_2019.pdf'] were not found in top retrieved chunks.
  - [A002] Stage: Reasoning & Calculation | Formula calculation error: predicted -5.25 but expected -3.5 (scale=million, derivation=(-2.5+(-4.5))/2).
```

---

### 🤝 What This Means for Teammates & Integration

1. **For Youssef (Member 3 – Reasoning Brain & Agent Lead)**:
   * Keep developing your LangGraph workflow in `services/agent_service/`.
   * As long as your agent outputs the `StrictAnswer` schema from `shared/models.py`, our benchmark runner can automatically grade your model against the 100+ questions in `questions_setA_practice.json`!
2. **For Omar (Member 2 – Search & Vector DB Lead)**:
   * Your `retrieval-api` should return chunks with `document_id` and `page`.
   * Our `retrieval_recall@K` metric will verify whether your hybrid search (BM25 + Qdrant dense + BGE Reranker) successfully captures the gold PDF evidence.
3. **For Member 4 (Schema, Validator & Tools Lead)**:
   * Your `answer-validator-api` protects the runtime queries from syntax errors and invalid citations.
   * Our evaluation engine will measure the quantitative accuracy of the validated outputs.

---

### 💻 Try It Yourself! (Local Reproduction)

To test the evaluation benchmark on your machine:

```bash
# 1. Fetch and checkout the feature branch
git fetch origin
git checkout feature/eval-service

# 2. Run the test suite (32 passing tests)
pytest tests/ -v

# 3. Run a test benchmark sweep (sample of 10 questions)
python3 -m services.eval_service.src.benchmark --sample-size 10
```

---

*Great progress, team! Let's keep the momentum going!* 💪
