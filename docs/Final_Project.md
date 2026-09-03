# MiA robotics AI TEAM TRAINING'27 Final Project

**MIA robotics**

## Project LEDGER: Financial Document Intelligence Agent

### Project Overview
This project asks you to build LEDGER, an Al financial analyst capable of ingesting a collection of financial-report PDFs and answering natural-language questions across the entire collection. You will build a complete document-to-answer pipeline, focusing on faithful document understanding and verifiable, evidence-grounded answers. A core challenge is to ensure the reasoning agent never fabricates a number or a citation — every answer must be checked against real retrieved evidence before it is shown to the user. The final product will be a fully demonstrable system that indexes real financial reports and answers questions about them without hallucinating.

### Core Objective
To design, build, and deploy a complete PDF Document Understanding → Retrieval → Reasoning → Answer pipeline. The system will parse raw financial-report PDFs (text, tables, layout), index them for retrieval, interpret a user's question, retrieve and reason over the relevant evidence, and then validate the generated answer against a strict schema - including a mandatory citation - before it is returned to the user. The application operates over the entire indexed corpus by default: a user should be able to ask a question without first telling the system which document contains the answer.

### Dataset
The primary dataset for the project is TAT-DQA (Towards Complex Document Understanding by Discrete Reasoning) - 16,558 real questions over 2,758 financial-report documents (3,067 pages), each containing both text and tables. Questions include direct lookups, multi-span answers, comparison, counting, and arithmetic (addition, subtraction, multiplication, division, sorting, and compositions of these). The dataset provides the original PDFs, a pre-parsed reference representation, question/answer pairs, ground-truth answers, and numerical derivations.

### Important Dataset Rule
The original PDFs must be the input to your production ingestion pipeline. You may NOT load the dataset's own pre-parsed JSON and use it as your application's document representation - that would be grading a shortcut, not a real PDF-understanding system. The provided structured content may only be used as ground truth to evaluate your own extraction, or as debugging reference. The supplied questions, answers, and derivations should be used to build the held-out evaluation benchmark.

### System Architecture & Technical Requirements
The system must be built using a microservices architecture: the seven components below as separate, independently runnable services communicating over HTTP. Containerizing them with Docker Compose is a bonus (see "Bonus Features"), not a requirement - teams should spend their limited time on retrieval quality, agent reasoning, and evaluation rather than container networking, and may choose whatever process/deployment model gets the services talking to each other reliably.

**Required Services:**

1. **Orchestrator Service (orchestrator-api)**
Framework: FastAPI.
Responsibility: The central nervous system. It routes documents and questions between services. If the agent generates an answer, the orchestrator sends it to the answer-validator-api for verification before it reaches the user.

2. **Document Processing Service (doc-processor-api)**
Framework: FastAPI, TorchServe, or similar.
Model Constraint: Must use a deep-learning-based OCR/layout model to convert raw PDFs into a structured representation (text, tables, headings, page numbers, bounding boxes).

3. **Retrieval Service (retrieval-api)**
Framework: FastAPI + a vector database (FAISS, Qdrant, Chroma, or similar).
Requirement: Must expose corpus-wide semantic (embedding) retrieval, AND at least one non-vector retrieval mechanism (e.g. BM25, metadata filtering, or direct table lookup) — not every information problem should be solved with embeddings.
Reranking: Candidates should be over-retrieved and reranked before being handed to the agent (e.g. top 30 → reranker → top 5); the value of reranking should be evaluated, not assumed.
Chunking: Must go beyond naive fixed-size chunking - investigate section-aware, table-aware, and/or parent-child chunking, and preserve document_id/page/section/content_type metadata on every chunk.

4. **Reasoning "Brain" Service (agent-service)**
Framework: LangGraph.
Core Logic: Its goal is to route the question, call retrieval and calculation tools as needed, and produce a parseable answer containing the result and its supporting evidence. Must use real conditional branches (e.g. text vs. table vs. numerical question, sufficient vs. insufficient evidence, retry on weak evidence) not a fixed node1 → node2 → node3 chain.
Tools: Must expose deterministic tools such as `search_documents` (query), `search_tables`(query), `calculate` (expression), and `filter_documents`(metadata). Arithmetic must go through the calculator tool, never be produced by the LLM from memory.
Model Constraint: Must use a resource-efficient LLM.

5. **Evaluation & Observability Service (eval-service)**
Framework: FastAPI + Langfuse.
Tracing: Must trace every step of a request — query classification, retrieval, reranking, tool calls, generation, and verification — with latency, prompts, outputs, and token usage visible per step, not just a pass/fail result.
Evaluation: Must run an automated benchmark against a held-out slice of the TAT-DQA question/answer pairs and report Exact Match, F1, and numerical accuracy for answers, plus Recall@K/Precision@K for retrieval where determinable.
Experiments: Langfuse datasets/experiments should be used to compare pipeline variants (e.g. chunking strategy, reranker on/off) and justify design choices with measured results, not intuition.

6. **User Interface (ui-service)**
Framework: Gradio.
Functionality: A chat view for corpus-wide questions, plus a dashboard view listing indexed documents and any extracted structured data. Every answer must display its source citation (document + page), and ideally let the user open that page.

7. **Answer Validator Service (answer-validator-api)**
Framework: FastAPI / Flask (or any other simple web framework).
Purpose: This service acts as the answer validator. It is the single source of truth for what constitutes a valid, grounded answer.
Endpoint: Must expose an API endpoint (e.g. `/validate_answer`) that accepts a POST request with a JSON payload.
Functionality: Upon receiving a JSON payload, it must perform two actions:
*   **Validation:** It strictly checks if the incoming JSON conforms to the "Strict Answer Schema" below — including the answer type, the presence of required evidence citations, and the data types of every value.
*   **Logging:** if valid, it logs a success message to its console: `[ANSWER-VALIDATOR-SUCCESS] Received and validated answer of type 'calculated' with evidence {'document_id': 'doc_041', 'page': 2}`. If invalid, it logs a detailed error: `[ANSWER-VALIDATOR-ERROR] Invalid answer. Reason: Missing required evidence citation.` or `[ANSWER-VALIDATOR-ERROR] Invalid answer for 'calculated': Missing required key 'formula'.`

**Orchestration Requirement:**
All seven services must run together and be launchable as one system with a single command, a startup script, or clearly documented manual steps in the README. Teams that choose to containerize should define this via a single `docker-compose.yml` (`docker-compose up`); this is a valid and encouraged approach but not the only acceptable one.

### Strict Answer Schema
The agent must generate answers that conform exactly to a strict, typed schema, and the `answer-validator-api` must reject any deviation. The four types below are the required minimum - TAT-DQA's questions include single-fact lookups, arithmetic, multi-item answers, and unanswerable cases, so all four must be supported. Teams may propose additional types (e.g. a dedicated comparison or ranking type) if their own data justifies it, as long as the same rule holds: every type requires cited evidence, and any addition is documented in the README.

**Base Structure:**
```json
{
"answer_type": "<type_name>",
"evidence": [ { "document_id": "...", "page": 0, "section": "..." } ],
"params": {...}
}
```

**Allowed Answer Types:**

#### 1. Type: `direct`
**Description:** Returns a fact retrieved directly from a document (e.g. "What was the operating income reported in 2020?").

| Parameter Name | Data Type | Constraints |
| :--- | :--- | :--- |
| value | string or number | Required. |
| evidence | array | Required. At least 1 citation with document_id and page. |

**Example:**
```json
{
"answer_type": "direct",
"evidence": [ { "document_id": "doc_017", "page": 1, "section": "Income Statement" } ],
"params": { "value": "$142.5M" }
}
```

#### 2. Type: `calculated`
**Description:** Returns a value derived through arithmetic — percentage change, comparison, sum, etc. — performed by a deterministic calculator tool, never by the LLM alone.

| Parameter Name | Data Type | Constraints |
| :--- | :--- | :--- |
| value | number | Required. The final computed result. |
| formula | string | Required. Shows the calculation performed, e.g. "(150-120)/120*100". |
| evidence | array | Required. One citation per operand used in the formula. |

**Example:**
```json
{
"answer_type": "calculated",
"evidence": [
{ "document_id": "doc_041", "page": 2, "section": "Operating Expenses" },
{ "document_id": "doc_041", "page": 1, "section": "Operating Expenses" }
],
"params": { "value": 13.4, "formula": "(3875-3410)/3410*100" }
}
```

#### 3. Type: `multi_span`
**Description:** Returns two or more distinct values pulled from the document(s) — a list of items, a set of line items, or several spans that together answer the question (e.g. "Which three expense categories increased in 2020?"). Do not force a multi-item answer into `direct` — a single value field cannot represent a list without losing information the validator needs to check.

| Parameter Name | Data Type | Constraints |
| :--- | :--- | :--- |
| values | array of string/number | Required. One entry per item in the answer, in the order they support the question. |
| evidence | array | Required. At least one citation per value; a single citation may cover multiple values if they come from the same cell/passage. |

**Example:**
```json
{
"answer_type": "multi_span",
"evidence": [
{ "document_id": "doc_022", "page": 3, "section": "Operating Expenses" }
],
"params": { "values": ["Marketing", "R&D", "Logistics"]}
}
```

#### 4. Type: `insufficient_evidence`
**Description:** Returned when the agent cannot find enough grounding to answer confidently. This must be used instead of guessing - hallucinating an answer is treated as a failure.

| Parameter Name | Data Type | Constraints |
| :--- | :--- | :--- |
| reason | string | Required. Brief explanation of what could not be found. |
| evidence | array | Optional. May be empty. |

**Example:**
```json
{
"answer_type": "insufficient_evidence",
"evidence": [],
"params": { "reason": "No document in the indexed corpus reports restructuring expenses." }
}
```

### Dashboard
In addition to chat, the Gradio UI must include a basic corpus-level dashboard: number of indexed documents, the document list, detected tables, any extracted structured values, and recent queries with their latency. Because TAT-DQA's PDFs are short excerpts (mostly 1-3 pages) rather than full annual reports, a rich company-level financial dashboard is a bonus, not the main objective — the dashboard's job is to make the pipeline's own state inspectable, not to be a finished BI product.

### Evaluation
A good demo is not sufficient - evaluation is mandatory. TAT-DQA already provides questions and ground-truth answers, so a held-out subset must be used to automatically score the complete system end to end (question → your system → predicted answer → compare to ground truth metrics).
*   **Answer quality:** Exact Match, F1, and numerical accuracy.
*   **Retrieval quality** (where determinable): Recall@K, Precision@K, MRR / Hit Rate.
*   **System performance:** average latency, LLM calls per query, token usage, approximate cost.

### Failure Analysis
Each team must select at least five failed evaluation examples and diagnose the real root cause using Langfuse traces — was it OCR, table extraction, chunking, retrieval, reranking, numerical reasoning, or generation? For example: expected 14.3%, but the wrong table from another document was retrieved → root cause: retrieval failure → fix: metadata-aware retrieval. This is graded — it is often the most valuable part of the project.

### Development Process & Collaboration Standards
*   **Version Control:** The project must be hosted on GitHub.
*   **Branching Strategy:** The main branch is to be protected. All development must be done on separate feature branches (e.g. `feature/retrieval-service`, `bugfix/docker-networking`).
*   **Code Integration:** All code must be merged into main via Pull Requests (PRs).
*   **Commit History:** Commit messages must be clear and descriptive, creating a readable log of the project's iterative development.

### Bonus Features
*   Full containerization of all seven services with Docker Compose for one-command deployment.
*   Bounding-box highlighting of the cited evidence on the source PDF page.
*   Query decomposition, multi-query retrieval, or HyDE.
*   Contextual retrieval, semantic caching, conversation memory.
*   SQL/structured-data querying alongside vector search.
*   Human-in-the-loop correction of extracted fields.
Bonus features only receive credit if their effect is actually demonstrated, not just implemented.

### Required User Interface & Final Demo Checklist

**Required User Interface**
The final application, built in Gradio, must contain at minimum:
*   **Document interface** - inspect the indexed documents.
*   **Chat** - ask questions across the complete document corpus (with optional document/company scoping, but corpus-wide is the default).
*   **Answer evidence** - the retrieved source document(s)/page(s) shown alongside every answer.
*   **Dashboard** - basic information about indexed documents and/or extracted structured data.

**Final Demo Checklist**
At the final demo, each team must show:
*   **Document ingestion** — show a raw PDF and its processed representation.
*   **Table understanding** - show one financial table extracted and represented.
*   **Corpus-wide question** — ask a question without naming its document; the system must find the right source.
*   **Numerical question** — show retrieval followed by a real calculation.
*   **Evidence** - show the source page/table backing the answer.
*   **LangGraph** — walk through one real conditional path.
*   **Langfuse** — open a real trace end to end.
*   **Evaluation** — present quantitative results from the held-out benchmark.
*   **Experiment** - show one change that measurably helped or hurt.
*   **Failure** — show one real failure and its root cause.

### Definition of "Done"
The project is considered complete when all of the following criteria are met:
*   [x] All seven services run correctly and can be launched together as one system (Docker Compose, a startup script, or documented manual steps).
*   [x] Document ingestion uses the raw PDFs, never the dataset's pre-parsed JSON.
*   [x] A Gradio UI allows for corpus-wide chat and a dashboard view of indexed documents.
*   [x] Retrieval combines dense (vector) search with at least one non-vector method, plus reranking.
*   [x] When a question is answered, a schema-compliant JSON answer is sent to and successfully validated by the `answer-validator-api`.
*   [x] The system correctly handles cases where the agent cannot find sufficient evidence, which will be returned as `insufficient_evidence` and logged by the `answer-validator-api`, rather than hallucinated.
*   [x] The complete pipeline is traced in Langfuse, with an automated benchmark run against a held-out TAT-DQA subset and reported metrics.
*   [x] A five-example failure analysis is documented, with each failure root-caused to a specific pipeline stage.
*   [x] The code is hosted on GitHub, with a clear history of Pull Requests from all team members.
*   [x] The README.md provides clear instructions on how to set up and run the project.
