"""
ui_service: Gradio Web Application for Project LEDGER.

Responsibilities:
- Corpus-wide financial Q&A.
- Render validated strict answer types.
- Display grounding evidence and citations.
- Display request latency and trace IDs.
- Display live microservice health.
- Display recent orchestrated queries.
- Never generate or display fake financial answers.

Owned by Ahmed.
"""

from __future__ import annotations

import logging
import os
import uuid
from pathlib import Path
from typing import Any

import gradio as gr
import httpx

from shared.config import ServiceName, get_service_url


# =============================================================================
# Logging
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
)

logger = logging.getLogger("UIService")


# =============================================================================
# Backend URLs
# =============================================================================

ORCHESTRATOR_BASE_URL = get_service_url(
    ServiceName.ORCHESTRATOR.value
)

ASK_URL = f"{ORCHESTRATOR_BASE_URL}/ask"

SERVICES_HEALTH_URL = (
    f"{ORCHESTRATOR_BASE_URL}/services/health"
)

RECENT_QUERIES_URL = (
    f"{ORCHESTRATOR_BASE_URL}/recent-queries"
)

DOCUMENT_UPLOAD_URL = (
    f"{ORCHESTRATOR_BASE_URL}/documents/upload"
)


# =============================================================================
# UI Constants
# =============================================================================

EMPTY_EVIDENCE = """
### Sources

_No sources yet. Ask LEDGER a question to see the supporting evidence._
"""

EMPTY_METADATA = """
### Request details

_No request has been sent yet._
"""


# =============================================================================
# Styling
# =============================================================================

ASSET_DIR = Path(__file__).resolve().parent
APP_CSS = (ASSET_DIR / "styles.css").read_text(encoding="utf-8")
APP_JS = (ASSET_DIR / "ui.js").read_text(encoding="utf-8")


# =============================================================================
# Generic Helpers
# =============================================================================

def _new_session_id() -> str:
    return str(uuid.uuid4())


def _safe_history(
    history: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:

    if history is None:
        return []

    return list(history)


def _append_message(
    history: list[dict[str, Any]],
    role: str,
    content: str,
) -> None:

    history.append(
        {
            "role": role,
            "content": content,
        }
    )


def _extract_error_message(
    response: httpx.Response,
) -> str:

    try:
        payload = response.json()

    except ValueError:
        return (
            f"The backend returned HTTP "
            f"{response.status_code}."
        )

    detail = payload.get("detail")

    if isinstance(
        detail,
        dict,
    ):
        message = detail.get(
            "message",
            "The request failed.",
        )

        reason = detail.get(
            "reason"
        )

        if reason:
            return (
                f"{message}\n\n"
                f"**Reason:** {reason}"
            )

        return str(
            message
        )

    if detail:
        return str(
            detail
        )

    return (
        f"The backend returned HTTP "
        f"{response.status_code}."
    )


# =============================================================================
# Answer Rendering
# =============================================================================

def _format_answer(
    answer_data: dict[str, Any],
) -> str:

    answer_type = answer_data.get(
        "answer_type",
        "unknown",
    )

    params = answer_data.get(
        "params",
        {},
    )

    if answer_type == "direct":

        value = params.get(
            "value",
            "N/A",
        )

        return (
            f"## {value}\n\n"
            "✓ **Verified answer**"
        )

    if answer_type == "calculated":

        value = params.get(
            "value",
            "N/A",
        )

        formula = params.get(
            "formula",
            "N/A",
        )

        return (
            f"## {value}\n\n"
            "**Calculated answer**\n\n"
            f"Formula: `{formula}`\n\n"
            "✓ **Verified**"
        )

    if answer_type == "multi_span":

        values = params.get(
            "values",
            [],
        )

        if values:

            value_lines = "\n".join(
                f"- {value}"
                for value in values
            )

        else:

            value_lines = (
                "_No answer values were returned._"
            )

        return (
            f"{value_lines}\n\n"
            "✓ **Verified multi-item answer**"
        )

    if answer_type == "insufficient_evidence":

        reason = params.get(
            "reason",
            (
                "The indexed corpus did not "
                "provide enough reliable evidence."
            ),
        )

        return (
            "I couldn't find enough reliable evidence "
            "to answer this question.\n\n"
            f"**Reason:** {reason}\n\n"
            "⚠️ **Insufficient evidence**"
        )

    logger.warning(
        "Unsupported answer type received: %s",
        answer_type,
    )

    return (
        "The backend returned an answer type "
        "that LEDGER does not recognize."
    )


def _format_evidence(
    evidence: list[dict[str, Any]] | None,
) -> str:

    evidence = evidence or []

    if not evidence:

        return (
            "### Sources\n\n"
            "_No evidence citations were returned._"
        )

    lines = [
        f"### Sources · {len(evidence)}",
        "",
    ]

    for index, citation in enumerate(
        evidence,
        start=1,
    ):

        document_id = citation.get(
            "document_id",
            "Unknown document",
        )

        page = citation.get(
            "page",
            "N/A",
        )

        section = citation.get(
            "section"
        )

        bbox = citation.get(
            "bbox"
        )

        lines.append(
            f"**{index}. {document_id}**"
        )

        details = (
            f"Page {page}"
        )

        if section:
            details += (
                f" · {section}"
            )

        lines.append(
            details
        )

        if bbox:
            lines.append(
                f"`bbox: {bbox}`"
            )

        lines.append("")

    return "\n".join(
        lines
    )


def _format_request_metadata(
    response_data: dict[str, Any],
) -> str:

    latency = response_data.get(
        "latency_ms",
        0,
    )

    trace_id = response_data.get(
        "trace_id",
        "N/A",
    )

    return (
        "### Request details\n\n"
        f"✓ Validated  \n"
        f"Latency: `{latency} ms`  \n"
        f"Trace: `{trace_id}`"
    )


# =============================================================================
# Chat
# =============================================================================

def query_ledger(
    user_question: str,
    history: list[dict[str, Any]] | None,
    session_id: str | None,
):

    history = _safe_history(
        history
    )

    question = (
        user_question
        or ""
    ).strip()

    if not question:

        return (
            history,
            "",
            (
                "### Sources\n\n"
                "Please enter a question first."
            ),
            (
                "### Request details\n\n"
                "_No request was sent._"
            ),
            session_id,
        )

    if not session_id:
        session_id = (
            _new_session_id()
        )

    _append_message(
        history,
        "user",
        question,
    )

    logger.info(
        "Forwarding query to Orchestrator: '%s'",
        question,
    )

    payload = {
        "query": question,
        "document_id": None,
        "session_id": session_id,
    }

    try:

        with httpx.Client(
            timeout=70.0,
        ) as client:

            response = client.post(
                ASK_URL,
                json=payload,
            )

    except httpx.TimeoutException:

        logger.error(
            "Request to Orchestrator timed out."
        )

        message = (
            "The request timed out before LEDGER "
            "could return a validated answer."
        )

        _append_message(
            history,
            "assistant",
            message,
        )

        return (
            history,
            "",
            (
                "### Sources\n\n"
                "_No sources returned._"
            ),
            (
                "### Request details\n\n"
                "Status: `timeout`"
            ),
            session_id,
        )

    except httpx.RequestError as exc:

        logger.error(
            "Could not connect to Orchestrator: %s",
            exc,
        )

        message = (
            "LEDGER's backend is currently unavailable. "
            "No financial answer was generated."
        )

        _append_message(
            history,
            "assistant",
            message,
        )

        return (
            history,
            "",
            (
                "### Sources\n\n"
                "_No sources returned._"
            ),
            (
                "### Request details\n\n"
                "Status: `backend unavailable`"
            ),
            session_id,
        )

    if response.status_code != 200:

        error_text = (
            _extract_error_message(
                response
            )
        )

        logger.warning(
            "Orchestrator returned HTTP %s: %s",
            response.status_code,
            error_text,
        )

        _append_message(
            history,
            "assistant",
            (
                "I couldn't return a validated answer.\n\n"
                f"{error_text}"
            ),
        )

        return (
            history,
            "",
            (
                "### Sources\n\n"
                "_No validated sources returned._"
            ),
            (
                "### Request details\n\n"
                f"HTTP status: "
                f"`{response.status_code}`"
            ),
            session_id,
        )

    try:

        response_data = (
            response.json()
        )

    except ValueError:

        logger.error(
            "Orchestrator returned malformed JSON."
        )

        _append_message(
            history,
            "assistant",
            (
                "The backend returned an invalid "
                "response."
            ),
        )

        return (
            history,
            "",
            (
                "### Sources\n\n"
                "_No valid sources returned._"
            ),
            (
                "### Request details\n\n"
                "Status: `invalid response`"
            ),
            session_id,
        )

    answer_data = response_data.get(
        "answer"
    )

    if not isinstance(
        answer_data,
        dict,
    ):

        logger.error(
            "Invalid answer object returned."
        )

        _append_message(
            history,
            "assistant",
            (
                "The backend returned an invalid "
                "answer structure."
            ),
        )

        return (
            history,
            "",
            (
                "### Sources\n\n"
                "_No valid sources returned._"
            ),
            (
                "### Request details\n\n"
                "Status: `invalid answer structure`"
            ),
            session_id,
        )

    answer_markdown = (
        _format_answer(
            answer_data
        )
    )

    evidence_markdown = (
        _format_evidence(
            answer_data.get(
                "evidence",
                [],
            )
        )
    )

    metadata_markdown = (
        _format_request_metadata(
            response_data
        )
    )

    _append_message(
        history,
        "assistant",
        answer_markdown,
    )

    return (
        history,
        "",
        evidence_markdown,
        metadata_markdown,
        session_id,
    )


def new_chat():

    return (
        [],
        "",
        EMPTY_EVIDENCE,
        EMPTY_METADATA,
        _new_session_id(),
    )


# =============================================================================
# Suggestions
# =============================================================================

def suggestion_compare():
    return (
        "What is the difference in finished goods "
        "between CTS and Jabil?"
    )


def suggestion_income():
    return (
        "What was the operating income reported "
        "in the financial reports?"
    )


def suggestion_percentage():
    return (
        "Calculate the percentage change between "
        "the relevant financial values."
    )


# =============================================================================
# Document Upload
# =============================================================================

def upload_document(
    pdf_path: str | None,
    document_id: str | None,
):
    """
    Upload one raw PDF to the Orchestrator document gateway.

    The UI never fabricates processing results. It only renders the
    ProcessPdfResponse returned by the backend.
    """

    if not pdf_path:
        return (
            "### Upload status\n\n"
            "Choose a PDF file before starting document processing.",
            None,
        )

    file_path = Path(pdf_path)

    if file_path.suffix.lower() != ".pdf":
        return (
            "### Upload status\n\n"
            "Only `.pdf` files are supported.",
            None,
        )

    if not file_path.exists():
        logger.error(
            "Uploaded PDF path no longer exists: %s",
            file_path,
        )

        return (
            "### Upload status\n\n"
            "The uploaded file is no longer available. "
            "Please choose the PDF again.",
            None,
        )

    clean_document_id = (
        document_id
        or ""
    ).strip()

    logger.info(
        "Uploading document to Orchestrator: "
        "filename='%s' document_id='%s'",
        file_path.name,
        clean_document_id or "auto",
    )

    try:
        with (
            file_path.open("rb") as pdf_file,
            httpx.Client(timeout=190.0) as client,
        ):
            files = {
                "file": (
                    file_path.name,
                    pdf_file,
                    "application/pdf",
                )
            }

            form_data: dict[str, str] = {}

            if clean_document_id:
                form_data["document_id"] = (
                    clean_document_id
                )

            response = client.post(
                DOCUMENT_UPLOAD_URL,
                files=files,
                data=form_data,
            )

    except httpx.TimeoutException:
        logger.error(
            "Document upload timed out: %s",
            file_path.name,
        )

        return (
            "### ⏱️ Processing timed out\n\n"
            "The backend took too long to process this PDF. "
            "No processed document was returned.",
            None,
        )

    except httpx.RequestError as exc:
        logger.error(
            "Could not reach Orchestrator document endpoint: %s",
            exc,
        )

        return (
            "### 🔴 Backend unavailable\n\n"
            "The Orchestrator could not be reached. "
            "No processed document was returned.",
            None,
        )

    if response.status_code != 200:
        error_text = _extract_error_message(
            response
        )

        logger.warning(
            "Document upload failed with HTTP %s: %s",
            response.status_code,
            error_text,
        )

        return (
            "### ❌ Processing failed\n\n"
            f"{error_text}\n\n"
            f"**HTTP status:** `{response.status_code}`",
            None,
        )

    try:
        processed_document = (
            response.json()
        )

    except ValueError:
        logger.error(
            "Orchestrator returned malformed JSON "
            "for document upload."
        )

        return (
            "### ❌ Invalid backend response\n\n"
            "The document was processed, but the backend "
            "returned malformed JSON.",
            None,
        )

    if not isinstance(
        processed_document,
        dict,
    ):
        logger.error(
            "Document upload response is not an object."
        )

        return (
            "### ❌ Invalid document response\n\n"
            "The backend returned an unexpected "
            "document structure.",
            None,
        )

    returned_document_id = (
        processed_document.get(
            "document_id",
            clean_document_id
            or file_path.stem,
        )
    )

    total_pages = processed_document.get(
        "total_pages",
        0,
    )

    blocks = processed_document.get(
        "blocks",
        [],
    )

    total_blocks = processed_document.get(
        "total_blocks",
        len(blocks)
        if isinstance(blocks, list)
        else 0,
    )

    processing_time = processed_document.get(
        "processing_time_s"
    )

    table_count = 0

    if isinstance(blocks, list):
        for block in blocks:
            if not isinstance(
                block,
                dict,
            ):
                continue

            if (
                block.get("content_type") == "table"
                or block.get("table_rows")
            ):
                table_count += 1

    summary_lines = [
        "### ✓ Document processed successfully",
        "",
        f"**Document ID:** `{returned_document_id}`  ",
        f"**Pages:** `{total_pages}`  ",
        f"**Blocks:** `{total_blocks}`  ",
        f"**Tables detected:** `{table_count}`",
    ]

    if processing_time is not None:
        summary_lines.append(
            f"  \n**Processing time:** "
            f"`{processing_time} s`"
        )

    summary_lines.extend(
        [
            "",
            "Open **Processed representation** below "
            "to inspect the structured output returned "
            "by the Document Processor.",
        ]
    )

    logger.info(
        "Document upload completed: "
        "document_id='%s' pages=%s blocks=%s",
        returned_document_id,
        total_pages,
        total_blocks,
    )

    return (
        "\n".join(summary_lines),
        processed_document,
    )


# =============================================================================
# Dashboard
# =============================================================================

def load_service_health():

    try:

        with httpx.Client(
            timeout=6.0,
        ) as client:

            response = client.get(
                SERVICES_HEALTH_URL
            )

    except httpx.TimeoutException:

        return (
            (
                "### System status\n\n"
                "Health check timed out."
            ),
            [],
        )

    except httpx.RequestError:

        return (
            (
                "### System status\n\n"
                "Orchestrator is offline."
            ),
            [],
        )

    if response.status_code != 200:

        return (
            (
                "### System status\n\n"
                f"Health check failed "
                f"(`{response.status_code}`)."
            ),
            [],
        )

    try:

        data = response.json()

    except ValueError:

        return (
            "### System status\n\nInvalid response.",
            [],
        )

    overall_status = data.get(
        "overall_status",
        "unknown",
    )

    healthy_count = data.get(
        "healthy_services",
        0,
    )

    total_services = data.get(
        "total_services",
        0,
    )

    icons = {
        "healthy": "🟢",
        "degraded": "🟡",
        "unavailable": "🔴",
    }

    icon = icons.get(
        overall_status,
        "⚪",
    )

    summary = (
        "### System status\n\n"
        f"{icon} **{overall_status.title()}**  \n"
        f"{healthy_count}/{total_services} "
        "backend services healthy"
    )

    rows: list[list[Any]] = []

    orchestrator = data.get(
        "orchestrator",
        {},
    )

    rows.append(
        [
            "orchestrator-api",
            orchestrator.get(
                "status",
                "unknown",
            ),
            orchestrator.get(
                "url",
                "",
            ),
            "-",
        ]
    )

    services = data.get(
        "services",
        {},
    )

    for service_name, info in services.items():

        rows.append(
            [
                service_name,
                info.get(
                    "status",
                    "unknown",
                ),
                info.get(
                    "url",
                    "",
                ),
                info.get(
                    "latency_ms",
                    "",
                ),
            ]
        )

    return (
        summary,
        rows,
    )


def load_recent_queries():

    try:

        with httpx.Client(
            timeout=6.0,
        ) as client:

            response = client.get(
                RECENT_QUERIES_URL,
                params={
                    "limit": 20,
                },
            )

    except httpx.TimeoutException:

        return (
            "### Activity\n\nQuery history timed out.",
            [],
        )

    except httpx.RequestError:

        return (
            "### Activity\n\nQuery data unavailable.",
            [],
        )

    if response.status_code != 200:

        return (
            (
                "### Activity\n\n"
                f"Failed to load "
                f"(`{response.status_code}`)."
            ),
            [],
        )

    try:

        data = response.json()

    except ValueError:

        return (
            "### Activity\n\nInvalid response.",
            [],
        )

    total = data.get(
        "total_recorded",
        0,
    )

    successful = data.get(
        "successful_queries",
        0,
    )

    average_latency = data.get(
        "average_latency_ms",
        0,
    )

    failed = max(
        total
        - successful,
        0,
    )

    summary = (
        "### Activity\n\n"
        f"**{total}** queries  \n"
        f"**{successful}** successful · "
        f"**{failed}** failed  \n"
        f"Avg latency: "
        f"`{average_latency} ms`"
    )

    rows: list[list[Any]] = []

    for item in data.get(
        "queries",
        [],
    ):

        rows.append(
            [
                item.get(
                    "timestamp",
                    "",
                ),
                item.get(
                    "query",
                    "",
                ),
                item.get(
                    "scope",
                    "",
                ),
                item.get(
                    "answer_type",
                    "",
                ),
                item.get(
                    "status",
                    "",
                ),
                item.get(
                    "latency_ms",
                    "",
                ),
                item.get(
                    "trace_id",
                    "",
                ),
            ]
        )

    return (
        summary,
        rows,
    )


def refresh_dashboard():

    health_summary, health_rows = (
        load_service_health()
    )

    query_summary, query_rows = (
        load_recent_queries()
    )

    return (
        health_summary,
        health_rows,
        query_summary,
        query_rows,
    )


# =============================================================================
# Navigation
# =============================================================================

def show_chat_view():

    return (
        gr.Column(
            visible=True
        ),
        gr.Column(
            visible=False
        ),
        gr.Column(
            visible=False
        ),
    )


def show_documents_view():

    return (
        gr.Column(
            visible=False
        ),
        gr.Column(
            visible=True
        ),
        gr.Column(
            visible=False
        ),
    )


def show_dashboard_view():

    return (
        gr.Column(
            visible=False
        ),
        gr.Column(
            visible=False
        ),
        gr.Column(
            visible=True
        ),
    )


# =============================================================================
# Gradio UI
# =============================================================================

def submit_question(user_question, history, session_id):
    """Keep the welcome screen and conversation visibility in sync."""
    history, question, evidence, metadata, session_id = query_ledger(
        user_question, history, session_id
    )
    return (
        gr.Chatbot(value=history, visible=bool(history)),
        question, evidence, metadata, session_id,
        gr.Group(visible=not bool(history)),
    )


def reset_workspace():
    """Start a fresh conversation and return to Chat from any view."""
    history, question, evidence, metadata, session_id = new_chat()
    return (
        gr.Chatbot(value=history, visible=False),
        question, evidence, metadata, session_id,
        gr.Group(visible=True),
        *show_chat_view(),
    )


def build_interface() -> gr.Blocks:
    with gr.Blocks(
        title="LEDGER — Financial Document Intelligence",
        fill_width=True,
    ) as demo:
        session_state = gr.State(value=_new_session_id)

        with gr.Sidebar(open=True, width=244, elem_id="ledger-sidebar"):
            gr.HTML("""
                <div class="brand-wrap">
                    <div class="brand-mark" aria-hidden="true">L</div>
                    <div>
                        <div class="brand-title">LEDGER</div>
                        <div class="brand-subtitle">Financial intelligence</div>
                    </div>
                </div>
            """)
            new_chat_button = gr.Button("+  New chat", elem_id="new-chat-btn")
            gr.HTML('<div class="nav-label">Workspace</div>')
            nav_chat = gr.Button("Chat", elem_id="nav-chat", elem_classes="ledger-nav-btn")
            nav_documents = gr.Button("Documents", elem_id="nav-documents", elem_classes="ledger-nav-btn")
            nav_dashboard = gr.Button("Dashboard", elem_id="nav-dashboard", elem_classes="ledger-nav-btn")
            gr.HTML("""
                <div class="sidebar-note">
                    <strong>Answers with evidence.</strong>
                    Explore your financial reports.<br>
                    Trace every answer to its source.
                    <div class="sidebar-version">LEDGER / V1.0</div>
                </div>
            """)

        gr.HTML("""
            <header id="workspace-header">
                <div>
                    <div class="workspace-title">Financial research workspace</div>
                    <div class="workspace-caption">From source documents to clear answers</div>
                </div>
                <div class="theme-control">
                    <span id="theme-label" class="theme-label">Light mode</span>
                    <button id="theme-toggle" type="button" role="switch"
                        aria-checked="false" aria-label="Dark mode"></button>
                </div>
            </header>
        """)

        with gr.Column(visible=True, elem_id="chat-view", elem_classes="ledger-view") as chat_view:
            gr.HTML("""
                <div class="view-heading">
                    <div><h1>Research chat</h1><p>Your questions. Grounded in your reports.</p></div>
                    <span class="page-badge">All reports</span>
                </div>
            """)
            with gr.Group(elem_id="welcome-panel") as welcome_panel:
                gr.HTML("""
                    <div class="welcome-area">
                        <div class="eyebrow">A clearer view of your financials</div>
                        <h2>Good questions.<br>Evidence-backed answers.</h2>
                        <p>Compare figures, find key metrics, and understand the numbers
                        across your financial reports. Start with a question below.</p>
                    </div>
                """)
                with gr.Row():
                    suggestion_1 = gr.Button("Compare finished goods", elem_classes="suggestion-btn")
                    suggestion_2 = gr.Button("Find operating income", elem_classes="suggestion-btn")
                    suggestion_3 = gr.Button("Calculate a change", elem_classes="suggestion-btn")

            chatbot = gr.Chatbot(
                value=[], visible=False,
                height="clamp(300px, 48vh, 560px)",
                label="Conversation", show_label=False,
                buttons=["copy"], layout="bubble",
                elem_id="ledger-chatbot",
            )
            with gr.Group(elem_classes="composer-shell"):
                with gr.Row(elem_id="composer-row"):
                    question_input = gr.Textbox(
                        placeholder="Ask a question about your financial reports…",
                        label="Your question", lines=1, max_lines=5,
                        show_label=False, container=False, scale=12, min_width=0,
                        elem_id="question-box",
                    )
                    ask_button = gr.Button("Send question", variant="primary", scale=0, min_width=40, elem_id="send-btn")
            gr.HTML("""
                <div class="composer-note">
                    <span>Verify important figures against the cited source documents.</span>
                    <span>Enter to send · Shift + Enter for a new line</span>
                </div>
            """)
            with gr.Row():
                with gr.Column(scale=3, min_width=230):
                    with gr.Accordion("Sources", open=False, elem_classes="ledger-details"):
                        evidence_output = gr.Markdown(value=EMPTY_EVIDENCE)
                with gr.Column(scale=2, min_width=230):
                    with gr.Accordion("Request details", open=False, elem_classes="ledger-details"):
                        metadata_output = gr.Markdown(value=EMPTY_METADATA)

        with gr.Column(visible=False, elem_id="documents-view", elem_classes="ledger-view") as documents_view:
            gr.HTML("""
                <div class="view-heading">
                    <div><h1>Documents</h1><p>The source behind every answer.</p></div>
                    <span class="page-badge">Report library</span>
                </div>
            """)

            with gr.Group(elem_id="document-upload-card"):
                gr.HTML("""
                    <div class="upload-card-header">
                        <div class="upload-card-icon" aria-hidden="true">
                            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor"
                                stroke-width="1.6" stroke-linecap="round"
                                stroke-linejoin="round">
                                <path d="M12 16V4"/>
                                <path d="m7 9 5-5 5 5"/>
                                <path d="M20 15v4a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2v-4"/>
                            </svg>
                        </div>
                        <div>
                            <div class="eyebrow upload-eyebrow">Document processing</div>
                            <h2>Upload a financial report</h2>
                            <p>
                                Send a raw PDF through LEDGER's Orchestrator to the
                                Document Processor and inspect the structured result.
                            </p>
                        </div>
                    </div>

                    <div class="upload-flow" aria-label="Processing flow">
                        <span>PDF</span>
                        <b>→</b>
                        <span>OCR + layout</span>
                        <b>→</b>
                        <span>Structured output</span>
                    </div>
                """)

                with gr.Row(elem_id="document-upload-fields"):
                    document_file = gr.File(
                        label="Financial report PDF",
                        file_types=[".pdf"],
                        file_count="single",
                        type="filepath",
                        elem_id="document-file",
                        scale=3,
                    )

                    document_id_input = gr.Textbox(
                        label="Document ID",
                        placeholder="Optional — e.g. annual_report_2024",
                        lines=1,
                        elem_id="document-id-input",
                        scale=2,
                    )

                upload_button = gr.Button(
                    "Upload & process",
                    variant="primary",
                    elem_id="upload-document-btn",
                )

                upload_status = gr.Markdown(
                    value=(
                        "### Upload status\n\n"
                        "_Choose a PDF to begin._"
                    ),
                    elem_id="upload-status",
                )

                with gr.Accordion(
                    "Processed representation",
                    open=False,
                    elem_classes="ledger-details processed-document-details",
                ):
                    processed_document_json = gr.JSON(
                        value=None,
                        label="Structured document output",
                        show_label=False,
                        elem_id="processed-document-json",
                    )

            gr.HTML("""
                <div class="empty-state">
                    <div class="empty-icon" aria-hidden="true">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor"
                            stroke-width="1.5" stroke-linejoin="round">
                            <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
                            <path d="M14 2v6h6M8 13h8M8 17h6"/>
                        </svg>
                    </div>
                    <h2>Your report library, in one place</h2>
                    <p>Document browsing is coming soon. For now, use Chat to ask
                    questions across indexed reports and see the sources behind each answer.</p>
                    <span class="page-badge">Coming soon</span>
                </div>
            """)

        with gr.Column(visible=False, elem_id="dashboard-view", elem_classes="ledger-view") as dashboard_view:
            gr.HTML("""
                <div class="view-heading">
                    <div><h1>Dashboard</h1><p>Service health and recent research activity.</p></div>
                    <span class="page-badge">System overview</span>
                </div>
            """)
            refresh_button = gr.Button("↻  Refresh data", elem_id="refresh-btn")
            with gr.Row():
                with gr.Column(elem_classes="dashboard-card"):
                    health_summary = gr.Markdown(
                        "### System status\n\nRefresh to check the current health of your services."
                    )
                with gr.Column(elem_classes="dashboard-card"):
                    query_summary = gr.Markdown(
                        "### Query activity\n\nRefresh to load recent questions and response times."
                    )
            gr.Markdown("### Services", elem_classes="section-title")
            health_table = gr.DataFrame(
                headers=["Service", "Status", "URL", "Latency (ms)"],
                value=[], interactive=False, label="Service health", show_label=False,
                elem_classes="ledger-table",
            )
            gr.Markdown("### Recent queries", elem_classes="section-title")
            recent_queries_table = gr.DataFrame(
                headers=["Timestamp", "Question", "Scope", "Answer Type", "Status", "Latency (ms)", "Trace ID"],
                value=[], interactive=False, label="Recent queries", show_label=False,
                elem_classes="ledger-table",
            )

        chat_inputs = [question_input, chatbot, session_state]
        chat_outputs = [chatbot, question_input, evidence_output, metadata_output, session_state, welcome_panel]
        for event in (ask_button.click, question_input.submit):
            event(
                fn=submit_question, inputs=chat_inputs, outputs=chat_outputs,
                concurrency_limit=1, concurrency_id="ledger-chat", show_progress="minimal",
            )

        for button, prompt in (
            (suggestion_1, suggestion_compare),
            (suggestion_2, suggestion_income),
            (suggestion_3, suggestion_percentage),
        ):
            button.click(fn=prompt, inputs=[], outputs=question_input, queue=False)

        view_outputs = [chat_view, documents_view, dashboard_view]
        new_chat_button.click(
            fn=reset_workspace, inputs=[], outputs=chat_outputs + view_outputs,
            concurrency_limit=1, concurrency_id="ledger-chat",
        )
        chatbot.clear(
            fn=reset_workspace, inputs=[], outputs=chat_outputs + view_outputs,
            concurrency_limit=1, concurrency_id="ledger-chat",
        )
        for button, navigate in (
            (nav_chat, show_chat_view),
            (nav_documents, show_documents_view),
            (nav_dashboard, show_dashboard_view),
        ):
            button.click(fn=navigate, inputs=[], outputs=view_outputs, queue=False)

        upload_button.click(
            fn=upload_document,
            inputs=[
                document_file,
                document_id_input,
            ],
            outputs=[
                upload_status,
                processed_document_json,
            ],
            concurrency_limit=1,
            concurrency_id="ledger-document-upload",
            show_progress="full",
        )

        refresh_button.click(
            fn=refresh_dashboard, inputs=[],
            outputs=[health_summary, health_table, query_summary, recent_queries_table],
        )

    return demo


# =============================================================================
# Entrypoint
# =============================================================================

if __name__ == "__main__":
    ui_port = int(os.getenv("UI_PORT", "8000"))

    app = build_interface()

    app.launch(
        server_name="0.0.0.0",
        server_port=ui_port,
        theme=gr.themes.Base(
            primary_hue="emerald",
            neutral_hue="slate",
        ),
        css=APP_CSS,
        js=APP_JS,
        footer_links=[],
        show_error=True,
    )