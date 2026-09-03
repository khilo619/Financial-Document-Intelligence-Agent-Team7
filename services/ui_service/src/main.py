"""
ui_service: Gradio Web Application for Project LEDGER.
Owned by Ahmed (Member 6) - Initial scaffold by Khaled (Repo Lead).
"""

import logging
import os

import gradio as gr
import httpx

from shared.config import ServiceName, get_service_url

logging.basicConfig(
    level=logging.INFO, format="[%(asctime)s] [%(levelname)s]: %(message)s"
)
logger = logging.getLogger("UIService")

ORCHESTRATOR_URL = f"{get_service_url(ServiceName.ORCHESTRATOR.value)}/ask"


def query_ledger(user_question: str, history: list):
    """
    Submits query to orchestrator-api and formats answer + evidence.
    """
    if not user_question.strip():
        return history, "", "Please enter a valid financial question."

    logger.info("UI forwarding user query: '%s'", user_question)
    try:
        with httpx.Client(timeout=10.0) as client:
            resp = client.post(ORCHESTRATOR_URL, json={"query": user_question})
            if resp.status_code == 200:
                data = resp.json()
                answer_data = data.get("answer", {})
                a_type = answer_data.get("answer_type", "unknown")
                params = answer_data.get("params", {})
                evidence = answer_data.get("evidence", [])
                latency = data.get("latency_ms", 0.0)

                # Format user-friendly response
                if a_type == "calculated":
                    bot_text = f"**Answer ({a_type}):** {params.get('value')}\n\n*Formula:* `{params.get('formula')}`"
                elif a_type == "direct":
                    bot_text = f"**Answer ({a_type}):** {params.get('value')}"
                elif a_type == "multi_span":
                    items = ", ".join(str(v) for v in params.get("values", []))
                    bot_text = f"**Answer ({a_type}):** {items}"
                else:
                    bot_text = f"**Insufficient Evidence:** {params.get('reason')}"

                bot_text += f"\n\n*(Latency: {latency} ms)*"

                # Evidence formatting
                evidence_text = (
                    f"### 📑 Cited Grounding Evidence ({len(evidence)} citations):\n"
                )
                for idx, ev in enumerate(evidence, 1):
                    evidence_text += (
                        f"{idx}. **Document:** `{ev.get('document_id')}` | "
                        f"**Page:** `{ev.get('page')}` | "
                        f"**Section:** `{ev.get('section', 'N/A')}`\n"
                    )

                history.append((user_question, bot_text))
                return history, "", evidence_text

            return (
                history,
                "",
                f"Error from Orchestrator: {resp.status_code} - {resp.text}",
            )

    except (httpx.RequestError, httpx.HTTPStatusError) as exc:
        logger.warning(
            "Connection error to orchestrator (%s). Falling back to demo mode.", exc
        )
        # Mock fallback for UI testing if orchestrator container is not running yet
        mock_response = (
            f"**[Demo / Standalone Mode]**\n\n"
            f"**Query:** {user_question}\n"
            f"**Answer (calculated):** 304,811 thousand\n\n"
            f"*Formula:* `abs(9447 - 314258)`"
        )
        mock_evidence = (
            "### 📑 Cited Grounding Evidence (Sample):\n"
            "1. **Document:** `cts-corporation_2019.pdf` | **Page:** `1` | **Section:** `Note 4: Inventories`\n"
            "2. **Document:** `jabil-circuit-inc_2019.pdf` | **Page:** `1` | **Section:** `Inventories`"
        )
        history.append((user_question, mock_response))
        return history, "", mock_evidence
    except Exception as exc:  # noqa: BLE001
        logger.error("Unexpected error in UI handler: %s", exc)
        return history, "", f"Unexpected error: {exc}"


def build_interface():
    """Builds the Gradio tabbed interface."""
    with gr.Blocks(title="Project LEDGER - Financial Document Intelligence") as demo:
        gr.Markdown(
            "# 📊 Project LEDGER: Financial Document Intelligence Agent\n"
            "*MIA Robotics AI Team 7 — Discrete Reasoning & Verified Evidence Grounding*"
        )

        with gr.Tabs():
            with gr.TabItem("💬 Corpus-Wide Chat"):
                chatbot = gr.Chatbot(height=450, label="Conversation")
                with gr.Row():
                    msg_input = gr.Textbox(
                        placeholder="Ask any question across all indexed financial reports (e.g., 'What was the difference in Finished Goods between CTS and Jabil in 2019?')...",
                        scale=9,
                        label="Your Question",
                    )
                    submit_btn = gr.Button("Submit Query", variant="primary", scale=2)

                evidence_box = gr.Markdown(
                    value="*Evidence citations backing the answer will appear here.*"
                )

                submit_btn.click(
                    query_ledger,
                    inputs=[msg_input, chatbot],
                    outputs=[chatbot, msg_input, evidence_box],
                )
                msg_input.submit(
                    query_ledger,
                    inputs=[msg_input, chatbot],
                    outputs=[chatbot, msg_input, evidence_box],
                )

            with gr.TabItem("📈 System Health & Corpus Dashboard"):
                gr.Markdown("### 🏛️ Microservice Topology & Port Registry")
                gr.DataFrame(
                    headers=["Service Name", "Owner", "Port", "Role"],
                    value=[
                        [
                            "ui-service",
                            "Ahmed (Member 6)",
                            "8000",
                            "Gradio UI & Dashboard",
                        ],
                        [
                            "orchestrator-api",
                            "Ahmed (Member 6)",
                            "8001",
                            "Central API Gateway",
                        ],
                        [
                            "doc-processor-api",
                            "Zeina (Member 1)",
                            "8002",
                            "Layout OCR & Table Extraction",
                        ],
                        [
                            "retrieval-api",
                            "Salma (Member 2)",
                            "8003",
                            "Hybrid Qdrant + BM25 + Reranker",
                        ],
                        [
                            "agent-service",
                            "Youssef (Member 3)",
                            "8004",
                            "LangGraph Reasoning Brain",
                        ],
                        [
                            "answer-validator-api",
                            "Omar (Member 4)",
                            "8005",
                            "Strict Schema Gatekeeper",
                        ],
                        [
                            "eval-service",
                            "Khaled (Member 5)",
                            "8006",
                            "Automated Benchmark & MLOps",
                        ],
                        ["qdrant", "Infrastructure", "6333", "Vector Database"],
                        [
                            "langfuse",
                            "Infrastructure",
                            "3000",
                            "Observability Platform",
                        ],
                    ],
                    interactive=False,
                )
                gr.Markdown(
                    "### 📁 Indexed Benchmark Corpus: `TAT-DQA` (questions_setA_practice.json)"
                )

    return demo


if __name__ == "__main__":
    port = int(os.getenv("UI_PORT", "8000"))
    demo_app = build_interface()
    demo_app.launch(server_name="0.0.0.0", server_port=port)
