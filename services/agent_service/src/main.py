"""
agent_service: LangGraph Cyclic Discrete Reasoning Brain microservice.
Owned by Youssef (Member 3) - Initial scaffold by Khaled (Repo Lead).
"""

import logging

from fastapi import FastAPI

from shared.config import ServiceName
from shared.models import AskRequest, Citation, StrictAnswer

logging.basicConfig(
    level=logging.INFO, format="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s"
)
logger = logging.getLogger("AgentService")

app = FastAPI(
    title="Project LEDGER - Agent Reasoning Service",
    description="LangGraph state machine for multi-hop retrieval and deterministic arithmetic.",
    version="0.1.0",
)


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": ServiceName.AGENT.value,
        "port": 8004,
    }


@app.post("/solve", response_model=StrictAnswer)
def solve_question(request: AskRequest):
    """
    Formulates multi-hop search queries, reasons over retrieved evidence,
    calls calculator tools, and outputs a validated StrictAnswer.
    """
    logger.info("Agent received reasoning request for: '%s'", request.query)

    # Day 1 Scaffold: Simulate LangGraph reasoning graph execution
    return StrictAnswer(
        answer_type="calculated",
        evidence=[
            Citation(
                document_id="cts-corporation_2019.pdf", page=1, section="Finished Goods"
            ),
            Citation(
                document_id="jabil-circuit-inc_2019.pdf",
                page=1,
                section="Finished Goods",
            ),
        ],
        params={
            "value": 304811.0,
            "formula": "abs(9447 - 314258)",
        },
    )
