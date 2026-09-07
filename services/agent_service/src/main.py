"""
agent_service: LangGraph Cyclic Discrete Reasoning Brain microservice.
Owned by Youssef (Member 3) - Initial scaffold by Khaled (Repo Lead).
"""

import logging

from fastapi import FastAPI
from langchain_core.messages import HumanMessage

from shared.config import ServiceName
from shared.models import AskRequest, StrictAnswer

from .graph import graph


logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
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
    Run the LangGraph reasoning workflow and return the final StrictAnswer.
    """

    logger.info(
        "Agent received reasoning request for: '%s'",
        request.query,
    )

    initial_state = {
        "query": request.query,
        "messages": [
            HumanMessage(content=request.query)
        ],
        "evidence": [],
    }

    final_state = graph.invoke(initial_state)

    return final_state["answer"]