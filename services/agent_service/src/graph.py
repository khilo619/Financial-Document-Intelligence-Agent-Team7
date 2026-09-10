import json

import requests
from langchain_core.messages import ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from shared.config import (
    DEFAULT_LLM_MODEL,
    DEFAULT_LLM_TEMPERATURE,
    ServiceName,
    get_service_url,
)
from shared.models import StrictAnswer

from .llm import get_llm, invoke_with_retry
from .prompts import FINALIZE_PROMPT, REPAIR_PROMPT, SYSTEM_PROMPT
from .state import AgentState
from .tools import tools

ANSWER_VALIDATOR_URL = f"{get_service_url(ServiceName.ANSWER_VALIDATOR.value)}/validate_answer"

llm = get_llm()

# adding tools
llm_with_tools = llm.bind_tools(tools)
# -------------------------------------------------
# tool node
tool_node = ToolNode(tools)


# --------------------------------------------
# decomposition node
# -------------------------------------------
def decompose(state: AgentState):
    last_message = state["messages"][-1]

    return {
        "messages": [
            {
                "role": "assistant",
                "content": (
                    "The original question has been decomposed into "
                    "the following sub-questions:\n\n"
                    + "\n".join(f"{i}. {question}" for i, question in enumerate(last_message.content, start=1))
                ),
            }
        ]
    }


# --------------------------------------
# reason node
# -----------------------------------------
def reason(state: AgentState):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *state.get("messages", []),
    ]

    response = invoke_with_retry(llm_with_tools, messages)

    return {
        "messages": [response],
    }


# -----------------------------------------------------
# router after reason node to :
# finalize -end if direct query , else go to tools
def route_after_reason(state: AgentState):
    last_message = state["messages"][-1]
    tool_calls = getattr(last_message, "tool_calls", None)

    if tool_calls:
        return "tools"

    return "end"


# -----------------------------------------------
# route after tools
def route_after_tools(state: AgentState):
    last_message = state["messages"][-1]

    if isinstance(last_message, ToolMessage):
        if last_message.name == "decompose_question":
            return "decompose"

        elif last_message.name in [
            "search_documents",
            "search_tables",
            "calculate",
        ]:
            return "evidence"

    return "evidence"


# --------------------------------------------
# evidence nood
def collect_evidence(state: AgentState):
    evidence = state.get("evidence", [])

    tool_messages = []

    # Collect only the ToolMessages produced
    # in the immediately previous ToolNode execution
    for message in reversed(state["messages"]):
        if not isinstance(message, ToolMessage):
            break

        tool_messages.append(message)

    # Restore original execution order
    for message in reversed(tool_messages):
        content = message.content

        # Tool output may arrive as a JSON string
        if isinstance(content, str):
            try:
                content = json.loads(content)
            except json.JSONDecodeError:
                continue

        if isinstance(content, dict):
            results = content.get("results", [])

            if isinstance(results, list):
                evidence.extend(results)

    return {"evidence": evidence}


# ---------------------------------------------------
# finalize nood

# llm for structured output with strict Answer
structured_llm = llm.with_structured_output(StrictAnswer)


def finalize(state: AgentState):
    messages = [
        {"role": "system", "content": FINALIZE_PROMPT},
        *state.get("messages", []),
    ]

    answer = invoke_with_retry(structured_llm, messages)

    return {
        "answer": answer,
    }


# --------------------------------------------------------
# validate from answer_validator_api
def validate(state: AgentState):
    answer = state["answer"]

    response = requests.post(
        ANSWER_VALIDATOR_URL,
        json=answer.model_dump(),
        timeout=30,
    )

    response.raise_for_status()

    validation_result = response.json()

    return {"validation": validation_result}


# --------------------------------------------------------
# router after validate
def route_after_validate(state: AgentState):
    validation = state["validation"]

    if validation.get("is_valid"):
        return "end"

    return "repair"


# ------------------------------------------------------
# reapir if not valid
def repair(state: AgentState):
    answer = state["answer"]
    validation = state["validation"]

    error = validation.get("error", "Unknown validation error.")

    repair_prompt = REPAIR_PROMPT.format(
        answer=answer.model_dump_json(indent=2),
        error=error,
        evidence=state.get("evidence", []),
    )

    repaired_answer = invoke_with_retry(
        structured_llm,
        [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": repair_prompt,
            },
        ],
    )

    return {"answer": repaired_answer}


# -------------------------------------------------------
# building graph
# --------------------------------------------------------
builder = StateGraph(AgentState)
builder.add_node("decompose", decompose)

builder.add_node("reason", reason)
builder.add_edge(START, "reason")

builder.add_node("tools", tool_node)
builder.add_node("finalize", finalize)
builder.add_node("evidence", collect_evidence)
builder.add_node("validate", validate)
builder.add_node("repair", repair)

builder.add_conditional_edges(
    "reason",
    route_after_reason,
    {
        "tools": "tools",
        "end": "finalize",
    },
)
builder.add_edge("finalize", "validate")
builder.add_conditional_edges(
    "validate",
    route_after_validate,
    {
        "end": END,
        "repair": "repair",
    },
)
builder.add_conditional_edges(
    "tools",
    route_after_tools,
    {
        "decompose": "decompose",
        "evidence": "evidence",
    },
)
builder.add_edge("evidence", "reason")
builder.add_edge("decompose", "reason")
builder.add_edge("repair", "validate")


# =========================
# Compile
# =========================

graph = builder.compile()
