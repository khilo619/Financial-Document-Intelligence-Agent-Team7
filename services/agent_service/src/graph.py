import json
import logging

import requests
from langchain_core.messages import ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from shared.config import (
    ServiceName,
    get_service_url,
)
from shared.models import StrictAnswer

from .llm import get_llm, invoke_with_retry
from .prompts import FINALIZE_PROMPT, REPAIR_PROMPT, SYSTEM_PROMPT
from .state import AgentState
from .tools import tools

logger = logging.getLogger("AgentService.Graph")
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
    content = getattr(last_message, "content", "")
    if isinstance(content, list):
        sub_questions = content
    elif isinstance(content, str):
        try:
            parsed = json.loads(content)
            sub_questions = parsed if isinstance(parsed, list) else [content]
        except Exception:
            sub_questions = [content]
    else:
        sub_questions = [str(content)]

    # Must be a user role so Gemini alternating turn validation succeeds
    return {
        "messages": [
            {
                "role": "user",
                "content": (
                    "The original question has been decomposed into "
                    "the following sub-questions:\n\n"
                    + "\n".join(f"{i}. {q}" for i, q in enumerate(sub_questions, start=1))
                    + "\n\nPlease proceed to investigate these sub-questions using available tools."
                ),
            }
        ]
    }


# --------------------------------------
# reason node
# -----------------------------------------
def reason(state: AgentState):
    raw_messages = list(state.get("messages", []))

    # Ensure Gemini does not receive a request ending with a model turn
    if raw_messages:
        last = raw_messages[-1]
        last_role = getattr(last, "type", None) or (last.get("role") if isinstance(last, dict) else None)
        if last_role in ("ai", "assistant", "model"):
            raw_messages.append(
                {
                    "role": "user",
                    "content": "Please proceed with reasoning based on the above information.",
                }
            )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *raw_messages,
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
    evidence = list(state.get("evidence", []))
    seen_blocks = {(e.get("document_id"), e.get("page"), e.get("chunk_id")) for e in evidence if isinstance(e, dict)}

    tool_messages = []

    # Collect only the ToolMessages produced
    # in the immediately previous ToolNode execution
    for message in reversed(state.get("messages", [])):
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
                for item in results:
                    if isinstance(item, dict):
                        key = (item.get("document_id"), item.get("page"), item.get("chunk_id"))
                        if key not in seen_blocks:
                            seen_blocks.add(key)
                            evidence.append(item)

    return {"evidence": evidence}


# ---------------------------------------------------
# finalize nood

# llm for structured output with strict Answer
structured_llm = llm.with_structured_output(StrictAnswer)


def finalize(state: AgentState):
    raw_messages = list(state.get("messages", []))

    # Always append a final user message to guarantee Gemini ends with a user turn
    # and strictly generates the StrictAnswer schema
    messages = [
        {"role": "system", "content": FINALIZE_PROMPT},
        *raw_messages,
        {
            "role": "user",
            "content": (
                "Produce the final structured answer strictly adhering to the StrictAnswer schema "
                "based on the conversation and retrieved evidence above."
            ),
        },
    ]

    try:
        answer = invoke_with_retry(structured_llm, messages)
    except Exception as exc:
        logger.error("Structured LLM invocation failed in finalize: %s", exc)
        answer = None

    if answer is None:
        answer = StrictAnswer(
            answer_type="insufficient_evidence",
            params={"reason": "Unable to produce a valid structured answer from evidence."},
            evidence=[],
        )

    return {
        "answer": answer,
    }


# --------------------------------------------------------
# validate from answer_validator_api
def validate(state: AgentState):
    answer = state.get("answer")
    if hasattr(answer, "model_dump"):
        answer_payload = answer.model_dump()
    elif isinstance(answer, dict):
        answer_payload = answer
    else:
        answer_payload = {
            "answer_type": "insufficient_evidence",
            "params": {"reason": "Missing candidate answer"},
            "evidence": [],
        }

    # ValidationRequest expects payload wrapped in {"answer": ...}
    response = requests.post(
        ANSWER_VALIDATOR_URL,
        json={"answer": answer_payload},
        timeout=30,
    )

    response.raise_for_status()

    validation_result = response.json()

    return {"validation": validation_result}


# --------------------------------------------------------
# router after validate
def route_after_validate(state: AgentState):
    validation = state.get("validation", {})

    if validation.get("is_valid"):
        return "end"

    # Stop after max repair attempts (2) to prevent infinite loops
    attempts = state.get("repair_attempts", 0)
    if attempts >= 2:
        return "end"

    return "repair"


# ------------------------------------------------------
# repair if not valid
def repair(state: AgentState):
    answer = state.get("answer")
    validation = state.get("validation", {})
    attempts = state.get("repair_attempts", 0) + 1

    error = validation.get("error", "Unknown validation error.")
    answer_str = (
        answer.model_dump_json(indent=2) if hasattr(answer, "model_dump_json") else json.dumps(answer, indent=2)
    )

    repair_prompt = REPAIR_PROMPT.format(
        answer=answer_str,
        error=error,
        evidence=state.get("evidence", []),
    )

    try:
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
    except Exception as exc:
        logger.error("Structured LLM invocation failed in repair: %s", exc)
        repaired_answer = None

    if repaired_answer is None:
        repaired_answer = answer

    return {"answer": repaired_answer, "repair_attempts": attempts}


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
