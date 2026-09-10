import ast
import json
import os

SERVICE_MAP = {
    "doc-processor-api": "services/doc_processor_api/Dockerfile",
    "retrieval-api": "services/retrieval_api/Dockerfile",
    "agent-service": "services/agent_service/Dockerfile",
    "answer-validator-api": "services/answer_validator_api/Dockerfile",
    "orchestrator-api": "services/orchestrator_api/Dockerfile",
    "eval-service": "services/eval_service/Dockerfile",
    "ui-service": "services/ui_service/Dockerfile",
}


def main():
    event_name = os.environ.get("EVENT_NAME", "")
    raw_changes = os.environ.get("RAW_CHANGES", "").strip()

    print(f"Event: {event_name}")
    print(f"Raw changes detected: {raw_changes}")

    if event_name == "workflow_dispatch":
        targets = list(SERVICE_MAP.keys())
    else:
        targets = []
        if raw_changes:
            # Try 1: Standard JSON
            try:
                targets = json.loads(raw_changes)
            except Exception:
                pass

            # Try 2: Python AST literal
            if not targets:
                try:
                    targets = ast.literal_eval(raw_changes)
                except Exception:
                    pass

            # Try 3: Direct bracket/comma splitting
            if not targets:
                cleaned = raw_changes.strip("[] \t\n\r\"'")
                targets = [s.strip(" \"'") for s in cleaned.split(",") if s.strip(" \"'")]

    print(f"Target services to build: {targets}")
    matrix_include = [{"name": s, "dockerfile": SERVICE_MAP[s]} for s in targets if s in SERVICE_MAP]
    has_changes = "true" if matrix_include else "false"

    output_matrix = (
        {"include": matrix_include}
        if matrix_include
        else {"include": [{"name": "none", "dockerfile": "none"}]}
    )
    print(f"has_changes: {has_changes}")
    print(f"Matrix: {json.dumps(output_matrix)}")

    output_file = os.environ.get("GITHUB_OUTPUT")
    if output_file:
        with open(output_file, "a") as f:
            f.write(f"has_changes={has_changes}\n")
            f.write(f"matrix={json.dumps(output_matrix)}\n")


if __name__ == "__main__":
    main()
