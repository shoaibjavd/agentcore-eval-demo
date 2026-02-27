# scripts/evaluate_agent.py
import boto3
import json
import os
import sys
import time

# All 14 built-in evaluators available in AgentCore
BUILTIN_EVALUATORS = {
    # Session-level evaluators
    "Builtin.GoalSuccessRate": {
        "level": "SESSION",
        "description": "Did the conversation achieve the user's goals?",
    },
    # Trace-level evaluators
    "Builtin.Helpfulness": {
        "level": "TRACE",
        "description": "How useful was the response from the user's perspective?",
    },
    "Builtin.Correctness": {
        "level": "TRACE",
        "description": "Is the response factually accurate?",
    },
    "Builtin.Coherence": {
        "level": "TRACE",
        "description": "Is the response logically consistent?",
    },
    "Builtin.Conciseness": {
        "level": "TRACE",
        "description": "Is the response efficiently communicated?",
    },
    "Builtin.ContextRelevance": {
        "level": "TRACE",
        "description": "Does the context contain the necessary information?",
    },
    "Builtin.Faithfulness": {
        "level": "TRACE",
        "description": "Is the response consistent with conversation history?",
    },
    "Builtin.Harmfulness": {
        "level": "TRACE",
        "description": "Does the response contain harmful content?",
    },
    "Builtin.InstructionFollowing": {
        "level": "TRACE",
        "description": "Does the response follow explicit instructions?",
    },
    "Builtin.Refusal": {
        "level": "TRACE",
        "description": "Did the agent refuse to answer when it shouldn't have?",
    },
    "Builtin.ResponseRelevance": {
        "level": "TRACE",
        "description": "Does the response address the user's question?",
    },
    "Builtin.Stereotyping": {
        "level": "TRACE",
        "description": "Does the response contain bias or stereotypes?",
    },
    # Tool-call-level evaluators
    "Builtin.ToolSelectionAccuracy": {
        "level": "TOOL_CALL",
        "description": "Was the correct tool selected for the task?",
    },
    "Builtin.ToolParameterAccuracy": {
        "level": "TOOL_CALL",
        "description": "Were the correct parameters passed to the tool?",
    },
}

# Default evaluators for CI — adjust based on your agent's priorities
DEFAULT_CI_EVALUATORS = [
    "Builtin.Helpfulness",
    "Builtin.Correctness",
    "Builtin.GoalSuccessRate",
    "Builtin.InstructionFollowing",
    "Builtin.ResponseRelevance",
]


def fetch_traces_from_cloudwatch(session_ids, region, agent_runtime_id=None):
    """
    Fetch OpenTelemetry spans AND log records from CloudWatch
    for the given session IDs.

    The Evaluate API requires OTel spans with gen_ai attributes.
    """
    logs_client = boto3.client("logs", region_name=region)

    # Determine the agent runtime log group
    if not agent_runtime_id:
        agent_runtime_id = os.environ.get("AGENT_RUNTIME_ID", "")

    # Discover available log groups for this runtime
    print("Discovering CloudWatch log groups...")
    available_groups = []
    try:
        paginator = logs_client.get_paginator("describe_log_groups")
        for prefix in ["aws/spans", "/aws/spans", "/aws/bedrock-agentcore"]:
            for page in paginator.paginate(logGroupNamePrefix=prefix):
                for group in page.get("logGroups", []):
                    name = group["logGroupName"]
                    available_groups.append(name)
                    print(f"  Found: {name}")
    except Exception as e:
        print(f"  Warning: Could not list log groups: {e}")

    if not available_groups:
        print("  No AgentCore or spans log groups found.")
        print("  Ensure CloudWatch Transaction Search is enabled in your account.")
        print("  See: https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/observability-get-started.html")

    # Build the list of log groups to query
    # Prefer discovered groups, fall back to expected names
    spans_group = None
    agent_group = None
    for g in available_groups:
        if "spans" in g and not spans_group:
            spans_group = g
        if agent_runtime_id and agent_runtime_id in g and not agent_group:
            agent_group = g

    log_groups = {}
    if spans_group:
        log_groups[spans_group] = "OTel spans"
    else:
        log_groups["aws/spans"] = "OTel spans (expected)"

    expected_agent_group = f"/aws/bedrock-agentcore/runtimes/{agent_runtime_id}-DEFAULT"
    if agent_group:
        log_groups[agent_group] = "OTel log records"
    else:
        log_groups[expected_agent_group] = "OTel log records (expected)"

    # Also try the runtime-logs subgroup which contains OTEL structured logs
    runtime_logs_group = f"/aws/bedrock-agentcore/runtimes/{agent_runtime_id}-DEFAULT/runtime-logs"
    for g in available_groups:
        if "runtime-logs" in g:
            log_groups[g] = "Runtime structured logs"
            break

    spans_by_session = {}

    for session_id in session_ids:
        print(f"\nFetching traces for session: {session_id}")
        session_spans = []

        for log_group, label in log_groups.items():
            print(f"  Querying {label} from {log_group}")
            query = f"""
                fields @timestamp, @message
                | filter @message like /"{session_id}"/
                | sort @timestamp asc
                | limit 200
            """

            try:
                response = logs_client.start_query(
                    logGroupName=log_group,
                    startTime=int((time.time() - 7200) * 1000),  # Look back 2 hours
                    endTime=int(time.time() * 1000),
                    queryString=query,
                )
            except logs_client.exceptions.ResourceNotFoundException:
                print(f"  Log group {log_group} not found — skipping.")
                continue
            except Exception as e:
                print(f"  Error querying {log_group}: {e}")
                continue

            query_id = response["queryId"]

            # Poll for results — CloudWatch queries are async
            for _ in range(30):  # Up to 30 seconds
                result = logs_client.get_query_results(queryId=query_id)
                if result["status"] == "Complete":
                    break
                time.sleep(1)

            count = 0
            for row in result.get("results", []):
                for field in row:
                    if field["field"] == "@message":
                        try:
                            span = json.loads(field["value"])
                            session_spans.append(span)
                            count += 1
                        except json.JSONDecodeError:
                            continue
            print(f"  Found {count} records.")

        if session_spans:
            spans_by_session[session_id] = session_spans

    total = sum(len(s) for s in spans_by_session.values())
    print(f"\nCollected {total} total records across {len(spans_by_session)} sessions (of {len(session_ids)} requested).")
    return spans_by_session


def run_evaluation(spans_by_session, evaluator_ids, region):
    """
    Run on-demand evaluation using the AgentCore Evaluate API.
    Evaluates each session separately (API requires single-session input),
    then averages scores across sessions.
    Returns a dict of evaluator_id -> {score, explanation}.
    """
    client = boto3.client("bedrock-agentcore", region_name=region)
    # Collect per-evaluator scores across all sessions
    evaluator_scores = {eid: [] for eid in evaluator_ids}

    for session_id, session_spans in spans_by_session.items():
        print(f"\n--- Evaluating session: {session_id} ({len(session_spans)} spans) ---")

        for evaluator_id in evaluator_ids:
            info = BUILTIN_EVALUATORS.get(evaluator_id, {})
            print(f"  {evaluator_id}: ", end="")

            try:
                response = client.evaluate(
                    evaluatorId=evaluator_id,
                    evaluationInput={"sessionSpans": session_spans},
                )

                eval_results = response.get("evaluationResults", [])
                if eval_results:
                    result = eval_results[0]
                    score = result.get("value", 0) or 0
                    error_msg = result.get("errorMessage", "")

                    if error_msg:
                        print(f"error — {error_msg[:100]}")
                    else:
                        evaluator_scores[evaluator_id].append(score)
                        print(f"{score:.2f}")
                else:
                    print("no results")

            except Exception as e:
                print(f"error — {e}")

    # Average scores across sessions
    results = {}
    for evaluator_id in evaluator_ids:
        scores = evaluator_scores[evaluator_id]
        if scores:
            avg = sum(scores) / len(scores)
            results[evaluator_id] = {
                "score": avg,
                "explanation": f"Average of {len(scores)} session(s)",
                "session_scores": scores,
            }
        else:
            results[evaluator_id] = {
                "score": 0,
                "explanation": "No successful evaluations",
            }

    return results


def enforce_quality_gate(results, threshold):
    """Check if all evaluation scores meet the threshold."""
    print(f"\n{'='*60}")
    print(f"QUALITY GATE — Threshold: {threshold}")
    print(f"{'='*60}")

    all_passed = True
    for evaluator_id, result in results.items():
        score = result["score"]
        status = "PASS" if score >= threshold else "FAIL"
        if status == "FAIL":
            all_passed = False
        print(f"  [{status}] {evaluator_id}: {score:.2f}")

    print(f"{'='*60}")

    if all_passed:
        print("✅ Quality gate PASSED. All evaluators meet the threshold.")
    else:
        print("❌ Quality gate FAILED. One or more evaluators below threshold.")

    return all_passed


def main():
    region = os.environ.get("AWS_REGION", "us-east-1")
    session_ids_raw = os.environ.get("SESSION_IDS", "[]")
    threshold = float(os.environ.get("EVAL_THRESHOLD", "0.7"))
    evaluator_ids = os.environ.get(
        "EVALUATOR_IDS", ",".join(DEFAULT_CI_EVALUATORS)
    ).split(",")
    agent_runtime_id = os.environ.get("AGENT_RUNTIME_ID", "")

    # Parse session IDs — handle both JSON array and comma-separated formats
    try:
        session_ids = json.loads(session_ids_raw)
    except json.JSONDecodeError:
        session_ids = [s.strip() for s in session_ids_raw.split(",") if s.strip()]

    if not session_ids:
        print("ERROR: No session IDs provided. Nothing to evaluate.")
        with open("evaluation_results.json", "w") as f:
            json.dump({"error": "No session IDs provided"}, f)
        sys.exit(1)

    print(f"Session IDs to evaluate: {session_ids}")
    print(f"Evaluators: {evaluator_ids}")

    # Fetch traces with retry — CloudWatch log ingestion can lag
    spans_by_session = {}
    for attempt in range(3):
        spans_by_session = fetch_traces_from_cloudwatch(session_ids, region, agent_runtime_id)
        if len(spans_by_session) >= len(session_ids):
            break
        if attempt < 2:
            wait = 30 * (attempt + 1)
            print(f"\nOnly found {len(spans_by_session)}/{len(session_ids)} sessions. Waiting {wait}s for more traces...")
            time.sleep(wait)

    if not spans_by_session:
        print("WARNING: No traces found in CloudWatch after retries.")
        print("Possible causes: Transaction Search not enabled, strands-agents[otel] not installed,")
        print("or aws-opentelemetry-distro not in requirements.txt.")
        diagnostic = {
            "error": "No traces found in CloudWatch",
            "session_ids": session_ids,
            "agent_runtime_id": agent_runtime_id,
            "hint": "Enable CloudWatch Transaction Search and ensure strands-agents[otel] is installed",
        }
        with open("evaluation_results.json", "w") as f:
            json.dump(diagnostic, f, indent=2)
        sys.exit(0)

    # Run evaluations (per-session, then averaged)
    results = run_evaluation(spans_by_session, evaluator_ids, region)

    # Write results to JSON for GitHub Actions artifact upload and PR comment
    with open("evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Enforce quality gate
    passed = enforce_quality_gate(results, threshold)

    if not passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
