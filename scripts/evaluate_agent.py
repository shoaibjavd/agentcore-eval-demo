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
    "Builtin.ToolSelectionAccuracy",
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
    agent_log_group = f"/aws/bedrock-agentcore/runtimes/{agent_runtime_id}-DEFAULT"

    # Two log groups to query: spans (trace metadata) and agent logs (conversation content)
    log_groups = {
        "/aws/spans": "OTel spans",
        agent_log_group: "OTel log records",
    }

    all_spans = []

    for session_id in session_ids:
        print(f"Fetching traces for session: {session_id}")

        for log_group, label in log_groups.items():
            print(f"  Querying {label} from {log_group}")
            query = f"""
                fields @timestamp, @message
                | filter @message like /"{session_id}"/
                | sort @timestamp asc
                | limit 100
            """

            try:
                response = logs_client.start_query(
                    logGroupName=log_group,
                    startTime=int((time.time() - 3600) * 1000),  # Look back 1 hour
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
                            all_spans.append(span)
                            count += 1
                        except json.JSONDecodeError:
                            continue
            print(f"  Found {count} records.")

    print(f"Collected {len(all_spans)} total records across {len(session_ids)} sessions.")
    return all_spans


def run_evaluation(spans, evaluator_ids, region):
    """
    Run on-demand evaluation using the AgentCore Evaluate API.
    Returns a dict of evaluator_id -> {score, explanation}.
    """
    client = boto3.client("bedrock-agentcore", region_name=region)
    results = {}

    for evaluator_id in evaluator_ids:
        info = BUILTIN_EVALUATORS.get(evaluator_id, {})
        print(f"\nEvaluating with: {evaluator_id}")
        print(f"  ({info.get('description', 'Custom evaluator')})")

        try:
            response = client.evaluate(
                evaluatorId=evaluator_id,
                evaluationInput={"sessionSpans": spans},
            )

            eval_results = response.get("evaluationResults", [])
            if eval_results:
                # Each result has: evaluatorId, value (double), explanation, label, etc.
                result = eval_results[0]
                score = result.get("value", 0) or 0
                explanation = result.get("explanation", "No explanation")
                error_msg = result.get("errorMessage", "")

                if error_msg:
                    print(f"  Evaluator error: {error_msg}")
                    results[evaluator_id] = {
                        "score": 0,
                        "explanation": error_msg,
                    }
                else:
                    results[evaluator_id] = {
                        "score": score,
                        "explanation": explanation,
                    }
                    print(f"  Score: {score:.2f}")
                    print(f"  Explanation: {explanation[:150]}...")
            else:
                print(f"  No results returned.")
                results[evaluator_id] = {"score": 0, "explanation": "No results"}

        except Exception as e:
            print(f"  ERROR: {e}")
            results[evaluator_id] = {"score": 0, "explanation": str(e)}

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
        # Write empty results so the PR comment step doesn't fail
        with open("evaluation_results.json", "w") as f:
            json.dump({"error": "No session IDs provided"}, f)
        sys.exit(1)

    print(f"Session IDs to evaluate: {session_ids}")
    print(f"Evaluators: {evaluator_ids}")

    # Fetch traces (spans + log records from CloudWatch)
    spans = fetch_traces_from_cloudwatch(session_ids, region, agent_runtime_id)
    if not spans:
        print("ERROR: No traces found. Check CloudWatch Transaction Search.")
        with open("evaluation_results.json", "w") as f:
            json.dump({"error": "No traces found in CloudWatch"}, f)
        sys.exit(1)

    # Run evaluations
    results = run_evaluation(spans, evaluator_ids, region)

    # Write results to JSON for GitHub Actions artifact upload and PR comment
    with open("evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Enforce quality gate
    passed = enforce_quality_gate(results, threshold)

    if not passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
