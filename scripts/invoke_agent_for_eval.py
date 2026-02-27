# scripts/invoke_agent_for_eval.py
import boto3
import json
import os
import sys
import time
import uuid


def get_cognito_token():
    """Get a JWT token from Cognito for the CI test user (optional)."""
    client_id = os.environ.get("COGNITO_CLIENT_ID")
    username = os.environ.get("CI_TEST_USERNAME")
    password = os.environ.get("CI_TEST_PASSWORD")

    if not all([client_id, username, password]):
        print("Cognito credentials not fully configured — skipping JWT auth.")
        print("Falling back to IAM SigV4 authentication.")
        return None

    try:
        cognito_client = boto3.client("cognito-idp")
        response = cognito_client.initiate_auth(
            AuthFlow="USER_PASSWORD_AUTH",
            ClientId=client_id,
            AuthParameters={
                "USERNAME": username,
                "PASSWORD": password,
            },
        )
        print("Successfully obtained Cognito JWT token.")
        return response["AuthenticationResult"]["IdToken"]
    except Exception as e:
        print(f"Warning: Cognito auth failed ({e}). Falling back to IAM SigV4.")
        return None


def invoke_agent(agent_runtime_arn, prompt, session_id):
    """Invoke the agent with a test prompt using IAM SigV4 auth."""
    client = boto3.client("bedrock-agentcore")

    payload = json.dumps({"prompt": prompt}).encode("utf-8")

    response = client.invoke_agent_runtime(
        agentRuntimeArn=agent_runtime_arn,
        runtimeSessionId=session_id,
        payload=payload,
    )

    # Handle the streaming response based on content type
    content_type = response.get("contentType", "")
    result_parts = []

    if "text/event-stream" in content_type:
        # Server-Sent Events streaming response
        for line in response["response"].iter_lines(chunk_size=10):
            if line:
                decoded = line.decode("utf-8")
                if decoded.startswith("data: "):
                    result_parts.append(decoded[6:])
                else:
                    result_parts.append(decoded)
    elif content_type == "application/json":
        # Standard JSON response — read chunks
        for chunk in response.get("response", []):
            if isinstance(chunk, bytes):
                result_parts.append(chunk.decode("utf-8"))
            else:
                result_parts.append(str(chunk))
    else:
        # Fallback: try reading as a stream
        stream = response.get("response", b"")
        if hasattr(stream, "read"):
            result_parts.append(stream.read().decode("utf-8"))
        elif hasattr(stream, "iter_lines"):
            for line in stream.iter_lines(chunk_size=10):
                if line:
                    result_parts.append(line.decode("utf-8"))
        else:
            result_parts.append(str(stream))

    return "\n".join(result_parts)


def run_test_sessions():
    agent_runtime_arn = os.environ.get("AGENT_RUNTIME_ARN", "")
    if not agent_runtime_arn:
        print("ERROR: AGENT_RUNTIME_ARN not set.")
        sys.exit(1)

    # Define test prompts — customize these for your agent
    test_cases = [
        {
            "prompt": "What are three best practices for writing clean Python code?",
            "description": "Knowledge query — tests helpfulness and correctness",
        },
        {
            "prompt": "Explain the difference between a list and a tuple in Python.",
            "description": "Explanation — tests coherence and response relevance",
        },
        {
            "prompt": "Write a short greeting message for a new team member named Alex.",
            "description": "Content generation — tests instruction following",
        },
    ]

    # Cognito JWT is optional — IAM SigV4 is the primary auth mechanism
    token = get_cognito_token()
    session_ids = []

    for i, test_case in enumerate(test_cases):
        # runtimeSessionId must be 33-256 characters
        session_id = f"ci-eval-{uuid.uuid4().hex}"  # 41 chars total
        print(f"\n[{i+1}/{len(test_cases)}] {test_case['description']}")
        print(f"  Session: {session_id}")
        print(f"  Prompt:  {test_case['prompt']}")

        try:
            result = invoke_agent(agent_runtime_arn, test_case["prompt"], session_id)
            preview = result[:200] if result else "(empty response)"
            print(f"  Response: {preview}...")
            session_ids.append(session_id)
        except Exception as e:
            print(f"  ERROR: {e}")
            # Continue with other test cases — don't fail the whole run
            continue

        # Brief pause between invocations to allow trace propagation
        time.sleep(2)

    if not session_ids:
        print("\nERROR: All test invocations failed. No sessions to evaluate.")
        sys.exit(1)

    # Allow traces to flush to CloudWatch
    print("\nWaiting 90s for traces to propagate to CloudWatch...")
    time.sleep(90)

    print(f"\nCompleted {len(session_ids)}/{len(test_cases)} test sessions.")
    return session_ids


if __name__ == "__main__":
    session_ids = run_test_sessions()

    # Write session IDs to GitHub Actions output
    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a") as f:
            f.write(f"session_ids={json.dumps(session_ids)}\n")
