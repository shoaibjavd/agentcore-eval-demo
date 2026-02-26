# scripts/invoke_agent_for_eval.py
import boto3
import json
import os
import time
import uuid

def get_cognito_token():
    """Get a JWT token from Cognito for the CI test user."""
    cognito_client = boto3.client("cognito-idp")

    response = cognito_client.initiate_auth(
        AuthFlow="USER_PASSWORD_AUTH",
        ClientId=os.environ["COGNITO_CLIENT_ID"],
        AuthParameters={
            "USERNAME": os.environ["CI_TEST_USERNAME"],
            "PASSWORD": os.environ["CI_TEST_PASSWORD"],
        },
    )
    return response["AuthenticationResult"]["IdToken"]


def invoke_agent(agent_runtime_arn, prompt, session_id, token):
    """Invoke the agent with a test prompt."""
    client = boto3.client("bedrock-agentcore")

    payload = json.dumps({"prompt": prompt}).encode("utf-8")

    response = client.invoke_agent_runtime(
        agentRuntimeArn=agent_runtime_arn,
        runtimeSessionId=session_id,
        payload=payload,
    )

    # Read the response stream
    result = ""
    event_stream = response.get("response", b"")
    if hasattr(event_stream, "read"):
        result = event_stream.read().decode("utf-8")
    else:
        result = str(event_stream)

    return result


def run_test_sessions():
    agent_runtime_arn = os.environ["AGENT_RUNTIME_ARN"]

    # Define test prompts — customize these for your agent
    test_cases = [
        {
            "prompt": "What meetings do I have tomorrow?",
            "description": "Calendar query — tests MCP tool selection",
        },
        {
            "prompt": "Summarize the key action items from my last team standup.",
            "description": "Multi-step reasoning — tests coherence and helpfulness",
        },
        {
            "prompt": "Draft a follow-up email to the client about the Q3 deliverables.",
            "description": "Content generation — tests instruction following",
        },
    ]

    token = get_cognito_token()
    session_ids = []

    for i, test_case in enumerate(test_cases):
        session_id = f"ci-eval-{uuid.uuid4().hex[:8]}"
        print(f"\n[{i+1}/{len(test_cases)}] {test_case['description']}")
        print(f"  Session: {session_id}")
        print(f"  Prompt:  {test_case['prompt']}")

        try:
            result = invoke_agent(
                agent_runtime_arn, test_case["prompt"], session_id, token
            )
            print(f"  Response: {result[:200]}...")
            session_ids.append(session_id)
        except Exception as e:
            print(f"  ERROR: {e}")
            # Continue with other test cases
            continue

        # Brief pause between invocations to allow trace propagation
        time.sleep(2)

    # Allow traces to flush to CloudWatch
    print("\nWaiting 30s for traces to propagate to CloudWatch...")
    time.sleep(30)

    print(f"\nCompleted {len(session_ids)}/{len(test_cases)} test sessions.")
    return session_ids


if __name__ == "__main__":
    session_ids = run_test_sessions()

    # Write session IDs to GitHub Actions output
    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a") as f:
            f.write(f"session_ids={json.dumps(session_ids)}\n")
