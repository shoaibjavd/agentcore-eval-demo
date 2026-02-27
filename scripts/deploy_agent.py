# scripts/deploy_agent.py
import boto3
import subprocess
import sys
import os
import time

def deploy_agent():
    region = os.environ["AWS_REGION"]
    account_id = os.environ["AWS_ACCOUNT_ID"]
    repo_name = os.environ["ECR_REPO_NAME"]
    agent_name = os.environ["AGENT_NAME"]
    runtime_role_arn = os.environ["AGENT_RUNTIME_ROLE_ARN"]
    image_tag = os.environ.get("IMAGE_TAG", "latest")

    ecr_uri = f"{account_id}.dkr.ecr.{region}.amazonaws.com/{repo_name}"
    full_image_uri = f"{ecr_uri}:{image_tag}"

    # Authenticate Docker with ECR
    ecr_client = boto3.client("ecr", region_name=region)
    token = ecr_client.get_authorization_token()
    endpoint = token["authorizationData"][0]["proxyEndpoint"]

    subprocess.run(
        f"aws ecr get-login-password --region {region} | "
        f"docker login --username AWS --password-stdin {endpoint}",
        shell=True, check=True,
    )

    # Build ARM64 image (required by AgentCore Runtime)
    # Use buildx for cross-platform build on x86_64 CI runners
    subprocess.run([
        "docker", "buildx", "build",
        "--platform", "linux/arm64",
        "-t", full_image_uri,
        "--load",
        "agent/"
    ], check=True)
    subprocess.run(["docker", "push", full_image_uri], check=True)
    print(f"Pushed image: {full_image_uri}")

    # Deploy to AgentCore Runtime
    control_client = boto3.client("bedrock-agentcore-control", region_name=region)

    runtime_config = {
        "agentRuntimeName": agent_name,
        "agentRuntimeArtifact": {
            "containerConfiguration": {
                "containerUri": full_image_uri,
            }
        },
        "roleArn": runtime_role_arn,
        "networkConfiguration": {"networkMode": "PUBLIC"},
    }

    # Add JWT authorizer if Cognito is configured
    cognito_discovery_url = os.environ.get("COGNITO_DISCOVERY_URL")
    cognito_client_id = os.environ.get("COGNITO_CLIENT_ID")
    cognito_audience = os.environ.get("COGNITO_AUDIENCE")

    if cognito_discovery_url:
        runtime_config["authorizerConfiguration"] = {
            "customJWTAuthorizerConfiguration": {
                "discoveryUrl": cognito_discovery_url,
                "allowedAudiences": [cognito_audience] if cognito_audience else [],
                "allowedClients": [cognito_client_id] if cognito_client_id else [],
            }
        }

    # Deploy: try to create, fall back to update if already exists
    agent_runtime_arn = None
    agent_runtime_id = None
    try:
        print(f"Creating new runtime: {agent_name}")
        response = control_client.create_agent_runtime(**runtime_config)
        agent_runtime_arn = response["agentRuntimeArn"]
        agent_runtime_id = response.get("agentRuntimeId")
    except control_client.exceptions.ConflictException:
        # Runtime already exists — find it and update
        print(f"Runtime '{agent_name}' already exists. Finding ARN...")
        try:
            runtimes = control_client.list_agent_runtimes()
            for rt in runtimes.get("agentRuntimes", []):
                if rt.get("agentRuntimeName") == agent_name:
                    agent_runtime_arn = rt["agentRuntimeArn"]
                    agent_runtime_id = rt.get("agentRuntimeId")
                    break
        except Exception as e:
            print(f"Warning: Could not list runtimes: {e}")

        if agent_runtime_arn:
            print(f"Updating existing runtime: {agent_runtime_arn}")
            control_client.update_agent_runtime(
                agentRuntimeId=agent_runtime_id,
                agentRuntimeArtifact=runtime_config["agentRuntimeArtifact"],
            )
        else:
            print("ERROR: Runtime exists but could not find ARN. Exiting.")
            sys.exit(1)

    print(f"Agent Runtime ARN: {agent_runtime_arn}")

    # Wait for runtime to become active (no SDK waiter available)
    print(f"Waiting for runtime to become ACTIVE (id={agent_runtime_id})...")
    for _ in range(60):  # Up to 5 minutes
        try:
            rt = control_client.get_agent_runtime(agentRuntimeId=agent_runtime_id)
            status = rt.get("status", "")
            print(f"  Status: {status}")
            if status == "READY":
                print("Runtime is ACTIVE.")
                break
            elif status in ("CREATE_FAILED", "UPDATE_FAILED"):
                reason = rt.get("failureReason", "unknown")
                print(f"ERROR: Runtime failed with status {status}: {reason}")
                sys.exit(1)
            time.sleep(5)
        except Exception as e:
            print(f"  Polling error: {e}")
            time.sleep(5)
    else:
        print("Warning: Timed out waiting for runtime. Proceeding anyway.")

    return agent_runtime_arn


if __name__ == "__main__":
    arn = deploy_agent()
    # Write ARN to GitHub Actions output
    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a") as f:
            f.write(f"agent_runtime_arn={arn}\n")
