# scripts/deploy_agent.py
import boto3
import subprocess
import sys
import os

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

    # Build and push
    subprocess.run(["docker", "build", "-t", full_image_uri, "."], check=True)
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

    # Check if runtime already exists
    try:
        existing = control_client.get_agent_runtime(agentRuntimeName=agent_name)
        agent_runtime_arn = existing["agentRuntimeArn"]
        print(f"Updating existing runtime: {agent_runtime_arn}")

        control_client.update_agent_runtime(
            agentRuntimeArn=agent_runtime_arn,
            agentRuntimeArtifact=runtime_config["agentRuntimeArtifact"],
        )
    except control_client.exceptions.ResourceNotFoundException:
        print(f"Creating new runtime: {agent_name}")
        response = control_client.create_agent_runtime(**runtime_config)
        agent_runtime_arn = response["agentRuntimeArn"]

    print(f"Agent Runtime ARN: {agent_runtime_arn}")

    # Wait for runtime to become active
    waiter = control_client.get_waiter("agent_runtime_active")
    print("Waiting for runtime to become ACTIVE...")
    waiter.wait(agentRuntimeArn=agent_runtime_arn)
    print("Runtime is ACTIVE.")

    return agent_runtime_arn


if __name__ == "__main__":
    arn = deploy_agent()
    # Write ARN to GitHub Actions output
    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a") as f:
            f.write(f"agent_runtime_arn={arn}\n")
