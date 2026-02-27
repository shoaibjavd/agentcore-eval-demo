# agent/main.py
from bedrock_agentcore.runtime import BedrockAgentCoreApp
from strands import Agent

app = BedrockAgentCoreApp()

agent = Agent(
    system_prompt="You are a helpful assistant. Answer questions clearly and concisely.",
)

@app.entrypoint
def handler(payload):
    user_input = payload.get("prompt", "")
    response = agent(user_input)
    return response.message["content"][0]["text"]

if __name__ == "__main__":
    app.run()
# test eval pipeline trigger
# iam fix
# ecr fix
