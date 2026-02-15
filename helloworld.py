import asyncio
import os

from dotenv import load_dotenv
from azure.core.credentials import AzureKeyCredential
from agent_framework.azure import AzureOpenAIResponsesClient

# Load environment variables from .env
load_dotenv()

# Azure OpenAI configuration
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
api_key = os.getenv("AZURE_OPENAI_API_KEY")
model = os.getenv("AZURE_OPENAI_MODEL", "gpt-4.1-mini")

# Validate that required env vars are set
if not endpoint or not api_key:
    raise ValueError("AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY must be set in .env")

# Create client using API key authentication
credential = AzureKeyCredential(api_key)
client = AzureOpenAIResponsesClient(
    project_endpoint=endpoint,
    deployment_name=model,
    credential=credential,
)

# Create an agent using create_agent
agent = client.create_agent(
    name="Assistant",
    instructions="You are a friendly assistant. Keep your answers brief.",
)


async def main():
    print("🤖 Testing Azure OpenAI Agent (Microsoft Agent Framework)...\n")

    # Non-streaming: get the complete response at once
    response = await agent.run("What is the capital of France?")
    print(f"✅ Agent Response: {response.text}")

    # Streaming: receive tokens as they are generated
    print("\n🤖 Streaming response: ", end="", flush=True)
    async for update in agent.run_stream("Tell me a one-sentence fun fact."):
        if update.text:
            print(update.text, end="", flush=True)
    print()

    print("\n🎉 API connection successful!")


if __name__ == "__main__":
    asyncio.run(main())
