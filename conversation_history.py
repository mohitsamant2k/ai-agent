import asyncio
import os
from typing import Annotated

from dotenv import load_dotenv
from azure.core.credentials import AzureKeyCredential
from agent_framework import tool
from agent_framework.azure import AzureOpenAIResponsesClient
from pydantic import Field

# Load environment variables from .env
load_dotenv()

# Azure OpenAI configuration
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
api_key = os.getenv("AZURE_OPENAI_API_KEY")
model = os.getenv("AZURE_OPENAI_MODEL", "gpt-4.1-mini")

if not endpoint or not api_key:
    raise ValueError("AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY must be set in .env")


# --- Define Tools ---


@tool
def get_weather(
    city: Annotated[str, Field(description="The city name to get weather for")],
) -> str:
    """Get the current weather for a given city."""
    weather_data = {
        "paris": "22 C, Sunny",
        "london": "15 C, Rainy",
        "tokyo": "28 C, Partly Cloudy",
        "new york": "18 C, Clear",
        "mumbai": "35 C, Hot and Humid",
        "delhi": "40 C, Very Hot",
    }
    result = weather_data.get(city.lower(), f"Weather data not available for {city}")
    print(f"   [Tool Called] get_weather(city='{city}') -> {result}")
    return result


# --- Create Agent ---

client = AzureOpenAIResponsesClient(
    endpoint=endpoint,
    deployment_name=model,
    api_key=api_key,
)

agent = client.as_agent(
    name="ChatBot",
    instructions=(
        "You are a helpful and friendly assistant. "
        "Use tools when needed. Always be concise."
    ),
    tools=[get_weather],
)


# --- Demo: Conversation with memory + tool calling ---


async def main():
    print("Conversation History + Tool Calling Demo\n")
    print("=" * 60)

    # Create a session to keep conversation history
    session = agent.create_session()

    # Call 1: Introduce yourself
    print("\nCall 1:")
    print("   User: Hi! My name is Mohit and I live in Delhi")
    response = await agent.run("Hi! My name is Mohit and I live in Delhi", session=session)
    print(f"   Agent: {response.text}")

    # Call 2: Ask weather - agent should remember the city from Call 1
    print("\nCall 2:")
    print("   User: What's the weather in my city?")
    response = await agent.run("What's the weather in my city?", session=session)
    print(f"   Agent: {response.text}")

    # Call 3: Ask about another city
    print("\nCall 3:")
    print("   User: How about Mumbai?")
    response = await agent.run("How about Mumbai?", session=session)
    print(f"   Agent: {response.text}")

    # Call 4: Compare - agent needs to remember both
    print("\nCall 4:")
    print("   User: Which city is hotter?")
    response = await agent.run("Which city is hotter?", session=session)
    print(f"   Agent: {response.text}")

    # Call 5: Verify memory - agent should know name + city
    print("\nCall 5:")
    print("   User: What do you know about me so far?")
    response = await agent.run("What do you know about me so far?", session=session)
    print(f"   Agent: {response.text}")

    print("\n" + "=" * 60)
    print("Done! The agent remembered your city and used the weather tool!")


if __name__ == "__main__":
    asyncio.run(main())
