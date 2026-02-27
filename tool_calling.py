import asyncio
import os
from typing import Annotated
from datetime import datetime

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


# ─── Define Tools ────────────────────────────────────────────────────────────


@tool
def get_weather(
    city: Annotated[str, Field(description="The city name to get weather for")],
) -> str:
    """Get the current weather for a given city."""
    # Simulated weather data
    weather_data = {
        "paris": "☀️ 22°C, Sunny",
        "london": "🌧️ 15°C, Rainy",
        "tokyo": "⛅ 28°C, Partly Cloudy",
        "new york": "🌤️ 18°C, Clear",
        "mumbai": "🌡️ 35°C, Hot and Humid",
    }
    result = weather_data.get(city.lower(), f"🌍 Weather data not available for {city}")
    print(f"   🔧 [Tool Called] get_weather(city='{city}') → {result}")
    return result


@tool
def calculate(
    expression: Annotated[str, Field(description="A math expression to evaluate, e.g. '2 + 3 * 4'")],
) -> str:
    """Evaluate a mathematical expression and return the result."""
    try:
        # Only allow safe math operations
        allowed_chars = set("0123456789+-*/.() ")
        if not all(c in allowed_chars for c in expression):
            return f"Error: Invalid characters in expression"
        result = eval(expression)  # safe: only numbers and operators
        print(f"   🔧 [Tool Called] calculate(expression='{expression}') → {result}")
        return str(result)
    except Exception as e:
        return f"Error: {e}"


@tool
def get_current_time() -> str:
    """Get the current date and time."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"   🔧 [Tool Called] get_current_time() → {now}")
    return now


# ─── Create Agent with Tools ────────────────────────────────────────────────

client = AzureOpenAIResponsesClient(
    endpoint=endpoint,
    deployment_name=model,
    api_key=api_key,
)

agent = client.as_agent(
    name="ToolAgent",
    instructions=(
        "You are a helpful assistant with access to tools. "
        "Use the tools when needed to answer questions accurately. "
        "Always be concise in your responses."
    ),
    tools=[get_weather, calculate, get_current_time],
)


# ─── Run Tests ───────────────────────────────────────────────────────────────


async def main():
    print("🤖 Tool Calling Test\n")
    print("=" * 60)

    # Test 1: Weather tool
    print("\n📌 Test 1: Weather lookup")
    print("   User: What's the weather in Tokyo and London?")
    response = await agent.run("What's the weather in Tokyo and London?")
    print(f"   Agent: {response.text}")

    # Test 2: Calculator tool
    print("\n📌 Test 2: Math calculation")
    print("   User: What is 145 * 37 + 89?")
    response = await agent.run("What is 145 * 37 + 89?")
    print(f"   Agent: {response.text}")

    # Test 3: Time tool
    print("\n📌 Test 3: Current time")
    print("   User: What time is it right now?")
    response = await agent.run("What time is it right now?")
    print(f"   Agent: {response.text}")

    # Test 4: Multiple tools in one query
    print("\n📌 Test 4: Multiple tools combined")
    print("   User: What's the weather in Mumbai and what time is it?")
    response = await agent.run("What's the weather in Mumbai and what time is it?")
    print(f"   Agent: {response.text}")

    # Test 5: No tool needed
    print("\n📌 Test 5: No tool needed (agent answers directly)")
    print("   User: What is Python?")
    response = await agent.run("What is Python?")
    print(f"   Agent: {response.text}")

    print("\n" + "=" * 60)
    print("✅ All tool calling tests completed!")


if __name__ == "__main__":
    asyncio.run(main())
