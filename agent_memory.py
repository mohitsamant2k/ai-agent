"""
Agent Memory with Context Providers (v2 — Session API)

Context providers inject dynamic instructions into each agent call.
This sample defines a provider that tracks the user's name and
personalizes every response.

The AI itself extracts the name using a tool — no brittle string
parsing needed. Works with any phrasing: "My name is Mohit",
"I'm Mohit", "Call me Mohit", "Mohit here", etc.

Uses the NEW agent_framework APIs:
  - BaseContextProvider  (hooks: before_run / after_run)
  - AgentSession         (lightweight session state)
  - SessionContext       (per-call context: instructions, messages, tools)
  - as_agent()           (creates an Agent)
  - @tool                (defines a callable tool for the AI)
"""

import asyncio
import os
from typing import Annotated, Any

from dotenv import load_dotenv
from pydantic import Field
from agent_framework import tool
from agent_framework._sessions import BaseContextProvider, AgentSession, SessionContext
from agent_framework.azure import AzureOpenAIResponsesClient

# Load environment variables from .env
load_dotenv()

# Azure OpenAI configuration
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
api_key = os.getenv("AZURE_OPENAI_API_KEY")
model = os.getenv("AZURE_OPENAI_MODEL", "gpt-4.1-mini")

if not endpoint or not api_key:
    raise ValueError("AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY must be set in .env")


# ─── Shared memory store ─────────────────────────────────────────────────────

# A simple dict to hold what we know about the user.
# Both the tool and the context provider access this.
user_info: dict[str, str | None] = {"name": None}


# ─── Tool: AI calls this when it detects a name ─────────────────────────────

@tool
def remember_user_name(
    name: Annotated[str, Field(description="The user's name extracted from the conversation")],
) -> str:
    """Store the user's name for personalization.

    Call this tool whenever the user introduces themselves or
    mentions their name in ANY way — e.g. "I'm Mohit", "Call me Mohit",
    "My name is Mohit", "Mohit here", etc.
    """
    user_info["name"] = name
    print(f"   🔧 [Tool Called] remember_user_name(name='{name}')")
    return f"Got it! I'll remember that the user's name is {name}."


# ─── Context Provider: injects personalized instructions ─────────────────────

class UserNameProvider(BaseContextProvider):
    """Injects extra instructions based on what we know about the user.

    - before_run()  → if we know the name, tell the AI to use it
    - after_run()   → nothing needed — the tool handles extraction
    """

    def __init__(self) -> None:
        super().__init__(source_id="user_name_provider")

    async def before_run(
        self,
        *,
        agent: Any,
        session: AgentSession,
        context: SessionContext,
        state: dict[str, Any],
    ) -> None:
        """Called BEFORE the AI model runs. Inject extra instructions."""
        if user_info["name"]:
            context.extend_instructions(
                self.source_id,
                f"The user's name is {user_info['name']}. Always address them by name.",
            )
        else:
            context.extend_instructions(
                self.source_id,
                (
                    "You don't know the user's name yet. "
                    "If the user mentions their name in any way, "
                    "call the remember_user_name tool to save it."
                ),
            )


# ─── Create Agent with Memory ───────────────────────────────────────────────

client = AzureOpenAIResponsesClient(
    endpoint=endpoint,
    deployment_name=model,
    api_key=api_key,
)

memory = UserNameProvider()

agent = client.as_agent(
    name="MemoryAgent",
    instructions="You are a friendly assistant. Always be concise.",
    tools=[remember_user_name],
    context_providers=[memory],
)


# ─── Run Demo ────────────────────────────────────────────────────────────────


async def main():
    print("Agent Memory with Tool-Based Name Extraction\n")
    print("=" * 60)

    # Create a session (keeps conversation history)
    session = agent.create_session()

    # Call 1: No name given yet
    print("\nCall 1:")
    print("   User: Hello! What's the square root of 9?")
    result = await agent.run("Hello! What's the square root of 9?", session=session)
    print(f"   Agent: {result.text}")
    print(f"   [Memory] name = {user_info['name']}")

    # Call 2: Give name in a casual way — NOT "my name is"!
    print("\nCall 2:")
    print("   User: I'm Mohit, nice to meet you!")
    result = await agent.run("I'm Mohit, nice to meet you!", session=session)
    print(f"   Agent: {result.text}")
    print(f"   [Memory] name = {user_info['name']}")

    # Call 3: The AI should now use the name
    print("\nCall 3:")
    print("   User: What is 2 + 2?")
    result = await agent.run("What is 2 + 2?", session=session)
    print(f"   Agent: {result.text}")
    print(f"   [Memory] name = {user_info['name']}")

    print("\n" + "=" * 60)
    print(f"Final memory state: name = '{user_info['name']}'")
    print("Done!")


if __name__ == "__main__":
    asyncio.run(main())
