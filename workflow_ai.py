"""
AI Workflow Demo — Customer Support Pipeline

A real workflow where AI does the heavy lifting:

  Customer Message
        │
        ▼
  ┌──────────────────────┐
  │ Step 1: AI Classifier │  ← AI reads the message and classifies it
  │ (question/complaint/  │     as question, complaint, or compliment
  │  compliment)          │
  └──────────┬───────────┘
             │
             ▼
  ┌──────────────────────┐
  │ Step 2: AI Responder  │  ← AI writes an appropriate reply based
  │ (generates reply)     │     on the classification
  └──────────┬───────────┘
             │
             ▼
  ┌──────────────────────┐
  │ Step 3: Formatter     │  ← Plain code: adds ticket number, timestamp
  │ (no AI needed)        │
  └──────────┬───────────┘
             │
             ▼
        Final Response
"""

import asyncio
import os
from datetime import datetime
from dataclasses import dataclass

from dotenv import load_dotenv
from typing_extensions import Never

from agent_framework import (
    Executor,
    WorkflowBuilder,
    WorkflowContext,
    executor,
    handler,
)
from agent_framework.azure import AzureOpenAIResponsesClient

# Load environment variables
load_dotenv()

endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
api_key = os.getenv("AZURE_OPENAI_API_KEY")
model = os.getenv("AZURE_OPENAI_MODEL", "gpt-4.1-mini")

if not endpoint or not api_key:
    raise ValueError("AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY must be set in .env")


# ─── Data passed between steps ───────────────────────────────────────────────

@dataclass
class ClassifiedMessage:
    """Passed from Step 1 → Step 2"""
    original: str          # what the customer said
    category: str          # question / complaint / compliment
    sentiment: str         # positive / negative / neutral


@dataclass
class SupportResponse:
    """Passed from Step 2 → Step 3"""
    original: str
    category: str
    sentiment: str
    reply: str             # AI-generated response


@dataclass
class FinalTicket:
    """Final output of the workflow"""
    ticket_id: str
    timestamp: str
    category: str
    sentiment: str
    customer_message: str
    agent_reply: str


# ─── Create the AI client (shared by all steps) ─────────────────────────────

client = AzureOpenAIResponsesClient(
    endpoint=endpoint,
    deployment_name=model,
    api_key=api_key,
)


# ─── Step 1: AI Classifier ──────────────────────────────────────────────────

class Classifier(Executor):
    """AI reads the customer message and classifies it."""

    def __init__(self, id: str):
        super().__init__(id=id)
        self.agent = client.as_agent(
            name="Classifier",
            instructions=(
                "You are a message classifier for a customer support system. "
                "Given a customer message, respond with EXACTLY two words separated by a comma: "
                "the category and the sentiment.\n"
                "Categories: question, complaint, compliment\n"
                "Sentiments: positive, negative, neutral\n"
                "Example response: complaint, negative\n"
                "Another example: question, neutral\n"
                "ONLY respond with those two words. Nothing else."
            ),
        )

    @handler
    async def classify(self, text: str, ctx: WorkflowContext[ClassifiedMessage]) -> None:
        print(f"   🔍 Step 1 (AI Classifier): analyzing message...")

        # Ask the AI to classify
        result = await self.agent.run(text)
        response = result.text.strip().lower()

        # Parse "complaint, negative" → category="complaint", sentiment="negative"
        parts = [p.strip() for p in response.split(",")]
        category = parts[0] if len(parts) >= 1 else "question"
        sentiment = parts[1] if len(parts) >= 2 else "neutral"

        print(f"   🔍 Step 1 result: category='{category}', sentiment='{sentiment}'")

        # Send structured data to Step 2
        await ctx.send_message(ClassifiedMessage(
            original=text,
            category=category,
            sentiment=sentiment,
        ))


# ─── Step 2: AI Responder ───────────────────────────────────────────────────

class Responder(Executor):
    """AI generates an appropriate reply based on the classification."""

    def __init__(self, id: str):
        super().__init__(id=id)
        self.agent = client.as_agent(
            name="Responder",
            instructions=(
                "You are a friendly customer support agent. "
                "You will receive a customer message along with its classification. "
                "Write a helpful, professional reply in 2-3 sentences. "
                "Match your tone to the situation:\n"
                "- For complaints: be empathetic and offer a solution\n"
                "- For questions: be clear and informative\n"
                "- For compliments: be grateful and warm"
            ),
        )

    @handler
    async def respond(self, data: ClassifiedMessage, ctx: WorkflowContext[SupportResponse]) -> None:
        print(f"   💬 Step 2 (AI Responder): generating reply for '{data.category}'...")

        # Give the AI the full context
        prompt = (
            f"Customer message: \"{data.original}\"\n"
            f"Category: {data.category}\n"
            f"Sentiment: {data.sentiment}\n\n"
            f"Write an appropriate reply:"
        )
        result = await self.agent.run(prompt)

        print(f"   💬 Step 2 result: reply generated ({len(result.text)} chars)")

        # Send to Step 3
        await ctx.send_message(SupportResponse(
            original=data.original,
            category=data.category,
            sentiment=data.sentiment,
            reply=result.text,
        ))


# ─── Step 3: Formatter (no AI — plain Python) ───────────────────────────────

@executor(id="formatter")
async def formatter(data: SupportResponse, ctx: WorkflowContext[Never, FinalTicket]) -> None:
    """Add ticket number and timestamp — pure code, no AI needed."""
    ticket_id = f"TKT-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print(f"   📋 Step 3 (Formatter): assigned ticket {ticket_id}")

    await ctx.yield_output(FinalTicket(
        ticket_id=ticket_id,
        timestamp=timestamp,
        category=data.category,
        sentiment=data.sentiment,
        customer_message=data.original,
        agent_reply=data.reply,
    ))


# ─── Build the workflow: Classifier → Responder → Formatter ─────────────────

def create_support_workflow():
    classifier = Classifier(id="classifier")
    responder = Responder(id="responder")
    return (
        WorkflowBuilder(start_executor=classifier)
        .add_edge(classifier, responder)     # Step 1 → Step 2
        .add_edge(responder, formatter)      # Step 2 → Step 3
        .build()
    )


# ─── Run Demo ───────────────────────────────────────────────────────────────

async def main():
    print("AI Customer Support Workflow\n")
    print("=" * 70)

    workflow = create_support_workflow()

    # Test messages — different types
    test_messages = [
        "My order #4521 arrived broken and I want a refund immediately!",
        "How do I change my shipping address for an existing order?",
        "Your product is amazing! Best purchase I've made this year!",
    ]

    for i, message in enumerate(test_messages, 1):
        print(f"\n{'─' * 70}")
        print(f"Customer #{i}: \"{message}\"\n")

        events = await workflow.run(message)

        # Get the final ticket
        ticket = events.get_outputs()[0]

        print(f"\n   ┌─────────────────────────────────────────────────────────")
        print(f"   │ Ticket:    {ticket.ticket_id}")
        print(f"   │ Time:      {ticket.timestamp}")
        print(f"   │ Category:  {ticket.category}")
        print(f"   │ Sentiment: {ticket.sentiment}")
        print(f"   │ Customer:  {ticket.customer_message[:60]}...")
        print(f"   │ Reply:     {ticket.agent_reply}")
        print(f"   └─────────────────────────────────────────────────────────")

    print(f"\n{'=' * 70}")
    print("Done! All customer messages processed through the AI pipeline.")


if __name__ == "__main__":
    asyncio.run(main())
