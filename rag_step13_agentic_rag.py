"""
Step 13: Agentic RAG — The AI Decides What to Search and When

THE BIG SHIFT:
  Steps 5-12: WE wrote the pipeline. The code decides:
    1. Search → 2. Re-rank → 3. Answer. Always the same steps.
  
  Step 13: The AI IS the pipeline. IT decides:
    - "Do I need to search?"
    - "What should I search for?"
    - "Do I have enough info, or should I search again?"
    - "Should I search a different way?"
    - "I'm done — here's the answer."

ANALOGY:
  Steps 5-12 = GPS navigation (fixed route, turn-by-turn)
  Step 13    = Human driver (looks around, makes decisions, adapts)

HOW IT WORKS:
  The LLM gets TOOLS — functions it can call:
    - search_knowledge_base(query) → search our docs
    - calculate(expression) → do math
    - summarize(text) → condense long text
  
  The LLM enters a REASONING LOOP:
    1. Think: "What do I need to answer this question?"
    2. Act:   Call a tool (e.g., search for "Python decorators")
    3. Observe: Read the results
    4. Think: "Is this enough? Do I need more info?"
    5. Repeat or Answer

  This is the ReAct pattern (Reasoning + Acting).

This file demonstrates:
  Part 1: Define tools the agent can use
  Part 2: The ReAct reasoning loop
  Part 3: Agent in action — multi-step queries
  Part 4: Comparison — fixed pipeline vs agentic
  Part 5: Architecture diagram + Summary
"""

import os
import re
import json
import time
import numpy as np
import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from dotenv import load_dotenv
from openai import OpenAI
from rag_utils import (
    recursive_chunk, distance_to_similarity,
    PYTHON_DOC, AI_DOC, WEBDEV_DOC,
)


# ═══════════════════════════════════════════════════════════════════════════════
# SETUP: Knowledge Base + LLM
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("SETUP: Building Knowledge Base for Agentic RAG")
print("=" * 70)

documents = {
    "python_guide": PYTHON_DOC,
    "ai_ml_guide": AI_DOC,
    "webdev_guide": WEBDEV_DOC,
}

all_chunks = []
all_ids = []
all_metadatas = []

for doc_name, content in documents.items():
    chunks = recursive_chunk(content, chunk_size=400, chunk_overlap=50)
    for i, chunk in enumerate(chunks):
        all_chunks.append(chunk)
        all_ids.append(f"{doc_name}::chunk_{i}")
        all_metadatas.append({"source": doc_name, "chunk_index": i})

print(f"   {len(all_chunks)} chunks from {len(documents)} documents")

tokenized_chunks = [chunk.lower().split() for chunk in all_chunks]
bm25 = BM25Okapi(tokenized_chunks)

client = chromadb.Client()
collection = client.create_collection(name="agentic_kb")
collection.add(documents=all_chunks, ids=all_ids, metadatas=all_metadatas)

cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

load_dotenv()
azure_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", "")
azure_api_key = os.environ.get("AZURE_OPENAI_API_KEY", "")
model_name = os.environ.get("AZURE_OPENAI_MODEL", "gpt-4.1-mini")
base_url = azure_endpoint.replace("/chat/completions", "")
llm_client = OpenAI(base_url=base_url, api_key=azure_api_key)

print(f"   LLM ready: {model_name}")
print("   Setup complete!\n")


# ═══════════════════════════════════════════════════════════════════════════════
# PART 1: Define Tools the Agent Can Use
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("PART 1: Agent Tools — What the AI Can Do")
print("=" * 70)
print()
print("The agent gets 3 tools:")
print("  1. search_knowledge_base(query) — search our documents")
print("  2. calculate(expression) — evaluate math expressions")
print("  3. get_document_list() — see what documents are available")
print()
print("The agent DECIDES which tools to call, in what order,")
print("and when it has enough information to answer.")
print()


# ── Tool 1: Search Knowledge Base ──────────────────────────────────────────

def search_knowledge_base(query, top_k=3):
    """
    Search the knowledge base using hybrid search + re-ranking.
    The agent calls this to find relevant information.
    
    Args:
        query: What to search for
        top_k: Number of results to return
    
    Returns:
        str: Formatted search results
    """
    # Vector search
    v_results = collection.query(query_texts=[query], n_results=top_k)
    
    # BM25 search
    bm25_scores = bm25.get_scores(query.lower().split())
    top_bm25 = np.argsort(bm25_scores)[::-1][:top_k]
    
    # Merge with RRF
    rrf_scores = {}
    doc_data = {}
    
    for rank in range(len(v_results["ids"][0])):
        did = v_results["ids"][0][rank]
        rrf_scores[did] = rrf_scores.get(did, 0) + 1.0 / (60 + rank + 1)
        doc_data[did] = {
            "chunk": v_results["documents"][0][rank],
            "id": did,
            "source": v_results["metadatas"][0][rank]["source"],
        }
    
    for rank, idx in enumerate(top_bm25):
        did = all_ids[idx]
        rrf_scores[did] = rrf_scores.get(did, 0) + 1.0 / (60 + rank + 1)
        doc_data[did] = {
            "chunk": all_chunks[idx],
            "id": did,
            "source": all_metadatas[idx]["source"],
        }
    
    sorted_ids = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)
    results = [doc_data[did] for did in sorted_ids[:top_k * 2]]
    
    # Re-rank
    if results:
        pairs = [(query, r["chunk"]) for r in results]
        ce_scores = cross_encoder.predict(pairs)
        for i, r in enumerate(results):
            r["ce_score"] = float(ce_scores[i])
        results = sorted(results, key=lambda x: x["ce_score"], reverse=True)
    
    results = results[:top_k]
    
    # Format as readable text for the agent
    output = f"Search results for: \"{query}\"\n"
    output += f"Found {len(results)} relevant passages:\n\n"
    for i, r in enumerate(results, 1):
        output += f"[Result {i}] Source: {r['source']}\n"
        output += f"{r['chunk']}\n\n"
    
    return output


# ── Tool 2: Calculate ──────────────────────────────────────────────────────

def calculate(expression):
    """
    Evaluate a mathematical expression safely.
    The agent calls this when it needs to do math.
    
    Args:
        expression: Math expression like "2 + 3 * 4"
    
    Returns:
        str: Result of the calculation
    """
    try:
        # Only allow safe math operations
        allowed = set("0123456789+-*/.() %")
        clean = expression.strip()
        if not all(c in allowed for c in clean):
            return f"Error: Invalid characters in expression: {clean}"
        
        result = eval(clean)  # Safe because we filtered chars
        return f"Calculation: {clean} = {result}"
    except Exception as e:
        return f"Error calculating '{expression}': {str(e)}"


# ── Tool 3: Get Document List ─────────────────────────────────────────────

def get_document_list():
    """
    List all available documents in the knowledge base.
    Helps the agent understand what information is available.
    
    Returns:
        str: List of documents with descriptions
    """
    doc_list = "Available documents in knowledge base:\n\n"
    doc_descriptions = {
        "python_guide": "Python Programming Guide — covers basics, error handling, "
                        "decorators, async programming, testing, package ecosystem",
        "ai_ml_guide": "AI & Machine Learning Guide — covers supervised/unsupervised/"
                       "reinforcement learning, deep learning, transformers, "
                       "transfer learning, RAG",
        "webdev_guide": "Web Development Guide — covers HTML, CSS, JavaScript, "
                        "React, REST APIs, Node.js, responsive design",
    }
    for name, desc in doc_descriptions.items():
        doc_list += f"  • {name}: {desc}\n"
    
    return doc_list


# ── Tool registry (maps name -> function) ──────────────────────────────────

TOOLS = {
    "search_knowledge_base": search_knowledge_base,
    "calculate": calculate,
    "get_document_list": get_document_list,
}

# Tool descriptions for the LLM (so it knows what each tool does)
TOOL_DESCRIPTIONS = """Available tools:

1. search_knowledge_base(query: str)
   Search the knowledge base for relevant information.
   Use this to find facts, explanations, and details from our documents.
   Example: search_knowledge_base("Python error handling")

2. calculate(expression: str)
   Evaluate a mathematical expression.
   Use this when you need to do arithmetic.
   Example: calculate("100 / 3 * 2.5")

3. get_document_list()
   List all available documents in the knowledge base.
   Use this to understand what topics are covered.
   Example: get_document_list()"""

print("-- Tools registered --\n")
for name in TOOLS:
    print(f"  ✅ {name}")
print()


# ═══════════════════════════════════════════════════════════════════════════════
# PART 2: The ReAct Reasoning Loop
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("PART 2: The ReAct Loop — Reasoning + Acting")
print("=" * 70)
print()
print("ReAct = Reason + Act. The agent alternates between:")
print("  THINK  → reason about what to do next")
print("  ACT    → call a tool")
print("  OBSERVE→ read the tool's output")
print("  ...repeat until ready to answer...")
print("  ANSWER → give the final response")
print()

AGENT_SYSTEM = """You are an intelligent research assistant with access to tools.
Your job is to answer the user's question by searching for information and reasoning step by step.

{tool_descriptions}

IMPORTANT RULES:
1. Think step by step before acting
2. Use tools to find information — don't make up facts
3. You can call multiple tools if needed
4. When you have enough information, provide a final answer

FORMAT YOUR RESPONSE EXACTLY LIKE THIS:

THINK: [your reasoning about what to do next]
ACT: [tool_name]([arguments])

After receiving tool results, continue with more THINK/ACT pairs or give your final answer:

THINK: [reasoning about the results]
ANSWER: [your final answer based on the information gathered]

RULES FOR ACT:
- For search: ACT: search_knowledge_base("your search query")
- For math: ACT: calculate("2 + 3 * 4")  
- For docs: ACT: get_document_list()
- Only ONE tool call per ACT line
- Always wrap string arguments in double quotes"""


def parse_agent_response(response):
    """
    Parse the agent's response to extract THINK, ACT, and ANSWER.
    
    Returns:
        dict with 'thinks', 'action', 'action_args', 'answer'
    """
    result = {
        "thinks": [],
        "action": None,
        "action_args": None,
        "answer": None,
        "raw": response,
    }
    
    lines = response.strip().split("\n")
    current_section = None
    current_text = []
    
    for line in lines:
        stripped = line.strip()
        
        if stripped.startswith("THINK:"):
            if current_section == "think" and current_text:
                result["thinks"].append(" ".join(current_text))
            current_section = "think"
            current_text = [stripped[6:].strip()]
        
        elif stripped.startswith("ACT:"):
            if current_section == "think" and current_text:
                result["thinks"].append(" ".join(current_text))
            current_section = "act"
            act_text = stripped[4:].strip()
            
            # Parse tool call: tool_name("args") or tool_name()
            match = re.match(r'(\w+)\((.*)\)$', act_text, re.DOTALL)
            if match:
                result["action"] = match.group(1)
                args_str = match.group(2).strip().strip('"').strip("'")
                result["action_args"] = args_str if args_str else None
            current_text = []
        
        elif stripped.startswith("ANSWER:"):
            if current_section == "think" and current_text:
                result["thinks"].append(" ".join(current_text))
            current_section = "answer"
            current_text = [stripped[7:].strip()]
        
        else:
            if current_section and stripped:
                current_text.append(stripped)
    
    # Capture last section
    if current_section == "think" and current_text:
        result["thinks"].append(" ".join(current_text))
    elif current_section == "answer" and current_text:
        result["answer"] = " ".join(current_text)
    
    return result


def run_agent(query, max_steps=5, verbose=True):
    """
    Run the ReAct agent loop.
    
    The agent thinks → acts → observes → repeats until it has an answer.
    
    Args:
        query: The user's question
        max_steps: Maximum tool calls before forcing an answer
        verbose: Print the agent's reasoning
    
    Returns:
        dict with answer, steps taken, tools used, and timing
    """
    if verbose:
        print(f"\n  🤖 Agent starting on: \"{query}\"\n")
    
    # Build the conversation
    system_msg = AGENT_SYSTEM.format(tool_descriptions=TOOL_DESCRIPTIONS)
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": query},
    ]
    
    steps = []
    tools_used = []
    t_start = time.time()
    
    for step in range(max_steps):
        # Get agent's response
        response = llm_client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0.2,
            max_tokens=600,
        )
        agent_text = response.choices[0].message.content
        
        # Parse the response
        parsed = parse_agent_response(agent_text)
        
        # Log thinks
        for think in parsed["thinks"]:
            if verbose:
                print(f"  💭 THINK: {think[:100]}{'...' if len(think) > 100 else ''}")
            steps.append({"type": "think", "content": think})
        
        # Check if agent wants to answer
        if parsed["answer"]:
            if verbose:
                print(f"\n  ✅ ANSWER: {parsed['answer'][:200]}{'...' if len(parsed['answer']) > 200 else ''}")
            
            return {
                "answer": parsed["answer"],
                "steps": steps,
                "tools_used": tools_used,
                "num_steps": step + 1,
                "time": time.time() - t_start,
            }
        
        # Execute the tool
        if parsed["action"]:
            tool_name = parsed["action"]
            tool_args = parsed["action_args"]
            
            if verbose:
                args_display = f'"{tool_args}"' if tool_args else ""
                print(f"  🔧 ACT: {tool_name}({args_display})")
            
            # Call the tool
            if tool_name in TOOLS:
                tool_fn = TOOLS[tool_name]
                if tool_args:
                    tool_result = tool_fn(tool_args)
                else:
                    tool_result = tool_fn()
                tools_used.append(tool_name)
            else:
                tool_result = f"Error: Unknown tool '{tool_name}'"
            
            if verbose:
                preview = tool_result[:100].replace("\n", " ")
                print(f"  👁️ OBSERVE: {preview}...")
            
            steps.append({
                "type": "action",
                "tool": tool_name,
                "args": tool_args,
                "result_preview": tool_result[:200],
            })
            
            # Add the tool result to the conversation
            messages.append({"role": "assistant", "content": agent_text})
            messages.append({
                "role": "user",
                "content": f"OBSERVATION:\n{tool_result}\n\nContinue your reasoning. "
                           f"If you have enough info, respond with ANSWER: [your answer]. "
                           f"Otherwise, use another tool.",
            })
        else:
            # No action and no answer — agent might be confused
            # Add the response and ask it to continue
            messages.append({"role": "assistant", "content": agent_text})
            messages.append({
                "role": "user",
                "content": "Please either use a tool (ACT:) or provide your answer (ANSWER:).",
            })
    
    # Max steps reached — force an answer
    if verbose:
        print(f"\n  ⚠️ Max steps ({max_steps}) reached, forcing answer...")
    
    messages.append({
        "role": "user",
        "content": "You've used all your steps. Please provide your ANSWER now "
                   "based on what you've gathered so far.",
    })
    
    response = llm_client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=0.2,
        max_tokens=600,
    )
    final = response.choices[0].message.content
    parsed = parse_agent_response(final)
    answer = parsed["answer"] or final
    
    if verbose:
        print(f"  ✅ ANSWER: {answer[:200]}...")
    
    return {
        "answer": answer,
        "steps": steps,
        "tools_used": tools_used,
        "num_steps": max_steps,
        "time": time.time() - t_start,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# PART 3: Agent in Action — Multi-Step Queries
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("PART 3: Agent in Action -- Solving Complex Queries")
print("=" * 70)
print()
print("Watch the agent THINK, decide which tools to use, and build")
print("its answer step by step. Each query shows the full reasoning.")
print()

agent_queries = [
    {
        "query": "What are the main differences between Python decorators and "
                 "JavaScript higher-order components in React?",
        "why": "Needs info from 2 different documents — agent must search twice",
    },
    {
        "query": "I have 3 documents in my knowledge base. If each document "
                 "produces about 10 chunks, and each chunk takes 1.3 seconds "
                 "to add context (like in Step 12), how long would it take total?",
        "why": "Needs both document lookup AND math — agent must use 2 tools",
    },
    {
        "query": "What technique would help me find documents about 'error handling' "
                 "when my chunks only say 'try-except blocks' without mentioning errors?",
        "why": "Agent needs to reason about RAG techniques from its search results",
    },
]

for test in agent_queries:
    print(f"  {'═' * 56}")
    print(f"  Query: \"{test['query'][:70]}{'...' if len(test['query']) > 70 else ''}\"")
    print(f"  Why:   {test['why']}")
    print(f"  {'═' * 56}")
    
    result = run_agent(test["query"], max_steps=5)
    
    print(f"\n  📊 Stats: {result['num_steps']} steps, "
          f"{len(result['tools_used'])} tool calls "
          f"({', '.join(result['tools_used']) or 'none'}), "
          f"{result['time']:.1f}s")
    print()


# ═══════════════════════════════════════════════════════════════════════════════
# PART 4: Comparison — Fixed Pipeline vs Agentic
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("PART 4: Fixed Pipeline vs Agentic RAG")
print("=" * 70)
print()
print("Comparing: does the agent actually give BETTER answers?")
print()


def fixed_pipeline_answer(query, top_k=3):
    """The fixed pipeline from previous steps — always same steps."""
    t0 = time.time()
    
    # Always: search → re-rank → answer
    v_results = collection.query(query_texts=[query], n_results=top_k)
    bm25_scores_arr = bm25.get_scores(query.lower().split())
    top_bm25 = np.argsort(bm25_scores_arr)[::-1][:top_k]
    
    rrf = {}
    data = {}
    for rank in range(len(v_results["ids"][0])):
        did = v_results["ids"][0][rank]
        rrf[did] = rrf.get(did, 0) + 1.0 / (60 + rank + 1)
        data[did] = v_results["documents"][0][rank]
    for rank, idx in enumerate(top_bm25):
        did = all_ids[idx]
        rrf[did] = rrf.get(did, 0) + 1.0 / (60 + rank + 1)
        data[did] = all_chunks[idx]
    
    sorted_ids = sorted(rrf, key=lambda x: rrf[x], reverse=True)[:top_k * 2]
    results = [{"chunk": data[did], "id": did} for did in sorted_ids]
    
    if results:
        pairs = [(query, r["chunk"]) for r in results]
        ce = cross_encoder.predict(pairs)
        for i, r in enumerate(results):
            r["ce_score"] = float(ce[i])
        results = sorted(results, key=lambda x: x["ce_score"], reverse=True)[:top_k]
    
    context = "\n\n---\n\n".join([r["chunk"] for r in results])
    
    response = llm_client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "Answer using ONLY the provided context."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"},
        ],
        temperature=0.3,
        max_tokens=400,
    )
    
    return {
        "answer": response.choices[0].message.content,
        "time": time.time() - t0,
        "searches": 1,
    }


comparison_queries = [
    "Compare Python async programming with JavaScript promises",
    "What RAG techniques help when search results are irrelevant?",
    "How does transfer learning reduce the cost of training AI models?",
]

for query in comparison_queries:
    print(f"  {'─' * 56}")
    print(f"  Query: \"{query}\"")
    print(f"  {'─' * 56}\n")
    
    # Fixed pipeline
    fixed = fixed_pipeline_answer(query)
    print(f"  [FIXED PIPELINE] ({fixed['time']:.1f}s, {fixed['searches']} search)")
    print(f"    {fixed['answer'][:150].replace(chr(10), ' ')}...")
    print()
    
    # Agentic
    agent_result = run_agent(query, max_steps=4, verbose=False)
    print(f"  [AGENTIC] ({agent_result['time']:.1f}s, "
          f"{len(agent_result['tools_used'])} tool calls)")
    answer_text = agent_result['answer'] or "No answer generated"
    print(f"    {answer_text[:150].replace(chr(10), ' ')}...")
    print()


# ═══════════════════════════════════════════════════════════════════════════════
# PART 5: Architecture Diagram + Summary
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("PART 5: Architecture + Summary")
print("=" * 70)
print("""
  AGENTIC RAG — The AI Drives the Pipeline
  ==========================================

  FIXED PIPELINE (Steps 5-12):
  
  Query → Search → Re-rank → Answer
  (always same steps, always same order)

  AGENTIC RAG (Step 13):

  User: "Compare Python async with JS promises"
         │
         v
  ┌──────────────────────────────────────────┐
  │  🤖 AGENT (LLM with tools)              │
  │                                          │
  │  💭 THINK: I need info about both        │
  │            Python async AND JS promises. │
  │            Let me search for each.       │
  │                     │                    │
  │  🔧 ACT: search_knowledge_base(         │
  │          "Python async programming")     │
  │                     │                    │
  │  👁️ OBSERVE: Found info about asyncio,   │
  │             await, async/await...        │
  │                     │                    │
  │  💭 THINK: Good, now I need JS promises. │
  │                     │                    │
  │  🔧 ACT: search_knowledge_base(         │
  │          "JavaScript promises async")    │
  │                     │                    │
  │  👁️ OBSERVE: Found info about promises,  │
  │             async/await in JS...         │
  │                     │                    │
  │  💭 THINK: I have both sides. Let me     │
  │            compare them now.             │
  │                     │                    │
  │  ✅ ANSWER: Both Python and JavaScript   │
  │            support async/await...        │
  └──────────────────────────────────────────┘

  THE ReAct PATTERN:
  ┌─────────┐    ┌─────────┐    ┌─────────┐
  │  THINK  │───→│   ACT   │───→│ OBSERVE │──┐
  │(reason) │    │(use tool)│    │(read    │  │
  │         │    │         │    │ result) │  │
  └─────────┘    └─────────┘    └─────────┘  │
       ↑                                      │
       └──────────────────────────────────────┘
                    (repeat until done)
                         │
                         v
                    ┌─────────┐
                    │ ANSWER  │
                    └─────────┘

  FIXED PIPELINE vs AGENTIC:
  ┌────────────────────┬──────────────┬──────────────────┐
  │ Aspect             │ Fixed        │ Agentic          │
  ├────────────────────┼──────────────┼──────────────────┤
  │ Who decides?       │ Code         │ LLM              │
  │ # of searches      │ Always 1     │ As many as needed│
  │ Search queries      │ User's exact │ LLM rewrites them│
  │ Can do math?       │ No           │ Yes (tool)       │
  │ Multi-doc queries  │ Weak         │ Strong           │
  │ Speed              │ Fast (1 LLM) │ Slower (N LLMs)  │
  │ Cost               │ Low          │ Higher           │
  │ Reliability        │ Predictable  │ Can go off-track │
  └────────────────────┴──────────────┴──────────────────┘

  WHEN TO USE EACH:
  • Fixed pipeline: Simple Q&A, speed matters, predictable
  • Agentic: Complex questions, multi-step reasoning,
             user questions that need multiple searches
""")

print("  PROGRESS SO FAR:")
print("    Steps 1-5:  RAG Foundations (embeddings → naive RAG)")
print("    Steps 6-8:  Better Retrieval (hybrid, re-rank, citations)")
print("    Steps 9-10: Self-Correction (CRAG, Self-RAG)")
print("    Step 11:    Query Transformation (fix the question)")
print("    Step 12:    Contextual Retrieval (fix the chunks)")
print("    Step 13:    Agentic RAG (AI drives the pipeline)")
print()
print("  🎉 This completes the Advanced RAG section!")
print()
print("  NEXT -> Step 14: Conversation Memory")
print("    Remember previous questions in a chat session.")
print("    Handle follow-ups like 'What about Q2?'")
print()
print("[OK] Step 13 complete! You now understand Agentic RAG.\n")
