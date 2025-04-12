# worker_agent_server.py
import uuid

import uvicorn
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from pydantic import ValidationError
import json
import os
import datetime
import time
from dotenv import load_dotenv

# Agno imports
from agno.agent import Agent
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.models.groq import Groq # Or OpenAIChat, AnthropicChat etc.

# Local A2A tool import
from agno_a2a_tools import register_with_registry
# Import models/helpers from registry (or duplicate if preferred)
from registry_server import AgentCardModel, create_a2a_task_response, create_jsonrpc_response, create_jsonrpc_error

load_dotenv()

# --- Worker Agent Configuration ---
WORKER_HOST = "0.0.0.0"
WORKER_PORT = int(os.getenv("WORKER_PORT", 8001))
WORKER_URL = os.getenv("WORKER_AGENT_URL", f"http://localhost:{WORKER_PORT}")
REGISTRY_URL = os.getenv("A2A_REGISTRY_URL", "http://localhost:8000")

# --- Define the Agent's Card ---
# Ensure this matches the AgentCardModel structure from registry_server
MY_AGENT_CARD = AgentCardModel(
    name="WebSearchWorkerAgent",
    type="text",
    description="An Agno agent that searches the web using DuckDuckGo and provides concise results.",
    url=f"{WORKER_URL}/a2a",
    version="1.1.0",
    capabilities={"streaming": False, "pushNotifications": False, "stateTransitionHistory": False},
    authentication={"schemes": []}, # No auth for simplicity
    skills=[
        {
            "id": "web_search",
            "type": "text",
            "name": "Web Search",
            "description": "Performs a web search for a given query string (provided in a TextPart) and returns results.",
            "inputModes": ["text"],
            "outputModes": ["text"]
        }
    ]
)
MY_AGENT_CARD_JSON = MY_AGENT_CARD.model_dump_json() # Get JSON string from Pydantic model

# --- Initialize Agno Agent Core Logic ---
print("[Worker Agent] Initializing Agno core...")
# Ensure you have GROQ_API_KEY in your .env or provide model credentials another way
try:
    # Check for API key before initializing the model
    if not os.getenv("GROQ_API_KEY"):
         print("WARNING: GROQ_API_KEY not found in environment. Agent may not function.")
         # You might want to raise an error or use a fallback model here
         # For demo purposes, we'll let it proceed, but tool calls will fail.

    web_searcher_agent = Agent(
       model=Groq(id="meta-llama/llama-4-scout-17b-16e-instruct"), # Using Groq Llama3-8b for speed
       # model=OpenAIChat(model="gpt-3.5-turbo"), # Or OpenAI
       description="You are a web search assistant. Use the DuckDuckGo tool to find information.",
       tools=[DuckDuckGoTools(fixed_max_results=3)], # Use DuckDuckGo, limit results
       instructions=[
           "Understand the user query provided.",
           "Use the DuckDuckGo search tool to find relevant web pages.",
           "Summarize the key findings from the search results concisely.",
           "Return only the summarized findings as plain text.",
           "Do not include conversational filler like 'Okay, here are the results...'."
        ],
       show_tool_calls=True, # Show tool calls for debugging
       debug_mode=True
    )
    print("[Worker Agent] Agno core initialized successfully.")
except Exception as e:
    print(f"[Worker Agent] FATAL ERROR initializing Agno agent: {e}")
    # Optionally, exit if the core agent fails to load
    # exit(1)
    web_searcher_agent = None # Ensure it's None if init fails

# --- Wrapper function for Agno Agent ---
def perform_search_task(query: str) -> str:
    """
    Uses the initialized Agno agent to perform a search.
    Returns the text response from the agent.
    """
    if not web_searcher_agent:
        return "Error: Web search agent core is not initialized."
    if not query or not isinstance(query, str):
        return "Error: Search query must be a non-empty string."

    print(f"\n[Worker Agent] Received search task for query: '{query}'")
    try:
        # Use .chat() for conversational interaction leading to tool use
        response = web_searcher_agent.run(query)
        print(f"[Worker Agent] Agno response: {response}")
        return response if isinstance(response, str) else str(response)
    except Exception as e:
        print(f"[Worker Agent] ERROR during Agno agent execution: {e}")
        return f"Error during search: {e}"

# --- FastAPI App for A2A Server ---
app = FastAPI(title=MY_AGENT_CARD.name)

@app.on_event("startup")
async def startup_event():
    """Register with the registry on startup."""
    print(f"\n[Worker Agent] Startup: Attempting registration with registry at {REGISTRY_URL}")
    if not REGISTRY_URL:
        print("[Worker Agent] WARNING: A2A_REGISTRY_URL not set. Skipping registration.")
        return

    # Retry registration a few times in case the registry isn't ready immediately
    max_retries = 3
    retry_delay = 5 # seconds
    for attempt in range(max_retries):
        print(f"[Worker Agent] Registration attempt {attempt + 1}/{max_retries}...")
        result_json = register_with_registry(REGISTRY_URL, MY_AGENT_CARD_JSON)
        try:
            result_data = json.loads(result_json)
            # Check for explicit error key OR if the task status is not completed
            if "error" in result_data or result_data.get("status", {}).get("state") != "completed":
                 print(f"[Worker Agent] WARNING: Registration failed (Attempt {attempt + 1}). Response: {result_json}")
                 if attempt < max_retries - 1:
                     print(f"[Worker Agent] Retrying registration in {retry_delay} seconds...")
                     time.sleep(retry_delay)
                 else:
                     print("[Worker Agent] ERROR: Max registration retries reached.")
            else:
                print("[Worker Agent] Successfully registered with the registry.")
                return # Exit loop on success
        except json.JSONDecodeError:
            print(f"[Worker Agent] WARNING: Could not parse registration response JSON (Attempt {attempt + 1}): {result_json}")
            if attempt < max_retries - 1:
                print(f"[Worker Agent] Retrying registration in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                print("[Worker Agent] ERROR: Max registration retries reached after JSON parse error.")
        except Exception as e:
            print(f"[Worker Agent] ERROR: Unexpected error during registration attempt {attempt + 1}: {e}")
            # Decide if retry makes sense for this error type
            break # Stop retrying on unexpected errors


@app.get("/.well-known/agent.json", response_model=AgentCardModel)
async def get_agent_card():
    """Serves this agent's Agent Card."""
    return JSONResponse(content=MY_AGENT_CARD.model_dump(exclude_none=True))

@app.post("/a2a", summary="A2A JSON-RPC Endpoint for Worker")
async def handle_a2a_task(request: Request):
    """Handles incoming A2A tasks for this worker agent."""
    request_id = None
    try:
        payload = await request.json()
        # print(f"\n[Worker Agent] Received A2A Request:\n{json.dumps(payload, indent=2)}") # Debug
        request_id = payload.get("id")

        # Basic JSON-RPC validation
        if not all(k in payload for k in ["jsonrpc", "method", "params", "id"]):
             return create_jsonrpc_error(-32600, "Invalid Request: Missing JSON-RPC fields.", request_id)
        if payload["jsonrpc"] != "2.0":
             return create_jsonrpc_error(-32600, "Invalid Request: Unsupported JSON-RPC version.", request_id)

        method = payload.get("method")
        params = payload.get("params", {})
        task = params.get("task", {})
        message = params.get("message", {})
        parts = message.get("parts", [])
        task_id = task.get("id", str(uuid.uuid4()))

        if method == "tasks/send":
            # Find the query text within the message parts
            query_text = None
            for part in parts:
                if part.get("type") == "text":
                    query_text = part.get("text")
                    break # Use the first text part found

            if query_text:
                # Check if the agent core is initialized
                if web_searcher_agent is None:
                     print("[Worker Agent] ERROR: Agent core not initialized, cannot process task.")
                     response_task = create_a2a_task_response(task_id, "failed", message_text="Agent core initialization failed.")
                     return create_jsonrpc_response(response_task, request_id)

                # Execute the core Agno logic
                search_result_text = perform_search_task(query_text)

                # Determine task status based on result
                if search_result_text.startswith("Error:"):
                     task_status = "failed"
                     message_text = search_result_text
                     artifacts = []
                else:
                     task_status = "completed"
                     message_text = "Search task completed." # Optional status message
                     # Package the result text into an A2A TextPart artifact
                     artifacts = [{"type": "text", "text": search_result_text}]

                response_task = create_a2a_task_response(task_id, task_status, artifacts=artifacts, message_text=message_text if task_status=="failed" else None)
                return create_jsonrpc_response(response_task, request_id)
            else:
                # No suitable TextPart found for the query
                print(f"[Worker Agent] ERROR: No 'text' part found in message for task {task_id}")
                response_task = create_a2a_task_response(task_id, "failed", message_text="Required 'text' part missing in the input message.")
                return create_jsonrpc_response(response_task, request_id)

        else:
            # Method not supported by this worker
            print(f"[Worker Agent] ERROR: Method '{method}' not implemented.")
            return create_jsonrpc_error(-32601, f"Method '{method}' not implemented by this agent.", request_id, status_code=501)

    except json.JSONDecodeError:
        print("[Worker Agent] ERROR: Invalid JSON payload.")
        return create_jsonrpc_error(-32700, "Parse error: Invalid JSON.", request_id)
    except Exception as e:
        print(f"[Worker Agent] ERROR: Internal server error: {e}")
        task_id_fallback = task.get("id", "unknown-task-id") if 'task' in locals() else "unknown-task-id"
        response_task = create_a2a_task_response(task_id_fallback, "failed", message_text=f"Internal server error processing task: {e}")
        request_id_fallback = request_id if request_id is not None else "unknown-req-id"
        return JSONResponse(status_code=500, content={"jsonrpc": "2.0", "result": response_task, "id": request_id_fallback})

@app.post("/a2a/a2a", summary="A2A JSON-RPC Endpoint for Worker")
async def handle_a2a_task2(request: Request):
    """Handles incoming A2A tasks for this worker agent."""
    request_id = None
    try:
        payload = await request.json()
        # print(f"\n[Worker Agent] Received A2A Request:\n{json.dumps(payload, indent=2)}") # Debug
        request_id = payload.get("id")

        # Basic JSON-RPC validation
        if not all(k in payload for k in ["jsonrpc", "method", "params", "id"]):
             return create_jsonrpc_error(-32600, "Invalid Request: Missing JSON-RPC fields.", request_id)
        if payload["jsonrpc"] != "2.0":
             return create_jsonrpc_error(-32600, "Invalid Request: Unsupported JSON-RPC version.", request_id)

        method = payload.get("method")
        params = payload.get("params", {})
        task = params.get("task", {})
        message = params.get("message", {})
        parts = message.get("parts", [])
        task_id = task.get("id", str(uuid.uuid4()))

        if method == "tasks/send":
            # Find the query text within the message parts
            query_text = None
            for part in parts:
                if part.get("type") == "text":
                    query_text = part.get("text")
                    break # Use the first text part found


            if query_text:
                web_searcher_agent = Agent(
                    model=Groq(id="meta-llama/llama-4-scout-17b-16e-instruct"),  # Using Groq Llama3-8b for speed
                    # model=OpenAIChat(model="gpt-3.5-turbo"), # Or OpenAI
                    description="You are a web search assistant. Use the DuckDuckGo tool to find information.",
                    tools=[DuckDuckGoTools(fixed_max_results=3)],  # Use DuckDuckGo, limit results
                    instructions=[
                        "Understand the user query provided.",
                        "Use the DuckDuckGo search tool to find relevant web pages.",
                        "Summarize the key findings from the search results concisely.",
                        "Return only the summarized findings as plain text.",
                        "Do not include conversational filler like 'Okay, here are the results...'."
                    ],
                    show_tool_calls=True,  # Show tool calls for debugging
                    debug_mode=True
                )
                print("[Worker Agent] Agno core initialized successfully.")
                # Check if the agent core is initialized
                if web_searcher_agent is None:
                     print("[Worker Agent] ERROR: Agent core not initialized, cannot process task.")
                     response_task = create_a2a_task_response(task_id, "failed", message_text="Agent core initialization failed.")
                     return create_jsonrpc_response(response_task, request_id)

                # Execute the core Agno logic
                search_result_text = perform_search_task(query_text)

                # Determine task status based on result
                if search_result_text.startswith("Error:"):
                     task_status = "failed"
                     message_text = search_result_text
                     artifacts = []
                else:
                     task_status = "completed"
                     message_text = "Search task completed." # Optional status message
                     # Package the result text into an A2A TextPart artifact
                     artifacts = [{"type": "text", "text": search_result_text}]

                response_task = create_a2a_task_response(task_id, task_status, artifacts=artifacts, message_text=message_text if task_status=="failed" else None)
                return create_jsonrpc_response(response_task, request_id)
            else:
                # No suitable TextPart found for the query
                print(f"[Worker Agent] ERROR: No 'text' part found in message for task {task_id}")
                response_task = create_a2a_task_response(task_id, "failed", message_text="Required 'text' part missing in the input message.")
                return create_jsonrpc_response(response_task, request_id)

        else:
            # Method not supported by this worker
            print(f"[Worker Agent] ERROR: Method '{method}' not implemented.")
            return create_jsonrpc_error(-32601, f"Method '{method}' not implemented by this agent.", request_id, status_code=501)

    except json.JSONDecodeError:
        print("[Worker Agent] ERROR: Invalid JSON payload.")
        return create_jsonrpc_error(-32700, "Parse error: Invalid JSON.", request_id)
    except Exception as e:
        print(f"[Worker Agent] ERROR: Internal server error: {e}")
        task_id_fallback = task.get("id", "unknown-task-id") if 'task' in locals() else "unknown-task-id"
        response_task = create_a2a_task_response(task_id_fallback, "failed", message_text=f"Internal server error processing task: {e}")
        request_id_fallback = request_id if request_id is not None else "unknown-req-id"
        return JSONResponse(status_code=500, content={"jsonrpc": "2.0", "result": response_task, "id": request_id_fallback})

# --- Run Server ---
if __name__ == "__main__":
    print(f"--- Starting Worker Agent Server ({MY_AGENT_CARD.name}) ---")
    print(f"Worker URL: {WORKER_URL}")
    print(f"Serving own Agent Card at: {WORKER_URL}/.well-known/agent.json")
    print(f"A2A Endpoint: {WORKER_URL}/a2a")
    print(f"Will attempt to register at Registry: {REGISTRY_URL}")
    print(f"----------------------------------------------")
    # Use configured host and port
    uvicorn.run(app, host=WORKER_HOST, port=WORKER_PORT)

# To run: python worker_agent_server.py