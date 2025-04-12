# registry_server.py
import uvicorn
from fastapi import FastAPI, Request, HTTPException, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ValidationError
from typing import Dict, List, Any, Optional
import json
import uuid
import datetime
import time
import os
from dotenv import load_dotenv

load_dotenv()

# --- Simple In-Memory Storage with TTL ---
# In production, replace this with a database (e.g., PostgreSQL, Redis)
registered_agents: Dict[str, Dict[str, Any]] = {}
agent_last_seen: Dict[str, float] = {}
AGENT_TTL_SECONDS = 300 # Agent entry expires after 5 minutes of inactivity

# --- Pydantic Models for Basic Validation ---
# Based loosely on A2A Spec - real validation would be more thorough
class A2ASkill(BaseModel):
    id: str
    name: Optional[str] = None
    description: Optional[str] = None
    # Add other fields as needed: inputModes, outputModes, examples...

class A2AAuthentication(BaseModel):
    schemes: List[str] # e.g., ["apiKey", "bearer"]

class A2ACapabilities(BaseModel):
    streaming: Optional[bool] = False
    pushNotifications: Optional[bool] = False
    stateTransitionHistory: Optional[bool] = False

class AgentCardModel(BaseModel):
    name: str
    description: Optional[str] = None
    url: str # This will be the key in our registry dict
    version: Optional[str] = None
    capabilities: Optional[A2ACapabilities] = None
    authentication: Optional[A2AAuthentication] = None
    skills: List[A2ASkill] = []
    # Add other fields like defaultInputModes, defaultOutputModes if needed

class A2APart(BaseModel):
    type: Optional[str] = 'text' # 'text', 'data', 'file'
    text: Optional[str] = None
    data: Optional[Any] = None # Could be dict, list, etc.
    # file fields omitted for simplicity

class A2AMessage(BaseModel):
    role: str # 'user', 'agent'
    parts: List[A2APart]

class A2ATask(BaseModel):
    id: str

class A2AParams(BaseModel):
    task: A2ATask
    message: A2AMessage

class A2ARequest(BaseModel):
    jsonrpc: str = "2.0"
    method: str
    params: A2AParams
    id: str # JSON-RPC request ID

# --- Registry Configuration ---
REGISTRY_HOST = "0.0.0.0"
REGISTRY_PORT = int(os.getenv("REGISTRY_PORT", 8000))
REGISTRY_URL = os.getenv("A2A_REGISTRY_URL", f"http://localhost:{REGISTRY_PORT}")

# --- Registry's Own Agent Card ---
REGISTRY_AGENT_CARD = {
 "name": "AgentDiscoveryRegistry",
 "description": "A central registry for discovering A2A-compliant agents. Supports registration and discovery by skill ID.",
 "url": f"{REGISTRY_URL}/a2a", # Use configured URL
 "version": "1.0.0",
 "capabilities": {"streaming": False, "pushNotifications": False, "stateTransitionHistory": False},
 "authentication": {"schemes": []}, # Example: No authentication for simplicity
 "skills": [
   {
     "id": "register_agent",
     "name": "Register Agent",
     "description": "Registers an agent by accepting its Agent Card in a DataPart.",
     "inputModes": ["application/json"], # Expects Agent Card in DataPart
     "outputModes": ["text"] # Simple success/failure message
   },
   {
     "id": "discover_agents",
     "name": "Discover Agents",
     "description": "Discovers agents based on skill ID provided in a DataPart {'skill_id': '...'}.",
     "inputModes": ["application/json"], # Expects query criteria in DataPart
     "outputModes": ["application/json"] # Returns list of Agent Cards as DataPart artifacts
   }
 ]
}

# --- FastAPI App ---
app = FastAPI(title="A2A Agent Registry")

# --- Helper Functions ---
def create_a2a_task_response(task_id: str, status: str, artifacts: List[Dict[str, Any]] = None, message_text: str = None) -> Dict[str, Any]:
    """Creates a basic A2A Task response structure."""
    response_task = {
        "id": task_id,
        "status": {
            "state": status,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()
        },
        "artifacts": artifacts if artifacts is not None else [],
        "history": [], # Keep history simple for this example
        "metadata": {}
    }
    if message_text:
        # Ensure message is structured correctly according to A2A spec
        response_task["status"]["message"] = {
            "role": "agent",
            "parts": [{"type": "text", "text": message_text}]
        }
    return response_task

def prune_expired_agents():
    """Removes agents that haven't checked in within the TTL."""
    now = time.time()
    expired_urls = [url for url, last_seen in agent_last_seen.items() if now - last_seen > AGENT_TTL_SECONDS]
    if expired_urls:
        print(f"\n[Registry] Pruning {len(expired_urls)} expired agents.")
        for url in expired_urls:
            if url in registered_agents:
                del registered_agents[url]
            if url in agent_last_seen:
                del agent_last_seen[url]
            print(f"[Registry] Pruned agent: {url}")

def create_jsonrpc_response(result: Any, request_id: str) -> JSONResponse:
    return JSONResponse(content={"jsonrpc": "2.0", "result": result, "id": request_id})

def create_jsonrpc_error(code: int, message: str, request_id: Optional[str], status_code: int = 400) -> JSONResponse:
    error_obj = {
        "jsonrpc": "2.0",
        "error": {"code": code, "message": message},
        "id": request_id
    }
    return JSONResponse(content=error_obj, status_code=status_code)


# --- A2A Endpoints ---
@app.get("/.well-known/agent.json", response_model=AgentCardModel)
async def get_registry_agent_card():
    """Serves the registry's own Agent Card."""
    return JSONResponse(content=REGISTRY_AGENT_CARD)

@app.get("/agents", summary="List Registered Agents (Debug)")
async def list_registered_agents():
    """Debug endpoint to view currently registered agents."""
    prune_expired_agents()
    return JSONResponse(content={"registered_agents": registered_agents})

@app.post("/a2a", summary="A2A JSON-RPC Endpoint")
async def handle_a2a_request(request: Request):
    """Handles incoming A2A JSON-RPC requests for registration and discovery."""
    prune_expired_agents() # Prune before processing request
    request_id = None # Keep track of JSON-RPC request ID
    try:
        payload = await request.json()
        # print(f"\n[Registry] Received A2A Request:\n{json.dumps(payload, indent=2)}") # Debug
        request_id = payload.get("id")

        # Validate basic JSON-RPC structure
        if not all(k in payload for k in ["jsonrpc", "method", "params", "id"]):
             return create_jsonrpc_error(-32600, "Invalid Request: Missing JSON-RPC fields.", request_id)

        if payload["jsonrpc"] != "2.0":
             return create_jsonrpc_error(-32600, "Invalid Request: Unsupported JSON-RPC version.", request_id)

        method = payload.get("method")
        params = payload.get("params", {})
        task = params.get("task", {})
        message = params.get("message", {})
        parts = message.get("parts", [])
        task_id = task.get("id", str(uuid.uuid4())) # Use provided task ID or generate one

        if method == "tasks/send":
            skill_to_invoke = None
            query_data = None
            agent_card_to_register_data = None

            # Look for DataParts first
            for part in parts:
                if part.get("type") == "data":
                    data_content = part.get("data", {})
                    if isinstance(data_content, dict):
                        # Heuristic for Agent Card (Registration)
                        if "url" in data_content and "skills" in data_content and "name" in data_content:
                            agent_card_to_register_data = data_content
                            skill_to_invoke = "register_agent"
                            print(f"[Registry] Detected Registration request for Task ID: {task_id}")
                            break # Found registration data
                        # Heuristic for Discovery Query
                        elif "skill_id" in data_content:
                            query_data = data_content
                            skill_to_invoke = "discover_agents"
                            print(f"[Registry] Detected Discovery request for Task ID: {task_id}")
                            # Don't break yet, maybe a more specific data part exists later? Usually not needed.
                            break # Found discovery data

            # --- Registration Logic ---
            if skill_to_invoke == "register_agent" and agent_card_to_register_data:
                try:
                    # Validate the received agent card data using Pydantic
                    agent_card = AgentCardModel.model_validate(agent_card_to_register_data)
                    agent_url = agent_card.url

                    # Use the validated data (agent_card.model_dump()) for storage
                    registered_agents[agent_url] = agent_card.model_dump(exclude_none=True)
                    agent_last_seen[agent_url] = time.time() # Update last seen time

                    print(f"[Registry] Registered/Updated agent: {agent_card.name} at {agent_url}")
                    response_task = create_a2a_task_response(task_id, "completed", message_text="Agent registered successfully.")
                    return create_jsonrpc_response(response_task, request_id)

                except ValidationError as e:
                    print(f"[Registry] ERROR: Invalid Agent Card format for registration: {e}")
                    response_task = create_a2a_task_response(task_id, "failed", message_text=f"Invalid Agent Card format: {e}")
                    # Return success at JSON-RPC level but failure at Task level
                    return create_jsonrpc_response(response_task, request_id)
                except Exception as e:
                     print(f"[Registry] ERROR: Internal error during registration: {e}")
                     # Handle unexpected errors during registration processing
                     response_task = create_a2a_task_response(task_id, "failed", message_text=f"Internal server error during registration: {e}")
                     return create_jsonrpc_response(response_task, request_id)


            # --- Discovery Logic ---
            elif skill_to_invoke == "discover_agents" and query_data:
                required_skill_id = query_data.get("skill_id")
                if not required_skill_id or not isinstance(required_skill_id, str):
                    response_task = create_a2a_task_response(task_id, "failed", message_text="Invalid discovery query: 'skill_id' (string) is required in data part.")
                    return create_jsonrpc_response(response_task, request_id)

                print(f"[Registry] Processing discovery for skill: '{required_skill_id}'")
                matching_cards_artifacts = []
                for agent_url, card_dict in registered_agents.items():
                    # Ensure skills is a list before iterating
                    agent_skills = card_dict.get("skills", [])
                    if isinstance(agent_skills, list):
                        for skill in agent_skills:
                            if isinstance(skill, dict) and skill.get("id") == required_skill_id:
                                # Wrap the matching agent card dict in an A2A DataPart structure for the artifact list
                                matching_cards_artifacts.append({"type": "data", "data": card_dict})
                                print(f"[Registry] Found match: {card_dict.get('name')} at {agent_url}")
                                break # Agent matches, no need to check other skills of the same agent

                print(f"[Registry] Discovery for skill '{required_skill_id}': Found {len(matching_cards_artifacts)} agents.")
                response_task = create_a2a_task_response(task_id, "completed", artifacts=matching_cards_artifacts)
                return create_jsonrpc_response(response_task, request_id)

            else:
                # If no specific skill was determined from DataParts
                print(f"[Registry] ERROR: Could not determine intended operation (register/discover) or missing required data in message parts for Task ID {task_id}.")
                response_task = create_a2a_task_response(task_id, "failed", message_text="Could not determine intended operation or missing required data. Use DataPart for agent card (register) or {'skill_id': '...'} (discover).")
                return create_jsonrpc_response(response_task, request_id)

        else:
            # Method not implemented (e.g., tasks/sendSubscribe, tasks/cancel)
            print(f"[Registry] ERROR: Method '{method}' not implemented.")
            # Send JSON-RPC error for unsupported method
            return create_jsonrpc_error(-32601, f"Method '{method}' not implemented.", request_id, status_code=501)

    except json.JSONDecodeError:
        print("[Registry] ERROR: Invalid JSON payload.")
        return create_jsonrpc_error(-32700, "Parse error: Invalid JSON.", request_id)
    except HTTPException as e:
        # Let FastAPI handle its own HTTPExceptions
        raise e
    except Exception as e:
        print(f"[Registry] ERROR: Internal server error: {e}")
        # Attempt to return a failed task response within JSON-RPC result field for unexpected errors
        # If task_id wasn't parsed, use a placeholder
        task_id_fallback = task.get("id", "unknown-task-id") if 'task' in locals() else "unknown-task-id"
        response_task = create_a2a_task_response(task_id_fallback, "failed", message_text=f"Internal server error: {e}")
        # Ensure request_id is available, fallback if needed
        request_id_fallback = request_id if request_id is not None else "unknown-req-id"

        # Best effort to return a valid JSON-RPC response containing the failed task
        # Use status code 500 for internal server errors
        return JSONResponse(status_code=500, content={"jsonrpc": "2.0", "result": response_task, "id": request_id_fallback})


# --- Run Server ---
if __name__ == "__main__":
    print(f"--- Starting A2A Agent Registry Server ---")
    print(f"Registry URL: {REGISTRY_URL}")
    print(f"Serving own Agent Card at: {REGISTRY_URL}/.well-known/agent.json")
    print(f"A2A Endpoint: {REGISTRY_URL}/a2a")
    print(f"Agent TTL: {AGENT_TTL_SECONDS} seconds")
    print(f"Debug Endpoint (List Agents): {REGISTRY_URL}/agents")
    print(f"-----------------------------------------")
    # Use configured host and port
    uvicorn.run(app, host=REGISTRY_HOST, port=REGISTRY_PORT)

# To run: python registry_server.py