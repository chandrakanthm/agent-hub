# agent_server.py
import uvicorn
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
import json
import os
import time
import uuid
from typing import Optional, List, Dict, Any, Callable
from pydantic import BaseModel, Field, ValidationError
from dotenv import load_dotenv
import inspect # To inspect agent tools later (optional enhancement)

# Agno imports
from agno.agent import Agent
from agno.agent import RunResponse

# Local imports
from a2a_utils import (
    AgentCardModel, A2ASkill, A2ACapabilities, A2AAuthentication,
    create_a2a_task_response, create_jsonrpc_response, create_jsonrpc_error
)
try:
    from agno_a2a_tools import register_with_registry
    REGISTRATION_TOOL_AVAILABLE = True
except ImportError:
    print("WARNING: Cannot import register_with_registry from agno_a2a_tools.py. Registration disabled.")
    REGISTRATION_TOOL_AVAILABLE = False
    def register_with_registry(*args, **kwargs):
        print("ERROR: register_with_registry called but is not available.")
        return json.dumps({"error": "Registration tool unavailable."})

load_dotenv()

# --- Configuration Model ---
class A2AServerConfig(BaseModel):
    """Configuration for serving an Agno Agent via A2A."""
    agent_name: str
    agent_description: str
    host: str = "0.0.0.0"
    port: int = 8001
    registry_url: Optional[str] = Field(default_factory=lambda: os.getenv("A2A_REGISTRY_URL"))
    agent_version: str = "1.0.0"
    # Skill details: Now a list to potentially support multiple skills per server instance
    skills: List[A2ASkill] = [
        A2ASkill(id="chat", name="Chat Processing", description="Handles general chat requests.", inputModes=["text"], outputModes=["text"])
    ]
    # Agent URL will be constructed based on host/port
    agent_url: Optional[str] = None # Will be constructed

    # Optional: You could add fields for default capabilities, auth schemes etc.

    def model_post_init(self, __context: Any) -> None:
        # Construct agent_url after basic validation
        self.agent_url = f"http://{self.host}:{self.port}"
        # Basic validation for skills list
        if not self.skills:
             raise ValueError("At least one skill must be defined in the configuration.")


# --- Internal Helper Logic ---

def _generate_agent_card(config: A2AServerConfig) -> AgentCardModel:
    """Generates the AgentCardModel based on configuration."""
    # Enhancement idea: Introspect agent.tools to auto-populate skills?
    # For now, uses skills defined in config.
    card = AgentCardModel(
        name=config.agent_name,
        description=config.agent_description,
        url=config.agent_url,
        version=config.agent_version,
        capabilities=A2ACapabilities(), # Default capabilities
        authentication=A2AAuthentication(schemes=[]), # Default: no auth
        skills=config.skills, # Use the list of skills from config
        # Optional: Set default modes based on the first skill?
        defaultInputModes=config.skills[0].inputModes if config.skills else ["text"],
        defaultOutputModes=config.skills[0].outputModes if config.skills else ["text"]
    )
    print(f"[{config.agent_name}] Generated Agent Card:")
    print(json.dumps(card.model_dump(exclude_none=True), indent=2))
    return card

async def _handle_tasks_send(
    payload: Dict[str, Any],
    agent: Agent,
    config: A2AServerConfig # Pass config for context
    ) -> JSONResponse:
    """Handles the logic for the A2A 'tasks/send' method."""
    request_id = payload.get("id")
    agent_name = config.agent_name # For logging
    try:
        params = payload.get("params", {})
        task_info = params.get("task", {})
        message = params.get("message", {})
        parts = message.get("parts", [])
        task_id = task_info.get("id", str(uuid.uuid4()))

        # Skill Routing (Basic Example: Assume first skill if multiple defined)
        # A real implementation might inspect message parts or task metadata
        target_skill_id = config.skills[0].id if config.skills else "chat"
        print(f"[{agent_name}] Routing task '{task_id}' to skill '{target_skill_id}'")

        # Extract input (simple text extraction for now)
        input_text = None
        for part in parts:
            if part.get("type") == "text":
                input_text = part.get("text")
                break

        if input_text is not None:
            print(f"\n[{agent_name}] Handling task '{task_id}'. Input: '{input_text[:100]}...'")
            try:
                response = agent.run(input_text) # Call the core Agno agent
                response_content = response.content if isinstance(response, RunResponse) else str(response)
                print(f"[{agent_name}] Agno agent response received for task '{task_id}'.")
                artifacts = [{"type": "text", "text": response_content}]
                response_task = create_a2a_task_response(task_id, "completed", artifacts=artifacts)
                return create_jsonrpc_response(response_task, request_id)
            except Exception as agent_error:
                print(f"[{agent_name}] ERROR executing Agno agent task '{task_id}': {agent_error}")
                response_task = create_a2a_task_response(task_id, "failed", message_text=f"Agent execution error: {agent_error}")
                return create_jsonrpc_response(response_task, request_id)
        else:
            print(f"[{agent_name}] ERROR: No 'text' part found for task '{task_id}'")
            response_task = create_a2a_task_response(task_id, "failed", message_text="Required 'text' part missing.")
            return create_jsonrpc_response(response_task, request_id)
    except Exception as e:
        print(f"[{agent_name}] ERROR processing 'tasks/send' for request '{request_id}': {e}")
        task_id_fallback = task_info.get("id", "unknown") if 'task_info' in locals() else "unknown"
        response_task = create_a2a_task_response(task_id_fallback, "failed", message_text=f"Internal error processing tasks/send: {e}")
        return create_jsonrpc_response(response_task, request_id)


async def _register_agent_on_startup(config: A2AServerConfig, agent_card_json: str):
    """Handles the agent registration logic on server startup."""
    # (Logic is identical to previous version - using config object now)
    agent_name = config.agent_name
    print(f"\n[{agent_name}] Startup Event: Attempting registration...")
    if not config.registry_url: print(f"[{agent_name}] Skipping: Registry URL not set."); return
    if not REGISTRATION_TOOL_AVAILABLE: print(f"[{agent_name}] Skipping: Registration tool unavailable."); return
    max_retries = 3; retry_delay = 5
    for attempt in range(max_retries):
        print(f"[{agent_name}] Reg attempt {attempt + 1}/{max_retries} to {config.registry_url}...")
        result_json = register_with_registry(config.registry_url, agent_card_json)
        try:
            result_data = json.loads(result_json); task_status = result_data.get("status", {}).get("state")
            if "error" in result_data or task_status != "completed":
                 print(f"[{agent_name}] WARNING: Reg failed (Attempt {attempt + 1}). Resp: {result_json}")
                 if attempt < max_retries - 1: time.sleep(retry_delay)
                 else: print(f"[{agent_name}] ERROR: Max registration retries.")
            else: print(f"[{agent_name}] Successfully registered."); return
        except Exception as e: print(f"[{agent_name}] ERROR processing reg resp (Attempt {attempt + 1}): {e}. Resp: {result_json}"); break


def _create_a2a_app(
    agent: Agent,
    config: A2AServerConfig,
    agent_card: AgentCardModel
    ) -> FastAPI:
    """Creates the FastAPI application using internal logic functions."""
    app = FastAPI(title=config.agent_name)
    agent_card_dict = agent_card.model_dump(exclude_none=True)
    agent_card_json_str = agent_card.model_dump_json(exclude_none=True)

    # --- Startup Event Handler ---
    @app.on_event("startup")
    async def startup_event_handler():
        await _register_agent_on_startup(config, agent_card_json_str)

    # --- Agent Card Endpoint ---
    @app.get("/.well-known/agent.json", response_model=AgentCardModel)
    async def get_agent_card_endpoint():
        return agent_card_dict # Return pre-generated dict

    # --- A2A Task Endpoint ---
    @app.post("/a2a")
    async def handle_a2a_task_endpoint(request: Request):
        request_id = None; payload = {}
        try:
            payload = await request.json(); request_id = payload.get("id")
            if not all(k in payload for k in ["jsonrpc", "method", "params", "id"]) or payload.get("jsonrpc") != "2.0":
                 return create_jsonrpc_error(-32600, "Invalid Request.", request_id)
            method = payload.get("method")
            # --- Route to specific method handlers ---
            if method == "tasks/send":
                # Pass necessary objects to the handler
                return await _handle_tasks_send(payload, agent, config)
            else:
                print(f"[{config.agent_name}] ERROR: Method '{method}' not implemented.")
                return create_jsonrpc_error(-32601, f"Method '{method}' not implemented.", request_id, status_code=501)
        # --- General Error Handling ---
        except json.JSONDecodeError: return create_jsonrpc_error(-32700, "Parse error.", request_id)
        except Exception as e:
            print(f"[{config.agent_name}] ERROR: Internal server error: {e}")
            task_id_fallback = payload.get("params", {}).get("task", {}).get("id", "unknown")
            response_task = create_a2a_task_response(task_id_fallback, "failed", message_text=f"Internal server error: {e}")
            request_id_fallback = request_id if request_id is not None else "unknown"
            return JSONResponse(status_code=500, content={"jsonrpc": "2.0", "result": response_task, "id": request_id_fallback})

    print(f"[{config.agent_name}] FastAPI app created.")
    return app

# --- The Public User-Facing Function ---

def serve_agno_agent_a2a(
    agent: Agent,
    config: A2AServerConfig,
    run_server: bool = True
    ) -> Optional[FastAPI]:
    """
    Takes a pre-configured Agno Agent and serves it as an A2A agent,
    optionally registering it with the AgentHub registry.

    This is the primary function users call in their agent scripts.

    Args:
        agent: The instantiated and configured agno.agent.Agent object.
        config: An A2AServerConfig object detailing how to serve the agent.
        run_server: If True (default), starts the Uvicorn server and blocks.
                    If False, creates and returns the FastAPI app instance
                    (useful for testing or embedding).

    Returns:
        The FastAPI app instance if run_server is False, otherwise None.
    """
    if not isinstance(agent, Agent):
        raise TypeError("The 'agent' argument must be an instance of agno.agent.Agent")
    if not isinstance(config, A2AServerConfig):
         raise TypeError("The 'config' argument must be an instance of A2AServerConfig")

    print(f"\n--- Preparing A2A Server for Agent: {config.agent_name} ---")

    # 1. Generate the Agent Card
    try:
        agent_card = _generate_agent_card(config)
    except Exception as e:
        print(f"FATAL ERROR: Failed to generate Agent Card: {e}"); return None

    # 2. Create the FastAPI App
    try:
        fastapi_app = _create_a2a_app(
            agent=agent,
            config=config,
            agent_card=agent_card
            )
    except Exception as e:
         print(f"FATAL ERROR: Failed to create FastAPI app: {e}"); return None

    # 3. Run the server (or return app)
    if run_server:
        print(f"[{config.agent_name}] Starting Uvicorn server on {config.host}:{config.port}...")
        try:
            # reload=True might be useful during development
            uvicorn.run(fastapi_app, host=config.host, port=config.port, reload=False)
            print(f"[{config.agent_name}] Server stopped.")
        except Exception as e:
             print(f"[{config.agent_name}] Error running Uvicorn server: {e}")
        return None # Return None because server run blocks
    else:
        print(f"[{config.agent_name}] FastAPI app created but server not started (run_server=False).")
        return fastapi_app # Return the app instance for testing/embedding