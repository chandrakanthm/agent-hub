# a2a_utils.py
import datetime
import uuid
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field, ValidationError
from fastapi.responses import JSONResponse

# --- Pydantic Models (Copied/adapted from registry_server.py) ---
class A2ASkill(BaseModel):
    id: str
    name: Optional[str] = None
    description: Optional[str] = None
    inputModes: Optional[List[str]] = None
    outputModes: Optional[List[str]] = None

class A2AAuthentication(BaseModel):
    schemes: List[str]

class A2ACapabilities(BaseModel):
    streaming: Optional[bool] = False
    pushNotifications: Optional[bool] = False
    stateTransitionHistory: Optional[bool] = False

class AgentCardModel(BaseModel):
    name: str
    description: Optional[str] = None
    url: str # A2A endpoint URL
    version: Optional[str] = None
    capabilities: Optional[A2ACapabilities] = Field(default_factory=A2ACapabilities)
    authentication: Optional[A2AAuthentication] = Field(default_factory=lambda: A2AAuthentication(schemes=[]))
    skills: List[A2ASkill] = []
    defaultInputModes: Optional[List[str]] = None
    defaultOutputModes: Optional[List[str]] = None

# --- Helper Functions (Copied/adapted from registry_server.py) ---
def create_a2a_task_response(task_id: str, status: str, artifacts: List[Dict[str, Any]] = None, message_text: str = None) -> Dict[str, Any]:
    """Creates a basic A2A Task response structure."""
    response_task = {
        "id": task_id,
        "status": {
            "state": status,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()
        },
        "artifacts": artifacts if artifacts is not None else [],
        "history": [],
        "metadata": {}
    }
    if message_text:
        response_task["status"]["message"] = {
            "role": "agent",
            "parts": [{"type": "text", "text": message_text}]
        }
    return response_task

def create_jsonrpc_response(result: Any, request_id: str) -> JSONResponse:
    """Creates a standard JSON-RPC 2.0 success response."""
    return JSONResponse(content={"jsonrpc": "2.0", "result": result, "id": request_id})

def create_jsonrpc_error(code: int, message: str, request_id: Optional[str], status_code: int = 400) -> JSONResponse:
    """Creates a standard JSON-RPC 2.0 error response."""
    error_obj = {
        "jsonrpc": "2.0",
        "error": {"code": code, "message": message},
        "id": request_id
    }
    return JSONResponse(content=error_obj, status_code=status_code)