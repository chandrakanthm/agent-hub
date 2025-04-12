# agno_a2a_tools.py
import requests
import json
import uuid
from typing import Dict, Any, List
import os
from dotenv import load_dotenv

load_dotenv() # Load environment variables from .env file

# --- Core A2A Client Tool ---

def call_a2a_agent(target_url: str, message_parts: List[Dict[str, Any]], task_id: str = None) -> str:
    """
    Sends a synchronous A2A task request (tasks/send) to a target A2A agent/server.

    Args:
        target_url (str): The base URL of the target A2A agent (e.g., "http://localhost:8000").
        message_parts (List[Dict[str, Any]]): A list of A2A Part objects for the initial message.
                                           Example: [{"type": "text", "text": "Hello"}]
                                           Example: [{"type": "data", "data": {"skill_id": "X"}}]
        task_id (str, optional): An existing task ID for multi-turn conversation.
                                If None, a new task ID is generated.

    Returns:
        str: A JSON string representation of the resulting A2A Task object, or an error string.
    """
    if not target_url:
        return json.dumps({"error": "Target URL cannot be empty."})
    if not target_url.endswith('/'):
        target_url = target_url.rstrip('/') # Ensure no trailing slash initially
    a2a_endpoint = f"{target_url}/a2a" # Standard A2A endpoint path

    current_task_id = task_id if task_id else str(uuid.uuid4())
    request_id = str(uuid.uuid4()) # JSON-RPC request ID

    payload = {
        "jsonrpc": "2.0",
        "method": "tasks/send",
        "params": {
            "task": {"id": current_task_id},
            "message": {
                "role": "user", # Messages from the client agent are 'user' role
                "parts": message_parts
            }
        },
        "id": request_id
    }

    print(f"\n[A2A Tool] Calling {a2a_endpoint} with Task ID: {current_task_id}")
    print(f"[A2A Tool] Payload: {json.dumps(payload, indent=2)}") # Uncomment for detailed debugging

    try:
        headers = {'Content-Type': 'application/json', 'Accept': 'application/json'}
        response = requests.post(a2a_endpoint, headers=headers, json=payload, timeout=20) # Increased timeout
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

        response_data = response.json()
        # print(f"[A2A Tool] Response Raw: {json.dumps(response_data, indent=2)}") # Uncomment for detailed debugging

        # Basic validation of A2A response structure
        if "result" in response_data and isinstance(response_data["result"], dict) and "id" in response_data["result"]:
            print(f"[A2A Tool] Received valid A2A Task Response for {current_task_id}")
            return json.dumps(response_data["result"]) # Return the Task object as JSON string
        elif "error" in response_data:
            print(f"[A2A Tool] ERROR: A2A Error from {target_url}: {response_data['error']}")
            return json.dumps({"error": response_data['error']}) # Propagate A2A error
        else:
            print(f"[A2A Tool] ERROR: Unexpected A2A response format from {target_url}: {response_data}")
            return json.dumps({"error": "Unexpected response format from A2A server"})

    except requests.exceptions.Timeout:
        print(f"[A2A Tool] ERROR: Timeout connecting to A2A agent at {target_url}")
        return json.dumps({"error": f"Timeout connecting to {target_url}"})
    except requests.exceptions.ConnectionError:
        print(f"[A2A Tool] ERROR: Connection error connecting to A2A agent at {target_url}")
        return json.dumps({"error": f"Could not connect to {target_url}"})
    except requests.exceptions.RequestException as e:
        print(f"[A2A Tool] ERROR: Error calling A2A agent at {target_url}: {e}")
        # Try to get error details from response if available
        error_detail = str(e)
        if e.response is not None:
            try:
                error_detail = e.response.json()
            except json.JSONDecodeError:
                error_detail = e.response.text
        return json.dumps({"error": f"A2A request failed: {error_detail}"})
    except Exception as e:
        print(f"[A2A Tool] ERROR: Unexpected error in call_a2a_agent: {e}")
        return json.dumps({"error": f"An unexpected error occurred: {e}"})

# --- Registration Tool ---

def register_with_registry(registry_url: str, agent_card_json: str) -> str:
    """
    Registers this agent with the specified A2A registry by sending its Agent Card.

    Args:
        registry_url (str): The base URL of the A2A Agent Registry (e.g., http://localhost:8000).
        agent_card_json (str): A JSON string representing the agent's own Agent Card.

    Returns:
        str: A JSON string of the resulting Task object from the registry, or an error string.
    """
    print(f"\n[Registration Tool] Attempting registration with {registry_url}")
    if not registry_url:
        return json.dumps({"error": "Registry URL is required."})
    try:
        agent_card_dict = json.loads(agent_card_json)
        # Basic validation of card structure
        if not isinstance(agent_card_dict, dict) or "url" not in agent_card_dict or "skills" not in agent_card_dict:
             return json.dumps({"error": "Invalid Agent Card structure: 'url' and 'skills' are required."})

        message_parts = [{"type": "data", "data": agent_card_dict}]
        result = call_a2a_agent(target_url=registry_url, message_parts=message_parts)
        print(f"[Registration Tool] Registry response: {result}")
        return result
    except json.JSONDecodeError:
        print("[Registration Tool] ERROR: Invalid agent_card_json format.")
        return json.dumps({"error": "Invalid agent_card_json format provided."})
    except Exception as e:
        print(f"[Registration Tool] ERROR: Failed during registration call: {e}")
        return json.dumps({"error": f"Failed to execute registration: {e}"})

# --- Discovery Tool ---

def discover_agents_from_registry(registry_url: str, required_skill_id: str) -> str:
    """
    Discovers agents with a specific skill ID from the A2A registry.

    Args:
        registry_url (str): The base URL of the A2A Agent Registry (e.g., http://localhost:8000).
        required_skill_id (str): The ID of the skill the desired agent must possess (e.g., 'web_search').

    Returns:
       str: A JSON string representing the LIST of matching Agent Card dictionaries found in the
            registry's task response artifacts. Returns an empty list '[]' if none found.
            Returns a JSON error object string '{"error": ...}' on failure.
            Example successful return: '[{"name": "Agent1",...}, {"name": "Agent2",...}]'
    """
    print(f"\n[Discovery Tool] Attempting discovery for skill '{required_skill_id}' from {registry_url}")
    if not registry_url:
        return json.dumps({"error": "Registry URL is required."})
    if not required_skill_id:
         return json.dumps({"error": "Required skill ID cannot be empty."})
    try:
        # Send skill ID in a DataPart for structured query
        message_parts = [{"type": "data", "data": {"skill_id": required_skill_id}}]
        task_response_json = call_a2a_agent(target_url=registry_url, message_parts=message_parts)

        # Parse the Task response from the registry
        task_response = json.loads(task_response_json)

        # Check if the call itself resulted in an error reported by call_a2a_agent
        if "error" in task_response:
            print(f"[Discovery Tool] ERROR: Call to registry failed: {task_response['error']}")
            return task_response_json # Propagate the error JSON

        # Check the status of the A2A Task returned by the registry
        task_status = task_response.get("status", {}).get("state")
        if task_status != "completed":
            error_msg = task_response.get("status", {}).get("message", {}).get("parts", [{}])[0].get("text", "Registry task did not complete successfully.")
            print(f"[Discovery Tool] ERROR: Registry task status: {task_status}. Message: {error_msg}")
            return json.dumps({"error": f"Registry task failed: {error_msg}", "status": task_status})

        # Extract Agent Cards from artifacts if task completed successfully
        artifacts = task_response.get("artifacts", [])
        discovered_cards = []
        for artifact in artifacts:
            # Expecting artifacts of type 'data' containing the agent card in the 'data' field
            if artifact.get("type") == "data" and isinstance(artifact.get("data"), dict):
                discovered_cards.append(artifact["data"]) # Append the agent card dictionary

        print(f"[Discovery Tool] Discovery result: Found {len(discovered_cards)} agents.")
        return json.dumps(discovered_cards) # Return list of cards as JSON string

    except json.JSONDecodeError:
        print(f"[Discovery Tool] ERROR: Failed to parse registry response JSON: {task_response_json}")
        return json.dumps({"error": "Failed to parse JSON response from registry."})
    except Exception as e:
        print(f"[Discovery Tool] ERROR: Unexpected error during discovery: {e}")
        return json.dumps({"error": f"Unexpected error occurred during agent discovery: {e}"})