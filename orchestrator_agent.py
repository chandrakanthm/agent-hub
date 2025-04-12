# orchestrator_agent.py
import os, sys, json, requests
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv

# --- Agno imports -----------------------------------------------------------
from agno.agent import Agent, RunResponse
from agno.models.groq import Groq

# --- Local tool imports ------------------------------------------------------
try:
    from agno_a2a_tools import call_a2a_agent, discover_agents_from_registry
    print("[Orchestrator] Tools imported from agno_a2a_tools.py")
except Exception as err:
    sys.exit(f"[Orchestrator] FATAL: cannot import A2A tools → {err}")

# --- Environment -------------------------------------------------------------
load_dotenv()
REGISTRY_URL = os.getenv("A2A_REGISTRY_URL")
GROQ_API_KEY  = os.getenv("GROQ_API_KEY")

if not REGISTRY_URL or not GROQ_API_KEY:
    sys.exit("FATAL: A2A_REGISTRY_URL or GROQ_API_KEY missing in environment.")

# --------------------------------------------------------------------------- #
#                               Helper functions                              #
# --------------------------------------------------------------------------- #
def get_registry_context(registry_base_url: str) -> str:
    """Fetch a snapshot of the registry for prompt‑context."""
    header = "== Current Registered Agents Context (fetched at startup) ==\n"
    footer = "== End of Context ==\n"
    if not registry_base_url:
        return header + "Error: registry URL not configured.\n" + footer

    agents_ep = f"{registry_base_url.rstrip('/')}/agents"
    try:
        print(f"[Context] GET {agents_ep}")
        resp = requests.get(agents_ep, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        body = ""
        for url, card in data.get("registered_agents", {}).items():
            name   = card.get("name", "Unnamed Agent")
            skills = ", ".join(
                s.get("id", "no_id") for s in card.get("skills", []) if isinstance(s, dict)
            ) or "None"
            body += f"  - Name: {name}, URL: {url}, Skill IDs: {skills}\n"
        return header + body + footer
    except (requests.RequestException, json.JSONDecodeError) as e:
        return header + f"- Error fetching registry: {e}\n" + footer

# --------------------------------------------------------------------------- #
#                               Orchestrator                                  #
# --------------------------------------------------------------------------- #
class OrchestratorAgent:
    def __init__(self, registry_url: str):
        if not registry_url:
            raise ValueError("Registry URL must be provided.")

        print(f"[Orchestrator] Initialising (registry={registry_url})")

        # Prompt‑context & instructions ---------------------------------------
        context = get_registry_context(registry_url)
        instructions = [
            "Your goal is to satisfy the user by orchestrating specialised agents.",
            "Steps:",
            "1. Decide which skill‑id is required (e.g. 'web_search').",
            "2. If the skill‑id is NOT in the registry‑context, reply with "
            "   'Error: No agent with skill ID \"<skill_id>\" listed in context.'",
            "3. Otherwise call discover_agents_from_registry(registry_url, skill_id).",
            "4. If it returns exactly '[]', reply with "
            "   'Error: Discovery failed for skill ID \"<skill_id>\".'",
            "5. Parse the JSON; take the FIRST agent card, extract its 'url'.",
            "6. Build message_parts **as a Python list of dicts**, e.g. "
            "   `[{'type':'text','text':'<user‑query>'}]`.",
            "7. Call call_a2a_agent(target_url, message_parts).",
            "8. Parse the returned JSON task; if status.state == 'completed' "
            "   present the answer (usually in artifacts). Otherwise report the failure.",
            "Only the two tools provided above may be used.",
        ]
        full_instructions = context + "\n---\n" + "\n".join(instructions)

        # Agent with BOTH tools ----------------------------------------------
        self.agent = Agent(
            model=Groq(id="meta-llama/llama-4-scout-17b-16e-instruct"),
            description="Research Orchestrator Agent",
            instructions=full_instructions,
            tools=[discover_agents_from_registry, call_a2a_agent],
            show_tool_calls=True,
            debug_mode=True,
        )
        print("[Orchestrator] Agent ready.")

    # -----------------------------------------------------------------------
    def run_task(self, user_query: str) -> str:
        if not user_query.strip():
            return "Error: user query cannot be empty."

        print(f"[Orchestrator] ⇢ {user_query}")
        try:
            resp = self.agent.run(user_query)
            if isinstance(resp, RunResponse):
                content = resp.content
            else:                       # safety‑net for raw strings etc.
                content = str(resp)
            print(f"[Orchestrator] ⇠ {content}")
            return content
        except Exception as e:
            print(f"[Orchestrator] ERROR: {e}")
            return f"Error during task execution: {e}"

# --------------------------------------------------------------------------- #
#                               CLI loop                                      #
# --------------------------------------------------------------------------- #
def main() -> None:
    orch = OrchestratorAgent(REGISTRY_URL)
    print("\nType 'quit' to exit.")
    try:
        while True:
            try:
                user_in = input("You: ").strip()
                if user_in.lower() in {"quit", "exit"}:
                    break
                if not user_in:
                    continue
                print("Orchestrator:", orch.run_task(user_in), "\n" + "-" * 20)
            except (EOFError, KeyboardInterrupt):
                break
    finally:
        print("\n[Orchestrator] Bye.")

if __name__ == "__main__":
    main()
