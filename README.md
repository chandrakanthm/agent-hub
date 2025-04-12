# AgentHub
## Agent Registry with A2A Discovery

## Overview
Solving the "how do agents find each other?" problem in Agno-based multi-agent systems. AgentHub implements a central discovery service – the "Hub" – where agents register using A2A Agent Card formats. This allows complex workflows where an orchestrator can dynamically query the Hub by required skill (e.g., 'web_search'), retrieve the necessary agent's details, and then interact with it directly, promoting decoupled and scalable agent architectures.

This project demonstrates a multi-agent system built using the [Agno framework](https://github.com/agno-agi/agno). It addresses the challenge of dynamic agent discovery by implementing a central **Agent Registry** service. Agents can register their capabilities (described using Google's [Agent2Agent (A2A) protocol](https://github.com/google/A2A) concepts, specifically the Agent Card structure) with the registry. Other agents can then query the registry to discover agents possessing specific skills needed to accomplish complex tasks.

The system consists of three main components running as separate processes:

1.  **Registry Server (`registry_server.py`):** An A2A-compliant server acting as the central directory. It handles agent registration and discovery requests.
2.  **Worker Agent Server (`worker_agent_server.py`):** An example Agno agent (e.g., Web Search) wrapped in an A2A server. It registers itself with the Registry upon startup and listens for tasks matching its skills.
3.  **Orchestrator Agent (`orchestrator_agent.py`):** An Agno agent that takes user requests, interacts with the Registry to discover appropriate Worker Agents, and then communicates directly with those Workers via A2A calls to delegate tasks and synthesize results.

## Features

*   **Dynamic Agent Discovery:** Agents aren't hardcoded; they find each other via the registry based on required skills.
*   **A2A Protocol Concepts:** Uses A2A Agent Card structure for registration and A2A Task semantics for communication (via custom Agno tools).
*   **Agno Framework:** Leverages the lightweight and efficient Agno library for building the core agent logic (LLM interaction, tool use).
*   **Modular Design:** Components run independently, allowing for easier scaling and modification.
*   **Example Workflow:** Demonstrates a research task involving web search coordinated by the orchestrator.

## Architecture

```mermaid
sequenceDiagram
    participant User
    participant Orchestrator (orchestrator_agent.py)
    participant Registry (registry_server.py)
    participant Worker (worker_agent_server.py)

    User->>+Orchestrator: "Find info on X"
    Orchestrator->>Orchestrator: Identify needed skill (e.g., 'web_search')
    Orchestrator->>+Registry: Discover(skill='web_search') via A2A Tool (HTTP POST /a2a)
    Registry->>Registry: Query internal storage
    Registry-->>-Orchestrator: Return Worker Agent Card(s) in A2A Task Response
    Orchestrator->>Orchestrator: Extract Worker URL from Card
    Orchestrator->>+Worker: Perform Search Task('X') via A2A Tool (HTTP POST /a2a to Worker URL)
    Worker->>Worker: Use Agno core + Tool (e.g., DuckDuckGo)
    Worker-->>-Orchestrator: Return Search Results in A2A Task Response
    Orchestrator->>Orchestrator: Process results
    Orchestrator-->>-User: Synthesized answer

    Note over Worker,Registry: On Startup
    Worker->>Registry: Register(My Agent Card) via A2A Tool (HTTP POST /a2a)
```
### Prerequisites

- **Python**: Version 3.8 or higher  
- **pip**: Python package installer (bundled with most Python installs)  
- **Git**: *Optional*—needed only if you’re cloning a repository  
- **LLM API key**: From your chosen provider (e.g., Groq, OpenAI, Anthropic)

---

### Setup Instructions

#### 1. Get the code

**Clone the repository (if applicable):**

```bash
git clone <repository-url>
cd <repository-directory>
```

Or manually create the files `agno_a2a_tools.py`, `registry_server.py`, `worker_agent_server.py`, and `orchestrator_agent.py`, then paste the provided code into each file.

---

#### 2. Create a virtual environment (recommended)

```bash
python -m venv .venv
# Activate the environment:
#   On Windows:
#     .venv\Scripts\activate
#   On macOS/Linux:
#     source .venv/bin/activate
```

---

#### 3. Install dependencies

Create a **`requirements.txt`** file in the project root with:

```text
agno
fastapi
uvicorn[standard]   # Installs standard server dependencies like watchfiles
requests
python-dotenv
groq                # Or openai, anthropic—match your LLM provider
duckduckgo-search   # Example worker‑agent tool
pydantic
```

Then install everything:

```bash
pip install -r requirements.txt
```

---

#### 4. Configure environment variables

Create a **`.env`** file in the project root (never commit secrets to Git) and add:

```dotenv
# --- API Keys (choose one or more, based on the model you use) ---
# Get from https://console.groq.com/keys
GROQ_API_KEY=gsk_YOUR_GROQ_API_KEY

# Get from https://platform.openai.com/api-keys
# OPENAI_API_KEY=sk-YOUR_OPENAI_API_KEY

# Get from Anthropic console
# ANTHROPIC_API_KEY=sk-ant-YOUR_ANTHROPIC_API_KEY

# --- Service URLs (defaults for local running) ---
# URL for the Agent Registry
A2A_REGISTRY_URL=http://localhost:8000

# URL the Worker Agent will listen on and register itself as
WORKER_AGENT_URL=http://localhost:8001

# Optional: Port overrides if defaults are taken
# REGISTRY_PORT=8000
# WORKER_PORT=8001
```

> **Important:** Use the API key that matches the LLM specified in your agent scripts—for example, `Groq(model="...")` in `worker_agent_server.py` and `orchestrator_agent.py` requires `GROQ_API_KEY`.
