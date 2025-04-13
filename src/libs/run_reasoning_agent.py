# run_reasoning_agent.py
import os
import sys
from dotenv import load_dotenv

# Agno imports
from agno.agent import Agent
from agno.models.groq import Groq
from agno.tools.reasoning import ReasoningTools
from agno.tools.yfinance import YFinanceTools

# Import the single function and config class needed
try:
    from agent_server import serve_agno_agent_a2a, A2AServerConfig, A2ASkill
except ImportError:
     print("ERROR: Failed to import from agent_server.")
     print("Ensure agent_server.py and a2a_utils.py are present.")
     sys.exit(1)

load_dotenv()

# --- Configuration ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
WORKER_URL = os.getenv("WORKER_AGENT_URL", f"http://localhost")

if not GROQ_API_KEY: print("FATAL ERROR: GROQ_API_KEY not set."); sys.exit(1)

# --- 1. Define and Configure the Core Agno Agent (User's Existing Code) ---
print("[FinancialAgent] Initializing core Agno agent...")
try:
    # This part remains exactly as the user defined it previously
    reasoning_agent = Agent(
        model=Groq(id="meta-llama/llama-4-scout-17b-16e-instruct"),
        tools=[
            ReasoningTools(think=True, analyze=True, add_instructions=True),
            YFinanceTools(stock_price=True, analyst_recommendations=True, company_info=True, company_news=True),
        ],
        instructions=[
            "You are a financial analyst.", "Use available tools.", "Provide comprehensive answers.",
            "Structure analysis clearly. Use tables for comparisons.", "Cite sources/tools.",
        ],
        show_tool_calls=True,
        markdown=True,
    )
    print("[FinancialAgent] Core Agno agent initialized.")
except Exception as e:
    print(f"FATAL ERROR: Failed to initialize core Agno agent: {e}"); sys.exit(1)


# --- 2. Define the A2A Server Configuration ---
# User only needs to configure *how* the agent should be served
server_config = A2AServerConfig(
    agent_name="FinancialReasoningAgent_V3", # Unique name for this server instance
    agent_description="Provides financial analysis and comparisons using YFinance and reasoning tools.",
    port=8001, # Ensure port is correct and free
    url=f"{WORKER_URL}",
    # Define the skill(s) this agent server provides
    skills=[
        A2ASkill(
            id="financial_analysis", # The ID orchestrators will search for
            name="Financial Analysis & Comparison",
            description="Analyzes and compares financial instruments like stocks.",
            inputModes=["text"], # Accepts text queries
            outputModes=["text"] # Returns text reports (with markdown)
        )
        # Add more skills here if the agent handles multiple distinct capabilities
    ]
    # registry_url will be picked from .env automatically if set
)


# --- 3. Serve the Agent ---
if __name__ == "__main__":
    print(f"\n[{server_config.agent_name}] Attempting to serve agent via A2A...")
    try:
        # Minimal change: Replace agent interaction (like print_response)
        # with this single function call.
        serve_agno_agent_a2a(
            agent=reasoning_agent,
            config=server_config
            # run_server=True # Default is True, starts blocking server
        )
    except Exception as serve_error:
         print(f"FATAL ERROR during server setup or run: {serve_error}")
         sys.exit(1)

    print(f"[{server_config.agent_name}] Server process finished.") # Only reached if server stops gracefully