"""🏠 MCP Airbnb Agent - Search for Airbnb listings!

This example shows how to create an agent that uses MCP and Gemini 2.5 Pro to search for Airbnb listings.

Run: `pip install google-genai mcp agno` to install the dependencies
"""

import asyncio
# from agno.models.google import Gemini
from agno.tools.mcp import MCPTools
from agno.agent import Agent
from agno.models.groq import Groq



async def run_agent(message: str) -> None:
    async with MCPTools(
        "npx -y @openbnb/mcp-server-airbnb --ignore-robots-txt"
    ) as mcp_tools:
        agent = Agent(
            model=Groq(id="meta-llama/llama-4-scout-17b-16e-instruct"),
            tools=[mcp_tools],
            markdown=True,
        )
        await agent.aprint_response(message, stream=True)


if __name__ == "__main__":
    asyncio.run(
        run_agent(
            "What listings are available in San Francisco for 2 people for 3 nights from 1 to 4 August 2025?"
        )
    )