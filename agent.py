import asyncio
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_mcp_adapters.tools import load_mcp_tools
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import logging

logging.getLogger("langchain_google_genai").setLevel(logging.ERROR)

MODEL_NAME = "gemini-2.5-flash"
TARGET_URL = "https://uni.sze.hu" 

# --- System Prompt ---
GDPR_AUDITOR_PROMPT = """
You are a GDPR Compliance Auditor. 
Your goal is to audit the provided URL for a Cookie Consent Banner.

**Instructions:**
1. Navigate to the URL.
2. Read the visible text on the page.
3. Analyze the text you retrieved:
   - Is there text mentioning "Cookies", "Privacy", or "Consent"?
   - Are there "Accept" or "Reject" options visible in the text?
4. Produce a FINAL REPORT summarizing your findings.

Do not stop until you have retrieved the text and generated the report.
"""

async def run_audit():
    # 1. Start Chrome MCP Server
    server_params = StdioServerParameters(
        command="npx",
        args=["-y", "chrome-devtools-mcp@latest"],
        env=os.environ.copy(),
    )

    llm = ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        temperature=0,
    )

    print("Connecting to Chrome DevTools MCP...")
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            # Load tools
            tools = await load_mcp_tools(session)
            llm_with_tools = llm.bind_tools(tools)
            print(f"Tools Loaded: {len(tools)} available.")

            # Initial Conversation State
            messages = [
                SystemMessage(content=GDPR_AUDITOR_PROMPT),
                HumanMessage(content=f"Audit this URL: {TARGET_URL}")
            ]

            # --- THE EXECUTION LOOP ---
            max_iterations = 10
            iteration = 0

            print(f"\nStarting Audit for {TARGET_URL}...\n")

            while iteration < max_iterations:
                iteration += 1
                
                # 1. Ask the Agent what to do
                response = await llm_with_tools.ainvoke(messages)
                messages.append(response)

                # 2. Check if the Agent wants to stop (Text Output) or call a tool
                if not response.tool_calls:
                    # Agent is done, it returned text!
                    print("\n" + "="*20 + " GDPR AUDIT REPORT " + "="*20 + "\n")
                    print(response.content)
                    print("\n" + "="*60)
                    break

                # 3. Agent wants to use tools
                print(f"Step {iteration}: Agent wants to use {len(response.tool_calls)} tool(s)...")
                
                for tool_call in response.tool_calls:
                    tool_name = tool_call["name"]
                    tool_args = tool_call["args"]
                    tool_id = tool_call["id"]
                    
                    print(f"   > Executing: {tool_name}")
                    
                    try:
                        # Execute Tool
                        tool_result = await session.call_tool(tool_name, arguments=tool_args)
                        
                        # Parse Content
                        content_str = ""
                        if hasattr(tool_result, 'content') and tool_result.content:
                            for item in tool_result.content:
                                if hasattr(item, 'text'):
                                    content_str += item.text
                        else:
                            content_str = "Tool executed (no text output returned)."

                        # Truncate extremely long HTML/Text to save tokens/sanity
                        if len(content_str) > 20000:
                            content_str = content_str[:20000] + "... [Truncated]"

                    except Exception as e:
                        content_str = f"Error executing tool: {str(e)}"

                    # Add result back to history
                    messages.append(ToolMessage(
                        content=content_str,
                        tool_call_id=tool_id,
                        name=tool_name
                    ))

            if iteration >= max_iterations:
                print("Max iterations reached. Agent got stuck in a loop.")

if __name__ == "__main__":
    try:
        asyncio.run(run_audit())
    except KeyboardInterrupt:
        print("Stopped by user.")