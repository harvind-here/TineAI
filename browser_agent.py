from openai import OpenAI
import os
from pydantic import BaseModel, Field, ValidationError
from typing import Optional, List
import json
import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from dotenv import load_dotenv

load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY not found in environment variables")

# Define Playwright MCP tools with CORRECT parameters
class BrowserNavigateTool(BaseModel):
    """Navigate to a URL in the browser"""
    url: str = Field(..., description="The URL to navigate to")

class BrowserSnapshotTool(BaseModel):
    """Capture accessibility snapshot of the current page to see current elements"""
    pass

class BrowserClickTool(BaseModel):
    """Click an element on the page"""
    element: str = Field(..., description="Human-readable element description")
    ref: str = Field(..., description="Exact target element reference from the page snapshot (e.g., e123)")
    doubleClick: Optional[bool] = Field(default=None, description="Whether to perform a double click")

class BrowserTypeTool(BaseModel):
    """Type text into an element"""
    element: str = Field(..., description="Human-readable element description")
    ref: str = Field(..., description="Exact target element reference from the page snapshot (e.g., e123)")
    text: str = Field(..., description="Text to type")
    submit: Optional[bool] = Field(default=None, description="Whether to submit (press Enter after)")

class BrowserFillFormTool(BaseModel):
    """Fill multiple form fields at once"""
    fields: List[dict] = Field(..., description="Array of form fields to fill")

class BrowserTakeScreenshotTool(BaseModel):
    """Take a screenshot of the current page"""
    filename: Optional[str] = Field(default=None, description="File name to save screenshot")
    fullPage: Optional[bool] = Field(default=None, description="Capture full scrollable page")

class BrowserCloseTool(BaseModel):
    """Close the browser"""
    pass

class BrowserWaitForTool(BaseModel):
    """Wait for text to appear or time to pass"""
    time: Optional[int] = Field(default=None, description="Time to wait in seconds")
    text: Optional[str] = Field(default=None, description="Text to wait for")

def pydantic_to_openai_schema(model: type[BaseModel]) -> dict:
    schema = model.model_json_schema()
    class_name = model.__name__.replace("Tool", "")
    tool_name = ''.join(['_'+c.lower() if c.isupper() else c for c in class_name]).lstrip('_')
    
    return {
        "type": "function",
        "function": {
            "name": tool_name,
            "description": model.__doc__ or "",
            "parameters": {
                "type": "object",
                "properties": schema.get("properties", {}),
                "required": schema.get("required", []),
            }
        }
    }

tools = [
    pydantic_to_openai_schema(BrowserNavigateTool),
    pydantic_to_openai_schema(BrowserSnapshotTool),
    pydantic_to_openai_schema(BrowserClickTool),
    pydantic_to_openai_schema(BrowserTypeTool),
    pydantic_to_openai_schema(BrowserFillFormTool),
    pydantic_to_openai_schema(BrowserTakeScreenshotTool),
    pydantic_to_openai_schema(BrowserCloseTool),
    pydantic_to_openai_schema(BrowserWaitForTool),
]

tool_models = {
    "browser_navigate": BrowserNavigateTool,
    "browser_snapshot": BrowserSnapshotTool,
    "browser_click": BrowserClickTool,
    "browser_type": BrowserTypeTool,
    "browser_fill_form": BrowserFillFormTool,
    "browser_take_screenshot": BrowserTakeScreenshotTool,
    "browser_close": BrowserCloseTool,
    "browser_wait_for": BrowserWaitForTool,
}

# Global MCP session management
class MCPSessionManager:
    def __init__(self):
        self.session = None
        self.read = None
        self.write = None
        self.client_context = None
        self.session_context = None
    
    async def initialize(self):
        """Initialize persistent MCP connection"""
        if self.session is None:
            server_params = StdioServerParameters(
                command="npx",
                args=["-y", "@playwright/mcp@latest"],
                env=None
            )
            
            self.client_context = stdio_client(server_params)
            self.read, self.write = await self.client_context.__aenter__()
            
            self.session_context = ClientSession(self.read, self.write)
            self.session = await self.session_context.__aenter__()
            await self.session.initialize()
            
            print("✅ Initialized persistent browser session")
    
    async def execute_tool(self, tool_name: str, args: dict):
        """Execute a tool on the persistent MCP session"""
        await self.initialize()
        result = await self.session.call_tool(tool_name, arguments=args)
        return result
    
    async def close(self):
        """Close the MCP session"""
        if self.session_context:
            await self.session_context.__aexit__(None, None, None)
        if self.client_context:
            await self.client_context.__aexit__(None, None, None)
        self.session = None
        print("🔒 Closed browser session")

# Global session manager
mcp_manager = MCPSessionManager()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

async def run_agent_loop(user_query: str, max_iterations: int = 20):
    """Main agent loop that handles tool calls iteratively"""
    messages = [
        {
            "role": "system",
            "content": """You are a browser automation agent with Playwright MCP tools.

CRITICAL RULES:
1. ALWAYS call browser_snapshot IMMEDIATELY after clicking buttons or navigating to see the NEW page state
2. NEVER reuse old 'ref' values - ALWAYS get fresh refs from the latest snapshot
3. If you see "Ref eXXX not found", it means the element is stale - call browser_snapshot to get new refs
4. For modals/popups: After clicking a button that should open a modal, IMMEDIATELY call browser_snapshot to verify it opened
5. If a modal didn't open, try clicking the button again or try a different ref
6. After typing text, call browser_snapshot to see if suggestions/dropdown appeared
7. For optional boolean fields (doubleClick, submit), either omit them entirely or set to true/false
8. If you see the same error 3 times in a row, try a different approach or acknowledge failure
9. Element refs change frequently - treat every snapshot as the source of truth"""
        },
        {
            "role": "user",
            "content": user_query
        }
    ]
    
    iteration = 0
    consecutive_errors = 0
    last_error = None
    
    while iteration < max_iterations:
        iteration += 1
        print(f"\n{'='*60}")
        print(f"Iteration {iteration}/{max_iterations}")
        print(f"{'='*60}")
        
        try:
            # Call LLM
            completion = client.chat.completions.create(
                model="google/gemini-2.5-flash-lite-preview-09-2025",
                messages=messages,
                tools=tools,
                tool_choice="auto",
            )
            
            # Handle None response
            if completion is None or completion.choices is None or len(completion.choices) == 0:
                print("⚠️  API returned empty response. Retrying...")
                await asyncio.sleep(2)
                continue
            
            response_message = completion.choices[0].message
            
            # Add assistant's response to messages
            message_dict = {
                "role": "assistant",
                "content": response_message.content,
            }
            
            if response_message.tool_calls:
                message_dict["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": tc.type,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    } for tc in response_message.tool_calls
                ]
            
            messages.append(message_dict)
            
            # If no tool calls, we're done
            if not response_message.tool_calls:
                print("\n✅ Agent completed task")
                if response_message.content:
                    print(f"\n📋 Final Response:\n{response_message.content}")
                break
            
            # Execute all tool calls
            print(f"\n🔧 Executing {len(response_message.tool_calls)} tool call(s)...")
            
            for tool_call in response_message.tool_calls:
                function_name = tool_call.function.name
                arguments_json = tool_call.function.arguments
                
                print(f"\n  📌 Tool: {function_name}")
                
                if function_name in tool_models:
                    try:
                        # Validate arguments
                        arguments = json.loads(arguments_json)
                        validated_args = tool_models[function_name](**arguments)
                        
                        # CRITICAL: Exclude None values to avoid sending null
                        clean_args = validated_args.model_dump(exclude_none=True)
                        
                        print(f"  📝 Arguments: {json.dumps(clean_args, indent=4)}")
                        
                        # Execute tool via persistent MCP session
                        result = await mcp_manager.execute_tool(function_name, clean_args)
                        
                        # Extract result content
                        result_text = ""
                        if hasattr(result, 'content') and result.content:
                            result_text = "\n".join([c.text for c in result.content if hasattr(c, 'text')])
                        
                        # Check if this is the same error as before
                        if "not found" in result_text.lower() or "error" in result_text.lower():
                            if result_text == last_error:
                                consecutive_errors += 1
                                if consecutive_errors >= 3:
                                    result_text += "\n\n⚠️ CRITICAL: Same error repeated 3 times. Try a completely different approach or acknowledge the task cannot be completed."
                                    consecutive_errors = 0
                            else:
                                consecutive_errors = 1
                            last_error = result_text
                        else:
                            consecutive_errors = 0
                            last_error = None
                        
                        # Truncate long results for display
                        display_result = result_text[:500] + "..." if len(result_text) > 500 else result_text
                        print(f"  ✅ Result: {display_result}")
                        
                        # Add tool result to messages
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": result_text
                        })
                        
                    except ValidationError as e:
                        error_msg = f"Validation error: {e}"
                        print(f"  ❌ {error_msg}")
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": error_msg
                        })
                        
                    except json.JSONDecodeError as e:
                        error_msg = f"JSON decode error: {e}"
                        print(f"  ❌ {error_msg}")
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": error_msg
                        })
                        
                    except Exception as e:
                        error_msg = f"Execution error: {str(e)}"
                        print(f"  ❌ {error_msg}")
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": error_msg
                        })
                else:
                    error_msg = f"Unknown tool: {function_name}"
                    print(f"  ❌ {error_msg}")
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": error_msg
                    })
        
        except Exception as e:
            print(f"\n❌ Error in iteration: {e}")
            import traceback
            traceback.print_exc()
            await asyncio.sleep(2)
            continue
    
    if iteration >= max_iterations:
        print("\n⚠️  Max iterations reached")
    
    return messages

async def main():
    """Main entry point"""
    print("="*60)
    print("🌐 Browser Automation Agent with Playwright MCP")
    print("="*60)
    print("\nAvailable commands:")
    print("  - Type your query to automate browser tasks")
    print("  - Type 'quit' or 'exit' to stop")
    print("="*60)
    
    try:
        while True:
            user_input = input("\n🤖 You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if not user_input:
                continue
            
            try:
                await run_agent_loop(user_input, max_iterations=20)
            except Exception as e:
                print(f"\n❌ Error: {e}")
                import traceback
                traceback.print_exc()
    finally:
        # Clean up browser session on exit
        await mcp_manager.close()

if __name__ == "__main__":
    asyncio.run(main())
