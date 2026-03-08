#!/usr/bin/env python3
"""MCP HTTP Bridge — expose MCP tools as REST endpoints for a web UI.

Spawns MCP services as stdio subprocesses, connects via the MCP SDK,
and serves tool listing / chat / direct-tool-call over HTTP.

Usage:
    # Start the bridge (default port 8100):
    python mcp_http_bridge.py

    # Or with uvicorn:
    uvicorn mcp_http_bridge:app --host 0.0.0.0 --port 8100

Env vars:
    MCP_BRIDGE_PORT       — listen port (default 8100)
    MCP_CONFIG            — path to server_config.json (default config/server_config.json)
    LLM_API_KEY           — OpenAI-compatible API key
    LLM_API_URL           — base URL for LLM API
    LLM_MODEL_NAME        — model name
    LLM_MODEL_TYPE        — 'openai' | 'deepseek' | 'ollama'
"""

import asyncio
import json
import os
from contextlib import AsyncExitStack, asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from openai import AsyncOpenAI
from pydantic import BaseModel

_PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(_PROJECT_ROOT / ".env")

# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    query: str
    history: Optional[List[Dict[str, str]]] = None

class ChatResponse(BaseModel):
    answer: str
    tool_calls: List[Dict[str, Any]] = []

class ToolCallRequest(BaseModel):
    tool: str
    arguments: Dict[str, Any] = {}

class ToolCallResponse(BaseModel):
    result: str
    is_error: bool = False

class ToolInfo(BaseModel):
    name: str
    description: str
    parameters: Dict[str, Any]
    server: str

# ---------------------------------------------------------------------------
# MCP Connection Manager
# ---------------------------------------------------------------------------

class MCPBridge:
    """Manages stdio connections to MCP servers and provides tool access."""

    def __init__(self):
        self.exit_stack = AsyncExitStack()
        self.sessions: Dict[str, ClientSession] = {}
        self.tools: List[ToolInfo] = []
        self._tool_server: Dict[str, str] = {}  # tool_name -> server_name
        self._tool_session: Dict[str, ClientSession] = {}  # tool_name -> session
        self.llm_client: Optional[AsyncOpenAI] = None
        self.model_name: str = ""

    async def start(self, config_path: str):
        """Load config and connect to all MCP servers."""
        with open(config_path) as f:
            config = json.load(f)

        servers = config.get("mcpServers", {})

        for name, srv in servers.items():
            try:
                command = srv["command"]
                args = srv.get("args", [])

                params = StdioServerParameters(
                    command=command, args=args, env=None
                )
                transport = await self.exit_stack.enter_async_context(
                    stdio_client(params)
                )
                reader, writer = transport
                session = await self.exit_stack.enter_async_context(
                    ClientSession(reader, writer)
                )
                await session.initialize()
                self.sessions[name] = session

                # Discover tools
                tools_resp = await session.list_tools()
                for tool in tools_resp.tools:
                    info = ToolInfo(
                        name=tool.name,
                        description=tool.description or "",
                        parameters=tool.inputSchema,
                        server=name,
                    )
                    self.tools.append(info)
                    self._tool_server[tool.name] = name
                    self._tool_session[tool.name] = session

                print(f"[MCP] Connected: {name} ({len(tools_resp.tools)} tools)")
            except Exception as e:
                print(f"[MCP] Failed to connect {name}: {e}")

        # Init LLM client
        api_key = os.getenv("LLM_API_KEY", "")
        base_url = os.getenv("LLM_API_URL", "")
        model_type = os.getenv("LLM_MODEL_TYPE", "deepseek")
        self.model_name = os.getenv("LLM_MODEL_NAME", "deepseek-chat")

        if api_key:
            self.llm_client = AsyncOpenAI(
                api_key=api_key,
                base_url=None if model_type == "openai" else base_url,
            )

        print(f"[MCP] Bridge ready — {len(self.tools)} tools from {len(self.sessions)} servers")

    async def stop(self):
        await self.exit_stack.aclose()

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        session = self._tool_session.get(tool_name)
        if not session:
            raise ValueError(f"Unknown tool: {tool_name}")
        result = await session.call_tool(tool_name, arguments)
        if result.content:
            return result.content[0].text
        return "(no result)"

    async def chat(self, query: str, history: Optional[List[Dict[str, str]]] = None) -> ChatResponse:
        """Send query through LLM with MCP tools available."""
        if not self.llm_client:
            raise ValueError("LLM not configured — set LLM_API_KEY in .env")

        # Build OpenAI-format tool definitions
        oai_tools = [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                },
            }
            for t in self.tools
        ]

        messages: List[Dict[str, Any]] = []
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": query})

        tool_calls_log: List[Dict[str, Any]] = []

        # LLM tool-call loop (max 5 rounds)
        for _ in range(5):
            resp = await self.llm_client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                tools=oai_tools if oai_tools else None,
                tool_choice="auto" if oai_tools else None,
            )
            msg = resp.choices[0].message
            messages.append(msg)

            if not msg.tool_calls:
                return ChatResponse(answer=msg.content or "", tool_calls=tool_calls_log)

            for tc in msg.tool_calls:
                fn_name = tc.function.name
                try:
                    fn_args = json.loads(tc.function.arguments) if isinstance(tc.function.arguments, str) else tc.function.arguments
                except json.JSONDecodeError:
                    fn_args = {"input": tc.function.arguments}

                try:
                    result_text = await self.call_tool(fn_name, fn_args)
                except Exception as e:
                    result_text = f"Error: {e}"

                tool_calls_log.append({
                    "tool": fn_name,
                    "arguments": fn_args,
                    "result": result_text[:500],
                })

                messages.append({
                    "role": "tool",
                    "content": result_text,
                    "tool_call_id": tc.id,
                    "name": fn_name,
                })

        # Fallback if loop exhausted
        return ChatResponse(answer="(max tool iterations reached)", tool_calls=tool_calls_log)


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

bridge = MCPBridge()


@asynccontextmanager
async def lifespan(app: FastAPI):
    config_path = os.getenv("MCP_CONFIG", str(_PROJECT_ROOT / "config" / "server_config.json"))
    await bridge.start(config_path)
    yield
    await bridge.stop()


app = FastAPI(title="MCP HTTP Bridge", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/tools", response_model=List[ToolInfo])
async def list_tools():
    """List all available MCP tools."""
    return bridge.tools


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """Chat with LLM + MCP tools."""
    try:
        return await bridge.chat(req.query, req.history)
    except ValueError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/tool", response_model=ToolCallResponse)
async def call_tool(req: ToolCallRequest):
    """Directly call an MCP tool."""
    try:
        result = await bridge.call_tool(req.tool, req.arguments)
        return ToolCallResponse(result=result)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        return ToolCallResponse(result=str(e), is_error=True)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "servers": len(bridge.sessions),
        "tools": len(bridge.tools),
    }


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("MCP_BRIDGE_PORT", "8100"))
    uvicorn.run(app, host="0.0.0.0", port=port)
