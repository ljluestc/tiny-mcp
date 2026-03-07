#!/usr/bin/env python3
"""MCP Client using Anthropic Claude for tool calling.

Usage:
    # Direct server script:
    python anthropic_mcp_client.py services/time_service.py

    # Via config file:
    python anthropic_mcp_client.py get_current_time config/mcp_config.json

Requires ANTHROPIC_API_KEY in .env or environment.
"""

import asyncio
import json
import os
import sys
from contextlib import AsyncExitStack
from typing import Any, Dict, List, Optional

import anthropic
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from pathlib import Path

# Load .env from script dir (project root) — dotenv loads relative to CWD by default
_PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(_PROJECT_ROOT / ".env")


class AnthropicMCPClient:
    def __init__(
        self,
        api_key: str,
        model_name: str = "claude-sonnet-4-20250514",
        max_tokens: int = 4096,
    ):
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self._cleanup_lock = asyncio.Lock()

        self.available_tools: List[Dict[str, Any]] = []
        self.anthropic_tools: List[Dict[str, Any]] = []

        self.client = anthropic.AsyncAnthropic(api_key=api_key)

    @staticmethod
    def parse_arguments(args: List[str]) -> StdioServerParameters:
        """Parse CLI args into StdioServerParameters."""
        if len(args) == 1:
            server_script = args[0]
            if not server_script.endswith((".py", ".js")):
                raise ValueError("Server script must be .py or .js")
            command = "python" if server_script.endswith(".py") else "node"
            return StdioServerParameters(command=command, args=[server_script], env=None)
        elif len(args) == 2:
            server_id, config_path = args
            with open(config_path) as f:
                config = json.load(f)
            srv = config.get("mcpServers", {}).get(server_id)
            if not srv:
                raise ValueError(f"Server '{server_id}' not found in config")
            return StdioServerParameters(
                command=srv["command"], args=srv["args"], env=None
            )
        else:
            raise ValueError(
                "Usage: python anthropic_mcp_client.py <script.py>\n"
                "   or: python anthropic_mcp_client.py <server_id> <config.json>"
            )

    async def connect_to_server(self, server_params: StdioServerParameters):
        """Connect to an MCP server via stdio."""
        transport = await self.exit_stack.enter_async_context(
            stdio_client(server_params)
        )
        reader, writer = transport
        self.session = await self.exit_stack.enter_async_context(
            ClientSession(reader, writer)
        )
        await self.session.initialize()

        # Discover tools
        tools_resp = await self.session.list_tools()
        for tool in tools_resp.tools:
            # Store in Anthropic tool format
            self.anthropic_tools.append(
                {
                    "name": tool.name,
                    "description": tool.description or "",
                    "input_schema": tool.inputSchema,
                }
            )

        tool_names = [t["name"] for t in self.anthropic_tools]
        print(f"[SYS] Connected. Available tools: {tool_names}")

    async def process_query(self, query: str) -> str:
        """Send query to Claude, handle tool calls, return final response."""
        messages = [{"role": "user", "content": query}]

        while True:
            resp = await self.client.messages.create(
                model=self.model_name,
                max_tokens=self.max_tokens,
                tools=self.anthropic_tools,
                messages=messages,
            )

            # Check if Claude wants to use tools
            if resp.stop_reason == "tool_use":
                # Collect all content blocks (text + tool_use)
                assistant_content = []
                for block in resp.content:
                    if block.type == "text":
                        assistant_content.append(
                            {"type": "text", "text": block.text}
                        )
                    elif block.type == "tool_use":
                        assistant_content.append(
                            {
                                "type": "tool_use",
                                "id": block.id,
                                "name": block.name,
                                "input": block.input,
                            }
                        )

                messages.append({"role": "assistant", "content": assistant_content})

                # Execute each tool call
                tool_results = []
                for block in resp.content:
                    if block.type == "tool_use":
                        print(f"[TOOL] Calling {block.name}({block.input})")
                        result = await self.session.call_tool(block.name, block.input)
                        result_text = (
                            result.content[0].text
                            if result.content
                            else "No result"
                        )
                        print(f"[TOOL] {block.name} -> {result_text[:200]}")
                        tool_results.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": block.id,
                                "content": result_text,
                            }
                        )

                messages.append({"role": "user", "content": tool_results})
                continue  # Let Claude process the tool results

            # No more tool calls — extract final text
            text_parts = [b.text for b in resp.content if b.type == "text"]
            return "\n".join(text_parts) if text_parts else "(no response)"

    async def chat_loop(self):
        """Interactive REPL."""
        print("[SYS] MCP Client ready. Type your query (or 'quit' to exit).\n")
        loop = asyncio.get_event_loop()

        while True:
            try:
                query = await loop.run_in_executor(
                    None, lambda: input("[YOU]: ").strip()
                )
                if not query:
                    continue
                if query.lower() in ("quit", "exit", "q"):
                    break

                response = await self.process_query(query)
                print(f"\n[Claude]: {response}\n")

            except (KeyboardInterrupt, EOFError):
                print("\n[SYS] Shutting down...")
                break
            except Exception as e:
                print(f"\n[ERR] {e}\n")

    async def cleanup(self):
        async with self._cleanup_lock:
            try:
                await self.exit_stack.aclose()
                self.session = None
            except Exception as e:
                print(f"[ERR] Cleanup error: {e}")


async def main():
    try:
        server_params = AnthropicMCPClient.parse_arguments(sys.argv[1:])
    except ValueError as e:
        print(f"[ERR] {e}")
        sys.exit(1)

    api_key = (
        os.getenv("ANTHROPIC_API_KEY")
        or os.getenv("ANTHROPIC_KEY")
        or ""
    )
    model = os.getenv("LLM_MODEL_NAME", "claude-sonnet-4-20250514")

    if not api_key:
        print("[ERR] ANTHROPIC_API_KEY or ANTHROPIC_KEY not set. Add to .env or environment.")
        sys.exit(1)

    print(f"[SYS] Model: {model}")

    client = AnthropicMCPClient(api_key=api_key, model_name=model)

    try:
        await client.connect_to_server(server_params)
        await client.chat_loop()
    except Exception as e:
        print(f"\n[ERR] {e}")
    finally:
        await client.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
