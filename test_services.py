#!/usr/bin/env python3
"""Test all MCP services end-to-end without needing an LLM API key.

Usage:
    source .venv/bin/activate
    python test_services.py
"""
import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from contextlib import AsyncExitStack


async def test_service(name: str, script: str, tool_name: str, tool_args: dict):
    """Connect to an MCP service, list tools, and call one tool."""
    print(f"\n{'='*60}")
    print(f"Testing: {name} ({script})")
    print(f"{'='*60}")

    import os
    base_dir = os.path.dirname(os.path.abspath(__file__))
    venv_python = os.path.join(base_dir, ".venv", "bin", "python")
    if not os.path.exists(venv_python):
        venv_python = "python"  # fallback

    exit_stack = AsyncExitStack()
    try:
        server_params = StdioServerParameters(
            command=venv_python,
            args=[os.path.join(base_dir, "services", script)],
            env=None
        )

        stdio_transport = await exit_stack.enter_async_context(
            stdio_client(server_params)
        )
        reader, writer = stdio_transport

        session = await exit_stack.enter_async_context(
            ClientSession(reader, writer)
        )
        await session.initialize()

        # List tools
        tools_response = await session.list_tools()
        tool_names = [t.name for t in tools_response.tools]
        print(f"  Connected! Available tools: {tool_names}")

        for tool in tools_response.tools:
            print(f"    - {tool.name}: {tool.description}")

        # Call tool
        print(f"\n  Calling: {tool_name}({tool_args})")
        result = await session.call_tool(tool_name, tool_args)
        print(f"  Result: {result.content[0].text}")
        print(f"\n  ✓ PASS")

    except Exception as e:
        print(f"\n  ✗ FAIL: {e}")
    finally:
        await exit_stack.aclose()


async def main():
    print("tiny-mcp Service Test Suite")
    print("=" * 60)

    await test_service(
        name="Time Service",
        script="time_service.py",
        tool_name="get_current_time",
        tool_args={"timezone": "America/New_York"},
    )

    await test_service(
        name="Calculator Service",
        script="calculator_service.py",
        tool_name="calculate",
        tool_args={"expression": "sqrt(144) + 3 ** 2"},
    )

    await test_service(
        name="Calculator - Unit Conversion",
        script="calculator_service.py",
        tool_name="unit_convert",
        tool_args={"value": 72.0, "from_unit": "fahrenheit", "to_unit": "celsius"},
    )

    await test_service(
        name="US Weather Service",
        script="weather_service_us.py",
        tool_name="get_alerts",
        tool_args={"state": "CA"},
    )

    print(f"\n{'='*60}")
    print("All tests completed!")
    print(f"{'='*60}")


if __name__ == "__main__":
    asyncio.run(main())
