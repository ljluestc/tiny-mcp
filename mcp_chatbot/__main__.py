"""Allow running as: python -m mcp_chatbot"""
import asyncio

from mcp_chatbot_main import main

if __name__ == "__main__":
    import platform
    if platform.system().lower() == "windows":
        loop = asyncio.get_event_loop()
        loop.run_until_complete(main())
    else:
        asyncio.run(main())
