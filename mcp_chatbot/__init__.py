from .config.configuration import Configuration
from .chat.chat_session import ChatSession
from .llm.llm_service import LLMService
from .mcp.mcp_client import MCPClient
from .mcp.mcp_tool import MCPTool

# Lazy import to avoid circular dependency; main lives in mcp_chatbot_main.py
def __getattr__(name: str):
    if name == "main":
        from mcp_chatbot_main import main
        return main
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")