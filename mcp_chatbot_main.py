import asyncio
import json
from pathlib import Path

from mcp_chatbot import Configuration, ChatSession, LLMService, MCPClient

async def main() -> None:
    """主入口函数
    """
    cfg = Configuration()
    cfg.print_config()
    config_path = Path(__file__).resolve().parent / "config" / "server_config.json"
    server_config = cfg.load_config(str(config_path))  # 加载服务器配置

    servers = [
        MCPClient(name, svc_cfg)
        for name, svc_cfg in server_config["mcpServers"].items()  # 创建服务器实例
    ]

    llm_service = LLMService(
        api_key=cfg.llm_api_key,
        model_name=cfg.model_name,
        base_url=cfg.base_url,
        model_type=cfg.model_type,
    )

    chat_session = ChatSession(servers, llm_service)
    await chat_session.start()


if __name__ == "__main__":
    import platform
    if platform.system().lower() == 'windows':
        loop = asyncio.get_event_loop()
        loop.run_until_complete(main())
    else:
        asyncio.run(main())
