"""HTTP 工具：内置 ``http`` 工具 + OpenAPI 规范驱动的工具（本地起服务，全程离线）。

要点：
- ``HttpTool`` 是自由 GET/POST；``HttpOpenApiTool`` 把 OpenAPI operation 变成有 schema 的工具。
- ``load_spec`` / ``build_url_template`` / ``operation_to_json_schema`` 可离线把规范转成工具元数据。
- ``McpTool`` 走同一个 ``BaseTool`` 协议，只需把 endpoint 指向真实 MCP 服务端（本例不发起请求）。
运行：``python example/http_tools.py``
"""

import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from _shared import ScriptedLLM, finish, tool_call

from mainspring import AgentDefinition, AgentRuntime, RunContext, RuntimeDeps
from mainspring.plugins.registry import PluginRegistry
from mainspring.tools.builtin.http import HttpTool
from mainspring.tools.http_openapi import HttpOpenApiTool, build_url_template, operation_to_json_schema
from mainspring.tools.mcp import McpTool


class Handler(BaseHTTPRequestHandler):
    """最小 JSON 服务：GET /weather?city=... 与 POST /echo。"""

    def _send(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        url = urlparse(self.path)
        if url.path == "/weather":
            city = parse_qs(url.query).get("city", ["?"])[0]
            self._send({"city": city, "weather": "晴", "temp_c": 22})
            return
        self._send({"error": "not found"}, 404)

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b"{}"
        self._send({"echo": json.loads(raw or b"{}")})

    def log_message(self, *args) -> None:  # 静默访问日志
        pass


def openapi_spec(base: str) -> dict:
    return {
        "openapi": "3.0.0",
        "servers": [{"url": base}],
        "paths": {
            "/weather": {
                "get": {
                    "operationId": "getWeather",
                    "description": "查询城市天气",
                    "parameters": [
                        {
                            "name": "city",
                            "in": "query",
                            "required": True,
                            "description": "城市名",
                            "schema": {"type": "string"},
                        }
                    ],
                }
            }
        },
    }


async def main() -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        # 1. 内置 http 工具：自由 GET
        builtin = HttpTool()
        got = await builtin.run(url=f"{base}/weather?city=Beijing")
        print("== 内置 HttpTool ==")
        print("success:", got.success, "| output:", got.output)

        # 2. OpenAPI 规范 → 带 JSON Schema 的工具
        spec = openapi_spec(base)
        operation = spec["paths"]["/weather"]["get"]
        weather = HttpOpenApiTool(
            id_="tool_weather",
            name="weather",
            description=operation["description"],
            parameters=operation_to_json_schema(operation),
            permission="read",
            url_template=build_url_template(spec["servers"], "/weather"),
            method="GET",
        )
        print("\n== HttpOpenApiTool ==")
        print("自动生成的 parameters:", json.dumps(weather.parameters, ensure_ascii=False))
        print("调用结果:", (await weather.run(city="Shanghai")).output)

        # 3. POST + requestBody
        echo = HttpOpenApiTool(
            id_="tool_echo", name="echo", description="回显请求体",
            parameters={"type": "object", "properties": {"body": {"type": "object"}}, "required": []},
            permission="read", url_template=f"{base}/echo", method="POST",
        )
        print("POST 回显:", (await echo.run(body={"hello": "world"})).output)

        # 4. 交给 Agent 调用
        registry = PluginRegistry()
        registry.register_tool(weather)
        llm = ScriptedLLM([
            tool_call("weather", {"city": "Shanghai"}),
            finish("上海今天晴，22℃。"),
        ])
        definition = AgentDefinition(model="mock-model", system_prompt="你是助手", plugins=registry.all())
        result = await AgentRuntime().run(
            definition, "上海天气如何", RunContext(run_id="http-tools"), RuntimeDeps(llm=llm)
        )
        print("\n== Agent 调用 OpenAPI 工具 ==")
        print("result:", result.result)

        # 5. MCP：同一协议，换成真实 MCP 服务端即可
        mcp = McpTool(
            id_="tool_mcp_search", name="mcp_search", description="MCP 搜索工具",
            parameters={"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
            permission="read", endpoint="http://127.0.0.1:9000/mcp",
        )
        print("\nMcpTool 已按同一 BaseTool 协议装配:", mcp.name, "->", mcp.endpoint)
    finally:
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    asyncio.run(main())
