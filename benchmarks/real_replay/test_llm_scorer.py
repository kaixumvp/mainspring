"""端到端验证：百炼 LLM 接入 Reflector 的写入选材打分。

运行：DASHSCOPE_API_KEY=sk-xxx .venv/Scripts/python benchmarks/real_replay/test_llm_scorer.py
"""
from __future__ import annotations

import asyncio
import os
from pathlib import Path

import dashscope
from http import HTTPStatus as HttpOK

from mainspring.memory.reflect import Reflector
from mainspring.ports import LLMResponse

_LLM_MODEL = "qwen-turbo-latest"


class DashScopeGateway:
    """百炼 chat 适配器（实现 mainspring LLMGateway 协议）——也叫聊天模型。"""

    def __init__(self, api_key: str) -> None:
        self._key = api_key
        dashscope.base_http_api_url = "https://cn-beijing.maas.qianwenaiapi.com/api/v1"

    async def chat(self, messages: list[dict], model: str | None = None,
                   tools: list[dict] | None = None, schema=None) -> LLMResponse:
        resp = await asyncio.to_thread(
            dashscope.Generation.call,
            model=model or _LLM_MODEL,
            messages=messages,
            api_key=self._key,
        )
        if resp.status_code != HttpOK.OK:
            return LLMResponse(content=f"ERROR: {resp.status_code} {resp.message}")
        return LLMResponse(content=resp.output.text)


async def main() -> None:
    key = os.environ.get("DASHSCOPE_API_KEY")
    if not key:
        print("请设置 DASHSCOPE_API_KEY 环境变量")
        return

    refl_rule = Reflector()
    refl_llm = Reflector(llm_gateway=DashScopeGateway(key), llm_model=_LLM_MODEL)

    corpus = Path(__file__).with_name("chat_log_anonymized.txt")
    msgs: list[str] = []
    for line in corpus.read_text(encoding="utf-8").splitlines():
        if ":" in line or "：" in line:
            text = line.split(":", 1)[-1].split("：", 1)[-1].strip()
            msgs.append(text)

    print(f"{'规则分':>6}  {'LLM分':>6}  消息预览")
    print("-" * 58)

    for text in msgs[:10]:
        rule = await refl_rule.score(text)
        llm = await refl_llm.score(text)
        mark = " ← LLM" if abs(llm - rule) > 0.2 else ""
        print(f"{rule:>6.3f}  {llm:>6.3f}   {text[:40]}{mark}")

    # 重点：那几条被旧规则误杀的关键结论
    key_texts = [
        "没有用到考勤组，登陆人——》考勤档案——》统计方案——》考勤周期",
        "薪福通逻辑是：修改当前加班规则，选择统计周期为当前周期的，则是当月周期",
        "修改规则后，再去进行手工核算日考勤，感觉会读到最新的打卡规则或加班规则吧？",
        "不是当月，是统计方案的周期，你改统计方案的周期看过吗",
    ]
    print("\n—— 关键结论（旧规则 0.3 被滤，修后 0.4，LLM 应给 >0.5）：——")
    for text in key_texts:
        rule = await refl_rule.score(text)
        llm = await refl_llm.score(text)
        print(f"  规则 {rule:.3f}  LLM {llm:.3f}   {text[:48]}")


if __name__ == "__main__":
    asyncio.run(main())