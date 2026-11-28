"""记忆模块的配置工厂——改环境变量，不改代码。

使用方式（宿主装配时）：
    from mainspring.memory.config import make_reflector, make_brain_config
    refl = make_reflector()  # 自动选 规则 / LLM
    cfg  = make_brain_config()

环境变量：
    DASHSCOPE_API_KEY      为空 = 规则模式；非空 = LLM 模式（自动启用百炼网关）
    REFLECTOR_MODE          "rule" 强制规则  "llm" 强制 LLM  "auto" 按 key 有无（默认）
    DASHSCOPE_LLM_MODEL     LLM 模型名（默认 qwen-turbo-latest）
    EXTRACT_MODE            "text" 仅存原文（默认） "fact" 仅存结构化 "both" 并存
    EXTRACTOR_MODE          同 REFLECTOR_MODE，控制提取器用规则还是 LLM

没网关时调用 make_reflector() 就是纯规则打分，跟改之前行为完全一致。
make_extractor() 同逻辑——无 key = 规则提取器（仅中文），有 key = LLM 提取器（全语言）。
"""
from __future__ import annotations

import os

from mainspring.memory.brain import BrainConfig
from mainspring.memory.extractor import Extractor, RuleExtractor, LLMExtractor
from mainspring.memory.reflect import Reflector


def make_reflector() -> Reflector:
    mode = os.environ.get("REFLECTOR_MODE", "auto")
    key = os.environ.get("DASHSCOPE_API_KEY")
    if mode == "rule" or (mode == "auto" and not key):
        return Reflector()
    from mainspring.llm.dashscope_gateway import DashScopeGateway
    model = os.environ.get("DASHSCOPE_LLM_MODEL", "qwen-turbo-latest")
    return Reflector(llm_gateway=DashScopeGateway(key), llm_model=model)


def make_brain_config(hot_budget: int = 150, **overrides) -> BrainConfig:
    return BrainConfig(hot_budget=hot_budget, **overrides)


def read_extract_mode() -> str:
    return os.environ.get("EXTRACT_MODE", "text")  # text | fact | both


def make_extractor() -> Extractor:
    mode = os.environ.get("EXTRACTOR_MODE", "auto")
    key = os.environ.get("DASHSCOPE_API_KEY")
    if mode == "rule" or (mode == "auto" and not key):
        return RuleExtractor()
    model = os.environ.get("DASHSCOPE_LLM_MODEL", "qwen-turbo-latest")
    return LLMExtractor(key, model)