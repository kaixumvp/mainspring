"""记忆提取器：文本 → 结构化事实/偏好/结论（可切换模式 + 可配置规则）。

模式（环境变量 EXTRACT_MODE）：
  text   仅存原文（默认，向后兼容）
  fact   仅存结构化节点（不存原文）
  both   原文 + 结构化节点并行

提取器类型（环境变量 DASHSCOPE_API_KEY / EXTRACTOR_MODE）：
  rule   零成本、规则驱动——默认中文模板，支持换规则本（见 RuleExtractor）
  LLM    任意语言，调用百炼 qwen 等模型 —— 生产路径

RuleExtractor 是可配置的——构造时传入自定义规则本替换或扩展默认值：
  # 换全套：
  extractor = RuleExtractor(fact=pats, pref=pats2, conc=pats3)
  # 追加：
  extractor = RuleExtractor(fact_extra=pats)
  # 全默认（中文技术场景开箱即用）：
  extractor = RuleExtractor()

每条规则 = (关键词数组, 匹配该关键词的正则模板)。fact 的模板固定是"主语 方向关键词 数字 单位"；
pref 是"偏好关键词 主体内容"；conc 是"主语 连接词 结论内容"。
"""
from __future__ import annotations

import asyncio
import logging
import re

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------- 默认中文规则本
_FACT_DIR = [
    "下降", "上涨", "涨了", "跌了", "增加", "减少", "降低", "升高", "提升", "提高",
    "激活", "命中", "关闭", "处理", "闭环", "解决", "搞定",
]
_PREF_KW = ["偏好", "喜欢", "想要", "希望", "必须", "不要", "别", "总是", "从不", "一直"]
_CONC_KW = [
    "的逻辑", "的结论", "的说法", "的意思是", "就是", "应该是", "还是该", "还是按", "按照",
]
_SUBJECT_SPLIT = re.compile(r"[后前中时日的 的，。、！；：?？\n\u3000]")


def _subject_window(text: str, tail_word: str) -> str:
    idx = text.find(tail_word)
    if idx < 0:
        return tail_word
    pre = text[:idx]
    tokens = _SUBJECT_SPLIT.split(pre)
    return tokens[-1].strip()[-24:] if tokens else tail_word


def _build_fact(direction_kw: list[str]) -> re.Pattern:
    alts = "|".join(direction_kw)
    return re.compile(
        rf"([一-龥\w·/−]{{1,40}}?)({alts})\s*(\d+\.?\d*)\s*(个|个百分点|倍|%|％|条|次)?")


def _build_pref(pref_kw: list[str]) -> re.Pattern:
    alts = "|".join(pref_kw)
    return re.compile(rf"({alts})(.{{2,40}}?)(?:，|。|；|哈|啊|吧|吗|嘛|$)")


def _build_conc(conc_kw: list[str]) -> re.Pattern:
    alts = "|".join(conc_kw)
    return re.compile(rf"([一-龥]{{2,24}})({alts})(.{{3,60}}?)(?:哈|啊|吧|吗|嘛|。|，|$)")


# ---------------------------------------------------------------- 公共接口
class Extractor:
    """提取器抽象——所有实现共用一个 extract() 签名。"""

    async def extract(self, text: str) -> list[dict]:
        raise NotImplementedError


# ---------------------------------------------------------------- 规则版（可配置）
class RuleExtractor(Extractor):
    """规则模板提取——传递自有的关键词数组，零成本，仅中文，但适用跨业务线。

    Defaults 提供面向中文技术群聊/业务流程的通用开箱设置。
    """

    def __init__(self, fact: list[str] | None = None, pref: list[str] | None = None,
                 conc: list[str] | None = None,
                 fact_extra: list[str] | None = None, pref_extra: list[str] | None = None,
                 conc_extra: list[str] | None = None) -> None:
        f_kw = (fact or _FACT_DIR) + (fact_extra or [])
        p_kw = (pref or _PREF_KW) + (pref_extra or [])
        c_kw = (conc or _CONC_KW) + (conc_extra or [])
        self._fact_re = _build_fact(f_kw) if fact is not None or fact_extra else _build_fact(_FACT_DIR)
        self._pref_re = _build_pref(p_kw) if pref is not None or pref_extra else _build_pref(_PREF_KW)
        self._conc_re = _build_conc(c_kw) if conc is not None or conc_extra else _build_conc(_CONC_KW)
        self._pref_kw = p_kw

    async def extract(self, text: str) -> list[dict]:
        result: list[dict] = []
        text = (text or "").strip()
        if not text:
            return result
        # -- 数字事实 --
        for m in self._fact_re.finditer(text):
            subject = _subject_window(text, m.group(1) + m.group(2))
            direction = m.group(2)
            num = m.group(3)
            unit = m.group(4) or ""
            result.append({
                "type": "fact", "subject": subject, "predicate": direction,
                "object": f"{direction} {num}{unit}", "confidence": 0.75,
            })
        # -- 偏好 --
        for m in self._pref_re.finditer(text):
            kw = m.group(1)
            body = m.group(2).strip()
            if len(body) < 2:
                continue
            subject = _subject_window(text, kw)
            predicate = body.split("用")[0] if "用" in body else "偏好"
            obj = body.split("用")[1].strip() if "用" in body else body
            result.append({
                "type": "preference", "subject": subject, "predicate": predicate,
                "object": obj, "confidence": 0.70,
            })
        # -- 结论 --
        for m in self._conc_re.finditer(text):
            subj = m.group(1).strip()
            conn = m.group(2)
            rest = m.group(3).strip()
            result.append({
                "type": "conclusion", "subject": subj,
                "predicate": f"{subj}{conn}",
                "object": rest[:60], "confidence": 0.60,
            })
        return result


# ---------------------------------------------------------------- LLM 版（任意语言）
class LLMExtractor(Extractor):
    """百炼 LLM 提取——任意语言，按调用计费。"""

    def __init__(self, api_key: str, model: str = "qwen-turbo-latest") -> None:
        self._key = api_key
        self._model = model

    async def extract(self, text: str) -> list[dict]:
        """发送给 LLM 以 JSON 格式抽取事实/偏好/结论。失败退空列表（不阻塞管线）。"""
        import dashscope
        from http import HTTPStatus as HttpOK
        dashscope.base_http_api_url = "https://cn-beijing.maas.qianwenaiapi.com/api/v1"

        prompt = (
            "Extract facts, preferences, and conclusions from this message.\n"
            "Return ONLY a JSON array: [{\"type\":\"fact|preference|conclusion\","
            "\"subject\":\"...\",\"predicate\":\"...\",\"object\":\"...\","
            "\"confidence\":0.0-1.0}].\n\nMessage: " + text
        )
        try:
            resp = await asyncio.to_thread(
                dashscope.Generation.call,
                model=self._model,
                messages=[{"role": "system", "content": "Extract facts. Reply JSON only."},
                          {"role": "user", "content": prompt}],
                api_key=self._key,
            )
            if resp.status_code != HttpOK.OK:
                logger.warning("LLM extraction failed: %s %s", resp.status_code, resp.message)
                return []
            import json as _json
            raw = resp.output.text
            start = raw.find("[")
            end = raw.rfind("]")
            if start >= 0 and end > start:
                return _json.loads(raw[start:end + 1])
            return _json.loads(raw)
        except Exception:
            logger.warning("LLM extraction exception", exc_info=True)
            return []