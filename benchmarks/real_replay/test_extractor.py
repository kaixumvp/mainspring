"""规则提取器验证：跑考勤匿名语料，看能抽出多少结构化事实。

运行：.venv/Scripts/python benchmarks/real_replay/test_extractor.py
"""
from __future__ import annotations

import asyncio
import re
from pathlib import Path

from mainspring.memory.extractor import RuleExtractor

CORPUS = Path(__file__).with_name("chat_log_anonymized.txt")


def parse_msgs() -> list[str]:
    out: list[str] = []
    for line in CORPUS.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^[^:：]+[:：]\s?(.*)$", line)
        if m:
            out.append(m.group(1).strip())
    return out


async def main() -> None:
    extractor = RuleExtractor()
    msgs = parse_msgs()
    total, extracted = 0, 0
    print(f"语料 {len(msgs)} 条消息，规则提取器扫描中…\n")
    for text in msgs:
        facts = await extractor.extract(text)
        total += 1
        if facts:
            extracted += 1
            print(f" [{facts[0]['type']}] {text[:60]}")
            for f in facts:
                print(f"       subject={f['subject'][:24]:<24} "
                      f"pred={f['predicate'][:20]:<20} "
                      f"obj={f['object'][:30]:<30} conf={f['confidence']:.2f}")
    print(f"\n抽到 {extracted}/{total} 条含结构化信息")


if __name__ == "__main__":
    asyncio.run(main())