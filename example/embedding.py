"""向量化：``HashEmbedder`` 离线实现 + 相似度去重（生产换 ``LiteLLMEmbedder``）。

要点：
- ``Embedder`` 是端口；``HashEmbedder`` 用确定性特征哈希做离线回退，适合本地/测试。
- 相同文本必得相同向量；近重复文本余弦相似度高，可用于长期记忆的去重阈值。
- 生产用 ``LiteLLMEmbedder(model, api_key)``（OpenAI 兼容 embedding 模型），协议不变。
运行：``python example/embedding.py``
"""

import asyncio
import math

from mainspring.llm.embedder import HashEmbedder


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb or 1.0)


async def main() -> None:
    embedder = HashEmbedder(dimension=128)

    texts = [
        "user prefers dark theme",
        "user prefers dark theme now",    # 近重复
        "user prefers dark theme a lot",  # 近重复
        "weather in Beijing",             # 无关
    ]
    vectors = await embedder.embed(texts)

    print("== HashEmbedder ==")
    print("维度:", len(vectors[0]))
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            print(f"  {texts[i]!r} vs {texts[j]!r}: {cosine(vectors[i], vectors[j]):.3f}")

    # 确定性：同一文本再次 embed 得到完全相同的向量
    again = await embedder.embed([texts[0]])
    print("确定性（同文本同向量）:", again[0] == vectors[0])

    # 记忆去重：相似度达到阈值即视为重复条目
    threshold = 0.75
    pairs = [(i, j) for i in range(len(texts)) for j in range(i + 1, len(texts))]
    duplicates = [(texts[i], texts[j]) for i, j in pairs if cosine(vectors[i], vectors[j]) >= threshold]
    print(f"阈值 {threshold} 下视为重复:", duplicates)


if __name__ == "__main__":
    asyncio.run(main())
