import math
import re
from collections import Counter

from app.models import KnowledgeDocument, KnowledgeHit


class InMemoryVectorStore:
    """Small deterministic vector-store substitute using character n-gram vectors."""

    def __init__(self, documents: list[KnowledgeDocument] | None = None) -> None:
        self._documents = documents or self._seed_documents()
        self._vectors = {
            document.id: self._embed(f"{document.title} {document.content}")
            for document in self._documents
        }

    async def similarity_search(self, query: str, *, limit: int = 3) -> list[KnowledgeHit]:
        query_vector = self._embed(query)
        hits = [
            KnowledgeHit(
                document=document,
                score=self._cosine(query_vector, self._vectors[document.id]),
            )
            for document in self._documents
        ]
        hits.sort(key=lambda hit: hit.score, reverse=True)
        return [hit for hit in hits[:limit] if hit.score > 0]

    @staticmethod
    def _embed(text: str) -> Counter[str]:
        normalized = re.sub(r"\s+", "", text.casefold())
        tokens = [normalized[index : index + 2] for index in range(max(0, len(normalized) - 1))]
        tokens.extend(re.findall(r"[a-z0-9]+", normalized))
        return Counter(tokens)

    @staticmethod
    def _cosine(left: Counter[str], right: Counter[str]) -> float:
        if not left or not right:
            return 0.0
        dot = sum(value * right.get(key, 0) for key, value in left.items())
        left_norm = math.sqrt(sum(value * value for value in left.values()))
        right_norm = math.sqrt(sum(value * value for value in right.values()))
        return dot / (left_norm * right_norm) if left_norm and right_norm else 0.0

    @staticmethod
    def _seed_documents() -> list[KnowledgeDocument]:
        return [
            KnowledgeDocument(
                id="kb-return-policy",
                title="七天无理由退货政策",
                content=(
                    "符合完好标准的商品，自签收之日起七天内可以申请无理由退货。定制商品、"
                    "已拆封的数字商品及影响二次销售的商品除外。"
                ),
                source="mock://knowledge/return-policy",
            ),
            KnowledgeDocument(
                id="kb-refund-time",
                title="退款到账时间",
                content="售后审核通过后，退款通常在一至三个工作日原路退回。",
                source="mock://knowledge/refund-time",
            ),
            KnowledgeDocument(
                id="kb-shipping",
                title="配送时效",
                content="普通订单付款后通常在四十八小时内发货，节假日和预售商品可能延迟。",
                source="mock://knowledge/shipping",
            ),
            KnowledgeDocument(
                id="kb-hours",
                title="人工客服时间",
                content="人工客服工作时间为每天九点至二十一点。",
                source="mock://knowledge/service-hours",
            ),
        ]
