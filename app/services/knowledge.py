from app.models import KnowledgeHit
from app.repositories import InMemoryVectorStore


class KnowledgeService:
    def __init__(self, vector_store: InMemoryVectorStore) -> None:
        self._vector_store = vector_store

    async def search(self, query: str, *, limit: int = 3) -> list[KnowledgeHit]:
        return await self._vector_store.similarity_search(query, limit=limit)
