"""FastAPI dependency factories for shared singletons."""

from __future__ import annotations

from functools import lru_cache

import anthropic
import redis
import structlog
from openai import OpenAI

from app.generation.cag.semantic import EstimationSemanticCache
from app.config import get_settings
from app.generation.rag.chunking.base import Chunker
from app.generation.rag.chunking.structural import JSONStructuralChunker
from app.generation.rag.embedding.embedder import OpenAIEmbedder
from app.generation.rag.chunking.strategies import (
    ContextualRetrievalChunker,
    FixedSizeChunker,
    HierarchicalChunker,
    PropositionalChunker,
    RecursiveChunker,
    SemanticChunker,
    SentenceWindowChunker,
)
from app.ingestion.catalog import DataCatalog, load_catalog
from app.ingestion.loaders.filesystem import FileSystemLoader
from app.ingestion.parsers.registry import ParserRegistry, default_registry
from app.generation.cag.exact import EstimationCache
from app.domain.estimation_service import EstimationService
from app.foundation.llm.runtime_config import RuntimeModelConfig, RuntimeRetrievalConfig
from app.foundation.llm.wrapper import LLMWrapper
from app.foundation.persistence.database import get_async_session_factory
from app.generation.rag.index_service import CorpusIndexService
from app.generation.rag.ingest_service import RagIngestService
from app.generation.rag.retriever import SemanticRetriever
from app.generation.rag.store.repository import ChunkStore
from app.generation.conversation.store import SessionStore

log = structlog.get_logger()


@lru_cache
def get_cache() -> EstimationCache:
    settings = get_settings()
    return EstimationCache.from_url(settings.REDIS_URL, ttl=settings.CACHE_TTL)


@lru_cache
def get_runtime_config() -> RuntimeModelConfig:
    """Redis-backed override store for the LLM model knobs (Settings UI).

    The singleton is just the Redis handle — freshness comes from reading
    Redis inside on every call, not from rebuilding this object.
    """
    settings = get_settings()
    return RuntimeModelConfig.from_url(settings.REDIS_URL, settings)


@lru_cache
def get_runtime_retrieval_config() -> RuntimeRetrievalConfig:
    """Redis-backed override store for the Session 10 retrieval toggles
    (search mode + reranking), read per call so a flip in the Ajustes UI takes
    effect on the next retrieval without a restart."""
    settings = get_settings()
    return RuntimeRetrievalConfig.from_url(settings.REDIS_URL, settings)


@lru_cache
def get_reranker():
    """Cross-encoder reranker singleton (Session 10). The model loads lazily on
    the first rerank, so building this is cheap and import-time has no torch cost."""
    from app.generation.rag.retrieval.reranker import CrossEncoderReranker

    return CrossEncoderReranker.from_settings()


@lru_cache
def get_llm_wrapper() -> LLMWrapper:
    settings = get_settings()
    return LLMWrapper(
        openai_api_key=settings.OPENAI_API_KEY,
        anthropic_api_key=settings.ANTHROPIC_API_KEY,
        primary_model=settings.PRIMARY_MODEL,
        fallback_model=settings.FALLBACK_MODEL,
        timeout=settings.LLM_TIMEOUT,
        num_retries=settings.LLM_RETRIES,
        cache=get_cache(),
        runtime_config=get_runtime_config(),
    )


@lru_cache
def get_openai_client() -> OpenAI | None:
    """Lazy OpenAI client used by ``check_input`` (Moderation API) and the
    semantic cache (Embeddings API)."""
    settings = get_settings()
    if not settings.OPENAI_API_KEY:
        return None
    return OpenAI(api_key=settings.OPENAI_API_KEY)


@lru_cache
def get_chunker() -> JSONStructuralChunker:
    """Stateless structural chunker for the embedding pipeline (Session 7)."""
    return JSONStructuralChunker()


@lru_cache
def get_embedder() -> OpenAIEmbedder | None:
    """OpenAI embedder for the embedding pipeline. ``None`` when no API key is
    configured (mirrors ``get_semantic_cache``); the router maps that to a 500."""
    settings = get_settings()
    client = get_openai_client()
    if client is None:
        log.warning("embedder_disabled", reason="no_openai_key")
        return None
    return OpenAIEmbedder(client=client, model=settings.EMBEDDING_MODEL)


# --- Session 8: pgvector persistence + semantic search ---------------------


@lru_cache
def get_chunk_store() -> ChunkStore:
    """Stateless async data-access layer over documents/chunks."""
    return ChunkStore()


@lru_cache
def get_rag_ingest_service() -> RagIngestService | None:
    """Chunk → embed → persist orchestration. ``None`` without an OpenAI key
    (mirrors ``get_embedder``); the router maps that to a 500."""
    embedder = get_embedder()
    if embedder is None:
        return None
    return RagIngestService(
        chunker=get_chunker(),
        embedder=embedder,
        session_factory=get_async_session_factory(),
        store=get_chunk_store(),
    )


@lru_cache
def get_corpus_index_service() -> CorpusIndexService | None:
    """Session 11 batch corpus expansion. ``None`` without an OpenAI key
    (mirrors ``get_rag_ingest_service``); the router maps that to a 500."""
    ingest = get_rag_ingest_service()
    if ingest is None:
        return None
    return CorpusIndexService(ingest=ingest)


@lru_cache
def get_semantic_retriever() -> SemanticRetriever | None:
    """Query-side counterpart of the ingest service. Same ``None`` contract."""
    embedder = get_embedder()
    if embedder is None:
        return None
    return SemanticRetriever(
        embedder=embedder,
        session_factory=get_async_session_factory(),
        store=get_chunk_store(),
    )


# --- Session 9: RAG estimation pipeline (transcript → grounded estimate) ----


@lru_cache
def get_idempotency_store():
    """Idempotency cache for ``POST /v1/estimate/from-transcript`` (singleton).

    Redis-backed when ``REDIS_URL`` is reachable, in-process dict otherwise."""
    from app.generation.rag.idempotency import IdempotencyStore

    return IdempotencyStore.from_settings(get_settings())


@lru_cache
def get_token_encoder():
    """tiktoken ``cl100k_base`` encoder used for the context token budget."""
    import tiktoken

    return tiktoken.get_encoding("cl100k_base")


@lru_cache
def get_anthropic_client() -> anthropic.Anthropic | None:
    """Lazy Anthropic client. ``None`` when no API key is configured."""
    settings = get_settings()
    if not settings.ANTHROPIC_API_KEY:
        return None
    return anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)


# --- Session 7 live: one factory per chunking strategy (§7) ----------------
# The no-API strategies are plain singletons. The LLM-backed ones raise a clear
# error if their key is missing; the comparison endpoint maps that to a 500.


@lru_cache
def get_fixed_size_chunker() -> FixedSizeChunker:
    return FixedSizeChunker()


@lru_cache
def get_recursive_chunker() -> RecursiveChunker:
    return RecursiveChunker()


@lru_cache
def get_sentence_window_chunker() -> SentenceWindowChunker:
    return SentenceWindowChunker()


@lru_cache
def get_hierarchical_chunker() -> HierarchicalChunker:
    return HierarchicalChunker()


@lru_cache
def get_semantic_chunker() -> SemanticChunker:
    settings = get_settings()
    # SemanticChunker raises a clear error if the OpenAI key is missing.
    return SemanticChunker(api_key=settings.OPENAI_API_KEY, model=settings.EMBEDDING_MODEL)


# NOT @lru_cache: these chunkers are rebuilt per /embeddings/compare request
# (construction is cheap — the underlying API clients stay singletons) so a
# runtime model override takes effect on the next comparison.
def get_propositional_chunker() -> PropositionalChunker:
    client = get_openai_client()
    if client is None:
        raise RuntimeError("PropositionalChunker requires OPENAI_API_KEY.")
    model = get_runtime_config().effective("PROPOSITIONAL_CHUNKER_MODEL")
    return PropositionalChunker(client=client, model=model)


def get_contextual_retrieval_chunker() -> ContextualRetrievalChunker:
    client = get_anthropic_client()
    if client is None:
        raise RuntimeError("ContextualRetrievalChunker requires ANTHROPIC_API_KEY.")
    model = get_runtime_config().effective("CONTEXTUAL_CHUNKER_MODEL")
    return ContextualRetrievalChunker(client=client, model=model)


# Registry: strategy name → factory. ``structural`` reuses ``get_chunker``.
# Order is the canonical comparison order used by "all".
CHUNKER_FACTORIES = {
    "structural": get_chunker,
    "fixed_size": get_fixed_size_chunker,
    "recursive": get_recursive_chunker,
    "sentence_window": get_sentence_window_chunker,
    "semantic": get_semantic_chunker,
    "propositional": get_propositional_chunker,
    "contextual_retrieval": get_contextual_retrieval_chunker,
    "hierarchical": get_hierarchical_chunker,
}
ALL_STRATEGIES = list(CHUNKER_FACTORIES)


def build_chunkers(names: list[str]) -> list[Chunker]:
    """Instantiate the requested chunkers by name.

    Raises ``KeyError`` for an unknown strategy and ``RuntimeError`` for a
    strategy whose API key is missing (both mapped to HTTP errors by the router).
    """
    chunkers: list[Chunker] = []
    for name in names:
        factory = CHUNKER_FACTORIES.get(name)
        if factory is None:
            raise KeyError(name)
        chunkers.append(factory())
    return chunkers


@lru_cache
def get_semantic_cache() -> EstimationSemanticCache | None:
    """Build the semantic cache, swallowing setup errors so the rest of the
    pipeline keeps working if Redis Stack / RediSearch is not available
    (e.g. running on vanilla redis:7-alpine)."""
    settings = get_settings()
    openai_client = get_openai_client()
    if openai_client is None:
        log.warning("semantic_cache_disabled", reason="no_openai_key")
        return None

    # We use redisvl's OpenAITextVectorizer; it lazy-loads the OpenAI client.
    try:
        from redisvl.utils.vectorize import OpenAITextVectorizer

        vectorizer = OpenAITextVectorizer(
            model=settings.EMBEDDING_MODEL,
            api_config={"api_key": settings.OPENAI_API_KEY},
        )
        redis_client = redis.from_url(settings.REDIS_URL, decode_responses=False)
        return EstimationSemanticCache(
            redis_client=redis_client,
            vectorizer=vectorizer,
            threshold=settings.SEMANTIC_CACHE_THRESHOLD,
            ttl=settings.SEMANTIC_CACHE_TTL,
            log_only=settings.SEMANTIC_CACHE_LOG_ONLY,
        )
    except Exception as exc:  # noqa: BLE001
        log.warning(
            "semantic_cache_disabled",
            reason="setup_failed",
            error_type=type(exc).__name__,
            error=str(exc)[:200],
        )
        return None


@lru_cache
def get_estimation_service() -> EstimationService:
    settings = get_settings()
    return EstimationService(
        llm_wrapper=get_llm_wrapper(),
        exact_cache=get_cache(),
        semantic_cache=get_semantic_cache(),
        openai_client=get_openai_client(),
        metadata_extractor_model=settings.METADATA_EXTRACTOR_MODEL,
        compression_model=settings.COMPRESSION_MODEL,
        anchor_detection_mode=settings.ANCHOR_DETECTION_MODE,
        conversational_prompt_version=settings.CONVERSATIONAL_PROMPT_VERSION,
        critic_model=settings.CRITIC_MODEL,
        boss_max_iterations=settings.BOSS_MAX_ITERATIONS,
        runtime_config=get_runtime_config(),
    )


def build_pseudonymizer(session):
    """Build a :class:`ConsistentPseudonymizer` backed by Postgres.

    Not a singleton — the mapping store wraps a Session, so callers (scripts,
    BackgroundTasks, tests) must pass their own. The analyzer (singleton via
    ``build_analyzer``) and the salt come from settings.
    """
    from app.ingestion.pii import (
        ConsistentPseudonymizer,
        PostgresMappingStore,
        build_analyzer,
    )

    settings = get_settings()
    return ConsistentPseudonymizer(
        analyzer=build_analyzer(),
        mapping_store=PostgresMappingStore(session),
        salt=settings.PSEUDONYM_HASH_SALT,
        faker_locale=settings.PSEUDONYM_FAKER_LOCALE,
        language="es",
    )


@lru_cache
def get_catalog() -> DataCatalog:
    """Load and cache the data-source catalog (Session 6).

    The catalog is read once at startup. Re-reading would invalidate the
    decisions baked into the running pipeline; rolling a new catalog version
    requires a process restart by design.
    """
    settings = get_settings()
    return load_catalog(settings.CATALOG_PATH)


@lru_cache
def get_filesystem_loader() -> FileSystemLoader:
    settings = get_settings()
    return FileSystemLoader(data_root=settings.INGESTION_DATA_ROOT)


@lru_cache
def get_parser_registry() -> ParserRegistry:
    """Registry of parsers available in this branch. XLSX/DOCX/PDF parsers
    live in ``guides/session-06-reference/`` and are not registered here."""
    return default_registry()


# --- Session 12: hand-rolled estimation agent ------------------------------
# The composition root is the ONLY place allowed to wire a `generation/rag`
# capability into the `agentic` layer (siblings never import each other). So the
# real `search_budgets` backend — embed → hybrid+rerank retrieve → map — is built
# here and injected into the agent as a plain async callable.


def _map_chunk_to_item(chunk) -> dict:
    """Map a ``RetrievedChunk`` onto the canonical agent search-item shape.

    Same shape as the offline stub (``reference_retrieval.py``) so the agent code
    is identical whether it runs on the real pipeline or the stub.
    """
    return {
        "id": chunk.id,
        "content_preview": chunk.content[:300],
        "sector": chunk.sector,
        "budget_id": chunk.budget_id,
        "estimated_hours": chunk.estimated_hours,
        "distance": round(chunk.distance, 4),
    }


def build_budget_retriever(embedder, reranker):
    """Build the async retrieval backend that ``search_budgets`` wraps.

    Reuses the Session 9/10 hybrid + rerank pipeline unchanged: embed the query,
    recall a wide candidate set, rerank, and map the surviving chunks onto the
    agent item shape. Returns ``(query, filters) -> list[item]``.
    """
    import asyncio

    from app.generation.rag.retrieval.collections import Collection
    from app.generation.rag.retrieval.pipeline import retrieve

    settings = get_settings()

    async def retriever(query: str, filters: dict | None) -> list[dict]:
        sectors = filters.get("sectors") if filters else None
        embedding = await asyncio.to_thread(embedder.embed_one, query)
        result = await retrieve(
            query_embedding=embedding,
            query_text=query,
            search_mode="hybrid",
            rerank=True,
            top_k=settings.RETRIEVAL_TOP_K,
            recall_k=settings.RETRIEVAL_RECALL_TOP_K,
            rerank_top_n=settings.RERANK_TOP_N,
            distance_threshold=settings.RETRIEVAL_DISTANCE_THRESHOLD,
            rrf_k=settings.RRF_K,
            collection=Collection.BUDGET,
            sectors=sectors,
            chunk_types=["budget_component"],
            reranker=reranker,
        )
        return [_map_chunk_to_item(chunk) for chunk in result.chunks]

    return retriever


def get_estimation_agent(retriever=None, *, model=None, reasoning_effort=None):
    """Build the Session 12 :class:`EstimationAgent`.

    ``retriever`` is injectable: the run script passes the offline stub for
    ``--stub``; when ``None`` we wire the real hybrid+rerank pipeline. ``model`` /
    ``reasoning_effort`` override the settings defaults (the run script's
    ``--model`` flag). Not a singleton — cheap to build and the retriever varies
    per caller.
    """
    from app.generation.agentic.agent_loop import EstimationAgent

    settings = get_settings()
    client = get_openai_client()
    if client is None:
        raise RuntimeError("The estimation agent requires OPENAI_API_KEY.")
    if retriever is None:
        embedder = get_embedder()
        if embedder is None:
            raise RuntimeError("The estimation agent requires an embedder (OPENAI_API_KEY).")
        retriever = build_budget_retriever(embedder, get_reranker())
    return EstimationAgent(
        client=client,
        retriever=retriever,
        model=model or settings.AGENT_MODEL,
        reasoning_effort=reasoning_effort or settings.AGENT_REASONING_EFFORT,
        max_iterations=settings.AGENT_MAX_ITERATIONS,
    )


@lru_cache
def get_session_store() -> SessionStore:
    """In-memory store of conversational sessions. Singleton per worker.

    State lives in process memory: docker compose restart clears it, and
    workers > 1 each get their own copy. Documented limitation for the
    Session 5 exercise; persistence is module-3 territory.
    """
    settings = get_settings()
    return SessionStore(max_turns=settings.MAX_CONVERSATION_TURNS)
