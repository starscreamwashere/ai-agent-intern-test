"""Factory helpers to assemble a SupportAgent from configuration.

Keeps wiring in one place so the CLI, the evaluation harness, and tests build
the agent the same way.
"""

from __future__ import annotations

from .agent import SupportAgent
from .config import Config, load_config
from .embeddings import build_embedder
from .llm import GeminiClient, LLMClient
from .observability import Tracer
from .order_lookup import OrderStore
from .retrieval import KnowledgeBase


def build_knowledge_base(config: Config | None = None) -> KnowledgeBase:
    config = config or load_config()
    embedder = build_embedder(
        config.effective_embed_backend,
        api_key=config.gemini_api_key,
        model=config.gemini_embed_model,
    )
    return KnowledgeBase.build(embedder)


def build_agent(
    config: Config | None = None,
    *,
    llm: LLMClient | None = None,
    kb: KnowledgeBase | None = None,
    tracer: Tracer | None = None,
) -> SupportAgent:
    """Build a fresh single-session agent.

    `llm` and `kb` can be injected (tests use a mock LLM / offline KB). When no
    llm is given, a GeminiClient is created and requires GEMINI_API_KEY.
    """
    config = config or load_config()
    kb = kb or build_knowledge_base(config)

    if llm is None:
        if not config.has_api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. Add it to .env to run the live agent, "
                "or inject a mock LLM for offline use."
            )
        llm = GeminiClient(api_key=config.gemini_api_key, model=config.gemini_model)

    return SupportAgent(kb, llm, OrderStore(), top_k=config.kb_top_k, tracer=tracer)
