"""Auto-select available Gemini models instead of hardcoding one.

Google renames and retires model ids (we hit 404s on both `text-embedding-004`
and `gemini-2.5-flash`), so by default we ask the API what this key can actually
use and pick the best match from a preference list. An explicit `GEMINI_MODEL` /
`GEMINI_EMBED_MODEL` (anything other than "auto") always wins.
"""

from __future__ import annotations

from dataclasses import replace

# Preference order, best first. We prefer models with a generous free tier and
# stable tool-calling for the chat model.
# Prefer the self-updating alias and the current flash model first; older ids
# (2.0/2.5) are being retired and may 404 even when the API still lists them.
PREFERRED_CHAT = [
    "gemini-flash-latest",
    "gemini-3.6-flash",
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-1.5-flash",
]
PREFERRED_EMBED = [
    "gemini-embedding-001",
    "text-embedding-004",
    "text-embedding-005",
]

_FALLBACK_CHAT = "gemini-2.0-flash"
_FALLBACK_EMBED = "gemini-embedding-001"


def _actions(model) -> list[str]:
    # google-genai names this `supported_actions`; be defensive across versions.
    return list(
        getattr(model, "supported_actions", None)
        or getattr(model, "supported_generation_methods", None)
        or []
    )


def _short(name: str) -> str:
    return name.split("/")[-1] if name else name


def _pick(available: dict[str, object], preferred: list[str], keyword: str, fallback: str) -> str:
    for p in preferred:
        if p in available:
            return p
    # No preferred id available: take any model whose name mentions the keyword.
    for name in available:
        if keyword in name:
            return name
    return next(iter(available), fallback)


def resolve_models(api_key: str, *, chat: str = "auto", embed: str = "auto") -> tuple[str, str]:
    """Return concrete (chat_model, embed_model), querying the API only for 'auto'."""
    if chat != "auto" and embed != "auto":
        return chat, embed

    try:
        from google import genai

        client = genai.Client(api_key=api_key)
        models = list(client.models.list())
    except Exception:
        # If discovery fails, fall back to sane defaults for any "auto" slot.
        return (
            _FALLBACK_CHAT if chat == "auto" else chat,
            _FALLBACK_EMBED if embed == "auto" else embed,
        )

    gen = {}
    emb = {}
    for m in models:
        actions = _actions(m)
        name = _short(getattr(m, "name", "") or "")
        if not name:
            continue
        if "generateContent" in actions:
            gen[name] = m
        if "embedContent" in actions:
            emb[name] = m

    resolved_chat = chat if chat != "auto" else _pick(gen, PREFERRED_CHAT, "flash", _FALLBACK_CHAT)
    resolved_embed = embed if embed != "auto" else _pick(emb, PREFERRED_EMBED, "embedding", _FALLBACK_EMBED)
    return resolved_chat, resolved_embed


def resolve_config_models(config):
    """Return a copy of config with any 'auto' model fields resolved.

    No-op when there is no API key (offline / TF-IDF) or nothing is set to auto.
    """
    if not config.has_api_key:
        return config
    if config.gemini_model != "auto" and config.gemini_embed_model != "auto":
        return config
    chat, embed = resolve_models(
        config.gemini_api_key, chat=config.gemini_model, embed=config.gemini_embed_model
    )
    return replace(config, gemini_model=chat, gemini_embed_model=embed)
