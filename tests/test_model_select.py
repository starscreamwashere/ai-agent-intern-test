"""Model auto-selection tests (offline; no API call for the paths tested)."""

from dataclasses import replace

from aster_agent.config import load_config
from aster_agent.model_select import _pick, resolve_config_models, resolve_models


def test_pick_prefers_first_available():
    available = {"gemini-2.5-flash": 1, "gemini-2.0-flash": 1}
    assert _pick(available, ["gemini-2.0-flash", "gemini-2.5-flash"], "flash", "fallback") == "gemini-2.0-flash"


def test_pick_falls_back_to_keyword_match():
    available = {"some-new-flash-model": 1, "other": 1}
    assert _pick(available, ["not-present"], "flash", "fb") == "some-new-flash-model"


def test_pick_uses_fallback_when_empty():
    assert _pick({}, ["x"], "flash", "gemini-2.0-flash") == "gemini-2.0-flash"


def test_resolve_models_skips_api_when_both_explicit():
    # Both explicit -> returns them unchanged without touching the network.
    chat, embed = resolve_models("no-key-needed", chat="gemini-2.0-flash", embed="gemini-embedding-001")
    assert (chat, embed) == ("gemini-2.0-flash", "gemini-embedding-001")


def test_resolve_config_models_noop_without_api_key():
    cfg = load_config()
    cfg = replace(cfg, gemini_api_key=None, gemini_model="auto", gemini_embed_model="auto")
    resolved = resolve_config_models(cfg)
    # No key -> unchanged (offline/TF-IDF path).
    assert resolved.gemini_model == "auto"
    assert resolved is cfg
