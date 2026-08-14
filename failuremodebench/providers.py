"""Multimodal LLM provider abstraction.

Used in two places:
  * the failure-mode JUDGE (`judge.py`) -- classify an error into F1-F8;
  * an optional API VLM *under test* (a "closed" model in the paper), and a
    handy CPU-only backend for smoke-testing the whole pipeline without a GPU.

Providers take (system_prompt, user_text, optional PIL image) and return raw
text. Keys are read from the environment so nothing secret is committed:
  ANTHROPIC_API_KEY  (default judge: Claude)
  OPENAI_API_KEY     (--provider openai)
"""
from __future__ import annotations

import base64
import io
import os
from typing import Optional

try:
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None


def _img_to_b64(image, fmt="PNG", max_side=1024) -> tuple[str, str]:
    """PIL image -> (base64_str, media_type). Downscales huge images."""
    if image.mode != "RGB":
        image = image.convert("RGB")
    if max(image.size) > max_side:
        s = max_side / max(image.size)
        image = image.resize((int(image.width * s), int(image.height * s)))
    buf = io.BytesIO()
    image.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode(), f"image/{fmt.lower()}"


class Provider:
    name = "base"

    def complete(self, system: str, user: str, image=None,
                 max_tokens: int = 1024, temperature: float = 0.0) -> str:
        raise NotImplementedError


class AnthropicProvider(Provider):
    name = "anthropic"

    def __init__(self, model: str = "claude-opus-4-8"):
        import anthropic
        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Export it before running the "
                "judge (see docs/LAUNCH.md).")
        self.client = anthropic.Anthropic(api_key=key)
        self.model = model

    def complete(self, system, user, image=None, max_tokens=1024, temperature=0.0):
        content: list = []
        if image is not None:
            b64, media = _img_to_b64(image)
            content.append({"type": "image", "source": {
                "type": "base64", "media_type": media, "data": b64}})
        content.append({"type": "text", "text": user})
        msg = self.client.messages.create(
            model=self.model, max_tokens=max_tokens, temperature=temperature,
            system=system, messages=[{"role": "user", "content": content}])
        return "".join(b.text for b in msg.content if b.type == "text")


class OpenAIProvider(Provider):
    name = "openai"

    def __init__(self, model: str = "gpt-4o"):
        from openai import OpenAI
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY is not set.")
        # OPENAI_BASE_URL lets this target OpenAI-compatible gateways (OpenRouter,
        # vLLM, ...). e.g. https://openrouter.ai/api/v1 with an OpenRouter key.
        base = os.environ.get("OPENAI_BASE_URL")
        self.client = OpenAI(api_key=key, base_url=base) if base else OpenAI(api_key=key)
        self.model = model

    def complete(self, system, user, image=None, max_tokens=1024, temperature=0.0):
        parts: list = [{"type": "text", "text": user}]
        if image is not None:
            b64, media = _img_to_b64(image)
            parts.append({"type": "image_url",
                          "image_url": {"url": f"data:{media};base64,{b64}"}})
        resp = self.client.chat.completions.create(
            model=self.model, max_tokens=max_tokens, temperature=temperature,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": parts}])
        return resp.choices[0].message.content or ""


class LocalVLMProvider(Provider):
    """Open-weight VLM judge, run locally via VLMEvalKit (no API key).

    Reuses the same VLMEvalKit model zoo as the models-under-test, so the judge
    runs entirely on rolf's GPUs. Default is Qwen2.5-VL-7B — strong and distinct
    from the tested Qwen2-VL-7B, avoiding the judge/test self-preference confound.
    """
    name = "local"

    def __init__(self, model: str = "Qwen2.5-VL-7B-Instruct"):
        from .backends import VLMEvalKitBackend
        self.backend = VLMEvalKitBackend(model)
        self.model = model

    def complete(self, system, user, image=None, max_tokens=1024, temperature=0.0):
        prompt = f"{system}\n\n{user}" if system else user
        # VLMEvalKit models take (image, prompt); text-only if no image.
        return self.backend.generate(image, prompt)


class LocalLLMEndpointProvider(OpenAIProvider):
    """Open-weight LLM served behind an OpenAI-compatible endpoint (vLLM/Ollama).

    Point OPENAI_BASE_URL at the server (e.g. http://localhost:8000/v1) and set
    the model name. Lets you judge with any served open model, image-capable or
    not. OPENAI_API_KEY can be any placeholder for local servers.
    """
    name = "local_endpoint"

    def __init__(self, model="Qwen2.5-VL-7B-Instruct"):
        from openai import OpenAI
        base = os.environ.get("OPENAI_BASE_URL", "http://localhost:8000/v1")
        key = os.environ.get("OPENAI_API_KEY", "EMPTY")
        self.client = OpenAI(api_key=key, base_url=base)
        self.model = model


class EchoProvider(Provider):
    """Offline stub for unit tests: deterministic, no network. Returns a fixed
    F8 verdict so the parser/aggregation paths can be exercised without a key."""
    name = "echo"

    def complete(self, system, user, image=None, max_tokens=1024, temperature=0.0):
        return ('{"failure_mode":"F8","confidence":0.5,'
                '"rationale":"echo-stub: no live judge configured",'
                '"secondary_mode":"none"}')


def get_provider(name: str = "anthropic", model: Optional[str] = None) -> Provider:
    name = (name or "anthropic").lower()
    if name == "anthropic":
        return AnthropicProvider(model or "claude-opus-4-8")
    if name == "openai":
        return OpenAIProvider(model or "gpt-4o")
    if name in ("local", "vlm", "local_vlm"):
        return LocalVLMProvider(model or "Qwen2.5-VL-7B-Instruct")
    if name == "local_endpoint":
        return LocalLLMEndpointProvider(model or "Qwen2.5-VL-7B-Instruct")
    if name == "echo":
        return EchoProvider()
    raise ValueError(f"unknown provider {name!r}")
