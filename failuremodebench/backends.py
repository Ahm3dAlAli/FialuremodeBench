"""VLM inference backends for the recognition runner.

A backend turns (PIL image, prompt) -> answer text. Two implementations:

  * VLMEvalKitBackend -- wraps VLMEvalKit's model zoo (Qwen2-VL, InternVL2.5,
    LLaVA-1.6, ...). This is the GPU path used on rolf; VLMEvalKit owns the
    per-model chat templates, image tokenisation and 4-bit loading.
  * ApiVLMBackend -- wraps providers.py (Claude / GPT-4o). CPU-only, used to
    smoke-test the recognition pipeline locally and as the paper's "closed" VLM.

VLMEvalKit is imported lazily so the package (and its unit tests) work without a
CUDA install.
"""
from __future__ import annotations

from typing import Optional

from .config import MODELS, ModelSpec


class VLMBackend:
    def generate(self, image, prompt: str) -> str:
        raise NotImplementedError


class ApiVLMBackend(VLMBackend):
    def __init__(self, provider_name="anthropic", model: Optional[str] = None):
        from .providers import get_provider
        self.provider = get_provider(provider_name, model)

    def generate(self, image, prompt: str) -> str:
        return self.provider.complete(
            system="You are a precise visual classifier.",
            user=prompt, image=image, max_tokens=64, temperature=0.0).strip()


class VLMEvalKitBackend(VLMBackend):
    def __init__(self, vlmevalkit_name: str, load_4bit: bool = False):
        from vlmeval.config import supported_VLM  # lazy: GPU only
        if vlmevalkit_name not in supported_VLM:
            raise KeyError(
                f"{vlmevalkit_name!r} not in VLMEvalKit supported_VLM "
                f"(have {len(supported_VLM)} models). Check the name / version.")
        self.model = supported_VLM[vlmevalkit_name]()

    def generate(self, image, prompt: str) -> str:
        import tempfile
        # VLMEvalKit's generate takes a message list of dicts (image path + text).
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            image.convert("RGB").save(f.name)
            msg = [{"type": "image", "value": f.name},
                   {"type": "text", "value": prompt}]
            try:
                return str(self.model.generate(message=msg, dataset="ImageNet")).strip()
            except TypeError:  # older signature
                return str(self.model.generate(msg)).strip()


def build_backend(model_key: str, prefer_api: bool = False,
                  api_provider="anthropic") -> tuple[VLMBackend, ModelSpec]:
    """Construct the right backend for a model key from the MODELS registry."""
    spec = MODELS[model_key]
    if spec.family == "api" or prefer_api:
        return ApiVLMBackend(api_provider), spec
    return VLMEvalKitBackend(spec.vlmevalkit_name, spec.load_4bit), spec
