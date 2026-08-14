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


class CLIPBackend(VLMBackend):
    """open_clip zero-shot classifier -- the contrastive contrast to generative
    VLMs. Not a generator: it scores an image against the dataset's label set and
    returns the argmax label, so it structurally cannot emit an out-of-set
    super-category (no F7). Exposes set_labels()/classify() for the CLIP runner.
    """
    def __init__(self, spec_str: str, fallbacks=None, device: Optional[str] = None):
        import open_clip
        import torch
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        specs = [spec_str] + list(fallbacks or [])
        last = None
        self.model = None
        for s in specs:
            arch, pretrained = s.split(":", 1)
            try:
                self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                    arch, pretrained=pretrained)
                self.tokenizer = open_clip.get_tokenizer(arch)
                self.spec_used = s
                break
            except Exception as e:  # arch/pretrained tag not in this open_clip build
                last = e
        if self.model is None:
            raise RuntimeError(f"open_clip could not load any of {specs}: {last}")
        self.model = self.model.to(self.device).eval()
        self.labels = None
        self.text_features = None

    def set_labels(self, labels, templates=("a photo of a {}.", "a photo of the {}.")):
        """Precompute (once per dataset) the mean-ensembled text embedding per label."""
        import torch
        self.labels = list(labels)
        embs = []
        with torch.no_grad():
            for lab in self.labels:
                toks = self.tokenizer([t.format(lab) for t in templates]).to(self.device)
                tf = self.model.encode_text(toks)
                tf = tf / tf.norm(dim=-1, keepdim=True)
                e = tf.mean(0)
                embs.append(e / e.norm())
            self.text_features = torch.stack(embs)

    def classify(self, image):
        """Return (pred_label, cosine_score) for the best-matching label."""
        import torch
        with torch.no_grad():
            img = self.preprocess(image.convert("RGB")).unsqueeze(0).to(self.device)
            f = self.model.encode_image(img)
            f = f / f.norm(dim=-1, keepdim=True)
            sims = (f @ self.text_features.T).squeeze(0)
            idx = int(sims.argmax())
            return self.labels[idx], float(sims[idx])

    def generate(self, image, prompt: str) -> str:  # not used; keeps the interface
        lab, _ = self.classify(image)
        return lab


def build_backend(model_key: str, prefer_api: bool = False,
                  api_provider="anthropic") -> tuple[VLMBackend, ModelSpec]:
    """Construct the right backend for a model key from the MODELS registry."""
    spec = MODELS[model_key]
    if spec.family == "api" or prefer_api:
        return ApiVLMBackend(api_provider), spec
    if spec.family == "clip":
        from .config import CLIP_FALLBACKS
        return CLIPBackend(spec.hf_id, CLIP_FALLBACKS.get(model_key)), spec
    return VLMEvalKitBackend(spec.vlmevalkit_name, spec.load_4bit), spec
