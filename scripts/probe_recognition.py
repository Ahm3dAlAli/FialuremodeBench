"""Find real HF repos + split sizes for the problem recognition datasets (metadata only)."""
from huggingface_hub import HfApi
from datasets import load_dataset_builder
api = HfApi()

def search(q, n=8):
    print(f"\n### SEARCH: {q}")
    try:
        for d in api.list_datasets(search=q, limit=n, sort="downloads", direction=-1):
            print(f"   {d.id:55} dl={getattr(d,'downloads','?')}")
    except Exception as e:
        print("   search err", e)

def splits(repo, config=None):
    try:
        b = load_dataset_builder(repo, config) if config else load_dataset_builder(repo)
        sp = {k: v.num_examples for k, v in (b.info.splits or {}).items()}
        cfgs = list(b.builder_configs) if hasattr(b, "builder_configs") else []
        print(f"   [{repo} cfg={config}] splits={sp} configs={cfgs}")
    except Exception as e:
        print(f"   [{repo} cfg={config}] ERR {type(e).__name__}: {str(e)[:120]}")

for q in ["imagenet sketch", "food101", "imagenetv2", "resisc45", "describable textures dtd"]:
    search(q)

print("\n### candidate split sizes")
for r, c in [("tanganke/dtd", None), ("tanganke/resisc45", None),
             ("clip-benchmark/wds_imagenet_sketch", None),
             ("clip-benchmark/wds_imagenet-a", None),
             ("clip-benchmark/wds_vtab-dtd", None),
             ("ethz/food101", None), ("food101", None),
             ("vaishaal/ImageNetV2", None)]:
    splits(r, c)
