from huggingface_hub import HfApi
from datasets import load_dataset_builder
api = HfApi()
def search(q, n=10):
    print(f"\n### SEARCH: {q}")
    try:
        for d in api.list_datasets(search=q, limit=n):
            print(f"   {d.id:60} dl={getattr(d,'downloads','?')}")
    except Exception as e:
        print("   err", e)
for q in ["imagenet-sketch", "imagenet sketch", "resisc45"]:
    search(q)
print("\n### split sizes for candidates")
for r in ["songweig/imagenet_sketch","mrm8488/ImageNet-Sketch","clip-benchmark/wds_imagenet-sketch",
          "timm/resisc45","jonathan-roberts1/RESISC45","tanganke/resisc45","blanchon/RESISC45"]:
    try:
        b=load_dataset_builder(r)
        print(f"   [{r}] {{k:v.num_examples for splits}} ->", {k:v.num_examples for k,v in (b.info.splits or {}).items()})
    except Exception as e:
        print(f"   [{r}] ERR {type(e).__name__}: {str(e)[:90]}")
