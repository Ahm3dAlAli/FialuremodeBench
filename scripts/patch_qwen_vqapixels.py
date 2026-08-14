"""Force a hard max_pixels cap on EVERY image in VLMEvalKit's Qwen2VL VQA path.
The constructor default isn't applied per-image (only OCRBench), so dynamic
resolution blows up (11.9GB allocations) on high-res math/chart images. Insert a
clamp right where each image item is built in _prepare_content."""
import re
p = "/local/scratch/alali/VLMEvalKit/vlmeval/vlm/qwen2_vl/model.py"
s = open(p).read()
if "FMB_PIXEL_CLAMP" in s:
    print("already patched"); raise SystemExit
# anchor: the line that constructs an image item
lines = s.split("\n")
out = []
patched = False
for ln in lines:
    out.append(ln)
    if "ensure_image_url(s['value'])" in ln and "item" in ln and not patched:
        indent = ln[:len(ln) - len(ln.lstrip())]
        out.append(f"{indent}item['max_pixels'] = 512 * 28 * 28  # FMB_PIXEL_CLAMP")
        out.append(f"{indent}item['min_pixels'] = 4 * 28 * 28")
        patched = True
open(p, "w").write("\n".join(out))
print("PATCHED" if patched else "ANCHOR_NOT_FOUND")
import glob, os
for f in glob.glob("/local/scratch/alali/VLMEvalKit/vlmeval/vlm/qwen2_vl/__pycache__/*.pyc"):
    os.remove(f)
