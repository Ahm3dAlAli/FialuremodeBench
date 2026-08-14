"""Patch VLMEvalKit Qwen2-VL loader for 11GB shared GPUs: spread model across
GPUs with headroom (max_memory) + moderate vision-token cap."""
import re, sys, glob
paths = glob.glob("/local/scratch/alali/VLMEvalKit/vlmeval/vlm/qwen2_vl/model.py")
p = paths[0]
s = open(p).read()
# 1) add max_memory to the from_pretrained device_map="auto" call
if "max_memory=" not in s:
    s = s.replace(
        'device_map="auto", attn_implementation=\'sdpa\'',
        'device_map="auto", max_memory={i: "5GiB" for i in range(__import__("torch").cuda.device_count())}, attn_implementation=\'sdpa\'')
# 2) firmer default max_pixels
s = s.replace("else 1024 * 28 * 28", "else 768 * 28 * 28")
open(p, "w").write(s)
print("max_memory present:", "max_memory=" in s)
print("cap 768:", "768 * 28 * 28" in s)
for f in glob.glob("/local/scratch/alali/VLMEvalKit/vlmeval/vlm/qwen2_vl/__pycache__/*.pyc"):
    import os; os.remove(f)
print("PATCH_OK")
