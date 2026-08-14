"""Load VLMEvalKit Qwen2-VL in 4-bit (bitsandbytes) so a 7B fits one 11GB GPU
with headroom on a contested box. Matches the paper's 4-bit protocol."""
import glob
p = glob.glob("/local/scratch/alali/VLMEvalKit/vlmeval/vlm/qwen2_vl/model.py")[0]
s = open(p).read()
old = ('model_path, torch_dtype=\'auto\', device_map="auto", '
       'max_memory={i: "5GiB" for i in range(__import__("torch").cuda.device_count())}, '
       'attn_implementation=\'sdpa\'')
new = ('model_path, torch_dtype=__import__("torch").float16, device_map="auto", '
       'quantization_config=__import__("transformers").BitsAndBytesConfig('
       'load_in_4bit=True, bnb_4bit_compute_dtype=__import__("torch").float16, '
       'bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True), '
       'attn_implementation=\'sdpa\'')
if old in s:
    s = s.replace(old, new); open(p, "w").write(s); print("4BIT_PATCHED")
elif "load_in_4bit=True" in s:
    print("ALREADY_4BIT")
else:
    print("PATTERN_NOT_FOUND — inspect model.py from_pretrained")
for f in glob.glob("/local/scratch/alali/VLMEvalKit/vlmeval/vlm/qwen2_vl/__pycache__/*.pyc"):
    import os; os.remove(f)
