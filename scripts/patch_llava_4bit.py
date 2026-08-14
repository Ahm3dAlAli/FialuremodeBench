"""Inject 4-bit + device_map into VLMEvalKit LLaVA-Next loader so a 7B fits an
11GB card, and disable the hard flash-attn requirement."""
p="/local/scratch/alali/VLMEvalKit/vlmeval/vlm/llava/llava.py"
s=open(p).read()
if "FMB_LLAVA_4BIT" not in s:
    q=('low_cpu_mem_usage=True, device_map="auto", '
       'quantization_config=__import__("transformers").BitsAndBytesConfig('
       'load_in_4bit=True, bnb_4bit_compute_dtype=__import__("torch").float16, '
       'bnb_4bit_quant_type="nf4"),  # FMB_LLAVA_4BIT')
    s=s.replace("low_cpu_mem_usage=True,", q)
    s=s.replace("use_flash_attention_2=True", "use_flash_attention_2=False")
    open(p,"w").write(s)
    print("PATCHED", s.count("FMB_LLAVA_4BIT"), "load sites")
else:
    print("already patched")
import glob,os
for f in glob.glob("/local/scratch/alali/VLMEvalKit/vlmeval/vlm/llava/__pycache__/*.pyc"): os.remove(f)
