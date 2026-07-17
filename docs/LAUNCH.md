# Launching the full FailureModeBench evaluation

The harness is built and validated offline. To produce the paper's numbers two
external things are required that this machine cannot self-provide:

## Blocker 1 — reach the rolf GPU box

`rolf.ifi.uzh.ch` (130.60.144.130) currently **times out on port 22** from this
laptop. UZH IFI GPU hosts are only reachable from the campus network. Fix one of:

- connect the **UZH VPN** that routes 130.60.144.0/… (the currently-up `utun*`
  tunnels do not), **or**
- SSH via the department **jump host**, e.g. add to `~/.ssh/config`:
  ```
  Host rolf
      HostName rolf.ifi.uzh.ch
      User alali
      ProxyJump alali@<login-host>.ifi.uzh.ch
  ```

Verify: `ssh rolf 'hostname; nvidia-smi --query-gpu=name,memory.total --format=csv'`
Login needs **password + OTP** — a human must complete it; the sync script
authenticates once and multiplexes.

> GPU note: rolf's RTX 2080 Ti has **11 GB**. The registry already sets
> `load_4bit=True` for the 7B/8B/13B models. For Qwen2-VL cap the vision tokens
> (dynamic resolution) to avoid OOM. Prefer `internvl25_4b` / `llava16_7b` first;
> add 13B / size-variant ablations once the 11 GB budget is confirmed.

## Blocker 2 — a judge API key

The failure-mode judge (paper's core contribution) defaults to Claude. Set on
rolf before Stage 3:

```bash
export ANTHROPIC_API_KEY=sk-ant-...        # or: --provider openai + OPENAI_API_KEY
```

No key on this machine (only a Claude Code OAuth entry, which is not an API key).
For a keyless plumbing test use `--echo` (returns a stub verdict).

## Verify VLMEvalKit dataset/model ids (do this once on rolf)

VLMEvalKit ids drift across versions. After `bootstrap_rolf.sh`, confirm the ids
in `config.py` (`vlmevalkit_name`) exist, and **fix two known-uncertain ones**:

```bash
python -c "from vlmeval.config import supported_VLM; print(len(supported_VLM))"
python -c "from vlmeval.dataset import SUPPORTED_DATASETS; print([d for d in SUPPORTED_DATASETS if 'CRPE' in d or 'Math' in d or 'RealWorld' in d])"
```
- `crpe` → pick the right CRPE split id (e.g. `CRPE_RELATION` / `CRPE_EXIST`).
- `capture` → **placeholder** `COCO_VAL` in config.py; set to the actual
  counterfactual "Capture" dataset id (or drop from the VQA set if unsupported).

## Run

```bash
# from laptop (once rolf reachable):
bash scripts/sync_to_rolf.sh
# on rolf:
cd ~/FailureModeBench && bash scripts/bootstrap_rolf.sh
export ANTHROPIC_API_KEY=sk-ant-...
screen -S fmb
conda activate fmb
# smoke first (20 samples/dataset) to shake out ids/memory:
RUN=smoke LIMIT=20 N_PER_FAMILY=20 CUDA_VISIBLE_DEVICES=0 nice -n 15 bash scripts/run_rolf.sh
# then the full run:
RUN=main CUDA_VISIBLE_DEVICES=0 nice -n 15 bash scripts/run_rolf.sh 2>&1 | tee run_rolf.log
```

Pull results back:
```bash
rsync -av rolf:~/FailureModeBench/results/main/ results/main/
```

## Rough cost / time

- Inference: ~185k recognition images + ~19k VQA items × 3 models on one 2080 Ti
  is the long pole — plan for **days**; parallelise across free GPUs
  (`CUDA_VISIBLE_DEVICES` per model in separate screens).
- Judge: ≤ 200 errors/family × 10 families × ~1 image call each ≈ a few thousand
  Claude calls total — **single-digit dollars to low-tens**, minutes-to-hours.
