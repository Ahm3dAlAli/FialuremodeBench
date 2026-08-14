"""Build a blind human-annotation study for judge validation (ICLR's key gap).

Samples N judged errors per family from a run's corpus, and emits a SELF-CONTAINED
HTML page where a human assigns each error one of F1-F8 -- WITHOUT seeing the LLM
judge's verdict (blind). The annotator labels in-browser and downloads a JSON;
`scripts/human_agreement.py` then computes human-vs-judge Cohen's kappa.

    python scripts/make_human_study.py --run main2b --n-per-family 20

Images: if an error's image_ref exists locally it is embedded (base64); otherwise
the item is text-only (gold/prediction/question), which still supports most modes.
Pull images first with:  rsync -az rolf:.../results/<run>/images/ results/<run>/images/
"""
import argparse
import base64
import json
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
from failuremodebench import taxonomy
from failuremodebench.judge import read_jsonl


def _img_data_uri(path, max_px=384, quality=78):
    """Embed a downscaled JPEG data URI. Images display at <=320px, so full-res
    PNGs just bloat the page (a 700-card study is ~400 MB unscaled); downscaling to
    max_px + JPEG keeps it browser-friendly (~10x smaller) with no visible loss."""
    if not (path and os.path.exists(path)):
        return ""
    try:
        from PIL import Image
        import io
        im = Image.open(path).convert("RGB")
        im.thumbnail((max_px, max_px))
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=quality)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
    except Exception:
        try:
            return "data:image/png;base64," + base64.b64encode(open(path, "rb").read()).decode()
        except Exception:
            return ""


def image_index(run_dir):
    """(model, dataset, sample_id) -> image_ref, read from the prediction files."""
    import glob
    idx = {}
    for f in glob.glob(os.path.join(run_dir, "predictions", "*.jsonl")):
        for r in read_jsonl(f):
            if r.get("image_ref"):
                idx[(r.get("model"), r.get("dataset"), r.get("sample_id"))] = r["image_ref"]
    return idx


def sample(corpus, n_per_family, run_dir, seed=0):
    from collections import defaultdict
    idx = image_index(run_dir)
    byf = defaultdict(list)
    for r in corpus:
        ref = idx.get((r.get("model"), r.get("dataset"), r.get("sample_id")), "")
        # VQA predictions saved no image; request it by naming convention so the
        # rolf-side extractor (scripts/extract_vqa_images.py) can recover it from
        # the VLMEvalKit TSV base64 image column.
        if not ref and r.get("model") and r.get("sample_id"):
            ref = f"{r['model']}-{r['sample_id']}.png"
        r["image_ref"] = ref
        byf[r.get("family", "?")].append(r)
    out = []
    for fam in sorted(byf):
        items = sorted(byf[fam], key=lambda r: r.get("error_key", ""))
        out.extend(items[:n_per_family])
    return out


def _gold_html(e, choices):
    """Render the gold answer with the choice text for multiple-choice items,
    so 'A' is not shown bare. Falls back to the raw gold otherwise."""
    gold = str(e.get("gold", ""))
    ch = choices.get(e.get("sample_id", ""))
    if not ch or not ch.get("options"):
        return gold
    opts = ch["options"]
    letter = gold.strip()
    gtext = ch.get("answer_text") or opts.get(letter, "")
    olist = " &nbsp; ".join(f"<b>{L}.</b> {t}" for L, t in opts.items())
    head = f"{gold}. {gtext}" if gtext else gold
    return f"{head}<div style='color:#666;font-size:12px;margin-top:2px'>options: {olist}</div>"


def build_html(items, run_dir, prefill=None):
    import json as _json
    prefill = prefill or {}
    cpath = os.path.join(run_dir, "choices.json")
    choices = _json.load(open(cpath)) if os.path.exists(cpath) else {}
    # Plain-English "pick this when" cues per mode (annotator-facing).
    cues = {
        "F1": "The model mentions an object/thing that is simply NOT in the image.",
        "F2": "The model makes a verbal claim with no support in the image or question.",
        "F3": "The model seems to ignore the image, or answers about the wrong thing entirely.",
        "F4": "The model read the individual facts right but combined/computed them wrongly.",
        "F5": "The model invented a whole step, formula, or story from nothing.",
        "F6": "Right property, WRONG object — it swapped which thing has which attribute/class "
              "(e.g. a sibling/look-alike category).",
        "F7": "Right idea, TOO BROAD — it named a parent/super-category instead of the specific "
              "one (e.g. 'bird' for 'sparrow', 'fish' for 'tench').",
        "F8": "The answer/reasoning sounds fluent and logical but the conclusion is just false.",
    }
    rubric = "".join(
        f"<tr><td><b>{m.code}</b></td><td><b>{m.name}</b></td>"
        f"<td>{m.definition}<br><span style='color:#2a6fb0'>Pick when:</span> {cues.get(m.code,'')}"
        f"<br><span style='color:#888;font-size:12px'>e.g. {m.manifestation}</span></td></tr>"
        for m in taxonomy.FAILURE_MODES)
    # items carry the judge verdict as data-judge (hidden) for later scoring; the
    # annotator never sees it. Cards whose error_key is in `prefill` are pre-checked
    # (labels carried over from a prior pass) so only new cards need attention.
    cards = []
    for i, e in enumerate(items):
        pre = prefill.get(e.get("error_key", ""))
        img = _img_data_uri(os.path.join(run_dir, "images",
              os.path.basename(e.get("image_ref", "") or "")) if e.get("image_ref") else "")
        img_html = f'<img src="{img}" style="max-width:320px;max-height:320px;border:1px solid #ccc">' if img else '<i>(no image)</i>'
        opts = "".join(
            f'<label style="margin-right:10px"><input type="radio" name="q{i}" value="{m.code}"'
            f'{" checked" if pre == m.code else ""}> {m.code}</label>'
            for m in taxonomy.FAILURE_MODES)
        # Escape hatch: the model was actually correct and the DATASET gold is wrong
        # (label noise, e.g. SEED "gym" for an outdoor field). Not an F1-F8 failure;
        # excluded from kappa and reported as a mislabel rate.
        opts += (f'<label style="margin-left:14px;color:#c0392b"><input type="radio" '
                 f'name="q{i}" value="GW"{" checked" if pre == "GW" else ""}> '
                 f'⚠ gold wrong / model OK</label>')
        badge = ' <span class="pre">✓ carried over — verify or change</span>' if pre else ''
        cards.append(f'''
        <div class="card{' done' if pre else ''}" data-key="{e.get('error_key','')}" data-judge="{e.get('failure_mode','')}"
             data-family="{e.get('family','')}" data-idx="{i}">
          <div class="hd">#{i+1} &middot; {e.get('dataset','')} ({e.get('family','')}){badge}</div>
          <div class="bd">
            <div>{img_html}</div>
            <div class="txt">
              <div><b>Task/question:</b> {e.get('question','')[:400]}</div>
              <div><b>Gold answer:</b> {_gold_html(e, choices)}</div>
              <div><b>Model output:</b> {e.get('prediction','')[:300]}</div>
              <div class="opts">{opts}</div>
            </div>
          </div>
        </div>''')
    html = f'''<!doctype html><html><head><meta charset="utf-8">
<title>FailureModeBench — human annotation</title>
<style>
 body{{font-family:system-ui,Arial;margin:20px;max-width:1000px}}
 .card{{border:1px solid #ddd;border-radius:8px;padding:12px;margin:12px 0}}
 .hd{{font-weight:bold;color:#444;margin-bottom:8px}}
 .bd{{display:flex;gap:16px}} .txt div{{margin:4px 0}}
 .opts{{margin-top:8px}} table{{border-collapse:collapse}} td{{border:1px solid #eee;padding:3px 6px;font-size:13px}}
 #bar{{position:sticky;top:0;background:#fff;padding:10px 0;border-bottom:2px solid #333;z-index:10}}
 button{{padding:8px 16px;font-size:15px}}
 .card.done{{opacity:.5}} .pre{{color:#2a8f2a;font-size:12px;font-weight:normal}}
 body.hidepre .card.done{{display:none}}
 .intro{{background:#f7f9fc;border:1px solid #dbe4ee;border-radius:8px;padding:14px 18px;margin:12px 0}}
 .intro h2{{margin:.2em 0}} .intro li{{margin:4px 0}}
 .dist td{{font-size:13px}} .dist th{{background:#eef3fa;text-align:left;padding:4px 6px}}
 details>summary{{cursor:pointer;font-size:16px;margin:10px 0}}
</style></head><body>
<div id="bar">
 <b>Assign each error ONE failure mode (F1–F8), based only on the image + text below.</b>
 <button onclick="save()">⬇ Download annotations</button>
 <button onclick="document.body.classList.toggle('hidepre')">Toggle carried-over</button>
 <button onclick="next()">Next unlabeled ▾</button> <span id="prog"></span>
</div>

<div class="intro">
 <h2>What am I doing here?</h2>
 <p>Each card below shows a case where a vision-language model gave a <b>wrong</b> answer.
 Your job: look at the image + question + the correct ("gold") answer + what the model
 said, and decide <b>WHY it failed</b> — assign it exactly <b>one</b> failure mode (F1–F8).
 This builds a human-validated ground truth for an automated failure-diagnosis benchmark.
 It takes ~10–20s per card; you don't have to finish all of them.</p>
 <ol>
  <li>Read the image + task, the <b>gold</b> answer, and the <b>model output</b>.</li>
  <li>Pick the <b>single</b> mode that best explains the error — the <i>earliest</i>
      departure from the evidence (see the guide below; it's short).</li>
  <li>If the model was actually <b>right and the gold label is wrong</b>, pick the red
      <b>"⚠ gold wrong / model OK"</b> option instead.</li>
  <li>Use <b>Next unlabeled ▾</b> to jump ahead; click <b>Download annotations</b> when
      you stop, and send the file back. Partial is fine.</li>
 </ol>
</div>

<details open><summary><b>📖 Failure-mode guide — read this once (2 min)</b></summary>
<table><tr><th>Code</th><th>Mode</th><th>What it means · when to pick it</th></tr>{rubric}</table>
<p style="margin-top:12px"><b>The distinctions people actually get wrong:</b></p>
<table class="dist">
 <tr><th>If you're torn between…</th><th>The rule</th></tr>
 <tr><td><b>F7</b> vs <b>F6</b></td><td><b>F7 = right object, too broad</b> ("bird"→"sparrow").
   <b>F6 = right property, wrong object</b> (calls it a look-alike / sibling class, or puts
   the right attribute on the wrong thing).</td></tr>
 <tr><td><b>F1</b> vs <b>F2</b></td><td><b>F1</b> = a <i>visual</i> thing that isn't in the
   image. <b>F2</b> = a <i>verbal</i> claim with no support (not about a missing object).</td></tr>
 <tr><td><b>F4</b> vs <b>F8</b></td><td><b>F4</b> = a mechanical slip (right numbers/facts,
   wrong combination). <b>F8</b> = a fluent narrative that sounds right but the conclusion
   is false.</td></tr>
 <tr><td><b>F5</b> vs <b>F8</b></td><td><b>F5</b> = <i>invents</i> content from nothing.
   <b>F8</b> = keeps it plausible but the reasoning/answer is wrong.</td></tr>
 <tr><td><b>F3</b></td><td>Use when the model seems to <b>ignore the image</b> entirely or
   answer about a different image/modality.</td></tr>
</table>
<p style="color:#666;font-size:13px">Tip: for recognition tasks the two most common are
 <b>F7</b> (named a broader category) and <b>F6</b> (confused a similar category). For VQA,
 watch for <b>F1</b> (hallucinated object) and <b>F6</b> (attribute on the wrong thing).</p>
</details>
{''.join(cards)}
<script>
function prog(){{let t=document.querySelectorAll('.card').length,
 d=[...document.querySelectorAll('.card')].filter(c=>c.querySelector('input:checked')).length;
 document.getElementById('prog').textContent=` ${{d}}/${{t}} labeled`;}}
document.addEventListener('change',prog); prog();
function next(){{let u=[...document.querySelectorAll('.card')].find(c=>!c.querySelector('input:checked'));
 if(u)u.scrollIntoView({{behavior:'smooth',block:'center'}}); else alert('all labeled');}}
function save(){{let out=[];document.querySelectorAll('.card').forEach(c=>{{
 let s=c.querySelector('input:checked');
 out.push({{error_key:c.dataset.key,family:c.dataset.family,judge:c.dataset.judge,
  human:s?s.value:null}});}});
 let b=new Blob([JSON.stringify(out,null,1)],{{type:'application/json'}});
 let a=document.createElement('a');a.href=URL.createObjectURL(b);
 a.download='human_annotations.json';a.click();}}
</script></body></html>'''
    return html


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="main2b")
    ap.add_argument("--n-per-family", type=int, default=20)
    ap.add_argument("--prefill", default=None,
                    help="annotations json from a prior pass; matching cards are pre-checked")
    ap.add_argument("--pin", default=None,
                    help="annotations json; restrict the study to EXACTLY these error_keys "
                         "(same items as a prior annotator, for multi-rater agreement)")
    a = ap.parse_args()
    run_dir = os.path.join(HERE, "results", a.run)
    corpus = read_jsonl(os.path.join(run_dir, "corpus.jsonl"))
    if a.pin and os.path.exists(a.pin):
        import json as _json
        order = [r["error_key"] for r in _json.load(open(a.pin)) if r.get("error_key")]
        idx = image_index(run_dir)
        by_key = {r.get("error_key"): r for r in corpus}
        items = []
        for k in order:
            r = by_key.get(k)
            if not r:
                continue
            ref = idx.get((r.get("model"), r.get("dataset"), r.get("sample_id")), "")
            if not ref and r.get("model") and r.get("sample_id"):
                ref = f"{r['model']}-{r['sample_id']}.png"
            r["image_ref"] = ref
            items.append(r)
        print(f"pinned to {len(items)} items from {a.pin}")
    else:
        items = sample(corpus, a.n_per_family, run_dir)
    prefill = {}
    if a.prefill and os.path.exists(a.prefill):
        import json as _json
        prefill = {r["error_key"]: r["human"] for r in _json.load(open(a.prefill))
                   if r.get("human") and r.get("error_key")}
    # write the exact list of image files the sampled errors need, so we can pull
    # just those from rolf instead of the whole images/ dir.
    needed = sorted({os.path.basename(e["image_ref"]) for e in items if e.get("image_ref")})
    open(os.path.join(run_dir, "images_needed.txt"), "w").write("\n".join(needed) + "\n")
    html = build_html(items, run_dir, prefill=prefill)
    out = os.path.join(HERE, "docs", f"human_study_{a.run}.html")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    open(out, "w").write(html)
    n_have = sum(1 for e in items if _img_data_uri(os.path.join(
        run_dir, "images", os.path.basename(e.get("image_ref") or "")) if e.get("image_ref") else ""))
    print(f"wrote {out}\n  {len(items)} errors across families; "
          f"{len(needed)} reference an image, {n_have} embedded locally")
    if len(needed) > n_have:
        print(f"  {len(needed)-n_have} images missing -> pull them:")
        print(f"     scripts/pull_study_images.sh {a.run}")
    print("  -> open in a browser, label all, click 'Download annotations',")
    print(f"     then: python scripts/human_agreement.py --run {a.run} "
          f"--annotations human_annotations.json")


if __name__ == "__main__":
    main()
