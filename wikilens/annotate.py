"""`python -m wikilens annotate RUN_ID`: write a single-file HTML page for
labelling alignment candidates. The page needs no server: open it in a
browser, label, and export the labels to eval/labels/.
"""
import json
from datetime import datetime, timezone
from pathlib import Path

from wikilens.alignment_eval import LABELS_FORMAT, build_candidates, default_query_lang, labels_filename, run_languages

PAGES_DIR = Path("eval/annotation_pages")
LABELS_DIR = Path("eval/labels")


def _embed_json(data) -> str:
    """JSON that is safe inside <script type="application/json">."""
    return (json.dumps(data, ensure_ascii=False)
            .replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026"))


def render_page(data: dict) -> str:
    return PAGE_TEMPLATE.replace("__DATA__", _embed_json(data)).replace(
        "__TITLE__", f"Align labels · {data['title'].replace('_', ' ')} {data['query_lang']}→{data['target_lang']}"
        .replace("&", "&amp;").replace("<", "&lt;"))


def cmd_annotate(args) -> int:
    from wikilens.cli import find_run
    meta = find_run(args.run)
    langs = run_languages(meta)
    query_lang = args.from_lang or default_query_lang(langs)
    data = build_candidates(meta["_dir"], meta, query_lang, k=args.k, sample=args.sample, seed=args.seed)
    data["format"] = LABELS_FORMAT
    data["labels_file"] = labels_filename(meta["run_id"], query_lang, langs)
    data["generated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")

    out = Path(args.output) if args.output else PAGES_DIR / f"{Path(data['labels_file']).stem}.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_page(data), encoding="utf-8")
    LABELS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Run       {meta['run_id']}  ({data['pair']})")
    print(f"Queries   {len(data['items'])} {query_lang} chunks"
          + (f" (sample of {data['n_query_chunks']}, seed {args.seed})" if args.sample else "")
          + f", top {args.k} {data['target_lang']} candidates each")
    print(f"Page      {out.resolve()}")
    print(f"Labels    export from the page, then save as {LABELS_DIR / data['labels_file']}")
    print("Guide     docs/annotation_guidelines.md")
    return 0


PAGE_TEMPLATE = r"""<!doctype html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
:root {
  color-scheme: light;
  --bg: #f7f7f5; --card: #ffffff; --ink: #1b1b1a; --muted: #5f5e5a; --line: #deddd8;
  --accent: #2a78d6; --match: #1d7f4e; --partial: #a86a00; --none: #b2403f;
  --match-bg: #e6f3ec; --partial-bg: #fbf1dc; --none-bg: #f9e7e6; --done: #e9f1fb;
}
@media (prefers-color-scheme: dark) {
  :root {
    color-scheme: dark;
    --bg: #161615; --card: #1f1f1e; --ink: #eeede9; --muted: #a9a8a1; --line: #3a3936;
    --accent: #5a9ff0; --match: #58c08a; --partial: #e2a83a; --none: #ec7b79;
    --match-bg: #19352a; --partial-bg: #3a2f17; --none-bg: #3d2120; --done: #1c2a3b;
  }
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--ink);
  font: 15px/1.55 system-ui, -apple-system, "PingFang SC", "Microsoft YaHei", sans-serif; }
header { position: sticky; top: 0; z-index: 2; background: var(--card); border-bottom: 1px solid var(--line);
  padding: 10px 16px; display: flex; flex-wrap: wrap; gap: 8px 16px; align-items: center; }
header h1 { font-size: 16px; margin: 0; font-weight: 600; }
.meta { color: var(--muted); font-size: 13px; }
.progress { flex: 1 1 160px; min-width: 140px; height: 8px; background: var(--line); border-radius: 4px; overflow: hidden; }
.progress > div { height: 100%; background: var(--accent); width: 0; transition: width .2s; }
.controls { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }
button, .filebtn { font: inherit; font-size: 13px; padding: 5px 10px; border-radius: 6px; cursor: pointer;
  border: 1px solid var(--line); background: var(--card); color: var(--ink); }
button.primary { background: var(--accent); border-color: var(--accent); color: #fff; }
input[type=text] { font: inherit; font-size: 13px; padding: 4px 8px; border-radius: 6px;
  border: 1px solid var(--line); background: var(--card); color: var(--ink); width: 9em; }
main { max-width: 1100px; margin: 0 auto; padding: 16px; }
details.ref { background: var(--card); border: 1px solid var(--line); border-radius: 8px; padding: 8px 12px; margin-bottom: 16px; }
details.ref summary { cursor: pointer; color: var(--muted); }
details.ref .chunk { border-top: 1px solid var(--line); padding: 6px 0; font-size: 14px; }
.item { scroll-margin-top: 120px; background: var(--card); border: 1px solid var(--line); border-radius: 10px; margin-bottom: 18px; overflow: hidden; }
.item.done { border-color: var(--accent); }
.item.done .qhead { background: var(--done); }
.qhead { padding: 10px 14px; border-bottom: 1px solid var(--line); }
.tag { display: inline-block; font-size: 12px; color: var(--muted); margin-right: 10px; }
.tag b { color: var(--ink); font-weight: 600; }
.text { white-space: pre-wrap; overflow-wrap: anywhere; }
.qtext { margin-top: 4px; }
.cand { display: grid; grid-template-columns: 1fr auto; gap: 8px 14px; padding: 10px 14px; border-bottom: 1px solid var(--line); }
.cand.l-match { background: var(--match-bg); } .cand.l-partial { background: var(--partial-bg); } .cand.l-none { background: var(--none-bg); }
.cand.disabled { opacity: .5; }
.choices { display: flex; flex-direction: column; gap: 4px; min-width: 8.5em; }
.choices label { font-size: 14px; cursor: pointer; white-space: nowrap; }
.choices label.match { color: var(--match); } .choices label.partial { color: var(--partial); } .choices label.none { color: var(--none); }
.foot { padding: 10px 14px; display: flex; flex-wrap: wrap; gap: 8px 18px; align-items: center; }
.foot label { cursor: pointer; }
.foot input[type=text] { flex: 1 1 200px; width: auto; }
.foot .idsrow { flex: 1 1 100%; display: flex; flex-wrap: wrap; gap: 6px 12px; align-items: center; }
.foot .idsrow input { flex: 0 1 22em; }
.warn { font-size: 13px; color: var(--none); }
.hidden { display: none; }
.status { font-size: 13px; color: var(--muted); }
@media (max-width: 640px) {
  header { position: static; }
  .item { scroll-margin-top: 8px; }
  .cand { grid-template-columns: 1fr; }
  .choices { flex-direction: row; flex-wrap: wrap; gap: 4px 14px; }
}
</style>
</head>
<body>
<header>
  <h1 id="title"></h1>
  <span class="meta" id="meta"></span>
  <div class="progress" title="完成度"><div id="bar"></div></div>
  <span class="meta" id="count"></span>
  <div class="controls">
    <input type="text" id="annotator" placeholder="标注者 annotator">
    <label class="meta"><input type="checkbox" id="onlyTodo"> 只看未完成</label>
    <button id="jump">下一个未完成 ↓</button>
    <label class="filebtn">导入 JSON<input type="file" id="importFile" accept=".json,application/json" class="hidden"></label>
    <button class="primary" id="export">导出 JSON</button>
  </div>
  <span class="status" id="status"></span>
</header>
<main>
  <details class="ref">
    <summary id="refSummary"></summary>
    <div id="ref"></div>
  </details>
  <div id="items"></div>
</main>
<script id="data" type="application/json">__DATA__</script>
<script>
(function () {
  "use strict";
  var DATA = JSON.parse(document.getElementById("data").textContent);
  var LABELS = [["match", "对应"], ["partial", "部分对应"], ["none", "不对应"]];
  var KEY = "wikilens-align-labels:" + DATA.run_id + ":" + DATA.query_lang + "-" + DATA.target_lang;
  var state = {};  // query_id -> {absent, elsewhere, elsewhere_ids, note, labels: {chunk_id: label}}
  var TARGET_IDS = {};
  DATA.target_chunks.forEach(function (c) { TARGET_IDS[c.chunk_id] = true; });
  var annotator = "";

  function el(tag, attrs, children) {
    var e = document.createElement(tag);
    if (attrs) for (var k in attrs) {
      if (k === "text") e.textContent = attrs[k];
      else if (k === "class") e.className = attrs[k];
      else e.setAttribute(k, attrs[k]);
    }
    (children || []).forEach(function (c) { if (c) e.appendChild(c); });
    return e;
  }
  function sectionText(o) { return o.section + (o.section_en ? "（" + o.section_en + "）" : ""); }
  function entry(qid) {
    if (!state[qid]) state[qid] = { absent: false, elsewhere: false, elsewhere_ids: [], note: "", labels: {} };
    return state[qid];
  }
  function isDone(item) {
    var s = state[item.query_id];
    if (!s) return false;
    if (s.absent) return true;
    return item.candidates.every(function (c) { return !!s.labels[c.chunk_id]; });
  }

  function save() {
    try { localStorage.setItem(KEY, JSON.stringify({ state: state, annotator: annotator })); }
    catch (e) { setStatus("浏览器无法自动保存，请定期导出 JSON。"); }
  }
  function load() {
    try {
      var raw = localStorage.getItem(KEY);
      if (raw) { var d = JSON.parse(raw); state = d.state || {}; annotator = d.annotator || ""; }
    } catch (e) { /* storage unavailable: start empty */ }
  }
  function setStatus(msg) { document.getElementById("status").textContent = msg; }

  function render() {
    var root = document.getElementById("items");
    root.textContent = "";
    DATA.items.forEach(function (item, n) {
      var s = entry(item.query_id);
      var card = el("section", { class: "item", id: "q-" + item.query_id });
      card.appendChild(el("div", { class: "qhead" }, [
        el("span", { class: "tag", text: "#" + (n + 1) + " / " + DATA.items.length }),
        el("span", { class: "tag" }, [document.createTextNode(DATA.query_lang.toUpperCase() + " "), el("b", { text: item.query_id })]),
        el("span", { class: "tag", text: "章节：" + sectionText(item) }),
        el("div", { class: "text qtext", text: item.text })
      ]));
      item.candidates.forEach(function (c, r) {
        var name = "l-" + item.query_id + "-" + c.chunk_id;
        var choices = el("div", { class: "choices" });
        LABELS.forEach(function (pair) {
          var input = el("input", { type: "radio", name: name, value: pair[0] });
          input.checked = s.labels[c.chunk_id] === pair[0];
          input.disabled = s.absent;
          input.addEventListener("change", function () {
            s.labels[c.chunk_id] = pair[0];
            changed(item, card);
          });
          choices.appendChild(el("label", { class: pair[0] }, [input, document.createTextNode(" " + pair[1])]));
        });
        var row = el("div", { class: "cand" }, [
          el("div", null, [
            el("span", { class: "tag" }, [document.createTextNode("候选 " + (r + 1) + " · " + DATA.target_lang.toUpperCase() + " "), el("b", { text: c.chunk_id })]),
            el("span", { class: "tag" }, [document.createTextNode("相似度 "), el("b", { text: c.chunk_sim.toFixed(3) })]),
            el("span", { class: "tag", text: "章节：" + sectionText(c) }),
            el("div", { class: "text", text: c.text })
          ]),
          choices
        ]);
        row.dataset.chunk = c.chunk_id;
        card.appendChild(row);
      });
      var absent = el("input", { type: "checkbox" });
      absent.checked = s.absent;
      absent.addEventListener("change", function () {
        s.absent = absent.checked;
        if (s.absent) {
          s.elsewhere = false;
          item.candidates.forEach(function (c) { s.labels[c.chunk_id] = "none"; });
        }
        changed(item, card);
        render();
        document.getElementById("q-" + item.query_id).scrollIntoView({ block: "nearest" });
      });
      var elsewhere = el("input", { type: "checkbox" });
      elsewhere.checked = !!s.elsewhere;
      elsewhere.disabled = s.absent;
      var ids = el("input", { type: "text", class: "ids", placeholder: "对应段落 ID（可选，如 " +
        (DATA.target_chunks[0] ? DATA.target_chunks[0].chunk_id : "en_0042") + "）" });
      ids.value = (s.elsewhere_ids || []).join(", ");
      var idsWarn = el("span", { class: "warn" });
      var idsRow = el("div", { class: "idsrow" + (s.elsewhere ? "" : " hidden") }, [ids, idsWarn]);
      function readIds() {
        var parts = ids.value.split(/[\s,，;；]+/).filter(Boolean);
        var cands = {};
        item.candidates.forEach(function (c) { cands[c.chunk_id] = true; });
        var bad = parts.filter(function (x) { return !TARGET_IDS[x]; });
        var inCands = parts.filter(function (x) { return cands[x]; });
        s.elsewhere_ids = parts.filter(function (x) { return TARGET_IDS[x] && !cands[x]; });
        var msgs = [];
        if (bad.length) msgs.push("不存在的 ID：" + bad.join(", "));
        if (inCands.length) msgs.push(inCands.join(", ") + " 是上面的候选，请直接标注");
        idsWarn.textContent = msgs.length ? msgs.join("；") + "（不会导出）" : "";
      }
      readIds();
      ids.addEventListener("input", function () { readIds(); save(); });
      elsewhere.addEventListener("change", function () {
        s.elsewhere = elsewhere.checked;
        if (s.elsewhere) s.absent = false;
        changed(item, card);
        render();
        document.getElementById("q-" + item.query_id).scrollIntoView({ block: "nearest" });
      });
      var note = el("input", { type: "text", placeholder: "备注（可选）" });
      note.value = s.note || "";
      note.addEventListener("input", function () { s.note = note.value; save(); });
      card.appendChild(el("div", { class: "foot" }, [
        el("label", null, [elsewhere, document.createTextNode(" 有对应段落，但不在这 " + DATA.k + " 个候选里")]),
        el("label", null, [absent, document.createTextNode(" 另一种语言里完全没有（" + DATA.target_lang.toUpperCase() + " 全文都没有这段内容）")]),
        idsRow,
        note
      ]));
      root.appendChild(card);
      paint(item, card);
    });
    applyFilter();
    updateProgress();
  }

  function paint(item, card) {
    var s = entry(item.query_id);
    card.classList.toggle("done", isDone(item));
    Array.prototype.forEach.call(card.querySelectorAll(".cand"), function (row) {
      var l = s.labels[row.dataset.chunk];
      row.className = "cand" + (l ? " l-" + l : "") + (s.absent ? " disabled" : "");
    });
  }
  function changed(item, card) { paint(item, card); save(); updateProgress(); }

  function updateProgress() {
    var done = DATA.items.filter(isDone).length;
    document.getElementById("bar").style.width = (100 * done / Math.max(1, DATA.items.length)) + "%";
    document.getElementById("count").textContent = done + " / " + DATA.items.length + " 已完成";
  }
  function applyFilter() {
    var only = document.getElementById("onlyTodo").checked;
    DATA.items.forEach(function (item) {
      document.getElementById("q-" + item.query_id).classList.toggle("hidden", only && isDone(item));
    });
  }

  function exportLabels() {
    var out = {
      format: DATA.format, run_id: DATA.run_id, title: DATA.title, pair: DATA.pair,
      query_lang: DATA.query_lang, target_lang: DATA.target_lang, k: DATA.k, sample: DATA.sample,
      annotator: annotator, page_generated_at: DATA.generated_at, exported_at: new Date().toISOString(),
      items: DATA.items.map(function (item) {
        var s = entry(item.query_id);
        return {
          query_id: item.query_id, section: item.section, absent: !!s.absent,
          elsewhere: !!s.elsewhere, elsewhere_ids: s.elsewhere ? (s.elsewhere_ids || []) : [],
          note: s.note || "",
          candidates: item.candidates.map(function (c) {
            return { chunk_id: c.chunk_id, chunk_sim: c.chunk_sim, section: c.section, label: s.labels[c.chunk_id] || null };
          })
        };
      })
    };
    var blob = new Blob([JSON.stringify(out, null, 2) + "\n"], { type: "application/json" });
    var a = el("a", { href: URL.createObjectURL(blob), download: DATA.labels_file });
    document.body.appendChild(a); a.click(); a.remove();
    var done = DATA.items.filter(isDone).length;
    setStatus("已导出 " + DATA.labels_file + "（" + done + "/" + DATA.items.length + " 完成），请放到 eval/labels/");
  }

  function importLabels(file) {
    var reader = new FileReader();
    reader.onload = function () {
      try {
        var d = JSON.parse(reader.result);
        if (d.run_id !== DATA.run_id || d.query_lang !== DATA.query_lang) {
          setStatus("导入失败：文件属于 " + d.run_id + " (" + d.query_lang + ")，不是本页的运行。");
          return;
        }
        var n = 0;
        d.items.forEach(function (it) {
          var s = { absent: !!it.absent, elsewhere: !!it.elsewhere, elsewhere_ids: it.elsewhere_ids || [],
                    note: it.note || "", labels: {} };
          it.candidates.forEach(function (c) { if (c.label) s.labels[c.chunk_id] = c.label; });
          state[it.query_id] = s; n++;
        });
        if (d.annotator) { annotator = d.annotator; document.getElementById("annotator").value = annotator; }
        save(); render();
        setStatus("已导入 " + n + " 条。");
      } catch (e) { setStatus("导入失败：" + e.message); }
    };
    reader.readAsText(file);
  }

  // ── Setup ──
  load();
  document.getElementById("title").textContent = DATA.title.replace(/_/g, " ") + " · " +
    DATA.query_lang.toUpperCase() + " → " + DATA.target_lang.toUpperCase();
  document.getElementById("meta").textContent = DATA.run_id + " · 每段 top " + DATA.k + " 候选";
  document.getElementById("refSummary").textContent = "查看 " + DATA.target_lang.toUpperCase() +
    " 全文（" + DATA.target_chunks.length + " 段，可用 Ctrl/⌘-F 搜索，用于判断「完全没有」）";
  var ref = document.getElementById("ref");
  DATA.target_chunks.forEach(function (c) {
    ref.appendChild(el("div", { class: "chunk" }, [
      el("span", { class: "tag" }, [el("b", { text: c.chunk_id })]),
      el("span", { class: "tag", text: sectionText(c) }),
      el("div", { class: "text", text: c.text })
    ]));
  });
  var ann = document.getElementById("annotator");
  ann.value = annotator;
  ann.addEventListener("input", function () { annotator = ann.value; save(); });
  document.getElementById("onlyTodo").addEventListener("change", applyFilter);
  document.getElementById("export").addEventListener("click", exportLabels);
  document.getElementById("importFile").addEventListener("change", function (e) {
    if (e.target.files[0]) importLabels(e.target.files[0]);
    e.target.value = "";
  });
  document.getElementById("jump").addEventListener("click", function () {
    var next = DATA.items.filter(function (i) { return !isDone(i); })[0];
    if (next) document.getElementById("q-" + next.query_id).scrollIntoView({ behavior: "smooth", block: "start" });
    else setStatus("全部完成，记得导出 JSON。");
  });
  window.addEventListener("beforeunload", save);
  render();
})();
</script>
</body>
</html>
"""
