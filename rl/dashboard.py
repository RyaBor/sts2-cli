"""Build a self-contained HTML dashboard from TensorBoard run logs.

    python rl/dashboard.py                          # all runs in rl/runs
    python rl/dashboard.py --run combat_v1_1        # one run
    python rl/dashboard.py --out report.html --open

Produces a single file with no external requests, so it can be opened
directly or shared. For live monitoring during a run, use TensorBoard
instead: python -m tensorboard.main --logdir rl/runs
"""
from __future__ import annotations

import argparse
import json
import os
import webbrowser

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "runs")

# Charts to render, in order: (tag, title, unit, higher_is_better|None)
PANELS = [
    ("combat/win_rate", "Win rate", "%", True),
    ("combat/hp_retained_on_win", "HP retained on win", "hp", True),
    ("rollout/ep_rew_mean", "Episode reward", "", True),
    ("rollout/ep_len_mean", "Episode length", "steps", False),
    ("train/explained_variance", "Value fit (explained variance)", "", True),
    ("train/entropy_loss", "Entropy loss", "", None),
    ("train/approx_kl", "Policy update size (approx KL)", "", None),
    ("time/fps", "Throughput", "fps", True),
]

HEADLINES = [
    ("combat/win_rate", "Win rate", "%"),
    ("combat/hp_retained_on_win", "HP retained", "hp"),
    ("rollout/ep_rew_mean", "Episode reward", ""),
    ("rollout/ep_len_mean", "Steps to finish", ""),
]


def load_run(path: str) -> dict:
    ea = EventAccumulator(path)
    ea.Reload()
    tags = ea.Tags().get("scalars", [])
    series = {}
    for tag in tags:
        pts = ea.Scalars(tag)
        series[tag] = {"x": [p.step for p in pts], "y": [round(p.value, 6) for p in pts]}
    return {"name": os.path.basename(path), "series": series}


def discover(runs_dir: str, only: str | None) -> list[dict]:
    if not os.path.isdir(runs_dir):
        return []
    out = []
    for entry in sorted(os.listdir(runs_dir)):
        full = os.path.join(runs_dir, entry)
        if not os.path.isdir(full):
            continue
        if only and entry != only:
            continue
        if not any(f.startswith("events.out") for f in os.listdir(full)):
            continue
        out.append(load_run(full))
    return out


def build_html(runs: list[dict]) -> str:
    payload = json.dumps({
        "runs": runs,
        "panels": [{"tag": t, "title": ti, "unit": u, "up": up} for t, ti, u, up in PANELS],
        "headlines": [{"tag": t, "label": l, "unit": u} for t, l, u in HEADLINES],
    })
    return _TEMPLATE.replace("__DATA__", payload)


_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>STS2 RL — training metrics</title>
<style>
  :root {
    color-scheme: light;
    --surface-1: #fcfcfb;
    --plane: #f9f9f7;
    --text-primary: #0b0b0b;
    --text-secondary: #52514e;
    --muted: #898781;
    --grid: #e1e0d9;
    --axis: #c3c2b7;
    --border: rgba(11,11,11,0.10);
    --series-1: #2a78d6;
    --series-2: #eb6834;
  }
  @media (prefers-color-scheme: dark) {
    :root:where(:not([data-theme="light"])) {
      color-scheme: dark;
      --surface-1: #1a1a19;
      --plane: #0d0d0d;
      --text-primary: #ffffff;
      --text-secondary: #c3c2b7;
      --muted: #898781;
      --grid: #2c2c2a;
      --axis: #383835;
      --border: rgba(255,255,255,0.10);
      --series-1: #3987e5;
      --series-2: #d95926;
    }
  }
  :root[data-theme="dark"] {
    color-scheme: dark;
    --surface-1: #1a1a19; --plane: #0d0d0d;
    --text-primary: #fff; --text-secondary: #c3c2b7; --muted: #898781;
    --grid: #2c2c2a; --axis: #383835; --border: rgba(255,255,255,0.10);
    --series-1: #3987e5; --series-2: #d95926;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; padding: 32px 24px 64px;
    background: var(--plane); color: var(--text-primary);
    font: 15px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif;
  }
  .wrap { max-width: 1180px; margin: 0 auto; }
  h1 { font-size: 22px; margin: 0 0 4px; letter-spacing: -0.01em; }
  .sub { color: var(--text-secondary); margin: 0 0 28px; font-size: 14px; }
  .tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(190px,1fr)); gap: 12px; margin-bottom: 28px; }
  .tile, .card {
    background: var(--surface-1); border: 1px solid var(--border);
    border-radius: 10px; padding: 16px 18px;
  }
  .tile .label { color: var(--text-secondary); font-size: 13px; margin-bottom: 6px; }
  .tile .value { font-size: 30px; font-weight: 600; letter-spacing: -0.02em; }
  .tile .unit { font-size: 15px; color: var(--muted); font-weight: 400; margin-left: 3px; }
  .tile .delta { font-size: 12.5px; color: var(--text-secondary); margin-top: 5px; }
  .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(430px,1fr)); gap: 14px; }
  .card h2 { font-size: 14px; font-weight: 600; margin: 0 0 2px; }
  .card .meta { font-size: 12.5px; color: var(--muted); margin-bottom: 10px; }
  .legend { display: flex; gap: 14px; flex-wrap: wrap; margin: 0 0 8px; font-size: 12.5px; color: var(--text-secondary); }
  .legend span { display: inline-flex; align-items: center; gap: 6px; }
  .swatch { width: 10px; height: 10px; border-radius: 2px; display: inline-block; }
  svg { display: block; width: 100%; height: auto; overflow: visible; }
  .tip {
    position: fixed; pointer-events: none; opacity: 0; transition: opacity .1s;
    background: var(--surface-1); border: 1px solid var(--border); border-radius: 7px;
    padding: 7px 10px; font-size: 12.5px; box-shadow: 0 4px 14px rgba(0,0,0,.13); z-index: 9;
    color: var(--text-primary);
  }
  .tip .k { color: var(--text-secondary); }
  details { margin-top: 12px; }
  summary { cursor: pointer; font-size: 13px; color: var(--text-secondary); }
  table { border-collapse: collapse; width: 100%; margin-top: 10px; font-size: 12.5px; font-variant-numeric: tabular-nums; }
  th, td { text-align: right; padding: 5px 8px; border-bottom: 1px solid var(--grid); }
  th:first-child, td:first-child { text-align: left; }
  th { color: var(--text-secondary); font-weight: 600; }
  .empty { color: var(--muted); font-size: 13px; padding: 18px 0; }
  .tablewrap { overflow-x: auto; max-height: 320px; overflow-y: auto; }
</style>
</head>
<body>
<div class="wrap">
  <h1>Slay the Spire 2 — RL training metrics</h1>
  <p class="sub" id="sub"></p>
  <div class="tiles" id="tiles"></div>
  <div class="grid" id="charts"></div>
</div>
<div class="tip" id="tip"></div>
<script>
const DATA = __DATA__;
const tip = document.getElementById('tip');
const COLORS = ['var(--series-1)','var(--series-2)'];

const fmt = (v, unit) => {
  if (v === null || v === undefined || Number.isNaN(v)) return '—';
  if (unit === '%') return (v * 100).toFixed(1) + '%';
  if (Math.abs(v) >= 1000) return Math.round(v).toLocaleString();
  if (Math.abs(v) >= 10) return v.toFixed(1);
  return v.toFixed(3);
};
const short = n => n >= 1e6 ? (n/1e6).toFixed(1)+'M' : n >= 1e3 ? Math.round(n/1e3)+'k' : String(n);

// Downsample for drawing; keeps first/last so endpoints are exact.
function thin(xs, ys, max) {
  if (xs.length <= max) return [xs, ys];
  const step = xs.length / max, ox = [], oy = [];
  for (let i = 0; i < max; i++) { const j = Math.min(xs.length-1, Math.round(i*step)); ox.push(xs[j]); oy.push(ys[j]); }
  ox.push(xs[xs.length-1]); oy.push(ys[ys.length-1]);
  return [ox, oy];
}

document.getElementById('sub').textContent =
  DATA.runs.length ? DATA.runs.map(r => r.name).join(' · ') : 'No runs found in rl/runs';

// ---- headline tiles: last value of each metric, with change from the start ----
// Take each metric from the most recent run that actually recorded it — older
// runs predate the combat/* metrics, so keying every tile off one run drops them.
const tiles = document.getElementById('tiles');
for (const h of DATA.headlines) {
  let src = null;
  for (let i = DATA.runs.length-1; i >= 0; i--) {
    const s = DATA.runs[i].series[h.tag];
    if (s && s.y.length) { src = {run: DATA.runs[i], s}; break; }
  }
  if (!src) continue;
  const y = src.s.y, last = y[y.length-1], d = last - y[0];
  const el = document.createElement('div');
  el.className = 'tile';
  el.innerHTML = `<div class="label">${h.label}</div>
    <div class="value">${fmt(last, h.unit)}${h.unit && h.unit!=='%' ? `<span class="unit">${h.unit}</span>`:''}</div>
    <div class="delta">${d>=0?'+':''}${fmt(d,h.unit)} since start${
      DATA.runs.length>1 ? ' · '+src.run.name : ''}</div>`;
  tiles.appendChild(el);
}
if (!tiles.children.length) {
  tiles.innerHTML = '<div class="empty">No scalar metrics found. Run training first.</div>';
}

// ---- line chart ----
function chart(panel) {
  const present = DATA.runs.filter(r => r.series[panel.tag] && r.series[panel.tag].y.length);
  if (!present.length) return null;

  const card = document.createElement('div');
  card.className = 'card';
  const multi = present.length > 1;
  card.innerHTML = `<h2>${panel.title}</h2><div class="meta">${panel.unit || 'value'} vs timesteps</div>`;

  if (multi) {
    const lg = document.createElement('div');
    lg.className = 'legend';
    lg.innerHTML = present.map((r,i)=>
      `<span><i class="swatch" style="background:${COLORS[i%2]}"></i>${r.name}</span>`).join('');
    card.appendChild(lg);
  }

  const W = 560, H = 210, P = {t: 12, r: 16, b: 30, l: 46};
  let xmin=Infinity, xmax=-Infinity, ymin=Infinity, ymax=-Infinity;
  for (const r of present) {
    const s = r.series[panel.tag];
    xmin=Math.min(xmin,...s.x); xmax=Math.max(xmax,...s.x);
    ymin=Math.min(ymin,...s.y); ymax=Math.max(ymax,...s.y);
  }
  if (ymin === ymax) { ymin -= 1; ymax += 1; }
  const pad = (ymax-ymin)*0.10; ymin -= pad; ymax += pad;
  if (panel.unit === '%') { ymin = Math.max(0, ymin); ymax = Math.min(1.02, ymax); }
  const sx = v => P.l + (v-xmin)/((xmax-xmin)||1) * (W-P.l-P.r);
  const sy = v => H-P.b - (v-ymin)/((ymax-ymin)||1) * (H-P.t-P.b);

  const ticks = 4;
  let g = '';
  for (let i=0;i<=ticks;i++) {
    const v = ymin + (ymax-ymin)*i/ticks, y = sy(v);
    g += `<line x1="${P.l}" x2="${W-P.r}" y1="${y}" y2="${y}" stroke="var(--grid)" stroke-width="1"/>`;
    g += `<text x="${P.l-8}" y="${y+4}" text-anchor="end" font-size="11" fill="var(--muted)">${fmt(v,panel.unit)}</text>`;
  }
  for (let i=0;i<=4;i++) {
    const v = xmin + (xmax-xmin)*i/4;
    g += `<text x="${sx(v)}" y="${H-P.b+18}" text-anchor="middle" font-size="11" fill="var(--muted)">${short(v)}</text>`;
  }
  g += `<line x1="${P.l}" x2="${W-P.r}" y1="${H-P.b}" y2="${H-P.b}" stroke="var(--axis)" stroke-width="1"/>`;

  const drawn = [];
  present.forEach((r,i) => {
    const s = r.series[panel.tag];
    const [xs, ys] = thin(s.x, s.y, 220);
    drawn.push({name:r.name, xs, ys});
    const d = xs.map((x,j)=>`${j?'L':'M'}${sx(x).toFixed(1)},${sy(ys[j]).toFixed(1)}`).join('');
    g += `<path d="${d}" fill="none" stroke="${COLORS[i%2]}" stroke-width="2"
           stroke-linejoin="round" stroke-linecap="round"/>`;
    // endpoint dot + direct label, so a single series needs no legend
    const lx = sx(xs[xs.length-1]), ly = sy(ys[ys.length-1]);
    g += `<circle cx="${lx}" cy="${ly}" r="3.5" fill="${COLORS[i%2]}" stroke="var(--surface-1)" stroke-width="2"/>`;
  });

  const svg = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${panel.title}">
    ${g}<rect id="hit" x="${P.l}" y="${P.t}" width="${W-P.l-P.r}" height="${H-P.t-P.b}" fill="transparent"/>
    <line class="cross" x1="0" x2="0" y1="${P.t}" y2="${H-P.b}" stroke="var(--axis)" stroke-width="1" opacity="0"/>
  </svg>`;
  card.insertAdjacentHTML('beforeend', svg);

  const el = card.querySelector('svg');
  const cross = el.querySelector('.cross');
  const hit = el.querySelector('#hit');
  hit.addEventListener('pointermove', ev => {
    const box = el.getBoundingClientRect();
    const px = (ev.clientX - box.left) / box.width * W;
    const xv = xmin + (px - P.l)/((W-P.l-P.r)||1) * (xmax-xmin);
    cross.setAttribute('x1', px); cross.setAttribute('x2', px); cross.setAttribute('opacity','1');
    const rows = drawn.map((s,i) => {
      let k = 0, best = Infinity;
      for (let j=0;j<s.xs.length;j++){ const dd=Math.abs(s.xs[j]-xv); if(dd<best){best=dd;k=j;} }
      return `<div><i class="swatch" style="background:${COLORS[i%2]}"></i>
        <span class="k">${multi ? s.name : 'step ' + short(s.xs[k])}</span>
        <strong>${fmt(s.ys[k], panel.unit)}</strong></div>`;
    }).join('');
    tip.innerHTML = rows;
    tip.style.opacity = 1;
    tip.style.left = Math.min(window.innerWidth-180, ev.clientX+14) + 'px';
    tip.style.top = (ev.clientY-10) + 'px';
  });
  hit.addEventListener('pointerleave', () => { tip.style.opacity=0; cross.setAttribute('opacity','0'); });

  // table view — every value reachable without hover
  const rowsN = Math.min(60, present[0].series[panel.tag].x.length);
  const [tx] = thin(present[0].series[panel.tag].x, present[0].series[panel.tag].y, rowsN);
  let body = '';
  for (let i=0;i<tx.length;i++) {
    const cells = present.map(r => {
      const s = r.series[panel.tag];
      let k=0,best=Infinity;
      for(let j=0;j<s.x.length;j++){const d=Math.abs(s.x[j]-tx[i]); if(d<best){best=d;k=j;}}
      return `<td>${fmt(s.y[k], panel.unit)}</td>`;
    }).join('');
    body += `<tr><td>${short(tx[i])}</td>${cells}</tr>`;
  }
  card.insertAdjacentHTML('beforeend',
    `<details><summary>Table view</summary><div class="tablewrap"><table>
      <thead><tr><th>step</th>${present.map(r=>`<th>${r.name}</th>`).join('')}</tr></thead>
      <tbody>${body}</tbody></table></div></details>`);
  return card;
}

const charts = document.getElementById('charts');
let any = false;
for (const p of DATA.panels) { const c = chart(p); if (c) { charts.appendChild(c); any = true; } }
if (!any) charts.innerHTML = '<div class="empty">No chartable metrics yet.</div>';
</script>
</body>
</html>
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default=RUNS)
    ap.add_argument("--run", default=None, help="only this run directory")
    ap.add_argument("--out", default=os.path.join(HERE, "dashboard.html"))
    ap.add_argument("--open", action="store_true", help="open in a browser when done")
    args = ap.parse_args()

    runs = discover(args.runs_dir, args.run)
    if not runs:
        print(f"No runs with event files under {args.runs_dir}")
    html = build_html(runs)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"wrote {args.out}  ({len(runs)} run(s), {sum(len(r['series']) for r in runs)} series)")
    if args.open:
        webbrowser.open("file://" + os.path.abspath(args.out))


if __name__ == "__main__":
    main()
