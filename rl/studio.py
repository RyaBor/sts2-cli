"""Live training dashboard.

    python rl/studio.py            # then open http://localhost:8778

Reads the TensorBoard event files while training is still writing them and
serves an auto-refreshing comparison of every run, plus checkpoint metadata.
Unlike dashboard.py (which renders a static file), this polls, so you can watch
runs progress and compare them against each other as they go.
"""
from __future__ import annotations

import argparse
import json
import os
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "runs")
CKPTS = os.path.join(HERE, "checkpoints")

# Charted in this order; everything else is still available in the table.
PANELS = [
    ("combat/win_rate", "Win rate", "%"),
    ("combat/hp_retained_on_win", "HP retained on win", "hp"),
    ("rollout/ep_rew_mean", "Episode reward", ""),
    ("rollout/ep_len_mean", "Episode length", "steps"),
    ("combat/timeout_rate", "Timeout rate", "%"),
    ("combat/invalid_actions", "Invalid actions", ""),
    ("train/explained_variance", "Value fit", ""),
    ("train/entropy_loss", "Entropy loss", ""),
    ("time/fps", "Throughput", "fps"),
]
HEADLINES = ["combat/win_rate", "combat/hp_retained_on_win",
             "rollout/ep_rew_mean", "time/fps"]

_cache: dict[str, tuple[float, dict]] = {}
_lock = threading.Lock()
LIVE_AFTER = 90.0  # a run whose events changed this recently counts as running


def _load_run(path: str) -> dict:
    """Reload a run's scalars, keyed on the event file's mtime."""
    files = [os.path.join(path, f) for f in os.listdir(path) if f.startswith("events.out")]
    if not files:
        return {}
    mtime = max(os.path.getmtime(f) for f in files)
    name = os.path.basename(path)
    with _lock:
        hit = _cache.get(name)
        if hit and hit[0] == mtime:
            return hit[1]

    ea = EventAccumulator(path, size_guidance={"scalars": 100000})
    ea.Reload()
    series = {}
    for tag in ea.Tags().get("scalars", []):
        pts = ea.Scalars(tag)
        series[tag] = {"x": [p.step for p in pts], "y": [round(p.value, 6) for p in pts]}
    data = {
        "name": name,
        "series": series,
        "updated": mtime,
        "live": (time.time() - mtime) < LIVE_AFTER,
        "steps": max((s["x"][-1] for s in series.values() if s["x"]), default=0),
    }
    with _lock:
        _cache[name] = (mtime, data)
    return data


def _meta_for(run_name: str) -> dict:
    """Config written by train.py. Run dirs are '<name>_<n>', metadata is
    keyed by the bare name, so strip the TensorBoard suffix."""
    base = run_name.rsplit("_", 1)[0] if run_name.rsplit("_", 1)[-1].isdigit() else run_name
    path = os.path.join(RUNS, "meta", f"{base}.json")
    if not os.path.exists(path):
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def collect_runs() -> list[dict]:
    if not os.path.isdir(RUNS):
        return []
    out = []
    for entry in sorted(os.listdir(RUNS)):
        p = os.path.join(RUNS, entry)
        if not os.path.isdir(p) or entry == "meta":
            continue
        d = _load_run(p)
        if not d:
            continue
        d["meta"] = _meta_for(entry)

        target = d["meta"].get("target_steps")
        d["target"] = target
        d["progress"] = min(1.0, d["steps"] / target) if target else None
        # ETA from recent throughput rather than the run average, so a run that
        # sped up or slowed down reports what it is doing now.
        fps = d["series"].get("time/fps", {}).get("y") or []
        recent = fps[-8:]
        rate = sum(recent) / len(recent) if recent else 0
        d["fps_now"] = rate
        d["eta"] = ((target - d["steps"]) / rate) if (target and rate > 0
                                                      and d["steps"] < target) else None
        out.append(d)
    out.sort(key=lambda r: (not r["live"], -r["updated"]))
    return out


def collect_checkpoints() -> list[dict]:
    if not os.path.isdir(CKPTS):
        return []
    rows = []
    for f in sorted(os.listdir(CKPTS)):
        if not f.endswith(".zip"):
            continue
        full = os.path.join(CKPTS, f)
        rows.append({
            "name": f[:-4],
            "mb": round(os.path.getsize(full) / 1e6, 1),
            "modified": os.path.getmtime(full),
        })
    rows.sort(key=lambda r: -r["modified"])
    return rows


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # keep the console quiet
        pass

    def _send(self, body: bytes, ctype: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.startswith("/api/data"):
            payload = {
                "runs": collect_runs(),
                "checkpoints": collect_checkpoints(),
                "panels": [{"tag": t, "title": ti, "unit": u} for t, ti, u in PANELS],
                "headlines": HEADLINES,
                "now": time.time(),
            }
            self._send(json.dumps(payload).encode(), "application/json")
        else:
            self._send(PAGE.encode(), "text/html; charset=utf-8")


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>STS2 RL — live training</title>
<style>
  :root{color-scheme:light;--surface-1:#fcfcfb;--plane:#f9f9f7;--text-primary:#0b0b0b;
    --text-secondary:#52514e;--muted:#898781;--grid:#e1e0d9;--axis:#c3c2b7;
    --border:rgba(11,11,11,.10);--good:#0ca30c;
    --s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;--s4:#eda100;--s5:#e87ba4;--s6:#008300;}
  @media (prefers-color-scheme:dark){:root:where(:not([data-theme="light"])){
    color-scheme:dark;--surface-1:#1a1a19;--plane:#0d0d0d;--text-primary:#fff;
    --text-secondary:#c3c2b7;--muted:#898781;--grid:#2c2c2a;--axis:#383835;
    --border:rgba(255,255,255,.10);
    --s1:#3987e5;--s2:#d95926;--s3:#199e70;--s4:#c98500;--s5:#d55181;--s6:#008300;}}
  :root[data-theme="dark"]{color-scheme:dark;--surface-1:#1a1a19;--plane:#0d0d0d;
    --text-primary:#fff;--text-secondary:#c3c2b7;--muted:#898781;--grid:#2c2c2a;
    --axis:#383835;--border:rgba(255,255,255,.10);
    --s1:#3987e5;--s2:#d95926;--s3:#199e70;--s4:#c98500;--s5:#d55181;--s6:#008300;}
  *{box-sizing:border-box}
  body{margin:0;padding:26px 22px 60px;background:var(--plane);color:var(--text-primary);
    font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
  .wrap{max-width:1240px;margin:0 auto}
  header{display:flex;align-items:baseline;gap:14px;flex-wrap:wrap;margin-bottom:6px}
  h1{font-size:21px;margin:0;letter-spacing:-.01em}
  .sub{color:var(--text-secondary);font-size:13.5px;margin:0 0 22px}
  .dot{width:8px;height:8px;border-radius:50%;display:inline-block;margin-right:6px}
  .live .dot{background:var(--good);animation:pulse 1.6s infinite}
  .idle .dot{background:var(--muted)}
  @keyframes pulse{0%,100%{opacity:1}50%{opacity:.35}}
  .runs{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:18px}
  .chip{display:flex;align-items:center;gap:7px;background:var(--surface-1);
    border:1px solid var(--border);border-radius:20px;padding:5px 12px;font-size:13px;
    cursor:pointer;user-select:none}
  .chip.off{opacity:.4}
  .sw{width:10px;height:10px;border-radius:3px}
  .tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:11px;margin-bottom:22px}
  .tile,.card{background:var(--surface-1);border:1px solid var(--border);border-radius:10px;padding:14px 16px}
  .tile .lbl{color:var(--text-secondary);font-size:12.5px;margin-bottom:5px}
  .tile .val{font-size:27px;font-weight:600;letter-spacing:-.02em}
  .tile .run{font-size:12px;color:var(--muted);margin-top:4px}
  .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(400px,1fr));gap:12px}
  .card h2{font-size:13.5px;font-weight:600;margin:0 0 1px}
  .card .meta{font-size:12px;color:var(--muted);margin-bottom:8px}
  svg{display:block;width:100%;height:auto;overflow:visible}
  table{border-collapse:collapse;width:100%;font-size:12.5px;font-variant-numeric:tabular-nums}
  th,td{text-align:right;padding:5px 8px;border-bottom:1px solid var(--grid)}
  th:first-child,td:first-child{text-align:left}
  th{color:var(--text-secondary);font-weight:600}
  .tip{position:fixed;pointer-events:none;opacity:0;transition:opacity .1s;background:var(--surface-1);
    border:1px solid var(--border);border-radius:7px;padding:7px 10px;font-size:12.5px;
    box-shadow:0 4px 14px rgba(0,0,0,.14);z-index:9;color:var(--text-primary)}
  .tip .k{color:var(--text-secondary)}
  h3{font-size:14px;margin:26px 0 8px}
  .prog{display:grid;grid-template-columns:1fr;gap:9px;margin-bottom:20px}
  .prow{background:var(--surface-1);border:1px solid var(--border);border-radius:10px;padding:12px 15px}
  .ptop{display:flex;align-items:center;gap:9px;font-size:13.5px;margin-bottom:8px;flex-wrap:wrap}
  .ptop .nm{font-weight:600}
  .ptop .cfg{color:var(--muted);font-size:12px}
  .ptop .right{margin-left:auto;color:var(--text-secondary);font-size:12.5px;font-variant-numeric:tabular-nums}
  .bar{height:7px;border-radius:4px;background:var(--grid);overflow:hidden}
  .bar i{display:block;height:100%;border-radius:4px;transition:width .6s ease}
  .bar.indet i{width:35%;animation:slide 1.7s ease-in-out infinite}
  @keyframes slide{0%{margin-left:-35%}100%{margin-left:100%}}
</style></head><body>
<div class="wrap">
  <header><h1>Slay the Spire 2 — live training</h1><span id="clock" class="sub"></span></header>
  <p class="sub" id="sub">connecting…</p>
  <div class="runs" id="runs"></div>
  <div class="prog" id="prog"></div>
  <div class="tiles" id="tiles"></div>
  <div class="grid" id="charts"></div>
  <h3>Runs</h3><div class="card"><table id="runtable"></table></div>
  <h3>Checkpoints</h3><div class="card"><table id="ckpt"></table></div>
</div>
<div class="tip" id="tip"></div>
<script>
const COLORS=['var(--s1)','var(--s2)','var(--s3)','var(--s4)','var(--s5)','var(--s6)'];
const tip=document.getElementById('tip');
let hidden=new Set(), DATA=null, colorOf={};

const fmt=(v,u)=>{ if(v==null||Number.isNaN(v))return '—';
  if(u==='%')return (v*100).toFixed(1)+'%';
  if(Math.abs(v)>=1000)return Math.round(v).toLocaleString();
  if(Math.abs(v)>=10)return v.toFixed(1); return v.toFixed(3); };
const short=n=>n>=1e6?(n/1e6).toFixed(2)+'M':n>=1e3?Math.round(n/1e3)+'k':String(n);
const ago=s=>{const d=Math.max(0,Math.round(s)); return d<60?d+'s':d<3600?Math.round(d/60)+'m':Math.round(d/3600)+'h';};
const last=s=>s&&s.y.length?s.y[s.y.length-1]:null;

function thin(xs,ys,max){ if(xs.length<=max)return [xs,ys];
  const st=xs.length/max,ox=[],oy=[];
  for(let i=0;i<max;i++){const j=Math.min(xs.length-1,Math.round(i*st));ox.push(xs[j]);oy.push(ys[j]);}
  ox.push(xs[xs.length-1]);oy.push(ys[ys.length-1]);return [ox,oy]; }

function visible(){ return DATA.runs.filter(r=>!hidden.has(r.name)); }

function drawChips(){
  const el=document.getElementById('runs'); el.innerHTML='';
  DATA.runs.forEach((r,i)=>{
    colorOf[r.name]=COLORS[i%COLORS.length];
    const d=document.createElement('div');
    d.className='chip '+(r.live?'live':'idle')+(hidden.has(r.name)?' off':'');
    d.innerHTML=`<span class="dot"></span><span class="sw" style="background:${colorOf[r.name]}"></span>`+
      `${r.name} <span style="color:var(--muted)">${short(r.steps)}</span>`;
    d.onclick=()=>{hidden.has(r.name)?hidden.delete(r.name):hidden.add(r.name);render();};
    el.appendChild(d);
  });
}

function drawProgress(){
  const el=document.getElementById('prog'); el.innerHTML='';
  for(const r of visible()){
    const c=colorOf[r.name], m=r.meta||{};
    const pct=r.progress==null?null:r.progress*100;
    const cfg=[m.envs?m.envs+' envs':null,
               m.n_snapshots?m.n_snapshots+' snapshots':null,
               m.deck_noise?Math.round(m.deck_noise*100)+'% deck noise':null,
               m.encounter&&m.encounter!=='default'
                 ? (m.encounter.split(',').length+' encounters') : null].filter(Boolean).join(' · ');
    const right = r.live
      ? (pct!=null?`${pct.toFixed(1)}% · ${short(r.steps)}/${short(r.target)}`:short(r.steps))
        + (r.eta?` · ~${ago(r.eta)} left`:'') + (r.fps_now?` · ${Math.round(r.fps_now)} fps`:'')
      : (pct!=null&&pct>=99.5?'complete':`stopped at ${short(r.steps)}`
         +(pct!=null?` (${pct.toFixed(0)}%)`:''));
    const d=document.createElement('div'); d.className='prow';
    d.innerHTML=`<div class="ptop"><span class="sw" style="background:${c}"></span>`+
      `<span class="nm">${r.name}</span><span class="cfg">${cfg}</span>`+
      `<span class="right">${right}</span></div>`+
      // No target recorded (runs started before metadata existed): show motion,
      // not a fake percentage.
      (pct==null && r.live
        ? `<div class="bar indet"><i style="background:${c}"></i></div>`
        : `<div class="bar"><i style="width:${Math.max(1,pct||0)}%;background:${c};${r.live?'':'opacity:.5'}"></i></div>`);
    el.appendChild(d);
  }
}

function drawTiles(){
  const el=document.getElementById('tiles'); el.innerHTML='';
  const runs=visible();
  for(const tag of DATA.headlines){
    const unit=(DATA.panels.find(p=>p.tag===tag)||{}).unit||'';
    // Best current value across visible runs, and who holds it.
    let best=null;
    for(const r of runs){ const v=last(r.series[tag]); if(v==null)continue;
      const better = tag==='rollout/ep_len_mean' ? (best&&v<best.v) : (!best||v>best.v);
      if(!best||better)best={v,run:r.name}; }
    if(!best)continue;
    const t=document.createElement('div'); t.className='tile';
    const title=(DATA.panels.find(p=>p.tag===tag)||{}).title||tag;
    t.innerHTML=`<div class="lbl">${title}</div><div class="val">${fmt(best.v,unit)}</div>`+
      `<div class="run">best: ${best.run}</div>`;
    el.appendChild(t);
  }
}

function chart(panel){
  const runs=visible().filter(r=>r.series[panel.tag]&&r.series[panel.tag].y.length);
  if(!runs.length)return null;
  const card=document.createElement('div'); card.className='card';
  card.innerHTML=`<h2>${panel.title}</h2><div class="meta">${panel.unit||'value'} vs timesteps</div>`;
  const W=560,H=200,P={t:10,r:14,b:26,l:48};
  let xmin=Infinity,xmax=-Infinity,ymin=Infinity,ymax=-Infinity;
  for(const r of runs){const s=r.series[panel.tag];
    xmin=Math.min(xmin,...s.x);xmax=Math.max(xmax,...s.x);
    ymin=Math.min(ymin,...s.y);ymax=Math.max(ymax,...s.y);}
  if(ymin===ymax){ymin-=1;ymax+=1;}
  const pad=(ymax-ymin)*.1; ymin-=pad; ymax+=pad;
  if(panel.unit==='%'){ymin=Math.max(-0.02,ymin);ymax=Math.min(1.02,ymax);}
  const sx=v=>P.l+(v-xmin)/((xmax-xmin)||1)*(W-P.l-P.r);
  const sy=v=>H-P.b-(v-ymin)/((ymax-ymin)||1)*(H-P.t-P.b);
  let g='';
  for(let i=0;i<=4;i++){const v=ymin+(ymax-ymin)*i/4,y=sy(v);
    g+=`<line x1="${P.l}" x2="${W-P.r}" y1="${y}" y2="${y}" stroke="var(--grid)" stroke-width="1"/>`;
    g+=`<text x="${P.l-7}" y="${y+4}" text-anchor="end" font-size="10.5" fill="var(--muted)">${fmt(v,panel.unit)}</text>`;}
  for(let i=0;i<=4;i++){const v=xmin+(xmax-xmin)*i/4;
    g+=`<text x="${sx(v)}" y="${H-P.b+16}" text-anchor="middle" font-size="10.5" fill="var(--muted)">${short(v)}</text>`;}
  g+=`<line x1="${P.l}" x2="${W-P.r}" y1="${H-P.b}" y2="${H-P.b}" stroke="var(--axis)" stroke-width="1"/>`;
  const drawn=[];
  runs.forEach(r=>{const s=r.series[panel.tag];const [xs,ys]=thin(s.x,s.y,200);
    drawn.push({name:r.name,xs,ys});
    const c=colorOf[r.name];
    g+=`<path d="${xs.map((x,j)=>`${j?'L':'M'}${sx(x).toFixed(1)},${sy(ys[j]).toFixed(1)}`).join('')}" fill="none" stroke="${c}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>`;
    g+=`<circle cx="${sx(xs[xs.length-1])}" cy="${sy(ys[ys.length-1])}" r="3.4" fill="${c}" stroke="var(--surface-1)" stroke-width="2"/>`;});
  card.insertAdjacentHTML('beforeend',
    `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${panel.title}">${g}
      <rect class="hit" x="${P.l}" y="${P.t}" width="${W-P.l-P.r}" height="${H-P.t-P.b}" fill="transparent"/>
      <line class="cross" x1="0" x2="0" y1="${P.t}" y2="${H-P.b}" stroke="var(--axis)" stroke-width="1" opacity="0"/></svg>`);
  const svg=card.querySelector('svg'),cross=svg.querySelector('.cross'),hit=svg.querySelector('.hit');
  hit.addEventListener('pointermove',ev=>{
    const box=svg.getBoundingClientRect(); const px=(ev.clientX-box.left)/box.width*W;
    const xv=xmin+(px-P.l)/((W-P.l-P.r)||1)*(xmax-xmin);
    cross.setAttribute('x1',px);cross.setAttribute('x2',px);cross.setAttribute('opacity','1');
    tip.innerHTML=`<div class="k">step ${short(Math.round(xv))}</div>`+drawn.map(s=>{
      let k=0,b=Infinity; for(let j=0;j<s.xs.length;j++){const d=Math.abs(s.xs[j]-xv);if(d<b){b=d;k=j;}}
      return `<div><span class="sw" style="display:inline-block;background:${colorOf[s.name]}"></span> ${s.name} <strong>${fmt(s.ys[k],panel.unit)}</strong></div>`;}).join('');
    tip.style.opacity=1;tip.style.left=Math.min(innerWidth-210,ev.clientX+14)+'px';tip.style.top=(ev.clientY-10)+'px';});
  hit.addEventListener('pointerleave',()=>{tip.style.opacity=0;cross.setAttribute('opacity','0');});
  return card;
}

function drawTables(){
  const runs=visible();
  const tags=DATA.panels.map(p=>p.tag);
  let head='<thead><tr><th>run</th><th>steps</th><th>state</th>'+
    DATA.panels.map(p=>`<th>${p.title}</th>`).join('')+'</tr></thead><tbody>';
  for(const r of runs){
    head+=`<tr><td>${r.name}</td><td>${short(r.steps)}</td>`+
      `<td>${r.live?'running':'idle '+ago(DATA.now-r.updated)}</td>`+
      tags.map((t,i)=>`<td>${fmt(last(r.series[t]),DATA.panels[i].unit)}</td>`).join('')+'</tr>';
  }
  document.getElementById('runtable').innerHTML=head+'</tbody>';
  let c='<thead><tr><th>checkpoint</th><th>size</th><th>age</th></tr></thead><tbody>';
  for(const k of DATA.checkpoints.slice(0,25))
    c+=`<tr><td>${k.name}</td><td>${k.mb} MB</td><td>${ago(DATA.now-k.modified)}</td></tr>`;
  document.getElementById('ckpt').innerHTML=c+'</tbody>';
}

function render(){
  drawChips(); drawProgress(); drawTiles();
  const el=document.getElementById('charts'); el.innerHTML='';
  for(const p of DATA.panels){const c=chart(p); if(c)el.appendChild(c);}
  drawTables();
  const live=DATA.runs.filter(r=>r.live).length;
  document.getElementById('sub').textContent =
    `${DATA.runs.length} run(s), ${live} training now — click a run to show/hide it`;
  document.getElementById('clock').textContent='updated '+new Date().toLocaleTimeString();
}

let first=true;
async function poll(){
  try{
    const r=await fetch('/api/data',{cache:'no-store'}); DATA=await r.json();
    if(first){
      first=false;
      // Focus on what is training now. Old runs are usually on a different
      // encounter set, so overlaying them silently compares unlike things —
      // they stay one click away rather than on by default.
      if(DATA.runs.some(x=>x.live)) DATA.runs.forEach(x=>{ if(!x.live) hidden.add(x.name); });
    }
    render();
  }
  catch(e){ document.getElementById('sub').textContent='lost connection to studio.py'; }
}
poll(); setInterval(poll,5000);
</script></body></html>
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8778)
    ap.add_argument("--no-open", action="store_true")
    args = ap.parse_args()

    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    url = f"http://localhost:{args.port}"
    print(f"live training dashboard: {url}   (ctrl-c to stop)")
    if not args.no_open:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
