"""Create additional SVG diagnostics from recorded repository results.

Uses only Python's standard library. Pass --history PATH to plot real epoch
records written by car_tse.train.utils.write_metrics.
"""
from __future__ import annotations
import argparse
import csv
import html
import json
import textwrap
from pathlib import Path

OUT = Path(__file__).resolve().parent
COLORS = ["#386cb0", "#f0027f", "#7fc97f", "#fdc086", "#beaed4"]


def esc(s): return html.escape(str(s))


def svg_start(title, width=1000, height=620):
    return [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
            '<rect width="100%" height="100%" fill="#fff"/>',
            '<style>text{font-family:Arial,sans-serif;fill:#253247}.title{font-size:23px;font-weight:bold}.label{font-size:14px}.small{font-size:12px}.grid{stroke:#d9e0e8;stroke-width:1}.axis{stroke:#536273;stroke-width:1.2}</style>',
            f'<text class="title" x="{width/2}" y="38" text-anchor="middle">{esc(title)}</text>']


def save_svg(name, parts):
    (OUT / f"{name}.svg").write_text("\n".join(parts + ["</svg>"]), encoding="utf-8")


def write_csv(name, rows):
    with (OUT / f"{name}.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)


def bar_chart(name, title, labels, values, xlabel, low=0, high=None, colors=None, footnote=""):
    high = high if high is not None else max(values) * 1.12
    foot_lines = textwrap.wrap(footnote, width=95) if footnote else []
    width, height, left, right, top, bottom = 1100, max(420, 95*len(labels)+140)+max(0,len(foot_lines)-1)*18, 420, 70, 78, 75
    plotw, ploth = width-left-right, height-top-bottom
    p = svg_start(title, width, height)
    for tick in range(6):
        v = low+(high-low)*tick/5; x=left+plotw*tick/5
        p += [f'<line class="grid" x1="{x}" y1="{top}" x2="{x}" y2="{top+ploth}"/>', f'<text class="small" x="{x}" y="{top+ploth+23}" text-anchor="middle">{v:.1f}</text>']
    rowh=ploth/len(labels)
    for i,(label,v) in enumerate(zip(labels,values)):
        y=top+i*rowh+rowh*.2; bh=rowh*.58
        x0=left+plotw*(0-low)/(high-low) if low<0<high else left
        xv=left+plotw*(v-low)/(high-low)
        p += [f'<text class="label" x="{left-12}" y="{y+bh*.72}" text-anchor="end">{esc(label)}</text>',
              f'<rect x="{min(x0,xv)}" y="{y}" width="{max(1,abs(xv-x0))}" height="{bh}" fill="{(colors or COLORS)[i%len(colors or COLORS)]}" rx="4"/>',
              f'<text class="small" x="{xv+8}" y="{y+bh*.72}">{v:.2f}</text>']
    p += [f'<line class="axis" x1="{left}" y1="{top+ploth}" x2="{left+plotw}" y2="{top+ploth}"/>', f'<text class="label" x="{left+plotw/2}" y="{height-48}" text-anchor="middle">{esc(xlabel)}</text>']
    for i,line in enumerate(foot_lines): p += [f'<text class="small" x="{left}" y="{height-12-(len(foot_lines)-1-i)*17}">{esc(line)}</text>']
    save_svg(name,p)


def line_chart(name, title, xs, series, xlabel, ylabel, y_min=0, y_max=None, footnote=""):
    vals=[v for points in series.values() for v in points if v is not None]
    y_max=y_max or max(vals)*1.1
    W,H,L,R,T,B=1050,610,100,80,85,130
    pw,ph=W-L-R,H-T-B
    p=svg_start(title,W,H)
    for j in range(6):
        val=y_min+(y_max-y_min)*j/5; y=T+ph*(1-j/5)
        p += [f'<line class="grid" x1="{L}" y1="{y}" x2="{L+pw}" y2="{y}"/>', f'<text class="small" x="{L-12}" y="{y+4}" text-anchor="end">{val:.1f}</text>']
    for i,xv in enumerate(xs):
        x=L+pw*i/max(1,len(xs)-1)
        p += [f'<text class="small" x="{x}" y="{T+ph+23}" text-anchor="middle">{esc(xv)}</text>']
    for si,(label,ys) in enumerate(series.items()):
        coords=[]
        for i,v in enumerate(ys):
            if v is None: continue
            x=L+pw*i/max(1,len(xs)-1); y=T+ph*(1-(v-y_min)/(y_max-y_min)); coords.append((x,y))
            p += [f'<circle cx="{x}" cy="{y}" r="6" fill="{COLORS[si%len(COLORS)]}"/>']
        if len(coords)>1: p += [f'<polyline points="{" ".join(f"{x},{y}" for x,y in coords)}" fill="none" stroke="{COLORS[si%len(COLORS)]}" stroke-width="3"/>']
        lx=L+si*240; ly=H-58
        p += [f'<line x1="{lx}" y1="{ly}" x2="{lx+28}" y2="{ly}" stroke="{COLORS[si%len(COLORS)]}" stroke-width="3"/>', f'<text class="small" x="{lx+35}" y="{ly+4}">{esc(label)}</text>']
    p += [f'<text class="label" x="{L+pw/2}" y="{T+ph+52}" text-anchor="middle">{esc(xlabel)}</text>',f'<text class="label" transform="translate(25 {T+ph/2}) rotate(-90)" text-anchor="middle">{esc(ylabel)}</text>']
    if footnote: p += [f'<text class="small" x="{L}" y="{H-14}">{esc(footnote)}</text>']
    save_svg(name,p)


def diagnostics():
    rows=[
      {"Evaluation":"CAVEMOVE synthetic test: mixture","SI-SDR (dB)":-2.41},
      {"Evaluation":"CAVEMOVE synthetic test: extractor","SI-SDR (dB)":8.57},
      {"Evaluation":"AISHELL-5 real target: far-field","SI-SDR (dB)":-37.31},
      {"Evaluation":"AISHELL-5 real target: current extractor","SI-SDR (dB)":-31.22},
      {"Evaluation":"AISHELL-5 real target: adapted frontend","SI-SDR (dB)":-30.57},]
    write_csv("stage1_domain_transfer",rows)
    bar_chart("fig11_stage1_domain_transfer","Stage 1: mean SI-SDR (dB), synthetic vs real cabin",[r["Evaluation"] for r in rows],[r["SI-SDR (dB)"] for r in rows],"",-40,15,footnote="Different datasets/protocols: compare within each evaluation pair only. Adaptation improved AISHELL by 0.65 dB but CAVEMOVE validation fell 11.00 to 9.90 dB.")

    rows=[{"Threshold":.50,"Recall (%)":100.,"False accept rate (%)":96.25},
          {"Threshold":.72,"Recall (%)":99.,"False accept rate (%)":36.25},
          {"Threshold":.90,"Recall (%)":91.13,"False accept rate (%)":17.5}]
    write_csv("stage2_validation_operating_points",rows)
    p=svg_start("Stage 2: presence-gate validation operating points",900,650); L,T,pw,ph=100,90,720,430
    for i in range(6):
        v=i*20; y=T+ph*(1-v/100); x=L+pw*v/100
        p += [f'<line class="grid" x1="{L}" y1="{y}" x2="{L+pw}" y2="{y}"/>',f'<text class="small" x="{L-10}" y="{y+4}" text-anchor="end">{v}</text>',f'<line class="grid" x1="{x}" y1="{T}" x2="{x}" y2="{T+ph}"/>',f'<text class="small" x="{x}" y="{T+ph+22}" text-anchor="middle">{v}</text>']
    for i,r in enumerate(rows):
        x=L+pw*r["False accept rate (%)"]/100; y=T+ph*(1-r["Recall (%)"]/100)
        if i == 0:
            label_x,anchor=x-12,"end"
        else:
            label_x,anchor=x+12,"start"
        p += [f'<circle cx="{x}" cy="{y}" r="10" fill="{COLORS[i]}"/>',f'<text class="small" x="{label_x}" y="{y-10}" text-anchor="{anchor}">threshold {r["Threshold"]:.2f}</text>']
    p += ['<text class="label" x="460" y="590" text-anchor="middle">False accept rate on target-absent validation (%)</text>','<text class="label" transform="translate(30 300) rotate(-90)" text-anchor="middle">Target-present recall (%)</text>','<text class="small" x="100" y="635">Validation: 800 target-present / 80 target-absent examples; points summarize one threshold sweep.</text>']
    save_svg("fig12_stage2_operating_points",p)

    rows=[{"Candidate":"Prior hybrid","Threshold":.81,"Smoothed false events/hour":9},
          {"Candidate":"AISHELL-augmented","Threshold":.81,"Smoothed false events/hour":30},
          {"Candidate":"AISHELL-augmented","Threshold":.90,"Smoothed false events/hour":12},
          {"Candidate":"AISHELL-augmented","Threshold":.95,"Smoothed false events/hour":3}]
    write_csv("stage3_aishell_threshold_stream",rows)
    line_chart("fig13_stage3_stream_threshold","Stage 3: threshold behavior on AISHELL-5 speech stream",[.81,.90,.95],{"Prior hybrid KWS":[9,None,None],"AISHELL-augmented KWS":[30,12,3]},"KWS score threshold","Smoothed false events per hour",0,35,"Target-absent stream only (20 min); this measures false events, not wake recall.")

    snrs=[-5,0,10,20]; one=[6.3,18.8,53.1,25.0]; two=[15.6,31.3,46.9,21.9]
    far1=[70.3,112.5,140.6,42.2]; far2=[None,None,112.5,51.6]
    rows=[]
    for i,snr in enumerate(snrs):
        rows += [{"SNR (dB)":snr,"Enrollment":"One clip","Smoothed wake recall (%)":one[i],"False events/hour":far1[i]},
                 {"SNR (dB)":snr,"Enrollment":"Two clips","Smoothed wake recall (%)":two[i],"False events/hour":far2[i] if far2[i] is not None else "not reported"}]
    write_csv("stage3_multispeaker_proxy",rows)
    line_chart("fig14a_stage3_proxy_recall","Stage 3: eight-speaker proxy wake recall",snrs,{"One-clip enrollment":one,"Two-clip enrollment":two},"SNR (dB)","Smoothed wake recall (%)",0,100,"32 expected wakes per SNR in synthetic cabin-noise scenes; not a real-cabin benchmark.")
    line_chart("fig14b_stage3_proxy_false_events","Stage 3: eight-speaker proxy false events",snrs,{"One-clip enrollment":far1,"Two-clip enrollment":far2},"SNR (dB)","Smoothed false events/hour",0,160,"Two-clip rates at -5 and 0 dB were not provided in the report and are intentionally omitted.")

    rows=[("CRNN bootstrap clean",41.53),("CRNN CAVEMOVE",41.10),("CRNN TSE-domain",48.62),
          ("wav2vec bootstrap clean",92.09),("wav2vec bootstrap raw cabin",88.98),("wav2vec bootstrap TSE-domain",89.41),
          ("wav2vec fine-tuned local clean",93.04),("wav2vec fine-tuned raw cabin",90.71),("wav2vec fine-tuned TSE-domain",90.49),
          ("wav2vec fine-tuned FSC validation",96.60),("wav2vec fine-tuned FSC test",99.31)]
    write_csv("stage4_domain_accuracy",[{"Model / domain":a,"Top-1 accuracy (%)":b} for a,b in rows])
    bar_chart("fig15_stage4_domain_accuracy","Stage 4: intent accuracy across models and domains",[a for a,b in rows],[b for a,b in rows],"Top-1 accuracy (%)",0,105,footnote="Official FSC clean test/validation and local noisy-domain validation are separate evaluations.")


def train_curves(history: Path, stem: str) -> bool:
    if not history.exists(): return False
    rows=[json.loads(line) for line in history.read_text(encoding="utf-8").splitlines() if line.strip()]
    flat=[]
    for row in rows:
        item={k:v for k,v in row.items() if isinstance(v,(int,float))}
        for k,v in row.get("val",{}).items():
            if isinstance(v,(int,float)): item[f"val_{k}"]=v
        item["epoch"]=row.get("epoch"); flat.append(item)
    metrics=[k for k in flat[0] if k!="epoch" and sum(isinstance(x.get(k),(int,float)) for x in flat)>1]
    if not metrics:return False
    W,H,L,R,T,B=1000,180+len(metrics)*130,110,40,90,50
    p=svg_start(f"Training history: {history.parent.name}",W,H)
    for mi,key in enumerate(metrics):
        vals=[r.get(key) for r in flat]; good=[v for v in vals if isinstance(v,(int,float))]
        lo,hi=min(good),max(good); hi=hi if hi>lo else lo+1
        y0=T+mi*130; pw=W-L-R; ph=75
        p += [f'<text class="label" x="{L}" y="{y0-10}">{esc(key.replace("_"," "))}</text>']
        pts=[]
        for i,v in enumerate(vals):
            if not isinstance(v,(int,float)): continue
            x=L+pw*i/max(1,len(vals)-1); y=y0+ph*(1-(v-lo)/(hi-lo)); pts.append(f"{x},{y}")
        p += [f'<polyline points="{" ".join(pts)}" fill="none" stroke="{COLORS[mi%len(COLORS)]}" stroke-width="3"/>',f'<text class="small" x="{L}" y="{y0+ph+18}">epoch range: {flat[0]["epoch"]} to {flat[-1]["epoch"]}; value range: {lo:.4g} to {max(good):.4g}</text>']
    save_svg(stem,p); return True


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--history",type=Path); ap.add_argument("--name",default="training_curves"); args=ap.parse_args()
    diagnostics()
    if args.history and not train_curves(args.history,args.name): raise SystemExit(f"No plot-ready epoch metrics in {args.history}")
    print(f"Wrote SVG diagnostics to {OUT}; training curves require an existing metrics_history.jsonl.")

if __name__=="__main__": main()
