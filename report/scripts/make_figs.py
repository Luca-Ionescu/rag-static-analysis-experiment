#!/usr/bin/env python
"""Generate all report figures into /tmp/report_work/figures/ (PDF).
Run with the analysis venv (matplotlib + package deps), PYTHONPATH=src."""
import csv, json, glob, sys, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, "/Users/nistoralex/personal_projects/rag-static-analysis-experiment/src")
from adaptive_retrieval.metrics import (edit_similarity, exact_match,
    truncate_to_function_body, truncate_to_line_count)

REPO="/Users/nistoralex/personal_projects/rag-static-analysis-experiment"
RES=f"{REPO}/data/_resweep"
OUT="/tmp/report_work/figures"; os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size":9,"font.family":"serif","axes.grid":True,
    "grid.alpha":0.3,"figure.dpi":150,"savefig.bbox":"tight","axes.axisbelow":True})

MODELS=[("qwen25_0.5b","0.5B","#1b9e77"),("qwen25_1.5b","1.5B","#d95f02"),("codellama_7b","7B","#7570b3")]
# dataset key -> (resweep suffix, pretty, main_mode)
DSETS=[("crosscodeeval_py","CrossCodeEval (line)","line"),
       ("repoeval_function","RepoEval (function)","body"),
       ("crosscodelongeval_function","CrossCodeLongEval (function)","body"),
       ("crosscodelongeval_chunk","CrossCodeLongEval (chunk)","lines")]

def load_sweep(tag,dskey):
    return list(csv.DictReader(open(f"{RES}/{tag}_{dskey}/sweep.csv")))
def F(x):
    try:return float(x)
    except:return None
def curve(rows,mode,cfg):
    pts=[(F(r["t_rag"]),F(r["retrieval_pct"]),F(r["edit_similarity"]),F(r["hall_A4B2"]),F(r["latency_ms"]))
         for r in rows if r["scoring"]==mode and r["config"]==cfg and r["t_rag"] not in ("",None)]
    return sorted(pts)
def endpoint(rows,mode,cfg):
    r=next(x for x in rows if x["scoring"]==mode and x["config"]==cfg)
    return F(r["retrieval_pct"]),F(r["edit_similarity"]),F(r["hall_A4B2"]),F(r["latency_ms"])

_FIM=("<|","▁<","<fim","<PRE>","<SUF>","<MID>","<EOT>","</s>","<｜","<repo_name>","<file_sep>","<|endoftext|>")
def _strip(t):
    for m in _FIM:
        i=t.find(m)
        if i!=-1: t=t[:i]
    return t
def clean(raw,gold,mode):
    s=_strip(raw)
    if mode=="line": return s.split("\n",1)[0]
    if mode=="lines": return truncate_to_line_count(gold,s)
    if mode=="body": return truncate_to_function_body(gold,s)
    return s

# ---------------- FIG 1: dataset GT-length distribution ----------------
def fig_datasets():
    raw={}
    cce=glob.glob(f"{REPO}/data/crosscodeeval/**/line_completion_rg1_bm25.jsonl",recursive=True)[0]
    raw["CrossCodeEval\n(line)"]=[json.loads(l)["groundtruth"].count("\n")+1 for l in open(cce)]
    re_=glob.glob(f"{REPO}/data/repoeval/datasets/function_level_completion_2k_context_codex.test.jsonl")[0]
    raw["RepoEval\n(function)"]=[json.loads(l)["metadata"]["ground_truth"].count("\n")+1 for l in open(re_)]
    for task,lab in (("function","CrossCodeLongEval\n(function)"),("chunk","CrossCodeLongEval\n(chunk)")):
        p=glob.glob(f"{REPO}/data/crosscodelongeval/cceval_{task}_eval_data/python_{task}_completion_sparse_rg1.jsonl")[0]
        raw[lab]=[json.loads(l)["groundtruth"].count("\n")+1 for l in open(p)]
    fig,ax=plt.subplots(figsize=(6.6,3.1))
    labels=list(raw); data=[raw[k] for k in labels]
    bp=ax.boxplot(data,vert=True,showfliers=False,patch_artist=True,widths=0.6,whis=(5,95))
    cols=["#4C72B0","#55A868","#C44E52","#8172B3"]
    for patch,c in zip(bp["boxes"],cols): patch.set_facecolor(c); patch.set_alpha(0.65)
    for med in bp["medians"]: med.set_color("black")
    ax.set_yscale("log"); ax.set_ylabel("ground-truth length (lines, log)")
    ax.set_ylim(0.8, 60)
    ax.set_xticklabels(labels,fontsize=8)
    for i,d in enumerate(data,1):
        ax.text(i,42,f"N={len(d)}",ha="center",fontsize=7.5,color="#333")
    ax.set_title("Completion-target length by benchmark (whiskers: 5–95th pct)",pad=10)
    fig.savefig(f"{OUT}/fig_datasets.pdf"); plt.close(fig)
    print("fig_datasets done")

# ---------------- FIG 2: cost-accuracy (ES vs %retrieval), cascade frontier ----------------
def fig_cost_accuracy():
    fig,axes=plt.subplots(1,4,figsize=(13,3.0))
    for ax,(dskey,pretty,mode) in zip(axes,DSETS):
        for tag,mlab,col in MODELS:
            rows=load_sweep(tag,dskey)
            c4=curve(rows,mode,"C4_cascade")
            xs=[p[1] for p in c4]; ys=[p[2] for p in c4]
            ax.plot(xs,ys,"-",color=col,lw=1.6,label=mlab)
            r1=endpoint(rows,mode,"C1_no_retrieve"); r2=endpoint(rows,mode,"C2_always_retrieve")
            ax.scatter([r1[0]],[r1[1]],color=col,marker="o",s=22,zorder=5)
            ax.scatter([r2[0]],[r2[1]],color=col,marker="s",s=22,zorder=5)
        ax.set_title(pretty,fontsize=8.5); ax.set_xlabel("retrieval rate (%)")
    axes[0].set_ylabel("edit similarity")
    axes[0].legend(title="model",fontsize=7.5,loc="lower right")
    # marker legend
    from matplotlib.lines import Line2D
    hh=[Line2D([0],[0],marker="o",color="gray",ls="",label="C1 no-retrieve"),
        Line2D([0],[0],marker="s",color="gray",ls="",label="C2 always-retrieve"),
        Line2D([0],[0],color="gray",lw=1.6,label="C4 cascade (sweep)")]
    axes[-1].legend(handles=hh,fontsize=7,loc="lower right")
    fig.suptitle("Accuracy–cost trade-off: the cascade reaches near-always-retrieve ES at a fraction of the retrieval budget",fontsize=9.5,y=1.02)
    fig.savefig(f"{OUT}/fig_cost_accuracy.pdf"); plt.close(fig)
    print("fig_cost_accuracy done")

# ---------------- FIG 3: hallucination vs %retrieval, CARD vs cascade ----------------
def fig_hallucination():
    fig,axes=plt.subplots(1,4,figsize=(13,3.0))
    for ax,(dskey,pretty,mode) in zip(axes,DSETS):
        for tag,mlab,col in MODELS:
            rows=load_sweep(tag,dskey)
            c3=curve(rows,mode,"C3_card"); c4=curve(rows,mode,"C4_cascade")
            ax.plot([p[1] for p in c3],[p[3] for p in c3],"--",color=col,lw=1.2,alpha=0.9)
            ax.plot([p[1] for p in c4],[p[3] for p in c4],"-",color=col,lw=1.7,label=mlab)
        ax.set_title(pretty,fontsize=8.5); ax.set_xlabel("retrieval rate (%)")
    axes[0].set_ylabel("invented-identifier rate\n(hall. $A_4\\wedge B_2$)")
    from matplotlib.lines import Line2D
    h1=[Line2D([0],[0],color=c,lw=1.7,label=m) for _,m,c in MODELS]
    h2=[Line2D([0],[0],color="gray",ls="--",lw=1.2,label="CARD (C3)"),
        Line2D([0],[0],color="gray",ls="-",lw=1.7,label="cascade (C4)")]
    axes[0].legend(handles=h1,fontsize=7.5,loc="upper right",title="model")
    axes[1].legend(handles=h2,fontsize=7.5,loc="upper right")
    fig.suptitle("The static gate (C4, solid) suppresses invented identifiers that CARD's confidence gate (C3, dashed) leaves through, at the same retrieval budget",fontsize=9,y=1.02)
    fig.savefig(f"{OUT}/fig_hallucination.pdf"); plt.close(fig)
    print("fig_hallucination done")

# ---------------- FIG 4: latency-accuracy dual axis (Repoformer Fig4 style) ----------------
def fig_latency():
    panels=[("crosscodelongeval_function","CrossCodeLongEval (function)","body"),
            ("crosscodelongeval_chunk","CrossCodeLongEval (chunk)","lines")]
    fig,axes=plt.subplots(1,2,figsize=(9.6,3.1))
    fig.subplots_adjust(wspace=0.55)
    tag,mlab,col="codellama_7b","7B","#7570b3"
    for ax,(dskey,pretty,mode) in zip(axes,panels):
        rows=load_sweep(tag,dskey); c4=curve(rows,mode,"C4_cascade")
        ts=[p[0] for p in c4]; es=[p[2] for p in c4]; lat=[p[4] for p in c4]
        l1,=ax.plot(ts,es,"-o",color="#1f77b4",ms=2.5,lw=1.4,label="edit similarity")
        ax.set_xlabel("CARD threshold $T_{RAG}$"); ax.set_ylabel("edit similarity",color="#1f77b4")
        ax.tick_params(axis="y",labelcolor="#1f77b4")
        ax2=ax.twinx()
        l2,=ax2.plot(ts,lat,"-s",color="#d62728",ms=2.5,lw=1.4,label="latency")
        ax2.set_ylabel("per-instance latency (ms)",color="#d62728"); ax2.tick_params(axis="y",labelcolor="#d62728")
        ax2.grid(False); ax.set_title(f"{pretty} · {mlab}",fontsize=8.5)
    fig.suptitle("Accuracy and latency vs the CARD threshold (cascade): higher $T_{RAG}$ $\\rightarrow$ more retrieval $\\rightarrow$ higher ES and higher latency",fontsize=9,y=1.04)
    fig.savefig(f"{OUT}/fig_latency_tradeoff.pdf"); plt.close(fig)
    print("fig_latency done")

# ---------------- FIG 5: per-instance Delta-ES histogram (retrieval helps few) ----------------
def per_instance_es(tag,dskey,mode):
    d=f"{RES}/{tag}_{dskey}"
    c1={json.loads(l)["instance_id"]:json.loads(l) for l in open(f"{d}/C1_no_retrieve.jsonl")}
    c2={json.loads(l)["instance_id"]:json.loads(l) for l in open(f"{d}/C2_always_retrieve.jsonl")}
    out=[]
    for i in set(c1)&set(c2):
        g=c1[i]["ground_truth"]
        e1=edit_similarity(g,clean(c1[i]["prediction"],g,mode))
        e2=edit_similarity(g,clean(c2[i]["prediction"],g,mode))
        out.append((e1,e2))
    return out
def fig_delta_es():
    tag,mlab="qwen25_1.5b","1.5B"
    fig,axes=plt.subplots(1,4,figsize=(13,2.8),sharey=False)
    for ax,(dskey,pretty,mode) in zip(axes,DSETS):
        pe=per_instance_es(tag,dskey,mode)
        d=[(e2-e1)*100 for e1,e2 in pe]
        ax.hist(d,bins=np.arange(-100,101,10),color="#d95f02",alpha=0.8,edgecolor="white",lw=0.4)
        helped=sum(1 for x in d if x>1); hurt=sum(1 for x in d if x<-1); same=len(d)-helped-hurt
        ax.axvline(0,color="black",lw=0.8)
        ax.set_title(f"{pretty}\nhelps {100*helped/len(d):.0f}% · hurts {100*hurt/len(d):.0f}% · ~same {100*same/len(d):.0f}%",fontsize=7.6)
        ax.set_xlabel("$\\Delta$ES (C2 $-$ C1), per inst.")
    axes[0].set_ylabel("# instances")
    fig.suptitle(f"Per-instance effect of always-retrieve ({mlab}): retrieval is broadly beneficial here, but a non-trivial slice is unchanged or hurt — the budget should be spent selectively",fontsize=8.8,y=1.05)
    fig.savefig(f"{OUT}/fig_delta_es_hist.pdf"); plt.close(fig)
    print("fig_delta_es done")

# ---------------- FIG 6: headline cascade hallucination reduction at low budget ----------------
def fig_cascade_bars():
    T=0.05
    fig,axes=plt.subplots(1,3,figsize=(12,3.0),sharey=False)
    dlabels=["CCE\n(line)","RepoEval\n(func)","CCLE\n(func)","CCLE\n(chunk)"]
    for ax,(tag,mlab,col) in zip(axes,MODELS):
        h3=[];h4=[];r4=[]
        for dskey,pretty,mode in DSETS:
            rows=load_sweep(tag,dskey)
            c3=next(p for p in curve(rows,mode,"C3_card") if abs(p[0]-T)<1e-6)
            c4=next(p for p in curve(rows,mode,"C4_cascade") if abs(p[0]-T)<1e-6)
            h3.append(c3[3]); h4.append(c4[3]); r4.append(c4[1])
        x=np.arange(len(dlabels)); w=0.38
        ax.bar(x-w/2,h3,w,label="CARD (C3)",color="#bdbdbd",edgecolor="black",lw=0.4)
        ax.bar(x+w/2,h4,w,label="cascade (C4)",color=col,edgecolor="black",lw=0.4)
        for i,(a,b,r) in enumerate(zip(h3,h4,r4)):
            fac = a/b if b>1e-9 else float('inf')
            txt = f"{fac:.0f}$\\times$" if fac!=float('inf') else "$\\rightarrow$0"
            ax.text(i+w/2,b+max(h3)*0.02,txt,ha="center",fontsize=7,color=col)
        ax.set_xticks(x); ax.set_xticklabels(dlabels,fontsize=7.5)
        ax.set_title(f"{mlab}",fontsize=9); ax.legend(fontsize=7,loc="upper right")
    axes[0].set_ylabel("invented-identifier rate\n(hall. $A_4\\wedge B_2$)")
    fig.suptitle(f"At a low retrieval budget ($T_{{RAG}}={T}$), the static gate cuts invented-identifier hallucination several-fold over CARD (factor annotated), at a few % extra retrieval",fontsize=9,y=1.03)
    fig.savefig(f"{OUT}/fig_cascade_bars.pdf"); plt.close(fig)
    print("fig_cascade_bars done")

if __name__=="__main__":
    fig_datasets(); fig_cost_accuracy(); fig_hallucination(); fig_latency(); fig_delta_es(); fig_cascade_bars()
    print("ALL FIGS DONE")
