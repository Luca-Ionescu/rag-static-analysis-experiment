#!/usr/bin/env python
"""Emit LaTeX table fragments + key-number macros from the sweeps."""
import csv, sys, os
RES="/Users/nistoralex/personal_projects/rag-static-analysis-experiment/data/_resweep"
OUT="/tmp/report_work/tables"; os.makedirs(OUT,exist_ok=True)
MODELS=[("qwen25_0.5b","0.5B"),("qwen25_1.5b","1.5B"),("codellama_7b","7B")]
DS=[("crosscodeeval_py","CCE (line)","line"),
    ("repoeval_function","RepoEval (func)","body"),
    ("crosscodelongeval_function","CCLE (func)","body"),
    ("crosscodelongeval_chunk","CCLE (chunk)","lines")]
def F(x):
    try:return float(x)
    except:return None
def load(tag,ds): return list(csv.DictReader(open(f"{RES}/{tag}_{ds}/sweep.csv")))
def get(rows,mode,cfg,t=None):
    for r in rows:
        if r["scoring"]==mode and r["config"]==cfg and (t is None or abs(F(r["t_rag"])-t)<1e-6):
            return r
def num(x,d=3): return f"{F(x):.{d}f}"

# paired-McNemar significance stars for the cascade hallucination reduction (C3 vs C4)
def stars(p):
    if p is None: return ""
    p=float(p)
    return r"$^{***}$" if p<1e-3 else (r"$^{**}$" if p<1e-2 else (r"$^{*}$" if p<0.05 else r"$^{\mathrm{ns}}$"))
_MCN={}
_mcn=os.path.join(os.path.dirname(os.path.abspath(__file__)),"mcnemar_results.csv")
if os.path.exists(_mcn):
    for _r in csv.DictReader(open(_mcn)): _MCN[(_r["dataset"],_r["model"])]=_r["p"]
_DSLAB={"crosscodeeval_py":"CCE-line","repoeval_function":"RepoEval-fn",
        "crosscodelongeval_function":"CCLE-fn","crosscodelongeval_chunk":"CCLE-chunk"}

# ---- Table I: endpoints C1/C2 (ES, EM, idF1, hallA4B2, latency) ----
rows_out=[]
rows_out.append(r"\begin{tabular}{ll rrrr r rrrr r}")
rows_out.append(r"\toprule")
rows_out.append(r" & & \multicolumn{5}{c}{\textbf{C1 no-retrieve}} & \multicolumn{5}{c}{\textbf{C2 always-retrieve}}\\")
rows_out.append(r"\cmidrule(lr){3-7}\cmidrule(lr){8-12}")
rows_out.append(r"Dataset & M & ES & EM & idF1 & h$_{A4B2}$ & ms & ES & EM & idF1 & h$_{A4B2}$ & ms\\")
rows_out.append(r"\midrule")
for di,(ds,pretty,mode) in enumerate(DS):
    for mi,(tag,ml) in enumerate(MODELS):
        rws=load(tag,ds); c1=get(rws,mode,"C1_no_retrieve"); c2=get(rws,mode,"C2_always_retrieve")
        lead = f"\\multirow{{3}}{{*}}{{{pretty}}}" if mi==0 else ""
        rows_out.append(f"{lead} & {ml} & {num(c1['edit_similarity'])} & {num(c1['exact_match'])} & {num(c1['identifier_f1'])} & {num(c1['hall_A4B2'],4)} & {F(c1['latency_ms']):.0f} & "
                        f"{num(c2['edit_similarity'])} & {num(c2['exact_match'])} & {num(c2['identifier_f1'])} & {num(c2['hall_A4B2'],4)} & {F(c2['latency_ms']):.0f}\\\\")
    rows_out.append(r"\midrule" if di<len(DS)-1 else r"\bottomrule")
rows_out.append(r"\end{tabular}")
open(f"{OUT}/table_endpoints.tex","w").write("\n".join(rows_out))

# ---- Table II: cascade C3 vs C4 at t=0.05 (retr%, ES, hallA4B2, reduction) ----
T=0.05; out=[]
out.append(r"\begin{tabular}{ll rrr r rrr r}")
out.append(r"\toprule")
out.append(r" & & \multicolumn{3}{c}{\textbf{C3 CARD}} & & \multicolumn{4}{c}{\textbf{C4 cascade (ours)}}\\")
out.append(r"\cmidrule(lr){3-5}\cmidrule(lr){7-10}")
out.append(r"Dataset & M & r\% & ES & h$_{A4B2}$ & & r\% & ES & h$_{A4B2}$ & $\downarrow$\\")
out.append(r"\midrule")
for di,(ds,pretty,mode) in enumerate(DS):
    for mi,(tag,ml) in enumerate(MODELS):
        rws=load(tag,ds); c3=get(rws,mode,"C3_card",T); c4=get(rws,mode,"C4_cascade",T)
        h3,h4=F(c3['hall_A4B2']),F(c4['hall_A4B2'])
        red = f"{h3/h4:.1f}$\\times$" if h4>1e-9 else (r"$\rightarrow$0" if h3>0 else "--")
        red = red + stars(_MCN.get((_DSLAB[ds],ml)))
        lead = f"\\multirow{{3}}{{*}}{{{pretty}}}" if mi==0 else ""
        out.append(f"{lead} & {ml} & {F(c3['retrieval_pct']):.0f} & {num(c3['edit_similarity'])} & {num(c3['hall_A4B2'],4)} & & "
                   f"{F(c4['retrieval_pct']):.0f} & {num(c4['edit_similarity'])} & {num(c4['hall_A4B2'],4)} & {red}\\\\")
    out.append(r"\midrule" if di<len(DS)-1 else r"\bottomrule")
out.append(r"\end{tabular}")
open(f"{OUT}/table_cascade.tex","w").write("\n".join(out))

# ---- Table III: dual metrics (truncated main vs raw full) C1/C2 ES ----
DUAL=[("repoeval_function","RepoEval (func)","body"),
      ("crosscodelongeval_function","CCLE (func)","body"),
      ("crosscodelongeval_chunk","CCLE (chunk)","lines")]
out=[]
out.append(r"\begin{tabular}{ll rr r rr}")
out.append(r"\toprule")
out.append(r" & & \multicolumn{2}{c}{\textbf{C1 ES}} & & \multicolumn{2}{c}{\textbf{C2 ES}}\\")
out.append(r"\cmidrule(lr){3-4}\cmidrule(lr){6-7}")
out.append(r"Dataset & M & trunc. & raw & & trunc. & raw\\")
out.append(r"\midrule")
for di,(ds,pretty,mode) in enumerate(DUAL):
    for mi,(tag,ml) in enumerate(MODELS):
        rws=load(tag,ds)
        c1=get(rws,mode,"C1_no_retrieve"); c1f=get(rws,"full","C1_no_retrieve")
        c2=get(rws,mode,"C2_always_retrieve"); c2f=get(rws,"full","C2_always_retrieve")
        lead = f"\\multirow{{3}}{{*}}{{{pretty}}}" if mi==0 else ""
        out.append(f"{lead} & {ml} & {num(c1['edit_similarity'])} & {num(c1f['edit_similarity'])} & & {num(c2['edit_similarity'])} & {num(c2f['edit_similarity'])}\\\\")
    out.append(r"\midrule" if di<len(DUAL)-1 else r"\bottomrule")
out.append(r"\end{tabular}")
open(f"{OUT}/table_dual.tex","w").write("\n".join(out))
print("tables written to",OUT)
for f in os.listdir(OUT): print(" ",f)
