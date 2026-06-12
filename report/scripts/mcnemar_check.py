#!/usr/bin/env python
"""Paired McNemar test (exact binomial) for C3 vs C4 invented-identifier flags
at T_RAG=0.05, per (model x dataset). Mirrors 13_sweep_eval's scoring exactly:
static signal on the COMPLETE (FIM-stripped, untruncated) prediction."""
import sys, json, csv
sys.path.insert(0, "/Users/nistoralex/personal_projects/rag-static-analysis-experiment/src")
sys.path.insert(0, "/Users/nistoralex/personal_projects/rag-static-analysis-experiment/scripts")
from adaptive_retrieval.eval.datasets import DATASET_LOADERS
from adaptive_retrieval.metrics import invented_identifier_flag, mcnemar_test
from adaptive_retrieval.static_analysis.pyflakes_checker import PyflakesChecker

RES="/Users/nistoralex/personal_projects/rag-static-analysis-experiment/data/_resweep"
_FIM=("<|","▁<","<fim","<PRE>","<SUF>","<MID>","<EOT>","</s>","<｜","<repo_name>","<file_sep>","<|endoftext|>")
def strip(t):
    for m in _FIM:
        i=t.find(m)
        if i!=-1: t=t[:i]
    return t

T=0.05
MODELS=[("qwen25_0.5b","0.5B"),("qwen25_1.5b","1.5B"),("codellama_7b","7B")]
DS=[("crosscodeeval_py","CCE-line"),("repoeval_function","RepoEval-fn"),
    ("crosscodelongeval_function","CCLE-fn"),("crosscodelongeval_chunk","CCLE-chunk")]

print(f"McNemar C3 vs C4 (h_A4B2 flags), T_RAG={T}")
rows_out=[]
for ds,dlab in DS:
    insts={i.instance_id:i for i in DATASET_LOADERS[ds]()}
    for tag,mlab in MODELS:
        d=f"{RES}/{tag}_{ds}"
        load=lambda n:{json.loads(l)["instance_id"]:json.loads(l) for l in open(f"{d}/{n}.jsonl")}
        c1,c2,c3=load("C1_no_retrieve"),load("C2_always_retrieve"),load("C3_card")
        ids=[i for i in sorted(set(c1)&set(c2)&set(c3)&set(insts)) if c3[i].get("s_hat_0") is not None]
        pf=PyflakesChecker()
        a_rec,b_rec=[],[]
        for i in ids:
            inst=insts[i]; gold=inst.ground_truth
            p1=strip(c1[i]["prediction"]); p2=strip(c2[i]["prediction"])
            u1=set(pf.analyze(p1,inst.x_left,inst.x_right).significant_out_of_scope)
            u2=set(pf.analyze(p2,inst.x_left,inst.x_right).significant_out_of_scope)
            h1=invented_identifier_flag(gold,p1,u1); h2=invented_identifier_flag(gold,p2,u2)
            shat=float(c3[i]["s_hat_0"]); trig=bool(u1)
            c3_h = h2 if shat<T else h1
            c4_h = h2 if (shat<T or trig) else h1
            a_rec.append({"h":bool(c3_h)}); b_rec.append({"h":bool(c4_h)})
        r=mcnemar_test(a_rec,b_rec,key="h")
        n3=sum(x["h"] for x in a_rec); n4=sum(x["h"] for x in b_rec)
        print(f"{dlab:12s} {mlab:5s} n={len(ids):5d} C3_hall={n3:4d} C4_hall={n4:4d} b={r['b']:3d} c={r['c']:4d} p={r['p_value']:.2e}")
        rows_out.append((dlab,mlab,len(ids),n3,n4,r['b'],r['c'],r['p_value']))
with open("/tmp/report_work/mcnemar_results.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(["dataset","model","n","c3_hall","c4_hall","b","c","p"]); w.writerows(rows_out)
print("saved /tmp/report_work/mcnemar_results.csv")
