#!/usr/bin/env python
"""Quantify gold-in-retrieval leakage EXACTLY as the pipeline ran it:
BM25Retriever(inst.repo_files) [no exclusion], query=make_query(x_left), top_k=10.
For each instance: did any retrieved chunk contain gold lines?"""
import sys, json, random
sys.path.insert(0, "/Users/nistoralex/personal_projects/rag-static-analysis-experiment/src")
from adaptive_retrieval.eval.datasets import DATASET_LOADERS
from adaptive_retrieval.retriever import BM25Retriever, make_query

def gold_lines(gt):
    return [l.strip() for l in gt.splitlines() if l.strip()]

def leak_stats(name, sample=None, cache_by_repo=False):
    insts = list(DATASET_LOADERS[name]())
    if sample and len(insts) > sample:
        random.seed(0); insts = random.sample(insts, sample)
    n=0; any_leak=0; top1_leak=0; frac_sum=0.0; tf_in_top=0
    cache={}
    for inst in insts:
        key = inst.repository if cache_by_repo else None
        retr = cache.get(key)
        if retr is None:
            retr = BM25Retriever(inst.repo_files)
            if cache_by_repo: cache[key]=retr
        chunks = retr.retrieve(make_query(inst.x_left), top_k=10)
        if not chunks: continue
        n+=1
        gl = gold_lines(inst.ground_truth)
        if not gl: continue
        joined = ["\n".join("  "+l for l in c.text.splitlines()) for c in chunks]
        # line-level containment per chunk set
        allchunks_text = "\n".join(c.text for c in chunks)
        hit = sum(1 for l in gl if l in allchunks_text)
        frac = hit/len(gl); frac_sum += frac
        if frac > 0: any_leak += 1
        if any(l in chunks[0].text for l in gl): top1_leak += 1
        # how often is a target-file chunk in top-10?
        tf = inst.target_file
        if any(c.file_path == tf or tf.endswith(c.file_path) or c.file_path.endswith(tf) for c in chunks):
            tf_in_top += 1
    print(f"{name}: n={n}  ANY gold line in top-10: {100*any_leak/n:.1f}%  "
          f"in top-1: {100*top1_leak/n:.1f}%  mean gold-line coverage: {100*frac_sum/n:.1f}%  "
          f"target-file chunk in top-10: {100*tf_in_top/n:.1f}%", flush=True)

leak_stats("crosscodeeval_py")
leak_stats("crosscodelongeval_chunk", sample=1200)
leak_stats("crosscodelongeval_function", sample=1200)
leak_stats("repoeval_function", cache_by_repo=True)
print("DONE")
