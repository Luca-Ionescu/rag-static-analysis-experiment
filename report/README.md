# ML4SE26 G17 — Report sources

Compile on Overleaf (or locally) with:
    pdflatex main → bibtex main → pdflatex main → pdflatex main

Contents
- main.tex        report source. Sections from "Datasets" (IV.G) to the end were
                  written/finalised; text added or changed in the sections BEFORE
                  the Datasets section is wrapped in \rev{...} and renders in RED
                  so reviewers can see what changed when the experiments were
                  finalised (e.g. RepoEval function split instead of line/API,
                  addition of CrossCodeLongEval, the Tier 1–3 definitions, RQs,
                  abstract). Remove the red by redefining \rev to be a no-op once
                  reviewed.
- main.bib        references (all verified against arXiv / the attached PDFs).
- figures/        six PDF figures used in the report.
- tables/         three LaTeX table fragments, \input by main.tex.
- scripts/        make_figs.py, gen_tables.py — regenerate the figures/tables from
                  the per-instance sweep results (data/_resweep/ in the code repo).
- IEEEtran.cls, IEEEtran*.bst   unchanged template files.

Notes
- \usepackage{minted} is commented out (it needs -shell-escape + Pygments and is
  unused); re-enable if you add minted code listings.
