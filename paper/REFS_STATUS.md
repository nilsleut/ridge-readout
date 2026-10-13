# Reference status (`paper/refs.bib`)

Neither `evaluation-resolution-rsa` (tag `arxiv-v3`) nor `learning-rules-rsa` (tag
`arxiv-v4`) contains a `.bib` file; both papers end in an inline `thebibliography`.
"Converted" entries carry the `\bibitem` text of the named tag field by field, with the
original `\bibitem` quoted as a comment in `refs.bib`. "New" entries are not in either
tag and are marked `% TO VERIFY` in `refs.bib`.

| Key | Reference | Origin | Used in |
|---|---|---|---|
| `hebart2023` | THINGS-data (THINGS-fMRI), eLife 2023 | converted: learning-rules-rsa `arxiv-v4`, `paper/arxiv_upload_learning_rules_v3/learning_rules_rsa_paper_v3.tex` (also in evaluation-resolution-rsa `arxiv-v3`) | 2.1 |
| `evalres` | arXiv:2608.12408 | **new, to verify** (title and author from evaluation-resolution-rsa `arxiv-v3` TeX lines 13-16; year and DOI not from a source) | 2.1, 2.2, 2.3, 2.4 |
| `leutenegger2026` | arXiv:2604.16875, v4 | converted: evaluation-resolution-rsa `arxiv-v3`, `paper/arxiv_upload_v3/Evaluation_Resolution_Confounds_paper_v3.tex` | not cited yet |
| `krizhevsky2009` | CIFAR-10 | **new, to verify, incomplete** (no DOI) | 2.2 |
| `kriegeskorte2008` | RSA | converted: learning-rules-rsa `arxiv-v4` (also in evaluation-resolution-rsa `arxiv-v3`, title with a colon) | not cited yet |
| `nili2014` | RSA toolbox, noise ceiling | converted: learning-rules-rsa `arxiv-v4` (also in evaluation-resolution-rsa `arxiv-v3`) | not cited yet |
| `lillicrap2016` | Feedback alignment | converted: learning-rules-rsa `arxiv-v4` (also in evaluation-resolution-rsa `arxiv-v3`) | 2.2 |
| `rao1999` | Predictive coding | converted: learning-rules-rsa `arxiv-v4` (evaluation-resolution-rsa `arxiv-v3` has the short title) | 2.2 |
| `whittington2017` | Predictive coding approximating backprop | converted: learning-rules-rsa `arxiv-v4` (also in evaluation-resolution-rsa `arxiv-v3`) | 2.2 |
| `bi1998` | STDP | converted: learning-rules-rsa `arxiv-v4` (evaluation-resolution-rsa `arxiv-v3` has the short title) | 2.2 |
| `masquelier2007` | STDP, visual features | converted: evaluation-resolution-rsa `arxiv-v3` | 2.2 |
| `hoerl1970` | Ridge regression | **new, to verify** (with DOI) | 2.5 |
| `paszke2019` | PyTorch (cited for torchvision) | **new, to verify** (arXiv DOI) | 2.2 |

The learning-rule references are the ones the two tags cite for the rules:
evaluation-resolution-rsa `arxiv-v3` line 43 (FA: `lillicrap2016`; PC: `rao1999`,
`whittington2017`; STDP: `bi1998`, `masquelier2007`) and learning-rules-rsa `arxiv-v4`
line 114 (`lillicrap2016`, `rao1999`, `bi1998`).

Build: `pdflatex main`, `bibtex main`, `pdflatex main` (twice). BibTeX reports no errors or
warnings; undefined citations: 0.
