# Reference status (`paper/refs.bib`)

Neither `evaluation-resolution-rsa` (tag `arxiv-v3`) nor `learning-rules-rsa` (tag
`arxiv-v4`) contains a `.bib` file; both papers end in an inline `thebibliography`.
"Converted" entries carry the `\bibitem` text of the named tag field by field, with the
original `\bibitem` quoted as a comment in `refs.bib`. "New" entries are not in either
tag and are marked `% TO VERIFY` in `refs.bib`.

| Key | Reference | Origin | Used in |
|---|---|---|---|
| `hebart2023` | THINGS-data (THINGS-fMRI), eLife 2023 | converted: learning-rules-rsa `arxiv-v4`, `paper/arxiv_upload_learning_rules_v3/learning_rules_rsa_paper_v3.tex` (also in evaluation-resolution-rsa `arxiv-v3`) | 1, 2.1 |
| `evalres` | arXiv:2608.12408 | **new, to verify** (title and author from evaluation-resolution-rsa `arxiv-v3` TeX lines 13-16; year, DOI, eprint, archivePrefix as specified by the author) | abstract, 1, 2.1-2.4, 3.3, 4 |
| `leutenegger2026` | arXiv:2604.16875, v4 | converted: evaluation-resolution-rsa `arxiv-v3`, `paper/arxiv_upload_v3/Evaluation_Resolution_Confounds_paper_v3.tex` | 1, 4 |
| `krizhevsky2009` | CIFAR-10 | **new, to verify** (techreport, University of Toronto; no DOI exists) | 2.2 |
| `kriegeskorte2008` | RSA | converted: learning-rules-rsa `arxiv-v4` (also in evaluation-resolution-rsa `arxiv-v3`, title with a colon) | 1 |
| `nili2014` | RSA toolbox, noise ceiling | converted: learning-rules-rsa `arxiv-v4` (also in evaluation-resolution-rsa `arxiv-v3`) | not cited yet |
| `lillicrap2016` | Feedback alignment | converted: learning-rules-rsa `arxiv-v4` (also in evaluation-resolution-rsa `arxiv-v3`) | 2.2 |
| `rao1999` | Predictive coding | converted: learning-rules-rsa `arxiv-v4` (evaluation-resolution-rsa `arxiv-v3` has the short title) | 2.2 |
| `whittington2017` | Predictive coding approximating backprop | converted: learning-rules-rsa `arxiv-v4` (also in evaluation-resolution-rsa `arxiv-v3`) | 2.2 |
| `bi1998` | STDP | converted: learning-rules-rsa `arxiv-v4` (evaluation-resolution-rsa `arxiv-v3` has the short title) | 2.2 |
| `masquelier2007` | STDP, visual features | converted: evaluation-resolution-rsa `arxiv-v3` | 2.2 |
| `hoerl1970` | Ridge regression | **new, to verify** (Technometrics 12(1):55-67, with DOI) | 2.5 |
| `paszke2019` | PyTorch (cited for torchvision) | **new, to verify** (arXiv DOI) | 2.2 |
| `marcel2010` | torchvision, ACM Multimedia 2010 | **new, to verify** (with DOI) | 2.2 |
| `kornblith2019` | CKA, similarity of representations | **new, to verify** (no DOI given) | 1 |
| `walther2016` | reliability of dissimilarity measures | **new, to verify** (with DOI; also an inline `\bibitem` in learning-rules-rsa `arxiv-v4`) | 1 |
| `diedrichsen2017` | representational models | **new, to verify** (with DOI) | 1 |
| `steegen2016` | multiverse analysis | **new, to verify** (with DOI) | 1 |
| `carandini2012` | normalisation | **new, to verify** (with DOI) | 4 |
| `truzzi2025` | untrained CNNs and early visual cortex, bioRxiv | **new, placeholder**: title, DOI and authors to be filled by the author (also an inline `\bibitem` in both tags) | abstract, 1 |

The learning-rule references are the ones the two tags cite for the rules:
evaluation-resolution-rsa `arxiv-v3` line 43 (FA: `lillicrap2016`; PC: `rao1999`,
`whittington2017`; STDP: `bi1998`, `masquelier2007`) and learning-rules-rsa `arxiv-v4`
line 114 (`lillicrap2016`, `rao1999`, `bi1998`).

Build: `pdflatex main`, `bibtex main`, `pdflatex main` (twice). BibTeX reports no errors or
warnings; undefined citations: 0.
