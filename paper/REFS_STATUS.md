# Reference status (`paper/refs.bib`)

Neither `evaluation-resolution-rsa` (tag `arxiv-v3`) nor `learning-rules-rsa` (tag
`arxiv-v4`) contains a `.bib` file; both papers end in an inline `thebibliography`.
"Converted" entries carry the `\bibitem` text of the named tag field by field, with the
original `\bibitem` quoted as a comment in `refs.bib`. "New" entries are not in either
tag; each carries a comment in `refs.bib` stating how and when it was verified.

| Key | Reference | Origin | Used in |
|---|---|---|---|
| `hebart2023` | THINGS-data (THINGS-fMRI), eLife 2023 | converted: learning-rules-rsa `arxiv-v4`, `paper/arxiv_upload_learning_rules_v3/learning_rules_rsa_paper_v3.tex` (also in evaluation-resolution-rsa `arxiv-v3`) | 1, 2.1 |
| `evalres` | arXiv:2608.12408 | **new, verified 2026-10-19** against the arXiv listing (title, author, version v3) | abstract, 1, 2.1-2.4, 3.3, 4 |
| `leutenegger2026` | arXiv:2604.16875, v4 | converted: evaluation-resolution-rsa `arxiv-v3`, `paper/arxiv_upload_v3/Evaluation_Resolution_Confounds_paper_v3.tex` | 1, 4 |
| `krizhevsky2009` | CIFAR-10 | **new** (techreport, University of Toronto); no DOI exists; standard citation of the technical report | 2.2 |
| `kriegeskorte2008` | RSA | converted: learning-rules-rsa `arxiv-v4` (also in evaluation-resolution-rsa `arxiv-v3`, title with a colon) | 1 |
| `nili2014` | RSA toolbox, noise ceiling | converted: learning-rules-rsa `arxiv-v4` (also in evaluation-resolution-rsa `arxiv-v3`) | not cited yet |
| `lillicrap2016` | Feedback alignment | converted: learning-rules-rsa `arxiv-v4` (also in evaluation-resolution-rsa `arxiv-v3`) | 2.2 |
| `rao1999` | Predictive coding | converted: learning-rules-rsa `arxiv-v4` (evaluation-resolution-rsa `arxiv-v3` has the short title) | 2.2 |
| `whittington2017` | Predictive coding approximating backprop | converted: learning-rules-rsa `arxiv-v4` (also in evaluation-resolution-rsa `arxiv-v3`) | 2.2 |
| `bi1998` | STDP | converted: learning-rules-rsa `arxiv-v4` (evaluation-resolution-rsa `arxiv-v3` has the short title) | 2.2 |
| `masquelier2007` | STDP, visual features | converted: evaluation-resolution-rsa `arxiv-v3` | 2.2 |
| `hoerl1970` | Ridge regression | **new, verified 2026-10-19** against the Crossref record (title, authors, volume, pages) | 2.5 |
| `paszke2019` | PyTorch (cited for torchvision) | **new, verified 2026-11-07** against NeurIPS and dblp-style listings (authors, pages 8024--8035; no DOI) | 2.2 |
| `marcel2010` | torchvision, ACM Multimedia 2010 | **new, verified 2026-10-19** against the Crossref record (title, authors, volume, pages) | 2.2 |
| `kornblith2019` | CKA, similarity of representations | **new, verified 2026-11-07** against the PMLR page (title, authors, volume, pages) | 1 |
| `walther2016` | reliability of dissimilarity measures | **new, verified 2026-10-19** against the Crossref record (title, authors, volume, pages) | 1 |
| `diedrichsen2017` | representational models | **new, verified 2026-10-19** against the Crossref record (title, authors, volume, pages) | 1 |
| `steegen2016` | multiverse analysis | **new, verified 2026-10-19** against the Crossref record (title, authors, volume, pages) | 1 |
| `carandini2012` | normalisation | **new, verified 2026-10-19** against the Crossref record (title, authors, volume, pages) | 4 |
| `truzzi2020` | untrained CNNs and human object representations, CogSci 2020 | **new, verified 2026-10-19** against the CogSci 2020 proceedings page (title and authors) | abstract, 1 |
| `kazemian2025` | convolutional architectures cortex-aligned de novo, Nature Machine Intelligence 2025 | **new, verified 2026-10-15** against the publisher page | abstract, 1 |
| `liu2022` | ConvNeXt, CVPR 2022 | **new, verified 2026-11-07** against the CVF open-access page (pages 11976--11986; dblp lists 11966--11976, not followed); DOI 10.1109/CVPR52688.2022.01167 | 4 |
| `cao2024` | explanatory models in neuroscience, part 2, Cognitive Systems Research 2024 | **new, verified 2026-11-07**: title, journal and DOI as listed in Kazemian et al. (2025); the arXiv version (2104.01489) is titled 'Part 2 -- constraint-based intelligibility'; author first names not verified | 4 |

`truzzi2025` (placeholder) was removed on 2026-10-14 and replaced by `truzzi2020` and
`kazemian2025` (published version of the bioRxiv preprint; replaced on 2026-10-15).

The learning-rule references are the ones the two tags cite for the rules:
evaluation-resolution-rsa `arxiv-v3` line 43 (FA: `lillicrap2016`; PC: `rao1999`,
`whittington2017`; STDP: `bi1998`, `masquelier2007`) and learning-rules-rsa `arxiv-v4`
line 114 (`lillicrap2016`, `rao1999`, `bi1998`).

Build: `pdflatex main`, `bibtex main`, `pdflatex main` (twice). BibTeX reports no errors or
warnings; undefined citations: 0.
## References added for version 2 (verified 2026-11-22)

| Key | Reference | Origin | Used in |
|---|---|---|---|
| `allen2022` | NSD, Nature Neuroscience 25:116-126 (2022) | **new, verified 2026-11-22** against the Crossref record of doi 10.1038/s41593-021-00962-x (title, 13 authors, volume, issue, pages; online 2021-12-16) | 2, 4 |
| `kay2013` | Two-stage cascade model of BOLD responses, PLoS Comput Biol 9(5):e1003079 | **new, verified 2026-11-22** against the Crossref record (title, authors, volume, issue, article number) | 6 |
| `ramirez2017` | Representational confusion (demeaning), bioRxiv 195271 | **new, verified 2026-11-22** against the Crossref record (title, author, posted 2017-09-28); no journal version found (a Journal of Vision 17(10):270 item with a similar title is a conference abstract), so preprint v1 is cited | 1, 6 |
| `taylor2025` | Framed RSA, CCN 2025 | **new, verified 2026-11-22** against the abstract PDF on 2025.ccneuro.org: title "Framed RSA: honoring representational geometry and regional-mean response preferences", authors JohnMark Taylor, Nikolaus Kriegeskorte; no DOI printed | 1, 6 |
| `bossio2025` | When do measured representational distances reflect the neural representational geometry?, eLife reviewed preprint 106812 | **new, verified 2026-11-22** against the Crossref record of doi 10.7554/eLife.106812.1 (title, authors Bossio Botero and Kriegeskorte, posted 2025-06-27, version 1); the eLife page itself returned a browser challenge | 1, 6 |
| `kremkow2014` | Dark dominance, PNAS 111(8):3170-3175 | **new, verified 2026-11-22** against the Crossref record (title, 9 authors, volume, issue, pages) | 6 |
