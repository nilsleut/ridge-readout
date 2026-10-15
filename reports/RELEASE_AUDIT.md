# Release audit

Date: 2026-10-15. Repository state: `git ls-files` at HEAD `11f1a7d` (178 tracked files),
full history (`git log -p --all`, 109 commits). Read-only checks; nothing was changed.

| Check | Result |
|---|---|
| a) Size and binary data | No tracked file above 5 MB (largest: `results/step7/multiverse.csv`, 2.7 MB). No `.npz`, `.npy`, `.pt`, `.h5`, `.pth` or `.pkl` files. Nothing tracked under `arm_b_out/`, `pilot_features/` or `data/`. **Pass.** |
| b) Environment files | `.env.local` is not tracked and is listed in `.gitignore` (line 10). `.env.example` has empty values and no personal paths. **Pass.** |
| c) Secrets | Patterns: `api_key`, `apikey`, `token`, `secret`, `password`, `sk-…`, `ghp_`, `AKIA`, `BEGIN … PRIVATE KEY`, and `modal` together with `key`. No hit in the history or in `results/`. Tracked files contain only the word "token" in the sense of ViT tokens or a Python token (`amendments/…step8…:23`, `paper/METHODS_LITERALS.md:51`, `paper/main.tex:85`, `scripts/export_macros.py:285`, `scripts/step8_extract_pretrained.py:15-16`); none is a credential. **Pass.** |
| d) Absolute user paths | 7 hits, all in documentation: `CHANGES_FROM_ORIGIN.md` (6, the original paths that were replaced, lines 25-43) and `reports/PHASE0_RECONNAISSANCE.md:24` (source repository path). No hit in code (`src/`, `scripts/`, `tests/`, configuration). **Pass (documentation hits only).** |
| e) Raw data | No THINGS-fMRI responses, betas or other raw data. `DATA_MANIFEST.txt` lists paths, sizes and sha256 only. The CSV files under `results/` are derived scores (RSA and ridge values, rule outcomes); `results/step2/d1a_real_voxels.csv` has 4 rows of control scores. **Pass.** |
| f) AI references | None in any commit message (`git log --all --format=%B`) or in tracked files other than `paper/main.tex`. All 109 commits are authored and committed by Nils Leutenegger. In `paper/main.tex` the placeholder comment `% AI disclosure sentence` is present and is replaced in a later commit. **Pass.** |
