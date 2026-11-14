"""Check that the TMLR version changes no number relative to the arXiv-v1 tag.

Run from the repository root:  python -I tmlr/verify_numbers.py
Reads the arXiv-v1 sources via `git show arxiv-v1:<path>` and the TMLR sources from tmlr/.
Writes tmlr/diff_vs_arxiv-v1.patch (unified diff of all text files) and prints a report.
"""
import difflib, io, os, re, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TAG = 'arxiv-v1'

def git_show(path):
    return subprocess.run(['git', 'show', f'{TAG}:{path}'], cwd=ROOT, capture_output=True,
                          check=True).stdout.decode('utf-8')

def read(path):
    return io.open(os.path.join(ROOT, path), encoding='utf-8').read()

orig_drafts = [os.path.basename(p) for p in subprocess.run(
    ['git', 'ls-tree', '--name-only', TAG, 'paper/drafts/'],
    cwd=ROOT, capture_output=True, check=True).stdout.decode().split()]
orig_files = {f: git_show('paper/' + f) for f in
              ['main.tex', 'refs.bib', 'numbers.tex'] + ['drafts/' + n for n in orig_drafts]}
new_files = {f: read('tmlr/' + f) for f in
             ['main.tex', 'refs.bib', 'numbers.tex'] +
             ['drafts/' + n for n in sorted(os.listdir(os.path.join(ROOT, 'tmlr', 'drafts'))) if n.endswith('.tex')]}

report = []
ok = True

# 1. numbers.tex must be byte-identical (all reported values are macros defined there)
same = orig_files['numbers.tex'] == new_files['numbers.tex']
report.append(f'[1] numbers.tex identical to {TAG}: {same}')
ok &= same

# 2. numeric literals: every literal in the TMLR text must occur in the arXiv-v1 text
NUM = re.compile(r'(?<![A-Za-z\\])[-+]?\d+(?:[.,{}]\d+)*(?:\\,?%|%)?')
def strip_comments(s):
    return '\n'.join(re.sub(r'(?<!\\)%.*', '', l) for l in s.split('\n'))
def literals(s):
    s = strip_comments(s)
    s = re.sub(r'\\(?:ref|label|cite[pt]?|citealt|input|includegraphics|url|bibliography\w*)(\[[^\]]*\])?\{[^}]*\}', ' ', s)
    # layout dimensions are not data: \vspace{..}, wrapfigure width, includegraphics width
    s = re.sub(r'\\vspace\{[^}]*\}', ' ', s)
    s = re.sub(r'\\begin\{wrapfigure\}\{[^}]*\}\{[^}]*\}', ' ', s)
    s = re.sub(r'\\[A-Za-z@]+', ' ', s)  # macro names (values come from numbers.tex)
    return set(m.group(0).lstrip('+') for m in NUM.finditer(s))
orig_lits = set().union(*(literals(v) for k, v in orig_files.items() if k != 'numbers.tex' and k.endswith('.tex')))
new_lits = set().union(*(literals(v) for k, v in new_files.items() if k != 'numbers.tex' and k.endswith('.tex')))
extra = sorted(new_lits - orig_lits)
report.append(f'[2] numeric literals in TMLR .tex not present in {TAG} .tex: {extra if extra else "none"}')
ok &= not extra

# 3. macros: every value macro used in TMLR text is defined (unchanged) in numbers.tex;
#    every macro used in arXiv-v1 is still used somewhere in the TMLR version (nothing silently dropped)
defined = set(re.findall(r'\\newcommand\{\\(\w+)\}', orig_files['numbers.tex']))
def used(files):
    return set().union(*(set(re.findall(r'\\(\w+)', strip_comments(v))) & defined
                         for k, v in files.items() if k != 'numbers.tex' and k.endswith('.tex')))
u_orig, u_new = used(orig_files), used(new_files)
report.append(f'[3] value macros used in {TAG}: {len(u_orig)}; in TMLR: {len(u_new)}; '
              f'dropped: {sorted(u_orig - u_new) or "none"}; new: {sorted(u_new - u_orig) or "none"}')
ok &= not (u_orig - u_new) and not (u_new - u_orig)

# 4. verbatim moves
def body(s):
    return '\n'.join(l for l in s.split('\n') if not l.startswith('%')).strip()
seg = orig_files['main.tex']
a = seg.index('\\subsection{Ridge readout}'); b = seg.index('\\input{drafts/sec_2_7}')
moved_orig = seg[a:b].replace('\\label{sec:ridge}', '\\label{app:ridge}').replace(
    '\\label{sec:gainmethods}', '\\label{app:gainmethods}').strip()
v = body(new_files['drafts/app_extmethods_ridge_gain.tex']) == moved_orig
report.append(f'[4a] Appendix I.1/I.2 = former Sections 2.5/2.6 verbatim (labels renamed): {v}')
ok &= v
for f in ['drafts/sec_3_4.tex', 'drafts/sec_3_ablation.tex', 'drafts/sec_3_5.tex', 'drafts/sec_3_1.tex',
          'drafts/sec_3_2.tex', 'drafts/sec_3_3.tex'] + [k for k in orig_files if k.startswith('drafts/app_')]:
    # only permitted difference: citation key leutenegger2026 -> lrrsa (anonymity of PDF link targets)
    v = orig_files[f].replace('{leutenegger2026}', '{lrrsa}') == new_files[f]
    report.append(f'[4b] {f} unchanged (except cite key rename): {v}')
    ok &= v
o27 = orig_files['drafts/sec_2_7.tex'].replace('https://github.com/nilsleut/ridge-readout',
                                                'https://anonymous.4open.science/r/PLACEHOLDER')
o27 = o27.replace('\\subsection{Analysis plan, amendments and inference}\n\\label{sec:inference}\n', '')
v = body(o27) == body(new_files['drafts/sec_2_7.tex'])
report.append(f'[4c] drafts/sec_2_7.tex differs only by heading removal and URL: {v}')
ok &= v

# 5. unified diff of all text files
out = []
for f in sorted(set(orig_files) | set(new_files)):
    o = orig_files.get(f, '').splitlines(keepends=True)
    n = new_files.get(f, '').splitlines(keepends=True)
    out.extend(difflib.unified_diff(o, n, f'{TAG}:paper/{f}', f'tmlr/{f}'))
io.open(os.path.join(ROOT, 'tmlr', 'diff_vs_arxiv-v1.patch'), 'w', encoding='utf-8', newline='\n').write(''.join(out))
report.append(f'[5] wrote tmlr/diff_vs_arxiv-v1.patch ({len(out)} lines)')

print('\n'.join(report))
print('ALL CHECKS PASSED' if ok else 'SOME CHECKS FAILED')
sys.exit(0 if ok else 1)
