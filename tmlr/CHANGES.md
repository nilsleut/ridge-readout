# TMLR-Einreichungsversion: Änderungen gegenüber `arxiv-v1`

Stand: 2026-11-14 · Branch `tmlr-submission` · Quellen und PDF in `tmlr/` · Original in `paper/` unverändert.

## 1. Ergebnis auf einen Blick

| Punkt | Ergebnis |
|---|---|
| Haupttext (Abstract bis Conclusion) | **12 Seiten** (S. 1–12; die Conclusion endet auf S. 12) |
| Danach | S. 13: „Use of large language models“ + Referenzen; S. 14–24: Appendix A–J |
| PDF gesamt | 24 Seiten (`pdfinfo`) |
| Build | `sh tmlr/build.sh` (pdflatex → bibtex → pdflatex ×2, bei Bedarf ×3): 0 LaTeX-Warnungen, 0 undefinierte Referenzen, 0 Overfull-Boxen, bibtex ohne Warnungen |
| Zahlen | unverändert (siehe Abschnitt 6) |
| Leak-Prüfung | ohne Treffer (siehe Abschnitt 5) |

**Seitenregel.** Die Author Guidelines (jmlr.org/tmlr/author-guide.html) nennen keine feste Grenze („Submissions may be any length …“). Die 12-Seiten-Schwelle steht im Action-Editor-Guide: Einreichungen „longer than 12 pages (excluding any appendix)“ gelten als Typ „long“ und bekommen eine längere Begutachtungszeit. Ob Referenzen mitzählen, sagt keine der beiden Seiten. Ich zähle Referenzen nicht mit. Aber auch mit Referenzen ist der Text vor dem Appendix nur 13 Seiten lang. Beim Einreichen in OpenReview also „regular submission“ wählen.

## 2. Git

- Den Tag `arxiv-v1` gab es schon (annotiert, „arXiv:2611.15533 v1, as submitted“, Commit `6f3bda2`, auf `origin` gepusht). **Ich habe ihn nicht umgesetzt.** Ein gepushter Tag, der auf die eingereichte arXiv-Version zeigt, sollte bleiben, wo er ist. `paper/` ist zwischen `arxiv-v1` und `main` (`d12f973`) byte-identisch geprüft. Nach dem Tag kamen nur README, CITATION, Lizenz und requirements dazu. Der Tag markiert also genau den Paper-Stand, um den es geht.
- Branch `tmlr-submission` von `main` (`d12f973`) angelegt. Nichts gepusht.

## 3. Template

- `tmlr.sty`, `tmlr.bst`, `fancyhdr.sty` stammen aus github.com/JmlrOrg/tmlr-style-file (HEAD `7bf90ef`, 2023-06-30). Das Netz war erreichbar.
- Einbindung mit `\usepackage{tmlr}` ohne Option, also anonyme Einreichungsvariante („Anonymous authors / Paper under double-blind review“, Kopfzeile „Under review as submission to TMLR“).
- Bibliografiestil `plainnat` → `tmlr`.
- **Lokale TeX-Installation:** `tmlr.sty` lädt `eso-pic`, und das Layout braucht `wrapfig`. Beide fehlten in MiKTeX, und die automatische Nachinstallation blieb ohne Rückmeldung hängen. Ich habe beide mit `miktex packages install eso-pic` bzw. `wrapfig` installiert. Das ist die einzige Änderung außerhalb des Repos.
- Abweichungen vom Template (alle zulässig, nichts an Rändern oder Schrift geändert): zusätzliche Pakete `wrapfig`, `enumitem`, `float`, `placeins`, `booktabs`, `array`, `amsmath`. `math_commands.tex` aus dem Template wird nicht verwendet (optional). `hyperref` wie im Template, mit leeren PDF-Metadaten.

## 4. Anonymisierung

| Was | Änderung |
|---|---|
| Autorenblock | Name, „Independent researcher, Zug, Switzerland“ und E-Mail entfernt (`\author{}`). Das Template setzt „Anonymous authors“ |
| Danksagung | Abschnitt „Acknowledgements“ entfernt (Dank an einen Feedbackgeber). **Für die Camera-ready-Version wieder einfügen.** Funding-Angaben gab es keine |
| LLM-Hinweis | Aus der Danksagung in einen eigenen Abschnitt „Use of large language models“ verschoben. Inhalt gleich, nur „The author … takes“ → „The authors … take“, damit die Zahl der Autoren nicht verraten wird |
| Literatur | `evalres` (arXiv 2608.12408) und `leutenegger2026` (arXiv 2604.16875) jetzt mit `author = {Anonymous}`, `howpublished = {Preprint}`. arXiv-ID, DOI, eprint und Versionsnummer entfernt. Titel behalten. Im PDF erscheinen sie als Anonymous (2026a) = Evaluation-Resolution-Paper und Anonymous (2026b) = Learning-Rules-Paper |
| Zitierschlüssel | `leutenegger2026` → `lrrsa`, weil hyperref Zitierschlüssel als Linkziele (`cite.<key>`) ins PDF schreibt |
| Bib-Kommentare | Herkunftskommentare in `tmlr/refs.bib` entfernt (nennen eigene Repo-Namen und Tags). Sie stehen weiter in `paper/refs.bib` |
| Repo-URL | `https://github.com/nilsleut/ridge-readout` → `https://anonymous.4open.science/r/PLACEHOLDER` (Haupttext 2.7 und Appendix I.3). Andere identifizierende Links gab es im Text nicht |
| PDF-Metadaten | `\pdfinfo{/Title … /Author …}` entfernt. `hyperref`: pdftitle, pdfauthor, pdfsubject, pdfkeywords, pdfcreator und pdfproducer leer. `\pdfsuppressptexinfo=-1` (keine PTEX-Dateinamen), `\pdftrailerid{}` (keine aus Pfaden abgeleitete ID), `\pdfinfoomitdate=1` (kein Datum mit Zeitzone +01:00) |
| Abbildungen | Alle 6 Abbildungs-PDFs geprüft (Text, Metadaten, Rohbytes): keine Namen, Pfade oder Nutzernamen. Dateinamen neutral |
| Selbstbezüge im Text | siehe Abschnitt 7b |

## 5. Ergebnis der Leak-Prüfung

1. `pdftotext tmlr/main.pdf - | grep -iE "leutenegger|nilsleut|zug|gmail|github.com/nilsleut"` → **leer** (Exit-Code 1). Es gibt auch keine Literatureinträge mit Namen, die erklärt werden müssten.
2. `exiftool` ist auf diesem Rechner **nicht installiert**. Ersatzweise geprüft:
   - `pdfinfo tmlr/main.pdf`: Title, Author, Subject, Keywords, Creator und Producer leer. Kein CreationDate oder ModDate. Kein XMP-Metadatenstrom (`pdfinfo -meta` leer).
   - Rohbytes des fertigen PDFs: 0 Treffer für `leutenegger|nilsl|nilsleut|gmail|Users`.
   - Unkomprimierter Testbuild (`\pdfcompresslevel=0`, `\pdfobjcompresslevel=0`): 0 Treffer für Name, Nutzername, `C:/Users` und „Zug,“. Kein `/Author`-Eintrag, kein `/ID`, keine `PTEX.*`-Schlüssel. `/Title` kommt nur in Lesezeichen vor (Abschnittstitel).
   - Bitte vor dem Upload trotzdem einmal `exiftool tmlr/main.pdf` laufen lassen.
3. Quellen in `tmlr/` (tex, bib, sh): keine Namen, IDs oder URLs. **Ausnahme:** Die Prüfhilfen `tmlr/verify_numbers.py` und `tmlr/diff_vs_arxiv-v1.patch` enthalten zwangsläufig die Originalstrings, denn der Diff zeigt die gelöschten Zeilen mit Name und arXiv-IDs. Diese Dateien gehören nicht in einen anonymen Mirror (siehe Abschnitt 9).

## 6. Prüfung, dass keine Zahl verändert wurde

`python -I tmlr/verify_numbers.py` (aus dem Repo-Wurzelverzeichnis): **ALL CHECKS PASSED**

- `numbers.tex` ist byte-identisch zu `arxiv-v1`. Alle berichteten Werte sind Makros aus dieser Datei.
- Alle 572 Wert-Makros, die in `arxiv-v1` vorkommen, kommen auch in der TMLR-Version vor. Keines fehlt, keines ist neu.
- Kein Zahlen-Literal im TMLR-Quelltext fehlt in `arxiv-v1`. Ausgenommen sind nur Layoutmaße (`\vspace`, Breite der Umfluss-Abbildung).
- Wortgleiche Verschiebungen geprüft: Appendix I.1/I.2 = frühere Abschnitte 2.5/2.6 (nur Labels umbenannt). `sec_2_7.tex` unterscheidet sich nur durch die entfernte Überschrift und die URL. `sec_3_1`–`3_6` und alle alten Appendix-Dateien sind unverändert, abgesehen vom Zitierschlüssel.
- Zusätzlich auf PDF-Ebene: Jede Zahl im TMLR-PDF (Haupttext und gesamtes PDF) kommt im arXiv-PDF vor. In der Gegenrichtung fehlen nur die zwei arXiv-IDs (absichtlich entfernt), die Seitenzahl „27“ und das Tabellenfragment „03)“ (aus „sub-03“).
- Vollständiger Diff aller Textdateien: `tmlr/diff_vs_arxiv-v1.patch`.

## 7. Inhaltliche Formulierungsänderungen (alt → neu)

### 7a. In den Appendix verschoben

| Bisher (arXiv) | Jetzt im Haupttext | Volltext jetzt in |
|---|---|---|
| 2.5 Ridge readout | auf 2 Absätze verdichtet | Appendix I.1 (wortgleich) |
| 2.6 Mean-profile components and decomposition | auf 2 Absätze verdichtet | Appendix I.2 (wortgleich) |
| 2.7 Analysis plan, amendments and inference | auf 2 Absätze verdichtet | Appendix I.3 (wortgleich, nur URL ersetzt) |
| 3.4 Mean-profile alignment | auf die tragenden Aussagen gekürzt | Appendix J.1 (wortgleich) |
| 3.5 Removing the mean-profile component | gekürzt | Appendix J.2 (wortgleich) |
| 3.6 Norm and angular components | gekürzt | Appendix J.3 (wortgleich) |
| Abb. 4 (LOC-Multiverse) | Zahlen bleiben in 3.3 | Appendix J, jetzt Abb. 6 |

Die Appendices A–H haben ihre Buchstaben behalten (**D = Failed predictions**). I und J sind neu.

Abbildungsnummern: arXiv 1 sign_flip, 2 prereg_ladder, 3 multiverse_v1, 4 multiverse_loc, 5 gain, 6 separation → TMLR 1 prereg_ladder (jetzt neben dem Text von 3.1), 2 sign_flip, 3 multiverse_v1, 4 gain, 5 separation, 6 multiverse_loc (Appendix). Inhalte, Größen und Bildunterschriften sind unverändert.

**Was nur noch im Appendix ausführlich steht** (im Haupttext kurz genannt, mit Verweis):
- 2.5: wie die Selektion nach MSE statt r per Simulation begründet ist, und dass das gewählte α r nicht maximieren muss; die Einzelheiten des Plateau-Gates. Haupttext: „not by Pearson r … (Appendix I.1)“.
- 2.7: die vier nachträglichen Entscheidungen im Einzelnen, die vier vor dem Primärlauf festgelegten operativen Details, der Implementierungsfehler bei Δmin im Detail, die Liste der nicht umgesetzten Instrumente (Kapazitäts-Sweep, PLS, Ridge auf dem Testset, LOSO-Grenze; Letztere steht weiter in den Limitations), die Commit-Hashes der kritikbedingten Ergänzungen. Haupttext: Nicht-Präregistrierung, Nicht-Verblindung samt bekannter Teilergebnisse, „four further decisions after data had been seen“, „several pre-specified instruments were not implemented“ und die Inferenzgrenzen bleiben stehen.
- 3.5: M2-Slope-Werte, LOC-Werte der Ablation, Einzelwerte der einseitigen Ablation. Die Selbstkritik an der Regelformulierung zu M3 ist auf „the wording of the pre-stated rule claimed more; Appendix J.2“ verkürzt; der volle Satz steht in J.2.
- 3.6: Rangkorrelationen N/A/W, R²-Werte der Regression, die Einzelwerte der Slopes (resnet50, own CNN, convnext_tiny).

### 7b. Selbstbezüge (Anonymisierung)

| Ort | alt | neu |
|---|---|---|
| Abstract | „and we had reported an untrained-network advantage …~\citep{evalres}“ | „and an earlier analysis reported an untrained-network advantage …~\citep{evalres}“ |
| Abstract | „in our networks the resolution dependence …“ | „in the CIFAR-trained networks the resolution dependence …“ |
| Abstract | „(Spearman … across our networks, …)“ | „(Spearman … across the CIFAR-trained networks, …)“ |
| Abstract | „at V1 (our network and 5 of 6 ImageNet models)“ | „at V1 (the CIFAR-trained network and 5 of 6 ImageNet models)“ |
| Intro ¶2 | „We reported a result of this kind ourselves. In THINGS fMRI \citep{hebart2023}, an untrained network …“ | „A result of this kind was reported in THINGS fMRI \citep{hebart2023}: an untrained network …“ |
| Intro ¶4 | „Here we test our earlier result in four ways.“ | „Here we test this result in four ways: …“ |
| Intro ¶4 | „… generalise beyond our networks.“ | „… generalise beyond the CIFAR-trained networks;“ |
| Intro ¶5 | „… and we revise the corresponding claims of our earlier work.“ | „… and state which earlier claims \citep{evalres, lrrsa} do not hold.“ |
| Diskussion | „The advantage … that we reported earlier \citep{evalres} was …“ | „The advantage … reported by \citet{evalres} was …“ |
| Diskussion | „the convention of the earlier work“ | „the convention of the earlier analysis“ |
| Diskussion | „In our CIFAR-trained networks, the dependence …“ | „In the CIFAR-trained networks, the dependence …“ |
| Diskussion | „but in our earlier work the size of the effect …“ | „but in the earlier analysis \citep{evalres} the size of the effect …“ |
| Diskussion | Überschrift „Consequences for our earlier work.“ | „Consequences for earlier work.“ |
| Diskussion | „… are not robust to the analysis choices examined here. An audit of our earlier preprints identified the affected statements; revised versions refer to this work.“ | „… are not robust to the analysis choices examined here.“ (zweiter Satz gestrichen, weil er die Autorschaft verrät; **für Camera-ready wiederherstellen**) |
| Conclusion | „The advantage we reported earlier was absent …“ | „The previously reported advantage \citep{evalres} was absent …“ |

„we chose the thresholds ourselves“ (2.7) bleibt, weil es sich auf diese Arbeit bezieht. „own CNN“ bleibt in Text und Tabellen, weil die Grafik `separation.pdf` diese Beschriftung fest enthält (siehe Abschnitt 9).

### 7c. Kürzungen der Introduction (inhaltlich)

| alt | neu |
|---|---|
| „Comparisons between artificial networks and neural data have become a standard way to ask which models capture the computations of visual cortex. Representational similarity analysis (RSA) is one of the most widely used tools for this purpose: it summarises …“ | „Representational similarity analysis (RSA) is one of the most widely used tools for asking which artificial networks capture the computations of visual cortex: it summarises …“ |
| „… exceeding them \citep{…}. If true, such results bear on how much of early visual representation follows from architecture alone and how much from learning.“ | Satz gestrichen |
| „That convention has a property that is easy to overlook. Correlation distance …“ | „Correlation distance …“ |
| Vier Sätze „First, … Second, … Third, … Fourth, an exact decomposition … separates a norm component from an angular component.“ | ein Satz mit (i)–(iv); (iv) „… separates norm and angular components.“ Inhalt gleich |

### 7d. Neu formulierte Kurzfassungen

Die Kurzfassungen in 2.5–2.7 und 3.4–3.6 (`drafts/methods_short.tex`, `drafts/sec_3_4_short.tex`, `drafts/sec_3_ablation_short.tex`, `drafts/sec_3_5_short.tex`) sind neu geschrieben. Sie übernehmen nur Aussagen und Makros, die im Original stehen. Zusammenfassende Formulierungen, die so nicht wörtlich im Original stehen:
- 3.4: „the partial slope remained above half of S_paper“ (Original: Wert und Intervall, „above half of S_paper“). „Descriptively, V1 voxel alignment correlated with stimulus mean luminance and RMS contrast, and at 224 px the untrained network's alignment had the same signs (Fig. 4c)“ (Original: die sechs Einzelwerte).
- 3.5: „the paired difference between the two variants was resolved (X1: met)“ (Original: Wert mit Intervall). „With the orthogonal remainder on one side only, both intervals included 0, so no attribution to a side is claimed; the point estimates lay close to that of the orthogonal variant (descriptive, no paired contrast tested)“ (Original: Werte, Intervalle, gleicher Inhalt).
- 3.6: „Descriptively (no decision rule) …“ fasst den Absatz zum Resolution-Slope zusammen (Original: „a descriptive point fixed in advance without a decision rule“). „convnext_tiny was an exception, with CIs including 0 for its V1 angle contrast and its angle slope“ (Original: Werte mit Intervallen).
- 2.5: „Under this pooled score, features without signal yield negative rather than zero scores.“ (Original gleichlautend, mit Begründung. Der Wert des Mittelwert-Modells steht weiter in 3.1.)
- 2.7: Der Satz zum kleinsten erreichbaren p (1/32) ist aus 2.7 gestrichen, weil er gleichlautend in 3.1 steht.

### 7e. Reines Layout (kein Inhalt)

Abb. 1 (prereg_ladder, native 3,5 in) wird von 3.1 umflossen (`wrapfigure`). Die Aufzählung „Ridge and RSA address different questions“ hat kompakte Abstände (`[nosep]`), Wortlaut unverändert. Neuer Einleitungssatz für Appendix I, J und Abb. 6.

### 7f. Abgeschwächte oder gestrichene Aussagen wegen fehlenden Belegs

Keine. Jede Aussage, die in Abstract, Diskussion, Tabelle 1 und Limitations steht, hat ihren Beleg weiterhin im Haupttext (Zahl oder Verweis auf eine Abbildung) oder hatte ihn schon im Original nur über einen Appendix-Verweis. Geprüft habe ich insbesondere: einseitige Ablation (3.5), G2-Restslope (3.4), Luminanz/Kontrast-Vorzeichen (3.4, Abb. 4c), Winkel-Slope der ImageNet-Modelle (3.6), 0,1 % Rangvarianz (3.6), X1 in Tabelle 1 (3.5), die sechs verfehlten Vorhersagen (2.7 und Limitations, Details in Appendix D).

## 8. Haupttext-Inhalt laut Vorgabe

- Abstract: unverändert bis auf die Selbstbezüge (7b).
- Introduction: leicht gekürzt (7c).
- Methoden 2.1–2.4: unverändert.
- Ergebnisse 3.1–3.3: unverändert (nur die Abbildungsplatzierung, 7e).
- Diskussion mit Tabelle 1 und den fünf Empfehlungen: vollständig, nur Selbstbezüge geändert (7b).
- Die sechs verfehlten Vorhersagen: im Haupttext in 2.7 („including the 6 that were not met (Appendix D)“), im Abstract („6 pre-stated predictions were not met“) und in den Limitations, außerdem einzeln in 3.2 (P_mech2), 3.3 (GEN3, GEN4), 3.4 (G2), 3.5 (M3) und 3.6 (SEP2).

## 9. Offene Punkte für dich

1. **Anonymer Mirror:** `PLACEHOLDER` in `drafts/methods_short.tex` und `drafts/sec_2_7.tex` durch die echte anonymous.4open.science-ID ersetzen und neu bauen (`sh tmlr/build.sh`). Im Mirror die Ersetzungsbegriffe setzen (mindestens Leutenegger, nilsleut, Zug, gmail, die arXiv-IDs 2604.16875, 2608.12408 und 2611.15533). README, CITATION.cff, LICENSE und PROVENANCE im Repo nennen dich.
2. **Nicht mitspiegeln:** `tmlr/diff_vs_arxiv-v1.patch`, `tmlr/verify_numbers.py` und diese `CHANGES.md` enthalten die Originalstrings bzw. beschreiben die Anonymisierung. Am einfachsten den ganzen Ordner `tmlr/` und `paper/` ausschließen.
3. **„the full commit history“:** Der Text (2.7 und Appendix I.3) verweist für die Zeitstempel der Regeln auf die Commit-Historie. anonymous.4open.science spiegelt meines Wissens nur einen Dateistand, keine Historie. Bitte prüfen. Wenn die Historie dort fehlt, stimmt der Satz für Gutachter nicht. Ich habe ihn nicht geändert, weil das eine inhaltliche Entscheidung ist.
4. **Commit-Hashes im Text** (`74853dc`, `ac59694` in Appendix I.3; Konfigurations-Hash `206fa875cca5` in 2.7): Über die GitHub-Suche lassen sie sich deinem öffentlichen Repo zuordnen. Sie belegen die Zeitstempel und sind deshalb stehen geblieben. Abwägen, ob sie für die Review-Phase raus sollen.
5. **Titel der eigenen Preprints** stehen weiter in den Anonymous-Einträgen und sind per Suche auffindbar. Gutachter sollen nicht suchen, aber wenn du strenger sein willst, die Titel durch „Anonymous preprint“ ersetzen. Ohne Titel können Gutachter allerdings nicht nachvollziehen, was revidiert wird.
6. **„own CNN“** steht fest in der Abbildung `separation.pdf` (Abb. 5) und deshalb auch in Text und Tabellen. Zusammen mit „the 30 checkpoints of Anonymous (2026a)“ deutet das auf dieselben Autoren hin. Zum Entschärfen die Abbildung mit „CIFAR CNN“ neu erzeugen (wie in `gain.pdf`) und die Textstellen anpassen.
7. **arXiv-Version dieser Arbeit** (2611.15533): Nach den TMLR-Regeln erlaubt, solange die Einreichung sie nicht verlinkt. Das tut sie nicht.
8. **Schon im Original:** Diskussion, Absatz „A luminance-and-contrast bank …“: „the norm of the voxel response vectors did not depend on luminance or contrast“. Für diese Aussage finde ich weder im Original noch hier einen Beleg (keine Zahl, kein Abschnitt). Nicht verändert, bitte prüfen.
9. **Lesbarkeit:** Abb. 4(c) hat sehr kleine Achsen- und Legendenbeschriftungen (etwa 5 pt). Das ist gleich wie im Original. Die übrigen Abbildungen sind bei 110 dpi gut lesbar. Gerenderte Seiten habe ich nicht ins Repo gelegt.
10. **Camera-ready:** Danksagung und den gestrichenen Audit-Satz (7b) wiederherstellen, Selbstbezüge zurück in die erste Person, `\usepackage[accepted]{tmlr}`.
11. **Optional:** TMLR empfiehlt bei Bedarf ein „Broader Impact Statement“. Für diese Arbeit vermutlich nicht nötig. Nicht ergänzt.
