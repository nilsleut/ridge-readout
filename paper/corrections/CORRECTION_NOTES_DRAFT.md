# Correction notes for earlier preprints — DRAFT

Status: draft, not posted. Source of every number: reports/AUDIT_RAW_RSA_CLAIMS.md
(values under z from results/step5/, commit 77e801a). "[NEW]" = arXiv ID of the
ridge-readout paper, to be filled in once it is announced.

Timing: post all replacements within one or two days AFTER [NEW] is announced, so that
every note can cite it. Do not post any of them before.

Form: as in the August correction notes, a short revision note at the start of the
abstract plus the arXiv "Comments" field. Original text otherwise unchanged; titles
unchanged (keeps the record citable; the note states that the title claim is not
supported).

---

## 2604.16875 (learning-rules-rsa) — title affected

Abstract prefix:

> Revision note (v5): A follow-up study [NEW] shows that the central result of this
> paper, that the untrained network exceeds backpropagation in V1 alignment at 224 px,
> depends on the analysis convention used here (globally pooled features, correlation
> distance, voxels normalised over all trials). When model units are standardised over
> the evaluated stimuli, the V1 difference reverses (untrained minus backpropagation
> -0.011 [-0.017, -0.005]), and a pre-specified ridge readout on the full Conv1 feature
> map also places backpropagation above the untrained network. The claim in the title,
> the V1 and V2 rankings and the conclusion that early visual alignment is driven more
> by architecture than by the learning rule are therefore not supported. The advantage
> of backpropagation at LOC survives. The text below is unchanged.

Comments field:
> v5: revision note on the V1 result; see arXiv:[NEW]. Main text unchanged.

---

## 2608.12408 (evaluation-resolution-rsa) — title affected

Abstract prefix:

> Revision note (v4): A follow-up study [NEW] shows that the evaluation-resolution
> dependence of the untrained-minus-backpropagation difference at V1 reported here is
> specific to raw correlation-distance RSA. When model units are standardised over the
> evaluated stimuli, the slope from 32 to 224 px is +0.002 [-0.004, +0.007] instead of
> +0.031 [0.024, 0.038], and the difference is negative at every resolution. The V1
> headline (+0.030 at 224 px) and the content-limited ratio R therefore do not hold
> under this analysis. The advantage of backpropagation at LOC, its loss under
> upsampling, and the widening of the V1 gap when content is limited survive.
> In ImageNet models, a smaller resolution dependence persists in the angular component
> of the representations [NEW]. The text below is unchanged.

Comments field:
> v4: revision note on the V1 resolution effect; see arXiv:[NEW]. Main text unchanged.

---

## 2605.22401 (Cross_Species_RSA) — human side affected, macaque side not tested

BEFORE writing the final note: download the arXiv source of the current version and
diff it against the working tree in Projekte_2/Cross_Species_RSA (the TeX there is
untracked and no tag exists). Commit and tag the exact posted version first.

Abstract prefix (draft):

> Revision note: The human V1 results in this paper (untrained network above
> backpropagation; human V1 ranking) rest on the same analysis convention that a
> follow-up study [NEW] shows to be sensitive to unit offsets; under unit
> standardisation the human V1 ranking changes. The macaque comparisons and the
> cross-species conclusions were not re-tested. Readers should treat the human-side
> V1 rankings as convention-dependent. The text below is unchanged.

---

## 2605.30556 (training_dynamics) — affected at the endpoint, point estimates only

BEFORE writing the final note: as above, establish the posted version (the folder is
not a git repository; initialise one, commit the posted source, tag it).

Abstract prefix (draft):

> Revision note: The degradation of V1 alignment with training reported here was
> measured with raw correlation-distance RSA. A follow-up study [NEW] finds that under
> unit standardisation, backpropagation after training scores above the untrained
> network at V1 (fixed layer, endpoint only). The trajectory and best-layer analyses of
> this paper were not re-tested. The claim that training degrades V1 alignment is
> therefore not established. The text below is unchanged.

Caveat to keep in mind: for this paper the under-z evidence is a point estimate at the
endpoint with a fixed layer (audit #1, #2). The note must not claim more than that.

---

## Checklist before posting

- [ ] [NEW] announced; ID inserted everywhere.
- [ ] Every number above re-checked against AUDIT_RAW_RSA_CLAIMS.md.
- [ ] 2605.22401 and 2605.30556: posted versions identified and tagged.
- [ ] Version numbers (v5, v4, ...) match the actual next arXiv version of each paper.
- [ ] Repo tags for the new versions set after announcement; old tags never moved.
