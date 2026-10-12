# Fixed full-description candidate identity trial

This is an offline public diagnostic. Production API/Web dispatch, detector
candidate generation, shared model internals, checkpoints and training are
unchanged. It does not implement the complete reference-first spatial search.

## Model and candidate boundary

`FrozenCLIPCropScorer` receives an already loaded, frozen CLIP plus the existing
`NativeTextEncoder.encode_descriptions` callable. It requires eval mode, frozen
parameters and the model's complete `visual.preprocess_cfg`. The official
`image_transform_v2(PreprocessCfg(**cfg), is_train=False)` is reconstructed without
creating another model or downloading weights. Each real full-description box
is clipped and rounded outward for a BGR-to-RGB crop. Candidate coordinates and
order are preserved; only the crop bounds are rounded. No ROI/subject/RefSeg
candidates are added. Each valid crop is encoded once and reused for all text
scores. Long, narrow crops may lose their ends through the official center crop;
crop bounds and aspect ratios are recorded rather than silently changing it.

The existing conservative parser runs before inference. Supported descriptions
separate target attributes from reference identities. Unsupported syntax keeps
the complete description and explicitly records that fallback. This parser does
not apply a target's absolute position to every reference object.

## Frozen comparison arms

All arms use exactly the same original full-description candidates:

- A: original NativeDetection selection.
- B: arithmetic mean of crop cosine with the bare target phrase and
  `a satellite remote sensing photo of {target}`.
- C: B minus the maximum reference score, using the same two text templates per
  reference. With no parsed reference C equals B.
- E, enabled by `--hybrid`: original raw native phrase score plus the original
  native absolute-position soft score with weight `.15`, then plus `.15` times
  C's minmax score over that same complete candidate pool. Raw native phrase
  scores are never minmax-normalized. A tied C pool contributes zero. The CLI
  directly reuses `NativeDetection._position_adjusted_scores` with float32 boxes,
  float32 raw phrase scores, original image dimensions and the original complete
  description after the original text cleaner. The task module receives these
  reconstructed baseline scores without importing API code. Missing original
  scores, CLIP failure or an incomplete valid crop pool explicitly falls E back
  to A.

Ties prefer the original selected index and then the lowest candidate index.
Detection confidence never enters these scores. Cosines and ranking margins are
not calibrated confidence. No coefficients or thresholds are selected using GT.

D (`--geometry`) is an optional, separate `C + .15 * soft_geometry` arm; it is
disabled in the registered ABC/E trial. It requires real predicted reference
anchors and never hard-filters candidates or generates a search ROI.

## Separate selection, prediction and scoring

`scripts/try_crop_identity.py` has four independent commands:

1. `prepare115` projects the original fixed115 report through a prediction-only
   whitelist. It binds the original selection SHA and model asset provenance.
2. `prepare80` freezes the existing public selection protocol: `Random(42)`,
   ascending integer class, each pool sorted by string ID, sample4 per class and
   sort those sampled IDs. The output contains no class labels or gold fields.
3. `predict` accepts a no-gold manifest, existing model directory and public
   image root; it has no annotation argument. It also accepts the independently
   audited fixed115/fixed80 input protocols with exact manifest SHA checks. The
   independent cached115 format additionally requires its frozen model-provenance
   JSON. Cached115 needs only CLIP residency; public80 generates real candidates
   with the unchanged full-description NativeDetection runtime and shares CLIP.
4. `score` reads public validation GT only after a complete prediction run. It
   requires prediction and frozen-run SHA bindings plus the matching annotation
   SHA. Partial/interrupted runs have no completion marker and cannot be scored.

Each prediction records all original candidates, original phrase scores, crop
bounds, per-template target/reference cosines, contrast, baseline adjusted
scores, enabled-arm decisions and bounded failure reasons. A failed scorer is
not represented as a successful phrase-score result. `frozen_run.json` records
configuration, source/model/manifest hashes and registration UTC before the
first prediction; `completion.json` records completion UTC; metrics record the
later scoring-start UTC. `--deadline-utc` stops new work and prevents an
interrupted trial from being marked complete. Existing output files are never
overwritten.

## Evidence boundary

The old fixed115 full-description pool has245 candidates: five empty pools,
23 single-candidate samples and87 multiple-candidate samples. Its oracle@.5 is
78/115 versus the original selection62/115; identity reranking diagnoses the16
selection misses but cannot solve the37 samples without a correct candidate.
The public80 cohort must be newly predicted from its frozen IDs because its old
per-sample report is unavailable. Only five IDs overlap fixed115.

Both cohorts are public RRSIS-D development diagnostics with800-square images.
The115 cohort contains one reference per sample, no target absolute-position
hint and no building/lawn target coverage. These experiments do not establish
accuracy for large official images, grass/building targets or multi-reference
ABC instructions. Original detector training overlap and geographic isolation
remain unknown. New prediction results must report candidate oracle, selected
accuracy at.5/.7, meanIoU, gains/losses versus A and missing-candidate versus
selection failures. RefSeg mask candidates and complete spatial search remain
separate later experiments.

## 2026-10-13 completed public comparison

The full115 and full80 runs completed before either run read public GT. Both
used CPU4, seed0, crop batch8, original detector/CLIP assets, the frozen A/B/C/E
configuration and a 2026-10-13 07:00 Asia/Shanghai deadline. These are public
development results, not official scores or independent test accuracy.

| Cohort | Arm | Correct at .5 | Correct at .7 | Mean IoU | Gains / losses at .5 |
| --- | --- | ---: | ---: | ---: | ---: |
| 115 single-reference | A | 62 | 52 | .493064 | 0 / 0 |
| 115 single-reference | B | 49 | 44 | .407344 | 6 / 19 |
| 115 single-reference | C | 57 | 49 | .464568 | 7 / 12 |
| 115 single-reference | E | 64 | 53 | .513511 | 2 / 0 |
| 80 class-balanced | A | 62 | 57 | .695778 | 0 / 0 |
| 80 class-balanced | B | 53 | 50 | .594729 | 3 / 12 |
| 80 class-balanced | C | 54 | 51 | .605028 | 3 / 11 |
| 80 class-balanced | E | 61 | 56 | .683383 | 0 / 1 |

Candidate oracle at .5 remains78/115 and66/80: the115 pool has245 boxes and five
empty samples; the80 pool has120 boxes and two empty samples. All115 parse
successfully; only5/80 parse, the remaining75 retain the complete description.
The80 E loss occurs in that fallback group. This motivates a later, separately
registered role-capability guard; it does not justify replacing these original
results or claiming that crop reranking improved every cohort. Pure B/C are
not recommended and unrestricted E has not been connected to production.

Full runtime was179.981s for115 (69.698s model load) and180.560s for80 (68.672s
model load), including multiple comparison outputs and hashing. These values
are not isolated single-request production latency. Prediction SHA256 is
`9aa7c847f3351023f6bbb31d1db8c8597fd7807d1c840d4d6c8d51e23399d051` (115)
and `f79be7a3e9ee329f11d3428bcbda5c16a7c85947366d00688a4cd31258875dfd` (80).
Local artifacts are under `runs/grounding/crop_identity{115,80}_20261013`; they,
models and data are excluded from Git. Necessary unit tests13/13 passed;
independent numerical audits match complete-run selections, metrics and frozen
source/input/model bindings. No official run was created in this stage.

Final independent verification completed:115/80 scoring audits passed16,356/
9,618 checks; a776-check strata/baseline audit confirmed the115 A outputs are
exactly the prior full-description baseline, including all candidate coordinates,
indices and IoUs. Four input-fixed real CLIP spot checks encoded11 crops and
passed91 checks. Direct cosine maximum difference was4.47e-08; official v2,
factory eval and a manual equivalent RGB transform produced identical tensors.
All source/model hashes remained unchanged. The empty pool performed no encoding
and returned no invented box. This verifies implementation, not new accuracy.

## Minimal local contract check

```bash
python -B -m unittest -v tasks.detection.tests.test_crop_identity
python -B scripts/try_crop_identity.py --help
```

The13 tests cover RGB/bounds, one visual encoding per crop, identity contrast,
tie behavior, frozen-model and feature failures, fixed soft/hybrid formulas,
explicit original-selection fallback, no-gold schemas, seed42/class4 selection
and completion-bound post-hoc scoring. They do not measure visual accuracy.
