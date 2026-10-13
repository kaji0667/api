# Full-object crop and complete-description matching trial

Both arms were rejected. The API continues its existing `clip_guarded` policy;
it does not import `whole_crop_trial.py`. The previous bbox-format cleanup did
not improve the subsequent development smoke's grounding score.

This 2026-10-13 trial uses the unchanged .25 native candidate pool, native
phrase/absolute scores and .15 crop correction, on the existing 115+80 public
memberships (190 unique IDs). Target/reference cues follow the existing parser.
`pad_guarded` preserves unsupported descriptions' original G selection.
`pad_full` instead scores their complete source text without inventing roles or
dropping attributes, relationships or negation. Candidate crops are centered
on a square filled with rounded CLIP mean RGB, before the original transform.
The encoder, model, thresholds and candidate coordinates remain unchanged.

| Arm | Correct @.5 / 115 | Gain / loss | Correct @.5 / 80 | Gain / loss |
| --- | ---: | ---: | ---: | ---: |
| Existing G | 64 | 0 / 0 | 62 | 0 / 0 |
| pad_guarded | 62 | 0 / 2 | 62 | 0 / 0 |
| pad_full | 62 | 0 / 2 | 61 | 0 / 1 |

@.7: G 53/115,57/80; pad_guarded 51/115,57/80; pad_full 51/115,56/80.
Mean IoU: G .5135112781/.6957778696, pad_guarded .4955566198/.6957778696,
pad_full .4955566198/.6840612822. Neither arm added a correct example.

All 195 predictions finished before a separate SHA-bound public annotation
score call. The inputs, images, original candidate receipt, CLIP and source
snapshots are frozen in `runs/grounding/whole_crop195_20261013`.
Prediction SHA: `122c2b51685dd36a7d0712bdcc667f791959ea78d1d62fc771b1afad5a53ee78`;
elapsed 85.5789 seconds. These are iterated public development results, with
unknown training/geographic overlap, not independent test or official gains.
Crop inference used GPU against a historical CPU baseline; no separate device
ablation was run, so the change cannot be attributed exclusively to padding.
The paired no-gain regression is sufficient to reject production adoption.

After prediction, an incomplete-evidence guard was added only to the offline
helper and tested. Registered source snapshots and predictions are preserved;
this guard does not change the completed run's fully valid nonempty records.

Five helper tests plus 15 existing production bbox contract tests passed.
The complete detection suite also passed all 115 tests.
The independent local audit recomputes cohort metrics, candidate membership,
full denominators and hashes. No API service restart, official run, production
model or VQA change was performed by this trial.

Reproduce prediction using the SHA-bound manifests and original complete
candidate receipt:

```bash
python -B scripts/try_whole_crop.py predict --help
python -B scripts/try_whole_crop.py score --help
python -m unittest tasks.detection.tests.test_whole_crop_trial
```

The failed experiment remains offline. Blindly routing unsupported grammar
through crop matching is not an established localization fix.
