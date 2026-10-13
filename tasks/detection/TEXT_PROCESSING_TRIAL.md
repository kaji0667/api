# Grounding text processing: public development trials (2026-10-13)

Production retains the original complete-description detector and `clip_guarded`
selection. `text_conditioning.py` and `text_consensus.py` are offline experiment
helpers; the API does not import them. No conditioning or visual-text ranking
arm below is enabled in production.

The fixed public inputs contain 115 relationship expressions and a separate
80-expression class-balanced cohort: 195 memberships, 190 unique IDs. They are
iterated public development data, with unknown detector training/geographic
overlap. Gold is read only after a complete, SHA-bound prediction receipt.

| Policy | Correct @.5 / 115 | Correct @.5 / 80 |
| --- | ---: | ---: |
| Existing guarded full description G | 64 | 62 |
| 75% full / 25% target, guarded new pool | 58 | 61 |
| 50% full / 50% target, guarded new pool | 56 | 60 |
| Both mixed conditions + target/contrast agreement, original pool | 63 | 62 |
| Remove explicit frame positions only from crop identity text | 64 | 61 |

All alternatives failed the no-regression requirement. Current thresholds,
models, full description, candidates and production ranking remain intact.
The visual-text helper also declines contradictory frame positions and keeps
unknown/nested/negated clauses; these contracts do not establish visual gains.

The accepted change is in `participant_api_starter-main/api_task_text.py`:
explicit bbox formatting such as `Return a bbox`, `Return the bounding box`,
`Please provide the bounding box` and Chinese output-coordinate directives is
removed at sentence/line boundaries. `Description：`, `目标描述：` and `描述:`
are supported payload markers. Object/reference text and internal newlines are
preserved, unknown instructions remain, multiple payloads fail explicitly.
Chinese payloads are not translated and encoder limits still apply.

`scripts/verify_bbox_text_wrapping.py` checked first-six frozen members in each
cohort, paired with six fixed formatting directives. All 12 real native/guarded
outputs for new wrapped versus bare text matched exactly, including prompt,
candidate order/boxes/confidences, phrase/adjusted scores and final index. Old
wrapped versus new wrapped correct @.5 was 7/12 -> 8/12, gain 1/loss 0; @.7
also 7 -> 8, mean IoU .5214230457 -> .6276771977. This is a small synthetic
protocol-format intervention on public imagery, not official/test accuracy or
evidence that ordinary buildings/lawns now have valid candidates.

Verification: API 253 tests; detection 110 tests. Trial scripts:

```bash
python -B scripts/try_grounding_text.py predict --help
python -B scripts/check_text_consensus.py --help
python -B scripts/try_visual_text.py --help
python -B scripts/verify_bbox_text_wrapping.py --help
python -B scripts/try_grounding_text.py score --help
```

Local receipts are under `runs/grounding/{text_conditioning195_20261013,
text_consensus195_20261013,visual_text195_20261013,bbox_wrap12_20261013}`.
Two trial source snapshots match their registered SHA values. The initial score
call had an outward-box helper signature mismatch; prediction was preserved,
the score-only conversion was fixed and its source SHA is in metrics.
