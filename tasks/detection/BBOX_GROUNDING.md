# Bbox grounding task executor

`BBoxGroundingExecutor` owns the optional production selection policy. The API
factory injects the existing native detector, description extractor, image
conversion function and shared frozen `FrozenCLIPCropScorer`. The task does not
import API modules, construct another model or modify detector internals.

`HFSA_GROUNDING_POLICY=native` is the default and retains the original bbox path.
`clip_guarded` requires the detection backend, native candidate confidence `.25`
and spatial weight `.15`. The factory immediately validates the scorer against
the already loaded frozen/eval CLIP and its official preprocessing; invalid
configuration or scorer initialization fails startup. There is no new threshold
configuration or request override. Presence continues to use native detection.

The task first extracts the raw description and parses target, reference clauses
and absolute position before native inference. It preserves interior newlines:
for example, `The lower-left lawn\nA road is to the right of it` is a two-clause
plan. Sanitizing these newlines before parsing changes the grammar. The native
runtime still receives the complete original question and prepares its original
full-description condition itself. Unsupported plans use its original call and
skip CLIP crop scoring.

For parsed plans, one original full-description native forward supplies every
candidate, its raw phrase score, the original selected index A and the same
float32 scores after the original absolute-position bonus and before assignment.
The frozen `rank_full_candidates(..., hybrid=True)` selects E (guarded G=E):
original phrase/position score + `.15` times the candidate-pool minmax of the
target-minus-reference CLIP contrast. A contrast tie adds zero. The exact
candidate order, coordinates and confidence scores are preserved; no search ROI,
reference detector forward, new box, mask conversion or threshold lowering is
introduced. Parsing early builds the plan; final scoring still compares boxes.

Missing/incomplete crop scores or optional ranking errors explicitly keep A from
the same forward. Native inference errors propagate. Empty native candidates
remain empty and fail the existing bbox answer contract. Diagnostics contain
only fixed policy/parse/outcome/reason values, counts and elapsed time; they do
not contain description text, boxes, crop scores or raw exception messages.
The server's existing `BoundedSemaphore(1)` continues to serialize inference
because the shared native phrase maps and encoder cache are mutable.

The `.25` choice follows the separate frozen candidate-threshold experiment;
lower thresholds did not improve final @.5 and worsened @.7/mIoU. This integration
does not claim a new accuracy result or an online evaluation improvement.

The separate public development comparison at `.25` gave G/A correct counts
`64/62` on 115 expressions and `62/62` on 80 expressions (195 memberships,
190 unique IDs). This limited result does not establish independent test or
official accuracy. Supplementary RefSegRS validation used five original
building/low-vegetation expressions from four image SHA groups: detector final
and oracle @.5 were `0/5` at all three thresholds; the existing RefSeg model
also had bbox @.5 `0/5`. Missing candidates remain unresolved. These samples
do not establish lawn/grass, multiple-ABC or large-image performance, and the
RefSeg branch is not enabled by this policy.

Synthetic task checks:

```bash
python -B -m unittest tasks.detection.tests.test_bbox_grounding -v
```
