# 目标查询与参照关系分离：公开离线候选试验

本轮复用原检测器和本地正式权重，先查询参照物、推导可选搜索窗口，再生成目标主体及属性的局部与全图候选。完整描述全图候选也保留。所有实际预测框在最后同一次主体全图 forward 的 phrase maps 中统一评分，避免混用不同视野的语义原分。未接入线上 API 或 Web。

## 模块与接口

- `reference_plan.py` 保守解析目标原文、参照原文、关系及目标全图方位；仍沿用上一版支持句法，不把目标方位自动附加给参照物。
- `candidate_grounding.py::predict_candidate_grounding(detector, image_bgr, plan)` 通过注入 detector 执行，不导入 API 或使用标注。固定顺序：参照全图 → 可用 ROI 主体 → 完整描述全图 → 主体全图。
- 现有 `NativeDetection.predict` 增加关闭默认的 `return_phrase_scores=False` 与 `score_boxes_xyxy=None`。`phrase_scores` 是本次 NMS 框的未加方位语义分；`extra_phrase_scores` 是外部实际预测框在同次 forward 中的分。原 `scores` 仍是检测 confidence。外部框不进入 NMS 或原 assignment，按真实每轴 resize/pad 转换，默认调用保持原行为。
- `scripts/try_candidate_grounding.py` 负责固定公开样本、实际模型注入和计分，GT 只在每样本全部预测与选择完成后访问。

参照可靠性使用同次 phrase 分 minmax 后的第一、第二名差；选中非原分最高者或并列时为0，单个有效候选约定为1。这不证明参照身份正确。任一参照缺失或不可靠时跳过 ROI，其它可靠参照仍可提供软关系分；区域冲突也跳过 ROI。

## 固定评分与候选保护

ROI 框先恢复原图坐标。接触内部裁边 ±2px 且与全图候选 IoU≥0.5 的局部片段被抑制；IoU≥0.8 的重复按主体全图、完整句全图、ROI 的优先级保留实际框，不拼接或合成 bbox。同源重复按原语义分和索引稳定选取。

只有可选候选参与同次主体全图语义分 minmax，相等为0。固定最终分：

```text
semantic_norm + 0.15 * geometry - 0.15 * self_match
```

左右/上下关系使用原图归一中心位移 `clip(0.5 + signed_displacement / axis_size, 0, 1)`；near 使用目标中心到参照 bbox 的距离除以图对角线的25%；绝对方位沿原连续轴位置分。geometry 分母为已解析参照数量加可用绝对位置约束数，参照贡献乘可靠性，缺失贡献为0。与有效参照 IoU≥0.8 时 self_match 为1，只作软惩罚。

无合法语义分时明确回退主体原描述匹配，再尝试完整句原匹配，不拿 confidence 猜身份。全部原 pool、抑制原因、分数来源、可选 pool、纯语义联合选择和最终选择均记录。若出现这种 fallback，可选 pool 的 oracle 不一定是 fallback 框的上界，应单独审计。

## 固定公开协议

RRSIS-D val 的1740条表达中，支持既有句法的115条全部按ID排序预选，不按GT或预测换样本。这批全为单参照，前12条与上一试验配对，新增103条单列。五臂为当前完整句加方位0.15、完整句纯语义、主体全图、联合候选纯语义、联合候选加独立关系。候选生成参数继续 cap1600/conf0.25/NMS0.7/max_det300/matching temperature0.7；试验内旧方位权重关闭。

在提交工程根目录运行（已有WSL环境，无下载）：

```bash
env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  ../../hfsa_env/bin/python -B scripts/try_candidate_grounding.py \
  --annotations ../../HFSA-main/pre_datasets/RRSIS-D_refseg/val.jsonl \
  --train-annotations ../../HFSA-main/pre_datasets/RRSIS-D_refseg/train.jsonl \
  --data-root ../../HFSA-main --models-dir . \
  --output runs/grounding/candidate_roles_20261013 --limit 115 --selection-only
```

移除 `--selection-only` 后执行真实预测。预选清单逐字核对；已有 `results.jsonl` 拒绝覆盖。CPU4线程、torch seed0，来源与检测/OpenCLIP文件前后SHA绑定。结果、权重、数据及环境不入源码Git。

合成测试：

```bash
../../hfsa_env/bin/python -B -m unittest discover -s tasks/detection/tests
../../hfsa_env/bin/python -B -m unittest discover -s participant_api_starter-main/tests
```

RRSIS-D 所选 train/val image ID 可检查无交集，但实际旧 RSVG-ALL 检测训练 split/cache 缺失，训练重叠与地理隔离未知。这是公开开发诊断，不能称独立 test 或官网提分。本地 RRSIS-D 与 VRSBench 元数据没有普通 building/lawn/grass 的目标 bbox，也没有长边>1600影像；不能把其它目标描述中的参照词对应GT挪作草坪/建筑标注。VRSBench val/test ID相同。多ABC仅合成流程可验证，真实视觉效果尚无相应样本证据。

## 2026-10-13 验证记录

定位任务合成测试50/50（新增候选18、参照流程17、解析15），0.209秒；native专项53/53（新增接口16、原检测37），4.238秒；完整API回归232/232，74.162秒，发现与冷导入合计111.882秒，无失败/错误/跳过。CLI帮助及三份新Python源码编译通过。接口和合成验证只证明契约，不代替真实视觉定位效果。完整本地收据在本轮输出目录的 `validation.json`。

全部115条真实公开试跑已完成：

| 流程 | 正确@0.5 | 正确@0.7 | 平均IoU | 候选覆盖@0.5 |
| --- | ---: | ---: | ---: | ---: |
| 当前完整句、旧方位0.15 | 62/115 | 52/115 | 0.493064 | 78/115 |
| 完整句纯语义 | 62/115 | 52/115 | 0.493064 | 78/115 |
| 主体全图 | 56/115 | 47/115 | 0.445804 | 76/115 |
| 联合候选纯语义 | 55/115 | 41/115 | 0.420320 | 82/115 |
| 联合候选及独立关系 | 55/115 | 41/115 | 0.425052 | 82/115 |

前12条current/subject/final为5/7/7，平均IoU .465102/.548223/.575414；新增103条为57/49/48，平均IoU .496322/.433872/.407534。扩大样本后主体与新策略都退化，不能沿用前12的收益说它们更好。最终相对current新增6、丢失13条正确（@0.5）；@0.7新增4、丢失15。相对主体@0.5新增1、丢失2；关系相对同一联合候选纯语义@0.5/@0.7均gain0/loss0，只小幅改变部分IoU。

630个原始pool候选抑制后307可选，107次ROI；all/active oracle均82/115，仍缺正确候选33条，已有正确可选框却最终选错27条。当前完整句对应缺37、选错16条。因此候选覆盖多4条并没有改善最终定位，目标角色及语义选择损失更大。同类vehicle指代保留完整句时可能更好，不能假定删去关系后主体身份仍唯一。

局部片段的同图语义ROI均值也可能高于完整目标：`val_10474`选中桥的裁短框，IoU从current0.886639/subject0.882609降到0.662364。ROI `[0,308,800,800]`，局部框top316.0835，距内部裁边8.0835px，未满足固定±2px规则；与主体全图桥框IoU0.631963又低于重复阈值0.8。纯语义联合已经选该局部框，关系进一步小幅加分，不能把这例仅归因方位权重。不依据GT修改阈值，失败证据保留。

CPU4线程/seed0，加载50.026秒、全部五臂及产物/hash总398.287秒，不能当单次线上延迟。报告列出的11源码/2权重前后SHA一致，预选清单逐字一致；全部五臂、候选、抑制和分项分数留在 `report.json`/`results.jsonl`。结论是**本轮不推荐接生产**，线上保持原策略；当前数据不能证明解决普通建筑、草坪或多ABC，也不是官网成绩。

独立 `independent_audit.json` 9132项检查通过、0错误：575最终IoU与575oracle、115全pool oracle、汇总/前12/新增103/transitions、同次分数映射/公式/抑制/坐标/调用顺序与13运行哈希均吻合独立基线。严格@0.7的all/active覆盖为69/66，抑制造成3条高精度候选损失（val_17430/5089/6415），不能称所有阈值无损。314重复、9裁边抑制，实际合法匹配fallback0；3例全部目标候选空，不应将粗略phrase_scores_unavailable标签解释为接口失败。

115条没有目标绝对方位，图均800²。92条参照可靠性为1，其中52单候选、40两候选；两分数minmax后的top-second天然1，不能据此声称参照确定。19条同类目标/参照表达中6条最终框与参照预测IoU≥0.8且GT失败，其中5条仍有正确active框；无独立参照GT，属于预测重叠诊断。两全图pool覆盖80条，ROI再补2条；ROI最终选中10次、5次正确。452次候选流水线调用、加current对照为567次，成本只记录全试验，不当线上性能承诺。
