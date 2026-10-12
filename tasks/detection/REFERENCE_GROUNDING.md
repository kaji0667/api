# 参照物前置定位：公开离线试验

本试验实现“全图检测参照物 → 推导目标搜索区域 → 在局部视野检测目标”。代码位于定位任务层，不接入线上 API，也不修改原模型、权重、训练或原测试脚本。

## 调用关系

`reference_plan.py` 将明确的中英文描述解析为 `ReferencePlan`：目标原文短语、最多三个参照物及目标相对参照物的 left_of/right_of/above/below/near 关系，以及只属于目标的全图绝对位置。嵌套、否定、between、对角相对关系、未知子句和绑定歧义不进入本试验。参照物名称不依赖类别白名单；属性中的颜色 and 不拆成多个物体。

`reference_grounding.py::predict_reference_grounding(detector, image_bgr, plan)` 只依赖 NumPy 和注入的 `detector.predict` 接口。返回框、候选、固定选择/回退原因和调用顺序。任务模块不导入 API，CLI 使用现有 native detector 作为实际模型的注入实现。

1. 在全图分别调用参照物描述，沿用各次描述匹配的 `grounding_index`，记录全部候选。
2. 以参照框中心定义方向半平面并求搜索区域；near 使用参照框向外扩展图宽/高的 25%。目标明确绝对方位使用有重叠的粗区：左/上覆盖前 60%，右/下覆盖后 60%，中部为 20%–80%。这些是预先固定的试验几何约定，不是官方空间判定标准。
3. 搜索区域向外扩展图宽/高的 10%，最小宽高各为全图的 25%，裁剪后仍使用完整原描述调用原检测器。局部框经原 detector 恢复到裁图坐标，再加裁图原点，得到原图坐标。
4. 候选选取沿用该视野的描述匹配结果，不跨不同视野比较 phrase 分数或用 confidence 替代描述匹配。目标中心满足解析关系且不与参照框近重合（IoU≥0.8）才采用局部结果。
5. 全图完整描述目标预测固定最后执行，作为配对基线和回退。缺参照、区域冲突/退化、局部预测失败、关系不符或自匹配时使用全图结果。此回退不保证准确率不退化；试验必须记录新增正确与丢失正确。

## 公开对照

使用公开 RRSIS-D val 元数据，仅按支持的描述句法和 ID 排序固定前 12 条；选择阶段不读取 GT bbox/mask 或模型结果。先保存 `selection.json`，真正运行时检查预选文件完全一致。原 RRSIS train/val image ID 隔离另行核验；原检测器训练 split/cache 和地理隔离未知，因此结果属于开发诊断。

同一组样本比较四臂：当前完整描述全图 + 0.15 末端方位分、完整描述全图纯语义匹配、目标主体全图、参照物前置局部推理 + 完整描述全图回退。参照物和局部视野关闭旧末端方位分，避免将全图方位错用成裁图内方位。GT 仅在全部推理完成后计分；报告最终 Acc@0.5、IoU、所有候选的 oracle 覆盖，以及 gain/regression。

首次真实图像试验如果仅包含单参照，应明确报告；两/三参照合成调用与几何验证不能当成真实多参照视觉效果。

在工程根目录执行（现有 WSL 环境）：

```bash
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
../../hfsa_env/bin/python -B scripts/try_reference_grounding.py \
  --annotations ../../HFSA-main/pre_datasets/RRSIS-D_refseg/val.jsonl \
  --train-annotations ../../HFSA-main/pre_datasets/RRSIS-D_refseg/train.jsonl \
  --data-root ../../HFSA-main --models-dir . \
  --output runs/grounding/reference_first_roi_20261012 --limit 12 --selection-only
```

移除 `--selection-only` 后运行真实对照。已存在 `results.jsonl` 的试验目录不会被覆盖，重试使用新目录。生产 API、Qwen、ngrok 和模型凭据均不参与本命令。

合成测试：

```bash
../../hfsa_env/bin/python -B -m unittest discover -s tasks/detection/tests -p 'test_reference*.py'
```

当前局限包括解析覆盖有限、参照身份可能错误、关系几何只是近似、裁图可能丢失语境或截断目标。裁图仍编码完整原描述会改变图像与文字语境；目标主体对照单列，不能将其收益归于参照策略。没有有效公开结果前不启用线上流程。

## 2026-10-13 实际公开试跑

固定样本清单和完整报告位于本地 `runs/grounding/reference_first_roi_20261012/selection.json`、`report.json`、`results.jsonl`；运行产物不纳入源码备份。1740 条 val 中支持当前句法的 115 条均为单参照，本轮预选前 12 条，全程不按标注或结果换样本。两/三参照只通过合成流程测试，没有真实多参照视觉效果证据。

| 流程 | 正确数（IoU≥0.5） | 平均 IoU | 候选覆盖（IoU≥0.5） |
| --- | ---: | ---: | ---: |
| 当前完整描述全图、末端方位分 0.15 | 5/12 | 0.465102 | 6/12 |
| 完整描述全图、方位分 0 | 5/12 | 0.465102 | 6/12 |
| 主体短语全图、方位分 0 | 7/12 | 0.548223 | 7/12 |
| 参照前置、完整描述 ROI、全图回退 | 6/12 | 0.478625 | 7/12 |

参照前置相对当前流程新增 1 条正确、丢失 0 条正确（仅指 IoU≥0.5）；12 次参照和 12 次 ROI 调用，11 次采用 ROI、1 次因 ROI 无可用目标回退全图。新召回的 `val_10416` 全图无目标候选，裁图后跑道框 IoU=0.737407；主体全图同样正确，不能把这条收益解释为参照关系理解已解决。`val_10972` 已有正确候选却仍选错，说明目标描述匹配问题仍存在。

前四条叠图核验也暴露两种失败：`val_10209` 仍误框参照服务区，裁图改变框范围后近重合检查未阻止它；`val_10474` 裁剪截断立交桥，IoU 从 0.886639 降至 0.662364。在更严格 IoU≥0.7 下，当前与参照前置同为 5/12：新增 `val_10416`，丢失 `val_10474`。搜索关系成立不证明参照身份和目标身份正确。候选覆盖确有新增，但最终结果低于主体短语对照，暂不接入线上 API。

现有 CPU/4线程、torch seed=0，加载约 59.218 秒，总计约 106.850 秒（含加载、四臂推理、评分/写出与末尾哈希核验）；这些不是单次 API 延迟。检测/OpenCLIP及报告列出的来源文件前后 SHA256 一致，RRSIS train/val 所选 image ID 无交集；原检测训练重叠与地理隔离仍未知。32 项解析/几何合成测试通过。独立 `independent_audit.json` 为 passed：48 个最终 IoU、48 个候选 oracle 及汇总、预选、JSONL、ROI 几何/索引/调用顺序均一致，9 个源码与 2 个权重哈希匹配。原全图共 23 个候选，ROI 新增 14 个，联合候选 37 个。
