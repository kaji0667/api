# 固定候选阈值公开实验

这是在上一轮公开 dev 反馈后提出的结构迭代。它检验候选不足和候选身份选择的问题，不是未接触验证集的独立最终成绩，也没有实现左下角 ROI 或 ABC 参考物的前置空间搜索。生产 API 保持原默认阈值 `.25`；新增的显式参数只供本轮独立 CLI 使用。

## 固定协议

- CLI：`scripts/try_crop_candidate_thresholds.py`，协议 `public_fixed_candidate_threshold_crop_identity_v1`。
- SHA 锁定的公开 115 条与 seed42/class4 的 80 条，分组保留 195 个 membership，190 个唯一 ID。5 个跨组重叠 ID 不合并。
- 每个 membership 固定 `.25/.10/.05` 三档，共 585 次独立完整 detector forward。同一个检测器和冻结 CLIP 顺序复用，NMS 的原地 xywh 转换不会污染其他档；不缓存 crop、不拼接不同阈值候选池。
- NMS 的 IoU `.7`、`agnostic=True`、`multi_label=False`、`max_det=300`、上游 `max_nms=30000` 保持。置信度阈值满足 `0<t<1`，拒绝 bool、非数字和非有限值；诊断和 NMS 读取同一次局部快照。
- A 是该档原完整描述的 native 选择；E 原公式保持为 raw native phrase + 原 native absolute `.15` + `.15` × C 在该档候选池的 minmax。C 同分归零，E 失败显式回 A。
- G 仅按解析能力守卫：`parse_status == parsed` 复制 E，否则 `complete_description_fallback` 复制 A。G 不读取 ID、类别、GT 或评价结果。
- 所有三档、两组预测写完并校验源、模型、输入与图像 SHA 后，才产生全局 `completion.json`。native/crop 失败保留记录和分母；过程级异常或截止中断没有 completion。
- 截止 `2026-10-12T23:00:00Z`（北京时间 2026-10-13 07:00）。predict 不接受 GT 路径；score 在 585 行唯一性、固定分组、原 manifest SHA、哈希、G 与选框一致性、UTC 顺序等全部检查后，才访问公开 GT。
- `.25` 的 115 条还逐条精确核对旧原候选顺序、raw phrase 与 A 选择，任何差异终止预测而非默默标记成功。80 条由独立审计核对，不用于选框。

## 命令

在 WSL 工程根运行已有环境，不下载资产：

```bash
/mnt/d/code/python/HFSA/hfsa_env/bin/python -B scripts/try_crop_candidate_thresholds.py predict \
  --manifest115 /path/to/frozen115_inputs.json \
  --manifest80 /path/to/frozen80_selection.json \
  --models-dir /mnt/d/code/python/HFSA/提交/HFSA-main \
  --data-root /mnt/d/code/python/HFSA/HFSA-main/data/RRSIS-D/images/rrsisd/JPEGImages \
  --output /path/to/new_threshold_run --threads 4 --crop-batch-size 8
```

只有全局预测完成后才可运行：

```bash
/mnt/d/code/python/HFSA/hfsa_env/bin/python -B scripts/try_crop_candidate_thresholds.py score \
  --run-dir /path/to/new_threshold_run \
  --annotations /mnt/d/code/python/HFSA/HFSA-main/pre_datasets/RRSIS-D_refseg/val.jsonl \
  --output /path/to/new_threshold_metrics.json
```

输出每行包含 cohort/id/row_key/实际阈值、真实候选、raw phrase、原选择、CLIP crop 明细、A/E/G、pre-NMS 数、native/crop 耗时与 UTC。每档结果分别以 115 和 80 为分母，报告 IoU `.5/.7`、mean IoU、oracle、缺候选与选错、候选分布、300 cap、pre-NMS 300/30000、成本与相对同档 A、固定 `.25` A/E 的 gain/loss。oracle 仅用于预测完成后的诊断，不能用于答案。

理论总上限 175500 个 crop；两档新增阈值最多增加 117000 个。每图单档最多 300 个 crop，现有 scorer 暂存全部 crop tensor；若预处理为 `224×224 RGB float32`，约 172 MiB，另需 CLIP、检测器和 batch8 内存。实际候选数未知，不能据上一轮小候选池推断新一轮耗时。降低阈值还会改变原 native 竞争和 E 的池内 minmax，新增框可能帮助召回，也可能使选择退化。

三份上轮 Python 源保持 SHA 冻结；检测 encoder、backbone、head、vendor NMS、权重、VQA、正式请求、API/Qwen/ngrok 服务均未修改。

## 2026-10-13 完整公开开发结果

585 次预测在北京时间 04:23:18 全部完成，实际 195 个 membership、190 个唯一 ID、1219 个候选裁块，CPU4 总耗时 921.204 秒（包含加载、哈希等）。预测 SHA256 为 `81625bd7f3ae7f3bb8ef9f017d8301ba207ff9dd78563cc86fba45b10aa830c3`。完成后才读取公开标注。

| 分组 | 阈值 | A 正确@.5 | E 正确@.5 | G 正确@.5 | G 正确@.7 | G mean IoU | oracle@.5 | 候选总数 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 115 | .25 | 62 | 64 | 64 | 53 | .513511278 | 78 | 245 |
| 115 | .10 | 62 | 64 | 64 | 52 | .504082494 | 81 | 280 |
| 115 | .05 | 62 | 64 | 64 | 52 | .504932735 | 82 | 301 |
| 80 | .25 | 62 | 61 | 62 | 57 | .695777870 | 66 | 120 |
| 80 | .10 | 62 | 61 | 62 | 55 | .686585928 | 66 | 130 |
| 80 | .05 | 61 | 60 | 61 | 55 | .683715944 | 66 | 143 |

`.25` 的 G 在 115 组相对原 A 新增 2 条正确、损失 0，mean IoU 从 .493063885 升至 .513511278；80 组 G 与原 A 完全相同，避免无限制 E 的 1 条损失。各档 115 组全部 parsed；80 组 5 条 parsed、75 条明确回 A。G 是根据已观察的公开 dev 结果提出的能力守卫，不能将这些结果称作独立 test。

降低阈值在 115 组改善候选覆盖，但没有增加最终正确数，并降低 @.7 和 mean IoU；80 组 `.05` 还损失 1 条 @.5 正确。生产不启用低阈值。后续只考虑固定 `.25` 的 G 接入，仍需真实 API 等价验收；本提交中的生产入口没有启用 G。

14 个空池保留在全部分母中，571 行 crop scoring succeeded，无 native/crop 运行失败、无 300 候选上限截断。每条 native+crop 平均耗时分别为 115 组 `.25/.10/.05` 的 1549.92/1519.58/1577.24 ms、80 组的 1252.91/1224.19/1288.31 ms；这是独立 CPU 离线流水线成本，不是公网 API 延迟。

Native 专项 41/41、新协议与能力守卫 8/8、完整 API 回归 237/237 通过。独立全量无 GT 数值审计 98112 项通过，`.25` 全部 195 条原候选/原 A/raw phrase/absolute 分精确一致；低阈值保留全部原候选。完整预测后独立评分审计 98761 项通过，逐行 IoU、oracle、转移和六组聚合均与 CLI 相同。审计只验证这些实现和公开开发结果，不能据此保证官网提分或解决普通建筑/草坪候选不足。
