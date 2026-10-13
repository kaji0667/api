# 文本回答交付整理记录

2026-10-13 后续：队友已提供完整 v4 源码和重新训练的十二类残差头包，原缺源码事项已解决。当前统一使用 `runs/vqa/qformer/` 的 v4，VQA 短文本/单选/枚举共用同一实例；原五头包移出交付工程留作对照，不单独部署或恢复兼容。当前状态见 [v4合并记录](TEAM_V4_MERGE.md)；以下保留首次收到五头资产时的历史审计。

2026-10-13 收到 `run_res_mlp/run_res_mlp` 模型包。完整 16 文件统一整理到 `runs/vqa/text/`，与其它任务一样使用 `tasks/<task>/` 放源码、`scripts/` 放入口、`runs/<task>/` 放权重及实验记录。模型包文件保持原字节，15 项 manifest SHA256 全部通过；权重 SHA256 为 `5cfd6f7354c9a216f17287589d1eec237346fbedb4b2574e6f2d25f50303f0fe`。

该包声明 Q-Former v2、五类答案头（category/color/shape/existence/scene），不是开放式文本生成模型。交付完整公开 val 分母 37392，训练记录 18 轮、最佳第 14 轮；记录的五类宏平均 53.765321%，全 QA accuracy 36.128049%。这些是交付方的公开选模验证记录，本次没有运行推理或复测准确率，不能与十二类模型使用不同归一口径直接比较。

本次 `torch.load(..., weights_only=True)` 仅核对元数据及张量键/形状，没有模型前向。权重共 113 张量，含题型分类器 LayerNorm 和 `answers.<type>.mlp` / `classifier`，现有十二头源码为线性答案头且键集合不同。包内没有 `model.py`、`inference.py` 或其它源码；架构 ID 和 inference 配置未说明这些 MLP 层的完整计算顺序、激活与残差连接，因此不能从权重猜测实现，也不能只放宽 strict 加载或添加兼容回退。

当前 API 仍使用 `runs/vqa/qformer/` 已验收十二头包；`runs/vqa/text/` 是待配套源码验收的交付资产，尚未接入 short_text 或替换任何当前执行器。需要交付方对应 `model.py` 和 `inference.py` 后，再核对模型方程、严格加载及按答案格式接线；当前没有新增环境变量、双模型启动或自动回退。

VQA 数据准备、训练、评估与其他任务一样直接使用任务模块：

```bash
# 先设置 ANNOTATIONS_TRAIN、ANNOTATIONS_VAL、IMAGES_DIR。
OUT_DIR=data/vqa bash scripts/prepare_vqa.sh
TRAIN_JSONL=data/vqa/train.jsonl VAL_JSONL=data/vqa/val.jsonl \
  SAVE_DIR=runs/vqa/retrain bash scripts/train_vqa.sh --cache-dir .cache/vqa/retrain
BUNDLE_DIR=runs/vqa/retrain VAL_JSONL=data/vqa/val.jsonl \
  OUT_DIR=runs/vqa/retrain/evaluation bash scripts/test_vqa.sh --cache-dir .cache/vqa/retrain
python -m tasks.vqa.inference --help
```

`scripts/eval_vqa.sh` 改名为 `scripts/test_vqa.sh`，保留脚本内容和参数；根目录 `run_vqa.py` 的重复编排删除。数据准备仍需设置 `ANNOTATIONS_TRAIN`、`ANNOTATIONS_VAL`、`IMAGES_DIR`，训练与评估可用原环境变量和 CLI 参数指定数据、设备及目录。旧命令需改用上述入口；模型包封装单独执行 `python scripts/finalize_vqa_bundle.py --bundle-dir <directory>`，不隐式训练、评测或演示。

上述为重新训练流程的命令说明，本轮未执行。直接训练/评估脚本的旧默认仍使用外部 `*_final.jsonl`；准备新数据后须像示例一样显式传新 JSONL、缓存与输出目录，不能覆盖当前已验收模型包或误用旧缓存。

模型、验证逐题产物与训练配置原路径只留本地模型包，不提交源码 Git。2026-10-13 北京时间 07:00 截止后，本轮只进行文件哈希、元数据、CLI 帮助和脚本静态验证，不运行模型、API 监听或官网测评。

静态验收完成：移动后全部15项manifest文件SHA一致、16文件齐全；现有纯文件API完整训练收据门禁通过（full_split/37392/18轮/最佳14）。prepare/train/test三个Shell语法检查通过，data/train/evaluate/inference四个模块 `--help` 通过；test脚本与改名前工作树字节完全相同。独立入口审查未发现已删runner的真实调用依赖。以上均没有执行模型前向。

独立复核交付的全部37392条旧验证预测：ID唯一、无重复，存储的正确标记与status/answer/normalized_answer逐条一致；13509正确、22327有效，五类22413条中12853正确、macro53.7653213487%，与绑定指标吻合。没有执行新推理；未训练的其它14979题中656条偶然正确不能当新增七类能力。现有本地交接源码及相关ZIP静态搜索未找到对应MLP实现。
