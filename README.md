# HFSA 运行说明

## 1. 安装环境

```bash
conda create -n hfsa python=3.11 -y
conda activate hfsa
cd HFSA-main
python -m pip install -r requirements.txt \
  --extra-index-url https://download.pytorch.org/whl/cu121
```

`wheels/` 中已包含 `en_core_web_sm` 本地安装文件。

## 2. 放置模型

正式权重现已复制到 `HFSA-main/` 内。API 与 Web 默认从工程内部读取；部署到其他机器时，将模型文件按以下位置一起携带：

```text
pretrain_model/yolov12m.pt
runs/classification/vrsbench_scene/weights/best.pt
runs/detect/DIOR-RSVG-ViT-L-14/weights/best.pt
runs/counting/counting-VRSBench-ViT-L-14/weights/best.pt
runs/semseg/srp_yolov12m_axis/weights/best_raw.pt
openclip/ViT-L-14.pt
runs/vqa/qformer/weights/vqa_head.pt
```

VQA 必须保留整个 `runs/vqa/qformer/` 正式模型包，包括 `config/`、`manifest.json` 和绑定的训练/评估记录。根目录 `config/vqa/qformer.json` 是训练参数，不替代模型包配置。权重和模型产物只保存在本地，不纳入源码 Git 备份。

各任务统一使用 `tasks/<task>/` 放实现，`scripts/train_<task>.sh` / `scripts/test_<task>.sh` 放训练与评估入口，`runs/<task>/` 放权重和实验记录。VQA 使用 `scripts/prepare_vqa.sh`、`scripts/train_vqa.sh`、`scripts/test_vqa.sh`，不再保留根目录 `run_vqa.py`。原 `scripts/eval_vqa.sh` 命令改用 `scripts/test_vqa.sh`，参数不变。

当前 VQA 为十二头残差答案头 v4（2026-10-14 冻结），配套源码、词表及完整模型包位于 `tasks/vqa` 和 `runs/vqa/qformer/`。head SHA-256：`2858c240b620915b9958ee3fa3f63f0bc01bc5c0c151b75f247852c2f551bf8b`。原五头版完整备份保留，其他版本不参与自动回退。

## 3. 一键启动

```bash
bash run_web.sh
```

浏览器访问 `http://服务器IP:7860`。

Web 单图推理不需要完整数据集。VRSBench、DIOR-RSVG 和 RRSIS-D 原始数据集不随作品提交；只有重新训练或整集评测时才需要按技术报告中的数据处理说明准备数据。

## 4. 官方本地评测 API

`participant_api_starter-main/` 已接入分类、检测、计数、正式 Q-Former VQA 及显式指代分割转bbox。默认权重根目录为 `HFSA-main/` 自身，可通过 `HFSA_MODELS_DIR` 显式覆盖；VQA bundle位于工程内 `runs/vqa/qformer/`。检测/计数使用工程内 `openclip/ViT-L-14.pt`，不依赖此前外部模型目录；双时相变化已取消。

启动命令、支持范围和验证方法见 [官方 API 说明](participant_api_starter-main/README.md)。`MODEL_MODE=real` 使用独立任务入口，按原测试的模型、文本、预处理及描述匹配执行，不调用Web Predictor。默认加载分类/检测/计数/Q-Former VQA，显式RefSeg才增加分割；终端显示 `API listening` 后才接题。VQA默认CPU以留出8GB显存，可通过`HFSA_VQA_DEVICE`修改。保持既有环境、9001、Key及Ngrok，实际运行状态和用户官网成绩见项目CURRENT_STATE；接口连通与准确率分别记录。

用户接受目标计数条件降级：带区域或关系限定的问题也按全图同类数量返回，不把全图数量称为正确区域计数。定位描述仍完整保留；代码修改后需重启API，已有Ngrok无需重新建立。

检测大图部署使用原测试可配置的输入尺寸和rect处理：`HFSA_DETECTION_MAX_IMGSZ` 默认1600，小图保留800；设800可恢复原默认尺寸。正式权重本机CPU重放的两条定位请求已返回合法框，并与同尺寸原test推理/描述匹配/坐标恢复一致；未验证官网IoU或新GPU服务状态。

## 5. 独立视觉问答第一阶段（历史记录）

当前 API 使用完整训练的十二头 Q-Former v4；下文 stage1 第一阶段实验仅作历史说明。当前实现和命令见 [VQA 模块说明](tasks/vqa/README_INTEGRATION.txt)。

以下保留旧第一阶段实验说明；当前 API 仅接受完成完整训练/验证的 Q-Former v2，使用 `runs/vqa/qformer/`，不回退旧架构。

问答代码位于 `tasks/vqa/`，主模型权重、配置、真实验证结果及公开样例位于 `runs/vqa/stage1/`；独立仅问题/仅图像基线分别位于 `runs/vqa/question_only/`、`runs/vqa/image_only/`。支持类别、颜色、形状、存在和场景五类答案；旧版已接入官方API的vqa分支，Web保持原状。

从本工程根目录运行 `python scripts/verify_vqa_examples.py --device cuda:0` 可核验20条实际样例输出。默认使用同级 `HFSA_models/` 的团队 YOLO/OpenCLIP 权重；可通过 CLI 指定其它资产路径。训练、评估及 `VQAPredictor` 用法见 [VQA 说明](tasks/vqa/README_INTEGRATION.txt)。本机沿用已有 `hfsa_env`，无需为此模块另建环境。

VQA 已新增从 [BLIP-2 官方源码](https://github.com/salesforce/LAVIS/blob/05cb670f47296522e24e7727e366b4a997d28582/lavis/models/blip2_models/Qformer.py) 迁入的 Q-Former 核心。`bash scripts/train_vqa.sh` 现在默认使用 `config/vqa/qformer.json` 训练该结构，输出到 `runs/vqa/qformer/`；旧模型仍在 `runs/vqa/stage1/`。这是轻量任务侧适配，需要重新训练验证，未引入完整 BLIP-2 大模型权重。许可、源码出处和差异说明位于 `tasks/vqa/third_party/lavis/`。


2026-10-14 正式部署：启动脚本默认 `HFSA_API_PACKAGE_DIR=../../rsu-formal800-v3-images`（相对本工程），可显式覆盖。正式根目录manifest含1000资产，dataset_id为rsu-formal800-v3；正式图片仅做完整性/读取检查，未用于选模。三版旧开发请求比较和登记填写内容在上级提交目录TXT中；正式测试须完成负责人签署登记及组委会核验。
