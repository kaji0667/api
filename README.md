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

将网盘下载的 `HFSA_models.zip` 直接解压到 `HFSA-main/`。解压后应存在：

```text
pretrain_model/yolov12m.pt
runs/classification/vrsbench_scene/weights/best.pt
runs/detect/DIOR-RSVG-ViT-L-14/weights/best.pt
runs/counting/counting-VRSBench-ViT-L-14/weights/best.pt
runs/semseg/srp_yolov12m_axis/weights/best_raw.pt
openclip/ViT-L-14.pt
```

## 3. 一键启动

```bash
bash run_web.sh
```

浏览器访问 `http://服务器IP:7860`。

Web 单图推理不需要完整数据集。VRSBench、DIOR-RSVG 和 RRSIS-D 原始数据集不随作品提交；只有重新训练或整集评测时才需要按技术报告中的数据处理说明准备数据。

## 4. 官方本地评测 API

`participant_api_starter-main/` 已接入分类、检测、计数、旧版独立VQA及显式指代分割转bbox。它使用同级 `HFSA_models/` 权重目录（可通过 `HFSA_MODELS_DIR` 修改），VQA bundle位于工程内`runs/vqa/stage1/`；与上述 Web 的权重解压位置分别按各自入口解析。

启动命令、支持范围和验证方法见 [官方 API 说明](participant_api_starter-main/README.md)。`MODEL_MODE=real` 使用独立任务入口，按原测试的模型、文本、预处理及描述匹配执行，不调用Web Predictor。默认加载分类/检测/计数/旧VQA，显式RefSeg才增加分割；终端显示 `API listening` 后才接题。VQA默认CPU以留出8GB显存，可通过`HFSA_VQA_DEVICE`修改。保持既有环境、9001、Key及Ngrok，实际运行状态和用户官网成绩见项目CURRENT_STATE；接口连通与准确率分别记录。

用户接受目标计数条件降级：带区域或关系限定的问题也按全图同类数量返回，不把全图数量称为正确区域计数。定位描述仍完整保留；代码修改后需重启API，已有Ngrok无需重新建立。

检测大图部署使用原测试可配置的输入尺寸和rect处理：`HFSA_DETECTION_MAX_IMGSZ` 默认1600，小图保留800；设800可恢复原默认尺寸。正式权重本机CPU重放的两条定位请求已返回合法框，并与同尺寸原test推理/描述匹配/坐标恢复一致；未验证官网IoU或新GPU服务状态。

## 5. 独立视觉问答第一阶段

当前接入队友十二类 Q-Former v2，使用 `runs/vqa/qformer/`，模型、归一与词表一起验收，不回退旧五头包。交付公开val复算53.797604%，本机验收和限制见 [合并记录](tasks/vqa/TEAM_V3_MERGE.md)。以下为旧五头历史记录，不代表当前支持范围。

问答代码位于 `tasks/vqa/`，主模型权重、配置、真实验证结果及公开样例位于 `runs/vqa/stage1/`；独立仅问题/仅图像基线分别位于 `runs/vqa/question_only/`、`runs/vqa/image_only/`。支持类别、颜色、形状、存在和场景五类答案；旧版已接入官方API的vqa分支，Web保持原状。

从本工程根目录运行 `python scripts/verify_vqa_examples.py --device cuda:0` 可核验20条实际样例输出。默认使用同级 `HFSA_models/` 的团队 YOLO/OpenCLIP 权重；可通过 CLI 指定其它资产路径。训练、评估及 `VQAPredictor` 用法见 [VQA 说明](tasks/vqa/README_INTEGRATION.txt)。本机沿用已有 `hfsa_env`，无需为此模块另建环境。

VQA 已新增从 [BLIP-2 官方源码](https://github.com/salesforce/LAVIS/blob/05cb670f47296522e24e7727e366b4a997d28582/lavis/models/blip2_models/Qformer.py) 迁入的 Q-Former 核心。`bash scripts/train_vqa.sh` 现在默认使用 `config/vqa/qformer.json` 训练该结构，输出到 `runs/vqa/qformer/`；旧模型仍在 `runs/vqa/stage1/`。这是轻量任务侧适配，需要重新训练验证，未引入完整 BLIP-2 大模型权重。许可、源码出处和差异说明位于 `tasks/vqa/third_party/lavis/`。
