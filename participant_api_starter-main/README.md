# 参赛队伍最小模型 API 示例（协议 2.0）

## 2026-10-12 正式权重位于工程内部（当前默认路径）

用户要求 `HFSA-main` 内权重齐全，分类/计数/检测/RefSeg 的正式权重及 YOLOv12m、OpenCLIP 基础权重已按原相对目录复制到工程内，6份文件逐项 SHA-256 与原文件相同。正式 VQA 模型包继续保留在 `runs/vqa/qformer/`，其配置、manifest 与验收文件不修改。

API 和启动脚本默认 `HFSA_MODELS_DIR=HFSA-main` 本身，检测/计数直接加载工程内 `openclip/ViT-L-14.pt`；显式 RefSeg 复用该同版本文本编码器。仍可设置 `HFSA_MODELS_DIR` 使用其他权重根目录。VQA 训练配置和评估/核验脚本的基础权重默认路径同步为 `pretrain_model/yolov12m.pt` 与 `openclip/ViT-L-14.pt`。

如果原 API 终端曾显式 export 外部 `HFSA_MODELS_DIR`，下次启动前执行 `unset HFSA_MODELS_DIR` 即可采用新默认；沿用原 Key 执行 `bash scripts/serve_hfsa_api.sh`。此次没有重启 API/Ollama/ngrok。只复制权重；环境和评测图片包仍按原路径提供，不将本目录称为含数据和环境的完整部署包。权重不提交到 GitHub 源码备份。下文旧外部路径说明仅作历史，以本节为准。

## 2026-10-12 取消双时相变化任务（当前状态）

用户已取消双时相任务。`tasks/change` 的推理源码、MCI/ChangeFormer 模型接线及专用下载、审计、验证脚本和测试已移除。API 启动不再读取变化模型包或相关环境变量；历史数据、权重、验证报告仍保存在 `runs/change/`。

当前任务标签为 `classification`、`presence`、`counting`、`visual_grounding`、`vqa`，仅执行单图请求。双图在任务路由和模型调用前以 `unsupported_image_count` 拒绝，经协议外壳返回 HTTP 500 `inference_failed`。单图 bbox 继续直接选择 `visual_grounding`，其余请求由 Qwen 仅选择 task；完整原题和现有单图执行器保持既有行为。

默认私有诊断文件为 `${XDG_DATA_HOME:-$HOME/.local/share}/hfsa/prediction-diagnostics.jsonl`；可用 `HFSA_API_DIAGNOSTIC_FILE` 配置，显式空值关闭。每次记录UTC/PID/task/实际executor/固定reason/耗时及安全标识/既有数值诊断，不记录题干/选项/答案/坐标/头信息/凭据/异常文本；文件写入失败不影响预测，新POSIX文件0600。修复前评测缺少真实诊断，不把旧错误码当本轮根因。

运行中的9001不会自动更新，本次没有启停 API、Ollama 或 ngrok。用户在原 API 终端沿用已经 export 的 Key，Ctrl+C 后执行以下命令才会载入删除后的代码：

```bash
cd /mnt/d/code/python/HFSA/提交/HFSA-main
bash scripts/serve_hfsa_api.sh
```

等待启动输出 `API listening` 后再自行确认 ready。更早的变化能力决定见根目录 ADR-0046/0047，现已由 ADR-0048 撤销；下文带日期的旧实现记录仅作历史，以本节为准。

## 定位专项公开诊断与坐标修复（2026-10-11）

`native_detection.py` 现在先移除 rect padding，再用实际 `resized_w/original_w` 和 `resized_h/original_h` 分别恢复 x/y。ceil resize 在非整比图像上产生不同的两轴比例；原 `scale_boxes` 统一用高度比例，极长图可能明显偏移。原检测 test/vendor 不改，模型、文本、输入、阈值、NMS 和 phrase-map 选择不改。`native_parity.py` 保留原恢复结果，另验独立几何真值，并报告旧恢复差异；不能再把原 scalar-gain 输出当几何真值。API 仍裁边并向外取整。见根目录 ADR-0043。

新增 `benchmark_grounding.py`，离线公共 RRSIS-D val 成对诊断：seed42、20类各4条、80条；来源/文件哈希审计、Detection/RefSeg、候选 oracle 召回、实际选框、annotation bbox 与 mask bbox 两种口径、4条原 tensor 数值对照及实际 Adapter 输出映射。GT 只参与计分，不参与候选生成/选择。缺失按 IoU0 纳入全部80条；oracle 与实际答案使用相同向外取整。这里固定任务，不启动 HTTP 或调用 Qwen。

在 WSL 工程根目录使用既有环境运行：

```bash
cd /mnt/d/code/python/HFSA/提交/HFSA-main
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  /mnt/d/code/python/HFSA/hfsa_env/bin/python -B -u \
  participant_api_starter-main/benchmark_grounding.py \
  --annotations /mnt/d/code/python/HFSA/HFSA-main/pre_datasets/RRSIS-D_refseg/val.jsonl \
  --train-annotations /mnt/d/code/python/HFSA/HFSA-main/pre_datasets/RRSIS-D_refseg/train.jsonl \
  --data-root /mnt/d/code/python/HFSA/HFSA-main \
  --models-dir /mnt/d/code/python/HFSA/提交/HFSA_models \
  --output runs/grounding/diagnostic_20261011/paired80.json \
  --per-class 4 --seed 42 --threads 4 --parity-samples 4 --overlays 8
```

结果（CPU/4线程、旧正式权重；本80条均800×800）：

| 后端 | 有效框 | Acc@0.5 | Acc@0.7 | 平均 bbox IoU |
| --- | --- | --- | --- | --- |
| Detection | 78/80 | 62/80（77.5%） | 57/80（71.25%） | 0.695778 |
| RefSeg | 76/80 | 53/80（66.25%） | 44/80（55%） | 0.613959 |

Detection 候选 recall@0.5=66/80（82.5%），14条没有合格候选、4条有合格候选但选错。RefSeg 原图 mask mIoU=0.641280。4条最终权重对照的输入/raw/text/maps/分数/恢复框误差均0，RefSeg probability误差0、阈值mask完全一致。162项API回归及修订后5项指标检查通过，非整比/长图坐标另有独立往返测试。本轮保留默认 detection、显式 refseg；未重启现有服务。

这不是整集或官网分数：RefSeg val 已用于选模，Detection 权重实际训练于 RSVG-ALL（epoch92/imgsz640），其原划分/cache本机缺失，不能确认本小集对Detection独立。80样本与RRSIS训练集无相同ID或字节相同JPEG，地理/重编码近重复未知。主指标用XML证实的annotation xyxy；annotation与mask外接框平均IoU0.802244，7/80低于0.5，二者不得混用。坐标修复不会改变本80张等比例输入的结果，不声称由此提分。

证据直接在 `runs/grounding/diagnostic_20261011/`：`paired80.json`、`public_data_audit.json`、`integrity.json`、测试/推理日志和8张公共overlay。后续区域候选/空间重排实验已按用户要求取消并回退；本轮仅完整训练和接入当前Q-Former VQA。正式 API 下次启动才加载本次坐标修复；服务状态以根目录CURRENT_STATE为准。

## 当前 Q-Former VQA 接入与本机启动（2026-10-11）

`MODEL_MODE=real` 的 `vqa` 分支当前默认引用 `runs/vqa/qformer/weights/vqa_head.pt`，只接受架构 `hfsa_frozen_qformer_vqa_v2`、融合 `blip2_qformer` 的正式产物。独立 `tasks.vqa.inference.VQAPredictor` 接收图片、完整原题和原选项，构造一次后连续复用。`classification` 继续使用原21类场景分类。本节覆盖下文历史 VQA 拒绝或复用分类的记录。

正式配置使用清洗后的85804条训练QA与37392条验证QA，seed42、batch128、最多15 epochs、patience4。冻结 YOLOv12m/CLIP，只训练新增 Q-Former Head 的2,864,405个参数；32 queries、2层、4头、192维。BLIP-2/LAVIS 核心来源与能力边界见 [VQA 模块说明](../tasks/vqa/README_INTEGRATION.txt)，没有导入其预训练 Q-Former/BERT/LLM 权重。

启动门禁要求 `manifest.evaluation_status="complete"`，以及经过 manifest SHA-256 校验的 `training_result.json`、`config/train.json` 和 `evaluation/metrics.json`。训练必须 `validation_scope="full_split"`，训练/验证 batch 限制均为0，config 融合为 `blip2_qformer`，epoch 与 best_epoch 为有效整数；验证数量与 metrics.n 一致。checkpoint/dataset 哈希须为非空64位小写十六进制且与全量评估绑定；五类宏平均须有限、与训练选模结果一致。未完成、有限 batch、旧架构或收据不一致的目录在模型构造前拒绝。

VQA 支持 category/color/shape/existence/scene 五类有限词表；short_text 返回模型答案，single_choice 返回原选项键，enum 返回原允许值。只将 constraint.values 声明的单选项传给模型；部分词表外候选仍允许比较已训练且可唯一映射的候选，未知候选不能选中，本地结果记录 choice_coverage。全部候选未知、已知别名重复、枚举含未知值、中文问题及未实现的计数/位置/比较/双图等明确失败。原题逐字传入，无目标提取、翻译或限定词删除；无 ROI、切片或定位证据，不代表完整官网问答能力。

`HFSA_VQA_BUNDLE_DIR` 默认工程内 `runs/vqa/qformer`；`HFSA_VQA_DEVICE` 默认 `cpu`，原任务设备由 `HFSA_API_DEVICE` 控制。VQA 使用权重根目录（默认工程自身）内 `pretrain_model/yolov12m.pt` 和 `openclip/ViT-L-14.pt`，启动严格核对配置、词表、state_dict、模型和外部资产哈希。

本轮正式15轮训练完成、最佳14，全部37392 QA复评通过并封包complete：五类宏平均53.730459%、五类22413 QA总体56.837550%、全QA35.643988%、有效返回58.039153%。同分母旧模型宏平均54.378791%、全QA36.379439%，公开val未提升；该val用于选模，不是独立test或官网成绩。用户已于2026-10-11 18:48重启当前工程API，PID399620监听9001；18:57用户官网smoke的6次HTTP200确认本次认证/公网连通和预测工作，当前入口仅接完整验证的Q-Former v2。部署/小样本分数解释见根目录CURRENT_STATE，原验收报告为训练验收时快照；无独立内存SHA接口，助手未读取进程Key或终端输出。

本机 Ollama 权重在 Windows `D:\Ollama\models`。若服务未启动，在 WSL 终端执行：

```bash
OLLAMA_MODELS=/mnt/d/Ollama/models ollama serve
```

在另一个 WSL 终端打开已有模型（API也会预热路由，交互终端可用 `/bye` 退出）：

```bash
ollama run qwen3:1.7b --keepalive 24h
```

正式产物通过验收后，在原 API 所属终端按 Ctrl+C，再从工程根目录执行：

```bash
cd /mnt/d/code/python/HFSA/提交/HFSA-main
bash scripts/serve_hfsa_api.sh
```

脚本复用该终端已 export 的原 `MODEL_API_KEY`，不输出或保存 Key；强制 `MODEL_MODE=real` 和正式 Q-Former 路径，设备、后端等保留已有环境值或工程默认值，监听 `127.0.0.1:9001`。若使用新终端，先手动输入原已保存、与官网配置一致的 Key：

```bash
cd /mnt/d/code/python/HFSA/提交/HFSA-main
read -rs -p 'MODEL_API_KEY: ' MODEL_API_KEY
printf '\n'
export MODEL_API_KEY
bash scripts/serve_hfsa_api.sh
```

等 `API listening` 后使用同 Key 验证 `/healthz` 为 ready。继续复用现有 ngrok agent，上游 `http://127.0.0.1:9001`；origin 以4040 agent API的实时值为准。API未就绪时隧道不能提供模型服务。

验收使用既有 WSL 环境，从工程根目录执行：

```bash
source /mnt/d/code/python/HFSA/hfsa_env/bin/activate
python -B scripts/verify_vqa_qformer.py \
  --bundle-dir runs/vqa/qformer \
  --jsonl ../vqa_work_20261011/data/val_final.jsonl \
  --cache-dir ../vqa_work_20261011/cache/full --device cpu \
  --out runs/vqa/qformer/reports/online_verification.json
python -B scripts/verify_vqa_api.py \
  --bundle-dir runs/vqa/qformer --device cpu \
  --output runs/vqa/qformer/reports/api_integration_verification.json
```

第一项核验当前权重的在线/缓存样本一致性、三种格式和冻结状态，完整准确率由37392条全量评估另行报告。第二项只读取 `runs/vqa/stage1/examples` 的既有公共图片和问题，当前模型直接输出作为 HTTP 一致性参考；不读取旧 expected_predictions 或金标。固定 VQA 任务组走真实 adapter 执行路径，真实 Qwen 组另验任务路由，路由到非 VQA 的样例记录为 skipped。临时服务只监听随机本机端口、使用随机测试认证值、关闭请求捕获；检查五类输出、三种格式、鉴权、图片哈希、不支持和双图拒绝、对象复用及冻结状态，不改变9001服务、不登记官网。`--skip-live-qwen` 会明确注明省略真实路由组，不能据此宣称完成路由验收。

验收已完成：CPU在线/缓存7组一致；固定任务真实权重HTTP21组（16成功/5预期失败）；未patch Qwen HTTP11组（7成功/4预期失败，另10非VQA路由skip）。三形式、五头输出、鉴权、坏哈希、双图、不支持、冻结与复用等13项检查全部通过、errors为空。177项完整API回归、22项VQA测试和15项封包测试通过。正式权重SHA256 `8f6ed01a42a22c7f82ff410ba4a5e6a8e56da79274b05d0eb4896408fc8cc706`；报告在 `runs/vqa/qformer/reports/` 与 `evaluation/`。接口一致性不代表模型准确率或官网分数。

## HFSA 原生任务接入

当前 `MODEL_MODE=real` 的 `hfsa_adapter.py` 使用官方 API 自有 `native_classification.py`、`native_detection.py`、`native_counting.py` 与显式模式的 `native_refseg.py`。不构造或调用 Web 单图 Predictor；直接使用提交源码内原测试的模型、tensor 推理、变换与匹配原语。下文旧 Predictor/最高置信框/内部文本不改的阶段记录由本节覆盖。

Ollama 仍只选 `task`。Adapter 将 `question` 原字符串传到 native task；`api_task_text.py` 在任务侧处理明确的 Description 与答题包装，不改写、翻译或缩写描述。检测直接编码完整描述，传入原 phrase weight/单对象依赖，再按 phrase-map 匹配候选框；不使用题号、标注、GT 数量或计分，也不按 confidence 选定位框。无候选/无匹配明确失败。

| 任务 | 当前原生流程 |
| --- | --- |
| 场景分类 | 原 Network + strict Head；checkpoint 类序、RGB/640 eval transform、softmax |
| 单图 VQA 回答 | 正式 Q-Former v2 独立VQA，完整问题参与融合；五类答案头，保留原单选键/枚举值 |
| 检测定位 | 原 PyTorch backend；小图沿用800，大图长边按32对齐且最多1600，ceil resize、单图 rect/pad=.5/stride 补边、RGB/255；conf=.25/NMS=.7/max_det300；描述匹配 temperature=.7；原图 xyxy |
| 有无判断 | 同检测候选流程，不做单框匹配；NMS 框是否为空映射 Yes/No |
| 类别计数 | 原 ObjectCounter、类别遥感模板、800 方形 letterbox、.15/.5/300；NMS 后框数 |
| 显式 RefSeg | 原 strict best_raw、fp32 token features、checkpoint 输入尺寸及 validation threshold；原图 probability 恢复后阈值化 |

计数直接从题干提取一个目标类别，例如 `How many ships are in the lower left?` 提取 `ship`，交原 ObjectCounter；保留 `small vehicle` / `storage tank` 等复合类名以及 plane/planes/aeroplane 的类别归一。不再生成或记录 `counting_scope`、`counting_conditions_ignored`。缺失或多目标查询仍明确失败。精确数值和明确区间选项必须唯一匹配；无匹配或重叠选项失败，不就近选答案。计数模型的区域/关系筛选能力没有因此增加。

单图 `vqa` 调用独立VQA，`classification`调用21类场景分类。Qwen仅选task，VQA原题逐字传入，不提取目标、翻译或删限定词；失败码为`vqa_unsupported_task`、`vqa_unsupported_language`、`vqa_choice_unmatched`等。计数、有无与bbox继续用专用执行器，双图/变化仍不支持。VQA不接受integer/bbox；诊断记录`executor=vqa`，不输出原题或答案。

检测/计数最终模板/RefSeg描述在编码前核对77-token长度，超限 `text_context_exceeded`，不静默截断。

检测部署尺寸由 `HFSA_DETECTION_MAX_IMGSZ` 控制，API默认1600，设800可恢复原test默认尺寸。按原图尺寸选择 `min(cap, max(800, ceil(long_side/32)*32))`，继续使用原eval的rect/ceil resize/pad原语，不切图、不改模型或阈值。独立NativeDetection类默认cap800，原单图流程仍可复现。真权重CPU受控对比中，两条原定位请求在800无候选、1600分别有2/7候选；本机原题重放的选框/坐标与同尺寸原test原语一致、raw误差0。尚无新版官网IoU、正确率或GPU服务重启验证；有候选不能当作答对。

启动命令使用顶部的既有环境、权重、包、9001、原Key和当前Ngrok。默认加载分类/检测/计数/正式Q-Former VQA；`HFSA_GROUNDING_BACKEND=refseg` 才增加分割。代码更新后必须重启，等待 `API listening` 和鉴权 `/healthz`；无需重新创建Ngrok隧道。本次VQA只新增和训练任务侧Q-Former Head；原分类、检测、计数、RefSeg及Web流程保持原有实现，YOLO/CLIP冻结。

合成回归：在代码根目录执行 `python -B -m unittest discover -s participant_api_starter-main/tests`。可选真实对照：启动前设置 `HFSA_API_VERIFY_PARITY=1` 和 `HFSA_API_PARITY_IMAGE=/absolute/path/to/local/image.jpg`，新服务复用将要接题的模型核对原 tensor 流程与 native/API 的数值结果，检查通过才监听；默认检测模式临时分割检查共享现有CLIP并在检查后释放分割模型。此检查不用标注、不上传图片或题目、不提交官网，日志只含误差/数量/状态。对照通过不等于准确率提高。生产不需要设置上述验证变量。

当前规则和数值/运行证据见项目 `ADR/0038-official-native-task-inference.md`、`CURRENT_STATE.md`；旧章节的 HTTP200 与接口回归不是原完整推理一致性证据。

在线共享CLIP固定FP32、一次归一化。检测旧缓存可按auto在CUDA生成FP16，原多图rect批次形状也可能不同；当前检查限定单图在线FP32部署流程，不能声称历史缓存、完整离线测试或历史指标逐位复现。RefSeg读图/resize使用原数据集的OpenCV/RGB/INTER_LINEAR。

定位返回 `empty_detections` 时，可设置 `HFSA_API_DIAGNOSTICS=1` 后重启API。同一 `hfsa_api_prediction` 日志会追加 `detection_inference` 数值摘要：原图/网络输入HW、最高候选置信度、现有.25阈值、阈值前后候选数量。此开关默认关闭，不改变forward、阈值、NMS、匹配或答案，也不记录题干、目标类、框坐标或答案。已经通过的单图parity检查不能代表官网定位正确；应先用本机已有请求取得诊断，不需要重复官网评测。计数将等价词plane/planes恢复到原VOC类airplane，仍按真实全图NMS计数，不依据答案选项选类别。

完整字段、题型与错误处理见 [模型 API 协议 2.0](PROTOCOL_2_0.md)；评测网站的 `/help` 页面也提供独立可读的接入说明。

可选的真实 Qwen3.5-4B 参考实现与启动步骤见 [QWEN_ADAPTER.md](QWEN_ADAPTER.md)；`MODEL_MODE=demo` 仍无需 GPU 或模型依赖。

这个文件夹可以单独复制给队伍。它提供标准库协议外壳、可选的 Qwen3.5-4B 适配代码、测试和部署说明；**不含题目、金标、图片、模型权重或密钥**。队伍可以使用任意 VLM、推理框架和 GPU，只需让 API 遵守下面的接口。`DemoModel` 不看图、不懂题，只用于确认接入通路；不要拿它的分数衡量模型。

## 1. 整体工作方式

```text
评测网站 --HTTPS JSON(题干、约束、图片编号/哈希)--> 队伍 API --本地读取--> 预置图片
                                              |
                                              +--> 队伍自己的模型/GPU
评测网站 <--JSON(答案)-------------------------+
```

开发集 100 题的图片资源包由组委会另行提供，先下载到**运行模型 API 的机器**，解开后应是：

```text
/path/to/dev-image-package/
  manifest.json
  assets/
    <32 位小写十六进制 asset_id>
    ...
```

网站的单题请求**不传图片字节、图片 URL、原始文件路径或金标**；只发 `asset_id`、SHA-256、MIME 类型。服务从本地 `assets/` 取图并核验哈希。协议允许双时相请求按 `images` 原序携带前后两图；当前 HFSA 执行器已取消双图支持，会明确拒绝。开发包的 `dataset_id` 应为 `rsu-dev100-v2`；正式包可能是另一编号，API 从本地 `manifest.json` 自动读取，不应硬编码开发编号。以组委会最终发布的包和校验值为准。

## 2. 文件说明

| 文件 | 用途 |
| --- | --- |
| `server.py` | `GET /healthz`、`POST /v1/predict`、Bearer 认证、本地图片核验 |
| `model_adapter.py` | 演示模型与需要队伍实现的 `RealModel` 接口 |
| `qwen_adapter.py`、`QWEN_ADAPTER.md` | Qwen3.5-4B 真实模型适配参考与逐步说明 |
| `requirements-qwen-reference.lock` | 本地 CUDA 13.0 环境参考版本，不是所有 GPU 的通用配置 |
| `check_api.py` | 对本机或公网地址做一次**不计分**的握手和预测检查 |
| `tests/test_server.py` | 用合成数据测试协议；不依赖真实赛题 |
| `Caddyfile.example` | 公网 GPU 服务器的 HTTPS 反向代理示意 |

协议外壳和演示模式只依赖 Python 3.10+ 标准库。`MODEL_MODE=qwen` 才需要 PyTorch、Transformers、Pillow 等；其他模型按队伍选择安装依赖。下文用 `python3` 表示选定环境中的 Python；如果机器上已有兼容的虚拟环境，可以直接用该环境的 `bin/python` 运行测试、服务和 `check_api.py`，无需重复安装依赖。运行真实 Qwen 时，先按 [Qwen 环境说明](QWEN_ADAPTER.md)核对 GPU、PyTorch 和模型目录。

## 3. 本地跑通协议

先在这个文件夹运行自带的合成测试，无需 GPU 或真实图片：

```bash
cd participant_api_starter
python3 -m unittest discover -s tests -v
```

然后把组委会提供的图片包解到推理机上的单独目录，确认 `manifest.json` 与 `assets/` 同级。设置一个**队伍自己生成**的模型 API token（不同于网站登录令牌、ngrok authtoken）：

```bash
# 从密码管理器取出至少 32 位随机值；此交互输入不会出现在 shell 命令历史。
read -rs -p 'MODEL_API_KEY: ' MODEL_API_KEY; echo
export MODEL_API_KEY
export MODEL_MODE=demo
python3 server.py --package /path/to/dev-image-package --host 127.0.0.1 --port 9001
```

上面的服务默认仅监听本机回环地址。另开一个终端，设置**同一个** `MODEL_API_KEY`，验证；两个终端的环境变量互不共享：

```bash
read -rs -p 'MODEL_API_KEY: ' MODEL_API_KEY; echo
export MODEL_API_KEY
python3 check_api.py --origin http://127.0.0.1:9001 --package /path/to/dev-image-package
```

`check_api.py` 从清单挑一张本地图片，发送合成问题，只检查协议、鉴权、图片查找和响应回显；它不会得到开发集分数。如果只是测健康接口，也可用：

```bash
curl -fsS -H "Authorization: Bearer $MODEL_API_KEY" http://127.0.0.1:9001/healthz
```

预期至少包含 `{"status":"ready","protocol_version":"2.0","dataset_id":"rsu-dev100-v2"}`。若 `dataset_id` 不符，检查是否指向了错误的图片包。如果 `/healthz` 或 `check_api.py` 返回 401，先确认检查终端使用的是**服务启动时**的密钥。服务在启动时读取 `MODEL_API_KEY`；之后只在某个终端修改或重新生成密钥，不会更新已运行的服务。更换密钥时，重启服务，并在检查终端和网站填写同一新密钥。

`check_api.py` 成功只证明一次合成题走通，演示模型也能通过。若要评测真实 Qwen，确认启动服务前设置了 `MODEL_MODE=qwen` 和 `QWEN_MODEL_DIR`，且启动日志没有 `demo mode` 警告。脚本在本地 HTTP 检查成功时也会打印 `HTTPS/API handshake`；这条固定提示不代表公网 HTTPS 已验证。

## 4. 接入自己的模型

想看真实模型接入，先读 [Qwen3.5-4B 适配示例](QWEN_ADAPTER.md)。准备本地权重和兼容环境后，设置 `MODEL_MODE=qwen`、`QWEN_MODEL_DIR=/absolute/path/to/Qwen3.5-4B`，继续使用同一个 `server.py`。其他 VLM 可借鉴其题干约束、双图顺序和 bbox 坐标转换。

编辑 `model_adapter.py` 中的 `RealModel`：

1. 在 `__init__` 里加载模型与权重，放到 GPU；服务启动时只做一次，**不要每题重载**。
2. 在 `predict(request, image_paths)` 中按 `image_paths` 顺序读取本地图像。不要把 `asset_id` 当成远程 URL，也不要向评测网站请求图片。
3. 将 `request["question"]`、可选的 `request["choices"]` 与 `request["response_constraint"]` 一起组织为模型输入。
4. 输出与题型相符的原始答案。短文本、单选、区域标签和整数题都返回**字符串**；`bbox` 返回原图像素坐标数组 `[xmin, ymin, xmax, ymax]`，必须满足 `xmin < xmax`、`ymin < ymax`。若模型产生 0–1000 归一化坐标，需要先乘原图宽高转换。
5. 不要在日志里打印 API token、完整题目、图片或模型私有信息。

改完再用 `export MODEL_MODE=real` 启动。未实现 `RealModel` 时服务会明确报错，这是为了防止把演示答案误当真实模型结果。这个示例一次只允许一个推理请求；评测站对同一队伍顺序发题，串行是安全的最小起点。若你的模型能并行推理，可按显存和实际吞吐自行调整 `server.py` 的并发限额。

### 接口样例

`GET /healthz` 和 `POST /v1/predict` 都要求 `Authorization: Bearer <MODEL_API_KEY>`。预测请求示意（编号和哈希取自本地图片包，此处不放真实题目）：

```json
{
  "protocol_version": "2.0",
  "request_id": "run-id:EXAMPLE:1",
  "item_id": "EXAMPLE",
  "question": "Is there anything in the image?",
  "images": [{
    "asset_id": "<manifest 中的 32 位编号>",
    "sha256": "<manifest 中的 64 位哈希>",
    "mime_type": "image/png"
  }],
  "response_constraint": {"type": "enum", "values": ["Yes", "No"]},
  "answer_type": "short_text"
}
```

成功时必须回显版本、请求 ID、题目 ID，并给出 `answer`：

```json
{"protocol_version":"2.0","request_id":"run-id:EXAMPLE:1","item_id":"EXAMPLE","answer":"Yes"}
```

`response_constraint.type` 可能为 `enum`、`single_choice`、`short_text`、`integer` 或 `bbox`。`single_choice` 另带 `choices`；`bbox` 另带 `image_width`、`image_height`。推理异常要返回非 2xx JSON，不能返回伪造的正确答案；失败题在网站上按缺失预测处理。

## 5. 最快公网接入：ngrok（适合调试）

如果 GPU 在本机、WSL 或不能开公网端口的机房，ngrok 是一个简单的起点：模型 API 继续监听 `127.0.0.1:9001`，ngrok 主动连出并提供公网 HTTPS 域名，不需要给 GPU 机开入站端口。**在运行 API 的同一系统/网络命名空间内**安装 [ngrok agent](https://ngrok.com/docs/getting-started/)，注册账号并按官网指引配置 authtoken；ngrok authtoken 与 `MODEL_API_KEY` 不能混用。

保持上一节的 API 终端运行，再开一个终端：

```bash
read -rs -p 'NGROK_AUTHTOKEN: ' NGROK_AUTHTOKEN; echo
ngrok config add-authtoken "$NGROK_AUTHTOKEN"
unset NGROK_AUTHTOKEN
ngrok http 9001
```

如果出现 [`ERR_NGROK_9009`](https://ngrok.com/docs/errors/err_ngrok_9009)，检查 ngrok 配置中的 `proxy_url`，以及当前终端的 `http_proxy`、`https_proxy`、`HTTP_PROXY`、`HTTPS_PROXY` 等变量。这个错误表示 ngrok agent 正尝试经 HTTP(S) 代理连接；若机器可以直连，可只对 ngrok 进程临时去掉代理变量后重试，不改变其他程序的代理设置：

```bash
env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
  -u all_proxy -u ALL_PROXY ngrok http 9001
```

如果网络必须通过代理才能出站，去掉变量后可能无法连接，应改用符合账号能力的连接方式或下节的自有 HTTPS 服务。

不要把 authtoken 写进仓库、截图或发给组委会。以 ngrok 终端**实际显示**的 HTTPS 地址为准，域名后缀可能是 `.ngrok-free.dev` 等。**只把 origin 填到评测网站**，不要附加 `/healthz`、`/v1/predict`、问号参数或末尾路径。将 `MODEL_API_KEY` 的原始值填到网站的“模型 API key”字段，不加 `Bearer ` 前缀；网站登录令牌只用于登录网站。

先在当前机器用公网地址检查，再视需要从其他网络重复检查。运行 `check_api.py` 的机器必须能读取同一图片包的 `manifest.json`，并设置相同的 `MODEL_API_KEY`；检查脚本不读取该机的 `assets/`。把示例地址替换成 ngrok 当前显示的完整 HTTPS origin：

```bash
PUBLIC_ORIGIN=https://your-ngrok-domain.ngrok-free.dev
python3 check_api.py --origin "$PUBLIC_ORIGIN" --package /path/to/dev-image-package
```

如果检查失败，先用 `curl -i -H "Authorization: Bearer $MODEL_API_KEY" "$PUBLIC_ORIGIN/healthz"` 看是否返回 **200 + JSON**，而不是 HTML、重定向或 ngrok 错误页。ngrok 免费档目前有流量/请求额度与浏览器访问提示页；评测程序是 API 客户端，但是否受到提示页影响仍要以上述端到端检查为准。**不要依赖评测网站会替你加 ngrok 专用跳过提示头**。ngrok 定价和限额会变化，请以 [官方定价页](https://ngrok.com/pricing) 为准。重启 ngrok 后也要重新确认地址和健康检查，不要假定隧道始终在线。

评测期间必须保持 GPU 机器、模型进程、ngrok 进程和网络一直在线；电脑休眠、WSL 关闭或隧道退出都会导致失败。建议先在网站跑 8 题 `smoke`，等待运行完成并检查超时、缺失预测等异常，再跑 100 题 `full`。ngrok 适合开发和短时验证；若比赛要求较长稳定在线，优先用下一节的自有公网服务。

## 6. 长时间接入：自有 HTTPS 服务

### A. GPU 服务器本身有公网 IP

为 GPU 服务器准备自己的域名，配置 DNS A/AAAA 记录，并在云安全组和主机防火墙放行 TCP 80/443。模型 API 仍绑定 `127.0.0.1:9001`，不要直接开放 9001。用 [Caddy 官方反向代理配置](https://caddyserver.com/docs/quick-starts/reverse-proxy)中的方式把 HTTPS 流量转给它；可将 `Caddyfile.example` 的域名改为自己的域名。Caddy 在满足 DNS 和端口条件时自动办理 HTTPS 证书。启动后从外网运行上一节相同的 `check_api.py`，成功才登记到评测网站。

### B. GPU 机器在内网，但队伍有自己的轻量公网主机

在公网主机配置域名 + Caddy，反代到该主机的 `127.0.0.1:19001`。在 GPU 机器上启动模型 API 后，再手动建立 [OpenSSH 反向端口转发](https://man.openbsd.org/ssh.1)：

```bash
ssh -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=15 -o ServerAliveCountMax=3 \
  -R 127.0.0.1:19001:127.0.0.1:9001 relay-user@relay.team.example.com
```

这里 `relay.team.example.com`、SSH 用户和反代域名都由**队伍自己**提供；不要默认占用组委会开发站的轻量云服务器。公网主机 Caddy 反代目标改为 `127.0.0.1:19001`。应给 SSH 隧道使用独立低权限账号，并限制其可监听端口；同样需要防止 GPU 机器休眠和 SSH 连接断开。这个方案的公网主机仅转发小型 JSON 请求与答案，图片仍留在队伍 GPU 机器的本地资源包。

这两种方式都将公网 HTTPS 终止在队伍控制的服务或中继上；ngrok 则经过第三方服务。队伍应依据自身安全和可用性要求选择。

## 7. 到评测网站登记与排错

1. 用组委会给的队伍账号登录开发评测网站；队伍只能看到本队成绩。打开模型 API Endpoint 设置，填上**自己**的 HTTPS origin（例如 ngrok URL），保存。
2. 在提交评测时填入与模型 API 服务端一致的 `MODEL_API_KEY`。网站会先检查 `/healthz`，包括 `protocol_version=2.0` 和 `dataset_id`，再开始计次。
3. 先跑 `smoke`，确认图片包、本地推理、响应格式和公网连通性；再跑 `full`。示例里的 `DemoModel` 只用于检查；真实评测可使用已配置的 `MODEL_MODE=qwen`，或设置 `MODEL_MODE=real` 并实现自己的模型适配器。

| 现象 | 优先检查 |
| --- | --- |
| 本地 `check_api.py` 返回 401 | 检查终端和服务启动时的 `MODEL_API_KEY` 是否相同；改过密钥就重启服务 |
| 网站提交报 `invalid_submission`、`participant preflight failed (http_error)` | 网站预检访问 `/healthz` 得到了非 200 状态；在 ngrok 的本地请求检查界面或反向代理日志中查看该请求的实际状态码；界面地址以 ngrok 终端显示的为准，端口不一定是 4040。若是 401，即使本地或公网 `check_api.py` 成功，也应重填网站提交时的模型 API key；不要填 ngrok authtoken、网站登录令牌或 `Bearer ` 前缀 |
| `ERR_NGROK_9009` | 检查 ngrok 是否继承了 HTTP(S) 代理变量，或配置了 `proxy_url`；可按上文只对 ngrok 进程临时取消代理并验证直连 |
| `/healthz` 不是 `ready` 或数据集不符 | 是否用了正确的 `manifest.json`、`assets/` 包，以及正确的 `--package` 目录 |
| 422 图片错误 | 图片文件是否缺失/被改动；文件 SHA-256 是否与 `manifest.json` 相同 |
| 502/HTML/重定向 | ngrok、Caddy、浏览器提示页或反代配置；程序必须收到 200 + `application/json` |
| 连接超时 | GPU 机器/模型/隧道是否在线；外网 DNS、HTTPS 证书和防火墙是否正常 |
| 有结果但分数低 | 先确认不是 `DemoModel`；再检查答案类型、单选字母、bbox 原图像素坐标与题目约束 |

ngrok 的请求检查界面可能记录 `Authorization` 请求头；排错时只分享请求路径和状态码，不要公开原始请求头。不要把图片资源包、模型权重、`.env`、私钥或真实 API token 一并打包给他人。这个文件夹的 `.gitignore` 默认忽略常见本地机密和大文件，但仍应在发布前自行检查归档内容。

## 8. 本地 HFSA 四任务接入（2026-10-09，历史记录）

本节及第10、11节记录当时实现和测试结果；涉及VQA支持范围、默认权重和启动方法以顶部当前Q-Former节为准。

`MODEL_MODE=real` 通过 `hfsa_adapter.py` 复用已有分类、检测、计数和 RefSeg Predictor。官方请求没有 `task_id`，因此只有这个 Adapter 按已支持的题干形式选择任务；原 Web 仍由用户手动选择。

当前电脑目录为 `提交/HFSA-main/`（代码），同级 `提交/HFSA_models/`（权重）。需要分类 `runs/classification/vrsbench_scene/weights/best.pt`、检测 `runs/detect/DIOR-RSVG-ViT-L-14/weights/best.pt`、计数 `runs/counting/counting-VRSBench-ViT-L-14/weights/best.pt`、分割 `runs/semseg/srp_yolov12m_axis/weights/best_raw.pt` 和 `pretrain_model/yolov12m.pt`。可用 `HFSA_MODELS_DIR` 指定其他权重根目录，`HFSA_API_DEVICE` 指定设备（默认 `auto`）。OpenCLIP ViT-L-14/openai 须已在运行账户缓存中。

在 **WSL 终端**执行以下本机命令，继续使用现有 `hfsa_env`，无需 Qwen 环境：

```bash
cd /mnt/d/code/python/HFSA/提交/HFSA-main
source /mnt/d/code/python/HFSA/hfsa_env/bin/activate
read -rs -p '输入你保存的 MODEL_API_KEY（至少32字符）: ' MODEL_API_KEY
printf '\n'
export MODEL_API_KEY
export MODEL_MODE=real
python participant_api_starter-main/server.py \
  --package /mnt/d/code/python/HFSA/rsu-dev100-v2-images \
  --host 127.0.0.1 --port 9001
```

四任务均在监听前加载一次，检测与计数共用一个既有 OpenCLIP 文本编码器；分割保留自己的 token encoder。启动会显示加载进度，等待 `API listening` 才表示可以接题。此前本机合成 smoke 启动约 101 秒，实际加载可能更久，以监听及 ready 为准，单次检查峰值 PyTorch 显存约 3.86 GiB；这不是正式效率评测。

| 任务 | 已支持题干示例 | 返回方式 |
| --- | --- | --- |
| 场景分类 | `Which scene category best describes the image?` / `图中属于什么场景？` | 单选返回字母，枚举返回允许值，短文本返回 checkpoint 类名 |
| 目标计数 | `How many ships are there in the image?` / `图中有多少架飞机？` | 整数字符串；数值选项返回匹配的标识 |
| 目标检测 | `Is there a ship in the image?` / `图中是否有飞机？` | 根据检测结果返回 Yes/No 或对应选项 |
| 目标检测 bbox | `Detect a windmill in the image.` / `检测图中的飞机` | 指定单一类别中置信度最高的原图像素框 |
| 指代分割转定位 | `Locate the airplane near the bridge.` / `框出图中灰色的小风车。` | 原图二值 Mask 的外接框；保留表达中的限定词 |

分类支持 checkpoint 中的 21 个场景类别，选项的英文别名或简单中文必须能映射到这些类别；含未知类或重复同义选项则报错。检测/计数要求明确的遥感目标类别。中文复杂关系仍由现有翻译器明确拒绝；英文 RefSeg 表达保持原样送入模型。计数保持原 `conf=0.15`、`IoU=0.5`、`max_det=300` 和独立计数权重，不增加空间关系计数策略。

未知题型、双图变化、通用问答、空间/多类别计数、空目标及空定位结果均失败，通过官方外壳返回 HTTP 500 `inference_failed`。本实现不是通用视觉语言模型，不保证覆盖整个开发集。官方 `check_api.py` 的 `Is there anything...` 合成题仍不支持，因为它没有指定目标类别；不要把该题失败误判为鉴权或连通故障。

本地 Adapter 现在会向 stderr 输出一行 `hfsa_api_prediction` JSON 诊断，包含题号、请求编号、答案约束、图片数量、所选任务、耗时及固定失败代码。它不记录题干、选项、答案、图片路径、认证头或原始异常文字；公网错误响应仍由官方 `server.py` 处理。代码更新后须重启 API 才能生效。

常见 `reason`：`unsupported_question_form` 为未覆盖的问法；`unsupported_image_count` 为当前不支持的双图任务；`unsupported_scene_options` 为场景选项超出分类权重类别；`unsupported_spatial_or_multi_category_count` 为未实现的计数条件；`empty_refseg_mask` / `empty_detections` 为模型未找到目标；`unexpected_exception` 为需要继续排查的异常，同时记录异常类型。诊断只解释失败类别，不会把未知题转为猜测答案，也不会自动提高成绩。

如果官网 `audit` 显示 HTTP 500，而 `predictions` 缺少相应行，应结合本机这些诊断定位；audit 本身只包含状态/摘要，不能还原题干或 Python 异常。先排查 8 题 smoke，不重复运行 full。后台运行时本机日志位置见项目 CURRENT_STATE，账号令牌及原始评测下载文件不纳入 Git。

### 用本地重放减少官网提交次数

提交次数有限时，先完成本机检查；不能为了每次收集日志就重新跑官网。只有接口合成测试通过还不足以确认真实题型已支持，缺少真实题干时须明确说明这一限制。

可选设置 `HFSA_API_REPLAY_DIR` 为项目外的本机私有目录，并重启API。Adapter会在路由/推理前保存已经通过外壳认证和图片核验的请求字段：题干、作答约束、选项、图片引用、尺寸及协议标识。默认关闭；不保存认证头、未知字段、模型答案或金标，同一请求去重，文件名使用内容哈希。日志仍不输出题干。请求保存失败不会改变模型作答；正式题的本地副本不上传Git，不按题号硬编码答案。

收到真实请求后可在WSL环境执行以下命令，保持与运行服务相同的 `MODEL_API_KEY`：

```bash
python participant_api_starter-main/replay_requests.py \
  --requests "$HFSA_API_REPLAY_DIR"
```

该工具只向loopback HTTP接口顺序重放，禁用HTTP代理及重定向，不调用评测网站或消耗官网提交额度。输出题号、HTTP状态、格式有效性与耗时，不输出题干、答案或密钥。格式有效不代表答案正确；没有金标时不报告准确率。最初没有捕获的请求不能从audit哈希恢复，须保留这一未验证范围，不承诺官方8题全部可答。

可重复检查：

```bash
cd participant_api_starter-main
python -m unittest discover -s tests -v
python tests/smoke_hfsa_real.py \
  --image /path/to/your/windmill-image.jpg \
  --target windmill --refseg-question 'The gray small windmill'
```

真实 smoke 用用户提供的单张图创建临时合成图片包，只监听随机本机端口，并检查四任务、鉴权、健康检查、标识回显、像素框和双图拒绝；完成后关闭服务并删除临时包。它不改官方图片包、不打印测试 Key，也不提交网站评测。后续公网接入状态见第9节。

## 9. 当前电脑复用已运行的 ngrok（2026-10-10 更新）

用户已选择 ngrok，现有 Windows agent 与账户配置已接通 WSL `127.0.0.1:9001`。优先复用已运行隧道，不重复启动、安装、配置令牌或复制工具。此前给出的 `%LOCALAPPDATA%\HFSA\PublicAPI\ngrok.exe` 路径在用户 PowerShell 中不可见，相关启动/令牌配置命令已作废；工具侧可见路径不能直接当作用户终端的安装路径。

当前 HTTPS origin 及实际健康检查记录见项目 `CURRENT_STATE.md`。官网填写 agent 实际 HTTPS 根地址，不加接口路径或末尾斜杠；模型 API Key 与启动 API 时完全一致，不加 `Bearer `。Ngrok Authtoken 仅用于隧道账户，两种凭据不能混用。

保持 API、Ollama、ngrok、电脑与网络在线。需要重启 ngrok 时，先核对当前环境实际已安装的可执行路径和账户配置，再启动并重新确认 HTTPS origin 和认证健康检查；本说明不提供未经核对的替代路径。先前 Cloudflare 隧道失败地址不能复用。官网评测由用户按当前授权自行提交，公网连通不代表评测已完成。

## 10. 真实题定位包装修复与能力补充评估（2026-10-09，历史记录）

明确的bounding box指令若带Description字段，Adapter先提取非空完整描述，保留多句关系与颜色/方位，再交给既有RefSeg。合成测试与两条真实请求的完整描述解析均通过；当前后台尚未重启到此次修复，尚无真实bbox/IoU或新官网得分结论。

已有四个专用模型仍存在通用视觉问答、未知类别有无、颜色、条件计数和双图变化的能力缺口。用户要求解决零分，因此已评估另接小型视觉语言模型的方案，详见项目Proposed ADR-0036；尚未下载/安装/上线，不能通过放开门禁或默认填值声称具有新能力。当前8GB显卡约1.6GB余量，不可照抄官方4B全BF16与四任务共驻；需要独立环境和经过实测的量化/互斥驻留策略。

## 11. 本地 LLM 路由与完整开发集评测（2026-10-09，历史记录）

本节按2026-10-10最新授权覆盖前文 HFSA 确定性模板、旧目标门禁和英文target/Description规划。`MODEL_MODE=real` 使用 `ollama_router.py` 调用已有 Ollama `qwen3:1.7b`，只选择任务，再显式调用 HFSA Predictor；`MODEL_MODE=qwen` 是另一个完整 Qwen 视觉后端，不能代替这个组合。

| LLM 标签 | 当前处理 |
| --- | --- |
| `classification` | 既有图像-only分类；题干仅选任务，保留选项概率映射 |
| `presence` | 官网question原文传既有检测，输出目标有无 |
| `counting` | 官网question原文传既有计数 |
| `visual_grounding` | 官网question原文传默认 detection；显式 `HFSA_GROUNDING_BACKEND=refseg` 才用原 mask 外接框 |
| `change` | 明确 `unsupported_change` |
| `vqa` | 当时明确 `unsupported_vqa` |

旧关键词任务分类和检测/计数目标白名单已删除，没有静默回退。2026-10-10 用户授权 bbox 默认改为 detection，后端由配置确定；非法计划、路由失败和任务/作答格式冲突明确失败。官方认证、图片哈希、答案和原图坐标校验仍保留。Web 保持手动任务选择及原分割入口，权重、Backbone、Neck、OpenCLIP、Head 及训练未变。

文本路由看不到图片，不补齐颜色/属性或变化视觉能力。它只输出task，不生成target/conditions。API把官网question原字符串传给需要文本的Predictor，不提取Description、不strip、不去包装或作答说明、不翻译中文。这个保证仅到API→Predictor，模型内部原有tokenizer/clean_prompt/模板/长度上限不改，不代表长原题会完整编码或被正确理解。默认检测无框明确失败；显式RefSeg模式空mask仍失败，不返回猜测框。

现有 WSL Ollama 为 `http://127.0.0.1:11434`，`qwen3:1.7b` 已下载；继续原 GPU 策略及 `hfsa_env`，不重装或下载。若有临时真实重放服务，先等其结束，避免两套模型服务共驻 GPU；若旧正式 API 在运行，在其所属终端按 Ctrl+C 后再启动更新服务。以下默认加载分类、检测、计数，不加载 RefSeg：

```bash
cd /mnt/d/code/python/HFSA/提交/HFSA-main

read -rs -p '粘贴已保存的 MODEL_API_KEY: ' MODEL_API_KEY
printf '\n'
export MODEL_API_KEY
export MODEL_MODE=real
export HFSA_ROUTER_URL=http://127.0.0.1:11434
export HFSA_ROUTER_MODEL=qwen3:1.7b
export HFSA_GROUNDING_BACKEND=detection
export HFSA_MODELS_DIR=/mnt/d/code/python/HFSA/提交/HFSA_models
export HFSA_API_DEVICE=auto

/mnt/d/code/python/HFSA/hfsa_env/bin/python -B -u \
  participant_api_starter-main/server.py \
  --package /mnt/d/code/python/HFSA/rsu-dev100-v2-images \
  --host 127.0.0.1 --port 9001
```

先预热路由再加载配置要求的模型，等待 `API listening` 和认证 `/healthz` 的 ready；加载日志不代表就绪。若确实需要旧 RefSeg bbox 分支，启动前显式设置 `HFSA_GROUNDING_BACKEND=refseg`，该模式才导入 RefSeg 推理、检查其权重并加载第四个模型。官网使用启动时相同的实际 Key，不加 `Bearer `，不把 ngrok Authtoken 当成模型 Key。

复用当前已运行的 ngrok 隧道；不继续使用已在用户终端证实不可见的 AppData 启动路径。若需要重启，按第9节先核对实际已安装路径，不臆造目录或复制工具。

以当前 ngrok 显示的 HTTPS 根地址为准，认证健康检查和真实本机/公网预测检查通过后，由用户登记 Endpoint 并执行一次 100 题完整开发集 `full`。本轮不为收集日志追加官网 smoke，不自动重复 full；用户自行操作网站，不检查其额度。保持 API、Ollama、ngrok、电脑和网络在线；排队/运行中不等于完成。协议记载按 UTC 日期重置，不能说北京时间午夜清零，实际站点配置优先。

合成路由 10/10，通过六类、条件计数和 bbox 关系表达；API 全套 45/45、4.612 秒，含七条 LLM HTTP mock 边界测试，WSL `hfsa_env` / `-B`。这些是路由/接口验证，不是视觉准确率或官网成绩；真实重放及完整评测结果在项目状态/实验日志中另记。根 Git 不可用，本轮未修复、提交或推送，尚未远端同步。详见 ADR-0037。

### 2026-10-10 路由契约修复

本轮按用户最新要求只修现有路由，不接入暂缺的视觉任务。每次请求单独复制 JSON Schema，根据 `response_constraint.type` 排除不兼容任务：文本/单选/枚举/整数禁止返回 `visual_grounding`，bbox 排除分类、有无与计数；两图只允许 `change` 或 `vqa`。这只限制非法计划，不把答案格式直接当作语义分类。模型返回后再检查允许任务，不做关键词回退或默认答案。

位置、方向和关系的文本/单选问答属于 `vqa`，当时仍明确 `unsupported_vqa`；双图变化仍 `unsupported_change`。格式冲突不再把已经校验的任务写成 `task=null`，所有诊断任务字段统一通过白名单，仍不记录题干、目标文本、答案、路径、密钥或异常文字。

回归55/55通过；真实Qwen6条针对性合成路由通过，包含短文本/单选位置问答、bbox、条件计数、场景分类和双图。另用旧私有请求验证一项原短文本定位误路由现为 `vqa`，尚无原16项全部恢复200或官网提分证据。7项计数选项不匹配没有完整原始选项，计数解析与模型保持原状，不提取任意数字、猜选项或强制就近匹配。

### 2026-10-10 bbox 原文描述传递修复

用户后续8题smoke发现两题分割为空。任务标签均为visual_grounding，但真实请求检查发现LLM仍可能缩写描述：其中一题新路由只产生54字符，旧提示/Schema在相同输入上产生232字符；原始Description实际273字符。任务Schema正确不能证明目标文本完整。

现在仅在已判定 `visual_grounding` 且约束 `bbox` 时读取明确 `Description:` 包装，已有英文描述直接从原题整段复制，保留全部句子、限定词、only和标点，仅去外围空白。标签不决定任务；不恢复关键词分类，也不按题号写答案。没有此包装时仍用LLM目标，中文仍沿用英文翻译。空描述或超过1800字符明确失败。

新增7项回归后62/62通过；两条实际bbox请求传入完整273/263字符描述，已逐字核验。这仅证明描述传递修复，真实RefSeg是否有mask和bbox IoU须分别核验；不降低阈值或用猜测框消除失败。

### 2026-10-10 bbox 默认检测与条件加载

用户已授权官网 bbox 默认使用现有检测，覆盖前文全部定位走 RefSeg 的决定。`HFSA_GROUNDING_BACKEND` 默认 `detection`，显式 `refseg` 保留旧分支；未知值在启动时失败，单题不会自动回退。默认只加载分类、检测、计数，不导入 `tasks.refseg.inference`、检查 RefSeg 权重或构造 RefSeg；原 Web 分割不变。

按用户后续原题直传要求，当前传入既有Predictor的是完整官网question，不再提取 `Description:`。检测仍使用原有 800 letterbox、conf=0.15、IoU=0.5、max_det=300 和原图坐标恢复；不能把输出原图坐标称作无需缩放。API 校验原图 `(H,W)`、框/分数的形状、数量和有限数值，选最高置信度的一个框，clip 到原图边界，左上 floor、右下 ceil。无框为 `empty_detections`，非法结果为 `invalid_detection_result`，裁剪后空面积为 `empty_detection_box`；尺寸不符仍为 `image_dimensions_mismatch`。不合并框、不猜坐标，也不承诺最高分框满足所有关系描述。

以下是后续原题直传之前的后端切换阶段证据：API 回归73/73、4.703秒（含三项from_local条件加载mock），当时服务ready、公网health200；两道原bbox重放均500/empty_detections，有无/计数各1项200，独立合成bbox200。它们不能作为当前task-only/raw-question策略的运行或准确率证据。此前一原题RefSeg有框而当时检测无框，因此也不能声称换检测改善有效返回；无新官网成绩或IoU。详见项目状态及实验日志的对应历史记录。

### 2026-10-10 仅选任务，官网question原文到Predictor（历史记录）

用户最新要求覆盖上文目标规划和描述提取：路由受约束JSON仅 `{"task":"..."}`，固定六标签，无target/conditions生成；格式和图片数量白名单校验保持。`parse_question` 仅保留任务及官网question原文，不extract/strip/translate。

presence调用检测、counting调用计数、grounding调用配置指定的detection/refseg时，第二参数都直接使用 `request["question"]`。官网包装、限定词、答题格式要求、中文、标点及空白均保留。classification Predictor是图像-only，题干只用于任务选择；choices/response_constraint仍属于路由及回答契约，不拼进视觉模型文本。

内部tokenizer、clean_prompt、模板和长度上限均保持现状，所以只保证API到Predictor的原字符串，不宣称内部token不截断/不归一化或模型能理解长原题。默认detection bbox及原错误/坐标校验、可选refseg、条件加载、Web和缺能力任务处理不变，不新增视觉能力。

本策略完整API回归78/78、4.671秒，真实Ollama合成任务选择10/10；四条私有原题（两bbox、一presence、一counting）真实Qwen解析task符合预期，QuestionPlan仅task/original_question且原字符串逐字相等。更新API一次启动113.48秒后本机鉴权health200/ready；同四题真实重放均200、严格response_valid通过，无503或路由错误，两bbox均有合法原图框。它们证明任务选择、原文传递及有效返回；没有金标/IoU，不能称准确率提升或官网提分。上文62/73项与空框结果为前阶段历史证据。

旧ngrok曾退出，已在Windows9001health200后复用实际已安装、签名有效的工具和原账号配置恢复隧道，未下载、复制或建立目录。原HTTPS origin与模型Key保持，上游为9001，公网鉴权health最终200/ready/2.0/rsu-dev100-v2。继续复用当前隧道，不使用此前失效的AppData命令；重启仍须核对实际安装路径。这里没有官网提交或得分结果，服务及公网就绪不等于完成评测。
