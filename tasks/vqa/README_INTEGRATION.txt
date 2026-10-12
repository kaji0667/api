HFSA 独立视觉问答模块（当前十二头 Q-Former v2）

2026-10-13更新：已合并队友十二类答案头及对应权重，当前支持全部TYPES，min_answer_count=1。API成功返回类型同步；整数计数与双图边界不变。交付逐题结果及本机验收限制见TEAM_V3_MERGE.md。以下五头训练参数/结果属于2026-10-11历史记录，不能用其验收报告证明新包在线/缓存一致。

输入为一张图片和原始完整英文问题。VQA 模块自身不依赖 API、Web 或 Ollama；官方 API 从外部调用 tasks.vqa.inference.VQAPredictor。当前 API 的唯一 VQA 架构为 hfsa_frozen_qformer_vqa_v2，fusion_backend=blip2_qformer，默认模型目录 runs/vqa/qformer。API 启动要求正式训练和全量验证收据通过门禁；短训练产物不作为服务模型。服务启动方法见 participant_api_starter-main/README.md。

架构与能力边界：
  从 BLIP-2（ICML 2023）官方 Salesforce LAVIS 源码直接迁入六个注意力/FFN 核心类，固定提交 05cb670f47296522e24e7727e366b4a997d28582。冻结 YOLOv12m 的 14/17/20 层输出，经 16/8/4 空间网格池化、投影和坐标/尺度编码，形成三尺度视觉 token。32 个可学习查询与一个完整问题 CLIP 向量共享自注意力，查询单独交叉注意力到视觉 token；查询和语言分别执行官方 FFN，经问题条件读出进入五个答案头。默认 2 层、4 头、192 维、4 倍 FFN、每 2 层交叉注意力。核心在 qformer_core.py，任务适配在 qformer.py，出处、许可证和原源码在 third_party/lavis。
  YOLO 和 ViT-L-14/OpenAI 文本编码器冻结并保持 eval，仅训练新增 Head 的 2,864,405 个参数。完整问题每 75 个内容 token 连续分块，按内容长度加权聚合；问题表示为池化 CLIP 向量，未改为单词级 BERT 序列。
  论文 https://proceedings.mlr.press/v202/li23q.html 的 SOTA 结论对应发表时的零样本 VQAv2 评测。这里只迁移核心结构并重新训练任务侧参数，没有导入 Q-Former、BERT、OPT、T5 预训练权重或 BLIP-2 预训练目标，不沿用论文分数。
  答案头为 category、color、shape、existence、scene 五类有限词表；数量、位置、比较、关系、方向和大小答案头未实现。问题中的位置限定保留，但不代表已验证关系推理。无切片、ROI 或显式 grounding 监督，evidence 中目标/参考框为空。

预处理：
  PIL 解码 RGB；640×640 双线性整图缩放；ImageNet 归一化。没有翻转或颜色扰动。每尺度保留 16/8/4 网格，视觉特征按 float16 舍入后以 float32 送 Head，在线与缓存一致。文字不提取目标、不加额外模板；编码器自身标准 BPE 规范化仍有效。超过 10000 字符或 750 内容 token 明确拒绝。

工程目录（以下命令均从提交/HFSA-main 根目录运行，使用既有 HFSA 环境）：
  tasks/vqa/                         数据、缓存、模型、训练、评估和推理代码
  config/vqa/qformer.json             正式训练配置
  scripts/train_vqa.sh                正式训练入口
  scripts/eval_vqa.sh                 全量验证入口
  scripts/verify_vqa_qformer.py       当前权重在线/缓存核验
  scripts/verify_vqa_api.py           当前权重与 HTTP 接口核验
  scripts/serve_hfsa_api.sh           使用原已 export Key 启动当前 API
  runs/vqa/qformer/                   weights/config/manifest/training_result/evaluation/reports
  runs/vqa/stage1/examples/           仅供 API verifier 读取既有公共图片和问题
主权重为 runs/vqa/qformer/weights/vqa_head.pt。bundle_dir 表示含 manifest.json 的模型目录；API 不再接受旧架构作为 VQA 服务模型。公共样例目录不是权重入口，API verifier 不读取其旧 expected_predictions 或金标文件。

数据准备：
  python -m tasks.vqa.data --annotations-dir /path/Annotations_train.zip --images-dir /path/JPEGImages --xml-dir /path/Annotations --split train --out data/vqa/train_raw.jsonl
  python -m tasks.vqa.data --annotations-dir /path/Annotations_val.zip --images-dir /path/JPEGImages --xml-dir /path/Annotations --split val --out data/vqa/val_raw.jsonl
  python -m tasks.vqa.data --clean-splits data/vqa/train_raw.jsonl data/vqa/val_raw.jsonl --out-dir data/vqa
核对原始 JSON 字段、图片 ID/解码尺寸、文件/像素哈希及 XML 描述，不用 XML 生成数量或不存在标签。原 train/val 父影像前缀隔离；地理位置身份未知，不声称地理完全隔离。去除文件名与 image 字段不一致的标注、同图重复问答和验证侧与训练像素相同的影像。部分错误 type 按问题修正，原问题与答案不变。
正式清洗划分为 train 85804 QA / val 37392 QA。词表仅由训练集建立，min_answer_count=2。

正式训练：
  bash scripts/train_vqa.sh
默认读取 ../vqa_work_20261011/data/train_final.jsonl 和 val_final.jsonl，使用 config/vqa/qformer.json 及 ../vqa_work_20261011/cache/full，输出 runs/vqa/qformer。可用 TRAIN_JSONL、VAL_JSONL、CONFIG、DEVICE、SAVE_DIR 覆盖；已有 manifest 时拒绝覆盖，重复实验须指定新的 SAVE_DIR。
种子 42，batch 128，最多 15 epochs，早停 patience 4，AdamW（lr=0.001、weight_decay=0.0001），题型均衡有放回采样，答案 CE 加 0.5 题型 CE，余弦学习率；按完整验证集上的五类宏平均选模。正式训练不得设置 max_train_batches 或 max_val_batches 为正值。
资产路径从启动时工程根目录解析：../HFSA_models/pretrain_model/yolov12m.pt 和 ../HFSA_models/openclip/ViT-L-14.pt。复用已有资产和冻结缓存，不安装新框架或下载新权重。其它机器须准备对应数据与资产并配置路径。

全量评估：
  BUNDLE_DIR=runs/vqa/qformer OUT_DIR=runs/vqa/qformer/evaluation bash scripts/eval_vqa.sh --cache-dir ../vqa_work_20261011/cache/full
默认 val 为 ../vqa_work_20261011/data/val_final.jsonl，完整分母为 37392。BUNDLE_DIR、VAL_JSONL、VISUAL_BACKBONE、TEXT_ENCODER、DEVICE、OUT_DIR 可覆盖。在线路径逐条执行；--cache-dir 可批量评估，校验数据/图片哈希、外部资产和预处理。--ablations --formats 增加输入移除消融与公共 QA 构造选项的格式诊断。未知、失败、不支持均计入完整分母；全 QA 与五类子集分别报告，不能将未实现题型从总分母删除。

API 完整训练门禁：
  manifest 的 architecture_id 必须为 hfsa_frozen_qformer_vqa_v2，fusion_backend 必须为 blip2_qformer，evaluation_status 必须为 complete。
  training_result.json、config/train.json、evaluation/metrics.json 必须通过 manifest 的 SHA-256 校验。validation_scope=full_split；训练与验证批次数限制均为 0；config 的融合结构为 blip2_qformer；epochs_completed 为正整数，best_epoch 为有效整数。
  evaluated_val_qa 为正整数且等于 metrics.n；checkpoint/dataset SHA-256 必须是非空 64 位小写十六进制，并分别绑定正式权重与 val_jsonl。全量评估的 first_stage_macro_accuracy 必须有限且与训练选模的 best_macro_accuracy 一致。未完成、有限 batch 或绑定不一致的产物在模型构造前拒绝。

本地接入：
  from tasks.vqa.inference import VQAPredictor
  predictor = VQAPredictor('runs/vqa/qformer', device='cpu', asset_paths={'visual_backbone': '../HFSA_models/pretrain_model/yolov12m.pt', 'text_encoder': '../HFSA_models/openclip/ViT-L-14.pt'})
  result = predictor.predict(image, question, choices=None, allowed_answers=None)
image 为 PIL、str 或 Path。构造时加载一次，严格核对配置、词表、state_dict、权重和外部资产哈希，可连续预测。短文本返回英文答案；单选将可唯一映射到已训练词表的候选按对应答案头概率比较，返回原选项键。部分词表外候选不会导致整题拒答，本地 choice_coverage 记录 provided、matched、unmatched_keys；未知候选无法选中，也不自动当作“都不是”。全部候选未知或已知候选别名重复仍失败。枚举要求所有值可唯一映射，保留允许值大小写。confidence 为原完整答案头概率，不对剩余候选重新归一化；不用 CLIP 相似度或默认选项。规范化包含 lowercase、空白、末尾标点及 grey->gray，多色答案不压缩为单色。

错误契约：
  空问题、非法选项或冲突模式抛 ValueError；缺文件、资产哈希或 checkpoint 不匹配在构造时报错。unsupported_task、unsupported_language、choice_unmatched、text_context_exceeded 等返回 answer=null；API 将不支持/失败映射为 HTTP 500 inference_failed，不回退分类或猜答案。双图、integer、bbox 不属于当前 VQA 接口。模型概率不是校准准确率。

核验命令：
  python -B -m unittest tasks.vqa.test_contract tasks.vqa.test_qformer -v
  python -B -m unittest discover -s participant_api_starter-main/tests -p test_vqa_api_current.py -v
  python -B scripts/verify_vqa_qformer.py --bundle-dir runs/vqa/qformer --jsonl ../vqa_work_20261011/data/val_final.jsonl --cache-dir ../vqa_work_20261011/cache/full --device cpu --out runs/vqa/qformer/reports/online_verification.json
  python -B scripts/verify_vqa_api.py --bundle-dir runs/vqa/qformer --device cpu --output runs/vqa/qformer/reports/api_integration_verification.json
核心测试包含官方六类 AST 对应、同权重层输出/输入和参数梯度一致、查询专属视觉交叉注意力、新增参数梯度、图像/问题敏感性和保存重载。online verifier 从明确传入的 val_final.jsonl 选取五种来源题型并检查在线/缓存对照、三种格式及冻结状态；这是样本一致性核验，不替代 37392 QA 全量评估。
API verifier 只使用既有公共样例图片/问题，用当前模型直接推理结果作为 HTTP 一致性参考，不使用旧预测或金标。固定 VQA 任务组调用实际 adapter 执行路径；另有真实 Qwen 任务路由组，路由到非 VQA 的样例单独记录为 skipped。临时服务仅监听随机本机端口，使用临时随机认证值，关闭请求捕获；检查鉴权、图片哈希、三种格式、五类输出、不支持情况、双图拒绝、模型对象复用和冻结状态。--skip-live-qwen 仅用于明确省略真实路由核验，报告会注明。该脚本不重启9001、不登记官网。

正式结果记录：
  正式15轮完成，最佳第14轮，训练循环697.54056s。checkpoint SHA-256为8f6ed01a42a22c7f82ff410ba4a5e6a8e56da79274b05d0eb4896408fc8cc706。全37392 QA准确率35.643988%、有效58.039153%；五类22413 QA宏平均53.730459%、总体56.837550%、未知4.528622%。旧版同分母宏平均54.378791%、全QA36.379439%，新版下降0.648332/0.735451个百分点，未获得提升；这是选模所用开发val，不是独立test或官网成绩。
  CPU在线/缓存7组通过；真实权重固定任务HTTP21组（16成功/5预期失败）、真实千问HTTP11组（7成功/4预期失败，10非VQA路由样例skip）与直接输出一致，13项检查全通过。22项核心/契约、15项封包、177项API回归通过。manifest已封包complete；原始配置/权重和完整训练/指标/逐题验证收据均哈希绑定。报告位于runs/vqa/qformer/reports及evaluation。接口一致性不代表模型准确率或官网分数。
  用户已于2026-10-11 18:48重启正式API，当前工程server进程PID399620监听9001；18:57官网smoke的6次成功请求确认本run认证/公网连通和预测工作。当前入口仅接受完整验证的Q-Former v2；来源/部署证据见根目录CURRENT_STATE。进程未暴露独立内存checkpoint SHA，助手未读取Key或终端输出。上述原本地验收报告是训练验收时快照，当前运行状态以CURRENT_STATE为准。
  本地推理不需要训练数据、缓存、API或联网，使用工程现有 HFSA 依赖。requirements-vqa.lock.txt 的注释版本只记录已有环境，不是依赖安装清单。
