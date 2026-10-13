# BLIP-VQA 公开独立试验

`tasks.vqa.blip_trial` 只用于公开样本上的完整预训练模型对照，不参与 API、Web、正式训练或默认模型分派。当前生产 VQA 继续使用 v4 十二头包。用户在侧对话中选择先试 BLIP-VQA，本次授权仅覆盖这一独立试验。

模型为 [Salesforce/blip-vqa-base](https://huggingface.co/Salesforce/blip-vqa-base)，固定 revision `787b3d35d57e49572baabd22884b3d5a05acf072`，模型卡声明 BSD-3-Clause。只使用官方 safetensors，不加载远程自定义代码或 pickle 权重。文件大小 1,538,800,584 字节，权重 SHA256 `33786eed34def0c95fa948128cb4386be9b9219aa2c2e25f1c9c744692121bb7`。完整模型包括视觉编码器、问题编码器与答案生成器，没有借用 HFSA 的 YOLO/OpenCLIP。

下载同版本的 README.md、config.json、preprocessor_config.json、special_tokens_map.json、tokenizer.json、tokenizer_config.json、vocab.txt 和 model.safetensors；本地模型目录须包含 `source_receipt.json`，其 `model_id`、`revision`、`files[{name,size,sha256}]` 记录来源和每个文件哈希。推理只读本地文件，检查全部哈希及 Transformers 加载的 missing/unexpected/mismatched/error 列表为空。

## 固定协议

- 原始公开 VRSBench val，每类按照 `SHA256(seed:sample_id)` 升序选 10 条，seed=20261013，共 120 条；不看答案值或预测挑样本。
- `prepare` 将完整题干、图像路径与身份哈希写入 `inputs.jsonl`，金标及金标题型另存 `gold.jsonl`。两个推理进程不读取 gold，不传 oracle type、候选答案或训练词表给 BLIP。
- 原图文件哈希和尺寸必须等于公开记录。解码像素哈希与来源不同时显式记录，冻结本机 RGB 哈希；两模型使用相同本机解码，不能声称复现了队友缓存特征。
- BLIP 使用官方全图 384×384 预处理、FP32、batch=1、beam=3、do_sample=false、max_new_tokens=20；问题不改写、不静默截断、不训练。当前 v4 使用原始 VQAPredictor 和配套完整模型包、YOLO/OpenCLIP 文件哈希，不改模型或解码。
- 首个固定输入仅用于预热计时，仍作为正式样本重跑。每条记录端到端模型耗时（包含模型预处理和生成，不含文件解码/哈希）；资源记录独立进程 RSS 和 GPU peak allocated/reserved。
- 两种模型全部完成并核对收据后才读 gold，以现有 normalize_answer 精确匹配，失败计错、分母不缩小，报告总体/十二类准确率、宏平均和成对 gain/loss。
- 本次 native=CPU、BLIP=RTX4060 CUDA，不据跨设备延迟宣称模型速度优劣。120 条均衡开发样本不代表完整 val、独立 test、官方成绩或原生单选能力。

## 运行

在具备 PyTorch、项目原依赖和 Transformers 4.48.3 的 Linux/WSL 环境运行。可用进程级 PYTHONPATH 复用已存在的隔离 Transformers 目录，避免改共享虚拟环境。准备阶段只依赖项目数据工具及 Pillow。

```bash
python -m tasks.vqa.blip_trial prepare \
  --jsonl ../vqa_work_20261011/data/val_final.jsonl \
  --out runs/vqa/blip_trial_20261013_111202 \
  --revision 787b3d35d57e49572baabd22884b3d5a05acf072

python -m tasks.vqa.blip_trial predict --backend blip \
  --model-dir pretrain_model/blip-vqa-base-787b3d3 \
  --out runs/vqa/blip_trial_20261013_111202

python -m tasks.vqa.blip_trial predict --backend native \
  --bundle-dir runs/vqa/qformer \
  --visual-backbone pretrain_model/yolov12m.pt \
  --text-encoder openclip/ViT-L-14.pt \
  --out runs/vqa/blip_trial_20261013_111202

python -m tasks.vqa.blip_trial score --out runs/vqa/blip_trial_20261013_111202
python -m unittest tasks.vqa.tests.test_blip_trial -v
```

已有选择、预测和完成收据拒绝覆盖；重做须换新的 out，重新固定协议。输出、模型和数据不提交源码 Git。没有自动安装依赖、启停服务、修改生产配置或消耗官网额度。

## 本次结果

用户确认效果不好不替换，并要求删除下载。官方权重与八个配套文件已全部移除，空模型目录也已删除；约释放1.54GB。当前生产VQA和服务配置没有切换；本试验代码及结果记录保留作失败对照证据。上述复跑命令需要重新提供匹配版本的模型资产，当前工程不再含BLIP下载。

2026-10-13，固定 120 个公开 QA / 120 张图全部完成，两路均无失败：

| 模型 | 正确数 | 均衡样本准确率/宏平均 | 设备 | 平均模型耗时 |
| --- | ---: | ---: | --- | ---: |
| 当前 v4 | 61/120 | 50.8333% | CPU，2 线程 | 0.3862 s |
| BLIP-VQA base | 49/120 | 40.8333% | RTX4060，FP32 | 0.0741 s |

BLIP 相对 v4 新增正确 17、丢失正确 29、双方正确 32、双方错误 42，净少答对 12，差 -10 个百分点。每类 10 条的 v4→BLIP 正确数依次为 category 0→2、color 5→5、shape 8→4、existence 10→8、scene 5→2、quantity 5→4、position 3→3、size 4→4、direction 2→2、comparison 6→3、relation 6→7、other 7→5。

BLIP 实际加载 384,672,572 参数，加载检查四项均空；GPU peak allocated 1,528.647 MiB、reserved 1,650 MiB，进程 peak RSS 2,117.477 MiB。v4 进程 peak RSS 5,913.289 MiB。冷加载分别 81.58/91.96 s（BLIP/v4），其中包含本机磁盘、文件校验、模块和模型加载，不是纯模型初始化基准。两设备不同，不作速度优劣结论。

公开原文件哈希和尺寸均匹配，108/120 张本机 RGB 哈希与交付元数据不同；差异已逐图记录，两路输入一致。没有证据将差异归因某个解码库，也不能将新 forward 当成队友缓存全量复现。三项专项协议测试通过；另行逐项复算选择、数据/金标绑定、源文件首尾哈希、完整 120 分母、分类型正确数、gain/loss、模型来源与完成收据，全部通过。

结论：这轮公开开发小样本不支持直接替换当前 VQA。没有在 API 中加载 BLIP，没有按本轮结果分题型混合两个模型，也没有新增训练或官网评测。下一候选可考虑具备指令微调的 BLIP-2/FLAN-T5 或 InstructBLIP；尚未测试，不能预先声称更好。

本机输出目录 `runs/vqa/blip_trial_20261013_111202/` 包含冻结协议、金标隔离文件、两路新预测、完成收据、comparison.json 和独立评分审计。输入 SHA256 `9b5012a00407497da9d5d5ab59e3addc5540e44cb53431c52bddaef25e6c618e`，comparison SHA256 `ea2b0849eadb0bdd097d817f8dd40e91733ade1f8c7e1c402f13734e717d5526`；当前 v4 head SHA256 `2858c240b620915b9958ee3fa3f63f0bc01bc5c0c151b75f247852c2f551bf8b`。这些结果只用于初步筛选，非完整 val、独立测试或官方成绩。
