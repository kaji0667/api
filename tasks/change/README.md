# 预训练双时相变化模块

## 2026-10-12 通用最大变化区域分支

当前保留下文原MCI，并新增作者 [ChangeFormerV6 DSIFN-CD](https://github.com/wgcban/ChangeFormer) 二值变化模型。通用最大变化区域用独立二值mask；明确道路/建筑变化区域、caption及全图是否变化继续用MCI。没有重训或修改YOLO/OpenCLIP/VQA/定位。

固定源码v0.1.0 revision `5ca45352d06e15a01e2d6525d891f9c5bbf54f52`，41,026,674参数，373张量strict加载。原archive 985,645,254bytes；抽取纯state 164,246,664bytes，SHA256 `e983f39209952cf7b3e341b54f4ff48115fcba979542d61bc5193c270396acfc`。完整来源/原文件hash/安全pickle审计在bundle与准备脚本；训练元数据只在独立准备过程以最小数字allowlist、weights_only=True读取，服务仅加载纯tensor，不允许strict=False/不安全兼容。release README要求研究/非商业用途，后续main的MIT LICENSE单独保存，不宣称统一无限制许可。

默认bundle `runs/change/changeformer_dsifn`，API独立配置 `HFSA_BINARY_CHANGE_BUNDLE_DIR` / `HFSA_BINARY_CHANGE_DEVICE=cpu`，一次构造复用且监听前完成校验。输入同尺寸、A前/B后；RGB/PIL bicubic256→ToTensor→mean/std0.5，末尺度二类logits argmax，不配准、不切片、无TTA。大图缩小和跨域能力仍有限。

支持合成例句 `Which region has the largest change area?`、`Which region shows the most significant change?`、`Which quadrant has changed the most?`、`哪个区域变化最明显？`；完整句法匹配不会丢弃未知限定。必须给完整四象限或九格enum/single_choice候选，按预测变化像素面积选唯一正最大，返回原值/键。九格top/bottom/left/right指边中格；四象限题干与九格候选冲突拒答。“显著”是面积近似，2×2/3×3几何属部署约定，尚未证明官网语义。无/并列变化、缺失/重复/未知区域、最小、数量、变化比例/密度、新增/拆除子集、特定地类/额外限定拒答；二值模型无类别/方向/计数头，不使用caption猜区域。

既有WSL环境中复现：

```bash
cd /mnt/d/code/python/HFSA/提交/HFSA-main
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 ../../hfsa_env/bin/python -B scripts/fetch_binary_change.py
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 ../../hfsa_env/bin/python -B -m unittest tasks.change.test_inference tasks.change.test_region_revision tasks.change.test_binary_change
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 ../../hfsa_env/bin/python -B scripts/verify_binary_change.py
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 ../../hfsa_env/bin/python -B scripts/verify_binary_change.py --preflight-only
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 ../../hfsa_env/bin/python -B scripts/verify_binary_change_api.py
```

最后一条使用已有本地Ollama、随机Key/端口和临时公开图片包；每次HTTP真实推理，不用生产Key/9001或官网。模型数值验证、GT指标和HTTP证据一致性分别记录，GT不会进入模型。最终API218项+变化66项=284合成回归通过；preflight11/11含精确A/B/GT路径绑定及公开派生319×287预处理误差0。4个预测前固定作者demo的输入、5尺度logits、mask与原作者全部相同；前景IoU0.65789、F1 0.79365，四/九区唯一最大各4/4，同图变化0。真实Qwen路由HTTP8/8与独立预测mask胜者一致。demo与预训练split独立性未知，不称完整基准/官网提分。CPU4线程单次约0.75–0.83s；双模型验证进程峰值RSS约1.47GB，不代表生产增量。报告在 `runs/change/changeformer_dsifn/reports`，原MCI报告保留。

服务脚本现默认保存安全diagnostic到 `${XDG_DATA_HOME:-$HOME/.local/share}/hfsa/prediction-diagnostics.jsonl`，字段含UTC/PID/task/实际executor/固定reason/耗时，不含题干/选项/答案/框/凭据/异常文本；显式 `HFSA_API_DIAGNOSTIC_FILE=''` 可关闭。代码完成不代表生产已切换：在原API终端沿用已export的Key，Ctrl+C后 `bash scripts/serve_hfsa_api.sh`，等API listening。

## 原MCI分支（保留）

本模块接入作者 [Change-Agent MCI（TGRS 2024）](https://github.com/Chen-Yang-Liu/Change-Agent) 的实际预训练模型，输出英文变化caption和道路／建筑变化mask。它是独立补充模型，没有更换现有HFSA的YOLO/OpenCLIP或VQA，也不声称是2026当前最佳模型。

权重：作者 [lcybuaa/Change-Agent](https://huggingface.co/lcybuaa/Change-Agent)，固定revision `1e679060173434cd22fd22a52effbf3c785c97a0`；913,762,855 bytes，SHA256 `34c6926342c40fdd6d50b43d50257cb46cb6b61763448de9ee551828a64b3eb9`。默认模型包为工程内 `runs/change/mci`，权重在 `weights/MCI_model.pth`。源码固定 `68cbaa7f388b36e4fc10872f7a2911482d26ae5b`，必要原文件、词表、许可与补丁记录在 `third_party/change_agent`；不复制整个上游工程，不安装mmcv/mmseg/einops或升级当前环境。

模型默认CPU、FP32，构造一次后复用；MiT-B1、3层双时相注意力融合、512维／8头、1层caption decoder、468词表、greedy k=1。源码结构参数130,502,679。mask是256×256的三类：0背景、1道路变化、2建筑变化；没有新增／拆除方向标签，也没有通用物体计数或任意地类转换。方向问题只使用caption明确表达的语义。模型需要同覆盖且配准的图像；同尺寸检查不证明已经配准。

官方checkpoint共2136个state，当前公开推理图需要1194个，全部严格加载。另942个fusion键在作者公开图中无模块，作者predict/test以strict=False排除；本模块要求固定权重/源码SHA及exact-key/shape/dtype白名单全部吻合后才排除，任何当前键缺失或新额外键均失败。白名单为`checkpoint_inactive_keys.json`；真实权重的原作者/适配流程在两公开图片对和同图控制中全部特征/logits误差0，mask/token完全一致，证据为`reports/checkpoint_compatibility.json`。不推断未公开训练图是否使用这些参数。

预处理保留原测试的RGB／float32、均值`[.39073,.38623,.32989]*255`及标准差`[.15329,.14628,.13648]*255`，Python乘255后转float32，逐值与原CHW处理一致。原数据为256²；额外部署尺寸先在HWC以OpenCV linear缩放到256，再转CHW，修正作者predict脚本非256时在CHW上resize的问题。不自动旋转EXIF。大图缩到256会丢失细节，目前无滑窗／配准网络。

`ChangePredictor.evidence(before, after)`仅产生真实模型输出。`predict(before, after, 原题, response_constraint=..., choices=...)`保留原题，随后本地EvidenceAnswerer使用既有Ollama Qwen对caption证据作受约束解释。它区别于仍只选task的`ollama_router.py`；不调用在线模型。只接受short_text、enum、single_choice，单选返回原键，枚举返回原值；不支持变化bbox/integer。最大变化区域按真实预测mask的像素面积计算，只支持完整四象限或完整3×3网格的明确标签；无变化、并列、未知标签/几何或无法确定目标范围时拒答。caption本身可能预测错误，文本解释也可能错误。

区域选项或`question_form=change_region`强制进入独立区域分支，不能由Qwen从caption直接选区域。只有`answering.maximum_region_focus`的有限完整句法能确认最大道路、建筑或二者并集变化面积时，直接按真实mask计算，Qwen不参与该答案。例如`Which quadrant has the most building change area?`和`哪个区域的道路与建筑变化面积最大？`。最小面积、新增/移除子集、物体数、其它地类、含未识别限定或模糊问法拒答；完整四/九个选项只确定几何，不能自行证明题意。其余caption路径只允许answer/unsupported操作。

全图是否变化采用`answering.global_change_query`的有限完整问法，例如`Do the earlier and later images show any change? Answer Yes or No.`和`前后两幅图像是否有变化？只回答是或否。`；明确无变化caption/肯定变化谓词直接映射原选项或Yes/No，不经Qwen。未知/模态/混合否定caption或具体事件二元问题拒答，未提及事实不等于No。明确数量query提前拒答。Qwen仅用于其余描述/多类映射，收到的视觉证据只含caption和能力范围，不含mask/像素统计；模型caption和文本解释仍可能错误。首轮真实公开验证的无变化误答和像素数误当物体数报告保留，未改模型/图片/阈值凑结果。

WSL工程目录中复现下载及验证（既有环境）：

```bash
cd /mnt/d/code/python/HFSA/提交/HFSA-main
../../hfsa_env/bin/python -B scripts/fetch_change_mci.py
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 ../../hfsa_env/bin/python -B -m unittest tasks.change.test_inference
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 ../../hfsa_env/bin/python -B scripts/verify_change_api.py
```

最后一条使用本机Ollama和预先固定的4对公开LEVIR-MCI test图片，不操作官网或生产9001。数据清单为`runs/change/public_validation/data_manifest.json`，GT只评分；该小集只用于接入与语义抽查，不替代完整测试。实际结果与加载收据在`runs/change/mci/reports`，以已完成报告为准。

2026-10-11最终验证：250项合成测试通过（API204+变化46）；真实模型公开binary4/4、A/A正确No，含背景三类mask mIoU91.8925%（道路91.2143%、建筑86.8940%）。22项检查及22次实际临时HTTP通过，含预期拒答/鉴权失败、真实Qwen路由和未缓存完整真实视觉请求。该未缓存HTTP耗1.3474s，其中视觉.21094s；验证进程峰值RSS约1.489GiB，不含现有API/Ollama。其它缓存HTTP时间不能当每次新视觉推理。报告保留原失败尝试。反向A/B的caption出现错误same-scene、mask仍有变化，不能据此声称方向理解可靠；大图缩到256的限制保留。

API默认 `HFSA_CHANGE_BUNDLE_DIR=runs/change/mci`、`HFSA_CHANGE_DEVICE=cpu`；原服务须重启才加载新模块。在原API终端Ctrl+C后运行：

```bash
bash scripts/serve_hfsa_api.sh
```

沿用原终端已export的Key，不读取／打印／保存生产密钥。已运行Ollama/ngrok可继续使用。源码和权重接入通过不等于9001已切换。

许可分别记录：仓库LICENSE是MIT，README另要求academic purposes only；权重HF card标Apache-2.0；LEVIR-MCI数据卡未单独声明许可。不将这些记录合并成统一许可，也不再分发完整数据集。本地公开小样例用于科研验证。
