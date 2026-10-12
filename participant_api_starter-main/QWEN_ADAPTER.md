# Qwen3.5-4B 适配示例：把本地视觉模型接入评测 API

此示例从组委会本地 Qwen3.5-4B 基线的推理逻辑抽出最小的模型适配层。它与 `server.py` 共用**协议 2.0**：网站发送题干和本地图片编号，`server.py` 核验图片包并给适配器传入图片路径，`qwen_adapter.py` 才在队伍 GPU 上读取图片、调用模型并返回答案。这里没有模型权重、赛题、图片、金标和实际密钥。

Qwen 只是可运行的参考实现，**不要求参赛队伍使用 Qwen**。其他 VLM 可以保留 `server.py` 与图片包逻辑，只替换 `model_adapter.py` 中的 `RealModel`，或以 `qwen_adapter.py` 为模板改写模型加载和生成部分。

## 一、准备环境和本地模型

需要支持 PyTorch 的 CUDA GPU、Python 3.10+ 与本地 Qwen3.5-4B 模型目录。模型目录必须包含 `config.json`、处理器/分词器文件和完整权重；示例通过 `local_files_only=True` 加载，启动时不会临时从公网下载权重。可从 [Qwen 官方模型页](https://huggingface.co/Qwen/Qwen3.5-4B)或[ModelScope 上的 Qwen 模型页](https://modelscope.cn/models/Qwen/Qwen3.5-4B)按其说明自行获取。请遵守模型自身许可和部署要求。

组委会本地基线固定的是 Qwen3.5-4B 的修订 `28657e97862998ba5b263fb68deb417a81785653`，并使用 BF16、CUDA、SDPA 与固定的 Transformers commit。`requirements-qwen-reference.lock` 是**该本地环境的参考锁定文件**，其中 PyTorch wheel 为 CUDA 13.0；不同显卡驱动或 CUDA 环境可能需要按 [PyTorch 官方安装说明](https://pytorch.org/get-started/locally/)选择兼容的 wheel，再安装其余依赖。不要盲目把 CUDA 13.0 锁定文件当成所有队伍的通用环境。

如果本机适配 CUDA 13.0，可尝试：

```bash
cd participant_api_starter
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-qwen-reference.lock
```

`QWEN_MODEL_DIR` 必须指向**本机**模型目录，不能填评测网站地址、模型下载网页或 API URL。模型权重建议放在这个 Git 仓库之外；本仓库也忽略 `models/` 以防意外提交。

## 二、启动真实模型 API

先把组委会提供的开发图片资源包解到运行 Qwen 的同一台机器；目录结构参见 [主 README](README.md#1-整体工作方式)。之后：

```bash
cd participant_api_starter
read -rs -p 'MODEL_API_KEY: ' MODEL_API_KEY; echo
export MODEL_API_KEY
export MODEL_MODE=qwen
export QWEN_MODEL_DIR=/absolute/path/to/Qwen3.5-4B
.venv/bin/python server.py --package /absolute/path/to/dev-image-package --host 127.0.0.1 --port 9001
```

服务器启动时加载一次 Qwen 模型；每道题仅调用 `predict()`，不会重载权重。`MODEL_API_KEY` 是队伍自己设置的至少 32 位随机密钥，不是网站登录令牌，也不是 ngrok authtoken。启动报 `CUDA is not available`、缺少 `config.json` 或依赖错误时，先修复本地环境，不要把演示模式作为真实评测结果。

另一个终端设置**同一个** `MODEL_API_KEY`，检查本地完整 API：

```bash
cd participant_api_starter
read -rs -p 'MODEL_API_KEY: ' MODEL_API_KEY; echo
export MODEL_API_KEY
.venv/bin/python check_api.py \
  --origin http://127.0.0.1:9001 \
  --package /absolute/path/to/dev-image-package
```

`check_api.py` 提交一张本地图片的**合成题**，只检验接口、图片读取和推理流程，不产生开发集成绩。随后按主 README 的 ngrok 或自有 HTTPS 步骤从公网检查，再在评测网站先跑 8 题 `smoke`、后跑 100 题 `full`。公网 URL 只需要是 API origin，不附加 `/healthz`。

## 三、四个适配点分别做什么

| 适配点 | Qwen 示例中的位置 | 换成自己模型时 |
| --- | --- | --- |
| 模型只加载一次 | `QwenModelAdapter.__init__` | 加载权重、processor/tokenizer，设置设备与推理模式 |
| 题干和约束转提示词 | `prompt_for_request()` | 把题干、候选项、答案格式写入自己模型的 prompt；双时相顺序不能反转 |
| 本地图像推理 | `QwenModelAdapter.predict()` | 按 `image_paths` 顺序读图和生成；不要从网络下载图片 |
| 模型文本转协议答案 | `parse_generated_answer()` | 对枚举、单选和整数输出做校验；bbox 转回原图像素坐标 |

`server.py` 管理的是 HTTP、Bearer 认证、`request_id`/`item_id` 回显和图片 SHA-256 核验。适配器只需要实现：

```python
def predict(self, request: dict, image_paths: list[Path]) -> str | list[int | float]:
    ...
```

`image_paths` 是 `server.py` 已核验的本地文件路径，顺序与 `request["images"]` 一致。`request["question"]` 是题干；`request["response_constraint"]` 规定输出类型；单选题还有 `request["choices"]`。不要在适配器中创建额外的 `/v1/predict` 服务、索取金标或上传图像。

### Qwen 的提示词和答案处理

- 枚举题只输出允许值；`Yes`/`No` 会按允许值规范化。单选只输出候选字母，不返回选项全文。
- `integer` 返回十进制非负整数**字符串**；短文本返回单行字符串。
- 变化理解的双图题：第一张为变化前/Image A，第二张为变化后/Image B。按数组顺序构造模型输入。
- 本地 Qwen 基线提示模型用 **0–1000** 归一化坐标回答 bbox。例如模型输出 `[100,200,900,800]`，若原图为 `200×100`，API 返回原图像素框 `[20,20,180,80]`。评测网站要求的是原图像素坐标，不是归一化坐标。其他模型若直接生成原图像素框，可设置 `QWEN_BBOX_COORDINATES=pixel_xyxy`，或在自己的适配器里直接返回像素框。
- 若模型没有给出可解析的答案，示例抛出错误，HTTP 层返回非 2xx；它不会用猜测答案掩盖适配故障。错误题会计为缺失预测。

## 四、怎么换成自己的 VLM

最小改动是在 `model_adapter.py` 的 `RealModel` 中实现一次性加载和 `predict()`，然后 `export MODEL_MODE=real`。可借鉴 Qwen 示例的提示词构造与答案解析，但要根据自己的模型调整 processor 调用、精度、设备、最大输出长度和 bbox 坐标系。尤其注意：

1. **模型输出不一定等于协议输出**。不要直接返回整段自然语言或 Markdown；单选是字母，bbox 是四个数字的数组。
2. **图片包是本地的**。每题请求只给 `asset_id` 和哈希，没有 URL；不要把 API 设计成接收上传文件或从网站拉图。
3. **资源与时延**。示例为单 GPU 串行推理；你可以自行优化批处理和并发，但要先确认显存充足、每题能在网站超时时间内完成。
4. **独立检查**。先运行 `python3 -m unittest discover -s tests -v`，再用本地 `check_api.py`、公网 `check_api.py`，最后才消耗网站评测次数。

这个独立仓库的 `qwen_adapter.py` 是给选手阅读和修改的薄适配层；组委会完整的 Qwen 基线还包含更严格的图像像素上限、环境审计和批量结果报告，不应把这个示例误认为生产级推理服务。
