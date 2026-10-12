# 模型 API 协议 2.0（选手实现规范）

本规范描述评测网站如何调用参赛队伍自行部署的模型 API。开发集与正式集使用相同的单题协议；二者的图片包和 `dataset_id` 不同。协议版本固定为字符串 `"2.0"`，与开发数据集版本 `2.0.0` 不同。

## 1. 数据流和边界

1. 组委会**另行交付**图片包；队伍在模型 API 所在机器预置并校验。图片包的首次交付可以经过网络，本协议仅保证**评测时的单题 HTTP 请求不携带图片字节或图片 URL**。
2. 网站每题发送题干、作答约束、图片 `asset_id` / SHA-256 / MIME 类型。队伍从本地 `assets/<asset_id>` 读取图片；网站不发送金标、原始图片路径或内部题图映射。
3. 模型 API 在队伍自己的算力上推理，返回答案。网站完成评分。开发站目前是 100 题，`smoke` 选取 8 题；开发成绩不等于正式赛成绩。

选手 API 只需实现 `GET /healthz` 和 `POST /v1/predict`。可以用本工程的 `server.py` 作为外壳，也可以用任意语言或框架实现。网站不是 OpenAI 兼容 `/v1/chat/completions` 客户端，不能直接填该路径。

## 2. Endpoint 与认证

在网站填写模型 API 的公网 HTTPS **origin**，例如 `https://model.example.org`。不要附加路径、末尾斜杠、查询参数或凭证。证书须可公开验证；localhost、内网地址和裸 IP 不适用。公网服务器配反向代理，或能提供稳定 HTTPS 域名的隧道，均可使用。

网站登录令牌和模型 API Key 是两种独立凭证。队伍自己生成模型 API Key，两个接口均校验请求头：

```http
Authorization: Bearer <MODEL_API_KEY>
```

建议使用至少 32 位随机值。请勿写进 URL、仓库、日志或截图。网站登录后可自行保存 Endpoint；提交评测时填写模型 API Key。修改 Endpoint 本身不消耗评测次数。

## 3. 图片包结构

开发图片包解开后至少有：

```text
<package>/
  manifest.json
  assets/
    <32 位小写十六进制 asset_id>
    ...
```

开发包 `dataset_id` 为 `rsu-dev100-v2`。`manifest.json` 的资产条目形如以下**合成示例**（编号、哈希、大小不是实物数据）：

```json
{
  "format": "rsu-sealed-images-v1",
  "dataset_id": "rsu-dev100-v2",
  "assets": [
    {
      "asset_id": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
      "sha256": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
      "size": 12345,
      "mime_type": "image/png"
    }
  ]
}
```

`assets` 是图片资产清单，**不是**题目或答案。每张图按 `asset_id` 存放，收到请求时应校验 SHA-256（以及清单中的大小），拒绝不存在或不匹配的图片。双图题按请求 `images` 数组顺序送入模型，不能交换。不要将 `asset_id` 拼接为任意客户端可控路径；应先验证其格式并从清单查找。正式包会使用不同的 `dataset_id`，请从本地清单读取，勿硬编码开发包编号。

## 4. 健康检查：`GET /healthz`

评测入队前，网站调用 `<Endpoint>/healthz`，携带 Bearer Key 和 `Accept: application/json`。成功时返回 HTTP 200、`Content-Type: application/json`，JSON 对象至少包含：

```json
{
  "status": "ready",
  "protocol_version": "2.0",
  "dataset_id": "rsu-dev100-v2"
}
```

网站核对 `status == "ready"`、协议版本以及当前图片包 `dataset_id`。预检失败时不入队，也不消耗提交次数。不要在图片尚未加载或模型未就绪时谎报 `ready`。

## 5. 预测：`POST /v1/predict`

请求头包含 Bearer Key、`Content-Type: application/json`、`Accept: application/json`。下面是**合成示例**；实际值以网站发送的请求为准：

```json
{
  "protocol_version": "2.0",
  "request_id": "example-run:EXAMPLE_0001:1",
  "item_id": "EXAMPLE_0001",
  "question": "图中是否有建筑物？",
  "images": [
    {
      "asset_id": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
      "sha256": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
      "mime_type": "image/png"
    }
  ],
  "response_constraint": {"type": "enum", "values": ["Yes", "No"]},
  "answer_type": "short_text"
}
```

| 字段 | 说明 |
| --- | --- |
| `protocol_version` | 固定字符串 `"2.0"` |
| `request_id`、`item_id` | 本次请求及题目标识；响应必须原样回显 |
| `question` | 题干文本 |
| `images` | 顺序数组，1–2 项；每项包含 32 位小写十六进制 `asset_id`、64 位小写十六进制 SHA-256 和 `mime_type`（`image/jpeg`、`image/png` 或 `image/webp`） |
| `response_constraint` | 作答约束；详见下一节 |
| `answer_type` | 控制 `answer` 的 JSON 类型：`bbox` 为数组，其余为字符串 |
| `choices` | 仅单选题出现；选项标识到选项文本的映射 |
| `image_width`、`image_height` | 仅 bbox 题出现；原图像素宽高 |

成功响应必须是 HTTP 200、`Content-Type: application/json`，且整个响应是 JSON 对象：

```json
{
  "protocol_version": "2.0",
  "request_id": "example-run:EXAMPLE_0001:1",
  "item_id": "EXAMPLE_0001",
  "answer": "Yes"
}
```

`protocol_version`、`request_id`、`item_id` 必须与请求完全一致。非 bbox 的 `answer` 必须为最多 4096 字符的字符串；bbox 则为四个有限数字的 JSON 数组，满足 `x1 < x2`、`y1 < y2`。网站对响应大小设 1 MiB 上限；默认单次请求超时 120 秒，实际站点配置以 `/help` 为准。返回非 200、非 JSON、重定向、错误标识或错误答案格式，都会使该题按缺失预测处理。

## 6. `response_constraint` 与答案类型

| `response_constraint.type` | 附加信息 | 应返回的 `answer` |
| --- | --- | --- |
| `enum` | `values`；可能有 `case_insensitive`、`question_form` | `values` 中的字符串。`question_form=change_region` 时 `answer_type=region_label`，仍返回字符串 |
| `single_choice` | `values` 和顶层 `choices` | 选项标识字符串，如 `"B"`；不要返回选项全文 |
| `short_text` | 可能有 `min_length` | 简短文本字符串 |
| `integer` | 可能有 `minimum` | 整数的**字符串**，如 `"3"`，不是 JSON 数字 |
| `bbox` | `coordinate_format=pixel_xyxy`、`length=4`，以及顶层图片宽高 | 原图像素坐标数组 `[x1,y1,x2,y2]` |

注意：`response_constraint.type` 是任务约束，`answer_type` 是网站对返回值的类型要求；两者不总是同名。bbox 不能直接返回 0–1 或 0–1000 归一化坐标，须转换成原图像素坐标。例如 1000×800 图像上的一题（仍为合成示例）：

```json
{
  "protocol_version": "2.0",
  "request_id": "example-run:BBOX_EXAMPLE:1",
  "item_id": "BBOX_EXAMPLE",
  "question": "框出图中的目标。",
  "images": [{
    "asset_id": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    "sha256": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
    "mime_type": "image/png"
  }],
  "response_constraint": {"type": "bbox", "coordinate_format": "pixel_xyxy", "length": 4},
  "answer_type": "bbox",
  "image_width": 1000,
  "image_height": 800
}
```

对应的成功响应：

```json
{"protocol_version":"2.0","request_id":"example-run:BBOX_EXAMPLE:1","item_id":"BBOX_EXAMPLE","answer":[100,80,500,400]}
```

格式合法不保证内容正确；枚举值不在 `values` 内、选择了错误的选项等，可能按错误答案计分。

## 7. 最短闭环与排错

1. 将组委会提供的图片包解到**运行 API 的机器**，检查 `manifest.json` 和 `assets/` 同级，核验清单和图片。
2. 用 `python3 -m unittest discover -s tests -v` 跑本工程的合成测试；再按 [README.md](README.md) 启动 `server.py`，或启动自己实现的 API。
3. 用独立随机 Key 从外部网络访问 `https://<队伍域名>/healthz`，确认返回 200、JSON、正确 `dataset_id`。可用本工程的 `check_api.py` 做不计分预测检查。
4. 登录评测网站，保存 HTTPS Endpoint；先跑 8 题 `smoke`，确认响应回显和答案格式，再跑完整开发集。

如果预检报 `dataset_mismatch`，先查本地打开的图片包；401/403 查 Key；404、HTML 或重定向查反向代理和 Endpoint 是否只填 origin；422 查图片编号、路径、SHA-256；超时查公网连通、隧道、模型加载及推理耗时。推理异常应返回非 2xx 的 JSON 错误，不要伪造答案。评测结束前保持服务在线。

当前开发站每队每日 `smoke` 最多 10 次、完整开发集最多 2 次（按 UTC 日期重置）；入队后的失败提交也计数。每队同时只允许一个排队或运行中的任务。**实际运行限额以网站 `/help` 显示为准。**
