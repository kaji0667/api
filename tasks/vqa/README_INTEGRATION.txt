HFSA 独立视觉问答模块：Q-Former v4（2026-10-14正式部署冻结）

配套源码：tasks/vqa；模型包：runs/vqa/qformer。
主权重：weights/vqa_head.pt
SHA-256：2858c240b620915b9958ee3fa3f63f0bc01bc5c0c151b75f247852c2f551bf8b
任务头参数：5,925,900。十二类残差答案头，类型LayerNorm；与v4词表/配置/state_dict严格配套。
冻结视觉YOLOv12m和OpenCLIP ViT-L-14/openai；32查询、2层、4头、192维的BLIP-2 Q-Former适配结构。LAVIS核心来源revision 05cb670f47296522e24e7727e366b4a997d28582；未导入BLIP-2/BERT/LLM预训练权重。原论文分数不能作为本模型成绩。

输入仍为一张原图、完整英文问题、原选项或枚举约束。640整图缩放、16/8/4多尺度网格、ImageNet归一化；长问题CLIP按75内容token连续分块和加权汇总。无ROI/切片监督。有限词表覆盖不保证开放位置描述能匹配；未知或冲突选项、非法约束、中文等继续显式失败。API分类/检测/计数/presence/定位继续独立路由。

bundle manifest绑定全部权重、词表、训练配置与完整37392验证QA收据；严格完整训练门禁保持。bundle中的历史训练绝对路径只作来源记录，运行时API显式传入工程内pretrain_model/yolov12m.pt和openclip/ViT-L-14.pt，不能改写已绑定训练记录。

三版比较：旧开发53条去重原请求，42人工确定、11不确定；先看图冻结人工参考，再查看真实API结果。三版确定题语义正确均14/42；真正VQA执行器29确定题均12/29。旧版VQA有效22/35，v3/v4均27/35；v4本次平均耗时2.155秒，v3为2.273秒，依约选择v4。sport field/sports field语义等价只影响一个v4结果；严格字符串副表也保留。小样本和单次顺序测试不能当正式分数或显著性结论，详见上级VQA版本对比.txt。

原五头源码/完整包已私有完整备份，工程runs/vqa/qformer-five-head-before-20261014保留原包。v3/v4来源备份在工程外。服务只使用当前qformer，不自动模型回退。

核验：python -B -m unittest tasks.vqa.test_contract tasks.vqa.test_qformer
API回归：python -B -m unittest discover -s participant_api_starter-main/tests
正式启动：先在运行环境设置MODEL_API_KEY，再bash scripts/serve_hfsa_api.sh。默认正式package根目录；VQA CPU，其余任务auto；需要鉴权health ready/2.0/rsu-formal800-v3。服务实际内存与源码/权重冻结收据在私有formal-release-20261014。
