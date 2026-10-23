# jevCabin · Qwen 车载智能助手

> 🎬 **在线 Demo**：<https://ranpin.github.io/jevCabin/>

车载座舱智能助手，专为汽车座舱场景设计：对话回答通过 OpenAI 兼容 API 调用 Qwen-2.5-Omni-7B；**信息提取使用 Jev 类型化决策模型**（本地 Laya 后端），从对话中抽取车内人员信息（姓名/身份/公司品牌）和座位位置并格式化输出。

核心思路：用类型化决策模型（单次前向返回校准概率、不生成任何文本）替代旧版「约 2000 token 巨型 prompt + LLM 生成 JSON + 字符串切割解析」——结构化输出错误率按构造为 0，每个判定带可按场景调节的置信度。

## 核心结果

零样本、纯本地、零 token 成本（Apple M4 Pro / MPS）：

- **人名 F1 0.964**（510 条扩充集，laya v5，追平 qwen3.8-flash；原始 51 条 F1 1.000、行级 50/50）
- **单条延迟 p50 0.028s、吞吐 1935 条/分钟**——比旧管线（p50 0.745s）**快 27 倍**
- **诚实边界**：分布外对抗子集（表外姓/真实否定/ASR 噪声）F1 0.474 → v5 规则修复后 **0.921**；余困难负例 FP 属 laya 零样本判别上限（微调可解）

📊 完整评测（51 条 / 510 条 / 对抗子集、按模板族分解、v1→v5 工程修复、全部复现命令）→ **[docs/evaluation.md](docs/evaluation.md)**

## 功能特点

1. 对话问答：OpenAI 兼容 API 调用 Qwen-2.5-Omni-7B（可指向本地边车服务）
2. 结构化提取：Jev（System One 决策模型）完成车内人员/位置/公司品牌提取（v5 新增 company 字段：~90 词汽车/互联网公司词典 + 就近绑定，零模型调用），人员与位置关联显示
3. 提取范围限定：只对用户输入提取，不对模型回答二次提取；不提取地理地址
4. 批量测试集处理与结果报告（P/R/F1 + 位置/身份 + 按模板族分解）
5. 通用人物提取试点（v4③）：同一管线复用到通用对话（去座位、加说话人关系），滩头四族 F1 1.000——记忆系统提取阶段的候选原型
6. 浏览器 live 推理：展示页可选「浏览器实时推理」——模型下载进访客浏览器（HF 直连，国内自动 hf-mirror），JS 移植管线（与 Python 对拍 51/51 逐字节一致）完全本地推理，零服务器、零 token 成本

## 应用场景

智能车载助手 · 车内人员识别与管理 · 座椅个性化设置 · 车内环境控制

## 演示 Demo

- **在线展示页**（预录回放 + 可选浏览器实时推理——模型跑在访客浏览器里；座舱图可交互、例句按场景分类、管线轨迹可展开看每步中间结果）：<https://ranpin.github.io/jevCabin/>
- **本地实时推理版**（支持任意输入、边打字边提取，debounce 350ms）：

```bash
python demo/server.py   # 启动时预加载模型约 3-5s，然后打开 http://127.0.0.1:8000
```

`demo/` = server.py（后端，仅标准库 http.server）+ index.html（座舱俯视图，随输入实时点亮座位）；`docs/index.html` = 线上展示页（模型 322M 无法在静态页运行，故用真实推理结果固化）。

## 快速开始

1. 安装依赖：

```bash
pip install -r requirements.txt   # 含 laya / torch
```

2. 配置对话模型 `.env`（信息提取默认走本地 laya，无需任何 key）：

```
OPENAI_API_KEY=your-api-key
OPENAI_BASE_URL=http://localhost:13984          # 可选：本地 OpenAI 兼容服务
OPENAI_DEFAULT_MODEL=your-local-model-name      # 可选：本地服务的模型名
OPENAI_EXTRACTION_MODEL=your-local-model-name   # 可选
```

3. 运行：

```bash
python qwen_chat.py         # 交互式：输入车内人员描述 → 对话回答 + 结构化提取
python scripts/process_test_set.py  # 批量评测 test/测试集.xlsx（--skip-chat 只跑提取，无需 LLM 服务；--out/--log 指定产物路径）
python jev_extract.py --selftest             # 提取自检（离线）：候选生成层覆盖 gold 全部人名（47/47）
JEV_BACKEND=dryrun python jev_extract.py     # 只打印将发送的问题结构，不调用模型
```

## 仓库结构

```
├── jev_extract.py          # 类型化决策提取管线核心（typesafe/laya/openjev/dryrun 四后端，
│                           #   fanout/forced 两种组装模式，姓氏候选过滤，否定/身份/位置证据门控，
│                           #   v5：介绍/品牌线索锚点 + 位置文本绑定 + 公司品牌提取 + 执行轨迹埋点）
├── qwen_chat.py            # 交互模式主入口：LLM 对话回答 + Jev/Laya 信息提取
├── model_config.py         # OpenAI 兼容客户端配置（对话模型，默认指向 13984 边车）
├── scripts/                # 评测与基准工具（一律从仓库根运行：python scripts/xxx.py）
│   ├── process_test_set.py # 批量评测（--in/--out/--log/--skip-chat）
│   ├── expand_test_set.py  # 扩充集生成器（seed=42 构造即标注，511 条 / 10 模板族）
│   ├── compare_results.py  # 结果 vs gold 对比（P/R/F1 + 位置/身份 + 按模板族分解）
│   ├── eval_adversarial.py # 对抗子集生成与评测（60 条分布外样本，三族各 20）
│   ├── eval_general.py     # 通用人物提取试点评测（7 族 95 条：自我介绍/介绍他人/打招呼/纠正/多人/负例/ASR）
│   ├── fit_calibration.py  # laya 温度校准拟合（510 条域内；实测结论：决策中性，不接入管线）
│   ├── eval_llm_baseline.py # 旧管线准确率评测（任意平台模型，--no-thinking 关思考）
│   ├── bench_perf.py       # Jev 侧性能基准 E1-E4（BENCH_BACKEND=openjev|laya）
│   └── bench_baseline_llm.py # 旧管线延迟基线（从 git 历史加载旧函数，需 13984 边车）
├── demo/                   # 实时推理 demo（server.py 后端 + index.html 座舱交互前端）
├── web/                    # 浏览器 live 推理（pipeline.js = jev_extract.py v5 逐函数移植、
│                           #   live.js = HF 直连+hf-mirror 回退、parity.html = 51 条对拍 harness）
├── docs/
│   ├── index.html          # 线上展示页（预录回放 + 可选浏览器实时推理，GitHub Pages 承载）
│   ├── assets/             # laya-live.bundle.js（esbuild 打包产物）
│   ├── architecture.md     # 信息提取架构详解（Jev 三原语、四步管线、后端配置、浏览器端部署）
│   └── evaluation.md       # 完整评测（51/510 条、对抗子集、浏览器 live 对拍、复现命令、产物清单）
└── test/                   # 测试集 + gold + results/（全部实验产物，清单见评测文档）
```

## 文档

- 🏗 **[信息提取架构（Jev）](docs/architecture.md)**——类型化决策三原语、四步提取管线、与旧管线对比、四后端配置
- 📊 **[评测结果与基准](docs/evaluation.md)**——51 条实测、510 条扩充集、对抗子集（诚实边界）、laya 怪癖与工程对策、复现命令与产物清单
