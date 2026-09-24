# Qwen车载智能助手

这是一个车载智能助手，专为汽车座舱场景设计，能够从对话中提取车内人员信息和空间位置信息并格式化输出。

对话回答通过 OpenAI 兼容 API 调用 Qwen-2.5-Omni-7B；信息提取使用 **Jev 类型化决策模型**（见下文「信息提取架构」）。

## 功能特点

1. 通过OpenAI兼容API调用Qwen-2.5-Omni-7B模型实现对话问答
2. 使用 Jev（System One 决策模型）完成车内人员/位置的结构化提取
3. 自动从用户输入中提取车内人员信息（姓名、身份等）和车内空间位置信息
4. 格式化输出对话内容和提取的信息，人员与位置信息关联显示
5. 支持批量处理测试集数据并生成结果报告

## 应用场景

专为汽车座舱场景设计，适用于：
- 智能车载助手
- 车内人员识别与管理
- 座椅个性化设置
- 车内环境控制

## 技术特性

### 特殊信息提取能力
1. **人物与位置关联提取**：
   - 提取人物姓名、称呼和身份关系
   - 同时提取每个人物所在的车内空间位置
   - 实现人员与位置的关联映射

2. **车内空间位置提取**：
   - 专门提取车内空间位置信息
   - 支持位置：主驾、副驾、前排、后排、后排左侧、后排右侧等
   - 不提取地理地址或外部地点信息

3. **信息提取范围限定**：
   - 只对用户输入的内容进行信息提取
   - 不对模型的回答内容进行二次提取

### 输出格式优化
- 人员信息与位置信息关联显示
- 清晰展示车内人员分布情况
- 便于后续系统处理和应用

## 信息提取架构（Jev）

[Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev) 是 TypeSafe AI 发布的「System One」类型化决策模型：输入一段文本（state）和一组带类型的问题，单次请求返回每个问题的**校准概率**，不生成任何文本。三种问题原语：

- `Choice`：从 ≤255 个候选中选一个（返回选项 + 概率分布 + 置信度）
- `Score`：2–10 级有序量表打分
- `Noul`：是/否判断，返回 p(yes)

Jev 不能生成文本，因此人名这类开放词表值由代码先给出候选、再让 Jev 挑选（官方推荐模式）。`jev_extract.py` 的提取管线：

1. **候选生成**（纯代码）：枚举输入中所有 2–4 字中文片段和拉丁词，过滤含功能词的片段
2. **第一轮提问**（一次请求，投机扇出）：对每个候选并行问 4 个问题——
   - `Noul` 『X』是这句话中提到的某个人物的名字或称呼
   - `Noul` 这句话否定或纠正了『X』这个称呼（处理"我不是陈总"类用例）
   - `Choice` 『X』坐在车内哪个位置（闭集：主驾/副驾/后排左边/…/未提及）
   - `Choice` 『X』的身份或称谓（半闭集：同事/同学/老师/…/未提及）
3. **第二轮消歧**（仅在需要时）：互相重叠的候选（如"小陈/陈同学/小陈同学"）用一次 `Choice` 让 Jev 选出最准确的称呼
4. **代码组装**：按概率阈值过滤、去重，输出与旧版一致的 `{"persons": [{"name", "identity", "location"}]}`

相比旧版「约 2000 token 的巨型 prompt + LLM 生成 JSON + 字符串切割解析」：
- 结构化输出错误率按构造为 0（不存在 JSON 解析失败）
- 品牌名（日产/吉利）、方位词、人称代词的排除从「prompt 规则」变为「可测量的概率」
- 每个判定带置信度，阈值可按场景调节（`JEV_IS_PERSON_THRESHOLD` 等）

### Jev 后端配置

通过 `.env` 或环境变量选择后端（默认 `auto`：有 `TYPESAFE_API_KEY` 走官方 API，否则依次尝试本地 laya / open-jev）：

```
JEV_BACKEND=auto            # auto | typesafe | laya | openjev | dryrun
TYPESAFE_API_KEY=sk-...     # TypeSafe 官方 API（早期访问需 waitlist）
JEV_MODEL=jev-latest        # 官方 API 模型版本（调过阈值后建议固定版本号）
LAYA_MODEL=convaiinnovations/laya          # 本地 Laya（推荐，多语言）
LAYA_SUBFOLDER=multilingual                # 中文必须走 multilingual
JEV_CANDIDATE_FILTER=surname               # laya 推荐：姓氏候选过滤
JEV_ASSEMBLY=forced                        # laya 推荐：迭代强制选择组装
OPEN_JEV_MODEL=com-kotobalabs/open-jev-deberta-v3-large   # 本地 open-jev（仅英文）
```

- **typesafe**：官方托管 API（`POST https://api.typesafe.ai/v1/systemone`），需要 `pip install requests`，key 从 console.typesafe.ai 获取
- **laya**：本地 Laya（Convai Innovations，Apache-2.0，mmBERT-base 多语言，MPS/CPU 可跑），需要 `pip install laya`；中文实测最优配置为 `JEV_CANDIDATE_FILTER=surname JEV_ASSEMBLY=forced`（见下文实测结果）
- **openjev**：本地开源复现（DeBERTa-v3-large，Apache-2.0，CPU 可跑），需要
  `pip install git+https://github.com/kotoba-lang/typed-decisions`；英文训练，中文零样本不可用（已实测）
- **dryrun**：不调用任何模型，只打印将要发送的问题结构，用于离线检查

> ⚠️ 官方 API 与开源模型均未公布中文专项基准，中文车载话语属于分布外场景，使用前请先用测试集实测（见下）。

### 实测结果（2026-09-22~24，Apple M4 Pro / 24GB / MPS）

数据：`test/测试集.xlsx` 全部 51 条；gold：`test/测试集_结果.xlsx`（旧管线 10.23 全量通过的输出）。

**准确率对比（零样本，无任何微调）**：

| 配置 | 人名 P/R/F1 | 位置准确率 | 身份准确率 | 行级全对 |
|---|---|---|---|---|
| 旧管线 Qwen-2.5-Omni-7B | （gold 来源） | — | — | 50/50 |
| open-jev（span 候选 + noul 扇出） | 0.253/0.532/0.342 | 0.680 | 0.600 | 8/50 |
| laya 扇出（姓氏过滤，noul 门控） | 1.000/0.170/0.291 | 0.875 | 0.000 | 11/50 |
| **laya 强制选择（姓氏过滤 + forced + 身份证据门控）** | **0.936/0.936/0.936** | **0.841** | **0.955** | **46/50** |

复现命令（laya 最优配置）：
```bash
JEV_BACKEND=laya JEV_CANDIDATE_FILTER=surname JEV_ASSEMBLY=forced \
  python process_test_set.py --skip-chat --out test/测试集_结果_jev_laya_forced.xlsx --log test/处理日志_jev_laya_forced.txt
python compare_results.py test/测试集_结果_jev_laya_forced.xlsx
```

**性能对比**（`BENCH_BACKEND=laya python bench_perf.py` / `python bench_perf.py` / `python bench_baseline_llm.py`，预热 3 条、正式 2 轮；原始数据 `test/perf_results*.json`、`test/perf_baseline_llm.json`）：

| 指标（同机 51 条） | 旧管线（边车→平台） | openjev-seq | **laya forced** |
|---|---|---|---|
| 延迟 p50 | 0.745s | 1.353s | **0.044s** |
| 延迟 p95 | 1.424s | 3.913s | **0.052s** |
| 吞吐 | ~68 条/分钟 | 38.8 条/分钟 | **1425 条/分钟** |
| 加载 / 内存 | 远端服务 | 4.4s / 0.7GB | 3.7s / 3.0GB |
| 每次调用成本 | 1400 prompt + 39 completion tokens | 本地机时 | 本地机时 |

**结论**：
- **laya-multilingual（mmBERT-base，Apache-2.0）+ 领域候选过滤 + 强制选择组装 = 比旧管线快 17 倍、人名 F1 0.936**，零样本、纯本地、零 token 成本。Jev 范式的性能主张（毫秒级、单前向）由 laya 在本地兑现
- 剩余 4 行错误集中在：『小陈同学』边界切分（2 行，gold 要 小陈+身份同学）、否定句（2 行，『我不是陈总我叫陈部长』类）——属微调可解范围（官方 Kaggle 2×T4 微调 notebook 现成）
- **laya 已知怪癖与本工程对策**（模型卡 Honest Limits 全部实测复现）：
  - noul 弱信号/选项顺序敏感（同一问题交换 A/B 位置分数 0.82↔0.44）→ 弃用独立 noul 扇出，改**迭代强制选择**（候选在同一 softmax 内竞争）
  - 单候选退化成二选一时随机倒向『没有人物』→ 单候选走 noul 阈值门
  - 身份高置信度幻觉（坐主驾→司机 0.97）→ **文本证据门控**：身份词必须在原文中与人名相邻出现，否则抑制（身份准确率 0.182→0.955）
  - 选项位置偏差 → 『未提及』放选项表第一位
- open-jev（英文 DeBERTa）零样本中文不可用（品牌名/方位词全高分误报）；其 512-token 上下文导致延迟随问题数严格线性（42ms/问），批量前向在 MPS 上反而慢 33%（批内 padding），默认串行
- ⚠️ 旧代码兼容性发现：main 分支提取函数用 `hasattr(response, '__iter__')` 判断流式，新版 openai SDK 的 ChatCompletion（pydantic v2）可迭代 → 非流式响应被误判、解析必失败（延迟测量不受影响）。如需在 main 上修复，改用 `if EXTRACTION_STREAM:` 分支即可
- 官方 Jev API（typesafe 后端）仍待 key，拿到后可补测第四列

### 扩充集评测（510 条，2026-09-24）

`expand_test_set.py`（seed=42，构造即标注）把测试集扩到 10 倍：原始 51 条 + 460 条模板生成，覆盖 10 个模板族（品牌位置/多人/否定句/无人物负例含困难负例/拉丁名/身份明示/同学边界等）。

| 指标（510 条） | laya 零样本（forced） | **qwen3.8-flash-aliyun（关思考）** |
|---|---|---|
| 人名 P / R / F1 | 0.885/0.719/0.793 | **0.964/0.964/0.964** |
| 位置 / 身份准确率 | 0.779 / 0.828 | **0.988 / 0.979** |
| 行级全对 | 368/510 (72.2%) | **491/510 (96.3%)** |
| 延迟 p50 / p95 | **0.044s / 0.052s** | 1.29s / 6.47s |
| 每次调用成本 | 0（本地） | ~1400 prompt + 43 completion tokens |

按模板族（行级全对率）：

| 族 | laya | flash |
|---|---|---|
| 品牌位置 / 无品牌位置 | 0.81 / 0.70 | **1.00 / 1.00** |
| 多人 | 0.38 | **1.00** |
| 否定句 | 0.42 | **1.00** |
| 无人物负例 / 身份明示 / 自我介绍 | 0.81 / 1.00 / 0.95 | **1.00 / 1.00 / 1.00** |
| 同学边界 / 称谓无品牌 | 0.56 / 0.55 | **1.00 / 1.00** |
| 拉丁名 | **0.72** | 0.24 |

结论：
- **flash（关思考）是新的准确率基线**：10/11 族满分，逼近原始 gold 水平；唯一短板「拉丁名」是称谓边界约定偏差（把『bob小姐』整体当名字，gold 约定剥离称谓），非能力问题
- **laya 的速度优势保持**（p50 44ms，比 flash 快 ~30 倍、零 token 成本），但零样本准确率差距在大样本下暴露：多人（迭代提前停止）与否定句（被否定名未抑制）两族是主要失分点，均有不依赖微调的工程解法（迭代指令带已提取上下文 / 代码级否定门控）
- 拉丁名族 laya 反超（0.72 vs 0.24）：代码级片段提取天然剥离称谓，LLM 反而把称谓包进名字
- Qwen3.8 系（max/flash）默认开思考；`enable_thinking:false` 平台可透传（`eval_llm_baseline.py --no-thinking`），关思考后 flash 单条 p50 1.29s，开思考的 max 单条 ≥7s（511 条跑不完，已弃）

复现：
```bash
python expand_test_set.py                                   # 生成 510 条 + gold + meta
python eval_llm_baseline.py qwen3.8-flash-aliyun --no-thinking   # 需 13984 边车
python compare_results.py test/测试集_扩充10x_结果_qwen3.8-flash-aliyun.xlsx test/测试集_扩充10x_gold.xlsx
```

## 仓库结构

```
├── model_config.py         # OpenAI 兼容客户端配置（对话模型，默认指向 13984 边车）
├── qwen_chat.py            # 交互模式主入口：LLM 对话回答 + Jev/Laya 信息提取
├── process_test_set.py     # 批量评测（--in/--out/--log/--skip-chat）
├── jev_extract.py          # 类型化决策提取管线核心（typesafe/laya/openjev/dryrun 四后端，
│                           #   fanout/forced 两种组装模式，姓氏候选过滤，身份证据门控）
├── expand_test_set.py      # 扩充集生成器（seed=42 构造即标注，511 条 / 10 模板族）
├── compare_results.py      # 结果 vs gold 对比（P/R/F1 + 位置/身份 + 按模板族分解）
├── bench_perf.py           # Jev 侧性能基准 E1-E4（BENCH_BACKEND=openjev|laya）
├── bench_baseline_llm.py   # 旧管线延迟基线（从 git 历史加载旧函数，需 13984 边车）
├── eval_llm_baseline.py    # 旧管线准确率评测（任意平台模型，--no-thinking 关思考）
└── test/
    ├── 测试集.xlsx                          # 原始 51 条用例
    ├── 测试集_结果.xlsx                     # 原始 gold（10.23 旧管线全量通过输出）
    ├── 测试集_扩充10x.xlsx                  # 扩充集 511 条（生成）
    ├── 测试集_扩充10x_gold.xlsx             # 扩充集 gold（构造即标注）
    ├── 测试集_扩充10x_meta.json             # 话语 → 模板族映射
    ├── 测试集_结果_jev_openjev.xlsx         # open-jev 原始集（F1 0.342）
    ├── 测试集_结果_jev_laya.xlsx            # laya 扇出模式（F1 0.291，被 forced 取代）
    ├── 测试集_结果_jev_laya_forced.xlsx     # laya 强制选择（F1 0.936）
    ├── 测试集_扩充10x_结果_laya.xlsx        # laya 扩充集（F1 0.793）
    ├── 测试集_扩充10x_结果_qwen3.8-flash-aliyun.xlsx  # flash 扩充集（F1 0.964）
    ├── perf_results.json                    # open-jev 性能原始数据
    ├── perf_results_laya.json               # laya 性能原始数据
    ├── perf_baseline_llm.json               # 旧管线（omni-7b）延迟原始数据
    └── 处理日志*.txt                        # 各轮跑批日志
```

## 环境准备

1. 安装依赖:
```bash
pip install -r requirements.txt
```

2. 配置API密钥:
   编辑`.env`文件，将`your-api-key-here`替换为您的实际API密钥:
   ```
   OPENAI_API_KEY=your-actual-api-key
   ```

## 使用本地部署的模型服务

如果您部署了与OpenAI API兼容的本地服务，可以通过配置`.env`文件中的`OPENAI_BASE_URL`来使用:

```
OPENAI_API_KEY=your-api-key-here
OPENAI_BASE_URL=http://localhost:13984
```

程序会自动使用您指定的URL作为模型服务的地址。

如果您的本地服务使用不同的模型名称，可以通过以下配置指定模型名称：

```
OPENAI_API_KEY=your-api-key-here
OPENAI_BASE_URL=http://localhost:13984
OPENAI_DEFAULT_MODEL=your-local-model-name
OPENAI_EXTRACTION_MODEL=your-local-model-name
```

## 使用方法

### 交互式对话模式

运行脚本:
```bash
python qwen_chat.py
```

根据提示输入车内人员情况描述，系统会自动分析并提取相关信息。

### 批量测试集处理模式

运行脚本:
```bash
python process_test_set.py
```

该脚本会读取`test/测试集.xlsx`文件中的测试用例，逐条处理并生成以下输出：
1. `test/测试集_结果.xlsx` - 包含详细处理结果的Excel文件
2. `test/处理日志.txt` - 完整的处理日志文件

可选参数：
```bash
python process_test_set.py --skip-chat --out test/测试集_结果_jev.xlsx --log test/处理日志_jev.txt
```
- `--skip-chat`：跳过对话模型调用，只跑 Jev 信息提取（无需本地 LLM 服务）
- `--out PATH`：指定结果输出路径，避免覆盖旧的对照结果
- `--log PATH`：指定日志输出路径（默认会覆盖 `test/处理日志.txt`）

### Jev 提取自检（离线，无需任何 API）

```bash
python jev_extract.py --selftest
```

检查候选生成层是否覆盖测试集 gold 标注中的全部人名（当前：47/47 全覆盖）。

```bash
JEV_BACKEND=dryrun python jev_extract.py
```

打印一条示例输入将要发送给 Jev 的全部问题结构，不调用模型。

## 输出格式

输出分为两个部分:
1. Qwen模型的完整回答
2. 结构化的车内人员分布信息（人员与位置关联显示）

## 示例

输入:
```
我是司机王五，副驾坐着我的同事小李，后排坐着我的孩子小宝。
```

输出:
```
============================================================
🤖 Qwen 车载助手回答:
--------------------------------------------------
您好王师傅！我已经记录了车内的乘坐情况。司机是您，副驾驶坐着您的同事小李，后排坐着您的孩子小宝。我会根据不同的乘坐需求为您提供个性化的服务。
============================================================
📋 车内人员分布:
--------------------------------------------------
   1. 王五（司机） 位于 主驾
   2. 小李（同事） 位于 副驾
   3. 小宝（孩子） 位于 后排
============================================================
```