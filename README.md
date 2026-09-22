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

通过 `.env` 或环境变量选择后端（默认 `auto`：有 `TYPESAFE_API_KEY` 走官方 API，否则尝试本地 open-jev）：

```
JEV_BACKEND=auto            # auto | typesafe | openjev | dryrun
TYPESAFE_API_KEY=sk-...     # TypeSafe 官方 API（早期访问需 waitlist）
JEV_MODEL=jev-latest        # 官方 API 模型版本（调过阈值后建议固定版本号）
OPEN_JEV_MODEL=com-kotobalabs/open-jev-deberta-v3-large   # 本地开源复现
```

- **typesafe**：官方托管 API（`POST https://api.typesafe.ai/v1/systemone`），需要 `pip install requests`，key 从 console.typesafe.ai 获取
- **openjev**：本地开源复现（DeBERTa-v3-large，Apache-2.0，CPU 可跑），需要
  `pip install git+https://github.com/kotoba-lang/typed-decisions`；注意该模型为英文训练，中文效果需实测
- **dryrun**：不调用任何模型，只打印将要发送的问题结构，用于离线检查

> ⚠️ 官方 API 与开源复现均未公布中文支持，中文车载话语属于分布外场景，使用前请先用测试集实测（见下）。

### 实测结果（openjev 本地后端，2026-09-22）

硬件：Apple M4 Pro / 24GB / MPS；数据：`test/测试集.xlsx` 全部 51 条；gold：`test/测试集_结果.xlsx`。

**准确率（零样本中文，阈值 0.5）**——`JEV_BACKEND=openjev python process_test_set.py --skip-chat --out test/测试集_结果_jev_openjev.xlsx --log test/处理日志_jev_openjev.txt` + `python compare_results.py test/测试集_结果_jev_openjev.xlsx`：

| 指标 | 数值 |
|---|---|
| 人名级 P / R / F1 | 0.253 / 0.532 / 0.342（TP=25 FP=74 FN=22） |
| 命中人名的位置 / 身份准确率 | 0.680 / 0.600 |
| 行级完全一致 | 8/50 |

结论：**英文训练的 open-jev 零样本中文不可用**——误报集中在品牌名（日产）、方位词（副驾/右边）和动词短语，模型对中文片段的 is_person 判别力失效（召回尚可、精确率崩盘）。

**性能**——`python bench_perf.py`（预热 3 条、正式 2 轮，原始数据 `test/perf_results.json`）：

| 指标 | seq（默认） | batch |
|---|---|---|
| 延迟 p50 / p95 | 1.353s / 3.913s | 1.876s / 4.801s |
| 吞吐 | 38.8 条/分钟 | 29.2 条/分钟 |
| 模型加载 / 峰值 RSS | 4.4s / 0.7GB | 2.0s / 0.89GB |

- 延迟随问题数**严格线性**（8问 0.34s → 116问 4.90s，约 42ms/问题）：512-token 上下文把问题切成串行分块，「加问题不加时」只在官方 API 的单请求架构下成立
- 批量前向（`JEV_OPENJEV_BATCHED=1`）与串行**数值一致**（160 问题最大概率差 9.8e-07）但在 MPS 上**慢 33%**（collator 批内 padding 到最大尺寸，浪费超过批处理收益），故默认串行
- 旧 LLM 基线（localhost:13984）与官方 API 本轮不可测（端点/key 不可得），以上仅为 open-jev 本地后端的绝对值，不构成新旧对比

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