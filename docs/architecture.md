# 信息提取架构（Jev 类型化决策）

> jevCabin 的架构详解：提取能力边界、Jev 三原语、四步提取管线、四后端配置。概览与快速上手见 [README](../README.md)，评测数字与复现命令见 [评测文档](evaluation.md)。

## 提取能力边界

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

## Jev 类型化决策模型

[Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev) 是 TypeSafe AI 发布的「System One」类型化决策模型：输入一段文本（state）和一组带类型的问题，单次请求返回每个问题的**校准概率**，不生成任何文本。三种问题原语：

- `Choice`：从 ≤255 个候选中选一个（返回选项 + 概率分布 + 置信度）
- `Score`：2–10 级有序量表打分
- `Noul`：是/否判断，返回 p(yes)

Jev 不能生成文本，因此人名这类开放词表值由代码先给出候选、再让 Jev 挑选（官方推荐模式）。

## 提取管线（jev_extract.py）

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

## 后端配置

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
- **laya**：本地 Laya（Convai Innovations，Apache-2.0，mmBERT-base 多语言，MPS/CPU 可跑），需要 `pip install laya`；中文实测最优配置为 `JEV_CANDIDATE_FILTER=surname JEV_ASSEMBLY=forced`（见[评测文档](evaluation.md)）
- **openjev**：本地开源复现（DeBERTa-v3-large，Apache-2.0，CPU 可跑），需要
  `pip install git+https://github.com/kotoba-lang/typed-decisions`；英文训练，中文零样本不可用（已实测）
- **dryrun**：不调用任何模型，只打印将要发送的问题结构，用于离线检查

> ⚠️ 官方 API 与开源模型均未公布中文专项基准，中文车载话语属于分布外场景，使用前请先用测试集实测（见[评测文档](evaluation.md)）。
