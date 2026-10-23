# 信息提取架构（Jev 类型化决策）

> jevCabin 的架构详解：提取能力边界、Jev 三原语、四步提取管线、四后端配置。概览与快速上手见 [README](../README.md)，评测数字与复现命令见 [评测文档](evaluation.md)。

## 提取能力边界

1. **人物与位置关联提取**：
   - 提取人物姓名、称呼和身份关系
   - 同时提取每个人物所在的车内空间位置
   - v5 起同时提取人物关联的公司/品牌（`company` 字段：日产/华为 等常见汽车与互联网公司，代码词表就近绑定，零模型调用）
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

1. **候选生成与否定门控**（纯代码）：枚举输入中所有 2–4 字中文片段和拉丁词，过滤含功能词/语气词的片段；surname 过滤模式只保留姓氏开头（含生僻姓）/ 小X / 阿X / 复姓 /「单字+称谓」（X老师），并做称谓碎片、叠字碎片、裸称谓（bob小姐 的 小姐）两遍清理 + v5 公司名清理（包含清理：阿里 ⊂ 阿里巴巴；两用词公司义清理：『小鹏的周平』的 小鹏 按公司义剔除人物候选）；被否定线索引导的候选（前向 不是/并非/当成… 3 字窗口 + 后向 没来/不是我/不在车上… 就近关联）在代码层剔除——laya 零样本否定判断是噪声，而否定句式是确定性表层句法
2. **第一轮提问**（一次请求，投机扇出）：对每个候选并行问 4 个问题——
   - `Noul` 『X』是这句话中提到的某个人物的名字或称呼
   - `Noul` 这句话否定或纠正了『X』这个称呼（处理"我不是陈总"类用例）
   - `Choice` 『X』坐在车内哪个位置（闭集：主驾/副驾/后排左边/…/未提及）
   - `Choice` 『X』的身份或称谓（半闭集：同事/同学/老师/…/未提及）
3. **第二轮消歧**（仅在需要时）：互相重叠的候选（如"小陈/陈同学/小陈同学"）用一次 `Choice` 让 Jev 选出最准确的称呼
4. **代码组装**：按概率阈值过滤、去重；称谓后缀拆分（陈同学 → 小陈 + 身份同学，仅当原文存在止于称谓起点的重叠候选时触发）与身份后缀抑制（陈部长 + 部长 → 身份空）；位置/身份文本证据门控（原文无任何位置词 / 身份词未在人名旁出现时置空——防幻觉）；**v5 确定性锚点**（『我是/我叫/这是…』介绍线索或『[公司词]的』品牌线索紧邻候选时直接定锚、跳过模型门控——laya 对『我是小张』类句式 noul 系统性低分 0.4407<0.5；同起点命中取最长者，田子 ⊂ 田子涵）；**v5 位置文本证据绑定**（原文与人名确定性绑定的座位词与模型回答冲突时以原文覆盖——修『后排左边』答成『后排右边』的左右混淆）；**v5 公司/品牌就近绑定**（~90 词词典，『[品牌]的[人名]』/『[人名]是[品牌]』，零模型调用）；输出 `{"persons": [{"name", "identity", "location", "company"}]}`

相比旧版「约 2000 token 的巨型 prompt + LLM 生成 JSON + 字符串切割解析」：
- 结构化输出错误率按构造为 0（不存在 JSON 解析失败）
- 品牌名（日产/吉利）、方位词、人称代词不再混入人物（v5 起品牌反而作为 `company` 字段显式提取）——排除从「prompt 规则」变为「可测量的概率 + 确定性词表」
- 每个判定带置信度，阈值可按场景调节（`JEV_IS_PERSON_THRESHOLD` 等）

**通用人物提取试点（v4③）**：`extract_entities_general` 复用上述管线（task="general"）——属性轮把位置问题换成说话人关系；关系不问 laya，由代码按 RELATION_CUES 表层线索直接指派（laya 零样本关系实测 0.565 不可用：叫我X 漏本人、多人句串人、中性句幻觉），身份仍走 laya + 证据门控。输出 `{"name", "identity", "relation"}`，通用集 F1 0.957、关系 88/88，详见[评测文档](evaluation.md)「通用人物提取试点」节。

**执行轨迹（v5）**：`PERF["trace"]` 记录管线每步中间结果——候选生成与否定剔除、锚点判定路径及模型概率（强制选择分布 / 单人门控分数 / 介绍与品牌线索）、增量扫描、属性轮问答（位置/身份的模型答案与概率）、代码门控裁决（身份证据抑制 / 位置绑定覆盖 / 公司绑定 / 称谓拆分）。展示页「管线轨迹」展开即渲染此数据（Python 与 JS 移植逐字段镜像，回放/live 两模式同构；纯埋点记录，不影响任何决策——51+510 全量重跑逐行验证零变更）。

## 为什么比长 prompt 方案快得多

一句话：**类型化决策把「生成」变成了「判别」**——旧管线让大模型把答案一个字一个字「写」出来，新管线让小模型一次性「读」出所有问题的概率。四层加速来源叠加：

| 维度 | 旧管线（Qwen-2.5-Omni-7B） | Jev/Laya 类型化决策 |
|---|---|---|
| **输出方式** | 自回归生成：逐 token 写 JSON，每个 token 一次**顺序**前向、无法并行 | 单前向判别：非自回归 encoder，一次前向读入文本+全部问题，**同时**输出所有问题的校准概率（never generates text） |
| **模型规模** | 7B（且多模态） | 322M（mmBERT-base），小约 22 倍 |
| **输入成本** | 每次 prefill ~2000 token 巨型 prompt（指令/schema/少样本示例） | 答案空间请求时定义，输入只有话语 + 紧凑问题 |
| **部署形态** | 远端边车→平台（网络 + 排队） | 纯本地，无网络往返 |

最大头是第一层：**自回归生成是顺序的**（输出 N 个 token ≈ N 次前向），而判别一次前向就出全部结果。实测 p50 0.745s → 0.043s（≈17×，详见[评测文档](evaluation.md)）。

这正是 TypeSafe 把它叫 "System One" 的原因——借卡尼曼的比喻：系统 1（快、判别、模式匹配）vs 系统 2（慢、审议、生成）。车载提取这类高频小判定只需系统 1，不必付出系统 2 的生成代价。

## 后端配置

通过 `.env` 或环境变量选择后端（默认 `auto`：有 `TYPESAFE_API_KEY` 走官方 API，否则依次尝试本地 laya / open-jev）：

```
JEV_BACKEND=auto            # auto | typesafe | laya | openjev | dryrun
TYPESAFE_API_KEY=sk-...     # TypeSafe 官方 API（闭源远程付费；现状见下文「模型选型现状」）
JEV_MODEL=jev-latest        # 官方 API 模型版本（调过阈值后建议固定版本号）
LAYA_MODEL=convaiinnovations/laya          # 本地 Laya（推荐，多语言）
LAYA_SUBFOLDER=multilingual                # 中文必须走 multilingual
JEV_CANDIDATE_FILTER=surname               # laya 推荐：姓氏候选过滤
JEV_ASSEMBLY=forced                        # laya 推荐：迭代强制选择组装
JEV_LAYA_CALIBRATION=                      # 可选：温度校准 JSON 路径（默认不加载，调查结论见 TODO 节）
OPEN_JEV_MODEL=com-kotobalabs/open-jev-deberta-v3-large   # 本地 open-jev（仅英文）
```

- **typesafe**：官方托管 API（`POST https://api.typesafe.ai/v1/systemone`），需要 `pip install requests`，key 从 console.typesafe.ai 获取
- **laya**：本地 Laya（Convai Innovations，Apache-2.0，mmBERT-base 多语言，MPS/CPU 可跑），需要 `pip install laya`；中文实测最优配置为 `JEV_CANDIDATE_FILTER=surname JEV_ASSEMBLY=forced`（见[评测文档](evaluation.md)）
- **openjev**（⚠️ 仅英文，留作消融对照，非中文场景可用后端）：本地开源复现（DeBERTa-v3-large，Apache-2.0，CPU 可跑），需要
  `pip install git+https://github.com/kotoba-lang/typed-decisions`；英文训练，中文零样本 F1 仅 0.342（品牌名/方位词全高分误报）——在以中文为主的本项目里仅作「英文 only 模型不迁移到中文」的对照
- **dryrun**：不调用任何模型，只打印将要发送的问题结构，用于离线检查

> ⚠️ 官方 API 与开源模型均未公布中文专项基准，中文车载话语属于分布外场景，使用前请先用测试集实测（见[评测文档](evaluation.md)）。

## 模型选型现状（2026-10 调研）

针对「有没有比 laya-multilingual 更好的开源模型」做过一轮调研（HF 模型榜 + 逐个模型卡），结论：**中文类型化决策场景，laya-multilingual 仍是当前唯一成熟的开源选项，本项目选型没有过时**。

- **Laya 家族**：只有 3 个 checkpoint——`laya`（ModernBERT-large 421M，纯英文）、`laya-multilingual`（mmBERT-base 322M，唯一多语言/中文可用）、`laya-typed-decisions`（421M 微调、通用基准 0.766 但纯英文）。没有更新/更大的多语言变体。
- **新出现的同类模型全部不可用于中文**：`rony/gemma3-270m-typed-decisions`（270M、个人实验级、0 likes、实际仅英文）、`MSGEncrypted/ocf-typed-decisions-mbert-base`（mBERT、中文未验证、个人上传）、`fastino/GLiNER2.5-Decide`（340M、是**分类器**而非类型化决策、且仅英文、厂商自测基准）。都是英文 only / 中文未验证 / 个人原型级，不构成替代。
- **官方 Jev API**：闭源远程付费（$0.042/1M tokens），高基数选项 + 校准比 laya 强但慢约 7.8×——不符合本项目「纯本地、零成本、车机/离线」约束。
- **⚠️ 诚实定位**：laya-multilingual **在通用 typed-decisions 基准上零样本近乎随机（0.342）**，模型卡明说「能力来自微调」。jevCabin 能做到 F1 0.964 **完全靠领域工程**（姓氏过滤 + 强制选择 + 证据/否定门控 + v4/v5 规则优化），不是模型通用能力——这正是本项目的核心贡献；对抗子集 0.474 → 0.921（v5 修复后）进一步印证瓶颈在代码边界而非模型。
- **最有价值的新发现——端侧变体**：社区已把 Laya 移植到多种端侧运行时——`litert-community/Laya-Multilingual-LiteRT`（Google LiteRT）、`AXERA-TECH/Laya`（爱芯元智边缘芯片），以及 GGUF / ONNX / CoreML / MXFP8 / MXFP4 / laya-browser。这为本项目「车机/嵌入式部署为未来工作」提供了现成路径（不必自己从头量化）。

## 浏览器端部署（live demo，2026-10-23）

展示页「浏览器实时推理」模式把整条提取管线跑在访客浏览器里（零服务器）：

- **运行时**：[@johnhenry/laya](https://www.npmjs.com/package/@johnhenry/laya)（Apache-2.0，laya-mlx 的纯 TS 移植，WebGPU/CPU 后端）——实测浏览器与 Python laya 概率四位有效数字一致
- **管线**：`web/pipeline.js` = jev_extract.py v5 的逐函数移植（候选生成/否定门控/线索锚点/强制选择/称谓拆分/证据门/位置绑定/公司绑定），51 条对拍 **51/51 逐字节一致**（详见[评测文档](evaluation.md)「浏览器 live 推理」节）
- **模型源**：HF repo `aac6fef/laya-multilingual-mlx`（fp16 ~678MB，revision 钉死），URL 直连 + Range 请求；hf.co 探活失败自动切 hf-mirror（国内可达）
- **关键约束**：Chromium Cache API 单条目上限实测在 200~300MB——643.8MB 模型无法走 Hub repo 缓存路径，只能 URL 直连（无持久缓存，每次启用重新下载）
- **打包**：`web/build.mjs`（esbuild）→ `docs/assets/laya-live.bundle.js`（260KB，Apache NOTICE 内联）

## 已知可优化项（laya 侧，TODO）

模型卡记录的两个 laya 固有坑，与本项目管线直接相关，值得后续处理：

1. **温度重校准——2026-10-21 已实测，关闭（不接入管线）**：`fit_calibration.py` 在 510 条域内集上收集 6797 条 logits 记录拟合——域内 ECE 0.50（严重过自信），拟合撞钳位（choice 型想要 T=7.5、choice:2 桶想要 T=6833，均被钳到 laya 上限 5.0），校准后 holdout ECE 0.37 仍高；laya 官方 0.314→0.106 不迁移到中文座舱域。**关键结论**：管线问题全为 choice 型，统一温度是单调变换——阈值重调后决策等价，零收益；降低分桶样本下限则 choice:11+ 被位置/身份问题（k=12/15）混入污染、温度不可信。`JEV_LAYA_CALIBRATION` 保留 opt-in 加载路径，默认不加载。
2. **noul 标签跟随 bug（laya issue #156）——workaround 已就位**：`LayaBackend` 已把 noul 问题统一改写为「A/B 中性键两选项 choice」（正是官方 workaround），P(A) 映射回 noul 概率，管线阈值不受影响。
3. **剩余零样本上限（v5 后）**：困难负例 FP（马上 noul 0.984 与真实人物不可区分，仅微调可解）；位置左右混淆已由 v5 规则 6（文本证据绑定）修复（位置准确率 0.799→0.879），余量为多人绑定与区域词 gold 口径行；多人族『这是我朋友X』第二人漏扫。

（微调 laya-multilingual 是更高杠杆的路径——微调版 laya-typed-decisions 通用基准 0.766 vs 零样本 0.342——但项目定位是零样本 + 纯工程，故仅备注、不做。）
