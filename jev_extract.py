"""基于 Jev（类型化决策模型）的车载人员信息提取。

替代原「超长 prompt + 生成式 LLM 吐 JSON」的提取管线：
1. 代码从用户输入生成人名候选（中文 2-4 字片段 + 拉丁词，过滤功能词）
2. 一次 Jev 请求对每个候选并行提问（is_person / negated / location / identity）
3. 代码按概率组装 {"persons": [...]}，输出结构天然合法，无需解析 JSON

Jev 不生成文本，只返回带类型的校准概率（Choice / Score / Noul），
因此人名这类开放词表值必须先由代码给出候选，再交给 Jev 挑选。

后端（JEV_BACKEND 环境变量，默认 auto）：
- typesafe: TypeSafe 官方 API（需要 TYPESAFE_API_KEY，早期访问）
- openjev:  本地开源复现 com-kotobalabs/open-jev-deberta-v3-large
            （需要 pip install git+https://github.com/kotoba-lang/typed-decisions）
- dryrun:   只打印将要发送的请求，不调用任何模型
"""

import json
import os
import re
import time

# ---------------------------------------------------------------- 配置

JEV_BACKEND = os.getenv("JEV_BACKEND", "auto")  # auto | typesafe | openjev | laya | dryrun
TYPESAFE_API_KEY = os.getenv("TYPESAFE_API_KEY", "")
TYPESAFE_API_URL = os.getenv("TYPESAFE_API_URL", "https://api.typesafe.ai/v1/systemone")
JEV_MODEL = os.getenv("JEV_MODEL", "jev-latest")
OPEN_JEV_MODEL = os.getenv("OPEN_JEV_MODEL", "com-kotobalabs/open-jev-deberta-v3-large")
LAYA_MODEL = os.getenv("LAYA_MODEL", "convaiinnovations/laya")
LAYA_SUBFOLDER = os.getenv("LAYA_SUBFOLDER", "multilingual")  # 中文必须走 multilingual
# 温度校准 opt-in（fit_calibration.py 产物的路径）。默认不加载：2026-10-21
# 510 条域内实测拟合撞钳位（想要 T=7.5~6833，钳到 5.0）、域内 ECE 0.50→0.37 仍高，
# 且管线问题全为 choice 型 → 统一温度是单调变换，阈值重调后决策等价（详见 docs/evaluation.md）
LAYA_CALIBRATION = os.getenv("JEV_LAYA_CALIBRATION", "")

IS_PERSON_THRESHOLD = float(os.getenv("JEV_IS_PERSON_THRESHOLD", "0.5"))
NEGATED_THRESHOLD = float(os.getenv("JEV_NEGATED_THRESHOLD", "0.5"))

# 组装模式：fanout=每候选独立 noul 扇出（openjev/typesafe 验证过）；
# forced=迭代强制选择（laya 验证形态：neg 门对 laya 零样本是噪声，
# 强制选择让候选在同一 softmax 内竞争，且天然处理否定句）
ASSEMBLY_MODE = os.getenv("JEV_ASSEMBLY", "fanout")
FORCED_MIN_P = float(os.getenv("JEV_FORCED_MIN_P", "0.3"))
# 强制选择判『没有人物』的置信度 ≥ 此值时信任（真负例实测 0.77-0.86），
# 低于此值视为含糊、回退普通 noul 扫描（多人句漏判实测 0.62）
NO_PERSON_CONFIDENT = float(os.getenv("JEV_NO_PERSON_CONFIDENT", "0.7"))
FORCED_MAX_PERSONS = int(os.getenv("JEV_FORCED_MAX_PERSONS", "5"))
FORCED_NO_PERSON = "没有人物"
FORCED_INSTRUCTIONS = (
    "这句话中提到的某个人物的名字或称呼是哪个？只根据这句话判断，"
    "品牌名、车内位置、方位词、普通名词都不算人物称呼；"
    "被否定或纠正的称呼（如『我不是X』『别叫我X』中的X）也不算。"
    "如果这句话没有提到任何人物，选『没有人物』。"
)

MAX_CANDIDATES = int(os.getenv("JEV_MAX_CANDIDATES", "60"))
# 候选过滤模式：span=全片段枚举（默认）；surname=只保留姓氏开头/小X/拉丁词
# （surname 模式把「品牌名/方位词/普通名词」类误报消灭在代码层，模型只裁决强候选）
CANDIDATE_FILTER = os.getenv("JEV_CANDIDATE_FILTER", "span")

# 常见单字姓（百家姓 Top ~100；刻意排除 司/老/同 等易与普通词冲突的生僻姓）
SURNAME_CHARS = set(
    "王李张刘陈杨黄赵吴周徐孙马朱胡郭何林罗高郑梁谢宋唐许韩冯邓曹彭曾肖田董袁潘"
    "于蒋蔡余杜叶程苏魏吕丁任沈姚卢姜崔钟谭陆汪范金石廖贾夏韦付方白邹孟熊秦邱江"
    "尹薛闫段雷侯龙史陶黎贺顾毛郝龚万钱严覃武戴莫孔向汤"
    # v4：无普通词冲突面的生僻姓（对抗集表外姓盲区）。商/迟/展/初/福/贵 等
    # 易撞词姓仍排除（同 司/老/同 原则），其「X+称谓」形态由 TITLE_SUFFIXES 覆盖
    "翦仝佟佘菅庹慕言"
)
COMPOUND_SURNAMES = frozenset((
    "欧阳", "司马", "诸葛", "上官", "夏侯", "皇甫", "尉迟", "公孙", "令狐", "长孙",
    "慕容", "司徒", "端木", "东方", "独孤", "南宫", "呼延", "西门", "第五", "淳于",
    "单于", "太叔", "申屠", "仲孙", "轩辕", "百里", "东郭", "南门", "羊舌", "微生",
))
# 多字称谓的首字（老师/部长/经理/医生/师傅/同学）：姓+这些字的 2 字片段
# 若存在同起点更长候选，几乎必是称谓碎片（陈部 ⊂ 陈部长，陈同 ⊂ 陈同学）
TITLE_HEAD_CHARS = frozenset("老部经医师同")

# v4：「单字+人物称谓」候选锚点：这些词几乎不会出现在非人物词的词尾，
# 可不开姓氏表就放行表外姓的称谓形态（翦老师/佟老板/撒师傅…）。
# 不含 总/工：撞 总结/汇总/工作/工程（同 马上/高架 困难负例类别）
TITLE_SUFFIXES = ("老师", "医生", "经理", "老板", "师傅")
# 称谓前的常见动词/量词（看医生/当老师/那位经理）不是称呼，不生成候选
TITLE_PREFIX_BLOCK = frozenset("看找做当像个位些问帮")

# v4 称谓后缀拆分：赢家形如『X+称谓』（陈同学）且原文恰有候选止于称谓起点
# （小陈）时，拆成 name=小陈 + 身份证据=同学。gold 约定：『小陈同学』=
# 小陈+身份同学；『陈同学』（文中无小陈）整体留名、身份为空。
SPLIT_SUFFIXES = ("同学", "老师", "师傅", "先生", "小姐", "女士")
# 裸称谓词：紧跟在另一候选之后时（bob小姐）是该人物的称谓，不是独立人物
BARE_TITLE_WORDS = frozenset(("同学", "老师", "师傅", "先生", "小姐", "女士",
                              "医生", "经理", "老板", "部长", "司机", "护士"))

# v4③：完整称谓词——候选片段不允许从这些词的内部起点。「工程师郑刚」会生成
# 跨界碎片「程师郑」（程∈姓氏表），laya 实测偏好它胜过真名 郑刚；这些称谓词
# 内部字符只有 程（工程师）是姓氏，故本规则对合法候选零影响
TITLE_WORDS_FULL = ("工程师", "老师", "医生", "经理", "老板", "师傅", "同学",
                    "学生", "护士", "司机", "部长", "先生", "小姐", "女士")

# 候选片段中出现这些字符即丢弃（功能词/动词/问候语，几乎不可能出现在姓名或称谓里）
STOPWORD_CHARS = set(
    "的我你他她它是在坐了有叫别不这那就都也还很吗吧啊呢给和跟向对把让被从到说"
    "快打招呼请问候兴见真早"
    # v4：句尾语气词与否定动词「没」（ASR 口语碎片「小吴哈/周总没错」），无姓名含这些字
    "哈嘛哦诶嗯呗啦咯呀哟哇嘞没"
    # v4③：敬称代词「您」（「郑工您好」会生成跨界碎片 郑工您，实测劫持赢家/稀释 softmax）
    "您"
)

LOCATION_OPTIONS = [
    "主驾", "副驾", "前排", "后排",
    "后排左边", "后排右边", "左后", "左后座", "右后", "右后座",
    "左边", "右边", "未提及",
]

# 未提及放第一位：laya 对选项顺序有位置偏差，身份 gold 绝大多数为空，
# 让偏差偏向『未提及』而不是幻觉出身份
IDENTITY_OPTIONS = [
    "未提及", "司机", "妻子", "丈夫", "儿子", "女儿", "孩子", "同事", "同学",
    "老师", "工程师", "朋友", "老板", "部长", "医生",
]

LOCATION_QUESTION = (
    "这个人物在这句话中坐在车内的哪个位置？按句子原话选择最匹配的说法，"
    "注意严格区分左边和右边；句子没说位置就选未提及"
)
IDENTITY_QUESTION = (
    "这个人物在这句话中的身份或称谓是什么？只有句子明确说明了身份"
    "（如『同事小李』『我的孩子小宝』『司机王五』）才选对应项，没有明确说明就选未提及"
)
NO_LOCATION = "未提及"
NO_IDENTITY = "未提及"

# ---------------------------------------------------------------- v4③ 通用人物提取试点
# 通用对话域（打招呼/自我介绍/介绍他人/纠正）：去掉车内位置，换成说话人关系。
# 战略用途：mirix-cpp Q4 提取阶段的候选原型——Mem0 的提取阶段用 LLM（φ），
# 类型化决策 ~44ms 本地提取是端侧替代；纠正句式（『不对我叫X』）是
# UPDATE/SUPERSEDE 的提取侧信号。两项目保持边界，本模块只验证通用域可行性。
#
# 关系由代码按表层线索直接指派，不问 laya：零样本关系回答实测 0.565 且
# 错误系统性（『叫我X』漏判本人、『这是我妻子X』答本人或空、多人句关系
# 串到另一人、中性句幻觉朋友）——介绍句法里关系是确定性表层措辞，
# 同否定门控方法论：代码管确定性句法，模型管语义判别。
RELATION_CUES = (
    ("说话人本人", ("我是", "我叫", "叫我", "喊我", "称呼我", "我的名字是")),
    ("家人", ("妻子", "丈夫", "儿子", "女儿", "老婆", "老公", "爸爸", "妈妈",
              "父亲", "母亲", "哥哥", "姐姐", "弟弟", "妹妹", "孩子", "家人",
              "我哥", "我姐", "我弟", "我妹", "媳妇", "爱人")),
    ("朋友", ("朋友", "哥们", "闺蜜", "发小")),
    ("同事", ("同事", "搭档")),
)
RELATION_CUE_MAX_GAP = 8
# 通用域身份 = 职业称谓（同事/朋友 归关系线索，不在此重复）
GENERAL_IDENTITY_OPTIONS = [
    "未提及", "老师", "医生", "经理", "老板", "工程师",
    "同学", "学生", "护士", "师傅",
]
GENERAL_IDENTITY_QUESTION = (
    "这个人物在这句话中的身份或称谓是什么？只有句子明确说明了身份"
    "（如『工程师小李』『我的同学小张』）才选对应项，没有明确说明就选未提及"
)


# ---------------------------------------------------------------- 候选生成

def _keep_cjk_span(s):
    """surname 过滤模式：只保留像中文人名的片段。

    规则：单字姓开头的 2-3 字片段 / 复姓开头的 3-4 字片段 /
    小X、阿X 昵称（2-3 字）/ 单字+人物称谓（X老师，v4）。
    span 模式（默认）全部保留。
    """
    if CANDIDATE_FILTER != "surname":
        return True
    w = len(s)
    if s[0] in ("小", "阿") and 2 <= w <= 3:
        return True
    if 2 <= w <= 3 and s[0] in SURNAME_CHARS:
        return True
    if 3 <= w <= 4 and s[:2] in COMPOUND_SURNAMES:
        return True
    if w == 3 and s[1:] in TITLE_SUFFIXES and s[0] not in TITLE_PREFIX_BLOCK:
        return True
    return False


def _inside_title_word(text, start):
    """v4③：start 是否落在某个称谓词出现的内部（非词首）——阻断跨界碎片。"""
    for w in TITLE_WORDS_FULL:
        pos = text.find(w)
        while pos >= 0:
            if pos < start < pos + len(w):
                return True
            pos = text.find(w, pos + 1)
    return False


def generate_candidates(text):
    """枚举人名候选片段，返回 [{'id', 'start', 'end', 'text'}, ...]。

    中文取所有 2-4 字子串（含功能词的丢弃，surname 模式下再过姓氏规则），
    拉丁文取连续字母词。同文本只保留首次出现位置。
    """
    seen = {}
    cands = []

    def add(start, end, s):
        if s in seen:
            return
        seen[s] = True
        cands.append({"id": len(cands), "start": start, "end": end, "text": s})

    for m in re.finditer(r"[A-Za-z]{2,}", text):
        add(m.start(), m.end(), m.group())

    for width in (2, 3, 4):
        for i in range(len(text) - width + 1):
            s = text[i:i + width]
            if not re.fullmatch(r"[\u4e00-\u9fff]+", s):
                continue
            if any(ch in STOPWORD_CHARS for ch in s):
                continue
            if not _keep_cjk_span(s):
                continue
            if _inside_title_word(text, i):
                continue
            add(i, i + width, s)

    # 碎片清理：2 字候选若形如「姓+多字称谓首字」（陈部 ⊂ 陈部长），
    # 且同起点存在更长候选，则剔除。碎片留在候选里会稀释强制选择的
    # softmax（实测 [陈部, 陈部长] 时 laya 倒向『没有人物』）。
    # 只认真正的称谓碎片前缀——若按「非完整称谓」宽泛剔除，会把
    # 江湖/宋城 这类困难负例也削成单候选、落进 noul 路径产生误报
    # （实测 江湖救 noul=0.73，而强制选择能正确以 0.86 拒绝）。
    cleaned = []
    for c in cands:
        if (len(c["text"]) == 2 and c["text"][0] != "小"
                and c["text"][1] in TITLE_HEAD_CHARS
                and any(o["start"] == c["start"] and len(o["text"]) > 2
                        for o in cands)):
            continue
        # v4 叠字碎片：ASR 重复「王总王总」会生成「王总王」——前 len-1 字
        # 已是候选且末字等于其首字，必是重复截断碎片，剔除
        t = c["text"]
        if (len(t) >= 3 and t[-1] == t[0]
                and any(o["text"] == t[:-1] for o in cands)):
            continue
        # v4 裸称谓：「bob小姐」的『小姐』紧跟另一候选，是其称谓而非独立人物
        if t in BARE_TITLE_WORDS and any(o["end"] == c["start"] for o in cands):
            continue
        cleaned.append(c)
    # v4 第二遍：称谓前缀碎片（小陈同 = 小陈 + 同）。前缀候选必须在第一遍
    # 存活——否则会把 冯老师（前缀 冯老 本身是待清理碎片）误杀成碎片
    final = []
    for c in cleaned:
        t = c["text"]
        if (len(t) >= 3 and t[-1] in TITLE_HEAD_CHARS
                and any(o["text"] == t[:-1] and o["start"] == c["start"]
                        for o in cleaned)):
            continue
        final.append(c)
    # v5 公司名包含清理：『阿里巴巴的小张』的 阿里 会按阿X昵称模式生成人物
    # 候选（实测误报成人物、且其名字区间会挡住公司绑定）——候选严格包含于
    # 更长公司词出现时剔除。小米/小鹏 等人名/公司两用词无更长公司词包含，不受影响
    comp_spans = _company_spans(text)
    if comp_spans:
        final = [c for c in final
                 if not any(ws <= c["start"] and c["end"] <= we
                            and we - ws > c["end"] - c["start"]
                            for ws, we in comp_spans)]
    # v5 规则7 两用词公司义清理：公司词候选后紧跟「的+另一候选」（『小鹏的周平』）
    # 时公司义优先，剔除该人物候选——510 集 4 条『小鹏的X』行实测 小鹏 既成人物
    # FP（gold 只有 X）又挡住公司绑定；『我是小鹏』『小鹏和小王』不匹配模式，
    # 人物义保留
    company_set = set(COMPANY_WORDS)
    final = [c for c in final
             if not (c["text"] in company_set
                     and text[c["end"]:c["end"] + 1] == "的"
                     and any(o["start"] == c["end"] + 1
                             for o in final if o is not c))]
    return final[:MAX_CANDIDATES]


# ---------------------------------------------------------------- 问题构造

def build_questions(cands):
    """官方 API 风格的问题字典：问题 id -> {type, instructions, criteria}。"""
    questions = {}
    for c in cands:
        i, name = c["id"], c["text"]
        questions[f"c{i}_is"] = {
            "type": "noul",
            "instructions": f"『{name}』是这句话中提到的某个人物的名字或称呼",
        }
        questions[f"c{i}_neg"] = {
            "type": "noul",
            "instructions": f"这句话否定或纠正了『{name}』这个称呼（例如说话人说自己不是、别叫这个称呼）",
        }
        questions[f"c{i}_loc"] = {
            "type": "choice",
            "instructions": f"『{name}』" + LOCATION_QUESTION,
            "criteria": {opt: "" for opt in LOCATION_OPTIONS},
        }
        questions[f"c{i}_id"] = {
            "type": "choice",
            "instructions": f"『{name}』" + IDENTITY_QUESTION,
            "criteria": {opt: "" for opt in IDENTITY_OPTIONS},
        }
    return questions


def overlaps(a, b):
    """两个候选在原文中的字符区间是否重叠（同一人物的不同写法会重叠）。"""
    return not (a["end"] <= b["start"] or b["end"] <= a["start"])


def _identity_supported(text, name, identity):
    """身份词必须在原文中与人名相邻出现（模型提议，文本证据裁决）。

    laya 零样本会高置信度幻觉身份（坐在主驾→司机 0.97），用原文字面
    证据门控：身份词不在人名附近出现就丢弃。
    """
    if not identity:
        return True
    idx = text.find(name)
    if idx < 0:
        return False
    window = text[max(0, idx - 4):idx + len(name) + 4]
    return identity in window


# 否定线索：出现在人名前 3 字窗口内即视为该次出现被否定。
# laya 零样本的否定判断是纯噪声（实测被否定 0.71 vs 未否定 0.79，
# 任何阈值切不开），而『不是X/别叫我X』是确定性表层句法，代码层解决。
# v4 泛化：并非（书面否定）/当成（误认『把我当成X』）/算不上/哪里是（反问否定）/
# 别记成（纠正）。窗口保持 3 字——扩到 4 会误杀『不是啊，小李在主驾』类
# 话语否定+真实人名句式，故『我没说我是X』（线索与人名字隔 2 字）为已知留白。
NEGATION_CUES = ("不是", "不叫", "别叫", "别说", "别喊", "不姓",
                 "并非", "当成", "算不上", "哪里是", "别记成")

# v4 后向缺席/纠正线索：线索出现在人名之后（『陈总不是我』『周老师没来』
# 『何医生今天休息，没来』），只关联最近的前置候选（距离 ≤ ABSENCE_MAX_GAP，
# 可跨逗号），避免『周老师来了，小李没来』类多人句误杀前者。
# 全部是人物级缺席/纠正谓词；位置否定只收完整的『不在车上/没上车』而不收
# 『不在』，防止『小李在主驾不在副驾』类位置纠正句误杀人物。
ABSENCE_CUES = ("不在车上", "没上车", "没来", "不来", "缺席", "不是我", "是别人")
ABSENCE_MAX_GAP = 8


def _is_negated(text, name):
    """name 在 text 中的所有出现是否都被否定线索引导。

    只要有一次未被否定的出现就保留（『我不是陈总，陈总在副驾』里
    第二个陈总是真实人物指称）。
    """
    total = negated = 0
    start = 0
    while True:
        idx = text.find(name, start)
        if idx < 0:
            break
        total += 1
        window = text[max(0, idx - 3):idx]
        if any(cue in window for cue in NEGATION_CUES):
            negated += 1
        start = idx + 1
    return total > 0 and negated == total


def _backward_negated(text, cands):
    """后向缺席/纠正线索命中的候选 id 集合。

    每个线索出现只否定距离最近的前置候选（≤ ABSENCE_MAX_GAP）及其
    重叠候选——『赵总并不在车上』会同时生成 赵总/赵总并，后者离线索
    更近，须连带否定重叠的真正人物候选；而『周老师来了，小李没来』的
    两个人物候选互不重叠，只杀最近的小李。
    """
    negated = set()
    for cue in ABSENCE_CUES:
        start = 0
        while True:
            q = text.find(cue, start)
            if q < 0:
                break
            near = [c for c in cands
                    if 0 <= q - c["end"] <= ABSENCE_MAX_GAP]
            if near:
                nearest = min(near, key=lambda c: q - c["end"])
                for c in near:
                    if overlaps(c, nearest):
                        negated.add(c["id"])
            start = q + 1
    return negated


# ---------------------------------------------------------------- v5 弱点修复
# v5 规则5 介绍线索锚点：『我是/我叫/这是/他叫…』紧邻候选时直接作为人物锚点，
# 跳过模型门控（同否定门控方法论：确定性表层句法由代码处理）。根因：laya 对
# 自我介绍句中的『小X』类名字 noul 系统性低分——实测『我是小张，坐在副驾』
# noul 0.4407 < 0.5 漏检，而同句位置问题 0.9995 全对（人物门是唯一失效环节）。
# 否定门控先行（『我不是X』『别叫我X』的 X 已被剔除），与本规则不冲突；
# 恰有一个候选被线索紧邻时才生效，多个命中视为歧义、交回模型裁决。
INTRO_CUES = ("我是", "我叫", "叫我", "喊我", "称呼我", "我的名字是",
              "这是", "那是", "他叫", "她叫", "他是", "她是")


def _cue_dedup(hits):
    """线索命中去歧义：同一起点的多个命中（田子 ⊂ 田子涵）是同一提及的
    不同写法，取最长者；命中跨多个起点才是真歧义，返回 None 交回模型裁决。"""
    if not hits:
        return None
    if len({c["start"] for c in hits}) != 1:
        return None
    return max(hits, key=lambda c: c["end"])


def _intro_cued(text, cands):
    """返回恰被介绍线索紧邻引导（线索止于候选起点）的唯一候选，否则 None。"""
    hits = [c for c in cands
            if any(text[max(0, c["start"] - len(cue)):c["start"]] == cue
                   for cue in INTRO_CUES)]
    return _cue_dedup(hits)


def _brand_cued(text, cands):
    """恰有一个候选被『[公司词]的』紧邻引导时返回该候选，否则 None。

    『小鹏的周平』『日产的小王』——公司词+的+候选 是确定性表层句法（该候选
    必是人物提及），同介绍线索方法论直接作锚点、跳过模型门控。与两用词清理
    配套：小鹏 已按公司义从候选剔除，但若剩余唯一候选走单候选 noul 门会因
    『X的Y』句式低分漏检（实测 周平 <0.5），本规则兜住该路径。
    """
    hits = []
    for c in cands:
        if text[max(0, c["start"] - 1):c["start"]] != "的":
            continue
        for w in COMPANY_WORDS_BY_LEN:
            s = c["start"] - 1 - len(w)
            if s >= 0 and text[s:c["start"] - 1] == w:
                hits.append(c)
                break
    return _cue_dedup(hits)


# v5 规则6 位置文本证据绑定：原文中与人名确定性绑定的座位词优先于模型回答。
# 根因：laya 位置 choice 对左右系统性混淆——实测『后排左边是我女儿小雨』答
# 后排右边 0.307 > 后排左边 0.189，证据词就在原文逐字出现。
# 只在「绑定词与模型回答都是座位级且属不同座位」时覆盖：不动区域级回答、
# 模型判『未提及』时不补填（避免与否定句/自我介绍 gold 位置为空的约定冲突）、
# 同座异名（右后→右后座）保留模型答案（gold 允许同义映射，51 集实测 2 行）。
LOC_SEAT_GROUP = {
    "主驾": "driver", "主驾驶": "driver", "驾驶位": "driver",
    "副驾": "passenger", "副驾驶": "passenger",
    "后排左边": "rear-left", "左后": "rear-left", "左后座": "rear-left",
    "后排左": "rear-left",
    "后排右边": "rear-right", "右后": "rear-right", "右后座": "rear-right",
    "后排右": "rear-right",
}
# 原文扫描词表：座位级（含同义变体）+ 区域级（区域词只参与绑定检测，不触发覆盖）。
# 按长度降序贪心匹配——『后排左边』不会再被拆成『后排』+『左边』两个词。
LOC_TEXT_WORDS = tuple(sorted(LOC_SEAT_GROUP, key=len, reverse=True)) + (
    "前排", "后排", "左边", "右边")
LOC_BIND_MAX_GAP = 8
LOC_BIND_STOP = "。！？；!?;"
LOC_FWD_CONNECT = ("是", "坐", "在")
LOC_BACK_CONNECT = ("在", "坐")


def _loc_occurrences(text):
    """按长度降序贪心扫描原文中的位置词，返回 [(start, end, word), ...]。"""
    occ = []
    i = 0
    while i < len(text):
        for w in LOC_TEXT_WORDS:
            if text.startswith(w, i):
                occ.append((i, i + len(w), w))
                i += len(w)
                break
        else:
            i += 1
    return occ


def _bound_loc_word(text, cand, others):
    """原文中与 cand 确定性绑定的位置词（最近者优先），无绑定返回 None。

    前向 [loc]…是/坐/在…[name]（『后排左边是我女儿小雨』）；
    后向 [name]…在/坐…[loc]（『王师傅在主驾』『陈部长，坐在主驾』）。
    窗口 ≤ LOC_BIND_MAX_GAP 字、不跨句读（。！？；）、不跨越其他赢家出现。
    """
    name_s, name_e = cand["start"], cand["end"]

    def region_ok(a, b):
        if any(ch in LOC_BIND_STOP for ch in text[a:b]):
            return False
        for o in others:
            p = text.find(o["text"])
            while p >= 0:
                if p < b and p + len(o["text"]) > a:
                    return False
                p = text.find(o["text"], p + 1)
        return True

    best = None  # (gap, word)
    for s, e, w in _loc_occurrences(text):
        if e <= name_s:  # 前向绑定
            gap = name_s - e
            if (gap <= LOC_BIND_MAX_GAP and region_ok(e, name_s)
                    and (gap == 0 or any(k in text[e:name_s]
                                         for k in LOC_FWD_CONNECT))
                    and (best is None or gap < best[0])):
                best = (gap, w)
        elif s >= name_e:  # 后向绑定
            gap = s - name_e
            if (gap <= LOC_BIND_MAX_GAP and region_ok(name_e, s)
                    and any(k in text[name_e:s] for k in LOC_BACK_CONNECT)
                    and (best is None or gap < best[0])):
                best = (gap, w)
    return best[1] if best else None


# v5 规则7 公司/品牌信息提取：常见汽车与互联网公司词典 + 就近绑定到人物。
# 品牌词是确定性词表、『[品牌]的[人名]』是表层句法——同否定/关系/位置绑定
# 方法论：代码处理，零模型调用。理想/大众/奔驰/小米/苹果 等与普通词同形的
# 条目保留（座舱语境下紧邻人名时几乎总是公司义），误绑风险由绑定闸门控制。
COMPANY_WORDS = (
    # 汽车整车（合资/自主/新势力）
    "日产", "吉利", "比亚迪", "特斯拉", "宝马", "奔驰", "奥迪", "大众",
    "丰田", "本田", "理想", "蔚来", "小鹏", "长城", "奇瑞", "沃尔沃",
    "马自达", "福特", "现代", "起亚", "零跑", "哪吒", "极氪", "问界",
    "智界", "享界", "领克", "坦克", "五菱", "红旗", "捷豹", "路虎",
    "保时捷", "雷克萨斯", "凯迪拉克", "别克", "雪佛兰", "斯巴鲁",
    "三菱", "标致", "雪铁龙", "斯柯达", "英菲尼迪", "宾利",
    "劳斯莱斯", "法拉利", "兰博基尼", "迈巴赫",
    # 汽车供应链/智驾
    "博世", "采埃孚", "宁德时代", "地平线", "黑芝麻", "亿咖通",
    "德赛西威", "均胜电子", "禾赛", "速腾聚创", "图达通", "Momenta",
    # 互联网/科技
    "阿里巴巴", "阿里", "腾讯", "百度", "华为", "小米", "字节跳动",
    "字节", "美团", "京东", "网易", "滴滴", "拼多多", "快手",
    "哔哩哔哩", "大疆", "商汤", "旷视", "科大讯飞", "联想", "海尔",
    "格力", "微软", "谷歌", "苹果", "亚马逊", "英伟达", "英特尔",
    "高通", "三星", "台积电", "大众点评",
)
# 长度降序 + 字典序（确定性迭代，跨进程一致——set 迭代序受哈希随机化影响）
COMPANY_WORDS_BY_LEN = tuple(sorted(set(COMPANY_WORDS), key=lambda w: (-len(w), w)))
COMPANY_BIND_MAX_GAP = 4   # 『日产的小王』gap=1；『日产公司的小王』gap=3
COMPANY_CONNECT = ("的", "是", "在", "来自", "属于")


def _company_spans(text):
    """文本中全部公司词出现区间 [(start, end), ...]（包含清理与绑定共用）。"""
    spans = []
    for w in COMPANY_WORDS_BY_LEN:
        p = text.find(w)
        while p >= 0:
            spans.append((p, p + len(w)))
            p = text.find(w, p + 1)
    return spans


def _company_for(text, cand, winners):
    """原文中与 cand 就近绑定的公司/品牌词（最近者优先，同距前向优先），无则 None。

    winners = 全部赢家候选（含 cand 自身）。防误绑三道闸：品牌出现与任何
    赢家人名重叠不绑（『我是小鹏』不把人名绑成公司）；绑定窗口跨越其他
    赢家出现不绑（『日产的小王和小李』不给小李绑日产）；gap>0 时窗口须含
    连接词（的/是/在/来自/属于——『[品牌]的[人名]』主模式，『小王和吉利』不绑）。
    """
    name_spans = []
    for w in winners:
        p = text.find(w["text"])
        while p >= 0:
            name_spans.append((p, p + len(w["text"])))
            p = text.find(w["text"], p + 1)

    def bindable(a, b, fwd, gap):
        if gap > COMPANY_BIND_MAX_GAP:
            return False
        if any(ch in LOC_BIND_STOP for ch in text[a:b]):
            return False
        if any(ns < b and ne > a for ns, ne in name_spans):
            return False
        if not (fwd and gap == 0) \
                and not any(k in text[a:b] for k in COMPANY_CONNECT):
            return False
        return True

    best = None  # (gap, rank, word)；rank: 前向0 后向1
    for word in COMPANY_WORDS_BY_LEN:
        p = text.find(word)
        while p >= 0:
            e = p + len(word)
            if not any(p < ne and e > ns for ns, ne in name_spans):
                if e <= cand["start"]:
                    gap = cand["start"] - e
                    if bindable(e, cand["start"], True, gap) \
                            and (best is None or (gap, 0) < (best[0], best[1])):
                        best = (gap, 0, word)
                elif p >= cand["end"]:
                    gap = p - cand["end"]
                    if bindable(cand["end"], p, False, gap) \
                            and (best is None or (gap, 1) < (best[0], best[1])):
                        best = (gap, 1, word)
            p = text.find(word, p + 1)
    return best[2] if best else None


def _title_split(cands, winner):
    """v4 称谓后缀拆分：返回 (替换候选, 称谓后缀)。

    赢家形如『X+称谓』（陈同学）且原文有候选恰止于称谓起点（小陈，
    须与赢家重叠——同一人物的不同写法）时返回该候选；无替换则为 None。
    """
    t = winner["text"]
    for suf in SPLIT_SUFFIXES:
        if t.endswith(suf) and len(t) > len(suf):
            split_pos = winner["end"] - len(suf)
            repl = next((c for c in cands
                         if c["end"] == split_pos and overlaps(c, winner)), None)
            return repl, suf
    return None, None


def _following_title(text, cand):
    """原文中紧跟候选之后的称谓后缀（小陈 后接 同学），无则 None。"""
    for suf in SPLIT_SUFFIXES:
        if text[cand["end"]:cand["end"] + len(suf)] == suf:
            return suf
    return None


def _relation_from_cues(text, cand, others):
    """v4③ 试点：按表层线索给赢家指派说话人关系。

    本人线索优先；其余取距人名最近的关系词。线索只向前关联（结束于人名
    之前）、距离 ≤ RELATION_CUE_MAX_GAP、且不跨越其他赢家——『我是胡平，
    这是我朋友吴刚』的 我是 不能越过 胡平 给 吴刚 指派本人。
    找不到线索返回 ""（未提及）。
    """
    start = cand["start"]

    def usable(cue_end):
        if not 0 <= start - cue_end <= RELATION_CUE_MAX_GAP:
            return False
        return not any(o["start"] >= cue_end and o["end"] <= start
                       for o in others)

    best_rel, best_end = "", -1
    for rel, cues in RELATION_CUES:
        for cue in cues:
            pos = text.rfind(cue, 0, start)
            if pos < 0:
                continue
            cue_end = pos + len(cue)
            if not usable(cue_end):
                continue
            if rel == "说话人本人":
                return rel
            if cue_end > best_end:
                best_rel, best_end = rel, cue_end
    return best_rel


def group_overlapping(passing):
    """把区间重叠的候选传递性地聚成组。"""
    groups = []
    for item in passing:
        for g in groups:
            if any(overlaps(item[0], other[0]) for other in g):
                g.append(item)
                break
        else:
            groups.append([item])
    return groups


def build_cluster_questions(clusters):
    """第二轮消歧：重叠候选组内让 Jev 选出最准确的称呼。"""
    questions = {}
    for j, cluster in enumerate(clusters):
        criteria = {item[0]["text"]: "" for item in cluster}
        criteria["都不是"] = ""
        questions[f"clu{j}"] = {
            "type": "choice",
            "instructions": (
                "下面几个片段都出自这句话且互相包含，可能是同一个人物的不同写法。"
                "这句话中对这个人最准确的名字或称呼是哪个？"
            ),
            "criteria": criteria,
        }
    return questions


# ---------------------------------------------------------------- 后端

def _normalize_answer(v):
    """把后端返回统一成 {'choice'|'noul': ..., 'probabilities': ..., 'confidence': ...}。"""
    if isinstance(v, (int, float)):
        return {"noul": float(v)}
    if isinstance(v, dict):
        return v
    return {}


class TypeSafeBackend:
    """TypeSafe 官方 API：POST /v1/systemone。"""

    def __init__(self):
        import requests
        self._requests = requests
        self.model = JEV_MODEL

    def ask(self, state, questions):
        payload = {"state": state, "questions": questions}
        if self.model:
            payload["model"] = self.model
        headers = {
            "Authorization": f"Bearer {TYPESAFE_API_KEY}",
            "Content-Type": "application/json",
        }
        last_err = None
        for attempt in range(3):
            try:
                resp = self._requests.post(
                    TYPESAFE_API_URL, json=payload, headers=headers, timeout=120
                )
                if resp.status_code == 429:
                    wait = float(resp.headers.get("retry-after", 2 * (attempt + 1)))
                    time.sleep(wait)
                    last_err = f"429 rate limited: {resp.text[:200]}"
                    continue
                resp.raise_for_status()
                data = resp.json()
                answers = data.get("answers", data)
                return {k: _normalize_answer(v) for k, v in answers.items()}
            except Exception as e:  # noqa: BLE001 - 网络错误统一重试
                last_err = str(e)
                time.sleep(2 * (attempt + 1))
        raise RuntimeError(f"TypeSafe API 调用失败: {last_err}")


class OpenJevBackend:
    """本地开源复现（kotoba-lang/typed-decisions）。

    DeBERTa 版上下文只有 512 token（中文在英文词表下 token 膨胀严重，
    超限 collator 会直接抛 ValueError），因此用模型自带 tokenizer
    精确计数，按预算把问题分块，每块一次前向，结果合并返回。
    """

    def __init__(self):
        from typed_decisions.open_jev import OpenJev
        self.model = OpenJev.from_pretrained(OPEN_JEV_MODEL)
        self._collator = self.model.collator
        # 实测（M4 Pro/MPS, bench_perf.py）：批量前向比串行慢 ~33%（p50 1.88s vs 1.35s），
        # 原因是 collator 把批内分块 padding 到最大尺寸，浪费超过批处理收益。
        # 数值上两者一致（160 问题最大概率差 9.8e-07），故默认串行，批量仅留作 CUDA 场景实验。
        self.batched = os.getenv("JEV_OPENJEV_BATCHED", "0") == "1"
        self.last_chunks = 0

    def _question_tokens(self, q):
        # 与 collator.encode_one 相同的计数：1 + 指令 + 每选项 (1 + token 数)
        n = 1 + len(self._collator._ids(q["instructions"]))
        n += sum(1 + len(self._collator._ids(o)) for o in q.get("criteria", {}))
        return n

    def _pack(self, state, questions):
        state_tok = min(len(self._collator._ids(state)), self._collator.max_state)
        budget = self._collator.max_len - 3 - state_tok - 8
        chunks, cur, used = [], {}, 0
        for qid, q in questions.items():
            cost = self._question_tokens(q)
            if cost > budget:
                raise RuntimeError(f"单个问题超出 open-jev 上下文预算: {qid}")
            if cur and used + cost > budget:
                chunks.append(cur)
                cur, used = {}, 0
            cur[qid] = q
            used += cost
        if cur:
            chunks.append(cur)
        return chunks

    @staticmethod
    def _to_openjev(q):
        if q["type"] == "choice":
            return {"type": "choice", "instructions": q["instructions"],
                    "options": list(q["criteria"].keys())}
        return {"type": q["type"], "instructions": q["instructions"]}

    def ask(self, state, questions):
        chunks = self._pack(state, questions)
        self.last_chunks = len(chunks)
        if self.batched:
            return self._ask_batched(state, chunks)
        out = {}
        for chunk in chunks:
            results = self.model.decide(
                state, [self._to_openjev(q) for q in chunk.values()])
            for qid, r in zip(chunk.keys(), results):
                out[qid] = _normalize_answer(r)
        return out

    def _ask_batched(self, state, chunks):
        """所有分块打包成一次 batch 前向。

        与 OpenJev.decide() 逐式对应（同样的 _question/collator/readout），
        只是 batch 维 > 1：N 个分块一次前向，而不是 N 次串行前向。
        """
        import torch
        from typed_decisions.schema import readout
        oj = self.model
        batch_qs = [
            [oj._question(i, self._to_openjev(q)) for i, q in enumerate(chunk.values())]
            for chunk in chunks
        ]
        b = oj.collator([(state, qs) for qs in batch_qs], oj.device)
        with torch.no_grad():
            logits = oj.model(b["input_ids"], b["attention_mask"], b["opt_pos"],
                              b["opt_mask"], b["q_pos"], b["seg"]).float()
        probs = (logits / oj.model.temperature).softmax(-1)
        out = {}
        for bi, (chunk, qs) in enumerate(zip(chunks, batch_qs)):
            qids = list(chunk.keys())
            for qi, q in enumerate(qs):
                p = probs[bi, qi, :len(q.options)].tolist()
                r = readout(q.kind, p)
                if q.kind == "choice":
                    ans = {"choice": q.options[r["choice"]],
                           "probabilities": dict(zip(q.options, p)),
                           "confidence": r["confidence"]}
                elif q.kind == "score":
                    ans = {"score": r["score"],
                           "probabilities": dict(zip(q.options, p)),
                           "confidence": r["confidence"]}
                else:
                    ans = {"noul": r["noul"]}
                out[qids[qi]] = ans
        return out


class LayaBackend:
    """Laya（convaiinnovations，Apache-2.0）本地后端。

    与 open-jev 的关键差异：所有问题在单次前向中回答（无 512-token 分块），
    multilingual checkpoint（mmBERT-base）支持 CJK。

    已知怪癖（模型卡 Honest Limits）：noul 在明确为真的输入上可能只给 ~0.5，
    官方建议改写成中性键两选项 choice。本后端自动做该改写并把 P("A")
    映射回 noul 概率，管线其余部分（阈值/组装）不感知。
    """

    def __init__(self):
        import laya
        self.agent = laya.load(LAYA_MODEL, subfolder=LAYA_SUBFOLDER,
                               calibration=LAYA_CALIBRATION or None)
        self.last_chunks = 1  # 单次前向，兼容 PERF 统计

    def ask(self, state, questions):
        transformed, noul_ids = {}, set()
        for qid, q in questions.items():
            if q["type"] == "noul":
                noul_ids.add(qid)
                transformed[qid] = {
                    "type": "choice",
                    "instructions": q["instructions"] + "？",
                    "criteria": {
                        "A": "是，" + q["instructions"],
                        "B": "否，" + q["instructions"] + " 不成立",
                    },
                }
            else:
                transformed[qid] = q
        result = self.agent.predict(state, transformed)
        out = {}
        for qid, ans in result.get("answers", {}).items():
            if qid in noul_ids:
                probs = ans.get("probabilities") or {}
                out[qid] = {"noul": float(probs.get("A", 0.0))}
            else:
                out[qid] = _normalize_answer(ans)
        return out


class DryRunBackend:
    """不调用模型，只打印将要发送的请求，便于离线检查问题设计。"""

    def ask(self, state, questions):
        print("=" * 60)
        print("[dryrun] state:", state)
        print(f"[dryrun] {len(questions)} 个问题:")
        for qid, q in questions.items():
            if q["type"] == "choice":
                opts = "/".join(q["criteria"].keys())
                print(f"  {qid} choice: {q['instructions']}  [{opts}]")
            else:
                print(f"  {qid} {q['type']}: {q['instructions']}")
        print("=" * 60)
        return {}


_backend = None


def get_backend():
    global _backend
    if _backend is not None:
        return _backend

    choice = JEV_BACKEND
    if choice == "auto":
        if TYPESAFE_API_KEY:
            choice = "typesafe"
        else:
            try:
                import laya  # noqa: F401
                choice = "laya"
            except ImportError:
                try:
                    import typed_decisions  # noqa: F401
                    choice = "openjev"
                except ImportError:
                    raise RuntimeError(
                        "未找到可用的 Jev 后端：请设置 TYPESAFE_API_KEY（官方 API），"
                        "或 pip install laya（本地 Laya，多语言），"
                        "或 pip install git+https://github.com/kotoba-lang/typed-decisions"
                        "（本地 open-jev），或设 JEV_BACKEND=dryrun 离线查看请求结构"
                    )
    if choice == "typesafe":
        if not TYPESAFE_API_KEY:
            raise RuntimeError("JEV_BACKEND=typesafe 需要 TYPESAFE_API_KEY")
        _backend = TypeSafeBackend()
    elif choice == "openjev":
        _backend = OpenJevBackend()
    elif choice == "laya":
        _backend = LayaBackend()
    elif choice == "dryrun":
        _backend = DryRunBackend()
    else:
        raise RuntimeError(f"未知 JEV_BACKEND: {choice}")
    print(f"[jev] 使用后端: {choice}")
    return _backend


# ---------------------------------------------------------------- 组装

def _answer_value(answers, qid, kind):
    a = answers.get(qid)
    if not a:
        return None
    return a.get(kind)


# 每次 extract_entities 调用的性能记录（bench_perf.py 读取；JEV_PERF=1 时打印）
PERF = {}


def _extract_forced(user_input, backend, cands, task="cockpit"):
    """锚点强制选择 + 增量 noul 扫描（laya 验证形态）。

    task="general"（通用人物提取试点）时属性轮改问 关系/身份，
    cockpit 路径的问题构造与顺序保持不变。

    第 1 轮：全部候选在同一个 choice 的 softmax 内竞争出锚点人物
    （单候选退化为 noul 阈值门；v5：介绍线索紧邻的候选直接作为
    确定性锚点，跳过模型门控）。第 2 轮：对剩余候选一次性做带
    锚点上下文的 noul 扫描，过阈值者全部保留、重叠者取概率最高。

    为什么第 2 轮不用强制选择：实测 laya 的 choice 对『小X』类名字
    系统性失效（三人句第 2 轮 小李 0.06/小宝 0.05，『没有人物』0.88），
    而同样的上下文 noul 给 小李 0.89/小宝 0.87。noul 扫描只在锚点
    确认后触发，纯负例句不会进入扫描，误报风险受控。
    否定句由代码层否定门控处理（见 _is_negated）。
    """
    t_round1 = t_round2 = 0.0
    n_q1 = n_q2 = 0

    # --- 第 1 轮：锚点 ---
    anchor = None
    fallback_winners = None
    intro = _intro_cued(user_input, cands)
    brand = None if intro is not None else _brand_cued(user_input, cands)
    t0 = time.perf_counter()
    try:
        if intro is not None:
            # v5 规则5：介绍线索紧邻 → 确定性锚点，跳过模型门控
            anchor = (intro, None)
        elif brand is not None:
            # v5 规则7：『[公司词]的』紧邻引导 → 确定性锚点（『小鹏的周平』）
            anchor = (brand, None)
        elif len(cands) == 1:
            c = cands[0]
            ans = backend.ask(user_input, {"pick_is": {
                "type": "noul",
                "instructions": f"『{c['text']}』是这句话中提到的某个人物的名字或称呼",
            }})
            p = (ans.get("pick_is") or {}).get("noul", 0.0)
            if p >= IS_PERSON_THRESHOLD:
                anchor = (c, p)
        else:
            criteria = {c["text"]: "" for c in cands}
            criteria[FORCED_NO_PERSON] = ""
            ans = backend.ask(user_input, {"pick": {
                "type": "choice",
                "instructions": FORCED_INSTRUCTIONS,
                "criteria": criteria,
            }})
            pick = ans.get("pick") or {}
            win = pick.get("choice")
            probs = pick.get("probabilities") or {}
            p_win = probs.get(win, 0.0)
            winner = next((c for c in cands if c["text"] == win), None)
            if winner is not None and win != FORCED_NO_PERSON and p_win >= FORCED_MIN_P:
                anchor = (winner, p_win)
            elif probs.get(FORCED_NO_PERSON, 0.0) < NO_PERSON_CONFIDENT:
                # 强制选择失败且『没有人物』置信度低（含糊）→ 普通 noul 扫描兜底。
                # 实测多人句 forced 常以低置信度漏判（没有人物 0.62 vs
                # noul 韦丽 0.85/田雪梅 0.96），而 forced 判对的负例
                # 置信度都高（0.77-0.86），据此门控回退。
                qs = {f"n{c['id']}": {
                    "type": "noul",
                    "instructions": f"『{c['text']}』是这句话中提到的某个人物的名字或称呼",
                } for c in cands}
                ans_n = backend.ask(user_input, qs)
                passed = [(c, (ans_n.get(f"n{c['id']}") or {}).get("noul", 0.0))
                          for c in cands]
                passed = [(c, p) for c, p in passed if p >= IS_PERSON_THRESHOLD]
                passed.sort(key=lambda item: -item[1])
                fallback_winners = []
                for item in passed:
                    if not any(overlaps(item[0], k[0]) for k in fallback_winners):
                        fallback_winners.append(item)
    except Exception as e:  # noqa: BLE001
        print(f"Jev 锚点轮失败: {e}")
    t_round1 = time.perf_counter() - t0
    n_q1 = 0 if (intro is not None or brand is not None) else 1
    if anchor is None and not fallback_winners:
        PERF.update({"t_round1": t_round1, "t_round2": 0.0,
                     "n_questions_r1": n_q1, "n_questions_r2": 0,
                     "winners": []})
        return {"persons": []}

    if fallback_winners:
        winners = fallback_winners[:FORCED_MAX_PERSONS]
        winner_paths = ["fallback"] * len(winners)
    else:
        winners = [anchor]
        winner_paths = ["intro_cue" if intro is not None
                        else ("brand_cue" if brand is not None
                              else ("single_noul" if len(cands) == 1 else "forced"))]

        # --- 第 2 轮：增量 noul 扫描剩余候选 ---
        a_cand = anchor[0]
        rest = [c for c in cands
                if c["text"] != a_cand["text"] and not overlaps(c, a_cand)]
        if rest:
            t0 = time.perf_counter()
            try:
                qs = {f"s{c['id']}": {
                    "type": "noul",
                    "instructions": f"除了已提到的『{a_cand['text']}』，"
                                    f"『{c['text']}』也是这句话中提到的某个人物的名字或称呼",
                } for c in rest}
                ans2 = backend.ask(user_input, qs)
                passed = [(c, (ans2.get(f"s{c['id']}") or {}).get("noul", 0.0))
                          for c in rest]
                passed = [(c, p) for c, p in passed if p >= IS_PERSON_THRESHOLD]
                passed.sort(key=lambda item: -item[1])
                for item in passed:
                    if not any(overlaps(item[0], k[0]) for k in winners):
                        winners.append(item)
                        winner_paths.append("scan")
            except Exception as e:  # noqa: BLE001
                print(f"Jev 增量扫描失败: {e}")
            t_round2 = time.perf_counter() - t0
            n_q2 = len(rest)
        winners = winners[:FORCED_MAX_PERSONS]
        winner_paths = winner_paths[:FORCED_MAX_PERSONS]

    # --- v4 称谓后缀拆分：陈同学 → 小陈 + 身份证据 同学（仅当原文存在
    # 止于称谓起点的重叠候选时触发；赢家已是小陈则只取后缀作身份证据）---
    split_winners = []
    for c, p in winners:
        repl, suf = _title_split(cands, c)
        if repl is None:
            suf = _following_title(user_input, c)
        split_winners.append((repl or c, p, suf))
    # 赢家分数/路径埋点（规则4阈值调优与 bench 诊断用）
    PERF["winners"] = [(c["text"], p, path)
                       for (c, p, _suf), path in zip(split_winners, winner_paths)]

    # v4 位置文本证据门（cockpit 专用）：原文没有任何位置词时，位置回答必是幻觉
    # （困难负例 马上/高架 全部幻觉出位置；否定句/自我介绍 gold 位置为空也被幻觉）。
    # 只验证「原文存在某个位置词」而非预测的具体选项词——gold 与问题措辞都允许
    # 右后→右后座 类同义映射，严格验证会误杀此类正确行（510 实测 2 行）
    loc_evidence = task != "general" and any(
        o in user_input for o in LOCATION_OPTIONS if o != NO_LOCATION)
    id_options = GENERAL_IDENTITY_OPTIONS if task == "general" else IDENTITY_OPTIONS
    id_question = GENERAL_IDENTITY_QUESTION if task == "general" else IDENTITY_QUESTION

    # --- 属性轮：cockpit 取 location/identity，general（试点）取 relation/identity ---
    persons = []
    for c, _p, suf in split_winners:
        t0 = time.perf_counter()
        if task == "general":
            # 关系由代码按 RELATION_CUES 直接指派（laya 零样本关系实测 0.565
            # 且错误系统性，见常量注释），只问 laya 身份
            qs = {
                "id": {"type": "choice",
                       "instructions": f"『{c['text']}』" + id_question,
                       "criteria": {o: "" for o in id_options}},
            }
        else:
            qs = {
                "loc": {"type": "choice",
                        "instructions": f"『{c['text']}』" + LOCATION_QUESTION,
                        "criteria": {o: "" for o in LOCATION_OPTIONS}},
                "id": {"type": "choice",
                       "instructions": f"『{c['text']}』" + IDENTITY_QUESTION,
                       "criteria": {o: "" for o in IDENTITY_OPTIONS}},
            }
        try:
            attr = backend.ask(user_input, qs)
        except Exception as e:  # noqa: BLE001
            print(f"Jev 属性轮失败: {e}")
            attr = {}
        t_round2 += time.perf_counter() - t0
        n_q2 += 1 if task == "general" else 2
        ident = _answer_value(attr, "id", "choice") or ""
        ident = "" if ident == NO_IDENTITY else ident
        if not _identity_supported(user_input, c["text"], ident):
            ident = ""
        # 拆分携带的身份证据优先（同学/老师 ∈ 选项表）；身份是名字自身
        # 后缀时按 gold 约定抑制（陈同学+同学 / 陈部长+部长 → 称谓留在名中，身份空）
        if suf and suf in id_options:
            ident = suf
        if ident and c["text"].endswith(ident):
            ident = ""
        if task == "general":
            others = [c2 for c2, _pw, _sf in split_winners if c2 is not c]
            rel = _relation_from_cues(user_input, c, others)
            persons.append({"name": c["text"], "identity": ident,
                            "relation": rel})
        else:
            loc = _answer_value(attr, "loc", "choice") or ""
            loc = "" if loc == NO_LOCATION else loc
            if loc and not loc_evidence:
                loc = ""
            if loc:
                # v5 规则6：原文确定性绑定的座位词与模型回答冲突 → 覆盖。
                # 座位级绑定词 vs 区域级粗答（后排左边 vs 后排）也覆盖——原文
                # 证据比模型粗答更具体；同座异名（右后→右后座）保留模型答案
                others = [c2 for c2, _pw, _sf in split_winners if c2 is not c]
                bound = _bound_loc_word(user_input, c, others)
                g_bound = LOC_SEAT_GROUP.get(bound) if bound else None
                if g_bound and LOC_SEAT_GROUP.get(loc) != g_bound:
                    loc = bound
            # v5 规则7：公司/品牌就近绑定（确定性词表，零模型调用）
            company = _company_for(user_input, c,
                                   [c2 for c2, _pw, _sf in split_winners]) or ""
            persons.append({"name": c["text"], "identity": ident,
                            "location": loc, "company": company})
    PERF.update({"t_round1": t_round1, "t_round2": t_round2,
                 "n_questions_r1": n_q1, "n_questions_r2": n_q2})
    persons.sort(key=lambda p: next(
        (c["start"] for c in cands if c["text"] == p["name"]), 0))
    return {"persons": persons}


def extract_entities(user_input):
    """与旧 extract_entities_with_model 相同的契约：输入文本，返回
    {"persons": [{"name", "identity", "location"}]}。
    """
    t_start = time.perf_counter()
    PERF.update({"n_candidates": 0, "n_questions_r1": 0, "n_chunks_r1": 0,
                 "n_questions_r2": 0, "t_candidates": 0.0, "t_round1": 0.0,
                 "t_round2": 0.0, "t_total": 0.0, "winners": []})
    try:
        backend = get_backend()
    except Exception as e:  # noqa: BLE001
        print(f"Jev 后端不可用: {e}")
        return {"persons": []}

    t0 = time.perf_counter()
    cands = generate_candidates(user_input)
    # 否定门控：被『不是/别叫我』引导的候选在代码层剔除，不进模型
    cands = [c for c in cands if not _is_negated(user_input, c["text"])]
    # v4 后向门控：人名后被缺席/纠正线索命中（『陈总不是我』『周老师没来』）的候选剔除
    back_neg = _backward_negated(user_input, cands)
    cands = [c for c in cands if c["id"] not in back_neg]
    PERF["t_candidates"] = time.perf_counter() - t0
    PERF["n_candidates"] = len(cands)
    if not cands:
        PERF["t_total"] = time.perf_counter() - t_start
        return {"persons": []}

    if ASSEMBLY_MODE == "forced":
        result = _extract_forced(user_input, backend, cands)
        PERF["t_total"] = time.perf_counter() - t_start
        if os.getenv("JEV_PERF"):
            print(f"[perf] 候选={PERF['n_candidates']} 选择轮={PERF['n_questions_r1']} "
                  f"属性问={PERF['n_questions_r2']} 一轮={PERF['t_round1']:.2f}s "
                  f"属性={PERF['t_round2']:.2f}s 总计={PERF['t_total']:.2f}s")
        return result

    questions = build_questions(cands)
    PERF["n_questions_r1"] = len(questions)
    t0 = time.perf_counter()
    try:
        answers = backend.ask(user_input, questions)
    except Exception as e:  # noqa: BLE001
        print(f"Jev 第一轮调用失败: {e}")
        return {"persons": []}
    PERF["t_round1"] = time.perf_counter() - t0
    PERF["n_chunks_r1"] = getattr(backend, "last_chunks", 0) or 0
    if not answers:  # dryrun
        PERF["t_total"] = time.perf_counter() - t_start
        return {"persons": []}

    passing = []
    for c in cands:
        p_is = _answer_value(answers, f"c{c['id']}_is", "noul")
        p_neg = _answer_value(answers, f"c{c['id']}_neg", "noul")
        if p_is is None:
            continue
        if p_is >= IS_PERSON_THRESHOLD and (p_neg or 0.0) < NEGATED_THRESHOLD:
            passing.append((c, p_is))

    groups = group_overlapping(passing)
    singletons = [g[0] for g in groups if len(g) == 1]
    ambiguous = [g for g in groups if len(g) > 1]

    winners = list(singletons)
    if ambiguous:
        cluster_questions = build_cluster_questions(ambiguous)
        PERF["n_questions_r2"] = len(cluster_questions)
        t0 = time.perf_counter()
        try:
            answers2 = backend.ask(user_input, cluster_questions)
        except Exception as e:  # noqa: BLE001
            print(f"Jev 第二轮消歧失败，按 is_person 概率回退: {e}")
            answers2 = {}
        PERF["t_round2"] = time.perf_counter() - t0
        for j, cluster in enumerate(ambiguous):
            picked = _answer_value(answers2, f"clu{j}", "choice")
            if picked == "都不是":
                continue
            match = [item for item in cluster if item[0]["text"] == picked]
            if match:
                winners.append(match[0])
            else:
                # 无有效回答（如第二轮调用失败）：回退按 is_person 概率取组内最高者
                winners.append(max(cluster, key=lambda item: item[1]))

    winners.sort(key=lambda item: item[0]["start"])
    loc_evidence = any(o in user_input for o in LOCATION_OPTIONS
                       if o != NO_LOCATION)
    persons = []
    for c, _p in winners:
        loc = _answer_value(answers, f"c{c['id']}_loc", "choice") or ""
        loc = "" if loc == NO_LOCATION else loc
        if loc and not loc_evidence:
            loc = ""
        ident = _answer_value(answers, f"c{c['id']}_id", "choice") or ""
        ident = "" if ident == NO_IDENTITY else ident
        if not _identity_supported(user_input, c["text"], ident):
            ident = ""
        # v4 同 forced 路径约定：身份是名字自身后缀时抑制（陈同学+同学 → 身份空）
        if ident and c["text"].endswith(ident):
            ident = ""
        persons.append({
            "name": c["text"],
            "identity": ident,
            "location": loc,
        })
    PERF["t_total"] = time.perf_counter() - t_start
    if os.getenv("JEV_PERF"):
        print(f"[perf] 候选={PERF['n_candidates']} 问题={PERF['n_questions_r1']}"
              f"+{PERF['n_questions_r2']} 块={PERF['n_chunks_r1']} "
              f"候选生成={PERF['t_candidates']*1000:.1f}ms "
              f"一轮={PERF['t_round1']:.2f}s 二轮={PERF['t_round2']:.2f}s "
              f"总计={PERF['t_total']:.2f}s")
    return {"persons": persons}


def extract_entities_general(user_input):
    """通用人物提取（v4③ 试点）：输入任意对话文本，返回
    {"persons": [{"name", "identity", "relation"}]}。

    与座舱版共享候选生成/否定门控/强制选择组装（task="general"），
    属性轮把「车内位置」换成「说话人关系」——打招呼/自我介绍/介绍他人/
    纠正句式为滩头，作为 mirix-cpp Q4 提取阶段的候选原型。
    始终走 forced 组装（laya 验证形态），不受 JEV_ASSEMBLY 影响。
    """
    t_start = time.perf_counter()
    PERF.update({"n_candidates": 0, "n_questions_r1": 0, "n_chunks_r1": 0,
                 "n_questions_r2": 0, "t_candidates": 0.0, "t_round1": 0.0,
                 "t_round2": 0.0, "t_total": 0.0, "winners": []})
    try:
        backend = get_backend()
    except Exception as e:  # noqa: BLE001
        print(f"Jev 后端不可用: {e}")
        return {"persons": []}

    t0 = time.perf_counter()
    cands = generate_candidates(user_input)
    cands = [c for c in cands if not _is_negated(user_input, c["text"])]
    back_neg = _backward_negated(user_input, cands)
    cands = [c for c in cands if c["id"] not in back_neg]
    PERF["t_candidates"] = time.perf_counter() - t0
    PERF["n_candidates"] = len(cands)
    if not cands:
        PERF["t_total"] = time.perf_counter() - t_start
        return {"persons": []}

    result = _extract_forced(user_input, backend, cands, task="general")
    PERF["t_total"] = time.perf_counter() - t_start
    if os.getenv("JEV_PERF"):
        print(f"[perf] 候选={PERF['n_candidates']} 一轮={PERF['t_round1']:.2f}s "
              f"属性={PERF['t_round2']:.2f}s 总计={PERF['t_total']:.2f}s")
    return result


# ---------------------------------------------------------------- 自检

def selftest():
    """离线检查：候选生成是否覆盖测试集 gold 中的全部人名（不需要任何 API）。"""
    import html
    import zipfile

    def read_rows(path):
        """返回 [(A 列文本, B 列文本), ...]，兼容内联字符串与共享字符串。"""
        z = zipfile.ZipFile(path)
        names = z.namelist()
        strings = []
        if "xl/sharedStrings.xml" in names:
            xml = z.read("xl/sharedStrings.xml").decode("utf-8")
            for s in re.findall(r"<si>(.*?)</si>", xml, re.S):
                ts = re.findall(r"<t[^>]*>(.*?)</t>", s, re.S)
                strings.append(html.unescape("".join(ts)))
        sheet = z.read("xl/worksheets/sheet1.xml").decode("utf-8")
        rows = []
        for r in re.findall(r"<row[^>]*>(.*?)</row>", sheet, re.S):
            cells = {}
            for col, attr, v in re.findall(
                    r'<c r="([A-Z]+)\d+"([^>]*)>(?:<v>(.*?)</v>)?', r):
                if v is None:
                    continue
                if 't="s"' in attr:
                    cells[col] = strings[int(v)]
                else:
                    cells[col] = html.unescape(v)
            rows.append((cells.get("A", ""), cells.get("B", "")))
        return rows

    gold_by_input = {}
    for inp, js in read_rows("test/测试集_结果.xlsx"):
        if not inp or not js.strip().startswith("{"):
            continue
        try:
            persons = json.loads(js).get("persons", [])
            gold_by_input[inp] = [p.get("name", "") for p in persons if p.get("name")]
        except json.JSONDecodeError:
            pass

    covered = missed = unmatched = 0
    for inp, _ in read_rows("test/测试集.xlsx"):
        if not inp:
            continue
        if inp not in gold_by_input:
            unmatched += 1
            continue
        cand_texts = {c["text"].lower() for c in generate_candidates(inp)}
        for name in gold_by_input[inp]:
            if name.lower() in cand_texts:
                covered += 1
            else:
                missed += 1
                print(f"  ✗ 未覆盖: {inp!r} 的 gold 人名 {name!r}")
    print(f"候选生成覆盖 gold 人名: {covered}/{covered + missed} "
          f"(无 gold 对照的用例 {unmatched} 条)")
    return missed == 0


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        sys.exit(0 if selftest() else 1)
    demo = "副驾是日产的小陈同学"
    print(f"输入: {demo}")
    print(json.dumps(extract_entities(demo), ensure_ascii=False, indent=2))
