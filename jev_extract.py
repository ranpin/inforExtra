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

JEV_BACKEND = os.getenv("JEV_BACKEND", "auto")  # auto | typesafe | openjev | dryrun
TYPESAFE_API_KEY = os.getenv("TYPESAFE_API_KEY", "")
TYPESAFE_API_URL = os.getenv("TYPESAFE_API_URL", "https://api.typesafe.ai/v1/systemone")
JEV_MODEL = os.getenv("JEV_MODEL", "jev-latest")
OPEN_JEV_MODEL = os.getenv("OPEN_JEV_MODEL", "com-kotobalabs/open-jev-deberta-v3-large")

IS_PERSON_THRESHOLD = float(os.getenv("JEV_IS_PERSON_THRESHOLD", "0.5"))
NEGATED_THRESHOLD = float(os.getenv("JEV_NEGATED_THRESHOLD", "0.5"))

MAX_CANDIDATES = int(os.getenv("JEV_MAX_CANDIDATES", "60"))

# 候选片段中出现这些字符即丢弃（功能词/动词，几乎不可能出现在姓名或称谓里）
STOPWORD_CHARS = set(
    "的我你他她它是在坐了有叫别不这那就都也还很吗吧啊呢给和跟向对把让被从到说"
)

LOCATION_OPTIONS = [
    "主驾", "副驾", "副驾驶", "前排", "后排",
    "后排左边", "后排右边", "左后", "左后座", "右后", "右后座",
    "左边", "右边", "未提及",
]

IDENTITY_OPTIONS = [
    "司机", "妻子", "丈夫", "儿子", "女儿", "孩子", "同事", "同学",
    "老师", "工程师", "朋友", "老板", "部长", "医生", "未提及",
]

LOCATION_QUESTION = "这个人物在这句话中坐在车内的哪个位置？按句子原话选择最匹配的说法"
IDENTITY_QUESTION = "这个人物在这句话中的身份或称谓是什么"
NO_LOCATION = "未提及"
NO_IDENTITY = "未提及"


# ---------------------------------------------------------------- 候选生成

def generate_candidates(text):
    """枚举人名候选片段，返回 [{'id', 'start', 'end', 'text'}, ...]。

    中文取所有 2-4 字子串（含功能词的丢弃），拉丁文取连续字母词。
    同文本只保留首次出现位置。
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
            add(i, i + width, s)

    return cands[:MAX_CANDIDATES]


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
                import typed_decisions  # noqa: F401
                choice = "openjev"
            except ImportError:
                raise RuntimeError(
                    "未找到可用的 Jev 后端：请设置 TYPESAFE_API_KEY（官方 API），"
                    "或 pip install git+https://github.com/kotoba-lang/typed-decisions"
                    "（本地 open-jev），或设 JEV_BACKEND=dryrun 离线查看请求结构"
                )
    if choice == "typesafe":
        if not TYPESAFE_API_KEY:
            raise RuntimeError("JEV_BACKEND=typesafe 需要 TYPESAFE_API_KEY")
        _backend = TypeSafeBackend()
    elif choice == "openjev":
        _backend = OpenJevBackend()
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


def extract_entities(user_input):
    """与旧 extract_entities_with_model 相同的契约：输入文本，返回
    {"persons": [{"name", "identity", "location"}]}。
    """
    t_start = time.perf_counter()
    PERF.update({"n_candidates": 0, "n_questions_r1": 0, "n_chunks_r1": 0,
                 "n_questions_r2": 0, "t_candidates": 0.0, "t_round1": 0.0,
                 "t_round2": 0.0, "t_total": 0.0})
    try:
        backend = get_backend()
    except Exception as e:  # noqa: BLE001
        print(f"Jev 后端不可用: {e}")
        return {"persons": []}

    t0 = time.perf_counter()
    cands = generate_candidates(user_input)
    PERF["t_candidates"] = time.perf_counter() - t0
    PERF["n_candidates"] = len(cands)
    if not cands:
        PERF["t_total"] = time.perf_counter() - t_start
        return {"persons": []}

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
    persons = []
    for c, _p in winners:
        loc = _answer_value(answers, f"c{c['id']}_loc", "choice") or ""
        ident = _answer_value(answers, f"c{c['id']}_id", "choice") or ""
        persons.append({
            "name": c["text"],
            "identity": "" if ident == NO_IDENTITY else ident,
            "location": "" if loc == NO_LOCATION else loc,
        })
    PERF["t_total"] = time.perf_counter() - t_start
    if os.getenv("JEV_PERF"):
        print(f"[perf] 候选={PERF['n_candidates']} 问题={PERF['n_questions_r1']}"
              f"+{PERF['n_questions_r2']} 块={PERF['n_chunks_r1']} "
              f"候选生成={PERF['t_candidates']*1000:.1f}ms "
              f"一轮={PERF['t_round1']:.2f}s 二轮={PERF['t_round2']:.2f}s "
              f"总计={PERF['t_total']:.2f}s")
    return {"persons": persons}


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
