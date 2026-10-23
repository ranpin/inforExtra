"""通用人物提取试点评测（v4③，2026-10-22）：打招呼/自我介绍/介绍他人/纠正为滩头。

战略用途：mirix-cpp Q4 提取阶段的候选原型——Mem0 的提取阶段用 LLM（φ），
类型化决策 ~44ms 本地提取是端侧替代；纠正句式（『不对我叫X』）是
UPDATE/SUPERSEDE 的提取侧信号。两项目保持边界不合并；本集只验证通用域可行性。

与座舱版的差异：去掉车内位置，新增「说话人关系」（本人/家人/朋友/同事/其他人/未提及）。
gold 约定：关系只认表层措辞（我→本人，妻子/儿子→家人，同事/朋友按原话），
未说明→未提及；身份只收职业称谓（同事/朋友归关系问题）；称谓留名中（王医生→身份空）。
gold 由构造时确定（构造即标注）。指标 = 人名级 P/R/F1 + 关系/身份准确率 + 行级全对。
产物：test/通用集.xlsx / test/通用集_gold.xlsx / test/results/通用集_laya_v5.xlsx
用法：.venv/bin/python eval_general.py
"""
import json
import os
import random

os.environ.setdefault("JEV_BACKEND", "laya")
os.environ.setdefault("JEV_CANDIDATE_FILTER", "surname")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import openpyxl  # noqa: E402

import jev_extract  # noqa: E402

RNG = random.Random(42)
SURNAMES = list("陈王李张刘杨黄赵吴周徐孙马朱胡郭何林罗高郑")
GIVENS = ["伟", "芳", "娜", "敏", "静", "磊", "军", "洋", "勇", "艳",
          "杰", "娟", "涛", "明", "超", "霞", "平", "刚", "桂英", "秀英"]
XIAO = ["小王", "小李", "小张", "小刘", "小陈", "小杨", "小赵", "小周", "小吴", "小徐"]
TITLES = ["总", "老师", "医生", "经理", "老板", "师傅", "工"]

SELF = "说话人本人"


def mkname():
    return RNG.choice(SURNAMES) + RNG.choice(GIVENS)


def p(n, relation="", identity=""):
    return {"name": n, "relation": relation, "identity": identity}


def fam_self_intro():
    """自我介绍族：relation=说话人本人（滩头句法）。"""
    ex = []
    plain = ["你好，我是{N}", "大家好，我是{N}", "我叫{N}",
             "诶你好，我是{N}，很高兴认识你", "你好你好，我是{N}", "我是{N}，请多关照"]
    for i in range(8):
        n = mkname()
        ex.append((plain[i % len(plain)].format(N=n), [p(n, SELF)]))
    xiao_t = ["叫我{X}就行", "我是{X}", "你好，叫我{X}就好", "他们都叫我{X}"]
    for i in range(6):
        x = RNG.choice(XIAO)
        ex.append((xiao_t[i % len(xiao_t)].format(X=x), [p(x, SELF)]))
    # 身份明示：relation=本人 + identity=职业称谓。
    # 工程师会造出姓氏尾巴碎片「程师」——通用域姓氏过滤的诚实探针
    ident_t = [("我是这里的工程师{N}", "工程师"), ("大家好，我是你们的老师{N}", "老师"),
               ("我是新来的护士{N}", "护士"), ("我是你们的经理{N}", "经理")]
    for i in range(6):
        n = mkname()
        t, ident = ident_t[i % len(ident_t)]
        ex.append((t.format(N=n), [p(n, SELF, ident)]))
    return ex


def fam_intro_other():
    """介绍他人族：家人/朋友/同事关系词。"""
    ex = []
    fam_words = ["妻子", "丈夫", "儿子", "女儿", "老婆", "哥"]
    for i in range(12):
        w = fam_words[i % len(fam_words)]
        n = mkname() if i % 2 == 0 else RNG.choice(XIAO)
        t = ["这是我{W}{N}", "这位是我{W}{N}", "给你介绍下，这是我{W}{N}"][i % 3]
        ex.append((t.format(W=w, N=n), [p(n, "家人")]))
    for i in range(6):
        w = ["同事", "朋友"][i % 2]
        n = mkname() if i % 3 else RNG.choice(XIAO)
        t = ["这位是我的{W}{N}", "这是我{W}{N}", "我{W}{N}也来了"][i % 3]
        ex.append((t.format(W=w, N=n), [p(n, w)]))
    return ex


def fam_greeting():
    """打招呼/称呼族：未说明关系 → 未提及。"""
    ex = []
    for i in range(8):
        s, t = RNG.choice(SURNAMES), RNG.choice(TITLES)
        tmpl = ["{S}{T}您好", "早上好，{S}{T}", "诶，{S}{T}也在啊", "{S}{T}，好久不见"][i % 4]
        ex.append((tmpl.format(S=s, T=t), [p(s + t)]))
    for i in range(4):
        x = RNG.choice(XIAO)
        t = ["你好{X}", "{X}，过来一下", "诶{X}也在啊", "{X}早上好"][i % 4]
        ex.append((t.format(X=x), [p(x)]))
    for i in range(3):
        n = mkname()
        t = ["{N}，好久不见", "哎{N}也在啊", "{N}你好呀"][i % 3]
        ex.append((t.format(N=n), [p(n)]))
    return ex


def fam_correction():
    """纠正族：UPDATE/SUPERSEDE 的提取侧信号（否定门控主战场）。"""
    ex = []
    for i in range(4):
        a, b = mkname(), mkname()
        ex.append((f"我不是{a}，我是{b}", [p(b, SELF)]))
    for i in range(3):
        a = RNG.choice(XIAO)
        b = RNG.choice([x for x in XIAO if x != a])
        ex.append((f"别叫我{a}，叫我{b}", [p(b, SELF)]))
    solo_t = ["不对，我叫{N}", "你认错人了，我是{N}", "搞错了，我是{N}", "诶不对不对，我叫{N}"]
    for i in range(4):
        n = mkname()
        ex.append((solo_t[i % len(solo_t)].format(N=n), [p(n, SELF)]))
    pair_t = ["我叫{B}，不是{A}", "记错了，并非{A}，我是{B}",
              "我不是{A}，叫我{B}就行", "别喊我{A}，喊我{B}"]
    for i in range(4):
        a, b = mkname(), mkname()
        ex.append((pair_t[i % len(pair_t)].format(A=a, B=b), [p(b, SELF)]))
    return ex


def fam_multi():
    """多人族：锚点 + 增量扫描。"""
    ex = []
    t4 = [("我是{A}，这是我朋友{B}", "朋友"), ("我是{A}，这位是我同事{B}", "同事"),
          ("大家好，我是{A}，旁边是我女儿{B}", "家人"), ("我是{A}，我妻子是{B}", "家人")]
    for t, rel_b in t4:
        a, b = mkname(), mkname()
        ex.append((t.format(A=a, B=b), [p(a, SELF), p(b, rel_b)]))
    t3 = [("这是{A}，那是{B}", ["", ""]), ("我带{A}和{B}一起来的", ["", ""]),
          ("他是{A}，我是{B}", ["", SELF])]
    for t, rels in t3:
        a, b = mkname(), mkname()
        ex.append((t.format(A=a, B=b), [p(a, rels[0]), p(b, rels[1])]))
    return ex


def fam_negative():
    """负例族：无人物 + 困难负例（姓氏开头普通词）。"""
    return [
        ("今天天气真好", []),
        ("帮我订个餐厅", []),
        ("马上就到了", []),
        ("张家口的天气怎么样", []),
        ("我想喝咖啡", []),
        ("我家狗叫旺财", []),
        ("高架上有点堵", []),
        ("江湖救急啊", []),
        ("这个包是小王的", [p("小王")]),
        ("会议室在三楼", []),
        ("帮我放首歌吧", []),
        ("白色的车真好看", []),
    ]


def fam_asr():
    """ASR 口语噪声（对照）：填充词/重复/句尾语气词。"""
    ex = []
    t4 = ["那个...你好我是{N}吧", "嗯，叫我{N}就行", "我是{N}，对，{N}", "诶那个，我是{N}啊"]
    for i in range(4):
        n = mkname()
        ex.append((t4[i % 4].format(N=n), [p(n, SELF)]))
    x4 = ["你好你好，我是{X}嗯", "就叫我{X}吧啊", "{X}...对，{X}也在", "喂，是{X}吗"]
    for i in range(4):
        x = RNG.choice(XIAO)
        rel = SELF if i < 2 else ""
        ex.append((x4[i % 4].format(X=x), [p(x, rel)]))
    return ex


FAMILIES = [
    ("自我介绍", fam_self_intro()),
    ("介绍他人", fam_intro_other()),
    ("打招呼称呼", fam_greeting()),
    ("纠正更新", fam_correction()),
    ("多人", fam_multi()),
    ("负例", fam_negative()),
    ("ASR噪声", fam_asr()),
]


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def main():
    print("通用人物提取试点评测（laya 最优配置，人名级 P/R/F1 + 关系/身份）")
    all_rows = []
    gtp = gfp = gfn = 0
    g_rel_ok = g_rel_tot = g_id_ok = g_row_ok = g_n = 0
    for fam, examples in FAMILIES:
        tp = fp = fn = rel_ok = rel_tot = id_ok = row_ok = 0
        fails = []
        for text, gold in examples:
            r = jev_extract.extract_entities_general(text)
            pred = r.get("persons", [])
            pmap = {x["name"]: x for x in pred}
            gmap = {x["name"]: x for x in gold}
            pn, gn = set(pmap), set(gmap)
            tp += len(pn & gn)
            fp += len(pn - gn)
            fn += len(gn - pn)
            for nm in pn & gn:
                rel_tot += 1
                if pmap[nm].get("relation", "") == gmap[nm].get("relation", ""):
                    rel_ok += 1
                if pmap[nm].get("identity", "") == gmap[nm].get("identity", ""):
                    id_ok += 1
            if pn == gn:
                row_ok += 1
            rel_bad = any(pmap[nm].get("relation", "") != gmap[nm].get("relation", "")
                          for nm in pn & gn)
            if pn != gn or rel_bad:
                fails.append((text, gold, pred))
            all_rows.append((fam, text, json.dumps(gold, ensure_ascii=False),
                             json.dumps(pred, ensure_ascii=False)))
        p_, r_, f_ = prf(tp, fp, fn)
        print(f"\n=== {fam}: P={p_:.3f} R={r_:.3f} F1={f_:.3f} "
              f"(TP={tp} FP={fp} FN={fn}) 关系={rel_ok}/{rel_tot} "
              f"身份={id_ok}/{rel_tot} 行级={row_ok}/{len(examples)} ===")
        for text, gold, pred in fails:
            print(f"  ✗ {text}\n      gold={json.dumps(gold, ensure_ascii=False)}"
                  f"\n      pred={json.dumps(pred, ensure_ascii=False)}")
        gtp += tp
        gfp += fp
        gfn += fn
        g_rel_ok += rel_ok
        g_rel_tot += rel_tot
        g_id_ok += id_ok
        g_row_ok += row_ok
        g_n += len(examples)
    gp, gr, gf = prf(gtp, gfp, gfn)
    print("\n" + "=" * 62)
    print(f"合计: P={gp:.3f} R={gr:.3f} F1={gf:.3f} (TP={gtp} FP={gfp} FN={gfn})")
    print(f"关系准确率={g_rel_ok}/{g_rel_tot}  身份准确率={g_id_ok}/{g_rel_tot}  "
          f"行级全对={g_row_ok}/{g_n}")

    os.makedirs("test/results", exist_ok=True)
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "通用集"
    ws.append(["用户输入"])
    for _, ex in FAMILIES:
        for t, _ in ex:
            ws.append([t])
    wb.save("test/通用集.xlsx")

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "gold"
    ws.append(["用户输入", "JSON输出"])
    for _, ex in FAMILIES:
        for t, g in ex:
            ws.append([t, json.dumps(g, ensure_ascii=False)])
    wb.save("test/通用集_gold.xlsx")

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "预测"
    ws.append(["族", "用户输入", "gold", "预测"])
    for row in all_rows:
        ws.append(list(row))
    wb.save("test/results/通用集_laya_v5.xlsx")
    print("\n产物已写入 test/通用集.xlsx, test/通用集_gold.xlsx, "
          "test/results/通用集_laya_v5.xlsx")


if __name__ == "__main__":
    main()
