"""模板族测试集生成器：把测试集扩充到 ~10 倍，构造即标注。

每条话语由模板嵌入 人名/身份/位置 参数生成，gold 标注随参数确定性产出
（无需人工标注，固定种子可复现）。原始 51 条及其 gold 原样并入。

模板族：品牌位置 / 无品牌位置 / 否定句 / 无人物负例（含姓氏字干扰的困难负例）/
多人 / 拉丁名 / 身份明示 / 自我介绍 / 同学边界 / 称谓无品牌

用法: .venv/bin/python expand_test_set.py
产物: test/测试集_扩充10x.xlsx        用户输入（511 条）
      test/测试集_扩充10x_gold.xlsx   用户输入 + JSON输出（gold）
      test/测试集_扩充10x_meta.json   话语 -> 模板族
"""

import json
import random
from collections import Counter

import pandas as pd

from compare_results import read_rows

SEED = 42
OUT_INPUT = "test/测试集_扩充10x.xlsx"
OUT_GOLD = "test/测试集_扩充10x_gold.xlsx"
OUT_META = "test/测试集_扩充10x_meta.json"

POSITIONS = ["主驾", "副驾", "前排", "后排", "后排左边", "后排右边",
             "左后", "右后", "左边", "右边", "左后座", "右后座"]
SIDES = ["左边", "右边"]
BRANDS = ["日产", "吉利", "比亚迪", "特斯拉", "宝马", "奔驰", "奥迪", "大众",
          "丰田", "本田", "理想", "蔚来", "小鹏", "长城", "奇瑞", "沃尔沃"]
BRAND_TYPOS = {"吉利": "激励"}
SURNAMES = list(
    "陈王李张刘杨黄赵吴周徐孙马朱胡郭何林罗高郑梁谢宋唐许韩冯邓曹彭曾肖田董袁潘"
    "蒋蔡余杜叶程苏魏吕丁任沈姚卢姜崔钟谭陆汪范金石廖贾夏韦方白邹孟熊秦邱江尹"
    "薛闫段雷侯龙史陶黎贺顾毛郝龚万钱严覃武戴莫孔向汤"
)
GIVENS = ["伟", "芳", "娜", "敏", "静", "磊", "军", "洋", "勇", "艳", "杰", "娟",
          "涛", "超", "明", "雪", "丽", "强", "平", "刚", "桂英", "秀兰", "建国",
          "晓燕", "雪梅", "子涵", "雨欣", "浩然", "欣怡", "俊杰"]
TITLES = ["总", "工", "老师", "部长", "经理", "医生", "师傅", "老板"]
IDENTITIES = ["同事", "孩子", "儿子", "女儿", "妻子", "丈夫", "朋友", "司机", "同学"]
LATIN = ["alice", "bob", "kevin", "lucy", "mike", "sara", "tom", "lily",
         "jack", "emma", "david", "nina"]
GREETINGS = ["给他打个招呼吧", "快给他打个招呼", "给他问个好", "跟他问个好",
             "问个好吧", "给他打个招呼"]
SIT = ["是", "坐的是", "坐着", "这位是"]

NEGATIVES_SAFE = [
    "今天天气不错", "帮我打开空调", "导航去最近的加油站", "把车窗关一下",
    "早上好今天心情不错", "车里有点冷", "下一首歌", "给我讲讲今天的新闻",
    "设置导航去公司", "打开座椅加热", "空调温度调高一点", "今天限号吗",
    "附近有什么好吃的", "帮我订个餐厅", "打开后备箱", "把音乐声音调小",
    "现在几点了", "外面下雨了吗", "打开双闪", "导航回家", "把空调调到25度",
    "给我讲个笑话", "今天有什么安排", "查一下电量还剩多少", "打开阅读灯",
    "我要听相声", "把天窗打开", "附近哪里有充电站", "播放新闻", "调低车内温度",
    "打开雾灯", "今天星期几", "帮我查一下限行政策", "把后排空调打开",
    "我想喝咖啡", "导航去机场", "关掉音乐", "打开导航", "把座椅放倒",
    "今天路上车多吗",
]
# 困难负例：含姓氏开头词但并非人名（考验模型判别，而非代码过滤）
NEGATIVES_HARD = [
    "张家口的天气怎么样", "高架上有点堵", "马上就到了", "石头上有只鸟",
    "田野里都是花", "白色的车好看", "黄金价格涨了吗", "孔明灯怎么放",
    "武则天的电视剧", "江湖救急", "张灯结彩的街道", "王炸炸弹",
    "李子的价格是多少", "宋城千古情好看吗",
]


def make_name(rng, style=None):
    """返回 (surface, gold_name, gold_identity)。"""
    style = style or rng.choice(["plain", "plain", "xiao", "title", "title"])
    if style == "plain":
        s = rng.choice(SURNAMES) + rng.choice(GIVENS)
    elif style == "xiao":
        s = "小" + rng.choice(SURNAMES)
    else:
        s = rng.choice(SURNAMES) + rng.choice(TITLES)
    return s, s, ""


def fam_brand_pos(rng):
    pos = rng.choice(POSITIONS)
    brand = rng.choice(BRANDS)
    if rng.random() < 0.05:
        brand = BRAND_TYPOS.get(brand, brand)
    surface, gname, gident = make_name(rng)
    r = rng.random()
    if r < 0.25:
        text = f"{pos}{rng.choice(SIT)}{brand}的{surface}"
    elif r < 0.45:
        v = "坐在" if rng.random() < 0.8 else rng.choice(["座在", "做在"])
        text = f"{v}{pos}的是{brand}的{surface}"
    elif r < 0.6:
        pos = rng.choice(SIDES)
        v = "坐" if rng.random() < 0.8 else rng.choice(["座", "做"])
        text = f"我{pos}{v}的是{brand}的{surface}"
    elif r < 0.75:
        text = f"我来介绍一下{pos}是{brand}的{surface}"
    else:
        text = f"{pos}是{brand}的{surface}"
    if rng.random() < 0.4:
        text += rng.choice(GREETINGS)
    return text, [{"name": gname, "identity": gident, "location": pos}]


def fam_pos_nobrand(rng):
    pos = rng.choice(POSITIONS)
    surface, gname, gident = make_name(rng)
    r = rng.random()
    if r < 0.4:
        text = f"{pos}是{surface}"
    elif r < 0.7:
        text = f"{pos}{rng.choice(SIT[1:])}{surface}"
    else:
        v = "坐在" if rng.random() < 0.8 else rng.choice(["座在", "做在"])
        text = f"{v}{pos}的是{surface}"
    if rng.random() < 0.3:
        text += rng.choice(GREETINGS)
    return text, [{"name": gname, "identity": gident, "location": pos}]


def fam_negation(rng):
    sur = rng.choice(SURNAMES)
    t1, t2 = rng.sample(TITLES, 2)
    r = rng.random()
    if r < 0.3:
        text = f"我不是{sur}{t1}我叫{sur}{t2}"
        gold_name = f"{sur}{t2}"
    elif r < 0.55:
        text = f"别叫我{sur}{t1}我是小{sur}"
        gold_name = f"小{sur}"
    elif r < 0.8:
        n1, _, _ = make_name(rng, "plain")
        n2, _, _ = make_name(rng, "plain")
        while n2 == n1:
            n2, _, _ = make_name(rng, "plain")
        text = f"我不是{n1}我是{n2}"
        gold_name = n2
    else:
        text = f"我叫{sur}{t2}不是{sur}{t1}"
        gold_name = f"{sur}{t2}"
    return text, [{"name": gold_name, "identity": "", "location": ""}]


def fam_negative(rng):
    pool = NEGATIVES_SAFE + NEGATIVES_HARD
    return rng.choice(pool), []


def fam_multi(rng):
    n = rng.choice([2, 2, 3])
    pos_pool = rng.sample(POSITIONS, n)
    persons, parts, used = [], [], set()
    for i in range(n):
        for _ in range(20):
            surface, gname, _ = make_name(rng)
            if gname not in used:
                break
        used.add(gname)
        pos = pos_pool[i]
        if rng.random() < 0.7:
            ident = rng.choice(IDENTITIES)
            parts.append(f"{pos}坐着我的{ident}{surface}")
            persons.append({"name": gname, "identity": ident, "location": pos})
        else:
            parts.append(f"{pos}是{surface}")
            persons.append({"name": gname, "identity": "", "location": pos})
    return "，".join(parts), persons


def fam_latin(rng):
    name = rng.choice(LATIN)
    honorific = rng.choice(["女士", "先生", "小姐"])
    r = rng.random()
    if r < 0.4:
        pos = rng.choice(POSITIONS)
        text = f"{pos}是{name}{honorific}"
    elif r < 0.7:
        pos = rng.choice(POSITIONS)
        text = f"{pos}坐着{rng.choice(BRANDS)}的{name}{honorific}"
    else:
        pos = None
        text = f"我是{name}"
    return text, [{"name": name, "identity": "",
                   "location": pos or ""}]


def fam_identity(rng):
    surface, gname, _ = make_name(rng, rng.choice(["plain", "xiao"]))
    ident = rng.choice(IDENTITIES)
    r = rng.random()
    if r < 0.5:
        pos = rng.choice(POSITIONS)
        text = f"{pos}坐着我的{ident}{surface}"
        loc = pos
    elif r < 0.75:
        text = f"这是我的{ident}{surface}"
        loc = ""
    else:
        pos = rng.choice(POSITIONS)
        text = f"{pos}是我的{ident}{surface}"
        loc = pos
    return text, [{"name": gname, "identity": ident, "location": loc}]


def fam_selfintro(rng):
    surface, gname, _ = make_name(rng)
    r = rng.random()
    if r < 0.3:
        text = f"大家好我是{surface}"
    elif r < 0.5:
        text = f"你好我是{surface}"
    elif r < 0.7:
        text = f"我叫{surface}"
    else:
        text = f"早上好我叫{surface}见到你们真高兴"
    return text, [{"name": gname, "identity": "", "location": ""}]


def fam_tongxue(rng):
    sur = rng.choice(SURNAMES)
    pos = rng.choice(POSITIONS)
    brand = rng.choice(BRANDS)
    if rng.random() < 0.5:
        text = f"{pos}是{brand}的小{sur}同学"
        gold = [{"name": f"小{sur}", "identity": "同学", "location": pos}]
    else:
        text = f"坐在{pos}的是{brand}的{sur}同学给他问个好"
        gold = [{"name": f"{sur}同学", "identity": "", "location": pos}]
    return text, gold


def fam_title_nobrand(rng):
    sur = rng.choice(SURNAMES)
    t = rng.choice(TITLES)
    pos = rng.choice(POSITIONS)
    if rng.random() < 0.5:
        text = f"{pos}是{sur}{t}"
    else:
        text = f"{pos}这位是{sur}{t}"
    if rng.random() < 0.3:
        text += rng.choice(GREETINGS)
    return text, [{"name": f"{sur}{t}", "identity": "", "location": pos}]


QUOTAS = [
    ("品牌位置", fam_brand_pos, 116),
    ("无品牌位置", fam_pos_nobrand, 50),
    ("否定句", fam_negation, 45),
    ("无人物负例", fam_negative, 54),
    ("多人", fam_multi, 60),
    ("拉丁名", fam_latin, 25),
    ("身份明示", fam_identity, 45),
    ("自我介绍", fam_selfintro, 20),
    ("同学边界", fam_tongxue, 25),
    ("称谓无品牌", fam_title_nobrand, 20),
]


def main():
    rng = random.Random(SEED)
    orig_inputs = [a for a, _ in read_rows("test/测试集.xlsx")
                   if a and a != "用户输入"]
    orig_gold = {}
    for a, b in read_rows("test/测试集_结果.xlsx"):
        if a and b.strip().startswith("{"):
            orig_gold[a] = b

    rows, seen = [], set()
    for inp in orig_inputs:
        if inp in seen:
            continue
        seen.add(inp)
        rows.append((inp, orig_gold.get(inp), "原始"))

    for fam_name, fn, quota in QUOTAS:
        made = tries = 0
        while made < quota and tries < quota * 30:
            tries += 1
            text, persons = fn(rng)
            if text in seen:
                continue
            seen.add(text)
            rows.append((text,
                         json.dumps({"persons": persons}, ensure_ascii=False),
                         fam_name))
            made += 1
        if made < quota:
            print(f"警告: {fam_name} 只生成 {made}/{quota} 条（模板空间耗尽）")

    pd.DataFrame({"用户输入": [r[0] for r in rows]}).to_excel(OUT_INPUT, index=False)
    gold_rows = [(t, g) for t, g, _ in rows if g is not None]
    pd.DataFrame({"用户输入": [r[0] for r in gold_rows],
                  "JSON输出": [r[1] for r in gold_rows]}).to_excel(OUT_GOLD, index=False)
    with open(OUT_META, "w", encoding="utf-8") as f:
        json.dump({t: fam for t, _, fam in rows}, f, ensure_ascii=False, indent=1)

    print(f"共 {len(rows)} 条（原始 {len(orig_inputs)}，新生成 {len(rows) - len(orig_inputs)}，"
          f"gold {len(gold_rows)} 条）")
    for fam_name, cnt in Counter(f for _, _, f in rows).most_common():
        print(f"  {fam_name}: {cnt}")


if __name__ == "__main__":
    main()
