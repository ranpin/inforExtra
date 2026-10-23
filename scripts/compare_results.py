"""把一份结果 xlsx 与 gold（test/测试集_结果.xlsx）逐条对比，输出指标与差异明细。

用法:
    python compare_results.py test/results/原始51_openjev.xlsx
"""

import html
import json
import os
import re
import sys
import zipfile

GOLD_FILE = "test/测试集_结果.xlsx"


def read_rows(path):
    """返回 [(A 列文本, B 列文本), ...]。

    兼容三种单元格形态：共享字符串（t="s" + <v> 索引）、内联值（t="str" + <v>）、
    openpyxl 写出的 inlineStr（<is><t>…</t></is>）。
    """
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
        for col, attr, inner in re.findall(
                r'<c r="([A-Z]+)\d+"([^>]*)(?:/>|>(.*?)</c>)', r, re.S):
            if inner is None:
                continue
            m = re.search(r"<v>(.*?)</v>", inner, re.S)
            if m:
                v = m.group(1)
                cells[col] = strings[int(v)] if 't="s"' in attr else html.unescape(v)
                continue
            ts = re.findall(r"<t[^>]*>(.*?)</t>", inner, re.S)
            if ts:
                cells[col] = html.unescape("".join(ts))
        rows.append((cells.get("A", ""), cells.get("B", "")))
    return rows


def load_persons(path):
    """返回 {用户输入: {人名小写: (identity, location)}}。"""
    out = {}
    for inp, js in read_rows(path):
        if not inp or not js.strip().startswith("{"):
            continue
        try:
            persons = json.loads(js).get("persons", [])
        except json.JSONDecodeError:
            persons = []
        out[inp] = {
            (p.get("name") or "").strip().lower():
                (p.get("identity") or "", p.get("location") or "")
            for p in persons if p.get("name")
        }
    return out


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    result_path = sys.argv[1]
    gold_path = sys.argv[2] if len(sys.argv) > 2 else GOLD_FILE

    # 模板族元数据（expand_test_set.py 产物）：按 gold 路径自动探测
    meta = {}
    if gold_path.endswith("_gold.xlsx"):
        meta_path = gold_path[: -len("_gold.xlsx")] + "_meta.json"
        if os.path.exists(meta_path):
            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)

    gold = load_persons(gold_path)
    result = load_persons(result_path)

    tp = fp = fn = 0
    loc_ok = loc_bad = id_ok = id_bad = 0
    row_exact = 0
    rows = 0
    errors = []
    fam = {}  # family -> dict(rows, exact, tp, fp, fn)

    def fstat(name):
        return fam.setdefault(name, {"rows": 0, "exact": 0, "tp": 0, "fp": 0, "fn": 0})

    for inp, gold_names in gold.items():
        if inp not in result:
            continue
        rows += 1
        f = fstat(meta.get(inp, "未分类"))
        f["rows"] += 1
        pred_names = result[inp]
        gold_keys = set(gold_names)
        pred_keys = set(pred_names)
        if gold_keys == pred_keys:
            row_exact += 1
            f["exact"] += 1
        for k in pred_keys & gold_keys:
            tp += 1
            f["tp"] += 1
            g_id, g_loc = gold_names[k]
            p_id, p_loc = pred_names[k]
            if g_loc == p_loc:
                loc_ok += 1
            else:
                loc_bad += 1
            if g_id == p_id:
                id_ok += 1
            else:
                id_bad += 1
        for k in pred_keys - gold_keys:
            fp += 1
            f["fp"] += 1
        for k in gold_keys - pred_keys:
            fn += 1
            f["fn"] += 1
        if gold_keys != pred_keys or any(
                gold_names[k][1] != pred_names[k][1] for k in pred_keys & gold_keys):
            errors.append((inp, sorted(gold_keys), sorted(pred_keys)))

    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0

    if rows == 0:
        print("没有可对比的用例（检查两份文件的用户输入列是否能对上）")
        sys.exit(1)

    print(f"对比: {result_path}  vs  {gold_path}")
    print(f"用例数: {rows}")
    print(f"人名级: P={prec:.3f}  R={rec:.3f}  F1={f1:.3f}  (TP={tp} FP={fp} FN={fn})")
    if tp:
        print(f"命中人名的位置准确率: {loc_ok}/{tp} = {loc_ok/tp:.3f}")
        print(f"命中人名的身份准确率: {id_ok}/{tp} = {id_ok/tp:.3f}")
    print(f"行级完全一致（人名集合相同）: {row_exact}/{rows} = {row_exact/rows:.3f}")

    if meta:
        print("\n按模板族分解:")
        print(f"  {'族':<14} {'行数':>5} {'行级全对':>8} {'TP':>4} {'FP':>4} {'FN':>4}")
        for name in sorted(fam, key=lambda n: -fam[n]["rows"]):
            s = fam[name]
            print(f"  {name:<14} {s['rows']:>5} {s['exact']:>5} ({s['exact']/s['rows']:.2f})"
                  f" {s['tp']:>4} {s['fp']:>4} {s['fn']:>4}")

    show = errors[:40]
    print(f"\n差异明细（共 {len(errors)} 条，显示前 {len(show)} 条）:")
    for inp, g, p in show:
        tag = f" [{meta[inp]}]" if inp in meta else ""
        print(f"  {inp}{tag}")
        print(f"    gold: {g if g else '[]'}")
        print(f"    pred: {p if p else '[]'}")


if __name__ == "__main__":
    main()
