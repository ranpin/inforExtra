"""把一份结果 xlsx 与 gold（test/测试集_结果.xlsx）逐条对比，输出指标与差异明细。

用法:
    python compare_results.py test/测试集_结果_jev_openjev.xlsx
"""

import html
import json
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
    gold = load_persons(GOLD_FILE)
    result = load_persons(result_path)

    tp = fp = fn = 0
    loc_ok = loc_bad = id_ok = id_bad = 0
    row_exact = 0
    rows = 0
    errors = []

    for inp, gold_names in gold.items():
        if inp not in result:
            continue
        rows += 1
        pred_names = result[inp]
        gold_keys = set(gold_names)
        pred_keys = set(pred_names)
        if gold_keys == pred_keys:
            row_exact += 1
        for k in pred_keys & gold_keys:
            tp += 1
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
        for k in gold_keys - pred_keys:
            fn += 1
        if gold_keys != pred_keys or any(
                gold_names[k][1] != pred_names[k][1] for k in pred_keys & gold_keys):
            errors.append((inp, sorted(gold_keys), sorted(pred_keys)))

    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0

    if rows == 0:
        print("没有可对比的用例（检查两份文件的用户输入列是否能对上）")
        sys.exit(1)

    print(f"对比: {result_path}  vs  {GOLD_FILE}")
    print(f"用例数: {rows}")
    print(f"人名级: P={prec:.3f}  R={rec:.3f}  F1={f1:.3f}  (TP={tp} FP={fp} FN={fn})")
    if tp:
        print(f"命中人名的位置准确率: {loc_ok}/{tp} = {loc_ok/tp:.3f}")
        print(f"命中人名的身份准确率: {id_ok}/{tp} = {id_ok/tp:.3f}")
    print(f"行级完全一致（人名集合相同）: {row_exact}/{rows} = {row_exact/rows:.3f}")
    print(f"\n差异明细（{len(errors)} 条）:")
    for inp, g, p in errors:
        print(f"  {inp}")
        print(f"    gold: {g if g else '[]'}")
        print(f"    pred: {p if p else '[]'}")


if __name__ == "__main__":
    main()
