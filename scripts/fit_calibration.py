"""在 510 条领域集上拟合 laya 温度校准（v4 Phase B）。

把管线同形态的问题（is_person A/B、锚点强制选择、位置/身份 choice、
二轮扫描式 A/B）按 gold 构造标签，records_from_labeled 收集原始 logits，
fit_temperatures 拟合类型级 + 按选项数分桶的温度，产物 laya_calibration.json
由 LayaBackend 自动加载（JEV_LAYA_CALIBRATION=0 可禁用）。

拟合只用 510 条模板集（分布内）；原始 51 条与对抗集保持 held-out 做验证。

用法: HF_HUB_OFFLINE=1 .venv/bin/python fit_calibration.py
"""
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import json  # noqa: E402
import sys  # noqa: E402

import laya  # noqa: E402
import laya.calibrate  # noqa: E402

# 脚本位于 scripts/ 下：仓库根保持在 sys.path（导入 jev_extract / compare_results）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import jev_extract  # noqa: E402
from jev_extract import (  # noqa: E402
    FORCED_INSTRUCTIONS, FORCED_NO_PERSON, IDENTITY_OPTIONS, IDENTITY_QUESTION,
    LOCATION_OPTIONS, LOCATION_QUESTION,
)
from compare_results import read_rows  # noqa: E402

IN_FILE = "test/测试集_扩充10x.xlsx"
GOLD_FILE = "test/测试集_扩充10x_gold.xlsx"
OUT_FILE = "test/results/laya_calibration.json"
MAX_SCAN_QUESTIONS = 4  # 每句二轮扫描式问题上限（控制单次前向规模）


def onehot(idx, k):
    v = [0.0] * k
    v[idx] = 1.0
    return v


def build_pairs():
    """(state, questions, targets) 三元组：问题与管线同形态，标签来自 gold。"""
    gold_by_input = {}
    for inp, js in read_rows(GOLD_FILE):
        if inp and js.strip().startswith("{"):
            gold_by_input[inp] = json.loads(js).get("persons", [])

    pairs = []
    for inp, _ in read_rows(IN_FILE):
        if not inp or inp == "用户输入" or inp not in gold_by_input:
            continue
        persons = gold_by_input[inp]
        gold_names = {p["name"] for p in persons if p.get("name")}
        # 候选与管线完全同源（生成 + 前向/后向否定门控）
        cands = jev_extract.generate_candidates(inp)
        cands = [c for c in cands if not jev_extract._is_negated(inp, c["text"])]
        back_neg = jev_extract._backward_negated(inp, cands)
        cands = [c for c in cands if c["id"] not in back_neg]
        if not cands:
            continue

        questions, targets = {}, {}
        # 1) is_person A/B（与 LayaBackend 的 noul→choice 改写同形）
        for c in cands:
            stmt = f"『{c['text']}』是这句话中提到的某个人物的名字或称呼"
            qid = f"is{c['id']}"
            questions[qid] = {
                "type": "choice", "instructions": stmt + "？",
                "criteria": {"A": "是，" + stmt, "B": "否，" + stmt + " 不成立"},
            }
            targets[qid] = onehot(0 if c["text"] in gold_names else 1, 2)
        # 2) 锚点强制选择（全体候选 + 没有人物）
        crit = {c["text"]: "" for c in cands}
        crit[FORCED_NO_PERSON] = ""
        opts = list(crit)
        anchor = next((c["text"] for c in cands if c["text"] in gold_names),
                      FORCED_NO_PERSON)
        questions["forced"] = {"type": "choice",
                               "instructions": FORCED_INSTRUCTIONS,
                               "criteria": crit}
        targets["forced"] = onehot(opts.index(anchor), len(opts))
        # 3) 每个 gold 人物的位置/身份
        for i, p in enumerate(persons):
            name = p.get("name", "")
            if not name:
                continue
            loc_opts = list(LOCATION_OPTIONS)
            questions[f"loc{i}"] = {
                "type": "choice", "instructions": f"『{name}』" + LOCATION_QUESTION,
                "criteria": {o: "" for o in loc_opts}}
            targets[f"loc{i}"] = onehot(
                loc_opts.index(p.get("location") or "未提及"), len(loc_opts))
            id_opts = list(IDENTITY_OPTIONS)
            questions[f"id{i}"] = {
                "type": "choice", "instructions": f"『{name}』" + IDENTITY_QUESTION,
                "criteria": {o: "" for o in id_opts}}
            targets[f"id{i}"] = onehot(
                id_opts.index(p.get("identity") or "未提及"), len(id_opts))
        # 4) 二轮扫描式（带锚点上下文），覆盖 round-2 的 k=2 分布
        anchor_cand = next((c for c in cands if c["text"] in gold_names), cands[0])
        others = [c for c in cands
                  if c["text"] != anchor_cand["text"]][:MAX_SCAN_QUESTIONS]
        for c in others:
            stmt = (f"除了已提到的『{anchor_cand['text']}』，"
                    f"『{c['text']}』也是这句话中提到的某个人物的名字或称呼")
            qid = f"s{c['id']}"
            questions[qid] = {
                "type": "choice", "instructions": stmt + "？",
                "criteria": {"A": "是，" + stmt, "B": "否，" + stmt + " 不成立"},
            }
            targets[qid] = onehot(0 if c["text"] in gold_names else 1, 2)
        pairs.append((inp, questions, targets))
    return pairs


def main():
    agent = laya.load(jev_extract.LAYA_MODEL, subfolder=jev_extract.LAYA_SUBFOLDER)
    temp_before = list(agent.temperature)
    by_opts_before = dict(agent.temperature_by_options)

    pairs = build_pairs()
    print(f"labeled pairs: {len(pairs)}")
    records = laya.calibrate.records_from_labeled(agent, pairs)
    print(f"records: {len(records)}")

    # 基线 ECE 用私有 _ece_report 在全量记录上算（与拟合后同口径对照）
    before = laya.calibrate._ece_report(records, temp_before, by_opts_before)
    tmap = agent.fit_temperatures(records, compute_ece=True, seed=42)

    print(f"checkpoint temperature: {temp_before}")
    print(f"fitted temperature:     {tmap['temperature']}")
    print(f"fitted by_options:      {tmap['temperature_by_options']}")
    print(f"n_by_bucket:            {tmap['n_by_bucket']}")
    print(f"clamped:                {tmap['temperature_clamped']}")
    print(f"ECE before (checkpoint temp, all records): {before}")
    print(f"ECE after  (fitted, 20% holdout):          {tmap.get('report')}")

    agent.save_calibration(OUT_FILE)
    print(f"saved -> {OUT_FILE}")


if __name__ == "__main__":
    main()
