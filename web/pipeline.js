/**
 * jevCabin 提取管线 JS 移植（浏览器 live 推理用）。
 *
 * 与 jev_extract.py（v4）逐函数对应：常量、候选生成、否定门控、锚点强制
 * 选择、增量扫描、称谓拆分、身份/位置证据门全部按行移植；唯一结构差异是
 * 后端适配层 askLaya（对应 Python LayaBackend.ask 的 noul→A/B 中性键改写）。
 * agent 由外部注入（@johnhenry/laya 的 load() 产物），本文件零依赖、
 * 可在 Node 用 mock agent 单测确定性部分。
 *
 * 对拍验证：51 条真实种子集 persons 与 Python 管线输出逐条一致
 * （test/results/原始51_laya强制选择v4.xlsx 为 gold）。
 */

// ---------------------------------------------------------------- 常量（与 jev_extract.py 逐字一致）

export const IS_PERSON_THRESHOLD = 0.5;
export const FORCED_MIN_P = 0.3;
export const NO_PERSON_CONFIDENT = 0.7;
export const FORCED_MAX_PERSONS = 5;
export const MAX_CANDIDATES = 60;

/** 每次调用的统计（镜像 Python PERF；live.js 读取供页面展示）。 */
export const PERF = { nCandidates: 0, nQ1: 0, nQ2: 0 };

export const FORCED_NO_PERSON = "没有人物";
export const FORCED_INSTRUCTIONS =
  "这句话中提到的某个人物的名字或称呼是哪个？只根据这句话判断，" +
  "品牌名、车内位置、方位词、普通名词都不算人物称呼；" +
  "被否定或纠正的称呼（如『我不是X』『别叫我X』中的X）也不算。" +
  "如果这句话没有提到任何人物，选『没有人物』。";

export const SURNAME_CHARS = new Set(
  "王李张刘陈杨黄赵吴周徐孙马朱胡郭何林罗高郑梁谢宋唐许韩冯邓曹彭曾肖田董袁潘" +
  "于蒋蔡余杜叶程苏魏吕丁任沈姚卢姜崔钟谭陆汪范金石廖贾夏韦付方白邹孟熊秦邱江" +
  "尹薛闫段雷侯龙史陶黎贺顾毛郝龚万钱严覃武戴莫孔向汤" +
  "翦仝佟佘菅庹慕言"
);
export const COMPOUND_SURNAMES = new Set([
  "欧阳", "司马", "诸葛", "上官", "夏侯", "皇甫", "尉迟", "公孙", "令狐", "长孙",
  "慕容", "司徒", "端木", "东方", "独孤", "南宫", "呼延", "西门", "第五", "淳于",
  "单于", "太叔", "申屠", "仲孙", "轩辕", "百里", "东郭", "南门", "羊舌", "微生",
]);
export const TITLE_HEAD_CHARS = new Set("老部经医师同");
export const TITLE_SUFFIXES = ["老师", "医生", "经理", "老板", "师傅"];
export const TITLE_PREFIX_BLOCK = new Set("看找做当像个位些问帮");
export const SPLIT_SUFFIXES = ["同学", "老师", "师傅", "先生", "小姐", "女士"];
export const BARE_TITLE_WORDS = new Set([
  "同学", "老师", "师傅", "先生", "小姐", "女士",
  "医生", "经理", "老板", "部长", "司机", "护士",
]);
export const TITLE_WORDS_FULL = [
  "工程师", "老师", "医生", "经理", "老板", "师傅", "同学",
  "学生", "护士", "司机", "部长", "先生", "小姐", "女士",
];
export const STOPWORD_CHARS = new Set(
  "的我你他她它是在坐了有叫别不这那就都也还很吗吧啊呢给和跟向对把让被从到说" +
  "快打招呼请问候兴见真早" +
  "哈嘛哦诶嗯呗啦咯呀哟哇嘞没" +
  "您"
);

export const LOCATION_OPTIONS = [
  "主驾", "副驾", "前排", "后排",
  "后排左边", "后排右边", "左后", "左后座", "右后", "右后座",
  "左边", "右边", "未提及",
];
// 未提及放第一位：对抗 laya 选项位置偏差（同 Python 约定）
export const IDENTITY_OPTIONS = [
  "未提及", "司机", "妻子", "丈夫", "儿子", "女儿", "孩子", "同事", "同学",
  "老师", "工程师", "朋友", "老板", "部长", "医生",
];
export const NO_LOCATION = "未提及";
export const NO_IDENTITY = "未提及";

export const LOCATION_QUESTION =
  "这个人物在这句话中坐在车内的哪个位置？按句子原话选择最匹配的说法，" +
  "注意严格区分左边和右边；句子没说位置就选未提及";
export const IDENTITY_QUESTION =
  "这个人物在这句话中的身份或称谓是什么？只有句子明确说明了身份" +
  "（如『同事小李』『我的孩子小宝』『司机王五』）才选对应项，没有明确说明就选未提及";

export const NEGATION_CUES = [
  "不是", "不叫", "别叫", "别说", "别喊", "不姓",
  "并非", "当成", "算不上", "哪里是", "别记成",
];
export const ABSENCE_CUES = [
  "不在车上", "没上车", "没来", "不来", "缺席", "不是我", "是别人",
];
export const ABSENCE_MAX_GAP = 8;

// ---------------------------------------------------------------- 候选生成

const CJK_FULL = /^[\u4e00-\u9fff]+$/;

function keepCjkSpan(s) {
  const w = s.length;
  if ((s[0] === "小" || s[0] === "阿") && w >= 2 && w <= 3) return true;
  if (w >= 2 && w <= 3 && SURNAME_CHARS.has(s[0])) return true;
  if (w >= 3 && w <= 4 && COMPOUND_SURNAMES.has(s.slice(0, 2))) return true;
  if (w === 3 && TITLE_SUFFIXES.includes(s.slice(1)) && !TITLE_PREFIX_BLOCK.has(s[0])) return true;
  return false;
}

function insideTitleWord(text, start) {
  for (const w of TITLE_WORDS_FULL) {
    let pos = text.indexOf(w);
    while (pos >= 0) {
      if (pos < start && start < pos + w.length) return true;
      pos = text.indexOf(w, pos + 1);
    }
  }
  return false;
}

export function generateCandidates(text) {
  const seen = new Set();
  const cands = [];
  const add = (start, end, s) => {
    if (seen.has(s)) return;
    seen.add(s);
    cands.push({ id: cands.length, start, end, text: s });
  };

  for (const m of text.matchAll(/[A-Za-z]{2,}/g)) {
    add(m.index, m.index + m[0].length, m[0]);
  }
  for (const width of [2, 3, 4]) {
    for (let i = 0; i + width <= text.length; i++) {
      const s = text.slice(i, i + width);
      if (!CJK_FULL.test(s)) continue;
      if ([...s].some((ch) => STOPWORD_CHARS.has(ch))) continue;
      if (!keepCjkSpan(s)) continue;
      if (insideTitleWord(text, i)) continue;
      add(i, i + width, s);
    }
  }

  // 碎片清理第一遍：称谓碎片（陈部⊂陈部长）/ 叠字碎片（王总王）/ 裸称谓（bob小姐 的 小姐）
  const cleaned = [];
  for (const c of cands) {
    const t = c.text;
    if (t.length === 2 && t[0] !== "小" && TITLE_HEAD_CHARS.has(t[1])
        && cands.some((o) => o.start === c.start && o.text.length > 2)) continue;
    if (t.length >= 3 && t[t.length - 1] === t[0]
        && cands.some((o) => o.text === t.slice(0, -1))) continue;
    if (BARE_TITLE_WORDS.has(t) && cands.some((o) => o.end === c.start)) continue;
    cleaned.push(c);
  }
  // 第二遍：称谓前缀碎片（小陈同 = 小陈 + 同）。前缀候选须在第一遍存活，
  // 否则会把 冯老师（前缀 冯老 本身是待清理碎片）误杀
  const final = [];
  for (const c of cleaned) {
    const t = c.text;
    if (t.length >= 3 && TITLE_HEAD_CHARS.has(t[t.length - 1])
        && cleaned.some((o) => o.text === t.slice(0, -1) && o.start === c.start)) continue;
    final.push(c);
  }
  return final.slice(0, MAX_CANDIDATES);
}

// ---------------------------------------------------------------- 门控与拆分

export function overlaps(a, b) {
  return !(a.end <= b.start || b.end <= a.start);
}

export function isNegated(text, name) {
  let total = 0;
  let negated = 0;
  let start = 0;
  for (;;) {
    const idx = text.indexOf(name, start);
    if (idx < 0) break;
    total += 1;
    const window = text.slice(Math.max(0, idx - 3), idx);
    if (NEGATION_CUES.some((cue) => window.includes(cue))) negated += 1;
    start = idx + 1;
  }
  return total > 0 && negated === total;
}

export function backwardNegated(text, cands) {
  const ids = new Set();
  for (const cue of ABSENCE_CUES) {
    let start = 0;
    for (;;) {
      const q = text.indexOf(cue, start);
      if (q < 0) break;
      const near = cands.filter((c) => q - c.end >= 0 && q - c.end <= ABSENCE_MAX_GAP);
      if (near.length) {
        let nearest = near[0];
        for (const c of near) if (q - c.end < q - nearest.end) nearest = c;
        for (const c of near) if (overlaps(c, nearest)) ids.add(c.id);
      }
      start = q + 1;
    }
  }
  return ids;
}

export function titleSplit(cands, winner) {
  const t = winner.text;
  for (const suf of SPLIT_SUFFIXES) {
    if (t.endsWith(suf) && t.length > suf.length) {
      const splitPos = winner.end - suf.length;
      const repl = cands.find((c) => c.end === splitPos && overlaps(c, winner)) || null;
      return [repl, suf];
    }
  }
  return [null, null];
}

export function followingTitle(text, cand) {
  for (const suf of SPLIT_SUFFIXES) {
    if (text.slice(cand.end, cand.end + suf.length) === suf) return suf;
  }
  return null;
}

export function identitySupported(text, name, identity) {
  if (!identity) return true;
  const idx = text.indexOf(name);
  if (idx < 0) return false;
  const window = text.slice(Math.max(0, idx - 4), idx + name.length + 4);
  return window.includes(identity);
}

// ---------------------------------------------------------------- 后端适配（对应 LayaBackend.ask）

/** noul→A/B 中性键 choice 改写（laya issue #156 workaround），P(A) 映射回 noul 概率。 */
export async function askLaya(agent, state, questions) {
  const transformed = {};
  const noulIds = new Set();
  for (const [qid, q] of Object.entries(questions)) {
    if (q.type === "noul") {
      noulIds.add(qid);
      transformed[qid] = {
        type: "choice",
        instructions: q.instructions + "？",
        criteria: {
          A: "是，" + q.instructions,
          B: "否，" + q.instructions + " 不成立",
        },
      };
    } else {
      transformed[qid] = q;
    }
  }
  const result = await agent.predict(state, transformed);
  const out = {};
  for (const [qid, ans] of Object.entries((result && result.answers) || {})) {
    if (noulIds.has(qid)) {
      const probs = (ans && ans.probabilities) || {};
      out[qid] = { noul: Number(probs.A || 0) };
    } else {
      out[qid] = ans && typeof ans === "object" ? ans : {};
    }
  }
  return out;
}

// ---------------------------------------------------------------- 组装（对应 _extract_forced cockpit 路径）

export async function extractForced(agent, userInput, cands) {
  PERF.nQ1 = 0;
  PERF.nQ2 = 0;
  let anchor = null;
  let fallbackWinners = null;
  try {
    if (cands.length === 1) {
      const c = cands[0];
      const ans = await askLaya(agent, userInput, {
        pick_is: {
          type: "noul",
          instructions: `『${c.text}』是这句话中提到的某个人物的名字或称呼`,
        },
      });
      const p = (ans.pick_is || {}).noul || 0;
      if (p >= IS_PERSON_THRESHOLD) anchor = [c, p];
    } else {
      const criteria = {};
      for (const c of cands) criteria[c.text] = "";
      criteria[FORCED_NO_PERSON] = "";
      const ans = await askLaya(agent, userInput, {
        pick: { type: "choice", instructions: FORCED_INSTRUCTIONS, criteria },
      });
      const pick = ans.pick || {};
      const win = pick.choice;
      const probs = pick.probabilities || {};
      const pWin = probs[win] || 0;
      const winner = cands.find((c) => c.text === win) || null;
      if (winner !== null && win !== FORCED_NO_PERSON && pWin >= FORCED_MIN_P) {
        anchor = [winner, pWin];
      } else if ((probs[FORCED_NO_PERSON] || 0) < NO_PERSON_CONFIDENT) {
        // 强制选择失败且『没有人物』置信度低（含糊）→ 普通 noul 扫描兜底
        const qs = {};
        for (const c of cands) {
          qs[`n${c.id}`] = {
            type: "noul",
            instructions: `『${c.text}』是这句话中提到的某个人物的名字或称呼`,
          };
        }
        const ansN = await askLaya(agent, userInput, qs);
        let passed = cands.map((c) => [c, (ansN[`n${c.id}`] || {}).noul || 0]);
        passed = passed.filter(([, p]) => p >= IS_PERSON_THRESHOLD);
        passed.sort((a, b) => b[1] - a[1]);
        fallbackWinners = [];
        for (const item of passed) {
          if (!fallbackWinners.some((k) => overlaps(item[0], k[0]))) fallbackWinners.push(item);
        }
      }
    }
  } catch (e) {
    console.warn("Jev 锚点轮失败:", e);
  }
  PERF.nQ1 = 1;
  // 注意：Python `if fallback_winners:` 对空列表为假——JS 空数组为真，须显式判 length
  if (anchor === null && (!fallbackWinners || fallbackWinners.length === 0)) {
    return { persons: [] };
  }

  let winners;
  if (fallbackWinners && fallbackWinners.length) {
    winners = fallbackWinners.slice(0, FORCED_MAX_PERSONS);
  } else {
    winners = [anchor];
    // 第 2 轮：增量 noul 扫描剩余候选（带锚点上下文）
    const aCand = anchor[0];
    const rest = cands.filter((c) => c.text !== aCand.text && !overlaps(c, aCand));
    if (rest.length) {
      try {
        const qs = {};
        for (const c of rest) {
          qs[`s${c.id}`] = {
            type: "noul",
            instructions: `除了已提到的『${aCand.text}』，` +
              `『${c.text}』也是这句话中提到的某个人物的名字或称呼`,
          };
        }
        const ans2 = await askLaya(agent, userInput, qs);
        let passed = rest.map((c) => [c, (ans2[`s${c.id}`] || {}).noul || 0]);
        passed = passed.filter(([, p]) => p >= IS_PERSON_THRESHOLD);
        passed.sort((a, b) => b[1] - a[1]);
        for (const item of passed) {
          if (!winners.some((k) => overlaps(item[0], k[0]))) winners.push(item);
        }
      } catch (e) {
        console.warn("Jev 增量扫描失败:", e);
      }
      PERF.nQ2 = rest.length;
    }
    winners = winners.slice(0, FORCED_MAX_PERSONS);
  }

  // v4 称谓后缀拆分：陈同学 → 小陈 + 身份证据 同学（仅当原文存在止于称谓起点的重叠候选）
  const splitWinners = [];
  for (const [c, p] of winners) {
    let [repl, suf] = titleSplit(cands, c);
    if (repl === null) suf = followingTitle(userInput, c);
    splitWinners.push([repl || c, p, suf]);
  }

  // v4 位置文本证据门（宽松版）：原文无任何位置词时位置回答必是幻觉
  const locEvidence = LOCATION_OPTIONS.some(
    (o) => o !== NO_LOCATION && userInput.includes(o));

  const persons = [];
  for (const [c, , suf] of splitWinners) {
    let attr = {};
    try {
      const locCriteria = {};
      for (const o of LOCATION_OPTIONS) locCriteria[o] = "";
      const idCriteria = {};
      for (const o of IDENTITY_OPTIONS) idCriteria[o] = "";
      attr = await askLaya(agent, userInput, {
        loc: {
          type: "choice",
          instructions: `『${c.text}』` + LOCATION_QUESTION,
          criteria: locCriteria,
        },
        id: {
          type: "choice",
          instructions: `『${c.text}』` + IDENTITY_QUESTION,
          criteria: idCriteria,
        },
      });
    } catch (e) {
      console.warn("Jev 属性轮失败:", e);
    }
    PERF.nQ2 += 2;
    let ident = ((attr.id || {}).choice) || "";
    ident = ident === NO_IDENTITY ? "" : ident;
    if (!identitySupported(userInput, c.text, ident)) ident = "";
    // 拆分携带的身份证据优先；身份是名字自身后缀时按 gold 约定抑制
    if (suf && IDENTITY_OPTIONS.includes(suf)) ident = suf;
    if (ident && c.text.endsWith(ident)) ident = "";
    let loc = ((attr.loc || {}).choice) || "";
    loc = loc === NO_LOCATION ? "" : loc;
    if (loc && !locEvidence) loc = "";
    persons.push({ name: c.text, identity: ident, location: loc });
  }
  persons.sort((a, b) => {
    const ca = cands.find((c) => c.text === a.name);
    const cb = cands.find((c) => c.text === b.name);
    return (ca ? ca.start : 0) - (cb ? cb.start : 0);
  });
  return { persons };
}

/** 完整管线入口（对应 extract_entities 的 forced 模式 = laya 最优配置）。 */
export async function extractEntities(agent, userInput) {
  let cands = generateCandidates(userInput);
  cands = cands.filter((c) => !isNegated(userInput, c.text));
  const backNeg = backwardNegated(userInput, cands);
  cands = cands.filter((c) => !backNeg.has(c.id));
  PERF.nCandidates = cands.length;
  if (!cands.length) {
    PERF.nQ1 = 0;
    PERF.nQ2 = 0;
    return { persons: [] };
  }
  return extractForced(agent, userInput, cands);
}
