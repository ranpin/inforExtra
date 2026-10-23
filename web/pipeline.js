/**
 * jevCabin 提取管线 JS 移植（浏览器 live 推理用）。
 *
 * 与 jev_extract.py（v5）逐函数对应：常量、候选生成、否定门控、介绍线索锚点、
 * 锚点强制选择、增量扫描、称谓拆分、身份证据门、位置文本证据绑定全部按行
 * 移植；唯一结构差异是后端适配层 askLaya（对应 Python LayaBackend.ask 的
 * noul→A/B 中性键改写）。agent 由外部注入（@johnhenry/laya 的 load() 产物），
 * 本文件零依赖、可在 Node 用 mock agent 单测确定性部分。
 *
 * 对拍验证：51 条真实种子集 persons 与 Python 管线输出逐条一致
 * （gold = web/public/parity51.json，由 v5 管线产物生成）。
 */

// ---------------------------------------------------------------- 常量（与 jev_extract.py 逐字一致）

export const IS_PERSON_THRESHOLD = 0.5;
export const FORCED_MIN_P = 0.3;
export const NO_PERSON_CONFIDENT = 0.7;
export const FORCED_MAX_PERSONS = 5;
export const MAX_CANDIDATES = 60;

/** 每次调用的统计（镜像 Python PERF；live.js 读取供页面展示）。 */
export const PERF = { nCandidates: 0, nQ1: 0, nQ2: 0, winners: [] };

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

// v5 规则5 介绍线索锚点（与 Python INTRO_CUES 逐字一致）：『我是/我叫/这是…』
// 紧邻候选时直接作为人物锚点、跳过模型门控（laya 对自我介绍句中小X类名字
// noul 系统性低分：『我是小张，坐在副驾』0.4407 < 0.5 漏检）。
export const INTRO_CUES = [
  "我是", "我叫", "叫我", "喊我", "称呼我", "我的名字是",
  "这是", "那是", "他叫", "她叫", "他是", "她是",
];

// v5 规则6 位置文本证据绑定（与 Python LOC_SEAT_GROUP/绑定规则逐字一致）：
// 原文中与人名确定性绑定的座位词优先于模型回答（laya 左右系统性混淆：
// 『后排左边是我女儿小雨』答 后排右边 0.307 > 后排左边 0.189）。
export const LOC_SEAT_GROUP = {
  "主驾": "driver", "主驾驶": "driver", "驾驶位": "driver",
  "副驾": "passenger", "副驾驶": "passenger",
  "后排左边": "rear-left", "左后": "rear-left", "左后座": "rear-left",
  "后排左": "rear-left",
  "后排右边": "rear-right", "右后": "rear-right", "右后座": "rear-right",
  "后排右": "rear-right",
};
export const LOC_TEXT_WORDS = [
  ...Object.keys(LOC_SEAT_GROUP).sort((a, b) => b.length - a.length),
  "前排", "后排", "左边", "右边",
];
export const LOC_BIND_MAX_GAP = 8;
export const LOC_BIND_STOP = "。！？；!?;";
export const LOC_FWD_CONNECT = ["是", "坐", "在"];
export const LOC_BACK_CONNECT = ["在", "坐"];

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
  // v5 公司名包含清理：候选严格包含于更长公司词出现时剔除（阿里 ⊂ 阿里巴巴）；
  // 小米/小鹏 等两用词无更长公司词包含，不受影响
  const compSpans = companySpans(text);
  const kept = compSpans.length
    ? final.filter((c) => !compSpans.some(([ws, we]) =>
        ws <= c.start && c.end <= we && we - ws > c.end - c.start))
    : final;
  // v5 规则7 两用词公司义清理：公司词候选后紧跟「的+另一候选」（小鹏的周平）
  // → 公司义优先，剔除人物候选；我是小鹏 / 小鹏和小王 不匹配，人物义保留
  const companySet = new Set(COMPANY_WORDS);
  const out = kept.filter((c) => !(companySet.has(c.text)
    && text.slice(c.end, c.end + 1) === "的"
    && kept.some((o) => o !== c && o.start === c.end + 1)));
  return out.slice(0, MAX_CANDIDATES);
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

export function cueDedup(hits) {
  if (!hits.length) return null;
  if (new Set(hits.map((c) => c.start)).size !== 1) return null;
  let best = hits[0];
  for (const c of hits) if (c.end > best.end) best = c;
  return best;
}

export function introCued(text, cands) {
  const hits = cands.filter((c) => INTRO_CUES.some(
    (cue) => text.slice(Math.max(0, c.start - cue.length), c.start) === cue));
  return cueDedup(hits);
}

export function brandCued(text, cands) {
  const hits = [];
  for (const c of cands) {
    if (text.slice(Math.max(0, c.start - 1), c.start) !== "的") continue;
    for (const w of COMPANY_WORDS_BY_LEN) {
      const s = c.start - 1 - w.length;
      if (s >= 0 && text.slice(s, c.start - 1) === w) {
        hits.push(c);
        break;
      }
    }
  }
  return cueDedup(hits);
}

export function locOccurrences(text) {
  const occ = [];
  let i = 0;
  while (i < text.length) {
    let matched = false;
    for (const w of LOC_TEXT_WORDS) {
      if (text.startsWith(w, i)) {
        occ.push([i, i + w.length, w]);
        i += w.length;
        matched = true;
        break;
      }
    }
    if (!matched) i += 1;
  }
  return occ;
}

export function boundLocWord(text, cand, others) {
  const regionOk = (a, b) => {
    for (const ch of text.slice(a, b)) {
      if (LOC_BIND_STOP.includes(ch)) return false;
    }
    for (const o of others) {
      let p = text.indexOf(o.text);
      while (p >= 0) {
        if (p < b && p + o.text.length > a) return false;
        p = text.indexOf(o.text, p + 1);
      }
    }
    return true;
  };
  let best = null; // [gap, word]
  for (const [s, e, w] of locOccurrences(text)) {
    if (e <= cand.start) { // 前向绑定
      const gap = cand.start - e;
      if (gap <= LOC_BIND_MAX_GAP && regionOk(e, cand.start)
          && (gap === 0 || LOC_FWD_CONNECT.some(
            (k) => text.slice(e, cand.start).includes(k)))
          && (best === null || gap < best[0])) best = [gap, w];
    } else if (s >= cand.end) { // 后向绑定
      const gap = s - cand.end;
      if (gap <= LOC_BIND_MAX_GAP && regionOk(cand.end, s)
          && LOC_BACK_CONNECT.some((k) => text.slice(cand.end, s).includes(k))
          && (best === null || gap < best[0])) best = [gap, w];
    }
  }
  return best ? best[1] : null;
}

// v5 规则7 公司/品牌信息提取（与 Python COMPANY_WORDS/绑定规则逐字一致）：
// 确定性词表 + 就近绑定，零模型调用。防误绑三道闸：品牌出现与任何赢家人名
// 重叠不绑（『我是小鹏』）；窗口跨越其他赢家出现不绑（『日产的小王和小李』
// 不给小李绑日产）；gap>0 时窗口须含连接词（『小王和吉利』不绑）。
export const COMPANY_WORDS = [
  // 汽车整车（合资/自主/新势力）
  "日产", "吉利", "比亚迪", "特斯拉", "宝马", "奔驰", "奥迪", "大众",
  "丰田", "本田", "理想", "蔚来", "小鹏", "长城", "奇瑞", "沃尔沃",
  "马自达", "福特", "现代", "起亚", "零跑", "哪吒", "极氪", "问界",
  "智界", "享界", "领克", "坦克", "五菱", "红旗", "捷豹", "路虎",
  "保时捷", "雷克萨斯", "凯迪拉克", "别克", "雪佛兰", "斯巴鲁",
  "三菱", "标致", "雪铁龙", "斯柯达", "英菲尼迪", "宾利",
  "劳斯莱斯", "法拉利", "兰博基尼", "迈巴赫",
  // 汽车供应链/智驾
  "博世", "采埃孚", "宁德时代", "地平线", "黑芝麻", "亿咖通",
  "德赛西威", "均胜电子", "禾赛", "速腾聚创", "图达通", "Momenta",
  // 互联网/科技
  "阿里巴巴", "阿里", "腾讯", "百度", "华为", "小米", "字节跳动",
  "字节", "美团", "京东", "网易", "滴滴", "拼多多", "快手",
  "哔哩哔哩", "大疆", "商汤", "旷视", "科大讯飞", "联想", "海尔",
  "格力", "微软", "谷歌", "苹果", "亚马逊", "英伟达", "英特尔",
  "高通", "三星", "台积电", "大众点评",
];
// 长度降序 + 字典序（确定性迭代，与 Python sorted(key=(-len, w)) 一致）
export const COMPANY_WORDS_BY_LEN = [...new Set(COMPANY_WORDS)]
  .sort((a, b) => b.length - a.length || (a < b ? -1 : a > b ? 1 : 0));
export const COMPANY_BIND_MAX_GAP = 4;
export const COMPANY_CONNECT = ["的", "是", "在", "来自", "属于"];

export function companySpans(text) {
  const spans = [];
  for (const w of COMPANY_WORDS_BY_LEN) {
    let p = text.indexOf(w);
    while (p >= 0) {
      spans.push([p, p + w.length]);
      p = text.indexOf(w, p + 1);
    }
  }
  return spans;
}

export function companyFor(text, cand, winners) {
  const nameSpans = [];
  for (const w of winners) {
    let p = text.indexOf(w.text);
    while (p >= 0) {
      nameSpans.push([p, p + w.text.length]);
      p = text.indexOf(w.text, p + 1);
    }
  }
  const bindable = (a, b, fwd, gap) => {
    if (gap > COMPANY_BIND_MAX_GAP) return false;
    if ([...text.slice(a, b)].some((ch) => LOC_BIND_STOP.includes(ch))) return false;
    if (nameSpans.some(([ns, ne]) => ns < b && ne > a)) return false;
    if (!(fwd && gap === 0)
        && !COMPANY_CONNECT.some((k) => text.slice(a, b).includes(k))) return false;
    return true;
  };
  let best = null; // [gap, rank, word]；rank: 前向0 后向1
  for (const word of COMPANY_WORDS_BY_LEN) {
    let p = text.indexOf(word);
    while (p >= 0) {
      const e = p + word.length;
      if (!nameSpans.some(([ns, ne]) => p < ne && e > ns)) {
        if (e <= cand.start) {
          const gap = cand.start - e;
          if (bindable(e, cand.start, true, gap)
              && (best === null || gap < best[0]
                  || (gap === best[0] && 0 < best[1]))) best = [gap, 0, word];
        } else if (p >= cand.end) {
          const gap = p - cand.end;
          if (bindable(cand.end, p, false, gap)
              && (best === null || gap < best[0]
                  || (gap === best[0] && 1 < best[1]))) best = [gap, 1, word];
        }
      }
      p = text.indexOf(word, p + 1);
    }
  }
  return best ? best[2] : null;
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
  PERF.winners = [];
  let anchor = null;
  let fallbackWinners = null;
  const intro = introCued(userInput, cands);
  const brand = intro !== null ? null : brandCued(userInput, cands);
  try {
    if (intro !== null) {
      // v5 规则5：介绍线索紧邻 → 确定性锚点，跳过模型门控
      anchor = [intro, null];
    } else if (brand !== null) {
      // v5 规则7：『[公司词]的』紧邻引导 → 确定性锚点（『小鹏的周平』）
      anchor = [brand, null];
    } else if (cands.length === 1) {
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
  PERF.nQ1 = (intro !== null || brand !== null) ? 0 : 1;
  // 注意：Python `if fallback_winners:` 对空列表为假——JS 空数组为真，须显式判 length
  if (anchor === null && (!fallbackWinners || fallbackWinners.length === 0)) {
    return { persons: [] };
  }

  let winners;
  let winnerPaths;
  if (fallbackWinners && fallbackWinners.length) {
    winners = fallbackWinners.slice(0, FORCED_MAX_PERSONS);
    winnerPaths = winners.map(() => "fallback");
  } else {
    winners = [anchor];
    winnerPaths = [intro !== null ? "intro_cue"
      : (brand !== null ? "brand_cue"
        : (cands.length === 1 ? "single_noul" : "forced"))];
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
          if (!winners.some((k) => overlaps(item[0], k[0]))) {
            winners.push(item);
            winnerPaths.push("scan");
          }
        }
      } catch (e) {
        console.warn("Jev 增量扫描失败:", e);
      }
      PERF.nQ2 = rest.length;
    }
    winners = winners.slice(0, FORCED_MAX_PERSONS);
    winnerPaths = winnerPaths.slice(0, FORCED_MAX_PERSONS);
  }

  // v4 称谓后缀拆分：陈同学 → 小陈 + 身份证据 同学（仅当原文存在止于称谓起点的重叠候选）
  const splitWinners = [];
  for (const [c, p] of winners) {
    let [repl, suf] = titleSplit(cands, c);
    if (repl === null) suf = followingTitle(userInput, c);
    splitWinners.push([repl || c, p, suf]);
  }
  PERF.winners = splitWinners.map(([c, p], i) => [c.text, p, winnerPaths[i]]);

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
    if (loc) {
      // v5 规则6：原文确定性绑定的座位词与模型回答冲突 → 覆盖（含区域级粗答）；
      // 同座异名（右后→右后座）保留模型答案
      const others = splitWinners.filter(([c2]) => c2 !== c).map(([c2]) => c2);
      const bound = boundLocWord(userInput, c, others);
      const gBound = bound ? LOC_SEAT_GROUP[bound] : undefined;
      if (gBound && LOC_SEAT_GROUP[loc] !== gBound) loc = bound;
    }
    // v5 规则7：公司/品牌就近绑定（确定性词表，零模型调用）
    const company = companyFor(userInput, c, splitWinners.map(([c2]) => c2)) || "";
    persons.push({ name: c.text, identity: ident, location: loc, company });
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
    PERF.winners = [];
    return { persons: [] };
  }
  return extractForced(agent, userInput, cands);
}
