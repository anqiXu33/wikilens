# Alignment annotation guidelines

These labels measure how well the aligner pairs paragraphs ("chunks") across two language editions of one Wikipedia article. They are the gold data behind `python -m wikilens evaluate` (item 2 of `docs/review.md`).

The current label set is **en-zh, Huawei and Social Credit System, 50 sampled Chinese chunks per article**. The tools themselves work for any language pair.

## The task

`python -m wikilens annotate RUN_ID` writes an HTML page. Each card shows one **query** chunk from the source language (by default the non-English side) and its **3 most similar chunks** from the other language. Similarity is LaBSE cosine; it is shown for reference, but do not let it decide your label. The page shows no thresholds and does not say which pairs the aligner currently matches, so the labels are independent of the system being tested.

For each candidate, choose one label:

| Label | In the export | Use it when |
|---|---|---|
| **对应** | `match` | The two chunks express **essentially the same content**: same facts, same claims, and neither has substantial material the other lacks. Different wording, sentence order, translation quality or small extra details (a date, a name in brackets, a clause) are fine. |
| **部分对应** | `partial` | The chunks **share at least one specific claim or fact**, but one or both also contain substantial content the other lacks. Typical cases: one chunk is a subset of the other; the two editions split or merge paragraphs differently; the chunks overlap on one event but go on to different things. |
| **不对应** | `none` | The chunks share **no specific claim**. Being about the same topic, section or period is not enough. |

For each card there is also a checkbox:

- **另一种语言里完全没有** (`absent`): nothing anywhere in the other article expresses any of this chunk's content. Check this only after searching the full text of the other article; the page has an expandable "查看全文" panel for this (use Ctrl/⌘-F with names, numbers and years). Checking it marks all three candidates `不对应`. **If the content is in the other article but not among the three candidates**, label the three candidates normally and leave the box unchecked, with a note such as `match is en_0042`.

A card is complete when all three candidates are labelled or the box is checked.

## Decision rules

1. **Compare propositions, not topics.** Ask: "Is there a specific statement (who did what, when, how much) that both chunks make?" If yes, the label is at least `partial`.
2. **`match` vs `partial`: is anything substantial left over?** If removing the shared content would leave one chunk with a sentence or more of its own content, the label is `partial`. A clause or a parenthetical does not count.
3. **Direction does not matter.** Judge the pair symmetrically. A short chunk that is fully contained in a long one is `partial` (the long one has leftover content), unless the leftover is trivial.
4. **Label content, not stance.** The same fact reported with different framing, tone or sourcing still counts as shared, e.g. "the system is used to punish dissent" against "the system improves trustworthiness in society". Each describes how the system is used, and if both refer to the same mechanism or event they are at least `partial`. Framing differences are what the bias scores measure, not alignment. Mark a chunk `absent` only when the **facts** themselves are missing.
5. **Numbers and dates that disagree** (different figures for the same quantity) still count as the same claim, so use `match`/`partial`, and say so in the note. This often matters for omission and bias analysis.
6. **Lists, infobox-like text and references** (e.g. a list of subsidiaries, a list of laws): use `partial` if the lists overlap substantially, otherwise `none`.
7. **When in doubt between two labels, pick the lower one** (`partial` over `match`, `none` over `partial`) and write a short note. Notes are kept in the export.

## Examples (Switzerland, ZH vs EN)

These come from `app/cached_articles/zh_瑞士.json` and `en_Switzerland.json` and only illustrate the rules. They are not part of the label set.

**对应 (match)**: same two facts, nothing left over.

> ZH 历史 > 早期歷史: 瑞士地區最早的人類活動可追溯至15萬年前，最早的農耕聚落遺跡於盖赫林根發現，該聚落存在於西元前5,300年。
>
> EN History > Beginnings: The oldest traces of hominid existence in Switzerland date to about 150,000 years ago. The oldest known farming settlements in Switzerland, which were found at Gächlingen, date to around 5300 BC.

**对应 (match)**: wording and detail differ, but every claim is shared (trade blockade by both sides, economic cooperation with Germany depending on invasion risk, the 1942 Vichy rail link, 300,000 refugees, the Red Cross in Geneva, later controversy). The English has a little more detail ("purchasing gold", "Liechtenstein"), but none of it is a separate claim.

> ZH 历史 > 現代歷史: 同盟國及轴心国雙方皆於戰時封鎖與瑞士間的貿易。受德國和義大利侵略的可能性及與其他國家貿易的影響，瑞士對德國的經濟合作及贷款的還款期限也隨之變動。瑞士經德國傀儡政權維希法國的重要鐵路路線於1942年中斷。在戰爭期間，瑞士收容超過30萬名難民，總部設於日內瓦的红十字国际委员会也於衝突中发挥了重要的作用。嚴格的移民及難民收容政策……使瑞士当时的中立政策于20世纪晚期遭受質疑。
>
> EN History > Modern history: Switzerland's trade was blockaded by both the Allies and the Axis. Economic cooperation with Nazi Germany … varied according to the perceived likelihood of invasion … a crucial rail link through Vichy France was severed in 1942 … Switzerland interned over 300,000 refugees aided by the International Red Cross, based in Geneva. Strict immigration and asylum policies and the financial relationships with Nazi Germany became controversial by the end of the 20th century.

**部分对应 (partial)**: the same ZH chunk against a different EN paragraph that shares only the refugee figure. The ZH chunk has a lot left over (blockade, rail link, Red Cross), and the EN chunk adds "refusing tens of thousands more, including Jews".

> EN History > Modern history: Switzerland's attitude towards refugees was complicated and controversial; over the course of the war, it admitted as many as 300,000 refugees while refusing tens of thousands more, including Jews persecuted by the Nazis.

**部分对应 (partial)**: paragraph boundaries differ between editions. ZH packs three events into one chunk. EN has Jura (1979) and the new constitution (1999) in one paragraph, and the Council of Europe (1963) at the end of a paragraph about a nuclear bomb.

> ZH 历史 > 現代歷史: 瑞士於1963年加入欧洲委员会。1979年汝拉州自伯恩州分出，成为新的一州。瑞士於1999年4月18日對聯邦憲法進行了全面的修改。
>
> EN History > Modern history: In 1979 areas from the canton of Bern attained independence from the Bernese, forming the new canton of Jura. On 18 April 1999, the Swiss population and the cantons voted in favour of a completely revised federal constitution. → **partial** (2 of 3 ZH facts; nothing extra on the EN side, but the ZH side has leftover content)
>
> EN History > Modern history: During the Cold War, Swiss authorities considered the construction of a Swiss nuclear bomb. … Plans for building nuclear weapons were dropped by 1988. Switzerland joined the Council of Europe in 1963. → **partial** (one shared fact, much leftover on both sides)

**部分对应 (partial), not 不对应**: same topic in different sections. ZH 政治 describes today's parliament (46 + 200 seats, 4-year terms, referendums). EN History > Federal state describes the 1848 constitution creating that bicameral parliament (two representatives per canton in the upper house, a lower house elected nationally). The shared claim is the structure of the two houses, so this is `partial` even though one chunk is about history and the other about current politics.

**不对应 (none)**: same section and era, but no shared claim. ZH 現代歷史 (1990s banking scandal: dormant accounts of Holocaust victims and the 1998 settlement of $1.25 billion) against EN History > Modern history (Cold War nuclear-bomb plans). Both are "modern Swiss history", but they share no fact, so the label is `none`. The dormant-accounts story appears nowhere in the cached English article, so for this ZH chunk all three candidates are `none` and **另一种语言里完全没有** is checked.

**另一种语言里完全没有 (absent)**: ZH 名稱來源 > 瑞士 explains the origin of the Chinese name 瑞士 (Gützlaff's 瑞国 of 1838, Lin Zexu's 四洲志, Xu Jiyu's 瀛寰志略, the southern-dialect transliteration of "Schwyz"). The English Etymology section covers "Switzer", "Schwyz" and "Helvetia", but not the Chinese name. Search the English full text for "Lin Zexu", "Gützlaff", "1838" and "Chinese" before checking the box.

## Workflow

See the PR description or `python -m wikilens annotate -h`. In short:

1. `python -m wikilens annotate RUN_ID --sample 50` → open `eval/annotation_pages/<run_id>.html`.
2. Label. Progress autosaves in the browser. **Export JSON** regularly anyway, because browser storage can be cleared.
3. Save the export as `eval/labels/<run_id>.json` (the page suggests this file name). To resume elsewhere, click **导入 JSON**.
4. `python -m wikilens evaluate`.

`--sample N --seed S` (default seed 0) picks the same N chunks every time for the same run. The seed and N are recorded in the page and in the exported labels (`"sample": {"n": 50, "seed": 0}`).

## How labels are scored

`python -m wikilens evaluate` re-runs the aligner's matching rule over a grid of `(chunk_thr, path_thr)` and compares the matched pairs with the labels:

- **TP**: a matched pair labelled `match` or `partial`. `partial` counts as positive by default; `--partial negative|ignore` changes this.
- **FP**: a matched pair labelled `none`, or any match for a chunk marked `absent`.
- **FN**: a `match`/`partial` pair the aligner did not match.
- **Pairs outside the top 3** that the aligner matched (`--unlabelled`): by default (`auto`) they count as FP when the chunk's third-ranked candidate was labelled `none`. In that case the ranking has already run out of matches, so lower-ranked chunks are taken as non-matches. Otherwise they are reported as "uncounted". Recall is relative to the top-3 pool, which is one more reason to write `match is …` notes when the real match is ranked lower.

The best grid point is chosen on the same labels, so its F1 is optimistic. With two or more articles, `evaluate` also reports a cross-article score: thresholds chosen on one article and scored on the other.
