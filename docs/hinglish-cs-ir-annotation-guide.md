# Human-Check Guide: Hinglish Queries (~120 items, 2–3 hours)

**Who:** you (a native Hinglish speaker) · **When:** sheet arrives ~Mon Oct 19; please return by **Wed Oct 21, 2026**
**File you get:** `hinglish-cs-ir/data/annotation/check_sheet.csv`
**File you return:** save it as `check_sheet_done.csv` in the same folder

## Why this matters
An LLM wrote the Hinglish queries. Reviewers will ask whether they look like what real people type. Your check gives us two things: (1) the percentage of queries that are natural and correct, which we report in the paper; (2) a small human-verified test set that we evaluate separately.

## What you see in each row
| Column | What it is |
|---|---|
| `item_id`, `dataset`, `qid` | IDs. Do not edit. |
| `english` | The original English search query. |
| `hinglish` | The LLM's romanized Hinglish version. |
| `natural` | **You fill:** Y or N |
| `meaning_preserved` | **You fill:** Y or N |
| `rewrite` | **Optional:** your own better version |
| `notes` | Optional comments |

## How to label

**`natural` = Y** if a typical urban Indian user could plausibly type this in a search box or a WhatsApp message. Small style differences are fine.
**`natural` = N** if any of these apply: it sounds like a word-for-word machine translation, it uses formal or bookish Hindi nobody types (e.g. *vishleshan*, *upchaar* in place of *analysis*, *treatment*), the mixing is odd (Hindi words placed where nobody would switch), or it contains Devanagari script.

**`meaning_preserved` = Y** if someone searching with the Hinglish query wants the **same information** as with the English query.
**`meaning_preserved` = N** if anything important is lost, added, or changed: a medical term, a number, a negation, or the subject.

**`rewrite`** (only when either label is N, or when you see a clearly more natural version). Rules:
- Use Roman script only.
- Keep technical, medical, and scientific terms and names in English, as people really do.
- Spell the way **you** naturally would. Don't try to be "correct"; there is no correct spelling.
- Keep the full meaning of the English query.

## Examples
| english | hinglish | natural | meaning | rewrite | why |
|---|---|---|---|---|---|
| Does vitamin D deficiency increase the risk of depression? | kya vitamin D deficiency se depression ka risk badhta hai? | Y | Y | | natural and complete |
| Does vitamin D deficiency increase the risk of depression? | kya vitamin D ki kami se avsaad ka khatra badhta hai? | N | Y | kya vitamin D deficiency se depression ka risk badhta hai? | bookish Hindi (*avsaad*, *khatra*) for common English terms |
| Statins reduce cholesterol in elderly patients | statins elderly patients mein cholesterol kam karte hain | Y | Y | | fine |
| Statins reduce cholesterol in elderly patients | statins cholesterol kam karte hain | Y | N | statins buzurg patients mein cholesterol kam karte hain | "elderly" was dropped, and the meaning changed |
| Is homework beneficial for students? | homework students ke liye beneficial hai kya | Y | Y | | different word order is fine |

## Tips
- Read the English first, then the Hinglish.
- Don't over-think spelling. *nahi* vs *nhi* doesn't matter here; we add spelling noise separately.
- ArguAna rows are long (whole arguments). For these, judge the overall naturalness and whether the main point is kept. A rewrite is optional and can be skipped for long rows.
- Spend about 1 minute per row; take a break every 40 rows.
- Use only Y or N (y/yes/n/no also work). Leave `rewrite` empty if you don't have one.
- If you open the file in Excel, save it as **CSV UTF-8** so the file stays readable.

## Optional: second annotator
If a friend labels the **same** 120 rows independently (in a copy saved as `check_sheet_done_2.csv`), we can report inter-annotator agreement (Cohen's κ). That strengthens the paper but is not required.
