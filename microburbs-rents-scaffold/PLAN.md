# Saturday 7 November — run sheet

Event 10:00–16:00, UTS Business School, 14 Ultimo Rd, Ultimo, Level 2 (or
remote). Key and files arrive **11:00**. Uploads close **15:00**, hard.

## Before you sleep tonight

- [ ] `pip install -r requirements.txt` on the machine you'll actually use
- [ ] `python make_synthetic.py && python pipeline.py --team X --trials 4` — must run clean
- [ ] Public GitHub repo created, this code pushed, `.gitignore` verified
- [ ] Phone camera tested for the pitch video; know where you'll record
- [ ] Team name decided (CSV name = lowercase, spaces → hyphens, letters/digits/hyphens only)

## The run sheet

| Time | Do | Done when |
|---|---|---|
| 11:00 | Files into `data/`. Run `pipeline.py --baseline-only`. | A valid CSV exists |
| 11:10 | Read the `[schema]` line. Fix `COLUMN_HINTS` if the leases columns differ. | Schema matches |
| 11:30 | **Upload the baseline CSV.** | You cannot score zero |
| 11:30–13:00 | Features. Building-level aggregates from past leases, time-since-last-advert, block rent levels. | Validation improves |
| 13:00–13:45 | `--trials 40`. ~3s a trial, so budget 2–5 min and spend the rest on features. | Best params logged |
| 13:45 | Final run. Re-upload CSV. | `out/rents_<team>.csv` final |
| 14:00 | **HARD STOP on modelling.** | — |
| 14:00–14:30 | Fill `FINDINGS` in `make_report.py`, run it, fill README TODOs, push. | report.html + README done |
| 14:30–14:50 | Record pitch (≤45s) and demo (≤3min). Each under 95 MB. | Both in repo |
| 14:50 | Final push. Check no `.env`, no supplied CSVs, no API output in the repo. | Clean `git status` |
| 15:00 | Closed. | — |

## Rules that disqualify or cost you

- Nothing dated after **31 July 2026**, from the API or anywhere
- All **65,777** gnaf_ids, no extras, **no blank cells** — blank scores worse than wrong
- Key, supplied files, API output and hidden judging data must stay **out of the repo**
- Videos under 95 MB each; pitch ≤45s, demo ≤3min
- CSV: UTF-8, one header, one row per required ID

## The API trap

65,777 per-address calls at ~10 req/s is about 110 minutes. **Do not do it.**
Build from the supplied past-leases file. If you want API enrichment, run it in
the background on a subset while the model trains, and only merge it if it
lands before 13:30.

---

# If you get a second team member

Two is the minimum, so a second person fixes the form. It does not change the
task and it should not change the plan. Give them work that is **off the
critical path**, so a teammate who underdelivers costs you nothing.

## Split A — they are non-technical or unknown to you

They own everything that is not the model:

- `report.html` prose — the FINDINGS blocks, written from numbers you call out
- README TODOs
- Both videos: script, record, compress, check the 95 MB limit
- The submission checklist at 14:50

This is genuinely 45–60 minutes of work that would otherwise come out of your
modelling time. It is the highest-value split available to you.

## Split B — they can actually code

They take the API enrichment track, entirely separately:

- Write an async client against `https://api-ht.microburbs.com.au`, 10 req/s,
  honouring 429s, on the ~16,000 flats that have never been advertised
- Output **one parquet/CSV keyed on gnaf_id**, merged by you only if it arrives
  before 13:30

You keep the main pipeline. One owner per file. The merge is one join, and if
their file is late or wrong you drop it and lose nothing.

## Rules for either split

- **You own `pipeline.py`.** Nobody else commits to it. Merge conflicts at
  14:30 are how teams miss the deadline.
- Everyone on the team must **already have been offered a place** — you cannot
  recruit someone who did not apply.
- Agree the 14:00 hard stop out loud before you start.
- They should not be a dependency for anything before 14:00.

## Tonight, in order

1. Reply to the Microburbs email and ask to be paired with another solo entrant
2. Check for the Slack invite — post your background and the task you're taking
3. Either way, run the pipeline test. The build does not depend on the answer.
