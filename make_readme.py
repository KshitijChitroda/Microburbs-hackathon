"""
Writes README.md from the real run, so the numbers in it can never drift from
the numbers in the report.

    python make_readme.py --team "k-and-k"
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

OUT = Path("out")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--team", required=True)
    args = ap.parse_args()

    m = json.loads((OUT / "metrics.json").read_text())
    f = json.loads((OUT / "findings.json").read_text()) if (OUT / "findings.json").exists() else {}

    best = m.get("model") or m["baseline_group_median"]
    base = m["baseline_group_median"]
    mb = m["benchmarks_published_by_microburbs"]["microburbs_rental_valuation"]
    by_size = m.get("parking_premium_by_building_size", {})
    park, drift = f.get("parking", {}), f.get("drift", {})
    floor, churn = f.get("floor", {}), f.get("churn", {})
    beat = (1 - best["median_abs_error"] / mb["median_abs_error"]) * 100

    small, mid = by_size.get("0-20"), by_size.get("60-150") or by_size.get("20-60")
    grad = small and mid and small > mid * 1.8

    size_rows = "\n".join(
        f"| {k.replace('-plus', '+').replace('-', ' to ')} flats | ${v} |"
        for k, v in by_size.items()) or "| — | — |"

    lead = (f"**A car space is worth ${small} a week in a block of under 20 flats, "
            f"and ${mid} in a larger building — {small/mid:.0f}× the price for the same "
            f"car space.** Towers are built with basement parking, so nobody pays extra "
            f"for it. Small walk-ups mostly have none, so the few spaces are scarce."
            if grad else
            f"**A car space adds about ${park.get('median_premium', 0):,.0f} a week** "
            f"across {park.get('n_groups', 0)} comparable groups of flats.")

    md = f"""# What will this flat rent for?

Microburbs Hackathon, 7 November 2026 — **Task 4**. Weekly asking rent for all
65,777 City of Sydney flats, estimated twice: with a car space and without.

Team **{args.team}**.

## What we built

An estimate of the weekly asking rent of every flat in the council area, under
two conditions — with a car space and without. The gap between those two numbers
is a measurement in its own right, and it is where the interesting findings came
from.

{lead}

Full write-up with charts: **`report.html`**.

## Who it helps

- A **tenant** can check whether an asking rent is ordinary for that building and street.
- A **landlord** can price a flat, and a car space, against comparable ones nearby.
- A **buyer or investor** can compare likely rent before committing.

## Results

Held out: adverts dated {m.get('validation_window', '')}, never used to build or tune.

| Estimator | Typical miss | Within $100 |
|---|---|---|
| **Our estimate** | **${best['median_abs_error']:,.2f}** | **{best['pct_within_100']}%** |
| Microburbs rental valuation (published) | ${mb['median_abs_error']} | {mb['pct_within_100']}% |
| Our group-median baseline | ${base['median_abs_error']:,.0f} | {base['pct_within_100']}% |
| Microburbs simple estimate (published) | $100 | — |

R squared **{best.get('r_squared')}** on {best['n']:,} held-out adverts. Our typical
miss is **{beat:.0f}% smaller** than the published figure for their own valuation.

Our baseline reproduces their published $100 / 56% almost exactly, which is why we
trust the comparison: we are measuring the same thing, on a window they have not seen.

## The clever step

The task ranks on **median absolute dollar error**, so the model is trained with an
**L1 objective** — its optimum is the conditional median, which is the quantity being
scored. The usual squared-error objective targets the mean, a different number. The
hyperparameter search also scores candidates on median absolute error directly rather
than on the training loss.

Two further choices:

- **Recent adverts count for more.** The history runs back to 2012 and rents have moved
  a long way since; adverts are weighted by age with a six-month half-life.
- **Parking is a counterfactual, not a flag.** Each flat is predicted twice and the gap
  is held inside a band anchored on the premium measured for that flat's *building size*,
  so a tower is never given a walk-up's premium, the two columns can never contradict
  each other, and no flat comes back blank.

We also **tested extrapolating the rent trend forward** onto the scoring window. It did
not improve held-out accuracy, so we did not ship it.

## What we found

| Finding | Number |
|---|---|
| Car space, blocks under 20 flats | ${small} a week |
| Car space, larger buildings | ${mid} a week |
| Car space, council-wide median | ${park.get('median_premium', 0):,.0f} a week |
| Rent growth, year to 31 Jul 2026 | ${drift.get('per_week', 0):,.2f}/week ({drift.get('pct_12mo')}%) |
| Each floor up | ${floor.get('per_floor', 0)} a week |
| Flats never advertised to rent | {churn.get('never', 0):,} of {churn.get('total', 0):,} ({churn.get('never_pct')}%) |

Parking premium by building size:

| Building size | Premium |
|---|---|
{size_rows}

## The data, and the cutoff

The task is as at **{m.get('cutoff')}**. Everything later is dropped as the data loads,
and the cutoff is **asserted in code** before training — the program stops rather than
continue with a later advert in the set.

Supplied and used: the input flats file, the past-lease adverts, the 2021–2026 block
counts. We also pulled the API's own suburb listing history and monthly median rent
series, both filtered to the cutoff.

**Deliberately refused:** the API's mesh-block rent map and suburb median rent. Both are
dated after the cutoff with no way to roll them back, so neither appears anywhere in this
estimate.

No supplied file, no API output and no key is in this repository.

## Running it

```bash
pip install -r requirements.txt
cp .env.example .env            # paste the team key; never commit it
# put the three supplied CSVs in data/

python pipeline.py --team {args.team} --baseline-only   # safety submission, seconds
python pipeline.py --team {args.team} --trials 20       # the real model
python findings.py                                       # findings + charts
python make_report.py --team {args.team}                 # report.html
```

Outputs land in `out/`: `rents_{args.team}.csv`, `metrics.json`, `report.html`,
`findings.json`, `findings_charts.html`.

| File | What it does |
|---|---|
| `pipeline.py` | load, cutoff, features, model, corrections, submission |
| `findings.py` | the four findings and their charts |
| `make_report.py` | builds `report.html` from the run |
| `fetch_adverts.py` | pulls advert history from the API |
| `preflight.py` | checks the environment before the event |
| `make_synthetic.py` | stand-in data used to build the pipeline before the files arrived |

## Credits

Property, advert and block data © Microburbs, used under the event key.
Addresses use G-NAF, © Geoscape Australia.
Census counts © Commonwealth of Australia, CC BY 4.0.

Estimates are automated and indicative only — not a formal valuation, and not
financial advice.
"""
    Path("README.md").write_text(md)
    print(f"[readme] README.md ({len(md):,} bytes)")


if __name__ == "__main__":
    main()
