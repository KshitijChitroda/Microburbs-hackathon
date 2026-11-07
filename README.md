# What will this flat rent for?

Microburbs Hackathon, 7 November 2026 — **Task 4**. Weekly asking rent for all
65,777 City of Sydney flats, estimated twice: with a car space and without.

Team **k-and-k**.

## What we built

An estimate of the weekly asking rent of every flat in the council area, under
two conditions — with a car space and without. The gap between those two numbers
is a measurement in its own right, and it is where the interesting findings came
from.

**A car space is worth $100 a week in a block of under 20 flats, and $18 in a larger building — 6× the price for the same car space.** Towers are built with basement parking, so nobody pays extra for it. Small walk-ups mostly have none, so the few spaces are scarce.

Full write-up with charts: **`report.html`**.

## Who it helps

- A **tenant** can check whether an asking rent is ordinary for that building and street.
- A **landlord** can price a flat, and a car space, against comparable ones nearby.
- A **buyer or investor** can compare likely rent before committing.

## Results

Held out: adverts dated 2026-05-01 to 2026-07-31, never used to build or tune.

| Estimator | Typical miss | Within $100 |
|---|---|---|
| **Our estimate** | **$57.27** | **70.76%** |
| Microburbs rental valuation (published) | $86 | 55.21% |
| Our group-median baseline | $100 | 56.87% |
| Microburbs simple estimate (published) | $100 | — |

R squared **0.7995** on 4,083 held-out adverts. Our typical
miss is **33% smaller** than the published figure for their own valuation.

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
| Car space, blocks under 20 flats | $100 a week |
| Car space, larger buildings | $18 a week |
| Car space, council-wide median | $72 a week |
| Rent growth, year to 31 Jul 2026 | $0.95/week (5.9%) |
| Each floor up | $7.3 a week |
| Flats never advertised to rent | 16,094 of 65,777 (24.5%) |

Parking premium by building size:

| Building size | Premium |
|---|---|
| 0 to 20 flats | $100 |
| 20 to 60 flats | $32 |
| 60 to 150 flats | $18 |
| 150+ flats | $25 |

## The data, and the cutoff

The task is as at **2026-07-31**. Everything later is dropped as the data loads,
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

python pipeline.py --team k-and-k --baseline-only   # safety submission, seconds
python pipeline.py --team k-and-k --trials 20       # the real model
python findings.py                                       # findings + charts
python make_report.py --team k-and-k                 # report.html
```

Outputs land in `out/`: `rents_k-and-k.csv`, `metrics.json`, `report.html`,
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
