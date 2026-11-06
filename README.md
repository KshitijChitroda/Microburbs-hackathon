# What will this flat rent for?

Microburbs Hackathon, 7 November 2026 — Task 4. Weekly asking rent for all
65,777 City of Sydney flats, estimated twice: with a car space and without.

**Fill in on the day:** team name, final held-out numbers, and the one-line
finding. Everything marked `TODO` below.

## What it is

TODO — one paragraph. What we built and the single number worth repeating.

## Who it helps

A tenant checking whether an asking rent is high for the building and bedroom
count. A landlord setting a rent that sits with comparable flats nearby. A
buyer or investor comparing likely rent before committing.

## The data

- `rents-input-flats-city-of-sydney.csv` — 65,777 flats across 33 suburbs, supplied
- `rents-past-leases-city-of-sydney.csv` — historic adverts, supplied
- `blocks-homes-and-people-2021-2026.csv` — 2021 Census counts and 2026 estimates, supplied

Nothing dated after **31 July 2026** is used, from these files or anywhere
else. The cutoff is applied in one place (`load_leases` in `pipeline.py`) and
asserted before training.

Supplied files, API output and the team key are **not** in this repository.

## The clever step

The contest ranks on **median absolute dollar error**. So the model is trained
with an **L1 objective**, whose optimum is the conditional median — the exact
quantity being scored — rather than the default squared-error objective, which
targets the mean. The hyperparameter search also scores candidates on median
absolute error directly, not on the training loss.

Parking is a counterfactual, not a feature flag: each flat is predicted twice
and the gap is clipped into a band anchored on the premium measured in the
training adverts, so the two output columns are always coherent and never
blank.

## How to run

```bash
pip install -r requirements.txt
cp .env.example .env          # paste the team key, never commit it
python pipeline.py --team TEAMNAME --baseline-only   # safety submission, seconds
python pipeline.py --team TEAMNAME --trials 40       # full model
python make_report.py --team TEAMNAME
```

Outputs land in `out/`: `rents_<team>.csv`, `metrics.json`, `report.html`.

`make_synthetic.py` generates stand-in files with the published column names.
It was used to build and test the pipeline before the real data was released;
it is not used in the submitted result.

## Results

| Estimator | Median abs. error | Within $100 |
|---|---|---|
| Our estimate | TODO | TODO |
| Our group-median baseline | TODO | TODO |
| Microburbs rental valuation (published) | $86 | 55.21% |

Held out: adverts dated 1 May to 31 July 2026, never used to build or tune.

## Credits

Property and block data © Microburbs, used under the event key.
Addresses use G-NAF, © Geoscape Australia. Census counts © Commonwealth of
Australia, CC BY 4.0.

Estimates are automated and indicative only — not a formal valuation or
financial advice.
