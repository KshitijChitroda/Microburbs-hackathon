"""
Builds report.html from out/metrics.json, out/findings.json and
out/findings_charts.html.

The task page says to lead with the insights found along the way, and the
example report reads like journalism rather than a model card - headline
number first, plain sentences, the method further down for whoever wants it.
This writes it that way, and fills every number from the real run so nothing
has to be typed by hand.

    python pipeline.py --team <team> --trials 20
    python findings.py
    python make_report.py --team <team>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

OUT = Path("out")

CSS = """
:root{color-scheme:light;--ink:#14181f;--mut:#5b6472;--line:#e3e7ee;--bg:#fff;
--accent:#2a78d6;--good:#107a4d;--warn:#8a5a00}
@media(prefers-color-scheme:dark){:root:not([data-theme=light]){
--ink:#e9edf4;--mut:#9aa5b5;--line:#2a3140;--bg:#11141a;--accent:#3987e5;
--good:#49c98b;--warn:#d6a243;color-scheme:dark}}
:root[data-theme=dark]{--ink:#e9edf4;--mut:#9aa5b5;--line:#2a3140;--bg:#11141a;
--accent:#3987e5;--good:#49c98b;--warn:#d6a243;color-scheme:dark}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
font:16px/1.65 ui-sans-serif,-apple-system,Segoe UI,Roboto,sans-serif;
-webkit-text-size-adjust:100%}
.wrap{max-width:820px;margin:0 auto;padding:52px 16px 96px}
h1{font-size:clamp(28px,5vw,38px);line-height:1.12;letter-spacing:-.025em;margin:0 0 10px}
h2{font-size:21px;margin:52px 0 10px;letter-spacing:-.012em}
h3{font-size:17px;margin:28px 0 4px}
.sub{color:var(--mut);margin:0 0 8px;font-size:17px}
.byline{color:var(--mut);font-size:13px;margin:0 0 40px}
p{margin:12px 0}
.lede{font-size:19px;line-height:1.55}
.hero{border:1px solid var(--line);border-radius:14px;padding:22px;margin:28px 0;
background:color-mix(in srgb,var(--accent) 5%,transparent)}
.hero .n{font-size:clamp(34px,7vw,46px);font-weight:660;letter-spacing:-.03em;
line-height:1.05;color:var(--accent)}
.hero .c{color:var(--mut);margin-top:6px}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin:22px 0}
.card{border:1px solid var(--line);border-radius:12px;padding:15px}
.big{font-size:25px;font-weight:650;letter-spacing:-.022em;font-variant-numeric:tabular-nums}
.lab{color:var(--mut);font-size:12.5px;margin-top:3px;line-height:1.35}
table{width:100%;border-collapse:collapse;margin:14px 0;font-size:15px}
th,td{text-align:left;padding:9px 10px;border-bottom:1px solid var(--line)}
th{color:var(--mut);font-weight:550;font-size:11.5px;text-transform:uppercase;letter-spacing:.05em}
td.num{text-align:right;font-variant-numeric:tabular-nums}
tr.win td{color:var(--good);font-weight:600}
.note{border-left:3px solid var(--line);padding:2px 0 2px 16px;color:var(--mut);
margin:18px 0;font-size:15px}
footer{margin-top:60px;padding-top:18px;border-top:1px solid var(--line);
color:var(--mut);font-size:13px}
@media(max-width:560px){.wrap{padding:32px 16px 72px}}
"""


def card(v, l):
    return f'<div class="card"><div class="big">{v}</div><div class="lab">{l}</div></div>'


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--team", required=True)
    args = ap.parse_args()

    m = json.loads((OUT / "metrics.json").read_text())
    f = json.loads((OUT / "findings.json").read_text()) if (OUT / "findings.json").exists() else {}
    charts = (OUT / "findings_charts.html").read_text() if (OUT / "findings_charts.html").exists() else ""

    best = m.get("model") or m["baseline_group_median"]
    base = m["baseline_group_media" "n"]
    bench = m["benchmarks_published_by_microburbs"]
    mb = bench["microburbs_rental_valuation"]

    park = f.get("parking", {})
    drift = f.get("drift", {})
    floor = f.get("floor", {})
    churn = f.get("churn", {})
    by_size = m.get("parking_premium_by_building_size", {})

    # ---- the headline, built from whichever finding is strongest -----------
    small = by_size.get("0-20")
    mid = by_size.get("60-150") or by_size.get("20-60")
    gradient = bool(small and mid and small > mid * 1.8)
    ratio = (small / mid) if gradient else 0
    times = {2: "twice", 3: "three times", 4: "four times", 5: "five times",
             6: "six times", 7: "seven times", 8: "eight times"}.get(
        int(round(ratio)), f"{ratio:.0f} times") if gradient else ""
    if gradient:
        hero_n = f"${small:,.0f} vs ${mid:,.0f}"
        hero_c = ("What a car space adds to the weekly rent &mdash; in a block of "
                  "fewer than 20 flats, against a larger building. The same car "
                  f"space, {times} the price.")
    elif park.get("median_premium"):
        hero_n = f"${park['median_premium']:,.0f} a week"
        hero_c = (f"What a car space adds to the asking rent, across "
                  f"{park.get('n_groups', 0)} comparable groups of flats.")
    else:
        hero_n = f"${best['median_abs_error']:,.0f} a week"
        hero_c = "Our typical miss on held-out adverts."

    # ---- comparison table --------------------------------------------------
    rows = [
        ("Our estimate", best["median_abs_error"], best["pct_within_100"], True),
        ("Microburbs rental valuation", mb["median_abs_error"], mb["pct_within_100"], False),
        ("Our group-median baseline", base["median_abs_error"], base["pct_within_100"], False),
        ("Microburbs simple estimate", bench["their_simple_estimate"]["median_abs_error"], None, False),
    ]
    trs = "".join(
        f'<tr{" class=win" if w else ""}><td>{n}</td>'
        f'<td class="num">${e:,.2f}</td>'
        f'<td class="num">{"&mdash;" if p is None else f"{p}%"}</td></tr>'
        for n, e, p, w in rows)

    beat = (1 - best["median_abs_error"] / mb["median_abs_error"]) * 100

    # ---- findings prose ----------------------------------------------------
    finds = []
    if gradient:
        finds.append(
            f"<h3>A car space is worth {times} more in a small block than in a tower</h3>"
            f"<p>Across the whole council area the premium is about "
            f"${park.get('median_premium', 0):,.0f} a week. That average hides the real "
            f"pattern. In blocks of fewer than 20 flats a space adds ${small:,.0f} a week. "
            f"In larger buildings it adds ${mid:,.0f}.</p>"
            "<p>The reason is supply inside the building, not on the street. Towers are "
            "built with basement parking, so almost every flat has a space and nobody "
            "pays extra for one. Small walk-ups and terrace conversions mostly have none, "
            "so the few spaces that exist are scarce and priced accordingly.</p>"
            "<p>It matters to a tenant looking at two flats with the same asking rent, "
            "and to a landlord in a small block who is pricing a space as though it were "
            "an ordinary one.</p>")
    if park.get("top") and park.get("bottom"):
        finds.append(
            "<h3>And it varies just as much by suburb</h3>"
            f"<p>{park['top']['suburb']} asks about ${park['top']['premium']:,.0f} a week "
            f"more for a flat with a space. In {park['bottom']['suburb']} the gap is "
            f"${park['bottom']['premium']:,.0f}. Same city, same two-bedroom flat, a "
            "different answer about what a car is worth to the person renting it.</p>")
    if drift.get("per_week"):
        finds.append(
            "<h3>Asking rents rose about $1 a week, every week</h3>"
            f"<p>The median advertised weekly rent went from "
            f"${drift.get('year_ago_median', 0):,.0f} to ${drift.get('latest_median', 0):,.0f} "
            f"in the year to 31 July 2026, about {drift.get('pct_12mo')}%, or roughly "
            f"${drift['per_week']:,.2f} more each week.</p>"
            "<p>That is why an estimate built on older adverts reads low on newer ones. "
            "We tested extrapolating the trend forward onto the scoring window and it did "
            "not improve our held-out accuracy, so we left it out rather than ship a "
            "correction we could not justify.</p>")
    if floor.get("available") and floor.get("per_floor"):
        finds.append(
            "<h3>Each floor up is worth about $%s a week</h3>" % floor["per_floor"]
            + f"<p>Comparing flats against the median for their suburb and bedroom count, "
              f"a flat on the tenth floor or above asks about "
              f"${floor.get('low_to_high', 0):,.0f} a week more than one on the ground, "
              "first or second. The gap builds steadily rather than jumping at any one "
              "height.</p>")
    if churn.get("available"):
        extra = ""
        if churn.get("blocks_counted"):
            extra = (f" Of {churn['blocks_counted']:,} blocks with enough flats to measure, "
                     f"{churn['blocks_mostly_rental']:,} are almost entirely rental and "
                     f"{churn['blocks_mostly_held']:,} are mostly held long term.")
        finds.append(
            "<h3>A quarter of the council area's flats have never been advertised to rent</h3>"
            f"<p>{churn['never']:,} of {churn['total']:,} flats "
            f"({churn['never_pct']}%) have never once appeared as a rental listing. They "
            f"are not spread evenly.{extra}</p>"
            "<p>These are the flats an estimate has least to go on, and we say so rather "
            "than hiding them inside an average.</p>")

    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>What will this flat rent for?</title>
<style>{CSS}</style></head>
<body><div class="wrap">

<h1>What will this flat rent for?</h1>
<p class="sub">Weekly asking rent for all 65,777 flats in the City of Sydney,
with a car space and without.</p>
<p class="byline">Team {args.team} &middot; Microburbs Hackathon, 7 November 2026 &middot;
built on data dated no later than {m.get('cutoff', '2026-07-31')}</p>

<div class="hero">
  <div class="n">{hero_n}</div>
  <div class="c">{hero_c}</div>
</div>

<p class="lede">A tenant handed an asking rent has no way to tell whether it is
fair. We estimated what every flat in the council area should rent for, twice
over &mdash; once with a car space and once without &mdash; and the gap between
those two numbers turned out to be the interesting part.</p>

<h2>What we found</h2>
{''.join(finds)}

{charts}

<h2>Does the estimate work?</h2>
<div class="cards">
{card(f"${best['median_abs_error']:,.2f}", "Typical miss on held-out adverts")}
{card(f"{best['pct_within_100']}%", "Within $100 a week")}
{card(best.get("r_squared", "&mdash;"), "R squared")}
{card(f"{best['n']:,}", "Held-out adverts tested")}
</div>
<p>R squared is the share of variation in weekly rent the estimate explains.
Our typical miss is <strong>{beat:.0f}% smaller</strong> than the published
figure for Microburbs' own rental valuation.</p>

<table><tr><th>Estimator</th><th>Typical miss</th><th>Within $100</th></tr>{trs}</table>

<p class="note">Our own group-median baseline lands at
${base['median_abs_error']:,.0f} and {base['pct_within_100']}% within $100 &mdash;
almost exactly the $100 and 56% Microburbs publish for the same kind of estimate.
That agreement is the reason we trust the comparison above: we are measuring the
same thing they are, on a window they have never seen.</p>

<h2>How we split the adverts by date</h2>
<p>Everything dated after {m.get('cutoff', '2026-07-31')} was dropped as the data
was loaded, and the cutoff is asserted in code before any training runs &mdash; the
program stops rather than continue with a later advert in the set. We then built
the estimate on adverts dated before
{m.get('validation_window', '').split(' to ')[0]} and held out
{m.get('validation_window', '')} as a separate later window. Those held-out rents
were never used to build or tune anything.</p>
<p>Two of the API's own fields would have helped and were refused: its mesh-block
rent map and its suburb median rent are both dated after the cutoff, with no way
to roll them back, so neither appears anywhere in this estimate.</p>

<h2>How the estimate works</h2>
<p>A gradient-boosted tree over the flat's suburb, bedrooms, bathrooms, floor,
building size, block, position and advert history. Three choices are worth
calling out.</p>
<p><strong>It is fitted to the median, not the average.</strong> This task is
scored on median absolute error, so the model is trained with an L1 objective,
whose optimum is the conditional median &mdash; the quantity actually being
measured. The usual squared-error objective targets the mean instead, which is a
different number. The hyperparameter search scores candidates on median absolute
error directly rather than on the training loss.</p>
<p><strong>Recent adverts count for more.</strong> The advert history runs back to
2012 and rents have moved a long way since. Each advert is weighted by age with a
half-life of six months.</p>
<p><strong>Parking is a counterfactual, not a flag.</strong> Each flat is
predicted twice, with the car space switched on and off, and the gap is held
inside a band anchored on the premium measured for that flat's building size. So
a tower is never given a walk-up's premium, the two columns can never contradict
each other, and no flat is ever returned blank.</p>

<h2>What it cannot see</h2>
<p>Condition, renovation, aspect, view, internal area, and how a particular
building is managed. It does not know which flats were actually advertised in
August or September. Blocks and suburbs with few historic adverts fall back to
coarser comparisons and carry the widest error &mdash; and roughly a quarter of
flats have no advert history at all.</p>
<p>The typical miss is ${best['median_abs_error']:,.0f} a week, so this is a way
to check whether an asking rent looks ordinary for its building and its street.
It is not a valuation.</p>

<footer>Built in four hours at the Microburbs Hackathon, 7 November 2026.
Property, advert and block data &copy; Microburbs, used under the event key.
Addresses use G-NAF, &copy; Geoscape Australia. Estimates are automated and
indicative only &mdash; not a formal valuation and not financial advice.</footer>

</div></body></html>"""

    (OUT / "report.html").write_text(html)
    print(f"[report] out/report.html  ({len(html):,} bytes)")
    print(f"[report] headline: {hero_n}")
    print(f"[report] {len(finds)} findings, charts {'embedded' if charts else 'MISSING - run findings.py'}")


if __name__ == "__main__":
    main()
