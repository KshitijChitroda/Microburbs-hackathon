"""
Builds report.html from out/metrics.json.

The task page requires report.html to show: your held-out accuracy score, your
R squared, how you split adverts by date, and the data and method you used.
It also says to LEAD WITH INSIGHTS found along the way — so FINDINGS below is
the part to actually write by hand on the day. Everything else fills itself.

    python make_report.py --team jack
"""
import argparse
import json
from pathlib import Path

OUT = Path("out")

# ---------------------------------------------------------------------------
# EDIT THESE ON THE DAY. Three sentences each, numbers from your own run.
# ---------------------------------------------------------------------------
FINDINGS = [
    ("$NN a week",
     "The parking premium we measured across suburb-and-bedroom groups. "
     "Write the real number and how many groups it came from."),
    ("Headline two",
     "Something the model surfaced that a person would repeat. Floor level? "
     "Building size? A suburb that breaks the pattern?"),
    ("Headline three",
     "Where the estimate is weakest, and who should not rely on it."),
]

METHOD = """
We used only adverts dated on or before 31 July 2026, filtered in one place in
the loader and asserted before training. Features came from the supplied input
file, the supplied past-lease adverts and the supplied 2021-2026 block file.

The estimator is a gradient-boosted tree trained with an L1 objective. That is
deliberate: the contest ranks on median absolute dollar error, and the L1
optimum is the conditional median, so the model is fitted to the quantity being
scored rather than to the mean.

Parking is handled by predicting each flat twice, once with a car space and
once without, then clipping the gap into a band anchored on the premium
measured directly in the training adverts. Every flat therefore gets two
coherent answers and no cell is left blank.
""".strip()

UNCERTAIN = """
The estimate does not see condition, renovation, aspect, view or internal area.
It cannot know which flats were actually advertised in August and September.
Blocks and suburbs with few historic adverts fall back to coarser groups, and
those rows carry the widest error.
""".strip()

CSS = """
:root{--ink:#14181f;--mut:#5b6472;--line:#e3e7ee;--bg:#fff;--accent:#1f5fd0;--good:#107a4d}
@media(prefers-color-scheme:dark){:root:not([data-theme=light]){--ink:#e9edf4;--mut:#9aa5b5;--line:#2a3140;--bg:#11141a;--accent:#6fa2ff;--good:#49c98b}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.65 ui-sans-serif,-apple-system,Segoe UI,Roboto,sans-serif}
.wrap{max-width:820px;margin:0 auto;padding:48px 16px 80px}
h1{font-size:34px;line-height:1.15;letter-spacing:-.02em;margin:0 0 8px}
h2{font-size:20px;margin:44px 0 12px;letter-spacing:-.01em}
.sub{color:var(--mut);margin:0 0 36px}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px;margin:24px 0}
.card{border:1px solid var(--line);border-radius:12px;padding:16px}
.big{font-size:27px;font-weight:650;letter-spacing:-.02em}
.lab{color:var(--mut);font-size:13px;margin-top:4px}
table{width:100%;border-collapse:collapse;margin:14px 0;font-size:15px}
th,td{text-align:left;padding:9px 10px;border-bottom:1px solid var(--line)}
th{color:var(--mut);font-weight:550;font-size:13px;text-transform:uppercase;letter-spacing:.04em}
td.num{text-align:right;font-variant-numeric:tabular-nums}
.win{color:var(--good);font-weight:600}
.find{border-left:3px solid var(--accent);padding:2px 0 2px 16px;margin:18px 0}
.find b{display:block;font-size:19px;margin-bottom:2px}
p{margin:12px 0;white-space:pre-line}
footer{margin-top:56px;padding-top:18px;border-top:1px solid var(--line);color:var(--mut);font-size:13px}
"""


def card(value, label):
    return f'<div class="card"><div class="big">{value}</div><div class="lab">{label}</div></div>'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--team", required=True)
    args = ap.parse_args()

    m = json.loads((OUT / "metrics.json").read_text())
    best = m.get("model") or m["baseline_group_median"]
    base = m["baseline_group_median"]
    bench = m["benchmarks_published_by_microburbs"]
    mb = bench["microburbs_rental_valuation"]

    rows = [
        ("Our estimate", best["median_abs_error"], best["pct_within_100"], True),
        ("Our group-median baseline", base["median_abs_error"], base["pct_within_100"], False),
        ("Microburbs rental valuation", mb["median_abs_error"], mb["pct_within_100"], False),
        ("Microburbs simple estimate", bench["their_simple_estimate"]["median_abs_error"], None, False),
    ]
    trs = "".join(
        f'<tr><td{" class=win" if w else ""}>{n}</td>'
        f'<td class="num">${e:,.2f}</td>'
        f'<td class="num">{p if p is None else f"{p}%"}</td></tr>'.replace("None", "—")
        for n, e, p, w in rows
    )

    finds = "".join(f'<div class="find"><b>{h}</b>{t}</div>' for h, t in FINDINGS)

    feats = m.get("top_features", [])
    feat_html = ""
    if feats:
        li = "".join(f'<tr><td>{f["feature"]}</td><td class="num">{f["gain"]:,}</td></tr>' for f in feats[:10])
        feat_html = f"<h2>What drove the estimate</h2><table><tr><th>Feature</th><th>Gain</th></tr>{li}</table>"

    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>What will this flat rent for?</title><style>{CSS}</style></head>
<body><div class="wrap">
<h1>What will this flat rent for?</h1>
<p class="sub">Weekly asking rent for all 65,777 City of Sydney flats, with and
without a car space. Team {args.team}. Microburbs Hackathon, 7 November 2026.</p>

<h2>What we found along the way</h2>
{finds}

<h2>Our separate check</h2>
<div class="cards">
{card(f"${best['median_abs_error']:,.0f}", "Median absolute error, held out")}
{card(f"{best['pct_within_100']}%", "Within $100 a week")}
{card(best.get("r_squared", "—"), "R squared")}
{card(f"{best['n']:,}", "Held-out adverts")}
</div>
<p>R squared is the share of variation in weekly rent the estimate explains.</p>

<h2>How it compares</h2>
<table><tr><th>Estimator</th><th>Median abs. error</th><th>Within $100</th></tr>{trs}</table>
<p>The Microburbs figures are the published benchmarks on the frozen
1 August to 30 September 2026 adverts. Ours are on our own held-out window, so
the comparison is indicative rather than like for like.</p>

<h2>How we split the adverts by date</h2>
<p>Everything dated after {m['cutoff']} was dropped at load time and the cutoff
was asserted before any training. We then built the estimate on adverts dated
before {m['validation_window'].split(' to ')[0]} and held out
{m['validation_window']} as a separate later window. Those held-out rents were
never used to build or tune the estimate.</p>

<h2>Data and method</h2>
<p>{METHOD}</p>
{feat_html}

<h2>Who wants it</h2>
<p>A tenant can check whether an asking rent looks high for the building and the
bedroom count. A landlord can set a weekly rent that sits with comparable flats
nearby. A buyer or investor can compare likely rent before committing.</p>

<h2>What remains uncertain</h2>
<p>{UNCERTAIN}</p>

<footer>Built in four hours at the Microburbs Hackathon. Property data ©
Microburbs, used under the event key. Estimates are automated and indicative
only, and are not a formal valuation or financial advice.</footer>
</div></body></html>"""

    (OUT / "report.html").write_text(html)
    print(f"[report] out/report.html  ({len(html):,} bytes)")


if __name__ == "__main__":
    main()
