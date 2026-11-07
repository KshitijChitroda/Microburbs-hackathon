"""
The findings, and the charts for them.

Runs on the supplied files alone - no model needed, so it can run while the
model trains. Writes:

    out/findings.json          the numbers, for the report and the pitch
    out/findings_charts.html   a self-contained section of inline-SVG charts

Four findings:
  1. Rent drift      - City of Sydney unit rents, monthly, to the cutoff
  2. Parking premium - by suburb, and whether it scales with building density
  3. Floor premium   - what a floor is worth, and where it stops mattering
  4. Churn           - the flats that have never been advertised, by suburb

Everything is filtered to 2026-07-31. Nothing dated later is read.

    python findings.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

CUTOFF = pd.Timestamp("2026-07-31")
RECENT_DAYS = 365
DATA, OUT = Path("data"), Path("out")
OUT.mkdir(exist_ok=True)

INPUT = DATA / "rents-input-flats-city-of-sydney.csv"
LEASES = DATA / "rents-past-leases-city-of-sydney.csv"
SERIES = DATA / "suburb-median-rent-series.csv"

MIN_N = 12          # minimum adverts before a group is reported

DATE_CANDS = ["listing_date", "advert_date", "date", "date_listed", "first_seen"]
RENT_CANDS = ["weekly_rent", "rent", "price", "rent_pw", "asking_rent"]


# --------------------------------------------------------------------------- io
def pick(df: pd.DataFrame, cands: list[str]) -> str | None:
    low = {c.lower(): c for c in df.columns}
    for c in cands:
        if c in low:
            return low[c]
    return next((orig for l, orig in low.items()
                 if any(c in l for c in cands)), None)


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    inp = pd.read_csv(INPUT, dtype={"gnaf_id": str, "mb": str})
    lea = pd.read_csv(LEASES, dtype={"gnaf_id": str})

    dcol, rcol = pick(lea, DATE_CANDS), pick(lea, RENT_CANDS)
    lea = lea.rename(columns={dcol: "date", rcol: "rent"})
    lea["date"] = pd.to_datetime(lea.date, errors="coerce")
    lea["rent"] = pd.to_numeric(lea.rent, errors="coerce")
    lea = lea.dropna(subset=["date", "rent"])
    lea = lea[(lea.rent > 100) & (lea.rent < 6000) & (lea.date <= CUTOFF)]

    # advert-level attributes win; otherwise take the flat's
    keep = [c for c in ("gnaf_id", "suburb", "bedrooms", "bathrooms", "car_space",
                        "floor_level", "flats_in_building", "building_address",
                        "property_type", "mb") if c in inp.columns]
    lea = lea.merge(inp[keep], on="gnaf_id", how="left", suffixes=("_adv", ""))
    for c in ("bedrooms", "bathrooms", "car_space"):
        a = f"{c}_adv"
        if a in lea.columns and c in lea.columns:
            lea[c] = pd.to_numeric(lea[a], errors="coerce").fillna(
                pd.to_numeric(lea[c], errors="coerce"))
    for c in ("bedrooms", "bathrooms", "car_space", "floor_level", "flats_in_building"):
        if c in lea.columns:
            lea[c] = pd.to_numeric(lea[c], errors="coerce")
    assert lea.date.max() <= CUTOFF, "post-cutoff advert survived"
    return inp, lea


# ---------------------------------------------------------------- 1. the drift
def f_drift(lea: pd.DataFrame) -> dict:
    m = (lea.assign(mo=lea.date.dt.to_period("M").dt.to_timestamp())
            .groupby("mo").rent.agg(["median", "size"]))
    m = m[m["size"] >= 30]
    recent = m[m.index >= m.index.max() - pd.Timedelta(days=RECENT_DAYS)]
    out = {"n_months": int(len(m))}
    if len(recent) >= 8:
        x = (recent.index - recent.index.min()).days.to_numpy(float)
        slope = float(np.polyfit(x, recent["median"].to_numpy(), 1)[0])
        out |= {"per_week": round(slope * 7, 2), "per_year": round(slope * 365, 0),
                "latest_median": float(recent["median"].iloc[-1]),
                "year_ago_median": float(recent["median"].iloc[0])}
        out["pct_12mo"] = round(
            (out["latest_median"] / out["year_ago_median"] - 1) * 100, 1)

    if SERIES.exists():
        s = pd.read_csv(SERIES)
        s["date"] = pd.to_datetime(s.date, errors="coerce")
        s = s[(s.date <= CUTOFF) & (s.series == "unit")]
        if len(s):
            agg = s.groupby("date").value.median()
            agg = agg[agg.index >= "2016-01-01"]
            out["series_points"] = [
                {"date": d.strftime("%Y-%m"), "value": float(v)}
                for d, v in agg.items()]
    if "series_points" not in out:
        out["series_points"] = [{"date": d.strftime("%Y-%m"), "value": float(v)}
                                for d, v in m["median"].items()]
    return out


# ------------------------------------------------------------- 2. the premium
def f_parking(lea: pd.DataFrame) -> dict:
    d = lea.dropna(subset=["car_space", "bedrooms", "suburb"]).copy()
    d = d[d.date >= d.date.max() - pd.Timedelta(days=RECENT_DAYS * 2)]
    d["car_space"] = (d.car_space >= 1).astype(int)

    def gap(keys):
        g = d.groupby(keys + ["car_space"], observed=True).rent.agg(["median", "size"])
        g = g[g["size"] >= MIN_N]["median"].unstack("car_space")
        if g.shape[1] < 2:
            return pd.Series(dtype=float)
        return (g[1] - g[0]).dropna()

    overall = gap(["suburb", "bedrooms"])
    out = {"median_premium": round(float(overall.median()), 0) if len(overall) else None,
           "n_groups": int(len(overall))}

    by_sub = (gap(["suburb", "bedrooms"]).groupby("suburb").median()
              .sort_values(ascending=False))
    out["by_suburb"] = [{"suburb": s.replace(" (NSW)", ""), "premium": round(float(v))}
                        for s, v in by_sub.items() if np.isfinite(v)]
    if len(by_sub) >= 4:
        out["top"] = {"suburb": out["by_suburb"][0]["suburb"],
                      "premium": out["by_suburb"][0]["premium"]}
        out["bottom"] = {"suburb": out["by_suburb"][-1]["suburb"],
                         "premium": out["by_suburb"][-1]["premium"]}

    # does it scale with density? bigger buildings = less street parking
    if "flats_in_building" in d.columns and d.flats_in_building.notna().any():
        d["band"] = pd.cut(d.flats_in_building, [0, 20, 60, 150, 10_000],
                           labels=["under 20", "20-60", "60-150", "150+"])
        rows = []
        for band, part in d.groupby("band", observed=True):
            g = part.groupby(["bedrooms", "car_space"], observed=True).rent.agg(["median", "size"])
            g = g[g["size"] >= MIN_N]["median"].unstack("car_space")
            if g.shape[1] == 2:
                v = (g[1] - g[0]).dropna()
                if len(v):
                    rows.append({"band": str(band), "premium": round(float(v.median())),
                                 "n": int(part.shape[0])})
        out["by_building_size"] = rows
    return out


# ---------------------------------------------------------------- 3. the floor
def f_floor(lea: pd.DataFrame) -> dict:
    if "floor_level" not in lea.columns or lea.floor_level.notna().sum() < 500:
        return {"available": False}
    d = lea.dropna(subset=["floor_level", "bedrooms", "suburb"]).copy()
    d = d[(d.floor_level >= 0) & (d.floor_level <= 40)]
    d = d[d.date >= d.date.max() - pd.Timedelta(days=RECENT_DAYS * 2)]

    # rent relative to the suburb x bedroom median, so floors are comparable
    base = d.groupby(["suburb", "bedrooms"], observed=True).rent.transform("median")
    d["rel"] = d.rent - base

    bands = [(0, 2), (3, 5), (6, 9), (10, 14), (15, 19), (20, 40)]
    rows = []
    for lo, hi in bands:
        part = d[(d.floor_level >= lo) & (d.floor_level <= hi)]
        if len(part) >= MIN_N * 4:
            rows.append({"band": f"{lo}-{hi}" if hi < 40 else f"{lo}+",
                         "rel": round(float(part.rel.median())), "n": int(len(part))})
    out = {"available": len(rows) >= 3, "bands": rows}
    if out["available"]:
        lowf = d[d.floor_level <= 2]
        highf = d[d.floor_level >= 10]
        if len(lowf) >= 50 and len(highf) >= 50:
            out["low_to_high"] = round(float(highf.rel.median() - lowf.rel.median()))
        fit = d[d.floor_level <= 12]
        if len(fit) >= 500:
            out["per_floor"] = round(float(np.polyfit(fit.floor_level, fit.rel, 1)[0]), 1)
    return out


# ---------------------------------------------------------------- 4. the churn
def f_churn(inp: pd.DataFrame) -> dict:
    if "ever_advertised_for_lease" not in inp.columns:
        return {"available": False}
    e = pd.to_numeric(inp.ever_advertised_for_lease, errors="coerce")
    out = {"available": True,
           "total": int(len(inp)),
           "never": int((e == 0).sum()),
           "never_pct": round(float((e == 0).mean() * 100), 1)}
    if "suburb" in inp.columns:
        g = inp.assign(never=(e == 0)).groupby("suburb").never.agg(["mean", "size"])
        g = g[g["size"] >= 200].sort_values("mean", ascending=False)
        out["by_suburb"] = [{"suburb": s.replace(" (NSW)", ""),
                             "never_pct": round(float(r["mean"] * 100), 1),
                             "n": int(r["size"])} for s, r in g.iterrows()]
    if "mb" in inp.columns:
        b = inp.assign(never=(e == 0)).groupby("mb").never.agg(["mean", "size"])
        b = b[b["size"] >= 20]
        if len(b):
            out["blocks_counted"] = int(len(b))
            out["blocks_mostly_rental"] = int((b["mean"] <= 0.10).sum())
            out["blocks_mostly_held"] = int((b["mean"] >= 0.60).sum())
    return out


# ----------------------------------------------------------------------- charts
CSS = """
.viz{color-scheme:light;--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;
--grid:#e6e5e1;--s1:#2a78d6;--s2:#eb6834;
font:14px/1.5 ui-sans-serif,-apple-system,Segoe UI,Roboto,sans-serif;color:var(--ink)}
@media(prefers-color-scheme:dark){:root:where(:not([data-theme=light])) .viz{
color-scheme:dark;--surface:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;
--grid:#333230;--s1:#3987e5;--s2:#d95926}}
:root[data-theme=dark] .viz{color-scheme:dark;--surface:#1a1a19;--ink:#f0efec;
--ink2:#c3c2b7;--grid:#333230;--s1:#3987e5;--s2:#d95926}
.viz figure{margin:28px 0}
.viz figcaption{color:var(--ink2);font-size:13px;margin-top:6px}
.viz h3{font-size:17px;margin:0 0 2px;color:var(--ink)}
.viz svg{display:block;width:100%;height:auto;overflow:visible}
.viz .ax{fill:var(--ink2);font-size:11px}
.viz .vl{fill:var(--ink);font-size:11px;font-variant-numeric:tabular-nums}
.viz .inbar{fill:#fff}
.viz table{width:100%;border-collapse:collapse;font-size:13px;margin-top:8px}
.viz th,.viz td{text-align:left;padding:5px 8px;border-bottom:1px solid var(--grid)}
.viz th{color:var(--ink2);font-weight:550;font-size:11px;text-transform:uppercase;letter-spacing:.04em}
.viz td.n{text-align:right;font-variant-numeric:tabular-nums}
.viz details{margin-top:6px}.viz summary{color:var(--ink2);font-size:12px;cursor:pointer}
"""


def esc(s) -> str:
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def money(v, unit="$") -> str:
    """-40 -> -$40, not $-40."""
    return f"-{unit}{abs(v):,}" if v < 0 else f"{unit}{v:,}"


def hbars(rows, label_key, value_key, unit="$", width=680, rowh=24) -> str:
    """
    Horizontal bars from a zero baseline, 4px rounded data-end, direct labels.
    When any value is negative the baseline moves inside the plot and bars run
    both ways, so a -$40 never draws longer than a +$25.
    """
    if not rows:
        return ""
    padl, padr = 150, 66
    vals = [r[value_key] for r in rows]
    lo, hi = min(min(vals), 0.0), max(max(vals), 0.0)
    span = (hi - lo) or 1
    inner = width - padl - padr
    zero = padl + (0.0 - lo) / span * inner
    h = rowh * len(rows) + 8

    out = [f'<svg viewBox="0 0 {width} {h}" role="img">']
    if lo < 0:
        out.append(f'<line x1="{zero:.1f}" y1="0" x2="{zero:.1f}" y2="{h-6}" '
                   f'stroke="var(--grid)" stroke-width="1"/>')
    for i, r in enumerate(rows):
        v = r[value_key]
        y = i * rowh + 4
        end = padl + (v - lo) / span * inner
        x, w = (zero, end - zero) if v >= 0 else (end, zero - end)
        w = max(2.0, w)
        lab = money(v, unit)
        # keep the value label off the category labels: if it would cross the
        # left gutter, set it inside the bar instead
        need = 8 + 7.2 * len(lab)
        if v >= 0:
            lx, anchor, cls = x + w + 8, "start", "vl"
        elif x - need < padl:
            lx, anchor, cls = x + 6, "start", "vl inbar"
        else:
            lx, anchor, cls = x - 8, "end", "vl"
        out.append(
            f'<text class="ax" x="{padl-10}" y="{y+13}" text-anchor="end">{esc(r[label_key])}</text>'
            f'<rect x="{x:.1f}" y="{y+3}" width="{w:.1f}" height="{rowh-11}" rx="4" '
            f'fill="var(--s1)"><title>{esc(r[label_key])}: {lab}</title></rect>'
            f'<text class="{cls}" x="{lx:.1f}" y="{y+13}" text-anchor="{anchor}">{lab}</text>')
    out.append("</svg>")
    return "".join(out)


def linechart(points, width=680, height=240, unit="$") -> str:
    if len(points) < 3:
        return ""
    padl, padr, padt, padb = 56, 14, 14, 28
    xs = list(range(len(points)))
    ys = [p["value"] for p in points]
    ylo, yhi = min(ys), max(ys)
    span = (yhi - ylo) or 1
    ylo, yhi = ylo - span * .12, yhi + span * .12

    def px(i): return padl + i / (len(points) - 1) * (width - padl - padr)
    def py(v): return padt + (yhi - v) / (yhi - ylo) * (height - padt - padb)

    ticks = np.linspace(ylo, yhi, 4)
    out = [f'<svg viewBox="0 0 {width} {height}" role="img">']
    for t in ticks:
        out.append(f'<line x1="{padl}" y1="{py(t):.1f}" x2="{width-padr}" y2="{py(t):.1f}" '
                   f'stroke="var(--grid)" stroke-width="1"/>'
                   f'<text class="ax" x="{padl-8}" y="{py(t)+4:.1f}" text-anchor="end">'
                   f'{unit}{t:,.0f}</text>')
    d = " ".join(f'{"M" if i == 0 else "L"}{px(i):.1f},{py(v):.1f}' for i, v in enumerate(ys))
    out.append(f'<path d="{d}" fill="none" stroke="var(--s1)" stroke-width="2" '
               f'stroke-linejoin="round"/>')
    step = max(1, len(points) // 6)
    for i in range(0, len(points), step):
        out.append(f'<text class="ax" x="{px(i):.1f}" y="{height-8}" text-anchor="middle">'
                   f'{esc(points[i]["date"])}</text>')
    for i, v in enumerate(ys):
        out.append(f'<circle cx="{px(i):.1f}" cy="{py(v):.1f}" r="7" fill="transparent">'
                   f'<title>{esc(points[i]["date"])}: {unit}{v:,.0f}</title></circle>')
    # label the last point only
    out.append(f'<circle cx="{px(xs[-1]):.1f}" cy="{py(ys[-1]):.1f}" r="4" fill="var(--s1)" '
               f'stroke="var(--surface)" stroke-width="2"/>')
    out.append("</svg>")
    return "".join(out)


def table(rows, cols) -> str:
    if not rows:
        return ""
    head = "".join(f"<th>{esc(h)}</th>" for h, _ in cols)
    body = "".join("<tr>" + "".join(
        f'<td class="n">{esc(r.get(k, ""))}</td>' if i else f"<td>{esc(r.get(k, ''))}</td>"
        for i, (_, k) in enumerate(cols)) + "</tr>" for r in rows)
    return (f"<details><summary>Show the numbers</summary>"
            f"<table><tr>{head}</tr>{body}</table></details>")


def charts_html(f: dict) -> str:
    s = [f"<style>{CSS}</style>", '<div class="viz">']
    d = f["drift"]
    if d.get("series_points"):
        pts = d["series_points"]
        s.append(
            "<figure><h3>City of Sydney unit rents, to 31 July 2026</h3>"
            + linechart(pts)
            + f'<figcaption>Monthly median advertised weekly rent. '
              f'{"Up $%s a week over the last year, %s%%. " % (d.get("per_week"), d.get("pct_12mo")) if d.get("per_week") else ""}'
              f'Nothing dated after the cutoff is shown.</figcaption></figure>')

    p = f["parking"]
    if p.get("by_suburb"):
        rows = p["by_suburb"][:14]
        s.append("<figure><h3>What a car space is worth, by suburb</h3>"
                 + hbars(rows, "suburb", "premium")
                 + f'<figcaption>Extra weekly rent asked for an otherwise comparable '
                   f'flat with a car space, within suburb and bedroom count. '
                   f'Median across {p["n_groups"]} groups: ${p["median_premium"]:,.0f}.'
                   f'</figcaption>'
                 + table(rows, [("Suburb", "suburb"), ("Premium", "premium")])
                 + "</figure>")
    if p.get("by_building_size"):
        s.append("<figure><h3>The premium rises with building size</h3>"
                 + hbars(p["by_building_size"], "band", "premium")
                 + "<figcaption>Flats in a building, against the car-space premium. "
                   "Bigger buildings mean less street parking - and a space worth more."
                   "</figcaption></figure>")

    fl = f["floor"]
    if fl.get("available"):
        rows = [{"band": b["band"], "rel": b["rel"]} for b in fl["bands"]]
        s.append("<figure><h3>What a floor is worth</h3>"
                 + hbars(rows, "band", "rel")
                 + f'<figcaption>Rent relative to the suburb and bedroom median, by floor. '
                   f'{"About $%s a week per floor up to the twelfth. " % fl["per_floor"] if fl.get("per_floor") else ""}'
                   f'</figcaption></figure>')

    c = f["churn"]
    if c.get("available") and c.get("by_suburb"):
        rows = [{"suburb": r["suburb"], "never_pct": r["never_pct"]}
                for r in c["by_suburb"][:14]]
        s.append("<figure><h3>Flats that have never been advertised for rent</h3>"
                 + hbars(rows, "suburb", "never_pct", unit="")
                 + f'<figcaption>{c["never"]:,} of {c["total"]:,} flats '
                   f'({c["never_pct"]}%) have never appeared as a rental advert. '
                   f'They are not spread evenly.</figcaption>'
                 + table(rows, [("Suburb", "suburb"), ("% never advertised", "never_pct")])
                 + "</figure>")
    s.append("</div>")
    return "\n".join(s)


# ------------------------------------------------------------------------ main
def main() -> None:
    inp, lea = load()
    print(f"[findings] {len(inp):,} flats, {len(lea):,} adverts "
          f"to {lea.date.max().date()}")

    f = {"drift": f_drift(lea), "parking": f_parking(lea),
         "floor": f_floor(lea), "churn": f_churn(inp), "cutoff": str(CUTOFF.date())}

    (OUT / "findings.json").write_text(json.dumps(f, indent=2, default=str))
    (OUT / "findings_charts.html").write_text(charts_html(f))

    print("\n================ HEADLINES ================")
    d = f["drift"]
    if d.get("per_week"):
        print(f"DRIFT     rents up ${d['per_week']}/week over the last year "
              f"({d.get('pct_12mo')}%), ${d['year_ago_median']:,.0f} -> ${d['latest_median']:,.0f}")
    p = f["parking"]
    if p.get("median_premium"):
        print(f"PARKING   ${p['median_premium']:,.0f}/week median, {p['n_groups']} groups")
        if p.get("top"):
            print(f"          {p['top']['suburb']} ${p['top']['premium']} vs "
                  f"{p['bottom']['suburb']} ${p['bottom']['premium']}")
        for r in p.get("by_building_size", []):
            print(f"          {r['band']:>8} flats/building -> ${r['premium']}")
    fl = f["floor"]
    if fl.get("available"):
        print(f"FLOOR     ${fl.get('per_floor')}/week per floor; "
              f"floor 10+ vs 0-2 = ${fl.get('low_to_high')}")
    c = f["churn"]
    if c.get("available"):
        print(f"CHURN     {c['never']:,} of {c['total']:,} flats ({c['never_pct']}%) "
              "never advertised")
        if c.get("blocks_counted"):
            print(f"          of {c['blocks_counted']:,} blocks: "
                  f"{c['blocks_mostly_rental']:,} are >=90% rental, "
                  f"{c['blocks_mostly_held']:,} are >=60% never-advertised")
    print("===========================================")
    print("\nout/findings.json, out/findings_charts.html")


if __name__ == "__main__":
    main()
