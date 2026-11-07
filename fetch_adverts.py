"""
Build the rent-advert training set from the API, replacing the missing
rents-past-leases-city-of-sydney.csv.

Source: GET /v1/suburbs/{suburb}/listings/rent-history
It returns real adverts with gnaf_pid, weekly_rent, listing_date, bedrooms,
bathrooms, car_spaces and property_type - paginated 200 at a time, with
meta.total. Surry Hills alone has ~12,000, so 29 suburbs is roughly 1,000
calls: about three minutes at 9 req/s.

Also pulls GET /v1/suburbs/{suburb}/market/median-rent-series, a dated monthly
median rent series per suburb. That gives a measured drift estimate instead of
an inferred one, and is a report finding in its own right.

THE CUTOFF. The task is as at 2026-07-31 and the endpoints have no as-of
switch, so every row dated after that is dropped here, and the assertion at
the end fails loudly rather than quietly leaking.

Deliberately NOT used, because they are dated after the cutoff and cannot be
rolled back: market/rent-map (as_of 2026-11-01), market/median-rent,
valuation/negative-gearing (undated, derived from current rents).

    python fetch_adverts.py --suburb "Surry Hills"     # one suburb, try first
    python fetch_adverts.py                            # all 29
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

CUTOFF = pd.Timestamp("2026-07-31")
BASE = os.environ.get("MICROBURBS_BASE_URL", "https://api-ht.microburbs.com.au")
DATA = Path("data")
RAW = DATA / "api_adverts_raw.jsonl"
OUT = DATA / "rents-past-leases-from-api.csv"
SERIES = DATA / "suburb-median-rent-series.csv"

PAGE = 200
RATE = 9.0
CONCURRENCY = 6

FLAT_TYPES = {"unit", "apartment", "studio", "flat", "apartment / unit / flat"}


def load_key() -> str:
    k = os.environ.get("MICROBURBS_API_KEY")
    if not k and Path(".env").exists():
        for line in Path(".env").read_text().splitlines():
            if line.startswith("MICROBURBS_API_KEY="):
                k = line.split("=", 1)[1].strip()
    if not k:
        sys.exit("No MICROBURBS_API_KEY - put it in .env")
    return k


class Limiter:
    def __init__(self, rate: float):
        self.interval = 1.0 / rate
        self.next_at = time.monotonic()
        self.lock = asyncio.Lock()

    async def wait(self) -> None:
        async with self.lock:
            now = time.monotonic()
            if self.next_at > now:
                await asyncio.sleep(self.next_at - now)
            self.next_at = max(now, self.next_at) + self.interval


async def get(client, path: str, lim: Limiter, params=None, tries: int = 5):
    for attempt in range(tries):
        await lim.wait()
        try:
            r = await client.get(f"{BASE}{path}", params=params)
        except Exception:
            await asyncio.sleep(0.4 * (attempt + 1) + random.random())
            continue
        if r.status_code == 200:
            return r.json().get("data")
        if r.status_code == 429:
            await asyncio.sleep(float(r.headers.get("Retry-After", 2)) + random.random())
            continue
        if r.status_code == 401:
            sys.exit("401 - key rejected")
        if r.status_code in (403, 404):
            return None
        await asyncio.sleep(0.4 * (attempt + 1))
    return None


async def suburb_adverts(client, suburb: str, lim: Limiter, sink) -> int:
    """Page through one suburb's rent-advert history."""
    path = f"/v1/suburbs/{suburb}/listings/rent-history"
    first = await get(client, path, lim, {"limit": PAGE, "offset": 0})
    if not first or "results" not in first:
        print(f"  {suburb}: no data")
        return 0

    total = int(first.get("meta", {}).get("total", len(first["results"])))
    rows = list(first["results"])

    offsets = list(range(PAGE, total, PAGE))
    if offsets:
        pages = await asyncio.gather(*[
            get(client, path, lim, {"limit": PAGE, "offset": o}) for o in offsets
        ])
        for p in pages:
            if p and p.get("results"):
                rows.extend(p["results"])

    for r in rows:
        r["_suburb"] = suburb
        sink.write(json.dumps(r) + "\n")
    print(f"  {suburb}: {len(rows):,} of {total:,}")
    return len(rows)


async def suburb_series(client, suburb: str, lim: Limiter) -> list[dict]:
    d = await get(client, f"/v1/suburbs/{suburb}/market/median-rent-series", lim)
    if not d:
        return []
    out = []
    for key, pts in d.items():
        if not isinstance(pts, list):
            continue
        for p in pts:
            if isinstance(p, dict) and p.get("date") and p.get("value") is not None:
                out.append({"suburb": suburb, "series": key,
                            "date": str(p["date"])[:10], "value": p["value"]})
    return out


async def run(suburbs: list[str], key: str) -> None:
    try:
        import httpx
    except ImportError:
        sys.exit("pip install httpx")

    lim = Limiter(RATE)
    t0 = time.time()
    sem = asyncio.Semaphore(CONCURRENCY)

    async with httpx.AsyncClient(headers={"Authorization": f"Bearer {key}"},
                                 timeout=40.0,
                                 limits=httpx.Limits(max_connections=CONCURRENCY + 4)) as client:
        print(f"adverts from {len(suburbs)} suburbs")
        with RAW.open("a") as sink:
            async def one(s):
                async with sem:
                    return await suburb_adverts(client, s, lim, sink)
            counts = await asyncio.gather(*(one(s) for s in suburbs))
        print(f"  {sum(counts):,} advert rows in {time.time()-t0:.0f}s\n")

        print("median rent series")
        series = await asyncio.gather(*(suburb_series(client, s, lim) for s in suburbs))

    flat = [r for sub in series for r in sub]
    if flat:
        s = pd.DataFrame(flat)
        s["date"] = pd.to_datetime(s.date, errors="coerce")
        s = s[s.date <= CUTOFF]
        s.to_csv(SERIES, index=False)
        print(f"  {SERIES}: {len(s):,} points, {s.suburb.nunique()} suburbs, "
              f"series={sorted(s.series.unique())}")


def build(all_types: bool = False) -> None:
    if not RAW.exists():
        print("nothing fetched yet")
        return

    rows, seen = [], set()
    with RAW.open() as f:
        for line in f:
            try:
                a = json.loads(line)
            except Exception:
                continue
            gid = a.get("gnaf_pid") or a.get("gnaf_id")
            date = a.get("listing_date") or a.get("date")
            rent = a.get("weekly_rent")
            if not gid or not date or rent is None:
                continue
            ptype = str(a.get("property_type", "")).strip().lower()
            if not all_types and ptype and ptype not in FLAT_TYPES:
                continue
            k = (gid, str(date)[:10], rent)
            if k in seen:
                continue
            seen.add(k)
            cs = a.get("car_spaces", a.get("car_space"))
            rows.append({
                "gnaf_id": gid,
                "advert_date": str(date)[:10],
                "weekly_rent": rent,
                "bedrooms": a.get("bedrooms"),
                "bathrooms": a.get("bathrooms"),
                "car_space": (1 if (cs or 0) >= 1 else 0) if cs is not None else None,
                "car_spaces": cs,
                "property_type": a.get("property_type"),
                "suburb": a.get("_suburb"),
                "leased": a.get("leased"),
                "address": a.get("address"),
                "latitude": a.get("lat"),
                "longitude": a.get("lng"),
            })

    if not rows:
        print("no usable adverts parsed")
        return

    df = pd.DataFrame(rows)
    df["advert_date"] = pd.to_datetime(df.advert_date, errors="coerce")
    df["weekly_rent"] = pd.to_numeric(df.weekly_rent, errors="coerce")
    raw_n = len(df)
    df = df.dropna(subset=["advert_date", "weekly_rent"])
    df = df[(df.weekly_rent > 100) & (df.weekly_rent < 6000)]

    after_cutoff = int((df.advert_date > CUTOFF).sum())
    df = df[df.advert_date <= CUTOFF]
    assert df.advert_date.max() <= CUTOFF, "LEAK: advert after cutoff survived"

    df = df.sort_values("advert_date")
    df["advert_date"] = df.advert_date.dt.strftime("%Y-%m-%d")
    df.to_csv(OUT, index=False)

    print(f"\n{OUT}")
    print(f"  {raw_n:,} raw -> {len(df):,} usable  "
          f"({after_cutoff:,} dropped as post-cutoff)")
    print(f"  {df.gnaf_id.nunique():,} flats, {df.advert_date.min()} to {df.advert_date.max()}")
    print(f"  rent median ${df.weekly_rent.median():,.0f}, "
          f"parking known on {df.car_space.notna().mean()*100:.0f}%")
    print(f"\nUse it:\n  cp {OUT} data/rents-past-leases-city-of-sydney.csv")


PROXY = DATA / "proxy-input-flats.csv"


def make_proxy_input() -> None:
    """
    A STAND-IN for rents-input-flats-city-of-sydney.csv, built from the adverts,
    so the whole pipeline can be built and validated before the real input file
    arrives. Same column names, so pipeline.py needs no changes.

    IT IS NOT THE SUBMISSION SET. The real file names the exact 65,777 gnaf_ids
    that get scored; this one holds whichever flats happen to have advertised.
    Swap the real file in the moment it lands.
    """
    if not OUT.exists():
        print("run the fetch first")
        return
    a = pd.read_csv(OUT, dtype={"gnaf_id": str})
    a["advert_date"] = pd.to_datetime(a.advert_date)

    latest = a.sort_values("advert_date").groupby("gnaf_id").tail(1).set_index("gnaf_id")
    first = a.groupby("gnaf_id").advert_date.min()

    df = pd.DataFrame(index=latest.index).reset_index()
    for c in ("address", "suburb", "property_type", "bedrooms", "bathrooms",
              "car_space", "car_spaces", "latitude", "longitude"):
        df[c] = latest[c].to_numpy()

    # building_address: drop the unit part of "UNIT 506, 20 DUNKERLEY PLACE"
    addr = df.address.fillna("").astype(str)
    df["building_address"] = addr.str.replace(r"^\s*(UNIT|APT|SUITE|SHOP|LEVEL)\s*[^,]*,\s*",
                                              "", regex=True, case=False).str.strip()

    # floor_level from the unit number, same trick the real file uses
    unit = addr.str.extract(r"^\s*(?:UNIT|APT|SUITE)\s*([0-9]{1,4})", expand=False)
    un = pd.to_numeric(unit, errors="coerce")
    df["floor_level"] = np.where(un >= 100, (un // 100), np.nan)

    df["flats_in_building"] = df.groupby("building_address").gnaf_id.transform("size")
    df["mb"] = pd.NA
    df["block_new_homes_2021_2026"] = pd.NA
    df["block_homes_2026"] = pd.NA
    df["ever_advertised_for_lease"] = 1
    df["first_ever_listed"] = first.reindex(df.gnaf_id).dt.strftime("%Y-%m-%d").to_numpy()
    df["attrs_last_seen"] = latest.advert_date.dt.strftime("%Y-%m-%d").to_numpy()

    df.to_csv(PROXY, index=False)
    print(f"\n{PROXY}: {len(df):,} flats  (PROXY - not the submission set)")
    print(f"  floor inferred for {df.floor_level.notna().sum():,}, "
          f"{df.building_address.nunique():,} buildings")
    print(f"\nTo develop against it:\n  cp {PROXY} data/rents-input-flats-city-of-sydney.csv")
    print("  ... and overwrite it with the real file the moment that arrives.")


SUBURBS = [
    "Alexandria", "Barangaroo", "Beaconsfield (NSW)", "Camperdown (NSW)",
    "Centennial Park (NSW)", "Chippendale", "Darlinghurst",
    "Darlington (Sydney - NSW)", "Dawes Point", "Elizabeth Bay", "Erskineville",
    "Eveleigh", "Forest Lodge", "Glebe (NSW)", "Haymarket", "Millers Point",
    "Moore Park", "Potts Point", "Pyrmont", "Redfern", "Rosebery (NSW)",
    "Rushcutters Bay", "Surry Hills", "Sydney", "The Rocks (Sydney - NSW)",
    "Ultimo", "Waterloo (NSW)", "Woolloomooloo", "Zetland",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suburb", action="append", help="one suburb; repeatable")
    ap.add_argument("--build-only", action="store_true")
    ap.add_argument("--proxy-input", action="store_true",
                    help="also write a stand-in input file for development")
    ap.add_argument("--all-types", action="store_true",
                    help="keep houses too (default: flats only)")
    args = ap.parse_args()

    DATA.mkdir(exist_ok=True)
    if not args.build_only:
        asyncio.run(run(args.suburb or SUBURBS, load_key()))
    build(args.all_types)
    if args.proxy_input:
        make_proxy_input()


if __name__ == "__main__":
    main()
