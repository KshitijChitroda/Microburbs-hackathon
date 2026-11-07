"""
Fallback / enrichment: pull rent-advert history from the Microburbs API and
write it in the same shape pipeline.py expects from the past-leases file.

Two uses:
  1. INSURANCE - if rents-past-leases-city-of-sydney.csv never arrives, this
     builds a training set from the API instead.
  2. ENRICHMENT - top up the supplied file for flats it covers thinly.

Key facts it is built around:
  * ~10 requests/second per key, 429 carries Retry-After.
  * Anything outside the 29 covered suburbs returns 403 - skipped, not retried.
  * "No data" is still HTTP 200 with {"data": null, "available": false}.
  * NOTHING dated after 2026-07-31 may be used, so the cutoff is applied here
    as well as in pipeline.py. Belt and braces.

Resumable: appends to a cache as it goes, so a crash or a Ctrl-C costs you
only the batch in flight. Re-running skips everything already fetched.

    python fetch_rent_history.py --limit 2000            # try it first
    python fetch_rent_history.py                         # everything
    python fetch_rent_history.py --only-advertised       # the 49,683 with history
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

import pandas as pd

CUTOFF = "2026-07-31"
BASE = os.environ.get("MICROBURBS_BASE_URL", "https://api-ht.microburbs.com.au")
DATA = Path("data")
CACHE = DATA / "rent_history_cache.jsonl"
OUT_CSV = DATA / "rents-past-leases-from-api.csv"

CONCURRENCY = 8          # with ~10 req/s cap, 8 in flight is about right
RATE = 9.0               # requests/second, deliberately under the limit


def load_key() -> str:
    key = os.environ.get("MICROBURBS_API_KEY")
    if not key:
        env = Path(".env")
        if env.exists():
            for line in env.read_text().splitlines():
                if line.startswith("MICROBURBS_API_KEY="):
                    key = line.split("=", 1)[1].strip()
    if not key:
        sys.exit("No MICROBURBS_API_KEY - put it in .env")
    return key


def already_done() -> set[str]:
    if not CACHE.exists():
        return set()
    done = set()
    with CACHE.open() as f:
        for line in f:
            try:
                done.add(json.loads(line)["gnaf_id"])
            except Exception:
                continue
    return done


class Limiter:
    """Simple token-bucket so we stay under the documented rate."""

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


async def fetch_one(client, gid: str, lim: Limiter, stats: dict) -> dict | None:
    url = f"{BASE}/v1/properties/{gid}/rent-history"
    for attempt in range(4):
        await lim.wait()
        try:
            r = await client.get(url)
        except Exception:
            await asyncio.sleep(0.5 * (attempt + 1) + random.random())
            continue

        if r.status_code == 200:
            body = r.json()
            if not body.get("data"):
                stats["empty"] += 1
                return {"gnaf_id": gid, "adverts": []}
            stats["ok"] += 1
            return {"gnaf_id": gid, "adverts": body["data"]}

        if r.status_code == 429:
            wait = float(r.headers.get("Retry-After", 2))
            stats["throttled"] += 1
            await asyncio.sleep(wait + random.random())
            continue

        if r.status_code == 403:          # outside coverage, never retry
            stats["outside"] += 1
            return {"gnaf_id": gid, "adverts": [], "outside": True}

        if r.status_code == 401:
            sys.exit("401 - key rejected. Check .env")

        stats["error"] += 1
        await asyncio.sleep(0.5 * (attempt + 1))
    stats["gaveup"] += 1
    return None


async def run(ids: list[str], key: str) -> None:
    try:
        import httpx
    except ImportError:
        sys.exit("pip install httpx")

    lim = Limiter(RATE)
    stats = dict(ok=0, empty=0, outside=0, throttled=0, error=0, gaveup=0)
    sem = asyncio.Semaphore(CONCURRENCY)
    t0 = time.time()
    written = 0

    headers = {"Authorization": f"Bearer {key}"}
    limits = httpx.Limits(max_connections=CONCURRENCY + 4)

    async with httpx.AsyncClient(headers=headers, timeout=25.0, limits=limits) as client:
        with CACHE.open("a") as sink:

            async def work(gid: str):
                nonlocal written
                async with sem:
                    rec = await fetch_one(client, gid, lim, stats)
                if rec is not None:
                    sink.write(json.dumps(rec) + "\n")
                    written += 1
                    if written % 500 == 0:
                        sink.flush()
                        rate = written / max(time.time() - t0, 1e-9)
                        left = (len(ids) - written) / max(rate, 1e-9) / 60
                        print(f"  {written:,}/{len(ids):,}  {rate:.1f}/s  "
                              f"~{left:.0f} min left  {stats}", flush=True)

            await asyncio.gather(*(work(g) for g in ids))

    print(f"\ndone in {(time.time()-t0)/60:.1f} min: {stats}")


def flatten() -> None:
    """Cache -> a flat advert table with the columns pipeline.py resolves."""
    if not CACHE.exists():
        print(f"No cache at {CACHE} yet - nothing to flatten.")
        return
    rows = []
    with CACHE.open() as f:
        for line in f:
            try:
                rec = json.loads(line)
            except Exception:
                continue
            for a in rec.get("adverts") or []:
                if not isinstance(a, dict):
                    continue
                date = next((a[k] for k in
                             ("advert_date", "date", "listed_date", "first_seen",
                              "date_listed", "start_date") if a.get(k)), None)
                rent = next((a[k] for k in
                             ("weekly_rent", "rent", "price", "rent_pw",
                              "asking_rent", "amount") if a.get(k) is not None), None)
                if date is None or rent is None:
                    continue
                rows.append({
                    "gnaf_id": rec["gnaf_id"],
                    "advert_date": str(date)[:10],
                    "weekly_rent": rent,
                    "bedrooms": a.get("bedrooms"),
                    "bathrooms": a.get("bathrooms"),
                    "car_space": a.get("car_space", a.get("parking")),
                })

    if not rows:
        print("No adverts parsed. Print one raw record and check its field names:")
        with CACHE.open() as f:
            for line in f:
                rec = json.loads(line)
                if rec.get("adverts"):
                    print(json.dumps(rec["adverts"][0], indent=2)[:900])
                    break
        return

    df = pd.DataFrame(rows)
    df["advert_date"] = pd.to_datetime(df.advert_date, errors="coerce")
    df["weekly_rent"] = pd.to_numeric(df.weekly_rent, errors="coerce")
    before = len(df)
    df = df.dropna(subset=["advert_date", "weekly_rent"])
    df = df[df.advert_date <= CUTOFF]                    # THE CUTOFF
    df = df.drop_duplicates(subset=["gnaf_id", "advert_date", "weekly_rent"])
    df["advert_date"] = df.advert_date.dt.strftime("%Y-%m-%d")
    df.to_csv(OUT_CSV, index=False)
    print(f"\n{OUT_CSV}: {before:,} raw -> {len(df):,} adverts "
          f"({df.gnaf_id.nunique():,} flats), {df.advert_date.min()} to {df.advert_date.max()}")
    print("\nTo use it instead of the supplied file:")
    print(f"  cp {OUT_CSV} data/rents-past-leases-city-of-sydney.csv")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="stop after N addresses (test first)")
    ap.add_argument("--only-advertised", action="store_true",
                    help="only flats with ever_advertised_for_lease == 1")
    ap.add_argument("--flatten-only", action="store_true", help="rebuild the CSV from cache")
    args = ap.parse_args()

    DATA.mkdir(exist_ok=True)
    if args.flatten_only:
        flatten()
        return

    inp_path = DATA / "rents-input-flats-city-of-sydney.csv"
    if not inp_path.exists():
        sys.exit(f"{inp_path} not found - the input file is required, it defines "
                 "which 65,777 gnaf_ids to predict and cannot be reconstructed.")

    inp = pd.read_csv(inp_path, dtype={"gnaf_id": str})
    if args.only_advertised and "ever_advertised_for_lease" in inp:
        inp = inp[inp.ever_advertised_for_lease == 1]

    ids = inp.gnaf_id.dropna().unique().tolist()
    done = already_done()
    todo = [g for g in ids if g not in done]
    if args.limit:
        todo = todo[: args.limit]

    print(f"{len(ids):,} candidates, {len(done):,} cached, fetching {len(todo):,}")
    print(f"at ~{RATE:.0f}/s that is about {len(todo)/RATE/60:.0f} minutes\n")

    if todo:
        asyncio.run(run(todo, load_key()))
    flatten()


if __name__ == "__main__":
    main()
