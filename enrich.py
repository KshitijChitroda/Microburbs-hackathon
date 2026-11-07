"""
enrich.py — side track for Task 4. Owner: Kiet. Writes out/extras.csv.

One row per gnaf_id with:
    floor_est             inferred floor, blank if genuinely unknown
    floor_est_source      given / parsed / unknown
    car_space_prob        0..1, chance the flat has a car space
    block_homes_filled    homes in the mesh block (2026), gaps filled
    block_new_homes_filled  new homes 2021-2026 in the mesh block, gaps filled

Rules this script keeps by construction:
  * reads ONLY the columns in ALLOWED from the input file (usecols), plus the
    supplied blocks file joined on mb
  * never opens the past-leases file, never reads a date or a rent column
  * one row per gnaf_id, no duplicates; blanks allowed

Usage:
    python enrich.py                       # reads data/, writes out/extras.csv
    python enrich.py --input path/to/rents-input-flats-city-of-sydney.csv
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path("data")
OUT = Path("out")
INPUT_FILE = DATA / "rents-input-flats-city-of-sydney.csv"
BLOCKS_FILE = DATA / "blocks-homes-and-people-2021-2026.csv"

# Columns we are allowed to read from the input file. Anything not here is
# never loaded. Edit this list if the agreed list changes.
ALLOWED = [
    "gnaf_id", "address", "building_address", "suburb", "mb",
    "latitude", "longitude", "flats_in_building", "floor_level", "car_spaces",
]
# Optional: only read if present in the file AND agreed with the team.
OPTIONAL_ALLOWED = ["block_homes_2026", "block_new_homes_2021_2026"]

MAX_FLOOR = 90           # nothing in City of Sydney is taller than this
BLANK_CAR_IS_ZERO = False  # set by --blank-car-is-zero
SHRINK = 5.0             # pseudo-count for car_space_prob shrinkage
N_FOLDS = 5              # for the floor-parsing accuracy check


# ---------------------------------------------------------------------------
# LOAD
# ---------------------------------------------------------------------------
def load_input(path: Path) -> pd.DataFrame:
    header = pd.read_csv(path, nrows=0).columns
    cols = [c for c in ALLOWED + OPTIONAL_ALLOWED if c in header]
    missing = [c for c in ALLOWED if c not in header]
    if missing:
        print(f"[load] !! not in input file: {missing}")
    # guard: nothing that looks like a date or a rent is read
    assert not any(re.search(r"date|rent|listed|seen", c, re.I) for c in cols)
    df = pd.read_csv(path, usecols=cols, dtype={"gnaf_id": str, "mb": str})
    for c in ("latitude", "longitude", "flats_in_building", "floor_level", "car_spaces",
              *OPTIONAL_ALLOWED):
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    n0 = len(df)
    df = df.drop_duplicates("gnaf_id").reset_index(drop=True)
    print(f"[load] {n0:,} rows, {len(df):,} unique gnaf_id, columns read: {cols}")
    return df


# ---------------------------------------------------------------------------
# FLOOR
# ---------------------------------------------------------------------------
LEVEL_RE = re.compile(r"\b(?:LEVEL|LVL|FLOOR|FLR|L)\s*(\d{1,2})\b", re.I)
GROUND_RE = re.compile(r"\b(?:GROUND|GRD|G)\s*(?:FLOOR|FLR|LEVEL)?\b(?=[ ,/])", re.I)
# unit number: "UNIT 1205,", "APARTMENT 12A", "FLAT 3", "SUITE 401", "1205/88 X ST"
UNIT_RE = re.compile(
    r"\b(?:UNIT|U|APARTMENT|APT|FLAT|SUITE|SHOP|PENTHOUSE|PH|VILLA|STUDIO)\s*([A-Z]?)(\d{1,5})[A-Z]?\b",
    re.I)
SLASH_RE = re.compile(r"^\s*([A-Z]?)(\d{1,5})[A-Z]?\s*/\s*\d", re.I)


def parse_address(addr: str) -> tuple[float, float]:
    """Return (explicit_level, unit_number) — NaN where absent."""
    if not isinstance(addr, str):
        return np.nan, np.nan
    m = LEVEL_RE.search(addr)
    level = float(m.group(1)) if m else np.nan
    if np.isnan(level) and re.search(r"\bGROUND\b", addr, re.I):
        level = 0.0
    m = UNIT_RE.search(addr) or SLASH_RE.search(addr)
    unit = float(m.group(2)) if m else np.nan
    return level, unit


# A "rule" maps a unit number to a floor: floor = unit // 10**k + offset.
# k=2 is the common Sydney pattern (1205 -> 12). Some buildings number
# 101 as ground (offset -1) or use 3-digit floors in towers (k=3 is rare).
RULES = [(k, off) for k in (2, 1, 3) for off in (0, -1, 1)]


def apply_rule(unit: np.ndarray, k: int, off: int) -> np.ndarray:
    f = np.floor(unit / 10 ** k) + off
    ok = (unit >= 10 ** k) & (f >= 0) & (f <= MAX_FLOOR)
    return np.where(ok, f, np.nan)


def fit_building_rules(known: pd.DataFrame, min_rows: int = 3, min_share: float = 0.6) -> dict:
    """Per building, pick the rule that best reproduces known floor_level."""
    rules = {}
    for b, g in known.groupby("building_address", sort=False):
        if len(g) < min_rows:
            continue
        best, best_share = None, 0.0
        for k, off in RULES:
            pred = apply_rule(g.unit.to_numpy(), k, off)
            share = np.mean(pred == g.floor_level.to_numpy())
            if share > best_share:
                best, best_share = (k, off), share
        if best and best_share >= min_share:
            rules[b] = best
    return rules


def fit_global_rule(known: pd.DataFrame) -> tuple[int, int]:
    """Best rule over all known rows with a 3+ digit unit number."""
    g = known[known.unit >= 100]
    if g.empty:
        return 2, 0
    scores = {r: np.mean(apply_rule(g.unit.to_numpy(), *r) == g.floor_level.to_numpy())
              for r in RULES if r[0] == 2}
    return max(scores, key=scores.get)


def predict_floor(df: pd.DataFrame, b_rules: dict, g_rule: tuple[int, int]) -> np.ndarray:
    unit = df.unit.to_numpy()
    out = df.level.to_numpy().copy()                     # 1. explicit "LEVEL n"
    # 2. building-specific rule
    rk = df.building_address.map(lambda b: b_rules.get(b, (np.nan, np.nan)))
    ks = np.array([r[0] for r in rk], dtype=float)
    offs = np.array([r[1] for r in rk], dtype=float)
    for k, off in RULES:
        ix = (ks == k) & (offs == off)
        if ix.any():
            fill = apply_rule(unit[ix], k, off)
            out[ix] = np.where(np.isnan(out[ix]), fill, out[ix])
    # 3. global rule, only for 3+ digit unit numbers (1-2 digits is ambiguous)
    g = apply_rule(unit, *g_rule)
    g = np.where(unit >= 100, g, np.nan)
    out = np.where(np.isnan(out), g, out)
    # sanity: a floor cannot exceed the number of flats in the building
    if "flats_in_building" in df:
        cap = df.flats_in_building.to_numpy()
        out = np.where(np.isfinite(cap) & (out > cap), np.nan, out)
    return out


def floor_features(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    parsed = df.address.map(parse_address)
    df = df.assign(level=[p[0] for p in parsed], unit=[p[1] for p in parsed])
    has_floor = "floor_level" in df
    known = df[df.floor_level.notna() & df.unit.notna()] if has_floor else df.iloc[:0]

    # --- accuracy: K-fold, labels of the scored rows are hidden from the rules
    acc = {}
    if len(known) >= 50:
        rng = np.random.default_rng(0)
        fold = rng.integers(0, N_FOLDS, len(known))
        preds = np.full(len(known), np.nan)
        for f in range(N_FOLDS):
            tr, te = known[fold != f], known[fold == f]
            p = predict_floor(te, fit_building_rules(tr), fit_global_rule(tr))
            preds[fold == f] = p
        truth = known.floor_level.to_numpy()
        hit = np.isfinite(preds)
        acc = {
            "rows_with_known_floor_and_unit": int(len(known)),
            "coverage_pct": round(100 * hit.mean(), 1),
            "exact_pct_of_covered": round(100 * np.mean(preds[hit] == truth[hit]), 1) if hit.any() else None,
            "within_1_pct_of_covered": round(100 * np.mean(np.abs(preds[hit] - truth[hit]) <= 1), 1) if hit.any() else None,
            "median_abs_floor_error": float(np.median(np.abs(preds[hit] - truth[hit]))) if hit.any() else None,
            "method": f"{N_FOLDS}-fold: rules fitted on 80% of known floors, scored on the hidden 20%",
        }
        print(f"[floor] accuracy on known floors: {acc}")

    # --- final fit on every known row
    b_rules, g_rule = fit_building_rules(known), fit_global_rule(known)
    est = predict_floor(df, b_rules, g_rule)
    print(f"[floor] building rules: {len(b_rules):,}; global rule k={g_rule[0]} offset={g_rule[1]}")

    given = df.floor_level.notna().to_numpy() if has_floor else np.zeros(len(df), bool)
    floor_est = np.where(given, df.floor_level if has_floor else np.nan, est)
    source = np.where(given, "given", np.where(np.isfinite(est), "parsed", "unknown"))
    df["floor_est"] = floor_est
    df["floor_est_source"] = source
    return df, acc


# ---------------------------------------------------------------------------
# CAR SPACE PROBABILITY  (hierarchical shrinkage: global > suburb > mb > building)
# ---------------------------------------------------------------------------
def car_space_prob(df: pd.DataFrame) -> np.ndarray:
    if "car_spaces" not in df:
        return np.full(len(df), np.nan)
    has = (df.car_spaces > 0).astype(float).where(df.car_spaces.notna())
    vc = df.car_spaces.value_counts(dropna=False).head(6).to_dict()
    print(f"[car] car_spaces values: {vc}")
    if not (df.car_spaces == 0).any():
        print("[car] !! car_spaces is never 0 - blank may mean 'no car space'. "
              "If so, run with --blank-car-is-zero")
        if BLANK_CAR_IS_ZERO:
            has = (df.car_spaces.fillna(0) > 0).astype(float)
    prior = pd.Series(has.mean(), index=df.index)
    for level in ("suburb", "mb", "building_address"):
        if level not in df:
            continue
        grp = has.groupby(df[level])
        s, n = df[level].map(grp.sum()), df[level].map(grp.count())
        prior = ((s.fillna(0) + SHRINK * prior) / (n.fillna(0) + SHRINK))
    # known rows: use the fact, unknown rows: use the shrunk group estimate
    p = has.where(has.notna(), prior)
    print(f"[car] known for {has.notna().mean()*100:.1f}% of flats; "
          f"mean prob {p.mean():.3f}")
    return p.clip(0, 1).round(4).to_numpy()


# ---------------------------------------------------------------------------
# BLOCK HOMES  (from the supplied blocks file on mb, gaps filled spatially)
# ---------------------------------------------------------------------------
def find_col(cols, *needles) -> str | None:
    for c in cols:
        low = c.lower()
        if all(n in low for n in needles):
            return c
    return None


def block_features(df: pd.DataFrame, blocks_path: Path, k: int = 8) -> pd.DataFrame:
    homes = df.get("block_homes_2026", pd.Series(np.nan, index=df.index)).copy()
    new = df.get("block_new_homes_2021_2026", pd.Series(np.nan, index=df.index)).copy()

    if blocks_path.exists():
        b = pd.read_csv(blocks_path, dtype={"mb": str})
        h26 = find_col(b.columns, "home", "2026") or find_col(b.columns, "dwell", "2026")
        h21 = find_col(b.columns, "home", "2021") or find_col(b.columns, "dwell", "2021")
        nw = find_col(b.columns, "new")
        print(f"[blocks] using columns homes_2026={h26} homes_2021={h21} new={nw}")
        b = b.drop_duplicates("mb").set_index("mb")
        if h26:
            homes = homes.fillna(df.mb.map(pd.to_numeric(b[h26], errors="coerce")))
        if nw:
            new = new.fillna(df.mb.map(pd.to_numeric(b[nw], errors="coerce")))
        elif h26 and h21:
            d = (pd.to_numeric(b[h26], errors="coerce") - pd.to_numeric(b[h21], errors="coerce")).clip(lower=0)
            new = new.fillna(df.mb.map(d))
    else:
        print(f"[blocks] !! {blocks_path} not found, spatial fill only")

    # remaining gaps: median of the k nearest flats (by lat/lon) that have a value
    if {"latitude", "longitude"} <= set(df.columns):
        try:
            from sklearn.neighbors import BallTree
        except ImportError:
            BallTree = None
        xy = np.radians(df[["latitude", "longitude"]].to_numpy())
        okxy = np.isfinite(xy).all(1)
        for name, s in (("homes", homes), ("new", new)):
            src = s.notna().to_numpy() & okxy
            dst = s.isna().to_numpy() & okxy
            if BallTree is None or not src.any() or not dst.any():
                continue
            tree = BallTree(xy[src], metric="haversine")
            _, ix = tree.query(xy[dst], k=min(k, src.sum()))
            vals = s.to_numpy()[src][ix]
            s.iloc[np.where(dst)[0]] = np.median(vals, axis=1)
            print(f"[blocks] {name}: filled {dst.sum():,} gaps from {k} nearest neighbours")

    df["block_homes_filled"] = homes.round(0)
    df["block_new_homes_filled"] = new.round(0)
    return df


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=INPUT_FILE)
    ap.add_argument("--blocks", type=Path, default=BLOCKS_FILE)
    ap.add_argument("--out", type=Path, default=OUT / "extras.csv")
    ap.add_argument("--blank-car-is-zero", action="store_true",
                    help="treat blank car_spaces as 0 (only if car_spaces is never 0)")
    args = ap.parse_args()
    global BLANK_CAR_IS_ZERO
    BLANK_CAR_IS_ZERO = args.blank_car_is_zero
    args.out.parent.mkdir(parents=True, exist_ok=True)

    df = load_input(args.input)
    df, acc = floor_features(df)
    df["car_space_prob"] = car_space_prob(df)
    df = block_features(df, args.blocks)

    out = df[["gnaf_id", "floor_est", "floor_est_source", "car_space_prob",
              "block_homes_filled", "block_new_homes_filled"]]
    assert out.gnaf_id.is_unique, "duplicate gnaf_id"
    out.to_csv(args.out, index=False)

    summary = {
        "rows": len(out),
        "floor_est_source": out.floor_est_source.value_counts().to_dict(),
        "floor_parsing_accuracy": acc,
        "car_space_prob_filled_pct": round(100 * out.car_space_prob.notna().mean(), 1),
        "block_homes_filled_pct": round(100 * out.block_homes_filled.notna().mean(), 1),
        "block_new_homes_filled_pct": round(100 * out.block_new_homes_filled.notna().mean(), 1),
    }
    (args.out.parent / "extras_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(f"[done] {args.out}  rows={len(out):,}")
    print(json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    main()
