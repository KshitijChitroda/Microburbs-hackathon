"""
Builds fake versions of the three files Microburbs sends at 11am, using the
EXACT column names published on the task page, so the pipeline can be written
and debugged tonight.

At 11:01 tomorrow: drop the real files into data/ and delete nothing. The
pipeline reads data/ either way.

The past-leases schema is NOT published. We guess a reasonable shape here and
pipeline.py resolves columns by fuzzy matching, so a mismatch tomorrow is a
one-line fix in COLUMN_HINTS rather than a rewrite.
"""
import numpy as np
import pandas as pd
from pathlib import Path

RNG = np.random.default_rng(7)
DATA = Path("data")
DATA.mkdir(exist_ok=True)

N_FLATS = 65_777
SUBURBS = [
    "Sydney (NSW)", "Surry Hills (NSW)", "Waterloo (NSW)", "Redfern (NSW)",
    "Pyrmont (NSW)", "Ultimo (NSW)", "Chippendale (NSW)", "Darlinghurst (NSW)",
    "Potts Point (NSW)", "Zetland (NSW)", "Alexandria (NSW)", "Glebe (NSW)",
    "Newtown (NSW)", "Erskineville (NSW)", "Rosebery (NSW)", "Haymarket (NSW)",
    "Millers Point (NSW)", "Barangaroo (NSW)", "Woolloomooloo (NSW)",
    "Elizabeth Bay (NSW)", "Rushcutters Bay (NSW)", "Paddington (NSW)",
    "Camperdown (NSW)", "Forest Lodge (NSW)", "Annandale (NSW)",
    "Beaconsfield (NSW)", "Eveleigh (NSW)", "Dawes Point (NSW)",
    "The Rocks (NSW)", "Centennial Park (NSW)", "Moore Park (NSW)",
    "Rushcutters Bay East (NSW)", "St Peters (NSW)",
]


def make_input() -> pd.DataFrame:
    n = N_FLATS
    suburb = RNG.choice(SUBURBS, n)
    beds = RNG.choice([0, 1, 2, 3, 4], n, p=[0.07, 0.42, 0.37, 0.12, 0.02])
    baths = np.clip(RNG.poisson(1.1, n), 1, 4)
    ptype = RNG.choice(["apartment", "unit", "studio"], n, p=[0.72, 0.21, 0.07])

    df = pd.DataFrame({
        "gnaf_id": [f"GANSW{700_000_000 + i}" for i in range(n)],
        "address": [f"UNIT {RNG.integers(1, 900)}, {RNG.integers(1, 200)} FAKE ST" for _ in range(n)],
        "building_address": [f"{i % 4000} FAKE ST" for i in range(n)],
        "suburb": suburb,
        "mb": RNG.integers(11_000_000_000, 11_999_999_999, n),
        "latitude": -33.88 + RNG.normal(0, 0.02, n),
        "longitude": 151.20 + RNG.normal(0, 0.02, n),
        "property_type": ptype,
        "bedrooms": beds,
        "bathrooms": baths,
        "car_space": RNG.choice([1, 0, np.nan], n, p=[0.45, 0.46, 0.09]),
        "car_spaces": RNG.choice([1, 2, np.nan], n, p=[0.40, 0.08, 0.52]),
        "floor_level": np.where(RNG.random(n) < 0.52, RNG.integers(1, 30, n), np.nan),
        "flats_in_building": RNG.integers(4, 300, n),
        "block_new_homes_2021_2026": np.where(RNG.random(n) < 0.94, RNG.integers(0, 400, n), np.nan),
        "block_homes_2026": np.where(RNG.random(n) < 0.94, RNG.integers(20, 1200, n), np.nan),
        "ever_advertised_for_lease": RNG.choice([1, 0], n, p=[0.755, 0.245]),
        "first_ever_listed": pd.to_datetime("2012-01-01") + pd.to_timedelta(RNG.integers(0, 5000, n), "D"),
        "attrs_last_seen": pd.to_datetime("2020-01-01") + pd.to_timedelta(RNG.integers(0, 2400, n), "D"),
    })
    df["first_ever_listed"] = df["first_ever_listed"].dt.strftime("%Y-%m-%d")
    df["attrs_last_seen"] = df["attrs_last_seen"].dt.strftime("%Y-%m-%d")
    return df


def make_leases(inp: pd.DataFrame) -> pd.DataFrame:
    """Historic adverts. One row per advert; a flat can appear many times."""
    pool = inp[inp.ever_advertised_for_lease == 1]
    idx = RNG.choice(len(pool), 42_000, replace=True)
    s = pool.iloc[idx].reset_index(drop=True)

    date = pd.to_datetime("2024-06-01") + pd.to_timedelta(RNG.integers(0, 790, len(s)), "D")
    park = s.car_space.fillna(pd.Series(RNG.choice([0, 1], len(s)))).astype(float)

    base = (
        420
        + 230 * s.bedrooms.to_numpy()
        + 60 * (s.bathrooms.to_numpy() - 1)
        + 95 * park.to_numpy()
        + 4.5 * np.nan_to_num(s.floor_level.to_numpy(), nan=5)
        + s.suburb.map({v: k for k, v in enumerate(SUBURBS)}).to_numpy() * 6
        + (date - pd.Timestamp("2024-06-01")).days * 0.09   # drift
        + RNG.normal(0, 85, len(s))
    )

    return pd.DataFrame({
        "gnaf_id": s.gnaf_id.to_numpy(),
        "advert_date": date.strftime("%Y-%m-%d"),
        "weekly_rent": np.round(np.clip(base, 250, 4000) / 5) * 5,
        "bedrooms": s.bedrooms.to_numpy(),
        "bathrooms": s.bathrooms.to_numpy(),
        "car_space": park.to_numpy(),
        "suburb": s.suburb.to_numpy(),
        "property_type": s.property_type.to_numpy(),
    })


def make_blocks(inp: pd.DataFrame) -> pd.DataFrame:
    mb = inp.mb.drop_duplicates().to_frame()
    mb["homes_2021"] = RNG.integers(10, 900, len(mb))
    mb["homes_2026"] = mb.homes_2021 + RNG.integers(0, 300, len(mb))
    mb["people_2021"] = mb.homes_2021 * RNG.uniform(1.4, 2.6, len(mb))
    mb["people_2026"] = mb.homes_2026 * RNG.uniform(1.4, 2.6, len(mb))
    return mb.round(0)


if __name__ == "__main__":
    inp = make_input()
    inp.to_csv(DATA / "rents-input-flats-city-of-sydney.csv", index=False)
    make_leases(inp).to_csv(DATA / "rents-past-leases-city-of-sydney.csv", index=False)
    make_blocks(inp).to_csv(DATA / "blocks-homes-and-people-2021-2026.csv", index=False)
    print("wrote synthetic files to data/")
