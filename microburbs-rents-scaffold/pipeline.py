"""
Microburbs Hackathon — Task 4: what will this flat rent for?

Scored on MEDIAN ABSOLUTE DOLLAR ERROR over 1,699 frozen adverts dated
1 Aug - 30 Sep 2026. Tie-break: share within $100. Blank answers score worse
than wrong ones.

Design decisions that follow from that:
  * objective = regression_l1 (MAE) -> its optimum IS the conditional median,
    which is the thing being scored. Default L2 optimises the mean. This is
    the single biggest edge available.
    (quantile alpha=0.5 is equivalent but LightGBM forbids monotone
    constraints with it, so L1 is the one to use.)
  * LightGBM forbids monotone constraints under any L1-type objective, so the
    parking premium is enforced AFTER prediction: predict the flat twice
    (parking on, parking off), then clip the gap into a sane band anchored on
    the empirical premium measured in the training adverts. This guarantees
    rent_with_car_space >= rent_without_car_space on every row.
  * every prediction passes through a fallback chain, so no cell is ever blank.
  * NOTHING dated after 2026-07-31 is used. The cutoff is enforced in one
    place and asserted before training.

Usage:
    python make_synthetic.py        # tonight only
    python pipeline.py --team jack --baseline-only
    python pipeline.py --team jack --trials 40
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# CONSTANTS
# ----------------------------------------------------------------------------
CUTOFF = pd.Timestamp("2026-07-31")      # nothing after this, from anywhere
VAL_START = pd.Timestamp("2026-05-01")   # our own held-out window, ends CUTOFF
DATA = Path("data")
OUT = Path("out")
OUT.mkdir(exist_ok=True)

INPUT_FILE = DATA / "rents-input-flats-city-of-sydney.csv"
LEASES_FILE = DATA / "rents-past-leases-city-of-sydney.csv"
BLOCKS_FILE = DATA / "blocks-homes-and-people-2021-2026.csv"

N_EXPECTED_ROWS = 65_777

# The leases schema is not published. Resolve by fuzzy match, print what was
# found, and override here in one line tomorrow if it guesses wrong.
COLUMN_HINTS = {
    "gnaf_id":  ["gnaf_id", "gnaf", "address_id", "id"],
    "date":     ["advert_date", "date", "listed", "listing_date", "first_seen", "date_listed"],
    "rent":     ["weekly_rent", "rent", "price", "asking_rent", "rent_pw", "weekly"],
    "bedrooms": ["bedrooms", "beds", "bedroom"],
    "bathrooms": ["bathrooms", "baths", "bathroom"],
    "car_space": ["car_space", "parking", "car", "carspace", "car_spaces"],
}


def resolve(df: pd.DataFrame, hints: dict) -> dict:
    """Map canonical name -> actual column, by exact then substring match."""
    cols = {c.lower().strip(): c for c in df.columns}
    found = {}
    for want, cands in hints.items():
        hit = next((cols[c] for c in cands if c in cols), None)
        if hit is None:
            hit = next((orig for low, orig in cols.items()
                        if any(c in low for c in cands)), None)
        if hit:
            found[want] = hit
    missing = set(hints) - set(found)
    print(f"[schema] matched: {found}")
    if missing:
        print(f"[schema] !! NOT FOUND: {sorted(missing)}  -> columns are: {list(df.columns)}")
    return found


# ----------------------------------------------------------------------------
# LOAD
# ----------------------------------------------------------------------------
def load_input() -> pd.DataFrame:
    df = pd.read_csv(INPUT_FILE, dtype={"gnaf_id": str, "mb": str})
    print(f"[load] input rows: {len(df):,}")
    if len(df) != N_EXPECTED_ROWS:
        print(f"[load] !! expected {N_EXPECTED_ROWS:,} rows")
    for c in ("first_ever_listed", "attrs_last_seen"):
        if c in df:
            df[c] = pd.to_datetime(df[c], errors="coerce")
            df.loc[df[c] < pd.Timestamp("1900-01-01"), c] = pd.NaT  # 0001-01-01 sentinel
    return df


def load_leases() -> tuple[pd.DataFrame, dict]:
    raw = pd.read_csv(LEASES_FILE, dtype=str)
    m = resolve(raw, COLUMN_HINTS)
    df = pd.DataFrame({
        "gnaf_id": raw[m["gnaf_id"]].astype(str),
        "date": pd.to_datetime(raw[m["date"]], errors="coerce"),
        "rent": pd.to_numeric(raw[m["rent"]], errors="coerce"),
    })
    for extra in ("bedrooms", "bathrooms", "car_space"):
        if extra in m:
            df[f"adv_{extra}"] = pd.to_numeric(raw[m[extra]], errors="coerce")

    before = len(df)
    df = df.dropna(subset=["date", "rent"])
    df = df[(df.rent > 100) & (df.rent < 6000)]

    # THE CUTOFF. Enforced once, asserted below.
    df = df[df.date <= CUTOFF]
    assert df.date.max() <= CUTOFF, "leakage: advert after 2026-07-31"
    print(f"[load] adverts {before:,} -> {len(df):,} usable, "
          f"{df.date.min().date()} to {df.date.max().date()}")
    return df, m


def load_blocks() -> pd.DataFrame | None:
    if not BLOCKS_FILE.exists():
        return None
    b = pd.read_csv(BLOCKS_FILE, dtype={"mb": str})
    print(f"[load] blocks: {len(b):,}")
    return b


# ----------------------------------------------------------------------------
# FEATURES
# ----------------------------------------------------------------------------
CAT = ["suburb", "property_type"]


def build_training_frame(leases: pd.DataFrame, inp: pd.DataFrame,
                         blocks: pd.DataFrame | None) -> pd.DataFrame:
    df = leases.merge(inp, on="gnaf_id", how="left", suffixes=("", "_inp"))

    # advert-level attributes beat the input snapshot where present
    for c in ("bedrooms", "bathrooms", "car_space"):
        a = f"adv_{c}"
        if a in df:
            df[c] = df[a].where(df[a].notna(), df[c])

    if blocks is not None and "mb" in df:
        keep = [c for c in blocks.columns if c != "mb"]
        df = df.merge(blocks[["mb"] + keep], on="mb", how="left", suffixes=("", "_blk"))

    return add_features(df, ref_date_col="date")


def add_features(df: pd.DataFrame, ref_date_col: str | None) -> pd.DataFrame:
    df = df.copy()
    ref = df[ref_date_col] if ref_date_col else pd.Series(CUTOFF, index=df.index)

    if "first_ever_listed" in df:
        df["days_since_first_listed"] = (ref - df.first_ever_listed).dt.days
    if "attrs_last_seen" in df:
        df["attrs_age_days"] = (ref - df.attrs_last_seen).dt.days

    # time index: lets the model learn market drift; frozen at CUTOFF when predicting
    df["t"] = (ref - pd.Timestamp("2024-01-01")).dt.days
    df["month"] = ref.dt.month

    for c in ("bedrooms", "bathrooms", "car_space", "car_spaces", "floor_level",
              "flats_in_building", "block_new_homes_2021_2026", "block_homes_2026",
              "latitude", "longitude", "ever_advertised_for_lease"):
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if {"floor_level", "flats_in_building"} <= set(df.columns):
        df["floor_ratio"] = df.floor_level / df.flats_in_building.clip(lower=1)
    if {"bedrooms", "bathrooms"} <= set(df.columns):
        df["bath_per_bed"] = df.bathrooms / df.bedrooms.clip(lower=1)
    if "flats_in_building" in df:
        df["log_building_size"] = np.log1p(df.flats_in_building)

    for c in CAT:
        if c in df:
            df[c] = df[c].astype("category")
    return df


FEATURES = [
    "bedrooms", "bathrooms", "car_space", "car_spaces", "floor_level",
    "flats_in_building", "log_building_size", "floor_ratio", "bath_per_bed",
    "block_new_homes_2021_2026", "block_homes_2026", "latitude", "longitude",
    "days_since_first_listed", "attrs_age_days", "t", "month",
    "suburb", "property_type",
]


def feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in FEATURES if c in df.columns]


# ----------------------------------------------------------------------------
# METRICS  (median absolute error is the ranked metric)
# ----------------------------------------------------------------------------
def score(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    err = np.abs(np.asarray(y_true) - np.asarray(y_pred))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return {
        "median_abs_error": round(float(np.median(err)), 2),
        "pct_within_100": round(float((err <= 100).mean() * 100), 2),
        "mean_abs_error": round(float(err.mean()), 2),
        "r_squared": round(1 - ss_res / ss_tot, 4) if ss_tot else None,
        "n": int(len(err)),
    }


# ----------------------------------------------------------------------------
# BASELINE  — group medians, no API, no model. Runs in seconds.
# This is what gets written at 11:30 so a submittable CSV always exists.
# ----------------------------------------------------------------------------
GROUPS = [
    ["suburb", "bedrooms", "car_space"],
    ["suburb", "bedrooms"],
    ["bedrooms", "car_space"],
    ["bedrooms"],
]
MIN_GROUP = 10


def baseline_tables(train: pd.DataFrame) -> tuple[list, float]:
    tables = []
    for keys in GROUPS:
        if not set(keys) <= set(train.columns):
            continue
        g = train.groupby(keys, observed=True).rent.agg(["median", "size"])
        tables.append((keys, g[g["size"] >= MIN_GROUP]["median"]))
    return tables, float(train.rent.median())


def baseline_predict(df: pd.DataFrame, tables: list, globalmed: float) -> np.ndarray:
    out = pd.Series(np.nan, index=df.index)
    for keys, tab in tables:
        if out.isna().sum() == 0:
            break
        idx = pd.MultiIndex.from_frame(df[keys]) if len(keys) > 1 else pd.Index(df[keys[0]])
        vals = pd.Series(tab.reindex(idx).to_numpy(), index=df.index)
        out = out.fillna(vals)
    return out.fillna(globalmed).to_numpy()


# ----------------------------------------------------------------------------
# MODEL
# ----------------------------------------------------------------------------
def train_model(train: pd.DataFrame, valid: pd.DataFrame, trials: int = 0):
    import lightgbm as lgb

    feats = feature_cols(train)

    base = dict(
        objective="regression_l1", metric="l1",
        learning_rate=0.05, num_leaves=63, min_data_in_leaf=40,
        feature_fraction=0.85, bagging_fraction=0.85, bagging_freq=1,
        lambda_l2=1.0, verbosity=-1, n_jobs=-1,
    )

    dtr = lgb.Dataset(train[feats], train.rent, categorical_feature=[c for c in CAT if c in feats])
    dva = lgb.Dataset(valid[feats], valid.rent, reference=dtr)

    def fit(params):
        return lgb.train(params, dtr, num_boost_round=3000, valid_sets=[dva],
                         callbacks=[lgb.early_stopping(100, verbose=False)])

    if trials > 0:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(t):
            p = dict(base)
            p.update(
                learning_rate=t.suggest_float("learning_rate", 0.02, 0.12, log=True),
                num_leaves=t.suggest_int("num_leaves", 31, 255),
                min_data_in_leaf=t.suggest_int("min_data_in_leaf", 20, 200),
                feature_fraction=t.suggest_float("feature_fraction", 0.6, 1.0),
                bagging_fraction=t.suggest_float("bagging_fraction", 0.6, 1.0),
                lambda_l2=t.suggest_float("lambda_l2", 0.1, 20.0, log=True),
            )
            m = fit(p)
            # tune on the ranked metric, not on the loss
            return score(valid.rent.to_numpy(), m.predict(valid[feats]))["median_abs_error"]

        st = optuna.create_study(direction="minimize")
        st.optimize(objective, n_trials=trials, show_progress_bar=False)
        print(f"[tune] best median abs error: ${st.best_value:.2f}  params={st.best_params}")
        base.update(st.best_params)

    model = fit(base)
    return model, feats


def empirical_premium(train: pd.DataFrame) -> float:
    """Median parking gap within suburb x bedrooms groups that have both kinds."""
    if "car_space" not in train:
        return 90.0
    t = train.dropna(subset=["car_space"])
    g = (t.groupby(["suburb", "bedrooms", "car_space"], observed=True)
           .rent.agg(["median", "size"]))
    g = g[g["size"] >= MIN_GROUP]["median"].unstack("car_space")
    if g.shape[1] < 2:
        return 90.0
    gaps = (g[1.0] - g[0.0]).dropna()
    val = float(gaps.median()) if len(gaps) else 90.0
    print(f"[premium] empirical median parking gap: ${val:.2f} over {len(gaps)} groups")
    return val


def predict_both(model, feats, frame: pd.DataFrame,
                 emp: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Same flat, parking on and off. LightGBM cannot enforce monotonicity under
    L1, so the gap is clipped into a band around the empirical premium. This
    makes rent_with >= rent_without true by construction.
    """
    a, b = frame.copy(), frame.copy()
    a["car_space"], b["car_space"] = 1.0, 0.0
    with_car = model.predict(a[feats])
    raw = with_car - model.predict(b[feats])

    lo, hi = max(10.0, 0.4 * emp), max(40.0, 2.5 * emp)
    premium = np.clip(np.nan_to_num(raw, nan=emp), lo, hi)
    print(f"[premium] applied band ${lo:.0f}-${hi:.0f}; "
          f"{(raw < lo).mean()*100:.1f}% floored, {(raw > hi).mean()*100:.1f}% capped")
    return with_car, with_car - premium


# ----------------------------------------------------------------------------
# OUTPUT
# ----------------------------------------------------------------------------
def round_half_dollar(x: np.ndarray) -> np.ndarray:
    return np.round(np.asarray(x, dtype=float) * 2) / 2


def write_submission(inp: pd.DataFrame, with_car, without, team: str) -> Path:
    sub = pd.DataFrame({
        "gnaf_id": inp.gnaf_id.to_numpy(),
        "rent_with_car_space": round_half_dollar(with_car),
        "rent_without_car_space": round_half_dollar(without),
    })

    # NO BLANKS, EVER. A blank scores worse than a wrong number.
    for c in ("rent_with_car_space", "rent_without_car_space"):
        bad = ~np.isfinite(sub[c])
        if bad.any():
            print(f"[write] !! filling {bad.sum()} non-finite in {c}")
            sub.loc[bad, c] = float(np.nanmedian(sub[c]))
        sub[c] = sub[c].clip(150, 6000)

    assert len(sub) == len(inp), "row count changed"
    assert sub.gnaf_id.is_unique, "duplicate gnaf_id"
    assert sub.notna().all().all(), "blank cell in submission"
    assert (sub.rent_with_car_space >= sub.rent_without_car_space).all(), "parking premium negative"

    slug = team.lower().replace(" ", "-")
    slug = "".join(ch for ch in slug if ch.isalnum() or ch == "-")
    path = OUT / f"rents_{slug}.csv"
    sub.to_csv(path, index=False)
    print(f"[write] {path}  rows={len(sub):,}  "
          f"median premium=${(sub.rent_with_car_space - sub.rent_without_car_space).median():.2f}")
    return path


# ----------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--team", required=True)
    ap.add_argument("--trials", type=int, default=0, help="Optuna trials; 0 = skip")
    ap.add_argument("--baseline-only", action="store_true")
    args = ap.parse_args()

    inp = load_input()
    leases, _ = load_leases()
    blocks = load_blocks()

    full = build_training_frame(leases, inp, blocks)
    train = full[full.date < VAL_START]
    valid = full[(full.date >= VAL_START) & (full.date <= CUTOFF)]
    print(f"[split] train {len(train):,} (to {VAL_START.date() - pd.Timedelta(days=1)})  "
          f"valid {len(valid):,} ({VAL_START.date()} to {CUTOFF.date()})")
    assert len(valid) > 200, "validation window too small - widen VAL_START"

    results = {}

    # --- baseline ------------------------------------------------------------
    tables, gmed = baseline_tables(train)
    results["baseline_group_median"] = score(valid.rent.to_numpy(), baseline_predict(valid, tables, gmed))
    print(f"[baseline] {results['baseline_group_median']}")

    # score frame = one row per flat, pretending today is the cutoff
    score_frame = add_features(
        inp.merge(blocks, on="mb", how="left") if blocks is not None and "mb" in inp else inp,
        ref_date_col=None,
    )

    if args.baseline_only:
        bw = baseline_predict(score_frame.assign(car_space=1.0), tables, gmed)
        bo = baseline_predict(score_frame.assign(car_space=0.0), tables, gmed)
        bo = np.minimum(bo, bw)
        write_submission(inp, bw, bo, args.team)
    else:
        model, feats = train_model(train, valid, args.trials)
        results["model"] = score(valid.rent.to_numpy(), model.predict(valid[feats]))
        print(f"[model] {results['model']}")

        imp = sorted(zip(feats, model.feature_importance("gain")), key=lambda x: -x[1])
        results["top_features"] = [{"feature": f, "gain": round(float(g))} for f, g in imp[:15]]

        w, o = predict_both(model, feats, score_frame, empirical_premium(train))
        write_submission(inp, w, o, args.team)
        results["parking_premium_median"] = round(float(np.median(w - o)), 2)

    results["benchmarks_published_by_microburbs"] = {
        "microburbs_rental_valuation": {"median_abs_error": 86, "pct_within_100": 55.21},
        "suburb_median": {"pct_within_100": 56.39},
        "their_simple_estimate": {"median_abs_error": 100, "with_parking": 93},
    }
    results["validation_window"] = f"{VAL_START.date()} to {CUTOFF.date()}"
    results["cutoff"] = str(CUTOFF.date())

    (OUT / "metrics.json").write_text(json.dumps(results, indent=2))
    print(f"[done] out/metrics.json")


if __name__ == "__main__":
    main()
