"""
Look up any City of Sydney flat and see what it should rent for, with a car
space and without. Built for the demo video.

    python -m pip install streamlit
    python -m streamlit run app.py

Reads the submission CSV and the input file. Nothing leaves the machine and no
API key is needed.
"""
from pathlib import Path

import pandas as pd
import streamlit as st

OUT, DATA = Path("out"), Path("data")
INPUT = DATA / "rents-input-flats-city-of-sydney.csv"

st.set_page_config(page_title="What will this flat rent for?",
                   page_icon="🏠", layout="centered")


@st.cache_data
def load() -> pd.DataFrame:
    subs = sorted(OUT.glob("rents_*.csv"))
    if not subs:
        st.error("No submission found. Run pipeline.py first.")
        st.stop()
    sub = pd.read_csv(subs[0], dtype={"gnaf_id": str})

    want = ["gnaf_id", "address", "suburb", "bedrooms", "bathrooms",
            "property_type", "flats_in_building", "floor_level"]
    have = [c for c in want if c in pd.read_csv(INPUT, nrows=1).columns]
    inp = pd.read_csv(INPUT, usecols=have, dtype={"gnaf_id": str})

    df = sub.merge(inp, on="gnaf_id", how="left")
    df["premium"] = df.rent_with_car_space - df.rent_without_car_space
    if "suburb" in df.columns:
        df["suburb"] = df.suburb.astype(str).str.replace(" (NSW)", "", regex=False)
    return df


def flat_card(r: pd.Series) -> None:
    bits = []
    for col, fmt in (("bedrooms", "{} bed"), ("bathrooms", "{} bath"),
                     ("floor_level", "floor {}"), ("flats_in_building", "{} in building")):
        v = r.get(col)
        if pd.notna(v):
            bits.append(fmt.format(int(v)))
    with st.container(border=True):
        st.markdown(f"**{r.address}**  \n{r.get('suburb', '')} · {' · '.join(bits)}")
        a, b = st.columns(2)
        a.metric("With a car space", f"${r.rent_with_car_space:,.0f}/wk")
        b.metric("Without", f"${r.rent_without_car_space:,.0f}/wk",
                 f"-${r.premium:,.0f}", delta_color="inverse")


df = load()

st.title("What will this flat rent for?")
st.caption(f"{len(df):,} flats across the City of Sydney. Weekly asking rent, "
           "with a car space and without. Built on adverts dated no later than "
           "31 July 2026.")

tab_addr, tab_sub, tab_why = st.tabs(
    ["Look up a flat", "Compare suburbs", "Why the premium moves"])

# ------------------------------------------------------------------ look up
with tab_addr:
    q = st.text_input("Search an address",
                      placeholder="a street, a suburb or a postcode — e.g. DUNKERLEY, ALEXANDRIA, 2015")
    if not q:
        c1, c2, c3 = st.columns(3)
        c1.metric("Flats priced", f"{len(df):,}")
        c2.metric("Median rent", f"${df.rent_with_car_space.median():,.0f}/wk")
        c3.metric("Median car-space premium", f"${df.premium.median():,.0f}/wk")
        st.info("Type part of a street, suburb or postcode to price a specific flat.")
    else:
        mask = df.address.fillna("").str.contains(q, case=False, regex=False)
        hits = df[mask]
        if hits.empty:
            st.warning("No match. Try part of a street name, a suburb, or a postcode.")
        else:
            st.caption(f"Showing 8 of {len(hits):,} matches · "
                       f"median premium here ${hits.premium.median():,.0f}/wk")
            for _, r in hits.head(8).iterrows():
                flat_card(r)

# ------------------------------------------------------------------ suburbs
with tab_sub:
    if "suburb" not in df.columns:
        st.info("No suburb column in the input file.")
    else:
        subs = sorted(df.suburb.dropna().unique())
        default = subs.index("Darlinghurst") if "Darlinghurst" in subs else 0
        pick = st.selectbox("Suburb", subs, index=default)
        s = df[df.suburb == pick]

        a, b, c = st.columns(3)
        a.metric("Flats", f"{len(s):,}")
        b.metric("Median rent", f"${s.rent_with_car_space.median():,.0f}/wk")
        c.metric("Car-space premium", f"${s.premium.median():,.0f}/wk")

        if "bedrooms" in s.columns and s.bedrooms.notna().any():
            st.subheader(f"{pick}: median rent by bedroom count")
            by = (s.dropna(subset=["bedrooms"])
                   .groupby(s.bedrooms.astype("Int64"), observed=True)
                   .rent_with_car_space.median())
            by.index = [f"{i} bed" for i in by.index]
            st.bar_chart(by.rename("Median weekly rent"), height=220)

        st.subheader("The car-space premium, every suburb")
        allp = (df.groupby("suburb").premium.median()
                  .sort_values(ascending=False).rename("Premium ($/week)"))
        st.bar_chart(allp, height=320)
        st.caption("Median extra weekly rent for a flat with a car space. "
                   "It is not the same number everywhere.")

# ---------------------------------------------------------------- the finding
with tab_why:
    st.subheader("A car space is worth more in a small block")
    if "flats_in_building" in df.columns and df.flats_in_building.notna().any():
        band = pd.cut(df.flats_in_building, [0, 20, 60, 150, 10 ** 6],
                      labels=["under 20", "20–60", "60–150", "150+"])
        by = df.groupby(band, observed=True).premium.median().rename("Premium ($/week)")
        st.bar_chart(by, height=260)
        st.markdown(
            "Flats in the building, against the weekly car-space premium.\n\n"
            "**Towers are built with basement parking**, so almost every flat has a "
            "space and nobody pays extra for one. **Small walk-ups and terrace "
            "conversions mostly have none**, so the few spaces that exist are scarce "
            "and priced accordingly.\n\n"
            "The premium tracks supply inside the building, not parking on the street.")
    else:
        st.info("No flats_in_building column in the input file.")

st.caption("Estimates are automated and indicative only — not a valuation. "
           "Data © Microburbs, used under the hackathon event key.")
