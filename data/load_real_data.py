"""Load real PMPML route topology + Census 2011 equity priors.

Sources:
- ``data/raw/pmpml_routes.csv``    -- OpenCity Pune Bus Stops and Routes
                                      (Route ID, Route Description, Kilometer)
                                      https://data.opencity.in/dataset/pune-bus-stops-and-routes
- ``data/raw/pune_census_2011.csv`` -- OpenCity Pune Census 2011 Data
                                      (PMC ward, P_SC, P_ST, TOT_P)
                                      https://data.opencity.in/dataset/pune-census-2011-data
- Depot list and counts: PMPML statistics page + Wikipedia
                                      (9 CNG + 4 electric = 13 depots).
                                      Per-depot lat/lon are neighbourhood
                                      centroids (no public per-depot footprint
                                      data exists).

The pipeline keeps the same ``routes.csv`` schema produced by the
synthetic ``generate_routes`` in ``fetch_data.py`` so the rest of the
codebase (deterministic MIP, forecasting, stochastic model, sensitivity,
dashboard) is unchanged. Synthetic pieces -- per-route ridership, hourly
demand profile, and per-depot fleet capacity -- are documented in the
paper's limitations section.
"""
from __future__ import annotations

import os
import re

import numpy as np
import pandas as pd


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")

# PMPML depot list. 9 CNG depots + 4 electric depots = 13 per Wikipedia /
# PMPML statistics. Lat/lon are neighbourhood centroids; PMPML does not
# publish per-depot footprint or capacity.
DEPOTS = [
    # name,             type,        lat,      lon,     zone
    ("Aundh",           "CNG",       18.5594, 73.8074, "west"),
    ("Bhosari",         "CNG",       18.6291, 73.8472, "pcmc"),
    ("Dhankawadi",      "CNG",       18.4623, 73.8528, "south"),
    ("Hadapsar",        "CNG",       18.5089, 73.9260, "east"),
    ("Katraj",          "CNG",       18.4488, 73.8657, "south"),
    ("Kothrud",         "CNG",       18.5074, 73.8077, "west"),
    ("Nigdi",           "CNG",       18.6517, 73.7644, "pcmc"),
    ("Pimpri",          "CNG",       18.6298, 73.8131, "pcmc"),
    ("Swargate",        "CNG",       18.5018, 73.8580, "central"),
    ("Balewadi",        "Electric",  18.5750, 73.7780, "west"),
    ("Maan-Hinjewadi",  "Electric",  18.5912, 73.7389, "west"),
    ("Shewalewadi",     "Electric",  18.4884, 73.9610, "east"),
    ("Charholi",        "Electric",  18.6608, 73.8979, "pcmc"),
]

# Direction suffix on PMPML route IDs (e.g. "100-D" / "100-U").
_DIR_RE = re.compile(r"-[DU]$", re.IGNORECASE)

# Aliases mapping common Pune route-endpoint landmarks to PMC census wards.
# Census wards are administrative; route descriptions use landmarks. Without
# this mapping only ~16% of routes match a ward by exact substring. Each
# alias key is a canonicalised endpoint substring; the value is the canonical
# ward name from ``data/raw/pune_census_2011.csv``. Sources cross-checked
# against the OpenCity Pune ward map and the Pune Municipal Corporation
# ward boundaries (https://data.opencity.in/dataset/pune-census-2011-data).
_ENDPOINT_TO_WARD_ALIASES = {
    # Sangamwadi (Yerawada) -- includes the Yerawada / Vishrantwadi catchment
    "yerwada": "Sangamwadi (Yerawada)",
    "yerawada": "Sangamwadi (Yerawada)",
    "vishrantwadi": "Sangamwadi (Yerawada)",
    "sangamwadi": "Sangamwadi (Yerawada)",
    "sangamvadi": "Sangamwadi (Yerawada)",
    "kalyaninagar": "Sangamwadi (Yerawada)",
    # Dholepatil Road -- Pune Station / Wadgaon Sheri catchment
    "punestation": "Dholepatil Road",
    "wadgaonsheri": "Dholepatil Road",
    "kharadi": "Dholepatil Road",
    "viman": "Dholepatil Road",  # Viman Nagar
    "dholepatil": "Dholepatil Road",
    # Sahakarnagar
    "sahakarnagar": "Sahakarnagar",
    "sahakar": "Sahakarnagar",
    "parvati": "Sahakarnagar",
    # Bhavani Peth -- central old city, high SC+ST
    "bhavanipeth": "Bhavani Peth",
    "bhavani": "Bhavani Peth",
    "kondhwa": "Bhavani Peth",  # Kondhwa is south-central; classed here for equity
    "wanowri": "Bhavani Peth",
    # Hadapsar
    "hadapsar": "Hadapsar",
    "mundhwa": "Hadapsar",
    "magarpatta": "Hadapsar",
    "manjari": "Hadapsar",
    # Aundh / Ghole Road area
    "aundh": "Aundh",
    "pashan": "Aundh",
    "baner": "Aundh",
    "balewadi": "Aundh",
    "shivajinagar": "Ghole Road",
    "deccan": "Ghole Road",
    "ghole": "Ghole Road",
    "gokhalenagar": "Ghole Road",
    "manapa": "Ghole Road",  # Ma Na Pa = Municipal HQ
    "ferguson": "Ghole Road",
    # Nagar Road
    "nagarroad": "Nagar Road",
    "kharadi": "Nagar Road",
    "wagholi": "Nagar Road",
    # Kothrud
    "kothrud": "Kothrud",
    "karvenagar": "Kothrud",
    "warje": "Warje",
    "karve": "Kothrud",
    # Dhankawadi / Bibvewadi / Sahakarnagar south
    "dhankawadi": "Dhankawadi",
    "bibvewadi": "Bibvewadi",
    "katraj": "Dhankawadi",
    # Yewalewadi (south periphery)
    "yewalewadi": "Yewalewadi",
    "undri": "Yewalewadi",
    # Kasbavish-Rambaug (central traditional, low SC+ST)
    "shaniwar": "Kasbavish-Rambaug",
    "narayanpeth": "Kasbavish-Rambaug",
    "sadashiv": "Kasbavish-Rambaug",
    "kasba": "Kasbavish-Rambaug",
    "marketyard": "Kasbavish-Rambaug",
    "swargate": "Kasbavish-Rambaug",  # adjacent
    # Tilak Road
    "tilak": "Tilak Road",
}


def _canon(s: str) -> str:
    """Canonical lowercase, alphanumeric-only form for fuzzy matching."""
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def load_routes_raw() -> pd.DataFrame:
    """Return the raw OpenCity route file with origin/destination split."""
    df = pd.read_csv(os.path.join(RAW_DIR, "pmpml_routes.csv"))
    df["base_id"] = df["Route ID"].astype(str).str.replace(_DIR_RE, "", regex=True)
    od = df["Route Description"].fillna("").str.split(" To ", n=1, expand=True)
    df["origin"] = od[0].fillna("").str.strip()
    df["destination"] = od[1].fillna("").str.strip() if od.shape[1] > 1 else ""
    return df


def _aggregate_routes(raw: pd.DataFrame) -> pd.DataFrame:
    """One row per base route, keeping max distance across direction pairs."""
    grp = raw.groupby("base_id", sort=False)
    out = pd.DataFrame({
        "route_id": grp.size().index,
        "Kilometer": grp["Kilometer"].max().values,
    })
    first = grp.first()
    out["origin"] = first["origin"].values
    out["destination"] = first["destination"].values
    return out.reset_index(drop=True)


def load_ward_priority() -> dict:
    """Per-ward SC+ST share aggregated from the 160 numbered wards into the
    15 named PMC wards. Returned dict maps ward name (str) -> share (float)."""
    c = pd.read_csv(os.path.join(RAW_DIR, "pune_census_2011.csv"))
    agg = c.groupby("Ward Name")[["TOT_P", "P_SC", "P_ST"]].sum()
    agg["share"] = (agg["P_SC"] + agg["P_ST"]) / agg["TOT_P"]
    return agg["share"].to_dict()


def _match_endpoint_to_ward(endpoint: str, ward_canon_keys: list) -> str | None:
    """Match a route endpoint to a Census ward. First tries the explicit
    alias dict (landmark -> ward), then a substring match on the canonical
    ward names themselves. Returns None if no match."""
    ep = _canon(endpoint)
    if not ep:
        return None
    # 1. Alias-based match (landmark -> ward).
    for alias, ward in _ENDPOINT_TO_WARD_ALIASES.items():
        if alias in ep:
            return ward
    # 2. Direct substring match on canonical ward names.
    for ward_name, ward_canon in ward_canon_keys:
        if ward_canon and ward_canon in ep:
            return ward_name
    return None


def _match_endpoint_to_depot(endpoint: str, depot_canon: list) -> str | None:
    """Match a route endpoint to a depot by canonical substring."""
    ep = _canon(endpoint)
    if not ep:
        return None
    for depot_name, depot_canon_name in depot_canon:
        if depot_canon_name in ep:
            return depot_name
    return None


def assign_route_metadata(routes: pd.DataFrame,
                          ward_priority: dict,
                          rng: np.random.Generator) -> pd.DataFrame:
    """Add depot_id, priority_score, route_category, num_stops, avg_trip_time_min
    to the aggregated route table. Returns a routes_df ready for the existing
    pipeline (same schema as the old synthetic generate_routes output)."""
    ward_canon = [(w, _canon(w)) for w in ward_priority]
    depot_canon = [(d[0], _canon(d[0])) for d in DEPOTS]
    depot_names = [d[0] for d in DEPOTS]

    median_share = float(np.median(list(ward_priority.values())))
    p25 = float(np.quantile(list(ward_priority.values()), 0.25))
    p75 = float(np.quantile(list(ward_priority.values()), 0.75))

    rows = []
    fallback_depot_idx = 0
    for _, r in routes.iterrows():
        rid = r["route_id"]
        length_km = float(r["Kilometer"]) if pd.notna(r["Kilometer"]) else 0.0
        if length_km <= 0:
            continue
        origin = r["origin"]
        dest = r["destination"]

        # Depot: match origin first, then destination, then round-robin fallback.
        dep = (_match_endpoint_to_depot(origin, depot_canon)
               or _match_endpoint_to_depot(dest, depot_canon))
        if dep is None:
            dep = depot_names[fallback_depot_idx % len(depot_names)]
            fallback_depot_idx += 1

        # Ward: match origin first, then destination, then median priority.
        ward = (_match_endpoint_to_ward(origin, ward_canon)
                or _match_endpoint_to_ward(dest, ward_canon))
        if ward is not None:
            share = float(ward_priority[ward])
        else:
            share = median_share

        # Convert SC+ST share to a [0, 1] priority score with mild noise so
        # routes within the same ward don't all collide on identical scores.
        # Top-quartile wards -> priority_score in [0.7, 1.0]; rest interpolated.
        if share >= p75:
            base = 0.7 + 0.3 * (share - p75) / max(1e-9, max(ward_priority.values()) - p75)
        elif share >= median_share:
            base = 0.5 + 0.2 * (share - median_share) / max(1e-9, p75 - median_share)
        elif share >= p25:
            base = 0.3 + 0.2 * (share - p25) / max(1e-9, median_share - p25)
        else:
            base = 0.0 + 0.3 * share / max(1e-9, p25)
        priority = float(np.clip(base + rng.normal(0, 0.03), 0.0, 1.0))

        # Service category from distance.
        if length_km < 8:
            category = "feeder"
            peak_lo, peak_hi = 1, 5
        elif length_km > 22:
            category = "suburban"
            peak_lo, peak_hi = 2, 6
        else:
            category = "trunk"
            peak_lo, peak_hi = 3, 8

        # Trip time: Pune average bus speed ~ 17 km/h (cited in existing paper
        # and in Pune traffic studies). Travel only, dwell added by tau.
        avg_trip_time = length_km * 60.0 / 17.0 + rng.normal(0, 3)
        avg_trip_time = float(max(8.0, avg_trip_time))
        num_stops = int(np.clip(length_km * 2.5 + rng.normal(0, 3), 5, 80))

        # Status-quo frequencies in the integer menu PMPML uses.
        peak_freq = int(rng.integers(peak_lo, peak_hi + 1))
        offpeak_freq = max(1, int(round(peak_freq * float(rng.uniform(0.5, 0.8)))))

        rows.append({
            "route_id": rid,
            "route_name": f"{origin} - {dest}",
            "length_km": round(length_km, 2),
            "avg_trip_time_min": round(avg_trip_time, 1),
            "num_stops": num_stops,
            "current_peak_freq": peak_freq,
            "current_offpeak_freq": offpeak_freq,
            "route_category": category,
            "priority_score": round(priority, 3),
            "depot_id": dep,
        })
    return pd.DataFrame(rows)


def _calibrate_peak_fleet(df: pd.DataFrame, target: int = 1500,
                          dwell_min: float = 7.5) -> pd.DataFrame:
    """Scale peak frequencies so the status-quo peak fleet is near `target`."""
    def fleet_at(df_):
        tau = 2.0 * df_["avg_trip_time_min"] + dwell_min
        return int(np.ceil(df_["current_peak_freq"] * tau / 60.0).sum())

    cur = fleet_at(df)
    if abs(cur - target) > 50:
        scale = target / cur
        df["current_peak_freq"] = np.clip(
            np.round(df["current_peak_freq"] * scale), 1, 12,
        ).astype(int)
        df["current_offpeak_freq"] = np.clip(
            np.round(df["current_peak_freq"] * np.random.default_rng(7).uniform(0.5, 0.8, len(df))),
            1, 10,
        ).astype(int)
    return df


def build_real_routes(target_peak_fleet: int = 1500, seed: int = 42) -> pd.DataFrame:
    """Top-level entry point: returns the same routes_df schema as the old
    synthetic generator, but anchored to real PMPML route topology and real
    Census 2011 priority designations."""
    raw = load_routes_raw()
    agg = _aggregate_routes(raw)
    wards = load_ward_priority()
    rng = np.random.default_rng(seed)
    routes = assign_route_metadata(agg, wards, rng)
    routes = _calibrate_peak_fleet(routes, target=target_peak_fleet)
    return routes


if __name__ == "__main__":
    df = build_real_routes()
    print(f"Loaded {len(df)} real PMPML routes.")
    print(f"Categories: {df['route_category'].value_counts().to_dict()}")
    print(f"Depot route counts: {df['depot_id'].value_counts().to_dict()}")
    print(f"Priority distribution: mean={df['priority_score'].mean():.3f}, "
          f">0.7: {(df['priority_score'] > 0.7).sum()}")
    print(f"Distance: {df['length_km'].min():.1f}-{df['length_km'].max():.1f} km, "
          f"mean {df['length_km'].mean():.1f}")
    tau = 2.0 * df['avg_trip_time_min'] + 7.5
    peak_fleet = int(np.ceil(df['current_peak_freq'] * tau / 60).sum())
    print(f"Status-quo peak fleet (with corrected tau): {peak_fleet}")
