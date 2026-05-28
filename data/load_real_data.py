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
# ward name from ``data/raw/pune_census_2011.csv``.
#
# The mapping has researcher-degrees-of-freedom: each alias is a judgement
# about which census ward best contains the landmark's catchment. We expose
# three rules so the sensitivity check in ``sensitivity.equity_mapping_sensitivity``
# can quantify how much the headline equity-cost figure depends on the
# table:
#
#   ``strict``       -- empty alias table; only literal ward-name substrings
#                       match. The most defensible mapping; produces the
#                       smallest priority set.
#   ``conservative`` -- alias entries with clear geographic justification
#                       only. Drops cross-ward equity classifications such
#                       as ``kondhwa -> Bhavani Peth``.
#   ``current``      -- full alias table, including landmark-to-equity-ward
#                       classifications that are defensible on welfare
#                       grounds (a landmark in a low-income area is mapped
#                       to a ward with comparable SC+ST share even if not
#                       strictly adjacent). Used by default.
#
# Geographic notes underlie each entry where the mapping is non-obvious.
_CONSERVATIVE_ALIASES = {
    # Sangamwadi (Yerawada) -- Yerawada / Vishrantwadi catchment (adjacent)
    "yerwada": "Sangamwadi (Yerawada)",
    "yerawada": "Sangamwadi (Yerawada)",
    "vishrantwadi": "Sangamwadi (Yerawada)",
    "sangamwadi": "Sangamwadi (Yerawada)",
    "sangamvadi": "Sangamwadi (Yerawada)",
    # Dholepatil Road -- Pune Station / Wadgaon Sheri catchment
    "punestation": "Dholepatil Road",
    "wadgaonsheri": "Dholepatil Road",
    "dholepatil": "Dholepatil Road",
    # Sahakarnagar
    "sahakarnagar": "Sahakarnagar",
    "sahakar": "Sahakarnagar",
    # Bhavani Peth -- central old city
    "bhavanipeth": "Bhavani Peth",
    "bhavani": "Bhavani Peth",
    # Hadapsar
    "hadapsar": "Hadapsar",
    "mundhwa": "Hadapsar",
    "magarpatta": "Hadapsar",
    # Aundh
    "aundh": "Aundh",
    # Ghole Road -- Shivajinagar / Deccan area
    "shivajinagar": "Ghole Road",
    "deccan": "Ghole Road",
    "ghole": "Ghole Road",
    # Nagar Road -- Kharadi / Wagholi catchment (Kharadi is east-Pune, on
    # Nagar Road corridor; this is the canonical Pune urban-planning
    # classification).
    "nagarroad": "Nagar Road",
    "kharadi": "Nagar Road",
    "wagholi": "Nagar Road",
    # Kothrud
    "kothrud": "Kothrud",
    "karve": "Kothrud",
    # Dhankawadi / Bibvewadi
    "dhankawadi": "Dhankawadi",
    "bibvewadi": "Bibvewadi",
    # Yewalewadi
    "yewalewadi": "Yewalewadi",
    # Kasbavish-Rambaug
    "shaniwar": "Kasbavish-Rambaug",
    "narayanpeth": "Kasbavish-Rambaug",
    "sadashiv": "Kasbavish-Rambaug",
    "kasba": "Kasbavish-Rambaug",
    # Tilak Road
    "tilak": "Tilak Road",
    # Warje
    "warje": "Warje",
}

# Additional aliases used by the "current" rule. These have weaker or more
# debatable geographic justification; the sensitivity analysis quantifies
# their effect on the equity-cost number.
_DEBATABLE_ALIASES = {
    # Kalyani Nagar sits between Yerawada and Koregaon Park; we classify it
    # with Sangamwadi because most of its bus catchment connects through
    # Yerawada landmarks.
    "kalyaninagar": "Sangamwadi (Yerawada)",
    # Viman Nagar is a Wadgaon Sheri / Dholepatil neighbour to the south.
    "viman": "Dholepatil Road",
    # Parvati is the southern slope below Sahakarnagar.
    "parvati": "Sahakarnagar",
    # Kondhwa / Wanowri are south-central low-income areas; not in Bhavani
    # Peth ward geographically, but classified here on welfare grounds
    # because their SC+ST shares are comparable.
    "kondhwa": "Bhavani Peth",
    "wanowri": "Bhavani Peth",
    # Manjari is east of Hadapsar.
    "manjari": "Hadapsar",
    # Pashan / Baner / Balewadi are western suburbs that historically share
    # an Aundh catchment for bus operations.
    "pashan": "Aundh",
    "baner": "Aundh",
    "balewadi": "Aundh",
    # Gokhalenagar / Ferguson / Ma Na Pa are in the Shivajinagar area.
    "gokhalenagar": "Ghole Road",
    "ferguson": "Ghole Road",
    "manapa": "Ghole Road",
    # Karve Nagar is between Kothrud and Warje.
    "karvenagar": "Kothrud",
    # Katraj is a south-Pune landmark adjacent to the Dhankawadi ward.
    "katraj": "Dhankawadi",
    # Undri is a southern peripheral landmark near Yewalewadi.
    "undri": "Yewalewadi",
    # Marketyard is in the Kasbavish-Rambaug central area.
    "marketyard": "Kasbavish-Rambaug",
    # Swargate is adjacent to the central old city, low SC+ST.
    "swargate": "Kasbavish-Rambaug",
}

_CURRENT_ALIASES = {**_CONSERVATIVE_ALIASES, **_DEBATABLE_ALIASES}


def _aliases_for(rule: str) -> dict:
    if rule == "strict":
        return {}
    if rule == "conservative":
        return _CONSERVATIVE_ALIASES
    if rule == "current":
        return _CURRENT_ALIASES
    raise ValueError(f"unknown priority_rule: {rule!r}")


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


def _match_endpoint_to_ward(endpoint: str, ward_canon_keys: list,
                            aliases: dict) -> str | None:
    """Match a route endpoint to a Census ward. First tries the supplied
    alias dict (landmark -> ward), then a substring match on the canonical
    ward names themselves. Returns None if no match."""
    ep = _canon(endpoint)
    if not ep:
        return None
    # 1. Alias-based match (landmark -> ward).
    for alias, ward in aliases.items():
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
                          rng: np.random.Generator,
                          priority_rule: str = "current") -> pd.DataFrame:
    """Add depot_id, priority_score, route_category, num_stops, avg_trip_time_min
    to the aggregated route table. ``priority_rule`` selects the
    landmark-to-ward alias table (see _aliases_for)."""
    aliases = _aliases_for(priority_rule)
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
        ward = (_match_endpoint_to_ward(origin, ward_canon, aliases)
                or _match_endpoint_to_ward(dest, ward_canon, aliases))
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


def build_real_routes(target_peak_fleet: int = 1500, seed: int = 42,
                      priority_rule: str = "current") -> pd.DataFrame:
    """Top-level entry point: returns the same routes_df schema as the old
    synthetic generator, but anchored to real PMPML route topology and real
    Census 2011 priority designations.

    ``priority_rule`` selects how route endpoints are mapped to Census
    wards for the equity / priority score (see _aliases_for):
    ``current`` (default), ``conservative``, or ``strict``.
    """
    raw = load_routes_raw()
    agg = _aggregate_routes(raw)
    wards = load_ward_priority()
    rng = np.random.default_rng(seed)
    routes = assign_route_metadata(agg, wards, rng, priority_rule=priority_rule)
    routes = _calibrate_peak_fleet(routes, target=target_peak_fleet)
    return routes


if __name__ == "__main__":
    for rule in ("current", "conservative", "strict"):
        df = build_real_routes(priority_rule=rule)
        tau = 2.0 * df['avg_trip_time_min'] + 7.5
        peak_fleet = int(np.ceil(df['current_peak_freq'] * tau / 60).sum())
        n_pri = (df['priority_score'] > 0.7).sum()
        print(f"[{rule:>12s}] n_routes={len(df)} n_priority={n_pri:3d} "
              f"(pri_share={n_pri/len(df):.1%}) peak_fleet={peak_fleet}")
