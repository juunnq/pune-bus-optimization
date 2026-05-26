"""
Data generation for pune-bus-optimization.
Produces routes.csv, weather.csv, demand.csv, demand_matrix.csv in data/processed/.
"""
import os
import sys
import random
import numpy as np
import pandas as pd
import requests

np.random.seed(42)
random.seed(42)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")

# Pune geography
PUNE_NEIGHBORHOODS = [
    "Swargate", "Katraj", "Shivajinagar", "Hinjewadi", "Kothrud", "Hadapsar",
    "Nigdi", "Pimpri", "Wakad", "Baner", "Aundh", "Pashan", "Deccan", "Camp",
    "Koregaon Park", "Viman Nagar", "Kharadi", "Magarpatta", "Undri", "PCMC",
    "Chinchwad", "Bhosari", "Akurdi", "Sangvi", "Kondhwa", "Warje", "Bibvewadi",
    "Yerwada", "Vishrantwadi", "Dhankawadi"
]

LOW_INCOME = {"Hadapsar", "Kondhwa", "Undri", "Bhosari", "Pimpri", "Yerwada", "Dhankawadi"}
AFFLUENT = {"Koregaon Park", "Baner", "Aundh", "Camp", "Magarpatta", "Viman Nagar"}

DEPOT_GROUPS = {
    "D1": {"Swargate", "Katraj", "Bibvewadi", "Dhankawadi"},
    "D2": {"Shivajinagar", "Deccan", "Camp"},
    "D3": {"Hinjewadi", "Wakad", "Baner"},
    "D4": {"Kothrud", "Warje", "Pashan"},
    "D5": {"Hadapsar", "Magarpatta", "Undri", "Kondhwa"},
    "D6": {"Nigdi", "Pimpri", "PCMC", "Chinchwad", "Akurdi", "Bhosari", "Sangvi"},
    "D7": {"Aundh"},
    "D8": {"Koregaon Park", "Viman Nagar", "Kharadi", "Yerwada", "Vishrantwadi"},
}


def assign_depot(neighborhood):
    for d, areas in DEPOT_GROUPS.items():
        if neighborhood in areas:
            return d
    return "D1"


# Routes
def generate_routes(n_routes=340):
    rows = []
    for i in range(1, n_routes + 1):
        rid = f"R{i:03d}"
        r = np.random.random()
        if r < 0.15:
            category = "trunk"
            length_km = np.random.uniform(15, 45)
            peak_lo, peak_hi = 5, 10
        elif r < 0.65:
            category = "feeder"
            length_km = np.random.uniform(5, 15)
            peak_lo, peak_hi = 1, 5
        else:
            category = "suburban"
            length_km = np.random.uniform(20, 40)
            peak_lo, peak_hi = 2, 6

        origin = random.choice(PUNE_NEIGHBORHOODS)
        dest = random.choice([n for n in PUNE_NEIGHBORHOODS if n != origin])

        avg_trip_time = max(10.0, length_km * 3.5 + np.random.normal(0, 5))
        num_stops = int(np.clip(length_km * 2.5 + np.random.normal(0, 3), 5, 80))
        peak_freq = int(np.random.randint(peak_lo, peak_hi + 1))
        offpeak_freq = max(1, int(round(peak_freq * np.random.uniform(0.5, 0.8))))

        if origin in LOW_INCOME or dest in LOW_INCOME:
            priority = float(np.random.uniform(0.7, 1.0))
        elif origin in AFFLUENT or dest in AFFLUENT:
            priority = float(np.random.uniform(0.0, 0.4))
        else:
            priority = float(np.random.uniform(0.3, 0.7))

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
            "depot_id": assign_depot(origin),
        })

    df = pd.DataFrame(rows)

    # Calibrate frequencies so the status-quo peak fleet leaves headroom for
    # the MIP's equity floor (priority routes at k>=3 during peaks). With the
    # round-trip cycle time tau = 2 * avg_trip + dwell (must match
    # deterministic_model.DWELL_MIN), bus consumption per route is ~2x what it
    # was under the old one-way-time approximation. Targeting peak_fleet ~1500
    # gives the MIP roughly 300 buses of room for the equity-floor lift before
    # hitting the operational fleet cap of B=1800.
    DWELL_MIN = 7.5

    def fleet(d):
        tau = 2.0 * d['avg_trip_time_min'] + DWELL_MIN
        return int(np.ceil(d['current_peak_freq'] * tau / 60).sum())

    cur = fleet(df)
    target = 1500
    if abs(cur - target) > 100:
        scale = target / cur
        df['current_peak_freq'] = np.clip(
            np.round(df['current_peak_freq'] * scale), 1, 12
        ).astype(int)
        df['current_offpeak_freq'] = np.clip(
            np.round(df['current_peak_freq'] * np.random.uniform(0.5, 0.8, len(df))), 1, 10
        ).astype(int)

    return df


# Weather
def fetch_or_generate_weather():
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": 18.5204,
        "longitude": 73.8567,
        "start_date": "2023-01-01",
        "end_date": "2024-12-31",
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,rain_sum",
        "timezone": "Asia/Kolkata",
    }
    try:
        resp = requests.get(url, params=params, timeout=20)
        resp.raise_for_status()
        d = resp.json()["daily"]
        df = pd.DataFrame({
            "date": pd.to_datetime(d["time"]),
            "temperature_max": d["temperature_2m_max"],
            "temperature_min": d["temperature_2m_min"],
            "precipitation_mm": d["precipitation_sum"],
        })
        # Drop any rows with nulls (API can return nulls at edges)
        df = df.dropna().reset_index(drop=True)
        if len(df) < 365:
            raise RuntimeError(f"weather API returned only {len(df)} rows")
        print(f"  weather source: open-meteo ({len(df)} days)")
        return df
    except Exception as e:
        print(f"  weather API failed ({e}); using synthetic")
        return generate_synthetic_weather()


def generate_synthetic_weather():
    dates = pd.date_range("2023-01-01", "2024-12-31", freq="D")
    n = len(dates)
    doy = dates.dayofyear.values
    # Peak ~mid-May (day ~135), trough ~late Dec
    temp_max = 33.5 + 8.5 * np.sin(2 * np.pi * (doy - 45) / 365.0) + np.random.normal(0, 3, n)
    temp_min = temp_max - 10 + np.random.normal(0, 2, n)

    months = dates.month.values
    monsoon = (months >= 6) & (months <= 9)
    precip = np.zeros(n)
    precip[monsoon] = np.random.exponential(8, monsoon.sum())
    pool = np.where(monsoon)[0]
    n_spike = int(0.03 * n)
    if len(pool) > 0 and n_spike > 0:
        spike_idx = np.random.choice(pool, size=min(n_spike, len(pool)), replace=False)
        precip[spike_idx] = np.random.uniform(50, 100, len(spike_idx))
    precip = np.maximum(precip + np.random.normal(0, 0.5, n), 0)

    return pd.DataFrame({
        "date": dates,
        "temperature_max": np.round(temp_max, 1),
        "temperature_min": np.round(temp_min, 1),
        "precipitation_mm": np.round(precip, 2),
    })


# Demand
HOUR_PROFILE = {
    6: 0.4, 7: 0.7, 8: 1.3, 9: 1.5, 10: 1.0, 11: 0.8,
    12: 0.7, 13: 0.7, 14: 0.7, 15: 0.8, 16: 0.9, 17: 1.3,
    18: 1.5, 19: 1.2, 20: 0.8, 21: 0.5, 22: 0.3,
}

INDIA_HOLIDAYS = {
    "2023-01-26", "2023-03-08", "2023-04-07", "2023-04-14", "2023-05-01",
    "2023-08-15", "2023-08-29", "2023-09-19", "2023-10-02", "2023-10-24",
    "2023-11-12", "2023-12-25",
    "2024-01-26", "2024-03-08", "2024-03-25", "2024-04-11", "2024-04-17",
    "2024-05-01", "2024-08-15", "2024-09-07", "2024-10-02", "2024-11-01",
    "2024-12-25",
}


def is_festival(d):
    if d.month == 9 and 10 <= d.day <= 17:
        return True
    if d.month == 11 and 5 <= d.day <= 12:
        return True
    return False


def generate_demand(routes_df, weather_df):
    """Top 50 routes by current peak frequency, full year 2024, hourly.

    The DGP includes three sources of irreducible noise that test the
    decision-focused learning claim more honestly than i.i.d. small-sigma
    Gaussian noise would:
      (i)   a latent per-day factor correlated across routes (unobserved by
            the predictor);
      (ii)  per-(route, date) hidden surge events (also unobserved);
      (iii) heavy-tailed (Student-t) multiplicative noise on a "volatile"
            subset of routes, light Gaussian noise on the rest.
    The latent variables are NOT included in the output feature columns, so a
    learner cannot recover them by inspection.
    """
    top50 = routes_df.nlargest(50, 'current_peak_freq').copy().reset_index(drop=True)
    top50['base_per_hour'] = (
        top50['current_peak_freq'] * 4 * 45
        + top50['current_offpeak_freq'] * 13 * 25
    ) / 17.0

    # Mark every 5th route (10 of 50) as volatile — heavy-tailed noise.
    volatile_routes = set(top50.iloc[::5]['route_id'].tolist())

    w24 = weather_df[weather_df['date'].dt.year == 2024].copy().reset_index(drop=True)
    w24['day_of_week'] = w24['date'].dt.dayofweek
    w24['is_weekend'] = (w24['day_of_week'] >= 5).astype(int)
    w24['month'] = w24['date'].dt.month
    w24['is_holiday'] = w24['date'].dt.strftime('%Y-%m-%d').isin(INDIA_HOLIDAYS).astype(int)
    w24['is_monsoon'] = ((w24['month'] >= 6) & (w24['month'] <= 9)).astype(int)
    w24['is_exam_season'] = w24['month'].isin([3, 4]).astype(int)
    w24['is_festival'] = w24['date'].apply(is_festival).astype(int)

    # Latent per-date factor (correlated demand shock across all routes).
    rng_day = np.random.default_rng(43)
    w24['_day_factor'] = rng_day.normal(0, 0.08, len(w24))

    hours_df = pd.DataFrame({'hour': list(range(24))})
    hours_df['profile'] = hours_df['hour'].map(lambda h: HOUR_PROFILE.get(h, 0.05))

    routes_keys = top50[['route_id', 'base_per_hour']].copy()
    routes_keys['_k'] = 1
    w24['_k'] = 1
    hours_df['_k'] = 1

    df = (routes_keys.merge(w24, on='_k')
                     .merge(hours_df, on='_k')
                     .drop(columns='_k'))

    # Per-(route, date) hidden surge events: ~3% of pairs get a 1.5-2.5x boost.
    rng_event = np.random.default_rng(44)
    n_routes = len(top50)
    n_dates = len(w24)
    event_mask = rng_event.random((n_routes, n_dates)) < 0.03
    event_size = rng_event.uniform(1.5, 2.5, (n_routes, n_dates))
    # Map (route_id, date) -> (i, j) indices for broadcast.
    route_idx = {r: i for i, r in enumerate(top50['route_id'].values)}
    date_idx = {pd.Timestamp(d): j for j, d in enumerate(w24['date'].values)}
    df['_ridx'] = df['route_id'].map(route_idx)
    df['_didx'] = df['date'].map(lambda d: date_idx[pd.Timestamp(d)])
    event_mult = np.where(event_mask[df['_ridx'].values, df['_didx'].values],
                          event_size[df['_ridx'].values, df['_didx'].values],
                          1.0)

    df['ridership'] = df['base_per_hour'] * df['profile']
    rain_mult = np.where(df['precipitation_mm'] > 10,
                         np.maximum(0.7, 1 - df['precipitation_mm'] / 100), 1.0)
    df['ridership'] *= rain_mult
    df['ridership'] *= np.where(df['is_weekend'] == 1, 0.6, 1.0)
    df['ridership'] *= np.where(df['is_holiday'] == 1, 0.5, 1.0)
    df['ridership'] *= np.where(df['is_exam_season'] == 1, 1.15, 1.0)
    df['ridership'] *= np.where(df['is_festival'] == 1, 0.75, 1.0)
    df['ridership'] *= np.where(df['temperature_max'] > 40, 0.9, 1.0)
    df['ridership'] *= (1 + df['_day_factor'])  # correlated latent shock
    df['ridership'] *= event_mult               # hidden surge events

    # Heavy-tailed Student-t noise on volatile routes; light Normal on rest.
    is_vol = df['route_id'].isin(volatile_routes).values
    rng_noise = np.random.default_rng(45)
    n = len(df)
    light = rng_noise.normal(0, 0.05, n)
    heavy = rng_noise.standard_t(df=3, size=n) * 0.10
    noise = np.where(is_vol, heavy, light)
    df['ridership'] *= (1 + noise)

    df['ridership'] = np.maximum(df['ridership'], 0).round(2)

    cols = ['route_id', 'date', 'hour', 'day_of_week', 'is_weekend', 'is_holiday',
            'month', 'temperature_max', 'precipitation_mm', 'is_monsoon',
            'is_exam_season', 'is_festival', 'ridership']
    return df[cols].sort_values(['route_id', 'date', 'hour']).reset_index(drop=True)


# Demand matrix
PERIOD_BOUNDS = [
    ("early_morning", 6, 8),
    ("AM_peak", 8, 10),
    ("midday", 10, 17),
    ("PM_peak", 17, 20),
    ("evening", 20, 23),
]


def hour_to_period(h):
    for name, lo, hi in PERIOD_BOUNDS:
        if lo <= h < hi:
            return name
    return None


def generate_demand_matrix(routes_df, demand_df):
    """340 routes x 5 periods.

    Modeled routes (50): per-(route, period) mean observed ridership.

    Unmodeled routes (290): parametric base_per_hour x period_profile, then
    multiplied by a per-period RESIDUAL RATIO estimated on the modeled subset.
    The residual ratio for period t is the empirical mean of
    (observed_demand_t / parametric_baseline_t) across modeled routes; it
    corrects the parametric baseline for systematic period-level effects that
    the hourly-profile constants alone do not capture.
    """
    d = demand_df.copy()
    d['period'] = d['hour'].apply(hour_to_period)
    d = d.dropna(subset=['period'])
    detailed = d.groupby(['route_id', 'period'])['ridership'].mean().unstack('period')

    period_avg = {name: float(np.mean([HOUR_PROFILE.get(h, 0.05) for h in range(lo, hi)]))
                  for name, lo, hi in PERIOD_BOUNDS}
    period_cols = [name for name, _, _ in PERIOD_BOUNDS]

    # Parametric baseline for the modeled 50 (same formula used for unmodeled).
    modeled = routes_df[routes_df['route_id'].isin(detailed.index)].copy()
    modeled['base_per_hour'] = (
        modeled['current_peak_freq'] * 4 * 45
        + modeled['current_offpeak_freq'] * 13 * 25
    ) / 17.0
    base_modeled = pd.DataFrame(index=modeled.set_index('route_id').index,
                                columns=period_cols, dtype=float)
    for _, r in modeled.iterrows():
        for name in period_cols:
            base_modeled.loc[r['route_id'], name] = r['base_per_hour'] * period_avg[name]

    # Per-period residual ratio = mean(observed / parametric_baseline).
    aligned = detailed.reindex(columns=period_cols)
    residual_ratio = {}
    for name in period_cols:
        actual = aligned[name]
        base = base_modeled[name]
        common = actual.notna() & (base > 0)
        if common.sum() > 0:
            residual_ratio[name] = float((actual[common] / base[common]).mean())
        else:
            residual_ratio[name] = 1.0
    print(f"  residual ratios (modeled subset): {residual_ratio}")

    other_ids = sorted(set(routes_df['route_id']) - set(detailed.index))
    other = routes_df[routes_df['route_id'].isin(other_ids)].copy()
    other['base_per_hour'] = (
        other['current_peak_freq'] * 4 * 45
        + other['current_offpeak_freq'] * 13 * 25
    ) / 17.0

    other_rows = []
    for _, r in other.iterrows():
        row = {'route_id': r['route_id']}
        for name in period_cols:
            row[name] = round(r['base_per_hour'] * period_avg[name] * residual_ratio[name], 2)
        other_rows.append(row)
    other_df = pd.DataFrame(other_rows).set_index('route_id')

    full = pd.concat([detailed, other_df]).round(2)
    full = full[period_cols].reset_index().rename(columns={'index': 'route_id'})
    return full


# Main
def main():
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    print("Generating routes...")
    routes_df = generate_routes(340)
    routes_path = os.path.join(PROCESSED_DIR, "routes.csv")
    routes_df.to_csv(routes_path, index=False)
    tau = 2.0 * routes_df['avg_trip_time_min'] + 7.5
    fleet = int(np.ceil(routes_df['current_peak_freq'] * tau / 60).sum())
    print(f"  routes={len(routes_df)}, peak_fleet={fleet}")

    print("Fetching weather...")
    weather_df = fetch_or_generate_weather()
    weather_path = os.path.join(PROCESSED_DIR, "weather.csv")
    weather_df.to_csv(weather_path, index=False)
    print(f"  weather days={len(weather_df)}")

    print("Generating demand...")
    demand_df = generate_demand(routes_df, weather_df)
    demand_path = os.path.join(PROCESSED_DIR, "demand.csv")
    demand_df.to_csv(demand_path, index=False)
    print(f"  demand rows={len(demand_df)}")

    print("Generating demand matrix...")
    matrix_df = generate_demand_matrix(routes_df, demand_df)
    matrix_path = os.path.join(PROCESSED_DIR, "demand_matrix.csv")
    matrix_df.to_csv(matrix_path, index=False)
    print(f"  matrix routes={len(matrix_df)}")

    print("\nDone.")


if __name__ == "__main__":
    main()
