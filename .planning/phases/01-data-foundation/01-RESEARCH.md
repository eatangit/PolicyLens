# Phase 1: Data Foundation — Research

**Researched:** 2026-04-18
**Domain:** EIA v2 API integration, data normalization, module architecture
**Confidence:** MEDIUM-HIGH (EIA route structure verified via live API; exact series IDs for some variables ASSUMED from naming convention — must be validated with actual API key during execution)

---

## Summary

Phase 1 establishes the data contract that every downstream component depends on. The core deliverables are: `VAR_INDEX` (single source of truth for 31 variable indices), `src/data.py` (EIA v2 loader with pagination and fallback), `src/normalizer.py` (RobustScaler wrapper with JSON serialization), `data/sample_state.json` (hardcoded 2023 baseline), and `data/normalizer_params.json` (scaler parameters fit on 1960–2005 only).

The most significant research findings for planning: (1) the `co2-emissions` route is **fully deprecated** — CO2 data must come from `/v2/seds/` using MSN series codes; (2) the `electricity/electric-power-operational-data` route starts in 2001 — generation-by-fuel variables have no data for 1960–2000, requiring a fallback strategy for the training split; (3) `densified-biomass` data starts only in ~2016, making ENV-15 useless for GRU training; (4) `nuclear-outages` provides daily/weekly data, not annual — aggregation required; (5) STEO data is a 18-month rolling forecast, not historical — ECON-16 needs special handling.

**Primary recommendation:** Write `src/data.py` as a two-layer system: a thin EIA fetch layer (one function per variable group, not per variable) and a state assembly layer that maps fetched values into the 31-dim VAR_INDEX-ordered array. Validate with actual EIA API key immediately at execution start, before writing other modules.

---

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| EIA HTTP fetching | `src/data.py` | None | Single module owns all external I/O |
| VAR_INDEX definition | `src/data.py` | None | Imported by parser, env, model — single source |
| State assembly (31-dim array) | `src/data.py` | None | Aggregation from multiple routes |
| Offline fallback | `src/data.py` | `data/sample_state.json` | Fallback file consumed by loader |
| Normalization fit | `src/normalizer.py` | `data/normalizer_params.json` | Serialized scaler params |
| Normalization transform | `src/normalizer.py` | None | Stateless after load |
| Physical bounds registry | `src/data.py` | Consumed by `src/env.py` | Derived from EIA historical range |

---

## Project Constraints (from CLAUDE.md)

- **Never shuffle time-series data** — chronological split only (train 1960–2005, val 2006–2015, test 2016–2023)
- **Clamp all LLM action magnitudes** to [0.0, 1.0] in application code
- **Clamp state variables** to physical bounds after every `step()`
- **Sync Anthropic client only** — not relevant to Phase 1
- **Pin numpy<2.0** — ABI break in 2.x breaks `torch.from_numpy()` silently
- **EIA values are JSON strings** since Jan 2024 — always `float(row.get("value", "nan"))`
- **Schema field order matters** — not relevant to Phase 1
- **31 state variables** defined in `VAR_INDEX` constant in `src/data.py`
- **RobustScaler** per variable, fitted on 1960–2005 only, serialized to `data/normalizer_params.json`

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| DATA-01 | Fetch live EIA baseline from ≥10 distinct EIA v2 endpoints | Confirmed feasible; see EIA Route Map section |
| DATA-02 | Fall back to `data/sample_state.json` when API unavailable | Load-and-fall-through pattern documented |
| DATA-03 | `VAR_INDEX` dict in single shared module | Single-module pattern documented |
| DATA-04 | Parse all values as `float`, nan fallback | Confirmed: all values are strings since v2.1.6 (Jan 2024) |
| DATA-05 | Handle 5000-row pagination, cache locally | Pagination via `offset`+`total` documented; X-Warning header confirmed |
| DATA-06 | RobustScaler fit on 1960–2005 only, serialize to JSON | `center_` + `scale_` attributes confirmed serializable |
| ENV-01 | `co2_total_emissions` — total US CO2 (MMT CO2) | SEDS route confirmed; series CO2TCA assumed |
| ENV-02 | `co2_electric_power` — electric power CO2 | SEDS route confirmed; series CO2TEA assumed |
| ENV-03 | `co2_transportation` — transportation CO2 | SEDS route confirmed; series CO2TXA assumed |
| ENV-04 | `co2_industrial` — industrial CO2 | SEDS route confirmed; series CO2TIA assumed |
| ENV-05 | `coal_consumption` — total coal consumption (thousand short tons) | `coal/consumption-and-quality` route confirmed |
| ENV-06 | `petroleum_supplied` — petroleum products supplied (kbd) | `petroleum/cons` route confirmed |
| ENV-07 | `natural_gas_consumed` — NG total consumption (Bcf) | `natural-gas/cons` route confirmed |
| ENV-08 | `renewable_generation_share` — renewables % of total generation | `electricity/electric-power-operational-data` confirmed; starts 2001 |
| ENV-09 | `solar_generation` — solar generation (GWh) | Same route; fueltypeid=SUN; starts 2001 |
| ENV-10 | `wind_generation` — wind generation (GWh) | Same route; fueltypeid=WND; starts 2001 |
| ENV-11 | `nuclear_generation` — nuclear generation (GWh) | Same route; fueltypeid=NUC; starts ~1957 |
| ENV-12 | `renewable_installed_capacity` — renewable capacity (GW) | `electricity/operating-generator-capacity` confirmed |
| ENV-13 | `nuclear_outage_rate` — nuclear capacity outage rate (%) | `nuclear-outages/us-nuclear-outages` confirmed; daily frequency only — needs aggregation |
| ENV-14 | `total_electricity_demand` — total demand (GWh) | `electricity/rto` confirmed but daily/hourly only; alternative: total-energy MSN |
| ENV-15 | `biomass_production` — densified biomass (thousand tons) | `densified-biomass/production-by-region` confirmed; starts ~2016 — not suitable for 1960 training |
| ECON-01 | `retail_electricity_price` — avg retail price (cents/kWh) | `electricity/retail-sales` confirmed |
| ECON-02 | `natural_gas_price_consumer` — NG delivered price ($/Mcf) | `natural-gas/pri` route confirmed |
| ECON-03 | `gasoline_retail_price` — regular gasoline price ($/gallon) | `petroleum/pri` route confirmed |
| ECON-04 | `diesel_retail_price` — diesel price ($/gallon) | `petroleum/pri` route confirmed |
| ECON-05 | `coal_market_price` — avg coal price ($/short ton) | `coal/market-sales-price` route confirmed |
| ECON-06 | `energy_expenditure_per_household` — annual HH expenditure ($) | SEDS expenditures; ASSUMED series available |
| ECON-07 | `govt_energy_expenditures` — total govt energy expenditure ($B) | SEDS expenditures; availability LOW confidence |
| ECON-08 | `coal_employment` — coal mining employment (thousands) | `coal/aggregate-production` route confirmed |
| ECON-09 | `natural_gas_production` — NG marketed production (Bcf) | `natural-gas/prod` route confirmed |
| ECON-10 | `crude_oil_production` — crude oil production (kbd) | `petroleum/crd` route confirmed |
| ECON-11 | `crude_oil_imports` — crude oil imports (kbd) | `crude-oil-imports` top-level route confirmed |
| ECON-12 | `petroleum_exports` — total petroleum exports (kbd) | `petroleum/move` route confirmed |
| ECON-13 | `natural_gas_imports` — NG imports (Bcf) | `natural-gas/move` route confirmed |
| ECON-14 | `petroleum_stocks` — total petroleum stocks (million barrels) | `petroleum/stoc` route confirmed |
| ECON-15 | `energy_trade_balance` — net energy trade balance ($B) | Derived from petroleum/move + natural-gas/move; requires computation |
| ECON-16 | `steo_price_index` — STEO energy price index | `steo` route confirmed; 18-month forward-looking only — historical values are rolling vintages |
</phase_requirements>

---

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| requests | >=2.31 (2.32.5 installed) | EIA v2 HTTP calls | Direct REST; no wrapper libs (both `myeia` and `eiapy` rejected per CLAUDE.md) |
| pandas | >=2.0,<3.0 (2.3.3 installed) | EIA JSON → DataFrame → array | Standard tabular bridge for EIA response |
| numpy | >=1.26,<2.0 (1.26.0 on project Python) | Array ops, state vector construction | Hard pin <2.0 per CLAUDE.md — ABI break |
| scikit-learn | current (NOT installed) | RobustScaler fit | Industry standard for normalization |
| python-dotenv | >=1.0 | Load EIA_API_KEY from .env | Simplest hackathon config |

**Installation required for Phase 1:**
```bash
pip install scikit-learn python-dotenv
```

**numpy version warning:** `pip show numpy` shows 2.4.1 but the project Python 3.11 (Windows Store) has 1.26.0. Do NOT upgrade numpy for this Python; the 2.4.1 belongs to a different Python installation. Verify with `python -c "import numpy; print(numpy.__version__)"` — should print 1.26.0.

---

## EIA v2 API Route Map (Verified)

### API Base

```
https://api.eia.gov/v2/{route}/data/
?api_key=KEY
&frequency={annual|monthly|daily}
&data[]={column_name}
&facets[{facet_key}][]={value}
&start={YYYY}
&end={YYYY}
&sort[0][column]=period
&sort[0][direction]=asc
&offset={N}
&length={5000}
```

[VERIFIED: live API query `https://api.eia.gov/v2/?api_key=DEMO_KEY`]

### Top-Level Routes Available

| Route | Description | Note |
|-------|-------------|------|
| `coal` | EIA coal energy data | Active |
| `crude-oil-imports` | Crude oil shipments by country | Active |
| `electricity` | EIA electricity survey data | Active |
| `natural-gas` | EIA natural gas survey data | Active |
| `nuclear-outages` | EIA nuclear outages survey data | Active |
| `petroleum` | EIA petroleum data | Active |
| `seds` | State Energy Data System | Active — CO2 data lives here |
| `steo` | Short-Term Energy Outlook | Active — 18-month projections |
| `densified-biomass` | EIA densified biomass data | Active — starts ~2016 |
| `total-energy` | Comprehensive energy statistics | Active — annual back to 1949 |
| `aeo` | Annual Energy Outlook | Active |
| `co2-emissions` | State CO2 (deprecated) | **DEPRECATED** — do not use |

[VERIFIED: live API query `https://api.eia.gov/v2/?api_key=DEMO_KEY`]

### Petroleum Sub-Routes

| Sub-route | Purpose | Full Path |
|-----------|---------|-----------|
| `pri` | Prices | `petroleum/pri/...` |
| `crd` | Crude reserves and production | `petroleum/crd/...` |
| `move` | Imports/exports and movements | `petroleum/move/...` |
| `stoc` | Stocks | `petroleum/stoc/...` |
| `cons` | Consumption/sales | `petroleum/cons/...` |
| `sum` | Summary | `petroleum/sum/...` |

[VERIFIED: live API query `https://api.eia.gov/v2/petroleum/?api_key=DEMO_KEY`]

### Natural Gas Sub-Routes

| Sub-route | Purpose | Full Path |
|-----------|---------|-----------|
| `pri` | Prices | `natural-gas/pri/...` |
| `prod` | Production | `natural-gas/prod/...` |
| `move` | Imports/exports/pipelines | `natural-gas/move/...` |
| `cons` | Consumption/end use | `natural-gas/cons/...` |
| `stor` | Storage | `natural-gas/stor/...` |
| `enr` | Exploration and reserves | `natural-gas/enr/...` |

[VERIFIED: live API query `https://api.eia.gov/v2/natural-gas/?api_key=DEMO_KEY`]

### Electricity Sub-Routes

| Sub-route | Purpose |
|-----------|---------|
| `retail-sales` | Sales to ultimate customers (price, revenue, MWh) |
| `electric-power-operational-data` | Generation, consumption, stocks by fuel type and sector — **starts 2001** |
| `rto` | Real-time grid operations (daily/hourly only) |
| `operating-generator-capacity` | Inventory of operable generators |
| `state-electricity-profiles` | State-level electricity profiles |
| `facility-fuel` | Plant-level fuel and generation data |

[CITED: https://www.eia.gov/opendata/browser/electricity per web search results]

### SEDS Route (CO2 and Expenditures)

The `seds` route holds state-level energy data back to 1960, including CO2 emissions by sector (formerly in the now-deprecated `co2-emissions` route).

**Facets for SEDS:** `seriesId` (MSN code), `stateId` (2-letter state code or "US" for national)

**CO2 MSN Series Codes** [ASSUMED: based on EIA naming convention and technical note references; must be verified with real API key]:

| Variable | MSN Code | Description |
|----------|----------|-------------|
| `co2_total_emissions` | `CO2TCA` | Total CO2 all sectors (MMT CO2) |
| `co2_electric_power` | `CO2TEA` | Electric power sector CO2 (MMT CO2) |
| `co2_transportation` | `CO2TXA` | Transportation sector CO2 (MMT CO2) |
| `co2_industrial` | `CO2TIA` | Industrial sector CO2 (MMT CO2) |

**Alternative if SEDS MSN codes don't resolve:** Use `total-energy` route with MSN facet — e.g., `facets[msn][]=TETCBUS` for total CO2. The total-energy route has annual data back to 1949 and uses the same MSN naming. [CITED: EIA Total Energy Annual Data tables at eia.gov/totalenergy/data/annual/]

---

## Data Shape and Field Names

### EIA v2 Response Structure

```json
{
  "response": {
    "total": 247,
    "dateFormat": "YYYY",
    "frequency": "annual",
    "data": [
      {
        "period": "2005",
        "location": "US",
        "stateDescription": "U.S. Total",
        "sectorid": 99,
        "sectorDescription": "All Sectors",
        "fueltypeid": "SUN",
        "fuelTypeDescription": "solar",
        "generation": "6.225",
        "generation-units": "thousand megawatthours"
      }
    ]
  },
  "request": { ... },
  "apiVersion": "2.1.12"
}
```

Key points [VERIFIED: live API query to `electricity/electric-power-operational-data`]:
- Data is under `response.data` (not top-level)
- `total` tells total matching rows — compare to `len(data)` to detect truncation
- `period` is a string: `"2005"` for annual, `"2005-06"` for monthly
- **All data values are strings** since v2.1.6 (January 2024) — `"6.225"` not `6.225`
- Unit description appears alongside data column: `generation-units: "thousand megawatthours"`
- Facet dimension field names in response match what you pass as facets

### Value Parsing Pattern (Required)

```python
val = float(row.get("value", "nan"))  # or column-specific:
gen = float(row.get("generation", "nan"))
price = float(row.get("price", "nan"))
```

Never trust `row["value"]` as a numeric type. Always cast. [VERIFIED: CLAUDE.md + EIA changelog v2.1.6]

---

## EIA Pagination

**5,000 row limit:** Every EIA v2 data request returns at most 5,000 rows. For national annual series (1960–2023 = ~63 rows), this is never an issue. For state-level series (63 years × 50 states = ~3,150 rows), may be close to limit. For monthly series (60 years × 12 months = 720 rows national, 36,000 state-level), pagination is required.

**Detection:** [VERIFIED: EIA API documentation v2.1.0]
- Check `response.total` vs `len(response.data)` — if `total > len(data)`, truncation occurred
- HTTP response header `X-Warning` is present when results are truncated (added in v2.1.0 November 2022)

**Pagination pattern:**

```python
def fetch_all_pages(url_base: str, params: dict) -> list[dict]:
    all_data = []
    offset = 0
    while True:
        params["offset"] = offset
        params["length"] = 5000
        resp = requests.get(url_base, params=params, timeout=30)
        resp.raise_for_status()
        body = resp.json()["response"]
        page = body["data"]
        all_data.extend(page)
        if len(all_data) >= int(body["total"]):
            break
        offset += len(page)
    return all_data
```

**Rate limits:** EIA recommends staying below ~9,000 requests/hour and below 5 requests/second burst. With ~31 variables across ~10 distinct API calls and annual data only, rate limiting is not a concern for this project. [CITED: eia.gov/opendata/faqs.php]

---

## RobustScaler Implementation

### Fitted Attributes [VERIFIED: scikit-learn 1.8.0 official docs]

| Attribute | Type | Content |
|-----------|------|---------|
| `center_` | ndarray shape (n_features,) | Median of training data, per feature |
| `scale_` | ndarray shape (n_features,) | IQR (Q75-Q25) of training data, per feature |

**Transformation formula:** `x_scaled = (x - center_) / scale_`
**Inverse formula:** `x_original = x_scaled * scale_ + center_`

### JSON Serialization Pattern

`center_` and `scale_` are numpy arrays and must be converted to Python lists for JSON:

```python
import json
import numpy as np
from sklearn.preprocessing import RobustScaler

# Fit (training split only — 1960–2005)
scaler = RobustScaler()
scaler.fit(X_train)  # X_train shape: (n_years, 31)

# Serialize
params = {
    var_name: {
        "center": float(scaler.center_[i]),
        "scale": float(scaler.scale_[i])
    }
    for i, var_name in enumerate(VAR_NAMES)
}
with open("data/normalizer_params.json", "w") as f:
    json.dump(params, f, indent=2)
```

### Pure Numpy Inference (No sklearn at inference time)

```python
def normalize(state: np.ndarray, params: dict, var_names: list[str]) -> np.ndarray:
    centers = np.array([params[n]["center"] for n in var_names])
    scales  = np.array([params[n]["scale"] for n in var_names])
    return (state - centers) / scales

def denormalize(state: np.ndarray, params: dict, var_names: list[str]) -> np.ndarray:
    centers = np.array([params[n]["center"] for n in var_names])
    scales  = np.array([params[n]["scale"] for n in var_names])
    return state * scales + centers
```

This allows `src/normalizer.py` to operate without sklearn at inference time — only needed during the fit step.

### Round-Trip Test

```python
x_orig = np.array([state_values])
x_norm = normalize(x_orig, params, VAR_NAMES)
x_back = denormalize(x_norm, params, VAR_NAMES)
assert np.allclose(x_orig, x_back, rtol=1e-6), "Round-trip failed"
```

**Edge case:** If `scale_` for any variable is 0 (zero IQR — constant variable), division by zero occurs. Clamp `scale_` to a minimum of 1e-8 before serializing. [ASSUMED]

---

## Architecture Patterns

### System Architecture Diagram

```
[EIA API v2 (HTTPS)]
        |
        | GET /v2/{route}/data/ × N calls
        v
[src/data.py — fetch_eia()]
    paginate → raw records list
    float() cast all values
    aggregate to annual if monthly input
    handle NaN for missing values
        |
        v
[State Assembly — assemble_state()]
    Map fetched values → 31-dim np.float32 array
    Order strictly by VAR_INDEX
    Write data/sample_state.json (2023 baseline)
        |
    [FALLBACK path]
    |   If API key absent → load data/sample_state.json
    |
    v
[src/normalizer.py — fit_scaler()]
    Load historical data (1960–2005 only)
    Fit RobustScaler per variable
    Extract center_ + scale_
    Write data/normalizer_params.json
        |
        v
[src/normalizer.py — Normalizer class]
    load(normalizer_params.json)
    .transform(state: np.ndarray) → np.ndarray
    .inverse_transform(state: np.ndarray) → np.ndarray
    (pure numpy, no sklearn at inference)
        |
        v
[Downstream consumers]
    src/parser.py  — imports VAR_INDEX
    src/env.py     — imports VAR_INDEX, Normalizer
    src/model.py   — imports VAR_INDEX
    src/train.py   — imports data loader, Normalizer
```

### Recommended Project Structure for Phase 1

```
src/
├── data.py          # VAR_INDEX, fetch_eia(), assemble_state(), load_or_fetch()
└── normalizer.py    # Normalizer class (fit, transform, inverse_transform, save, load)
data/
├── sample_state.json          # Written by data.py --write-sample
├── normalizer_params.json     # Written by normalizer.py --fit
└── README.md                  # EIA download instructions (for TRAIN-05 later)
```

### VAR_INDEX Definition Pattern

```python
# src/data.py

VAR_INDEX: dict[str, int] = {
    # Environmental (0–14)
    "co2_total_emissions":         0,
    "co2_electric_power":          1,
    "co2_transportation":          2,
    "co2_industrial":              3,
    "coal_consumption":            4,
    "petroleum_supplied":          5,
    "natural_gas_consumed":        6,
    "renewable_generation_share":  7,
    "solar_generation":            8,
    "wind_generation":             9,
    "nuclear_generation":          10,
    "renewable_installed_capacity": 11,
    "nuclear_outage_rate":         12,
    "total_electricity_demand":    13,
    "biomass_production":          14,
    # Economic (15–30)
    "retail_electricity_price":    15,
    "natural_gas_price_consumer":  16,
    "gasoline_retail_price":       17,
    "diesel_retail_price":         18,
    "coal_market_price":           19,
    "energy_expenditure_per_household": 20,
    "govt_energy_expenditures":    21,
    "coal_employment":             22,
    "natural_gas_production":      23,
    "crude_oil_production":        24,
    "crude_oil_imports":           25,
    "petroleum_exports":           26,
    "natural_gas_imports":         27,
    "petroleum_stocks":            28,
    "energy_trade_balance":        29,
    "steo_price_index":            30,
}

# Validation at import time:
assert len(VAR_INDEX) == 31, "VAR_INDEX must have exactly 31 entries"
assert set(VAR_INDEX.values()) == set(range(31)), "VAR_INDEX must cover indices 0–30"
```

### `load_or_fetch()` Public Interface

```python
def load_or_fetch(
    api_key: str | None = None,
    year: int = 2023,
    write_sample: bool = False,
) -> np.ndarray:
    """
    Returns a (31,) float32 array of the 31 state variables for the given year.

    If api_key is None or fetch fails, falls back to data/sample_state.json.
    If write_sample=True and fetch succeeds, overwrites data/sample_state.json.
    """
```

### EIA Fetch Helper Pattern

```python
def fetch_eia(
    route: str,
    frequency: str,
    data_cols: list[str],
    facets: dict[str, list[str]],
    start: str,
    end: str,
    api_key: str,
) -> pd.DataFrame:
    url = f"https://api.eia.gov/v2/{route}/data/"
    params: dict = {
        "api_key": api_key,
        "frequency": frequency,
        "sort[0][column]": "period",
        "sort[0][direction]": "asc",
    }
    for col in data_cols:
        params.setdefault("data[]", []).append(col)
    for facet_key, vals in facets.items():
        for v in vals:
            params[f"facets[{facet_key}][]"] = v  # Note: requests handles repeated keys

    all_rows = []
    offset = 0
    while True:
        params["offset"] = offset
        params["length"] = 5000
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        body = resp.json()["response"]
        page = body["data"]
        all_rows.extend(page)
        if len(all_rows) >= int(body["total"]):
            break
        offset += len(page)

    df = pd.DataFrame(all_rows)
    for col in data_cols:
        if col in df.columns:
            df[col] = df[col].apply(lambda v: float(v) if v not in (None, "", "NA") else float("nan"))
    return df
```

---

## Variable-by-Variable Route Reference

### CO2 Variables (ENV-01 to ENV-04)

**Route:** `seds/data/`
**Frequency:** annual
**Facets:** `seriesId` (MSN code), `stateId`=US
**Data col:** `value`
**Units:** million metric tons CO2
**Coverage:** 1960–2023 [CITED: eia.gov/environment/emissions/state/ — "data for 1960 forward"]

| Variable | MSN Code | Confidence |
|----------|----------|------------|
| co2_total_emissions | CO2TCA | ASSUMED — verify with API key |
| co2_electric_power | CO2TEA | ASSUMED — verify with API key |
| co2_transportation | CO2TXA | ASSUMED — verify with API key |
| co2_industrial | CO2TIA | ASSUMED — verify with API key |

**Alternative if MSN codes wrong:** Use `total-energy` route with appropriate MSN facet values (TETCBUS pattern seen in EIA data browser URLs). [CITED: eia.gov/opendata URL structure]

### Coal Variables (ENV-05, ECON-05, ECON-08)

**Route:** `coal/consumption-and-quality/data/` (ENV-05), `coal/market-sales-price/data/` (ECON-05), `coal/aggregate-production/data/` (ECON-08)
**Frequency:** annual
**Coverage:** Multi-decade back to 1960s [ASSUMED — coal survey data predates modern API]
**Data cols:** `consumption` / `price` / `production` / `employees`

### Petroleum Variables (ENV-06, ECON-03, ECON-04, ECON-10, ECON-11, ECON-12, ECON-14, ECON-15)

| Variable | Route | Key Data Col |
|----------|-------|-------------|
| petroleum_supplied | `petroleum/cons/...` | `value` (kbd) |
| gasoline_retail_price | `petroleum/pri/...` | `value` ($/gallon) |
| diesel_retail_price | `petroleum/pri/...` | `value` ($/gallon) |
| crude_oil_production | `petroleum/crd/...` | `value` (kbd) |
| crude_oil_imports | `crude-oil-imports/data/` | `value` (kbd) |
| petroleum_exports | `petroleum/move/...` | `value` (kbd) |
| petroleum_stocks | `petroleum/stoc/...` | `value` (million barrels) |
| energy_trade_balance | derived: `petroleum/move` + `natural-gas/move` | computed |

**Coverage:** Crude oil and petroleum data generally available from 1949/1960 [CITED: EIA Total Energy Annual tables]

### Natural Gas Variables (ENV-07, ECON-02, ECON-09, ECON-13)

| Variable | Route | Key Data Col |
|----------|-------|-------------|
| natural_gas_consumed | `natural-gas/cons/...` | `value` (Bcf) |
| natural_gas_price_consumer | `natural-gas/pri/...` | `value` ($/Mcf) |
| natural_gas_production | `natural-gas/prod/...` | `value` (Bcf) |
| natural_gas_imports | `natural-gas/move/...` | `value` (Bcf) |

**Coverage:** Generally from 1960s/1970s [ASSUMED]

### Electricity Generation Variables (ENV-08 to ENV-11)

**Route:** `electricity/electric-power-operational-data/data/`
**Frequency:** annual (also monthly/quarterly available)
**Location facet:** `US` (national aggregate)
**Data col:** `generation` (thousand MWh — divide by 1000 for GWh)
**⚠️ CRITICAL: Data starts 2001** — no pre-2001 data in this route [CITED: API metadata confirms "2001-01 through 2026-01"]

| Variable | fueltypeid | sectorid |
|----------|-----------|---------|
| solar_generation | SUN | 99 (all sectors) |
| wind_generation | WND | 99 |
| nuclear_generation | NUC | 99 |
| renewable_generation_share | ALL + SUN + WND + HYC + GEO | computed from all-sector totals |

**Coverage gap strategy:** For 1960–2000 training data, use `total-energy` MSN series which go back to 1949:
- Solar: negligible before 1990, treat as 0 or use SEDS proxy
- Wind: negligible before 1990, treat as 0 or use SEDS proxy
- Nuclear: available via `total-energy` from 1957 [CITED: nuclear power plants operational from 1957]

### Electricity Capacity (ENV-12)

**Route:** `electricity/operating-generator-capacity/data/`
**Frequency:** annual
**Coverage:** Available from early 1990s [ASSUMED]

### Nuclear Outage Rate (ENV-13)

**Route:** `nuclear-outages/us-nuclear-outages/data/`
**⚠️ Data is DAILY only** — no annual frequency available in this route [CITED: EIA nuclear outages page describes daily plant status]
**Strategy:** Fetch weekly data and compute annual average outage rate (% capacity unavailable). This is a heavy fetch (52 weeks × N years). Alternative: compute from nuclear generation capacity vs actual generation from `electric-power-operational-data`.

**Alternative approach (recommended):** Derive `nuclear_outage_rate` from:
```
nuclear_outage_rate = 1 - (actual_nuclear_gen / nameplate_capacity)
```
Using nuclear generation (ENV-11 route) and operating capacity (ENV-12 route). This avoids the daily-data aggregation problem entirely.

### Total Electricity Demand (ENV-14)

**Route:** `electricity/rto/...` is hourly/daily only
**Alternative:** Use `total-energy` route with MSN for total electricity generation as proxy for demand (generation ≈ demand at annual scale).
**Or:** Use `electricity/electric-power-operational-data` with all-sector, all-fuel aggregation.
[ASSUMED — specific MSN code for total electricity demand TBD]

### Biomass Production (ENV-15)

**Route:** `densified-biomass/production-by-region/data/`
**⚠️ CRITICAL: Data starts ~2016** — Form EIA-63C survey launched January 2016 [CITED: search results confirming survey launch date]
**Impact:** Cannot be used for GRU training split (1960–2005 has zero data). For training: set biomass_production=0 for all pre-2016 records. For demo baseline (2023 value): will have real data.

### Retail Electricity Price (ECON-01)

**Route:** `electricity/retail-sales/data/`
**Frequency:** annual
**Facets:** sectorid=ALL, stateid=US
**Data col:** `price` (cents/kWh)

### SEDS Expenditures (ECON-06, ECON-07)

**Route:** `seds/data/`
**Frequency:** annual
**Data col:** `value`
**MSN codes for expenditures:** [ASSUMED — SEDS contains expenditure MSNs but specific codes need verification]

| Variable | Likely MSN | Confidence |
|----------|-----------|------------|
| energy_expenditure_per_household | ESRCHUS or similar | LOW — need API key to discover |
| govt_energy_expenditures | EFTCBUS or similar | LOW — need API key to discover |

**Fallback:** If specific expenditure MSN codes cannot be found, use SEDS total energy expenditure per capita or per household derived values.

### STEO Price Index (ECON-16)

**Route:** `steo/data/`
**⚠️ STEO is a rolling 18-month forward projection** — not historical data [CITED: EIA STEO description]
**Coverage:** Historical STEO vintages back to 1990 [CITED: search results]
**Special handling for baseline:** The 2023 state value is the most-recent STEO release's near-term price index. For GRU training, use the actual historical STEO-published values if available as "actuals", or substitute with a different energy price index from SEDS.
**MSN code:** Likely `PAPR_WORLD` or similar; exact code must be verified with API key [ASSUMED]

---

## Historical Data Availability Summary

| Variable Group | Route | Coverage Start | Notes |
|---------------|-------|---------------|-------|
| CO2 (all sectors) | seds | 1960 | Confirmed 1960 forward |
| Coal consumption/price | coal/* | ~1960 | EIA coal survey long-running |
| Petroleum consumption/price/production | petroleum/* | ~1949 | Total energy tables go to 1949 |
| Natural gas consumption/price/production | natural-gas/* | ~1960 | SEDS covers from 1960 |
| Nuclear generation | total-energy MSN | 1957 | Nuclear plants operational from 1957 |
| Solar/wind generation | electric-power-operational-data | **2001** | Pre-2001: use 0 or proxy |
| Renewable capacity | operating-generator-capacity | ~1990 | EIA-860 form starts 1990 |
| Nuclear outage rate | Derived from gen/capacity | 2001+ | Or nuclear-outages daily |
| Total electricity demand | total-energy MSN | ~1949 | Proxy via generation |
| Biomass production | densified-biomass | **~2016** | New survey; pre-2016 = 0 |
| Retail electricity price | electricity/retail-sales | ~1960 | SEDS price data back to 1960 |
| NG/gasoline/diesel prices | petroleum/pri, natural-gas/pri | ~1974 | Oil embargo era onwards |
| Coal employment | coal/aggregate-production | ~1970s | [ASSUMED] |
| Crude oil/NG production, imports | petroleum/crd, crude-oil-imports | ~1960 | [ASSUMED] |
| STEO price index | steo | 1990 | [CITED: search results] |

**Training split implications (1960–2005):**
- Solar/wind (2001+): Zero-fill for 1960–2000, real values 2001–2005 — clearly labeled
- Biomass (2016+): Zero-fill for 1960–2005 entirely — labeled in README
- GRU will learn these as near-constant zero for most of training range — acceptable

---

## normalizer_params.json Schema

```json
{
  "co2_total_emissions": {
    "center": 5234.7,
    "scale": 312.4
  },
  "co2_electric_power": {
    "center": 2145.2,
    "scale": 198.1
  },
  "...": "one entry per VAR_INDEX key"
}
```

**Schema rules:**
- Exactly 31 keys, matching VAR_INDEX names exactly
- `center` = float (median of training split 1960–2005)
- `scale` = float (IQR of training split 1960–2005, clamped ≥1e-8)
- Order irrelevant (load by key name)

---

## sample_state.json Schema

```json
{
  "year": 2023,
  "source": "EIA API v2",
  "fetched": "2026-04-18",
  "state": {
    "co2_total_emissions": 4874.3,
    "co2_electric_power": 1541.2,
    "...": "one entry per VAR_INDEX key"
  }
}
```

**Fallback loading pattern:**

```python
def _load_sample_state() -> np.ndarray:
    path = Path(__file__).parent.parent / "data" / "sample_state.json"
    with open(path) as f:
        data = json.load(f)
    state = np.zeros(31, dtype=np.float32)
    for name, idx in VAR_INDEX.items():
        state[idx] = float(data["state"].get(name, float("nan")))
    return state
```

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Median/IQR scaling | Custom scaler class | `sklearn.preprocessing.RobustScaler` | Handles edge cases (zero IQR, sparse data) |
| HTTP retry/timeout | Custom retry loop | `requests` with `timeout=30` + try/except | Built-in timeout handling |
| EIA pagination | First-page-only fetch | `fetch_all_pages()` using `total` field | Silent truncation without pagination |
| JSON type coercion | Assume numeric types | Always `float(row.get(col, "nan"))` | EIA changed all values to strings in Jan 2024 |
| Scaler re-fit at inference | Re-run fit step | Load `normalizer_params.json` + pure numpy | Ensures consistency with training |

**Key insight:** The only non-trivial custom implementation needed is the multi-route state assembly — mapping 10+ EIA fetch results into a consistent 31-dim array. Everything else has a library solution.

---

## Common Pitfalls

### Pitfall 1: Using Deprecated `co2-emissions` Route

**What goes wrong:** `https://api.eia.gov/v2/co2-emissions/...` returns no new data — the route is deprecated and officially directs users to SEDS.

**Why it happens:** REQUIREMENTS.md mentions "SEDS / electric-power-operational-data" for CO2 variables but the deprecated route name is tempting.

**How to avoid:** Always use `seds/data/` with `facets[seriesId][]=CO2TCA` (and related MSN codes) for CO2 data. Never use the `co2-emissions` top-level route.

**Warning signs:** Route returns data that stops at a date several years ago.

### Pitfall 2: Missing the 2001 Start Date for Generation-by-Fuel

**What goes wrong:** Fetching `electricity/electric-power-operational-data` with `start=1960` silently returns only 2001+ data. Assembling the training matrix with expected shape (46, 31) for 1960–2005 produces a partially-populated array with NaN for solar/wind/renewable columns for 1960–2000.

**How to avoid:** Zero-fill pre-2001 values for solar, wind, and renewable share. Document this in the data/README.md. Do NOT use `ffill` or `bfill` — zeros are the physically correct value for pre-commercial-solar era.

**Warning signs:** Training CSV has 46 rows for 1960–2005 but solar/wind columns have only 5 non-zero rows (2001–2005).

### Pitfall 3: String Type Propagation from EIA Response

**What goes wrong:** Building a pandas DataFrame directly from `response.data` and then calling `.values` produces a string-typed array that passes `pd.DataFrame()` construction but crashes silently in numpy arithmetic.

**How to avoid:** Cast inside the fetch function, not after. Apply `float(row.get(col, "nan"))` to each data column immediately when converting the response JSON.

**Warning signs:** `df.dtypes` shows `object` for data columns; `np.isfinite(arr)` raises `TypeError`.

### Pitfall 4: Fetching Monthly Data Without Annual Aggregation

**What goes wrong:** Several EIA routes offer monthly data that must be aggregated to annual. Requesting `frequency=annual` sometimes isn't available for a route, so the code falls back to monthly but forgets to aggregate — producing a 12× larger dataset with the wrong frequency.

**How to avoid:** Explicitly request `frequency=annual` for all Phase 1 fetches. If a route doesn't support annual, aggregate with: means for rates/prices, sums for volumes.

**Warning signs:** Training CSV has ~720 rows instead of ~60 for a US-national series.

### Pitfall 5: Zero-Scale Variables Causing Division by Zero

**What goes wrong:** Variables that were constant or near-constant during 1960–2005 (e.g., solar generation = 0 for most of the period, biomass = 0 for entire period) produce `scale_` (IQR) = 0. RobustScaler raises `RuntimeWarning` and produces `inf` normalized values.

**How to avoid:** After fitting RobustScaler, check `(scaler.scale_ == 0).any()`. For zero-scale variables, substitute scale with 1.0 (identity transform) or a small constant (1e-8). Document which variables have zero IQR in the training split.

**Warning signs:** `normalizer_params.json` has `"scale": 0.0` for any variable; `normalize()` returns `inf` or `nan` values.

### Pitfall 6: nuclear_outage_rate Has No Native Annual Data

**What goes wrong:** The `nuclear-outages/us-nuclear-outages` route only has daily data. Fetching for "annual" frequency returns nothing or errors.

**How to avoid:** Derive `nuclear_outage_rate` from existing annual data: `1 - (nuclear_generation_GWh / (nuclear_capacity_GW × 8760 hours))`. This is a computed variable — document that it is not directly fetched.

### Pitfall 7: STEO Is Forward-Looking, Not Historical

**What goes wrong:** The STEO route returns energy price projections (18 months forward). For training historical data from 1960–2005, STEO provides no actual historical values.

**How to avoid:** For ECON-16 (`steo_price_index`), use STEO only for the baseline state (2023 value = current forecast). For training data, use a historical energy price index from SEDS/total-energy as a proxy, or drop ECON-16 from training input and fill with actual-at-time values.

---

## Code Examples

### Fetch EIA with Pagination

```python
# Source: EIA API v2 documentation + verified via live queries
import requests
import pandas as pd

def fetch_eia(route: str, frequency: str, data_cols: list[str],
              facets: dict[str, list[str]], start: str, end: str,
              api_key: str) -> pd.DataFrame:
    url = f"https://api.eia.gov/v2/{route}/data/"
    base_params = {
        "api_key": api_key,
        "frequency": frequency,
        "sort[0][column]": "period",
        "sort[0][direction]": "asc",
    }
    # Build facet params (requests handles list values correctly when passed as list)
    for col in data_cols:
        base_params[f"data[]"] = col  # Will be overwritten — use list format below

    # Use list of tuples for repeated keys
    query_params = [
        ("api_key", api_key),
        ("frequency", frequency),
        ("start", start),
        ("end", end),
        ("sort[0][column]", "period"),
        ("sort[0][direction]", "asc"),
    ]
    for col in data_cols:
        query_params.append(("data[]", col))
    for facet_key, vals in facets.items():
        for v in vals:
            query_params.append((f"facets[{facet_key}][]", v))

    all_rows = []
    offset = 0
    while True:
        page_params = query_params + [("offset", offset), ("length", 5000)]
        resp = requests.get(url, params=page_params, timeout=30)
        resp.raise_for_status()
        body = resp.json()["response"]
        page = body["data"]
        all_rows.extend(page)
        if len(all_rows) >= int(body["total"]):
            break
        offset += len(page)
        if not page:
            break  # Safety: no data in page

    df = pd.DataFrame(all_rows)
    for col in data_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df
```

### RobustScaler Fit and Serialize

```python
# Source: scikit-learn 1.8.0 official docs (verified)
import numpy as np
import json
from sklearn.preprocessing import RobustScaler

def fit_and_save(X_train: np.ndarray, var_names: list[str], path: str) -> dict:
    """Fit per-variable RobustScaler on training split, save params to JSON."""
    params = {}
    for i, name in enumerate(var_names):
        col = X_train[:, i].reshape(-1, 1)
        scaler = RobustScaler()
        scaler.fit(col)
        center = float(scaler.center_[0])
        scale = float(scaler.scale_[0])
        if scale == 0.0 or np.isnan(scale):
            scale = 1.0  # Identity transform for constant variables
        params[name] = {"center": center, "scale": scale}
    with open(path, "w") as f:
        json.dump(params, f, indent=2)
    return params
```

### Normalize / Denormalize (Pure Numpy)

```python
# Source: derived from RobustScaler formula (x - center) / scale
import numpy as np
import json

class Normalizer:
    def __init__(self, params_path: str):
        with open(params_path) as f:
            self._params = json.load(f)
        # Build ordered arrays matching VAR_INDEX order
        from src.data import VAR_INDEX
        names = sorted(VAR_INDEX.keys(), key=lambda n: VAR_INDEX[n])
        self._centers = np.array([self._params[n]["center"] for n in names], dtype=np.float32)
        self._scales  = np.array([self._params[n]["scale"] for n in names], dtype=np.float32)

    def transform(self, state: np.ndarray) -> np.ndarray:
        return ((state - self._centers) / self._scales).astype(np.float32)

    def inverse_transform(self, state: np.ndarray) -> np.ndarray:
        return (state * self._scales + self._centers).astype(np.float32)
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `co2-emissions` API route | `seds` route with MSN codes | 2023 | All CO2 data must use SEDS |
| EIA v1 API (`eiapy` library) | EIA v2 REST API (direct requests) | 2022 | v1 routes deprecated; no wrapper libs |
| Numeric JSON values | String-typed JSON values | Jan 2024 (v2.1.6) | Must cast with `float()` |
| No truncation warning | `X-Warning` header + `total` field | Nov 2022 (v2.1.0) | Pagination detection available |

**Deprecated/outdated:**
- `co2-emissions` route: deprecated, no new data, redirects to SEDS
- EIA v1 API: functionally deprecated; all new development should use v2
- `myeia` Python library: marked inactive (Snyk), sole maintainer, 21 stars
- `eiapy` Python library: targets v1 API only

---

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Python 3.11 | Runtime | ✓ | 3.11.9 | — |
| requests | EIA HTTP calls | ✓ | 2.32.5 | — |
| pandas | Data wrangling | ✓ | 2.3.3 | — |
| numpy | Array ops | ✓ | 1.26.0 (correct) | — |
| scikit-learn | RobustScaler fit | ✗ | NOT INSTALLED | — |
| python-dotenv | API key loading | ✗ | NOT INSTALLED | os.environ fallback |
| EIA API key | Live data fetch | ✗ (unknown) | — | sample_state.json fallback |

**⚠️ numpy version conflict:** `pip show numpy` reports 2.4.1 (different Python install), but project Python 3.11 has 1.26.0. The correct version is installed on the project Python. Do NOT run `pip install numpy` to "fix" this — it may upgrade the project numpy to 2.x and break PyTorch.

**Missing dependencies with no fallback:**
- `scikit-learn`: Required for `normalizer.py --fit` (training-time only). Install: `pip install scikit-learn`
- `python-dotenv`: Required for clean API key loading. Install: `pip install python-dotenv`

**Missing dependencies with fallback:**
- EIA API key: `load_or_fetch()` falls back to `data/sample_state.json`. The sample state JSON must be created manually (hardcoded 2023 values) before Phase 3 demo.

**Also needed for Phase 1 but not Phase 1 blocker:**
- plotly 6.6.0 installed — CLAUDE.md requires <6.0. Must be downgraded before Phase 3 (UI).
- streamlit 1.55.0 installed — CLAUDE.md requires 1.40–1.52. Must be downgraded before Phase 3.

---

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | SEDS MSN code for total CO2 is `CO2TCA` | EIA Route Map | Fetch returns empty; must query SEDS facets endpoint to discover actual codes |
| A2 | SEDS MSN code for electric power CO2 is `CO2TEA` | EIA Route Map | Same as A1 |
| A3 | SEDS MSN code for transportation CO2 is `CO2TXA` | EIA Route Map | Same as A1 |
| A4 | SEDS MSN code for industrial CO2 is `CO2TIA` | EIA Route Map | Same as A1 |
| A5 | SEDS has MSN codes for household and govt energy expenditures | Variable-by-Variable | ECON-06 and ECON-07 may need different route or approximation |
| A6 | Coal employment data available in `coal/aggregate-production` back to 1970s | Historical Data | May need to check `total-energy` route for pre-1990 employment data |
| A7 | natural-gas and petroleum routes have annual data from 1960 | Historical Data | Some sub-routes may only start in 1970s/1980s |
| A8 | STEO route has historical vintage data back to 1990 | Variable-by-Variable | Training proxy for ECON-16 may need different series |
| A9 | `scale_=0` edge case for biomass (all zeros in training) requires manual scale=1.0 | Normalizer | Confirmed logic — risk is in implementation |
| A10 | fueltypeid codes SUN, WND, NUC are valid for `electric-power-operational-data` | Variable-by-Variable | Codes may differ; use API facet metadata query to discover |

**Mitigation for A1–A4 (CRITICAL):** First task in Phase 1 execution must be: query `https://api.eia.gov/v2/seds/facet/seriesId/?api_key=KEY` to retrieve all available MSN codes. Filter for CO2-related codes. This takes 1 API call and resolves all CO2 series uncertainty.

---

## Open Questions

1. **SEDS CO2 MSN codes (A1–A4)**
   - What we know: SEDS holds CO2 data, redirected from deprecated co2-emissions route; codes follow MSN naming convention
   - What's unclear: Exact 5-character MSN codes for total, electric power, transportation, industrial CO2
   - Recommendation: First execution task — query SEDS facet endpoint for seriesId values

2. **ECON-06/ECON-07 expenditure series**
   - What we know: SEDS covers "production, consumption, price, and expenditure data"
   - What's unclear: Whether per-household and govt expenditure MSNs exist or need derivation
   - Recommendation: Query SEDS facet endpoint for seriesId values containing "expenditure" — fall back to SEDS total expenditure per capita

3. **ENV-14 (total_electricity_demand) annual series**
   - What we know: `electricity/rto` is hourly/daily; SEDS/total-energy may have annual demand proxy
   - What's unclear: Exact MSN or route for US annual total electricity demand back to 1960
   - Recommendation: Use total net generation from `total-energy` route as proxy (generation ≈ demand annually)

4. **Coal employment pre-1990 availability**
   - What we know: EIA has coal survey data; `coal/aggregate-production` route confirmed
   - What's unclear: How far back coal employment data is available (1970? 1980?)
   - Recommendation: Check `coal/aggregate-production` facets on first API call; fall back to SEDS

5. **physical bounds for ENV-RL-06 (Phase 3)**
   - What we know: Physical bounds must be derived from EIA historical range during Phase 1
   - What's unclear: Exact min/max for all 31 variables — must be computed from fetched data
   - Recommendation: After fetching all historical data (1960–2023), compute per-variable min/max with 10% buffer. Export as `data/physical_bounds.json` as Phase 1 deliverable (even though consumed in Phase 3).

---

## Sources

### Primary (HIGH confidence)
- EIA API v2 live response (`api.eia.gov/v2/?api_key=DEMO_KEY`) — top-level routes and deprecation status
- EIA API v2 live response (`api.eia.gov/v2/petroleum/?api_key=DEMO_KEY`) — petroleum sub-routes
- EIA API v2 live response (`api.eia.gov/v2/natural-gas/?api_key=DEMO_KEY`) — natural gas sub-routes
- EIA API v2 live response (`api.eia.gov/v2/electricity/electric-power-operational-data/data/`) — field names, value types, 2001 start date
- scikit-learn 1.8.0 official docs (`scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html`) — center_, scale_ attributes

### Secondary (MEDIUM confidence)
- EIA Open Data FAQs (`eia.gov/opendata/faqs.php`) — pagination limits, rate limits, X-Warning header
- EIA environment/emissions page (`eia.gov/environment/emissions/state/`) — SEDS CO2 coverage 1960 forward
- EIA API v2.1.0 documentation PDF — pagination and warning headers
- EIA total energy annual tables (`eia.gov/totalenergy/data/annual/`) — sector CO2 table structure
- Web search results confirming densified-biomass survey start ~2016

### Tertiary (LOW confidence — verify during execution)
- SEDS CO2 MSN codes (CO2TCA, CO2TEA, CO2TXA, CO2TIA) — inferred from naming convention
- STEO historical coverage back to 1990 — single source reference
- Coal employment data availability — assumed from general EIA coal survey history

---

## Metadata

**Confidence breakdown:**
- EIA route structure (top-level + sub-routes): HIGH — verified via live API
- EIA data field names and string type: HIGH — verified via live API query + official changelog
- EIA pagination mechanics: HIGH — official docs + FAQs
- SEDS CO2 MSN series codes: LOW — naming convention assumption only
- RobustScaler API and serialization: HIGH — official scikit-learn docs
- Historical data coverage: MEDIUM — most confirmed, some ASSUMED
- Expenditure series availability: LOW — SEDS documentation is sparse

**Research date:** 2026-04-18
**Valid until:** 2026-05-18 (EIA API structure is stable; MSN codes change rarely)
