# Data Center Energy Impact Analysis

A comprehensive project for scraping, analyzing, and predicting the energy impact of data centers across the United States. This project collects data from 4,592+ data centers, estimates their electricity consumption using physics-based models, and trains machine learning models to understand their impact on regional electricity demand.

## 🔍 Key Findings

- **96% DC Signal**: XGBoost BA-level model explains 96% of variance with data center features alone
- **R² = 0.997**: Extremely high accuracy in predicting regional electricity consumption  
- **92% dc_count**: Data center count alone explains 92% of regional electricity variance
- **4,592 Data Centers**: Comprehensive database scraped from datacentermap.com
- **9.7% CAGR**: Data center capacity growing 9.7% annually (2019-2024)

---

## 📁 Project Structure

```
env_forecasting/
├── scrapers/                    # Web scraping modules
│   ├── usa_scraper.py          # Scrapes main USA page
│   ├── state_scraper.py        # Scrapes each state page
│   ├── city_scraper.py         # Scrapes city-level data
│   ├── specs_scraper.py        # Scrapes DC specifications
│   ├── abilene_scraper.py      # Abilene-specific scraper
│   └── texas_scraper.py        # Texas-specific scraper
│
├── parsers/                     # Data extraction & parsing
│   ├── parse_state_links.py    # Extracts state URLs from HTML
│   ├── parse_city_links.py     # Extracts city URLs
│   ├── parse_specs_to_csv.py   # Converts specs to CSV
│   ├── extract_datacenters.py  # Extracts DC data from pages
│   ├── extract_specs.py        # Extracts DC specifications
│   └── merge_datacenter_specs.py # Merges DC specs with locations
│
├── models/                      # Machine learning models
│   ├── scripts/                # Training scripts
│   │   ├── ba_multiyear_predictor.py   # BA-level multi-year model (best)
│   │   ├── ba_level_predictor.py       # BA-level single-year model
│   │   ├── electricity_predictor.py    # State-level predictor
│   │   ├── ai_datacenter_model.py      # AI DC classification
│   │   ├── datacenter_energy_model.py  # Physics-based energy model
│   │   ├── energy_model_v5_real.py     # Production energy model
│   │   ├── enrich_and_train_ml.py      # Feature enrichment
│   │   ├── granular_predictor.py       # Sub-state analysis
│   │   └── download_eia_data.py        # EIA data downloader
│   └── trained/                # Saved model files (.joblib)
│       ├── energy_model_v5_real.joblib
│       └── ai_dc_prediction_model.joblib
│
├── data/                        # All datasets
│   ├── raw/                    # Raw data
│   │   ├── html/              # Scraped HTML files
│   │   │   ├── usa.txt        # Main datacentermap.com page
│   │   │   ├── state_links.txt
│   │   │   └── state/         # Per-state HTML files
│   │   │       └── {state}/city/{city}/dc/{dc-name}/
│   │   └── csv/               # Initial CSVs
│   │       └── data_centers.csv
│   ├── processed/              # Cleaned & processed data
│   │   ├── datacenter_specs.csv
│   │   ├── datacenter_enriched.csv
│   │   ├── datacenter_energy_estimates_v5.csv
│   │   ├── training_data_real_eia.csv
│   │   └── ml_features.csv
│   ├── eia/                    # EIA-861 electricity data
│   │   ├── 2019/
│   │   ├── 2020/
│   │   ├── 2021/
│   │   ├── 2022/
│   │   ├── 2023/
│   │   └── 2024/
│   └── outputs/                # Model outputs & results
│       ├── ba_multiyear_model_results.json
│       └── usa_datacenters_twbx.twbx
│
├── website/                     # Interactive web application
│   ├── index.html              # Interactive US map
│   ├── report.html             # Technical report
│   ├── datacenters.json        # DC data for map
│   ├── state_data.json         # State aggregations
│   └── generate_state_data.py  # Data generator
│
└── README.md
```

---

## ⚡ Quick Start (For Teammates)

```bash
# Clone and run with ONE COMMAND
git clone https://github.com/wijayaju/env_forecasting.git
cd env_forecasting
make start
```

That's it! Opens at http://localhost:8080

### Other Commands
| Command | What it does |
|---------|-------------|
| `make start` | **ONE COMMAND** - installs deps + starts website |
| `make install` | Install all Python dependencies |
| `make website` | Start local website at http://localhost:8080 |
| `make model` | Run the XGBoost prediction model |
| `make scrape` | Run the DC scraper (~2 hours) |
| `make test` | Check all dependencies are installed |
| `make clean` | Remove generated cache files |

---

## 🚀 Installation (Manual)

### Prerequisites

```bash
# Python 3.10+
pip install -r requirements.txt
```

### Clone & Setup

```bash
git clone https://github.com/wijayaju/env_forecasting.git
cd env_forecasting
pip install -r requirements.txt
```

---

## 🕷️ Web Scraping Pipeline

The scrapers collect data center information from [datacentermap.com](https://www.datacentermap.com/usa/) in a hierarchical manner.

### Step 1: Scrape USA Page
```bash
cd scrapers
python usa_scraper.py
```
This scrapes the main USA data centers page and saves it to `data/raw/html/usa.txt`.

### Step 2: Parse State Links
```bash
cd parsers
python parse_state_links.py
```
Extracts links to all 50 state pages from the USA page.

### Step 3: Scrape Each State
```bash
cd scrapers
python state_scraper.py
```
Iterates through all states and saves each state's HTML to `data/raw/html/state/{state-name}/{state-name}.txt`.

### Step 4: Parse City Links
```bash
cd parsers
python parse_city_links.py
```
Extracts city URLs from each state page.

### Step 5: Scrape Cities
```bash
cd scrapers
python city_scraper.py
```
Scrapes each city page for data center listings.

### Step 6: Scrape Data Center Specs
```bash
cd scrapers
python specs_scraper.py
```
This is the main scraper that visits each data center's `/specs/` page to extract detailed specifications:
- **Power capacity** (MW)
- **Total area** (sq ft)
- **Opened date**
- **Operator/Owner**
- **Coordinates** (lat/long)

**Rate Limiting Features:**
- Configurable delay between requests (default: 3s + random jitter)
- Automatic retry on rate limiting
- Resumable: skips already-scraped pages
- Rotates user agents

### Step 7: Convert to CSV
```bash
cd parsers
python parse_specs_to_csv.py
```
Parses all scraped specs into `data/processed/datacenter_specs.csv`.

---

## 🧠 Machine Learning Pipeline

### Data Enrichment

```bash
cd models/scripts
python enrich_and_train_ml.py
```

This script:
1. Loads raw datacenter data
2. Geocodes missing coordinates
3. Categorizes DCs (AI, Crypto, Enterprise, Colocation)
4. Calculates physics-based energy estimates
5. Creates training features for ML models

### Physics-Based Energy Model

The energy model uses first-principles physics to estimate power consumption:

```
Power (MW) = Area × PUE × Power_Density × Utilization
```

Where:
- **Power_Density**: 150-500 W/sq ft depending on DC type (AI DCs up to 500W/sq ft)
- **PUE**: Power Usage Effectiveness (1.2-1.6 typical)
- **Utilization**: Capacity utilization factor (0.6-0.8)

AI/GPU data centers have dramatically higher power density than traditional colocation facilities.

### Training the Predictor

#### State-Level Model (Baseline)
```bash
python electricity_predictor.py
```
- Uses EIA state electricity data
- R² = 0.978, but only 3% DC signal (population dominates)

#### BA-Level Single-Year Model
```bash
python ba_level_predictor.py
```
- Uses Balancing Authority (BA) level EIA data
- R² = 0.952, 54% DC signal

#### BA-Level Multi-Year Model (Best)
```bash
python ba_multiyear_predictor.py
```
- Uses 6 years of BA-level data (2019-2024)
- 326 observations (55 BAs × 6 years)
- R² = 0.997, **96% DC signal** (XGBoost)
- Key insight: dc_count explains 92% of BA electricity variance

### Model Results Summary

| Model | R² | DC Signal | Observations |
|-------|-----|-----------|--------------|
| State-Level | 0.978 | 3% | 51 |
| BA Single-Year | 0.952 | 54% | 54 |
| **BA Multi-Year (XGBoost)** | **0.997** | **96%** | 326 |

The project uses EIA-861 Form data for ground-truth electricity consumption.

### Download EIA Data
```bash
cd models/scripts
python download_eia_data.py
```

Downloads annual utility-level sales data from [EIA](https://www.eia.gov/electricity/data/eia861/).

### Data Files Used
- `Sales_Ult_Cust_{year}.xlsx` - Utility sales by state/BA
- `Balancing_Authority_{year}.xlsx` - BA assignments

---

## 🌐 Website / Visualization

### Running the Website
```bash
cd website
python -m http.server 8080
```
Then open `http://localhost:8080` in your browser.

### Features
- **Interactive Map**: Click on states to see data center counts and energy estimates
- **Energy Calculator**: Physics-based calculator for individual DCs
- **Technical Report**: Full methodology documentation with charts

### Regenerating Data Files
```bash
cd website
python generate_state_data.py
```
Creates `state_data.json` with aggregated statistics per state.

---

## 📈 Key Metrics

### Data Center Coverage
- **4,592** total data centers scraped
- **434** have power capacity data
- **2,100+** have area (sq ft) data

### Energy Impact
- **Estimated Total DC Power**: 18.2 GW
- **Annual Electricity**: ~160 TWh (4% of US total)
- **Top State**: Virginia (highest concentration, 500+ DCs)

### Model Performance
- **Best Model**: XGBoost Regressor (BA Multi-Year)
- **Cross-Validation R²**: 0.73 ± 0.07
- **Test Set R²**: 0.997
- **Feature Importance**: dc_count (92%), utility_count (4%)

---

## 🛠️ Configuration

### Scraper Settings (in `scrapers/specs_scraper.py`)
```python
REQUEST_DELAY = 3           # Seconds between requests
JITTER_MAX = 1              # Random jitter (0-1 seconds)
RATE_LIMIT_WAIT = 0.25      # Minutes to wait if rate limited
MAX_RATE_LIMIT_RETRIES = 10 # Max retries before giving up
```

### Model Parameters
```python
# XGBoost Regressor
import xgboost as xgb

xgb.XGBRegressor(
    n_estimators=150,
    max_depth=4,
    learning_rate=0.1,
    min_child_weight=5,
    objective='reg:squarederror',
    random_state=42
)
```

---

## 📚 Data Sources

1. **Data Center Locations**: [datacentermap.com](https://www.datacentermap.com/)
2. **Electricity Data**: [EIA-861](https://www.eia.gov/electricity/data/eia861/)

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/new-feature`
3. Commit changes: `git commit -am 'Add new feature'`
4. Push: `git push origin feature/new-feature`
5. Submit a Pull Request

---

## 📄 License

This project is for educational purposes (AI Club Advanced Project).

---

## 👥 Authors

- Michigan State University - AI Club
