# Air Quality LATAM

**Open-source framework for air quality analysis in data-scarce regions of Latin America**

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![OpenAQ](https://img.shields.io/badge/Data-OpenAQ-teal.svg)](https://openaq.org)
[![NASA](https://img.shields.io/badge/Data-NASA%20EarthData-orange.svg)](https://earthdata.nasa.gov)

---

## Why this project exists

Air quality monitoring in Latin America is severely limited. Countries like Guatemala have fewer than 5 official monitoring stations for a population of 17 million people. Existing tools are built for data-rich environments — dense sensor networks, clean APIs, standardized formats.

This project was built for the opposite scenario: **regions where data is scarce, fragmented, or simply missing.**

We combine ground-level station data with satellite remote sensing (NASA MERRA-2, MODIS AOD, Copernicus CAMS) to produce air quality analyses where traditional monitoring falls short.

---

## What it does

- Ingests data from local stations (CSV/Excel), OpenAQ API, NASA EarthData, and Copernicus CAMS
- Cleans, standardizes, and validates multi-source air quality data
- Computes EPA AQI, IMECA, and WHO 2021 standards
- Performs spatial interpolation (IDW, kriging) to fill monitoring gaps
- Generates interactive maps, time series analysis, and correlation studies
- Produces automated PDF reports by region and time period

---

## Pollutants covered

PM2.5 · PM10 · O3 · NO2 · CO · SO2 · Saharan dust (transoceanic transport)

---

## Target regions

- Guatemala (primary focus)
- Mexico — underserved states
- Expanding to Central America and broader LATAM

---

## Data sources

| Source | Type | Coverage |
|--------|------|----------|
| MARN Guatemala | Station data | Guatemala |
| SEMARNAT / SINAICA | Station data | Mexico |
| OpenAQ API | Multi-source aggregator | Global |
| NASA MERRA-2 | Reanalysis | Global |
| MODIS AOD | Satellite aerosol | Global |
| Copernicus CAMS | Reanalysis | Global |

---

## Quickstart

```bash
git clone https://github.com/rodrigopell/air-quality-latam.git
cd air-quality-latam
pip install -r requirements.txt

# Run with synthetic Guatemala data
python 01_ingestion/load_stations.py

# Launch interactive dashboard
streamlit run 04_visualization/dashboard.py
```

Open http://localhost:8501 in your browser.

---

## Project structure

```
air-quality-latam/
├── 01_ingestion/      # Data loaders: stations, OpenAQ, NASA, Copernicus
├── 02_processing/     # Cleaning, spatial merge, AQI calculation
├── 03_analysis/       # Spatial, descriptive, time series, correlations
├── 04_visualization/  # Maps, dashboard, automated reports
├── data/              # Raw and processed data
├── outputs/           # Figures, maps, PDF reports
└── notebooks/         # Exploratory analysis
```

---

## Contributing

Contributions welcome — especially from researchers, environmental engineers, and data scientists working in data-scarce regions. Open an issue or submit a PR.

---

## Author

Rodrigo Pellecer · Environmental Engineer  
Guatemala · rodrigo.pellecergodoy@gmail.com

---

## License

MIT