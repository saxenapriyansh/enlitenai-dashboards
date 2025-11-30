# 🏥 Seizure Analytics Dashboard

A comprehensive, interactive Streamlit application for analyzing 52-week seizure, medication, and assessment data.

## 🚀 Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### Running the Application
```bash
streamlit run app.py
```

Open your browser to `http://localhost:8501`

### Data Options
- **Option A**: Use pre-loaded demo data (default)
- **Option B**: Upload your own three CSV files (seizures, medications, assessments)

## 📊 Dashboard Sections

### 1. Overview Dashboard
Summary KPIs and quick visualizations showing total seizures, weekly averages, and key statistics.

### 2. Seizure Activity
- Daily and weekly seizure timelines
- 52-week × 7-day calendar heatmap
- Trend analysis and pattern identification

### 3. Medication Dashboard
- Dose trajectories for 5 medications
- Automatic dose change detection
- Medication-seizure correlation analysis

### 4. Mental Health & QoL
- Quality of Life, Anxiety, Depression, and Behavioral assessments
- QoL vs Seizures comparison
- Patient-reported outcome tracking

### 5. Integrated Clinical View
Multi-tab interface with disease activity, medication changes, patient outcomes, and correlation analysis.

### 6. TachyGrid Heatmap
GitHub-style contribution map showing daily seizure distribution across 52 weeks with interactive hover tooltips.

### 7. Circadian Pattern Analysis
- **6 time scales**: 1-hour, 6-hour, day, week, month, year
- **Polar visualizations**: Beautiful radial charts for each time scale
- **Severe seizure toggle**: Filter to show only severe seizures
- **Pattern insights**: Automated clinical recommendations
- **Color-coded views**: Different palettes for each granularity

## 📋 Data Format

The app uses **three CSV files**:

### Seizures CSV (364 rows)
- Columns: `date`, `day`, `week`, `day_of_week`, `daily_total`, `daily_severe`

### Medications CSV (364 rows)
- Columns: `date`, `day`, `Lamictal`, `Clonazepam`, `Vimpat`, `Zonisamide`, `Fycompa`

### Assessments CSV (sparse, ~40-50 rows)
- Columns: `date`, `day`, `week`, `QoL`, `Anxiety`, `Depression`, `Behavioral`

### Demo Data Included
Located in the `data/` folder:
- `data/seizures_dummy.csv` - 325 seizures over 364 days
- `data/medications_dummy.csv` - 5 medications with realistic dose changes
- `data/assessments_dummy.csv` - 45 assessment entries

## 🎨 Key Features

- **Interactive visualizations**: Plotly charts with hover tooltips and zoom
- **Responsive design**: Adapts to different screen sizes
- **Multiple time scales**: Analyze from hourly to yearly patterns
- **Clinical insights**: Automated pattern detection and recommendations
- **Severe seizure filtering**: Toggle to focus on critical events
- **Correlation analysis**: Pearson correlations between variables
- **Cached processing**: Fast performance with Streamlit caching

## 🔧 Technology Stack

- **Streamlit** - Web application framework
- **Plotly** - Interactive visualizations
- **Pandas** - Data manipulation
- **NumPy** - Numerical computations
- **SciPy** - Statistical analysis

## 💡 Tips

- Start with demo data to explore features
- Use circadian analysis to identify time-based patterns
- Toggle severe seizures to focus on critical events
- Check correlation panel for medication effectiveness
- Export insights for clinical discussions

---

**Built for comprehensive epilepsy care analytics** 💙
