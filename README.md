# MTE Consistency Analyzer

A sophisticated web-based tool for advanced statistical process control analysis using I-MR (Individual-Moving Range) control charts. Designed for manufacturing test engineering teams to analyze test data consistency and process stability.

## Features

- **I-MR Control Charts**: Generate Individual-Moving Range control charts with proper statistical control limits
- **EIF-Based Analysis**: Color-coded analysis by Equipment Interface (EIF) categories
- **Interactive Plotting**: Dynamic plot generation with real-time filtering and bin selection
- **Stable Process Detection**: Automatic identification of dominant histogram bins for control limit calculation
- **Professional PDF Reports**: Comprehensive consistency analysis reports with detailed statistics
- **Advanced Data Processing**: Handles large datasets, zip files, and complex test structures
- **Modern Web Interface**: Beautiful, responsive UI with light theme and subtle animations

## Setup

1. **Install Dependencies**:
   ```bash
   cd ~/Documents/w3a_flask_app
   pip3 install -r requirements.txt
   ```

2. **Run the Application**:
   ```bash
   python3 app.py
   ```

3. **Open in Browser**:
   Navigate to `http://localhost:5000`

## Usage

### Phase 1: Data Processing
1. Enter the path to your test logs directory or upload a ZIP file
2. Click "Process All CSV Data" to analyze your dataset
3. Wait for processing to complete (displays total tests and data points found)

### Phase 2: Interactive Analysis
1. **Select Tests**: Choose from numeric test steps in the left panel
2. **View Plots**: Interactive consistency plots with EIF-based color coding
3. **Adjust Histograms**: Change bin counts and click bins to select stable process data
4. **Filter Data**: Use EIF checkboxes to filter analysis by equipment interface
5. **Generate Reports**: Create comprehensive PDF reports with all analyzed tests

### Key Features
- **Bin Selection**: Click histogram bins to define stable process data for I-MR control limits
- **EIF Filtering**: Toggle different equipment interfaces on/off for focused analysis  
- **Test Management**: Move numeric tests to non-numeric category if conceptually inappropriate
- **Navigation**: Use arrow keys or Previous/Next buttons to browse through tests

## Supported Data Formats

- **Zip Files**: Automatically extracts and processes parametric CSV files
- **CSV Files**: Directly processes `*_parametric.csv` files
- **Flexible Structure**: Handles missing tests and varying test plans
- **Numeric Filtering**: Only processes float-type measurements

## Project Structure

```
w3a_flask_app/
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── templates/            # HTML templates
│   ├── base.html         # Base template
│   ├── index.html        # Home page
│   └── consistency/      # Consistency plots templates
├── static/              # CSS and JavaScript
│   ├── css/style.css    # Main stylesheet
│   └── js/main.js       # JavaScript utilities
└── routes/              # Route modules
    ├── main.py          # Main routes
    └── consistency.py   # Consistency plots routes
```

## Example Paths

- `/Volumes/slimthicc/0112 testing`
- `/Users/zachstanziano/Downloads/aws-logs-w3a-w3a_mlb_test-2026-01-12T03-54-06`
- `/path/to/your/extracted/logs`

## Output

### Interactive Analysis
- **Real-time Plots**: Dynamic consistency plots with immediate updates
- **Live Statistics**: Current test statistics and control limits
- **Histogram Analysis**: Interactive bin selection for stable process identification

### PDF Reports
- **Comprehensive Reports**: Professional multi-page PDF documents
- **All Tests Included**: Complete analysis of all numeric tests (no 20-test limit)
- **Detailed Statistics**: Dataset overview, pass rates, and methodology explanation
- **I-MR Control Charts**: Proper Individual-Moving Range control limit calculations
- **Custom Branding**: Program name, build, and date customization

### Report Types
- **Station Consistency Report**: For station testing scenarios (3x20 approach)
- **Process Consistency Report**: For process testing scenarios (20+x1 approach)
