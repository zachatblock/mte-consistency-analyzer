# W3A Test Analysis Flask Application

A web-based tool for analyzing W3A manufacturing test data and generating consistency plots (statistical process control charts).

## Features

- **Consistency Plots**: Generate control charts with ±3σ limits for parametric test data
- **Serial Number Color Coding**: Red markers for SN 549xxx, blue markers for SN 602xxx
- **Pass/Fail Tracking**: Distinguish between measurements from passing vs failing units
- **Flexible Data Processing**: Handles zip files, missing tests, and varying test plans
- **Web Interface**: Clean, responsive web UI for easy use

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

1. Go to the **Consistency Plots** page
2. Enter the path to your W3A test logs directory
3. Click "Generate Consistency Plots"
4. View and download the generated plots and summary report

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

Generated plots are saved to `/tmp/w3a_consistency_[timestamp]/` with:
- Individual PNG files for each test
- Summary CSV report with statistics
- Downloadable results through the web interface
