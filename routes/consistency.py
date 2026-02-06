"""
Consistency plots routes for the W3A Flask application - REFACTORED VERSION
Two-phase approach: 1) Data Processing, 2) Interactive Plotting
"""

from flask import Blueprint, render_template, request, jsonify, send_file, flash, redirect, url_for, session
from werkzeug.utils import secure_filename
import os
import csv
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
import warnings
import re
import zipfile
import tempfile
import shutil
from datetime import datetime
import yaml
from scipy import stats
from sklearn.ensemble import IsolationForest
import json
import uuid
import pickle

warnings.filterwarnings('ignore')

consistency_bp = Blueprint('consistency', __name__)

# Simple state tracking file - in current directory
STATE_FILE = 'w3a_gui_state.json'

def load_gui_state():
    """Load GUI state from file."""
    try:
        if os.path.exists(STATE_FILE):
            with open(STATE_FILE, 'r') as f:
                return json.load(f)
    except Exception as e:
        print(f"DEBUG: Error loading state: {e}")
    return {}

def save_gui_state(state):
    """Save GUI state to file."""
    try:
        print(f"DEBUG: Attempting to save state to {STATE_FILE}")
        print(f"DEBUG: State content to save: {state}")
        with open(STATE_FILE, 'w') as f:
            json.dump(state, f, indent=2)
        print(f"DEBUG: State successfully saved to {STATE_FILE}")
        
        # Verify what was actually written
        with open(STATE_FILE, 'r') as f:
            saved_content = f.read()
        print(f"DEBUG: Verified saved content: {saved_content[:200]}...")
        
    except Exception as e:
        print(f"DEBUG: Error saving state: {e}")
        import traceback
        print(f"DEBUG: Full traceback: {traceback.format_exc()}")

def update_test_state(test_id, **kwargs):
    """Update state for a specific test."""
    state = load_gui_state()
    if 'tests' not in state:
        state['tests'] = {}
    if test_id not in state['tests']:
        state['tests'][test_id] = {}
    
    # Update with new values
    state['tests'][test_id].update(kwargs)
    save_gui_state(state)

def get_test_state(test_id):
    """Get state for a specific test."""
    state = load_gui_state()
    return state.get('tests', {}).get(test_id, {})

def extract_sn_from_filename(filename):
    """Extract serial number from filename."""
    match = re.match(r'([\dA-Z]+)_', filename)
    if match:
        return match.group(1)
    return None

def extract_eif_from_serial(serial_number):
    """
    Extract EIF (Engineering Identification Flag) from serial number.
    Serial number format: YWWLMMMMRRFSSSSS
    where F is the EIF at position 11 (0-indexed: position 10)
    
    Breakdown:
    - Y (year): position 0
    - WW (week): positions 1-2
    - L (location): position 3
    - MMMM (model): positions 4-7
    - RR (revision): positions 8-9
    - F (EIF): position 10
    - SSSSS (sequential): positions 11-15
    
    Args:
        serial_number: Serial number string (e.g., "604WS27201500009")
    
    Returns:
        EIF character or None if serial number is too short
    """
    if serial_number and len(serial_number) >= 11:
        return serial_number[10]  # Position 11 (0-indexed: 10)
    return None

def find_parametric_csv(log_dir):
    """Find all parametric CSV files in directory tree, extracting nested ZIPs if needed."""
    csv_files = []
    temp_extracts = []  # Keep track of temporary extraction directories
    
    try:
        for root, dirs, files in os.walk(log_dir):
            # Look for direct CSV files first
            for file in files:
                if file.endswith('_parametric.csv'):
                    csv_files.append(os.path.join(root, file))
                
                # Also extract any ZIP files we find (unit logs)
                elif file.endswith('.zip'):
                    zip_path = os.path.join(root, file)
                    try:
                        # Create temp extraction directory for this ZIP in system temp directory
                        temp_base = tempfile.gettempdir()
                        extract_dir = os.path.join(temp_base, f'w3a_nested_extract_{file.replace(".zip", "")}_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
                        os.makedirs(extract_dir, exist_ok=True)
                        temp_extracts.append(extract_dir)
                        
                        print(f"DEBUG: Extracting nested ZIP {zip_path} to temp dir: {extract_dir}")
                        
                        # Extract the ZIP
                        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                            zip_ref.extractall(extract_dir)
                        
                        # Look for parametric CSV files in the extracted content
                        for sub_root, sub_dirs, sub_files in os.walk(extract_dir):
                            for sub_file in sub_files:
                                if sub_file.endswith('_parametric.csv'):
                                    csv_files.append(os.path.join(sub_root, sub_file))
                                    
                    except Exception as e:
                        print(f"DEBUG: Error extracting nested ZIP {zip_path}: {e}")
                        continue
        
        print(f"DEBUG: Found {len(csv_files)} parametric CSV files after nested extraction")
        return csv_files
        
    except Exception as e:
        print(f"DEBUG: Error in find_parametric_csv: {e}")
        return csv_files

def parse_csv_file(csv_path):
    """Parse CSV file and extract test data with limits."""
    numeric_test_data = []
    non_numeric_test_data = []
    filename = os.path.basename(csv_path)
    serial_number = extract_sn_from_filename(filename)
    
    try:
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    test_id = row.get('test_id', '')
                    test_type = row.get('type', '')
                    ret_value = row.get('ret', '')
                    unit = row.get('unit', '')
                    execution_time = row.get('execution_time', '')
                    test_result = row.get('test_result', '')
                    lo_limit = row.get('lo_limit', '')
                    hi_limit = row.get('hi_limit', '')
                    
                    # Try to parse as numeric even if not explicitly marked as FLOAT
                    try:
                        value = float(ret_value)
                        if abs(value) > 1e30:
                            continue
                        
                        # Parse limits if they exist
                        low_limit = None
                        high_limit = None
                        try:
                            if lo_limit and lo_limit.strip():
                                low_limit = float(lo_limit)
                        except (ValueError, TypeError):
                            pass
                        
                        try:
                            if hi_limit and hi_limit.strip():
                                high_limit = float(hi_limit)
                        except (ValueError, TypeError):
                            pass
                        
                        numeric_test_data.append({
                            'test_id': test_id,
                            'value': value,
                            'unit': unit,
                            'timestamp': execution_time,
                            'test_result': test_result,
                            'serial_number': serial_number,
                            'source_file': filename,
                            'low_limit': low_limit,
                            'high_limit': high_limit
                        })
                    except (ValueError, TypeError):
                        # Not a numeric value, collect as non-numeric
                        non_numeric_test_data.append({
                            'test_id': test_id,
                            'test_type': test_type,
                            'ret_value': ret_value,
                            'unit': unit,
                            'timestamp': execution_time,
                            'test_result': test_result,
                            'serial_number': serial_number,
                            'source_file': filename
                        })
                        continue
                except Exception as e:
                    continue
            
    except Exception as e:
        print(f"Error reading {csv_path}: {e}")
    
    return numeric_test_data, non_numeric_test_data

def collect_all_test_data(log_dir):
    """Collect all test data from all CSV files and return organized by test_id."""
    csv_files = find_parametric_csv(log_dir)
    all_tests = defaultdict(list)
    all_non_numeric_tests = defaultdict(list)
    
    print(f"DEBUG: Starting to process {len(csv_files)} CSV files for test data collection")
    
    files_processed = 0
    files_with_data = 0
    total_data_points = 0
    total_non_numeric_points = 0
    
    for i, csv_file in enumerate(csv_files):
        if i % 100 == 0:  # Progress update every 100 files
            print(f"DEBUG: Processing CSV file {i+1}/{len(csv_files)}: {os.path.basename(csv_file)}")
        
        # Check if file exists before trying to parse
        if not os.path.exists(csv_file):
            print(f"DEBUG: WARNING - CSV file does not exist: {csv_file}")
            continue
            
        numeric_test_data, non_numeric_test_data = parse_csv_file(csv_file)
        files_processed += 1
        
        if numeric_test_data or non_numeric_test_data:
            files_with_data += 1
            
            # Process numeric test data
            for test in numeric_test_data:
                all_tests[test['test_id']].append(test)
                total_data_points += 1
            
            # Process non-numeric test data
            for test in non_numeric_test_data:
                all_non_numeric_tests[test['test_id']].append(test)
                total_non_numeric_points += 1
    
    print(f"DEBUG: Collection complete - Processed {files_processed} files, {files_with_data} had data")
    print(f"DEBUG: Total numeric data points collected: {total_data_points}")
    print(f"DEBUG: Total non-numeric data points collected: {total_non_numeric_points}")
    print(f"DEBUG: Unique numeric test IDs found: {len(all_tests)}")
    print(f"DEBUG: Unique non-numeric test IDs found: {len(all_non_numeric_tests)}")
    
    return all_tests, all_non_numeric_tests, len(csv_files)

def convert_numpy_to_python_types(obj):
    """Recursively convert NumPy types to native Python types for JSON serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.generic):
        return obj.item()
    elif isinstance(obj, dict):
        return {key: convert_numpy_to_python_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_to_python_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_to_python_types(item) for item in obj)
    else:
        return obj

def generate_histogram_bins(values, num_bins=5):
    """Generate histogram bins with precise bin count control."""
    values = np.array(values)
    values = values[np.isfinite(values)]  # Remove any infinite or NaN values
    
    if len(values) == 0:
        return [], [], {}
    
    # Enforce minimum bin count of 2
    num_bins = max(2, int(num_bins))
    
    min_val, max_val = np.min(values), np.max(values)
    print(f"DEBUG: Histogram data range: {min_val:.2f} to {max_val:.2f} ({len(values)} samples)")
    print(f"DEBUG: Using precise linear binning with exactly {num_bins} bins")
    
    # Precise linear binning - use exactly the requested number of bins
    data_range = max_val - min_val
    
    if data_range > 0:
        # Create exactly num_bins bins with equal width
        bin_edges = np.linspace(min_val, max_val, num_bins + 1)
    else:
        # Fallback for zero range - create small spread around the single value
        spread = max(abs(min_val) * 0.01, 0.5)  # 1% of value or 0.5, whichever is larger
        bin_edges = np.linspace(min_val - spread, max_val + spread, num_bins + 1)
    
    # Calculate histogram
    counts, _ = np.histogram(values, bins=bin_edges)
    
    # Create bin info with centers and ranges
    bin_centers = []
    bin_ranges = []
    for i in range(len(bin_edges) - 1):
        # Arithmetic mean for linear scale
        center = (bin_edges[i] + bin_edges[i + 1]) / 2
        bin_centers.append(center)
        bin_ranges.append((bin_edges[i], bin_edges[i + 1]))
    
    # Find dominant bin (>51% of samples)
    total_samples = len(values)
    dominant_bin = None
    max_count = 0
    max_bin = 0
    for i, count in enumerate(counts):
        if count > max_count:
            max_count = count
            max_bin = i
        if count / total_samples > 0.51:
            dominant_bin = i
            break
    
    print(f"DEBUG: Histogram bins - Max bin: {max_bin} ({max_count} samples, {100*max_count/total_samples:.1f}%)")
    if dominant_bin is not None:
        print(f"DEBUG: Dominant bin found: {dominant_bin} ({counts[dominant_bin]} samples, {100*counts[dominant_bin]/total_samples:.1f}%)")
    else:
        print(f"DEBUG: No dominant bin (>51%), will use max bin {max_bin}")
    
    histogram_data = {
        'bin_edges': bin_edges.tolist(),
        'bin_centers': bin_centers,
        'bin_ranges': bin_ranges,
        'counts': counts.tolist(),
        'total_samples': total_samples,
        'dominant_bin': dominant_bin,
        'use_log_bins': False  # Always linear now
    }
    
    return bin_centers, counts, histogram_data

def calculate_imr_control_limits_from_stable_data(all_values, stable_indices=None):
    """Calculate I-MR control limits from stable process data only."""
    all_values = np.array(all_values)
    
    # If no stable indices provided, use all data (fallback to original behavior)
    if stable_indices is None or len(stable_indices) == 0:
        stable_values = all_values
    else:
        stable_values = all_values[stable_indices]
    
    n = len(stable_values)
    
    if n < 2:
        # Not enough stable data for I-MR chart
        mean_val = np.mean(all_values) if len(all_values) > 0 else 0
        return {
            'mean': float(mean_val),
            'ucl': float(mean_val),
            'lcl': float(mean_val),
            'sigma_hat': 0.0,
            'mr_bar': 0.0,
            'moving_ranges': [],
            'stable_data_count': n,
            'method': 'insufficient_stable_data'
        }
    
    # 1. Calculate center line from stable data only
    mean_val = np.mean(stable_values)
    
    # 2. Calculate moving ranges from stable data MR_i = |X_i - X_{i-1}|
    moving_ranges = []
    for i in range(1, n):
        mr = abs(stable_values[i] - stable_values[i-1])
        moving_ranges.append(mr)
    
    # 3. Calculate average moving range from stable data
    mr_bar = np.mean(moving_ranges) if moving_ranges else 0
    
    # 4. Estimate process sigma using d2 = 1.128 for moving range of 2
    d2 = 1.128
    sigma_hat = mr_bar / d2 if mr_bar > 0 else 0
    
    # 5. Calculate I-MR control limits based on stable process
    ucl = mean_val + 3 * sigma_hat
    lcl = mean_val - 3 * sigma_hat
    
    return convert_numpy_to_python_types({
        'mean': float(mean_val),
        'ucl': float(ucl),
        'lcl': float(lcl),
        'sigma_hat': float(sigma_hat),
        'mr_bar': float(mr_bar),
        'moving_ranges': [float(mr) for mr in moving_ranges],
        'stable_data_count': int(n),
        'method': 'imr_from_stable_data'
    })

def find_stable_data_indices(values, histogram_data, selected_bin=None):
    """Find indices of data points that fall within the selected stable bin."""
    values = np.array(values)
    
    # Validate inputs
    if len(values) == 0:
        print("DEBUG: No values provided to find_stable_data_indices")
        return []
    
    if not histogram_data or 'bin_ranges' not in histogram_data or len(histogram_data['bin_ranges']) == 0:
        print("DEBUG: No valid histogram data available")
        return []
    
    # Use dominant bin if no specific bin selected
    if selected_bin is None:
        selected_bin = histogram_data.get('dominant_bin')
        print(f"DEBUG: No bin selected, using dominant bin: {selected_bin}")
    
    # If no dominant bin found, find the bin with the most samples
    if selected_bin is None:
        if len(histogram_data['counts']) > 0:
            selected_bin = np.argmax(histogram_data['counts'])
            print(f"DEBUG: No dominant bin, using max bin: {selected_bin}")
        else:
            print("DEBUG: No histogram data available")
            return []
    
    # Validate selected bin
    if selected_bin < 0 or selected_bin >= len(histogram_data['bin_ranges']):
        print(f"DEBUG: Selected bin {selected_bin} out of range (max: {len(histogram_data['bin_ranges'])-1})")
        return []
    
    # Get the range for the selected bin
    bin_min, bin_max = histogram_data['bin_ranges'][selected_bin]
    print(f"DEBUG: Using bin {selected_bin} range: {bin_min:.2f} to {bin_max:.2f}")
    
    # Find indices of values that fall within this bin
    stable_indices = np.where((values >= bin_min) & (values < bin_max))[0]
    
    if len(stable_indices) == 0:
        print(f"DEBUG: No samples found in selected bin {selected_bin}")
        return []
    
    stable_values = values[stable_indices]
    
    # Safe min/max calculation for non-empty array
    if len(stable_values) > 0:
        min_val = np.min(stable_values)
        max_val = np.max(stable_values)
        print(f"DEBUG: Found {len(stable_indices)} stable samples (range: {min_val:.2f} to {max_val:.2f})")
    else:
        print(f"DEBUG: Found {len(stable_indices)} stable samples (empty range)")
    
    return stable_indices.tolist()

def generate_plot_data(test_id, test_data, plot_options=None):
    """Generate plot data for a specific test with histogram-based control limits."""
    if plot_options is None:
        plot_options = {
            'show_spec_limits': True,
            'show_control_limits': True,
            'selected_bin': None,  # Auto-select dominant bin
            'num_bins': 5  # Default number of bins
        }
    
    df = pd.DataFrame(test_data)
    df = df.sort_values('timestamp')
    
    values = df['value'].values
    test_results = df['test_result'].values
    serial_numbers = df['serial_number'].values
    
    n = len(values)
    if n < 2:
        return None
    
    # Extract limits from the data
    usl = None
    lsl = None
    for data_point in test_data:
        if data_point.get('high_limit') is not None:
            usl = data_point['high_limit']
            break
    
    for data_point in test_data:
        if data_point.get('low_limit') is not None:
            lsl = data_point['low_limit']
            break
    
    # Generate histogram for stable data selection
    num_bins = plot_options.get('num_bins', 5)  # Get bin count from options
    bin_centers, bin_counts, histogram_data = generate_histogram_bins(values, num_bins)
    
    # Find stable data indices based on selected bin (or auto-selected dominant bin)
    selected_bin = plot_options.get('selected_bin')
    stable_indices = find_stable_data_indices(values, histogram_data, selected_bin)
    
    # Calculate I-MR control limits from stable data only
    imr_results = calculate_imr_control_limits_from_stable_data(values, stable_indices)
    print(f"DEBUG: Control limits calculated - Mean: {imr_results['mean']:.2f}, UCL: {imr_results['ucl']:.2f}, LCL: {imr_results['lcl']:.2f} (from {imr_results['stable_data_count']} stable samples)")
    
    # Identify out-of-control points using stable-based I-MR limits
    out_of_control = (values > imr_results['ucl']) | (values < imr_results['lcl'])
    
    # Categorize by EIF (Engineering Identification Flag)
    eif_categories = {}
    for i, sn in enumerate(serial_numbers):
        eif = extract_eif_from_serial(str(sn)) if sn else None
        if eif not in eif_categories:
            eif_categories[eif] = []
        eif_categories[eif].append(i)
    
    # Create data series for plotting
    plot_data = {
        'x_values': list(range(len(values))),
        'values': values.tolist(),
        'test_results': test_results.tolist(),
        'serial_numbers': serial_numbers.tolist(),
        'eif_categories': {str(k): v for k, v in eif_categories.items()},
        'pass_indices': np.where(test_results == 'PASS')[0].tolist(),
        'fail_indices': np.where(test_results == 'FAIL')[0].tolist(),
        'ooc_indices': np.where(out_of_control)[0].tolist(),
        'stable_indices': stable_indices,  # Indices used for control limit calculation
        
        # I-MR Statistics (from stable data)
        'mean': imr_results['mean'],
        'std': imr_results['sigma_hat'],
        'ucl': imr_results['ucl'],
        'lcl': imr_results['lcl'],
        'usl': float(usl) if usl is not None else None,
        'lsl': float(lsl) if lsl is not None else None,
        
        # I-MR specific data
        'mr_bar': imr_results['mr_bar'],
        'sigma_hat': imr_results['sigma_hat'],
        'moving_ranges': imr_results['moving_ranges'],
        'control_method': imr_results['method'],
        'stable_data_count': imr_results['stable_data_count'],
        
        # Histogram data
        'histogram': {
            'bin_centers': histogram_data['bin_centers'],
            'bin_counts': histogram_data['counts'],
            'bin_ranges': histogram_data['bin_ranges'],
            'dominant_bin': histogram_data['dominant_bin'],
            'selected_bin': selected_bin if selected_bin is not None else histogram_data['dominant_bin'],
            'use_log_bins': histogram_data['use_log_bins'],
            'total_samples': histogram_data['total_samples']
        },
        
        # Counts
        'n_samples': int(n),
        'n_pass': int(np.sum(test_results == 'PASS')),
        'n_fail': int(np.sum(test_results == 'FAIL')),
        'eif_counts': {str(k): len(v) for k, v in eif_categories.items()},
        'n_ooc': int(np.sum(out_of_control)),
        'usl_violations': int(np.sum(values > usl)) if usl is not None else 0,
        'lsl_violations': int(np.sum(values < lsl)) if lsl is not None else 0,
        
        # Metadata
        'test_id': test_id,
        'unit': df['unit'].iloc[0] if 'unit' in df.columns and df['unit'].iloc[0] else '',
        'has_limits': usl is not None or lsl is not None,
        'plot_options': plot_options
    }
    
    return convert_numpy_to_python_types(plot_data)

# ===== ROUTES =====

@consistency_bp.route('/')
def index():
    """Consistency plots main page - NEW INTERACTIVE VERSION."""
    # Clear state file when opening GUI (simple overwrite approach)
    print(f"DEBUG: GUI launched - attempting to create state file: {STATE_FILE}")
    print(f"DEBUG: Current working directory: {os.getcwd()}")
    print(f"DEBUG: Full state file path: {os.path.abspath(STATE_FILE)}")
    
    try:
        initial_state = {
            'session_start': datetime.now().isoformat(),
            'tests': {}
        }
        save_gui_state(initial_state)
        print(f"DEBUG: Successfully created GUI state file for new session")
        
        # Verify file was created
        if os.path.exists(STATE_FILE):
            file_size = os.path.getsize(STATE_FILE)
            print(f"DEBUG: State file exists, size: {file_size} bytes")
            with open(STATE_FILE, 'r') as f:
                content = f.read()
                print(f"DEBUG: State file content: {content[:200]}...")
        else:
            print(f"DEBUG: ERROR - State file was not created!")
            
    except Exception as e:
        print(f"DEBUG: Error creating state file: {e}")
        import traceback
        print(f"DEBUG: Full traceback: {traceback.format_exc()}")
    
    return render_template('consistency/index_interactive.html')

@consistency_bp.route('/process_data', methods=['POST'])
def process_data():
    """Phase 1: Process all CSV data and return test list + summary."""
    try:
        data = request.get_json()
        log_path = data.get('log_path', '').strip()
        
        if not log_path or not os.path.exists(log_path):
            return jsonify({'error': 'No log directory or file path provided'}), 400
        
        # Handle ZIP files - extract if needed
        processing_dir = log_path
        temp_extract_dir = None
        
        if os.path.isfile(log_path) and log_path.endswith('.zip'):
            print("DEBUG: Extracting ZIP file for data processing...")
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            temp_extract_dir = os.path.join(tempfile.gettempdir(), f'w3a_data_extract_{timestamp}')
            os.makedirs(temp_extract_dir, exist_ok=True)
            
            try:
                with zipfile.ZipFile(log_path, 'r') as zip_ref:
                    zip_ref.extractall(temp_extract_dir)
                processing_dir = temp_extract_dir
                print(f"DEBUG: Extracted ZIP to: {processing_dir}")
            except Exception as e:
                if temp_extract_dir and os.path.exists(temp_extract_dir):
                    shutil.rmtree(temp_extract_dir)
                return jsonify({'error': f'Error extracting ZIP file: {str(e)}'}), 400
        
        try:
            # Collect all test data
            print("DEBUG: Collecting test data from all CSV files...")
            all_tests, all_non_numeric_tests, csv_count = collect_all_test_data(processing_dir)
            
            if not all_tests:
                return jsonify({'error': 'No parametric test data found'}), 400
            
            print(f"DEBUG: Found {len(all_tests)} unique numeric tests from {csv_count} CSV files")
            print(f"DEBUG: Found {len(all_non_numeric_tests)} unique non-numeric tests from {csv_count} CSV files")
            
            # Create session ID to store data
            session_id = str(uuid.uuid4())
            
            print(f"DEBUG: Creating session {session_id} to store data")
            
            # Store data in temporary file instead of Flask session to avoid size limits
            session_data_file = os.path.join(tempfile.gettempdir(), f'w3a_session_{session_id}.pkl')
            all_tests_dict = dict(all_tests)
            all_non_numeric_tests_dict = dict(all_non_numeric_tests)
            
            session_data = {
                'all_tests': all_tests_dict,
                'all_non_numeric_tests': all_non_numeric_tests_dict,
                'csv_count': csv_count,
                'timestamp': datetime.now().isoformat(),
                'temp_extract_dir': temp_extract_dir
            }
            
            print(f"DEBUG: Storing {len(all_tests_dict)} tests in file: {session_data_file}")
            
            # Save to pickle file
            with open(session_data_file, 'wb') as f:
                pickle.dump(session_data, f)
            
            # Store only session metadata in Flask session
            session[session_id] = {
                'data_file': session_data_file,
                'timestamp': datetime.now().isoformat()
            }
            
            print(f"DEBUG: Session data stored successfully in file")
            
            # ADD ALL DETECTED TESTS TO STATE FILE
            print(f"DEBUG: Adding {len(all_tests)} detected tests to state file")
            state = load_gui_state()
            if 'tests' not in state:
                state['tests'] = {}
            
            # Add each detected test to the state with default settings
            for test_id in all_tests.keys():
                if test_id not in state['tests']:
                    state['tests'][test_id] = {
                        'detected_during_parse': True,
                        'num_bins': 5,  # Default bins
                        'selected_bin': None  # Will auto-select dominant bin
                    }
            
            # Save the updated state with all tests
            save_gui_state(state)
            print(f"DEBUG: State file updated with all {len(all_tests)} detected tests")
            
            # Generate test list with sample counts (limit response size)
            test_list = []
            total_samples = 0
            for test_id, test_data in all_tests.items():
                total_samples += len(test_data)
                test_info = {
                    'test_id': test_id,
                    'sample_count': len(test_data),
                    'has_limits': any(d.get('high_limit') is not None or d.get('low_limit') is not None for d in test_data),
                    'unit': test_data[0].get('unit', '') if test_data else ''
                }
                test_list.append(test_info)
            
            # Sort by sample count (descending)
            test_list.sort(key=lambda x: x['sample_count'], reverse=True)
            
            print(f"DEBUG: Preparing JSON response with {len(test_list)} tests")
            
            # Create minimal response to avoid timeout
            minimal_test_list = []
            for test in test_list[:10]:  # Only top 10 tests initially
                minimal_test_list.append({
                    'test_id': test['test_id'][:50],  # Truncate long test names
                    'sample_count': test['sample_count'],
                    'has_limits': test['has_limits'],
                    'unit': test['unit'][:10] if test['unit'] else ''  # Truncate units
                })
            
            response_data = {
                'success': True,
                'session_id': session_id,
                'total_tests': len(all_tests),
                'total_samples': total_samples,
                'csv_files_processed': csv_count,
                'test_list': minimal_test_list
            }
            
            print(f"DEBUG: Response data prepared with {len(minimal_test_list)} tests")
            print(f"DEBUG: Estimated response size: {len(str(response_data))} characters")
            return jsonify(response_data)
            
        finally:
            # Note: Don't cleanup temp_extract_dir here - we need it for plotting
            pass
            
    except Exception as e:
        print(f"DEBUG: Data processing error: {str(e)}")
        return jsonify({'error': f'Data processing error: {str(e)}'}), 500

@consistency_bp.route('/get_full_test_list', methods=['POST'])
def get_full_test_list():
    """Get the complete test list for a session."""
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        
        if not session_id or session_id not in session:
            return jsonify({'error': 'Invalid or expired session'}), 400
        
        # Load data from file
        session_meta = session[session_id]
        data_file = session_meta['data_file']
        
        if not os.path.exists(data_file):
            return jsonify({'error': 'Session data file not found'}), 400
        
        with open(data_file, 'rb') as f:
            session_data = pickle.load(f)
        
        all_tests = session_data['all_tests']
        all_non_numeric_tests = session_data.get('all_non_numeric_tests', {})
        
        # Generate complete numeric test list
        test_list = []
        for test_id, test_data in all_tests.items():
            test_info = {
                'test_id': test_id,
                'sample_count': len(test_data),
                'has_limits': any(d.get('high_limit') is not None or d.get('low_limit') is not None for d in test_data),
                'unit': test_data[0].get('unit', '') if test_data else ''
            }
            test_list.append(test_info)
        
        # Sort by sample count (descending)
        test_list.sort(key=lambda x: x['sample_count'], reverse=True)
        
        # Generate non-numeric test list
        non_numeric_test_list = []
        for test_id, test_data in all_non_numeric_tests.items():
            test_info = {
                'test_id': test_id,
                'sample_count': len(test_data),
                'test_type': test_data[0].get('test_type', '') if test_data else '',
                'unit': test_data[0].get('unit', '') if test_data else ''
            }
            non_numeric_test_list.append(test_info)
        
        # Sort by sample count (descending)
        non_numeric_test_list.sort(key=lambda x: x['sample_count'], reverse=True)
        
        return jsonify({
            'success': True,
            'test_list': test_list,
            'non_numeric_test_list': non_numeric_test_list
        })
        
    except Exception as e:
        print(f"DEBUG: Get full test list error: {str(e)}")
        return jsonify({'error': f'Get full test list error: {str(e)}'}), 500

@consistency_bp.route('/get_plot_data', methods=['POST'])
def get_plot_data():
    """Phase 2: Generate plot data for a specific test."""
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        test_id = data.get('test_id')
        plot_options = data.get('plot_options', {})
        
        if not session_id or session_id not in session:
            return jsonify({'error': 'Invalid or expired session'}), 400
        
        if not test_id:
            return jsonify({'error': 'Test ID required'}), 400
        
        # Load data from file
        session_meta = session[session_id]
        data_file = session_meta['data_file']
        
        if not os.path.exists(data_file):
            return jsonify({'error': 'Session data file not found'}), 400
        
        with open(data_file, 'rb') as f:
            session_data = pickle.load(f)
        
        all_tests = session_data['all_tests']
        
        if test_id not in all_tests:
            return jsonify({'error': f'Test {test_id} not found'}), 404
        
        # DON'T restore saved state during interactive use - state is ONLY for PDF generation!
        # The frontend should control all interactive behavior
        
        # If frontend provided new values, save them to state for PDF generation
        if 'num_bins' in plot_options or 'selected_bin' in plot_options:
            print(f"DEBUG: Frontend provided new values, saving to state for PDF: {plot_options}")
            update_test_state(test_id, 
                             selected_bin=plot_options.get('selected_bin'),
                             num_bins=plot_options.get('num_bins', 5))
        
        # Generate plot data
        test_data = all_tests[test_id]
        plot_data = generate_plot_data(test_id, test_data, plot_options)
        
        if not plot_data:
            return jsonify({'error': 'Insufficient data for plotting'}), 400
        
        print(f"DEBUG: Plot data refresh complete - returning updated control limits UCL: {plot_data['ucl']:.2f}, LCL: {plot_data['lcl']:.2f} to frontend")
        return jsonify({
            'success': True,
            'plot_data': plot_data
        })
        
    except Exception as e:
        print(f"DEBUG: Plot data generation error: {str(e)}")
        return jsonify({'error': f'Plot data generation error: {str(e)}'}), 500

@consistency_bp.route('/update_control_limits', methods=['POST'])
def update_control_limits():
    """Update control limits based on selected histogram bin."""
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        test_id = data.get('test_id')
        selected_bin = data.get('selected_bin')
        plot_options = data.get('plot_options', {})
        
        if not session_id or session_id not in session:
            return jsonify({'error': 'Invalid or expired session'}), 400
        
        if not test_id:
            return jsonify({'error': 'Test ID required'}), 400
        
        if selected_bin is None:
            return jsonify({'error': 'Selected bin required'}), 400
        
        # Load data from file
        session_meta = session[session_id]
        data_file = session_meta['data_file']
        
        if not os.path.exists(data_file):
            return jsonify({'error': 'Session data file not found'}), 400
        
        with open(data_file, 'rb') as f:
            session_data = pickle.load(f)
        
        all_tests = session_data['all_tests']
        
        if test_id not in all_tests:
            return jsonify({'error': f'Test {test_id} not found'}), 404
        
        # Update plot options with selected bin
        plot_options['selected_bin'] = selected_bin
        
        # Save state for this test
        print(f"DEBUG: Saving state for test '{test_id}' - selected_bin: {selected_bin}, num_bins: {plot_options.get('num_bins', 5)}")
        update_test_state(test_id, 
                         selected_bin=selected_bin,
                         num_bins=plot_options.get('num_bins', 5))
        print(f"DEBUG: State save completed for test '{test_id}'")
        
        # Verify state was saved
        if os.path.exists(STATE_FILE):
            print(f"DEBUG: State file exists after save, size: {os.path.getsize(STATE_FILE)} bytes")
        else:
            print(f"DEBUG: ERROR - State file does not exist after save attempt!")
        
        # Generate updated plot data
        test_data = all_tests[test_id]
        plot_data = generate_plot_data(test_id, test_data, plot_options)
        
        if not plot_data:
            return jsonify({'error': 'Insufficient data for plotting'}), 400
        
        print(f"DEBUG: Control limits update complete - returning updated control limits UCL: {plot_data['ucl']:.2f}, LCL: {plot_data['lcl']:.2f} to frontend")
        return jsonify({
            'success': True,
            'plot_data': plot_data
        })
        
    except Exception as e:
        print(f"DEBUG: Update control limits error: {str(e)}")
        return jsonify({'error': f'Update control limits error: {str(e)}'}), 500

@consistency_bp.route('/upload_log_file', methods=['POST'])
def upload_log_file():
    """Handle log file uploads for analysis."""
    try:
        if 'log_file' not in request.files:
            return jsonify({'error': 'No log file provided'}), 400
        
        file = request.files['log_file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Validate file type
        filename = secure_filename(file.filename)
        if not (filename.lower().endswith('.zip') or filename.lower().endswith('.csv')):
            return jsonify({'error': 'Invalid file type. Expected .zip or .csv'}), 400
        
        # Create uploads directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        upload_dir = os.path.join(tempfile.gettempdir(), f'w3a_uploads_{timestamp}')
        os.makedirs(upload_dir, exist_ok=True)
        
        # Save the uploaded file
        file_path = os.path.join(upload_dir, filename)
        file.save(file_path)
        
        print(f"DEBUG: File uploaded successfully to: {file_path}")
        print(f"DEBUG: File size: {os.path.getsize(file_path)} bytes")
        
        return jsonify({
            'success': True,
            'file_path': file_path,
            'filename': filename,
            'size': os.path.getsize(file_path)
        })
        
    except Exception as e:
        print(f"DEBUG: Upload error: {str(e)}")
        return jsonify({'error': f'Upload error: {str(e)}'}), 500

@consistency_bp.route('/move_test_to_non_numeric', methods=['POST'])
def move_test_to_non_numeric():
    """Move a test from numeric to non-numeric category."""
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        test_id = data.get('test_id')
        
        if not session_id or session_id not in session:
            return jsonify({'error': 'Invalid or expired session'}), 400
        
        if not test_id:
            return jsonify({'error': 'Test ID required'}), 400
        
        # Load session data from file
        session_meta = session[session_id]
        data_file = session_meta['data_file']
        
        if not os.path.exists(data_file):
            return jsonify({'error': 'Session data file not found'}), 400
        
        with open(data_file, 'rb') as f:
            session_data = pickle.load(f)
        
        all_tests = session_data['all_tests']
        all_non_numeric_tests = session_data.get('all_non_numeric_tests', {})
        
        # Check if test exists in numeric tests
        if test_id not in all_tests:
            return jsonify({'error': f'Test {test_id} not found in numeric tests'}), 404
        
        # Move test data from numeric to non-numeric
        test_data = all_tests[test_id]
        
        # Convert numeric test data to non-numeric format
        non_numeric_data = []
        for data_point in test_data:
            non_numeric_data.append({
                'test_id': test_id,
                'test_type': 'MOVED_FROM_NUMERIC',
                'ret_value': str(data_point['value']),
                'unit': data_point.get('unit', ''),
                'timestamp': data_point.get('timestamp', ''),
                'test_result': data_point.get('test_result', ''),
                'serial_number': data_point.get('serial_number', ''),
                'source_file': data_point.get('source_file', '')
            })
        
        # Add to non-numeric tests
        all_non_numeric_tests[test_id] = non_numeric_data
        
        # Remove from numeric tests
        del all_tests[test_id]
        
        # Update session data
        session_data['all_tests'] = all_tests
        session_data['all_non_numeric_tests'] = all_non_numeric_tests
        
        # Save updated data back to file
        with open(data_file, 'wb') as f:
            pickle.dump(session_data, f)
        
        # UPDATE STATE FILE TO REMOVE THE MOVED TEST
        print(f"DEBUG: Updating state file to remove moved test '{test_id}' from numeric section")
        state = load_gui_state()
        if 'tests' in state and test_id in state['tests']:
            # Remove the test from state since it's no longer numeric
            del state['tests'][test_id]
            save_gui_state(state)
            print(f"DEBUG: Removed test '{test_id}' from state file (moved to non-numeric)")
        else:
            print(f"DEBUG: Test '{test_id}' was not found in state file")
        
        print(f"DEBUG: Moved test '{test_id}' from numeric to non-numeric ({len(non_numeric_data)} data points)")
        
        return jsonify({
            'success': True,
            'message': f'Test "{test_id}" moved to non-numeric tests',
            'moved_data_points': len(non_numeric_data)
        })
        
    except Exception as e:
        print(f"DEBUG: Move test error: {str(e)}")
        return jsonify({'error': f'Move test error: {str(e)}'}), 500

@consistency_bp.route('/generate_consistency_report', methods=['POST'])
def generate_consistency_report():
    """Generate a comprehensive PDF consistency report."""
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        report_config = data.get('report_config', {})
        
        if not session_id or session_id not in session:
            return jsonify({'error': 'Invalid or expired session'}), 400
        
        # Load session data
        session_meta = session[session_id]
        data_file = session_meta['data_file']
        
        if not os.path.exists(data_file):
            return jsonify({'error': 'Session data file not found'}), 400
        
        with open(data_file, 'rb') as f:
            session_data = pickle.load(f)
        
        all_tests = session_data['all_tests']
        
        # Extract report configuration
        program_name = report_config.get('program_name', 'Unknown Program')
        build_name = report_config.get('build_name', 'Unknown Build')
        report_date = report_config.get('report_date', datetime.now().strftime('%Y-%m-%d'))
        report_type = report_config.get('report_type', 'process')
        report_title = report_config.get('report_title', 'Consistency Analysis Report')
        
        print(f"DEBUG: Generating PDF report for {len(all_tests)} tests")
        
        # Create temporary file for PDF
        temp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
        temp_pdf.close()
        
        try:
            # Generate PDF using matplotlib
            from matplotlib.backends.backend_pdf import PdfPages
            import matplotlib.pyplot as plt
            
            with PdfPages(temp_pdf.name) as pdf:
                # Title page
                fig, ax = plt.subplots(figsize=(8.5, 11))
                ax.axis('off')
                
                # Title page content
                ax.text(0.5, 0.8, report_title, ha='center', va='center', 
                       fontsize=20, fontweight='bold', transform=ax.transAxes)
                
                ax.text(0.5, 0.7, f'Program: {program_name}', ha='center', va='center', 
                       fontsize=14, transform=ax.transAxes)
                
                ax.text(0.5, 0.65, f'Build: {build_name}', ha='center', va='center', 
                       fontsize=14, transform=ax.transAxes)
                
                ax.text(0.5, 0.6, f'Date: {report_date}', ha='center', va='center', 
                       fontsize=14, transform=ax.transAxes)
                
                type_text = 'Station Consistency Report' if report_type == 'station' else 'Process Consistency Report'
                ax.text(0.5, 0.55, f'Type: {type_text}', ha='center', va='center', 
                       fontsize=14, transform=ax.transAxes)
                
                ax.text(0.5, 0.45, f'Total Tests Analyzed: {len(all_tests)}', ha='center', va='center', 
                       fontsize=12, transform=ax.transAxes)
                
                ax.text(0.5, 0.4, f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', 
                       ha='center', va='center', fontsize=10, transform=ax.transAxes)
                
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)
                
                # Generate plots for all tests (or limit based on report config)
                max_tests = report_config.get('max_tests_in_pdf', len(all_tests))  # Allow user to control limit
                test_items = list(all_tests.items())[:max_tests]
                
                for i, (test_id, test_data) in enumerate(test_items):
                    print(f"DEBUG: Generating plots for test {i+1}/{len(test_items)}: {test_id}")
                    
                    # Generate plot data with default settings
                    plot_options = {
                        'show_spec_limits': True,
                        'show_control_limits': True,
                        'num_bins': 5
                    }
                    
                    # Check if user made adjustments to this test
                    saved_state = get_test_state(test_id)
                    if saved_state:
                        # Apply user's saved settings
                        if 'selected_bin' in saved_state:
                            plot_options['selected_bin'] = saved_state['selected_bin']
                        if 'num_bins' in saved_state:
                            plot_options['num_bins'] = saved_state['num_bins']
                        print(f"DEBUG: PDF using saved state for {test_id}: {saved_state}")
                    else:
                        print(f"DEBUG: PDF using defaults for {test_id}")
                    
                    plot_data = generate_plot_data(test_id, test_data, plot_options)
                    if not plot_data:
                        continue
                    
                    # Create three plots per test on one page
                    fig = plt.figure(figsize=(11, 8.5))  # Landscape orientation
                    
                    # Plot 1: Complete Data Analysis (full range)
                    ax1 = plt.subplot(2, 2, 1)
                    values = np.array(plot_data['values'])
                    x_values = np.array(plot_data['x_values'])
                    
                    # Plot data points
                    pass_indices = plot_data['pass_indices']
                    fail_indices = plot_data['fail_indices']
                    ooc_indices = plot_data['ooc_indices']
                    
                    if pass_indices:
                        ax1.scatter(np.array(x_values)[pass_indices], values[pass_indices], 
                                   c='blue', s=20, alpha=0.7, label='PASS')
                    if fail_indices:
                        ax1.scatter(np.array(x_values)[fail_indices], values[fail_indices], 
                                   c='red', s=30, alpha=0.7, marker='^', label='FAIL')
                    if ooc_indices:
                        ax1.scatter(np.array(x_values)[ooc_indices], values[ooc_indices], 
                                   c='orange', s=40, alpha=0.8, marker='x', label='Out of Control')
                    
                    # Add control limits
                    if plot_data['ucl'] and plot_data['lcl']:
                        ax1.axhline(y=plot_data['ucl'], color='orange', linestyle='--', alpha=0.8, label='UCL')
                        ax1.axhline(y=plot_data['lcl'], color='orange', linestyle='--', alpha=0.8, label='LCL')
                        ax1.axhline(y=plot_data['mean'], color='green', linestyle='--', alpha=0.8, label='Mean')
                    
                    # Add spec limits
                    if plot_data['usl'] is not None:
                        ax1.axhline(y=plot_data['usl'], color='red', linestyle='-', alpha=0.8, label='USL')
                    if plot_data['lsl'] is not None:
                        ax1.axhline(y=plot_data['lsl'], color='red', linestyle='-', alpha=0.8, label='LSL')
                    
                    ax1.set_title('Complete Data Analysis', fontsize=10, fontweight='bold')
                    ax1.set_xlabel('Sample Number', fontsize=8)
                    unit_text = f' ({plot_data["unit"]})' if plot_data["unit"] else ''
                    ax1.set_ylabel(f'Value{unit_text}', fontsize=8)
                    ax1.legend(fontsize=6, loc='upper right')
                    ax1.grid(True, alpha=0.3)
                    
                    # Plot 2: Process Control View (zoomed to ±20% of max limit)
                    ax2 = plt.subplot(2, 2, 2)
                    
                    # Calculate zoom range
                    limit_values = []
                    if plot_data['usl'] is not None:
                        limit_values.append(plot_data['usl'])
                    if plot_data['lsl'] is not None:
                        limit_values.append(plot_data['lsl'])
                    if plot_data['ucl']:
                        limit_values.append(plot_data['ucl'])
                    if plot_data['lcl']:
                        limit_values.append(plot_data['lcl'])
                    
                    if limit_values:
                        max_limit = max(abs(max(limit_values)), abs(min(limit_values)))
                        zoom_range = max_limit * 1.2  # 20% more than the greater limit
                        center = plot_data['mean']
                        ax2.set_ylim(center - zoom_range, center + zoom_range)
                    
                    # Same plotting as above but zoomed
                    if pass_indices:
                        ax2.scatter(np.array(x_values)[pass_indices], values[pass_indices], 
                                   c='blue', s=20, alpha=0.7, label='PASS')
                    if fail_indices:
                        ax2.scatter(np.array(x_values)[fail_indices], values[fail_indices], 
                                   c='red', s=30, alpha=0.7, marker='^', label='FAIL')
                    if ooc_indices:
                        ax2.scatter(np.array(x_values)[ooc_indices], values[ooc_indices], 
                                   c='orange', s=40, alpha=0.8, marker='x', label='Out of Control')
                    
                    # Add limits (same as above)
                    if plot_data['ucl'] and plot_data['lcl']:
                        ax2.axhline(y=plot_data['ucl'], color='orange', linestyle='--', alpha=0.8, label='UCL')
                        ax2.axhline(y=plot_data['lcl'], color='orange', linestyle='--', alpha=0.8, label='LCL')
                        ax2.axhline(y=plot_data['mean'], color='green', linestyle='--', alpha=0.8, label='Mean')
                    
                    if plot_data['usl'] is not None:
                        ax2.axhline(y=plot_data['usl'], color='red', linestyle='-', alpha=0.8, label='USL')
                    if plot_data['lsl'] is not None:
                        ax2.axhline(y=plot_data['lsl'], color='red', linestyle='-', alpha=0.8, label='LSL')
                    
                    ax2.set_title('Process Control View', fontsize=10, fontweight='bold')
                    ax2.set_xlabel('Sample Number', fontsize=8)
                    unit_text = f' ({plot_data["unit"]})' if plot_data["unit"] else ''
                    ax2.set_ylabel(f'Value{unit_text}', fontsize=8)
                    ax2.grid(True, alpha=0.3)
                    
                    # Plot 3: Distribution Analysis (histogram)
                    ax3 = plt.subplot(2, 2, 3)
                    
                    if 'histogram' in plot_data and plot_data['histogram']:
                        hist_data = plot_data['histogram']
                        bin_centers = hist_data['bin_centers']
                        bin_counts = hist_data['bin_counts']
                        
                        # Create bar colors (highlight selected bin)
                        colors = ['lightblue' if i == hist_data.get('selected_bin') else 'lightgray' 
                                 for i in range(len(bin_counts))]
                        
                        ax3.bar(bin_centers, bin_counts, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
                        ax3.set_title('Distribution Analysis', fontsize=10, fontweight='bold')
                        unit_text = f' ({plot_data["unit"]})' if plot_data["unit"] else ''
                        ax3.set_xlabel(f'Value{unit_text}', fontsize=8)
                        ax3.set_ylabel('Count', fontsize=8)
                        ax3.grid(True, alpha=0.3)
                    
                    # Plot 4: Statistics table
                    ax4 = plt.subplot(2, 2, 4)
                    ax4.axis('off')
                    
                    # Create statistics text with safe formatting for None values
                    # Safe formatting helper function
                    def safe_format(value, format_spec='.3f'):
                        if value is None:
                            return 'None'
                        try:
                            return f"{value:{format_spec}}"
                        except (ValueError, TypeError):
                            return str(value)
                    
                    stats_text = f"""Test: {test_id}
                    
Sample Statistics:
• Total Samples: {plot_data['n_samples']}
• PASS: {plot_data['n_pass']} ({100*plot_data['n_pass']/plot_data['n_samples']:.1f}%)
• FAIL: {plot_data['n_fail']} ({100*plot_data['n_fail']/plot_data['n_samples']:.1f}%)
• Out of Control: {plot_data['n_ooc']}

Statistical Measures:
• Mean: {safe_format(plot_data['mean'])}
• Std Dev: {safe_format(plot_data['std'])}
• Min: {safe_format(np.min(values))}
• Max: {safe_format(np.max(values))}

Control Limits (I-MR):
• UCL: {safe_format(plot_data['ucl'])}
• LCL: {safe_format(plot_data['lcl'])}
• Stable Data Count: {plot_data['stable_data_count']}

Specification Limits:
• USL: {safe_format(plot_data['usl'])}
• LSL: {safe_format(plot_data['lsl'])}"""

                    if plot_data['usl'] is not None or plot_data['lsl'] is not None:
                        stats_text += f"\n• USL Violations: {plot_data['usl_violations']}"
                        stats_text += f"\n• LSL Violations: {plot_data['lsl_violations']}"
                    
                    ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes, fontsize=8, 
                            verticalalignment='top', fontfamily='monospace')
                    
                    plt.suptitle(f'{test_id} - Consistency Analysis', fontsize=12, fontweight='bold')
                    plt.tight_layout()
                    pdf.savefig(fig, bbox_inches='tight')
                    plt.close(fig)
                
                # Summary page
                fig, ax = plt.subplots(figsize=(8.5, 11))
                ax.axis('off')
                
                ax.text(0.5, 0.9, 'Analysis Summary', ha='center', va='center', 
                       fontsize=16, fontweight='bold', transform=ax.transAxes)
                
                # Calculate summary statistics across all tests
                total_samples = sum(len(test_data) for test_data in all_tests.values())
                tests_with_limits = sum(1 for test_data in all_tests.values() 
                                      if any(d.get('high_limit') is not None or d.get('low_limit') is not None 
                                           for d in test_data))
                
                # Calculate overall pass/fail rates
                all_pass = sum(sum(1 for d in test_data if d.get('test_result') == 'PASS') 
                             for test_data in all_tests.values())
                all_fail = sum(sum(1 for d in test_data if d.get('test_result') == 'FAIL') 
                             for test_data in all_tests.values())
                pass_rate = (all_pass / (all_pass + all_fail) * 100) if (all_pass + all_fail) > 0 else 0
                
                summary_text = f"""CONSISTENCY ANALYSIS SUMMARY

Dataset Overview:
• Total Tests Available: {len(all_tests)}
• Total Data Points: {total_samples:,}
• Tests with Specification Limits: {tests_with_limits}
• Overall Pass Rate: {pass_rate:.1f}% ({all_pass:,} PASS, {all_fail:,} FAIL)

Report Details:
• Tests Analyzed in PDF: {len(test_items)}
• Report Type: {type_text}
• Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
• Program: {program_name}
• Build: {build_name}

Analysis Components (per test):
• Complete Data Analysis - Full chronological view of all measurements
• Process Control View - Focused view highlighting control limit violations
• Distribution Analysis - Histogram showing measurement distribution patterns
• Statistical Summary - Key metrics, control limits, and specification violations

Statistical Control Method:
• I-MR (Individual-Moving Range) Control Charts
• Control limits calculated from stable process data only
• Stable data identified using dominant histogram bin (>51% of samples)
• UCL/LCL = Process Mean ± 3 × (Moving Range Average / d2 constant)
• Moving Range calculated from consecutive stable measurements
• d2 = 1.128 for subgroup size of 2 (consecutive measurements)

Process Capability Assessment:
• Out-of-Control points identified using I-MR limits
• Specification limit violations tracked separately
• Control vs. specification limits clearly differentiated
• Stable process capability estimated from histogram-selected data

Interactive Analysis:
Use the web interface to analyze all {len(all_tests)} tests with full interactivity,
custom bin selection, and real-time control limit recalculation."""
                
                ax.text(0.1, 0.7, summary_text, ha='left', va='top', 
                       fontsize=11, transform=ax.transAxes, fontfamily='monospace')
                
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)
            
            print(f"DEBUG: PDF report generated successfully: {temp_pdf.name}")
            
            # Return the PDF file
            return send_file(temp_pdf.name, 
                           as_attachment=True, 
                           download_name=f'{program_name}_{build_name}_Consistency_Report.pdf',
                           mimetype='application/pdf')
                           
        except Exception as e:
            # Clean up temp file on error
            if os.path.exists(temp_pdf.name):
                os.unlink(temp_pdf.name)
            raise e
            
    except Exception as e:
        print(f"DEBUG: PDF generation error: {str(e)}")
        return jsonify({'error': f'PDF generation error: {str(e)}'}), 500

@consistency_bp.route('/cleanup_session', methods=['POST'])
def cleanup_session():
    """Clean up session data and temporary files."""
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        
        if session_id and session_id in session:
            session_meta = session[session_id]
            data_file = session_meta['data_file']
            
            # Load session data to get temp directories
            if os.path.exists(data_file):
                with open(data_file, 'rb') as f:
                    session_data = pickle.load(f)
                
                # Clean up temporary extraction directory if it exists
                temp_extract_dir = session_data.get('temp_extract_dir')
                if temp_extract_dir and os.path.exists(temp_extract_dir):
                    shutil.rmtree(temp_extract_dir)
                    print(f"DEBUG: Cleaned up temp directory: {temp_extract_dir}")
                
                # Remove the session data file
                os.remove(data_file)
                print(f"DEBUG: Cleaned up session data file: {data_file}")
            
            # Remove from session
            del session[session_id]
            
        return jsonify({'success': True})
        
    except Exception as e:
        print(f"DEBUG: Cleanup error: {str(e)}")
        return jsonify({'error': f'Cleanup error: {str(e)}'}), 500
