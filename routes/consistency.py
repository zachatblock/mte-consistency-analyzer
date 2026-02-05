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

def generate_histogram_bins(values, num_bins=20):
    """Generate histogram bins with smart binning standardized around orders of magnitude."""
    values = np.array(values)
    values = values[np.isfinite(values)]  # Remove any infinite or NaN values
    
    if len(values) == 0:
        return [], [], {}
    
    min_val, max_val = np.min(values), np.max(values)
    print(f"DEBUG: Histogram data range: {min_val:.2f} to {max_val:.2f} ({len(values)} samples)")
    
    # Check if we need logarithmic binning (wide dynamic range)
    use_log_bins = (max_val / min_val > 100) and (min_val > 0)
    print(f"DEBUG: Using {'logarithmic' if use_log_bins else 'linear'} binning (ratio: {max_val/min_val:.1f})")
    
    if use_log_bins:
        # Logarithmic binning with standardized edges
        log_min = np.log10(min_val)
        log_max = np.log10(max_val)
        
        # Round to nice order-of-magnitude boundaries
        log_min_rounded = np.floor(log_min * 2) / 2  # Round to nearest 0.5 decade
        log_max_rounded = np.ceil(log_max * 2) / 2
        
        # Create standardized log edges
        log_edges = np.linspace(log_min_rounded, log_max_rounded, num_bins + 1)
        bin_edges = 10 ** log_edges
    else:
        # Linear binning with nice round numbers
        data_range = max_val - min_val
        
        # Determine appropriate step size based on order of magnitude
        if data_range > 0:
            order_of_magnitude = 10 ** np.floor(np.log10(data_range))
            nice_step = order_of_magnitude / 10  # Start with 1/10th of the order of magnitude
            
            # Adjust step size to get reasonable number of bins
            while (data_range / nice_step) > num_bins * 1.5:
                nice_step *= 2
            while (data_range / nice_step) < num_bins * 0.5:
                nice_step /= 2
            
            # Round min/max to nice boundaries
            nice_min = np.floor(min_val / nice_step) * nice_step
            nice_max = np.ceil(max_val / nice_step) * nice_step
            
            # Generate nice bin edges
            n_steps = int(np.ceil((nice_max - nice_min) / nice_step)) + 1
            bin_edges = np.linspace(nice_min, nice_min + (n_steps - 1) * nice_step, n_steps)
        else:
            # Fallback for zero range
            bin_edges = np.linspace(min_val - 0.5, max_val + 0.5, num_bins + 1)
    
    # Calculate histogram
    counts, _ = np.histogram(values, bins=bin_edges)
    
    # Create bin info with centers and ranges
    bin_centers = []
    bin_ranges = []
    for i in range(len(bin_edges) - 1):
        if use_log_bins:
            # Geometric mean for log scale
            center = np.sqrt(bin_edges[i] * bin_edges[i + 1])
        else:
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
        'use_log_bins': use_log_bins
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
    
    if selected_bin >= len(histogram_data['bin_ranges']):
        print(f"DEBUG: Selected bin {selected_bin} out of range (max: {len(histogram_data['bin_ranges'])-1})")
        return []
    
    # Get the range for the selected bin
    bin_min, bin_max = histogram_data['bin_ranges'][selected_bin]
    print(f"DEBUG: Using bin {selected_bin} range: {bin_min:.2f} to {bin_max:.2f}")
    
    # Find indices of values that fall within this bin
    stable_indices = np.where((values >= bin_min) & (values < bin_max))[0]
    stable_values = values[stable_indices]
    
    print(f"DEBUG: Found {len(stable_indices)} stable samples (range: {np.min(stable_values):.2f} to {np.max(stable_values):.2f})")
    
    return stable_indices.tolist()

def generate_plot_data(test_id, test_data, plot_options=None):
    """Generate plot data for a specific test with histogram-based control limits."""
    if plot_options is None:
        plot_options = {
            'show_spec_limits': True,
            'show_control_limits': True,
            'selected_bin': None  # Auto-select dominant bin
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
    bin_centers, bin_counts, histogram_data = generate_histogram_bins(values)
    
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
        
        # Generate plot data
        test_data = all_tests[test_id]
        plot_data = generate_plot_data(test_id, test_data, plot_options)
        
        if not plot_data:
            return jsonify({'error': 'Insufficient data for plotting'}), 400
        
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
        
        # Generate updated plot data
        test_data = all_tests[test_id]
        plot_data = generate_plot_data(test_id, test_data, plot_options)
        
        if not plot_data:
            return jsonify({'error': 'Insufficient data for plotting'}), 400
        
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
        
        print(f"DEBUG: Moved test '{test_id}' from numeric to non-numeric ({len(non_numeric_data)} data points)")
        
        return jsonify({
            'success': True,
            'message': f'Test "{test_id}" moved to non-numeric tests',
            'moved_data_points': len(non_numeric_data)
        })
        
    except Exception as e:
        print(f"DEBUG: Move test error: {str(e)}")
        return jsonify({'error': f'Move test error: {str(e)}'}), 500

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
