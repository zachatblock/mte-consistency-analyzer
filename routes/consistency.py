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

def detect_outliers_multiple_methods(values, test_id=""):
    """Detect outliers using multiple methods and return comprehensive outlier information."""
    values = np.array(values)
    n = len(values)
    
    if n < 10:  # Need reasonable sample size for outlier detection
        return {
            'has_outliers': False,
            'outlier_indices': [],
            'outlier_values': [],
            'methods_used': [],
            'outlier_summary': f"Insufficient samples ({n}) for outlier detection"
        }
    
    outlier_methods = {}
    
    # Method 1: IQR (Interquartile Range) - Classic and robust
    q1 = np.percentile(values, 25)
    q3 = np.percentile(values, 75)
    iqr = q3 - q1
    
    if iqr > 0:  # Avoid division by zero
        iqr_lower = q1 - 1.5 * iqr
        iqr_upper = q3 + 1.5 * iqr
        iqr_outliers = (values < iqr_lower) | (values > iqr_upper)
        outlier_methods['IQR'] = {
            'outliers': iqr_outliers,
            'count': np.sum(iqr_outliers),
            'bounds': (iqr_lower, iqr_upper),
            'description': f"IQR method: Q1={q1:.2f}, Q3={q3:.2f}, bounds=[{iqr_lower:.2f}, {iqr_upper:.2f}]"
        }
    
    # Method 2: Modified Z-Score - Good for detecting extreme outliers
    median = np.median(values)
    mad = np.median(np.abs(values - median))  # Median Absolute Deviation
    
    if mad > 0:  # Avoid division by zero
        modified_z_scores = 0.6745 * (values - median) / mad
        z_threshold = 3.5  # Common threshold for modified z-score
        z_outliers = np.abs(modified_z_scores) > z_threshold
        outlier_methods['Modified_Z'] = {
            'outliers': z_outliers,
            'count': np.sum(z_outliers),
            'threshold': z_threshold,
            'description': f"Modified Z-Score: median={median:.2f}, MAD={mad:.2f}, threshold={z_threshold}"
        }
    
    # Method 3: Isolation Forest - Machine learning approach
    try:
        iso_forest = IsolationForest(contamination=0.1, random_state=42)  # Expect 10% outliers max
        outlier_predictions = iso_forest.fit_predict(values.reshape(-1, 1))
        iso_outliers = outlier_predictions == -1
        outlier_methods['Isolation_Forest'] = {
            'outliers': iso_outliers,
            'count': np.sum(iso_outliers),
            'contamination': 0.1,
            'description': f"Isolation Forest: contamination=0.1, detected {np.sum(iso_outliers)} outliers"
        }
    except Exception as e:
        print(f"DEBUG: Isolation Forest failed for {test_id}: {e}")
    
    # Method 4: Statistical outliers (beyond 3 standard deviations)
    mean_val = np.mean(values)
    std_val = np.std(values)
    if std_val > 0:
        stat_outliers = np.abs(values - mean_val) > 3 * std_val
        outlier_methods['Statistical_3Sigma'] = {
            'outliers': stat_outliers,
            'count': np.sum(stat_outliers),
            'bounds': (mean_val - 3*std_val, mean_val + 3*std_val),
            'description': f"3-Sigma: mean={mean_val:.2f}, std={std_val:.2f}, bounds=[{mean_val-3*std_val:.2f}, {mean_val+3*std_val:.2f}]"
        }
    
    # Combine methods using voting (at least 2 methods must agree)
    if len(outlier_methods) >= 2:
        all_outlier_arrays = [method['outliers'] for method in outlier_methods.values()]
        outlier_votes = np.sum(all_outlier_arrays, axis=0)
        
        # Require at least 2 methods to agree for an outlier
        consensus_outliers = outlier_votes >= 2
        consensus_count = np.sum(consensus_outliers)
        
        # For resistance tests specifically, be more aggressive with outlier detection
        if 'res' in test_id.lower() or 'resistance' in test_id.lower():
            # For resistance tests, look for values that are >100x the median (likely open circuits)
            resistance_threshold = 100 * median if median > 0 else 1e6
            extreme_outliers = values > resistance_threshold
            consensus_outliers = consensus_outliers | extreme_outliers
            consensus_count = np.sum(consensus_outliers)
            
            if np.sum(extreme_outliers) > 0:
                outlier_methods['Resistance_Extreme'] = {
                    'outliers': extreme_outliers,
                    'count': np.sum(extreme_outliers),
                    'threshold': resistance_threshold,
                    'description': f"Resistance extreme: >100x median ({resistance_threshold:.0f}Ω)"
                }
        
    else:
        consensus_outliers = np.zeros(n, dtype=bool)
        consensus_count = 0
    
    # Calculate impact on statistics
    original_mean = np.mean(values)
    original_std = np.std(values)
    
    if consensus_count > 0:
        clean_values = values[~consensus_outliers]
        if len(clean_values) > 3:  # Need minimum samples for statistics
            clean_mean = np.mean(clean_values)
            clean_std = np.std(clean_values)
            
            mean_change_pct = abs(clean_mean - original_mean) / abs(original_mean) * 100 if original_mean != 0 else 0
            std_change_pct = abs(clean_std - original_std) / abs(original_std) * 100 if original_std != 0 else 0
            
            impact_significant = mean_change_pct > 5 or std_change_pct > 10  # 5% mean change or 10% std change
        else:
            clean_mean = original_mean
            clean_std = original_std
            mean_change_pct = 0
            std_change_pct = 0
            impact_significant = False
    else:
        clean_mean = original_mean
        clean_std = original_std
        mean_change_pct = 0
        std_change_pct = 0
        impact_significant = False
    
    # Create summary
    method_names = list(outlier_methods.keys())
    outlier_summary = f"Methods: {', '.join(method_names)}. "
    outlier_summary += f"Consensus: {consensus_count}/{n} outliers ({consensus_count/n*100:.1f}%). "
    outlier_summary += f"Mean change: {mean_change_pct:.1f}%, Std change: {std_change_pct:.1f}%"
    
    return convert_numpy_to_python_types({
        'has_outliers': bool(consensus_count > 0),
        'outlier_indices': np.where(consensus_outliers)[0].tolist(),
        'outlier_values': values[consensus_outliers].tolist(),
        'clean_indices': np.where(~consensus_outliers)[0].tolist(),
        'clean_values': values[~consensus_outliers].tolist(),
        'methods_used': method_names,
        'consensus_count': int(consensus_count),
        'total_samples': int(n),
        'outlier_percentage': float(consensus_count/n*100),
        'original_mean': float(original_mean),
        'original_std': float(original_std),
        'clean_mean': float(clean_mean),
        'clean_std': float(clean_std),
        'mean_change_percent': float(mean_change_pct),
        'std_change_percent': float(std_change_pct),
        'impact_significant': bool(impact_significant),
        'outlier_summary': outlier_summary
    })

def generate_plot_data(test_id, test_data, plot_options=None):
    """Generate plot data for a specific test without creating the actual plot."""
    if plot_options is None:
        plot_options = {
            'show_spec_limits': True,
            'show_control_limits': True,
            'show_outliers': True,
            'clean_outliers': False
        }
    
    df = pd.DataFrame(test_data)
    df = df.sort_values('timestamp')
    
    values = df['value'].values
    test_results = df['test_result'].values
    serial_numbers = df['serial_number'].values
    
    n = len(values)
    if n < 3:
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
    
    # Detect outliers
    outlier_info = detect_outliers_multiple_methods(values, test_id)
    
    # Apply outlier cleaning if requested
    if plot_options.get('clean_outliers') and outlier_info['has_outliers']:
        clean_indices = outlier_info['clean_indices']
        values = values[clean_indices]
        test_results = test_results[clean_indices]
        serial_numbers = serial_numbers[clean_indices]
        n = len(values)
    
    # Calculate statistics
    mean = np.mean(values)
    std = np.std(values, ddof=1)
    ucl = mean + 3 * std
    lcl = mean - 3 * std
    
    # Identify out-of-control points
    out_of_control = (values > ucl) | (values < lcl)
    
    # Categorize by EIF (Engineering Identification Flag) instead of hardcoded SN prefixes
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
        'eif_categories': {str(k): v for k, v in eif_categories.items()},  # Convert keys to strings for JSON
        'pass_indices': np.where(test_results == 'PASS')[0].tolist(),
        'fail_indices': np.where(test_results == 'FAIL')[0].tolist(),
        'ooc_indices': np.where(out_of_control)[0].tolist(),
        'outlier_indices': outlier_info.get('outlier_indices', []) if plot_options.get('show_outliers') else [],
        
        # Statistics
        'mean': float(mean),
        'std': float(std),
        'ucl': float(ucl),
        'lcl': float(lcl),
        'usl': float(usl) if usl is not None else None,
        'lsl': float(lsl) if lsl is not None else None,
        
        # Counts
        'n_samples': int(n),
        'n_pass': int(np.sum(test_results == 'PASS')),
        'n_fail': int(np.sum(test_results == 'FAIL')),
        'eif_counts': {str(k): len(v) for k, v in eif_categories.items()},  # Count by EIF instead of hardcoded SNs
        'n_ooc': int(np.sum(out_of_control)),
        'usl_violations': int(np.sum(values > usl)) if usl is not None else 0,
        'lsl_violations': int(np.sum(values < lsl)) if lsl is not None else 0,
        
        # Metadata
        'test_id': test_id,
        'unit': df['unit'].iloc[0] if 'unit' in df.columns and df['unit'].iloc[0] else '',
        'has_limits': usl is not None or lsl is not None,
        'outlier_info': outlier_info,
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
