"""
Consistency plots routes for the W3A Flask application.
Handles the consistency plot generation functionality.
"""

from flask import Blueprint, render_template, request, jsonify, send_file, flash, redirect, url_for
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

warnings.filterwarnings('ignore')

consistency_bp = Blueprint('consistency', __name__)

def extract_sn_from_filename(filename):
    """Extract serial number from filename."""
    match = re.match(r'([\dA-Z]+)_', filename)
    if match:
        return match.group(1)
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
                        # NOT in the log_dir which might be the user's Desktop!
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
        print(f"DEBUG: Created {len(temp_extracts)} temporary extraction directories")
        print(f"DEBUG: Sample CSV paths: {csv_files[:5] if csv_files else 'None'}")
        return csv_files
        
    except Exception as e:
        print(f"DEBUG: Error in find_parametric_csv: {e}")
        return csv_files

def parse_test_plan_yaml(yaml_path):
    """Parse test plan YAML file to get test groups and steps."""
    try:
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        
        test_info = {
            'imports': data.get('imports', []),
            'groups': {}
        }
        
        groups = data.get('groups', {})
        for group_name, group_data in groups.items():
            test_info['groups'][group_name] = {
                'index': group_data.get('index', 0),
                'skip': group_data.get('skip', False),
                'steps': group_data.get('steps', [])
            }
        
        return test_info
    except Exception as e:
        print(f"Error parsing test plan YAML {yaml_path}: {e}")
        return None

def parse_step_config_yaml(yaml_path):
    """Parse step config YAML file to get test step definitions."""
    try:
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        
        step_configs = {}
        for step_name, step_data in data.items():
            if isinstance(step_data, dict):
                step_configs[step_name] = {
                    'description': step_data.get('DESCRIPTION', ''),
                    'function': step_data.get('FUNCTION', ''),
                    'arguments': step_data.get('ARGUMENTS', ''),
                    'timeout': step_data.get('TIMEOUT', ''),
                    'type': step_data.get('TYPE', ''),
                    'unit': step_data.get('UNIT', ''),
                    'comp': step_data.get('COMP', ''),
                    'low': step_data.get('LOW', ''),
                    'high': step_data.get('HIGH', ''),
                    'skip': step_data.get('SKIP', ''),
                    'fail_count': step_data.get('FAIL_COUNT', '1')
                }
        
        return step_configs
    except Exception as e:
        print(f"Error parsing step config YAML {yaml_path}: {e}")
        return None

def get_test_limits_from_configs(test_plan_path, step_config_path):
    """Extract test limits from config files for comparison with actual results."""
    test_plan = parse_test_plan_yaml(test_plan_path) if test_plan_path else None
    step_config = parse_step_config_yaml(step_config_path) if step_config_path else None
    
    if not test_plan or not step_config:
        return {}
    
    test_limits = {}
    
    # Go through each group and step in the test plan
    for group_name, group_data in test_plan['groups'].items():
        if group_data.get('skip', False):
            continue
            
        for step_info in group_data['steps']:
            if isinstance(step_info, list) and len(step_info) >= 1:
                step_name = step_info[0]
                
                if step_name in step_config:
                    config = step_config[step_name]
                    if config['type'] == 'FLOAT' and config['low'] and config['high']:
                        try:
                            low_limit = float(config['low'])
                            high_limit = float(config['high'])
                            test_limits[step_name] = {
                                'low': low_limit,
                                'high': high_limit,
                                'unit': config.get('unit', ''),
                                'description': config.get('description', ''),
                                'group': group_name
                            }
                        except (ValueError, TypeError):
                            continue
    
    return test_limits

def extract_zip_files(zip_dir, temp_dir):
    """Extract all zip files to temporary directory."""
    extracted_count = 0
    for root, dirs, files in os.walk(zip_dir):
        for file in files:
            if file.endswith('.zip'):
                zip_path = os.path.join(root, file)
                try:
                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        zip_ref.extractall(temp_dir)
                        extracted_count += 1
                except Exception as e:
                    print(f"Error extracting {zip_path}: {e}")
    return extracted_count

def parse_csv_file(csv_path):
    """Parse CSV file and extract test data with limits."""
    test_data = []
    filename = os.path.basename(csv_path)
    serial_number = extract_sn_from_filename(filename)
    
    # Debug: Track what we find in each file
    total_rows = 0
    float_rows = 0
    other_types = set()
    
    try:
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                total_rows += 1
                try:
                    test_id = row.get('test_id', '')
                    test_type = row.get('type', '')
                    ret_value = row.get('ret', '')
                    unit = row.get('unit', '')
                    execution_time = row.get('execution_time', '')
                    test_result = row.get('test_result', '')
                    lo_limit = row.get('lo_limit', '')
                    hi_limit = row.get('hi_limit', '')
                    
                    # Track all test types we see
                    if test_type:
                        other_types.add(test_type)
                    
                    # Try to parse as numeric even if not explicitly marked as FLOAT
                    # This catches INT, DOUBLE, or unmarked numeric values
                    try:
                        value = float(ret_value)
                        if abs(value) > 1e30:
                            continue
                        
                        float_rows += 1
                        
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
                        
                        test_data.append({
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
                        # Not a numeric value, skip
                        continue
                except Exception as e:
                    continue
        
        # Debug output for files with no data
        if not test_data and total_rows > 0:
            print(f"DEBUG: {filename} - {total_rows} total rows, {float_rows} numeric, types found: {other_types}")
            
    except Exception as e:
        print(f"Error reading {csv_path}: {e}")
    
    return test_data

def collect_all_test_data(log_dir):
    """Collect all test data from all CSV files."""
    csv_files = find_parametric_csv(log_dir)
    all_tests = defaultdict(list)
    
    print(f"DEBUG: Starting to process {len(csv_files)} CSV files for test data collection")
    
    files_processed = 0
    files_with_data = 0
    total_data_points = 0
    
    for i, csv_file in enumerate(csv_files):
        if i % 100 == 0:  # Progress update every 100 files
            print(f"DEBUG: Processing CSV file {i+1}/{len(csv_files)}: {os.path.basename(csv_file)}")
        
        # Check if file exists before trying to parse
        if not os.path.exists(csv_file):
            print(f"DEBUG: WARNING - CSV file does not exist: {csv_file}")
            continue
            
        test_data = parse_csv_file(csv_file)
        files_processed += 1
        
        if test_data:
            files_with_data += 1
            for test in test_data:
                all_tests[test['test_id']].append(test)
                total_data_points += 1
        else:
            print(f"DEBUG: No test data extracted from: {os.path.basename(csv_file)}")
    
    print(f"DEBUG: Collection complete - Processed {files_processed} files, {files_with_data} had data")
    print(f"DEBUG: Total data points collected: {total_data_points}")
    print(f"DEBUG: Unique test IDs found: {len(all_tests)}")
    
    # Show sample counts for first few tests
    test_sample_counts = [(test_id, len(data)) for test_id, data in all_tests.items()]
    test_sample_counts.sort(key=lambda x: x[1], reverse=True)
    
    print(f"DEBUG: Top 5 tests by sample count:")
    for test_id, count in test_sample_counts[:5]:
        print(f"DEBUG:   {test_id}: {count} samples")
    
    return all_tests, len(csv_files)

def create_consistency_plot(test_id, test_data, output_dir):
    """Create a consistency plot colored by serial number prefix."""
    df = pd.DataFrame(test_data)
    df = df.sort_values('timestamp')
    
    values = df['value'].values
    test_results = df['test_result'].values
    serial_numbers = df['serial_number'].values
    
    n = len(values)
    if n < 3:
        return None
    
    # Calculate statistics
    mean = np.mean(values)
    std = np.std(values, ddof=1)
    ucl = mean + 3 * std
    lcl = mean - 3 * std
    
    # Identify out-of-control points
    out_of_control = (values > ucl) | (values < lcl)
    
    # Extract limits from the data itself (they're in the CSV files!)
    usl = None
    lsl = None
    has_limits = False
    
    # Check if any data points have limit information
    for data_point in test_data:
        if data_point.get('high_limit') is not None:
            usl = data_point['high_limit']
            has_limits = True
            break
    
    for data_point in test_data:
        if data_point.get('low_limit') is not None:
            lsl = data_point['low_limit']
            has_limits = True
            break
    
    print(f"DEBUG: Test {test_id} - Found limits in data: USL={usl}, LSL={lsl}")
    
    # Separate by SN prefix (549 or 602)
    sn_starts_549 = np.array([str(sn).startswith('549') if sn else False for sn in serial_numbers])
    sn_starts_602 = np.array([str(sn).startswith('602') if sn else False for sn in serial_numbers])
    
    # Combine SN and pass/fail and OOC
    sn549_pass_ic = sn_starts_549 & (test_results == 'PASS') & ~out_of_control
    sn549_fail_ic = sn_starts_549 & (test_results == 'FAIL') & ~out_of_control
    sn549_pass_ooc = sn_starts_549 & (test_results == 'PASS') & out_of_control
    sn549_fail_ooc = sn_starts_549 & (test_results == 'FAIL') & out_of_control
    
    sn602_pass_ic = sn_starts_602 & (test_results == 'PASS') & ~out_of_control
    sn602_fail_ic = sn_starts_602 & (test_results == 'FAIL') & ~out_of_control
    sn602_pass_ooc = sn_starts_602 & (test_results == 'PASS') & out_of_control
    sn602_fail_ooc = sn_starts_602 & (test_results == 'FAIL') & out_of_control
    
    # Create plot
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(14, 6))
    x = range(len(values))
    
    # Plot SN starting with 549 (RED)
    if np.any(sn549_pass_ic):
        ax.plot(np.where(sn549_pass_ic)[0], values[sn549_pass_ic], 'ro', markersize=4, alpha=0.6, label='SN 549 PASS')
    if np.any(sn549_fail_ic):
        ax.plot(np.where(sn549_fail_ic)[0], values[sn549_fail_ic], 'r^', markersize=6, alpha=0.6, label='SN 549 FAIL')
    if np.any(sn549_pass_ooc):
        ax.plot(np.where(sn549_pass_ooc)[0], values[sn549_pass_ooc], 'r^', markersize=10, markeredgewidth=2, markerfacecolor='none', label='SN 549 OOC PASS', zorder=5)
    if np.any(sn549_fail_ooc):
        ax.plot(np.where(sn549_fail_ooc)[0], values[sn549_fail_ooc], 'rx', markersize=10, markeredgewidth=2, label='SN 549 OOC FAIL', zorder=5)
    
    # Plot SN starting with 602 (BLUE)
    if np.any(sn602_pass_ic):
        ax.plot(np.where(sn602_pass_ic)[0], values[sn602_pass_ic], 'bo', markersize=4, alpha=0.6, label='SN 602 PASS')
    if np.any(sn602_fail_ic):
        ax.plot(np.where(sn602_fail_ic)[0], values[sn602_fail_ic], 'b^', markersize=6, alpha=0.6, label='SN 602 FAIL')
    if np.any(sn602_pass_ooc):
        ax.plot(np.where(sn602_pass_ooc)[0], values[sn602_pass_ooc], 'b^', markersize=10, markeredgewidth=2, markerfacecolor='none', label='SN 602 OOC PASS', zorder=5)
    if np.any(sn602_fail_ooc):
        ax.plot(np.where(sn602_fail_ooc)[0], values[sn602_fail_ooc], 'bx', markersize=10, markeredgewidth=2, label='SN 602 OOC FAIL', zorder=5)
    
    # Connect all points with a line
    ax.plot(x, values, 'gray', linewidth=0.5, alpha=0.3, zorder=1)
    
    # Plot control limits (±3σ)
    ax.axhline(y=mean, color='g', linestyle='-', linewidth=2, label=f'Mean = {mean:.2f}')
    ax.axhline(y=ucl, color='orange', linestyle='--', linewidth=2, label=f'UCL (+3σ) = {ucl:.2f}')
    ax.axhline(y=lcl, color='orange', linestyle='--', linewidth=2, label=f'LCL (-3σ) = {lcl:.2f}')
    
    # Plot USL/LSL limits if available from the CSV data (blue dashed lines)
    usl_violations = 0
    lsl_violations = 0
    if has_limits:
        if usl is not None:
            ax.axhline(y=usl, color='blue', linestyle='--', linewidth=2, alpha=0.8, label=f'USL = {usl:.2f}')
            usl_violations = np.sum(values > usl)
        if lsl is not None:
            ax.axhline(y=lsl, color='blue', linestyle='--', linewidth=2, alpha=0.8, label=f'LSL = {lsl:.2f}')
            lsl_violations = np.sum(values < lsl)
    
    unit = df['unit'].iloc[0] if 'unit' in df.columns and df['unit'].iloc[0] else ''
    unit_str = f" ({unit})" if unit else ""
    
    ax.set_xlabel('Sample Number', fontsize=12)
    ax.set_ylabel(f'Measured Value{unit_str}', fontsize=12)
    ax.set_title(f'Consistency Plot: {test_id}\nn={n}, μ={mean:.2f}, σ={std:.2f}', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    
    n_pass = np.sum(test_results == 'PASS')
    n_fail = np.sum(test_results == 'FAIL')
    n_sn549 = np.sum(sn_starts_549)
    n_sn602 = np.sum(sn_starts_602)
    
    stats_text = f'Statistics:\n'
    stats_text += f' Total Samples: {n}\n'
    stats_text += f' SN 549: {n_sn549}\n'
    stats_text += f' SN 602: {n_sn602}\n'
    stats_text += f' PASS: {n_pass} ({100*n_pass/n:.1f}%)\n'
    stats_text += f' FAIL: {n_fail} ({100*n_fail/n:.1f}%)\n'
    stats_text += f' Mean: {mean:.2f}\n'
    stats_text += f' Std Dev: {std:.2f}\n'
    stats_text += f' Min: {np.min(values):.2f}\n'
    stats_text += f' Max: {np.max(values):.2f}\n'
    stats_text += f' OOC Total: {np.sum(out_of_control)}'
    
    if has_limits:
        if usl is not None:
            stats_text += f'\n USL Violations: {usl_violations}'
        if lsl is not None:
            stats_text += f'\n LSL Violations: {lsl_violations}'
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            fontsize=9, family='monospace')
    
    plt.tight_layout()
    
    safe_filename = test_id.replace('/', '_').replace('\\', '_')
    output_path = os.path.join(output_dir, f'{safe_filename}.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return {
        'test_id': test_id,
        'n_samples': n,
        'n_sn549': int(n_sn549),
        'n_sn602': int(n_sn602),
        'n_pass': int(n_pass),
        'n_fail': int(n_fail),
        'mean': mean,
        'std': std,
        'ucl': ucl,
        'lcl': lcl,
        'min': np.min(values),
        'max': np.max(values),
        'out_of_control_total': int(np.sum(out_of_control)),
        'usl_violations': int(usl_violations),
        'lsl_violations': int(lsl_violations),
        'has_limits': bool(has_limits),
        'usl': float(usl) if usl is not None else None,
        'lsl': float(lsl) if lsl is not None else None,
        'plot_path': output_path
    }

@consistency_bp.route('/')
def index():
    """Consistency plots main page."""
    # Set default example paths
    example_config_dir = "/Users/zachstanziano/Documents/Github/hw-factory_test_nextgen_w3a/project_config"
    return render_template('consistency/index.html', example_config_dir=example_config_dir)

@consistency_bp.route('/generate', methods=['POST'])
def generate_plots():
    """Generate consistency plots from uploaded data."""
    try:
        # Get form data
        log_directory = request.form.get('log_directory', '').strip()
        test_plan_path = request.form.get('test_plan_path', '').strip()
        step_config_path = request.form.get('step_config_path', '').strip()
        
        if not log_directory or not os.path.exists(log_directory):
            flash('Please provide a valid log directory path.', 'error')
            return redirect(url_for('consistency.index'))
        
        # Validate config files if provided
        test_limits = {}
        if test_plan_path and step_config_path:
            if not os.path.exists(test_plan_path):
                flash('Test plan file does not exist.', 'error')
                return redirect(url_for('consistency.index'))
            if not os.path.exists(step_config_path):
                flash('Step config file does not exist.', 'error')
                return redirect(url_for('consistency.index'))
            
            # Parse config files to get test limits
            test_limits = get_test_limits_from_configs(test_plan_path, step_config_path)
            flash(f'Loaded {len(test_limits)} test limits from config files.', 'info')
        
        # Create output directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = os.path.join(tempfile.gettempdir(), f'w3a_consistency_{timestamp}')
        os.makedirs(output_dir, exist_ok=True)
        
        # Check if directory contains zip files
        zip_files = [f for f in os.listdir(log_directory) if f.endswith('.zip')]
        
        if zip_files:
            # Extract zip files to temporary directory
            temp_extract_dir = os.path.join(tempfile.gettempdir(), f'w3a_extract_{timestamp}')
            os.makedirs(temp_extract_dir, exist_ok=True)
            
            extracted_count = extract_zip_files(log_directory, temp_extract_dir)
            flash(f'Extracted {extracted_count} zip files for processing.', 'info')
            
            # Process extracted files
            all_tests, csv_count = collect_all_test_data(temp_extract_dir)
        else:
            # Process directory directly
            all_tests, csv_count = collect_all_test_data(log_directory)
        
        if not all_tests:
            flash('No parametric test data found in the specified directory.', 'error')
            return redirect(url_for('consistency.index'))
        
        # Generate plots
        stats_list = []
        plot_files = []
        
        for test_id, test_data in all_tests.items():
            stats = create_consistency_plot(test_id, test_data, output_dir)
            if stats:
                stats_list.append(stats)
                plot_files.append(stats['plot_path'])
        
        # Generate summary report
        if stats_list:
            df = pd.DataFrame(stats_list)
            df = df.sort_values('n_samples', ascending=False)
            
            summary_path = os.path.join(output_dir, 'summary_report.csv')
            df.to_csv(summary_path, index=False)
        
        # Store results in session or return data
        results = {
            'output_dir': output_dir,
            'total_tests': len(all_tests),
            'total_samples': sum(len(data) for data in all_tests.values()),
            'csv_files_processed': csv_count,
            'plots_generated': len(plot_files),
            'summary_path': summary_path if stats_list else None,
            'timestamp': timestamp
        }
        
        flash(f'Successfully generated {len(plot_files)} consistency plots!', 'success')
        return render_template('consistency/results.html', results=results, stats=stats_list)
        
    except Exception as e:
        flash(f'Error generating plots: {str(e)}', 'error')
        return redirect(url_for('consistency.index'))

@consistency_bp.route('/download/<path:filename>')
def download_file(filename):
    """Download generated files."""
    try:
        # Security: only allow files from temp directories
        if not filename.startswith('/tmp/'):
            return "Access denied", 403
        
        if os.path.exists(filename):
            return send_file(filename, as_attachment=True)
        else:
            return "File not found", 404
    except Exception as e:
        return f"Error downloading file: {str(e)}", 500

@consistency_bp.route('/upload_file', methods=['POST'])
def upload_file():
    """Handle file uploads from the file input fields."""
    try:
        file_type = request.form.get('file_type')
        file_path = request.form.get('file_path', '').strip()
        
        if not file_type or file_type not in ['csv', 'yaml', 'step']:
            return jsonify({'error': 'Invalid file type'}), 400
        
        # Handle file upload vs file path
        if 'file' in request.files and request.files['file'].filename:
            # File upload
            file = request.files['file']
            filename = secure_filename(file.filename)
            
            # Basic file type validation
            if file_type == 'csv' and not filename.lower().endswith('.csv'):
                return jsonify({'error': 'Invalid file type. Expected .csv'}), 400
            elif file_type in ['yaml', 'step'] and not (filename.lower().endswith('.yaml') or filename.lower().endswith('.yml')):
                return jsonify({'error': 'Invalid file type. Expected .yaml or .yml'}), 400
            
            # Read file content
            file_content = file.read().decode('utf-8')
            
            return jsonify({
                'success': True,
                'filename': filename,
                'content': file_content[:2000] + ('\n\n... (content truncated)' if len(file_content) > 2000 else ''),
                'full_length': len(file_content)
            })
            
        elif file_path:
            # File path
            if not os.path.exists(file_path):
                return jsonify({'error': 'File path does not exist'}), 400
            
            # Basic file type validation
            path_lower = file_path.lower()
            if file_type == 'csv' and not path_lower.endswith('.csv'):
                return jsonify({'error': 'Invalid file type. Expected .csv'}), 400
            elif file_type in ['yaml', 'step'] and not (path_lower.endswith('.yaml') or path_lower.endswith('.yml')):
                return jsonify({'error': 'Invalid file type. Expected .yaml or .yml'}), 400
            
            # Read file content
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    file_content = f.read()
                
                return jsonify({
                    'success': True,
                    'filename': os.path.basename(file_path),
                    'content': file_content[:2000] + ('\n\n... (content truncated)' if len(file_content) > 2000 else ''),
                    'full_length': len(file_content)
                })
            except Exception as e:
                return jsonify({'error': f'Error reading file: {str(e)}'}), 500
        
        else:
            return jsonify({'error': 'No file or file path provided'}), 400
            
    except Exception as e:
        return jsonify({'error': f'Upload error: {str(e)}'}), 500

@consistency_bp.route('/read_file_path', methods=['POST'])
def read_file_path():
    """Read file content from a file path."""
    try:
        data = request.get_json()
        file_path = data.get('file_path', '').strip()
        file_type = data.get('file_type', '')
        
        if not file_path or not file_type:
            return jsonify({'error': 'File path and type required'}), 400
        
        if not os.path.exists(file_path):
            return jsonify({'error': 'File does not exist'}), 400
        
        # Basic file type validation
        path_lower = file_path.lower()
        if file_type == 'csv' and not path_lower.endswith('.csv'):
            return jsonify({'error': 'Invalid file type. Expected .csv'}), 400
        elif file_type in ['yaml', 'step'] and not (path_lower.endswith('.yaml') or path_lower.endswith('.yml')):
            return jsonify({'error': 'Invalid file type. Expected .yaml or .yml'}), 400
        
        # Read and return file content
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            return jsonify({
                'success': True,
                'filename': os.path.basename(file_path),
                'content': content[:2000] + ('\n\n... (content truncated)' if len(content) > 2000 else ''),
                'full_length': len(content)
            })
        except Exception as e:
            return jsonify({'error': f'Error reading file: {str(e)}'}), 500
            
    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'}), 500

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

# Failure Analysis Functions
def find_failed_logs(log_dir):
    """Find all failed log files (ending with _F.zip) or failed directories with parametric CSVs."""
    print(f"DEBUG: Searching for failed logs in directory: {log_dir}")
    failed_files = []
    failed_dirs = []
    total_files = 0
    
    for root, dirs, files in os.walk(log_dir):
        print(f"DEBUG: Checking directory: {root}")
        
        # Look for _F.zip files (compressed failed logs)
        for file in files:
            total_files += 1
            if file.endswith('_F.zip'):
                failed_file_path = os.path.join(root, file)
                failed_files.append(failed_file_path)
                print(f"DEBUG: Found failed ZIP log: {failed_file_path}")
        
        # Look for directories that contain failed units (with parametric CSVs)
        # These might be named like: 549WP92002000160_FactoryTestProduct_w3a_mlb_test_20251211_161247_slot1_P
        # or: 550WP92002100121_w3a_mlb_test_20251218_151444_slot1_P
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            # Check if this directory contains a parametric CSV (indicating it's a unit log directory)
            for subroot, subdirs, subfiles in os.walk(dir_path):
                for subfile in subfiles:
                    if subfile.endswith('_parametric.csv'):
                        # Check if this is a failed unit by looking at the directory name or CSV content
                        # For now, we'll assume any directory with parametric CSV could be failed
                        # The actual failure determination will be done during CSV parsing
                        failed_dirs.append(dir_path)
                        print(f"DEBUG: Found potential failed directory with parametric CSV: {dir_path}")
                        break
                break  # Only check the first level
    
    print(f"DEBUG: Searched {total_files} total files")
    print(f"DEBUG: Found {len(failed_files)} failed ZIP files")
    print(f"DEBUG: Found {len(failed_dirs)} directories with parametric CSVs")
    
    # Return both ZIP files and directories - we'll handle both in the parsing function
    return failed_files + failed_dirs

def extract_and_parse_failed_log(path, temp_dir):
    """Extract a failed log ZIP or process a directory and parse its parametric CSV"""
    try:
        print(f"DEBUG: Processing path: {path}")
        
        # Check if this is a ZIP file or a directory
        if os.path.isfile(path) and path.endswith('.zip'):
            print(f"DEBUG: Extracting unit ZIP: {path}")
            
            # Create extraction directory
            extract_dir = os.path.join(temp_dir, os.path.basename(path).replace('.zip', ''))
            os.makedirs(extract_dir, exist_ok=True)
            print(f"DEBUG: Created extraction directory: {extract_dir}")
            
            # Extract the ZIP
            with zipfile.ZipFile(path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
                print(f"DEBUG: Extracted {len(zip_ref.namelist())} files from unit ZIP")
            
            # Show what was extracted
            print(f"DEBUG: Contents of extracted unit ZIP:")
            for root, dirs, files in os.walk(extract_dir):
                level = root.replace(extract_dir, '').count(os.sep)
                indent = ' ' * 2 * level
                print(f"DEBUG: {indent}{os.path.basename(root)}/")
                subindent = ' ' * 2 * (level + 1)
                for file in files:
                    print(f"DEBUG: {subindent}{file}")
            
            search_dir = extract_dir
            
        elif os.path.isdir(path):
            print(f"DEBUG: Processing directory: {path}")
            search_dir = path
            
            # Show directory contents
            print(f"DEBUG: Contents of directory:")
            for root, dirs, files in os.walk(search_dir):
                level = root.replace(search_dir, '').count(os.sep)
                indent = ' ' * 2 * level
                print(f"DEBUG: {indent}{os.path.basename(root)}/")
                subindent = ' ' * 2 * (level + 1)
                for file in files:
                    print(f"DEBUG: {subindent}{file}")
        else:
            print(f"DEBUG: Invalid path type: {path}")
            return {}
        
        # Find the parametric CSV file
        for root, dirs, files in os.walk(search_dir):
            for file in files:
                if file.endswith('_parametric.csv'):
                    csv_path = os.path.join(root, file)
                    print(f"DEBUG: Found parametric CSV: {csv_path}")
                    
                    # Parse the CSV and check if it actually represents a failed unit
                    failure_data = parse_parametric_csv_for_failures(csv_path)
                    if failure_data and (failure_data.get('first_error_item') or failure_data.get('overall_result') == 'FAIL'):
                        print(f"DEBUG: Confirmed failed unit - first error: {failure_data.get('first_error_item')}")
                        return failure_data
                    else:
                        print(f"DEBUG: Unit appears to have passed - skipping")
                        return {}
        
        print(f"DEBUG: No parametric CSV found in {path}")
        return {}
        
    except Exception as e:
        print(f"DEBUG: Error processing {path}: {str(e)}")
        return {}

def parse_parametric_csv_for_failures(csv_path):
    """Parse parametric CSV and extract only the first failure information."""
    filename = os.path.basename(csv_path)
    serial_number = extract_sn_from_filename(filename)
    
    failure_info = {
        'serial_number': serial_number,
        'source_file': filename,
        'first_error_code': None,
        'first_error_item': None,
        'overall_result': None
    }
    
    try:
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                test_id = row.get('test_id', '')
                
                # Extract only the summary information - this is what we care about
                if test_id == 'FIRST_ERROR_CODE':
                    failure_info['first_error_code'] = row.get('ret', '')
                    print(f"DEBUG: Found first error code: {failure_info['first_error_code']}")
                    continue
                elif test_id == 'FIRST_ERROR_ITEM_TYPE':
                    failure_info['first_error_item'] = row.get('ret', '')
                    print(f"DEBUG: Found first error item: {failure_info['first_error_item']}")
                    continue
                elif test_id == 'OVERALL_TEST_RESULT':
                    failure_info['overall_result'] = row.get('ret', '')
                    print(f"DEBUG: Found overall result: {failure_info['overall_result']}")
                    continue
    
    except Exception as e:
        print(f"Error reading CSV {csv_path}: {e}")
        return None
    
    return failure_info

def create_pareto_chart(failure_counts, title, output_path):
    """Create a Pareto chart from failure counts."""
    if not failure_counts:
        return
    
    # Sort by count (descending)
    sorted_failures = failure_counts.most_common()
    
    # Prepare data
    labels = [item[0] for item in sorted_failures]
    counts = [item[1] for item in sorted_failures]
    
    # Calculate cumulative percentages
    total = sum(counts)
    cumulative_counts = np.cumsum(counts)
    cumulative_percentages = (cumulative_counts / total) * 100
    
    # Create figure with two y-axes
    fig, ax1 = plt.subplots(figsize=(14, 8))
    
    # Bar chart for counts
    bars = ax1.bar(range(len(labels)), counts, color='steelblue', alpha=0.7)
    ax1.set_xlabel('Failure Type', fontsize=12)
    ax1.set_ylabel('Failure Count', fontsize=12, color='steelblue')
    ax1.tick_params(axis='y', labelcolor='steelblue')
    
    # Rotate x-axis labels for better readability
    ax1.set_xticks(range(len(labels)))
    ax1.set_xticklabels(labels, rotation=45, ha='right', fontsize=10)
    
    # Add count labels on bars
    for i, (bar, count) in enumerate(zip(bars, counts)):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts)*0.01,
                str(count), ha='center', va='bottom', fontsize=9)
    
    # Line chart for cumulative percentage
    ax2 = ax1.twinx()
    ax2.plot(range(len(labels)), cumulative_percentages, color='red', marker='o', linewidth=2)
    ax2.set_ylabel('Cumulative Percentage (%)', fontsize=12, color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.set_ylim(0, 105)
    
    # Add 80% line
    ax2.axhline(y=80, color='orange', linestyle='--', linewidth=1, alpha=0.7)
    ax2.text(len(labels)*0.7, 82, '80% Line', color='orange', fontsize=10)
    
    # Title and grid
    plt.title(f'Pareto Chart: {title}\\nTotal Failures: {total}', fontsize=14, fontweight='bold', pad=20)
    ax1.grid(True, alpha=0.3)
    
    # Tight layout and save
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

@consistency_bp.route('/analyze_failures', methods=['POST'])
def analyze_failures():
    """Analyze failed logs and generate Pareto charts."""
    try:
        data = request.get_json()
        log_path = data.get('log_path', '').strip()
        
        print(f"DEBUG: Received log_path: '{log_path}'")
        print(f"DEBUG: log_path exists: {os.path.exists(log_path) if log_path else 'No path provided'}")
        print(f"DEBUG: log_path is file: {os.path.isfile(log_path) if log_path else 'No path provided'}")
        print(f"DEBUG: log_path is dir: {os.path.isdir(log_path) if log_path else 'No path provided'}")
        
        if not log_path:
            return jsonify({'error': 'No log directory or file path provided'}), 400
            
        if not os.path.exists(log_path):
            return jsonify({'error': f'Path does not exist: {log_path}'}), 400
        
        # Handle both single ZIP file and directory
        failed_files = []
        
        if os.path.isfile(log_path) and log_path.endswith('.zip'):
            print(f"DEBUG: Processing ZIP file: {log_path}")
            # Single ZIP file provided
            if '_F.zip' in log_path:
                print("DEBUG: This is a single failed log file")
                # This is a single failed log file
                failed_files = [log_path]
            else:
                print("DEBUG: This is a ZIP archive - extracting to search for failed logs")
                # This might be a ZIP containing multiple logs - extract and search
                temp_extract_dir = os.path.join(tempfile.gettempdir(), f'w3a_zip_extract_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
                print(f"DEBUG: Creating temp extract dir: {temp_extract_dir}")
                os.makedirs(temp_extract_dir, exist_ok=True)
                
                try:
                    # Extract the main ZIP file
                    print("DEBUG: Extracting main ZIP file...")
                    with zipfile.ZipFile(log_path, 'r') as zip_ref:
                        zip_ref.extractall(temp_extract_dir)
                    print("DEBUG: Main ZIP extraction complete")
                    
                    # The extracted content contains individual unit ZIP files
                    # We need to look for _F.zip files (failed units) in the extracted content
                    print("DEBUG: Searching for failed unit ZIP files in extracted content...")
                    failed_files = find_failed_logs(temp_extract_dir)
                    print(f"DEBUG: Found {len(failed_files)} failed unit ZIP files")
                    
                    # Debug: show what we found
                    print("DEBUG: Structure of extracted main ZIP:")
                    for root, dirs, files in os.walk(temp_extract_dir):
                        level = root.replace(temp_extract_dir, '').count(os.sep)
                        indent = ' ' * 2 * level
                        print(f"DEBUG: {indent}{os.path.basename(root)}/")
                        subindent = ' ' * 2 * (level + 1)
                        for file in files[:10]:  # Limit to first 10 files per directory
                            print(f"DEBUG: {subindent}{file}")
                        if len(files) > 10:
                            print(f"DEBUG: {subindent}... and {len(files) - 10} more files")
                    
                    if not failed_files:
                        print("DEBUG: No failed logs found - cleaning up and returning error")
                        shutil.rmtree(temp_extract_dir)
                        return jsonify({'error': 'No failed log files (_F.zip) found inside the ZIP archive'}), 400
                        
                except Exception as e:
                    print(f"DEBUG: Error during ZIP extraction: {str(e)}")
                    if os.path.exists(temp_extract_dir):
                        shutil.rmtree(temp_extract_dir)
                    return jsonify({'error': f'Error extracting ZIP file: {str(e)}'}), 400
        elif os.path.isdir(log_path):
            # Directory provided - find all failed logs
            failed_files = find_failed_logs(log_path)
        else:
            return jsonify({'error': 'Path must be either a directory or a ZIP file'}), 400
        
        if not failed_files:
            return jsonify({'error': 'No failed log files (_F.zip) found'}), 400
        
        # Create output directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = os.path.join(tempfile.gettempdir(), f'w3a_failure_analysis_{timestamp}')
        os.makedirs(output_dir, exist_ok=True)
        
        # Create temporary directory for extraction
        temp_dir = os.path.join(tempfile.gettempdir(), f'w3a_failure_temp_{timestamp}')
        os.makedirs(temp_dir, exist_ok=True)
        
        try:
            # Process each failed log - focus only on first error items
            all_failures = []
            first_error_counts = Counter()
            processing_stats = {
                'total_files_to_process': len(failed_files),
                'successfully_parsed': 0,
                'failed_to_parse': 0,
                'no_first_error': 0,
                'extraction_errors': 0
            }
            
            print(f"DEBUG: Starting to process {len(failed_files)} failed files")
            
            for i, failed_file in enumerate(failed_files):
                print(f"DEBUG: Processing file {i+1}/{len(failed_files)}: {failed_file}")
                
                failure_data = extract_and_parse_failed_log(failed_file, temp_dir)
                
                if not failure_data:
                    processing_stats['extraction_errors'] += 1
                    print(f"DEBUG: Failed to extract/parse {failed_file}")
                    continue
                    
                if not failure_data.get('first_error_item'):
                    processing_stats['no_first_error'] += 1
                    print(f"DEBUG: No first error found in {failed_file} - SN: {failure_data.get('serial_number')}")
                    continue
                
                # Successfully processed
                processing_stats['successfully_parsed'] += 1
                all_failures.append(failure_data)
                
                # Count only the first error (which should be the root cause)
                first_error_counts[failure_data['first_error_item']] += 1
                print(f"DEBUG: SUCCESS - Counted first error: {failure_data['first_error_item']} for SN {failure_data.get('serial_number')}")
            
            print(f"DEBUG: Processing complete - Stats: {processing_stats}")
            
            if not all_failures:
                return jsonify({'error': 'No valid failure data found in logs'}), 400
            
            # Generate Pareto chart for first errors only
            create_pareto_chart(first_error_counts, "First Error Types (Root Cause Analysis)", 
                              os.path.join(output_dir, 'first_errors_pareto.png'))
            
            # Generate summary statistics
            top_first_errors = first_error_counts.most_common(10)
            
            return jsonify({
                'success': True,
                'total_failed_logs': len(all_failures),
                'total_failed_files': len(failed_files),
                'unique_first_errors': len(first_error_counts),
                'top_first_errors': top_first_errors,
                'output_dir': output_dir,
                'pareto_charts': {
                    'first_errors': os.path.join(output_dir, 'first_errors_pareto.png')
                }
            })
            
        finally:
            # Cleanup temp directory
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
                
    except Exception as e:
        return jsonify({'error': f'Analysis error: {str(e)}'}), 500

@consistency_bp.route('/generate_consistency_plots', methods=['POST'])
def generate_consistency_plots():
    """Generate consistency plots from uploaded log files."""
    try:
        data = request.get_json()
        log_path = data.get('log_path', '').strip()
        test_plan_path = data.get('test_plan_path', '').strip()
        step_config_path = data.get('step_config_path', '').strip()
        
        print(f"DEBUG: Consistency plots - log_path: '{log_path}'")
        print(f"DEBUG: Consistency plots - test_plan_path: '{test_plan_path}'")
        print(f"DEBUG: Consistency plots - step_config_path: '{step_config_path}'")
        
        if not log_path or not os.path.exists(log_path):
            return jsonify({'error': 'No log directory or file path provided'}), 400
        
        # Create output directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = os.path.join(tempfile.gettempdir(), f'w3a_consistency_{timestamp}')
        os.makedirs(output_dir, exist_ok=True)
        
        # Handle ZIP files - extract if needed
        processing_dir = log_path
        temp_extract_dir = None
        
        if os.path.isfile(log_path) and log_path.endswith('.zip'):
            print("DEBUG: Extracting ZIP file for consistency plot processing...")
            temp_extract_dir = os.path.join(tempfile.gettempdir(), f'w3a_consistency_extract_{timestamp}')
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
            # Get test limits from config files if provided
            test_limits = {}
            if test_plan_path and step_config_path and os.path.exists(test_plan_path) and os.path.exists(step_config_path):
                test_limits = get_test_limits_from_configs(test_plan_path, step_config_path)
                print(f"DEBUG: Loaded {len(test_limits)} test limits from config files")
            
            # Collect all test data
            print("DEBUG: Collecting test data from all CSV files...")
            all_tests, csv_count = collect_all_test_data(processing_dir)
            
            if not all_tests:
                return jsonify({'error': 'No parametric test data found'}), 400
            
            print(f"DEBUG: Found {len(all_tests)} unique tests from {csv_count} CSV files")
            
            # Generate consistency plots for each test
            stats_list = []
            plot_files = []
            
            for test_id, test_data in all_tests.items():
                print(f"DEBUG: Generating plot for test: {test_id} ({len(test_data)} samples)")
                
                # Add USL/LSL limits if available from config
                if test_id in test_limits:
                    limits = test_limits[test_id]
                    # Add limit information to test data for plotting
                    for data_point in test_data:
                        data_point['usl'] = limits['high']
                        data_point['lsl'] = limits['low']
                        data_point['limit_unit'] = limits['unit']
                
                stats = create_consistency_plot_with_limits(test_id, test_data, output_dir, test_limits.get(test_id))
                if stats:
                    stats_list.append(stats)
                    plot_files.append(stats['plot_path'])
            
            # Generate summary report
            summary_path = None
            if stats_list:
                df = pd.DataFrame(stats_list)
                df = df.sort_values('n_samples', ascending=False)
                summary_path = os.path.join(output_dir, 'consistency_summary.csv')
                df.to_csv(summary_path, index=False)
                print(f"DEBUG: Summary report saved to: {summary_path}")
            
            return jsonify({
                'success': True,
                'total_tests': len(all_tests),
                'plots_generated': len(plot_files),
                'csv_files_processed': csv_count,
                'output_dir': output_dir,
                'summary_path': summary_path,
                'test_limits_loaded': len(test_limits),
                'plot_files': plot_files[:10],  # Return first 10 plot paths
                'plot_results': stats_list  # Include plot data with base64 images
            })
            
        finally:
            # Cleanup temp extraction directory
            if temp_extract_dir and os.path.exists(temp_extract_dir):
                shutil.rmtree(temp_extract_dir)
                
    except Exception as e:
        print(f"DEBUG: Consistency plot error: {str(e)}")
        return jsonify({'error': f'Consistency plot generation error: {str(e)}'}), 500

def detect_outliers_multiple_methods(values, test_id=""):
    """Detect outliers using multiple methods and return comprehensive outlier information."""
    values = np.array(values)
    n = len(values)
    
    if n < 10:  # Need reasonable sample size for outlier detection
        return {
            'has_outliers': False,
            'outlier_indices': np.array([]),
            'outlier_values': np.array([]),
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
        from sklearn.ensemble import IsolationForest
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
        # If we see values that are orders of magnitude different, flag them
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
    
    return {
        'has_outliers': consensus_count > 0,
        'outlier_indices': np.where(consensus_outliers)[0],
        'outlier_values': values[consensus_outliers],
        'clean_indices': np.where(~consensus_outliers)[0],
        'clean_values': values[~consensus_outliers],
        'methods_used': method_names,
        'method_details': outlier_methods,
        'consensus_count': consensus_count,
        'total_samples': n,
        'outlier_percentage': consensus_count/n*100,
        'original_mean': original_mean,
        'original_std': original_std,
        'clean_mean': clean_mean,
        'clean_std': clean_std,
        'mean_change_percent': mean_change_pct,
        'std_change_percent': std_change_pct,
        'impact_significant': impact_significant,
        'outlier_summary': outlier_summary
    }

def calculate_limit_recommendations(values, mean, std, current_usl=None, current_lsl=None):
    """Calculate recommended USL/LSL limits based on statistical analysis."""
    recommendations = {}
    
    # Calculate statistical limits (±3σ from mean)
    statistical_ucl = mean + 3 * std
    statistical_lcl = mean - 3 * std
    
    # Recommended limits: slightly wider than ±3σ for manufacturing tolerance
    # Use ±3.5σ or ±4σ depending on the application
    recommended_usl = mean + 3.5 * std
    recommended_lsl = mean - 3.5 * std
    
    # Check if current limits need adjustment
    needs_usl_recommendation = False
    needs_lsl_recommendation = False
    
    if current_usl is None:
        # Missing USL - recommend one
        needs_usl_recommendation = True
        recommendations['reason_usl'] = "Missing USL limit"
    elif abs(current_usl - statistical_ucl) > 0.1 * abs(statistical_ucl):
        # Current USL is more than 10% different from statistical limit
        if current_usl < statistical_ucl:
            needs_usl_recommendation = True
            recommendations['reason_usl'] = f"Current USL ({current_usl:.3f}) is too tight (< UCL {statistical_ucl:.3f})"
        elif current_usl > recommended_usl * 1.2:
            needs_usl_recommendation = True
            recommendations['reason_usl'] = f"Current USL ({current_usl:.3f}) may be too loose (> 1.2×recommended)"
    
    if current_lsl is None:
        # Missing LSL - recommend one
        needs_lsl_recommendation = True
        recommendations['reason_lsl'] = "Missing LSL limit"
    elif abs(current_lsl - statistical_lcl) > 0.1 * abs(statistical_lcl):
        # Current LSL is more than 10% different from statistical limit
        if current_lsl > statistical_lcl:
            needs_lsl_recommendation = True
            recommendations['reason_lsl'] = f"Current LSL ({current_lsl:.3f}) is too tight (> LCL {statistical_lcl:.3f})"
        elif current_lsl < recommended_lsl * 1.2:
            needs_lsl_recommendation = True
            recommendations['reason_lsl'] = f"Current LSL ({current_lsl:.3f}) may be too loose (< 1.2×recommended)"
    
    if needs_usl_recommendation or needs_lsl_recommendation:
        recommendations['has_recommendations'] = True
        recommendations['current_lsl'] = current_lsl
        recommendations['current_usl'] = current_usl
        recommendations['recommended_lsl'] = recommended_lsl
        recommendations['recommended_usl'] = recommended_usl
        recommendations['statistical_lcl'] = statistical_lcl
        recommendations['statistical_ucl'] = statistical_ucl
    else:
        recommendations['has_recommendations'] = False
    
    return recommendations

def create_single_consistency_plot(test_id, values, test_results, serial_numbers, 
                                 usl=None, lsl=None, plot_title_suffix="", 
                                 outlier_info=None, clean_data=False):
    """Create a single consistency plot with the given data."""
    n = len(values)
    if n < 3:
        return None
    
    # Calculate statistics
    mean = np.mean(values)
    std = np.std(values, ddof=1)
    ucl = mean + 3 * std
    lcl = mean - 3 * std
    
    # Identify out-of-control points
    out_of_control = (values > ucl) | (values < lcl)
    
    # Separate by SN prefix (549 or 602)
    sn_starts_549 = np.array([str(sn).startswith('549') if sn else False for sn in serial_numbers])
    sn_starts_602 = np.array([str(sn).startswith('602') if sn else False for sn in serial_numbers])
    
    # Combine SN and pass/fail and OOC
    sn549_pass_ic = sn_starts_549 & (test_results == 'PASS') & ~out_of_control
    sn549_fail_ic = sn_starts_549 & (test_results == 'FAIL') & ~out_of_control
    sn549_pass_ooc = sn_starts_549 & (test_results == 'PASS') & out_of_control
    sn549_fail_ooc = sn_starts_549 & (test_results == 'FAIL') & out_of_control
    
    sn602_pass_ic = sn_starts_602 & (test_results == 'PASS') & ~out_of_control
    sn602_fail_ic = sn_starts_602 & (test_results == 'FAIL') & ~out_of_control
    sn602_pass_ooc = sn_starts_602 & (test_results == 'PASS') & out_of_control
    sn602_fail_ooc = sn_starts_602 & (test_results == 'FAIL') & out_of_control
    
    # Create plot
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(14, 6))
    x = range(len(values))
    
    # Plot SN starting with 549 (RED)
    if np.any(sn549_pass_ic):
        ax.plot(np.where(sn549_pass_ic)[0], values[sn549_pass_ic], 'ro', markersize=4, alpha=0.6, label='SN 549 PASS')
    if np.any(sn549_fail_ic):
        ax.plot(np.where(sn549_fail_ic)[0], values[sn549_fail_ic], 'r^', markersize=6, alpha=0.6, label='SN 549 FAIL')
    if np.any(sn549_pass_ooc):
        ax.plot(np.where(sn549_pass_ooc)[0], values[sn549_pass_ooc], 'r^', markersize=10, markeredgewidth=2, markerfacecolor='none', label='SN 549 OOC PASS', zorder=5)
    if np.any(sn549_fail_ooc):
        ax.plot(np.where(sn549_fail_ooc)[0], values[sn549_fail_ooc], 'rx', markersize=10, markeredgewidth=2, label='SN 549 OOC FAIL', zorder=5)
    
    # Plot SN starting with 602 (BLUE)
    if np.any(sn602_pass_ic):
        ax.plot(np.where(sn602_pass_ic)[0], values[sn602_pass_ic], 'bo', markersize=4, alpha=0.6, label='SN 602 PASS')
    if np.any(sn602_fail_ic):
        ax.plot(np.where(sn602_fail_ic)[0], values[sn602_fail_ic], 'b^', markersize=6, alpha=0.6, label='SN 602 FAIL')
    if np.any(sn602_pass_ooc):
        ax.plot(np.where(sn602_pass_ooc)[0], values[sn602_pass_ooc], 'b^', markersize=10, markeredgewidth=2, markerfacecolor='none', label='SN 602 OOC PASS', zorder=5)
    if np.any(sn602_fail_ooc):
        ax.plot(np.where(sn602_fail_ooc)[0], values[sn602_fail_ooc], 'bx', markersize=10, markeredgewidth=2, label='SN 602 OOC FAIL', zorder=5)
    
    # Highlight outliers if this is the original plot and we have outlier info
    if outlier_info and outlier_info['has_outliers'] and not clean_data:
        outlier_indices = outlier_info['outlier_indices']
        if len(outlier_indices) > 0:
            ax.scatter(outlier_indices, values[outlier_indices], 
                      s=100, facecolors='none', edgecolors='purple', linewidths=3, 
                      label=f'Outliers ({len(outlier_indices)})', zorder=10)
    
    # Connect all points with a line
    ax.plot(x, values, 'gray', linewidth=0.5, alpha=0.3, zorder=1)
    
    # Plot control limits (±3σ)
    ax.axhline(y=mean, color='g', linestyle='-', linewidth=2, label=f'Mean = {mean:.2f}')
    ax.axhline(y=ucl, color='orange', linestyle='--', linewidth=2, label=f'UCL (+3σ) = {ucl:.2f}')
    ax.axhline(y=lcl, color='orange', linestyle='--', linewidth=2, label=f'LCL (-3σ) = {lcl:.2f}')
    
    # Plot USL/LSL limits if available
    usl_violations = 0
    lsl_violations = 0
    has_limits = usl is not None or lsl is not None
    
    if usl is not None:
        ax.axhline(y=usl, color='red', linestyle='-', linewidth=2, alpha=0.8, label=f'USL = {usl:.2f}')
        usl_violations = np.sum(values > usl)
    if lsl is not None:
        ax.axhline(y=lsl, color='red', linestyle='-', linewidth=2, alpha=0.8, label=f'LSL = {lsl:.2f}')
        lsl_violations = np.sum(values < lsl)
    
    ax.set_xlabel('Sample Number', fontsize=12)
    ax.set_ylabel('Measured Value', fontsize=12)
    
    title = f'Consistency Plot: {test_id}{plot_title_suffix}\nn={n}, μ={mean:.2f}, σ={std:.2f}'
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    
    n_pass = np.sum(test_results == 'PASS')
    n_fail = np.sum(test_results == 'FAIL')
    n_sn549 = np.sum(sn_starts_549)
    n_sn602 = np.sum(sn_starts_602)
    
    stats_text = f'Statistics:\n'
    stats_text += f' Total Samples: {n}\n'
    stats_text += f' SN 549: {n_sn549}\n'
    stats_text += f' SN 602: {n_sn602}\n'
    stats_text += f' PASS: {n_pass} ({100*n_pass/n:.1f}%)\n'
    stats_text += f' FAIL: {n_fail} ({100*n_fail/n:.1f}%)\n'
    stats_text += f' Mean: {mean:.2f}\n'
    stats_text += f' Std Dev: {std:.2f}\n'
    stats_text += f' Min: {np.min(values):.2f}\n'
    stats_text += f' Max: {np.max(values):.2f}\n'
    stats_text += f' OOC Total: {np.sum(out_of_control)}'
    
    if has_limits:
        if usl is not None:
            stats_text += f'\n USL Violations: {usl_violations}'
        if lsl is not None:
            stats_text += f'\n LSL Violations: {lsl_violations}'
    
    # Add outlier information to stats if available
    if outlier_info and outlier_info['has_outliers']:
        if clean_data:
            stats_text += f'\n Outliers Removed: {outlier_info["consensus_count"]}'
            stats_text += f'\n Mean Change: {outlier_info["mean_change_percent"]:.1f}%'
            stats_text += f'\n Std Change: {outlier_info["std_change_percent"]:.1f}%'
        else:
            stats_text += f'\n Outliers Detected: {outlier_info["consensus_count"]} ({outlier_info["outlier_percentage"]:.1f}%)'
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            fontsize=9, family='monospace')
    
    plt.tight_layout()
    
    return fig, {
        'n_samples': int(n),
        'n_sn549': int(n_sn549),
        'n_sn602': int(n_sn602),
        'n_pass': int(n_pass),
        'n_fail': int(n_fail),
        'mean': float(mean),
        'std': float(std),
        'ucl': float(ucl),
        'lcl': float(lcl),
        'min': float(np.min(values)),
        'max': float(np.max(values)),
        'out_of_control_total': int(np.sum(out_of_control)),
        'usl_violations': int(usl_violations),
        'lsl_violations': int(lsl_violations),
        'has_limits': bool(has_limits),
        'usl': float(usl) if usl is not None else None,
        'lsl': float(lsl) if lsl is not None else None
    }

def create_consistency_plot_with_limits(test_id, test_data, output_dir, limit_info=None):
    """Create consistency plots with USL/LSL limits and outlier detection - generates dual plots if outliers found."""
    df = pd.DataFrame(test_data)
    df = df.sort_values('timestamp')
    
    values = df['value'].values
    test_results = df['test_result'].values
    serial_numbers = df['serial_number'].values
    
    n = len(values)
    if n < 3:
        return None
    
    # Extract limits from the data itself (they're in the CSV files!)
    usl = None
    lsl = None
    has_limits = False
    
    # Check if any data points have limit information
    for data_point in test_data:
        if data_point.get('high_limit') is not None:
            usl = data_point['high_limit']
            has_limits = True
            break
    
    for data_point in test_data:
        if data_point.get('low_limit') is not None:
            lsl = data_point['low_limit']
            has_limits = True
            break
    
    print(f"DEBUG: Test {test_id} - Found limits in data: USL={usl}, LSL={lsl}")
    
    # Detect outliers using multiple methods
    outlier_info = detect_outliers_multiple_methods(values, test_id)
    print(f"DEBUG: Test {test_id} - Outlier detection: {outlier_info['outlier_summary']}")
    
    # Calculate limit recommendations (using original data)
    original_mean = np.mean(values)
    original_std = np.std(values, ddof=1)
    limit_recommendations = calculate_limit_recommendations(values, original_mean, original_std, usl, lsl)
    
    # Create plots
    plots_generated = []
    
    # Always create the original plot
    fig_original, stats_original = create_single_consistency_plot(
        test_id, values, test_results, serial_numbers, usl, lsl, 
        plot_title_suffix="", outlier_info=outlier_info, clean_data=False
    )
    
    if fig_original:
        # Save original plot
        safe_filename = test_id.replace('/', '_').replace('\\', '_')
        original_path = os.path.join(output_dir, f'{safe_filename}_consistency_original.png')
        fig_original.savefig(original_path, dpi=150, bbox_inches='tight')
        
        # Convert to base64 for inline display
        import io
        import base64
        buffer = io.BytesIO()
        fig_original.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        plot_data = buffer.getvalue()
        buffer.close()
        original_base64 = base64.b64encode(plot_data).decode('utf-8')
        plt.close(fig_original)
        
        plots_generated.append({
            'plot_type': 'original',
            'plot_path': original_path,
            'plot_base64': original_base64,
            'stats': stats_original
        })
    
    # Create cleaned plot if significant outliers are detected
    cleaned_base64 = None
    cleaned_path = None
    stats_cleaned = None
    
    if outlier_info['has_outliers'] and outlier_info['impact_significant']:
        print(f"DEBUG: Test {test_id} - Creating cleaned plot (outliers impact significant)")
        
        # Create cleaned dataset
        clean_indices = outlier_info['clean_indices']
        clean_values = values[clean_indices]
        clean_test_results = test_results[clean_indices]
        clean_serial_numbers = serial_numbers[clean_indices]
        
        # Create cleaned plot
        fig_cleaned, stats_cleaned = create_single_consistency_plot(
            test_id, clean_values, clean_test_results, clean_serial_numbers, usl, lsl,
            plot_title_suffix=" (Outliers Removed)", outlier_info=outlier_info, clean_data=True
        )
        
        if fig_cleaned:
            # Save cleaned plot
            cleaned_path = os.path.join(output_dir, f'{safe_filename}_consistency_cleaned.png')
            fig_cleaned.savefig(cleaned_path, dpi=150, bbox_inches='tight')
            
            # Convert to base64 for inline display
            buffer = io.BytesIO()
            fig_cleaned.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            buffer.seek(0)
            plot_data = buffer.getvalue()
            buffer.close()
            cleaned_base64 = base64.b64encode(plot_data).decode('utf-8')
            plt.close(fig_cleaned)
            
            plots_generated.append({
                'plot_type': 'cleaned',
                'plot_path': cleaned_path,
                'plot_base64': cleaned_base64,
                'stats': stats_cleaned
            })
    
    # Return comprehensive results
    result = {
        'test_id': test_id,
        'has_outliers': outlier_info['has_outliers'],
        'outlier_info': outlier_info,
        'limit_recommendations': limit_recommendations,
        'plots_generated': plots_generated,
        'dual_plots': len(plots_generated) > 1
    }
    
    # Add original plot stats to top level for backward compatibility
    if plots_generated:
        original_stats = plots_generated[0]['stats']
        result.update({
            'n_samples': original_stats['n_samples'],
            'n_sn549': original_stats['n_sn549'],
            'n_sn602': original_stats['n_sn602'],
            'n_pass': original_stats['n_pass'],
            'n_fail': original_stats['n_fail'],
            'mean': original_stats['mean'],
            'std': original_stats['std'],
            'ucl': original_stats['ucl'],
            'lcl': original_stats['lcl'],
            'min': original_stats['min'],
            'max': original_stats['max'],
            'out_of_control_total': original_stats['out_of_control_total'],
            'usl_violations': original_stats['usl_violations'],
            'lsl_violations': original_stats['lsl_violations'],
            'has_limits': original_stats['has_limits'],
            'usl': original_stats['usl'],
            'lsl': original_stats['lsl'],
            'plot_path': plots_generated[0]['plot_path'],
            'plot_base64': plots_generated[0]['plot_base64']
        })
    
    return result


@consistency_bp.route('/export_plots', methods=['POST'])
def export_plots():
    """Export all plots from a directory as a ZIP archive"""
    try:
        data = request.get_json()
        output_dir = data.get('output_dir')
        
        if not output_dir or not os.path.exists(output_dir):
            return jsonify({'error': 'Invalid output directory'}), 400
        
        # Create a temporary ZIP file
        import tempfile
        import zipfile
        from flask import send_file
        
        temp_zip = tempfile.NamedTemporaryFile(delete=False, suffix='.zip')
        temp_zip.close()
        
        try:
            with zipfile.ZipFile(temp_zip.name, 'w', zipfile.ZIP_DEFLATED) as zipf:
                # Add all PNG files from the output directory
                for root, dirs, files in os.walk(output_dir):
                    for file in files:
                        if file.endswith('.png') or file.endswith('.csv'):
                            file_path = os.path.join(root, file)
                            # Add to zip with relative path
                            arcname = os.path.relpath(file_path, output_dir)
                            zipf.write(file_path, arcname)
            
            # Send the ZIP file
            return send_file(
                temp_zip.name,
                as_attachment=True,
                download_name=f'w3a_plots_{datetime.now().strftime("%Y%m%d_%H%M%S")}.zip',
                mimetype='application/zip'
            )
            
        except Exception as e:
            # Clean up temp file on error
            if os.path.exists(temp_zip.name):
                os.unlink(temp_zip.name)
            raise e
            
    except Exception as e:
        print(f"Error in plot export: {str(e)}")
        return jsonify({'error': str(e)}), 400
