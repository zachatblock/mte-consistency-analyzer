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

warnings.filterwarnings('ignore')

consistency_bp = Blueprint('consistency', __name__)

def extract_sn_from_filename(filename):
    """Extract serial number from filename."""
    match = re.match(r'([\dA-Z]+)_', filename)
    if match:
        return match.group(1)
    return None

def find_parametric_csv(log_dir):
    """Find all parametric CSV files in directory tree."""
    csv_files = []
    for root, dirs, files in os.walk(log_dir):
        for file in files:
            if file.endswith('_parametric.csv'):
                csv_files.append(os.path.join(root, file))
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
    """Parse CSV file and extract test data."""
    test_data = []
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
                    
                    if test_type != 'float':
                        continue
                    
                    try:
                        value = float(ret_value)
                        if abs(value) > 1e30:
                            continue
                        
                        test_data.append({
                            'test_id': test_id,
                            'value': value,
                            'unit': unit,
                            'timestamp': execution_time,
                            'test_result': test_result,
                            'serial_number': serial_number,
                            'source_file': filename
                        })
                    except (ValueError, TypeError):
                        continue
                except Exception as e:
                    continue
    except Exception as e:
        print(f"Error reading {csv_path}: {e}")
    
    return test_data

def collect_all_test_data(log_dir):
    """Collect all test data from all CSV files."""
    csv_files = find_parametric_csv(log_dir)
    all_tests = defaultdict(list)
    
    for csv_file in csv_files:
        test_data = parse_csv_file(csv_file)
        for test in test_data:
            all_tests[test['test_id']].append(test)
    
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
    
    # Plot control limits
    ax.axhline(y=mean, color='g', linestyle='-', linewidth=2, label=f'Mean = {mean:.2f}')
    ax.axhline(y=ucl, color='orange', linestyle='--', linewidth=2, label=f'UCL (+3σ) = {ucl:.2f}')
    ax.axhline(y=lcl, color='orange', linestyle='--', linewidth=2, label=f'LCL (-3σ) = {lcl:.2f}')
    
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
                'plot_files': plot_files[:10]  # Return first 10 plot paths
            })
            
        finally:
            # Cleanup temp extraction directory
            if temp_extract_dir and os.path.exists(temp_extract_dir):
                shutil.rmtree(temp_extract_dir)
                
    except Exception as e:
        print(f"DEBUG: Consistency plot error: {str(e)}")
        return jsonify({'error': f'Consistency plot generation error: {str(e)}'}), 500

def create_consistency_plot_with_limits(test_id, test_data, output_dir, limit_info=None):
    """Create a consistency plot with USL/LSL limits if available."""
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
    
    # Plot USL/LSL limits if available
    usl_violations = 0
    lsl_violations = 0
    if limit_info:
        usl = limit_info['high']
        lsl = limit_info['low']
        ax.axhline(y=usl, color='red', linestyle='-', linewidth=2, alpha=0.8, label=f'USL = {usl:.2f}')
        ax.axhline(y=lsl, color='red', linestyle='-', linewidth=2, alpha=0.8, label=f'LSL = {lsl:.2f}')
        
        # Count limit violations
        usl_violations = np.sum(values > usl)
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
    
    if limit_info:
        stats_text += f'\n USL Violations: {usl_violations}'
        stats_text += f'\n LSL Violations: {lsl_violations}'
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            fontsize=9, family='monospace')
    
    plt.tight_layout()
    
    safe_filename = test_id.replace('/', '_').replace('\\', '_')
    output_path = os.path.join(output_dir, f'{safe_filename}_consistency.png')
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
        'usl_violations': usl_violations,
        'lsl_violations': lsl_violations,
        'has_limits': limit_info is not None,
        'plot_path': output_path
    }
