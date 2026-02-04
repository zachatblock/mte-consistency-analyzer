"""
Consistency plots routes for the W3A Flask application.
Handles the consistency plot generation functionality.
"""

from flask import Blueprint, render_template, request, jsonify, send_file, flash, redirect, url_for
import os
import csv
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from collections import defaultdict
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
