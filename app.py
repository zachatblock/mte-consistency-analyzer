#!/usr/bin/env python3
"""
W3A Flask Application
Main Flask application entry point with modular routing structure.
"""

from flask import Flask, render_template
import os
import tempfile

def create_app():
    """Create and configure the Flask application."""
    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'w3a-consistency-analysis-secret-key-2025'
    app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size
    app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()
    app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0  # Disable caching
    app.config['PERMANENT_SESSION_LIFETIME'] = 3600  # 1 hour session timeout
    
    # Register blueprints
    from routes.consistency import consistency_bp
    
    # Register blueprints (no URL prefix to match original structure)
    app.register_blueprint(consistency_bp)
    
    # Root route redirect
    @app.route('/')
    def index():
        return render_template('index.html')
    
    return app

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True, host='localhost', port=5001)
