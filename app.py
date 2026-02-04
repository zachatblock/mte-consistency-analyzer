#!/usr/bin/env python3
"""
W3A Flask Application
Main Flask application entry point with modular routing structure.
"""

from flask import Flask, render_template
import os

def create_app():
    """Create and configure the Flask application."""
    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'your-secret-key-here'
    
    # Register blueprints
    from routes.consistency import consistency_bp
    
    app.register_blueprint(consistency_bp)
    
    return app

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True, host='localhost', port=5001)
