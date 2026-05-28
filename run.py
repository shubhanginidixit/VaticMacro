#!/usr/bin/env python3
"""
VaticMacro Flask Application Launcher
Run this file to start the application: python run.py
Then open http://localhost:5000 in your browser
"""

import importlib.util
import os


APP_PATH = os.path.join(os.path.dirname(__file__), "app.py")
SPEC = importlib.util.spec_from_file_location("vaticmacro_app", APP_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Unable to load Flask app from {APP_PATH}")

MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
app = MODULE.app

if __name__ == '__main__':
    print("\n" + "="*60)
    print("  VaticMacro - Institutional Research Dashboard")
    print("="*60)
    print("\n✓ Starting Flask server...")
    print("✓ Access the application at: http://localhost:5000")
    print("✓ Press CTRL+C to stop the server\n")
    print("="*60 + "\n")
    
    app.run(debug=False, host='127.0.0.1', port=5000, use_reloader=False)
