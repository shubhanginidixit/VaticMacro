#!/usr/bin/env python3
"""
VaticMacro Flask Application Launcher
Run this file to start the application: python run.py
Then open http://localhost:5000 in your browser
"""

from app import app

if __name__ == '__main__':
    print("\n" + "="*60)
    print("  VaticMacro - Institutional Research Dashboard")
    print("="*60)
    print("\n✓ Starting Flask server...")
    print("✓ Access the application at: http://localhost:5000")
    print("✓ Press CTRL+C to stop the server\n")
    print("="*60 + "\n")
    
    app.run(debug=True, host='127.0.0.1', port=5000)
