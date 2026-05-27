#!/usr/bin/env python3
import os
import sys

# Add current folder to python path to resolve modules correctly
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Run the Streamlit Dashboard
if __name__ == "__main__":
    from dashboard.app import DashboardApp
    app = DashboardApp()
    app.run()
