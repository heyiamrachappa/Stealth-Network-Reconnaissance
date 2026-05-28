#!/usr/bin/env python3
import os
import sys

# Add project root to PYTHONPATH so all modules resolve correctly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import and run the PhantomTrace forensics dashboard
from dashboard.app import ForensicWorkstationApp

app = ForensicWorkstationApp()
app.run()
