"""Shared paths for CLAS12 fast MC."""

import os
import socket

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))

# Auto-detect machine
if 'ifarm' in socket.gethostname() or os.path.isdir('/volatile/clas12'):
    DATA_ROOT = "/volatile/clas12/vpk/fastmc"
else:
    DATA_ROOT = "/Users/vpk/Downloads/GEMC/fast_MC"

PIDS = {"electron": 11, "positron": -11, "proton": 2212,
        "pi+": 211, "pi-": -211, "K+": 321, "K-": -321}

DETECTOR_CUTS = {
    "FD": 2000,          # Forward Detector (DC tracking): |status| >= 2000
    "CD": 4000,          # Central Detector: |status| >= 4000
    "FT": (1000, 2000),  # Forward Tagger: 1000 <= |status| < 2000
    "all": None,         # No status cut
}
